"""Bounded FP16 graph eviction/reuse diagnostic; no checkpoint or graph copies."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
from weakref import WeakKeyDictionary


def execute(args):
    import torch
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    from npunlock._native import InferenceSession
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[2]
    out=Path(args.output_dir).resolve() if args.output_dir else root/'npunlock-probe/research/results/smollm-cache-lifetime'
    model=Path(args.model).resolve() if args.model else root/'models/SmolLM2-135M-Instruct'
    out.mkdir(parents=True,exist_ok=True)
    tag=f'{args.layers}-{args.budget}-{args.retention}'+('-'+args.label if args.label else '')
    if (out/(tag+'.jsonl')).exists():raise FileExistsError('Use a fresh --label for each run')
    events=[];active=dict(phase='setup',layer=None,cycle=0);owners=WeakKeyDictionary();held=[]
    begin=time.perf_counter()
    report=dict(status='running',config=vars(args),phases=[],events_file=tag+'.jsonl',sources={
        p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ('torch_intel_npu.py','torch_npu_smollm.py',
        'torch_npu_graph_cache.py','npunlock/python/npunlock/_native.py','npunlock/src/graphinfer/session_win32.c',
        'npunlock-probe/session-runtime/npunlock_session.dll','npunlock-probe/torch-device/_torch_intel_npu.pyd')})
    npu._MAX_SESSION_BYTES=args.budget*1024*1024
    original_room=npu._make_session_room;original_cache=npu._cache_program;original_close=InferenceSession.close
    def label(key):return key[0]+':'+(key[1].hex()[:12] if len(key)>1 and isinstance(key[1],bytes) else repr(key[1:])[:80])
    def log(kind,**values):
        row=dict(event=kind,seconds=time.perf_counter()-begin,**active,**values);events.append(row)
        # Flush metadata before native submissions, including a device-loss failure.
        with (out/(tag+'.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    def close(session):
        if not session._closed:log('close',graph=owners.get(session,'root/unmapped'))
        return original_close(session)
    def cache(key,program,**options):
        before=npu._session_creations
        result=original_cache(key,program,**options)
        if npu._session_creations!=before:
            owners[result[0]]=label(key)
            log('import',graph=label(key),bytes=len(program.graph_blob),cached_bytes=npu._session_blob_bytes())
        return result
    def room(incoming_bytes=0):
        prior=dict(npu._sessions) if args.retention=='hold' else tuple(npu._sessions)
        if args.retention=='flush' and (len(npu._sessions)>=npu._MAX_SESSIONS or
            npu._session_blob_bytes()+incoming_bytes>npu._MAX_SESSION_BYTES):
            keys=[k for k in npu._sessions if k in npu._rebuildable_sessions and k[0].startswith('smollm_')]
            for key in keys:
                session,program=npu._sessions.pop(key);npu._rebuildable_sessions.remove(key)
                npu._retired_sessions.add(session);del session,program;npu._session_evictions+=1
        original_room(incoming_bytes)
        removed=[key for key in prior if key not in npu._sessions]
        if args.retention=='hold':held.extend(prior[key] for key in removed)
        if removed:log('evict',graphs=[label(key) for key in removed],cached_bytes=npu._session_blob_bytes(),held=len(held))
    npu._make_session_room=room;npu._cache_program=cache;InferenceSession.close=close
    def checkpoint(name):
        gc.collect();report['phases'].append(dict(name=name,cycle=active['cycle'],stats=npu.stats(),held_graphs=len(held)))
        print(name,'calls',npu.stats()['npu_calls'],'evictions',npu.stats()['graph_cache_evictions'],
              'blob MiB',round(npu._session_blob_bytes()/2**20,2),'held',len(held),flush=True)
    def work():
        stack=FrozenSmolStack.from_checkpoint(model,num_layers=args.layers,mode='fp16')
        host=(torch.randn(1,64,576,generator=torch.Generator().manual_seed(451))/4).half()
        active.update(phase='prefill1',layer=None);value,state=stack.prefill(host[:,:1].contiguous().to('intel_npu'))
        checkpoint('prefill1')
        for i in range(2):
            active.update(phase='decode1',layer=None)
            value,state=stack.decode(host[:,i+1:i+2].contiguous().to('intel_npu'),state)
        checkpoint('decode1')
        expected=None
        for iteration in range(2):
            active.update(phase=f'forward64-{iteration}',layer=None)
            value=host.to('intel_npu')
            for i,layer in enumerate(stack.layers):
                active['layer']=i
                log('before',cached_bytes=npu._session_blob_bytes(),evictions=npu.stats()['graph_cache_evictions'])
                value=layer(value)
                log('after',calls=npu.stats()['npu_calls'])
            actual=value.cpu()
            if expected is None:expected=actual
            else:torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            checkpoint(f'forward64-{iteration}')
        # Check rebuild after switching back to KV graphs; old cache owners remain valid.
        active.update(phase='decode-after-forward',layer=None)
        x=host[:,3:4].contiguous().to('intel_npu')
        value,new_state=stack.decode(x,state)
        assert all(c.length==4 for c in new_state)
        again,_=stack.decode(x,state)
        torch.testing.assert_close(value.cpu(),again.cpu(),rtol=0,atol=0)
        checkpoint('decode-after-forward')
    try:
        with torch.no_grad():
            for cycle in range(args.cycles):
                active['cycle']=cycle
                try:work()
                except RuntimeError as error:
                    if not args.retry_after_guard or 'prepared/pinned family' not in str(error):raise
                    report['guard']=dict(message=str(error),stats=npu.stats())
                    assert not report['guard']['stats']['context_failed'] and npu._smol_plan_depth==0
                    assert torch.intel_npu.set_graph_cache_budget is npu.set_graph_cache_budget
                    npu.set_graph_cache_budget(512*1024**2)
                    print('guard rejected safely; retry at 512 MiB',flush=True)
                    work()
            if args.retry_after_guard:assert 'guard' in report
        report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    report['before_close']=npu.stats()
    active.update(phase='cleanup',layer=None)
    held.clear();gc.collect()
    npu.shutdown();gc.collect();report['shutdown']=npu.stats()
    npu._make_session_room=original_room;npu._cache_program=original_cache;InferenceSession.close=original_close
    final=report['shutdown']
    report['healthy']=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_sessions','cached_scalar_buffers','quarantined_buffer_bytes'))
    report['elapsed_seconds']=time.perf_counter()-begin
    (out/(tag+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FINAL',tag,report['status'],'healthy',report['healthy'],flush=True)
    return int(report['status']!='passed' or not report['healthy'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers',type=int,choices=(1,30),default=30)
    parser.add_argument('--budget',type=int,choices=(512,768,256,16),default=512)
    parser.add_argument('--retention',choices=('normal','hold','flush'),default='normal')
    parser.add_argument('--label',default='')
    parser.add_argument('--cycles',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--retry-after-guard',action='store_true')
    parser.add_argument('--model',help='Existing local checkpoint directory; no model is downloaded')
    parser.add_argument('--output-dir',help='Directory for this run\'s temporary scalar/event reports')
    raise SystemExit(execute(parser.parse_args()))

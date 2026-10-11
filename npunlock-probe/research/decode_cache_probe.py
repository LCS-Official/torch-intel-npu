"""Isolated allocator controls with the same kernel, graphs and KV snapshots."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/decode-cache'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def bind(runtime):
    if runtime=='before':
        source=(OUT/'before/torch_intel_npu.py').read_text(encoding='utf-8')
        old='_EXT = _ROOT / "npunlock-probe/torch-device/_torch_intel_npu.pyd"';assert source.count(old)==1
        source=source.replace(old,f'_EXT = Path({str(OUT/"before/_torch_intel_npu.pyd")!r})')
        module=types.ModuleType('torch_intel_npu');module.__file__=str(ROOT/'torch_intel_npu.py')
        sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
    import torch_intel_npu as npu
    return npu


def worker(case,runtime,label):
    import numpy as np
    import torch
    npu=bind(runtime);torch.set_num_threads(8)
    report=dict(status='running',case=case,runtime=runtime,source_sha256=sha(Path(__file__)),checks=[],timings=[],
        build=read(OUT/'before/build.json' if runtime=='before' else ROOT/'npunlock-probe/torch-device/build.json'))
    fields=('npu_calls','buffer_allocations','driver_allocations','allocation_reuses','host_upload_copies','host_download_copies','session_creations')
    def delta(before,after):return {f:after[f]-before[f] for f in fields}
    def prime():
        npu.empty_cache()
        limit=npu.stats()['allocation_cache_limit_blocks']
        blocks=[torch.empty(i+1,dtype=torch.float32,device='intel_npu') for i in range(limit)]
        del blocks
        assert npu.stats()['cached_buffer_blocks']==limit
    try:
        with torch.no_grad():
            if case=='allocation':
                prime();before=npu.stats();samples=[]
                # Sizes of a B4 decoder output, two KV snapshots and a position.
                for _ in range(21):
                    start=time.perf_counter_ns()
                    values=[torch.empty(shape,dtype=dtype,device='intel_npu') for shape,dtype in
                            [((4,1,576),torch.float16),((4,3,64,64),torch.float16),((4,3,64,64),torch.float16),((1,),torch.float32)]]
                    del values;samples.append((time.perf_counter_ns()-start)/1e6)
                after=npu.stats();report['checks'].append(dict(name='full_pool_shape_switch',counter_delta=delta(before,after),
                    cached_blocks=after['cached_buffer_blocks'],cached_bytes=after['cached_buffer_bytes']))
                report['timings'].append(dict(name='allocate_and_release_four',samples_ms=samples,median_ms=statistics.median(samples)))
            else:
                from safetensors import safe_open
                from transformers import AutoTokenizer
                from torch_npu_smollm import FrozenSmolStack
                directory=ROOT/'models/SmolLM2-135M-Instruct'
                tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
                ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
                with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as file:embedding=file.get_tensor('model.embed_tokens.weight')
                arrays={}
                for batch,prefix,mode in ((1,32,'fp16'),(1,32,'w8a16'),(4,16,'fp16'),(4,16,'w8a16')):
                    tag=f'b{batch}-s{prefix}-{mode}'
                    tokens=ids.repeat(1,(prefix+33+ids.shape[1])//ids.shape[1])[:,:prefix+32].repeat(batch,1)
                    host=embedding[tokens].half().contiguous();stack=FrozenSmolStack.from_checkpoint(directory,num_layers=4,mode=mode)
                    x=host[:,:prefix].contiguous().to('intel_npu');one=host[:,prefix:prefix+1].contiguous().to('intel_npu')
                    output,seed=stack.prefill(x);decoded,updated=stack.decode(one,seed)
                    def save(name,value,caches):
                        arrays[f'{tag}/{name}/output']=value.cpu().numpy().copy()
                        for i,c in enumerate(caches):
                            for field in ('keys','values','_position'):arrays[f'{tag}/{name}/{i}/{field}']=getattr(c,field).cpu().numpy().copy()
                    save('prefill',output,seed);save('decode',decoded,updated);del decoded,updated
                    # Diagnose both the normal pool and a saturated prior-shape pool.
                    for scope in ('natural','shape-switch'):
                        if scope=='shape-switch':prime()
                        for _ in range(5):value=stack.decode(one,seed);del value
                        before=npu.stats();samples=[];cleanup=[]
                        for _ in range(31):
                            start=time.perf_counter_ns();value=stack.decode(one,seed);returned=time.perf_counter_ns()
                            del value;freed=time.perf_counter_ns()
                            samples.append((returned-start)/1e6);cleanup.append((freed-returned)/1e6)
                        after=npu.stats();d=delta(before,after)
                        assert d['npu_calls']==31*4 and all(d[f]==0 for f in ('host_upload_copies','host_download_copies','session_creations'))
                        record=dict(tag=tag,scope=scope,operation='single-decode',samples_ms=samples,median_ms=statistics.median(samples),
                            cleanup_ms=cleanup,cleanup_median_ms=statistics.median(cleanup),counter_delta=d)
                        report['timings'].append(record);print(tag,scope,record['median_ms'],d['driver_allocations'],flush=True)
                    # Remove synthetic stale blocks before the real chain. Its
                    # ordinary released buffers carry into the next config.
                    npu.empty_cache()
                    # A real position-advancing chain, 32 tokens with resident inputs.
                    inputs=[host[:,prefix+i:prefix+i+1].contiguous().to('intel_npu') for i in range(32)]
                    before=npu.stats();state=seed;samples=[]
                    for i,item in enumerate(inputs):
                        start=time.perf_counter_ns();value,state=stack.decode(item,state);samples.append((time.perf_counter_ns()-start)/1e6)
                        assert all(c.length==prefix+i+1 for c in state)
                        if i in (0,15,31):save(f'chain-{i+1}',value,state)
                    after=npu.stats();d=delta(before,after)
                    # Diagnostic downloads occur outside the recorded per-step interval.
                    assert d['npu_calls']==128 and d['host_upload_copies']==d['session_creations']==0
                    save('preserved',output,seed)
                    for key in [k for k in arrays if k.startswith(f'{tag}/prefill/')]:
                        np.testing.assert_array_equal(arrays[key].view(np.uint8),arrays[key.replace('/prefill/','/preserved/')].view(np.uint8))
                    report['timings'].append(dict(tag=tag,scope='natural',operation='chain-32',samples_ms=samples,
                        median_ms=statistics.median(samples),total_step_ms=sum(samples),counter_delta=d))
                    report['checks'].append(dict(tag=tag,seed_preserved=True,chain_length=32,final_length=state[0].length,
                        graph_sha256={f'{i}-{key[-1]}':hashlib.sha256(npu._sessions[session][1].graph_blob).hexdigest()
                            for i,layer in enumerate(stack.layers) for key,session in layer._keys.items()}))
                    del inputs,item,value,state,stack,seed,x,one,output
                path=OUT/f'{label}-outputs.npz';np.savez(path,**arrays);report['outputs_sha256']=sha(path)
            gc.collect();npu.shutdown();report['final_stats']=npu.stats()
            assert not report['final_stats']['context_failed']
            assert report['final_stats']['live_buffers']==report['final_stats']['cached_buffer_bytes']==0
        report['status']='passed'
    except Exception:
        import traceback
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/f'{label}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return int(report['status']!='passed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('allocation','model'),required=True)
    parser.add_argument('--runtime',choices=('before','after'),default='after');parser.add_argument('--label',required=True)
    parser.add_argument('--child',action='store_true');args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.child:return worker(args.case,args.runtime,args.label)
    with (OUT/f'{args.label}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--case',args.case,'--runtime',args.runtime,'--label',args.label,'--child'],
            cwd=ROOT,stdout=log,stderr=log,timeout=600)
    print((OUT/f'{args.label}.log').read_text(encoding='utf-8'),flush=True);return result.returncode


if __name__=='__main__':raise SystemExit(main())

"""Rebuild frozen graphs under count/byte pressure; retain only Markdown reports."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
import weakref


def execute(case,mode='fp16',execution='whole'):
    import torch
    import torch_intel_npu as npu
    from torch_npu_lowbit import QuantizedLinear,quantize_weight
    root=Path(__file__).resolve().parent;started=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(833)
    files=('torch_intel_npu.py','torch_npu_lowbit.py','torch_npu_rect_gemm.py','torch_npu_transformer.py',
           'torch_npu_smollm.py','torch_npu_frozen_capacity_demo.py',
           'npunlock-probe/session-runtime/npunlock_session.dll','npunlock-probe/torch-device/_torch_intel_npu.pyd')
    source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files};checks=[]
    def memory():
        state=npu.stats()
        assert state['cached_sessions']<=npu._MAX_SESSIONS
        assert state['cached_graph_blob_bytes']<=npu._MAX_SESSION_BYTES
        assert not state['context_failed']
        return state
    def pressure():
        for size in range(41,49):
            value=-torch.arange(size,dtype=torch.float32).to('intel_npu')
        del value
    def apply(model,x):
        before=npu.stats();value=model(x)
        assert npu.stats()['host_download_copies']==before['host_download_copies']
        return value.cpu()
    def lowbit():
        npu._MAX_SESSIONS=2;weight=torch.randn(32,64)/8
        models=[QuantizedLinear(quantize_weight(weight,format=fmt)) for fmt in ('int8','int4','nf4')]
        data={m:(torch.randn(m,64)/8).half() for m in (1,3,7)};baseline={};before=npu.stats()
        for repeat in range(3):
            for model in models:
                for batch,host in data.items():
                    actual=apply(model,host.to('intel_npu'));key=(model.weight.format,batch)
                    expected=(host.float()@torch.from_numpy(model.weight.dequantize()).float().t()).half()
                    torch.testing.assert_close(actual,expected,rtol=3e-3,atol=3e-4)
                    if repeat:assert torch.equal(actual.view(torch.int16),baseline[key].view(torch.int16))
                    else:baseline[key]=actual
                    state=memory();assert state['pinned_graph_sessions']==0
            print('lowbit repeat',repeat+1,'evictions',npu.stats()['graph_cache_evictions'],flush=True)
        state=memory();assert state['graph_factory_calls']-before['graph_factory_calls']==27
        assert state['live_graph_factories']==0
        owner=weakref.ref(models[-1]);del model,models;gc.collect();assert owner() is None
        checks.append(dict(formats=['int8','int4','nf4'],batches=[1,3,7],repeats=3,
                           reference_rtol=.003,reference_atol=.0003,bit_exact_reloads=True,model_not_retained=True))
    def bytes():
        npu._MAX_SESSIONS=64
        models=[QuantizedLinear(quantize_weight(torch.randn(32,64)/8,format='int8')) for _ in range(3)]
        host=(torch.randn(3,64)/8).half();x=host.to('intel_npu')
        baseline=[apply(model,x) for model in models]
        sizes=[len(npu._sessions[model._keys[3]][1].graph_blob) for model in models]
        npu._MAX_SESSION_BYTES=max(sizes);npu._make_session_room();before=npu.stats()
        for repeat in range(3):
            for model,expected in zip(models,baseline):
                assert torch.equal(apply(model,x).view(torch.int16),expected.view(torch.int16))
                memory()
        state=memory();assert state['graph_cache_evictions']>before['graph_cache_evictions']
        assert len(npu._sessions)==1 and state['pinned_graph_sessions']==0
        limit=npu._MAX_SESSION_BYTES;calls=npu.stats()['npu_calls'];transient=npu.stats()['transient_graph_imports']
        try:
            npu._MAX_SESSION_BYTES=min(sizes)-1;npu._make_session_room()
            assert torch.equal(apply(models[0],x).view(torch.int16),baseline[0].view(torch.int16))
            assert npu.stats()['npu_calls']==calls+1
            assert npu.stats()['transient_graph_imports']==transient+1
            assert npu.stats()['cached_graph_blob_bytes']==npu.stats()['deferred_graph_sessions']==0
        finally:npu._MAX_SESSION_BYTES=limit
        assert torch.equal(apply(models[0],x).view(torch.int16),baseline[0].view(torch.int16))
        checks.append(dict(graph_sizes=sizes,byte_budget=limit,item_limit=64,graphs_retained=1,
                           bit_exact_reloads=True,oversize_runs_without_caching=True,recovery=True))
    def transformer():
        from torch_npu_transformer import FrozenTransformer
        source=torch.nn.TransformerEncoderLayer(128,4,dim_feedforward=512,dropout=0,
            activation='gelu',batch_first=True,norm_first=True).eval()
        models=[FrozenTransformer.from_float(source,mode=m) for m in ('fp16','w8a16')]
        hosts=[(torch.randn(1,length,128)/8).half() for length in (2,3)];baseline={}
        for model in models:
            for host in hosts:
                actual=apply(model,host.to('intel_npu'));reference=model.reference_cpu(host)[0]
                torch.testing.assert_close(actual,reference,rtol=.004,atol=.002)
                relative=float(torch.linalg.vector_norm(actual.float()-reference.float())/torch.linalg.vector_norm(reference.float()))
                assert relative<.002
                baseline[(model.mode,host.shape[1])]=actual
        npu._MAX_SESSIONS=2;npu._make_session_room();before=npu.stats();pressure()
        for repeat in range(2):
            for model in models:
                for host in hosts:
                    actual=apply(model,host.to('intel_npu'))
                    assert torch.equal(actual.view(torch.int16),baseline[(model.mode,host.shape[1])].view(torch.int16))
                    memory()
        assert npu.stats()['graph_factory_calls']>before['graph_factory_calls']
        checks.append(dict(modes=['fp16','w8a16'],lengths=[2,3],cpu_reference=True,bit_exact_reloads=True))
    def smollm(plan=False):
        from torch_npu_smollm import FrozenSmolStack
        layers=4 if plan else 1
        model=FrozenSmolStack.from_checkpoint(root/'models/SmolLM2-135M-Instruct',num_layers=layers,mode=mode)
        for layer in model.layers:
            layer._fuse_norm=execution!='plain';layer._whole_layer=execution=='whole'
        host=(torch.randn(1,2,576)/8).half();tokens=[(torch.randn(1,1,576)/8).half() for _ in range(2)]
        deferred=[];original=npu._run_chain
        def chain(entries,group_size=4):
            deferred.append(npu.stats()['deferred_graph_sessions']);return original(entries,group_size)
        npu._run_chain=chain
        def snapshot(out,caches):
            return (out.cpu().view(torch.int16),*[(c.keys.cpu().view(torch.int16),c.values.cpu().view(torch.int16),
                     c.length,c._position.cpu().view(torch.int32)) for c in caches])
        def same(left,right):
            assert torch.equal(left[0],right[0])
            for a,b in zip(left[1:],right[1:]):
                assert a[2]==b[2] and all(torch.equal(a[i],b[i]) for i in (0,1,3))
        def sequence():
            before=npu.stats();out,caches=model.prefill(host.to('intel_npu'))
            assert npu.stats()['host_download_copies']==before['host_download_copies']
            result=[snapshot(out,caches)]
            for token in tokens:
                old=caches;old_snapshot=snapshot(out,old);before=npu.stats()
                out,caches=model.decode(token.to('intel_npu'),old,group_size=4 if plan else 1)
                assert npu.stats()['host_download_copies']==before['host_download_copies']
                same((old_snapshot[0],*snapshot(out,old)[1:]),old_snapshot)
                result.append(snapshot(out,caches))
            return result
        try:
            baseline=sequence()
            if not plan:
                reference=model.reference_cpu(host)[0];actual=baseline[0][0].view(torch.float16)
                relative=float(torch.linalg.vector_norm(actual.float()-reference.float())/torch.linalg.vector_norm(reference.float()))
                rms=reference.float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-5)
                scaled=float(((actual.float()-reference.float()).abs()/rms).max())
                assert relative<.003 and scaled<.01,(relative,scaled)
            if plan:
                sizes=[len(program.graph_blob) for key,(session,program) in npu._sessions.items() if key[0]=='smollm_fused_norm']
                budget=npu._MAX_SESSION_BYTES;guards=[]
                try:
                    npu._MAX_SESSION_BYTES=min(sizes)-1;npu._make_session_room()
                    try:sequence()
                    except RuntimeError as error:
                        assert 'set_graph_cache_budget' in str(error);guards.append(str(error))
                    else:raise AssertionError('Oversized Smol graph must reject transient execution')
                finally:npu.set_graph_cache_budget(budget)
                npu._MAX_SESSIONS=2;npu._make_session_room()
                try:sequence()
                except RuntimeError as error:
                    assert 'prepared/pinned family' in str(error);guards.append(str(error))
                else:raise AssertionError('Small-cache grouped Smol plan must reject partial retirement')
                assert not npu.stats()['context_failed'] and npu._smol_plan_depth==0
                npu._MAX_SESSIONS=16
            else:npu._MAX_SESSIONS=2
            npu._make_session_room();before=npu.stats();pressure()
            for repeat in range(2):
                current=sequence()
                for a,b in zip(current,baseline):same(a,b)
                state=memory();assert state['pinned_graph_sessions']==0 and state['deferred_graph_sessions']==0
                print('smollm',mode,execution,'repeat',repeat+1,'evictions',state['graph_cache_evictions'],flush=True)
            assert npu.stats()['graph_factory_calls']>before['graph_factory_calls']
            if plan:
                assert max(deferred)==0
                checks.append(dict(smol_capacity_guards=guards,recovered_in_same_context=True,entry_limit=16))
            checks.append(dict(mode=mode,execution=execution,layers=layers,prefix_length=2,decode_tokens=2,
                               repeats=2,bit_exact_outputs_and_kv=True,old_snapshot_unchanged=True,
                               cpu_reference=not plan,maximum_deferred_before_chain=max(deferred,default=0)))
        finally:npu._run_chain=original
    def plan():smollm(True)
    def gemm():
        from torch_npu_rect_gemm import matrix_fp32
        npu._MAX_SESSIONS=2
        for replicas in (4,8):
            a=torch.randn(7,33)/8;b=torch.randn(33,9)/8;x,y=a.to('intel_npu'),b.to('intel_npu')
            baseline=matrix_fp32(x,y,replicas=replicas).cpu();torch.testing.assert_close(baseline,a@b,rtol=4e-5,atol=2e-6)
            before=npu.stats();pressure();actual=matrix_fp32(x,y,replicas=replicas).cpu()
            assert torch.equal(actual.view(torch.int32),baseline.view(torch.int32));memory()
            assert npu.stats()['graph_factory_calls']>before['graph_factory_calls']
            checks.append(dict(replicas=replicas,shape=[7,33,9],cpu_reference=True,bit_exact_reload=True))
    error=None
    try:
        with torch.no_grad():locals()[case]()
    except Exception:error=traceback.format_exc();print(error,flush=True)
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes',
        'deferred_graph_sessions','cached_graph_blob_bytes','live_graph_factories'))
    report=dict(status='passed' if error is None and healthy else 'failed',case=case,mode=mode,execution=execution,
                checks=checks,error=error,stats=before,shutdown=final,elapsed_seconds=time.perf_counter()-started,source_sha256=source_sha256)
    target=root/'npunlock-probe/research/results/frozen-capacity';target.mkdir(exist_ok=True)
    (target/f'{case}-{mode}-{execution}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(case,mode,execution,report['status'],'seconds',report['elapsed_seconds'],flush=True)
    return int(report['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('lowbit','bytes','transformer','smollm','plan','gemm'),required=True)
    parser.add_argument('--mode',choices=('fp16','w8a16'),default='fp16')
    parser.add_argument('--execution',choices=('whole','split','plain'),default='whole')
    args=parser.parse_args()
    if args.case=='plan' and args.execution!='whole':parser.error('plan requires whole-layer execution')
    raise SystemExit(execute(args.case,args.mode,args.execution))

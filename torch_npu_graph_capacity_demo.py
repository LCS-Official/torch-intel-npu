"""Bounded graph recycling, retained plans, and dynamic-shape training on NPU."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback


def execute(case,count=1600,limit=1536):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.manual_seed(832);started=time.perf_counter();checks=[]
    npu._MAX_SESSIONS=limit
    def churn():
        for dtype in (torch.float32,torch.float16):
            baseline=npu.stats();host=torch.linspace(-1,1,count+32,dtype=dtype)
            for i in range(count):
                x=host[:i+17].to('intel_npu');result=-x
                assert torch.equal(result.cpu(),-host[:i+17])
                del x,result
                if (i+1)%256==0:print(dtype,i+1,'graphs',len(npu._sessions),'evictions',npu.stats()['graph_cache_evictions'],flush=True)
            # Revisit an evicted shape, with independent result storage.
            x=host[:17].to('intel_npu');result=-x
            assert torch.equal(result.cpu(),-host[:17]);del x,result
            after=npu.stats()
            assert after['cached_sessions']<=limit and after['graph_cache_evictions']>baseline['graph_cache_evictions']
            checks.append(dict(dtype=str(dtype),different_counts=count,evictions=after['graph_cache_evictions']-baseline['graph_cache_evictions']))
    def retained():
        # A prepared plan and a custom compiled graph must survive ordinary LRU pressure.
        host=torch.arange(16,dtype=torch.float32);x=host.to('intel_npu');out=torch.empty_like(x)
        key=('neg','f32',(16,));session,program=npu._session(key)
        plan=[(session,program,(x,),(out,))]
        pinned=('capacity-pinned',)
        npu._cache_program(pinned,program)
        for count in range(17,17+2*limit):
            result=-torch.arange(count,dtype=torch.float32).to('intel_npu')
            del result
        assert key not in npu._sessions and not session._closed and pinned in npu._sessions
        npu._run_chain(plan,1);assert torch.equal(out.cpu(),-host)
        assert npu.stats()['deferred_graph_sessions']==1
        del plan,session,program,x,out;gc.collect()
        assert npu.stats()['deferred_graph_sessions']==0
        for i in range(limit-1):npu._cache_program(('capacity-pinned',i),npu._sessions[pinned][1])
        x=host.to('intel_npu');before=npu.stats()
        try:torch.neg(x)
        except RuntimeError as error:assert 'pinned' in str(error)
        else:raise AssertionError('All-pinned cache accepted a new graph')
        assert npu.stats()['npu_calls']==before['npu_calls']
        result=npu._run(pinned,[x],(16,),torch.float32)
        assert torch.equal(result.cpu(),-host)
        checks.append(dict(prepared_chain_survived_eviction=True,pinned_graph_survived=True,
                           all_pinned_refusal_before_execution=True,existing_graph_recovery=True))
    def training():
        for dtype in (torch.float32,torch.float16):
            cpu=torch.nn.Sequential(torch.nn.Linear(17,23),torch.nn.Tanh(),torch.nn.Linear(23,7)).to(dtype)
            model=copy.deepcopy(cpu).to('intel_npu')
            opts=[torch.optim.SGD(m.parameters(),lr=.01,momentum=.9,foreach=False) for m in (cpu,model)]
            baseline=npu.stats();losses=[]
            for step in range(6):
                host=torch.randn(step+2,17,dtype=dtype)*.1;value=host.to('intel_npu')
                for opt in opts:opt.zero_grad(set_to_none=True)
                expected=cpu(host).float().square().mean();before=npu.stats();actual=model(value).float().square().mean()
                actual.backward();after=npu.stats();expected.backward()
                assert after['host_download_copies']==before['host_download_copies']
                tol=dict(rtol=4e-5,atol=2e-6) if dtype==torch.float32 else dict(rtol=.004,atol=.001)
                torch.testing.assert_close(actual.detach().cpu(),expected,**tol)
                for p,q in zip(model.parameters(),cpu.parameters()):torch.testing.assert_close(p.grad.cpu(),q.grad,**tol)
                for opt in opts:opt.step()
                for p,q in zip(model.parameters(),cpu.parameters()):torch.testing.assert_close(p.detach().cpu(),q,**tol)
                losses.append(actual.item());print(dtype,'step',step+1,'evictions',npu.stats()['graph_cache_evictions'],flush=True)
            after=npu.stats();assert after['graph_cache_evictions']>baseline['graph_cache_evictions']
            checks.append(dict(dtype=str(dtype),independent_cpu_steps=6,losses=losses,
                               evictions=after['graph_cache_evictions']-baseline['graph_cache_evictions']))
    error=None
    try:locals()[case]()
    except Exception:error=traceback.format_exc();print(error,flush=True)
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes','deferred_graph_sessions'))
    root=Path(__file__).resolve().parent
    files=('torch_intel_npu.py','torch_npu_graph_capacity_demo.py','npunlock/python/npunlock/npu4000.py',
           'npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/_torch_intel_npu.pyd',
           'npunlock/src/graphinfer/session_win32.c','npunlock-probe/session-runtime/npunlock_session.dll')
    report=dict(status='passed' if error is None and healthy else 'failed',case=case,error=error,checks=checks,
                stats=before,shutdown=final,elapsed_seconds=time.perf_counter()-started,
                source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files})
    target=root/'npunlock-probe/research/results/graph-capacity';target.mkdir(exist_ok=True)
    (target/f'{case}-{limit}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(case,report['status'],'seconds',report['elapsed_seconds'],flush=True)
    return int(report['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('churn','retained','training'),required=True)
    parser.add_argument('--count',type=int,default=1600);parser.add_argument('--limit',type=int,default=1536)
    args=parser.parse_args()
    if not 1<=args.limit<=1536 or not 1<=args.count<=8192:parser.error('limit 1..1536 and count 1..8192 required')
    raise SystemExit(execute(args.case,args.count,args.limit))

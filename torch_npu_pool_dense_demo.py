"""Dense-plane pool timing and correctness; retain reports rather than raw tensors."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import statistics
import faulthandler
import traceback


def execute(case,precision,baseline=False):
    import torch
    import torch.nn.functional as F
    import torch_intel_npu as npu
    from torch_npu_memory_scale_demo import memory_status
    root=Path(__file__).resolve().parent;started=time.perf_counter();dtype=getattr(torch,precision)
    faulthandler.dump_traceback_later(60,repeat=True)
    torch.set_num_threads(1);torch.manual_seed(838);checks=[]
    names=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_geometry.py','torch_npu_pool_dense_demo.py',
        'npunlock-probe/research/pool_dense_before.py','torch_npu_pool_dense.py','test_torch_npu_pool_dense.py',
        'npunlock-probe/torch-device/backend.cpp',
        'npunlock-probe/torch-device/_torch_intel_npu.pyd','npunlock-probe/session-runtime/npunlock_session.dll',
        'npunlock/tests/fixtures/npu4000/resident/manifest.json')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    if baseline:
        path=root/'npunlock-probe/research/pool_dense_before.py'
        exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),npu.__dict__)
    def upload(x):return x.detach().to('intel_npu').requires_grad_(x.requires_grad)
    def close(x,y):torch.testing.assert_close(x.detach().cpu(),y.detach(),rtol=0,atol=0,equal_nan=True)
    def dense(size):
        if min(memory_status()['physical_available'],memory_status()['commit_available'])<2*2**30:
            raise MemoryError('Dense pool probe needs 2 GiB of physical/commit headroom')
        host=(torch.randn(1,1,size,size,dtype=dtype)/8).requires_grad_();x=upload(host)
        expected,ids=F.max_pool2d(host,3,1,1,return_indices=True);seed=torch.randn_like(expected)/32;gradient=upload(seed)
        expected.backward(seed);measurements=[]
        for repeat in range(3):
            x.grad=None;before=npu.stats();tick=time.perf_counter()
            actual,index=F.max_pool2d(x,3,1,1,return_indices=True);forward=time.perf_counter()-tick
            tick=time.perf_counter();actual.backward(gradient);backward=time.perf_counter()-tick;after=npu.stats()
            close(actual,expected);close(index,ids);close(x.grad,host.grad)
            measurements.append(dict(forward_seconds=forward,backward_seconds=backward,npu_calls=after['npu_calls']-before['npu_calls']))
            print(case,precision,'iteration',repeat+1,measurements[-1],flush=True)
            del actual,index
        checks.append(dict(shape=list(host.shape),source_elements=host.numel(),output_elements=expected.numel(),
            full_cpu_values_indices_gradient=True,measurements=measurements,
            warm_forward_median_seconds=statistics.median(r['forward_seconds'] for r in measurements[1:]),
            warm_backward_median_seconds=statistics.median(r['backward_seconds'] for r in measurements[1:])))
    error=None
    try:
        if case=='small':dense(257)
        elif case=='large':dense(4097)
        else:raise ValueError(case)
    except Exception:error=traceback.format_exc();print(error,flush=True)
    gc.collect();stats=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes',
        'deferred_graph_sessions','cached_graph_blob_bytes','live_graph_factories'))
    result=dict(status='passed' if error is None and healthy else 'failed',case=case,precision=precision,
        baseline=baseline,checks=checks,error=error,stats=stats,shutdown=final,
        elapsed_seconds=time.perf_counter()-started,source_sha256=hashes)
    target=root/'npunlock-probe/research/results/pool-dense';target.mkdir(exist_ok=True)
    (target/f'{"before" if baseline else "after"}-{case}-{precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    faulthandler.cancel_dump_traceback_later()
    print(case,precision,'before' if baseline else 'after',result['status'],checks,flush=True)
    return int(result['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('small','large'),required=True)
    parser.add_argument('--precision',choices=('float32','float16'),default='float32')
    parser.add_argument('--baseline',action='store_true')
    args=parser.parse_args()
    if args.case=='large':
        path=Path(__file__).resolve().parent/f'npunlock-probe/research/results/pool-dense/after-small-{args.precision}.json'
        if path.exists():
            previous=json.loads(path.read_text())
            times=previous.get('checks',[{}])[0].get('measurements',[])
            if previous['status']!='passed' or any(row['forward_seconds']>60 for row in times):
                print('Large run deferred until the small-packet timing check passes',flush=True);raise SystemExit(0)
    raise SystemExit(execute(args.case,args.precision,args.baseline))

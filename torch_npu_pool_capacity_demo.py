"""Pool planes and large extrema with complete CPU controls, without new kernels."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback


def execute(case,precision):
    import torch
    import torch.nn.functional as F
    import torch_intel_npu as npu
    root=Path(__file__).resolve().parent;started=time.perf_counter();dtype=getattr(torch,precision)
    torch.set_num_threads(1);torch.manual_seed(836);checks=[]
    names=('torch_intel_npu.py','torch_npu_geometry.py','torch_npu_autograd.py','torch_npu_sort.py',
        'torch_npu_vision_extended.py','torch_npu_pool_capacity_demo.py','test_torch_npu_pool_capacity.py',
        'npunlock-probe/session-runtime/npunlock_session.dll','npunlock-probe/torch-device/_torch_intel_npu.pyd',
        'npunlock/tests/fixtures/npu4000/resident/manifest.json')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    def upload(x):return x.detach().to('intel_npu').requires_grad_(x.requires_grad)
    def close(x,y,exact=True):
        torch.testing.assert_close(x.detach().cpu(),y.detach(),equal_nan=True,
            rtol=0 if exact else .004 if dtype==torch.float16 else 4e-5,
            atol=0 if exact else .0005 if dtype==torch.float16 else 2e-6)
    def pool(shape,kernel,stride,padding=0,backward=True):
        host=(torch.randn(shape,dtype=dtype)/8).requires_grad_(backward);x=upload(host)
        expected,ids=F.max_pool2d(host,kernel,stride,padding,return_indices=True)
        seed=torch.randn_like(expected)/32;g=upload(seed)
        actual,index=F.max_pool2d(x,kernel,stride,padding,return_indices=True)
        close(actual,expected);close(index,ids)
        if backward:
            actual.backward(g);expected.backward(seed);close(x.grad,host.grad)
        checks.append(dict(shape=list(shape),output_elements=actual.numel(),source_elements=x.numel(),
            kernel=kernel,stride=stride,padding=padding,full_cpu_forward=True,full_cpu_gradient=backward,
            exact_values_and_indices=True))
    def spatial():pool((1,1,1,524289),1,1,backward=False)
    def sampled():pool((1,4,513,513),3,1,1)
    def planes():pool((8193,1,32,64),2,2)
    def extrema():
        host=torch.zeros(1048593,dtype=dtype);host[3]=-9;host[1048513]=7;host[-1]=7;host.requires_grad_();x=upload(host)
        for kind in ('max','min'):
            host.grad=None;x.grad=None;fn=getattr(torch,kind)
            expected,ids=fn(host,0);actual,index=fn(x,0);actual.backward();expected.backward()
            close(actual,expected);close(index,ids);close(x.grad,host.grad)
        with torch.no_grad():
            host[1048514]=float('nan');host[-1]=float('nan');x.copy_(host)
            close(torch.argmax(x),torch.argmax(host));close(torch.argmin(x),torch.argmin(host))
        checks.append(dict(long_axis=host.numel(),first_tie=True,nan_after_old_offset_bound=True,
                           full_cpu_paired_gradient=True))
        host=(torch.arange(524289*3).reshape(524289,3)%3).to(dtype)/8;host.requires_grad_();x=upload(host)
        expected,ids=torch.max(host,1);actual,index=torch.max(x,1);g=torch.full_like(expected,.125)
        actual.backward(upload(g));expected.backward(g);close(actual,expected);close(index,ids);close(x.grad,host.grad)
        checks.append(dict(shape=list(host.shape),full_cpu_rows_values_indices_gradient=True))
    def contracts():
        host=torch.empty(0,2,5,7,dtype=dtype,requires_grad=True);x=upload(host)
        expected,ids=F.max_pool2d(host,2,return_indices=True);actual,index=F.max_pool2d(x,2,return_indices=True)
        actual.backward(torch.empty_like(actual));expected.backward(torch.empty_like(expected))
        close(actual,expected);close(index,ids);close(x.grad,host.grad)
        data=upload(torch.randn(1,2,5,7,dtype=dtype));values,ids=F.max_pool2d(data,2,return_indices=True)
        bad=upload(torch.full(ids.shape,2**40,dtype=torch.int64));gradient=torch.ones_like(values)
        try:torch.ops.aten.max_pool2d_with_indices_backward(gradient,data,[2,2],[2,2],[0,0],[1,1],False,bad)
        except IndexError:pass
        else:raise AssertionError('Invalid high-word pool index accepted')
        close(F.max_pool2d(data,2),F.max_pool2d(data.cpu(),2))
        leaf=upload(torch.randn(1,2,5,7,dtype=dtype,requires_grad=True));values,ids=F.max_pool2d(leaf,2,return_indices=True)
        with torch.no_grad():ids.zero_()
        gradient=torch.ones_like(values);before=npu.stats()['npu_calls']
        try:values.backward(gradient)
        except RuntimeError as exc:assert 'version' in str(exc) or 'inplace' in str(exc)
        else:raise AssertionError('Missing pool index version guard')
        assert npu.stats()['npu_calls']==before
        host=torch.randn(2,5,dtype=dtype,requires_grad=True);x=upload(host)
        a,_=torch.topk(x,0);b,_=torch.topk(host,0)
        a.backward(torch.empty_like(a));b.backward(torch.empty_like(b));close(x.grad,host.grad)
        checks.append(dict(empty_batch_forward_backward=True,invalid_int64_and_recovery=True,
                           saved_index_version=True,empty_topk_gradient=True))
    def training():
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.bias=torch.nn.Parameter(torch.zeros(1,205,1,1));self.head=torch.nn.Linear(205,3)
            def forward(self,x):
                y=F.max_pool2d(x+self.bias,3,2,1)
                return self.head(y.float().mean((-2,-1)).to(x.dtype))
        cpu=Model().to(dtype);model=copy.deepcopy(cpu).to('intel_npu')
        host=(torch.arange(17*19,dtype=torch.float32).view(1,1,17,19)/512).to(dtype).expand(1,205,17,19).clone()
        data=upload(host);opts=[torch.optim.SGD(net.parameters(),lr=.001,foreach=False) for net in (cpu,model)];losses=[]
        for step in range(2):
            for opt in opts:opt.zero_grad(set_to_none=True)
            expected=cpu(host).float().square().mean();actual=model(data).float().square().mean();actual.backward();expected.backward()
            close(actual,expected,False)
            for p,q in zip(model.parameters(),cpu.parameters()):close(p.grad,q.grad,False)
            for opt in opts:opt.step()
            for p,q in zip(model.parameters(),cpu.parameters()):close(p,q,False)
            losses.append(actual.item());print('training',step+1,'passed',flush=True)
        assert losses[-1]<losses[0]
        checks.append(dict(independent_cpu_steps=2,losses=losses,full_parameter_gradient_and_update=True,planes=205))
    error=None
    try:locals()[case]()
    except Exception:error=traceback.format_exc();print(error,flush=True)
    gc.collect();stats=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes',
        'deferred_graph_sessions','cached_graph_blob_bytes','live_graph_factories'))
    result=dict(status='passed' if error is None and healthy else 'failed',case=case,precision=precision,checks=checks,error=error,
        stats=stats,shutdown=final,elapsed_seconds=time.perf_counter()-started,source_sha256=hashes)
    target=root/'npunlock-probe/research/results/pool-capacity';target.mkdir(exist_ok=True)
    (target/f'{case}-{precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(case,precision,result['status'],'seconds',result['elapsed_seconds'],flush=True)
    return int(result['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=('spatial','sampled','planes','extrema','contracts','training'))
    parser.add_argument('--precision',choices=('float32','float16'),default='float32')
    args=parser.parse_args();raise SystemExit(execute(args.case,args.precision))

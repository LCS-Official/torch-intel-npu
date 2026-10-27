"""Bounded gather/scatter, pooling and clipping migration on the real Intel NPU."""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/index-pool'


def model(kind):
    import torch
    if kind == 'cnn':
        return torch.nn.Sequential(torch.nn.Conv2d(2,4,3,padding=1,groups=2),torch.nn.ReLU6(),
                                   torch.nn.AvgPool2d(3,2,1,ceil_mode=True,count_include_pad=False),
                                   torch.nn.AdaptiveAvgPool2d((2,2)),torch.nn.Flatten(),torch.nn.Linear(16,3))
    class Indexed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first=torch.nn.Linear(4,6);self.head=torch.nn.Linear(6,3)
            self.low=torch.nn.Parameter(torch.full((1,6),-.25));self.high=torch.nn.Parameter(torch.full((1,6),.25))
            self.register_buffer('indices',torch.tensor([[0,2,2,5],[1,1,3,4]],dtype=torch.int64))
        def forward(self,x):
            value=self.first(x)
            selected=torch.gather(value,1,self.indices)
            combined=torch.zeros_like(value).scatter_add(1,self.indices,selected)
            return self.head(torch.clamp(combined,self.low,self.high))
    return Indexed()


def execute(case, precision):
    import torch
    import torch.nn.functional as F
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.manual_seed(359)
    dtype = getattr(torch, precision)
    rows = []

    def upload(x):
        return x.detach().to('intel_npu', memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)

    def close(x, y, exact=False):
        torch.testing.assert_close(x.detach().cpu(), y.detach(), equal_nan=True,
                                   rtol=0 if exact else .004 if dtype == torch.float16 else 5e-5,
                                   atol=0 if exact else .0005 if dtype == torch.float16 else 6e-6)

    def resident(before, flags=0, metadata=0):
        after = npu.stats()
        assert after['host_download_copies']-before['host_download_copies'] == flags
        assert after['host_device_copies'] == before['host_device_copies']
        assert after['host_upload_copies']-before['host_upload_copies'] == after['cached_scalar_buffers']-before['cached_scalar_buffers']+metadata

    def probe():
        x = torch.ones(2, 3, device='intel_npu')
        index = torch.tensor([[0, 1], [2, 0]], dtype=torch.int64, device='intel_npu')
        image = torch.ones(1, 2, 4, 4, device='intel_npu')
        actions = dict(gather=lambda: x.gather(1, index),
                       scatter_add=lambda: x.scatter_add(1, index, x[:, :2].clone()),
                       avg_pool=lambda: F.avg_pool2d(image, 2),
                       adaptive_pool=lambda: F.adaptive_avg_pool2d(image, (2, 2)),
                       clamp=lambda: x.clamp(-.5, .5), minimum=lambda: torch.minimum(x, x))
        for name, action in actions.items():
            try:
                action(); rows.append(dict(name=name, status='supported'))
            except Exception as error:
                rows.append(dict(name=name, status='unsupported', error=str(error).splitlines()[0]))
        return dict(probes=rows)

    def indexed():
        configurations = 0
        for shape in ((5,), (2, 4), (2, 3, 4), (1, 2, 3, 4)):
            for axis in range(len(shape)):
                index_shape = list(shape); index_shape[axis] = 3
                ids = torch.randint(0, shape[axis], index_shape)
                index = upload(ids)
                host = (torch.randn(shape, dtype=dtype)/8).requires_grad_()
                x = upload(host)
                for negative in (False, True):
                    dim = axis-len(shape) if negative else axis
                    host.grad = x.grad = None
                    before = npu.stats()
                    actual = torch.gather(x, dim, index)
                    actual.sum().backward()
                    other_rows = host.numel()//shape[axis]
                    count,extent=ids.numel(),host.numel()
                    flags=1+(count+4095)//4096+((count+2047)//2048)*((extent+8191)//8192)+((extent+65534)//65535)*((count+21841)//21842)
                    resident(before,flags,metadata=len(shape)-1)
                    expected = host.gather(dim, ids); expected.sum().backward()
                    close(actual, expected, True); close(x.grad, host.grad)
                    src = (torch.randn(index_shape, dtype=dtype)/8).requires_grad_()
                    source = upload(src)
                    host.grad = x.grad = None
                    before = npu.stats()
                    actual = torch.scatter_add(x, dim, index, source)
                    actual.sum().backward()
                    resident(before,flags,metadata=len(shape)-1)
                    expected = host.scatter_add(dim, ids, src); expected.sum().backward()
                    close(actual, expected); close(x.grad, host.grad); close(source.grad, src.grad)
                    with torch.no_grad():
                        out = torch.empty_like(x)
                        assert torch.scatter_add(x, dim, index, source, out=out) is out
                        close(out, expected)
                    configurations += 1
        # CPU scatter_add initializes its FP32 accumulator with self before updates.
        host = torch.tensor([65504.], dtype=dtype)
        src = torch.tensor([-65504., .01], dtype=dtype)
        ids = torch.zeros(2, dtype=torch.int64)
        x, source, index = upload(host), upload(src), upload(ids)
        close(x.scatter_add(0, index, source), host.scatter_add(0, ids, src), True)
        with torch.no_grad():
            x.scatter_add_(0, index, source)
        close(x, host.scatter_add(0, ids, src), True)
        return dict(configurations=configurations, duplicate_indices=True, initialized_accumulation_exact=True)

    def extrema():
        for operation in (torch.maximum, torch.minimum):
            for shape_x, shape_y in (((2, 3), (2, 3)), ((2, 1), (1, 3)), ((), (2, 3))):
                host = (torch.randn(shape_x, dtype=dtype)/8).requires_grad_()
                other = (torch.randn(shape_y, dtype=dtype)/8).requires_grad_()
                x, y = upload(host), upload(other)
                before = npu.stats(); actual = operation(x, y); actual.sum().backward(); resident(before)
                expected = operation(host, other); expected.sum().backward()
                close(actual, expected, True); close(x.grad, host.grad, True); close(y.grad, other.grad, True)
            host = torch.tensor([float('nan'), 1., 2., -0., float('inf'), -float('inf')], dtype=dtype, requires_grad=True)
            other = torch.tensor([1., float('nan'), 2., 0., float('inf'), 1.], dtype=dtype, requires_grad=True)
            x, y = upload(host), upload(other)
            actual = operation(x, y); actual.sum().backward()
            expected = operation(host, other); expected.sum().backward()
            close(actual, expected, True); close(x.grad, host.grad, True); close(y.grad, other.grad, True)
            with torch.no_grad():
                out = torch.empty_like(x); assert operation(x, y, out=out) is out; close(out, expected, True)
        return dict(broadcast_forward_backward=True, equal_split_gradients=True, nan_gradient_semantics=True)

    def clipping():
        count = 0
        for bounds in ((-1., 1.), (2., 1.), (0., 0.), (None, 1.), (-1., None), (float('nan'), 1.)):
            for tensor_bounds in (False, True):
                host = torch.tensor([-2., -1., 0., 1., 2., float('nan')], dtype=dtype, requires_grad=True)
                limits = [None if b is None else torch.tensor(b, dtype=dtype, requires_grad=True) if tensor_bounds else b for b in bounds]
                x = upload(host); actual_limits = [upload(b) if isinstance(b, torch.Tensor) else b for b in limits]
                before = npu.stats(); actual = torch.clamp(x, *actual_limits); actual.sum().backward(); resident(before)
                expected = torch.clamp(host, *limits); expected.sum().backward()
                close(actual, expected, True); close(x.grad, host.grad, True)
                for a, b in zip(actual_limits, limits):
                    if isinstance(b, torch.Tensor): close(a.grad, b.grad, True)
                count += 1
        for operation in (torch.clamp_min, torch.clamp_max):
            for tensor_bound in (False, True):
                host = torch.tensor([-1., 0., 1., float('nan')], dtype=dtype, requires_grad=True)
                x = upload(host); bound = torch.tensor(0., dtype=dtype, requires_grad=True) if tensor_bound else 0.
                actual_bound = upload(bound) if tensor_bound else bound
                actual = operation(x, actual_bound); actual.sum().backward()
                expected = operation(host, bound); expected.sum().backward()
                close(actual, expected, True); close(x.grad, host.grad, True)
                if tensor_bound: close(actual_bound.grad, bound.grad, True)
        for module in (torch.nn.Hardtanh(), torch.nn.ReLU6()):
            host = torch.tensor([-2., -1., 0., 1., 6., 7., float('nan')], dtype=dtype, requires_grad=True)
            x = upload(host); actual = module(x); actual.sum().backward()
            expected = module(host); expected.sum().backward()
            close(actual, expected, True); close(x.grad, host.grad, True)
        p = torch.nn.Parameter(torch.zeros(5, dtype=dtype, device='intel_npu'))
        p.grad = upload(torch.tensor([-2., -.5, 0., .5, 2.], dtype=dtype))
        torch.nn.utils.clip_grad_value_([p], .5, foreach=False)
        close(p.grad, torch.tensor([-.5, -.5, 0., .5, .5], dtype=dtype), True)
        return dict(clamp_cases=count, one_bound_and_hardtanh=True, clip_grad_value=True)

    def pools():
        from torch_npu_geometry import avg_pool2d_geometry
        import numpy as np
        count = 0
        for shape in ((2, 5, 6), (2, 2, 5, 6)):
            for kernel, stride, padding, ceil, include, divisor in (
                    (2, 2, 0, False, True, None), (3, 2, 1, False, True, None),
                    (3, 2, 1, True, False, None), ((2, 3), (2, 2), (1, 1), True, True, None),
                    (3, 2, 1, True, True, 7), (2, 2, 0, False, True, -2)):
                host = (torch.randn(shape, dtype=dtype)/8).requires_grad_(); x = upload(host)
                divisors = avg_pool2d_geometry(shape[-2:], kernel, (stride,) if isinstance(stride,int) else stride,
                                              padding, ceil, include, divisor)[-1]
                before = npu.stats()
                actual = F.avg_pool2d(x, kernel, stride, padding, ceil, include, divisor)
                actual.sum().backward(); resident(before, metadata=int(not np.all(divisors == divisors.flat[0])))
                expected = F.avg_pool2d(host, kernel, stride, padding, ceil, include, divisor); expected.sum().backward()
                close(actual, expected); close(x.grad, host.grad)
                count += 1
                if shape == (2,2,5,6) and divisor == -2:
                    with torch.no_grad():
                        out=torch.empty_like(actual)
                        assert torch.ops.aten.avg_pool2d.out(x,[2,2],[2,2],[0,0],False,True,-2,out=out) is out
                        close(out,expected)
        for shape in ((2, 7), (2, 2, 7)):
            host = torch.randn(shape, dtype=dtype).requires_grad_(); x = upload(host)
            actual = F.avg_pool1d(x, 3, 2, 1, True, False); actual.sum().backward()
            expected = F.avg_pool1d(host, 3, 2, 1, True, False); expected.sum().backward()
            close(actual, expected); close(x.grad, host.grad)
            with torch.no_grad():
                out=torch.empty_like(actual)
                assert torch.ops.aten.avg_pool1d.out(x,[3],[2],[1],True,False,out=out) is out
                close(out,expected)
        return dict(avg_pool2d_cases=count, avg_pool1d_cases=2, no_payload_downloads=True)

    def adaptive():
        count = 0
        for shape in ((2, 3, 5), (2, 2, 3, 5)):
            for size in ((1, 1), (2, 3), (4, 2)):
                host = (torch.randn(shape, dtype=dtype)/4).requires_grad_(); x = upload(host)
                before = npu.stats()
                actual = F.adaptive_avg_pool2d(x, size); actual.sum().backward(); resident(before)
                expected = F.adaptive_avg_pool2d(host, size); expected.sum().backward()
                close(actual, expected); close(x.grad, host.grad)
                count += 1
        for shape in ((2, 5), (2, 2, 5)):
            host = torch.randn(shape, dtype=dtype).requires_grad_(); x = upload(host)
            actual = F.adaptive_avg_pool1d(x, 3); actual.sum().backward()
            expected = F.adaptive_avg_pool1d(host, 3); expected.sum().backward()
            close(actual, expected); close(x.grad, host.grad)
            with torch.no_grad():
                out=torch.empty_like(actual)
                assert torch.ops.aten.adaptive_avg_pool1d.out(x,[3],out=out) is out
                close(out,expected)
        return dict(adaptive_2d_cases=count, adaptive_1d_cases=2, overlapping_bins=True)

    def training(kind, amp=False):
        from torch_npu_transformer_demo import cpu_tree
        cpu=model(kind).to(torch.float32 if amp else dtype);network=copy.deepcopy(cpu).to('intel_npu')
        inputs=torch.randn((2,2,5,6) if kind=='cnn' else (2,4)).to(torch.float32 if amp else dtype)/4
        labels=torch.tensor([0,2]);data,target=upload(inputs),upload(labels)
        optimizers=[torch.optim.SGD(m.parameters(),lr=.01,momentum=.5,foreach=False,fused=False) for m in (network,cpu)]
        scalers=[torch.intel_npu.amp.GradScaler(init_scale=8.,growth_interval=2),
                 torch.amp.GradScaler('cpu',init_scale=8.,growth_interval=2)] if amp else None
        history=[];errors=[];saved=io.BytesIO()
        tolerance=dict(rtol=.008,atol=.0006) if amp or dtype==torch.float16 else dict(rtol=6e-5,atol=7e-6)
        for step in range(4):
            losses=[]
            for i,(net,opt,x,y) in enumerate(zip((network,cpu),optimizers,(data,inputs),(target,labels))):
                opt.zero_grad(set_to_none=True);before=npu.stats()
                with torch.autocast('intel_npu' if i==0 else 'cpu',dtype=torch.float16,enabled=amp):
                    output=net(x);loss=F.cross_entropy(output.float(),y,label_smoothing=.1)
                if amp:
                    scalers[i].scale(loss).backward();scalers[i].unscale_(opt)
                else:loss.backward()
                torch.nn.utils.clip_grad_value_(net.parameters(),.25,foreach=False)
                if i==0:resident(before,(1 if kind=='cnn' else 9)+int(amp),metadata=int(kind=='cnn'))
                losses.append(loss)
            torch.testing.assert_close(losses[0].detach().cpu(),losses[1].detach(),**tolerance)
            differences=[]
            for p,q in zip(network.parameters(),cpu.parameters()):
                assert p.grad is not None and q.grad is not None
                torch.testing.assert_close(p.grad.cpu(),q.grad,**tolerance)
                differences.append(float((p.grad.cpu()-q.grad).abs().max()))
                if amp:assert p.dtype==p.grad.dtype==torch.float32
            errors.append(max(differences))
            for i,opt in enumerate(optimizers):
                if amp:scalers[i].step(opt);scalers[i].update()
                else:opt.step()
            for p,q in zip(network.parameters(),cpu.parameters()):
                torch.testing.assert_close(p.detach().cpu(),q.detach(),**tolerance)
            if amp:assert scalers[0].state_dict()==scalers[1].state_dict()
            history.append(float(losses[0].detach().cpu()));print('train',kind,amp,step+1,history[-1],flush=True)
            if step==2:torch.save(dict(model=cpu_tree(network.state_dict()),optimizer=cpu_tree(optimizers[0].state_dict()),
                                       scaler=scalers[0].state_dict() if amp else None),saved)
        saved.seek(0);state=torch.load(saved,weights_only=True)
        restored=model(kind).to(torch.float32 if amp else dtype).to('intel_npu');restored.load_state_dict(state['model'])
        optimizer=torch.optim.SGD(restored.parameters(),lr=.01,momentum=.5,foreach=False,fused=False);optimizer.load_state_dict(state['optimizer'])
        if amp:
            scaler=torch.intel_npu.amp.GradScaler();scaler.load_state_dict(state['scaler'])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('intel_npu',dtype=torch.float16,enabled=amp):loss=F.cross_entropy(restored(data).float(),target,label_smoothing=.1)
        if amp:scaler.scale(loss).backward();scaler.unscale_(optimizer)
        else:loss.backward()
        torch.nn.utils.clip_grad_value_(restored.parameters(),.25,foreach=False)
        if amp:scaler.step(optimizer);scaler.update();assert scaler.state_dict()==scalers[0].state_dict()
        else:optimizer.step()
        assert float(loss.detach().cpu())==history[-1]
        for p,q in zip(network.parameters(),restored.parameters()):
            torch.testing.assert_close(p.detach().cpu(),q.detach().cpu(),rtol=0,atol=0)
            torch.testing.assert_close(optimizers[0].state[p]['momentum_buffer'].cpu(),optimizer.state[q]['momentum_buffer'].cpu(),rtol=0,atol=0)
        return dict(kind=kind,amp=amp,steps=4,parameters=sum(p.numel() for p in network.parameters()),losses=history,
                    max_gradient_abs=errors,tolerance=tolerance,checkpoint_next_step_bitwise=True,compute_copy_contract=True)

    def edges():
        host=torch.tensor([[1.,2.,3.],[4.,5.,6.]],dtype=dtype,requires_grad=True)
        ids=torch.tensor([[2,0],[1,1]],dtype=torch.int64)
        x,index=upload(host),upload(ids)
        actual=torch.take_along_dim(x,index,dim=1);actual.sum().backward()
        expected=torch.take_along_dim(host,ids,dim=1);expected.sum().backward()
        close(actual,expected,True);close(x.grad,host.grad,True)
        with torch.no_grad():
            out=torch.empty_like(actual)
            assert torch.gather(x,1,index,out=out) is out
            close(out,expected,True)
            assert torch.take_along_dim(x,index,dim=1,out=out) is out
            close(out,expected,True)
        scalar=torch.tensor(2.,dtype=dtype,requires_grad=True);x=upload(scalar);index=upload(torch.tensor(0))
        actual=torch.take_along_dim(x,index);actual.sum().backward()
        expected=torch.take_along_dim(scalar,torch.tensor(0));expected.sum().backward()
        close(actual,expected,True);close(x.grad,scalar.grad,True)
        tiny_host=torch.tensor([[1.,2.,3.]],dtype=dtype);tiny_index=torch.tensor([[2**63-1]],dtype=torch.int64)
        close(torch.take_along_dim(upload(tiny_host),upload(tiny_index),dim=1),
              torch.take_along_dim(tiny_host,tiny_index,dim=1),True)
        for op in (torch.maximum,torch.minimum):
            host=torch.ones(1,dtype=torch.float16,requires_grad=True);other=torch.tensor(1.,requires_grad=True)
            x,y=upload(host),upload(other);seed=torch.tensor([2**-24],dtype=torch.float16)
            actual=op(x,y);actual.backward(upload(seed));expected=op(host,other);expected.backward(seed)
            close(x.grad,host.grad,True);close(y.grad,other.grad,True)
        for shape_x,shape_y in (((2,1),(1,3)),((3,),())):
            host=torch.tensor([[.1],[-.2]] if shape_x==(2,1) else [.1,-.2,.3],dtype=torch.float16)
            other=torch.full(shape_y,.2,dtype=torch.float32,requires_grad=True);host.requires_grad_()
            x,y=upload(host),upload(other)
            for op in (torch.maximum,torch.minimum):
                host.grad=other.grad=x.grad=y.grad=None
                actual=op(x,y);actual.sum().backward();expected=op(host,other);expected.sum().backward()
                close(actual,expected,True);close(x.grad,host.grad,True);close(y.grad,other.grad,True)
        base=torch.randn(2,8,dtype=dtype,requires_grad=True);low=torch.tensor([[-.25],[-.5]],dtype=dtype,requires_grad=True)
        high=torch.tensor([[.25],[.5]],dtype=dtype,requires_grad=True)
        x,a,b=upload(base),upload(low),upload(high)
        actual=torch.clamp(x[:,::2],a,b);actual.sum().backward()
        expected=torch.clamp(base[:,::2],low,high);expected.sum().backward()
        close(actual,expected,True);close(x.grad,base.grad,True);close(a.grad,low.grad,True);close(b.grad,high.grad,True)
        with torch.no_grad():
            out=torch.empty_like(actual)
            assert torch.clamp(x[:,::2],a,b,out=out) is out
            close(out,expected,True)
        host=torch.ones(1,dtype=dtype,requires_grad=True);ids=torch.zeros(128,dtype=torch.int64)
        x,index=upload(host),upload(ids);seed=torch.full((128,),2**-24,dtype=dtype)
        actual=x.gather(0,index);actual.backward(upload(seed));expected=host.gather(0,ids);expected.backward(seed)
        close(x.grad,host.grad,True)
        value=x.gather(0,index)
        with torch.no_grad():index.fill_(0)
        try:value.sum().backward()
        except RuntimeError as error:assert 'version' in str(error) or 'inplace' in str(error)
        else:raise AssertionError('Index mutation was not detected')
        return dict(take_along_dim=True,mixed_precision=True,partial_clamp=True,subnormal_repeated_gradient=True,index_version_guard=True)

    def index_wrap():
        samples=[-2**63,2**63-1,-2**60,2**60,-4,-3,-1,0,2,3,4]
        ids=torch.tensor((samples*((63*521+10)//11))[:63*521],dtype=torch.int64).view(63,521)
        host=(torch.arange(63*3,dtype=dtype).view(63,3)/8).requires_grad_();x,index=upload(host),upload(ids)
        before=npu.stats();actual=torch.take_along_dim(x,index,dim=1);actual.sum().backward();resident(before,2+2*63)
        expected=torch.take_along_dim(host,ids,dim=1);expected.sum().backward()
        close(actual,expected,True);close(x.grad,host.grad,True)
        with torch.no_grad():
            ids=torch.tensor([[2**63-1,-2**63,-1]],dtype=torch.int64);index=upload(ids)
            close(torch.take_along_dim(x,index,dim=-1),torch.take_along_dim(host,ids,dim=-1),True)
            flat_ids=torch.tensor([[0,188,3]],dtype=torch.int64)
            close(torch.take_along_dim(x,upload(flat_ids)),torch.take_along_dim(host,flat_ids),True)
        return dict(signed_int64_extremes=True,indices=63*521,wrap_packets=2,broadcast_and_flatten=True)

    def contracts():
        x=torch.ones(2,3,dtype=dtype,device='intel_npu');index=torch.zeros(2,2,dtype=torch.int64,device='intel_npu')
        image=torch.ones(1,2,5,6,dtype=dtype,device='intel_npu');wrong=torch.empty(1,dtype=dtype,device='intel_npu')
        many=torch.ones(129,3,dtype=dtype,device='intel_npu');many_ids=torch.zeros(129,2,dtype=torch.int64,device='intel_npu')
        src=torch.ones(2,2,dtype=dtype,device='intel_npu')
        actions=[lambda:x.gather(2,index),lambda:torch.gather(x,1,index,sparse_grad=True),
                 lambda:x.gather(1,index[:1]),lambda:many.gather(1,many_ids),
                 lambda:torch.scatter_add(x,1,index,x),lambda:torch.gather(x,1,index,out=wrong),
                 lambda:F.avg_pool2d(image,0),lambda:F.avg_pool2d(image,2,padding=2),
                 lambda:F.avg_pool2d(image,2,divisor_override=0),lambda:F.adaptive_avg_pool2d(image,(9,8)),
                 lambda:torch.clamp(x),lambda:torch.clamp(x,-.5,.5,out=wrong)]
        for operation in actions:
            before=npu.stats()
            try:operation()
            except (ValueError,IndexError,RuntimeError,NotImplementedError,TypeError):pass
            else:raise AssertionError('Invalid parameters accepted')
            assert npu.stats()['npu_calls']==before['npu_calls']
        bad=upload(torch.tensor([[0,0],[0,2**40]],dtype=torch.int64))
        for op in (lambda:x.gather(1,bad),lambda:x.scatter_add(1,bad,src),lambda:torch.take_along_dim(x,bad)):
            try:op()
            except IndexError:pass
            else:raise AssertionError('Huge INT64 index accepted')
        return dict(metadata_rejections=len(actions),device_index_rejections=3)

    actions = dict(probe=probe, indexed=indexed, extrema=extrema, clipping=clipping, pools=pools, adaptive=adaptive,
                   cnn=lambda:training('cnn'),index_model=lambda:training('index'),
                   cnn_amp=lambda:training('cnn',True),index_amp=lambda:training('index',True),edges=edges,contracts=contracts,
                   **{'index-wrap':index_wrap})
    before = npu.stats()
    try:
        detail = actions[case]()
        status = 'passed'
    except Exception as error:
        status = 'failed'; detail = dict(error=str(error), traceback=traceback.format_exc())
    gc.collect()
    after = npu.stats()
    if after['live_buffers']-after['cached_scalar_buffers'] != before['live_buffers']-before['cached_scalar_buffers']:
        status = 'failed'; detail['lifecycle_error'] = 'Leaked device storage'
    print(case, precision, status, detail.get('error', '')[:700], flush=True)
    npu.shutdown()
    final = npu.stats()
    if final['context_failed'] or any(final[k] for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes')):
        status = 'failed'
    names = ('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_geometry.py','torch_npu_index_pool_demo.py',
             'npunlock/tests/fixtures/npu4000/resident_index.c','npunlock/tests/fixtures/npu4000/resident/manifest.json',
             'npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/_torch_intel_npu.pyd')
    return dict(status=status, case=case, precision=precision, details=detail, checks=rows,
                npu_calls=after['npu_calls']-before['npu_calls'], final_stats=final,
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('probe','indexed','extrema','clipping','pools','adaptive','cnn','index_model',
                                         'cnn_amp','index_amp','edges','contracts','index-wrap'), default='probe')
    parser.add_argument('--precision', choices=('float16','float32'), default='float32')
    parser.add_argument('--label', default='after')
    args = parser.parse_args()
    result = execute(args.case, args.precision)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{args.label}-{args.case}-{args.precision}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    raise SystemExit(result['status'] != 'passed')

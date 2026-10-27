"""Device byte-DMA cat/stack, view reads, slice/select gradients and model migration."""
import argparse
import copy
import gc
import hashlib
import json
from math import prod
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/layout-migration'
CASES = ('cat', 'stack', 'views', 'edges', 'contracts', 'training')


def execute(case, precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(127)
    checks = []

    def upload(x):
        return x.detach().to('intel_npu', memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)

    def close(a, b, exact=True):
        torch.testing.assert_close(a.detach().cpu(), b.detach().cpu(), equal_nan=True,
                                   rtol=0 if exact else 3e-5, atol=0 if exact else 3e-6)

    def no_transfers(before):
        after = npu.stats()
        assert all(after[k] == before[k] for k in ('host_upload_copies','host_download_copies','host_device_copies'))

    def check(name, function):
        before = npu.stats()
        row = dict(name=name)
        try:
            row.update(function() or {})
            row['status'] = 'passed'
        except Exception as error:
            row.update(status='failed', error=str(error), traceback=traceback.format_exc())
        gc.collect()
        after = npu.stats()
        row['npu_calls'] = after['npu_calls']-before['npu_calls']
        row['live_buffer_delta'] = after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
        if row['live_buffer_delta'] or after['context_failed']:
            row.update(status='failed', lifecycle_error='Live buffers or failed context')
        checks.append(row)
        print(name, row['status'], row.get('error','')[:350], flush=True)

    def assembly(dtype, stack):
        function = torch.stack if stack else torch.cat
        count = 0
        for rank in range(0 if stack else 1, 4 if stack else 5):
            shape = tuple(range(2,rank+2))
            for axis in range(rank+int(stack)):
                for negative in (False,True):
                    inputs = []
                    for index in range(2):
                        current = list(shape)
                        if not stack:
                            current[axis] += index
                        value = torch.arange(prod(current),dtype=dtype).reshape(current)+index
                        inputs.append(value.requires_grad_(dtype != torch.int64 and index == 0))
                    tensors = [upload(x) for x in inputs]
                    dim = axis-rank-int(stack) if negative else axis
                    expected = function(inputs,dim)
                    seed = torch.arange(expected.numel(),dtype=dtype).reshape(expected.shape)/16 if dtype != torch.int64 else None
                    gradient = upload(seed) if seed is not None else None
                    before = npu.stats()
                    actual = function(tensors,dim)
                    if seed is not None:
                        actual.backward(gradient)
                    no_transfers(before)
                    if seed is not None:
                        expected.backward(seed)
                        close(tensors[0].grad,inputs[0].grad)
                        assert tensors[1].grad is None
                    close(actual,expected)
                    assert actual.is_contiguous()
                    assert all(actual.untyped_storage()._cdata != x.untyped_storage()._cdata for x in tensors)
                    with torch.no_grad():
                        out = torch.empty_like(actual)
                        assert function(tensors,dim,out=out) is out
                        close(out,expected)
                    count += 1
        value = upload(torch.arange(6,dtype=dtype).reshape(2,3))
        single = function((value,),-1)
        assert single.untyped_storage()._cdata != value.untyped_storage()._cdata
        close(single,function((value.cpu(),),-1))
        # Repeated input appears twice in the graph; both gradient contributions matter.
        if dtype != torch.int64:
            host = torch.arange(6,dtype=dtype).reshape(2,3).requires_grad_()
            x = upload(host)
            function((x,x),1).sum().backward()
            function((host,host),1).sum().backward()
            close(x.grad,host.grad)
        host = torch.arange(24,dtype=dtype).reshape(4,6).requires_grad_(dtype != torch.int64)
        x = upload(host)
        view, reference = x.t()[::2],host.t()[::2]
        actual, expected = function((view,view),-1),function((reference,reference),-1)
        close(actual,expected)
        if dtype != torch.int64:
            seed = torch.arange(expected.numel(),dtype=dtype).reshape(expected.shape)/8
            gradient = upload(seed.transpose(-1,-2)).transpose(-1,-2)
            before = npu.stats()
            actual.backward(gradient)
            no_transfers(before)
            expected.backward(seed)
            close(x.grad,host.grad)
        return dict(cases=count, one_and_two_inputs=True, single_input_storage_independent=True,
                    positive_negative_axes=True, first_order=dtype != torch.int64, out=True,
                    transposed_partial_inputs=True, strided_output_gradient=dtype != torch.int64,
                    zero_payload_transfers=True)

    def views(dtype):
        factories = (
            lambda x:x[:,1:6:2], lambda x:x[1], lambda x:x[:,1:5],
            lambda x:x[1:3,2:5], lambda x:x.t()[1:5:2], lambda x:x.narrow(1,1,3),
            lambda x:torch.select(x,-1,-2), lambda x:x[1,2],
        )
        count = 0
        for factory in factories:
            host = torch.arange(24,dtype=dtype).reshape(4,6).requires_grad_(dtype != torch.int64)
            x = upload(host)
            view, reference = factory(x), factory(host)
            assert view.untyped_storage()._cdata == x.untyped_storage()._cdata
            assert view.storage_offset() == reference.storage_offset() and view.stride() == reference.stride()
            seed = torch.arange(view.numel(),dtype=dtype).reshape(view.shape)/8 if dtype != torch.int64 else None
            gradient = upload(seed) if seed is not None else None
            before = npu.stats()
            result = view.clone()
            if seed is not None:
                result.backward(gradient)
            no_transfers(before)
            assert result.untyped_storage()._cdata != x.untyped_storage()._cdata
            assert result.stride() == reference.clone().stride()
            close(result,reference)
            close(view,reference)
            if seed is not None:
                reference.clone().backward(seed)
                close(x.grad,host.grad)
            with torch.no_grad():
                target = torch.empty(view.shape,dtype=dtype,device='intel_npu')
                assert target.copy_(view) is target
                close(target,reference)
                stacked = torch.stack((view,view),-1)
                close(stacked,torch.stack((reference,reference),-1))
            count += 1
        for name, operation in (
                ('slice_copy',lambda x:torch.slice_copy(x,1,1,6,2)),
                ('select_copy',lambda x:torch.select_copy(x,1,-2)),
                ('narrow_copy',lambda x:torch.narrow_copy(x,1,1,3))):
            host = torch.arange(24,dtype=dtype).reshape(4,6).requires_grad_(dtype != torch.int64 and name != 'narrow_copy')
            x = upload(host)
            actual, expected = operation(x), operation(host)
            assert actual.untyped_storage()._cdata != x.untyped_storage()._cdata
            close(actual,expected)
            if host.requires_grad:
                actual.sum().backward()
                expected.sum().backward()
                close(x.grad,host.grad)
            with torch.no_grad():
                target = torch.empty_like(actual)
                if name == 'slice_copy':
                    returned = torch.slice_copy(x,1,1,6,2,out=target)
                elif name == 'select_copy':
                    returned = torch.select_copy(x,1,-2,out=target)
                else:
                    returned = torch.narrow_copy(x,1,1,3,out=target)
                assert returned is target
                close(target,expected)
        if dtype != torch.int64:
            host = torch.arange(24,dtype=dtype).reshape(4,6).requires_grad_()
            x = upload(host)
            other = torch.float32 if dtype == torch.float16 else torch.float16
            x[:,1::2].to(other).sum().backward()
            host[:,1::2].to(other).sum().backward()
            close(x.grad,host.grad)
            host = torch.ones(1,dtype=dtype,requires_grad=True)
            x = upload(host)
            x.expand(2,3).clone().sum().backward()
            host.expand(2,3).clone().sum().backward()
            close(x.grad,host.grad)
            for function in (lambda x:torch.split(x,[1,3,2],dim=1), lambda x:torch.unbind(x,dim=0)):
                host = torch.arange(24,dtype=dtype).reshape(4,6).requires_grad_()
                x = upload(host)
                parts, refs = function(x), function(host)
                assert all(v.untyped_storage()._cdata == x.untyped_storage()._cdata for v in parts)
                # Leave one split output unused to exercise zero-gradient creation.
                torch.stack([v.clone().sum() for v in parts[1:]]).sum().backward()
                torch.stack([v.clone().sum() for v in refs[1:]]).sum().backward()
                close(x.grad,host.grad)
        return dict(view_cases=count, composite_copy_operators=3, slice_select_backward=dtype != torch.int64,
                    composite_copy_out=True, partial_dtype_backward=dtype != torch.int64,
                    broadcast_clone_backward=dtype != torch.int64, split_unbind_backward=dtype != torch.int64,
                    aliases_preserved=True, zero_payload_transfers=True)

    def edges(dtype):
        if dtype == torch.int64:
            host = torch.tensor([-(2**63),2**63-1,2**53+1,-2**53-1,0x7fc0012300000001,0])
        else:
            bits = torch.tensor([0,-2147483648,1,0x7f800000,-8388608,0x7fc00123,0x3f800000],dtype=torch.int32)
            host = bits.view(torch.float32).to(dtype)
        x = upload(host)
        actual = torch.cat((x[1:],x[:1])).cpu()
        expected = torch.cat((host[1:],host[:1]))
        assert actual.numpy().tobytes() == expected.numpy().tobytes()
        many = torch.cat([x]*65)
        close(many,torch.cat([host]*65))
        large = torch.arange(70000,dtype=torch.int64).remainder(127).to(dtype).reshape(10000,7)
        z = upload(large)
        part = z[1::5,1::3].clone()
        close(part,large[1::5,1::3])
        permuted = upload(torch.arange(24,dtype=dtype).reshape(4,6)).t()
        result = permuted.clone()
        assert result.stride() == permuted.stride()
        close(result,torch.arange(24,dtype=dtype).reshape(4,6).t())
        repeated = x[:1].expand(2,host.numel())
        close(repeated.clone(),host[:1].expand(2,host.numel()))
        if dtype == torch.float16:
            codes = torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
            full = upload(codes)
            result = torch.cat((full[::2],full[1::2])).cpu()
            expected = torch.cat((codes[::2],codes[1::2]))
            assert result.numpy().tobytes() == expected.numpy().tobytes()
        # Native binding must continue rejecting partial allocations.
        before = npu.stats()['npu_calls']
        try:
            npu._C.copy(torch.empty(2,device='intel_npu',dtype=dtype),x[:2],False)
        except (ValueError,RuntimeError):
            pass
        else:
            raise AssertionError('Native whole-allocation guard bypassed')
        assert npu.stats()['npu_calls'] == before
        return dict(raw_bits_exact=True, int64_high_words=dtype == torch.int64, inputs_65=True,
                    source_elements=70000, gather_span_tiling=True, native_binding_guard=True,
                    all_half_encodings=dtype == torch.float16, broadcast_read=True)

    def contracts():
        x = upload(torch.arange(12,dtype=torch.float32).reshape(3,4))
        half = upload(torch.ones(3,4,dtype=torch.float16))
        wrong = torch.empty(3,4,device='intel_npu')
        rank4 = torch.empty((1,)*npu.MAX_RANK,device='intel_npu')
        large = torch.empty(npu._storage_limit(torch.float32),device='intel_npu')
        actions = {
            'cat-axis':lambda:torch.cat((x,x),2), 'stack-axis':lambda:torch.stack((x,x),-4),
            'cat-shape':lambda:torch.cat((x,x.t()),0), 'stack-shape':lambda:torch.stack((x,x.t())),
            'mixed-device':lambda:torch.cat((x,torch.ones(3,4))),
            'cat-scalar':lambda:torch.cat((x[0,0],x[0,0])), 'stack-rank65':lambda:torch.stack((rank4,rank4)),
            'output-limit':lambda:torch.cat((large,large)), 'out-shape':lambda:torch.cat((x,x),out=wrong),
            'out-dtype':lambda:torch.cat((x,x),out=torch.empty(6,4,dtype=torch.float16,device='intel_npu')),
            'out-alias':lambda:torch.cat((x,),out=x),
            'narrow-out-alias':lambda:torch.narrow_copy(x,0,0,3,out=x),
            'narrow-out-shape':lambda:torch.narrow_copy(x,1,1,2,out=wrong),
            'out-partial':lambda:torch.cat((x,),out=torch.empty(24,device='intel_npu')[:12].view(3,4)),
            'bad-backward-shape':lambda:torch.ops.aten.slice_backward(x,[3,5],1,1,4,1),
            'bad-backward-axis':lambda:torch.ops.aten.select_backward(x,[2,3,4],3,0),
            'out-grad':lambda:torch.cat((x.detach().requires_grad_(),),out=wrong),
        }
        errors = {}
        for name, action in actions.items():
            before = npu.stats()['npu_calls']
            try:
                action()
            except (ValueError,RuntimeError,NotImplementedError,IndexError) as error:
                errors[name] = str(error).splitlines()[0][:220]
            else:
                raise AssertionError('Missing rejection: '+name)
            assert npu.stats()['npu_calls'] == before, name
        # Views continue sharing the base's version counter after read materialization.
        host = torch.ones(3,4,requires_grad=True)
        base = upload(host)
        view = base[:,1:3]
        version = base._version
        with torch.no_grad():
            base.add_(1)
        assert view._version == base._version == version+1
        close(view,torch.full((3,2),2.))
        return dict(rejections=errors, no_submission_on_metadata_rejection=True, shared_version_counter=True)

    def training():
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.left, self.right, self.head = torch.nn.Linear(6,4),torch.nn.Linear(6,4),torch.nn.Linear(12,3)

            def forward(self,x):
                a,b = torch.relu(self.left(x)),torch.relu(self.right(x))
                combined = torch.cat((a[:,::2],b[:,1::2]),dim=1)
                folded = torch.stack((a,b),dim=1).mean(1)
                return self.head(torch.cat((combined,folded,a),dim=-1))
        cpu = Model()
        model = copy.deepcopy(cpu).to('intel_npu')
        host,target = torch.randn(4,6)/4,torch.tensor([0,1,2,1])
        x,labels = upload(host),upload(target)
        optimizers = [torch.optim.SGD(net.parameters(),lr=.03,momentum=.5,foreach=False,fused=False) for net in (cpu,model)]
        losses = []
        for _ in range(4):
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            expected = torch.nn.functional.cross_entropy(cpu(host),target)
            actual = torch.nn.functional.cross_entropy(model(x),labels)
            losses.append(actual.item())
            expected.backward()
            actual.backward()
            close(actual,expected,False)
            for p,q in zip(model.parameters(),cpu.parameters()):
                close(p.grad,q.grad,False)
            for opt in optimizers:
                opt.step()
            for p,q in zip(model.parameters(),cpu.parameters()):
                close(p,q,False)
        assert losses[-1] < losses[0]
        return dict(independent_steps=4, losses=losses, precision='FP32', ordinary_branched_model=True)

    dtypes = {'float16':torch.float16,'float32':torch.float32,'int64':torch.int64}
    for name in CASES if case == 'all' else (case,):
        if name in ('contracts','training'):
            check(name,{'contracts':contracts,'training':training}[name])
        else:
            for label in dtypes if precision == 'all' else (precision,):
                dtype = dtypes[label]
                check(f'{name}-{label}',lambda n=name,d=dtype: assembly(d,n=='stack') if n in ('cat','stack') else {'views':views,'edges':edges}[n](d))
    gc.collect()
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    sources = ('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_layout_demo.py','npunlock-probe/torch-device/_torch_intel_npu.pyd')
    return dict(status='passed' if all(c['status']=='passed' for c in checks) else 'failed',checks=checks,final_stats=final,
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('all',)+CASES,default='all')
    parser.add_argument('--precision',choices=('all','float16','float32','int64'),default='all')
    parser.add_argument('--label',default='after')
    args = parser.parse_args()
    if args.case == 'all':
        # Each process stays inside the backend's deliberately bounded graph cache.
        status = 0
        for case in CASES:
            precisions = ('float16','float32','int64') if args.precision == 'all' else (args.precision,)
            for precision in ('all',) if case in ('contracts','training') else precisions:
                result = subprocess.run([sys.executable,'-B',__file__,'--case',case,'--precision',precision,'--label',args.label])
                status |= int(result.returncode != 0)
        return status
    result = execute(args.case,args.precision)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{args.label}-{args.case}-{args.precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())

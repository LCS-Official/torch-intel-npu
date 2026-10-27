"""Resident INT64 index selection, dense embedding gradients and model migration."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/embedding'
CASES = ('index', 'embedding', 'edges', 'tiled', 'contracts', 'training')


def execute(case):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(113)
    checks = []

    def upload(x):
        return x.detach().to('intel_npu', memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)

    def close(a, b, exact=False):
        torch.testing.assert_close(a.detach().cpu(), b.detach().cpu(), equal_nan=True,
                                   rtol=0 if exact else .003 if a.dtype == torch.float16 else 3e-5,
                                   atol=0 if exact else .0005 if a.dtype == torch.float16 else 3e-6)

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
        print(name, row['status'], row.get('error', '')[:250], flush=True)

    def transfers(before, flags):
        after = npu.stats()
        copies = sum(after[k]-before[k] for k in ('host_upload_copies','host_download_copies','host_device_copies'))
        scalars = after['cached_scalar_buffers']-before['cached_scalar_buffers']
        assert copies == scalars+flags, (copies, scalars, flags)
        assert after['host_download_copies']-before['host_download_copies'] == flags

    def index(dtype):
        count = 0
        for shape, dim in (((4,),0), ((4,3),0), ((3,4),-1), ((2,4,3),1), ((2,3,4,2),-2)):
            for dense in (False, True):
                host = (torch.randn(shape,dtype=dtype)/4).requires_grad_()
                x = upload(host)
                if dense and x.ndim > 1:
                    x = x.detach().transpose(-1,-2).contiguous().transpose(-1,-2).requires_grad_()
                labels = torch.tensor([2,0,2,3])
                indices = upload(labels)
                expected = torch.index_select(host,dim,labels)
                gradient = torch.randn_like(expected)/8
                seed = upload(gradient)
                if dense and seed.ndim > 1:
                    seed = seed.transpose(-1,-2).contiguous().transpose(-1,-2)
                before = npu.stats()
                actual = torch.index_select(x,dim,indices)
                actual.backward(seed)
                transfers(before,2)
                expected.backward(gradient)
                assert actual.is_contiguous() and actual.data_ptr() != x.data_ptr()
                close(actual,expected,True)
                close(x.grad,host.grad)
                with torch.no_grad():
                    out = torch.empty_like(actual)
                    assert torch.index_select(x,dim,indices,out=out) is out
                    close(out,expected,True)
                count += 1
        return dict(precision=str(dtype), cases=count, resident_int64=True, duplicate_index_gradients=True,
                    complete_dense_permutations=True, out=True, flags_per_forward_backward=2)

    def embedding(dtype):
        count = 0
        for shape in ((), (4,), (2,2), (1,2,2)):
            labels = torch.tensor([2,0,2,6]) if shape else torch.tensor(2)
            labels = labels.reshape(shape)
            for padding in (None,0,-1):
                host = (torch.randn(7,3,dtype=dtype)/4).requires_grad_()
                weight,indices = upload(host),upload(labels)
                expected = torch.nn.functional.embedding(labels,host,padding_idx=padding)
                gradient = torch.randn_like(expected)/8
                seed = upload(gradient)
                if seed.ndim > 1:
                    seed = seed.transpose(-1,-2).contiguous().transpose(-1,-2)
                before = npu.stats()
                actual = torch.nn.functional.embedding(indices,weight,padding_idx=padding)
                actual.backward(seed)
                transfers(before,2)
                expected.backward(gradient)
                close(actual,expected,True)
                close(weight.grad,host.grad)
                count += 1
        source = torch.arange(21,dtype=dtype).reshape(7,3)/8
        cpu = torch.nn.Embedding.from_pretrained(source,freeze=True,padding_idx=0)
        model = copy.deepcopy(cpu).to('intel_npu')
        labels = torch.tensor([[0,2,2]])
        close(model(upload(labels)),cpu(labels),True)
        assert not model.weight.requires_grad
        return dict(precision=str(dtype), cases=count, padding_none_zero_negative=True,
                    repeated_indices=True, noncontiguous_output_gradient=True, frozen_module=True)

    def edges(dtype):
        bits = torch.tensor([0,-2147483648,1,0x7f800000,-8388608,0x7fc00123,0x3f800000],dtype=torch.int32)
        host = bits.view(torch.float32).to(dtype)
        labels = torch.tensor([5,1,2,0,6,3,4,5])
        x,indices = upload(host),upload(labels)
        actual = torch.index_select(x,0,indices).cpu()
        expected = torch.index_select(host,0,labels)
        integer = torch.int16 if dtype == torch.float16 else torch.int32
        assert torch.equal(actual.view(integer),expected.view(integer))
        rows = 65528
        large = torch.linspace(-.25,.5,rows,dtype=dtype).requires_grad_()
        device,indices = upload(large),upload(torch.tensor([rows-1]))
        selected = torch.index_select(device,0,indices)
        selected.backward(upload(torch.ones(1,dtype=dtype)))
        close(selected,large[-1:].detach(),True)
        gradient = torch.zeros_like(large)
        gradient[-1] = 1
        close(device.grad,gradient,True)
        data = upload(torch.arange(12,dtype=dtype).reshape(4,3))
        indices = upload(torch.tensor([2,0,2]))
        torch.index_select(data,0,indices)
        indices.copy_(torch.tensor([3,1,0]))
        before = npu.stats()
        result = torch.index_select(data,0,indices)
        transfers(before,1)
        assert npu.stats()['session_creations'] == before['session_creations']
        close(result,torch.arange(12,dtype=dtype).reshape(4,3)[[3,1,0]],True)
        # More than two repeats distinguishes per-add half rounding from FP32 accumulation.
        for function in ('index_select','embedding'):
            for tiny in (False,True):
                host = torch.zeros(4,1,dtype=dtype,requires_grad=True)
                x = upload(host)
                labels = torch.full((128,),2,dtype=torch.int64)
                seed = (torch.full((128,1),2**(-24 if dtype==torch.float16 else -149),dtype=dtype)
                        if tiny else torch.tensor([.3333,-.2,.125,-.1]*32,dtype=dtype).view(128,1))
                indices = upload(labels)
                expected = (torch.index_select(host,0,labels) if function=='index_select' else
                            torch.nn.functional.embedding(labels,host))
                actual = (torch.index_select(x,0,indices) if function=='index_select' else
                          torch.nn.functional.embedding(indices,x))
                expected.backward(seed)
                actual.backward(upload(seed))
                close(x.grad,host.grad,True)
        host = torch.zeros(4,1,dtype=dtype,requires_grad=True)
        x = upload(host)
        labels = torch.tensor([0,2,2])
        seed = torch.tensor([[float('nan')],[.25],[.5]],dtype=dtype)
        torch.nn.functional.embedding(labels,host,padding_idx=0).backward(seed)
        torch.nn.functional.embedding(upload(labels),x,padding_idx=0).backward(upload(seed))
        close(x.grad,host.grad,True)
        return dict(precision=str(dtype), raw_forward_bits=True, packet_elements_boundary=65536,
                    warm_graph_reused_with_changed_indices=True, repeats_128_exact_gradient=True,
                    subnormal_gradient=True, padding_nan_isolated=True)

    def tiled(dtype):
        from torch_npu_geometry import index_tile_width
        rows,width,count=2048,35,32
        labels=torch.tensor([rows-1,0,2,2,17,rows-1,2,1]*4)
        indices=upload(labels);tiles=(width+index_tile_width(rows,width,count)-1)//index_tile_width(rows,width,count)
        cases=0
        for embedding,dim in ((False,0),(False,1),(True,0)):
            shape=(rows,width) if dim==0 else (width,rows)
            host=(torch.randn(shape,dtype=dtype)/4).requires_grad_();x=upload(host)
            expected=torch.nn.functional.embedding(labels,host,padding_idx=0) if embedding else torch.index_select(host,dim,labels)
            seed=torch.randn_like(expected)/8;gradient=upload(seed)
            before=npu.stats()
            actual=torch.nn.functional.embedding(indices,x,padding_idx=0) if embedding else torch.index_select(x,dim,indices)
            actual.backward(gradient);transfers(before,2*tiles);expected.backward(seed)
            close(actual,expected,True);close(x.grad,host.grad,True)
            if embedding:assert not torch.count_nonzero(x.grad.cpu()[0])
            else:
                with torch.no_grad():
                    out=torch.empty_like(actual);assert torch.index_select(x,dim,indices,out=out) is out;close(out,expected,True)
            cases+=1
        # Invalid high words/negative indices must still fail in the first complete-index packet.
        for invalid in (-1,rows,2**32,2**63-1):
            bad=labels.clone();bad[-1]=invalid;device_indices=upload(bad);before=npu.stats()
            try:torch.index_select(x,0,device_indices)
            except IndexError:pass
            else:raise AssertionError('Tiled index accepted an invalid INT64 value')
            transfers(before,1)
        return dict(cases=cases,shape=[rows,width],indices=count,column_tiles=tiles,
                    raw_forward_and_gradient_exact=True,partial_last_tile=True,invalid_indices=4,
                    flags_per_forward_backward=2*tiles)

    def contracts():
        weight = upload(torch.ones(4,3))
        indices = upload(torch.tensor([0,2]))
        float_indices = upload(torch.ones(2))
        wrong = torch.empty(1,device='intel_npu')
        too_large = upload(torch.ones(65529))
        long_indices = upload(torch.zeros(1025,dtype=torch.int64))
        budget_weight = upload(torch.ones(1024,1))
        wide_rank = upload(torch.zeros(1,1,1,1,dtype=torch.int64))
        identity = upload(torch.tensor([0,1,2,3]))
        calls = {
            'index-cpu': lambda: torch.index_select(weight,0,torch.tensor([0,1])),
            'index-dtype': lambda: torch.index_select(weight,0,float_indices),
            'index-rank': lambda: torch.index_select(weight,0,indices.view(1,2)),
            'dim': lambda: torch.index_select(weight,2,indices),
            'partial-index': lambda: torch.index_select(weight,0,indices[:1]),
            'out-shape': lambda: torch.index_select(weight,0,indices,out=wrong),
            'out-alias': lambda: torch.index_select(weight,0,identity,out=weight),
            'packet-bound': lambda: torch.index_select(too_large,0,indices),
            'work-bound': lambda: torch.index_select(budget_weight,0,long_indices),
            'embedding-rank': lambda: torch.nn.functional.embedding(wide_rank,weight),
            'embedding-freq': lambda: torch.nn.functional.embedding(indices,weight,scale_grad_by_freq=True),
            'embedding-sparse': lambda: torch.nn.functional.embedding(indices,weight,sparse=True),
            'embedding-max-norm': lambda: torch.nn.functional.embedding(indices,weight,max_norm=1.),
        }
        before = npu.stats()['npu_calls']
        errors = {}
        for name,function in calls.items():
            try:
                function()
            except (ValueError,RuntimeError,NotImplementedError,IndexError) as error:
                errors[name] = str(error).splitlines()[0][:200]
            else:
                raise AssertionError(f'Expected rejection: {name}')
        assert npu.stats()['npu_calls'] == before
        invalid = []
        for value in (-1,4,2**32,2**53+1,2**63-1,-2**63,0x7fc00000):
            labels = upload(torch.tensor([0,value]))
            before = npu.stats()
            try:
                torch.index_select(weight,0,labels)
            except IndexError as error:
                invalid.append(value)
            else:
                raise AssertionError(f'Accepted invalid index: {value}')
            transfers(before,1)
        tensor = upload(torch.ones(4,3,requires_grad=True))
        output = torch.nn.functional.embedding(indices,tensor)
        seed = upload(torch.ones(2,3))
        indices.copy_(torch.tensor([1,3]))
        before = npu.stats()['npu_calls']
        try:
            output.backward(seed)
        except RuntimeError as error:
            assert 'modified by an inplace operation' in str(error)
        else:
            raise AssertionError('Missing index version guard')
        assert npu.stats()['npu_calls'] == before
        return dict(metadata_rejections=errors, invalid_int64_values=invalid, index_version_guard=True,
                    no_submission_on_metadata_rejection=True)

    def training():
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16,6,padding_idx=0)
                self.norm = torch.nn.LayerNorm(6)
                self.head = torch.nn.Linear(6,3)

            def forward(self,tokens):
                return self.head(torch.relu(self.norm(self.embedding(tokens).mean(1))))
        cpu = Model()
        model = copy.deepcopy(cpu).to('intel_npu')
        tokens = torch.tensor([[0,2,2,3],[1,4,1,0],[5,2,5,2],[6,7,6,7]])
        target = torch.tensor([0,1,2,1])
        data,labels = upload(tokens),upload(target)
        reference,opt = [torch.optim.SGD(net.parameters(),lr=.02,momentum=.5,foreach=False,fused=False) for net in (cpu,model)]
        losses = []
        for _ in range(4):
            reference.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            expected = torch.nn.functional.cross_entropy(cpu(tokens),target)
            actual = torch.nn.functional.cross_entropy(model(data),labels)
            losses.append(actual.item())
            expected.backward()
            actual.backward()
            close(actual,expected)
            for p,q in zip(model.parameters(),cpu.parameters()): close(p.grad,q.grad)
            reference.step()
            opt.step()
            for p,q in zip(model.parameters(),cpu.parameters()): close(p,q)
        assert losses[-1] < losses[0]
        return dict(precision='float32', independent_steps=4, losses=losses, model='Embedding/mean/LayerNorm/ReLU/Linear/CrossEntropy')

    for name in CASES if case == 'all' else (case,):
        if name in ('index','embedding','edges','tiled'):
            for dtype in (torch.float16,torch.float32):
                check(f'{name}-{dtype}',lambda d=dtype,n=name: {'index':index,'embedding':embedding,'edges':edges,'tiled':tiled}[n](d))
        else:
            check(name,{'contracts':contracts,'training':training}[name])
    gc.collect()
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    sources = ('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_embedding_demo.py',
               'npunlock/tests/fixtures/npu4000/resident_index.c','npunlock/tests/fixtures/npu4000/resident/manifest.json',
               'npunlock-probe/torch-device/_torch_intel_npu.pyd')
    return dict(status='passed' if all(r['status']=='passed' for r in checks) else 'failed',checks=checks,final_stats=final,
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('all',)+CASES,default='all')
    parser.add_argument('--label',default='after')
    args = parser.parse_args()
    result = execute(args.case)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{args.label}-{args.case}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(result['status'] != 'passed')

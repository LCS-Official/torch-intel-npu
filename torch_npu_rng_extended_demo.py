"""NPU Bernoulli/randperm contracts, independent Philox references and replay."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import struct
import time
import traceback

from torch_npu_rng_demo import philox
from torch_npu_discrete_demo import integer_reference


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    dtype=getattr(torch,precision);started=time.perf_counter();records=[]
    gen=torch.Generator(device='intel_npu').manual_seed(739)
    cpu_rng=torch.get_rng_state();default_rng=torch.intel_npu.get_rng_state()
    def state():return struct.unpack('<8sQQ',bytes(gen.get_state().tolist()))[1:]
    def resident(before,flags=0):
        after=npu.stats()
        assert after['host_device_copies']==before['host_device_copies']
        assert after['host_download_copies']-before['host_download_copies']==flags
    def bernoulli():
        host=torch.tensor([0.,1.,.25,.5,.75,.125,1.],dtype=dtype).view(1,7)
        p=host.to('intel_npu');prototype=torch.empty((3,7),dtype=dtype,device='intel_npu')
        out=torch.empty_like(prototype)
        actions=(('functional',lambda:torch.bernoulli(p,generator=gen),host),
                 ('tensor-fill',lambda:prototype.bernoulli_(p,generator=gen),host.expand(3,7)),
                 ('tensor-aten',lambda:torch.ops.aten.bernoulli.Tensor(prototype,p,generator=gen),host.expand(3,7)),
                 ('tensor-out',lambda:torch.ops.aten.bernoulli.Tensor_out(prototype,p,generator=gen,out=out),host.expand(3,7)),
                 ('scalar-functional',lambda:torch.ops.aten.bernoulli.p(prototype,.375,generator=gen),torch.full((3,7),.375)),
                 ('scalar-out',lambda:torch.ops.aten.bernoulli.float_out(prototype,.375,generator=gen,out=out),torch.full((3,7),.375)))
        for name,fn,prob in actions:
            seed,offset=state();before=npu.stats();actual=fn();resident(before,0 if name.startswith('scalar') else 1)
            expected=torch.tensor([(philox(seed,offset+i)[0]>>8)*2**-24 for i in range(actual.numel())]).view(prob.shape)<prob.float()
            torch.testing.assert_close(actual.cpu(),expected.to(actual.dtype),rtol=0,atol=0)
            assert state()==(seed,offset+actual.numel())
            records.append(dict(name=name,shape=list(actual.shape),exact_philox=True))
        functional_out=torch.empty_like(p,dtype=torch.int64)
        torch.bernoulli(p,generator=gen,out=functional_out)
        assert set(functional_out.cpu().view(-1).tolist())<={0,1}
        # Both probabilities and prototype derivatives are zero under native Bernoulli autograd.
        leaf=p.detach().requires_grad_();actual=torch.bernoulli(leaf,generator=gen)
        actual.sum().backward();assert torch.count_nonzero(leaf.grad.cpu())==0
        source=torch.ones((3,7),dtype=dtype,device='intel_npu',requires_grad=True)
        probability=p.detach().requires_grad_();target=source.clone()
        target.bernoulli_(probability,generator=gen).sum().backward()
        assert torch.count_nonzero(source.grad.cpu())==0 and torch.count_nonzero(probability.grad.cpu())==0
        for target_dtype in npu._STORAGE_DTYPES:
            target=torch.empty((2,7),dtype=target_dtype,device='intel_npu')
            saved=gen.get_state();seed,offset=state();before=npu.stats()
            target.bernoulli_(p,generator=gen);resident(before,1)
            expected=torch.tensor([(philox(seed,offset+i)[0]>>8)*2**-24 for i in range(14)]).view(2,7)<host.float()
            torch.testing.assert_close(target.cpu(),expected.to(target_dtype),rtol=0,atol=0)
            gen.set_state(saved);replayed=torch.empty_like(target).bernoulli_(p,generator=gen)
            torch.testing.assert_close(replayed.cpu(),target.cpu(),rtol=0,atol=0)
        base=torch.empty((7,3),dtype=dtype,device='intel_npu')
        target=base.t();assert target.bernoulli_(p,generator=gen) is target
        empty=torch.empty((0,7),dtype=dtype,device='intel_npu');saved=gen.get_state()
        empty.bernoulli_(p,generator=gen);assert torch.equal(saved,gen.get_state())
        # A probability read may alias the destination; read everything before overwriting it.
        alias=p.clone();saved=gen.get_state();alias.bernoulli_(alias,generator=gen)
        gen.set_state(saved);reference=torch.bernoulli(p,generator=gen)
        torch.testing.assert_close(alias.cpu(),reference.cpu(),rtol=0,atol=0)
        for probability_dtype in (torch.float16,torch.float32,torch.bfloat16):
            backing=torch.tensor([0.,.9,.25,.9,1.,.9],dtype=probability_dtype).to('intel_npu')
            probability=backing[::2];seed,offset=state()
            target=torch.empty((2,3),dtype=dtype,device='intel_npu').bernoulli_(probability,generator=gen)
            expected=torch.tensor([(philox(seed,offset+i)[0]>>8)*2**-24 for i in range(6)]).view(2,3)<torch.tensor([0.,.25,1.])
            torch.testing.assert_close(target.cpu(),expected.to(dtype),rtol=0,atol=0)
        return dict(variants=records,nine_output_dtypes=True,zero_probability_gradient=True,
                    dense_transpose=True,empty_no_rng=True,alias_read_before_write=True,replay=True,
                    mixed_probability_dtypes_and_strided_reads=True,inplace_zero_gradients=True)
    def permutation():
        rows=[]
        for n,target_dtype in ((0,torch.int64),(1,torch.int64),(17,torch.int64),(129,torch.int32),
                               (128,torch.int8),(256,torch.uint8),(129,torch.int16),
                               (257,torch.bfloat16),(2049,torch.float16),(8193,torch.float32)):
            saved=gen.get_state();seed,offset=state();before=npu.stats()
            actual=torch.randperm(n,device='intel_npu',dtype=target_dtype,generator=gen)
            resident(before,1 if n>1 else 0)
            expected=torch.argsort(integer_reference(seed,offset,n,-(1<<63),1<<63),stable=True) if n>1 else torch.arange(n)
            torch.testing.assert_close(actual.cpu(),expected.to(target_dtype),rtol=0,atol=0)
            torch.testing.assert_close(actual.cpu().to(torch.int64).sort().values,torch.arange(n),rtol=0,atol=0)
            assert state()==(seed,offset+(n if n>1 else 0))
            gen.set_state(saved);out=torch.empty((n,),dtype=target_dtype,device='intel_npu')
            assert torch.randperm(n,generator=gen,out=out) is out
            torch.testing.assert_close(out.cpu(),actual.cpu(),rtol=0,atol=0)
            rows.append(dict(n=n,dtype=str(target_dtype),exact_signed_int64_key_oracle=True))
        saved=torch.intel_npu.get_rng_state();independent=gen.get_state()
        seed,offset=struct.unpack('<8sQQ',bytes(saved.tolist()))[1:]
        actual=torch.randperm(37,device='intel_npu')
        expected=integer_reference(seed,offset,37,-(1<<63),1<<63).argsort(stable=True)
        torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
        torch.intel_npu.set_rng_state(saved);out=torch.empty((37,),dtype=torch.int64,device='intel_npu')
        assert torch.randperm(37,out=out) is out
        torch.testing.assert_close(actual.cpu(),out.cpu(),rtol=0,atol=0)
        assert torch.equal(independent,gen.get_state())
        torch.intel_npu.set_rng_state(saved)
        return dict(cases=rows,exact_permutations=True,empty_and_singleton_no_rng=True,
                    explicit_generator_out_and_replay=True,default_generator_and_out=True)
    def contracts():
        prototype=torch.zeros((2,3),dtype=dtype,device='intel_npu')
        good=torch.full((1,3),.5,dtype=dtype,device='intel_npu')
        bad_shape=torch.ones((2,2),device='intel_npu')
        actions=[lambda:torch.randperm(-1,device='intel_npu',generator=gen),
                 lambda:torch.randperm(2050,device='intel_npu',dtype=torch.float16,generator=gen),
                 lambda:torch.randperm(258,device='intel_npu',dtype=torch.bfloat16,generator=gen),
                 lambda:torch.randperm(129,device='intel_npu',dtype=torch.int8,generator=gen),
                 lambda:torch.randperm(3,device='intel_npu',dtype=torch.bool,generator=gen),
                 lambda:torch.randperm(3,device='intel_npu',generator=torch.Generator()),
                 lambda:torch.randperm(4,generator=gen,out=torch.empty(3,device='intel_npu',dtype=torch.int64)),
                 lambda:prototype.bernoulli_(torch.ones(3),generator=gen),
                 lambda:prototype.bernoulli_(bad_shape,generator=gen),
                 lambda:prototype.bernoulli_(good,generator=torch.Generator()),
                 lambda:prototype.bernoulli_(1.1,generator=gen)]
        for fn in actions:
            saved=gen.get_state();before=npu.stats()
            try:fn()
            except (ValueError,RuntimeError,NotImplementedError):pass
            else:raise AssertionError('Missing metadata rejection')
            assert torch.equal(saved,gen.get_state())
            assert npu.stats()['npu_calls']==before['npu_calls']
        for invalid in (-.1,1.1,float('nan'),float('inf')):
            p=torch.tensor([.5,invalid,.5],dtype=dtype).to('intel_npu');out=prototype.clone()
            saved=gen.get_state()
            try:torch.ops.aten.bernoulli.Tensor_out(prototype,p,generator=gen,out=out)
            except ValueError:pass
            else:raise AssertionError('Missing probability validation')
            assert torch.equal(saved,gen.get_state())
            torch.testing.assert_close(out.cpu(),torch.zeros((2,3),dtype=dtype),rtol=0,atol=0)
        return dict(metadata_rejections=len(actions),bad_probabilities=4,rejection_before_rng_or_output=True)
    row=dict(name=case)
    try:
        row.update(dict(bernoulli=bernoulli,permutation=permutation,contracts=contracts)[case]())
        assert torch.equal(cpu_rng,torch.get_rng_state())
        assert torch.equal(default_rng,torch.intel_npu.get_rng_state())
        row.update(status='passed',cpu_and_default_rng_unchanged=True)
    except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
    print(case,precision,row['status'],row.get('error','')[:500],flush=True)
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes'))
    root=Path(__file__).resolve().parent
    sources=('torch_intel_npu.py','torch_npu_rng.py','torch_npu_rng_extended_demo.py','test_torch_npu_rng.py')
    result=dict(status='passed' if healthy and row['status']=='passed' else 'failed',case=case,precision=precision,
                checks=[row],stats=before,shutdown=final,elapsed_seconds=time.perf_counter()-started,
                source_sha256={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources})
    target=root/'npunlock-probe/research/results/attention-rng-boundaries';target.mkdir(exist_ok=True)
    (target/f'{case}-{precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('bernoulli','permutation','contracts'),required=True)
    parser.add_argument('--precision',choices=('float16','float32'),default='float32')
    args=parser.parse_args();raise SystemExit(execute(args.case,args.precision))

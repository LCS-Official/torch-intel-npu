"""Device INT64 RNG, categorical sampling and sampled-token model checks."""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/discrete-rng'


def integer_reference(seed,offset,count,low,high):
    import torch
    span=high-low;threshold=(1<<64)%span;result=[]
    for i in range(count):
        attempt=0
        while True:
            # Independent Python Philox oracle also supplies the rejection stream.
            c=[(offset+i)&0xffffffff,(offset+i)>>32,attempt&0xffffffff,attempt>>32]
            k=[seed&0xffffffff,seed>>32]
            for _ in range(10):
                a,b=c[0]*0xd2511f53,c[2]*0xcd9e8d57
                c=[((b>>32)^c[1]^k[0])&0xffffffff,b&0xffffffff,((a>>32)^c[3]^k[1])&0xffffffff,a&0xffffffff]
                k=[(k[0]+0x9e3779b9)&0xffffffff,(k[1]+0xbb67ae85)&0xffffffff]
            draw=(c[1]<<32)|c[0];attempt+=1
            if draw>=threshold:break
        result.append(low+draw%span)
    return torch.tensor(result,dtype=torch.int64)


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.manual_seed(271)
    dtype=getattr(torch,precision);checks=[]
    def state():return struct.unpack('<8sQQ',bytes(torch.intel_npu.get_rng_state().tolist()))[1:]
    def no_payload(before,downloads=0):
        after=npu.stats()
        assert after['host_download_copies']-before['host_download_copies']==downloads
        assert after['host_device_copies']==before['host_device_copies']
    def check(name,action):
        before=npu.stats();row=dict(name=name)
        try:row.update(action() or {});row['status']='passed'
        except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
        gc.collect();after=npu.stats();row['npu_calls']=after['npu_calls']-before['npu_calls']
        if after['live_buffers']-after['cached_scalar_buffers']!=before['live_buffers']-before['cached_scalar_buffers']:
            row.update(status='failed',lifecycle_error='Leaked tensor storage')
        checks.append(row);print(name,row['status'],row.get('error','')[:250],flush=True)
    def probe():
        observations=[]
        for name,action in (('randint',lambda:torch.randint(17,(2,3),device='intel_npu')),
                            ('multinomial',lambda:torch.multinomial(torch.tensor([1.,2.,3.]).to('intel_npu'),1))):
            try:action();observations.append(dict(name=name,status='passed'))
            except Exception as error:observations.append(dict(name=name,status='blocked',error=str(error).splitlines()[0][:220]))
        return dict(observations=observations)
    def integers():
        ranges=((0,17),(-23,37),(-(1<<63),-(1<<63)+97),((1<<63)-101,(1<<63)-1),
                (-(1<<63),(1<<63)-1),(-7,(1<<63)-1),(5,6))
        cpu_rng=torch.get_rng_state()
        for low,high in ranges:
            seed,offset=state();before=npu.stats()
            actual=torch.randint(low,high,(2,3,17),device='intel_npu');no_payload(before)
            assert actual.dtype==torch.int64 and state()==(seed,offset+actual.numel())
            assert torch.equal(actual.cpu().reshape(-1),integer_reference(seed,offset,actual.numel(),low,high))
        for count in (1,8193,65537):
            seed,offset=state();before=npu.stats();actual=torch.randint(19,(count,),device='intel_npu');no_payload(before)
            expected=integer_reference(seed,offset,count,0,19);assert torch.equal(actual.cpu(),expected)
        x=torch.empty((2,3),dtype=torch.int64,device='intel_npu')
        for low,high in ((0,1<<63),(-(1<<63),1<<63)):
            seed,offset=state();before=npu.stats()
            assert (x.random_() if low==0 else x.random_(low,None)) is x;no_payload(before)
            assert torch.equal(x.cpu().reshape(-1),integer_reference(seed,offset,6,low,high))
        out=torch.empty((2,3),dtype=torch.int64,device='intel_npu');seed,offset=state()
        assert torch.randint(-5,9,(2,3),out=out) is out
        assert torch.equal(out.cpu().reshape(-1),integer_reference(seed,offset,6,-5,9))
        like=torch.randint_like(out,7);assert like.shape==out.shape and like.dtype==torch.int64
        assert torch.equal(cpu_rng,torch.get_rng_state())
        return dict(full_signed_int64_ranges=True,exact_philox_rejection_reference=True,tiled_65537=True,
                    random_fill_and_factory_out=True,cpu_rng_unchanged=True)

    def sampling():
        cpu_rng=torch.get_rng_state()
        weights=torch.tensor([[1.,3.,6.],[0.,1.,0.]],dtype=dtype).to('intel_npu')
        for replacement,count in ((True,17),(False,1)):
            saved=torch.intel_npu.get_rng_state();seed,offset=state();before=npu.stats()
            sampled=torch.multinomial(weights,count,replacement);no_payload(before,1)
            assert sampled.dtype==torch.int64 and sampled.shape==(2,count) and state()==(seed,offset+2*count)
            host=sampled.cpu();assert torch.all(host[1]==1)
            torch.intel_npu.set_rng_state(saved)
            assert torch.equal(torch.multinomial(weights,count,replacement).cpu(),host)
        source=torch.tensor([1.,3.,6.],dtype=dtype).to('intel_npu');before=npu.stats()
        drawn=torch.multinomial(source,10000,True);no_payload(before,1)
        counts=torch.bincount(drawn.cpu(),minlength=3)
        probabilities=counts.double()/10000
        assert torch.all((probabilities-torch.tensor([.1,.3,.6])).abs()<.025),probabilities
        uniform=torch.ones(16,dtype=dtype,device='intel_npu');before=npu.stats()
        permutation=torch.multinomial(uniform,16,False);no_payload(before,1)
        assert torch.equal(permutation.cpu().sort().values,torch.arange(16))
        out=torch.empty((2,3),dtype=torch.int64,device='intel_npu')
        full=torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.]],dtype=dtype).to('intel_npu')
        assert torch.multinomial(full,3,False,out=out) is out
        assert all(row.unique().numel()==3 for row in out.cpu())
        # Re-normalizing remaining original weights must survive ratio underflow.
        tiny=torch.finfo(dtype).tiny*torch.finfo(dtype).eps
        for host in (torch.tensor([tiny,0.,0.],dtype=dtype),
                     torch.tensor([torch.finfo(dtype).max,tiny],dtype=dtype)):
            x=host.to('intel_npu');before=npu.stats();actual=torch.multinomial(x,int(torch.count_nonzero(host)),False);no_payload(before,1)
            assert set(actual.cpu().tolist())==set(torch.nonzero(host).view(-1).tolist())
        wide=torch.zeros(49152,dtype=dtype);wide[-1]=1.;x=wide.to('intel_npu');before=npu.stats()
        actual=torch.multinomial(x,1);no_payload(before,1);assert actual.item()==49151
        batch=torch.zeros(4,20000,dtype=dtype)
        for row,column in enumerate((7,19999,16384,123)):batch[row,column]=1.
        before=npu.stats();drawn=torch.multinomial(batch.to('intel_npu'),1);no_payload(before,2)
        assert drawn.cpu().view(-1).tolist()==[7,19999,16384,123]
        differentiable=torch.ones(7,dtype=dtype,device='intel_npu',requires_grad=True)
        drawn=torch.multinomial(differentiable,2)
        assert not drawn.requires_grad and differentiable.grad is None
        assert torch.equal(cpu_rng,torch.get_rng_state())
        return dict(replacement_counts=counts.tolist(),without_replacement_unique=True,
                    exact_state_replay=True,zero_and_extreme_weights=True,categories_49152=True,
                    row_tiling_and_requires_grad=True,cpu_rng_unchanged=True)

    def generators():
        gen=torch.Generator(device='intel_npu').manual_seed(-1)
        assert isinstance(gen,torch.Generator) and gen.device==torch.device('intel_npu:0')
        assert gen.initial_seed()==(1<<64)-1
        default=torch.intel_npu.get_rng_state();cpu_rng=torch.get_rng_state()
        weights=torch.tensor([1.,3.,6.],dtype=dtype).to('intel_npu')
        actions=(lambda:torch.rand(19,device='intel_npu',dtype=dtype,generator=gen),
                 lambda:torch.randn(23,device='intel_npu',dtype=dtype,generator=gen),
                 lambda:torch.empty(17,device='intel_npu',dtype=dtype).uniform_(-.5,.5,generator=gen),
                 lambda:torch.empty(17,device='intel_npu',dtype=dtype).normal_(.25,.5,generator=gen),
                 lambda:torch.empty(17,device='intel_npu',dtype=dtype).bernoulli_(.3,generator=gen),
                 lambda:torch.randint(-11,19,(13,),device='intel_npu',generator=gen),
                 lambda:torch.multinomial(weights,5,True,generator=gen))
        for action in actions:
            saved=gen.get_state();offset=gen.get_offset();actual=action()
            assert gen.get_offset()==offset+actual.numel()
            expected=actual.cpu();gen.set_state(saved)
            assert torch.equal(action().cpu(),expected)
        gen.set_offset((1<<32)-3);offset=gen.get_offset();seed=gen.initial_seed()
        actual=torch.randint(-7,17,(9,),device='intel_npu',generator=gen)
        assert torch.equal(actual.cpu(),integer_reference(seed,offset,9,-7,17))
        clone=gen.clone_state();assert torch.equal(clone.get_state(),gen.get_state())
        a=torch.rand(17,device='intel_npu',generator=gen);b=torch.rand(17,device='intel_npu',generator=clone)
        assert torch.equal(a.cpu(),b.cpu())
        saved=gen.get_state();clone.manual_seed(4);assert torch.equal(saved,gen.get_state())
        assert torch.equal(default,torch.intel_npu.get_rng_state()) and torch.equal(cpu_rng,torch.get_rng_state())
        assert torch.intel_npu.default_generators[0] is npu._rng
        torch.intel_npu.default_generators[0].manual_seed(277);assert torch.intel_npu.initial_seed()==277
        torch.intel_npu.set_rng_state(default)
        return dict(native_torch_generator=True,distributions=len(actions),clone_and_offset=True,
                    default_and_cpu_rng_isolated=True,legacy_24_byte_state=True)

    def contracts():
        x=torch.empty(4,dtype=torch.int64,device='intel_npu');other=torch.empty(4,device='intel_npu')
        p=torch.ones(2,3,dtype=dtype,device='intel_npu');wrong=torch.empty(2,dtype=torch.int64,device='intel_npu')
        cpu_gen=torch.Generator().manual_seed(7);cpu_state=cpu_gen.get_state()
        before=npu.stats();saved=torch.intel_npu.get_rng_state();rejected=0
        actions=(lambda:x.random_(5,5),lambda:x.random_(7,3),lambda:other.random_(),
                 lambda:x.random_(generator=cpu_gen),lambda:x.view(2,2).t().random_(),
                 lambda:torch.randint(7,(2,2),out=x),lambda:torch.multinomial(p,0),
                 lambda:torch.multinomial(p,4,False),lambda:torch.multinomial(p,1,out=wrong),
                 lambda:torch.multinomial(p,1,generator=cpu_gen),
                 lambda:torch.multinomial(x,1),lambda:torch.multinomial(p.view(1,2,3),1))
        for action in actions:
            try:action()
            except (ValueError,RuntimeError,TypeError):rejected+=1
            else:raise AssertionError('Invalid discrete RNG metadata accepted')
        assert npu.stats()['npu_calls']==before['npu_calls'] and torch.equal(saved,torch.intel_npu.get_rng_state())
        assert torch.equal(cpu_state,cpu_gen.get_state())
        invalids=([-1.,1.,2.],[float('nan'),1.,2.],[float('inf'),1.,2.],[0.,0.,0.])
        for values in invalids:
            source=torch.tensor(values,dtype=dtype).to('intel_npu');out=torch.full((1,),55,dtype=torch.int64,device='intel_npu')
            before=npu.stats();saved=torch.intel_npu.get_rng_state()
            try:torch.multinomial(source,1,out=out)
            except ValueError:pass
            else:raise AssertionError('Invalid probabilities accepted')
            no_payload(before,1);assert torch.equal(saved,torch.intel_npu.get_rng_state()) and out.item()==55
        gen=torch.Generator(device='intel_npu').manual_seed(11);gen.set_offset((1<<64)-1);saved=gen.get_state()
        try:torch.randint(7,(1,),device='intel_npu',generator=gen)
        except RuntimeError:pass
        else:raise AssertionError('Exhausted RNG accepted')
        assert torch.equal(saved,gen.get_state())
        for invalid in (torch.zeros(24,dtype=torch.uint8),torch.zeros(24),saved[:23],torch.zeros(48,dtype=torch.uint8)[::2]):
            try:gen.set_state(invalid)
            except (RuntimeError,ValueError):pass
            else:raise AssertionError('Invalid Generator state accepted')
            assert torch.equal(saved,gen.get_state())
        # A bad later packet must not consume RNG or partially overwrite caller output.
        batch=torch.ones(4,20000,dtype=dtype);batch[-1].zero_()
        source=batch.to('intel_npu');out=torch.full((4,1),55,dtype=torch.int64,device='intel_npu')
        before=npu.stats();saved=torch.intel_npu.get_rng_state()
        try:torch.multinomial(source,1,out=out)
        except ValueError:pass
        else:raise AssertionError('Invalid later probability packet accepted')
        no_payload(before,2);assert torch.equal(saved,torch.intel_npu.get_rng_state()) and torch.all(out.cpu()==55)
        sparse=torch.tensor([0.,1.,0.],dtype=dtype).to('intel_npu')
        try:torch.multinomial(sparse,2,False)
        except ValueError:pass
        else:raise AssertionError('Too few positive weights accepted')
        assert torch.equal(saved,torch.intel_npu.get_rng_state())
        alias=torch.ones(4 if dtype==torch.float16 else 2,dtype=dtype,device='intel_npu')
        before=npu.stats()
        try:torch.multinomial(alias,1,out=alias.view(torch.int64))
        except ValueError:pass
        else:raise AssertionError('Aliased probability/output storage accepted')
        assert npu.stats()['npu_calls']==before['npu_calls'] and torch.equal(saved,torch.intel_npu.get_rng_state())
        return dict(metadata_rejections=rejected,invalid_weight_cases=4,no_rng_on_rejection=True,
                    invalid_out_unchanged=True,exhaustion_and_invalid_state=True,late_packet_and_alias_rejection=True)

    def training():
        from torch_npu_transformer_demo import cpu_tree
        class TokenNet(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embedding=torch.nn.Embedding(17,4);self.head=torch.nn.Linear(4,17)
            def forward(self,tokens):return self.head(self.embedding(tokens).mean(1))
        cpu=TokenNet().to(dtype);model=copy.deepcopy(cpu).to('intel_npu')
        reference,opt=[torch.optim.SGD(net.parameters(),lr=.02,momentum=.5,foreach=False,fused=False) for net in (cpu,model)]
        gen=torch.Generator(device='intel_npu').manual_seed(281);default=torch.intel_npu.get_rng_state();checkpoint=io.BytesIO();losses=[]
        def cycle(net,optimizer):
            optimizer.zero_grad(set_to_none=True)
            tokens=torch.randint(17,(2,3),device='intel_npu',generator=gen)
            labels=torch.randint(17,(2,),device='intel_npu',generator=gen)
            before=npu.stats();cpu_rng=torch.get_rng_state();logits=net(tokens)
            loss=torch.nn.functional.cross_entropy(logits.float(),labels);loss.backward();no_payload(before,3)
            assert torch.equal(cpu_rng,torch.get_rng_state())
            optimizer.step()
            before=npu.stats()
            with torch.no_grad():
                sampled=torch.multinomial(torch.softmax(logits.detach().float(),-1),1,generator=gen)
                next_logits=net(sampled)
            no_payload(before,2)
            return loss,tokens,labels,sampled,next_logits
        def close(a,b):torch.testing.assert_close(a.detach().cpu(),b.detach(),rtol=.003 if dtype==torch.float16 else 3e-5,atol=.0005 if dtype==torch.float16 else 3e-6)
        for step in range(2):
            actual,tokens,labels,sampled,next_logits=cycle(model,opt)
            reference.zero_grad(set_to_none=True);host_tokens=tokens.cpu();host_labels=labels.cpu()
            expected=torch.nn.functional.cross_entropy(cpu(host_tokens).float(),host_labels);expected.backward();close(actual,expected)
            for a,b in zip(model.parameters(),cpu.parameters()):close(a.grad,b.grad)
            reference.step()
            for a,b in zip(model.parameters(),cpu.parameters()):close(a,b)
            with torch.no_grad():close(next_logits,cpu(sampled.cpu()))
            losses.append(actual.item())
            state=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),generator=gen.get_state(),sampled=sampled.cpu())
            if step==0:torch.save(state,checkpoint)
        checkpoint.seek(0);saved=torch.load(checkpoint,weights_only=True);restored=TokenNet().to(dtype).to('intel_npu')
        restored.load_state_dict(saved['model']);replay=torch.optim.SGD(restored.parameters(),lr=.02,momentum=.5,foreach=False,fused=False)
        replay.load_state_dict(saved['optimizer']);gen.set_state(saved['generator'])
        loss,_,_,sampled,_=cycle(restored,replay)
        assert loss.item()==losses[-1] and torch.equal(sampled.cpu(),state['sampled']) and torch.equal(gen.get_state(),state['generator'])
        for name,value in cpu_tree(restored.state_dict()).items():torch.testing.assert_close(value,state['model'][name],rtol=0,atol=0)
        for key,value in cpu_tree(replay.state_dict())['state'].items():
            torch.testing.assert_close(value['momentum_buffer'],state['optimizer']['state'][key]['momentum_buffer'],rtol=0,atol=0)
        assert torch.equal(default,torch.intel_npu.get_rng_state())
        return dict(steps=2,losses=losses,npu_tokens_labels_sampling_and_embedding=True,all_gradients=True,
                    checkpoint_and_sampling_bitwise=True,default_rng_unchanged=True)

    check(case,dict(probe=probe,integers=integers,sampling=sampling,generators=generators,contracts=contracts,training=training)[case])
    gc.collect();before=npu.stats();npu.shutdown();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes'))
    sources=('torch_intel_npu.py','torch_npu_rng.py','torch_npu_discrete_demo.py',
             'npunlock/python/npunlock/npu4000.py','npunlock/tests/fixtures/npu4000/resident/manifest.json',
             'npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/_torch_intel_npu.pyd')
    return dict(status='passed' if healthy and all(r['status']=='passed' for r in checks) else 'failed',checks=checks,
                before_shutdown=before,final_stats=final,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=('all','probe','integers','sampling','generators','contracts','training'),default='all')
    parser.add_argument('--precision',choices=('all','float16','float32'),default='all');parser.add_argument('--label',default='after');args=parser.parse_args()
    if args.case=='all' or args.precision=='all':
        cases=('integers','sampling','generators','contracts','training') if args.case=='all' else (args.case,)
        for case in cases:
            precisions=('float32',) if case in ('integers','probe') else (('float32','float16') if args.precision=='all' else (args.precision,))
            for precision in precisions:
                subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--case',case,'--precision',precision,'--label',args.label],check=True)
        raise SystemExit(0)
    result=execute(args.case,args.precision);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{args.label}-{args.case}-{args.precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(result['status']!='passed')

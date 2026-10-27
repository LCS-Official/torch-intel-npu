"""On-device RNG, replayable state, Dropout gradients and checkpoint training."""
import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/rng-dropout'
CASES=('smoke','uniform','normal','state','dropout','contracts','training')


def philox(seed,counter):
    c=[counter&0xffffffff,counter>>32,0,0]
    k=[seed&0xffffffff,seed>>32]
    for _ in range(10):
        a,b=c[0]*0xd2511f53,c[2]*0xcd9e8d57
        c=[((b>>32)^c[1]^k[0])&0xffffffff,b&0xffffffff,((a>>32)^c[3]^k[1])&0xffffffff,a&0xffffffff]
        k=[(k[0]+0x9e3779b9)&0xffffffff,(k[1]+0xbb67ae85)&0xffffffff]
    return c


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(149)
    dtype={'float16':torch.float16,'float32':torch.float32}[precision]
    checks=[]

    def check(name,fn):
        gc.collect();before=npu.stats();row=dict(name=name,status='passed')
        try:row['detail']=fn()
        except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
        gc.collect();after=npu.stats()
        row['npu_calls']=after['npu_calls']-before['npu_calls']
        row['live_buffer_delta']=after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
        if row['live_buffer_delta'] or after['context_failed']:row.update(status='failed',lifecycle_error='Buffers or context')
        checks.append(row);print(name,row['status'],row.get('error','')[:500],flush=True)

    def no_download(before):
        assert npu.stats()['host_download_copies']==before['host_download_copies']

    def smoke():
        assert philox(0,0)==[0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8]
        torch.manual_seed(0)
        before=npu.stats();x=torch.rand(8,device='intel_npu');no_download(before)
        expected=torch.tensor([(philox(0,i)[0]>>8)*2**-24 for i in range(8)])
        torch.testing.assert_close(x.cpu(),expected,rtol=0,atol=0)
        return dict(known_philox_vector=True,exact_uniform=True,no_payload_download=True)

    def uniform():
        seed=(1<<63)+117
        torch.intel_npu.manual_seed(seed)
        cpu_state=torch.get_rng_state().clone()
        before=npu.stats();x=torch.rand(8199,device='intel_npu',dtype=dtype);no_download(before)
        shift=21 if dtype==torch.float16 else 8
        factor=2**-11 if dtype==torch.float16 else 2**-24
        expected=torch.tensor([(philox(seed,i)[0]>>shift)*factor for i in range(8199)],dtype=dtype)
        torch.testing.assert_close(x.cpu(),expected,rtol=0,atol=0)
        assert torch.equal(cpu_state,torch.get_rng_state())
        before=npu.stats()
        y=torch.rand_like(x);z=torch.empty(7,11,device='intel_npu',dtype=dtype).uniform_(-.3,.7)
        t=torch.empty(2,3,device='intel_npu',dtype=dtype).t()
        t_state=torch.intel_npu.get_rng_state();t.uniform_()
        b=torch.empty(8199,device='intel_npu',dtype=dtype).bernoulli_(.25)
        no_download(before)
        assert ((z.cpu()>=torch.tensor(-.3,dtype=dtype)) & (z.cpu()<torch.tensor(.7,dtype=dtype))).all()
        assert ((b.cpu()==0)|(b.cpu()==1)).all() and abs(b.cpu().float().mean().item()-.25)<.025
        assert not torch.equal(x.cpu(),y.cpu()) and t.stride()==(1,3)
        _,t_seed,t_counter=struct.unpack('<8sQQ',bytes(t_state.tolist()))
        expected_t=torch.tensor([(philox(t_seed,t_counter+i)[0]>>shift)*factor for i in range(6)],dtype=dtype).view(3,2)
        torch.testing.assert_close(t.cpu(),expected_t,rtol=0,atol=0)
        scalar=torch.rand((),device='intel_npu',dtype=dtype,requires_grad=True)
        rank4=torch.randn(1,2,1,3,device='intel_npu',dtype=dtype,requires_grad=True)
        assert scalar.is_leaf and rank4.is_leaf
        (scalar+rank4).sum().backward()
        torch.testing.assert_close(scalar.grad.cpu(),torch.tensor(6,dtype=dtype),rtol=0,atol=0)
        torch.testing.assert_close(rank4.grad.cpu(),torch.ones(1,2,1,3,dtype=dtype),rtol=0,atol=0)
        linear=torch.nn.Linear(4,3,device='intel_npu',dtype=dtype)
        assert all(p.device.type=='intel_npu' for p in linear.parameters())
        assert torch.isfinite(linear.weight.detach().cpu()).all()
        return dict(elements=8199,exact_philox=True,chunk_boundary=True,cpu_rng_unchanged=True,
                    rand_like=True,dense_transposed_fill=True,scalar_bernoulli=True,
                    scalar_and_rank4_factory_gradients=True,direct_linear_initialization=True)

    def normal():
        seed=418
        torch.intel_npu.manual_seed(seed)
        before=npu.stats();x=torch.randn(8199,device='intel_npu',dtype=dtype);no_download(before)
        def oracle(i):
            a,b,*_=philox(seed,i)
            return math.sqrt(-2*math.log(((a>>9)+.5)*2**-23))*math.cos((b>>8)*2**-24*2*math.pi)
        expected=torch.tensor([oracle(i) for i in range(8199)],dtype=dtype)
        torch.testing.assert_close(x.cpu(),expected,rtol=.002 if dtype==torch.float16 else 3e-5,
                                   atol=.002 if dtype==torch.float16 else 3e-6)
        host=x.cpu().float()
        assert abs(host.mean().item())<.04 and abs(host.std().item()-1)<.04
        before=npu.stats()
        y=torch.randn_like(x);z=torch.empty(16384,device='intel_npu',dtype=dtype).normal_(2.,.5)
        no_download(before)
        assert not torch.equal(x.cpu(),y.cpu())
        mean,std=z.cpu().float().mean().item(),z.cpu().float().std().item()
        assert abs(mean-2)<.025 and abs(std-.5)<.025
        return dict(elements=8199,box_muller_reference=True,mean=host.mean().item(),std=host.std().item(),
                    shifted_mean=mean,shifted_std=std,no_payload_download=True)

    def state():
        torch.manual_seed(-1)
        assert torch.intel_npu.initial_seed()==(1<<64)-1
        before=npu.stats();a=torch.rand(17,device='intel_npu');no_download(before)
        saved=torch.intel_npu.get_rng_state()
        expected=torch.randn(19,device='intel_npu')
        torch.intel_npu.set_rng_state(saved)
        actual=torch.randn(19,device='intel_npu')
        torch.testing.assert_close(actual.cpu(),expected.cpu(),rtol=0,atol=0)
        outer=torch.intel_npu.get_rng_state()
        with torch.random.fork_rng(devices=[0],device_type='intel_npu'):
            torch.manual_seed(2);temp=torch.rand(11,device='intel_npu')
        assert torch.equal(outer,torch.intel_npu.get_rng_state())
        all_states=torch.intel_npu.get_rng_state_all()
        torch.intel_npu.seed();torch.intel_npu.seed_all()
        torch.intel_npu.set_rng_state_all(all_states)
        assert torch.equal(outer,torch.intel_npu.get_rng_state())
        seed=(1<<64)-1;counter=(1<<32)-3
        carry=torch.tensor(list(struct.pack('<8sQQ',b'NPURNG01',seed,counter)),dtype=torch.uint8)
        torch.intel_npu.set_rng_state(carry)
        value=torch.rand(9,device='intel_npu')
        expected=torch.tensor([(philox(seed,counter+i)[0]>>8)*2**-24 for i in range(9)])
        torch.testing.assert_close(value.cpu(),expected,rtol=0,atol=0)
        torch.manual_seed(53);first=torch.rand(8199,device='intel_npu')
        torch.manual_seed(53);pieces=[torch.rand(n,device='intel_npu') for n in (3,8189,7)]
        torch.testing.assert_close(first.cpu(),torch.cat([v.cpu() for v in pieces]),rtol=0,atol=0)
        return dict(roundtrip=True,fork_rng=True,unsigned_seed=True,counter_32bit_carry=True,call_and_tile_partition_invariant=True)

    def dropout():
        for p,train in ((.25,True),(0.,True),(1.,True),(.4,False)):
            host=torch.linspace(-2,2,97,dtype=dtype).view(1,97)
            x=host.to('intel_npu').requires_grad_()
            g=torch.linspace(-1,1,97,dtype=dtype).view(1,97)
            gradient=g.to('intel_npu');saved=torch.intel_npu.get_rng_state()
            before=npu.stats();y,mask=torch.ops.aten.native_dropout(x,p,train);y.backward(gradient);no_download(before)
            assert mask.dtype==torch.bool and not mask.requires_grad
            m=mask.cpu();scale=1. if not train else 0. if p==1 else 1./(1.-p)
            expected=(host.float()*m.float()*scale).to(dtype)
            expected_grad=(g.float()*m.float()*scale).to(dtype)
            torch.testing.assert_close(y.detach().cpu(),expected,rtol=0,atol=0)
            torch.testing.assert_close(x.grad.cpu(),expected_grad,rtol=0,atol=0)
            if not train or p in (0,1):assert torch.equal(saved,torch.intel_npu.get_rng_state())
            else:assert m.any() and (~m).any()
        module=torch.nn.Dropout(.3)
        x=torch.ones(3,5,device='intel_npu',dtype=dtype,requires_grad=True)
        y=module(x);y.sum().backward()
        torch.testing.assert_close(x.grad.cpu(),y.detach().cpu(),rtol=0,atol=0)
        module.eval();saved=torch.intel_npu.get_rng_state();assert module(x) is x
        assert torch.equal(saved,torch.intel_npu.get_rng_state())
        return dict(native_cases=4,exact_forward_backward=True,ordinary_nn_dropout=True,eval_no_rng_consumption=True)

    def contracts():
        x=torch.ones(2,3,device='intel_npu');rng=torch.intel_npu.get_rng_state()
        before=npu.stats()
        actions=(lambda:x.uniform_(2,1),lambda:x.normal_(0,-1),lambda:x.normal_(float('nan'),1),
                 lambda:x.uniform_(generator=torch.Generator()),lambda:x.bernoulli_(1.1),
                 lambda:torch.nn.functional.dropout(x,p=-.1),lambda:torch.intel_npu.get_rng_state(1),
                 lambda:torch.intel_npu.set_rng_state(torch.ones(24)),lambda:torch.intel_npu.manual_seed(1<<64))
        for fn in actions:
            try:fn()
            except (ValueError,RuntimeError,NotImplementedError):pass
            else:raise AssertionError('Missing RNG rejection')
        assert torch.equal(rng,torch.intel_npu.get_rng_state())
        assert npu.stats()['npu_calls']==before['npu_calls']
        host=torch.ones(2,3,requires_grad=True);leaf=host.detach().to('intel_npu').requires_grad_()
        for fn in (lambda:leaf.uniform_(),lambda:leaf.normal_(),lambda:leaf.bernoulli_()):
            try:fn()
            except (ValueError,RuntimeError):pass
            else:raise AssertionError('Missing in-place leaf guard')
        y,mask=torch.ops.aten.native_dropout(leaf,.2,True)
        with torch.no_grad():mask.fill_(False)
        before=npu.stats()
        try:y.sum().backward()
        except RuntimeError as error:assert 'version' in str(error) or 'inplace' in str(error)
        else:raise AssertionError('Missing saved mask version check')
        return dict(metadata_rejections=len(actions),state_and_submission_unchanged=True,leaf_and_saved_mask_guards=True)

    def training():
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.fc=torch.nn.Linear(4,8);self.drop=torch.nn.Dropout(.25);self.head=torch.nn.Linear(8,2)
            def forward(self,x):return self.head(self.drop(torch.nn.functional.relu(self.fc(x))))
        cpu=Model();model=copy.deepcopy(cpu).to('intel_npu')
        host=torch.randn(3,4)/4;target=torch.randn(3,2)/4
        x,labels=host.to('intel_npu'),target.to('intel_npu')
        opt=torch.optim.SGD(model.parameters(),lr=.03,foreach=False,fused=False)
        cpu_opt=torch.optim.SGD(cpu.parameters(),lr=.03,foreach=False,fused=False)
        losses=[];checkpoint=None;next_parameters=None;next_state=None
        cpu_drop=cpu.drop.forward
        def cpu_copy():return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        for step in range(4):
            saved=torch.intel_npu.get_rng_state()
            _,mask=torch.ops.aten.native_dropout(torch.ones(3,8,device='intel_npu'),.25,True)
            m=mask.cpu();torch.intel_npu.set_rng_state(saved)
            cpu.drop.forward=lambda v:(v*m.to(v.dtype))*(1/.75)
            opt.zero_grad(set_to_none=True);cpu_opt.zero_grad(set_to_none=True)
            before=npu.stats();loss=torch.nn.functional.mse_loss(model(x),labels);loss.backward();opt.step();no_download(before)
            expected=torch.nn.functional.mse_loss(cpu(host),target);expected.backward();cpu_opt.step()
            torch.testing.assert_close(loss.detach().cpu(),expected.detach(),rtol=3e-5,atol=3e-6)
            for p,q in zip(model.parameters(),cpu.parameters()):
                torch.testing.assert_close(p.grad.cpu(),q.grad,rtol=3e-5,atol=3e-6)
                torch.testing.assert_close(p.detach().cpu(),q.detach(),rtol=3e-5,atol=3e-6)
            losses.append(loss.item())
            if step==1:checkpoint=(cpu_copy(),copy.deepcopy(opt.state_dict()),torch.intel_npu.get_rng_state())
            if step==2:next_parameters=cpu_copy();next_state=torch.intel_npu.get_rng_state();third_loss=loss.item()
        cpu.drop.forward=cpu_drop
        model.load_state_dict({k:v.to('intel_npu') for k,v in checkpoint[0].items()})
        opt.load_state_dict(checkpoint[1]);torch.intel_npu.set_rng_state(checkpoint[2]);opt.zero_grad(set_to_none=True)
        before=npu.stats();loss=torch.nn.functional.mse_loss(model(x),labels);loss.backward();opt.step();no_download(before)
        assert loss.item()==third_loss and torch.equal(next_state,torch.intel_npu.get_rng_state())
        for k,v in cpu_copy().items():torch.testing.assert_close(v,next_parameters[k],rtol=0,atol=0)
        assert losses[-1]<losses[0]
        return dict(steps=4,losses=losses,cpu_same_mask_gradient_and_update_reference=True,
                    checkpoint_next_step_bitwise_equal=True,no_training_payload_download=True)

    check(case,locals()[case])
    gc.collect();npu.shutdown();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    sources=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_rng.py','torch_npu_rng_demo.py',
             'npunlock/python/npunlock/npu4000.py','npunlock/tests/fixtures/npu4000/resident_rng.c',
             'npunlock/tests/fixtures/npu4000/resident/rng-packet.elf')
    return dict(status='passed' if healthy and all(c['status']=='passed' for c in checks) else 'failed',
                checks=checks,final_stats=final,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('all',)+CASES,default='all')
    parser.add_argument('--precision',choices=('all','float16','float32'),default='all')
    args=parser.parse_args()
    if args.case=='all' or args.precision=='all':
        for case in CASES if args.case=='all' else (args.case,):
            for precision in ('float16','float32') if case in ('uniform','normal','dropout') and args.precision=='all' else (args.precision if args.precision!='all' else 'float32',):
                if subprocess.run([sys.executable,'-B',__file__,'--case',case,'--precision',precision]).returncode:return 1
        return 0
    result=execute(args.case,args.precision);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{args.case}-{args.precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

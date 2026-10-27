"""SDPA Dropout: independent same-mask references, gradients, RNG and training replay."""
import argparse
import copy
import gc
import hashlib
import io
import json
from math import prod
from pathlib import Path
import struct
import subprocess
import sys
import traceback
from torch_npu_rng_demo import philox

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/sdpa-dropout'
CASES=('smoke','operators','state','tiled','contracts','training')


def keep_mask(state,shape,p):
    import torch
    if p in (0,1):return torch.full(shape,p==0,dtype=torch.bool)
    _,seed,offset=struct.unpack('<8sQQ',bytes(state.tolist()))
    return torch.tensor([(philox(seed,offset+i)[0]>>8)*2**-24>=p for i in range(prod(shape))],dtype=torch.bool).view(shape)


def reference(q,k,v,mask=None,*,p=0.,causal=False,scale=None,keep=None):
    """CPU math reference; no NPU implementation or RNG is called."""
    import torch
    factor=(q.shape[-1] or 1)**-.5 if scale is None else scale
    root=abs(factor)**.5
    scores=(q.float()*(-root if factor<0 else root))@(k.float()*root).transpose(-2,-1)
    if causal:
        mask=torch.ones(q.shape[-2],k.shape[-2],dtype=torch.bool).tril()
    if mask is not None:
        scores=scores.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else scores+mask.float()
    blocked=torch.isneginf(scores).all(-1,keepdim=True)
    probabilities=scores.masked_fill(blocked,0.).softmax(-1).masked_fill(blocked,0.)
    if p:
        if keep is None:raise ValueError('Reference Dropout requires an independently generated mask')
        probabilities=probabilities*keep*(0. if p==1 else 1./(1.-p))
    return (probabilities@v.float()).to(q.dtype)


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.manual_seed(187)
    dtype={'float16':torch.float16,'float32':torch.float32}[precision]
    F=torch.nn.functional.scaled_dot_product_attention
    checks=[]

    def upload(x):
        return x.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)

    def close(actual,expected,exact=False):
        torch.testing.assert_close(actual.detach().cpu(),expected.detach(),rtol=0 if exact else .003 if actual.dtype==torch.float16 else 3e-5,
                                   atol=0 if exact else .0005 if actual.dtype==torch.float16 else 3e-6,equal_nan=True)

    def no_download(before):
        assert npu.stats()['host_download_copies']==before['host_download_copies']

    def check(name,fn):
        gc.collect();before=npu.stats();row=dict(name=name,status='passed')
        try:row.update(fn())
        except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
        gc.collect();after=npu.stats()
        row['npu_calls']=after['npu_calls']-before['npu_calls']
        row['live_buffer_delta']=after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
        if row['live_buffer_delta'] or after['context_failed']:row.update(status='failed',lifecycle_error='Buffer or context')
        checks.append(row);print(name,row['status'],row.get('error','')[:500],flush=True)

    def one(name,shapes,*,p=.25,mask_kind=None,causal=False,scale=None,needs=(True,True,True),dense=False):
        hosts=[(torch.randn(shape,dtype=dtype)/4).requires_grad_(need) for shape,need in zip(shapes,needs)]
        tensors=[upload(v) for v in hosts]
        if dense:
            tensors=[x.detach().transpose(-2,-1).contiguous().transpose(-2,-1).requires_grad_(need) for x,need in zip(tensors,needs)]
        shape=tuple(torch.broadcast_shapes(*(v.shape[:-2] for v in hosts)))+(hosts[0].shape[-2],hosts[1].shape[-2])
        mask=None
        if mask_kind:
            mask=torch.ones(shape[-2:],dtype=torch.bool)
            mask[1]=False;mask[:,-1]=False
            if mask_kind!='bool':
                mask=torch.where(mask,torch.tensor(.125),torch.tensor(float('-inf')))
                mask=mask.to(torch.float32 if mask_kind=='mixed' else dtype).requires_grad_()
        device_mask=upload(mask) if mask is not None else None
        saved=torch.intel_npu.get_rng_state();keep=keep_mask(saved,shape,p)
        expected=reference(*hosts,mask,p=p,causal=causal,scale=scale,keep=keep)
        seed=torch.randn_like(expected)/8;gradient=upload(seed)
        cpu_state=torch.get_rng_state();before=npu.stats()
        actual=F(*tensors,attn_mask=device_mask,dropout_p=p,is_causal=causal,scale=scale)
        after_forward=torch.intel_npu.get_rng_state();actual.backward(gradient);no_download(before)
        assert torch.equal(after_forward,torch.intel_npu.get_rng_state()) and torch.equal(cpu_state,torch.get_rng_state())
        if p in (0,1):assert torch.equal(saved,after_forward)
        else:
            _,_,offset=struct.unpack('<8sQQ',bytes(saved.tolist()))
            assert struct.unpack('<8sQQ',bytes(after_forward.tolist()))[2]==offset+prod(shape)
        expected.backward(seed);close(actual,expected)
        for tensor,host in zip(tensors,hosts):
            if host.requires_grad:close(tensor.grad,host.grad)
        if mask is not None and mask.requires_grad:close(device_mask.grad,mask.grad)
        if mask is not None:
            assert torch.count_nonzero(actual.detach().cpu()[...,1,:])==0
            if needs[0]:assert torch.count_nonzero(tensors[0].grad.cpu()[...,1,:])==0
        if p==1:
            assert torch.count_nonzero(actual.detach().cpu())==0
            assert all(torch.count_nonzero(x.grad.cpu())==0 for x in tensors if x.requires_grad)
        return name

    def smoke():
        return dict(cases=[one('plain',((1,2,3),(1,3,3),(1,3,2)))],same_mask_cpu_reference=True)

    def operators():
        rank3=((2,3,4),(2,5,4),(2,5,2))
        broadcast=((2,1,3,4),(1,2,5,4),(2,1,5,2))
        cases=[one('rank3',rank3),one('broadcast-negative-scale',broadcast,p=.5,scale=-.7,dense=True),
               one('bool-fully-masked',broadcast,mask_kind='bool'),one('float-mask-gradient',broadcast,mask_kind='float'),
               one('fp32-mask-gradient',broadcast,mask_kind='mixed',scale=.7),one('causal-rectangle',rank3,causal=True),
               one('p0',rank3,p=0),one('p1',rank3,p=1)]
        for index in range(3):
            cases.append(one('only-'+('q','k','v')[index],rank3,needs=tuple(i==index for i in range(3))))
        cases.append(one('only-mask',broadcast,mask_kind='mixed',needs=(False,False,False)))
        return dict(cases=cases,qkv_and_float_mask_gradients=True,zero_fully_masked_rows=True,no_payload_download=True)

    def state():
        inputs=[upload(torch.randn(1,3,4).requires_grad_()) for _ in range(3)]
        saved=torch.intel_npu.get_rng_state();gradient=upload(torch.ones(1,3,4)/8)
        before=npu.stats();y=F(*inputs,dropout_p=.25);y.backward(gradient);no_download(before)
        outputs=[y.detach().cpu()]+[v.grad.cpu() for v in inputs];advanced=torch.intel_npu.get_rng_state()
        for v in inputs:v.grad=None
        torch.intel_npu.set_rng_state(saved)
        before=npu.stats();again=F(*inputs,dropout_p=.25);again.backward(gradient);no_download(before)
        for actual,expected in zip([again]+[v.grad for v in inputs],outputs):close(actual,expected,True)
        assert torch.equal(advanced,torch.intel_npu.get_rng_state())
        with torch.no_grad():
            before_state=torch.intel_npu.get_rng_state();other=F(*inputs,dropout_p=.25)
            assert not torch.equal(before_state,torch.intel_npu.get_rng_state())
            for p in (0,1):
                before_state=torch.intel_npu.get_rng_state();value=F(*inputs,dropout_p=p)
                assert torch.equal(before_state,torch.intel_npu.get_rng_state())
        return dict(bitwise_output_and_gradient_replay=True,backward_no_rng_consumption=True,
                    no_grad_still_applies_dropout=True,p0_p1_no_rng_consumption=True)

    def tiled():
        return dict(cases=[one('8193-scores',((1,1,1),(1,8193,1),(1,8193,1)))],score_tile_boundary=True)

    def contracts():
        inputs=[upload(torch.randn(1,2,3).requires_grad_()) for _ in range(3)]
        saved=torch.intel_npu.get_rng_state();before=npu.stats()
        for p in (-.1,1.01,float('nan'),float('inf')):
            try:F(*inputs,dropout_p=p)
            except (ValueError,RuntimeError):pass
            else:raise AssertionError('Accepted invalid SDPA Dropout probability')
        assert torch.equal(saved,torch.intel_npu.get_rng_state()) and npu.stats()['npu_calls']==before['npu_calls']
        captured=[];original=npu._native_dropout
        def record(*args,**kwargs):
            result=original(*args,**kwargs);captured.append(result[1]);return result
        try:
            npu._native_dropout=record
            output=F(*inputs,dropout_p=.25)
        finally:npu._native_dropout=original
        assert len(captured)==1
        gradient=upload(torch.ones(tuple(output.shape),dtype=output.dtype))
        with torch.no_grad():captured[0].fill_(False)
        before=npu.stats();saved=torch.intel_npu.get_rng_state()
        try:output.backward(gradient)
        except RuntimeError as error:assert 'version' in str(error) or 'inplace' in str(error)
        else:raise AssertionError('Missing saved Dropout mask version protection')
        assert npu.stats()['npu_calls']==before['npu_calls'] and torch.equal(saved,torch.intel_npu.get_rng_state())
        return dict(invalid_probability_rejections=4,no_submission_or_rng_change=True,saved_dropout_mask_version_guard=True)

    def training():
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q,self.k,self.v,self.proj=[torch.nn.Linear(4,4) for _ in range(4)]
                self.head=torch.nn.Linear(4,3)
                self.register_buffer('causal',torch.ones(3,3,dtype=torch.bool).tril())
                self.reference_mask=None
            def forward(self,x):
                q,k,v=[layer(x).view(2,3,2,2).transpose(1,2) for layer in (self.q,self.k,self.v)]
                p=.25 if self.training else 0.
                h=(reference(q,k,v,self.causal,p=p,keep=self.reference_mask) if self.reference_mask is not None else
                   F(q,k,v,attn_mask=self.causal,dropout_p=p))
                h=h.transpose(1,2).contiguous().view(2,3,4)
                return self.head((x+self.proj(h)).mean(1))
        cpu=Model();model=copy.deepcopy(cpu).to('intel_npu')
        host=torch.randn(2,3,4)/4;target=torch.randn(2,3)/4
        x,labels=upload(host),upload(target)
        opts=[torch.optim.SGD(net.parameters(),lr=.03,foreach=False,fused=False) for net in (cpu,model)]
        losses=[];checkpoint=io.BytesIO()
        def cpu_copy():return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        for step in range(4):
            cpu.reference_mask=keep_mask(torch.intel_npu.get_rng_state(),(2,2,3,3),.25)
            for opt in opts:opt.zero_grad(set_to_none=True)
            before=npu.stats();loss=torch.nn.functional.mse_loss(model(x),labels);loss.backward();opts[1].step();no_download(before)
            expected=torch.nn.functional.mse_loss(cpu(host),target);expected.backward();opts[0].step()
            close(loss,expected)
            for a,b in zip(model.parameters(),cpu.parameters()):close(a.grad,b.grad);close(a,b)
            losses.append(loss.item())
            if step==1:torch.save(dict(model=cpu_copy(),optimizer=opts[1].state_dict(),rng=torch.intel_npu.get_rng_state()),checkpoint)
            if step==2:third=cpu_copy();third_loss=loss.item();third_rng=torch.intel_npu.get_rng_state()
        checkpoint.seek(0);saved=torch.load(checkpoint,weights_only=True)
        model.load_state_dict({k:v.to('intel_npu') for k,v in saved['model'].items()})
        opts[1].load_state_dict(saved['optimizer']);torch.intel_npu.set_rng_state(saved['rng']);opts[1].zero_grad(set_to_none=True)
        before=npu.stats();loss=torch.nn.functional.mse_loss(model(x),labels);loss.backward();opts[1].step();no_download(before)
        assert loss.item()==third_loss and torch.equal(third_rng,torch.intel_npu.get_rng_state())
        for name,value in cpu_copy().items():torch.testing.assert_close(value,third[name],rtol=0,atol=0)
        model.eval();cpu.load_state_dict(third);cpu.eval();cpu.reference_mask=None
        saved_rng=torch.intel_npu.get_rng_state()
        with torch.no_grad():close(model(x),cpu(host))
        assert torch.equal(saved_rng,torch.intel_npu.get_rng_state())
        return dict(steps=4,losses=losses,cpu_independent_philox_mask_reference=True,all_parameter_gradients_and_updates=True,
                    serialized_cpu_checkpoint_next_step_bitwise_equal=True,eval_switch_disables_dropout=True,no_training_payload_download=True)

    check(case+'-'+precision,locals()[case])
    gc.collect();npu.shutdown();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    sources=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_sdpa_dropout_demo.py','torch_npu_rng_demo.py',
             'torch_npu_rng.py','npunlock/tests/fixtures/npu4000/resident/rng-packet.elf')
    return dict(status='passed' if healthy and all(c['status']=='passed' for c in checks) else 'failed',checks=checks,final_stats=final,
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('all',)+CASES,default='all')
    parser.add_argument('--precision',choices=('all','float16','float32'),default='all')
    args=parser.parse_args()
    if args.case=='all' or args.precision=='all':
        for case in CASES if args.case=='all' else (args.case,):
            for precision in ('float16','float32') if case=='operators' and args.precision=='all' else (args.precision if args.precision!='all' else 'float32',):
                if subprocess.run([sys.executable,'-B',__file__,'--case',case,'--precision',precision]).returncode:return 1
        return 0
    result=execute(args.case,args.precision);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{args.case}-{args.precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

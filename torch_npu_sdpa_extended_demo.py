"""Rank/broadcast/GQA and larger SDPA, with independent CPU gradients/training."""
import argparse
import copy
import gc
import hashlib
import json
from math import prod
from pathlib import Path
import struct
import time
import traceback
from torch_npu_sdpa_dropout_demo import keep_mask,reference


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    from torch.nn.attention import SDPBackend,sdpa_kernel
    started=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(428)
    dtype=getattr(torch,precision);F=torch.nn.functional.scaled_dot_product_attention
    checks=[];errors=[]
    def upload(x):return x.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)
    def close(actual,expected,gradient=False):
        a,b=actual.detach().cpu(),expected.detach().cpu()
        torch.testing.assert_close(a,b,rtol=.004 if dtype==torch.float16 else 4e-5,
                                   atol=.001 if dtype==torch.float16 else 2e-6,equal_nan=True)
        if gradient and a.numel():
            error=float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30))
            errors.append(error);assert error<=(.01 if dtype==torch.float16 else 1e-4),error
    def resident(before):
        after=npu.stats()
        assert after['host_download_copies']==before['host_download_copies']
        assert after['host_device_copies']==before['host_device_copies']
    def one(name,shapes,*,gqa=False,causal=False,mask_kind=None,p=0.,partial=False,needs=(True,True,True),scale=-.7):
        bases=[(torch.randn(s[:-1]+(s[-1]*2,) if partial else s,dtype=dtype)*.125).requires_grad_(need)
               for s,need in zip(shapes,needs)]
        leaves=[upload(x) for x in bases]
        hosts=[x[...,::2] for x in bases] if partial else bases
        values=[x[...,::2] for x in leaves] if partial else leaves
        heads=shapes[0][-3] if gqa else None
        cpu_kv=[x.repeat_interleave(heads//x.shape[-3],dim=-3) if gqa and x.shape[-3] else x for x in hosts[1:]]
        kv_batches=[tuple(x.shape[:-3])+(heads,) if gqa else tuple(x.shape[:-2]) for x in hosts[1:]]
        score_shape=tuple(torch.broadcast_shapes(hosts[0].shape[:-2],*kv_batches))+(shapes[0][-2],shapes[1][-2])
        score_elements=0 if not hosts[2].shape[-1] or gqa and not hosts[1].shape[-3] else prod(score_shape)
        mask=None
        if mask_kind:
            keep=torch.ones(score_shape[-2:],dtype=torch.bool)
            if keep.shape[0]>1:keep[1]=False
            if keep.shape[1]:keep[:,-1]=False
            mask=keep if mask_kind=='bool' else torch.where(keep,.125,float('-inf')).requires_grad_()
        device_mask=upload(mask) if mask is not None else None
        state=torch.intel_npu.get_rng_state()
        if p and score_elements:
            expected=reference(hosts[0],*cpu_kv,mask,p=p,causal=causal,scale=scale,keep=keep_mask(state,score_shape,p))
        else:
            with sdpa_kernel(SDPBackend.MATH):expected=F(*hosts,attn_mask=mask,dropout_p=p,is_causal=causal,scale=scale,enable_gqa=gqa)
        seed=torch.randn_like(expected)*.05;device_seed=upload(seed)
        cpu_state=torch.get_rng_state();before=npu.stats()
        actual=F(*values,attn_mask=device_mask,dropout_p=p,is_causal=causal,scale=scale,enable_gqa=gqa)
        advanced=torch.intel_npu.get_rng_state();actual.backward(device_seed);resident(before)
        assert torch.equal(cpu_state,torch.get_rng_state())
        assert torch.equal(advanced,torch.intel_npu.get_rng_state())
        start=struct.unpack('<8sQQ',bytes(state.tolist()))[2];end=struct.unpack('<8sQQ',bytes(advanced.tolist()))[2]
        assert end-start==(score_elements if 0<p<1 else 0),(start,end,score_shape)
        expected.backward(seed);close(actual,expected)
        for a,b,need in zip(leaves,bases,needs):
            if need:close(a.grad,b.grad,True)
            else:assert a.grad is None and b.grad is None
        if mask is not None and mask.requires_grad:
            if mask.grad is None:assert device_mask.grad is None
            else:close(device_mask.grad,mask.grad,True)
        if mask_kind and score_shape[-2]>1:assert torch.count_nonzero(actual.detach().cpu()[...,1,:])==0
        return dict(name=name,shapes=shapes,gqa=gqa,causal=causal,dropout=p,partial=partial,
                    score_elements=score_elements,needs=needs,no_payload_download=True,backward_rng_unchanged=True)

    def boundaries():
        cases=[]
        for name,s in (
            ('empty-batch',((0,2,3,4),(1,2,5,4),(1,2,5,6))),
            ('empty-query',((2,0,4),(1,5,4),(2,5,6))),
            ('empty-kv',((2,3,4),(1,0,4),(2,0,6))),
            ('zero-qk-feature',((2,3,0),(1,5,0),(2,5,6))),
            ('zero-value-feature',((2,3,4),(1,5,4),(2,5,0)))):
            for p in (0.,.25,1.):
                cases.append(one(name,s,p=p,scale=None,mask_kind='float'))
        for qh,kh in ((0,2),(2,0),(0,0)):
            cases.append(one('empty-gqa-heads',((2,qh,3,4),(1,kh,5,4),(1,kh,5,6)),gqa=True,p=.25))
        cases.append(one('zero-qk-causal',((4,3,0),(2,5,0),(2,5,3)),gqa=True,causal=True,p=.25,scale=None))
        cases.append(one('zero-qk-bool',((2,3,0),(1,5,0),(2,5,3)),mask_kind='bool',partial=True))
        cases.append(one('rank64-empty-query',((1,)*61+(2,0,4),(2,5,4),(2,5,3)),causal=True))
        return dict(cases=cases)

    def shapes():
        cases=[('rank2',((3,4),(5,4),(5,2)),False),
               ('mixed-rank',((2,1,3,2),(5,2),(1,3,5,3)),False),
               ('rank5',((2,1,3,4,2),(1,4,1,5,2),(2,1,1,5,3)),False),
               ('rank64',((1,)*61+(2,3,2),(2,5,2),(2,5,3)),True)]
        return dict(cases=[one(name,s,causal=c) for name,s,c in cases])
    def gqa():
        records=[one('gqa',((2,4,3,2),(1,2,5,2),(2,2,5,3)),gqa=True),
                 one('mqa-causal',((2,6,4,2),(1,1,5,2),(1,1,5,3)),gqa=True,causal=True),
                 one('mixed-rank-mask',((2,4,3,2),(2,5,2),(2,5,3)),gqa=True,mask_kind='float'),
                 one('partial-bool',((2,4,3,2),(1,2,5,2),(1,2,5,3)),gqa=True,mask_kind='bool',partial=True),
                 one('rank64-gqa',((1,)*61+(4,3,2),(2,5,2),(2,5,3)),gqa=True),
                 one('equal-heads',((2,3,2),(2,5,2),(2,5,3)),gqa=True)]
        for i in range(3):records.append(one(f'one-sided-{i}',((4,3,2),(2,5,2),(2,5,3)),gqa=True,needs=tuple(j==i for j in range(3))))
        return dict(cases=records)
    def dropout():
        return dict(cases=[one('gqa-dropout',((2,4,3,2),(1,2,5,2),(2,2,5,3)),gqa=True,mask_kind='float',p=.25),
                           one('mqa-causal-dropout',((4,5,2),(1,3,2),(1,3,3)),gqa=True,causal=True,p=.25),
                           one('gqa-full-dropout',((4,3,2),(2,5,2),(2,5,3)),gqa=True,p=1.)])
    def large():
        return one('larger-than-old-score-cap',((1,1,1025,1),(1,1,1024,1),(1,1,1024,1)),causal=True)
    def training():
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.norm=torch.nn.LayerNorm(8)
                self.q=torch.nn.Linear(8,8);self.k=torch.nn.Linear(8,4);self.v=torch.nn.Linear(8,4)
                self.proj=torch.nn.Linear(8,8);self.head=torch.nn.Linear(8,3)
            def forward(self,x):
                z=self.norm(x);prefix=tuple(z.shape[:-2]);length=z.shape[-2]
                q=self.q(z).view(prefix+(length,4,2)).transpose(-3,-2)
                k=self.k(z).view(prefix+(length,2,2)).transpose(-3,-2)
                v=self.v(z).view(prefix+(length,2,2)).transpose(-3,-2)
                h=F(q,k,v,is_causal=True,enable_gqa=True).transpose(-3,-2).contiguous().view(z.shape)
                return self.head((x+self.proj(h)).mean(-2))
        cpu=Model().to(dtype);model=copy.deepcopy(cpu).to('intel_npu')
        host=(torch.randn(2,1,9,8,dtype=dtype)*.125).requires_grad_();value=upload(host)
        target=torch.randn(2,1,3)*.1;labels=upload(target)
        opts=[torch.optim.SGD(m.parameters(),lr=.05,foreach=False,fused=False) for m in (cpu,model)]
        losses=[];graphs=[];near_zero=[]
        for step in range(3):
            for opt in opts:opt.zero_grad(set_to_none=True)
            host.grad=value.grad=None
            with sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.mse_loss(cpu(host).float(),target)
            before=npu.stats();actual=torch.nn.functional.mse_loss(model(value).float(),labels)
            actual.backward();resident(before);expected.backward();close(actual,expected);close(value.grad,host.grad,True)
            actual_gradients=[];reference_gradients=[]
            for (name,a),b in zip(model.named_parameters(),cpu.parameters()):
                close(a.grad,b.grad)
                actual_grad=a.grad.detach().cpu();reference_grad=b.grad.detach()
                actual_gradients.append(actual_grad.view(-1));reference_gradients.append(reference_grad.view(-1))
                if reference_grad.float().norm()<1e-7:
                    near_zero.append(dict(step=step+1,parameter=name,reference_norm=reference_grad.double().norm().item(),
                                          max_absolute_error=(actual_grad.double()-reference_grad.double()).abs().max().item()))
            # Key bias is shift-invariant; its near-zero gradient makes per-parameter relative norms ill-conditioned.
            close(torch.cat(actual_gradients),torch.cat(reference_gradients),True)
            for opt in opts:opt.step()
            for a,b in zip(model.parameters(),cpu.parameters()):close(a,b)
            losses.append(actual.item());graphs.append(npu.stats()['cached_sessions'])
            print('step',step+1,'loss',losses[-1],'graphs',graphs[-1],flush=True)
        assert losses[-1]<losses[0] and graphs[1:]==graphs[:1]*2
        return dict(parameters=sum(p.numel() for p in cpu.parameters()),rank5_attention=True,q_heads=4,kv_heads=2,
                    independent_steps=3,losses=losses,cached_graphs=graphs,
                    per_parameter_elementwise_and_global_gradient_l2=True,near_zero_gradients=near_zero)
    def contracts():
        q,k,v=[upload(torch.ones(s,dtype=dtype)) for s in ((4,3,2),(2,5,2),(2,5,3))]
        bad=upload(torch.ones(3,3,2,dtype=dtype));different=upload(torch.ones(1,5,3,dtype=dtype))
        longq=upload(torch.ones(1,npu.MAX_DMA_ELEMENTS//16384+1,1,dtype=dtype));longk=upload(torch.ones(1,16384,1,dtype=dtype))
        small=upload(torch.ones(1,1,1,dtype=dtype));wide=upload(torch.ones(1,65537,1,dtype=dtype))
        causal=upload(torch.ones(1,1048577,1,dtype=dtype))
        device_mask=upload(torch.ones(3,5,dtype=dtype))
        functions=[lambda:F(bad,k,v,enable_gqa=True,dropout_p=.25),lambda:F(q,k,different,enable_gqa=True),
                   lambda:F(q[0],k[0],v[0],enable_gqa=True),lambda:F(longq,longk,longk),
                   lambda:F(small,wide,wide),lambda:F(causal,small,small,is_causal=True),
                   lambda:F(q,k,v,enable_gqa=True,attn_mask=torch.ones(3,5)),
                   lambda:F(q,k,v,enable_gqa=True,is_causal=True,attn_mask=device_mask)]
        before=npu.stats()['npu_calls'];state=torch.intel_npu.get_rng_state()
        for fn in functions:
            try:fn()
            except (ValueError,RuntimeError,TypeError):pass
            else:raise AssertionError('Invalid SDPA accepted')
        assert npu.stats()['npu_calls']==before and torch.equal(state,torch.intel_npu.get_rng_state())
        leaves=[upload(torch.ones(s,dtype=dtype,requires_grad=True)) for s in ((4,3,2),(2,5,2),(2,5,3))]
        actual=F(*leaves,enable_gqa=True)
        with torch.no_grad():leaves[1].add_(1)
        try:actual.sum().backward()
        except RuntimeError as error:assert 'version' in str(error) or 'inplace' in str(error)
        else:raise AssertionError('Missing saved KV version guard')
        return dict(rejections=len(functions),no_submit_or_rng_advance=True,saved_kv_version=True)

    row=dict(name=case)
    try:row.update(dict(shapes=shapes,gqa=gqa,dropout=dropout,large=large,training=training,contracts=contracts,boundaries=boundaries)[case]());row['status']='passed'
    except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
    row['max_gradient_relative_l2']=max(errors,default=0.)
    print(case,precision,row['status'],row.get('error','')[:500],flush=True)
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes'))
    root=Path(__file__).resolve().parent
    sources=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_sdpa_extended_demo.py',
             'test_torch_npu_sdpa_extended.py','npunlock/tests/fixtures/npu4000/resident/manifest.json')
    result=dict(status='passed' if healthy and row['status']=='passed' else 'failed',case=case,precision=precision,
                checks=[row],stats=before,shutdown=final,elapsed_seconds=time.perf_counter()-started,
                source_sha256={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources})
    target=root/'npunlock-probe/research/results/sdpa-extended';target.mkdir(exist_ok=True)
    (target/f'{case}-{precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('shapes','gqa','dropout','large','training','contracts','boundaries'),required=True)
    parser.add_argument('--precision',choices=('float16','float32'),default='float32')
    args=parser.parse_args();raise SystemExit(execute(args.case,args.precision))

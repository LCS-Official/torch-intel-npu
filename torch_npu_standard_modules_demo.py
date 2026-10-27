"""Standard PyTorch MultiheadAttention/TransformerEncoderLayer migration checks."""
import argparse
import copy
import gc
import hashlib
import io
import json
from contextlib import contextmanager
from math import prod
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch
from torch_npu_sdpa_dropout_demo import keep_mask,reference

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/standard-modules'
CASES=('primitives','mha','encoder','inference','training','contracts')


@contextmanager
def cpu_rng_reference(state):
    """Only CPU test callbacks change; device modules always run outside this scope."""
    import torch
    from torch_npu_rng import RNGState
    rng=RNGState(0);rng.set_state(state)
    def draw(shape,p):
        keep=keep_mask(rng.get_state(),shape,p)
        if 0<p<1:rng.reserve(prod(shape))
        return keep
    def dropout(x,p=.5,training=True,inplace=False):
        assert x.device.type=='cpu' and not inplace
        if not training or p==0:return x
        return (x.float()*draw(tuple(x.shape),p)*(0. if p==1 else 1./(1.-p))).to(x.dtype)
    def sdpa(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,*,scale=None,enable_gqa=False):
        assert q.device.type=='cpu' and not enable_gqa
        shape=tuple(torch.broadcast_shapes(*(x.shape[:-2] for x in (q,k,v))))+(q.shape[-2],k.shape[-2])
        return reference(q,k,v,attn_mask,p=dropout_p,causal=is_causal,scale=scale,keep=draw(shape,dropout_p))
    with patch('torch.nn.functional.dropout',dropout),patch('torch.nn.functional.scaled_dot_product_attention',sdpa):
        yield rng


def probe():
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.manual_seed(193)
    rows=[]
    def run(kind,train,weights):
        cpu=(torch.nn.MultiheadAttention(4,2,dropout=0,batch_first=True) if kind=='mha' else
             torch.nn.TransformerEncoderLayer(4,2,dim_feedforward=8,dropout=0,batch_first=True))
        model=copy.deepcopy(cpu).to('intel_npu');cpu.train(train);model.train(train)
        host=(torch.randn(2,3,4)/4).requires_grad_(train);x=host.detach().to('intel_npu').requires_grad_(train)
        row=dict(kind=kind,training=train,need_weights=weights)
        try:
            with torch.set_grad_enabled(train):
                expected=cpu(host,host,host,need_weights=weights)[0] if kind=='mha' else cpu(host)
                actual=model(x,x,x,need_weights=weights)[0] if kind=='mha' else model(x)
                if train:actual.sum().backward();expected.sum().backward()
            torch.testing.assert_close(actual.detach().cpu(),expected.detach(),rtol=3e-5,atol=3e-6)
            row['status']='passed'
        except Exception as error:
            row.update(status='blocked',error=str(error),traceback=traceback.format_exc())
        return row
    for args in (('mha',True,False),('mha',True,True),('mha',False,False),('encoder',True,False),('encoder',False,False)):
        row=run(*args);rows.append(row);gc.collect();print(args,row['status'],(row.get('error','').splitlines() or [''])[0][:180],flush=True)
    npu.shutdown()
    return dict(status='probe-complete',checks=rows,final_stats=npu.stats())


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.manual_seed(197)
    dtype={'float16':torch.float16,'float32':torch.float32}[precision]
    initial_fastpath=torch.backends.mha.get_fastpath_enabled()
    checks=[]
    def upload(x):return x.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)
    def close(actual,expected,exact=False):
        torch.testing.assert_close(actual.detach().cpu(),expected.detach(),rtol=0 if exact else .003 if actual.dtype==torch.float16 else 3e-5,
                                   atol=0 if exact else .0005 if actual.dtype==torch.float16 else 3e-6,equal_nan=True)
    def no_download(before):assert npu.stats()['host_download_copies']==before['host_download_copies']
    def check(name,fn):
        gc.collect();before=npu.stats();row=dict(name=name,status='passed')
        try:row.update(fn())
        except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
        gc.collect();after=npu.stats();row['npu_calls']=after['npu_calls']-before['npu_calls']
        row['live_buffer_delta']=after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
        if row['live_buffer_delta'] or after['context_failed']:row.update(status='failed',lifecycle_error='Buffer or context')
        checks.append(row);print(name,row['status'],row.get('error','')[:500],flush=True)
    def compare_module(cpu,args,kwargs,*,train=True,weights=False):
        model=copy.deepcopy(cpu).to('intel_npu');cpu.train(train);model.train(train)
        cache={}
        def convert(x):
            if x is None:return None
            if id(x) not in cache:cache[id(x)]=upload(x)
            return cache[id(x)]
        values=tuple(convert(x) for x in args)
        options={k:convert(v) if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
        saved=torch.intel_npu.get_rng_state()
        with torch.set_grad_enabled(train):
            # This CPU's oneDNN GELU backward cannot run FP16. Keep the
            # standard CPU math reference, without changing the MHA fastpath.
            with torch.backends.mkldnn.flags(enabled=False),cpu_rng_reference(saved) as expected_rng:
                expected=cpu(*args,**kwargs)
            expected_state=expected_rng.get_state()
            expected_tuple=expected if isinstance(expected,tuple) else (expected,)
            seeds=[torch.randn_like(t)/8 if t is not None else None for t in expected_tuple]
            gradients=[upload(t) if t is not None else None for t in seeds]
            cpu_state=torch.get_rng_state();before=npu.stats();actual=model(*values,**options)
            actual_tuple=actual if isinstance(actual,tuple) else (actual,)
            if train:
                torch.autograd.backward([t for t in actual_tuple if t is not None],[g for g in gradients if g is not None])
            no_download(before)
            assert torch.equal(cpu_state,torch.get_rng_state()) and torch.equal(expected_state,torch.intel_npu.get_rng_state())
            if train:
                with torch.backends.mkldnn.flags(enabled=False):
                    torch.autograd.backward([t for t in expected_tuple if t is not None],[g for g in seeds if g is not None])
        for a,b in zip(actual_tuple,expected_tuple):
            if b is None:assert a is None
            else:close(a,b)
        if train:
            for a,b in zip(model.parameters(),cpu.parameters()):
                if b.grad is None:assert a.grad is None
                else:close(a.grad,b.grad)
            for host in args:
                if host.requires_grad:close(cache[id(host)].grad,host.grad)
        return True

    def primitives():
        for kind in ('add','sub','mul','div'):
            host=(torch.randn(4,3,4,dtype=dtype)/4).requires_grad_();x=upload(host)
            a,b=host[1:3,:,::2],x[1:3,:,::2]
            expected=getattr(torch,kind)(a,.75);actual=getattr(torch,kind)(b,.75)
            g=torch.ones_like(expected)/8;before=npu.stats();actual.backward(upload(g));no_download(before)
            expected.backward(g);close(actual,expected);close(x.grad,host.grad)
        hosts=[(torch.randn(shape,dtype=dtype)/4).requires_grad_() for shape in ((2,3,2),(3,3,4),(3,4,2))]
        tensors=[upload(x) for x in hosts]
        for beta,alpha in ((1.,1.),(.5,-.75),(0.,1.)):
            for t in hosts+tensors:t.grad=None
            h,a,b=hosts;v,x,y=tensors
            expected=torch.baddbmm(h[:1],a[1:],b[1:],beta=beta,alpha=alpha)
            before=npu.stats();actual=torch.baddbmm(v[:1],x[1:],y[1:],beta=beta,alpha=alpha)
            gradient=upload(torch.ones(tuple(expected.shape),dtype=dtype)/8);actual.backward(gradient);no_download(before)
            expected.backward(torch.ones_like(expected)/8);close(actual,expected)
            for t,h in zip(tensors,hosts):close(t.grad,h.grad)
        with torch.no_grad():
            x=upload(torch.ones(1,2,2,dtype=dtype));nan=upload(torch.full((1,2,2),float('nan'),dtype=dtype))
            out=torch.empty_like(x);assert torch.baddbmm(nan,x,x,beta=0,out=out) is out
            close(out,torch.full((1,2,2),2.,dtype=dtype),True)
            close(torch.baddbmm(x,nan,x,alpha=0),torch.full((1,2,2),float('nan'),dtype=dtype),True)
            base=upload(torch.arange(24,dtype=dtype));alias=torch.ops.aten._reshape_alias(base,[3,8],[8,1])
            assert alias.untyped_storage()._cdata==base.untyped_storage()._cdata
            version=alias._version;base.add_(1);assert alias._version==version+1;close(alias,torch.arange(1,25,dtype=dtype).view(3,8),True)
        return dict(partial_arithmetic_gradients=4,partial_baddbmm_coefficients=3,beta_zero_ignores_nan=True,
                    alpha_zero_preserves_nan=True,reshape_alias_storage_and_version=True)

    def mha():
        cases=[]
        for mode in ('sdpa','weights','cross','separate','unbatched'):
            seq_first=mode=='separate';unbatched=mode=='unbatched'
            model=torch.nn.MultiheadAttention(4,2,dropout=.25 if mode in ('sdpa','weights','separate') else 0.,
                batch_first=not seq_first,kdim=3 if mode=='separate' else None,vdim=2 if mode=='separate' else None,dtype=dtype)
            shape=(3,4) if unbatched else (3,2,4) if seq_first else (2,3,4)
            q=(torch.randn(shape,dtype=dtype)/4).requires_grad_()
            if mode in ('sdpa','weights'):k=v=q
            else:
                def source(e):return (torch.randn((4,e) if unbatched else (4,2,e) if seq_first else (2,4,e),dtype=dtype)/4).requires_grad_()
                k=source(3 if mode=='separate' else 4);v=k if mode=='cross' else source(2 if mode=='separate' else 4)
            options=dict(need_weights=mode!='sdpa' and mode!='separate',average_attn_weights=mode!='weights')
            if mode in ('sdpa','weights'):
                mask=torch.ones(3,3,dtype=torch.bool).triu(1);padding=torch.tensor([[False,False,True],[False,True,False]])
                if mode=='weights':mask=torch.where(mask,float('-inf'),0.).to(dtype);padding=torch.where(padding,float('-inf'),0.).to(dtype)
                options.update(attn_mask=mask,key_padding_mask=padding)
            if mode=='separate':options['key_padding_mask']=torch.tensor([[False,False,False,True],[False,True,False,False]])
            compare_module(model,(q,k,v),options);cases.append(mode)
        return dict(cases=cases,all_parameters_inputs_and_attention_weight_gradients=True,same_mask_cpu_reference=True)

    def encoder():
        cases=[]
        for norm_first,activation in ((False,'relu'),(True,'gelu')):
            model=torch.nn.TransformerEncoderLayer(4,2,dim_feedforward=8,dropout=.25,batch_first=True,
                                                   norm_first=norm_first,activation=activation,dtype=dtype)
            host=(torch.randn(2,3,4,dtype=dtype)/4).requires_grad_()
            options=dict(src_mask=torch.ones(3,3,dtype=torch.bool).triu(1),
                         src_key_padding_mask=torch.tensor([[False,False,True],[False,True,False]]))
            compare_module(model,(host,),options);cases.append(dict(norm_first=norm_first,activation=activation))
        return dict(cases=cases,attention_and_ffn_dropouts=True,all_parameters_and_input_gradients=True)

    def inference():
        cases=[]
        for mode in ('plain','merged','all-padding'):
            model=torch.nn.MultiheadAttention(4,2,dropout=.25,batch_first=True,dtype=dtype)
            host=torch.randn(2,3,4,dtype=dtype)/4;options=dict(need_weights=mode=='merged')
            if mode!='plain':options['key_padding_mask']=torch.tensor([[False,False,True],[False,True,False]])
            if mode=='merged':options['attn_mask']=torch.ones(3,3,dtype=torch.bool).triu(1)
            if mode=='all-padding':options['key_padding_mask'][1]=True
            compare_module(model,(host,host,host),options,train=False);cases.append('mha-'+mode)
        for norm_first,activation in ((False,'relu'),(True,'gelu')):
            model=torch.nn.TransformerEncoderLayer(4,2,dim_feedforward=8,dropout=.25,batch_first=True,
                                                   norm_first=norm_first,activation=activation,dtype=dtype)
            host=torch.randn(2,3,4,dtype=dtype)/4
            mask=torch.tensor([[0.,-.5,0.],[0.,0.,.125],[0.,0.,0.]])
            with torch.inference_mode():compare_module(model,(host,),dict(src_mask=mask),train=False)
            cases.append('encoder-'+activation)
        return dict(cases=cases,default_native_entry_points=True,global_fastpath_unchanged=True,
                    native_full_mask_nan_matches_cpu=True,inference_mode=True)

    def training():
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers=torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(4,2,dim_feedforward=8,dropout=.25,
                    batch_first=True,activation='gelu',norm_first=True) for _ in range(2)])
                self.head=torch.nn.Linear(4,3)
                self.register_buffer('mask',torch.ones(3,3,dtype=torch.bool).triu(1))
            def forward(self,x):
                for layer in self.layers:x=layer(x,src_mask=self.mask)
                return self.head(x.mean(1))
        cpu=Model();model=copy.deepcopy(cpu).to('intel_npu')
        host=torch.randn(2,3,4)/4;target=torch.randn(2,3)/4;x,labels=upload(host),upload(target)
        opts=[torch.optim.SGD(net.parameters(),lr=.02,foreach=False,fused=False) for net in (cpu,model)]
        checkpoint=io.BytesIO();losses=[]
        def cpu_copy():return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        for step in range(3):
            for opt in opts:opt.zero_grad(set_to_none=True)
            with cpu_rng_reference(torch.intel_npu.get_rng_state()) as tracker:expected=torch.nn.functional.mse_loss(cpu(host),target)
            before=npu.stats();loss=torch.nn.functional.mse_loss(model(x),labels);loss.backward();opts[1].step();no_download(before)
            assert torch.equal(tracker.get_state(),torch.intel_npu.get_rng_state())
            expected.backward();opts[0].step();close(loss,expected)
            for a,b in zip(model.parameters(),cpu.parameters()):close(a.grad,b.grad);close(a,b)
            losses.append(loss.item())
            if step==0:torch.save(dict(model=cpu_copy(),optimizer=opts[1].state_dict(),rng=torch.intel_npu.get_rng_state()),checkpoint)
            if step==1:second=cpu_copy();second_loss=loss.item();second_rng=torch.intel_npu.get_rng_state()
        checkpoint.seek(0);saved=torch.load(checkpoint,weights_only=True)
        model.load_state_dict({k:v.to('intel_npu') for k,v in saved['model'].items()});opts[1].load_state_dict(saved['optimizer'])
        torch.intel_npu.set_rng_state(saved['rng']);opts[1].zero_grad(set_to_none=True)
        before=npu.stats();loss=torch.nn.functional.mse_loss(model(x),labels);loss.backward();opts[1].step();no_download(before)
        assert loss.item()==second_loss and torch.equal(second_rng,torch.intel_npu.get_rng_state())
        for name,value in cpu_copy().items():torch.testing.assert_close(value,second[name],rtol=0,atol=0)
        model.eval();cpu.load_state_dict(second);cpu.eval();saved_rng=torch.intel_npu.get_rng_state()
        with torch.no_grad():close(model(x),cpu(host))
        assert torch.equal(saved_rng,torch.intel_npu.get_rng_state())
        return dict(layers=2,parameters=sum(p.numel() for p in model.parameters()),steps=3,losses=losses,
                    independent_cpu_same_mask_updates=True,serialized_checkpoint_replay_bitwise=True,default_eval_native_path=True)

    def contracts():
        base=upload(torch.ones(24));a=upload(torch.ones(2,3,4));b=upload(torch.ones(2,4,2));bias=upload(torch.ones(2,3,2))
        partial=base[:12].view(2,3,2);overlap=base.as_strided((3,3),(1,1))
        before=npu.stats();saved=torch.intel_npu.get_rng_state();rejected=0
        actions=[lambda:torch.add(overlap,1),lambda:torch.add(a,a,out=a[:1]),lambda:torch.baddbmm(bias,a,b,alpha=1e39),
                 lambda:torch.baddbmm(bias,a,b,out=partial),lambda:torch.baddbmm(bias.double(),a,b),
                 lambda:torch.ops.aten._reshape_alias(base,[100,100],[100,1]).clone()]
        for fn in actions:
            try:fn()
            except (ValueError,RuntimeError,NotImplementedError):rejected+=1
            else:raise AssertionError('Missing read/out/baddbmm metadata guard')
        assert npu.stats()['npu_calls']==before['npu_calls'] and torch.equal(saved,torch.intel_npu.get_rng_state())
        layer=torch.nn.TransformerEncoderLayer(4,2,dim_feedforward=8,dropout=0,batch_first=True,device='intel_npu')
        cpu=torch.nn.TransformerEncoderLayer(4,2,dim_feedforward=8,dropout=0,batch_first=True)
        cpu.load_state_dict({k:v.detach().cpu() for k,v in layer.state_dict().items()});cpu.eval();layer.eval()
        host=torch.randn(2,3,4)/4;value=upload(host)
        with torch.no_grad():close(layer(value),cpu(host))
        return dict(metadata_rejections=rejected,no_submission_on_rejection=True,direct_npu_encoder_initialization=True)

    check(case+'-'+precision,locals()[case]);gc.collect()
    assert torch.backends.mha.get_fastpath_enabled()==initial_fastpath
    npu.shutdown();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    sources=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_standard_modules_demo.py',
             'npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/_torch_intel_npu.pyd')
    return dict(status='passed' if healthy and all(c['status']=='passed' for c in checks) else 'failed',checks=checks,
                final_stats=final,fastpath_enabled=initial_fastpath,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('all','probe')+CASES,default='all')
    parser.add_argument('--precision',choices=('all','float16','float32'),default='all');args=parser.parse_args()
    if args.case=='probe':result=probe();name='probe'
    elif args.case=='all' or args.precision=='all':
        for case in CASES if args.case=='all' else (args.case,):
            for precision in ('float16','float32') if case in ('primitives','mha','encoder','inference') and args.precision=='all' else (args.precision if args.precision!='all' else 'float32',):
                if subprocess.run([sys.executable,'-B',__file__,'--case',case,'--precision',precision]).returncode:return 1
        return 0
    else:result=execute(args.case,args.precision);name=args.case+'-'+args.precision
    OUT.mkdir(parents=True,exist_ok=True);(OUT/(name+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status'] not in ('passed','probe-complete'))


if __name__=='__main__':raise SystemExit(main())

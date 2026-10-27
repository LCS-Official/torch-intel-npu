"""Bounded standard TransformerEncoder and token-classifier migration checks."""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import traceback
from torch_npu_standard_modules_demo import cpu_rng_reference

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/transformer-container'
PRECISIONS={'primitives':('float16','float32','int64','bool'),'boolean':('bool',),'masked':('float16',),
            'container':('float16','float32'),'classifier':('float32',),'contracts':('float32',)}
CASES=tuple(PRECISIONS)


def token_classifier(dtype=None, *, vocabulary=16, width=4, heads=2, length=3,
                     layers=2, feedforward=8, dropout=.25):
    import torch
    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding=torch.nn.Embedding(vocabulary,width,padding_idx=0)
            self.position=torch.nn.Parameter(torch.randn(1,length,width)/10)
            layer=torch.nn.TransformerEncoderLayer(width,heads,dim_feedforward=feedforward,dropout=dropout,batch_first=True,activation='gelu')
            # shortcut: dense storage only; add NestedTensor after backend storage support exists.
            self.encoder=torch.nn.TransformerEncoder(layer,layers,norm=torch.nn.LayerNorm(width),enable_nested_tensor=False)
            self.head=torch.nn.Linear(width,3)
            self.register_buffer('causal',torch.ones(length,length,dtype=torch.bool).triu(1))
        def forward(self,tokens):
            padding=tokens==0
            x=self.encoder(self.embedding(tokens)+self.position,mask=self.causal,src_key_padding_mask=padding)
            pooled=x.masked_fill(padding.unsqueeze(-1),0).sum(1)/padding.logical_not().to(x.dtype).sum(1,keepdim=True)
            return self.head(pooled)
    model=Classifier()
    return model.to(dtype=dtype) if dtype is not None else model


def cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def probe():
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.manual_seed(211)
    rows=[]
    def one(kind):
        layer=torch.nn.TransformerEncoderLayer(4,2,dim_feedforward=8,dropout=0,batch_first=True)
        cpu=torch.nn.TransformerEncoder(layer,2,enable_nested_tensor=kind!='dense-padding')
        model=copy.deepcopy(cpu).to('intel_npu');train=kind.endswith('train')
        cpu.train(train);model.train(train)
        host=(torch.randn(2,3,4)/4).requires_grad_(train);x=host.detach().to('intel_npu').requires_grad_(train)
        kwargs={}
        if kind=='causal-train':kwargs['mask']=torch.ones(3,3,dtype=torch.bool).triu(1)
        if kind.endswith('padding'):kwargs['src_key_padding_mask']=torch.tensor([[False,False,True],[False,True,True]])
        options={k:v.to('intel_npu') for k,v in kwargs.items()};row=dict(name=kind)
        try:
            with torch.set_grad_enabled(train):
                expected=cpu(host,**kwargs);actual=model(x,**options)
                if train:actual.sum().backward();expected.sum().backward()
            torch.testing.assert_close(actual.detach().cpu(),expected.detach(),rtol=3e-5,atol=3e-6,equal_nan=True)
            row['status']='passed'
        except Exception as error:row.update(status='blocked',error=str(error).splitlines()[0],traceback=traceback.format_exc())
        return row
    for kind in ('plain-train','causal-train','default-padding','dense-padding'):
        row=one(kind);rows.append(row);gc.collect();print(row['name'],row['status'],row.get('error','')[:200],flush=True)
    npu.shutdown()
    return dict(status='probe-complete',checks=rows,final_stats=npu.stats())


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.manual_seed(223)
    dtype=getattr(torch,precision);checks=[];initial_fastpath=torch.backends.mha.get_fastpath_enabled()
    def upload(x):return x.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)
    def close(a,b,exact=False):
        torch.testing.assert_close(a.detach().cpu(),b.detach(),rtol=0 if exact else .003 if a.dtype==torch.float16 else 3e-5,
                                   atol=0 if exact else .0005 if a.dtype==torch.float16 else 3e-6,equal_nan=True)
    def bits(a,b):
        assert torch.equal(a.detach().cpu().contiguous().view(torch.uint8),b.detach().contiguous().view(torch.uint8))
    def transfers(before,flags,rng_headers=0):
        after=npu.stats()
        downloads=after['host_download_copies']-before['host_download_copies']
        uploads=after['host_upload_copies']-before['host_upload_copies']
        constants=after['cached_scalar_buffers']-before['cached_scalar_buffers']
        assert downloads==flags,('download count',downloads,flags)
        # Each small Dropout uploads one 32-byte seed/counter header.
        assert uploads==constants+rng_headers,('upload count',uploads,constants,rng_headers)
        assert after['host_device_copies']==before['host_device_copies']
    def check(name,fn):
        gc.collect();before=npu.stats();row=dict(name=name,status='passed')
        try:row.update(fn())
        except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
        gc.collect();after=npu.stats();row['npu_calls']=after['npu_calls']-before['npu_calls']
        row['live_buffer_delta']=after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
        if row['live_buffer_delta'] or after['context_failed']:row.update(status='failed',lifecycle_error='Buffer or context')
        checks.append(row);print(name,row['status'],row.get('error','')[:600],flush=True)

    def primitives():
        count=0
        for shape in ((3,5),(2,3,4),(1,2,3,4)):
            size=1
            for dim in shape:size*=dim
            samples=([0,1,-1,2**60,-2**60,2**63-1,-2**63] if dtype==torch.int64 else
                     [True,False] if dtype==torch.bool else [0.,-0.,float('nan'),float('inf'),-float('inf'),2**-24,1.])
            host=torch.tensor((samples*((size+len(samples)-1)//len(samples)))[:size],dtype=dtype).reshape(shape)
            backing=torch.zeros((*shape[:-1],shape[-1]*2),dtype=dtype);backing[...,::2]=host;x=upload(backing)[...,::2]
            for upper in (True,False):
                operation=torch.triu if upper else torch.tril
                for diagonal in (-2**63,-1,0,2,2**63-1):
                    before=npu.stats();actual=operation(x,diagonal);transfers(before,0);bits(actual,operation(host,diagonal));count+=1
                    assert actual.untyped_storage()._cdata!=x.untyped_storage()._cdata
            out=torch.empty(shape,dtype=dtype,device='intel_npu')
            assert torch.triu(x,1,out=out) is out;bits(out,torch.triu(host,1))
        gradients=0
        if dtype in (torch.float16,torch.float32):
            for operation in (torch.triu,torch.tril):
                host=torch.randn(2,3,4,dtype=dtype,requires_grad=True);x=upload(host)
                seed=torch.randn_like(host);g=upload(seed);before=npu.stats()
                actual=operation(x,-1);actual.backward(g);transfers(before,0)
                expected=operation(host,-1);expected.backward(seed);bits(actual,expected);bits(x.grad,host.grad);gradients+=1
        return dict(triangles=count,partial_read=True,bitwise_storage=True,out=True,gradient_cases=gradients)

    def boolean():
        count=0
        for shape in ((),(2,3,4)):
            host=torch.ones(shape,dtype=torch.bool)
            if shape:host[0,1,2]=False
            x=upload(host)
            for operation in (torch.all,torch.any):
                for dim in (None,-1,(),(0,),tuple(range(len(shape)))):
                    for keepdim in (False,True):
                        before=npu.stats();actual=operation(x,dim=dim,keepdim=keepdim);transfers(before,0)
                        close(actual,operation(host,dim=dim,keepdim=keepdim),True);count+=1
        host=torch.tensor([[True,False,True],[True,True,True]]);x=upload(host).t()
        for operation in (torch.all,torch.any):
            out=torch.empty(2,dtype=torch.bool,device='intel_npu');assert operation(x,dim=0,out=out) is out
            close(out,operation(host.t(),dim=0),True)
        for size in (8193,65537):
            host=torch.ones(size,dtype=torch.bool);x=upload(host)
            assert torch.all(x).item() is True and torch.any(x).item() is True
            host[-1]=False;x=upload(host);assert torch.all(x).item() is False
            host.zero_();x=upload(host);assert torch.any(x).item() is False
        return dict(shape_dim_cases=count,scalar_and_empty_dims=True,transposed_out=True,tiled_sizes=[8193,65537])

    def masked():
        cases=[]
        for mode in ('random','nonfinite','subnormal','wide'):
            shape=(1,1,1,8193) if mode=='wide' else (1,1,1,5) if mode=='subnormal' else (2,2,3,4)
            host=torch.randn(shape,dtype=torch.float16)
            mask=torch.rand(shape)<.25
            if mode=='wide':host.zero_();mask.zero_()
            elif mode=='subnormal':host=torch.tensor([[[[0.,-17.,-16.,-14.,-10.]]]],dtype=torch.float16);mask.zero_()
            else:mask[1,1,2]=True
            if mode=='nonfinite':
                host[0,0,0,0]=float('nan');mask[0,0,0,0]=True
                host[0,1,1,1]=float('inf');mask[0,1,1,1]=False
            expected=torch.ops.aten._masked_softmax(host,mask,-1,2);x,m=upload(host),upload(mask)
            before=npu.stats()
            with torch.no_grad(),npu._lock:actual=npu._native_half_masked_softmax(x,m)
            transfers(before,0);value=actual.cpu();close(actual,expected,True)
            finite=~torch.isnan(expected)
            assert torch.equal(value[finite].view(torch.int16),expected[finite].view(torch.int16))
            cases.append(mode)
        cpu=torch.nn.MultiheadAttention(4,2,batch_first=True,dtype=torch.float16).eval()
        with torch.no_grad():cpu.in_proj_bias.copy_(torch.linspace(-.2,.2,12).half());cpu.out_proj.bias.fill_(.03125)
        model=copy.deepcopy(cpu).to('intel_npu');host=torch.randn(2,3,4,dtype=torch.float16)/4;x=upload(host)
        padding=torch.tensor([[False,False,True],[False,True,False]]);mask=upload(padding)
        with torch.no_grad():
            for average in (False,True):
                expected=cpu(host,host,host,key_padding_mask=padding,need_weights=True,average_attn_weights=average)
                before=npu.stats();actual=model(x,x,x,key_padding_mask=mask,need_weights=True,average_attn_weights=average);transfers(before,0)
                for a,b in zip(actual,expected):close(a,b)
        cases.append('biased-native-mha-averaged-and-per-head')
        return dict(cases=cases,native_cpu_half_rounding=True,finite_bits_exact=True,masked_nan_ignored=True,
                    fully_masked_rows_nan=True,subnormal_and_tile_boundary=True)

    def compare_container(cpu,host,kwargs,train,flags):
        model=copy.deepcopy(cpu).to('intel_npu');cpu.train(train);model.train(train);x=upload(host)
        options={k:upload(v) if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
        with torch.set_grad_enabled(train):
            saved=torch.intel_npu.get_rng_state()
            with torch.backends.mkldnn.flags(enabled=False),cpu_rng_reference(saved) as tracker:expected=cpu(host,**kwargs)
            seed=torch.randn_like(expected)/8;gradient=upload(seed)
            cpu_state=torch.get_rng_state();before=npu.stats();actual=model(x,**options)
            if train:actual.backward(gradient)
            transfers(before,flags,8 if train else 0)
            assert torch.equal(cpu_state,torch.get_rng_state()) and torch.equal(tracker.get_state(),torch.intel_npu.get_rng_state())
            if train:
                with torch.backends.mkldnn.flags(enabled=False):expected.backward(seed)
        close(actual,expected)
        if train:
            close(x.grad,host.grad)
            for a,b in zip(model.parameters(),cpu.parameters()):close(a.grad,b.grad)

    def container():
        cases=[]
        for mode in ('default-train','causal-padding-train','hinted-train','prenorm-train','sequence-train','default-eval','dense-padding-eval','dense-causal-eval'):
            train=mode.endswith('train');sequence=mode=='sequence-train';pre=mode=='prenorm-train'
            dense=pre or sequence or mode.startswith('dense-')
            layer=torch.nn.TransformerEncoderLayer(4,2,dim_feedforward=8,dropout=.25,batch_first=not sequence,
                norm_first=pre,activation='gelu' if pre or sequence else 'relu',dtype=dtype)
            cpu=torch.nn.TransformerEncoder(layer,2,norm=torch.nn.LayerNorm(4,dtype=dtype),enable_nested_tensor=not dense)
            host=(torch.randn((3,2,4) if sequence else (2,3,4),dtype=dtype)/4).requires_grad_(train)
            kwargs={};flags=0
            if mode in ('causal-padding-train','hinted-train','prenorm-train','dense-causal-eval'):
                kwargs['mask']=torch.ones(3,3,dtype=torch.bool).triu(1);flags=1
            if mode=='hinted-train':kwargs['is_causal']=True;flags=0
            if mode in ('causal-padding-train','dense-padding-eval','dense-causal-eval'):
                kwargs['src_key_padding_mask']=torch.tensor([[False,False,True],[False,True,True]])
            if mode=='dense-causal-eval':
                with torch.inference_mode():compare_container(cpu,host,kwargs,False,flags)
            else:compare_container(cpu,host,kwargs,train,flags)
            cases.append(mode);print('container',mode,'passed',flush=True)
        return dict(cases=cases,layers=2,final_layer_norm=True,all_parameter_and_input_gradients=True,
                    same_mask_cpu_reference=True,default_fastpath_unchanged=True)

    def classifier():
        cpu=token_classifier();model=copy.deepcopy(cpu).to('intel_npu')
        host=torch.tensor([[2,2,0],[3,4,0]],dtype=torch.int64);target=torch.tensor([0,2]);x,labels=upload(host),upload(target)
        def optimizer(net):return torch.optim.SGD(net.parameters(),lr=.02,momentum=.5,foreach=False,fused=False)
        opts=[optimizer(net) for net in (cpu,model)];checkpoint=io.BytesIO();losses=[]
        for step in range(3):
            for opt in opts:opt.zero_grad(set_to_none=True)
            with cpu_rng_reference(torch.intel_npu.get_rng_state()) as tracker:expected=torch.nn.functional.cross_entropy(cpu(host),target)
            cpu_state=torch.get_rng_state();before=npu.stats();logits=model(x);loss=torch.nn.functional.cross_entropy(logits,labels)
            loss.backward();opts[1].step();transfers(before,4,8)
            assert torch.equal(cpu_state,torch.get_rng_state()) and torch.equal(tracker.get_state(),torch.intel_npu.get_rng_state())
            expected.backward();opts[0].step();close(loss,expected)
            for a,b in zip(model.parameters(),cpu.parameters()):close(a.grad,b.grad);close(a,b)
            assert torch.count_nonzero(model.embedding.weight.grad.cpu()[0])==0
            losses.append(loss.item());print('classifier step',step+1,losses[-1],flush=True)
            state=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opts[1].state_dict()),rng=torch.intel_npu.get_rng_state())
            if step==0:torch.save(state,checkpoint)
            if step==1:second=state;second_loss=loss.item()
        checkpoint.seek(0);saved=torch.load(checkpoint,weights_only=True)
        restored=token_classifier().to('intel_npu');restored.load_state_dict(saved['model']);opt=optimizer(restored);opt.load_state_dict(saved['optimizer'])
        torch.intel_npu.set_rng_state(saved['rng']);opt.zero_grad(set_to_none=True)
        before=npu.stats();loss=torch.nn.functional.cross_entropy(restored(x),labels);loss.backward();opt.step();transfers(before,4,8)
        assert loss.item()==second_loss and torch.equal(second['rng'],torch.intel_npu.get_rng_state())
        for name,value in cpu_tree(restored.state_dict()).items():torch.testing.assert_close(value,second['model'][name],rtol=0,atol=0)
        current=cpu_tree(opt.state_dict())
        assert current['param_groups']==second['optimizer']['param_groups']
        for key,state in current['state'].items():
            for name,value in state.items():torch.testing.assert_close(value,second['optimizer']['state'][key][name],rtol=0,atol=0)
        cpu.load_state_dict(second['model']);cpu.eval();restored.eval();saved_rng=torch.intel_npu.get_rng_state()
        with torch.no_grad():
            expected=cpu(host);before=npu.stats();actual=restored(x);transfers(before,2);close(actual,expected)
        assert torch.equal(saved_rng,torch.intel_npu.get_rng_state())
        return dict(parameters=sum(p.numel() for p in model.parameters()),steps=3,losses=losses,embedding_positions_encoder_pool_head_ce=True,
                    independent_cpu_gradients_and_momentum_updates=True,fresh_model_optimizer_checkpoint_bitwise=True,
                    repeated_indices_and_zero_padding_gradient=True,training_scalar_downloads=4,eval_scalar_downloads=2)

    def contracts():
        x=upload(torch.ones(2,3));mask=upload(torch.ones(2,3,dtype=torch.bool));out=torch.empty(2,device='intel_npu',dtype=torch.bool)
        before=npu.stats();rng=torch.intel_npu.get_rng_state();count=0
        actions=[lambda:torch.triu(x.view(-1)),lambda:torch.triu(x,out=x),lambda:torch.tril(x,out=mask),
                 lambda:torch.all(x),lambda:torch.any(mask,dim=3),lambda:torch.all(mask,dim=(0,-2)),
                 lambda:torch.all(mask,dim=0,out=out),lambda:torch.all(mask,dim=(),out=mask)]
        for fn in actions:
            try:fn()
            except (ValueError,RuntimeError,NotImplementedError,IndexError):count+=1
            else:raise AssertionError('Missing triangle/BOOL reduction guard')
        assert npu.stats()['npu_calls']==before['npu_calls'] and torch.equal(rng,torch.intel_npu.get_rng_state())
        layer=torch.nn.TransformerEncoderLayer(4,2,dim_feedforward=8,dropout=0,batch_first=True)
        model=torch.nn.TransformerEncoder(layer,2).to('intel_npu').eval();src=upload(torch.randn(2,3,4)/4)
        padding=upload(torch.tensor([[False,False,True],[False,True,True]]))
        with torch.no_grad():
            try:model(src,src_key_padding_mask=padding)
            except (ValueError,NotImplementedError) as error:unsupported=str(error).splitlines()[0]
            else:raise AssertionError('Unexpected unverified default NestedTensor path')
        return dict(metadata_rejections=count,no_submission_on_rejection=True,default_nested_padding_not_supported=unsupported,
                    dense_padding_requires_enable_nested_tensor_false=True)

    check(case+'-'+precision,locals()[case]);gc.collect()
    assert torch.backends.mha.get_fastpath_enabled()==initial_fastpath
    npu.shutdown();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    sources=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_geometry.py','torch_npu_transformer_demo.py',
             'npunlock-probe/torch-device/_torch_intel_npu.pyd')
    return dict(status='passed' if healthy and all(r['status']=='passed' for r in checks) else 'failed',checks=checks,final_stats=final,
                fastpath_enabled=initial_fastpath,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=('all','probe')+CASES,default='all')
    parser.add_argument('--precision',choices=('all','float16','float32','int64','bool'),default='all');args=parser.parse_args()
    if args.case in PRECISIONS and args.precision!='all' and args.precision not in PRECISIONS[args.case]:
        parser.error(f'{args.case} supports precisions {PRECISIONS[args.case]}')
    if args.case=='probe':result=probe();name='probe'
    elif args.case=='all' or args.precision=='all':
        for case in CASES if args.case=='all' else (args.case,):
            for precision in PRECISIONS[case] if args.precision=='all' else tuple(p for p in PRECISIONS[case] if p==args.precision):
                if subprocess.run([sys.executable,'-B',__file__,'--case',case,'--precision',precision]).returncode:return 1
        return 0
    else:result=execute(args.case,args.precision);name=args.case+'-'+args.precision
    OUT.mkdir(parents=True,exist_ok=True);(OUT/(name+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status'] not in ('passed','probe-complete'))


if __name__=='__main__':raise SystemExit(main())

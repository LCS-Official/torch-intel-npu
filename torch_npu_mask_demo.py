"""BOOL storage, exact comparisons/selection, gradients and masked attention."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/bool-mask'
CASES = ('storage','compare','where','mixed','sdpa','contracts','training')


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(131)
    checks = []

    def upload(x):
        return x.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)

    def close(x,y,exact=True):
        torch.testing.assert_close(x.detach().cpu(),y.detach().cpu(),equal_nan=True,
            rtol=0 if exact else .005 if x.dtype==torch.float16 else 3e-5,
            atol=0 if exact else .0006 if x.dtype==torch.float16 else 3e-6)

    def check(name,function):
        before = npu.stats()
        row = dict(name=name)
        try:
            row.update(function() or {})
            row['status'] = 'passed'
        except Exception as error:
            row.update(status='failed',error=str(error),traceback=traceback.format_exc())
        gc.collect()
        after = npu.stats()
        row['npu_calls'] = after['npu_calls']-before['npu_calls']
        row['live_buffer_delta'] = after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
        if row['live_buffer_delta'] or after['context_failed']:
            row.update(status='failed',lifecycle_error='Live buffers or context failure')
        checks.append(row)
        print(name,row['status'],row.get('error','')[:400],flush=True)

    def no_download(before):
        assert npu.stats()['host_download_copies']==before['host_download_copies']

    def storage():
        count=0
        for shape in ((),(1,),(5,),(3,5),(2,3,5),(1,2,3,5)):
            host=torch.arange(torch.Size(shape).numel()).remainder(3).ne(0).reshape(shape)
            x=upload(host)
            assert x.dtype==torch.bool and x.element_size()==1
            assert x.untyped_storage().nbytes()==x.numel()
            close(x,host)
            clone=x.clone()
            assert clone.untyped_storage()._cdata!=x.untyped_storage()._cdata
            close(clone,host)
            function,dim=(torch.stack,-1) if x.ndim<4 else (torch.cat,0)
            close(function((x,x),dim),function((host,host),dim))
            x.fill_(True);close(x,torch.ones(shape,dtype=torch.bool))
            x.zero_();close(x,torch.zeros(shape,dtype=torch.bool))
            count+=1
        host=torch.arange(35).reshape(5,7).remainder(3).ne(0)
        x=upload(host)
        for v,h in ((x.t(),host.t()),(x[1::2,::3],host[1::2,::3]),(x[:1,:1].expand(3,5),host[:1,:1].expand(3,5))):
            clone=v.clone()
            assert clone.stride()==h.clone().stride()
            close(clone,h)
            close(v,h)
        for dtype in (torch.float16,torch.float32,torch.int64):
            close(x.to(dtype),host.to(dtype))
            values=torch.tensor([0,1,-1,2,0],dtype=dtype)
            if dtype.is_floating_point:
                values=torch.tensor([0.,-0.,float('nan'),float('inf'),2.**(-24 if dtype==torch.float16 else -149)],dtype=dtype)
            elif dtype==torch.int64:
                values=torch.tensor([0,2**32,2**53+1,-(2**63),2**63-1])
            data=upload(values)
            before=npu.stats()
            converted=data.bool()
            no_download(before)
            close(converted,values.bool())
        assert upload(torch.tensor(True)).item() is True
        return dict(shape_cases=count,odd_byte_lengths=True,clone_views_broadcast=True,bool_casts=6,scalar_item=True)

    def compare(dtype):
        if dtype==torch.int64:
            values=torch.tensor([-(2**63),-2**53-1,-1,0,1,2**53+1,2**63-1])
        elif dtype==torch.bool:
            values=torch.tensor([False,True,False,True])
        else:
            values=torch.tensor([0,-2147483648,1,-2147483647,0x00800000,-2139095040,0x3f800000,
                                 -1082130432,0x7f800000,-8388608,0x7fc00123,-4194303],dtype=torch.int32).view(torch.float32).to(dtype)
        a,b=values[:,None],values[None,:]
        x,y=upload(a),upload(b)
        count=0
        for kind in ('eq','ne','lt','le','gt','ge'):
            print('comparison',str(dtype),kind,flush=True)
            function=getattr(torch,kind)
            before=npu.stats()
            actual=function(x,y)
            no_download(before)
            assert actual.dtype==torch.bool and not actual.requires_grad
            close(actual,function(a,b))
            out=torch.empty_like(actual)
            assert function(x,y,out=out) is out
            close(out,function(a,b))
            close(function(x,0),function(a,0))
            count+=1
        if dtype==torch.float16:
            all_codes=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
            codes=upload(all_codes)
            for kind in ('eq','ne','lt','le','gt','ge'):
                close(getattr(torch,kind)(codes,0),getattr(torch,kind)(all_codes,0))
        if dtype==torch.bool:
            for kind in ('and','or','xor'):
                function=getattr(torch,'logical_'+kind)
                close(function(x,y),function(a,b))
                close(getattr(torch,'bitwise_'+kind)(x,y),function(a,b))
                out=torch.empty(a.shape[0],b.shape[1],device='intel_npu',dtype=torch.bool)
                assert function(x,y,out=out) is out
                close(out,function(a,b))
            close(~x,~a)
            close(torch.logical_not(x),torch.logical_not(a))
        else:
            h=torch.ones(3,dtype=dtype,requires_grad=dtype.is_floating_point)
            tensor=upload(h)
            assert not (tensor>0).requires_grad
        return dict(binary_comparisons=count,pairwise_special_values=True,scalar_out=True,
                    no_host_download=True,all_half_encodings=dtype==torch.float16,logical=dtype==torch.bool)

    def where(dtype):
        floating=dtype.is_floating_point
        a=torch.arange(3).to(dtype).reshape(3,1).requires_grad_(floating)
        b=torch.arange(5).to(dtype).reshape(1,5).requires_grad_(floating)
        mask=torch.arange(15).reshape(3,5).remainder(2).bool()
        x,y,m=upload(a),upload(b),upload(mask)
        before=npu.stats()
        actual=torch.where(m,x,y)
        no_download(before)
        expected=torch.where(mask,a,b)
        close(actual,expected)
        if floating:
            seed=torch.arange(15,dtype=dtype).reshape(3,5)/8
            gradient=upload(seed.t()).t()
            before=npu.stats()
            actual.backward(gradient)
            no_download(before)
            expected.backward(seed)
            close(x.grad,a.grad);close(y.grad,b.grad)
        with torch.no_grad():
            out=torch.empty_like(actual)
            assert torch.where(m,x,y,out=out) is out
            close(out,expected)
        for av,bv,ah,bh in ((x,0,a,0),(1,y,1,b),(True,False,True,False)):
            close(torch.where(m,av,bv),torch.where(mask,ah,bh))
        if floating:
            special=torch.tensor([0.,-0.,float('nan'),float('inf'),2.**-24],dtype=dtype)
        elif dtype==torch.int64:
            special=torch.tensor([2**63-1,-(2**63),2**53+1,-1,0])
        else:
            special=torch.tensor([True,False,False,True,True])
        labels=torch.tensor([True,False,True,False,True])
        v,w,condition=upload(special),upload(special.flip(0)),upload(labels)
        selected=torch.where(condition,v,w).cpu()
        reference=torch.where(labels,special,special.flip(0))
        assert selected.numpy().tobytes()==reference.numpy().tobytes()
        if floating:
            left=special.detach().requires_grad_()
            right=special.flip(0).detach().requires_grad_()
            a,b=upload(left),upload(right)
            seed=torch.tensor([float('nan'),float('inf'),-float('inf'),1.,-0.],dtype=dtype)
            torch.where(condition,a,b).backward(upload(seed))
            torch.where(labels,left,right).backward(seed)
            close(a.grad,left.grad);close(b.grad,right.grad)
        h=torch.arange(3).to(dtype).reshape(1,3).requires_grad_(floating)
        data=upload(h)
        host_mask=torch.tensor([[True,False,True],[False,True,False]])
        device_mask=upload(host_mask)
        value=torch.tensor(.25 if floating else 1,dtype=dtype,requires_grad=floating)
        scalar=upload(value)
        actual=data.masked_fill(device_mask,scalar)
        expected=h.masked_fill(host_mask,value)
        close(actual,expected)
        if floating:
            actual.sum().backward();expected.sum().backward()
            close(data.grad,h.grad);close(scalar.grad,value.grad)
        with torch.no_grad():
            out=torch.empty_like(actual)
            assert torch.ops.aten.masked_fill.Scalar_out(data,device_mask,value.item(),out=out) is out
            close(out,expected)
            local=upload(torch.arange(6).to(dtype).reshape(2,3))
            ref=torch.arange(6).to(dtype).reshape(2,3)
            before=local._version
            assert local.masked_fill_(device_mask,0) is local
            assert local._version==before+1
            close(local,ref.masked_fill(host_mask,0))
        return dict(broadcast_where=True,gradient=floating,raw_selected_bytes_exact=True,scalar_overloads=3,
                    masked_fill_tensor_value_gradient=floating,masked_fill_out=True,inplace_version=True)

    def mixed():
        rows=[]
        for count in (7,100,127):
            a=torch.ones(count,dtype=torch.float16,requires_grad=True)
            b=torch.ones((),dtype=torch.float32,requires_grad=True)
            condition=torch.zeros(count,dtype=torch.bool)
            x,y,mask=upload(a),upload(b),upload(condition)
            seed=torch.full((count,),.3333,dtype=torch.float16)
            gradient=upload(seed)
            before=npu.stats()
            actual=torch.where(mask,x,y)
            actual.backward(gradient)
            no_download(before)
            expected=torch.where(condition,a,b)
            expected.backward(seed)
            assert actual.dtype==expected.dtype==torch.float16
            rows.append(dict(count=count,cpu_gradient=b.grad.item(),npu_gradient=y.grad.item()))
            print('mixed scalar gradient',rows[-1],flush=True)
            close(x.grad,a.grad);close(y.grad,b.grad)
        return dict(cases=rows,fp16_output_fp32_scalar=True,exact_output_dtype_reduction_rounding=True)

    def sdpa(dtype):
        from torch.nn.attention import sdpa_kernel,SDPBackend
        count=0
        for scalar in (None,True,False):
            hosts=[(torch.randn(shape,dtype=dtype)/4).requires_grad_() for shape in ((2,2,3,4),(1,2,5,4),(2,1,5,2))]
            tensors=[upload(v) for v in hosts]
            mask=torch.tensor([[True,False,True,True,False],[False]*5,[True]*5]) if scalar is None else torch.tensor(scalar)
            if scalar is None:
                expanded=torch.zeros(3,10,dtype=torch.bool)
                expanded[:,::2]=mask
                device_mask=upload(expanded)[:,::2]
            else:device_mask=upload(mask)
            with sdpa_kernel(SDPBackend.MATH):
                expected=torch.nn.functional.scaled_dot_product_attention(*hosts,attn_mask=mask)
            seed=torch.randn_like(expected)/8
            gradient=upload(seed)
            before=npu.stats()
            actual=torch.nn.functional.scaled_dot_product_attention(*tensors,attn_mask=device_mask)
            actual.backward(gradient)
            no_download(before)
            expected.backward(seed)
            close(actual,expected,False)
            for x,h in zip(tensors,hosts):close(x.grad,h.grad,False)
            if scalar is False:
                assert torch.count_nonzero(actual.detach().cpu())==0
                assert all(torch.count_nonzero(v.grad.cpu())==0 for v in tensors)
            elif scalar is None:
                assert torch.count_nonzero(actual[:,:,1,:].detach().cpu())==0
                assert torch.count_nonzero(tensors[0].grad[:,:,1,:].cpu())==0
            count+=1
        return dict(cases=count,true_means_keep=True,partial_mask=True,batch_head_broadcast=True,
                    fully_masked_rows_zero=True,qkv_gradients=True,no_host_download=True)

    def contracts():
        mask=upload(torch.ones(3,5,dtype=torch.bool))
        x=upload(torch.ones(3,5))
        integer=upload(torch.ones(3,5,dtype=torch.int64))
        large=torch.empty(npu._storage_limit(torch.int64)+1,device='intel_npu',dtype=torch.bool)
        out=torch.empty(3,5,device='intel_npu')
        half=upload(torch.ones(3,5,dtype=torch.float16))
        actions={
            'where-float-condition':lambda:torch.where(x,x,x),
            'where-cpu-mask':lambda:torch.where(torch.ones(3,5,dtype=torch.bool),x,x),
            'where-shape':lambda:torch.where(mask,x,x.t()),
            'where-out-alias':lambda:torch.where(mask,x,x,out=x),
            'where-out-dtype':lambda:torch.where(mask,x,x,out=mask),
            'comparison-out-alias':lambda:torch.eq(mask,mask,out=mask),
            'fill-nonscalar':lambda:x.masked_fill(mask,x),
            'fill-wrong-tensor-dtype':lambda:x.masked_fill(mask,torch.tensor(1,dtype=torch.int64)),
            'fill-half-overflow':lambda:half.masked_fill(mask,65520.),
            'fill-out-alias':lambda:torch.ops.aten.masked_fill.Scalar_out(x,mask,0,out=x),
            'bool-to-int64-bound':lambda:large.to(torch.int64),
            'bool-rank65':lambda:torch.empty((1,)*(npu.MAX_RANK+1),device='intel_npu',dtype=torch.bool),
            'async-bool-copy':lambda:mask.to('cpu',non_blocking=True),
        }
        errors={}
        for name,action in actions.items():
            before=npu.stats()['npu_calls']
            try:action()
            except (ValueError,RuntimeError,NotImplementedError,IndexError) as error:errors[name]=str(error).splitlines()[0][:200]
            else:raise AssertionError('Missing rejection: '+name)
            assert npu.stats()['npu_calls']==before,name
        leaf=upload(torch.ones(3,5,requires_grad=True))
        result=torch.where(mask,leaf,0.)
        seed=upload(torch.ones(3,5))
        mask.zero_()
        before=npu.stats()['npu_calls']
        try:result.backward(seed)
        except RuntimeError as error:assert 'version' in str(error) or 'inplace' in str(error)
        else:raise AssertionError('Missing saved-mask version guard')
        assert npu.stats()['npu_calls']==before
        return dict(rejections=errors,no_submission_on_metadata_rejection=True,saved_mask_version_guard=True)

    def training():
        from torch.nn.attention import sdpa_kernel,SDPBackend
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q,self.k,self.v,self.proj=[torch.nn.Linear(4,4) for _ in range(4)]
                self.head=torch.nn.Linear(4,3)
                self.register_buffer('causal',torch.ones(3,3,dtype=torch.bool).tril())
            def forward(self,x,tokens):
                valid=tokens!=0
                mask=valid[:,None,None,:] & self.causal[None,None,:,:]
                q,k,v=[layer(x).view(2,3,2,2).transpose(1,2) for layer in (self.q,self.k,self.v)]
                output=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
                output=self.proj(output.transpose(1,2).contiguous().view(2,3,4))
                output=output.masked_fill((~valid)[:,:,None],0.)
                return self.head(output.mean(1))
        cpu=Model();model=copy.deepcopy(cpu).to('intel_npu')
        host,tokens,target=torch.randn(2,3,4)/4,torch.tensor([[1,2,0],[0,3,4]]),torch.tensor([0,2])
        x,ids,labels=upload(host),upload(tokens),upload(target)
        opts=[torch.optim.SGD(net.parameters(),lr=.03,momentum=.5,foreach=False,fused=False) for net in (cpu,model)]
        losses=[]
        for _ in range(4):
            for opt in opts:opt.zero_grad(set_to_none=True)
            with sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.cross_entropy(cpu(host,tokens),target)
            actual=torch.nn.functional.cross_entropy(model(x,ids),labels)
            losses.append(actual.item())
            expected.backward();actual.backward()
            close(actual,expected,False)
            for p,q in zip(model.parameters(),cpu.parameters()):close(p.grad,q.grad,False)
            for opt in opts:opt.step()
            for p,q in zip(model.parameters(),cpu.parameters()):close(p,q,False)
        assert losses[-1]<losses[0]
        return dict(precision='FP32',independent_steps=4,losses=losses,
                    int64_padding_comparison=True,bool_causal_padding_mask=True,masked_sdpa=True,masked_fill_backward=True)

    functions={'storage':storage,'mixed':mixed,'contracts':contracts,'training':training}
    if case in functions:check(case,functions[case])
    else:
        dtype={'float16':torch.float16,'float32':torch.float32,'int64':torch.int64,'bool':torch.bool}[precision]
        check(f'{case}-{precision}',lambda:{'compare':compare,'where':where,'sdpa':sdpa}[case](dtype))
    gc.collect()
    npu.shutdown()
    final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    sources=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_mask_demo.py',
             'npunlock/python/npunlock/npu4000.py','npunlock/tests/fixtures/npu4000/resident_mask.c',
             'npunlock/tests/fixtures/npu4000/resident/manifest.json','npunlock-probe/torch-device/backend.cpp',
             'npunlock-probe/torch-device/_torch_intel_npu.pyd')
    return dict(status='passed' if healthy and all(c['status']=='passed' for c in checks) else 'failed',checks=checks,final_stats=final,
                source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('all',)+CASES,default='all')
    parser.add_argument('--precision',choices=('all','float16','float32','int64','bool'),default='all')
    parser.add_argument('--label',default='after')
    args=parser.parse_args()
    if args.case=='all' or args.precision=='all' and args.case in ('compare','where','sdpa'):
        for case in CASES if args.case=='all' else (args.case,):
            precisions=('float16','float32') if case=='sdpa' else ('float16','float32','int64','bool')
            selected=('all',) if case in ('storage','mixed','contracts','training') else precisions if args.precision=='all' else (args.precision,)
            for precision in selected:
                if subprocess.run([sys.executable,'-B',__file__,'--case',case,'--precision',precision,'--label',args.label]).returncode:
                    return 1
        return 0
    result=execute(args.case,args.precision)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{args.label}-{args.case}-{args.precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status']!='passed')


if __name__=='__main__':
    raise SystemExit(main())

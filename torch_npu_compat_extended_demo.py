"""Independent CPU references for storage/index/math compatibility (real NPU)."""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.manual_seed(719)
    dtype=getattr(torch,precision);checks=[];started=time.perf_counter()
    def up(x):return x.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)
    def close(x,y,*,exact=False):
        a=x.detach().cpu();b=y.detach()
        if exact or not b.is_floating_point():assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8)),(a,b)
        else:torch.testing.assert_close(a,b,rtol=4e-5 if dtype==torch.float32 else .004,atol=2e-6 if dtype==torch.float32 else .001,equal_nan=True)
    def check(name,fn):
        row=dict(name=name);before=npu.stats()
        try:row.update(fn() or {});row['status']='passed'
        except Exception as e:row.update(status='failed',error=str(e),traceback=traceback.format_exc())
        gc.collect();after=npu.stats();row['npu_calls']=after['npu_calls']-before['npu_calls']
        checks.append(row);print(name,row['status'],row.get('error','')[:240],flush=True)
    def capacity():
        count=1048576+17;host=torch.linspace(-.75,.75,count,dtype=torch.float32).to(dtype);x=up(host)
        close(x+x,host+host);close(x.mean(),host.mean());close(x.clone(),host,exact=True)
        w=torch.zeros((512,512,3,3),dtype=dtype);close(up(w).clone(),w,exact=True)
        shape=(1,2,1,2,1,2,1,2);a=torch.randn(shape,dtype=dtype,requires_grad=True);b=up(a)
        (b*b).sum().backward();(a*a).sum().backward();close(b.grad,a.grad)
        boundary=torch.empty((16777216,),dtype=torch.float32,device='intel_npu')
        assert boundary.untyped_storage().nbytes()==64*1024*1024
        host_boundary=torch.zeros(16777216,dtype=torch.float32);host_boundary[-1]=-123.25
        boundary.copy_(host_boundary)
        close(npu._slice(boundary,0,1),host_boundary[:1],exact=True)
        close(npu._slice(boundary,16777215,1),host_boundary[-1:],exact=True)
        try:torch.empty((npu._storage_limit(torch.float32)+1,),dtype=torch.float32,device='intel_npu')
        except RuntimeError:pass
        else:raise AssertionError('Storage limit was not enforced')
        return dict(elements=count,large_conv_weight=w.numel(),allocation_boundary_mib=64,boundary_dma_readback=True,rank8_autograd=True)
    def types():
        for dt in (torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64):
            a=torch.tensor([-9,-2,-1,0,1,2,7,12],dtype=torch.int64).to(dt);b=torch.tensor([3,2,3,2,3,2,3,2],dtype=dt)
            x,y=up(a),up(b)
            for fn in (torch.add,torch.sub,torch.mul,torch.bitwise_and,torch.bitwise_or,torch.bitwise_xor,torch.floor_divide,torch.remainder):close(fn(x,y),fn(a,b),exact=True)
            for mode in ('floor','trunc'):close(torch.div(x,y,rounding_mode=mode),torch.div(a,b,rounding_mode=mode),exact=True)
            close(torch.add(x,y,alpha=3),torch.add(a,b,alpha=3),exact=True)
            close(x.sum(),a.sum(),exact=True);close(x.reshape(2,4).sum(0),a.reshape(2,4).sum(0),exact=True)
            for target in (torch.float16,torch.float32,torch.bfloat16,torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool):close(x.to(target),a.to(target),exact=True)
            close(torch.all(x),torch.all(a));close(torch.any(x),torch.any(a))
        a=torch.tensor([2**53+1,-2**53-3,2**62-1,-2**63,2**63-1]);b=torch.tensor([7,-11,3,1,1]);x,y=up(a),up(b)
        for fn in (torch.add,torch.sub,torch.mul,torch.floor_divide,torch.remainder):close(fn(x,y),fn(a,b),exact=True)
        a=torch.tensor([-.0,.0,.125,-.5,1.5,12.25,65504.],dtype=torch.float32)
        for target in (torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool,torch.bfloat16):
            close(up(a).to(target),a.to(target),exact=True)
        for dt in (torch.bfloat16,torch.int32,torch.int8,torch.uint8):
            a=torch.arange(24).reshape(4,6).to(dt);x=up(a);close(x[:,1:5].clone(),a[:,1:5],exact=True)
            close(x.t().clone(),a.t(),exact=True)
        a=torch.tensor([0,1,2,128,255],dtype=torch.uint8).view(torch.bool);x=up(a)
        for target in (torch.int64,torch.int32,torch.uint8,torch.float16,torch.float32):close(x.to(target),a.to(target),exact=True)
        close(x.sum(),a.sum(),exact=True)
        return dict(integer_types=5,casts=True,exact_above_2pow53=True,integer_sum=True)
    def math_values():
        for kind in ('sin','cos','tan','expm1','log1p','floor','ceil','trunc','round'):
            a=(torch.randn(3,4)*.5).to(dtype).requires_grad_();x=up(a);seed=torch.randn_like(a)
            y=getattr(torch,kind)(x);z=getattr(torch,kind)(a);close(y,z)
            y.backward(up(seed));z.backward(seed);close(x.grad,a.grad)
            out=torch.empty_like(x);getattr(torch,kind)(x.detach(),out=out);close(out,z)
        a=torch.tensor([-0.,0.,.5,-.5,1.5,-1.5,2.5,-2.5,float('inf'),float('-inf'),float('nan')],dtype=dtype)
        for kind in ('sin','cos','floor','ceil','trunc','round','expm1','log1p'):close(getattr(torch,kind)(up(a)),getattr(torch,kind)(a))
        return dict(unary=9,first_order=True,special_values=True,out=True)
    def trig_range():
        import numpy as np
        host=torch.from_numpy(np.random.default_rng(313).integers(0,2**32,8192,dtype=np.uint32).view(np.float32))
        maxima={}
        for name in ('sin','cos'):
            result=getattr(torch,name)(up(host));expected=getattr(torch,name)(host)
            torch.testing.assert_close(result.cpu(),expected,rtol=3e-5,atol=3e-7,equal_nan=True)
            finite=torch.isfinite(expected);maxima[name]=float((result.cpu()[finite]-expected[finite]).abs().max())
        return dict(random_float32_encodings=host.numel(),max_absolute=maxima)
    def powers():
        a=torch.tensor([.25,.5,1.,2.,3.],dtype=dtype,requires_grad=True);b=torch.tensor([-.5,0.,.5,1.5,3.],dtype=dtype,requires_grad=True)
        x,y=up(a),up(b);actual=x.pow(y);expected=a.pow(b);close(actual,expected);actual.sum().backward();expected.sum().backward();close(x.grad,a.grad);close(y.grad,b.grad)
        special=torch.tensor([-float('inf'),-4.,-1.,-0.,0.,.5,1.,4.,float('inf'),float('nan')],dtype=dtype)
        for exponent in (-3.,-2.,-.5,0.,.5,2.,3.,float('inf'),float('-inf'),float('nan')):close(up(special).pow(exponent),special.pow(exponent))
        for base in (0.,1e-300,1.0001,2.,-.5):
            a=torch.tensor([0.,1.,2.,3.],dtype=dtype,requires_grad=True);x=up(a)
            expected=torch.pow(base,a);actual=torch.pow(base,x);close(actual,expected)
            actual.sum().backward();expected.sum().backward();close(x.grad,a.grad)
        return dict(tensor_exponent=True,scalar_base=True,both_gradients=True,special_values=True)
    def indexing():
        a=torch.arange(120).reshape(4,5,6).to(dtype);x=up(a)
        for mask in (a%3==0,a<0,a>=0):
            m=up(mask);close(torch.nonzero(m),torch.nonzero(mask),exact=True);close(x[m],a[mask],exact=True)
            for p,q in zip(torch.where(m),torch.where(mask)):close(p,q,exact=True)
        for indices in ((torch.tensor([3,1,-1]),),
                        (torch.tensor([[0],[3]]),None,torch.tensor([[1,4,5]])),
                        (None,torch.tensor([4,1]),torch.tensor([5,0])),
                        (torch.tensor([True,False,True,False]),),
                        (torch.tensor([],dtype=torch.int64),)):
            device=[None if v is None else up(v) for v in indices];cpu=tuple(slice(None) if v is None else v for v in indices)
            close(torch.ops.aten.index.Tensor(x,device),a[cpu],exact=True)
        a=torch.randn(4,5,dtype=dtype,requires_grad=True);x=up(a);idx=torch.tensor([3,1,3,0]);g=torch.randn(4,5,dtype=dtype)
        x[up(idx)].backward(up(g));a[idx].backward(g);close(x.grad,a.grad)
        source=torch.randn(6,7,dtype=dtype);mask=source>0
        a=source.requires_grad_();x=up(a);x[up(mask)].sum().backward();a[mask].sum().backward();close(x.grad,a.grad)
        return dict(nonzero=True,boolean=True,broadcast_separated_indices=True,duplicates_gradient=True)
    def writing():
        for dt in (dtype,torch.int64,torch.int32,torch.uint8,torch.bool):
            a=torch.arange(30).reshape(5,6).to(dt);x=up(a)
            value=torch.tensor(1,dtype=dt);a[1:4,::2]=value;x[1:4,::2]=up(value);close(x,a,exact=True)
            a.t()[1:3].fill_(0);x.t()[1:3].fill_(0);close(x,a,exact=True)
            idx=torch.tensor([4,1,4]);v=torch.ones((3,6),dtype=dt);a[idx]=v;x[up(idx)]=up(v);close(x,a,exact=True)
            ids=torch.tensor([[0,2,2],[1,0,3]]);source=torch.arange(6).reshape(2,3).to(dt)
            z=torch.zeros((2,4),dtype=dt);n=up(z);z.scatter_(1,ids,source);n.scatter_(1,up(ids),up(source));close(n,z,exact=True)
        a=torch.arange(12,dtype=dtype).reshape(3,4);x=up(a);a[1:]=a[:-1].clone();x[1:]=x[:-1].clone();close(x,a,exact=True)
        return dict(partial_copy=True,strided_fill=True,index_assignment=True,scatter=True,alias_snapshot=True)
    def gradients():
        idx=torch.tensor([3,1,3]);g=torch.randn((5,4),dtype=dtype)
        for accumulate in (False,True):
            a=torch.randn((5,4),dtype=dtype,requires_grad=True);v=torch.randn((3,1),dtype=dtype,requires_grad=True)
            x,y=up(a),up(v);actual=x.index_put((up(idx),),y,accumulate=accumulate);expected=a.index_put((idx,),v,accumulate=accumulate)
            close(actual,expected);actual.backward(up(g));expected.backward(g);close(x.grad,a.grad);close(y.grad,v.grad)
        a=torch.randn((2,4),dtype=dtype,requires_grad=True);v=torch.randn((2,3),dtype=dtype,requires_grad=True);ids=torch.tensor([[0,2,2],[1,0,3]])
        x,y=up(a),up(v);actual=x.scatter(1,up(ids),y);expected=a.scatter(1,ids,v);seed=torch.randn_like(a)
        actual.backward(up(seed));expected.backward(seed);close(actual,expected);close(x.grad,a.grad);close(y.grad,v.grad)
        for kind in ('copy','fill','put','scatter'):
            a=torch.randn((5,4),dtype=dtype,requires_grad=True);v=torch.randn((2,2),dtype=dtype,requires_grad=True);x,y=up(a),up(v)
            cpu=a*2;device=x*2
            if kind=='copy':cpu[1:3,::2]=v;device[1:3,::2]=y
            elif kind=='fill':cpu[1:3,::2].fill_(.75);device[1:3,::2].fill_(.75)
            elif kind=='put':
                i=torch.tensor([0,3]);cpu[i]=v.repeat(1,2);device[up(i)]=torch.cat((y,y),1)
            else:
                i=torch.tensor([[0,2],[1,0],[3,1],[0,2],[1,3]]);source=cpu[:,:2].clone();payload=device[:,:2].clone()
                cpu.scatter_(1,i,source);device.scatter_(1,up(i),payload)
            close(device,cpu);device.sum().backward();cpu.sum().backward();close(x.grad,a.grad)
            if v.grad is not None:close(y.grad,v.grad)
        return dict(index_put=True,accumulate=True,scatter=True,view_copy_fill=True,inplace_index_autograd=True)
    def contracts():
        cases=[lambda:up(torch.ones(3))[up(torch.tensor([3]))],lambda:up(torch.ones(3))[up(torch.tensor([-4]))],
               lambda:up(torch.ones(3))[up(torch.tensor([True,False]))],
               lambda:up(torch.ones(1)).expand(3).fill_(1),
               lambda:torch.floor_divide(up(torch.ones(3,dtype=torch.int64)),up(torch.zeros(3,dtype=torch.int64)))]
        for fn in cases:
            try:fn()
            except (ValueError,IndexError,RuntimeError,ZeroDivisionError):pass
            else:raise AssertionError('Invalid operation was accepted')
        return dict(rejections=len(cases))
    def model():
        import copy
        class Large(torch.nn.Module):
            def __init__(self):
                super().__init__();self.first=torch.nn.Linear(2048,1024);self.head=torch.nn.Linear(1024,4)
            def forward(self,x):return self.head(torch.sin(self.first(x)))
        cpu=Large().to(dtype=dtype);device=copy.deepcopy(cpu).to('intel_npu');inp=torch.randn(1,2048,dtype=dtype)*.1;target=torch.randn(1,4,dtype=dtype)*.1
        optim=torch.optim.SGD(cpu.parameters(),lr=.01);other=torch.optim.SGD(device.parameters(),lr=.01);losses=[]
        for step in range(2):
            optim.zero_grad();other.zero_grad();expected=(cpu(inp)-target).square().mean();actual=(device(up(inp))-up(target)).square().mean()
            close(actual,expected);actual.backward();expected.backward()
            for a,b in zip(device.parameters(),cpu.parameters()):close(a.grad,b.grad)
            other.step();optim.step()
            for a,b in zip(device.parameters(),cpu.parameters()):close(a,b)
            losses.append(float(actual.item()))
        return dict(parameters=sum(p.numel() for p in cpu.parameters()),steps=2,independent_cpu_reference=True,losses=losses)
    def expanded():
        from concurrent.futures import ThreadPoolExecutor
        for dt in (torch.int64,torch.int32,dtype):
            a=torch.arange(650).reshape(130,5).to(dt);ids=torch.tensor([[4,0,4]]).expand(130,3).clone();x=up(a);i=up(ids)
            close(torch.gather(x,1,i),torch.gather(a,1,ids),exact=True)
            close(torch.index_select(x,1,up(torch.tensor([4,0,4],dtype=torch.int32))),a[:,[4,0,4]],exact=True)
            src=torch.arange(390).reshape(130,3).to(dt);close(x.scatter_add(1,i,up(src)),a.scatter_add(1,ids,src))
            close(x.index_put((up(torch.tensor([1,1,3])),),up(torch.ones((3,5),dtype=dt)),accumulate=True),a.index_put((torch.tensor([1,1,3]),),torch.ones((3,5),dtype=dt),accumulate=True))
        for fn in ('gather','index_select','scatter_add'):
            a=torch.randn(3,5,dtype=dtype,requires_grad=True);x=up(a);ids=torch.tensor([[4,0,4],[2,1,2],[3,0,3]]);i=up(ids)
            if fn=='gather':actual=x.gather(1,i);expected=a.gather(1,ids)
            elif fn=='index_select':actual=x.index_select(1,i[0].clone());expected=a.index_select(1,ids[0])
            else:
                v=torch.randn((3,3),dtype=dtype,requires_grad=True);y=up(v);actual=x.scatter_add(1,i,y);expected=a.scatter_add(1,ids,v)
            seed=torch.randn_like(expected);actual.backward(up(seed));expected.backward(seed);close(actual,expected);close(x.grad,a.grad)
            if fn=='scatter_add':close(y.grad,v.grad)
        for dt in (torch.int8,torch.int32,torch.int64):
            a=torch.tensor([-3,-1,0,2,5],dtype=dt);x=up(a);close(x.pow(3),a.pow(3),exact=True)
            with torch.no_grad():x[1:4].add_(2);a[1:4].add_(2)
            close(x,a,exact=True);close(torch.logical_not(x),torch.logical_not(a))
            close(torch.where(x>0,x,up(torch.zeros_like(a))),torch.where(a>0,a,torch.zeros_like(a)),exact=True)
        a=torch.randn(3,5,dtype=dtype,requires_grad=True);x=up(a);expected=a*2;actual=x*2
        mask=torch.tensor([[True,False],[False,True],[True,True]])
        expected[:,1:3].masked_fill_(mask,.25);actual[:,1:3].masked_fill_(up(mask),.25)
        close(actual,expected);actual.sum().backward();expected.sum().backward();close(x.grad,a.grad)
        a=torch.tensor([.25,-.5,1.25],dtype=torch.bfloat16,requires_grad=True);x=up(a)
        expected=(a*a+1).exp().sum();actual=(x*x+1).exp().sum();close(actual,expected);actual.backward();expected.backward();close(x.grad,a.grad)
        a=torch.randn(2,3,dtype=torch.float16,requires_grad=True);b=torch.randn(2,3,dtype=torch.float32,requires_grad=True);x,y=up(a),up(b)
        actual=torch.cat((x,y),1);expected=torch.cat((a,b),1);close(actual,expected);actual.sum().backward();expected.sum().backward();close(x.grad,a.grad);close(y.grad,b.grad)
        source=torch.arange(12).reshape(3,4).to(dtype);x=up(source)
        for flag in (True,False):close(x[torch.tensor(flag)],source[torch.tensor(flag)],exact=True)
        # Saving original indices protects backward even though offsets are computed afresh.
        a=up(torch.ones(4,3,dtype=dtype,requires_grad=True));ids=up(torch.tensor([1,1,3]));result=a[ids]
        ids.fill_(0)
        try:result.sum().backward()
        except RuntimeError as error:assert 'version' in str(error) or 'inplace' in str(error)
        else:raise AssertionError('Missing original index version guard')
        def worker(offset):
            a=torch.arange(17,dtype=torch.int64)+2**53+offset;x=up(a);close(x*3+7,a*3+7,exact=True)
            return True
        with ThreadPoolExecutor(max_workers=2) as pool:assert all(pool.map(worker,(1,19,31,41)))
        return dict(rows_130=True,gather_and_scatter_add=True,integer_pow_inplace=True,bfloat16_autograd=True,mixed_cat_gradients=True,scalar_boolean=True,index_version_guard=True)
    actions={'capacity':capacity,'types':types,'math':math_values,'trig':trig_range,'pow':powers,'index':indexing,'write':writing,'gradients':gradients,'contracts':contracts,'model':model,'expanded':expanded}
    for name,fn in actions.items():
        if case in ('all',name):check(name,fn)
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();after=npu.stats()
    report=dict(case=case,precision=precision,checks=checks,stats=before,shutdown=after,elapsed_seconds=time.perf_counter()-started)
    target=Path(__file__).resolve().parent/'npunlock-probe/research/results/compat-foundation';target.mkdir(exist_ok=True)
    (target/f'{case}-{precision}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert not after['context_failed'] and not after['quarantined_buffer_bytes']
    assert all(after[k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions'))
    return 0 if all(row['status']=='passed' for row in checks) else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',default='all',choices=('all','capacity','types','math','trig','pow','index','write','gradients','contracts','model','expanded'))
    parser.add_argument('--precision',default='float32',choices=('float32','float16'));args=parser.parse_args()
    raise SystemExit(execute(args.case,args.precision))

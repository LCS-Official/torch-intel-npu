"""S24: independent CPU references for the remaining vision/sorting families."""
import argparse
import gc
import json
from pathlib import Path
import traceback


def execute(case,precision):
    import torch
    import torch.nn.functional as F
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False;torch.manual_seed(824)
    dtype=getattr(torch,precision);checks=0
    def upload(x):return x.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)
    def close(a,b,exact=False):
        torch.testing.assert_close(a.detach().cpu(),b.detach().cpu(),equal_nan=True,
            rtol=0 if exact else .008 if dtype==torch.float16 else .0002,
            atol=0 if exact else .002 if dtype==torch.float16 else 2e-5)
    def check(fn,host,*,exact=False,gradient_exact=None):
        nonlocal checks
        data=upload(host);a=fn(data);b=fn(host)
        if isinstance(a,tuple):
            for left,right in zip(a[1:],b[1:]):close(left,right,True)
            a,b=a[0],b[0]
        close(a,b,exact)
        if host.requires_grad:
            seed=torch.randn_like(b);a.backward(upload(seed));b.backward(seed);close(data.grad,host.grad,exact if gradient_exact is None else gradient_exact)
        checks+=1
    before=npu.stats()
    def run():
        nonlocal checks
        if case=='foundation':
            for dt in (torch.float16,torch.float32,torch.int64,torch.bool):
                for shape in ((0,),(2,0,3),(0,2,3,4,5)):
                    x=torch.empty(shape,dtype=dt,requires_grad=dt in (torch.float16,torch.float32));a=upload(x)
                    v,i=torch.sort(a);close(v,x,True);close(i,torch.empty(shape,dtype=torch.int64),True)
                    close(a.clone(),x,True);close(torch.topk(a,0).values,x if shape[-1]==0 else x.narrow(-1,0,0),True)
                    if x.requires_grad:v.sum().backward();close(a.grad,torch.zeros_like(x),True)
                    checks+=1
            for shape in ((2,3,4,5,6),(1,2,2,3,4)):
                check(lambda x:(x.permute(0,2,1,4,3).contiguous()*2.+1.).sum((0,2)),torch.randn(shape,dtype=dtype,requires_grad=True))
                x=torch.randn(shape,dtype=dtype);check(lambda x:x+x.mean((2,3,4),keepdim=True),x.requires_grad_())
            check(lambda x:torch.topk(x,0),torch.tensor(2.,dtype=dtype,requires_grad=True),exact=True)
            check(lambda x:torch.topk(x,0),torch.randn(2,5,dtype=dtype,requires_grad=True),exact=True)
            check(lambda x:torch.where(x>0,x,-x),torch.randn(1,2,3,4,5,dtype=dtype,requires_grad=True),exact=True)
            check(lambda x:torch.sort(x[:0],stable=True),torch.randn(4,5,dtype=dtype,requires_grad=True),exact=True)
            check(lambda x:torch.sort(x[:,:0],stable=True),torch.randn(4,5,dtype=dtype,requires_grad=True),exact=True)
        elif case=='integer':
            for dt in (torch.int64,torch.bool):
                values=[-(2**63),2**63-1,2**53+1,2**53,-1,0,2**32+7,-2**32-7,0]
                x=torch.tensor(values,dtype=torch.int64).to(dt).repeat(2,1)
                for axis in (0,1):
                    for desc in (False,True):
                        a,i=torch.sort(upload(x),dim=axis,descending=desc,stable=True);b,j=torch.sort(x,dim=axis,descending=desc,stable=True)
                        close(a,b,True);close(i,j,True);checks+=1
                for k in (0,1,5,9):
                    a,i=torch.topk(upload(x),k);b,j=torch.sort(x,descending=True,stable=True)
                    close(a,b[:,:k],True);checks+=1
                for kind in ('max','min'):
                    a,i=getattr(torch,kind)(upload(x),dim=1);b,j=getattr(torch,kind)(x,dim=1);close(a,b,True);close(i,j,True);checks+=1
                if dt==torch.int64:
                    for kind in ('kthvalue','median','mode'):
                        a,i=getattr(torch,kind)(upload(x),3,dim=1) if kind=='kthvalue' else getattr(torch,kind)(upload(x),dim=1)
                        b,j=getattr(torch,kind)(x,3,dim=1) if kind=='kthvalue' else getattr(torch,kind)(x,dim=1)
                        close(a,b,True);checks+=1
            x=torch.randint(-(2**60),2**60,(1,9001),dtype=torch.int64)
            a,i=torch.sort(upload(x),stable=True);b,j=torch.sort(x,stable=True);close(a,b,True);close(i,j,True);checks+=1
        elif case=='statistics':
            for axis in (0,1,-1):
                for keep in (False,True):
                    for kind in ('kthvalue','median','nanmedian'):
                        fn=lambda x:getattr(torch,kind)(x,2,dim=axis,keepdim=keep) if kind=='kthvalue' else getattr(torch,kind)(x,dim=axis,keepdim=keep)
                        check(fn,torch.randn(3,5,dtype=dtype,requires_grad=True))
            for kind in ('median','nanmedian'):
                check(lambda x:getattr(torch,kind)(x),torch.tensor([1.,2.,2.,3.,4.],dtype=dtype,requires_grad=True))
            for kind in ('mode','nanmedian','median'):
                x=torch.tensor([[4.,1.,1.,2.,3.],[2.,4.,3.,2.,5.]],dtype=dtype,requires_grad=True)
                if kind=='mode':check(lambda x:torch.mode(x,dim=1),x)
                else:check(lambda x:getattr(torch,kind)(x,dim=1),x)
            if dtype==torch.float32:
                for kind in ('quantile','nanquantile'):
                    for interp in ('linear','lower','higher','nearest','midpoint'):
                        for axis in (None,0,1):
                            x=torch.randn(3,5,requires_grad=True)
                            check(lambda x:getattr(torch,kind)(x,.37,dim=axis,keepdim=True,interpolation=interp),x)
                    x=torch.randn(3,5,requires_grad=True)
                    check(lambda x:getattr(torch,kind)(x,torch.tensor([0.,.5,1.],device=x.device),dim=1),x)
            for kind in ('median','nanmedian'):
                x=torch.tensor([[1.,float('nan'),2.,3.,4.],[float('nan')]*5],dtype=dtype)
                close(getattr(torch,kind)(upload(x),dim=1).values,getattr(torch,kind)(x,dim=1).values)
                checks+=1
                check(lambda x:getattr(torch,kind)(x),torch.tensor([float('nan')]*3,dtype=dtype,requires_grad=True))
            for kind in ('kthvalue','median','nanmedian','mode'):
                host=torch.tensor([[1.,4.,2.,5.,3.]],dtype=dtype,requires_grad=True);x=upload(host)
                a=getattr(torch,kind)(x,2,dim=1).values if kind=='kthvalue' else getattr(torch,kind)(x,dim=1).values
                b=getattr(torch,kind)(host,2,dim=1).values if kind=='kthvalue' else getattr(torch,kind)(host,dim=1).values
                gradient=torch.full_like(b,float('inf'));a.backward(upload(gradient));b.backward(gradient);close(x.grad,host.grad);checks+=1
            if dtype==torch.float32:
                for kind in ('quantile','nanquantile'):
                    for interp in ('linear','lower','higher','midpoint','nearest'):
                        x=torch.tensor([1.,float('nan'),2.,3.])
                        close(getattr(torch,kind)(upload(x),.4,interpolation=interp),getattr(torch,kind)(x,.4,interpolation=interp));checks+=1
                        x=torch.tensor([float('inf')]);close(torch.quantile(upload(x),.5,interpolation=interp),torch.quantile(x,.5,interpolation=interp));checks+=1
                x=torch.randn(3,5,requires_grad=True)
                check(lambda x:torch.quantile(x,torch.empty(0,device=x.device),dim=1),x)
                check(lambda x:torch.quantile(x,1e-39),torch.tensor([0.,1e38],requires_grad=True))
        elif case=='contracts':
            def reject(fn):
                nonlocal checks
                try:fn()
                except (ValueError,RuntimeError,IndexError):checks+=1
                else:raise AssertionError('Invalid input accepted')
            x=upload(torch.randn(2,5,dtype=dtype));v=torch.empty(2,dtype=dtype,device='intel_npu');i=torch.empty(2,dtype=torch.int64,device='intel_npu')
            reject(lambda:torch.kthvalue(x,0));reject(lambda:torch.kthvalue(x,6))
            before=npu.stats()['npu_calls'];reject(lambda:torch.median(x,dim=1,out=(x,i)));assert npu.stats()['npu_calls']==before
            if dtype==torch.float32:
                reject(lambda:torch.quantile(x,1.1));reject(lambda:torch.quantile(x,upload(torch.tensor([-1.,.5]))))
            coords=upload(torch.full((1,2,3,2),float('inf'),dtype=dtype));image=upload(torch.randn(1,1,3,4,dtype=dtype))
            reject(lambda:F.grid_sample(image,coords,align_corners=False))
            before=npu.stats()['npu_calls'];reject(lambda:F.pad(image,(4,0),'reflect'));assert npu.stats()['npu_calls']==before
            # A failed index/grid check must leave the context usable.
            close(x+x,2*x.cpu());checks+=1
            for fn in (lambda a:F.conv3d(a,torch.ones(1,1,2,2,2,device=a.device,dtype=dtype)),
                       lambda a:F.max_pool3d(a,2),lambda a:F.interpolate(a,size=(3,4,5),mode='trilinear',align_corners=False)):
                with torch.inference_mode():
                    host=torch.randn(1,1,3,4,5,dtype=dtype);close(fn(upload(host)),fn(host));checks+=1
            for cls in (torch.nn.BatchNorm3d,torch.nn.InstanceNorm3d):
                import copy
                cpu=cls(2,affine=True,track_running_stats=True).to(dtype).eval();net=copy.deepcopy(cpu).to('intel_npu')
                with torch.inference_mode():
                    host=torch.randn(1,2,3,4,5,dtype=dtype);close(net(upload(host)),cpu(host));checks+=1
            for family in ('conv','grid'):
                host=upload(torch.randn(1,1,3,4,5,dtype=dtype,requires_grad=True)) if family=='conv' else upload(torch.randn(1,1,3,4,dtype=dtype,requires_grad=True))
                saved=upload(torch.randn(1,1,2,2,2,dtype=dtype,requires_grad=True)) if family=='conv' else upload(torch.zeros(1,2,3,2,dtype=dtype,requires_grad=True))
                result=F.conv3d(host,saved) if family=='conv' else F.grid_sample(host,saved,align_corners=False)
                with torch.no_grad():saved.add_(1.)
                reject(lambda:result.sum().backward())
        elif case=='outputs':
            with torch.no_grad():
                for dt in (torch.int64,torch.bool,dtype):
                    host=torch.tensor([[5,2,4,1,3],[1,7,3,5,2]],dtype=dt);x=upload(host)
                    values=torch.empty_like(x);ids=torch.empty_like(x,dtype=torch.int64)
                    pair=torch.sort(x,stable=True,out=(values,ids));expected=torch.sort(host,stable=True)
                    assert pair.values is values and pair.indices is ids;close(values,expected.values,True);close(ids,expected.indices,True);checks+=1
                    assert torch.argsort(x,stable=True,out=ids) is ids;close(ids,expected.indices,True);checks+=1
                    v=torch.empty(2,1,dtype=dt,device=x.device);i=torch.empty(2,1,dtype=torch.int64,device=x.device)
                    pair=torch.max(x,1,True,out=(v,i));ref=torch.max(host,1,True);assert pair.values is v and pair.indices is i;close(v,ref.values,True);close(i,ref.indices,True);checks+=1
                    if dt!=torch.bool:
                        pair=torch.kthvalue(x,3,dim=1,keepdim=True,out=(v,i));ref=torch.kthvalue(host,3,dim=1,keepdim=True)
                        assert pair.values is v and pair.indices is i;close(v,ref.values,True);close(i,ref.indices,True);checks+=1
                if dtype==torch.float32:
                    host=torch.randn(2,5);x=upload(host);out=torch.empty(3,2,device=x.device);q=upload(torch.tensor([0.,.5,1.]))
                    assert torch.quantile(x,q,dim=1,out=out) is out;close(out,torch.quantile(host,q.cpu(),dim=1));checks+=1
        else:
            extra=run_vision(case,dtype,upload,close,check)
            if extra:checks+=extra
    try:
        run()
        detail=dict(checks=checks);status='passed'
    except Exception as error:status='failed';detail=dict(checks=checks,error=str(error),traceback=traceback.format_exc())
    gc.collect();after=npu.stats();npu.shutdown();final=npu.stats()
    if final['context_failed'] or any(final[k] for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes')):status='failed'
    print(case,precision,status,detail,flush=True)
    return dict(status=status,case=case,precision=precision,details=detail,npu_calls=after['npu_calls']-before['npu_calls'],final_stats=final)


def run_vision(case,dtype,upload,close,check):
    import copy
    import torch
    import torch.nn.functional as F
    if case=='training':
        class Volume(torch.nn.Module):
            def __init__(self):
                super().__init__();self.conv=torch.nn.Conv3d(2,4,3,padding=1,groups=2);self.norm=torch.nn.GroupNorm(2,4)
                self.up=torch.nn.ConvTranspose3d(4,2,2,stride=2);self.head=torch.nn.Linear(16,3)
            def forward(self,x):
                x=F.relu(self.norm(self.conv(x)),inplace=True)
                x=F.avg_pool3d(x.float(),2).to(x.dtype);x=self.up(x)
                x=F.interpolate(x,size=(4,4,4),mode='trilinear',align_corners=False)
                x=F.avg_pool3d(x.float(),2).to(x.dtype)
                return self.head(x.flatten(1))
        cpu=Volume().to(dtype);net=copy.deepcopy(cpu).to('intel_npu');host=torch.randn(2,2,4,4,4,dtype=dtype)/3;data=upload(host)
        optimizers=[torch.optim.SGD(m.parameters(),lr=.003,momentum=.5,foreach=False) for m in (net,cpu)]
        target=torch.randn(2,3,dtype=dtype)/3;reference=upload(target)
        for step in range(3):
            losses=[]
            for model,opt,x,y in zip((net,cpu),optimizers,(data,host),(reference,target)):
                opt.zero_grad(set_to_none=True);loss=F.mse_loss(model(x).float(),y.float());loss.backward();losses.append(loss)
            close(*losses)
            for p,q in zip(net.parameters(),cpu.parameters()):close(p.grad,q.grad)
            for opt in optimizers:opt.step()
            for p,q in zip(net.parameters(),cpu.parameters()):close(p,q)
        return 3
    if case=='warp':
        host=torch.randn(1,2,4,5,dtype=dtype)/3;data=upload(host)
        theta=(torch.eye(2,3,dtype=dtype).unsqueeze(0)*.8).requires_grad_();a=upload(theta)
        for step in range(3):
            results=[]
            for x,t in ((data,a),(host,theta)):
                grid=F.affine_grid(t,x.shape,align_corners=False)
                value=F.grid_sample(x.float() if x.device.type=='cpu' else x,grid.float() if x.device.type=='cpu' else grid,align_corners=False,padding_mode='reflection').to(dtype)
                loss=(value.float()*value.float()).mean();loss.backward();results.append(loss)
            close(*results);close(a.grad,theta.grad)
            with torch.no_grad():a.add_(a.grad,alpha=-.01);theta.add_(theta.grad,alpha=-.01)
            a.grad=None;theta.grad=None;close(a,theta)
        return 3
    if case=='grid':
        for dims in (2,3):
            shape=(1,2,4,5) if dims==2 else (1,2,3,4,5)
            for mode in (('nearest','bilinear','bicubic') if dims==2 else ('nearest','bilinear')):
                for padding in ('zeros','border','reflection'):
                    for align in (False,True):
                        host=torch.randn(shape,dtype=dtype,requires_grad=True)
                        coords=(torch.rand((1,2,3,2) if dims==2 else (1,2,2,3,3),dtype=dtype)*3.6-1.8).requires_grad_()
                        a,b=upload(host),upload(coords)
                        # Torch CPU grid_sampler does not support Half; use explicit
                        # FP32 opmath on the same rounded input and grid values.
                        expected=F.grid_sample(host.float(),coords.float(),mode=mode,padding_mode=padding,align_corners=align).to(dtype)
                        actual=F.grid_sample(a,b,mode=mode,padding_mode=padding,align_corners=align);close(actual,expected)
                        seed=torch.randn_like(expected);actual.backward(upload(seed));expected.backward(seed)
                        close(a.grad,host.grad);close(b.grad,coords.grad)
            theta=torch.randn((1,dims,dims+1),dtype=dtype,requires_grad=True)/3;theta=theta.detach().requires_grad_()
            for align in (False,True):
                check(lambda t:F.affine_grid(t,shape,align_corners=align),theta.detach().clone().requires_grad_())
        return 30
    if case=='images':
        for mode in ('bilinear','bicubic'):
            for align in (False,True):
                for size in ((3,4),(8,9)):
                    check(lambda x:F.interpolate(x.float() if x.device.type=='cpu' else x,size=size,mode=mode,align_corners=align,antialias=True).to(x.dtype),torch.randn(1,2,5,6,dtype=dtype,requires_grad=True))
        for shape,padding in (((2,3,5),(2,1)),((1,2,4,5),(2,1,1,2)),((1,2,3,4,5),(1,2,1,1,1,0))):
            for mode in ('constant','reflect','replicate','circular'):
                check(lambda x:F.pad(x,padding,mode,value=.3 if mode=='constant' else None),torch.randn(shape,dtype=dtype,requires_grad=True))
        for fn,shape in ((lambda x:F.pixel_shuffle(x,2),(2,8,3,4)),(lambda x:F.pixel_unshuffle(x,2),(1,2,6,8)),(lambda x:torch.channel_shuffle(x,2),(1,6,3,4))):
            check(fn,torch.randn(shape,dtype=dtype,requires_grad=True),exact=True)
        for mode in ('constant','reflect','replicate','circular'):
            for padding in ((-1,2),(2,-1)):
                check(lambda x:F.pad(x,padding,mode),torch.randn(1,2,5,dtype=dtype,requires_grad=True))
        for value in (float('inf'),-float('inf'),float('nan'),-0.):
            check(lambda x:F.pad(x,(1,2),value=value),torch.randn(1,2,5,dtype=dtype,requires_grad=True))
        for kernel,dilation,padding,stride in (((2,3),(1,1),(1,1),(1,2)),((2,2),(2,1),(1,0),(2,1))):
            check(lambda x:F.unfold(x,kernel,dilation,padding,stride),torch.randn(1,2,5,6,dtype=dtype,requires_grad=True),exact=True,gradient_exact=False)
            cols=F.unfold(torch.randn(1,2,5,6,dtype=dtype),kernel,dilation,padding,stride).requires_grad_()
            check(lambda x:F.fold(x,(5,6),kernel,dilation,padding,stride),cols)
        return
    if case=='pooling':
        for unbatched in (False,True):
            shape=(2,4,5,6) if unbatched else (1,2,4,5,6)
            for ceil in (False,True):
                check(lambda x:F.max_pool3d(x,(2,3,2),(1,2,2),(0,1,0),(1,1,2),ceil,return_indices=True),torch.randn(shape,dtype=dtype,requires_grad=True))
                for include in (False,True):
                    check(lambda x:F.avg_pool3d(x.float() if x.device.type=='cpu' else x,(2,3,2),(1,2,2),(0,1,0),ceil,include).to(x.dtype),torch.randn(shape,dtype=dtype,requires_grad=True))
            check(lambda x:F.adaptive_avg_pool3d(x.float() if x.device.type=='cpu' else x,(3,2,4)).to(x.dtype),torch.randn(shape,dtype=dtype,requires_grad=True))
            check(lambda x:F.adaptive_max_pool3d(x,(3,2,4),return_indices=True),torch.randn(shape,dtype=dtype,requires_grad=True))
        check(lambda x:F.adaptive_max_pool2d(x,(3,4),return_indices=True),torch.randn(1,2,5,6,dtype=dtype,requires_grad=True))
        check(lambda x:F.adaptive_avg_pool2d(x,(9,9)),torch.randn(1,2,5,6,dtype=dtype,requires_grad=True))
        for dims,shape in ((1,(1,2,8)),(2,(1,2,6,8)),(3,(1,2,4,6,8))):
            def unpool(x):
                values,ids=getattr(F,'max_pool'+str(dims)+'d')(x,2,return_indices=True)
                return getattr(F,'max_unpool'+str(dims)+'d')(values,ids,2,output_size=x.shape)
            check(unpool,torch.randn(shape,dtype=dtype,requires_grad=True),exact=True)
        return
    if case=='volume':
        for transpose in (False,True):
            for groups in (1,2):
                fn=F.conv_transpose3d if transpose else F.conv3d
                shape=(2,2,3,4,5);wshape=(2,2 if transpose else 1,2,2,2) if groups==2 else (2,3,2,2,2) if transpose else (3,2,2,2,2)
                x=torch.randn(shape,dtype=dtype,requires_grad=True)/3;x=x.detach().requires_grad_()
                w=(torch.randn(wshape,dtype=dtype)/3).requires_grad_();bias=torch.randn(wshape[1]*groups if transpose else wshape[0],dtype=dtype,requires_grad=True)
                a,b,c=upload(x),upload(w),upload(bias);kwargs=dict(stride=(2,1,1),padding=(1,0,1),dilation=(2,1,1),groups=groups)
                if transpose:kwargs['output_padding']=(1,0,0)
                out=fn(a,b,c,**kwargs);expected=fn(x,w,bias,**kwargs);close(out,expected)
                seed=torch.randn_like(expected);out.backward(upload(seed));expected.backward(seed)
                for left,right in ((a.grad,x.grad),(b.grad,w.grad),(c.grad,bias.grad)):close(left,right)
        for mode in ('nearest','nearest-exact','trilinear'):
            for align in ((False,True) if mode=='trilinear' else (None,)):
                check(lambda x:F.interpolate(x,size=(4,3,6),mode=mode,align_corners=align),torch.randn(1,2,3,4,5,dtype=dtype,requires_grad=True))
        check(lambda x:F.softmax(x,dim=1),torch.randn(1,3,2,3,4,dtype=dtype,requires_grad=True))
        labels=torch.randint(0,3,(1,2,3,4))
        check(lambda x:F.cross_entropy(x,labels.to(x.device),label_smoothing=.1),torch.randn(1,3,2,3,4,dtype=dtype,requires_grad=True))
        for norm in (torch.nn.BatchNorm3d(2),torch.nn.InstanceNorm3d(2,affine=True,track_running_stats=True),torch.nn.GroupNorm(2,2)):
            cpu=norm.to(dtype);net=copy.deepcopy(cpu).to('intel_npu');x=torch.randn(2,2,3,4,5,dtype=dtype,requires_grad=True);a=upload(x)
            out=F.relu(net(a),inplace=True);ref=F.relu(cpu(x),inplace=True);seed=torch.randn_like(ref)
            close(out,ref);out.backward(upload(seed));ref.backward(seed);close(a.grad,x.grad)
            for p,q in zip(net.parameters(),cpu.parameters()):close(p.grad,q.grad)
        return 7
    raise ValueError('Unknown case '+case)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',required=True,
        choices=('foundation','integer','statistics','images','pooling','volume','grid','training','warp','contracts','outputs'))
    parser.add_argument('--precision',choices=('float16','float32'),default='float32');parser.add_argument('--label',default='after');args=parser.parse_args()
    result=execute(args.case,args.precision)
    out=Path(__file__).resolve().parent/'npunlock-probe/research/results/vision-sort-extended';out.mkdir(parents=True,exist_ok=True)
    (out/f'{args.label}-{args.case}-{args.precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(result['status']!='passed')

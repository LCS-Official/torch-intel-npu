"""Weighted/soft classification and broadcast MSE against independent CPU autograd."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
import warnings
from math import prod


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    F=torch.nn.functional;A=torch.ops.aten;dtype=getattr(torch,precision)
    torch.manual_seed(429);torch.set_num_threads(1);started=time.perf_counter();checks=[];errors=[]
    def upload(x):return x.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)
    def close(a,b):
        a,b=a.detach().cpu(),b.detach().cpu()
        torch.testing.assert_close(a,b,rtol=.004 if dtype==torch.float16 else 4e-5,
                                   atol=.001 if dtype==torch.float16 else 2e-6,equal_nan=True)
        if a.numel() and torch.isfinite(a).all() and torch.isfinite(b).all():
            errors.append(float((a.double()-b.double()).abs().max()))
    def compare(name,fn,hosts,*,downloads=0):
        hosts=[x.detach().clone().requires_grad_(x.requires_grad) for x in hosts];values=[upload(x) for x in hosts]
        expected=fn(*hosts);g=torch.randn_like(expected);dg=upload(g);before=npu.stats()
        actual=fn(*values);actual.backward(dg);after=npu.stats()
        assert after['host_download_copies']-before['host_download_copies']==downloads,(name,'downloads',before,after)
        assert after['host_device_copies']==before['host_device_copies'],(name,'uploads')
        expected.backward(g);close(actual,expected)
        for a,b in zip(values,hosts):
            if b.requires_grad:close(a.grad,b.grad)
            else:assert a.grad is None
        checks.append(dict(name=name,output_shape=list(actual.shape),output_dtype=str(actual.dtype),
                           gradients=sum(x.requires_grad for x in hosts),validation_scalar_downloads=downloads))
        print(name,'passed',flush=True)
        return values,hosts
    def hard():
        for shape in ((3,),(2,3),(2,3,2),(2,3,2,1,1),(0,3,2),(2,3,0,1)):
            target_shape=() if len(shape)==1 else (shape[0],*shape[2:])
            x=torch.randn(shape,dtype=dtype,requires_grad=True);target=torch.zeros(target_shape,dtype=torch.int64)
            if target.numel()>1:target.view(-1)[-1]=-2**63
            w=torch.tensor([.5,2.,0.],dtype=dtype)
            for red in ('none','mean','sum'):
                for smooth in (0.,.2,1.):
                    compare(f'ce-{shape}-{red}-s{smooth}',lambda x,t,w:F.cross_entropy(x,t,weight=w,reduction=red,
                            ignore_index=-2**63,label_smoothing=smooth),[x,target,w],downloads=int(x.numel()>0))
        for weights in ([0.,0.,0.],[-.5,2.,1.],[2.,float('nan'),float('inf')]):
            x=torch.randn(2,3,dtype=dtype,requires_grad=True);t=torch.tensor([0,-100]);w=torch.tensor(weights,dtype=dtype)
            for red in ('none','sum','mean'):
                compare(f'nll-{weights}-{red}',lambda x,t,w:F.nll_loss(x,t,weight=w,reduction=red),[x,t,w],downloads=1)
        x=torch.randn(2,3,dtype=dtype,requires_grad=True);t=torch.full((2,),-100);w=torch.ones(3,dtype=dtype)
        for smooth in (0.,.2,1.):
            compare(f'ce-all-ignored-s{smooth}',lambda x,t,w:F.cross_entropy(x,t,weight=w,label_smoothing=smooth),[x,t,w],downloads=1)
        compare('ce-partial-view',lambda x,t,w:F.cross_entropy(x[:,::2],t,weight=w[::2],label_smoothing=.2),
                [torch.randn(2,6,dtype=dtype,requires_grad=True),torch.tensor([0,2]),torch.rand(6,dtype=dtype)],downloads=1)
        for scale in (60000.,1.e-7):
            for shape in ((3,),(2,3)):
                x=torch.full(shape,-1.25,dtype=dtype,requires_grad=True);t=torch.zeros(shape[:-1],dtype=torch.int64)
                w=torch.full((3,),scale,dtype=dtype)
                for red in ('none','mean','sum'):
                    compare(f'nll-extreme-{scale}-{shape}-{red}',lambda x,t,w:F.nll_loss(x,t,weight=w,reduction=red),
                            [x,t,w],downloads=1)
    def tiled():
        for shape in ((513,257),(2,17,33,65)):
            target_shape=(shape[0],*shape[2:]);classes=shape[1]
            x=torch.randn(shape,dtype=dtype,requires_grad=True);t=torch.arange(prod(target_shape)).reshape(target_shape)%classes
            t.view(-1)[-1]=-2**63;w=torch.linspace(.5,1.5,classes,dtype=dtype)
            for red in ('none','mean','sum'):
                compare(f'tiled-ce-{shape}-{red}',lambda x,t,w:F.cross_entropy(x,t,weight=w,reduction=red,
                        ignore_index=-2**63,label_smoothing=.2),[x,t,w],downloads=1)
        x=torch.randn(513,257,dtype=dtype,requires_grad=True);w=torch.ones(257,dtype=dtype)
        compare('tiled-all-ignored',lambda x,t,w:F.nll_loss(x,t,weight=w),
                [x,torch.full((513,),-100,dtype=torch.int64),w],downloads=1)
        w[1]=float('nan');w[2]=float('inf');t=torch.zeros(513,dtype=torch.int64);t[-1]=-100
        compare('tiled-unselected-nonfinite-weights',lambda x,t,w:F.nll_loss(x,t,weight=w),[x,t,w],downloads=1)
        compare('maximum-class-row',lambda x,t:F.nll_loss(x,t),
                [-torch.ones(1,65532,dtype=dtype,requires_grad=True),torch.tensor([65531])],downloads=1)
        x=torch.randn(513,257,dtype=dtype);t=torch.arange(513)%257;w=torch.ones(257,dtype=dtype)
        dx,dt,dw=(upload(v) for v in (x,t,w))
        output,total,packet=npu._nll_forward(dx,dt,dw,1,-100);assert packet.numel()==0
        expected,ref_total=A.nll_loss_forward(x,t,w,1,-100);close(output,expected);close(total,ref_total)
        grad=torch.ones((),dtype=dtype);dest=torch.empty_like(dx);before=npu.stats()
        got=A.nll_loss_backward.grad_input(upload(grad),dx,dt,dw,1,-100,total,grad_input=dest)
        assert got is dest and npu.stats()['host_download_copies']-before['host_download_copies']==1
        close(got,A.nll_loss_backward(grad,x,t,w,1,-100,ref_total))
        checks.append(dict(name='tiled-native',rows=513,classes=257,compact_saved_packet=True,native_backward_out=True))
        for bad in (257,2**40):
            labels=t.clone();labels[-1]=bad;dlabels=upload(labels)
            try:A.nll_loss_forward(dx,dlabels,dw,1,-100)
            except IndexError:pass
            else:raise AssertionError('Invalid label in last tile accepted')
            try:A.nll_loss_backward(upload(grad),dx,dlabels,dw,1,-100,total)
            except IndexError:pass
            else:raise AssertionError('Invalid backward label in last tile accepted')
        for what in ('target','weight'):
            xx=upload(x.requires_grad_());labels=upload(t);weights=upload(w);loss=F.nll_loss(xx,labels,weight=weights)
            with torch.no_grad():(labels if what=='target' else weights).add_(1)
            try:loss.backward()
            except RuntimeError as error:assert 'version' in str(error) or 'inplace' in str(error)
            else:raise AssertionError('Missing tiled NLL saved version guard')
        xx=upload(torch.randn(2,17,33,65,dtype=dtype,requires_grad=True));labels=upload(torch.zeros(2,33,65,dtype=torch.int64))
        loss=F.cross_entropy(xx,labels)
        with torch.no_grad():labels.add_(1)
        try:loss.backward()
        except RuntimeError as error:assert 'version' in str(error) or 'inplace' in str(error)
        else:raise AssertionError('Missing original contiguous spatial label version guard')
        model=torch.nn.Conv2d(2,17,1,dtype=dtype);cpu=copy.deepcopy(model);model=model.to('intel_npu')
        host=torch.randn(2,2,33,65,dtype=dtype)*.1;data=upload(host);labels=torch.arange(4290).reshape(2,33,65)%17;dl=upload(labels)
        opts=[torch.optim.SGD(m.parameters(),lr=.01,foreach=False) for m in (model,cpu)];losses=[]
        for step in range(2):
            for opt in opts:opt.zero_grad(set_to_none=True)
            with torch.backends.mkldnn.flags(enabled=False):expected=F.cross_entropy(cpu(host).float(),labels)
            before=npu.stats();actual=F.cross_entropy(model(data).float(),dl)
            actual.backward();assert npu.stats()['host_download_copies']-before['host_download_copies']==1
            with torch.backends.mkldnn.flags(enabled=False):expected.backward()
            close(actual,expected)
            for a,b in zip(model.parameters(),cpu.parameters()):close(a.grad,b.grad)
            for opt in opts:opt.step()
            for a,b in zip(model.parameters(),cpu.parameters()):close(a,b)
            losses.append(actual.item())
        checks.append(dict(name='tiled-segmentation-training',steps=2,losses=losses,independent_cpu_gradients_and_updates=True))
    def tall():
        if dtype!=torch.float32:raise ValueError('Tall scalar-reduction probe uses FP32 loss')
        rows=1048593;x=-torch.ones(rows,3,dtype=dtype);x.requires_grad_()
        labels=torch.zeros(rows,dtype=torch.int64);labels[::11]=-100
        values,hosts=compare('million-rows-nll-mean',lambda x,t:F.nll_loss(x,t),[x,labels],downloads=1)
        actual,expected=values[0].grad.cpu().double(),hosts[0].grad.double()
        relative=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected))
        assert relative<1e-4,relative
        assert npu.stats()['cached_sessions']<256,npu.stats()['cached_sessions']
        checks.append(dict(name='tall-storage',rows=rows,input_elements=x.numel(),
                           packet_count=len(npu.nll_row_tiles(rows,3,max_elements=npu.MAX_DMA_ELEMENTS)),
                           cache_below_256=True,gradient_relative_l2=relative))
    def soft():
        for shape in ((3,),(2,3),(2,3,2,1),(0,3,2)):
            for mixed in (False,True):
                target_dtype=torch.float32 if mixed else dtype
                hosts=[torch.randn(shape,dtype=dtype,requires_grad=True),
                       torch.randn(shape,dtype=target_dtype,requires_grad=True),
                       torch.rand(shape[0] if len(shape)==1 else shape[1],dtype=target_dtype,requires_grad=True)]
                for red in ('none','mean','sum'):
                    for smooth in (0.,.2,1.):
                        compare(f'soft-{shape}-{mixed}-{red}-s{smooth}',lambda x,t,w:F.cross_entropy(x,t,weight=w,reduction=red,
                                label_smoothing=smooth,ignore_index=-2),hosts)
        compare('soft-partial-views',lambda x,t,w:F.cross_entropy(x[:,::2],t[:,::2],weight=w[::2]),
                [torch.randn(2,6,dtype=dtype,requires_grad=True),torch.rand(2,6,dtype=dtype,requires_grad=True),
                 torch.rand(6,dtype=dtype,requires_grad=True)])
    def mse():
        for shapes in (((2,1),(1,3)),((),(2,3)),((2,3),(3,)),((0,1),(1,3)),((1,)*63+(2,),(2,))):
            for mixed in (False,True):
                hosts=[torch.randn(shapes[0],dtype=dtype,requires_grad=True),
                       torch.randn(shapes[1],dtype=torch.float32 if mixed else dtype,requires_grad=True)]
                for red in ('none','mean','sum'):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',UserWarning)
                        compare(f'mse-{shapes}-{mixed}-{red}',lambda x,t:F.mse_loss(x,t,reduction=red),hosts)
        compare('mse-strided',lambda x,t:F.mse_loss(x[:,::2],t[:,1::2]),
                [torch.randn(2,6,dtype=dtype,requires_grad=True),torch.randn(1,6,dtype=dtype,requires_grad=True)])
        for red in (0,1,2):
            compare(f'mse-raw-aten-{red}',lambda x,t:A.mse_loss(x,t,red),
                    [torch.randn(2,1,dtype=dtype,requires_grad=True),torch.randn(1,3,dtype=dtype,requires_grad=True)])
    def native():
        with torch.no_grad():
            for spatial in (False,True):
                x=torch.randn((2,3,2,2) if spatial else (2,3),dtype=dtype)
                t=torch.zeros((2,2,2) if spatial else (2,),dtype=torch.int64);t.reshape(-1)[-1]=-100
                w=torch.tensor([.5,2.,1.],dtype=dtype);values=[upload(v) for v in (x,t,w)]
                forward=A.nll_loss2d_forward if spatial else A.nll_loss_forward
                backward=A.nll_loss2d_backward if spatial else A.nll_loss_backward
                for red in (0,1,2):
                    expected,total=forward(x,t,w,red,-100);actual,weight=forward(*values,red,-100)
                    close(actual,expected);close(weight,total)
                    out,tw=torch.empty_like(actual),torch.empty_like(weight)
                    got=forward.output(*values,red,-100,output=out,total_weight=tw)
                    assert got[0] is out and got[1] is tw;close(out,expected);close(tw,total)
                    g=torch.randn_like(expected);dg=upload(g)
                    # Native backward must use the supplied total_weight, not recompute it.
                    custom=total*2 if red==1 else total;dtotal=upload(custom)
                    result=backward(dg,*values,red,-100,dtotal);ref=backward(g,x,t,w,red,-100,custom)
                    close(result,ref);dest=torch.empty_like(result)
                    assert backward.grad_input(dg,*values,red,-100,dtotal,grad_input=dest) is dest;close(dest,ref)
                    checks.append(dict(name=f'native-nll-{spatial}-{red}',out_identity=True,caller_total=True))
            for mixed in (False,True):
                x=torch.randn(2,1,dtype=dtype);t=torch.randn(1,3,dtype=torch.float32 if mixed else dtype)
                dx,dt=upload(x),upload(t)
                for red in (0,1,2):
                    ref=A.mse_loss(x,t,red);out=torch.empty(ref.shape,dtype=ref.dtype,device='intel_npu')
                    assert A.mse_loss.out(dx,dt,red,out=out) is out;close(out,ref)
                    g=torch.randn_like(ref);dg=upload(g);dest=torch.empty(2,3,dtype=dtype,device='intel_npu')
                    refback=A.mse_loss_backward.grad_input(g,x,t,red,grad_input=torch.empty(2,3,dtype=dtype))
                    close(A.mse_loss_backward(dg,dx,dt,red),refback)
                    assert A.mse_loss_backward.grad_input(dg,dx,dt,red,grad_input=dest) is dest;close(dest,refback)
                    checks.append(dict(name=f'native-mse-{mixed}-{red}',out_identity=True,broadcast_backward=True))
        compare('native-nll2d-autograd',lambda x,t,w:A.nll_loss2d_forward(x,t,w,1,-100)[0],
                [torch.randn(2,3,2,2,dtype=dtype,requires_grad=True),torch.zeros(2,2,2,dtype=torch.int64),
                 torch.rand(3,dtype=dtype)],downloads=1)
    def contracts():
        x=upload(torch.ones(2,3,dtype=dtype));t=upload(torch.tensor([0,1]));w=upload(torch.ones(3,dtype=dtype))
        short=upload(torch.ones(2,dtype=dtype));integer=upload(torch.ones(2,3,dtype=torch.int64))
        wg=upload(torch.ones(3,dtype=dtype,requires_grad=True));wrong=upload(torch.ones(3,dtype=torch.float32 if dtype==torch.float16 else torch.float16))
        scalar=upload(torch.ones((),dtype=dtype));badout=torch.empty(1,dtype=dtype,device='intel_npu')
        actions=[lambda:F.cross_entropy(x,t,weight=short),lambda:F.cross_entropy(x,t,weight=wrong),
                 lambda:F.cross_entropy(x,t,weight=wg),lambda:F.cross_entropy(x,t,weight=torch.ones(3,dtype=dtype)),
                 lambda:F.cross_entropy(x,integer),lambda:F.cross_entropy(x,x,ignore_index=0),
                 lambda:F.cross_entropy(x,x,label_smoothing=1.1),lambda:A.nll_loss_forward(x,t,w,3,-100),
                 lambda:A.mse_loss(x,short,1),lambda:A.mse_loss.out(x,x,0,out=x),
                 lambda:A.mse_loss.out(x,x,1,out=badout),
                 lambda:A.mse_loss_backward(x,x,x,1),lambda:A.nll_loss_backward(scalar,x,t,w,1,-100,short),
                 lambda:A.nll_loss_forward.output(x,t,w,1,-100,output=scalar,total_weight=scalar)]
        before=npu.stats()['npu_calls']
        for fn in actions:
            try:fn()
            except (ValueError,RuntimeError,TypeError):pass
            else:raise AssertionError('invalid loss metadata accepted')
        assert npu.stats()['npu_calls']==before
        checks.append(dict(name='metadata',rejections=len(actions),no_submit=True))
        invalid=upload(torch.tensor([0,2**40]))
        try:F.cross_entropy(x,invalid,weight=w)
        except IndexError:pass
        else:raise AssertionError('invalid INT64 label accepted')
        for change in ('weight','target'):
            xx=upload(torch.ones(2,3,dtype=dtype,requires_grad=True));tt=upload(torch.tensor([0,1]));ww=upload(torch.ones(3,dtype=dtype))
            output=F.cross_entropy(xx,tt,weight=ww)
            with torch.no_grad():(ww if change=='weight' else tt).add_(1)
            try:output.backward()
            except RuntimeError as error:assert 'version' in str(error) or 'inplace' in str(error)
            else:raise AssertionError('missing saved tensor version guard')
        checks.append(dict(name='labels-and-saved-versions',invalid_int64=True,weight_version=True,target_version=True))
    def versions():
        mismatches=[]
        scenarios=((False,False,'none'),(True,True,'none'),(False,True,'target'),
                   (False,False,'target'),(True,True,'input'),(False,True,'input'),
                   (False,True,'weight'),(False,True,'unrelated'))
        def version_error(loss,*,retain_graph=False):
            try:loss.backward(retain_graph=retain_graph)
            except RuntimeError as error:
                if 'version' in str(error) or 'inplace' in str(error):return True
                raise
            return False
        def function(kind,x,t,w):
            if kind=='native2d':return A.nll_loss2d_forward(x,t,w,1,-100)[0]
            if kind=='nll':return F.nll_loss(x,t,weight=w)
            return F.cross_entropy(x,t,weight=w,label_smoothing=.2 if kind=='smooth' else 0.)
        for shape in ((2,3),(2,3,2),(2,3,2,2),(2,3,2,2,2),(2,3,33,365)):
            for kind in ('ce','smooth','nll','native2d'):
                if kind=='native2d' and len(shape)!=4:continue
                selected=scenarios if prod(shape)<=65536 else ((False,True,'none'),(False,True,'target'),(False,True,'input'))
                for strided_x,strided_t,change in selected:
                    size=(*shape[:-1],shape[-1]*(2 if strided_x else 1))
                    hx=torch.randn(size,dtype=dtype);dx=upload(hx)
                    x=(hx[...,::2] if strided_x else hx).detach().requires_grad_()
                    value=(dx[...,::2] if strided_x else dx).detach().requires_grad_()
                    target_shape=(shape[0],*shape[2:]);size=(*target_shape[:-1],target_shape[-1]*(2 if strided_t else 1))
                    ht=(torch.arange(prod(size)).reshape(size)%3).long();dt=upload(ht)
                    target=ht[...,::2] if strided_t else ht;labels=dt[...,::2] if strided_t else dt
                    hw=torch.tensor([.5,1.25,2.],dtype=dtype);dw=upload(hw)
                    expected=function(kind,x,target,hw);before=npu.stats()['npu_calls']
                    actual=function(kind,value,labels,dw);close(actual,expected)
                    if change!='none':
                        a,b=((ht.clone(),dt.clone()) if change=='unrelated' else
                             {'target':(ht,dt),'input':(hx,dx),'weight':(hw,dw)}[change])
                        with torch.no_grad():a.add_(0);b.add_(0)
                    wanted=version_error(expected);got=version_error(actual)
                    calls=npu.stats()['npu_calls']-before
                    name=f'{kind}-{shape}-x{strided_x}-t{strided_t}-{change}'
                    if wanted!=got:mismatches.append(name)
                    elif not wanted:
                        close(value.grad,x.grad)
                        a,b=value.grad.detach().cpu().double(),x.grad.double()
                        relative=float((a-b).norm()/b.norm().clamp_min(1e-30))
                        assert relative<(.01 if dtype==torch.float16 else 1e-4),(name,relative)
                    checks.append(dict(name=name,cpu_version_error=wanted,npu_version_error=got,npu_calls=calls))
        for kind in ('ce','nll','native2d'):
            hx=torch.randn(2,3,2,2,dtype=dtype);dx=upload(hx).requires_grad_();hx.requires_grad_()
            ht=torch.zeros(2,2,4,dtype=torch.int64);dt=upload(ht);hw=torch.ones(3,dtype=dtype);dw=upload(hw)
            expected=function(kind,hx,ht[...,::2],hw);actual=function(kind,dx,dt[...,::2],dw)
            assert not version_error(expected,retain_graph=True) and not version_error(actual,retain_graph=True)
            close(dx.grad,hx.grad)
            with torch.no_grad():ht.add_(0);dt.add_(0)
            wanted=version_error(expected);got=version_error(actual)
            name=f'{kind}-retained-graph'
            if wanted!=got:mismatches.append(name)
            checks.append(dict(name=name,cpu_version_error=wanted,npu_version_error=got))
            with torch.no_grad():
                expected=function(kind,hx,ht[...,::2],hw);actual=function(kind,dx,dt[...,::2],dw)
                assert not actual.requires_grad;close(actual,expected)
            checks.append(dict(name=kind+'-no-grad',requires_grad=False))
        print('version checks',len(checks),'mismatches',len(mismatches),mismatches[:8],flush=True)
        assert not mismatches,mismatches
    def training():
        cpu=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Tanh(),torch.nn.Linear(8,3)).to(dtype)
        model=copy.deepcopy(cpu).to('intel_npu');x=torch.randn(3,4,dtype=dtype)*.2;dx=upload(x)
        t=torch.tensor([0,1,2]);w=torch.tensor([.5,2.,1.]);soft=torch.rand(3,3);reg=torch.tensor([[.1,.2,.3]])
        dt,dw,ds,dr=(upload(v) for v in (t,w,soft,reg));losses=[];graphs=[];l2=[]
        opts=[torch.optim.SGD(m.parameters(),lr=.05,foreach=False,fused=False) for m in (cpu,model)]
        def loss(z,t,w,s,r):return F.cross_entropy(z.float(),t,weight=w,label_smoothing=.2)+.1*F.cross_entropy(z.float(),s,weight=w)+.2*F.mse_loss(z.float(),r.expand(3,3))
        for step in range(3):
            for opt in opts:opt.zero_grad(set_to_none=True)
            expected=loss(cpu(x),t,w,soft,reg);before=npu.stats();actual=loss(model(dx),dt,dw,ds,dr);actual.backward()
            after=npu.stats();assert after['host_download_copies']-before['host_download_copies']==1
            assert after['host_device_copies']==before['host_device_copies'];expected.backward();close(actual,expected)
            ag=[];bg=[]
            for a,b in zip(model.parameters(),cpu.parameters()):
                close(a.grad,b.grad);ag.append(a.grad.detach().cpu().reshape(-1));bg.append(b.grad.reshape(-1))
            ag,bg=torch.cat(ag).double(),torch.cat(bg).double();error=float((ag-bg).norm()/bg.norm().clamp_min(1e-30));l2.append(error)
            assert error<=(.01 if dtype==torch.float16 else 1e-4),error
            for opt in opts:opt.step()
            for a,b in zip(model.parameters(),cpu.parameters()):close(a,b)
            losses.append(actual.item());graphs.append(npu.stats()['cached_sessions']);print('step',step+1,losses[-1],flush=True)
        assert losses[-1]<losses[0] and graphs[1:]==graphs[:1]*2
        checks.append(dict(name='weighted-soft-regression-training',steps=3,parameters=sum(p.numel() for p in cpu.parameters()),
                           losses=losses,gradient_relative_l2=l2,cached_graphs=graphs))
    row=dict(name=case)
    try:locals()[case]();row['status']='passed'
    except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
    print(case,precision,row['status'],row.get('error','')[:800],flush=True)
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes'))
    root=Path(__file__).resolve().parent
    sources=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_loss_extended_demo.py','test_torch_npu_losses.py',
             'npunlock/tests/fixtures/npu4000/resident/manifest.json')
    result=dict(status='passed' if healthy and row['status']=='passed' else 'failed',case=case,precision=precision,
                result=row,checks=checks,stats=before,shutdown=final,max_absolute_error=max(errors,default=0.),
                elapsed_seconds=time.perf_counter()-started,
                source_sha256={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources})
    target=root/'npunlock-probe/research/results/loss-extended';target.mkdir(exist_ok=True)
    (target/f'{case}-{precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('hard','soft','mse','native','contracts','versions','training','tiled','tall'),required=True)
    parser.add_argument('--precision',choices=('float16','float32'),default='float32')
    args=parser.parse_args();raise SystemExit(execute(args.case,args.precision))

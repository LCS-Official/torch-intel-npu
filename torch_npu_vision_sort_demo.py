"""Independent CPU / real-NPU checks for vision and floating sorting."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/vision-sort'


def execute(case,precision):
    import torch
    import torch.nn.functional as F
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False;torch.manual_seed(823);dtype=getattr(torch,precision)
    def upload(x):return x.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)
    def close(a,b,exact=False):
        torch.testing.assert_close(a.detach().cpu(),b.detach().cpu(),equal_nan=True,
            rtol=0 if exact else .006 if dtype==torch.float16 else 8e-5,
            atol=0 if exact else .001 if dtype==torch.float16 else 1e-5)
    def check(function,host,exact=False):
        x=upload(host)
        actual=function(x);expected=function(host)
        if isinstance(actual,tuple):actual=actual[0];expected=expected[0]
        seed=torch.randn_like(expected);actual.backward(upload(seed));expected.backward(seed)
        close(actual,expected,exact);close(x.grad,host.grad,exact)
        return actual

    def sorting():
        count=0
        for shape in ((),(9,),(3,7),(2,3,5),(2,2,3,5)):
            for axis in range(len(shape) or 1):
                for descending in (False,True):
                    host=torch.randint(-2,3,shape).to(dtype).requires_grad_();x=upload(host)
                    actual,ids=torch.sort(x,dim=axis,descending=descending,stable=True)
                    expected,ref=torch.sort(host,dim=axis,descending=descending,stable=True)
                    seed=torch.randn_like(expected);actual.backward(upload(seed));expected.backward(seed)
                    close(actual,expected,True);close(ids,ref,True);close(x.grad,host.grad,True)
                    close(torch.argsort(x,dim=axis,descending=descending,stable=True),ref,True)
                    with torch.no_grad():
                        v=torch.empty_like(actual);i=torch.empty_like(ids)
                        result=torch.sort(x,dim=axis,descending=descending,stable=True,out=(v,i))
                        assert result.values is v and result.indices is i
                        close(v,expected,True);close(i,ref,True)
                        assert torch.argsort(x,dim=axis,stable=True,descending=descending,out=i) is i
                    count+=1
        for values in ([float('nan'),1.,float('nan'),-float('inf'),float('inf')],
                       [0.,-0.,0.,-0.],[2**-149,-2**-149,0.,2**-148],[-1.,-1.,-2.,-2.]):
            host=torch.tensor(values,dtype=dtype)
            for desc in (False,True):
                a,i=torch.sort(upload(host),stable=True,descending=desc);b,j=torch.sort(host,stable=True,descending=desc)
                close(a,b,True);close(i,j,True)
                raw=torch.int16 if dtype==torch.float16 else torch.int32
                torch.testing.assert_close(a.cpu().view(raw),b.view(raw),rtol=0,atol=0)
        host=torch.randn(3,12,dtype=dtype,requires_grad=True);x=upload(host)
        a,i=torch.sort(x[:,::2],dim=0,descending=True);b,j=torch.sort(host[:,::2],dim=0,descending=True)
        a.sum().backward();b.sum().backward();close(a,b,True);close(i,j,True);close(x.grad,host.grad,True)
        host=torch.randn(2,257,dtype=dtype,requires_grad=True);x=upload(host)
        a,i=torch.topk(x,200);a.sum().backward()
        b,j=torch.sort(host,descending=True,stable=True);b[:,:200].sum().backward()
        close(a,b[:,:200],True);close(x.grad,host.grad,True)
        return dict(axis_cases=count,stable_ties_nan_zero_bits=True,partial_view=True,large_k=200,paired_out=True)

    def extrema():
        count=0
        for shape in ((),(7,),(2,4),(2,3,4)):
            for axis in range(len(shape) or 1):
                for kind in ('max','min'):
                    for keep in (False,True):
                        host=torch.randint(-2,3,shape).to(dtype).requires_grad_();x=upload(host)
                        a,i=getattr(torch,kind)(x,axis,keep);b,j=getattr(torch,kind)(host,axis,keep)
                        seed=torch.randn_like(b);a.backward(upload(seed));b.backward(seed)
                        close(a,b,True);close(i,j,True);close(x.grad,host.grad,True);count+=1
                        with torch.no_grad():
                            v=torch.empty_like(a);ids=torch.empty_like(i)
                            result=getattr(torch,kind)(x,axis,keep,out=(v,ids));assert result.values is v and result.indices is ids
                            close(v,b,True);close(ids,j,True)
        for kind in ('max','min'):
            host=torch.tensor([[float('nan'),2.,float('nan')],[1.,1.,1.]],dtype=dtype,requires_grad=True);x=upload(host)
            a,i=getattr(torch,kind)(x,1);b,j=getattr(torch,kind)(host,1)
            a.sum().backward();b.sum().backward();close(a,b,True);close(i,j,True);close(x.grad,host.grad,True)
            host=torch.tensor([1.,1.,2.,2.],dtype=dtype,requires_grad=True);check(lambda x:getattr(torch,kind)(x),host,True)
            host=torch.tensor([float('nan'),2.,float('nan')],dtype=dtype,requires_grad=True);check(lambda x:getattr(torch,kind)(x),host,True)
        return dict(axis_cases=count,first_tie_nan=True,global_shared_gradient=True)

    def wide():
        host=torch.randint(-100,101,(21847,)).to(dtype).requires_grad_();x=upload(host)
        a,i=torch.sort(x,descending=True,stable=True);b,j=torch.sort(host,descending=True,stable=True)
        seed=torch.arange(host.numel()).remainder(11).to(dtype);a.backward(upload(seed));b.backward(seed)
        close(a,b,True);close(i,j,True);close(x.grad,host.grad,True)
        return dict(length=host.numel(),bitonic_cross_carrier=True,stable=True)

    def interpolation():
        count=0
        for shape,modes,output in (((2,2,5),('nearest','nearest-exact','linear'),(8,)),
                                   ((2,2,4,5),('nearest','nearest-exact','bilinear','bicubic'),(7,3))):
            for mode in modes:
                for align in ((False,True) if mode in ('linear','bilinear','bicubic') else (None,)):
                    for scale in (False,True):
                        options=dict(mode=mode)
                        if align is not None:options['align_corners']=align
                        if scale:options['scale_factor']=(1.7,) if len(shape)==3 else (1.6,.7)
                        else:options['size']=output
                        host=(torch.randn(shape,dtype=dtype)/4).requires_grad_()
                        check(lambda x:F.interpolate(x,**options),host);count+=1
        host=torch.randn(1,2,5,10,dtype=dtype,requires_grad=True);x=upload(host)
        a=F.interpolate(x[:,:,:,::2],size=(3,8),mode='bilinear',align_corners=False);b=F.interpolate(host[:,:,:,::2],size=(3,8),mode='bilinear',align_corners=False)
        a.sum().backward();b.sum().backward();close(a,b);close(x.grad,host.grad)
        host=torch.randn(1,1,1,1,dtype=dtype,requires_grad=True);check(lambda x:F.interpolate(x,size=(5,7),mode='bicubic',align_corners=False),host)
        for size in ((4,5),(1,1)):
            host=torch.randn(1,1,4,5,dtype=dtype,requires_grad=True);check(lambda x:F.interpolate(x,size=size,mode='bilinear',align_corners=True),host)
        for shape,modes in (((1,1,4),('nearest','nearest-exact','linear')),((1,1,4,4),('nearest','nearest-exact','bilinear','bicubic'))):
            for mode in modes:
                host=torch.randn(shape,dtype=dtype,requires_grad=True)
                options=dict(mode=mode,scale_factor=1.24)
                if mode in ('linear','bilinear','bicubic'):options['align_corners']=False
                if (len(shape)==3 and mode=='nearest') or mode=='bicubic':
                    x=upload(host);before=npu.stats()['npu_calls']
                    try:F.interpolate(x,**options)
                    except ValueError as e:assert 'legacy CPU scale shortcut' in str(e)
                    else:raise AssertionError('Inconsistent legacy interpolation training accepted')
                    assert npu.stats()['npu_calls']==before
                    with torch.no_grad():close(F.interpolate(x,**options),F.interpolate(host,**options))
                    options['recompute_scale_factor']=True
                check(lambda x:F.interpolate(x,**options),host)
        x=upload(torch.randn(1,1,3,4,dtype=dtype));before=npu.stats()
        result=F.interpolate(x,size=(5,7),mode='bilinear',align_corners=False);after=npu.stats()
        assert after['host_download_copies']==before['host_download_copies'] and after['host_device_copies']==before['host_device_copies']
        with torch.no_grad():
            out=torch.empty_like(result);actual=torch.ops.aten.upsample_bilinear2d.out(x,[5,7],False,None,None,out=out)
            assert actual is out;close(actual,result,True)
            grad=torch.randn_like(result);destination=torch.empty_like(x)
            actual=torch.ops.aten.upsample_bilinear2d_backward.grad_input(grad,[5,7],list(x.shape),False,None,None,grad_input=destination)
            assert actual is destination
            with torch.enable_grad():
                ref=x.cpu().requires_grad_();F.interpolate(ref,size=(5,7),mode='bilinear',align_corners=False).backward(grad.cpu())
            close(destination,ref.grad)
        return dict(cases=count,partial_view=True,degenerate_and_identity=True,native_out_backward=True,no_payload_download=True)

    def normalization():
        count=0
        for shape in ((2,6),(2,6,5),(2,6,3,4)):
            for groups in (1,2,6):
                host=(torch.randn(shape,dtype=dtype)/3).requires_grad_();x=upload(host)
                cpu=torch.nn.GroupNorm(groups,6).to(dtype);net=copy.deepcopy(cpu).to('intel_npu')
                a=net(x);b=cpu(host);seed=torch.randn_like(b);a.backward(upload(seed));b.backward(seed)
                close(a,b);close(x.grad,host.grad)
                for p,q in zip(net.parameters(),cpu.parameters()):close(p.grad,q.grad)
                count+=1
        for shape,klass in (((2,3,7),torch.nn.InstanceNorm1d),((2,3,4,5),torch.nn.InstanceNorm2d),((3,4,5),torch.nn.InstanceNorm2d)):
            for tracking in (False,True):
                host=torch.randn(shape,dtype=dtype,requires_grad=True);x=upload(host)
                cpu=klass(3,affine=True,track_running_stats=tracking).to(dtype);net=copy.deepcopy(cpu).to('intel_npu')
                for training in (True,False):
                    cpu.train(training);net.train(training);host.grad=None;x.grad=None;cpu.zero_grad();net.zero_grad()
                    a=net(x);b=cpu(host);seed=torch.randn_like(b);a.backward(upload(seed));b.backward(seed)
                    close(a,b);close(x.grad,host.grad)
                    for (name,p),q in zip(net.named_parameters(),cpu.parameters()):
                        try:close(p.grad,q.grad)
                        except AssertionError as e:
                            raise AssertionError(f'InstanceNorm {shape=} {tracking=} {training=} {name=}: NPU={p.grad.cpu().tolist()} CPU={q.grad.tolist()}') from e
                    if tracking:close(net.running_mean,cpu.running_mean);close(net.running_var,cpu.running_var)
                    count+=1
        host=torch.randn(2,4,3,4,dtype=dtype,requires_grad=True);x=upload(host)
        a=F.group_norm(x,2);b=F.group_norm(host,2);a.sum().backward();b.sum().backward();close(a,b);close(x.grad,host.grad)
        if dtype==torch.float16:
            host=torch.randn(2,4,3,4,dtype=dtype,requires_grad=True);x=upload(host)
            cpu=torch.nn.GroupNorm(2,4);net=copy.deepcopy(cpu).to('intel_npu')
            a=net(x);b=cpu(host);seed=torch.randn_like(b);a.backward(upload(seed));b.backward(seed)
            close(a,b);close(x.grad,host.grad)
            for p,q in zip(net.parameters(),cpu.parameters()):close(p.grad,q.grad)
        return dict(module_cases=count,affine_gradients=True,running_statistics=True,eval_and_no_affine=True,mixed_affine=dtype=='float16')

    def transposed():
        count=0
        for dims in (1,2):
            for groups,stride,padding,dilation,output_padding in ((1,1,0,1,0),(2,2,1,1,1),(2,1,2,3,2)):
                shape=(2,4,4) if dims==1 else (2,4,3,4)
                cpu=getattr(torch.nn,f'ConvTranspose{dims}d')(4,6,3,stride=stride,padding=padding,dilation=dilation,output_padding=output_padding,groups=groups).to(dtype)
                net=copy.deepcopy(cpu).to('intel_npu');host=(torch.randn(shape,dtype=dtype)/3).requires_grad_();x=upload(host)
                a=net(x);b=cpu(host);seed=torch.randn_like(b);a.backward(upload(seed));b.backward(seed)
                close(a,b);close(x.grad,host.grad)
                for p,q in zip(net.parameters(),cpu.parameters()):close(p.grad,q.grad)
                count+=1
        cpu=torch.nn.ConvTranspose2d(2,2,(2,3),stride=(2,1),padding=(0,1),output_padding=(1,0),bias=False).to(dtype)
        net=copy.deepcopy(cpu).to('intel_npu');host=torch.randn(1,2,3,4,dtype=dtype,requires_grad=True);x=upload(host)
        a=net(x);b=cpu(host);a.sum().backward();b.sum().backward();close(a,b);close(x.grad,host.grad);close(net.weight.grad,cpu.weight.grad)
        return dict(module_cases=count+1,groups=True,dilation=True,output_padding_greater_than_stride=True,asymmetric_kernel=True)

    def tiled():
        host=torch.randint(-20,21,(257,257)).to(dtype).requires_grad_();x=upload(host)
        a,i=torch.sort(x,dim=1,stable=True);b,j=torch.sort(host,dim=1,stable=True)
        seed=torch.randn_like(b);a.backward(upload(seed));b.backward(seed)
        close(a,b,True);close(i,j,True);close(x.grad,host.grad,True)
        host=torch.randn(1,1,260,260,dtype=dtype,requires_grad=True)
        check(lambda x:F.interpolate(x,size=(257,257),mode='bilinear',align_corners=False),host)
        cpu=torch.nn.ConvTranspose2d(1,1,2,stride=2).to(dtype);net=copy.deepcopy(cpu).to('intel_npu')
        host=torch.randn(1,1,130,130,dtype=dtype,requires_grad=True);x=upload(host)
        a=net(x);b=cpu(host);seed=torch.randn_like(b);a.backward(upload(seed));b.backward(seed)
        close(a,b);close(x.grad,host.grad)
        for p,q in zip(net.parameters(),cpu.parameters()):close(p.grad,q.grad)
        return dict(sort_elements=257*257,interpolation_input=260*260,transpose_output=260*260,forward_backward=True)

    def contracts():
        count=0
        def reject(action):
            nonlocal count
            before=npu.stats()['npu_calls']
            try:action()
            except (ValueError,IndexError,RuntimeError,TypeError):pass
            else:raise AssertionError('Invalid operator contract accepted')
            assert npu.stats()['npu_calls']==before;count+=1
        x=upload(torch.randn(2,4,dtype=dtype));ids=torch.empty(2,4,dtype=torch.int64,device='intel_npu')
        reject(lambda:torch.sort(x,dim=2));reject(lambda:torch.sort(x,out=(x,ids)))
        reject(lambda:torch.topk(x,-1));reject(lambda:torch.topk(x,5));reject(lambda:torch.sort(ids,dim=2))
        wide=torch.empty(npu.MAX_TENSOR_ELEMENTS+1,dtype=dtype,device='intel_npu');reject(lambda:torch.sort(wide))
        image=upload(torch.randn(2,4,3,4,dtype=dtype));w=upload(torch.randn(4,2,3,3,dtype=dtype));bias=upload(torch.randn(4,dtype=dtype))
        reject(lambda:F.interpolate(image,size=(0,2),mode='bilinear',align_corners=False))
        reject(lambda:F.group_norm(image,3));reject(lambda:F.group_norm(image,2,eps=0))
        reject(lambda:F.instance_norm(image,running_mean=bias,running_var=bias))
        reject(lambda:F.conv_transpose2d(image,w,stride=1,output_padding=1));reject(lambda:F.conv_transpose2d(image,w,groups=3))
        image.requires_grad_();y,i=torch.sort(image,stable=True)
        with torch.no_grad():i.fill_(0)
        try:y.sum().backward()
        except RuntimeError as e:assert 'version' in str(e) or 'inplace' in str(e)
        else:raise AssertionError('Mutated sorted indices accepted')
        a=upload(torch.randn(2,4,3,4,dtype=dtype,requires_grad=True));weight=upload(torch.ones(4,dtype=dtype,requires_grad=True))
        y=F.group_norm(a,2,weight)
        with torch.no_grad():weight.add_(1.)
        try:y.sum().backward()
        except RuntimeError as e:assert 'version' in str(e) or 'inplace' in str(e)
        else:raise AssertionError('Mutated norm weight accepted')
        host=torch.tensor([-1.,2.,-3.,4.],dtype=dtype).view(1,1,2,2).requires_grad_();x=upload(host);version=x._version
        a=F.relu(F.interpolate(x,size=(2,2),mode='bilinear',align_corners=False),inplace=True)
        b=F.relu(F.interpolate(host,size=(2,2),mode='bilinear',align_corners=False),inplace=True)
        a.sum().backward();b.sum().backward();close(x,host,True);close(a,b,True);close(x.grad,host.grad,True);assert x._version==version
        for layer,shape in ((torch.nn.Conv1d(2,2,3,padding=1),(2,2,5)),(torch.nn.ConvTranspose1d(2,2,2,stride=2),(2,2,4)),
                            (torch.nn.ConvTranspose2d(2,2,2,stride=2),(2,2,3,4)),(torch.nn.GroupNorm(2,4),(2,4,3,4)),
                            (torch.nn.InstanceNorm2d(2,affine=True),(2,2,3,4))):
            cpu=torch.nn.Sequential(layer,torch.nn.ReLU(inplace=True)).to(dtype);net=copy.deepcopy(cpu).to('intel_npu')
            host=torch.randn(shape,dtype=dtype,requires_grad=True);x=upload(host);a=net(x);b=cpu(host)
            a.sum().backward();b.sum().backward();close(a,b);close(x.grad,host.grad)
            for p,q in zip(net.parameters(),cpu.parameters()):close(p.grad,q.grad)
        return dict(rejected_before_submit=count,index_and_parameter_version_guards=True)

    def training(amp=False,sorting=True):
        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__();self.conv=torch.nn.Conv2d(1,4,3,padding=1);self.norm=torch.nn.GroupNorm(2,4)
                self.up=torch.nn.ConvTranspose2d(4,2,2,stride=2);self.instance=torch.nn.InstanceNorm2d(2,affine=True,track_running_stats=True)
                self.head=torch.nn.Linear(136,3)
            def forward(self,x,route=None):
                x=F.relu(self.norm(self.conv(x)),inplace=True);x=F.max_pool2d(x,2)
                x=self.up(x);x=F.interpolate(x,size=(9,9),mode='bilinear',align_corners=False)
                x=self.instance(x).flatten(1)
                self.pre_sort=x.detach()
                if sorting:
                    ordered,self.sort_ids=torch.sort(x,dim=1,stable=True)
                    x=ordered if route is None else x.gather(1,route)
                x=x[:,26:]
                return self.head(x)
        model_dtype=torch.float32 if amp else dtype;cpu=Vision().to(model_dtype);net=copy.deepcopy(cpu).to('intel_npu')
        host=torch.randn(2,1,6,6,dtype=model_dtype)/3;data=upload(host);labels=torch.tensor([0,2]);target=upload(labels)
        optimizers=[torch.optim.SGD(m.parameters(),lr=.01,momentum=.5,foreach=False,fused=False) for m in (net,cpu)]
        scalers=[torch.intel_npu.amp.GradScaler(init_scale=8,growth_interval=2),torch.amp.GradScaler('cpu',init_scale=8,growth_interval=2)] if amp else None
        tolerance=dict(rtol=.012,atol=.002) if amp or dtype==torch.float16 else dict(rtol=.0002,atol=2e-5)
        losses=[];errors=[];payload=[]
        for step in range(3):
            pair=[]
            for j,(m,opt,x,t) in enumerate(zip((net,cpu),optimizers,(data,host),(target,labels))):
                opt.zero_grad(set_to_none=True);before=npu.stats()
                with torch.autocast('intel_npu' if j==0 else 'cpu',dtype=torch.float16,enabled=amp):loss=F.cross_entropy(m(x).float(),t)
                if amp:scalers[j].scale(loss).backward();scalers[j].unscale_(opt)
                else:loss.backward()
                if j==0:
                    after=npu.stats();assert after['host_device_copies']==before['host_device_copies']
                    # Reads are scalar classification/index validation and scaler flags.
                    assert after['host_download_copies']-before['host_download_copies']==2+int(sorting)+int(amp)
                    payload.append(dict(uploads=after['host_upload_copies']-before['host_upload_copies'],flag_reads=after['host_download_copies']-before['host_download_copies']))
                pair.append(loss)
            torch.testing.assert_close(pair[0].detach().cpu(),pair[1].detach(),**tolerance);delta=[]
            for (name,p),q in zip(net.named_parameters(),cpu.parameters()):
                try:torch.testing.assert_close(p.grad.cpu(),q.grad,**tolerance)
                except AssertionError as e:
                    diagnostic=dict(step=step,parameter=name,sorting=sorting,activation_max_error=float((net.pre_sort.cpu()-cpu.pre_sort).abs().max()))
                    if sorting:
                        route=net.sort_ids.cpu();diagnostic['different_indices']=int((route!=cpu.sort_ids).sum())
                        if diagnostic['different_indices']:
                            # Diagnostic only: hold discrete choices fixed to isolate
                            # sensitivity from an incorrect continuous derivative.
                            cpu.zero_grad(set_to_none=True)
                            with torch.autocast('cpu',dtype=torch.float16,enabled=amp):reference_loss=F.cross_entropy(cpu(host,route=route).float(),labels)
                            scale=scalers[1].get_scale() if amp else 1.;(reference_loss*scale).backward();fixed=[]
                            for left,right in zip(net.parameters(),cpu.parameters()):
                                expected=right.grad/scale;torch.testing.assert_close(left.grad.cpu(),expected,**tolerance)
                                fixed.append(float((left.grad.cpu()-expected).abs().max()))
                            diagnostic.update(fixed_route_reference_passed=True,fixed_route_max_gradient_error=max(fixed),
                                strict_failure=str(e),npu_loss=float(pair[0].detach().cpu()),cpu_loss=float(pair[1].detach()),tolerance=tolerance)
                            return dict(strict_cpu_passed=False,steps_completed=step,sorting=sorting,amp=amp,routing_boundary=diagnostic)
                    raise AssertionError(str(diagnostic)) from e
                delta.append(float((p.grad.cpu()-q.grad).abs().max()))
            for j,opt in enumerate(optimizers):
                if amp:scalers[j].step(opt);scalers[j].update()
                else:opt.step()
            for p,q in zip(net.parameters(),cpu.parameters()):torch.testing.assert_close(p.detach().cpu(),q.detach(),**tolerance)
            if amp:assert scalers[0].state_dict()==scalers[1].state_dict()
            close(net.instance.running_mean,cpu.instance.running_mean);close(net.instance.running_var,cpu.instance.running_var)
            losses.append(float(pair[0].detach().cpu()));errors.append(max(delta))
        return dict(steps=3,amp=amp,sorting=sorting,losses=losses,max_gradient_error=errors,tolerance=tolerance,device_copy_accounting=payload)

    actions=dict(sort=sorting,extrema=extrema,wide=wide,interpolation=interpolation,normalization=normalization,transpose=transposed,tiled=tiled,
                 contracts=contracts,training=training,amp=lambda:training(True))
    actions.update({'vision-training':lambda:training(False,False),'vision-amp':lambda:training(True,False)})
    before=npu.stats()
    try:detail=actions[case]();status='passed_with_limitations' if detail.get('strict_cpu_passed') is False else 'passed'
    except Exception as error:status='failed';detail=dict(error=str(error),traceback=traceback.format_exc())
    gc.collect();after=npu.stats()
    if after['live_buffers']-after['cached_scalar_buffers']!=before['live_buffers']-before['cached_scalar_buffers']:
        status='failed';detail['lifecycle_error']='Device storage leak'
    npu.shutdown();final=npu.stats()
    if final['context_failed'] or any(final[k] for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes')):status='failed'
    names=['torch_intel_npu.py','torch_npu_autograd.py','torch_npu_sort.py','torch_npu_geometry.py','torch_npu_vision_sort_demo.py',
           'npunlock/tests/fixtures/npu4000/resident_sort.c','npunlock/tests/fixtures/npu4000/resident/manifest.json']
    if (ROOT/'torch_npu_vision.py').exists():names.append('torch_npu_vision.py')
    print(case,precision,status,detail.get('error','')[:1500],flush=True)
    if status!='passed':print(detail,flush=True)
    return dict(status=status,case=case,precision=precision,details=detail,npu_calls=after['npu_calls']-before['npu_calls'],final_stats=final,
                source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',required=True,
        choices=('sort','extrema','wide','interpolation','normalization','transpose','contracts','tiled','training','amp','vision-training','vision-amp'))
    parser.add_argument('--precision',choices=('float16','float32'),default='float32');parser.add_argument('--label',default='after');args=parser.parse_args()
    result=execute(args.case,args.precision);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{args.label}-{args.case}-{args.precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(result['status'] not in ('passed','passed_with_limitations'))

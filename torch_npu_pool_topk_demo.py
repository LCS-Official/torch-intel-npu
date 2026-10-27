"""MaxPool1d/2d, differentiable in-place ReLU and topk on the real Intel NPU."""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/pool-topk'


def model():
    import torch
    class CNN(torch.nn.Module):
        def __init__(self):
            super().__init__();self.conv=torch.nn.Conv2d(1,2,3,padding=1);self.relu=torch.nn.ReLU(inplace=True)
            self.pool=torch.nn.MaxPool2d(2);self.head=torch.nn.Linear(4,3)
        def forward(self,x):
            h=self.relu(self.conv(x));h=self.relu(h+.125)
            return self.head(torch.topk(self.pool(h).flatten(1),4,dim=1).values)
    return CNN()


def execute(case,precision):
    import torch
    import torch.nn.functional as F
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False;torch.manual_seed(607);dtype=getattr(torch,precision)
    def upload(x):return x.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)
    def close(a,b,exact=False):
        torch.testing.assert_close(a.detach().cpu(),b.detach().cpu(),equal_nan=True,
            rtol=0 if exact else .006 if dtype==torch.float16 else 6e-5,
            atol=0 if exact else .0007 if dtype==torch.float16 else 8e-6)
    def resident(before,flags=0,metadata=0):
        s=npu.stats();assert s['host_download_copies']-before['host_download_copies']==flags
        assert s['host_device_copies']==before['host_device_copies']
        assert s['host_upload_copies']-before['host_upload_copies']==s['cached_scalar_buffers']-before['cached_scalar_buffers']+metadata

    def relu():
        host=torch.tensor([-float('inf'),-1.,-0.,0.,2**-24,1.,float('inf'),float('nan')],dtype=dtype,requires_grad=True)
        x=upload(host);y=x*1.;reference=host*1.;before=npu.stats();version=y._version
        actual=F.relu(y,inplace=True);assert actual is y and y._version==version+1;actual.sum().backward();resident(before)
        F.relu(reference,inplace=True).sum().backward();close(actual,reference,True);close(x.grad,host.grad,True)
        x=upload(torch.tensor([-1.,2.],dtype=dtype,requires_grad=True));before=npu.stats()
        try:x.relu_()
        except RuntimeError as e:assert 'leaf' in str(e)
        else:raise AssertionError('Leaf mutation accepted')
        assert npu.stats()['npu_calls']==before['npu_calls'];close(x,torch.tensor([-1.,2.],dtype=dtype),True)
        view=(x*1.).view(2);before=npu.stats()
        try:view.relu_()
        except (ValueError,RuntimeError):pass
        else:raise AssertionError('Differentiable view mutation accepted')
        assert npu.stats()['npu_calls']==before['npu_calls']
        clone=view.clone();assert clone.relu_() is clone;clone.sum().backward();close(x.grad,torch.tensor([0.,1.],dtype=dtype),True)
        x=upload(torch.tensor([-1.,2.],dtype=dtype,requires_grad=True));saved=(x*x).sum()
        with torch.no_grad():assert x.relu_() is x
        try:saved.backward()
        except RuntimeError as e:assert 'version' in str(e) or 'inplace' in str(e)
        else:raise AssertionError('Saved input mutation not detected')
        with torch.inference_mode():
            value=torch.tensor([-1.,2.],dtype=dtype,device='intel_npu');assert value.relu_() is value
            close(value,torch.tensor([0.,2.],dtype=dtype),True)
        for cpu in (torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.ReLU(True)),
                    torch.nn.Sequential(torch.nn.Conv2d(1,2,3,padding=1),torch.nn.BatchNorm2d(2),torch.nn.ReLU(True))):
            cpu=cpu.to(dtype);net=copy.deepcopy(cpu).to('intel_npu');shape=(2,3) if isinstance(cpu[0],torch.nn.Linear) else (2,1,4,4)
            host=torch.randn(shape,dtype=dtype,requires_grad=True);x=upload(host)
            actual=net(x);actual.sum().backward();expected=cpu(host);expected.sum().backward();close(actual,expected);close(x.grad,host.grad)
            for p,q in zip(net.parameters(),cpu.parameters()):close(p.grad,q.grad)
        return dict(inplace_identity=True,version_guard=True,leaf_view_preflight=True,clone_escape=True,linear_conv_bn_modules=True)

    def topk():
        count=0
        for shape in ((),(7,),(2,5),(2,3,4),(2,2,3,4)):
            for axis in range(len(shape) or 1):
                width=shape[axis] if shape else 1;k=min(3,width)
                for largest in (False,True):
                    host=torch.randn(shape,dtype=dtype,requires_grad=True);x=upload(host);before=npu.stats()
                    actual,index=torch.topk(x,k,axis,largest,True);actual.sum().backward();resident(before,1)
                    expected,cpu_ids=torch.topk(host,k,axis,largest,True);expected.sum().backward()
                    close(actual,expected,True)
                    # CPU topk may choose other equal-valued indices; use actual indices for derivative certification.
                    ids=index.cpu();reference=torch.zeros_like(host)
                    if shape:reference.scatter_(axis,ids,1.)
                    else:reference.fill_(1.)
                    close(x.grad,reference,True)
                    with torch.no_grad():
                        v=torch.empty_like(actual);i=torch.empty_like(index)
                        result=torch.topk(x,k,axis,largest,out=(v,i));assert result.values is v and result.indices is i
                        close(v,actual,True);close(i,index,True)
                    count+=1
        host=torch.randn(3,10,dtype=dtype,requires_grad=True);x=upload(host)
        actual,ids=torch.topk(x[:,::2],2,dim=-1,sorted=False);actual.sum().backward()
        expected=host[:,::2].gather(-1,ids.cpu());expected.sum().backward();close(actual,expected,True);close(x.grad,host.grad,True)
        return dict(axis_cases=count,partial_view=True,out=True,tied_indices_cpu_order_not_promised=True)

    def pools():
        count=0
        for shape in ((2,5,6),(2,2,5,6)):
            for kernel,stride,pad,dilation,ceil in ((2,2,0,1,False),(3,2,1,1,True),((2,3),(1,2),(1,1),(2,1),False),(2,1,0,1,True)):
                host=torch.randn(shape,dtype=dtype,requires_grad=True);x=upload(host);before=npu.stats()
                actual,ids=F.max_pool2d(x,kernel,stride,pad,dilation,ceil,return_indices=True);actual.sum().backward()
                resident(before,1,metadata=host.numel()//(shape[-1]*shape[-2]))
                expected,cpu_ids=F.max_pool2d(host,kernel,stride,pad,dilation,ceil,return_indices=True);expected.sum().backward()
                close(actual,expected,True);close(ids,cpu_ids,True);close(x.grad,host.grad,True);count+=1
                if count==1:
                    with torch.no_grad():
                        out=torch.empty_like(actual);ind=torch.empty_like(ids)
                        a,b=torch.ops.aten.max_pool2d_with_indices.out(x,[2,2],[2,2],[0,0],[1,1],False,out=out,indices=ind)
                        assert a is out and b is ind;close(out,actual,True);close(ind,ids,True)
                        grad=torch.ones_like(actual);destination=torch.empty_like(x)
                        assert torch.ops.aten.max_pool2d_with_indices_backward.grad_input(grad,x,[2,2],[2,2],[0,0],[1,1],False,ind,grad_input=destination) is destination
                        close(destination,x.grad,True)
        for shape in ((2,7),(2,2,7)):
            for returns in (False,True):
                host=torch.randn(shape,dtype=dtype,requires_grad=True);x=upload(host)
                a=F.max_pool1d(x,3,2,1,ceil_mode=True,return_indices=returns);b=F.max_pool1d(host,3,2,1,ceil_mode=True,return_indices=returns)
                actual,expected=(a[0],b[0]) if returns else (a,b);actual.sum().backward();expected.sum().backward()
                close(actual,expected,True);close(x.grad,host.grad,True)
                if returns:close(a[1],b[1],True)
        return dict(pool2d_cases=count,pool1d_cases=4,dilation_ceil_padding=True,out_backward=True)

    def edges():
        host=torch.tensor([1.,2.],dtype=dtype,requires_grad=True);x=upload(host)
        seed=torch.tensor([-0.,1.],dtype=dtype);actual,_=torch.topk(x,2);actual.backward(upload(seed))
        expected,_=torch.topk(host,2);expected.backward(seed)
        raw=torch.int16 if dtype==torch.float16 else torch.int32
        torch.testing.assert_close(x.grad.cpu().view(raw),host.grad.view(raw),rtol=0,atol=0)
        for values in ([float('nan'),1.,float('nan'),2.],[0.,-0.,0.,-0.],[float('-inf')]*4,[2**-149,-2**-149,0.,2**-148]):
            host=torch.tensor(values,dtype=dtype,requires_grad=True).view(1,1,1,4);host.retain_grad();x=upload(host)
            actual,ids=F.max_pool2d(x,(1,4),return_indices=True);actual.sum().backward()
            expected,cpu_ids=F.max_pool2d(host,(1,4),return_indices=True);expected.sum().backward()
            close(actual,expected,True);close(ids,cpu_ids,True);close(x.grad,host.grad,True)
            flat=host.detach().view(-1)
            for largest in (False,True):
                a,i=torch.topk(upload(flat),3,largest=largest);b,_=torch.topk(flat,3,largest=largest)
                close(a,b,True);close(a,flat.gather(0,i.cpu()),True);assert len(set(i.cpu().tolist()))==3
        host=torch.zeros(1,1,5,5,dtype=dtype,requires_grad=True);host.data[0,0,2,2]=1.;x=upload(host)
        seed=torch.tensor([65504.,-65504.,.01,2**-24,1.,-1.,.03,2**-24,.01],dtype=dtype).view(1,1,3,3)
        actual=F.max_pool2d(x,3,1);actual.backward(upload(seed));expected=F.max_pool2d(host,3,1);expected.backward(seed)
        close(x.grad,host.grad,True)
        host=torch.randn(1,1,5,12,dtype=dtype,requires_grad=True);x=upload(host)
        actual=F.max_pool2d(x[:,:,:,::2],3,2,1,ceil_mode=True);actual.sum().backward()
        expected=F.max_pool2d(host[:,:,:,::2],3,2,1,ceil_mode=True);expected.sum().backward()
        close(actual,expected,True);close(x.grad,host.grad,True)
        x=upload(torch.arange(8,dtype=dtype).requires_grad_());a,i=torch.topk(x,3)
        with torch.no_grad():i.fill_(0)
        try:a.sum().backward()
        except RuntimeError as e:assert 'version' in str(e) or 'inplace' in str(e)
        else:raise AssertionError('Mutated topk indices accepted')
        return dict(last_pool_nan=True,topk_ties_valid=True,overlap_rounding=True,index_version=True,topk_negative_zero_gradient_bits=True)

    def tiled():
        host=torch.zeros(65537,dtype=dtype,requires_grad=True);host.data[0]=2;host.data[65530]=5;host.data[-1]=3;x=upload(host)
        before=npu.stats();a,i=torch.topk(x,3);a.sum().backward();resident(before,2)
        b,j=torch.topk(host,3);b.sum().backward();close(a,b,True);close(i,j,True);close(x.grad,host.grad,True)
        host=torch.randn(1,1,260,260,dtype=dtype,requires_grad=True);x=upload(host);before=npu.stats()
        a,i=F.max_pool2d(x,3,2,return_indices=True);a.sum().backward();resident(before,2,5)
        b,j=F.max_pool2d(host,3,2,return_indices=True);b.sum().backward();close(a,b,True);close(i,j,True);close(x.grad,host.grad,True)
        host=torch.arange(129,dtype=dtype);a,i=torch.topk(upload(host),128,largest=False);b,j=torch.topk(host,128,largest=False)
        close(a,b,True);close(i,j,True)
        return dict(long_axis=65537,pool_input=67600,topk_k128=True)

    def contracts():
        x=torch.ones(2,5,dtype=dtype,device='intel_npu');image=torch.ones(1,1,4,4,dtype=dtype,device='intel_npu')
        many=torch.empty(npu.MAX_TENSOR_ELEMENTS+1,1,dtype=dtype,device='intel_npu');full_ids=torch.empty(2,5,dtype=torch.int64,device='intel_npu')
        wrong=torch.empty(1,dtype=dtype,device='intel_npu');ids=torch.empty(1,dtype=torch.int64,device='intel_npu')
        actions=[lambda:torch.topk(x,-1),lambda:torch.topk(x,6),lambda:torch.topk(x,1,2),lambda:torch.topk(x,1,out=(wrong,ids)),
                 lambda:torch.topk(many,1,dim=1),lambda:torch.topk(many.view(-1),128),lambda:torch.topk(x,5,out=(x,full_ids)),
                 lambda:F.max_pool2d(image,0),lambda:F.max_pool2d(image,2,padding=2),
                 lambda:F.max_pool2d(image,2,dilation=0),lambda:torch.ops.aten.max_pool2d_with_indices.out(image,[2,2],out=wrong,indices=ids)]
        for op in actions:
            before=npu.stats()
            try:op()
            except (ValueError,IndexError,RuntimeError,NotImplementedError):pass
            else:raise AssertionError('Invalid parameters accepted')
            assert npu.stats()['npu_calls']==before['npu_calls']
        grad=torch.ones(1,1,2,2,dtype=dtype,device='intel_npu');bad=upload(torch.full((1,1,2,2),2**40,dtype=torch.int64))
        try:torch.ops.aten.max_pool2d_with_indices_backward(grad,image,[2,2],[2,2],[0,0],[1,1],False,bad)
        except IndexError:pass
        else:raise AssertionError('Malformed INT64 pool indices accepted')
        return dict(metadata_rejections=len(actions),invalid_device_index=True)

    def training(amp=False):
        from torch_npu_transformer_demo import cpu_tree
        host=model().to(torch.float32 if amp else dtype);net=copy.deepcopy(host).to('intel_npu')
        inputs=torch.randn(2,1,6,6).to(torch.float32 if amp else dtype)/4;labels=torch.tensor([0,2]);data,target=upload(inputs),upload(labels)
        optimizers=[torch.optim.SGD(m.parameters(),lr=.02,momentum=.5,foreach=False,fused=False) for m in (net,host)]
        scalers=[torch.intel_npu.amp.GradScaler(init_scale=8,growth_interval=2),torch.amp.GradScaler('cpu',init_scale=8,growth_interval=2)] if amp else None
        tolerance=dict(rtol=.01,atol=.001) if amp or dtype==torch.float16 else dict(rtol=6e-5,atol=8e-6)
        losses=[];errors=[];saved=io.BytesIO()
        for step in range(4):
            pair=[]
            for j,(m,opt,x,y) in enumerate(zip((net,host),optimizers,(data,inputs),(target,labels))):
                opt.zero_grad(set_to_none=True);before=npu.stats()
                with torch.autocast('intel_npu' if j==0 else 'cpu',dtype=torch.float16,enabled=amp):loss=F.cross_entropy(m(x).float(),y)
                if amp:scalers[j].scale(loss).backward();scalers[j].unscale_(opt)
                else:loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),.1,foreach=False)
                if j==0:resident(before,3+int(amp),4)
                pair.append(loss)
            torch.testing.assert_close(pair[0].detach().cpu(),pair[1].detach(),**tolerance);differences=[]
            for p,q in zip(net.parameters(),host.parameters()):
                torch.testing.assert_close(p.grad.cpu(),q.grad,**tolerance);differences.append(float((p.grad.cpu()-q.grad).abs().max()))
            for j,opt in enumerate(optimizers):
                if amp:scalers[j].step(opt);scalers[j].update()
                else:opt.step()
            for p,q in zip(net.parameters(),host.parameters()):torch.testing.assert_close(p.detach().cpu(),q.detach(),**tolerance)
            if amp:assert scalers[0].state_dict()==scalers[1].state_dict()
            losses.append(float(pair[0].detach().cpu()));errors.append(max(differences))
            if step==2:torch.save(dict(model=cpu_tree(net.state_dict()),optimizer=cpu_tree(optimizers[0].state_dict()),scaler=scalers[0].state_dict() if amp else None),saved)
        saved.seek(0);state=torch.load(saved,weights_only=True);restored=model().to(torch.float32 if amp else dtype).to('intel_npu');restored.load_state_dict(state['model'])
        opt=torch.optim.SGD(restored.parameters(),lr=.02,momentum=.5,foreach=False,fused=False);opt.load_state_dict(state['optimizer']);opt.zero_grad(set_to_none=True)
        if amp:scaler=torch.intel_npu.amp.GradScaler();scaler.load_state_dict(state['scaler'])
        with torch.autocast('intel_npu',dtype=torch.float16,enabled=amp):loss=F.cross_entropy(restored(data).float(),target)
        if amp:scaler.scale(loss).backward();scaler.unscale_(opt)
        else:loss.backward()
        torch.nn.utils.clip_grad_norm_(restored.parameters(),.1,foreach=False)
        if amp:scaler.step(opt);scaler.update();assert scaler.state_dict()==scalers[0].state_dict()
        else:opt.step()
        assert float(loss.detach().cpu())==losses[-1]
        for p,q in zip(net.parameters(),restored.parameters()):
            torch.testing.assert_close(p.detach().cpu(),q.detach().cpu(),rtol=0,atol=0)
            torch.testing.assert_close(optimizers[0].state[p]['momentum_buffer'].cpu(),opt.state[q]['momentum_buffer'].cpu(),rtol=0,atol=0)
        return dict(steps=4,amp=amp,losses=losses,max_gradient_error=errors,checkpoint_bitwise=True,tolerance=tolerance)

    actions=dict(relu=relu,topk=topk,pools=pools,edges=edges,tiled=tiled,contracts=contracts,training=training,amp=lambda:training(True))
    before=npu.stats()
    try:detail=actions[case]();status='passed'
    except Exception as error:status='failed';detail=dict(error=str(error),traceback=traceback.format_exc())
    gc.collect();after=npu.stats()
    if after['live_buffers']-after['cached_scalar_buffers']!=before['live_buffers']-before['cached_scalar_buffers']:status='failed';detail['lifecycle_error']='Device storage leak'
    npu.shutdown();final=npu.stats()
    if final['context_failed'] or any(final[k] for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes')):status='failed'
    names=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_geometry.py','torch_npu_pool_topk_demo.py',
           'npunlock/tests/fixtures/npu4000/resident_selection.c','npunlock/tests/fixtures/npu4000/resident/manifest.json')
    print(case,precision,status,detail.get('error','')[:1500],flush=True)
    return dict(status=status,case=case,precision=precision,details=detail,npu_calls=after['npu_calls']-before['npu_calls'],final_stats=final,
                source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=('relu','topk','pools','edges','tiled','contracts','training','amp'),required=True)
    parser.add_argument('--precision',choices=('float16','float32'),default='float32');parser.add_argument('--label',default='after');args=parser.parse_args()
    result=execute(args.case,args.precision);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{args.label}-{args.case}-{args.precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(result['status']!='passed')

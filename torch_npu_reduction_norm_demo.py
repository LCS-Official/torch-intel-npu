"""NPU arg/extreme reductions, vector norms and standard gradient-norm clipping."""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/reduction-norm'


def model():
    import torch
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.first=torch.nn.Linear(4,6);self.last=torch.nn.Linear(6,3)
        def forward(self,x):
            h=torch.tanh(self.first(x))
            h=h/(torch.linalg.vector_norm(h,dim=-1,keepdim=True)+.5)
            return self.last(h+.1*torch.amax(h,dim=-1,keepdim=True))
    return Network()


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False;torch.manual_seed(521)
    dtype=getattr(torch,precision)
    def upload(x):
        return x.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)
    def close(a,b,exact=False):
        torch.testing.assert_close(a.detach().cpu(),b.detach().cpu(),equal_nan=True,
            rtol=0 if exact else .006 if dtype==torch.float16 else 6e-5,
            atol=0 if exact else .0007 if dtype==torch.float16 else 8e-6)
    def resident(before,flags=0):
        s=npu.stats();assert s['host_download_copies']-before['host_download_copies']==flags
        assert s['host_device_copies']==before['host_device_copies']
        assert s['host_upload_copies']-before['host_upload_copies']==s['cached_scalar_buffers']-before['cached_scalar_buffers']

    def reductions():
        count=0
        for shape in ((),(7,),(3,5),(2,3,4),(2,2,3,4)):
            host=torch.randn(shape,dtype=dtype).requires_grad_();x=upload(host)
            for dim in (None,*range(-len(shape),0)):
                for keep in (False,True):
                    for op in (torch.argmax,torch.argmin):
                        before=npu.stats();actual=op(x,dim=dim,keepdim=keep);resident(before)
                        close(actual,op(host,dim=dim,keepdim=keep),True);assert actual.dtype==torch.int64 and not actual.requires_grad
                        with torch.no_grad():
                            out=torch.empty_like(actual);assert op(x,dim=dim,keepdim=keep,out=out) is out;close(out,actual,True)
                    for op in (torch.amax,torch.amin):
                        host.grad=x.grad=None;before=npu.stats()
                        actual=op(x,dim=dim,keepdim=keep);actual.sum().backward();resident(before)
                        expected=op(host,dim=dim,keepdim=keep);expected.sum().backward()
                        close(actual,expected,True);close(x.grad,host.grad,True)
                    count+=1
        host=torch.randn(2,3,4,dtype=dtype).requires_grad_();x=upload(host)
        for dim in ((0,2),(-1,-3),()):
            host.grad=x.grad=None;actual=torch.amax(x,dim,True);actual.sum().backward()
            expected=torch.amax(host,dim,True);expected.sum().backward();close(actual,expected,True);close(x.grad,host.grad,True)
        return dict(axis_keepdim_cases=count,multi_axis=True,out=True)

    def norms():
        count=0
        for shape in ((),(7,),(2,3),(2,2,3)):
            for dim in (None,[-1],[],[0,-1] if len(shape)>1 else [0]):
                for p in (0,1,2,float('inf'),float('-inf')):
                    for keep in (False,True):
                        host=(torch.randn(shape,dtype=dtype)/4).requires_grad_();x=upload(host)
                        before=npu.stats();actual=torch.linalg.vector_norm(x,p,dim,keep);actual.sum().backward();resident(before)
                        expected=torch.linalg.vector_norm(host,p,dim,keep);expected.sum().backward();close(actual,expected)
                        if p==0:assert x.grad is host.grad is None
                        else:close(x.grad,host.grad)
                        count+=1
        host=torch.randn(2,3,dtype=torch.float16).requires_grad_();x=upload(host)
        actual=torch.linalg.vector_norm(x,dim=1,dtype=torch.float32);actual.sum().backward()
        expected=torch.linalg.vector_norm(host,dim=1,dtype=torch.float32);expected.sum().backward();close(actual,expected);close(x.grad,host.grad)
        with torch.no_grad():
            x=upload(host);out=torch.empty(2,device='intel_npu',dtype=torch.float32)
            assert torch.linalg.vector_norm(x,dim=1,dtype=torch.float32,out=out) is out;close(out,expected)
            close(torch.norm(x),torch.norm(host));close(x.norm(p=1,dim=0),host.norm(p=1,dim=0))
        return dict(norm_cases=count,orders=[0,1,2,'inf','-inf'],fp16_to_fp32=True,legacy_norm=True)

    def edges():
        count=0
        for values in ([0.,-0.,0.],[2.,2.,-2.],[float('inf'),float('-inf'),float('inf')],
                       [1.,float('nan'),float('nan')],[0.,0.],[3e-30,4e-30],[3e30,4e30]):
            host=torch.tensor(values,dtype=dtype)
            for op in (torch.argmax,torch.argmin):close(op(upload(host)),op(host),True)
            for op in (torch.amax,torch.amin):
                reference=host.clone().requires_grad_();x=upload(reference)
                actual=op(x);actual.backward();expected=op(reference);expected.backward()
                close(actual,expected,True);close(x.grad,reference.grad)
            for p in (0,1,2,float('inf'),float('-inf')):
                reference=host.clone().requires_grad_();x=upload(reference)
                actual=torch.linalg.vector_norm(x,p);actual.backward();expected=torch.linalg.vector_norm(reference,p);expected.backward()
                close(actual,expected)
                if p:close(x.grad,reference.grad)
                count+=1
        # Lowest nonzero FP32 magnitudes must remain ordered rather than compare as zero.
        host=torch.tensor([0.,-2**-149,2**-149,2**-148],dtype=torch.float32)
        for op in (torch.argmax,torch.argmin,torch.amax,torch.amin):close(op(upload(host)),op(host),True)
        host=torch.tensor([[3e30],[3e-30]],dtype=torch.float32,requires_grad=True);x=upload(host)
        actual=torch.linalg.vector_norm(x,dim=-1);actual.sum().backward()
        expected=torch.linalg.vector_norm(host,dim=-1);expected.sum().backward();close(actual,expected,True);close(x.grad,host.grad,True)
        host=torch.tensor([0.,-0.,float('nan'),float('inf'),float('-inf'),2**-149],dtype=dtype,requires_grad=True);x=upload(host)
        for op in (torch.isnan,torch.isinf,torch.isfinite):
            before=npu.stats();actual=op(x);resident(before);close(actual,op(host),True)
            assert actual.dtype==torch.bool and not actual.requires_grad
            if 'out' in getattr(torch.ops.aten,op.__name__)._schemas:
                out=torch.empty_like(actual);getattr(torch.ops.aten,op.__name__).out(x,out=out);close(out,op(host),True)
        for length in (2051,65537):
            host=torch.ones(length,dtype=dtype,requires_grad=True)
            for op in (torch.amax,lambda x:torch.linalg.vector_norm(x,float('inf'))):
                host.grad=None;x=upload(host);actual=op(x);actual.backward();expected=op(host);expected.backward()
                close(actual,expected,True);close(x.grad,host.grad,True)
        base=torch.randn(3,10,dtype=dtype).requires_grad_();x=upload(base)
        actual=torch.linalg.vector_norm(x[:,::2],dim=0);actual.sum().backward()
        expected=torch.linalg.vector_norm(base[:,::2],dim=0);expected.sum().backward();close(actual,expected);close(x.grad,base.grad)
        close(torch.argmax(x[:,::2],dim=1),torch.argmax(base[:,::2],dim=1),True)
        view=upload(torch.tensor([1.,2.],dtype=dtype)).view(2,1).expand(2,4)
        close(torch.argmin(view,dim=1),torch.tensor([0,0]),True)
        return dict(special_norm_cases=count,first_tie_and_nan=True,fp32_subnormals=True,partial_broadcast=True)

    def tiled():
        cases=0
        for shape in ((65533,),(2,65537),(65537,1),(1048576,)):
            host=torch.zeros(shape,dtype=dtype);flat=host.view(-1);flat[-1]=3.;flat[0]=-2.
            x=upload(host)
            for op in (torch.argmax,torch.argmin,torch.amax,torch.amin):
                dim=1 if len(shape)==2 else None
                before=npu.stats();actual=op(x,dim=dim);resident(before);close(actual,op(host,dim=dim),True);cases+=1
        host=torch.zeros(2,65537,dtype=dtype);host[:,65532]=5;host[:,-1]=5;host[1,-1]=float('nan')
        for op in (torch.argmax,torch.argmin):close(op(upload(host),dim=1),op(host,dim=1),True)
        host=torch.full((65537,),.01,dtype=dtype);x=upload(host);cpu_differences={}
        for p in (0,1,2,float('inf'),float('-inf')):
            actual=torch.linalg.vector_norm(x,p);cpu=torch.linalg.vector_norm(host,p)
            reference=torch.linalg.vector_norm(host.double(),p).to(dtype) if p in (1,2) else cpu
            close(actual,reference);cpu_differences[str(p)]=float((actual.cpu()-cpu).abs())
        return dict(reduction_cases=cases,max_input=1048576,large_arg_indices=True,cross_chunk_ties_nan=True,
                    large_l1_l2_reference='independent CPU FP64, rounded to result dtype',cpu_large_norm_abs_differences=cpu_differences)

    def clipping():
        count=0
        for p in (0.,1.,2.,float('inf'),float('-inf')):
            for foreach in (None,False):
                for limit in (0.,.25,10.):
                    hosts=[torch.nn.Parameter(torch.ones(2,3,dtype=dtype)),torch.nn.Parameter(torch.ones(5,dtype=dtype)),torch.nn.Parameter(torch.ones(1,dtype=dtype))]
                    params=[torch.nn.Parameter(t.detach().to('intel_npu')) for t in hosts]
                    for i,(a,b) in enumerate(zip(params,hosts)):
                        if i<2:b.grad=torch.randn_like(b);a.grad=upload(b.grad)
                    before=npu.stats();actual=torch.nn.utils.clip_grad_norm_(params,limit,p,foreach=foreach);resident(before)
                    expected=torch.nn.utils.clip_grad_norm_(hosts,limit,p,foreach=False);close(actual,expected)
                    for a,b in zip(params[:2],hosts[:2]):close(a.grad,b.grad)
                    assert params[-1].grad is None;count+=1
        flags=0
        for v in (float('inf'),float('nan')):
            param=torch.nn.Parameter(torch.ones(2,dtype=dtype,device='intel_npu'));param.grad=upload(torch.tensor([v,1.],dtype=dtype))
            before=npu.stats()
            try:torch.nn.utils.clip_grad_norm_([param],1.,error_if_nonfinite=True,foreach=False)
            except RuntimeError as e:assert 'non-finite' in str(e)
            else:raise AssertionError('Nonfinite norm accepted')
            flags+=npu.stats()['host_download_copies']-before['host_download_copies']
            close(param.grad,torch.tensor([v,1.],dtype=dtype),True)
            cpu=torch.nn.Parameter(torch.ones(2,dtype=dtype));cpu.grad=torch.tensor([v,1.],dtype=dtype)
            before=npu.stats();actual=torch.nn.utils.clip_grad_norm_([param],1.,foreach=False);resident(before)
            expected=torch.nn.utils.clip_grad_norm_([cpu],1.,foreach=False);close(actual,expected);close(param.grad,cpu.grad)
        param=torch.nn.Parameter(torch.ones(2,dtype=dtype,device='intel_npu'));param.grad=upload(torch.tensor([3.,4.],dtype=dtype))
        before=npu.stats();actual=torch.nn.utils.clip_grad_norm_([param],1.,error_if_nonfinite=True,foreach=False);resident(before,1)
        close(actual,torch.tensor(5.,dtype=dtype),True)
        return dict(clipping_cases=count,nonfinite_rejected=2,nonfinite_scalar_flags=flags)

    def contracts():
        x=torch.ones(2,3,dtype=dtype,device='intel_npu');wrong=torch.empty(1,device='intel_npu',dtype=dtype)
        many=torch.ones(524289,1,dtype=dtype,device='intel_npu')
        actions=[lambda:torch.argmax(x,2),lambda:torch.argmax(x,out=wrong),lambda:torch.argmax(many,1),lambda:torch.amax(x,[0,0]),
                 lambda:torch.linalg.vector_norm(x,3),lambda:torch.linalg.vector_norm(x,dim=[3]),
                 lambda:torch.linalg.vector_norm(x,out=wrong),lambda:torch.linalg.vector_norm(x,dtype=torch.float64)]
        if dtype==torch.float32:actions.append(lambda:torch.linalg.vector_norm(x,dtype=torch.float16))
        for action in actions:
            before=npu.stats()
            try:action()
            except (ValueError,IndexError,RuntimeError,NotImplementedError):pass
            else:raise AssertionError('Invalid arguments accepted')
            assert npu.stats()['npu_calls']==before['npu_calls']
        with torch.no_grad():
            value=torch.ops.aten.norm.ScalarOpt_dim_dtype(x,2,[1],False,dtype=torch.float32)
            close(value,torch.ops.aten.norm.ScalarOpt_dim_dtype(torch.ones(2,3,dtype=dtype),2,[1],False,dtype=torch.float32))
        host=torch.ones(2,3,dtype=dtype,requires_grad=True);x=upload(host);value=torch.linalg.vector_norm(x)
        with torch.no_grad():x.fill_(2.)
        try:value.backward()
        except RuntimeError as e:assert 'version' in str(e) or 'inplace' in str(e)
        else:raise AssertionError('Saved input mutation accepted')
        return dict(metadata_rejections=len(actions),version_guard=True)

    def training(amp=False):
        from torch_npu_transformer_demo import cpu_tree
        host=model().to(torch.float32 if amp else dtype);network=copy.deepcopy(host).to('intel_npu')
        inputs=torch.randn(3,4).to(torch.float32 if amp else dtype)/4;labels=torch.tensor([0,1,2]);data,target=upload(inputs),upload(labels)
        optimizers=[torch.optim.SGD(m.parameters(),lr=.02,momentum=.5,foreach=False,fused=False) for m in (network,host)]
        scalers=[torch.intel_npu.amp.GradScaler(init_scale=8,growth_interval=2),torch.amp.GradScaler('cpu',init_scale=8,growth_interval=2)] if amp else None
        losses=[];errors=[];norms=[];saved=io.BytesIO();tolerance=dict(rtol=.008,atol=.0008) if amp or dtype==torch.float16 else dict(rtol=8e-5,atol=8e-6)
        for step in range(4):
            values=[];totals=[]
            for i,(net,opt,x,y) in enumerate(zip((network,host),optimizers,(data,inputs),(target,labels))):
                opt.zero_grad(set_to_none=True);before=npu.stats()
                with torch.autocast('intel_npu' if i==0 else 'cpu',dtype=torch.float16,enabled=amp):
                    output=net(x);loss=torch.nn.functional.cross_entropy(output.float(),y)
                if amp:scalers[i].scale(loss).backward();scalers[i].unscale_(opt)
                else:loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),.1,foreach=False)
                prediction=torch.argmax(output,dim=-1)
                if i==0:resident(before,1+int(amp))
                values.append(loss);totals.append(norm)
            close(values[0],values[1]);close(totals[0],totals[1]);difference=[]
            for a,b in zip(network.parameters(),host.parameters()):
                torch.testing.assert_close(a.grad.cpu(),b.grad,**tolerance);difference.append(float((a.grad.cpu()-b.grad).abs().max()))
            for i,opt in enumerate(optimizers):
                if amp:scalers[i].step(opt);scalers[i].update()
                else:opt.step()
            for a,b in zip(network.parameters(),host.parameters()):torch.testing.assert_close(a.detach().cpu(),b.detach(),**tolerance)
            if amp:assert scalers[0].state_dict()==scalers[1].state_dict()
            losses.append(float(values[0].detach().cpu()));norms.append(float(totals[0].cpu()));errors.append(max(difference))
            if step==2:torch.save(dict(model=cpu_tree(network.state_dict()),optimizer=cpu_tree(optimizers[0].state_dict()),scaler=scalers[0].state_dict() if amp else None),saved)
        saved.seek(0);state=torch.load(saved,weights_only=True);restored=model().to(torch.float32 if amp else dtype).to('intel_npu');restored.load_state_dict(state['model'])
        opt=torch.optim.SGD(restored.parameters(),lr=.02,momentum=.5,foreach=False,fused=False);opt.load_state_dict(state['optimizer']);opt.zero_grad(set_to_none=True)
        if amp:scaler=torch.intel_npu.amp.GradScaler();scaler.load_state_dict(state['scaler'])
        with torch.autocast('intel_npu',dtype=torch.float16,enabled=amp):loss=torch.nn.functional.cross_entropy(restored(data).float(),target)
        if amp:scaler.scale(loss).backward();scaler.unscale_(opt)
        else:loss.backward()
        torch.nn.utils.clip_grad_norm_(restored.parameters(),.1,foreach=False)
        if amp:scaler.step(opt);scaler.update();assert scaler.state_dict()==scalers[0].state_dict()
        else:opt.step()
        assert float(loss.detach().cpu())==losses[-1]
        for a,b in zip(network.parameters(),restored.parameters()):
            torch.testing.assert_close(a.detach().cpu(),b.detach().cpu(),rtol=0,atol=0)
            torch.testing.assert_close(optimizers[0].state[a]['momentum_buffer'].cpu(),opt.state[b]['momentum_buffer'].cpu(),rtol=0,atol=0)
        return dict(steps=4,amp=amp,losses=losses,norms=norms,max_gradient_error=errors,checkpoint_bitwise=True,tolerance=tolerance)

    actions=dict(reductions=reductions,norms=norms,edges=edges,tiled=tiled,clipping=clipping,contracts=contracts,training=training,amp=lambda:training(True))
    before=npu.stats()
    try:detail=actions[case]();status='passed'
    except Exception as error:status='failed';detail=dict(error=str(error),traceback=traceback.format_exc())
    gc.collect();after=npu.stats()
    if after['live_buffers']-after['cached_scalar_buffers']!=before['live_buffers']-before['cached_scalar_buffers']:
        status='failed';detail['lifecycle_error']='Device storage leak'
    npu.shutdown();final=npu.stats()
    if final['context_failed'] or any(final[k] for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes')):status='failed'
    names=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_geometry.py','torch_npu_reduction_norm_demo.py',
           'npunlock/tests/fixtures/npu4000/resident_row_extrema.c','npunlock/tests/fixtures/npu4000/resident/manifest.json')
    print(case,precision,status,detail.get('error','')[:1200],flush=True)
    return dict(status=status,case=case,precision=precision,details=detail,npu_calls=after['npu_calls']-before['npu_calls'],final_stats=final,
                source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=('reductions','norms','edges','tiled','clipping','contracts','training','amp'),required=True)
    parser.add_argument('--precision',choices=('float16','float32'),default='float32');parser.add_argument('--label',default='after');args=parser.parse_args()
    result=execute(args.case,args.precision);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{args.label}-{args.case}-{args.precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(result['status']!='passed')

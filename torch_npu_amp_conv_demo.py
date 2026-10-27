"""Isolate FP16 batch rounding, then exercise independent AMP convolution training."""
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import traceback
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/amp-conv'


class _HalfConvFP32(torch.autograd.Function):
    """CPU-only reference with FP32 accumulation and FP16 saved/output tensors."""
    @staticmethod
    def forward(ctx,x,w,b,stride,padding,dilation):
        if any(t.device.type!='cpu' or t.dtype!=torch.float16 for t in (x,w,b) if t is not None):
            raise ValueError('The AMP opmath reference requires CPU FP16 tensors')
        ctx.save_for_backward(x,w);ctx.params=(stride,padding,dilation);ctx.bias=b is not None
        return F.conv2d(x.float(),w.float(),None if b is None else b.float(),stride,padding,dilation).half()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        x,w=ctx.saved_tensors;s,p,d=ctx.params;v=g.float()
        gx=torch.nn.grad.conv2d_input(x.shape,w.float(),v,s,p,d).half() if ctx.needs_input_grad[0] else None
        gw=torch.nn.grad.conv2d_weight(x.float(),w.shape,v,s,p,d).half() if ctx.needs_input_grad[1] else None
        gb=v.sum((0,2,3)).half() if ctx.bias and ctx.needs_input_grad[2] else None
        return gx,gw,gb,None,None,None


def reference_conv(x,w,b,stride=(1,1),padding=(0,0),dilation=(1,1)):
    return _HalfConvFP32.apply(x.half(),w.half(),None if b is None else b.half(),stride,padding,dilation)


def geometry_cases():
    base=((2,4,5),(3,3,3),(1,1),(1,1),(1,1))
    return [(f'batch{b}',(b,*base[0]),base[1],*base[2:]) for b in (1,2,4)]+[
        ('stride',(2,2,5,7),(2,2,3),(2,1),(1,1),(1,1)),
        ('dilation',(2,1,7,6),(2,2,2),(1,1),(1,0),(2,1)),
        ('pointwise',(2,2,5,5),(3,1,1),(1,1),(0,0),(1,1))]


def sample(case):
    name,shape,(channels,kh,kw),stride,padding,dilation=case
    gen=torch.Generator().manual_seed(307)
    x=(torch.randn(shape,generator=gen)/3).half()
    output=F.conv2d(x.float(),torch.zeros(channels,shape[1],kh,kw),stride=stride,padding=padding,dilation=dilation)
    g=(torch.randn(output.shape,generator=gen)/3).half()
    w=(torch.randn(channels,shape[1],kh,kw,generator=gen)/3).half()
    b=(torch.randn(channels,generator=gen)/8).half()
    return x,w,b,g


def batch_rounding_weight(x,g,weight_shape,stride=(1,1),padding=(0,0),dilation=(1,1)):
    columns=F.unfold(x.float(),weight_shape[2:],dilation,padding,stride)
    gradients=g.float().flatten(2);result=torch.zeros(weight_shape[0],columns.shape[1],dtype=torch.float16)
    for batch in range(x.shape[0]):result=(result.float()+gradients[batch]@columns[batch].t()).half()
    return result.view(weight_shape)


def conv_oracles(x,w,b,g,stride,padding,dilation):
    result={}
    for dtype,label in ((torch.float32,'opmath'),(torch.float64,'fp64')):
        a,k,seed=x.to(dtype),w.to(dtype),g.to(dtype)
        result[label]=(
            F.conv2d(a,k,b.to(dtype),stride,padding,dilation),
            torch.nn.grad.conv2d_input(x.shape,k,seed,stride,padding,dilation),
            torch.nn.grad.conv2d_weight(a,w.shape,seed,stride,padding,dilation),
            seed.sum((0,2,3)))
    result['native']=torch.nn.grad.conv2d_weight(x,w.shape,g,stride,padding,dilation)
    result['batch_rounding']=batch_rounding_weight(x,g,w.shape,stride,padding,dilation)
    return result


def model(kind,reference='npu'):
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem=torch.nn.Conv2d(1,2,3,padding=1)
            self.block=torch.nn.ModuleList([torch.nn.Conv2d(2,2,3,padding=1) for _ in range(2)]) if kind=='residual' else torch.nn.ModuleList()
            self.head=torch.nn.Linear(2,3)
        def conv(self,layer,x):
            if reference=='npu':return layer(x)
            if reference=='opmath':return reference_conv(x,layer.weight,layer.bias,layer.stride,layer.padding,layer.dilation)
            return F.conv2d(x.half(),layer.weight.half(),layer.bias.half(),layer.stride,layer.padding,layer.dilation)
        def forward(self,x):
            stem=self.conv(self.stem,x)
            x=F.silu(stem) if kind=='small' else F.relu(stem)
            if self.block:x=F.relu(self.conv(self.block[1],F.relu(self.conv(self.block[0],x)))+x)
            x=F.adaptive_avg_pool2d(x,1).flatten(1)
            return self.head(x) if reference=='npu' else F.linear(x.half(),self.head.weight.half(),self.head.bias.half())
    return Net()


def execute(case,steps=4,scaler_api='legacy'):
    import torch_intel_npu as npu
    from torch_npu_transformer_demo import cpu_tree
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False;torch.manual_seed(293)
    def no_payload(before,downloads=0):
        after=npu.stats();assert after['host_download_copies']-before['host_download_copies']==downloads
        assert after['host_device_copies']==before['host_device_copies']

    def local():
        rows=[]
        for geometry in geometry_cases():
            name,_,_,stride,padding,dilation=geometry
            x,w,b,g=sample(geometry);oracles=conv_oracles(x,w,b,g,stride,padding,dilation)
            leaves=[t.float().to('intel_npu').requires_grad_() for t in (x,w,b)]
            seed=g.to('intel_npu');before=npu.stats()
            with torch.autocast('intel_npu'):output=F.conv2d(*leaves,stride=stride,padding=padding,dilation=dilation)
            output.backward(seed);no_payload(before)
            actual=[output.detach().cpu(),*[t.grad.cpu() for t in leaves]]
            expected=[v.half().float() for v in oracles['opmath']]
            strict=[];fp64_errors=[]
            for label,a,e,truth in zip(('output','input','weight','bias'),actual,expected,oracles['fp64']):
                torch.testing.assert_close(a.float(),e,rtol=.003,atol=.0005)
                try:torch.testing.assert_close(a.float(),e,rtol=3e-5,atol=3e-6)
                except AssertionError:strict.append(label)
                fp64_errors.append(float((a.double()-truth).abs().max()))
            native=oracles['native'];batched=oracles['batch_rounding'];single=oracles['opmath'][2].half()
            reproduction=torch.equal(native,batched)
            if name.startswith('batch'):assert reproduction
            rows.append(dict(name=name,strict_opmath_failures=strict,
                native_vs_single_round=int(torch.count_nonzero(native!=single)),native_batch_rounding_reproduced=reproduction,
                npu_vs_fp64_rounded_weight=int(torch.count_nonzero(actual[2]!=oracles['fp64'][2].half().float())),
                fp64_max_absolute=dict(zip(('output','input','weight','bias'),fp64_errors))))
            del output,leaves,seed
        return dict(cases=rows,fp16_bounds=True,limitations=[r['name'] for r in rows if r['strict_opmath_failures']])

    def training(kind):
        controls=[model(kind,reference) for reference in ('opmath','native')]
        rng=torch.get_rng_state();net=model(kind);torch.set_rng_state(rng)
        for control in controls[1:]:control.load_state_dict(controls[0].state_dict())
        net.load_state_dict(controls[0].state_dict());net.to('intel_npu')
        inputs=torch.randn(2 if kind=='small' else 4,1,4 if kind=='small' else 6,4 if kind=='small' else 6)/4
        labels=torch.tensor([0,2]) if kind=='small' else torch.arange(inputs.shape[0])%3
        x,y=inputs.to('intel_npu'),labels.to('intel_npu')
        opts=[torch.optim.SGD(m.parameters(),lr=.02,momentum=.5,foreach=False,fused=False) for m in (net,*controls)]
        scaler=(torch.amp.GradScaler('intel_npu',init_scale=128.,growth_interval=2) if scaler_api=='standard' else
                torch.intel_npu.amp.GradScaler(init_scale=128.,growth_interval=2))
        cpu_scalers=[torch.amp.GradScaler('cpu',init_scale=128.,growth_interval=2) for _ in controls]
        losses=[];native_failures=[];strict_failures=[];checkpoint=io.BytesIO()
        def cycle(m,opt):
            opt.zero_grad(set_to_none=True);before=npu.stats()
            with torch.autocast('intel_npu'):loss=F.cross_entropy(m(x),y)
            scaler.scale(loss).backward();scaler.unscale_(opt);no_payload(before,2)
            assert all(p.dtype==p.grad.dtype==torch.float32 for p in m.parameters())
            return loss
        for step in range(steps):
            loss=cycle(net,opts[0]);compared=[]
            for control,opt,cpu_scaler in zip(controls,opts[1:],cpu_scalers):
                opt.zero_grad(set_to_none=True);expected=F.cross_entropy(control(inputs).float(),labels)
                cpu_scaler.scale(expected).backward();cpu_scaler.unscale_(opt)
                errors=[]
                for name,a,b in [('loss',loss,expected)]+[(n,p.grad,dict(control.named_parameters())[n].grad) for n,p in net.named_parameters()]:
                    try:torch.testing.assert_close(a.detach().cpu(),b.detach(),rtol=3e-5,atol=3e-6)
                    except AssertionError:errors.append(dict(name=name,max_absolute=float((a.detach().cpu()-b.detach()).abs().max())))
                    torch.testing.assert_close(a.detach().cpu(),b.detach(),rtol=.003,atol=.0005)
                compared.append(errors)
            if compared[0]:strict_failures.append(dict(step=step,errors=compared[0]))
            if compared[1]:native_failures.append(dict(step=step,errors=compared[1]))
            before=npu.stats();scaler.step(opts[0]);scaler.update();no_payload(before)
            for opt,cpu_scaler in zip(opts[1:],cpu_scalers):cpu_scaler.step(opt);cpu_scaler.update()
            for p,q in zip(net.parameters(),controls[0].parameters()):torch.testing.assert_close(p.detach().cpu(),q.detach(),rtol=3e-5,atol=3e-6)
            assert all(s.state_dict()==scaler.state_dict() for s in cpu_scalers)
            losses.append(loss.item())
            final=dict(model=cpu_tree(net.state_dict()),optimizer=cpu_tree(opts[0].state_dict()),scaler=scaler.state_dict())
            if step==steps-2:torch.save(final,checkpoint)
        checkpoint.seek(0);saved=torch.load(checkpoint,weights_only=True);restored=model(kind).to('intel_npu')
        restored.load_state_dict(saved['model']);opt=torch.optim.SGD(restored.parameters(),lr=.02,momentum=.5,foreach=False,fused=False)
        opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        replay=cycle(restored,opt);scaler.step(opt);scaler.update()
        assert replay.item()==losses[-1] and scaler.state_dict()==final['scaler']
        for name,v in cpu_tree(restored.state_dict()).items():torch.testing.assert_close(v,final['model'][name],rtol=0,atol=0)
        for key,v in cpu_tree(opt.state_dict())['state'].items():torch.testing.assert_close(v['momentum_buffer'],final['optimizer']['state'][key]['momentum_buffer'],rtol=0,atol=0)
        return dict(steps=steps,parameters=sum(p.numel() for p in net.parameters()),losses=losses,
                    opmath_strict_failures=strict_failures,native_cpu_strict_failures=native_failures,
                    independent_reference=True,fp16_bounds=True,checkpoint_bitwise=True,
                    limitations=['Independent FP32-opmath reference exceeded strict threshold'] if strict_failures else [])

    before=npu.stats();row=dict(name=case)
    try:
        row.update(local() if case=='local' else training(case));row['status']='passed_with_limitations' if row.get('limitations') else 'passed'
    except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
    gc.collect();after=npu.stats();row['npu_calls']=after['npu_calls']-before['npu_calls']
    if after['live_buffers']-after['cached_scalar_buffers']!=before['live_buffers']-before['cached_scalar_buffers']:row.update(status='failed',lifecycle_error='Leaked tensor storage')
    print(case,row['status'],row.get('error','')[:300],flush=True)
    npu.shutdown();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in ('live_buffers','cached_scalar_buffers','cached_buffer_blocks','cached_sessions','quarantined_buffer_bytes'))
    sources=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_amp.py','torch_npu_amp_conv_demo.py',
             'npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/_torch_intel_npu.pyd')
    return dict(status=row['status'] if healthy else 'failed',scaler_api=scaler_api,checks=[row],final_stats=final,
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=('all','local','small','residual'),default='all')
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--scaler-api',choices=('legacy','standard'),default='legacy');args=parser.parse_args()
    if args.steps<2:parser.error('steps must be >=2 for checkpoint replay')
    if args.case=='all':
        for case in ('local','small','residual'):subprocess.run([sys.executable,'-B',__file__,'--case',case,'--steps',str(args.steps),'--scaler-api',args.scaler_api],check=True)
        raise SystemExit(0)
    result=execute(args.case,args.steps,args.scaler_api);OUT.mkdir(parents=True,exist_ok=True)
    name=f'standard-{args.case}' if args.scaler_api=='standard' else args.case
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(result['status'] not in ('passed','passed_with_limitations'))

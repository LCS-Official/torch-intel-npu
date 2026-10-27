"""Bounded native autocast and NPU GradScaler hardware acceptance."""
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
import warnings

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/amp'


def amp_model(kind,reference=False):
    import torch
    import torch.nn.functional as F
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if kind=='mlp':
                self.first=torch.nn.Linear(4,8);self.norm=torch.nn.LayerNorm(8);self.head=torch.nn.Linear(8,3)
            else:
                self.first=torch.nn.Conv2d(1,2,3,padding=1);self.head=torch.nn.Linear(2,3)
        def forward(self,x):
            if reference:
                x=(F.linear(x.half(),self.first.weight.half(),self.first.bias.half()) if kind=='mlp' else
                   F.conv2d(x.half(),self.first.weight.half(),self.first.bias.half(),padding=1))
                if kind=='mlp':x=F.layer_norm(x.float(),(8,),self.norm.weight,self.norm.bias,self.norm.eps)
                x=F.silu(x)
                if kind=='conv':x=F.adaptive_avg_pool2d(x,1).flatten(1)
                return F.linear(x.half(),self.head.weight.half(),self.head.bias.half())
            x=self.first(x)
            if kind=='conv' and getattr(self,'capture',None) is not None and x.requires_grad:
                x.register_hook(lambda gradient:self.capture.update(gradient=gradient.detach()))
            if kind=='mlp':x=self.norm(x)
            x=F.silu(x)
            if kind=='conv':x=F.adaptive_avg_pool2d(x,1).flatten(1)
            return self.head(x)
    return Model()


def execute(case,scaler_api='legacy'):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.manual_seed(293);torch.backends.mkldnn.enabled=False
    def make_scaler(**kwargs):
        return torch.amp.GradScaler('intel_npu',**kwargs) if scaler_api=='standard' else torch.intel_npu.amp.GradScaler(**kwargs)
    def probe():
        x=torch.ones(2,4,device='intel_npu');model=torch.nn.Linear(4,3).to('intel_npu')
        with warnings.catch_warnings(record=True) as caught:
            with torch.autocast('intel_npu',dtype=torch.float16):
                y=model(x);loss=torch.nn.functional.mse_loss(y,torch.zeros(2,3,device='intel_npu'))
                enabled=torch.is_autocast_enabled('intel_npu')
        try:
            scaler=torch.amp.GradScaler('intel_npu');scaler.scale(loss).backward()
            scale_status='scale accepted'
        except Exception as error:scale_status=str(error).splitlines()[0]
        return dict(autocast_enabled=enabled,output_dtype=str(y.dtype),loss_dtype=str(loss.dtype),
                    warnings=[str(w.message) for w in caught],standard_scaler=scale_status)

    def no_payload(before,downloads=0):
        after=npu.stats()
        assert after['host_download_copies']-before['host_download_copies']==downloads,(before,after)
        assert after['host_device_copies']==before['host_device_copies']

    def policy():
        from concurrent.futures import ThreadPoolExecutor
        assert torch.intel_npu.get_amp_supported_dtype()==[torch.float16]
        assert torch.get_autocast_dtype('intel_npu')==torch.float16 and not torch.is_autocast_enabled('intel_npu')
        host=torch.randn(2,4)/4;weight=torch.randn(3,4)/4;bias=torch.randn(3)/4
        x=host.to('intel_npu').requires_grad_();w=weight.to('intel_npu').requires_grad_();b=bias.to('intel_npu').requires_grad_()
        target=torch.zeros(2,3,device='intel_npu');labels=torch.tensor([0,2],device='intel_npu')
        before=npu.stats()
        with torch.autocast('intel_npu'):
            assert torch.is_autocast_enabled('intel_npu')
            values=[torch.nn.functional.linear(x,w,b),x@w.t(),torch.mm(x,w.t()),
                    torch.addmm(b,x,w.t()),torch.mv(w,x[0]),
                    torch.bmm(x.view(1,2,4),w.t().contiguous().view(1,4,3))]
            assert all(v.dtype==torch.float16 for v in values)
            losses=[torch.nn.functional.mse_loss(values[0],target),torch.nn.functional.cross_entropy(values[0],labels)]
            assert all(v.dtype==torch.float32 for v in losses)
            h=values[0]
            guarded=[torch.softmax(h,-1),torch.log_softmax(h,-1),h.sum(),h.mean(),torch.exp(h),torch.log(h.abs()+1.),
                     torch.reciprocal(h.abs()+1.),torch.rsqrt(h.abs()+1.),
                     torch.nn.functional.layer_norm(h,(3,)),torch.nn.functional.rms_norm(h,(3,))]
            assert all(v.dtype==torch.float32 for v in guarded)
            assert h.sum(dtype=torch.float16).dtype==torch.float16 and h.softmax(-1,dtype=torch.float16).dtype==torch.float16
            assert torch.nn.functional.silu(h).dtype==torch.float16
            with torch.autocast('intel_npu',enabled=False):assert torch.mm(x,w.t()).dtype==torch.float32
            with ThreadPoolExecutor(max_workers=1) as pool:
                assert pool.submit(torch.is_autocast_enabled,'intel_npu').result() is False
            with torch.no_grad():
                out=torch.empty(2,3,device='intel_npu');assert torch.mm(x.detach(),w.detach().t(),out=out).dtype==torch.float32
        no_payload(before,1) # NLL label validation only.
        assert not torch.is_autocast_enabled('intel_npu')
        expected=torch.nn.functional.linear(host.half(),weight.half(),bias.half())
        torch.testing.assert_close(values[0].detach().cpu(),expected,rtol=.003,atol=.0005)
        losses[0].backward();assert all(t.grad.dtype==torch.float32 for t in (x,w,b))
        module=torch.nn.Linear(4,3).to('intel_npu');plain=module(host.to('intel_npu'))
        with torch.autocast('intel_npu',cache_enabled=True):
            start=npu.stats()['npu_calls'];a=module(host.to('intel_npu'));first=npu.stats()['npu_calls']-start
            start=npu.stats()['npu_calls'];a=module(host.to('intel_npu'));second=npu.stats()['npu_calls']-start
        assert first>second,(first,second)
        with torch.no_grad():module.weight.add_(.25)
        with torch.autocast('intel_npu'):changed=module(host.to('intel_npu'))
        assert not torch.equal(a.detach().cpu(),changed.detach().cpu())
        with warnings.catch_warnings(record=True) as caught:
            with torch.autocast('intel_npu',dtype=torch.bfloat16):assert not torch.is_autocast_enabled('intel_npu')
        assert caught and plain.dtype==torch.float32
        return dict(fp16_matrix_and_fp32_sensitive_policy=True,fp32_master_gradients=True,
                    explicit_dtype_and_out_preserved=True,nested_and_thread_local=True,cache_calls=[first,second],unsupported_bf16_disabled=True)

    def scaler():
        p=torch.nn.Parameter(torch.tensor([.5,-1.,2.,.25],device='intel_npu'))
        q=torch.nn.Parameter(p.detach().cpu());opt=torch.optim.SGD([p],lr=.02,momentum=.5,foreach=False,fused=False)
        cpu_opt=torch.optim.SGD([q],lr=.02,momentum=.5,foreach=False,fused=False)
        actual=make_scaler(init_scale=8.,growth_interval=2)
        expected=torch.amp.GradScaler('cpu',init_scale=8.,growth_interval=2);records=[]
        for special in (None,None,float('inf'),None,float('nan'),None,None):
            actual.scale(p.sum());expected.scale(q.sum())
            factor=actual.get_scale();grad=torch.tensor([2.,-4.,.25,0.])*factor
            if special is not None:grad[1]=special
            p.grad=grad.to('intel_npu');q.grad=grad.clone()
            before=npu.stats();actual.unscale_(opt);no_payload(before,1);expected.unscale_(cpu_opt)
            torch.testing.assert_close(p.grad.cpu(),q.grad,rtol=0,atol=0,equal_nan=True)
            before=npu.stats();actual.step(opt);actual.update();no_payload(before)
            expected.step(cpu_opt);expected.update()
            torch.testing.assert_close(p.detach().cpu(),q.detach(),rtol=3e-5,atol=3e-6)
            assert actual.state_dict()==expected.state_dict()
            if opt.state[p]:torch.testing.assert_close(opt.state[p]['momentum_buffer'].cpu(),cpu_opt.state[q]['momentum_buffer'],rtol=3e-5,atol=3e-6)
            records.append(actual.get_scale())
        # All microbatch losses share a scale until one optimizer update.
        opt.zero_grad(set_to_none=True);cpu_opt.zero_grad(set_to_none=True)
        for _ in range(2):actual.scale(p.sum()).backward();expected.scale(q.sum()).backward()
        before=npu.stats();actual.step(opt);actual.update();no_payload(before,1)
        expected.step(cpu_opt);expected.update();torch.testing.assert_close(p.detach().cpu(),q.detach(),rtol=3e-5,atol=3e-6)
        saved=actual.state_dict();replacement=make_scaler();replacement.load_state_dict(saved)
        assert replacement.state_dict()==saved
        # A real FP16 backward overflow, using the default scale, must skip the update.
        net=torch.nn.Linear(1,1,bias=False).to('intel_npu');original=net.weight.detach().cpu()
        default=make_scaler();optimizer=torch.optim.SGD(net.parameters(),lr=.1,foreach=False,fused=False)
        with torch.autocast('intel_npu'):loss=net(torch.ones(1,1,device='intel_npu')).float().sum()*2.
        default.scale(loss).backward()
        before=npu.stats();default.step(optimizer);default.update();no_payload(before,1)
        assert default.get_scale()==32768.;torch.testing.assert_close(net.weight.detach().cpu(),original,rtol=0,atol=0)
        # Per-optimizer flags aggregate into a single scale update.
        a=torch.nn.Parameter(torch.ones(2,device='intel_npu'));b=torch.nn.Parameter(torch.ones(2,device='intel_npu'))
        multi=make_scaler(init_scale=8.)
        scaled=multi.scale([a.sum(),b.sum()]);assert isinstance(scaled,list)
        sum(scaled).backward();b.grad.fill_(float('inf'))
        optimizers=[torch.optim.SGD([t],lr=.1,foreach=False,fused=False) for t in (a,b)]
        before=npu.stats()
        for optimizer in optimizers:multi.step(optimizer)
        multi.update();no_payload(before,2)
        assert multi.get_scale()==4.;torch.testing.assert_close(a.detach().cpu(),torch.full((2,),.9));torch.testing.assert_close(b.detach().cpu(),torch.ones(2))
        return dict(scale_history=records,finite_inf_nan_skip_and_growth=True,independent_cpu_scaler=True,
                    fp32_master_gradients=True,accumulation_and_scaler_state=True,default_backward_overflow=True,multiple_optimizers=True)

    def contracts():
        p=torch.nn.Parameter(torch.ones(4,device='intel_npu'));opt=torch.optim.SGD([p],lr=.02,foreach=False,fused=False)
        scaler=make_scaler(init_scale=8.);scaler.scale(p.sum());p.grad=torch.ones_like(p)
        bad=torch.nn.Parameter(torch.ones(4,device='intel_npu',dtype=torch.float16));bad.grad=torch.ones_like(bad)
        mixed=torch.optim.SGD([p,bad],lr=.02,foreach=False,fused=False)
        before=npu.stats();cpu_grad=p.grad.cpu();before=npu.stats()
        try:scaler.unscale_(mixed)
        except ValueError:pass
        else:raise AssertionError('FP16 master accepted')
        assert npu.stats()['npu_calls']==before['npu_calls'];torch.testing.assert_close(p.grad.cpu(),cpu_grad,rtol=0,atol=0)
        # Grad aliases and unsupported optimizers must fail before any unscale writes.
        alias=torch.nn.Parameter(torch.ones(4,device='intel_npu'));alias.grad=p.grad
        aliased=torch.optim.SGD([p,alias],lr=.02,foreach=False,fused=False)
        for wrong in (aliased,torch.optim.SGD([p],lr=.02,foreach=True,fused=False)):
            before=npu.stats()
            try:scaler.unscale_(wrong)
            except ValueError:pass
            else:raise AssertionError('Unsafe optimizer accepted')
            assert npu.stats()['npu_calls']==before['npu_calls']
        scaler.unscale_(opt)
        for action in (lambda:scaler.unscale_(opt),lambda:scaler.step(opt,closure=lambda:None)):
            before=npu.stats()
            try:action()
            except RuntimeError:pass
            else:raise AssertionError('Bad scaler stage accepted')
            assert npu.stats()['npu_calls']==before['npu_calls']
        scaler.step(opt)
        try:scaler.step(opt)
        except RuntimeError:pass
        else:raise AssertionError('Double step accepted')
        scaler.update();saved=scaler.state_dict()
        for bad_state in ({**saved,'scale':float('nan')},{**saved,'growth_interval':0},{**saved,'_growth_tracker':-1}):
            try:scaler.load_state_dict(bad_state)
            except ValueError:pass
            else:raise AssertionError('Invalid scaler state accepted')
            assert scaler.state_dict()==saved
        disabled=make_scaler(enabled=False)
        assert disabled.scale(p) is p and disabled.state_dict()=={}
        return dict(master_dtype_alias_optimizer_preflight=True,stage_guards=True,checkpoint_validation=True,disabled_identity=True)

    def training(kind):
        from torch_npu_transformer_demo import cpu_tree
        cpu=amp_model(kind,True);model=amp_model(kind);model.load_state_dict(cpu.state_dict());model.to('intel_npu')
        inputs=torch.randn(2,4)/4 if kind=='mlp' else torch.randn(2,1,4,4)/4
        labels=torch.tensor([0,2]);x=inputs.to('intel_npu');y=labels.to('intel_npu')
        opts=[torch.optim.SGD(net.parameters(),lr=.02,momentum=.5,foreach=False,fused=False) for net in (model,cpu)]
        scaler=make_scaler(init_scale=128.,growth_interval=2)
        reference=torch.amp.GradScaler('cpu',init_scale=128.,growth_interval=2)
        losses=[];checkpoint=io.BytesIO();trajectory=[];strict_failures=[];local_weight_checks=[]
        if kind=='conv':model.capture={}
        def cycle(net,opt):
            opt.zero_grad(set_to_none=True);before=npu.stats()
            with torch.autocast('intel_npu'):
                logits=net(x);loss=torch.nn.functional.cross_entropy(logits,y)
            assert logits.dtype==torch.float16 and loss.dtype==torch.float32
            scaler.scale(loss).backward();scaler.unscale_(opt);no_payload(before,2)
            assert all(p.dtype==p.grad.dtype==torch.float32 for p in net.parameters())
            return loss
        def close(a,b):torch.testing.assert_close(a.detach().cpu(),b.detach(),rtol=3e-5,atol=3e-6)
        for step in range(3):
            loss=cycle(model,opts[0]);opts[1].zero_grad(set_to_none=True)
            expected=torch.nn.functional.cross_entropy(cpu(inputs).float(),labels)
            reference.scale(expected).backward();reference.unscale_(opts[1]);close(loss,expected)
            for name,p in model.named_parameters():
                q=dict(cpu.named_parameters())[name]
                try:close(p.grad,q.grad)
                except AssertionError:
                    if kind!='conv':raise
                    delta=(p.grad.cpu()-q.grad).abs()
                    strict_failures.append(dict(step=step,parameter=name,max_absolute=delta.max().item()))
                    # Keep the strict failure; separately check the existing FP16-compute bound.
                    torch.testing.assert_close(p.grad.cpu(),q.grad,rtol=.003,atol=.0005)
            if kind=='conv':
                # A local FP32 accumulation control uses the actual saved NPU upstream gradient.
                seed=model.capture['gradient'].cpu().float()
                local=torch.nn.grad.conv2d_weight(inputs.half().float(),cpu.first.weight.shape,seed,padding=1).half().float()/scaler.get_scale()
                local_weight_checks.append((model.first.weight.grad.cpu()-local).abs().max().item())
            before=npu.stats();scaler.step(opts[0]);scaler.update();no_payload(before)
            reference.step(opts[1]);reference.update()
            for p,q in zip(model.parameters(),cpu.parameters()):close(p,q)
            assert scaler.state_dict()==reference.state_dict()
            losses.append(loss.item());trajectory.append(scaler.get_scale())
            current=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opts[0].state_dict()),scaler=scaler.state_dict())
            if step==1:torch.save(current,checkpoint)
        checkpoint.seek(0);saved=torch.load(checkpoint,weights_only=True);restored=amp_model(kind).to('intel_npu');restored.load_state_dict(saved['model'])
        opt=torch.optim.SGD(restored.parameters(),lr=.02,momentum=.5,foreach=False,fused=False);opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        loss=cycle(restored,opt);scaler.step(opt);scaler.update()
        assert loss.item()==losses[-1] and scaler.state_dict()==current['scaler']
        for name,value in cpu_tree(restored.state_dict()).items():torch.testing.assert_close(value,current['model'][name],rtol=0,atol=0)
        for key,value in cpu_tree(opt.state_dict())['state'].items():
            torch.testing.assert_close(value['momentum_buffer'],current['optimizer']['state'][key]['momentum_buffer'],rtol=0,atol=0)
        return dict(steps=3,losses=losses,scale_history=trajectory,fp32_master_and_gradient=True,
                    strict_cpu_reference=not strict_failures,strict_gradient_failures=strict_failures,
                    local_fp32_accumulation_max_abs=local_weight_checks,
                    fp16_compute_bound=True,checkpoint_next_step_bitwise=True,
                    limitations=['Native CPU FP16 convolution gradient exceeds strict FP32 threshold'] if strict_failures else [])

    actions=dict(probe=probe,policy=policy,scaler=scaler,contracts=contracts,mlp=lambda:training('mlp'),conv=lambda:training('conv'))
    row=dict(name=case,scaler_api=scaler_api)
    before=npu.stats()
    try:
        row.update(actions[case]());row['status']='passed_with_limitations' if row.get('limitations') else 'passed'
    except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
    gc.collect();after=npu.stats();row['npu_calls']=after['npu_calls']-before['npu_calls']
    if after['live_buffers']-after['cached_scalar_buffers']!=before['live_buffers']-before['cached_scalar_buffers']:
        row.update(status='failed',lifecycle_error='Leaked tensor storage')
    print(case,row['status'],row.get('error','')[:300],flush=True)
    npu.shutdown();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes'))
    sources=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_amp.py','npunlock-probe/torch-device/backend.cpp',
             'npunlock-probe/torch-device/_torch_intel_npu.pyd','torch_npu_amp_demo.py')
    return dict(status=row['status'] if healthy else 'failed',checks=[row],final_stats=final,
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=('all','probe','policy','scaler','contracts','mlp','conv'),default='all');parser.add_argument('--label',default='after')
    parser.add_argument('--scaler-api',choices=('legacy','standard'),default='legacy')
    args=parser.parse_args()
    if args.case=='all':
        for case in ('policy','scaler','contracts','mlp','conv'):
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--case',case,'--label',args.label,'--scaler-api',args.scaler_api],check=True)
        raise SystemExit(0)
    result=execute(args.case,args.scaler_api);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{args.label}-{args.case}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(result['status'] not in ('passed','passed_with_limitations'))

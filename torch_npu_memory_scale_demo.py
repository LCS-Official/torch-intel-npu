"""Bounded native ranges and model-memory validation; never keep a full large CPU model."""
import argparse
import ctypes
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import threading


def memory_status():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(name,ctypes.c_uint64) for name in
                 ('total','available','commit_total','commit_available','virtual_total','virtual_available','extended')]
    value=Memory();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):raise ctypes.WinError()
    return dict(physical_total=value.total,physical_available=value.available,commit_available=value.commit_available)


def training_plan(target_gib,width,layers=None):
    if not math.isfinite(target_gib) or not 0<target_gib<=14 or type(width) is not int or width<1 or width>8192:
        raise ValueError('Training probe requires target in (0,14] GiB and width in [1,8192]')
    parameter_bytes=(width*width+width)*4
    count=max(1,int(target_gib*2**30//(4*parameter_bytes))) if layers is None else layers
    if type(count) is not int or not 1<=count<=256:raise ValueError('Training probe requires 1-256 layers')
    resident=count*parameter_bytes*4
    peak=resident+4*parameter_bytes+16*1024**2
    if peak>14.25*2**30:raise ValueError('Planned training peak exceeds 14.25 GiB')
    return dict(target_gib=target_gib,width=width,layers=count,parameters=count*(width*width+width),
                resident_state_bytes=resident,estimated_peak_bytes=peak)


def execute(case,*,target_gib=2.,width=4096,layers=None):
    import torch
    import torch_intel_npu as npu
    started=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(830);checks=[]
    def guard(required=0):
        status=memory_status()
        if min(status['physical_available'],status['commit_available'])<required+3*2**30:
            raise MemoryError('Physical/commit headroom must retain 3 GiB beyond planned additional allocations')
        return status
    def ranges():
        size=65536
        for dtype in (torch.float32,torch.float16,torch.int64,torch.bool,torch.uint8):
            host=(torch.arange(size*3,dtype=torch.int64)%97).to(dtype)
            src=host.to('intel_npu');dst=torch.zeros_like(src);width=src.element_size();count=size*width
            key=('broadcast','u8',(count,),(count,));session,program=npu._session(key)
            before=npu.stats()
            for offset in (0,size,size*2):
                read=src.view(torch.uint8).narrow(0,offset*width,count)
                write=dst.view(torch.uint8).narrow(0,offset*width,count)
                npu._C.run_ranges(session._result.session,[read],[program.graph.inputs[0].name],write,program.graph.outputs[0].name)
            after=npu.stats()
            assert after['npu_calls']-before['npu_calls']==3
            assert after['host_download_copies']==before['host_download_copies']
            assert after['host_device_copies']==before['host_device_copies']
            actual=dst.cpu();assert torch.equal(actual.view(torch.uint8),host.view(torch.uint8))
            calls=npu.stats()['npu_calls']
            for fn in (lambda:npu._C.run_ranges(session._result.session,[read],[program.graph.inputs[0].name],read,program.graph.outputs[0].name),
                       lambda:npu._C.run_ranges(session._result.session,[src.view(torch.uint8).narrow(0,1,count)],
                              [program.graph.inputs[0].name],write,program.graph.outputs[0].name),
                       lambda:npu._C.run(session._result.session,[read],[program.graph.inputs[0].name],write,program.graph.outputs[0].name)):
                try:fn()
                except RuntimeError:pass
                else:raise AssertionError('Invalid range accepted')
            assert npu.stats()['npu_calls']==calls
            checks.append(dict(dtype=str(dtype),page_aligned_ranges=3,byte_exact=True,rejections=3))
            print('ranges',dtype,'passed',flush=True)
    def offsets():
        value=torch.arange(2050*1024,dtype=torch.float32).to('intel_npu');baseline=npu.stats();pieces=[]
        for i in range(2050):pieces.append(npu._slice(value,i*1024,1))
        after=npu.stats();assert after['cached_sessions']-baseline['cached_sessions']<=1
        assert after['host_download_copies']==baseline['host_download_copies']
        actual=npu._concat(pieces,(2050,),torch.float32).cpu();assert torch.equal(actual,torch.arange(2050,dtype=torch.float32)*1024)
        # Nonaligned offsets keep the original whole-buffer Slice route.
        assert npu._slice(value,1,3).cpu().tolist()==[1.,2.,3.]
        checks.append(dict(offsets=2050,new_graphs=after['cached_sessions']-baseline['cached_sessions'],unaligned_fallback=True))
    def storage():
        for dt in npu._STORAGE_DTYPES:
            guard(npu.MAX_ALLOCATION_BYTES)
            count=npu._storage_limit(dt);value=torch.empty(count,device='intel_npu',dtype=dt)
            raw=value.view(torch.uint8);tail=npu._slice(raw,raw.numel()-4096,4096).cpu()
            assert torch.count_nonzero(tail)==0
            before=npu.stats()['npu_calls']
            try:torch.empty(count+1,device='intel_npu',dtype=dt)
            except (ValueError,RuntimeError):pass
            else:raise AssertionError('Oversize allocation accepted')
            assert npu.stats()['npu_calls']==before
            checks.append(dict(dtype=str(dt),elements=count,allocation_bytes=value.nbytes,gpu_last_page_read=True))
            del value,raw,tail;gc.collect();print('storage',dt,'passed',flush=True)
    def shapes():
        shapes=[(1,)*prefix+(2**a,2**b,2**c,2**(12-a-b-c))
                for prefix in range(4) for a in range(13) for b in range(13-a) for c in range(13-a-b)]
        assert len(set(shapes))==1820
        for dtype in (torch.float32,torch.float16):
            host=torch.linspace(-1,1,4096,dtype=dtype);value=host.to('intel_npu');out=torch.empty_like(value)
            before=npu.stats()
            with torch.no_grad():
                for shape in shapes:
                    result=torch.neg(value.view(shape),out=out.view(shape))
                    assert tuple(result.shape)==shape
                    assert torch.equal(result.cpu().view(-1),-host)
                    fresh=torch.neg(value.view(shape))
                    assert fresh._base is None and tuple(fresh.shape)==shape
                    assert torch.equal(fresh.cpu().view(-1),-host)
            after=npu.stats();assert after['cached_sessions']-before['cached_sessions']==1
            checks.append(dict(dtype=str(dtype),logical_shapes=len(shapes),new_graphs=1,full_cpu_comparison=True))
            print('shapes',dtype,'passed',flush=True)
    def large():
        count=npu.MAX_DMA_ELEMENTS//2+17;guard(5*count*4)
        host=torch.linspace(-.5,.5,count,requires_grad=True);value=host.detach().to('intel_npu').requires_grad_()
        before=npu.stats();actual=(value*.25+.125).square().sum();actual.backward();after=npu.stats()
        assert after['host_download_copies']==before['host_download_copies']
        expected=(host*.25+.125).square().sum();expected.backward()
        torch.testing.assert_close(actual.detach().cpu(),expected,rtol=4e-5,atol=2e-6)
        torch.testing.assert_close(value.grad.cpu(),host.grad,rtol=4e-5,atol=2e-6)
        assert after['cached_sessions']<256,after['cached_sessions']
        checks.append(dict(elements=count,input_bytes=value.nbytes,full_cpu_gradient_comparison=True,
                           cached_graphs=after['cached_sessions'],loss=actual.item()))
    def resident():
        if not math.isfinite(target_gib) or not 0<target_gib<=14 or target_gib!=int(target_gib):
            raise ValueError('Resident probe requires an integer target of 1..14 GiB')
        blocks=int(target_gib);chunk=npu.MAX_ALLOCATION_BYTES
        assert chunk==2**30
        guard(blocks*chunk+1024**2);buffers=[]
        host=torch.linspace(-1,1,1024);source=host.to('intel_npu');before=npu.stats()
        for i in range(blocks):
            guard((blocks-i)*chunk+1024**2)
            value=torch.empty(chunk//4,device='intel_npu',dtype=torch.float32);buffers.append(value)
            for start in (0,value.numel()-1024):
                view=value.narrow(0,start,1024)
                npu._run_ranges(('neg','f32',(1024,)),[source],view)
                torch.testing.assert_close(npu._slice(value,start,1024).cpu(),-host,rtol=0,atol=0)
            assert torch.count_nonzero(npu._slice(value,value.numel()//2,1024).cpu())==0
            guard();print('resident',i+1,'GiB verified; free GiB',round(memory_status()['physical_available']/2**30,2),flush=True)
        after=npu.stats();assert sum(x.nbytes for x in buffers)==blocks*chunk
        assert after['host_device_copies']==before['host_device_copies']
        checks.append(dict(resident_bytes=blocks*chunk,buffers=blocks,allocation_bytes=chunk,
                           npu_head_and_tail_write_read=True,middle_zero_read=True,training=False))
    class Model(torch.nn.Module):
        def __init__(self,dimension,count,device='cpu'):
            super().__init__();self.layers=torch.nn.ModuleList(torch.nn.Linear(dimension,dimension,device=device) for _ in range(count))
            self.scale=1/math.sqrt(count)
        def forward(self,x):
            for i,layer in enumerate(self.layers):
                x=x+layer(torch.tanh(x))*self.scale
                if case=='training' and (i+1)%8==0:
                    guard();print('forward layer',i+1,'live GiB',npu.stats()['live_buffer_bytes']/2**30,flush=True)
            return x
    def paired():
        import copy
        for dtype in (torch.float32,torch.float16):
            reference=torch.tensor([[-2.,-1.,0.],[.5,1.,2.]],dtype=dtype,requires_grad=True)
            leaf=reference.detach().to('intel_npu').requires_grad_();value=-leaf
            assert value._base is None
            torch.relu_(value).sum().backward();expected=-reference;torch.relu_(expected).sum().backward()
            torch.testing.assert_close(value.detach().cpu(),expected,rtol=0,atol=0)
            torch.testing.assert_close(leaf.grad.cpu(),reference.grad,rtol=0,atol=0)
        cpu=Model(32,3);model=copy.deepcopy(cpu).to('intel_npu')
        host=torch.randn(2,32)*.05;data=host.to('intel_npu');opts=[torch.optim.AdamW(m.parameters(),lr=1e-4,foreach=False,fused=False) for m in (cpu,model)]
        losses=[]
        for step in range(3):
            for opt in opts:opt.zero_grad(set_to_none=True)
            expected=cpu(host).square().mean();before=npu.stats();actual=model(data).square().mean();actual.backward()
            assert npu.stats()['host_download_copies']==before['host_download_copies'];expected.backward()
            torch.testing.assert_close(actual.detach().cpu(),expected,rtol=4e-5,atol=2e-6)
            for a,b in zip(model.parameters(),cpu.parameters()):torch.testing.assert_close(a.grad.cpu(),b.grad,rtol=4e-5,atol=2e-6)
            for opt in opts:opt.step()
            for a,b in zip(model.parameters(),cpu.parameters()):torch.testing.assert_close(a.detach().cpu(),b,rtol=4e-5,atol=2e-6)
            losses.append(actual.item())
        assert losses[-1]<losses[0]
        checks.append(dict(independent_cpu_steps=3,parameters=sum(p.numel() for p in cpu.parameters()),losses=losses))
    def training():
        plan=training_plan(target_gib,width,layers);initial=guard(plan['estimated_peak_bytes']+128*1024**2)
        model=Model(width,plan['layers'],device='meta').to_empty(device='intel_npu')
        host=torch.empty(width,width)
        with torch.no_grad():
            for i,layer in enumerate(model.layers):
                host.normal_(0,1/math.sqrt(width));layer.weight.copy_(host);layer.bias.zero_()
                if (i+1)%8==0:guard();print('initialized layer',i+1,flush=True)
        del host;gc.collect()
        data=(torch.randn(1,width)*.05).to('intel_npu')
        opt=torch.optim.AdamW(model.parameters(),lr=1e-5,foreach=False,fused=False);losses=[];graphs=[];samples=0
        completed=[0]
        def backward_progress(gradient):
            completed[0]+=1
            if completed[0]%8==0:guard();print('backward layer',completed[0],flush=True)
            return gradient
        hooks=[layer.weight.register_hook(backward_progress) for layer in model.layers]
        def sample(tensor,start,count):return npu._slice(tensor.detach().view(-1),start,count).cpu()
        for step in range(2):
            opt.zero_grad(set_to_none=True);completed[0]=0;guard()
            start=npu.stats();actual=model(data).square().mean();actual.backward()
            after=npu.stats();assert after['host_download_copies']==start['host_download_copies']
            assert after['host_device_copies']==start['host_device_copies']
            assert completed[0]==plan['layers']
            loss=actual.item();assert math.isfinite(loss)
            records=[]
            # Check local AdamW updates from sampled real gradients, without a full CPU mirror.
            for p in model.parameters():
                size=min(1024,p.numel());starts=(0,) if p.ndim==1 else (0,(p.numel()//2//1024)*1024,p.numel()-size)
                for offset in starts:
                    parameter=torch.nn.Parameter(sample(p,offset,size));parameter.grad=sample(p.grad,offset,size)
                    assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
                    reference=torch.optim.AdamW([parameter],lr=1e-5,foreach=False,fused=False)
                    if step:
                        reference.state[parameter]={'step':opt.state[p]['step'].detach().cpu().clone(),
                            'exp_avg':sample(opt.state[p]['exp_avg'],offset,size),
                            'exp_avg_sq':sample(opt.state[p]['exp_avg_sq'],offset,size)}
                    reference.step();records.append((p,offset,size,parameter.detach(),reference.state[parameter]))
            remaining=max(0,plan['estimated_peak_bytes']-npu.stats()['live_buffer_bytes']);guard(remaining)
            print('optimizer step',step+1,'start; live GiB',npu.stats()['live_buffer_bytes']/2**30,flush=True)
            opt.step();guard()
            for p,offset,size,reference,state in records:
                torch.testing.assert_close(sample(p,offset,size),reference,rtol=4e-5,atol=2e-6)
                for key in ('exp_avg','exp_avg_sq'):
                    torch.testing.assert_close(sample(opt.state[p][key],offset,size),state[key],rtol=4e-5,atol=2e-6)
                samples+=size
            losses.append(loss);graphs.append(npu.stats()['cached_sessions'])
            print('training step',step+1,'loss',loss,'live GiB',npu.stats()['live_buffer_bytes']/2**30,'graphs',graphs[-1],flush=True)
        assert losses[-1]<losses[0]
        assert graphs[1]-graphs[0]<32,(graphs[0],graphs[1])
        for hook in hooks:hook.remove()
        actual_state=sum(p.nbytes+p.grad.nbytes+opt.state[p]['exp_avg'].nbytes+opt.state[p]['exp_avg_sq'].nbytes for p in model.parameters())
        assert actual_state==plan['resident_state_bytes']
        checks.append(dict(plan=plan,initial_memory=initial,final_memory=memory_status(),losses=losses,
                           cached_graphs=graphs,actual_state_bytes=actual_state,all_parameters_have_gradients=True,
                           sampled_adamw_update_elements=samples,independent_full_cpu_model=False))
    row=dict(name=case);monitor_stop=threading.Event();monitor_errors=[];minimum_memory=None;monitor=None
    if case in ('training','resident'):
        initial_memory=memory_status()
        minimum_memory={key:initial_memory[key] for key in ('physical_available','commit_available')}
        def monitor_memory():
            while not monitor_stop.wait(.25):
                try:
                    state=memory_status()
                    for key in minimum_memory:minimum_memory[key]=min(minimum_memory[key],state[key])
                except Exception as error:monitor_errors.append(str(error));return
        monitor=threading.Thread(target=monitor_memory,daemon=True);monitor.start()
    try:locals()[case]();row['status']='passed'
    except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
    monitor_stop.set()
    if monitor is not None:
        monitor.join(timeout=1)
        if monitor_errors or min(minimum_memory.values())<3*2**30:
            row.update(status='failed',error='Memory monitor failed or headroom dropped below 3 GiB',monitor_errors=monitor_errors)
    print(case,row['status'],row.get('error','')[:700],flush=True)
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes'))
    root=Path(__file__).resolve().parent
    sources=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_memory_scale_demo.py','test_torch_npu_memory_scale.py',
             'npunlock/python/npunlock/npu4000.py','npunlock/tests/fixtures/npu4000/resident/manifest.json',
             'npunlock-probe/torch-device/backend.cpp',
             'npunlock-probe/torch-device/_torch_intel_npu.pyd','npunlock/src/graphinfer/session_win32.c',
             'npunlock/include/npunlock/graphinfer.h','npunlock-probe/session-runtime/npunlock_session.dll')
    result=dict(status='passed' if healthy and row['status']=='passed' else 'failed',result=row,checks=checks,
                stats=before,shutdown=final,minimum_memory_250ms=minimum_memory,elapsed_seconds=time.perf_counter()-started,
                source_sha256={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources})
    target=root/'npunlock-probe/research/results/memory-scale';target.mkdir(exist_ok=True)
    label=f'{case}-w{width}-l{layers if layers is not None else "auto"}' if case=='training' else case
    if case=='resident':label=f'resident-{int(target_gib)}gib'
    (target/f'{label}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=('ranges','offsets','shapes','storage','large','paired','training','resident'),required=True)
    parser.add_argument('--target-gib',type=float,default=2.);parser.add_argument('--width',type=int,default=4096)
    parser.add_argument('--layers',type=int)
    args=parser.parse_args();raise SystemExit(execute(args.case,target_gib=args.target_gib,width=args.width,layers=args.layers))

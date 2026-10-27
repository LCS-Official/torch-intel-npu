"""Compare bounded Conv2d workspaces and assembly against the retained S34 path."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback


def execute(case,precision,baseline=False):
    import torch
    import torch_intel_npu as npu
    root=Path(__file__).resolve().parent;started=time.perf_counter();dtype=getattr(torch,precision)
    torch.set_num_threads(1);torch.manual_seed(835);torch.backends.mkldnn.enabled=False
    names=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_geometry.py','torch_npu_conv_capacity_demo.py',
           'npunlock-probe/research/conv_workspace_before.py','npunlock-probe/session-runtime/npunlock_session.dll',
           'npunlock-probe/torch-device/_torch_intel_npu.pyd','npunlock/tests/fixtures/npu4000/resident/manifest.json')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    if baseline:
        path=root/'npunlock-probe/research/conv_workspace_before.py'
        exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),npu.__dict__)
    checks=[]
    def upload(x):return x.detach().to('intel_npu').requires_grad_(x.requires_grad)
    def close(x,y):
        torch.testing.assert_close(x.detach().cpu(),y.detach(),equal_nan=True,
            rtol=.004 if dtype==torch.float16 else 7e-5,atol=.0005 if dtype==torch.float16 else 6e-6)
    def conv(shape,channels,kernel,stride=1,padding=0,dilation=1,groups=1):
        host=(torch.randn(shape,dtype=dtype)/16).requires_grad_()
        weight=(torch.randn(channels,shape[1]//groups,*kernel,dtype=dtype)/16).requires_grad_()
        bias=(torch.randn(channels,dtype=dtype)/32).requires_grad_()
        x,w,b=map(upload,(host,weight,bias));expected=torch.nn.functional.conv2d(host,weight,bias,stride,padding,dilation,groups)
        seed=torch.randn_like(expected)/max(16,expected.shape[-1]);gradient=upload(seed)
        before=npu.stats();actual=torch.nn.functional.conv2d(x,w,b,stride,padding,dilation,groups)
        actual.backward(gradient);after=npu.stats()
        assert after['host_download_copies']==before['host_download_copies']
        assert after['host_device_copies']==before['host_device_copies']
        expected.backward(seed);close(actual,expected)
        for a,q in ((x,host),(w,weight),(b,bias)):close(a.grad,q.grad)
        checks.append(dict(shape=list(shape),channels=channels,kernel=list(kernel),stride=stride,padding=padding,
            dilation=dilation,groups=groups,output_elements=actual.numel(),full_cpu_output_and_gradients=True,
            saved_cpu_index_entries=npu._conv_indices.cache_info().currsize))
    def workspace():conv((1,4,96,97),4,(3,3),padding=1)
    def output():conv((1,1,1,4097),257,(1,1))
    def geometry():
        for args in (((2,2,11,13),3,(3,2),(2,1),(1,1),(1,2),1),
                     ((1,4,17,19),4,(3,3),1,1,1,2)):
            conv(*args)
    def assembly():
        import weakref
        refs=[];sizes=(17,1007,1,65535,33,2049,3,65536,65536,65536,65536);host=torch.linspace(-1,1,sum(sizes))
        def values():
            offset=0
            for count in sizes:
                assert baseline or sum(ref() is not None for ref in refs)<=1
                tile=host[offset:offset+count].to('intel_npu');offset+=count;refs.append(weakref.ref(tile))
                yield tile
        before=npu.stats()
        actual=(npu._concat(list(values()),(sum(sizes),),torch.float32) if baseline else
                npu._stream_concat(values(),(sum(sizes),)))
        after=npu.stats()
        assert after['host_download_copies']==before['host_download_copies']
        assert torch.equal(actual.cpu().view(torch.int32),host.view(torch.int32))
        assert all(ref() is None for ref in refs)
        checks.append(dict(sizes=list(sizes),raw_bits_equal=True,old_tiles_released=True))
    def training():
        cpu=torch.nn.Sequential(torch.nn.Conv2d(2,3,3,padding=1),torch.nn.Tanh(),torch.nn.Conv2d(3,2,1)).to(dtype)
        model=copy.deepcopy(cpu).to('intel_npu');host=torch.randn(1,2,17,19,dtype=dtype)/16;x=upload(host)
        opts=[torch.optim.SGD(net.parameters(),lr=.001,foreach=False) for net in (cpu,model)];losses=[]
        # Force the bounded path at this small size, rather than a long large-network test.
        original=npu._CONV_WORKSPACE_ELEMENTS;npu._CONV_WORKSPACE_ELEMENTS=1
        try:
            for step in range(2):
                for opt in opts:opt.zero_grad(set_to_none=True)
                expected=cpu(host).float().square().mean();before=npu.stats();actual=model(x).float().square().mean();actual.backward()
                assert npu.stats()['host_download_copies']==before['host_download_copies'];expected.backward();close(actual,expected)
                for p,q in zip(model.parameters(),cpu.parameters()):close(p.grad,q.grad)
                for opt in opts:opt.step()
                for p,q in zip(model.parameters(),cpu.parameters()):close(p,q)
                losses.append(actual.item());print('training step',step+1,'passed',flush=True)
        finally:npu._CONV_WORKSPACE_ELEMENTS=original
        assert losses[-1]<losses[0]
        checks.append(dict(independent_cpu_steps=2,losses=losses,all_parameter_gradients_and_updates=True,forced_streaming=True))
    error=None
    try:
        if case=='geometry' and not baseline:
            original=npu._CONV_WORKSPACE_ELEMENTS;npu._CONV_WORKSPACE_ELEMENTS=1
            try:geometry()
            finally:npu._CONV_WORKSPACE_ELEMENTS=original
        else:locals()[case]()
    except Exception:error=traceback.format_exc();print(error,flush=True)
    gc.collect();stats=npu.stats();cpu_cache=npu._conv_indices.cache_info().currsize
    npu._conv_indices.cache_clear();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes',
        'deferred_graph_sessions','cached_graph_blob_bytes','live_graph_factories'))
    result=dict(status='passed' if error is None and healthy else 'failed',case=case,precision=precision,baseline=baseline,
        checks=checks,error=error,stats=stats,cpu_index_cache_entries=cpu_cache,shutdown=final,
        elapsed_seconds=time.perf_counter()-started,source_sha256=hashes)
    target=root/'npunlock-probe/research/results/conv-capacity';target.mkdir(exist_ok=True)
    (target/f'{"before" if baseline else "after"}-{case}-{precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('before' if baseline else 'after',case,precision,result['status'],'seconds',result['elapsed_seconds'],flush=True)
    return int(result['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=('workspace','output','geometry','assembly','training'))
    parser.add_argument('--precision',choices=('float32','float16'),default='float32');parser.add_argument('--baseline',action='store_true')
    args=parser.parse_args();raise SystemExit(execute(args.case,args.precision,args.baseline))

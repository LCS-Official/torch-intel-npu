"""Validate byte-sized storage caps and complete maximum MaxPool shapes."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import struct
import time
import traceback


def execute(case,precision):
    import torch
    import torch.nn.functional as F
    import torch_intel_npu as npu
    from torch_npu_memory_scale_demo import memory_status
    root=Path(__file__).resolve().parent;started=time.perf_counter();dtype=getattr(torch,precision);checks=[]
    torch.set_num_threads(1);torch.manual_seed(838)
    names=('torch_intel_npu.py','torch_npu_geometry.py','torch_npu_autograd.py','torch_npu_pool_dense.py',
        'torch_npu_storage_shape_demo.py','test_torch_npu_pool_dense.py','npunlock/python/npunlock/npu4000.py',
        'npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/_torch_intel_npu.pyd',
        'npunlock-probe/session-runtime/npunlock_session.dll','npunlock/tests/fixtures/npu4000/resident/manifest.json')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    def guard(additional):
        state=memory_status()
        if min(state['physical_available'],state['commit_available'])<additional+3*2**30:
            raise MemoryError('Probe preserves 3 GiB of physical/commit headroom')
        return state
    def close(x,y):torch.testing.assert_close(x.detach().cpu(),y,rtol=0,atol=0,equal_nan=True)
    def upload(x):return x.detach().to('intel_npu').requires_grad_(x.requires_grad)
    def contracts():
        host=torch.randn(1,2,129,131,dtype=dtype,requires_grad=True);x=upload(host)
        for kernel,stride,padding,dilation,ceil in ((3,1,1,1,False),((2,3),(2,1),(1,1),(2,2),True)):
            x.grad=None;host.grad=None
            expected,ids=F.max_pool2d(host,kernel,stride,padding,dilation,ceil,return_indices=True)
            actual,index=F.max_pool2d(x,kernel,stride,padding,dilation,ceil,return_indices=True)
            seed=torch.randn_like(expected)/32;actual.backward(upload(seed));expected.backward(seed)
            close(actual,expected);close(index,ids);close(x.grad,host.grad)
        # Force the new forward for ties, signed zero and last-NaN selection.
        host=torch.full((1,2,257,259),-0.,dtype=dtype);host[...,1,1]=float('nan');host[...,2,2]=float('nan')
        actual,index=F.max_pool2d(upload(host),3,1,1,return_indices=True)
        expected,ids=F.max_pool2d(host,3,1,1,return_indices=True);close(actual,expected);close(index,ids)
        # Manually supplied indices may lie outside the corresponding window.
        host=torch.zeros(1,1,257,257,dtype=dtype);g=torch.zeros_like(host);g[...,0,0]=.125
        ids=torch.zeros(host.shape,dtype=torch.int64);ids[...,0,0]=host.numel()-1
        expected=torch.ops.aten.max_pool2d_with_indices_backward(g,host,[3,3],[1,1],[1,1],[1,1],False,ids)
        actual=torch.ops.aten.max_pool2d_with_indices_backward(upload(g),upload(host),[3,3],[1,1],[1,1],[1,1],False,upload(ids));close(actual,expected)
        leaf=upload(torch.randn(1,1,257,257,dtype=dtype,requires_grad=True));values,ids=F.max_pool2d(leaf,3,1,1,return_indices=True)
        with torch.no_grad():ids.zero_()
        seed=torch.ones_like(values);before=npu.stats()['npu_calls']
        try:values.backward(seed)
        except RuntimeError as exc:assert 'version' in str(exc) or 'inplace' in str(exc)
        else:raise AssertionError('Dense pool saved-index version guard missing')
        assert before==npu.stats()['npu_calls']
        # Direct malformed carrier packets must return status, not access outside the source.
        header=[1,0,1,1,1,1,1,1,1,1,0,0,1,1,0,1]
        def packet(fields):
            raw=struct.pack('<16I',*fields)+struct.pack('<f',-.0)+bytes(4)
            source=torch.frombuffer(bytearray(raw),dtype=torch.float32).to('intel_npu').view(1,18)
            return npu._run(('pool-dense-f32','f32',(1,18)),[source,source],(1,18),torch.float32).cpu().view(-1)
        result=packet(header);assert result[-1]==0 and result[0].view(torch.int32).item()==-2147483648
        rejected=0
        for field,value in ((0,100),(1,1),(2,0),(6,32766),(8,0),(10,1),(12,1073741825),(14,1),(15,0)):
            invalid=header.copy();invalid[field]=value;assert packet(invalid)[-1]!=0;rejected+=1
        assert packet(header)[-1]==0
        checks.append(dict(ceil_dilation_full_cpu=True,nan_ties=True,explicit_arbitrary_indices=True,
            saved_index_version=True,malformed_packets_rejected=rejected,negative_zero_bits=True))
    def storage():
        guard(2*2**30);rows=[];marker=torch.arange(4096,dtype=torch.int32).remainder(127).to(torch.uint8).to('intel_npu')
        for dt in npu._STORAGE_DTYPES:
            count=npu._storage_limit(dt);assert count*dt.itemsize==2**30
            x=torch.empty((count,),dtype=dt,device='intel_npu');raw=x.view(torch.uint8)
            for offset in (0,2**29,2**30-4096):
                npu._range_copy(raw.narrow(0,offset,4096),marker)
                close(npu._slice(raw,offset,4096),marker.cpu())
            before=npu.stats()['npu_calls']
            try:torch.empty((count+1,),dtype=dt,device='intel_npu')
            except (ValueError,RuntimeError):pass
            else:raise AssertionError('Oversize storage accepted')
            assert before==npu.stats()['npu_calls']
            rows.append(dict(dtype=str(dt),elements=count,bytes=x.nbytes));del raw,x;gc.collect()
        checks.append(dict(maximum_storage=rows,first_middle_last_page_gpu_dma=True,oversize_before_submission=True))
    def pointwise():
        assert dtype==torch.float16;guard(4*2**30);count=npu._storage_limit(dtype)
        host=torch.full((count,),.25,dtype=dtype);x=upload(host);del host
        before=npu.stats()['npu_calls'];tick=time.perf_counter();value=torch.neg(x);elapsed=time.perf_counter()-tick
        for start in range(0,count,1048576):
            length=min(1048576,count-start)
            close(npu._slice(value,start,length),torch.full((length,),-.25,dtype=dtype))
        checks.append(dict(elements=count,bytes=x.nbytes,full_output_checked=True,negate_seconds=elapsed,
            npu_calls=npu.stats()['npu_calls']-before))
    def maximum():
        guard(7*2**30);count=npu._storage_limit(dtype);stride=count//npu.MAX_INT64_ELEMENTS
        host=torch.full((1,1,1,count),.25,dtype=dtype,requires_grad=True);x=upload(host);del host;gc.collect()
        tick=time.perf_counter();actual,indices=F.max_pool2d(x,(1,1),(1,stride),return_indices=True);forward=time.perf_counter()-tick
        assert actual.numel()==npu.MAX_INT64_ELEMENTS and indices.nbytes==2**30
        seed=torch.full(actual.shape,.125,dtype=dtype).to('intel_npu');tick=time.perf_counter();actual.backward(seed);backward=time.perf_counter()-tick
        for start in range(0,actual.numel(),1048576):
            length=min(1048576,actual.numel()-start)
            close(npu._slice(actual.view(-1),start,length),torch.full((length,),.25,dtype=dtype))
            close(npu._slice(indices.view(-1),start,length),torch.arange(start*stride,(start+length)*stride,stride,dtype=torch.int64))
        gradient=x.grad.view(-1)
        for start in range(0,count,1048576):
            length=min(1048576,count-start);pos=torch.arange(start,start+length,dtype=torch.int64)
            expected=torch.where(pos%stride==0,.125,0.).to(dtype);close(npu._slice(gradient,start,length),expected)
            if start%(16*1048576)==0:print('maximum gradient checked',start+length,'/',count,flush=True)
        checks.append(dict(source_elements=count,source_bytes=x.nbytes,output_elements=actual.numel(),
            index_bytes=indices.nbytes,stride=stride,forward_seconds=forward,backward_seconds=backward,
            full_values_indices_input_gradient_checked=True,reference='exact kernel1/stride analytic reference in CPU chunks'))
    error=None
    try:locals()[case]()
    except Exception:error=traceback.format_exc();print(error,flush=True)
    gc.collect();stats=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes',
        'deferred_graph_sessions','cached_graph_blob_bytes','live_graph_factories'))
    result=dict(status='passed' if error is None and healthy else 'failed',case=case,precision=precision,
        checks=checks,error=error,stats=stats,shutdown=final,source_sha256=hashes,elapsed_seconds=time.perf_counter()-started)
    target=root/'npunlock-probe/research/results/storage-shape';target.mkdir(exist_ok=True)
    (target/f'{case}-{precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(case,precision,result['status'],checks,flush=True);return int(result['status']!='passed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=('contracts','storage','pointwise','maximum'),required=True)
    p.add_argument('--precision',choices=('float32','float16'),default='float32');a=p.parse_args();raise SystemExit(execute(a.case,a.precision))

"""Wide single-plane selection metadata, checked against complete CPU results."""
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
    torch.set_num_threads(1);dtype=getattr(torch,precision);root=Path(__file__).resolve().parent
    started=time.perf_counter();checks=[]
    names=('torch_intel_npu.py','torch_npu_geometry.py','torch_npu_selection_capacity_demo.py',
        'test_torch_npu_selection_capacity.py','npunlock/tests/fixtures/npu4000/resident_selection.c',
        'npunlock/tests/fixtures/npu4000/resident/manifest.json',
        'npunlock/tests/fixtures/npu4000/resident/selection-backward-u32.elf',
        'npunlock-probe/session-runtime/npunlock_session.dll','npunlock-probe/torch-device/_torch_intel_npu.pyd')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    def upload(x):return x.detach().to('intel_npu').requires_grad_(x.requires_grad)
    def close(x,y):torch.testing.assert_close(x.detach().cpu(),y.detach(),rtol=0,atol=0,equal_nan=True)
    def packet():
        width,per=16777219,16777221;base=width-2
        fields=[3,per-1,per,width,base,4,0,2*width]
        ids=torch.tensor([width-1,1,1],dtype=torch.int64)
        grad=torch.tensor([-.0,1.,-1.],dtype=dtype)
        initial=torch.tensor([-.0,2.,3.,2048.],dtype=torch.float32)
        def run(header,labels):
            size=max(9+3*len(grad),header[5]+1)
            payload=struct.pack('<8I',*header)+grad.float().numpy().tobytes()+labels.numpy().tobytes()
            data=torch.frombuffer(bytearray(payload+bytes(4*size-len(payload))),dtype=torch.float32)
            seed=torch.zeros(size);seed[:len(initial)]=initial
            return npu._run(('selection-backward-u32','f32',(1,size)),
                [upload(data.view(1,size)),upload(seed.view(1,size))],(1,size),torch.float32).cpu().view(-1)
        for accumulate in (False,True):
            fields[6]=int(dtype==torch.float16)+2*int(not accumulate)
            expected=initial.clone()
            for value,pos in zip(grad,[1,3,3]):
                value=value.float() if not accumulate else expected[pos]+value.float()
                expected[pos]=value.half().float() if accumulate and dtype==torch.float16 else value
            actual=run(fields,ids);assert actual[-1]==0
            assert torch.equal(actual[:4].view(torch.uint8),expected.view(torch.uint8))
        rejected=0
        for field,value in ((0,0),(1,2*per),(2,0),(3,0),(4,2*width),(6,8),(7,1073741825)):
            invalid=fields.copy();invalid[field]=value
            assert run(invalid,ids)[-1]!=0;rejected+=1
        fields[6]=0
        for value in (-2,-1,width,2**40):
            invalid=ids.clone();invalid[0]=value
            assert run(fields,invalid)[-1]!=0;rejected+=1
        fields[6]=4;labels=ids.clone();labels[0]=-1
        actual=run(fields,labels);assert actual[-1]==0
        assert torch.equal(actual[:3].view(torch.uint8),initial[:3].view(torch.uint8))
        fields[6]=0;assert run(fields,ids)[-1]==0
        checks.append(dict(raw_odd_metadata_above_2p24=True,duplicate_half_rounding=True,
            negative_zero_assignment_bits=True,malformed_packets_rejected=rejected,padding_and_recovery=True))
    def pool():
        size=16777219;host=torch.zeros(1,1,1,size,dtype=dtype)
        host[...,1::32768]=1.;host[...,2::32768]=.5;host.requires_grad_();x=upload(host)
        expected,ids=F.max_pool2d(host,(1,3),(1,32768),return_indices=True)
        actual,index=F.max_pool2d(x,(1,3),(1,32768),return_indices=True)
        g=(torch.arange(expected.numel()).remainder(17).to(dtype)/32).reshape(expected.shape)
        actual.backward(upload(g));expected.backward(g)
        close(actual,expected);close(index,ids);close(x.grad,host.grad)
        assert index.cpu().view(-1)[-1]==16777217
        checks.append(dict(shape=list(host.shape),source_elements=size,output_elements=actual.numel(),
            final_index=16777217,full_cpu_values_indices_and_gradient=True))
    def extrema():
        size=16777219;host=torch.zeros(size,dtype=dtype);host[5]=-9;host[-2:]=9;host.requires_grad_();x=upload(host)
        for kind in ('max','min'):
            host.grad=None;x.grad=None
            expected,ids=getattr(torch,kind)(host,0);actual,index=getattr(torch,kind)(x,0)
            actual.backward();expected.backward();close(actual,expected);close(index,ids);close(x.grad,host.grad)
        with torch.no_grad():
            host[-2:]=float('nan');x.copy_(host)
            close(torch.argmax(x),torch.argmax(host));close(torch.argmin(x),torch.argmin(host))
        checks.append(dict(source_elements=size,first_tie_and_nan_index=16777217,
            full_cpu_paired_gradient=True))
    def dense():
        count=16798727;host=torch.zeros(count,dtype=dtype)
        positions=[0,21503,21504,16777217,count-1];values=[2048.,1.,-2048.,.5,.25]
        host[positions]=torch.tensor(values,dtype=dtype)
        ids=torch.zeros(count,dtype=torch.int64);initial=torch.tensor([.125,-.0,5.],dtype=dtype)
        expected=initial.clone()
        for value in values:expected[0]=(expected[0].float()+value).to(dtype)
        result=npu._selection_backward(upload(host),upload(ids),(3,),count,3,initial_value=upload(initial))
        close(result,expected)
        assert torch.equal(result.cpu()[1:].view(torch.uint8),expected[1:].view(torch.uint8))
        checks.append(dict(gradient_elements=count,destination_elements=3,initial_value=True,
            expected=expected.tolist(),sequential_dtype_reference=True))
    error=None
    try:locals()[case]()
    except Exception:error=traceback.format_exc();print(error,flush=True)
    gc.collect();stats=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes',
        'deferred_graph_sessions','cached_graph_blob_bytes','live_graph_factories'))
    result=dict(status='passed' if error is None and healthy else 'failed',case=case,precision=precision,
        checks=checks,error=error,stats=stats,shutdown=final,elapsed_seconds=time.perf_counter()-started,source_sha256=hashes)
    target=root/'npunlock-probe/research/results/selection-capacity';target.mkdir(exist_ok=True)
    (target/f'{case}-{precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(case,precision,result['status'],'seconds',result['elapsed_seconds'],flush=True)
    return int(result['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=('packet','pool','extrema','dense'))
    parser.add_argument('--precision',choices=('float32','float16'),default='float32')
    args=parser.parse_args();raise SystemExit(execute(args.case,args.precision))

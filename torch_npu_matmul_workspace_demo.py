"""S27: rolling FP32 compensation, wide-K GEMM and rank/strided MatMul training."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(727)
    dtype=getattr(torch,precision)
    started=time.perf_counter();errors=[]
    def upload(x):
        return x.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)
    def close(actual,expected,*,gradient=False):
        a,b=actual.detach().cpu(),expected.detach().cpu()
        torch.testing.assert_close(a,b,rtol=4e-5 if dtype==torch.float32 else .004,
                                   atol=2e-6 if dtype==torch.float32 else .001,equal_nan=True)
        if gradient and b.numel():
            delta=(a.double()-b.double()).norm().item();denom=b.double().norm().item()
            error=delta/denom if denom else 0. if delta==0 else float('inf')
            assert error<=(1e-4 if dtype==torch.float32 else .01),error
            errors.append(error)
    def reject(fn):
        before=npu.stats()['npu_calls']
        try:fn()
        except (ValueError,RuntimeError):assert npu.stats()['npu_calls']==before
        else:raise AssertionError('Missing metadata rejection')

    def stream():
        # Compare the new rolling state against the existing full Neumaier reduction.
        shape=(257,129);host=[torch.full(shape,v) for v in (.125,.25,-.125)]
        special=((1e8,1.,-1e8),(2.**-149,2.**-149,-2.**-149),
                 (torch.finfo(torch.float32).max,torch.finfo(torch.float32).max,-torch.finfo(torch.float32).max),
                 (float('inf'),-float('inf'),0.),(float('nan'),1.,2.),(-0.,-0.,-0.),(1e-37,-1e-37,2.**-149))
        for index,values in enumerate(special):
            for part,value in zip(host,values):part.view(-1)[index]=value
        actual=npu._stream_sum((upload(v) for v in host),shape)
        legacy=npu._reduce(npu._concat([upload(v) for v in host],(3,*shape),torch.float32),0,kind='sum')
        a,b=actual.cpu(),legacy.cpu();nan=torch.isnan(b)
        assert torch.equal(torch.isnan(a),nan)
        assert torch.equal(a.reshape(-1).view(torch.int32)[~nan.reshape(-1)],b.reshape(-1).view(torch.int32)[~nan.reshape(-1)])
        assert a.view(-1)[0].item()==1. and a.view(-1)[1].item()==2.**-149
        out=torch.empty(shape,dtype=torch.float32,device='intel_npu')
        assert npu._stream_sum((upload(v) for v in host),shape,out=out) is out
        close(out,actual)
        reject(lambda:npu._stream_sum(iter(()),(1,1)))
        from npunlock.npu4000 import inspect
        for count in (32768,385):
            _,program=npu._session(('stream-sum','f32',(1,2*count)))
            evidence=inspect(program.graph_blob,memref_count=3)
            assert len(evidence['invocations'])==1
            assert all(ref['elements']==2*count for ref in evidence['invocations'][0]['memrefs'])
        return dict(shape=shape,partials=3,two_output_tiles=True,full_reference_bits_equal_except_nan_payload=True,
                    cancellation_and_subnormal_exact=True,inf_nan_categories=True,complete_state_carrier=True)

    def stream_large():
        shape=(257,257);parts=1025;logical=parts*257*257
        assert logical>npu._storage_limit(torch.float32)
        def values():
            for index in range(parts):
                value=(1e8,1.,-1e8)[index%3] if index<1023 else 0.
                yield npu._broadcast(npu._scalar(value,torch.float32),shape)
        actual=npu._stream_sum(values(),shape)
        assert torch.equal(actual.cpu(),torch.full(shape,341.,dtype=torch.float32))
        assert npu.stats()['peak_buffer_bytes']<16*1024*1024
        return dict(shape=shape,partials=parts,logical_stack_bytes=logical*4,
                    old_stack_exceeds_single_allocation=True,known_cancellation_exact=True,
                    tracked_peak_below_16mib=True,synthetic_partials_not_full_gemm=True)

    def wide():
        hosts=[(torch.randn(shape,dtype=dtype)*.125).requires_grad_() for shape in ((3,65539),(65539,5))]
        values=[upload(v) for v in hosts];seed=torch.randn(3,5,dtype=dtype)*.01
        actual=torch.mm(*values);expected=torch.mm(*hosts);close(actual,expected)
        actual.backward(upload(seed));expected.backward(seed)
        for a,b in zip(values,hosts):close(a.grad,b.grad,gradient=True)
        out=torch.empty(3,5,dtype=dtype,device='intel_npu')
        with torch.no_grad():
            assert torch.mm(*values,out=out) is out
            close(out,expected)
        return dict(m=3,k=65539,n=5,forward_and_both_gradients=True,out=True,fp32_partial_compensation=True)

    def batch():
        configurations=(((2,1,3,2,3),(1,4,1,3,5)),
                        ((1,)*61+(2,2,3),(1,)*62+(3,4)),
                        ((1,)*61+(2,2,3),(3,)))
        for left,right in configurations:
            hosts=[(torch.randn(shape,dtype=dtype)*.125).requires_grad_() for shape in (left,right)]
            values=[upload(v) for v in hosts]
            expected=torch.matmul(*hosts);actual=torch.matmul(*values);close(actual,expected)
            seed=torch.randn(expected.shape,dtype=dtype)*.01
            actual.backward(upload(seed));expected.backward(seed)
            for a,b in zip(values,hosts):close(a.grad,b.grad,gradient=True)
            out=torch.empty(expected.shape,dtype=dtype,device='intel_npu')
            with torch.no_grad():
                assert torch.matmul(*values,out=out) is out
                close(out,expected)
        return dict(cases=3,max_rank=64,broadcast_gradients=True,vector_promotion=True,out=True)

    def views():
        hosts=[torch.randn(shape,dtype=dtype).requires_grad_() for shape in ((6,6),(8,7))]
        values=[upload(v) for v in hosts]
        actual=torch.mm(values[0][1:5,::2],values[1][1:7:2,1:6])
        expected=torch.mm(hosts[0][1:5,::2],hosts[1][1:7:2,1:6]);close(actual,expected)
        seed=torch.randn(expected.shape,dtype=dtype)*.01
        actual.backward(upload(seed));expected.backward(seed)
        for a,b in zip(values,hosts):close(a.grad,b.grad,gradient=True)
        host=torch.randn(1,3,dtype=dtype).requires_grad_();value=upload(host)
        weight=torch.randn(3,2,dtype=dtype);resident=upload(weight)
        actual=torch.mm(value.expand(4,3),resident);expected=torch.mm(host.expand(4,3),weight)
        close(actual,expected);actual.sum().backward();expected.sum().backward();close(value.grad,host.grad,gradient=True)
        host=torch.arange(9,dtype=dtype).requires_grad_();value=upload(host)
        weight=torch.tensor([.25,-.5,.125],dtype=dtype)
        actual=torch.dot(value[1:7:2],upload(weight));expected=torch.dot(host[1:7:2],weight)
        close(actual,expected);actual.backward();expected.backward();close(value.grad,host.grad,gradient=True)
        for left,right in (((0,65539),(65539,5)),((2,65539),(65539,0)),((0,2,3),(1,3,4))):
            a,b=upload(torch.empty(left,dtype=dtype)),upload(torch.empty(right,dtype=dtype))
            before=npu.stats()['npu_calls'];result=torch.matmul(a,b)
            assert npu.stats()['npu_calls']==before and result.numel()==0
        for left,right in (((2,0),(0,3)),((0,2,3),(1,3,4)),((2,3,0),(1,0,4)),
                           ((2,1,3,0),(1,4,0,5)),((0,),(0,)),((2,3,0),(0,))):
            hosts=[torch.empty(shape,dtype=dtype,requires_grad=True) for shape in (left,right)]
            values=[upload(v) for v in hosts]
            actual=torch.matmul(*values);expected=torch.matmul(*hosts);close(actual,expected)
            actual.sum().backward();expected.sum().backward()
            for a,b in zip(values,hosts):close(a.grad,b.grad,gradient=True)
            with torch.no_grad():
                out=torch.empty(expected.shape,dtype=dtype,device='intel_npu')
                assert torch.matmul(*values,out=out) is out
                close(out,expected)
        for left,right in (((2,0,3),(2,3,4)),((2,3,4),(2,4,0)),((0,2,3),(0,3,4))):
            a,b=upload(torch.empty(left,dtype=dtype)),upload(torch.empty(right,dtype=dtype))
            before=npu.stats()['npu_calls'];result=torch.bmm(a,b)
            assert npu.stats()['npu_calls']==before and result.numel()==0
        hosts=[torch.empty(shape,dtype=dtype,requires_grad=True) for shape in ((2,3,0),(2,0,4))]
        values=[upload(v) for v in hosts]
        actual=torch.bmm(*values);expected=torch.bmm(*hosts);close(actual,expected)
        actual.sum().backward();expected.sum().backward()
        for a,b in zip(values,hosts):close(a.grad,b.grad,gradient=True)
        matrix=upload(torch.ones(2,2,dtype=dtype))
        reject(lambda:torch.mm(matrix,matrix,out=matrix))
        return dict(offset_step_views=True,overlap_reads=True,partial_dot=True,base_gradients=True,
                    empty_m_n_batch_no_submission=True,zero_k_and_empty_batch_backward=True,
                    zero_k_broadcast_and_vector_out_backward=True,
                    public_bmm_empty_and_zero_k=True,out_alias_rejected=True)

    def train():
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wide=torch.nn.Linear(65539,3,dtype=dtype)
                self.head=torch.nn.Parameter(torch.randn(1,1,2,3,4,dtype=dtype)*.1)
            def forward(self,x):
                return torch.matmul(torch.tanh(self.wide(x)),self.head).mean((0,1,2,3))
        cpu=Model();device=copy.deepcopy(cpu).to('intel_npu')
        host=(torch.randn(1,2,1,2,65539,dtype=dtype)*.125).requires_grad_();value=upload(host)
        target=torch.tensor([.2,-.1,.05,-.15],dtype=dtype);labels=upload(target)
        opts=[torch.optim.SGD(net.parameters(),lr=.1,foreach=False,fused=False) for net in (cpu,device)]
        losses=[];graphs=[]
        for step in range(2):
            for opt in opts:opt.zero_grad(set_to_none=True)
            host.grad=value.grad=None
            actual=torch.nn.functional.mse_loss(device(value),labels)
            expected=torch.nn.functional.mse_loss(cpu(host),target);close(actual,expected)
            actual.backward();expected.backward();close(value.grad,host.grad,gradient=True)
            for a,b in zip(device.parameters(),cpu.parameters()):close(a.grad,b.grad,gradient=True)
            for opt in opts:opt.step()
            for a,b in zip(device.parameters(),cpu.parameters()):close(a,b)
            losses.append(actual.item());graphs.append(npu.stats()['cached_sessions'])
            print('training step',step+1,'loss',losses[-1],'graphs',graphs[-1],flush=True)
        assert losses[-1]<losses[0] and graphs[-1]==graphs[0]
        return dict(parameters=sum(p.numel() for p in device.parameters()),input_shape=list(host.shape),
                    wide_k=65539,rank5_matmul=True,independent_steps=2,losses=losses,cached_graphs=graphs)

    row=dict(name=case)
    try:row.update({'stream':stream,'stream-large':stream_large,'wide':wide,'batch':batch,'views':views,'train':train}[case]());row['status']='passed'
    except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
    row['max_gradient_relative_l2']=max(errors,default=0.)
    print(case,precision,row['status'],row.get('error','')[:320],flush=True)
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes'))
    root=Path(__file__).resolve().parent
    sources=('torch_intel_npu.py','torch_npu_compat_extended.py','torch_npu_matmul_workspace_demo.py','test_torch_npu_matmul_workspace.py',
             'npunlock/tests/fixtures/npu4000/resident_stream_sum.c','npunlock/tests/fixtures/npu4000/resident/manifest.json')
    report=dict(status='passed' if healthy and row['status']=='passed' else 'failed',case=case,precision=precision,
                checks=[row],stats=before,shutdown=final,elapsed_seconds=time.perf_counter()-started,
                source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources})
    out=root/'npunlock-probe/research/results/matmul-workspace';out.mkdir(exist_ok=True)
    (out/f'{case}-{precision}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return int(report['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('stream','stream-large','wide','batch','views','train'),required=True)
    parser.add_argument('--precision',choices=('float32','float16'),default='float32')
    args=parser.parse_args();raise SystemExit(execute(args.case,args.precision))

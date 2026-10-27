"""Larger embedding, indexed gradients and sorting; preserve only the report."""
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
    root=Path(__file__).resolve().parent;started=time.perf_counter()
    torch.set_num_threads(1);torch.manual_seed(834);dtype=getattr(torch,precision);checks=[]
    names=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_compat_extended.py','torch_npu_sort.py',
           'torch_npu_index_capacity_demo.py','test_torch_npu_operator_capacity.py',
           'npunlock-probe/session-runtime/npunlock_session.dll','npunlock-probe/torch-device/_torch_intel_npu.pyd',
           'npunlock/tests/fixtures/npu4000/resident/manifest.json')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    def upload(value):return value.detach().to('intel_npu').requires_grad_(value.requires_grad)
    def close(actual,expected,exact=False):
        torch.testing.assert_close(actual.detach().cpu(),expected.detach(),equal_nan=True,
            rtol=0 if exact else .004 if dtype==torch.float16 else 4e-5,
            atol=0 if exact else .001 if dtype==torch.float16 else 2e-6)
    def embedding():
        for rows,width,count in ((65537,17,17),(4,1,32769)):
            host=(torch.randn(rows,width,dtype=dtype)/8).requires_grad_()
            ids=(torch.arange(count,dtype=torch.int64)%4);ids[1]=rows-1;ids[-2]=rows-1
            indices=upload(ids);device=upload(host)
            seed=(torch.randn(count,width,dtype=dtype)/32 if width>1 else
                  ((torch.arange(count)%2).to(dtype)*2-1).view(-1,1)/1024)
            seed[ids==0]=float('nan')
            expected=torch.nn.functional.embedding(ids,host,padding_idx=0);expected.backward(seed)
            before=npu.stats();actual=torch.nn.functional.embedding(indices,device,padding_idx=0)
            actual.backward(upload(seed));after=npu.stats()
            close(actual,expected,True);close(device.grad,host.grad,True)
            assert after['host_device_copies']==before['host_device_copies']
            assert torch.count_nonzero(device.grad.cpu()[0])==0
            checks.append(dict(rows=rows,width=width,indices=count,source_elements=host.numel(),
                output_elements=actual.numel(),full_cpu_forward_and_gradient=True,bit_exact=True,
                padding_nan_isolated=True,validation_downloads=after['host_download_copies']-before['host_download_copies']))
            print('embedding',rows,width,count,'passed',flush=True)
        # Invalid high words and negatives must reject, then valid execution must recover.
        data=upload(torch.arange(65537,dtype=dtype).view(-1,1))
        for bad in (-1,65537,2**32,2**63-1):
            try:torch.nn.functional.embedding(upload(torch.tensor([0,bad])),data)
            except (IndexError,RuntimeError):pass
            else:raise AssertionError('Large embedding accepted invalid INT64 index')
        close(torch.nn.functional.embedding(upload(torch.tensor([0,1])),data),torch.arange(2,dtype=dtype).view(2,1),True)
        checks.append(dict(invalid_indices=4,context_recovered=True))
    def indexing():
        host=(torch.randn(17,65537,dtype=dtype)/8).requires_grad_();ids=torch.tensor([65536,0,7,7,1])
        device=upload(host);indices=upload(ids);expected=torch.index_select(host,1,ids)
        seed=torch.randn_like(expected)/32;expected.backward(seed)
        actual=torch.index_select(device,1,indices);actual.backward(upload(seed))
        close(actual,expected,True);close(device.grad,host.grad,True)
        # Exercise the enlarged legacy internal route as well as the public S25 route.
        with torch.no_grad():
            close(npu._index_select(device,1,indices),expected,True)
            close(npu._index_backward(upload(seed),indices,tuple(host.shape),1),host.grad,True)
        checks.append(dict(shape=list(host.shape),indices=5,nonfirst_axis=True,full_cpu_forward_and_gradient=True,
                           public_and_internal_routes=True,bit_exact=True))
    def columns():
        rows,width=(131073,128) if dtype==torch.float32 else (65537,17)
        old_limit=npu.MAX_TENSOR_ELEMENTS
        try:
            if dtype==torch.float16:npu.MAX_TENSOR_ELEMENTS=131074
            host=(torch.randn(rows,width,dtype=dtype)/8).requires_grad_();ids=torch.tensor([rows-1,0,rows-1])
            device=upload(host);indices=upload(ids);seed=torch.randn(3,width,dtype=dtype)/32
            expected=torch.nn.functional.embedding(ids,host,padding_idx=0);expected.backward(seed)
            actual=torch.nn.functional.embedding(indices,device,padding_idx=0);actual.backward(upload(seed))
            close(actual,expected,True);close(device.grad,host.grad,True)
            checks.append(dict(rows=rows,width=width,source_elements=host.numel(),metadata_limit=npu.MAX_TENSOR_ELEMENTS,
                               column_tiles=(width+(npu.MAX_TENSOR_ELEMENTS//rows)-1)//(npu.MAX_TENSOR_ELEMENTS//rows),
                               full_cpu_forward_and_gradient=True,bit_exact=True,lowered_budget=dtype==torch.float16))
        finally:npu.MAX_TENSOR_ELEMENTS=old_limit
    def sorting():
        rows,width=4097,129
        host=((torch.arange(rows*width).reshape(rows,width)*37)%width).to(dtype)/256
        host.requires_grad_();device=upload(host);expected,ids=torch.sort(host,stable=True)
        seed=torch.linspace(-.125,.125,width,dtype=dtype).expand(rows,width).clone()
        expected.backward(seed);actual,index=torch.sort(device,stable=True);actual.backward(upload(seed))
        close(actual,expected,True);close(index,ids,True);close(device.grad,host.grad,True)
        with torch.no_grad():
            top,chosen=torch.topk(device,width)
            close(top,torch.sort(host.detach(),descending=True)[0],True)
            assert chosen.shape==device.shape
            values=torch.empty_like(device);indices=torch.empty_like(index)
            got=torch.sort(device,stable=True,out=(values,indices))
            assert got.values is values and got.indices is indices;close(values,expected,True);close(indices,ids,True)
        checks.append(dict(sort_elements=host.numel(),topk_output_elements=top.numel(),
                           full_cpu_values_indices_and_gradient=True,exact_out=True,bit_exact=True))
        # More than the old million-element topk input, but a small selected output.
        host=(torch.arange(8193*129).reshape(8193,129)%129).to(dtype)/256
        host.requires_grad_();device=upload(host);actual,index=torch.topk(device,1)
        expected,ids=torch.topk(host,1);actual.sum().backward();expected.sum().backward()
        close(actual,expected,True);close(index,ids,True);close(device.grad,host.grad,True)
        checks.append(dict(topk_input_elements=host.numel(),k=1,full_cpu_gradient=True,bit_exact=True))
    def integers():
        for typ in (torch.int64,torch.bool):
            base=torch.arange(65537*8).reshape(65537,8)%7
            host=(base.to(typ) if typ==torch.bool else base*2**54-2**62)
            device=upload(host);expected,ids=torch.sort(host,stable=True);actual,index=torch.sort(device,stable=True)
            close(actual,expected,True);close(index,ids,True)
            checks.append(dict(dtype=str(typ),elements=host.numel(),stable_ties=True,raw_int64=True,bit_exact=True))
    def training():
        cpu=torch.nn.Sequential(torch.nn.Embedding(65537,17,padding_idx=0),torch.nn.Flatten(1),torch.nn.Linear(4*17,5)).to(dtype)
        model=copy.deepcopy(cpu).to('intel_npu');ids=torch.tensor([[0,65536,7,7],[4,5,65536,0]])
        data=upload(ids);target=torch.randn(2,5)*.1
        optimizers=[torch.optim.SGD(net.parameters(),lr=.001,foreach=False) for net in (cpu,model)];losses=[]
        for step in range(2):
            for opt in optimizers:opt.zero_grad(set_to_none=True)
            expected=(cpu(ids).float()-target).square().mean();actual=(model(data).float()-upload(target)).square().mean()
            expected.backward();actual.backward();close(actual,expected)
            for p,q in zip(model.parameters(),cpu.parameters()):close(p.grad,q.grad)
            for opt in optimizers:opt.step()
            for p,q in zip(model.parameters(),cpu.parameters()):close(p,q)
            losses.append(actual.item());print('training step',step+1,'passed',flush=True)
        assert losses[-1]<losses[0]
        checks.append(dict(parameters=sum(p.numel() for p in cpu.parameters()),independent_cpu_steps=2,
                           full_parameter_gradient_and_update=True,losses=losses))
    error=None
    try:locals()[case]()
    except Exception:error=traceback.format_exc();print(error,flush=True)
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes',
        'deferred_graph_sessions','cached_graph_blob_bytes','live_graph_factories'))
    report=dict(status='passed' if error is None and healthy else 'failed',case=case,precision=precision,checks=checks,
                error=error,stats=before,shutdown=final,source_sha256=hashes,elapsed_seconds=time.perf_counter()-started)
    target=root/'npunlock-probe/research/results/index-capacity';target.mkdir(exist_ok=True)
    (target/f'{case}-{precision}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(case,precision,report['status'],'seconds',report['elapsed_seconds'],flush=True)
    return int(report['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=('embedding','indexing','columns','sorting','integers','training'))
    parser.add_argument('--precision',choices=('float16','float32'),default='float32')
    args=parser.parse_args();raise SystemExit(execute(args.case,args.precision))

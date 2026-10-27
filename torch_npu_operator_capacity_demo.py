"""Hardware controls beyond legacy operator workspaces, with bounded host data."""
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
    torch.set_num_threads(1);torch.manual_seed(731);dtype=getattr(torch,precision)
    started=time.perf_counter();checks=[]
    def close(actual,expected):
        torch.testing.assert_close(actual.detach().cpu(),expected.detach(),rtol=.004 if dtype==torch.float16 else 4e-5,
                                   atol=.001 if dtype==torch.float16 else 2e-6,equal_nan=True)
    def upload(value):return value.detach().to('intel_npu').requires_grad_(value.requires_grad)
    def resident(before):
        after=npu.stats()
        for key in ('host_download_copies','host_device_copies','host_pack_copies'):assert after[key]==before[key],(key,before,after)
    def conv():
        length,channels=16385,1024
        x=torch.linspace(.5,1.,length,dtype=dtype).view(1,1,1,length).requires_grad_()
        w=torch.full((channels,1,1,1),.03125,dtype=dtype,requires_grad=True)
        b=torch.linspace(0.,.03125,channels,dtype=dtype).requires_grad_();dx,dw,db=(upload(v) for v in (x,w,b))
        with torch.backends.mkldnn.flags(enabled=False):expected=torch.nn.functional.conv2d(x,w,b)
        gradient=torch.full_like(expected,1./1024);dg=upload(gradient)
        before=npu.stats();actual=torch.nn.functional.conv2d(dx,dw,db);actual.backward(dg);resident(before)
        with torch.backends.mkldnn.flags(enabled=False):expected.backward(gradient)
        close(actual,expected)
        for a,ref in ((dx,x),(dw,w),(db,b)):close(a.grad,ref.grad)
        checks.append(dict(output_elements=actual.numel(),input_elements=x.numel(),full_cpu_output_and_three_gradients=True))
    def encoder():
        layer=torch.nn.TransformerEncoderLayer(2,1,dim_feedforward=16384,dropout=0.,batch_first=True,
                                              activation='relu',norm_first=True,dtype=dtype).eval()
        model=copy.deepcopy(layer).to('intel_npu').eval();host=torch.randn(1,1025,2,dtype=dtype)*.1;data=upload(host)
        def run(src,mod):
            return torch._transformer_encoder_layer_fwd(src,2,1,mod.self_attn.in_proj_weight,mod.self_attn.in_proj_bias,
                mod.self_attn.out_proj.weight,mod.self_attn.out_proj.bias,False,True,mod.norm1.eps,
                mod.norm1.weight,mod.norm1.bias,mod.norm2.weight,mod.norm2.bias,
                mod.linear1.weight,mod.linear1.bias,mod.linear2.weight,mod.linear2.bias,None,None)
        with torch.no_grad():
            expected=run(host,layer);before=npu.stats();actual=run(data,model);resident(before);close(actual,expected)
        checks.append(dict(ffn_elements=1025*16384,parameters=sum(p.numel() for p in layer.parameters()),
                           full_cpu_fused_encoder_output=True,inference_only=True))
    def dma():
        a=torch.arange(16777216,dtype=torch.float32).remainder(97).to(dtype);b=torch.linspace(-2,2,17,dtype=dtype)
        da,db=upload(a),upload(b);before=npu.stats()
        results=npu._concat_many(((da,db),(db,da)),((a.numel()+17,),(a.numel()+17,)),dtype);resident(before)
        for actual,expected in zip(results,(torch.cat((a,b)),torch.cat((b,a)))):
            assert torch.equal(actual.cpu().view(torch.uint8),expected.view(torch.uint8))
        checks.append(dict(output_elements_each=a.numel()+17,outputs=2,byte_exact=True))
    def masked():
        if dtype!=torch.float16:raise ValueError('Native rounded masked-softmax test requires float16')
        x=torch.randn(256,4100,dtype=dtype)*.1;mask=torch.arange(x.numel()).reshape(x.shape)%11==0;mask[::23]=True
        dx,dm=upload(x),upload(mask);expected=torch.ops.aten._masked_softmax(x,mask,-1,2)
        before=npu.stats();actual=npu._native_half_masked_softmax(dx,dm);resident(before);close(actual,expected)
        checks.append(dict(elements=x.numel(),all_masked_rows=True,cpu_native_half_reference=True))
    row=dict(name=case)
    try:locals()[case]();row['status']='passed'
    except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
    print(case,precision,row['status'],row.get('error','')[:650],flush=True)
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes'))
    root=Path(__file__).resolve().parent;sources=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_geometry.py',
        'torch_npu_operator_capacity_demo.py','test_torch_npu_operator_capacity.py',
        'npunlock/tests/fixtures/npu4000/resident/manifest.json')
    result=dict(status='passed' if healthy and row['status']=='passed' else 'failed',case=case,precision=precision,
                result=row,checks=checks,stats=before,shutdown=final,elapsed_seconds=time.perf_counter()-started,
                source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources})
    target=root/'npunlock-probe/research/results/operator-capacity';target.mkdir(exist_ok=True)
    (target/f'{case}-{precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=('conv','encoder','dma','masked'),required=True)
    parser.add_argument('--precision',choices=('float32','float16'),default='float32')
    args=parser.parse_args();raise SystemExit(execute(args.case,args.precision))

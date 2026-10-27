"""Complete-row softmax tiling and larger SDPA with global causal positions."""
import argparse
import gc
import hashlib
import json
from math import prod
from pathlib import Path
import struct
import subprocess
import sys
import traceback
from torch_npu_sdpa_dropout_demo import keep_mask,reference

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/attention-rows'
PRECISIONS={'rows':('float16','float32'),'sdpa':('float16','float32'),
            'native':('float16',),'contracts':('float32',)}


def execute(case,precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.manual_seed(233)
    dtype=getattr(torch,precision);checks=[]
    def upload(value):return value.detach().to('intel_npu',memory_format=torch.contiguous_format).requires_grad_(value.requires_grad)
    def close(actual,expected,exact=False):
        torch.testing.assert_close(actual.detach().cpu(),expected.detach(),equal_nan=True,
            rtol=0 if exact else .003 if actual.dtype==torch.float16 else 3e-5,
            atol=0 if exact else .0005 if actual.dtype==torch.float16 else 3e-6)
    def no_download(before):
        now=npu.stats()
        assert now['host_download_copies']==before['host_download_copies']
        assert now['host_device_copies']==before['host_device_copies']
    def check(name,function):
        gc.collect();before=npu.stats();row=dict(name=name,status='passed')
        try:row.update(function())
        except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
        gc.collect();after=npu.stats();row['npu_calls']=after['npu_calls']-before['npu_calls']
        row['live_buffer_delta']=after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
        if row['live_buffer_delta'] or after['context_failed']:row.update(status='failed',lifecycle_error='Buffer or context')
        checks.append(row);print(name,row['status'],row.get('error','')[:800],flush=True)

    def rows():
        count=0
        for shape in ((257,257),(256,4096),(513,129)):
            for logarithmic in (False,True):
                host=(torch.randn(shape,dtype=dtype)/4).requires_grad_()
                if shape==(513,129):
                    with torch.no_grad():
                        host[507].fill_(float('-inf'));host[508,0]=float('nan');host[509,1]=float('inf')
                x=upload(host);operation=torch.log_softmax if logarithmic else torch.softmax
                expected=operation(host,-1);seed=torch.randn_like(host)/8;gradient=upload(seed)
                before=npu.stats();actual=operation(x,-1);actual.backward(gradient);no_download(before)
                expected.backward(seed);close(actual,expected);close(x.grad,host.grad)
                with torch.no_grad():
                    out=torch.empty_like(x)
                    primitive=torch.ops.aten._log_softmax.out if logarithmic else torch.ops.aten._softmax.out
                    before=npu.stats();assert primitive(x,-1,False,out=out) is out;no_download(before);close(out,expected)
                count+=1;print('rows',shape,'log' if logarithmic else 'softmax','passed',flush=True)
        return dict(cases=count,forward_backward_out=True,max_elements=1048576,
                    nonfinite_rows_straddle_tile_boundary=True,no_payload_download=True)

    def sdpa_one(mode):
        tail=mode=='single-key-tail';causal=mode in ('causal','single-key-tail');p=0. if tail else .25
        shapes=((1,65537,1),(1,1,1),(1,1,1)) if tail else ((2,1,129,3),(1,4,129,3),(2,1,129,2))
        hosts=[(torch.randn(shape,dtype=dtype)/4).requires_grad_() for shape in shapes];inputs=[upload(t) for t in hosts]
        shape=tuple(torch.broadcast_shapes(*(t.shape[:-2] for t in hosts)))+(hosts[0].shape[-2],hosts[1].shape[-2])
        mask=None
        if mode in ('bool','float'):
            keep=torch.ones(129,129,dtype=torch.bool);keep[121]=False;keep[:,-1]=False
            mask=keep if mode=='bool' else torch.where(keep,.125,float('-inf')).requires_grad_()
        device_mask=upload(mask) if mask is not None else None
        saved=torch.intel_npu.get_rng_state();keep=keep_mask(saved,shape,p)
        expected=reference(*hosts,mask,p=p,causal=causal,scale=-.7,keep=keep)
        seed=torch.randn_like(expected)/8;gradient=upload(seed);cpu_state=torch.get_rng_state();before=npu.stats()
        actual=torch.nn.functional.scaled_dot_product_attention(*inputs,attn_mask=device_mask,dropout_p=p,is_causal=causal,scale=-.7)
        advanced=torch.intel_npu.get_rng_state();actual.backward(gradient);no_download(before)
        assert torch.equal(advanced,torch.intel_npu.get_rng_state()) and torch.equal(cpu_state,torch.get_rng_state())
        begin=struct.unpack('<8sQQ',bytes(saved.tolist()))[2];end=struct.unpack('<8sQQ',bytes(advanced.tolist()))[2]
        assert end-begin==(0 if p==0 else prod(shape))
        expected.backward(seed);close(actual,expected)
        for a,b in zip(inputs,hosts):close(a.grad,b.grad)
        if mask is not None and mask.requires_grad:close(device_mask.grad,mask.grad)
        if mask is not None:
            assert not torch.count_nonzero(actual.detach().cpu()[...,121,:])
            assert not torch.count_nonzero(inputs[0].grad.cpu()[...,121,:])
        return dict(score_elements=prod(shape),causal=causal,same_mask_dropout=p,
                    broadcast_qkv_and_mask_gradients=True,backward_rng_unchanged=True,no_payload_download=True)

    def native():
        host=torch.zeros(2,4,129,129,dtype=torch.float16);mask=torch.zeros_like(host,dtype=torch.bool)
        mask[...,::3]=True;mask.reshape(-1,129)[507]=True;host.reshape(-1,129)[508,0]=float('nan')
        expected=torch.ops.aten._masked_softmax(host,mask,-1,2);x,m=upload(host),upload(mask)
        before=npu.stats()
        with torch.no_grad(),npu._lock:actual=npu._native_half_masked_softmax(x,m)
        no_download(before);close(actual,expected,True)
        finite=~torch.isnan(expected);assert torch.equal(actual.cpu()[finite].view(torch.int16),expected[finite].view(torch.int16))
        return dict(elements=host.numel(),uniform_finite_bits_exact=True,masked_nan_ignored=True,all_masked_nan=True)

    def contracts():
        q=upload(torch.ones(1,1,npu.MAX_DMA_ELEMENTS//16384+1,1));k=upload(torch.ones(1,1,16384,1))
        small=upload(torch.ones(1,1,1,1));wide=upload(torch.ones(1,1,65537,1));x=upload(torch.ones(1,65537))
        F=torch.nn.functional.scaled_dot_product_attention;before=npu.stats();rng=torch.intel_npu.get_rng_state()
        for function in (lambda:F(q,k,k),lambda:F(small,wide,wide),lambda:torch.softmax(x,-1),lambda:torch.log_softmax(x,-1)):
            try:function()
            except (ValueError,RuntimeError):pass
            else:raise AssertionError('Missing row/storage rejection')
        assert npu.stats()['npu_calls']==before['npu_calls'] and torch.equal(rng,torch.intel_npu.get_rng_state())
        return dict(rejections=4,no_submission_or_rng_advance=True)

    if case=='sdpa':
        for mode in ('causal','bool','float','single-key-tail'):check(mode,lambda m=mode:sdpa_one(m))
    else:check(case,locals()[case])
    gc.collect();before=npu.stats();npu.shutdown();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes'))
    sources=('torch_intel_npu.py','torch_npu_geometry.py','torch_npu_autograd.py','torch_npu_attention_rows_demo.py',
        'npunlock/tests/fixtures/npu4000/resident_sdpa_softmax.c','npunlock/tests/fixtures/npu4000/resident/manifest.json')
    return dict(status='passed' if healthy and all(r['status']=='passed' for r in checks) else 'failed',checks=checks,
        stats_before_shutdown=before,final_stats=final,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=('all',)+tuple(PRECISIONS),default='all')
    parser.add_argument('--precision',choices=('all','float16','float32'),default='all');args=parser.parse_args()
    if args.case!='all' and args.precision!='all' and args.precision not in PRECISIONS[args.case]:parser.error('Unsupported precision for this case')
    if args.case=='all' or args.precision=='all':
        for case in PRECISIONS if args.case=='all' else (args.case,):
            for precision in PRECISIONS[case] if args.precision=='all' else tuple(p for p in PRECISIONS[case] if p==args.precision):
                if subprocess.run([sys.executable,'-B',__file__,'--case',case,'--precision',precision]).returncode:return 1
        return 0
    result=execute(args.case,args.precision);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{args.case}-{args.precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

"""Bounded online attention and genuine NestedTensor device acceptance."""
import argparse
import copy
import gc
import json
import time
import hashlib
from pathlib import Path
import traceback


def execute(case,precision='float32'):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.manual_seed(421)
    dtype=getattr(torch,precision)
    started=time.perf_counter()
    rows=[]
    def check(name,fn):
        try:rows.append(dict(name=name,status='passed',result=fn()))
        except Exception as error:rows.append(dict(name=name,status='blocked',error=str(error),traceback=traceback.format_exc()))
        print(name,rows[-1]['status'],rows[-1].get('error','')[:350],flush=True)
    def probe():
        values=[torch.arange(24.).view(3,8).to('intel_npu'),torch.ones(1,8).to('intel_npu')]
        def factory():
            x=torch.nested.nested_tensor(values,device='intel_npu')
            return dict(nested=x.is_nested,device=str(x.device),keys=str(torch._C._dispatch_keys(x)))
        def view():
            buf=values[0].view(-1)
            x=torch.ops.aten._nested_view_from_buffer.default.redispatch(
                torch._C.DispatchKeySet(torch._C.DispatchKey.CPU),buf,
                torch.tensor([[2,8],[1,8]]),torch.tensor([[8,1],[8,1]]),torch.tensor([0,16]))
            return dict(nested=x.is_nested,device=str(x.device),keys=str(torch._C._dispatch_keys(x)),sizes=x._nested_tensor_size().tolist())
        check('original-nested-factory',factory);check('metadata-view-constructor',view)
    def nested():
        host=[torch.randn(3,8,dtype=dtype),torch.randn(1,8,dtype=dtype)]
        expected=torch.nested.nested_tensor(host)
        actual=torch.nested.nested_tensor([t.to('intel_npu') for t in host],device='intel_npu')
        assert actual.is_nested and actual.device.type=='intel_npu'
        for a,b in zip(actual.unbind(),host):torch.testing.assert_close(a.cpu(),b,rtol=0,atol=0)
        torch.testing.assert_close(actual.to_padded_tensor(-2).cpu(),expected.to_padded_tensor(-2),rtol=0,atol=0)
        for converted in (actual.clone(),actual.cpu().to('intel_npu'),expected.to('intel_npu')):
            torch.testing.assert_close(converted.to_padded_tensor(-2).cpu(),expected.to_padded_tensor(-2),rtol=0,atol=0)
        actual.copy_(expected);actual.copy_(actual)
        alias=torch.ops.aten._nested_view_from_buffer(actual.values(),actual._nested_tensor_size(),actual._nested_tensor_strides(),actual._nested_tensor_storage_offsets())
        before=npu.stats()['npu_calls']
        try:actual.copy_(alias)
        except ValueError:pass
        else:raise AssertionError('Aliased nested copy accepted')
        assert npu.stats()['npu_calls']==before
        with torch.no_grad():
            a,am,ar=torch.native_layer_norm(actual,(8,),torch.ones(8,dtype=dtype,device='intel_npu'),torch.zeros(8,dtype=dtype,device='intel_npu'),1e-5)
            b,bm,br=torch.native_layer_norm(expected,(8,),torch.ones(8,dtype=dtype),torch.zeros(8,dtype=dtype),1e-5)
        for aa,bb in ((a.to_padded_tensor(0),b.to_padded_tensor(0)),(am,bm),(ar,br)):
            torch.testing.assert_close(aa.cpu(),bb,rtol=.005 if dtype==torch.float16 else 5e-5,atol=.001 if dtype==torch.float16 else 4e-6)
        train=torch.nested.nested_tensor([t.to('intel_npu') for t in host],device='intel_npu',requires_grad=True)
        before=npu.stats()['npu_calls']
        try:train.to_padded_tensor(0)
        except ValueError:pass
        else:raise AssertionError('Nested autograd silently accepted')
        assert npu.stats()['npu_calls']==before
        for norm in (None,torch.nn.LayerNorm(8,dtype=dtype)):
            cpu=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(8,2,16,dropout=0.,batch_first=True,dtype=dtype),2,norm=norm).eval()
            model=copy.deepcopy(cpu).to('intel_npu')
            x=expected.to_padded_tensor(0,output_size=(2,4,8));mask=torch.tensor([[False,False,False,True],[False,True,True,True]])
            with torch.no_grad():a=model(x.to('intel_npu'),src_key_padding_mask=mask.to('intel_npu'));b=cpu(x,src_key_padding_mask=mask)
            torch.testing.assert_close(a.cpu(),b,rtol=.005 if dtype==torch.float16 else 5e-5,atol=.001 if dtype==torch.float16 else 3e-6)
        return dict(genuine_nested_tensor=True,default_encoder_nested_fastpath=True,cpu_roundtrip_and_clone=True,copy_and_alias_guard=True,native_layernorm_stats=True,training_rejected=True)
    def flash():
        from torch.nn.attention import sdpa_kernel,SDPBackend
        cpu=[torch.randn(s,dtype=dtype).requires_grad_() for s in ((2,3,7),(2,5,7),(2,5,5))]
        dev=[x.detach().to('intel_npu').requires_grad_() for x in cpu]
        with sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.scaled_dot_product_attention(*cpu)
        actual=torch.intel_npu.flash_attention(*dev,query_block=2,key_block=3)
        seed=torch.randn_like(expected);actual.backward(seed.to('intel_npu'));expected.backward(seed)
        for a,b in [(actual,expected)]+[(a.grad,b.grad) for a,b in zip(dev,cpu)]:
            torch.testing.assert_close(a.detach().cpu(),b.detach(),rtol=.01 if dtype==torch.float16 else 5e-5,atol=.002 if dtype==torch.float16 else 5e-6)
        return dict(forward_and_recomputed_qkv_gradients=True,multiple_query_and_key_tiles=True)
    def compare_attention(shapes,options,mask=None,blocks=None):
        from torch.nn.attention import sdpa_kernel,SDPBackend
        cpu=[(torch.randn(s,dtype=dtype)/4).requires_grad_() for s in shapes]
        dev=[x.detach().to('intel_npu').requires_grad_() for x in cpu]
        expected_options=dict(options,attn_mask=mask)
        with sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.scaled_dot_product_attention(*cpu,**expected_options)
        actual_options=dict(options,attn_mask=None if mask is None else mask.to('intel_npu'))
        if blocks:actual=torch.intel_npu.flash_attention(*dev,**actual_options,query_block=blocks[0],key_block=blocks[1])
        else:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):actual=torch.nn.functional.scaled_dot_product_attention(*dev,**actual_options)
        seed=torch.randn_like(expected);actual.backward(seed.to('intel_npu'));expected.backward(seed)
        for a,b in [(actual,expected)]+[(a.grad,b.grad) for a,b in zip(dev,cpu)]:
            torch.testing.assert_close(a.detach().cpu(),b.detach(),rtol=.01 if dtype==torch.float16 else 5e-5,atol=.002 if dtype==torch.float16 else 5e-6)
    def masks():
        shapes=((1,2,19,8),(1,2,23,8),(1,2,23,5))
        compare_attention(shapes,dict(is_causal=True),blocks=(8,7))
        mask=torch.rand(19,23)>.25;mask[0]=False
        compare_attention(shapes,dict(scale=-.7),mask,blocks=(8,7))
        mask=torch.randn(1,1,19,23)/8;mask[:,:,5]=float('-inf')
        compare_attention(shapes,dict(scale=.4),mask,blocks=(8,7))
        compare_attention(((2,4,3,7),(1,2,5,7),(2,5,5)),dict(enable_gqa=True,scale=-.7),blocks=(2,3))
        compare_attention(((2,3,7),(2,5,7),(2,5,5)),dict(is_causal=True))
        compare_attention(((1,1,17,128),(1,1,65,128),(1,1,65,128)),dict(is_causal=True))
        compare_attention(((1,1,33,128),(1,1,129,128),(1,1,129,128)),dict(is_causal=True))
        return dict(causal_offsets_and_tail_tiles=True,fully_masked_rows=True,float_mask=True,gqa_and_broadcast_gradients=True,standard_flash_selection=True)
    def special():
        from torch.nn.attention import sdpa_kernel,SDPBackend
        records=[]
        for kind in ('nan-q','inf-v','inf-mask','fully-masked-inf-v'):
            q,k,v=[torch.randn(s,dtype=dtype)/4 for s in ((3,7),(5,7),(5,5))];mask=None
            if kind=='nan-q':q[0,0]=float('nan')
            elif kind=='inf-v':v[0,0]=float('inf')
            elif kind=='inf-mask':mask=torch.zeros(3,5,dtype=dtype);mask[0,0]=float('inf')
            else:v[0,0]=float('inf');mask=torch.zeros(3,5,dtype=torch.bool)
            with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
            actual=torch.intel_npu.flash_attention(*(t.to('intel_npu') for t in (q,k,v)),attn_mask=None if mask is None else mask.to('intel_npu'),query_block=2,key_block=3)
            torch.testing.assert_close(actual.cpu(),expected,rtol=.005 if dtype==torch.float16 else 5e-5,atol=.001 if dtype==torch.float16 else 4e-6,equal_nan=True)
            records.append(kind)
        return dict(forward_ieee_cases=records)
    def underflow():
        if dtype!=torch.float32:raise ValueError('Gradual underflow acceptance uses FP32 storage')
        from torch.nn.attention import sdpa_kernel,SDPBackend
        records=[]
        for kind in ('tiny-value','tiny-query','tiny-scale'):
            values=[torch.randn(s)/4 for s in ((3,7),(5,7),(5,5))]
            if kind=='tiny-value':values[2]*=2.**-125
            elif kind=='tiny-query':values[0]*=2.**-125
            scale=2.**-132 if kind=='tiny-scale' else .4
            cpu=[t.double().requires_grad_() for t in values]
            dev=[t.to('intel_npu').requires_grad_() for t in values]
            with sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.scaled_dot_product_attention(*cpu,scale=scale)
            actual=torch.intel_npu.flash_attention(*dev,scale=scale,query_block=2,key_block=3)
            seed=torch.randn_like(actual,device='cpu');expected.backward(seed.double());actual.backward(seed.to('intel_npu'))
            errors=[]
            for index,(a,b) in enumerate([(actual,expected)]+[(a.grad,b.grad) for a,b in zip(dev,cpu)]):
                actual_host=a.detach().cpu().double();reference=b.detach()
                torch.testing.assert_close(actual_host,reference,rtol=2e-4,atol=32*2.**-149)
                if index=={'tiny-value':0,'tiny-query':2,'tiny-scale':1}[kind]:
                    assert torch.count_nonzero(actual_host)>0, 'Subnormal results were flushed to zero'
                errors.append(float((actual_host-reference).abs().max()))
            records.append(dict(case=kind,max_abs_output_qkv_gradient_errors=errors))
        return dict(gradual_underflow_cases=records)
    def nested_attention():
        from torch.nn.attention import sdpa_kernel,SDPBackend
        # Ensure the selective AutogradNestedTensor entry keeps CPU forward and autograd intact.
        reference=[[torch.randn(s)/4 for s in shapes] for shapes in (
            ((2,3,4),(2,1,4)),((2,5,4),(2,2,4)),((2,5,5),(2,2,5)))]
        tensors=[torch.nested.nested_tensor(group,requires_grad=True) for group in reference]
        with sdpa_kernel(SDPBackend.MATH):out=torch.nn.functional.scaled_dot_product_attention(*tensors)
        out.values().sum().backward()
        for index,a in enumerate(out.unbind()):
            values=[group[index].detach().requires_grad_() for group in reference]
            with sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.scaled_dot_product_attention(*values)
            expected.sum().backward();torch.testing.assert_close(a,expected,rtol=5e-5,atol=3e-6)
            for t,v in zip(tensors,values):torch.testing.assert_close(t.grad.unbind()[index],v.grad,rtol=5e-5,atol=4e-6)
        host=[[torch.randn(s,dtype=dtype)/4 for s in shapes] for shapes in (
            ((4,3,7),(4,1,7)),((2,5,7),(2,2,7)),((2,5,5),(2,2,5)))]
        dev=[torch.nested.nested_tensor([t.to('intel_npu') for t in group],device='intel_npu') for group in host]
        with torch.no_grad(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            actual=torch.nn.functional.scaled_dot_product_attention(*dev,is_causal=True,enable_gqa=True)
        assert actual.is_nested and actual.device.type=='intel_npu'
        for index,a in enumerate(actual.unbind()):
            with sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.scaled_dot_product_attention(*(group[index] for group in host),is_causal=True,enable_gqa=True)
            torch.testing.assert_close(a.cpu(),expected,rtol=.005 if dtype==torch.float16 else 5e-5,atol=.001 if dtype==torch.float16 else 4e-6)
        parts=[torch.randn(5,8,dtype=dtype)/4,torch.randn(2,8,dtype=dtype)/4,torch.empty(0,8,dtype=dtype)]
        cpu_nt=torch.nested.nested_tensor(parts);nt=cpu_nt.to('intel_npu')
        for average in (False,True):
            cpu=torch.nn.MultiheadAttention(8,2,batch_first=True,dtype=dtype).eval();model=copy.deepcopy(cpu).to('intel_npu')
            with torch.no_grad():a,aw=model(nt,nt,nt,average_attn_weights=average);b,bw=cpu(cpu_nt,cpu_nt,cpu_nt,average_attn_weights=average)
            torch.testing.assert_close(a.to_padded_tensor(0).cpu(),b.to_padded_tensor(0),rtol=.005 if dtype==torch.float16 else 5e-5,atol=.001 if dtype==torch.float16 else 4e-6)
            torch.testing.assert_close(aw.cpu(),bw,rtol=.005 if dtype==torch.float16 else 5e-5,atol=.001 if dtype==torch.float16 else 4e-6)
        return dict(ragged_flash_sdpa=True,gqa_causal=True,native_mha_weights=True,empty_constituent=True,cpu_nested_forward_backward_preserved=True)
    def contracts():
        q,k,v=[torch.randn(s,device='intel_npu',dtype=dtype) for s in ((2,3,7),(2,5,7),(2,5,5))]
        mask=torch.zeros(3,5,dtype=dtype,device='intel_npu',requires_grad=True)
        actions=(lambda:torch.intel_npu.flash_attention(q,k,v,dropout_p=.1),
                 lambda:torch.intel_npu.flash_attention(q,k,v,key_block=129),
                 lambda:torch.intel_npu.flash_attention(q,k,v,query_block=0),
                 lambda:torch.intel_npu.flash_attention(q,k,v,attn_mask=mask),
                 lambda:torch.intel_npu.flash_attention(q,k,v,attn_mask=torch.empty(2,3,4,device='intel_npu',dtype=dtype)))
        before=npu.stats()['npu_calls'];state=torch.intel_npu.get_rng_state()
        for action in actions:
            try:action()
            except (ValueError,RuntimeError):pass
            else:raise AssertionError('Invalid flash configuration accepted')
        assert npu.stats()['npu_calls']==before and torch.equal(state,torch.intel_npu.get_rng_state())
        for shapes in (((2,0,7),(2,5,7),(2,5,5)),((2,3,7),(2,0,7),(2,0,5)),((0,2,3,7),(1,2,5,7),(0,2,5,5))):
            compare_attention(shapes,{})
        q.requires_grad_();a=torch.intel_npu.flash_attention(q,k,v)
        with torch.no_grad():k.add_(1)
        seed=torch.ones_like(a)
        before=npu.stats()['npu_calls']
        try:a.backward(seed)
        except RuntimeError as error:assert 'version' in str(error) or 'inplace' in str(error)
        else:raise AssertionError('Saved KV mutation not detected')
        assert npu.stats()['npu_calls']==before
        return dict(metadata_rejections=len(actions),no_submit_or_rng_on_rejection=True,empty_gradients=True,saved_kv_version=True)
    def measure():
        from torch.nn.attention import sdpa_kernel,SDPBackend
        mode=SDPBackend.MATH if case=='memory-math' else SDPBackend.FLASH_ATTENTION
        cpu=[(torch.randn(1,2,n,16,dtype=dtype)/4).requires_grad_() for n in (128,256,256)]
        dev=[t.detach().to('intel_npu').requires_grad_() for t in cpu]
        with sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.scaled_dot_product_attention(*cpu)
        begin=time.perf_counter()
        saved={}
        def pack(t):saved[t.untyped_storage()._cdata]=dict(shape=tuple(t.shape),bytes=t.untyped_storage().nbytes());return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t),sdpa_kernel(mode):actual=torch.nn.functional.scaled_dot_product_attention(*dev)
        seed=torch.randn_like(expected);actual.backward(seed.to('intel_npu'));duration=time.perf_counter()-begin
        expected.backward(seed)
        for a,b in [(actual,expected)]+[(a.grad,b.grad) for a,b in zip(dev,cpu)]:
            torch.testing.assert_close(a.detach().cpu(),b.detach(),rtol=.01 if dtype==torch.float16 else 5e-5,atol=.002 if dtype==torch.float16 else 5e-6)
        return dict(mode=case,query=tuple(cpu[0].shape),key=tuple(cpu[1].shape),cold_forward_backward_seconds=duration,all_gradients_match_cpu=True,
                    saved_unique_storage_bytes=sum(v['bytes'] for v in saved.values()),saved_storages=list(saved.values()))
    if case=='probe':probe()
    else:check(case,dict(nested=nested,flash=flash,masks=masks,contracts=contracts,special=special,underflow=underflow,**{'memory-math':measure,'memory-flash':measure,'nested-attention':nested_attention})[case])
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();after=npu.stats()
    root=Path(__file__).resolve().parent/'npunlock-probe/research/results/flash-nested'
    root.mkdir(exist_ok=True)
    healthy=after['closed'] and not after['context_failed'] and all(after[k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes'))
    data=dict(status='probe-complete' if case=='probe' else 'passed' if healthy and all(x['status']=='passed' for x in rows) else 'failed',checks=rows,stats=before,shutdown=after,elapsed_seconds=time.perf_counter()-started,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_flash_attention.py','torch_npu_nested.py','npunlock/tests/fixtures/npu4000/resident/manifest.json')})
    (root/(case+'-'+precision+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
    return int(data['status']=='failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=('probe','nested','flash','masks','contracts','special','underflow','nested-attention','memory-math','memory-flash'),default='probe')
    parser.add_argument('--precision',choices=('float16','float32'),default='float32')
    args=parser.parse_args();raise SystemExit(execute(args.case,args.precision))

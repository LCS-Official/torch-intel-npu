"""Strided NestedTensor inference with NPU payload and CPU shape metadata."""
from math import prod
import operator
import torch
import torch_intel_npu as npu

_nested=torch.library.Library('aten','IMPL','NestedTensorPrivateUse1')
_autograd_nested=torch.library.Library('aten','IMPL','AutogradNestedTensor')
_cpu_nested=torch._C.DispatchKeySet(torch._C.DispatchKey.NestedTensorCPU)
_cpu=torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)


def metadata(name,x):
    # These native kernels only return NestedTensorImpl's CPU shape/offset tensors.
    return getattr(torch.ops.aten,name).default.redispatch(_cpu_nested,x)


def parts(x):
    if not x.is_nested or x.layout!=torch.strided or x.device.type!='intel_npu':
        raise ValueError('NPU nested inference requires strided NestedTensor on intel_npu')
    npu._inference_only((x,))
    if x.dtype not in (torch.float16,torch.float32):raise ValueError('NPU nested inference requires FP16/FP32')
    # Native unbind constructs dense storage views; its payload kernels dispatch on NPU.
    return tuple(torch.ops.aten.unbind.int.redispatch(_cpu_nested,x,0))


def packed(values):
    if not values:raise ValueError('NPU nested inference requires a nonempty batch')
    return torch.nested.nested_tensor(list(values),dtype=values[0].dtype,device=values[0].device)


def from_buffer(buffer,sizes,strides,offsets):
    npu._validate_read(buffer);npu._inference_only((buffer,))
    if (buffer.ndim!=1 or not buffer.is_contiguous() or buffer.storage_offset() or
            any(t.device.type!='cpu' or t.dtype!=torch.int64 or not t.is_contiguous() for t in (sizes,strides,offsets)) or
            sizes.ndim!=2 or strides.shape!=sizes.shape or offsets.shape!=(sizes.shape[0],) or
            sizes.shape[1]>=npu.MAX_RANK or (sizes<0).any() or (strides<0).any() or (offsets<0).any()):
        raise ValueError('NPU nested view requires flat storage and nonnegative CPU INT64 layout metadata')
    # This constructor checks storage bounds and wraps storage; it never reads payload values.
    return torch.ops.aten._nested_view_from_buffer.default.redispatch(_cpu,buffer,sizes,strides,offsets)


def left_aligned(t,mask):
    npu._validate_float_read(t);npu._validate_read(mask)
    if t.ndim!=3 or mask.dtype!=torch.bool or tuple(mask.shape)!=tuple(t.shape[:2]):
        raise ValueError('NPU nested mask requires batched sequence input and matching BOOL keep mask')
    if mask.shape[1]<2:return True
    return not (mask[:,1:] & ~mask[:,:-1]).any().item()


def from_mask(t,mask,mask_check=True):
    npu._inference_only((t,))
    with npu._lock,torch.no_grad():
        # Validate shape even when the caller has already checked alignment.
        npu._validate_float_read(t);npu._validate_read(mask)
        if t.ndim!=3 or not t.shape[0] or mask.dtype!=torch.bool or tuple(mask.shape)!=tuple(t.shape[:2]):
            raise ValueError('NPU nested conversion requires nonempty batch and matching BOOL keep mask')
        if mask_check and not left_aligned(t,mask):raise ValueError('NPU nested keep mask must be left aligned')
        lengths=mask.sum(1,dtype=torch.int64).cpu().tolist()
        return packed([t[i,:length].contiguous() for i,length in enumerate(lengths)])


def padded(x,padding,output_size=None,*,out=None):
    npu._inference_only((x,))
    with npu._lock,torch.no_grad():
        values=parts(x)
        if not values:raise ValueError('NPU nested padded conversion requires nonempty batch')
        rank=values[0].ndim
        shape=(len(values),)+tuple(max(t.shape[a] for t in values) for a in range(rank))
        if output_size is not None:
            requested=tuple(operator.index(v) for v in output_size)
            if len(requested)!=len(shape) or any(a<b for a,b in zip(requested,shape)):
                raise ValueError('NPU nested output_size may only enlarge the padded shape')
            shape=requested
        if prod(shape)>npu._storage_limit(x.dtype):raise ValueError('NPU nested padded output exceeds storage')
        npu._check_out(out,shape,x.dtype,(x,))
        result=torch.full(shape,padding,dtype=x.dtype,device=x.device)
        for i,value in enumerate(values):
            if value.numel():npu._copy(result[(i,)+tuple(slice(0,n) for n in value.shape)],value)
        return npu._copy(out,result) if out is not None else result


def mha(query,key,value,embed_dim,num_head,qkv_weight,qkv_bias,proj_weight,proj_bias,
        mask=None,need_weights=True,average_attn_weights=True,mask_type=None):
    with npu._lock:
        npu._inference_only((query,key,value,qkv_weight,qkv_bias,proj_weight,proj_bias))
        if query is not key or query is not value or mask is not None:
            raise ValueError('NPU nested MHA requires self attention without a separate mask')
        values=parts(query)
        if not values or any(t.ndim!=2 or t.shape[1]!=embed_dim for t in values):
            raise ValueError('NPU nested MHA requires ragged sequences with a common embedding width')
        # Preflight every item before submitting the first projection.
        for t in values:
            q=t.unsqueeze(0)
            npu._native_mha_spec(q,q,q,embed_dim,num_head,qkv_weight,qkv_bias,proj_weight,proj_bias,None,None)
        results=[];weights=[]
        for t in values:
            q=t.unsqueeze(0)
            a,w=npu._native_multi_head_attention(q,q,q,embed_dim,num_head,qkv_weight,qkv_bias,proj_weight,proj_bias,
                                                None,need_weights,average_attn_weights,None)
            results.append(a.squeeze(0));weights.append(w)
        scores=None
        if need_weights:
            length=max(t.shape[0] for t in values)
            shape=(len(values),)+(() if average_attn_weights else (num_head,))+(length,length)
            scores=torch.zeros(shape,dtype=query.dtype,device=query.device)
            for i,w in enumerate(weights):
                if w is not None and w.numel():npu._copy(scores[i,...,:w.shape[-2],:w.shape[-1]],w.squeeze(0))
        return packed(results),scores


def encoder(src,*args,**kwargs):
    with npu._lock:
        npu._inference_only((src,)+tuple(v for v in args if isinstance(v,torch.Tensor)))
        if kwargs.get('mask') is not None or len(args)>17 and args[17] is not None:
            raise ValueError('NPU nested encoder does not accept a separate mask')
        values=parts(src)
        if not values or any(t.ndim!=2 for t in values):raise ValueError('NPU nested encoder requires ragged sequences')
        return packed([npu._transformer_encoder_layer_fwd(t.unsqueeze(0),*args,**kwargs).squeeze(0) for t in values])


def layer_norm(x,shape,weight=None,bias=None,eps=1e-5):
    with npu._lock:
        npu._inference_only((x,weight,bias));values=parts(x)
        outputs=[];means=[];inverses=[]
        for t in values:
            a,m,r=npu._layer_norm(npu._materialize(t),shape,weight,bias,eps)
            outputs.append(a);means.append(m.reshape(-1));inverses.append(r.reshape(-1))
        return packed(outputs),npu._concat(means,(sum(t.numel() for t in means),),means[0].dtype),npu._concat(inverses,(sum(t.numel() for t in inverses),),inverses[0].dtype)


def sdpa(query,key,value,attn_mask=None,dropout_p=0.,is_causal=False,*,scale=None,enable_gqa=False):
    with npu._lock:
        npu._inference_only((query,key,value,attn_mask))
        if attn_mask is not None or dropout_p!=0:raise ValueError('NPU nested SDPA requires no external mask and dropout_p=0')
        q,k,v=parts(query),parts(key),parts(value)
        if not q or len(q)!=len(k) or len(q)!=len(v):raise ValueError('NPU nested SDPA batch sizes differ or are empty')
        streaming=not torch.backends.cuda.math_sdp_enabled()
        for a,b,c in zip(q,k,v):npu._sdpa_spec(a,b,c,None,0,is_causal,scale=scale,enable_gqa=enable_gqa,streaming=streaming)
        return packed([npu._sdpa(a,b,c,None,0,is_causal,scale=scale,enable_gqa=enable_gqa) for a,b,c in zip(q,k,v)])


def sdpa_dispatch(keyset,query,key,value,attn_mask=None,dropout_p=0.,is_causal=False,*,scale=None,enable_gqa=False):
    if query.device.type=='intel_npu':
        return sdpa(query,key,value,attn_mask,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa)
    # The upstream composite AutogradNestedTensor kernel runs before backend keys.
    # Other devices keep that same composite through their original lower backend kernel.
    return torch.ops.aten.scaled_dot_product_attention.default.redispatch(
        keyset.remove(torch._C.DispatchKey.AutogradNestedTensor),query,key,value,attn_mask,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa)


def to_copy(x,*,dtype=None,layout=None,device=None,pin_memory=None,non_blocking=False,memory_format=None):
    with npu._lock:
        npu._inference_only((x,));dtype=dtype or x.dtype;device=torch.device(device) if device is not None else x.device
        if dtype not in (torch.float16,torch.float32) or device.type not in ('cpu','intel_npu') or layout not in (None,torch.strided) or pin_memory or non_blocking or memory_format not in (None,torch.preserve_format,torch.contiguous_format):
            raise ValueError('NPU nested copy supports synchronous contiguous FP16/FP32 CPU/intel_npu only')
        if device.type=='intel_npu':npu._check_device(device)
        source=parts(x) if x.device.type=='intel_npu' else x.unbind() if x.is_nested and x.device.type=='cpu' else None
        if source is None or x.dtype not in (torch.float16,torch.float32):raise ValueError('NPU nested copy requires FP16/FP32 CPU/intel_npu input')
        return packed([t.to(device=device,dtype=dtype,copy=True) for t in source])


def copy(dst,src,non_blocking=False):
    with npu._lock:
        npu._inference_only((dst,src))
        if non_blocking or not src.is_nested or src.device.type not in ('cpu','intel_npu'):
            raise ValueError('NPU nested copy requires synchronous nested CPU/intel_npu input')
        if dst is src:return dst
        if src.dtype not in (torch.float16,torch.float32) or dst.untyped_storage()._cdata==src.untyped_storage()._cdata:
            raise ValueError('NPU nested copy requires FP16/FP32 and nonaliased source/destination')
        target=parts(dst);source=parts(src) if src.device.type=='intel_npu' else src.unbind()
        if len(target)!=len(source) or any(a.shape!=b.shape for a,b in zip(target,source)):
            raise ValueError('NPU nested copy shapes differ')
        with torch.no_grad():
            for a,b in zip(target,source):npu._copy(a,b)
        return dst


def register(aten):
    for name in ('_nested_tensor_size','_nested_tensor_strides','_nested_tensor_storage_offsets'):
        _nested.impl(name,lambda x,name=name:metadata(name,x))
    _nested.impl('unbind.int',lambda x,dim=0:parts(x) if dim in (0,-x.ndim) else (_ for _ in ()).throw(ValueError('NPU nested unbind requires batch axis')))
    _nested.impl('values',lambda x:torch.ops.aten.values.default.redispatch(_cpu_nested,x))
    _nested.impl('to_padded_tensor',padded);_nested.impl('to_padded_tensor.out',padded)
    _nested.impl('_native_multi_head_attention',mha);_nested.impl('_transformer_encoder_layer_fwd',encoder)
    _nested.impl('native_layer_norm',layer_norm)
    _nested.impl('scaled_dot_product_attention',sdpa)
    _autograd_nested.impl('scaled_dot_product_attention',sdpa_dispatch,with_keyset=True)
    _nested.impl('_to_copy',to_copy);_nested.impl('copy_',copy)
    _nested.impl('clone',lambda x,*,memory_format=None:to_copy(x,memory_format=memory_format))
    aten.impl('_nested_view_from_buffer',from_buffer)
    aten.impl('_nested_tensor_from_mask_left_aligned',left_aligned)
    aten.impl('_nested_tensor_from_mask',from_mask)

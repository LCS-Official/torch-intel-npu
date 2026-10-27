"""Experimental FlashAttention-style CMX tiles, with recomputed first gradients."""
from itertools import product
from math import prod
import struct
import torch
from torch.autograd.function import once_differentiable
import torch_intel_npu as npu
from torch_npu_compat_extended import packet


def selected():
    return not torch.backends.cuda.math_sdp_enabled() and torch.backends.cuda.flash_sdp_enabled()


def coordinates(shape):return product(*(range(n) for n in shape))


def operand_coords(t,coords,gqa=False,heads=None):
    indices=list(coords[-(t.ndim-2):]) if t.ndim>2 else []
    if gqa:indices[-1]//=heads//t.shape[-3]
    return tuple(0 if n==1 else i for i,n in zip(indices,t.shape[:-2]))


def tile(t,coords,start,count):
    return npu._cast(npu._materialize(t[coords+(slice(start,start+count),slice(None))]),torch.float32)


def mask_tile(mask,shape,coords,qs,nq,ks,nk):
    if mask is None:return None
    return npu._materialize(mask.expand(shape)[coords+(slice(qs,qs+nq),slice(ks,ks+nk))])


def frame(q,k,v,state,mask,qs,ks,causal,scale,first,go=None):
    nr,d=q.shape;ns,dv=v.shape
    root=struct.unpack('<I',struct.pack('<f',abs(scale)**.5))[0]
    values=[q,k,v,state]+([] if go is None else [go])+([] if mask is None else [mask])
    header=(nr,ns,d,dv,qs,ks,int(causal),root,int(scale<0),0 if mask is None else 2 if mask.dtype==torch.bool else 1,int(first))
    count=nr*(dv+2) if go is None else q.numel()+k.numel()+v.numel()
    raw=packet('flash-step-f32' if go is None else 'flash-backward-f32',header,values,count*4).view(-1)
    if go is None:return npu._slice(raw,0,count).view(nr,dv+2)
    return tuple(npu._slice(raw,start,length).view(shape) for start,length,shape in (
        (0,q.numel(),tuple(q.shape)),(q.numel(),k.numel(),tuple(k.shape)),
        (q.numel()+k.numel(),v.numel(),tuple(v.shape))))


def packet_words(nr,ns,d,dv,mask):
    words=16+nr*d+ns*d+ns*dv+nr*(dv+2)+nr*dv
    return words+(0 if mask is None else (nr*ns*(1 if mask.dtype==torch.bool else 4)+3)//4)


def block_shape(q,k,v,mask,qblock,kblock):
    """Prefer fewer packets, retaining the complete backward carrier budget."""
    autoq,autok=qblock is None,kblock is None
    qb,kb=32 if autoq else qblock,128 if autok else kblock
    if type(qb) is not int or type(kb) is not int or not 1<=qb<=32 or not 1<=kb<=128:
        raise ValueError('NPU flash query/key block bounds are 1..32 and 1..128')
    while packet_words(min(qb,q.shape[-2]),min(kb,k.shape[-2]),q.shape[-1],v.shape[-1],mask)>32768:
        if autok and kb>1:kb=max(1,kb//2)
        elif autoq and qb>1:qb=max(1,qb//2)
        else:raise ValueError('NPU flash complete forward/backward packet exceeds 32768 words; reduce block sizes')
    return qb,kb


def validate(q,k,v,mask,p,causal,scale,gqa,qblock,kblock):
    shape,scale=npu._sdpa_spec(q,k,v,mask,p,causal,scale=scale,enable_gqa=gqa,streaming=True)
    if p!=0:raise ValueError('NPU experimental flash attention currently requires dropout_p=0')
    if not 1<=q.shape[-1]<=128 or not 1<=v.shape[-1]<=128:
        raise ValueError('NPU flash QK/V head dimensions must be in [1,128]')
    if type(qblock) is not int or type(kblock) is not int or not 1<=qblock<=32 or not 1<=kblock<=128:
        raise ValueError('NPU flash query/key block bounds are 1..32 and 1..128')
    if packet_words(min(qblock,shape[-2]),min(kblock,shape[-1]),q.shape[-1],v.shape[-1],mask)>32768:
        raise ValueError('NPU flash complete forward/backward packet exceeds 32768 words; reduce block sizes')
    return shape,scale


def forward(q,k,v,mask,causal,scale,gqa,qblock,kblock):
    shape,scale=validate(q,k,v,mask,0,causal,scale,gqa,qblock,kblock)
    output_shape=shape[:-1]+(v.shape[-1],);states=[];layout=[]
    if not prod(shape) or gqa and not k.shape[-3]:return torch.zeros(output_shape,dtype=q.dtype,device=q.device),states,layout
    def rows():
        for coords in coordinates(shape[:-2]):
            qi=operand_coords(q,coords);ki=operand_coords(k,coords,gqa,q.shape[-3] if gqa else None)
            vi=operand_coords(v,coords,gqa,q.shape[-3] if gqa else None)
            for qs in range(0,shape[-2],qblock):
                nr=min(qblock,shape[-2]-qs);qt=tile(q,qi,qs,nr)
                state=torch.zeros((nr,v.shape[-1]+2),dtype=torch.float32,device=q.device)
                for ks in range(0,shape[-1],kblock):
                    nk=min(kblock,shape[-1]-ks)
                    mt=mask_tile(mask,shape,coords,qs,nr,ks,nk)
                    if mt is not None and mt.dtype!=torch.bool:mt=npu._cast(mt,torch.float32)
                    state=frame(qt,tile(k,ki,ks,nk),tile(v,vi,ks,nk),state,mt,qs,ks,causal,scale,ks==0)
                states.append(state);layout.append((coords,qi,ki,vi,qs,nr))
                denominator=npu._materialize(state[:,1:2]);numerator=npu._materialize(state[:,2:])
                yield npu._binary_op(numerator,npu._where(npu._compare(denominator,0.,kind='eq'),1.,denominator),kind='div')
    result=npu._stream_concat(rows(),output_shape)
    return npu._cast(result,q.dtype),states,layout


def backward(q,k,v,mask,states,layout,g,causal,scale,gqa,qblock,kblock,needs):
    shape,scale=validate(q,k,v,mask,0,causal,scale,gqa,qblock,kblock)
    accum=[{}, {}, {}]
    def add(which,key,value):
        if not needs[which]:return
        old=accum[which].get(key)
        accum[which][key]=value if old is None else npu._binary_op(old,value,kind='add')
    for state,(coords,qi,ki,vi,qs,nr) in zip(states,layout):
        qt=tile(q,qi,qs,nr);go=tile(g,coords,qs,nr)
        for ks in range(0,shape[-1],kblock):
            nk=min(kblock,shape[-1]-ks);mt=mask_tile(mask,shape,coords,qs,nr,ks,nk)
            if mt is not None and mt.dtype!=torch.bool:mt=npu._cast(mt,torch.float32)
            dq,dk,dv=frame(qt,tile(k,ki,ks,nk),tile(v,vi,ks,nk),state,mt,qs,ks,causal,scale,False,go)
            add(0,(qi,qs),dq);add(1,(ki,ks),dk);add(2,(vi,ks),dv)
    gradients=[]
    for i,(t,block) in enumerate(zip((q,k,v),(qblock,kblock,kblock))):
        if not needs[i]:gradients.append(None);continue
        if not layout:
            gradients.append(torch.zeros(tuple(t.shape),dtype=t.dtype,device=t.device));continue
        pieces=(accum[i][(coords,start)] for coords in coordinates(tuple(t.shape[:-2])) for start in range(0,t.shape[-2],block))
        gradients.append(npu._cast(npu._stream_concat(pieces,tuple(t.shape)),t.dtype))
    return tuple(gradients)


class Flash(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mask,causal,scale,gqa,qblock,kblock):
        with npu._lock:
            out,states,ctx.layout=forward(q,k,v,mask,causal,scale,gqa,qblock,kblock)
            ctx.save_for_backward(q,k,v,mask,*states)
            ctx.params=(causal,scale,gqa,qblock,kblock)
            return out

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        q,k,v,mask,*states=ctx.saved_tensors
        with npu._lock:
            return backward(q,k,v,mask,states,ctx.layout,g,*ctx.params,ctx.needs_input_grad[:3])+(None,)*6


def flash_attention(query,key,value,attn_mask=None,dropout_p=0.,is_causal=False,*,scale=None,enable_gqa=False,query_block=None,key_block=None):
    """Opt-in tiled SDPA; automatic blocks stay within the 32768-word budget."""
    active=torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (query,key,value,attn_mask))
    if active and attn_mask is not None and attn_mask.requires_grad:
        raise ValueError('NPU experimental flash does not yet support trainable masks; select math SDPA')
    with npu._lock,torch.no_grad():
        npu._sdpa_spec(query,key,value,attn_mask,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa,streaming=True)
        query_block,key_block=block_shape(query,key,value,attn_mask,query_block,key_block)
        validate(query,key,value,attn_mask,dropout_p,is_causal,scale,enable_gqa,query_block,key_block)
    args=(query,key,value,attn_mask,is_causal,scale,enable_gqa,query_block,key_block)
    if active:return Flash.apply(*args)
    with npu._lock,torch.no_grad():return forward(*args)[0]

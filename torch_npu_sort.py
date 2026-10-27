"""Floating sorting and indexed extreme reductions on the real Intel NPU."""
import operator
from math import prod
import numpy as np
import torch
from torch.autograd.function import once_differentiable
import torch_intel_npu as npu


def _axis(x,dim):
    npu._validate_read(x);dim=operator.index(dim);rank=x.ndim or 1
    if not -rank<=dim<rank:raise IndexError('NPU sort/reduction axis out of range')
    return dim%rank


def _packet(values,indices=None,*,rows=1,width=None,descending=False,pos=0,stage=0,stride=0,partner=None):
    count=values.numel();width=count if width is None else width;merge=indices is not None
    integer=values.dtype==torch.int64;words=2 if integer else 1;mode=int(merge)+2*integer
    size=max(13,7+(words+2*merge)*count,((words+2)*(1 if merge else 2))*count)
    header=npu._concat([npu._scalar(float(v),torch.float32).view(1) for v in
                       (mode,rows,width,int(descending),pos,stage,stride)],(7,),torch.float32)
    def pad(parts):
        used=sum(p.numel() for p in parts)
        if used<size:parts.append(npu._broadcast(npu._scalar(0.,torch.float32),(size-used,)))
        return npu._concat(parts,(1,size),torch.float32)
    parts=[header,values.view(-1).view(torch.float32) if integer else values.view(-1)]
    if merge:parts.append(indices.view(-1).view(torch.float32))
    data=pad(parts)
    second=data if not merge else pad([partner[0].view(-1).view(torch.float32) if integer else partner[0].view(-1),partner[1].view(-1).view(torch.float32)])
    result=npu._run(('sort-f32','f32',(1,size)),[data,second],(1,size),torch.float32)
    value=npu._slice(result,0,words*count)
    return (value.view(torch.int64) if integer else value),npu._slice(result,words*count,2*count).view(torch.int64)


def _wide_row(row,descending):
    """Shape-only XOR gathers tile a complete bitonic network; no CPU values."""
    width=row.numel();length=1<<(width-1).bit_length()
    value=row.view(-1);integer=row.dtype==torch.int64
    if length>width:value=npu._cat([value,npu._broadcast(npu._scalar(0 if integer else 0.,row.dtype),(length-width,))],0)
    ids=np.arange(length,dtype=np.int64);ids[width:]=0xffffffff
    index=torch.from_numpy(ids).to('intel_npu');stage=2
    while stage<=length:
        stride=stage//2
        while stride:
            values=[];indices=[]
            words=index.view(torch.float32)
            for start in range(0,length,8192 if integer else 16384):
                count=min(8192 if integer else 16384,length-start);partner=(np.arange(start,start+count,dtype=np.int64)^stride)[:,None]
                other=npu._gather_window(value.view(torch.float32) if integer else value,(partner*2+np.arange(2)).reshape(-1,2) if integer else partner,1).view(-1)
                if integer:other=other.view(torch.int64)
                raw=npu._gather_window(words,(partner*2+np.arange(2)).reshape(-1,2),1).view(-1).view(torch.int64)
                a=npu._slice(value.view(torch.float32),2*start,2*count).view(torch.int64) if integer else npu._slice(value,start,count);b=npu._slice(words,2*start,2*count).view(torch.int64)
                v,i=_packet(a,b,descending=descending,pos=start,stage=stage,stride=stride,partner=(other,raw))
                values.append(v);indices.append(i)
            value=npu._cat(values,0);index=npu._cat(indices,0)
            stride//=2
        stage*=2
    return (npu._slice(value.view(torch.float32),0,2*width).view(torch.int64) if integer else npu._slice(value,0,width)),npu._slice(index.view(torch.float32),0,2*width).view(torch.int64)


def sort(x,dim=-1,descending=False,*,stable=None,values=None,indices=None):
    with npu._lock:
        axis=_axis(x,dim)
        if x.dtype not in (torch.float16,torch.float32,torch.int64,torch.bool):raise NotImplementedError('NPU sort currently requires FP16/FP32/INT64/BOOL')
        if x.numel()>npu.MAX_TENSOR_ELEMENTS:raise ValueError('NPU sort retains the 16777216-element metadata bound')
        shape=tuple(x.shape)
        npu._check_out(values,shape,x.dtype,(x,indices));npu._check_out(indices,shape,torch.int64,(x,values))
        if (values is None)!=(indices is None):raise ValueError('NPU sort requires both output tensors')
        if not x.numel():return (torch.empty_like(x),torch.empty(shape,dtype=torch.int64,device=x.device)) if values is None else (values,indices)
        logical=shape or (1,);width=logical[axis];order=tuple(a for a in range(len(logical)) if a!=axis)+(axis,)
        matrix=npu._materialize(x.permute(order) if shape else x.view(1))
        integer=x.dtype==torch.int64
        if not integer:matrix=npu._cast(matrix,torch.float32)
        matrix=matrix.view(-1,width)
        rows=matrix.shape[0];vals=[];ids=[]
        bound=8192 if integer else 10922
        if width<=bound:
            batch=bound//width
            for start in range(0,rows,batch):
                count=min(batch,rows-start)
                data=npu._slice(matrix.view(-1).view(torch.float32),2*start*width,2*count*width).view(torch.int64) if integer else npu._slice(matrix,start*width,count*width)
                v,i=_packet(data,rows=count,width=width,descending=descending)
                vals.append(v);ids.append(i)
        else:
            for row in range(rows):
                data=npu._slice(matrix.view(-1).view(torch.float32),2*row*width,2*width).view(torch.int64) if integer else npu._slice(matrix,row*width,width)
                v,i=_wide_row(data,descending);vals.append(v);ids.append(i)
        arranged=tuple(logical[a] for a in order);inverse=tuple(order.index(a) for a in range(len(logical)))
        result=npu._cat(vals,0).view(arranged)
        if not integer:result=npu._cast(result,x.dtype)
        index=npu._cat(ids,0).view(arranged)
        result=npu._materialize(result.permute(inverse)).view(shape);index=npu._materialize(index.permute(inverse)).view(shape)
        return (npu._copy(values,result),npu._copy(indices,index)) if values is not None else (result,index)


def argsort(x,dim=-1,descending=False,*,stable=False,out=None):
    with npu._lock,torch.no_grad():
        _axis(x,dim);npu._check_out(out,tuple(x.shape),torch.int64,(x,))
        result=sort(x,dim,descending,stable=stable)[1]
        return npu._copy(out,result) if out is not None else result


def maxmin(x,dim,keepdim=False,*,kind,values=None,indices=None):
    with npu._lock:
        axis=_axis(x,dim);shape=tuple(npu.reduction_geometry(tuple(x.shape),axis,keepdim)[1])
        npu._check_out(values,shape,x.dtype,(x,indices));npu._check_out(indices,shape,torch.int64,(x,values))
        if (values is None)!=(indices is None):raise ValueError('NPU max/min requires both output tensors')
        result,index=npu._extreme_reduce(x,axis,keepdim,kind=kind,pair=True)
        return (npu._copy(values,result),npu._copy(indices,index)) if values is not None else (result,index)


def integer_extreme(x,dim,keepdim,kind,arg=False,pair=False,out=None):
    axes,shape=npu.reduction_geometry(tuple(x.shape),dim,keepdim);logical=tuple(x.shape) or (1,)
    npu._check_out(out,shape,torch.int64 if arg else x.dtype,(x,))
    width=prod(logical[a] for a in axes)
    if not width:raise ValueError('Cannot reduce an empty extrema axis')
    order=tuple(a for a in range(len(logical)) if a not in axes)+axes
    matrix=npu._materialize(x.view(logical).permute(order)).view(-1,width)
    values,indices=sort(matrix,1,kind=='max')
    value=npu._materialize(values.narrow(1,0,1)).view(shape);index=npu._materialize(indices.narrow(1,0,1)).view(shape)
    result=index if arg else value
    return (value,index) if pair else npu._copy(out,result) if out is not None else result


class IndexedSelection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,dim,descending,keepdim,kind):
        ctx.set_materialize_grads(False)
        result,index=sort(x,dim,descending) if kind=='sort' else maxmin(x,dim,keepdim,kind=kind)
        ctx.shape=tuple(x.shape);ctx.axis=dim%(x.ndim or 1);ctx.kind=kind
        ctx.save_for_backward(index);ctx.mark_non_differentiable(index)
        return result.detach(),index

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient,unused):
        index,=ctx.saved_tensors
        with npu._lock:
            shape=ctx.shape or (1,);axis=ctx.axis;order=tuple(a for a in range(len(shape)) if a!=axis)+(axis,)
            selected=list(shape);selected[axis]=shape[axis] if ctx.kind=='sort' else 1
            g=npu._materialize(gradient.view(selected).permute(order));ids=npu._materialize(index.view(selected).permute(order))
            result=npu._selection_backward_planes(g,ids,tuple(shape[a] for a in order),selected[axis],shape[axis],accumulate=False)
            result=npu._materialize(result.permute(tuple(order.index(a) for a in range(len(shape)))))
            return result.view(ctx.shape),None,None,None,None


class GlobalExtreme(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,kind):
        value=npu._extreme_reduce(x,kind=kind);ctx.save_for_backward(x,value)
        return value.detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        x,value=ctx.saved_tensors
        with npu._lock:
            mask=npu._logical(npu._compare(x,value,kind='eq'),
                npu._logical(npu._compare(x,x,kind='ne'),npu._compare(value,value,kind='ne'),kind='and'),kind='or')
            count=npu._reduce(npu._mask_cast(mask,torch.float32),kind='sum')
            scale=npu._cast(npu._binary_op(npu._cast(npu._materialize(gradient),torch.float32),count,kind='div'),x.dtype)
            return npu._where(mask,scale,0.),None


def register(aten,autograd):
    from torch_npu_autograd import _active,Extrema
    for name in ('sort','sort.stable','sort.values','sort.values_stable'):aten.impl(name,sort)
    for name in ('sort','sort.stable'):
        autograd.impl(name,lambda x,dim=-1,descending=False,stable=None:
            IndexedSelection.apply(x,dim,descending,False,'sort') if _active(x) else sort(x,dim,descending,stable=stable))
    for name in ('argsort','argsort.stable','argsort.stable_out'):
        aten.impl(name,argsort);autograd.impl(name,argsort)
    for kind in ('max','min'):
        def pair(x,dim,keepdim=False,kind=kind,**outputs):
            names=('max','max_values') if kind=='max' else ('min','min_indices')
            return maxmin(x,dim,keepdim,kind=kind,values=outputs.get(names[0]),indices=outputs.get(names[1]))
        aten.impl(kind+'.dim',pair);aten.impl(kind+('.dim_max' if kind=='max' else '.dim_min'),pair)
        autograd.impl(kind+'.dim',lambda x,dim,keepdim=False,kind=kind:
            IndexedSelection.apply(x,dim,False,keepdim,kind) if _active(x) else maxmin(x,dim,keepdim,kind=kind))
        aten.impl(kind,lambda x,kind=kind:npu._extreme_reduce(x,kind=kind))
        aten.impl(kind+'.unary_out',lambda x,*,out,kind=kind:npu._extreme_reduce(x,kind=kind,out=out))
        autograd.impl(kind,lambda x,kind=kind:
            GlobalExtreme.apply(x,kind) if _active(x) else npu._extreme_reduce(x,kind=kind))
        other='maximum' if kind=='max' else 'minimum'
        aten.impl(kind+'.other',lambda x,y,other=other:npu._extreme(x,y,kind=other))
        aten.impl(kind+'.out',lambda x,y,*,out,other=other:npu._extreme(x,y,kind=other,out=out))
        autograd.impl(kind+'.other',lambda x,y,other=other:
            Extrema.apply(x,y,other) if _active(x,y) else npu._extreme(x,y,kind=other))

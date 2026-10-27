"""Device-side kthvalue, median, mode and quantiles over bounded sorted rows."""
import operator
from math import prod,isfinite
import numpy as np
import torch
from torch.autograd.function import once_differentiable
import torch_intel_npu as npu
from torch_npu_sort import sort,_axis

MODES={'kthvalue':0,'median':1,'mode':2,'quantile':3,'nanmedian':4,'nanquantile':5}


def _stat(x,dim,keepdim,kind,q,interpolation):
    npu._validate_read(x);axis=_axis(x,-1 if dim is None else dim)
    if x.dtype not in (torch.float16,torch.float32,torch.int64,torch.bool):raise NotImplementedError('NPU order statistics currently requires FP16/FP32/INT64/BOOL')
    if kind in ('quantile','nanquantile') and x.dtype!=torch.float32:raise ValueError('NPU quantile requires FP32, as Torch does for floating storage supported here')
    if isinstance(q,torch.Tensor):
        npu._validate_float_read(q)
        if q.dtype!=x.dtype or q.ndim>1 or q.requires_grad:raise ValueError('NPU quantile q requires same dtype, rank 0/1, without gradients')
        queries=q.numel();vector=q.ndim==1;query=npu._materialize(q).view(-1)
    else:
        if kind in ('quantile','nanquantile') and (not isfinite(q) or not 0<=q<=1):raise ValueError('Quantile q must lie in [0,1]')
        queries=1;vector=False;query=npu._scalar(float(q),torch.float32).view(1)
    if queries>128:raise ValueError('NPU statistics supports up to 128 simultaneous quantiles')
    if interpolation not in ('linear','lower','higher','midpoint','nearest'):raise ValueError('Unknown quantile interpolation')
    if dim is None:
        logical=(x.numel(),);axis=0;values=npu._materialize(x).view(logical);shape=(1,)*x.ndim if keepdim else ()
    else:
        logical=tuple(x.shape) or (1,);values=x.view(logical);shape=tuple(1 if a==axis else n for a,n in enumerate(x.shape)) if keepdim else tuple(n for a,n in enumerate(x.shape) if a!=axis)
    width=logical[axis]
    if width>32760:raise ValueError('NPU order-statistic sorted row exceeds 32760 elements')
    if kind=='kthvalue' and (q<0 or q>=width or int(q)!=q):raise ValueError('kthvalue k must lie in [1,axis length]')
    if not width:
        if dim is not None or kind in ('kthvalue','mode','quantile','nanquantile'):raise ValueError('Cannot reduce an empty statistics axis')
        return npu._scalar(float('nan'),x.dtype).view(shape),None,None,None,logical,axis,vector
    if not queries:
        rows=prod(logical)//width
        value=torch.empty((0,)+shape,dtype=x.dtype,device=x.device);ids=torch.empty((rows,0),dtype=torch.int64,device=x.device)
        return value,ids,ids,torch.empty((rows,0),dtype=torch.float32,device=x.device),logical,axis,vector
    order=tuple(a for a in range(len(logical)) if a!=axis)+(axis,)
    sorted_values,indices=sort(values,axis)
    integer=x.dtype==torch.int64;raw=npu._materialize(sorted_values.permute(order)).view(-1,width)
    if not integer:raw=npu._cast(raw,torch.float32)
    ids=npu._materialize(indices.permute(order)).view(-1,width);parts=[];first=[];second=[];weights=[]
    for row in range(raw.shape[0]):
        count=width*(2 if integer else 1)
        data=npu._slice(raw.view(-1).view(torch.float32),row*count,count) if integer else npu._slice(raw,row*width,width)
        index=npu._slice(ids.view(-1).view(torch.float32),2*row*width,2*width)
        size=max(16,5+count+queries,2*width,8*queries+1)
        header=torch.tensor([MODES[kind],width,int(integer),queries,('linear','lower','higher','midpoint','nearest').index(interpolation)],dtype=torch.float32).to(x.device)
        def pad(parts):
            used=sum(v.numel() for v in parts)
            if used<size:parts.append(npu._broadcast(npu._scalar(0.,torch.float32),(size-used,)))
            return npu._concat(parts,(1,size),torch.float32)
        a=pad([header,data,query]);b=pad([index]);result=npu._run(('order-stats-f32','f32',(1,size)),[a,b],(1,size),torch.float32)
        if npu._slice(result,size-1,1).item():raise ValueError('NPU statistic rejected q or row metadata')
        offsets=np.arange(queries,dtype=np.int64)[:,None]*8
        value=npu._gather_window(result,offsets+np.arange(2 if integer else 1),1).view(-1)
        weight=npu._gather_window(result,offsets+6,1).view(-1)
        if kind in ('quantile','nanquantile'):
            other=npu._gather_window(result,offsets+7,1).view(-1)
            difference=npu._execute('sub',other,value)
            low=npu._execute('add',value,npu._execute('mul',difference,weight))
            high=npu._execute('sub',other,npu._execute('mul',difference,npu._binary_op(1.,weight,kind='sub')))
            mixed=npu._where(npu._compare(weight,.5,kind='lt'),low,high)
            if interpolation in ('linear','midpoint'):value=mixed
        parts.append(value.view(torch.int64) if integer else npu._cast(value,x.dtype))
        first.append(npu._gather_window(result,offsets+np.arange(2,4),1).view(-1).view(torch.int64))
        second.append(npu._gather_window(result,offsets+np.arange(4,6),1).view(-1).view(torch.int64))
        weights.append(weight)
    rows=raw.shape[0];target=((queries,)+shape) if vector else shape
    output=npu._cat(parts,0).view(rows,queries) if parts else torch.empty((rows,queries),dtype=x.dtype,device=x.device)
    output=npu._materialize(output.t()).view(target)
    f=npu._cat(first,0).view(rows,queries) if first else torch.empty((rows,queries),dtype=torch.int64,device=x.device)
    s=npu._cat(second,0).view(rows,queries) if second else torch.empty_like(f)
    w=npu._concat(weights,(rows,queries),torch.float32)
    return output,f,s,w,logical,axis,vector


class Statistic(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,dim,keepdim,kind,q,interpolation):
        value,a,b,w,logical,axis,vector=_stat(x,dim,keepdim,kind,q,interpolation)
        ctx.shape=tuple(x.shape);ctx.logical=logical;ctx.axis=axis;ctx.global_median=dim is None and kind in ('median','nanmedian')
        ctx.weighted=kind in ('quantile','nanquantile') and interpolation in ('linear','midpoint')
        if ctx.global_median:ctx.save_for_backward(x,value)
        else:ctx.save_for_backward(a,b,w)
        index=None if a is None else a.view(value.shape)
        if index is not None:ctx.mark_non_differentiable(index)
        return value.detach(),index

    @staticmethod
    @once_differentiable
    def backward(ctx,g,unused):
        with npu._lock:
            if ctx.global_median:
                x,value=ctx.saved_tensors
                mask=npu._logical(npu._compare(x,value,kind='eq'),npu._logical(npu._compare(x,x,kind='ne'),npu._compare(value,value,kind='ne'),kind='and'),kind='or')
                count=npu._reduce(npu._mask_cast(mask,torch.float32),kind='sum')
                # Scalar median shares gradients across equal values, including NaNs.
                scaled=npu._binary_op(npu._cast(npu._materialize(g),torch.float32),count,kind='div')
                result=npu._where(mask,scaled,0.)
                return npu._cast(result,g.dtype),None,None,None,None,None
            a,b,w=ctx.saved_tensors;rows,queries=a.shape
            if not a.numel():return npu._fill(torch.empty(ctx.shape,dtype=g.dtype,device=g.device),0.),None,None,None,None,None
            gradient=npu._materialize(g.reshape(queries,rows).t())
            if ctx.weighted:
                g1=npu._execute('mul',npu._cast(gradient,torch.float32),npu._binary_op(1.,w,kind='sub'))
                g2=npu._execute('mul',npu._cast(gradient,torch.float32),w)
                grad=npu._cast(npu._cat([g1,g2],1),g.dtype);indices=npu._cat([a,b],1);per=2*queries
            else:grad=gradient;indices=a;per=queries
            order=tuple(i for i in range(len(ctx.logical)) if i!=ctx.axis)+(ctx.axis,)
            shape=tuple(ctx.logical[i] for i in order)
            result=npu._selection_backward(grad,indices,shape,per,ctx.logical[ctx.axis],accumulate=ctx.weighted)
            return npu._materialize(result.permute(tuple(order.index(i) for i in range(len(order))))).view(ctx.shape),None,None,None,None,None


def statistic(x,dim=None,keepdim=False,*,kind,q=0,interpolation='linear',out=None,values=None,indices=None):
    from torch_npu_autograd import _active
    paired=kind in ('kthvalue','mode') or dim is not None and kind in ('median','nanmedian')
    with torch.no_grad():
        axis=_axis(x,-1 if dim is None else dim)
        shape=(1,)*x.ndim if keepdim and dim is None else () if dim is None else tuple(1 if a==axis else n for a,n in enumerate(x.shape)) if keepdim else tuple(n for a,n in enumerate(x.shape) if a!=axis)
        if kind in ('quantile','nanquantile') and isinstance(q,torch.Tensor) and q.ndim==1:shape=(q.numel(),)+shape
        npu._check_out(out,shape,x.dtype,(x,q))
        npu._check_out(values,shape,x.dtype,(x,indices));npu._check_out(indices,shape,torch.int64,(x,values))
        if paired and (values is None)!=(indices is None):raise ValueError('Statistics requires paired out tensors')
    if kind=='kthvalue':q=operator.index(q)-1
    if _active(x):
        if out is not None or values is not None or indices is not None:raise ValueError('Statistics out= does not support autograd')
        result,index=Statistic.apply(x,dim,keepdim,kind,q,interpolation)
    else:
        with npu._lock,torch.no_grad():
            result,index,_,_,_,_,_=_stat(x,dim,keepdim,kind,q,interpolation)
            if index is not None:index=index.view(result.shape)
    if paired:
        if (values is None)!=(indices is None):raise ValueError('Statistics requires paired out tensors')
        npu._check_out(values,tuple(result.shape),x.dtype,(x,indices));npu._check_out(indices,tuple(result.shape),torch.int64,(x,values))
        return (npu._copy(values,result),npu._copy(indices,index)) if values is not None else (result,index)
    npu._check_out(out,tuple(result.shape),x.dtype,(x,))
    return npu._copy(out,result) if out is not None else result


def register(aten,autograd):
    for library in (aten,autograd):
        for kind in ('median','nanmedian'):
            library.impl(kind,lambda x,kind=kind:statistic(x,kind=kind))
            library.impl(kind+'.out',lambda x,*,out,kind=kind:statistic(x,kind=kind,out=out))
            for suffix in ('.dim','.dim_values'):
                library.impl(kind+suffix,lambda x,dim,keepdim=False,kind=kind,**outputs:statistic(x,dim,keepdim,kind=kind,**outputs))
        for suffix in ('','.values'):
            library.impl('kthvalue'+suffix,lambda x,k,dim=-1,keepdim=False,**outputs:statistic(x,dim,keepdim,kind='kthvalue',q=k,**outputs))
            library.impl('mode'+suffix,lambda x,dim=-1,keepdim=False,**outputs:statistic(x,dim,keepdim,kind='mode',**outputs))
        for kind in ('quantile','nanquantile'):
            for suffix in ('','.scalar','.out','.scalar_out'):
                library.impl(kind+suffix,lambda x,q,dim=None,keepdim=False,interpolation='linear',kind=kind,**outputs:
                             statistic(x,dim,keepdim,kind=kind,q=q,interpolation=interpolation,**outputs))

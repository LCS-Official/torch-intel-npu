"""Larger storage, exact integer arithmetic, dynamic indexing and writable views.

Tensor payload stays on the NPU. Dynamic result sizes/validation flags synchronize.
"""
from math import prod, log
from functools import wraps
import operator
import torch
from torch.autograd.function import once_differentiable
import torch_intel_npu as npu

DTYPES=(torch.float32,torch.float16,torch.int64,torch.bool,torch.int8,torch.int32,
        torch.int16,None,torch.uint8,torch.bfloat16)
INTEGERS=(torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool)
_old={name:getattr(npu,'_'+name) for name in ('cast','copy','fill','binary_op','execute','compare','logical','reduce','mask_cast','math_unary','where','float_predicate')}


def packet(kind,header,values,output_bytes):
    header=tuple(header)+(0,)*(15-len(header))+(output_bytes+4,)
    sizes=tuple(v.numel()*v.element_size() for v in values)
    words=(max(64+sum(sizes),output_bytes+4)+7)//8*2
    raw=npu._run(('byte-pack','f32',(1,words),header,sizes),
        [v.view(-1).view(torch.uint8).view(1,-1) for v in values],(1,words),torch.float32)
    return npu._run((kind,'f32',(1,words)),[raw,raw],(1,words),torch.float32)


def typed_cast(x,dtype,out=None):
    npu._validate(x,storage=True);npu._check_out(out,tuple(x.shape),dtype,[x])
    if dtype is None or dtype not in DTYPES:raise ValueError('Unsupported NPU conversion dtype')
    limit=npu._storage_limit(dtype)
    if x.numel()>limit:raise ValueError('NPU conversion output exceeds dtype storage bound')
    if dtype==x.dtype:return npu._copy(out,x) if out is not None and out is not x else x
    if x.dtype==torch.bool:
        result=_old['mask_cast'](x,dtype if dtype in (torch.float16,torch.float32,torch.int64) else torch.float32)
        if result.dtype!=dtype:result=typed_cast(result,dtype)
        return npu._copy(out,result) if out is not None else result
    pieces=[];width=torch.empty((),dtype=dtype).element_size()
    for start in range(0,x.numel(),4096):
        count=min(4096,x.numel()-start)
        tile=npu._storage_slice(x,start,count)
        raw=packet('typed-cast',(count,0,DTYPES.index(x.dtype),DTYPES.index(dtype)),[tile],count*width)
        pieces.append(npu._slice(raw.view(-1).view(torch.uint8),0,count*width).view(dtype))
    result=npu._concat(pieces,tuple(x.shape),dtype)
    return npu._copy(out,result) if out is not None else result


def cast(x,dtype,out=None):
    if x.dtype==torch.bfloat16 or dtype==torch.bfloat16 or x.dtype in INTEGERS and dtype!=x.dtype or dtype in INTEGERS and dtype!=x.dtype:
        return typed_cast(x,dtype,out)
    if x.dtype in INTEGERS:return npu._copy(out,x) if out is not None else x
    return _old['cast'](x,dtype,out)


def _operand(value,dtype,shape):
    if isinstance(value,torch.Tensor) and value.device.type=='intel_npu':return npu._broadcast(cast(npu._materialize(value),dtype),shape)
    if isinstance(value,torch.Tensor):
        if value.device.type!='cpu' or value.ndim or value.requires_grad:raise ValueError('Only CPU scalar constants may join NPU arithmetic')
        value=value.item()
    return npu._broadcast(npu._scalar(value,dtype),shape)


def integer_binary(x,y,kind,*,out=None,alpha=1,rounding_mode=None):
    shape=tuple(torch.broadcast_shapes(*(v.shape if isinstance(v,torch.Tensor) else () for v in (x,y))))
    dtype=torch.result_type(x,y)
    if kind=='div' and rounding_mode is None:
        return npu._binary_op(_operand(x,torch.get_default_dtype(),shape),_operand(y,torch.get_default_dtype(),shape),kind='div',out=out)
    if dtype not in INTEGERS:raise ValueError('Integer operation needs an integer result dtype')
    if type(alpha) is not int:raise TypeError('Integer alpha must be an integer')
    if dtype==torch.bool:
        if kind in ('eq','ne','lt','le','gt','ge'):return _old['compare'](x,y,kind=kind,out=out)
        if kind not in ('add','mul','and','or','xor','not'):raise RuntimeError('This arithmetic operation is unsupported for BOOL')
        npu._check_out(out,shape,torch.bool,[x,y])
        a,b=_operand(x,dtype,shape),_operand(y,dtype,shape)
        if kind=='add' and alpha==0:return npu._copy(out,a) if out is not None else npu._clone(a)
        return _old['logical'](a,None if kind=='not' else b,kind='or' if kind=='add' else 'and' if kind=='mul' else kind,out=out)
    codes={'add':0,'sub':1,'mul':2,'and':3,'or':4,'xor':5,'not':6,'neg':7,'abs':8,
           'floor_divide':9,'remainder':10,'trunc':11,'eq':12,'ne':13,'lt':14,'le':15,'gt':16,'ge':17,'sign':18,'pow':21}
    if kind=='pow' and type(y) is int and y<0:raise RuntimeError('Integers to negative integer powers are not allowed')
    op=codes['floor_divide' if kind=='div' and rounding_mode=='floor' else 'trunc' if kind=='div' else kind]
    if alpha!=1 and kind in ('add','sub'):op=19 if kind=='add' else 20
    result_dtype=torch.bool if kind in ('eq','ne','lt','le','gt','ge') else dtype
    npu._check_out(out,shape,result_dtype,[x,y]);a,b=(_operand(v,dtype,shape) for v in (x,y));pieces=[]
    width=torch.empty((),dtype=result_dtype).element_size();bits=alpha%(1<<64)
    for start in range(0,prod(shape),4096):
        count=min(4096,prod(shape)-start)
        raw=packet('typed-math',(count,op,DTYPES.index(dtype),DTYPES.index(result_dtype),bits&0xffffffff,bits>>32),
                   [npu._storage_slice(v,start,count) for v in (a,b)],count*width)
        if op in (9,10,11) and npu._slice(raw.view(-1),raw.numel()-1,1).view(torch.int32).item():raise ZeroDivisionError('NPU integer division by zero')
        pieces.append(npu._slice(raw.view(-1).view(torch.uint8),0,count*width).view(result_dtype))
    result=npu._concat(pieces,shape,result_dtype)
    return npu._copy(out,result) if out is not None else result


def binary(x,y,*,kind,out=None,alpha=1):
    if kind=='sub' and any(type(v) is bool or isinstance(v,torch.Tensor) and v.dtype==torch.bool for v in (x,y)):
        raise RuntimeError('Subtraction with a bool operand is unsupported')
    tensors=[v for v in (x,y) if isinstance(v,torch.Tensor) and v.device.type=='intel_npu']
    dtype=torch.result_type(x,y)
    if dtype in INTEGERS:return integer_binary(x,y,kind,out=out,alpha=alpha)
    if any(v.dtype in INTEGERS or v.dtype==torch.bfloat16 for v in tensors):
        shape=tuple(torch.broadcast_shapes(*(v.shape if isinstance(v,torch.Tensor) else () for v in (x,y))))
        work=torch.float32 if dtype==torch.bfloat16 else dtype
        result=_old['binary_op'](*[_operand(v,work,shape) for v in (x,y)],kind=kind,alpha=alpha)
        result=cast(result,dtype)
        return npu._copy(out,result) if out is not None else result
    return _old['binary_op'](x,y,kind=kind,out=out,alpha=alpha)


def execute(kind,x,y=None,*,out=None):
    if x.dtype in INTEGERS and kind in ('neg','abs','sign'):return integer_binary(x,x,kind,out=out)
    if x.dtype==torch.bfloat16:
        result=_old['execute'](kind,cast(npu._materialize(x),torch.float32),None if y is None else cast(npu._materialize(y),torch.float32))
        return cast(result,x.dtype,out)
    return _old['execute'](kind,x,y,out=out)


def compare(x,y,*,kind,out=None):
    if any(isinstance(v,torch.Tensor) and v.dtype in (*INTEGERS,torch.bfloat16) for v in (x,y)):
        if torch.result_type(x,y) in INTEGERS:return integer_binary(x,y,kind,out=out)
        shape=tuple(torch.broadcast_shapes(*(v.shape if isinstance(v,torch.Tensor) else () for v in (x,y))))
        return _old['compare'](*[_operand(v,torch.float32,shape) for v in (x,y)],kind=kind,out=out)
    return _old['compare'](x,y,kind=kind,out=out)


def logical(x,y=None,*,kind,out=None):
    with torch.no_grad():return _old['logical'](cast(npu._materialize(x),torch.bool),None if y is None else cast(npu._materialize(y),torch.bool),kind=kind,out=out)


def where(condition,x,y,*,out=None):
    dtype=torch.result_type(x,y)
    if dtype not in (torch.float16,torch.float32,torch.int64,torch.bool) or any(isinstance(v,torch.Tensor) and v.dtype in INTEGERS for v in (x,y)):
        shape=tuple(torch.broadcast_shapes(condition.shape,*(v.shape if isinstance(v,torch.Tensor) else () for v in (x,y))))
        work=torch.int64 if dtype in INTEGERS and dtype!=torch.bool else torch.float32 if dtype==torch.bfloat16 else dtype
        result=_old['where'](condition,*[_operand(v,work,shape) for v in (x,y)])
        return cast(result,dtype,out)
    return _old['where'](condition,x,y,out=out)


def reduce(x,dim=None,keepdim=False,*,dtype=None,out=None,kind):
    if x.dtype==torch.bfloat16:
        return cast(_old['reduce'](cast(npu._materialize(x),torch.float32),dim,keepdim,kind=kind),dtype or x.dtype,out)
    if x.dtype not in INTEGERS:return _old['reduce'](x,dim,keepdim,dtype=dtype,out=out,kind=kind)
    if kind!='sum' and dtype not in (torch.float32,torch.float16):raise ValueError('Integer mean requires floating dtype')
    if dtype in (torch.float32,torch.float16):return _old['reduce'](cast(npu._materialize(x),dtype),dim,keepdim,kind=kind,out=out)
    if dtype==torch.bool:return boolean_reduce(x,dim,keepdim,out=out,kind='any')
    axes,shape=npu.reduction_geometry(tuple(x.shape),dim,keepdim);result_dtype=dtype or torch.int64
    if result_dtype not in INTEGERS:raise ValueError('Unsupported NPU integer reduction dtype')
    npu._check_out(out,shape,result_dtype,[x]);order=tuple(a for a in range(x.ndim) if a not in axes)+axes
    matrix=npu._materialize(x.permute(order));cols=prod(x.shape[a] for a in axes);rows=prod(shape);pieces=[]
    if cols==0:return npu._fill(torch.empty(shape,dtype=result_dtype,device=x.device),0)
    for r in range(rows):
        total=None
        for start in range(0,cols,4096):
            count=min(4096,cols-start);tile=npu._storage_slice(matrix,r*cols+start,count)
            if tile.dtype==torch.bool:tile=_old['mask_cast'](tile,torch.int64)
            raw=packet('typed-reduce',(count,0,DTYPES.index(tile.dtype),DTYPES.index(result_dtype),1,count),[tile],8)
            value=npu._slice(raw.view(-1).view(torch.uint8),0,torch.empty((),dtype=result_dtype).element_size()).view(result_dtype)
            total=value if total is None else integer_binary(total,value,'add')
        pieces.append(total)
    result=npu._concat(pieces,shape,result_dtype)
    return npu._copy(out,result) if out is not None else result


def fill(x,value):
    if x.dtype in (*npu.bridge._DTYPES,torch.int64,torch.bool) and x.storage_offset()==0 and x.numel()*x.element_size()==x.untyped_storage().nbytes():return _old['fill'](x,value)
    with npu._lock:
        npu._validate_read(x)
        if x.numel()==0:return x
        scalar=npu._scalar(value,x.dtype)
        result=npu._materialize(scalar.expand(tuple(x.shape)))
        return copy(x,result)


def _address_metadata(x):
    import numpy as np
    offsets=np.full(tuple(x.shape),x.storage_offset(),dtype=np.int64)
    for axis,(count,stride) in enumerate(zip(x.shape,x.stride())):
        shape=[1]*x.ndim;shape[axis]=count
        offsets+=np.arange(count,dtype=np.int64).reshape(shape)*stride
    return offsets


def _writable(x):
    npu._validate_read(x);span=1
    for axis in sorted(range(x.ndim),key=lambda a:x.stride(a)):
        if x.shape[axis]<=1:continue
        if x.stride(axis)<span:raise ValueError('NPU destination view overlaps itself')
        span+=(x.shape[axis]-1)*x.stride(axis)


def dynamic(source,ids,values=None,*,accumulate=False):
    """Flat exact byte gather or deterministic last-write-wins scatter."""
    # shortcut: scan source/target blocks; replace after direct indexed DMA is validated.
    source=npu._materialize(source).view(-1);ids=npu._materialize(ids).view(-1)
    if ids.dtype!=torch.int64:raise TypeError('NPU indices must be INT64')
    if values is not None:
        values=npu._materialize(values).view(-1)
        if values.dtype!=source.dtype or values.numel()!=ids.numel():raise ValueError('Scatter values differ from index shape/dtype')
    width=source.element_size();pieces=[]
    if ids.numel()==0:return npu._clone(source) if values is not None else torch.empty((0,),dtype=source.dtype,device=source.device)
    if not source.numel():raise IndexError('Cannot index an empty NPU tensor')
    outer=range(0,ids.numel(),2048) if values is None else range(0,source.numel(),8192)
    for start in outer:
        count=min(2048,ids.numel()-start) if values is None else min(8192,source.numel()-start)
        initial=None if values is None else npu._storage_slice(source,start,count)
        inner=range(0,source.numel(),8192) if values is None else range(0,ids.numel(),2048)
        for base in inner:
            span=min(8192,source.numel()-base) if values is None else count
            n=count if values is None else min(2048,ids.numel()-base)
            index=npu._storage_slice(ids,start if values is None else base,n)
            data=npu._storage_slice(source,base,span) if values is None else npu._storage_slice(values,base,n)
            header=(n,0 if values is None else 2 if accumulate else 1,width,base if values is None else start,span,source.numel(),DTYPES.index(source.dtype))
            sizes=(index.numel()*8,data.numel()*width)
            words=(max(64+sum(sizes),count*width+4)+7)//8*2
            h=header+(0,)*(15-len(header))+(count*width+4,)
            raw=npu._run(('byte-pack','f32',(1,words),h,sizes),[index.view(torch.uint8).view(1,-1),data.view(torch.uint8).view(1,-1)],(1,words),torch.float32)
            if initial is None:second=torch.zeros((words,),device=source.device,dtype=torch.float32)
            else:
                b=initial.view(torch.uint8)
                second=npu._concat([b,torch.zeros((words*4-b.numel(),),dtype=torch.uint8,device=source.device)],(words*4,),torch.uint8).view(torch.float32)
            result=npu._run(('dynamic-index','f32',(1,words)),[raw,second],(1,words),torch.float32)
            if npu._slice(result.view(-1),words-1,1).view(torch.int32).item():raise IndexError('NPU index out of bounds')
            initial=npu._slice(result.view(-1).view(torch.uint8),0,count*width).view(source.dtype)
        pieces.append(initial)
    return npu._concat(pieces,(ids.numel() if values is None else source.numel(),),source.dtype)


def copy(dst,src,non_blocking=False):
    if non_blocking:raise ValueError('NPU copies remain synchronous')
    if torch.is_grad_enabled() and (dst.requires_grad or src.requires_grad):raise ValueError('Use a registered autograd copy operator or no_grad')
    if dst.device.type=='intel_npu' and (dst.storage_offset()!=0 or not dst.is_contiguous() or dst.numel()*dst.element_size()!=dst.untyped_storage().nbytes()):
        _writable(dst)
        if src.device.type=='cpu':src=src.to(device=dst.device,dtype=dst.dtype)
        src=cast(npu._materialize(src),dst.dtype)
        if tuple(torch.broadcast_shapes(src.shape,dst.shape))!=tuple(dst.shape):raise ValueError('Copy cannot broadcast to destination')
        values=npu._materialize(src.expand(tuple(dst.shape)))
        if dst.storage_offset()==0 and dst.numel()*dst.element_size()==dst.untyped_storage().nbytes() and npu._is_dense(dst):
            # Complete dense gradient destinations need no dynamic scatter or validation readback.
            base,axes=npu._physical_view(dst)
            copy(base,npu._materialize(values.permute(axes)))
            return dst
        backing=dst.as_strided((dst.untyped_storage().nbytes()//dst.element_size(),),(1,),storage_offset=0)
        ids=torch.from_numpy(_address_metadata(dst).reshape(-1)).to(dst.device)
        result=dynamic(backing,ids,values)
        copy(backing,result)
        return dst
    if src.shape!=dst.shape:
        if tuple(torch.broadcast_shapes(src.shape,dst.shape))!=tuple(dst.shape):raise ValueError('Copy cannot broadcast to destination')
        if src.device.type=='cpu':src=src.to(dst.device)
        src=npu._materialize(src.expand(tuple(dst.shape)))
    if src.dtype!=dst.dtype:
        if src.device.type=='cpu':src=src.to('intel_npu')
        src=cast(npu._materialize(src),dst.dtype)
    if src.dtype in (torch.int8,torch.uint8,torch.int16,torch.int32,torch.bfloat16):
        if src.device.type=='intel_npu':src=npu._materialize(src)
        if src.device.type==dst.device.type=='intel_npu':
            if src.data_ptr()==dst.data_ptr():return dst
            _old['copy'](dst.view(-1).view(torch.uint8),src.view(-1).view(torch.uint8),False)
            return dst
        return npu._C.copy(dst,src,False)
    return _old['copy'](dst,src,False)


def normalize(ids,size):
    ids=cast(npu._materialize(ids),torch.int64);pieces=[]
    for start in range(0,ids.numel(),4096):
        count=min(4096,ids.numel()-start)
        raw=packet('index-offset',(count,0,8,0,size),[npu._storage_slice(ids,start,count)],count*8)
        if npu._slice(raw.view(-1),raw.numel()-1,1).view(torch.int32).item():raise IndexError('NPU index out of bounds')
        pieces.append(npu._slice(raw.view(-1).view(torch.uint8),0,count*8).view(torch.int64))
    return npu._concat(pieces,tuple(ids.shape),torch.int64)


def nonzero(x,*,out=None):
    npu._validate_read(x)
    mask=cast(npu._materialize(x),torch.bool).view(-1);pieces=[];count=0
    for start in range(0,mask.numel(),4096):
        size=min(4096,mask.numel()-start)
        raw=packet('compact-index',(size,0,1,start),[npu._slice(mask,start,size)],size*8)
        kept=npu._slice(raw.view(-1),raw.numel()-1,1).view(torch.int32).item()
        if kept:pieces.append(npu._slice(raw.view(-1).view(torch.uint8),0,kept*8).view(torch.int64));count+=kept
    flat=npu._concat(pieces,(count,),torch.int64)
    if not x.ndim:result=torch.empty((count,0),device=x.device,dtype=torch.int64)
    else:
        columns=[]
        for axis,size in enumerate(x.shape):
            stride=prod(x.shape[axis+1:]);columns.append(integer_binary(integer_binary(flat,stride,'floor_divide'),size,'remainder').view(count,1))
        result=npu._cat(columns,1)
    if out is not None:npu._check_out(out,tuple(result.shape),torch.int64,[x]);return copy(out,result)
    return result


def index_geometry(x,indices):
    """ATen advanced index layout, including separated axes and boolean masks."""
    tensors=[];axis=0
    for idx in indices:
        if idx is not None and idx.dtype==torch.bool and idx.ndim==0:
            x=x.unsqueeze(axis)
            idx=idx.reshape(1)
        if axis>=x.ndim:raise IndexError('Too many indices for NPU tensor')
        if idx is None:tensors.append(None);axis+=1;continue
        if idx.device.type=='cpu':idx=idx.to(x.device)
        if idx.device!=x.device:raise ValueError('Advanced indices must be on CPU or the same NPU')
        if idx.dtype==torch.bool:
            if tuple(idx.shape)!=tuple(x.shape[axis:axis+idx.ndim]):raise IndexError('Boolean index shape mismatch')
            nz=nonzero(idx)
            tensors.extend(npu._materialize(nz[:,a]) for a in range(idx.ndim));axis+=idx.ndim
        elif idx.dtype in (torch.int64,torch.int32):tensors.append(normalize(idx,x.shape[axis]));axis+=1
        else:raise IndexError('Advanced indices require INT64/INT32 or BOOL')
    tensors.extend([None]*(x.ndim-len(tensors)));selected=[a for a,v in enumerate(tensors) if v is not None]
    if not selected:raise ValueError('Advanced indexing requires an index tensor')
    broadcast=tuple(torch.broadcast_shapes(*(tensors[a].shape for a in selected)))
    continuous=selected==list(range(selected[0],selected[-1]+1));other=[a for a in range(x.ndim) if a not in selected]
    before=[a for a in other if a<selected[0]] if continuous else []
    after=[a for a in other if a not in before]
    shape=tuple(x.shape[a] for a in before)+broadcast+tuple(x.shape[a] for a in after)
    if len(shape)>npu.MAX_RANK:raise ValueError('Advanced index result rank exceeds NPU storage bound')
    offsets=torch.zeros(shape,dtype=torch.int64,device=x.device)
    for a in range(x.ndim):
        if a in selected:
            dims=(1,)*len(before)+broadcast+(1,)*len(after)
            coord=npu._materialize(tensors[a].expand(broadcast)).view(dims)
        else:
            position=before.index(a) if a in before else len(before)+len(broadcast)+after.index(a)
            dims=[1]*len(shape);dims[position]=x.shape[a]
            coord=torch.arange(x.shape[a],dtype=torch.int64).to(x.device).view(dims)
        offsets=integer_binary(offsets,integer_binary(coord,prod(x.shape[a+1:]),'mul'),'add')
    return offsets,shape


class AdvancedIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,ids,shape,opmath,originals):
        ctx.save_for_backward(ids,*originals);ctx.shape=tuple(x.shape);ctx.dtype=x.dtype;ctx.opmath=opmath
        return dynamic(x,ids).view(shape).detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        ids=ctx.saved_tensors[0]
        # Existing selection kernel preserves duplicate accumulation/half rounding.
        with npu._lock:
            source=npu._materialize(g).view(-1)
            if ctx.opmath or source.dtype==torch.bfloat16:source=cast(source,torch.float32)
            result=npu._selection_backward(source,ids.view(-1),ctx.shape,source.numel(),prod(ctx.shape),accumulate=True)
            return cast(result,ctx.dtype),None,None,None,None


def index(x,indices,*,out=None):
    from torch_npu_autograd import _active
    with torch.no_grad():ids,shape=index_geometry(x,indices)
    result=AdvancedIndex.apply(x,ids,shape,False,tuple(v for v in indices if v is not None)) if _active(x) else dynamic(x,ids).view(shape)
    if out is not None:
        if result.requires_grad:raise ValueError('Index out= does not support autograd')
        npu._check_out(out,shape,x.dtype,[x,*[v for v in indices if v is not None]])
        return copy(out,result)
    return result


class IndexedWrite(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,ids,values,accumulate,originals,opmath):
        ctx.save_for_backward(ids,*originals);ctx.accumulate=accumulate;ctx.shape=tuple(x.shape);ctx.values_shape=tuple(values.shape)
        if accumulate:
            initial=cast(npu._materialize(x),torch.float32) if opmath else x
            data=cast(npu._materialize(values),torch.float32) if opmath else values
            return cast(npu._selection_backward(data.view(-1),ids.view(-1),tuple(x.shape),ids.numel(),x.numel(),initial_value=initial),x.dtype).detach()
        return dynamic(x,ids,values).view(x.shape).detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        ids=ctx.saved_tensors[0]
        with npu._lock:
            gx=gv=None
            if ctx.needs_input_grad[0]:
                gx=npu._materialize(g) if ctx.accumulate else dynamic(g,ids,torch.zeros(tuple(ids.shape),dtype=g.dtype,device=g.device)).view(ctx.shape)
            if ctx.needs_input_grad[2]:gv=dynamic(g,ids).view(ctx.values_shape)
            return gx,None,gv,None,None,None


def index_put(x,indices,values,accumulate=False,*,inplace=False,out=None):
    from torch_npu_autograd import _active
    with torch.no_grad():ids,shape=index_geometry(x,indices)
    if values.dtype!=x.dtype or values.device!=x.device:raise ValueError('Index assignment requires matching device/dtype')
    if tuple(torch.broadcast_shapes(values.shape,shape))!=shape:raise ValueError('Index values cannot broadcast to selected shape')
    values=values.expand(shape).clone()
    if _active(x,values):
        if inplace or out is not None:raise NotImplementedError('Use out-of-place index_put for autograd')
        return IndexedWrite.apply(x,ids,values,accumulate,tuple(v for v in indices if v is not None),False)
    if accumulate:
        if x.dtype in INTEGERS:result=dynamic(x,ids,values,accumulate=True).view(x.shape)
        elif x.dtype in (torch.float16,torch.float32):result=npu._selection_backward(values.view(-1),ids.view(-1),tuple(x.shape),ids.numel(),x.numel(),initial_value=x)
        else:raise NotImplementedError('Accumulating BF16 index_put is not supported')
    else:result=dynamic(x,ids,values).view(x.shape)
    return copy(x,result) if inplace else copy(out,result) if out is not None else result


def scatter(x,dim,index,src,*,out=None,inplace=False):
    npu._check_out(out,tuple(x.shape),x.dtype,[x,index,src])
    dim=operator.index(dim)
    if not -x.ndim<=dim<x.ndim:raise IndexError('Scatter axis out of range')
    dim%=x.ndim
    if index.ndim!=x.ndim or any(index.shape[a]>x.shape[a] for a in range(x.ndim) if a!=dim):raise ValueError('Scatter index shape mismatch')
    if not isinstance(src,torch.Tensor):src=torch.full(index.shape,src,dtype=x.dtype,device=x.device)
    if src.ndim!=x.ndim or any(src.shape[a]<index.shape[a] for a in range(x.ndim)):raise ValueError('Scatter source too small')
    src=src[tuple(slice(0,n) for n in index.shape)]
    # Negative scatter indices are invalid (advanced indices may wrap).
    if index.dtype!=torch.int64:raise TypeError('Scatter indices must be INT64')
    if npu._boolean_reduce(npu._compare(index,0,kind='lt'),kind='any').item():raise IndexError('Negative scatter index')
    ids=torch.zeros(index.shape,dtype=torch.int64,device=x.device)
    for axis,size in enumerate(index.shape):
        if axis==dim:coord=normalize(index,x.shape[axis])
        else:
            shape=[1]*x.ndim;shape[axis]=size;coord=torch.arange(size,dtype=torch.int64).to(x.device).view(shape)
        ids=integer_binary(ids,integer_binary(coord,prod(x.shape[axis+1:]),'mul'),'add')
    from torch_npu_autograd import _active
    if _active(x,src):
        if inplace or out is not None:raise NotImplementedError('Use out-of-place scatter for autograd')
        return IndexedWrite.apply(x,ids,src,False,(index,),False)
    result=dynamic(x,ids,src).view(x.shape)
    return copy(x,result) if inplace else copy(out,result) if out is not None else result


def indexed_offsets(x,dim,index):
    dim=operator.index(dim)
    if not -x.ndim<=dim<x.ndim:raise IndexError('Indexed axis out of range')
    dim%=x.ndim
    npu._validate_read(x);npu._validate_read(index)
    if index.dtype!=torch.int64 or index.ndim!=x.ndim:raise ValueError('Gather/scatter needs same-rank INT64 indices')
    if any(index.shape[a]>x.shape[a] for a in range(x.ndim) if a!=dim):raise ValueError('Indexed dimensions exceed source')
    if boolean_reduce(compare(index,0,kind='lt'),kind='any').item():raise IndexError('Negative gather/scatter index')
    ids=torch.zeros(index.shape,dtype=torch.int64,device=x.device)
    for axis,size in enumerate(index.shape):
        if axis==dim:coord=normalize(index,x.shape[axis])
        else:
            shape=[1]*x.ndim;shape[axis]=size;coord=torch.arange(size,dtype=torch.int64).to(x.device).view(shape)
        ids=integer_binary(ids,integer_binary(coord,prod(x.shape[axis+1:]),'mul'),'add')
    return ids


def gather(x,dim,index,*,sparse_grad=False,out=None):
    from torch_npu_autograd import _active
    if sparse_grad:raise NotImplementedError('Sparse gather gradients are not supported')
    npu._check_out(out,tuple(index.shape),x.dtype,[x,index])
    if out is not None and _active(x):raise ValueError('Gather out= does not support autograd')
    with torch.no_grad():ids=indexed_offsets(x,dim,index)
    if _active(x):
        if out is not None:raise ValueError('Gather out= does not support autograd')
        return AdvancedIndex.apply(x,ids,tuple(index.shape),True,(index,))
    result=dynamic(x,ids).view(index.shape)
    npu._check_out(out,tuple(index.shape),x.dtype,[x,index])
    return copy(out,result) if out is not None else result


def index_select(x,dim,indices,*,out=None):
    from torch_npu_autograd import _active
    if indices.dtype not in (torch.int64,torch.int32) or indices.ndim>1:raise ValueError('index_select needs scalar/vector integer indices')
    dim=operator.index(dim)
    if not -x.ndim<=dim<x.ndim:raise IndexError('index_select axis out of range')
    dim%=x.ndim
    output=list(x.shape);output[dim]=indices.numel()
    npu._check_out(out,tuple(output),x.dtype,[x,indices])
    if out is not None and _active(x):raise ValueError('index_select out= does not support autograd')
    with torch.no_grad():
        if indices.device.type=='cpu':indices=indices.to(x.device)
        if boolean_reduce(compare(indices,0,kind='lt'),kind='any').item():raise IndexError('Negative index_select index')
        args=[None]*dim+[indices.reshape(-1)];ids,shape=index_geometry(x,args)
    if _active(x):
        if out is not None:raise ValueError('index_select out= does not support autograd')
        return AdvancedIndex.apply(x,ids,shape,True,(indices,))
    result=dynamic(x,ids).view(shape);npu._check_out(out,shape,x.dtype,[x,indices])
    return copy(out,result) if out is not None else result


def scatter_add(x,dim,index,src,*,out=None,inplace=False):
    from torch_npu_autograd import _active
    if src.dtype!=x.dtype or src.device!=x.device or src.ndim!=index.ndim or any(src.shape[a]<index.shape[a] for a in range(src.ndim)):raise ValueError('scatter_add source shape/dtype mismatch')
    npu._check_out(out,tuple(x.shape),x.dtype,[x,index,src])
    if out is not None and _active(x,src):raise ValueError('scatter_add out= does not support autograd')
    with torch.no_grad():ids=indexed_offsets(x,dim,index)
    values=src[tuple(slice(0,n) for n in index.shape)].clone()
    if _active(x,values):
        if inplace or out is not None:raise NotImplementedError('Use out-of-place scatter_add for autograd')
        return IndexedWrite.apply(x,ids,values,True,(index,),True)
    if x.dtype in INTEGERS:result=dynamic(x,ids,values,accumulate=True).view(x.shape)
    elif x.dtype in (torch.float16,torch.float32):
        # CPU scatter_add accumulates Half in FP32, then rounds once per output.
        result=cast(npu._selection_backward(cast(npu._materialize(values),torch.float32).view(-1),ids.view(-1),tuple(x.shape),ids.numel(),x.numel(),initial_value=cast(npu._materialize(x),torch.float32)),x.dtype)
    else:raise NotImplementedError('BF16 scatter_add is not supported')
    npu._check_out(out,tuple(x.shape),x.dtype,[x,index,src])
    return copy(x,result) if inplace else copy(out,result) if out is not None else result


MATH=('sin','cos','pow','tan','floor','ceil','trunc','round','expm1','log1p')


def math_forward(x,kind,y=None,*,out=None):
    npu._validate_read(x)
    dtype=x.dtype if y is None else torch.result_type(x,y)
    if dtype in INTEGERS and kind=='pow':return integer_binary(x,y,'pow',out=out)
    if dtype in INTEGERS:dtype=torch.get_default_dtype()
    if dtype not in (torch.float16,torch.float32,torch.bfloat16):raise ValueError('NPU math requires real floating dtype')
    shape=tuple(x.shape) if y is None else tuple(torch.broadcast_shapes(x.shape,y.shape if isinstance(y,torch.Tensor) else ()))
    npu._check_out(out,shape,dtype,[x,y])
    if kind=='pow' and type(y) in (int,float) and y in (1,2) and x.dtype in (torch.float16,torch.float32):
        # Reuse tiled copy/multiply; 4096-element pow packets exhaust the graph cache on large squares.
        value=npu._materialize(x)
        result=npu._clone(value) if y==1 else npu._execute('mul',value,value)
        return copy(out,result) if out is not None else result
    a=_operand(x,torch.float32,shape);b=a if y is None else _operand(y,torch.float32,shape);pieces=[]
    if kind=='pow' and dtype==torch.float32 and type(y) in (int,float) and y in (.5,-.5,-1.):
        result=npu._reciprocal(a) if y==-1. else npu._math_unary(a,'sqrt' if y==.5 else 'rsqrt')
        return cast(result,dtype,out)
    for start in range(0,prod(shape),4096):
        count=min(4096,prod(shape)-start)
        raw=packet('math-'+kind,(count,MATH.index(kind)),[npu._slice(v,start,count) for v in (a,b)],count*4)
        pieces.append(npu._slice(raw.view(-1),0,count))
    result=cast(npu._concat(pieces,shape,torch.float32),dtype)
    npu._check_out(out,shape,dtype,[x,y])
    return copy(out,result) if out is not None else result


def unary_math(x,kind,*,out=None):
    if x.dtype==torch.bfloat16:return cast(_old['math_unary'](cast(npu._materialize(x),torch.float32),kind),x.dtype,out)
    return _old['math_unary'](x,kind,out=out)


def predicate(x,*,kind,out=None):
    with torch.no_grad():
        if x.dtype==torch.bfloat16:return _old['float_predicate'](cast(npu._materialize(x),torch.float32),kind=kind,out=out)
        if x.dtype in INTEGERS:
            npu._validate_read(x);npu._check_out(out,tuple(x.shape),torch.bool,[x])
            result=torch.full(tuple(x.shape),kind=='isfinite',dtype=torch.bool,device=x.device)
            return copy(out,result) if out is not None else result
        return _old['float_predicate'](x,kind=kind,out=out)


class Math(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,kind):
        result=math_forward(x,kind,y);ctx.kind=kind;ctx.scalar=y if not isinstance(y,torch.Tensor) else None
        ctx.shape,ctx.dtype=tuple(x.shape),x.dtype
        if kind in ('floor','ceil','trunc','round'):
            ctx.tensor=False
            return result.detach()
        ctx.tensor=isinstance(y,torch.Tensor);ctx.save_for_backward(x,y if ctx.tensor else None,result)
        return result.detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        from torch_npu_autograd import _sum_to
        if ctx.kind in ('floor','ceil','trunc','round'):
            return torch.zeros(ctx.shape,dtype=ctx.dtype,device=g.device),None,None
        x,y,value=ctx.saved_tensors;y=y if ctx.tensor else ctx.scalar
        with npu._lock:
            a=cast(npu._materialize(x),torch.float32);gradient=cast(npu._materialize(g),torch.float32);kind=ctx.kind
            gy=None
            if kind=='sin':derivative=math_forward(a,'cos')
            elif kind=='cos':derivative=npu._execute('neg',math_forward(a,'sin'))
            elif kind=='tan':derivative=binary(1.,binary(cast(value,torch.float32),cast(value,torch.float32),kind='mul'),kind='add')
            elif kind=='expm1':derivative=cast(binary(value,1.,kind='add'),torch.float32)
            elif kind=='log1p':derivative=binary(1.,binary(a,1.,kind='add'),kind='div')
            elif kind=='pow':
                b=cast(npu._materialize(y),torch.float32) if ctx.tensor else y
                derivative=binary(b,math_forward(a,'pow',binary(b,1.,kind='sub') if ctx.tensor else b-1),kind='mul')
                derivative=npu._where(npu._compare(b,0.,kind='eq'),0.,derivative) if ctx.tensor else torch.zeros_like(a) if b==0 else derivative
                if ctx.tensor and ctx.needs_input_grad[1]:
                    log=npu._math_unary(a,'log')
                    term=binary(cast(value,torch.float32),log,kind='mul')
                    term=npu._where(npu._logical(npu._compare(a,0.,kind='eq'),npu._compare(b,0.,kind='ge'),kind='and'),0.,term)
                    gy=_sum_to(binary(gradient,term,kind='mul'),y)
            else:derivative=torch.zeros_like(a)
            gx=_sum_to(binary(gradient,derivative,kind='mul'),x) if ctx.needs_input_grad[0] else None
            return gx,gy,None


def math(x,kind,y=None,*,out=None):
    from torch_npu_autograd import _active
    if out is not None and _active(x,y):raise ValueError('Math out= does not support autograd')
    return Math.apply(x,y,kind) if _active(x,y) else math_forward(x,kind,y,out=out)


class ScalarPower(torch.autograd.Function):
    @staticmethod
    def forward(ctx,base,exponent):
        dtype=torch.result_type(base,exponent)
        result=math_forward(npu._scalar(base,dtype),'pow',exponent)
        ctx.base=base;ctx.logbase=log(base) if base>0 else float('nan');ctx.save_for_backward(exponent,result)
        return result.detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        exponent,result=ctx.saved_tensors
        with npu._lock:
            if ctx.base==0:
                factor=npu._where(npu._compare(exponent,0.,kind='ge'),0.,float('-inf'))
            else:
                # Scalar-base forward rounds to the output dtype; its derivative
                # uses the original scalar's logarithm in Torch's opmath dtype.
                # Fold this Python Scalar constant before dtype conversion;
                # no Tensor values are inspected or computed on the CPU.
                factor=npu._binary_op(cast(npu._materialize(result),torch.float32),ctx.logbase,kind='mul')
            gradient=npu._binary_op(cast(npu._materialize(g),torch.float32),factor,kind='mul')
            return None,cast(gradient,exponent.dtype)


def scalar_power(base,exponent,*,out=None):
    from torch_npu_autograd import _active
    dtype=torch.result_type(base,exponent);npu._check_out(out,tuple(exponent.shape),dtype,[exponent])
    if out is not None and _active(exponent):raise ValueError('Scalar power out= does not support autograd')
    result=ScalarPower.apply(base,exponent) if _active(exponent) else math_forward(npu._scalar(base,dtype),'pow',exponent)
    return copy(out,result) if out is not None else result


def boolean_reduce(x,dim=None,keepdim=False,*,out=None,kind):
    with torch.no_grad():
        if x.dtype!=torch.bool:x=typed_cast(npu._materialize(x),torch.bool)
        return _original_boolean_reduce(x,dim,keepdim,out=out,kind=kind)


_original_boolean_reduce=npu._boolean_reduce


def register(aten,autograd):
    for name,fn in dict(cast=cast,copy=copy,fill=fill,binary_op=binary,execute=execute,compare=compare,logical=logical,reduce=reduce,
                       mask_cast=typed_cast,where=where,math_unary=unary_math,float_predicate=predicate).items():setattr(npu,'_'+name,fn)
    def copy_kernel(dst,src,non_blocking=False):
        if torch.is_grad_enabled() and src.requires_grad and src.device.type!=dst.device.type:
            raise ValueError('Differentiable CPU/NPU copy chains are not supported')
        with torch.no_grad():return copy(dst,src,non_blocking)
    def fill_kernel(x,value):
        with torch.no_grad():return fill(x,value)
    aten.impl('copy_',copy_kernel);aten.impl('fill_.Scalar',fill_kernel)
    for kind in ('and','or','xor','not'):
        def bitwise(x,y=None,*,out=None,kind=kind):
            if x.dtype==torch.bool and (y is None or isinstance(y,torch.Tensor) and y.dtype==torch.bool):return _old['logical'](x,y,kind=kind,out=out)
            return integer_binary(x,x if y is None else y,kind,out=out)
        for overload in (('','.out') if kind=='not' else ('.Tensor','.Scalar','.Tensor_out','.Scalar_out')):
            if overload[1:] not in getattr(torch.ops.aten,'bitwise_'+kind)._schemas:continue
            aten.impl('bitwise_'+kind+overload,bitwise)
    def update(x,y,*,alpha=1,kind):
        if torch.is_grad_enabled() and x.requires_grad:raise NotImplementedError('Arithmetic inplace autograd is not supported')
        if isinstance(y,torch.Tensor) and y.device.type=='intel_npu' and y.untyped_storage()._cdata==x.untyped_storage()._cdata and (y.shape!=x.shape or y.stride()!=x.stride() or y.storage_offset()!=x.storage_offset()):
            raise ValueError('Inplace source overlaps destination with a different layout')
        result=binary(x,y,kind=kind,alpha=alpha)
        if not torch.can_cast(result.dtype,x.dtype):raise RuntimeError('Inplace result cannot cast to destination dtype')
        return copy(x,cast(result,x.dtype))
    for kind in ('add','sub','mul','div'):
        for overload in ('.Tensor','.Scalar'):
            aten.impl(kind+'_'+overload,lambda x,y,alpha=1,kind=kind:update(x,y,alpha=alpha,kind=kind))
    npu._boolean_reduce=boolean_reduce
    for name in ('all','any'):
        def reduction(x,dim=None,keepdim=False,*,out=None,kind=name):return boolean_reduce(x,dim,keepdim,out=out,kind=kind)
        for overload in ('','.dim','.dims','.out','.dims_out','.all_out'):aten.impl(name+overload,reduction)
    for name in MATH:
        if name=='pow':
            for library in (aten,autograd):
                library.impl('pow.Tensor_Scalar',lambda x,y:math(x,'pow',y))
                library.impl('pow.Tensor_Tensor',lambda x,y:math(x,'pow',y))
                library.impl('pow.Scalar',scalar_power)
            aten.impl('pow.Tensor_Scalar_out',lambda x,y,*,out:math(x,'pow',y,out=out))
            aten.impl('pow.Tensor_Tensor_out',lambda x,y,*,out:math(x,'pow',y,out=out))
            aten.impl('pow.Scalar_out',scalar_power)
        else:
            for library in (aten,autograd):library.impl(name,lambda x,kind=name:math(x,kind))
            aten.impl(name+'.out',lambda x,*,out,kind=name:math(x,kind,out=out))
            def inplace(x,kind=name):
                if torch.is_grad_enabled() and x.requires_grad:raise NotImplementedError('Differentiable math inplace is not supported')
                return copy(x,math_forward(x,kind))
            aten.impl(name+'_',inplace)
    for library in (aten,autograd):
        def divide(x,y,*,rounding_mode=None,out=None):
            if rounding_mode is None:return torch.div(x,y) if out is None else binary(x,y,kind='div',out=out)
            if rounding_mode not in ('floor','trunc'):raise ValueError('Unknown rounding mode')
            return integer_binary(x,y,'div',rounding_mode=rounding_mode,out=out)
        for name in ('div.Tensor_mode','div.Scalar_mode','div.out_mode','div.Scalar_mode_out'):library.impl(name,divide)
        library.impl('gather',gather);library.impl('gather.out',gather)
        library.impl('index_select',index_select);library.impl('index_select.out',index_select)
        library.impl('scatter_add',scatter_add);library.impl('scatter_add.out',scatter_add)
        def nonzero_kernel(x,*,out=None):
            with torch.no_grad():return nonzero(x,out=out)
        library.impl('nonzero',nonzero_kernel);library.impl('nonzero.out',nonzero_kernel)
        library.impl('nonzero_numpy',lambda x:tuple(npu._materialize(nonzero(x)[:,a]) for a in range(x.ndim)) if x.ndim else (nonzero(x.reshape(1))[:,0],))
        library.impl('where',lambda condition:tuple(npu._materialize(nonzero(condition)[:,a]) for a in range(condition.ndim)))
        library.impl('index.Tensor',index);library.impl('index.Tensor_out',index)
        library.impl('masked_select',lambda x,mask:index(npu._materialize(x.expand(torch.broadcast_shapes(x.shape,mask.shape))),[mask.expand(torch.broadcast_shapes(x.shape,mask.shape))]))
        library.impl('index_put',index_put)
        if library is aten:
            def put_inplace(x,indices,values,accumulate=False):
                with torch.no_grad():return index_put(x,indices,values,accumulate,inplace=True)
            library.impl('index_put_',put_inplace)
        for name in ('scatter.src','scatter.value'):
            library.impl(name,scatter)
        if library is aten:
            def scatter_inplace(x,dim,index,src):
                with torch.no_grad():return scatter(x,dim,index,src,inplace=True)
            for name in ('scatter_.src','scatter_.value'):library.impl(name,scatter_inplace)
            def add_inplace(x,dim,index,src):
                with torch.no_grad():return scatter_add(x,dim,index,src,inplace=True)
            library.impl('scatter_add_',add_inplace)
        for name in ('remainder','floor_divide'):
            for overload in ('.Tensor','.Scalar','' if name=='floor_divide' else '.Scalar_Tensor'):
                if (overload[1:] if overload else '') not in getattr(torch.ops.aten,name)._schemas:continue
                library.impl(name+overload,lambda x,y,kind=name:integer_binary(x,y,kind))
            for overload in ('.out','.Scalar_out') if name=='floor_divide' else ('.Tensor_out','.Scalar_out','.Scalar_Tensor_out'):
                library.impl(name+overload,lambda x,y,*,out,kind=name:integer_binary(x,y,kind,out=out))
    import torch_npu_sequence
    torch_npu_sequence.register(aten)


def _locked(function):
    @wraps(function)
    def call(*args,**kwargs):
        with npu._lock:return function(*args,**kwargs)
    return call


# Sessions and scalar buffers share the backend's existing synchronous lock.
for _name in ('packet','typed_cast','cast','integer_binary','binary','execute','compare','logical','where',
              'reduce','fill','dynamic','copy','normalize','nonzero','index_geometry','index','index_put',
              'scatter','indexed_offsets','gather','index_select','scatter_add','math_forward','math',
              'unary_math','predicate','boolean_reduce','scalar_power'):
    globals()[_name]=_locked(globals()[_name])

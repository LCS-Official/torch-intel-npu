"""Sequence metadata factories and device operators used by model migration."""
import torch
import torch_intel_npu as npu
from math import prod
import operator

INTEGERS=(torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool)


def arange_spec(start,end,step,dtype,max_bytes):
    meta=torch.arange(start,end,step,dtype=dtype,device='meta')
    if meta.dtype not in (*INTEGERS,torch.float16,torch.float32,torch.bfloat16) or meta.dtype==torch.bool or meta.numel()*meta.element_size()>max_bytes:
        raise ValueError('NPU arange exceeds its dtype or storage bound')
    return meta


def arange(start,end=None,step=1,*,dtype=None,layout=None,device=None,pin_memory=None,out=None):
    with npu._lock,torch.no_grad():
        if end is None:start,end=0,start
        target=torch.device(out.device if out is not None else 'intel_npu' if device is None else device)
        if target.type!='intel_npu' or target.index not in (None,0) or layout not in (None,torch.strided) or pin_memory:
            raise ValueError('NPU arange requires strided intel_npu:0 output without pinned memory')
        dtype=out.dtype if out is not None else dtype
        meta=arange_spec(start,end,step,dtype,npu.MAX_ALLOCATION_BYTES)
        npu._check_out(out,tuple(meta.shape),meta.dtype,())
        # Position/index data is determined entirely by scalar metadata, as in the existing out factory.
        host=torch.arange(start,end,step,dtype=meta.dtype,device='cpu')
        if host.shape!=meta.shape:raise ValueError('NPU arange metadata and host range sizes differ')
        return npu._copy(out,host) if out is not None else host.to(target)


def cumsum(x,dim,*,dtype=None,out=None):
    with npu._lock,torch.no_grad():
        npu._validate_read(x)
        if x.dtype not in INTEGERS:raise ValueError('NPU cumsum currently supports integer/BOOL inputs')
        dtype=out.dtype if out is not None and dtype is None else dtype or torch.int64
        if dtype not in INTEGERS or dtype==torch.bool:raise ValueError('NPU integer cumsum requires an integer output dtype')
        if type(dim) is not int or not -(x.ndim or 1)<=dim<(x.ndim or 1):raise IndexError('NPU cumsum axis out of range')
        axis=dim%(x.ndim or 1);shape=tuple(x.shape)
        npu._check_out(out,shape,dtype,(x,))
        values=npu._cast(npu._materialize(x),dtype)
        if not x.ndim:result=values
        elif not x.numel():result=values
        else:
            values=npu._materialize(values.transpose(axis,-1));width=values.shape[-1]
            matrix=values.view(-1,width);stride=1
            def columns(start,count):
                # Each row is contiguous; reuse byte-exact range DMA instead of a Gather graph.
                return npu._concat([npu._storage_slice(matrix,row*width+start,count)
                                    for row in range(matrix.shape[0])],(matrix.shape[0],count),dtype)
            while stride<width:
                prefix=columns(0,stride)
                tail=npu._binary_op(columns(stride,width-stride),columns(0,width-stride),kind='add')
                matrix=npu._cat((prefix,tail),1);stride*=2
            result=npu._materialize(matrix.view(values.shape).transpose(axis,-1))
        return npu._copy(out,result) if out is not None else npu._clone(result)


def isin(elements,test_elements,*,assume_unique=False,invert=False,out=None):
    with npu._lock,torch.no_grad():
        tensors=[x for x in (elements,test_elements) if isinstance(x,torch.Tensor)]
        for x in tensors:npu._validate_read(x)
        shape=tuple(elements.shape) if isinstance(elements,torch.Tensor) else ()
        dtype=torch.result_type(elements,test_elements)
        if dtype not in npu._STORAGE_DTYPES or dtype==torch.bool:
            raise ValueError('NPU isin requires supported real numeric inputs')
        if dtype not in INTEGERS and any(x.dtype in INTEGERS for x in tensors):
            raise ValueError('NPU isin does not mix integer tensors with floating operands')
        count=prod(shape);tests=test_elements.numel() if isinstance(test_elements,torch.Tensor) else 1
        if count*tests>16777216:raise ValueError('NPU isin exceeds its 16777216 pair comparison budget')
        npu._check_out(out,shape,torch.bool,(elements,test_elements))
        def values(x):return npu._cast(npu._materialize(x),dtype).view(-1) if isinstance(x,torch.Tensor) else npu._scalar(x,dtype).view(1)
        a,b=values(elements),values(test_elements);pieces=[]
        for start in range(0,count,1024):
            rows=min(1024,count-start);found=None
            first=npu._storage_slice(a,start,rows).view(rows,1)
            for offset in range(0,tests,max(1,16384//rows)):
                second=npu._storage_slice(b,offset,min(16384//rows,tests-offset)).view(1,-1)
                hit=npu._boolean_reduce(npu._compare(first,second,kind='eq'),1,kind='any')
                found=hit if found is None else npu._logical(found,hit,kind='or')
            if found is None:found=torch.zeros((rows,),dtype=torch.bool,device=a.device)
            pieces.append(npu._logical(found,kind='not') if invert else found)
        result=npu._concat(pieces,shape,torch.bool) if count else torch.empty(shape,dtype=torch.bool,device=a.device)
        return npu._copy(out,result) if out is not None else result


def integer_pad(x,padding,value=0):
    with npu._lock,torch.no_grad():
        npu._validate_read(x)
        padding=tuple(operator.index(v) for v in padding)
        if len(padding)%2 or len(padding)>2*x.ndim:raise ValueError('Invalid integer padding rank')
        shape=list(x.shape);source=[slice(None)]*x.ndim;destination=list(source)
        for i in range(len(padding)//2):
            axis=x.ndim-1-i;left,right=padding[2*i:2*i+2];size=x.shape[axis]
            length=size-max(-left,0)-max(-right,0)
            shape[axis]=size+left+right
            if length<0 or shape[axis]<0:raise ValueError('NPU integer padding crops beyond its input')
            source[axis]=slice(max(-left,0),max(-left,0)+length)
            destination[axis]=slice(max(left,0),max(left,0)+length)
        if prod(shape)>npu._storage_limit(x.dtype):raise ValueError('NPU padded output exceeds dtype storage')
        result=torch.full(shape,value,dtype=x.dtype,device=x.device)
        values=x[tuple(source)]
        if values.numel():npu._copy(result[tuple(destination)],npu._materialize(values))
        return result


def register(aten):
    for name in ('arange','arange.start','arange.start_step','arange.start_out'):
        aten.impl(name,arange)
    aten.impl('arange.out',lambda end,*,out:arange(0,end,out=out))
    aten.impl('cumsum',cumsum);aten.impl('cumsum.out',cumsum)
    for suffix in ('Tensor_Tensor','Tensor_Scalar','Scalar_Tensor'):
        aten.impl('isin.'+suffix,isin);aten.impl('isin.'+suffix+'_out',isin)

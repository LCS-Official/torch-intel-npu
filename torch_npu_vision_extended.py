"""3D and geometry-based vision operators on device data, without CPU fallback."""
import operator
from math import prod
import numpy as np
import torch
from torch.autograd.function import once_differentiable
import torch_intel_npu as npu
from torch_npu_geometry import convolution_windows_nd,padding_axis_geometry,transpose_resample_geometry


def _last(x):
    return npu._materialize(x.permute((0,)+tuple(range(2,x.ndim))+(1,)))


def _first(x):
    return npu._materialize(x.permute((0,x.ndim-1)+tuple(range(1,x.ndim-1))))


def _add_tree(levels,value):
    level=0
    while level<len(levels) and levels[level] is not None:
        value=npu._execute('add',levels[level],value);levels[level]=None;level+=1
    if level==len(levels):levels.append(value)
    else:levels[level]=value


def _tree_total(levels):
    result=None
    for value in reversed(levels):
        if value is not None:result=value if result is None else npu._execute('add',result,value)
    return result


def _windows(source,original,target,params,transpose):
    width=prod(params[0])*original[1]
    if width>65536:raise ValueError('NPU convolution window exceeds 65536 elements')
    rows=target[0]*prod(target[2:]);batch=max(1,65536//width)
    for start in range(0,rows,batch):
        count=min(batch,rows-start)
        indices=convolution_windows_nd(original,target,*params,start,count,transpose=transpose)
        yield start,count,npu._gather_window(source,indices,1)


def _conv_spec(x,w,b,s,p,d,t,o,g):
    npu._validate_float_read(x);npu._validate_float_read(w)
    if x.ndim!=5 or w.ndim!=5 or x.dtype!=w.dtype or type(g) is not int or g<1:
        raise ValueError('NPU Conv3d requires matching NCDHW FP16/FP32 tensors and positive groups')
    for values,minimum in ((s,1),(p,0),(d,1),(o,0)):
        if len(values)!=3 or any(type(v) is not int or v<minimum for v in values):raise ValueError('Invalid NPU Conv3d geometry')
    if (t and (x.shape[1]!=w.shape[0] or x.shape[1]%g or any(a>=b and a>=c for a,b,c in zip(o,s,d))) or (
            not t and (x.shape[1]!=w.shape[1]*g or w.shape[0]%g or any(o)))):
        raise ValueError('NPU Conv3d channels/output_padding mismatch')
    cout=w.shape[1]*g if t else w.shape[0]
    spatial=tuple((n-1)*a-2*b+c*(k-1)+e+1 if t else (n+2*b-c*(k-1)-1)//a+1
                  for n,k,a,b,c,e in zip(x.shape[2:],w.shape[2:],s,p,d,o))
    shape=(x.shape[0],cout,*spatial)
    if any(v<1 for v in spatial) or prod(shape)>npu.MAX_TENSOR_ELEMENTS or prod(w.shape[2:])*max(x.shape[1]//g,cout//g)>65536:
        raise ValueError('NPU Conv3d output/window exceeds storage bounds')
    if b is not None:
        npu._validate_float_read(b)
        if b.shape!=(cout,) or b.dtype!=x.dtype:raise ValueError('NPU Conv3d bias mismatch')
    return shape


def convolution3d(x,w,b,s,p,d,t,o,g):
    from torch_npu_autograd import _active
    return Conv3d.apply(x,w,b,s,p,d,t,o,g) if _active(x,w,b) else conv_forward(x,w,b,s,p,d,t,o,g)


def conv_forward(x,w,b,s,p,d,t,o,g):
    with npu._lock:
        shape=_conv_spec(x,w,b,s,p,d,t,o,g);cin=x.shape[1]//g;cout=shape[1]//g;pieces=[]
        params=tuple(w.shape[2:]),tuple(s),tuple(p),tuple(d)
        for group in range(g):
            a=npu._cast(npu._materialize(x.narrow(1,group*cin,cin)),torch.float32)
            weight=npu._cast(npu._materialize(w.narrow(0,group*(cin if t else cout),cin if t else cout)),torch.float32)
            order=(2,3,4,0,1) if t else (2,3,4,1,0)
            matrix=npu._materialize(weight.permute(order)).view(-1,cout);target=(shape[0],cout,*shape[2:])
            parts=[npu._matmul(window,matrix) for _,_,window in _windows(_last(a),tuple(a.shape),target,params,t)]
            result=_first(npu._concat(parts,(shape[0],*shape[2:],cout),torch.float32))
            if b is not None:result=npu._binary_op(result,npu._cast(npu._materialize(b.narrow(0,group*cout,cout)),torch.float32).view(1,cout,1,1,1),kind='add')
            pieces.append(npu._cast(result,x.dtype))
        return pieces[0] if g==1 else npu._cat(pieces,1)


class Conv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b,s,p,d,t,o,g):
        result=conv_forward(x,w,b,s,p,d,t,o,g);ctx.save_for_backward(x,w)
        ctx.params=(tuple(w.shape[2:]),tuple(s),tuple(p),tuple(d));ctx.transpose=t;ctx.groups=g
        ctx.bias_dtype=None if b is None else b.dtype
        return result.detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        with npu._lock:
            x,w=ctx.saved_tensors;cin=x.shape[1]//ctx.groups;cout=gradient.shape[1]//ctx.groups;t=ctx.transpose;xs=[];ws=[]
            for group in range(ctx.groups):
                a=npu._cast(npu._materialize(x.narrow(1,group*cin,cin)),torch.float32)
                weight=npu._cast(npu._materialize(w.narrow(0,group*(cin if t else cout),cin if t else cout)),torch.float32)
                grad=npu._cast(npu._materialize(gradient.narrow(1,group*cout,cout)),torch.float32)
                source=_last(grad);gx=[];levels=[]
                if ctx.needs_input_grad[0]:
                    matrix=npu._materialize(weight.permute((2,3,4,1,0) if t else (2,3,4,0,1))).view(-1,cin)
                    gx=[npu._matmul(window,matrix) for _,_,window in _windows(source,tuple(grad.shape),tuple(a.shape),ctx.params,not t)]
                    xs.append(npu._cast(_first(npu._concat(gx,(a.shape[0],*a.shape[2:],cin),torch.float32)),x.dtype))
                if ctx.needs_input_grad[1]:
                    data=_last(a).view(-1,cin) if t else source.view(-1,cout)
                    original,target=(tuple(grad.shape),tuple(a.shape)) if t else (tuple(a.shape),tuple(grad.shape))
                    for start,count,window in _windows(source if t else _last(a),original,target,ctx.params,False):
                        rows=npu._slice(data,start*(cin if t else cout),count*(cin if t else cout)).view(count,-1)
                        _add_tree(levels,npu._matmul(rows.t(),window))
                    total=_tree_total(levels)
                    # Rows are input/output channels; columns are (kernel, opposite channels).
                    raw=total.view((cin if t else cout),*w.shape[2:],cout if t else cin)
                    ws.append(npu._cast(npu._materialize(raw.permute(0,4,1,2,3)),w.dtype))
            gb=npu._reduce(npu._cast(npu._materialize(gradient),torch.float32),(0,2,3,4),dtype=ctx.bias_dtype,kind='sum') if ctx.needs_input_grad[2] else None
            return (npu._cat(xs,1) if xs else None),(npu._cat(ws,0) if ws else None),gb,None,None,None,None,None,None


def batch_norm(x,w,b,rm,rv,training,momentum,eps,cudnn):
    from torch_npu_autograd import BatchNorm,_active
    if x.ndim not in (2,3,4,5):raise ValueError('NPU BatchNorm requires rank 2-5')
    value=x if x.ndim==4 else x.reshape(x.shape[0],x.shape[1],1,-1)
    result=BatchNorm.apply(value,w,b,rm,rv,training,momentum,eps) if _active(x,w,b) else npu._batch_norm(value,w,b,rm,rv,training,momentum,eps)[0]
    result=result.view(x.shape)
    return result.clone() if _active(result) and x.ndim!=4 else result


class StaticAxes(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,plans,shape,constant):
        from torch_npu_vision import _sample_axis
        ctx.shape=tuple(x.shape);ctx.plans=plans
        result=npu._cast(npu._materialize(x),torch.float32)
        for axis,positions,weights in plans:
            result=_sample_axis(result,axis,positions,weights)
            if np.any(positions<0):
                mask=(positions[:,0]<0);layout=[1]*result.ndim;layout[axis]=len(mask)
                # Selection preserves original values with +/-Inf/NaN padding;
                # multiplying a zero mask by Inf would corrupt the image.
                result=npu._where(torch.from_numpy(mask).to(x.device).view(layout),constant,result)
        return npu._cast(result,x.dtype).view(shape).detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        from torch_npu_vision import _sample_axis
        result=npu._cast(npu._materialize(g),torch.float32)
        for axis,positions,weights in reversed(ctx.plans):
            result=_sample_axis(result,axis,*transpose_resample_geometry(ctx.shape[axis],positions,weights))
        return npu._cast(result,g.dtype),None,None,None


def pad(x,padding,mode='constant',value=None):
    if mode=='constant' and x.dtype in (torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool):
        from torch_npu_sequence import integer_pad
        return integer_pad(x,padding,0 if value is None else value)
    npu._validate_float_read(x) if not x.requires_grad else None
    padding=tuple(operator.index(v) for v in padding)
    if len(padding)%2 or len(padding)>2*x.ndim:raise ValueError('Invalid NPU padding rank')
    shape=list(x.shape);plans=[]
    for i in range(len(padding)//2):
        axis=x.ndim-1-i;left,right=padding[2*i:2*i+2]
        positions,weights=padding_axis_geometry(shape[axis],left,right,mode);shape[axis]=len(positions)
        plans.append((axis,positions,weights))
    if prod(shape)>npu.MAX_TENSOR_ELEMENTS:raise ValueError('NPU padded output exceeds storage bound')
    if not prod(shape):raise ValueError('NPU padding to an empty output is not yet supported')
    if mode!='constant' and value not in (None,0,0.):raise ValueError('Nonconstant padding cannot specify a value')
    # Contract first so an intermediate never exceeds the larger endpoint.
    plans.sort(key=lambda p:len(p[1])/x.shape[p[0]])
    return StaticAxes.apply(x,tuple(plans),tuple(shape),0. if value is None else value)


class Rearrange(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,indices,shape):
        npu._validate_float_read(x)
        ctx.shape=tuple(x.shape);ctx.indices=indices
        return npu._gather_window(npu._materialize(x).view(-1),indices.reshape(1,-1),1).view(shape).detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        inverse=np.argsort(ctx.indices.reshape(-1))
        return npu._gather_window(npu._materialize(g).view(-1),inverse.reshape(1,-1),1).view(ctx.shape),None,None


def pixel(x,factor,*,inverse=False):
    factor=operator.index(factor)
    if x.ndim<3 or factor<1:raise ValueError('NPU pixel shuffle requires rank>=3 and a positive factor')
    prefix=tuple(x.shape[:-3]);c,h,w=x.shape[-3:];indices=np.arange(x.numel(),dtype=np.int64)
    if inverse:
        if h%factor or w%factor:raise ValueError('Pixel unshuffle spatial dimensions must divide the factor')
        shape=prefix+(c*factor*factor,h//factor,w//factor)
        raw=indices.reshape(prod(prefix),c,h//factor,factor,w//factor,factor).transpose(0,1,3,5,2,4)
    else:
        if c%(factor*factor):raise ValueError('Pixel shuffle channels must divide factor squared')
        shape=prefix+(c//(factor*factor),h*factor,w*factor)
        raw=indices.reshape(prod(prefix),c//(factor*factor),factor,factor,h,w).transpose(0,1,4,2,5,3)
    return Rearrange.apply(x,raw.reshape(-1),shape)


def channel_shuffle(x,groups):
    groups=operator.index(groups)
    if x.ndim<3 or groups<1 or x.shape[1]%groups:raise ValueError('Invalid NPU channel shuffle groups')
    raw=np.arange(x.numel(),dtype=np.int64).reshape(x.shape[0],groups,x.shape[1]//groups,-1).transpose(0,2,1,3)
    return Rearrange.apply(x,raw.reshape(-1),tuple(x.shape))


def _tuple(value,dims,default=None):
    if value is None or isinstance(value,(tuple,list)) and not value:return default
    return (operator.index(value),)*dims if isinstance(value,int) else tuple(operator.index(v) for v in value)


def pool_geometry(spatial,kernel,stride,padding,dilation,ceil,*,adaptive=None,include=True,divisor=None):
    dims=len(spatial)
    if adaptive is not None:
        output=_tuple(adaptive,dims)
        if len(output)!=dims or any(v<1 for v in output):raise ValueError('Invalid adaptive pooling output')
        bounds=[[(i*n//m,((i+1)*n+m-1)//m) for i in range(m)] for n,m in zip(spatial,output)]
        width=prod(max(b-a for a,b in axis) for axis in bounds)
    else:
        kernel=_tuple(kernel,dims);stride=_tuple(stride,dims,kernel);padding=_tuple(padding,dims);dilation=_tuple(dilation,dims)
        if any(len(v)!=dims for v in (kernel,stride,padding,dilation)) or any(v<1 for v in (*kernel,*stride,*dilation)) or any(p<0 or p>k//2 for p,k in zip(padding,kernel)):
            raise ValueError('Invalid NPU pooling geometry')
        output=[]
        for n,k,s,p,d in zip(spatial,kernel,stride,padding,dilation):
            m=(n+2*p-d*(k-1)-1+(s-1 if ceil else 0))//s+1
            if ceil and (m-1)*s>=n+p:m-=1
            if m<1:raise ValueError('Pooling output is empty')
            output.append(m)
        output=tuple(output);width=prod(kernel)
    if prod(output)*width>8388608 or width>32765:raise ValueError('NPU pooling exceeds sampled-cell/carrier bounds')
    positions=np.full((prod(output),width),-1,dtype=np.int64);weights=np.zeros_like(positions,dtype=np.float32)
    for row,coord in enumerate(np.ndindex(output)):
        if adaptive is not None:
            samples=[np.arange(*bounds[a][i]) for a,i in enumerate(coord)];padded=prod(map(len,samples))
        else:
            samples=[i*s-p+np.arange(k)*d for i,s,p,k,d in zip(coord,stride,padding,kernel,dilation)]
            padded=prod(np.count_nonzero((v>=-p)&(v<n+p)) for v,n,p in zip(samples,spatial,padding))
        grids=np.meshgrid(*samples,indexing='ij');flat=np.zeros(tuple(map(len,samples)),dtype=np.int64);valid=np.ones_like(flat,dtype=bool)
        for axis,(grid,n) in enumerate(zip(grids,spatial)):
            valid&=(grid>=0)&(grid<n);flat+=grid*prod(spatial[axis+1:])
        divisor_value=divisor if divisor is not None else padded if include else int(valid.sum())
        if not divisor_value:raise ValueError('Pooling window divisor must not be zero')
        positions[row,:flat.size]=np.where(valid,flat,-1).reshape(-1)
        weights[row,:flat.size]=valid.reshape(-1).astype(np.float32)/divisor_value
    return output,positions,weights


class AveragePool(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,output,positions,weights):
        from torch_npu_vision import _sample_axis
        dims=len(output);ctx.shape=tuple(x.shape);ctx.dims=dims;ctx.plans=(positions,weights)
        data=npu._cast(npu._materialize(x),torch.float32).view(-1,prod(x.shape[-dims:]))
        value=_sample_axis(data,1,positions,weights)
        return npu._cast(value,x.dtype).view(*x.shape[:-dims],*output).detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        from torch_npu_vision import _sample_axis
        count=prod(ctx.shape[-ctx.dims:]);positions,weights=ctx.plans
        value=npu._cast(npu._materialize(g),torch.float32).view(-1,len(positions))
        result=_sample_axis(value,1,*transpose_resample_geometry(count,positions,weights))
        return npu._cast(result,g.dtype).view(ctx.shape),None,None,None


class MaximumPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,output,positions):
        ctx.set_materialize_grads(False)
        dims=len(output);plane=prod(x.shape[-dims:]);shape=(*x.shape[:-dims],*output)
        source=npu._cast(npu._materialize(x),torch.float32).view(-1);parts=[];ids=[];width=positions.shape[1];batch=max(1,32765//width)
        for channel in range(x.numel()//plane):
            for start in range(0,len(positions),batch):
                locations=positions[start:start+batch]
                window=npu._gather_window(source,np.where(locations<0,-1,locations+channel*plane),1)
                value,index=npu._select_rows(window,1,indices=torch.from_numpy(locations.copy()).to(x.device),pool=True)
                parts.append(value.view(-1));ids.append(index.view(-1))
        value=npu._cast(npu._concat(parts,shape,torch.float32),x.dtype);indices=npu._cat(ids,0).view(shape)
        ctx.shape=tuple(x.shape);ctx.per=prod(output);ctx.width=plane;ctx.save_for_backward(indices);ctx.mark_non_differentiable(indices)
        return value.detach(),indices

    @staticmethod
    @once_differentiable
    def backward(ctx,g,unused):
        indices,=ctx.saved_tensors
        return npu._selection_backward_planes(g,indices,ctx.shape,ctx.per,ctx.width),None,None


def pooling(x,kernel=None,stride=(),padding=0,dilation=1,ceil=False,include=True,divisor=None,*,dims=3,adaptive=None,maximum=False,out=None,indices=None):
    with torch.no_grad():
        npu._validate_float_read(x)
        if x.ndim not in (dims+1,dims+2):raise ValueError('NPU pooling input rank mismatch')
        if divisor is not None and (type(divisor) is not int or divisor==0):raise ValueError('Pooling divisor_override must be nonzero integer')
        output,positions,weights=pool_geometry(tuple(x.shape[-dims:]),kernel,stride,padding,dilation,ceil,adaptive=adaptive,include=include,divisor=divisor)
        shape=(*x.shape[:-dims],*output)
        if prod(shape)>(npu.MAX_INT64_ELEMENTS if maximum else npu.MAX_TENSOR_ELEMENTS) or prod(shape)*positions.shape[1]>8388608:raise ValueError('NPU pooling exceeds dtype storage or sampled-cell bounds')
        if maximum and np.any(np.all(positions<0,axis=1)):raise ValueError('NPU MaxPool requires a valid sample in every window')
        npu._check_out(out,shape,x.dtype,(x,indices))
        if maximum:npu._check_out(indices,shape,torch.int64,(x,out))
    if out is not None and torch.is_grad_enabled() and x.requires_grad:raise ValueError('Pooling out= does not support autograd')
    if maximum:
        if (out is None)!=(indices is None):raise ValueError('Maximum pooling requires paired out tensors')
        v,i=MaximumPool.apply(x,output,positions)
        return (npu._copy(out,v),npu._copy(indices,i)) if out is not None else (v,i)
    value=AveragePool.apply(x,output,positions,weights)
    return npu._copy(out,value) if out is not None else value


def indexed_read(source,index,per,width):
    """Linear-time dynamic gathers, tiled without reading model indices on the host."""
    npu._validate_float_read(source);npu._validate_read(index)
    if index.dtype!=torch.int64 or per<1 or width<1 or index.numel()%per or source.numel()//width!=index.numel()//per:
        raise ValueError('Invalid NPU indexed read geometry')
    data=npu._cast(npu._materialize(source),torch.float32).view(-1);words=npu._materialize(index).view(-1).view(torch.float32);parts=[]
    for start in range(0,index.numel(),8192):
        count=min(8192,index.numel()-start);initial=npu._broadcast(npu._scalar(0.,torch.float32),(count,))
        ids=npu._slice(words,2*start,2*count)
        # Shape metadata bounds the relevant planes; no device index download.
        lower=(start//per)*width;upper=((start+count-1)//per+1)*width
        for base in range(lower,upper,32768):
            span=min(32768,upper-base);size=max(8+2*count+span,count+1)
            header=torch.tensor([count,start,per,width,base,span,0,source.numel()],dtype=torch.float32).to(source.device)
            packet=npu._concat([header,ids,npu._slice(data,base,span)],(1,size),torch.float32)
            second=npu._concat([initial,npu._broadcast(npu._scalar(0.,torch.float32),(size-count,))],(1,size),torch.float32)
            result=npu._run(('indexed-read-f32','f32',(1,size)),[packet,second],(1,size),torch.float32)
            if npu._slice(result,size-1,1).item():raise IndexError('NPU dynamic read index out of bounds')
            initial=npu._slice(result,0,count)
        parts.append(initial)
    return npu._cast(npu._concat(parts,tuple(index.shape),torch.float32),source.dtype)


class Unpool(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,indices,output_size):
        dims=len(output_size);shape=(*x.shape[:-dims],*output_size)
        npu._validate_float_read(x);npu._validate_read(indices)
        if x.shape!=indices.shape or indices.dtype!=torch.int64 or any(n<1 for n in output_size) or prod(shape)>npu.MAX_TENSOR_ELEMENTS:
            raise ValueError('Invalid NPU unpool shape/indices')
        ctx.save_for_backward(indices);ctx.per=prod(x.shape[-dims:]);ctx.width=prod(output_size)
        return npu._selection_backward(x,indices,shape,ctx.per,ctx.width,accumulate=False).detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        indices,=ctx.saved_tensors
        return indexed_read(g,indices,ctx.per,ctx.width),None,None


def _unfold_geometry(shape,kernel,dilation,padding,stride):
    full=shape if len(shape)==4 else (1,)+shape
    if len(full)!=4:raise ValueError('NPU Unfold requires CHW/NCHW')
    kernel=_tuple(kernel,2);stride=_tuple(stride,2);padding=_tuple(padding,2);dilation=_tuple(dilation,2)
    if any(len(v)!=2 for v in (kernel,stride,padding,dilation)) or any(v<1 for v in (*kernel,*stride,*dilation)) or any(v<0 for v in padding):raise ValueError('Invalid Unfold geometry')
    spatial=tuple((n+2*p-d*(k-1)-1)//s+1 for n,k,s,p,d in zip(full[2:],kernel,stride,padding,dilation))
    output=(full[0],full[1]*prod(kernel),prod(spatial))
    if any(v<1 for v in spatial) or prod(output)>npu.MAX_TENSOR_ELEMENTS or prod(full)>npu.MAX_TENSOR_ELEMENTS:raise ValueError('Unfold exceeds output/storage bounds')
    indices=convolution_windows_nd(full,(full[0],full[1],*spatial),kernel,stride,padding,dilation,0,full[0]*prod(spatial))
    raw=(indices//(prod(full[1:])))*prod(full[1:])+(indices%full[1])*prod(full[2:])+(indices//full[1])%prod(full[2:])
    raw=np.where(indices<0,-1,raw).reshape(full[0],prod(spatial),prod(kernel),full[1]).transpose(0,3,2,1)
    return raw.reshape(-1),output if len(shape)==4 else output[1:]


def _adjoint_gather(value,positions,size):
    from torch_npu_vision import _sample_axis
    inverse,weights=transpose_resample_geometry(size,positions.reshape(-1,1),np.ones((positions.size,1),dtype=np.float32))
    return _sample_axis(npu._cast(npu._materialize(value),torch.float32).view(1,-1),1,inverse,weights).view(-1)


class Unfold(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,kernel,dilation,padding,stride):
        npu._validate_float_read(x);positions,shape=_unfold_geometry(tuple(x.shape),kernel,dilation,padding,stride)
        ctx.shape=tuple(x.shape);ctx.positions=positions
        return npu._gather_window(npu._materialize(x).view(-1),positions.reshape(1,-1),1).view(shape).detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        return npu._cast(_adjoint_gather(g,ctx.positions,prod(ctx.shape)),g.dtype).view(ctx.shape),None,None,None,None


class Fold(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,output_size,kernel,dilation,padding,stride):
        npu._validate_float_read(x);kernel=_tuple(kernel,2);output_size=tuple(output_size)
        if x.ndim not in (2,3) or len(output_size)!=2 or any(v<1 for v in output_size) or x.shape[-2]%prod(kernel):raise ValueError('Invalid Fold input')
        shape=tuple(x.shape[:-2])+(x.shape[-2]//prod(kernel),)+output_size
        positions,expected=_unfold_geometry(shape,kernel,dilation,padding,stride)
        if tuple(x.shape)!=expected:raise ValueError('Fold input does not match output geometry')
        ctx.shape=tuple(x.shape);ctx.positions=positions
        return npu._cast(_adjoint_gather(x,positions,prod(shape)),x.dtype).view(shape).detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        return npu._gather_window(npu._materialize(g).view(-1),ctx.positions.reshape(1,-1),1).view(ctx.shape),None,None,None,None,None


def register(aten,autograd):
    for library in (aten,autograd):
        library.impl('pad',pad)
        library.impl('constant_pad_nd',lambda x,pad,value=0:globals()['pad'](x,pad,'constant',value))
        library.impl('pixel_shuffle',pixel)
        library.impl('pixel_unshuffle',lambda x,factor:pixel(x,factor,inverse=True))
        library.impl('channel_shuffle',channel_shuffle)
        for mode,base in (('reflect','reflection_pad'),('replicate','replication_pad')):
            for dim in (1,2,3):
                library.impl(base+str(dim)+'d',lambda x,padding,mode=mode:pad(x,padding,mode))
        library.impl('avg_pool3d',lambda x,kernel_size,stride=(),padding=0,ceil_mode=False,count_include_pad=True,divisor_override=None:
                     pooling(x,kernel_size,stride,padding,1,ceil_mode,count_include_pad,divisor_override))
        for dims in (1,2,3):
            if dims==1:continue # Torch's public wrapper lowers to adaptive_max_pool2d.
            library.impl('adaptive_max_pool'+str(dims)+'d',lambda x,output_size,dims=dims:pooling(x,dims=dims,adaptive=output_size,maximum=True))
        library.impl('adaptive_avg_pool3d',lambda x,output_size:pooling(x,adaptive=output_size))
        library.impl('_adaptive_avg_pool3d',lambda x,output_size:pooling(x,adaptive=output_size))
        library.impl('max_pool3d_with_indices',lambda x,kernel_size,stride=(),padding=0,dilation=1,ceil_mode=False:
                     pooling(x,kernel_size,stride,padding,dilation,ceil_mode,maximum=True))
        library.impl('max_pool3d',lambda x,kernel_size,stride=(),padding=0,dilation=1,ceil_mode=False:
                     pooling(x,kernel_size,stride,padding,dilation,ceil_mode,maximum=True)[0])
        library.impl('max_unpool2d',lambda x,indices,output_size:Unpool.apply(x,indices,tuple(output_size)))
        library.impl('max_unpool3d',lambda x,indices,output_size,stride,padding:Unpool.apply(x,indices,tuple(output_size)))
        library.impl('im2col',lambda x,kernel_size,dilation,padding,stride:Unfold.apply(x,kernel_size,dilation,padding,stride))
        library.impl('col2im',lambda x,output_size,kernel_size,dilation,padding,stride:Fold.apply(x,output_size,kernel_size,dilation,padding,stride))

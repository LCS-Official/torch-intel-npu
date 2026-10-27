"""Bounded 1D/2D vision operators, with device arithmetic and static geometry."""
import operator
from math import prod,isfinite
import numpy as np
import torch
from torch.autograd.function import once_differentiable
import torch_intel_npu as npu
from torch_npu_geometry import (resample_axis_geometry,transpose_resample_geometry,transpose_conv_windows,
                                validate_resample_backward_geometry,antialias_axis_geometry)


def _resample_plan(size,output,mode,align,scale,dims):
    return (antialias_axis_geometry(size,output,mode[:-3],align,scale) if mode.endswith('-aa') else
            resample_axis_geometry(size,output,mode,align,scale,legacy_nearest=dims!=1))


def _identity_plans(shape,plans):
    return all(p.shape==(n,1) and np.array_equal(p[:,0],np.arange(n)) and np.all(w==1)
               for n,(p,w) in zip(shape,plans))


def _sample_axis(x,axis,positions,weights):
    order=tuple(a for a in range(x.ndim) if a!=axis)+(axis,);inverse=tuple(order.index(a) for a in range(x.ndim))
    source=npu._materialize(x.permute(order)).view(-1);old=x.shape[axis];new,taps=positions.shape
    shape=tuple(x.shape[a] for a in order[:-1])+(new,);parts=[];batch=max(1,8192//min(taps,8192))
    for start in range(0,prod(shape),batch):
        count=min(batch,prod(shape)-start);linear=np.arange(start,start+count,dtype=np.int64)
        rows,columns=linear//new,linear%new;total=None
        for offset in range(0,taps,8192):
            pos=positions[columns,offset:offset+8192];coeff=weights[columns,offset:offset+8192]
            window=npu._gather_window(source,np.where(pos<0,-1,rows[:,None]*old+pos),1)
            if taps==1 and np.all(coeff==1):result=window.view(-1)
            else:
                constants=torch.from_numpy(np.ascontiguousarray(coeff)).to('intel_npu')
                result=npu._reduce(npu._execute('mul',window,constants),1,kind='sum')
            total=result if total is None else npu._execute('add',total,result)
        parts.append(total)
    result=npu._concat(parts,shape,torch.float32)
    return npu._materialize(result.permute(inverse))


def _upsample_spec(x,output_size,align_corners,scales,mode,out=None):
    npu._validate_float_read(x);dims=len(output_size)
    if dims not in (1,2,3) or x.ndim!=dims+2:raise ValueError('NPU interpolation requires NCL/NCHW/NCDHW')
    output=tuple(operator.index(v) for v in output_size)
    if any(v<1 for v in output) or prod(x.shape[:2])*prod(output)>npu.MAX_TENSOR_ELEMENTS:
        raise ValueError('NPU interpolation output must be nonempty and <=1048576 elements')
    plans=[_resample_plan(x.shape[a+2],output[a],mode,align_corners,scales[a],dims) for a in range(dims)]
    # Contracting axes first keep every separable temporary within storage bounds.
    order=tuple(sorted(range(dims),key=lambda a:output[a]/x.shape[a+2]))
    npu._check_out(out,tuple(x.shape[:2])+output,x.dtype,(x,))
    return output,plans,order


def upsample(x,output_size,align_corners=False,scales=None,*,mode,out=None):
    with npu._lock:
        scales=(None,)*len(output_size) if scales is None else tuple(scales)
        output,plans,order=_upsample_spec(x,output_size,align_corners,scales,mode,out)
        if _identity_plans(tuple(x.shape[2:]),plans):
            result=npu._clone(x)
            return npu._copy(out,result) if out is not None else result
        result=npu._cast(npu._materialize(x),torch.float32)
        for axis in order:result=_sample_axis(result,axis+2,*plans[axis])
        result=npu._cast(result,x.dtype)
        return npu._copy(out,result) if out is not None else result


def upsample_backward(gradient,output_size,input_size,align_corners=False,scales=None,*,mode,grad_input=None):
    with npu._lock:
        npu._validate_float_read(gradient);shape=tuple(input_size);dims=len(output_size)
        if len(shape)!=dims+2 or tuple(gradient.shape)!=shape[:2]+tuple(output_size):
            raise ValueError('NPU interpolation backward geometry mismatch')
        if prod(shape)>npu.MAX_TENSOR_ELEMENTS or any(v<1 for v in shape):raise ValueError('NPU interpolation input exceeds storage bounds')
        scales=(None,)*dims if scales is None else tuple(scales)
        validate_resample_backward_geometry(shape[2:],output_size,mode,align_corners,scales)
        plans=[_resample_plan(shape[a+2],output_size[a],mode,align_corners,scales[a],dims) for a in range(dims)]
        npu._check_out(grad_input,shape,gradient.dtype,(gradient,))
        order=tuple(sorted(range(dims),key=lambda a:output_size[a]/shape[a+2]));result=npu._cast(npu._materialize(gradient),torch.float32)
        if not _identity_plans(shape[2:],plans):
            for axis in reversed(order):result=_sample_axis(result,axis+2,*transpose_resample_geometry(shape[axis+2],*plans[axis]))
        result=npu._cast(result,gradient.dtype)
        return npu._copy(grad_input,result) if grad_input is not None else result


class Upsample(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,output,align,scales,mode):
        validate_resample_backward_geometry(tuple(x.shape[2:]),output,mode,align,scales)
        ctx.shape=tuple(x.shape);ctx.params=(tuple(output),align,tuple(scales));ctx.mode=mode
        return upsample(x,output,align,scales,mode=mode).detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        output,align,scales=ctx.params
        return upsample_backward(gradient,output,ctx.shape,align,scales,mode=ctx.mode),None,None,None,None


def _norm_spec(x,weight,bias,groups,eps):
    npu._validate_float_read(x);groups=operator.index(groups)
    if x.ndim not in (2,3,4,5) or groups<1 or x.shape[1]%groups:raise ValueError('NPU GroupNorm requires rank2-5, positive groups dividing channels')
    if not isfinite(eps) or not 2**-149<=eps<=torch.finfo(torch.float32).max:raise ValueError('NPU normalization requires positive finite FP32 eps')
    for value in (weight,bias):
        if value is not None:
            npu._validate_float_read(value)
            if value.shape!=(x.shape[1],) or value.dtype not in (x.dtype,torch.float32):raise ValueError('NPU norm affine shape/dtype mismatch')
    affine=[v.dtype for v in (weight,bias) if v is not None]
    if len(set(affine))>1:raise ValueError('NPU norm affine tensors must share a dtype')
    if x.dtype==torch.float16 and weight is None and bias is not None and bias.dtype==torch.float32:
        raise ValueError('NPU mixed norm requires an FP32 weight')
    return groups,affine[0] if affine else x.dtype


def native_group_norm(x,weight,bias,N,C,HxW,group,eps):
    with npu._lock:
        groups,stats_dtype=_norm_spec(x,weight,bias,group,eps)
        if (N,C,HxW)!=(x.shape[0],x.shape[1],prod(x.shape[2:])):raise ValueError('NPU GroupNorm metadata does not match input')
        grouped=npu._cast(npu._materialize(x),torch.float32).view(N,groups,-1)
        mean=npu._reduce(grouped,2,True,kind='mean');centered=npu._binary_op(grouped,mean,kind='sub')
        variance=npu._reduce(npu._execute('mul',centered,centered),2,True,kind='mean')
        inverse=npu._execute('rsqrt',npu._binary_op(variance,eps,kind='add'))
        result=npu._binary_op(centered,inverse,kind='mul').view(x.shape);affine_shape=(1,C)+(1,)*(x.ndim-2)
        if weight is not None:result=npu._binary_op(result,npu._cast(npu._materialize(weight),torch.float32).view(affine_shape),kind='mul')
        if bias is not None:result=npu._binary_op(result,npu._cast(npu._materialize(bias),torch.float32).view(affine_shape),kind='add')
        return npu._cast(result,x.dtype),npu._cast(mean.view(N,groups),stats_dtype),npu._cast(inverse.view(N,groups),stats_dtype)


def native_group_norm_backward(gradient,x,mean,inverse,weight,N,C,HxW,group,output_mask):
    with npu._lock:
        groups,stats_dtype=_norm_spec(x,weight,None,group,1e-5);npu._validate_float_read(gradient)
        if (N,C,HxW)!=(x.shape[0],x.shape[1],prod(x.shape[2:])) or gradient.shape!=x.shape or gradient.dtype!=x.dtype or len(output_mask)!=3:
            raise ValueError('NPU GroupNorm backward metadata mismatch')
        for value in (mean,inverse):
            npu._validate_float_read(value)
            if value.shape!=(N,groups) or value.dtype!=stats_dtype:raise ValueError('NPU GroupNorm saved statistics mismatch')
        values=npu._cast(npu._materialize(x),torch.float32).view(N,groups,-1)
        inv=npu._cast(npu._materialize(inverse),torch.float32).view(N,groups,1)
        normalized=npu._binary_op(npu._binary_op(values,npu._cast(npu._materialize(mean),torch.float32).view(N,groups,1),kind='sub'),inv,kind='mul')
        g=npu._cast(npu._materialize(gradient),torch.float32);axes=(0,)+tuple(range(2,x.ndim));gx=gw=gb=None
        if output_mask[1] and weight is not None:gw=npu._reduce(npu._execute('mul',g,normalized.view(x.shape)),axes,dtype=weight.dtype,kind='sum')
        if output_mask[2]:gb=npu._reduce(g,axes,dtype=stats_dtype,kind='sum')
        if output_mask[0]:
            scaled=g if weight is None else npu._binary_op(g,npu._cast(npu._materialize(weight),torch.float32).view((1,C)+(1,)*(x.ndim-2)),kind='mul')
            scaled=scaled.view(N,groups,-1)
            centered=npu._binary_op(scaled,npu._reduce(scaled,2,True,kind='mean'),kind='sub')
            projection=npu._reduce(npu._execute('mul',scaled,normalized),2,True,kind='mean')
            gx=npu._cast(npu._execute('mul',npu._binary_op(centered,npu._binary_op(normalized,projection,kind='mul'),kind='sub'),npu._broadcast(inv,tuple(scaled.shape))),x.dtype).view(x.shape)
        return gx,gw,gb


class GroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,N,C,HxW,group,eps):
        output,mean,inverse=native_group_norm(x,weight,bias,N,C,HxW,group,eps)
        ctx.save_for_backward(x,weight,mean,inverse);ctx.params=(N,C,HxW,group);ctx.bias_dtype=None if bias is None else bias.dtype
        ctx.mark_non_differentiable(mean,inverse)
        return output.detach(),mean,inverse

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient,_mean,_inverse):
        x,weight,mean,inverse=ctx.saved_tensors
        gx,gw,gb=native_group_norm_backward(gradient,x,mean,inverse,weight,*ctx.params,ctx.needs_input_grad[:3])
        if gb is not None:gb=npu._cast(gb,ctx.bias_dtype)
        return gx,gw,gb,None,None,None,None,None


def instance_norm(x,weight=None,bias=None,running_mean=None,running_var=None,use_input_stats=True,momentum=.1,eps=1e-5,cudnn_enabled=True):
    # InstanceNorm is BatchNorm over (1,N*C,H,W); all replication is device DMA.
    with npu._lock:
        with torch.no_grad():
            if x.ndim not in (3,4,5):raise ValueError('NPU InstanceNorm requires NCL/NCHW/NCDHW')
            _norm_spec(x,weight,bias,x.shape[1],eps)
            N,C=x.shape[:2]
            if not isfinite(momentum) or (running_mean is None)!=(running_var is None) or not use_input_stats and running_mean is None:
                raise ValueError('NPU InstanceNorm requires finite momentum and paired running statistics in eval')
            if use_input_stats and prod(x.shape[2:])<=1:raise ValueError('NPU InstanceNorm input statistics require >1 spatial value')
            for value in (running_mean,running_var):
                if value is not None:
                    npu._validate(value,dense=True)
                    if value.requires_grad or value.shape!=(C,) or value.dtype not in (x.dtype,torch.float32):raise ValueError('NPU InstanceNorm running buffer mismatch')
            owners=[v.untyped_storage()._cdata for v in (x,weight,bias,running_mean,running_var) if v is not None]
            if len(owners)!=len(set(owners)):raise ValueError('NPU InstanceNorm tensors must not alias')
        def repeated(v):return None if v is None else v.view(1,C).expand(N,C).contiguous().view(-1)
        w,b,rm,rv=(repeated(v) for v in (weight,bias,running_mean,running_var))
        value=x.reshape(1,N*C,1,-1)
        result=torch.ops.aten.batch_norm(value,w,b,rm,rv,use_input_stats,momentum,eps,False).view(x.shape)
        if use_input_stats and running_mean is not None:
            with torch.no_grad():
                for destination,source in ((running_mean,rm),(running_var,rv)):
                    npu._copy(destination,npu._reduce(source.view(N,C),0,dtype=destination.dtype,kind='mean'))
        # The public reshape is differentiable. Give following in-place
        # activations real independent storage through the registered clone.
        return result.clone() if torch.is_grad_enabled() and result.requires_grad else result


def _transpose_spec(x,weight,bias,stride,padding,dilation,output_padding,groups):
    npu._validate_float_read(x);npu._validate_float_read(weight)
    if x.ndim!=4 or weight.ndim!=4 or x.dtype!=weight.dtype or type(groups) is not int or groups<1 or x.shape[1]!=weight.shape[0] or x.shape[1]%groups:
        raise ValueError('NPU ConvTranspose requires matching NCHW tensors and channel groups')
    for name,values,minimum in (('stride',stride,1),('padding',padding,0),('dilation',dilation,1),('output_padding',output_padding,0)):
        if len(values)!=2 or any(type(v) is not int or v<minimum for v in values):raise ValueError('Invalid NPU ConvTranspose '+name)
    if any(o>=s and o>=d for o,s,d in zip(output_padding,stride,dilation)):raise ValueError('NPU ConvTranspose output_padding must be smaller than stride or dilation')
    spatial=tuple((n-1)*s-2*p+d*(k-1)+o+1 for n,s,p,d,k,o in zip(x.shape[2:],stride,padding,dilation,weight.shape[2:],output_padding))
    shape=(x.shape[0],weight.shape[1]*groups,*spatial)
    if any(v<1 for v in spatial) or prod(shape)>npu.MAX_TENSOR_ELEMENTS or prod(weight.shape[2:])*max(x.shape[1]//groups,weight.shape[1])>65536:
        raise ValueError('NPU ConvTranspose output/window exceeds storage bounds')
    if bias is not None:
        npu._validate_float_read(bias)
        if bias.shape!=(shape[1],) or bias.dtype!=x.dtype:raise ValueError('NPU ConvTranspose bias mismatch')
    return shape


def _transpose_tiles(source,input_shape,output_shape,params,backward=False):
    rows=prod(input_shape)//input_shape[1] if backward else prod(output_shape)//output_shape[1]
    width=prod(params[0])*(output_shape[1] if backward else input_shape[1]);batch=max(1,65536//width)
    for start in range(0,rows,batch):
        count=min(batch,rows-start)
        indices=transpose_conv_windows(input_shape,output_shape,*params,start,count,backward=backward)
        yield start,count,npu._gather_window(source,indices,1)


def transpose_forward(x,weight,bias,stride,padding,dilation,output_padding,groups):
    with npu._lock:
        shape=_transpose_spec(x,weight,bias,stride,padding,dilation,output_padding,groups)
        cin=x.shape[1]//groups;cout=weight.shape[1];params=tuple(weight.shape[2:]),tuple(stride),tuple(padding),tuple(dilation);pieces=[]
        for group in range(groups):
            a=npu._cast(npu._materialize(x.narrow(1,group*cin,cin)),torch.float32)
            w=npu._cast(npu._materialize(weight.narrow(0,group*cin,cin)),torch.float32)
            original=tuple(a.shape);target=(shape[0],cout,*shape[2:])
            source=npu._materialize(a.permute(0,2,3,1));matrix=npu._materialize(w.permute(2,3,0,1)).view(-1,cout)
            parts=[npu._matmul(window,matrix) for _,_,window in _transpose_tiles(source,original,target,params)]
            result=npu._materialize(npu._concat(parts,(shape[0],*shape[2:],cout),torch.float32).permute(0,3,1,2))
            if bias is not None:result=npu._binary_op(result,npu._cast(npu._materialize(bias.narrow(0,group*cout,cout)),torch.float32).view(1,cout,1,1),kind='add')
            pieces.append(npu._cast(result,x.dtype))
        return pieces[0] if groups==1 else npu._cat(pieces,1)


class ConvTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,stride,padding,dilation,output_padding,groups):
        result=transpose_forward(x,weight,bias,stride,padding,dilation,output_padding,groups)
        ctx.save_for_backward(x,weight);ctx.params=(tuple(weight.shape[2:]),tuple(stride),tuple(padding),tuple(dilation));ctx.groups=groups;ctx.bias_dtype=None if bias is None else bias.dtype
        return result.detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        x,weight=ctx.saved_tensors;cin=x.shape[1]//ctx.groups;cout=weight.shape[1];xs=[];ws=[]
        with npu._lock:
            for group in range(ctx.groups):
                a=npu._materialize(x.narrow(1,group*cin,cin));w=npu._materialize(weight.narrow(0,group*cin,cin))
                g=npu._cast(npu._materialize(gradient.narrow(1,group*cout,cout)),torch.float32);source=npu._materialize(g.permute(0,2,3,1))
                inputs=npu._materialize(npu._cast(a,torch.float32).permute(0,2,3,1)).view(-1,cin)
                matrix=npu._materialize(npu._cast(w,torch.float32).permute(2,3,1,0)).view(-1,cin)
                gx=[];levels=[]
                if any(ctx.needs_input_grad[:2]):
                    for start,count,window in _transpose_tiles(source,tuple(a.shape),tuple(g.shape),ctx.params,True):
                        if ctx.needs_input_grad[0]:gx.append(npu._matmul(window,matrix))
                        if ctx.needs_input_grad[1]:
                            part=npu._matmul(npu._slice(inputs,start*cin,count*cin).view(count,cin).t(),window);level=0
                            while level<len(levels) and levels[level] is not None:
                                part=npu._execute('add',levels[level],part);levels[level]=None;level+=1
                            if level==len(levels):levels.append(part)
                            else:levels[level]=part
                if gx:xs.append(npu._cast(npu._materialize(npu._concat(gx,(x.shape[0],*x.shape[2:],cin),torch.float32).permute(0,3,1,2)),x.dtype))
                if levels:
                    total=None
                    for part in reversed(levels):
                        if part is not None:total=part if total is None else npu._execute('add',total,part)
                    ws.append(npu._cast(npu._materialize(total.view(cin,*weight.shape[2:],cout).permute(0,3,1,2)),weight.dtype))
            gb=npu._reduce(npu._cast(npu._materialize(gradient),torch.float32),(0,2,3),dtype=ctx.bias_dtype,kind='sum') if ctx.needs_input_grad[2] else None
            return (npu._cat(xs,1) if xs else None),(npu._cat(ws,0) if ws else None),gb,None,None,None,None,None


def conv_transpose(x,w,b,s,p,d,o,g):
    from torch_npu_autograd import _active
    return ConvTranspose.apply(x,w,b,s,p,d,o,g) if _active(x,w,b) else transpose_forward(x,w,b,s,p,d,o,g)


def register(aten,autograd):
    from torch_npu_autograd import _active
    for dims in (1,2,3):
        for mode,name in (('nearest','upsample_nearest'),('nearest-exact','_upsample_nearest_exact'),
                          ('linear','upsample_linear' if dims==1 else 'upsample_bilinear' if dims==2 else 'upsample_trilinear'),('cubic','upsample_bicubic')):
            if dims!=2 and mode=='cubic':continue
            op=f'{name}{dims}d';linear=mode in ('linear','cubic')
            def raw(x,output_size,*args,out=None,mode=mode,linear=linear,dims=dims):
                align=bool(args[0]) if linear else False;scales=args[1:] if linear else args
                return upsample(x,output_size,align,tuple(scales)+(None,)*(dims-len(scales)),mode=mode,out=out)
            def active(x,output_size,*args,mode=mode,linear=linear,dims=dims):
                align=bool(args[0]) if linear else False;scales=tuple(args[1:] if linear else args);scales+=(None,)*(dims-len(scales))
                return Upsample.apply(x,output_size,align,scales,mode) if _active(x) else upsample(x,output_size,align,scales,mode=mode)
            def backward(gradient,output_size,input_size,*args,grad_input=None,mode=mode,linear=linear,dims=dims):
                align=bool(args[0]) if linear else False;scales=tuple(args[1:] if linear else args);scales+=(None,)*(dims-len(scales))
                return upsample_backward(gradient,output_size,input_size,align,scales,mode=mode,grad_input=grad_input)
            aten.impl(op,raw);aten.impl(op+'.out',raw);autograd.impl(op,active)
            aten.impl(op+'_backward',backward);aten.impl(op+'_backward.grad_input',backward)
    for mode,op in (('linear-aa','_upsample_bilinear2d_aa'),('cubic-aa','_upsample_bicubic2d_aa')):
        def raw(x,output_size,align_corners,scales_h=None,scales_w=None,*,out=None,mode=mode):
            return upsample(x,output_size,align_corners,(scales_h,scales_w),mode=mode,out=out)
        def active(x,output_size,align_corners,scales_h=None,scales_w=None,mode=mode):
            return Upsample.apply(x,output_size,align_corners,(scales_h,scales_w),mode) if _active(x) else raw(x,output_size,align_corners,scales_h,scales_w,mode=mode)
        def backward(g,output_size,input_size,align_corners,scales_h=None,scales_w=None,*,grad_input=None,mode=mode):
            return upsample_backward(g,output_size,input_size,align_corners,(scales_h,scales_w),mode=mode,grad_input=grad_input)
        aten.impl(op,raw);aten.impl(op+'.out',raw);autograd.impl(op,active)
        aten.impl(op+'_backward',backward);aten.impl(op+'_backward.grad_input',backward)
    aten.impl('native_group_norm',native_group_norm)
    aten.impl('native_group_norm_backward',native_group_norm_backward)
    def native(x,w,b,N,C,HxW,g,eps):
        return GroupNorm.apply(x,w,b,N,C,HxW,g,eps) if _active(x,w,b) else native_group_norm(x,w,b,N,C,HxW,g,eps)
    def group_norm(x,num_groups,weight=None,bias=None,eps=1e-5,cudnn_enabled=True):
        return native(x,weight,bias,x.shape[0],x.shape[1],prod(x.shape[2:]),num_groups,eps)[0]
    autograd.impl('native_group_norm',native);aten.impl('group_norm',group_norm);autograd.impl('group_norm',group_norm)
    aten.impl('instance_norm',instance_norm);autograd.impl('instance_norm',instance_norm)

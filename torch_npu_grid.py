"""Differentiable affine grids and dynamic SHAVE sampling (2D / 3D)."""
from math import prod
import numpy as np
import torch
from torch.autograd.function import once_differentiable
import torch_intel_npu as npu
from torch_npu_vision_extended import indexed_read


def _base(size,align):
    axes=[np.linspace(-1,1,n,dtype=np.float32) if align and n>1 else
          ((np.arange(n,dtype=np.float32)+.5)*2/n-1) for n in size[2:]]
    grid=np.meshgrid(*axes,indexing='ij');values=np.stack(tuple(reversed(grid))+(np.ones(size[2:],dtype=np.float32),),axis=-1)
    return torch.from_numpy(values.reshape(1,-1,len(size)-1)).to('intel_npu')


class AffineGrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx,theta,size,align):
        npu._validate_float_read(theta);dims=len(size)-2
        if dims not in (2,3) or theta.shape!=(size[0],dims,dims+1) or any(n<1 for n in size):raise ValueError('Invalid NPU affine grid geometry')
        shape=(size[0],*size[2:],dims)
        if prod(shape)>npu.MAX_TENSOR_ELEMENTS:raise ValueError('Affine grid exceeds storage bound')
        base=npu._broadcast(_base(size,align),(size[0],prod(size[2:]),dims+1));ctx.save_for_backward(base);ctx.dtype=theta.dtype;ctx.dims=dims
        result=npu._bmm(base,npu._materialize(npu._cast(npu._materialize(theta),torch.float32).transpose(1,2)))
        return npu._cast(result,theta.dtype).view(shape).detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        base,=ctx.saved_tensors
        value=npu._cast(npu._materialize(g),torch.float32).view(base.shape[0],-1,ctx.dims)
        return npu._cast(npu._bmm(npu._materialize(value.transpose(1,2)),base),ctx.dtype),None,None


def _spec(x,grid,mode,padding,align):
    npu._validate_float_read(x);npu._validate_float_read(grid);dims=x.ndim-2
    if dims not in (2,3) or grid.ndim!=x.ndim or grid.shape[0]!=x.shape[0] or grid.shape[-1]!=dims or grid.dtype!=x.dtype:
        raise ValueError('NPU grid_sample requires matching dtype NCHW/NCDHW input and grid')
    if mode not in (0,1,2) or dims==3 and mode==2 or padding not in (0,1,2):raise ValueError('Invalid grid_sample mode/padding')
    taps=1 if mode==1 else 16 if mode==2 else 2**dims;shape=tuple(x.shape[:2])+tuple(grid.shape[1:-1])
    if not x.numel() or not prod(shape) or prod(shape)>npu.MAX_TENSOR_ELEMENTS or prod(shape)*taps>8388608:
        raise ValueError('NPU grid_sample exceeds storage or sampled-cell bounds')
    return dims,taps,shape


def _plans(grid,spatial,mode,padding,align):
    dims=len(spatial);taps=1 if mode==1 else 16 if mode==2 else 2**dims;fields=3+dims
    data=npu._cast(npu._materialize(grid),torch.float32).view(-1);points=grid.numel()//dims
    for start in range(0,points,512):
        count=min(512,points-start);size=max(16,8+count*dims,count*taps*fields+1)
        extent=tuple(reversed(spatial))+(1,)*(3-dims)
        header=torch.tensor([dims,mode,padding,int(align),count,*extent],dtype=torch.float32).to(grid.device)
        parts=[header,npu._slice(data,start*dims,count*dims)]
        if size>8+count*dims:parts.append(npu._broadcast(npu._scalar(0.,torch.float32),(size-8-count*dims,)))
        packet=npu._concat(parts,(1,size),torch.float32)
        raw=npu._run(('grid-stencil-f32','f32',(1,size)),[packet,packet],(1,size),torch.float32)
        if npu._slice(raw,size-1,1).item():raise ValueError('Grid coordinates exceed the finite NPU coordinate bound')
        offsets=np.arange(count*taps,dtype=np.int64)[:,None]*fields
        indices=npu._gather_window(raw,offsets+np.arange(2),1).view(-1).view(torch.int64).view(count,taps)
        weights=npu._gather_window(raw,offsets+2,1).view(count,taps)
        derivatives=[npu._gather_window(raw,offsets+3+a,1).view(count,taps) for a in range(dims)]
        yield start,count,indices,weights,derivatives


def forward(x,grid,mode,padding,align):
    with npu._lock:
        dims,taps,shape=_spec(x,grid,mode,padding,align);plane=prod(x.shape[2:]);outputs=[]
        data=npu._cast(npu._materialize(x),torch.float32).view(-1)
        for batch in range(x.shape[0]):
            parts=[[] for _ in range(x.shape[1])];coordinates=npu._materialize(grid.narrow(0,batch,1))
            for start,count,indices,weights,_ in _plans(coordinates,tuple(x.shape[2:]),mode,padding,align):
                for channel in range(x.shape[1]):
                    source=npu._slice(data,(batch*x.shape[1]+channel)*plane,plane)
                    samples=indexed_read(source,indices,count*taps,plane)
                    parts[channel].append(npu._reduce(npu._execute('mul',samples,weights),1,kind='sum'))
            outputs.extend(npu._concat(p,(prod(shape[2:]),),torch.float32) for p in parts)
        return npu._cast(npu._concat(outputs,shape,torch.float32),x.dtype)


def backward(g,x,grid,mode,padding,align,output_mask=(True,True)):
    with npu._lock:
        dims,taps,shape=_spec(x,grid,mode,padding,align)
        npu._validate_float_read(g)
        if tuple(g.shape)!=shape or g.dtype!=x.dtype:raise ValueError('NPU grid_sample gradient mismatch')
        plane=prod(x.shape[2:]);outplane=prod(shape[2:]);xs=[];grids=[]
        data=npu._cast(npu._materialize(x),torch.float32).view(-1);grad=npu._cast(npu._materialize(g),torch.float32).view(-1)
        for batch in range(x.shape[0]):
            input_parts=[None]*x.shape[1];grid_parts=[];coordinates=npu._materialize(grid.narrow(0,batch,1))
            for start,count,indices,weights,derivatives in _plans(coordinates,tuple(x.shape[2:]),mode,padding,align):
                coordinate_parts=[None]*dims
                for channel in range(x.shape[1]):
                    values=npu._slice(grad,(batch*x.shape[1]+channel)*outplane+start,count).view(count,1)
                    if output_mask[0]:
                        weighted=npu._binary_op(values,weights,kind='mul')
                        partial=npu._selection_backward(weighted,indices,(plane,),count*taps,plane,allow_padding=True)
                        input_parts[channel]=partial if input_parts[channel] is None else npu._execute('add',input_parts[channel],partial)
                    if output_mask[1]:
                        if mode==1:
                            for a in range(dims):coordinate_parts[a]=npu._broadcast(npu._scalar(0.,torch.float32),(count,))
                            continue
                        samples=indexed_read(npu._slice(data,(batch*x.shape[1]+channel)*plane,plane),indices,count*taps,plane)
                        for a in range(dims):
                            partial=npu._execute('mul',values.view(count),npu._reduce(npu._execute('mul',samples,derivatives[a]),1,kind='sum'))
                            coordinate_parts[a]=partial if coordinate_parts[a] is None else npu._execute('add',coordinate_parts[a],partial)
                if output_mask[1]:grid_parts.append(npu._stack(coordinate_parts,1))
            if output_mask[0]:xs.extend(input_parts)
            if output_mask[1]:grids.append(npu._concat(grid_parts,(outplane,dims),torch.float32))
        gx=npu._cast(npu._concat(xs,tuple(x.shape),torch.float32),x.dtype) if output_mask[0] else None
        gg=npu._cast(npu._concat(grids,tuple(grid.shape),torch.float32),grid.dtype) if output_mask[1] else None
        return gx,gg


class GridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,grid,mode,padding,align):
        result=forward(x,grid,mode,padding,align);ctx.save_for_backward(x,grid);ctx.params=(mode,padding,align)
        return result.detach()

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        x,grid=ctx.saved_tensors;gx,gg=backward(g,x,grid,*ctx.params,ctx.needs_input_grad[:2])
        return gx,gg,None,None,None


def register(aten,autograd):
    from torch_npu_autograd import _active
    for library in (aten,autograd):
        library.impl('affine_grid_generator',lambda theta,size,align_corners:AffineGrid.apply(theta,tuple(size),align_corners))
        for dims in (2,3):
            def sample(x,grid,interpolation_mode,padding_mode,align_corners):
                return GridSample.apply(x,grid,interpolation_mode,padding_mode,align_corners) if _active(x,grid) else forward(x,grid,interpolation_mode,padding_mode,align_corners)
            library.impl('grid_sampler_'+str(dims)+'d',sample)
    for dims in (2,3):
        aten.impl('grid_sampler_'+str(dims)+'d_backward',backward)

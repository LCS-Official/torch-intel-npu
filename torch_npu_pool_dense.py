"""Bounded ordinary MaxPool windows and geometry-proven autograd pruning."""
from math import prod
import struct
import numpy as np
import torch
import torch_intel_npu as npu
from torch_npu_geometry import pool_dense_tiles,pool_gradient_ranges,max_pool_windows


def aligned_slice(x,start,count):
    page=4096//x.element_size();base=start//page*page
    cover=npu._slice(x,base,start-base+count)
    return npu._slice(cover,start-base,count) if start!=base else cover


def forward(x,params,shape,*,out=None,indices=None):
    data=npu._materialize(x).view(-1);spatial=tuple(x.shape[-2:]);plane=prod(spatial);per=prod(params[-1])
    values=torch.empty(shape,dtype=x.dtype,device=x.device) if out is None else out
    ids=torch.empty(shape,dtype=torch.int64,device=x.device) if indices is None else indices
    cursors=[0,0];tails=[None,None]
    for start,count,base,span in pool_dense_tiles(spatial,params,x.numel()//plane,4096//x.element_size()):
        if base is None:
            locations=max_pool_windows(spatial,params,start,count)
            channels=np.arange(start,start+count,dtype=np.int64)//per
            absolute=np.where(locations<0,-1,locations+channels[:,None]*plane)
            windows=npu._cast(npu._gather_window(data,absolute,1),torch.float32)
            v,i=npu._select_rows(windows,1,indices=torch.from_numpy(locations).to(x.device),pool=True)
        else:
            fields=(count,start,*spatial,*params[-1],*params[0],*params[1],*params[2],*params[3],base,span)
            header=torch.frombuffer(bytearray(struct.pack('<16I',*fields)),dtype=torch.float32).to(x.device)
            source=npu._cast(npu._slice(data,base,span),torch.float32);size=32768
            packet=npu._concat([header,source,npu._broadcast(npu._scalar(0.,torch.float32),(size-16-span,))],(1,size),torch.float32)
            result=npu._run(('pool-dense-f32','f32',(1,size)),[packet,packet],(1,size),torch.float32)
            if npu._slice(result,size-1,1).item()!=0:raise RuntimeError('NPU dense pool geometry/source span rejected')
            v=npu._slice(result,0,count);i=npu._slice(result,count,2*count).view(torch.int64)
        for lane,part in enumerate((npu._cast(v.view(-1),x.dtype),i.view(-1))):
            cursors[lane],tails[lane]=npu._stream_write((values,ids)[lane],cursors[lane],tails[lane],part)
    return tuple(npu._stream_finish(t,c,p) for t,c,p in zip((values,ids),cursors,tails))


def backward(gradient,indices,shape,params):
    """Only for saved, version-checked indices produced by the forward above."""
    spatial=tuple(shape[-2:]);plane=prod(spatial);per=prod(params[-1]);g=npu._materialize(gradient).view(-1)
    ids=npu._materialize(indices).view(-1)
    def pieces():
        for channel in range(prod(shape)//plane):
            for base in range(0,plane,61440):
                span=min(61440,plane-base);initial=npu._broadcast(npu._scalar(0.,torch.float32),(span,))
                for start,count in pool_gradient_ranges(spatial,params,base,span):
                    for offset in range(0,count,21504):
                        length=min(21504,count-offset);position=start+offset;absolute=channel*per+position
                        a=aligned_slice(g,absolute,length);b=aligned_slice(ids,absolute,length)
                        initial=npu._selection_accumulate(a,b,per,plane,base,span,plane,position,initial,
                            half=gradient.dtype==torch.float16,wide=True)
                yield npu._cast(initial,gradient.dtype)
    return npu._stream_concat(pieces(),shape,dtype=gradient.dtype)

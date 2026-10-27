"""S37 pool/selection baseline; execute inside torch_intel_npu globals.
Backward accepts and ignores S38 known_windows to preserve the old algorithm.
"""

def _selection_backward(gradient,indices,shape,per,width,*,accumulate=True,allow_padding=False,initial_value=None):
    with _lock:
        _validate_float_read(gradient);_validate_read(indices)
        if indices.dtype!=torch.int64 or gradient.shape!=indices.shape:
            raise ValueError('NPU selection gradient requires matching FP16/FP32 gradient and INT64 indices')
        total=prod(shape);count=gradient.numel()
        if (any(type(n) is not int or n<0 for n in shape) or
                any(type(n) is not int or n<0 for n in (per,width)) or
                max(total,count,per,width)>_storage_limit(gradient.dtype)):
            raise ValueError('NPU selection backward exceeds geometry/storage bounds')
        if initial_value is not None:
            _validate_float_read(initial_value)
            if tuple(initial_value.shape)!=tuple(shape) or initial_value.dtype!=gradient.dtype:raise ValueError('Selection initial value shape/dtype mismatch')
            initial_value=_cast(_materialize(initial_value),torch.float32).view(-1)
        if not count:return _cast(_clone(initial_value).view(shape),gradient.dtype) if initial_value is not None else _fill(torch.empty(shape,dtype=gradient.dtype,device=gradient.device),0.)
        if not per or not width or count%per or total%width or total//width!=count//per:
            raise ValueError('NPU selection gradient geometry mismatch')
        wide=max(total,count,per,width)>MAX_TENSOR_ELEMENTS
        target_batch,source_batch=(64512,21504) if wide else (65535,21842)
        g=_cast(_materialize(gradient),torch.float32).view(-1)
        words=_materialize(indices).view(-1).view(torch.float32);pieces=[]
        for base in range(0,total,target_batch):
            span=min(target_batch,total-base);initial=_slice(initial_value,base,span) if initial_value is not None else _broadcast(_scalar(0.,torch.float32),(span,))
            for start in range(0,count,source_batch):
                length=min(source_batch,count-start);size=max(9+3*length,span+1)
                fields=(length,start,per,width,base,span,int(gradient.dtype==torch.float16)+2*int(not accumulate)+4*int(allow_padding),total)
                header=(torch.frombuffer(bytearray(struct.pack('<8I',*fields)),dtype=torch.float32).to(gradient.device)
                        if wide else _concat([_scalar(float(v),torch.float32).view(1) for v in fields],(8,),torch.float32))
                parts=[header,_slice(g,start,length),_slice(words,2*start,2*length),
                       _broadcast(_scalar(0.,torch.float32),(size-8-3*length,))]
                packet=_concat(parts,(1,size),torch.float32)
                second=_concat([initial,_broadcast(_scalar(0.,torch.float32),(size-span,))],(1,size),torch.float32)
                result=_run(('selection-backward-u32' if wide else 'selection-backward-f32','f32',(1,size)),[packet,second],(1,size),torch.float32)
                if _slice(result,size-1,1).item()!=0:raise IndexError('NPU selection gradient index is out of bounds')
                initial=_slice(result,0,span)
            pieces.append(initial)
        return _cast(_stream_concat(iter(pieces),shape) if wide else _concat(pieces,shape,torch.float32),gradient.dtype)


def _selection_backward_planes(gradient,indices,shape,per,width,*,accumulate=True):
    with _lock:
        synchronize();_validate_float_read(gradient);_validate_read(indices)
        if indices.dtype!=torch.int64 or gradient.shape!=indices.shape:
            raise ValueError('NPU plane gradient requires matching INT64 indices')
        total=prod(shape)
        if (type(per) is not int or type(width) is not int or min(per,width)<0 or
                total>_storage_limit(gradient.dtype)):
            raise ValueError('NPU gradient plane geometry/storage mismatch')
        if not gradient.numel():return _selection_backward(gradient,indices,shape,per,width,accumulate=accumulate)
        if min(per,width)<1 or total%width or gradient.numel()!=total//width*per:
            raise ValueError('NPU gradient plane geometry/storage mismatch')
        from torch_npu_geometry import selection_plane_tiles
        plan=selection_plane_tiles(total//width,per,width,max_elements=_storage_limit(gradient.dtype))
        if len(plan)==1:
            return _selection_backward(gradient,indices,shape,per,width,accumulate=accumulate)
        data=_materialize(gradient).view(-1);ids=_materialize(indices).view(-1)
        def pieces():
            for start,count in plan:
                g=_slice(data,start*per,count*per);i=_slice(ids,start*per,count*per)
                yield _cast(_selection_backward(g,i,(count,width),per,width,accumulate=accumulate),torch.float32)
        return _cast(_stream_concat(pieces(),shape),gradient.dtype)


def _max_pool2d(x,kernel_size,stride=(),padding=0,dilation=1,ceil_mode=False,*,out=None,indices=None):
    import numpy as np
    with _lock:
        synchronize();params,shape=_max_pool2d_spec(x,kernel_size,stride,padding,dilation,ceil_mode,out,indices)
        if not prod(shape):
            return ((torch.empty(shape,dtype=x.dtype,device=x.device),torch.empty(shape,dtype=torch.int64,device=x.device))
                    if out is None else (out,indices))
        plane=prod(x.shape[-2:]);outputs=prod(params[-1]);width=prod(params[0]);planes=x.numel()//plane
        batch=min(65530//width,65536//(2*width),65536//3)
        data=_cast(_materialize(x),torch.float32).view(-1);values=[];ids=[]
        for start in range(0,planes*outputs,batch):
            count=min(batch,planes*outputs-start);locations=max_pool_windows(tuple(x.shape[-2:]),params,start,count)
            channels=np.arange(start,start+count,dtype=np.int64)//outputs
            absolute=np.where(locations<0,-1,locations+channels[:,None]*plane)
            windows=_gather_window(data,absolute,1)
            metadata=torch.from_numpy(locations).to('intel_npu')
            v,i=_select_rows(windows,1,indices=metadata,pool=True)
            values.append(v.view(-1));ids.append(i.view(-1))
        value=_cast(_concat(values,shape,torch.float32),x.dtype);index=_cat(ids,0).view(shape)
        return (_copy(out,value),_copy(indices,index)) if out is not None else (value,index)


def _max_pool2d_backward(gradient,x,kernel_size,stride,padding,dilation,ceil_mode,indices,*,grad_input=None,known_windows=False):
    with _lock:
        params,shape=_max_pool2d_spec(x,kernel_size,stride,padding,dilation,ceil_mode)
        if tuple(gradient.shape)!=shape or gradient.dtype!=x.dtype:
            raise ValueError('NPU MaxPool gradient shape/dtype mismatch')
        _check_out(grad_input,tuple(x.shape),x.dtype,(gradient,x,indices))
        result=_selection_backward_planes(gradient,indices,tuple(x.shape),prod(params[-1]),prod(x.shape[-2:]))
        return _copy(grad_input,result) if grad_input is not None else result

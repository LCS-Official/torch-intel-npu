"""Bounded operator shape/index metadata; no Torch tensors or activation values."""
import operator
from math import prod
import numpy as np


def resample_axis_geometry(size,output,mode,align_corners=False,scale=None,*,legacy_nearest=True):
    """One-dimensional interpolation coordinates/weights, without tensor data."""
    if any(type(v) is not int or v<1 for v in (size,output)) or mode not in ('nearest','nearest-exact','linear','cubic'):
        raise ValueError('NPU interpolation requires positive sizes and a supported mode')
    if scale is not None and (not np.isfinite(scale) or scale<=0):raise ValueError('NPU interpolation scale must be positive and finite')
    if size==output and (mode=='linear' or mode=='nearest' and legacy_nearest):
        return np.arange(output,dtype=np.int64)[:,None],np.ones((output,1),dtype=np.float32)
    ratio=np.float32((size-1)/(output-1) if align_corners and output>1 else
                     0 if align_corners else 1/scale if scale is not None else size/output)
    target=np.arange(output,dtype=np.float32)
    if mode in ('nearest','nearest-exact'):
        source=(np.arange(output)//2 if mode=='nearest' and legacy_nearest and output==2*size else
                np.floor((target+(np.float32(.5) if mode=='nearest-exact' else np.float32(0)))*ratio))
        return np.clip(source.astype(np.int64),0,size-1)[:,None],np.ones((output,1),dtype=np.float32)
    source=target*ratio if align_corners else (target+np.float32(.5))*ratio-np.float32(.5)
    if mode=='linear':source=np.maximum(source,0)
    base=np.floor(source).astype(np.int64);fraction=source-base.astype(np.float32)
    if mode=='linear':
        positions=base[:,None]+np.arange(2);weights=np.stack((1-fraction,fraction),axis=1)
    else:
        positions=base[:,None]+np.arange(-1,3)
        distance=np.abs(source[:,None]-positions.astype(np.float32));a=np.float32(-.75)
        weights=np.where(distance<=1,((a+2)*distance-(a+3))*distance*distance+1,
                         ((a*distance-5*a)*distance+8*a)*distance-4*a)
    return np.clip(positions,0,size-1),weights.astype(np.float32)


def transpose_resample_geometry(size,positions,weights):
    """Transpose a static sampling stencil, including repeated edge positions."""
    entries=[[] for _ in range(size)]
    for row in range(len(positions)):
        for pos,weight in zip(positions[row],weights[row]):
            if pos<0:continue
            entries[int(pos)].append((row,weight))
    width=max(1,max(map(len,entries)));indices=np.full((size,width),-1,dtype=np.int64);coeff=np.zeros((size,width),dtype=np.float32)
    for row,items in enumerate(entries):
        for col,(index,weight) in enumerate(items):indices[row,col]=index;coeff[row,col]=weight
    return indices,coeff


def antialias_axis_geometry(size,output,mode,align=False,scale=None):
    """Normalized separable triangle / Keys(-0.5) filter, clipped to the image."""
    if any(type(v) is not int or v<1 for v in (size,output)) or mode not in ('linear','cubic') or scale is not None and (not np.isfinite(scale) or scale<=0):
        raise ValueError('Invalid antialias size/mode/scale')
    if size==output:return np.arange(size,dtype=np.int64)[:,None],np.ones((size,1),dtype=np.float32)
    ratio=np.float32((size-1)/(output-1) if align and output>1 else 0 if align else 1/scale if scale is not None else size/output)
    support=max(1.,float(ratio))*(2 if mode=='cubic' else 1);items=[]
    for row in range(output):
        center=np.float32(ratio*np.float32(row+.5))
        left=max(0,int(center-support+.5));right=min(size,int(center+support+.5))
        pos=np.arange(left,right,dtype=np.int64);distance=np.abs((pos.astype(np.float32)-center+np.float32(.5))/max(1.,float(ratio)))
        if mode=='linear':coeff=np.maximum(0,1-distance)
        else:
            a=np.float32(-.5)
            coeff=np.where(distance<1,((a+2)*distance-(a+3))*distance*distance+1,
                           np.where(distance<2,((a*distance-5*a)*distance+8*a)*distance-4*a,0))
        coeff=coeff/coeff.sum(dtype=np.float32);items.append((pos,coeff))
    width=max(map(lambda p:len(p[0]),items));positions=np.full((output,width),-1,dtype=np.int64);weights=np.zeros((output,width),dtype=np.float32)
    for row,(pos,coeff) in enumerate(items):positions[row,:len(pos)]=pos;weights[row,:len(pos)]=coeff
    return positions,weights


def padding_axis_geometry(size,left,right,mode):
    output=size+left+right
    if size<1 or output<0:raise ValueError('Invalid NPU padding extent')
    pos=np.arange(output,dtype=np.int64)-left
    if mode=='constant':pos=np.where((pos>=0)&(pos<size),pos,-1)
    elif mode=='replicate':pos=np.clip(pos,0,size-1)
    elif mode=='reflect':
        if max(left,right)>=size:raise ValueError('Reflection padding must be smaller than input')
        pos=np.where(pos<0,-pos,np.where(pos>=size,2*size-2-pos,pos))
    elif mode=='circular':
        # Circular pad first crops negative extents, then wraps the remaining
        # image. Reflect/replicate instead address the original image directly.
        low=max(-left,0);remaining=size-low-max(-right,0)
        if remaining<1 or max(left,right)>remaining:raise ValueError('Circular padding cannot wrap more than once after cropping')
        pos=(np.arange(output,dtype=np.int64)-max(left,0))%remaining+low
    else:raise ValueError('Unsupported NPU padding mode')
    return pos[:,None],np.ones((output,1),dtype=np.float32)


def validate_resample_backward_geometry(spatial,output,mode,align_corners,scales):
    """Reject CPU legacy scale shortcuts whose backward is not the forward adjoint."""
    if mode=='nearest' and len(spatial)==1:
        actual,_=resample_axis_geometry(spatial[0],output[0],mode,False,scales[0],legacy_nearest=False)
        legacy,_=resample_axis_geometry(spatial[0],output[0],mode,False,scales[0],legacy_nearest=True)
        inconsistent=not np.array_equal(actual,legacy)
    else:
        inconsistent=(mode=='cubic' and tuple(spatial)==tuple(output) and not align_corners and
                      any(s is not None and s!=1 for s in scales))
    if inconsistent:
        raise ValueError('NPU interpolation training rejects a legacy CPU scale shortcut; use size= or recompute_scale_factor=True')


def transpose_conv_windows(input_shape,output_shape,kernel,stride,padding,dilation,start,count,*,backward=False):
    """NHWC DMA window indices for ConvTranspose forward or its adjoint."""
    batch,channels,ih,iw=input_shape;_,outputs,oh,ow=output_shape;kh,kw=kernel
    height,width,source_channels=(ih,iw,outputs) if backward else (oh,ow,channels)
    rows=batch*height*width;columns=kh*kw*source_channels
    if start<0 or count<1 or start+count>rows:raise ValueError('Invalid NPU ConvTranspose window tile')
    row=np.arange(start,start+count,dtype=np.int64)[:,None];col=np.arange(columns,dtype=np.int64)[None,:]
    b,y,x=row//(height*width),row//width%height,row%width
    ky,kx,c=col//(kw*source_channels),col//source_channels%kw,col%source_channels
    if backward:
        sy=y*stride[0]-padding[0]+ky*dilation[0];sx=x*stride[1]-padding[1]+kx*dilation[1]
        valid=(sy>=0)&(sy<oh)&(sx>=0)&(sx<ow);h,w=oh,ow
    else:
        uy=y+padding[0]-ky*dilation[0];ux=x+padding[1]-kx*dilation[1]
        sy,sx=uy//stride[0],ux//stride[1]
        valid=(uy%stride[0]==0)&(ux%stride[1]==0)&(sy>=0)&(sy<ih)&(sx>=0)&(sx<iw);h,w=ih,iw
    return np.where(valid,((b*h+sy)*w+sx)*source_channels+c,-1)


def convolution_windows_nd(input_shape,output_shape,kernel,stride,padding,dilation,start,count,*,transpose=False):
    """Channels-last windows for any 1D/2D/3D convolution direction."""
    spatial=input_shape[2:];target=output_shape[2:];channels=input_shape[1]
    rows=input_shape[0]*prod(target);columns=prod(kernel)*channels
    if start<0 or count<1 or start+count>rows:raise ValueError('Invalid NPU convolution tile')
    row=np.arange(start,start+count,dtype=np.int64)[:,None];col=np.arange(columns,dtype=np.int64)[None,:]
    result=(row//prod(target))*prod(spatial)*channels+col%channels;valid=np.ones((count,columns),dtype=bool)
    for axis,(n,m,k,s,p,d) in enumerate(zip(spatial,target,kernel,stride,padding,dilation)):
        coordinate=row//prod(target[axis+1:])%m;tap=col//(channels*prod(kernel[axis+1:]))%k
        raw=coordinate+p-tap*d if transpose else coordinate*s-p+tap*d
        source=raw//s if transpose else raw
        valid&=(source>=0)&(source<n)
        if transpose:valid&=raw%s==0
        result=result+source*prod(spatial[axis+1:])*channels
    return np.where(valid,result,-1)


def row_extrema_packet(rows, width, candidate_indices=False):
    if any(type(n) is not int or n < 1 for n in (rows, width)):
        raise ValueError('NPU row extrema require positive integer dimensions')
    size = max(4+rows*width, 3*rows, 2*rows*width if candidate_indices else 0)
    if rows > 21845 or size > 65536:
        raise ValueError('NPU row extrema packet exceeds 65536 words')
    return size


def selection_packet(rows,width,k,supplied=False):
    if any(type(n) is not int or n < 1 for n in (rows,width,k)) or k > min(width,128):
        raise ValueError('NPU selection requires positive dimensions and k<=min(width,128)')
    size=max(6+rows*width,3*rows*k,2*rows*width if supplied else 0)
    if size > 65536:raise ValueError('NPU selection packet exceeds 65536 words')
    return size


def max_pool2d_geometry(spatial,kernel,stride,padding,dilation,ceil_mode):
    kernel=pool_pair(kernel);stride=pool_pair(stride,kernel);padding=pool_pair(padding);dilation=pool_pair(dilation)
    if len(spatial)!=2 or any(type(n) is not int or n < 1 for n in spatial):
        raise ValueError('NPU MaxPool requires positive spatial dimensions')
    if (any(v < 1 for v in (*kernel,*stride,*dilation)) or
        any(p < 0 or p > k//2 for p,k in zip(padding,kernel)) or prod(kernel)>32765):
        raise ValueError('Invalid or oversized NPU MaxPool kernel/stride/padding/dilation')
    output=[]
    for n,k,s,p,d in zip(spatial,kernel,stride,padding,dilation):
        count=(n+2*p-d*(k-1)-1+(s-1 if ceil_mode else 0))//s+1
        if ceil_mode and (count-1)*s>=n+p:count-=1
        if count < 1:raise ValueError('NPU MaxPool has empty output')
        if count>134217728:raise ValueError('NPU MaxPool spatial output exceeds index storage bounds')
        if (count-1)*s-p>=n:raise ValueError('NPU MaxPool does not support windows without a valid sample')
        # Interior starts already name a valid sample; only left padding needs enumeration.
        starts=np.arange(min(count,(p+s-1)//s),dtype=np.int64)*s-p
        first=starts+np.maximum(0,(-starts+d-1)//d)*d
        if np.any(first >= np.minimum(starts+d*(k-1)+1,n)):
            raise ValueError('NPU MaxPool does not support windows without a valid sample')
        output.append(count)
    if prod(output)>134217728:raise ValueError('NPU MaxPool spatial output exceeds index storage bounds')
    return kernel,stride,padding,dilation,tuple(output)


def max_pool_windows(spatial,params,start,count):
    kernel,stride,padding,dilation,output=params
    # A tile may span multiple planes; returned indices remain plane-relative.
    locations=np.arange(start,start+count,dtype=np.int64)%prod(output)
    y=locations//output[1]*stride[0]-padding[0];x=locations%output[1]*stride[1]-padding[1]
    ys=y[:,None,None]+np.arange(kernel[0])[None,:,None]*dilation[0]
    xs=x[:,None,None]+np.arange(kernel[1])[None,None,:]*dilation[1]
    return np.where((ys>=0)&(ys<spatial[0])&(xs>=0)&(xs<spatial[1]),ys*spatial[1]+xs,-1).reshape(count,-1)


def selection_plane_tiles(planes,per,width,*,max_elements=16777216):
    if (type(planes) is not int or planes<0 or
            any(type(v) is not int or v<1 for v in (per,width,max_elements))):
        raise ValueError('Invalid NPU plane selection geometry')
    if not planes:return ()
    if max(per,width)>max_elements:raise ValueError('NPU selection plane exceeds metadata bounds')
    # Prefer a single result/input packet per group, retaining complete planes.
    batch=max(1,min(65535//width,21842//per))
    return tuple((start,min(batch,planes-start)) for start in range(0,planes,batch))


def pool_dense_tiles(spatial,params,planes,alignment=1024):
    """Bound contiguous source spans; a wide single window uses the old gather."""
    kernel,stride,padding,dilation,output=params;ih,iw=spatial;per=prod(output);plane=ih*iw
    start=0;batch=max(1,min(10922,32768//prod(kernel)));total=planes*per
    supported=(max(*spatial,*kernel,*stride,*padding,*dilation,*output)<=1073741824 and
        all((o-1)*s<=1073741824 and (k-1)*d<=1073741824 for o,s,k,d in zip(output,stride,kernel,dilation)))
    while start<total:
        count=min(batch,total-start)
        if count>=512:count=count//512*512
        while True:
            first,pos=divmod(start,per);last,end=divmod(start+count-1,per)
            y0,x0=divmod(pos,output[1]);y1,x1=divmod(end,output[1])
            same=first==last and y0==y1
            lo=first*plane+max(0,y0*stride[0]-padding[0])*iw+(max(0,x0*stride[1]-padding[1]) if same else 0)
            hi=last*plane+min(ih-1,y1*stride[0]-padding[0]+(kernel[0]-1)*dilation[0])*iw
            hi+=min(iw-1,x1*stride[1]-padding[1]+(kernel[1]-1)*dilation[1]) if same else iw-1
            base=lo//alignment*alignment
            span=min(planes*plane,((hi+alignment)//alignment)*alignment)-base
            if supported and span<=32751:break
            if count==1:base=span=None;break
            count=max(1,count//2)
            if count>=512:count=count//512*512
        yield start,count,base,span
        start+=count


def pool_gradient_ranges(spatial,params,base,span):
    """Conservative output windows reaching a contiguous input tile, in order."""
    kernel,stride,padding,dilation,output=params;ih,iw=spatial
    first,x0=divmod(base,iw);last,x1=divmod(base+span-1,iw)
    ceildiv=lambda a,b:-(-a//b)
    low=max(0,ceildiv(first+padding[0]-(kernel[0]-1)*dilation[0],stride[0]))
    high=min(output[0]-1,(last+padding[0])//stride[0]);pending=None
    for row in range(low,high+1):
        origin=row*stride[0]-padding[0]
        a=max(0,ceildiv(first-origin,dilation[0]));b=min(kernel[0]-1,(last-origin)//dilation[0])
        if a>b:continue
        ymin,ymax=origin+a*dilation[0],origin+b*dilation[0]
        left=x0 if ymin==ymax==first else 0
        right=x1 if ymin==ymax==last else iw-1
        begin=max(0,ceildiv(left+padding[1]-(kernel[1]-1)*dilation[1],stride[1]))
        end=min(output[1]-1,(right+padding[1])//stride[1])+1
        if begin>=end:continue
        start=row*output[1]+begin;count=end-begin
        if pending is not None and pending[0]+pending[1]==start:pending=(pending[0],pending[1]+count)
        else:
            if pending is not None:yield pending
            pending=(start,count)
    if pending is not None:yield pending


def indexed_rows_geometry(shape, index_shape, dim):
    if not 1 <= len(shape) <= 4 or len(shape) != len(index_shape) or any(type(n) is not int or n < 1 for n in (*shape, *index_shape)):
        raise ValueError('NPU gather/scatter requires nonempty rank 1-4 data and same-rank indices')
    dim = operator.index(dim)
    if not -len(shape) <= dim < len(shape):
        raise IndexError('NPU gather/scatter axis is out of range')
    dim %= len(shape)
    if any(shape[a] != index_shape[a] for a in range(len(shape)) if a != dim):
        raise ValueError('NPU gather/scatter currently requires equal non-indexed dimensions')
    rows = prod(shape)//shape[dim]
    # shortcut: row packets reuse current kernels; add fused address arithmetic for larger batches.
    if rows > 128 or max(prod(shape), prod(index_shape)) > 1048576:
        raise ValueError('NPU gather/scatter row path requires <=128 rows and bounded storage')
    index_tile_width(shape[dim], 1, index_shape[dim])
    return dim, rows, shape[dim], index_shape[dim]


def pool_pair(values, default=None):
    if isinstance(values, int):
        values = (values, values)
    else:
        values = tuple(values)
        if not values and default is not None:
            values = tuple(default)
        if len(values) == 1:
            values = values*2
    if len(values) != 2 or any(type(v) is not int for v in values):
        raise ValueError('NPU pooling requires integer scalar or one/two spatial parameters')
    return values


def avg_pool2d_geometry(spatial, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override):
    if len(spatial) != 2 or any(type(n) is not int or n < 1 for n in spatial):
        raise ValueError('NPU pooling requires two positive spatial dimensions')
    kernel = pool_pair(kernel); stride = pool_pair(stride, kernel); padding = pool_pair(padding)
    if (any(v < 1 for v in (*kernel, *stride)) or
        any(p < 0 or p > k//2 for p, k in zip(padding, kernel)) or
        divisor_override is not None and (type(divisor_override) is not int or divisor_override == 0 or abs(divisor_override) > 2**24)):
        raise ValueError('Invalid NPU average-pooling geometry or divisor')
    output, extra = [], []
    for length, k, s, p in zip(spatial, kernel, stride, padding):
        count = (length+2*p-k+(s-1 if ceil_mode else 0))//s+1
        if ceil_mode and (count-1)*s >= length+p:
            count -= 1
        if count < 1:
            raise ValueError('NPU average pooling has empty output')
        output.append(count); extra.append(max(0, (count-1)*s+k-length-2*p))
    if prod(output) > 1048576:
        raise ValueError('NPU average pooling spatial output exceeds storage bound')
    counts = []
    for length, count, k, s, p in zip(spatial, output, kernel, stride, padding):
        start = np.arange(count, dtype=np.int64)*s-p
        end = np.minimum(start+k, length+p)
        counts.append(end-start if count_include_pad else np.minimum(end, length)-np.maximum(start, 0))
    divisors = counts[0][:,None]*counts[1][None,:]
    if divisor_override is not None:
        divisors.fill(divisor_override)
    return kernel, stride, padding, tuple(output), tuple(extra), divisors.astype(np.float32)


def softmax_row_tiles(rows, columns, *, max_elements=1048576):
    """Split complete rows without changing an individual reduction's order."""
    if any(type(n) is not int or n<1 for n in (rows,columns,max_elements)) or rows*columns>max_elements:
        raise ValueError('NPU row storage requires positive dimensions within its caller storage bound')
    if columns>65536:
        raise ValueError('NPU softmax/log_softmax requires row width <=65536')
    batch=65536//columns
    return tuple((start,min(batch,rows-start)) for start in range(0,rows,batch))


def nll_row_tiles(rows, columns, *, max_elements):
    """Keep class rows intact; four extra words carry labels and validation."""
    if (any(type(n) is not int for n in (rows,columns,max_elements)) or rows<0 or columns<1 or
            max_elements<1 or rows*columns>max_elements or rows and columns+4>65536):
        raise ValueError('NPU NLL requires bounded storage and class width <=65532 per nonempty row')
    batch=65536//(columns+4) if rows else 1
    # Align FP32 rows and INT64 labels so tall batches reuse native range graphs.
    if rows>batch and batch>=1024:batch=batch//1024*1024
    return tuple((start,min(batch,rows-start)) for start in range(0,rows,batch))


def index_tile_width(rows, width, count):
    """Keep every complete-index packet/workspace bounded; split only columns."""
    if any(type(n) is not int or n<1 for n in (rows,width,count)):
        raise ValueError('NPU index dimensions must be positive integers')
    if max(rows,count)*width>1048576:
        raise ValueError('NPU index source/output exceeds 1048576 elements')
    tile=min(width,(65530-2*count)//max(rows,count),1048576//(rows*count))
    if tile<1:
        raise ValueError('NPU index packet/work limit cannot hold one complete column')
    return tile


def reduction_geometry(shape, dim=None, keepdim=False):
    requested=dim if isinstance(dim,(tuple,list)) else (() if dim is None else (dim,))
    rank=len(shape) or 1;axes=[]
    for value in requested:
        axis=operator.index(value)
        if not -rank<=axis<rank:raise IndexError('NPU reduction axis is out of range')
        axis%=rank
        if axis in axes:raise ValueError('NPU reduction axes must be distinct')
        axes.append(axis)
    axes=tuple(sorted(axes)) if axes else tuple(range(rank))
    result=tuple(1 if i in axes else d for i,d in enumerate(shape)) if keepdim else tuple(d for i,d in enumerate(shape) if i not in axes)
    return axes,result


def triangle_indices(shape, diagonal=0, *, upper=True):
    if not 2<=len(shape)<=4 or any(n<=0 for n in shape) or prod(shape)>1048576:
        raise ValueError('NPU triangle requires a bounded nonempty rank 2-4 input')
    diagonal=operator.index(diagonal);rows,columns=shape[-2:]
    flat=np.arange(prod(shape),dtype=np.int64)
    delta=flat%columns-(flat//columns)%rows
    return np.where(delta>=diagonal if upper else delta<=diagonal,flat,-1).reshape(1,-1)


def conv_geometry(shape, kernel, stride, padding, dilation):
    batch, channels, height, width = shape
    kh, kw = kernel
    oh = (height+2*padding[0]-dilation[0]*(kh-1)-1)//stride[0]+1
    ow = (width+2*padding[1]-dilation[1]*(kw-1)-1)//stride[1]+1
    if oh <= 0 or ow <= 0:
        raise ValueError('NPU convolution has empty output')
    return (oh, ow), (batch*oh*ow, channels*kh*kw)


def conv_window_indices(shape, kernel, stride, padding, dilation, start, count, *, output_channels=None):
    """A row tile of forward/input-gradient windows from an NHWC source.

    Forward columns follow (input_channel, ky, kx). Input-gradient columns
    follow (ky, kx, output_channel), including stride divisibility checks.
    Negative indices denote zero padding.
    """
    batch, channels, height, width = shape
    kh, kw = kernel
    (oh, ow), (rows, columns) = conv_geometry(shape, kernel, stride, padding, dilation)
    if output_channels is not None:
        rows, columns = batch*height*width, kh*kw*output_channels
    if start < 0 or count <= 0 or start+count > rows:
        raise ValueError('Invalid convolution window row tile')
    row = np.arange(start, start+count, dtype=np.int64)[:, None]
    col = np.arange(columns, dtype=np.int64)[None, :]
    if output_channels is None:
        b, y, x = row//(oh*ow), row//ow % oh, row % ow
        c, ky, kx = col//(kh*kw), col//kw % kh, col % kw
        iy = y*stride[0]-padding[0]+ky*dilation[0]
        ix = x*stride[1]-padding[1]+kx*dilation[1]
        valid = (iy >= 0) & (iy < height) & (ix >= 0) & (ix < width)
        indices = ((b*height+iy)*width+ix)*channels+c
    else:
        b, y, x = row//(height*width), row//width % height, row % width
        ky, kx, c = col//(kw*output_channels), col//output_channels % kw, col % output_channels
        uy, ux = y+padding[0]-ky*dilation[0], x+padding[1]-kx*dilation[1]
        oy, ox = uy//stride[0], ux//stride[1]
        valid = ((uy % stride[0] == 0) & (ux % stride[1] == 0) &
                 (oy >= 0) & (oy < oh) & (ox >= 0) & (ox < ow))
        indices = ((b*oh+oy)*ow+ox)*output_channels+c
    return np.where(valid, indices, -1)

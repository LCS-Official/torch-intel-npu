"""Conv2d index metadata only; no Torch tensors or activation values."""
import numpy as np


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

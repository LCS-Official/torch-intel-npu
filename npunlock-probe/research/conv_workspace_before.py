"""S34 convolution reference functions; installed only by the capacity benchmark.
Execute in torch_intel_npu globals after import; not a standalone backend.
"""

def _conv_streaming(original, params, shape):
    # Keep the validated small path. Stream when either full temporary could
    # exceed storage capacity; the fold upper bound is kernel area per pixel.
    return max(prod(shape), prod(original)*prod(params[0])) > MAX_TENSOR_ELEMENTS


def _conv_window_tiles(x, original, params, *, output_channels=None):
    (_, row_width), (rows, width) = conv_geometry(original, *params)
    if output_channels is not None:
        rows, width = original[0]*original[2]*original[3], prod(params[0])*output_channels
        row_width = original[3]
    # Each window buffer fits one compute tile, or one wide row if needed.
    batch = max(1, MAX_ELEMENTS//width)
    if batch >= row_width:
        batch = batch//row_width*row_width
    for start in range(0, rows, batch):
        count = min(batch, rows-start)
        indices = conv_window_indices(original, *params, start, count, output_channels=output_channels)
        yield start, count, _gather_window(x, indices, row_width)


def _conv_forward(x, weight, bias, spatial, params, shape):
    matrix = _materialize(weight).view(weight.shape[0], shape[1]).t()
    if _conv_streaming(tuple(x.shape), params, shape):
        # Compute in FP32 throughout. Share the same DMA carriers with backward
        # instead of caching a second complete set of FP16 gather/slice graphs.
        source = _materialize(_cast(_materialize(x), torch.float32).permute(0, 2, 3, 1))
        matrix = _cast(_materialize(matrix), torch.float32)
        bias = _cast(bias, torch.float32) if bias is not None else None
        tiles = []
        for _, _, window in _conv_window_tiles(source, tuple(x.shape), params):
            tiles.append(_matmul(window, matrix) if bias is None else _addmm(bias, window, matrix))
        result, columns = _concat(tiles, (shape[0], weight.shape[0]), torch.float32), None
    else:
        columns = _unfold(x, params, shape)
        result = _matmul(columns, matrix) if bias is None else _addmm(bias, columns, matrix)
    output = _materialize(result.view(x.shape[0], *spatial, weight.shape[0]).permute(0, 3, 1, 2))
    return _cast(output, x.dtype), columns


def _conv_stream_backward(x, weight, g, params, needs):
    """Recompute bounded windows; avoid a full im2col or input-gradient fold."""
    original = tuple(x.shape)
    channels = weight.shape[0]
    gx = gw = None
    if needs[0]:
        # Transposed geometry gathers the relevant output gradients for each
        # input pixel. Contract against (ky,kx,out_channel,in_channel) weights.
        w = _materialize(_cast(_materialize(weight), torch.float32).permute(2, 3, 0, 1)).view(-1, original[1])
        tiles = [_matmul(window, w) for _, _, window in
                 _conv_window_tiles(g, original, params, output_channels=channels)]
        nhwc = _concat(tiles, (original[0], original[2], original[3], original[1]), torch.float32)
        gx = _cast(_materialize(nhwc.permute(0, 3, 1, 2)), x.dtype)
    if needs[1]:
        source = _materialize(_cast(_materialize(x), torch.float32).permute(0, 2, 3, 1))
        # A binary reduction tree bounds live parameter-gradient buffers, even
        # when a stacked list of all partial gradients would exceed 1M items.
        levels = []
        for start, count, window in _conv_window_tiles(source, original, params):
            grad = _slice(g, start*channels, count*channels).view(count, channels)
            partial = _matmul(grad.t(), _cast(window, torch.float32))
            level = 0
            while level < len(levels) and levels[level] is not None:
                partial = _execute('add', levels[level], partial)
                levels[level] = None
                level += 1
            if level == len(levels):
                levels.append(partial)
            else:
                levels[level] = partial
        partials = [value for value in reversed(levels) if value is not None]
        total = partials[0]
        for value in partials[1:]:
            total = _execute('add', total, value)
        gw = _cast(total, weight.dtype).view(weight.shape)
    return gx, gw

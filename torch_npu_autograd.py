"""First-order autograd for the bounded intel_npu backend; no CPU payload fallback."""
from math import prod
import torch
from torch.autograd.function import once_differentiable
import torch_intel_npu as npu


def _fresh_output(value):
    # Only newly allocated results: discard internal reshape views without copying data.
    return value.detach()


def _sum_to(gradient, tensor):
    return _sum_to_shape(gradient,tuple(tensor.shape),tensor.dtype)


def _sum_to_shape(gradient, shape, dtype):
    padding = gradient.ndim - len(shape)
    axes = tuple(a for a in range(gradient.ndim) if a < padding or shape[a-padding] == 1 and gradient.shape[a] != 1)
    if axes:
        gradient = npu._reduce(gradient, axes, True, dtype=torch.float32, kind='sum')
    return npu._cast(gradient.view(shape), dtype)


class Dropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,p,train):
        result,mask=npu._native_dropout(x,p,train)
        ctx.save_for_backward(mask)
        ctx.scale=1. if train is False else 0. if p == 1 else 1./(1.-p)
        ctx.mark_non_differentiable(mask)
        return result,mask

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient,mask_gradient):
        return npu._native_dropout_backward(gradient,ctx.saved_tensors[0],ctx.scale),None,None


class Triangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,diagonal,upper):
        ctx.diagonal,ctx.upper=diagonal,upper
        return npu._triangle(x,diagonal,upper=upper)

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        return npu._triangle(gradient,ctx.diagonal,upper=ctx.upper),None,None


class Binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, kind, alpha):
        tensors = [v for v in (x, y) if isinstance(v, torch.Tensor)]
        if any(v.device.type != 'intel_npu' for v in tensors):
            raise ValueError('Training scalar operands must be Python scalars or NPU tensors')
        ctx.save_for_backward(*tensors)
        ctx.arguments = (isinstance(x, torch.Tensor), x if not isinstance(x, torch.Tensor) else None,
                         isinstance(y, torch.Tensor), y if not isinstance(y, torch.Tensor) else None, kind, alpha)
        return _fresh_output(npu._binary_op(x, y, kind=kind, alpha=alpha))

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        ix, sx, iy, sy, kind, alpha = ctx.arguments
        saved = iter(ctx.saved_tensors)
        x, y = next(saved) if ix else sx, next(saved) if iy else sy
        with npu._lock:
            g = npu._cast(npu._materialize(gradient), torch.float32)
            values = [npu._cast(npu._materialize(v), torch.float32) if isinstance(v, torch.Tensor) else v for v in (x, y)]
            gx = gy = None
            if ix and ctx.needs_input_grad[0]:
                gx = g if kind in ('add', 'sub') else npu._binary_op(g, values[1], kind='mul' if kind == 'mul' else 'div')
                gx = _sum_to(gx, x)
            if iy and ctx.needs_input_grad[1]:
                if kind == 'add':
                    gy = g
                elif kind == 'sub':
                    gy = npu._execute('neg', g)
                elif kind == 'mul':
                    gy = npu._binary_op(g, values[0], kind='mul')
                else:
                    # Match Torch's division order: y*y can overflow/underflow
                    # even when both the quotient and its derivative are finite.
                    scale = npu._execute('neg', npu._binary_op(g, values[1], kind='div'))
                    quotient = npu._binary_op(values[0], values[1], kind='div')
                    gy = npu._binary_op(scale, quotient, kind='mul')
                if kind in ('add', 'sub') and alpha != 1:
                    gy = npu._binary_op(gy, alpha, kind='mul')
                gy = _sum_to(gy, y)
        return gx, gy, None, None


class Extrema(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, kind):
        result = npu._extreme(x, y, kind=kind)
        ctx.save_for_backward(x, y)
        ctx.kind = kind
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, y = ctx.saved_tensors
        with npu._lock:
            g = npu._cast(npu._materialize(gradient), torch.float32)
            equal = npu._compare(x, y, kind='eq')
            half = npu._cast(npu._binary_op(g, .5, kind='mul'), gradient.dtype)
            results = []
            for needed, tensor, opposite in zip(ctx.needs_input_grad[:2], (x, y),
                                                ('lt','gt') if ctx.kind == 'maximum' else ('gt','lt')):
                if needed:
                    selected = npu._where(equal, half, npu._where(npu._compare(x, y, kind=opposite), 0., g))
                    results.append(_sum_to(selected, tensor))
                else:
                    results.append(None)
            return *results, None


class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, minimum, maximum, kind):
        result = npu._clamp(x, minimum, maximum)
        ctx.tensor_bounds = any(isinstance(v, torch.Tensor) for v in (minimum, maximum))
        ctx.scalars = tuple(None if isinstance(v, torch.Tensor) else v for v in (minimum, maximum))
        ctx.save_for_backward(x, minimum if isinstance(minimum, torch.Tensor) else None,
                              maximum if isinstance(maximum, torch.Tensor) else None)
        ctx.kind = kind
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, minimum, maximum = ctx.saved_tensors
        minimum = ctx.scalars[0] if minimum is None else minimum
        maximum = ctx.scalars[1] if maximum is None else maximum
        with npu._lock:
            g = npu._cast(npu._materialize(gradient), torch.float32)
            conditions = []
            for bound, lower in ((minimum, True), (maximum, False)):
                if bound is not None:
                    operation = ('lt' if lower else 'gt') if ctx.tensor_bounds else (
                        ('le' if lower else 'ge') if ctx.kind == 'hardtanh' else ('gt' if lower else 'lt'))
                    conditions.append(npu._compare(x, bound, kind=operation))
            mask = conditions[0]
            for condition in conditions[1:]:
                mask = npu._logical(mask, condition, kind='or' if ctx.tensor_bounds or ctx.kind == 'hardtanh' else 'and')
            if ctx.tensor_bounds or ctx.kind == 'hardtanh':
                mask = npu._logical(mask, kind='not')
            selected_gradient = g
            if ctx.tensor_bounds:
                half = npu._cast(npu._binary_op(g, .5, kind='mul'), gradient.dtype)
                ordered = npu._compare(minimum, maximum, kind='lt')
                tied = npu._logical(npu._compare(x, minimum, kind='eq'), npu._compare(x, maximum, kind='eq'), kind='or')
                tied = npu._logical(tied, ordered, kind='and')
                selected_gradient = npu._where(tied, half, g)
            gx = _sum_to(npu._where(mask, selected_gradient, 0.), x) if ctx.needs_input_grad[0] else None
            low = high = None
            if ctx.needs_input_grad[1]:
                selected = npu._where(npu._compare(x, minimum, kind='lt'), g,
                                      npu._where(npu._logical(npu._compare(x, minimum, kind='eq'), ordered, kind='and'),
                                                 half, 0.))
                low = _sum_to(npu._where(npu._compare(minimum, maximum, kind='le'), selected, 0.), minimum)
            if ctx.needs_input_grad[2]:
                selected = npu._where(npu._compare(x, maximum, kind='gt'), g,
                                      npu._where(npu._logical(npu._compare(x, maximum, kind='eq'), ordered, kind='and'),
                                                 half, 0.))
                selected = npu._where(npu._compare(minimum, maximum, kind='le'), selected,
                                      npu._where(npu._compare(minimum, maximum, kind='gt'), g, 0.))
                high = _sum_to(selected, maximum)
            return gx, low, high, None


class ExtremeReduction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,dim,keepdim,kind):
        result = npu._extreme_reduce(x,dim,keepdim,kind=kind)
        ctx.axes = npu.reduction_geometry(tuple(x.shape),dim,keepdim)[0]
        ctx.save_for_backward(x,result)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        x,result = ctx.saved_tensors
        with npu._lock:
            shape = tuple(1 if a in ctx.axes else n for a,n in enumerate(x.shape))
            mask = npu._compare(x,result.view(shape),kind='eq')
            count = npu._reduce(npu._mask_cast(mask,torch.float32),ctx.axes,True,kind='sum')
            scale = npu._cast(npu._binary_op(npu._cast(npu._materialize(gradient),torch.float32).view(shape),count,kind='div'),x.dtype)
            return npu._binary_op(scale,npu._mask_cast(mask,x.dtype),kind='mul'),None,None,None


class VectorNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,ord,dim,keepdim,dtype):
        result = npu._vector_norm(x,ord,dim,keepdim,dtype=dtype)
        ctx.axes = npu.reduction_geometry(tuple(x.shape),dim,keepdim)[0]
        ctx.ord = ord
        ctx.save_for_backward(x,result)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        if ctx.ord == 0: return None,None,None,None,None
        x,result = ctx.saved_tensors
        with npu._lock:
            shape = tuple(1 if a in ctx.axes else n for a,n in enumerate(x.shape))
            value = npu._cast(npu._materialize(x),result.dtype)
            norm = result.view(shape); g = npu._materialize(gradient).view(shape)
            if ctx.ord == 1:
                gx = npu._binary_op(npu._execute('sign',value),g,kind='mul')
            elif ctx.ord == 2:
                zero = npu._compare(norm,0.,kind='eq')
                ratio = npu._binary_op(value,npu._where(zero,1.,norm),kind='div')
                gx = npu._where(zero,0.,npu._binary_op(ratio,g,kind='mul'))
            else:
                mask = npu._compare(npu._execute('abs',value),norm,kind='eq')
                count = npu._reduce(npu._mask_cast(mask,torch.float32),ctx.axes,True,kind='sum')
                count = npu._where(npu._compare(count,0.,kind='eq'),1.,count)
                scale = npu._cast(npu._binary_op(npu._cast(g,torch.float32),count,kind='div'),result.dtype)
                gx = npu._binary_op(npu._where(mask,scale,0.),npu._execute('sign',value),kind='mul')
            return npu._cast(gx,x.dtype),None,None,None,None


class Reduction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, keepdim, dtype, kind):
        result = npu._reduce(x, dim, keepdim, dtype=dtype, kind=kind)
        axes = tuple(range(x.ndim or 1)) if dim is None or isinstance(dim, (list, tuple)) and not dim else (
            tuple(dim) if isinstance(dim, (list, tuple)) else (dim,))
        ctx.axes = tuple(a % (x.ndim or 1) for a in axes)
        ctx.shape, ctx.dtype, ctx.keepdim, ctx.kind = tuple(x.shape), x.dtype, keepdim, kind
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        with npu._lock:
            shape = tuple(1 if a in ctx.axes else d for a, d in enumerate(ctx.shape))
            # Mean's division is rounded in the output/gradient dtype before
            # autograd converts the returned gradient to the input dtype.
            gradient = npu._materialize(gradient).view(shape)
            gradient = npu._broadcast(gradient, ctx.shape)
            if ctx.kind == 'mean':
                gradient = npu._binary_op(gradient, prod(ctx.shape[a] for a in ctx.axes) if ctx.shape else 1, kind='div')
            return npu._cast(gradient, ctx.dtype), None, None, None, None


class Matrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return npu._matmul(x, y)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, y = ctx.saved_tensors
        with npu._lock:
            g = npu._cast(npu._materialize(gradient), torch.float32)
            left, right = (npu._cast(npu._materialize(t), torch.float32) for t in (x, y))
            gx = npu._cast(npu._matmul(g, right.t()), x.dtype) if ctx.needs_input_grad[0] else None
            gy = npu._cast(npu._matmul(left.t(), g), y.dtype) if ctx.needs_input_grad[1] else None
        return gx, gy


class Clone(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, memory_format):
        return _fresh_output(npu._clone(x, memory_format=memory_format))

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        with npu._lock:
            npu.synchronize()
            return npu._materialize(gradient), None


class Where(torch.autograd.Function):
    @staticmethod
    def forward(ctx, condition, x, y):
        result = npu._where(condition,x,y)
        ctx.save_for_backward(condition)
        ctx.operands = tuple((tuple(v.shape),v.dtype) if isinstance(v,torch.Tensor) else None for v in (x,y))
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        condition, = ctx.saved_tensors
        with npu._lock:
            g = npu._materialize(gradient)
            result = []
            for index,metadata in enumerate(ctx.operands):
                if ctx.needs_input_grad[index+1]:
                    selected = npu._where(condition,g,0.) if index == 0 else npu._where(condition,0.,g)
                    shape,dtype = metadata
                    # Torch reduces in the output dtype before converting to the input dtype.
                    reduced = _sum_to_shape(selected,shape,selected.dtype)
                    result.append(npu._cast(reduced,dtype))
                else:
                    result.append(None)
        return None,*result


class Concatenate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dim, stack, *tensors):
        _, ctx.dim, _ = npu._cat_spec(tensors, dim, stack=stack)
        ctx.shapes, ctx.stack = tuple(tuple(x.shape) for x in tensors), stack
        ctx.dtypes=tuple(x.dtype for x in tensors)
        return npu._stack(tensors, dim) if stack else npu._cat(tensors, dim)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        with npu._lock:
            g = npu._materialize(gradient)
            result, start = [], 0
            for needed, shape, dtype in zip(ctx.needs_input_grad[2:], ctx.shapes, ctx.dtypes):
                count = 1 if ctx.stack else 0 if shape==(0,) else shape[ctx.dim]
                if needed:
                    sliced = list(g.shape)
                    sliced[ctx.dim] = count
                    view = g.as_strided(sliced, g.stride(), storage_offset=start*g.stride(ctx.dim))
                    result.append(npu._cast(npu._materialize(view).view(shape),dtype))
                else:
                    result.append(None)
                start += count
        return None, None, *result


class Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, kind):
        ctx.save_for_backward(x, y)
        return npu._vector_product(x, y, kind=kind) if kind is not None else npu._matmul_nd(x, y)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, y = ctx.saved_tensors
        with npu._lock:
            g = npu._cast(npu._materialize(gradient), torch.float32)
            a, b = npu._matmul_promote(x, y)
            left, right = (npu._cast(npu._materialize(t), torch.float32) for t in (a, b))
            batch = tuple(torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
            g = g.view(batch+(a.shape[-2], b.shape[-1]))
            gx = _sum_to(npu._matmul_nd(g, right.transpose(-2, -1)), a).view(x.shape) if ctx.needs_input_grad[0] else None
            gy = _sum_to(npu._matmul_nd(left.transpose(-2, -1), g), b).view(y.shape) if ctx.needs_input_grad[1] else None
        return gx, gy, None


class IndexSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, indices):
        result = npu._index_select(x, dim, indices)
        ctx.save_for_backward(indices)
        ctx.shape, ctx.dim = tuple(x.shape), dim % x.ndim
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        return npu._index_backward(gradient, indices, ctx.shape, ctx.dim), None, None


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, index, sparse_grad):
        result = npu._gather(x, dim, index, sparse_grad=sparse_grad)
        ctx.save_for_backward(x, index)
        ctx.dim = dim
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, index = ctx.saved_tensors
        with npu._lock:
            initial = torch.zeros(tuple(x.shape), dtype=torch.float32, device='intel_npu')
            result = npu._scatter_add(initial, ctx.dim, index, npu._cast(npu._materialize(gradient), torch.float32))
            return npu._cast(result, x.dtype), None, None, None


class ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, index, src):
        result = npu._scatter_add(x, dim, index, src)
        ctx.save_for_backward(index)
        ctx.dim = dim
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        index, = ctx.saved_tensors
        with npu._lock:
            g = npu._materialize(gradient)
            return (g if ctx.needs_input_grad[0] else None, None, None,
                    npu._gather(g, ctx.dim, index) if ctx.needs_input_grad[3] else None)


class Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, indices, padding_idx, scale_grad_by_freq, sparse):
        result = npu._embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)
        ctx.save_for_backward(indices)
        ctx.shape, ctx.padding_idx = tuple(weight.shape), padding_idx
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        with npu._lock:
            g = npu._materialize(gradient).view(-1, ctx.shape[1])
            # CPU Embedding rounds each half update; index_select accumulates in FP32.
            result = npu._index_backward(g, indices.view(-1), ctx.shape, 0, ctx.padding_idx, round_half=True)
        return result, None, None, None, None


class SDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        result, probabilities, left, right, used, dropout_mask = npu._sdpa_forward(
            query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
        ctx.save_for_backward(query, key, value, attn_mask, probabilities, left, right, used, dropout_mask)
        ctx.gqa = enable_gqa
        ctx.empty_scores = probabilities is None
        ctx.empty_mask_grad = bool(query.shape[-2] and key.shape[-2] and value.shape[-1] and
                                   (not enable_gqa or key.shape[-3]))
        ctx.dropout_scale = 0. if dropout_p == 1 else 1./(1.-dropout_p)
        factor = (query.shape[-1] or 1)**-.5 if scale is None else scale
        ctx.root, ctx.negative = abs(factor)**.5, factor < 0
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        query, key, value, mask, probabilities, left, right, used, dropout_mask = ctx.saved_tensors
        with npu._lock:
            if ctx.empty_scores:
                return tuple(torch.zeros(tuple(t.shape),dtype=t.dtype,device=t.device)
                             if need and (i<3 or ctx.empty_mask_grad) else None
                             for i,(t,need) in enumerate(zip((query,key,value,mask),ctx.needs_input_grad[:4])))+(None,)*4
            g = npu._cast(npu._materialize(gradient), torch.float32)
            gq = gk = gv = gm = None
            def reduce_kv(gradient,original):
                if ctx.gqa:gradient = npu._sdpa_sum_heads(gradient,original.shape[-3])
                return _sum_to(gradient,original)
            if any(ctx.needs_input_grad[i] for i in (0, 1, 3)):
                v = npu._sdpa_repeat_heads(value,query.shape[-3]) if ctx.gqa else value
                v = npu._cast(npu._materialize(v), torch.float32)
                gp = npu._matmul_nd(g, v.transpose(-2, -1))
                if dropout_mask is not None:
                    gp = npu._native_dropout_backward(gp,dropout_mask,ctx.dropout_scale)
                ds = npu._softmax_backward(gp, probabilities, -1, torch.float32)
                if ctx.needs_input_grad[0]:
                    gq = _sum_to(npu._binary_op(npu._matmul_nd(ds, right), -ctx.root if ctx.negative else ctx.root, kind='mul'), query)
                if ctx.needs_input_grad[1]:
                    gk = reduce_kv(npu._binary_op(npu._matmul_nd(ds.transpose(-2, -1), left), ctx.root, kind='mul'), key)
                if ctx.needs_input_grad[3]:
                    gm = _sum_to(ds, mask)
            if ctx.needs_input_grad[2]:
                gv = reduce_kv(npu._matmul_nd(used.transpose(-2, -1), g), value)
        return gq, gk, gv, gm, None, None, None, None


class BatchedAddmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,bias,x,y,beta,alpha):
        ctx.save_for_backward(bias,x,y);ctx.beta,ctx.alpha=beta,alpha
        return npu._baddbmm(bias,x,y,beta=beta,alpha=alpha)

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        bias,x,y=ctx.saved_tensors
        with npu._lock:
            g=npu._cast(npu._materialize(gradient),torch.float32)
            gb=_sum_to(npu._binary_op(g,ctx.beta,kind='mul'),bias) if ctx.needs_input_grad[0] else None
            scaled=npu._binary_op(g,ctx.alpha,kind='mul') if ctx.alpha!=1 else g
            gx=_sum_to(npu._bmm(scaled,npu._cast(npu._materialize(y),torch.float32).transpose(1,2)),x) if ctx.needs_input_grad[1] else None
            gy=_sum_to(npu._bmm(npu._cast(npu._materialize(x),torch.float32).transpose(1,2),scaled),y) if ctx.needs_input_grad[2] else None
        return gb,gx,gy,None,None


class BatchedMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return npu._bmm(x, y)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, y = ctx.saved_tensors
        with npu._lock:
            g = npu._cast(npu._materialize(gradient), torch.float32)
            left, right = (npu._cast(npu._materialize(t), torch.float32) for t in (x, y))
            gx = npu._cast(npu._bmm(g, right.transpose(1, 2)), x.dtype) if ctx.needs_input_grad[0] else None
            gy = npu._cast(npu._bmm(left.transpose(1, 2), g), y.dtype) if ctx.needs_input_grad[1] else None
        return gx, gy


class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return _fresh_output(npu._linear(x, weight, bias))

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, weight = ctx.saved_tensors
        with npu._lock:
            g = npu._cast(npu._materialize(gradient), torch.float32).view(-1, weight.shape[0])
            matrix = npu._cast(npu._materialize(x), torch.float32).view(-1, weight.shape[1])
            w = npu._cast(npu._materialize(weight), torch.float32)
            gx = npu._cast(npu._matmul(g, w), x.dtype).view(x.shape) if ctx.needs_input_grad[0] else None
            gw = npu._cast(npu._matmul(g.t(), matrix), weight.dtype) if ctx.needs_input_grad[1] else None
            gb = npu._reduce(g, 0, dtype=ctx.bias_dtype, kind='sum') if ctx.needs_input_grad[2] else None
        return gx, gw, gb


class Addmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, x, y, beta, alpha):
        ctx.save_for_backward(bias, x, y)
        ctx.beta, ctx.alpha = beta, alpha
        return npu._addmm(bias, x, y, beta=beta, alpha=alpha)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        bias, x, y = ctx.saved_tensors
        with npu._lock:
            g = npu._cast(npu._materialize(gradient), torch.float32)
            gb = _sum_to(npu._binary_op(g, ctx.beta, kind='mul'), bias) if ctx.needs_input_grad[0] else None
            if ctx.alpha != 1:
                g = npu._binary_op(g, ctx.alpha, kind='mul')
            left, right = (npu._cast(npu._materialize(t), torch.float32) for t in (x, y))
            gx = npu._cast(npu._matmul(g, right.t()), x.dtype) if ctx.needs_input_grad[1] else None
            gy = npu._cast(npu._matmul(left.t(), g), y.dtype) if ctx.needs_input_grad[2] else None
        return gb, gx, gy, None, None


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
        with npu._lock:
            spatial, params, shape = npu._conv_spec(x, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
            results, saved = [], []
            for a, w, b in npu._conv_groups(x, weight, bias, groups):
                result, columns = npu._conv_forward(a, w, b, spatial, params, shape)
                results.append(result)
                if columns is not None:
                    saved.append(columns)
            ctx.streaming, ctx.groups = not saved, groups
            ctx.save_for_backward(x, weight, *saved)
            ctx.params, ctx.bias_dtype = params, bias.dtype if bias is not None else None
            return _fresh_output(results[0] if groups == 1 else npu._cat(results, 1))

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, weight, *saved = ctx.saved_tensors
        with npu._lock:
            pieces = [[], [], []]
            for group, (a, w, _) in enumerate(npu._conv_groups(x, weight, None, ctx.groups)):
                gradient_part = gradient if ctx.groups == 1 else gradient.narrow(1, group*w.shape[0], w.shape[0])
                g = npu._cast(npu._materialize(gradient_part), torch.float32)
                g = npu._materialize(g.permute(0, 2, 3, 1)).view(-1, w.shape[0])
                gx = gw = gb = None
                if ctx.streaming:
                    gx, gw = npu._conv_stream_backward(a, w, g, ctx.params, ctx.needs_input_grad)
                elif ctx.needs_input_grad[0]:
                    matrix = npu._cast(npu._materialize(w), torch.float32).view(w.shape[0], -1)
                    patches = npu._matmul(g, matrix)
                    folded = npu._fold(patches, tuple(a.shape), ctx.params)
                    gx = npu._cast(npu._reduce(folded, 1, kind='sum'), a.dtype).view(a.shape)
                if ctx.needs_input_grad[1] and not ctx.streaming:
                    gw = npu._matmul(g.t(), npu._cast(saved[group], torch.float32))
                    gw = npu._cast(gw, w.dtype).view(w.shape)
                if ctx.needs_input_grad[2]:
                    gb = npu._reduce(g, 0, dtype=ctx.bias_dtype, kind='sum')
                for outputs, value in zip(pieces, (gx, gw, gb)):
                    if value is not None:
                        outputs.append(value)
            gx, gw, gb = (None if not values else values[0] if ctx.groups == 1 else npu._cat(values, axis)
                          for values, axis in zip(pieces, (1, 0, 0)))
        return gx, gw, gb, None, None, None, None, None, None


class TopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,k,dim,largest,sorted):
        ctx.set_materialize_grads(False)
        values,indices=npu._topk(x,k,dim,largest,sorted)
        ctx.save_for_backward(indices);ctx.shape=tuple(x.shape);ctx.axis=dim%(x.ndim or 1);ctx.k=k
        ctx.mark_non_differentiable(indices)
        return _fresh_output(values),indices

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient,unused):
        indices,=ctx.saved_tensors
        with npu._lock:
            shape=ctx.shape or (1,);order=tuple(a for a in range(len(shape)) if a!=ctx.axis)+(ctx.axis,)
            g=npu._materialize(gradient.permute(order) if ctx.shape else gradient.view(1))
            ids=npu._materialize(indices.permute(order) if ctx.shape else indices.view(1))
            result=npu._selection_backward_planes(g,ids,tuple(shape[a] for a in order),ctx.k if ctx.shape else 1,shape[ctx.axis],accumulate=False)
            result=npu._materialize(result.permute(tuple(order.index(a) for a in range(len(shape)))))
            return result.view(ctx.shape),None,None,None,None


class MaxPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,kernel,stride,padding,dilation,ceil):
        ctx.set_materialize_grads(False)
        values,indices=npu._max_pool2d(x,kernel,stride,padding,dilation,ceil)
        ctx.save_for_backward(x,indices);ctx.params=(kernel,stride,padding,dilation,ceil)
        ctx.mark_non_differentiable(indices)
        return _fresh_output(values),indices

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient,unused):
        x,indices=ctx.saved_tensors
        return npu._max_pool2d_backward(gradient,x,*ctx.params,indices,known_windows=True),None,None,None,None,None


class ReluInplace(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        npu._relu_inplace(x,bump_version=False);ctx.mark_dirty(x);ctx.save_for_backward(x)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        return npu._execute('relu_grad',ctx.saved_tensors[0],npu._materialize(gradient))


def relu_inplace(x):
    with npu._lock:
        with torch.no_grad():npu._validate(x,dense=True)
        if _active(x):
            if x.is_leaf:raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation')
            if x._base is not None:raise ValueError('NPU differentiable relu_ requires a non-view result; clone the view first')
            return ReluInplace.apply(x)
        return npu._relu_inplace(x)


class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _fresh_output(npu._execute('relu', x))

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, = ctx.saved_tensors
        return npu._execute('relu_grad', x, npu._materialize(gradient))


class Unary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kind):
        result = npu._reciprocal(x) if kind == 'reciprocal' else npu._execute(kind, x)
        ctx.kind, ctx.dtype, ctx.shape = kind, x.dtype, tuple(x.shape)
        if kind in ('abs', 'reciprocal'):
            ctx.save_for_backward(result if kind == 'reciprocal' else x)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        with npu._lock:
            if ctx.kind == 'sign':
                return torch.zeros(ctx.shape, dtype=ctx.dtype, device='intel_npu'), None
            g = npu._cast(npu._materialize(gradient), torch.float32)
            if ctx.kind == 'neg':
                result = npu._execute('neg', g)
            elif ctx.kind == 'abs':
                value, = ctx.saved_tensors
                sign = npu._cast(npu._execute('sign', value), torch.float32)
                result = npu._execute('mul', g, sign)
            else:
                value, = ctx.saved_tensors
                square = npu._cast(npu._execute('mul', value, value), torch.float32)
                result = npu._execute('mul', npu._execute('neg', g), square)
            return npu._cast(result, ctx.dtype), None


class UnaryMath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kind):
        result = npu._math_unary(x, kind)
        ctx.kind, ctx.dtype = kind, x.dtype
        ctx.save_for_backward(x if kind in ('log', 'silu') else result)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        with npu._lock:
            saved, = ctx.saved_tensors
            value = npu._cast(npu._materialize(saved), torch.float32)
            g = npu._cast(npu._materialize(gradient), torch.float32)
            binary = npu._binary_op
            if ctx.kind == 'exp':
                result = binary(g, value, kind='mul')
            elif ctx.kind == 'log':
                result = binary(g, value, kind='div')
            elif ctx.kind == 'sqrt':
                result = binary(g, binary(value, 2., kind='mul'), kind='div')
            elif ctx.kind == 'rsqrt':
                # Match Torch's result.pow(3) dtype/overflow boundary before scaling.
                cube = npu._cast(binary(binary(value, value, kind='mul'), value, kind='mul'), ctx.dtype)
                scaled = binary(npu._materialize(gradient), -.5, kind='mul')
                return binary(scaled, cube, kind='mul'), None
            elif ctx.kind == 'tanh':
                result = binary(g, binary(1., binary(value, value, kind='mul'), kind='sub'), kind='mul')
            else:
                sigmoid = npu._math_unary(value, 'sigmoid') if ctx.kind == 'silu' else value
                tail = binary(1., sigmoid, kind='sub')
                if ctx.kind == 'silu':
                    tail = binary(1., binary(value, tail, kind='mul'), kind='add')
                result = binary(binary(g, sigmoid, kind='mul'), tail, kind='mul')
            return npu._cast(result, ctx.dtype), None


class MSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, reduction):
        ctx.save_for_backward(x, target)
        ctx.reduction = reduction
        return npu._mse_loss(x, target, reduction)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, target = ctx.saved_tensors
        with npu._lock:
            gx = _sum_to(npu._mse_loss_backward(gradient,x,target,ctx.reduction),x) if ctx.needs_input_grad[0] else None
            gy = _sum_to(npu._mse_loss_backward(gradient,target,x,ctx.reduction),target) if ctx.needs_input_grad[1] else None
        return gx, gy, None


class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, half_to_float):
        result = npu._log_softmax(x, dim, half_to_float)
        ctx.save_for_backward(result)
        ctx.dim, ctx.dtype = dim, x.dtype
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        output, = ctx.saved_tensors
        return npu._log_softmax_backward(gradient, output, ctx.dim, ctx.dtype), None, None


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, half_to_float):
        result = npu._softmax(x, dim, half_to_float)
        ctx.save_for_backward(result)
        ctx.dim, ctx.dtype = dim, x.dtype
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        output, = ctx.saved_tensors
        return npu._softmax_backward(gradient, output, ctx.dim, ctx.dtype), None, None


class NLL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, weight, reduction, ignore_index):
        result, total, packet = npu._nll_forward(x, target, weight, reduction, ignore_index)
        ctx.save_for_backward(x, target, weight, total, packet)
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        ctx.mark_non_differentiable(total)
        return result, total

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient, _):
        x, target, weight, total, packet = ctx.saved_tensors
        return npu._nll_backward(gradient,x,packet,total,ctx.reduction,weight,
                                 target=target,ignore_index=ctx.ignore_index), None, None, None, None


class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, training, momentum, eps):
        result, normalized, inverse, centered = npu._batch_norm(x, weight, bias, running_mean, running_var, training, momentum, eps,save_centered=True)
        ctx.save_for_backward(x, weight, normalized, inverse, centered)
        ctx.training, ctx.bias_dtype = training, bias.dtype if bias is not None else None
        return _fresh_output(result)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, weight, normalized, inverse, centered = ctx.saved_tensors
        with npu._lock:
            g = npu._cast(npu._materialize(gradient), torch.float32)
            shape, axes = (1, x.shape[1], 1, 1), (0, 2, 3)
            gx = gw = gb = None
            if ctx.needs_input_grad[1]:
                # CPU forms a centered dot product before multiplying invstd.
                # Moving invstd into each product can cross a Half rounding
                # midpoint, amplified when repeated InstanceNorm weights sum.
                dot = npu._reduce(npu._execute('mul',g,centered),axes,kind='sum')
                gw = npu._cast(npu._execute('mul',dot,inverse),weight.dtype)
            if ctx.needs_input_grad[2]:
                gb = npu._reduce(g, axes, dtype=ctx.bias_dtype, kind='sum')
            if ctx.needs_input_grad[0]:
                scaled = g if weight is None else npu._binary_op(g, npu._cast(weight, torch.float32).view(shape), kind='mul')
                if ctx.training:
                    average = npu._reduce(scaled, axes, True, kind='mean')
                    projection = npu._reduce(npu._execute('mul', scaled, normalized), axes, True, kind='mean')
                    scaled = npu._binary_op(scaled, average, kind='sub')
                    scaled = npu._binary_op(scaled, npu._binary_op(normalized, projection, kind='mul'), kind='sub')
                gx = npu._cast(npu._binary_op(scaled, inverse.view(shape), kind='mul'), x.dtype)
        return gx, gw, gb, None, None, None, None, None


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shape, weight, bias, eps):
        output, mean, inverse = npu._layer_norm(x, shape, weight, bias, eps)
        ctx.save_for_backward(x, mean, inverse, weight, bias)
        ctx.shape = shape
        ctx.mark_non_differentiable(mean, inverse)
        return output, mean, inverse

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient, _mean_gradient, _inverse_gradient):
        x, mean, inverse, weight, bias = ctx.saved_tensors
        gx, gw, gb = npu._layer_norm_backward(gradient, x, ctx.shape, mean, inverse, weight, bias,
                                             (ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]))
        return gx, None, gw, gb, None


class Gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, approximate):
        ctx.save_for_backward(x)
        ctx.approximate = approximate
        return npu._gelu(x, approximate=approximate)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, = ctx.saved_tensors
        return npu._gelu_backward(gradient, x, approximate=ctx.approximate), None


class GlobalAvgPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_size):
        ctx.shape, ctx.dtype = tuple(x.shape), x.dtype
        return npu._adaptive_avg_pool2d(x, output_size)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        with npu._lock:
            g = npu._cast(npu._materialize(gradient), torch.float32)
            g = npu._binary_op(g, ctx.shape[2]*ctx.shape[3], kind='div')
            return npu._cast(npu._broadcast(g, ctx.shape), ctx.dtype), None


_library = torch.library.Library('aten', 'IMPL', 'AutogradPrivateUse1')


def _active(*args):
    return torch.is_grad_enabled() and any(isinstance(x, torch.Tensor) and x.requires_grad for x in args)


_library.impl('topk',lambda x,k,dim=-1,largest=True,sorted=True:
              TopK.apply(x,k,dim,largest,sorted) if _active(x) else npu._topk(x,k,dim,largest,sorted))
def max_pool2d(x,kernel_size,stride=(),padding=0,dilation=1,ceil_mode=False):
    return (MaxPool2d.apply(x,kernel_size,stride,padding,dilation,ceil_mode) if _active(x) else
            npu._max_pool2d(x,kernel_size,stride,padding,dilation,ceil_mode))
_library.impl('max_pool2d_with_indices',max_pool2d)
_library.impl('max_pool2d',lambda x,kernel_size,stride=(),padding=0,dilation=1,ceil_mode=False:
              max_pool2d(x,kernel_size,stride,padding,dilation,ceil_mode)[0])
_library.impl('max_pool1d',npu._max_pool1d)
_library.impl('max_pool1d_with_indices',lambda x,kernel_size,stride=(),padding=(0,),dilation=(1,),ceil_mode=False:
              npu._max_pool1d(x,kernel_size,stride,padding,dilation,ceil_mode,return_indices=True))
_library.impl('relu_',relu_inplace)
_library.impl('native_dropout',lambda x,p,train:
              Dropout.apply(x,p,train) if _active(x) else npu._native_dropout(x,p,train))
_library.impl('dropout',lambda x,p,train:
              x if train is False or p == 0 else Dropout.apply(x,p,train)[0] if _active(x) else npu._dropout(x,p,train))
_library.impl('multinomial', npu._multinomial)
_library.impl('multinomial.out', npu._multinomial)


for _kind in ('add', 'sub', 'mul', 'div'):
    def binary(x, y, *, alpha=1, kind=_kind):
        return Binary.apply(x, y, kind, alpha) if _active(x, y) else npu._binary_op(x, y, kind=kind, alpha=alpha)
    for _overload in ('Tensor', 'Scalar'):
        _library.impl(f'{_kind}.{_overload}', binary)
for _overload in ('Tensor', 'Scalar'):
    def reverse_subtract(x, y, alpha=1):
        return Binary.apply(y, x, 'sub', alpha) if _active(x, y) else npu._rsub(x, y, alpha=alpha)
    _library.impl('rsub.'+_overload, reverse_subtract)
for _kind in ('sum', 'mean'):
    def reduction(x, dim=None, keepdim=False, *, dtype=None, kind=_kind):
        return Reduction.apply(x, dim, keepdim, dtype, kind) if _active(x) else npu._reduce(x, dim, keepdim, dtype=dtype, kind=kind)
    for _overload in (('sum', 'sum.dim_IntList') if _kind == 'sum' else ('mean', 'mean.dim')):
        _library.impl(_overload, reduction)
_library.impl('mm', lambda x, y: Matrix.apply(x, y) if _active(x, y) else npu._matmul(x, y))
_library.impl('clone', lambda x, memory_format=None:
              Clone.apply(x, memory_format) if _active(x) else npu._clone(x, memory_format=memory_format))
for _overload in ('self','ScalarOther','ScalarSelf','Scalar'):
    _library.impl('where.'+_overload,lambda condition,x,y:
                  Where.apply(condition,x,y) if _active(x,y) else npu._where(condition,x,y))


def masked_fill(x,mask,value):
    with torch.no_grad():
        npu._masked_fill_spec(x,mask,value)
    if _active(x,value):
        constant = value if isinstance(value,torch.Tensor) else npu._scalar(value,x.dtype)
        return Where.apply(mask,constant,x)
    return npu._masked_fill(x,mask,value)


for _overload in ('Scalar','Tensor'):
    _library.impl('masked_fill.'+_overload,masked_fill)
for _name in ('cat', 'stack'):
    def concatenate(tensors, dim=0, name=_name):
        return (Concatenate.apply(dim, name == 'stack', *tensors) if _active(*tensors) else
                getattr(npu, '_'+name)(tensors, dim))
    _library.impl(_name, concatenate)
_library.impl('matmul', lambda x, y: Matmul.apply(x, y, None) if _active(x, y) else npu._matmul_nd(x, y))
_library.impl('index_select', lambda x, dim, indices:
              IndexSelect.apply(x, dim, indices) if _active(x) else npu._index_select(x, dim, indices))
for _name in ('isnan','isinf','isfinite'):
    _library.impl(_name,lambda x,kind=_name:npu._float_predicate(x,kind=kind))
for _kind in ('max','min'):
    def extreme_reduction(x,dim=(),keepdim=False,kind=_kind):
        return ExtremeReduction.apply(x,dim,keepdim,kind) if _active(x) else npu._extreme_reduce(x,dim,keepdim,kind=kind)
    _library.impl('a'+_kind,extreme_reduction)
    _library.impl('arg'+_kind,lambda x,dim=None,keepdim=False,kind=_kind:npu._arg_reduce(x,dim,keepdim,kind=kind))
def vector_norm(x,ord=2,dim=None,keepdim=False,*,dtype=None):
    return VectorNorm.apply(x,ord,dim,keepdim,dtype) if _active(x) else npu._vector_norm(x,ord,dim,keepdim,dtype=dtype)
_library.impl('linalg_vector_norm',vector_norm)
def legacy_norm(x,p=2,dim=None,keepdim=False,*,dtype=None):
    return vector_norm(x,2 if p is None else p,dim,keepdim,dtype=dtype)
for _name in ('Scalar','ScalarOpt_dim','ScalarOpt_dtype','ScalarOpt_dim_dtype'):
    _library.impl('norm.'+_name,legacy_norm)
_library.impl('gather', lambda x, dim, index, sparse_grad=False:
              Gather.apply(x, dim, index, sparse_grad) if _active(x) else npu._gather(x, dim, index, sparse_grad=sparse_grad))
_library.impl('scatter_add', lambda x, dim, index, src:
              ScatterAdd.apply(x, dim, index, src) if _active(x, src) else npu._scatter_add(x, dim, index, src))
_library.impl('take_along_dim', npu._take_along_dim)
_library.impl('embedding', lambda w, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False:
              Embedding.apply(w, indices, padding_idx, scale_grad_by_freq, sparse) if _active(w) else
              npu._embedding(w, indices, padding_idx, scale_grad_by_freq, sparse))
for _kind in ('dot', 'mv'):
    def vector(x, y, kind=_kind):
        return Matmul.apply(x, y, kind) if _active(x, y) else npu._vector_product(x, y, kind=kind)
    _library.impl(_kind, vector)
_library.impl('bmm', lambda x, y: BatchedMatrix.apply(x, y) if _active(x, y) else npu._bmm(x, y))
for _upper,_name in ((True,'triu'),(False,'tril')):
    def triangle(x,diagonal=0,upper=_upper):
        return Triangle.apply(x,diagonal,upper) if _active(x) else npu._triangle(x,diagonal,upper=upper)
    _library.impl(_name,triangle)
_library.impl('baddbmm',lambda bias,x,y,beta=1,alpha=1:
              BatchedAddmm.apply(bias,x,y,beta,alpha) if _active(bias,x,y) else npu._baddbmm(bias,x,y,beta=beta,alpha=alpha))
_library.impl('_native_multi_head_attention',npu._native_multi_head_attention)
_library.impl('_transformer_encoder_layer_fwd',npu._transformer_encoder_layer_fwd)
def scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,*,scale=None,enable_gqa=False):
    if not torch.backends.cuda.math_sdp_enabled():
        return npu._sdpa(q,k,v,attn_mask,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa)
    return (SDPA.apply(q,k,v,attn_mask,dropout_p,is_causal,scale,enable_gqa) if _active(q,k,v,attn_mask) else
            npu._sdpa(q,k,v,attn_mask,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa))

_library.impl('scaled_dot_product_attention',scaled_dot_product_attention)
_library.impl('linear', lambda x, weight, bias=None: Linear.apply(x, weight, bias) if _active(x, weight, bias) else npu._linear(x, weight, bias))
_library.impl('addmm', lambda b, x, y, beta=1, alpha=1: Addmm.apply(b, x, y, beta, alpha) if _active(b, x, y) else npu._addmm(b, x, y, beta=beta, alpha=alpha))
def convolution(x, w, b, s, p, d, t, o, g):
    if x.ndim==5:
        from torch_npu_vision_extended import convolution3d
        return convolution3d(x,w,b,s,p,d,t,o,g)
    one_dimensional = x.ndim == 3
    args = npu._conv1d_arguments(x, w, b, s, p, d, t, o, g) if one_dimensional else (x, w, b, s, p, d, t, o, g)
    if t:
        from torch_npu_vision import conv_transpose
        a,weight,bias,stride,padding,dilation,_,output_padding,groups=args
        result=conv_transpose(a,weight,bias,stride,padding,dilation,output_padding,groups)
        return (result.squeeze(2).clone() if _active(x,w,b) else result.squeeze(2)) if one_dimensional else result
    result = Conv2d.apply(*args) if _active(x, w, b) else npu._convolution(*args)
    return (result.squeeze(2).clone() if _active(x,w,b) else result.squeeze(2)) if one_dimensional else result


_library.impl('convolution', convolution)
_library.impl('relu', lambda x: Relu.apply(x) if _active(x) else npu._execute('relu', x))
for _kind in ('neg', 'abs', 'sign', 'reciprocal'):
    def unary(x, kind=_kind):
        return Unary.apply(x, kind) if _active(x) else (
            npu._reciprocal(x) if kind == 'reciprocal' else npu._execute(kind, x))
    _library.impl(_kind, unary)
for _kind in npu._MATH_UNARY:
    def unary_math(x, kind=_kind):
        return UnaryMath.apply(x, kind) if _active(x) else npu._math_unary(x, kind)
    _library.impl(_kind, unary_math)
for _kind in ('maximum', 'minimum'):
    def extrema(x, y, kind=_kind):
        return Extrema.apply(x, y, kind) if _active(x, y) else npu._extreme(x, y, kind=kind)
    _library.impl(_kind, extrema)
for _name in ('clamp', 'clip'):
    def clipping(x, min=None, max=None):
        if (min is None) != (max is None) and isinstance(max if min is None else min, torch.Tensor):
            bound = max if min is None else min
            kind = 'minimum' if min is None else 'maximum'
            return Extrema.apply(x, bound, kind) if _active(x, bound) else npu._extreme(x, bound, kind=kind)
        return Clamp.apply(x, min, max, 'clamp') if _active(x, min, max) else npu._clamp(x, min, max)
    _library.impl(_name, clipping)
    _library.impl(_name+'.Tensor', clipping)
for _name in ('clamp_min', 'clamp_max'):
    def bound_clipping(x, bound, lower=_name == 'clamp_min'):
        if isinstance(bound, torch.Tensor):
            return (Extrema.apply(x, bound, 'maximum' if lower else 'minimum') if _active(x, bound) else
                    npu._extreme(x, bound, kind='maximum' if lower else 'minimum'))
        return clipping(x, bound if lower else None, None if lower else bound)
    _library.impl(_name, bound_clipping)
    _library.impl(_name+'.Tensor', bound_clipping)
_library.impl('hardtanh', lambda x, min_val=-1., max_val=1.:
              Clamp.apply(x, min_val, max_val, 'hardtanh') if _active(x) else npu._clamp(x, min_val, max_val))
_library.impl('mse_loss', lambda x, target, reduction=1: MSE.apply(x, target, reduction) if _active(x, target) else npu._mse_loss(x, target, reduction))
_library.impl('_log_softmax', lambda x, dim, half_to_float=False:
              LogSoftmax.apply(x, dim, half_to_float) if _active(x) else npu._log_softmax(x, dim, half_to_float))
_library.impl('_softmax', lambda x, dim, half_to_float=False:
              Softmax.apply(x, dim, half_to_float) if _active(x) else npu._softmax(x, dim, half_to_float))


def nll(x, target, weight=None, reduction=1, ignore_index=-100):
    return NLL.apply(x, target, weight, reduction, ignore_index) if _active(x) else npu._nll_forward(x, target, weight, reduction, ignore_index)[:2]


_library.impl('nll_loss_forward', nll)


def nll2d_forward(x,target,weight,reduction,ignore_index):
    if x.ndim!=4:raise ValueError('nll_loss2d requires rank-4 NCHW input')
    shape=target.shape
    values,labels=npu._classification_input(x,target,weight,reduction,ignore_index)
    result,total=nll(values,labels,weight,reduction,ignore_index)
    return (result.view(shape) if reduction==0 else result),total


_library.impl('nll_loss2d_forward',nll2d_forward)


def nll_nd(x, target, weight=None, reduction=1, ignore_index=-100):
    shape = target.shape
    x, target = npu._classification_input(x, target, weight, reduction, ignore_index)
    result = nll(x, target, weight, reduction, ignore_index)[0]
    return result.view(shape) if reduction == 0 else result


for _name in ('nll_loss', 'nll_loss_nd'):
    _library.impl(_name, nll_nd)


_library.impl('cross_entropy_loss', npu._cross_entropy)
def layer_norm(x, shape, weight=None, bias=None, eps=1e-5):
    return (LayerNorm.apply(x, shape, weight, bias, eps) if _active(x, weight, bias) else
            npu._layer_norm(x, shape, weight, bias, eps))


_library.impl('layer_norm', lambda x, shape, weight=None, bias=None, eps=1e-5, cudnn_enable=True:
              layer_norm(x, shape, weight, bias, eps)[0])
_library.impl('native_layer_norm', layer_norm)
_library.impl('rms_norm', npu._rms_norm)
_library.impl('gelu', lambda x, approximate='none':
              Gelu.apply(x, approximate) if _active(x) else npu._gelu(x, approximate=approximate))
def batch_norm(x,w,b,rm,rv,training,momentum,eps,cudnn):
    from torch_npu_vision_extended import batch_norm as implementation
    return implementation(x,w,b,rm,rv,training,momentum,eps,cudnn)
_library.impl('batch_norm',batch_norm)
for _name in ('adaptive_avg_pool2d', '_adaptive_avg_pool2d'):
    _library.impl(_name, lambda x, output_size: GlobalAvgPool.apply(x, output_size) if _active(x) and x.ndim == 4 and tuple(output_size) == (1,1) else npu._adaptive_avg_pool2d(x, output_size))
_library.impl('adaptive_avg_pool1d', npu._adaptive_avg_pool1d)
_library.impl('avg_pool1d', npu._avg_pool1d)
_library.impl('avg_pool2d', npu._avg_pool2d)

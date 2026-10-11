"""First-order autograd for the bounded intel_npu backend; no CPU payload fallback."""
from math import prod
import torch
from torch.autograd.function import once_differentiable
import torch_intel_npu as npu


def _sum_to(gradient, tensor):
    shape = tuple(tensor.shape)
    padding = gradient.ndim - tensor.ndim
    axes = tuple(a for a in range(gradient.ndim) if a < padding or tensor.shape[a-padding] == 1 and gradient.shape[a] != 1)
    if axes:
        gradient = npu._reduce(gradient, axes, True, dtype=torch.float32, kind='sum')
    return npu._cast(gradient.view(shape), tensor.dtype)


class Binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, kind):
        tensors = [v for v in (x, y) if isinstance(v, torch.Tensor)]
        if any(v.device.type != 'intel_npu' for v in tensors):
            raise ValueError('Training scalar operands must be Python scalars or NPU tensors')
        ctx.save_for_backward(*tensors)
        ctx.arguments = (isinstance(x, torch.Tensor), x if not isinstance(x, torch.Tensor) else None,
                         isinstance(y, torch.Tensor), y if not isinstance(y, torch.Tensor) else None, kind)
        return npu._binary_op(x, y, kind=kind)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        ix, sx, iy, sy, kind = ctx.arguments
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
                gy = _sum_to(gy, y)
        return gx, gy, None


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


class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return npu._linear(x, weight, bias)

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
            result, columns = npu._conv_forward(x, weight, bias, spatial, params, shape)
            ctx.streaming = columns is None
            if ctx.streaming:
                ctx.save_for_backward(x, weight)
            else:
                ctx.save_for_backward(x, weight, columns)
            ctx.params, ctx.bias_dtype = params, bias.dtype if bias is not None else None
            return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, weight, *saved = ctx.saved_tensors
        with npu._lock:
            g = npu._cast(npu._materialize(gradient), torch.float32)
            g = npu._materialize(g.permute(0, 2, 3, 1)).view(-1, weight.shape[0])
            gx = gw = gb = None
            if ctx.streaming:
                gx, gw = npu._conv_stream_backward(x, weight, g, ctx.params, ctx.needs_input_grad)
            elif ctx.needs_input_grad[0]:
                w = npu._cast(npu._materialize(weight), torch.float32).view(weight.shape[0], -1)
                patches = npu._matmul(g, w)
                folded = npu._fold(patches, tuple(x.shape), ctx.params)
                gx = npu._cast(npu._reduce(folded, 1, kind='sum'), x.dtype).view(x.shape)
            if ctx.needs_input_grad[1] and not ctx.streaming:
                gw = npu._matmul(g.t(), npu._cast(saved[0], torch.float32))
                gw = npu._cast(gw, weight.dtype).view(weight.shape)
            if ctx.needs_input_grad[2]:
                gb = npu._reduce(g, 0, dtype=ctx.bias_dtype, kind='sum')
        return gx, gw, gb, None, None, None, None, None, None


class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return npu._execute('relu', x)

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
            difference = npu._binary_op(npu._cast(npu._materialize(x), torch.float32),
                                        npu._cast(npu._materialize(target), torch.float32), kind='sub')
            scale = 2./x.numel() if ctx.reduction == 1 else 2.
            g = npu._binary_op(difference, scale, kind='mul')
            g = npu._binary_op(g, npu._cast(npu._materialize(gradient), torch.float32), kind='mul')
            gx = npu._cast(g, x.dtype) if ctx.needs_input_grad[0] else None
            gy = npu._cast(npu._execute('neg', g), target.dtype) if ctx.needs_input_grad[1] else None
        return gx, gy, None


class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, training, momentum, eps):
        result, normalized, inverse = npu._batch_norm(x, weight, bias, running_mean, running_var, training, momentum, eps)
        ctx.save_for_backward(x, weight, normalized, inverse)
        ctx.training, ctx.bias_dtype = training, bias.dtype if bias is not None else None
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x, weight, normalized, inverse = ctx.saved_tensors
        with npu._lock:
            g = npu._cast(npu._materialize(gradient), torch.float32)
            shape, axes = (1, x.shape[1], 1, 1), (0, 2, 3)
            gx = gw = gb = None
            if ctx.needs_input_grad[1]:
                gw = npu._reduce(npu._execute('mul', g, normalized), axes, dtype=weight.dtype, kind='sum')
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


for _kind in ('add', 'sub', 'mul', 'div'):
    def binary(x, y, *, alpha=1, kind=_kind):
        if type(alpha) not in (int, float) or alpha != 1:
            raise NotImplementedError('Differentiable NPU add/sub require alpha=1')
        return Binary.apply(x, y, kind) if _active(x, y) else npu._binary_op(x, y, kind=kind)
    for _overload in ('Tensor', 'Scalar'):
        _library.impl(f'{_kind}.{_overload}', binary)
for _kind in ('sum', 'mean'):
    def reduction(x, dim=None, keepdim=False, *, dtype=None, kind=_kind):
        return Reduction.apply(x, dim, keepdim, dtype, kind) if _active(x) else npu._reduce(x, dim, keepdim, dtype=dtype, kind=kind)
    for _overload in (('sum', 'sum.dim_IntList') if _kind == 'sum' else ('mean', 'mean.dim')):
        _library.impl(_overload, reduction)
for _name in ('mm', 'matmul'):
    _library.impl(_name, lambda x, y: Matrix.apply(x, y) if _active(x, y) else npu._matmul(x, y))
_library.impl('linear', lambda x, weight, bias=None: Linear.apply(x, weight, bias) if _active(x, weight, bias) else npu._linear(x, weight, bias))
_library.impl('addmm', lambda b, x, y, beta=1, alpha=1: Addmm.apply(b, x, y, beta, alpha) if _active(b, x, y) else npu._addmm(b, x, y, beta=beta, alpha=alpha))
_library.impl('convolution', lambda x, w, b, s, p, d, t, o, g: Conv2d.apply(x, w, b, s, p, d, t, o, g) if _active(x, w, b) else npu._convolution(x, w, b, s, p, d, t, o, g))
_library.impl('relu', lambda x: Relu.apply(x) if _active(x) else npu._execute('relu', x))
for _kind in ('neg', 'abs', 'sign', 'reciprocal'):
    def unary(x, kind=_kind):
        return Unary.apply(x, kind) if _active(x) else (
            npu._reciprocal(x) if kind == 'reciprocal' else npu._execute(kind, x))
    _library.impl(_kind, unary)
_library.impl('mse_loss', lambda x, target, reduction=1: MSE.apply(x, target, reduction) if _active(x, target) else npu._mse_loss(x, target, reduction))
_library.impl('batch_norm', lambda x, w, b, rm, rv, training, momentum, eps, cudnn:
              BatchNorm.apply(x, w, b, rm, rv, training, momentum, eps) if _active(x, w, b) else
              npu._batch_norm(x, w, b, rm, rv, training, momentum, eps)[0])
for _name in ('adaptive_avg_pool2d', '_adaptive_avg_pool2d'):
    _library.impl(_name, lambda x, output_size: GlobalAvgPool.apply(x, output_size) if _active(x) else npu._adaptive_avg_pool2d(x, output_size))

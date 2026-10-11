"""FP16/FP32 Torch arithmetic on Lunar Lake NPU with CPU tensor staging.

No forward CPU arithmetic fallback. FP32 MatMul uses SHAVE multiply/pairwise add;
the driver's ordinary MatMul can silently lower to FP16 on the DPU.
"""
import hashlib
import json
import math
import operator
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import npunlock as npu
from npunlock.npu4000 import MAX_ELEMENTS, compile_unary, compile_binary, compile_cast
import torch
from torch._dynamo import register_backend

_KERNELS = Path(__file__).resolve().parent / "npunlock/tests/fixtures/npu4000/precision"
# ponytail: copied execution is serialized until shared sessions are reliable.
_lock = Lock()
_native = None
_stats = {"successful_npu_calls": 0, "compiled_graphs": 0, "carrier_compilations": 0,
          "last_device": None, "last_shape": None, "last_blob_sha256": None, "operations": {}}
_DTYPES = {torch.float16: ("f16", np.float16), torch.float32: ("f32", np.float32)}


def _validate(x, *, max_elements=MAX_ELEMENTS):
    if x.device.type != "cpu" or x.layout != torch.strided:
        raise ValueError("Requires a strided CPU tensor for NPU host staging")
    if x.dtype not in _DTYPES or x.ndim > 4 or not 0 < x.numel() <= max_elements:
        raise ValueError(f"Requires FP16/FP32, rank 0-4, and 1-{max_elements} elements")


@lru_cache(maxsize=32)
def _kernel(kind, dtype):
    manifest = json.loads((_KERNELS / "manifest.json").read_text())
    if hashlib.sha256((_KERNELS.parent / "arithmetic.c").read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Kernel source changed; rebuild precision artifacts")
    stem = f"{kind}-{dtype}"
    data = (_KERNELS / f"{stem}.elf").read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["kernels"][stem]["elf_sha256"]:
        raise ValueError("Kernel ELF differs from its build manifest")
    return data


@lru_cache(maxsize=128)
def _program(kind, dtype, shape, axes=()):
    global _native
    if _native is None:
        _native = npu.NativeLibraries()
    kwargs = dict(libraries=_native, timeout_ms=20_000)
    if kind in ("add1", "neg", "abs", "sign"):
        program = compile_unary(shape, _kernel(kind, dtype), dtype=dtype, **kwargs)
    elif kind in ("add", "sub", "mul", "div"):
        program = compile_binary(shape, _kernel(kind, dtype), dtype=dtype, **kwargs)
    elif kind == "cast":
        program = compile_cast(shape, dtype=dtype, kernel_elf=_kernel("cast", dtype), **kwargs)
    else:
        raise NotImplementedError(kind)
    _stats["carrier_compilations"] += 1
    return program


def execution_info():
    with _lock:
        return {**_stats, "operations": dict(_stats["operations"]),
            "last_device": dict(_stats["last_device"]) if _stats["last_device"] else None,
            "tensor_storage": "cpu", "arithmetic_device": "Intel NPU4000",
            "fp32_path": "SHAVE FP32; no implicit FP16 lowering",
            "cached_carriers": _program.cache_info().currsize}


def _execute(kind, dtype, shape, arrays, axes=()):
    with _lock:
        program = _program(kind, dtype, shape, axes)
        result = _native.infer_graph(program.graph_blob,
            [npu.InferenceInput(t.name, np.ascontiguousarray(a).tobytes()) for t, a in zip(program.graph.inputs, arrays)],
            timeout_ms=10_000)
        if (result.vendor_id, result.device_id) != (0x8086, 0x643E) or len(result.outputs) != 1:
            raise RuntimeError("Unexpected NPU device/output contract")
        output, spec = result.outputs[0], program.graph.outputs[0]
        npdtype = {"f16": np.float16, "f32": np.float32}[spec.dtype]
        if (output.dtype != spec.dtype or output.shape != spec.shape or
            len(output.data) != math.prod(spec.shape) * np.dtype(npdtype).itemsize):
            raise RuntimeError("Unexpected NPU output shape/dtype/byte count")
        if kind == "cast" and dtype == "f32":
            # Storage unpacking only; IEEE rounding happened in the NPU kernel.
            array = np.frombuffer(output.data, dtype=np.uint32).astype(np.uint16).view(np.float16)
        else:
            array = np.frombuffer(output.data, dtype=npdtype).copy()
        y = torch.from_numpy(array.reshape(spec.shape))
        _stats["successful_npu_calls"] += 1
        key = f"{kind}.{dtype}"
        _stats["operations"][key] = _stats["operations"].get(key, 0) + 1
        _stats["last_device"] = {"vendor_id": hex(result.vendor_id), "device_id": hex(result.device_id),
            "driver_version": result.driver_version, "driver_index": result.driver_index, "device_index": result.device_index}
        _stats["last_shape"] = list(shape)
        _stats["last_blob_sha256"] = hashlib.sha256(program.graph_blob).hexdigest()
        return y


@torch.library.custom_op("npunlock_npu::unary", mutates_args=(), device_types="cpu")
def unary(x: torch.Tensor, kind: str) -> torch.Tensor:
    _validate(x)
    if kind not in ("add1", "neg", "abs", "sign"):
        raise NotImplementedError(kind)
    shape = tuple(x.shape) or (1,)
    return _execute(kind, _DTYPES[x.dtype][0], shape, [x.detach().numpy().reshape(shape)]).reshape(x.shape)


@unary.register_fake
def _fake_unary(x, kind):
    _validate(x)
    if kind not in ("add1", "neg", "abs", "sign"):
        raise NotImplementedError(kind)
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("npunlock_npu::binary", mutates_args=(), device_types="cpu")
def binary(x: torch.Tensor, y: torch.Tensor, kind: str) -> torch.Tensor:
    _validate(x)
    _validate(y)
    if kind not in ("add", "sub", "mul", "div") or x.shape != y.shape or x.dtype != y.dtype:
        raise ValueError("Custom binary kernel requires same-shape, same-dtype inputs")
    shape = tuple(x.shape) or (1,)
    return _execute(kind, _DTYPES[x.dtype][0], shape,
        [v.detach().numpy().reshape(shape) for v in (x, y)]).reshape(x.shape)


@binary.register_fake
def _fake_binary(x, y, kind):
    _validate(x)
    _validate(y)
    if kind not in ("add", "sub", "mul", "div") or x.shape != y.shape or x.dtype != y.dtype:
        raise ValueError("Custom binary kernel requires same-shape, same-dtype inputs")
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("npunlock_npu::cast", mutates_args=(), device_types="cpu")
def cast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    _validate(x)
    if dtype not in _DTYPES:
        raise ValueError("Only FP16/FP32 casts are supported")
    if x.dtype == dtype:
        return x.clone()
    shape = tuple(x.shape) or (1,)
    return _execute("cast", _DTYPES[x.dtype][0], shape, [x.detach().numpy().reshape(shape)]).reshape(x.shape)


@cast.register_fake
def _fake_cast(x, dtype):
    _validate(x)
    if dtype not in _DTYPES:
        raise ValueError("Only FP16/FP32 casts are supported")
    return torch.empty(x.shape, dtype=dtype, device=x.device)


@torch.library.custom_op("npunlock_npu::reduce", mutates_args=(), device_types="cpu")
def _reduce(x: torch.Tensor, kind: str, axes: list[int], keepdim: bool) -> torch.Tensor:
    _validate(x)
    if x.dtype != torch.float32 or kind not in ("sum", "mean"):
        raise ValueError("Reduction kernel requires FP32")
    remaining = tuple(i for i in range(x.ndim) if i not in axes)
    # Host work reorders/copies bytes; every sum and division runs on the NPU.
    shape = tuple(x.shape)
    rows = math.prod(shape[i] for i in remaining)
    count = math.prod(shape[i] for i in axes)
    values = x.detach().numpy().transpose(remaining + tuple(axes)).reshape(rows, count)
    if count == 1:
        y = _execute("add", "f32", (rows, 1), [values, np.zeros_like(values)])
    while count > 1:
        pairs = (count + 1) // 2
        left = values[:, ::2]
        right = np.zeros((rows, pairs), dtype=np.float32)
        right[:, :count // 2] = values[:, 1::2]
        y = _execute("add", "f32", (rows, pairs), [left, right])
        values = y.numpy()
        count = pairs
    # Torch reductions start with +0; retain that behavior for all-negative-zero inputs.
    y = _execute("add", "f32", (rows, 1), [y.numpy(), np.zeros((rows, 1), dtype=np.float32)])
    if kind == "mean":
        divisor = np.full((rows, 1), math.prod(shape[i] for i in axes), dtype=np.float32)
        y = _execute("div", "f32", (rows, 1), [y.numpy(), divisor])
    outshape = tuple(1 if i in axes else d for i, d in enumerate(x.shape)) if keepdim else tuple(
        d for i, d in enumerate(x.shape) if i not in axes)
    return y.reshape(outshape)


@_reduce.register_fake
def _fake_reduce(x, kind, axes, keepdim):
    _validate(x)
    if x.dtype != torch.float32 or kind not in ("sum", "mean"):
        raise ValueError("Reduction kernel requires FP32")
    shape = tuple(1 if i in axes else d for i, d in enumerate(x.shape)) if keepdim else tuple(
        d for i, d in enumerate(x.shape) if i not in axes)
    return torch.empty(shape, dtype=x.dtype, device=x.device)


def _matrix_contract(x, y):
    _validate(x)
    _validate(y)
    if x.ndim != 2 or y.ndim != 2 or x.dtype != y.dtype or x.shape[1] != y.shape[0]:
        raise ValueError("MatMul requires compatible 2D matrices of the same dtype")
    m, k, n = x.shape[0], x.shape[1], y.shape[1]
    # ponytail: materialized products; tile when M*K*N needs to exceed 65536.
    if m * k * n > MAX_ELEMENTS:
        raise ValueError("SHAVE MatMul currently supports M*K*N <= 65536")
    return m, k, n


@torch.library.custom_op("npunlock_npu::matmul", mutates_args=(), device_types="cpu")
def _matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k, n = _matrix_contract(x, y)
    if x.dtype != torch.float32:
        raise ValueError("Internal SHAVE MatMul requires FP32")
    a = np.broadcast_to(x.detach().numpy()[:, None, :], (m, n, k))
    b = np.broadcast_to(y.detach().numpy().T[None, :, :], (m, n, k))
    products = _execute("mul", "f32", (m, n, k), [a, b])
    return _reduce(products, "sum", [2], False).reshape(m, n)


@_matmul.register_fake
def _fake_matmul(x, y):
    m, _, n = _matrix_contract(x, y)
    if x.dtype != torch.float32:
        raise ValueError("Internal SHAVE MatMul requires FP32")
    return torch.empty((m, n), dtype=x.dtype, device=x.device)


def add1(x): return unary(x, "add1")
def neg(x): return unary(x, "neg")
def abs(x): return unary(x, "abs")
def sign(x): return unary(x, "sign")


def _binary_dtype(x, y):
    tensor = x if isinstance(x, torch.Tensor) else y
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("At least one operand must be a tensor")
    _validate(tensor)
    for value in (x, y):
        if isinstance(value, torch.Tensor):
            _validate(value)
    shape = torch.broadcast_shapes(*(v.shape if isinstance(v, torch.Tensor) else () for v in (x, y)))
    if len(shape) > 4 or not 0 < math.prod(shape) <= MAX_ELEMENTS:
        raise ValueError("Broadcast result exceeds supported shape/size")
    outdtype = torch.result_type(x, y)
    if outdtype not in _DTYPES:
        raise ValueError("Only FP16/FP32 results are supported")
    return outdtype


def _binary(x, y, kind):
    outdtype = _binary_dtype(x, y)
    # Match Torch's wrapped-scalar promotion, including FP16 scalar rounding.
    dtype = torch.float32 if outdtype == torch.float16 and kind in ("mul", "div") and (
        not isinstance(y, torch.Tensor) or y.ndim == 0 and y.dtype != torch.float16) else outdtype
    operands = []
    for value in (x, y):
        if isinstance(value, torch.Tensor):
            _validate(value)
            operands.append(cast(value, dtype) if value.dtype != dtype else value)
        elif type(value) in (int, float):
            operands.append(torch.tensor(value, dtype=dtype))
        else:
            raise TypeError("Operand must be a tensor or real numeric scalar")
    a, b = torch.broadcast_tensors(*operands)
    _validate(a)
    z = binary(a, b, kind)
    return cast(z, outdtype) if z.dtype != outdtype else z


def add(x, y, *, alpha=1):
    if alpha != 1:
        outdtype = _binary_dtype(x, y)
        alpha = torch.tensor(alpha, dtype=outdtype).item()
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=outdtype).item()
        x = cast(x, torch.float32) if isinstance(x, torch.Tensor) and x.dtype != torch.float32 else x
        y = cast(y, torch.float32) if isinstance(y, torch.Tensor) and y.dtype != torch.float32 else y
        y = mul(y, alpha) if isinstance(y, torch.Tensor) else y * alpha
        z = _binary(x, y, "add")
        return cast(z, outdtype) if z.dtype != outdtype else z
    return _binary(x, y, "add")


def sub(x, y, *, alpha=1):
    if alpha != 1:
        outdtype = _binary_dtype(x, y)
        alpha = torch.tensor(alpha, dtype=outdtype).item()
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=outdtype).item()
        x = cast(x, torch.float32) if isinstance(x, torch.Tensor) and x.dtype != torch.float32 else x
        y = cast(y, torch.float32) if isinstance(y, torch.Tensor) and y.dtype != torch.float32 else y
        y = mul(y, alpha) if isinstance(y, torch.Tensor) else y * alpha
        z = _binary(x, y, "sub")
        return cast(z, outdtype) if z.dtype != outdtype else z
    return _binary(x, y, "sub")


def mul(x, y): return _binary(x, y, "mul")


def div(x, y, *, rounding_mode=None):
    if rounding_mode is not None:
        raise NotImplementedError("NPU division supports rounding_mode=None")
    return _binary(x, y, "div")


def _axes(x, dim):
    values = tuple(range(x.ndim)) if dim is None else (dim,) if isinstance(dim, int) else tuple(dim)
    if not values:
        values = tuple(range(x.ndim))
    if x.ndim == 0:
        if any(i not in (0, -1) for i in values):
            raise IndexError("Scalar reduction dimension out of range")
        return ()
    if any(type(i) is not int or not -x.ndim <= i < x.ndim for i in values):
        raise IndexError("Reduction dimension out of range")
    values = tuple(i % x.ndim for i in values)
    if len(set(values)) != len(values):
        raise ValueError("Duplicate reduction dimensions")
    return tuple(sorted(values))


def _reduction(x, kind, dim, keepdim, dtype):
    _validate(x)
    if dtype is not None and dtype not in _DTYPES:
        raise ValueError("Only FP16/FP32 reductions are supported")
    axes = _axes(x, dim)
    if kind == "sum" and dtype is not None and x.dtype != dtype:
        x = cast(x, dtype)
    work = cast(x, torch.float32) if x.dtype != torch.float32 else x
    y = _reduce(work, kind, list(axes), keepdim)
    target = dtype or x.dtype
    return cast(y, target) if y.dtype != target else y


def sum(x, dim=None, keepdim=False, *, dtype=None): return _reduction(x, "sum", dim, keepdim, dtype)
def mean(x, dim=None, keepdim=False, *, dtype=None): return _reduction(x, "mean", dim, keepdim, dtype)


def matmul(x, y):
    _matrix_contract(x, y)
    dtype = x.dtype
    a, b = (cast(v, torch.float32) if v.dtype != torch.float32 else v for v in (x, y))
    z = _matmul(a, b)
    return cast(z, dtype) if z.dtype != dtype else z


def _setup_unary(ctx, inputs, output):
    x, ctx.kind = inputs
    ctx.save_for_backward(x)


def _backward_unary(ctx, grad):
    if ctx.kind == "add1": return grad, None
    if ctx.kind == "neg": return neg(grad), None
    if ctx.kind == "sign": return torch.zeros_like(grad), None
    return mul(grad, sign(ctx.saved_tensors[0])), None


unary.register_autograd(_backward_unary, setup_context=_setup_unary)


def _setup_binary(ctx, inputs, output):
    x, y, ctx.kind = inputs
    ctx.save_for_backward(x, y)


def _backward_binary(ctx, grad):
    x, y = ctx.saved_tensors
    if ctx.kind == "add": return grad, grad, None
    if ctx.kind == "sub": return grad, neg(grad), None
    if ctx.kind == "mul": return mul(grad, y), mul(grad, x), None
    gx = div(grad, y)
    return gx, neg(div(mul(gx, x), y)), None


binary.register_autograd(_backward_binary, setup_context=_setup_binary)


def _setup_cast(ctx, inputs, output):
    ctx.dtype = inputs[0].dtype


cast.register_autograd(lambda ctx, grad: (cast(grad, ctx.dtype), None), setup_context=_setup_cast)


def _setup_reduce(ctx, inputs, output):
    x, ctx.kind, axes, ctx.keepdim = inputs
    ctx.shape, ctx.axes = tuple(x.shape), tuple(axes)
    ctx.count = math.prod(x.shape[i] for i in axes)


def _backward_reduce(ctx, grad):
    if not ctx.keepdim:
        for axis in ctx.axes:
            grad = grad.unsqueeze(axis)
    grad = grad.expand(ctx.shape)
    if ctx.kind == "mean":
        grad = div(grad, ctx.count)
    return grad, None, None, None


_reduce.register_autograd(_backward_reduce, setup_context=_setup_reduce)


def _setup_matmul(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward_matmul(ctx, grad):
    x, y = ctx.saved_tensors
    return matmul(grad, y.T), matmul(x.T, grad)


_matmul.register_autograd(_backward_matmul, setup_context=_setup_matmul)

_FUNCTIONS = {
    operator.add: add, torch.add: add, torch.ops.aten.add.Tensor: add,
    operator.sub: sub, torch.sub: sub, torch.subtract: sub, torch.ops.aten.sub.Tensor: sub,
    operator.mul: mul, torch.mul: mul, torch.multiply: mul, torch.ops.aten.mul.Tensor: mul,
    operator.truediv: div, torch.div: div, torch.divide: div, torch.ops.aten.div.Tensor: div,
    operator.neg: neg, torch.neg: neg, torch.ops.aten.neg.default: neg,
    operator.abs: abs, torch.abs: abs, torch.ops.aten.abs.default: abs,
    torch.sign: sign, torch.ops.aten.sign.default: sign,
    torch.sum: sum, torch.mean: mean,
    operator.matmul: matmul, torch.matmul: matmul, torch.mm: matmul, torch.ops.aten.mm.default: matmul,
}
_METHODS = {"add": add, "sub": sub, "mul": mul, "div": div, "neg": neg, "abs": abs, "sign": sign,
            "sum": sum, "mean": mean, "matmul": matmul, "mm": matmul}


@register_backend(name="npunlock_npu")
def npu_backend(gm, example_inputs):
    """Lower a static supported FX graph; unknown operations fail without fallback."""
    error = "npunlock_npu supports FP16/FP32 arithmetic, casts, reductions and 2D MatMul; no CPU fallback"
    for value in example_inputs:
        if not isinstance(value, torch.Tensor):
            raise NotImplementedError(error)
        _validate(value)
    rewrites = []
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output", "get_attr"): continue
        if node.op == "call_function" and node.target in _FUNCTIONS:
            rewrites.append((node, _FUNCTIONS[node.target], node.args, node.kwargs))
        elif node.op == "call_method" and node.target in _METHODS:
            rewrites.append((node, _METHODS[node.target], node.args, node.kwargs))
        elif node.op == "call_method" and node.target in ("half", "float", "to"):
            dtype = {"half": torch.float16, "float": torch.float32}.get(node.target)
            if node.target == "to":
                if len(node.args) > 2 or set(node.kwargs) - {"dtype"}:
                    raise NotImplementedError(error)
                dtype = node.kwargs.get("dtype", node.args[1] if len(node.args) == 2 else None)
            if dtype not in _DTYPES:
                raise NotImplementedError(error)
            rewrites.append((node, torch.ops.npunlock_npu.cast.default, (node.args[0], dtype), {}))
        else:
            raise NotImplementedError(f"{error}: {node.op} {node.target}")
    if not rewrites: raise NotImplementedError(error)
    for node, function, args, kwargs in rewrites:
        node.op, node.target, node.args, node.kwargs = "call_function", function, args, kwargs
    gm.graph.lint()
    gm.recompile()
    with _lock: _stats["compiled_graphs"] += 1
    return gm.forward


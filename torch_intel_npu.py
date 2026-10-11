"""Minimal real PrivateUse1 device for the validated Lunar Lake NPU profile.

Import registers intel_npu:0. FP16/FP32 tensors own Level Zero shared storage;
ordinary ATen arithmetic reuses resident graph sessions without host staging.
"""
import atexit
import hashlib
import importlib.util
import json
import operator
from math import prod, isfinite
from functools import lru_cache
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import npunlock as npu
import torch
import torch_npu_minimal as bridge
from torch_npu_geometry import conv_geometry, conv_window_indices
from npunlock.npu4000 import (compile_halfcode_slice, compile_reduce, compile_broadcast, compile_permute,
                             compile_gather, compile_matrix_concat, compile_concat_many, compile_slice, compile_unary, compile_binary,
                             MAX_ELEMENTS, MAX_TENSOR_ELEMENTS)
from torch_npu_runtime import SESSION_DLL

_ROOT = Path(__file__).resolve().parent
_EXT = _ROOT / "npunlock-probe/torch-device/_torch_intel_npu.pyd"
_BUILD = _EXT.with_name("build.json")
if torch._C._get_privateuse1_backend_name() != "privateuseone":
    raise RuntimeError("PrivateUse1 is already reserved by another backend")
if not _EXT.is_file() or not _BUILD.is_file():
    raise FileNotFoundError("Run build_torch_npu_device.ps1 first")
_provenance = json.loads(_BUILD.read_text(encoding="utf-8-sig"))
for path, key in ((_EXT, "extension_sha256"), (SESSION_DLL, "session_sha256"),
                  (_EXT.with_name("backend.cpp"), "source_sha256")):
    if hashlib.sha256(path.read_bytes()).hexdigest() != _provenance[key]:
        raise RuntimeError(f"Stale NPU extension build: {path}; rebuild the extension")
if torch.__version__ != _provenance["torch_version"]:
    raise RuntimeError("PyTorch version changed; rebuild the NPU C++ extension")

_lock = RLock()
_native = npu.NativeLibraries(session_library=SESSION_DLL)
with bridge._lock:
    _bootstrap_program = bridge._program("add", "f32", (1,))
_root_session = _native.create_inference_session(_bootstrap_program.graph_blob, timeout_ms=10_000)
_info = _root_session._result
_driver_version = _info.driver_version
if (_info.device_vendor_id, _info.device_id) != (0x8086, 0x643E):
    _root_session.close()
    raise RuntimeError("This backend only supports the validated Intel 8086:643e NPU")

torch.utils.rename_privateuse1_backend("intel_npu")
_spec = importlib.util.spec_from_file_location("_torch_intel_npu", _EXT)
_C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_C)
if _C.max_tensor_elements != MAX_TENSOR_ELEMENTS:
    raise RuntimeError("NPU storage bounds differ; rebuild the C++ extension")
_C.configure(str(SESSION_DLL), _info.session)
_sessions = OrderedDict()
_scalars = OrderedDict()
_session_creations = 1
_MAX_SESSIONS = 512
_GATHER_TILE_ELEMENTS = 8192  # Conservative byte-Gather output bound on this driver.
_closed = False


def _check_device(device=None):
    if device is not None:
        device = torch.device(device) if not isinstance(device, int) else torch.device("intel_npu", device)
        if device.type != "intel_npu" or device.index not in (None, 0):
            raise ValueError("Only intel_npu:0 is available")


def synchronize(device=None):
    _check_device(device)
    with _lock:
        if _closed or _C.stats()["context_failed"]:
            raise RuntimeError("NPU backend is closed or failed")
        # Every submission already waits on its native completion fence.


def stats():
    with _lock:
        return {**_C.stats(), "session_creations": _session_creations,
                "cached_sessions": len(_sessions), "graph_cache_limit": _MAX_SESSIONS,
                "cached_scalar_buffers": len(_scalars), "closed": _closed,
                "max_tensor_elements": MAX_TENSOR_ELEMENTS, "max_compute_tile_elements": MAX_ELEMENTS,
                "device": "intel_npu:0", "hardware_device": "8086:643e",
                "driver_version": _driver_version, "torch_version": torch.__version__,
                "allocation_api": "zeMemAllocShared", "synchronization": "synchronous fence per graph or chain",
                "command_policy": "re-record current bindings; warmed chains share a list/fence; reset after completion",
                "allocation_cache_limit_bytes": 16 * 1024 * 1024, "allocation_cache_limit_blocks": 256,
                "copy_counter_scope": "explicit tensor copies/packing; excludes driver DMA and compiler"}


def empty_cache():
    """Release unused shared-memory blocks; live tensors and graphs stay valid."""
    with _lock:
        _C.empty_cache()


def shutdown():
    """Disable operations and close graphs. Live tensors retain their allocations."""
    global _closed
    with _lock:
        if not _closed:
            _C.shutdown()
            _scalars.clear()
            for session, _ in reversed(tuple(_sessions.values())):
                session.close()
            _sessions.clear()
            _root_session.close()
            _closed = True


def _unsupported(func, *args, **kwargs):
    raise NotImplementedError(f"intel_npu does not implement {func}; no CPU arithmetic fallback")


def _validate(tensor, *, dense=False):
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "intel_npu" or
            tensor.dtype not in bridge._DTYPES):
        raise ValueError("Requires intel_npu FP16/FP32 tensors")
    if tensor.requires_grad and torch.is_grad_enabled():
        raise ValueError("Requires an implemented NPU autograd operator, or no_grad/inference_mode")
    if tensor.storage_offset() != 0 or tensor.is_neg() or tensor.is_conj():
        raise ValueError("Requires contiguous, offset-zero NPU tensors")
    if tensor.ndim > 4 or not 0 < tensor.numel() <= MAX_TENSOR_ELEMENTS:
        raise ValueError("Requires rank 0-4 and 1-1048576 elements")
    if tensor.numel() * tensor.element_size() != tensor.untyped_storage().nbytes():
        raise ValueError("NPU binding must cover its entire allocation")
    if tensor.is_contiguous():
        return
    if dense:
        expected = 1
        for axis in sorted(range(tensor.ndim), key=lambda a: tensor.stride(a)):
            if tensor.shape[axis] != 1:
                if tensor.stride(axis) != expected:
                    raise ValueError("NPU supports complete dense permutations only")
                expected *= tensor.shape[axis]
    else:
        raise ValueError("Requires contiguous NPU tensors; call contiguous() on a dense permutation")


def _physical_view(x):
    axes = sorted(range(x.ndim), key=lambda a: x.stride(a), reverse=True)
    base = x.as_strided(tuple(x.shape[a] for a in axes), tuple(x.stride(a) for a in axes))
    return base, axes


def _materialize(x, out=None):
    _validate(x, dense=True)
    if x.is_contiguous():
        return _copy(out, x) if out is not None else x
    base, axes = _physical_view(x)
    order = tuple(axes.index(a) for a in range(x.ndim))
    return _run(("permute", bridge._DTYPES[x.dtype][0], tuple(base.shape), order),
                [base], tuple(x.shape), x.dtype, out=out)


def _resident_kernel(kind):
    directory = _ROOT / "npunlock/tests/fixtures/npu4000"
    manifest = json.loads((directory / "resident/manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Resident kernel source changed; run build_resident_kernels.py")
    data = (directory / f"resident/{kind}.elf").read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["kernels"][kind]["elf_sha256"]:
        raise RuntimeError("Resident kernel differs from its build manifest")
    return data


def _session(key):
    global _session_creations
    if key not in _sessions:
        # Destroy/reimport during execution reproducibly lost the device in tiled training.
        # Retain bounded graphs until shutdown; never silently evict a live driver graph.
        if len(_sessions) >= _MAX_SESSIONS:
            raise RuntimeError(f"NPU graph cache reached {_MAX_SESSIONS} entries; start a new process for new shapes")
        kind, dtype, shape, *extra = key
        if kind == "halfcode_slice":
            program = compile_halfcode_slice(shape[0], libraries=_native)
        elif kind == "broadcast":
            program = compile_broadcast(shape, extra[0], dtype=dtype, libraries=_native)
        elif kind == "permute":
            program = compile_permute(shape, extra[0], dtype=dtype, libraries=_native)
        elif kind == "slice":
            program = compile_slice(shape[0], *extra, dtype=dtype, libraries=_native)
        elif kind == "matrix_rows":
            import numpy as np
            start, count = extra
            indices = np.arange(start*shape[1], (start+count)*shape[1], dtype=np.int64).reshape(count, shape[1])
            program = compile_gather(prod(shape), indices, dtype=dtype, libraries=_native)
        elif kind == "matrix_concat":
            program = compile_matrix_concat(extra[0], shape[1], dtype=dtype, libraries=_native)
        elif kind == "concat_many":
            program = compile_concat_many(extra[0], dtype=dtype, libraries=_native)
        elif kind == "gather_tile":
            import numpy as np
            indices = np.frombuffer(extra[0], dtype=np.int64).reshape(1, -1)
            program = compile_gather(shape[0], indices, dtype=dtype, libraries=_native)
        elif kind in ("unfold", "fold"):
            original, kernel, stride, padding, dilation = (shape, *extra) if kind == "unfold" else extra
            _, indices, inverse = _conv_indices(original, kernel, stride, padding, dilation)
            program = compile_gather(prod(shape), indices if kind == "unfold" else inverse,
                                     dtype=dtype, libraries=_native)
        elif kind in ("unfold_tile", "fold_tile"):
            import numpy as np
            original, kernel, stride, padding, dilation, start, count, base = extra
            _, indices, inverse = _conv_indices(original, kernel, stride, padding, dilation)
            selected = (indices if kind == "unfold_tile" else inverse).reshape(-1)[start:start+count]
            program = compile_gather(shape[0], np.where(selected < 0, -1, selected-base).reshape(1, count),
                                     dtype=dtype, libraries=_native)
        elif kind in ("relu", "relu_grad", "rsqrt"):
            compile_fn = compile_binary if kind == "relu_grad" else compile_unary
            program = compile_fn(shape, _resident_kernel(f"{kind}-{dtype}"), dtype=dtype, libraries=_native)
        elif kind in ("sum", "mean"):
            program = compile_reduce(shape, extra[0], _resident_kernel(kind), kind=kind, libraries=_native)
        else:
            with bridge._lock:
                program = bridge._program(kind, dtype, shape)
        session = _native.create_inference_session(program.graph_blob, timeout_ms=10_000, parent=_root_session)
        _sessions[key] = session, program
        _session_creations += 1
    else:
        _sessions.move_to_end(key)
    return _sessions[key]


def _run(key, inputs, shape, dtype, out=None):
    if out is None:
        out = torch.empty(shape, dtype=dtype, device="intel_npu:0")
    _validate(out)
    if out.shape != shape or out.dtype != dtype:
        raise ValueError("NPU output has the wrong shape or dtype")
    if any(out.untyped_storage()._cdata == x.untyped_storage()._cdata for x in inputs):
        raise ValueError("NPU output may not alias an input")
    session, program = _session(key)
    try:
        _C.run(session._result.session, inputs, [t.name for t in program.graph.inputs],
               out, program.graph.outputs[0].name)
    except RuntimeError as error:
        error.add_note(f"NPU graph key: {key}")
        raise
    return out


def _prepare_many(key, inputs, specs, outs=None):
    """Internal binding preparation; outputs become readable after submission."""
    with _lock:
        synchronize()
        for value in inputs:
            _validate(value)
        if not specs or (outs is not None and len(outs) != len(specs)):
            raise ValueError("NPU multi-output requires one shape/dtype per output")
        if any(dtype not in bridge._DTYPES for shape, dtype in specs):
            raise ValueError("NPU multi-output requires FP16/FP32")
        session, program = _session(key)
        if len(specs) != len(program.graph.outputs):
            raise ValueError("NPU output count differs from the graph")
        outputs = tuple(torch.empty(shape, dtype=dtype, device="intel_npu:0") for shape, dtype in specs) if outs is None else tuple(outs)
        # Fresh allocations cannot alias live inputs or each other; explicit outs can.
        for output, (shape, dtype) in zip(outputs, specs):
            _check_out(output, tuple(shape), dtype, inputs if outs is not None else ())
        if outs is not None and len({output.untyped_storage()._cdata for output in outputs}) != len(outputs):
            raise ValueError("NPU outputs may not alias each other")
        return session, program, tuple(inputs), outputs


def _run_many(key, inputs, specs, outs=None):
    """Bind independent, complete output allocations in one synchronous call."""
    with _lock:
        session, program, inputs, outputs = _prepare_many(key, inputs, specs, outs)
        try:
            _C.run_many(session._result.session, inputs, [t.name for t in program.graph.inputs],
                        outputs, [t.name for t in program.graph.outputs])
        except RuntimeError as error:
            error.add_note(f"NPU graph key: {key}")
            raise
        return outputs


def _run_chain(plan, group_size=4):
    """Internal synchronous submission; the plan retains sessions and Tensors."""
    with _lock:
        synchronize()
        if not plan or type(group_size) is not int or group_size not in (1, 2, 4):
            raise ValueError("NPU chain requires a nonempty plan and group_size 1, 2 or 4")
        _C.run_chain([session._result.session for session, program, inputs, outputs in plan],
            [inputs for session, program, inputs, outputs in plan],
            [[t.name for t in program.graph.inputs] for session, program, inputs, outputs in plan],
            [outputs for session, program, inputs, outputs in plan],
            [[t.name for t in program.graph.outputs] for session, program, inputs, outputs in plan], group_size)


def _execute(kind, x, y=None, *, out=None):
    with _lock:
        synchronize()
        _validate(x, dense=True)
        inputs = [x]
        if y is not None:
            _validate(y, dense=True)
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError("NPU arithmetic requires identical shapes and dtypes; broadcasting is not implemented")
            inputs.append(y)
        _check_out(out, tuple(x.shape), x.dtype, inputs)
        inputs = [value if value.is_contiguous() else _materialize(value) for value in inputs]
        if x.numel() > MAX_ELEMENTS:
            return _map_tiles(lambda *tiles: _execute(kind, *tiles), inputs, tuple(x.shape), x.dtype, out)
        key = kind, bridge._DTYPES[x.dtype][0], tuple(x.shape) or (1,)
        return _run(key, inputs, tuple(x.shape), x.dtype, out)


def _check_out(out, shape, dtype, inputs):
    if out is not None:
        _validate(out)
        if tuple(out.shape) != shape or out.dtype != dtype:
            raise ValueError("NPU out requires the exact result shape and dtype")
        if any(isinstance(x, torch.Tensor) and x.device.type == "intel_npu" and
               out.untyped_storage()._cdata == x.untyped_storage()._cdata for x in inputs):
            raise ValueError("NPU output may not alias an input")


def _slice(x, start, count):
    """Device DMA tile extraction; CPU only supplies shape/offset metadata."""
    if start < 0 or count <= 0 or start+count > x.numel():
        raise ValueError('Invalid NPU contiguous slice range')
    if start == 0 and count == x.numel():
        return x.view(count)
    if count > MAX_ELEMENTS:
        tiles = [_slice(x, start+offset, min(MAX_ELEMENTS, count-offset))
                 for offset in range(0, count, MAX_ELEMENTS)]
        return _concat(tiles, (count,), x.dtype)
    return _run(("slice", bridge._DTYPES[x.dtype][0], (x.numel(),), start, count),
                [x], (count,), x.dtype)


def _concat(tiles, shape, dtype, out=None):
    if sum(t.numel() for t in tiles) != prod(shape):
        raise ValueError("NPU tile assembly does not match the requested shape")
    # The native argument limit includes the output. All assembly is byte DMA.
    while len(tiles) > 63:
        tiles = [_concat(tiles[i:i+63], (sum(t.numel() for t in tiles[i:i+63]),), dtype)
                 for i in range(0, len(tiles), 63)]
    if len(tiles) == 1:
        result = tiles[0].view(shape)
        return _copy(out, result) if out is not None else result
    sizes = tuple(t.numel() for t in tiles)
    result = _run(("matrix_concat", bridge._DTYPES[dtype][0], (prod(shape), 1), sizes),
                  [t.view(-1, 1) for t in tiles], (prod(shape), 1), dtype,
                  out.view(-1, 1) if out is not None else None)
    return out if out is not None else result.view(shape)


def _concat_many(groups, shapes, dtype, outs=None):
    """Merge independent full-buffer concatenations without numeric operations."""
    with _lock:
        synchronize();groups=tuple(tuple(group) for group in groups);shapes=tuple(tuple(shape) for shape in shapes)
        if (dtype not in bridge._DTYPES or len(groups)!=len(shapes) or len(groups)<2
            or sum(map(len,groups))+len(groups)>64 or any(not 2<=len(group)<=63 for group in groups)):
            raise ValueError('Invalid NPU multi-concat groups/dtype/binding count')
        for group in groups:
            for value in group:
                _validate(value)
                if value.dtype!=dtype:raise ValueError('NPU multi-concat requires one dtype')
        sizes=tuple(tuple(value.numel() for value in group) for group in groups)
        for counts,shape in zip(sizes,shapes):
            if not 1<=len(shape)<=4 or any(type(v) is not int or v<1 for v in shape) or sum(counts)!=prod(shape) or prod(shape)>MAX_TENSOR_ELEMENTS:
                raise ValueError('NPU multi-concat output shape differs or exceeds storage bound')
        key=('concat_many',bridge._DTYPES[dtype][0],tuple(sum(group) for group in sizes),sizes)
        return _run_many(key,[value for group in groups for value in group],[(shape,dtype) for shape in shapes],outs)


def _map_tiles(function, inputs, shape, dtype, out=None):
    _check_out(out, shape, dtype, inputs)
    tiles = [function(*[_slice(x, start, min(MAX_ELEMENTS, prod(shape)-start)) for x in inputs])
             for start in range(0, prod(shape), MAX_ELEMENTS)]
    return _concat(tiles, shape, dtype, out)


def _scalar(value, dtype):
    # Scalar constants are host arguments, not staged device Tensor payloads.
    host = torch.tensor(value, dtype=dtype)
    key = dtype, host.numpy().tobytes()
    if key not in _scalars:
        _scalars[key] = host.to("intel_npu")
        # ponytail: 128 tiny cached constants; embed graph constants if measured useful.
        if len(_scalars) > 128:
            _scalars.popitem(last=False)
    else:
        _scalars.move_to_end(key)
    return _scalars[key]


def _broadcast(x, shape):
    _validate(x)
    shape = tuple(shape)
    if len(shape) > 4 or not 0 < prod(shape) <= MAX_TENSOR_ELEMENTS or tuple(torch.broadcast_shapes(x.shape, shape)) != shape:
        raise ValueError("Invalid NPU broadcast shape")
    if tuple(x.shape) == shape:
        return x
    if x.numel() == prod(shape):
        return x.view(shape)
    return _run(("broadcast", bridge._DTYPES[x.dtype][0], tuple(x.shape) or (1,), shape),
                [x], shape, x.dtype)


def _binary_op(x, y, *, kind, out=None):
    with _lock:
        synchronize()
        tensors = [v for v in (x, y) if isinstance(v, torch.Tensor)]
        if not tensors or not any(v.device.type == "intel_npu" for v in tensors):
            raise ValueError("NPU arithmetic requires an NPU Tensor")
        for value in (x, y):
            if isinstance(value, torch.Tensor):
                if value.device.type == "intel_npu":
                    _validate(value, dense=True)
                elif value.device.type != "cpu" or value.ndim or value.requires_grad or value.is_complex():
                    raise ValueError("CPU arithmetic operands must be real scalar tensors without autograd")
            elif type(value) not in (int, float, bool):
                raise TypeError("NPU operand must be a Tensor or a real scalar")
        if kind == "sub" and any(type(v) is bool or isinstance(v, torch.Tensor) and v.dtype == torch.bool for v in (x, y)):
            raise ValueError("Torch subtraction does not support bool operands")
        shape = tuple(torch.broadcast_shapes(*(tuple(v.shape) if isinstance(v, torch.Tensor) else () for v in (x, y))))
        if len(shape) > 4 or not 0 < prod(shape) <= MAX_TENSOR_ELEMENTS:
            raise ValueError("NPU broadcast result requires rank 0-4 and 1-1048576 elements")
        dtype = torch.result_type(x, y)
        if dtype not in bridge._DTYPES:
            raise ValueError("NPU arithmetic result dtype must be FP16/FP32")
        _check_out(out, shape, dtype, (x, y))
        # CPU Torch's wrapped-scalar FP16 mul/div use FP32 opmath scalars;
        # add/sub round their scalar input to FP16 first. Preserve this boundary.
        work_dtype = torch.float32 if dtype == torch.float16 and kind in ("mul", "div") and (
            not isinstance(y, torch.Tensor) or y.ndim == 0 and y.dtype != torch.float16) else dtype
        operands = []
        for value in (x, y):
            if isinstance(value, torch.Tensor) and value.device.type == "intel_npu":
                value = _cast(value if value.is_contiguous() else _materialize(value), work_dtype)
            else:
                value = _scalar(value.item() if isinstance(value, torch.Tensor) else value, work_dtype)
            operands.append(_broadcast(value, shape))
        result = _execute(kind, *operands, out=out if dtype == work_dtype else None)
        return _cast(result, dtype, out=out)


def _matmul(x, y, *, out=None):
    with _lock:
        synchronize()
        m, k, n = _matrix_shape(x, y)
        _check_out(out, (m, n), x.dtype, (x, y))
        result = _matrix_fp32(x, y, out=out if out is not None and out.dtype == torch.float32 else None)
        return _cast(result, x.dtype, out=out)


def _matrix_shape(x, y):
    _validate(x, dense=True)
    _validate(y, dense=True)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[0] or x.dtype != y.dtype:
        raise ValueError("NPU MatMul requires matching-dtype 2D matrices with compatible inner dimensions")
    m, k = x.shape
    n = y.shape[1]
    if m * n > MAX_TENSOR_ELEMENTS:
        raise ValueError("NPU MatMul output requires M*N <= 1048576")
    return m, k, n


def _matrix_fp32(x, y, *, out=None):
    # ponytail: only measured 128/256 squares; expand dispatch after shape-specific benchmarks.
    if (x.shape[0], x.shape[1], y.shape[1]) in ((128, 128, 128), (256, 256, 256)):
        from torch_npu_rect_gemm import matrix_fp32
        return matrix_fp32(x, y, out=out)
    return _matrix_fp32_expanded(x, y, out=out)


def _matrix_fp32_expanded(x, y, *, out=None):
    m, k, n = x.shape[0], x.shape[1], y.shape[1]
    left, right = (_cast(_materialize(v), torch.float32) for v in (x, y))
    if k > MAX_ELEMENTS:
        # Both operands fit the public storage bound. Slice the transposed left
        # and row-major right so each K block uses only bounded contiguous DMA.
        width = MAX_ELEMENTS//max(m, n)
        column_major = _materialize(left.t())
        partials = []
        for start in range(0, k, width):
            count = min(width, k-start)
            a = _slice(column_major, start*m, count*m).view(count, m).t()
            b = _slice(right, start*n, count*n).view(count, n)
            partials.append(_matrix_fp32(a, b))
        # Keep the partial results in FP32, including for FP16 inputs. Reuse
        # compensated reduction before the single final output conversion.
        stacked = _concat(partials, (len(partials), m, n), torch.float32)
        return _reduce(stacked, 0, dtype=torch.float32, kind='sum', out=out)
    # Tile M and N while preserving the full K reduction order inside each tile.
    if m*k*n > MAX_ELEMENTS:
        columns = min(n, MAX_ELEMENTS//k)
        rows = MAX_ELEMENTS//(k*columns)
        # Large byte Gather lost the device on this driver. Column-major DMA
        # permutation plus contiguous slices avoids that unverified carrier.
        column_major = _materialize(right.t()) if columns < n else None
        tiles = []
        for start in range(0, m, rows):
            count = min(rows, m-start)
            row = _slice(left, start*k, count*k).view(count, k)
            pieces = []
            for column in range(0, n, columns):
                width = min(columns, n-column)
                block = right if width == n else _slice(column_major, column*k, width*k).view(width, k).t()
                pieces.append(_matrix_fp32(row, block))
            tiles.append(pieces[0] if len(pieces) == 1 else _materialize(
                _concat([_materialize(t.t()) for t in pieces], (n, count), torch.float32).t()))
        return _concat(tiles, (m, n), torch.float32, out)
    # ponytail: expanded products within each tile; use a dedicated GEMM kernel for speed.
    left = _broadcast(left.view(m, k, 1), (m, k, n))
    right = _broadcast(right.view(1, k, n), (m, k, n))
    return _reduce(_execute("mul", left, right), 1, dtype=torch.float32, kind="sum", out=out)


def _addmm(bias, x, y, *, beta=1, alpha=1, out=None):
    with _lock:
        synchronize()
        m, _, n = _matrix_shape(x, y)
        _validate(bias, dense=True)
        if bias.dtype != x.dtype or tuple(torch.broadcast_shapes(bias.shape, (m, n))) != (m, n):
            raise ValueError("NPU addmm bias must match the matrices' dtype and broadcast to (M,N)")
        if any(type(v) not in (int, float) for v in (alpha, beta)):
            raise ValueError("NPU addmm coefficients must be real numbers")
        if any(abs(v) > torch.finfo(torch.float32).max and abs(v) != float('inf') for v in (alpha, beta)):
            raise ValueError("NPU addmm coefficient cannot be represented in FP32 opmath")
        _check_out(out, (m, n), x.dtype, (bias, x, y))
        final_out = out if x.dtype == torch.float32 else None
        result = _matrix_fp32(x, y, out=final_out if beta == 0 and alpha == 1 else None)
        if alpha != 1:
            result = _binary_op(result, alpha, kind="mul", out=final_out if beta == 0 else None)
        if beta != 0:  # Torch ignores even NaN/Inf bias when beta=0.
            value = _cast(_materialize(bias), torch.float32)
            if beta != 1:
                value = _binary_op(value, beta, kind="mul")
            result = _binary_op(result, value, kind="add", out=final_out)
        return _cast(result, x.dtype, out=out)


def _linear(x, weight, bias=None):
    with _lock:
        synchronize()
        _validate(x, dense=True)
        _validate(weight, dense=True)
        if not 1 <= x.ndim <= 4 or weight.ndim != 2 or x.shape[-1] != weight.shape[1] or x.dtype != weight.dtype:
            raise ValueError("NPU Linear requires rank 1-4 input and matching (out_features,in_features) weights")
        m, k, n = prod(x.shape[:-1]), x.shape[-1], weight.shape[0]
        if m * n > MAX_TENSOR_ELEMENTS:
            raise ValueError("NPU Linear output requires M*N <= 1048576")
        if bias is not None:
            _validate(bias, dense=True)
            if bias.shape != (n,) or bias.dtype != x.dtype:
                raise ValueError("NPU Linear bias must have shape (out_features,) and matching dtype")
        matrix = _materialize(x).view(m, k)
        result = (_matmul(matrix, weight.t()) if bias is None else _addmm(bias, matrix, weight.t()))
        return result.view((*x.shape[:-1], n))


@lru_cache(maxsize=32)
def _conv_indices(shape, kernel, stride, padding, dilation):
    """CPU shape/index metadata only. No tensor payload is inspected/computed."""
    import numpy as np
    batch, channels, height, width = shape
    kh, kw = kernel
    oh = (height + 2*padding[0] - dilation[0]*(kh-1) - 1)//stride[0] + 1
    ow = (width + 2*padding[1] - dilation[1]*(kw-1) - 1)//stride[1] + 1
    m, k = batch*oh*ow, channels*kh*kw
    if oh <= 0 or ow <= 0 or m*k > MAX_TENSOR_ELEMENTS:
        raise ValueError("NPU convolution window storage exceeds 1048576 elements or has empty output")
    indices = np.full((m, k), -1, dtype=np.int64)
    inverse_lists = [[] for _ in range(prod(shape))]
    for b in range(batch):
        for y in range(oh):
            for x in range(ow):
                row = (b*oh+y)*ow+x
                for c in range(channels):
                    for ky in range(kh):
                        for kx in range(kw):
                            iy, ix = y*stride[0]-padding[0]+ky*dilation[0], x*stride[1]-padding[1]+kx*dilation[1]
                            column = (c*kh+ky)*kw+kx
                            if 0 <= iy < height and 0 <= ix < width:
                                src = ((b*channels+c)*height+iy)*width+ix
                                indices[row, column] = src
                                inverse_lists[src].append(row*k+column)
    overlap = max(1, max(map(len, inverse_lists)))
    if prod(shape)*overlap > MAX_TENSOR_ELEMENTS:
        raise ValueError("NPU convolution input-gradient fold storage exceeds 1048576 elements")
    inverse = np.full((prod(shape), overlap), -1, dtype=np.int64)
    for i, entries in enumerate(inverse_lists):
        inverse[i, :len(entries)] = entries
    indices.setflags(write=False)
    inverse.setflags(write=False)
    return (oh, ow), indices, inverse


def _conv_spec(x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    _validate(x, dense=True)
    _validate(weight, dense=True)
    if (transposed or groups != 1 or any(output_padding) or x.ndim != 4 or weight.ndim != 4 or
        x.dtype != weight.dtype or x.shape[1] != weight.shape[1]):
        raise ValueError("NPU convolution requires matching NCHW FP16/FP32, groups=1, non-transposed Conv2d")
    for name, values, minimum in (("stride", stride, 1), ("padding", padding, 0), ("dilation", dilation, 1)):
        if len(values) != 2 or any(type(v) is not int or v < minimum for v in values):
            raise ValueError(f"Invalid NPU convolution {name}")
    if bias is not None:
        _validate(bias)
        if bias.dtype != x.dtype or bias.shape != (weight.shape[0],):
            raise ValueError("NPU convolution bias must match output channels/dtype")
    params = tuple(weight.shape[2:]), tuple(stride), tuple(padding), tuple(dilation)
    spatial, shape = conv_geometry(tuple(x.shape), *params)
    if shape[0]*weight.shape[0] > MAX_TENSOR_ELEMENTS:
        raise ValueError("NPU convolution output exceeds 1048576 elements")
    return spatial, params, shape


def _conv_streaming(original, params, shape):
    # Keep the validated small path. Stream when either full temporary could
    # exceed storage capacity; the fold upper bound is kernel area per pixel.
    return max(prod(shape), prod(original)*prod(params[0])) > MAX_TENSOR_ELEMENTS


def _gather_window(x, indices, row_width):
    """Bounded byte DMA gathers; key contains immutable CPU index metadata."""
    import numpy as np
    flat = indices.reshape(-1)
    width = indices.shape[1]
    batch = max(1, _GATHER_TILE_ELEMENTS//width)
    if batch >= row_width:
        batch = batch//row_width*row_width
    block = min(_GATHER_TILE_ELEMENTS, batch*width)
    tiles, start = [], 0
    while start < flat.size:
        count = min(block, flat.size-start)
        while True:
            selected = flat[start:start+count]
            valid = selected[selected >= 0]
            base = int(valid.min()) if valid.size else 0
            span = int(valid.max())-base+1 if valid.size else 1
            if span <= MAX_ELEMENTS:
                break
            count //= 2
        relative = np.where(selected < 0, -1, selected-base).astype(np.int64).tobytes()
        source = _slice(x, base, span)
        tiles.append(_run(('gather_tile', bridge._DTYPES[x.dtype][0], (span,), relative),
                          [source], (count,), x.dtype))
        start += count
    return _concat(tiles, tuple(indices.shape), x.dtype)


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


def _unfold(x, params, shape):
    x = _materialize(x)
    _, indices, _ = _conv_indices(tuple(x.shape), *params)
    return _conv_transport('unfold', x, indices, shape, tuple(x.shape), params)


def _fold(x, original, params):
    _, _, inverse = _conv_indices(original, *params)
    return _conv_transport('fold', x, inverse, tuple(inverse.shape), original, params)


def _conv_transport(kind, x, indices, shape, original, params):
    if x.numel() <= MAX_ELEMENTS and indices.size <= MAX_ELEMENTS:
        extra = params if kind == 'unfold' else (original, *params)
        return _run((kind, bridge._DTYPES[x.dtype][0], tuple(x.shape), *extra), [x], shape, x.dtype)
    # Large Gather is unverified on this driver. Each gather binds a bounded
    # contiguous source slice; padding and reassembly remain exact byte DMA.
    flat = indices.reshape(-1)
    tiles, start = [], 0
    while start < flat.size:
        count = min(_GATHER_TILE_ELEMENTS, flat.size-start)
        while True:
            selected = flat[start:start+count]
            valid = selected[selected >= 0]
            base = int(valid.min()) if valid.size else 0
            span = int(valid.max())-base+1 if valid.size else 1
            if span <= MAX_ELEMENTS:
                break
            count //= 2
        tile = _slice(x, base, span)
        key = f'{kind}_tile', bridge._DTYPES[x.dtype][0], (span,), original, *params, start, count, base
        tiles.append(_run(key, [tile], (count,), x.dtype))
        start += count
    return _concat(tiles, shape, x.dtype)


def _convolution(x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    with _lock:
        synchronize()
        spatial, params, shape = _conv_spec(x, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
        return _conv_forward(x, weight, bias, spatial, params, shape)[0]


def _mse_loss(x, target, reduction=1):
    with _lock:
        synchronize()
        _validate(x, dense=True)
        _validate(target, dense=True)
        if x.shape != target.shape or x.dtype != target.dtype or reduction not in (0, 1, 2):
            raise ValueError("NPU MSE requires matching shape/dtype and reduction none/mean/sum")
        difference = _binary_op(_cast(_materialize(x), torch.float32),
                                _cast(_materialize(target), torch.float32), kind="sub")
        result = _execute("mul", difference, difference)
        if reduction:
            result = _reduce(result, kind="mean" if reduction == 1 else "sum")
        return _cast(result, x.dtype)


def _batch_norm(x, weight, bias, running_mean, running_var, training, momentum, eps):
    """NCHW BatchNorm, including functional running buffers; all statistics use FP32."""
    with _lock:
        synchronize()
        _validate(x, dense=True)
        if x.ndim != 4:
            raise ValueError("NPU BatchNorm currently requires rank-4 NCHW")
        if not isfinite(eps) or eps < 0 or training and eps == 0 or not isfinite(momentum):
            raise ValueError("NPU BatchNorm requires finite eps/momentum and positive training eps")
        channels, count = x.shape[1], x.numel()//x.shape[1]
        if training and count <= 1:
            raise ValueError("NPU training BatchNorm requires more than one value per channel")
        if (running_mean is None) != (running_var is None) or not training and running_mean is None:
            raise ValueError("NPU eval BatchNorm requires both running buffers")
        buffers = [v for v in (running_mean, running_var) if v is not None]
        for v in (weight, bias, *buffers):
            if v is not None:
                _validate(v)
                if v.shape != (channels,) or v.dtype not in (x.dtype, torch.float32):
                    raise ValueError("NPU BatchNorm parameters/buffers require shape (C,) and input/FP32 dtype")
        if any(v.requires_grad for v in buffers):
            raise ValueError("NPU running statistics must not require gradients")
        owners = [v.untyped_storage()._cdata for v in (x, weight, bias, *buffers) if v is not None]
        if len(set(owners)) != len(owners):
            raise ValueError("NPU BatchNorm inputs and mutable running buffers must not alias")
        values = _cast(_materialize(x), torch.float32)
        axes, shape = (0, 2, 3), (1, channels, 1, 1)
        mean = (_reduce(values, axes, kind="mean") if training else _cast(running_mean, torch.float32))
        centered = _binary_op(values, mean.view(shape), kind="sub")
        variance = (_reduce(_execute("mul", centered, centered), axes, kind="mean")
                    if training else _cast(running_var, torch.float32))
        inverse = _execute("rsqrt", _binary_op(variance, eps, kind="add"))
        normalized = _binary_op(centered, inverse.view(shape), kind="mul")
        result = normalized
        if weight is not None:
            result = _binary_op(result, _cast(weight, torch.float32).view(shape), kind="mul")
        if bias is not None:
            result = _binary_op(result, _cast(bias, torch.float32).view(shape), kind="add")
        if training and buffers:
            unbiased = _binary_op(variance, count/(count-1), kind="mul")
            for buffer, current in zip(buffers, (mean, unbiased)):
                old = _binary_op(_cast(buffer, torch.float32), 1.-momentum, kind="mul")
                update = _binary_op(old, _binary_op(current, momentum, kind="mul"), kind="add")
                buffer.copy_(_cast(update, buffer.dtype))
        return _cast(result, x.dtype), normalized, inverse


def _adaptive_avg_pool2d(x, output_size):
    _validate(x, dense=True)
    if x.ndim != 4 or tuple(output_size) != (1, 1):
        raise ValueError("NPU adaptive average pooling currently supports NCHW output_size=(1,1)")
    return _reduce(_materialize(x), (2, 3), True, kind="mean")


def _fill(x, value):
    with _lock:
        synchronize()
        _validate(x, dense=True)
        if type(value) not in (int, float, bool):
            raise ValueError("NPU fill requires a real scalar")
        target = _physical_view(x)[0] if not x.is_contiguous() else x
        shape = tuple(target.shape) or (1,)
        _run(("broadcast", bridge._DTYPES[x.dtype][0], (1,), shape),
             [_scalar(value, x.dtype)], tuple(target.shape), x.dtype, out=target)
        return x


def _add_inplace(x, y, *, alpha=1):
    with _lock:
        synchronize()
        if torch.is_grad_enabled():
            raise ValueError("NPU in-place updates require no_grad; differentiable optimizers are unsupported")
        _validate(x, dense=True)
        if not isinstance(y, torch.Tensor) or y.dtype != x.dtype or y.device != x.device or y.shape != x.shape:
            raise ValueError("NPU in-place add requires matching shape/dtype/device")
        _validate(y, dense=True)
        if type(alpha) not in (int, float):
            raise ValueError("NPU in-place alpha must be real")
        left, right = _cast(_materialize(x), torch.float32), _cast(_materialize(y), torch.float32)
        if alpha != 1:
            right = _binary_op(right, alpha, kind="mul")
        result = _cast(_execute("add", left, right), x.dtype)
        return _copy(x, result)


def _cast(x, dtype, out=None):
    _validate(x)
    if dtype not in bridge._DTYPES:
        raise ValueError("NPU conversion supports FP16/FP32 only")
    if out is not None:
        _validate(out)
        if out.shape != x.shape or out.dtype != dtype:
            raise ValueError("NPU cast output has the wrong shape or dtype")
    if x.dtype == dtype:
        return _copy(out, x) if out is not None and out is not x else x
    if x.numel() > MAX_ELEMENTS:
        return _map_tiles(lambda tile: _cast(tile, dtype), [x], tuple(x.shape), dtype, out)
    key = "cast", bridge._DTYPES[x.dtype][0], tuple(x.shape) or (1,)
    raw = _run(key, [x], tuple(x.shape), torch.float32, out if dtype == torch.float32 else None)
    if dtype == torch.float32:
        return raw
    # Metadata-only reinterpretation. Low 16-bit lane compaction runs as NPU DMA.
    transport = raw.view(-1).view(torch.float16).view(x.numel(), 2)
    result = _run(("halfcode_slice", "f16", (x.numel(),)), [transport], (x.numel(), 1), dtype,
                  out.view(x.numel(), 1) if out is not None else None)
    return out if out is not None else result.view(tuple(x.shape))


def _copy(dst, src, non_blocking=False):
    with _lock:
        synchronize()
        if non_blocking or dst.shape != src.shape or dst.dtype not in bridge._DTYPES or src.dtype not in bridge._DTYPES:
            raise ValueError("NPU copy requires identical shapes, FP16/FP32, and synchronous execution")
        if torch.is_grad_enabled() and (dst.requires_grad or src.requires_grad):
            raise ValueError("NPU device autograd is not implemented")
        if dst.device.type not in ("cpu", "intel_npu") or src.device.type not in ("cpu", "intel_npu"):
            raise ValueError("Only CPU/intel_npu copies are implemented")
        for tensor in (dst, src):
            if tensor.device.type == "intel_npu":
                _validate(tensor, dense=True)
        if (dst.device.type == 'cpu' and not dst.is_contiguous() or
                any(t.is_neg() or t.is_conj() for t in (dst, src))):
            raise ValueError("Copy requires a contiguous destination and resolved source value bits")
        if not dst.is_contiguous():
            if src.device.type == 'cpu':
                src = _copy(torch.empty(src.shape, dtype=dst.dtype, device=dst.device), src)
            else:
                src = _cast(_materialize(src), dst.dtype)
            base, axes = _physical_view(dst)
            # Write the inverse permutation through a complete contiguous view;
            # no partial pointers or host payload arithmetic enter native binding.
            _copy(base, _materialize(src.permute(axes)))
            return dst
        if src.device.type == "intel_npu":
            if not src.is_contiguous() and dst.device.type == "intel_npu" and dst.dtype == src.dtype:
                return _materialize(src, out=dst)
            src = _materialize(src)
        if dst.dtype != src.dtype:
            if src.device.type == "cpu":
                uploaded = torch.empty(src.shape, device="intel_npu:0", dtype=src.dtype)
                _C.copy(uploaded, src, False)
                src = uploaded
            if dst.device.type == "intel_npu":
                return _cast(src, dst.dtype, out=dst)
            src = _cast(src, dst.dtype)
        if dst.device.type == src.device.type == "intel_npu":
            if dst.data_ptr() == src.data_ptr():
                return dst
            shape = tuple(src.shape) or (1,)
            return _run(("broadcast", bridge._DTYPES[src.dtype][0], shape, shape),
                        [src], tuple(src.shape), src.dtype, out=dst)
        return _C.copy(dst, src, False)


def _to_copy(x, *, dtype=None, layout=None, device=None, pin_memory=None,
             non_blocking=False, memory_format=None, out=None):
    target_device = torch.device(device) if device is not None else x.device
    active = x.requires_grad or out is not None and out.requires_grad
    if torch.is_grad_enabled() and active and (
            x.device.type != "intel_npu" or target_device.type != "intel_npu" or out is not None):
        raise ValueError("Differentiable NPU conversion requires an in-device FP16/FP32 copy without out=")
    # Torch's native ToCopyBackward handles the reverse dtype/layout conversion.
    # This backend kernel only performs the forward SHAVE/DMA operation.
    with _lock, torch.no_grad():
        synchronize()
        if x.device.type == "cpu":
            if not x.is_contiguous() and memory_format != torch.contiguous_format:
                raise ValueError("Noncontiguous CPU upload requires memory_format=torch.contiguous_format")
            bridge._validate(x, max_elements=MAX_TENSOR_ELEMENTS)
        else:
            _validate(x, dense=True)
        if (layout not in (None, torch.strided) or pin_memory or non_blocking or
            memory_format not in (None, torch.preserve_format, torch.contiguous_format)):
            raise ValueError("NPU conversion requires synchronous contiguous, unpinned, strided tensors")
        if out is not None:
            return _copy(out, x)
        dtype = dtype or x.dtype
        device = target_device
        if dtype not in bridge._DTYPES or device.type not in ("cpu", "intel_npu"):
            raise ValueError("NPU conversion supports FP16/FP32 and CPU/intel_npu only")
        if device.type == "intel_npu":
            _check_device(device)
        if x.device.type == "cpu":
            return _copy(torch.empty(tuple(x.shape), dtype=dtype, device=device), x)
        converted = _cast(_materialize(x), dtype)
        if device.type == "intel_npu" and converted is not x:
            return converted
        result = torch.empty(tuple(x.shape), dtype=dtype, device=device)
        return _copy(result, converted)


def _reduce(x, dim=None, keepdim=False, *, dtype=None, out=None, kind):
    with _lock:
        synchronize()
        _validate(x, dense=True)
        requested = dim if isinstance(dim, (tuple, list)) else (() if dim is None else (dim,))
        rank = x.ndim or 1
        axes = []
        for value in requested:
            axis = operator.index(value)
            if not -rank <= axis < rank:
                raise IndexError("NPU reduction axis is out of range")
            axis %= rank
            if axis in axes:
                raise ValueError("NPU reduction axes must be distinct")
            axes.append(axis)
        axes = tuple(sorted(axes)) if axes else tuple(range(rank))
        result_dtype = dtype or (out.dtype if out is not None else x.dtype)
        if result_dtype not in bridge._DTYPES:
            raise ValueError("NPU reduction dtype must be FP16/FP32")
        shape = tuple(1 if i in axes else d for i, d in enumerate(x.shape)) if keepdim else tuple(
            d for i, d in enumerate(x.shape) if i not in axes)
        if out is not None:
            _validate(out)
            if out.shape != shape or out.dtype != result_dtype:
                raise ValueError("NPU reduction out requires the exact result shape and dtype")
            if out.untyped_storage()._cdata == x.untyped_storage()._cdata:
                raise ValueError("NPU reduction out may not alias its input")
        # Torch sum casts each input to its requested dtype; CPU mean instead
        # accumulates FP32 inputs in FP32 even when its output is FP16.
        values = x if x.is_contiguous() else _materialize(x)
        values = _cast(values, result_dtype) if kind == "sum" else values
        values = _cast(values, torch.float32)
        key = kind, "f32", tuple(x.shape) or (1,), axes
        destination = out if out is not None and result_dtype == torch.float32 else None
        result = (_reduce_large(values, axes, shape, kind, destination) if x.numel() > MAX_ELEMENTS else
                  _run(key, [values], shape, torch.float32, destination))
        return _cast(result, result_dtype, out=out)


def _reduce_large(x, axes, shape, kind, out):
    order = tuple(a for a in range(x.ndim) if a not in axes) + axes
    matrix = _materialize(x.permute(order))
    width = prod(x.shape[a] for a in axes)
    rows = x.numel()//width
    tiles = []
    if width <= MAX_ELEMENTS:
        batch = MAX_ELEMENTS//width
        for start in range(0, rows, batch):
            count = min(batch, rows-start)
            tile = _slice(matrix, start*width, count*width).view(count, width)
            tiles.append(_reduce(tile, 1, kind=kind))
    else:
        # Chunked reductions reassociate FP32 sums; means divide by the full count once.
        for row in range(rows):
            partials = [_reduce(_slice(matrix, row*width+start, min(MAX_ELEMENTS, width-start)), kind='sum')
                        for start in range(0, width, MAX_ELEMENTS)]
            total = _reduce(_concat(partials, (len(partials),), torch.float32), kind='sum')
            tiles.append(_binary_op(total, width, kind='div') if kind == 'mean' else total)
    return _concat(tiles, shape, torch.float32, out)


def _add(x, y, *, alpha=1, out=None):
    if type(alpha) not in (int, float) or alpha != 1:
        raise NotImplementedError("NPU add/sub currently requires alpha=1")
    return _binary_op(x, y, kind="add", out=out)


def _sub(x, y, *, alpha=1, out=None):
    if type(alpha) not in (int, float) or alpha != 1:
        raise NotImplementedError("NPU add/sub currently requires alpha=1")
    return _binary_op(x, y, kind="sub", out=out)


def _rsub(x, y, *, alpha=1, out=None):
    if type(alpha) not in (int, float) or alpha != 1:
        raise NotImplementedError("NPU add/sub currently requires alpha=1")
    return _binary_op(y, x, kind="sub", out=out)


_aten = torch.library.Library("aten", "IMPL", "PrivateUse1")
_aten.impl("copy_", _copy)
_aten.impl("_to_copy", _to_copy)
_aten.impl("_to_copy.out", _to_copy)
for _kind, _overloads in (("sum", ("sum", "sum.dim_IntList", "sum.out", "sum.IntList_out")),
                         ("mean", ("mean", "mean.dim", "mean.dtype_out", "mean.out"))):
    def _reduction(x, dim=None, keepdim=False, *, dtype=None, out=None, kind=_kind):
        return _reduce(x, dim, keepdim, dtype=dtype, out=out, kind=kind)
    for _overload in _overloads:
        _aten.impl(_overload, _reduction)
for _kind, _function in (("add", _add), ("sub", _sub), ("rsub", _rsub)):
    for _overload in (("Tensor", "Scalar", "Tensor_out", "Scalar_out") if _kind == "rsub" else
                      ("Tensor", "Scalar", "out", "Scalar_out")):
        _aten.impl(f"{_kind}.{_overload}", _function)
for _kind in ("mul", "div"):
    def _binary(x, y, *, out=None, rounding_mode=None, kind=_kind):
        if rounding_mode is not None:
            raise NotImplementedError("NPU division requires rounding_mode=None")
        return _binary_op(x, y, kind=kind, out=out)
    for _overload in ("Tensor", "Scalar", "out", "Scalar_out"):
        _aten.impl(f"{_kind}.{_overload}", _binary)
    if _kind == "div":
        for _overload in ("Tensor_mode", "Scalar_mode", "out_mode", "Scalar_mode_out"):
            _aten.impl(f"div.{_overload}", _binary)
for _kind in ("mm", "matmul"):
    _aten.impl(_kind, _matmul)
    _aten.impl(f"{_kind}.out", _matmul)
_aten.impl("addmm", _addmm)
_aten.impl("addmm.out", _addmm)
_aten.impl("linear", _linear)
_aten.impl("convolution", _convolution)
_aten.impl("mse_loss", _mse_loss)
_aten.impl("batch_norm", lambda x, w, b, rm, rv, training, momentum, eps, cudnn:
           _batch_norm(x, w, b, rm, rv, training, momentum, eps)[0])
for _name in ("adaptive_avg_pool2d", "_adaptive_avg_pool2d"):
    _aten.impl(_name, _adaptive_avg_pool2d)
_aten.impl("fill_.Scalar", _fill)
_aten.impl("zero_", lambda x: _fill(x, 0.))
_aten.impl("add_.Tensor", _add_inplace)
_aten.impl("relu", lambda x: _execute("relu", x))
def _reciprocal(x, *, out=None):
    return _binary_op(1., x, kind="div", out=out)

_aten.impl("reciprocal", _reciprocal)
_aten.impl("reciprocal.out", _reciprocal)
for _kind in ("neg", "abs", "sign"):
    def _unary(x, *, out=None, kind=_kind):
        return _execute(kind, x, out=out)
    _aten.impl(_kind, _unary)
    _aten.impl(f"{_kind}.out", _unary)
_fallback = torch.library.Library("_", "IMPL", "PrivateUse1")
_fallback.fallback(_unsupported)


def _manual_seed_all(seed):
    # torch.manual_seed also calls registered device modules. This backend has
    # no RNG state; random ATen operators still reject through the fallback.
    pass


torch._register_device_module("intel_npu", SimpleNamespace(
    is_available=lambda: not _closed and not _C.stats()["context_failed"],
    device_count=lambda: 1, current_device=lambda: 0, synchronize=synchronize, empty_cache=empty_cache,
    _is_in_bad_fork=lambda: False, manual_seed_all=_manual_seed_all,
    get_amp_supported_dtype=lambda: []))
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=False)
atexit.register(shutdown)

# Register only the explicitly implemented first-order device derivatives.
import torch_npu_autograd

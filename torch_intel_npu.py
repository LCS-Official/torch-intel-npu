"""Minimal real PrivateUse1 device for the validated Lunar Lake NPU profile.

Import registers intel_npu:0. Supported tensors own Level Zero shared storage;
ordinary ATen arithmetic reuses resident graph sessions without host staging.
"""
import atexit
import hashlib
import importlib.util
import json
import operator
import struct
from math import prod, isfinite
from functools import lru_cache
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from weakref import WeakSet, WeakValueDictionary

import npunlock as npu
import torch
import torch_npu_minimal as bridge
from torch_npu_rng import integer_range, permutation_size
from torch_npu_geometry import (conv_geometry, conv_window_indices, reduction_geometry, triangle_indices,
                               index_tile_width, softmax_row_tiles, nll_row_tiles, indexed_rows_geometry, avg_pool2d_geometry, pool_pair, row_extrema_packet,
                               selection_packet, max_pool2d_geometry, max_pool_windows)
from npunlock.npu4000 import (compile_halfcode_slice, compile_reduce, compile_broadcast, compile_permute, compile_int64_counter,
                             compile_mask_packet, mask_packet_spec, compile_rng_packet, rng_packet_words,
                             compile_discrete_packet, discrete_packet_words,
                             compile_classification_rows, compile_classification_pack,
                             compile_gather, compile_matrix_concat, compile_concat_many, compile_slice, compile_unary, compile_binary,
                             MAX_ELEMENTS, MAX_STORAGE_ELEMENTS as MAX_TENSOR_ELEMENTS,
                             MAX_ALLOCATION_BYTES, MAX_DMA_ELEMENTS, MAX_STORAGE_RANK)
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
if (_C.max_tensor_elements != MAX_DMA_ELEMENTS or _C.max_storage_rank != MAX_STORAGE_RANK or
        getattr(_C,'max_allocation_bytes',None)!=MAX_ALLOCATION_BYTES):
    raise RuntimeError("NPU storage bounds differ; rebuild the C++ extension")
_C.configure(str(SESSION_DLL), _info.session)
_sessions = OrderedDict()
_SMOL_SESSION_KINDS = ('smollm_fused_norm','smollm_prefill_recipe','smollm_rmsnorm')
_smol_plan_depth = 0
_rebuildable_sessions = set()
_retired_sessions = WeakSet()
_graph_factories = WeakValueDictionary()
_session_evictions = 0
_graph_factory_calls = 0
_transient_graph_imports = 0
_scalars = OrderedDict()
_session_creations = 1
_MAX_SESSIONS = 1536
_MAX_SESSION_BYTES = 512 * 1024 * 1024
_GATHER_TILE_ELEMENTS = 8192  # Conservative byte-Gather output bound on this driver.
_SLICE_COMPILE_TIMEOUT_MS = 60_000  # Cold DMA Slice compilation can exceed the old 20s budget.
_STORAGE_DTYPES = (*bridge._DTYPES, torch.int64, torch.bool, torch.int32, torch.int16,
                   torch.int8, torch.uint8, torch.bfloat16)
_MATH_UNARY = ('exp', 'log', 'tanh', 'sigmoid', 'silu', 'sqrt', 'rsqrt')
MAX_INT64_ELEMENTS = MAX_DMA_ELEMENTS // 2
MAX_RANK = MAX_STORAGE_RANK
_CONV_WORKSPACE_ELEMENTS = 4 * MAX_ELEMENTS
# Float-metadata kernels retain MAX_TENSOR_ELEMENTS; ordinary storage uses _storage_limit.
_closed = False
_rng = _C.default_generator()
_rng.manual_seed(torch.initial_seed())


def _storage_limit(dtype):
    return MAX_ALLOCATION_BYTES//dtype.itemsize


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
                "graph_cache_evictions": _session_evictions,
                "cached_graph_blob_bytes": _session_blob_bytes(), "graph_cache_limit_bytes": _MAX_SESSION_BYTES,
                "graph_factory_calls": _graph_factory_calls, "live_graph_factories": len(_graph_factories),
                "transient_graph_imports": _transient_graph_imports,
                "pinned_graph_sessions": sum(key not in _rebuildable_sessions for key in _sessions),
                "deferred_graph_sessions": sum(not s._closed for s in _retired_sessions),
                "cached_scalar_buffers": len(_scalars), "closed": _closed,
                "max_tensor_elements": MAX_DMA_ELEMENTS, "max_compute_tile_elements": MAX_ELEMENTS,
                "dtype_storage_limit_elements": {str(dtype):_storage_limit(dtype) for dtype in _STORAGE_DTYPES},
                "legacy_operator_max_elements": MAX_TENSOR_ELEMENTS,
                "max_allocation_bytes": MAX_ALLOCATION_BYTES, "max_storage_rank": MAX_RANK,
                "storage_dtypes": [str(dtype) for dtype in _STORAGE_DTYPES],
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
            _rebuildable_sessions.clear()
            _graph_factories.clear()
            for session in tuple(_retired_sessions):
                session.close()
            _retired_sessions.clear()
            _root_session.close()
            _closed = True


def _unsupported(func, *args, **kwargs):
    raise NotImplementedError(f"intel_npu does not implement {func}; no CPU arithmetic fallback")


def _validate(tensor, *, dense=False, storage=False):
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "intel_npu" or
            tensor.dtype not in (_STORAGE_DTYPES if storage else bridge._DTYPES)):
        raise ValueError("Requires intel_npu FP16/FP32 compute tensors or supported storage dtype")
    if tensor.requires_grad and torch.is_grad_enabled():
        raise ValueError("Requires an implemented NPU autograd operator, or no_grad/inference_mode")
    if tensor.storage_offset() != 0 or tensor.is_neg() or tensor.is_conj():
        raise ValueError("Requires contiguous, offset-zero NPU tensors")
    limit = _storage_limit(tensor.dtype)
    if tensor.ndim > MAX_RANK or not 0 <= tensor.numel() <= limit:
        raise ValueError("Tensor exceeds its NPU dtype/rank storage bound")
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


def _dma_dtype(dtype):
    return 'u8' if dtype in (torch.bool, torch.uint8) else bridge._DTYPES[dtype][0]


def _is_dense(x):
    expected = 1
    for axis in sorted(range(x.ndim), key=lambda a: x.stride(a)):
        if x.shape[axis] != 1:
            if x.stride(axis) != expected:
                return False
            expected *= x.shape[axis]
    return True


def _validate_read(x):
    """Permit bounded strided/broadcast views for reads, never native bindings."""
    if (not isinstance(x, torch.Tensor) or x.device.type != 'intel_npu' or x.dtype not in _STORAGE_DTYPES):
        raise ValueError('NPU layout reads require FP16/FP32/INT64/BOOL device tensors')
    if x.requires_grad and torch.is_grad_enabled():
        raise ValueError('NPU layout read requires an implemented autograd operator or no_grad')
    limit = _storage_limit(x.dtype)
    size = x.untyped_storage().nbytes()
    if (x.ndim > MAX_RANK or not 0 <= x.numel() <= limit or x.is_neg() or x.is_conj() or
            size % x.element_size() or not 0 <= size // x.element_size() <= limit or x.storage_offset() < 0):
        raise ValueError('NPU layout read exceeds its dtype/rank/storage bounds')
    if not x.numel():
        if any(v<0 for v in x.stride()):raise ValueError('NPU empty view requires nonnegative strides')
        return
    footprint = 1
    for axis in sorted(range(x.ndim), key=lambda a: x.stride(a)):
        stride, count = x.stride(axis), x.shape[axis]
        if stride < 0:
            raise ValueError('NPU layout read requires nonnegative strides')
        footprint += (count-1)*stride
    if x.storage_offset()+footprint > size // x.element_size():
        raise ValueError('NPU layout read exceeds its backing allocation')


def _materialize(x, out=None):
    _validate_read(x)
    _check_out(out, tuple(x.shape), x.dtype, [x])
    if not x.numel():return torch.empty(tuple(x.shape),dtype=x.dtype,device=x.device) if out is None else out
    complete = x.storage_offset() == 0 and x.numel()*x.element_size() == x.untyped_storage().nbytes()
    if complete and x.is_contiguous():
        return _copy(out, x) if out is not None else x
    if complete and _is_dense(x) and x.dtype in (*bridge._DTYPES, torch.bool) and x.ndim<=4:
        base, axes = _physical_view(x)
        order = tuple(axes.index(a) for a in range(x.ndim))
        return _run(("permute", _dma_dtype(x.dtype), tuple(base.shape), order),
                    [base], tuple(x.shape), x.dtype, out=out)
    import numpy as np
    # Offsets are shape/stride metadata. All payload reads remain byte DMA.
    offsets = np.full(tuple(x.shape), x.storage_offset(), dtype=np.int64)
    for axis, (count, stride) in enumerate(zip(x.shape, x.stride())):
        shape = [1]*x.ndim
        shape[axis] = count
        offsets += np.arange(count, dtype=np.int64).reshape(shape)*stride
    backing = x.as_strided((x.untyped_storage().nbytes()//x.element_size(),), (1,), storage_offset=0)
    if x.dtype not in (*bridge._DTYPES, torch.bool):
        width = x.element_size()
        offsets = (offsets.reshape(-1, 1)*width+np.arange(width)).reshape(1, -1)
        result = _gather_window(backing.view(torch.uint8), offsets, 1).view(-1).view(x.dtype)
    else:
        result = _gather_window(backing, offsets.reshape(1, -1), 1)
    result = result.view(tuple(x.shape))
    return _copy(out, result) if out is not None else result


def _validate_float_read(x):
    _validate_read(x)
    if x.dtype not in bridge._DTYPES:
        raise ValueError('NPU compute reads require FP16/FP32')


def _triangle(x,diagonal=0,*,upper=True,out=None):
    with _lock:
        synchronize()
        _validate_read(x)
        indices=triangle_indices(tuple(x.shape),diagonal,upper=upper)
        _check_out(out,tuple(x.shape),x.dtype,(x,))
        values=_materialize(x).view(-1)
        if x.dtype==torch.int64:
            import numpy as np
            indices=np.where(indices.reshape(-1,1)<0,-1,indices.reshape(-1,1)*2+np.arange(2)).reshape(1,-1)
            values=values.view(torch.float32)
        result=_gather_window(values,indices,1).view(-1)
        if x.dtype==torch.int64:result=result.view(torch.int64)
        result=result.view(x.shape)
        return _copy(out,result) if out is not None else result


def _boolean_reduce(x,dim=None,keepdim=False,*,out=None,kind):
    with _lock:
        synchronize()
        _validate_read(x)
        if x.dtype!=torch.bool:raise ValueError('NPU all/any currently require BOOL input')
        empty_dims=isinstance(dim,(tuple,list)) and not dim
        axes,shape=((),tuple(x.shape)) if empty_dims else reduction_geometry(tuple(x.shape),dim,keepdim)
        _check_out(out,shape,torch.bool,(x,))
        # BOOL all/any with dim=[] copy the input; sum instead reduces all axes.
        if empty_dims:return _copy(out,x) if out is not None else _clone(x)
        values=_mask_cast(_materialize(x),torch.float32)
        counts=_reduce(values,axes,keepdim,dtype=torch.float32,kind='sum')
        count=prod(x.shape[a] for a in axes) if x.ndim else 1
        result=_compare(counts,count if kind=='all' else 0,kind='eq' if kind=='all' else 'gt')
        return _copy(out,result) if out is not None else result


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


def _session_blob_bytes():
    return sum(len(program.graph_blob) for session,program in _sessions.values())


def set_graph_cache_budget(max_bytes):
    """Change the graph-blob budget (1 byte..1 GiB) without evicting live entries."""
    global _MAX_SESSION_BYTES
    with _lock:
        synchronize()
        if type(max_bytes) is not int or not 1<=max_bytes<=1024**3:
            raise ValueError('Graph cache budget must be an integer in 1..1 GiB bytes')
        if _smol_plan_depth:raise RuntimeError('Cannot resize the graph cache while building a SmolLM plan')
        if max_bytes<_session_blob_bytes():raise ValueError('Graph cache budget is below the cached graph bytes; increase it or restart')
        _MAX_SESSION_BYTES=max_bytes


@contextmanager
def _smol_graph_plan():
    """Do not retire a Smol graph family while its decode plan is being built."""
    global _smol_plan_depth
    _smol_plan_depth+=1
    try:yield
    finally:_smol_plan_depth-=1


def _make_session_room(incoming_bytes=0):
    """Drop an LRU cache reference; an executing call/plan retains its session."""
    global _session_evictions
    synchronize()
    if type(_MAX_SESSIONS) is not int or _MAX_SESSIONS<1:
        raise RuntimeError('NPU graph cache reached its limit; cache is full or disabled')
    if type(_MAX_SESSION_BYTES) is not int or type(incoming_bytes) is not int or not 0<=incoming_bytes<=_MAX_SESSION_BYTES or _MAX_SESSION_BYTES<1:
        raise RuntimeError('NPU graph exceeds the graph cache byte budget')
    while len(_sessions) >= _MAX_SESSIONS or _session_blob_bytes()+incoming_bytes>_MAX_SESSION_BYTES:
        key=next((key for key in _sessions if key in _rebuildable_sessions),None)
        if key is None:
            raise RuntimeError('NPU graph cache reached its limit; cache is full of pinned graphs or disabled')
        victims=[key]
        if isinstance(key,tuple) and key and key[0] in _SMOL_SESSION_KINDS:
            family=[other for other in _sessions if isinstance(other,tuple) and other and other[0] in _SMOL_SESSION_KINDS]
            if _smol_plan_depth or any(other not in _rebuildable_sessions for other in family):
                raise RuntimeError('SmolLM graph cache cannot retire a prepared/pinned family; increase the graph cache budget before retrying')
            # Partial teardown reproducibly invalidated surviving Smol graphs on
            # this Windows driver. Retire the family together, then rebuild cold.
            victims=family
        for victim in victims:
            session,program=_sessions.pop(victim)
            _rebuildable_sessions.remove(victim)
            _retired_sessions.add(session)
            # The last Python owner closes after synchronous submission returns.
            del session,program
            _session_evictions+=1


def _cache_program(key,program,*,rebuildable=False):
    """Only callers with a rebuild recipe opt into graph recycling."""
    global _session_creations,_transient_graph_imports
    with _lock:
        synchronize()
        if key in _sessions:
            _sessions.move_to_end(key)
            return _sessions[key]
        size=len(program.graph_blob)
        transient=rebuildable and type(_MAX_SESSION_BYTES) is int and size>_MAX_SESSION_BYTES
        if transient and isinstance(key,tuple) and key and key[0] in _SMOL_SESSION_KINDS:
            raise RuntimeError('SmolLM graph exceeds the graph cache budget; use set_graph_cache_budget before retrying')
        _make_session_room(0 if transient else size)
        session=_native.create_inference_session(program.graph_blob,timeout_ms=10_000,parent=_root_session)
        _session_creations+=1
        if transient:
            _retired_sessions.add(session)
            _transient_graph_imports+=1
            return session,program
        _sessions[key]=session,program
        if rebuildable:_rebuildable_sessions.add(key)
        return _sessions[key]


def _register_graph_factory(key,factory):
    """The caller retains its recipe through submission; the cache owns no model."""
    with _lock:
        synchronize()
        if not callable(factory):raise TypeError('Graph factory must be callable')
        _graph_factories[key]=factory
        if key in _sessions:_rebuildable_sessions.add(key)


def _session(key):
    global _graph_factory_calls
    with _lock:
        synchronize()
        factory=_graph_factories.get(key) if key not in _sessions else None
        if factory is not None:
            _make_session_room()
            _graph_factory_calls+=1
            return _cache_program(key,factory(),rebuildable=True)
        return _compile_session(key)


def _compile_session(key):
    if key not in _sessions:
        _make_session_room()
        kind, dtype, shape, *extra = key
        if kind == 'byte-pack':
            from npunlock.npu4000 import compile_byte_packet
            program = compile_byte_packet(extra[0], extra[1], libraries=_native)
        elif kind == 'stream-sum':
            program = compile_classification_rows(shape, _resident_kernel('stream-sum-f32'), libraries=_native)
        elif kind in ('typed-cast', 'typed-math', 'typed-reduce', 'dynamic-index', 'compact-index', 'index-offset', 'flash-step-f32', 'flash-backward-f32') or kind.startswith('math-'):
            program = compile_classification_rows(shape, _resident_kernel(kind), libraries=_native)
        elif kind == "halfcode_slice":
            program = compile_halfcode_slice(shape[0], libraries=_native)
        elif kind == "broadcast":
            program = compile_broadcast(shape, extra[0], dtype=dtype, libraries=_native)
        elif kind == "permute":
            program = compile_permute(shape, extra[0], dtype=dtype, libraries=_native)
        elif kind == "slice":
            program = compile_slice(shape[0], *extra, dtype=dtype, libraries=_native, timeout_ms=_SLICE_COMPILE_TIMEOUT_MS)
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
        elif kind in (*_MATH_UNARY, "relu", "relu_grad", "maximum",
                      "gelu-none", "gelu-tanh", "gelu-backward-none", "gelu-backward-tanh"):
            compile_fn = compile_binary if kind in ("relu_grad", "maximum", *_MATH_UNARY[:5]) or kind.startswith("gelu-") else compile_unary
            program = compile_fn(shape, _resident_kernel(f"{kind}-{dtype}"), dtype=dtype, libraries=_native)
        elif kind == "int64_add":
            program = compile_int64_counter(_resident_kernel("int64-add"), libraries=_native)
        elif kind in ("log-softmax", "log-softmax-backward", "softmax", "softmax-backward", "native-half-softmax", "sdpa-softmax", "nll-packet", "nll-backward", "layernorm-stats", "index-gather", "index-backward", "index-wrap-i64", "row-max-f32", "row-min-f32", "topk-f32", "pool-select-f32", "selection-backward-f32", "selection-backward-u32", "pool-dense-f32", "indexed-read-f32", "sort-f32", "order-stats-f32", "grid-stencil-f32"):
            name = kind+"-f32" if kind in ("layernorm-stats", "index-gather", "index-backward") else kind
            program = compile_classification_rows(shape, _resident_kernel(name), libraries=_native)
        elif kind == "layernorm-fma-test":
            program = compile_classification_rows(shape, _resident_kernel(kind+"-f32"), libraries=_native)
        elif kind == "class-pack":
            program = compile_classification_pack(*shape, libraries=_native)
        elif kind == 'mask-pack':
            program = compile_mask_packet(*shape, libraries=_native)
        elif kind == 'mask-eval':
            program = compile_classification_rows(shape, _resident_kernel('mask-packet'), libraries=_native)
        elif kind == 'rng-pack':
            program = compile_rng_packet(shape[0], libraries=_native)
        elif kind == 'rng-eval':
            program = compile_classification_rows(shape, _resident_kernel('rng-packet'), libraries=_native)
        elif kind == 'discrete-pack':
            program = compile_discrete_packet(*shape, libraries=_native)
        elif kind in ('integer-rng', 'multinomial'):
            program = compile_classification_rows(shape, _resident_kernel(kind+'-packet'), libraries=_native)
        elif kind == "class-columns":
            import numpy as np
            start, count = extra
            indices = (np.arange(shape[0], dtype=np.int64)[:, None]*shape[1] +
                       np.arange(start, start+count, dtype=np.int64)[None, :])
            program = compile_gather(prod(shape), indices, dtype="f32", libraries=_native)
        elif kind in ("sum", "mean"):
            program = compile_reduce(shape, extra[0], _resident_kernel(kind), kind=kind, libraries=_native)
        else:
            with bridge._lock:
                program = bridge._program(kind, dtype, shape)
        return _cache_program(key,program,rebuildable=True)
    else:
        _sessions.move_to_end(key)
    return _sessions[key]


def _run(key, inputs, shape, dtype, out=None):
    if out is None:
        out = torch.empty(shape, dtype=dtype, device="intel_npu:0")
    _validate(out, storage=True)
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
        _validate_float_read(x)
        inputs = [x]
        if y is not None:
            _validate_float_read(y)
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError("NPU arithmetic requires identical shapes and dtypes; broadcasting is not implemented")
            inputs.append(y)
        _check_out(out, tuple(x.shape), x.dtype, inputs)
        inputs = [_materialize(value) for value in inputs]
        if x.numel() > MAX_ELEMENTS:
            result=torch.empty(tuple(x.shape),dtype=x.dtype,device=x.device) if out is None else out
            sources=[value.view(-1) for value in inputs];destination=result.view(-1)
            for start in range(0,x.numel(),MAX_ELEMENTS):
                count=min(MAX_ELEMENTS,x.numel()-start)
                key=(kind,bridge._DTYPES[x.dtype][0],(count,))
                _run_ranges(key,[value.narrow(0,start,count) for value in sources],destination.narrow(0,start,count))
            return result
        if not x.numel():return torch.empty(tuple(x.shape),dtype=x.dtype,device=x.device) if out is None else out
        # Elementwise graphs depend on element count, not the logical view shape.
        shape=(x.numel(),)
        key=kind,bridge._DTYPES[x.dtype][0],shape
        return _run(key,inputs,tuple(x.shape),x.dtype,out)


def _check_out(out, shape, dtype, inputs):
    if out is not None:
        _validate(out, storage=True)
        if tuple(out.shape) != shape or out.dtype != dtype:
            raise ValueError("NPU out requires the exact result shape and dtype")
        if any(isinstance(x, torch.Tensor) and x.device.type == "intel_npu" and
               out.untyped_storage()._cdata == x.untyped_storage()._cdata for x in inputs):
            raise ValueError("NPU output may not alias an input")


def _run_ranges(key,inputs,out):
    session,program=_session(key)
    try:
        _C.run_ranges(session._result.session,inputs,[v.name for v in program.graph.inputs],out,program.graph.outputs[0].name)
    except RuntimeError as error:
        error.add_note(f'NPU range graph key: {key}')
        raise
    return out


def _range_copy(dst,src):
    """Internal page-aligned byte DMA; Tensor views keep both owners alive."""
    if not dst.is_contiguous() or not src.is_contiguous() or dst.nbytes!=src.nbytes or not src.numel():
        raise ValueError('NPU range copy requires matching nonempty contiguous byte spans')
    count=src.nbytes
    source=src.view(-1).view(torch.uint8);target=dst.view(-1).view(torch.uint8)
    _run_ranges(('broadcast','u8',(count,),(count,)),[source],target)
    return dst


def _slice(x, start, count):
    """Device DMA tile extraction; CPU only supplies shape/offset metadata."""
    if start < 0 or count < 0 or start+count > x.numel():
        raise ValueError('Invalid NPU contiguous slice range')
    if not count:return torch.empty((0,),dtype=x.dtype,device=x.device)
    if x.dtype not in (*bridge._DTYPES, torch.bool, torch.uint8):
        return _slice(x.view(-1).view(torch.uint8), start*x.element_size(), count*x.element_size()).view(x.dtype)
    if start == 0 and count == x.numel():
        return x.view(count)
    if x.is_contiguous() and (x.storage_offset()+start)*x.element_size()%4096==0:
        return _range_copy(torch.empty((count,),dtype=x.dtype,device=x.device),x.view(-1).narrow(0,start,count))
    if count > MAX_ELEMENTS:
        tiles = [_slice(x, start+offset, min(MAX_ELEMENTS, count-offset))
                 for offset in range(0, count, MAX_ELEMENTS)]
        return _concat(tiles, (count,), x.dtype)
    return _run(("slice", _dma_dtype(x.dtype), (x.numel(),), start, count),
                [x], (count,), x.dtype)


def _concat(tiles, shape, dtype, out=None):
    if sum(t.numel() for t in tiles) != prod(shape):
        raise ValueError("NPU tile assembly does not match the requested shape")
    if dtype not in (*bridge._DTYPES, torch.bool, torch.uint8):
        width = torch.empty((),dtype=dtype).element_size()
        result = _concat([t.view(-1).view(torch.uint8) for t in tiles], (prod(shape)*width,), torch.uint8).view(dtype).view(shape)
        return _copy(out,result) if out is not None else result
    tiles=[t for t in tiles if t.numel()]
    if not prod(shape):return torch.empty(shape,dtype=dtype,device='intel_npu') if out is None else out
    # The native argument limit includes the output. All assembly is byte DMA.
    while len(tiles) > 63:
        tiles = [_concat(tiles[i:i+63], (sum(t.numel() for t in tiles[i:i+63]),), dtype)
                 for i in range(0, len(tiles), 63)]
    if len(tiles) == 1:
        result = tiles[0].view(shape)
        return _copy(out, result) if out is not None else result
    sizes = tuple(t.numel() for t in tiles)
    result = _run(("matrix_concat", _dma_dtype(dtype), (prod(shape), 1), sizes),
                  [t.view(-1, 1) for t in tiles], (prod(shape), 1), dtype,
                  out.view(-1, 1) if out is not None else None)
    return out if out is not None else result.view(shape)


def _stream_write(result,cursor,pending,part):
    _validate(part,storage=True)
    if part.dtype!=result.dtype:raise ValueError('NPU streaming tile dtype mismatch')
    if not part.numel():return cursor,pending
    tile=_materialize(part).view(-1);dtype=result.dtype;page=4096//dtype.itemsize
    if pending is not None:tile=_concat([pending,tile],(pending.numel()+tile.numel(),),dtype)
    if cursor+tile.numel()>result.numel():raise ValueError('NPU streaming tiles exceed result shape')
    count=tile.numel()//page*page
    if count:_range_copy(result.view(-1).narrow(0,cursor,count),_slice(tile,0,count));cursor+=count
    return cursor,_slice(tile,count,tile.numel()-count) if count<tile.numel() else None


def _stream_finish(result,cursor,pending):
    tail=0 if pending is None else pending.numel()
    if cursor+tail!=result.numel():raise ValueError('NPU streaming tiles do not fill result shape')
    if tail:_range_copy(result.view(-1).narrow(0,cursor,tail),pending)
    return result


def _stream_concat(tiles, shape, dtype=torch.float32):
    """Assemble byte-exact results with one destination and a sub-page tail."""
    if len(shape)>MAX_RANK or any(type(n) is not int or n<0 for n in shape) or prod(shape)>MAX_DMA_ELEMENTS*4//dtype.itemsize:
        raise ValueError('NPU streaming result exceeds rank/storage bounds')
    result=torch.empty(shape,dtype=dtype,device='intel_npu')
    cursor=0;pending=None
    for part in tiles:
        cursor,pending=_stream_write(result,cursor,pending,part)
    return _stream_finish(result,cursor,pending)


def _cat_spec(tensors, dim, *, stack=False):
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError('NPU cat/stack requires at least one tensor')
    for x in tensors:
        _validate_read(x)
    shaped=tensors if stack else tuple(x for x in tensors if tuple(x.shape)!=(0,)) or (tensors[0],)
    first = shaped[0]
    rank = first.ndim+int(stack)
    dim = operator.index(dim)
    if not 1 <= rank <= MAX_RANK or not -rank <= dim < rank:
        raise ValueError('NPU cat/stack requires a valid axis and output rank 1-64')
    dim %= rank
    if any(x.ndim != first.ndim for x in shaped):
        raise ValueError('NPU cat/stack requires identical input rank')
    dtype=tensors[0].dtype
    for x in tensors[1:]:dtype=torch.promote_types(dtype,x.dtype)
    if stack:
        if any(x.shape != first.shape for x in tensors):
            raise ValueError('NPU stack requires identical input shapes')
        shape = tuple(first.shape[:dim])+(len(tensors),)+tuple(first.shape[dim:])
    else:
        if any(tuple(x.shape[:dim])+tuple(x.shape[dim+1:]) != tuple(first.shape[:dim])+tuple(first.shape[dim+1:]) for x in shaped):
            raise ValueError('NPU cat input shapes differ outside the concatenation axis')
        shape = tuple(first.shape[:dim])+(sum(x.shape[dim] for x in shaped),)+tuple(first.shape[dim+1:])
    limit = _storage_limit(dtype)
    if prod(shape) > limit:
        raise ValueError('NPU cat/stack output exceeds its dtype storage bound')
    return tensors, dim, shape


def _cat(tensors, dim=0, *, out=None):
    with _lock:
        synchronize()
        tensors, dim, shape = _cat_spec(tensors, dim)
        dtype = tensors[0].dtype
        for x in tensors[1:]:dtype=torch.promote_types(dtype,x.dtype)
        _check_out(out, shape, dtype, tensors)
        if not prod(shape):return torch.empty(shape,dtype=dtype,device=tensors[0].device) if out is None else out
        order = (dim,)+tuple(a for a in range(len(shape)) if a != dim)
        values = [_cast(_materialize(x.permute(order)),dtype) for x in tensors if tuple(x.shape)!=(0,)]
        byte_storage = dtype not in (*bridge._DTYPES, torch.bool)
        raw = [x.view(-1).view(torch.uint8) for x in values] if byte_storage else values
        flat = _concat(raw, (sum(x.numel() for x in raw),), raw[0].dtype)
        flat = flat.view(dtype) if byte_storage else flat
        front = flat.view(tuple(shape[a] for a in order))
        result = _materialize(front.permute(tuple(order.index(a) for a in range(len(shape)))))
        if out is not None:
            return _copy(out, result)
        # Even cat([x]) must own storage rather than return an alias.
        return _clone(result) if any(result.untyped_storage()._cdata == x.untyped_storage()._cdata for x in tensors) else result


def _stack(tensors, dim=0, *, out=None):
    with _lock:
        synchronize()
        tensors, dim, shape = _cat_spec(tensors, dim, stack=True)
        dtype=tensors[0].dtype
        for x in tensors[1:]:dtype=torch.promote_types(dtype,x.dtype)
        _check_out(out, shape, dtype, tensors)
        values = [_materialize(x).view(tuple(x.shape[:dim])+(1,)+tuple(x.shape[dim:])) for x in tensors]
        return _cat(values, dim, out=out)


def _layout_gradient(gradient, input_sizes, dim, start, end, step, *, select=False, out=None):
    """Inverse slice/select by byte Gather, with -1 mapping to exact zero."""
    with _lock:
        synchronize()
        _validate_read(gradient)
        shape = tuple(operator.index(n) for n in input_sizes)
        dim, step = operator.index(dim), operator.index(step)
        if gradient.dtype not in bridge._DTYPES or not 1 <= len(shape) <= MAX_RANK or any(n < 0 for n in shape) or prod(shape) > MAX_TENSOR_ELEMENTS:
            raise ValueError('NPU slice/select backward requires bounded FP16/FP32 shapes')
        if not -len(shape) <= dim < len(shape) or step <= 0:
            raise ValueError('Invalid NPU slice/select backward axis or step')
        dim %= len(shape)
        if select:
            start = operator.index(start)
            if not -shape[dim] <= start < shape[dim]:
                raise IndexError('NPU select backward index is out of range')
            start %= shape[dim]
            end = start+1
        else:
            start, end, step = slice(start, end, step).indices(shape[dim])
        length = len(range(start, end, step))
        expected = shape[:dim]+(() if select else (length,))+shape[dim+1:]
        if tuple(gradient.shape) != expected:
            raise ValueError('NPU slice/select backward gradient shape differs')
        _check_out(out, shape, gradient.dtype, [gradient])
        if not length or not prod(shape):
            return _fill(torch.empty(shape,dtype=gradient.dtype,device=gradient.device) if out is None else out,0.)
        values = _materialize(gradient)
        if select:
            values = values.view(shape[:dim]+(1,)+shape[dim+1:])
        order = (dim,)+tuple(a for a in range(len(shape)) if a != dim)
        values = _materialize(values.permute(order))
        import numpy as np
        width = prod(shape[a] for a in order[1:])
        indices = np.full((shape[dim], width), -1, dtype=np.int64)
        indices[start:end:step] = np.arange(length*width, dtype=np.int64).reshape(length, width)
        result = _gather_window(values.view(-1), indices, 1).view(tuple(shape[a] for a in order))
        return _materialize(result.permute(tuple(order.index(a) for a in range(len(shape)))), out=out)


def _slice_backward(gradient, input_sizes, dim, start, end, step, *, out=None):
    return _layout_gradient(gradient, input_sizes, dim, start, end, step, out=out)


def _select_backward(gradient, input_sizes, dim, index, *, out=None):
    return _layout_gradient(gradient, input_sizes, dim, index, None, 1, select=True, out=out)


def _narrow_copy_out(x, dim, start, length, *, out):
    with _lock:
        synchronize()
        _validate_read(x)
        view = x.narrow(dim, start, length)
        _validate_read(view)
        _check_out(out, tuple(view.shape), x.dtype, [x])
        return _copy(out, view)


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
            if not 1<=len(shape)<=4 or any(type(v) is not int or v<1 for v in shape) or sum(counts)!=prod(shape) or prod(shape)>MAX_DMA_ELEMENTS:
                raise ValueError('NPU multi-concat output shape differs or exceeds storage bound')
        key=('concat_many',bridge._DTYPES[dtype][0],tuple(sum(group) for group in sizes),sizes)
        return _run_many(key,[value for group in groups for value in group],[(shape,dtype) for shape in shapes],outs)


def _map_tiles(function, inputs, shape, dtype, out=None):
    _check_out(out, shape, dtype, inputs)
    result=torch.empty(shape,dtype=dtype,device='intel_npu') if out is None else out
    for start in range(0,prod(shape),MAX_ELEMENTS):
        count=min(MAX_ELEMENTS,prod(shape)-start)
        tile=function(*[_slice(x,start,count) for x in inputs])
        _range_copy(result.view(-1).narrow(0,start,count),tile)
    return result


def _scalar(value, dtype):
    # Scalar constants are host arguments, not staged device Tensor payloads.
    host = torch.tensor(value, dtype=dtype)
    key = dtype, host.reshape(-1).view(torch.uint8).numpy().tobytes()
    if key not in _scalars:
        _scalars[key] = host.to("intel_npu")
        # ponytail: 128 tiny cached constants; embed graph constants if measured useful.
        if len(_scalars) > 128:
            _scalars.popitem(last=False)
    else:
        _scalars.move_to_end(key)
    return _scalars[key]


def _broadcast(x, shape):
    _validate(x, storage=True)
    shape = tuple(shape)
    if len(shape) > MAX_RANK or not 0 <= prod(shape) <= _storage_limit(x.dtype) or tuple(torch.broadcast_shapes(x.shape, shape)) != shape:
        raise ValueError("Invalid NPU broadcast shape")
    if tuple(x.shape) == shape:
        return x
    if x.numel() == prod(shape):
        return x.view(shape)
    if x.dtype not in (*bridge._DTYPES, torch.bool) or len(shape)>4 or not prod(shape):
        return _materialize(x.expand(shape))
    return _run(("broadcast", _dma_dtype(x.dtype), tuple(x.shape) or (1,), shape),
                [x], shape, x.dtype)


_MASK_CODES = {torch.float32: 0, torch.float16: 1, torch.int64: 2, torch.bool: 3}
_COMPARISONS = ('eq', 'ne', 'lt', 'le', 'gt', 'ge')


def _check_generator(generator):
    if generator is not None and (not isinstance(generator, torch.Generator) or generator.device != torch.device('intel_npu:0')):
        raise ValueError('NPU RNG requires an intel_npu:0 Generator')


def _reserve_rng(count, generator=None):
    _check_generator(generator)
    return _C.reserve_generator(_rng if generator is None else generator, count)


def _random_values(shape, dtype, normal=False, generator=None):
    if not prod(shape):
        _check_generator(generator)
        return torch.empty(shape,dtype=torch.float32,device='intel_npu')
    seed, offset = _reserve_rng(prod(shape), generator)
    tiles = []
    for start in range(0, prod(shape), _GATHER_TILE_ELEMENTS):
        count = min(_GATHER_TILE_ELEMENTS, prod(shape)-start)
        counter = offset+start
        header = struct.pack('<8I', seed & 0xffffffff, seed >> 32, counter & 0xffffffff,
                             counter >> 32, count, int(normal), int(dtype == torch.float16), 0)
        metadata = torch.frombuffer(bytearray(header), dtype=torch.float32).to('intel_npu').view(1,8)
        words = rng_packet_words(count)
        packet = _run(('rng-pack','f32',(count,)), [metadata], (1,words), torch.float32)
        raw = _run(('rng-eval','f32',(1,words)), [packet,packet], (1,words), torch.float32)
        tiles.append(_slice(raw.view(-1), 0, count))
    return _concat(tiles, shape, torch.float32)


def _random_fill(x, a=0., b=1., *, generator=None, kind='uniform'):
    with _lock:
        synchronize()
        _validate(x, dense=True)
        _check_generator(generator)
        a, b = float(a), float(b)
        if not isfinite(a) or not isfinite(b) or kind == 'normal' and b < 0:
            raise ValueError('NPU RNG requires finite parameters and nonnegative normal std')
        if kind == 'uniform':
            limit = torch.finfo(x.dtype).max
            if a > b or max(abs(a),abs(b)) > limit or b-a > torch.finfo(torch.float32).max:
                raise ValueError('NPU uniform requires ordered bounds representable in its floating dtype')
            # PyTorch converts uniform bounds to the destination floating dtype.
            low, high = (torch.tensor(v, dtype=x.dtype, device='cpu').item() for v in (a,b))
            previous = torch.nextafter(torch.tensor(high,dtype=x.dtype,device='cpu'),
                                       torch.tensor(float('-inf'),dtype=x.dtype,device='cpu')).item()
        elif kind != 'normal' or max(abs(a),abs(b)) > torch.finfo(torch.float32).max:
            raise ValueError('Unsupported NPU random distribution/parameter range')
        with torch.no_grad():
            result = _random_values(tuple(x.shape), x.dtype, kind == 'normal', generator)
            factor, shift = (b,a) if kind == 'normal' else (high-low,low)
            if factor != 1: result = _binary_op(result,factor,kind='mul')
            if shift != 0: result = _binary_op(result,shift,kind='add')
            result = _cast(result,x.dtype)
            if kind == 'uniform' and high != low and (a != 0. or b != 1.):
                result = _where(_compare(result,high,kind='lt'),result,previous)
            return _copy(x,result)


def _discrete_packet(mode, rows, columns=0, samples=1, replacement=False, *, seed=0, offset=0, bounds=(0,0,0), weights=None):
    low, span, threshold = bounds
    values = (seed & 0xffffffff, seed >> 32, offset & 0xffffffff, offset >> 32,
              mode, rows, columns, samples, int(replacement), low & 0xffffffff, low >> 32,
              span & 0xffffffff, span >> 32, threshold & 0xffffffff, threshold >> 32, 0)
    metadata = torch.frombuffer(bytearray(struct.pack('<16I', *values)), dtype=torch.float32).to('intel_npu').view(1,16)
    words = discrete_packet_words(rows, columns, samples)
    inputs = [metadata] if weights is None else [metadata, weights]
    packet = _run(('discrete-pack','f32',(rows,columns,samples)), inputs, (1,words), torch.float32)
    return _run(('integer-rng' if mode == 0 else 'multinomial','f32',(1,words)),
                [packet,packet], (1,words), torch.float32)


def _random_integer_fill(x, low=0, high=None, *, generator=None):
    with _lock, torch.no_grad():
        synchronize()
        _validate(x, storage=True)
        if x.dtype != torch.int64:
            raise ValueError('NPU discrete random_ currently requires contiguous INT64 storage')
        _check_generator(generator)
        bounds = integer_range(low, high)
        seed, offset = _reserve_rng(x.numel(), generator)
        tiles = []
        for start in range(0, x.numel(), _GATHER_TILE_ELEMENTS):
            count = min(_GATHER_TILE_ELEMENTS, x.numel()-start)
            raw = _discrete_packet(0, count, seed=seed, offset=offset+start, bounds=bounds)
            tiles.append(_slice(raw.view(-1), 0, count*2))
        result = _concat(tiles, (x.numel()*2,), torch.float32).view(torch.int64).view(x.shape)
        return _copy(x, result)


def _resize_same(x, size, *, memory_format=None):
    with _lock:
        synchronize()
        _validate(x, storage=True)
        if tuple(size) != tuple(x.shape) or memory_format not in (None, torch.contiguous_format):
            raise ValueError('NPU out factories require the existing exact shape and contiguous storage; resize is unsupported')
        return x


def _multinomial(x, num_samples, replacement=False, *, generator=None, out=None):
    with _lock, torch.no_grad():
        synchronize()
        _validate_float_read(x)
        count = operator.index(num_samples)
        _check_generator(generator)
        if x.ndim not in (1,2) or count < 1:
            raise ValueError('NPU multinomial requires 1D/2D FP16/FP32 weights and positive samples')
        rows, columns = (1,x.shape[0]) if x.ndim == 1 else tuple(x.shape)
        shape = (count,) if x.ndim == 1 else (rows,count)
        if rows*count > MAX_INT64_ELEMENTS or rows*columns*count > MAX_TENSOR_ELEMENTS:
            raise ValueError('NPU multinomial output/work exceeds its bounded storage/scan limit')
        if not replacement and count > columns:
            raise ValueError('NPU multinomial without replacement requires samples <= categories')
        batch = min(rows, (MAX_ELEMENTS-16)//columns, (MAX_ELEMENTS-2)//(columns+2*count))
        if batch < 1:
            raise ValueError('NPU multinomial cannot fit a complete category row and output in its packet')
        _check_out(out, shape, torch.int64, (x,))
        values = _cast(_materialize(x), torch.float32).view(rows,columns)
        groups = [(start,min(batch,rows-start)) for start in range(0,rows,batch)]
        weights = [_slice(values.view(-1),start*columns,n*columns).view(n,columns) for start,n in groups]
        # All rows are validated before consuming RNG or writing the caller's out.
        for (_, n), data in zip(groups,weights):
            raw = _discrete_packet(1,n,columns,count,replacement,weights=data)
            flag = _slice(raw.view(-1),raw.numel()-1,1).item()
            if flag != 0:
                raise ValueError('NPU multinomial needs finite nonnegative weights with positive row mass; without replacement needs enough positive categories')
        seed, offset = _reserve_rng(rows*count, generator)
        tiles = []
        for (start,n), data in zip(groups,weights):
            raw = _discrete_packet(2,n,columns,count,replacement,seed=seed,offset=offset+start*count,weights=data)
            tiles.append(_slice(raw.view(-1),0,n*count*2))
        result = _concat(tiles, (rows*count*2,), torch.float32).view(torch.int64).view(shape)
        return _copy(out,result) if out is not None else result


def _bernoulli(x, p=None, *, generator=None, out=None, inplace=False):
    with _lock, torch.no_grad():
        synchronize()
        _validate_read(x)
        if inplace:_validate(x,dense=True,storage=True)
        _check_generator(generator)
        shape=tuple(x.shape);dtype=out.dtype if out is not None else x.dtype;p=x if p is None else p
        if x.numel()>MAX_DMA_ELEMENTS:
            raise ValueError('NPU Bernoulli FP32 workspace exceeds its storage bound')
        _check_out(out,shape,dtype,(x,p))
        if isinstance(p,torch.Tensor):
            _validate_read(p)
            if p.dtype not in (torch.float16,torch.float32,torch.bfloat16) or tuple(torch.broadcast_shapes(p.shape,shape))!=shape:
                raise ValueError('NPU Bernoulli needs broadcastable floating device probabilities')
            p=_broadcast(_cast(_materialize(p),torch.float32),shape)
            valid=_logical(_compare(p,0.,kind='ge'),_compare(p,1.,kind='le'),kind='and')
            # Only the validation flag reaches the CPU; reject before RNG consumption or output writes.
            if not _boolean_reduce(valid,kind='all').item():
                raise ValueError('NPU Bernoulli probabilities must be finite and in [0,1]')
        else:
            p=float(p)
            if not 0<=p<=1:raise ValueError('NPU scalar Bernoulli requires p in [0,1]')
        if not x.numel():return x if inplace else out if out is not None else torch.empty(shape,dtype=dtype,device=x.device)
        samples=_random_values(shape,torch.float32,generator=generator)
        result=_cast(_compare(samples,p,kind='lt'),dtype)
        return _copy(x if inplace else out,result) if inplace or out is not None else result


def _bernoulli_fill(x, p=.5, *, generator=None):
    return _bernoulli(x,p,generator=generator,inplace=True)


def _randperm(n, *, generator=None, dtype=None, layout=None, device=None, pin_memory=None, out=None):
    with _lock, torch.no_grad():
        synchronize()
        _check_generator(generator)
        dtype=out.dtype if out is not None else torch.int64 if dtype is None else dtype
        device=torch.device(out.device if out is not None else 'intel_npu' if device is None else device)
        if device.type!='intel_npu' or device.index not in (None,0) or layout not in (None,torch.strided) or pin_memory:
            raise ValueError('NPU randperm requires strided intel_npu:0 output without pinned memory')
        n=permutation_size(n,dtype)
        _check_out(out,(n,),dtype,())
        if n<2:
            result=torch.zeros((n,),dtype=dtype,device=device)
        else:
            from torch_npu_sort import sort
            keys=torch.empty((n,),dtype=torch.int64,device=device)
            for _ in range(8):
                _random_integer_fill(keys,-(1<<63),generator=generator)
                ordered,result=sort(keys)
                duplicate=_compare(_materialize(ordered[1:]),_materialize(ordered[:-1]),kind='eq')
                if not _boolean_reduce(duplicate,kind='any').item():break
            else:raise RuntimeError('NPU randperm exhausted eight distinct-key attempts')
            # Redraw all keys on collision: accepting stable ties would bias the permutation.
            result=_cast(result,dtype)
        return _copy(out,result) if out is not None else result


def _dropout_spec(x,p,train):
    _validate_read(x)
    if x.dtype not in bridge._DTYPES or not 0 <= p <= 1 or train not in (None,True,False):
        raise ValueError('NPU Dropout requires FP16/FP32, p in [0,1] and a boolean training flag')


def _native_dropout(x,p,train=True):
    with _lock, torch.no_grad():
        synchronize()
        _dropout_spec(x,p,train)
        values = _materialize(x)
        if train is False or p == 0 or p == 1:
            mask = torch.empty(tuple(x.shape),device=x.device,dtype=torch.bool).fill_(train is False or p != 1)
            result = _clone(values) if train is False or p == 0 else _binary_op(values,0.,kind='mul')
        else:
            samples = _random_values(tuple(x.shape),torch.float32)
            mask = _compare(samples,float(p),kind='ge')
            result = _native_dropout_backward(values,mask,1./(1.-p))
        return result, mask


def _native_dropout_backward(gradient,mask,scale):
    with _lock, torch.no_grad():
        _validate_read(gradient)
        _validate_read(mask)
        if gradient.dtype not in bridge._DTYPES or mask.dtype != torch.bool or mask.shape != gradient.shape or not isfinite(scale):
            raise ValueError('NPU Dropout backward requires a matching BOOL mask and finite scale')
        g = _cast(_materialize(gradient),torch.float32)
        m = _mask_cast(_materialize(mask),torch.float32)
        result = _binary_op(_binary_op(g,m,kind='mul'),scale,kind='mul')
        return _cast(result,gradient.dtype)


def _dropout(x,p=.5,train=True):
    with _lock, torch.no_grad():
        _dropout_spec(x,p,train)
        return x if train is False or p == 0 else _native_dropout(x,p,train)[0]


def _storage_slice(x, start, count):
    if x.dtype == torch.int64:
        return _slice(x.view(-1).view(torch.float32), start*2, count*2).view(torch.int64)
    return _slice(x, start, count)


def _mask_compute(inputs, kind, code, shape, dtype, out=None):
    """Complete bounded packets; only static header/shape metadata is on CPU."""
    tiles = []
    for start in range(0, prod(shape), _GATHER_TILE_ELEMENTS):
        count = min(_GATHER_TILE_ELEMENTS, prod(shape)-start)
        values = [_storage_slice(x, start, count) for x in inputs]
        sizes, _, words = mask_packet_spec(count, kind, code)
        if tuple(x.numel()*x.element_size() for x in values) != sizes:
            raise ValueError('NPU mask packet payload differs from its ABI')
        packet = _run(('mask-pack','f32',(count,kind,code)), values, (1,words), torch.float32)
        raw = _run(('mask-eval','f32',(1,words)), [packet,packet], (1,words), torch.float32)
        if dtype == torch.int64:
            result = _slice(raw.view(-1), 0, count*2).view(torch.int64)
        else:
            result = _slice(raw.view(-1).view(dtype), 0, count)
        tiles.append(result)
    if dtype == torch.int64:
        result = _concat([x.view(torch.float32) for x in tiles], (prod(shape)*2,), torch.float32).view(dtype).view(shape)
    else:
        result = _concat(tiles, shape, dtype)
    return _copy(out, result) if out is not None else result


def _mask_cast(x, dtype):
    _validate(x, storage=True)
    if dtype not in _STORAGE_DTYPES or torch.bool not in (x.dtype, dtype):
        raise ValueError('NPU mask casts require BOOL and a supported storage dtype')
    if x.dtype == dtype:
        return x
    if dtype == torch.int64 and x.numel() > MAX_INT64_ELEMENTS:
        raise ValueError('BOOL to INT64 cast exceeds the INT64 storage bound')
    kind, code = (8, _MASK_CODES[dtype]) if x.dtype == torch.bool else (7, _MASK_CODES[x.dtype])
    return _mask_compute([x], kind, code, tuple(x.shape), dtype)


def _mask_operand(value, dtype, shape):
    if isinstance(value, torch.Tensor) and value.device.type == 'intel_npu':
        result = _materialize(value)
        result = _cast(result, dtype) if result.dtype != dtype else result
    else:
        result = _scalar(value.item() if isinstance(value, torch.Tensor) else value, dtype)
    return _broadcast(result, shape)


def _mask_operands_spec(values):
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.device.type == 'intel_npu':
                _validate_read(value)
            elif value.device.type != 'cpu' or value.ndim or value.requires_grad or value.dtype not in _STORAGE_DTYPES:
                raise ValueError('NPU mask operations require device tensors or non-grad CPU scalar constants')
        elif type(value) not in (int, float, bool):
            raise ValueError('NPU mask operands must be supported tensors or real scalars')


def _compare(x, y, *, kind, out=None):
    with _lock, torch.no_grad():
        synchronize()
        _mask_operands_spec((x,y))
        if not any(isinstance(v,torch.Tensor) and v.device.type == 'intel_npu' for v in (x,y)):
            raise ValueError('NPU comparison requires a device operand')
        dtype = torch.result_type(x,y)
        if dtype not in _STORAGE_DTYPES or any(isinstance(v,torch.Tensor) and v.dtype == torch.int64 for v in (x,y)) and dtype in bridge._DTYPES:
            raise ValueError('NPU comparison does not mix INT64 with floating-point operands')
        shape = tuple(torch.broadcast_shapes(*(tuple(v.shape) if isinstance(v,torch.Tensor) else () for v in (x,y))))
        if len(shape)>MAX_RANK or not 0<=prod(shape)<=_storage_limit(dtype):
            raise ValueError('NPU comparison broadcast exceeds its dtype/rank bound')
        _check_out(out, shape, torch.bool, (x,y))
        inputs = [_mask_operand(v,dtype,shape) for v in (x,y)]
        return _mask_compute(inputs, _COMPARISONS.index(kind), _MASK_CODES[dtype], shape, torch.bool, out)


def _logical(x, y=None, *, kind, out=None):
    with _lock:
        synchronize()
        inputs = [x] if y is None else [x,y]
        for v in inputs:
            _validate_read(v)
            if v.dtype != torch.bool:
                raise ValueError('NPU logical operations currently require BOOL device tensors')
        shape = tuple(torch.broadcast_shapes(*(v.shape for v in inputs)))
        if len(shape)>MAX_RANK or not 0<=prod(shape)<=_storage_limit(torch.bool):
            raise ValueError('NPU logical output exceeds its storage bound')
        _check_out(out,shape,torch.bool,inputs)
        return _mask_compute([_broadcast(_materialize(v),shape) for v in inputs],
                             {'not':9,'and':10,'or':11,'xor':12}[kind],3,shape,torch.bool,out)


def _where_spec(condition, x, y, out=None):
    _validate_read(condition)
    if condition.dtype != torch.bool:
        raise ValueError('NPU where requires a BOOL condition')
    _mask_operands_spec((x,y))
    dtype = torch.result_type(x,y)
    if dtype not in _STORAGE_DTYPES or any(isinstance(v,torch.Tensor) and v.dtype==torch.int64 for v in (x,y)) and dtype in bridge._DTYPES:
        raise ValueError('NPU where requires supported result dtype without INT64/float promotion')
    shape = tuple(torch.broadcast_shapes(condition.shape,*(tuple(v.shape) if isinstance(v,torch.Tensor) else () for v in (x,y))))
    if len(shape)>MAX_RANK or not 0<=prod(shape)<=_storage_limit(dtype):
        raise ValueError('NPU where broadcast exceeds its dtype/rank bound')
    _check_out(out,shape,dtype,(condition,x,y))
    return shape,dtype


def _where(condition, x, y, *, out=None):
    with _lock:
        synchronize()
        shape,dtype = _where_spec(condition,x,y,out)
        values = [_broadcast(_materialize(condition),shape),*[_mask_operand(v,dtype,shape) for v in (x,y)]]
        return _mask_compute(values,6,_MASK_CODES[dtype],shape,dtype,out)


def _masked_fill_spec(x, mask, value):
    _validate_read(x)
    if isinstance(value,torch.Tensor):
        if value.ndim or value.dtype != x.dtype:
            raise ValueError('NPU masked_fill Tensor value must be scalar and match input dtype')
    else:
        # Match scalar overflow validation without inspecting device payloads.
        torch.empty((),dtype=x.dtype,device='cpu').masked_fill(torch.tensor(True,device='cpu'),value)
    _mask_operands_spec((value,))
    shape,_ = _where_spec(mask,x,x)
    return shape


def _masked_fill(x, mask, value):
    with _lock:
        _masked_fill_spec(x,mask,value)
        constant = value if isinstance(value,torch.Tensor) else _scalar(value,x.dtype)
        return _where(mask,constant,x)


def _masked_fill_inplace(x, mask, value):
    with _lock, torch.no_grad():
        synchronize()
        from torch_npu_compat_extended import _writable
        _writable(x)
        if _masked_fill_spec(x,mask,value) != tuple(x.shape):
            raise ValueError('NPU masked_fill_ requires an unchanged shape and writable storage')
        constant = value if isinstance(value,torch.Tensor) else _scalar(value,x.dtype)
        return _copy(x,_where(mask,constant,x))


def _masked_fill_out(x, mask, value, *, out):
    with _lock:
        shape = _masked_fill_spec(x,mask,value)
        _check_out(out,shape,x.dtype,(x,mask,value))
        return _copy(out,_masked_fill(x,mask,value))


def _binary_op(x, y, *, kind, out=None, alpha=1):
    with _lock:
        synchronize()
        if type(alpha) not in (int, float) or isfinite(alpha) and abs(alpha) > 3.4028234663852886e38:
            raise ValueError("NPU add/sub alpha must be real and representable in FP32")
        tensors = [v for v in (x, y) if isinstance(v, torch.Tensor)]
        if not tensors or not any(v.device.type == "intel_npu" for v in tensors):
            raise ValueError("NPU arithmetic requires an NPU Tensor")
        for value in (x, y):
            if isinstance(value, torch.Tensor):
                if value.device.type == "intel_npu":
                    _validate_float_read(value)
                elif value.device.type != "cpu" or value.ndim or value.requires_grad or value.is_complex():
                    raise ValueError("CPU arithmetic operands must be real scalar tensors without autograd")
            elif type(value) not in (int, float, bool):
                raise TypeError("NPU operand must be a Tensor or a real scalar")
        if kind == "sub" and any(type(v) is bool or isinstance(v, torch.Tensor) and v.dtype == torch.bool for v in (x, y)):
            raise ValueError("Torch subtraction does not support bool operands")
        shape = tuple(torch.broadcast_shapes(*(tuple(v.shape) if isinstance(v, torch.Tensor) else () for v in (x, y))))
        if len(shape) > MAX_RANK or not 0 <= prod(shape) <= MAX_DMA_ELEMENTS:
            raise ValueError("NPU broadcast result exceeds its rank/storage bound")
        dtype = torch.result_type(x, y)
        if dtype not in bridge._DTYPES:
            raise ValueError("NPU arithmetic result dtype must be FP16/FP32")
        if dtype == torch.float16 and isfinite(alpha) and abs(alpha) > 65504:
            raise ValueError("NPU add/sub alpha must be representable in FP16")
        _check_out(out, shape, dtype, (x, y))
        # CPU Torch's wrapped-scalar FP16 mul/div use FP32 opmath scalars;
        # add/sub round their scalar input to FP16 first. Preserve this boundary.
        work_dtype = torch.float32 if dtype == torch.float16 and kind in ("mul", "div") and (
            not isinstance(y, torch.Tensor) or y.ndim == 0 and y.dtype != torch.float16) else dtype
        if alpha != 1:
            work_dtype = torch.float32
        operands = []
        for value in (x, y):
            if isinstance(value, torch.Tensor) and value.device.type == "intel_npu":
                value = _cast(_materialize(value), work_dtype)
            else:
                scalar_dtype = dtype if kind in ("add", "sub") else work_dtype
                value = _cast(_scalar(value.item() if isinstance(value, torch.Tensor) else value, scalar_dtype), work_dtype)
            operands.append(_broadcast(value, shape))
        if alpha != 1:
            scale = _cast(_scalar(alpha, dtype), work_dtype)
            operands[1] = _execute("mul", operands[1], _broadcast(scale, shape))
        result = _execute(kind, *operands, out=out if dtype == work_dtype else None)
        return _cast(result, dtype, out=out)


def _matmul(x, y, *, out=None):
    with _lock:
        synchronize()
        m, k, n = _matrix_shape(x, y)
        _check_out(out, (m, n), x.dtype, (x, y))
        result = _matrix_fp32(x, y, out=out if out is not None and out.dtype == torch.float32 else None)
        return _cast(result, x.dtype, out=out)


def _matmul_promote(x, y):
    return x.view(1, -1) if x.ndim == 1 else x, y.view(-1, 1) if y.ndim == 1 else y


def _matmul_nd_spec(x, y):
    _validate_float_read(x)
    _validate_float_read(y)
    if not 1 <= x.ndim <= MAX_RANK or not 1 <= y.ndim <= MAX_RANK or x.dtype != y.dtype:
        raise ValueError('NPU matmul requires matching-dtype rank 1-64 tensors with compatible inner dimensions')
    a, b = _matmul_promote(x, y)
    if a.shape[-1] != b.shape[-2]:
        raise ValueError('NPU matmul requires compatible inner dimensions')
    batch = tuple(torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
    left, right = batch+tuple(a.shape[-2:]), batch+tuple(b.shape[-2:])
    shape = batch + (() if x.ndim == 1 else (a.shape[-2],)) + (() if y.ndim == 1 else (b.shape[-1],))
    if any(prod(s) > MAX_DMA_ELEMENTS for s in (left, right, shape)):
        raise ValueError('NPU matmul output and broadcast operands must each fit the storage bound')
    return left, right, shape


def _matmul_nd(x, y, *, out=None):
    with _lock:
        synchronize()
        left_shape, right_shape, shape = _matmul_nd_spec(x, y)
        _check_out(out, shape, x.dtype, (x, y))
        if not prod(shape):
            return torch.empty(shape, dtype=x.dtype, device=x.device) if out is None else out
        if not x.shape[-1]:
            result = torch.empty(shape,dtype=x.dtype,device=x.device) if out is None else out
            return _fill(result,0.)
        x, y = (_materialize(v) for v in (x, y))
        x, y = _matmul_promote(x, y)
        if x.ndim == y.ndim == 2:
            target = out.view(x.shape[-2], y.shape[-1]) if out is not None else None
            result = _matmul(x, y, out=target).view(shape)
            return out if out is not None else result
        left = _broadcast(_materialize(x), left_shape).view(-1, *left_shape[-2:])
        right = _broadcast(_materialize(y), right_shape).view(-1, *right_shape[-2:])
        result = _bmm(left, right).view(shape)
        return _copy(out, result) if out is not None else result


def _vector_product(x, y, *, kind, out=None):
    if (x.ndim, y.ndim) != ((1, 1) if kind == 'dot' else (2, 1)):
        raise ValueError(f'NPU {kind} requires rank {(1, 1) if kind == "dot" else (2, 1)} operands')
    return _matmul_nd(x, y, out=out)


def _bmm_spec(x, y):
    _validate_float_read(x)
    _validate_float_read(y)
    if (x.ndim != 3 or y.ndim != 3 or x.dtype != y.dtype or
            x.shape[0] != y.shape[0] or x.shape[2] != y.shape[1]):
        raise ValueError("NPU bmm requires matching-dtype 3D matrices with equal batches and compatible inner dimensions")
    batch, m, k = x.shape
    n = y.shape[2]
    if batch*m*n > MAX_DMA_ELEMENTS:
        raise ValueError("NPU bmm output exceeds the storage bound")
    return batch, m, k, n


def _index_limits(rows, width, count):
    size = 6 + 2*count + max(rows, count)*width
    if size > MAX_ELEMENTS or rows*width*count > 1048576:
        raise ValueError('NPU index packet requires 6+2*L+max(R,L)*D <= 65536 and R*D*L <= 1048576')
    return size


def _index_select_spec(x, dim, indices):
    _validate(x, dense=True)
    _validate(indices, storage=True)
    if (not 1 <= x.ndim <= 4 or type(dim) is not int or not -x.ndim <= dim < x.ndim or
            indices.dtype != torch.int64 or indices.ndim != 1):
        raise ValueError('NPU index_select requires rank 1-4 data, a valid dim and contiguous 1D INT64 device indices')
    dim %= x.ndim
    rows, count = x.shape[dim], indices.numel()
    width = prod(x.shape[:dim]+x.shape[dim+1:])
    _index_storage_bound(rows, width, count)
    shape = list(x.shape)
    shape[dim] = count
    return dim, rows, width, tuple(shape)


def _index_storage_bound(rows, width, count):
    if any(type(n) is not int or n<1 for n in (rows,width,count)):
        raise ValueError('NPU indexed dimensions must be positive integers')
    if max(rows,count)>MAX_TENSOR_ELEMENTS or max(rows,count)*width>MAX_DMA_ELEMENTS:
        raise ValueError('NPU indexed dimensions exceed metadata or tensor storage bounds')


def _index_large(data, indices, rows, width, *, backward=False, padding_idx=-1, round_half=False, initial=None):
    from torch_npu_compat_extended import index_select, normalize, integer_binary
    if not backward:
        return index_select(data.view(rows,width),0,indices).view(-1)
    if _boolean_reduce(_compare(indices,0,kind='lt'),kind='any').item():
        raise IndexError('Negative NPU index_select/embedding index')
    ids=normalize(indices,rows)
    columns=torch.arange(width,dtype=torch.int64).to(data.device)
    offsets=integer_binary(integer_binary(ids.view(-1,1),width,'mul'),columns.view(1,-1),'add').view(-1)
    source=data.view(-1,width)
    if padding_idx>=0:
        source=_where(_compare(ids,padding_idx,kind='eq').view(-1,1),0.,source)
    # Embedding rounds each Half update; index_select/scatter accumulate in FP32.
    if not round_half:source=_cast(_materialize(source),torch.float32)
    result=_selection_backward(source.view(-1),offsets,(rows,width),offsets.numel(),rows*width,
                               initial_value=initial)
    return _cast(result,data.dtype).view(-1)


def _index_packet(data, indices, rows, width, *, backward=False, padding_idx=-1, round_half=False, initial=None):
    count = indices.numel()
    _index_storage_bound(rows,width,count)
    if initial is not None:
        _validate(initial)
        if not backward or initial.dtype!=torch.float32 or initial.numel()!=rows*width:
            raise ValueError('NPU indexed initial values require a complete FP32 backward packet')
    try:columns=index_tile_width(rows,width,count)
    except ValueError:
        columns=min(width,MAX_TENSOR_ELEMENTS//max(rows,count))
        if columns==width:
            if initial is not None and round_half and data.dtype==torch.float16:
                raise ValueError('Large Half indexed updates with FP32 initial values require FP32 opmath')
            return _index_large(data,indices,rows,width,backward=backward,padding_idx=padding_idx,
                                round_half=round_half,initial=initial)
    if initial is not None:
        _validate(initial)
        if not backward or columns < width or initial.dtype != torch.float32 or initial.numel() != rows*width:
            raise ValueError('NPU indexed initial values require a complete FP32 backward packet')
    if columns<width:
        # Keep all indices together so duplicate-gradient accumulation order is unchanged.
        source_rows=count if backward else rows
        result_rows=rows if backward else count
        column_major=_materialize(data.view(source_rows,width).t())
        pieces=[]
        for start in range(0,width,columns):
            size=min(columns,width-start)
            block=_slice(column_major,start*source_rows,size*source_rows).view(size,source_rows).t()
            result=_index_packet(block,indices,rows,size,backward=backward,
                                 padding_idx=padding_idx,round_half=round_half)
            pieces.append(_materialize(result.view(result_rows,size).t()))
        return _materialize(_concat(pieces,(width,result_rows),data.dtype).t()).view(-1)
    size = _index_limits(rows, width, count)
    header = _concat([_scalar(v, torch.float32).view(1) for v in
                      (rows, width, count, padding_idx+1, int(round_half and data.dtype == torch.float16)+2*int(initial is not None))], (5,), torch.float32)
    words = indices.view(-1).view(torch.float32)
    values = _cast(_materialize(data), torch.float32).view(-1)
    parts = [header, words, values]
    remaining = size-5-words.numel()-values.numel()
    if remaining:
        parts.append(_broadcast(_scalar(0., torch.float32), (remaining,)))
    packet = _concat(parts, (1, size), torch.float32)
    kind = 'index-backward' if backward else 'index-gather'
    second = packet if initial is None else _concat(
        [initial.view(-1), _broadcast(_scalar(0., torch.float32), (size-initial.numel(),))], (1, size), torch.float32)
    result = _run((kind, 'f32', (1, size)), [packet, second], (1, size), torch.float32)
    if _slice(result, size-1, 1).item() != 0:
        raise IndexError('NPU index_select/embedding index is out of bounds')
    elements = rows*width if backward else count*width
    return _cast(_slice(result, 0, elements), data.dtype)


def _index_select(x, dim, indices, *, out=None):
    with _lock:
        synchronize()
        dim, rows, width, shape = _index_select_spec(x, dim, indices)
        _check_out(out, shape, x.dtype, (x, indices))
        axes = (dim,)+tuple(a for a in range(x.ndim) if a != dim)
        matrix = _materialize(x.permute(axes)).view(rows, width)
        result = _index_packet(matrix, indices, rows, width).view(tuple(shape[a] for a in axes))
        result = _materialize(result.permute(tuple(axes.index(a) for a in range(x.ndim))))
        return _copy(out, result) if out is not None else result


def _index_backward(gradient, indices, shape, dim, padding_idx=-1, *, round_half=False):
    with _lock:
        synchronize()
        _validate(gradient, dense=True)
        _validate(indices, storage=True)
        rows, width = shape[dim], prod(shape[:dim]+shape[dim+1:])
        expected = list(shape)
        expected[dim] = indices.numel()
        if tuple(gradient.shape) != tuple(expected) or indices.dtype != torch.int64 or indices.ndim != 1:
            raise ValueError('NPU index gradient has incompatible shape or indices')
        _index_storage_bound(rows,width,indices.numel())
        axes = (dim,)+tuple(a for a in range(len(shape)) if a != dim)
        matrix = _materialize(gradient.permute(axes)).view(indices.numel(), width)
        result = _index_packet(matrix, indices, rows, width, backward=True, padding_idx=padding_idx, round_half=round_half)
        result = result.view(tuple(shape[a] for a in axes))
        return _materialize(result.permute(tuple(axes.index(a) for a in range(len(shape)))))


def _indexed_rows(x, dim, index, src=None, *, out=None):
    with _lock:
        synchronize()
        _validate_float_read(x); _validate_read(index)
        if index.dtype != torch.int64:
            raise ValueError('NPU gather/scatter requires INT64 device indices')
        dim, rows, width, count = indexed_rows_geometry(tuple(x.shape), tuple(index.shape), dim)
        if src is not None:
            _validate_float_read(src)
            if src.shape != index.shape or src.dtype != x.dtype:
                raise ValueError('NPU scatter_add requires src shape equal to index and dtype equal to self')
        shape = tuple(index.shape) if src is None else tuple(x.shape)
        _check_out(out, shape, x.dtype, (x, index, src))
        axes = tuple(a for a in range(x.ndim) if a != dim)+(dim,)
        values = _materialize(x.permute(axes)).view(rows, width)
        indices = _materialize(index.permute(axes)).view(rows, count)
        source = None if src is None else _cast(_materialize(src.permute(axes)), torch.float32).view(rows, count)
        pieces = []
        for row in range(rows):
            ids = _materialize(indices.narrow(0, row, 1)).view(-1)
            data = _slice(values, row*width, width)
            if src is None:
                pieces.append(_index_packet(data, ids, width, 1))
            else:
                part = _slice(source, row*count, count)
                pieces.append(_cast(_index_packet(part, ids, width, 1, backward=True,
                                                  initial=_cast(data, torch.float32)), x.dtype))
        result = _concat(pieces, tuple(shape[a] for a in axes), x.dtype)
        result = _materialize(result.permute(tuple(axes.index(a) for a in range(x.ndim))))
        return _copy(out, result) if out is not None else result


def _gather(x, dim, index, *, sparse_grad=False, out=None):
    if sparse_grad:
        raise ValueError('NPU gather supports dense gradients only')
    return _indexed_rows(x, dim, index, out=out)


def _scatter_add(x, dim, index, src, *, out=None):
    return _indexed_rows(x, dim, index, src, out=out)


def _scatter_add_inplace(x, dim, index, src):
    with _lock:
        _validate(x, dense=True)
        return _copy(x, _scatter_add(x, dim, index, src))


def _wrap_indices(index, width):
    with _lock:
        index = _materialize(index)
        words = index.view(-1).view(torch.float32)
        pieces = []
        for start in range(0, index.numel(), 32765):
            count = min(32765, index.numel()-start); size = max(9, 6+2*count)
            header = _concat([_scalar(float(n), torch.float32).view(1) for n in (width,1,count,0,0)], (5,), torch.float32)
            packet = _concat([header, _slice(words, 2*start, 2*count),
                              _broadcast(_scalar(0., torch.float32), (size-5-2*count,))], (1,size), torch.float32)
            result = _run(('index-wrap-i64', 'f32', (1,size)), [packet,packet], (1,size), torch.float32)
            if _slice(result, size-1, 1).item() != 0:
                raise RuntimeError('NPU take_along_dim index packet failed validation')
            pieces.append(_slice(result, 0, 2*count))
        return _concat(pieces, (2*index.numel(),), torch.float32).view(torch.int64).view(index.shape)


def _take_along_dim(x, indices, dim=None, *, out=None):
    with _lock:
        with torch.no_grad():
            _validate_float_read(x); _validate_read(indices)
            if indices.dtype != torch.int64:
                raise ValueError('NPU take_along_dim requires INT64 indices')
            if dim is None:
                axis = 0; input_shape = (x.numel(),); index_shape = (indices.numel(),)
            else:
                axis = operator.index(dim)
                if not 1 <= x.ndim <= 4 or not -x.ndim <= axis < x.ndim or x.ndim != indices.ndim:
                    raise ValueError('NPU take_along_dim requires equal rank and a valid axis')
                axis %= x.ndim
                a = list(x.shape); b = list(indices.shape); a[axis] = b[axis] = 1
                common = list(torch.broadcast_shapes(a, b))
                input_shape = list(common); input_shape[axis] = x.shape[axis]
                index_shape = list(common); index_shape[axis] = indices.shape[axis]
                input_shape, index_shape = tuple(input_shape), tuple(index_shape)
            indexed_rows_geometry(input_shape, index_shape, axis)
            if prod(index_shape) > MAX_INT64_ELEMENTS:
                raise ValueError('NPU take_along_dim broadcast indices exceed INT64 storage bound')
            _check_out(out, index_shape, x.dtype, (x, indices))
            if out is not None: _inference_only((x,))
        value = x.clone(memory_format=torch.contiguous_format).view(-1) if dim is None else x.expand(input_shape)
        index = indices.reshape(-1) if dim is None else indices
        normalized = index if dim is None else _wrap_indices(index, input_shape[axis]).expand(index_shape)
        result = torch.gather(value, axis, normalized)
        return _copy(out, result) if out is not None else result


def _embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    with _lock:
        synchronize()
        _validate(weight, dense=True)
        _validate(indices, storage=True)
        if weight.ndim != 2 or indices.ndim > 3 or indices.dtype != torch.int64:
            raise ValueError('NPU embedding requires 2D FP16/FP32 weight and rank 0-3 contiguous INT64 device indices')
        if type(padding_idx) is not int or not -1 <= padding_idx < weight.shape[0] or scale_grad_by_freq or sparse:
            raise ValueError('NPU embedding supports valid padding_idx, scale_grad_by_freq=False and sparse=False')
        return _index_select(weight, 0, indices.view(-1)).view(tuple(indices.shape)+(weight.shape[1],))


def _bmm(x, y, *, out=None):
    with _lock:
        synchronize()
        batch, m, k, n = _bmm_spec(x, y)
        _check_out(out, (batch, m, n), x.dtype, (x, y))
        if not batch or not m or not n or not k:
            result = torch.empty((batch,m,n),dtype=x.dtype,device=x.device) if out is None else out
            return _fill(result,0.) if result.numel() else result
        left, right = (_cast(_materialize(v), torch.float32) for v in (x, y))
        tiles = []
        batches = max(1, MAX_ELEMENTS//(m*k*n))
        # shortcut: expanded products per batch tile; dedicated batched GEMM is future performance work.
        for start in range(0, batch, batches):
            count = min(batches, batch-start)
            a = _slice(left, start*m*k, count*m*k).view(count, m, k)
            b = _slice(right, start*k*n, count*k*n).view(count, k, n)
            if m*k*n > MAX_ELEMENTS:
                tiles.append(_matrix_fp32(a.view(m, k), b.view(k, n)).view(1, m, n))
            else:
                shape = (count, m, k, n)
                a = _broadcast(a.view(count, m, k, 1), shape)
                b = _broadcast(b.view(count, 1, k, n), shape)
                tiles.append(_reduce(_execute('mul', a, b), 2, dtype=torch.float32, kind='sum'))
        result = _cast(_concat(tiles, (batch, m, n), torch.float32), x.dtype)
        return _copy(out, result) if out is not None else result


def _baddbmm(bias,x,y,*,beta=1,alpha=1,out=None):
    with _lock:
        synchronize()
        batch,m,_,n=_bmm_spec(x,y)
        _validate_float_read(bias)
        if bias.dtype!=x.dtype or tuple(torch.broadcast_shapes(bias.shape,(batch,m,n)))!=(batch,m,n):
            raise ValueError('NPU baddbmm bias must have matching dtype and broadcast to (B,M,N)')
        if any(type(v) not in (int,float) or isfinite(v) and abs(v)>torch.finfo(torch.float32).max for v in (alpha,beta)):
            raise ValueError('NPU baddbmm coefficients must fit FP32 opmath')
        _check_out(out,(batch,m,n),x.dtype,(bias,x,y))
        a,b=(_cast(_materialize(v),torch.float32) for v in (x,y))
        result=_bmm(a,b)
        if alpha!=1:result=_binary_op(result,alpha,kind='mul')
        if beta!=0:
            value=_cast(_materialize(bias),torch.float32)
            if beta!=1:value=_binary_op(value,beta,kind='mul')
            result=_binary_op(result,value,kind='add')
        return _cast(result,x.dtype,out=out)


def _matrix_shape(x, y):
    _validate_float_read(x)
    _validate_float_read(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[0] or x.dtype != y.dtype:
        raise ValueError("NPU MatMul requires matching-dtype 2D matrices with compatible inner dimensions")
    m, k = x.shape
    n = y.shape[1]
    if m * n > _storage_limit(torch.float32):
        raise ValueError("NPU MatMul output exceeds the storage bound")
    return m, k, n


def _matrix_fp32(x, y, *, out=None):
    # ponytail: only measured 128/256 squares; expand dispatch after shape-specific benchmarks.
    if (x.shape[0], x.shape[1], y.shape[1]) in ((128, 128, 128), (256, 256, 256)):
        from torch_npu_rect_gemm import matrix_fp32
        return matrix_fp32(x, y, out=out)
    return _matrix_fp32_expanded(x, y, out=out)


def _stream_sum(values, shape, *, out=None):
    """Neumaier sum of FP32 matrix partials with two rolling values per output."""
    _check_out(out, shape, torch.float32, ())
    block = MAX_ELEMENTS//2
    counts = tuple(min(block,prod(shape)-start) for start in range(0,prod(shape),block))
    if not counts:
        raise ValueError('NPU streamed sum requires a nonempty output')
    states = None
    for value in values:
        _validate(value)
        if tuple(value.shape) != tuple(shape) or value.dtype != torch.float32:
            raise ValueError('NPU streamed sum requires same-shape FP32 partials')
        if states is None:
            zero = _scalar(0.,torch.float32)
            states = [_broadcast(zero,(2*count,)) for count in counts]
            pads = {count:_broadcast(zero,(count,)) for count in set(counts)}
        for index,count in enumerate(counts):
            packed = _concat([_slice(value,index*block,count),pads[count]],(2*count,),torch.float32)
            states[index] = _run(('stream-sum','f32',(1,2*count)),
                                 [states[index].view(1,-1),packed.view(1,-1)],(1,2*count),torch.float32).view(-1)
    if states is None:
        raise ValueError('NPU streamed sum needs at least one partial')
    pieces = [_execute('add',_slice(state,0,count),_slice(state,count,count))
              for state,count in zip(states,counts)]
    return _concat(pieces,shape,torch.float32,out)


def _matrix_fp32_expanded(x, y, *, out=None):
    m, k, n = x.shape[0], x.shape[1], y.shape[1]
    if not m or not n:
        return torch.empty((m,n), dtype=torch.float32, device=x.device) if out is None else out
    left, right = (_cast(_materialize(v), torch.float32) for v in (x, y))
    if k > MAX_ELEMENTS:
        # Both operands fit the public storage bound. Slice the transposed left
        # and row-major right so each K block uses only bounded contiguous DMA.
        width = MAX_ELEMENTS//max(m, n)
        column_major = _materialize(left.t())
        def partials():
            for start in range(0, k, width):
                count = min(width, k-start)
                a = _slice(column_major, start*m, count*m).view(count, m).t()
                b = _slice(right, start*n, count*n).view(count, n)
                yield _matrix_fp32(a, b)
        return _stream_sum(partials(), (m,n), out=out)
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
        _validate_float_read(x)
        _validate_float_read(weight)
        if not 1 <= x.ndim <= MAX_RANK or weight.ndim != 2 or x.shape[-1] != weight.shape[1] or x.dtype != weight.dtype:
            raise ValueError("NPU Linear requires rank 1-64 input and matching (out_features,in_features) weights")
        m, k, n = prod(x.shape[:-1]), x.shape[-1], weight.shape[0]
        if m * n > _storage_limit(torch.float32):
            raise ValueError("NPU Linear output exceeds the storage bound")
        if bias is not None:
            _validate_float_read(bias)
            if bias.shape != (n,) or bias.dtype != x.dtype:
                raise ValueError("NPU Linear bias must have shape (out_features,) and matching dtype")
        matrix = _materialize(x).view(m, k)
        weight = _materialize(weight)
        bias = _materialize(bias) if bias is not None else None
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
    if (transposed or type(groups) is not int or groups < 1 or any(output_padding) or
        x.ndim != 4 or weight.ndim != 4 or x.dtype != weight.dtype or
        x.shape[1] != weight.shape[1]*groups or weight.shape[0] % groups):
        raise ValueError("NPU convolution requires matching NCHW FP16/FP32, valid channel groups and non-transposed Conv2d")
    for name, values, minimum in (("stride", stride, 1), ("padding", padding, 0), ("dilation", dilation, 1)):
        if len(values) != 2 or any(type(v) is not int or v < minimum for v in values):
            raise ValueError(f"Invalid NPU convolution {name}")
    if bias is not None:
        _validate(bias)
        if bias.dtype != x.dtype or bias.shape != (weight.shape[0],):
            raise ValueError("NPU convolution bias must match output channels/dtype")
    params = tuple(weight.shape[2:]), tuple(stride), tuple(padding), tuple(dilation)
    spatial, shape = conv_geometry((x.shape[0], x.shape[1]//groups, *x.shape[2:]), *params)
    if shape[0]*weight.shape[0] > MAX_DMA_ELEMENTS:
        raise ValueError("NPU convolution output exceeds the storage bound")
    return spatial, params, shape


def _conv_groups(x, weight, bias, groups):
    """Channel groups use bounded DMA copies and the existing single-group kernel."""
    if groups == 1:
        yield x, weight, bias
        return
    inputs, outputs = x.shape[1]//groups, weight.shape[0]//groups
    for group in range(groups):
        yield (_materialize(x.narrow(1, group*inputs, inputs)),
               _materialize(weight.narrow(0, group*outputs, outputs)),
               None if bias is None else _materialize(bias.narrow(0, group*outputs, outputs)))


def _conv_streaming(original, params, shape):
    # Bound retained windows/CPU maps as well as the unfolded fold upper bound.
    return max(prod(shape),prod(original)*prod(params[0]))>_CONV_WORKSPACE_ELEMENTS


def _gather_window(x, indices, row_width):
    """Bounded byte DMA gathers; key contains immutable CPU index metadata."""
    import numpy as np
    flat = indices.reshape(-1)
    width = indices.shape[1]
    batch = max(1, _GATHER_TILE_ELEMENTS//width)
    if batch >= row_width:
        batch = batch//row_width*row_width
    block = min(_GATHER_TILE_ELEMENTS, batch*width)
    alignment=4096//x.element_size() if x.is_contiguous() and x.storage_offset()==0 else 1
    tiles, start = [], 0
    while start < flat.size:
        count = min(block, flat.size-start)
        while True:
            selected = flat[start:start+count]
            valid = selected[selected >= 0]
            base = int(valid.min()) if valid.size else 0
            base=base//alignment*alignment
            span = int(valid.max())-base+1 if valid.size else 1
            if span <= MAX_ELEMENTS:
                break
            count //= 2
        relative = np.where(selected < 0, -1, selected-base).astype(np.int64).tobytes()
        source = _slice(x, base, span)
        tiles.append(_run(('gather_tile', _dma_dtype(x.dtype), (span,), relative),
                          [source], (count,), x.dtype))
        start += count
    return _concat(tiles, tuple(indices.shape), x.dtype)


def _conv_window_tiles(x, original, params, *, output_channels=None, result_width=1):
    (_, row_width), (rows, width) = conv_geometry(original, *params)
    if output_channels is not None:
        rows, width = original[0]*original[2]*original[3], prod(params[0])*output_channels
        row_width = original[3]
    if type(result_width) is not int or result_width<1:raise ValueError('Invalid NPU convolution result width')
    # Bound both the input window and its matrix result, except a single wide row.
    batch = max(1, MAX_ELEMENTS//max(width,result_width))
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
        tiles=(_matmul(window,matrix) if bias is None else _addmm(bias,window,matrix)
               for _,_,window in _conv_window_tiles(source,tuple(x.shape),params,result_width=weight.shape[0]))
        result,columns=_stream_concat(tiles,(shape[0],weight.shape[0])),None
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
        tiles=(_matmul(window,w) for _,_,window in
               _conv_window_tiles(g,original,params,output_channels=channels,result_width=original[1]))
        nhwc=_stream_concat(tiles,(original[0],original[2],original[3],original[1]))
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


def _conv1d_arguments(x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    if weight.ndim != 3 or any(len(v) != 1 for v in (stride, padding, dilation, output_padding)):
        raise ValueError('NPU Conv1d requires rank-3 weight and one spatial parameter per argument')
    return (x.unsqueeze(2), weight.unsqueeze(2), bias, (1, *stride), (0, *padding),
            (1, *dilation), transposed, (0, *output_padding), groups)


def _convolution(x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    if x.ndim==5:
        from torch_npu_vision_extended import conv_forward
        return conv_forward(x,weight,bias,stride,padding,dilation,transposed,output_padding,groups)
    with _lock:
        synchronize()
        if x.ndim == 3:
            return _convolution(*_conv1d_arguments(x, weight, bias, stride, padding, dilation,
                                                  transposed, output_padding, groups)).squeeze(2)
        if transposed:
            from torch_npu_vision import transpose_forward
            return transpose_forward(x,weight,bias,stride,padding,dilation,output_padding,groups)
        spatial, params, shape = _conv_spec(x, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
        results = [_conv_forward(a, w, b, spatial, params, shape)[0]
                   for a, w, b in _conv_groups(x, weight, bias, groups)]
        return results[0] if groups == 1 else _cat(results, 1)


def _mse_spec(x,target,reduction):
    _validate_float_read(x);_validate_float_read(target)
    if type(reduction) is not int or reduction not in (0,1,2):
        raise ValueError('NPU MSE requires reduction none/mean/sum')
    shape=tuple(torch.broadcast_shapes(x.shape,target.shape))
    if len(shape)>MAX_RANK or prod(shape)>MAX_DMA_ELEMENTS:
        raise ValueError('NPU MSE broadcast result exceeds storage bounds')
    return shape,torch.promote_types(x.dtype,target.dtype)


def _mse_loss(x, target, reduction=1, *, out=None):
    with _lock:
        synchronize()
        shape,dtype=_mse_spec(x,target,reduction)
        _check_out(out,shape if reduction==0 else (),dtype,(x,target))
        difference = _binary_op(_cast(_materialize(x), torch.float32),
                                _cast(_materialize(target), torch.float32), kind="sub")
        result = _execute("mul", difference, difference)
        if reduction:
            result = _reduce(result, kind="mean" if reduction == 1 else "sum")
        return _cast(result,dtype,out=out)


def _mse_loss_backward(gradient,x,target,reduction,*,grad_input=None):
    with _lock:
        shape,dtype=_mse_spec(x,target,reduction);_validate_float_read(gradient)
        if tuple(gradient.shape)!=(shape if reduction==0 else ()) or gradient.dtype!=dtype:
            raise ValueError('NPU MSE backward gradient shape/dtype differs from its output')
        _check_out(grad_input,shape,x.dtype,(gradient,x,target))
        difference=_binary_op(_cast(_materialize(x),torch.float32),_cast(_materialize(target),torch.float32),kind='sub')
        # Match native ATen: functional MSE expands both inputs before this entry point.
        factor=(2./x.numel() if x.numel() else float('inf')) if reduction==1 else 2.
        result=_binary_op(_binary_op(difference,factor,kind='mul'),_cast(_materialize(gradient),torch.float32),kind='mul')
        return _cast(result,x.dtype,out=grad_input)


def _row_softmax_spec(x, dim, half_to_float=False):
    _validate_float_read(x)
    if (not 1 <= x.ndim <= MAX_RANK or type(dim) is not int or not -x.ndim <= dim < x.ndim or
            half_to_float):
        raise ValueError("NPU softmax/log_softmax requires a valid axis, rank 1-64 and half_to_float=False")
    rows, columns = x.numel()//x.shape[dim], x.shape[dim]
    softmax_row_tiles(rows,columns,max_elements=MAX_DMA_ELEMENTS)
    return rows, columns


def _softmax_rows(x, dim, shape):
    return _cast(_materialize(x.transpose(dim, -1)), torch.float32).view(shape)


def _softmax_restore(result, x, dim, dtype):
    value = _cast(result, dtype).view(x.transpose(dim, -1).shape).transpose(dim, -1)
    return _materialize(value)


def _run_rowwise(kind, values, other=None, *, causal_length=0):
    rows,columns=values.shape;pieces=[]
    for start,count in softmax_row_tiles(rows,columns,max_elements=MAX_DMA_ELEMENTS):
        shape=(count,columns);size=count*columns
        first=_slice(values,start*columns,size).view(shape)
        if kind=='sdpa-softmax':
            if causal_length and size>1:
                parts=[_scalar(causal_length,torch.float32).view(1),_scalar(start%causal_length,torch.float32).view(1)]
                if size>2:parts.append(_broadcast(_scalar(0.,torch.float32),(size-2,)))
                second=_concat(parts,shape,torch.float32)
            else:second=_broadcast(_scalar(causal_length,torch.float32),shape)
        else:second=first if other is None else _slice(other,start*columns,size).view(shape)
        pieces.append(_run((kind,'f32',shape),[first,second],shape,torch.float32))
    return _concat(pieces,(rows,columns),torch.float32)


def _log_softmax(x, dim, half_to_float=False, *, out=None):
    with _lock:
        synchronize()
        shape = _row_softmax_spec(x, dim, half_to_float)
        _check_out(out, tuple(x.shape), x.dtype, (x,))
        values = _softmax_rows(x, dim, shape)
        result = _run_rowwise('log-softmax',values)
        result = _softmax_restore(result, x, dim, x.dtype)
        return _copy(out, result) if out is not None else result


def _log_softmax_backward(gradient, output, dim, input_dtype, *, out=None):
    with _lock:
        synchronize()
        shape = _row_softmax_spec(output, dim)
        _validate_float_read(gradient)
        if gradient.shape != output.shape or gradient.dtype != output.dtype or input_dtype not in bridge._DTYPES:
            raise ValueError("NPU log_softmax backward requires matching gradient/output and FP16/FP32 input dtype")
        _check_out(out, tuple(output.shape), input_dtype, (gradient, output))
        values = _softmax_rows(output, dim, shape)
        g = _softmax_rows(gradient, dim, shape)
        result = _run_rowwise('log-softmax-backward',values,g)
        result = _softmax_restore(result, output, dim, input_dtype)
        return _copy(out, result) if out is not None else result


def _softmax(x, dim, half_to_float=False, *, out=None):
    with _lock:
        synchronize()
        shape = _row_softmax_spec(x, dim, half_to_float)
        _check_out(out, tuple(x.shape), x.dtype, (x,))
        values = _softmax_rows(x, dim, shape)
        result = _run_rowwise('softmax',values)
        result = _softmax_restore(result, x, dim, x.dtype)
        return _copy(out, result) if out is not None else result


def _softmax_backward(gradient, output, dim, input_dtype, *, out=None):
    with _lock:
        synchronize()
        shape = _row_softmax_spec(output, dim)
        _validate_float_read(gradient)
        if gradient.shape != output.shape or gradient.dtype != output.dtype or input_dtype not in bridge._DTYPES:
            raise ValueError("NPU softmax backward requires matching gradient/output and FP16/FP32 input dtype")
        _check_out(out, tuple(output.shape), input_dtype, (gradient, output))
        values = _softmax_rows(output, dim, shape)
        g = _softmax_rows(gradient, dim, shape)
        result = _run_rowwise('softmax-backward',values,g)
        result = _softmax_restore(result, output, dim, input_dtype)
        return _copy(out, result) if out is not None else result


def _sdpa_spec(query, key, value, attn_mask=None, dropout_p=0., is_causal=False, *, scale=None, enable_gqa=False, streaming=False):
    for tensor in (query, key, value):
        _validate_float_read(tensor)
    if (any(not 2 <= t.ndim <= MAX_RANK for t in (query,key,value)) or
            query.dtype != key.dtype or query.dtype != value.dtype or
            query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]):
        raise ValueError('NPU SDPA requires rank 2-64 FP16/FP32 Q/K/V with matching dtype and compatible dimensions')
    if type(dropout_p) not in (int, float) or not 0 <= dropout_p <= 1:
        raise ValueError('NPU SDPA requires dropout_p in [0,1]')
    if type(is_causal) is not bool or type(enable_gqa) is not bool:
        raise ValueError('NPU SDPA requires boolean is_causal and enable_gqa')
    shapes = [tuple(t.shape) for t in (query,key,value)]
    if enable_gqa:
        if (any(t.ndim < 3 for t in (query,key,value)) or key.shape[-3] != value.shape[-3] or
                key.shape[-3] and query.shape[-3] % key.shape[-3]):
            raise ValueError('NPU GQA requires rank >=3, equal K/V heads and Q heads divisible by K/V heads')
        shapes[1:] = [s[:-3]+(query.shape[-3],)+s[-2:] for s in shapes[1:]]
    if is_causal and attn_mask is not None:
        raise ValueError('NPU SDPA does not combine attn_mask with is_causal')
    # A zero contraction has zero scores; a finite neutral scale also keeps its gradients well-defined.
    scale = (query.shape[-1] or 1)**-.5 if scale is None else scale
    if type(scale) not in (int, float) or not isfinite(scale) or abs(scale) > torch.finfo(torch.float32).max:
        raise ValueError('NPU SDPA scale must be finite and representable in FP32')
    # The existing SHAVE causal header encodes query length up to 1,048,576.
    if is_causal and query.shape[-2] > 1048576:
        raise ValueError('NPU causal SDPA query length exceeds its kernel header bound')
    batch = tuple(torch.broadcast_shapes(*(s[:-2] for s in shapes)))
    shape = batch+(query.shape[-2], key.shape[-2])
    output = batch+(query.shape[-2], value.shape[-1])
    if prod(shape) and not streaming:
        softmax_row_tiles(prod(shape)//shape[-1],shape[-1],max_elements=MAX_DMA_ELEMENTS)
    if any(prod(batch+s[-2:]) > MAX_DMA_ELEMENTS for s in shapes) or prod(output) > MAX_DMA_ELEMENTS:
        raise ValueError('NPU SDPA broadcast operands/output exceed the storage bound')
    if attn_mask is not None:
        _validate_read(attn_mask)
        if (attn_mask.dtype not in (query.dtype, torch.float32, torch.bool) or
                tuple(torch.broadcast_shapes(attn_mask.shape, shape)) != shape):
            raise ValueError('NPU SDPA requires a broadcastable device BOOL or additive floating mask')
    return shape, scale


def _sdpa_repeat_heads(x, heads):
    if x.shape[-3] == heads:return x
    prefix,old,rows,columns = tuple(x.shape[:-3]),*x.shape[-3:]
    # Flatten the prefix so even rank-64 inputs need only a rank-3 DMA carrier.
    source = _materialize(x).view(prod(prefix)*old,1,rows*columns)
    return _broadcast(source,(prod(prefix)*old,heads//old,rows*columns)).view(prefix+(heads,rows,columns))


def _sdpa_sum_heads(gradient, heads):
    prefix,current,rows,columns = tuple(gradient.shape[:-3]),*gradient.shape[-3:]
    if current == heads:return gradient
    values = _materialize(gradient).view(prod(prefix)*heads,current//heads,rows*columns)
    return _reduce(values,1,dtype=torch.float32,kind='sum').view(prefix+(heads,rows,columns))


def _sdpa_forward(query, key, value, attn_mask=None, dropout_p=0., is_causal=False, *, scale=None, enable_gqa=False):
    with _lock:
        synchronize()
        shape, scale = _sdpa_spec(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
        if not prod(shape) or not value.shape[-1] or enable_gqa and not key.shape[-3]:
            result = torch.zeros(shape[:-1]+(value.shape[-1],),dtype=query.dtype,device=query.device)
            return result, None, None, None, None, None
        if enable_gqa:
            key,value = (_sdpa_repeat_heads(t,query.shape[-3]) for t in (key,value))
        # Scale both inputs before the dot product, as Torch math SDPA does to avoid avoidable overflow.
        root = abs(scale)**.5
        left = _binary_op(_cast(_materialize(query), torch.float32), -root if scale < 0 else root, kind='mul')
        right = _binary_op(_cast(_materialize(key), torch.float32), root, kind='mul')
        scores = _broadcast(_matmul_nd(left, right.transpose(-2, -1)), shape)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = _where(attn_mask,scores,float('-inf'))
            else:
                mask = _cast(_materialize(attn_mask), torch.float32)
                scores = _binary_op(scores, mask, kind='add')
        rows = (prod(shape)//shape[-1], shape[-1])
        probabilities = _run_rowwise('sdpa-softmax',scores.view(rows),causal_length=shape[-2] if is_causal else 0).view(shape)
        used, dropout_mask = _native_dropout(probabilities,dropout_p,True) if dropout_p else (probabilities,None)
        result = _matmul_nd(used, _cast(_materialize(value), torch.float32))
        return _cast(result, query.dtype), probabilities, left, right, used, dropout_mask


def _sdpa(query, key, value, attn_mask=None, dropout_p=0., is_causal=False, *, scale=None, enable_gqa=False):
    if not torch.backends.cuda.math_sdp_enabled():
        import torch_npu_flash_attention as flash
        if flash.selected():return flash.flash_attention(query,key,value,attn_mask,dropout_p,is_causal,scale=scale,enable_gqa=enable_gqa)
        raise ValueError('NPU SDPA requires math or the experimental FLASH_ATTENTION selection')
    return _sdpa_forward(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)[0]


def _inference_only(tensors):
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
        raise ValueError('NPU native attention/encoder entry points are inference-only; use the module training path')


def _native_mha_spec(query,key,value,embed_dim,num_head,qkv_weight,qkv_bias,proj_weight,proj_bias,mask,mask_type):
    tensors=(query,key,value,qkv_weight,qkv_bias,proj_weight,proj_bias)
    for t in tensors:_validate_float_read(t)
    if (query.ndim!=3 or type(embed_dim) is not int or type(num_head) is not int or
            embed_dim<1 or num_head<1 or embed_dim%num_head or query.shape[-1]!=embed_dim):
        raise ValueError('NPU native MHA requires batched self attention with divisible positive embedding/heads')
    if any(t.dtype!=query.dtype for t in tensors):raise ValueError('NPU native MHA requires matching floating dtypes')
    if any(t.untyped_storage()._cdata!=query.untyped_storage()._cdata or t.shape!=query.shape or
           t.stride()!=query.stride() or t.storage_offset()!=query.storage_offset() for t in (key,value)):
        raise ValueError('NPU native MHA supports self attention aliases only')
    e=embed_dim;b,l,_=query.shape
    if (qkv_weight.shape!=(3*e,e) or qkv_bias.shape!=(3*e,) or proj_weight.shape!=(e,e) or proj_bias.shape!=(e,) or
            b*num_head*l*l>MAX_DMA_ELEMENTS or b*l*3*e>MAX_DMA_ELEMENTS):
        raise ValueError('NPU native MHA projection/score/workspace bounds or parameter shapes exceeded')
    if mask is not None:
        _validate_read(mask)
        if mask.dtype not in (torch.bool,query.dtype,torch.float32) or mask_type not in (0,1,2):
            raise ValueError('NPU native MHA requires a supported device mask/type')
        expected=(l,l) if mask_type==0 else (b,l) if mask_type==1 else (b,num_head,l,l)
        if tuple(mask.shape)!=expected:raise ValueError('NPU native MHA merged mask shape differs from mask_type')


def _native_multi_head_attention(query,key,value,embed_dim,num_head,qkv_weight,qkv_bias,proj_weight,proj_bias,
                                 mask=None,need_weights=True,average_attn_weights=True,mask_type=None):
    with _lock:
        synchronize()
        _inference_only((query,key,value,qkv_weight,qkv_bias,proj_weight,proj_bias,mask))
        _native_mha_spec(query,key,value,embed_dim,num_head,qkv_weight,qkv_bias,proj_weight,proj_bias,mask,mask_type)
        if not query.numel():return _clone(query),None
        if query.dtype==torch.float16 and mask is not None:
            return _native_half_masked_attention(query,embed_dim,num_head,qkv_weight,qkv_bias,proj_weight,proj_bias,
                                                 mask,need_weights,average_attn_weights,mask_type)
        q=_materialize(query).transpose(0,1)
        padding=attention=None
        if mask is not None:
            # Native CPU fused attention interprets merged masks as exclusion bits.
            mask=_mask_cast(_materialize(mask),torch.bool) if mask.dtype!=torch.bool else _materialize(mask)
            if mask_type==1:padding=mask
            else:attention=mask if mask_type==0 else mask.reshape(-1,query.shape[1],query.shape[1])
        output,weights=torch.nn.functional.multi_head_attention_forward(q,q,q,embed_dim,num_head,
            _materialize(qkv_weight),_materialize(qkv_bias),None,None,False,0.,
            _materialize(proj_weight),_materialize(proj_bias),training=False,key_padding_mask=padding,
            need_weights=True,attn_mask=attention,average_attn_weights=average_attn_weights if need_weights else False)
        return output.transpose(0,1),weights if need_weights else None


def _native_half_masked_softmax(scores,mask):
    _validate_float_read(scores);_validate_read(mask)
    if scores.dtype!=torch.float16 or scores.ndim<1 or mask.dtype!=torch.bool or mask.shape!=scores.shape:
        raise ValueError('Native half masked-softmax requires matching bounded FP16 scores and BOOL mask')
    shape=(scores.numel()//scores.shape[-1],scores.shape[-1])
    softmax_row_tiles(*shape,max_elements=MAX_DMA_ELEMENTS)
    values=_cast(_materialize(scores),torch.float32).view(shape)
    excluded=_mask_cast(_materialize(mask),torch.float32).view(shape)
    return _cast(_run_rowwise('native-half-softmax',values,excluded),torch.float16).view(scores.shape)


def _native_half_masked_attention(query,e,heads,qkv_weight,qkv_bias,proj_weight,proj_bias,mask,need_weights,average,mask_type):
    b,l,_=query.shape;d=e//heads
    projected=_linear(query,qkv_weight,None);parts=[]
    for i in range(3):
        values=_binary_op(_cast(_materialize(projected[:,:,i*e:(i+1)*e]),torch.float32),
                          _cast(_materialize(qkv_bias[i*e:(i+1)*e]),torch.float32),kind='add')
        if i==0:values=_binary_op(values,d**-.5,kind='mul')
        parts.append(_materialize(_cast(values,torch.float16).view(b,l,heads,d).transpose(1,2)).view(b*heads,l,d))
    scores=_bmm(parts[0],parts[1].transpose(1,2)).view(b,heads,l,l)
    mask=_mask_cast(_materialize(mask),torch.bool) if mask.dtype!=torch.bool else _materialize(mask)
    if mask_type==1:mask=mask.view(b,1,1,l)
    probability=_native_half_masked_softmax(scores,_broadcast(mask,tuple(scores.shape))).view(b*heads,l,l)
    mixed=_bmm(probability,parts[2]).view(b,heads,l,d)
    output=_linear(_materialize(mixed.transpose(1,2)).view(b,l,e),proj_weight,proj_bias)
    weights=probability.view(b,heads,l,l) if need_weights else None
    if need_weights and average:weights=_binary_op(_reduce(weights,1,dtype=torch.float16,kind='sum'),heads,kind='div')
    return output,weights


def _transformer_encoder_layer_fwd(src,embed_dim,num_heads,qkv_weight,qkv_bias,proj_weight,proj_bias,
        use_gelu,norm_first,eps,norm_weight_1,norm_bias_1,norm_weight_2,norm_bias_2,
        ffn_weight_1,ffn_bias_1,ffn_weight_2,ffn_bias_2,mask=None,mask_type=None):
    with _lock:
        synchronize()
        tensors=(src,qkv_weight,qkv_bias,proj_weight,proj_bias,norm_weight_1,norm_bias_1,
                 norm_weight_2,norm_bias_2,ffn_weight_1,ffn_bias_1,ffn_weight_2,ffn_bias_2)
        _inference_only((*tensors,mask))
        _native_mha_spec(src,src,src,embed_dim,num_heads,qkv_weight,qkv_bias,proj_weight,proj_bias,mask,mask_type)
        for t in tensors:_validate_float_read(t)
        if any(t.dtype!=src.dtype for t in tensors) or not isfinite(eps) or eps<=0:
            raise ValueError('NPU native encoder requires matching dtypes and finite positive eps')
        e=embed_dim
        if (any(t.shape!=(e,) for t in (norm_weight_1,norm_bias_1,norm_weight_2,norm_bias_2)) or
            ffn_weight_1.ndim!=2 or ffn_weight_1.shape[1]!=e or ffn_weight_2.shape!=(e,ffn_weight_1.shape[0]) or
            ffn_bias_1.shape!=(ffn_weight_1.shape[0],) or ffn_bias_2.shape!=(e,) or
            src.shape[0]*src.shape[1]*ffn_weight_1.shape[0]>MAX_DMA_ELEMENTS):
            raise ValueError('NPU native encoder FFN/norm parameter shapes or workspace bound exceeded')
        if not src.numel():return _clone(src)
        def norm(x,w,b):return _layer_norm(x,(e,),w,b,eps)[0]
        x=_materialize(src)
        h=norm(x,norm_weight_1,norm_bias_1) if norm_first else x
        attention=_native_multi_head_attention(h,h,h,e,num_heads,qkv_weight,qkv_bias,proj_weight,proj_bias,
                                              mask,False,False,mask_type)[0]
        x=_binary_op(x,attention,kind='add')
        if not norm_first:x=norm(x,norm_weight_1,norm_bias_1)
        h=norm(x,norm_weight_2,norm_bias_2) if norm_first else x
        h=_linear(h,ffn_weight_1,ffn_bias_1)
        h=_gelu(h) if use_gelu else _execute('relu',h)
        x=_binary_op(x,_linear(h,ffn_weight_2,ffn_bias_2),kind='add')
        return x if norm_first else norm(x,norm_weight_2,norm_bias_2)


def _nll_spec(x, target, weight, reduction, ignore_index):
    _validate_float_read(x)
    _validate_read(target)
    if x.ndim not in (1, 2) or x.shape[-1]<1 or target.dtype != torch.int64 or target.shape != x.shape[:-1]:
        raise ValueError("NPU NLL requires (C) or (N,C) and matching INT64 target on intel_npu")
    _classification_weight(x,weight,x.shape[-1])
    if type(reduction) is not int or reduction not in (0, 1, 2):
        raise ValueError("NPU NLL requires reduction none/mean/sum")
    if type(ignore_index) is not int or not -(1 << 63) <= ignore_index < 1 << 63:
        raise ValueError("NPU ignore_index must be a signed 64-bit Python integer")
    rows, columns = x.numel()//x.shape[-1], x.shape[-1]
    nll_row_tiles(rows,columns,max_elements=MAX_DMA_ELEMENTS)
    return rows, columns


def _class_columns(x, start, count):
    shape = tuple(x.shape)
    return _run(("class-columns", "f32", shape, start, count), [x], (shape[0], count), torch.float32)


def _classification_weight(x,weight,classes,*,soft=False):
    if weight is None:return
    _validate_float_read(weight)
    if weight.shape!=(classes,) or not soft and weight.dtype!=x.dtype:
        raise ValueError('NPU class weight requires shape (C,) and matching hard-label input dtype')
    if not soft and weight.requires_grad:
        raise RuntimeError('NLL/hard-label class weight is not differentiable')


def _nll_row_weights(packet,columns,weight):
    selected=_compare(_class_columns(packet,0,columns),1.,kind='eq')
    values=_cast(_materialize(weight),torch.float32).view(1,columns)
    return _reduce(_where(selected,values,0.),1,kind='sum')


def _nll_packet(values, target, ignore_index):
    rows,columns=values.shape;shape=(rows,columns+4)
    labels=_materialize(target).view(-1).view(torch.float32).view(rows,2)
    ignored=_broadcast(_scalar(ignore_index,torch.int64).view(1).view(torch.float32).view(1,2),(rows,2))
    packed=_run(('class-pack','f32',(rows,columns)),[values,labels,ignored],shape,torch.float32)
    return _run(('nll-packet','f32',shape),[packed,packed],shape,torch.float32)


def _nll_forward(x, target, weight=None, reduction=1, ignore_index=-100):
    with _lock:
        synchronize()
        rows, columns = _nll_spec(x, target, weight, reduction, ignore_index)
        shape = rows, columns+4
        if not rows:
            result=(torch.empty(tuple(target.shape),dtype=x.dtype,device=x.device) if reduction==0 else
                    _scalar(float('nan') if reduction==1 else 0.,x.dtype).clone())
            return result,_scalar(0.,x.dtype).clone(),torch.empty(shape,dtype=torch.float32,device=x.device)
        values=_cast(_materialize(x),torch.float32).view(-1);labels=_materialize(target).view(-1)
        tiles=nll_row_tiles(rows,columns,max_elements=MAX_DMA_ELEMENTS)
        loss_parts=[];weight_parts=[];flags=[]
        for start,count in tiles:
            packet=_nll_packet(_slice(values,start*columns,count*columns).view(count,columns),
                               _slice(labels,start,count),ignore_index)
            flags.append(_reduce(_class_columns(packet,columns+1,1),kind='sum').view(1))
            losses=_class_columns(packet,columns,1).view(-1)
            row_weights=(_class_columns(packet,columns+2,1).view(-1) if weight is None else
                         _nll_row_weights(packet,columns,weight).view(-1))
            if weight is not None:
                losses=_cast(_cast(_binary_op(losses,row_weights,kind='mul'),x.dtype),torch.float32)
            loss_parts.append(losses);weight_parts.append(row_weights)
        # Only the device validation flag crosses to the host; labels stay resident.
        invalid=_reduce(_concat(flags,(len(flags),),torch.float32),kind='sum')
        if invalid.cpu().item() != 0:
            raise IndexError("NPU NLL target is out of bounds")
        losses=_concat(loss_parts,(rows,),torch.float32).view(target.shape)
        row_weights=_concat(weight_parts,(rows,),torch.float32).view(target.shape)
        total_weight = (_cast(_reduce(row_weights, kind="sum"), x.dtype)
                        if reduction or x.ndim == 1 else _scalar(0., x.dtype).clone())
        result = losses
        if reduction:
            # Native NLL rounds the reduced numerator before its mean division.
            result = _cast(_cast(_reduce(losses, kind="sum"),x.dtype),torch.float32)
            if reduction == 1:
                result = _binary_op(result, _cast(total_weight, torch.float32), kind="div")
        # Large batches recompute bounded masks from saved labels in backward.
        saved=packet if len(tiles)==1 else torch.empty((0,),dtype=torch.float32,device=x.device)
        return _cast(result, x.dtype), total_weight, saved


def _nll_backward(gradient, x, packet, total_weight, reduction, weight=None, *, target=None, ignore_index=-100, validate_target=False):
    with _lock:
        synchronize()
        _validate_float_read(gradient)
        expected_shape = tuple(x.shape[:-1]) if reduction == 0 else ()
        if tuple(gradient.shape) != expected_shape or gradient.dtype != x.dtype:
            raise ValueError("NPU NLL gradient has the wrong shape/dtype")
        if not x.numel():return torch.empty(tuple(x.shape),dtype=x.dtype,device=x.device)
        if not packet.numel():
            if target is None:raise ValueError('Tiled NLL backward requires saved device labels')
            rows,columns=_nll_spec(x,target,weight,reduction,ignore_index);pieces=[];flags=[]
            labels=_materialize(target).view(-1);tile_reduction=reduction
            if reduction==1:
                gradient=_cast(_binary_op(_cast(_materialize(gradient),torch.float32),
                                         _cast(total_weight,torch.float32),kind='div'),x.dtype)
                tile_reduction=2
            flat=_materialize(gradient).view(-1)
            for start,count in nll_row_tiles(rows,columns,max_elements=MAX_DMA_ELEMENTS):
                dummy=_broadcast(_scalar(0.,x.dtype),(count,columns))
                selected=_nll_packet(_cast(dummy,torch.float32),_slice(labels,start,count),ignore_index)
                if validate_target:flags.append(_reduce(_class_columns(selected,columns+1,1),kind='sum').view(1))
                g=_slice(flat,start,count) if reduction==0 else gradient
                pieces.append(_nll_backward(g,dummy,selected,total_weight,tile_reduction,weight))
            if validate_target and _reduce(_concat(flags,(len(flags),),torch.float32),kind='sum').cpu().item()!=0:
                raise IndexError('NPU NLL target is out of bounds')
            return _concat(pieces,tuple(x.shape),x.dtype)
        g = _cast(_materialize(gradient), torch.float32)
        if reduction == 1:
            g = _binary_op(g, _cast(total_weight, torch.float32), kind="div")
            g = _cast(_cast(g,x.dtype),torch.float32)
        if weight is not None:
            g=_binary_op(g,_nll_row_weights(packet,x.shape[-1],weight).view(expected_shape if reduction==0 else (-1,)),kind='mul')
        g = _broadcast(g.view(-1, 1), tuple(packet.shape))
        result = _run(("nll-backward", "f32", tuple(packet.shape)), [packet, g], tuple(packet.shape), torch.float32)
        return _cast(_class_columns(result, 0, x.shape[-1]), x.dtype).view(x.shape)


def _classification_input(x, target, weight, reduction, ignore_index, label_smoothing=0.0):
    # Public class-axis losses flatten only metadata/layout; labels stay on NPU.
    with torch.no_grad():
        _validate_float_read(x)
        _validate_read(target)
        classes = x.shape[0] if x.ndim == 1 else x.shape[1] if 2 <= x.ndim <= 5 else 0
        expected = () if x.ndim == 1 else (x.shape[0], *x.shape[2:])
        if not classes or target.dtype != torch.int64 or tuple(target.shape) != expected:
            raise ValueError('NPU class-axis loss requires rank 1-5 and matching INT64 hard labels')
        _classification_weight(x,weight,classes)
        if type(reduction) is not int or reduction not in (0, 1, 2):
            raise ValueError('NPU class-axis loss requires reduction none/mean/sum')
        if type(ignore_index) is not int or not -(1 << 63) <= ignore_index < 1 << 63:
            raise ValueError('NPU ignore_index must be a signed 64-bit Python integer')
        nll_row_tiles(target.numel(),classes,max_elements=MAX_DMA_ELEMENTS)
        if type(label_smoothing) not in (int, float) or not 0 <= label_smoothing <= 1:
            raise ValueError('NPU label_smoothing must be in [0,1]')
    if x.ndim > 2:
        x = x.movedim(1, -1).clone(memory_format=torch.contiguous_format).view(-1, classes)
        # shortcut: copied strided spatial labels still need source-version tracking.
        target = target.contiguous().view(-1)
    return x, target


def _smooth_classification_loss(log_probs, target, result, total, reduction, smoothing, ignore_index, weight=None):
    if smoothing == 0:
        return result
    values=log_probs if weight is None else log_probs*weight
    smooth = torch.where(target != ignore_index, -values.sum(-1), 0.)
    if reduction:
        smooth = smooth.sum()
        if reduction == 1:
            smooth = smooth / total
    return result*(1.-smoothing) + smooth*(smoothing/log_probs.shape[-1])


def _nll_nd(x, target, weight=None, reduction=1, ignore_index=-100):
    shape = target.shape
    x, target = _classification_input(x, target, weight, reduction, ignore_index)
    result = _nll_forward(x, target, weight, reduction, ignore_index)[0]
    return result.view(shape) if reduction == 0 else result


def _cross_entropy(x, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0):
    if target.shape==x.shape:
        with torch.no_grad():
            _validate_float_read(x);_validate_float_read(target)
            classes=x.shape[0] if x.ndim==1 else x.shape[1] if 2<=x.ndim<=5 else 0
            if not classes or type(reduction) is not int or reduction not in (0,1,2):
                raise ValueError('NPU soft-label CE requires rank 1-5, nonempty classes and reduction none/mean/sum')
            _classification_weight(x,weight,classes,soft=True)
            if type(ignore_index) is not int or ignore_index>=0:
                raise ValueError('ignore_index is not supported for floating point target')
            if type(label_smoothing) not in (int,float) or not 0<=label_smoothing<=1:
                raise ValueError('NPU label_smoothing must be in [0,1]')
        axis=0 if x.ndim==1 else 1
        output=torch.log_softmax(x,axis) if x.numel() else x
        labels=target*(1.-label_smoothing)+label_smoothing/classes if label_smoothing else target
        terms=output*labels
        if weight is not None:
            shape=[1]*x.ndim;shape[axis]=classes
            terms=terms*weight.view(shape)
        if reduction==0:return -terms.sum(axis)
        result=-terms.sum()
        return result/(x.numel()//classes) if reduction==1 else result
    shape = target.shape
    x, target = _classification_input(x, target, weight, reduction, ignore_index, label_smoothing)
    output = torch.log_softmax(x,-1) if x.numel() else x
    result,total=torch.ops.aten.nll_loss_forward(output,target,weight,reduction,ignore_index)
    result = _smooth_classification_loss(output, target, result, total, reduction, label_smoothing, ignore_index,weight)
    return result.view(shape) if reduction == 0 else result


def _nll_native_forward(x,target,weight,reduction,ignore_index,*,spatial=False,output=None,total_weight=None):
    with _lock:
        _validate_float_read(x);_validate_read(target)
        if spatial and x.ndim!=4:raise ValueError('nll_loss2d requires rank-4 NCHW input')
        if not spatial:_nll_spec(x,target,weight,reduction,ignore_index)
        if (output is None)!=(total_weight is None):raise ValueError('NLL out requires both output and total_weight')
        inputs=(x,target)+(() if weight is None else (weight,))
        _check_out(output,tuple(target.shape) if reduction==0 else (),x.dtype,inputs+(() if total_weight is None else (total_weight,)))
        _check_out(total_weight,(),x.dtype,inputs+(() if output is None else (output,)))
        values,labels=_classification_input(x,target,weight,reduction,ignore_index) if spatial else (x,target)
        result,total,_=_nll_forward(values,labels,weight,reduction,ignore_index)
        if reduction==0:result=result.view(target.shape)
        return (_copy(output,result),_copy(total_weight,total)) if output is not None else (result,total)


def _nll_native_backward(gradient,x,target,weight,reduction,ignore_index,total_weight,*,spatial=False,grad_input=None):
    with _lock:
        _validate_float_read(x);_validate_read(target);_validate_float_read(gradient);_validate_float_read(total_weight)
        if spatial and x.ndim!=4:raise ValueError('nll_loss2d backward requires rank-4 NCHW input')
        if not spatial:_nll_spec(x,target,weight,reduction,ignore_index)
        if total_weight.shape!=() or total_weight.dtype!=x.dtype:
            raise ValueError('NLL total_weight requires a matching-dtype device scalar')
        if tuple(gradient.shape)!=(tuple(target.shape) if reduction==0 else ()) or gradient.dtype!=x.dtype:
            raise ValueError('NLL native gradient shape/dtype differs from its output')
        _check_out(grad_input,tuple(x.shape),x.dtype,(gradient,x,target,total_weight)+(() if weight is None else (weight,)))
        values,labels=_classification_input(x,target,weight,reduction,ignore_index) if spatial else (x,target)
        tiled=values.numel()//values.shape[-1]*(values.shape[-1]+4)>MAX_ELEMENTS
        packet=(torch.empty((0,),dtype=torch.float32,device=x.device) if tiled else
                _nll_forward(values,labels,None,0,ignore_index)[2])
        g=_materialize(gradient).view(labels.shape if reduction==0 else ())
        result=_nll_backward(g,values,packet,total_weight,reduction,weight,target=labels,
                             ignore_index=ignore_index,validate_target=tiled)
        if spatial:result=_materialize(result.view(x.movedim(1,-1).shape).movedim(-1,1))
        return _copy(grad_input,result) if grad_input is not None else result


def _batch_norm(x, weight, bias, running_mean, running_var, training, momentum, eps, *, save_centered=False):
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
        # CPU half BatchNorm saves rounded mean/invstd; use the same boundary
        # before normalization/backward, while running updates retain FP32 stats.
        if training and x.dtype == torch.float16 and not any(
                v is not None and v.dtype == torch.float32 for v in (weight, bias, *buffers)):
            centered = _binary_op(values, _cast(_cast(mean, x.dtype), torch.float32).view(shape), kind="sub")
            inverse = _cast(_cast(inverse, x.dtype), torch.float32)
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
        saved=(_cast(result, x.dtype),normalized,inverse)
        return (*saved,centered) if save_centered else saved


def _layer_norm_spec(x, normalized_shape, weight, bias, eps):
    _validate(x, dense=True)
    shape = tuple(operator.index(v) for v in normalized_shape)
    if not shape or any(v <= 0 for v in shape) or len(shape) > x.ndim or tuple(x.shape[-len(shape):]) != shape:
        raise ValueError("NPU LayerNorm normalized_shape must match nonempty trailing dimensions")
    if not isfinite(eps) or eps <= 0 or eps > 3.4028234663852886e38 or eps < 2**-149:
        raise ValueError("NPU LayerNorm requires positive finite FP32 eps")
    affine = [v for v in (weight, bias) if v is not None]
    for value in affine:
        _validate(value, dense=True)
        if tuple(value.shape) != shape or value.dtype not in (x.dtype, torch.float32):
            raise ValueError("NPU LayerNorm affine tensors require normalized_shape and input/FP32 dtype")
    if affine and any(v.dtype != affine[0].dtype for v in affine):
        raise ValueError("NPU LayerNorm affine tensors must share a dtype")
    if x.dtype == torch.float16 and bias is not None and bias.dtype == torch.float32 and weight is None:
        raise ValueError("NPU mixed LayerNorm requires FP32 weight (CPU bias-only mixed backward is unsupported)")
    if x.dtype == torch.float16 and affine and affine[0].dtype == torch.float32 and prod(shape) > MAX_ELEMENTS:
        raise ValueError("NPU mixed LayerNorm requires normalized width <=65536")
    return tuple(range(x.ndim-len(shape), x.ndim)), (affine[0].dtype if affine else x.dtype)


def _layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    with _lock:
        axes, stats_dtype = _layer_norm_spec(x, normalized_shape, weight, bias, eps)
        values = _cast(_materialize(x), torch.float32)
        if x.dtype == torch.float16 and stats_dtype == torch.float32 and prod(normalized_shape) >= 2:
            mean, inverse = _layer_norm_ordered_stats(values, normalized_shape, eps)
            centered = _binary_op(values, mean, kind="sub")
        else:
            mean = _reduce(values, axes, True, kind="mean")
            centered = _binary_op(values, mean, kind="sub")
            variance = _reduce(_execute("mul", centered, centered), axes, True, kind="mean")
            inverse = _execute("rsqrt", _binary_op(variance, eps, kind="add"))
        result = _binary_op(centered, inverse, kind="mul")
        if weight is not None:
            result = _binary_op(result, _cast(_materialize(weight), torch.float32), kind="mul")
        if bias is not None:
            result = _binary_op(result, _cast(_materialize(bias), torch.float32), kind="add")
        # CPU forward uses full statistics, then saves them in the affine/input dtype.
        return _cast(result, x.dtype), _cast(mean, stats_dtype), _cast(inverse, stats_dtype)


def _layer_norm_ordered_stats(values, normalized_shape, eps):
    """Keep the CPU half-input Welford order before FP16 affine-gradient rounding."""
    width = prod(normalized_shape)
    rows, batch = values.numel()//width, MAX_ELEMENTS//width
    means, inverses = [], []
    for start in range(0, rows, batch):
        count = min(batch, rows-start)
        tile = _slice(values, start*width, count*width).view(count, width)
        epsilon = _broadcast(_scalar(eps, torch.float32), tuple(tile.shape))
        packet = _run(("layernorm-stats", "f32", tuple(tile.shape)), [tile, epsilon], tile.shape, torch.float32)
        means.append(_class_columns(packet, 0, 1))
        inverses.append(_class_columns(packet, 1, 1))
    shape = tuple(values.shape[:-len(normalized_shape)])+(1,)*len(normalized_shape)
    return _concat(means, shape, torch.float32), _concat(inverses, shape, torch.float32)


def _layer_norm_affine_gradient(value, x, normalized_shape, dtype):
    leading = tuple(range(x.ndim-len(normalized_shape)))
    if x.dtype == torch.float16:
        # CPU LayerNorm uses input-dtype per-thread buffers even for FP32 affine tensors.
        # Match single-thread row accumulation; parallel CPU reassociation can differ.
        width = prod(normalized_shape)
        result = _cast(_slice(value, 0, width), x.dtype)
        for start in range(width, value.numel(), width):
            part = _slice(value, start, width)
            result = _cast(_execute("add", _cast(result, torch.float32), part), x.dtype)
        return _cast(result.view(normalized_shape), dtype)
    result = _reduce(value, leading, kind="sum") if leading else value
    return _cast(result.view(normalized_shape), dtype)


def _layer_norm_backward(gradient, x, normalized_shape, mean, inverse, weight, bias, output_mask):
    with _lock:
        axes, stats_dtype = _layer_norm_spec(x, normalized_shape, weight, bias, 1e-5)
        _validate(gradient, dense=True)
        stats_shape = tuple(1 if a in axes else d for a, d in enumerate(x.shape))
        if gradient.shape != x.shape or gradient.dtype != x.dtype or len(output_mask) != 3:
            raise ValueError("NPU LayerNorm backward requires matching gradient and three output flags")
        for value in (mean, inverse):
            _validate(value)
            if tuple(value.shape) != stats_shape or value.dtype != stats_dtype:
                raise ValueError("NPU LayerNorm backward statistics have incorrect shape/dtype")
        g, values, m, a = (_cast(_materialize(v), torch.float32) for v in (gradient, x, mean, inverse))
        gx = gw = gb = None
        if output_mask[1] and weight is not None:
            normalized = _binary_op(_binary_op(values, a, kind="mul"), _execute("mul", m, a), kind="sub")
            gw = _execute("mul", g, normalized)
            gw = _layer_norm_affine_gradient(gw, x, normalized_shape, weight.dtype)
        if output_mask[2] and bias is not None:
            gb = _layer_norm_affine_gradient(g, x, normalized_shape, bias.dtype)
        if output_mask[0]:
            scaled = g if weight is None else _binary_op(g, _cast(_materialize(weight), torch.float32), kind="mul")
            db = _reduce(scaled, axes, True, kind="mean")
            ds = _reduce(_execute("mul", scaled, values), axes, True, kind="mean")
            b = _binary_op(_execute("mul", db, m), ds, kind="sub")
            for _ in range(3):
                b = _execute("mul", b, a)
            c = _binary_op(_execute("neg", _execute("mul", b, m)), _execute("mul", db, a), kind="sub")
            gx = _binary_op(_binary_op(scaled, a, kind="mul"), _binary_op(values, b, kind="mul"), kind="add")
            gx = _cast(_binary_op(gx, c, kind="add"), x.dtype)
        return gx, gw, gb


def _gelu(x, *, approximate="none", out=None):
    return _gelu_compute(x, approximate, out=out)


def _gelu_backward(gradient, x, *, approximate="none", grad_input=None):
    return _gelu_compute(x, approximate, gradient=gradient, out=grad_input)


def _gelu_compute(x, approximate, *, gradient=None, out=None):
    with _lock:
        _validate(x, dense=True)
        if approximate not in ("none", "tanh"):
            raise ValueError("NPU GELU approximate must be 'none' or 'tanh'")
        if gradient is not None:
            _validate(gradient, dense=True)
            if gradient.shape != x.shape or gradient.dtype != x.dtype:
                raise ValueError("NPU GELU backward gradient must match input shape/dtype")
        if out is not None:
            _check_out(out, x.shape, x.dtype, [x] + ([] if gradient is None else [gradient]))
        value = _cast(_materialize(x), torch.float32)
        # The verified Power carrier has room for GELU; its second read-only binding
        # is unused in forward. Keep its original code extent and scheduling intact.
        g = value if gradient is None else _cast(_materialize(gradient), torch.float32)
        kind = ("gelu-" if gradient is None else "gelu-backward-") + approximate
        return _cast(_execute(kind, value, g), x.dtype, out=out)


def _adaptive_avg_pool2d(x, output_size):
    if prod(output_size)>64:
        from torch_npu_vision_extended import pooling
        return pooling(x,dims=2,adaptive=output_size)
    with _lock:
        with torch.no_grad():
            _validate_float_read(x)
            size = tuple(output_size)
            if len(size) != 2 or x.ndim not in (3, 4) or any(n < 1 for n in size) or prod(size) > 64:
                raise ValueError('NPU adaptive average pooling requires CHW/NCHW and 1-64 positive output bins')
            if prod(x.shape[:-2])*prod(size) > MAX_TENSOR_ELEMENTS:
                raise ValueError('NPU adaptive average pooling output exceeds storage bound')
        if x.ndim == 4 and size == (1, 1):
            return _reduce(_materialize(x), (2, 3), True, kind='mean')
        with torch.autocast('intel_npu', enabled=False):
            value = x.clone(memory_format=torch.contiguous_format).float()
            height, width = x.shape[-2:]
            rows = []
            for y in range(size[0]):
                start_y, end_y = y*height//size[0], ((y+1)*height+size[0]-1)//size[0]
                pieces = []
                for z in range(size[1]):
                    start_x, end_x = z*width//size[1], ((z+1)*width+size[1]-1)//size[1]
                    window = value[..., start_y:end_y, start_x:end_x].clone(memory_format=torch.contiguous_format)
                    pieces.append(window.mean((-2, -1), keepdim=True))
                rows.append(torch.cat(pieces, -1))
            return torch.cat(rows, -2).to(x.dtype)


def _avg_pool2d(x, kernel_size, stride=(), padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, *, out=None):
    import numpy as np
    with _lock:
        with torch.no_grad():
            _validate_float_read(x)
            if x.ndim not in (3, 4):
                raise ValueError('NPU average pooling requires CHW/NCHW')
            kernel, step, pad, size, extra, divisors = avg_pool2d_geometry(
                tuple(x.shape[-2:]), kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
            shape = (*x.shape[:-2], *size)
            if max(prod(shape), prod(x.shape[:-2])*(x.shape[-2]+extra[0])*(x.shape[-1]+extra[1]), x.shape[-3]*prod(kernel)) > MAX_TENSOR_ELEMENTS:
                raise ValueError('NPU average pooling output, padded input or kernel exceeds storage bound')
            _check_out(out, shape, x.dtype, (x,))
            if out is not None:
                _inference_only((x,))
        with torch.autocast('intel_npu', enabled=False):
            value = x.clone(memory_format=torch.contiguous_format).float()
            if x.ndim == 3:
                value = value.unsqueeze(0)
            if extra[1]:
                value = torch.cat((value, torch.zeros((*value.shape[:-1], extra[1]), dtype=torch.float32, device='intel_npu')), -1)
            if extra[0]:
                value = torch.cat((value, torch.zeros((*value.shape[:-2], extra[0], value.shape[-1]), dtype=torch.float32, device='intel_npu')), -2)
            weight = torch.ones((x.shape[-3], 1, *kernel), dtype=torch.float32, device='intel_npu')
            sums = torch.conv2d(value, weight, stride=step, padding=pad, groups=x.shape[-3])
            # Divisors depend only on public geometry, never on input payloads.
            denominator = (_scalar(float(divisors.flat[0]), torch.float32) if np.all(divisors == divisors.flat[0]) else
                           torch.from_numpy(divisors).to('intel_npu'))
            result = (sums/denominator).to(x.dtype).view(shape)
            return _copy(out, result) if out is not None else result


def _avg_pool1d(x, kernel_size, stride=(), padding=(0,), ceil_mode=False, count_include_pad=True, *, out=None):
    if x.ndim not in (2, 3) or len(kernel_size) != 1 or any(len(v) not in (0, 1) for v in (stride, padding)):
        raise ValueError('NPU AvgPool1d requires CL/NCL and one spatial parameter')
    kernel = (1, *kernel_size)
    step = (1, stride[0] if stride else kernel_size[0])
    if out is not None:
        with torch.no_grad():
            _validate_float_read(x)
            size = avg_pool2d_geometry((1, x.shape[-1]), kernel, step, (0, padding[0] if padding else 0),
                                       ceil_mode, count_include_pad, None)[3]
            _check_out(out, (*x.shape[:-1], size[1]), x.dtype, (x,))
        _inference_only((x,))
    result = _avg_pool2d(x.unsqueeze(-2), kernel, step, (0, padding[0] if padding else 0), ceil_mode, count_include_pad)
    result = result.squeeze(-2)
    if out is not None:
        with torch.no_grad():
            _check_out(out, tuple(result.shape), result.dtype, (x,))
        _inference_only((x,))
        return _copy(out, result)
    return result


def _adaptive_avg_pool1d(x, output_size, *, out=None):
    if x.ndim not in (2, 3) or len(output_size) != 1:
        raise ValueError('NPU AdaptiveAvgPool1d requires CL/NCL and one output size')
    if out is not None:
        with torch.no_grad():
            _validate_float_read(x)
            if not 1 <= output_size[0] <= 64:
                raise ValueError('NPU adaptive average pooling requires 1-64 output bins')
            _check_out(out, (*x.shape[:-1], output_size[0]), x.dtype, (x,))
        _inference_only((x,))
    result = torch.nn.functional.adaptive_avg_pool2d(x.unsqueeze(-2), (1, output_size[0])).squeeze(-2)
    if out is not None:
        with torch.no_grad():
            _check_out(out, tuple(result.shape), result.dtype, (x,))
        _inference_only((x,))
        return _copy(out, result)
    return result


def _fill(x, value):
    with _lock:
        synchronize()
        if not x.numel():
            _validate(x,dense=True,storage=True)
            return x
        if x.ndim>4:
            _fill(x.view(-1),value)
            return x
        if x.dtype == torch.bool:
            _validate(x, dense=True, storage=True)
            if type(value) not in (int, float, bool):
                raise ValueError('BOOL fill requires a real scalar')
            target = _physical_view(x)[0] if not x.is_contiguous() else x
            _run(('broadcast','u8',(1,),tuple(target.shape) or (1,)),
                 [_scalar(bool(value),torch.bool)],tuple(target.shape),torch.bool,out=target)
            return x
        if x.dtype == torch.int64:
            _validate(x, storage=True)
            if type(value) is float and isfinite(value) and value.is_integer():
                value = int(value)
            if type(value) is not int or not -(1 << 63) <= value < 1 << 63:
                raise ValueError("NPU INT64 fill requires a signed 64-bit Python integer")
            raw = x.view(-1).view(torch.float32).view(x.numel(), 2)
            constant = _scalar(value, torch.int64).view(1).view(torch.float32).view(1, 2)
            _run(("broadcast", "f32", (1, 2), tuple(raw.shape)), [constant], tuple(raw.shape), torch.float32, out=raw)
            return x
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
        if x.dtype == torch.int64:
            if isinstance(y, torch.Tensor) and y.device.type == 'cpu' and y.dtype == torch.int64 and y.ndim == 0:
                y = y.item()
            if type(y) is not int:
                raise ValueError("NPU INT64 add_ requires a Python integer counter increment")
            return _add_scalar_inplace(x, y, alpha=alpha)
        if type(y) in (int, float):
            return _add_scalar_inplace(x, y, alpha=alpha)
        if (isinstance(y, torch.Tensor) and y.device.type == 'cpu' and y.ndim == 0 and
                y.dtype != torch.bool and not y.is_complex() and not y.requires_grad):
            return _add_scalar_inplace(x, y.item(), alpha=alpha)
        if torch.is_grad_enabled():
            raise ValueError("NPU in-place updates require no_grad; differentiable optimizers are unsupported")
        _validate(x, dense=True)
        if not isinstance(y, torch.Tensor) or y.dtype != x.dtype or y.device != x.device or y.shape != x.shape:
            raise ValueError("NPU in-place add requires matching shape/dtype/device")
        _validate(y, dense=True)
        _update_coefficient(alpha)
        if x.dtype == torch.float16 and isfinite(alpha) and abs(alpha) > 65504:
            raise ValueError("NPU in-place alpha must be representable in FP16")
        left, right = _cast(_materialize(x), torch.float32), _cast(_materialize(y), torch.float32)
        if alpha != 1:
            # Tensor add_ converts alpha to the input dtype before FP32 opmath.
            scale = _cast(_scalar(alpha, x.dtype), torch.float32)
            right = _binary_op(right, scale, kind="mul")
        result = _cast(_execute("add", left, right), x.dtype)
        return _copy(x, result)


def _add_scalar_inplace(x, value, *, alpha=1):
    with _lock:
        synchronize()
        if x.dtype in bridge._DTYPES:
            if torch.is_grad_enabled():
                raise ValueError("NPU in-place updates require no_grad; differentiable optimizers are unsupported")
            if type(value) not in (int, float):
                raise ValueError("NPU floating add_ requires a real Python scalar")
            return _copy(x, _binary_op(x, value, kind="add", alpha=alpha))
        _validate(x, storage=True)
        if x.dtype != torch.int64 or x.ndim != 0 or type(value) is not int or type(alpha) is not int or alpha != 1:
            raise ValueError("NPU integer add_ supports an INT64 rank-0 counter, Python integer and alpha=1")
        if not -(1 << 63) <= value < 1 << 63:
            raise ValueError("NPU INT64 increment must fit signed 64-bit")
        # Raw-word carry arithmetic runs on SHAVE; neither word is converted to float.
        left = x.view(1).view(torch.float32)
        right = _scalar(value, torch.int64).view(1).view(torch.float32)
        result = _run(("int64_add", "f32", (2,)), [left, right], (2,), torch.float32)
        return _copy(x, result.view(torch.int64).view(()))


def _local_scalar_dense(x):
    with _lock, torch.no_grad():
        synchronize()
        _validate(x, storage=True)
        if x.numel() != 1:
            raise ValueError("NPU scalar extraction requires exactly one element")
        host = torch.empty((), dtype=x.dtype)
        _copy(host, x.view(()))
        return host.item()


def _mul_inplace(x, y):
    with _lock:
        synchronize()
        if torch.is_grad_enabled():
            raise ValueError("NPU in-place updates require no_grad; differentiable optimizers are unsupported")
        _validate(x, dense=True)
        if isinstance(y, torch.Tensor):
            if y.dtype != x.dtype or y.device != x.device or (y.shape != x.shape and y.ndim != 0):
                raise ValueError("NPU in-place mul requires matching shape/dtype/device or a real scalar")
            _validate(y, dense=True)
            if y.untyped_storage()._cdata == x.untyped_storage()._cdata and y.stride() != x.stride():
                raise ValueError("NPU in-place mul rejects overlapping views with different strides")
        elif type(y) not in (int, float):
            raise ValueError("NPU in-place mul requires matching shape/dtype/device or a real scalar")
        # Read into a distinct result before overwriting any aliased dense input.
        return _copy(x, _binary_op(x, y, kind="mul"))


def _update_inputs(x, *others):
    if torch.is_grad_enabled():
        raise ValueError("NPU in-place updates require no_grad; differentiable optimizers are unsupported")
    _validate(x, dense=True)
    for value in others:
        _validate(value, dense=True)
        if value.shape != x.shape or value.dtype != x.dtype or value.device != x.device:
            raise ValueError("NPU optimizer updates require matching shape/dtype/device without broadcasting")
        if value.untyped_storage()._cdata == x.untyped_storage()._cdata and value.stride() != x.stride():
            raise ValueError("NPU optimizer updates reject overlapping views with different strides")


def _update_coefficient(value):
    if type(value) not in (int, float) or isfinite(value) and abs(value) > 3.4028234663852886e38:
        raise ValueError("NPU optimizer coefficient must be a real Python scalar representable in FP32")


def _lerp_inplace(x, end, weight):
    with _lock:
        synchronize()
        _update_inputs(x, end)
        _update_coefficient(weight)
        weight = struct.unpack('<f', struct.pack('<f', weight))[0]
        left, right = (_cast(_materialize(t), torch.float32) for t in (x, end))
        delta = _binary_op(right, left, kind="sub")
        # Use the stable branch in PyTorch's lerp formula, with FP32 opmath.
        base, scale = (left, weight) if abs(weight) < .5 else (right, weight-1)
        result = _binary_op(base, _binary_op(delta, scale, kind="mul"), kind="add")
        return _copy(x, _cast(result, x.dtype))


def _add_pointwise_inplace(x, a, b, *, value=1, kind):
    with _lock:
        synchronize()
        _update_inputs(x, a, b)
        _update_coefficient(value)
        left, first, second = (_cast(_materialize(t), torch.float32) for t in (x, a, b))
        # Match the CPU opmath order: value*a before multiplication/division by b.
        scaled = _binary_op(first, value, kind="mul")
        update = _binary_op(scaled, second, kind=kind)
        return _copy(x, _cast(_binary_op(left, update, kind="add"), x.dtype))


def _sqrt(x, *, out=None):
    return _math_unary(x, 'sqrt', out=out)


def _rms_norm(x, normalized_shape, weight=None, eps=None):
    with _lock:
        with torch.no_grad():
            _validate_float_read(x)
            shape = tuple(operator.index(n) for n in normalized_shape)
            if not shape or any(n < 1 for n in shape) or len(shape) > x.ndim or tuple(x.shape[-len(shape):]) != shape:
                raise ValueError('NPU RMSNorm normalized_shape must match nonempty trailing dimensions')
            if weight is not None:
                _validate_float_read(weight)
                if tuple(weight.shape) != shape:
                    raise ValueError('NPU RMSNorm weight must match normalized_shape')
            # Torch 2.14 uses the computation dtype for the default epsilon.
            epsilon = torch.finfo(torch.float32).eps if eps is None else eps
            if not isfinite(epsilon) or not 0 <= epsilon <= 3.4028234663852886e38 or 0 < epsilon < 2**-149:
                raise ValueError('NPU RMSNorm requires a nonnegative finite FP32 eps')
        axes = tuple(range(x.ndim-len(shape), x.ndim))
        value = x.float()
        result = value * torch.rsqrt((value*value).mean(axes, keepdim=True) + epsilon)
        if weight is not None:
            result = result * weight.float()
        return result.to(x.dtype)


def _math_unary(x, kind, *, out=None):
    with _lock:
        synchronize()
        _validate_float_read(x)
        if kind not in _MATH_UNARY:
            raise ValueError('Unknown NPU unary math operation')
        _check_out(out, tuple(x.shape), x.dtype, (x,))
        value = _cast(_materialize(x), torch.float32)
        result = _cast(_execute(kind, value, value if kind in _MATH_UNARY[:5] else None), x.dtype)
        return _copy(out, result) if out is not None else result


def _maximum(x, y, *, out=None):
    with _lock:
        synchronize()
        for tensor in (x, y):
            _validate(tensor, dense=True)
        if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device:
            raise ValueError("NPU maximum requires matching shape/dtype/device without broadcasting")
        if out is not None:
            _validate(out, dense=True)
            if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
                raise ValueError("NPU maximum output must match shape/dtype/device")
            for tensor in (x, y):
                if out.untyped_storage()._cdata == tensor.untyped_storage()._cdata and out.stride() != tensor.stride():
                    raise ValueError("NPU maximum rejects overlapping output views with different strides")
        left, right = (_cast(_materialize(t), torch.float32) for t in (x, y))
        result = _cast(_execute("maximum", left, right), x.dtype)
        return _copy(out, result) if out is not None else result


def _extreme(x, y, *, kind, out=None):
    with _lock:
        synchronize()
        _validate_float_read(x); _validate_float_read(y)
        if (kind == 'maximum' and x.shape == y.shape and x.dtype == y.dtype and
            all(_is_dense(t) and t.storage_offset() == 0 and t.numel()*t.element_size() == t.untyped_storage().nbytes() for t in (x, y))):
            # Preserve AdamW's existing exact-alias output path.
            return _maximum(x, y, out=out)
        shape = tuple(torch.broadcast_shapes(x.shape, y.shape))
        if len(shape) > 5 or prod(shape) > MAX_TENSOR_ELEMENTS:
            raise ValueError('NPU extrema broadcast exceeds storage bound')
        dtype = torch.result_type(x, y)
        _check_out(out, shape, dtype, (x, y))
        left = _broadcast(_cast(_materialize(x), dtype), shape)
        right = _broadcast(_cast(_materialize(y), dtype), shape)
        choose = _logical(_compare(left, right, kind='gt' if kind == 'maximum' else 'lt'),
                          _compare(left, left, kind='ne'), kind='or')
        result = _where(choose, left, right)
        return _copy(out, result) if out is not None else result


def _clamp_spec(x, minimum, maximum, out=None):
    _validate_float_read(x)
    if minimum is None and maximum is None:
        raise ValueError('NPU clamp requires at least one bound')
    bounds = [v for v in (minimum, maximum) if v is not None]
    tensors = [x]
    for value in bounds:
        if isinstance(value, torch.Tensor):
            _validate_float_read(value); tensors.append(value)
        elif type(value) not in (int, float, bool):
            raise ValueError('NPU clamp bounds must be real scalars or floating NPU tensors')
    if any(isinstance(v, torch.Tensor) for v in bounds) and any(not isinstance(v, torch.Tensor) for v in bounds):
        raise ValueError('NPU clamp does not mix Tensor and scalar bounds')
    ranked = [t for t in tensors if t.ndim] or tensors
    dtype = ranked[0].dtype
    for tensor in ranked[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    shape = tuple(torch.broadcast_shapes(*(v.shape for v in tensors)))
    if len(shape) > 5 or prod(shape) > MAX_TENSOR_ELEMENTS:
        raise ValueError('NPU clamp broadcast exceeds storage bound')
    for value in bounds:
        if not isinstance(value, torch.Tensor):
            torch.full((), value, dtype=dtype, device='cpu')
    _check_out(out, shape, dtype, tensors)
    return shape, dtype


def _clamp(x, min=None, max=None, *, out=None):
    with _lock:
        synchronize()
        shape, dtype = _clamp_spec(x, min, max, out)
        result = _broadcast(_cast(_materialize(x), dtype), shape)
        for value, kind in ((min, 'maximum'), (max, 'minimum')):
            if value is not None:
                bound = (_cast(_materialize(value), dtype) if isinstance(value, torch.Tensor) else _scalar(value, dtype))
                result = _extreme(result, bound, kind=kind)
        return _copy(out, result) if out is not None else result


def _clamp_inplace(x, min=None, max=None):
    with _lock:
        _validate(x, dense=True)
        shape, dtype = _clamp_spec(x, min, max)
        if shape != tuple(x.shape) or dtype != x.dtype:
            raise ValueError('NPU inplace clamp requires unchanged shape and dtype')
        return _copy(x, _clamp(x, min, max))


def _cast(x, dtype, out=None):
    if torch.bool in (x.dtype, dtype):
        _validate(x, storage=True)
        _check_out(out, tuple(x.shape), dtype, [x])
        result = x if x.dtype == dtype else _mask_cast(x, dtype)
        return _copy(out, result) if out is not None else result
    _validate(x)
    if not x.numel():return torch.empty(tuple(x.shape),dtype=dtype,device=x.device) if out is None else out
    if dtype not in bridge._DTYPES:
        raise ValueError("NPU conversion supports FP16/FP32 only")
    if x.numel()>_storage_limit(dtype):raise ValueError('NPU cast result exceeds target dtype byte storage bound')
    if out is not None:
        _validate(out)
        if out.shape != x.shape or out.dtype != dtype:
            raise ValueError("NPU cast output has the wrong shape or dtype")
    if x.dtype == dtype:
        return _copy(out, x) if out is not None and out is not x else x
    if x.numel() > MAX_ELEMENTS:
        return _map_tiles(lambda tile: _cast(tile, dtype), [x], tuple(x.shape), dtype, out)
    # Casts only depend on element count; flatten complete bindings to share graphs across ranks.
    shape = (x.numel(),)
    key = "cast", bridge._DTYPES[x.dtype][0], shape
    raw = _run(key, [x.view(shape)], shape, torch.float32,
               out.view(shape) if dtype == torch.float32 and out is not None else None)
    if dtype == torch.float32:
        return out if out is not None else raw.view(tuple(x.shape))
    # Metadata-only reinterpretation. Low 16-bit lane compaction runs as NPU DMA.
    transport = raw.view(-1).view(torch.float16).view(x.numel(), 2)
    result = _run(("halfcode_slice", "f16", (x.numel(),)), [transport], (x.numel(), 1), dtype,
                  out.view(x.numel(), 1) if out is not None else None)
    return out if out is not None else result.view(tuple(x.shape))


def _copy(dst, src, non_blocking=False):
    with _lock:
        synchronize()
        if non_blocking or dst.shape != src.shape or dst.dtype not in _STORAGE_DTYPES or src.dtype not in _STORAGE_DTYPES:
            raise ValueError("NPU copy requires identical shapes, supported storage dtype, and synchronous execution")
        if torch.is_grad_enabled() and (dst.requires_grad or src.requires_grad):
            raise ValueError("NPU device autograd is not implemented")
        if dst.device.type not in ("cpu", "intel_npu") or src.device.type not in ("cpu", "intel_npu"):
            raise ValueError("Only CPU/intel_npu copies are implemented")
        if dst.device.type == 'intel_npu':
            _validate(dst, dense=True, storage=True)
        if src.device.type == 'intel_npu':
            _validate_read(src)
        if (dst.device.type == 'cpu' and not dst.is_contiguous() or
                any(t.is_neg() or t.is_conj() for t in (dst, src))):
            raise ValueError("Copy requires a contiguous destination and resolved source value bits")
        if not src.numel():return dst
        if torch.bool in (src.dtype, dst.dtype) and src.dtype != dst.dtype:
            if src.device.type == 'cpu':
                uploaded = torch.empty(src.shape, device='intel_npu', dtype=src.dtype)
                _C.copy(uploaded, src, False)
                src = uploaded
            return _copy(dst, _mask_cast(_materialize(src), dst.dtype))
        if torch.int64 in (dst.dtype, src.dtype):
            if src.dtype != dst.dtype or not dst.is_contiguous():
                raise ValueError("NPU INT64 copy requires identical dtype and a contiguous destination")
            if src.device.type == 'intel_npu':
                src = _materialize(src)
            if src.device.type == dst.device.type == 'intel_npu':
                if src.data_ptr() != dst.data_ptr():
                    source = src.view(-1).view(torch.float32)
                    target = dst.view(-1).view(torch.float32)
                    shape = tuple(source.shape)
                    _run(("broadcast", "f32", shape, shape), [source], shape, torch.float32, out=target)
                return dst
            return _C.copy(dst, src, False)
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
            shape = tuple(src.shape) if src.ndim<=4 else (src.numel(),)
            shape = shape or (1,)
            return _run(("broadcast", _dma_dtype(src.dtype), shape, shape),
                        [src.view(shape)], shape, src.dtype, out=dst.view(shape)).view(dst.shape)
        return _C.copy(dst, src, False)


def _clone(x, *, memory_format=None):
    with _lock:
        synchronize()
        _validate_read(x)
        if memory_format not in (None, torch.preserve_format, torch.contiguous_format):
            raise ValueError('NPU clone supports preserve_format or contiguous_format')
        result = torch.empty_like(x, memory_format=memory_format or torch.preserve_format)
        if not result.is_contiguous():
            # Copy through complete contiguous physical views, including INT64.
            base, axes = _physical_view(result)
            _copy(base, _materialize(x.permute(axes)))
            return result
        return _copy(result, x)


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
            if x.dtype in (torch.int64, torch.bool):
                limit = _storage_limit(x.dtype)
                if x.ndim > MAX_RANK or not 0 <= x.numel() <= limit:
                    raise ValueError("NPU integer/bool storage exceeds its rank or element bound")
            else:
                if x.dtype not in _STORAGE_DTYPES or x.ndim>MAX_RANK or x.numel()>_storage_limit(x.dtype):raise ValueError('NPU upload exceeds dtype/rank/storage bounds')
        else:
            _validate_read(x)
        if (layout not in (None, torch.strided) or pin_memory or non_blocking or
            memory_format not in (None, torch.preserve_format, torch.contiguous_format)):
            raise ValueError("NPU conversion requires synchronous contiguous, unpinned, strided tensors")
        if out is not None:
            return _copy(out, x)
        dtype = dtype or x.dtype
        device = target_device
        if dtype not in _STORAGE_DTYPES or device.type not in ("cpu", "intel_npu"):
            raise ValueError("NPU conversion requires supported storage dtype and CPU/intel_npu")
        if device.type == "intel_npu":
            _check_device(device)
        if x.dtype == dtype == torch.int64:
            return _copy(torch.empty(tuple(x.shape), dtype=dtype, device=device), x)
        if x.device.type == "cpu":
            return _copy(torch.empty(tuple(x.shape), dtype=dtype, device=device), x)
        converted = _cast(_materialize(x), dtype)
        if device.type == "intel_npu" and converted is not x:
            return converted
        result = torch.empty(tuple(x.shape), dtype=dtype, device=device)
        return _copy(result, converted)


def _select_rows(matrix,k,largest=True,*,indices=None,pool=False,offset=0):
    rows,width=matrix.shape;size=selection_packet(rows,width,k,indices is not None)
    header=_concat([_scalar(float(v),torch.float32).view(1) for v in
                    (rows,width,k,int(largest),int(indices is not None),offset)],(6,),torch.float32)
    parts=[header,matrix.view(-1)]
    if size>6+matrix.numel():parts.append(_broadcast(_scalar(0.,torch.float32),(size-6-matrix.numel(),)))
    packet=_concat(parts,(1,size),torch.float32);second=packet
    if indices is not None:
        words=indices.view(-1).view(torch.float32)
        second=words.view(1,size) if words.numel()==size else _concat(
            [words,_broadcast(_scalar(0.,torch.float32),(size-words.numel(),))],(1,size),torch.float32)
    result=_run(('pool-select-f32' if pool else 'topk-f32','f32',(1,size)),[packet,second],(1,size),torch.float32)
    count=rows*k
    return _slice(result,0,count).view(rows,k),_slice(result,count,2*count).view(torch.int64).view(rows,k)


def _topk_spec(x,k,dim,values=None,indices=None):
    _validate_read(x);k=operator.index(k);dim=operator.index(dim);rank=x.ndim or 1
    if x.dtype not in (torch.float16,torch.float32,torch.int64,torch.bool):raise NotImplementedError('NPU topk currently requires FP16/FP32/INT64/BOOL')
    if x.numel()>MAX_TENSOR_ELEMENTS:raise ValueError('NPU topk input exceeds the 16777216-element metadata bound')
    if not -rank<=dim<rank:raise IndexError('NPU topk axis out of range')
    axis=dim%rank;width=x.shape[axis] if x.ndim else 1
    if not 0<=k<=width:raise ValueError('NPU topk requires 0<=k<=axis length')
    shape=list(x.shape);shape[axis:axis+1]=[k] if x.ndim else [];shape=tuple(shape)
    if prod(shape)>MAX_TENSOR_ELEMENTS:raise ValueError('NPU topk output exceeds the 16777216-element metadata bound')
    _check_out(values,shape,x.dtype,(x,indices));_check_out(indices,shape,torch.int64,(x,values))
    if (values is None)!=(indices is None):raise ValueError('NPU topk requires both output tensors')
    return k,axis,width,shape


def _topk(x,k,dim=-1,largest=True,sorted=True,*,values=None,indices=None):
    with _lock:
        synchronize();k,axis,width,shape=_topk_spec(x,k,dim,values,indices)
        if x.ndim==0 and k==0:
            value=_clone(x);index=_scalar(0,torch.int64).clone()
            return (_copy(values,value),_copy(indices,index)) if values is not None else (value,index)
        if not prod(shape):
            return (torch.empty(shape,dtype=x.dtype,device=x.device),torch.empty(shape,dtype=torch.int64,device=x.device)) if values is None else (values,indices)
        if x.dtype not in bridge._DTYPES or k>128 or x.numel()*k>16777216 or width>1048576:
            from torch_npu_sort import sort
            v,i=sort(x,dim,largest)
            value=_materialize(v.narrow(axis,0,k));index=_materialize(i.narrow(axis,0,k))
            return (_copy(values,value),_copy(indices,index)) if values is not None else (value,index)
        order=tuple(a for a in range(x.ndim) if a!=axis)+(axis,) if x.ndim else (0,)
        matrix=_cast(_materialize(x.permute(order) if x.ndim else x.view(1)),torch.float32).view(-1,width)
        rows=matrix.shape[0];parts=[];ids=[]
        if width<=65530:
            batch=min(65530//width,65536//(3*k))
            for start in range(0,rows,batch):
                count=min(batch,rows-start)
                v,i=_select_rows(_slice(matrix,start*width,count*width).view(count,width),k,largest)
                parts.append(v.view(-1));ids.append(i.view(-1))
        else:
            for row in range(rows):
                candidates=[];candidate_ids=[]
                for start in range(0,width,65530):
                    count=min(65530,width-start);take=min(k,count)
                    v,i=_select_rows(_slice(matrix,row*width+start,count).view(1,count),take,largest,offset=start)
                    candidates.append(v.view(-1));candidate_ids.append(i.view(-1))
                count=sum(v.numel() for v in candidates)
                v,i=_select_rows(_concat(candidates,(1,count),torch.float32),k,largest,
                                 indices=_cat(candidate_ids,0).view(1,count))
                parts.append(v.view(-1));ids.append(i.view(-1))
        matrix_shape=tuple(shape[a] for a in order) if x.ndim else (1,)
        value=_cast(_concat(parts,matrix_shape,torch.float32),x.dtype)
        index=_cat(ids,0).view(matrix_shape)
        if x.ndim:
            inverse=tuple(order.index(a) for a in range(x.ndim))
            value=_materialize(value.permute(inverse));index=_materialize(index.permute(inverse))
        else:value=value.view(());index=index.view(())
        # sorted=False permits a sorted valid selection too; ties are deterministic here.
        return (_copy(values,value),_copy(indices,index)) if values is not None else (value,index)


def _selection_accumulate(gradient,indices,per,width,base,span,total,start,initial,*,half=False,accumulate=True,allow_padding=False,wide=True):
    length=gradient.numel();size=max(9+3*length,span+1)
    fields=(length,start,per,width,base,span,int(half)+2*int(not accumulate)+4*int(allow_padding),total)
    header=(torch.frombuffer(bytearray(struct.pack('<8I',*fields)),dtype=torch.float32).to(gradient.device)
            if wide else _concat([_scalar(float(v),torch.float32).view(1) for v in fields],(8,),torch.float32))
    g=_cast(_materialize(gradient),torch.float32).view(-1)
    words=_materialize(indices).view(-1).view(torch.float32)
    packet=_concat([header,g,words,_broadcast(_scalar(0.,torch.float32),(size-8-3*length,))],(1,size),torch.float32)
    second=_concat([initial,_broadcast(_scalar(0.,torch.float32),(size-span,))],(1,size),torch.float32)
    result=_run(('selection-backward-u32' if wide else 'selection-backward-f32','f32',(1,size)),[packet,second],(1,size),torch.float32)
    if _slice(result,size-1,1).item()!=0:raise IndexError('NPU selection gradient index is out of bounds')
    return _slice(result,0,span)


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
            initial_value=_materialize(initial_value).view(-1)
        if not count:return _clone(initial_value).view(shape) if initial_value is not None else _fill(torch.empty(shape,dtype=gradient.dtype,device=gradient.device),0.)
        if not per or not width or count%per or total%width or total//width!=count//per:
            raise ValueError('NPU selection gradient geometry mismatch')
        wide=max(total,count,per,width)>MAX_TENSOR_ELEMENTS
        target_batch,source_batch=(61440,21504) if wide else (65535,21842)
        g=_cast(_materialize(gradient),torch.float32).view(-1);ids=_materialize(indices).view(-1)
        def pieces():
            for base in range(0,total,target_batch):
                span=min(target_batch,total-base)
                initial=(_cast(_slice(initial_value,base,span),torch.float32) if initial_value is not None
                         else _broadcast(_scalar(0.,torch.float32),(span,)))
                for start in range(0,count,source_batch):
                    length=min(source_batch,count-start)
                    initial=_selection_accumulate(_slice(g,start,length),_slice(ids,start,length),per,width,base,span,total,start,initial,
                        half=gradient.dtype==torch.float16,accumulate=accumulate,allow_padding=allow_padding,wide=wide)
                yield _cast(initial,gradient.dtype)
        return (_stream_concat(pieces(),shape,dtype=gradient.dtype) if wide else _concat(list(pieces()),shape,gradient.dtype))


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
                yield _selection_backward(g,i,(count,width),per,width,accumulate=accumulate)
        return _stream_concat(pieces(),shape,dtype=gradient.dtype)


def _max_pool2d_spec(x,kernel_size,stride,padding,dilation,ceil_mode,out=None,indices=None):
    _validate_float_read(x)
    if x.ndim not in (3,4) or x.shape[-3]<1:raise ValueError('NPU MaxPool2d requires CHW/NCHW with positive channels')
    params=max_pool2d_geometry(tuple(x.shape[-2:]),kernel_size,stride,padding,dilation,ceil_mode)
    shape=(*x.shape[:-2],*params[-1])
    if prod(shape)>min(MAX_INT64_ELEMENTS,_storage_limit(x.dtype)):
        raise ValueError('NPU MaxPool output exceeds dtype storage bounds')
    _check_out(out,shape,x.dtype,(x,indices));_check_out(indices,shape,torch.int64,(x,out))
    if (out is None)!=(indices is None):raise ValueError('NPU MaxPool requires both out tensors')
    return params,shape


def _max_pool2d(x,kernel_size,stride=(),padding=0,dilation=1,ceil_mode=False,*,out=None,indices=None):
    import numpy as np
    with _lock:
        synchronize();params,shape=_max_pool2d_spec(x,kernel_size,stride,padding,dilation,ceil_mode,out,indices)
        if not prod(shape):
            return ((torch.empty(shape,dtype=x.dtype,device=x.device),torch.empty(shape,dtype=torch.int64,device=x.device))
                    if out is None else (out,indices))
        if prod(shape)*prod(params[0])>=65536:
            from torch_npu_pool_dense import forward
            return forward(x,params,shape,out=out,indices=indices)
        plane=prod(x.shape[-2:]);outputs=prod(params[-1]);width=prod(params[0]);planes=x.numel()//plane
        batch=min(65530//width,65536//(2*width),65536//3)
        data=_materialize(x).view(-1);values=[];ids=[]
        for start in range(0,planes*outputs,batch):
            count=min(batch,planes*outputs-start);locations=max_pool_windows(tuple(x.shape[-2:]),params,start,count)
            channels=np.arange(start,start+count,dtype=np.int64)//outputs
            absolute=np.where(locations<0,-1,locations+channels[:,None]*plane)
            windows=_cast(_gather_window(data,absolute,1),torch.float32)
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
        _validate_read(indices)
        if indices.dtype!=torch.int64 or indices.shape!=gradient.shape:raise ValueError('NPU pool gradient requires matching INT64 indices')
        if known_windows and prod(x.shape[-2:])>65535 and gradient.numel():
            from torch_npu_pool_dense import backward
            result=backward(gradient,indices,tuple(x.shape),params)
        else:result=_selection_backward_planes(gradient,indices,tuple(x.shape),prod(params[-1]),prod(x.shape[-2:]))
        return _copy(grad_input,result) if grad_input is not None else result


def _max_pool1d(x,kernel_size,stride=(),padding=(0,),dilation=(1,),ceil_mode=False,*,return_indices=False):
    if x.ndim not in (2,3) or len(kernel_size)!=1 or any(len(v) not in (0,1) for v in (stride,padding,dilation)):
        raise ValueError('NPU MaxPool1d requires CL/NCL and one spatial parameter')
    values,indices=torch.ops.aten.max_pool2d_with_indices(x.unsqueeze(-2),(1,kernel_size[0]),
        (1,stride[0] if stride else kernel_size[0]),(0,padding[0] if padding else 0),(1,dilation[0] if dilation else 1),ceil_mode)
    values,indices=values.squeeze(-2),indices.squeeze(-2)
    return (values,indices) if return_indices else values


def _relu_inplace(x,*,bump_version=True):
    with _lock:
        _validate(x,dense=True)
        _copy(x,_execute('relu',x))
        if bump_version:torch.autograd.graph.increment_version(x)
        return x


def _row_extreme(matrix, kind, *, offset=0, indices=None):
    rows, width = matrix.shape
    size = row_extrema_packet(rows, width, indices is not None)
    header = _concat([_scalar(float(v), torch.float32).view(1) for v in
                      (rows, width, offset, int(indices is not None))], (4,), torch.float32)
    pieces = [header, matrix.view(-1)]
    if size > 4+matrix.numel(): pieces.append(_broadcast(_scalar(0., torch.float32), (size-4-matrix.numel(),)))
    packet = _concat(pieces, (1,size), torch.float32)
    second = packet
    if indices is not None:
        words = indices.view(-1).view(torch.float32)
        second = (words.view(1,size) if size == words.numel() else
                  _concat([words, _broadcast(_scalar(0., torch.float32), (size-words.numel(),))], (1,size), torch.float32))
    result = _run(('row-'+kind+'-f32', 'f32', (1,size)), [packet,second], (1,size), torch.float32)
    return _slice(result, 0, rows), _slice(result, rows, 2*rows).view(torch.int64)


def _extreme_reduce(x, dim=None, keepdim=False, *, kind, arg=False, out=None, pair=False):
    with _lock:
        synchronize(); _validate_read(x)
        if x.numel()>_storage_limit(x.dtype):raise ValueError('NPU row-extrema input exceeds dtype storage bounds')
        if x.dtype not in bridge._DTYPES:
            from torch_npu_sort import integer_extreme
            return integer_extreme(x,dim,keepdim,kind,arg,pair,out)
        if not x.numel():
            axes,shape=reduction_geometry(tuple(x.shape),dim,keepdim)
            if not prod((tuple(x.shape) or (1,))[a] for a in axes):raise ValueError('Cannot reduce an empty extrema axis')
            _check_out(out,shape,torch.int64 if arg else x.dtype,(x,))
            value=torch.empty(shape,dtype=x.dtype,device=x.device);index=torch.empty(shape,dtype=torch.int64,device=x.device)
            return (value,index) if pair else out if out is not None else index if arg else value
        if arg and dim is not None: dim = operator.index(dim)
        axes, shape = reduction_geometry(tuple(x.shape), dim, keepdim)
        dtype = torch.int64 if arg else x.dtype
        if (arg or pair) and prod(shape) > MAX_INT64_ELEMENTS:
            raise ValueError('NPU arg reduction output exceeds INT64 storage bound')
        _check_out(out, shape, dtype, (x,))
        logical = tuple(x.shape) or (1,)
        order = tuple(a for a in range(len(logical)) if a not in axes)+axes
        width = prod(logical[a] for a in axes); rows = x.numel()//width
        matrix = _materialize(x.view(1) if x.ndim == 0 else x.permute(order)).view(rows,width)
        values, ids = [], []
        if width <= 65532:
            batch = min(21845,65532//width)
            for start in range(0,rows,batch):
                count = min(batch,rows-start)
                v,i = _row_extreme(_cast(_slice(matrix,start*width,count*width),torch.float32).view(count,width),kind)
                if arg or pair: ids.append(i)
                if not arg or pair: values.append(v)
        else:
            for row in range(rows):
                partial_values, partial_ids = [], []
                block=61440 if width>MAX_TENSOR_ELEMENTS or matrix.dtype==torch.float16 else 65532
                for start in range(0,width,block):
                    count = min(block,width-start)
                    offset=start if start+count<=1048576 else 0
                    v,i = _row_extreme(_cast(_slice(matrix,row*width+start,count),torch.float32).view(1,count),kind,offset=offset)
                    if start!=offset:i=_binary_op(i,start,kind='add')
                    partial_values.append(v); partial_ids.append(i)
                count = len(partial_values)
                v,i = _row_extreme(_concat(partial_values,(1,count),torch.float32),kind,
                                   indices=_cat(partial_ids,0).view(1,count))
                if arg or pair: ids.append(i)
                if not arg or pair: values.append(v)
        if pair:return _cast(_concat(values,shape,torch.float32),x.dtype),_cat(ids,0).view(shape)
        result = _cat(ids,0).view(shape) if arg else _cast(_concat(values,shape,torch.float32),x.dtype)
        return _copy(out,result) if out is not None else result


def _arg_reduce(x, dim=None, keepdim=False, *, kind, out=None):
    # Integer results never retain an autograd graph, even for floating leaves.
    with torch.no_grad(): return _extreme_reduce(x,dim,keepdim,kind=kind,arg=True,out=out)


def _float_predicate(x, *, kind, out=None):
    with _lock, torch.no_grad():
        _validate_float_read(x); _check_out(out,tuple(x.shape),torch.bool,(x,))
        if kind == 'isnan': result = _compare(x,x,kind='ne')
        elif kind == 'isinf': result = _compare(_execute('abs',_materialize(x)),float('inf'),kind='eq')
        else:
            result = _logical(_compare(x,x,kind='eq'),
                              _compare(_execute('abs',_materialize(x)),float('inf'),kind='ne'),kind='and')
        return _copy(out,result) if out is not None else result


def _vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None):
    with _lock:
        synchronize(); _validate_float_read(x)
        if type(ord) not in (int,float) or ord not in (0,1,2,float('inf'),float('-inf')):
            raise ValueError('NPU vector norm supports orders 0, 1, 2 and +/-inf')
        axes, shape = reduction_geometry(tuple(x.shape),dim,keepdim)
        dtype = x.dtype if dtype is None else dtype
        if dtype not in bridge._DTYPES or (x.dtype == torch.float32 and dtype == torch.float16):
            raise ValueError('NPU vector norm requires FP16/FP32 dtype without narrowing')
        _check_out(out,shape,dtype,(x,))
        value = _cast(_materialize(x),torch.float32)
        logical = tuple(x.shape) or (1,)
        if ord == 2 and prod(logical[a] for a in axes) == 1:
            # Torch uses abs for singleton reductions, avoiding a spurious square overflow.
            result = _execute('abs',value).view(shape)
        elif ord == 0:
            result = _reduce(_mask_cast(_compare(value,0.,kind='ne'),torch.float32),axes,keepdim,kind='sum')
        elif ord in (float('inf'),float('-inf')):
            result = _extreme_reduce(_execute('abs',value),axes,keepdim,kind='max' if ord > 0 else 'min')
        else:
            data = _execute('abs',value) if ord == 1 else _binary_op(value,value,kind='mul')
            result = _reduce(data,axes,keepdim,kind='sum')
            if ord == 2: result = _math_unary(result,'sqrt')
        return _cast(result,dtype,out=out)


def _reduce(x, dim=None, keepdim=False, *, dtype=None, out=None, kind):
    with _lock:
        synchronize()
        _validate(x, dense=True)
        axes,shape = reduction_geometry(tuple(x.shape),dim,keepdim)
        result_dtype = dtype or (out.dtype if out is not None else x.dtype)
        if result_dtype not in bridge._DTYPES:
            raise ValueError("NPU reduction dtype must be FP16/FP32")
        if out is not None:
            _validate(out)
            if out.shape != shape or out.dtype != result_dtype:
                raise ValueError("NPU reduction out requires the exact result shape and dtype")
            if out.untyped_storage()._cdata == x.untyped_storage()._cdata:
                raise ValueError("NPU reduction out may not alias its input")
        # Torch sum casts each input to its requested dtype; CPU mean instead
        # accumulates FP32 inputs in FP32 even when its output is FP16.
        if not x.numel():
            result=torch.empty(shape,dtype=result_dtype,device=x.device)
            return _fill(result,0. if kind=='sum' else float('nan')) if out is None else _copy(out,_fill(result,0. if kind=='sum' else float('nan')))
        values = x if x.is_contiguous() else _materialize(x)
        values = _cast(values, result_dtype) if kind == "sum" else values
        values = _cast(values, torch.float32)
        key = kind, "f32", tuple(x.shape) or (1,), axes
        destination = out if out is not None and result_dtype == torch.float32 else None
        result = (_reduce_large(values, axes, shape, kind, destination) if x.numel() > MAX_ELEMENTS or x.ndim>4 else
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
    return _binary_op(x, y, kind="add", out=out, alpha=alpha)


def _sub(x, y, *, alpha=1, out=None):
    return _binary_op(x, y, kind="sub", out=out, alpha=alpha)


def _rsub(x, y, alpha=1, *, out=None):
    return _binary_op(y, x, kind="sub", out=out, alpha=alpha)


_aten = torch.library.Library("aten", "IMPL", "PrivateUse1")
_aten.impl('topk',_topk);_aten.impl('topk.values',_topk)
_aten.impl('max_pool2d_with_indices',_max_pool2d);_aten.impl('max_pool2d_with_indices.out',_max_pool2d)
_aten.impl('max_pool2d_with_indices_backward',_max_pool2d_backward)
_aten.impl('max_pool2d_with_indices_backward.grad_input',_max_pool2d_backward)
_aten.impl('max_pool2d',lambda x,kernel_size,stride=(),padding=0,dilation=1,ceil_mode=False:
           _max_pool2d(x,kernel_size,stride,padding,dilation,ceil_mode)[0])
_aten.impl('max_pool1d',_max_pool1d)
_aten.impl('max_pool1d_with_indices',lambda x,kernel_size,stride=(),padding=(0,),dilation=(1,),ceil_mode=False:
           _max_pool1d(x,kernel_size,stride,padding,dilation,ceil_mode,return_indices=True))
_aten.impl('relu_',_relu_inplace)
for _name in ('isnan','isinf','isfinite'):
    def _predicate(x,*,out=None,kind=_name): return _float_predicate(x,kind=kind,out=out)
    _aten.impl(_name,_predicate)
    if 'out' in getattr(torch.ops.aten,_name)._schemas: _aten.impl(_name+'.out',_predicate)
for _kind in ('max','min'):
    def _arg(x,dim=None,keepdim=False,*,out=None,kind=_kind):
        return _arg_reduce(x,dim,keepdim,kind=kind,out=out)
    def _extreme_reduction(x,dim=(),keepdim=False,*,out=None,kind=_kind):
        return _extreme_reduce(x,dim,keepdim,kind=kind,out=out)
    _aten.impl('arg'+_kind,_arg); _aten.impl('arg'+_kind+'.out',_arg)
    _aten.impl('a'+_kind,_extreme_reduction); _aten.impl('a'+_kind+'.out',_extreme_reduction)
_aten.impl('linalg_vector_norm',_vector_norm)
_aten.impl('linalg_vector_norm.out',_vector_norm)
def _norm(x,p=2,dim=None,keepdim=False,*,dtype=None,out=None):
    return _vector_norm(x,2 if p is None else p,dim,keepdim,dtype=dtype,out=out)
for _name in ('Scalar','ScalarOpt_dim','ScalarOpt_dtype','ScalarOpt_dtype_out','Scalar_out','ScalarOpt_dim_dtype','dtype_out','out'):
    _aten.impl('norm.'+_name,_norm)
_aten.impl("copy_", _copy)
_aten.impl('clone', _clone)
for _upper,_name in ((True,'triu'),(False,'tril')):
    def _triangular(x,diagonal=0,*,out=None,upper=_upper):
        return _triangle(x,diagonal,upper=upper,out=out)
    _aten.impl(_name,_triangular)
    _aten.impl(_name+'.out',_triangular)
for _name in ('all','any'):
    def _bool_reduction(x,dim=None,keepdim=False,*,out=None,kind=_name):
        return _boolean_reduce(x,dim,keepdim,out=out,kind=kind)
    for _overload in ('','.dim','.dims','.out','.dims_out','.all_out'):
        _aten.impl(_name+_overload,_bool_reduction)
_aten.impl("_to_copy", _to_copy)
_aten.impl("_to_copy.out", _to_copy)
_aten.impl('narrow_copy.out', _narrow_copy_out)
for _kind in _COMPARISONS:
    def _comparison(x,y,*,out=None,kind=_kind):
        return _compare(x,y,kind=kind,out=out)
    for _overload in ('Tensor','Scalar','Tensor_out','Scalar_out'):
        _aten.impl(f'{_kind}.{_overload}',_comparison)
for _kind in ('not','and','or','xor'):
    def _logic(x,y=None,*,out=None,kind=_kind):
        return _logical(x,y,kind=kind,out=out)
    _aten.impl(f'logical_{_kind}',_logic)
    _aten.impl(f'logical_{_kind}.out',_logic)
    _aten.impl(f'bitwise_{_kind}'+('' if _kind=='not' else '.Tensor'),_logic)
    _aten.impl(f'bitwise_{_kind}'+('.out' if _kind=='not' else '.Tensor_out'),_logic)
for _overload in ('self','ScalarOther','ScalarSelf','Scalar','self_out'):
    _aten.impl(f'where.{_overload}',_where)
for _overload in ('Scalar','Tensor'):
    _aten.impl(f'masked_fill.{_overload}',_masked_fill)
    _aten.impl(f'masked_fill.{_overload}_out',_masked_fill_out)
    _aten.impl(f'masked_fill_.{_overload}',_masked_fill_inplace)
for _name, _function in (('cat', _cat), ('stack', _stack), ('slice_backward', _slice_backward),
                         ('select_backward', _select_backward)):
    _aten.impl(_name, _function)
    _aten.impl(f'{_name}.out', _function)
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
_aten.impl("mm", _matmul)
_aten.impl("mm.out", _matmul)
_aten.impl("matmul", _matmul_nd)
_aten.impl("matmul.out", _matmul_nd)
_aten.impl('index_select', _index_select)
_aten.impl('index_select.out', _index_select)
_aten.impl('embedding', _embedding)
for _kind in ('dot', 'mv'):
    def _vector(x, y, *, out=None, kind=_kind):
        return _vector_product(x, y, kind=kind, out=out)
    _aten.impl(_kind, _vector)
    _aten.impl(_kind+'.out', _vector)
_aten.impl("addmm", _addmm)
_aten.impl("bmm", _bmm)
_aten.impl("bmm.out", _bmm)
_aten.impl('baddbmm', _baddbmm)
_aten.impl('baddbmm.out', _baddbmm)
_aten.impl('_native_multi_head_attention', _native_multi_head_attention)
_aten.impl('_transformer_encoder_layer_fwd', _transformer_encoder_layer_fwd)
_aten.impl("addmm.out", _addmm)
_aten.impl("linear", _linear)
_aten.impl("convolution", _convolution)
_aten.impl('gather', _gather)
_aten.impl('gather.out', _gather)
_aten.impl('scatter_add', _scatter_add)
_aten.impl('scatter_add.out', _scatter_add)
_aten.impl('scatter_add_', _scatter_add_inplace)
_aten.impl('take_along_dim', _take_along_dim)
_aten.impl('take_along_dim.out', _take_along_dim)
_aten.impl("mse_loss", _mse_loss)
_aten.impl('mse_loss.out',_mse_loss)
_aten.impl('mse_loss_backward',_mse_loss_backward)
_aten.impl('mse_loss_backward.grad_input',_mse_loss_backward)
_aten.impl("_log_softmax", _log_softmax)
_aten.impl("_log_softmax.out", _log_softmax)
_aten.impl("_log_softmax_backward_data", _log_softmax_backward)
_aten.impl("_log_softmax_backward_data.out", _log_softmax_backward)
_aten.impl("_softmax", _softmax)
_aten.impl("_softmax.out", _softmax)
_aten.impl("_softmax_backward_data", _softmax_backward)
_aten.impl("_softmax_backward_data.out", lambda gradient, output, dim, input_dtype, *, grad_input:
           _softmax_backward(gradient, output, dim, input_dtype, out=grad_input))
_aten.impl('scaled_dot_product_attention', _sdpa)
_aten.impl("nll_loss_forward", lambda x, target, weight=None, reduction=1, ignore_index=-100:
           _nll_forward(x, target, weight, reduction, ignore_index)[:2])
_aten.impl('nll_loss_forward.output',_nll_native_forward)
_aten.impl('nll_loss_backward',_nll_native_backward)
_aten.impl('nll_loss_backward.grad_input',_nll_native_backward)
_aten.impl('nll_loss2d_forward',lambda *args,**kwargs:_nll_native_forward(*args,spatial=True,**kwargs))
_aten.impl('nll_loss2d_forward.output',lambda *args,**kwargs:_nll_native_forward(*args,spatial=True,**kwargs))
_aten.impl('nll_loss2d_backward',lambda *args,**kwargs:_nll_native_backward(*args,spatial=True,**kwargs))
_aten.impl('nll_loss2d_backward.grad_input',lambda *args,**kwargs:_nll_native_backward(*args,spatial=True,**kwargs))
for _name in ("nll_loss", "nll_loss_nd"):
    _aten.impl(_name, _nll_nd)
_aten.impl("cross_entropy_loss", _cross_entropy)
_aten.impl("layer_norm", lambda x, shape, weight=None, bias=None, eps=1e-5, cudnn_enable=True:
           _layer_norm(x, shape, weight, bias, eps)[0])
_aten.impl("native_layer_norm", _layer_norm)
_aten.impl("native_layer_norm_backward", _layer_norm_backward)
_aten.impl('rms_norm', _rms_norm)
_aten.impl("gelu", _gelu)
_aten.impl("gelu.out", _gelu)
_aten.impl("gelu_backward", _gelu_backward)
_aten.impl("gelu_backward.grad_input", _gelu_backward)
def _batch_norm_entry(x,w,b,rm,rv,training,momentum,eps,cudnn):
    from torch_npu_vision_extended import batch_norm
    return batch_norm(x,w,b,rm,rv,training,momentum,eps,cudnn)
_aten.impl('batch_norm',_batch_norm_entry)
for _name in ("adaptive_avg_pool2d", "_adaptive_avg_pool2d"):
    _aten.impl(_name, _adaptive_avg_pool2d)
_aten.impl('adaptive_avg_pool1d', _adaptive_avg_pool1d)
_aten.impl('adaptive_avg_pool1d.out', _adaptive_avg_pool1d)
for _name, _function in (('avg_pool2d', _avg_pool2d), ('avg_pool1d', _avg_pool1d)):
    _aten.impl(_name, _function)
    _aten.impl(_name+'.out', _function)
_aten.impl("fill_.Scalar", _fill)
_aten.impl("zero_", lambda x: _fill(x, 0))
_aten.impl('uniform_', _random_fill)
_aten.impl('normal_', lambda x, mean=0., std=1., *, generator=None:
           _random_fill(x,mean,std,generator=generator,kind='normal'))
_aten.impl('random_', lambda x, *, generator=None: _random_integer_fill(x,generator=generator))
_aten.impl('random_.from', lambda x, low, high=None, *, generator=None: _random_integer_fill(x,low,high,generator=generator))
_aten.impl('random_.to', lambda x, high, *, generator=None: _random_integer_fill(x,0,high,generator=generator))
_aten.impl('resize_', _resize_same)
_aten.impl('multinomial', _multinomial)
_aten.impl('multinomial.out', _multinomial)
_aten.impl('bernoulli_.float', _bernoulli_fill)
_aten.impl('bernoulli_.Tensor', _bernoulli_fill)
for _name in ('bernoulli','bernoulli.out','bernoulli.p','bernoulli.Tensor','bernoulli.Tensor_out','bernoulli.float_out'):
    _aten.impl(_name,_bernoulli)
for _name in ('randperm','randperm.generator','randperm.out','randperm.generator_out'):
    _aten.impl(_name,_randperm)
_aten.impl('native_dropout', _native_dropout)
_aten.impl('native_dropout_backward', _native_dropout_backward)
_aten.impl('dropout', _dropout)
_aten.impl("add_.Tensor", _add_inplace)
_aten.impl("add_.Scalar", _add_scalar_inplace)
_aten.impl("_local_scalar_dense", _local_scalar_dense)
_aten.impl("mul_.Tensor", _mul_inplace)
_aten.impl("mul_.Scalar", _mul_inplace)
_aten.impl("lerp_.Scalar", _lerp_inplace)
_aten.impl("addcmul_", lambda x, a, b, value=1: _add_pointwise_inplace(x, a, b, value=value, kind="mul"))
_aten.impl("addcdiv_", lambda x, a, b, value=1: _add_pointwise_inplace(x, a, b, value=value, kind="div"))
for _kind in _MATH_UNARY:
    def _math(x, *, out=None, kind=_kind):
        return _math_unary(x, kind, out=out)
    _aten.impl(_kind, _math)
    _aten.impl(_kind+'.out', _math)
for _kind in ('maximum', 'minimum'):
    def _extrema(x, y, *, out=None, kind=_kind):
        return _extreme(x, y, kind=kind, out=out)
    _aten.impl(_kind, _extrema)
    _aten.impl(_kind+'.out', _extrema)
for _name in ('clamp', 'clip'):
    for _suffix in ('', '.Tensor', '.out', '.Tensor_out'):
        _aten.impl(_name+_suffix, _clamp)
    _aten.impl(_name+'_', _clamp_inplace)
for _name in ('clamp_min', 'clamp_max'):
    def _one_bound(x, bound, *, out=None, lower=_name == 'clamp_min'):
        return _clamp(x, bound if lower else None, None if lower else bound, out=out)
    def _one_bound_inplace(x, bound, lower=_name == 'clamp_min'):
        return _clamp_inplace(x, bound if lower else None, None if lower else bound)
    for _suffix in ('', '.Tensor', '.out', '.Tensor_out'):
        _aten.impl(_name+_suffix, _one_bound)
    _aten.impl(_name+'_', _one_bound_inplace)
_aten.impl('hardtanh', lambda x, min_val=-1., max_val=1.: _clamp(x, min_val, max_val))
_aten.impl('hardtanh.out', lambda x, min_val=-1., max_val=1., out=None: _clamp(x, min_val, max_val, out=out))
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


def manual_seed(seed):
    with _lock:
        _rng.manual_seed(seed)


def initial_seed():
    with _lock:
        return _rng.initial_seed()


def seed():
    with _lock:
        return _rng.seed()


def get_rng_state(device=None):
    _check_device(device)
    with _lock:
        return _rng.get_state()


def set_rng_state(state, device=None):
    _check_device(device)
    with _lock:
        _rng.set_state(state)


def set_rng_state_all(states):
    states = list(states)
    if len(states) != 1:
        raise ValueError('NPU RNG requires exactly one device state')
    set_rng_state(states[0])


from torch_npu_amp import GradScaler as _GradScaler
from torch_npu_amp import register_grad_scaler as _register_grad_scaler

torch._register_device_module("intel_npu", SimpleNamespace(
    is_available=lambda: not _closed and not _C.stats()["context_failed"],
    device_count=lambda: 1, current_device=lambda: 0, synchronize=synchronize, empty_cache=empty_cache,
    set_graph_cache_budget=set_graph_cache_budget,
    _is_in_bad_fork=lambda: False, manual_seed_all=manual_seed, manual_seed=manual_seed,
    initial_seed=initial_seed, get_rng_state=get_rng_state, set_rng_state=set_rng_state,
    seed=seed, seed_all=seed, get_rng_state_all=lambda: [get_rng_state()], set_rng_state_all=set_rng_state_all,
    default_generators=(_rng,),
    amp=SimpleNamespace(GradScaler=_GradScaler),
    get_amp_supported_dtype=lambda: [torch.float16]))
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=False)
atexit.register(shutdown)
_register_grad_scaler()

# Register only the explicitly implemented first-order device derivatives.
import torch_npu_autograd
import torch_npu_sort
torch_npu_sort.register(_aten,torch_npu_autograd._library)
import torch_npu_order_stats
torch_npu_order_stats.register(_aten,torch_npu_autograd._library)

import torch_npu_nested
torch_npu_nested.register(_aten)

import torch_npu_flash_attention
torch.intel_npu.flash_attention=torch_npu_flash_attention.flash_attention
import torch_npu_vision
torch_npu_vision.register(_aten,torch_npu_autograd._library)
import torch_npu_vision_extended
torch_npu_vision_extended.register(_aten,torch_npu_autograd._library)

import torch_npu_grid
torch_npu_grid.register(_aten,torch_npu_autograd._library)
import torch_npu_compat_extended
torch_npu_compat_extended.register(_aten,torch_npu_autograd._library)

"""Synchronous NPU-resident buffers; this is not a PyTorch Tensor device yet.

Graphs share one Level Zero context. Only upload/download copy tensor payloads
on the host; intermediate graph outputs are rebound directly as graph inputs.
"""
import ctypes
import math
from pathlib import Path
from threading import RLock

import numpy as np
import npunlock as npu
import torch
import torch_npu_minimal as bridge

SESSION_DLL = Path(__file__).resolve().parent / "npunlock-probe/session-runtime/npunlock_session.dll"
_BINARY = {"add", "sub", "mul", "div"}
_UNARY = {"add1", "neg", "abs", "sign"}


class NpuBuffer:
    """Owned Level Zero allocation, with no exposed dangling NumPy views."""
    def __init__(self, runtime, native):
        self._runtime = runtime
        self._native = native

    @property
    def address(self):
        with self._runtime._lock:
            self._runtime._validate(self)
            return self._native.address

    def release(self):
        with self._runtime._lock:
            if not self._native._released:
                self._native.release()
                self._runtime._stats["live_buffers"] -= 1
                self._runtime._stats["live_buffer_bytes"] -= self._runtime.nbytes
                self._runtime._buffers.remove(self)


class NpuRuntime:
    """Fixed shape/dtype execution context for the validated Lunar Lake profile.

    CPU tensors are explicitly uploaded/downloaded. No silent arithmetic
    fallback, autograd, broadcasting, in-place output, or dtype casts here.
    """
    def __init__(self, shape, dtype=torch.float32, *, timeout_ms=10_000):
        shape = tuple(shape)
        if (dtype not in bridge._DTYPES or len(shape) > 4 or
                any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in shape) or
                not 0 < math.prod(shape) <= 65536):
            raise ValueError("Requires FP16/FP32, rank 0-4, and 1-65536 elements")
        self.shape, self.dtype = shape, dtype
        self._dtype_name, self._npdtype = bridge._DTYPES[dtype]
        self.nbytes = math.prod(shape) * np.dtype(self._npdtype).itemsize
        self._timeout_ms = timeout_ms
        self._lock = RLock()
        self._closed = False
        self._failed = False
        self._buffers, self._sessions, self._programs = [], {}, {}
        if not SESSION_DLL.is_file():
            raise FileNotFoundError("Run build_npu_session.ps1 to build the session runtime first")
        self._native = npu.NativeLibraries(session_library=SESSION_DLL)
        self._stats = {"session_creations": 0, "buffer_allocations": 0, "live_buffers": 0,
                       "live_buffer_bytes": 0, "peak_buffer_bytes": 0, "npu_calls": 0,
                       "host_upload_copies": 0, "host_upload_bytes": 0,
                       "host_download_copies": 0, "host_download_bytes": 0,
                       "host_pack_copies": 0, "host_pack_bytes": 0,
                       "quarantined_buffer_bytes": 0, "operations": {}}
        self._session("add")
        self._driver_version = self._sessions["add"]._result.driver_version

    def _session(self, kind):
        if kind not in self._sessions:
            # Reuse the existing compiler and precision/ELF validation.
            with bridge._lock:
                program = bridge._program(kind, self._dtype_name, self.shape or (1,))
            parent = self._sessions.get("add")
            session = self._native.create_inference_session(program.graph_blob,
                timeout_ms=self._timeout_ms, parent=parent)
            info = session._result
            if (info.device_vendor_id, info.device_id) != (0x8086, 0x643E):
                session.close()
                raise RuntimeError("Unexpected NPU device")
            self._sessions[kind], self._programs[kind] = session, program
            self._stats["session_creations"] += 1
        return self._sessions[kind]

    def _validate(self, *buffers):
        if self._closed:
            raise RuntimeError("NPU runtime is closed")
        if self._failed:
            raise RuntimeError("NPU execution failed; restart process before accessing its buffers")
        for buffer in buffers:
            if not isinstance(buffer, NpuBuffer) or buffer._runtime is not self:
                raise ValueError("Buffer belongs to another NPU runtime/context")
            if buffer._native._released:
                raise RuntimeError("NPU buffer has been released")

    def empty(self):
        with self._lock:
            self._validate()
            buffer = NpuBuffer(self, self._sessions["add"].create_buffer(self.nbytes))
            self._buffers.append(buffer)
            self._stats["buffer_allocations"] += 1
            self._stats["live_buffers"] += 1
            self._stats["live_buffer_bytes"] += self.nbytes
            self._stats["peak_buffer_bytes"] = max(self._stats["peak_buffer_bytes"], self._stats["live_buffer_bytes"])
            return buffer

    def upload(self, tensor, *, out=None):
        with self._lock:
            self._validate()
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or
                    tensor.layout != torch.strided or tensor.dtype != self.dtype or tuple(tensor.shape) != self.shape):
                raise ValueError("Upload requires a CPU tensor with the runtime's shape and dtype")
            if tensor.requires_grad:
                raise ValueError("Resident runtime does not implement autograd")
            if tensor.is_neg() or tensor.is_conj():
                raise ValueError("Upload requires resolved tensor value bits")
            if out is None:
                out = self.empty()
            self._validate(out)
            values = tensor.contiguous().numpy()
            ctypes.memmove(out._native.address, values.ctypes.data, self.nbytes)
            if not tensor.is_contiguous():
                self._stats["host_pack_copies"] += 1
                self._stats["host_pack_bytes"] += self.nbytes
            self._stats["host_upload_copies"] += 1
            self._stats["host_upload_bytes"] += self.nbytes
            return out

    def download(self, buffer):
        with self._lock:
            self._validate(buffer)
            result = torch.empty(self.shape, dtype=self.dtype)
            ctypes.memmove(result.data_ptr(), buffer._native.address, self.nbytes)
            self._stats["host_download_copies"] += 1
            self._stats["host_download_bytes"] += self.nbytes
            return result

    def run(self, kind, x, y=None, *, out=None):
        with self._lock:
            inputs = (x, y) if kind in _BINARY else (x,)
            if kind not in _BINARY | _UNARY or (kind in _BINARY) != (y is not None):
                raise ValueError("Requires a supported unary or binary operation")
            self._validate(*inputs)
            if out is not None:
                self._validate(out)
                if out in inputs:
                    raise ValueError("Output may not alias an input")
            session = self._session(kind)
            if out is None:
                out = self.empty()
            program = self._programs[kind]
            try:
                session.infer([(t.name, b._native) for t, b in zip(program.graph.inputs, inputs)],
                              [(program.graph.outputs[0].name, out._native)])
            except npu.NativeError as error:
                if error.status == 10:  # NPUNLOCK_STATUS_DRIVER_FAILED
                    self._failed = True
                    if b"zeFenceHostSynchronize" in error.diagnostic or b"zeCommandQueueExecuteCommandLists" in error.diagnostic:
                        self._stats["quarantined_buffer_bytes"] = self._stats["live_buffer_bytes"]
                raise
            self._stats["npu_calls"] += 1
            self._stats["operations"][kind] = self._stats["operations"].get(kind, 0) + 1
            return out

    def stats(self):
        with self._lock:
            return {**self._stats, "operations": dict(self._stats["operations"]), "closed": self._closed,
                    "context_failed": self._failed,
                    "shape": list(self.shape), "dtype": str(self.dtype), "allocation_api": "zeMemAllocShared",
                    "driver_version": self._driver_version,
                    "device": "8086:643e", "synchronization": "fence per graph; context serialized",
                    "queue_policy": "one persistent command queue per context",
                    "command_policy": "one list/fence per graph; re-record bindings and reset after completion",
                    "copy_counter_scope": "explicit runtime upload/download/packing; excludes driver DMA and compiler"}

    def close(self):
        with self._lock:
            if not self._closed:
                for buffer in list(self._buffers):
                    buffer.release()
                for session in reversed(tuple(self._sessions.values())):
                    session.close()
                self._sessions.clear()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

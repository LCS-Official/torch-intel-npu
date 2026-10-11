"""Bounded hardware/lifetime checks for the genuine intel_npu Tensor backend."""
import argparse
import faulthandler
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "npunlock-probe/research/results"
DESTINATION = OUT / "torch-npu-device.json"


def execute(quick):
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    import torch
    import torch_intel_npu as npu
    report = {"status": "running", "checks": [],
              "openvino_installed": importlib.util.find_spec("openvino") is not None,
              "extension_sha256": hashlib.sha256(npu._EXT.read_bytes()).hexdigest(),
              "session_dll_sha256": hashlib.sha256(npu.SESSION_DLL.read_bytes()).hexdigest()}
    started = time.perf_counter()

    def save(stage):
        report["stage"] = stage
        DESTINATION.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)

    def check(name, function):
        save(name)
        before = npu.stats()
        baseline = before["live_buffers"] - before["cached_scalar_buffers"]
        result = function()
        gc.collect()
        after = npu.stats()
        assert after["live_buffers"] - after["cached_scalar_buffers"] == baseline, f"Leaked NPU storage in {name}"
        report["checks"].append({"name": name, "status": "passed", **(result or {})})

    def device_and_copy(dtype):
        torch.manual_seed(123)  # Registering the NPU must not break CPU seeding.
        host = torch.tensor([[1., 2., -3.], [4., 5., 6.]], dtype=dtype)
        x = host.to("intel_npu")
        assert type(x) is torch.Tensor
        assert x.device == torch.device("intel_npu:0") == x.untyped_storage().device
        assert "PrivateUse1" in str(torch._C._dispatch_keys(x))
        torch.testing.assert_close(x.cpu(), host, rtol=0, atol=0)
        y = torch.empty_like(x)
        assert y.device == x.device
        y.copy_(x)
        torch.testing.assert_close(y.cpu(), host, rtol=0, atol=0)
        changed = torch.full_like(host, 2.25)
        x.copy_(changed)
        torch.testing.assert_close(x.cpu(), changed, rtol=0, atol=0)
        assert x.to("intel_npu") is x
        converted = host.intel_npu()
        torch.testing.assert_close(converted.cpu(), host, rtol=0, atol=0)
        created = torch.tensor([1., 2.], dtype=dtype, device="intel_npu")
        torch.testing.assert_close(created.cpu(), torch.tensor([1., 2.], dtype=dtype), rtol=0, atol=0)
        # Packing is explicit host copy work, not CPU arithmetic fallback.
        before = npu.stats()
        strided = torch.arange(12, dtype=dtype).reshape(3, 4).t()
        packed = strided.to("intel_npu", memory_format=torch.contiguous_format)
        torch.testing.assert_close(packed.cpu(), strided, rtol=0, atol=0)
        assert npu.stats()["host_pack_copies"] == before["host_pack_copies"] + 1
        clone = x.clone()
        assert clone.data_ptr() != x.data_ptr()
        torch.testing.assert_close(clone.cpu(), changed, rtol=0, atol=0)
        return {"tensor_type": "torch.Tensor", "tensor_device": str(x.device),
                "storage_device": str(x.untyped_storage().device),
                "dispatch_keys": str(torch._C._dispatch_keys(x))}

    def chain(dtype, shape, iterations, reuse):
        host = torch.linspace(-2, 2, math.prod(shape), dtype=dtype).reshape(shape)
        factor = torch.full(shape, 0.5, dtype=dtype)
        expected = host.clone()
        x, y = host.to("intel_npu"), factor.to("intel_npu")
        temp, spare = torch.empty_like(x), torch.empty_like(x)
        torch.add(x, y, out=temp)
        torch.mul(temp, y, out=spare)
        before = npu.stats()
        pointers = sorted(v.data_ptr() for v in (x, y, temp, spare))
        for _ in range(iterations):
            if reuse:
                torch.add(x, y, out=temp)
                torch.mul(temp, y, out=spare)
                x, spare = spare, x
            else:
                x = (x + y) * y
            expected = (expected + factor) * factor
        after = npu.stats()
        assert after["npu_calls"] - before["npu_calls"] == 2 * iterations
        for key in ("host_upload_copies", "host_download_copies", "host_device_copies"):
            assert before[key] == after[key], f"Host payload copy in arithmetic loop: {key}"
        assert before["session_creations"] == after["session_creations"]
        if reuse:
            assert before["buffer_allocations"] == after["buffer_allocations"]
            assert pointers == sorted(v.data_ptr() for v in (x, y, temp, spare))
        else:
            assert after["live_buffers"] == before["live_buffers"]
        torch.testing.assert_close(x.cpu(), expected, rtol=0, atol=0)
        return {"iterations": iterations, "shape": list(shape), "reused_out": reuse,
                "npu_calls": iterations * 2, "loop_host_payload_copies": 0,
                "loop_allocations": after["buffer_allocations"] - before["buffer_allocations"]}

    def arithmetic(dtype):
        vals = [0., -0., 1., -1., 2.25, -4.75, float('inf'), -float('inf'), float('nan')]
        if dtype == torch.float32:
            vals += [70000., -70000., 1.0001, 2.**-149, -(2.**-149), 2.**-126]
        else:
            vals += [65504., -65504., 2.**-24, -(2.**-24)]
        a = torch.tensor(vals, dtype=dtype)
        b = torch.full_like(a, .5)
        x, y = a.to('intel_npu'), b.to('intel_npu')
        before = npu.stats()
        functions = [torch.add, torch.sub, torch.mul, torch.div, torch.neg, torch.abs, torch.sign]
        for f in functions:
            args = (x, y) if f in functions[:4] else (x,)
            cpu_args = (a, b) if len(args) == 2 else (a,)
            result, expected = f(*args).cpu(), f(*cpu_args)
            torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)
            zeros = expected == 0
            assert torch.equal(torch.signbit(result[zeros]), torch.signbit(expected[zeros]))
        assert npu.stats()['npu_calls'] - before['npu_calls'] == 7
        return {"operations": [f.__name__ for f in functions], "elements": len(vals), "npu_calls": 7}

    def views():
        base = torch.arange(6, dtype=torch.float32).to("intel_npu")
        pointer = base.data_ptr()
        view = base.view(2, 3)
        assert view.data_ptr() == pointer
        assert view.untyped_storage()._cdata == base.untyped_storage()._cdata
        live = npu.stats()['live_buffers']
        del base
        gc.collect()
        assert npu.stats()['live_buffers'] == live
        torch.testing.assert_close(view.cpu(), torch.arange(6, dtype=torch.float32).view(2, 3))
        offset = torch.as_strided(view, (3,), (1,), 1)
        try:
            offset.cpu()
        except (ValueError, RuntimeError) as error:
            assert "offset-zero" in str(error)
        else:
            raise AssertionError("Offset view accepted unsupported native binding")
        return {"same_storage": True, "base_deleted_view_valid": True, "offset_binding_rejected": True}

    def cast_precision():
        import numpy as np
        # Every binary16 bit pattern, including signed zeros, subnormals and NaNs.
        host = torch.from_numpy(np.arange(65536, dtype=np.uint16).view(np.float16))
        x = host.to('intel_npu')
        before = npu.stats()
        widened = x.float()
        narrowed = widened.half()
        after = npu.stats()
        for key in ('host_upload_copies', 'host_download_copies', 'host_device_copies'):
            assert after[key] == before[key]
        assert after['npu_calls'] - before['npu_calls'] == 3
        torch.testing.assert_close(widened.cpu(), host.float(), rtol=0, atol=0, equal_nan=True)
        actual = narrowed.cpu()
        finite = ~torch.isnan(host)
        assert torch.equal(actual.view(torch.int16)[finite], host.view(torch.int16)[finite])
        rng = np.random.default_rng(123)
        bits = rng.integers(0, 2**32, 8192, dtype=np.uint32)
        floats = torch.from_numpy(bits.view(np.float32))
        result = floats.to('intel_npu').half().cpu()
        expected = floats.half()
        torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)
        valid = ~torch.isnan(expected)
        assert torch.equal(result.view(torch.int16)[valid], expected.view(torch.int16)[valid])
        # Same-dtype to(copy=True), device/CPU copy with dtype conversion, scalar cast.
        for shape in ((), (1,), (2, 3, 7)):
            original = torch.linspace(-.5, .5, math.prod(shape), dtype=torch.float32).reshape(shape)
            source = original.to('intel_npu')
            clone = source.to(dtype=torch.float32, copy=True)
            assert clone.data_ptr() != source.data_ptr()
            destination = torch.empty(shape, device='intel_npu', dtype=torch.float16)
            destination.copy_(source)
            torch.testing.assert_close(destination.cpu(), original.half(), rtol=0, atol=0)
            destination.copy_(original)
            torch.testing.assert_close(destination.float().cpu(), original.half().float(), rtol=0, atol=0)
            cpu_half = torch.empty(shape, dtype=torch.float16)
            cpu_half.copy_(source)
            torch.testing.assert_close(cpu_half, original.half(), rtol=0, atol=0)
            torch.testing.assert_close(source.to('cpu', dtype=torch.float16), original.half(), rtol=0, atol=0)
            torch.testing.assert_close(original.to('intel_npu', dtype=torch.float16).cpu(), original.half(), rtol=0, atol=0)
        return {'half_bit_patterns': 65536, 'random_fp32_patterns': 8192,
                'resident_roundtrip_npu_calls': 3, 'resident_roundtrip_host_copies': 0}

    def reductions():
        comparisons = 0
        cases = [((), (None, 0, -1, ())), ((1,), (None, 0)),
                 ((3, 5), (None, 0, 1, -1, (0, 1))),
                 ((2, 3, 7), (None, 0, 1, 2, (0, 2))),
                 ((2, 3, 4, 5), (None, 0, 1, 2, 3, (1, 3))),
                 ((65536,), (None,)), ((256, 256), (0, 1, None))]
        for dtype in (torch.float16, torch.float32):
            for shape, axes in cases:
                # Dyadic inputs make exact accumulation possible, so a layout
                # mistake cannot hide behind broad numerical tolerances.
                host = ((torch.arange(math.prod(shape)) % 19).float() * .125 - 1.).to(dtype).reshape(shape)
                x = host.to('intel_npu')
                for dim in axes:
                    for keepdim in (False, True):
                        for kind in ('sum', 'mean'):
                            result = getattr(x, kind)(dim=dim, keepdim=keepdim)
                            expected = getattr(host, kind)(dim=dim, keepdim=keepdim)
                            actual = result.cpu()
                            torch.testing.assert_close(actual, expected,
                                rtol=1e-3 if dtype == torch.float16 else 1e-6, atol=1e-7)
                            # CPU FP16 parallel sum can round partial results in
                            # FP16. Our contract is FP32 accumulation then cast.
                            reference = getattr(host.float(), kind)(dim=dim, keepdim=keepdim).to(dtype)
                            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
                            comparisons += 1
                if len(shape) > 1:
                    for kind in ('sum', 'mean'):
                        expected = getattr(host, kind)(dim=1)
                        out = torch.empty(expected.shape, dtype=dtype, device='intel_npu')
                        result = getattr(torch, kind)(x, dim=1, out=out)
                        assert result is out
                        torch.testing.assert_close(out.cpu(), expected, rtol=1e-6, atol=1e-7)
                        comparisons += 1
        return {'shape_axis_dtype_comparisons': comparisons, 'max_elements': 65536}

    def reduction_edges():
        comparisons = 0
        values = ([2.**-149] * 17, [-2.**-149] * 17, [-0.] * 17,
                  [70000., 70000.], [2049., -2048.], [float('nan'), 1.],
                  [float('inf'), 1.], [float('inf'), -float('inf')], [2.**-24] * 17)
        for dtype in (torch.float32, torch.float16):
            for vals in values:
                host = torch.tensor(vals, dtype=dtype)
                x = host.to('intel_npu')
                for output_dtype in (torch.float32, torch.float16):
                    for kind in ('sum', 'mean'):
                        result = getattr(x, kind)(dtype=output_dtype).cpu()
                        expected = getattr(host, kind)(dtype=output_dtype)
                        torch.testing.assert_close(result, expected, rtol=1e-6, atol=0, equal_nan=True)
                        if expected == 0:
                            assert torch.signbit(result) == torch.signbit(expected)
                        comparisons += 1
        # Distinguish sum's input rounding from mean's FP32 accumulation.
        host = torch.tensor([2049., -2048.])
        x = host.to('intel_npu')
        for kind in ('sum', 'mean'):
            out = torch.empty((), dtype=torch.float16, device='intel_npu')
            getattr(torch, kind)(x, dim=(), out=out)
            before = npu.stats()
            for _ in range(10):
                getattr(torch, kind)(x, dim=(), out=out)
            after = npu.stats()
            for key in ('host_upload_copies', 'host_download_copies', 'host_device_copies'):
                assert after[key] == before[key], key
            cpu_out = torch.empty((), dtype=torch.float16)
            getattr(torch, kind)(host, dim=(), out=cpu_out)
            torch.testing.assert_close(out.cpu(), cpu_out, rtol=0, atol=0)
        return {'special_value_dtype_comparisons': comparisons,
                'fp32_subnormals_preserved': True, 'mean_fp32_input_not_rounded_to_fp16': True}

    def resident_cast_reduce_chain():
        host = (torch.arange(51).float() * .125).reshape(3, 17)
        factor = torch.full_like(host, .5)
        x, y = host.to('intel_npu'), factor.to('intel_npu')
        # Warm the exact graph shapes once; then measure the reusable sessions.
        z = (x.half().float() * y).sum(dim=1).half().float().mean()
        before = npu.stats()
        for _ in range(10 if quick else 100):
            z = (x.half().float() * y).sum(dim=1).half().float().mean()
        after = npu.stats()
        for key in ('host_upload_copies', 'host_download_copies', 'host_device_copies', 'session_creations'):
            assert before[key] == after[key], key
        expected = (host.half().float() * factor).sum(dim=1).half().float().mean()
        torch.testing.assert_close(z.cpu(), expected, rtol=1e-6, atol=1e-7)
        return {'iterations': 10 if quick else 100, 'npu_calls': after['npu_calls'] - before['npu_calls'],
                'loop_host_copies': 0, 'loop_graph_compilations': 0}

    def broadcast_bits():
        import numpy as np
        comparisons = 0
        arrays = [np.arange(i, i + 32768, dtype=np.uint16).view(np.float16).reshape(1, -1)
                  for i in (0, 32768)]
        random_bits = np.random.default_rng(123).integers(0, 2**32, 8192, dtype=np.uint32)
        arrays.append(random_bits.view(np.float32).reshape(1, -1))
        for array in arrays:
            host = torch.from_numpy(array)
            x = host.to('intel_npu')
            shape = (65536 // host.numel(), host.numel())
            before = npu.stats()
            result = npu._broadcast(x, shape)
            after = npu.stats()
            for key in ('host_upload_copies', 'host_download_copies', 'host_device_copies'):
                assert before[key] == after[key]
            integer = torch.int16 if x.dtype == torch.float16 else torch.int32
            assert torch.equal(result.cpu().view(integer), host.view(integer).expand(shape))
            comparisons += 1
        host = torch.tensor([2.**-149, -0., float('nan'), float('inf'), 70000.]).reshape(1, 1, 1, 5)
        x = host.to('intel_npu')
        result = npu._broadcast(x, (2, 3, 4, 5)).cpu()
        assert torch.equal(result.view(torch.int32), host.view(torch.int32).expand(2, 3, 4, 5))
        return {'half_bit_patterns_preserved': 65536, 'random_fp32_patterns_preserved': 8192,
                'max_output_elements': 65536, 'broadcast_host_payload_copies': 0}

    def scalar_and_promotion():
        comparisons = 0
        for dtype in (torch.float16, torch.float32):
            host = torch.tensor([-65504., -1., -0., 0., .25, 1.000244140625, 2.**-24, float('inf')], dtype=dtype)
            x = host.to('intel_npu')
            for scalar in (1., 1.0001, 70000., -2):
                for function in (lambda z: z + scalar, lambda z: scalar - z, lambda z: z * scalar,
                                 lambda z: z / scalar, lambda z: scalar / z):
                    torch.testing.assert_close(function(x).cpu(), function(host),
                        rtol=1e-3 if dtype == torch.float16 else 3e-7, atol=0, equal_nan=True)
                    comparisons += 1
            torch.testing.assert_close(torch.div(x, 2., rounding_mode=None).cpu(), torch.div(host, 2., rounding_mode=None),
                                       rtol=1e-3 if dtype == torch.float16 else 3e-7, atol=0, equal_nan=True)
        half = torch.tensor([0., .25], dtype=torch.float16)
        scalar = torch.tensor(70000.)
        a, b = half.to('intel_npu'), scalar.to('intel_npu')
        for function in (torch.mul, torch.div):
            for left, right, cpu_left, cpu_right in ((a, b, half, scalar), (b, a, scalar, half)):
                torch.testing.assert_close(function(left, right).cpu(), function(cpu_left, cpu_right),
                                           rtol=1e-3, atol=0, equal_nan=True)
                comparisons += 1
        shapes = (((3, 5), (5,)), ((2, 1, 4), (1, 3, 1)), ((), (2, 3)), ((2, 3), ()), ((), ()))
        for left_dtype in (torch.float16, torch.float32):
            for right_dtype in (torch.float16, torch.float32):
                for a_shape, b_shape in shapes:
                    a = (torch.arange(math.prod(a_shape)).float() * .125 - .25).to(left_dtype).reshape(a_shape)
                    b = (torch.arange(math.prod(b_shape)).float() * .125 + .5).to(right_dtype).reshape(b_shape)
                    x, y = a.to('intel_npu'), b.to('intel_npu')
                    for function in (torch.add, torch.sub, torch.mul, torch.div):
                        actual, expected = function(x, y), function(a, b)
                        assert actual.dtype == expected.dtype
                        torch.testing.assert_close(actual.cpu(), expected,
                            rtol=1e-3 if actual.dtype == torch.float16 else 3e-7, atol=0, equal_nan=True)
                        out = torch.empty(expected.shape, dtype=expected.dtype, device='intel_npu')
                        assert function(x, y, out=out) is out
                        torch.testing.assert_close(out.cpu(), expected,
                            rtol=1e-3 if actual.dtype == torch.float16 else 3e-7, atol=0, equal_nan=True)
                        comparisons += 2
        return {'scalar_broadcast_promotion_comparisons': comparisons}

    def matrices():
        comparisons = 0
        shapes = ((1, 1, 1), (3, 5, 4), (2, 7, 3), (1, 65536, 1), (16, 16, 16), (32, 32, 64))
        for dtype in (torch.float16, torch.float32):
            for m, k, n in shapes:
                a = ((torch.arange(m*k) % 17).float() * .125 - 1).to(dtype).reshape(m, k)
                b = ((torch.arange(k*n) % 13).float() * .125 - .5).to(dtype).reshape(k, n)
                x, y = a.to('intel_npu'), b.to('intel_npu')
                expected = a.float().matmul(b.float()).to(dtype)
                for function in (torch.mm, torch.matmul):
                    result = function(x, y)
                    torch.testing.assert_close(result.cpu(), expected, rtol=1e-6, atol=1e-7)
                    out = torch.empty((m, n), dtype=dtype, device='intel_npu')
                    assert function(x, y, out=out) is out
                    torch.testing.assert_close(out.cpu(), expected, rtol=1e-6, atol=1e-7)
                    comparisons += 2
                # Reusing the same graph must observe a newly uploaded input.
                x.copy_(a * .5)
                torch.testing.assert_close((x @ y).cpu(), ((a*.5).float() @ b.float()).to(dtype), rtol=1e-6, atol=1e-7)
                comparisons += 1
        generator = torch.Generator().manual_seed(77)
        for dtype in (torch.float16, torch.float32):
            a, b = [torch.randn(shape, generator=generator).to(dtype) for shape in ((3, 17), (17, 7))]
            actual = (a.to('intel_npu') @ b.to('intel_npu')).cpu()
            torch.testing.assert_close(actual, a @ b, rtol=2e-3 if dtype == torch.float16 else 3e-5,
                                       atol=2e-3 if dtype == torch.float16 else 2e-5)
            comparisons += 1
        return {'matmul_comparisons': comparisons, 'max_expanded_products': 65536, 'arithmetic': 'SHAVE FP32'}

    def matrix_edges():
        cases = [(torch.tensor([[70000., 1.000244140625, 2.**-149]]), torch.eye(3)),
                 (torch.tensor([[300., 300.]], dtype=torch.float16), torch.tensor([[300.], [-300.]], dtype=torch.float16)),
                 (torch.full((1, 17), 2.**-24, dtype=torch.float16), torch.ones(17, 1, dtype=torch.float16)),
                 (torch.tensor([[float('inf'), 1.], [float('nan'), -0.]]), torch.tensor([[0., 1.], [1., 2.]]))]
        for a, b in cases:
            expected = a.float().matmul(b.float()).to(a.dtype)
            actual = (a.to('intel_npu') @ b.to('intel_npu')).cpu()
            torch.testing.assert_close(actual, expected, rtol=3e-7, atol=0, equal_nan=True)
        return {'cases': len(cases), 'fp32_range_and_subnormals': True, 'fp16_products_accumulate_in_fp32': True}

    def resident_matrix_chain():
        calls = 0
        for dtype in (torch.float16, torch.float32):
            a = (torch.arange(15).reshape(3, 5).float() * .125).to(dtype)
            b = (torch.arange(20).reshape(5, 4).float() * .125).to(dtype)
            bias = torch.arange(4, dtype=dtype)
            x, w, y = [t.to('intel_npu') for t in (a, b, bias)]
            out = torch.empty(3, 4, dtype=dtype, device='intel_npu')
            tmp, scaled = torch.empty_like(out), torch.empty_like(out)
            def step():
                torch.mm(x, w, out=out)
                torch.add(out, y, out=tmp)
                torch.mul(tmp, .5, out=scaled)
                return scaled.mean()
            result = step()
            before = npu.stats()
            for _ in range(10 if quick else 100):
                result = step()
            after = npu.stats()
            for key in ('host_upload_copies', 'host_download_copies', 'host_device_copies', 'session_creations'):
                assert before[key] == after[key], key
            expected = ((a @ b + bias) * .5).mean()
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-3 if dtype == torch.float16 else 1e-6, atol=1e-6)
            calls += after['npu_calls'] - before['npu_calls']
        return {'rounds_per_dtype': 10 if quick else 100, 'npu_calls': calls,
                'loop_host_payload_copies': 0, 'loop_graph_compilations': 0}

    def scalar_cache_lifetime():
        x = torch.tensor(.25).to('intel_npu')
        for i in range(160):
            result = x + float(1000 + i)
        assert npu.stats()['cached_scalar_buffers'] == 128
        before = npu.stats()
        for _ in range(20):
            result = x + 1159.
        after = npu.stats()
        assert before['host_upload_copies'] == after['host_upload_copies']
        torch.testing.assert_close(result.cpu(), torch.tensor(1159.25), rtol=0, atol=0)
        return {'cached_scalar_limit': 128, 'evictions_exercised': True, 'repeated_constant_uploads': 0}

    def transpose_bits():
        import numpy as np
        comparisons = 0
        for dtype, integer in ((torch.float16, np.uint16), (torch.float32, np.uint32)):
            bits = (np.arange(65536, dtype=integer) if dtype == torch.float16 else
                    np.random.default_rng(92).integers(0, 2**32, 65536, dtype=integer))
            if dtype == torch.float32:
                bits[:4] = [1, 0x7f800000, 0x7fc00001, 0x80000000]
            host = torch.from_numpy(bits.view(np.float16 if dtype == torch.float16 else np.float32)).view(256, 256)
            x = host.to('intel_npu')
            before = npu.stats()
            view = x.t()
            assert view.untyped_storage()._cdata == x.untyped_storage()._cdata
            assert npu.stats()['npu_calls'] == before['npu_calls']  # Metadata only.
            packed = view.contiguous()
            after = npu.stats()
            assert after['npu_calls'] == before['npu_calls'] + 1
            for key in ('host_upload_copies', 'host_download_copies', 'host_device_copies'):
                assert before[key] == after[key], key
            assert packed.is_contiguous() and packed.data_ptr() != x.data_ptr()
            assert np.array_equal(packed.cpu().numpy().view(integer), bits.reshape(256, 256).T)
            comparisons += 1
            for shape, order in (((3, 5), (1, 0)), ((2, 3, 5), (2, 0, 1)),
                                 ((2, 3, 4, 5), (3, 1, 0, 2)), ((2, 1, 3, 5), (3, 1, 0, 2))):
                cpu = host.flatten()[:math.prod(shape)].reshape(shape).clone()
                device = cpu.to('intel_npu')
                permutation = device.permute(order)
                expected = cpu.permute(order).numpy().view(integer)
                assert np.array_equal(permutation.contiguous().cpu().numpy().view(integer), expected)
                assert np.array_equal(permutation.cpu().numpy().view(integer), expected)
                copied = torch.empty_like(cpu.permute(order).contiguous()).to('intel_npu')
                copied.copy_(permutation)
                assert np.array_equal(copied.cpu().numpy().view(integer), expected)
                comparisons += 3
            base = torch.arange(15, dtype=dtype).reshape(3, 5).to('intel_npu')
            transpose = base.transpose(-1, -2)
            detached = transpose.detach()
            assert detached.untyped_storage()._cdata == base.untyped_storage()._cdata
            replacement = torch.full((3, 5), 2.25, dtype=dtype)
            base.copy_(replacement)
            del base
            gc.collect()
            torch.testing.assert_close(detached.cpu(), replacement.t(), rtol=0, atol=0)
            comparisons += 1
        return {'comparisons': comparisons, 'half_bit_patterns': 65536, 'fp32_bit_patterns': 65536,
                'transpose_is_storage_alias': True, 'materialization': 'DMA only', 'resident_host_copies': 0}

    def transposed_matrices():
        comparisons = 0
        for dtype in (torch.float16, torch.float32):
            for m, k, n in ((3, 5, 4), (1, 256, 256), (32, 32, 64)):
                a = (torch.arange(k * m, dtype=torch.float32).reshape(k, m) % 17 - 8).to(dtype) / 8
                b = (torch.arange(n * k, dtype=torch.float32).reshape(n, k) % 13 - 6).to(dtype) / 8
                left, right = a.to('intel_npu').t(), b.to('intel_npu').t()
                expected = (a.t().float() @ b.t().float()).to(dtype)
                out = torch.empty((m, n), dtype=dtype, device='intel_npu')
                result = left @ right
                torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
                assert torch.mm(left, right, out=out) is out
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                comparisons += 2
        return {'comparisons': comparisons, 'largest_MKN': 65536}

    def addmm_precision():
        comparisons = 0
        for dtype in (torch.float16, torch.float32):
            a = torch.tensor([[.25, -.5, 1.], [2., .5, -.25]], dtype=dtype)
            b = torch.tensor([[1., 2., -.5, 1.], [-1., .5, 2., -2.], [.25, -1., 1., .5]], dtype=dtype)
            x, y = a.to('intel_npu'), b.to('intel_npu')
            for shape in ((), (4,), (1, 4), (2, 1), (2, 4)):
                bias = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).to(dtype) / 4
                device = bias.to('intel_npu')
                out = torch.empty((2, 4), dtype=dtype, device='intel_npu')
                for beta, alpha in ((1, 1), (0, 1), (2, .5), (-3, -2)):
                    expected = torch.addmm(bias, a, b, beta=beta, alpha=alpha)
                    torch.testing.assert_close(torch.addmm(device, x, y, beta=beta, alpha=alpha).cpu(), expected, rtol=0, atol=0)
                    assert torch.addmm(device, x, y, beta=beta, alpha=alpha, out=out) is out
                    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                    comparisons += 2
            nan_bias = torch.full((4,), float('nan'), dtype=dtype).to('intel_npu')
            torch.testing.assert_close(torch.addmm(nan_bias, x, y, beta=0).cpu(), a @ b, rtol=0, atol=0)
            comparisons += 1
            # No intermediate FP16 overflow/rounding before adding bias.
            large = torch.tensor([[300.]], dtype=dtype)
            bias = torch.tensor([-60000.], dtype=dtype)
            actual = torch.addmm(bias.to('intel_npu'), large.to('intel_npu'), large.to('intel_npu')).cpu()
            torch.testing.assert_close(actual, torch.addmm(bias, large, large), rtol=0, atol=0)
            comparisons += 1
            one = torch.ones((1, 1), dtype=dtype).to('intel_npu')
            torch.testing.assert_close(torch.addmm(one, one, one, alpha=70000., beta=-70000.).cpu(),
                                       torch.zeros((1, 1), dtype=dtype), rtol=0, atol=0)
            comparisons += 1
            # Warm out= uses the final allocation without host memmove.
            torch.addmm(one, one, one, out=torch.empty_like(one))
            out = torch.empty_like(one)
            before = npu.stats()
            for _ in range(5):
                torch.addmm(one, one, one, out=out)
            after = npu.stats()
            for key in ('host_upload_copies', 'host_download_copies', 'host_device_copies', 'session_creations'):
                assert before[key] == after[key], key
        return {'comparisons': comparisons, 'fp16_bias_added_before_output_rounding': True, 'beta_zero_ignores_nan_bias': True}

    def linear_forward():
        import copy
        comparisons = 0
        for dtype in (torch.float16, torch.float32):
            for has_bias in (False, True):
                cpu = torch.nn.Linear(5, 4, bias=has_bias, dtype=dtype).eval()
                model = copy.deepcopy(cpu).to('intel_npu')
                assert all(p.device.type == 'intel_npu' and p.requires_grad for p in model.parameters())
                for shape in ((5,), (2, 5), (2, 3, 5), (2, 2, 3, 5)):
                    host = torch.linspace(-1, 1, math.prod(shape), dtype=dtype).reshape(shape)
                    data = host.to('intel_npu').requires_grad_()
                    expected = cpu(host).detach()
                    for context in (torch.no_grad, torch.inference_mode):
                        with context():
                            result = model(data)
                            assert result.device.type == 'intel_npu' and not result.requires_grad
                            torch.testing.assert_close(result.cpu(), expected,
                                rtol=2e-3 if dtype == torch.float16 else 2e-6, atol=2e-4 if dtype == torch.float16 else 2e-6)
                            comparisons += 1
                with torch.no_grad():
                    replacement = torch.full_like(cpu.weight, .125)
                    model.weight.copy_(replacement)
                    cpu.weight.copy_(replacement)
                    model.load_state_dict(cpu.state_dict())
                    # Forward observes the current resident weight, not cached values.
                    torch.testing.assert_close(model(data).cpu(), cpu(host), rtol=2e-3, atol=2e-6)
                    model.cpu()
                    assert model.weight.requires_grad and model.weight.device.type == 'cpu'
                    torch.testing.assert_close(model.weight, cpu.weight, rtol=0, atol=0)
                    comparisons += 2
        # Distinguish real FP32 from an implicit half-precision Linear carrier.
        edge = torch.tensor([[70000., 1.000244140625, 2.0**-149]], dtype=torch.float32)
        with torch.inference_mode():
            actual = torch.nn.functional.linear(edge.to('intel_npu'), torch.eye(3).to('intel_npu'),
                                                 torch.zeros(3).to('intel_npu'))
            torch.testing.assert_close(actual.cpu(), edge, rtol=0, atol=0)
            half = torch.tensor([[300.]], dtype=torch.float16)
            actual = torch.nn.functional.linear(half.to('intel_npu'), half.to('intel_npu'),
                                                 torch.tensor([-60000.], dtype=torch.float16).to('intel_npu'))
            torch.testing.assert_close(actual.cpu(), torch.tensor([[30000.]], dtype=torch.float16), rtol=0, atol=0)
        comparisons += 2
        return {'comparisons': comparisons, 'input_ranks': [1, 2, 3, 4], 'bias_and_no_bias': True,
                'normal_module_transfer': True, 'requires_grad_metadata_preserved': True,
                'contexts': ['no_grad', 'inference_mode'], 'weight_updates_and_state_dict': True}

    def linear_chain():
        import copy
        calls = 0
        for dtype in (torch.float16, torch.float32):
            cpu = torch.nn.Sequential(torch.nn.Linear(5, 4, dtype=dtype), torch.nn.Linear(4, 3, dtype=dtype)).eval()
            model = copy.deepcopy(cpu).to('intel_npu')
            host = torch.linspace(-1, 1, 10, dtype=dtype).view(2, 5)
            data = host.to('intel_npu')
            with torch.inference_mode():
                expected = cpu(host)
                model(data)
                before = npu.stats()
                for _ in range(5 if quick else 100):
                    result = model(data)
                after = npu.stats()
                for key in ('host_upload_copies', 'host_download_copies', 'host_device_copies', 'session_creations'):
                    assert before[key] == after[key], key
                torch.testing.assert_close(result.cpu(), expected,
                    rtol=2e-3 if dtype == torch.float16 else 2e-6, atol=2e-4 if dtype == torch.float16 else 2e-6)
            calls += after['npu_calls'] - before['npu_calls']
        return {'rounds_per_dtype': 5 if quick else 100, 'layers': 2, 'npu_calls': calls,
                'loop_host_payload_copies': 0, 'loop_graph_compilations': 0}

    def rejections():
        x = torch.tensor([1., 2.]).to('intel_npu')
        y = torch.tensor([3., 4.]).to('intel_npu')
        half = torch.tensor([3., 4.], dtype=torch.float16).to('intel_npu')
        short = torch.tensor([1.]).to('intel_npu')
        grad = torch.tensor([1., 2.]).to('intel_npu').requires_grad_()
        actions = {
            "unsupported_sin": lambda: torch.sin(x),
            "unsupported_rng": lambda: torch.rand(2, device='intel_npu'),
            "unsupported_batched_matmul": lambda: x.view(1, 1, 2) @ y.view(1, 2, 1),
            "incompatible_broadcast": lambda: x + torch.ones(3).to('intel_npu'),
            "broadcast_oversized": lambda: torch.ones(1024, 1).to('intel_npu') + torch.ones(1, 1025).to('intel_npu'),
            "complex_scalar": lambda: x + 1j,
            "cpu_mixed_input": lambda: x + torch.ones(2),
            "alpha": lambda: torch.add(x, y, alpha=2),
            "boolean_alpha": lambda: torch.add(x, y, alpha=True),
            "boolean_subtraction": lambda: x - True,
            "output_alias": lambda: torch.mul(x, y, out=x),
            "autograd_upload": lambda: torch.ones(2, requires_grad=True).to('intel_npu'),
            "float64": lambda: torch.empty(2, dtype=torch.float64, device='intel_npu'),
            "int8": lambda: torch.empty(2, dtype=torch.int8, device='intel_npu'),
            "empty": lambda: torch.empty(0, device='intel_npu'),
            "oversized": lambda: torch.empty(npu.MAX_TENSOR_ELEMENTS+1, device='intel_npu'),
            "rank5": lambda: torch.empty((1, 1, 1, 1, 1), device='intel_npu'),
            "factory_with_storage_holes": lambda: torch.empty_strided((2, 3), (1, 4), device='intel_npu'),
            "factory_with_overlapping_storage": lambda: torch.empty_strided((2, 3), (1, 1), device='intel_npu'),
            "noncontiguous_upload_without_format": lambda: torch.ones(2, 3).t().to('intel_npu'),
            "device1": lambda: torch.empty(2, device='intel_npu:1'),
            "asynchronous_copy": lambda: x.to('cpu', non_blocking=True),
            "unsupported_bfloat16": lambda: x.bfloat16(),
            "reduction_duplicate_axis": lambda: x.sum(dim=(0, 0)),
            "reduction_bad_axis": lambda: x.mean(dim=1),
            "reduction_int_dtype": lambda: x.sum(dtype=torch.int32),
            "reduction_wrong_out_shape": lambda: torch.sum(x, dim=0, out=y),
            "reduction_dtype_out_mismatch": lambda: torch.sum(x, dim=0, dtype=torch.float32,
                                                               out=torch.empty((), dtype=torch.float16, device='intel_npu')),
            "reduction_output_alias": lambda: torch.mean(short, dim=0, keepdim=True, out=short),
            "matmul_inner_mismatch": lambda: x.view(1, 2) @ y.view(1, 2),
            "matmul_dtype_mismatch": lambda: x.view(1, 2) @ half.view(2, 1),
            "matmul_oversized_output": lambda: torch.ones(1025, 1).to('intel_npu') @ torch.ones(1, 1024).to('intel_npu'),
            "matmul_out_alias": lambda: torch.mm(x.view(1, 2), torch.eye(2).to('intel_npu'), out=x.view(1, 2)),
            "division_rounding": lambda: torch.div(x, y, rounding_mode='floor'),
            "addmm_cpu_bias": lambda: torch.addmm(torch.ones(2), x.view(1, 2), torch.eye(2).to('intel_npu')),
            "addmm_bias_dtype": lambda: torch.addmm(half, x.view(1, 2), torch.eye(2).to('intel_npu')),
            "addmm_bias_shape": lambda: torch.addmm(y.view(2, 1), x.view(1, 2), torch.eye(2).to('intel_npu')),
            "addmm_out_alias": lambda: torch.addmm(y, x.view(1, 2), torch.eye(2).to('intel_npu'), out=y.view(1, 2)),
            "addmm_coefficient_overflow": lambda: torch.addmm(y, x.view(1, 2), torch.eye(2).to('intel_npu'), alpha=1e50),
            "linear_bias_shape": lambda: torch.nn.functional.linear(x, y.view(1, 2), y),
            "linear_bad_weight": lambda: torch.nn.functional.linear(x, y),
            "dense_partial_storage": lambda: x.as_strided((1,), (1,), 0).cpu(),
            "dense_expanded_storage": lambda: short.as_strided((2,), (0,), 0).cpu(),
        }
        before = npu.stats()['npu_calls']
        rejected = []
        for name, action in actions.items():
            try:
                action()
            except (ValueError, IndexError, TypeError, RuntimeError, NotImplementedError):
                rejected.append(name)
            else:
                raise AssertionError(f"Accepted {name}")
        assert before == npu.stats()['npu_calls']
        return {"rejected": rejected, "npu_calls": 0}

    def threads():
        from concurrent.futures import ThreadPoolExecutor
        baseline = npu.stats()['npu_calls']
        def worker(i):
            a = torch.full((17,), float(i)).to('intel_npu')
            b = torch.full((17,), .5).to('intel_npu')
            for _ in range(5 if quick else 50):
                z = (a + b) * b
            torch.testing.assert_close(z.cpu(), torch.full((17,), (i + .5) * .5))
            reduced = z.half().float().sum().mean()
            torch.testing.assert_close(reduced.cpu(), torch.tensor(17 * (i + .5) * .5))
            matrix = (a.view(1, 17) @ b.view(17, 1)) + 1.
            torch.testing.assert_close(matrix.mean().cpu(), torch.tensor(17 * i * .5 + 1.))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, range(4)))
        calls = npu.stats()['npu_calls'] - baseline
        assert calls == 4 * ((5 if quick else 50) * 2 + 9)
        return {"threads": 4, "npu_calls": calls, "submission": "serialized"}

    def closed_lifetime():
        x = torch.tensor([3., 4.]).to('intel_npu')
        view = x.view(1, 2)
        del x
        baseline = npu.stats()['live_buffers']
        constants = npu.stats()['cached_scalar_buffers']
        npu.shutdown()
        assert baseline == 1 + constants
        assert npu.stats()['live_buffers'] == 1
        for action in (lambda: view.cpu(), lambda: torch.empty(2, device='intel_npu'), lambda: view + view):
            try:
                action()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Shutdown backend permitted buffer access")
        del view
        gc.collect()
        assert npu.stats()['live_buffers'] == npu.stats()['live_buffer_bytes'] == 0
        npu.shutdown()
        return {"live_storage_retained_after_shutdown": True, "last_view_release_frees_storage": True}

    try:
        for dtype in (torch.float16, torch.float32):
            check(f"{dtype}_device_and_copy", lambda d=dtype: device_and_copy(d))
            check(f"{dtype}_arithmetic", lambda d=dtype: arithmetic(d))
            for reuse in (False, True):
                check(f"{dtype}_chain_reuse_{reuse}", lambda d=dtype, r=reuse:
                      chain(d, (1, 17), 10 if quick else 1000, r))
            for shape in ((), (2, 3, 7), (1, 65536)):
                check(f"{dtype}_shape_{shape}", lambda d=dtype, s=shape: chain(d, s, 2 if quick else 10, True))
        check("view_storage_lifetime", views)
        check("resident_cast_precision", cast_precision)
        check("resident_reduction_axes", reductions)
        check("resident_reduction_special_values", reduction_edges)
        check("resident_cast_reduce_chain", resident_cast_reduce_chain)
        check("resident_broadcast_bit_patterns", broadcast_bits)
        check("resident_scalar_broadcast_promotion", scalar_and_promotion)
        check("resident_matrices", matrices)
        check("resident_matrix_special_values", matrix_edges)
        check("resident_matrix_chain", resident_matrix_chain)
        check("transpose_bit_patterns_and_lifetime", transpose_bits)
        check("resident_transposed_matrices", transposed_matrices)
        check("resident_addmm_precision", addmm_precision)
        check("linear_module_forward", linear_forward)
        check("resident_two_linear_chain", linear_chain)
        check("scalar_cache_lifetime", scalar_cache_lifetime)
        check("unsupported_contracts", rejections)
        check("threaded_tensor_lifetime", threads)
        assert npu.stats()['live_buffers'] == npu.stats()['cached_scalar_buffers']
        report['stats_before_shutdown'] = npu.stats()
        check("shutdown_storage_lifetime", closed_lifetime)
        report.update(status="passed", seconds=time.perf_counter() - started, final_stats=npu.stats())
        faulthandler.cancel_dump_traceback_later()
        save("complete")
        return 0
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc(), final_stats=npu.stats())
        save(report['stage'])
        print(report['traceback'], flush=True)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.child:
        raise SystemExit(execute(args.quick))
    command = [sys.executable, __file__, '--child'] + (['--quick'] if args.quick else [])
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300, env=os.environ.copy())
        (OUT / 'torch-npu-device.log').write_text(result.stdout + result.stderr)
        print(result.stdout + result.stderr, flush=True)
        raise SystemExit(result.returncode)
    except subprocess.TimeoutExpired:
        report = json.loads(DESTINATION.read_text()) if DESTINATION.exists() else {}
        report.update(status='timeout', error='Outer process deadline: 300 seconds')
        DESTINATION.write_text(json.dumps(report, indent=2))
        raise SystemExit(1)

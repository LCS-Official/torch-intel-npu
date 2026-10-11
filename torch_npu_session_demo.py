"""Bounded hardware checks for resident buffers and reusable NPU graph sessions."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "npunlock-probe/research/results"
DESTINATION = OUT / "torch-npu-session.json"


def execute(quick):
    import torch
    import torch_npu_minimal as bridge
    from torch_npu_runtime import NpuRuntime, SESSION_DLL

    report = {"status": "running", "checks": [], "stage": "start",
              "openvino_installed": importlib.util.find_spec("openvino") is not None,
              "session_dll_sha256": hashlib.sha256(SESSION_DLL.read_bytes()).hexdigest(),
              "timing_scope": "synchronous host submission + NPU execution + fences; not pure kernel time"}
    started = time.perf_counter()

    def save(stage):
        report["stage"] = stage
        DESTINATION.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)

    def check(name, function):
        save(name)
        result = function()
        report["checks"].append({"name": name, "status": "passed", **(result or {})})

    def chain(dtype, shape, iterations):
        x = torch.linspace(-2, 2, torch.tensor(shape).prod().item() if shape else 1, dtype=dtype).reshape(shape)
        y = torch.full(shape, 0.5, dtype=dtype)
        expected = x.clone()
        for _ in range(iterations):
            expected = (expected + y) * y
        with NpuRuntime(shape, dtype) as runtime:
            a, b = runtime.upload(x), runtime.upload(y)
            intermediate, output = runtime.empty(), runtime.empty()
            runtime.run("add", a, b, out=intermediate)
            runtime.run("mul", intermediate, b, out=output)
            addresses = [v.address for v in (a, b, intermediate, output)]
            before = runtime.stats()
            samples = []
            for _ in range(iterations):
                start = time.perf_counter()
                runtime.run("add", a, b, out=intermediate)
                runtime.run("mul", intermediate, b, out=output)
                a, output = output, a
                samples.append((time.perf_counter() - start) * 1000)
            after = runtime.stats()
            assert before["host_upload_copies"] == after["host_upload_copies"] == 2
            assert before["host_download_copies"] == after["host_download_copies"] == 0
            assert before["buffer_allocations"] == after["buffer_allocations"] == 4
            assert before["session_creations"] == after["session_creations"] == 2
            assert after["npu_calls"] - before["npu_calls"] == iterations * 2
            assert sorted(addresses) == sorted(v.address for v in (a, b, intermediate, output))
            torch.testing.assert_close(runtime.download(a), expected, rtol=0, atol=0, equal_nan=True)
            # Update both existing input allocations; no stale pointer/data cache.
            changed_x, changed_y = torch.full(shape, -3., dtype=dtype), torch.full(shape, 0.25, dtype=dtype)
            runtime.upload(changed_x, out=a)
            runtime.upload(changed_y, out=b)
            runtime.run("add", a, b, out=intermediate)
            runtime.run("mul", intermediate, b, out=output)
            torch.testing.assert_close(runtime.download(output), (changed_x + changed_y) * changed_y, rtol=0, atol=0)
            stats = runtime.stats()
        closed = runtime.stats()
        assert closed["live_buffers"] == closed["live_buffer_bytes"] == 0
        return {"iterations": iterations, "resident_calls": iterations * 2,
                "loop_host_uploads": 0, "loop_host_downloads": 0,
                "loop_allocations": 0, "median_chain_ms": statistics.median(samples),
                "p95_chain_ms": sorted(samples)[int((len(samples) - 1) * .95)],
                "stats": stats, "closed_live_buffer_bytes": closed["live_buffer_bytes"]}

    def arithmetic(dtype):
        x = torch.tensor([0., -0., 1., -1., .5, -2., 65504., -65504., 2**-24, -2**-24,
                          float("inf"), -float("inf"), float("nan"), 0., -0., 3., .25], dtype=dtype)
        y = torch.tensor([0., -0., 2., 3., .5, -2., 1., -1., .5, 2., 0., -1., 1., 1., -1., 0., 4.], dtype=dtype)
        if dtype == torch.float32:
            x[6:10] = torch.tensor([70000., 1.000244140625, 2**-140, -2**-149])
        with NpuRuntime(x.shape, dtype) as runtime:
            a, b, out = runtime.upload(x), runtime.upload(y), runtime.empty()
            for kind in ("add", "sub", "mul", "div", "add1", "neg", "abs", "sign"):
                runtime.run(kind, a, b if kind in {"add", "sub", "mul", "div"} else None, out=out)
                expected = x + 1 if kind == "add1" else getattr(torch, kind)(x, y) if kind in {"add", "sub", "mul", "div"} else getattr(torch, kind)(x)
                actual = runtime.download(out)
                torch.testing.assert_close(actual, expected, rtol=3e-7 if kind == "div" else 0, atol=0, equal_nan=True, msg=kind)
                zeros = expected == 0
                assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))
            return {"stats": runtime.stats()}

    def native_lifetime():
        import npunlock as npu
        from npunlock._native import NativeError
        native = npu.NativeLibraries(session_library=SESSION_DLL)
        p = bridge._program("add", "f32", (1, 17))
        q = bridge._program("mul", "f32", (1, 17))
        values = torch.full((17,), 2.)
        copied = native.infer_graph(p.graph_blob,
            [npu.InferenceInput(t.name, values.numpy().tobytes()) for t in p.graph.inputs], timeout_ms=10_000)
        assert (copied.vendor_id, copied.device_id) == (0x8086, 0x643E)
        copied_tensor = torch.frombuffer(bytearray(copied.outputs[0].data), dtype=torch.float32)
        torch.testing.assert_close(copied_tensor, torch.full((17,), 4.), rtol=0, atol=0)
        parent = native.create_inference_session(p.graph_blob)
        child = native.create_inference_session(q.graph_blob, parent=parent)
        buffers = [parent.create_buffer(17 * 4) for _ in range(3)]
        for buffer in buffers[:2]:
            ctypes.memmove(buffer.address, values.data_ptr(), 17 * 4)
        parent.close()
        parent.close()
        # Parent's graph is gone, but the child's context and old allocations live.
        child.infer([(t.name, b) for t, b in zip(q.graph.inputs, buffers)], [(q.graph.outputs[0].name, buffers[2])])
        result = torch.empty(17)
        ctypes.memmove(result.data_ptr(), buffers[2].address, 17 * 4)
        torch.testing.assert_close(result, torch.full((17,), 4.), rtol=0, atol=0)
        # Check native byte-size/alias guards, independently of NpuRuntime.
        small = child.create_buffer(4)
        for invalid_inputs, invalid_output, message in (
                ((small, buffers[1]), buffers[2], "byte size"),
                ((buffers[0], buffers[1]), buffers[0], "aliases")):
            try:
                child.infer([(t.name, b) for t, b in zip(q.graph.inputs, invalid_inputs)],
                            [(q.graph.outputs[0].name, invalid_output)])
            except NativeError as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError("Native binding guard accepted invalid buffers")
        small.release()
        grandchild = native.create_inference_session(q.graph_blob, parent=child)
        child.close()
        grandchild.infer([(t.name, b) for t, b in zip(q.graph.inputs, buffers)],
                        [(q.graph.outputs[0].name, buffers[2])])
        grandchild.close()
        # Buffer memory remains host readable after all graphs have closed.
        ctypes.memmove(result.data_ptr(), buffers[2].address, 17 * 4)
        torch.testing.assert_close(result, torch.full((17,), 4.), rtol=0, atol=0)
        for buffer in buffers:
            buffer.release()
            buffer.release()

    def concurrency():
        from concurrent.futures import ThreadPoolExecutor
        with NpuRuntime((1, 17)) as runtime:
            bundles = []
            for index in range(4):
                x = torch.full((1, 17), index + 1.)
                y = torch.full((1, 17), .25 * (index + 1))
                bundles.append((runtime.upload(x), runtime.upload(y), runtime.empty(), runtime.empty(), (x + y) * y))
            # Compile/load both graphs before concurrent submissions.
            runtime.run("mul", bundles[0][0], bundles[0][1], out=bundles[0][3])
            before = runtime.stats()
            iterations = 25 if quick else 250
            def worker(bundle):
                a, b, tmp, out, expected = bundle
                for _ in range(iterations):
                    runtime.run("add", a, b, out=tmp)
                    runtime.run("mul", tmp, b, out=out)
                torch.testing.assert_close(runtime.download(out), expected, rtol=0, atol=0)
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(worker, bundles))
            after = runtime.stats()
            assert after["buffer_allocations"] == before["buffer_allocations"] == 16
            assert after["session_creations"] == 2
            assert after["npu_calls"] - before["npu_calls"] == 8 * iterations
            return {"threads": 4, "serialized_npu_calls": 8 * iterations, "stats": after}

    def repeated_close():
        for _ in range(20):
            with NpuRuntime((17,)) as runtime:
                x = torch.ones(17)
                a, b, out = runtime.upload(x), runtime.upload(x), runtime.empty()
                runtime.run("add", a, b, out=out)
                torch.testing.assert_close(runtime.download(out), x + x, rtol=0, atol=0)
            assert runtime.stats()["live_buffer_bytes"] == 0
        # Explicit packing copy is counted for noncontiguous CPU input.
        x = torch.arange(34.).reshape(17, 2)[:, 0]
        with NpuRuntime((17,)) as runtime:
            a = runtime.upload(x)
            torch.testing.assert_close(runtime.download(a), x, rtol=0, atol=0)
            assert runtime.stats()["host_pack_copies"] == 1
        return {"create_run_close_cycles": 20}

    def timing_comparison(dtype):
        x, y = torch.linspace(-1, 1, 17, dtype=dtype).reshape(1, 17), torch.full((1, 17), .5, dtype=dtype)
        expected = (x + y) * y
        samples = {"copied": [], "resident": []}
        with NpuRuntime(x.shape, dtype) as runtime:
            a, b, tmp, out = runtime.upload(x), runtime.upload(y), runtime.empty(), runtime.empty()
            for index in range(35):
                # Alternate order to reduce drift; first five iterations are warmup.
                for path in (("copied", "resident") if index % 2 else ("resident", "copied")):
                    start = time.perf_counter()
                    if path == "copied":
                        actual = bridge.mul(bridge.add(x, y), y)
                    else:
                        runtime.run("add", a, b, out=tmp)
                        runtime.run("mul", tmp, b, out=out)
                    elapsed = (time.perf_counter() - start) * 1000
                    if index >= 5:
                        samples[path].append(elapsed)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(runtime.download(out), expected, rtol=0, atol=0)
        copied, resident = (statistics.median(samples[k]) for k in ("copied", "resident"))
        return {"shape": [1, 17], "samples_per_path": 30,
                "copied_chain_median_ms": copied, "resident_chain_median_ms": resident,
                "copied_over_resident": copied / resident,
                "note": "same two SHAVE kernels; resident excludes initial upload/final download; no CPU/GPU comparison"}

    def unary_bulk(dtype):
        evidence = []
        for elements in (2048, 65536):
            x = torch.linspace(-2, 2, elements, dtype=dtype).reshape(1, elements)
            if dtype == torch.float32:
                x[0, :2] = torch.tensor([-2**-149, -2**-140])
            with NpuRuntime(x.shape, dtype) as runtime:
                a, out = runtime.upload(x), runtime.empty()
                iterations = 5 if quick else 50
                for _ in range(iterations):
                    runtime.run("abs", a, out=out)
                torch.testing.assert_close(runtime.download(out), x.abs(), rtol=0, atol=0)
                stats = runtime.stats()
                assert stats["host_upload_copies"] == stats["host_download_copies"] == 1
                evidence.append({"elements": elements, "iterations": iterations, "stats": stats})
        return {"cases": evidence}

    def runtime_failure_guard():
        import npunlock as npu
        from unittest.mock import patch
        with NpuRuntime((17,)) as runtime:
            x = torch.ones(17)
            a, b, out = runtime.upload(x), runtime.upload(x), runtime.empty()
            # Native quarantine is checked separately by test_session_failure.c.
            error = npu.NativeError("graphinfer_session", 10, "driver_failed", b"zeFenceHostSynchronize returned error")
            with patch.object(runtime._sessions["add"], "infer", side_effect=error):
                try:
                    runtime.run("add", a, b, out=out)
                except npu.NativeError:
                    pass
                else:
                    raise AssertionError("Injected failure did not propagate")
            for action in (lambda: runtime.download(out), lambda: runtime.upload(x, out=a),
                           lambda: runtime.run("add", a, b, out=out)):
                try:
                    action()
                except RuntimeError as failure:
                    assert "restart process" in str(failure)
                else:
                    raise AssertionError("Failed runtime allowed access")
            assert runtime.stats()["context_failed"]
        return {"fault_injected": "Python driver-error propagation; no real device failure"}

    def rejection_checks():
        rejected = []
        with NpuRuntime((17,)) as runtime, NpuRuntime((17,)) as other:
            x = torch.ones(17)
            a, b, out = runtime.upload(x), runtime.upload(x), runtime.empty()
            foreign = other.empty()
            dead = runtime.empty()
            dead.release()
            actions = {
                "foreign_context": lambda: runtime.run("add", a, foreign, out=out),
                "released_buffer": lambda: runtime.run("add", a, dead, out=out),
                "output_alias": lambda: runtime.run("mul", a, b, out=a),
                "unsupported_op": lambda: runtime.run("sin", a, out=out),
                "wrong_dtype": lambda: runtime.upload(x.half(), out=out),
                "wrong_shape": lambda: runtime.upload(torch.ones(1, 17), out=out),
                "requires_grad": lambda: runtime.upload(x.requires_grad_(), out=out),
            }
            before = runtime.stats()["npu_calls"]
            for name, action in actions.items():
                try:
                    action()
                except (ValueError, RuntimeError):
                    rejected.append(name)
                else:
                    raise AssertionError(f"Accepted {name}")
            assert runtime.stats()["npu_calls"] == before
            runtime.close()
            try:
                runtime.download(a)
            except RuntimeError:
                rejected.append("closed_runtime")
            else:
                raise AssertionError("Closed runtime accepted")
        return {"rejections": rejected}

    try:
        for dtype in (torch.float16, torch.float32):
            check(f"{dtype}_chain_{20 if quick else 1000}", lambda d=dtype: chain(d, (1, 17), 20 if quick else 1000))
            for shape in [(), (3, 5), (2, 3, 7), (1, 2048), (1, 65536)]:
                check(f"{dtype}_chain_{shape}", lambda d=dtype, s=shape: chain(d, s, 3 if quick else 20))
        check("native_context_lifetime", native_lifetime)
        check("invalid_inputs", rejection_checks)
        check("runtime_failure_guard", runtime_failure_guard)
        check("concurrent_submissions", concurrency)
        check("repeated_close_and_packing", repeated_close)
        for dtype in (torch.float16, torch.float32):
            check(f"{dtype}_resident_arithmetic", lambda d=dtype: arithmetic(d))
            check(f"{dtype}_resident_abs_bulk", lambda d=dtype: unary_bulk(d))
            check(f"{dtype}_submission_timing", lambda d=dtype: timing_comparison(d))
        report.update(status="passed", seconds=time.perf_counter() - started)
        save("complete")
        return 0
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        save(report["stage"])
        print(report["traceback"], flush=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.child:
        raise SystemExit(execute(args.quick))
    command = [sys.executable, __file__, "--child"] + (["--quick"] if args.quick else [])
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180, env=os.environ.copy())
        (OUT / "torch-npu-session.log").write_text(result.stdout + result.stderr)
        print(result.stdout + result.stderr, flush=True)
        raise SystemExit(result.returncode)
    except subprocess.TimeoutExpired:
        report = json.loads(DESTINATION.read_text()) if DESTINATION.exists() else {}
        report.update(status="timeout", error="Outer deadline: 180 seconds")
        DESTINATION.write_text(json.dumps(report, indent=2))
        raise SystemExit(1)

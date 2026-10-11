"""Repeatable hardware checks for FP16/FP32 Torch arithmetic and precision."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "npunlock-probe/research/results"
DESTINATION = OUT / "torch-npu-precision.json"


def execute(quick):
    import numpy as np
    import torch
    import torch_npu_minimal as bridge
    from npunlock.npu4000 import compile_precision_graph
    import npunlock as npu

    report = {"status": "running", "stage": "import", "checks": [],
        "torch_version": torch.__version__, "npunlock_source": importlib.util.find_spec("npunlock").origin,
        "openvino_installed": importlib.util.find_spec("openvino") is not None}
    start = time.perf_counter()
    torch._dynamo.config.recompile_limit = 64

    def save(stage):
        report["stage"] = stage
        report["execution"] = bridge.execution_info()
        DESTINATION.write_text(json.dumps(report, indent=2))
        print(f"{stage}: {len(report['checks'])} checks, {report['execution']['successful_npu_calls']} NPU calls", flush=True)

    def check(name, function, expected, *, rtol=0., atol=0., require_npu=True):
        before = bridge.execution_info()["successful_npu_calls"]
        actual = function()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        finite_zero = expected == 0
        if torch.any(finite_zero) and atol == 0:
            assert torch.equal(torch.signbit(actual[finite_zero]), torch.signbit(expected[finite_zero])), name
        calls = bridge.execution_info()["successful_npu_calls"] - before
        assert calls > 0 or not require_npu, f"{name} did not execute on NPU"
        item = {"name": name, "elements": actual.numel(), "npu_calls": calls,
                "rtol": rtol, "atol": atol, "status": "passed"}
        if actual.dtype == torch.float32:
            mask = torch.isfinite(actual) & torch.isfinite(expected)
            if torch.any(mask):
                aa = actual[mask].contiguous().view(torch.int32).long()
                bb = expected[mask].contiguous().view(torch.int32).long()
                item["max_ulp_error_finite"] = int((aa - bb).abs().max())
        report["checks"].append(item)
        return actual

    try:
        generator = torch.Generator().manual_seed(456)
        save("elementwise")
        for dtype in (torch.float16, torch.float32):
            a = torch.tensor([0., -0., 1., -1., 1.000244140625, 2**-20, 2**-14, 65504., -65504.,
                              100., -100., float("inf"), -float("inf"), float("nan"), .5, -.5, 2.], dtype=dtype)
            b = torch.tensor([1., -1., 2., 3., 2**-12, 2., .5, -65504., 65504., 3., -3., 1., -1., 2., 2., 0., 4.], dtype=dtype)
            for op in ("add", "sub", "mul", "div"):
                expected = getattr(torch, op)(a, b)
                check(f"{dtype}_{op}_specials", lambda: getattr(bridge, op)(a, b), expected,
                      rtol=3e-7 if op == "div" and dtype == torch.float32 else 0)
            for op in ("neg", "abs", "sign"):
                check(f"{dtype}_{op}_specials", lambda: getattr(bridge, op)(a), getattr(torch, op)(a))
            check(f"{dtype}_add1", lambda: bridge.add1(a), a + 1)
            check(f"{dtype}_scalar", lambda: bridge.add(a, 1.0001), a + 1.0001)
            x = torch.tensor([-65504., -1., .25], dtype=dtype)
            check(f"{dtype}_scalar_range", lambda: bridge.add(x, 70000.), x + 70000.)
            for scalar_op in ("mul", "div"):
                check(f"{dtype}_{scalar_op}_scalar_range", lambda op=scalar_op: getattr(bridge, op)(x, 70000.),
                      getattr(torch, scalar_op)(x, 70000.), rtol=3e-7 if dtype == torch.float32 else 0)
            check(f"{dtype}_alpha", lambda: bridge.add(x, x, alpha=.0001), torch.add(x, x, alpha=.0001),
                  rtol=2e-7 if dtype == torch.float32 else 0)
            check(f"{dtype}_alpha_rounding", lambda: bridge.add(x, torch.ones_like(x), alpha=1.0001),
                  torch.add(x, torch.ones_like(x), alpha=1.0001))
            for shape in [(), (1,), (1, 17), (3, 5), (2, 3, 7), (1, 3, 16, 17)]:
                x = torch.randn(shape, generator=generator, dtype=dtype)
                y = torch.randn(shape, generator=generator, dtype=dtype)
                check(f"{dtype}_shape{shape}", lambda: bridge.mul(bridge.add(x, y), y), (x + y) * y)
            x = torch.randn((3, 10), generator=generator, dtype=dtype)[:, ::2]
            y = torch.randn((1, 5), generator=generator, dtype=dtype)
            check(f"{dtype}_strides_broadcast", lambda: bridge.add(x, y), x + y)
        x = torch.tensor([70000., -70000., 1.000244140625, 2**-140], dtype=torch.float32)
        check("fp32_range_add", lambda: bridge.add(x, x), x + x)
        check("fp32_precision_mul", lambda: bridge.mul(x, 2), x * 2)
        tiny = torch.tensor([2**-149, -2**-149, 2**-140, -2**-140, 2**-126, -(2**-126),
                             2**-125, 0., -0.], dtype=torch.float32)
        for op in ("neg", "abs", "sign"):
            check(f"fp32_subnormal_{op}", lambda op=op: getattr(bridge, op)(tiny), getattr(torch, op)(tiny))
        for op, value in [("add", tiny), ("sub", tiny), ("mul", torch.full_like(tiny, 2)),
                          ("div", torch.full_like(tiny, 2))]:
            check(f"fp32_subnormal_{op}", lambda: getattr(bridge, op)(tiny, value), getattr(torch, op)(tiny, value))
        edge_a = torch.tensor([2**-149, 0., -0., float("inf"), -float("inf"), 2**-149], dtype=torch.float32)
        edge_b = torch.tensor([0., 2**-149, -2**-149, 2**-149, -2**-149, float("inf")], dtype=torch.float32)
        for op in ("mul", "div"):
            check(f"fp32_subnormal_inf_zero_{op}", lambda: getattr(bridge, op)(edge_a, edge_b),
                  getattr(torch, op)(edge_a, edge_b))

        save("casts_and_ieee_edges")
        h = torch.tensor([0., -0., 2**-24, -2**-24, 2**-20, 2**-14, 65504., float("inf"), float("nan")], dtype=torch.float16)
        check("half_to_float_subnormals", lambda: bridge.cast(h, torch.float32), h.float())
        check("float_to_half_subnormals", lambda: bridge.cast(h.float(), torch.float16), h)
        ties = torch.tensor([2**-25, 2**-25 + 2**-48, -(2**-25), 1 + 2**-11, 1 + 3*2**-11,
                             65504., 65519., 65520., 70000., -70000., 2**-149], dtype=torch.float32)
        check("half_round_to_nearest_even", lambda: bridge.cast(ties, torch.float16), ties.half())
        if not quick:
            all_half = torch.from_numpy(np.arange(65536, dtype=np.uint16).view(np.float16).copy())
            lifted = check("all_65536_half_encodings_to_float", lambda: bridge.cast(all_half, torch.float32), all_half.float())
            check("all_65536_half_encodings_roundtrip", lambda: bridge.cast(lifted, torch.float16), all_half)
            for op in ("neg", "abs"):
                check(f"all_65536_half_encodings_{op}", lambda: getattr(bridge, op)(all_half), getattr(torch, op)(all_half))
            rng = np.random.default_rng(123)
            a = torch.from_numpy(rng.integers(0, 2**32, 8192, dtype=np.uint32).view(np.float32).copy())
            b = torch.from_numpy(rng.integers(0, 2**32, 8192, dtype=np.uint32).view(np.float32).copy())
            for op in ("add", "sub", "mul", "div"):
                check(f"8192_random_float_bits_{op}", lambda: getattr(bridge, op)(a, b), getattr(torch, op)(a, b),
                      rtol=3e-7 if op == "div" else 0)
            check("8192_random_float_bits_to_half", lambda: bridge.cast(a, torch.float16), a.half())
        save("reductions_and_matmul")
        for dtype in (torch.float16, torch.float32):
            x = torch.randn((2, 3, 5), generator=generator, dtype=dtype)
            for op in ("sum", "mean"):
                for dim, keepdim in [(None, False), (1, True), ((0, 2), False), (-1, False)]:
                    check(f"{dtype}_{op}_{dim}_{keepdim}", lambda: getattr(bridge, op)(x, dim, keepdim),
                          getattr(torch, op)(x, dim=dim, keepdim=keepdim),
                          rtol=2e-3 if dtype == torch.float16 else 2e-6, atol=2e-3 if dtype == torch.float16 else 2e-6)
            for m, k, n in [(1, 1, 1), (3, 5, 4), (8, 17, 7)] + ([] if quick else [(32, 32, 64)]):
                a = torch.randn((m, k), generator=generator, dtype=dtype)
                b = torch.randn((k, n), generator=generator, dtype=dtype)
                check(f"{dtype}_matmul_{m}_{k}_{n}", lambda: bridge.matmul(a, b), a @ b,
                      rtol=3e-3 if dtype == torch.float16 else 3e-5, atol=3e-3 if dtype == torch.float16 else 3e-6)
        a = torch.tensor([[70000., 1.000244140625, 2**-140]], dtype=torch.float32)
        b = torch.eye(3, dtype=torch.float32)
        check("fp32_matmul_preserves_range_precision_subnormal", lambda: bridge.matmul(a, b), a @ b)
        check("fp32_sum_subnormals", lambda: bridge.sum(tiny), tiny.sum())
        check("fp32_mean_subnormals", lambda: bridge.mean(tiny), tiny.mean(), rtol=3e-7)
        z = torch.tensor([-0., -0.])
        check("sum_negative_zeros", lambda: bridge.sum(z), z.sum())
        check("explicit_reduction_dtype", lambda: bridge.sum(ties, dtype=torch.float16), ties.sum(dtype=torch.float16))
        rounding = torch.tensor([2049., -2048.])
        check("mean_fp32_input_half_output", lambda: bridge.mean(rounding, dtype=torch.float16),
              rounding.mean(dtype=torch.float16))
        a = torch.tensor([1., 2., 3.], dtype=torch.float16)
        b = torch.tensor([.00001, .00002, .00003], dtype=torch.float32)
        check("mixed_tensor_dtype", lambda: bridge.add(a, b), a + b)

        save("torch_compile")
        functions = [
            ("chain", lambda a, b: ((a + b) * b - a) / 2),
            ("methods_casts", lambda a, b: a.float().add(b.float()).half()),
            ("reduce", lambda a, b: (a * b).sum(dim=1)),
            ("mean", lambda a, b: (a + b).mean()),
            ("unary", lambda a, b: (-a).abs()),
            ("matrix", lambda a, b: a @ b),
        ]
        for dtype in (torch.float16, torch.float32):
            for name, function in functions:
                a = torch.randn((3, 5), generator=generator, dtype=dtype)
                b = torch.randn((5, 4) if name == "matrix" else (3, 5), generator=generator, dtype=dtype)
                compiled = torch.compile(function, backend="npunlock_npu", fullgraph=True, dynamic=False)
                check(f"compile_{dtype}_{name}", lambda: compiled(a, b), function(a, b),
                      rtol=2e-3 if dtype == torch.float16 else 3e-6, atol=2e-3 if dtype == torch.float16 else 3e-6)
        save("autograd")
        for dtype in (torch.float16, torch.float32):
            for name in ("chain", "matrix", "cast", "abs"):
                a = torch.randn((3, 5), generator=generator, dtype=dtype).requires_grad_(True)
                b = torch.randn((5, 4) if name == "matrix" else (3, 5), generator=generator, dtype=dtype).requires_grad_(True)
                aa, bb = a.detach().clone().requires_grad_(True), b.detach().clone().requires_grad_(True)
                if name == "matrix":
                    got, expected = bridge.mean(bridge.matmul(a, b)), (aa @ bb).mean()
                elif name == "cast":
                    got, expected = bridge.mean(bridge.cast(a, torch.float32)), aa.float().mean()
                elif name == "abs":
                    got, expected = bridge.sum(bridge.abs(a)), aa.abs().sum()
                else:
                    got = bridge.mean(bridge.div(bridge.mul(bridge.add(a, b), b), 3))
                    expected = (((aa + bb) * bb) / 3).mean()
                got.backward()
                expected.backward()
                torch.testing.assert_close(a.grad, aa.grad, rtol=5e-3 if dtype == torch.float16 else 3e-6,
                                           atol=2e-3 if dtype == torch.float16 else 2e-6)
                if b.grad is not None:
                    torch.testing.assert_close(b.grad, bb.grad, rtol=5e-3 if dtype == torch.float16 else 3e-6,
                                               atol=2e-3 if dtype == torch.float16 else 2e-6)
                report["checks"].append({"name": f"autograd_{dtype}_{name}", "status": "passed"})
        if not quick:
            save("opcheck")
            x = torch.randn((3, 5), generator=generator, requires_grad=True)
            y = torch.randn((3, 5), generator=generator, requires_grad=True)
            report["opcheck"] = {
                "unary": torch.library.opcheck(bridge.unary, (x, "abs")),
                "binary": torch.library.opcheck(bridge.binary, (x, y, "mul")),
                "cast": torch.library.opcheck(bridge.cast, (x, torch.float16)),
                "reduce": torch.library.opcheck(bridge._reduce, (x, "sum", [1], False)),
                "matmul": torch.library.opcheck(bridge._matmul, (x, torch.randn((5, 4), requires_grad=True))),
            }
        save("rejections")
        before = bridge.execution_info()["successful_npu_calls"]
        rejected = []
        for name, function in [
            ("fp64", lambda: bridge.add1(torch.ones(3, dtype=torch.float64))),
            ("bf16", lambda: bridge.add1(torch.ones(3, dtype=torch.bfloat16))),
            ("empty", lambda: bridge.add1(torch.empty(0))),
            ("rank5", lambda: bridge.add1(torch.ones(1, 1, 1, 1, 1))),
            ("oversize", lambda: bridge.add1(torch.empty(65537))),
            ("cast_integer", lambda: bridge.cast(torch.ones(3), torch.int32)),
            ("matrix_product_bound", lambda: bridge.matmul(torch.ones(64, 64), torch.ones(64, 64))),
            ("duplicate_axes", lambda: bridge.sum(torch.ones(2, 3), dim=(1, 1))),
            ("division_rounding", lambda: bridge.div(torch.ones(3), 2, rounding_mode="trunc")),
            ("unknown_fx", lambda: torch.compile(lambda x: torch.sin(x), backend="npunlock_npu", fullgraph=True)(torch.ones(3))),
        ]:
            try:
                function()
            except (ValueError, NotImplementedError, torch._dynamo.exc.BackendCompilerFailed):
                rejected.append(name)
            else:
                raise AssertionError(f"Unsupported {name} accepted")
        assert bridge.execution_info()["successful_npu_calls"] == before
        x = npu.input("x", shape=(3, 5), dtype="f32")
        b = npu.input("b", shape=(5, 4), dtype="f32")
        y = npu.op("MatMul", x, b, transpose_a=False, transpose_b=False, _shape=(3, 4), _dtype="f32")
        try:
            compile_precision_graph(npu.Graph([x, b], [y]))
        except ValueError as error:
            assert "DPU" in str(error)
            rejected.append("driver_fp32_matmul_lowering")
        else:
            raise AssertionError("Unverified DPU FP32 MatMul accepted")
        report["rejections"] = rejected
        report.update(status="passed", seconds=time.perf_counter() - start)
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
    if args.child:
        raise SystemExit(execute(args.quick))
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    prefix = Path(sys.executable).parent
    env["PATH"] = f"{prefix};{prefix / 'Library/bin'};{prefix / 'Scripts'};" + env["PATH"]
    command = [sys.executable, __file__, "--child"] + (["--quick"] if args.quick else [])
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=240, env=env)
        (OUT / "torch-npu-precision.log").write_text(result.stdout + result.stderr)
        print(result.stdout + result.stderr, flush=True)
        raise SystemExit(result.returncode)
    except subprocess.TimeoutExpired:
        report = json.loads(DESTINATION.read_text()) if DESTINATION.exists() else {}
        report.update(status="timeout", error="Outer deadline: 240 seconds")
        DESTINATION.write_text(json.dumps(report, indent=2))
        raise SystemExit(1)

"""Device execution checks for custom FP16/FP32 unary and binary kernels."""
import json
from pathlib import Path
import numpy as np
import npunlock as npu
from npunlock.npu4000 import compile_unary, compile_binary

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "npunlock/tests/fixtures/npu4000/precision"
OUT = ROOT / "npunlock-probe/research/results/precision"


def main():
    native = npu.NativeLibraries()
    report = []
    for dtype in ("f16", "f32"):
        npdtype = np.float16 if dtype == "f16" else np.float32
        a = np.array([0., -0., 1., -1., 1.000244140625, 2**-20, 2**-14, 65504.,
                      -65504., 70000. if dtype == "f32" else 100., -70000. if dtype == "f32" else -100.,
                      np.inf, -np.inf, np.nan, 0.5, -0.5, 2.], dtype=npdtype).reshape(1, -1)
        b = np.array([1., -1., 2., 3., 2**-12, 2., 0.5, -65504., 65504., -69999. if dtype == "f32" else 3.,
                      69999. if dtype == "f32" else -3., 1., -1., 2., 2., 0., 4.], dtype=npdtype).reshape(1, -1)
        for operation in ("add1", "neg", "abs", "add", "sub", "mul", "div"):
            kernel = (KERNELS / f"{operation}-{dtype}.elf").read_bytes()
            unary = operation in ("add1", "neg", "abs")
            program = (compile_unary if unary else compile_binary)(a.shape, kernel, dtype=dtype, libraries=native)
            inputs = [npu.InferenceInput("x", a.tobytes())]
            if not unary:
                inputs += [npu.InferenceInput("b", b.tobytes())]
            result = native.infer_graph(program.graph_blob, inputs, timeout_ms=10_000)
            assert result.device_id == 0x643e
            actual = np.frombuffer(result.outputs[0].data, dtype=npdtype).reshape(a.shape)
            with np.errstate(all="ignore"):
                af, bf = a.astype(np.float32), b.astype(np.float32)
                expected = {"add1": lambda: af + 1, "neg": lambda: -af, "abs": lambda: np.abs(af),
                    "add": lambda: af + bf, "sub": lambda: af - bf, "mul": lambda: af * bf,
                    "div": lambda: af / bf}[operation]().astype(npdtype)
            np.testing.assert_allclose(actual, expected, rtol=0 if operation != "div" else 3e-7, atol=0, equal_nan=True)
            zeros = (expected == 0)
            assert np.array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))
            report.append({"operation": operation, "dtype": dtype, "status": "passed",
                           "device_id": hex(result.device_id), "patch": json.loads(program.patch_reports[0])})
            print(json.dumps({k: v for k, v in report[-1].items() if k != "patch"}), flush=True)
    (OUT / "custom-kernels-hardware.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

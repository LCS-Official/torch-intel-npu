"""Compile, validate and execute variable-shape NPU4000 unary kernels."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/unary-shapes"
SHAPES = [(1,), (1, 8), (1, 16), (1, 17), (1, 31), (1, 32), (1, 33),
          (1, 2048), (1, 4097), (3, 5), (32, 64), (1, 3, 16, 17), (1, 65536), (256, 256), (2, 3, 7)]


def compile_carrier(shape):
    import numpy as np
    import npunlock as npu
    from npunlock.npu4000 import Elf, compile_unary, inspect
    kernel = (ROOT / "results/4000xx.elf").read_bytes()
    program = compile_unary(shape, kernel, timeout_ms=20_000)
    original = program.ir_provenance.graph_blob
    stem = "x".join(map(str, shape))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}-abs.blob").write_bytes(original)
    program.save(OUT / f"{stem}-add1.blob")
    (OUT / f"{stem}-patch.json").write_bytes(program.patch_reports[0])
    evidence = inspect(original)
    (OUT / f"{stem}-inspection.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    elf = Elf(original)
    refs = [r for task in evidence["invocations"] for r in task.get("unary_memrefs", [])]
    summary = {"shape": shape, "device": hex(program.ir_provenance.device_id),
               "ranges": len(evidence["ranges"]), "invocations": len(evidence["invocations"]),
               "windows": sorted(set((t["code_size"], t["text_relocation"]["symbol_value"] + t["text_relocation"]["addend"])
                                      for t in evidence["ranges"] if t["workload_type"] == 0)),
               "memrefs": [dict(dims=r["dims"], strides=r["strides_bits"], order=r["order"],
                                data=r["unrelocated_data"], relocs=r["data_relocations"]) for r in refs],
               "text_size": elf.named["shave.text"]["size"]}
    rng = np.random.default_rng(456)
    cases = [(np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 127 - 63).astype(np.float16)]
    cases += [rng.uniform(-4, 4, shape).astype(np.float16) for _ in range(2)]
    native = npu.NativeLibraries()
    baseline = npu.load_native(original, graph=program.graph)
    np.testing.assert_array_equal(baseline.run({"x": cases[0]})["y"], np.abs(cases[0]))
    comparisons = []
    for value in cases:
        result = native.infer_graph(program.graph_blob, [npu.InferenceInput("x", value.tobytes())], timeout_ms=10_000)
        if (result.vendor_id, result.device_id) != (0x8086, 0x643e):
            raise AssertionError("Wrong device selected")
        output = result.outputs[0]
        if output.shape != shape or output.dtype != "f16" or len(output.data) != value.nbytes:
            raise AssertionError("Incorrect output contract")
        actual = np.frombuffer(output.data, dtype=np.float16).reshape(shape)
        expected = (value.astype(np.float32) + 1).astype(np.float16)
        np.testing.assert_array_equal(actual, expected)
        comparisons.append({"max_abs_error": float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))})
    summary.update(status="passed", baseline_abs=True, comparisons=comparisons,
                   execution_device=hex(result.device_id), driver_version=result.driver_version)
    (OUT / f"{stem}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "memrefs"}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        compile_carrier(tuple(map(int, sys.argv[1].split("x"))))
    else:
        env = os.environ.copy()
        prefix = Path(sys.executable).parent
        env["PATH"] = f"{prefix};{prefix / 'Library/bin'};{prefix / 'Scripts'};" + env["PATH"]
        results = []
        OUT.mkdir(parents=True, exist_ok=True)
        for shape in SHAPES:
            stem = "x".join(map(str, shape))
            result = subprocess.run([sys.executable, __file__, "x".join(map(str, shape))],
                                    env=env, capture_output=True, text=True, timeout=45)
            print(result.stdout + result.stderr, flush=True)
            (OUT / f"{stem}-hardware.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise SystemExit(result.returncode)
            results.append(json.loads((OUT / f"{stem}-summary.json").read_text()))
        (OUT / "hardware-matrix.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

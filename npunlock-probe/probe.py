"""Bounded, staged Lunar Lake smoke tests using the published npunlock wheel."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
CASES = ("abs", "add", "subtract", "multiply", "matmul", "custom", "toolchain")

KERNEL = b"""
#include <npunlock/npu3720_kernel.h>
void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *in = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    __fp16 *out = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i)
        out[i] = (__fp16)((float)in[i] + 1.0f);
}
"""


def run_case(case, copied_only=False):
    report = {"case": case, "stage": "import", "status": "running", "python": sys.version}
    path = RESULTS / f"{case}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def checkpoint(stage):
        report["stage"] = stage
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{case}: {stage}", flush=True)

    try:
        import numpy as np
        import npunlock as npu
        report["npunlock_module"] = npu.__file__
        if case == "toolchain":
            checkpoint("compile_custom_kernel")
            elf = npu.NativeLibraries().compile_shave(
                KERNEL, movi_dll_dir=os.environ["NPUNLOCK_MOVITOOLS_DIR"], timeout_ms=20000)
            (RESULTS / "add1.elf").write_bytes(elf)
            report.update(status="passed", elf_bytes=len(elf))
            checkpoint("complete")
            return 0
        checkpoint("build_graph")
        rng = np.random.default_rng(123)
        shape = (32, 32) if case == "matmul" else (1, 2048)
        if case == "custom":
            shape = (1, 16)
        a = rng.uniform(-2, 2, shape).astype(np.float16)
        b = rng.uniform(-2, 2, shape).astype(np.float16)
        if case == "custom":
            a = np.arange(-8, 8, dtype=np.float16).reshape(shape)
        x = npu.input("x", shape=shape, dtype="f16")
        inputs, arrays = [x], {"x": a}
        if case in ("abs", "custom"):
            y = npu.Abs(x, _shape=shape, _dtype="f16", _name="y")
            expected = np.abs(a) if case == "abs" else a + np.float16(1)
        else:
            z = npu.input("z", shape=shape, dtype="f16")
            inputs.append(z)
            arrays["z"] = b
            operations = {"add": (npu.Add, np.add),
                          "subtract": (npu.Subtract, np.subtract),
                          "multiply": (npu.Multiply, np.multiply),
                          "matmul": (npu.MatMul, np.matmul)}
            factory, oracle = operations[case]
            attributes = {"transpose_a": False, "transpose_b": False} if case == "matmul" else {}
            y = factory(x, z, _shape=shape, _dtype="f16", _name="y", **attributes)
            expected = oracle(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
        graph = npu.Graph(inputs=inputs, outputs=[y], name=f"lnl_{case}")
        serialized = npu.serialize_ir(graph)
        (RESULTS / f"{case}.xml").write_bytes(serialized.xml)
        checkpoint("compile_ir")
        start = time.perf_counter()
        program = npu.compile(graph, timeout_ms=20000)
        report["compile_ms"] = (time.perf_counter() - start) * 1000
        provenance = dataclasses.asdict(program.ir_provenance)
        provenance.pop("graph_blob")
        report["provenance"] = provenance
        report["blob_bytes"] = len(program.graph_blob)
        program.save(RESULTS / f"{case}.blob")
        if case == "custom":
            native = npu.NativeLibraries()
            checkpoint("execute_unpatched_carrier")
            carrier_output = program.run(arrays)["y"]
            np.testing.assert_array_equal(carrier_output, np.abs(a))
            report["carrier_correctness_passed"] = True
            np.savez(RESULTS / "custom-carrier-tensors.npz", x=a, actual=carrier_output, expected=np.abs(a))
            checkpoint("discover_act_targets")
            groups = native.discover_patch_targets(program.graph_blob)
            report["target_groups"] = [[dataclasses.asdict(t) for t in group] for group in groups]
            if not groups:
                raise RuntimeError("No patchable ACT target found; no offsets will be forced.")
            checkpoint("compile_custom_kernel")
            elf = native.compile_shave(KERNEL, movi_dll_dir=os.environ["NPUNLOCK_MOVITOOLS_DIR"], timeout_ms=20000)
            (RESULTS / "add1.elf").write_bytes(elf)
            checkpoint("patch_blob")
            patched = native.patch_graph(program.graph_blob, elf, groups[0])
            (RESULTS / "custom-patch.json").write_bytes(patched.report_json)
            (RESULTS / "custom-patched.blob").write_bytes(patched.graph_blob)
            program = npu.load_native(patched.graph_blob, graph=graph, timeout_ms=20000)
        checkpoint("execute_and_check")
        start = time.perf_counter()
        actual = program.run(arrays)["y"]
        report["first_run_ms"] = (time.perf_counter() - start) * 1000
        report["max_abs_error"] = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
        tolerance = {"rtol": 0.02, "atol": 0.02} if case == "matmul" else {"rtol": 0, "atol": 0}
        report["tolerance"] = tolerance
        np.testing.assert_allclose(actual, expected, **tolerance)
        report["correctness_passed"] = True
        np.savez(RESULTS / f"{case}-tensors.npz", **arrays, actual=actual, expected=expected)
        copied_timings = [report["first_run_ms"]]
        for _ in range(2):
            start = time.perf_counter()
            repeated = program.run(arrays)["y"]
            copied_timings.append((time.perf_counter() - start) * 1000)
            np.testing.assert_allclose(repeated, expected, **tolerance)
        report["copied_run_median_ms"] = float(np.median(copied_timings))
        report["copied_run_samples"] = len(copied_timings)
        if copied_only:
            report["status"] = "passed"
            checkpoint("complete")
            return 0
        checkpoint("reused_session_timing")
        shared_inputs = {name: program.shared_array(value.shape, value.dtype) for name, value in arrays.items()}
        for name, value in arrays.items():
            shared_inputs[name][:] = value
        shared_outputs = {"y": program.shared_array(shape, "f16")}
        for _ in range(3):
            program.run(shared_inputs, outputs=shared_outputs)
        timings = []
        for _ in range(20):
            start = time.perf_counter()
            program.run(shared_inputs, outputs=shared_outputs)
            timings.append((time.perf_counter() - start) * 1000)
        np.testing.assert_allclose(shared_outputs["y"], expected, **tolerance)
        report["shared_run_median_ms"] = float(np.median(timings))
        report["shared_run_samples"] = len(timings)
        report["status"] = "passed"
        checkpoint("complete")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report["traceback"], flush=True)
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--copied-only", action="store_true", help="Skip experimental shared session timing")
    args = parser.parse_args()
    if args.case:
        raise SystemExit(run_case(args.case, args.copied_only))
    failed = False
    for case in args.cases:
        try:
            command = [sys.executable, __file__, "--case", case]
            if args.copied_only:
                command.append("--copied-only")
            completed = subprocess.run(command,
                                       capture_output=True, text=True, timeout=60)
            log = completed.stdout + completed.stderr
            print(log, flush=True)
            (RESULTS / f"{case}.log").write_text(log, encoding="utf-8")
            path = RESULTS / f"{case}.json"
            report = json.loads(path.read_text()) if path.exists() else {"case": case}
            report["process_exit_code"] = completed.returncode
            failed |= completed.returncode != 0
            if completed.returncode and report.get("status") == "running":
                report.update(status="process_error", error=f"Process exited with code {completed.returncode}")
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except subprocess.TimeoutExpired as error:
            failed = True
            path = RESULTS / f"{case}.json"
            report = json.loads(path.read_text()) if path.exists() else {"case": case}
            report.update(status="timeout", error="Outer process deadline: 60 seconds")
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"{case}: timed out", flush=True)
    raise SystemExit(1 if failed else 0)

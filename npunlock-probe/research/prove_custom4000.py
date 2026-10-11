"""Bounded hardware proof: replace only the verified unary carrier's code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import traceback

from inspect_npu4000 import Elf, bounded, inspect, string_at

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PROJECT = ROOT.parent.parent
PREFIX = PROJECT / ".conda/npunlock"


def text32(data):
    if data[:6] != b"\x7fELF\x01\x01" or struct.unpack_from("<HH", data, 16) != (2, 2):
        raise ValueError("Requires executable little-endian SPARC/SHAVE ELF32")
    entry, shoff = struct.unpack_from("<I", data, 24)[0], struct.unpack_from("<I", data, 32)[0]
    size, count, names_index = struct.unpack_from("<HHH", data, 46)
    if size != 40 or names_index >= count:
        raise ValueError("Unsupported ELF32 section table")
    table = bounded(data, shoff, count * size)
    sections = [struct.unpack_from("<10I", table, i * size) for i in range(count)]
    names_header = sections[names_index]
    names = bounded(data, names_header[4], names_header[5])
    named = {string_at(names, h[0]): h for h in sections}
    text = named[".text"]
    if text[1] != 1 or text[3] != 0x1D000000 or entry != text[3]:
        raise ValueError("Unexpected text entry/address")
    arg = named[".arg.data"]
    if arg[3] != 0x1E000000 or arg[5] != 0:
        raise ValueError("This proof only supports empty kernel data")
    if any(h[1] in (4, 9) and h[5] for h in sections):
        raise ValueError("Kernel ELF has unresolved relocations")
    return bounded(data, text[4], text[5])


def build_candidate():
    original = (ROOT.parent / "results/custom.blob").read_bytes()
    elf = Elf(original)
    evidence = inspect(original)
    official = text32((ROOT / "sources/npu_compiler/sw_runtime_kernels/kernels/prebuild/act_shave_bin/activation_abs.4000xx.elf").read_bytes())
    new_code = text32((OUT / "4000xx.elf").read_bytes())
    text = elf.named["shave.text"]
    if elf.content(text)[:len(official)] != official:
        raise ValueError("Carrier does not exactly match the verified official NPU4000 Abs code")
    if len(evidence["ranges"]) != 8 or len(evidence["invocations"]) != 8:
        raise ValueError("This proof is limited to the observed 1x16 carrier")
    for task in evidence["ranges"]:
        reloc = task["text_relocation"]
        if (task["workload_type"] != 0 or task["kernel_entry"] != "0x1d000000"
            or task["code_size"] != len(official) or reloc["type"] != 0
            or reloc["symbol_section"] != "shave.text" or reloc["symbol_value"] + reloc["addend"] != 0):
            raise ValueError("Unexpected code window/range contract")
    for task in evidence["invocations"]:
        for ref in task["unary_memrefs"]:
            if (ref["static"] != 1 or ref["dtype"] != 2 or ref["location"] != 2
                or ref["elements"] != 8 or ref["data_relocations"]):
                raise ValueError("Unexpected parameter contract")
    if not 0 < len(new_code) <= len(official):
        raise ValueError("New kernel must fit in the existing code window")
    patched = bytearray(original)
    begin, end = text["offset"], text["offset"] + len(new_code)
    patched[begin:end] = new_code
    if (len(patched) != len(original) or patched[:begin] != original[:begin]
        or patched[end:] != original[end:]):
        raise ValueError("Unexpected modification outside the replacement code")
    (OUT / "add1-npu4000-inplace.blob").write_bytes(patched)
    report = dict(status="candidate_built", original_sha256=hashlib.sha256(original).hexdigest(),
                  patched_sha256=hashlib.sha256(patched).hexdigest(), original_bytes=len(original),
                  code_offset=begin, replacement_code_bytes=len(new_code), original_code_window_bytes=len(official),
                  changed_bytes=sum(a != b for a, b in zip(original, patched)),
                  limitation="Fixed 1x16 unary carrier; no ELF growth, generic mapping, or production backend")
    (OUT / "inplace-patch.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def execute():
    import numpy as np
    import npunlock as npu
    report = dict(status="running", stage="load", device_id_expected="0x643e")
    destination = OUT / "custom4000-hardware.json"
    try:
        x = npu.input("x", shape=(1, 16), dtype="f16")
        y = npu.Abs(x, _shape=(1, 16), _dtype="f16", _name="y")
        graph = npu.Graph(inputs=[x], outputs=[y], name="lnl_custom")
        original = npu.load_native((ROOT.parent / "results/custom.blob").read_bytes(), graph=graph, timeout_ms=10000)
        patched = npu.load_native((OUT / "add1-npu4000-inplace.blob").read_bytes(), graph=graph, timeout_ms=10000)
        a = np.arange(-8, 8, dtype=np.float16).reshape(1, 16)
        report["stage"] = "baseline_abs"
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.testing.assert_array_equal(original.run({"x": a})["y"], np.abs(a))
        report["baseline_passed"] = True
        report["stage"] = "custom_add1"
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        rng = np.random.default_rng(456)
        cases = [a] + [rng.uniform(-4, 4, (1, 16)).astype(np.float16) for _ in range(2)]
        comparisons = []
        for index, a in enumerate(cases):
            expected = (a.astype(np.float32) + 1).astype(np.float16)
            if index == 0:
                inference = npu.NativeLibraries().infer_graph(
                    patched.graph_blob, [npu.InferenceInput("x", a.tobytes())], timeout_ms=10000)
                report["execution_provenance"] = dict(
                    vendor_id=hex(inference.vendor_id), device_id=hex(inference.device_id),
                    driver_version=inference.driver_version, driver_index=inference.driver_index,
                    device_index=inference.device_index)
                if inference.device_id != 0x643E or inference.vendor_id != 0x8086:
                    raise ValueError("Execution did not select the expected Intel NPU")
                actual = np.frombuffer(inference.outputs[0].data, dtype=np.float16).reshape(1, 16)
            else:
                actual = patched.run({"x": a})["y"]
            error = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
            comparisons.append(dict(index=index, x=a.tolist(), actual=actual.tolist(), expected=expected.tolist(), max_abs_error=error))
            np.testing.assert_array_equal(actual, expected)
        report.update(status="passed", stage="complete", comparisons=comparisons)
        print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}), flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
        return 1
    finally:
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-child", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.execute_child:
        raise SystemExit(execute())
    print(json.dumps(build_candidate()), flush=True)
    if not args.build_only:
        env = os.environ.copy()
        env["PATH"] = f"{PREFIX};{PREFIX / 'Library/bin'};{PREFIX / 'Scripts'};" + env["PATH"]
        try:
            result = subprocess.run([str(PREFIX / "python.exe"), __file__, "--execute-child"],
                                    capture_output=True, text=True, timeout=45, env=env)
            log = result.stdout + result.stderr
            (OUT / "custom4000-hardware.log").write_text(log, encoding="utf-8")
            print(log, flush=True)
            raise SystemExit(result.returncode)
        except subprocess.TimeoutExpired:
            destination = OUT / "custom4000-hardware.json"
            report = json.loads(destination.read_text()) if destination.exists() else {}
            report.update(status="timeout", error="Outer process deadline: 45 seconds")
            destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise SystemExit(1)

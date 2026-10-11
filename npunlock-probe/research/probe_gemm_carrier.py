"""Compile and inspect candidate GEMM carriers; never execute patched code."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import npunlock as npu
from npunlock.npu4000 import Elf, inspect, _dense_float
from torch_npu_runtime import SESSION_DLL

OUT = Path(__file__).resolve().parent / "results/gemm"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    native = npu.NativeLibraries(session_library=SESSION_DLL)
    records = []
    for size in (4, 8, 16, 32, 64):
        for tiles in (None, 1, "reference"):
            name = f"power-{size}-tiles-{tiles or 'auto'}"
            record = {"name": name, "size": size, "tiles": tiles}
            try:
                x = npu.input("x", shape=(size, size), dtype="f32")
                b = npu.input("b", shape=(size, size), dtype="f32")
                y = npu.op("Power", x, b, _shape=x.shape, _dtype="f32", _name="y")
                ir = npu.serialize_ir(npu.Graph([x, b], [y], name="gemm_carrier_probe"))
                xml = ET.fromstring(ir.xml)
                for layer in xml.findall("./layers/layer"):
                    if layer.get("type") == "Power":
                        info = ET.SubElement(layer, "rt_info")
                        ET.SubElement(info, "attribute", name="DisablePrecisionConversion",
                                      version="0", value="dynamic:f16")
                ir = replace(ir, xml=ET.tostring(xml))
                flags = '--config EXECUTION_MODE_HINT="ACCURACY"'
                if tiles == "reference":
                    flags += ' NPU_TILES="1" NPU_COMPILATION_MODE="ReferenceSW"'
                elif tiles:
                    flags += f' NPU_TILES="{tiles}"'
                result = native.compile_ir(ir.xml, ir.weights, build_flags=flags, timeout_ms=20000)
                assert (result.vendor_id, result.device_id, tuple(result.compiler_version)) == (0x8086, 0x643E, (8, 3))
                blob = result.graph_blob
                evidence = inspect(blob, memref_count=3)
                elf = Elf(blob)
                assert not any(s["size"] and s["name"].startswith("task.dpu.") for s in elf.sections)
                (OUT / f"{name}.blob").write_bytes(blob)
                (OUT / f"{name}.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
                contracts = []
                for task in evidence["invocations"]:
                    refs = task["memrefs"]
                    record["observed_memrefs"] = refs
                    spans = [_dense_float(ref, task["tile"], elf, "f32") for ref in refs]
                    contracts.append({"tile": task["tile"], "dims": [r["dims"] for r in refs],
                                      "strides_bits": [r["strides_bits"] for r in refs],
                                      "spans": spans, "elements": [r["elements"] for r in refs]})
                record.update(status="inspected", flags=flags, invocations=len(contracts), contracts=contracts,
                              whole_matrices=all(r["elements"] == [size * size] * 3 for r in contracts))
                windows = set()
                for task in evidence["ranges"]:
                    reloc = task["text_relocation"]
                    windows.add((reloc["symbol_value"] + reloc["addend"], task["code_size"]))
                record["code_windows"] = [{"offset": off, "bytes": extent,
                    "sha256": hashlib.sha256(elf.content(elf.named["shave.text"])[off:off + extent]).hexdigest()}
                    for off, extent in sorted(windows)]
            except Exception as error:
                record.update(status="rejected", error=str(error))
            records.append(record)
            (OUT / "carrier-summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            print({k: v for k, v in record.items() if k not in ("contracts", "observed_memrefs")}, flush=True)
    for m, k, n in ((4, 4, 4), (3, 5, 7), (16, 16, 16), (64, 64, 64)):
        name = f"matmul-{m}-{k}-{n}-reference"
        record = {"name": name}
        try:
            x = npu.input("x", shape=(m, k), dtype="f32")
            b = npu.input("b", shape=(k, n), dtype="f32")
            y = npu.op("MatMul", x, b, transpose_a=False, transpose_b=False,
                       _shape=(m, n), _dtype="f32", _name="y")
            ir = npu.serialize_ir(npu.Graph([x, b], [y], name="gemm_matmul_probe"))
            xml = ET.fromstring(ir.xml)
            for layer in xml.findall("./layers/layer"):
                if layer.get("type") == "MatMul":
                    info = ET.SubElement(layer, "rt_info")
                    ET.SubElement(info, "attribute", name="DisablePrecisionConversion",
                                  version="0", value="dynamic:f16")
            flags = '--config EXECUTION_MODE_HINT="ACCURACY" NPU_TILES="1" NPU_COMPILATION_MODE="ReferenceSW"'
            result = native.compile_ir(ET.tostring(xml), ir.weights, build_flags=flags, timeout_ms=20000)
            blob = result.graph_blob
            (OUT / f"{name}.blob").write_bytes(blob)
            # MatMul lowers to a mix of unary permute/cast and binary FC tasks.
            evidence = inspect(blob, memref_count=2)
            (OUT / f"{name}.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            lowered = any(r['dtype'] != 1 for t in evidence['invocations'] for r in t.get('memrefs', []))
            record.update(status="rejected_precision_lowering" if lowered else "inspected",
                          ranges=len(evidence['ranges']),
                          invocations=len(evidence['invocations']),
                          refs=[[(r['dims'], r['dtype'], r['location']) for r in t.get('memrefs', [])]
                                for t in evidence['invocations']])
        except Exception as error:
            record.update(status="rejected", error=str(error))
        records.append(record)
        (OUT / "carrier-summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(record, flush=True)


if __name__ == "__main__":
    main()

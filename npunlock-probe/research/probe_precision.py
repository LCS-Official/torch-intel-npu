"""Inspect precision-preserving driver carriers; no custom hardware execution."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import npunlock as npu
from npunlock.npu4000 import Elf, inspect

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/precision"


def carrier(op="Abs", dtype="f32", shape=(1, 17), *, sensitive=True, accuracy=True):
    x = npu.input("x", shape=shape, dtype=dtype)
    inputs = [x]
    if op in ("Add", "Subtract", "Multiply", "Divide", "Power"):
        inputs.append(npu.input("b", shape=shape, dtype=dtype))
    y = npu.op(op, *inputs, _shape=shape, _dtype=dtype, _name="y")
    graph = npu.Graph(inputs, [y])
    ir = npu.serialize_ir(graph)
    xml = ET.fromstring(ir.xml)
    if sensitive:
        layer = xml.find("./layers/layer[@name='y']")
        info = ET.SubElement(layer, "rt_info")
        ET.SubElement(info, "attribute", name="DisablePrecisionConversion", version="0", value="dynamic:f16")
    native = npu.NativeLibraries()
    result = native.compile_ir(ET.tostring(xml), ir.weights,
        build_flags='--config EXECUTION_MODE_HINT="ACCURACY"' if accuracy else "", timeout_ms=20_000)
    return result, graph


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for op, dtype in [("Abs", "f32"), ("Add", "f32"), ("Multiply", "f32"), ("Power", "f16")]:
        name = f"{op}-{dtype}"
        try:
            result, graph = carrier(op, dtype)
            (OUT / f"{name}.blob").write_bytes(result.graph_blob)
            evidence = inspect(result.graph_blob)
            (OUT / f"{name}-inspection.json").write_text(json.dumps(evidence, indent=2))
            elf = Elf(result.graph_blob)
            import hashlib
            windows = set((t["text_relocation"]["symbol_value"] + t["text_relocation"]["addend"], t["code_size"])
                          for t in evidence["ranges"] if t["workload_type"] == 0)
            summary = dict(name=name, windows=[(offset, size, hashlib.sha256(elf.content(elf.named["shave.text"])[offset:offset+size]).hexdigest())
                                               for offset, size in windows],
                refs=[task.get("unary_memrefs", []) for task in evidence["invocations"]][:2])
            print(json.dumps(summary), flush=True)
        except Exception as error:
            print(json.dumps(dict(name=name, error=str(error))), flush=True)

"""Bounded probes for driver-generated cast/reduction/matmul contracts."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import npunlock as npu
from npunlock.npu4000 import inspect, Elf

OUT = Path(__file__).resolve().parent / "results/precision"
OUT.mkdir(exist_ok=True)


def probe(kind, dtype):
    x = npu.input("x", shape=(3, 5), dtype=dtype)
    args = [x]
    if kind == "Convert":
        target = "f32" if dtype == "f16" else "f16"
        y = npu.op(kind, x, destination_type=target, _shape=x.shape, _dtype=target, _name="y")
    elif kind in ("ReduceSum", "ReduceMean"):
        axes = npu.constant(np.array([1], dtype=np.int64))
        y = npu.op(kind, x, axes, keep_dims=False, _shape=(3,), _dtype=dtype, _name="y")
    else:
        b = npu.input("b", shape=(5, 4), dtype=dtype)
        args += [b]
        y = npu.op(kind, x, b, transpose_a=False, transpose_b=False,
                   _shape=(3, 4), _dtype=dtype, _name="y")
    graph = npu.Graph(args, [y])
    ir = npu.serialize_ir(graph)
    xml = ET.fromstring(ir.xml)
    info = ET.SubElement(xml.find("./layers/layer[@name='y']"), "rt_info")
    ET.SubElement(info, "attribute", name="DisablePrecisionConversion", version="0", value="dynamic:f16")
    native = npu.NativeLibraries()
    result = native.compile_ir(ET.tostring(xml), ir.weights,
        build_flags='--config EXECUTION_MODE_HINT="ACCURACY"', timeout_ms=20_000)
    stem = f"{kind}-{dtype}"
    (OUT / f"{stem}.blob").write_bytes(result.graph_blob)
    evidence = inspect(result.graph_blob)
    (OUT / f"{stem}-inspection.json").write_text(json.dumps(evidence, indent=2))
    summary = {"name": stem, "bytes": len(result.graph_blob),
        "ranges": len(evidence["ranges"]),
        "memref_dtypes": sorted(set(r["dtype"] for t in evidence["invocations"] for r in t.get("unary_memrefs", []))),
        "kernels": [(t["code_size"], t["text_relocation"]["addend"]) for t in evidence["ranges"] if t["workload_type"] == 0]}
    # FP32 includes values and differences that cannot survive FP16 conversion.
    values = np.array([[70000., 1.000244140625, -70000., 0.000244140625, 2.],
                       [1., 2., 3., 4., 5.], [1., -1., 1., -1., 1.]], dtype=np.float32)
    if dtype == "f16":
        values = np.arange(15, dtype=np.float16).reshape(3, 5) / 8
    inputs = [npu.InferenceInput("x", values.tobytes())]
    if kind == "MatMul":
        b = np.ones((5, 4), dtype=values.dtype)
        inputs += [npu.InferenceInput("b", b.tobytes())]
        expected = values @ b
    elif kind == "Convert":
        expected = values.astype(np.float32 if dtype == "f16" else np.float16)
    else:
        expected = values.sum(axis=1) if kind == "ReduceSum" else values.mean(axis=1)
    inferred = native.infer_graph(result.graph_blob, inputs, timeout_ms=10_000)
    output = inferred.outputs[0]
    actual = np.frombuffer(output.data, dtype=expected.dtype).reshape(output.shape)
    summary.update(actual=actual.tolist(), expected=expected.tolist(),
                   execution_device=hex(inferred.device_id))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert inferred.device_id == 0x643e
    summary["status"] = "passed"
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    for kind in ("Convert", "ReduceSum", "ReduceMean", "MatMul"):
        for dtype in ("f16", "f32"):
            try:
                probe(kind, dtype)
            except Exception as error:
                print(json.dumps({"name": f"{kind}-{dtype}", "error": str(error)}), flush=True)

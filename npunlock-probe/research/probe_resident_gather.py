"""Test exact DMA gather/zero padding needed by resident convolution."""
import json
from pathlib import Path
import numpy as np
import npunlock as npu
from npunlock.npu4000 import compile_gather
from torch_npu_runtime import SESSION_DLL

native = npu.NativeLibraries(session_library=SESSION_DLL)
out = Path(__file__).resolve().parent / 'results'
reports = []
for width in (2, 4):
    for elements, shape in ((36, (16, 9)), (72, (32, 18)), (512, (128, 27))):
        indices = np.random.default_rng(3).integers(-1, elements, shape, dtype=np.int64)
        entry = dict(width=width, elements=elements, shape=shape)
        try:
            dtype = 'f16' if width == 2 else 'f32'
            program = compile_gather(elements, indices, dtype=dtype, libraries=native)
            bits = np.random.default_rng(92).integers(0, 256, (elements, width), dtype=np.uint8)
            result = native.infer_graph(program.graph_blob, [npu.InferenceInput('x', bits.tobytes())], timeout_ms=10000)
            actual = np.frombuffer(result.outputs[0].data, dtype=np.uint8).reshape(*shape, width)
            expected = np.concatenate([bits, np.zeros((1, width), dtype=np.uint8)])[np.where(indices < 0, elements, indices)]
            assert np.array_equal(actual, expected)
            if elements == 36:
                (out / f'gather-{dtype}-36.blob').write_bytes(program.graph_blob)
            entry['status'] = 'passed'
        except Exception as error:
            entry.update(status='failed', error=str(error))
        reports.append(entry)
        (out / 'resident-gather-probe.json').write_text(json.dumps(reports, indent=2))
        print(entry, flush=True)
assert all(r['status'] == 'passed' for r in reports)

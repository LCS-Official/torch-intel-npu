"""Bounded driver/DMA probe before enabling transpose materialization."""
import json
from pathlib import Path
import numpy as np
import npunlock as npu
from npunlock.npu4000 import compile_permute
from torch_npu_runtime import SESSION_DLL

native = npu.NativeLibraries(session_library=SESSION_DLL)
out = Path(__file__).resolve().parent / 'results'
reports = []
for dtype in ('f16', 'f32'):
    for shape, order in (((3, 5), (1, 0)), ((256, 256), (1, 0)),
                         ((2, 3, 5), (2, 0, 1)), ((2, 3, 4, 5), (3, 1, 0, 2)),
                         ((1, 5), (1, 0))):
        entry = dict(dtype=dtype, shape=shape, order=order)
        try:
            program = compile_permute(shape, order, dtype=dtype, libraries=native)
            integer = np.uint16 if dtype == 'f16' else np.uint32
            bits = np.arange(np.prod(shape), dtype=integer)
            if dtype == 'f32':
                bits = np.random.default_rng(45).integers(0, 2**32, len(bits), dtype=integer)
                bits[:4] = [1, 0x7f800000, 0x7fc00001, 0x80000000]
            result = native.infer_graph(program.graph_blob, [npu.InferenceInput('x', bits.tobytes())], timeout_ms=10000)
            assert result.outputs[0].dtype == 'u8'
            actual = np.frombuffer(result.outputs[0].data, dtype=integer).reshape(tuple(shape[a] for a in order))
            assert np.array_equal(actual, bits.reshape(shape).transpose(order))
            if shape == (3, 5):
                (out / f'permute-{dtype}-3x5.blob').write_bytes(program.graph_blob)
            entry['status'] = 'passed'
        except Exception as error:
            entry.update(status='failed', error=str(error))
        reports.append(entry)
        (out / 'resident-permute-probe.json').write_text(json.dumps(reports, indent=2))
        print(entry, flush=True)
assert all(r['status'] == 'passed' for r in reports)

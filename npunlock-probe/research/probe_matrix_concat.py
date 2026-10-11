"""Verify exact DMA row assembly and the native 64-argument boundary."""
import json
from pathlib import Path
import numpy as np
import npunlock as npu
from npunlock.npu4000 import compile_matrix_concat
from torch_npu_runtime import SESSION_DLL

native = npu.NativeLibraries(session_library=SESSION_DLL)
output = Path(__file__).resolve().parent / 'results'
reports = []
for dtype, width in (('f16', 2), ('f32', 4)):
    for rows, columns in (((3, 2, 1), 5), ((64, 1), 32), ((1,)*63, 1)):
        program = compile_matrix_concat(rows, columns, dtype=dtype, libraries=native)
        values = [np.random.default_rng(i).integers(0, 256, (r, columns, width), dtype=np.uint8)
                  for i, r in enumerate(rows)]
        result = native.infer_graph(program.graph_blob, [npu.InferenceInput(f'x{i}', v.tobytes())
                                    for i, v in enumerate(values)], timeout_ms=10000)
        expected = np.concatenate(values)
        assert result.outputs[0].data == expected.tobytes()
        if rows == (3, 2, 1):
            (output / f'matrix-concat-{dtype}.blob').write_bytes(program.graph_blob)
        reports.append(dict(dtype=dtype, rows=rows, columns=columns, status='passed'))
        print(reports[-1], flush=True)
(output / 'matrix-concat-probe.json').write_text(json.dumps(reports, indent=2))

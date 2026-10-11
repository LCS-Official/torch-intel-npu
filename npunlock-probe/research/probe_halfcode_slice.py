"""Test NPU-only extraction of binary16 lanes from the existing cast carrier."""
import json
from pathlib import Path
import numpy as np
import npunlock as npu
from npunlock.npu4000 import compile_precision_graph, Elf
from torch_npu_runtime import SESSION_DLL

OUT = Path(__file__).resolve().parent / 'results'
native = npu.NativeLibraries(session_library=SESSION_DLL)
reports = []
for size in (1, 17, 65536):
    entry = dict(elements=size)
    try:
        x = npu.input('x', shape=(size, 2), dtype='f16')
        begin, end, strides = [npu.constant(np.array(v, dtype=np.int64)) for v in ([0, 0], [size, 1], [1, 1])]
        y = npu.op('StridedSlice', x, begin, end, strides,
                   begin_mask=[0, 0], end_mask=[0, 0], new_axis_mask=[0, 0],
                   shrink_axis_mask=[0, 0], ellipsis_mask=[0, 0],
                   _shape=(size, 1), _dtype='f16', _name='y')
        program = compile_precision_graph(npu.Graph([x], [y]), libraries=native)
        blob = program.graph_blob
        (OUT / f'halfcode-slice-{size}.blob').write_bytes(blob)
        elf = Elf(blob)
        dpu = [s['name'] for s in elf.sections if s['size'] and s['name'].startswith('task.dpu.')]
        assert not dpu, dpu
        codes = np.arange(size, dtype=np.uint32).astype(np.uint16)
        transport = codes.astype(np.uint32).view(np.float16).reshape(size, 2)
        result = native.infer_graph(blob, [npu.InferenceInput('x', transport.tobytes())], timeout_ms=10000)
        output = np.frombuffer(result.outputs[0].data, dtype=np.uint16)
        assert np.array_equal(output, codes)
        entry.update(status='passed', bytes=len(blob), dpu_tasks=dpu,
                     shave_sections=[s['name'] for s in elf.sections if s['name'].startswith('shave.')])
    except Exception as error:
        entry.update(status='failed', error=str(error))
    reports.append(entry)
    (OUT / 'halfcode-slice.json').write_text(json.dumps(reports, indent=2))
    print(entry, flush=True)

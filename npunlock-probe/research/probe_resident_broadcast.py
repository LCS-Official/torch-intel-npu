"""Check driver Broadcast is a bit-preserving DMA transport before using it."""
import json
from pathlib import Path
import numpy as np
import npunlock as npu
from npunlock.npu4000 import compile_broadcast
from torch_npu_runtime import SESSION_DLL

native = npu.NativeLibraries(session_library=SESSION_DLL)
OUT = Path(__file__).resolve().parent / 'results'
reports = []
for dtype in ('f16', 'f32'):
    for shape, target in (((1,), (3, 5)), ((1, 5), (3, 5)), ((3, 1), (3, 5)),
                          ((3, 5, 1), (3, 5, 4)), ((1, 5, 4), (3, 5, 4)),
                          ((1, 3, 1, 5), (2, 3, 4, 5)), ((1,), (65536,))):
        entry = dict(dtype=dtype, shape=shape, target=target)
        try:
            program = compile_broadcast(shape, target, dtype=dtype, libraries=native)
            bits = np.arange(np.prod(shape), dtype=np.uint32)
            if dtype == 'f32':
                bits[:min(4, len(bits))] = np.array([1, 0x7f800000, 0x7fc00001, 0x80000000], dtype=np.uint32)[:min(4, len(bits))]
                values = bits.view(np.float32).reshape(shape)
                integer = np.uint32
            else:
                values = bits.astype(np.uint16).view(np.float16).reshape(shape)
                integer = np.uint16
            result = native.infer_graph(program.graph_blob, [npu.InferenceInput('x', values.tobytes())], timeout_ms=10000)
            output = np.frombuffer(result.outputs[0].data, dtype=integer).reshape(target)
            assert np.array_equal(output, np.broadcast_to(values.view(integer), target))
            if shape == (1, 5):
                (OUT / f'broadcast-{dtype}-1x5-to-3x5.blob').write_bytes(program.graph_blob)
            entry['status'] = 'passed'
        except Exception as error:
            entry.update(status='failed', error=str(error))
        reports.append(entry)
        (OUT / 'resident-broadcast-probe.json').write_text(json.dumps(reports, indent=2))
        print(entry, flush=True)

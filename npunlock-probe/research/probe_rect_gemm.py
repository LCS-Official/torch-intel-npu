"""Compile packed rectangular carriers; inspect and reject before execution."""
import json
from pathlib import Path
import npunlock as npu
from npunlock.npu4000 import compile_precision_graph, _patch_four_tile_buffers, rect_gemm_capacity
from torch_npu_runtime import SESSION_DLL

OUT = Path(__file__).resolve().parent / 'results/gemm-rect'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    native = npu.NativeLibraries(session_library=SESSION_DLL)
    kernel = Path('npunlock/tests/fixtures/npu4000/gemm_parallel/square-f32-four-tiles.elf').read_bytes()
    records = []
    for shape in ((3, 5, 7), (16, 48, 24), (64, 128, 64), (63, 256, 64)):
        count = rect_gemm_capacity(shape)
        x = npu.input('x', shape=(4, 2, count//2), dtype='f32')
        b = npu.input('b', shape=x.shape, dtype='f32')
        y = npu.op('Power', x, b, _shape=x.shape, _dtype='f32', _name='y')
        program = compile_precision_graph(npu.Graph([x, b], [y], name='rect_probe'), libraries=native)
        (OUT/f'capacity-{count}.blob').write_bytes(program.graph_blob)
        # Reuse an existing ELF only for offline byte-window validation, never infer.
        _, report = _patch_four_tile_buffers(program.graph_blob, kernel, count=count,
            vendor_id=program.ir_provenance.vendor_id, device_id=program.ir_provenance.device_id,
            compiler_version=program.ir_provenance.compiler_version)
        records.append(dict(shape=shape, capacity=count, report=report))
        print('validated', shape, count, flush=True)
    (OUT/'carrier-validation.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()

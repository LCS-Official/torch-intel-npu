"""Inspect compiler task layouts only; never run a speculative carrier."""
import json
from pathlib import Path
import npunlock as npu
from npunlock.npu4000 import compile_precision_graph, inspect, _gemm_dma, Elf
from torch_npu_runtime import SESSION_DLL


def main():
    out = Path(__file__).resolve().parent/'results/gemm-eight'
    out.mkdir(parents=True, exist_ok=True)
    native = npu.NativeLibraries(session_library=SESSION_DLL)
    records = []
    for count in (1152, 8256, 16384):
        for rows in (8,):
            name = f'capacity-{count}-replicas-8'
            shape = (8, rows, count//rows)
            x = npu.input('x', shape=shape, dtype='f32')
            b = npu.input('b', shape=shape, dtype='f32')
            y = npu.op('Power', x, b, _shape=shape, _dtype='f32', _name='y')
            p = compile_precision_graph(npu.Graph([x,b],[y], name='eight_probe'), libraries=native)
            e = inspect(p.graph_blob, memref_count=3)
            tasks = [(t['tile'], [(r['dims'], r['strides_bits'], r['unrelocated_data']) for r in t['memrefs']])
                     for t in e['invocations']]
            (out/f'{name}.blob').write_bytes(p.graph_blob)
            (out/f'{name}.json').write_text(json.dumps(e, indent=2))
            record = dict(name=name, shape=shape, tasks=tasks)
            records.append(record); print(record, flush=True)
    (out/'layout-summary.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()

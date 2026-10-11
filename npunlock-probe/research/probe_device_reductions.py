"""Inspect the driver carriers before choosing resident cast/reduction kernels."""
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import npunlock as npu
from npunlock.npu4000 import compile_precision_graph, inspect, Elf
from torch_npu_runtime import SESSION_DLL

OUT = Path(__file__).resolve().parent / 'results'
native = npu.NativeLibraries(session_library=SESSION_DLL)
reports = []
for kind in ('Convert', 'ReduceSum', 'ReduceMean'):
    for shape in ((1, 17), (3, 5), (1, 65536)):
        name = kind.lower() + '-' + 'x'.join(map(str, shape))
        entry = dict(kind=kind, shape=shape)
        try:
            x = npu.input('x', shape=shape, dtype='f32')
            if kind == 'Convert':
                y = npu.op(kind, x, destination_type='f16', _shape=shape, _dtype='f16', _name='y')
            else:
                axes = npu.constant(np.array([1], dtype=np.int64))
                y = npu.op(kind, x, axes, keep_dims=True, _shape=(shape[0], 1), _dtype='f32', _name='y')
            program = compile_precision_graph(npu.Graph([x], [y]), libraries=native)
            blob = program.graph_blob
            (OUT / (name + '.blob')).write_bytes(blob)
            elf, evidence = Elf(blob), inspect(blob)
            entry.update(invocations=len(evidence['invocations']),
                         refs=[t.get('memrefs') for t in evidence['invocations']],
                         ranges=evidence['ranges'])
            text = elf.content(elf.named['shave.text']) if 'shave.text' in elf.named else b''
            entry['code_hashes'] = []
            for task in evidence['ranges']:
                reloc = task.get('text_relocation')
                if reloc:
                    offset = reloc['symbol_value'] + reloc['addend']
                    entry['code_hashes'].append(hashlib.sha256(text[offset:offset+task['code_size']]).hexdigest())
            if shape[1] < 100:
                values = np.ones(shape, dtype=np.float32)
                values[:, :5] = [70000, 1.0001, 2.**-149, -0., 2.**-24]
                result = native.infer_graph(blob, [npu.InferenceInput('x', values.tobytes())], timeout_ms=10000)
                dtype = np.float16 if kind == 'Convert' else np.float32
                output = np.frombuffer(result.outputs[0].data, dtype=dtype)
                entry.update(output=output.tolist(), output_bits=output.view(np.uint16 if kind == 'Convert' else np.uint32).tolist())
                entry['edges'] = []
                for v in (2.**-149, 2.**-24, -0.):
                    values.fill(v)
                    result = native.infer_graph(blob, [npu.InferenceInput('x', values.tobytes())], timeout_ms=10000)
                    output = np.frombuffer(result.outputs[0].data, dtype=dtype)
                    entry['edges'].append(dict(value=v, output_bits=output.view(np.uint16 if kind == 'Convert' else np.uint32).tolist()))
            entry['status'] = 'compiled'
        except Exception as error:
            entry.update(status='failed', error=str(error), traceback=traceback.format_exc())
        reports.append(entry)
        (OUT / 'device-carriers.json').write_text(json.dumps(reports, indent=2))
        print(name, entry['status'], entry.get('error', ''), entry.get('code_hashes'), entry.get('output'), flush=True)

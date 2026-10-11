"""Inspect OpenVINO exports with the existing bounded NPU40XX ELF parser."""
import hashlib
import json
from pathlib import Path

from npunlock.npu4000 import Elf
from torch_npu_lowbit import dpu_modes

OUT = Path(__file__).resolve().parent/'results/dpu-diagnostic'
rows = []
for path in sorted(path for prefix in ('openvino-','paired-','tiles-') for path in OUT.glob(prefix+'*.blob')):
    blob = path.read_bytes()
    modes = dpu_modes(blob)
    matrices = [value for value in modes if not value['elementwise']]
    assert matrices and all((value['amode'],value['wmode']) == (1,1) for value in matrices)
    elf = Elf(blob)
    row = dict(file=path.name,sha256=hashlib.sha256(blob).hexdigest(),
               matrix_modes=sorted(set((value['amode'],value['wmode']) for value in matrices)),
               invariants=len(modes),sections={prefix:[dict(name=s['name'],bytes=s['size']) for s in elf.sections
               if s['size'] and s['name'].startswith(prefix)] for prefix in
               ('task.dpu.invariant.','task.dpu.variant.','task.shave.invocation.','task.dma.')})
    rows.append(row)
    print(path.name, row['matrix_modes'],{key:sum(v['bytes'] for v in values) for key,values in row['sections'].items()})
(OUT/'openvino-graphs.json').write_text(json.dumps(dict(
    inspector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    abi_scope='Existing NPU40XX 352-byte invariant layout; matrix dtype fields only, no descriptor modification',rows=rows),indent=2),encoding='utf-8')

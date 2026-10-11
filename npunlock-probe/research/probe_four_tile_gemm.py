import sys, json
sys.path.insert(0,'npunlock/python')
import npunlock as npu
from npunlock.npu4000 import compile_precision_graph,inspect,Elf
from torch_npu_runtime import SESSION_DLL
from pathlib import Path
native=npu.NativeLibraries(session_library=SESSION_DLL); out=Path('npunlock-probe/research/results/gemm-parallel')
out.mkdir(parents=True, exist_ok=True)
for size in (8,16,32,64):
    shape=(4,2,size*size//2); a=npu.input('x',shape=shape,dtype='f32'); b=npu.input('b',shape=shape,dtype='f32'); c=npu.op('Power',a,b,_shape=shape,_dtype='f32',_name='y')
    program=compile_precision_graph(npu.Graph([a,b],[c],name='gemm_four_tiles'),libraries=native)
    blob=program.graph_blob; evidence=inspect(blob,memref_count=3)
    (out/f'four-tiles-{size}.blob').write_bytes(blob); (out/f'four-tiles-{size}.json').write_text(json.dumps(evidence,indent=2))
    print(size,[(t['tile'],[(r['dims'],r['unrelocated_data']) for r in t['memrefs']]) for t in evidence['invocations']])

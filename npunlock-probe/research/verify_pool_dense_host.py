"""CPU oracle for the unchanged pool C body; only entry/MemRef pointers are widened.

The temporary DLL is built from resident_pool_dense.c using the existing MSVC
toolchain. This verifies source logic, not SHAVE scheduling or NPU performance.
"""
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import random
import sys


def emit_source(destination):
    root=Path('npunlock/tests/fixtures/npu4000')
    comparison=(root/'resident_selection.c').read_text(encoding='utf-8').split('void controlled_act(unsigned p) {',1)[0]
    body=(root/'resident_pool_dense.c').read_text(encoding='utf-8').replace(
        'void controlled_act(unsigned p)','__declspec(dllexport) void controlled_act(uintptr_t p)')
    abi='#include <stdint.h>\ntypedef struct { uintptr_t lo; unsigned hi,isStatic,rank; uintptr_t dims,strides; unsigned dtype,orderLo,orderHi,location; } MemRef;\n'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(abi+comparison+body,encoding='utf-8')


def main(dll):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch_npu_geometry import max_pool2d_geometry,max_pool_windows,pool_dense_tiles
    torch.set_num_threads(1);torch.manual_seed(838);rng=random.Random(838)
    class Ref(c.Structure):
        _fields_=[('lo',c.c_size_t),('hi',c.c_uint),('static',c.c_uint),('rank',c.c_uint),
            ('dims',c.c_size_t),('strides',c.c_size_t),('dtype',c.c_uint),('orderLo',c.c_uint),('orderHi',c.c_uint),('location',c.c_uint)]
    native=c.CDLL(str(dll.resolve()));native.controlled_act.argtypes=[c.c_size_t]
    def run(fields,source):
        packet=np.zeros(32768,dtype=np.uint32);packet[:16]=fields;packet[16:16+len(source)]=source
        output=np.zeros_like(packet);dims=(c.c_uint*4)(1,1,1,len(packet));refs=(Ref*3)()
        for ref,address in zip(refs,(packet.ctypes.data,packet.ctypes.data,output.ctypes.data)):
            ref.lo=address;ref.static=1;ref.rank=4;ref.dims=c.addressof(dims);ref.dtype=1
        native.controlled_act(c.addressof(refs));assert output[-1]==0,fields
        count=fields[0]
        return torch.from_numpy(output[:count].copy().view(np.float32)),torch.from_numpy(output[count:3*count].copy().view(np.int64))
    packets=cases=0
    geometries=[((257,259),(3,3),(1,1),(1,1),(1,1),False),((7,11),(2,3),(2,1),(1,1),(2,2),True)]
    for _ in range(250):
        spatial=(rng.randint(1,39),rng.randint(1,43));kernel=(rng.randint(1,3),rng.randint(1,3))
        geometries.append((spatial,kernel,(rng.randint(1,4),rng.randint(1,4)),tuple(rng.randint(0,k//2) for k in kernel),
            (rng.randint(1,3),rng.randint(1,3)),bool(rng.randrange(2))))
    for spatial,kernel,stride,pad,dilation,ceil in geometries:
        try:params=max_pool2d_geometry(spatial,kernel,stride,pad,dilation,ceil)
        except ValueError:continue
        for dtype in (torch.float32,torch.float16):
            host=(torch.randn(3,1,*spatial)/8).to(dtype);flat=host.view(-1)
            flat[::17]=-0.;flat[::31]=float('nan');flat[::43]=float('inf')
            expected,ids=F.max_pool2d(host,kernel,stride,pad,dilation,ceil,return_indices=True)
            words=host.float().view(torch.int32).numpy().view(np.uint32).reshape(-1)
            for start,count,base,span in pool_dense_tiles(spatial,params,3,4096//dtype.itemsize):
                assert base is not None
                values,index=run((count,start,*spatial,*params[-1],*kernel,*stride,*pad,*dilation,base,span),words[base:base+span])
                torch.testing.assert_close(values.to(dtype),expected.view(-1)[start:start+count],rtol=0,atol=0,equal_nan=True)
                assert torch.equal(index,ids.view(-1)[start:start+count]);packets+=1
            cases+=1
    width=536870912;outputs=134217728;start=outputs-2;base=4*start
    source=np.array([.125,0,0,0,.25],dtype=np.float32).view(np.uint32)
    value,index=run((2,start,1,width,1,outputs,1,1,1,4,0,0,1,1,base,5),source)
    assert torch.equal(index,torch.tensor([base,base+4]))
    assert torch.equal(value,torch.tensor([.125,.25]));packets+=1
    result=dict(status='passed',geometry_dtype_cases=cases,packets=packets,maximum_uint32_index=int(index[-1]),
        source_sha256=hashlib.sha256(Path('npunlock/tests/fixtures/npu4000/resident_pool_dense.c').read_bytes()).hexdigest())
    target=Path('npunlock-probe/research/results/storage-shape');target.mkdir(exist_ok=True)
    (target/'host-c.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dll',type=Path,default=Path('npunlock-probe/research/diagnostics/pool_host.dll'))
    parser.add_argument('--emit-source',type=Path)
    args=parser.parse_args()
    if args.emit_source:emit_source(args.emit_source)
    else:main(args.dll)

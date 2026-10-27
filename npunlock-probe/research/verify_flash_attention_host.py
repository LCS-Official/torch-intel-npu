"""CPU oracle for the actual C attention body, widening only entry/MemRef pointers."""
import argparse
import ctypes as c
import json
from pathlib import Path
import struct


def emit_source(destination):
    root=Path('npunlock/tests/fixtures/npu4000')
    helpers=''.join((root/name).read_text().split('void controlled_act(unsigned p) {',1)[0]
                    for name in ('arithmetic.c','resident_classification.c'))
    helpers=helpers[helpers.index('typedef struct'):]
    helpers=helpers[helpers.index('} MemRef;')+len('} MemRef;'):]
    prefix,body=(root/'resident_flash_attention.c').read_text().split('void controlled_act(unsigned p)',1)
    abi='#include <stdint.h>\ntypedef struct { uintptr_t lo; unsigned hi,isStatic,rank; uintptr_t dims,strides; unsigned dtype,orderLo,orderHi,location; } MemRef;\n'
    # MSVC shares the exact arithmetic body; only the GNU inline spelling differs.
    source=(abi+helpers+prefix).replace('__attribute__((always_inline)) ', '')
    for mode,name in ((0,'flash_forward'),(1,'flash_backward')):
        source+=f'\n#undef FLASH_BACKWARD\n#define FLASH_BACKWARD {mode}\n__declspec(dllexport) void {name}(uintptr_t p)'+body
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(source,encoding='utf-8')


def verify(path):
    import numpy as np
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    torch.set_num_threads(1);torch.manual_seed(422)
    class Ref(c.Structure):
        _fields_=[('lo',c.c_size_t),('hi',c.c_uint),('static',c.c_uint),('rank',c.c_uint),
            ('dims',c.c_size_t),('strides',c.c_size_t),('dtype',c.c_uint),('orderLo',c.c_uint),('orderHi',c.c_uint),('location',c.c_uint)]
    dll=c.CDLL(str(path.resolve()))
    for name in ('flash_forward','flash_backward'):getattr(dll,name).argtypes=[c.c_size_t]
    packets=0
    def run(q,k,v,state,mask,qs,ks,causal,scale,first,go=None):
        nonlocal packets
        nr,d=q.shape;ns,dv=v.shape;root=struct.unpack('<I',struct.pack('<f',abs(scale)**.5))[0]
        count=nr*(dv+2) if go is None else q.numel()+k.numel()+v.numel()
        header=(nr,ns,d,dv,qs,ks,int(causal),root,int(scale<0),0 if mask is None else 2 if mask.dtype==torch.bool else 1,int(first),0,0,0,0,count*4+4)
        values=[q,k,v,state]+([] if go is None else [go])+([] if mask is None else [mask])
        payload=b''.join(t.detach().contiguous().numpy().tobytes() for t in values)
        size=(max(64+len(payload),count*4+4)+7)//8*8
        packet=np.zeros(size//4,dtype=np.uint32);raw=struct.pack('<16I',*header)+payload
        packet.view(np.uint8)[:len(raw)]=np.frombuffer(raw,dtype=np.uint8)
        output=np.zeros_like(packet);dims=(c.c_uint*4)(1,1,1,len(packet));refs=(Ref*3)()
        for ref,ptr in zip(refs,(packet.ctypes.data,packet.ctypes.data,output.ctypes.data)):
            ref.lo=ptr;ref.static=1;ref.rank=4;ref.dims=c.addressof(dims);ref.dtype=1
        getattr(dll,'flash_forward' if go is None else 'flash_backward')(c.addressof(refs));packets+=1
        return torch.from_numpy(output[:count].copy().view(np.float32))
    cases=0
    for causal in (False,True):
        for scale in (-.7,0.,.4,1.25):
            for kind in ('none','bool','float') if not causal else ('none',):
                q,k,v=[torch.randn(s)/4 for s in ((7,3),(11,3),(11,5))]
                mask=None if kind=='none' else torch.rand(7,11)>.3 if kind=='bool' else torch.randn(7,11)/8
                if mask is not None:mask[2]=False if kind=='bool' else float('-inf')
                expected_inputs=[t.double().requires_grad_() for t in (q,k,v)]
                with sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.scaled_dot_product_attention(*expected_inputs,attn_mask=mask if kind=='bool' or mask is None else mask.double(),is_causal=causal,scale=scale)
                states=[];outputs=[]
                for qs in range(0,7,3):
                    qt=q[qs:qs+3];state=torch.zeros(len(qt),7)
                    for ks in range(0,11,4):
                        mt=None if mask is None else mask[qs:qs+len(qt),ks:ks+4]
                        state=run(qt,k[ks:ks+4],v[ks:ks+4],state,mt,qs,ks,causal,scale,ks==0).view(len(qt),7)
                    states.append(state);outputs.append(state[:,2:]/state[:,1:2].masked_fill(state[:,1:2]==0,1))
                actual=torch.cat(outputs)
                torch.testing.assert_close(actual,expected.float(),rtol=3e-5,atol=3e-6)
                seed=torch.randn_like(actual);expected.backward(seed.double());grads=[torch.zeros_like(t) for t in (q,k,v)]
                for state,qs in zip(states,range(0,7,3)):
                    qt=q[qs:qs+3];nr=len(qt)
                    for ks in range(0,11,4):
                        kt,vt=k[ks:ks+4],v[ks:ks+4];nk=len(kt);mt=None if mask is None else mask[qs:qs+nr,ks:ks+nk]
                        result=run(qt,kt,vt,state,mt,qs,ks,causal,scale,False,seed[qs:qs+nr]);a=qt.numel();b=kt.numel()
                        grads[0][qs:qs+nr]+=result[:a].view_as(qt)
                        grads[1][ks:ks+nk]+=result[a:a+b].view_as(kt)
                        grads[2][ks:ks+nk]+=result[a+b:].view_as(vt)
                for a,b in zip(grads,expected_inputs):torch.testing.assert_close(a,b.grad.float(),rtol=5e-5,atol=4e-6)
                cases+=1
    result=dict(status='passed',cases=cases,packets=packets,reference='independent CPU FP64 SDPA forward and autograd')
    out=Path('npunlock-probe/research/results/flash-nested/host-c.json');out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--emit-source',type=Path);parser.add_argument('--dll',type=Path)
    args=parser.parse_args()
    if args.emit_source:emit_source(args.emit_source)
    else:verify(args.dll)

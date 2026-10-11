"""Compare cached binding graphs with full graphs across boundary shapes."""
import argparse
import gc
import json
import os
from pathlib import Path
from types import MethodType
import numpy as np
import torch
import torch_intel_npu as npu
import torch_npu_smollm as smol
import torch_npu_graph_cache as cache

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/startup-ir-cache'


def main(phase):
    os.environ['NPUNLOCK_GRAPH_CACHE']=str(OUT/'shape-cache')
    torch.set_num_threads(8);torch.manual_seed(9)
    checks=[]
    def arrays(output,state):
        return [t.cpu().numpy().copy() for t in (output,state.keys,state.values,state._position)]
    def equal(left,right):
        for a,b in zip(left,right):np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8))
    for shape in ((1,1,576),(1,24,576),(4,1,576),(8,64,576)):
        left=smol.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16')
        right=smol.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16')
        original=smol._PREFILL_GRAPH
        left.graph=MethodType(lambda self,*a,**kw:original(self,*a,**kw),left)
        with torch.no_grad():
            x=(torch.randn(shape)*.1).half().to('intel_npu')
            a,ca=left.prefill(x);b,cb=right.prefill(x)
            old_a=arrays(a,ca);old_b=arrays(b,cb);equal(old_a,old_b)
            assert ca.length==cb.length==shape[1]
            checks.append(dict(shape=shape,operation='prefill',bit_equal_arrays=4))
            if shape==(1,24,576):
                clones=[smol.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16') for _ in range(2)]
                clones[0].graph=MethodType(lambda self,*a,**kw:original(self,*a,**kw),clones[0])
                for clone in clones:
                    for name,scale in [('input_layernorm.weight',.5),('post_attention_layernorm.weight',.75)]:
                        data,wshape=clone._weights[name]
                        clone._weights[name]=((np.frombuffer(data,dtype=np.float16)*np.float16(scale)).tobytes(),wshape)
                before=npu.stats()['session_creations']
                ga,gca=clones[0].prefill(x);gb,gcb=clones[1].prefill(x)
                equal(arrays(ga,gca),arrays(gb,gcb))
                assert npu.stats()['session_creations']==before and not np.array_equal(ga.cpu().numpy(),old_a[0])
                checks.append(dict(shape=shape,operation='different_gamma_shared_graph',bit_equal_arrays=4,new_sessions=0))
                del clones,clone,ga,gb,gca,gcb
            if shape[1]<64:
                one=(torch.randn(shape[0],1,576)*.1).half().to('intel_npu')
                da,dca=left.decode(one,ca);db,dcb=right.decode(one,cb)
                equal(arrays(da,dca),arrays(db,dcb));equal(old_a,arrays(a,ca));equal(old_b,arrays(b,cb))
                assert dca.length==dcb.length==shape[1]+1
                checks.append(dict(shape=shape,operation='decode',bit_equal_arrays=4,original_cache_unchanged=True))
                del one,da,db,dca,dcb
            del x,a,b,ca,cb
        del left,right;gc.collect()
    counts=cache.stats()
    assert counts['compiles']==(10 if phase=='fill' else 0)
    assert counts['ir_skipped']==(0 if phase=='fill' else 4)
    assert counts['hits']==(0 if phase=='fill' else 10) and counts['rejected']==0
    npu.shutdown();final=npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
    (OUT/('shape-'+phase+'.json')).write_text(json.dumps(dict(status='passed',checks=checks,cache=counts,final_stats=final),indent=2))
    print(phase,len(checks),'hardware shape cases passed; cache',counts,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('fill','load'),required=True)
    main(parser.parse_args().phase)

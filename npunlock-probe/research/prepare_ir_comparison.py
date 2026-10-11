"""Reuse verified identical blobs for paired full-IR/early-lookup benchmarks."""
import hashlib
import json
import os
from pathlib import Path
import zipfile
import torch_npu_graph_cache as cache
import torch_npu_smollm as smol
from npunlock.npu4000 import compile_fused_rmsnorm

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/startup-ir-cache'


def main():
    directory=OUT/'candidate-cache';os.environ['NPUNLOCK_GRAPH_CACHE']=str(directory)
    prior=set()
    for path in (OUT/'baseline-cache').glob('*.npc'):
        with zipfile.ZipFile(path) as archive:prior.add(hashlib.sha256(archive.read('graph.blob')).hexdigest())
    if prior:assert len(prior)==30
    with zipfile.ZipFile(next(directory.glob('*.npc'))) as archive:profile=json.loads(archive.read('manifest.json'))['profile']
    stack=smol.FrozenSmolStack.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',num_layers=30,mode='fp16')
    shape=(1,24,576);kernel=smol._rmsnorm_kernel();rows=[]
    for i,layer in enumerate(stack.layers):
        stub,recipe=cache.prefill_cache_graph(layer,shape,kernel,compile_fused_rmsnorm)
        fast=cache._key(recipe,profile,True)
        blob,reports=cache._read(directory/(fast+'.npc'),fast,profile,contract=cache._contract(stub))
        blob_sha=hashlib.sha256(blob).hexdigest()
        if prior:assert blob_sha in prior
        graph,xml,weights=layer.graph(shape,segment='layer',kv_cache='prefill',fused_norm=True)
        assert cache._contract(graph)==cache._contract(stub)
        digest=hashlib.sha256(xml+weights+kernel).digest();legacy=cache._key(digest,profile,True)
        assert cache._write(OUT/'paired-full-ir-cache',legacy,profile,blob,reports)
        assert cache._read(OUT/'paired-full-ir-cache'/(legacy+'.npc'),legacy,profile)==(blob,reports)
        rows.append(dict(layer=i,fast_key=fast,full_ir_key=legacy,blob_sha256=blob_sha))
    assert len({r['blob_sha256'] for r in rows})==30
    (OUT/'paired-cache-map.json').write_text(json.dumps(rows,indent=2))
    print('30 identical compiled blobs, legacy and early keys verified',flush=True)


if __name__=='__main__':main()

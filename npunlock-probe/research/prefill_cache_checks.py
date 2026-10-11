"""No-device checks for early cache identity and lazy graph construction."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import torch_npu_graph_cache as cache
import torch_npu_smollm as smol
from npunlock.npu4000 import compile_fused_rmsnorm

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/startup-ir-cache'


def main():
    checks=[]
    layer=smol.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16')
    shape=(1,24,576);kernel=smol._rmsnorm_kernel()
    with tempfile.TemporaryDirectory(prefix='prefill-cache-',dir=OUT) as directory,patch.dict(os.environ,NPUNLOCK_GRAPH_CACHE=directory):
        def identity(shape=shape,kernel=kernel):
            return cache.prefill_cache_graph(layer,shape,kernel,compile_fused_rmsnorm)
        graph,digest=identity()
        for actual_shape in ((1,24,576),(4,1,576),(8,64,576)):
            stub,_=identity(actual_shape)
            actual,_,_=layer.graph(actual_shape,segment='layer',kv_cache='prefill',fused_norm=True)
            assert cache._contract(stub)==cache._contract(actual)
            checks.append('real graph contract '+str(actual_shape))
        assert identity()[1]==digest;checks.append('stable recipe')
        assert identity((1,25,576))[1]!=digest;checks.append('shape invalidation')
        assert identity(kernel=kernel+b'changed')[1]!=digest;checks.append('kernel invalidation')
        name='self_attn.q_proj.weight';data,wshape=layer._weights[name]
        layer._weights[name]=(bytes([data[0]^1])+data[1:],wshape)
        assert identity()[1]!=digest;layer._weights[name]=(data,wshape);checks.append('projection invalidation')
        for name in ('_cos','_sin'):
            data=getattr(layer,name);setattr(layer,name,bytes([data[0]^1])+data[1:])
            assert identity()[1]!=digest;setattr(layer,name,data);checks.append(name+' invalidation')
        name='input_layernorm.weight';data,wshape=layer._weights[name]
        layer._weights[name]=(b'\0'*len(data),wshape)
        assert identity()[1]==digest;layer._weights[name]=(data,wshape);checks.append('dynamic gamma excluded')
        with patch.dict(os.environ,NPUNLOCK_GRAPH_CACHE=''):assert identity() is None
        checks.append('disabled cache unchanged')
        with patch.object(layer,'_mode','w8a16'):assert identity() is None
        checks.append('quantized path unchanged')
        assert cache.prefill_cache_graph(layer,shape,kernel,lambda:None) is None
        checks.append('custom compiler bypass')
        original=smol.FrozenSmolLayer.graph
        with patch.object(smol.FrozenSmolLayer,'graph',lambda *a,**kw:original(*a,**kw)):assert identity() is None
        checks.append('custom graph bypass')
        with patch.object(smol.s,'serialize_ir',lambda *a,**kw:None):assert identity() is None
        checks.append('custom serializer bypass')
        calls=[]
        @dataclass
        class Program:
            graph:object
            graph_blob:bytes=b'test-compiled-blob'
            patch_reports:tuple=(b'{"operation":"test"}',)
        def compiler(graph,*args,**kwargs):calls.append('compile');return Program(graph)
        def loader(blob,*,graph,libraries):calls.append('load');return Program(graph,blob)
        fake=SimpleNamespace(_native=object(),_driver_version=1005540,
            _info=SimpleNamespace(device_vendor_id=0x8086,device_id=0x643e),
            _bootstrap_program=SimpleNamespace(ir_provenance=SimpleNamespace(compiler_version=(8,3),graph_extension_version=1)))
        def factory():calls.append('build');return graph,b'xml',b'weights'
        def run(factory=factory):
            return cache.compile_fused_cached(fake,graph,None,None,kernel,graph_digest=digest,whole_layer=True,
                select_kernel_elf=None,compiler_factory=compiler,graph_factory=factory)
        with patch('npunlock.npu4000.compile_fused_rmsnorm',compiler),patch('npunlock.load_native',loader):
            result=run();assert calls==['build','compile'];checks.append('miss builds before compile')
            calls.clear();assert run().graph_blob==result.graph_blob and calls==['load']
            checks.append('hit skips graph factory')
            entry=next(Path(directory).glob('*.npc'))
            with zipfile.ZipFile(entry) as archive:
                manifest=json.loads(archive.read('manifest.json'));blob=archive.read('graph.blob')
            manifest['io_contract']['outputs'][0]['shape']=[1,1,576]
            with zipfile.ZipFile(entry,'w') as archive:
                archive.writestr('manifest.json',json.dumps(manifest));archive.writestr('graph.blob',blob)
            calls.clear();run();assert calls==['build','compile'];checks.append('contract corruption rebuilds')
            entry.unlink();calls.clear()
            other,_,_=layer.graph((1,25,576),segment='layer',kv_cache='prefill',fused_norm=True)
            try:run(lambda:(other,b'xml',b'weights'))
            except ValueError as error:assert 'contract' in str(error)
            else:raise AssertionError('mismatched factory accepted')
            assert not calls and not list(Path(directory).glob('*.npc'));checks.append('factory contract mismatch rejects before compile')
    report=dict(status='passed',checks=checks,sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(cache.__file__),Path(smol.__file__),Path(__file__))})
    (OUT/'prefill-checks.json').write_text(json.dumps(report,indent=2))
    print(len(checks),'early prefill cache checks passed',flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);main()

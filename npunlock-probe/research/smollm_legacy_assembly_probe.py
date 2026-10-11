"""Check that the historical assembly control uses compatible saved graphs."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-kv-batch'
PHASE=OUT.parent/'smollm-kv-profile'


def main():
    import torch
    import torch_intel_npu as npu
    sources={};modules=[]
    for name in ['before','after']:
        path=PHASE/name/'torch_npu_smollm.py';sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        spec=importlib.util.spec_from_file_location('legacy_assembly_'+name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module;spec.loader.exec_module(module);module.__file__=str(ROOT/'torch_npu_smollm.py');modules.append(module)
    previous,current=modules
    def work():
        torch.set_num_threads(8);torch.manual_seed(1091)
        model=current.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='w8a16')
        host=torch.randn(2,4,576).half()*.1;_,state=model.prefill(host[:,:3].contiguous().to('intel_npu'))
        one=host[:,-1:].contiguous().to('intel_npu');outputs=[];calls=[]
        for method in [previous.FrozenSmolLayer._run_cached_attention,current.FrozenSmolLayer._run_cached_attention]:
            with patch.object(current.FrozenSmolLayer,'_run_cached_attention',method):
                model.decode(one,state);before=npu.stats();value,cache=model.decode(one,state);after=npu.stats()
            calls.append(after['npu_calls']-before['npu_calls']);outputs.append([v.cpu() for v in [value,cache.keys,cache.values]])
            for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
        assert calls==[8,6],calls
        assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(*outputs))
        return dict(batch=2,graph_calls=calls,all_bits_equal=True)
    with torch.no_grad():checks=[work()]
    gc.collect();assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
    npu.shutdown();assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
    sources['npunlock-probe/research/smollm_legacy_assembly_probe.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'legacy.json').write_text(json.dumps(dict(status='passed',checks=checks,sources=sources,final_stats=npu.stats()),indent=2),encoding='utf-8')
    print(checks,flush=True)


if __name__=='__main__':main()

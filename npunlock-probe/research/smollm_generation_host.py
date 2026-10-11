"""Attribute warm decode cost at the Python/native boundary; diagnostic only."""
import cProfile
import json
from pathlib import Path
import pstats
import time
import numpy as np
from decode_chain_api import arrays, equal, delta, provenance
from w8_quality_probe import ROOT, MODEL, sha
from shave_fp32_pipeline import final
from smollm_generation_profile import OUT


def main():
    import torch
    import torch_intel_npu as npu
    from safetensors import safe_open
    from torch_npu_smollm import FrozenSmolStack
    torch.set_num_threads(8)
    report=dict(status='running',sources=provenance(),source_sha256=sha(Path(__file__)),repetitions=30)
    def work():
        original=json.loads((OUT/'fp16.json').read_text())
        assert original['status']=='passed'
        with np.load(OUT/'fp16-outputs.npz',allow_pickle=False) as saved:tokens=torch.from_numpy(saved['tokens'].copy())
        with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as file:
            embedding=file.get_tensor('model.embed_tokens.weight').half()
        length=original['input_tokens'];stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=30,mode='fp16')
        _,seed=stack.prefill(embedding[tokens[:,:length]].contiguous().to('intel_npu'))
        one=embedding[tokens[:,length:length+1]].contiguous().to('intel_npu')
        expected=stack.decode(one,seed,group_size=1);anchor=arrays(expected);del expected
        for _ in range(3):value=stack.decode(one,seed);del value
        profiler=cProfile.Profile()
        for enabled in (False,True):
            before=npu.stats();start=time.perf_counter_ns()
            if enabled:profiler.enable()
            for _ in range(30):value=stack.decode(one,seed);del value
            if enabled:profiler.disable()
            elapsed=(time.perf_counter_ns()-start)/1e6;change=delta(before,npu.stats())
            assert change['npu_calls']==change['chain_graphs']==900 and change['chain_groups']==240
            assert all(v==0 for k,v in change.items() if k not in ('npu_calls','chain_graphs','chain_groups'))
            report['profiled' if enabled else 'uninstrumented']=dict(elapsed_ms=elapsed,per_decode_ms=elapsed/30,delta=change)
        stats=pstats.Stats(profiler)
        rows=[dict(file=file,line=line,function=name,calls=nc,self_ms=tt*1000,cumulative_ms=ct*1000)
              for (file,line,name),(cc,nc,tt,ct,callers) in stats.stats.items()]
        native=[row for row in rows if row['file']=='~' and 'run_chain' in row['function']]
        assert len(native)==1 and native[0]['calls']==30
        report.update(native_chain_ms_per_decode=native[0]['cumulative_ms']/30,
                      largest_self_time=sorted(rows,key=lambda row:row['self_ms'],reverse=True)[:30],functions=rows)
        equal(arrays(stack.decode(one,seed)),anchor)
        print(json.dumps({k:report[k] for k in ('uninstrumented','profiled','native_chain_ms_per_decode','largest_self_time')},indent=2),flush=True)
    with torch.no_grad():work();final(npu,report)
    report['status']='passed';(OUT/'host.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()

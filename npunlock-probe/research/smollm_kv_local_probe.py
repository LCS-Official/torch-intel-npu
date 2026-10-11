"""Isolate local cache error from accumulated predecessor error, offline."""
from pathlib import Path
import gc
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_npu_smollm import FrozenSmolStack
from torch_npu_smollm_stack_demo import errors
import torch_intel_npu as npu

ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/'models/SmolLM2-135M-Instruct'
os.environ['HF_HUB_OFFLINE']='1'
torch.set_num_threads(8)

def main():
    cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    stream=ids.repeat(1,(70+ids.shape[1]-1)//ids.shape[1])
    ids=torch.cat([stream[:,i:i+63] for i in range(8)])
    report=[]
    with torch.no_grad():
        host=cpu.model.embed_tokens(ids);stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,mode='fp16')
        x=host.to('intel_npu');out,caches=stack.prefill(x);reference,traces=stack.reference_cpu(host)
        report.append(dict(whole_output=errors(out.cpu(),reference)))
        for i,(layer,cache,trace) in enumerate(zip(stack.layers,caches,traces)):
            local,local_trace=layer.reference_cpu(x.cpu())
            value,state=layer.prefill(x)
            entry=dict(layer=i,local_output=errors(value.cpu(),local))
            for name in ['keys','values']:
                actual=getattr(cache,name).cpu()[:,:,:63]
                assert torch.equal(actual,getattr(state,name).cpu()[:,:,:63])
                entry[name]=dict(accumulated=errors(actual,trace[name]),local=errors(actual,local_trace[name]))
            report.append(entry);x=value
        print(json.dumps(report,indent=2),flush=True)
        (ROOT/'npunlock-probe/research/results/smollm-kv/local-cache-probe.json').write_text(json.dumps(report,indent=2))

main();gc.collect();npu.shutdown();assert npu.stats()['live_buffers']==0

"""Same-model ABBA/BAAB control for serial versus folded-batch decode."""
from contextlib import ExitStack
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/smollm-kv-batch'
MODEL=ROOT/'models/SmolLM2-135M-Instruct'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'


def execute():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from torch_npu_smollm import FrozenSmolLayer,FrozenSmolStack
    from torch_npu_smollm_stack_demo import errors
    import torch_intel_npu as npu
    names=['torch_npu_batch_benchmark.py','torch_npu_smollm.py','torch_intel_npu.py','npunlock/python/npunlock/npu4000.py']
    report=dict(status='running',checks=[],timings=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names})
    try:
        def work():
            torch.set_num_threads(8)
            tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
            cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
            path=OUT/'before/torch_npu_smollm.py';spec=importlib.util.spec_from_file_location('smollm_batch_before',path)
            old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old);old.__file__=str(ROOT/'torch_npu_smollm.py')
            report['baseline_adapter_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            methods=dict(previous=(old.FrozenSmolLayer.graph,old.FrozenSmolLayer._run_cached_attention),
                         current=(FrozenSmolLayer.graph,FrozenSmolLayer._run_cached_attention))
            for count,batch,prefix,mode in [(4,1,32,'fp16'),(4,1,32,'w8a16'),(4,4,63,'fp16'),(4,4,63,'w8a16'),(1,8,63,'fp16'),(1,8,63,'w8a16')]:
                ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
                stream=ids.repeat(1,(prefix+batch+ids.shape[1]-1)//ids.shape[1]);ids=torch.cat([stream[:,i:i+prefix+1] for i in range(batch)])
                host=cpu.model.embed_tokens(ids).contiguous();one=host[:,-1:].contiguous().to('intel_npu')
                model=(FrozenSmolLayer.from_checkpoint(MODEL,mode=mode) if count==1 else FrozenSmolStack.from_checkpoint(MODEL,num_layers=count,mode=mode))
                _,state=model.prefill(host[:,:prefix].contiguous().to('intel_npu'));layers=[model] if count==1 else list(model.layers)
                keys={name:[(dict(layer._keys),dict(layer._evidence)) for layer in layers] for name in methods}
                def strategy(name):
                    stack=ExitStack();graph,attention=methods[name]
                    stack.enter_context(patch.object(FrozenSmolLayer,'graph',graph))
                    stack.enter_context(patch.object(FrozenSmolLayer,'_run_cached_attention',attention))
                    for layer,(sessions,evidence) in zip(layers,keys[name]):
                        stack.enter_context(patch.object(layer,'_keys',sessions));stack.enter_context(patch.object(layer,'_evidence',evidence))
                    return stack
                calls={};outputs={}
                for name in methods:
                    with strategy(name):
                        model.decode(one,state);before=npu.stats();value,cache=model.decode(one,state);after=npu.stats()
                    calls[name]=after['npu_calls']-before['npu_calls'];outputs[name]=[value.cpu()]
                    for item in cache if type(cache) is tuple else (cache,):outputs[name].extend([item.keys.cpu(),item.values.cpu(),item._position.cpu()])
                    for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                comparisons=[errors(b,a) for a,b in zip(outputs['previous'],outputs['current'])]
                equal=all(torch.equal(a.view(torch.int16 if a.dtype==torch.float16 else torch.int32),b.view(torch.int16 if b.dtype==torch.float16 else torch.int32))
                          for a,b in zip(outputs['previous'],outputs['current']))
                assert equal,comparisons
                report['checks'].append(dict(num_layers=count,batch=batch,mode=mode,all_bits_equal=equal,comparisons=comparisons))
                samples={name:[] for name in methods};rounds=[];before=npu.stats()
                for index in range(41):
                    order=['previous','current','current','previous'] if index%2==0 else ['current','previous','previous','current']
                    local={name:[] for name in methods}
                    for name in order:
                        with strategy(name):
                            begin=time.perf_counter();model.decode(one,state);elapsed=(time.perf_counter()-begin)*1000
                        samples[name].append(elapsed);local[name].append(elapsed)
                    rounds.append({name:statistics.mean(values) for name,values in local.items()})
                after=npu.stats()
                for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                assert after['npu_calls']-before['npu_calls']==41*2*sum(calls.values())
                medians={name:statistics.median(values) for name,values in samples.items()};deltas=[e['previous']-e['current'] for e in rounds]
                entry=dict(num_layers=count,batch=batch,mode=mode,prefix_tokens=prefix,rounds=41,samples_ms=samples,round_pairs_ms=rounds,
                    median_ms=medians,latency_reduction_percent=100*(1-medians['current']/medians['previous']),
                    median_paired_delta_ms=statistics.median(deltas),positive_rounds=sum(delta>0 for delta in deltas),graph_calls=calls,
                    warm_host_copies_and_compiles=0,same_model_input_cache=True)
                report['timings'].append(entry);print({k:v for k,v in entry.items() if k not in ['samples_ms','round_pairs_ms']},flush=True)
                del model,state,value,cache,one,outputs,keys,layers
        with torch.inference_mode():work()
        gc.collect();report['stats_before_shutdown']=npu.stats();assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown();report['final_stats']=npu.stats();assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'control.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':
    if '--child' in sys.argv:raise SystemExit(execute())
    with (OUT/'control.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--child'],stdout=log,stderr=log,timeout=600)
    print((OUT/'control.log').read_text(encoding='utf-8'),flush=True);raise SystemExit(result.returncode)

"""Fixed-length greedy generation: four NPU layers and 26 CPU layers."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from decode_cache_probe import ROOT,OUT,bind,sha


def worker(runtime,label):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
    from torch_npu_smollm import FrozenSmolStack
    npu=bind(runtime);torch.set_num_threads(8)
    directory=ROOT/'models/SmolLM2-135M-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    cpu=AutoModelForCausalLM.from_pretrained(directory,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
    text=tokenizer.apply_chat_template([dict(role='system',content='Tell stories.'),dict(role='user',content='Tell a story about a bird.')],
        tokenize=False,add_generation_prompt=True)
    ids=tokenizer(text,return_tensors='pt').input_ids;assert ids.shape[1]+31<=64
    report=dict(status='running',runtime=runtime,source_sha256=sha(Path(__file__)),checks=[],rows=[],input_tokens=ids.shape[1],
        new_tokens=32,npu_layers=4,cpu_layers=26,stop_at_eos=False,rounds=3)
    arrays={}
    try:
        with torch.no_grad():
            baseline=cpu(ids,use_cache=True).logits[:,-1].float()
            # The new prompt exceeded the existing 3% W8A16 quality gate on
            # the old runtime. Keep that failure; benchmark FP16 without
            # loosening the gate. Both precisions remain in the layer controls.
            for mode in ('fp16',):
                stack=FrozenSmolStack.from_checkpoint(directory,num_layers=4,mode=mode)
                def run(*,cold=False):
                    cpu_cache=DynamicCache(config=cpu.config);generated=ids.clone();state=None;steps=[];npu_ms=[];snapshots={}
                    before=npu.stats();start=time.perf_counter_ns()
                    for step in range(32):
                        begin=time.perf_counter_ns();tokens=ids if step==0 else generated[:,-1:]
                        position=0 if state is None else state[0].length
                        host=cpu.model.embed_tokens(tokens);length=host.shape[1]
                        positions=torch.arange(position,position+length)[None]
                        mask=torch.where(torch.arange(position+length)[None,:]>torch.arange(position,position+length)[:,None],
                            torch.tensor(-float('inf'),dtype=torch.float16),torch.tensor(0,dtype=torch.float16))[None,None]
                        prepared=dict(position_ids=positions,attention_mask=mask,position_embeddings=cpu.model.rotary_emb(host,positions),use_cache=True)
                        segment=time.perf_counter_ns()
                        hidden,state=stack.prefill(host.to('intel_npu')) if state is None else stack.decode(host.to('intel_npu'),state)
                        hidden=hidden.cpu();npu_ms.append((time.perf_counter_ns()-segment)/1e6)
                        for layer in cpu.model.layers[4:]:hidden=layer(hidden,past_key_values=cpu_cache,**prepared)
                        logits=cpu.lm_head(cpu.model.norm(hidden))[:,-1].float();next_token=logits.argmax(-1,keepdim=True)
                        generated=torch.cat((generated,next_token),dim=1);steps.append((time.perf_counter_ns()-begin)/1e6)
                        if step in (0,15,31):snapshots[f'logits-{step}']=logits.numpy().copy()
                    elapsed=(time.perf_counter_ns()-start)/1e6;after=npu.stats()
                    assert all(cpu_cache.get_seq_length(i)==state[0].length for i in range(4,30))
                    calls=after['npu_calls']-before['npu_calls']
                    # The first pass also prepares resident gamma broadcasts.
                    assert (calls>=128 if cold else calls==128) and torch.isfinite(logits).all(),calls
                    snapshots['tokens']=generated.numpy().copy()
                    return dict(elapsed_ms=elapsed,steps_ms=steps,npu_with_transfers_ms=npu_ms,cache_length=state[0].length,
                        counter_delta={f:after[f]-before[f] for f in ('npu_calls','driver_allocations','allocation_reuses','host_upload_copies','host_download_copies','session_creations')}),snapshots
                warm,anchors=run(cold=True)
                first=torch.from_numpy(anchors['logits-0']);error=float(torch.linalg.vector_norm(first-baseline)/torch.linalg.vector_norm(baseline))
                assert error<.03,(mode,error)
                samples=[]
                for index in range(3):
                    row,actual=run();samples.append(row);assert row['counter_delta']['session_creations']==0
                    for name,value in actual.items():
                        np.testing.assert_array_equal(value.view(np.uint8),anchors[name].view(np.uint8))
                        arrays[f'{mode}/{index}/{name}']=value
                median=statistics.median(r['elapsed_ms'] for r in samples)
                # 31 decode iterations exclude the prefill step.
                decode=statistics.median(sum(r['steps_ms'][1:]) for r in samples)
                record=dict(mode=mode,samples=samples,median_elapsed_ms=median,end_to_end_tokens_per_second=32000/median,
                    median_decode_ms=decode,decode_tokens_per_second=31000/decode)
                report['rows'].append(record);report['checks'].append(dict(mode=mode,first_logits_relative_l2_cpu=error,
                    repeated_tokens_and_logits_bit_equal=True,cache_length=samples[0]['cache_length']))
                print(mode,record['end_to_end_tokens_per_second'],record['decode_tokens_per_second'],flush=True)
                del stack,actual,anchors,first
            del cpu;gc.collect();npu.shutdown();report['final_stats']=npu.stats()
            assert not report['final_stats']['context_failed'] and report['final_stats']['live_buffers']==report['final_stats']['cached_buffer_bytes']==0
        path=OUT/f'{label}-outputs.npz';np.savez(path,**arrays);report.update(status='passed',outputs_sha256=sha(path))
    except Exception:
        import traceback
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/f'{label}.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',choices=('before','after'),required=True)
    parser.add_argument('--label',required=True);parser.add_argument('--child',action='store_true');args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.child:return worker(args.runtime,args.label)
    with (OUT/f'{args.label}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--runtime',args.runtime,'--label',args.label,'--child'],stdout=log,stderr=log,cwd=ROOT,timeout=600)
    print((OUT/f'{args.label}.log').read_text(encoding='utf-8'),flush=True);return result.returncode


if __name__=='__main__':raise SystemExit(main())

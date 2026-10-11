"""Independent held-out prompts, unchanged quality/execution thresholds."""
import gc
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from w8_calibration import OUT,CALIBRATION,SOURCES,load_weights
from w8_quality_probe import ROOT,MODEL,sha
from torch_npu_smollm_stack_demo import errors

HOLDOUT=[
    ('bird','Tell stories.','Tell a story about a bird.'),
    ('math','You are a helpful assistant.','What is two plus two?'),
    ('paris','You are a helpful assistant.','Complete: Paris is the capital of'),
    ('ice','You are a helpful assistant.','Explain why ice floats on water.'),
    ('code','You are a helpful assistant.','Write a Python function that returns the larger of two numbers.'),
    ('trip','You are a helpful assistant.','If a train leaves at nine and arrives at eleven, how long is the trip?'),
    ('garden','Tell stories.','Tell a story about a gardener.'),
    ('dog','Tell stories.','Write a short story about a lost dog.'),
]
FRESH=[
    ('moon','You are a helpful assistant.','Why does the Moon appear to change shape?'),
    ('multiply','You are a helpful assistant.','What is seven times eight?'),
    ('sorting','You are a helpful assistant.','Explain sorting a list of numbers to a beginner.'),
    ('language','You are a helpful assistant.','Translate thank you into German.'),
    ('cat','Tell stories.','Tell a story about a cat.'),
    ('fox','Tell stories.','Tell a story about a fox.'),
    ('rain','Tell stories.','Write a short story about a rainy day.'),
    ('letter','Tell stories.','Write a story about someone finding a letter.'),
]


def metric(actual,baseline):
    logp=baseline.float().log_softmax(-1);logq=actual.float().log_softmax(-1)
    result=errors(actual,baseline);kl=float((logp.exp()*(logp-logq)).sum())
    return dict(**result,next_token_kl=kl,next_token_equal=bool(actual.argmax(-1)==baseline.argmax(-1)),
        quality_gate_passed=result['relative_l2']<.03 and kl<.02)


def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    import torch_intel_npu as npu
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from torch_npu_smollm import FrozenSmolStack
    torch.set_num_threads(8)
    prompts=HOLDOUT+FRESH
    assert set((system,prompt) for _,system,prompt in prompts).isdisjoint(CALIBRATION)
    report=dict(status='running',source_sha256=sha(Path(__file__)),calibration_weights_sha256=sha(OUT/'weights.npz'),
        sources={name:sha(ROOT/name) for name in SOURCES},calibration_prompts_disjoint=True,
        criteria=dict(logits_relative_l2=.03,next_token_kl=.02,local_execution_relative_l2=.003,local_max_scaled_error=.01),prompts=[])
    arrays={}
    def work():
        cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        stacks={label:FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,mode='w8a16',
            packed_weights=load_weights() if label=='calibrated' else None) for label in ('original','calibrated')}
        for tag,system,prompt in prompts:
            text=tokenizer.apply_chat_template([dict(role='system',content=system),dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
            ids=tokenizer(text,return_tensors='pt').input_ids;host=cpu.model.embed_tokens(ids);length=host.shape[1];assert length<=64
            baseline=cpu(ids,use_cache=False).logits[:,-1].float();positions=torch.arange(length)[None]
            prepared=dict(position_ids=positions,attention_mask=torch.triu(torch.full((length,length),-float('inf'),dtype=torch.float16),diagonal=1)[None,None],
                position_embeddings=cpu.model.rotary_emb(host,positions),use_cache=False)
            def tail(value):
                for layer in cpu.model.layers[4:]:value=layer(value,**prepared)
                return cpu.lm_head(cpu.model.norm(value))[:,-1].float()
            official=host
            for layer in cpu.model.layers[:4]:official=layer(official,**prepared)
            assert torch.equal(tail(official),baseline)
            entry=dict(tag=tag,split='regression' if tag in {row[0] for row in HOLDOUT} else 'fresh',
                system=system,prompt=prompt,input_tokens=length,manual_bridge_bit_equal=True,variants={})
            report['prompts'].append(entry);arrays[f'{tag}/baseline-logits']=baseline.numpy().copy();arrays[f'{tag}/ids']=ids.numpy().copy()
            for label,stack in stacks.items():
                x=host.to('intel_npu');device=x;staged=host;local=[]
                for i,layer in enumerate(stack.layers):
                    previous=device.cpu();reference,_=layer.reference_cpu(previous)
                    device,cache=layer.prefill(device);actual=device.cpu();error=errors(actual,reference)
                    assert error['relative_l2']<.003 and error['max_scaled_error']<.01,(tag,label,i,error)
                    assert cache.length==length
                    local.append(dict(layer=i,**error,evidence=list(layer._evidence.values())))
                    arrays[f'{tag}/{label}/layer-{i}-actual']=actual.numpy().copy();arrays[f'{tag}/{label}/layer-{i}-reference']=reference.numpy().copy()
                    staged,_=layer.reference_cpu(staged)
                    del cache
                logits=tail(actual);reference_logits=tail(staged);quality=metric(logits,baseline)
                arrays[f'{tag}/{label}/logits']=logits.numpy().copy();arrays[f'{tag}/{label}/cpu-staged-logits']=reference_logits.numpy().copy()
                before=npu.stats();warm,state=stack.prefill(x);after=npu.stats()
                assert torch.equal(warm.cpu().view(torch.int16),actual.view(torch.int16))
                assert after['npu_calls']-before['npu_calls']==4
                assert all(after[key]==before[key] for key in ('session_creations','host_upload_copies','host_download_copies'))
                entry['variants'][label]=dict(**quality,cpu_staged_quality=metric(reference_logits,baseline),local_layers=local,
                    first_four_layers_quality=errors(actual,official),warm_bit_equal=True,warm_graph_calls=4)
                del device,x,warm,state
            print(tag,{label:row['relative_l2'] for label,row in entry['variants'].items()},flush=True)
    try:
        with torch.no_grad():work()
        gc.collect();npu.shutdown();report['final_stats']=npu.stats()
        assert not report['final_stats']['context_failed'] and report['final_stats']['live_buffers']==report['final_stats']['cached_buffer_bytes']==0
        path=OUT/'holdout-outputs.npz';np.savez(path,**arrays);report.update(status='passed',outputs_sha256=sha(path),output_array_count=len(arrays),
            candidate_all_quality_gates_passed=all(row['variants']['calibrated']['quality_gate_passed'] for row in report['prompts']))
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'holdout.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

"""Three fixed CPU continuations, FP16/original W8/calibrated W8 controls."""
import gc
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from w8_calibration import OUT,SOURCES,load_weights
from w8_calibration_probe import metric
from w8_quality_probe import ROOT,MODEL,sha
from torch_npu_smollm_stack_demo import errors

PROMPTS=[('bird','Tell a story about a bird.'),('garden','Tell a story about a gardener.'),('cat','Tell a story about a cat.')]


def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    import torch_intel_npu as npu
    from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
    from torch_npu_smollm import FrozenSmolStack
    torch.set_num_threads(8)
    report=dict(status='running',source_sha256=sha(Path(__file__)),weights_sha256=sha(OUT/'weights.npz'),
        sources={name:sha(ROOT/name) for name in SOURCES},quality_gate_relative_l2=.03,quality_gate_kl=.02,prompts=[])
    arrays={}
    def work():
        cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        stacks={label:FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,mode='fp16' if label=='fp16' else 'w8a16',
            packed_weights=load_weights() if label=='calibrated' else None) for label in ('fp16','original','calibrated')}
        for tag,prompt in PROMPTS:
            text=tokenizer.apply_chat_template([dict(role='system',content='Tell stories.'),dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
            ids=tokenizer(text,return_tensors='pt').input_ids;generated=ids.clone();assert ids.shape[1]+31<=64
            states={label:None for label in stacks};tails={label:DynamicCache(config=cpu.config) for label in stacks}
            cpu_cache=DynamicCache(config=cpu.config);before=npu.stats()
            entry=dict(tag=tag,prompt=prompt,input_tokens=ids.shape[1],new_tokens=32,stop_at_eos=False,steps=[]);report['prompts'].append(entry)
            for step in range(32):
                tokens=ids if step==0 else generated[:,-1:]
                baseline=cpu(tokens,past_key_values=cpu_cache,use_cache=True,output_hidden_states=True)
                logits=baseline.logits[:,-1].float();arrays[f'{tag}/{step}/baseline-logits']=logits.numpy().copy()
                host=cpu.model.embed_tokens(tokens);length=host.shape[1];position=0 if step==0 else ids.shape[1]+step-1
                positions=torch.arange(position,position+length)[None]
                mask=torch.where(torch.arange(position+length)[None,:]>torch.arange(position,position+length)[:,None],
                    torch.tensor(-float('inf'),dtype=torch.float16),torch.tensor(0,dtype=torch.float16))[None,None]
                prepared=dict(position_ids=positions,attention_mask=mask,position_embeddings=cpu.model.rotary_emb(host,positions),use_cache=True)
                row=dict(step=step,position=position,variants={});entry['steps'].append(row)
                for label,stack in stacks.items():
                    old=states[label];x=host.to('intel_npu')
                    snapshots=None
                    if step in (1,15,31):snapshots=[(c.keys.cpu().view(torch.int16).clone(),c.values.cpu().view(torch.int16).clone()) for c in old]
                    hidden,state=stack.prefill(x) if old is None else stack.decode(x,old)
                    value=hidden.cpu();hidden_error=errors(value,baseline.hidden_states[4])
                    for layer in cpu.model.layers[4:]:value=layer(value,past_key_values=tails[label],**prepared)
                    actual=cpu.lm_head(cpu.model.norm(value))[:,-1].float()
                    row['variants'][label]=dict(**metric(actual,logits),first_four_layers=hidden_error)
                    arrays[f'{tag}/{step}/{label}-logits']=actual.numpy().copy()
                    assert torch.isfinite(actual).all() and all(c.length==position+length for c in state)
                    assert all(tails[label].get_seq_length(i)==position+length for i in range(4,30))
                    if snapshots is not None:
                        for previous,(keys,values) in zip(old,snapshots):
                            assert torch.equal(previous.keys.cpu().view(torch.int16),keys) and torch.equal(previous.values.cpu().view(torch.int16),values)
                        row['variants'][label]['old_snapshot_unchanged']=True
                    if step==31:
                        start=npu.stats();repeat,repeated_state=stack.decode(x,old);end=npu.stats()
                        assert torch.equal(repeat.cpu().view(torch.int16),hidden.cpu().view(torch.int16))
                        for original,repeated in zip(state,repeated_state):
                            for field in ('keys','values','_position'):
                                assert torch.equal(getattr(original,field).cpu().view(torch.uint8),getattr(repeated,field).cpu().view(torch.uint8))
                        assert end['npu_calls']-start['npu_calls']==4
                        assert all(end[field]==start[field] for field in ('session_creations','host_upload_copies','host_download_copies'))
                        row['variants'][label].update(warm_repeat_bit_equal=True,warm_graph_calls=4,warm_host_copies_and_compiles=0)
                    states[label]=state
                generated=torch.cat((generated,logits.argmax(-1,keepdim=True)),dim=1)
                print(tag,step,{label:round(result['relative_l2'],6) for label,result in row['variants'].items()},flush=True)
            after=npu.stats();entry['counter_delta']={field:after[field]-before[field] for field in ('npu_calls','session_creations','driver_allocations','host_upload_copies','host_download_copies')}
            arrays[f'{tag}/tokens']=generated.numpy().copy()
            entry.update(cache_length=states['calibrated'][0].length,forced_continuation=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True))
    try:
        with torch.no_grad():work()
        gc.collect();npu.shutdown();report['final_stats']=npu.stats()
        assert not report['final_stats']['context_failed'] and report['final_stats']['live_buffers']==report['final_stats']['cached_buffer_bytes']==0
        path=OUT/'decode-outputs.npz';np.savez(path,**arrays);report.update(status='passed',outputs_sha256=sha(path),output_array_count=len(arrays),
            candidate_all_quality_gates_passed=all(row['variants']['calibrated']['quality_gate_passed'] for entry in report['prompts'] for row in entry['steps']))
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'decode.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

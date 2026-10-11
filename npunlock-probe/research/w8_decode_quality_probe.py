"""Teacher-forced 32-token control: identical CPU-generated tokens per mode."""
import gc
import json
import os
from pathlib import Path
import traceback
from w8_quality_probe import ROOT,OUT,MODEL,sha
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'


def main():
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
    from torch_npu_smollm import FrozenSmolStack
    from torch_npu_smollm_stack_demo import errors
    torch.set_num_threads(8)
    report=dict(status='running',source_sha256=sha(Path(__file__)),quality_gate_relative_l2=.03,steps=[],
        sources={name:sha(ROOT/name) for name in ['torch_npu_smollm.py','torch_npu_lowbit.py','torch_intel_npu.py',
            'npunlock-probe/torch-device/_torch_intel_npu.pyd','npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf']})
    arrays={}
    def work():
        cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        text=tokenizer.apply_chat_template([dict(role='system',content='Tell stories.'),dict(role='user',content='Tell a story about a bird.')],tokenize=False,add_generation_prompt=True)
        ids=tokenizer(text,return_tensors='pt').input_ids;generated=ids.clone();assert ids.shape[1]+31<=64
        stacks={mode:FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,mode=mode) for mode in ('fp16','w8a16')}
        states={mode:None for mode in stacks};tails={mode:DynamicCache(config=cpu.config) for mode in stacks}
        cache=DynamicCache(config=cpu.config);before=npu.stats()
        for step in range(32):
            tokens=ids if step==0 else generated[:,-1:]
            baseline=cpu(tokens,past_key_values=cache,use_cache=True,output_hidden_states=True)
            logits=baseline.logits[:,-1].float();arrays[f'{step}/baseline-logits']=logits.numpy().copy()
            host=cpu.model.embed_tokens(tokens);length=host.shape[1];position=0 if step==0 else ids.shape[1]+step-1
            positions=torch.arange(position,position+length)[None]
            mask=torch.where(torch.arange(position+length)[None,:]>torch.arange(position,position+length)[:,None],
                torch.tensor(-float('inf'),dtype=torch.float16),torch.tensor(0,dtype=torch.float16))[None,None]
            prepared=dict(position_ids=positions,attention_mask=mask,position_embeddings=cpu.model.rotary_emb(host,positions),use_cache=True)
            entry=dict(step=step,position=position,variants={});report['steps'].append(entry)
            for mode,stack in stacks.items():
                old=states[mode];x=host.to('intel_npu')
                hidden,state=stack.prefill(x) if old is None else stack.decode(x,old)
                value=hidden.cpu();hidden_error=errors(value,baseline.hidden_states[4])
                for layer in cpu.model.layers[4:]:value=layer(value,past_key_values=tails[mode],**prepared)
                actual=cpu.lm_head(cpu.model.norm(value))[:,-1].float();logp=logits.log_softmax(-1);logq=actual.log_softmax(-1)
                error=errors(actual,logits)
                entry['variants'][mode]=dict(**error,first_four_layers=hidden_error,next_token_kl=float((logp.exp()*(logp-logq)).sum()),
                    next_token_equal=bool(actual.argmax(-1)==logits.argmax(-1)),quality_gate_passed=error['relative_l2']<.03)
                arrays[f'{step}/{mode}-logits']=actual.numpy().copy()
                assert torch.isfinite(actual).all() and all(c.length==position+length for c in state)
                assert all(tails[mode].get_seq_length(i)==position+length for i in range(4,30))
                if step==31:
                    start=npu.stats();repeat,repeated_state=stack.decode(x,old);end=npu.stats()
                    assert torch.equal(repeat.cpu().view(torch.int16),hidden.cpu().view(torch.int16))
                    assert end['npu_calls']-start['npu_calls']==4
                    assert all(end[field]==start[field] for field in ('session_creations','host_upload_copies','host_download_copies'))
                    entry['variants'][mode]['warm_repeat_bit_equal']=True
                states[mode]=state
            generated=torch.cat((generated,logits.argmax(-1,keepdim=True)),dim=1)
            print(step,{mode:round(row['relative_l2'],6) for mode,row in entry['variants'].items()},flush=True)
        after=npu.stats();report['counter_delta']={key:after[key]-before[key] for key in ('npu_calls','session_creations','driver_allocations','host_upload_copies','host_download_copies')}
        arrays['tokens']=generated.numpy().copy()
        report.update(input_tokens=ids.shape[1],new_tokens=32,cache_length=states['fp16'][0].length,
            forced_continuation=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True),stop_at_eos=False)
    try:
        with torch.no_grad():work()
        gc.collect();npu.shutdown();report['final_stats']=npu.stats()
        assert not report['final_stats']['context_failed'] and report['final_stats']['live_buffers']==report['final_stats']['cached_buffer_bytes']==0
        path=OUT/'decode-outputs.npz';np.savez(path,**arrays);report.update(status='passed',outputs_sha256=sha(path),output_array_count=len(arrays))
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'decode.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

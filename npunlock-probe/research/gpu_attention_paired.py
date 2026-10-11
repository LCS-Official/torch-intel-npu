"""Interleave eager/SDPA on the same GPU model, preserving the generation contract."""
import json
import time
import importlib.metadata as metadata
from datetime import datetime
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_cross_device import ROOT, MODEL, TOKENS, ROUNDS, sha, write, metric, distribution
from model_cross_device_followup import OUT, BASE


def main():
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    assert torch.xpu.is_available()
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,
        attn_implementation='eager',local_files_only=True).eval().to('xpu')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    with np.load(BASE/'fixture.npz',allow_pickle=False) as data:
        reference=data['logits'].copy();expected_tokens=data['tokens'].copy()
    ids=torch.from_numpy(np.repeat(expected_tokens[:,:24],8,axis=0))
    assert ids.shape==(8,24)

    def select(mode):
        model.set_attn_implementation(mode)
        assert model.config._attn_implementation==model.model.config._attn_implementation==mode

    def run(replay=None):
        torch.xpu.synchronize();begin=time.perf_counter_ns()
        generated=ids.clone().to('xpu');state=None
        rows=[];emitted=[];steps=[]
        for step in range(TOKENS):
            mark=begin if step==0 else time.perf_counter_ns()
            source=generated if step==0 else generated[:,-1:]
            result=model.model(source,past_key_values=state,use_cache=True);state=result.past_key_values
            logits=model.lm_head(result.last_hidden_state)[:,-1].float()
            token=logits.argmax(-1,keepdim=True);emitted.append(token.cpu())
            chosen=token if replay is None else replay[:,24+step:25+step]
            generated=torch.cat((generated,chosen),dim=1);rows.append(logits)
            torch.xpu.synchronize();steps.append((time.perf_counter_ns()-mark)/1e6)
        elapsed=(time.perf_counter_ns()-begin)/1e6
        logits_cpu=torch.stack(rows).cpu().numpy().copy()
        tokens=generated.cpu().numpy().copy()
        assert np.isfinite(logits_cpu).all() and tokens.shape==(8,56)
        # Identical prompts produce identical batch rows; retain one row as compact evidence.
        assert np.array_equal(logits_cpu,np.repeat(logits_cpu[:,:1],8,axis=1))
        assert np.array_equal(tokens,np.repeat(tokens[:1],8,axis=0))
        return dict(elapsed_ms=elapsed,first_token_ms=steps[0],step_ms=steps,
                    overall_tokens_per_second=8*32000/elapsed),logits_cpu[:,:1].copy(),tokens[:1].copy()

    anchors={};pairs=[];evidence={}
    with torch.no_grad():
        for mode in ('eager','sdpa'):
            select(mode)
            for _ in range(3):row,logits,tokens=run()
            anchors[mode]=(logits,tokens)
        for index in range(ROUNDS):
            order=('eager','sdpa') if index%2==0 else ('sdpa','eager')
            pair=dict(index=index,order=order)
            for mode in order:
                select(mode);row,logits,tokens=run()
                np.testing.assert_array_equal(logits.view(np.uint8),anchors[mode][0].view(np.uint8))
                np.testing.assert_array_equal(tokens,anchors[mode][1]);pair[mode]=row
            pairs.append(pair);print('pair',index,[(m,round(pair[m]['elapsed_ms'],2)) for m in order],flush=True)
        accuracy={}
        for mode in ('eager','sdpa'):
            select(mode)
            replay=torch.from_numpy(np.repeat(expected_tokens,8,axis=0)).to('xpu')
            _,logits,tokens=run(replay);np.testing.assert_array_equal(tokens,expected_tokens)
            errors=[metric(a,b) for a,b in zip(logits[:,0],reference[:,0])]
            accuracy[mode]=dict(max_relative_l2=max(r['relative_l2'] for r in errors),errors=errors,
                greedy_equal=bool(np.array_equal(anchors[mode][1],expected_tokens)))
            assert accuracy[mode]['max_relative_l2']<.03 and accuracy[mode]['greedy_equal']
            evidence.update({mode+'_replay_logits':logits,mode+'_greedy_logits':anchors[mode][0],
                             mode+'_greedy_tokens':anchors[mode][1]})
    path=OUT/'xpu-b8-paired-outputs.npz';np.savez_compressed(path,**evidence)
    summary={mode:{key:distribution([pair[mode][key] for pair in pairs]) for key in
        ('elapsed_ms','first_token_ms','overall_tokens_per_second')} for mode in ('eager','sdpa')}
    speedup=distribution([p['eager']['elapsed_ms']/p['sdpa']['elapsed_ms'] for p in pairs])
    report=dict(status='passed',started_at=datetime.now().astimezone().isoformat(),device='xpu',batch=8,
        dtype='float16',packages={name:metadata.version(name) for name in ('torch','transformers','numpy')},
        device_name=torch.xpu.get_device_name(0),checkpoint_sha256=sha(MODEL/'model.safetensors'),
        fixture_sha256=sha(BASE/'fixture.npz'),source_sha256=sha(__file__),pairs=pairs,summary=summary,
        sdpa_speedup=speedup,accuracy=accuracy,outputs_sha256=sha(path),
        protocol='one model; three warm generations per mode; seven alternating-order pairs; host token and per-step sync; 24 prompt + 32 generated tokens',
        evidence_scope='all eight identical batch rows checked equal; one row retained; full logits/greedy repeat checks')
    write(OUT/'xpu-b8-paired.json',report)
    print(json.dumps(dict(summary=summary,sdpa_speedup=speedup),indent=2),flush=True)


if __name__=='__main__':main()

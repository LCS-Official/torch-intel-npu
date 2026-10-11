"""Measure true cold/warm first-token stages; keep only temporary compact evidence."""
import time
ENTRY=time.perf_counter_ns()
import argparse
import gc
import json
import os
from pathlib import Path
from functools import wraps

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/startup-first-token'
MODEL=ROOT/'models/SmolLM2-135M-Instruct'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'


def main(args):
    imported=time.perf_counter_ns()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from model_cross_device import sha,metric,distribution
    imports_ms=(time.perf_counter_ns()-imported)/1e6
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    started=time.perf_counter_ns()
    import torch_intel_npu as npu
    import torch_npu_smollm as smol
    import torch_npu_graph_cache as cache
    backend_ms=(time.perf_counter_ns()-started)/1e6
    totals={}
    def hook(owner,name,label):
        original=getattr(owner,name)
        @wraps(original)
        def wrapped(*a,**kw):
            begin=time.perf_counter_ns()
            try:return original(*a,**kw)
            finally:
                row=totals.setdefault(label,dict(calls=0,ms=0.))
                row['calls']+=1;row['ms']+=(time.perf_counter_ns()-begin)/1e6
        setattr(owner,name,wrapped)
    hook(smol.FrozenSmolLayer,'graph','graph_build_serialize')
    hook(smol.FrozenSmolLayer,'_inspect_blob','inspect_blob')
    hook(smol.FrozenSmolLayer,'_norm_parameter','norm_parameter')
    hook(smol,'compile_fused_cached','cache_or_compile')
    hook(cache,'_read','cache_read')
    hook(cache,'_key','cache_key')
    hook(npu._native,'create_inference_session','session_create')
    hook(npu,'_run_many','layer_run_many')
    started=time.perf_counter_ns()
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,
        attn_implementation='eager',local_files_only=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    text=tokenizer.apply_chat_template([dict(role='system',content='Tell stories.'),
        dict(role='user',content='Tell a story about a bird.')],tokenize=False,add_generation_prompt=True)
    ids=tokenizer(text,return_tensors='pt').input_ids
    assert ids.shape==(1,24)
    stack=smol.FrozenSmolStack.from_checkpoint(MODEL,num_layers=30,mode='fp16')
    setup_ms=(time.perf_counter_ns()-started)/1e6
    report=dict(status='running',label=args.label,imports_ms=imports_ms,backend_ms=backend_ms,
                setup_ms=setup_ms,torch=torch.__version__,cache_directory=os.environ.get('NPUNLOCK_GRAPH_CACHE'))
    def first():
        before={k:dict(v) for k,v in totals.items()};stats=npu.stats()
        begin=time.perf_counter_ns();host=model.model.embed_tokens(ids)
        embedded=time.perf_counter_ns();x=host.to('intel_npu');uploaded=time.perf_counter_ns()
        hidden,state=stack.prefill(x);prefilled=time.perf_counter_ns()
        output=hidden.cpu();downloaded=time.perf_counter_ns()
        logits=model.lm_head(model.model.norm(output))[:,-1].float();headed=time.perf_counter_ns()
        token=logits.argmax(-1).item();end=time.perf_counter_ns()
        row=dict(ttft_ms=(end-begin)/1e6,embed_ms=(embedded-begin)/1e6,
            upload_ms=(uploaded-embedded)/1e6,prefill_ms=(prefilled-uploaded)/1e6,
            download_ms=(downloaded-prefilled)/1e6,head_ms=(headed-downloaded)/1e6,argmax_ms=(end-headed)/1e6,
            hooks={k:dict(calls=v['calls']-before.get(k,{}).get('calls',0),ms=v['ms']-before.get(k,{}).get('ms',0)) for k,v in totals.items()},
            stats_delta={k:npu.stats()[k]-stats[k] for k in ('npu_calls','session_creations','driver_allocations','host_upload_copies','host_download_copies')})
        assert len(state)==30 and all(k.length==24 for k in state)
        return row,output.numpy().copy(),logits.numpy().copy(),token,(end-ENTRY)/1e6
    with torch.no_grad():
        print(args.label,'cold prefill',flush=True)
        cold,hidden,logits,token,entry=first()
        report.update(cold=cold,entry_to_token_ms=entry,token=token,cold_cache=cache.stats())
        warm=[]
        for index in range(9):
            row,h,l,t,_=first()
            np.testing.assert_array_equal(h.view(np.uint8),hidden.view(np.uint8))
            np.testing.assert_array_equal(l.view(np.uint8),logits.view(np.uint8));assert t==token
            if index>=2:warm.append(row)
        reference=model(ids,use_cache=False).logits[:,-1].float().numpy()
        error=metric(logits,reference)
        assert error['relative_l2']<.03 and error['argmax_equal']
    evidence=OUT/(args.label+'-outputs.npz');np.savez_compressed(evidence,hidden=hidden,logits=logits)
    report.update(warm=warm,warm_ttft=distribution([r['ttft_ms'] for r in warm]),quality=error,
        cache=cache.stats(),outputs_sha256=sha(evidence),sources={str(p.relative_to(ROOT)):sha(p) for p in
        (Path(__file__),ROOT/'torch_npu_smollm.py',ROOT/'torch_npu_graph_cache.py')},
        checkpoint_sha256=sha(MODEL/'model.safetensors'),hook_scope='inclusive hook times; do not add nested timers')
    del stack,model;gc.collect();npu.shutdown();final=npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
    report.update(status='passed',final_stats=final)
    (OUT/(args.label+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(label=args.label,entry_to_token_ms=entry,cold=cold,warm_ttft=report['warm_ttft'],cache=report['cache']),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);main(args)

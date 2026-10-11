"""Full NPU decoder, explicit CPU embedding/head: paired greedy generation."""
import argparse
import json
from pathlib import Path
import statistics as st
import time
import numpy as np
from decode_chain_api import provenance, delta, equal
from w8_quality_probe import ROOT, MODEL, sha
from shave_fp32_pipeline import final

OUT=ROOT/'npunlock-probe/research/results/smollm-generation'
STAGES=('embedding','upload','decoder','download','norm','head','greedy','bookkeeping')


def main(mode,*,baseline_prepare=None,output=OUT):
    import torch
    import torch_intel_npu as npu
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch_npu_smollm import FrozenSmolStack
    from torch_npu_smollm_stack_demo import errors
    torch.set_num_threads(8)
    output.mkdir(parents=True,exist_ok=True)
    current_prepare=npu._prepare_many
    report=dict(status='running',mode=mode,sources=provenance(),source_sha256=sha(Path(__file__)),
                decoder_layers=30,cpu_decoder_layers=0,new_tokens=32,stop_at_eos=False,
                sampling='greedy argmax',rounds=7,samples=[],pairs=[])
    report['comparison']='output preparation, both group4' if baseline_prepare else 'group1 vs default group4'
    saved={}
    def work():
        cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,
                    attn_implementation='eager',local_files_only=True).eval()
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        text=tokenizer.apply_chat_template([dict(role='system',content='Tell stories.'),
                    dict(role='user',content='Tell a story about a bird.')],tokenize=False,add_generation_prompt=True)
        ids=tokenizer(text,return_tensors='pt').input_ids
        assert ids.shape==(1,ids.shape[1]) and ids.shape[1]+31<=64
        report['input_tokens']=ids.shape[1]
        stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=30,mode=mode)
        def run(group,*,cold=False):
            actual_group=None if baseline_prepare else group
            if baseline_prepare:npu._prepare_many=baseline_prepare if group==1 else current_prepare
            generated=ids.clone();state=None;logits_rows=[];steps=[];prefill_hidden=None
            before=npu.stats();begin=time.perf_counter_ns()
            for step in range(32):
                marks=[time.perf_counter_ns()]
                host=cpu.model.embed_tokens(ids if step==0 else generated[:,-1:]);marks.append(time.perf_counter_ns())
                x=host.to('intel_npu');marks.append(time.perf_counter_ns())
                hidden,state=stack.prefill(x) if state is None else stack.decode(x,state,group_size=actual_group)
                marks.append(time.perf_counter_ns())
                hidden=hidden.cpu();marks.append(time.perf_counter_ns())
                normalized=cpu.model.norm(hidden);marks.append(time.perf_counter_ns())
                logits=cpu.lm_head(normalized)[:,-1].float();marks.append(time.perf_counter_ns())
                token=logits.argmax(-1,keepdim=True);marks.append(time.perf_counter_ns())
                generated=torch.cat((generated,token),dim=1);logits_rows.append(logits)
                if step==0:prefill_hidden=hidden
                marks.append(time.perf_counter_ns())
                steps.append(dict(zip(STAGES,[(b-a)/1e6 for a,b in zip(marks,marks[1:])])) )
            elapsed=(time.perf_counter_ns()-begin)/1e6;change=delta(before,npu.stats())
            assert change['npu_calls']>=960 if cold else change['npu_calls']==960
            if not cold:
                assert change['session_creations']==0
                assert change['chain_graphs']==(0 if actual_group==1 else 930)
                assert change['chain_groups']==(0 if actual_group==1 else 248)
                assert change['host_upload_copies']==change['host_download_copies']==32
            assert all(c.length==ids.shape[1]+31 for c in state)
            assert all(torch.isfinite(row).all() for row in logits_rows)
            decode=sum(sum(row.values()) for row in steps[1:])
            return dict(group=group,elapsed_ms=elapsed,steps=steps,delta=change,
                first_token_ms=sum(steps[0].values()),decode_ms=decode,
                overall_tokens_per_second=32000/elapsed,decode_tokens_per_second=31000/decode),logits_rows,generated,prefill_hidden
        print(mode,'warming full 30-layer generation',ids.shape[1],'prompt tokens',flush=True)
        cold,anchor,tokens,prefill_hidden=run(1,cold=True)
        warm,actual,generated,_=run(None)
        assert torch.equal(tokens,generated)
        equal([row.numpy() for row in anchor],[row.numpy() for row in actual])
        baseline=cpu(ids,use_cache=False).logits[:,-1].float()
        report['first_logits_vs_cpu']=errors(anchor[0],baseline)
        report['first_logits_vs_cpu']['within_existing_3_percent_gate']=report['first_logits_vs_cpu']['relative_l2']<.03
        for step in (0,15,31):saved[f'logits-{step}']=anchor[step].numpy().copy()
        saved['tokens']=tokens.numpy().copy();saved['prefill-hidden']=prefill_hidden.numpy().copy()
        report['continuation']=tokenizer.decode(tokens[0,ids.shape[1]:],skip_special_tokens=True)
        for index in range(7):
            pair={1:[],4:[]}
            for group in ((1,None,None,1) if index%2==0 else (None,1,1,None)):
                row,actual,generated,_=run(group)
                assert torch.equal(tokens,generated)
                equal([v.numpy() for v in anchor],[v.numpy() for v in actual])
                row['all_32_logits_and_tokens_bit_equal']=True
                report['samples'].append(row);pair[1 if group==1 else 4].append(row['elapsed_ms'])
            report['pairs'].append({str(k):st.mean(v) for k,v in pair.items()})
        report['summary']={}
        for group in (1,4):
            rows=[r for r in report['samples'] if (1 if r['group']==1 else 4)==group]
            total=st.median(r['elapsed_ms'] for r in rows);decode=st.median(r['decode_ms'] for r in rows)
            report['summary'][str(group)]=dict(elapsed_ms=total,first_token_ms=st.median(r['first_token_ms'] for r in rows),
                overall_tokens_per_second=32000/total,decode_tokens_per_second=31000/decode,
                median_decode_stage_ms={k:st.median(st.mean(r['steps'][i][k] for i in range(1,32)) for r in rows) for k in STAGES})
        report['positive_rounds']=sum(p['4']<p['1'] for p in report['pairs'])
        # Isolate the last-position head candidate before changing any public path.
        full=cpu.lm_head(cpu.model.norm(prefill_hidden))[:,-1].float()
        last=cpu.lm_head(cpu.model.norm(prefill_hidden[:,-1:]))[:,0].float()
        report['last_position_head']=dict(bit_equal=bool(torch.equal(full.view(torch.int32),last.view(torch.int32))),**errors(last,full))
        measurements={False:[],True:[]}
        for index in range(21):
            for last_only in ((False,True,True,False) if index%2==0 else (True,False,False,True)):
                start=time.perf_counter_ns()
                value=cpu.lm_head(cpu.model.norm(prefill_hidden[:,-1:] if last_only else prefill_hidden))[:,-1].float()
                elapsed=(time.perf_counter_ns()-start)/1e6;measurements[last_only].append(elapsed)
                assert torch.equal(value.view(torch.int32),(last if last_only else full).view(torch.int32))
        report['last_position_head'].update(full_median_ms=st.median(measurements[False]),
            last_only_median_ms=st.median(measurements[True]),samples_ms={str(k):v for k,v in measurements.items()})
        print(mode,json.dumps(report['summary']),report['positive_rounds'],'/ 7 wins',flush=True)
        print('head',report['last_position_head']['bit_equal'],report['last_position_head']['full_median_ms'],
              report['last_position_head']['last_only_median_ms'],flush=True)
    with torch.no_grad():work();final(npu,report)
    path=output/(mode+'-outputs.npz');np.savez_compressed(path,**saved)
    report.update(status='passed',outputs_sha256=sha(path))
    (output/(mode+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('fp16','w8a16'),required=True)
    main(parser.parse_args().mode)

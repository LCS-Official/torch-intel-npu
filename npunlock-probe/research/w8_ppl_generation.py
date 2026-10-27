"""Frozen-policy, short-context WikiText NLL and independent greedy generation.

No quality-driven tuning, production changes or raw logit/activation storage.
Window loss is summed over unique next-token targets before computing PPL.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import traceback

import torch

import w8_layer_sensitivity as layers
from w8_ppl_data import REVISION,OFFSETS

OUT=layers.mixed.common.ROOT/'npunlock-probe/research/results/w8-ppl-generation'
layers.OUT=OUT
layers.mixed.OUT=OUT
UP=(0,1,3,4,7,8,9,13,14,15,16,17,18,19,20,22,23,24,25,26,27,29)
DOWN=(0,3,4,5,6,7,8,9,10,11,14,15,16,17,21,22,23,24,25,26,27,28,29)
POLICY=frozenset([(i,'mlp.up_proj.weight') for i in UP]+[(i,'mlp.down_proj.weight') for i in DOWN])
PROMPTS=(
    ('divide','What is twelve divided by three?'),
    ('planet','Name the planet closest to the Sun.'),
    ('format','Return only the word blue.'),
    ('square','Write a Python function that squares a number.'),
    ('paperclip','List three uses of a paperclip.'),
    ('shadow','Explain why a shadow changes length during a day.'),
    ('spanish','Translate good morning into Spanish.'),
    ('clock','Write two sentences about a clockmaker who loses a key.'),
)
SOURCE_NAMES=('npunlock-probe/research/w8_ppl_generation.py','npunlock-probe/research/w8_ppl_data.py',
    'npunlock-probe/research/w8_layer_sensitivity.py','npunlock-probe/research/w8_projection_mixed.py',
    'npunlock-probe/research/w8_sequential_calibration.py','torch_npu_smollm.py','torch_npu_lowbit.py',
    'models/SmolLM2-135M-Instruct/config.json','models/SmolLM2-135M-Instruct/tokenizer.json',
    'models/SmolLM2-135M-Instruct/tokenizer_config.json','models/SmolLM2-135M-Instruct/generation_config.json',
    'npunlock-probe/research/results/w8-layer-sensitivity/REPORT.md')


def read(name):return layers.mixed.read(name)
def write(name,value):layers.mixed.write(name,value)
def sha(path):return layers.mixed.common.sha(path)
def sources():return {name:sha(layers.mixed.common.ROOT/name) for name in SOURCE_NAMES}


def windows(tokens):
    assert len(tokens)==257
    covered=[]
    for start in range(0,193,32):
        first=0 if start==0 else 32
        positions=list(range(start+first+1,start+65))
        covered.extend(positions)
        yield dict(start=start,first=first,input_ids=tokens[start:start+64],targets=tokens[start+1:start+65])
    assert covered==list(range(1,257))


def loss(logits, targets, first):
    x=logits.reshape(-1,logits.shape[-1]).double()[first:]
    labels=torch.tensor(targets,dtype=torch.long)[first:]
    assert x.shape[0]==labels.numel() and torch.isfinite(x).all()
    values=-x.log_softmax(-1).gather(1,labels[:,None]).squeeze(1)
    return dict(nll_sum=float(values.sum()),count=values.numel())


def aggregate(rows):
    count=sum(row['count'] for row in rows)
    nll=sum(row['nll_sum'] for row in rows)/count
    return dict(count=count,nll=nll,ppl=math.exp(nll))


def greedy(step, prompt, stops, maximum):
    assert 1<=len(prompt) and len(prompt)+maximum-1<=64
    for token in prompt:logits=step(token)
    generated=[]
    for index in range(maximum):
        token=int(logits.argmax(-1))
        generated.append(token)
        if token in stops:break
        if index+1<maximum:logits=step(token)
    return dict(token_ids=generated,consumed=len(prompt)+len(generated)-1,
                stop_reason='eos' if generated[-1] in stops else 'max_new_tokens')


def freeze():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(layers.mixed.common.MODEL,local_files_only=True)
    corpus=read('corpus')
    assert corpus['revision']==REVISION and [row['offset'] for row in corpus['slices']]==list(OFFSETS)
    cases=[]
    for row in corpus['slices']:
        encoded=json.dumps(row['texts'],ensure_ascii=False,separators=(',',':')).encode()
        assert hashlib.sha256(encoded).hexdigest()==row['text_sha256']
        text='\n\n'.join(row['texts'])
        assert all(prompt not in text for _,prompt in layers.mixed.PROMPTS)
        ids=tokenizer(text,add_special_tokens=False).input_ids[:257]
        assert len(ids)==257
        cases.append(dict(offset=row['offset'],text_sha256=row['text_sha256'],tokens=ids))
    generation=[]
    previous={(system,prompt) for _,system,prompt in layers.mixed.HOLDOUT+layers.mixed.FRESH}|set(layers.mixed.PROMPTS)
    for tag,prompt in PROMPTS:
        system='You are a helpful assistant.'
        assert (system,prompt) not in previous
        ids=layers.mixed.common.chat(tokenizer,system,prompt)[0].tolist()
        assert len(ids)+31<=64
        generation.append(dict(tag=tag,system=system,prompt=prompt,tokens=ids))
    assert len(POLICY)==45
    old_report=(layers.mixed.common.ROOT/SOURCE_NAMES[-1]).read_text(encoding='utf-8')
    old_policy=set()
    for i,flags in re.findall(r'\| (\d+) \| \d+ \| \d+ \| ([^|]+)\|',old_report):
        for short in ('Up','Down'):
            if short in flags:old_policy.add((int(i),'mlp.'+short.lower()+'_proj.weight'))
    assert old_policy==POLICY
    selection=dict(selected=[list(key) for key in sorted(POLICY)],
        origin='Frozen top45 policy from the previous layer-sensitivity report; no new selection.',
        report_sha256=sha(layers.mixed.common.ROOT/SOURCE_NAMES[-1]))
    write('selection',selection)
    data=dict(status='frozen',sources=sources(),corpus_sha256=sha(OUT/'corpus.json'),
        selection_sha256=sha(OUT/'selection.json'),weights_sha256=read('prepare')['weights_sha256'],
        cases=cases,generation=generation,eos_ids=[tokenizer.eos_token_id],max_new_tokens=32,
        protocol='8 fixed WikiText slices, first 257 tokens each; 64 input positions, stride32, '
                 'unique next-token targets 1..256. Token-weighted FP64 NLL. '
                 '8 fresh chat prompts, own-path greedy, EOS stop, 32 new tokens, capacity64. '
                 'No new adoption threshold, policy tuning or benchmark-wide claim.')
    assert sum(64-window['first'] for row in cases for window in windows(row['tokens']))==2048
    write('freeze',data)
    print('FROZEN: 2048 real targets, 56 windows, 8 prompts; prefix lengths',[len(row['tokens']) for row in generation],flush=True)


def frozen():
    data=read('freeze')
    assert data['sources']==sources() and data['corpus_sha256']==sha(OUT/'corpus.json')
    assert data['selection_sha256']==sha(OUT/'selection.json')
    assert data['weights_sha256']==read('prepare')['weights_sha256']
    return data


def provenance():
    return dict(sources=sources(),freeze_sha256=sha(OUT/'freeze.json'),weights_sha256=read('prepare')['weights_sha256'])


def policy(label, modules):
    return frozenset(modules) if label=='fp16' else layers.GLOBAL if label=='mixed' else POLICY if label=='selected' else frozenset()


def cpu_run():
    from transformers import DynamicCache
    data=frozen()
    cpu,tokenizer=layers.mixed.common.model()
    # Cache each official-window reference in RAM before replacing the Linears.
    references=[]
    for case in data['cases']:
        for window in windows(case['tokens']):
            ids=torch.tensor([window['input_ids']])
            references.append(cpu(ids,use_cache=False).logits.float())
    modules,_=layers.install(cpu,layers.mixed.weights())
    report=dict(status='running',**provenance(),variants={})
    for label in ('fp16','w8','mixed','selected'):
        layers.set_policy(modules,policy(label,modules))
        record=dict(windows=[],generation=[])
        j=0
        for case in data['cases']:
            for window in windows(case['tokens']):
                actual=cpu(torch.tensor([window['input_ids']]),use_cache=False).logits.float()
                if label=='fp16':assert torch.equal(actual,references[j])
                current=loss(actual,window['targets'],window['first'])
                baseline=loss(references[j],window['targets'],window['first'])
                record['windows'].append(dict(offset=case['offset'],start=window['start'],**current,reference_nll_sum=baseline['nll_sum']))
                j+=1
        record['summary']=aggregate(record['windows'])
        for prompt in data['generation']:
            cache=DynamicCache(config=cpu.config)
            def step(token):
                return cpu(torch.tensor([[token]]),past_key_values=cache,use_cache=True).logits[:,-1].float()
            row=greedy(step,prompt['tokens'],set(data['eos_ids']),data['max_new_tokens'])
            assert cache.get_seq_length()==row['consumed']
            record['generation'].append(dict(tag=prompt['tag'],**row,text=tokenizer.decode(row['token_ids'],skip_special_tokens=True)))
        report['variants'][label]=record
        print('CPU',label,record['summary'],'generation lengths',[len(row['token_ids']) for row in record['generation']],flush=True)
    report.update(status='passed',npu_calls=0)
    write('cpu',report)


def npu_run(label):
    import torch_intel_npu as npu
    data=frozen()
    reference=read('cpu')['variants']['fp16']
    report=dict(status='running',label=label,**provenance(),windows=[],generation=[],graphs=[],local_checks=[],warm=[])
    def digest(value):return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
    def work():
        cpu,tokenizer=layers.mixed.common.model()
        stack=layers.make_stack(label,None if label=='fp16' else layers.mixed.weights())
        report['protected']=[list((i,name)) for i,layer in enumerate(stack.layers) for name in sorted(layer.fp16_projections)]
        j=0
        for case in data['cases']:
            for window in windows(case['tokens']):
                value=cpu.model.embed_tokens(torch.tensor([window['input_ids']])).contiguous().to('intel_npu')
                if j==0:
                    for i,layer in enumerate(stack.layers):
                        local,_=layer.reference_cpu(value.cpu())
                        value=layer(value)
                        error=layers.mixed.common.errors(value.cpu(),local)
                        assert error['relative_l2']<.003 and error['max_scaled_error']<.01,(i,error)
                        report['local_checks'].append(dict(layer=i,**error))
                else:value=stack(value)
                logits=cpu.lm_head(cpu.model.norm(value.cpu())).float()
                current=loss(logits,window['targets'],window['first'])
                row=dict(offset=case['offset'],start=window['start'],**current,
                         reference_nll_sum=reference['windows'][j]['nll_sum'])
                report['windows'].append(row)
                j+=1
            print('NPU',label,'PPL slice',case['offset'],aggregate(report['windows']),flush=True)
        report['summary']=aggregate(report['windows'])
        for prompt in data['generation']:
            state=None;last=None;seed=None
            def step(token):
                nonlocal state,last,seed
                x=cpu.model.embed_tokens(torch.tensor([[token]])).contiguous().to('intel_npu')
                previous=state
                output,state=stack.prefill(x) if state is None else stack.decode(x,state)
                last=(x,previous,output,state)
                if seed is None:seed=state
                return cpu.lm_head(cpu.model.norm(output.cpu()))[:,-1].float()
            # Hash the first seed before later steps can expose a mutation.
            logits=step(prompt['tokens'][0])
            seed_hash=[digest(v) for c in seed for v in (c.keys,c.values,c._position)]
            def seeded_step(token):
                if not seeded_step.used:
                    seeded_step.used=True
                    assert token==prompt['tokens'][0]
                    return logits
                return step(token)
            seeded_step.used=False
            row=greedy(seeded_step,prompt['tokens'],set(data['eos_ids']),data['max_new_tokens'])
            assert all(cache.length==row['consumed'] for cache in state)
            x,previous,output,current=last
            before=npu.stats();again,again_state=stack.decode(x,previous);after=npu.stats()
            assert after['npu_calls']-before['npu_calls']==30
            assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
            first=[output]+[v for c in current for v in (c.keys,c.values,c._position)]
            second=[again]+[v for c in again_state for v in (c.keys,c.values,c._position)]
            assert [digest(v) for v in first]==[digest(v) for v in second]
            assert seed_hash==[digest(v) for c in seed for v in (c.keys,c.values,c._position)]
            report['warm'].append(dict(tag=prompt['tag'],calls=30,repeated_arrays=91,seed_arrays=90,payload_copies_and_compiles=0))
            report['generation'].append(dict(tag=prompt['tag'],**row,text=tokenizer.decode(row['token_ids'],skip_special_tokens=True)))
            print('NPU',label,'free generation',prompt['tag'],len(row['token_ids']),row['stop_reason'],flush=True)
        for i,layer in enumerate(stack.layers):
            assert len(layer._evidence)==3
            for key,evidence in layer._evidence.items():
                integer=[m for m in evidence['modes'] if not m['elementwise'] and m['wmode'] in (1,2)]
                assert bool(integer)==(label!='fp16') and all(m['amode']==0 and m['wmode']==2 for m in integer)
                report['graphs'].append(dict(layer=i,key=list(key),evidence=evidence))
    try:
        work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    gc.collect();npu.shutdown();final=npu.stats();report['final_stats']=final
    assert final['closed'] and not final['context_failed']
    assert all(final[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
    if report['status']=='passed':
        expected=30*(56+sum(row['consumed'] for row in report['generation'])+8)+120
        assert final['npu_calls']==expected,(final['npu_calls'],expected)
    write('npu-'+label,report)
    print('NPU final',label,report.get('summary'),final['npu_calls'],flush=True)
    return int(report['status']!='passed')


def selfcheck():
    w=list(windows(list(range(257))))
    assert len(w)==7 and sum(64-row['first'] for row in w)==256
    logits=torch.zeros(1,64,4)
    result=loss(logits,[1]*64,32)
    assert result['count']==32 and abs(math.exp(result['nll_sum']/32)-4)<1e-12
    assert aggregate([dict(nll_sum=1,count=1),dict(nll_sum=9,count=3)])['nll']==2.5
    calls=[]
    def step(token):calls.append(token);return torch.tensor([[0.,1.,2.]])
    eos=greedy(step,[4,5],{2},32)
    assert eos['token_ids']==[2] and eos['consumed']==2 and calls==[4,5]
    calls.clear();full=greedy(step,[4,5],{9},3)
    assert full['token_ids']==[2,2,2] and full['consumed']==4 and calls==[4,5,2,2]
    assert len(POLICY)==45 and POLICY<=layers.GLOBAL
    print('PASS: unique targets, token-weighted NLL/PPL, EOS and consumption accounting',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=('selfcheck','freeze','cpu','npu-fp16','npu-w8','npu-mixed','npu-selected'),required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    assert not os.environ.get('NPUNLOCK_GRAPH_CACHE')
    torch.set_num_threads(8)
    with torch.no_grad():
        selfcheck()
        if args.case.startswith('npu-'):return npu_run(args.case[4:])
        if args.case!='selfcheck':{'freeze':freeze,'cpu':cpu_run}[args.case]()
    return 0


if __name__=='__main__':raise SystemExit(main())

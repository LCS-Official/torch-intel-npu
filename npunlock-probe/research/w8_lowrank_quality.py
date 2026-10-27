"""Fit W8 residual factors on calibration inputs; select before held-out quality.

Only packed weights/factors, token IDs and scalar metrics are temporary files.
No activation/logit tensors, model downloads, runtime dependencies or graphs.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import w8_projection_mixed as mixed
from torch_npu_lowbit_correction import LowRankCorrection,fit_lowrank_correction

OUT=mixed.common.ROOT/'npunlock-probe/research/results/w8-lowrank-quality'
mixed.OUT=OUT
RANKS=(8,16,32)


def write(name,data):mixed.write(name,data)
def read(name):return mixed.read(name)
def sha(path):return mixed.common.sha(path)


def prepare():
    assert 'torch_intel_npu' not in sys.modules
    mixed.prepare()
    cpu,_=mixed.common.model();packed=mixed.weights();metadata=read('prepare')
    ids=torch.cat([torch.tensor(row['token_ids']) for row in metadata['trajectories'] if row['index']%5!=4])
    hidden=cpu.model.embed_tokens(ids)
    positions=torch.arange(64)[None].expand(8,-1)
    mask=torch.triu(torch.full((64,64),-float('inf'),dtype=torch.float16),diagonal=1)[None,None]
    arrays={};started=time.perf_counter()
    for i,layer in enumerate(cpu.model.layers):
        captured={key:[] for key in ('norm1','attention','norm2','gated')};hooks=[]
        for name in ('self_attn.q_proj','self_attn.o_proj','mlp.gate_proj','mlp.down_proj'):
            key=mixed.input_name(name+'.weight')
            def capture(module,args,key=key):captured[key].append(args[0].detach().reshape(-1,args[0].shape[-1]).clone())
            hooks.append(layer.get_submodule(name).register_forward_pre_hook(capture))
        try:
            hidden=torch.cat([layer(chunk,position_ids=positions,attention_mask=mask,
                position_embeddings=cpu.model.rotary_emb(chunk,positions),use_cache=False) for chunk in hidden.split(8)])
        finally:
            for hook in hooks:hook.remove()
        observed={key:torch.cat(values) for key,values in captured.items()}
        for name in mixed.NAMES:
            weight=layer.get_parameter(name).detach().half().contiguous()
            correction=fit_lowrank_correction(weight,packed[i][name],observed[mixed.input_name(name)],rank=max(RANKS))
            a,b=correction.tensors()
            arrays[f'{i}/{name}/down']=a.numpy();arrays[f'{i}/{name}/up']=b.numpy()
        print('residual factors',i+1,'/30',flush=True)
    assert len(arrays)==420
    path=OUT/'corrections.npz';np.savez_compressed(path,**arrays)
    write('factors',dict(status='passed',rank=max(RANKS),ranks=list(RANKS),matrices=210,
        fit_rows=2048,ridge_fraction=.01,factor_sha256=sha(path),weights_sha256=metadata['weights_sha256'],
        helper_sha256=sha(mixed.common.ROOT/'torch_npu_lowbit_correction.py'),
        preparation_sha256=hashlib.sha256(inspect.getsource(prepare).encode()).hexdigest(),
        study_sha256=sha(Path(__file__)),seconds=time.perf_counter()-started,npu_calls=0))


def corrections(rank):
    metadata=read('factors')
    assert metadata['weights_sha256']==read('prepare')['weights_sha256']
    assert metadata['helper_sha256']==sha(mixed.common.ROOT/'torch_npu_lowbit_correction.py')
    assert metadata['preparation_sha256']==hashlib.sha256(inspect.getsource(prepare).encode()).hexdigest()
    assert sha(OUT/'corrections.npz')==metadata['factor_sha256'] and rank in RANKS
    result={i:{} for i in range(30)}
    with np.load(OUT/'corrections.npz',allow_pickle=False) as arrays:
        assert len(arrays.files)==420
        for key in arrays.files:
            if not key.endswith('/down'):continue
            index,name,_=key.split('/');a=arrays[key];b=arrays[key[:-4]+'up']
            assert a.dtype==b.dtype==np.float16 and a.shape[0]==b.shape[1]==max(RANKS)
            result[int(index)][name]=LowRankCorrection((b.shape[0],a.shape[1]),max(RANKS),a.tobytes(),b.tobytes()).truncated(rank)
    return result


class ReferenceLinear(mixed.ReferenceLinear):
    def __init__(self,original,packed,correction):
        super().__init__(original,packed)
        a,b=correction.tensors()
        self.register_buffer('down',a);self.register_buffer('up',b)
        self.rank=0

    def forward(self,value):
        output=super().forward(value)
        if self.protected or self.rank==0:return output
        z=F.linear(value.float(),self.down[:self.rank].float()).half()
        delta=F.linear(z.float(),self.up[:,:self.rank].float()).half()
        return (output+delta).half()


def install(cpu):
    packed=mixed.weights();factors=corrections(max(RANKS));modules=[]
    for i,layer in enumerate(cpu.model.layers):
        for name in mixed.NAMES:
            parent,field=name[:-7].rsplit('.',1)
            reference=ReferenceLinear(layer.get_submodule(name[:-7]),packed[i][name],factors[i][name])
            setattr(layer.get_submodule(parent),field,reference);modules.append((name,reference))
    return modules


def evaluate(cpu,modules,rank,cases,protected=()):
    for name,module in modules:module.rank=rank;module.protected=name in protected
    rows=[]
    for tag,offset,ids,baseline in cases:
        actual=cpu(ids,use_cache=False).logits[:,offset:].float()
        assert torch.isfinite(actual).all()
        for j in range(actual.shape[1]):
            a,b=actual[:,j],baseline[:,j]
            centered=float(torch.linalg.vector_norm((a-a.mean(-1,keepdim=True))-(b-b.mean(-1,keepdim=True)))/
                torch.linalg.vector_norm(b-b.mean(-1,keepdim=True)).clamp_min(1e-20))
            rows.append(dict(tag=tag,position=offset+j,centered_relative_l2=centered,**mixed.metric(a,b)))
    summary=mixed.common.summary(rows)
    summary['mean_centered_relative_l2']=float(np.mean([r['centered_relative_l2'] for r in rows]))
    return dict(summary=summary,positions=rows)


def select():
    cpu,_=mixed.common.model();cases=[]
    for row in read('prepare')['trajectories']:
        if row['index']%5==4:
            ids=torch.tensor(row['token_ids']);offset=row['prefix']-1
            cases.append((str(row['index']),offset,ids,cpu(ids,use_cache=False).logits[:,offset:].float()))
    modules=install(cpu);report=dict(status='running',ranks=list(RANKS),profiles={},
        criteria=read('prepare')['criteria'],factors_sha256=sha(OUT/'factors.json'),
        policy='Only internal 8 trajectories; rank failed positions/max raw L2/mean raw L2, then factor bytes. '
               'Ranks 8/16/32 fixed before evaluation. FP16/mixed50 are controls; no held-out tuning.')
    for rank in (0,*RANKS):
        row=evaluate(cpu,modules,rank,cases)
        row['factor_bytes']=0 if rank==0 else sum(len(q.down)+len(q.up) for layer in corrections(rank).values() for q in layer.values())
        report['profiles'][str(rank)]=row
        print('internal residual rank',rank,row['summary'],flush=True)
    report['mixed50_control']=evaluate(cpu,modules,0,cases,('mlp.up_proj.weight','mlp.down_proj.weight'))['summary']
    report['fp16_control']=evaluate(cpu,modules,0,cases,mixed.NAMES)['summary']
    assert report['fp16_control']['max_relative_l2']==0
    report['selected_rank']=int(min(report['profiles'],key=lambda r:(*mixed.common.ranking(report['profiles'][r]['summary']),report['profiles'][r]['factor_bytes'])))
    report.update(status='passed',source_sha256=sha(Path(__file__)),npu_calls=0)
    assert 'torch_intel_npu' not in sys.modules
    write('selection',report);print('SELECTED RANK',report['selected_rank'],flush=True)


def independent():
    selection=read('selection');assert selection['factors_sha256']==sha(OUT/'factors.json')
    cpu,tokenizer=mixed.common.model();cases=[]
    for tag,system,prompt in mixed.HOLDOUT+mixed.FRESH:
        ids=mixed.common.chat(tokenizer,system,prompt);count=32 if tag in ('bird','garden','cat') else 1
        tokens=mixed.common.trajectory(cpu,ids,ids.shape[1]+count-1);offset=ids.shape[1]-1
        cases.append((tag,offset,tokens,cpu(tokens,use_cache=False).logits[:,offset:].float()))
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    ids=ids.repeat(1,(32+ids.shape[1])//ids.shape[1])[:,:32];tokens=mixed.common.trajectory(cpu,ids,64)
    cases.append(('repeat32',31,tokens,cpu(tokens,use_cache=False).logits[:,31:].float()))
    modules=install(cpu);report=dict(status='running',selected_rank=selection['selected_rank'],
        selection_sha256=sha(OUT/'selection.json'),cases=[dict(tag=t,offset=o,token_ids=i.tolist()) for t,o,i,_ in cases],variants={})
    for label,rank,protected in (('w8',0,()),('corrected',selection['selected_rank'],()),
        ('mixed50',0,('mlp.up_proj.weight','mlp.down_proj.weight')),('fp16',0,mixed.NAMES)):
        report['variants'][label]=evaluate(cpu,modules,rank,cases,protected)
        print('independent CPU',label,report['variants'][label]['summary'],flush=True)
    assert report['variants']['fp16']['summary']['max_relative_l2']==0
    report.update(status='passed',source_sha256=sha(Path(__file__)),npu_calls=0)
    write('independent',report)


def language_helpers():
    import w8_ppl_generation as language
    # The historical runner sets a shared research output directory at import.
    mixed.OUT=OUT
    return language


def data():
    import w8_ppl_data as dataset
    from transformers import AutoTokenizer
    dataset.OUT=OUT;dataset.main()
    tokenizer=AutoTokenizer.from_pretrained(mixed.common.MODEL,local_files_only=True)
    corpus=read('corpus');language=language_helpers();cases=[];generation=[]
    assert corpus['revision']==dataset.REVISION
    for row in corpus['slices']:
        encoded=json.dumps(row['texts'],ensure_ascii=False,separators=(',',':')).encode()
        assert hashlib.sha256(encoded).hexdigest()==row['text_sha256']
        tokens=tokenizer('\n\n'.join(row['texts']),add_special_tokens=False).input_ids[:257]
        assert len(tokens)==257
        cases.append(dict(offset=row['offset'],text_sha256=row['text_sha256'],tokens=tokens))
    previous=set(mixed.PROMPTS)|{(s,p) for _,s,p in mixed.HOLDOUT+mixed.FRESH}
    for tag,prompt in language.PROMPTS:
        system='You are a helpful assistant.'
        assert (system,prompt) not in previous
        tokens=mixed.common.chat(tokenizer,system,prompt)[0].tolist()
        assert len(tokens)+31<=64
        generation.append(dict(tag=tag,prompt=prompt,tokens=tokens))
    write('language-data',dict(status='frozen',corpus_sha256=sha(OUT/'corpus.json'),
        selection_sha256=sha(OUT/'selection.json'),cases=cases,generation=generation,
        eos_ids=[tokenizer.eos_token_id],max_new_tokens=32))


def language_quality(cpu,tokenizer,*,stack=None):
    from transformers import DynamicCache
    language=language_helpers();data=read('language-data')
    assert data['selection_sha256']==sha(OUT/'selection.json') and data['corpus_sha256']==sha(OUT/'corpus.json')
    npu=None
    if stack is not None:import torch_intel_npu as npu
    result=dict(windows=[],generation=[],local_checks=[],warm=[])
    for case in data['cases']:
        for window in language.windows(case['tokens']):
            ids=torch.tensor([window['input_ids']])
            if stack is None:logits=cpu(ids,use_cache=False).logits.float()
            else:
                value=cpu.model.embed_tokens(ids).contiguous().to('intel_npu')
                if not result['windows']:
                    for i,layer in enumerate(stack.layers):
                        reference,_=layer.reference_cpu(value.cpu())
                        value=layer(value)
                        error=mixed.common.errors(value.cpu(),reference)
                        assert error['relative_l2']<.003 and error['max_scaled_error']<.01,(i,error)
                        result['local_checks'].append(dict(layer=i,**error))
                else:value=stack(value)
                logits=cpu.lm_head(cpu.model.norm(value.cpu())).float()
            result['windows'].append(dict(offset=case['offset'],start=window['start'],**language.loss(logits,window['targets'],window['first'])))
        print('language slice',case['offset'],language.aggregate(result['windows']),flush=True)
    result['summary']=language.aggregate(result['windows'])
    assert result['summary']['count']==2048 and len(result['windows'])==56
    for prompt in data['generation']:
        state=None;cache=DynamicCache(config=cpu.config);last=None;seed=None;seed_hash=None
        def digest(t):return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()
        def step(token):
            nonlocal state,last,seed,seed_hash
            ids=torch.tensor([[token]])
            if stack is None:return cpu(ids,past_key_values=cache,use_cache=True).logits[:,-1].float()
            x=cpu.model.embed_tokens(ids).contiguous().to('intel_npu');previous=state
            output,state=stack.prefill(x) if state is None else stack.decode(x,state)
            last=(x,previous,output,state)
            if seed is None:
                seed=state;seed_hash=[digest(t) for c in seed for t in (c.keys,c.values,c._position)]
            return cpu.lm_head(cpu.model.norm(output.cpu()))[:,-1].float()
        row=language.greedy(step,prompt['tokens'],set(data['eos_ids']),data['max_new_tokens'])
        if stack is None:assert cache.get_seq_length()==row['consumed']
        else:
            assert all(c.length==row['consumed'] for c in state)
            x,previous,output,current=last;before=npu.stats();again,repeated=stack.decode(x,previous);after=npu.stats()
            assert after['npu_calls']-before['npu_calls']==30
            assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
            assert [digest(t) for t in [output]+[t for c in current for t in (c.keys,c.values,c._position)]]==[
                digest(t) for t in [again]+[t for c in repeated for t in (c.keys,c.values,c._position)]]
            assert seed_hash==[digest(t) for c in seed for t in (c.keys,c.values,c._position)]
            result['warm'].append(dict(tag=prompt['tag'],calls=30,repeated_arrays=91,seed_arrays=90))
        result['generation'].append(dict(tag=prompt['tag'],**row,text=tokenizer.decode(row['token_ids'],skip_special_tokens=True)))
        print('free generation',prompt['tag'],len(row['token_ids']),row['stop_reason'],flush=True)
    return result


def cpu_language():
    cpu,tokenizer=mixed.common.model();language=language_helpers();data=read('language-data')
    window=next(language.windows(data['cases'][0]['tokens']))
    reference=cpu(torch.tensor([window['input_ids']]),use_cache=False).logits.float()
    reference_hash=hashlib.sha256(reference.numpy().tobytes()).hexdigest()
    del reference
    modules=install(cpu);rank=read('selection')['selected_rank'];variants={}
    for label in ('fp16','w8','corrected'):
        for _,module in modules:module.rank=rank if label=='corrected' else 0;module.protected=label=='fp16'
        if label=='fp16':
            check=cpu(torch.tensor([window['input_ids']]),use_cache=False).logits.float()
            assert hashlib.sha256(check.numpy().tobytes()).hexdigest()==reference_hash
        variants[label]=language_quality(cpu,tokenizer)
        print('CPU language',label,variants[label]['summary'],flush=True)
    assert 'torch_intel_npu' not in sys.modules
    write('cpu-language',dict(status='passed',variants=variants,npu_calls=0,
        language_data_sha256=sha(OUT/'language-data.json'),source_sha256=sha(Path(__file__))))


def npu_run(label,graph_cache_mib=512):
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    npu._MAX_SESSION_BYTES=graph_cache_mib*1024*1024
    selection=read('selection');report=dict(status='running',label=label,positions=[],graphs=[],warm=[],guards=[],
        graph_cache_mib=graph_cache_mib,
        selection_sha256=sha(OUT/'selection.json'),weights_sha256=read('prepare')['weights_sha256'],
        factors_sha256=sha(OUT/'factors.json'),source_sha256=sha(Path(__file__)),
        adapter_sha256=sha(mixed.common.ROOT/'torch_npu_smollm.py'))
    def work():
        cpu,tokenizer=mixed.common.model();packed=None if label=='fp16' else mixed.weights()
        factors=corrections(selection['selected_rank']) if label in ('probe','corrected') else None
        if label=='probe':
            for bad in ({0:[]},{1:factors[0]},{}):
                before=npu.stats()
                try:FrozenSmolStack.from_checkpoint(mixed.common.MODEL,num_layers=1,packed_weights={0:packed[0]},corrections=bad)
                except ValueError:
                    after=npu.stats()
                    assert all(after[k]==before[k] for k in ('npu_calls','session_creations','driver_allocations','host_upload_copies'))
                    report['guards'].append(repr(list(bad)))
                else:raise AssertionError('Invalid correction map accepted')
            for rank in RANKS:
                layer=FrozenSmolStack.from_checkpoint(mixed.common.MODEL,num_layers=1,packed_weights={0:packed[0]},
                    corrections={0:{name:q.truncated(rank) for name,q in factors[0].items()}}).layers[0]
                x=(torch.randn(1,3,576,generator=torch.Generator().manual_seed(444))/4).half()
                reference,_=layer.reference_cpu(x);actual,cache=layer.prefill(x.to('intel_npu'))
                error=mixed.common.errors(actual.cpu(),reference)
                assert error['relative_l2']<.003 and error['max_scaled_error']<.01,(rank,error)
                report['graphs'].append(dict(rank=rank,error=error,evidence=list(layer._evidence.values())))
                print('NPU correction probe',rank,error,flush=True)
            return
        stack=FrozenSmolStack.from_checkpoint(mixed.common.MODEL,num_layers=30,mode='fp16' if label=='fp16' else 'w8a16',
            packed_weights=packed,corrections=factors)
        for case in read('independent')['cases']:
            ids=torch.tensor(case['token_ids']);offset=case['offset'];host=cpu.model.embed_tokens(ids)
            baseline=cpu(ids,use_cache=False).logits[:,offset:].float();state=None
            for position in range(ids.shape[1]):
                x=host[:,position:position+1].contiguous().to('intel_npu');previous=state
                output,state=stack.prefill(x) if state is None else stack.decode(x,state)
                assert all(c.length==position+1 for c in state)
                if position>=offset:
                    actual=cpu.lm_head(cpu.model.norm(output.cpu()))[:,-1].float()
                    report['positions'].append(dict(tag=case['tag'],position=position,**mixed.metric(actual,baseline[:,position-offset])))
            before=npu.stats();again,repeated=stack.decode(x,previous);after=npu.stats()
            assert after['npu_calls']-before['npu_calls']==30
            assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
            assert torch.equal(again.cpu().view(torch.int16),output.cpu().view(torch.int16))
            report['warm'].append(dict(tag=case['tag'],calls=30))
            print('NPU independent',label,case['tag'],mixed.common.summary([r for r in report['positions'] if r['tag']==case['tag']]),flush=True)
        report['summary']=mixed.common.summary(report['positions']);assert len(report['positions'])==142
        report['language']=language_quality(cpu,tokenizer,stack=stack)
        for i,layer in enumerate(stack.layers):
            for key,evidence in layer._evidence.items():
                integer=[m for m in evidence['modes'] if not m['elementwise'] and m['wmode'] in (1,2)]
                assert bool(integer)==(label!='fp16') and all(m['amode']==0 and m['wmode']==2 for m in integer)
                report['graphs'].append(dict(layer=i,key=list(key),evidence=evidence))
    try:
        work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    gc.collect();npu.shutdown();final=npu.stats();report['final_stats']=final
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','live_buffer_bytes','cached_buffer_bytes','cached_sessions','quarantined_buffer_bytes'))
    if not healthy:report['status']='failed'
    write('npu-'+label,report);print('NPU final',label,report.get('summary'),final['npu_calls'],report['status'],flush=True)
    return int(report['status']!='passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('prepare','select','independent','data','cpu-language','npu-probe','npu-corrected','npu-w8','npu-fp16'),required=True)
    parser.add_argument('--graph-cache-mib',type=int,choices=(512,768),default=512)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    torch.set_num_threads(8)
    with torch.no_grad():
        if args.case.startswith('npu-'):return npu_run(args.case[4:],args.graph_cache_mib)
        {'prepare':prepare,'select':select,'independent':independent,'data':data,'cpu-language':cpu_language}[args.case]()
    return 0


if __name__=='__main__':raise SystemExit(main())

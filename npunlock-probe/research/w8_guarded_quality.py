"""S45: guard rank-32 residual rows on internal calibration, then freeze evaluation.

Uses the S44 fitting protocol and existing factor API. Alpha is folded into
FP16 up factors; no extra inference operator, dependency or quality-gate change.
"""
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import w8_lowrank_quality as base
from torch_npu_lowbit_correction import LowRankCorrection

OUT=base.mixed.common.ROOT/'npunlock-probe/research/results/w8-guarded-quality'
base.OUT=base.mixed.OUT=OUT
FRESH=(
    ('s45-reason','If every red box is empty and this box is red, what is inside it?'),
    ('s45-add','What is 27 plus 36? Give just the number.'),
    ('s45-sub','A shelf has 24 books. Nine are removed. How many remain?'),
    ('s45-code','Write a Python expression that squares every number in a list named xs.'),
    ('s45-json','Return a JSON object with city set to Oslo and country set to Norway.'),
    ('s45-sort','Sort these numbers in ascending order: 7, 2, 9, 1.'),
    ('s45-science','Why does a metal spoon feel colder than a wooden spoon?'),
    ('s45-language','Translate into English: 这本书很有意思。'),
    ('s45-list','Name three vegetables that can be eaten raw.'),
    ('s45-grammar','Correct this sentence: They was waiting for the train.'),
    ('s45-compare','How are a square and a rectangle related?'),
    ('s45-dict','In Python, how do you safely look up a missing dictionary key?'),
    ('s45-safety','Why should you back up important files?'),
    ('s45-summary','Summarize: The bus arrived late, so Mira walked to school.'),
    ('s45-instruction','Write exactly the word blue, without punctuation.'),
    ('s45-chinese','请用一句话解释为什么月亮有时是圆的，有时是弯的。'),
)


def guard_rows(weight,decoded,inputs,down,up,*,group_rows=64):
    """Choose alpha in {0,1/2,1} per output; no group may exceed W8 SSE."""
    assert inputs.ndim==2 and len(inputs)%group_rows==0
    target=F.linear(inputs,weight).half()
    plain=(inputs.double()@decoded.t()).half()
    projected=F.linear(inputs.float(),down.float()).half().float()
    def error(candidate):return (candidate.double()-target.double()).square().reshape(-1,group_rows,len(up)).sum(1)
    limit=error(plain);best=limit.sum(0);scales=torch.zeros(len(up))
    for alpha in (.5,1.):
        candidate=(plain+F.linear(projected,(up*alpha).float()).half()).half()
        e=error(candidate);safe=(e<=limit+1e-18).all(0);improve=e.sum(0)<best
        chosen=safe&improve;scales[chosen]=alpha;best[chosen]=e.sum(0)[chosen]
    guarded=(up*scales[:,None]).half().contiguous()
    final=(plain+F.linear(projected,guarded.float()).half()).half()
    assert (error(final)<=limit+1e-18).all()
    return guarded,dict(rows=len(up),off=int((scales==0).sum()),half=int((scales==.5).sum()),
        full=int((scales==1).sum()),plain_sse=float(limit.sum()),guarded_sse=float(error(final).sum()))


@torch.no_grad()
def fit():
    cpu,_=base.mixed.common.model();metadata=base.read('prepare');packed=base.mixed.weights();factors=base.corrections(32)
    ids=torch.cat([torch.tensor(r['token_ids']) for r in metadata['trajectories'] if r['index']%5==4])
    hidden=cpu.model.embed_tokens(ids);positions=torch.arange(64)[None].expand(8,-1)
    mask=torch.triu(torch.full((64,64),-float('inf'),dtype=torch.float16),diagonal=1)[None,None]
    arrays={};rows=[];start=time.perf_counter()
    for i,layer in enumerate(cpu.model.layers):
        observed={};hooks=[]
        for name in ('self_attn.q_proj','self_attn.o_proj','mlp.gate_proj','mlp.down_proj'):
            key=base.mixed.input_name(name+'.weight')
            hooks.append(layer.get_submodule(name).register_forward_pre_hook(
                lambda module,args,key=key:observed.__setitem__(key,args[0].detach().reshape(-1,args[0].shape[-1]))))
        try:hidden=layer(hidden,position_ids=positions,attention_mask=mask,
            position_embeddings=cpu.model.rotary_emb(hidden,positions),use_cache=False)
        finally:
            for hook in hooks:hook.remove()
        for name in base.mixed.NAMES:
            q=packed[i][name];a,b=factors[i][name].tensors()
            decoded=torch.from_numpy(np.frombuffer(q.data,dtype=np.int8).reshape(q.shape).astype(np.float64))*torch.from_numpy(np.frombuffer(q.scales,dtype=np.float16).astype(np.float64))[:,None]
            b,metrics=guard_rows(layer.get_parameter(name).detach(),decoded,observed[base.mixed.input_name(name)],a,b)
            arrays[f'{i}/{name}/down']=a.numpy();arrays[f'{i}/{name}/up']=b.numpy()
            rows.append(dict(layer=i,name=name,**metrics))
        print('guarded layer',i+1,'/30',flush=True)
    np.savez_compressed(OUT/'guarded.npz',**arrays)
    base.write('guard',dict(status='passed',matrices=rows,seconds=time.perf_counter()-start,
        parent_sha256=base.sha(OUT/'factors.json'),data_sha256=base.sha(OUT/'guarded.npz'),
        method_sha256=hashlib.sha256(inspect.getsource(guard_rows).encode()).hexdigest(),npu_calls=0,
        policy='S44 fit 32 trajectories; guard rows using only internal 8 trajectories, alpha=0/.5/1. '
        'Per-output local SSE may not exceed plain W8 in any of the 8 groups. No held-out tuning.'))


def corrections():
    metadata=base.read('guard')
    assert metadata['parent_sha256']==base.sha(OUT/'factors.json')
    assert metadata['data_sha256']==base.sha(OUT/'guarded.npz')
    assert metadata['method_sha256']==hashlib.sha256(inspect.getsource(guard_rows).encode()).hexdigest()
    result={i:{} for i in range(30)}
    with np.load(OUT/'guarded.npz',allow_pickle=False) as arrays:
        assert len(arrays.files)==420
        for key in arrays.files:
            if not key.endswith('/down'):continue
            i,name,_=key.split('/');a=arrays[key];b=arrays[key[:-4]+'up']
            assert a.dtype==b.dtype==np.float16 and a.shape[0]==b.shape[1]==32
            result[int(i)][name]=LowRankCorrection((b.shape[0],a.shape[1]),32,a.tobytes(),b.tobytes())
    return result


def apply_guard(modules,factors):
    for (name,module),(i,key) in zip(modules,((i,key) for i in range(30) for key in base.mixed.NAMES),strict=True):
        assert name==key
        a,b=factors[i][name].tensors();module.down.copy_(a);module.up.copy_(b)


@torch.no_grad()
def evaluate():
    cpu,tokenizer=base.mixed.common.model();cases=[];internal=[]
    old=set(base.mixed.PROMPTS)|{(s,p) for _,s,p in base.mixed.HOLDOUT+base.mixed.FRESH}
    assert not any(('You are a helpful assistant.',prompt) in old for _,prompt in FRESH)
    for row in base.read('prepare')['trajectories']:
        if row['index']%5==4:
            ids=torch.tensor(row['token_ids']);offset=row['prefix']-1
            internal.append((str(row['index']),offset,ids,cpu(ids,use_cache=False).logits[:,offset:].float()))
    for tag,system,prompt in (*base.mixed.HOLDOUT,*base.mixed.FRESH,
        *((tag,'You are a helpful assistant.',prompt) for tag,prompt in FRESH)):
        ids=base.mixed.common.chat(tokenizer,system,prompt);count=32 if tag in ('bird','garden','cat') else 1
        assert ids.shape[1]+count-1<=64,(tag,ids.shape)
        tokens=base.mixed.common.trajectory(cpu,ids,ids.shape[1]+count-1);offset=ids.shape[1]-1
        cases.append((tag,offset,tokens,cpu(tokens,use_cache=False).logits[:,offset:].float()))
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    ids=ids.repeat(1,(32+ids.shape[1])//ids.shape[1])[:,:32];tokens=base.mixed.common.trajectory(cpu,ids,64)
    cases.append(('repeat32',31,tokens,cpu(tokens,use_cache=False).logits[:,31:].float()))
    modules=base.install(cpu);report=dict(status='running',internal={},regression={},fresh={},
        guard_sha256=base.sha(OUT/'guard.json'),cases=[dict(tag=t,offset=o,token_ids=i.tolist()) for t,o,i,_ in cases])
    for label,rank in (('w8',0),('corrected',32),('guarded',32)):
        if label=='guarded':apply_guard(modules,corrections())
        for group,selected in (('internal',internal),('regression',[c for c in cases if not c[0].startswith('s45-')]),
            ('fresh',[c for c in cases if c[0].startswith('s45-')])):
            report[group][label]=base.evaluate(cpu,modules,rank,selected)
            print(group,label,report[group][label]['summary'],flush=True)
    ranking=base.mixed.common.ranking
    report['selected']='guarded' if ranking(report['internal']['guarded']['summary'])<ranking(report['internal']['corrected']['summary']) else 'corrected'
    report.update(status='passed',npu_calls=0,source_sha256=base.sha(Path(__file__)))
    base.write('guard-evaluation',report)


@torch.no_grad()
def strength():
    """A fixed scalar sweep uses internal data only; alpha=1 remains a control."""
    cpu,tokenizer=base.mixed.common.model();cases=[]
    for row in base.read('prepare')['trajectories']:
        if row['index']%5==4:
            ids=torch.tensor(row['token_ids']);offset=row['prefix']-1
            cases.append((str(row['index']),offset,ids,cpu(ids,use_cache=False).logits[:,offset:].float()))
    modules=base.install(cpu);original=[m.up.clone() for _,m in modules];report=dict(internal={},alphas=[.5,.75,1.,1.25])
    def apply(alpha):
        for (_,m),up in zip(modules,original,strict=True):m.up.copy_((up*alpha).half())
    for alpha in report['alphas']:
        apply(alpha);result=base.evaluate(cpu,modules,32,cases)
        report['internal'][str(alpha)]=result
        print('strength internal',alpha,result['summary'],flush=True)
    rank=base.mixed.common.ranking
    selected=min(report['alphas'],key=lambda a:(*rank(report['internal'][str(a)]['summary']),abs(a-1)))
    report.update(selected_alpha=selected,parent_sha256=base.sha(OUT/'factors.json'),
        policy='Fixed alpha=.5/.75/1/1.25; select by internal failed count/max raw L2/mean raw L2, then proximity to 1. '
        'Fold alpha into FP16 up factors. Freeze selection before regression and 8-token fresh continuations.')
    base.write('strength-selection',report)
    # The 142 historical positions are regression data; the 16 new prompts are
    # extended to at most 8 continuation positions without changing the choice.
    frozen=base.sha(OUT/'strength-selection.json');cases=[]
    for row in base.read('guard-evaluation')['cases']:
        ids=torch.tensor(row['token_ids']);offset=row['offset']
        if row['tag'].startswith('s45-'):ids=base.mixed.common.trajectory(cpu,ids,min(64,len(ids[0])+7))
        cases.append((row['tag'],offset,ids,None))
    for _,m in modules:m.protected=True
    cases=[(t,o,ids,cpu(ids,use_cache=False).logits[:,o:].float()) for t,o,ids,_ in cases]
    report=dict(selection_sha256=frozen,selected_alpha=selected,cases=[dict(tag=t,offset=o,token_ids=ids.tolist()) for t,o,ids,_ in cases],variants={})
    for label,alpha in (('w8',0),('corrected',1),('strength',selected)):
        apply(alpha)
        report['variants'][label]={}
        for group,chosen in (('regression',[c for c in cases if not c[0].startswith('s45-')]),('fresh',[c for c in cases if c[0].startswith('s45-')])):
            report['variants'][label][group]=base.evaluate(cpu,modules,0 if alpha==0 else 32,chosen)
            print('strength',group,label,report['variants'][label][group]['summary'],flush=True)
    report.update(status='passed',npu_calls=0,source_sha256=base.sha(Path(__file__)))
    assert frozen==base.sha(OUT/'strength-selection.json')
    base.write('strength-evaluation',report)


@torch.no_grad()
def diagnose():
    cpu,tokenizer=base.mixed.common.model();cases=[]
    for row in base.read('strength-evaluation')['cases']:
        ids=torch.tensor(row['token_ids']);o=row['offset']
        cases.append((row['tag'],o,ids,cpu(ids,use_cache=False).logits[:,o:].float()))
    modules=base.install(cpu)
    for _,module in modules:module.rank=32
    rows=[]
    for tag,offset,ids,expected in cases:
        actual=cpu(ids,use_cache=False).logits[:,offset:].float()
        for j in range(actual.shape[1]):
            a,b=actual[:,j],expected[:,j];metrics=base.mixed.metric(a,b)
            if metrics['relative_l2']<.03 and metrics['next_token_kl']<.02:continue
            rows.append(dict(tag=tag,position=offset+j,**failure_detail(a,b,tokenizer)))
    base.write('diagnosis',dict(status='passed',failed=rows,npu_calls=0,
        parent_sha256=base.sha(OUT/'strength-evaluation.json'),source_sha256=base.sha(Path(__file__)),
        note='CPU exact-W8/FP16-factor reference. Offset decomposition diagnoses the existing raw-logit gate; it does not replace or relax that gate.'))
    print('diagnosis failures',len(rows),flush=True)


def failure_detail(actual,expected,tokenizer):
    error=(actual-expected).double();energy=error.square().sum();centered=error-error.mean(-1,keepdim=True)
    top=expected.topk(2,-1)
    return dict(**base.mixed.metric(actual,expected),mean_offset=float(error.mean()),
        offset_energy_fraction=float((error.mean().square()*error.numel())/energy.clamp_min(1e-30)),
        centered_error_same_denominator=float(torch.linalg.vector_norm(centered)/torch.linalg.vector_norm(expected.double()).clamp_min(1e-20)),
        reference_top_margin=float(top.values[0,0]-top.values[0,1]),
        reference_token=tokenizer.decode([int(expected.argmax(-1))]),actual_token=tokenizer.decode([int(actual.argmax(-1))]))


@torch.no_grad()
def hardware_kv():
    """Replay the four historical KV cases containing S44's 16 failed positions."""
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    report=dict(status='running',positions=[],failed=[],warm=[],route='S44 one-token prefill / grouped KV decode',
        evaluation_sha256=base.sha(OUT/'strength-evaluation.json'),source_sha256=base.sha(Path(__file__)),
        core_sha256=base.sha(base.mixed.common.ROOT/'torch_intel_npu.py'))
    def work():
        cpu,tokenizer=base.mixed.common.model()
        stack=FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,
            packed_weights=base.mixed.weights(),corrections=base.corrections(32))
        for case in base.read('strength-evaluation')['cases']:
            if case['tag'] not in ('bird','cat','fox','repeat32'):continue
            ids=torch.tensor(case['token_ids']);offset=case['offset'];host=cpu.model.embed_tokens(ids)
            expected=cpu(ids,use_cache=False).logits[:,offset:].float();state=None
            for position in range(ids.shape[1]):
                x=host[:,position:position+1].contiguous().to('intel_npu');previous=state
                output,state=stack.prefill(x) if state is None else stack.decode(x,state)
                assert all(c.length==position+1 for c in state)
                if position>=offset:
                    actual=cpu.lm_head(cpu.model.norm(output.cpu()))[:,-1].float()
                    metrics=base.mixed.metric(actual,expected[:,position-offset])
                    report['positions'].append(dict(tag=case['tag'],position=position,**metrics))
                    if not metrics['quality_gate_passed']:
                        report['failed'].append(dict(tag=case['tag'],position=position,**failure_detail(actual,expected[:,position-offset],tokenizer)))
            before=npu.stats();again,_=stack.decode(x,previous);after=npu.stats()
            assert after['npu_calls']-before['npu_calls']==30
            assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
            torch.testing.assert_close(again.cpu(),output.cpu(),rtol=0,atol=0)
            report['warm'].append(dict(tag=case['tag'],calls=30))
            print('KV diagnostic',case['tag'],flush=True)
        report['summary']=base.mixed.common.summary(report['positions']);assert len(report['positions'])==98
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    gc.collect();npu.shutdown();report['shutdown']=npu.stats()
    report['healthy']=report['shutdown']['closed'] and not report['shutdown']['context_failed'] and all(
        report['shutdown'][k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_sessions','quarantined_buffer_bytes'))
    if not report['healthy']:report['status']='failed'
    base.write('hardware-kv',report);print('KV diagnostic',report.get('summary'),report['healthy'],flush=True)
    return int(report['status']!='passed')


@torch.no_grad()
def hardware():
    """Separate fixed-shape, right-padded causal-forward evaluation, not KV decode."""
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    cpu,_=base.mixed.common.model();cases=[];metadata=base.read('strength-evaluation')
    for row in metadata['cases']:
        ids=torch.tensor(row['token_ids']);length=ids.shape[1]
        padded=F.pad(ids,(0,64-length),value=0)
        expected=cpu(padded,use_cache=False).logits[:,row['offset']:length].float()
        cases.append((row,padded,expected))
    report=dict(status='running',variants={},route='64-token right-padded causal forward; original positions only; not KV decode',
        evaluation_sha256=base.sha(OUT/'strength-evaluation.json'),sources={str(p):base.sha(base.mixed.common.ROOT/p) for p in (
            Path(__file__).relative_to(base.mixed.common.ROOT),Path('torch_npu_smollm.py'),
            Path('npunlock/src/graphinfer/session_win32.c'),Path('npunlock-probe/session-runtime/npunlock_session.dll'))})
    try:
        for label in ('w8','corrected','fp16'):
            stack=FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,
                mode='fp16' if label=='fp16' else 'w8a16',packed_weights=None if label=='fp16' else base.mixed.weights(),
                corrections=base.corrections(32) if label=='corrected' else None)
            rows=[];checks=[];start=time.perf_counter()
            for index,(case,ids,expected) in enumerate(cases):
                x=cpu.model.embed_tokens(ids).contiguous().to('intel_npu')
                if index==0:
                    for i,layer in enumerate(stack.layers):
                        reference,_=layer.reference_cpu(x.cpu());x=layer(x)
                        errors=base.mixed.common.errors(x.cpu(),reference)
                        assert errors['relative_l2']<.003 and errors['max_scaled_error']<.01,(label,i,errors)
                        checks.append(dict(layer=i,**errors))
                    del layer
                else:x=stack(x)
                actual=cpu.lm_head(cpu.model.norm(x.cpu())).float()[:,case['offset']:len(case['token_ids'][0])]
                for j in range(actual.shape[1]):
                    rows.append(dict(tag=case['tag'],position=case['offset']+j,**base.mixed.metric(actual[:,j],expected[:,j])))
            groups={group:base.mixed.common.summary([r for r in rows if r['tag'].startswith('s45-')==fresh]) for group,fresh in (('regression',False),('fresh',True))}
            report['variants'][label]=dict(positions=rows,groups=groups,local_checks=checks,seconds=time.perf_counter()-start,stats=npu.stats())
            print('NPU forward',label,groups,flush=True)
            if label=='fp16':assert all(r['quality_gate_passed'] for r in rows)
            del stack,x;gc.collect()
        report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    gc.collect();npu.shutdown();report['shutdown']=npu.stats()
    report['healthy']=report['shutdown']['closed'] and not report['shutdown']['context_failed'] and all(
        report['shutdown'][k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_sessions','quarantined_buffer_bytes'))
    if not report['healthy']:report['status']='failed'
    base.write('hardware-forward',report)
    return int(report['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','fit','evaluate','strength','diagnose','hardware','hardware-kv'))
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    with torch.no_grad():result={'prepare':base.prepare,'fit':fit,'evaluate':evaluate,'strength':strength,'diagnose':diagnose,'hardware':hardware,'hardware-kv':hardware_kv}[args.stage]()
    raise SystemExit(result or 0)

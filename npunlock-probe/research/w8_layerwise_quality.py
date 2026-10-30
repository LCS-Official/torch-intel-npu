"""S47: locate W8 error; freeze FP16 rescue on internal trajectories only.

Reuses the unchanged S44 preparation and existing per-layer FrozenSmol API.
Only token IDs, packed weights/factors and scalar metrics are saved temporarily.
Raw-logit L2 < .03 and next-token KL < .02 remain the acceptance criteria.
"""
import argparse
import gc
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F
import w8_lowrank_quality as base

ROOT=base.mixed.common.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-layerwise-quality'
base.OUT=base.mixed.OUT=OUT
NAMES=base.mixed.NAMES


def relative(value,reference):
    return float(torch.linalg.vector_norm(value.float())/torch.linalg.vector_norm(reference.float()).clamp_min(1e-20))


def error_parts(actual,reference,isolated,inherited):
    """Decompose signed errors; their norms are not additive contributions."""
    reference=reference.float();total=actual.float()-reference
    local=isolated.float()-reference;prefix=inherited.float()-reference
    interaction=total-local-prefix
    denominator=(torch.linalg.vector_norm(local)*torch.linalg.vector_norm(prefix)).clamp_min(1e-20)
    return dict(total=relative(total,reference),local=relative(local,reference),
        inherited=relative(prefix,reference),interaction=relative(interaction,reference),
        local_inherited_cosine=float((local*prefix).sum()/denominator),
        reconstruction_error=relative(total-(local+prefix+interaction),reference))


def offset_detail(actual,reference):
    actual=actual.float();reference=reference.float();delta=actual-reference
    shift=delta.mean(-1,keepdim=True)
    return dict(relative_l2=relative(delta,reference),
        centered_relative_l2=relative(delta-shift,reference-reference.mean(-1,keepdim=True)),
        offset_energy_fraction=float((shift.square().sum()*delta.shape[-1])/delta.square().sum().clamp_min(1e-20)))


def validated_policy(policy):
    result=frozenset((i,name) for i,name in policy)
    if len(result)!=len(policy) or any(type(i) is not int or not 0<=i<30 or name not in NAMES for i,name in result):
        raise ValueError('Expected distinct (layer 0..29, projection weight name) pairs')
    return result


def set_policy(modules,policy=(),rank=32):
    policy=validated_policy(policy)
    assert len(modules)==30*len(NAMES)
    for index,(name,module) in enumerate(modules):
        assert name==NAMES[index%len(NAMES)]
        module.rank=rank;module.protected=(index//len(NAMES),name) in policy


def extra_parameter_bytes(modules,policy):
    policy=validated_policy(policy);total=0
    for index,(name,module) in enumerate(modules):
        if (index//len(NAMES),name) in policy:
            count=module.original.weight.numel()
            total+=count*2-(count+2*module.original.out_features+2*(module.down.numel()+module.up.numel()))
    return total


def candidate_key(record):
    return (*base.mixed.common.ranking(record['summary']),record['extra_parameter_bytes'],record['label'])


def score(actual,reference,metadata):
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    rows=[]
    for batch,row in enumerate(metadata):
        offset=row['prefix']-1
        for position in range(offset,64):
            a,b=actual[batch:batch+1,position],reference[batch:batch+1,position]
            rows.append(dict(tag=str(row['index']),position=position,**base.mixed.metric(a,b)))
    assert len(rows)==290
    return dict(summary=base.mixed.common.summary(rows),positions=rows)


def layer_forward(cpu,layer,hidden):
    positions=torch.arange(hidden.shape[1])[None].expand(len(hidden),-1)
    mask=torch.triu(torch.full((hidden.shape[1],hidden.shape[1]),-float('inf'),dtype=hidden.dtype),diagonal=1)[None,None]
    return layer(hidden,position_ids=positions,attention_mask=mask,
        position_embeddings=cpu.model.rotary_emb(hidden,positions),use_cache=False)


def internal():
    cpu,_=base.mixed.common.model()
    metadata=[r for r in base.read('prepare')['trajectories'] if r['index']%5==4]
    ids=torch.cat([torch.tensor(r['token_ids']) for r in metadata])
    observed=torch.arange(64)[None]>=torch.tensor([r['prefix']-1 for r in metadata])[:,None]
    assert observed.sum()==290
    reference_states=[cpu.model.embed_tokens(ids)];hooks=[]
    for layer in cpu.model.layers:
        hooks.append(layer.register_forward_hook(lambda module,args,output:reference_states.append(output.detach().clone())))
    try:reference=cpu(ids,use_cache=False).logits
    finally:
        for hook in hooks:hook.remove()
    assert len(reference_states)==31
    # Verify manual suffix replay against the official full-model path first.
    hidden=reference_states[0]
    for i,layer in enumerate(cpu.model.layers):
        hidden=layer_forward(cpu,layer,hidden)
        torch.testing.assert_close(hidden,reference_states[i+1],rtol=0,atol=0)
    torch.testing.assert_close(cpu.lm_head(cpu.model.norm(hidden)),reference,rtol=0,atol=0)
    modules=base.install(cpu);set_policy(modules)
    corrected_states=[reference_states[0]];layers=[];projections=[]
    start=time.perf_counter()
    for i,layer in enumerate(cpu.model.layers):
        group=modules[i*len(NAMES):(i+1)*len(NAMES)]
        isolated=layer_forward(cpu,layer,reference_states[i])
        for _,module in group:module.protected=True
        inherited=layer_forward(cpu,layer,corrected_states[-1])
        for _,module in group:module.protected=False
        hooks=[]
        for name,module in group:
            def capture(module,args,output,name=name):
                target=module.original(args[0]);plain=base.mixed.ReferenceLinear.forward(module,args[0])
                projections.append(dict(layer=i,name=name,
                    plain=offset_detail(plain[observed],target[observed]),
                    corrected=offset_detail(output[observed],target[observed])))
            hooks.append(module.register_forward_hook(capture))
        try:hidden=layer_forward(cpu,layer,corrected_states[-1])
        finally:
            for hook in hooks:hook.remove()
        corrected_states.append(hidden)
        layers.append(dict(layer=i,**error_parts(hidden[observed],reference_states[i+1][observed],
            isolated[observed],inherited[observed])))
        print('layer error',i, layers[-1],flush=True)
    norm=cpu.model.norm(hidden);logits=cpu.lm_head(norm)
    terminal=dict(hidden=offset_detail(hidden[observed],reference_states[-1][observed]),
        norm=offset_detail(norm[observed],cpu.model.norm(reference_states[-1])[observed]),
        logits=offset_detail(logits[observed],reference[observed]))
    candidates=[]
    def record(label,policy,actual):
        row=dict(label=label,policy=sorted(policy),extra_parameter_bytes=extra_parameter_bytes(modules,policy),
            **score(actual,reference,metadata))
        candidates.append(row);print('candidate',label,row['summary'],flush=True)
        return row
    record('corrected',(),logits)
    def replay(policy):
        set_policy(modules,policy);first=min(i for i,_ in policy);value=corrected_states[first]
        for layer in cpu.model.layers[first:]:value=layer_forward(cpu,layer,value)
        return cpu.lm_head(cpu.model.norm(value))
    for i in range(30):
        policy=tuple((i,name) for name in NAMES)
        record(f'layer-{i:02d}',policy,replay(policy))
    best_block=min(candidates[1:],key=candidate_key)
    i=best_block['policy'][0][0]
    for name in NAMES:
        policy=((i,name),)
        record(f'layer-{i:02d}/{name}',policy,replay(policy))
    selected=min(candidates,key=candidate_key)
    set_policy(modules,selected['policy'])
    torch.testing.assert_close(cpu(ids,use_cache=False).logits,replay(selected['policy'])
        if selected['policy'] else logits,rtol=0,atol=0)
    set_policy(modules,tuple((i,name) for i in range(30) for name in NAMES))
    torch.testing.assert_close(cpu(ids,use_cache=False).logits,reference,rtol=0,atol=0)
    report=dict(status='passed',layers=layers,projections=projections,terminal=terminal,candidates=candidates,
        selected=selected,fp16_control=score(reference,reference,metadata)['summary'],
        factors_sha256=base.sha(OUT/'factors.json'),source_sha256=base.sha(Path(__file__)),
        seconds=time.perf_counter()-start,npu_calls=0,
        selection_policy='Internal 8 trajectories only, batch 8, 290 positions. 30 single-block and 7 single-projection '
        'rescues in the best block; compare with corrected baseline by failed positions/max raw L2/mean raw L2, '
        'then added projection bytes and label. Official/manual FP16 and selected replay must agree exactly. '
        'No held-out reading before selection; offset metrics do not replace raw L2/KL gates.')
    assert 'torch_intel_npu' not in sys.modules
    base.write('selection',report)
    print('SELECTED',selected['label'],selected['summary'],flush=True)


def frozen_selection():
    record=base.read('selection')
    assert record['status']=='passed' and record['factors_sha256']==base.sha(OUT/'factors.json')
    assert record['source_sha256']==base.sha(Path(__file__))
    validated_policy(record['selected']['policy'])
    return record


def independent():
    selection=frozen_selection();frozen=base.sha(OUT/'selection.json')
    # Import only after freezing: this historical script changes the shared OUT.
    from w8_guarded_quality import FRESH
    base.OUT=base.mixed.OUT=OUT
    cpu,tokenizer=base.mixed.common.model();cases=[]
    for tag,system,prompt in (*base.mixed.HOLDOUT,*base.mixed.FRESH,
        *((tag,'You are a helpful assistant.',prompt) for tag,prompt in FRESH)):
        ids=base.mixed.common.chat(tokenizer,system,prompt)
        count=8 if tag.startswith('s45-') else 32 if tag in ('bird','garden','cat') else 1
        tokens=base.mixed.common.trajectory(cpu,ids,min(64,len(ids[0])+count-1));offset=len(ids[0])-1
        cases.append((tag,offset,tokens,cpu(tokens,use_cache=False).logits[:,offset:].float()))
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    ids=ids.repeat(1,(32+ids.shape[1])//ids.shape[1])[:,:32];tokens=base.mixed.common.trajectory(cpu,ids,64)
    cases.append(('repeat32',31,tokens,cpu(tokens,use_cache=False).logits[:,31:].float()))
    modules=base.install(cpu);report=dict(status='running',selection_sha256=frozen,variants={},
        cases=[dict(tag=t,offset=o,token_ids=ids.tolist()) for t,o,ids,_ in cases])
    for label,rank,policy in (('w8',0,()),('corrected',32,()),('selected',32,selection['selected']['policy']),
        ('fp16',0,tuple((i,name) for i in range(30) for name in NAMES))):
        set_policy(modules,policy,rank);rows=[]
        for tag,offset,ids,expected in cases:
            actual=cpu(ids,use_cache=False).logits[:,offset:].float()
            for j in range(actual.shape[1]):
                a,b=actual[:,j],expected[:,j]
                rows.append(dict(tag=tag,position=offset+j,**{**offset_detail(a,b),**base.mixed.metric(a,b)}))
        groups={group:base.mixed.common.summary([r for r in rows if r['tag'].startswith('s45-')==fresh])
            for group,fresh in (('regression',False),('fresh',True))}
        assert groups['regression']['positions']==142 and groups['fresh']['positions']==128
        report['variants'][label]=dict(positions=rows,groups=groups)
        print('CPU independent',label,groups,flush=True)
        if label=='fp16':assert all(r['relative_l2']==0 for r in rows)
    assert frozen==base.sha(OUT/'selection.json') and 'torch_intel_npu' not in sys.modules
    report.update(status='passed',source_sha256=base.sha(Path(__file__)),npu_calls=0)
    base.write('independent',report)


def build_stack(label,policy):
    from torch_npu_smollm import FrozenSmolLayer,FrozenSmolStack
    packed=None if label=='fp16' else base.mixed.weights()
    factors=base.corrections(32) if label!='fp16' else None
    stack=FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,
        mode='fp16' if label=='fp16' else 'w8a16',packed_weights=packed,corrections=factors)
    if label=='selected':
        for i in sorted({i for i,_ in policy}):
            protected=tuple(name for j,name in policy if j==i)
            if len(protected)==len(NAMES):
                stack.layers[i]=FrozenSmolLayer.from_checkpoint(base.mixed.common.MODEL,layer_index=i,mode='fp16')
            else:
                stack.layers[i]=FrozenSmolLayer.from_checkpoint(base.mixed.common.MODEL,layer_index=i,
                    packed_weights=packed[i],fp16_projections=protected,
                    corrections={name:q for name,q in factors[i].items() if name not in protected})
    return stack


def hardware_kv(label):
    """Focused replay of the original failure trajectories; no policy tuning."""
    selection=frozen_selection();metadata=base.read('independent')
    assert metadata['selection_sha256']==base.sha(OUT/'selection.json')
    import torch_intel_npu as npu
    report=dict(status='running',label=label,positions=[],warm=[],
        selection_sha256=base.sha(OUT/'selection.json'),evaluation_sha256=base.sha(OUT/'independent.json'),
        source_sha256=base.sha(Path(__file__)),route='one-token prefill / grouped KV decode')
    def work():
        cpu,_=base.mixed.common.model();stack=build_stack(label,selection['selected']['policy'])
        start=time.perf_counter()
        for case in metadata['cases']:
            if case['tag'] not in ('bird','cat','fox','repeat32'):continue
            ids=torch.tensor(case['token_ids']);offset=case['offset'];host=cpu.model.embed_tokens(ids)
            expected=cpu(ids,use_cache=False).logits[:,offset:].float();state=None
            for position in range(ids.shape[1]):
                x=host[:,position:position+1].contiguous().to('intel_npu');previous=state
                output,state=stack.prefill(x) if state is None else stack.decode(x,state)
                assert output.device.type=='intel_npu' and all(c.length==position+1 for c in state)
                if position>=offset:
                    a=cpu.lm_head(cpu.model.norm(output.cpu()))[:,-1].float();b=expected[:,position-offset]
                    report['positions'].append(dict(tag=case['tag'],position=position,
                        **{**offset_detail(a,b),**base.mixed.metric(a,b)}))
            before=npu.stats();again,_=stack.decode(x,previous);after=npu.stats()
            assert after['npu_calls']-before['npu_calls']==30
            assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
            torch.testing.assert_close(again.cpu(),output.cpu(),rtol=0,atol=0)
            report['warm'].append(dict(tag=case['tag'],calls=30))
            print('NPU KV',label,case['tag'],flush=True)
        assert len(report['positions'])==98
        report.update(summary=base.mixed.common.summary(report['positions']),seconds=time.perf_counter()-start,stats=npu.stats())
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    gc.collect();npu.shutdown();report['shutdown']=npu.stats()
    report['healthy']=report['shutdown']['closed'] and not report['shutdown']['context_failed'] and all(
        report['shutdown'][k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_sessions','quarantined_buffer_bytes'))
    if not report['healthy']:report['status']='failed'
    base.write('hardware-kv-'+label,report);print('KV result',label,report.get('summary'),report['healthy'],flush=True)
    return int(report['status']!='passed')


def hardware(label):
    """One variant per process; default cache budgets and 64-token causal forward."""
    selection=frozen_selection();metadata=base.read('independent')
    assert metadata['status']=='passed' and metadata['selection_sha256']==base.sha(OUT/'selection.json')
    import torch_intel_npu as npu
    report=dict(status='running',label=label,positions=[],local_checks=[],
        selection_sha256=base.sha(OUT/'selection.json'),evaluation_sha256=base.sha(OUT/'independent.json'),
        route='64-token right-padded causal forward; original positions only; not KV decode',
        sources={name:base.sha(ROOT/name) for name in ('torch_intel_npu.py','torch_npu_smollm.py',
            'npunlock-probe/research/w8_layerwise_quality.py','npunlock-probe/session-runtime/npunlock_session.dll')})
    def work():
        cpu,_=base.mixed.common.model();stack=build_stack(label,selection['selected']['policy'])
        start=time.perf_counter()
        for index,case in enumerate(metadata['cases']):
            ids=torch.tensor(case['token_ids']);length=ids.shape[1];padded=F.pad(ids,(0,64-length),value=0)
            expected=cpu(padded,use_cache=False).logits[:,case['offset']:length].float()
            x=cpu.model.embed_tokens(padded).contiguous().to('intel_npu')
            if index==0:
                for i,layer in enumerate(stack.layers):
                    reference,_=layer.reference_cpu(x.cpu());x=layer(x)
                    errors=base.mixed.common.errors(x.cpu(),reference)
                    assert errors['relative_l2']<.003 and errors['max_scaled_error']<.01,(label,i,errors)
                    report['local_checks'].append(dict(layer=i,**errors))
                del layer
            else:x=stack(x)
            actual=cpu.lm_head(cpu.model.norm(x.cpu())).float()[:,case['offset']:length]
            for j in range(actual.shape[1]):
                a,b=actual[:,j],expected[:,j]
                report['positions'].append(dict(tag=case['tag'],position=case['offset']+j,
                    **{**offset_detail(a,b),**base.mixed.metric(a,b)}))
            print('NPU forward',label,index+1,'/33',flush=True)
        report['groups']={group:base.mixed.common.summary([r for r in report['positions'] if r['tag'].startswith('s45-')==fresh])
            for group,fresh in (('regression',False),('fresh',True))}
        if label=='fp16':assert all(r['quality_gate_passed'] for r in report['positions'])
        report.update(seconds=time.perf_counter()-start,stats=npu.stats())
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    gc.collect();npu.shutdown();report['shutdown']=npu.stats()
    report['healthy']=report['shutdown']['closed'] and not report['shutdown']['context_failed'] and all(
        report['shutdown'][k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_sessions','quarantined_buffer_bytes'))
    if not report['healthy']:report['status']='failed'
    base.write('hardware-'+label,report);print('hardware result',label,report.get('groups'),report['healthy'],flush=True)
    return int(report['status']!='passed')


def prepare():
    if (OUT/'weights.npz').exists():raise FileExistsError('Use the existing verified preparation or clear this study output first')
    base.prepare()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','internal','independent','hardware','hardware-kv'))
    parser.add_argument('--variant',choices=('corrected','selected','fp16'),default='selected')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    with torch.no_grad():
        result={'hardware':hardware,'hardware-kv':hardware_kv}[args.stage](args.variant) if args.stage.startswith('hardware') else globals()[args.stage]()
    raise SystemExit(result or 0)


if __name__=='__main__':main()

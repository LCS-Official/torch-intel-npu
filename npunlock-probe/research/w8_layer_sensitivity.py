"""Research-only logit diagnostics and Up/Down protection ablations.

Selection uses only the eight internal trajectories. No production API,
quality gate or kernel changes; raw logits and activations remain in RAM.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch

import w8_projection_mixed as mixed

OUT = mixed.common.ROOT/'npunlock-probe/research/results/w8-layer-sensitivity'
GLOBAL = frozenset((i,name) for i in range(30) for name in ('mlp.up_proj.weight','mlp.down_proj.weight'))
mixed.OUT = OUT


def write(name, value):
    mixed.write(name,value)


def read(name):
    return mixed.read(name)


def provenance():
    return dict(runner_sha256=mixed.common.sha(Path(__file__)),fit_sources=mixed.sources(),
                adapter_sha256=mixed.common.sha(mixed.common.ROOT/'torch_npu_smollm.py'),
                weights_sha256=read('prepare')['weights_sha256'])


def diagnostics(actual, reference):
    original = mixed.metric(actual,reference)
    x,y = actual.double().reshape(-1),reference.double().reshape(-1)
    delta = x-y
    offset = delta.mean()
    centered = delta-offset
    energy = delta.square().sum()
    shift = delta.numel()*offset.square()
    assert torch.allclose(energy,shift+centered.square().sum(),rtol=1e-10,atol=1e-12)
    logp,logq = y.log_softmax(0),x.log_softmax(0)
    p,q = logp.exp(),logq.exp()
    top = y.topk(2)
    a,b = top.indices.tolist()
    return dict(**original,reference_mean=float(y.mean()),mean_offset=float(offset),
        offset_energy_fraction=float(shift/energy) if energy else 0.,
        centered_delta_relative_l2=float(centered.norm()/y.norm().clamp_min(1e-30)),
        centered_logits_relative_l2=float(centered.norm()/(y-y.mean()).norm().clamp_min(1e-30)),
        kl_fp64=float((p*(logp-logq)).sum()),total_variation=float((p-q).abs().sum()*.5),
        max_probability_error=float((p-q).abs().max()),reference_top1_id=a,actual_top1_id=int(x.argmax()),
        reference_top2_margin=float(top.values[0]-top.values[1]),actual_margin_at_reference_top2=float(x[a]-x[b]),
        reference_top1_probability=float(p[a]),actual_reference_top1_probability=float(q[a]))


def summary(rows):
    result = mixed.common.summary(rows)
    if 'offset_energy_fraction' in rows[0]:
        result.update(mean_offset_energy_fraction=float(np.mean([r['offset_energy_fraction'] for r in rows])),
                      mean_centered_logits_relative_l2=float(np.mean([r['centered_logits_relative_l2'] for r in rows])),
                      max_centered_logits_relative_l2=max(r['centered_logits_relative_l2'] for r in rows),
                      max_total_variation=max(r['total_variation'] for r in rows),
                      mean_total_variation=float(np.mean([r['total_variation'] for r in rows])),
                      max_kl_fp64=max(r['kl_fp64'] for r in rows))
    return result


def install(cpu, packed):
    modules,counts = mixed.install_reference(cpu,packed)
    assert [name for name,_ in modules]==list(mixed.NAMES)*30
    mapping = {(j//7,name):module for j,(name,module) in enumerate(modules)}
    assert len(mapping)==210
    return mapping,sum(counts.values())


def set_policy(modules, policy):
    assert set(policy)<=set(modules)
    for key,module in modules.items():
        module.protected=key in policy


def internal_cases(cpu):
    cases = []
    for row in read('prepare')['trajectories']:
        if row['index']%5==4:
            ids = torch.tensor(row['token_ids'])
            offset = row['prefix']-1
            cases.append((str(row['index']),offset,ids,cpu(ids,use_cache=False).logits[:,offset:].float()))
    assert len(cases)==8 and sum(64-offset for _,offset,_,_ in cases)==290
    return cases


def evaluate(cpu, modules, policy, cases, *, detailed=True):
    set_policy(modules,policy)
    rows = []
    for tag,offset,ids,reference in cases:
        actual = cpu(ids,use_cache=False).logits[:,offset:].float()
        assert actual.shape==reference.shape and torch.isfinite(actual).all()
        for j in range(actual.shape[1]):
            fn = diagnostics if detailed else mixed.metric
            rows.append(dict(tag=tag,position=offset+j,**fn(actual[:,j],reference[:,j])))
    return dict(protected=[list(key) for key in sorted(policy)],summary=summary(rows),positions=rows)


def layer_arguments(cpu, value):
    length = value.shape[1]
    positions = torch.arange(length)[None]
    return dict(position_ids=positions,
                attention_mask=torch.triu(torch.full((length,length),-float('inf'),dtype=torch.float16),diagonal=1)[None,None],
                position_embeddings=cpu.model.rotary_emb(value,positions),use_cache=False)


def score_tail(cpu, value, start, args, offset):
    for layer in cpu.model.layers[start:]:
        value = layer(value,**args)
    return cpu.lm_head(cpu.model.norm(value))[:,offset:].float()


def select():
    cpu,_ = mixed.common.model()
    cases = internal_cases(cpu)
    modules,total = install(cpu,mixed.weights())
    original = evaluate(cpu,modules,GLOBAL,cases)
    cached = []
    for tag,offset,ids,reference in cases:
        value = cpu.model.embed_tokens(ids)
        args = layer_arguments(cpu,value)
        prefixes = []
        for layer in cpu.model.layers:
            prefixes.append(value)
            value = layer(value,**args)
        logits = cpu.lm_head(cpu.model.norm(value))[:,offset:].float()
        assert torch.equal(logits,cpu(ids,use_cache=False).logits[:,offset:].float())
        cached.append((tag,offset,prefixes,args,reference))
    report = dict(status='running',**provenance(),projection_total_elements=total,
                  ranking='Removal impact on failed count/max raw L2/mean raw L2; deterministic layer/name tie break.',
                  selection_rule='Minimize failed count/max raw L2/mean raw L2, then protected elements.',
                  criteria=dict(logits_relative_l2=.03,next_token_kl=.02),ablations=[],candidates=[])
    # One omission at a time: prefixes before that layer are exactly unchanged.
    for index,key in enumerate(sorted(GLOBAL)):
        modules[key].protected=False
        rows = []
        for tag,offset,prefixes,args,reference in cached:
            actual = score_tail(cpu,prefixes[key[0]],key[0],args,offset)
            if index==0:
                ids=next(ids for case_tag,_,ids,_ in cases if case_tag==tag)
                assert torch.equal(actual,cpu(ids,use_cache=False).logits[:,offset:].float())
            for j in range(actual.shape[1]):
                rows.append(dict(tag=tag,position=offset+j,**mixed.metric(actual[:,j],reference[:,j])))
        modules[key].protected=True
        report['ablations'].append(dict(removed=list(key),summary=summary(rows)))
        if (index+1)%10==0:
            print('internal ablations',index+1,'/60',flush=True)
    ordering = sorted(report['ablations'],key=lambda row:(*mixed.common.ranking(row['summary']),
                      -row['removed'][0],row['removed'][1]),reverse=True)
    ordered = [tuple(row['removed']) for row in ordering]
    report['sensitivity_order']=[list(key) for key in ordered]
    policies = [('w8',frozenset()),('top12',frozenset(ordered[:12])),('top30',frozenset(ordered[:30])),
                ('top45',frozenset(ordered[:45])),('top55',frozenset(ordered[:55])),('mixed',GLOBAL)]
    for label,policy in policies:
        record = original if label=='mixed' else evaluate(cpu,modules,policy,cases)
        record.update(label=label,fp16_elements=sum(modules[key].original.weight.numel() for key in policy))
        record['fp16_weight_fraction']=record['fp16_elements']/total
        assert record['fp16_weight_fraction']<=.5
        report['candidates'].append(record)
        print('internal candidate',label,record['summary'],'FP16',record['fp16_weight_fraction'],flush=True)
    best = min(report['candidates'],key=lambda row:(*mixed.common.ranking(row['summary']),row['fp16_elements']))
    control = evaluate(cpu,modules,frozenset(modules),cases)
    assert control['summary']['max_relative_l2']==0 and control['summary']['quality_passed']==290
    report.update(status='passed',selected=best['protected'],selected_label=best['label'],
                  selected_summary=best['summary'],selected_fp16_weight_fraction=best['fp16_weight_fraction'],
                  fp16_control=control['summary'],npu_calls=0)
    write('selection',report)
    print('SELECTED',best['label'],best['summary'],flush=True)


def independent_cases(cpu, tokenizer):
    cases = []
    for tag,system,prompt in mixed.HOLDOUT+mixed.FRESH:
        ids = mixed.common.chat(tokenizer,system,prompt)
        count = 32 if tag in ('bird','garden','cat') else 1
        tokens = mixed.common.trajectory(cpu,ids,ids.shape[1]+count-1)
        offset = ids.shape[1]-1
        cases.append((tag,offset,tokens,cpu(tokens,use_cache=False).logits[:,offset:].float()))
    ids = tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    ids = ids.repeat(1,(32+ids.shape[1])//ids.shape[1])[:,:32]
    tokens = mixed.common.trajectory(cpu,ids,64)
    cases.append(('repeat32',31,tokens,cpu(tokens,use_cache=False).logits[:,31:].float()))
    return cases


def independent():
    selection = read('selection')
    cpu,tokenizer = mixed.common.model()
    cases = independent_cases(cpu,tokenizer)
    modules,_ = install(cpu,mixed.weights())
    policies = [('w8',frozenset()),('mixed',GLOBAL),('fp16',frozenset(modules))]
    selected = frozenset(tuple(key) for key in selection['selected'])
    if selected!=GLOBAL and selected:
        policies.insert(2,('selected',selected))
    report = dict(status='running',**provenance(),selection_sha256=mixed.common.sha(OUT/'selection.json'),
                  cases=[dict(tag=tag,offset=offset,token_ids=ids.tolist()) for tag,offset,ids,_ in cases],variants={})
    for label,policy in policies:
        record = evaluate(cpu,modules,policy,cases)
        report['variants'][label]=record
        print('independent',label,record['summary'],flush=True)
    assert report['variants']['fp16']['summary']['max_relative_l2']==0
    report.update(status='passed',npu_calls=0)
    write('independent',report)


def make_stack(label, packed):
    from torch_npu_smollm import FrozenSmolLayer,FrozenSmolStack
    if label=='fp16':
        return FrozenSmolStack.from_checkpoint(mixed.common.MODEL,num_layers=30,mode='fp16')
    if label=='mixed':
        return FrozenSmolStack.from_checkpoint(mixed.common.MODEL,num_layers=30,mode='w8a16',packed_weights=packed,
            fp16_projections=('mlp.up_proj.weight','mlp.down_proj.weight'))
    policy = GLOBAL if label=='mixed' else frozenset(tuple(key) for key in read('selection')['selected']) if label=='selected' else frozenset()
    stack = FrozenSmolStack.from_checkpoint(mixed.common.MODEL,num_layers=30,mode='w8a16',packed_weights=packed)
    config = json.loads((mixed.common.MODEL/'config.json').read_text())
    # Research composition through existing layer APIs; no heterogeneous public stack option.
    for i,layer in enumerate(stack.layers):
        names = tuple(sorted(name for index,name in policy if index==i))
        if names:
            weights = {name:layer._cpu_weight(name) for name in layer._weights}
            stack.layers[i]=FrozenSmolLayer(config,weights,mode='w8a16',packed_weights=packed[i],fp16_projections=names)
    actual = frozenset((i,name) for i,layer in enumerate(stack.layers) for name in layer.fp16_projections)
    assert actual==policy
    return stack


def npu_run(label):
    import torch_intel_npu as npu
    data = read('independent')
    assert data['selection_sha256']==mixed.common.sha(OUT/'selection.json')
    assert label in data['variants']
    report = dict(status='running',label=label,**provenance(),selection_sha256=data['selection_sha256'],
                  positions=[],graphs=[],warm=[])
    def digest(value):
        return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
    def work():
        cpu,_ = mixed.common.model()
        stack = make_stack(label,None if label=='fp16' else mixed.weights())
        report['protected']=[list((i,name)) for i,layer in enumerate(stack.layers) for name in sorted(layer.fp16_projections)]
        for case in data['cases']:
            ids = torch.tensor(case['token_ids'])
            host = cpu.model.embed_tokens(ids)
            baseline = cpu(ids,use_cache=False).logits[:,case['offset']:].float()
            state = None
            for position in range(ids.shape[1]):
                x = host[:,position:position+1].contiguous().to('intel_npu')
                previous = state
                output,state = stack.prefill(x) if state is None else stack.decode(x,state)
                assert all(cache.length==position+1 for cache in state)
                if position==0:
                    seed=state
                    seed_hash=[digest(v) for c in seed for v in (c.keys,c.values,c._position)]
                if position>=case['offset']:
                    logits=cpu.lm_head(cpu.model.norm(output.cpu()))[:,-1].float()
                    assert torch.isfinite(logits).all()
                    report['positions'].append(dict(tag=case['tag'],position=position,**diagnostics(logits,baseline[:,position-case['offset']])))
            before=npu.stats()
            again,again_state=stack.decode(x,previous)
            after=npu.stats()
            assert after['npu_calls']-before['npu_calls']==30
            assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
            first=[output]+[v for c in state for v in (c.keys,c.values,c._position)]
            second=[again]+[v for c in again_state for v in (c.keys,c.values,c._position)]
            assert [digest(v) for v in first]==[digest(v) for v in second]
            assert seed_hash==[digest(v) for c in seed for v in (c.keys,c.values,c._position)]
            report['warm'].append(dict(tag=case['tag'],calls=30,repeated_arrays=91,seed_arrays=90,payload_copies_and_compiles=0))
            print('NPU',label,case['tag'],summary([r for r in report['positions'] if r['tag']==case['tag']]),flush=True)
        for i,layer in enumerate(stack.layers):
            evidence=[value for key,value in layer._evidence.items() if key[-1]=='decode']
            assert len(evidence)==1
            integer=[m for m in evidence[0]['modes'] if not m['elementwise'] and m['wmode'] in (1,2)]
            assert bool(integer)==(label!='fp16')
            assert all(m['amode']==0 and m['wmode']==2 for m in integer)
            report['graphs'].append(dict(layer=i,evidence=evidence[0]))
        report['summary']=summary(report['positions'])
    try:
        work()
        report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    gc.collect()
    npu.shutdown()
    report['final_stats']=final=npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
    write('npu-'+label,report)
    print('NPU final',label,report.get('summary'),final['npu_calls'],flush=True)
    return int(report['status']!='passed')


def selfcheck():
    y=torch.tensor([[1.,2.,4.,-1.]])
    shift=diagnostics(y+3,y)
    assert shift['offset_energy_fraction']==1 and shift['centered_logits_relative_l2']==0
    assert abs(shift['kl_fp64'])<1e-12 and shift['total_variation']<1e-12
    assert shift['next_token_equal'] and not shift['quality_gate_passed']
    temperature=diagnostics(y*.4,y)
    assert temperature['next_token_equal'] and temperature['total_variation']>.1
    x=y.clone();x[0,1]=5
    flip=diagnostics(x,y)
    assert not flip['next_token_equal'] and flip['actual_margin_at_reference_top2']<flip['reference_top2_margin']
    assert len(GLOBAL)==60
    print('PASS: shift/temperature/argmax diagnostics and original quality gates',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=('prepare','selfcheck','select','independent','npu-w8','npu-mixed','npu-selected','npu-fp16'),required=True)
    args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    assert not os.environ.get('NPUNLOCK_GRAPH_CACHE')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    with torch.no_grad():
        selfcheck()
        if args.case.startswith('npu-'):
            return npu_run(args.case[4:])
        if args.case!='selfcheck':
            {'prepare':mixed.prepare,'select':select,'independent':independent}[args.case]()
    return 0


if __name__=='__main__':
    raise SystemExit(main())

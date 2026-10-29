"""S54: whole-model logits/KL fitting of existing QKV output factors."""
import argparse
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import w8_qkv_quality as previous
from torch_npu_lowbit_correction import LowRankCorrection

base=previous.base
study=previous.study
ROOT=previous.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-logits-quality'
previous.OUT=study.OUT=study.previous.OUT=study.previous.previous.OUT=study.probe.OUT=study.device.OUT=study.device.previous.OUT=base.OUT=base.mixed.OUT=OUT
NAMES=previous.NAMES
PROFILES=('logits-e1','logits-e2')
FRESH=(
    ('sand','Describe how sand becomes glass in simple words.'),
    ('compass','A compass points north. Explain what its needle responds to.'),
    ('parcel','Give three steps for packing a fragile ceramic cup.'),
    ('moon','Why can the Moon be visible during the day?'),
    ('library','Write a polite one-sentence reminder to return a library book.'),
    ('pattern','Complete the number pattern 2, 5, 8, 11 and explain the rule.'),
    ('circuit','Describe what happens when a simple electrical circuit is opened.'),
    ('moss','Compare moss growing on a rock with a flowering garden plant.'))


def logits_loss(actual,target):
    if (not isinstance(actual,torch.Tensor) or not isinstance(target,torch.Tensor)
        or actual.shape!=target.shape or actual.ndim!=2 or not all(actual.shape)
        or actual.dtype!=torch.float32 or target.dtype!=torch.float32
        or actual.device.type!='cpu' or target.device.type!='cpu'
        or not torch.isfinite(actual).all() or not torch.isfinite(target).all()):
        raise ValueError('Requires finite matching CPU FP32 position-by-vocabulary logits')
    target=target.detach()
    l2=(actual-target).square().sum(-1)/target.square().sum(-1).clamp_min(1e-20)
    kl=F.kl_div(actual.log_softmax(-1),target.log_softmax(-1),reduction='none',log_target=True).sum(-1).clamp_min(0)
    return (l2+.1*kl+2*torch.relu(l2-.03**2)).mean()


def trainable_outputs(model,modules):
    model.requires_grad_(False);result=[]
    for index,(name,module) in enumerate(modules):
        module.rank=32;module.protected=False
        if name in NAMES:
            if module.up.dtype!=torch.float16 or module.up.shape[1]!=32:raise ValueError('Expected rank32 Half factors')
            prior=module.up.detach().float().clone();master=torch.nn.Parameter(prior.clone())
            result.append((index//7,name,module,master,prior))
    if len(result)!=90:raise ValueError('Expected 90 QKV output factors')
    return result


def bind_outputs(tunables):
    for _,_,module,master,_ in tunables:
        if not torch.isfinite(master).all():raise ValueError('Nonfinite master output factor')
        module.up=master.half()
        if not torch.isfinite(module.up).all():raise ValueError('Output factor overflows Half')


def source_identity():
    return dict(source_sha256=base.sha(Path(__file__)),weights_sha256=base.read('prepare')['weights_sha256'],
        factors_sha256=base.read('factors')['factor_sha256'],fresh_sha256=base.sha(OUT/'fresh.json'))


def identity():return dict(**source_identity(),fit_sha256=base.sha(OUT/'logits-fit.json'))


def fresh_data():
    if (OUT/'fresh.json').exists():raise FileExistsError('Refusing to overwrite frozen fresh cases')
    cpu,tokenizer=base.mixed.common.model();cases=[];system='You are a helpful assistant.'
    old=set(base.mixed.PROMPTS)|{(s,p) for _,s,p in (*base.mixed.HOLDOUT,*base.mixed.FRESH)}
    from w8_guarded_quality import FRESH as old_fresh
    base.OUT=base.mixed.OUT=OUT
    old|={(system,p) for _,p in old_fresh}
    assert len(FRESH)==len(set(FRESH))==8 and all((system,prompt) not in old for _,prompt in FRESH)
    with torch.no_grad():
        for tag,prompt in FRESH:
            ids=base.mixed.common.chat(tokenizer,system,prompt)
            assert 1<=ids.shape[1]<64
            tokens=base.mixed.common.trajectory(cpu,ids,64)
            cases.append(dict(tag=tag,prompt=prompt,offset=ids.shape[1]-1,token_ids=tokens.tolist()))
    base.write('fresh',dict(status='frozen',cases=cases,source_sha256=base.sha(Path(__file__)),
        policy='Eight new prompts frozen before fitting or internal selection; teacher-forced positions only.'))


def fit():
    if (OUT/'logits-fit.npz').exists():raise FileExistsError('Refusing to overwrite a logits fit')
    report=dict(status='running',steps=[],checkpoints={},**source_identity())
    cpu,_=base.mixed.common.model();rows=previous.partition(base.read('prepare')['trajectories'],True);cases=[]
    with torch.no_grad():
        for row in rows:
            ids=torch.tensor(row['token_ids']);offset=row['prefix']-1
            cases.append((row['index'],ids,offset,cpu(ids,use_cache=False).logits[:,offset:].float().detach()))
    modules=base.install(cpu);tunables=trainable_outputs(cpu,modules);masters=[r[3] for r in tunables]
    bind_outputs(tunables)
    def score():
        with torch.no_grad():
            return float(np.mean([float(logits_loss(cpu(ids,use_cache=False).logits[:,offset:].float()[0],target[0]))
                for _,ids,offset,target in cases]))
    report['before_objective']=score();optimizer=torch.optim.AdamW(masters,lr=2e-6,weight_decay=0.)
    anchor=sum(float(prior.square().sum()) for *_,prior in tunables);arrays={};started=time.perf_counter()
    for epoch,profile in enumerate(PROFILES,1):
        for index,ids,offset,target in cases:
            optimizer.zero_grad(set_to_none=True);bind_outputs(tunables)
            actual=cpu(ids,use_cache=False).logits[:,offset:].float()[0]
            objective=logits_loss(actual,target[0])
            penalty=sum((master-prior).square().sum() for _,_,_,master,prior in tunables)/max(anchor,1e-20)
            loss=objective+1e-5*penalty
            (loss*512).backward()
            for master in masters:
                if master.grad is None or not torch.isfinite(master.grad).all():raise ValueError('Missing or nonfinite output-factor gradient')
                master.grad.div_(512)
            norm=torch.nn.utils.clip_grad_norm_(masters,1.,error_if_nonfinite=True);optimizer.step()
            record=dict(epoch=epoch,trajectory=index,objective=float(objective.detach()),grad_norm=float(norm))
            report['steps'].append(record);print('logits fit step',len(report['steps']),'/64',record,flush=True)
        bind_outputs(tunables)
        for i,name,_,master,_ in tunables:arrays[f'{profile}/{i}/{name}/up']=master.detach().half().numpy().copy()
        report['checkpoints'][profile]=dict(objective=score())
        print('logits checkpoint',profile,report['checkpoints'][profile],flush=True)
    assert len(arrays)==180 and len(report['steps'])==64
    path=OUT/'logits-fit.npz';np.savez_compressed(path,**arrays)
    report.update(status='passed',fitted_factors_sha256=base.sha(path),fit_indices=[r['index'] for r in rows],
        trainable_parameters=sum(p.numel() for p in masters),seconds=time.perf_counter()-started,npu_calls=0,
        policy='CPU differentiable reference, 32 calibration trajectories, 2 fixed epochs. Only QKV rank32 up factors. '
        'FP32 masters are rounded to Half on every forward; INT8 main, scales, down and other factors remain fixed. '
        'AdamW lr=2e-6 weight_decay=0; fixed scale512, grad clip1, relative anchor penalty1e-5. '
        'Objective mean(relative logits L2 squared + .1 KL + 2 relu(L2 squared-.03 squared)). '
        'No independent logits used in fitting and no NPU-training claim.')
    base.write('logits-fit',report)


def parameters(profile):
    if profile not in ('baseline',*PROFILES):raise ValueError('Unknown logits profile')
    packed=base.mixed.weights();factors=base.corrections(32)
    if profile=='baseline':return packed,factors
    metadata=base.read('logits-fit');path=OUT/'logits-fit.npz'
    assert metadata['status']=='passed' and all(metadata[k]==v for k,v in source_identity().items())
    assert base.sha(path)==metadata['fitted_factors_sha256']
    with np.load(path,allow_pickle=False) as arrays:
        assert len(arrays.files)==180
        for i in range(30):
            for name in NAMES:
                old=factors[i][name];up=arrays[f'{profile}/{i}/{name}/up']
                assert up.dtype==np.float16 and up.shape==(old.shape[0],32)
                factors[i][name]=LowRankCorrection(old.shape,32,old.down,up.tobytes())
    return packed,factors


def stack_for(profile):
    from torch_npu_smollm import FrozenSmolStack
    packed,factors=parameters(profile)
    return FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,packed_weights=packed,corrections=factors)


def internal(profile):
    runner=study.previous;old_stack,old_identity=runner.stack_for,runner.identity
    runner.stack_for=stack_for;runner.identity=identity
    try:return runner.internal(profile)
    finally:runner.stack_for=old_stack;runner.identity=old_identity


def select():
    records={p:base.read('internal-'+p) for p in ('baseline',*PROFILES)};current=identity()
    assert all(all(r[k]==v for k,v in current.items()) for r in records.values())
    selected,candidate,rejected=study.previous.choose(records)
    base.write('selection',dict(status='frozen',selected=selected,candidate=candidate,rejected=rejected,**current,
        summaries={p:r['summary'] for p,r in records.items() if 'summary' in r},
        records_sha256={p:base.sha(OUT/('internal-'+p+'.json')) for p in records},
        policy='Unchanged internal gates and promotion rule. Freeze the better of two fixed epochs before new fresh NPU evaluation. '
        'Fresh metrics never select or refit parameters; an unpromoted candidate remains diagnostic only.'))
    print('LOGITS SELECTED',selected,'CANDIDATE',candidate,flush=True)


def frozen_selection():
    row=base.read('selection');assert row['status']=='frozen' and all(row[k]==v for k,v in identity().items())
    assert row['candidate'] in (*PROFILES,None) and row['selected'] in ('baseline',*PROFILES)
    for p,digest in row['records_sha256'].items():assert base.sha(OUT/('internal-'+p+'.json'))==digest
    return row


def fresh(label):
    selection=frozen_selection();profile='baseline' if label=='baseline' else selection['candidate']
    if profile is None:raise ValueError('No internally valid candidate for fresh evaluation')
    import torch_intel_npu as npu
    report=dict(status='running',label=label,profile=profile,positions=[],local_checks=[],
        selection_sha256=base.sha(OUT/'selection.json'),**identity())
    def work():
        cpu,_=base.mixed.common.model();stack=stack_for(profile);cases=base.read('fresh')['cases']
        for index,row in enumerate(cases):
            ids=torch.tensor(row['token_ids']);offset=row['offset'];reference=cpu(ids,use_cache=False).logits[:,offset:].float()
            inputs=cpu.model.embed_tokens(ids).contiguous().to('intel_npu');x=inputs
            if index==0:
                for i,layer in enumerate(stack.layers):
                    expected,_=layer.reference_cpu(x.cpu());x=layer(x);error=study.device.compare(x.cpu(),expected)
                    report['local_checks'].append(dict(layer=i,**error))
                    if error['relative_l2']>=.003 or error['max_scaled_error']>=.01:
                        report['local_guard_failure']=dict(layer=i,**error);raise ValueError(('Unchanged fresh local guard',profile,i,error))
            else:x=stack(x)
            actual=cpu.lm_head(cpu.model.norm(x.cpu())).float()[:,offset:]
            for j in range(actual.shape[1]):report['positions'].append(dict(tag=row['tag'],position=offset+j,**base.mixed.metric(actual[:,j],reference[:,j])))
            print('fresh logits',label,row['tag'],flush=True)
        assert len(report['positions'])==sum(64-r['offset'] for r in cases)
        before=npu.stats();again=stack(inputs);after=npu.stats();torch.testing.assert_close(again.cpu(),x.cpu(),rtol=0,atol=0)
        assert after['npu_calls']-before['npu_calls']==30 and all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
        report.update(summary=base.mixed.common.summary(report['positions']),stats=npu.stats(),warm_calls=30)
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return study.device.finish(report,'fresh-'+label,npu)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','fresh-data','fit','internal','select','fresh'))
    parser.add_argument('--profile',choices=('baseline',*PROFILES),default='baseline')
    parser.add_argument('--label',choices=('baseline','candidate'),default='candidate')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    if args.stage=='fit':result=fit()
    else:
        with torch.no_grad():
            if args.stage=='prepare':previous.prepare();result=fresh_data()
            elif args.stage=='fresh-data':result=fresh_data()
            elif args.stage=='internal':result=internal(args.profile)
            elif args.stage=='fresh':result=fresh(args.label)
            else:result=select()
    raise SystemExit(result or 0)

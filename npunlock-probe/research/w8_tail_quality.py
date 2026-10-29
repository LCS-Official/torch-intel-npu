"""S55: bounded upper-tail logits/KL fitting with disjoint prompt groups."""
import argparse
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import w8_logits_quality as previous
from torch_npu_lowbit_correction import LowRankCorrection

base=previous.base
study=previous.study
ROOT=previous.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-tail-quality'
previous.OUT=previous.previous.OUT=study.OUT=study.previous.OUT=study.previous.previous.OUT=study.probe.OUT=study.device.OUT=study.device.previous.OUT=base.OUT=base.mixed.OUT=OUT
PROFILES=('tail-e1','tail-e2')
REGRESSION=previous.FRESH
CALIBRATION=(
    ('river','Explain why a river can develop bends instead of flowing straight.'),
    ('leaf','Describe the job of a leaf on a growing tree.'),
    ('recipe','Give three steps for making a simple fruit salad.'),
    ('queue','Explain why people form a queue at a busy ticket counter.'),
    ('raincoat','Compare a raincoat with an umbrella in two sentences.'),
    ('sum','A tray has eight cups. Three more are added. How many cups are there?'),
    ('multiply','What is six times seven? Answer with a number and a short explanation.'),
    ('subset','Every oak is a tree. Is every tree necessarily an oak? Explain.'),
    ('code','Write a Python expression that takes the first three items of a list called data.'),
    ('string','In Python, how would you join the words red and blue with a hyphen?'),
    ('schedule','Suggest a simple plan for tidying a desk before studying.'),
    ('story','Write two sentences about a traveler who finds a small wooden bridge.'),
    ('grammar','Rewrite this sentence correctly: She have two green notebooks.'),
    ('compare','Compare a handwritten note with a text message.'),
    ('sound','Why can a large empty room produce an echo?'),
    ('frost','Describe what frost on a cold window looks like.'))
FRESH=(
    ('shadow','Explain why a shadow changes length during a day.'),
    ('sentence','Turn this list into a sentence: bread, apples, and tea.'),
    ('marbles','A box holds six red marbles and four blue ones. How many marbles are there?'),
    ('robot','Suggest a short title for a story about a lost robot.'),
    ('map','What is the difference between a map and a photograph?'),
    ('invitation','Write a friendly invitation to a neighborhood chess evening.'),
    ('bicycle','Explain why a bicycle needs a chain.'),
    ('team','Describe a fair way for three people to share a task.'))


def components(actual,target):
    if (not isinstance(actual,torch.Tensor) or not isinstance(target,torch.Tensor)
        or actual.shape!=target.shape or actual.ndim!=2 or not all(actual.shape)
        or actual.dtype!=torch.float32 or target.dtype!=torch.float32
        or actual.device.type!='cpu' or target.device.type!='cpu'
        or not torch.isfinite(actual).all() or not torch.isfinite(target).all()):
        raise ValueError('Requires finite matching CPU FP32 position-by-vocabulary logits')
    target=target.detach()
    l2=(actual-target).square().sum(-1)/target.square().sum(-1).clamp_min(1e-20)
    kl=F.kl_div(actual.log_softmax(-1),target.log_softmax(-1),reduction='none',log_target=True).sum(-1).clamp_min(0)
    return torch.stack((l2,kl),-1)


def tail_loss(actual,target,budget):
    values=components(actual,target)
    if (not isinstance(budget,torch.Tensor) or budget.shape!=values.shape or budget.dtype!=torch.float32
        or budget.device.type!='cpu' or not torch.isfinite(budget).all() or (budget<0).any()):
        raise ValueError('Requires matching finite nonnegative baseline budgets')
    l2,kl=values.unbind(-1);count=max(1,(len(l2)+9)//10)
    mean=(l2+.1*kl+2*torch.relu(l2-.03**2)).mean()
    tail=.5*(l2.topk(count).values.mean()+kl.topk(count).values.mean())
    # The symmetric subgradient also protects positions exactly on the budget.
    preservation=2*torch.maximum(values-budget.detach(),torch.zeros_like(values)).mean(0).sum()
    return mean+tail+preservation


def prompt_groups():
    groups=(CALIBRATION,REGRESSION,FRESH);texts=[{p for _,p in group} for group in groups]
    if tuple(map(len,groups))!=(16,8,8) or any(len(g)!=len(t) or len({tag for tag,_ in g})!=len(g) for g,t in zip(groups,texts)):
        raise ValueError('Expected distinct calibration/regression/fresh prompts')
    if any(texts[i]&texts[j] for i in range(3) for j in range(i)):raise ValueError('Prompt groups overlap')
    return groups


def data():
    if any((OUT/(name+'.json')).exists() for name in ('tail-data','regression','fresh')):
        raise FileExistsError('Refusing to overwrite frozen prompt groups')
    prompt_groups();system='You are a helpful assistant.'
    old=set(base.mixed.PROMPTS)|{(s,p) for _,s,p in (*base.mixed.HOLDOUT,*base.mixed.FRESH)}
    try:
        from w8_guarded_quality import FRESH as old_fresh
    finally:base.OUT=base.mixed.OUT=OUT
    old|={(system,p) for _,p in old_fresh}
    assert all((system,p) not in old for _,p in (*CALIBRATION,*FRESH))
    cpu,tokenizer=base.mixed.common.model()
    for group,prompts in zip(('tail-data','regression','fresh'),prompt_groups(),strict=True):
        cases=[]
        for index,(tag,prompt) in enumerate(prompts):
            ids=base.mixed.common.chat(tokenizer,system,prompt);assert 1<=ids.shape[1]<64
            tokens=base.mixed.common.trajectory(cpu,ids,64)
            cases.append(dict(tag=tag,prompt=prompt,offset=ids.shape[1]-1,prefix=ids.shape[1],
                index=40+index,token_ids=tokens.tolist()))
        base.write(group,dict(status='frozen',group=group,cases=cases,source_sha256=base.sha(Path(__file__)),
            policy='Freeze all groups before fitting; only tail-data augments calibration. Regression is already viewed S54 data.'))


def source_identity():
    return dict(source_sha256=base.sha(Path(__file__)),helper_sha256=base.sha(Path(previous.__file__)),
        weights_sha256=base.read('prepare')['weights_sha256'],factors_sha256=base.read('factors')['factor_sha256'],
        fresh_sha256=base.sha(OUT/'fresh.json'),regression_sha256=base.sha(OUT/'regression.json'),
        calibration_sha256=base.sha(OUT/'tail-data.json'))


def fit_rows(original,added):
    if (len(original)!=32 or [r['index'] for r in original]!=[i for i in range(40) if i%5!=4]
        or len(added)!=16 or [r['index'] for r in added]!=list(range(40,56))):
        raise ValueError('Expected disjoint original32 and added16 calibration rows')
    return [row for i in range(16) for row in (*original[2*i:2*i+2],added[i])]


def trainable_outputs(model,modules):
    model.requires_grad_(False);result=[]
    for index,(name,module) in enumerate(modules):
        module.rank=32;module.protected=False
        if module.up.dtype!=torch.float16 or module.up.shape[1]!=32:raise ValueError('Expected rank32 Half factors')
        prior=module.up.detach().float().clone();master=torch.nn.Parameter(prior.clone())
        result.append((index//7,name,module,master,prior))
    if len(result)!=210:raise ValueError('Expected all 210 output factors')
    return result


def fit():
    if (OUT/'logits-fit.npz').exists():raise FileExistsError('Refusing to overwrite a tail fit')
    report=dict(status='running',steps=[],checkpoints={},**source_identity())
    cpu,_=base.mixed.common.model();rows=fit_rows(previous.previous.partition(base.read('prepare')['trajectories'],True),base.read('tail-data')['cases'])
    assert len(rows)==48 and len({r['index'] for r in rows})==48 and not {r['index'] for r in rows}&set(range(4,40,5))
    cases=[]
    with torch.no_grad():
        for row in rows:
            ids=torch.tensor(row['token_ids']);offset=row['prefix']-1
            cases.append([row['index'],ids,offset,cpu(ids,use_cache=False).logits[:,offset:].float().detach(),None])
    modules=base.install(cpu);tunables=trainable_outputs(cpu,modules);masters=[r[3] for r in tunables]
    previous.bind_outputs(tunables)
    with torch.no_grad():
        for case in cases:
            _,ids,offset,target,_=case;case[4]=components(cpu(ids,use_cache=False).logits[:,offset:].float()[0],target[0]).detach()
    def score():
        objectives=[];values=[]
        with torch.no_grad():
            for _,ids,offset,target,budget in cases:
                actual=cpu(ids,use_cache=False).logits[:,offset:].float()[0]
                objectives.append(float(tail_loss(actual,target[0],budget)));values.append(components(actual,target[0]))
        merged=torch.cat(values)
        return dict(objective=float(np.mean(objectives)),positions=len(merged),mean_relative_l2=float(merged[:,0].sqrt().mean()),
            max_relative_l2=float(merged[:,0].sqrt().max()),max_kl=float(merged[:,1].max()))
    report['before']=score();current=report['before']
    anchor=sum(float(prior.square().sum()) for *_,prior in tunables);arrays={};started=time.perf_counter()
    report['updates']=[]
    for epoch,profile in enumerate(PROFILES,1):
        for master in masters:master.grad=None
        previous.bind_outputs(tunables)
        for index,ids,offset,target,budget in cases:
            actual=cpu(ids,use_cache=False).logits[:,offset:].float()[0];objective=tail_loss(actual,target[0],budget)
            penalty=sum((master-prior).square().sum() for _,_,_,master,prior in tunables)/max(anchor,1e-20)
            ((objective+1e-5*penalty)*512/len(cases)).backward()
            report['steps'].append(dict(epoch=epoch,trajectory=index,objective=float(objective.detach())))
            print('tail gradient',len(report['steps']),'/96',report['steps'][-1],flush=True)
        for master in masters:
            if master.grad is None or not torch.isfinite(master.grad).all():raise ValueError('Missing or nonfinite gradient')
            master.grad.div_(512)
        norm=float(torch.nn.utils.clip_grad_norm_(masters,1.,error_if_nonfinite=True))
        saved=[master.detach().clone() for master in masters];proposals=[];accepted=False
        for relative in (.001,.00025,.0000625,.000015625):
            step=relative*anchor**.5/max(min(norm,1.),1e-30)
            with torch.no_grad():
                for master,original in zip(masters,saved,strict=True):master.copy_(original-step*master.grad)
            previous.bind_outputs(tunables);candidate=score()
            accepted=candidate['objective']<current['objective']
            proposals.append(dict(relative_step=relative,accepted=accepted,**candidate))
            print('tail proposal',profile,proposals[-1],flush=True)
            if accepted:current=candidate;break
        if not accepted:
            with torch.no_grad():
                for master,original in zip(masters,saved,strict=True):master.copy_(original)
        previous.bind_outputs(tunables)
        for i,name,_,master,_ in tunables:arrays[f'{profile}/{i}/{name}/up']=master.detach().half().numpy().copy()
        report['updates'].append(dict(epoch=epoch,grad_norm=norm,accepted=accepted,proposals=proposals))
        report['checkpoints'][profile]=current;print('tail checkpoint',profile,current,flush=True)
    assert len(arrays)==420 and len(report['steps'])==96
    path=OUT/'logits-fit.npz';np.savez_compressed(path,**arrays)
    report.update(status='passed',fitted_factors_sha256=base.sha(path),fit_indices=[r['index'] for r in rows],
        trainable_parameters=sum(p.numel() for p in masters),seconds=time.perf_counter()-started,npu_calls=0,
        budget_sha256=hashlib.sha256(b''.join(case[4].numpy().tobytes() for case in cases)).hexdigest(),
        policy='48 calibration trajectories, 2 full-gradient updates, CPU preparation only. All 210 rank32 output factors; Half forward. '
        'S54 mean objective + .5 top10% L2 squared + .5 top10% teacher KL + 2 mean positive per-position excess over CPU baseline. '
        'Budgets are detached and fixed; symmetric subgradient at zero excess. Interleave two original and one added prompt. '
        'Scale512, clip1, relative anchor1e-5. Full-gradient direction; try relative Frobenius radii .001/.00025/.0000625/.000015625. '
        'Accept strictly lower calibration objective, otherwise restore all masters. '
        'No regression or fresh logits in fitting; independent metrics never refit or select parameters.')
    base.write('logits-fit',report)


def parameters(profile):
    if profile not in ('baseline',*PROFILES):raise ValueError('Unknown tail profile')
    packed=base.mixed.weights();factors=base.corrections(32)
    if profile=='baseline':return packed,factors
    metadata=base.read('logits-fit');path=OUT/'logits-fit.npz'
    assert metadata['status']=='passed' and all(metadata[k]==v for k,v in source_identity().items())
    assert base.sha(path)==metadata['fitted_factors_sha256']
    with np.load(path,allow_pickle=False) as arrays:
        assert len(arrays.files)==420
        for i in range(30):
            for name in base.mixed.NAMES:
                old=factors[i][name];up=arrays[f'{profile}/{i}/{name}/up']
                assert up.dtype==np.float16 and up.shape==(old.shape[0],32)
                factors[i][name]=LowRankCorrection(old.shape,32,old.down,up.tobytes())
    return packed,factors


def evaluate_group(label,group):
    selection=previous.frozen_selection();profile='baseline' if label=='baseline' else selection['candidate']
    if group not in ('regression','fresh') or profile is None:raise ValueError('Invalid frozen evaluation')
    import torch_intel_npu as npu
    report=dict(status='running',group=group,label=label,profile=profile,positions=[],local_checks=[],
        selection_sha256=base.sha(OUT/'selection.json'),**previous.identity())
    def work():
        cpu,_=base.mixed.common.model();stack=previous.stack_for(profile);cases=base.read(group)['cases']
        for index,row in enumerate(cases):
            ids=torch.tensor(row['token_ids']);offset=row['offset'];reference=cpu(ids,use_cache=False).logits[:,offset:].float()
            inputs=cpu.model.embed_tokens(ids).contiguous().to('intel_npu');x=inputs
            if index==0:
                for i,layer in enumerate(stack.layers):
                    expected,_=layer.reference_cpu(x.cpu());x=layer(x);error=study.device.compare(x.cpu(),expected)
                    report['local_checks'].append(dict(layer=i,**error))
                    if error['relative_l2']>=.003 or error['max_scaled_error']>=.01:
                        report['local_guard_failure']=dict(layer=i,**error);raise ValueError(('Unchanged local guard',profile,i,error))
            else:x=stack(x)
            actual=cpu.lm_head(cpu.model.norm(x.cpu())).float()[:,offset:]
            for j in range(actual.shape[1]):report['positions'].append(dict(tag=row['tag'],position=offset+j,**base.mixed.metric(actual[:,j],reference[:,j])))
            print('tail evaluation',group,label,row['tag'],flush=True)
        assert len(report['positions'])==sum(64-r['offset'] for r in cases)
        before=npu.stats();again=stack(inputs);after=npu.stats();torch.testing.assert_close(again.cpu(),x.cpu(),rtol=0,atol=0)
        assert after['npu_calls']-before['npu_calls']==30 and all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
        report.update(summary=base.mixed.common.summary(report['positions']),stats=npu.stats(),warm_calls=30)
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return study.device.finish(report,group+'-'+label,npu)


previous.PROFILES=PROFILES
previous.source_identity=source_identity
previous.parameters=parameters

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','data','fit','internal','select','evaluate'))
    parser.add_argument('--profile',choices=('baseline',*PROFILES),default='baseline')
    parser.add_argument('--label',choices=('baseline','candidate'),default='candidate')
    parser.add_argument('--group',choices=('regression','fresh'),default='fresh')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    if args.stage=='fit':result=fit()
    else:
        with torch.no_grad():
            if args.stage=='prepare':previous.previous.prepare();result=data()
            elif args.stage=='data':result=data()
            elif args.stage=='internal':result=previous.internal(args.profile)
            elif args.stage=='evaluate':result=evaluate_group(args.label,args.group)
            else:result=previous.select()
    raise SystemExit(result or 0)

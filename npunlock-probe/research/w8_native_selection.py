"""S56: select bounded correction updates on real-NPU calibration outputs."""
import argparse
import math
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import w8_tail_quality as previous
from torch_npu_lowbit_correction import LowRankCorrection

base=previous.base
parent=previous.previous
PARENT_FROZEN=parent.frozen_selection
study=previous.study
ROOT=previous.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-native-selection'
previous.OUT=parent.OUT=parent.previous.OUT=study.OUT=study.previous.OUT=study.previous.previous.OUT=study.probe.OUT=study.device.OUT=study.device.previous.OUT=base.OUT=base.mixed.OUT=OUT
PROPOSALS={f'{seed}-a{percent}':(seed,percent/100) for seed in ('tail-e1','tail-e2') for percent in (25,50,100)}
PROFILES=tuple(PROPOSALS)
REGRESSION=(*previous.REGRESSION,*previous.FRESH)
FRESH=(
    ('zipper','Describe how a zipper closes a jacket.'),
    ('magnet','Why does a magnet stick to a steel door?'),
    ('books','A shelf has five red books and seven green books. How many books are there?'),
    ('thanks','Write a brief thank-you note to a neighbor who watered your plants.'),
    ('seesaw','Explain how a seesaw works.'),
    ('dawn','Turn the words quiet, lake, and dawn into one descriptive sentence.'),
    ('station','How is a train station different from an airport?'),
    ('postcards','Suggest three ways to organize a collection of postcards.'))


def data():
    if any((OUT/(n+'.json')).exists() for n in ('tail-data','regression','fresh')):raise FileExistsError('Frozen data already exists')
    groups=(previous.CALIBRATION,REGRESSION,FRESH);texts=[{p for _,p in g} for g in groups]
    assert tuple(map(len,groups))==(16,16,8) and all(len(g)==len(t) for g,t in zip(groups,texts,strict=True))
    assert not any(texts[i]&texts[j] for i in range(3) for j in range(i))
    system='You are a helpful assistant.'
    old=set(base.mixed.PROMPTS)|{(s,p) for _,s,p in (*base.mixed.HOLDOUT,*base.mixed.FRESH)}
    try:
        from w8_guarded_quality import FRESH as old_fresh
    finally:base.OUT=base.mixed.OUT=OUT
    old|={(system,p) for _,p in old_fresh}
    assert all((system,p) not in old for _,p in FRESH)
    cpu,tokenizer=base.mixed.common.model()
    for group,prompts in zip(('tail-data','regression','fresh'),groups,strict=True):
        cases=[]
        for i,(tag,prompt) in enumerate(prompts):
            ids=base.mixed.common.chat(tokenizer,system,prompt);assert 1<=ids.shape[1]<64
            cases.append(dict(tag=tag,prompt=prompt,index=40+i,prefix=ids.shape[1],offset=ids.shape[1]-1,
                token_ids=base.mixed.common.trajectory(cpu,ids,64).tolist()))
        base.write(group,dict(status='frozen',group=group,cases=cases,source_sha256=base.sha(Path(__file__)),
            policy='Same 48 fit/calibration trajectories as S55; 16 already viewed regression prompts; eight new independent prompts.'))


def source_identity():
    return dict(source_sha256=base.sha(Path(__file__)),fit_helper_sha256=base.sha(Path(previous.__file__)),
        reference_helper_sha256=base.sha(Path(parent.__file__)),weights_sha256=base.read('prepare')['weights_sha256'],
        factors_sha256=base.read('factors')['factor_sha256'],calibration_sha256=base.sha(OUT/'tail-data.json'),
        regression_sha256=base.sha(OUT/'regression.json'),fresh_sha256=base.sha(OUT/'fresh.json'))


def identity():
    result=dict(**source_identity(),fit_sha256=base.sha(OUT/'logits-fit.json'))
    if (OUT/'device-selection.json').exists():result['device_selection_sha256']=base.sha(OUT/'device-selection.json')
    return result


def blend_factor(prior,fitted,alpha):
    if (not isinstance(prior,LowRankCorrection) or not isinstance(fitted,LowRankCorrection)
        or prior.shape!=fitted.shape or prior.rank!=fitted.rank or prior.down!=fitted.down
        or type(alpha) not in (float,int) or alpha not in (.25,.5,1.)):
        raise ValueError('Requires fixed matching input factors and a predefined blend')
    up=(prior.tensors()[1].float()+alpha*(fitted.tensors()[1].float()-prior.tensors()[1].float())).half().contiguous()
    return LowRankCorrection(prior.shape,prior.rank,prior.down,up.numpy().tobytes())


def parameters(profile):
    if profile=='baseline':return previous.parameters('baseline')
    if profile not in PROPOSALS:raise ValueError('Unknown native-selection profile')
    seed,alpha=PROPOSALS[profile];packed,factors=previous.parameters(seed);original=base.corrections(32)
    for i in range(30):
        for name in base.mixed.NAMES:factors[i][name]=blend_factor(original[i][name],factors[i][name],alpha)
    return packed,factors


def calibration_rows():
    return previous.fit_rows(parent.previous.partition(base.read('prepare')['trajectories'],True),base.read('tail-data')['cases'])


def budget_for(baseline,tag,offset,count):
    lookup={}
    for r in baseline['positions']:
        key=(r['tag'],r['position'])
        if key in lookup:raise ValueError('Duplicate native baseline position')
        lookup[key]=r['components']
    try:result=torch.tensor([lookup[(tag,offset+j)] for j in range(count)],dtype=torch.float32)
    except KeyError as error:raise ValueError('Missing native baseline position') from error
    if result.shape!=(count,2) or not torch.isfinite(result).all() or (result<0).any():raise ValueError('Invalid native baseline budget')
    return result


def calibrate(profile):
    if (OUT/'device-selection.json').exists() or (OUT/('calibration-'+profile+'.json')).exists():
        raise FileExistsError('Refusing to overwrite native calibration')
    import torch_intel_npu as npu
    baseline=None if profile=='baseline' else base.read('calibration-baseline')
    current=identity()
    if baseline is not None:
        assert baseline['status']=='passed' and baseline['healthy'] and all(baseline[k]==v for k,v in current.items())
    report=dict(status='running',profile=profile,positions=[],local_checks=[],losses=[],**current)
    if baseline is not None:report['baseline_sha256']=base.sha(OUT/'calibration-baseline.json')
    def work():
        cpu,_=base.mixed.common.model();stack=parent.stack_for(profile);rows=calibration_rows()
        for index,row in enumerate(rows):
            ids=torch.tensor(row['token_ids']);offset=row['prefix']-1;tag=str(row['index'])
            expected=cpu(ids,use_cache=False).logits[:,offset:].float()[0]
            inputs=cpu.model.embed_tokens(ids).contiguous().to('intel_npu');x=inputs
            if index==0:
                for i,layer in enumerate(stack.layers):
                    reference,_=layer.reference_cpu(x.cpu());x=layer(x);error=study.device.compare(x.cpu(),reference)
                    report['local_checks'].append(dict(layer=i,**error))
                    if error['relative_l2']>=.003 or error['max_scaled_error']>=.01:
                        report['local_guard_failure']=dict(layer=i,**error);raise ValueError(('Unchanged native calibration guard',profile,i,error))
            else:x=stack(x)
            actual=cpu.lm_head(cpu.model.norm(x.cpu())).float()[0,offset:]
            values=previous.components(actual,expected)
            budget=values.detach() if baseline is None else budget_for(baseline,tag,offset,len(actual))
            report['losses'].append(float(previous.tail_loss(actual,expected,budget)))
            for j in range(len(actual)):report['positions'].append(dict(tag=tag,position=offset+j,components=values[j].tolist(),
                **base.mixed.metric(actual[j:j+1],expected[j:j+1])))
            print('native calibration',profile,index+1,'/48',report['losses'][-1],flush=True)
        assert len(rows)==48 and len(report['positions'])==1657 and len(report['losses'])==48
        before=npu.stats();again=stack(inputs);after=npu.stats();torch.testing.assert_close(again.cpu(),x.cpu(),rtol=0,atol=0)
        assert after['npu_calls']-before['npu_calls']==30 and all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
        report.update(summary=base.mixed.common.summary(report['positions']),objective=float(np.mean(report['losses'])),stats=npu.stats(),warm_calls=30)
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return study.device.finish(report,'calibration-'+profile,npu)


def choose_calibration(records,positions=1657):
    baseline=records['baseline'];valid=[];rejected={}
    for profile,row in records.items():
        if not row['healthy']:raise ValueError('Unhealthy native calibration: '+profile)
        if row['status']!='passed':
            guard=row.get('local_guard_failure')
            if profile=='baseline' or row['status']!='failed' or not guard or not (guard['relative_l2']>=.003 or guard['max_scaled_error']>=.01):
                raise ValueError('Unexpected native calibration failure: '+profile)
            rejected[profile]=guard;continue
        if (row['summary']['positions']!=positions or not math.isfinite(row['objective']) or row['objective']<0):
            raise ValueError('Incomplete native calibration: '+profile)
        if profile!='baseline':valid.append(profile)
    candidate=min(valid,key=lambda p:(records[p]['objective'],*base.mixed.common.ranking(records[p]['summary']),p)) if valid else None
    selected=candidate if candidate is not None and records[candidate]['objective']<baseline['objective'] else 'baseline'
    return selected,candidate,rejected


def select_device():
    if (OUT/'device-selection.json').exists():raise FileExistsError('Native calibration choice is already frozen')
    records={p:base.read('calibration-'+p) for p in ('baseline',*PROFILES)};current=identity()
    assert all(all(r[k]==v for k,v in current.items()) for r in records.values())
    for p,r in records.items():
        if p!='baseline':assert r['baseline_sha256']==base.sha(OUT/'calibration-baseline.json')
    selected,candidate,rejected=choose_calibration(records)
    base.write('device-selection',dict(status='frozen',selected=selected,candidate=candidate,rejected=rejected,**current,
        objectives={p:r.get('objective') for p,r in records.items()},summaries={p:r.get('summary') for p,r in records.items()},
        records_sha256={p:base.sha(OUT/('calibration-'+p+'.json')) for p in records},
        policy='Six fixed CPU update blends; choose minimum real-NPU calibration objective on original48 only. '
        'Original local guard rejects candidates; unknown failures remain fatal. Freeze before internal/regression/fresh evaluation.'))
    print('DEVICE CHOICE',selected,'DIAGNOSTIC CANDIDATE',candidate,flush=True)


def device_choice():
    row=base.read('device-selection');assert row['status']=='frozen'
    current=dict(**source_identity(),fit_sha256=base.sha(OUT/'logits-fit.json'))
    assert all(row[k]==v for k,v in current.items()) and row['candidate'] in (*PROFILES,None)
    for p,digest in row['records_sha256'].items():assert base.sha(OUT/('calibration-'+p+'.json'))==digest
    return row


def internal(profile):
    choice=device_choice()
    if profile not in ('baseline',choice['candidate']):raise ValueError('Only the frozen native calibration candidate may be evaluated')
    return parent.internal(profile)


def frozen_selection():
    device_choice()
    return PARENT_FROZEN()


def select_internal():
    if (OUT/'selection.json').exists():raise FileExistsError('Internal choice is already frozen')
    choice=device_choice();candidate=choice['candidate']
    if candidate is None:raise ValueError('No locally valid native calibration candidate')
    records={p:base.read('internal-'+p) for p in ('baseline',candidate)};current=identity()
    assert all(all(r[k]==v for k,v in current.items()) for r in records.values())
    selected,valid,rejected=study.previous.choose(records)
    if choice['selected']=='baseline':selected='baseline'
    base.write('selection',dict(status='frozen',selected=selected,candidate=valid,rejected=rejected,**current,
        summaries={p:r.get('summary') for p,r in records.items()},
        records_sha256={p:base.sha(OUT/('internal-'+p+'.json')) for p in records},
        policy='One candidate chosen before internal metrics; unchanged S51 internal gates and promotion rule. '
        'No promotion without real-NPU calibration improvement; regression/fresh never select or refit parameters.'))
    print('INTERNAL CHOICE',selected,'DIAGNOSTIC CANDIDATE',valid,flush=True)


parent.PROFILES=PROFILES
parent.source_identity=source_identity
parent.identity=identity
parent.parameters=parameters
parent.frozen_selection=frozen_selection
previous.source_identity=source_identity

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','data','fit','calibrate','select-device','internal','select','evaluate','cpu-audit'))
    parser.add_argument('--profile',choices=('baseline',*PROFILES),default='baseline')
    parser.add_argument('--group',choices=('regression','fresh'),default='fresh')
    parser.add_argument('--label',choices=('baseline','candidate'),default='candidate')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    if args.stage=='fit':result=previous.fit()
    else:
        with torch.no_grad():
            if args.stage=='prepare':parent.previous.prepare();result=data()
            elif args.stage=='data':result=data()
            elif args.stage=='calibrate':result=calibrate(args.profile)
            elif args.stage=='select-device':result=select_device()
            elif args.stage=='internal':result=internal(args.profile)
            elif args.stage=='select':result=select_internal()
            elif args.stage=='evaluate':result=previous.evaluate_group(args.label,args.group)
            else:
                import w8_tail_audit as audit
                audit.study=SimpleNamespace(base=base,OUT=OUT,previous=parent,source_identity=source_identity,parameters=parameters)
                result=audit.main()
    raise SystemExit(result or 0)

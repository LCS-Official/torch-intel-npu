"""S51: bounded row-scale / rank-32 calibration, with unchanged INT8 codes.

Only the original 32 calibration trajectories fit parameters. Two fixed profiles
are selected on the other eight trajectories using actual NPU execution.
"""
import argparse
import hashlib
import inspect
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import w8_gate_experiment as previous
from torch_npu_lowbit import PackedWeight
from torch_npu_lowbit_correction import LowRankCorrection,fit_lowrank_correction

base=previous.base
device=previous.study.device
ROOT=base.mixed.common.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-scale-quality'
previous.OUT=previous.study.OUT=device.OUT=device.previous.OUT=base.OUT=base.mixed.OUT=OUT
NAMES=tuple('mlp.'+name+'_proj.weight' for name in ('gate','up','down'))
PROFILES=('gate-up','ffn')


@torch.no_grad()
def refine_scales(weight,packed,inputs,correction):
    """Two bounded least-squares scale updates, alternating with rank refits.

    Codes and rank stay fixed. Stored FP16 scale candidates include the nearest
    value and its two neighbors, within +/-2% of the original scale. Accept
    scales rowwise and refits globally only when staged calibration SSE falls.
    This independently implemented coordinate fit is CPU preparation only.
    """
    if (not isinstance(packed,PackedWeight) or packed.format!='int8' or packed.group_size!=packed.shape[1]
        or not isinstance(correction,LowRankCorrection) or correction.shape!=packed.shape or correction.rank>32
        or not isinstance(weight,torch.Tensor) or weight.device.type!='cpu' or tuple(weight.shape)!=packed.shape
        or weight.dtype not in (torch.float16,torch.float32) or not torch.isfinite(weight).all()
        or not isinstance(inputs,torch.Tensor) or inputs.device.type!='cpu' or inputs.dtype not in (torch.float16,torch.float32)
        or inputs.ndim!=2 or inputs.shape[1]!=packed.shape[1] or not 1<=len(inputs)<=4096
        or max(packed.shape)>4096 or not torch.isfinite(inputs).all()):
        raise ValueError('Requires bounded finite matching CPU W8 weights, inputs and rank<=32 correction')
    codes=torch.from_numpy(np.frombuffer(packed.data,dtype=np.int8).reshape(packed.shape).astype(np.float64))
    original=torch.from_numpy(np.frombuffer(packed.scales,dtype=np.float16).astype(np.float64))
    projected=inputs.double()@codes.t();energy=projected.square().sum(0)
    target=F.linear(inputs.float(),weight.float()).half().double()
    def delta(factor):
        a,b=factor.tensors()
        return F.linear(F.linear(inputs.float(),a.float()).half().float(),b.float()).half()
    def loss(scales,correction_value):
        prediction=((projected*scales[None]).half()+correction_value).half().double()
        if not torch.isfinite(prediction).all():raise ValueError('Staged calibration output overflows FP16')
        return (prediction-target).square().sum(0)
    if not torch.isfinite(target).all():raise ValueError('Calibration target overflows FP16')
    scale=original.clone();factor=correction;before=loss(scale,delta(factor));steps=[]
    for _ in range(2):
        correction_value=delta(factor);best=loss(scale,correction_value)
        optimal=((projected*(target-correction_value.double())).sum(0)+.01*energy*scale)/(1.01*energy.clamp_min(1e-300))
        optimal=torch.where(energy>0,optimal,scale).clamp(original*.98,original*1.02)
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            rounded=optimal.numpy().astype(np.float16)
            trials=(rounded,np.nextafter(rounded,np.float16(0)),np.nextafter(rounded,np.float16(np.inf)))
        for trial in trials:
            proposed=torch.from_numpy(trial.astype(np.float64))
            valid=torch.isfinite(proposed)&(proposed>0)&(proposed>=original*.98)&(proposed<=original*1.02)&(energy>0)
            proposed=torch.where(valid,proposed,scale)
            candidate=loss(proposed,correction_value);better=candidate<best
            scale=torch.where(better,proposed,scale);best=torch.where(better,candidate,best)
        updated=PackedWeight(packed.shape,'int8',packed.group_size,packed.data,scale.half().numpy().tobytes())
        fitted=fit_lowrank_correction(weight,updated,inputs,rank=correction.rank)
        candidate=loss(scale,delta(fitted));accepted=bool(candidate.sum()<best.sum())
        if accepted:factor=fitted;best=candidate
        steps.append(dict(sse=float(best.sum()),refit_accepted=accepted))
    final=PackedWeight(packed.shape,'int8',packed.group_size,packed.data,scale.half().numpy().tobytes())
    after=loss(scale,delta(factor))
    assert after.sum()<=before.sum() and final.data==packed.data
    return final,factor,dict(before_sse=float(before.sum()),after_sse=float(after.sum()),steps=steps,
        changed_rows=int((scale!=original).sum()),rows=len(scale),
        min_scale_ratio=float((scale/original).min()),max_scale_ratio=float((scale/original).max()))


def prepare():
    if (OUT/'weights.npz').exists():raise FileExistsError('Refusing to overwrite existing preparation')
    base.prepare()


def fit():
    if (OUT/'scale-fit.npz').exists():raise FileExistsError('Refusing to overwrite existing scale fit')
    cpu,_=base.mixed.common.model();packed=base.mixed.weights();factors=base.corrections(32)
    rows=[r for r in base.read('prepare')['trajectories'] if r['index']%5!=4]
    assert len(rows)==32
    hidden=cpu.model.embed_tokens(torch.cat([torch.tensor(r['token_ids']) for r in rows]));arrays={};records=[]
    started=time.perf_counter()
    for i,layer in enumerate(cpu.model.layers):
        captured={name:[] for name in ('norm2','gated')};hooks=[]
        for name,key in (('mlp.gate_proj','norm2'),('mlp.down_proj','gated')):
            def capture(module,args,key=key):captured[key].append(args[0].detach().reshape(-1,args[0].shape[-1]).clone())
            hooks.append(layer.get_submodule(name).register_forward_pre_hook(capture))
        try:hidden=device.previous.run_layer(cpu,layer,hidden)
        finally:
            for hook in hooks:hook.remove()
        observed={key:torch.cat(value) for key,value in captured.items()}
        for name in NAMES:
            x=observed['gated' if name==NAMES[-1] else 'norm2'];assert len(x)==2048
            q,f,record=refine_scales(layer.get_parameter(name).detach(),packed[i][name],x,factors[i][name])
            a,b=f.tensors();arrays[f'{i}/{name}/scales']=np.frombuffer(q.scales,dtype=np.float16)
            arrays[f'{i}/{name}/down']=a.numpy();arrays[f'{i}/{name}/up']=b.numpy()
            records.append(dict(layer=i,name=name,**record))
        print('row-scale fit',i+1,'/30',records[-3:],flush=True)
    path=OUT/'scale-fit.npz';np.savez_compressed(path,**arrays)
    assert len(records)==90 and len(arrays)==270 and 'torch_intel_npu' not in sys.modules
    base.write('scale-fit',dict(status='passed',matrices=records,profiles=list(PROFILES),fit_indices=[r['index'] for r in rows],
        scale_fit_sha256=base.sha(path),weights_sha256=base.read('prepare')['weights_sha256'],
        factors_sha256=base.read('factors')['factor_sha256'],source_sha256=base.sha(Path(__file__)),
        method_sha256=hashlib.sha256(inspect.getsource(refine_scales).encode()).hexdigest(),
        seconds=time.perf_counter()-started,npu_calls=0))


def parameters(profile):
    if profile not in ('baseline',*PROFILES):raise ValueError('Unknown scale profile')
    packed=base.mixed.weights();factors=base.corrections(32)
    if profile=='baseline':return packed,factors
    metadata=base.read('scale-fit');path=OUT/'scale-fit.npz'
    assert metadata['status']=='passed' and metadata['source_sha256']==base.sha(Path(__file__))
    assert metadata['weights_sha256']==base.read('prepare')['weights_sha256']
    assert metadata['factors_sha256']==base.read('factors')['factor_sha256'] and base.sha(path)==metadata['scale_fit_sha256']
    with np.load(path,allow_pickle=False) as arrays:
        assert len(arrays.files)==270
        for i in range(30):
            for name in (NAMES[:2] if profile=='gate-up' else NAMES):
                prefix=f'{i}/{name}/';old=packed[i][name];s=arrays[prefix+'scales'];a=arrays[prefix+'down'];b=arrays[prefix+'up']
                assert s.dtype==a.dtype==b.dtype==np.float16 and s.shape==(old.shape[0],)
                packed[i][name]=PackedWeight(old.shape,'int8',old.group_size,old.data,s.tobytes())
                factors[i][name]=LowRankCorrection(old.shape,32,a.tobytes(),b.tobytes())
    return packed,factors


def stack_for(profile):
    from torch_npu_smollm import FrozenSmolStack
    packed,factors=parameters(profile)
    return FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,packed_weights=packed,corrections=factors)


def identity():
    return dict(source_sha256=base.sha(Path(__file__)),fit_sha256=base.sha(OUT/'scale-fit.json'))


def internal(profile):
    import torch_intel_npu as npu
    report=dict(status='running',profile=profile,positions=[],local_checks=[],**identity())
    def work():
        cpu,_=base.mixed.common.model();stack=stack_for(profile)
        rows=[r for r in base.read('prepare')['trajectories'] if r['index']%5==4]
        assert [r['index'] for r in rows]==[4,9,14,19,24,29,34,39]
        for index,row in enumerate(rows):
            ids=torch.tensor(row['token_ids']);offset=row['prefix']-1
            expected=cpu(ids,use_cache=False).logits[:,offset:].float();input_device=cpu.model.embed_tokens(ids).contiguous().to('intel_npu');x=input_device
            if index==0:
                for i,layer in enumerate(stack.layers):
                    reference,_=layer.reference_cpu(x.cpu());x=layer(x)
                    error=base.mixed.common.errors(x.cpu(),reference)
                    report['local_checks'].append(dict(layer=i,**error))
                    if error['relative_l2']>=.003 or error['max_scaled_error']>=.01:
                        report['local_guard_failure']=dict(layer=i,**error)
                        raise ValueError(('Candidate violates the unchanged local precision gate',profile,i,error))
            else:x=stack(x)
            actual=cpu.lm_head(cpu.model.norm(x.cpu())).float()[:,offset:]
            for j in range(actual.shape[1]):
                report['positions'].append(dict(tag=str(row['index']),position=offset+j,**base.mixed.metric(actual[:,j],expected[:,j])))
            print('scale internal',profile,row['index'],flush=True)
        assert len(report['positions'])==290
        before=npu.stats();started=time.perf_counter();again=stack(input_device);elapsed=time.perf_counter()-started
        after=npu.stats();torch.testing.assert_close(again.cpu(),x.cpu(),rtol=0,atol=0)
        assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
        assert after['npu_calls']-before['npu_calls']==30
        report.update(summary=base.mixed.common.summary(report['positions']),stats=npu.stats(),
            warm=dict(seconds=elapsed,calls=30,new_sessions=0,host_copies=0))
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return device.finish(report,'internal-'+profile,npu)


def choose(records):
    rejected={};eligible=[]
    for profile,row in records.items():
        if not row['healthy']:raise ValueError('Unhealthy device execution: '+profile)
        if row['status']!='passed':
            guard=row.get('local_guard_failure')
            if profile=='baseline' or not guard or not (guard['relative_l2']>=.003 or guard['max_scaled_error']>=.01):
                raise ValueError('Unexpected execution failure: '+profile)
            rejected[profile]=guard
        else:
            if row['summary']['positions']!=290 or len(row['local_checks'])!=30:raise ValueError('Incomplete internal evaluation: '+profile)
            if profile!='baseline':eligible.append(profile)
    if records['baseline']['status']!='passed':raise ValueError('Baseline is not validated')
    candidate=min(eligible,key=lambda p:(*base.mixed.common.ranking(records[p]['summary']),p)) if eligible else None
    selected=candidate if candidate and previous.promotable(records['baseline']['summary'],records[candidate]['summary']) else 'baseline'
    return selected,candidate,rejected


def select():
    records={p:base.read('internal-'+p) for p in ('baseline',*PROFILES)};current=identity()
    assert all(all(row[k]==v for k,v in current.items()) for row in records.values())
    selected,candidate,rejected=choose(records)
    base.write('selection',dict(status='frozen',candidate=candidate,selected=selected,**current,
        summaries={p:r['summary'] for p,r in records.items() if 'summary' in r},rejected=rejected,
        records_sha256={p:base.sha(OUT/('internal-'+p+'.json')) for p in records},
        policy='Two fixed profiles, gate/up or all FFN. Two rounds, +/-2% FP16 row scales, unchanged INT8 codes and rank32. '
        'Fit only 32 calibration trajectories; select on eight internal NPU trajectories. Rank failed/max/mean L2; '
        'Unchanged local guards reject candidates before ranking; unexpected failures remain fatal. '
        'Advance only with more passes and no worse mean/max L2, maximum KL or matching next tokens.'))
    print('SCALE SELECTED',selected,'CANDIDATE',candidate,{p:r.get('summary',r.get('local_guard_failure')) for p,r in records.items()},flush=True)


def frozen_selection():
    row=base.read('selection');assert row['status']=='frozen' and all(row[k]==v for k,v in identity().items())
    assert row['candidate'] in (*PROFILES,None) and row['selected'] in ('baseline',*PROFILES)
    for p,digest in row['records_sha256'].items():assert base.sha(OUT/('internal-'+p+'.json'))==digest
    return row


def apply_profile(modules,profile):
    packed,factors=parameters(profile)
    for index,(name,module) in enumerate(modules):
        i=index//7;q=packed[i][name];a,b=factors[i][name].tensors()
        codes=np.frombuffer(q.data,dtype=np.int8).reshape(q.shape).astype(np.float64)
        scales=np.frombuffer(q.scales,dtype=np.float16).astype(np.float64)
        module.decoded=torch.from_numpy(codes*scales[:,None]);module.down.copy_(a);module.up.copy_(b)
        module.rank=32;module.protected=False


def independent():
    selection=frozen_selection();assert selection['selected']!='baseline'
    from w8_guarded_quality import FRESH
    cpu,tokenizer=base.mixed.common.model();cases=[]
    for tag,system,prompt in (*base.mixed.HOLDOUT,*base.mixed.FRESH,*((tag,'You are a helpful assistant.',prompt) for tag,prompt in FRESH)):
        ids=base.mixed.common.chat(tokenizer,system,prompt);count=8 if tag.startswith('s45-') else 32 if tag in ('bird','garden','cat') else 1
        tokens=base.mixed.common.trajectory(cpu,ids,min(64,len(ids[0])+count-1));offset=len(ids[0])-1
        cases.append((tag,offset,tokens,cpu(tokens,use_cache=False).logits[:,offset:].float()))
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    ids=ids.repeat(1,(32+ids.shape[1])//ids.shape[1])[:,:32];tokens=base.mixed.common.trajectory(cpu,ids,64)
    cases.append(('repeat32',31,tokens,cpu(tokens,use_cache=False).logits[:,31:].float()))
    modules=base.install(cpu);report=dict(status='passed',selection_sha256=base.sha(OUT/'selection.json'),**identity(),
        cases=[dict(tag=t,offset=o,token_ids=ids.tolist()) for t,o,ids,_ in cases],variants={})
    for label,profile in (('baseline','baseline'),('candidate',selection['candidate']),('fp16','baseline')):
        apply_profile(modules,profile);report['variants'][label]={}
        for group,fresh in (('regression',False),('fresh',True)):
            result=base.evaluate(cpu,modules,0 if label=='fp16' else 32,[c for c in cases if c[0].startswith('s45-')==fresh],
                base.mixed.NAMES if label=='fp16' else ())
            assert result['summary']['positions']==(128 if fresh else 142)
            if label=='fp16':assert result['summary']['max_relative_l2']==0
            report['variants'][label][group]=result
            print('scale independent',label,group,result['summary'],flush=True)
    assert 'torch_intel_npu' not in sys.modules;base.write('independent',report)


def hardware(label,route):
    selection=frozen_selection();assert selection['selected']!='baseline'
    packed,factors=parameters(selection['candidate'] if label=='candidate' else 'baseline')
    old_weights=base.mixed.weights;old_factors=device.previous.profile_factors;old_selection=device.previous.frozen_selection
    base.mixed.weights=lambda:packed;device.previous.profile_factors=lambda profile:factors;device.previous.frozen_selection=frozen_selection
    try:result=device.previous.hardware(label,route)
    finally:base.mixed.weights=old_weights;device.previous.profile_factors=old_factors;device.previous.frozen_selection=old_selection
    name='hardware-'+route+'-'+label;report=base.read(name);report.update(identity());base.write(name,report)
    return result


def data():
    assert frozen_selection()['selected']!='baseline'
    base.data()


def cpu_language():
    selection=frozen_selection();assert selection['selected']!='baseline'
    cpu,tokenizer=base.mixed.common.model();variants={'fp16':base.language_quality(cpu,tokenizer)}
    modules=base.install(cpu)
    for label,profile in (('baseline','baseline'),('candidate',selection['candidate'])):
        apply_profile(modules,profile);variants[label]=base.language_quality(cpu,tokenizer)
        print('scale CPU language',label,variants[label]['summary'],flush=True)
    assert 'torch_intel_npu' not in sys.modules
    base.write('language-cpu',dict(status='passed',variants=variants,selection_sha256=base.sha(OUT/'selection.json'),**identity()))


def hardware_language(label):
    selection=frozen_selection();assert selection['selected']!='baseline'
    import torch_intel_npu as npu
    report=dict(status='running',label=label,selection_sha256=base.sha(OUT/'selection.json'),**identity())
    def work():
        cpu,tokenizer=base.mixed.common.model();stack=stack_for(selection['candidate'] if label=='candidate' else 'baseline')
        result=base.language_quality(cpu,tokenizer,stack=stack)
        assert len(result['local_checks'])==30 and len(result['generation'])==len(result['warm'])==8
        report.update(result=result,stats=npu.stats())
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return device.finish(report,'language-npu-'+label,npu)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','fit','internal','select','independent','hardware','data','cpu-language','hardware-language'))
    parser.add_argument('--profile',choices=('baseline',*PROFILES),default='baseline')
    parser.add_argument('--variant',choices=('baseline','candidate','fp16'),default='candidate')
    parser.add_argument('--route',choices=('forward','kv'),default='forward')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():
        if args.stage=='internal':result=internal(args.profile)
        elif args.stage=='hardware':result=hardware(args.variant,args.route)
        elif args.stage=='hardware-language':result=hardware_language(args.variant)
        else:result=globals()[args.stage.replace('-','_')]()
    raise SystemExit(result or 0)

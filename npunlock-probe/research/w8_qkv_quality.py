"""S53: fit the existing Q/K/V output factors using actual NPU observations."""
import argparse
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import w8_attention_quality as study
from torch_npu_lowbit_correction import LowRankCorrection

base=study.base
ROOT=study.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-qkv-quality'
study.OUT=study.previous.OUT=study.previous.previous.OUT=study.probe.OUT=study.device.OUT=study.device.previous.OUT=base.OUT=base.mixed.OUT=OUT
NAMES=tuple('self_attn.'+name+'_proj.weight' for name in ('q','k','v'))
PROFILES=('qkv-0.25','qkv-0.50','qkv-1.00')


def partition(rows,fit):
    if len(rows)!=40 or [r['index'] for r in rows]!=list(range(40)):
        raise ValueError('Expected the original 40 ordered trajectories')
    return [r for r in rows if (r['index']%5!=4)==fit]


@torch.no_grad()
def fit_device_basis(reduced,main,target,prior,alpha):
    """Ridge fit against the FP16 projection; keep the observed input basis.

    The NPU supplies main and reduced. The FP64 solve is preparation only;
    acceptance uses a CPU proxy with the existing two Half output boundaries.
    """
    if (not isinstance(prior,LowRankCorrection) or prior.rank>32 or max(prior.shape)>4096
        or type(alpha) not in (float,int) or alpha not in (.25,.5,1.)
        or any(not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=torch.float16
            or v.ndim!=2 or not torch.isfinite(v).all() for v in (reduced,main,target))
        or not 1<=len(reduced)<=4096 or reduced.shape[1]!=prior.rank
        or main.shape!=target.shape or tuple(main.shape)!=(len(reduced),prior.shape[0])):
        raise ValueError('Requires bounded finite CPU Half NPU observations and a matching rank<=32 prior')
    _,original=prior.tensors();z=reduced.double();y=target.double()-main.double()
    h=z.t()@z/len(z);ridge=.01*h.diagonal().mean().clamp_min(1e-20)
    h.diagonal().add_(ridge)
    cross=y.t()@z/len(z)+ridge*original.double()
    fitted=torch.linalg.solve(h,cross,left=False)
    candidate=(original.double()+alpha*(fitted-original.double())).half().contiguous()
    if not torch.isfinite(candidate).all():raise ValueError('Fitted output factor overflows Half')
    def loss(up):
        output=(main+F.linear(reduced.float(),up.float()).half()).half()
        if not torch.isfinite(output).all():raise ValueError('Staged projection overflows Half')
        return float((output.double()-target.double()).square().sum())
    before=loss(original);proposal=loss(candidate);accepted=proposal<before
    if not accepted:candidate=original
    factor=LowRankCorrection(prior.shape,prior.rank,prior.down,candidate.numpy().tobytes())
    return factor,dict(proxy_before_sse=before,proxy_proposal_sse=proposal,
        proxy_after_sse=proposal if accepted else before,accepted=accepted,
        changed_coefficients=int((candidate!=original).sum()))


def prepare():
    if (OUT/'weights.npz').exists():raise FileExistsError('Refusing to overwrite existing preparation')
    base.prepare()


def source_identity():
    return dict(source_sha256=base.sha(Path(__file__)),weights_sha256=base.read('prepare')['weights_sha256'],
        factors_sha256=base.read('factors')['factor_sha256'])


def identity():return dict(**source_identity(),fit_sha256=base.sha(OUT/'qkv-fit.json'))


def parameters(profile):
    if profile not in ('baseline',*PROFILES):raise ValueError('Unknown QKV profile')
    packed=base.mixed.weights();factors=base.corrections(32)
    if profile=='baseline':return packed,factors
    metadata=base.read('qkv-fit');path=OUT/'qkv-fit.npz'
    assert metadata['status']=='passed' and metadata['healthy']
    assert all(metadata[k]==v for k,v in source_identity().items())
    assert base.sha(path)==metadata['fitted_factors_sha256']
    with np.load(path,allow_pickle=False) as arrays:
        assert len(arrays.files)==270
        for i in range(30):
            for name in NAMES:
                old=factors[i][name];up=arrays[f'{profile}/{i}/{name}/up']
                assert up.dtype==np.float16 and up.shape==(old.shape[0],old.rank)
                factors[i][name]=LowRankCorrection(old.shape,old.rank,old.down,up.tobytes())
    return packed,factors


def stack_for(profile):
    from torch_npu_smollm import FrozenSmolStack
    packed,factors=parameters(profile)
    return FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,packed_weights=packed,corrections=factors)


def calibrate():
    if (OUT/'qkv-fit.npz').exists():raise FileExistsError('Refusing to overwrite a device fit')
    import torch_intel_npu as npu
    report=dict(status='running',layers=[],profiles=list(PROFILES),**source_identity())
    def work():
        cpu,_=base.mixed.common.model();stack=stack_for('baseline')
        rows=partition(base.read('prepare')['trajectories'],True)
        hidden=[cpu.model.embed_tokens(torch.tensor(r['token_ids'])).contiguous().to('intel_npu') for r in rows]
        arrays={};started=time.perf_counter()
        for i,layer in enumerate(stack.layers):
            graph,xml,_=layer.graph((1,64,576),segment='attention')
            recipe=study.attention_probe_graph(graph,xml)
            captured={name:{key:[] for key in ('main','reduced','target','actual')} for name in NAMES}
            next_hidden=[];controls=[]
            for row,x in zip(rows,hidden,strict=True):
                assert tuple(x.shape)==(1,64,576)
                normalized=layer._norm_device(x,'input_layernorm.weight')
                values,_=study.probe.run_graph(npu,recipe,[x,normalized])
                attention=layer._run_fused(x,segment='attention')[0];whole=layer(x)
                split=layer._run_fused(values['output'],segment='ffn')[0]
                checks=dict(attention=study.device.compare(values['output'].cpu(),attention.cpu()),
                    split=study.device.compare(split.cpu(),whole.cpu()))
                if any(r['exact_fraction']!=1 for r in checks.values()):raise ValueError(('Observation changes inference',i,row['index'],checks))
                controls.append(dict(trajectory=row['index'],**checks));next_hidden.append(whole)
                host=normalized.cpu().reshape(-1,576)
                for name in NAMES:
                    key=name.split('.')[1][0];data=captured[name]
                    data['main'].append(values[key+'_main'].cpu());data['reduced'].append(values[key+'_reduced'].cpu())
                    data['actual'].append(values[key+'_corrected'].cpu())
                    data['target'].append(F.linear(host.float(),layer._cpu_weight(name).float()).half())
            hidden=next_hidden;records=[]
            for name,data in captured.items():
                data={key:torch.cat(value) for key,value in data.items()}
                assert all(len(v)==2048 for v in data.values())
                actual_sse=float((data['actual'].double()-data['target'].double()).square().sum())
                digest=hashlib.sha256(b''.join(data[key].numpy().tobytes() for key in sorted(data))).hexdigest()
                for profile in PROFILES:
                    factor,metrics=fit_device_basis(data['reduced'],data['main'],data['target'],layer._corrections[name],float(profile[4:]))
                    assert factor.down==layer._corrections[name].down
                    arrays[f'{profile}/{i}/{name}/up']=factor.tensors()[1].numpy()
                    records.append(dict(name=name,profile=profile,actual_before_sse=actual_sse,observations_sha256=digest,**metrics))
            report['layers'].append(dict(layer=i,controls=controls,matrices=records))
            print('device QKV fit',i+1,'/30',[(r['profile'],r['name'],r['proxy_before_sse'],r['proxy_after_sse']) for r in records],flush=True)
        assert len(arrays)==270 and len(report['layers'])==30
        path=OUT/'qkv-fit.npz';np.savez_compressed(path,**arrays)
        report.update(fit_indices=[r['index'] for r in rows],fit_rows=2048,
            internal_indices=[r['index'] for r in partition(base.read('prepare')['trajectories'],False)],
            fitted_factors_sha256=base.sha(path),stats=npu.stats(),seconds=time.perf_counter()-started,
            policy='Original 32 calibration trajectories only; actual baseline NPU prefixes, main and reduced outputs. '
            'Target original FP16 projection on the same NPU-normalized input. Fixed rank32 input basis, codes and scales. '
            'Ridge 1%, output-factor blend .25/.50/1.00; accept only lower staged CPU-proxy SSE per matrix. '
            'No stored activations and no whole-model quality claim from calibration SSE.')
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return study.device.finish(report,'qkv-fit',npu)


def select():
    records={p:base.read('internal-'+p) for p in ('baseline',*PROFILES)};current=identity()
    assert all(all(row[k]==v for k,v in current.items()) for row in records.values())
    selected,candidate,rejected=study.previous.choose(records)
    base.write('selection',dict(status='frozen',candidate=candidate,selected=selected,**current,rejected=rejected,
        summaries={p:r['summary'] for p,r in records.items() if 'summary' in r},
        records_sha256={p:base.sha(OUT/('internal-'+p+'.json')) for p in records},
        policy='Unchanged S51 NPU local guards and promotion rule. Three predefined output-factor blends; '
        'fit 32 trajectories and select on the other eight. Original FP16 whole-model logits reference; '
        'more passes and no worse mean/max L2, maximum KL or next-token matches. No execution-path change.'))
    print('QKV SELECTED',selected,'CANDIDATE',candidate,{p:r.get('summary',r.get('local_guard_failure')) for p,r in records.items()},flush=True)


def frozen_selection():
    row=base.read('selection');assert row['status']=='frozen' and all(row[k]==v for k,v in identity().items())
    assert row['candidate'] in (*PROFILES,None) and row['selected'] in ('baseline',*PROFILES)
    for p,digest in row['records_sha256'].items():assert base.sha(OUT/('internal-'+p+'.json'))==digest
    return row


def evaluate(stage,*args):
    """Reuse the unchanged S51 guards and independent forward/KV evaluators."""
    previous=study.previous;names=('parameters','stack_for','identity','frozen_selection')
    original={name:getattr(previous,name) for name in names}
    for name in names:setattr(previous,name,globals()[name])
    try:return getattr(previous,stage)(*args)
    finally:
        for name,value in original.items():setattr(previous,name,value)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','calibrate','internal','select','independent','hardware','data','cpu_language','hardware_language'))
    parser.add_argument('--profile',choices=('baseline',*PROFILES),default='baseline')
    parser.add_argument('--label',choices=('baseline','candidate','fp16'),default='candidate')
    parser.add_argument('--route',choices=('forward','decode'),default='forward')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():
        if args.stage in ('prepare','calibrate','select'):result=globals()[args.stage]()
        else:
            operands=(args.profile,) if args.stage=='internal' else (args.label,args.route) if args.stage=='hardware' else (args.label,) if args.stage=='hardware_language' else ()
            result=evaluate(args.stage,*operands)
    raise SystemExit(result or 0)

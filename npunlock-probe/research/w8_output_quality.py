"""S48: fit FFN-down residuals on quantized prefixes, then freeze evaluation.

The existing rank-32 factor API executes every correction on the NPU.
Fit 32 calibration trajectories; select only on the other eight trajectories.
"""
import argparse
import gc
import hashlib
import inspect
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import w8_lowrank_quality as base
from torch_npu_lowbit_correction import LowRankCorrection,fit_lowrank_correction

ROOT=base.mixed.common.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-output-quality'
base.OUT=base.mixed.OUT=OUT
DOWN='mlp.down_proj.weight'
PROFILES=('input-local','block-0.25','block-0.50','block-1.00',
    'basis-local','basis-0.25','basis-0.50','basis-1.00')


def decoded_weight(packed):
    codes=torch.from_numpy(np.frombuffer(packed.data,dtype=np.int8).reshape(packed.shape).astype(np.float64))
    scales=torch.from_numpy(np.frombuffer(packed.scales,dtype=np.float16).astype(np.float64))
    return codes*scales[:,None]


@torch.no_grad()
def fit_output_residual(weight,packed,inputs,target,*,rank=32):
    """Reduced-rank ridge regression, anchored to the original W-W8 residual.

    Minimize ||X C' - target||^2/m + lambda ||C-(W-W8)||^2,
    rank(C)<=rank. The FP16 factors use the existing two-GEMM inference path.
    """
    if (not isinstance(packed,base.mixed.PackedWeight) or packed.format!='int8' or packed.group_size!=packed.shape[1]
        or not isinstance(weight,torch.Tensor) or weight.device.type!='cpu' or tuple(weight.shape)!=packed.shape
        or weight.dtype not in (torch.float16,torch.float32) or not torch.isfinite(weight).all()
        or not isinstance(inputs,torch.Tensor) or inputs.device.type!='cpu' or inputs.dtype not in (torch.float16,torch.float32)
        or inputs.ndim!=2 or inputs.shape[1]!=packed.shape[1] or not 1<=inputs.shape[0]<=4096
        or max(packed.shape)>4096 or not torch.isfinite(inputs).all()
        or not isinstance(target,torch.Tensor) or target.device.type!='cpu' or target.dtype not in (torch.float16,torch.float32,torch.float64)
        or tuple(target.shape)!=(len(inputs),packed.shape[0]) or not torch.isfinite(target).all()
        or type(rank) is not int or not 1<=rank<=min(packed.shape)):
        raise ValueError('Requires bounded finite matching CPU weights, inputs, output residuals and rank')
    prior=weight.detach().double()-decoded_weight(packed);x=inputs.detach().double();y=target.detach().double()
    h=x.t()@x/len(x);ridge=.01*h.diagonal().mean().clamp_min(1e-20)
    h.diagonal().add_(ridge);lower=torch.linalg.cholesky(h)
    cross=y.t()@x/len(x)+ridge*prior
    whitened=torch.linalg.solve_triangular(lower,cross.t(),upper=False).t()
    u,s,vh=torch.linalg.svd(whitened,full_matrices=False);root=s[:rank].sqrt()
    up=u[:,:rank]*root
    down=torch.linalg.solve_triangular(lower.t(),vh[:rank].t(),upper=True).t()*root[:,None]
    # Balance after undoing whitening, including unobserved input directions.
    a=down.abs().amax(1);b=up.abs().amax(0)
    balance=torch.where((a>0)&(b>0),(a/b.clamp_min(1e-300)).sqrt(),torch.ones_like(a))
    down=(down/balance[:,None]).half().contiguous();up=(up*balance).half().contiguous()
    return LowRankCorrection(packed.shape,rank,down.numpy().tobytes(),up.numpy().tobytes())


@torch.no_grad()
def fit_basis_residual(inputs,target,prior):
    """Keep the original input basis; fit only rank-many output coefficients."""
    if (not isinstance(prior,LowRankCorrection) or prior.rank>32 or max(prior.shape)>4096
        or not isinstance(inputs,torch.Tensor) or inputs.device.type!='cpu' or inputs.dtype not in (torch.float16,torch.float32)
        or inputs.ndim!=2 or inputs.shape[1]!=prior.shape[1] or not 1<=len(inputs)<=4096 or not torch.isfinite(inputs).all()
        or not isinstance(target,torch.Tensor) or target.device.type!='cpu' or target.dtype not in (torch.float16,torch.float32,torch.float64)
        or tuple(target.shape)!=(len(inputs),prior.shape[0]) or not torch.isfinite(target).all()):
        raise ValueError('Requires finite matching CPU inputs/targets and a bounded rank<=32 prior')
    down,up=prior.tensors();z=F.linear(inputs.float(),down.float()).half().double()
    h=z.t()@z/len(z);ridge=.01*h.diagonal().mean().clamp_min(1e-20)
    h.diagonal().add_(ridge)
    cross=target.detach().double().t()@z/len(z)+ridge*up.double()
    fitted=torch.linalg.solve(h,cross,left=False).half().contiguous()
    return LowRankCorrection(prior.shape,prior.rank,prior.down,fitted.numpy().tobytes())


def run_layer(cpu,layer,hidden):
    positions=torch.arange(64)[None].expand(8,-1)
    mask=torch.triu(torch.full((64,64),-float('inf'),dtype=torch.float16),diagonal=1)[None,None]
    return torch.cat([layer(chunk,position_ids=positions,attention_mask=mask,
        position_embeddings=cpu.model.rotary_emb(chunk,positions),use_cache=False) for chunk in hidden.split(8)])


def replace_down(modules,factors):
    for i in range(30):
        module=dict(modules[i*7:(i+1)*7])[DOWN]
        down,up=factors[i][DOWN].tensors();module.down.copy_(down);module.up.copy_(up)


def fit():
    cpu,_=base.mixed.common.model();metadata=base.read('prepare')
    fit_rows=[row for row in metadata['trajectories'] if row['index']%5!=4]
    assert len(fit_rows)==32
    ids=torch.cat([torch.tensor(row['token_ids']) for row in fit_rows]);teacher=[cpu.model.embed_tokens(ids)]
    for layer in cpu.model.layers:teacher.append(run_layer(cpu,layer,teacher[-1]))
    torch.testing.assert_close(cpu(ids[:8],use_cache=False).logits,
        cpu.lm_head(cpu.model.norm(teacher[-1][:8])),rtol=0,atol=0)
    modules=base.install(cpu);original=base.corrections(32);packed=base.mixed.weights()
    for _,module in modules:module.rank=32
    reports={};start=time.perf_counter()
    for profile in PROFILES:
        replace_down(modules,original);hidden=teacher[0];arrays={};rows=[]
        alpha=None if profile=='input-local' else 0. if profile=='basis-local' else float(profile.rsplit('-',1)[1])
        for i,layer in enumerate(cpu.model.layers):
            module=dict(modules[i*7:(i+1)*7])[DOWN];captured={'inputs':[],'residual':[]}
            hooks=[module.register_forward_pre_hook(lambda m,args:captured['inputs'].append(args[0].detach().clone())),
                layer.post_attention_layernorm.register_forward_pre_hook(lambda m,args:captured['residual'].append(args[0].detach().clone()))]
            try:before=run_layer(cpu,layer,hidden)
            finally:
                for hook in hooks:hook.remove()
            x=torch.cat(captured['inputs']).reshape(-1,1536);residual=torch.cat(captured['residual']).reshape(-1,576)
            assert len(x)==len(residual)==2048
            if alpha is None:
                correction=fit_lowrank_correction(module.original.weight,packed[i][DOWN],x,rank=32)
            else:
                local=x.double()@(module.original.weight.detach().double()-decoded_weight(packed[i][DOWN])).t()
                plain=base.mixed.ReferenceLinear.forward(module,x)
                block=teacher[i+1].reshape(-1,576).double()-residual.double()-plain.double()
                target=(1-alpha)*local+alpha*block
                correction=fit_basis_residual(x,target,original[i][DOWN]) if profile.startswith('basis-') else \
                    fit_output_residual(module.original.weight,packed[i][DOWN],x,target,rank=32)
            down,up=correction.tensors();module.down.copy_(down);module.up.copy_(up)
            hidden=run_layer(cpu,layer,hidden)
            assert torch.isfinite(hidden).all()
            arrays[f'{i}/down']=down.numpy();arrays[f'{i}/up']=up.numpy()
            reference=teacher[i+1].float()
            error=lambda value:float(torch.linalg.vector_norm(value.float()-reference)/torch.linalg.vector_norm(reference))
            rows.append(dict(layer=i,before=error(before),after=error(hidden)))
            print('fit',profile,i+1,'/30',rows[-1],flush=True)
        path=OUT/(profile+'.npz');np.savez_compressed(path,**arrays)
        reports[profile]=dict(factors_sha256=base.sha(path),layers=rows)
    assert 'torch_intel_npu' not in sys.modules
    base.write('output-fit',dict(status='passed',profiles=reports,fit_rows=2048,fit_indices=[r['index'] for r in fit_rows],
        prior='original W-W8, ridge 1% mean activation energy',rank=32,
        baseline_sha256=base.sha(OUT/'factors.json'),method_sha256=hashlib.sha256((inspect.getsource(fit_output_residual)+inspect.getsource(fit_basis_residual)).encode()).hexdigest(),
        source_sha256=base.sha(Path(__file__)),seconds=time.perf_counter()-start,npu_calls=0))


def profile_factors(profile):
    result=base.corrections(32)
    if profile=='baseline':return result
    metadata=base.read('output-fit')
    assert metadata['status']=='passed' and metadata['baseline_sha256']==base.sha(OUT/'factors.json')
    assert metadata['source_sha256']==base.sha(Path(__file__))
    assert metadata['method_sha256']==hashlib.sha256((inspect.getsource(fit_output_residual)+inspect.getsource(fit_basis_residual)).encode()).hexdigest()
    assert profile in PROFILES and base.sha(OUT/(profile+'.npz'))==metadata['profiles'][profile]['factors_sha256']
    with np.load(OUT/(profile+'.npz'),allow_pickle=False) as arrays:
        assert len(arrays.files)==60
        for i in range(30):
            down,up=arrays[f'{i}/down'],arrays[f'{i}/up']
            assert down.dtype==up.dtype==np.float16 and down.shape==(32,1536) and up.shape==(576,32)
            result[i][DOWN]=LowRankCorrection((576,1536),32,down.tobytes(),up.tobytes())
    return result


def select():
    cpu,_=base.mixed.common.model();cases=[]
    for row in base.read('prepare')['trajectories']:
        if row['index']%5==4:
            ids=torch.tensor(row['token_ids']);offset=row['prefix']-1
            cases.append((str(row['index']),offset,ids,cpu(ids,use_cache=False).logits[:,offset:].float()))
    modules=base.install(cpu);profiles={}
    for profile in ('baseline',*PROFILES):
        replace_down(modules,profile_factors(profile))
        profiles[profile]=base.evaluate(cpu,modules,32,cases)
        print('internal',profile,profiles[profile]['summary'],flush=True)
    rank=lambda profile:(*base.mixed.common.ranking(profiles[profile]['summary']),profile!='baseline',profile)
    selected=min(profiles,key=rank);candidate=min(PROFILES,key=rank)
    control=base.evaluate(cpu,modules,0,cases,base.mixed.NAMES)['summary']
    assert control['max_relative_l2']==0 and control['positions']==290 and 'torch_intel_npu' not in sys.modules
    base.write('selection',dict(status='passed',profiles=profiles,selected=selected,candidate=candidate,fp16_control=control,
        fit_sha256=base.sha(OUT/'output-fit.json'),source_sha256=base.sha(Path(__file__)),npu_calls=0,
        policy='Input-local, full-factor block alpha=.25/.50/1.00, and fixed-basis local/alpha=.25/.50/1.00; '
        'the fixed-basis family was added after internal-only overfit review, before independent evaluation. '
        'Internal 8 trajectories, B1, 290 positions only. '
        'Rank by failed positions/max raw L2/mean raw L2, prefer baseline on ties. Freeze best new candidate '
        'and baseline-vs-candidate choice before independent evaluation; no raw-gate or rank change.'))
    print('SELECTED',selected,'CANDIDATE',candidate,flush=True)


def frozen_selection():
    selection=base.read('selection')
    assert selection['status']=='passed' and selection['source_sha256']==base.sha(Path(__file__))
    assert selection['fit_sha256']==base.sha(OUT/'output-fit.json')
    assert selection['candidate'] in PROFILES and selection['selected'] in ('baseline',*PROFILES)
    return selection


def independent():
    selection=frozen_selection();frozen=base.sha(OUT/'selection.json')
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
    modules=base.install(cpu);report=dict(status='running',selection_sha256=frozen,candidate=selection['candidate'],variants={},
        cases=[dict(tag=t,offset=o,token_ids=ids.tolist()) for t,o,ids,_ in cases])
    for label,profile,rank,protected in (('baseline','baseline',32,()),('candidate',selection['candidate'],32,()),
        ('fp16','baseline',0,base.mixed.NAMES)):
        replace_down(modules,profile_factors(profile));report['variants'][label]={}
        for group,fresh in (('regression',False),('fresh',True)):
            result=base.evaluate(cpu,modules,rank,[c for c in cases if c[0].startswith('s45-')==fresh],protected)
            assert result['summary']['positions']==(128 if fresh else 142)
            report['variants'][label][group]=result
            print('CPU independent',label,group,result['summary'],flush=True)
        if label=='fp16':assert all(r['summary']['max_relative_l2']==0 for r in report['variants'][label].values())
    assert frozen==base.sha(OUT/'selection.json') and 'torch_intel_npu' not in sys.modules
    report.update(status='passed',source_sha256=base.sha(Path(__file__)),npu_calls=0)
    base.write('independent',report)


def hardware(label,route):
    selection=frozen_selection();metadata=base.read('independent')
    assert metadata['status']=='passed' and metadata['selection_sha256']==base.sha(OUT/'selection.json')
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    report=dict(status='running',label=label,route=route,candidate=selection['candidate'],positions=[],local_checks=[],warm=[],
        selection_sha256=base.sha(OUT/'selection.json'),evaluation_sha256=base.sha(OUT/'independent.json'),
        sources={name:base.sha(ROOT/name) for name in ('torch_intel_npu.py','torch_npu_smollm.py',
            'npunlock-probe/research/w8_output_quality.py','npunlock-probe/session-runtime/npunlock_session.dll')})
    def work():
        cpu,_=base.mixed.common.model();profile=selection['candidate'] if label=='candidate' else 'baseline'
        stack=FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,
            mode='fp16' if label=='fp16' else 'w8a16',packed_weights=None if label=='fp16' else base.mixed.weights(),
            corrections=None if label=='fp16' else profile_factors(profile))
        cases=[r for r in metadata['cases'] if route=='forward' or r['tag'] in ('bird','cat','fox','repeat32')]
        started=time.perf_counter()
        def append(case,position,actual,expected):
            report['positions'].append(dict(tag=case['tag'],position=position,**base.mixed.metric(actual,expected)))
        for index,case in enumerate(cases):
            ids=torch.tensor(case['token_ids']);length=ids.shape[1];offset=case['offset']
            if route=='kv':
                expected=cpu(ids,use_cache=False).logits[:,offset:].float();host=cpu.model.embed_tokens(ids);state=None
                for position in range(length):
                    x=host[:,position:position+1].contiguous().to('intel_npu');previous=state
                    output,state=stack.prefill(x) if state is None else stack.decode(x,state)
                    assert output.device.type=='intel_npu' and all(c.length==position+1 for c in state)
                    if position>=offset:
                        append(case,position,cpu.lm_head(cpu.model.norm(output.cpu()))[:,-1].float(),expected[:,position-offset])
                before=npu.stats();again,_=stack.decode(x,previous);after=npu.stats()
                assert after['npu_calls']-before['npu_calls']==30
                assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
                torch.testing.assert_close(again.cpu(),output.cpu(),rtol=0,atol=0)
                report['warm'].append(dict(tag=case['tag'],calls=30))
            else:
                padded=F.pad(ids,(0,64-length),value=0)
                expected=cpu(padded,use_cache=False).logits[:,offset:length].float()
                x=cpu.model.embed_tokens(padded).contiguous().to('intel_npu')
                if index==0:
                    for i,layer in enumerate(stack.layers):
                        reference,_=layer.reference_cpu(x.cpu());x=layer(x)
                        errors=base.mixed.common.errors(x.cpu(),reference)
                        assert errors['relative_l2']<.003 and errors['max_scaled_error']<.01,(label,i,errors)
                        report['local_checks'].append(dict(layer=i,**errors))
                    del layer
                else:x=stack(x)
                actual=cpu.lm_head(cpu.model.norm(x.cpu())).float()[:,offset:length]
                for j in range(actual.shape[1]):append(case,offset+j,actual[:,j],expected[:,j])
            print('NPU',route,label,index+1,'/',len(cases),flush=True)
        assert len(report['positions'])==(98 if route=='kv' else 270)
        report.update(summary=base.mixed.common.summary(report['positions']),seconds=time.perf_counter()-started,stats=npu.stats())
        if route=='forward':
            report['groups']={g:base.mixed.common.summary([r for r in report['positions'] if r['tag'].startswith('s45-')==fresh])
                for g,fresh in (('regression',False),('fresh',True))}
        if label=='fp16':assert all(r['quality_gate_passed'] for r in report['positions'])
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    gc.collect();npu.shutdown();report['shutdown']=npu.stats()
    report['healthy']=report['shutdown']['closed'] and not report['shutdown']['context_failed'] and all(
        report['shutdown'][k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_sessions','quarantined_buffer_bytes'))
    if not report['healthy']:report['status']='failed'
    base.write('hardware-'+route+'-'+label,report);print('hardware result',route,label,report.get('summary'),report['healthy'],flush=True)
    return int(report['status']!='passed')


def prepare():
    if (OUT/'weights.npz').exists():raise FileExistsError('Refusing to overwrite an existing preparation')
    base.prepare()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','fit','select','independent','hardware'))
    parser.add_argument('--variant',choices=('baseline','candidate','fp16'),default='candidate')
    parser.add_argument('--route',choices=('forward','kv'),default='forward')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    with torch.no_grad():
        result=hardware(args.variant,args.route) if args.stage=='hardware' else globals()[args.stage]()
    raise SystemExit(result or 0)


if __name__=='__main__':main()

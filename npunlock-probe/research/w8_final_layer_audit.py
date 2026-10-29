"""S57: replay the S56 final-layer guard without changing quality gates."""
import argparse
import hashlib
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F
import w8_native_selection as previous
from w8_scale_audit import scaled_peak

base=previous.base
attention=previous.study
probe=attention.probe
device=attention.device
ROOT=previous.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-final-layer-audit'
for module in (previous,previous.previous,previous.parent,previous.parent.previous,attention,attention.previous,
               attention.previous.previous,probe,device,device.previous,base,base.mixed):module.OUT=OUT


def ffn_reference(layer,residual,normalized,overrides=None):
    """CPU staged reference with explicit, cumulative Half boundary replacements."""
    overrides={} if overrides is None else overrides
    if (set(overrides)-{'gate','up','silu','gated','down'} or residual.ndim!=3 or not all(residual.shape)
        or normalized.shape!=residual.shape or any(t.device.type!='cpu' or t.dtype!=torch.float16
        or not torch.isfinite(t).all() for t in (residual,normalized))):
        raise ValueError('Requires matching nonempty finite CPU Half FFN inputs and known boundaries')
    def choose(name,value):
        result=overrides.get(name,value)
        if (result.shape!=value.shape or result.device.type!='cpu' or result.dtype!=torch.float16
            or not torch.isfinite(result).all()):raise ValueError('Invalid FFN boundary: '+name)
        return result
    x=normalized.reshape(-1,normalized.shape[-1]);result={}
    for name in ('gate','up'):
        result[name]=choose(name,probe.cpu_projection(layer,'mlp.'+name+'_proj.weight',x)[0])
    result['silu']=choose('silu',F.silu(result['gate'].float()).half())
    result['gated']=choose('gated',(result['silu']*result['up']).half())
    result['down']=choose('down',probe.cpu_projection(layer,'mlp.down_proj.weight',result['gated'])[0])
    result['output']=(residual+result['down'].view_as(residual)).half()
    return result


def replay_rows(reference,stages,coordinate):
    if (reference.device.type!='cpu' or not reference.numel() or not torch.isfinite(reference).all()
        or len(coordinate)!=reference.ndim or any(type(i) is not int or not 0<=i<n for i,n in zip(coordinate,reference.shape))):
        raise ValueError('Invalid reference or replay coordinate')
    coordinate=tuple(coordinate);prior=float(reference[coordinate]);rows=[];names=set()
    for name,value in stages:
        if (not name or name in names or value.device.type!='cpu' or value.shape!=reference.shape
            or value.dtype!=reference.dtype or not torch.isfinite(value).all()):raise ValueError('Invalid or duplicate replay stage')
        names.add(name);current=float(value[coordinate])
        rows.append(dict(stage=name,peak_value=current,signed_delta=current-prior,output=device.compare(value,reference)))
        prior=current
    if not rows:raise ValueError('Empty replay')
    assert sum(r['signed_delta'] for r in rows)==prior-float(reference[coordinate])
    return rows


def identity():
    paths={'source':Path(__file__),'selection_helper':Path(previous.__file__),'attention_helper':Path(attention.__file__),
           'ffn_helper':Path(probe.__file__),'weights':OUT/'weights.npz','factors':OUT/'corrections.npz',
           'fit':OUT/'logits-fit.npz','fit_record':OUT/'logits-fit.json','data':OUT/'fresh.json'}
    return {name+'_sha256':base.sha(path) for name,path in paths.items()}


def ffn_audit(npu,layer,x):
    compare=device.compare;shared=x.cpu();whole=layer(x);reference,trace=layer.reference_cpu(shared)
    residual=layer._run_fused(x,segment='attention')[0];normalized=layer._norm_device(residual,'post_attention_layernorm.weight')
    normal=layer._run_graph(residual,normalized,segment='ffn');graph,xml,_=layer.graph(tuple(x.shape),segment='ffn')
    values,key=probe.run_graph(npu,probe.ffn_probe_graph(graph,xml),[residual,normalized])
    actual={name:value.cpu() for name,value in values.items()};host_residual=residual.cpu();host_norm=normalized.cpu()
    norm1=layer._norm_device(x,'input_layernorm.weight').cpu()
    from_norm1,_=layer.reference_cpu(shared,segment='attention',normalized=norm1)
    def continue_ffn(value):return ffn_reference(layer,value,attention.norm_cpu(layer,value))['output']
    staged=ffn_reference(layer,host_residual,host_norm)
    existing,_=layer.reference_cpu(host_residual,segment='ffn',normalized=host_norm)
    torch.testing.assert_close(staged['output'],existing,rtol=0,atol=0)
    controls=dict(split_vs_whole=compare(normal.cpu(),whole.cpu()),exported_vs_normal=compare(actual['output'],normal.cpu()),
                  cpu_reference=compare(staged['output'],existing))
    for check in controls.values():assert check['exact_fraction']==1,('FFN observation changes arithmetic',controls)
    checks=dict(norm1_same_input=compare(norm1,trace['norm1']),
                norm2_same_input=compare(host_norm,attention.norm_cpu(layer,host_residual)))
    for name in ('gate','up','down'):
        inputs=host_norm.reshape(-1,576) if name!='down' else actual['gated']
        corrected,local,b=probe.cpu_projection(layer,'mlp.'+name+'_proj.weight',inputs)
        for label,observed,expected in (
            ('main_exact',actual[name+'_main'],local['main']),('main_rounded',actual[name+'_main'],local['rounded']),
            ('reduced',actual[name+'_reduced'],local['reduced']),
            ('delta_same_input',actual[name+'_delta'],F.linear(actual[name+'_reduced'].float(),b.float()).half()),
            ('add_same_input',actual[name+'_corrected'],(actual[name+'_main']+actual[name+'_delta']).half()),
            ('corrected',actual[name+'_corrected'],corrected)):
            checks[name+'_'+label]=compare(observed,expected)
    gate=actual['gate_corrected'];up=actual['up_corrected'];silu=actual['silu'].reshape_as(gate)
    checks['silu_same_input']=compare(silu,F.silu(gate.float()).half())
    checks['multiply_same_input']=compare(actual['gated'],(silu*up).half())
    checks['residual_same_input']=compare(actual['output'],(host_residual+actual['down_corrected'].view_as(host_residual)).half())
    peak=scaled_peak(whole.cpu(),reference);coordinate=peak['index']
    stages=[('norm1',continue_ffn(from_norm1)),('attention',continue_ffn(host_residual)),('norm2',staged['output'])]
    overrides={}
    for name,boundaries in (('gate-up',{'gate':gate,'up':up}),('silu',{'silu':silu}),('multiply',{'gated':actual['gated']}),
                            ('down',{'down':actual['down_corrected']})):
        overrides.update(boundaries);stages.append((name,ffn_reference(layer,host_residual,host_norm,overrides)['output']))
    stages.append(('residual-add',actual['output']))
    replays=replay_rows(reference,stages,coordinate)
    # A same-input SHAVE activation diagnostic; it does not select or modify model parameters.
    strict=(F.silu(values['gate_corrected'].float()).half()*values['up_corrected']).half()
    tail=probe.down_tail_graph(graph,xml)
    control,_=probe.run_graph(npu,tail,[residual,values['gated']]);changed,_=probe.run_graph(npu,tail,[residual,strict])
    controls['tail_control']=compare(control['output'].cpu(),actual['output']);assert controls['tail_control']['exact_fraction']==1
    rms=reference.float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-5)
    scores=(whole.cpu().float()-reference.float()).abs()/(reference.float().abs()+rms)
    return dict(peak=peak,baseline=compare(whole.cpu(),reference),controls=controls,checks=checks,replays=replays,
        guard_violations=int((scores>=.01).sum()),strict_silu=dict(output=compare(changed['output'].cpu(),reference),
        peak_value=float(changed['output'].cpu()[tuple(coordinate)])),input_sha256=hashlib.sha256(shared.numpy().tobytes()).hexdigest(),
        dpu_evidence=layer._inspect_blob(npu._sessions[key][1].graph_blob))


def diagnose(profile):
    if (OUT/('diagnostic-'+profile+'.json')).exists():raise FileExistsError('Refusing to overwrite diagnostic')
    import torch_intel_npu as npu
    report=dict(status='running',profile=profile,cases=[],**identity())
    def work():
        cpu,tokenizer=base.mixed.common.model();stack=previous.parent.stack_for(profile)
        zipper=base.read('fresh')['cases'][0];assert zipper['tag']=='zipper' and zipper['prompt']==previous.FRESH[0][1]
        control=next(r for r in base.read('prepare')['trajectories'] if r['index']==4)
        for tag,row in (('zipper',zipper),('original-4',control)):
            ids=torch.tensor(row['token_ids']);x=cpu.model.embed_tokens(ids).contiguous().to('intel_npu');checks=[]
            for i,layer in enumerate(stack.layers):
                reference,_=layer.reference_cpu(x.cpu());output=layer(x);error=device.compare(output.cpu(),reference)
                checks.append(dict(layer=i,**error))
                if i==29:break
                x=output
            assert len(checks)==30
            for error in checks[:29]:assert error['relative_l2']<.003 and error['max_scaled_error']<.01
            last=checks[-1];rejected=last['relative_l2']>=.003 or last['max_scaled_error']>=.01
            # Do not continue a whole-model quality run after a local precision rejection.
            attention_result=attention.diagnose_case(npu,layer,x,dict(profile=profile,prompt=tag,layer=29))
            ffn=ffn_audit(npu,layer,x)
            assert attention_result['baseline']==ffn['baseline']=={k:v for k,v in checks[-1].items() if k!='layer'}
            assert attention_result['input_sha256']==ffn['input_sha256']
            pos=ffn['peak']['index'][1]
            result=dict(tag=tag,prefix=row['prefix'],token_ids=row['token_ids'],local_checks=checks,
                guard_rejected=rejected,attention=attention_result,ffn=ffn,peak_token=tokenizer.decode(ids[0,pos:pos+1]),
                peak_in_scored_suffix=pos>=row['prefix']-1)
            report['cases'].append(result)
            print('final-layer replay',profile,tag,'guard',checks[-1], 'FFN replays',[(r['stage'],r['signed_delta']) for r in ffn['replays']],flush=True)
        assert report['cases'][0]['guard_rejected'] and not report['cases'][1]['guard_rejected']
        assert identity()=={k:report[k] for k in identity()}
        report['stats']=npu.stats()
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return device.finish(report,'diagnostic-'+profile,npu)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','fit','diagnose'))
    parser.add_argument('--profile',choices=('baseline','tail-e2-a100'),default='baseline')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    if args.stage=='fit':result=previous.previous.fit()
    else:
        with torch.no_grad():
            if args.stage=='prepare':previous.parent.previous.prepare();result=previous.data()
            else:result=diagnose(args.profile)
    raise SystemExit(result or 0)

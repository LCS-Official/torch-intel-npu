"""S52: same-input attention stage diagnosis and bounded NPU counterfactuals."""
import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import traceback

import numpy as np
import npunlock as symbolic
import torch
import w8_scale_quality as previous
from w8_scale_audit import scaled_peak

base=previous.base
probe=previous.previous.study
device=previous.device
ROOT=previous.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-attention-quality'
previous.OUT=previous.previous.OUT=probe.OUT=device.OUT=device.previous.OUT=base.OUT=base.mixed.OUT=OUT


def prepare():
    previous.prepare()
    previous.fit()


def attention_nodes(graph):
    """Recognize the existing dense causal graph; reject different topologies."""
    softmax=[n for n in graph.nodes if n.op=='Softmax']
    if len(softmax)!=1 or softmax[0].attrs.get('axis')!=3:raise ValueError('Expected one axis-3 attention softmax')
    probs=softmax[0].outputs[0];masked=softmax[0].inputs[0]
    if masked.producer.op!='Add':raise ValueError('Expected additive causal mask')
    multiply=[v for v in masked.producer.inputs if v.producer.op=='Multiply']
    if len(multiply)!=1:raise ValueError('Expected scaled attention scores')
    scaled=multiply[0];raw=[v for v in scaled.producer.inputs if v.producer.op=='MatMul']
    scales=[v for v in scaled.producer.inputs if v.producer.op=='Const']
    if len(raw)!=1 or len(scales)!=1 or float(np.asarray(scales[0].producer.metadata['value']))!=.125:
        raise ValueError('Expected QK scale 0.125')
    raw=raw[0];qk=raw.producer
    if qk.attrs.get('transpose_a',False) or not qk.attrs.get('transpose_b',False) or len(raw.shape)!=4 or raw.shape[-2]!=raw.shape[-1]:
        raise ValueError('Expected square causal QK transpose')
    masks=[v for v in masked.producer.inputs if v is not scaled]
    expected=np.where(np.arange(raw.shape[-1])[None,:]>np.arange(raw.shape[-2])[:,None],-np.inf,0).astype(np.float16)
    if len(masks)!=1 or masks[0].producer.op!='Const' or not np.array_equal(masks[0].producer.metadata['value'],expected):
        raise ValueError('Unexpected causal mask')
    products=[n for n in graph.nodes if n.op=='MatMul' and n.inputs[0] is probs]
    if len(products)!=1 or products[0].attrs.get('transpose_a',False) or products[0].attrs.get('transpose_b',False):
        raise ValueError('Expected PV product')
    pv=products[0];chains={key:probe.linear_chain(graph,'self_attn.'+key+'_proj.weight') for key in ('q','k','v','o')}
    values={key+'_'+stage:chain[stage] for key,chain in chains.items() for stage in ('main','reduced','delta','corrected')}
    values.update(q_rot=qk.inputs[0],k_repeat=qk.inputs[1],v_repeat=pv.inputs[1],raw=raw,scaled=scaled,masked=masked,
        probs=probs,pv=pv.outputs[0],attended=chains['o']['input'],output=graph.outputs[0])
    return values


def attention_probe_graph(graph,xml):
    return probe.serialize_probe(graph.inputs,attention_nodes(graph),xml,'attention_stages')


def attention_tail_graph(graph,xml,stage):
    values=attention_nodes(graph)
    if stage not in ('raw','masked','probs','attended'):raise ValueError('Unknown attention tail boundary')
    residual=symbolic.input('residual',shape=graph.inputs[0].shape,dtype='f16')
    boundary=symbolic.input(stage,shape=values[stage].shape,dtype='f16')
    inputs=[residual,boundary];mapped={graph.inputs[0]:residual,values[stage]:boundary}
    if stage!='attended':
        value=symbolic.input('value',shape=values['v_repeat'].shape,dtype='f16')
        inputs.append(value);mapped[values['v_repeat']]=value
    def clone(t):
        if t not in mapped:
            n=t.producer
            if n.op=='Parameter':raise ValueError('Tail retains an unexpected normalization dependency')
            copied=replace(n,inputs=tuple(clone(v) for v in n.inputs));mapped.update(zip(n.outputs,copied.outputs,strict=True))
        return mapped[t]
    return probe.serialize_probe(inputs,dict(output=clone(graph.outputs[0])),xml,'attention_tail_'+stage)


def cpu_attention(q,k,v,cos,sin,overrides=None):
    """Explicit Half boundaries and FP32 GEMM/softmax CPU reference only."""
    overrides={} if overrides is None else overrides
    allowed={'q','k','v','q_rot','k_repeat','v_repeat','raw','scaled','masked','probs','attended'}
    if set(overrides)-allowed:raise ValueError('Unknown attention boundary')
    if (q.ndim!=4 or k.ndim!=4 or v.ndim!=4 or not all(q.shape) or not all(k.shape) or k.shape!=v.shape
        or q.shape[0]!=k.shape[0] or q.shape[2:]!=k.shape[2:] or q.shape[1]%k.shape[1] or q.shape[-1]%2
        or any(t.device.type!='cpu' or t.dtype!=torch.float16 for t in (q,k,v,cos,sin))):
        raise ValueError('Requires matching CPU Half attention heads and even feature width')
    def choose(name,expected):
        value=overrides.get(name,expected)
        if (value.device.type!='cpu' or value.dtype!=torch.float16 or value.shape!=expected.shape or torch.isnan(value).any()
            or not torch.equal(torch.isinf(value),torch.isinf(expected)) or not torch.equal(value[torch.isinf(value)],expected[torch.isinf(expected)])):
            raise ValueError('Invalid attention boundary: '+name)
        return value
    q,k,v=(choose(name,value) for name,value in zip(('q','k','v'),(q,k,v),strict=True))
    def rotate(t):
        d=t.shape[-1]//2
        return (t*cos+torch.cat((-t[...,d:],t[...,:d]),-1)*sin).half()
    ratio=q.shape[1]//k.shape[1]
    q=choose('q_rot',rotate(q));k=choose('k_repeat',rotate(k).repeat_interleave(ratio,dim=1));v=choose('v_repeat',v.repeat_interleave(ratio,dim=1))
    raw=choose('raw',(q.float()@k.float().transpose(-1,-2)).half());scaled=choose('scaled',(raw*.125).half())
    length=raw.shape[-1];mask=torch.triu(torch.full((length,length),-float('inf'),dtype=torch.float16),diagonal=1)
    masked=choose('masked',(scaled+mask).half());probs=choose('probs',masked.float().softmax(-1).half())
    pv=(probs.float()@v.float()).half();attended=pv.transpose(1,2).reshape(-1,q.shape[1]*q.shape[-1])
    attended=choose('attended',attended)
    return dict(q_rot=q,k_repeat=k,v_repeat=v,raw=raw,scaled=scaled,masked=masked,probs=probs,attended=attended)


def compare_masked(actual,reference):
    if (actual.shape!=reference.shape or torch.isnan(actual).any() or torch.isnan(reference).any()
        or not torch.equal(torch.isinf(actual),torch.isinf(reference))
        or not torch.equal(actual[torch.isinf(actual)],reference[torch.isinf(reference)])):
        raise ValueError('Nonfinite attention score structure differs')
    finite=torch.isfinite(reference)
    if not finite.any():raise ValueError('No finite attention scores to compare')
    return dict(**device.compare(actual[finite],reference[finite]),nonfinite_entries=int((~finite).sum()))


def norm_cpu(layer,value):
    x=value.float();normalized=(x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)).half()
    return (normalized*layer._cpu_weight('post_attention_layernorm.weight')).half()


def diagnose_case(npu,layer,x,tag):
    compare=device.compare;shared=x.cpu();full=layer(x)
    residual=layer._run_fused(x,segment='attention')[0];normalized=layer._norm_device(x,'input_layernorm.weight')
    normal=layer._run_graph(x,normalized,segment='attention');graph,xml,_=layer.graph(tuple(x.shape),segment='attention')
    values,key=probe.run_graph(npu,attention_probe_graph(graph,xml),[x,normalized])
    actual={name:value.cpu() for name,value in values.items()}
    controls=dict(exported_vs_normal=compare(actual['output'],normal.cpu()),segment_vs_fused=compare(normal.cpu(),residual.cpu()))
    for check in controls.values():assert check['exact_fraction']==1,('Observation changes attention arithmetic',tag,controls)
    reference,trace=layer.reference_cpu(shared);reference_r1,cpu_trace=layer.reference_cpu(shared,segment='attention',normalized=normalized.cpu())
    projected={};checks={}
    for name in ('q','k','v'):
        inp=normalized.cpu().reshape(-1,576);corrected,local,b=probe.cpu_projection(layer,'self_attn.'+name+'_proj.weight',inp)
        projected[name]=corrected
        for stage,expected in (('main_exact',local['main']),('main_rounded',local['rounded']),('reduced',local['reduced']),
            ('delta',torch.nn.functional.linear(actual[name+'_reduced'].float(),b.float()).half()),
            ('add',(actual[name+'_main']+actual[name+'_delta']).half())):
            observed=actual[name+('_main' if stage.startswith('main_') else '_corrected' if stage=='add' else '_'+stage)]
            checks[name+'_'+stage]=compare(observed,expected)
    def heads(t,count):return t.view(1,64,count,64).transpose(1,2)
    cpu_heads={name:heads(projected[name],9 if name=='q' else 3) for name in ('q','k','v')}
    observed={name:heads(actual[name+'_corrected'],9 if name=='q' else 3) for name in ('q','k','v')}
    observed.update({name:actual[name] for name in ('q_rot','k_repeat','v_repeat','raw','scaled','masked','probs','attended')})
    cos=torch.from_numpy(np.frombuffer(layer._cos,dtype=np.float16).reshape(64,64).copy())[None,None]
    sin=torch.from_numpy(np.frombuffer(layer._sin,dtype=np.float16).reshape(64,64).copy())[None,None]
    ideal=cpu_attention(**cpu_heads,cos=cos,sin=sin)
    torch.testing.assert_close(ideal['attended'].view(1,64,576),cpu_trace['attention'],rtol=0,atol=0)
    from_projections=cpu_attention(**cpu_heads,cos=cos,sin=sin,overrides={n:observed[n] for n in ('q','k','v')})
    from_rope=cpu_attention(**cpu_heads,cos=cos,sin=sin,overrides={n:observed[n] for n in ('q_rot','k_repeat','v_repeat')})
    checks.update({name:compare(actual[name],from_projections[name]) for name in ('q_rot','k_repeat','v_repeat')})
    checks['qk_same_input']=compare(actual['raw'],from_rope['raw'])
    checks['scale_same_input']=compare(actual['scaled'],(actual['raw']*.125).half())
    mask=torch.triu(torch.full((64,64),-float('inf'),dtype=torch.float16),diagonal=1)
    checks['mask_same_input']=compare_masked(actual['masked'],(actual['scaled']+mask).half())
    checks['softmax_same_input']=compare(actual['probs'],actual['masked'].float().softmax(-1).half())
    pv=(actual['probs'].float()@actual['v_repeat'].float()).half().transpose(1,2).reshape(-1,576)
    checks['pv_same_input']=compare(actual['attended'],pv)
    o,local,b=probe.cpu_projection(layer,'self_attn.o_proj.weight',actual['attended'])
    checks['o_main_same_input']=compare(actual['o_main'],local['main'])
    checks['o_delta_same_input']=compare(actual['o_delta'],torch.nn.functional.linear(actual['o_reduced'].float(),b.float()).half())
    checks['o_add_same_input']=compare(actual['o_corrected'],(actual['o_main']+actual['o_delta']).half())
    checks['residual_add_same_input']=compare(actual['output'],(shared+actual['o_corrected'].view_as(shared)).half())
    def continuation(r1):return layer.reference_cpu(r1,segment='ffn',normalized=norm_cpu(layer,r1))[0]
    peak=scaled_peak(full.cpu(),reference);coordinate=peak['index'];replays=[];previous_output=reference
    def append(name,r1):
        nonlocal previous_output
        y=continuation(r1);replays.append(dict(stage=name,peak_value=float(y[coordinate]),
            signed_delta=float(y[coordinate])-float(previous_output[coordinate]),attention=compare(r1,reference_r1),output=compare(y,reference)))
        previous_output=y
    append('norm1',reference_r1);overrides={}
    for stage,names in (('qkv',('q','k','v')),('rope-repeat',('q_rot','k_repeat','v_repeat')),
        ('qk',('raw',)),('scale',('scaled',)),('mask',('masked',)),('softmax',('probs',)),('pv',('attended',))):
        overrides.update({name:observed[name] for name in names})
        attended=cpu_attention(**cpu_heads,cos=cos,sin=sin,overrides=overrides)['attended']
        corrected,_,_=probe.cpu_projection(layer,'self_attn.o_proj.weight',attended)
        append(stage,(shared+corrected.view_as(shared)).half())
    append('o-projection',(shared+actual['o_corrected'].view_as(shared)).half());append('residual-add',actual['output'])
    replays.append(dict(stage='remaining-norm2-ffn',peak_value=float(full.cpu()[coordinate]),
        signed_delta=float(full.cpu()[coordinate])-float(previous_output[coordinate])))
    assert sum(r['signed_delta'] for r in replays)==peak['actual']-peak['reference']
    strict={};tails={}
    for stage in ('raw','masked','probs','attended'):
        tail=attention_tail_graph(graph,xml,stage);tails[stage]=tail
        operands=[x,values[stage]]+([] if stage=='attended' else [values['v_repeat']])
        control,_=probe.run_graph(npu,tail,operands);controls['tail-'+stage]=compare(control['output'].cpu(),actual['output'])
        assert controls['tail-'+stage]['exact_fraction']==1,('Tail changes attention',tag,stage,controls['tail-'+stage])
    original_ffn=layer._run_fused(residual,segment='ffn')[0]
    controls['split_ffn_vs_whole']=compare(original_ffn.cpu(),full.cpu());assert controls['split_ffn_vs_whole']['exact_fraction']==1
    start=npu.stats()['npu_calls']
    raw=(values['q_rot'].float()@values['k_repeat'].float().transpose(-1,-2)).half()
    strict['qk-f32']=(raw,'raw',compare(raw.cpu(),from_rope['raw']))
    probabilities=values['masked'].float().softmax(-1).half()
    strict['softmax-f32']=(probabilities,'probs',compare(probabilities.cpu(),actual['masked'].float().softmax(-1).half()))
    attended=(values['probs'].float()@values['v_repeat'].float()).half().transpose(1,2).reshape(-1,576).contiguous()
    strict['pv-f32']=(attended,'attended',compare(attended.cpu(),pv))
    combined_probs=(raw*.125+mask.to('intel_npu')).float().softmax(-1).half()
    combined=(combined_probs.float()@values['v_repeat'].float()).half().transpose(1,2).reshape(-1,576).contiguous()
    strict['all-f32']=(combined,'attended',None);results={}
    for name,(tensor,stage,error) in strict.items():
        operands=[x,tensor]+([] if stage=='attended' else [values['v_repeat']])
        r1,_=probe.run_graph(npu,tails[stage],operands);y=layer._run_fused(r1['output'],segment='ffn')[0]
        results[name]=dict(operator_same_input=error,attention=compare(r1['output'].cpu(),reference_r1),
            output=compare(y.cpu(),reference),peak_value=float(y.cpu()[coordinate]),peak_delta=float(y.cpu()[coordinate])-float(reference[coordinate]))
    result=dict(tag=tag,controls=controls,checks=checks,replays=replays,peak=peak,candidates=results,
        baseline=compare(full.cpu(),reference),probs_row_sum_max_error=float((actual['probs'].float().sum(-1)-1).abs().max()),
        extra_counterfactual_calls=npu.stats()['npu_calls']-start,input_sha256=hashlib.sha256(shared.numpy().tobytes()).hexdigest(),
        dpu_evidence=layer._inspect_blob(npu._sessions[key][1].graph_blob))
    assert hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()==result['input_sha256']
    print('attention sample',tag,'peak',peak,'replays',[(r['stage'],r['signed_delta']) for r in replays],
        'candidates',{n:r['output']['max_scaled_error'] for n,r in results.items()},flush=True)
    return result


def diagnose(scope):
    import torch_intel_npu as npu
    report=dict(status='running',scope=scope,samples=[],source_sha256=base.sha(Path(__file__)),
        weights_sha256=base.read('prepare')['weights_sha256'],factors_sha256=base.read('factors')['factor_sha256'],
        scale_fit_sha256=base.read('scale-fit')['scale_fit_sha256'])
    def work():
        cpu,_=base.mixed.common.model();evaluation=previous.stack_for('baseline')
        prefixes=('ffn',) if scope=='focus' else ('baseline','ffn')
        trajectories=(4,) if scope=='focus' else (4,19);layers=(29,) if scope=='focus' else (14,29)
        for prefix in prefixes:
            source=previous.stack_for(prefix)
            for case in [r for r in base.read('prepare')['trajectories'] if r['index'] in trajectories]:
                x=cpu.model.embed_tokens(torch.tensor(case['token_ids'])).contiguous().to('intel_npu')
                for i in range(max(layers)+1):
                    if i in layers:report['samples'].append(diagnose_case(npu,evaluation.layers[i],x,dict(prefix=prefix,trajectory=case['index'],layer=i)))
                    if i<max(layers):x=source.layers[i](x)
        assert len(report['samples'])==(1 if scope=='focus' else 8)
        report['stats']=npu.stats()
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return device.finish(report,scope,npu)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','focus','cross'))
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():result=prepare() if args.stage=='prepare' else diagnose(args.stage)
    raise SystemExit(result or 0)

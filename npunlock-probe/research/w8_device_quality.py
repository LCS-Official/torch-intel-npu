"""S49: same-input device stage probes and real-NPU internal selection."""
import argparse
import copy
import gc
import hashlib
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import npunlock as symbolic
import torch
import torch.nn.functional as F
import w8_output_quality as previous

base=previous.base
ROOT=previous.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-device-quality'
previous.OUT=base.OUT=base.mixed.OUT=OUT
PROFILES=('baseline',*previous.PROFILES)
PROBE_LAYERS=(0,6,14,24,29)


def projection_probe_graph(graph,xml):
    """Export the existing down chain, keeping both original graph inputs live."""
    prefix='mlp_down_proj_weight_'
    def unique(values,description):
        values=list(values)
        if len(values)!=1:raise ValueError('Expected one '+description)
        return values[0]
    down=unique((n for n in graph.nodes if n.op=='MatMul' and n.inputs[1].producer.name==prefix+'correction_down'),'correction-down matrix')
    up=unique((n for n in graph.nodes if n.op=='MatMul' and n.inputs[1].producer.name==prefix+'correction_up'),'correction-up matrix')
    if up.inputs[0] is not down.outputs[0]:raise ValueError('Correction matrices are not consecutive')
    add=unique((n for n in graph.nodes if n.op=='Add' and up.outputs[0] in n.inputs),'correction add')
    main=unique((v for v in add.inputs if v is not up.outputs[0]),'main projection input')
    values=(down.inputs[0],main,down.outputs[0],up.outputs[0],add.outputs[0],graph.outputs[0])
    names=('gated','main','reduced','delta','corrected','output')
    outputs=[symbolic.op('Reshape',v,symbolic.constant(np.array(v.shape,dtype=np.int64)),
        special_zero=False,_shape=v.shape,_dtype='f16',_name='probe_'+name) for name,v in zip(names,values,strict=True)]
    probe=symbolic.Graph(graph.inputs,outputs,name='w8_down_probe');ir=symbolic.serialize_ir(probe)
    original=ET.fromstring(xml);updated=ET.fromstring(ir.xml)
    annotations={layer.attrib['name']:layer.find('rt_info') for layer in original.findall('./layers/layer') if layer.find('rt_info') is not None}
    for layer in updated.findall('./layers/layer'):
        if layer.attrib['type']=='Swish':layer.set('version','opset4')
        if layer.attrib['name'] in annotations:layer.append(copy.deepcopy(annotations[layer.attrib['name']]))
    return probe,ET.tostring(updated),ir.weights,names


def device_rank(profile,records):
    return (*base.mixed.common.ranking(records[profile]['summary']),profile!='baseline',profile)


def finish(report,name,npu):
    gc.collect();npu.shutdown();report['shutdown']=npu.stats()
    report['healthy']=report['shutdown']['closed'] and not report['shutdown']['context_failed'] and all(
        report['shutdown'][k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_sessions','quarantined_buffer_bytes'))
    if not report['healthy']:report['status']='failed'
    base.write(name,report);print(name,report.get('summary'),report['status'],report['healthy'],flush=True)
    return int(report['status']!='passed')


def source_identity():
    return dict(source_sha256=base.sha(Path(__file__)),
        fitting_source_sha256=base.sha(Path(previous.__file__)),fit_sha256=base.sha(OUT/'output-fit.json'))


def compare(actual,reference):
    assert actual.shape==reference.shape and torch.isfinite(actual).all() and torch.isfinite(reference).all()
    return dict(**base.mixed.common.errors(actual,reference),exact_fraction=float((actual==reference).float().mean()))


def probe():
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    report=dict(status='running',layers=[],**source_identity())
    def work():
        cpu,_=base.mixed.common.model();row=next(r for r in base.read('prepare')['trajectories'] if r['index']==4)
        stack=FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,
            packed_weights=base.mixed.weights(),corrections=base.corrections(32))
        x=cpu.model.embed_tokens(torch.tensor(row['token_ids'])).contiguous().to('intel_npu')
        for i,layer in enumerate(stack.layers):
            whole=layer(x)
            if i in PROBE_LAYERS:
                residual=layer._run_fused(x,segment='attention')[0]
                normalized=layer._norm_device(residual,'post_attention_layernorm.weight')
                segment=layer._run_graph(residual,normalized,segment='ffn')
                graph,xml,_=layer.graph(tuple(x.shape),segment='ffn')
                graph,xml,data,names=projection_probe_graph(graph,xml)
                key=('w8_down_probe',hashlib.sha256(xml+data).digest())
                def factory():
                    result=npu._native.compile_ir(xml,data,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
                    if (result.vendor_id,result.device_id,tuple(result.compiler_version))!=(0x8086,0x643e,(8,3)):
                        raise ValueError('Unsupported diagnostic profile')
                    return symbolic.load_native(result.graph_blob,graph=graph,libraries=npu._native)
                npu._register_graph_factory(key,factory)
                values=npu._run_many(key,[residual,normalized],[(v.shape,torch.float16) for v in graph.outputs])
                actual={name:value.cpu() for name,value in zip(names,values,strict=True)}
                host_residual=residual.cpu();host_norm=normalized.cpu();gated=actual['gated']
                _,trace=layer.reference_cpu(host_residual,segment='ffn',normalized=host_norm)
                a,b=layer._corrections[previous.DOWN].tensors();weight=previous.decoded_weight(layer._packed[previous.DOWN])
                main=(gated.double()@weight.t()).half()
                rounded=(gated.float()@weight.half().float().t()).half()
                reduced=F.linear(gated.float(),a.float()).half()
                delta_same_input=F.linear(actual['reduced'].float(),b.float()).half()
                ideal_norm=(host_residual.float()*torch.rsqrt(host_residual.float().square().mean(-1,keepdim=True)+1e-5)).half()
                ideal_norm=(ideal_norm*layer._cpu_weight('post_attention_layernorm.weight')).half()
                checks=dict(norm2=compare(host_norm,ideal_norm),gated=compare(gated,trace['gated'].reshape_as(gated)),
                    main_exact_weight=compare(actual['main'],main),main_rounded_weight=compare(actual['main'],rounded),
                    reduced=compare(actual['reduced'],reduced),delta_same_input=compare(actual['delta'],delta_same_input),
                    correction_add_same_input=compare(actual['corrected'],(actual['main']+actual['delta']).half()),
                    residual_add_same_input=compare(actual['output'],(host_residual+actual['corrected'].reshape_as(host_residual)).half()),
                    exported_vs_normal_ffn=compare(actual['output'],segment.cpu()),
                    split_vs_whole=compare(segment.cpu(),whole.cpu()))
                evidence=layer._inspect_blob(npu._sessions[key][1].graph_blob)
                report['layers'].append(dict(layer=i,checks=checks,evidence=evidence))
                print('probe layer',i,{k:v['relative_l2'] for k,v in checks.items()},flush=True)
                del values,residual,normalized,segment
            x=whole
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return finish(report,'probe',npu)


def stack_for(profile):
    from torch_npu_smollm import FrozenSmolStack
    return FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,
        mode='fp16' if profile=='fp16' else 'w8a16',packed_weights=None if profile=='fp16' else base.mixed.weights(),
        corrections=None if profile=='fp16' else previous.profile_factors(profile))


def internal(profile):
    import torch_intel_npu as npu
    report=dict(status='running',profile=profile,positions=[],local_checks=[],**source_identity())
    def work():
        cpu,_=base.mixed.common.model();stack=stack_for(profile)
        rows=[r for r in base.read('prepare')['trajectories'] if r['index']%5==4]
        assert [r['index'] for r in rows]==[4,9,14,19,24,29,34,39]
        for index,row in enumerate(rows):
            ids=torch.tensor(row['token_ids']);offset=row['prefix']-1
            expected=cpu(ids,use_cache=False).logits[:,offset:].float()
            x=cpu.model.embed_tokens(ids).contiguous().to('intel_npu')
            if index==0:
                for i,layer in enumerate(stack.layers):
                    reference,_=layer.reference_cpu(x.cpu());x=layer(x)
                    errors=base.mixed.common.errors(x.cpu(),reference)
                    assert errors['relative_l2']<.003 and errors['max_scaled_error']<.01,(profile,i,errors)
                    report['local_checks'].append(dict(layer=i,**errors))
                del layer
            else:x=stack(x)
            actual=cpu.lm_head(cpu.model.norm(x.cpu())).float()[:,offset:]
            for j in range(actual.shape[1]):
                report['positions'].append(dict(tag=str(row['index']),position=offset+j,**base.mixed.metric(actual[:,j],expected[:,j])))
            print('NPU internal',profile,row['index'],flush=True)
        assert len(report['positions'])==290
        if profile=='fp16':assert all(r['quality_gate_passed'] for r in report['positions'])
        report.update(summary=base.mixed.common.summary(report['positions']),stats=npu.stats())
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return finish(report,'internal-'+profile,npu)


def select():
    records={profile:base.read('internal-'+profile) for profile in (*PROFILES,'fp16')};identity=source_identity()
    for profile,row in records.items():
        assert row['healthy'] and row['status']=='passed' and all(row[k]==v for k,v in identity.items()),profile
        assert row['summary']['positions']==290 and len(row['local_checks'])==30
    selected=min(PROFILES,key=lambda p:device_rank(p,records))
    candidate=min(previous.PROFILES,key=lambda p:device_rank(p,records))
    base.write('selection',dict(status='passed',selected=selected,candidate=candidate,
        internal={p:r['summary'] for p,r in records.items()},**identity,
        internal_sha256={p:base.sha(OUT/('internal-'+p+'.json')) for p in records},
        policy='Real NPU B1/S64, only eight internal trajectories / 290 positions. '
        'Nine pre-existing profiles; failed positions, max/mean raw L2, prefer baseline on ties. '
        'FP16 is a control, not a ranked candidate. No independent or language rows used.'))
    print('DEVICE SELECTED',selected,'CANDIDATE',candidate,flush=True)


def frozen_selection():
    r=base.read('selection');identity=source_identity()
    assert r['status']=='passed' and all(r[k]==v for k,v in identity.items())
    assert r['selected'] in PROFILES and r['candidate'] in previous.PROFILES
    for p,digest in r['internal_sha256'].items():assert base.sha(OUT/('internal-'+p+'.json'))==digest
    return r


def evaluate(stage,*args):
    """Reuse S48's unchanged evaluators with this run's frozen device choice."""
    original=previous.frozen_selection
    previous.frozen_selection=frozen_selection
    try:
        if stage in ('data','cpu-language','hardware-language'):
            import w8_output_language as language
            return {'data':language.data,'cpu-language':language.cpu,'hardware-language':language.hardware}[stage](*args)
        return getattr(previous,stage)(*args)
    finally:previous.frozen_selection=original


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','fit','probe','internal','select','independent','hardware','data','cpu-language','hardware-language'))
    parser.add_argument('--profile',choices=(*PROFILES,'fp16'),default='baseline')
    parser.add_argument('--variant',choices=('baseline','candidate','fp16'),default='candidate')
    parser.add_argument('--route',choices=('forward','kv'),default='forward')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():
        if args.stage in ('prepare','fit'):result=getattr(previous,args.stage)()
        elif args.stage=='internal':result=internal(args.profile)
        elif args.stage=='hardware':result=evaluate('hardware',args.variant,args.route)
        elif args.stage=='hardware-language':result=evaluate('hardware-language',args.variant)
        elif args.stage in ('independent','data','cpu-language'):result=evaluate(args.stage)
        else:result=globals()[args.stage]()
    raise SystemExit(result or 0)

"""S50: isolate gate/up/SiLU and W8-down arithmetic on actual device inputs."""
import argparse
import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import npunlock as symbolic
import torch
import torch.nn.functional as F
import w8_device_quality as device

previous=device.previous
base=device.base
ROOT=device.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-gate-quality'
device.OUT=previous.OUT=base.OUT=base.mixed.OUT=OUT
LAYERS=(0,6,14,24,29)
TRAJECTORIES=(4,19)


def linear_chain(graph,name):
    prefix=name.replace('.','_')+'_'
    def unique(values,label):
        values=list(values)
        if len(values)!=1:raise ValueError('Expected one '+label)
        return values[0]
    down=unique((n for n in graph.nodes if n.op=='MatMul' and n.inputs[1].producer.name==prefix+'correction_down'),'reduced matrix')
    up=unique((n for n in graph.nodes if n.op=='MatMul' and n.inputs[1].producer.name==prefix+'correction_up'),'delta matrix')
    if up.inputs[0] is not down.outputs[0]:raise ValueError('Nonconsecutive correction matrices')
    add=unique((n for n in graph.nodes if n.op=='Add' and up.outputs[0] in n.inputs),'correction add')
    main=unique((v for v in add.inputs if v is not up.outputs[0]),'main input')
    return dict(input=down.inputs[0],main=main,reduced=down.outputs[0],delta=up.outputs[0],corrected=add.outputs[0])


def serialize_probe(inputs,values,xml,name):
    outputs=[symbolic.op('Reshape',v,symbolic.constant(np.array(v.shape,dtype=np.int64)),special_zero=False,
        _shape=v.shape,_dtype='f16',_name='trace_'+key) for key,v in values.items()]
    graph=symbolic.Graph(inputs,outputs,name=name);ir=symbolic.serialize_ir(graph)
    original=ET.fromstring(xml);updated=ET.fromstring(ir.xml)
    annotations={layer.attrib['name']:layer.find('rt_info') for layer in original.findall('./layers/layer') if layer.find('rt_info') is not None}
    for layer in updated.findall('./layers/layer'):
        if layer.attrib['type']=='Swish':layer.set('version','opset4')
        if layer.attrib['name'] in annotations:layer.append(copy.deepcopy(annotations[layer.attrib['name']]))
    return graph,ET.tostring(updated),ir.weights,tuple(values)


def ffn_probe_graph(graph,xml):
    chains={key:linear_chain(graph,'mlp.'+key+'_proj.weight') for key in ('gate','up','down')}
    swish=[n for n in graph.nodes if n.op=='Swish']
    if len(swish)!=1:raise ValueError('Expected one FFN Swish')
    def unshape(t):
        while t.producer.op=='Reshape':t=t.producer.inputs[0]
        return t
    gated=unshape(chains['down']['input'])
    if (unshape(swish[0].inputs[0]) is not chains['gate']['corrected'] or gated.producer.op!='Multiply'
        or unshape(gated.producer.inputs[0]) is not swish[0].outputs[0]
        or unshape(gated.producer.inputs[1]) is not chains['up']['corrected']):
        raise ValueError('FFN gate/up/Swish topology differs')
    values={key+'_'+stage:chain[stage] for key,chain in chains.items() for stage in ('main','reduced','delta','corrected')}
    values.update(silu=swish[0].outputs[0],gated=chains['down']['input'],output=graph.outputs[0])
    return serialize_probe(graph.inputs,values,xml,'w8_ffn_stages')


def down_tail_graph(graph,xml):
    chain=linear_chain(graph,'mlp.down_proj.weight')
    residual=symbolic.input('residual',shape=graph.inputs[0].shape,dtype='f16')
    gated=symbolic.input('gated',shape=chain['input'].shape,dtype='f16')
    mapped={graph.inputs[0]:residual,chain['input']:gated}
    def clone(t):
        if t not in mapped:
            n=t.producer
            if n.op=='Parameter':raise ValueError('Tail still depends on normalization input')
            cloned=replace(n,inputs=tuple(clone(v) for v in n.inputs))
            mapped.update(zip(n.outputs,cloned.outputs,strict=True))
        return mapped[t]
    values={key:clone(value) for key,value in dict(corrected=chain['corrected'],output=graph.outputs[0]).items()}
    return serialize_probe((residual,gated),values,xml,'w8_down_tail')


def gated_error_parts(actual,gate,up,silu,cpu_gate,cpu_up):
    reference=(F.silu(cpu_gate.float()).half()*cpu_up).half().float()
    strict=(F.silu(gate.float()).half()*up).half().float()
    observed=(silu*up).half().float()
    parts=dict(total=actual.float()-reference,projection=strict-reference,
        activation=observed-strict,multiply=actual.float()-observed)
    denominator=torch.linalg.vector_norm(reference).clamp_min(1e-20)
    result={key:float(torch.linalg.vector_norm(value)/denominator) for key,value in parts.items()}
    result['reconstruction']=float(torch.linalg.vector_norm(parts['total']-parts['projection']-parts['activation']-parts['multiply'])/denominator)
    return result


def run_graph(npu,recipe,inputs):
    graph,xml,data,names=recipe;key=('w8_gate_probe',hashlib.sha256(xml+data).digest())
    def factory():
        result=npu._native.compile_ir(xml,data,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
        if (result.vendor_id,result.device_id,tuple(result.compiler_version))!=(0x8086,0x643e,(8,3)):
            raise ValueError('Unsupported diagnostic profile')
        return symbolic.load_native(result.graph_blob,graph=graph,libraries=npu._native)
    npu._register_graph_factory(key,factory)
    values=npu._run_many(key,inputs,[(v.shape,torch.float16) for v in graph.outputs])
    return dict(zip(names,values,strict=True)),key


def cpu_projection(layer,name,x):
    weight=previous.decoded_weight(layer._packed[name]);a,b=layer._corrections[name].tensors()
    main=(x.double()@weight.t()).half()
    reduced=F.linear(x.float(),a.float()).half()
    delta=F.linear(reduced.float(),b.float()).half()
    return (main+delta).half(),dict(main=main,rounded=(x.float()@weight.half().float().t()).half(),reduced=reduced),b


def diagnose(route):
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    report=dict(status='running',route=route,samples=[],source_sha256=base.sha(Path(__file__)),
        weights_sha256=base.read('prepare')['weights_sha256'],factors_sha256=base.read('factors')['factor_sha256'])
    compare=device.compare
    def work():
        cpu,_=base.mixed.common.model();stack=FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,
            packed_weights=base.mixed.weights(),corrections=base.corrections(32))
        for case in [r for r in base.read('prepare')['trajectories'] if r['index'] in TRAJECTORIES]:
            host=cpu.model.embed_tokens(torch.tensor(case['token_ids']));caches=None
            if route=='decode':
                _,caches=stack.prefill(host[:,:63].contiguous().to('intel_npu'))
                x=host[:,63:].contiguous().to('intel_npu')
            else:x=host.contiguous().to('intel_npu')
            for i,layer in enumerate(stack.layers):
                old_cache=None if caches is None else caches[i]
                snapshots=None
                if old_cache is not None and i in LAYERS:snapshots=[v.cpu().clone() for v in (old_cache.keys,old_cache.values,old_cache._position)]
                whole=layer(x) if old_cache is None else layer.decode(x,old_cache)[0]
                if i in LAYERS:
                    residual=layer._run_fused(x,segment='attention',cache=old_cache,kv=old_cache is not None)[0]
                    normalized=layer._norm_device(residual,'post_attention_layernorm.weight')
                    normal=layer._run_graph(residual,normalized,segment='ffn')
                    graph,xml,_=layer.graph(tuple(x.shape),segment='ffn')
                    values,key=run_graph(npu,ffn_probe_graph(graph,xml),[residual,normalized])
                    actual={name:value.cpu() for name,value in values.items()};host_norm=normalized.cpu().reshape(-1,576)
                    checks=dict(exported_vs_normal=compare(actual['output'],normal.cpu()),split_vs_whole=compare(normal.cpu(),whole.cpu()))
                    references={}
                    for name in ('gate','up','down'):
                        inp=host_norm if name!='down' else actual['gated']
                        reference,local,b=cpu_projection(layer,'mlp.'+name+'_proj.weight',inp)
                        references[name]=reference
                        for target,ref in (('main_exact',local['main']),('main_rounded',local['rounded']),('reduced',local['reduced']),
                            ('delta',F.linear(actual[name+'_reduced'].float(),b.float()).half()),
                            ('add',(actual[name+'_main']+actual[name+'_delta']).half())):
                            observed=actual[name+'_main'] if target.startswith('main_') else actual[name+'_'+('corrected' if target=='add' else target)]
                            checks[name+'_'+target]=compare(observed,ref)
                    gate=actual['gate_corrected'];up=actual['up_corrected'];silu=actual['silu'].reshape_as(gate)
                    checks['silu_same_input']=compare(silu,F.silu(gate.float()).half())
                    checks['multiply_same_input']=compare(actual['gated'],(silu*up).half())
                    parts=gated_error_parts(actual['gated'],gate,up,silu,references['gate'],references['up'])
                    strict=F.silu(values['gate_corrected'].float()).half()
                    strict_gated=strict*values['up_corrected']
                    checks['shave_silu_same_input']=compare(strict.cpu(),F.silu(gate.float()).half())
                    tail=down_tail_graph(graph,xml)
                    control,_=run_graph(npu,tail,[residual,values['gated']])
                    changed,_=run_graph(npu,tail,[residual,strict_gated])
                    checks['tail_control_down']=compare(control['corrected'].cpu(),actual['down_corrected'])
                    checks['tail_control_output']=compare(control['output'].cpu(),actual['output'])
                    reference,_=layer.reference_cpu(residual.cpu(),segment='ffn',normalized=normalized.cpu())
                    checks['normal_vs_cpu_ffn']=compare(normal.cpu(),reference)
                    checks['strict_silu_tail_vs_cpu_ffn']=compare(changed['output'].cpu(),reference)
                    if snapshots is not None:
                        for value,saved in zip((old_cache.keys,old_cache.values,old_cache._position),snapshots,strict=True):
                            torch.testing.assert_close(value.cpu(),saved,rtol=0,atol=0)
                    report['samples'].append(dict(trajectory=case['index'],layer=i,checks=checks,gated_parts=parts,
                        evidence=layer._inspect_blob(npu._sessions[key][1].graph_blob),cache_unchanged=snapshots is not None))
                    print('gate probe',route,case['index'],i,'SiLU',checks['silu_same_input']['relative_l2'],
                        'multiply',checks['multiply_same_input']['relative_l2'],'export',checks['exported_vs_normal']['relative_l2'],flush=True)
                    del values,control,changed,strict,strict_gated,residual,normalized,normal
                x=whole
        assert len(report['samples'])==10
        report['stats']=npu.stats()
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return device.finish(report,'probe-'+route,npu)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','probe'))
    parser.add_argument('--route',choices=('forward','decode'),default='forward')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():result=previous.prepare() if args.stage=='prepare' else diagnose(args.route)
    raise SystemExit(result or 0)

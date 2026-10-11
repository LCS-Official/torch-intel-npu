"""Paired full generation: original decode vs shared positional Gather."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import weakref

from decode_mask_table import OUT
from w8_quality_probe import ROOT, sha


def main(mode):
    import torch_intel_npu as npu
    import torch_npu_smollm as smol
    graph_source=OUT/'joint/joined-graph.py';scope=dict(vars(smol))
    exec(compile(graph_source.read_text(),str(graph_source),'exec'),scope);graph=scope['graph']
    baseline_source=OUT/'joint/baseline-graph.py';baseline_scope=dict(vars(smol))
    exec(compile(baseline_source.read_text(),str(baseline_source),'exec'),baseline_scope)
    source=ROOT/'npunlock-probe/research/smollm_generation_profile.py';text=source.read_text()
    old='            actual_group=None if baseline_prepare else group';assert text.count(old)==1
    text=text.replace(old,'            actual_group=None\n            select_graph(group)')
    old="    report['comparison']='output preparation, both group4' if baseline_prepare else 'group1 vs default group4'"
    assert text.count(old)==1
    text=text.replace(old,"    report['comparison']='original vs shared positional Gather, both group4'")
    directory=OUT/'generation';directory.mkdir(exist_ok=True)
    runner=directory/'runner.py';runner.write_text(text,encoding='utf-8')
    (directory/'wrapper.py').write_bytes(Path(__file__).read_bytes())
    state={};graphs=[];original_construct=smol.FrozenSmolStack.from_checkpoint.__func__
    original_graph=smol.FrozenSmolLayer.graph
    def construct(cls,*args,**options):
        stack=original_construct(cls,*args,**options);assert len(stack.layers)==30
        state['stack']=weakref.ref(stack);return stack
    def select(group):
        stack=state['stack']();assert stack is not None
        if group==1 and 'keys' not in state:return
        if 'keys' not in state:
            state['keys']={name:[] for name in ('baseline','joined')}
            kernel=smol._rmsnorm_kernel();selection=smol._kv_select_kernel()
            for i,layer in enumerate(stack.layers):
                baseline=dict(layer._keys);assert len(baseline)==2
                state['keys']['baseline'].append(baseline);keys=dict(baseline)
                for local,key in baseline.items():
                    if local[-1]!='decode':continue
                    _,previous=npu._sessions[key]
                    ir,xml,weights=graph(layer,(1,1,576),segment='layer',kv_cache='decode',fused_norm=True)
                    candidate=smol.compile_fused_rmsnorm(ir,xml,weights,kernel,libraries=npu._native,
                        whole_layer=True,select_kernel_elf=selection)
                    new_key=('decode-mask-generation',mode,i);assert new_key not in npu._sessions
                    npu._sessions[new_key]=npu._native.create_inference_session(candidate.graph_blob,
                        timeout_ms=10000,parent=npu._root_session),candidate
                    npu._session_creations+=1;keys[local]=new_key
                    graphs.append(dict(layer=i,baseline_blob_sha256=hashlib.sha256(previous.graph_blob).hexdigest(),
                        joined_blob_sha256=hashlib.sha256(candidate.graph_blob).hexdigest(),xml_sha256=hashlib.sha256(xml).hexdigest(),
                        ir_nodes=dict(Counter(node.op for node in ir.nodes)),weight_bytes=len(weights),
                        patches=[json.loads(p) for p in candidate.patch_reports]))
                state['keys']['joined'].append(keys)
        for layer,keys in zip(stack.layers,state['keys']['baseline' if group==1 else 'joined']):layer._keys=keys
    smol.FrozenSmolStack.from_checkpoint=classmethod(construct)
    smol.FrozenSmolLayer.graph=baseline_scope['graph']
    namespace=dict(__name__='decode_mask_generation_runner',__file__=str(runner),select_graph=select)
    try:
        exec(compile(text,str(runner),'exec'),namespace);namespace['main'](mode,output=directory)
    finally:
        smol.FrozenSmolStack.from_checkpoint=classmethod(original_construct)
        smol.FrozenSmolLayer.graph=original_graph
    path=directory/(mode+'.json');report=json.loads(path.read_text())
    assert len(graphs)==30
    allocation_counts={str(label):[s['delta']['driver_allocations'] for s in report['samples']
        if (1 if s['group']==1 else 4)==label] for label in (1,4)}
    assert allocation_counts['1']==allocation_counts['4']
    report.update(template_sha256=sha(source),wrapper_sha256=sha(Path(__file__)),graph_source_sha256=sha(graph_source),
        baseline_graph_source_sha256=sha(baseline_source),driver_allocations_per_generation=allocation_counts,
        graph_selection='before generation timer; prefill keys shared; both default group4',graphs=graphs)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('fp16','w8a16'),required=True)
    main(parser.parse_args().mode)

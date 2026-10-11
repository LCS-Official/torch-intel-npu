"""Cut the rejected batch/head graph at projection, heads, RoPE and cache."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-kv-batch'


def main():
    import npunlock as s
    import torch_intel_npu as npu
    path=OUT.parent/'smollm-kv-profile/failures/batch-head-fold/torch_npu_smollm.py'
    spec=importlib.util.spec_from_file_location('smollm_rejected_heads',path);module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    model=module.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16')
    graph,_,_=model.graph((2,1,576),segment='attention',kv_cache='decode')
    targets=[('projection',next(n.outputs[0] for n in graph.nodes if n.op=='MatMul')),
        ('heads',next(n.outputs[0] for n in graph.nodes if n.op=='Transpose')),
        ('rope',next(n.outputs[0] for n in graph.nodes if n.op=='Add')),
        ('cache',next(n.outputs[0] for n in graph.nodes if n.op=='Select' and n.outputs[0].shape==(2,3,64,64))),
        ('full',graph.outputs)]
    report=dict(status='running',rejected_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checks=[])
    for name,outputs in targets:
        outputs=outputs if type(outputs) is tuple else (outputs,);seen=set();todo=[v.producer for v in outputs]
        while todo:
            node=todo.pop()
            if node in seen:continue
            seen.add(node);todo.extend(v.producer for v in node.inputs)
        cut=s.Graph([v for v in graph.inputs if v.producer in seen],outputs);ir=s.serialize_ir(cut);root=ET.fromstring(ir.xml)
        for layer in root.findall('./layers/layer'):
            if layer.attrib['type']=='Gather':layer.find('data').attrib.pop('batch_dims',None)
        xml=ET.tostring(root);(OUT/f'lowering-{name}.xml').write_bytes(xml);(OUT/f'lowering-{name}.bin').write_bytes(ir.weights)
        entry=dict(stage=name,status='running',output_shapes=[v.shape for v in outputs]);report['checks'].append(entry)
        try:
            result=npu._native.compile_ir(xml,ir.weights,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
            entry.update(status='compiled',blob_sha256=hashlib.sha256(result.graph_blob).hexdigest())
        except Exception as error:entry.update(status='compile_failed',error=str(error))
        print(entry,flush=True)
    npu.shutdown();report.update(status='passed',final_stats=npu.stats())
    assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
    (OUT/'lowering.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()

"""Offline, isolated compiler probe for an N=1 batch/head decode layout."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-kv-batch'


def main():
    import torch
    import npunlock as s
    import torch_intel_npu as npu
    from torch_npu_smollm_stack_demo import errors
    path=OUT/'candidate.py';spec=importlib.util.spec_from_file_location('smollm_batch_candidate',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.__file__=str(ROOT/'torch_npu_smollm.py')
    report=dict(status='running',candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checks=[])
    try:
        with torch.no_grad():
            def work():
                torch.set_num_threads(8);torch.manual_seed(1091)
                for mode in ['fp16','w8a16']:
                    model=module.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode)
                    for batch in [2,4,8]:
                        tag=f'{mode}-b{batch}';graph,xml,data=model.graph((batch,1,576),segment='attention',kv_cache='decode')
                        (OUT/f'{tag}.xml').write_bytes(xml);(OUT/f'{tag}.bin').write_bytes(data)
                        # Cut at the first static projection to distinguish its
                        # lowering from cache update and dynamic attention.
                        first=next(node.outputs[0] for node in graph.nodes if node.op=='MatMul')
                        reachable=set();pending=[first.producer]
                        while pending:
                            node=pending.pop()
                            if node in reachable:continue
                            reachable.add(node);pending.extend(value.producer for value in node.inputs)
                        cut=s.Graph([value for value in graph.inputs if value.producer in reachable],[first])
                        ir=s.serialize_ir(cut);root=ET.fromstring(ir.xml)
                        for layer in root.findall('./layers/layer'):
                            if layer.attrib['name'].endswith('_decode'):
                                info=ET.SubElement(layer,'rt_info');ET.SubElement(info,'attribute',name='decompression',version='0')
                        for stage,source,weights in [('projection',ET.tostring(root),ir.weights),('decode',xml,data)]:
                            entry=dict(tag=tag,stage=stage,status='running');report['checks'].append(entry)
                            try:
                                result=npu._native.compile_ir(source,weights,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
                                entry.update(status='compiled',blob_sha256=hashlib.sha256(result.graph_blob).hexdigest())
                            except Exception as error:entry.update(status='compile_failed',error=str(error))
                            print(entry,flush=True)
                        if report['checks'][-1]['status']!='compiled':continue
                        host=torch.randn(batch,9,576).half()*.1
                        _,cache=model.prefill(host[:,:8].contiguous().to('intel_npu'))
                        one=host[:,-1:].contiguous().to('intel_npu');before=npu.stats();actual,cache=model.decode(one,cache);after=npu.stats()
                        reference,_=model.reference_cpu(host);error=errors(actual.cpu(),reference[:,-1:])
                        entry=dict(tag=tag,stage='execute',execution=error,graph_calls=after['npu_calls']-before['npu_calls'])
                        report['checks'].append(entry);print(entry,flush=True)
                        assert error['relative_l2']<.003 and error['max_scaled_error']<.01
                del model
            work();gc.collect();npu.shutdown();report['final_stats']=npu.stats()
            assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'layout-probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

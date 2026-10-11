"""Keep single-token IR and reference/graph guards unchanged."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-prefill-fusion'


def main():
    import torch
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module);module.__file__=str(ROOT/'torch_npu_smollm.py');return module
    old=load('prefill_integrity_before',OUT/'before/torch_npu_smollm.py')
    new=load('prefill_integrity_candidate',OUT/'candidate.py')
    checks=[];torch.set_num_threads(8)
    for mode in ['fp16','w8a16']:
        a=old.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode)
        b=new.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode)
        for batch in range(1,9):
            for kind in [None,'prefill','decode']:
                options=dict(segment='layer',kv_cache=kind,fused_norm=True)
                _,xml,weights=a.graph((batch,1,576),**options);_,current,data=b.graph((batch,1,576),**options)
                assert xml==current and weights==data,(mode,batch,kind)
                checks.append(dict(mode=mode,batch=batch,kind=kind,single_token_ir_weights_equal=True,
                    xml_sha256=hashlib.sha256(xml).hexdigest(),weights_sha256=hashlib.sha256(weights).hexdigest()))
    def reject(name,fn):
        try:fn()
        except ValueError as error:checks.append(dict(name=name,rejected=True,error=str(error)))
        else:raise AssertionError(name+' was accepted')
    for name,shape in [('zero_length',(1,0,576)),('over_length',(1,65,576)),('over_batch',(9,2,576)),('wrong_width',(1,2,575))]:
        reject(name,lambda shape=shape:b.graph(shape,segment='layer',fused_norm=True))
    reject('unfused_whole_layer',lambda:b.graph((1,7,576),segment='layer'))
    reject('diagnostic_stage',lambda:b.graph((1,7,576),segment='layer',fused_norm=True,stage='norm1'))
    reject('multi_token_decode',lambda:b.graph((1,2,576),segment='layer',fused_norm=True,kv_cache='decode'))
    host=torch.zeros(1,2,576,dtype=torch.float16)
    reject('internal_layer_cpu_reference',lambda:b.reference_cpu(host,segment='layer',normalized=host))
    report=dict(status='passed',checks=checks,sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['npunlock-probe/research/smollm_prefill_integrity.py',
            'npunlock-probe/research/results/smollm-prefill-fusion/candidate.py',
            'npunlock-probe/research/results/smollm-prefill-fusion/before/torch_npu_smollm.py']})
    (OUT/'integrity.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report['status'],len(checks))


if __name__=='__main__':main()

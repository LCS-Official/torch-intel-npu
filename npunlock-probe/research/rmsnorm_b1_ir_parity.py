"""Prove B2..B8 fused IR/weights are unchanged, without device execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-b1-fusion'


def main():
    from torch_npu_smollm import FrozenSmolLayer
    path=OUT/'before/torch_npu_smollm.py';spec=importlib.util.spec_from_file_location('smollm_before_b1_ir',path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old);old.__file__=str(ROOT/'torch_npu_smollm.py')
    checks=[]
    for mode in ['fp16','w8a16']:
        a=old.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode)
        b=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode)
        for batch in range(2,9):
            for segment,kind in [('attention',None),('attention','prefill'),('attention','decode'),('ffn',None)]:
                _,xml_a,data_a=a.graph((batch,1,576),segment=segment,kv_cache=kind,fused_norm=True)
                _,xml_b,data_b=b.graph((batch,1,576),segment=segment,kv_cache=kind,fused_norm=True)
                assert xml_a==xml_b and data_a==data_b,(mode,batch,segment,kind)
                checks.append(dict(mode=mode,batch=batch,segment=segment,kind=kind,xml_weights_equal=True,
                    xml_sha256=hashlib.sha256(xml_b).hexdigest(),weights_sha256=hashlib.sha256(data_b).hexdigest()))
    report=dict(status='passed',checks=checks,baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['torch_npu_smollm.py','npunlock-probe/research/rmsnorm_b1_ir_parity.py']})
    (OUT/'ir-parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('passed',len(checks))


if __name__=='__main__':main()

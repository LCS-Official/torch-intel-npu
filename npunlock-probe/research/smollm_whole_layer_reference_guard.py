"""Check the final CPU-reference guard without rerunning unchanged device code."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-whole-layer'


def main():
    import torch
    from torch_npu_smollm import FrozenSmolLayer
    path=OUT/'candidate.py';spec=importlib.util.spec_from_file_location('reference_measured',path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    torch.set_num_threads(8);torch.manual_seed(891)
    current=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct')
    measured=old.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct')
    x=torch.randn(3,4,576).half()*.1;normalized=torch.randn_like(x);checks=[]
    with torch.no_grad():
        for segment in [None,'attention','ffn']:
            kwargs=dict(segment=segment,normalized=None if segment is None else normalized)
            a,traces=current.reference_cpu(x,**kwargs);b,previous=measured.reference_cpu(x,**kwargs)
            assert traces.keys()==previous.keys() and torch.equal(a.view(torch.int16),b.view(torch.int16))
            assert all(torch.equal(v,previous[k]) for k,v in traces.items())
            checks.append(dict(segment=segment,status='passed',all_outputs_and_traces_equal=True))
        try:current.reference_cpu(x,segment='layer',normalized=normalized)
        except ValueError as error:checks.append(dict(segment='layer',status='rejected',error=str(error)))
        else:raise AssertionError('internal layer segment was accepted by the CPU reference')
    report=dict(status='passed',checks=checks,sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['torch_npu_smollm.py','npunlock-probe/research/smollm_whole_layer_reference_guard.py']},
        measured_candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT/'reference-guard.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report['status'],len(checks))


if __name__=='__main__':main()

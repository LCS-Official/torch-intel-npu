"""CPU-only tail precision diagnosis; does not select a production policy.

Reuses reference_cpu(quantized=False) for tail layers, without changing layer
objects, public runtime APIs, calibration weights, or historical artifacts.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import w8_continuation_calibration as experiment
from w8_quality_probe import ROOT,MODEL,sha
from w8_calibration_probe import metric


def main():
    from transformers import AutoModelForCausalLM
    from torch_npu_smollm import FrozenSmolStack
    experiment.configure();out=experiment.OUT;torch.set_num_threads(8)
    packed=experiment.full.weights();reports={case:json.loads((out/(case+'.json')).read_text()) for case in ('holdout','decode')}
    for case,report in reports.items():assert report['status']=='passed' and sha(out/(case+'-outputs.npz'))==report['outputs_sha256']
    cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
    stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=30,mode='w8a16',packed_weights=packed)
    report=dict(status='running',source_sha256=sha(Path(__file__)),sources={name:sha(ROOT/name) for name in experiment.SOURCES},
        weights_sha256=sha(out/'weights.npz'),input_reports_sha256={case:sha(out/(case+'.json')) for case in reports},
        npu_calls=0,policy='CPU reference sensitivity diagnosis on existing regression cases; no production selection',cases=[])
    cases=[];saved={}
    with np.load(out/'decode-outputs.npz',allow_pickle=False) as arrays:
        for tag in ('bird','garden','cat','repeat32'):
            worst=max((row for row in reports['decode']['positions'] if row['tag'].startswith(tag+'/')),key=lambda row:row['variants']['calibrated']['relative_l2'])
            cases.append((worst['tag'],arrays[tag+'/tokens'][:,:worst['cache_length']].copy(),arrays[worst['tag']+'/baseline-logits'].copy()))
    with np.load(out/'holdout-outputs.npz',allow_pickle=False) as arrays:
        worst=max((row for row in reports['holdout']['positions'] if row['execution']=='incremental_prompt'),key=lambda row:row['variants']['calibrated']['relative_l2'])
        cases.append((worst['tag'],arrays[worst['tag']+'/ids'].copy(),arrays[worst['tag']+'/baseline-logits'].copy()))
    for tag,ids,baseline in cases:
        baseline=torch.from_numpy(baseline);host=cpu.model.embed_tokens(torch.from_numpy(ids))
        saved[tag+'/ids']=ids;saved[tag+'/baseline-logits']=baseline.numpy().copy();entry=dict(tag=tag,variants={});report['cases'].append(entry)
        for tail in (0,1,2,4,8,30):
            value=host
            for i,layer in enumerate(stack.layers):value,_=layer.reference_cpu(value,quantized=i<30-tail)
            logits=cpu.lm_head(cpu.model.norm(value))[:,-1].float();entry['variants'][str(tail)]=metric(logits,baseline)
            saved[f'{tag}/tail-{tail}-logits']=logits.numpy().copy()
        print(tag,{tail:round(row['relative_l2'],6) for tail,row in entry['variants'].items()},flush=True)
    path=out/'tail-precision-outputs.npz';np.savez(path,**saved)
    with np.load(path,allow_pickle=False) as arrays:
        for row in report['cases']:
            for tail,record in row['variants'].items():
                actual=metric(torch.from_numpy(arrays[f'{row["tag"]}/tail-{tail}-logits']),torch.from_numpy(arrays[row['tag']+'/baseline-logits']))
                for key,value in actual.items():assert np.isclose(value,record[key],rtol=1e-6,atol=1e-9)
    assert 'torch_intel_npu' not in sys.modules
    report.update(status='passed',npu_backend_imported=False,metric_rechecks=30,outputs_sha256=sha(path),output_array_count=len(saved),
        summary={str(tail):dict(max_relative_l2=max(row['variants'][str(tail)]['relative_l2'] for row in report['cases']),
            quality_passed=sum(row['variants'][str(tail)]['quality_gate_passed'] for row in report['cases'])) for tail in (0,1,2,4,8,30)})
    (out/'tail-precision.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    with torch.no_grad():main()

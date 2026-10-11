"""Reproduce worst continuation errors with the same W8 weights on CPU.

This never imports torch_intel_npu or executes an NPU graph. The staged
reference retains the driver's unrounded INT8-code * FP16-scale semantics.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from w8_full_calibration import OUT,SOURCES,weights
from w8_quality_probe import ROOT,MODEL,sha
from w8_calibration_probe import metric
from torch_npu_smollm_stack_demo import errors


def main():
    from transformers import AutoModelForCausalLM
    from torch_npu_smollm import FrozenSmolStack
    torch.set_num_threads(8);packed=weights()
    source=json.loads((OUT/'decode.json').read_text());assert source['status']=='passed'
    assert sha(OUT/'decode-outputs.npz')==source['outputs_sha256']
    report=dict(status='running',source_sha256=sha(Path(__file__)),sources={name:sha(ROOT/name) for name in SOURCES},
        weights_sha256=sha(OUT/'weights.npz'),decode_report_sha256=sha(OUT/'decode.json'),npu_calls=0,cases=[])
    saved={}
    cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
    stacks={label:FrozenSmolStack.from_checkpoint(MODEL,num_layers=30,mode='fp16' if label=='fp16' else 'w8a16',
        packed_weights=packed if label=='calibrated' else None) for label in ('fp16','original','calibrated')}
    captured=[]
    def collect(module,args,value):captured.append(value.detach().clone())
    hooks=[layer.register_forward_hook(collect) for layer in cpu.model.layers]
    try:
        with np.load(OUT/'decode-outputs.npz',allow_pickle=False) as data:
            for tag in ('bird','garden','cat','repeat32'):
                worst=max((row for row in source['positions'] if row['tag'].startswith(tag+'/')),key=lambda row:row['variants']['calibrated']['relative_l2'])
                length=worst['cache_length'];ids=torch.from_numpy(data[tag+'/tokens'][:,:length].copy())
                baseline=torch.from_numpy(data[worst['tag']+'/baseline-logits'].copy())
                captured.clear();official=cpu(ids,use_cache=False).logits[:,-1].float();assert len(captured)==30
                saved[tag+'/ids']=ids.numpy().copy();saved[tag+'/baseline-logits']=baseline.numpy().copy();saved[tag+'/official-prefill-logits']=official.numpy().copy()
                cpu_cache_error=metric(official,baseline);assert cpu_cache_error['relative_l2']<.03
                entry=dict(tag=tag,position=worst['tag'],cache_length=length,official_prefill_vs_cached_cpu=cpu_cache_error,
                    npu_variants=worst['variants'],cpu_staged_variants={},calibrated_layer_drift=[]);report['cases'].append(entry)
                for label,stack in stacks.items():
                    value=cpu.model.embed_tokens(ids)
                    for i,layer in enumerate(stack.layers):
                        value,_=layer.reference_cpu(value)
                        if label=='calibrated':
                            entry['calibrated_layer_drift'].append(dict(layer=i,**errors(value,captured[i]),
                                last_token=errors(value[:,-1:],captured[i][:,-1:])))
                            saved[f'{tag}/layer-{i}-calibrated']=value.numpy().copy();saved[f'{tag}/layer-{i}-official']=captured[i].numpy().copy()
                    logits=cpu.lm_head(cpu.model.norm(value))[:,-1].float();entry['cpu_staged_variants'][label]=metric(logits,baseline)
                    saved[f'{tag}/{label}-staged-logits']=logits.numpy().copy()
                print(tag,entry['position'],'NPU',worst['variants']['calibrated']['relative_l2'],
                    'CPU same W8 weights',entry['cpu_staged_variants']['calibrated']['relative_l2'],flush=True)
    finally:
        for hook in hooks:hook.remove()
    path=OUT/'cpu-reproduction-outputs.npz';np.savez(path,**saved)
    # Recompute saved evidence and retain a runnable check of the diagnosis.
    with np.load(path,allow_pickle=False) as data:
        for case in report['cases']:
            tag=case['tag'];baseline=torch.from_numpy(data[tag+'/baseline-logits'])
            for label,expected in case['cpu_staged_variants'].items():
                actual=metric(torch.from_numpy(data[f'{tag}/{label}-staged-logits']),baseline)
                for key,value in actual.items():assert np.isclose(value,expected[key],rtol=1e-6,atol=1e-9)
            for row in case['calibrated_layer_drift']:
                key=f'{tag}/layer-{row["layer"]}';actual=errors(torch.from_numpy(data[key+'-calibrated']),torch.from_numpy(data[key+'-official']))
                for name,value in actual.items():assert np.isclose(value,row[name],rtol=1e-6,atol=1e-9)
                last=errors(torch.from_numpy(data[key+'-calibrated'])[:,-1:],torch.from_numpy(data[key+'-official'])[:,-1:])
                for name,value in last.items():assert np.isclose(value,row['last_token'][name],rtol=1e-6,atol=1e-9)
    assert 'torch_intel_npu' not in sys.modules
    report.update(status='passed',npu_backend_imported=False,outputs_sha256=sha(path),output_array_count=len(saved),staged_metric_rechecks=12,layer_drift_rechecks=120)
    (OUT/'cpu-reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    with torch.no_grad():main()

"""Offline check of raw outputs, original tolerances, and source fingerprints."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from w8_quality_probe import ROOT,OUT,MODEL,sha
from torch_npu_lowbit import quantize_weight
from w8_scale_candidate import refine_weight
from torch_npu_smollm_stack_demo import errors


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def same(actual,record):
    for key in ('relative_l2','max_abs','max_scaled_error'):
        assert math.isclose(actual[key],record[key],rel_tol=1e-6,abs_tol=1e-9),(key,actual[key],record[key])


def logits_same(actual,baseline,record):
    same(errors(actual,baseline),record)
    logp=baseline.log_softmax(-1);logq=actual.log_softmax(-1)
    kl=float((logp.exp()*(logp-logq)).sum())
    assert math.isclose(kl,record['next_token_kl'],rel_tol=1e-6,abs_tol=1e-9)
    assert record['next_token_equal']==bool(actual.argmax(-1)==baseline.argmax(-1))


def main():
    torch.set_num_threads(8);summaries=[];array_checks=0;source_checks=0
    for tag,script in [('baseline','w8_quality_probe.py'),('refined','w8_quality_probe.py'),
        ('linear-semantics','w8_linear_semantics_probe.py'),('decode','w8_decode_quality_probe.py')]:
        report=read(f'{tag}.json');assert report['status']=='passed'
        assert report['source_sha256']==sha(OUT/'sources'/script)
        for name,digest in report['sources'].items():
            assert digest==sha(ROOT/name)==sha(OUT/'sources'/name);source_checks+=1
        final=report['final_stats']
        assert final['closed'] and not final['context_failed']
        assert all(final[key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
        path=OUT/f'{tag}-outputs.npz';assert sha(path)==report['outputs_sha256']
        arrays=np.load(path)
        if tag in ('baseline','refined'):
            assert report['quality_gate_relative_l2']==.03 and report['execution_gate_relative_l2']==.003 and report['execution_gate_max_scaled']==.01
            if tag=='refined':assert report['candidate_source_sha256']==sha(OUT/'sources/w8_scale_candidate.py')
            for entry in report['prompts']:
                name=entry['tag'];assert entry['manual_cpu_bridge_bit_equal']
                baseline=torch.from_numpy(arrays[f'{name}/baseline'])
                for mode in ('fp16','w8a16'):
                    actual=torch.from_numpy(arrays[f'{name}/{mode}/npu-logits'])
                    record=entry['variants'][f'npu_{mode}'];logits_same(actual,baseline,record)
                    assert record['quality_gate_passed']==(record['relative_l2']<.03)
                    for local in entry['local_layers']:
                        if local['mode']!=mode:continue
                        i=local['layer'];a=torch.from_numpy(arrays[f'{name}/{mode}/npu-layer-{i}'])
                        b=torch.from_numpy(arrays[f'{name}/{mode}/local-reference-{i}'])
                        same(errors(a,b),local['output']);assert local['output']['relative_l2']<.003 and local['output']['max_scaled_error']<.01
                        array_checks+=1
                assert all(np.isfinite(value).all() for value in (arrays[f'{name}/baseline'],arrays[f'{name}/fp16/npu-logits'],arrays[f'{name}/w8a16/npu-logits']))
            summaries.append(dict(tag=tag,logit_relative_l2={row['tag']:row['variants']['npu_w8a16']['relative_l2'] for row in report['prompts']}))
        elif tag=='linear-semantics':
            for row in report['rows']:
                name=row['name'];actual=torch.from_numpy(arrays[f'{name}/npu'])
                for label,field in [('cpu-unrounded','npu_vs_unrounded_weight'),('cpu-rounded','npu_vs_fp16_dequantized_weight'),('npu-fp16-control','npu_quant_vs_fp16_control')]:
                    same(errors(actual,torch.from_numpy(arrays[f'{name}/{label}'])),row[field]);array_checks+=1
                assert row['npu_vs_unrounded_weight']['relative_l2']<row['npu_vs_fp16_dequantized_weight']['relative_l2']<.001
                assert not row['quant_fp16_control_bit_equal']
                assert {(r['amode'],r['wmode']) for r in row['quant_dpu_modes'] if not r['elementwise']}=={(0,2)}
            summaries.append(dict(tag=tag,projection_count=len(report['rows'])))
        else:
            assert report['quality_gate_relative_l2']==.03 and report['cache_length']==55 and len(report['steps'])==32
            for mode in ('fp16','w8a16'):
                for row in report['steps']:
                    index=row['step'];actual=torch.from_numpy(arrays[f'{index}/{mode}-logits'])
                    baseline=torch.from_numpy(arrays[f'{index}/baseline-logits']);record=row['variants'][mode]
                    logits_same(actual,baseline,record);array_checks+=1
                    assert record['quality_gate_passed']==(record['relative_l2']<.03)
                    assert record['next_token_equal']==bool(actual.argmax(-1)==baseline.argmax(-1))
                summaries.append(dict(tag=f'decode_{mode}',failed_positions=[r['step'] for r in report['steps'] if not r['variants'][mode]['quality_gate_passed']],
                    max_relative_l2=max(r['variants'][mode]['relative_l2'] for r in report['steps']),
                    max_kl=max(r['variants'][mode]['next_token_kl'] for r in report['steps']),next_tokens_equal=sum(r['variants'][mode]['next_token_equal'] for r in report['steps'])))
    baseline_arrays=np.load(OUT/'baseline-outputs.npz');refined_arrays=np.load(OUT/'refined-outputs.npz')
    fp16_repeated=0
    for name in baseline_arrays.files:
        if '/fp16/' in name or name.endswith('/baseline') or '/official-layer-' in name:
            assert np.array_equal(baseline_arrays[name].view(np.uint8),refined_arrays[name].view(np.uint8)),name
            fp16_repeated+=1
    assert fp16_repeated==57
    # The least-squares candidate improves the stored, unrounded W8 matrix
    # objective per row, without claiming it improves model-level quality.
    rows=0;before_mse=0.;after_mse=0.
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as file:
        for name in file.keys():
            if not any(name.startswith(f'model.layers.{i}.') for i in range(4)) or not name.endswith('_proj.weight'):continue
            weight=file.get_tensor(name).float().half();original=quantize_weight(weight,format='int8');candidate=refine_weight(weight)
            value=weight.double().numpy()
            def loss(q):
                codes=np.frombuffer(q.data,dtype=np.int8).reshape(q.shape).astype(np.float64)
                scales=np.frombuffer(q.scales,dtype=np.float16).astype(np.float64)[:,None]
                return np.square(value-codes*scales).sum(-1)
            before=loss(original);after=loss(candidate)
            assert np.all(after<=before+1e-20)
            assert len(original.data)==len(candidate.data) and len(original.scales)==len(candidate.scales)
            rows+=len(before);before_mse+=float(before.sum());after_mse+=float(after.sum())
    assert rows==4*(3*576+2*192+2*1536)
    # Zero and a constant row exercise the zero-denominator/grid limits.
    for value in (torch.zeros(3,64),torch.full((3,64),.125)):
        original=quantize_weight(value,format='int8');candidate=refine_weight(value)
        assert original==candidate
    result=dict(status='passed',source_sha256=sha(Path(__file__)),source_checks=source_checks,array_metric_checks=array_checks,
        fp16_repeated_arrays=fp16_repeated,refined_weight_rows=rows,
        refined_weight_squared_error_ratio=after_mse/before_mse,summary=summaries,
        checkpoint_sha256=sha(MODEL/'model.safetensors'))
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())

"""Offline evidence check for calibration, public API, and NPU outputs."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from w8_calibration import OUT,SOURCES,input_name,load_weights,unpack
from w8_quality_probe import ROOT,MODEL,sha
from w8_quality_audit import same,logits_same
from torch_npu_lowbit import quantize_weight
from torch_npu_smollm_stack_demo import errors


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(8);prepare=read('prepare.json');assert prepare['status']=='passed' and prepare['method']=='gptq'
    assert sha(OUT/'calibration-inputs.npz')==prepare['inputs_sha256']
    assert sha(OUT/'weights.npz')==prepare['weights_sha256']
    assert sha(MODEL/'model.safetensors')==prepare['checkpoint_sha256']
    prepared=load_weights();inputs=np.load(OUT/'calibration-inputs.npz');source_checks=0;array_checks=0
    assert prepare['source_sha256']==sha(ROOT/'npunlock-probe/research/w8_calibration.py')==sha(OUT/'sources/w8_calibration.py')
    for name,digest in prepare['sources'].items():
        assert digest==sha(ROOT/name)==sha(OUT/'sources'/name);source_checks+=1
    before_sse=0.;after_sse=0.;weight_rows=0
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as file:
        for row in prepare['weights']:
            i,name=row['layer'],row['name'];weight=file.get_tensor(f'model.layers.{i}.{name}').float().half()
            original=quantize_weight(weight,format='int8');q=prepared[i][name]
            assert q.scales==original.scales and len(q.data)==len(original.data) and q.shape==original.shape
            activation=torch.from_numpy(inputs[f'{i}/{input_name(name)}']).double();target=activation@weight.double().t()
            def loss(packed):
                c,s=unpack(packed);prediction=(activation@torch.from_numpy(c.astype(np.float64)).t())*torch.from_numpy(s.astype(np.float64))[None]
                return (prediction-target).square().sum(0)
            before=loss(original);after=loss(q)
            assert torch.all(after<=before+1e-18)
            for value,field in [(float(before.sum()),'original_output_squared_error'),(float(after.sum()),'calibrated_output_squared_error')]:
                assert math.isclose(value,row[field],rel_tol=1e-8,abs_tol=1e-10)
            before_sse+=float(before.sum());after_sse+=float(after.sum());weight_rows+=len(before)
    assert weight_rows==20736
    summaries={};calls=0
    for label,script in [('holdout','w8_calibration_probe.py'),('decode','w8_calibration_decode.py'),('checks','w8_calibration_checks.py')]:
        report=read(label+'.json');assert report['status']=='passed'
        assert report['source_sha256']==sha(ROOT/'npunlock-probe/research'/script)==sha(OUT/'sources'/script)
        for name,digest in report['sources'].items():
            assert digest==sha(ROOT/name)==sha(OUT/'sources'/name);source_checks+=1
        final=report['final_stats'];calls+=final['npu_calls']
        assert final['closed'] and not final['context_failed']
        assert all(final[key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
        if label=='checks':
            assert len(report['checks'])==16;continue
        assert report['calibration_weights_sha256' if label=='holdout' else 'weights_sha256']==prepare['weights_sha256']
        assert report['candidate_all_quality_gates_passed']
        path=OUT/f'{label}-outputs.npz';assert sha(path)==report['outputs_sha256'];arrays=np.load(path)
        if label=='holdout':
            assert report['criteria']==dict(logits_relative_l2=.03,next_token_kl=.02,local_execution_relative_l2=.003,local_max_scaled_error=.01)
            calibration=set(map(tuple,prepare['calibration_texts']))
            holdout={(entry['system'],entry['prompt']) for entry in report['prompts']}
            assert len(holdout)==16 and calibration.isdisjoint(holdout)
            rows=[]
            for entry in report['prompts']:
                tag=entry['tag'];baseline=torch.from_numpy(arrays[f'{tag}/baseline-logits'])
                for mode in ('original','calibrated'):
                    actual=torch.from_numpy(arrays[f'{tag}/{mode}/logits']);record=entry['variants'][mode]
                    logits_same(actual,baseline,record);array_checks+=1
                    assert record['quality_gate_passed']==(record['relative_l2']<.03 and record['next_token_kl']<.02)
                    for local in record['local_layers']:
                        i=local['layer'];a=torch.from_numpy(arrays[f'{tag}/{mode}/layer-{i}-actual'])
                        b=torch.from_numpy(arrays[f'{tag}/{mode}/layer-{i}-reference']);same(errors(a,b),local);array_checks+=1
                        assert local['relative_l2']<.003 and local['max_scaled_error']<.01
                        assert all(e['int8_matrix_tasks']>0 for e in local['evidence'])
                        assert all(m['amode']==0 for e in local['evidence'] for m in e['modes'] if m['wmode'] in (1,2) and not m['elementwise'])
                for a,b in zip(entry['variants']['original']['local_layers'],entry['variants']['calibrated']['local_layers']):
                    assert [e['constant_bytes'] for e in a['evidence']]==[e['constant_bytes'] for e in b['evidence']]
                rows.append(dict(tag=tag,split=entry['split'],original=entry['variants']['original']['relative_l2'],
                    calibrated=entry['variants']['calibrated']['relative_l2']))
            summaries[label]=dict(rows=rows,original_failed=sum(not e['variants']['original']['quality_gate_passed'] for e in report['prompts']),
                calibrated_failed=0,calibrated_next_tokens_equal=sum(e['variants']['calibrated']['next_token_equal'] for e in report['prompts']))
        else:
            assert report['quality_gate_relative_l2']==.03 and report['quality_gate_kl']==.02
            rows=[]
            for entry in report['prompts']:
                assert entry['cache_length']==entry['input_tokens']+31 and len(entry['steps'])==32
                for step in entry['steps']:
                    for mode in ('fp16','original','calibrated'):
                        actual=torch.from_numpy(arrays[f'{entry["tag"]}/{step["step"]}/{mode}-logits'])
                        baseline=torch.from_numpy(arrays[f'{entry["tag"]}/{step["step"]}/baseline-logits']);record=step['variants'][mode]
                        logits_same(actual,baseline,record);array_checks+=1
                        assert record['quality_gate_passed']==(record['relative_l2']<.03 and record['next_token_kl']<.02)
                        if step['step'] in (1,15,31):assert record['old_snapshot_unchanged']
                        if step['step']==31:assert record['warm_repeat_bit_equal'] and record['warm_graph_calls']==4 and record['warm_host_copies_and_compiles']==0
                rows.append(dict(tag=entry['tag'],modes={mode:dict(max_l2=max(s['variants'][mode]['relative_l2'] for s in entry['steps']),
                    max_kl=max(s['variants'][mode]['next_token_kl'] for s in entry['steps']),failed=sum(not s['variants'][mode]['quality_gate_passed'] for s in entry['steps']),
                    next_tokens_equal=sum(s['variants'][mode]['next_token_equal'] for s in entry['steps'])) for mode in ('fp16','original','calibrated')}))
            summaries[label]=rows
    repeated=0
    for name in ('weights','calibration-inputs','holdout-outputs','decode-outputs'):
        before=np.load(OUT/'candidate-before-api'/f'{name}.npz');after=np.load(OUT/f'{name}.npz')
        assert set(before.files)==set(after.files)
        for key in before.files:
            assert np.array_equal(before[key].view(np.uint8),after[key].view(np.uint8)),(name,key);repeated+=1
    references=read('references/sources.json')
    for row in references['files']:assert sha(OUT/'references'/row['filename'])==row['sha256']
    assert sha(ROOT/'LICENSES/GPTQ.txt')==sha(OUT/'references/GPTQ_LICENSE')
    old=json.loads((ROOT/'npunlock-probe/research/results/w8-quality/baseline.json').read_text())
    for name in SOURCES:
        if name not in ('torch_npu_lowbit.py','torch_npu_smollm.py'):assert old['sources'][name]==sha(ROOT/name)
    tests=(OUT/'cpu-tests.log').read_text(encoding='utf-8');assert 'Ran 5 tests' in tests and '\nOK' in tests
    result=dict(status='passed',source_sha256=sha(Path(__file__)),source_checks=source_checks,array_metric_checks=array_checks,
        before_after_api_bit_equal_arrays=repeated,weight_rows=weight_rows,calibration_projection_squared_error_ratio=after_sse/before_sse,
        public_phase_npu_calls=calls,summary=summaries,upstream_commit=references['commit'])
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())

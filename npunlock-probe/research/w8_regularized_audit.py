"""Replay the CPU-only damping comparison; verify frozen sources and artifacts."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
import w8_regularized_calibration as experiment
from w8_quality_audit import logits_same
from w8_quality_probe import ROOT,MODEL,sha


def main():
    torch.set_num_threads(8);out=experiment.OUT;report=json.loads((out/'selection.json').read_text(encoding='utf-8'))
    assert report['status']=='passed' and report['npu_calls']==0 and not report['npu_backend_imported']
    assert report['sources']=={name:sha(ROOT/name) for name in experiment.SOURCES}
    assert set(report['fit_trajectory_indices']).isdisjoint(report['validation_trajectory_indices'])
    assert sorted(report['fit_trajectory_indices']+report['validation_trajectory_indices'])==list(range(40))
    assert len(report['fit_trajectory_indices'])==32 and len(report['validation_trajectory_indices'])==8
    assert sha(out/'selection-outputs.npz')==report['outputs_sha256']
    before=json.loads((experiment.BASE/'prepare.json').read_text(encoding='utf-8'))
    assert sha(experiment.BASE/'calibration-inputs.npz')==report['inputs_sha256']==before['inputs_sha256']
    assert sha(MODEL/'model.safetensors')==before['checkpoint_sha256']
    assert sha(experiment.BASE/'weights.npz')==before['weights_sha256']
    for name,digest in before['sources'].items():assert sha(ROOT/name)==digest
    metrics=arrays_checked=0;summaries=[]
    with np.load(out/'selection-outputs.npz',allow_pickle=False) as saved:
        for record in report['candidates']:
            tag=experiment.label(record['damping']);path=out/(tag+'-weights.npz');assert sha(path)==record['weights_sha256']
            manifest=json.loads((out/(tag+'-fit.json')).read_text(encoding='utf-8'));assert manifest['sources']==report['sources']
            assert manifest['inputs_sha256']==report['inputs_sha256'] and manifest['weights_sha256']==record['weights_sha256']
            with np.load(path,allow_pickle=False) as weights:
                assert len(weights.files)==420 and len(record['weights'])==210
                for row in record['weights']:
                    prefix=f'{row["layer"]}/{row["name"]}';codes=weights[prefix+'/codes'];scales=weights[prefix+'/scales']
                    assert codes.dtype==np.int8 and scales.dtype==np.float16 and codes.ndim==2 and scales.shape==(codes.shape[0],)
                    assert np.isfinite(scales).all() and np.all(scales>0) and codes.min()>=-127
                    import hashlib
                    assert hashlib.sha256(codes.tobytes()).hexdigest()==row['calibrated_codes_sha256']
                    assert row['activation_rows']==2048 and row['calibrated_output_squared_error']<=row['original_output_squared_error']
                    arrays_checked+=2
            for row in record['positions']:
                case=next(c for c in report['validation_cases'] if c['index']==row['index']);j=row['position']-case['offset']
                logits_same(torch.from_numpy(saved[f'{row["index"]}/{tag}-logits'])[:,j],
                    torch.from_numpy(saved[f'{row["index"]}/baseline-logits'])[:,j],row)
                assert row['quality_gate_passed']==(row['relative_l2']<.03 and row['next_token_kl']<.02);metrics+=1
            assert record['quality_failed']==sum(not r['quality_gate_passed'] for r in record['positions'])
            summaries.append({key:record[key] for key in ('damping','quality_failed','max_relative_l2','mean_relative_l2',
                'max_kl','next_token_equal','weight_mse_ratio_vs_original')})
    selected=min(report['candidates'],key=lambda r:(r['quality_failed'],r['max_relative_l2'],r['mean_relative_l2']))
    assert report['selected_damping']==selected['damping']==.01 and metrics==1160 and arrays_checked==1680
    experiment.selfcheck();assert 'torch_intel_npu' not in sys.modules
    snapshots=[]
    for name,digest in {**report['sources'],'npunlock-probe/research/w8_regularized_audit.py':sha(Path(__file__))}.items():
        target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        assert sha(target)==digest;snapshots.append(dict(path=name,sha256=digest))
    (out/'snapshots.json').write_text(json.dumps(snapshots,indent=2),encoding='utf-8')
    result=dict(status='passed',scope='CPU-only internal calibration split; no fresh NPU run or deployment change',
        source_sha256=sha(Path(__file__)),selection_sha256=sha(out/'selection.json'),metric_rechecks=metrics,
        weight_arrays_checked=arrays_checked,source_snapshots=len(snapshots),npu_calls=0,npu_backend_imported=False,
        selected_damping=.01,new_regularization_strategy_selected=False,previous_profile_fingerprints_passed=True,candidates=summaries)
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    with torch.no_grad():main()

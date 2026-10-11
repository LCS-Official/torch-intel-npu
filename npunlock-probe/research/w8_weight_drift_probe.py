"""Offline weight displacement audit, using unrounded DPU code*scale weights."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from torch_npu_lowbit import quantize_weight
from w8_quality_probe import ROOT,MODEL,sha

OUT=ROOT/'npunlock-probe/research/results/w8-regularized-calibration'
PROFILES={name:ROOT/'npunlock-probe/research/results'/folder for name,folder in (
    ('prompt349','w8-full-calibration'),('trajectory2560','w8-continuation-calibration'))}


def main():
    torch.set_num_threads(8);OUT.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',npu_calls=0,source_sha256=sha(Path(__file__)),
        checkpoint_sha256=sha(MODEL/'model.safetensors'),profiles={},matrices=[])
    loaded={}
    for name,path in PROFILES.items():
        metadata=json.loads((path/'prepare.json').read_text(encoding='utf-8'))
        assert metadata['status']=='passed' and sha(path/'weights.npz')==metadata['weights_sha256']
        report['profiles'][name]=dict(weights_sha256=metadata['weights_sha256'],prepare_sha256=sha(path/'prepare.json'))
        loaded[name]=np.load(path/'weights.npz',allow_pickle=False)
    try:
        with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as checkpoint:
            for key in loaded['trajectory2560'].files:
                if not key.endswith('/codes'):continue
                prefix=key.removesuffix('/codes');i,name=prefix.split('/',1)
                weight=checkpoint.get_tensor(f'model.layers.{i}.{name}').half().double().numpy()
                original=quantize_weight(torch.from_numpy(weight).half(),format='int8')
                c0=np.frombuffer(original.data,dtype=np.int8).reshape(weight.shape)
                scale=np.frombuffer(original.scales,dtype=np.float16).astype(np.float64)
                e0=((c0*scale[:,None]-weight)**2).sum(1)
                entry=dict(layer=int(i),name=name,weight_squared_norm=float((weight**2).sum()),
                    original_weight_squared_error=float(e0.sum()),profiles={})
                for profile,arrays in loaded.items():
                    codes=arrays[key];np.testing.assert_array_equal(arrays[prefix+'/scales'].astype(np.float64),scale)
                    error=((codes*scale[:,None]-weight)**2).sum(1)
                    entry['profiles'][profile]=dict(weight_squared_error=float(error.sum()),
                        weight_mse_ratio_vs_original=float(error.sum()/e0.sum()),
                        max_row_weight_mse_ratio=float((error/np.maximum(e0,1e-30)).max()),
                        changed_code_fraction=float(np.mean(codes!=c0)))
                report['matrices'].append(entry)
    finally:
        for data in loaded.values():data.close()
    assert len(report['matrices'])==210
    report['summary']={name:dict(weight_mse_ratio_vs_original=sum(r['profiles'][name]['weight_squared_error'] for r in report['matrices'])/
        sum(r['original_weight_squared_error'] for r in report['matrices']),worst_matrices=sorted(report['matrices'],
        key=lambda r:r['profiles'][name]['weight_mse_ratio_vs_original'],reverse=True)[:7]) for name in loaded}
    cpu=json.loads((PROFILES['trajectory2560']/'cpu-reproduction.json').read_text(encoding='utf-8'))
    report['cpu_reproduction_sha256']=sha(PROFILES['trajectory2560']/'cpu-reproduction.json')
    report['last_token_drift']={row['position']:[dict(layer=r['layer'],relative_l2=r['last_token']['relative_l2'])
        for r in row['calibrated_layer_drift']] for row in cpu['cases']}
    assert 'torch_intel_npu' not in sys.modules
    report.update(status='passed',npu_backend_imported=False)
    (OUT/'weight-drift.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for name,row in report['summary'].items():
        print(name,'total weight MSE ratio',row['weight_mse_ratio_vs_original'],flush=True)
        for r in row['worst_matrices'][:4]:print(r['layer'],r['name'],r['profiles'][name],flush=True)
    for name,rows in report['last_token_drift'].items():print(name,[(r['layer'],round(r['relative_l2']*100,2)) for r in rows],flush=True)


if __name__=='__main__':main()

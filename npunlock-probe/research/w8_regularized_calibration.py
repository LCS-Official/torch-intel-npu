"""Choose GPTQ damping on calibration-only trajectories, then refit all 2560 rows.

Research-only adaptation of torch_npu_lowbit.calibrate_int8 (LICENSES/GPTQ.txt).
The public calibrator, historic profiles, FP16 scales and DPU kernels are frozen.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from safetensors import safe_open
import w8_continuation_calibration as previous
import w8_full_calibration as full
from w8_calibration import input_name,unpack
from w8_calibration_probe import metric
from w8_quality_probe import ROOT,MODEL,sha

BASE=previous.OUT
OUT=ROOT/'npunlock-probe/research/results/w8-regularized-calibration'
SOURCES=previous.SOURCES+['npunlock-probe/research/w8_regularized_calibration.py',
    'npunlock-probe/research/w8_weight_drift_probe.py']
DAMPINGS=(.01,.1,.3,1.)
VALIDATION=tuple(i for i in range(40) if i%5==4)


def configure():
    full.OUT=OUT;full.SOURCES=SOURCES;full.CALIBRATION=previous.PROMPTS
    previous.OUT=OUT;previous.SOURCES=SOURCES;previous.BEFORE=BASE


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def label(damping):return 'damping-'+str(damping)


def calibrate(weight,inputs,damping):
    from torch_npu_lowbit import PackedWeight,quantize_weight
    from npunlock.npu4000 import MAX_TENSOR_ELEMENTS
    original=quantize_weight(weight,format='int8')
    assert inputs.ndim==2 and inputs.device.type=='cpu' and inputs.dtype in (torch.float16,torch.float32)
    assert inputs.shape[1]==weight.shape[1] and 0<inputs.shape[0]<=4096 and inputs.shape[1]<=4096
    assert max(inputs.numel(),len(inputs)*len(weight))<=4*MAX_TENSOR_ELEMENTS and torch.isfinite(inputs).all()
    assert damping in DAMPINGS
    x=inputs.double();h=x.t()@x/len(x);energy=h.diagonal().clone()
    order=torch.argsort(energy,descending=True,stable=True);h=h[order][:,order];h.diagonal()[energy[order]==0]=1
    h.diagonal().add_(damping*h.diagonal().mean().clamp_min(1e-20))
    upper=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)),upper=True)
    values=weight.double()[:,order].clone();scales=torch.from_numpy(np.frombuffer(original.scales,dtype=np.float16).astype(np.float64))
    codes=torch.empty_like(values,dtype=torch.int8);width=weight.shape[1]
    for start in range(0,width,128):
        end=min(start+128,width);block=values[:,start:end].clone();residual=torch.zeros_like(block)
        for i in range(end-start):
            column=block[:,i].clone();code=(column/scales).round().clamp(-127,127);codes[:,start+i]=code.to(torch.int8)
            error=(column-code*scales)/upper[start+i,start+i]
            block[:,i:].sub_(error[:,None]*upper[start+i,start+i:end][None]);residual[:,i]=error
        values[:,end:].sub_(residual@upper[start:end,end:])
    proposed=codes[:,torch.argsort(order)].contiguous().numpy();original_codes,_=unpack(original)
    before=loss(weight,inputs,original_codes,scales.numpy());after=loss(weight,inputs,proposed,scales.numpy())
    original_codes[after<before]=proposed[after<before]
    return PackedWeight(original.shape,'int8',original.group_size,original_codes.tobytes(),original.scales)


def loss(weight,inputs,codes,scales):
    # DPU fuses scale with MatMul: do not round dequantized weights to FP16.
    error=torch.from_numpy(codes.astype(np.float64)*scales.astype(np.float64)[:,None])-weight.double()
    return (inputs.double()@error.t()).square().sum(0).numpy()


def selfcheck():
    from torch_npu_lowbit import calibrate_int8
    generator=torch.Generator().manual_seed(1907)
    for width in (17,192):
        weight=torch.randn(23,width,generator=generator).half();x=torch.randn(47,width,generator=generator).half()
        weight[0]=0;x[:,3]=0
        assert calibrate(weight,x,.01)==calibrate_int8(weight,x)
    assert 'torch_intel_npu' not in sys.modules


def fit(damping,indices,path):
    from torch_npu_lowbit import quantize_weight
    arrays={};rows=[];weight_sse=original_sse=0.
    with np.load(BASE/'calibration-inputs.npz',allow_pickle=False) as captured, safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as checkpoint:
        for i in range(30):
            for name in ('self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
                         'self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight'):
                weight=checkpoint.get_tensor(f'model.layers.{i}.{name}').half().contiguous()
                all_inputs=captured[f'{i}/{input_name(name)}'];assert len(all_inputs)==2560
                observed=torch.from_numpy(all_inputs.reshape(40,64,-1)[list(indices)].reshape(-1,all_inputs.shape[-1]).copy())
                original=quantize_weight(weight,format='int8');q=calibrate(weight,observed,damping)
                c0,s0=unpack(original);c,s=unpack(q);np.testing.assert_array_equal(s,s0)
                before=loss(weight,observed,c0,s0);after=loss(weight,observed,c,s)
                assert np.all(after<=before+1e-18) and c.min()>=-127
                original_error=float(((c0.astype(np.float64)*s0.astype(np.float64)[:,None]-weight.double().numpy())**2).sum())
                calibrated_error=float(((c.astype(np.float64)*s.astype(np.float64)[:,None]-weight.double().numpy())**2).sum())
                weight_sse+=calibrated_error;original_sse+=original_error
                key=f'{i}/{name}';arrays[key+'/codes']=c;arrays[key+'/scales']=s
                rows.append(dict(layer=i,name=name,rows=len(s),activation_rows=len(observed),changed_rows=int(np.any(c0!=c,axis=1).sum()),
                    original_output_squared_error=float(before.sum()),calibrated_output_squared_error=float(after.sum()),
                    original_weight_squared_error=original_error,calibrated_weight_squared_error=calibrated_error,
                    original_codes_sha256=hashlib.sha256(original.data).hexdigest(),calibrated_codes_sha256=hashlib.sha256(q.data).hexdigest()))
            print('fit',label(damping),'layer',i+1,'/30',len(indices)*64,'rows',flush=True)
    assert len(rows)==210 and len(arrays)==420;np.savez(path,**arrays)
    return dict(damping=damping,weights_sha256=sha(path),weights=rows,weight_mse_ratio_vs_original=weight_sse/original_sse)


def load(path):
    from torch_npu_lowbit import PackedWeight
    packed={i:{} for i in range(30)}
    with np.load(path,allow_pickle=False) as arrays:
        for key in arrays.files:
            if not key.endswith('/codes'):continue
            prefix=key.removesuffix('/codes');i,name=prefix.split('/',1);codes=arrays[key];scales=arrays[prefix+'/scales']
            packed[int(i)][name]=PackedWeight(codes.shape,'int8',codes.shape[1],codes.tobytes(),scales.tobytes())
    return packed


def select():
    from transformers import AutoModelForCausalLM
    from torch_npu_smollm import FrozenSmolStack
    start=time.perf_counter();metadata=json.loads((BASE/'prepare.json').read_text(encoding='utf-8'))
    assert sha(BASE/'calibration-inputs.npz')==metadata['inputs_sha256'] and metadata['activation_rows']==2560
    assert metadata['checkpoint_sha256']==sha(MODEL/'model.safetensors')
    report=dict(status='running',sources={name:sha(ROOT/name) for name in SOURCES},npu_calls=0,
        base_prepare_sha256=sha(BASE/'prepare.json'),inputs_sha256=metadata['inputs_sha256'],
        fit_trajectory_indices=[i for i in range(40) if i not in VALIDATION],validation_trajectory_indices=VALIDATION,
        policy='Fit 32 calibration trajectories; validate remaining 8 teacher-forced CPU FP16 trajectories. '
            'Rank by failed-position count, max relative L2, then mean relative L2. Refit chosen damping on all 40. '
            'Independent holdout/decode prompts never enter fitting or selection.',
        criteria=dict(logits_relative_l2=.03,next_token_kl=.02),candidates=[],validation_cases=[])
    selfcheck()
    for damping in DAMPINGS:
        path=OUT/(label(damping)+'-weights.npz');checkpoint=OUT/(label(damping)+'-fit.json')
        if checkpoint.exists():
            record=json.loads(checkpoint.read_text(encoding='utf-8'))
            assert record['sources']==report['sources'] and record['inputs_sha256']==report['inputs_sha256']
            assert record['weights_sha256']==sha(path) and record['fit_trajectory_indices']==report['fit_trajectory_indices']
            print('reuse verified fit',damping,flush=True)
        else:
            record=fit(damping,report['fit_trajectory_indices'],path)
            record.update(sources=report['sources'],inputs_sha256=report['inputs_sha256'],fit_trajectory_indices=report['fit_trajectory_indices'])
            checkpoint.write_text(json.dumps(record,indent=2),encoding='utf-8')
        report['candidates'].append(record)
    cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
    saved={};cases=[]
    for index in VALIDATION:
        row=metadata['inputs'][index];ids=torch.tensor(row['token_ids'],dtype=torch.long);offset=row['input_tokens']-1
        baseline=cpu(ids,use_cache=False).logits[:,offset:].float();cases.append((index,ids,offset,baseline))
        saved[f'{index}/ids']=ids.numpy();saved[f'{index}/baseline-logits']=baseline.numpy()
        report['validation_cases'].append(dict(index=index,system=row['system'],prompt=row['prompt'],offset=offset,positions=baseline.shape[1]))
    for record in report['candidates']:
        damping=record['damping'];stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=30,mode='w8a16',packed_weights=load(OUT/(label(damping)+'-weights.npz')))
        metrics=[]
        for index,ids,offset,baseline in cases:
            value=cpu.model.embed_tokens(ids)
            for layer in stack.layers:value,_=layer.reference_cpu(value)
            logits=cpu.lm_head(cpu.model.norm(value))[:,offset:].float();saved[f'{index}/{label(damping)}-logits']=logits.numpy()
            for position in range(logits.shape[1]):metrics.append(dict(index=index,position=position+offset,**metric(logits[:,position],baseline[:,position])))
        record.update(positions=metrics,quality_failed=sum(r['relative_l2']>=.03 or r['next_token_kl']>=.02 for r in metrics),
            max_relative_l2=max(r['relative_l2'] for r in metrics),mean_relative_l2=float(np.mean([r['relative_l2'] for r in metrics])),
            max_kl=max(r['next_token_kl'] for r in metrics),next_token_equal=sum(r['next_token_equal'] for r in metrics))
        print('validation',damping,'failed',record['quality_failed'],'/',len(metrics),'max',record['max_relative_l2'],
            'mean',record['mean_relative_l2'],'weight MSE ratio',record['weight_mse_ratio_vs_original'],flush=True)
        del stack;gc.collect()
    selected=min(report['candidates'],key=lambda r:(r['quality_failed'],r['max_relative_l2'],r['mean_relative_l2']))
    report['selected_damping']=selected['damping'];path=OUT/'selection-outputs.npz';np.savez(path,**saved)
    # One replayable check covers the candidate ranking and saved CPU logits.
    from w8_quality_audit import logits_same
    with np.load(path,allow_pickle=False) as data:
        checks=0
        for record in report['candidates']:
            for row in record['positions']:
                offset=next(c['offset'] for c in report['validation_cases'] if c['index']==row['index']);j=row['position']-offset
                logits_same(torch.from_numpy(data[f'{row["index"]}/{label(record["damping"])}-logits'])[:,j],
                    torch.from_numpy(data[f'{row["index"]}/baseline-logits'])[:,j],row);checks+=1
    assert 'torch_intel_npu' not in sys.modules
    report.update(status='passed',npu_backend_imported=False,metric_rechecks=checks,outputs_sha256=sha(path),elapsed_seconds=time.perf_counter()-start)
    full.write('selection',report);print('selected damping',selected['damping'],flush=True)


def prepare():
    start=time.perf_counter();selection=read('selection.json');assert selection['status']=='passed'
    assert selection['sources']=={name:sha(ROOT/name) for name in SOURCES}
    assert sha(OUT/'selection-outputs.npz')==selection['outputs_sha256'];damping=selection['selected_damping']
    metadata=json.loads((BASE/'prepare.json').read_text(encoding='utf-8'))
    assert sha(BASE/'calibration-inputs.npz')==metadata['inputs_sha256']==selection['inputs_sha256']
    for name,digest in metadata['sources'].items():assert sha(ROOT/name)==digest
    record=fit(damping,range(40),OUT/'weights.npz')
    # Keep manifest compatible with the existing strict loader and quality auditor.
    report={**metadata,**record,'sources':selection['sources'],'method':'gptq-regularized',
        'calibrator':'w8_regularized_calibration.calibrate','selection_sha256':sha(OUT/'selection.json'),
        'gptq_parameters':dict(damping=damping,block_size=128,activation_order=True,solver_dtype='float64',row_scale='original_fp16'),
        'policy':'Reuse bit-identical 40 CPU FP16 calibration trajectories; damping chosen only on internal 32/8 trajectory split',
        'elapsed_seconds':time.perf_counter()-start,'status':'passed'}
    shutil.copyfile(BASE/'calibration-inputs.npz',OUT/'calibration-inputs.npz')
    snapshots=[]
    for name,digest in report['sources'].items():
        assert sha(ROOT/name)==digest;target=OUT/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        snapshots.append(dict(path=name,sha256=digest))
    (OUT/'snapshots.json').write_text(json.dumps(snapshots,indent=2),encoding='utf-8');full.write('prepare',report)
    print('refit damping',damping,'weight MSE ratio',record['weight_mse_ratio_vs_original'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('select','prepare','holdout','decode','diagnostic','cpu','audit'),required=True);args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';torch.set_num_threads(8);OUT.mkdir(parents=True,exist_ok=True);configure()
    with torch.no_grad():
        if args.case=='select':select();return 0
        if args.case=='prepare':prepare();return 0
        if args.case=='audit':previous.audit();return 0
        if args.case=='cpu':
            import w8_full_cpu_diagnostic as check
            check.OUT=OUT;check.SOURCES=SOURCES;check.weights=full.weights;check.main();return 0
        return full.evaluate(args.case)


if __name__=='__main__':raise SystemExit(main())

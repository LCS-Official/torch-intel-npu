"""Full 30-layer controls, isolated per precision/batch; reuse the API checks."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import statistics as st
import numpy as np
import decode_chain_api as common
from w8_quality_probe import ROOT, MODEL, sha
from shave_fp32_pipeline import final

OUT=ROOT/'npunlock-probe/research/results/decode-chain-full'


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def snapshot():
    before=OUT/'before';before.mkdir(parents=True,exist_ok=True)
    state=common.provenance()
    state.update({name:sha(ROOT/name) for name in ('TORCH_NPU_README.md','torch_npu_warm_profile.py',
        'torch_npu_device_profile.py','npunlock-probe/research/shave_fp32_pipeline.py',
        'npunlock-probe/research/decode_chain_api.py','npunlock-probe/research/decode_chain_full.py')})
    if not (before/'sources.json').exists():
        for name in state:
            target=before/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        write(before/'sources.json',state)
    for name,digest in read(before/'sources.json').items():assert sha(before/name)==digest


def benchmark(mode,batch):
    snapshot();directory=OUT/f'{mode}-b{batch}';directory.mkdir(exist_ok=True)
    common.OUT=directory
    common.models([(mode,batch,30,32)],steps=32,boundary=True)
    report=read(directory/'models.json')
    report['runner_sha256']=sha(Path(__file__));report['shared_runner_sha256']=sha(Path(common.__file__))
    write(directory/'models.json',report)


def defaults():
    import torch
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    torch.set_num_threads(8);torch.manual_seed(2039)
    report=dict(status='running',sources=common.provenance(),checks=[]);saved={}
    def work():
        for mode,batch in [(m,b) for m in ('fp16','w8a16') for b in (1,4,8)]+[('fp16',2)]:
            tag=f'{mode}-b{batch}';stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,mode=mode)
            host=torch.randn(batch,8,576).half();x=host.to('intel_npu');one=host[:,:1].contiguous().to('intel_npu')
            _,seed=stack.prefill(x)
            expected=stack.decode(one,seed,group_size=1);anchor=common.arrays(expected);del expected
            before=npu.stats();value=stack.decode(one,seed);change=common.delta(before,npu.stats())
            actual=common.arrays(value);common.equal(anchor,actual)
            assert change['chain_graphs']==(4 if batch in (1,4,8) else 0)
            assert change['chain_groups']==(1 if batch in (1,4,8) else 0)
            report['checks'].append(dict(tag=tag,bit_equal_arrays=13,delta=change))
            for i,(a,b) in enumerate(zip(anchor,actual)):
                saved[f'{tag}/baseline/{i}']=a;saved[f'{tag}/default/{i}']=b
            if (mode,batch)==('fp16',1):
                stack.layers[-1]._whole_layer=False
                expected=stack.decode(one,seed,group_size=1);before=npu.stats();value=stack.decode(one,seed)
                assert common.delta(before,npu.stats())['chain_graphs']==0
                common.equal(common.arrays(expected),common.arrays(value));del expected
                report['checks'].append(dict(tag='unfused-default-fallback',bit_equal_arrays=13))
            del stack,seed,x,one,value
    with torch.no_grad():work();final(npu,report)
    np.savez_compressed(OUT/'default-outputs.npz',**saved)
    report.update(status='passed',outputs_sha256=sha(OUT/'default-outputs.npz'))
    write(OUT/'defaults.json',report);print('default selection checks passed',len(report['checks']),flush=True)


def audit():
    baseline=read(OUT/'before/sources.json');summary=[];comparisons=0
    changed={'torch_npu_smollm.py','npunlock-probe/research/decode_chain_api.py',
             'npunlock-probe/research/shave_fp32_pipeline.py','TORCH_NPU_README.md'}
    for name,digest in baseline.items():
        assert sha(OUT/'before'/name)==digest
        if name not in changed:assert sha(ROOT/name)==digest,name
    for mode in ('fp16','w8a16'):
        for batch in (1,4,8):
            tag=f'{mode}-b{batch}';directory=OUT/tag;record=read(directory/'models.json')
            assert record['status']=='passed' and len(record['cases'])==1
            for name,digest in record['sources'].items():assert digest==baseline[name],name
            assert record['shared_runner_sha256']==baseline['npunlock-probe/research/decode_chain_api.py']
            assert record['runner_sha256']==sha(Path(__file__))
            assert sha(directory/'models-outputs.npz')==record['outputs_sha256']
            final_stats=record['final_stats'];assert final_stats['closed'] and not final_stats['context_failed']
            assert all(final_stats[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
            case=record['cases'][0];assert case['cold_bit_equal_arrays']==91
            assert case['continuous_decode_steps']==32 and case['continuous_final_position']==64
            assert case['kv_boundary']=='actual prefill63 -> decode64 -> reject65'
            with np.load(directory/'models-outputs.npz',allow_pickle=False) as arrays:
                common.equal([arrays[f'final/{mode}-b{batch}-l30/baseline/{i}'] for i in range(91)],
                             [arrays[f'final/{mode}-b{batch}-l30/group4/{i}'] for i in range(91)])
                comparisons+=91
                assert all(arrays[f'final/{mode}-b{batch}-l30/group4/{3+3*layer}'].item()==64 for layer in range(30))
            timing=case['timing'];assert len(timing['samples'])==84 and len(timing['pairs'])==21
            for group in (1,4):assert st.median(r['total_ms'] for r in timing['samples'] if r['group']==group)==timing['median_total_ms'][str(group)]
            delta=timing['warm_delta'];assert delta['npu_calls']==2520 and delta['chain_graphs']==1260 and delta['chain_groups']==336
            assert all(v==0 for k,v in delta.items() if k not in ('npu_calls','chain_graphs','chain_groups'))
            a=timing['median_total_ms']['1'];b=timing['median_total_ms']['4']
            wins=sum(p['4']<p['1'] for p in timing['pairs']);assert wins==timing['positive_rounds']
            summary.append(dict(tag=tag,original_ms=a,group4_ms=b,reduction_percent=100*(1-b/a),positive_rounds=wins))
    defaults=read(OUT/'defaults.json');assert defaults['status']=='passed' and len(defaults['checks'])==8
    for name,digest in defaults['sources'].items():assert sha(ROOT/name)==digest
    assert sha(OUT/'default-outputs.npz')==defaults['outputs_sha256']
    final_stats=defaults['final_stats'];assert final_stats['closed'] and not final_stats['context_failed']
    assert all(final_stats[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
    with np.load(OUT/'default-outputs.npz',allow_pickle=False) as arrays:
        for tag in [f'{m}-b{b}' for m in ('fp16','w8a16') for b in (1,4,8)]+['fp16-b2']:
            common.equal([arrays[f'{tag}/baseline/{i}'] for i in range(13)],[arrays[f'{tag}/default/{i}'] for i in range(13)]);comparisons+=13
    old=(OUT/'before/torch_npu_smollm.py').read_text(encoding='utf-8');current=(ROOT/'torch_npu_smollm.py').read_text(encoding='utf-8')
    old=old.replace('# ponytail: default enabled only for measured B1 FP16 decode; widen after model benchmarks.',
                    '# ponytail: default limited to measured batches; expand after full-stack benchmarks.')
    old=old.replace("if supported and self._mode=='fp16' and x.shape[0]==1 else 1",'if supported and self.num_layers>1 and x.shape[0] in (1,4,8) else 1')
    assert old==current,'Unexpected model computation change'
    result=dict(status='passed',full_model_cases=6,continuous_steps_per_case=32,final_position=64,
                default_checks=8,exact_saved_array_comparisons=comparisons,timings=summary,
                native_runtime_and_kernels_unchanged=True,default_only_model_change=True,
                source_sha256=sha(Path(__file__)),current_model_sha256=sha(ROOT/'torch_npu_smollm.py'))
    write(OUT/'audit.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('benchmark','defaults','audit'),required=True)
    parser.add_argument('--mode',choices=('fp16','w8a16'),default='fp16');parser.add_argument('--batch',type=int,choices=(1,4,8),default=1)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.case=='benchmark':benchmark(args.mode,args.batch)
    elif args.case=='defaults':defaults()
    else:audit()

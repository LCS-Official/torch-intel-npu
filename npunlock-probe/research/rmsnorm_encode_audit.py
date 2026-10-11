"""Audit the measured candidates, promoted kernel and current model checks."""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import torch
from npunlock.npu4000 import kernel_text
from rmsnorm_encode import OUT, ROOT
from w8_quality_probe import sha
from w8_quality_audit import logits_same
from torch_npu_smollm_stack_demo import errors


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    fixture=ROOT/'npunlock/tests/fixtures/npu4000';manifest=read(fixture/'rmsnorm/manifest.json')
    winner=read(OUT/'mask/build.json');old=winner['production']['sources'];calls=0;comparisons=0;timings=[]
    def closed(report):
        nonlocal calls
        assert report['status']=='passed';s=report['final_stats'];calls+=s['npu_calls']
        assert s['closed'] and not s['context_failed']
        assert all(s[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
    changed={'npunlock/tests/fixtures/npu4000/rmsnorm.c','npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf'}
    for name,digest in old.items():
        if name not in changed:assert sha(ROOT/name)==digest,name
    for name,digest in manifest['source_sha256'].items():assert sha(fixture/name)==digest
    for name,field in (('rmsnorm-f16.elf','elf_sha256'),('rmsnorm-f16.s','assembly_sha256')):
        assert sha(fixture/'rmsnorm'/name)==manifest[field]==winner['kernels']['codec'][field]
    assert sha(fixture/'rmsnorm.c')==winner['kernels']['codec']['source_sha256']
    assert (fixture/'rmsnorm.c').read_bytes()==(OUT/'mask/codec.c').read_bytes()
    assert manifest['text_bytes']==4528
    for name in ('rmsnorm.c','rmsnorm-f16.elf'):
        expected=old['npunlock/tests/fixtures/npu4000/'+('rmsnorm/'+name if name.endswith('.elf') else name)]
        assert sha(OUT/'before'/name)==expected
    assert sha(fixture/'rmsnorm/FP16_LICENSE')==sha(OUT/'before/FP16_LICENSE')
    previous=(OUT/'before/rmsnorm.c').read_text(encoding='utf-8');current=(fixture/'rmsnorm.c').read_text(encoding='utf-8')
    marker='static inline HalfBits4 encode4(';entry='void controlled_act('
    assert previous[:previous.index(marker)]==current[:current.index(marker)]
    assert previous[previous.index(entry):]==current[current.index(entry):], 'Arithmetic or MemRef guards changed'

    reports=[('mask','numeric','probe'),('lanes','numeric','probe'),('mask','model','process0'),('lanes','model','process0'),('confirmation/mask','model','confirm')]
    for directory,case,label in reports:
        base=OUT/directory;build=read(base/'build.json');report=read(base/f'{case}-{label}.json');closed(report)
        assert build['builder_sha256']==sha(ROOT/'npunlock-probe/research/rmsnorm_encode.py')==report['wrapper_sha256']
        assert report['source_sha256']==sha(ROOT/'npunlock-probe/research/rmsnorm_reduce_probe.py')
        assert report['production']==build['production'] and report['build']==build
        for name,record in build['kernels'].items():
            path=base/f'{name}.elf';assert sha(path)==record['elf_sha256'] and len(kernel_text(path.read_bytes()))==record['text_bytes']
        if case=='numeric':
            assert len(report['checks'])==104 and all(check['bit_differences']==0 for check in report['checks'])
            for rows,records in report['carrier_evidence'].items():
                layout=None
                for patch in records:
                    data=(base/f"norm-{rows}-{patch['kernel']}.blob").read_bytes();a=patch['code_offset'];b=a+patch['window_bytes']
                    digest=hashlib.sha256(data[:a]+bytes(b-a)+data[b:]).hexdigest()
                    if layout is None:layout=digest
                    assert digest==layout==patch['layout_sha256']
                    assert hashlib.sha256(data).hexdigest()==patch['patched_sha256']
        else:
            assert len(report['checks'])==4
            for check in report['checks']:
                assert check['bit_equal_arrays']==26 and check['seed_preserved']
                path=base/f"{check['tag']}-outputs.npz";assert sha(path)==check['outputs_sha256']
                with np.load(path,allow_pickle=False) as arrays:
                    for i in range(26):
                        np.testing.assert_array_equal(arrays[f"{check['tag']}/baseline/{i}"].view(np.uint8),arrays[f"{check['tag']}/codec/{i}"].view(np.uint8));comparisons+=1
                    for i in range(12):
                        np.testing.assert_array_equal(arrays[f"{check['tag']}/baseline/{i+1}"].view(np.uint8),arrays[f"{check['tag']}/preserved/{i}"].view(np.uint8));comparisons+=1
            for evidence in report['carrier_evidence']:
                patch=evidence['baseline'];a=patch['code_offset'];b=a+patch['window_bytes']
                before=(base/f"{evidence['tag']}-baseline.blob").read_bytes();after=(base/f"{evidence['tag']}-codec.blob").read_bytes()
                assert before[:a]+before[b:]==after[:a]+after[b:]
                for name,data in (('baseline',before),('codec',after)):
                    assert hashlib.sha256(data).hexdigest()==evidence[name]['patched_sha256']
                    text=kernel_text((base/f'{name}.elf').read_bytes());assert data[a:a+len(text)]==text
        for row in report['timings']:
            rounds=report['rounds'];samples=row['samples_ms'];pairs=row['pairs_ms']
            assert len(pairs)==rounds and all(len(values)==rounds*2 for values in samples.values())
            assert row['median_ms']=={name:statistics.median(values) for name,values in samples.items()}
            for name,values in samples.items():
                assert all(statistics.mean(values[2*i:2*i+2])==pairs[i][name] for i in range(rounds))
            assert row['positive_rounds']['codec']==sum(p['codec']<p['baseline'] for p in pairs)
            assert row['warm_delta']['npu_calls']==rounds*(4 if case=='numeric' else 16)
            baseline=row['median_ms']['baseline'];candidate=row['median_ms']['codec']
            timings.append(dict(candidate=directory,case=case,tag=row.get('tag',str(row.get('rows'))),operation=row.get('operation','rmsnorm'),
                baseline_ms=baseline,candidate_ms=candidate,latency_reduction_percent=100*(1-candidate/baseline),
                positive_rounds=row['positive_rounds']['codec'],rounds=rounds))
    regressions=[]
    for module,cases in (('torch_npu_smollm_demo',('norm','numeric','boundary','guards','causality','hybrid')),('torch_npu_smollm_stack_demo',('numeric','boundary'))):
        for case in cases:
            report=read(OUT/'regressions'/module/f'{case}.json');closed(report)
            assert report['kernel_manifest']==manifest
            for name,digest in report['sources'].items():assert sha(ROOT/name)==digest
            regressions.append(dict(module=module,case=case,checks=len(report['checks']),npu_calls=report['final_stats']['npu_calls']))

    calibration=OUT/'calibrated-current';prepare=read(calibration/'prepare.json')
    assert prepare['status']=='passed' and prepare['method']=='gptq'
    assert prepare['entry_runner_sha256']==sha(ROOT/'npunlock-probe/research/rmsnorm_encode_checks.py')
    assert prepare['source_sha256']==sha(ROOT/'npunlock-probe/research/w8_calibration.py')
    for name,digest in prepare['sources'].items():assert sha(ROOT/name)==digest
    assert sha(calibration/'weights.npz')==prepare['weights_sha256'] and sha(calibration/'calibration-inputs.npz')==prepare['inputs_sha256']
    assert sha(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors')==prepare['checkpoint_sha256']
    matching=0
    for name in ('weights','calibration-inputs','holdout-outputs','decode-outputs'):
        with np.load(ROOT/f'npunlock-probe/research/results/w8-calibration/{name}.npz',allow_pickle=False) as before, np.load(calibration/f'{name}.npz',allow_pickle=False) as after:
            assert set(before.files)==set(after.files)
            for key in before.files:np.testing.assert_array_equal(before[key].view(np.uint8),after[key].view(np.uint8));matching+=1
    quality={};metrics=0
    for case in ('holdout','decode'):
        report=read(calibration/f'{case}.json');closed(report)
        assert report['entry_runner_sha256']==prepare['entry_runner_sha256'] and report['sources']==prepare['sources']
        script='w8_calibration_probe.py' if case=='holdout' else 'w8_calibration_decode.py'
        assert report['source_sha256']==sha(ROOT/'npunlock-probe/research'/script)
        if case=='holdout':
            assert report['criteria']==dict(logits_relative_l2=.03,next_token_kl=.02,local_execution_relative_l2=.003,local_max_scaled_error=.01)
        else:
            assert report['quality_gate_relative_l2']==.03 and report['quality_gate_kl']==.02
        assert report['candidate_all_quality_gates_passed']
        path=calibration/f'{case}-outputs.npz';assert sha(path)==report['outputs_sha256']
        values=[]
        with np.load(path,allow_pickle=False) as arrays:
            for prompt in report['prompts']:
                if case=='holdout':
                    entries=[(prompt['variants'],f"{prompt['tag']}/",f"{prompt['tag']}/baseline-logits")]
                else:
                    assert prompt['cache_length']==55 and len(prompt['steps'])==32
                    entries=[(step['variants'],f"{prompt['tag']}/{step['step']}/",f"{prompt['tag']}/{step['step']}/baseline-logits") for step in prompt['steps']]
                for variants,prefix,baseline_key in entries:
                    baseline=torch.from_numpy(arrays[baseline_key])
                    for mode,record in variants.items():
                        key=prefix+(mode+'/logits' if case=='holdout' else mode+'-logits')
                        logits_same(torch.from_numpy(arrays[key]),baseline,record);metrics+=1
                    values.append(variants['calibrated'])
        assert all(r['relative_l2']<.03 and r['next_token_kl']<.02 for r in values)
        quality[case]=dict(positions=len(values),max_relative_l2=max(r['relative_l2'] for r in values),next_tokens_equal=sum(r['next_token_equal'] for r in values))
    snapshots=read(OUT/'snapshots.json')
    for name,digest in snapshots['sources'].items():assert sha(ROOT/name)==digest==sha(OUT/'sources'/name)
    for name,digest in snapshots['after'].items():assert sha(fixture/name)==digest==sha(OUT/'after'/Path(name).name)
    result=dict(status='passed',default_kernel_elf_sha256=manifest['elf_sha256'],default_text_bytes=manifest['text_bytes'],
        total_npu_calls=calls,candidate_model_bit_comparisons=comparisons,before_after_default_bit_equal_arrays=matching,
        recomputed_quality_metrics=metrics,regressions=regressions,regression_checks=sum(r['checks'] for r in regressions),
        quality=quality,timings=timings,snapshot_files=sum(len(group) for group in snapshots.values()),auditor_sha256=sha(Path(__file__)))
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

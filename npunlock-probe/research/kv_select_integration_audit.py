"""Recheck default Select integration and saved outputs without NPU execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
from npunlock.npu4000 import kernel_text, patch_kv_select
from kv_select_checks import OUT, ROOT, sha
from w8_quality_audit import logits_same, same
from torch_npu_device_profile_audit import kernel_patches
from torch_npu_smollm_stack_demo import errors


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    torch.set_num_threads(8);counts=Counter();runs=[];full_quality={};profile_negative_checked=False
    before=read(OUT/'before.json');fixture=ROOT/'npunlock/tests/fixtures/npu4000';manifest=read(fixture/'kv_select/manifest.json')
    kernel=(fixture/'kv_select/kv-select-f16.elf').read_bytes();profile=dict(vendor_id=0x8086,device_id=0x643e,compiler_version=(8,3))
    assert sha(fixture/'kv_select.c')==manifest['source_sha256']['kv_select.c']
    assert sha(fixture/'kv_select/kv-select-f16.elf')==manifest['elf_sha256']==sha(ROOT/'npunlock-probe/research/results/kv-group/select-kernel/select.elf')
    assert sha(fixture/'kv_select/kv-select-f16.s')==manifest['assembly_sha256'] and len(kernel_text(kernel))==manifest['text_bytes']==1984
    for name,digest in before.items():
        assert sha(OUT/'before'/name)==digest
        if name not in ('torch_npu_smollm.py','npunlock/python/npunlock/npu4000.py','torch_npu_device_profile_audit.py'):assert sha(ROOT/name)==digest
    def closed(report):
        assert report['status']=='passed';s=report['final_stats'];counts['npu_calls']+=s['npu_calls']
        assert s['closed'] and not s['context_failed']
        assert all(s[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
    for label in ('batch','full-fp16','full-w8a16'):
        directory=OUT/label;report=read(directory/'integration.json');closed(report)
        assert report['source_sha256']==sha(ROOT/'npunlock-probe/research/kv_select_integration.py')
        for name,digest in report['sources'].items():assert sha(ROOT/name)==digest
        assert sha(directory/'outputs.npz')==report['outputs_sha256']
        with np.load(directory/'outputs.npz',allow_pickle=False) as arrays:
            assert len(arrays.files)==report['output_array_count']
            for check in report['checks']:
                for i in range(check['bit_equal_arrays']):
                    a=arrays[f"{check['tag']}/baseline/{i}"];b=arrays[f"{check['tag']}/current/{i}"]
                    np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8));assert np.isfinite(b).all()
                    counts['default_path_array_comparisons']+=1
            if label=='batch':
                assert len(report['checks'])==176 and len(report['warm_checks'])==64 and len(report['graphs'])==32
                tags={c['tag'] for c in report['checks']}
                for mode in ('fp16','w8a16'):
                    for batch in range(1,9):
                        for prefix in (1,7,32,63):assert f'b{batch}-s{prefix}-{mode}/decode' in tags
            else:
                assert report['npu_decoder_layers']==30 and len(report['graphs'])==60
                assert len(report['checks'])==33 and len(report['cpu_quality'])==33 and len(report['warm_checks'])==1
                assert {r['tag'] for r in report['checks']}=={'prefill',*[f'chain-{i}' for i in range(33,65)]}
                for row in report['cpu_quality']:
                    tag=row['tag'];a=arrays[f'{tag}/baseline/logits'];b=arrays[f'{tag}/current/logits'];reference=arrays[f'{tag}/cpu/logits']
                    np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8));assert np.isfinite(b).all()
                    same(errors(torch.from_numpy(b),torch.from_numpy(reference)),row)
                    assert row['next_token_equal']==bool(b.argmax(-1)==reference.argmax(-1));counts['full_logit_comparisons']+=1
                full_quality[label]=dict(positions=33,max_relative_l2=max(r['relative_l2'] for r in report['cpu_quality']),
                    next_token_equal=sum(r['next_token_equal'] for r in report['cpu_quality']))
            assert all(r['host_copies_and_compiles']==0 and r['graph_calls_per_variant']==(1 if label=='batch' else 30) for r in report['warm_checks'])
        for graph in report['graphs']:
            stem=graph['stem'];path=directory/f'{stem}.blob';blob=path.read_bytes();assert sha(path)==graph['blob_sha256']
            assert sha(directory/f'{stem}.xml')==graph['xml_sha256'];reports=graph['patch_reports']
            kernel_patches(blob,reports,read(fixture/'rmsnorm/manifest.json'));counts['profile_patch_compatibility_checks']+=1
            if len(reports)==2 and not profile_negative_checked:
                bad=json.loads(json.dumps(reports));bad[1]['original_sha256']='broken-chain'
                damaged=bytearray(blob);damaged[reports[1]['code_offset']]^=1
                for image,records in ((blob,bad),(bytes(damaged),reports)):
                    try:kernel_patches(image,records,read(fixture/'rmsnorm/manifest.json'))
                    except AssertionError:counts['profile_patch_rejection_checks']+=1
                    else:raise RuntimeError('Malformed profiling patch evidence accepted')
                profile_negative_checked=True
            if '-current-' in stem:
                original=(directory/f"{stem.replace('-current-','-baseline-')}.blob").read_bytes()
                rebuilt,selection=patch_kv_select(original,kernel,**profile);selection['operation']='select_fp16_copy'
                assert rebuilt==blob and json.loads(json.dumps(selection))==reports[1]
                assert reports[0]['patched_sha256']==reports[1]['original_sha256'] and reports[1]['patched_sha256']==graph['blob_sha256']
                assert (directory/f"{stem.replace('-current-','-baseline-')}.xml").read_bytes()==(directory/f'{stem}.xml').read_bytes()
                counts['same_blob_select_replacements']+=1
            else:assert len(reports)==1 and reports[0]['patched_sha256']==graph['blob_sha256']
            counts['saved_decode_graphs']+=1
        runs.append(dict(label=label,checks=len(report['checks']),npu_calls=report['final_stats']['npu_calls']))

    quality=OUT/'calibrated-current';prepare=read(quality/'prepare.json');assert prepare['status']=='passed' and prepare['method']=='gptq'
    assert prepare['entry_runner_sha256']==sha(ROOT/'npunlock-probe/research/kv_select_checks.py')
    assert prepare['source_sha256']==sha(ROOT/'npunlock-probe/research/w8_calibration.py')
    for name,digest in prepare['sources'].items():assert sha(ROOT/name)==digest
    assert sha(quality/'weights.npz')==prepare['weights_sha256'] and sha(quality/'calibration-inputs.npz')==prepare['inputs_sha256']
    previous=ROOT/'npunlock-probe/research/results/rmsnorm-encode/calibrated-current'
    previous_prepare=read(previous/'prepare.json')
    for name in ('weights','calibration-inputs','holdout-outputs','decode-outputs'):
        old_digest=(previous_prepare['weights_sha256'] if name=='weights' else previous_prepare['inputs_sha256']
            if name=='calibration-inputs' else read(previous/f"{name.split('-')[0]}.json")['outputs_sha256'])
        assert sha(previous/f'{name}.npz')==old_digest
        with np.load(previous/f'{name}.npz',allow_pickle=False) as old,np.load(quality/f'{name}.npz',allow_pickle=False) as current:
            assert set(old.files)==set(current.files)
            for key in old.files:
                np.testing.assert_array_equal(old[key].view(np.uint8),current[key].view(np.uint8));counts['historical_quality_array_comparisons']+=1
    for case in ('holdout','decode'):
        report=read(quality/f'{case}.json');closed(report)
        assert report['entry_runner_sha256']==prepare['entry_runner_sha256'] and report['sources']==prepare['sources']
        script='w8_calibration_probe.py' if case=='holdout' else 'w8_calibration_decode.py'
        assert report['source_sha256']==sha(ROOT/'npunlock-probe/research'/script) and report['candidate_all_quality_gates_passed']
        if case=='holdout':assert report['criteria']==dict(logits_relative_l2=.03,next_token_kl=.02,local_execution_relative_l2=.003,local_max_scaled_error=.01)
        else:assert report['quality_gate_relative_l2']==.03 and report['quality_gate_kl']==.02
        assert sha(quality/f'{case}-outputs.npz')==report['outputs_sha256']
        with np.load(quality/f'{case}-outputs.npz',allow_pickle=False) as arrays:
            for prompt in report['prompts']:
                rows=[(prompt['variants'],f"{prompt['tag']}/")] if case=='holdout' else [(r['variants'],f"{prompt['tag']}/{r['step']}/") for r in prompt['steps']]
                for variants,prefix in rows:
                    reference=torch.from_numpy(arrays[prefix+'baseline-logits'])
                    for name,row in variants.items():
                        key=prefix+(name+'/logits' if case=='holdout' else name+'-logits')
                        logits_same(torch.from_numpy(arrays[key]),reference,row);counts['recomputed_quality_metrics']+=1
                    assert variants['calibrated']['quality_gate_passed'];counts['calibrated_quality_positions']+=1
    for module,cases in (('torch_npu_smollm_demo',('norm','numeric','boundary','guards','causality','hybrid')),
                         ('torch_npu_smollm_stack_demo',('numeric','boundary'))):
        for case in cases:
            report=read(OUT/'regressions'/module/f'{case}.json');closed(report)
            for name,digest in report['sources'].items():assert sha(ROOT/name)==digest
            counts['existing_model_checks']+=len(report['checks'])
    for name in ('guards','guards-optimized'):
        report=read(OUT/f'{name}.json');assert report['status']=='passed' and report['optimized_python']==(name=='guards-optimized')
        assert report['source_sha256']==sha(ROOT/'npunlock-probe/research/kv_select_guards.py')
        assert report['production_sha256']==sha(ROOT/'npunlock/python/npunlock/npu4000.py') and report['kernel_sha256']==manifest['elf_sha256']
        assert len(report['checks'])==35;counts['offline_select_checks']+=len(report['checks'])
    log=(ROOT/'npunlock-probe/research/kv-select-integration-offline.log').read_text(encoding='utf-8-sig')
    assert re.search(r'Ran 16 tests',log) and re.search(r'^OK$',log,re.M);counts['existing_offline_tests']=16
    snapshots=read(OUT/'snapshots.json')
    for name,digest in snapshots.items():assert sha(ROOT/name)==digest==sha(OUT/'sources'/name)
    result=dict(status='passed',default_enabled=True,counts=dict(counts),runs=runs,full_decoder_quality=full_quality,
        kernel_manifest=manifest,snapshot_files=len(snapshots),auditor_sha256=sha(Path(__file__)))
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

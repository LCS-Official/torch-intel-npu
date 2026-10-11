"""Offline verification of generation controls, frozen sources and regressions."""
import json
from pathlib import Path
import statistics as st
import numpy as np
from decode_chain_api import equal
from smollm_generation_profile import OUT
from w8_quality_probe import ROOT, sha


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    checkpoint=sha(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors')
    assert checkpoint==read(ROOT/'npunlock-probe/research/results/decode-chain-full/fp16-b1/models.json')['checkpoint_sha256']
    original=OUT/'before/torch_intel_npu.py'
    old=original.read_text(encoding='utf-8');current=(ROOT/'torch_intel_npu.py').read_text(encoding='utf-8')
    patched=old.replace('        for output, (shape, dtype) in zip(outputs, specs):',
        '        # Fresh allocations cannot alias live inputs or each other; explicit outs can.\n        for output, (shape, dtype) in zip(outputs, specs):')
    patched=patched.replace('_check_out(output, tuple(shape), dtype, inputs)',
        '_check_out(output, tuple(shape), dtype, inputs if outs is not None else ())')
    patched=patched.replace('if len({output.untyped_storage()._cdata for output in outputs}) != len(outputs):',
        'if outs is not None and len({output.untyped_storage()._cdata for output in outputs}) != len(outputs):')
    assert current==patched,'Unexpected production change'
    results=[];saved_pairs=0
    def check_state(report):
        state=report['final_stats'];assert state['closed'] and not state['context_failed']
        assert all(state[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
    for mode in ('fp16','w8a16'):
        baseline=None
        for candidate in (False,True):
            comparison='output preparation, both group4' if candidate else 'group1 vs default group4'
            directory=OUT/'preparation' if candidate else OUT
            report=read(directory/(mode+'.json'));assert report['status']=='passed'
            assert report['first_logits_vs_cpu']['within_existing_3_percent_gate']==(report['first_logits_vs_cpu']['relative_l2']<.03)
            check_state(report)
            assert report['input_tokens']==24 and report['new_tokens']==32 and report['decoder_layers']==30 and report['cpu_decoder_layers']==0
            assert report['rounds']==7 and len(report['samples'])==28 and len(report['pairs'])==7
            assert report['positive_rounds']==sum(p['4']<p['1'] for p in report['pairs'])
            assert sha(directory/(mode+'-outputs.npz'))==report['outputs_sha256']
            for name,digest in report['sources'].items():
                expected=sha(original) if name=='torch_intel_npu.py' and not candidate else sha(ROOT/name)
                assert digest==expected,(name,mode,candidate)
            expected_script=(ROOT/'npunlock-probe/research/smollm_generation_profile.py' if candidate else
                OUT/'before/npunlock-probe/research/smollm_generation_profile.py')
            assert report['source_sha256']==sha(expected_script)
            if candidate:
                assert report['comparison']==comparison
                assert report['baseline_source_sha256']==sha(original)
                assert report['wrapper_sha256']==sha(ROOT/'npunlock-probe/research/smollm_preparation_probe.py')
            for row in report['samples']:
                assert row['all_32_logits_and_tokens_bit_equal'] and len(row['steps'])==32
                group=1 if row['group']==1 else 4
                change=row['delta'];assert change['npu_calls']==960 and change['session_creations']==0
                assert change['chain_graphs']==(930 if candidate or group==4 else 0)
                assert change['chain_groups']==(248 if candidate or group==4 else 0)
                assert change['host_upload_copies']==change['host_download_copies']==32
                assert row['first_token_ms']==sum(row['steps'][0].values())
                assert row['decode_ms']==sum(sum(step.values()) for step in row['steps'][1:])
            for group in (1,4):
                samples=[r for r in report['samples'] if (1 if r['group']==1 else 4)==group]
                summary=report['summary'][str(group)]
                assert summary['elapsed_ms']==st.median(r['elapsed_ms'] for r in samples)
                assert summary['overall_tokens_per_second']==32000/summary['elapsed_ms']
                assert summary['decode_tokens_per_second']==31000/st.median(r['decode_ms'] for r in samples)
                assert len(report['last_position_head']['samples_ms'][str(group==4)])==42
            with np.load(directory/(mode+'-outputs.npz'),allow_pickle=False) as saved:
                values={key:saved[key].copy() for key in saved.files}
            assert values['tokens'].shape==(1,56)
            if baseline is None:baseline=values
            else:
                assert set(baseline)==set(values)
                equal(list(baseline.values()),[values[key] for key in baseline]);saved_pairs+=len(values)
            a=report['summary']['1'];b=report['summary']['4']
            results.append(dict(mode=mode,comparison=comparison,old_ms=a['elapsed_ms'],new_ms=b['elapsed_ms'],
                old_tokens_per_second=a['overall_tokens_per_second'],new_tokens_per_second=b['overall_tokens_per_second'],
                latency_reduction_percent=100*(1-b['elapsed_ms']/a['elapsed_ms']),positive_rounds=report['positive_rounds'],
                first_logits_cpu_relative_l2=report['first_logits_vs_cpu']['relative_l2'],
                cpu_quality_gate_passed=report['first_logits_vs_cpu']['within_existing_3_percent_gate']))
    checks={}
    for filename,count in (('bindings.json',17),('checks.json',20),('defaults.json',8),('model-guards.json',21),('device.json',33),('training.json',11)):
        record=read(OUT/'regressions'/filename)
        assert record['status'] in ('passed','passed_with_limitations'),filename
        assert len(record['checks'])==count,filename
        check_state(record);checks[filename]=dict(count=len(record['checks']),status=record['status'])
        if record.get('limitations'):checks[filename]['limitations']=record['limitations']
        for name,digest in record.get('sources',{}).items():assert sha(ROOT/name)==digest,name
    host=read(OUT/'host.json');assert host['status']=='passed';check_state(host)
    assert host['sources']['torch_intel_npu.py']==sha(original)
    assert host['source_sha256']==sha(ROOT/'npunlock-probe/research/smollm_generation_host.py')
    # Separate tokenwise normalization from the changed CPU GEMM shape.
    import torch
    from safetensors import safe_open
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    torch.set_num_threads(8);head_diagnostic={}
    norm=LlamaRMSNorm(576,eps=1e-5).half().eval()
    with torch.no_grad(),safe_open(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors',framework='pt',device='cpu') as file:
        norm.weight.copy_(file.get_tensor('model.norm.weight').half())
        weight=file.get_tensor('model.embed_tokens.weight').half()
        for mode in ('fp16','w8a16'):
            with np.load(OUT/(mode+'-outputs.npz'),allow_pickle=False) as saved:
                hidden=torch.from_numpy(saved['prefill-hidden'].copy());anchor=torch.from_numpy(saved['logits-0'].copy())
            normalized=norm(hidden);last_normalized=norm(hidden[:,-1:])
            assert torch.equal(normalized[:,-1:].view(torch.int16),last_normalized.view(torch.int16))
            full=torch.nn.functional.linear(normalized,weight)[:,-1].float()
            last=torch.nn.functional.linear(last_normalized,weight)[:,0].float()
            assert torch.equal(full.view(torch.int32),anchor.view(torch.int32))
            head_diagnostic[mode]=dict(norm_bit_equal=True,head_bit_equal=bool(torch.equal(full.view(torch.int32),last.view(torch.int32))),
                differing_logits=int((full!=last).sum()),max_abs=float((full-last).abs().max()),
                argmax_equal=bool(torch.equal(full.argmax(-1),last.argmax(-1))))
    result=dict(status='passed',production_change='Skip redundant alias checks for fresh outputs only',
        checkpoint_sha256=checkpoint,
        last_position_head_diagnostic=head_diagnostic,
        native_and_model_kernels_unchanged=True,saved_baseline_candidate_array_pairs=saved_pairs,
        generation_comparisons=results,regressions=checks,sources_sha256={name:sha(ROOT/name) for name in
        ('torch_intel_npu.py','npunlock-probe/research/smollm_generation_profile.py',
         'npunlock-probe/research/smollm_preparation_probe.py','npunlock-probe/research/smollm_generation_host.py',
         'npunlock-probe/research/smollm_generation_regressions.py','npunlock-probe/research/smollm_generation_audit.py')})
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

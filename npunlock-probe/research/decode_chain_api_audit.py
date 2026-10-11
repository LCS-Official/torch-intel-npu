"""Offline provenance and result audit for the production chain integration."""
import ast
import json
from pathlib import Path
import statistics as st
import numpy as np
from decode_chain_api import ROOT, OUT, sha, write, equal


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def healthy(record):
    final=record['final_stats']
    assert final['closed'] and not final['context_failed']
    assert all(final[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_blocks','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))


def function(path, cls, name):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return ast.dump(next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)


def main():
    before=OUT/'before';saved=read(before/'sources.json')
    for name,digest in saved.items():assert sha(before/name)==digest
    records={name:read(OUT/(name+'.json')) for name in ('checks','checks-profile','models','device','training','model-guards')}
    for name,record in records.items():
        assert record['status'] in ('passed','passed_with_limitations'),(name,record['status'])
        healthy(record)
        for source,digest in record.get('sources',{}).items():assert sha(ROOT/source)==digest,source
    assert len(records['checks']['checks'])==len(records['checks-profile']['checks'])==20
    assert len(records['device']['checks'])==33 and len(records['training']['checks'])==11
    for row in records['checks-profile']['checks']:
        if 'warm_group' in row:
            assert row['stage_counts']['append_graph']==7
            assert all(row['stage_counts'][k]==(7+row['warm_group']-1)//row['warm_group'] for k in ('submit','wait','close_list','reset_list','reset_fence'))
    models=records['models'];assert len(models['cases'])==7
    assert sha(OUT/'models-outputs.npz')==models['outputs_sha256']
    comparisons=0;summary=[]
    with np.load(OUT/'models-outputs.npz',allow_pickle=False) as outputs:
        for case in models['cases']:
            tag=case['tag'];layers=30 if tag.endswith('l30') else 7
            assert case['cold_bit_equal_arrays']==layers*3+1 and case['continuous_decode_steps']==4
            if layers==7:assert case['kv_boundary']=='actual prefill63 -> decode64 -> reject65'
            assert len(case['checks'])==3
            for group in case['checks']:assert group['bit_equal_arrays']==layers*3+1
            timing=case['timing'];rounds=21 if layers==30 else 15;assert len(timing['samples'])==4*rounds
            for group in (1,4):
                values=[r['total_ms'] for r in timing['samples'] if r['group']==group]
                assert len(values)==rounds*2 and st.median(values)==timing['median_total_ms'][str(group)]
            change=timing['warm_delta']
            assert change['npu_calls']==rounds*4*layers and change['chain_graphs']==rounds*2*layers
            assert change['chain_groups']==rounds*2*((layers+3)//4)
            assert all(v==0 for k,v in change.items() if k not in ('npu_calls','chain_groups','chain_graphs'))
            assert timing['positive_rounds']==sum(p['4']<p['1'] for p in timing['pairs'])
            for i in range(layers*3+1):assert outputs[f'{tag}/{i}'].size>0;comparisons+=1
            a=timing['median_total_ms']['1'];b=timing['median_total_ms']['4']
            summary.append(dict(tag=tag,original_ms=a,group4_ms=b,reduction_percent=100*(1-b/a),wins=timing['positive_rounds'],rounds=rounds))
        with np.load(ROOT/'npunlock-probe/research/results/decode-chain/plain-outputs.npz',allow_pickle=False) as old:
            equal([outputs[f'fp16-b1-l30/{i}'] for i in range(91)],[old[f'baseline/{i}'] for i in range(91)])
    for cls,names in [('FrozenSmolStack',('forward','prefill')),('FrozenSmolLayer',('forward','prefill','graph','_validate_input','_validate_cache'))]:
        for name in names:assert function(before/'torch_npu_smollm.py',cls,name)==function(ROOT/'torch_npu_smollm.py',cls,name)
    original=(before/'npunlock/src/graphinfer/session_win32.c').read_text(encoding='utf-8')
    original=original.replace('  npunlock_ze_command_list_close_fn command_list_close;',
        '  npunlock_ze_command_list_close_fn command_list_close;\n  npunlock_ze_command_list_append_barrier_fn command_list_append_barrier;')
    assert (ROOT/'npunlock/src/graphinfer/session_win32.c').read_text(encoding='utf-8').startswith(original)
    ext=read(ROOT/'npunlock-probe/torch-device/build.json');native=read(ROOT/'npunlock-probe/session-runtime/build.json')
    assert sha(ROOT/'npunlock-probe/torch-device/_torch_intel_npu.pyd')==ext['extension_sha256']
    assert sha(ROOT/'npunlock-probe/torch-device/backend.cpp')==ext['source_sha256']
    assert sha(ROOT/'npunlock-probe/session-runtime/npunlock_session.dll')==ext['session_sha256']==native['dllSha256'].lower()
    for item in native['sources']:assert sha(ROOT/'npunlock'/item['path'])==item['sha256'].lower()
    for item in native['headers']:assert sha(ROOT/'npunlock'/item['path'])==item['sha256'].lower()
    previous=read(ROOT/'npunlock-probe/research/results/decode-chain/before/sources.json')['sources']
    for name in ('npunlock/tests/fixtures/npu4000/rmsnorm.c','npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf',
                 'npunlock/tests/fixtures/npu4000/kv_select/kv-select-f16.elf'):
        assert sha(ROOT/name)==previous[name]
    task=read(OUT/'task-profile/b1-s32-fp16.json');healthy(task);assert task['status']=='passed'
    for source,digest in task['production_sha256'].items():assert sha(ROOT/source)==digest
    assert sha(OUT/'task-profile/session_task_profile.dll')==task['profiling_dll_sha256']
    native_log=(OUT/'native-build.log').read_text(encoding='utf-8-sig')
    assert '11 rejections, 9 failures and safe recovery passed' in native_log
    result=dict(status='passed',before_snapshot_files=len(saved),model_cases=7,saved_model_arrays=comparisons,
                archived_output_bit_equal_arrays=91,native_chain_cases=23,production_checks=20,instrumented_checks=20,
                device_checks=33,training_checks=11,model_guard_checks=len(records['model-guards']['checks']),
                all_final_states_healthy=True,unchanged_compute_kernels=True,unchanged_prefill_and_forward=True,
                timings=summary,training_status=records['training']['status'],training_limitations=records['training'].get('limitations',[]),
                tool_sources={name:sha(ROOT/name) for name in ('torch_npu_warm_profile.py','torch_npu_device_profile.py',
                    'npunlock-probe/research/shave_fp32_pipeline.py','npunlock-probe/research/decode_chain_api.py',
                    'npunlock-probe/research/decode_chain_regressions.py','npunlock-probe/research/decode_chain_api_audit.py')})
    write('audit.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

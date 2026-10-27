"""Audit mixed-projection artifacts before their storage-policy cleanup."""
import importlib.util
import json
from pathlib import Path
import sys

import torch

import w8_projection_mixed as experiment
from torch_npu_smollm import FrozenSmolLayer


def main():
    torch.set_num_threads(8)
    root, out = experiment.common.ROOT, experiment.OUT
    prepare, selection, independent = (experiment.read(name) for name in ('prepare','selection','independent'))
    assert prepare['sources']==selection['sources']==independent['sources']==experiment.sources()
    assert all(record['status']=='passed' and record['npu_calls']==0 for record in (prepare,selection,independent))
    assert independent['selection_sha256']==experiment.common.sha(out/'selection.json')
    assert len(prepare['matrices'])==210 and len(prepare['trajectories'])==40
    assert all(row['calibrated_sse']<=row['rtn_sse']+1e-15 for row in prepare['matrices'])
    fit = [row for row in prepare['trajectories'] if row['index']%5!=4]
    internal = [row for row in prepare['trajectories'] if row['index']%5==4]
    assert len(fit)==32 and len(internal)==8 and sum(len(row['token_ids'][0]) for row in fit)==2048
    assert set((row['system'],row['prompt']) for row in prepare['trajectories']).isdisjoint(
        (system,prompt) for _,system,prompt in experiment.HOLDOUT+experiment.FRESH)
    def check(record, count):
        rows = record['positions']
        assert len(rows)==count and record['summary']==experiment.common.summary(rows)
        assert all(row['quality_gate_passed']==(row['relative_l2']<.03 and row['next_token_kl']<.02) for row in rows)
    for record in selection['profiles']:
        check(record,290)
        assert record['fp16_elements']<=selection['projection_total_elements']//2
    best = min(selection['profiles'],key=lambda row:(*experiment.common.ranking(row['summary']),row['fp16_elements']))
    assert best['protected']==selection['selected'] and best['summary']==selection['selected_summary']
    for record in independent['variants'].values():
        check(record,142)
    expected = [(row['tag'],p) for row in independent['cases'] for p in range(row['offset'],len(row['token_ids'][0]))]
    assert len(independent['cases'])==17 and len(expected)==142
    tokens = sum(len(row['token_ids'][0]) for row in independent['cases'])
    assert tokens==592
    totals = {}
    for label in ('w8','mixed','fp16'):
        record = experiment.read('npu-'+label)
        assert record['status']=='passed' and record['adapter_sha256']==experiment.common.sha(root/'torch_npu_smollm.py')
        assert record['sources']==experiment.sources() and record['source_sha256']==experiment.common.sha(Path(experiment.__file__))
        assert record['weights_sha256']==prepare['weights_sha256'] and record['selection_sha256']==independent['selection_sha256']
        check(record,142)
        assert [(row['tag'],row['position']) for row in record['positions']]==expected
        assert len(record['graphs'])==30 and len(record['warm'])==17
        assert all(row['calls']==30 and row['repeated_arrays']==91 and row['seed_arrays']==90
                   and row['payload_copies_and_compiles']==0 for row in record['warm'])
        final = record['final_stats']
        assert final['npu_calls']==30*(tokens+17)+60==18330 and final['closed'] and not final['context_failed']
        assert all(final[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
        for graph in record['graphs']:
            integer = [m for m in graph['evidence']['modes'] if not m['elementwise'] and m['wmode'] in (1,2)]
            assert bool(integer)==(label!='fp16')
            assert all(m['amode']==0 and m['wmode']==2 for m in integer)
        totals[label] = dict(**record['summary'],constant_bytes=sum(g['evidence']['constant_bytes'] for g in record['graphs']),
                             int8_matrix_tasks=sum(g['evidence']['int8_matrix_tasks'] for g in record['graphs']))
    checks = experiment.read('checks')
    assert checks['status']=='passed' and len(checks['guards'])==4 and len(checks['local_layers'])==30
    assert checks['adapter_sha256']==experiment.common.sha(root/'torch_npu_smollm.py') and checks['ranged_checkpoint_policy_passed']
    assert all(row['relative_l2']<.003 and row['max_scaled_error']<.01 for row in checks['local_layers'])
    spec = importlib.util.spec_from_file_location('_published_smollm',root/'source-release/torch_npu_smollm.py')
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    assert experiment.common.sha(Path(original.__file__))=='a9ce3949141ebd3fae258a73578d9f13ed7bd0da4e0d3809a97ca0c46296c869'
    packed = experiment.weights()[0]
    x = (torch.randn(1,7,576,generator=torch.Generator().manual_seed(934))*.1).half()
    golden = 0
    with torch.no_grad():
        for mode in ('fp16','w8a16'):
            kwargs = dict(layer_index=0,mode=mode,packed_weights=packed if mode=='w8a16' else None)
            old, new = (cls.from_checkpoint(experiment.common.MODEL,**kwargs) for cls in (original.FrozenSmolLayer,FrozenSmolLayer))
            assert torch.equal(old.reference_cpu(x)[0],new.reference_cpu(x)[0])
            for kind,shape in (('prefill',(1,7,576)),('decode',(1,1,576))):
                options = dict(segment='layer',fused_norm=True,kv_cache=kind)
                assert old.graph(shape,**options)[1:]==new.graph(shape,**options)[1:]
                golden += 1
    summary = dict(status='passed',npu=totals,local_max_l2=max(row['relative_l2'] for row in checks['local_layers']),
                   local_max_scaled=max(row['max_scaled_error'] for row in checks['local_layers']),
                   default_graphs_unchanged=golden,total_npu_calls=54990+checks['final_stats']['npu_calls'],
                   projection_sse_ratio=sum(row['calibrated_sse'] for row in prepare['matrices'])/sum(row['rtn_sse'] for row in prepare['matrices']),
                   hashes={name:experiment.common.sha(root/name) for name in ('torch_npu_smollm.py','torch_npu_lowbit.py',
                     'torch_intel_npu.py','npunlock-probe/torch-device/backend.cpp','npunlock-probe/session-runtime/npunlock_session.dll',
                     'npunlock-probe/torch-device/_torch_intel_npu.pyd')})
    experiment.write('audit',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

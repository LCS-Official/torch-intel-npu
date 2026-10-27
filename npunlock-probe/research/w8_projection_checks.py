"""Real-NPU local arithmetic checks for the selected mixed projection policy."""
import gc
import json
from pathlib import Path
import traceback

import torch

import w8_projection_mixed as experiment
from torch_npu_smollm import FrozenSmolLayer, FrozenSmolStack
from torch_npu_smollm_stack_demo import errors


def main():
    import torch_intel_npu as npu
    torch.set_num_threads(8)
    selection = experiment.read('selection')
    policy = tuple(selection['selected'])
    report = dict(status='running',source_sha256=experiment.common.sha(Path(__file__)),
                  adapter_sha256=experiment.common.sha(experiment.common.ROOT/'torch_npu_smollm.py'),
                  protected=list(policy),selection_sha256=experiment.common.sha(experiment.OUT/'selection.json'),
                  local_layers=[],guards=[])
    def work():
        packed = experiment.weights()
        # Contract errors must be rejected before native allocation/submission.
        for bad in (['mlp.up_proj.weight'],('unknown',),policy+policy,tuple(experiment.NAMES)):
            before = npu.stats()
            try:
                FrozenSmolStack.from_checkpoint(experiment.common.MODEL,num_layers=30,fp16_projections=bad)
            except ValueError:
                after = npu.stats()
                assert all(after[k]==before[k] for k in ('npu_calls','session_creations','driver_allocations','host_upload_copies'))
                report['guards'].append(repr(bad))
            else:
                raise AssertionError('Missing policy rejection')
        stack = FrozenSmolStack.from_checkpoint(experiment.common.MODEL,num_layers=30,packed_weights=packed,fp16_projections=policy)
        assert stack.fp16_projections==frozenset(policy)
        assert all(layer.fp16_projections==stack.fp16_projections for layer in stack.layers)
        ranged = FrozenSmolStack.from_checkpoint(experiment.common.MODEL,start_layer=3,num_layers=2,
                    packed_weights={i:packed[i] for i in (3,4)},fp16_projections=policy)
        assert ranged.start_layer==3 and all(layer.fp16_projections==frozenset(policy) for layer in ranged.layers)
        report['ranged_checkpoint_policy_passed']=True
        cpu,_ = experiment.common.model()
        case = next(row for row in experiment.read('independent')['cases'] if row['tag']=='fox')
        ids = torch.tensor(case['token_ids'])
        value = cpu.model.embed_tokens(ids).to('intel_npu')
        for i,layer in enumerate(stack.layers):
            reference,_ = layer.reference_cpu(value.cpu())
            value,cache = layer.prefill(value)
            error = errors(value.cpu(),reference)
            assert error['relative_l2']<.003 and error['max_scaled_error']<.01,(i,error)
            evidence = next(iter(layer._evidence.values()))
            assert evidence['int8_matrix_tasks']>0
            report['local_layers'].append(dict(layer=i,**error,constant_bytes=evidence['constant_bytes'],
                                              int8_matrix_tasks=evidence['int8_matrix_tasks']))
        print('PASS: 30 local mixed prefills; max L2',max(row['relative_l2'] for row in report['local_layers']),flush=True)
    try:
        with torch.no_grad():
            work()
        report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    gc.collect()
    npu.shutdown()
    final = npu.stats()
    report['final_stats']=final
    assert final['closed'] and not final['context_failed']
    assert all(final[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
    experiment.write('checks',report)
    print(report['status'],'NPU calls',final['npu_calls'],flush=True)
    return int(report['status']!='passed')


if __name__=='__main__':
    raise SystemExit(main())

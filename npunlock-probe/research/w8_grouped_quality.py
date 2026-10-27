"""Research-only grouped W8 full decoder; production contracts stay unchanged.

The existing graph builder supports grouped weights. This local subclass uses
it without relaxing FrozenSmolLayer's public per-row-only constructor contract.
Grouped code*FP16 scale is exact in FP32; the staged CPU path uses FP32 GEMM.
Compiler DPU modes must be inspected before claiming compressed W8 execution.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

import w8_sequential_calibration as study
from torch_npu_lowbit import quantize_weight, compile_weight
from torch_npu_smollm import FrozenSmolLayer
from w8_calibration_probe import metric


class GroupedLayer(FrozenSmolLayer):
    def __init__(self, config, weights, *, group_size=32, mode='w8a16', packed_weights=None):
        assert mode == 'w8a16' and packed_weights is None and group_size in (4, 8, 16, 32, 64)
        super().__init__(config, weights, mode=mode)
        self._reference_weights = {}
        for name, value in weights.items():
            if value.ndim == 2:
                self._packed[name] = quantize_weight(value.float().half(), format='int8', group_size=group_size)
                q = self._packed[name]
                codes = np.frombuffer(q.data, dtype=np.int8).reshape(q.grouped_shape).astype(np.float32)
                scales = np.frombuffer(q.scales, dtype=np.float16).reshape(*q.grouped_shape[:2], 1).astype(np.float32)
                self._reference_weights[name] = torch.from_numpy((codes * scales).reshape(q.shape))

    def _cpu_weight(self, name):
        if name in self._reference_weights:
            return self._reference_weights[name]
        return super()._cpu_weight(name)

    def reference_cpu(self, x, **options):
        assert not options
        return super().reference_cpu(x, quantized=False)


def layers(group_size):
    from safetensors import safe_open
    config = json.loads((study.MODEL / 'config.json').read_text(encoding='utf-8'))
    result = []
    with safe_open(study.MODEL / 'model.safetensors', framework='pt', device='cpu') as file:
        for i in range(30):
            prefix = f'model.layers.{i}.'
            weights = {name[len(prefix):]: file.get_tensor(name) for name in file.keys() if name.startswith(prefix)}
            result.append(GroupedLayer(config, weights, group_size=group_size))
    return result


def cpu():
    metadata = study.read('fit')
    model, _ = study.model()
    cases = [row for row in metadata['trajectories'] if row['index'] % 5 == 4]
    report = dict(status='running', source_sha256=study.sha(Path(__file__)), sources=study.fingerprints(),
                  checkpoint_sha256=metadata['checkpoint_sha256'], criteria=metadata['criteria'], candidates={})
    for size in (64, 32, 16, 8, 4):
        stack = layers(size)
        rows = []
        for case in cases:
            ids = torch.tensor(case['token_ids'])
            offset = case['prefix'] - 1
            baseline = model(ids, use_cache=False).logits[:, offset:].float()
            hidden = model.model.embed_tokens(ids)
            for layer in stack:
                hidden, _ = layer.reference_cpu(hidden)
            logits = model.lm_head(model.model.norm(hidden))[:, offset:].float()
            for j in range(logits.shape[1]):
                assert torch.isfinite(logits[:, j]).all()
                rows.append(dict(tag=str(case['index']), position=offset+j, **metric(logits[:, j], baseline[:, j])))
        report['candidates'][str(size)] = dict(summary=study.summary(rows), positions=rows,
                                              packed_bytes=sum(len(q.data)+len(q.scales) for layer in stack for q in layer._packed.values()))
        print('group', size, report['candidates'][str(size)]['summary'], flush=True)
        del stack
        gc.collect()
    # Larger groups have less scale overhead; only prefer smaller groups when quality improves.
    report['selected_group_size'] = int(min(report['candidates'], key=lambda size: (*study.ranking(report['candidates'][size]['summary']), -int(size))))
    assert 'torch_intel_npu' not in __import__('sys').modules
    report.update(status='passed', npu_calls=0)
    study.write('group-selection', report)


def probe():
    import torch_intel_npu as npu
    report = dict(status='running', source_sha256=study.sha(Path(__file__)), cases=[])
    def work():
        from safetensors import safe_open
        with safe_open(study.MODEL / 'model.safetensors', framework='pt', device='cpu') as checkpoint:
            for name in ('self_attn.q_proj.weight', 'mlp.down_proj.weight'):
                weight = checkpoint.get_tensor('model.layers.0.' + name).float().half()
                for size in (32, 16, 8, 4):
                    q = quantize_weight(weight, format='int8', group_size=size)
                    program, evidence = compile_weight(q, 1, npu._native)
                    # Direct native inference against a CPU FP64 decode, independent of the model wrapper.
                    generator = torch.Generator().manual_seed(947)
                    x = torch.randn(1, q.shape[1], generator=generator).half()
                    c = np.frombuffer(q.data, dtype=np.int8).reshape(q.grouped_shape).astype(np.float64)
                    s = np.frombuffer(q.scales, dtype=np.float16).reshape(*q.grouped_shape[:2], 1).astype(np.float64)
                    target = (x.double() @ torch.from_numpy((c*s).reshape(q.shape)).t()).half()
                    key = ('group_quality_probe', name, size)
                    session = npu._native.create_inference_session(program.graph_blob, timeout_ms=10000, parent=npu._root_session)
                    npu._sessions[key] = session, program
                    npu._session_creations += 1
                    device = x.to('intel_npu')
                    actual = npu._run(key, [device], target.shape, torch.float16).cpu()
                    error = study.errors(actual, target)
                    assert error['relative_l2'] < .003 and error['max_scaled_error'] < .01, error
                    report['cases'].append(dict(name=name, group_size=size, error=error, evidence=evidence))
                    print('group DPU probe', name, size, error, flush=True)
    try:
        work()
        gc.collect()
        npu.shutdown()
        report['final_stats'] = npu.stats()
        assert report['final_stats']['closed'] and not report['final_stats']['context_failed']
        assert all(report['final_stats'][key] == 0 for key in ('live_buffer_bytes', 'cached_buffer_bytes', 'quarantined_buffer_bytes'))
        report['status'] = 'passed'
    except Exception:
        report.update(status='failed', traceback=traceback.format_exc(), failure_stats=npu.stats())
        print(report['traceback'], flush=True)
    study.write('group-probe', report)
    return int(report['status'] != 'passed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('cpu', 'probe'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    with torch.no_grad():
        if args.case == 'cpu':
            cpu()
            return 0
        return probe()


if __name__ == '__main__':
    raise SystemExit(main())

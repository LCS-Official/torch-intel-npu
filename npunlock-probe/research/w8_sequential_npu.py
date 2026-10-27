"""Real-NPU checks on the fixed independent cases from sequential calibration.

One variant per process bounds sessions/memory. No graph disk cache, raw logits
or graph blobs are saved. Embedding/final norm/head remain explicitly on CPU.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch

import w8_sequential_calibration as study
from torch_npu_smollm import FrozenSmolStack
from w8_calibration_probe import metric


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run(label):
    import torch_intel_npu as npu
    started = time.perf_counter()
    independent = study.read('independent-cpu')
    metadata = study.read('fit')
    assert independent['sources'] == metadata['sources'] == study.fingerprints()
    assert independent['weight_sha256'] == metadata['weight_sha256']
    report = dict(status='running', label=label, source_sha256=study.sha(Path(__file__)),
                  sources=metadata['sources'], weight_sha256=metadata['weight_sha256'],
                  independent_sha256=study.sha(study.OUT / 'independent-cpu.json'),
                  criteria=metadata['criteria'], initial_stats=npu.stats(), positions=[], warm=[], graphs=[],
                  policy='All 30 decoder layers on NPU; incremental prompt then fixed CPU teacher tokens. '
                         'CPU embedding/final norm/head; compare with official full-prefix CPU FP16 logits.')
    if label == 'grouped':
        import w8_grouped_quality as grouped
        selection = study.read('group-selection')
        assert selection['source_sha256'] == study.sha(Path(grouped.__file__))
        report.update(group_size=selection['selected_group_size'], group_selection_sha256=study.sha(study.OUT / 'group-selection.json'))
    def work():
        cpu, _ = study.model()
        stack = FrozenSmolStack.from_checkpoint(
            study.MODEL, num_layers=30, mode='fp16' if label == 'fp16' else 'w8a16',
            packed_weights=None if label in ('fp16', 'grouped') else study.load_weights(label))
        if label == 'grouped':
            stack.layers = torch.nn.ModuleList(grouped.layers(report['group_size']))
        assert len(stack.layers) == 30
        for case in independent['cases']:
            ids = torch.tensor(case['token_ids'])
            offset = case['offset']
            host = cpu.model.embed_tokens(ids)
            baseline = cpu(ids, use_cache=False).logits[:, offset:].float()
            states = None
            for position in range(ids.shape[1]):
                x = host[:, position:position+1].contiguous().to('intel_npu')
                previous = states
                output, states = stack.prefill(x) if states is None else stack.decode(x, states)
                assert all(cache.length == position + 1 for cache in states)
                if position == 0:
                    seed = states
                    seed_hashes = [digest(value) for cache in seed for value in (cache.keys, cache.values, cache._position)]
                if position >= offset:
                    actual = cpu.lm_head(cpu.model.norm(output.cpu()))[:, -1].float()
                    assert torch.isfinite(actual).all()
                    report['positions'].append(dict(tag=case['tag'], position=position,
                                                    **metric(actual, baseline[:, position-offset])))
            before = npu.stats()
            again, again_states = stack.decode(x, previous)
            after = npu.stats()
            assert after['npu_calls'] - before['npu_calls'] == 30
            assert all(after[key] == before[key] for key in ('session_creations', 'host_upload_copies', 'host_download_copies'))
            first = [output] + [value for cache in states for value in (cache.keys, cache.values, cache._position)]
            second = [again] + [value for cache in again_states for value in (cache.keys, cache.values, cache._position)]
            assert [digest(value) for value in first] == [digest(value) for value in second]
            assert seed_hashes == [digest(value) for cache in seed for value in (cache.keys, cache.values, cache._position)]
            report['warm'].append(dict(tag=case['tag'], calls=30, bit_equal_arrays=len(first),
                                       payload_copies_and_compiles=0, immutable_seed_arrays=len(seed_hashes)))
            rows = [row for row in report['positions'] if row['tag'] == case['tag']]
            print('NPU', label, case['tag'], study.summary(rows), flush=True)
        for i, layer in enumerate(stack.layers):
            evidence = [value for key, value in layer._evidence.items() if key[-1] == 'decode']
            assert len(evidence) == 1
            modes = [mode for mode in evidence[0]['modes'] if not mode['elementwise']]
            assert modes
            if label == 'fp16':
                assert all(mode['wmode'] not in (1, 2) for mode in modes)
            elif label != 'grouped':
                # Non-elementwise FP16 tasks can also perform layout changes.
                integer = [mode for mode in modes if mode['wmode'] in (1, 2)]
                assert integer and all(mode['amode'] == 0 and mode['wmode'] == 2 for mode in integer)
            else:
                assert all(mode['amode'] == 0 and mode['wmode'] in (0, 2) for mode in modes)
            report['graphs'].append(dict(layer=i, evidence=evidence[0]))
        report['summary'] = study.summary(report['positions'])
    try:
        work()
        gc.collect()
        npu.shutdown()
        final = npu.stats()
        report['final_stats'] = final
        assert final['closed'] and not final['context_failed']
        assert all(final[key] == 0 for key in ('live_buffers', 'live_buffer_bytes', 'cached_buffer_bytes', 'quarantined_buffer_bytes'))
        report.update(status='passed', elapsed_seconds=time.perf_counter()-started,
                      all_quality_gates_passed=report['summary']['quality_passed'] == report['summary']['positions'])
    except Exception:
        report.update(status='failed', traceback=traceback.format_exc(), failure_stats=npu.stats())
        print(report['traceback'], flush=True)
    if report['status'] == 'failed':
        gc.collect()
        npu.shutdown()
        report['final_stats'] = npu.stats()
    study.write('npu-' + label, report)
    print(json.dumps({key: report.get(key) for key in ('status', 'summary', 'elapsed_seconds', 'final_stats')}, indent=2), flush=True)
    return int(report['status'] != 'passed')


def cpu_grouped():
    import w8_grouped_quality as grouped
    independent = study.read('independent-cpu')
    selection = study.read('group-selection')
    assert selection['source_sha256'] == study.sha(Path(grouped.__file__))
    size = selection['selected_group_size']
    cpu, _ = study.model()
    stack = grouped.layers(size)
    rows = []
    for case in independent['cases']:
        ids = torch.tensor(case['token_ids'])
        offset = case['offset']
        baseline = cpu(ids, use_cache=False).logits[:, offset:].float()
        hidden = cpu.model.embed_tokens(ids)
        for layer in stack:
            hidden, _ = layer.reference_cpu(hidden)
        logits = cpu.lm_head(cpu.model.norm(hidden))[:, offset:].float()
        for j in range(logits.shape[1]):
            assert torch.isfinite(logits[:, j]).all()
            rows.append(dict(tag=case['tag'], position=offset+j, **metric(logits[:, j], baseline[:, j])))
        print('CPU grouped', size, case['tag'], study.summary([row for row in rows if row['tag'] == case['tag']]), flush=True)
    result = dict(status='passed', group_size=size, source_sha256=study.sha(Path(__file__)),
                  group_source_sha256=study.sha(Path(grouped.__file__)), group_selection_sha256=study.sha(study.OUT / 'group-selection.json'),
                  independent_sha256=study.sha(study.OUT / 'independent-cpu.json'), sources=study.fingerprints(),
                  positions=rows, summary=study.summary(rows), npu_calls=0)
    assert 'torch_intel_npu' not in __import__('sys').modules
    study.write('independent-grouped-cpu', result)
    print(json.dumps(result['summary'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('fp16', 'simultaneous', 'sequential', 'grouped', 'grouped-cpu'), required=True)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    assert not os.environ.get('NPUNLOCK_GRAPH_CACHE'), 'This experiment must not create a disk graph cache'
    torch.set_num_threads(8)
    with torch.no_grad():
        if args.variant == 'grouped-cpu':
            cpu_grouped()
            return 0
        return run(args.variant)


if __name__ == '__main__':
    raise SystemExit(main())

"""Aggregate per-graph task interval unions and audit saved local evidence."""
from collections import defaultdict
import json
from pathlib import Path
import statistics as st

from w8_quality_probe import ROOT, sha
from torch_npu_device_profile import union_ns

OUT = ROOT / 'npunlock-probe/research/results/decoder-native'


def summarize():
    host = json.loads((OUT / 'host.json').read_text())
    assert host['status'] == 'passed'
    assert sha(OUT / 'host-outputs.npz') == host['outputs_sha256']
    report = dict(host=[], modes={}, files={}, current_sha256={})
    for row in host['rows']:
        samples = row['host_samples']
        assert len(samples) == 62 and row['warm_delta']['driver_allocations'] == 0
        total = row['median_ms']['baseline']
        measured = st.median(sum(s['times_ms'].values()) for s in samples)
        report['host'].append(dict(operation=row['operation'], total_ms=total,
            measured_native_ms=measured, outside_instrumented_calls_ms=total-measured,
            median_stage_ms=row['median_host_stage_ms'], counts=samples[0]['counts']))
    for mode in ('fp16', 'w8a16'):
        path = OUT / 'profile' / ('b1-s24-' + mode + '.json')
        if not path.exists():continue
        value = json.loads(path.read_text())
        assert value['status'] == 'passed' and len(value['graphs']) == 60
        assert sha(path.with_name(path.stem + '-outputs.npz')) == value['outputs_sha256']
        aggregate = {}
        for op in ('prefill', 'decode'):
            graphs = [g for g in value['graphs'] if g['operation'] == op]
            assert sorted(g['layer'] for g in graphs) == list(range(30))
            totals = defaultdict(list)
            categories = defaultdict(list)
            # Combine across layers only by adding each graph's own interval union.
            for i in range(3):
                sums = defaultdict(float)
                groups = defaultdict(float)
                for graph in graphs:
                    sample = graph['samples'][i]
                    assert len({s['raw_sha256'] for s in graph['samples']}) == 3
                    raw = path.parent / (graph['tag'] + '-' + str(i) + '-tasks.bin')
                    assert sha(raw) == sample['raw_sha256']
                    for engine in ('dpu', 'shave', 'dma'):
                        sums[engine] += sample['engines'][engine]['union_ms']
                    sums['task_span'] += sample['timeline_span_ms']
                    sums['observed_union'] += sample['observed_union_ms']
                    sums['unobserved_gaps'] += sample['unobserved_gaps_ms']
                    by_type = defaultdict(list)
                    for task in sample['tasks']:
                        by_type[(task['exec_type'], task['layer_type'])].append(task)
                    for (engine, kind), tasks in by_type.items():
                        groups[str(engine) + ':' + kind] += union_ns(tasks) / 1e6
                    norm = [t for t in sample['tasks'] if t['name'].startswith('rmsnorm_')]
                    assert all(t['layer_type'] == 'Power' and t['exec_type'] == 2 for t in norm)
                    groups['rmsnorm_named_tasks'] += union_ns(norm) / 1e6
                for key, val in sums.items():totals[key].append(val)
                for key, val in groups.items():categories[key].append(val)
            aggregate[op] = dict(profiled_wall_ms=value[op]['median_ms'],
                profiled_host_wait_ms=st.median(s['host_wait_ms'] for s in value[op]['samples']),
                summed_graph_union_ms={key:st.median(vals) for key, vals in totals.items()},
                summed_graph_type_union_ms=dict(sorted(((key, st.median(vals)) for key, vals in categories.items()), key=lambda p:-p[1])),
                dma_duration_available=any(s['dma_duration_available'] for g in graphs for s in g['samples']))
        report['modes'][mode] = aggregate
        report['files'][str(path.relative_to(OUT))] = sha(path)
        report['profile_production_sha256'] = value['production_sha256']
        stats = value['final_stats']
        assert not stats['context_failed'] and all(stats[k] == 0 for k in ('live_buffers', 'cached_buffer_bytes', 'quarantined_buffer_bytes'))
    for name in ('torch_intel_npu.py', 'torch_npu_smollm.py', 'npunlock/src/graphinfer/session_win32.c',
                 'npunlock-probe/session-runtime/npunlock_session.dll', 'npunlock-probe/torch-device/_torch_intel_npu.pyd',
                 'npunlock/tests/fixtures/npu4000/rmsnorm.c', 'npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf'):
        report['current_sha256'][name] = sha(ROOT / name)
    report.update(status='passed', source_sha256=sha(Path(__file__)),
        interpretation='Interval unions are within each graph; sums across graphs/engines/types are not a whole-run critical-path decomposition. PERF_COUNT changes graph timing. Missing DMA records do not imply zero IO.')
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(host=report['host'], modes=report['modes']), indent=2), flush=True)


if __name__ == '__main__':summarize()

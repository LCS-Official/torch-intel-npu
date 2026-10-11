"""Offline provenance, units and completeness audit of the published hardware runs."""
import hashlib
import json
import math
import statistics
from pathlib import Path

from dpu_diagnostic import OUT, ROOT, SHAPES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(mapping):
    matched = {}
    for name,expected in mapping.items():
        original = Path(name)
        if not original.is_absolute(): original = ROOT/original
        candidates = [original,OUT/'initial-runtime'/original.name]
        if original.name == 'dpu_diagnostic.py': candidates.append(OUT/'scan-script.py')
        if original.name == 'dpu_replay_benchmark.py': candidates.append(OUT/'replay-script.py')
        candidates = [path for path in candidates if path.is_file() and sha(path) == expected]
        assert candidates,(name,expected)
        matched[name] = str(candidates[0].relative_to(ROOT))
    return matched


def rate(timing,shape,key):
    values = timing['samples_ms']; median = statistics.median(values)
    assert math.isclose(median,timing['median_ms'],rel_tol=1e-12)
    operations = 2*math.prod(shape)
    assert math.isclose(operations/(median*1e9),timing[key],rel_tol=1e-12)


def execute():
    audit = dict(status='running',files={},provenance={},published_successful_graph_calls=0)
    for shape in SHAPES:
        tag = '-'.join(map(str,shape)); path = OUT/(tag+'.json')
        report = json.loads(path.read_text(encoding='utf-8'))
        assert report['status'] == 'passed' and report['shape_mkn'] == list(shape)
        assert report['released_buffers'] == 4 and report['closed_sessions'] == 2
        assert report['successful_graph_calls'] == 74
        for row in report['checks']:
            assert row['bit_exact_grid']
            blob = OUT/f"{tag}-{row['path']}.blob"
            assert sha(blob) == row['blob_sha256']
            matrices = [v for v in row['dpu_modes'] if not v['elementwise']]
            assert matrices and all((v['amode'],v['wmode']) == (1,1) for v in matrices)
        for timing in report['timings'].values():
            assert len(timing['samples_ms']) == 31
            rate(timing,shape,'effective_teraops_per_second')
        audit['provenance'][path.name] = provenance(report['source_sha256'])
        audit['files'][path.name] = sha(path)
        audit['published_successful_graph_calls'] += report['successful_graph_calls']
    for tag in ('profile-128-1024-1024','profile-1024-1024-1024'):
        path = OUT/(tag+'.json'); report = json.loads(path.read_text(encoding='utf-8'))
        assert report['status'] == 'passed' and report['successful_graph_calls'] == 18
        assert report['released_buffers'] == 4 and report['closed_sessions'] == 2
        for data in report['device_profiling'].values():
            tasks = data['tasks']
            assert data['bytes'] == 344*len(tasks) and all(v['exec_type'] == 1 for v in tasks)
            span = max(v['start_time_ns']+v['duration_ns'] for v in tasks)-min(v['start_time_ns'] for v in tasks)
            assert math.isclose(data['timeline_span_ms'],span/1e6,rel_tol=1e-12)
        audit['provenance'][path.name] = provenance(report['source_sha256'])
        audit['files'][path.name] = sha(path)
        audit['published_successful_graph_calls'] += report['successful_graph_calls']
    for name in ('replay.json','openvino-paired.json'):
        path = OUT/name; report = json.loads(path.read_text(encoding='utf-8'))
        assert report['status'] == 'passed' and report['all_buffers_released'] and report['all_sessions_closed']
        for row in report['rows']:
            assert row['changed_input_and_restore_passed']
            for timing in row['timings'].values():
                assert len(timing['samples_ms']) == 31
                rate(timing,row['shape_mkn'],'effective_teraops_per_second' if name == 'replay.json' else 'effective_tops')
            if name == 'replay.json': assert row['released_input_rejected']
            else:
                assert row['execution_devices'] == 'NPU' and row['bit_exact_grid']
                tag = '-'.join(map(str,row['shape_mkn']))
                assert sha(OUT/f"paired-{tag}-{row['path']}-{row['compiler'].lower()}.blob") == row['shared_compiled_blob_sha256']
        expected_count = 5216 if name == 'replay.json' else 1936
        calls = report['successful_graph_calls'] if name == 'replay.json' else report['successful_native_graph_calls']+report['successful_openvino_graph_calls']
        assert calls == expected_count
        audit['published_successful_graph_calls'] += calls
        audit['provenance'][name] = provenance(report['source_sha256'])
        audit['files'][name] = sha(path)
    for name,source,count,expected_rows in (
            ('openvino.json',OUT/'openvino-script.py',624,16),
            ('openvino-tiles.json',ROOT/'npunlock-probe/research/openvino_dpu_benchmark.py',312,8)):
        path = OUT/name; report = json.loads(path.read_text(encoding='utf-8'))
        assert report['status'] == 'passed' and len(report['rows']) == expected_rows
        assert report['successful_graph_calls'] == count and sha(source) == report['source_sha256']
        for row in report['rows']:
            assert row['bit_exact_grid'] and row['changed_input_and_restore_passed']
            rate(row,row['shape_mkn'],'effective_tops')
            assert len(row['samples_ms']) == 31 and sha(OUT/row['blob_file']) == row['export_sha256']
            assert row['compiled_properties']['EXECUTION_DEVICES'] == 'NPU'
            tag = '-'.join(map(str,row['shape_mkn']))
            assert sha(OUT/f"{tag}-{row['path']}.xml") == row['ir_sha256']
            assert sha(OUT/f"{tag}-{row['path']}.bin") == row['weights_sha256']
            assert sha(OUT/f'fixture-{tag}.npz') == row['fixture_sha256']
        audit['files'][name] = sha(path)
        audit['published_successful_graph_calls'] += count
    graph_report = json.loads((OUT/'openvino-graphs.json').read_text(encoding='utf-8'))
    assert len(graph_report['rows']) == 40
    assert graph_report['inspector_sha256'] == sha(ROOT/'npunlock-probe/research/inspect_openvino_graphs.py')
    for row in graph_report['rows']:
        assert row['matrix_modes'] == [[1,1]] and sha(OUT/row['file']) == row['sha256']
        assert not row['sections']['task.shave.invocation.']
        if row['file'].startswith('tiles-'):
            tiles = int(Path(row['file']).stem.rsplit('-',1)[1])
            assert len(row['sections']['task.dpu.variant.']) == tiles
    for path in (OUT/'build.json',OUT/'initial-runtime/build.json'):
        report = json.loads(path.read_text(encoding='utf-8'))
        audit['provenance'][str(path.relative_to(OUT))] = provenance(report['source_sha256'])
        assert sha(path.parent/'session_diagnostic.dll') == report['dll_sha256']
    production = {
        'npunlock-probe/session-runtime/npunlock_session.dll':'5d145c0f1dd455994d47b4ca0433afb4812347075e673bf3fa653ca75b5d1e44',
        'npunlock-probe/torch-device/_torch_intel_npu.pyd':'7ff42b9b3a528fa2efbf91989008dab0ac91a809225cef3dfb299a6a1d4aa267',
        'npunlock/src/graphinfer/session_win32.c':'0ce6e0178f273637907aefbf0b2c9bc0b6d567f84cc3f8aac3e58f877994388c',
        'torch_npu_lowbit.py':'29e4554d1c6234097040ab8ec7d1fd61243ab9bf11aee8d2edfa79829c687404',
        'torch_intel_npu.py':'539d97fa2b180d4f7deca40abed4e52d96397c968f4f60a48ceb7c9291014e35',
        'torch_npu_smollm.py':'cb51080272e443b95241900edebf92690ca23842e057497585bb13ab8e009b89'}
    assert all(sha(ROOT/name) == value for name,value in production.items())
    audit['production_unchanged_sha256'] = production
    assert audit['published_successful_graph_calls'] == 8716
    audit['auditor_sha256'] = sha(Path(__file__))
    audit['files']['REPORT.md'] = sha(OUT/'REPORT.md')
    audit['status'] = 'passed'
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=audit['status'],published_successful_graph_calls=audit['published_successful_graph_calls'],
                         compiled_graphs_checked=40,production_unchanged=True)))


if __name__ == '__main__':
    execute()

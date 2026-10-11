"""Verify the full-stack profiles and rejected candidates without NPU work."""
import json
from pathlib import Path
import statistics as st

import numpy as np
from w8_quality_probe import ROOT, sha
from decoder_native_summary import OUT, summarize


def read(path):return json.loads(path.read_text())


def released(report):
    state = report['final_stats']
    assert state['closed'] and not state['context_failed']
    assert all(state[k] == 0 for k in ('live_buffers', 'cached_buffer_bytes', 'quarantined_buffer_bytes', 'cached_sessions'))


def main():
    summarize()
    summary = read(OUT/'summary.json')
    assert set(summary['modes']) == {'fp16', 'w8a16'}
    report = dict(status='running', candidates={}, profiles={}, verified_files={}, source_sha256=sha(Path(__file__)))
    build = read(OUT/'profile/build.json')
    assert sha(ROOT/'torch_npu_device_profile.py') == build['builder_sha256']
    assert sha(OUT/'profile/session_task_profile.dll') == build['dll_sha256']
    assert sha(OUT/'profile/session_task_profile.c') == build['generated_source_sha256']
    host = read(OUT/'host.json');released(host)
    for name, digest in host['sources'].items():assert sha(ROOT/name) == digest, name
    with np.load(OUT/'host-outputs.npz') as h, np.load(OUT/'profile/b1-s24-fp16-outputs.npz') as d:
        for i in range(182):
            plain = 0 if i == 0 else 1 if i == 91 else i+1 if i <= 90 else i
            np.testing.assert_array_equal(h[f'b1-s24/{i}'].view(np.uint8), d[f'plain/{plain}'].view(np.uint8))
    report['host_cross_profile_bit_equal_arrays'] = 182
    for mode in ('fp16', 'w8a16'):
        path = OUT/'profile'/f'b1-s24-{mode}.json'
        data = read(path);released(data)
        for name, digest in data['production_sha256'].items():assert sha(ROOT/name) == digest, name
        with np.load(path.with_name(path.stem+'-outputs.npz')) as values:
            count = len([key for key in values if key.startswith('plain/')]);assert count == 182
            for i in range(count):
                np.testing.assert_array_equal(values[f'plain/{i}'].view(np.uint8), values[f'profile/{i}'].view(np.uint8))
            for i in range(2, 92):
                np.testing.assert_array_equal(values[f'plain/{i}'].view(np.uint8), values[f'seed/{i}'].view(np.uint8))
        report['profiles'][mode] = dict(graphs=60, fresh_task_records=180, bit_equal_arrays=count, immutable_seed_arrays=90)
    for name, label, expected in (('duplicate-rows', 'duplicate', 184), ('scalar-bits', 'scalar', 158)):
        directory = OUT/name;build = read(directory/'build.json')
        assert sha(directory/'builder.py') == build['builder_sha256']
        for kernel, fields in build['kernels'].items():
            for suffix, field in (('.c', 'source_sha256'), ('.s', 'assembly_sha256'), ('.elf', 'elf_sha256')):
                assert sha(directory/(kernel+suffix)) == fields[field]
        for relative, suffix in (('rmsnorm.c', '.c'), ('rmsnorm/rmsnorm-f16.s', '.s'), ('rmsnorm/rmsnorm-f16.elf', '.elf')):
            assert sha(ROOT/'npunlock/tests/fixtures/npu4000'/relative) == sha(directory/('baseline'+suffix))
        numeric = read(directory/('numeric-'+label+'.json'));model = read(directory/'full30.json')
        assert numeric['status'] == model['status'] == 'passed'
        assert len(numeric['checks']) == expected and all(c['bit_differences'] == 0 for c in numeric['checks'])
        assert model['checks'][0]['bit_equal_arrays'] == 182 and model['immutable_seed_arrays'] == 90
        assert sha(directory/'full30-outputs.npz') == model['outputs_sha256']
        released(numeric);released(model)
        assert len(model['graphs']) == 60
        for graph in model['graphs']:
            assert graph['baseline'][0]['kernel_elf_sha256'] == build['kernels']['baseline']['elf_sha256']
            assert graph['codec'][0]['kernel_elf_sha256'] == build['kernels']['codec']['elf_sha256']
            if graph['operation'] == 'decode':
                assert all(c['elements'] == 576 for c in graph['baseline'][0]['contracts'])
        with np.load(directory/'full30-outputs.npz') as values:
            for i in range(182):np.testing.assert_array_equal(values[f'baseline/{i}'].view(np.uint8), values[f'codec/{i}'].view(np.uint8))
            for i in range(90):np.testing.assert_array_equal(values[f'baseline/{1+i}'].view(np.uint8), values[f'preserved/{i}'].view(np.uint8))
        timings = {}
        for row in model['timings']:
            assert len(row['pairs_ms']) == 31
            assert row['warm_delta']['driver_allocations'] == 0
            for kernel in ('baseline', 'codec'):
                assert len(row['samples_ms'][kernel]) == 62
                assert st.median(row['samples_ms'][kernel]) == row['median_ms'][kernel]
            timings[row['operation']] = dict(median_ms=row['median_ms'], wins=row['positive_rounds']['codec'],
                latency_change_percent=100*(row['median_ms']['codec']/row['median_ms']['baseline']-1))
        report['candidates'][name] = dict(adopted=False, numeric_checks=expected, bit_equal_arrays=182,
            immutable_seed_arrays=90, timings=timings, reason='No full-stack decode benefit')
    for path in OUT.rglob('*'):
        if path.is_file() and path.name not in ('audit.json', 'REPORT.md'):
            report['verified_files'][str(path.relative_to(OUT))] = sha(path)
    report.update(status='passed', production_unchanged=True, summary_sha256=sha(OUT/'summary.json'))
    (OUT/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('status', 'production_unchanged', 'profiles', 'candidates')}, indent=2))


if __name__ == '__main__':main()

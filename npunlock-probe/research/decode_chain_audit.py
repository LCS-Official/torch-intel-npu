"""Build fake-driver failure checks and audit the isolated decode-chain results."""
import argparse
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import numpy as np
from decode_chain import OUT, HERE, read, write
from w8_quality_probe import ROOT, sha


def failures():
    source = HERE / 'test_decode_chain_failure.c'; output = OUT / 'test-decode-chain-failure.exe'
    args = read(OUT / 'host/profile-build.json')['arguments']
    args = [str(source) if p == str(OUT / 'host/session_profile.c') else
            str(output) if p == str(OUT / 'host/session_profile.dll') else p for p in args if p != '-shared']
    built = subprocess.run(args, env=dict(os.environ, PATH=str(Path(args[0]).parent)+';'+os.environ['PATH']), capture_output=True, text=True)
    (OUT / 'failure-build.log').write_text(built.stdout + built.stderr, encoding='utf-8'); built.check_returncode()
    tested = subprocess.run([str(output)], capture_output=True, text=True, timeout=20)
    (OUT / 'failures.log').write_text(tested.stdout + tested.stderr, encoding='utf-8'); tested.check_returncode()
    write(OUT / 'failures.json', dict(status='passed', chain_cases=10, production_failure_suite='passed',
          source_sha256=sha(source), chain_sha256=sha(HERE / 'decode_chain_extra.c'), executable_sha256=sha(output), arguments=args))
    print(tested.stdout, flush=True)


def same(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


def audit():
    before = read(OUT / 'before/sources.json'); build = read(OUT / 'build.json')
    wrappers = {'npunlock-probe/research/decode_chain.py'}
    # The runner acquired extra validation after the first host-profile run;
    # its initial copy and the exact plain runner are both retained.
    frozen = {name: digest for name, digest in before['sources'].items() if name not in wrappers}
    for name, digest in before['sources'].items(): assert sha(OUT / 'before' / name) == digest
    for name, digest in frozen.items(): assert sha(ROOT / name) == digest, name
    for name, value in build['runtimes'].items():
        assert sha(OUT / (name+'.dll')) == value['dll_sha256']
        assert sha(OUT / (name+'.c')) == value['source_sha256']
        # The chain is appended to either the original or counter-instrumented session source.
        base = ROOT / 'npunlock/src/graphinfer/session_win32.c' if name == 'plain' else OUT / 'host/session_profile.c'
        assert (OUT / (name+'.c')).read_text(encoding='utf-8') == base.read_text(encoding='utf-8') + '\n' + (HERE / 'decode_chain_extra.c').read_text(encoding='utf-8')
    extension = read(OUT / 'extension/build.json')
    assert sha(OUT / 'extension/_torch_intel_npu.pyd') == extension['extension_sha256']
    assert sha(OUT / 'extension/backend.cpp') == extension['source_sha256']
    checks = read(OUT / 'checks.json'); failed = read(OUT / 'failures.json')
    assert checks['status'] == failed['status'] == 'passed' and len(checks['checks']) == 8 and failed['chain_cases'] == 10
    assert checks['final_stats']['closed'] and not checks['final_stats']['context_failed']
    assert all(checks['final_stats'][k] == 0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_blocks','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
    assert failed['source_sha256'] == sha(HERE / 'test_decode_chain_failure.c')
    assert failed['chain_sha256'] == sha(HERE / 'decode_chain_extra.c')
    assert failed['executable_sha256'] == sha(OUT / 'test-decode-chain-failure.exe')
    records = {label: read(OUT / (label+'.json')) for label in ('host', 'plain')}
    comparisons = 0
    for label, report in records.items():
        assert report['status'] == 'passed'
        assert all(report['experiment']['sources'][name] == sha(ROOT / name) for name in report['experiment']['sources'])
        assert sha(OUT / (label+'-outputs.npz')) == report['outputs_sha256']
        if label == 'plain': assert sha(OUT / 'plain-runner.py') == report['wrapper_sha256']
        final = report['final_stats']
        assert final['closed'] and not final['context_failed']
        assert all(final[k] == 0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_blocks','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
        assert len(report['graphs']) == 60
        with np.load(OUT / (label+'-outputs.npz'), allow_pickle=False) as arrays:
            for strategy in ('baseline','group1','group2','group4'):
                for i in range(91): same(arrays[f'baseline/{i}'], arrays[f'{strategy}/{i}']); comparisons += 1
            if label == 'plain':
                for i in range(90): same(arrays[f'seed-before/{i}'], arrays[f'seed/{i}']); comparisons += 1
                assert report['advancing_decode'] == dict(steps=8, bit_equal_arrays=728, final_position=40)
                for step in range(8):
                    for i in range(91): same(arrays[f'advance/{step}/baseline/{i}'], arrays[f'advance/{step}/group4/{i}']); comparisons += 1
                    for layer in range(30):
                        assert arrays[f'advance/{step}/group4/{3+layer*3}'].item() == 33 + step
        for timing in report['timings']:
            assert len(timing['samples']) == 84 and len(timing['pairs_total_ms']) == 21
            assert timing['warm_delta']['npu_calls'] == 2520 and all(v == 0 for k,v in timing['warm_delta'].items() if k != 'npu_calls')
            for strategy in ('baseline', timing['candidate']):
                rows = [r for r in timing['samples'] if r['strategy'] == strategy]
                assert len(rows) == 42
                for field in ('api_ms','cleanup_ms','total_ms'):
                    assert timing['median_ms'][strategy][field] == st.median(r[field] for r in rows)
                if label == 'host':
                    count = 30 if strategy == 'baseline' else (30+int(strategy[-1])-1)//int(strategy[-1])
                    for row in rows:
                        assert row['stage_counts']['append_graph'] == 30
                        assert all(row['stage_counts'][stage] == count for stage in ('close_list','submit','wait','reset_list','reset_fence'))
    assert records['host']['graphs'] == records['plain']['graphs']
    assert records['host']['experiment']['checkpoint_sha256'] == records['plain']['experiment']['checkpoint_sha256']
    with np.load(OUT / 'host-outputs.npz', allow_pickle=False) as host, np.load(OUT / 'plain-outputs.npz', allow_pickle=False) as plain:
        for i in range(91): same(host[f'baseline/{i}'], plain[f'baseline/{i}']); comparisons += 1
    result = dict(status='passed', frozen_files=len(frozen), exact_array_comparisons=comparisons,
                  shared_graph_hashes=60, real_device_checks=8, synthetic_chain_cases=10,
                  final_stats_healthy=True, production_unchanged=True,
                  files={p.name:sha(p) for p in (HERE/'decode_chain.py', HERE/'decode_chain_extra.c', HERE/'decode_chain_extra.cpp',
                                               HERE/'test_decode_chain_failure.c', Path(__file__))})
    write(OUT / 'audit.json', result); print(json.dumps(result, indent=2), flush=True)
    for timing in records['plain']['timings']:
        name = timing['candidate']; base = timing['median_ms']['baseline']['total_ms']; candidate = timing['median_ms'][name]['total_ms']
        pairs = [100*(1-p[name]/p['baseline']) for p in timing['pairs_total_ms']]
        print(name, 'baseline',base,'candidate',candidate,'reduction_pct',100*(1-candidate/base),
              'paired_median_pct',st.median(pairs),'wins',timing['positive_rounds'])
    for timing in records['host']['timings']:
        print('HOST', timing['candidate'])
        for strategy in ('baseline', timing['candidate']):
            rows = [r for r in timing['samples'] if r['strategy'] == strategy]
            stages = {k:st.median(r['stages_ms'][k] for r in rows) for k in rows[0]['stages_ms']}
            residual = st.median(r['api_ms']-sum(r['stages_ms'].values()) for r in rows)
            print(strategy, json.dumps(stages), 'unattributed_api_ms',residual)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--case', choices=('failures','audit'), required=True)
    args = parser.parse_args()
    if args.case == 'failures': failures()
    else: audit()

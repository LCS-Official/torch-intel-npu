"""Bitwise old/new resident-reduction comparison and interleaved kernel timings."""
import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'npunlock-probe/research/results/compatibility'
OLD = ROOT / 'npunlock-probe/research/compat-before/npunlock/tests/fixtures/npu4000/resident'


def execute():
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from npunlock.npu4000 import compile_reduce
    torch.manual_seed(73)
    report = {'status': 'running', 'bitwise_checks': [], 'timings': [],
              'source_sha256': hashlib.sha256((ROOT/'npunlock/tests/fixtures/npu4000/resident_reduce.c').read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256((ROOT/'npunlock/tests/fixtures/npu4000/resident/manifest.json').read_bytes()).hexdigest(),
              'timing_scope': 'resident synchronous wall time; compilation/uploads/check downloads excluded'}
    old_manifest = json.loads((OLD/'manifest.json').read_text())

    def reference(shape, axes, kind):
        key = ('old_reduce', kind, shape, axes)
        if key not in npu._sessions:
            elf = (OLD/f'{kind}.elf').read_bytes()
            assert hashlib.sha256(elf).hexdigest() == old_manifest['kernels'][kind]['elf_sha256']
            program = compile_reduce(shape, axes, elf, kind=kind, libraries=npu._native)
            session = npu._native.create_inference_session(program.graph_blob, timeout_ms=10000, parent=npu._root_session)
            assert len(npu._sessions) < npu._MAX_SESSIONS
            npu._sessions[key] = session, program
            npu._session_creations += 1
        session, program = npu._sessions[key]

        def run(x, out):
            npu._C.run(session._result.session, [x], [program.graph.inputs[0].name], out, program.graph.outputs[0].name)
        return run

    def compare(shape, axes, kind):
        run = reference(shape, axes, kind)
        output_shape = tuple(1 if i in axes else d for i, d in enumerate(shape))
        for pattern in ('finite', 'ieee_edges'):
            host = torch.randn(shape, dtype=torch.float32)/4
            if pattern == 'ieee_edges':
                values = np.array([0x80000000, 1, 0x007fffff, 0x00800000, 0x80800000,
                                   0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc12345], dtype=np.uint32)
                host = torch.from_numpy(np.resize(values.view(np.float32), host.numel()).reshape(shape).copy())
            x = host.to('intel_npu')
            before = torch.empty(output_shape, dtype=torch.float32, device='intel_npu')
            run(x, before)
            after = npu._reduce(x, axes, True, kind=kind)
            np.testing.assert_array_equal(before.cpu().numpy().view(np.uint32), after.cpu().numpy().view(np.uint32))
        report['bitwise_checks'].append({'shape': shape, 'axes': axes, 'kind': kind, 'patterns': 2, 'status': 'passed'})

    def measure(shape, axes, kind):
        run = reference(shape, axes, kind)
        output_shape = tuple(1 if i in axes else d for i, d in enumerate(shape))
        x = (torch.randn(shape)/4).to('intel_npu')
        old = torch.empty(output_shape, dtype=torch.float32, device='intel_npu')
        new = torch.empty_like(old)
        for _ in range(3):
            run(x, old)
            npu._reduce(x, axes, True, kind=kind, out=new)
        baseline = npu.stats()
        samples = {'old': [], 'new': []}
        for i in range(9):
            for version in (('old', 'new') if i % 2 == 0 else ('new', 'old')):
                started = time.perf_counter()
                if version == 'old':
                    run(x, old)
                else:
                    npu._reduce(x, axes, True, kind=kind, out=new)
                samples[version].append((time.perf_counter()-started)*1000)
        stats = npu.stats()
        for field in ('host_upload_copies', 'host_download_copies', 'host_device_copies', 'session_creations'):
            assert stats[field] == baseline[field], field
        np.testing.assert_array_equal(old.cpu().numpy().view(np.uint32), new.cpu().numpy().view(np.uint32))
        old_ms, new_ms = (statistics.median(samples[name]) for name in ('old', 'new'))
        report['timings'].append({'shape': shape, 'axes': axes, 'kind': kind, 'old_ms': old_ms, 'new_ms': new_ms,
                                  'speedup': old_ms/new_ms, 'samples_ms': samples, 'warm_host_copies_and_compiles': 0})
        print(shape, axes, kind, f'{old_ms:.3f} -> {new_ms:.3f} ms', flush=True)

    try:
        with torch.no_grad():
            for shape in ((1,), (31,), (7, 5), (3, 4, 5), (2, 3, 4, 5), (1, 3, 1, 5)):
                for size in range(1, len(shape)+1):
                    for axes in itertools.combinations(range(len(shape)), size):
                        for kind in ('sum', 'mean'):
                            compare(shape, axes, kind)
            for shape, axes, kind in (((1, 256, 256), (1,), 'sum'), ((64, 64, 16), (1,), 'sum'),
                                      ((65536,), (0,), 'sum'), ((4, 4, 32, 32), (0, 2, 3), 'mean')):
                measure(shape, axes, kind)
        gc.collect()
        report['stats_before_shutdown'] = npu.stats()
        assert report['stats_before_shutdown']['live_buffers'] == 0
        npu.shutdown()
        report['final_stats'] = npu.stats()
        assert not report['final_stats']['context_failed'] and report['final_stats']['live_buffers'] == 0
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc(), failure_stats=npu.stats())
        print(report['traceback'], flush=True)
    (OUT/'reduce-optimization.json').write_text(json.dumps(report, indent=2))
    return int(report['status'] != 'passed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.child:
        return execute()
    completed = subprocess.run([sys.executable, __file__, '--child'], capture_output=True, text=True, timeout=400)
    (OUT/'reduce-optimization.log').write_text(completed.stdout+completed.stderr)
    print(completed.stdout+completed.stderr)
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())

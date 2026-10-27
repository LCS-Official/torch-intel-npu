"""Real-device cast graph reuse, IEEE values, layouts and bounded tiling checks."""
import argparse
import gc
import hashlib
import json
from math import prod
from pathlib import Path
from statistics import median
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'npunlock-probe/research/results/cast-reuse'


def execute():
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from torch_npu_sustained_train import process_memory
    torch.set_num_threads(1)
    result = dict(status='running', groups=[])

    def close(actual, expected):
        actual = actual.cpu()
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        integer = torch.int16 if expected.dtype == torch.float16 else torch.int32
        valid = ~torch.isnan(expected)
        assert torch.equal(actual.view(integer)[valid], expected.view(integer)[valid])

    def no_transfer(before):
        after = npu.stats()
        for key in ('host_upload_copies', 'host_download_copies', 'host_device_copies', 'host_pack_copies'):
            assert after[key] == before[key], key

    def workload():
        rng = np.random.default_rng(241)
        for shapes in (((96,), (8, 12), (2, 3, 16), (2, 2, 4, 6)),
                       ((), (1,), (1, 1), (1, 1, 1), (1, 1, 1, 1))):
            rows = []
            for source_dtype, target_dtype in ((torch.float16, torch.float32), (torch.float32, torch.float16)):
                for shape in shapes:
                    count = prod(shape)
                    # Different payloads in each shape detect stale bindings on a reused graph.
                    bits = rng.integers(0, 2**32, count, dtype=np.uint32)
                    host = torch.from_numpy(bits.view(np.float32)).to(source_dtype).reshape(shape)
                    source = host.to('intel_npu')
                    out = torch.empty(shape, dtype=target_dtype, device='intel_npu')
                    before = npu.stats(); start = time.perf_counter()
                    converted = source.to(target_dtype)
                    cold = time.perf_counter() - start
                    assert out.copy_(source) is out
                    no_transfer(before)
                    after = npu.stats()
                    created = after['session_creations'] - before['session_creations']
                    close(converted, host.to(target_dtype)); close(out, host.to(target_dtype))
                    samples = []
                    for _ in range(9):
                        before = npu.stats(); start = time.perf_counter()
                        assert out.copy_(source) is out
                        samples.append(time.perf_counter() - start)
                        no_transfer(before)
                        assert npu.stats()['session_creations'] == before['session_creations']
                    rows.append(dict(shape=shape, source=str(source_dtype), cold_seconds=cold,
                        warm_median_seconds=median(samples), new_sessions=created))
            assert sum(row['new_sessions'] for row in rows) == 3, 'Equal-count casts did not share graphs'
            result['groups'].append(rows)
            print('same-count shapes', shapes, 'passed', flush=True)

        for count in (65536, 65537, 1048576):
            # Exhaustive half encodings, also spanning full-tile and one-element tail boundaries.
            host = torch.from_numpy(np.resize(np.arange(65536, dtype=np.uint16), count).view(np.float16))
            x = host.to('intel_npu'); before = npu.stats()
            wide = x.float(); narrow = wide.half()
            no_transfer(before); close(wide, host.float()); close(narrow, host.float().half())
        print('exhaustive half / tile boundaries passed', flush=True)

        for source_dtype, target_dtype in ((torch.float16, torch.float32), (torch.float32, torch.float16)):
            host = torch.linspace(-2, 2, 24).to(source_dtype).view(4, 6)
            x = host.to('intel_npu')
            for view, expected in ((x.t(), host.t()), (x[:, 1:5], host[:, 1:5])):
                out = torch.empty(expected.shape, dtype=target_dtype, device='intel_npu')
                before = npu.stats(); actual = view.to(target_dtype)
                assert out.copy_(view) is out
                no_transfer(before); close(actual, expected.to(target_dtype)); close(out, expected.to(target_dtype))
            out = torch.empty(6, 4, dtype=target_dtype, device='intel_npu').t()
            before = npu.stats(); assert out.copy_(x) is out
            no_transfer(before); close(out, host.to(target_dtype))
            before = npu.stats()
            for action in (lambda: npu._cast(x.t(), target_dtype),
                           lambda: npu._cast(x[:, 1:5], target_dtype),
                           lambda: npu._cast(x, target_dtype, out=torch.empty(24, dtype=target_dtype, device='intel_npu')),
                           lambda: npu._cast(x, target_dtype, out=x)):
                try: action()
                except ValueError: pass
                else: raise AssertionError('Cast accepted an invalid native binding or output')
            assert npu.stats()['npu_calls'] == before['npu_calls']
        result.update(exhaustive_half_encodings=65536, max_elements=1048576,
            scalar_rank_restored=True, public_dense_partial_and_dense_out=True,
            invalid_native_bindings_rejected=8, finite_bits_exact=True)

    try:
        workload()
        result['status'] = 'passed'
    except Exception as error:
        result.update(status='failed', error=str(error), traceback=traceback.format_exc())
    finally:
        gc.collect(); result['before_shutdown'] = npu.stats(); result['process_memory'] = process_memory()
        npu.shutdown(); result['final_stats'] = npu.stats()
        final = result['final_stats']
        if final['context_failed'] or not final['closed'] or any(final[key] for key in (
                'live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'cached_sessions', 'quarantined_buffer_bytes')):
            result.update(status='failed', lifecycle_error='Unhealthy final device state')
        result['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'torch_intel_npu.py', 'torch_npu_cast_reuse_demo.py', 'npunlock/tests/fixtures/npu4000/arithmetic.c',
            'npunlock/tests/fixtures/npu4000/precision/manifest.json')}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='after'); args = parser.parse_args()
    result = execute(); OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{args.label}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['status'], result.get('traceback', ''), flush=True)
    raise SystemExit(result['status'] != 'passed')

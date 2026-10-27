"""Same-input warmed attention timings and host-side submission attribution."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time


def execute(args):
    import torch
    import torch_intel_npu as npu
    from torch.nn.attention import sdpa_kernel, SDPBackend

    torch.set_num_threads(1)
    torch.manual_seed(431)
    dtype = getattr(torch, args.precision)
    host = [(torch.randn(1, 2, length, args.head_dim, dtype=dtype) / 4).requires_grad_()
            for length in (args.query_length, args.key_length, args.key_length)]
    dev = [x.detach().to('intel_npu').requires_grad_() for x in host]
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(*host)
    seed = torch.randn_like(expected)
    expected.backward(seed)
    upstream = seed.to('intel_npu')
    profile = defaultdict(lambda: dict(calls=0, seconds=0.))
    originals = {}
    enabled = False

    def instrument(name):
        original = getattr(npu, name)
        originals[name] = original

        def wrapped(*values, **kwargs):
            if not enabled:
                return original(*values, **kwargs)
            category = values[0][0] if name in ('_run', '_run_ranges') else name
            started = time.perf_counter()
            try:
                return original(*values, **kwargs)
            finally:
                row = profile[str(category)]
                row['calls'] += 1
                row['seconds'] += time.perf_counter() - started
        setattr(npu, name, wrapped)

    for name in ('_run', '_run_ranges', '_run_many', '_run_chain'):
        instrument(name)
    saved = {}

    def pack(t):
        saved[t.untyped_storage()._cdata] = t.untyped_storage().nbytes()
        return t

    durations = []
    warm_before = npu.stats()
    actual = None
    try:
        for iteration in range(args.warmup + args.repeat):
            actual = None
            for t in dev:
                t.grad = None
            saved.clear()
            enabled = iteration >= args.warmup
            before = npu.stats()
            begin = time.perf_counter()
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                if args.mode == 'math':
                    with sdpa_kernel(SDPBackend.MATH):
                        actual = torch.nn.functional.scaled_dot_product_attention(*dev)
                elif args.query_block is None and args.key_block is None:
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        actual = torch.nn.functional.scaled_dot_product_attention(*dev)
                else:
                    blocks = {k: v for k, v in dict(query_block=args.query_block, key_block=args.key_block).items() if v is not None}
                    actual = torch.intel_npu.flash_attention(*dev, **blocks)
            middle = time.perf_counter()
            actual.backward(upstream)
            end = time.perf_counter()
            after = npu.stats()
            row = dict(iteration=iteration, measured=enabled, forward_seconds=middle-begin,
                       backward_seconds=end-middle, seconds=end-begin,
                       delta={k: after[k]-before[k] for k in ('npu_calls', 'chain_groups', 'chain_graphs',
                           'session_creations', 'graph_cache_evictions', 'host_upload_copies', 'host_download_copies')})
            durations.append(row)
            print(json.dumps(row), flush=True)
    finally:
        enabled = False
        for name, original in originals.items():
            setattr(npu, name, original)
    measured = durations[args.warmup:]
    assert all(row['delta']['session_creations'] == 0 and row['delta']['graph_cache_evictions'] == 0 for row in measured), 'Measurement was not fully warmed'
    assert all(row['delta']['host_upload_copies'] == 0 and row['delta']['host_download_copies'] == 0 for row in measured), 'Host payload transfer in measured loop'
    errors = []
    for a, b in [(actual, expected)] + [(a.grad, b.grad) for a, b in zip(dev, host)]:
        value = a.detach().cpu()
        torch.testing.assert_close(value, b.detach(), rtol=.01 if dtype == torch.float16 else 5e-5,
                                   atol=.002 if dtype == torch.float16 else 5e-6)
        errors.append(float((value.float()-b.detach().float()).abs().max()))
    before_close = npu.stats()
    saved_bytes = sum(saved.values())
    del actual, dev, upstream, a, t
    gc.collect()
    npu.shutdown()
    gc.collect()
    after_close = npu.stats()
    healthy = after_close['closed'] and not after_close['context_failed'] and all(after_close[k] == 0 for k in (
        'live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'cached_sessions', 'quarantined_buffer_bytes'))
    assert healthy, after_close
    paths = ('torch_intel_npu.py', 'torch_npu_flash_attention.py', 'torch_npu_compat_extended.py',
             'npunlock/tests/fixtures/npu4000/resident/manifest.json')
    result = dict(status='passed', config=vars(args), durations=durations,
        median_seconds=statistics.median(row['seconds'] for row in measured),
        median_forward_seconds=statistics.median(row['forward_seconds'] for row in measured),
        median_backward_seconds=statistics.median(row['backward_seconds'] for row in measured),
        profile=dict(sorted(profile.items(), key=lambda pair: -pair[1]['seconds'])),
        profile_scope='Host wall time including Python, session lookup, native submit, fence and kernel; not hardware-only time',
        saved_unique_storage_bytes_including_inputs=saved_bytes, max_abs_output_qkv_gradient_errors=errors,
        warm_before=warm_before, stats=before_close, shutdown=after_close,
        source_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths})
    destination = Path(__file__).resolve().parent/'npunlock-probe/research/results/flash-optimization'
    destination.mkdir(exist_ok=True)
    (destination/f'{args.label}-{args.mode}-{args.precision}-d{args.head_dim}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='passed', median_seconds=result['median_seconds'], profile=result['profile'],
        peak_buffer_bytes=before_close['peak_buffer_bytes'], shutdown_healthy=healthy)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='current')
    parser.add_argument('--mode', choices=('math', 'flash'), default='flash')
    parser.add_argument('--precision', choices=('float16', 'float32'), default='float32')
    parser.add_argument('--head-dim', type=int, default=16)
    parser.add_argument('--query-length', type=int, default=128)
    parser.add_argument('--key-length', type=int, default=256)
    parser.add_argument('--query-block', type=int)
    parser.add_argument('--key-block', type=int)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=3)
    arguments = parser.parse_args()
    if arguments.warmup < 1 or arguments.repeat < 1:
        parser.error('At least one warmup and one measured repetition are required')
    execute(arguments)

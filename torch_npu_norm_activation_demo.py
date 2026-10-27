"""CPU/NPU LayerNorm, GELU and residual feed-forward training acceptance."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/norm-activation'
CASES = ('layernorm', 'gelu', 'contracts', 'training')


def execute(case):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(73)
    torch.backends.mkldnn.enabled = False
    checks = []
    maximum_error = {'forward': 0., 'backward': 0.}

    def close(actual, expected, category='forward'):
        a, b = actual.detach().cpu(), expected.detach().cpu()
        finite = torch.isfinite(a) & torch.isfinite(b)
        if finite.any():
            maximum_error[category] = max(maximum_error[category], float((a[finite]-b[finite]).abs().max()))
        torch.testing.assert_close(a, b, rtol=.003 if a.dtype == torch.float16 else 3e-5,
                                   atol=.0005 if a.dtype == torch.float16 else 3e-6, equal_nan=True)

    def check(name, function):
        before = npu.stats()
        row = {'name': name}
        try:
            row.update(function() or {})
            row['status'] = 'passed'
        except Exception as error:
            row.update(status='failed', error=str(error), traceback=traceback.format_exc())
        gc.collect()
        after = npu.stats()
        row['npu_calls'] = after['npu_calls']-before['npu_calls']
        row['live_buffer_delta'] = after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
        if row['live_buffer_delta'] or after['context_failed']:
            row.update(status='failed', lifecycle_error='Buffer leak or failed context')
        checks.append(row)
        print(name, row['status'], row.get('error', '')[:300], flush=True)

    def layernorm(dtype, mixed=False):
        comparisons = 0
        specifications = [((4,), (4,)), ((3, 4), (4,)), ((2, 3, 5), (3, 5)),
                          ((2, 2, 3, 5), (3, 5)), ((2, 3, 5), (2, 3, 5)),
                          ((2, 1), (1,)), ((1, 4096), (4096,)), ((17, 4096), (4096,))]
        for shape, normalized in specifications:
            for affine in ('both', 'none', 'weight', 'bias'):
                if mixed and affine == 'bias':
                    continue  # CPU native backward rejects FP16 + bias-only FP32.
                host = torch.randn(shape, dtype=dtype).requires_grad_()
                # Exercise complete dense permutations without a partial storage view.
                if len(shape) == 2 and shape == (3, 4):
                    host = host.detach().t().contiguous().t().requires_grad_()
                param_dtype = torch.float32 if mixed else dtype
                w = torch.randn(normalized, dtype=param_dtype).requires_grad_() if affine in ('both', 'weight') else None
                b = torch.randn(normalized, dtype=param_dtype).requires_grad_() if affine in ('both', 'bias') else None
                values = [v.detach().to('intel_npu', memory_format=torch.contiguous_format).requires_grad_()
                          if v is not None else None for v in (host, w, b)]
                if not host.is_contiguous():
                    values[0] = values[0].detach().t().contiguous().t().requires_grad_()
                expected = torch.native_layer_norm(host, normalized, w, b, 1e-5)
                actual = torch.native_layer_norm(values[0], normalized, values[1], values[2], 1e-5)
                for a, e in zip(actual, expected):
                    assert a.dtype == e.dtype and a.shape == e.shape
                    close(a, e)
                    comparisons += 1
                seed = (torch.randn_like(host)/4).contiguous()
                expected[0].backward(seed)
                actual[0].backward(seed.to('intel_npu'))
                for a, e in zip(values, (host, w, b)):
                    if a is not None:
                        try:
                            close(a.grad, e.grad, 'backward')
                        except AssertionError as error:
                            raise AssertionError(f'{shape}/{normalized}/{affine}, gradient {tuple(a.shape)}: {error}') from error
                        comparisons += 1
        for data in (torch.ones(2, 4), torch.tensor([[100., 100.125, 99.875, 100.25]]),
                     torch.tensor([[float('inf'), 0., -1., 2.], [float('nan'), 1., 2., 3.]])):
            host = data.to(dtype).requires_grad_()
            x = host.detach().to('intel_npu').requires_grad_()
            expected = torch.nn.functional.layer_norm(host, (4,))
            actual = torch.nn.functional.layer_norm(x, (4,))
            seed = torch.tensor([.1, -.2, .3, -.4], dtype=dtype).expand_as(host).contiguous()
            expected.backward(seed)
            actual.backward(seed.to('intel_npu'))
            close(actual, expected)
            close(x.grad, host.grad, 'backward')
            comparisons += 2
        return {'comparisons': comparisons, 'mixed_fp32_affine': mixed,
                'rank_1_to_4_dense_permutation_affine_options_singleton_and_tiled': True}

    def gelu(dtype, mode):
        comparisons = 0
        samples = [torch.linspace(-10, 10, 4097).to(dtype),
                   torch.tensor([-float('inf'), -1e20, -65504., -6., -0., 0., 6., 65504., 1e20,
                                 float('inf'), float('nan')], dtype=dtype),
                   torch.randn(2, 3, 5, 7, dtype=dtype).permute(0, 1, 3, 2),
                   torch.randn(65537, dtype=dtype)]
        if dtype == torch.float16:
            samples.append(torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16))
        else:
            samples.append(torch.randint(-2**31, 2**31, (8192,), dtype=torch.int64).to(torch.int32).view(torch.float32))
        samples.append(torch.tensor(-.3, dtype=dtype))
        for sample in samples:
            host = sample.detach().requires_grad_()
            x = sample.to('intel_npu', memory_format=torch.contiguous_format).requires_grad_()
            if not sample.is_contiguous():
                x = x.detach().permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2).requires_grad_()
            expected = torch.nn.functional.gelu(host, approximate=mode)
            actual = torch.nn.functional.gelu(x, approximate=mode)
            seed = torch.full_like(host, .75).contiguous()
            expected.backward(seed)
            actual.backward(seed.to('intel_npu'))
            close(actual, expected)
            close(x.grad, host.grad, 'backward')
            with torch.no_grad():
                out = torch.empty_like(x, memory_format=torch.contiguous_format)
                assert torch.ops.aten.gelu.out(x, approximate=mode, out=out) is out
                close(out, expected)
                assert torch.ops.aten.gelu_backward.grad_input(seed.to('intel_npu'), x, approximate=mode, grad_input=out) is out
                close(out, host.grad, 'backward')
            comparisons += 4
        return {'comparisons': comparisons, 'all_half_bit_patterns': dtype == torch.float16,
                'mode': mode, 'rank_1_to_4_dense_special_values_and_tile_boundary': True}

    def contracts():
        x = torch.ones(2, 4, device='intel_npu')
        w = torch.ones(4, device='intel_npu')
        half = torch.ones(4, dtype=torch.float16, device='intel_npu')
        bad = [lambda: npu._layer_norm(x, (), None, None),
               lambda: npu._layer_norm(x, (3,), None, None),
               lambda: npu._layer_norm(x, (4,), w, None, 0.),
               lambda: npu._layer_norm(x, (4,), w, None, float('nan')),
               lambda: npu._layer_norm(x, (4,), w, None, 1e-50),
               lambda: npu._layer_norm(x, (4,), half, None),
               lambda: npu._layer_norm(x, (4,), torch.ones(4), None),
               lambda: npu._layer_norm(x[:, 1:], (3,)),
               lambda: npu._gelu(x, approximate='other'),
               lambda: npu._gelu(x, out=x),
               lambda: npu._gelu(x, out=w),
               lambda: npu._gelu_backward(w, x),
               lambda: npu._gelu_backward(x.to(torch.float16), x)]
        # Allocate the mismatched gradient before counting submits.
        mismatched = x.to(torch.float16)
        bad[-1] = lambda: npu._gelu_backward(mismatched, x)
        rejected = 0
        for function in bad:
            before = npu.stats()['npu_calls']
            try:
                function()
            except (ValueError, NotImplementedError):
                rejected += 1
            else:
                raise AssertionError('Invalid operator request accepted')
            assert npu.stats()['npu_calls'] == before
        for operation in (lambda v: torch.nn.functional.layer_norm(v, (4,)), torch.nn.functional.gelu):
            v = torch.randn(2, 4).to('intel_npu').requires_grad_()
            y = operation(v)
            with torch.no_grad():
                v.add_(1.)
            try:
                y.sum().backward()
            except RuntimeError as error:
                assert 'modified by an inplace operation' in str(error)
            else:
                raise AssertionError('Saved tensor version guard missing')
        return {'rejected_without_submit': rejected, 'saved_tensor_version_guards': 2}

    class FeedForward(torch.nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.norm = torch.nn.LayerNorm(8)
            self.up = torch.nn.Linear(8, 16)
            self.activation = torch.nn.GELU(approximate=mode)
            self.down = torch.nn.Linear(16, 8)
            self.head = torch.nn.Linear(8, 3)

        def forward(self, x):
            return self.head(x+self.down(self.activation(self.up(self.norm(x)))))

    def train(dtype, mode, optimizer):
        host = FeedForward(mode).to(dtype)
        model = copy.deepcopy(host).to('intel_npu')
        options = {'lr': .01, 'momentum': .8} if optimizer == 'sgd' else {'lr': .001}
        cls = torch.optim.SGD if optimizer == 'sgd' else torch.optim.AdamW
        reference, actual = (cls(v.parameters(), foreach=False, fused=False, **options) for v in (host, model))
        data, labels = torch.randn(4, 8, dtype=dtype), torch.tensor([0, 1, 2, 1])
        x, target = data.to('intel_npu'), labels.to('intel_npu')
        losses = []
        for step in range(4):
            reference.zero_grad(set_to_none=True)
            actual.zero_grad(set_to_none=True)
            expected = torch.nn.functional.cross_entropy(host(data), labels)
            result = torch.nn.functional.cross_entropy(model(x), target)
            close(result, expected)
            expected.backward()
            result.backward()
            for a, e in zip(model.parameters(), host.parameters()):
                close(a.grad, e.grad, 'backward')
            reference.step()
            actual.step()
            for a, e in zip(model.parameters(), host.parameters()):
                close(a, e)
            losses.append(float(result.detach().cpu()))
        assert losses[-1] < losses[0]
        return {'dtype': str(dtype), 'optimizer': optimizer, 'mode': mode, 'steps': 4, 'losses': losses,
                'independent_cpu_trajectory': True}

    if case == 'layernorm':
        for dtype, mixed in ((torch.float32, False), (torch.float16, False), (torch.float16, True)):
            check(f'layernorm-{dtype}-mixed-{mixed}', lambda d=dtype, m=mixed: layernorm(d, m))
    elif case == 'gelu':
        for dtype in (torch.float16, torch.float32):
            for mode in ('none', 'tanh'):
                check(f'gelu-{dtype}-{mode}', lambda d=dtype, m=mode: gelu(d, m))
    elif case == 'contracts':
        check('contracts', contracts)
    else:
        for dtype, optimizer in ((torch.float16, 'sgd'), (torch.float32, 'sgd'), (torch.float32, 'adamw')):
            for mode in ('none', 'tanh'):
                check(f'feed-forward-{dtype}-{optimizer}-{mode}', lambda d=dtype, m=mode, o=optimizer: train(d, m, o))
    gc.collect()
    before_shutdown = npu.stats()
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert final['live_buffers'] == final['cached_buffer_blocks'] == final['quarantined_buffer_bytes'] == final['cached_sessions'] == 0
    return {'case': case, 'checks': checks, 'maximum_absolute_errors': maximum_error,
            'status': 'failed' if any(r['status'] != 'passed' for r in checks) else 'passed',
            'stats_before_shutdown': before_shutdown, 'final_stats': final,
            'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('torch_npu_norm_activation_demo.py', 'torch_intel_npu.py', 'torch_npu_autograd.py',
                 'npunlock/tests/fixtures/npu4000/resident_gelu.c')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES+('all',), default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.case != 'all':
        result = execute(args.case)
        (OUT/f'{args.case}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return int(result['status'] != 'passed')
    status = 0
    for case in CASES:
        process = subprocess.run([sys.executable, __file__, '--case', case], capture_output=True,
                                 text=True, encoding='utf-8', timeout=900)
        (OUT/f'{case}.log').write_text(process.stdout+process.stderr, encoding='utf-8')
        print(process.stdout+process.stderr, flush=True)
        status |= int(process.returncode != 0)
    return status


if __name__ == '__main__':
    raise SystemExit(main())

"""Bounded standard softmax/bmm, attention gradients and tiny-network training."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/attention-primitives'
CASES = ('softmax', 'bmm', 'bmm-large', 'contracts', 'attention', 'training')


def execute(case):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(83)
    torch.backends.mkldnn.enabled = False
    checks = []

    def close(a, b):
        a, b = a.detach().cpu(), b.detach().cpu()
        torch.testing.assert_close(a, b, rtol=.003 if a.dtype == torch.float16 else 3e-5,
                                   atol=.0005 if a.dtype == torch.float16 else 3e-6, equal_nan=True)

    def upload(x):
        return x.detach().to('intel_npu', memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)

    def copies_since(before, error_flags=0):
        after = npu.stats()
        copies = sum(after[k]-before[k] for k in ('host_upload_copies', 'host_download_copies', 'host_device_copies'))
        scalars = after['cached_scalar_buffers']-before['cached_scalar_buffers']
        assert copies == scalars+error_flags, (copies, scalars, error_flags)
        assert after['npu_calls'] > before['npu_calls']
        return {'copies': copies, 'scalar_uploads': scalars, 'error_flag_reads': error_flags}

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
        print(name, row['status'], row.get('error', '')[:400], flush=True)

    def softmax(dtype):
        comparisons, copy_checks = 0, []
        for shape in ((1,), (3,), (2, 3), (3, 17), (2, 3, 5), (1, 2, 3, 7), (2, 257), (1, 4096), (1, 65536)):
            host = (torch.randn(shape)*4).to(dtype).requires_grad_()
            x, seed = upload(host), torch.randn(shape, dtype=dtype)/4
            g = seed.to('intel_npu')
            expected = torch.softmax(host, -1)
            before = npu.stats()
            actual = torch.nn.functional.softmax(x, dim=-1)
            actual.backward(g)
            copy_checks.append(copies_since(before))
            expected.backward(seed)
            close(actual, expected)
            close(x.grad, host.grad)
            with torch.no_grad():
                out, grad_out = torch.empty_like(x), torch.empty_like(x)
                assert torch.ops.aten._softmax.out(x, x.ndim-1, False, out=out) is out
                assert torch.ops.aten._softmax_backward_data.out(g, out, -1, dtype, grad_input=grad_out) is grad_out
                close(out, expected)
                close(grad_out, host.grad)
            comparisons += 4
        for values in ([[10000., 10000.125, 9999.875]], [[0., -80., -90., -100., -103., -104.]],
                       [[float('inf'), 0., -1.]], [[-float('inf')]*3], [[float('nan'), 0., 1.]],
                       [[-float('inf'), 0., -1.]]):
            host = torch.tensor(values, dtype=dtype).requires_grad_()
            x = upload(host)
            expected, actual = torch.softmax(host, -1), torch.nn.Softmax(-1)(x)
            seed = torch.tensor([[.1, -.3, .2, -.4, .5, -.6][:host.shape[-1]]], dtype=dtype)
            expected.backward(seed)
            actual.backward(seed.to('intel_npu'))
            close(actual, expected)
            close(x.grad, host.grad)
            if dtype == torch.float32 and host.shape[-1] == 6:
                tail = actual.detach().cpu()[0, 2:5]
                assert bool(((tail > 0) & (tail < torch.finfo(torch.float32).tiny)).all())
            comparisons += 2
        host = torch.randn(3, 4, dtype=dtype).t().requires_grad_()
        x = upload(host.t()).t().detach().requires_grad_()
        expected, actual = torch.softmax(host, -1, dtype=torch.float32), torch.softmax(x, -1, dtype=torch.float32)
        seed = torch.randn_like(expected)
        expected.backward(seed)
        actual.backward(seed.to('intel_npu'))
        close(actual, expected)
        close(x.grad, host.grad)
        probabilities = torch.tensor([[.2, .3, .5]], dtype=dtype)
        for seed in (torch.tensor([[float('inf'), 0., 1.]], dtype=dtype),
                     torch.tensor([[float('nan'), 0., 1.]], dtype=dtype)):
            close(torch.ops.aten._softmax_backward_data(seed.to('intel_npu'), probabilities.to('intel_npu'), -1, dtype),
                  torch.ops.aten._softmax_backward_data(seed, probabilities, -1, dtype))
        return {'comparisons': comparisons+4, 'compute_copies': copy_checks, 'explicit_fp32_dense_transpose': True,
                'special_values_and_subnormals': True}

    def bmm(dtype, large=False):
        specs = ((3, 16, 64, 32), (2, 16, 257, 17), (5, 128, 1, 128), (65537, 1, 1, 1), (1, 1, 65537, 1)) if large else (
            (1, 2, 3, 4), (3, 2, 5, 4), (2, 3, 1, 5))
        copies = []
        for batch, m, k, n in specs:
            a, b = (torch.randn(shape, dtype=dtype)/4 for shape in ((batch, m, k), (batch, k, n)))
            # Small cases also exercise one-sided gradients and complete dense permutations.
            for need_a, need_b in (((True, True),) if large else ((True, True), (True, False), (False, True))):
                left, right = a.detach().requires_grad_(need_a), b.detach().requires_grad_(need_b)
                x, y = upload(left), upload(right)
                if not large:
                    x = x.detach().transpose(1, 2).contiguous().transpose(1, 2).requires_grad_(need_a)
                    y = y.detach().transpose(1, 2).contiguous().transpose(1, 2).requires_grad_(need_b)
                seed = torch.randn(batch, m, n, dtype=dtype)/16
                g = seed.to('intel_npu')
                expected = torch.bmm(left, right)
                before = npu.stats()
                actual = torch.bmm(x, y)
                actual.backward(g)
                copies.append(copies_since(before))
                expected.backward(seed)
                close(actual, expected)
                if need_a:
                    close(x.grad, left.grad)
                if need_b:
                    close(y.grad, right.grad)
                with torch.no_grad():
                    out = torch.empty_like(actual)
                    assert torch.bmm(x, y, out=out) is out
                    close(out, expected)
        return {'shapes_bmkn': specs, 'compute_copies': copies, 'forward_out_backward': True}

    def contracts():
        x, y = torch.ones(2, 3, 4).to('intel_npu'), torch.ones(2, 4, 3).to('intel_npu')
        oversized_a = torch.ones(1, 1025, 1).to('intel_npu')
        oversized_b = torch.ones(1, 1, 1024).to('intel_npu')
        big = torch.ones(1, 65537).to('intel_npu')
        wrong = torch.empty_like(x)
        half_y = y.half()
        square_y = torch.ones(2, 4, 4).to('intel_npu')
        before, rejected = npu.stats()['npu_calls'], []
        for name, function in {
            'softmax-axis': lambda: torch.softmax(x, 3),
            'softmax-half-to-float': lambda: torch.ops.aten._softmax(x, -1, True),
            'softmax-size': lambda: torch.softmax(big, -1),
            'softmax-alias': lambda: torch.softmax(x, -1, out=x),
            'softmax-grad-shape': lambda: torch.ops.aten._softmax_backward_data(y, x, -1, x.dtype),
            'bmm-rank': lambda: torch.bmm(x.view(6, 4), y),
            'bmm-batch': lambda: torch.bmm(x, y.view(1, 8, 3)),
            'bmm-dtype': lambda: torch.bmm(x, half_y),
            'bmm-inner': lambda: torch.bmm(x, x),
            'bmm-output-bound': lambda: torch.bmm(oversized_a, oversized_b),
            'bmm-out-shape': lambda: torch.bmm(x, y, out=wrong),
            'bmm-alias': lambda: torch.bmm(x, square_y, out=x),
        }.items():
            try:
                function()
            except (ValueError, RuntimeError, TypeError, NotImplementedError):
                rejected.append(name)
            else:
                raise AssertionError(f'Accepted {name}')
        assert npu.stats()['npu_calls'] == before
        for name in ('softmax', 'bmm'):
            a = x.detach().requires_grad_()
            output = torch.softmax(a, -1) if name == 'softmax' else torch.bmm(a, y)
            loss = output.sum()
            with torch.no_grad():
                (output if name == 'softmax' else a).add_(1)
            try:
                loss.backward()
            except RuntimeError as error:
                assert 'version' in str(error) or 'inplace' in str(error)
            else:
                raise AssertionError(f'{name} saved version bypassed')
        return {'rejected_without_submit': rejected, 'saved_version_guards': True}

    def attend(q, k, v):
        scores = torch.bmm(q, k.transpose(1, 2))*(q.shape[-1]**-.5)
        return torch.bmm(torch.softmax(scores, -1), v)

    def attention(dtype):
        values = [(torch.randn(2, 4, 3, dtype=dtype)/3).requires_grad_() for _ in range(3)]
        inputs = [upload(x) for x in values]
        seed = torch.randn(2, 4, 3, dtype=dtype)/4
        g = seed.to('intel_npu')
        expected = attend(*values)
        before = npu.stats()
        actual = attend(*inputs)
        actual.backward(g)
        copies = copies_since(before)
        expected.backward(seed)
        close(actual, expected)
        for x, ref in zip(inputs, values):
            close(x.grad, ref.grad)
        return {'forward_qkv_gradients': True, 'compute_copies': copies}

    def train(dtype):
        class TinyAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(4)
                self.q, self.k, self.v, self.proj = (torch.nn.Linear(4, 4) for _ in range(4))
                self.head = torch.nn.Linear(4, 3)

            def forward(self, x):
                h = self.norm(x)
                h = x+self.proj(attend(self.q(h), self.k(h), self.v(h)))
                return self.head(h.mean(1))

        cpu = TinyAttention().to(dtype)
        model = copy.deepcopy(cpu).to('intel_npu')
        host = torch.randn(2, 3, 4, dtype=dtype)/3
        target = torch.tensor([0, 2])
        data, labels = host.to('intel_npu'), target.to('intel_npu')
        reference, opt = (torch.optim.SGD(m.parameters(), lr=.025, foreach=False, fused=False) for m in (cpu, model))
        losses, copies = [], []
        for _ in range(4):
            reference.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            expected = cpu(host)
            cpu_loss = torch.nn.functional.cross_entropy(expected.float(), target)
            cpu_loss.backward()
            reference.step()
            before = npu.stats()
            actual = model(data)
            loss = torch.nn.functional.cross_entropy(actual.float(), labels)
            loss.backward()
            opt.step()
            copies.append(copies_since(before, error_flags=1))
            close(actual, expected)
            close(loss, cpu_loss)
            for a, b in zip(model.parameters(), cpu.parameters()):
                close(a.grad, b.grad)
                close(a, b)
            losses.append(float(loss.detach().cpu()))
        assert losses[-1] < losses[0], losses
        return {'losses': losses, 'steps': 4, 'all_parameters_and_gradients': True, 'step_copies': copies,
                'explicit_fp32_classification_loss': True}

    if case == 'contracts':
        check('contracts', contracts)
    else:
        for dtype in (torch.float16, torch.float32):
            check(f'{case}-{dtype}', lambda d=dtype: softmax(d) if case == 'softmax' else (
                bmm(d, case == 'bmm-large') if case.startswith('bmm') else attention(d) if case == 'attention' else train(d)))
    gc.collect()
    before = npu.stats()
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
    return {'case': case, 'checks': checks, 'status': 'failed' if any(c['status'] == 'failed' for c in checks) else 'passed',
            'stats_before_shutdown': before, 'final_stats': final, 'python': sys.version,
            'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                'torch_intel_npu.py', 'torch_npu_autograd.py', 'torch_npu_attention_demo.py',
                'npunlock/tests/fixtures/npu4000/resident_softmax.c', 'npunlock/tests/fixtures/npu4000/resident/manifest.json')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result = execute(args.case)
    (OUT/f'{args.case}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    raise SystemExit(result['status'] == 'failed')

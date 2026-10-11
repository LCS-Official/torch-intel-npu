"""Fresh-process compatibility regressions; run with run_torch_npu.ps1 -Script."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'npunlock-probe/research/results/compatibility'
CASES = ('division', 'dense', 'unary', 'reduction-cast', 'late-import')


def execute(case):
    import torch
    if case == 'late-import':
        torch.tensor(2., requires_grad=True).square().backward()
    import torch_intel_npu as npu
    report = {'case': case, 'status': 'running', 'checks': [], 'source_sha256': {
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('torch_intel_npu.py', 'torch_npu_autograd.py', 'npunlock-probe/torch-device/backend.cpp')}}

    def close(actual, expected, *, rtol=3e-6, atol=0.):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach(), rtol=rtol, atol=atol, equal_nan=True)

    def check(name, function):
        before = npu.stats()
        try:
            details = function()
            report['checks'].append({'name': name, 'status': 'passed', **(details or {})})
        except Exception as error:
            report['checks'].append({'name': name, 'status': 'failed', 'error': str(error),
                                     'traceback': traceback.format_exc()})
        gc.collect()
        after = npu.stats()
        assert not after['context_failed'], 'NPU context failed'
        assert after['live_buffers']-after['cached_scalar_buffers'] == before['live_buffers']-before['cached_scalar_buffers'], name
        print(name, report['checks'][-1]['status'], flush=True)

    def division():
        x = torch.tensor([1e20, 1e-20, 3e19, 3e-19, 1., -2.], requires_grad=True)
        y = torch.tensor([1e20, 1e-20, -2e19, -2e-19, 0., float('inf')], requires_grad=True)
        g = torch.tensor([1., 1., .25, .25, 0., 1.])
        a, b = (t.detach().to('intel_npu').requires_grad_() for t in (x, y))
        (x/y).backward(g)
        (a/b).backward(g.to('intel_npu'))
        close(a.grad, x.grad)
        close(b.grad, y.grad)
        return {'large_and_small_finite_denominators': True, 'zero_inf_nan_semantics': True}

    def dense():
        for dtype in (torch.float16, torch.float32):
            x = (torch.arange(6, dtype=dtype).view(2, 3)/8+.25).t().detach().requires_grad_()
            y = torch.tensor([[.5, 1.], [1.5, 2.], [2.5, 3.]], dtype=dtype, requires_grad=True)
            a = x.detach().t().contiguous().to('intel_npu').t().detach().requires_grad_()
            b = y.detach().to('intel_npu').requires_grad_()
            grad = torch.arange(6, dtype=dtype).view(2, 3).t()/8+.125
            expected, actual = (x*y+x), (a*b+a)
            expected.backward(grad)
            actual.backward(grad.t().contiguous().to('intel_npu').t())
            close(actual, expected, atol=.002 if dtype == torch.float16 else 0.)
            close(a.grad, x.grad, atol=.002 if dtype == torch.float16 else 0.)
            assert a.grad.stride() == a.stride(), 'Leaf gradient layout was not preserved'
            # A second backward accumulates into the existing dense gradient.
            (x*y+x).backward(grad)
            (a*b+a).backward(grad.t().contiguous().to('intel_npu').t())
            close(a.grad, x.grad, atol=.002 if dtype == torch.float16 else 0.)
            cpu_opt = torch.optim.SGD([x], lr=.125, foreach=False, fused=False)
            npu_opt = torch.optim.SGD([a], lr=.125, foreach=False, fused=False)
            cpu_opt.step(); npu_opt.step()
            close(a, x, atol=.002 if dtype == torch.float16 else 0.)
            with torch.no_grad():
                target = torch.empty_strided(a.shape, a.stride(), device='intel_npu', dtype=dtype)
                target.copy_(a)
                close(target, x)
                clone = target.clone()
                assert clone.stride() == target.stride() and clone.data_ptr() != target.data_ptr()
                close(clone, x)
                # Copy through dense destinations also supports a host source
                # and precision changes without host activation arithmetic.
                target.copy_(x.detach().float() if dtype == torch.float16 else x.detach().half())
                close(target, x, atol=.002 if dtype == torch.float16 else 0.)
                close(target.sum(dim=0), x.sum(dim=0), atol=.002 if dtype == torch.float16 else 0.)
                target.zero_()
                close(target, torch.zeros_like(x))
            del a, b, actual, target, clone, npu_opt
        with torch.no_grad():
            square = torch.arange(9, dtype=torch.float32).view(3, 3).to('intel_npu')
            square.t().copy_(square)
            close(square, torch.arange(9, dtype=torch.float32).view(3, 3).t())
        return {'precisions': ['float16', 'float32'], 'dense_leaf_accumulation_and_sgd': True}

    def unary():
        for dtype in (torch.float16, torch.float32):
            for function in (torch.neg, torch.abs, torch.sign, torch.reciprocal):
                x = torch.tensor([-2., -.5, .25, 3., .001, 0., float('inf'), float('nan')], dtype=dtype, requires_grad=True)
                a = x.detach().to('intel_npu').requires_grad_()
                g = torch.tensor([.25, 1., -.5, 2., .01, 0., 1., 1.], dtype=dtype)
                expected, actual = function(x), function(a)
                expected.backward(g); actual.backward(g.to('intel_npu'))
                close(actual, expected, rtol=.001 if dtype == torch.float16 else 3e-6)
                close(a.grad, x.grad, rtol=.001 if dtype == torch.float16 else 3e-6)
        return {'unary_first_order_cases': 8}

    def reduction_cast():
        for kind in ('sum', 'mean'):
            for incoming, outgoing in ((torch.float32, torch.float16), (torch.float16, torch.float32)):
                x = torch.tensor([1.125, -.75, .375], dtype=incoming, requires_grad=True)
                a = x.detach().to('intel_npu').requires_grad_()
                expected, actual = getattr(x, kind)(dtype=outgoing), getattr(a, kind)(dtype=outgoing)
                g = torch.tensor(1.1, dtype=outgoing)
                expected.backward(g); actual.backward(g.to('intel_npu'))
                close(actual, expected, rtol=0.)
                close(a.grad, x.grad, rtol=0.)
        return {'explicit_reduction_dtype_gradients': 4}

    def late_import():
        a = torch.tensor([1., 2.]).to('intel_npu').requires_grad_()
        ordinary_error = None
        try:
            (a*2.).sum().backward()
        except RuntimeError as error:
            ordinary_error = str(error)
            assert 'device_ready_queues_' in ordinary_error
        # Supported PyTorch context changes scheduling, not where tensor math executes.
        before = torch.autograd.is_multithreading_enabled()
        with torch.autograd.set_multithreading_enabled(False):
            (a*2.).sum().backward()
        assert torch.autograd.is_multithreading_enabled() == before
        close(a.grad, torch.tensor([2., 2.]))
        return {'ordinary_backward_after_late_registration': 'upstream_limitation' if ordinary_error else 'passed',
                'scoped_single_thread_backward': 'passed', 'scheduling_context_restored': True}

    try:
        check(case, {'division': division, 'dense': dense, 'unary': unary,
                     'reduction-cast': reduction_cast, 'late-import': late_import}[case])
        report['stats_before_shutdown'] = npu.stats()
        npu.shutdown()
        report['final_stats'] = npu.stats()
        assert report['final_stats']['live_buffers'] == 0
        report['status'] = 'passed' if all(c['status'] == 'passed' for c in report['checks']) else 'failed'
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc(), failure_stats=npu.stats())
    (OUT/f'{case}.json').write_text(json.dumps(report, indent=2))
    return int(report['status'] != 'passed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', choices=CASES)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.child:
        return execute(args.child)
    reports = []
    for case in CASES:
        completed = subprocess.run([sys.executable, __file__, '--child', case], capture_output=True, text=True, timeout=180)
        (OUT/f'{case}.log').write_text(completed.stdout+completed.stderr)
        print(case, completed.stdout+completed.stderr, flush=True)
        reports.append(json.loads((OUT/f'{case}.json').read_text()))
    summary = {'status': 'passed' if all(r['status'] == 'passed' for r in reports) else 'failed', 'cases': reports}
    (OUT/'compatibility.json').write_text(json.dumps(summary, indent=2))
    return int(summary['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())

"""Bounded hardware checks for large resident tensors and unchanged compute tiles."""
import argparse
import copy
import faulthandler
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

OUT = Path(__file__).resolve().parent / 'npunlock-probe/research/results'


def execute(case):
    import numpy as np
    import torch
    import torch_intel_npu as npu
    torch.backends.mkldnn.enabled = False
    torch.manual_seed(35)
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    report = {'status': 'running', 'case': case, 'checks': [],
              'openvino_installed': importlib.util.find_spec('openvino') is not None}
    started = time.perf_counter()

    def save():
        (OUT / f'torch-npu-large-{case}.json').write_text(json.dumps(report, indent=2))

    def check(name, function):
        report['stage'] = name
        save()
        print(name, flush=True)
        before = npu.stats()
        result = function()
        gc.collect()
        after = npu.stats()
        assert after['live_buffers']-after['cached_scalar_buffers'] == before['live_buffers']-before['cached_scalar_buffers']
        assert not after['context_failed']
        report['checks'].append({'name': name, 'status': 'passed', **result})

    def pointwise(dtype):
        comparisons = 0
        for shape in ((65537,), (257, 257), (1024, 1024)):
            host = ((torch.arange(int(np.prod(shape))) % 17 - 8)/8).to(dtype).reshape(shape)
            x = host.to('intel_npu')
            torch.testing.assert_close(((x+.25)*.5).cpu(), (host+.25)*.5, rtol=0, atol=0)
            out = torch.empty_like(x)
            torch.add(x, x, out=out)
            torch.testing.assert_close(out.cpu(), host+host, rtol=0, atol=0)
            torch.testing.assert_close(x.half().float().cpu(), host.float(), rtol=0, atol=0)
            torch.testing.assert_close(x.sum().cpu(), host.sum(), rtol=0, atol=0)
            torch.testing.assert_close(x.mean().cpu(), host.mean(), rtol=2e-6, atol=1e-8)
            out.fill_(.25)
            torch.testing.assert_close(out.cpu(), torch.full_like(host, .25), rtol=0, atol=0)
            comparisons += 6
            if len(shape) == 2:
                for axis in (0, 1):
                    torch.testing.assert_close(x.mean(axis).cpu(), host.mean(axis), rtol=2e-6, atol=1e-8)
                    comparisons += 1
        # Unequal final reduction chunks must be weighted by their element counts.
        host = torch.ones(65537, dtype=dtype)
        host[-1] = 7
        x = host.to('intel_npu')
        torch.testing.assert_close(x.mean().cpu(), host.mean(), rtol=0, atol=0)
        return {'comparisons': comparisons+1, 'max_elements': npu.MAX_TENSOR_ELEMENTS}

    def bit_transports():
        for dtype, integer, values in (
                (torch.float16, torch.int16, np.arange(131072, dtype=np.uint32).astype(np.uint16).view(np.float16)),
                (torch.float32, torch.int32, np.random.default_rng(35).integers(0, 2**32, 131072,
                                                                            dtype=np.uint32).view(np.float32))):
            host = torch.from_numpy(values).reshape(256, 512)
            x = host.to('intel_npu')
            torch.testing.assert_close(x.clone().cpu().view(integer), host.view(integer), rtol=0, atol=0)
            torch.testing.assert_close(x.t().contiguous().cpu().view(integer),
                                       host.t().contiguous().view(integer), rtol=0, atol=0)
        return {'comparisons': 4, 'elements_per_tensor': 131072, 'includes_nan_inf_subnormals': True}

    def broadcast_autograd():
        host = torch.linspace(-.5, .5, 257).view(257, 1).requires_grad_()
        bias = torch.linspace(.1, .4, 257).view(1, 257).requires_grad_()
        x, y = (v.detach().to('intel_npu').requires_grad_() for v in (host, bias))
        expected = (host+bias).half().float().square().mean()
        converted = (x+y).half().float()
        actual = (converted*converted).mean()
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual.detach().cpu(), expected.detach(), rtol=2e-6, atol=1e-8)
        for device, reference in ((x, host), (y, bias)):
            torch.testing.assert_close(device.grad.cpu(), reference.grad, rtol=7e-5, atol=6e-6)
        return {'output_elements': 66049, 'differentiable_casts': True, 'broadcast_gradients': True}

    def contracts():
        x = torch.empty(65537, device='intel_npu')
        before = npu.stats()['npu_calls']
        actions = {'allocation_bound': lambda: torch.empty(npu._storage_limit(torch.float32)+1, device='intel_npu'),
                   'alias_out': lambda: torch.neg(x, out=x)}
        for name, action in actions.items():
            try:
                action()
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError(f'Accepted {name}')
        assert npu.stats()['npu_calls'] == before
        return {'rejected': list(actions)}

    def training(dtype, large_weights=False, conv=False):
        if conv:
            cpu = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1),
                torch.nn.BatchNorm2d(8, track_running_stats=False), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 4))
            shape = (2, 3, 65, 65)
        else:
            cpu = torch.nn.Linear(256, 512, dtype=dtype) if large_weights else torch.nn.Linear(16, 16, dtype=dtype)
            shape = (2, 256) if large_weights else (4097, 16)
        model = copy.deepcopy(cpu).to('intel_npu')
        host = (torch.randn(shape, dtype=dtype)/3).requires_grad_()
        x = host.detach().to('intel_npu').requires_grad_()
        target = torch.zeros_like(cpu(host).detach())
        labels = target.to('intel_npu')
        optimizer = torch.optim.SGD(model.parameters(), lr=.05, foreach=False)
        reference_optimizer = torch.optim.SGD(cpu.parameters(), lr=.05, foreach=False)
        tolerance = dict(rtol=4e-3, atol=5e-4) if dtype == torch.float16 else dict(rtol=7e-5, atol=6e-6)
        rows = []
        for step in range(3):
            with torch.no_grad():
                for p, q in zip(model.parameters(), cpu.parameters()):
                    q.copy_(p.detach().cpu())
            host.grad = x.grad = None
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected = (cpu(host).float()-target.float()).square().mean().to(dtype)
            expected.backward()
            reference_optimizer.step()
            before = npu.stats()
            step_start = time.perf_counter()
            loss = torch.nn.functional.mse_loss(model(x), labels)
            loss.backward()
            optimizer.step()
            after = npu.stats()
            duration = time.perf_counter()-step_start
            differences, references = [], []
            for p, q in zip(model.parameters(), cpu.parameters()):
                assert p.grad.device.type == 'intel_npu' and p.grad.dtype == p.dtype
                grad = p.grad.cpu()
                torch.testing.assert_close(grad, q.grad, **tolerance)
                torch.testing.assert_close(p.detach().cpu(), q.detach(), **tolerance)
                differences.append((grad.float()-q.grad.float()).flatten())
                references.append(q.grad.float().flatten())
            torch.testing.assert_close(x.grad.cpu(), host.grad, **tolerance)
            value = float(loss.detach().cpu())
            torch.testing.assert_close(torch.tensor(value, dtype=dtype), expected.detach(), **tolerance)
            row = {'step': step, 'loss': value, 'seconds': duration,
                   'gradient_relative_l2': float(torch.cat(differences).norm()/torch.cat(references).norm().clamp_min(1e-30)),
                   'host_payload_copies': sum(after[k]-before[k] for k in ('host_upload_copies', 'host_download_copies', 'host_device_copies')),
                   'graph_compilations': after['session_creations']-before['session_creations'],
                   'npu_calls': after['npu_calls']-before['npu_calls']}
            if step:
                assert row['host_payload_copies'] == row['graph_compilations'] == 0
            rows.append(row)
            print(row, flush=True)
        assert rows[-1]['loss'] < rows[0]['loss'], rows
        return {'steps': rows, 'input_shape': shape, 'max_parameter_elements': max(p.numel() for p in model.parameters()),
                'cpu_reference_rebased_each_step': True, 'gradient_update_tolerance': tolerance,
                'convolution_output_elements': 67600 if conv else None,
                'convolution_window_elements': 228150 if conv else None}

    try:
        if case == 'pointwise':
            for dtype in (torch.float16, torch.float32):
                check(f'{dtype}_million_elements', lambda d=dtype: pointwise(d))
            check('large_bit_transports', bit_transports)
            check('large_broadcast_cast_autograd', broadcast_autograd)
            check('large_contracts', contracts)
        elif case == 'training':
            for dtype in (torch.float16, torch.float32):
                for weights in (True, False):
                    check(f'{dtype}_large_{"weights" if weights else "activations"}',
                          lambda d=dtype, w=weights: training(d, w))
        else:
            check('large_conv_batchnorm_training', lambda: training(torch.float32, conv=True))
        report['stats_before_shutdown'] = npu.stats()
        npu.shutdown()
        assert npu.stats()['live_buffers'] == npu.stats()['live_buffer_bytes'] == 0
        report.update(status='passed', seconds=time.perf_counter()-started, final_stats=npu.stats())
        faulthandler.cancel_dump_traceback_later()
        save()
        return 0
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc(), final_stats=npu.stats())
        save()
        print(report['traceback'], flush=True)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--case', choices=('all', 'pointwise', 'training', 'conv'), default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.child:
        raise SystemExit(execute(args.case))
    cases = ('pointwise', 'training', 'conv') if args.case == 'all' else (args.case,)
    reports = []
    summary = OUT / 'torch-npu-large.json'
    summary.write_text(json.dumps({'status': 'running', 'cases': reports}))
    for case in cases:
        try:
            result = subprocess.run([sys.executable, __file__, '--child', '--case', case],
                capture_output=True, text=True, timeout=600, env=os.environ.copy())
            (OUT / f'torch-npu-large-{case}.log').write_text(result.stdout+result.stderr)
            print(result.stdout+result.stderr, flush=True)
            reports.append(json.loads((OUT / f'torch-npu-large-{case}.json').read_text()))
            summary.write_text(json.dumps({'status': 'failed' if result.returncode else 'running',
                                          'cases': reports}, indent=2))
            if result.returncode:
                raise SystemExit(result.returncode)
        except subprocess.TimeoutExpired:
            (OUT / f'torch-npu-large-{case}.json').write_text(json.dumps({'status': 'timeout'}))
            summary.write_text(json.dumps({'status': 'timeout', 'case': case, 'cases': reports}, indent=2))
            raise SystemExit(1)
    summary.write_text(json.dumps({'status': 'passed', 'cases': reports}, indent=2))

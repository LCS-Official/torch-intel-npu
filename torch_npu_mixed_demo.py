"""FP16 convolution weight storage + FP32 activations/compute, all on device."""
import argparse
import copy
import faulthandler
import gc
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from torch_npu_training_demo import make_small_resnet, OUT

REPORT = OUT / 'torch-npu-mixed.json'


def execute(quick):
    import torch
    import torch_intel_npu as npu
    torch.backends.mkldnn.enabled = False
    torch.manual_seed(21)
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    started = time.perf_counter()
    report = {'status': 'running', 'checks': [], 'openvino_installed': importlib.util.find_spec('openvino') is not None}

    def save(stage):
        report['stage'] = stage
        REPORT.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)

    def check(name, function):
        save(name)
        before = npu.stats()
        result = function()
        gc.collect()
        after = npu.stats()
        assert after['live_buffers']-after['cached_scalar_buffers'] == before['live_buffers']-before['cached_scalar_buffers']
        assert not after['context_failed']
        report['checks'].append({'name': name, 'status': 'passed' if result.get('pointwise_cpu_checks_passed', True)
                                 else 'measured_with_limitations', **result})

    def casts():
        comparisons = 0
        grad_fns = set()
        for source, destination in ((torch.float16, torch.float32), (torch.float32, torch.float16)):
            for transpose in (False, True):
                host = torch.tensor([[.5, -.25, 1.], [2., .125, -3.]], dtype=source, requires_grad=True)
                data = host.detach().to('intel_npu').requires_grad_()
                x, y = (host.t(), data.t()) if transpose else (host, data)
                reference, actual = x.to(destination), y.to(destination)
                assert actual.device.type == 'intel_npu' and actual.requires_grad
                grad_fns.add(type(actual.grad_fn).__name__)
                seed = torch.linspace(-.7, .9, host.numel()).reshape(reference.shape).to(destination)
                actual.backward(seed.to('intel_npu'))
                reference.backward(seed)
                torch.testing.assert_close(actual.detach().cpu(), reference.detach(), rtol=0, atol=0)
                torch.testing.assert_close(data.grad.cpu(), host.grad, rtol=0, atol=0)
                comparisons += 2
                assert data.grad.dtype == source
        host = torch.tensor([.1, -.2, .3], requires_grad=True)
        data = host.detach().to('intel_npu').requires_grad_()
        expected = host.half().float()+host
        actual = data.half().float()+data
        expected.sum().backward()
        actual.sum().backward(retain_graph=True)  # Warm the implicit scalar gradient.
        data.grad = None
        before = npu.stats()
        actual.sum().backward()
        after = npu.stats()
        assert all(after[k] == before[k] for k in ('host_upload_copies', 'host_download_copies', 'host_device_copies'))
        torch.testing.assert_close(data.grad.cpu(), host.grad, rtol=0, atol=0)
        comparisons += 1
        return {'comparisons': comparisons, 'native_autograd_functions': sorted(grad_fns),
                'cast_chain_and_branch_gradient_accumulation': True}

    def contracts():
        data = torch.ones(2).to('intel_npu').requires_grad_()
        actions = {'active_download': lambda: data.cpu(),
                   'active_upload': lambda: torch.ones(2, requires_grad=True).to('intel_npu'),
                   'bfloat16_cast': lambda: data.bfloat16(),
                   'float64_cast': lambda: data.double(),
                   'async_cast': lambda: data.to(dtype=torch.float16, non_blocking=True),
                   'cast_out': lambda: torch.ops.aten._to_copy.out(data,
                       out=torch.empty(2, dtype=torch.float16, device='intel_npu')),
                   'raw_cast_grad_out': lambda: npu._to_copy(data.detach(), out=data)}
        before = npu.stats()['npu_calls']
        for name, action in actions.items():
            try:
                action()
            except (ValueError, RuntimeError, NotImplementedError):
                pass
            else:
                raise AssertionError(f'Accepted {name}')
        assert npu.stats()['npu_calls'] == before
        return {'rejected': list(actions)}

    def train(mode, saved, steps, scenario):
        dtype = torch.float32 if mode == 'fp32' else torch.float16
        cpu = make_small_resnet(dtype, mixed_precision=mode == 'mixed')
        cpu.load_state_dict(saved['state'])
        model = copy.deepcopy(cpu).to('intel_npu')
        host = saved['input'].to(dtype)
        data = host.to('intel_npu')
        output_dtype = torch.float16 if mode == 'fp16' else torch.float32
        # All policies see the same numeric target, including FP16's representation of .2.
        target = torch.full((2, 4), float(torch.tensor(.2, dtype=torch.float16)), dtype=output_dtype)
        labels = target.to('intel_npu')
        optimizer = torch.optim.SGD(model.parameters(), lr=.05, foreach=False)
        reference_optimizer = torch.optim.SGD(cpu.parameters(), lr=.05, foreach=False)
        tolerance = {'rtol': 7e-5, 'atol': 6e-6} if mode == 'fp32' else {'rtol': 4e-3, 'atol': 5e-4}
        rows = []
        for step in range(steps):
            # Verification copies only. Every CPU step starts from exactly the NPU weights.
            with torch.no_grad():
                for p, q in zip(model.parameters(), cpu.parameters()):
                    q.copy_(p.detach().cpu())
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            reference = cpu(host)
            expected_loss = (reference.float()-target.float()).square().mean().to(output_dtype)
            expected_loss.backward()
            reference_optimizer.step()
            before = npu.stats()
            step_start = time.perf_counter()
            actual = model(data)
            assert actual.dtype == output_dtype
            loss = torch.nn.functional.mse_loss(actual, labels)
            loss.backward()
            assert all(p.grad.device.type == 'intel_npu' and p.grad.dtype == p.dtype for p in model.parameters())
            optimizer.step()
            duration = time.perf_counter()-step_start
            after = npu.stats()
            differences, references, actual_gradients = [], [], []
            max_absolute = 0.
            mismatches = []
            for (name, p), q in zip(model.named_parameters(), cpu.parameters()):
                gradient = p.grad.cpu()
                assert torch.isfinite(gradient).all(), name
                try:
                    torch.testing.assert_close(gradient, q.grad, **tolerance)
                    torch.testing.assert_close(p.detach().cpu(), q.detach(), **tolerance)
                except AssertionError as error:
                    # FP16 is a diagnostic baseline, not a claimed correctness pass.
                    if mode != 'fp16':
                        error.add_note(f'{scenario} {mode} step={step}, parameter={name}')
                        raise
                    mismatches.append(name)
                difference = gradient.float()-q.grad.float()
                max_absolute = max(max_absolute, float(difference.abs().max()))
                differences.append(difference.flatten())
                references.append(q.grad.float().flatten())
                actual_gradients.append(gradient.float().flatten())
            difference, reference_gradient, actual_gradient = map(torch.cat, (differences, references, actual_gradients))
            value = float(loss.detach().cpu())
            torch.testing.assert_close(torch.tensor(value, dtype=output_dtype), expected_loss.detach(), **tolerance)
            row = {'step': step, 'loss': value, 'gradient_relative_l2': float(difference.norm()/reference_gradient.norm().clamp_min(1e-30)),
                   'gradient_cosine': float(torch.nn.functional.cosine_similarity(actual_gradient, reference_gradient, dim=0)),
                   'gradient_max_absolute': max_absolute, 'seconds': duration,
                   'cpu_pointwise_mismatching_parameters': mismatches,
                   'npu_calls': after['npu_calls']-before['npu_calls'],
                   'host_payload_copies': sum(after[k]-before[k] for k in ('host_upload_copies', 'host_download_copies', 'host_device_copies')),
                   'graph_compilations': after['session_creations']-before['session_creations']}
            if step:
                assert row['host_payload_copies'] == row['graph_compilations'] == 0, row
            rows.append(row)
            print(scenario, mode, row, flush=True)
        assert rows[-1]['loss'] < rows[0]['loss'], rows
        if mode == 'mixed':
            assert max(r['gradient_relative_l2'] for r in rows) <= .01, rows
        counts = {}
        for p in cpu.parameters():
            counts[str(p.dtype)] = counts.get(str(p.dtype), 0)+p.numel()
        return {'mode': mode, 'scenario': scenario, 'steps': rows, 'parameter_elements_by_dtype': counts,
                'pointwise_cpu_checks_passed': not any(r['cpu_pointwise_mismatching_parameters'] for r in rows),
                'cpu_reference_rebased_each_step': True, 'gradient_update_tolerance': tolerance,
                'gradient_relative_l2_max': max(r['gradient_relative_l2'] for r in rows),
                'warm_step_median_seconds': statistics.median(r['seconds'] for r in rows[1:]),
                'convolution_compute_dtype': str(torch.float32 if mode != 'fp16' else torch.float16),
                'relative_gradient_error_limit': .01 if mode == 'mixed' else None,
                'autocast': False, 'loss_scaling': False, 'fp32_master_weights': False}

    try:
        check('differentiable_dtype_conversions', casts)
        check('conversion_contracts', contracts)
        initial = make_small_resnet(torch.float16)
        saved = {'state': {k: v.detach().clone() for k, v in initial.state_dict().items()},
                 'input': torch.randn(2, 3, 8, 8, dtype=torch.float16)/3}
        for mode in ('fp16', 'mixed', 'fp32'):
            check(f'{mode}_training', lambda m=mode: train(m, saved, 3 if quick else 15, 'initial'))
        late = OUT / 'fp16-residual-late.pt'
        if late.exists():
            saved = torch.load(late, weights_only=True)
            for mode in ('fp16', 'mixed', 'fp32'):
                check(f'{mode}_near_convergence', lambda m=mode: train(m, saved, 3 if quick else 5, 'near_convergence'))
        report['stats_before_shutdown'] = npu.stats()
        npu.shutdown()
        assert npu.stats()['live_buffers'] == npu.stats()['live_buffer_bytes'] == 0
        report.update(status='passed_with_limitations' if any(c['status'] != 'passed' for c in report['checks'])
                      else 'passed', seconds=time.perf_counter()-started, final_stats=npu.stats())
        faulthandler.cancel_dump_traceback_later()
        save('complete')
        return 0
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc(), final_stats=npu.stats())
        save(report['stage'])
        print(report['traceback'], flush=True)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.child:
        raise SystemExit(execute(args.quick))
    try:
        result = subprocess.run([sys.executable, __file__, '--child']+(['--quick'] if args.quick else []),
                                capture_output=True, text=True, timeout=600, env=os.environ.copy())
        REPORT.with_suffix('.log').write_text(result.stdout+result.stderr)
        print(result.stdout+result.stderr, flush=True)
        raise SystemExit(result.returncode)
    except subprocess.TimeoutExpired:
        REPORT.write_text(json.dumps({'status': 'timeout', 'error': 'Outer process deadline: 600 seconds'}, indent=2))
        raise SystemExit(1)

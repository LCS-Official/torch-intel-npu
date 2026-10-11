"""Bounded hardware checks for K tiling and streaming Conv2d training."""
import argparse
import copy
import faulthandler
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

OUT = Path(__file__).resolve().parent / 'npunlock-probe/research/results'
CASES = ('k-f16', 'k-f32', 'conv-f16', 'conv-f32', 'conv-geometry', 'conv-dense')


def execute(case):
    import torch
    import torch_intel_npu as npu
    torch.manual_seed(46)
    torch.backends.mkldnn.enabled = False
    dtype = torch.float16 if case.endswith('f16') else torch.float32
    tolerance = dict(rtol=4e-3, atol=5e-4) if dtype == torch.float16 else dict(rtol=7e-5, atol=6e-6)
    report = {'status': 'running', 'case': case, 'checks': [],
              'openvino_installed': importlib.util.find_spec('openvino') is not None,
              'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ('torch_intel_npu.py', 'torch_npu_autograd.py',
                                             'torch_npu_geometry.py', 'torch_npu_stream_demo.py')}}
    started = time.perf_counter()
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)

    def save():
        (OUT / f'torch-npu-stream-{case}.json').write_text(json.dumps(report, indent=2))

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
        save()

    def matrices():
        comparisons = 0
        for m, k, n in ((2, 65537, 3), (1, 1048576, 1)):
            left = (torch.randn(m, k)/32).to(dtype)
            right = (torch.randn(k, n)/32).to(dtype)
            x, y = left.to('intel_npu'), right.to('intel_npu')
            reference = (left.double() @ right.double()).to(dtype)
            torch.testing.assert_close((x@y).cpu(), reference, **tolerance)
            out = torch.empty((m, n), dtype=dtype, device='intel_npu')
            torch.mm(x, y, out=out)
            torch.testing.assert_close(out.cpu(), reference, **tolerance)
            comparisons += 2
        # Exactly representable products, cancellation, and an unequal K tail.
        left = torch.ones(1, 65537, dtype=dtype)
        right = torch.zeros(65537, 1, dtype=dtype)
        right[0], right[1], right[2], right[-1] = 1024, 1, -1024, 7
        x, y = left.to('intel_npu'), right.to('intel_npu')
        torch.testing.assert_close((x@y).cpu(), torch.tensor([[8]], dtype=dtype), rtol=0, atol=0)
        bias = torch.tensor([float('nan')], dtype=dtype).to('intel_npu')
        torch.testing.assert_close(torch.addmm(bias, x, y, beta=0).cpu(),
                                   torch.tensor([[8]], dtype=dtype), rtol=0, atol=0)
        return {'comparisons': comparisons+2, 'max_k': 1048576,
                'unequal_tail_and_cancellation': True, 'out_and_beta_zero': True}

    def sliced_transport():
        import numpy as np
        integer = torch.int16 if dtype == torch.float16 else torch.int32
        if dtype == torch.float16:
            values = np.arange(131072, dtype=np.uint32).astype(np.uint16).view(np.float16)
        else:
            values = np.random.default_rng(46).integers(0, 2**32, 131072, dtype=np.uint32).view(np.float32)
        host = torch.from_numpy(values)
        x = host.to('intel_npu')
        torch.testing.assert_close(npu._slice(x, 3, 65537).cpu().view(integer),
                                   host[3:65540].view(integer), rtol=0, atol=0)
        before = npu.stats()['npu_calls']
        for start, count in ((-1, 1), (0, 0), (131071, 2)):
            try:
                npu._slice(x, start, count)
            except ValueError:
                pass
            else:
                raise AssertionError('Accepted invalid slice range')
        assert npu.stats()['npu_calls'] == before
        return {'elements': 65537, 'bit_exact': True, 'includes_nan_inf_subnormals': True}

    def dense_conv():
        host_base = (torch.randn(1, 4, 176, 176)/3).requires_grad_()
        weight_base = (torch.randn(4, 4, 3, 3)/8).requires_grad_()
        host, weight_host = host_base.transpose(2, 3), weight_base.transpose(0, 1)
        gradient_host = torch.randn(1, 4, 176, 176).transpose(2, 3)
        x_base = host_base.detach().to('intel_npu').requires_grad_()
        w_base = weight_base.detach().to('intel_npu').requires_grad_()
        x, weight = x_base.transpose(2, 3), w_base.transpose(0, 1)
        gradient = gradient_host.transpose(2, 3).to('intel_npu').transpose(2, 3)
        assert not x.is_contiguous() and not weight.is_contiguous() and not gradient.is_contiguous()
        expected = torch.nn.functional.conv2d(host, weight_host, padding=1)
        output = torch.nn.functional.conv2d(x, weight, padding=1)
        expected.backward(gradient_host)
        output.backward(gradient)
        for actual, reference in ((output.detach(), expected.detach()), (x_base.grad, host_base.grad),
                                  (w_base.grad, weight_base.grad)):
            torch.testing.assert_close(actual.cpu(), reference, **tolerance)
        return {'comparisons': 3, 'dense_input_weight_upstream_gradient': True,
                'convolution_window_elements': 1115136}

    def train(conv=False):
        if conv:
            # Old full-window and input-gradient fold both exceeded 1M.
            if case == 'conv-geometry':
                cpu = torch.nn.Conv2d(2, 3, 3, stride=2, padding=2, dilation=2, bias=False, dtype=dtype)
                shape = (2, 2, 176, 176)
            else:
                cpu = torch.nn.Conv2d(4, 4, 3, padding=1, dtype=dtype)
                shape = (1, 4, 176, 176)
        else:
            cpu = torch.nn.Linear(2, 2, dtype=dtype)
            shape = (65537, 2)  # Weight-gradient MatMul reduces K=batch.
        model = copy.deepcopy(cpu).to('intel_npu')
        host = (torch.randn(shape, dtype=dtype)/3).requires_grad_()
        x = host.detach().to('intel_npu').requires_grad_()
        target = torch.zeros_like(cpu(host).detach())
        labels = target.to('intel_npu')
        optimizer = torch.optim.SGD(model.parameters(), lr=.05, foreach=False)
        reference_optimizer = torch.optim.SGD(cpu.parameters(), lr=.05, foreach=False)
        rows = []
        for step in range(3):
            with torch.no_grad():
                for p, q in zip(model.parameters(), cpu.parameters()):
                    q.copy_(p.detach().cpu())
            host.grad = x.grad = None
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            reference_output = cpu(host)
            expected = (reference_output.float()-target.float()).square().mean().to(dtype)
            expected.backward()
            reference_optimizer.step()
            before = npu.stats()
            step_start = time.perf_counter()
            output = model(x)
            if conv:
                # Ensure autograd saves x/weight rather than a full im2col Tensor.
                saved = output.grad_fn.saved_tensors
                assert len(saved) == 2 and saved[0].numel() == x.numel()
                del saved
            loss = torch.nn.functional.mse_loss(output, labels)
            loss.backward()
            optimizer.step()
            after = npu.stats()
            seconds = time.perf_counter()-step_start
            torch.testing.assert_close(output.detach().cpu(), reference_output.detach(), **tolerance)
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
            row = {'step': step, 'loss': value, 'seconds': seconds,
                   'gradient_relative_l2': float(torch.cat(differences).norm()/torch.cat(references).norm().clamp_min(1e-30)),
                   'host_payload_copies': sum(after[k]-before[k] for k in ('host_upload_copies', 'host_download_copies', 'host_device_copies')),
                   'graph_compilations': after['session_creations']-before['session_creations'],
                   'npu_calls': after['npu_calls']-before['npu_calls'],
                   'cached_graphs': after['cached_sessions']}
            if step:
                assert row['host_payload_copies'] == row['graph_compilations'] == 0
            rows.append(row)
            print(row, flush=True)
            report['active_steps'] = rows
            save()
            del output, loss
        assert rows[-1]['loss'] < rows[0]['loss'], rows
        report.pop('active_steps', None)
        return {'steps': rows, 'input_shape': shape, 'input_gradient_checked': True,
                'weight_gradient_k': shape[0] if not conv else None,
                'convolution_window_elements': shape[0]*target.shape[2]*target.shape[3]*shape[1]*9 if conv else None,
                'convolution_fold_upper_bound': x.numel()*9 if conv else None,
                'stride': cpu.stride if conv else None, 'dilation': cpu.dilation if conv else None,
                'bias': cpu.bias is not None,
                'cpu_reference_rebased_each_step': True, 'gradient_update_tolerance': tolerance}

    try:
        if case.startswith('k-'):
            check('large_k_matrix', matrices)
            check('large_contiguous_slice_transport', sliced_transport)
            check('batch_65537_linear_training', train)
        elif case == 'conv-dense':
            check('dense_streaming_conv_gradients', dense_conv)
        else:
            check('streaming_conv_training', lambda: train(True))
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
    parser.add_argument('--case', choices=('all', *CASES), default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.child:
        raise SystemExit(execute(args.case))
    summary = OUT / 'torch-npu-stream.json'
    reports = []
    cases = CASES if args.case == 'all' else (args.case,)
    for case in cases:
        summary.write_text(json.dumps({'status': 'running', 'cases': reports, 'stage': case}, indent=2))
        try:
            result = subprocess.run([sys.executable, __file__, '--child', '--case', case],
                capture_output=True, text=True, timeout=600, env=os.environ.copy())
            (OUT / f'torch-npu-stream-{case}.log').write_text(result.stdout+result.stderr)
            print(result.stdout+result.stderr, flush=True)
            reports.append(json.loads((OUT / f'torch-npu-stream-{case}.json').read_text()))
            summary.write_text(json.dumps({'status': 'failed' if result.returncode else 'running',
                                          'cases': reports}, indent=2))
            if result.returncode:
                raise SystemExit(result.returncode)
        except subprocess.TimeoutExpired:
            summary.write_text(json.dumps({'status': 'timeout', 'case': case, 'cases': reports}, indent=2))
            raise SystemExit(1)
    summary.write_text(json.dumps({'status': 'passed', 'cases': reports}, indent=2))

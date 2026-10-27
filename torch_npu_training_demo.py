"""Bounded genuine-device autograd/training checks. Run through run_torch_npu.ps1."""
import argparse
import copy
import faulthandler
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'npunlock-probe/research/results'
REPORT = OUT / 'torch-npu-training.json'


def make_small_resnet(dtype, *, mixed_precision=False):
    import torch
    if mixed_precision and dtype != torch.float16:
        raise ValueError('Explicit mixed SmallResNet requires FP16 convolution weights')

    class MixedConv2d(torch.nn.Conv2d):
        def forward(self, x):
            # Half weight storage, FP32 activation/compute. Rounding every convolution
            # output to half still amplified small cross-device differences in training.
            bias = self.bias.float() if self.bias is not None else None
            return self._conv_forward(x.float(), self.weight.float(), bias)

    conv = MixedConv2d if mixed_precision else torch.nn.Conv2d

    class ResidualBlock(torch.nn.Module):
        def __init__(self, incoming, outgoing, stride, dtype):
            super().__init__()
            self.branch = torch.nn.Sequential(
                conv(incoming, outgoing, 3, stride=stride, padding=1, bias=False, dtype=dtype),
                torch.nn.BatchNorm2d(outgoing, track_running_stats=False), torch.nn.ReLU(),
                conv(outgoing, outgoing, 3, padding=1, bias=False, dtype=dtype),
                torch.nn.BatchNorm2d(outgoing, track_running_stats=False))
            self.skip = (torch.nn.Identity() if incoming == outgoing and stride == 1 else torch.nn.Sequential(
                conv(incoming, outgoing, 1, stride=stride, bias=False, dtype=dtype),
                torch.nn.BatchNorm2d(outgoing, track_running_stats=False)))
            self.activation = torch.nn.ReLU()

        def forward(self, x):
            return self.activation(self.branch(x) + self.skip(x))

    class SmallResNet(torch.nn.Module):
        """Four residual blocks, channel changes/downsampling, ordinary PyTorch modules."""
        def __init__(self, dtype):
            super().__init__()
            self.layers = torch.nn.Sequential(
                conv(3, 8, 3, padding=1, bias=False, dtype=dtype),
                torch.nn.BatchNorm2d(8, track_running_stats=False), torch.nn.ReLU(),
                ResidualBlock(8, 8, 1, dtype), ResidualBlock(8, 8, 1, dtype),
                ResidualBlock(8, 16, 2, dtype), ResidualBlock(16, 16, 1, dtype),
                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                torch.nn.Linear(16, 4, dtype=torch.float32 if mixed_precision else dtype))

        def forward(self, x):
            return self.layers(x)

    return SmallResNet(dtype)


def execute(quick, models_only):
    import numpy as np
    import torch
    import torch_intel_npu as npu
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    torch.manual_seed(21)
    torch.backends.mkldnn.enabled = False  # CPU FP16 reference; oneDNN rejects half backward here.
    report = {'status': 'running', 'checks': [], 'openvino_installed': importlib.util.find_spec('openvino') is not None}
    started = time.perf_counter()

    def save(stage):
        report['stage'] = stage
        REPORT.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)

    def check(name, function):
        save(name)
        before = npu.stats()
        baseline = before['live_buffers'] - before['cached_scalar_buffers']
        result = function()
        gc.collect()
        after = npu.stats()
        assert after['live_buffers'] - after['cached_scalar_buffers'] == baseline, f'Leaked {name}'
        assert not after['context_failed']
        report['checks'].append({'name': name, 'status': 'passed', **(result or {})})

    def close(actual, expected, dtype):
        if actual.device.type == 'intel_npu':
            assert actual.dtype == expected.dtype
            actual = actual.detach().cpu()
        torch.testing.assert_close(actual, expected.detach(), rtol=4e-3 if dtype == torch.float16 else 7e-5,
                                   atol=5e-4 if dtype == torch.float16 else 6e-6, equal_nan=True)

    def resident(tensor):
        return tensor.detach().to('intel_npu').requires_grad_(tensor.requires_grad)

    def arithmetic_gradients():
        comparisons = 0
        for dtype in (torch.float16, torch.float32):
            for kind in ('add', 'sub', 'mul', 'div'):
                x = torch.tensor([[.5], [1.5], [2.]], dtype=dtype, requires_grad=True)
                y = torch.tensor([[1., 2., .5, 4.]], dtype=dtype, requires_grad=True)
                a, b = resident(x), resident(y)
                expected = getattr(torch, kind)(x, y).mean()
                actual = getattr(torch, kind)(a, b).mean()
                expected.backward()
                actual.backward()
                close(actual, expected, dtype)
                close(a.grad, x.grad, dtype)
                close(b.grad, y.grad, dtype)
                assert a.grad.device.type == b.grad.device.type == 'intel_npu'
                comparisons += 3
            for axes, keepdim in ((None, False), ((0, 2), False), ((1,), True)):
                x = torch.linspace(-.5, .5, 30, dtype=dtype).view(2, 3, 5).requires_grad_()
                a = resident(x)
                expected = x.sum(axes, keepdim=keepdim, dtype=torch.float32).mean()
                actual = a.sum(axes, keepdim=keepdim, dtype=torch.float32).mean()
                expected.backward()
                actual.backward()
                close(a.grad, x.grad, dtype)
                comparisons += 1
            x = torch.tensor([.5, -.25], dtype=dtype, requires_grad=True)
            a = resident(x)
            expected = ((x*3. + x) / 2.).sum()
            actual = ((a*3. + a) / 2.).sum()
            expected.backward()
            actual.backward()
            close(a.grad, x.grad, dtype)  # Branch accumulation into the same leaf.
            actual = (a*a).sum()
            actual.backward()  # Accumulate a second graph into existing .grad.
            expected = (x*x).sum()
            expected.backward()
            close(a.grad, x.grad, dtype)
            comparisons += 2
        return {'comparisons': comparisons, 'broadcast_reduction_and_branch_accumulation': True}

    def matrix_gradients():
        comparisons = 0
        for dtype in (torch.float16, torch.float32):
            for add_bias in (False, True):
                x = torch.linspace(-.5, .5, 15, dtype=dtype).view(3, 5).requires_grad_()
                y = torch.linspace(-.25, .5, 20, dtype=dtype).view(5, 4).requires_grad_()
                bias = torch.tensor([.1, .2, -.3, .5], dtype=dtype, requires_grad=True)
                a, b, c = resident(x), resident(y), resident(bias)
                expected = torch.addmm(bias, x, y, alpha=.5, beta=2.) if add_bias else x @ y
                actual = torch.addmm(c, a, b, alpha=.5, beta=2.) if add_bias else a @ b
                seed = torch.linspace(-.5, .5, 12, dtype=dtype).view(3, 4)
                expected.backward(seed)
                actual.backward(seed.to('intel_npu'))
                close(actual, expected, dtype)
                close(a.grad, x.grad, dtype)
                close(b.grad, y.grad, dtype)
                if add_bias:
                    close(c.grad, bias.grad, dtype)
                comparisons += 4 if add_bias else 3
            for shape in ((5,), (2, 5), (2, 3, 5), (2, 2, 3, 5)):
                cpu = torch.nn.Linear(5, 4, dtype=dtype)
                model = copy.deepcopy(cpu).to('intel_npu')
                x = torch.linspace(-.5, .5, int(np.prod(shape)), dtype=dtype).reshape(shape).requires_grad_()
                data = resident(x)
                expected, actual = cpu(x), model(data)
                seed = torch.full_like(expected, .25)
                expected.backward(seed)
                actual.backward(seed.to('intel_npu'))
                close(actual, expected, dtype)
                close(data.grad, x.grad, dtype)
                for p, q in zip(model.parameters(), cpu.parameters()):
                    close(p.grad, q.grad, dtype)
                comparisons += 4
        return {'comparisons': comparisons, 'linear_input_ranks': [1, 2, 3, 4]}

    def relu_precision():
        comparisons = 0
        for dtype, integer, floating in ((torch.float16, np.uint16, np.float16), (torch.float32, np.uint32, np.float32)):
            bits = np.arange(65536, dtype=integer) if dtype == torch.float16 else np.array(
                [0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00001, 0xffc00001], dtype=integer)
            x = torch.from_numpy(bits.view(floating)).requires_grad_()
            a = resident(x)
            expected, actual = torch.relu(x), torch.relu(a)
            close(actual, expected, dtype)
            seed = torch.ones_like(x)
            seed[0] = float('nan')
            expected.backward(seed)
            actual.backward(seed.to('intel_npu'))
            close(a.grad, x.grad, dtype)
            # Exact finite encodings and sign of zero are retained.
            expected_bits = expected.detach().numpy().view(integer)
            actual_bits = actual.detach().cpu().numpy().view(integer)
            finite = np.isfinite(expected.detach().numpy())
            assert np.array_equal(expected_bits[finite], actual_bits[finite])
            comparisons += len(bits)
        return {'relu_bit_patterns': comparisons, 'negative_zero_subnormal_inf_nan': True}

    def conv_gradients():
        comparisons = 0
        configs = [((2, 1, 5, 5), 2, 3, 1, 1, 1, True),
                   ((1, 2, 6, 5), 3, (2, 3), (2, 1), (1, 0), 1, False),
                   ((1, 1, 7, 7), 2, 3, 1, 2, 2, True)]
        for dtype in (torch.float16, torch.float32):
            for shape, channels, kernel, stride, padding, dilation, bias in configs:
                cpu = torch.nn.Conv2d(shape[1], channels, kernel, stride=stride, padding=padding,
                                      dilation=dilation, bias=bias, dtype=dtype)
                model = copy.deepcopy(cpu).to('intel_npu')
                x = torch.linspace(-.5, .5, int(np.prod(shape)), dtype=dtype).reshape(shape).requires_grad_()
                a = resident(x)
                expected, actual = cpu(x), model(a)
                seed = torch.linspace(-.2, .3, expected.numel(), dtype=dtype).view(expected.shape)
                expected.backward(seed)
                actual.backward(seed.to('intel_npu'))
                close(actual, expected, dtype)
                close(a.grad, x.grad, dtype)
                for p, q in zip(model.parameters(), cpu.parameters()):
                    close(p.grad, q.grad, dtype)
                comparisons += 2 + len(list(model.parameters()))
        return {'comparisons': comparisons, 'configurations_per_dtype': len(configs), 'stride_padding_dilation_and_bias': True}

    def rsqrt_precision():
        # Full exponent range, random mantissas, zero/subnormal/inf/NaN/negative boundaries.
        bits = np.random.default_rng(23).integers(1, 0x7f800000, 8192, dtype=np.uint32)
        special = np.array([0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x7f7fffff,
                            0x7f800000, 0xff800000, 0x7fc00001, 0xffc00001, 0xbf800000], dtype=np.uint32)
        host = torch.from_numpy(np.concatenate([bits, special]).view(np.float32))
        with torch.no_grad():
            actual = npu._execute('rsqrt', host.to('intel_npu')).cpu()
        torch.testing.assert_close(actual, host.rsqrt(), rtol=3e-7, atol=0., equal_nan=True)
        assert torch.signbit(actual[8192:8194]).tolist() == [False, True]
        return {'patterns': host.numel(), 'relative_tolerance': 3e-7, 'arithmetic': 'SHAVE FP32, integer sqrt + FP32 reciprocal'}

    def tiled_gradients():
        comparisons = 0
        for dtype in (torch.float16, torch.float32):
            for m, k, n in ((65, 32, 32), (129, 32, 32), (33, 64, 64)):
                print(f'tiled matrix {dtype} {m}x{k}x{n}', flush=True)
                x = (torch.randn(m, k, dtype=dtype)/8).requires_grad_()
                y = (torch.randn(k, n, dtype=dtype)/8).requires_grad_()
                a, b = resident(x), resident(y)
                expected, actual = x@y, a@b
                seed = torch.randn_like(expected)/8
                expected.backward(seed)
                actual.backward(seed.to('intel_npu'))
                for u, v in ((actual, expected), (a.grad, x.grad), (b.grad, y.grad)):
                    close(u, v, dtype)
                    comparisons += 1
            cpu = torch.nn.Conv2d(8, 8, 3, padding=1, dtype=dtype)
            model = copy.deepcopy(cpu).to('intel_npu')
            host = (torch.randn(2, 8, 8, 8, dtype=dtype)/4).requires_grad_()
            data = resident(host)
            expected, actual = cpu(host), model(data)
            seed = torch.randn_like(expected)/8
            expected.backward(seed)
            actual.backward(seed.to('intel_npu'))
            for u, v in ((actual, expected), (data.grad, host.grad),
                         (model.weight.grad, cpu.weight.grad), (model.bias.grad, cpu.bias.grad)):
                close(u, v, dtype)
                comparisons += 1
        return {'comparisons': comparisons, 'matmul_MKN': [65*32*32, 129*32*32, 33*64*64],
                'conv_MKN': 2*8*8*72*8, 'forward_backward_tiled': True}

    def batchnorm_gradients():
        comparisons = 0
        for dtype in (torch.float16, torch.float32):
            for affine in (False, True):
                for parameter_dtype in ((dtype, torch.float32) if dtype == torch.float16 and affine else (dtype,)):
                    cpu = torch.nn.BatchNorm2d(3, affine=affine, track_running_stats=False, dtype=parameter_dtype)
                    if affine:
                        with torch.no_grad():
                            cpu.weight.copy_(torch.tensor([.5, -1., 1.25], dtype=parameter_dtype))
                            cpu.bias.copy_(torch.tensor([.1, -.2, .3], dtype=parameter_dtype))
                    model = copy.deepcopy(cpu).to('intel_npu')
                    host = (torch.randn(2, 3, 5, 4, dtype=dtype)/3).requires_grad_()
                    data = resident(host)
                    expected, actual = cpu(host), model(data)
                    seed = torch.randn_like(expected)/8
                    expected.backward(seed)
                    actual.backward(seed.to('intel_npu'))
                    for u, v in ((actual, expected), (data.grad, host.grad)):
                        close(u, v, dtype)
                        comparisons += 1
                    for p, q in zip(model.parameters(), cpu.parameters()):
                        close(p.grad, q.grad, dtype)
                        comparisons += 1
                    with torch.no_grad():
                        model.eval()
                        cpu.eval()
                        close(model(data), cpu(host), dtype)  # No running stats: eval also uses batch stats.
                        comparisons += 1
            # Constant inputs exercise zero variance. Functional API checks running stats/eval gradients.
            for training in (True, False):
                host = torch.full((2, 3, 3, 4), .25, dtype=dtype, requires_grad=True)
                data = resident(host)
                mean = torch.tensor([.1, -.1, .2], dtype=torch.float32)
                variance = torch.tensor([.5, 1., 2.], dtype=torch.float32)
                w = torch.tensor([.5, -1., 1.25], dtype=torch.float32, requires_grad=True)
                b = torch.tensor([.1, -.2, .3], dtype=torch.float32, requires_grad=True)
                dm, dv, dw, db = [resident(t) for t in (mean, variance, w, b)]
                expected = torch.nn.functional.batch_norm(host, mean, variance, w, b, training, .2, 1e-5)
                actual = torch.nn.functional.batch_norm(data, dm, dv, dw, db, training, .2, 1e-5)
                seed = torch.linspace(-.3, .2, host.numel(), dtype=dtype).view(host.shape)
                expected.backward(seed)
                actual.backward(seed.to('intel_npu'))
                for u, v in ((actual, expected), (data.grad, host.grad), (dw.grad, w.grad), (db.grad, b.grad), (dm, mean), (dv, variance)):
                    close(u, v, dtype)
                    comparisons += 1
        return {'comparisons': comparisons, 'affine_optional': True, 'mixed_fp16_fp32': True,
                'functional_running_buffers': True, 'module_track_running_stats': False}

    def pooling_gradients():
        comparisons = 0
        for dtype in (torch.float16, torch.float32):
            for shape in ((2, 3, 5, 7), (1, 4, 1, 1)):
                host = torch.randn(shape, dtype=dtype).requires_grad_()
                data = resident(host)
                pool = torch.nn.AdaptiveAvgPool2d(1)
                expected, actual = pool(host), pool(data)
                seed = torch.randn_like(expected)
                expected.backward(seed)
                actual.backward(seed.to('intel_npu'))
                close(actual, expected, dtype)
                close(data.grad, host.grad, dtype)
                comparisons += 2
        return {'comparisons': comparisons, 'output_size': [1, 1]}

    class TinyResNet(torch.nn.Module):
        """One two-convolution residual block; intentionally no BatchNorm."""
        def __init__(self, dtype):
            super().__init__()
            self.stem = torch.nn.Conv2d(1, 2, 3, padding=1, dtype=dtype)
            self.block1 = torch.nn.Conv2d(2, 2, 3, padding=1, dtype=dtype)
            self.block2 = torch.nn.Conv2d(2, 2, 3, padding=1, dtype=dtype)
            self.head = torch.nn.Linear(2, 2, dtype=dtype)

        def forward(self, x):
            stem = torch.relu(self.stem(x))
            branch = self.block2(torch.relu(self.block1(stem)))
            x = torch.relu(branch + stem)
            return self.head(x.mean(dim=(2, 3)))

    def train(kind, dtype):
        cpu = (torch.nn.Sequential(torch.nn.Linear(5, 4, dtype=dtype), torch.nn.ReLU(), torch.nn.Linear(4, 2, dtype=dtype))
               if kind == 'mlp' else
               torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1, dtype=dtype), torch.nn.ReLU())
               if kind == 'conv' else TinyResNet(dtype) if kind == 'tiny_resnet' else
               torch.nn.Sequential(torch.nn.Linear(32, 64, dtype=dtype), torch.nn.ReLU(), torch.nn.Linear(64, 16, dtype=dtype))
               if kind == 'wide_mlp' else make_small_resnet(dtype))
        model = copy.deepcopy(cpu).to('intel_npu')
        shape = (2, 5) if kind == 'mlp' else (64, 32) if kind == 'wide_mlp' else (2, 3, 8, 8) if kind == 'small_resnet' else (2, 1, 5, 5)
        host = (torch.randn(shape, dtype=dtype)/3 if models_only else
                torch.linspace(-.5, .5, int(np.prod(shape)), dtype=dtype).reshape(shape))
        data = host.to('intel_npu')
        with torch.no_grad():
            target = torch.full_like(cpu(host), .2)
        labels = target.to('intel_npu')
        reference_optimizer = torch.optim.SGD(cpu.parameters(), lr=.05, foreach=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.05, foreach=False)
        steps = 3 if quick else 15
        losses, calls, warm_host_copies, warm_compilations, timings = [], 0, 0, 0, []
        rebased_reference = kind == 'small_resnet' and dtype == torch.float16
        gradient_relative_errors, gradient_cosines = [], []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            if rebased_reference and step:
                # Verify each half-precision step from identical weights. Independent
                # FP16 trajectories can cross different ReLU boundaries after rounding.
                # These are verification downloads, outside the measured NPU step.
                with torch.no_grad():
                    for p, q in zip(model.parameters(), cpu.parameters()):
                        q.copy_(p.detach().cpu())
            reference = cpu(host)
            expected_loss = (reference.float()-target.float()).square().mean().to(dtype)
            expected_loss.backward()
            weights_before = [p.detach().clone() for p in cpu.parameters()] if rebased_reference else None
            reference_optimizer.step()
            before = npu.stats()
            step_start = time.perf_counter()
            output = model(data)
            loss = torch.nn.functional.mse_loss(output, labels)
            loss.backward()
            assert all(p.grad.device.type == 'intel_npu' and p.grad.dtype == p.dtype for p in model.parameters())
            optimizer.step()
            timings.append(time.perf_counter()-step_start)
            after = npu.stats()
            calls += after['npu_calls'] - before['npu_calls']
            if step:
                warm_host_copies += sum(after[k]-before[k] for k in ('host_upload_copies', 'host_download_copies', 'host_device_copies'))
                warm_compilations += after['session_creations'] - before['session_creations']
            # Explicit validation downloads occur after the measured training step.
            close(loss, expected_loss, dtype)
            losses.append(float(loss.detach().cpu()))
            if models_only:
                print(f'{kind} {dtype} step={step} loss={losses[-1]:.7g}', flush=True)
            with torch.no_grad():
                differences, references, actual_gradients = [], [], []
                for index, ((name, p), q) in enumerate(zip(model.named_parameters(), cpu.parameters())):
                    try:
                        if rebased_reference:
                            gradient = p.grad.detach().cpu()
                            assert torch.isfinite(gradient).all(), f'Nonfinite gradient: {name}'
                            close(gradient, q.grad, dtype)
                            differences.append((gradient.float()-q.grad.float()).reshape(-1))
                            references.append(q.grad.float().reshape(-1))
                            actual_gradients.append(gradient.float().reshape(-1))
                            # Exact SGD check against the gradient actually produced by NPU.
                            expected_update = weights_before[index].add_(gradient, alpha=-.05)
                            close(p, expected_update, dtype)
                            close(p, q, dtype)
                        else:
                            close(p.grad, q.grad, dtype)
                            close(p, q, dtype)
                    except AssertionError as error:
                        error.add_note(f'{kind} {dtype} step={step}, parameter={name}')
                        raise
                if rebased_reference:
                    difference, reference_gradient, actual_gradient = map(torch.cat, (differences, references, actual_gradients))
                    relative = float(difference.norm()/reference_gradient.norm().clamp_min(1e-30))
                    cosine = float(torch.nn.functional.cosine_similarity(actual_gradient, reference_gradient, dim=0))
                    gradient_relative_errors.append(relative)
                    gradient_cosines.append(cosine)
                    print(f'half residual CPU gradient relative_L2={relative:.6g}, cosine={cosine:.6g}', flush=True)
        assert losses[-1] < losses[0], losses
        assert warm_host_copies == 0
        if not models_only:
            assert warm_compilations == 0
        return {'network': kind, 'dtype': str(dtype), 'steps': steps, 'loss_first': losses[0], 'loss_last': losses[-1],
                'npu_calls': calls, 'warm_step_host_payload_copies': warm_host_copies,
                'warm_step_graph_compilations': warm_compilations, 'gradients_and_updates_match_cpu': True,
                'batchnorm': kind == 'small_resnet', 'parameters': sum(p.numel() for p in model.parameters()),
                'warm_step_median_seconds': statistics.median(timings[1:]),
                'residual_blocks': 4 if kind == 'small_resnet' else 1 if kind == 'tiny_resnet' else 0,
                'cpu_reference_rebased_each_step': rebased_reference,
                'strict_cpu_gradient_steps': steps,
                'cpu_gradient_relative_l2_per_step': gradient_relative_errors,
                'cpu_gradient_relative_l2_max': max(gradient_relative_errors, default=None),
                'cpu_gradient_cosine_per_step': gradient_cosines,
                'sgd_updates_checked_against_actual_npu_gradients': rebased_reference,
                'batchnorm_track_running_stats': False if kind == 'small_resnet' else None}

    def contracts():
        x = torch.ones(1, 1, 4, 4).to('intel_npu').requires_grad_()
        w = torch.ones(1, 1, 3, 3).to('intel_npu').requires_grad_()
        actions = {
            'groups': lambda: torch.nn.functional.conv2d(x, w, groups=2),
            'transposed_output_padding': lambda: torch.nn.functional.conv_transpose2d(x, w, output_padding=1),
            'oversized': lambda: torch.nn.functional.conv2d(torch.ones(1, 16, 384, 384).to('intel_npu'),
                                                            torch.ones(npu.MAX_DMA_ELEMENTS//(382*382)+1, 16, 3, 3).to('intel_npu')),
            'wrong_bias': lambda: torch.nn.functional.conv2d(x, w, torch.ones(2).to('intel_npu')),
            'mse_incompatible_broadcast': lambda: torch.nn.functional.mse_loss(x, torch.ones(3).to('intel_npu')),
            'cpu_training_scalar_tensor': lambda: x * torch.tensor(2., requires_grad=True),
            'batchnorm_single_value': lambda: torch.nn.functional.batch_norm(
                torch.ones(1, 2, 1, 1).to('intel_npu'), None, None, training=True),
            'batchnorm_bad_parameter': lambda: torch.nn.functional.batch_norm(x, None, None,
                torch.ones(2).to('intel_npu'), training=True),
            'batchnorm_eval_without_buffers': lambda: torch.nn.functional.batch_norm(x, None, None),
            'batchnorm_negative_eps': lambda: torch.nn.functional.batch_norm(x, None, None, training=True, eps=-1.),
            'empty_adaptive_pool_output': lambda: torch.nn.functional.adaptive_avg_pool2d(x, (0, 8)),
        }
        before = npu.stats()['npu_calls']
        for name, action in actions.items():
            try:
                action()
            except (ValueError, RuntimeError, NotImplementedError):
                pass
            else:
                raise AssertionError(f'Accepted {name}')
        assert npu.stats()['npu_calls'] == before
        capacity = npu._MAX_SESSIONS
        try:
            npu._MAX_SESSIONS = 0
            try:
                torch.relu(torch.ones(19, 23).to('intel_npu'))
            except RuntimeError as error:
                assert 'graph cache reached' in str(error)
            else:
                raise AssertionError('Graph cache capacity was not enforced')
        finally:
            npu._MAX_SESSIONS = capacity
        assert npu.stats()['npu_calls'] == before
        # Version tracking must reject weight mutation between forward/backward.
        model = torch.nn.Linear(2, 1).to('intel_npu')
        data = torch.ones(1, 2).to('intel_npu').requires_grad_()
        loss = model(data).sum()
        with torch.no_grad():
            model.weight.add_(torch.ones_like(model.weight))
        try:
            loss.backward()
        except RuntimeError as error:
            assert 'version' in str(error) or 'inplace' in str(error), str(error)
        else:
            raise AssertionError('Saved tensor version check missing')
        a = torch.ones(2).to('intel_npu').requires_grad_()
        gradient, = torch.autograd.grad((a*a).sum(), a, create_graph=True)
        try:
            torch.autograd.grad(gradient.sum(), a)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Higher-order differentiation silently accepted')
        return {'rejected': list(actions), 'saved_tensor_version_guard': True, 'higher_order_autograd': False,
                'graph_cache_capacity_guard': True}

    try:
        if models_only:
            check('rsqrt_precision', rsqrt_precision)
            check('tiled_matrix_and_conv_gradients', tiled_gradients)
            check('batchnorm_train_eval_and_gradients', batchnorm_gradients)
            check('global_average_pool_gradients', pooling_gradients)
        else:
            check('arithmetic_broadcast_reduction_gradients', arithmetic_gradients)
            check('matmul_addmm_linear_gradients', matrix_gradients)
            check('relu_precision_and_gradients', relu_precision)
            check('conv2d_input_weight_bias_gradients', conv_gradients)
        for dtype in ((torch.float32, torch.float16) if models_only else (torch.float16, torch.float32)):
            for kind in (('wide_mlp', 'small_resnet') if models_only else ('mlp', 'conv', 'tiny_resnet')):
                check(f'{kind}_{dtype}_training', lambda k=kind, d=dtype: train(k, d))
        check('training_contracts', contracts)
        assert npu.stats()['live_buffers'] == npu.stats()['cached_scalar_buffers']
        report['stats_before_shutdown'] = npu.stats()
        npu.shutdown()
        assert npu.stats()['live_buffers'] == npu.stats()['live_buffer_bytes'] == 0
        # Same-weight FP16 checks pass the original pointwise tolerance, but this
        # does not certify an independent trajectory or small relative error near zero.
        limited = any(c.get('cpu_reference_rebased_each_step') for c in report['checks'])
        report.update(status='passed_with_limitations' if limited else 'passed', seconds=time.perf_counter()-started, final_stats=npu.stats())
        if limited:
            report['limitations'] = ['All recorded same-weight FP16 gradient/update checks pass the original pointwise tolerance. The CPU reference is rebased each step; near-convergence relative gradient differences remain. See per-step relative L2/cosine; independent FP16 trajectories are not certified.']
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
    parser.add_argument('--models', action='store_true')
    args = parser.parse_args()
    if args.models:
        REPORT = OUT / 'torch-npu-models.json'
    OUT.mkdir(parents=True, exist_ok=True)
    if args.child:
        raise SystemExit(execute(args.quick, args.models))
    command = [sys.executable, __file__, '--child'] + (['--quick'] if args.quick else []) + (['--models'] if args.models else [])
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600, env=os.environ.copy())
        REPORT.with_suffix('.log').write_text(result.stdout+result.stderr)
        print(result.stdout+result.stderr, flush=True)
        raise SystemExit(result.returncode)
    except subprocess.TimeoutExpired:
        report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
        report.update(status='timeout', error='Outer process deadline: 600 seconds')
        REPORT.write_text(json.dumps(report, indent=2))
        raise SystemExit(1)

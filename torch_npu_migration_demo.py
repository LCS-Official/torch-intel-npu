"""Bounded ordinary-PyTorch model migration checks; no frozen-model adapter."""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'npunlock-probe/research/results/torch-migration'
CASES = ('models', 'sgd', 'updates', 'boundaries')


def execute(case):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(43)
    torch.backends.mkldnn.enabled = False  # The CPU FP16 backward reference needs this.
    checks = []

    def close(actual, expected, dtype):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(),
                                   rtol=5e-3 if dtype == torch.float16 else 1e-4,
                                   atol=8e-4 if dtype == torch.float16 else 8e-6, equal_nan=True)

    def check(name, function, blocked=None):
        before = npu.stats()
        row = {'name': name, 'expectation': 'blocked' if blocked else 'supported'}
        try:
            details = function()
            if isinstance(details, dict):
                row.update(details)
            del details
            row['status'] = 'failed' if blocked else 'passed'
            if blocked:
                row['error'] = 'Expected an explicit unsupported error'
        except Exception as error:
            row['error'] = str(error)
            if blocked and isinstance(error, (ValueError, RuntimeError, NotImplementedError)) and blocked in str(error):
                row['status'] = 'blocked'
            else:
                row.update(status='failed', traceback=traceback.format_exc())
        gc.collect()
        after = npu.stats()
        row['npu_calls'] = after['npu_calls'] - before['npu_calls']
        if row['status'] == 'passed' and not row['npu_calls']:
            row.update(status='failed', error='Supported compute check did not submit an NPU graph')
        row['live_buffer_delta'] = ((after['live_buffers'] - after['cached_scalar_buffers']) -
                                    (before['live_buffers'] - before['cached_scalar_buffers']))
        if row['live_buffer_delta'] or after['context_failed']:
            row.update(status='failed', lifecycle_error='Live buffer leak or failed NPU context')
        checks.append(row)
        print(name, row['status'], row.get('error', '')[:150], flush=True)

    class TinyResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Conv2d(1, 2, 3, padding=1)
            self.branch = torch.nn.Sequential(torch.nn.Conv2d(2, 2, 3, padding=1), torch.nn.ReLU(),
                                              torch.nn.Conv2d(2, 2, 3, padding=1))
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.head = torch.nn.Linear(2, 2)

        def forward(self, x):
            x = torch.relu(self.stem(x))
            return self.head(self.pool(torch.relu(x + self.branch(x))).flatten(1))

    def model(kind, dtype):
        net = (torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
               if kind == 'mlp' else
               torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1), torch.nn.ReLU(),
                                   torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(2, 2))
               if kind == 'conv' else TinyResNet())
        return net.to(dtype=dtype)

    def train(kind, dtype):
        cpu = model(kind, dtype)
        device_model = copy.deepcopy(cpu).to('intel_npu')
        shape = (2, 5) if kind == 'mlp' else (2, 1, 4, 4)
        host = torch.linspace(-.4, .5, torch.Size(shape).numel(), dtype=dtype).reshape(shape).requires_grad_()
        data = host.detach().to('intel_npu').requires_grad_()
        target = torch.tensor([[.2, -.1], [-.1, .3]], dtype=dtype)
        labels = target.to('intel_npu')
        options = dict(lr=.025, momentum=.8, dampening=.1, foreach=False, fused=False)
        expected_opt = torch.optim.SGD(cpu.parameters(), **options)
        opt = torch.optim.SGD(device_model.parameters(), **options)
        losses, copies, scalar_uploads = [], [], []

        def step():
            expected_opt.zero_grad(set_to_none=False)
            opt.zero_grad(set_to_none=False)
            host.grad = data.grad = None
            expected = cpu(host)
            expected_loss = torch.nn.functional.mse_loss(expected.float(), target.float()).to(dtype)
            expected_loss.backward()
            expected_opt.step()
            before = npu.stats()
            actual = device_model(data)
            loss = torch.nn.functional.mse_loss(actual, labels)
            loss.backward()
            opt.step()
            after = npu.stats()
            payload_copies = sum(after[k] - before[k] for k in
                                 ('host_upload_copies', 'host_download_copies', 'host_device_copies'))
            new_scalars = after['cached_scalar_buffers'] - before['cached_scalar_buffers']
            # Momentum's mul/scale constants are first needed at the second step.
            assert payload_copies == new_scalars, (payload_copies, new_scalars)
            close(actual, expected, dtype)
            close(loss, expected_loss, dtype)
            close(data.grad, host.grad, dtype)
            assert data.grad.device.type == 'intel_npu'
            for p, q in zip(device_model.parameters(), cpu.parameters()):
                assert p.device.type == p.grad.device.type == 'intel_npu'
                close(p, q, dtype)
                close(p.grad, q.grad, dtype)
                close(opt.state[p]['momentum_buffer'], expected_opt.state[q]['momentum_buffer'], dtype)
            losses.append(float(loss.detach().cpu()))
            copies.append(payload_copies)
            scalar_uploads.append(new_scalars)

        for _ in range(3):
            step()
        # Portable checkpoint: explicitly download detached weights and optimizer buffers.
        state = {'model': {k: v.detach().cpu() for k, v in device_model.state_dict().items()},
                 'optimizer': opt.state_dict()}
        state['optimizer']['state'] = {k: {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                                          for key, value in values.items()}
                                        for k, values in state['optimizer']['state'].items()}
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=True)
        restored = model(kind, dtype).to('intel_npu')
        restored.load_state_dict(state['model'])
        restored_opt = torch.optim.SGD(restored.parameters(), **options)
        restored_opt.load_state_dict(state['optimizer'])
        device_model, opt = restored, restored_opt
        step()  # Resume from the same CPU trajectory, including the momentum state.
        device_model.eval()
        cpu.eval()
        with torch.inference_mode():
            close(device_model(data.detach()), cpu(host.detach()), dtype)
        assert all(v == 0 for v in copies[2:]), copies
        return {'precision': str(dtype), 'steps': 4, 'losses': losses, 'step_host_copies': copies,
                'scalar_cache_uploads': scalar_uploads, 'steady_host_payload_copies': copies[2:],
                'portable_checkpoint_resume': True, 'independent_cpu_trajectory': True}

    def sgd(dtype, options):
        cpu = torch.nn.Parameter(torch.tensor([[.4, -.7], [.25, .9]], dtype=dtype).t())
        p = torch.nn.Parameter(cpu.detach().t().contiguous().to('intel_npu').t())
        expected = torch.optim.SGD([cpu], lr=.03, foreach=False, fused=False, **options)
        actual = torch.optim.SGD([p], lr=.03, foreach=False, fused=False, **options)
        for index in range(4):
            gradient = torch.tensor([[.2 + index*.01, -.3], [.5, -.1]], dtype=dtype).t()
            cpu.grad = gradient
            p.grad = gradient.t().contiguous().to('intel_npu').t()
            expected.step()
            actual.step()
            close(p, cpu, dtype)
            if options.get('momentum'):
                close(actual.state[p]['momentum_buffer'], expected.state[cpu]['momentum_buffer'], dtype)
            assert p.stride() == cpu.stride()
        return {'precision': str(dtype), 'steps': 4, 'options': options, 'dense_transposed_parameter': True}

    def updates(dtype):
        for transposed in (False, True):
            source = torch.tensor([[.3, -.7], [.5, 1.2]], dtype=dtype)
            for operand in (.8, source, source.t()):
                expected = source.t().clone() if transposed else source.clone()
                actual = source.to('intel_npu').t() if transposed else source.to('intel_npu')
                other = operand.t().contiguous().to('intel_npu').t() if isinstance(operand, torch.Tensor) else operand
                version = actual._version
                with torch.no_grad():
                    expected.mul_(operand)
                    assert actual.mul_(other) is actual
                assert actual._version == version + 1
                close(actual, expected, dtype)
        source = torch.tensor([[.3, -.7], [.5, 1.2]], dtype=dtype)
        actual = source.to('intel_npu')
        with torch.no_grad():
            actual.mul_(actual)
        close(actual, source*source, dtype)
        return {'precision': str(dtype), 'scalar_tensor_dense_and_version': True}

    def scaled_add(dtype):
        for kind in ('add', 'sub'):
            for alpha in (0., .8, -2., 60000., float('inf'), float('nan')):
                x = torch.tensor([[.125], [-.25]], dtype=dtype).requires_grad_()
                y = torch.tensor([[.25, -.125, 0.]], dtype=dtype).requires_grad_()
                a, b = (v.detach().to('intel_npu').requires_grad_() for v in (x, y))
                expected = getattr(torch, kind)(x, y, alpha=alpha)
                actual = getattr(torch, kind)(a, b, alpha=alpha)
                gradient = torch.tensor([[.5, -.25, 1.], [.125, .5, -.25]], dtype=dtype)
                expected.backward(gradient)
                actual.backward(gradient.to('intel_npu'))
                close(actual, expected, dtype)
                close(a.grad, x.grad, dtype)
                close(b.grad, y.grad, dtype)
                with torch.no_grad():
                    out = torch.empty_like(actual)
                    assert getattr(torch, kind)(a, b, alpha=alpha, out=out) is out
                    close(out, expected, dtype)
            x = torch.tensor([.3333, -.2], dtype=dtype)
            a = x.to('intel_npu')
            close(getattr(torch, kind)(a, .3333, alpha=.8), getattr(torch, kind)(x, .3333, alpha=.8), dtype)
        return {'precision': str(dtype), 'broadcast_and_first_order_gradients': True,
                'zero_negative_large_inf_nan_alpha_and_out': True}

    def update_contracts():
        x = torch.ones(2, 2).to('intel_npu')
        half = torch.ones(2, 2, dtype=torch.float16).to('intel_npu')
        small = torch.ones(1).to('intel_npu')
        partial = x.view(-1)[1:]
        before = npu.stats()['npu_calls']
        rejected = []
        with torch.no_grad():
            for name, function in {
                'mul-host-tensor': lambda: x.mul_(torch.ones(2, 2)),
                'mul-overlapping-transpose': lambda: x.mul_(x.t()),
                'add-bool-alpha': lambda: torch.add(x, x, alpha=True),
                'add-overflow-alpha': lambda: torch.add(x, x, alpha=1e50),
                'add-half-overflow-alpha': lambda: torch.add(half, half, alpha=70000.),
                'sub-overflow-alpha': lambda: torch.sub(x, x, alpha=1e50),
                'scaled-output-alias': lambda: torch.add(x, x, alpha=.8, out=x),
            }.items():
                try:
                    function()
                except (ValueError, RuntimeError, TypeError, NotImplementedError):
                    rejected.append(name)
                else:
                    raise AssertionError(f'Accepted {name}')
        try:
            x.detach().requires_grad_().mul_(.8)
        except (ValueError,RuntimeError,NotImplementedError) as error:
            assert 'grad' in str(error) or 'leaf' in str(error)
            rejected.append('mul-grad-leaf')
        else:
            raise AssertionError('Accepted grad-enabled inplace update')
        assert npu.stats()['npu_calls'] == before
        close(x, torch.ones(2, 2), torch.float32)
        # An inplace update must invalidate any saved tensor used by a backward.
        a = x.detach().requires_grad_()
        loss = (a*a).sum()
        with torch.no_grad():
            a.mul_(.8)
        try:
            loss.backward()
        except RuntimeError as error:
            assert 'version' in str(error) or 'inplace' in str(error)
        else:
            raise AssertionError('Saved tensor version check was bypassed')
        return {'rejected_without_graph_submit': rejected, 'saved_tensor_version_guard': True}

    def probe(name):
        if name == 'batchnorm-default':
            cpu = torch.nn.BatchNorm2d(2)
            module = copy.deepcopy(cpu).to('intel_npu')
            host = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)/8
            with torch.no_grad():
                close(module(host.to('intel_npu')), cpu(host), torch.float32)
            assert module.num_batches_tracked.item() == 1
            return {'default_running_buffers_and_counter': True}
        if name == 'integer-target':
            labels = torch.tensor([0, 1], dtype=torch.int64).to('intel_npu').clone()
            assert labels.dtype == torch.int64 and torch.equal(labels.cpu(), torch.tensor([0, 1]))
            return {'integer_target_storage_only': True}
        if name == 'cross-entropy':
            host = torch.tensor([[.1, .2], [.3, -.4]], requires_grad=True)
            x = host.detach().to('intel_npu').requires_grad_()
            target = torch.tensor([0, 1])
            expected = torch.nn.functional.cross_entropy(host, target)
            actual = torch.nn.functional.cross_entropy(x, target.to('intel_npu'))
            expected.backward()
            actual.backward()
            close(actual, expected, torch.float32)
            close(x.grad, host.grad, torch.float32)
            return {'resident_int64_target_and_classification_backward': True}
        if name == 'adamw':
            host = torch.nn.Parameter(torch.ones(2))
            p = torch.nn.Parameter(host.detach().to('intel_npu'))
            reference, opt = (torch.optim.AdamW([v],foreach=False,fused=False) for v in (host,p))
            for gradient in (1.,.75):
                host.grad = torch.full_like(host,gradient)
                p.grad = host.grad.to('intel_npu')
                reference.step()
                opt.step()
                close(p,host,torch.float32)
                for key,value in opt.state[p].items():
                    close(value,reference.state[host][key],torch.float32)
            return {'adamw_parameter_and_moment_state': True}
        if name in ('layernorm', 'gelu'):
            host = torch.tensor([[.2, -.7, .4, 1.1], [-.4, .8, -.2, .3]], requires_grad=True)
            x = host.detach().to('intel_npu').requires_grad_()
            cpu = torch.nn.LayerNorm(4) if name == 'layernorm' else torch.nn.GELU()
            model = copy.deepcopy(cpu).to('intel_npu')
            expected, actual = cpu(host), model(x)
            seed = torch.tensor([[.1, .2, -.3, .4], [-.5, .6, .7, -.8]])
            expected.backward(seed)
            actual.backward(seed.to('intel_npu'))
            close(actual, expected, torch.float32)
            close(x.grad, host.grad, torch.float32)
            for a, b in zip(model.parameters(), cpu.parameters()):
                close(a.grad, b.grad, torch.float32)
            return {'forward_backward_cpu_reference': True}
        x = torch.ones(2, 4).to('intel_npu')
        if name == 'softmax':
            return torch.nn.Softmax(dim=-1)(x)
        if name == 'embedding':
            cpu = torch.nn.Embedding(8, 4, padding_idx=0)
            model = copy.deepcopy(cpu).to('intel_npu')
            labels = torch.tensor([[0, 1, 1]])
            expected = cpu(labels)
            actual = model(labels.to('intel_npu'))
            seed = torch.linspace(-.3, .4, 12).reshape_as(expected)
            expected.backward(seed)
            actual.backward(seed.to('intel_npu'))
            close(actual, expected, torch.float32)
            close(model.weight.grad, cpu.weight.grad, torch.float32)
            return {'resident_int64_embedding_and_dense_weight_gradient': True}
        if name == 'batched-matmul':
            host = torch.tensor([[[.2, -.3], [.7, .4]], [[-.1, .5], [.3, -.8]]], requires_grad=True)
            right = torch.tensor([[[.1, -.4], [.6, .2]]], requires_grad=True)
            a, b = (v.detach().to('intel_npu').requires_grad_() for v in (host, right))
            expected, actual = host@right, a@b
            seed = torch.arange(8).reshape(2, 2, 2).float()/8
            expected.backward(seed)
            actual.backward(seed.to('intel_npu'))
            for value, reference in ((actual, expected), (a.grad, host.grad), (b.grad, right.grad)):
                close(value, reference, torch.float32)
            return {'batched_broadcast_forward_backward': True}
        if name == 'sdpa':
            from torch.nn.attention import SDPBackend, sdpa_kernel
            values = [(torch.arange(16).reshape(1, 2, 2, 4).float()/16+i/10).requires_grad_() for i in range(3)]
            inputs = [v.detach().to('intel_npu').requires_grad_() for v in values]
            with sdpa_kernel(SDPBackend.MATH):
                expected = torch.nn.functional.scaled_dot_product_attention(*values, is_causal=True)
            actual = torch.nn.functional.scaled_dot_product_attention(*inputs, is_causal=True)
            seed = torch.linspace(-.3, .4, 16).reshape_as(expected)
            expected.backward(seed)
            actual.backward(seed.to('intel_npu'))
            close(actual, expected, torch.float32)
            for value, reference in zip(inputs, values):
                close(value.grad, reference.grad, torch.float32)
            return {'causal_multihead_forward_backward': True}
        if name == 'native-checkpoint':
            stream = io.BytesIO()
            torch.save({'x': x}, stream)
            stream.seek(0)
            loaded = torch.load(stream, map_location='cpu', weights_only=True)
            close(loaded['x'], torch.ones(2, 4), torch.float32)
            return {'native_device_checkpoint_to_cpu': True}
        raise ValueError(name)

    if case == 'models':
        for kind in ('mlp', 'conv', 'tiny-resnet'):
            for dtype in (torch.float16, torch.float32):
                check(f'{kind}-{dtype}', lambda k=kind, d=dtype: train(k, d))
    elif case == 'sgd':
        for name, options in (
                ('basic', {}), ('momentum', {'momentum': .8}), ('dampening', {'momentum': .8, 'dampening': .2}),
                ('nesterov', {'momentum': .8, 'nesterov': True}), ('weight-decay', {'weight_decay': .05}),
                ('combined', {'momentum': .8, 'nesterov': True, 'weight_decay': .05}),
                ('maximize', {'momentum': .8, 'maximize': True})):
            for dtype in (torch.float16, torch.float32):
                check(f'sgd-{name}-{dtype}', lambda d=dtype, o=options: sgd(d, o))
    elif case == 'updates':
        for dtype in (torch.float16, torch.float32):
            check(f'mul-inplace-{dtype}', lambda d=dtype: updates(d))
            check(f'scaled-add-sub-{dtype}', lambda d=dtype: scaled_add(d))
        check('update-contracts', update_contracts)
    else:
        for name, error in {
            'batchnorm-default': None,
            'integer-target': None,
            'cross-entropy': None,
            'adamw': None,
            'layernorm': None,
            'gelu': None,
            'softmax': None,
            'embedding': None,
            'batched-matmul': None,
            'sdpa': None,
            'native-checkpoint': 'aten.set_.source_Storage',
        }.items():
            check(name, lambda name=name: probe(name), blocked=error)
    gc.collect()
    before_shutdown = npu.stats()
    npu.shutdown()
    final = npu.stats()
    assert final['live_buffers'] == final['cached_buffer_blocks'] == final['quarantined_buffer_bytes'] == final['cached_sessions'] == 0
    assert final['closed'] and not final['context_failed']
    return {'case': case, 'checks': checks, 'stats_before_shutdown': before_shutdown, 'final_stats': final,
            'status': 'failed' if any(c['status'] == 'failed' for c in checks) else 'passed',
            'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                              ('torch_npu_migration_demo.py', 'torch_intel_npu.py', 'torch_npu_autograd.py',
                               'npunlock-probe/torch-device/backend.cpp', 'npunlock-probe/torch-device/_torch_intel_npu.pyd')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', choices=CASES)
    parser.add_argument('--label', default='after')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.child:
        result = execute(args.child)
        (OUT/f'{args.label}-{args.child}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return int(result['status'] != 'passed')
    status = 0
    for case in CASES:
        completed = subprocess.run([sys.executable, __file__, '--child', case, '--label', args.label],
                                   capture_output=True, text=True, encoding='utf-8', timeout=240)
        (OUT/f'{args.label}-{case}.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
        print(completed.stdout + completed.stderr, flush=True)
        status |= int(completed.returncode != 0)
    return status


if __name__ == '__main__':
    raise SystemExit(main())

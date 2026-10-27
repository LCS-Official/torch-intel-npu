"""Bounded classification operators and ordinary PyTorch training on intel_npu."""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/classification'


def execute(case, optimizer='sgd', bn_bias=True):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(53)
    torch.backends.mkldnn.enabled = False
    checks = []

    def close(actual, expected):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(),
            rtol=.003 if actual.dtype == torch.float16 else 3e-5,
            atol=.0005 if actual.dtype == torch.float16 else 3e-6, equal_nan=True)

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
            row.update(status='failed', lifecycle_error='Live buffer leak or failed context')
        checks.append(row)
        print(name, row['status'], row.get('error', '')[:220], flush=True)

    def logsoftmax(dtype):
        comparisons = 0
        for shape in ((1,), (3,), (2, 3), (1, 17), (3, 17), (2, 3, 5), (1, 2, 3, 7), (2, 257), (1, 4096), (1, 65536)):
            host = (torch.randn(shape)*4).to(dtype).requires_grad_()
            data = host.detach().to('intel_npu').requires_grad_()
            expected = torch.nn.functional.log_softmax(host, dim=-1)
            actual = torch.nn.functional.log_softmax(data, dim=-1)
            seed = torch.randn(shape, dtype=dtype)/4
            expected.backward(seed)
            actual.backward(seed.to('intel_npu'))
            close(actual, expected)
            close(data.grad, host.grad)
            with torch.no_grad():
                out = torch.empty_like(data)
                assert torch.ops.aten._log_softmax.out(data, -1, False, out=out) is out
                close(out, expected)
            comparisons += 3
        for values in ([[10000., 10000.125, 9999.875]], [[0., -80., -90., -100., -103., -104.]],
                       [[float('inf'), 0., -1.]], [[-float('inf')]*3], [[float('nan'), 0., 1.]],
                       [[-float('inf'), 0., -1.]]):
            host = torch.tensor(values, dtype=dtype).requires_grad_()
            data = host.detach().to('intel_npu').requires_grad_()
            expected, actual = host.log_softmax(-1), data.log_softmax(-1)
            expected.sum().backward()
            actual.sum().backward()
            close(actual, expected)
            close(data.grad, host.grad)
            comparisons += 2
        host = torch.tensor([[.2, -.3, .5]], dtype=dtype).requires_grad_()
        data = host.detach().to('intel_npu').requires_grad_()
        expected = host.log_softmax(-1, dtype=torch.float32)
        actual = data.log_softmax(-1, dtype=torch.float32)
        expected.sum().backward()
        actual.sum().backward()
        close(actual, expected)
        close(data.grad, host.grad)
        for seed_values in ([float('inf'), 1., 2.], [float('nan'), 1., 2.]):
            host = torch.tensor([[.2, -.3, .5]], dtype=dtype).requires_grad_()
            data = host.detach().to('intel_npu').requires_grad_()
            seed = torch.tensor([seed_values], dtype=dtype)
            host.log_softmax(-1).backward(seed)
            data.log_softmax(-1).backward(seed.to('intel_npu'))
            close(data.grad, host.grad)
        return {'dtype': str(dtype), 'comparisons': comparisons+4, 'last_axis_rank_1_to_4_and_special_values': True}

    def nll(dtype):
        comparisons = 0
        for kind in ('nll', 'cross-entropy'):
            function = torch.nn.functional.nll_loss if kind == 'nll' else torch.nn.functional.cross_entropy
            for labels, ignore in (([0, 2, 1], -100), ([0, -100, 1], -100), ([-100]*3, -100),
                                   ([0, 2, 1], 2), ([0, -2**63, 1], -2**63)):
                for reduction in ('none', 'mean', 'sum'):
                    host = torch.tensor([[.5, -.2, 1.], [.1, -.4, .7], [-.2, .5, -.8]], dtype=dtype).requires_grad_()
                    if kind == 'nll':
                        host = host.detach().log_softmax(-1).requires_grad_()
                    data = host.detach().to('intel_npu').requires_grad_()
                    target = torch.tensor(labels, dtype=torch.int64)
                    expected = function(host, target, reduction=reduction, ignore_index=ignore)
                    actual = function(data, target.to('intel_npu'), reduction=reduction, ignore_index=ignore)
                    seed = torch.full_like(expected, .75)
                    expected.backward(seed)
                    actual.backward(seed.to('intel_npu'))
                    close(actual, expected)
                    close(data.grad, host.grad)
                    comparisons += 2
            host = torch.tensor([.5, -.3, .7], dtype=dtype, requires_grad=True)
            data = host.detach().to('intel_npu').requires_grad_()
            target = torch.tensor(1, dtype=torch.int64)
            expected, actual = function(host, target), function(data, target.to('intel_npu'))
            expected.backward()
            actual.backward()
            close(actual, expected)
            close(data.grad, host.grad)
            comparisons += 2
        # NLL must skip unselected/ignored NaNs, not multiply them by zero.
        host = torch.tensor([[float('nan'), -.3, float('nan')], [float('nan')]*3], dtype=dtype, requires_grad=True)
        data = host.detach().to('intel_npu').requires_grad_()
        target = torch.tensor([1, -100])
        expected = torch.nn.functional.nll_loss(host, target)
        actual = torch.nn.functional.nll_loss(data, target.to('intel_npu'))
        expected.backward()
        actual.backward()
        close(actual, expected)
        close(data.grad, host.grad)
        for shape, labels in (((3,), 1), ((2, 3), [1, 2])):
            host = torch.randn(shape, dtype=dtype)
            target = torch.tensor(labels, dtype=torch.int64)
            for reduction in (0, 1, 2):
                expected = torch.ops.aten.nll_loss_forward(host, target, None, reduction, -100)
                actual = torch.ops.aten.nll_loss_forward(host.to('intel_npu'), target.to('intel_npu'), None, reduction, -100)
                for a, b in zip(actual, expected):
                    close(a, b)
        # Valid-count rounding crosses FP16's exact integer range at 2049.
        host = -torch.ones(2049, 2, dtype=dtype, requires_grad=True)
        host = host.detach().requires_grad_()
        data = host.detach().to('intel_npu').requires_grad_()
        target = torch.zeros(2049, dtype=torch.int64)
        expected = torch.nn.functional.nll_loss(host, target)
        actual = torch.nn.functional.nll_loss(data, target.to('intel_npu'))
        expected.backward()
        actual.backward()
        close(actual, expected)
        close(data.grad, host.grad)
        return {'dtype': str(dtype), 'comparisons': comparisons+16, 'ignore_all_ignored_reductions_and_unselected_nan': True,
                'nll_forward_total_weight': True}

    def contracts():
        x = torch.ones(2, 3).to('intel_npu')
        target = torch.tensor([0, 1], dtype=torch.int64).to('intel_npu')
        weights = torch.ones(2).to('intel_npu')
        integer_soft = torch.ones(2,3,dtype=torch.int64).to('intel_npu')
        before = npu.stats()['npu_calls']
        actions = {
            'logsoftmax-axis': lambda: x.log_softmax(2),
            'logsoftmax-capacity': lambda: torch.ones(1, 65537).to('intel_npu').log_softmax(-1),
            'half-to-float': lambda: torch.ops.aten._log_softmax(x, -1, True),
            'class-weights': lambda: torch.nn.functional.cross_entropy(x, target, weight=weights),
            'label-smoothing': lambda: torch.nn.functional.cross_entropy(x, target, label_smoothing=1.1),
            'integer-soft-targets': lambda: torch.nn.functional.cross_entropy(x, integer_soft),
            'cpu-targets': lambda: torch.nn.functional.cross_entropy(x, torch.tensor([0, 1])),
            'spatial-input': lambda: torch.nn.functional.cross_entropy(x.view(1, 2, 3), target),
            'target-shape': lambda: torch.nn.functional.cross_entropy(x, target.view(1, 2)),
            'nll-capacity': lambda: torch.nn.functional.nll_loss(torch.ones(1, 65533).to('intel_npu'), torch.tensor([0]).to('intel_npu')),
            'logsoftmax-out-alias': lambda: torch.ops.aten._log_softmax.out(x, -1, False, out=x),
            'nll-reduction': lambda: torch.ops.aten.nll_loss_forward(x, target, None, 3, -100),
        }
        rejected = []
        for name, function in actions.items():
            try:
                function()
            except (ValueError, RuntimeError, NotImplementedError):
                rejected.append(name)
            else:
                raise AssertionError(f'Accepted {name}')
        assert npu.stats()['npu_calls'] == before
        for labels in ([0, 3], [0, -1], [0, 2**32], [0, 2**53+1]):
            invalid = torch.tensor(labels, dtype=torch.int64).to('intel_npu')
            try:
                torch.nn.functional.cross_entropy(x, invalid)
            except IndexError as error:
                assert 'out of bounds' in str(error)
            else:
                raise AssertionError(f'Accepted invalid target {labels}')
            assert not npu.stats()['context_failed']
        leaf = x.detach().requires_grad_()
        loss = torch.nn.functional.cross_entropy(leaf, target)
        target.copy_(torch.tensor([1, 2]))
        try:
            loss.backward()
        except RuntimeError as error:
            assert 'version' in str(error) or 'inplace' in str(error)
        else:
            raise AssertionError('Mutated target bypassed saved-tensor version check')
        return {'structural_rejections_without_submit': rejected, 'device_checked_invalid_targets': 4,
                'target_version_guard': True}

    def train(kind, dtype):
        class Residual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1), torch.nn.BatchNorm2d(2), torch.nn.ReLU())
                self.branch = torch.nn.Sequential(torch.nn.Conv2d(2, 2, 3, padding=1), torch.nn.BatchNorm2d(2), torch.nn.ReLU(), torch.nn.Conv2d(2, 2, 3, padding=1))
                self.head = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(2, 3))

            def forward(self, x):
                x = self.stem(x)
                return self.head(torch.relu(x+self.branch(x)))

        cpu = (torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)) if kind == 'mlp' else
               torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1), torch.nn.BatchNorm2d(2), torch.nn.ReLU(),
                   torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(2, 3)) if kind == 'conv' else Residual()).to(dtype=dtype)
        # Remove only biases immediately followed by BatchNorm; preserve the
        # original RNG consumption and all other weights for a controlled check.
        if not bn_bias and kind == 'conv':
            cpu[0].bias = None
        elif not bn_bias and kind == 'residual':
            cpu.stem[0].bias = cpu.branch[0].bias = None
        model = copy.deepcopy(cpu).to('intel_npu')
        host = torch.linspace(-.6, .8, 20 if kind == 'mlp' else 64, dtype=dtype).reshape((4, 5) if kind == 'mlp' else (4, 1, 4, 4)).requires_grad_()
        data = host.detach().to('intel_npu').requires_grad_()
        target = torch.tensor([0, 1, 2, 0])
        labels = target.to('intel_npu')
        loss_fn = torch.nn.CrossEntropyLoss()
        options = (dict(lr=.025, momentum=.8, foreach=False, fused=False) if optimizer == 'sgd' else
                   dict(lr=.001, eps=1e-4 if dtype == torch.float16 else 1e-8, foreach=False, fused=False))
        optimizer_type = torch.optim.SGD if optimizer == 'sgd' else torch.optim.AdamW
        opt_cpu, opt = (optimizer_type(m.parameters(), **options) for m in (cpu, model))
        losses, copies, scalar_uploads = [], [], []
        for index in range(4):
            opt_cpu.zero_grad(set_to_none=False)
            opt.zero_grad(set_to_none=False)
            host.grad = data.grad = None
            expected = cpu(host)
            expected_loss = loss_fn(expected, target)
            expected_loss.backward()
            opt_cpu.step()
            if optimizer == 'adamw':
                # Independent diagnostic: identical NPU weights/moments and later
                # the identical gradient must give the same CPU optimizer update.
                shadow_parameters = [torch.nn.Parameter(p.detach().cpu()) for p in model.parameters()]
                shadow = optimizer_type(shadow_parameters, **options)
                shadow_state = opt.state_dict()
                shadow_state['state'] = {k:{n:v.detach().cpu().clone() for n,v in s.items()}
                                        for k,s in shadow_state['state'].items()}
                shadow.load_state_dict(shadow_state)
            before = npu.stats()
            actual = model(data)
            actual.retain_grad()
            loss = loss_fn(actual, labels)
            loss.backward()
            opt.step()
            after = npu.stats()
            copied = sum(after[k]-before[k] for k in ('host_upload_copies','host_download_copies','host_device_copies'))
            scalars = after['cached_scalar_buffers']-before['cached_scalar_buffers']
            assert copied == scalars+1, (copied, scalars)
            if optimizer == 'adamw':
                for source, target_parameter in zip(model.parameters(), shadow_parameters):
                    target_parameter.grad = source.grad.detach().cpu()
                shadow.step()
                for source, target_parameter in zip(model.parameters(), shadow_parameters):
                    close(source, target_parameter)
                    for name,value in opt.state[source].items():
                        close(value, shadow.state[target_parameter][name])
            close(actual, expected)
            close(loss, expected_loss)
            # Isolate the new loss derivative from existing half Conv/BatchNorm differences.
            reference_logits = actual.detach().cpu().requires_grad_()
            loss_fn(reference_logits, target).backward()
            close(actual.grad, reference_logits.grad)
            close(data.grad, host.grad)
            for (parameter_name,p), q in zip(model.named_parameters(), cpu.parameters()):
                try:
                    close(p, q)
                    close(p.grad, q.grad)
                    assert opt.state[p].keys() == opt_cpu.state[q].keys()
                    for name, value in opt.state[p].items():
                        close(value, opt_cpu.state[q][name])
                        if name == 'step':
                            assert value.device.type == 'cpu' and value.item() == index+1
                except AssertionError as error:
                    error.add_note(f'{kind} {dtype}, step={index+1}, parameter={parameter_name}; '
                        f'identical-gradient AdamW update checked={optimizer == "adamw"}')
                    raise
            for name, buffer in model.named_buffers():
                reference = dict(cpu.named_buffers())[name]
                if buffer.dtype == torch.int64:
                    assert torch.equal(buffer.cpu(), reference)
                else:
                    close(buffer, reference)
            losses.append(float(loss.detach().cpu()))
            copies.append(copied)
            scalar_uploads.append(scalars)
            if index == 2:
                state = {'model': {k:v.detach().cpu() for k,v in model.state_dict().items()}, 'optimizer': opt.state_dict()}
                state['optimizer']['state'] = {k:{n:v.detach().cpu() if isinstance(v,torch.Tensor) else v for n,v in s.items()}
                                               for k,s in state['optimizer']['state'].items()}
                stream = io.BytesIO()
                torch.save(state, stream)
                stream.seek(0)
                state = torch.load(stream, map_location='cpu', weights_only=True)
                restored = copy.deepcopy(cpu).to('intel_npu')
                restored.load_state_dict(state['model'])
                restored_opt = optimizer_type(restored.parameters(), **options)
                restored_opt.load_state_dict(state['optimizer'])
                model, opt = restored, restored_opt
        cpu.eval()
        model.eval()
        with torch.inference_mode():
            close(model(data), cpu(host))
        return {'dtype': str(dtype), 'kind': kind, 'losses': losses, 'step_host_copies': copies,
                'scalar_uploads': scalar_uploads, 'checkpoint_resume': True, 'independent_cpu_trajectory': True,
                'loss_gradient_on_identical_logits': True, 'optimizer':optimizer, 'optimizer_options':options,
                'identical_gradient_optimizer_check':optimizer == 'adamw', 'bn_bias':bn_bias}

    if case in ('logsoftmax', 'nll'):
        for dtype in (torch.float16, torch.float32):
            check(f'{case}-{dtype}', lambda d=dtype: logsoftmax(d) if case == 'logsoftmax' else nll(d))
    elif case == 'contracts':
        check('classification-contracts', contracts)
    else:
        for kind in ('mlp','conv','residual'):
            for dtype in (torch.float16,torch.float32):
                check(f'{kind}-{dtype}', lambda k=kind,d=dtype: train(k,d))
    gc.collect()
    before = npu.stats()
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    status = 'failed' if any(c['status'] == 'failed' for c in checks) else 'passed'
    return {'case':case, 'optimizer':optimizer, 'bn_bias':bn_bias, 'status':status, 'checks':checks,
            'stats_before_shutdown':before, 'final_stats':final, 'python':sys.version,
            'source_sha256': {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_classification_demo.py','npunlock/python/npunlock/npu4000.py',
                 'npunlock/tests/fixtures/npu4000/resident_classification.c','npunlock-probe/torch-device/_torch_intel_npu.pyd')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('logsoftmax','nll','contracts','training'), required=True)
    parser.add_argument('--optimizer', choices=('sgd','adamw'), default='sgd')
    parser.add_argument('--bn-bias', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.optimizer != 'sgd' and args.case != 'training':
        parser.error('--optimizer=adamw is only applicable to --case=training')
    output = OUT if args.optimizer == 'sgd' else OUT.parent/'adamw'
    output.mkdir(parents=True,exist_ok=True)
    result = execute(args.case, args.optimizer, args.bn_bias)
    filename = args.case if args.bn_bias else args.case+'-no-bn-bias'
    (output/f'{filename}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(result['status'] == 'failed')

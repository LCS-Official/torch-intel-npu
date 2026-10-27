"""NPU acceptance for grouped convolution, class-axis losses and AMP migration."""
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
OUT = ROOT/'npunlock-probe/research/results/portability'


def spatial_model():
    import torch
    return torch.nn.Sequential(
        torch.nn.Conv2d(2, 4, 3, padding=1, groups=2), torch.nn.SiLU(),
        torch.nn.Conv2d(4, 4, 3, padding=1, groups=4), torch.nn.GELU(),
        torch.nn.Conv2d(4, 3, 1))


def execute(case, precision):
    import torch
    import torch.nn.functional as F
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(337)
    torch.backends.mkldnn.enabled = False
    dtype = getattr(torch, precision)
    records = []

    def upload(x):
        return x.detach().to('intel_npu', memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)

    def close(a, b, exact=False):
        torch.testing.assert_close(a.detach().cpu(), b.detach(), equal_nan=True,
                                   rtol=0 if exact else .004 if dtype == torch.float16 else 4e-5,
                                   atol=0 if exact else .0005 if dtype == torch.float16 else 5e-6)

    def resident(before, flags=0):
        after = npu.stats()
        assert after['host_download_copies']-before['host_download_copies'] == flags
        assert after['host_device_copies'] == before['host_device_copies']
        assert after['host_upload_copies']-before['host_upload_copies'] == after['cached_scalar_buffers']-before['cached_scalar_buffers']

    def probe():
        x = torch.ones(2, 4, 4, 4, device='intel_npu')
        weight = torch.ones(4, 2, 3, 3, device='intel_npu')
        target = torch.zeros(2, 4, 4, dtype=torch.int64, device='intel_npu')
        actions = dict(grouped_conv=lambda: F.conv2d(x, weight, padding=1, groups=2),
                       channel_softmax=lambda: x.softmax(1), spatial_cross_entropy=lambda: F.cross_entropy(x, target),
                       rsub_alpha=lambda: torch.rsub(x, 2., alpha=.5))
        rows = []
        for name, operation in actions.items():
            try:
                operation()
                rows.append(dict(name=name, status='supported'))
            except Exception as error:
                rows.append(dict(name=name, status='unsupported', error=str(error).splitlines()[0]))
        return dict(probes=rows)

    def operators():
        total = 0
        for shape in ((3, 4), (2, 3, 4), (2, 3, 2, 4)):
            for axis in range(-len(shape), len(shape)):
                for logarithmic in (False, True):
                    host = (torch.randn(shape, dtype=dtype)/4).requires_grad_()
                    x = upload(host)
                    operation = torch.log_softmax if logarithmic else torch.softmax
                    seed = torch.randn(shape, dtype=dtype)/8
                    upstream = upload(seed)
                    before = npu.stats()
                    actual = operation(x, axis)
                    actual.backward(upstream)
                    resident(before)
                    expected = operation(host, axis)
                    expected.backward(seed)
                    close(actual, expected)
                    close(x.grad, host.grad)
                    with torch.no_grad():
                        out = torch.empty_like(x)
                        packet = torch.ops.aten._log_softmax.out if logarithmic else torch.ops.aten._softmax.out
                        assert packet(x, axis, False, out=out) is out
                        close(out, expected)
                    total += 1
        # Strided reads and exceptional rows must follow the same axis mapping.
        host = torch.tensor([[[0., -float('inf')], [float('inf'), float('nan')], [1., 2.]],
                             [[-float('inf'), 1.], [-float('inf'), 2.], [-float('inf'), 3.]]], dtype=dtype)
        x = upload(host)
        for operation in (torch.softmax, torch.log_softmax):
            close(operation(x.transpose(0, 2), 1), operation(host.transpose(0, 2), 1))
        for tensor_other in (False, True):
            for alpha in (1., .5, -.75, 0.):
                host = (torch.randn(2, 3, dtype=dtype)/4).requires_grad_()
                other = (torch.randn(1, 3, dtype=dtype)/4).requires_grad_() if tensor_other else 2.
                x, y = upload(host), upload(other) if tensor_other else other
                before = npu.stats()
                actual = torch.rsub(x, y, alpha=alpha)
                actual.sum().backward()
                resident(before)
                expected = torch.rsub(host, other, alpha=alpha)
                expected.sum().backward()
                close(actual, expected)
                close(x.grad, host.grad)
                if tensor_other:
                    close(y.grad, other.grad)
                with torch.no_grad():
                    out = torch.empty_like(x)
                    packet = torch.ops.aten.rsub.Tensor_out if tensor_other else torch.ops.aten.rsub.Scalar_out
                    assert packet(x, y, alpha=alpha, out=out) is out
                    close(out, expected)
        return dict(axis_softmax_cases=total, reverse_subtract_cases=8, no_payload_downloads=True)

    def convolution(stream=False):
        geometries = [(2, 4, 6, (5, 6), (3, 2), (2, 1), (1, 2), (1, 2), 2),
                      (1, 3, 6, (4, 5), (3, 3), (1, 1), (1, 1), (1, 1), 3),
                      (2, 4, 4, (3, 4), (1, 1), (1, 1), (0, 0), (1, 1), 4)]
        if stream:
            geometries = [(1, 4, 4, (82, 82), (9, 9), (1, 1), (4, 4), (1, 1), 2)]
        for batch, inputs, outputs, spatial, kernel, stride, padding, dilation, groups in geometries:
            for bias_enabled in ((True,) if stream else (False, True)):
                hosts = [(torch.randn(batch, inputs, *spatial, dtype=dtype)/16).requires_grad_(),
                         (torch.randn(outputs, inputs//groups, *kernel, dtype=dtype)/16).requires_grad_()]
                if bias_enabled:
                    hosts.append((torch.randn(outputs, dtype=dtype)/16).requires_grad_())
                tensors = [upload(h) for h in hosts]
                options = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
                expected = F.conv2d(*hosts, **options)
                seed = torch.randn_like(expected)/16
                upstream = upload(seed)
                before = npu.stats()
                actual = F.conv2d(*tensors, **options)
                actual.backward(upstream)
                resident(before)
                expected.backward(seed)
                close(actual, expected)
                for x, h in zip(tensors, hosts):
                    close(x.grad, h.grad)
                records.append(dict(groups=groups, input=list(hosts[0].shape), kernel=list(kernel), bias=bias_enabled,
                                    output_max_abs=float((actual.detach().cpu()-expected.detach()).abs().max())))
        # Saving original tensors preserves mutation detection across group copies.
        x = upload(torch.randn(1, 2, 3, 3, dtype=dtype).requires_grad_())
        w = upload(torch.randn(2, 1, 1, 1, dtype=dtype).requires_grad_())
        value = F.conv2d(x, w, groups=2)
        with torch.no_grad():
            w.add_(1.)
        try:
            value.sum().backward()
        except RuntimeError as error:
            assert 'modified by an inplace operation' in str(error)
        else:
            raise AssertionError('Grouped convolution lost the saved-weight version guard')
        return dict(configurations=len(records), original_version_guard=True, no_payload_downloads=True)

    def losses():
        configurations = 0
        for shape in ((4,), (3, 4), (2, 4, 3), (2, 4, 2, 3)):
            target_shape = () if len(shape) == 1 else (shape[0], *shape[2:])
            labels = torch.randint(0, 4, target_shape)
            if labels.numel() > 1:
                labels.view(-1)[1] = -100
            target = upload(labels)
            for reduction in ('none', 'mean', 'sum'):
                for negative_log_likelihood in (False, True):
                    raw = torch.randn(shape, dtype=dtype)/4
                    host = (raw.log_softmax(0 if len(shape) == 1 else 1) if negative_log_likelihood else raw).requires_grad_()
                    x = upload(host)
                    operation = F.nll_loss if negative_log_likelihood else F.cross_entropy
                    before = npu.stats()
                    actual = operation(x, target, reduction=reduction)
                    actual.sum().backward()
                    resident(before, 1)
                    expected = operation(host, labels, reduction=reduction)
                    expected.sum().backward()
                    close(actual, expected)
                    close(x.grad, host.grad)
                    configurations += 1
        for reduction in ('none', 'mean', 'sum'):
            host = torch.randn(2, 3, 2, 2, dtype=dtype).requires_grad_()
            labels = torch.full((2, 2, 2), -100, dtype=torch.int64)
            x, target = upload(host), upload(labels)
            actual = F.cross_entropy(x, target, reduction=reduction)
            actual.sum().backward()
            expected = F.cross_entropy(host, labels, reduction=reduction)
            expected.sum().backward()
            close(actual, expected)
            close(x.grad, host.grad)
        return dict(hard_label_cases=configurations, all_ignored_cases=3, flag_downloads_only=True)

    def smoothing():
        count = 0
        for shape in ((4,), (3, 4), (2, 4, 3), (2, 4, 2, 3)):
            label_shape = () if len(shape) == 1 else (shape[0], *shape[2:])
            labels = torch.randint(0, 4, label_shape)
            if labels.numel() > 1:
                labels.view(-1)[1] = -100
            target = upload(labels)
            for strength in (.1, 1.):
                for reduction in ('none', 'mean', 'sum'):
                    host = (torch.randn(shape, dtype=dtype)/4).requires_grad_()
                    x = upload(host)
                    before = npu.stats()
                    actual = F.cross_entropy(x, target, label_smoothing=strength, reduction=reduction)
                    actual.sum().backward()
                    resident(before, 1)
                    expected = F.cross_entropy(host, labels, label_smoothing=strength, reduction=reduction)
                    expected.sum().backward()
                    close(actual, expected)
                    close(x.grad, host.grad)
                    count += 1
        for strength in (.1, 1.):
            for reduction in ('none', 'mean', 'sum'):
                host = torch.randn(2, 3, 2, 2, dtype=dtype).requires_grad_()
                labels = torch.full((2, 2, 2), -100, dtype=torch.int64)
                x, target = upload(host), upload(labels)
                actual = F.cross_entropy(x, target, label_smoothing=strength, reduction=reduction)
                actual.sum().backward()
                expected = F.cross_entropy(host, labels, label_smoothing=strength, reduction=reduction)
                expected.sum().backward()
                close(actual, expected)
                close(x.grad, host.grad, True)
        return dict(smoothing_cases=count, all_ignored_cases=6, flag_downloads_only=True)

    def conv1d():
        for groups in (1, 2, 4):
            hosts = [(torch.randn(2, 4, 9, dtype=dtype)/8).requires_grad_(),
                     (torch.randn(8, 4//groups, 3, dtype=dtype)/8).requires_grad_(),
                     (torch.randn(8, dtype=dtype)/8).requires_grad_()]
            tensors = [upload(h) for h in hosts]
            options = dict(stride=2, padding=2, dilation=2, groups=groups)
            expected = F.conv1d(*hosts, **options)
            upstream = torch.randn_like(expected)/16
            gradient = upload(upstream)
            before = npu.stats()
            actual = F.conv1d(*tensors, **options)
            actual.backward(gradient)
            resident(before)
            expected.backward(upstream)
            close(actual, expected)
            for x, h in zip(tensors, hosts):
                close(x.grad, h.grad)
            with torch.no_grad():
                close(F.conv1d(*tensors, **options), expected)
        return dict(groups=[1, 2, 4], stride=2, dilation=2, fp32_opmath=True, no_payload_downloads=True)

    def axis_tiled():
        host = (torch.randn(257, 16, 34, dtype=dtype)/8).requires_grad_()
        x = upload(host)
        for logarithmic in (False, True):
            host.grad = x.grad = None
            operation = torch.log_softmax if logarithmic else torch.softmax
            seed = torch.randn(257, 16, 17, dtype=dtype)/16
            gradient = upload(seed)
            before = npu.stats()
            actual = operation(x[..., ::2], 1)
            actual.backward(gradient)
            resident(before)
            expected = operation(host[..., ::2], 1)
            expected.backward(seed)
            close(actual, expected)
            close(x.grad, host.grad)
        return dict(elements=257*16*17, axis=1, partial_view_gradient=True, complete_row_tiling=True)

    def models(amp=False, transformer=False):
        from torch_npu_transformer_demo import cpu_tree, token_classifier
        cpu = token_classifier(dropout=0.) if transformer else spatial_model()
        cpu = cpu.float() if amp else cpu.to(dtype)
        model = copy.deepcopy(cpu).to('intel_npu')
        inputs = torch.tensor([[1, 2, 0], [3, 4, 5]]) if transformer else torch.randn(2, 2, 4, 4)/4
        inputs = inputs if transformer else inputs.to(torch.float32 if amp else dtype)
        labels = torch.tensor([0, 2]) if transformer else torch.randint(0, 3, (2, 4, 4))
        data, target = upload(inputs), upload(labels)
        optimizers = [torch.optim.SGD(m.parameters(), lr=.01, momentum=.5, foreach=False, fused=False) for m in (model, cpu)]
        scalers = [torch.intel_npu.amp.GradScaler(init_scale=8., growth_interval=2),
                   torch.amp.GradScaler('cpu', init_scale=8., growth_interval=2)] if amp else None
        history, differences, checkpoint = [], [], io.BytesIO()
        for step in range(4):
            losses_at_step = []
            for index, (network, optimizer, x, y) in enumerate(zip((model, cpu), optimizers, (data, inputs), (target, labels))):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('intel_npu' if index == 0 else 'cpu', dtype=torch.float16, enabled=amp):
                    output = network(x)
                    loss = F.cross_entropy(output.float(), y, label_smoothing=.1)
                if amp:
                    scalers[index].scale(loss).backward()
                    scalers[index].unscale_(optimizer)
                else:
                    loss.backward()
                losses_at_step.append(loss)
            # AMP comparisons allow FP16 arithmetic differences, and record them.
            tolerance = dict(rtol=.006, atol=.0005) if amp or dtype == torch.float16 else dict(rtol=5e-5, atol=6e-6)
            torch.testing.assert_close(losses_at_step[0].detach().cpu(), losses_at_step[1].detach(), **tolerance)
            maximum = 0.
            for p, q in zip(model.parameters(), cpu.parameters()):
                assert p.grad is not None and q.grad is not None
                host_gradient = p.grad.cpu()
                torch.testing.assert_close(host_gradient, q.grad, **tolerance)
                maximum = max(maximum, float((host_gradient-q.grad).abs().max()))
                if amp:
                    assert p.dtype == p.grad.dtype == torch.float32
            differences.append(maximum)
            for index, optimizer in enumerate(optimizers):
                if amp:
                    scalers[index].step(optimizer)
                    scalers[index].update()
                else:
                    optimizer.step()
            for p, q in zip(model.parameters(), cpu.parameters()):
                torch.testing.assert_close(p.detach().cpu(), q.detach(), **tolerance)
            if amp:
                assert scalers[0].state_dict() == scalers[1].state_dict()
            history.append(float(losses_at_step[0].detach().cpu()))
            print('training', 'transformer' if transformer else 'spatial', 'amp' if amp else precision,
                  step+1, history[-1], flush=True)
            if step == 2:
                torch.save(dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizers[0].state_dict()),
                                scaler=scalers[0].state_dict() if amp else None), checkpoint)
        checkpoint.seek(0)
        state = torch.load(checkpoint, weights_only=True)
        restored = (token_classifier(dropout=0.) if transformer else spatial_model()).to(torch.float32 if amp else dtype).to('intel_npu')
        restored.load_state_dict(state['model'])
        optimizer = torch.optim.SGD(restored.parameters(), lr=.01, momentum=.5, foreach=False, fused=False)
        optimizer.load_state_dict(state['optimizer'])
        if amp:
            scaler = torch.intel_npu.amp.GradScaler()
            scaler.load_state_dict(state['scaler'])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('intel_npu', dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(restored(data).float(), target, label_smoothing=.1)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            assert scaler.state_dict() == scalers[0].state_dict()
        else:
            loss.backward()
            optimizer.step()
        assert float(loss.detach().cpu()) == history[-1]
        for p, q in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(p.detach().cpu(), q.detach().cpu(), rtol=0, atol=0)
        for p, q in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(optimizers[0].state[p]['momentum_buffer'].cpu(), optimizer.state[q]['momentum_buffer'].cpu(), rtol=0, atol=0)
        return dict(steps=4, losses=history, max_gradient_abs=differences, comparison_tolerance=tolerance,
                    parameters=sum(p.numel() for p in model.parameters()), amp=amp, checkpoint_next_step_bitwise=True)

    def contracts():
        x = torch.ones(1, 4, 4, 4, dtype=dtype, device='intel_npu')
        w = torch.ones(4, 2, 3, 3, dtype=dtype, device='intel_npu')
        target = torch.zeros(1, 4, 4, dtype=torch.int64, device='intel_npu')
        bad_shape = target[:1, :3].clone()
        class_weight = torch.ones(3, dtype=dtype, device='intel_npu')
        actions = [lambda: F.conv2d(x, w, groups=3), lambda: F.conv2d(x, w, groups=0),
                   lambda: F.conv_transpose2d(x, w, groups=3), lambda: x.softmax(4),
                   lambda: F.cross_entropy(x, target, label_smoothing=1.1),
                   lambda: F.cross_entropy(x, target, weight=class_weight),
                   lambda: F.cross_entropy(x, bad_shape),
                   lambda: torch.rsub(x, 1., alpha=True)]
        for index,action in enumerate(actions):
            before = npu.stats()
            try:
                action()
            except (ValueError, IndexError, RuntimeError, NotImplementedError, TypeError):
                pass
            else:
                raise AssertionError(f'Invalid configuration accepted at action {index}')
            after = npu.stats()
            assert after['npu_calls'] == before['npu_calls']
        bad = upload(torch.tensor([[[0, 0], [0, 2**40]]], dtype=torch.int64))
        try:
            F.cross_entropy(x[:, :, :2, :2].clone(), bad)
        except IndexError:
            pass
        else:
            raise AssertionError('Out-of-range spatial target accepted')
        return dict(rejected=len(actions)+1, validation_before_compute=True)

    actions = dict(probe=probe, operators=operators, conv=convolution, losses=losses, smoothing=smoothing,
                   conv1d=conv1d, models=models, **{'axis-tiled': axis_tiled},
                   contracts=contracts, **{'conv-stream': lambda: convolution(True), 'conv-amp': lambda: models(True),
                                           'transformer-amp': lambda: models(True, True)})
    before = npu.stats()
    try:
        detail = actions[case]()
        status = 'passed'
    except Exception as error:
        detail = dict(error=str(error), traceback=traceback.format_exc())
        status = 'failed'
    gc.collect()
    after = npu.stats()
    calls = after['npu_calls']-before['npu_calls']
    if after['live_buffers']-after['cached_scalar_buffers'] != before['live_buffers']-before['cached_scalar_buffers']:
        status = 'failed'
        detail['lifecycle_error'] = 'Leaked tensor storage'
    print(case, precision, status, detail.get('error', '')[:700], flush=True)
    npu.shutdown()
    final = npu.stats()
    if final['context_failed'] or any(final[k] for k in (
            'live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'cached_sessions', 'quarantined_buffer_bytes')):
        status = 'failed'
    sources = ('torch_intel_npu.py', 'torch_npu_autograd.py', 'torch_npu_portability_demo.py',
               'npunlock-probe/torch-device/backend.cpp', 'npunlock-probe/torch-device/_torch_intel_npu.pyd')
    return dict(status=status, case=case, precision=precision, npu_calls=calls, details=detail,
                checks=records, final_stats=final,
                source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('probe', 'operators', 'conv', 'conv-stream', 'losses', 'smoothing',
                                         'conv1d', 'models', 'axis-tiled', 'contracts', 'conv-amp', 'transformer-amp'), default='probe')
    parser.add_argument('--precision', choices=('float16', 'float32'), default='float32')
    parser.add_argument('--label', default='after')
    args = parser.parse_args()
    result = execute(args.case, args.precision)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{args.label}-{args.case}-{args.precision}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    raise SystemExit(result['status'] != 'passed')

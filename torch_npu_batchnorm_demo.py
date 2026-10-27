"""INT64 storage/counters and ordinary default BatchNorm migration checks."""
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
OUT = ROOT / 'npunlock-probe/research/results/int64-batchnorm'


def execute(case):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(47)
    torch.backends.mkldnn.enabled = False
    checks = []

    def close(actual, expected):
        if actual.dtype == torch.int64:
            assert actual.dtype == expected.dtype
            assert torch.equal(actual.detach().cpu(), expected.detach().cpu()), 'INT64 values differ'
            return
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(),
                                   rtol=.005 if actual.dtype == torch.float16 else 1e-4,
                                   atol=.0008 if actual.dtype == torch.float16 else 1e-5)

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
        row['npu_calls'] = after['npu_calls'] - before['npu_calls']
        row['live_buffer_delta'] = (after['live_buffers'] - after['cached_scalar_buffers'] -
                                    before['live_buffers'] + before['cached_scalar_buffers'])
        if row['live_buffer_delta'] or after['context_failed']:
            row.update(status='failed', lifecycle_error='Live buffer leak or failed context')
        checks.append(row)
        print(name, row['status'], row.get('error', '')[:200], flush=True)

    def storage():
        samples = [0, 1, -1, 2**24+1, 2**32-1, 2**53+1, -(2**53+1), 2**63-1, -2**63,
                   0x7fc1234580000000, 0x7f80000000000000]
        for shape in ((), (1,), (1, 1), (1, 1, 1), (1, 1, 1, 1)):
            x = torch.tensor(samples[5], dtype=torch.int64).reshape(shape).to('intel_npu')
            assert x.dtype == torch.int64 and x.device.type == 'intel_npu'
            assert x.item() == samples[5]
            close(x.clone(), torch.full(shape, samples[5], dtype=torch.int64))
        cpu = torch.tensor(samples, dtype=torch.int64)
        x = cpu.to('intel_npu')
        close(x, cpu)
        clone = x.clone()
        assert clone.data_ptr() != x.data_ptr()
        close(clone, cpu)
        clone.copy_(torch.tensor(list(reversed(samples)), dtype=torch.int64))
        x.copy_(clone)
        close(x, torch.tensor(list(reversed(samples)), dtype=torch.int64))
        noncontiguous = torch.arange(12, dtype=torch.int64).reshape(3, 4).t()
        dense = noncontiguous.to('intel_npu', memory_format=torch.contiguous_format)
        close(dense, noncontiguous)
        for value in (0, 1, -3, 2**53+1, -2**63, 0x7fc1234580000000):
            full = torch.full((3, 4), value, dtype=torch.int64, device='intel_npu')
            close(full, torch.full((3, 4), value, dtype=torch.int64))
        close(torch.zeros(3, dtype=torch.int64, device='intel_npu'), torch.zeros(3, dtype=torch.int64))
        close(torch.ones(3, dtype=torch.int64, device='intel_npu'), torch.ones(3, dtype=torch.int64))
        full.zero_()
        close(full, torch.zeros(3, 4, dtype=torch.int64))
        maximum = torch.full((npu.MAX_INT64_ELEMENTS,), -1, dtype=torch.int64, device='intel_npu')
        close(maximum.clone(), torch.full((npu.MAX_INT64_ELEMENTS,), -1, dtype=torch.int64))
        return {'exact_roundtrip_values': samples, 'max_elements': npu.MAX_INT64_ELEMENTS,
                'factories_copy_clone_zero_item_and_cpu_dense_pack': True}

    def counters():
        pairs = [(0, 1), (2**24, 1), (2**32-1, 1), (2**53, 1), (2**63-1, 1),
                 (-2**63, -1), (-1, 1), (0, -1), (2**53+17, -23),
                 (0, 0x7fc1234580000000), (0x7fc1234580000000, 0x7f80000000000000)]
        for base, increment in pairs:
            cpu = torch.tensor(base, dtype=torch.int64)
            x = cpu.to('intel_npu')
            version = x._version
            before = npu.stats()['npu_calls']
            cpu.add_(increment)
            assert x.add_(increment) is x  # Integer counters update with grad mode enabled.
            assert x._version == version + 1
            assert npu.stats()['npu_calls'] - before == 2
            close(x, cpu)
        x = torch.tensor(2**53, dtype=torch.int64).to('intel_npu')
        for _ in range(20):
            x.add_(1)
        assert x.item() == 2**53+20
        program = npu._session(('int64_add', 'f32', (2,)))[1]
        return {'exact_pairs': pairs, 'repeat_at_2pow53': 20, 'signed_wraparound_and_carry': True,
                'carrier': json.loads(program.patch_reports[0])}

    def contracts():
        x = torch.tensor(0, dtype=torch.int64).to('intel_npu')
        array = torch.zeros(2, dtype=torch.int64).to('intel_npu')
        floating = torch.ones(2).to('intel_npu')
        before = npu.stats()['npu_calls']
        rejected = []
        for name, function in {
            'integer-capacity': lambda: torch.empty(npu.MAX_INT64_ELEMENTS+1, dtype=torch.int64, device='intel_npu'),
            'empty': lambda: torch.empty(0, dtype=torch.int64, device='intel_npu'),
            'integer-cast': lambda: array.float(),
            'float-cast-to-int': lambda: floating.long(),
            'integer-tensor-add': lambda: array+array,
            'integer-array-add-inplace': lambda: array.add_(1),
            'integer-float-increment': lambda: x.add_(1.5),
            'integer-alpha': lambda: x.add_(1, alpha=2),
            'integer-bool-increment': lambda: x.add_(True),
            'integer-float-fill': lambda: array.fill_(1.5),
            'integer-item-array': lambda: array.item(),
            'integer-copy-float': lambda: array.copy_(floating),
            'integer-nonblocking': lambda: array.to('cpu', non_blocking=True),
            'integer-transpose-copy': lambda: torch.empty((2, 2), dtype=torch.int64, device='intel_npu').t().copy_(torch.ones(2, 2, dtype=torch.int64)),
        }.items():
            try:
                function()
            except (ValueError, TypeError, RuntimeError, NotImplementedError):
                rejected.append(name)
            else:
                raise AssertionError(f'Accepted {name}')
        assert before == npu.stats()['npu_calls']
        assert x.item() == 0
        return {'rejected_without_graph_submit': rejected}

    def train(dtype, momentum, affine):
        cpu = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(2, momentum=momentum, affine=affine), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(2, 2)).to(dtype=dtype)
        model = copy.deepcopy(cpu).to('intel_npu')
        assert model[1].num_batches_tracked.device.type == 'intel_npu'
        assert model[1].num_batches_tracked.dtype == torch.int64
        opt_cpu = torch.optim.SGD(cpu.parameters(), lr=.01, momentum=.8, foreach=False, fused=False)
        opt = torch.optim.SGD(model.parameters(), lr=.01, momentum=.8, foreach=False, fused=False)
        target = torch.tensor([[.2, -.1], [.1, .3]], dtype=dtype)
        labels = target.to('intel_npu')
        losses, copies, scalar_uploads = [], [], []
        for index in range(4):
            host = (torch.linspace(-.5, .7, 32).reshape(2, 1, 4, 4) + index*.1).to(dtype).requires_grad_()
            data = host.detach().to('intel_npu').requires_grad_()
            opt.zero_grad(set_to_none=False)
            opt_cpu.zero_grad(set_to_none=False)
            expected = cpu(host)
            loss_cpu = torch.nn.functional.mse_loss(expected.float(), target.float()).to(dtype)
            loss_cpu.backward()
            opt_cpu.step()
            before = npu.stats()
            actual = model(data)
            loss = torch.nn.functional.mse_loss(actual, labels)
            loss.backward()
            opt.step()
            after = npu.stats()
            copied = sum(after[k] - before[k] for k in ('host_upload_copies','host_download_copies','host_device_copies'))
            new_scalars = after['cached_scalar_buffers'] - before['cached_scalar_buffers']
            assert copied == new_scalars + int(momentum is None), (copied, new_scalars, momentum)
            close(actual, expected)
            close(loss, loss_cpu)
            close(data.grad, host.grad)
            for p, q in zip(model.parameters(), cpu.parameters()):
                close(p, q)
                close(p.grad, q.grad)
                close(opt.state[p]['momentum_buffer'], opt_cpu.state[q]['momentum_buffer'])
            for name, buffer in model.named_buffers():
                close(buffer, dict(cpu.named_buffers())[name])
            assert model[1].num_batches_tracked.item() == index+1
            losses.append(float(loss.detach().cpu()))
            copies.append(copied)
            scalar_uploads.append(new_scalars)
            if index == 2:
                checkpoint = {'model': {k: v.detach().cpu() for k,v in model.state_dict().items()}, 'optimizer': opt.state_dict()}
                checkpoint['optimizer']['state'] = {k: {n: v.detach().cpu() if isinstance(v, torch.Tensor) else v for n,v in state.items()}
                                                    for k,state in checkpoint['optimizer']['state'].items()}
                stream = io.BytesIO()
                torch.save(checkpoint, stream)
                stream.seek(0)
                state = torch.load(stream, map_location='cpu', weights_only=True)
                restored = copy.deepcopy(cpu).to('intel_npu')
                restored.load_state_dict(state['model'])
                restored_opt = torch.optim.SGD(restored.parameters(), lr=.01, momentum=.8, foreach=False, fused=False)
                restored_opt.load_state_dict(state['optimizer'])
                model, opt = restored, restored_opt
        model.eval()
        cpu.eval()
        counter = model[1].num_batches_tracked.item()
        with torch.inference_mode():
            close(model(data.detach()), cpu(host.detach()))
        assert model[1].num_batches_tracked.item() == counter == 4
        model[1].reset_running_stats()
        assert model[1].num_batches_tracked.item() == 0
        close(model[1].running_mean, torch.zeros_like(cpu[1].running_mean))
        close(model[1].running_var, torch.ones_like(cpu[1].running_var))
        return {'dtype': str(dtype), 'momentum': momentum, 'affine': affine, 'losses': losses,
                'step_host_copies': copies, 'scalar_uploads': scalar_uploads, 'checkpoint_resume': True,
                'running_statistics_counter_eval_and_reset': True}

    if case == 'integer':
        check('int64-storage', storage)
        check('int64-shave-counters', counters)
        check('int64-contracts', contracts)
    else:
        for dtype in (torch.float16, torch.float32):
            for momentum, affine in ((.1, True), (None, True), (.1, False)):
                check(f'batchnorm-{dtype}-momentum-{momentum}-affine-{affine}',
                      lambda d=dtype, m=momentum, a=affine: train(d, m, a))
    gc.collect()
    before_shutdown = npu.stats()
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    return {'case': case, 'status': 'failed' if any(c['status'] != 'passed' for c in checks) else 'passed',
            'checks': checks, 'stats_before_shutdown': before_shutdown, 'final_stats': final,
            'python': sys.version, 'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('torch_intel_npu.py','torch_npu_batchnorm_demo.py','npunlock-probe/torch-device/backend.cpp',
                         'npunlock-probe/torch-device/_torch_intel_npu.pyd','npunlock/python/npunlock/npu4000.py',
                         'npunlock/tests/fixtures/npu4000/resident_integer.c',
                         'npunlock/tests/fixtures/npu4000/resident/int64-add.elf')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('integer','models'), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result = execute(args.case)
    (OUT/f'{args.case}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    raise SystemExit(result['status'] != 'passed')

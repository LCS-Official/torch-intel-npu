"""S26 real-device storage, rank, large arithmetic, softmax and Linear training."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback


def execute(case, precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(726)
    dtype = getattr(torch, precision)
    started = time.perf_counter()
    checks = []

    def upload(x):
        return x.detach().to('intel_npu', memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)

    def close(actual, expected, *, exact=False):
        a, b = actual.detach().cpu(), expected.detach()
        if exact:
            assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
        else:
            torch.testing.assert_close(a, b, rtol=4e-5 if dtype == torch.float32 else .004,
                                       atol=2e-6 if dtype == torch.float32 else .001, equal_nan=True)

    def rejects(fn):
        before = npu.stats()['npu_calls']
        try:
            fn()
        except (ValueError, RuntimeError):
            assert npu.stats()['npu_calls'] == before
        else:
            raise AssertionError('Invalid input was accepted')

    def storage():
        boundaries = {}
        for dt in npu._STORAGE_DTYPES:
            count = npu._storage_limit(dt)
            value = torch.empty(count, dtype=dt, device='intel_npu')
            boundaries[str(dt)] = value.untyped_storage().nbytes()
            rejects(lambda: torch.empty(count+1, dtype=dt, device='intel_npu'))
            close(npu._slice(value, count-3, 3), torch.zeros(3, dtype=dt), exact=True)
            del value
        count = npu._storage_limit(torch.float32)
        host = torch.zeros(count, dtype=torch.float32)
        host[:3], host[-3:] = torch.tensor([1., -0., 3.]), torch.tensor([-7., 123.25, -9.])
        value = upload(host)
        close(npu._slice(value, 0, 3), host[:3], exact=True)
        close(npu._slice(value, count-3, 3), host[-3:], exact=True)
        rejects(lambda: torch.empty((1,)*(npu.MAX_RANK+1), device='intel_npu'))
        gradient = upload(torch.ones(1))
        indices = upload(torch.zeros(1, dtype=torch.int64))
        rejects(lambda: npu._selection_backward(gradient, indices, (count+1,), 1, count+1))
        assert npu.stats()['max_allocation_bytes'] == npu.MAX_ALLOCATION_BYTES
        return dict(boundary_bytes=boundaries, full_fp32_upload_bytes=value.untyped_storage().nbytes(),
                    boundary_dma_readback=True, oversize_rejected_before_submission=True)

    def rank():
        shape = (2,)+(1,)*62+(3,)
        host = torch.arange(6, dtype=dtype).reshape(shape).requires_grad_()
        value = upload(host)
        close(value.clone(), host, exact=True)
        close(value.transpose(0, -1).contiguous(), host.transpose(0, -1).contiguous(), exact=True)
        actual, expected = (value*.25+.5).square().sum(), (host*.25+.5).square().sum()
        close(actual, expected)
        actual.backward()
        expected.backward()
        close(value.grad, host.grad)
        host.grad = value.grad = None
        weight = torch.randn(4, 3, dtype=dtype, requires_grad=True)*.1
        weight = weight.detach().requires_grad_()
        resident_weight = upload(weight)
        actual, expected = torch.nn.functional.linear(value, resident_weight), torch.nn.functional.linear(host, weight)
        close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        close(value.grad, host.grad)
        close(resident_weight.grad, weight.grad)
        close(torch.softmax(value.detach(), -1), torch.softmax(host.detach(), -1))
        return dict(rank=64, clone=True, permutation=True, arithmetic_and_backward=True,
                    linear_and_gradients=True, softmax=True)

    def pointwise():
        count = 16777216+17
        host = torch.linspace(-.5, .5, count, dtype=torch.float32).to(dtype).requires_grad_()
        value = upload(host)
        actual = (value*.25+.125).square().sum(dtype=torch.float32)
        expected = (host*.25+.125).square().sum(dtype=torch.float32)
        close(actual, expected)
        actual.backward()
        expected.backward()
        close(value.grad, host.grad)
        close(value.clone(), host, exact=True)
        return dict(elements=count, input_bytes=value.untyped_storage().nbytes(),
                    reduction_and_backward=True, loss=actual.item())

    def softmax():
        shape = (1,)*7+(4097, 256)
        host = (torch.randn(shape, dtype=dtype)*.1).requires_grad_()
        gradient = torch.randn(shape, dtype=dtype)*.01
        value = upload(host)
        for fn in (torch.softmax, torch.log_softmax):
            value.grad = host.grad = None
            actual, expected = fn(value, -1), fn(host, -1)
            close(actual, expected)
            actual.backward(upload(gradient))
            expected.backward(gradient)
            close(value.grad, host.grad)
        rejects(lambda: torch.softmax(torch.empty(1, 65537, device='intel_npu', dtype=dtype), -1))
        return dict(shape=shape, elements=host.numel(), softmax_and_log_softmax_backward=True,
                    row_width_bound_preserved=True)

    def model():
        cpu = torch.nn.Linear(4096, 4160, dtype=dtype)
        device = copy.deepcopy(cpu).to('intel_npu')
        host = torch.randn(1, 4096, dtype=dtype)*.1
        target = torch.randn(1, 4160, dtype=dtype)*.1
        value, labels = upload(host), upload(target)
        opts = [torch.optim.SGD(net.parameters(), lr=3., foreach=False, fused=False) for net in (cpu, device)]
        losses = []
        for _ in range(2):
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            actual = torch.nn.functional.mse_loss(device(value), labels)
            expected = torch.nn.functional.mse_loss(cpu(host), target)
            close(actual, expected)
            actual.backward()
            expected.backward()
            for a, b in zip(device.parameters(), cpu.parameters()):
                close(a.grad, b.grad)
            for opt in opts:
                opt.step()
            for a, b in zip(device.parameters(), cpu.parameters()):
                close(a, b)
            losses.append(actual.item())
            print('model step', len(losses), 'loss', losses[-1], 'graphs', npu.stats()['cached_sessions'], flush=True)
        assert losses[-1] < losses[0]
        return dict(parameters=sum(p.numel() for p in device.parameters()),
                    weight_bytes=device.weight.untyped_storage().nbytes(), independent_steps=2,
                    losses=losses, all_parameter_gradients_and_updates=True)

    actions = dict(storage=storage, rank=rank, pointwise=pointwise, softmax=softmax, model=model)
    row = dict(name=case)
    try:
        row.update(actions[case]())
        row['status'] = 'passed'
    except Exception as error:
        row.update(status='failed', error=str(error), traceback=traceback.format_exc())
    checks.append(row)
    print(case, precision, row['status'], row.get('error', '')[:320], flush=True)
    gc.collect()
    before = npu.stats()
    npu.shutdown()
    gc.collect()
    final = npu.stats()
    healthy = not final['context_failed'] and all(final[k] == 0 for k in (
        'live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'cached_sessions', 'quarantined_buffer_bytes'))
    root = Path(__file__).resolve().parent
    sources = ('torch_intel_npu.py', 'torch_npu_capacity_demo.py', 'npunlock/python/npunlock/npu4000.py',
               'npunlock/src/graphinfer/session_win32.c', 'npunlock-probe/torch-device/backend.cpp',
               'npunlock-probe/torch-device/_torch_intel_npu.pyd', 'npunlock-probe/session-runtime/npunlock_session.dll')
    result = dict(status='passed' if healthy and row['status'] == 'passed' else 'failed',
                  precision=precision, checks=checks, stats=before, shutdown=final,
                  elapsed_seconds=time.perf_counter()-started,
                  source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources})
    out = root/'npunlock-probe/research/results/capacity-model'
    out.mkdir(exist_ok=True)
    (out/f'{case}-{precision}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('storage', 'rank', 'pointwise', 'softmax', 'model'), required=True)
    parser.add_argument('--precision', choices=('float32', 'float16'), default='float32')
    args = parser.parse_args()
    raise SystemExit(execute(args.case, args.precision))

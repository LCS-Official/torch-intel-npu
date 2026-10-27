"""Vector matmul, synchronous scalar extraction and ordinary-model migration checks."""
import argparse
import copy
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/vector-migration'
CASES = ('vectors', 'large', 'scalar', 'contracts', 'training', 'gaps')


def execute(case):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(107)
    rows = []

    def upload(x):
        return x.detach().to('intel_npu', memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)

    def close(actual, expected):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(),
                                   rtol=.003 if actual.dtype == torch.float16 else 3e-5,
                                   atol=.0005 if actual.dtype == torch.float16 else 3e-6, equal_nan=True)

    def check(name, function):
        before = npu.stats()
        row = dict(name=name)
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
            row.update(status='failed', lifecycle_error='Live buffers or failed context')
        rows.append(row)
        print(name, row['status'], row.get('error', '')[:240], flush=True)

    def vectors(dtype, large=False):
        shapes = (((65537,), (65537,)), ((2, 1, 65537), (65537,))) if large else (
            ((5,), (5,)), ((5,), (5, 3)), ((3, 5), (5,)),
            ((5,), (2, 5, 3)), ((2, 3, 5), (5,)),
            ((5,), (2, 3, 5, 4)), ((2, 3, 4, 5), (5,)), ((1,), (1,)))
        count, out_count = 0, 0
        for ashape, bshape in shapes:
            for needs in (((True, True),) if large else ((True, True), (True, False), (False, True))):
                for dense in ((False,) if large else (False, True)):
                    a, b = [(torch.randn(s, dtype=dtype)/8).requires_grad_(need)
                            for s, need in zip((ashape, bshape), needs)]
                    x, y = upload(a), upload(b)
                    if dense:
                        x, y = [v.detach().transpose(-2, -1).contiguous().transpose(-2, -1).requires_grad_(need)
                                if v.ndim > 1 else v for v, need in zip((x, y), needs)]
                    expected = a@b
                    gradient = torch.randn_like(expected)/16
                    seed = upload(gradient)
                    before = npu.stats()
                    actual = x@y
                    actual.backward(seed)
                    after = npu.stats()
                    assert after['npu_calls'] > before['npu_calls']
                    assert all(after[k] == before[k] for k in
                               ('host_upload_copies', 'host_download_copies', 'host_device_copies'))
                    expected.backward(gradient)
                    close(actual, expected)
                    for value, reference, need in zip((x, y), (a, b), needs):
                        if need:
                            assert value.grad.device.type == 'intel_npu' and value.grad.shape == reference.shape
                            close(value.grad, reference.grad)
                        else:
                            assert value.grad is None
                    with torch.no_grad():
                        functions = [torch.matmul]
                        if len(ashape) == len(bshape) == 1:
                            functions.append(torch.dot)
                        if len(ashape) == 2 and len(bshape) == 1:
                            functions.append(torch.mv)
                        for function in functions:
                            out = torch.empty_like(actual)
                            assert function(x, y, out=out) is out
                            close(out, expected)
                            out_count += 1
                    if not large and (x.ndim, y.ndim) in ((1, 1), (2, 1)):
                        x.grad = y.grad = None
                        result = (torch.dot if x.ndim == 1 else torch.mv)(x, y)
                        result.backward(seed)
                        for value, reference, need in zip((x, y), (a, b), needs):
                            if need:
                                close(value.grad, reference.grad)
                    count += 1
        return dict(precision=str(dtype), forward_backward_cases=count, out_cases=out_count,
                    complete_dense_permutations=True, no_compute_host_payload_copies=True)

    def scalar(dtype):
        count = 0
        values = [0., -0., .5, -2., float('inf'), -float('inf'), float('nan')]
        values += [2**-24, 65504.] if dtype == torch.float16 else [2**-149, 3.4028234663852886e38]
        for value in values:
            for shape in ((), (1,), (1, 1, 1, 1)):
                host = torch.tensor(value, dtype=dtype).reshape(shape)
                x = upload(host.requires_grad_())
                version, before = x._version, npu.stats()
                result = x.item()
                after = npu.stats()
                assert type(result) is float
                expected = host.item()
                assert math.isnan(result) if math.isnan(expected) else result == expected
                if expected == 0:
                    assert math.copysign(1., result) == math.copysign(1., expected)
                assert after['host_download_copies']-before['host_download_copies'] == 1
                assert all(after[k] == before[k] for k in ('npu_calls', 'host_upload_copies', 'host_device_copies'))
                assert x._version == version and x.grad is None
                count += 1
        x = upload(torch.tensor(.5, dtype=dtype, requires_grad=True))
        assert float(x.detach()) == .5
        assert bool(x.detach())
        assert int(x.detach()) == 0
        assert not bool(upload(torch.tensor(0., dtype=dtype)))
        (x*x).backward()
        close(x.grad, torch.tensor(1., dtype=dtype))
        for value in (-(1 << 63), (1 << 53)+1, (1 << 63)-1):
            assert upload(torch.tensor(value, dtype=torch.int64)).item() == value
        return dict(precision=str(dtype), scalar_cases=count, python_conversions=True,
                    exactly_one_download_per_item=True, integer_item_preserved=True,
                    live_autograd_graph_preserved=True)

    def contracts():
        a, b = upload(torch.ones(3)), upload(torch.ones(3))
        matrix = upload(torch.ones(2, 3))
        bad = upload(torch.ones(2))
        scalar = upload(torch.tensor(1.))
        half = upload(torch.ones(3, dtype=torch.float16))
        wrong_out = torch.empty(1, device='intel_npu')
        wrong_dtype = torch.empty((), dtype=torch.float16, device='intel_npu')
        differentiable = upload(torch.ones(3, requires_grad=True))
        identity = upload(torch.eye(3))
        actions = {
            'scalar-matmul': lambda: scalar@scalar,
            'inner-dim': lambda: a@bad,
            'dtype': lambda: a@half,
            'cpu-input': lambda: a@torch.ones(3),
            'dot-rank': lambda: torch.dot(matrix, b),
            'mv-rank': lambda: torch.mv(a, b),
            'out-shape': lambda: torch.matmul(a, b, out=wrong_out),
            'out-dtype': lambda: torch.dot(a, b, out=wrong_dtype),
            'differentiable-out': lambda: torch.matmul(differentiable, b, out=scalar),
            'out-alias': lambda: torch.matmul(a, identity, out=a),
            'partial-scalar': lambda: torch.ops.aten._local_scalar_dense(a[:1]),
            'non-scalar-item': lambda: a.item(),
            'partial-vector-shape': lambda: a[:2]@b,
        }
        errors = {}
        before = npu.stats()
        for name, action in actions.items():
            try:
                action()
            except (ValueError, RuntimeError, NotImplementedError) as error:
                errors[name] = str(error).splitlines()[0][:220]
            else:
                raise AssertionError(f'Expected rejection: {name}')
        after = npu.stats()
        assert after['npu_calls'] == before['npu_calls']
        assert all(after[k] == before[k] for k in ('host_upload_copies', 'host_download_copies', 'host_device_copies'))
        x, y = upload(torch.ones(3, requires_grad=True)), upload(torch.ones(3, requires_grad=True))
        output = torch.dot(x, y)
        with torch.no_grad():
            x.add_(y)
        seed = upload(torch.tensor(1.))
        before = npu.stats()['npu_calls']
        try:
            output.backward(seed)
        except RuntimeError as error:
            assert 'modified by an inplace operation' in str(error), str(error)
        else:
            raise AssertionError('Missing saved-tensor version guard')
        assert npu.stats()['npu_calls'] == before
        return dict(rejections=errors, rejection_count=len(errors), no_submission_on_rejection=True, version_guard=True)

    def training():
        class VectorHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(5, 4)
                self.direction = torch.nn.Parameter(torch.linspace(-.3, .4, 4))

            def forward(self, x):
                return torch.relu(self.projection(x))@self.direction
        cpu = VectorHead()
        model = copy.deepcopy(cpu).to('intel_npu')
        host = torch.randn(2, 3, 5)/4
        target = torch.randn(2, 3)/8
        x, labels = upload(host), upload(target)
        reference, opt = [torch.optim.SGD(net.parameters(), lr=.02, momentum=.5, foreach=False, fused=False)
                          for net in (cpu, model)]
        losses = []
        for _ in range(4):
            reference.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            expected = torch.nn.functional.mse_loss(cpu(host), target)
            actual = torch.nn.functional.mse_loss(model(x), labels)
            logged = actual.item()
            expected.backward()
            actual.backward()
            close(actual, expected)
            for p, q in zip(model.parameters(), cpu.parameters()):
                close(p.grad, q.grad)
            reference.step()
            opt.step()
            for p, q in zip(model.parameters(), cpu.parameters()):
                close(p, q)
            losses.append(logged)
        return dict(precision='float32', independent_training_steps=4, losses=losses, normal_loss_item=True,
                    batch_vector_gradient_reduced=True)

    def gaps():
        x = upload(torch.ones(2, 3))
        actions = {
            'async-copy': lambda: x.to('cpu', non_blocking=True),
            'native-save': lambda: torch.save(x, io.BytesIO()),
        }
        result = {}
        for name, action in actions.items():
            try:
                value = action()
            except (ValueError, RuntimeError, NotImplementedError) as error:
                result[name] = dict(status='blocked', error=str(error).splitlines()[0][:300])
            else:
                del value
                result[name] = dict(status='observed-only-not-certified')
        assert all(row['status'] == 'blocked' for row in result.values()), result
        return dict(migration_gaps=result)

    for name in CASES if case == 'all' else (case,):
        if name in ('vectors', 'large', 'scalar'):
            for dtype in (torch.float16, torch.float32):
                check(f'{name}-{dtype}', lambda d=dtype, n=name: scalar(d) if n == 'scalar' else vectors(d, n == 'large'))
        else:
            check(name, {'contracts': contracts, 'training': training, 'gaps': gaps}[name])
    gc.collect()
    npu.shutdown()
    final = npu.stats()
    assert final['closed']
    assert all(final[k] == 0 for k in ('live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
    assert not final['context_failed']
    sources = ('torch_npu_vector_demo.py', 'torch_intel_npu.py', 'torch_npu_autograd.py',
               'npunlock-probe/torch-device/_torch_intel_npu.pyd')
    return dict(status='passed' if all(r['status'] == 'passed' for r in rows) else 'failed', checks=rows,
                final_stats=final, torch_version=torch.__version__,
                sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all',)+CASES, default='all')
    parser.add_argument('--label', default='after')
    args = parser.parse_args()
    result = execute(args.case)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{args.label}-{args.case}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    raise SystemExit(result['status'] != 'passed')

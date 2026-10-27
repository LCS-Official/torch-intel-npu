"""Check Tensor add_ alpha conversion, half encodings, layout and update guards."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/mixed-training-rounding'


def main(label):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    checks = []

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
        print(name, row['status'], row.get('error', '')[:240], flush=True)

    def update(x, y, alpha, exact=True, transposed=False, alias=False):
        host = x.clone().t() if transposed else x.clone()
        data = x.to('intel_npu').t() if transposed else x.to('intel_npu')
        other = host if alias else y.t() if transposed else y
        right = data if alias else y.to('intel_npu').t() if transposed else y.to('intel_npu')
        version, stride = data._version, data.stride()
        with torch.no_grad():
            host.add_(other, alpha=alpha)
            assert data.add_(right, alpha=alpha) is data
        assert data._version == version+1 and data.stride() == stride
        got = data.cpu()
        rtol, atol = (.003, .0005) if host.dtype == torch.float16 else (3e-5, 3e-6)
        torch.testing.assert_close(got, host, rtol=0 if exact else rtol,
                                   atol=0 if exact else atol, equal_nan=True)
        return {'unequal_finite_or_inf': int((got != host).logical_and(~torch.isnan(host)).sum()),
                'max_abs': float((got.float()-host.float()).abs().max())}

    def replay():
        x = torch.tensor([.00015676021575927734, .0007448196411132812,
                          -.01300811767578125, -.002628326416015625], dtype=torch.float16)
        y = torch.tensor([.1925048828125, .132080078125, .09649658203125, -.1959228515625], dtype=torch.float16)
        # Preserve the vector-loop context of the original 256-element parameter.
        update(x.repeat(16), y.repeat(16), -.01)
        return {'four_original_update_samples_in_64_element_vector_exact': True}

    def scalar_tail():
        x = torch.tensor([.00015676021575927734, .0007448196411132812,
                          -.01300811767578125, -.002628326416015625], dtype=torch.float16)
        y = torch.tensor([.1925048828125, .132080078125, .09649658203125, -.1959228515625], dtype=torch.float16)
        return dict(update(x, y, -.01, exact=False), cpu_half_product_tail_bit_parity_not_certified=True)

    def encodings():
        x = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
        y = torch.roll(x, 17431)
        for alpha in (-.01, .03, .8, 1e-8, 0., 1., -2., 60000., float('inf'), float('nan')):
            update(x, y, alpha)
        return {'half_patterns': 65536, 'alphas': 10, 'nan_positions_checked': True, 'signed_zero_payload_not_certified': True}

    def layouts(dtype):
        x = torch.tensor([[.00015676021575927734, -.25], [.3, -.01300811767578125]], dtype=dtype)
        y = torch.tensor([[.1925048828125, .75], [-.8, .09649658203125]], dtype=dtype)
        for transposed in (False, True):
            for alias in (False, True):
                update(x, y, -.01, exact=dtype==torch.float16, transposed=transposed, alias=alias)
        return {'transpose_self_alias_and_version': True}

    def guards():
        x = torch.ones(2, 2, dtype=torch.float16, device='intel_npu')
        y = torch.ones_like(x)
        before = npu.stats()['npu_calls']
        rejected = []
        with torch.no_grad():
            for alpha in (True, 70000., 1e50):
                try: x.add_(y, alpha=alpha)
                except (ValueError, TypeError, RuntimeError): rejected.append(str(alpha))
                else: raise AssertionError(f'Accepted alpha={alpha}')
        assert npu.stats()['npu_calls'] == before
        try: x.add_(y, alpha=.8)
        except ValueError as error: assert 'no_grad' in str(error)
        else: raise AssertionError('Accepted grad-enabled update')
        assert npu.stats()['npu_calls'] == before
        leaf = x.detach().requires_grad_()
        loss = (leaf*leaf).sum()
        with torch.no_grad(): leaf.add_(y, alpha=-.01)
        try: loss.backward()
        except RuntimeError as error: assert 'version' in str(error) or 'inplace' in str(error)
        else: raise AssertionError('Saved version guard bypassed')
        return {'rejected_before_submit': rejected+['grad-enabled'], 'saved_version_guard': True}

    check('original-update-replay', replay)
    check('scalar-tail-tolerance-boundary', scalar_tail)
    check('half-encoding-alpha', encodings)
    for dtype in (torch.float16, torch.float32): check(str(dtype)+'-layouts', lambda d=dtype: layouts(d))
    check('alpha-contracts', guards)
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
    result = {'status': 'passed' if all(c['status']=='passed' for c in checks) else 'failed',
              'checks': checks, 'final_stats': final, 'source_sha256': {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
              for name in ('torch_intel_npu.py', 'torch_npu_autograd.py', 'npunlock-probe/research/add_alpha_regression.py')}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'updates-{label}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result['status'] != 'passed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', choices=('before', 'after'), default='after')
    raise SystemExit(main(parser.parse_args().label))

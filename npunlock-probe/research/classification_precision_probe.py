"""Compare native-half and explicit-FP32 classification with an independent FP64 oracle."""
import gc
import hashlib
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/classification-precision'


def main():
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.manual_seed(127)
    checks = []

    def metrics(actual, expected):
        original_dtype = actual.dtype
        actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
        difference = actual-expected
        return {'max_abs': float(difference.abs().max()), 'rms': float(difference.square().mean().sqrt()),
                'nonfinite': int((~torch.isfinite(actual)).sum()),
                'different_from_rounded_reference': int((actual != expected.to(original_dtype).double()).sum())}

    def check(name, function):
        before = npu.stats()
        row = {'name': name}
        try:
            row.update(function())
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

    def classification(values, labels):
        reference = values.double().requires_grad_()
        ref_log = reference.log_softmax(-1)
        ref_loss = torch.nn.functional.cross_entropy(reference, labels)
        ref_loss.backward()
        candidates = {}
        for device in ('cpu', 'intel_npu'):
            for promoted in (False, True):
                x = values.to(device).detach().requires_grad_()
                logits = x.float() if promoted else x
                log = logits.log_softmax(-1)
                loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
                loss.backward()
                # The original FP16 input still receives FP16 gradients after explicit promotion.
                entry = {'log_softmax': metrics(log, ref_log), 'loss': metrics(loss, ref_loss),
                         'input_gradient': metrics(x.grad, reference.grad), 'loss_value': float(loss.detach().cpu())}
                if device == 'intel_npu':
                    rtol, atol = (3e-5, 3e-6) if promoted else (.003, .0005)
                    torch.testing.assert_close(log.detach().cpu().double(), ref_log, rtol=rtol, atol=atol)
                    torch.testing.assert_close(loss.detach().cpu().double(), ref_loss, rtol=rtol, atol=atol)
                    torch.testing.assert_close(x.grad.cpu().double(), reference.grad, rtol=.003, atol=.0005)
                    assert not entry['log_softmax']['nonfinite'] and not entry['input_gradient']['nonfinite']
                candidates[device+('-fp32-loss' if promoted else '-half-loss')] = entry
        return {'shape': list(values.shape), 'input_sha256': hashlib.sha256(values.numpy().tobytes()).hexdigest(),
                'oracle': 'FP64 logsumexp/CrossEntropy on the exact same quantized logits', 'candidates': candidates}

    # Preserve the initialization/input order of the two original failed networks.
    for mode in ('none', 'tanh'):
        host = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.LayerNorm(16),
                                   torch.nn.GELU(approximate=mode), torch.nn.Linear(16, 3)).half()
        host[1].float()
        data = torch.randn(4, 16, dtype=torch.float16)
        values = host(data).detach()
        labels = torch.tensor([0, 1, 2, 1])
        check('original-logits-'+mode, lambda v=values, t=labels: classification(v, t))
    generator = torch.Generator().manual_seed(701)
    for rows, width in ((1, 3), (8, 7), (3, 17), (2, 257), (1, 4096)):
        values = (torch.randn(rows, width, generator=generator)*4).half()
        labels = torch.arange(rows) % width
        check(f'random-{rows}x{width}', lambda v=values, t=labels: classification(v, t))
    values = torch.tensor([[10000., 10008., 9992.], [-30., 0., 30.]], dtype=torch.float16)
    check('large-offset-and-range', lambda: classification(values, torch.tensor([0, 2])))
    values = torch.zeros(1, 65532, dtype=torch.float16)
    check('wide-zero-cross-entropy', lambda: classification(values, torch.tensor([65531])))

    def maximum_logsoftmax():
        values = torch.zeros(1, 65536, dtype=torch.float16)
        oracle = values.double().log_softmax(-1)
        cpu = values.log_softmax(-1)
        data = values.to('intel_npu')
        native = data.log_softmax(-1).cpu()
        promoted = data.log_softmax(-1, dtype=torch.float32).cpu()
        torch.testing.assert_close(native, oracle.half(), rtol=0, atol=0)
        torch.testing.assert_close(promoted, oracle.float(), rtol=3e-5, atol=3e-6)
        assert torch.isneginf(cpu).all() and torch.isfinite(native).all()
        return {'cpu_half_nonfinite': int((~torch.isfinite(cpu)).sum()),
                'npu_half_value': float(native[0, 0]), 'npu_fp32_value': float(promoted[0, 0]),
                'fp64_value': float(oracle[0, 0])}

    check('maximum-65536-logsoftmax', maximum_logsoftmax)
    del data, values
    gc.collect()
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
    result = {'status': 'passed' if all(c['status']=='passed' for c in checks) else 'failed', 'checks': checks,
              'final_stats': final, 'source_sha256': {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
               ('torch_intel_npu.py', 'torch_npu_autograd.py', 'npunlock-probe/research/classification_precision_probe.py',
                'npunlock/tests/fixtures/npu4000/resident_classification.c', 'npunlock/tests/fixtures/npu4000/resident/manifest.json')}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'quality.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())

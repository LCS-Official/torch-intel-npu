"""Hardware regressions for cancellation, half GEMM, and IEEE exceptional values."""
import gc
import json
import torch
import torch_intel_npu as npu
from diagnose_fp16_training import OUT


def verify():
    checks = []
    host = torch.tensor([[1e8, 1, -1e8], [1e12, 3, -1e12], [1, 2, 3]], dtype=torch.float32)
    data = host.to('intel_npu')
    for kind in ('sum', 'mean'):
        expected = getattr(host.double(), kind)(1).float()
        actual = getattr(data, kind)(1).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        checks.append({'operation': kind, 'actual': actual.tolist(), 'fp64_reference': expected.tolist()})
    for dtype in (torch.float16, torch.float32):
        x = torch.tensor([[65504, 1, 65504]], dtype=dtype).to('intel_npu')
        y = torch.tensor([[65504], [1], [-65504]], dtype=dtype).to('intel_npu')
        torch.testing.assert_close((x @ y).cpu(), torch.ones(1, 1, dtype=dtype), rtol=0, atol=0)
        checks.append({'operation': 'mm', 'dtype': str(dtype), 'cancellation_result': 1})
    subnormal = torch.tensor(1, dtype=torch.int32).view(torch.float32).item()
    host = torch.tensor([[float('inf'), 1, 2], [float('inf'), -float('inf'), 1],
                         [float('nan'), 1, 2], [subnormal, subnormal, -subnormal]], dtype=torch.float32)
    data = host.to('intel_npu')
    for kind in ('sum', 'mean'):
        actual = getattr(data, kind)(1).cpu()
        expected = getattr(host.double(), kind)(1).float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        checks.append({'operation': kind, 'ieee_inf_nan_subnormal': True})
    return checks


if __name__ == '__main__':
    checks = verify()
    gc.collect()
    stats = npu.stats()
    assert not stats['context_failed']
    assert stats['live_buffers'] == stats['cached_scalar_buffers'], stats
    npu.shutdown()
    report = {'status': 'passed', 'checks': checks, 'final_stats': npu.stats()}
    (OUT / 'compensated-reduce-regression.json').write_text(json.dumps(report, indent=2))
    print('Compensated accumulation:', len(checks), 'checks passed', flush=True)

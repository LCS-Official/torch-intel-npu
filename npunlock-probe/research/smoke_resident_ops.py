"""Small resident cast/reduction checks before the full boundary validation."""
import torch
import torch_intel_npu as npu

for dtype in (torch.float32, torch.float16):
    a = torch.tensor([[1., -2., 3., 4., 5.], [6., 7., -8., 9., 10.], [11., 12., 13., 14., 15.]], dtype=dtype)
    x = a.to('intel_npu')
    print('uploaded', dtype, flush=True)
    converted = x.half() if dtype == torch.float32 else x.float()
    torch.testing.assert_close(converted.cpu(), a.to(converted.dtype), rtol=0, atol=0)
    print('cast', dtype, flush=True)
    for dim in (None, 0, 1, (0, 1), -1):
        for kind in ('sum', 'mean'):
            print(kind, dtype, dim, flush=True)
            result = getattr(x, kind)(dim=dim)
            torch.testing.assert_close(result.cpu(), getattr(a, kind)(dim=dim), rtol=1e-5, atol=1e-5)
    print('complete', dtype, npu.stats(), flush=True)

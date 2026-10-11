"""Resident broadcast/scalar/MatMul smoke check with explicit CPU references."""
import torch
import torch_intel_npu as npu

for dtype in (torch.float16, torch.float32):
    host = torch.arange(15, dtype=dtype).reshape(3, 5)
    x = host.to('intel_npu')
    print('dtype', dtype, 'result_type', torch.result_type(x, 1.), flush=True)
    for function in (lambda z: z + 1., lambda z: 2. - z, lambda z: z / 2., lambda z: 2. / (z + 1.)):
        torch.testing.assert_close(function(x).cpu(), function(host), rtol=1e-3, atol=1e-6)
    b = torch.arange(5, dtype=dtype).to('intel_npu')
    torch.testing.assert_close((x + b).cpu(), host + torch.arange(5, dtype=dtype))
    weights = (torch.arange(20).reshape(5, 4).float() * .125).to(dtype)
    w = weights.to('intel_npu')
    before = npu.stats()
    result = x @ w
    after = npu.stats()
    for key in ('host_upload_copies', 'host_download_copies', 'host_device_copies'):
        assert before[key] == after[key], key
    torch.testing.assert_close(result.cpu(), host @ weights, rtol=1e-3, atol=1e-6)
    print('passed', dtype, npu.stats(), flush=True)

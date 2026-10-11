"""Run with .\run_torch_npu.ps1 -Script .\torch_npu_linear_demo.py."""
import torch
import torch_intel_npu as npu

torch.manual_seed(7)
for dtype in (torch.float16, torch.float32):
    model = torch.nn.Sequential(torch.nn.Linear(5, 4, dtype=dtype),
                                torch.nn.Linear(4, 3, dtype=dtype)).eval()
    host = torch.linspace(-1, 1, 10, dtype=dtype).view(2, 5)
    with torch.no_grad():
        expected = model(host)
    model.to('intel_npu')
    data = host.to('intel_npu')
    with torch.inference_mode():
        model(data)  # Warm up static graph compilation.
        before = npu.stats()
        for _ in range(100):
            result = model(data)
        after = npu.stats()
        for key in ('host_upload_copies', 'host_download_copies', 'host_device_copies', 'session_creations'):
            assert before[key] == after[key], key
        actual = result.cpu()
        torch.testing.assert_close(actual, expected, rtol=2e-3 if dtype == torch.float16 else 2e-6,
                                   atol=2e-4 if dtype == torch.float16 else 2e-6)
        print(dtype, result.device, actual)
        print('100 forwards:', after['npu_calls'] - before['npu_calls'],
              'NPU submissions; 0 intermediate host copies; 0 new graph compilations')

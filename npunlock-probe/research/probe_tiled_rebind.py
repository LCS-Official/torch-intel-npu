"""Check repeated buffer rebinding of a tiled FP32 matrix in a fresh process."""
import torch
import torch_intel_npu as npu
import npunlock.npu4000 as compiler

print(compiler.__file__, flush=True)
print(npu.__file__, flush=True)
for index in range(20):
    for dtype in (torch.float16, torch.float32):
        x = torch.ones(65, 32, dtype=dtype).to('intel_npu')
        y = torch.ones(32, 32, dtype=dtype).to('intel_npu')
        actual = x@y
        torch.testing.assert_close(actual.cpu(), torch.full((65, 32), 32., dtype=dtype))
    print(index, npu.stats()['npu_calls'], flush=True)
print(npu.stats(), flush=True)

"""Inspect per-operator wrapped-scalar rounding at the FP16 range boundary."""
import torch
import torch_intel_npu

for dtype in (torch.float16, torch.float32):
    a = torch.tensor([-65504., -1., -0., 0., .25, 1.000244140625, 2.**-24, float('inf')], dtype=dtype)
    x = a.to('intel_npu')
    for scalar in (1.0001, 70000., -2):
        for name, function in (('add', lambda z: z+scalar), ('rsub', lambda z: scalar-z),
                               ('mul', lambda z: z*scalar), ('div', lambda z: z/scalar), ('rdiv', lambda z: scalar/z)):
            cpu, device = function(a), function(x).cpu()
            try:
                torch.testing.assert_close(device, cpu, rtol=1e-3 if dtype == torch.float16 else 3e-7,
                                            atol=0, equal_nan=True)
            except AssertionError:
                print(dtype, scalar, name, 'CPU', cpu.tolist(), 'NPU', device.tolist(), flush=True)

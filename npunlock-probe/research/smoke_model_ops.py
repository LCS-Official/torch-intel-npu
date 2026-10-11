"""Bounded hardware smoke before the full training regression."""
import torch
import torch_intel_npu as npu

torch.backends.mkldnn.enabled = False
torch.manual_seed(21)
x = torch.logspace(-38, 38, 511)
x = torch.cat([x, torch.tensor([0., -0., 1e-45, float('inf'), -1., float('nan')])])
with torch.no_grad():
    actual = npu._execute('rsqrt', x.to('intel_npu')).cpu()
torch.testing.assert_close(actual, x.rsqrt(), rtol=3e-7, atol=0., equal_nan=True)
print('rsqrt passed', flush=True)
for dtype in (torch.float16, torch.float32):
    x = torch.randn(65, 32, dtype=dtype)/8
    y = torch.randn(32, 32, dtype=dtype)/8
    actual = x.to('intel_npu') @ y.to('intel_npu')
    torch.testing.assert_close(actual.cpu(), (x.float() @ y.float()).to(dtype), rtol=.004, atol=5e-5)
    print('tiled matmul', dtype, flush=True)
    x = torch.randn(2, 3, 5, 4, dtype=dtype, requires_grad=True)
    cpu = torch.nn.BatchNorm2d(3, track_running_stats=False, dtype=dtype)
    import copy
    model = copy.deepcopy(cpu).to('intel_npu')
    data = x.detach().to('intel_npu').requires_grad_()
    expected, actual = cpu(x), model(data)
    seed = torch.randn_like(expected)
    expected.backward(seed)
    actual.backward(seed.to('intel_npu'))
    for a, b in ((actual, expected), (data.grad, x.grad), (model.weight.grad, cpu.weight.grad), (model.bias.grad, cpu.bias.grad)):
        torch.testing.assert_close(a.detach().cpu(), b.detach(), rtol=.006 if dtype == torch.float16 else 1e-5, atol=.004 if dtype == torch.float16 else 1e-5)
    print('batchnorm passed', dtype, flush=True)
print(npu.stats(), flush=True)

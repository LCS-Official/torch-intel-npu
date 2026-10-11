"""Small forward-only smoke check; broader coverage lives in the device demo."""
import torch
import torch_intel_npu as npu

for dtype in (torch.float16, torch.float32):
    host = torch.arange(15, dtype=dtype).reshape(3, 5)
    x = host.to('intel_npu')
    view = x.t()
    assert view.untyped_storage()._cdata == x.untyped_storage()._cdata
    assert not view.is_contiguous()
    torch.testing.assert_close(view.contiguous().cpu(), host.t(), rtol=0, atol=0)
    torch.testing.assert_close(view.cpu(), host.t(), rtol=0, atol=0)
    torch.testing.assert_close((view @ x).cpu(), host.t() @ host, rtol=0, atol=0)
    model = torch.nn.Linear(5, 4, dtype=dtype).eval()
    data = torch.linspace(-1, 1, 10, dtype=dtype).reshape(2, 5)
    expected = model(data).detach()
    model.to('intel_npu')
    assert model.weight.requires_grad and model.weight.device.type == 'intel_npu'
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            actual = model(data.to('intel_npu'))
            assert actual.device.type == 'intel_npu' and not actual.requires_grad
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3 if dtype == torch.float16 else 2e-6, atol=2e-6)
    actual = model(data.to('intel_npu'))
    assert actual.requires_grad and actual.grad_fn is not None
    torch.testing.assert_close(actual.detach().cpu(), expected, rtol=2e-3 if dtype == torch.float16 else 2e-6, atol=2e-6)
    print(dtype, 'views, addmm/Linear, no_grad/inference_mode passed', flush=True)
print(npu.stats())

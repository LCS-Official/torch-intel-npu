"""Initial linear/conv backward smoke; comprehensive checks live in training demo."""
import copy
import torch
import torch_intel_npu as npu

torch.manual_seed(9)
for cpu, shape in ((torch.nn.Linear(5, 3), (2, 5)), (torch.nn.Conv2d(1, 2, 3, padding=1), (2, 1, 5, 5))):
    model = copy.deepcopy(cpu).to('intel_npu')
    host = torch.linspace(-1, 1, torch.tensor(shape).prod().item()).reshape(shape).requires_grad_()
    data = host.detach().to('intel_npu').requires_grad_()
    expected = cpu(host)
    target = torch.full_like(expected, .1)
    expected_loss = torch.nn.functional.mse_loss(expected, target)
    expected_loss.backward()
    result = model(data)
    loss = torch.nn.functional.mse_loss(result, target.to('intel_npu'))
    print(type(cpu).__name__, 'forward:', result.requires_grad, type(result.grad_fn).__name__, flush=True)
    loss.backward()
    torch.testing.assert_close(result.detach().cpu(), expected.detach(), rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(data.grad.cpu(), host.grad, rtol=3e-5, atol=3e-6)
    for p, reference in zip(model.parameters(), cpu.parameters()):
        assert p.grad.device.type == 'intel_npu'
        torch.testing.assert_close(p.grad.cpu(), reference.grad, rtol=3e-5, atol=3e-6)
    torch.optim.SGD(model.parameters(), lr=.01, foreach=False).step()
    torch.optim.SGD(cpu.parameters(), lr=.01, foreach=False).step()
    with torch.no_grad():
        for p, reference in zip(model.parameters(), cpu.parameters()):
            torch.testing.assert_close(p.cpu(), reference, rtol=3e-5, atol=3e-6)
    print(type(cpu).__name__, 'backward + SGD passed', flush=True)
print(npu.stats())

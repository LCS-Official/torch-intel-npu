import torch
import torch_intel_npu as npu

x = torch.tensor([1., 2., 70000.]).to('intel_npu')
y = torch.tensor([2., 3., 0.5]).to('intel_npu')
print(type(x), x.device, x.untyped_storage().device, torch._C._dispatch_keys(x), flush=True)
print('x cpu', x.cpu(), flush=True)
z = (x + y) * y
print('result', z.cpu(), flush=True)
print(npu.stats(), flush=True)
v = z.view(1, 3)
del z
print('view', v.cpu(), flush=True)

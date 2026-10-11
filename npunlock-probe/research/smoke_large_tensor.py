"""First boundary probe for tiled storage/compute, with no CPU arithmetic fallback."""
import gc
import torch
import torch_intel_npu as npu


def check():
    host = (torch.arange(65537, dtype=torch.float32) % 17 - 8)/8
    x = host.to('intel_npu')
    torch.testing.assert_close((x+.25).cpu(), host+.25, rtol=0, atol=0)
    torch.testing.assert_close(x.half().float().cpu(), host, rtol=0, atol=0)
    torch.testing.assert_close(x.sum().cpu(), host.sum(), rtol=0, atol=0)
    print('65537-element arithmetic, casts and reduction passed', npu.stats(), flush=True)


check()
gc.collect()
assert npu.stats()['live_buffers'] == npu.stats()['cached_scalar_buffers'], npu.stats()
npu.shutdown()
assert npu.stats()['live_buffers'] == 0

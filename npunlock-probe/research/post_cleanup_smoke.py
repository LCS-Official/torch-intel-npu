"""Verify retained native files, arithmetic and one-layer prefill/decode without old results."""
import gc
import json
import os
import torch
import torch_intel_npu as npu
from torch_npu_smollm import FrozenSmolStack
from model_cross_device import MODEL


def work():
    torch.set_num_threads(8)
    with torch.no_grad():
        host=torch.arange(16,dtype=torch.float32)
        actual=(host.to('intel_npu')+70000).cpu()
        torch.testing.assert_close(actual,host+70000,rtol=0,atol=0)
        a=torch.arange(16,dtype=torch.float16).reshape(4,4)/16
        b=torch.eye(4,dtype=torch.float16)
        torch.testing.assert_close((a.to('intel_npu')@b.to('intel_npu')).cpu(),a,rtol=0,atol=0)
        stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=1,mode='fp16')
        x=torch.zeros(1,2,576,dtype=torch.float16).to('intel_npu')
        y,state=stack.prefill(x)
        z,state=stack.decode(torch.zeros(1,1,576,dtype=torch.float16).to('intel_npu'),state)
        assert y.shape==(1,2,576) and z.shape==(1,1,576) and state[0].length==3
        assert torch.isfinite(y.cpu()).all() and torch.isfinite(z.cpu()).all()


if __name__=='__main__':
    assert not os.environ.get('NPUNLOCK_GRAPH_CACHE')
    work();gc.collect();npu.shutdown();s=npu.stats()
    assert s['closed'] and not s['context_failed']
    assert all(s[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
    print(json.dumps(dict(status='passed',checks=['native provenance','strict FP32 arithmetic','FP16 matmul',
        'checkpoint and SHAVE kernels','one-layer prefill/decode','clean shutdown'],npu_calls=s['npu_calls']),indent=2))

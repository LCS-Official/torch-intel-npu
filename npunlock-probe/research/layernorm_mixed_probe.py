"""Strict hardware checks for ordered statistics and integer fused multiply-add."""
import ctypes
import gc
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/layernorm-mixed'


def main():
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from layernorm_mixed_diagnose import sample
    torch.set_num_threads(1)
    torch.manual_seed(113)
    crt = ctypes.CDLL('ucrtbase.dll')
    fma = crt.fmaf
    fma.argtypes = [ctypes.c_float]*3
    fma.restype = ctypes.c_float
    random = torch.randint(-2**31, 2**31, (8192,3), dtype=torch.int64).to(torch.int32).view(torch.float32)
    special = torch.tensor([[0., -1., -0.], [-0., 1., -0.], [1.+2**-23, 1.-2**-23, -1.],
        [float('inf'), 0., 1.], [float('inf'), 1., -float('inf')], [float('nan'), 1., 0.],
        [torch.finfo(torch.float32).max, 2., -float('inf')],
        [torch.finfo(torch.float32).max, 2., -torch.finfo(torch.float32).max],
        [2**-149, .5, 2**-149], [-2**-149, .5, -2**-149]], dtype=torch.float32)
    data = torch.cat((random, special))
    expected = torch.tensor([fma(*row) for row in data.tolist()], dtype=torch.float32)
    x = data.to('intel_npu')
    actual = npu._execute('layernorm-fma-test', x, x).cpu()[:,0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    assert torch.equal(actual[finite].view(torch.int32), expected[finite].view(torch.int32))
    del x
    comparisons = []
    values, w, b, _ = sample()
    for host in (values, torch.randn(3,17,dtype=torch.float16), torch.ones(2,4,dtype=torch.float16)):
        width = host.shape[-1]
        weight = torch.ones(width, dtype=torch.float32)
        bias = torch.zeros_like(weight)
        _, mean, inverse = torch.native_layer_norm(host, (width,), weight, bias, 1e-5)
        x = host.float().to('intel_npu')
        eps = torch.full(host.shape, 1e-5).to('intel_npu')
        packet = npu._run(('layernorm-stats','f32',tuple(host.shape)),[x,eps],host.shape,torch.float32).cpu()
        m, a = packet[:,:1], packet[:,1:2]
        print('stats',tuple(host.shape),'mean',m.view(torch.int32).tolist(),mean.view(torch.int32).tolist(),
              'inverse',a.view(torch.int32).tolist(),inverse.view(torch.int32).tolist(),flush=True)
        assert torch.equal(m.view(torch.int32), mean.view(torch.int32))
        assert torch.equal(a.view(torch.int32), inverse.view(torch.int32))
        comparisons.append({'shape':list(host.shape),'statistics_bit_equal':True})
        del x,eps
    gc.collect()
    npu.shutdown()
    final = npu.stats()
    assert final['live_buffers'] == final['cached_buffer_blocks'] == final['cached_sessions'] == 0
    assert final['closed'] and not final['context_failed']
    result = {'fma_finite_bits_equal':True,'fma_samples':len(data),'statistics':comparisons,'final_stats':final,'status':'passed'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('FMA and ordered moments passed',flush=True)


if __name__ == '__main__':
    main()

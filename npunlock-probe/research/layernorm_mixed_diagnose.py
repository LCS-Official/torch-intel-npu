"""Replay the saved mixed-affine failure and isolate statistics versus arithmetic."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/layernorm-mixed'


def sample():
    import torch
    torch.manual_seed(73)
    specs = [((4,), (4,)), ((3, 4), (4,)), ((2, 3, 5), (3, 5)),
             ((2, 2, 3, 5), (3, 5)), ((2, 3, 5), (2, 3, 5)),
             ((2, 1), (1,)), ((1, 4096), (4096,)), ((17, 4096), (4096,))]
    for dtype, mixed in ((torch.float32, False), (torch.float16, False), (torch.float16, True)):
        for shape, normalized in specs:
            for affine in ('both', 'none', 'weight', 'bias'):
                if mixed and affine == 'bias':
                    continue
                x = torch.randn(shape, dtype=dtype)
                if shape == (3, 4):
                    x = x.t().contiguous().t()
                pdtype = torch.float32 if mixed else dtype
                w = torch.randn(normalized, dtype=pdtype) if affine in ('both', 'weight') else None
                b = torch.randn(normalized, dtype=pdtype) if affine in ('both', 'bias') else None
                g = (torch.randn_like(x)/4).contiguous()
                if mixed and shape == (1, 4096) and affine == 'both':
                    return x, w, b, g
    raise AssertionError('Missing replay case')


def main():
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    x, w, b, g = sample()
    _, cpu_mean, cpu_inverse = torch.native_layer_norm(x, (4096,), w, b, 1e-5)
    expected = torch.ops.aten.native_layer_norm_backward(g, x, (4096,), cpu_mean, cpu_inverse,
                                                       w, b, (True, True, True))
    xn, wn, bn, gn = (t.to('intel_npu') for t in (x, w, b, g))
    _, nm, na = npu._layer_norm(xn, (4096,), wn, bn)
    actual = npu._layer_norm_backward(gn, xn, (4096,), nm, na, wn, bn, (True, True, True))
    mean, inverse, got = nm.cpu(), na.cpu(), actual[1].cpu()
    def compare(v):
        v = v.view(-1)
        target = expected[1].view(-1)
        difference = (v-target).abs()
        bad = difference > 3e-6+3e-5*target.abs()
        return {'bad': int(bad.sum()), 'max_abs': float(difference.max()),
                'indices': bad.nonzero().flatten().tolist()[:10]}
    comparisons = {'actual_npu': compare(got)}
    for name, m, a in (('cpu_stats', cpu_mean, cpu_inverse), ('npu_stats', mean, inverse)):
        offset = -m*a
        separate = x.float()*a+offset
        fused = (x.double()*a.double()+offset.double()).float()
        for mode, norm in (('separate', separate), ('fused', fused)):
            comparisons[name+'_'+mode] = compare((g.float()*norm).half().float())
    index = 1228
    result = {'comparisons': comparisons, 'cpu_mean': cpu_mean.tolist(), 'npu_mean': mean.tolist(),
              'cpu_inverse': cpu_inverse.tolist(), 'npu_inverse': inverse.tolist(),
              'at_1228': {'input': float(x[0,index]), 'gradient': float(g[0,index]),
                          'expected_weight_gradient': float(expected[1][index]),
                          'actual_weight_gradient': float(got[index])},
              'input_sha256': hashlib.sha256(x.numpy().tobytes()+g.numpy().tobytes()).hexdigest()}
    del xn, wn, bn, gn, nm, na, actual, _
    npu.shutdown()
    result['final_stats'] = npu.stats()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'diagnosis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

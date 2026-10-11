"""Native Torch ToCopyBackward with resident dtype conversions."""
import gc
import torch
import torch_intel_npu as npu


def check():
    for source, target in ((torch.float16, torch.float32), (torch.float32, torch.float16)):
        x = torch.tensor([[.5, -.25, 1.], [2., .125, -3.]], dtype=source).to('intel_npu').requires_grad_()
        y = x.t().to(target)
        print(source, target, type(y.grad_fn).__name__, y.stride(), flush=True)
        assert y.requires_grad and type(y.grad_fn).__name__ == 'ToCopyBackward0'
        seed = torch.tensor([[.5, 1.], [2., -1.], [3., .25]], dtype=target)
        y.backward(seed.to('intel_npu'))
        torch.testing.assert_close(x.grad.cpu(), seed.t().to(source), rtol=0, atol=0)
        try:
            x.cpu()
        except ValueError:
            pass
        else:
            raise AssertionError('Differentiable download accepted')
        before = npu.stats()['npu_calls']
        try:
            npu._to_copy(x.detach(), out=x)
        except ValueError:
            pass
        else:
            raise AssertionError('Differentiable raw out= accepted')
        assert npu.stats()['npu_calls'] == before


if __name__ == '__main__':
    check()
    gc.collect()
    assert npu.stats()['live_buffers'] == npu.stats()['cached_scalar_buffers'], npu.stats()
    npu.shutdown()
    print('differentiable casts passed', flush=True)

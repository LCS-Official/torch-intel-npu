"""Offline CPU controls for public RMSNorm and gated/decoder migration tests."""
import unittest
import torch
from torch_npu_rng import RNGState
from torch_npu_standard_modules_demo import cpu_rng_reference
from torch_npu_unary_math_demo import gated_model, decoder_model


class UnaryMathControls(unittest.TestCase):
    def test_rms_norm_default_is_computation_epsilon_and_mixed_weight(self):
        for dtype in (torch.float16, torch.float32):
            x = torch.tensor([[.3, -.7, 1., .2], [.1, .5, -.3, .8]], dtype=dtype, requires_grad=True)
            w = torch.tensor([.5, 1., 1.5, 2.], requires_grad=True)
            actual = torch.rms_norm(x, (4,), w)
            v = x.float()
            expected = (v*torch.rsqrt((v*v).mean(-1, keepdim=True)+torch.finfo(torch.float32).eps)*w).to(dtype)
            torch.testing.assert_close(actual, expected, rtol=.003, atol=.0005)
            seed = torch.tensor([[.125, -.25, .5, 1.], [-.25, .125, .5, -.5]], dtype=dtype)
            g = torch.autograd.grad(actual, (x, w), seed, retain_graph=True)
            reference = torch.autograd.grad(expected, (x, w), seed)
            for a, b in zip(g, reference): torch.testing.assert_close(a, b, rtol=.003, atol=.0005)
            if dtype == torch.float16:
                wrong = torch.rms_norm(x, (4,), w, torch.finfo(torch.float16).eps)
                self.assertFalse(torch.equal(actual, wrong))

    def test_gated_standard_modules_all_receive_gradients(self):
        for dtype in (torch.float16, torch.float32):
            model = gated_model(dtype)
            output = model(torch.randn(2, 3, 8).to(dtype))
            output.float().square().mean().backward()
            self.assertEqual(sum(p.numel() for p in model.parameters()), 355)
            self.assertEqual(tuple(output.shape), (2, 3))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_decoder_same_mask_control_covers_self_and_cross_attention(self):
        model = decoder_model(torch.float32)
        hosts = [torch.randn(2, n, 5)/4 for n in (3, 4)]
        before = torch.get_rng_state()
        with cpu_rng_reference(RNGState(257).get_state()) as tracker:
            output = model(*hosts); output.square().mean().backward()
        self.assertEqual(tracker.offset, 648)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1947)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__': unittest.main()

"""Offline checks of the CPU-only same-mask control for standard modules."""
import unittest
import torch
from torch_npu_rng import RNGState
from torch_npu_standard_modules_demo import cpu_rng_reference


class CPUReferenceTests(unittest.TestCase):
    def test_callback_restoration_and_no_cpu_rng_use(self):
        dropout,sdpa=torch.nn.functional.dropout,torch.nn.functional.scaled_dot_product_attention
        before=torch.get_rng_state().clone();x=torch.ones(8,requires_grad=True)
        with self.assertRaisesRegex(RuntimeError,'control'):
            with cpu_rng_reference(RNGState(19).get_state()) as tracker:
                output=torch.nn.functional.dropout(x,p=.25);output.sum().backward()
                torch.testing.assert_close(x.grad,output.detach(),rtol=0,atol=0)
                self.assertEqual(tracker.offset,8)
                raise RuntimeError('control')
        self.assertIs(torch.nn.functional.dropout,dropout)
        self.assertIs(torch.nn.functional.scaled_dot_product_attention,sdpa)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_mha_replay_and_counter(self):
        model=torch.nn.MultiheadAttention(4,2,dropout=.25,batch_first=True)
        x=torch.randn(2,3,4,requires_grad=True);state=RNGState(29).get_state();before=torch.get_rng_state()
        with cpu_rng_reference(state) as tracker:expected,_=model(x,x,x,need_weights=False)
        self.assertEqual(tracker.offset,36)
        with cpu_rng_reference(state):actual,_=model(x,x,x,need_weights=False)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        actual.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_encoder_all_dropout_sites_and_eval(self):
        model=torch.nn.TransformerEncoderLayer(4,2,dim_feedforward=8,dropout=.25,batch_first=True)
        x=torch.randn(2,3,4,requires_grad=True);state=RNGState(31).get_state();before=torch.get_rng_state()
        with cpu_rng_reference(state) as tracker:output=model(x)
        self.assertEqual(tracker.offset,36+24+48+24)
        output.sum().backward();model.eval()
        with torch.no_grad(),cpu_rng_reference(state) as tracker:model(x)
        self.assertEqual(tracker.offset,0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()

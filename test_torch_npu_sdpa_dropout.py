"""Validate the CPU oracle against PyTorch's math SDPA before device comparisons."""
import unittest
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch_npu_rng import RNGState
from torch_npu_sdpa_dropout_demo import keep_mask,reference


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        generator=torch.Generator().manual_seed(83)
        self.values=[torch.randn(shape,generator=generator,requires_grad=True) for shape in ((2,1,3,4),(1,2,5,4),(2,1,5,2))]

    def test_zero_dropout_masks_and_causal(self):
        mask=torch.tensor([[True,False,True,True,False],[False]*5,[True]*5])
        for mode in ('bool','float','causal'):
            m=None if mode=='causal' else mask if mode=='bool' else torch.where(mask,0.,float('-inf'))
            with sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.scaled_dot_product_attention(*self.values,attn_mask=m,is_causal=mode=='causal',scale=-.7)
            actual=reference(*self.values,m,causal=mode=='causal',scale=-.7)
            torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
            for a,b in zip(torch.autograd.grad(actual.sum(),self.values),torch.autograd.grad(expected.sum(),self.values)):
                torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)

    def test_supplied_dropout_mask_math_reference(self):
        mask=keep_mask(RNGState(19).get_state(),(2,2,3,5),.25)
        actual=reference(*self.values,p=.25,keep=mask)
        expected,_=torch.ops.aten._scaled_dot_product_attention_math(*self.values,dropout_p=.25,dropout_mask=mask)
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
        for a,b in zip(torch.autograd.grad(actual.sum(),self.values),torch.autograd.grad(expected.sum(),self.values)):
            torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)

    def test_large_causal_reference_with_broadcast_and_supplied_mask(self):
        generator=torch.Generator().manual_seed(239)
        values=[(torch.randn(shape,generator=generator)/4).requires_grad_() for shape in
                ((2,1,129,3),(1,4,129,3),(2,1,129,2))]
        mask=keep_mask(RNGState(71).get_state(),(2,4,129,129),.25)
        actual=reference(*values,p=.25,causal=True,scale=-.7,keep=mask)
        expected,_=torch.ops.aten._scaled_dot_product_attention_math(*values,dropout_p=.25,
            is_causal=True,scale=-.7,dropout_mask=mask)
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
        for a,b in zip(torch.autograd.grad(actual.sum(),values),torch.autograd.grad(expected.sum(),values)):
            torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)

    def test_full_dropout(self):
        actual=reference(*self.values,p=1.,keep=torch.zeros(2,2,3,5,dtype=torch.bool))
        with sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.scaled_dot_product_attention(*self.values,dropout_p=1.)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(torch.count_nonzero(actual).item(),0)
        self.assertTrue(all(torch.count_nonzero(g).item()==0 for g in torch.autograd.grad(actual.sum(),self.values)))
        with self.assertRaises(ValueError):reference(*self.values,p=.25)


if __name__=='__main__':unittest.main()

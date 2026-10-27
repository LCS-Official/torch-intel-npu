"""Independent algebra and immutable calibration contracts, without NPU calls."""
import unittest
import numpy as np
import torch
from torch_npu_lowbit import PackedWeight,quantize_weight
from torch_npu_lowbit_correction import LowRankCorrection,fit_lowrank_correction


class CorrectionTests(unittest.TestCase):
    def test_exact_low_rank_error_reconstruction_including_dead_inputs(self):
        torch.manual_seed(441)
        x=torch.randn(64,7);x[:,-1]=0
        weight=(torch.randn(5,2)@torch.randn(2,7)/32).half()
        packed=PackedWeight((5,7),'int8',7,bytes(35),np.ones(5,dtype=np.float16).tobytes())
        q=fit_lowrank_correction(weight,packed,x,rank=2)
        a,b=q.tensors()
        actual=(x.double()@a.double().t())@b.double().t()
        expected=x.double()@weight.double().t()
        torch.testing.assert_close(actual,expected,rtol=.02,atol=2e-4)
        self.assertLess(float((actual-expected).square().sum()),float(expected.square().sum())*1e-5)

    def test_full_rank_matches_original_weight_and_truncation(self):
        torch.manual_seed(442)
        weight=torch.randn(4,7).half();x=torch.randn(50,7)
        packed=quantize_weight(weight,format='int8')
        q=fit_lowrank_correction(weight,packed,x,rank=4)
        a,b=q.tensors()
        decoded=np.frombuffer(packed.data,dtype=np.int8).reshape(packed.shape)*np.frombuffer(packed.scales,dtype=np.float16).astype(np.float64)[:,None]
        actual=torch.from_numpy(decoded)+b.double()@a.double()
        torch.testing.assert_close(actual,weight.double(),rtol=1e-3,atol=1e-5)
        truncated=q.truncated(2)
        aa,bb=truncated.tensors()
        torch.testing.assert_close(aa,a[:2],rtol=0,atol=0)
        torch.testing.assert_close(bb,b[:,:2],rtol=0,atol=0)
        a.fill_(99)
        self.assertNotEqual(q.tensors()[0][0,0],99)

    def test_zero_error_is_finite_zero_correction(self):
        weight=torch.full((4,8),.125)
        packed=quantize_weight(weight,format='int8')
        # Use an exactly representable code/scale product.
        packed=PackedWeight((4,8),'int8',8,bytes([1])*32,np.full(4,.125,dtype=np.float16).tobytes())
        q=fit_lowrank_correction(weight,packed,torch.zeros(10,8),rank=2)
        a,b=q.tensors();self.assertEqual(int(torch.count_nonzero(a)),0);self.assertEqual(int(torch.count_nonzero(b)),0)

    def test_invalid_data_and_fit_reject(self):
        weight=torch.ones(4,8);packed=quantize_weight(weight,format='int8');x=torch.ones(12,8)
        for rank in (0,5,True):
            with self.assertRaises(ValueError):fit_lowrank_correction(weight,packed,x,rank=rank)
        for bad in (torch.ones(12,7),torch.full((12,8),float('nan')),torch.ones(4097,8)):
            with self.assertRaises(ValueError):fit_lowrank_correction(weight,packed,bad,rank=2)
        with self.assertRaises(ValueError):fit_lowrank_correction(weight,quantize_weight(weight,format='int4'),x,rank=2)
        for shape,rank,down,up in (((4,8),True,bytes(32),bytes(16)),((4,8),2,bytearray(32),bytes(16)),
                                 ((4,8),2,np.full(16,np.inf,dtype=np.float16).tobytes(),bytes(16)),
                                 ((4,8),2,bytes(30),bytes(16))):
            with self.assertRaises(ValueError):LowRankCorrection(shape,rank,down,up)


if __name__=='__main__':unittest.main()

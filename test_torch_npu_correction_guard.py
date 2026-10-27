"""Calibration row guard must respect every local trajectory, not just the mean."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch

research=Path(__file__).with_name('npunlock-probe')/'research'


@unittest.skipUnless(importlib.util.find_spec('safetensors'), 'optional model calibration dependency')
class CorrectionGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(research))
        try:
            spec=importlib.util.spec_from_file_location('_s45_guard',research/'w8_guarded_quality.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            cls.guard=staticmethod(module.guard_rows)
            cls.detail=staticmethod(module.failure_detail)
        finally:sys.path.pop(0)

    def test_selects_off_half_full_without_changing_down(self):
        w=torch.full((3,1),2.,dtype=torch.float16);decoded=torch.tensor([[0.],[1.],[2.]],dtype=torch.float64)
        down=torch.ones(1,1,dtype=torch.float16);up=torch.tensor([[2.],[2.],[1.]],dtype=torch.float16)
        result,stats=self.guard(w,decoded,torch.ones(2,1,dtype=torch.float16),down,up,group_rows=1)
        torch.testing.assert_close(result,torch.tensor([[2.],[1.],[0.]],dtype=torch.float16),rtol=0,atol=0)
        self.assertEqual((stats['off'],stats['half'],stats['full']),(1,1,1))
        self.assertLess(stats['guarded_sse'],stats['plain_sse'])
        self.assertTrue(torch.equal(down,torch.ones_like(down)))

    def test_rejects_a_factor_that_hurts_one_group(self):
        w=torch.ones(1,2,dtype=torch.float16);decoded=torch.zeros(1,2,dtype=torch.float64)
        result,stats=self.guard(w,decoded,torch.eye(2,dtype=torch.float16),
            torch.eye(2,dtype=torch.float16),torch.tensor([[.5,4.]],dtype=torch.float16),group_rows=1)
        torch.testing.assert_close(result,torch.tensor([[.25,2.]],dtype=torch.float16),rtol=0,atol=0)
        self.assertEqual(stats['half'],1)

    def test_common_offset_diagnosis_does_not_relax_the_quality_gate(self):
        from types import SimpleNamespace
        expected=torch.tensor([[1.,0.,0.]])
        detail=self.detail(expected+4,expected,SimpleNamespace(decode=lambda ids:str(ids)))
        self.assertAlmostEqual(detail['offset_energy_fraction'],1.)
        self.assertEqual(detail['centered_error_same_denominator'],0.)
        self.assertTrue(detail['next_token_equal']);self.assertFalse(detail['quality_gate_passed'])


if __name__=='__main__':unittest.main()

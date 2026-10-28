"""Offline contracts for bounded W8 scale/correction preparation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch_npu_lowbit import PackedWeight,quantize_weight
from torch_npu_lowbit_correction import LowRankCorrection,fit_lowrank_correction


def refine():
    path=Path(__file__).parent/'npunlock-probe/research/w8_scale_quality.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='refine_scales')
    env=dict(torch=torch,np=np,F=F,PackedWeight=PackedWeight,LowRankCorrection=LowRankCorrection,fit_lowrank_correction=fit_lowrank_correction)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env['refine_scales']


class ScaleCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.weight=torch.tensor([[.5,.2,-.8,1.],[-.4,.7,.3,-.2],[.1,-.5,.8,.3]],dtype=torch.float16)
        self.packed=quantize_weight(self.weight,format='int8')
        self.x=torch.tensor([[1.,2.,3.,4.],[2.,-1.,.5,3.],[-2.,1.,2.,-.5],[.5,-2.,1.,3.]],dtype=torch.float16)
        self.factor=fit_lowrank_correction(self.weight,self.packed,self.x,rank=1)

    def test_preserves_codes_shape_rank_and_calibration_nonincrease(self):
        packed,factor,row=refine()(self.weight,self.packed,self.x,self.factor)
        self.assertEqual(packed.data,self.packed.data);self.assertEqual(packed.shape,self.packed.shape)
        self.assertEqual(factor.rank,1);self.assertEqual(factor.shape,packed.shape)
        self.assertLessEqual(row['after_sse'],row['before_sse'])
        self.assertGreaterEqual(row['min_scale_ratio'],.98);self.assertLessEqual(row['max_scale_ratio'],1.02)
        self.assertEqual(len(row['steps']),2)

    def test_known_scale_offset_is_improved(self):
        weight=torch.from_numpy(self.packed.dequantize().copy())*1.01
        factor=LowRankCorrection(self.packed.shape,1,np.zeros((1,4),np.float16).tobytes(),np.zeros((3,1),np.float16).tobytes())
        packed,_,row=refine()(weight,self.packed,self.x,factor)
        self.assertGreater(row['changed_rows'],0);self.assertLess(row['after_sse'],row['before_sse'])
        self.assertEqual(packed.data,self.packed.data)

    def test_inputs_weights_and_original_immutable_parameters_are_preserved(self):
        x=self.x.clone();weight=self.weight.clone();packed=self.packed;factor=self.factor
        first=refine()(self.weight,packed,self.x,factor);second=refine()(self.weight,packed,self.x,factor)
        self.assertEqual(first,second);self.assertEqual(self.packed,packed);self.assertEqual(self.factor,factor)
        torch.testing.assert_close(self.x,x,rtol=0,atol=0);torch.testing.assert_close(self.weight,weight,rtol=0,atol=0)

    def test_zero_energy_keeps_original_parameters(self):
        packed,factor,row=refine()(self.weight,self.packed,torch.zeros_like(self.x),self.factor)
        self.assertEqual(packed,self.packed);self.assertEqual(factor,self.factor)
        self.assertEqual(row['changed_rows'],0);self.assertEqual(row['before_sse'],row['after_sse'])

    def test_nonfinite_or_mismatched_inputs_reject(self):
        for x in (self.x[:,:3],self.x[:0],self.x.double(),self.x.clone().fill_(float('nan'))):
            with self.subTest(shape=x.shape,dtype=x.dtype),self.assertRaises(ValueError):
                refine()(self.weight,self.packed,x,self.factor)

    def test_unsupported_format_and_factor_shape_reject(self):
        with self.assertRaises(ValueError):refine()(self.weight,quantize_weight(self.weight,format='int4'),self.x,self.factor)
        bad=LowRankCorrection((4,4),1,np.zeros((1,4),np.float16).tobytes(),np.zeros((4,1),np.float16).tobytes())
        with self.assertRaises(ValueError):refine()(self.weight,self.packed,self.x,bad)

    def test_fp16_target_overflow_rejects(self):
        with self.assertRaisesRegex(ValueError,'overflows FP16'):
            refine()(self.weight,self.packed,self.weight[0].sign().expand_as(self.x)*65504,self.factor)

    def test_model_reference_uses_new_decoded_scales_and_factors_at_each_layer(self):
        path=Path(__file__).parent/'npunlock-probe/research/w8_scale_quality.py'
        node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='apply_profile')
        packed={};factors={};modules=[]
        for i in range(2):
            packed[i]={};factors[i]={}
            for j in range(7):
                name=str(j);codes=np.full((40,40),j+1,np.int8);scales=np.full(40,(i+1)/16,np.float16)
                packed[i][name]=PackedWeight((40,40),'int8',40,codes.tobytes(),scales.tobytes())
                factors[i][name]=LowRankCorrection((40,40),32,np.full((32,40),i+1,np.float16).tobytes(),np.full((40,32),j+1,np.float16).tobytes())
                modules.append((name,SimpleNamespace(decoded=torch.zeros(40,40),down=torch.zeros(32,40),up=torch.zeros(40,32),rank=0,protected=True)))
        env=dict(np=np,torch=torch,parameters=lambda profile:(packed,factors))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
        env['apply_profile'](modules,'ffn')
        for index,(_,module) in enumerate(modules):
            i,j=divmod(index,7)
            self.assertTrue(torch.all(module.decoded==(i+1)*(j+1)/16))
            self.assertTrue(torch.all(module.down==i+1));self.assertTrue(torch.all(module.up==j+1))
            self.assertEqual(module.rank,32);self.assertFalse(module.protected)


class CandidateQualificationTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).parent
        path=root/'npunlock-probe/research/w8_scale_quality.py'
        choose=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='choose')
        prior=root/'npunlock-probe/research/w8_gate_experiment.py'
        promote=next(n for n in ast.parse(prior.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='promotable')
        env={};exec(compile(ast.Module(body=[promote],type_ignores=[]),str(prior),'exec'),env)
        env.update(previous=SimpleNamespace(promotable=env['promotable']),base=SimpleNamespace(mixed=SimpleNamespace(common=SimpleNamespace(
            ranking=lambda s:(s['positions']-s['quality_passed'],s['max_relative_l2'],s['mean_relative_l2'])))))
        exec(compile(ast.Module(body=[choose],type_ignores=[]),str(path),'exec'),env);self.choose=env['choose']
        summary=dict(positions=290,quality_passed=243,mean_relative_l2=.02,max_relative_l2=.13,max_kl=.004,next_token_equal=286)
        self.baseline=dict(status='passed',healthy=True,summary=summary,local_checks=[{}]*30)
        self.rejected=dict(status='failed',healthy=True,local_guard_failure=dict(layer=29,relative_l2=.0006,max_scaled_error=.0135))

    def test_local_guard_rejection_is_excluded_before_ranking(self):
        improved=dict(self.baseline,summary=dict(self.baseline['summary'],quality_passed=244,mean_relative_l2=.019))
        selected,candidate,rejected=self.choose(dict(baseline=self.baseline,**{'gate-up':improved,'ffn':self.rejected}))
        self.assertEqual((selected,candidate),('gate-up','gate-up'));self.assertEqual(list(rejected),['ffn'])
        selected,candidate,_=self.choose(dict(baseline=self.baseline,**{'gate-up':self.rejected,'ffn':self.rejected}))
        self.assertEqual(selected,'baseline');self.assertIsNone(candidate)

    def test_unhealthy_or_unexpected_errors_are_not_hidden_as_rejections(self):
        for failed in (dict(self.rejected,healthy=False),dict(status='failed',healthy=True),
            dict(self.rejected,local_guard_failure=dict(relative_l2=0,max_scaled_error=0))):
            with self.assertRaises(ValueError):self.choose(dict(baseline=self.baseline,ffn=failed))
        with self.assertRaises(ValueError):self.choose(dict(baseline=self.rejected,ffn=self.baseline))

    def test_passed_result_must_include_all_local_and_internal_checks(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            self.choose(dict(baseline=self.baseline,ffn=dict(self.baseline,local_checks=[{}]*29)))


class ScaleDiagnosticTests(unittest.TestCase):
    def test_peak_reports_the_worst_scaled_coordinate_and_original_values(self):
        path=Path(__file__).parent/'npunlock-probe/research/w8_scale_audit.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='scaled_peak')
        env=dict(np=np,torch=torch)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
        reference=torch.tensor([[[1.,2.],[3.,4.]]]);actual=reference.clone();actual[0,1,0]+=.5
        peak=env['scaled_peak'](actual,reference)
        self.assertEqual(peak['index'],(0,1,0));self.assertEqual((peak['actual'],peak['reference']),(3.5,3.))
        self.assertAlmostEqual(peak['row_rms'],(12.5)**.5,places=6)
        self.assertAlmostEqual(peak['scaled_error'],.5/(3+12.5**.5),places=7)
        with self.assertRaises(ValueError):env['scaled_peak'](actual[:,:0],reference[:,:0])


if __name__=='__main__':unittest.main()

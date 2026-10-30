"""Offline contracts for the paired FP16-decode fitting pilot."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch_npu_lowbit import PackedWeight
from torch_npu_lowbit_correction import LowRankCorrection


def helpers(*names,**extra):
    path=Path(__file__).parent/'npunlock-probe/research/w8_decode_proxy.py'
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    env=dict(np=np,torch=torch,PackedWeight=PackedWeight,LowRankCorrection=LowRankCorrection,NAME='mlp.down_proj.weight')
    env.update(extra);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return SimpleNamespace(**{name:env[name] for name in names})


class DecodeProxyTests(unittest.TestCase):
    def setUp(self):
        self.call=helpers('proxy_main').proxy_main
        self.weight=PackedWeight((1,2),'int8',2,np.array([3,5],np.int8).tobytes(),np.array([.1],np.float16).tobytes())

    def test_decoding_rounds_before_multiply_and_exposes_cancellation(self):
        x=torch.tensor([[5.,-3.]]).half()
        self.assertEqual(float(self.call(self.weight,x,'exact')),0.)
        self.assertEqual(float(self.call(self.weight,x,'decoded')),-.0009765625)
        expected=(x.double()@torch.from_numpy(self.weight.dequantize()).double().t()).half()
        torch.testing.assert_close(self.call(self.weight,x,'decoded'),expected,rtol=0,atol=0)

    def test_output_is_half_and_original_weight_and_input_are_preserved(self):
        x=torch.tensor([[1.,2.],[.5,-.25]]).half();saved=x.clone();data=self.weight.data;scales=self.weight.scales
        for mode in ('exact','decoded'):
            result=self.call(self.weight,x,mode);self.assertEqual(result.dtype,torch.float16);self.assertEqual(tuple(result.shape),(2,1))
        torch.testing.assert_close(x,saved,rtol=0,atol=0)
        self.assertEqual(self.weight.data,data);self.assertEqual(self.weight.scales,scales)

    def test_invalid_input_or_unlisted_mode_rejects(self):
        for x,mode in ((torch.ones(1,2),'decoded'),(torch.ones(1,3).half(),'decoded'),(torch.ones(0,2).half(),'decoded'),
                       (torch.ones(4097,2).half(),'decoded'),(torch.full((1,2),float('nan')).half(),'decoded'),(torch.ones(1,2).half(),'other')):
            with self.assertRaises(ValueError):self.call(self.weight,x,mode)

    def test_grouped_and_other_lowbit_formats_cannot_silently_use_row_scales(self):
        grouped=PackedWeight((1,4),'int8',2,bytes([1,2,3,4]),np.array([.1,.2],np.float16).tobytes())
        nf4=PackedWeight((1,2),'nf4',2,bytes([0x12]),self.weight.scales)
        for weight in (grouped,nf4):
            with self.assertRaises(ValueError):self.call(weight,torch.ones(1,weight.shape[1]).half(),'decoded')

    def test_overflow_in_weight_decode_or_half_output_is_rejected(self):
        weight=PackedWeight((1,1),'int8',1,bytes([127]),np.array([65504.],np.float16).tobytes())
        with self.assertRaisesRegex(ValueError,'decoded weight'):self.call(weight,torch.zeros(1,1).half(),'decoded')
        weight=PackedWeight((1,1),'int8',1,bytes([127]),np.array([511.],np.float16).tobytes())
        with self.assertRaisesRegex(ValueError,'Projection'):self.call(weight,torch.full((1,1),65504.).half(),'exact')


class FactorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.call=helpers('replace_last_output').replace_last_output
        self.name='mlp.down_proj.weight';a=torch.zeros(32,1536).half();up=torch.zeros(576,32).half()
        self.prior=LowRankCorrection((576,1536),32,a.numpy().tobytes(),up.numpy().tobytes())
        self.factors={i:{'untouched':object()} for i in range(30)};self.factors[29][self.name]=self.prior

    def test_only_the_final_down_output_changes_without_mutating_prior(self):
        up=torch.full((576,32),.125).half();result=self.call(self.factors,up)
        for i in range(29):self.assertIs(result[i],self.factors[i])
        self.assertIsNot(result[29],self.factors[29]);self.assertIs(result[29]['untouched'],self.factors[29]['untouched'])
        self.assertEqual(result[29][self.name].down,self.prior.down);self.assertEqual(result[29][self.name].rank,32)
        self.assertEqual(self.factors[29][self.name],self.prior)
        torch.testing.assert_close(result[29][self.name].tensors()[1],up,rtol=0,atol=0)

    def test_bad_output_shape_dtype_nonfinite_or_layer_set_is_rejected(self):
        for up in (torch.zeros(576,16).half(),torch.zeros(576,32),torch.full((576,32),float('nan')).half()):
            with self.assertRaises(ValueError):self.call(self.factors,up)
        with self.assertRaises(ValueError):self.call({i:v for i,v in self.factors.items() if i!=0},torch.zeros(576,32).half())


class FitIdentityTests(unittest.TestCase):
    def record(self,**changes):return dict(status='passed',healthy=True,source_sha256='source',fitted_factors_sha256='factors',**changes)

    def call(self,row,digest='factors'):
        base=SimpleNamespace(read=lambda n:row,sha=lambda p:digest if p.suffix=='.npz' else 'fit-record')
        return helpers('identity',base=base,OUT=Path('unused'),source_identity=lambda:dict(source_sha256='source')).identity()

    def test_complete_fit_binds_parameters_and_record(self):
        self.assertEqual(self.call(self.record()),dict(source_sha256='source',fit_sha256='fit-record'))

    def test_changed_coefficients_or_source_cannot_reach_evaluation(self):
        with self.assertRaises(AssertionError):self.call(self.record(),digest='changed')
        row=self.record();row['source_sha256']='changed'
        with self.assertRaises(AssertionError):self.call(row)

    def test_failed_or_unhealthy_fit_cannot_reach_evaluation(self):
        for key,value in (('status','failed'),('healthy',False)):
            row=self.record();row[key]=value
            with self.assertRaises(AssertionError):self.call(row)


if __name__=='__main__':unittest.main()

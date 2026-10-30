"""Independent CPU checks of anchored low-rank output fitting and down routing."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch_npu_lowbit import PackedWeight,quantize_weight
from torch_npu_lowbit_correction import LowRankCorrection,fit_lowrank_correction


def helpers():
    path=Path(__file__).parent/'npunlock-probe/research/w8_output_quality.py'
    names={'decoded_weight','fit_output_residual','fit_basis_residual','replace_down'}
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    env=dict(torch=torch,np=np,F=torch.nn.functional,LowRankCorrection=LowRankCorrection,
        base=SimpleNamespace(mixed=SimpleNamespace(PackedWeight=PackedWeight)),DOWN='mlp.down_proj.weight')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return SimpleNamespace(**{n:env[n] for n in names})


def zeros(shape):
    return PackedWeight(shape,'int8',shape[1],bytes(shape[0]*shape[1]),np.ones(shape[0],dtype=np.float16).tobytes())


class OutputFitTests(unittest.TestCase):
    def test_full_rank_matches_independent_anchored_normal_equations(self):
        torch.manual_seed(4801);h=helpers();w=torch.randn(4,7).half();q=quantize_weight(w,format='int8')
        x=torch.randn(40,7).half();target=torch.randn(40,4).double()/16
        result=h.fit_output_residual(w,q,x,target,rank=4);a,b=result.tensors()
        covariance=x.double().t()@x.double()/len(x);ridge=.01*covariance.diagonal().mean()
        prior=w.double()-h.decoded_weight(q)
        expected=torch.linalg.solve(covariance+ridge*torch.eye(7),target.t()@x.double()/len(x)+ridge*prior,left=False)
        torch.testing.assert_close(b.double()@a.double(),expected,rtol=.004,atol=2e-5)

    def test_rank_two_target_reduces_output_error(self):
        torch.manual_seed(4802);h=helpers();w=torch.zeros(5,7).half();q=zeros((5,7))
        x=torch.eye(7).repeat(8,1).half();coefficient=torch.randn(5,2).double()@torch.randn(2,7).double()/8
        target=x.double()@coefficient.t();result=h.fit_output_residual(w,q,x,target,rank=2);a,b=result.tensors()
        actual=(x.double()@a.double().t())@b.double().t()
        self.assertLess(float((actual-target).square().sum()/target.square().sum()),.0002)

    def test_unobserved_inputs_preserve_prior_without_factor_overflow(self):
        torch.manual_seed(4803);h=helpers();w=torch.randn(4,7).half();q=quantize_weight(w,format='int8')
        result=h.fit_output_residual(w,q,torch.zeros(8,7),torch.zeros(8,4),rank=4);a,b=result.tensors()
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
        torch.testing.assert_close(b.double()@a.double(),w.double()-h.decoded_weight(q),rtol=.004,atol=1e-6)

    def test_weight_error_target_matches_existing_covariance_objective(self):
        torch.manual_seed(4804);h=helpers();w=torch.randn(5,7).half();q=quantize_weight(w,format='int8');x=torch.randn(50,7).half()
        target=x.double()@(w.double()-h.decoded_weight(q)).t()
        old=fit_lowrank_correction(w,q,x,rank=3);new=h.fit_output_residual(w,q,x,target,rank=3)
        a,b=old.tensors();aa,bb=new.tensors()
        torch.testing.assert_close(bb.double()@aa.double(),b.double()@a.double(),rtol=.01,atol=2e-6)

    def test_invalid_inputs_targets_rank_and_formats_reject(self):
        h=helpers();w=torch.ones(4,7).half();q=zeros((4,7));x=torch.ones(8,7);target=torch.zeros(8,4)
        for bad in (torch.zeros(8,3),torch.zeros(7,4),torch.full((8,4),float('nan')),torch.zeros(8,4,dtype=torch.int64)):
            with self.subTest(target=bad.shape),self.assertRaises(ValueError):h.fit_output_residual(w,q,x,bad,rank=2)
        for bad in (torch.zeros(0,7),torch.zeros(4097,7),torch.zeros(8,6),torch.full((8,7),float('inf'))):
            with self.subTest(inputs=bad.shape),self.assertRaises(ValueError):h.fit_output_residual(w,q,bad,target,rank=2)
        for rank in (0,5,True):
            with self.subTest(rank=rank),self.assertRaises(ValueError):h.fit_output_residual(w,q,x,target,rank=rank)
        with self.assertRaises(ValueError):h.fit_output_residual(w,quantize_weight(w,format='int4'),x,target,rank=2)

    def test_only_down_factors_are_replaced_at_every_layer(self):
        h=helpers();names=('q','k','v','o','gate','up','mlp.down_proj.weight');modules=[];factors={}
        for i in range(30):
            for name in names:modules.append((name,SimpleNamespace(down=torch.zeros(2,3),up=torch.zeros(4,2))))
            factors[i]={names[-1]:SimpleNamespace(tensors=lambda i=i:(torch.full((2,3),i+1.),torch.full((4,2),i+2.)))}
        h.replace_down(modules,factors)
        for index,(name,module) in enumerate(modules):
            if name==names[-1]:
                self.assertTrue((module.down==index//7+1).all() and (module.up==index//7+2).all())
            else:self.assertEqual(int(torch.count_nonzero(module.down))+int(torch.count_nonzero(module.up)),0)

    def test_basis_fit_matches_independent_regression_and_keeps_input_bytes(self):
        torch.manual_seed(4805);h=helpers();a=torch.randn(2,7).half()/4;b=torch.randn(5,2).half()/4
        prior=LowRankCorrection((5,7),2,a.numpy().tobytes(),b.numpy().tobytes());x=torch.randn(40,7).half()
        target=torch.randn(40,5).double()/8;fitted=h.fit_basis_residual(x,target,prior)
        self.assertEqual(fitted.down,prior.down)
        z=torch.nn.functional.linear(x.float(),a.float()).half().double()
        covariance=z.t()@z/len(z);ridge=.01*covariance.diagonal().mean()
        expected=(target.t()@z/len(z)+ridge*b.double())@torch.linalg.inv(covariance+ridge*torch.eye(2))
        torch.testing.assert_close(fitted.tensors()[1].double(),expected,rtol=.002,atol=2e-5)

    def test_basis_zero_activations_preserve_prior(self):
        h=helpers();a=torch.full((2,7),.125).half();b=torch.full((5,2),.25).half()
        prior=LowRankCorrection((5,7),2,a.numpy().tobytes(),b.numpy().tobytes())
        fitted=h.fit_basis_residual(torch.zeros(8,7),torch.zeros(8,5),prior)
        self.assertEqual(fitted,prior)

    def test_basis_invalid_shapes_and_targets_reject(self):
        h=helpers();a=torch.ones(2,7).half();b=torch.ones(5,2).half()
        prior=LowRankCorrection((5,7),2,a.numpy().tobytes(),b.numpy().tobytes())
        for x,y in ((torch.zeros(8,6),torch.zeros(8,5)),(torch.zeros(0,7),torch.zeros(0,5)),
            (torch.zeros(8,7),torch.zeros(8,4)),(torch.zeros(8,7),torch.full((8,5),float('nan')))):
            with self.subTest(x=x.shape,y=y.shape),self.assertRaises(ValueError):h.fit_basis_residual(x,y,prior)


if __name__=='__main__':unittest.main()

"""Offline contracts for device-observed QKV output-factor calibration."""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch_npu_lowbit_correction import LowRankCorrection

PROFILES=('qkv-0.25','qkv-0.50','qkv-1.00')
NAMES=tuple('self_attn.'+name+'_proj.weight' for name in ('q','k','v'))


def helpers(*names,**extra):
    path=Path(__file__).parent/'npunlock-probe/research/w8_qkv_quality.py'
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    env=dict(torch=torch,F=F,np=np,LowRankCorrection=LowRankCorrection,PROFILES=PROFILES,NAMES=NAMES,**extra)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return SimpleNamespace(**{name:env[name] for name in names})


class DeviceBasisTests(unittest.TestCase):
    def setUp(self):
        self.z=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[-1.,1.]]).half()
        down=torch.tensor([[1.,0.,.5],[0.,1.,.25]]).half()
        up=torch.zeros(2,2).half()
        self.prior=LowRankCorrection((2,3),2,down.numpy().tobytes(),up.numpy().tobytes())
        self.main=torch.full((4,2),.5).half()
        self.target=(self.main+self.z@torch.tensor([[.25,-.5],[.5,.25]]).half()).half()
        self.fit=helpers('fit_device_basis').fit_device_basis

    def test_actual_main_and_observed_basis_fit_the_fp16_target(self):
        factor,metrics=self.fit(self.z,self.main,self.target,self.prior,1.)
        output=(self.main+F.linear(self.z.float(),factor.tensors()[1].float()).half()).half()
        self.assertTrue(metrics['accepted'])
        self.assertLess(metrics['proxy_after_sse'],metrics['proxy_before_sse']*.001)
        self.assertEqual(float((output.double()-self.target.double()).square().sum()),metrics['proxy_after_sse'])

    def test_closed_ridge_solution_and_half_output_boundaries(self):
        factor,_=self.fit(self.z,self.main,self.target,self.prior,1.)
        z=self.z.double();h=z.t()@z/len(z);h+=torch.eye(2)*(.01*h.diagonal().mean())
        rhs=z.t()@(self.target-self.main).double()/len(z)
        expected=torch.linalg.inv(h)@rhs
        torch.testing.assert_close(factor.tensors()[1],expected.t().half(),rtol=0,atol=0)

    def test_blending_preserves_rank_down_factor_and_inputs(self):
        snapshots=[t.clone() for t in (self.z,self.main,self.target)];norms=[]
        for alpha in (.25,.5,1.):
            factor,metrics=self.fit(self.z,self.main,self.target,self.prior,alpha)
            self.assertEqual((factor.shape,factor.rank,factor.down),(self.prior.shape,2,self.prior.down))
            self.assertLessEqual(metrics['proxy_after_sse'],metrics['proxy_before_sse'])
            norms.append(float(factor.tensors()[1].float().norm()))
        self.assertEqual(norms,sorted(norms))
        for actual,expected in zip((self.z,self.main,self.target),snapshots):torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_unobserved_basis_and_already_exact_target_keep_the_prior(self):
        for z,target in ((torch.zeros_like(self.z),self.target),(self.z,self.main)):
            factor,metrics=self.fit(z,self.main,target,self.prior,1.)
            self.assertEqual(factor,self.prior);self.assertFalse(metrics['accepted'])

    def test_invalid_observations_or_blends_reject(self):
        nan=self.z.clone();nan[0,0]=float('nan')
        for z,main,target,alpha in ((self.z.float(),self.main,self.target,1.),(nan,self.main,self.target,1.),
            (self.z[:0],self.main[:0],self.target[:0],1.),(self.z[:,:1],self.main,self.target,1.),
            (self.z,self.main[:,:1],self.target,1.),(self.z,self.main,self.target,.75),(self.z,self.main,self.target,True)):
            with self.subTest(alpha=alpha),self.assertRaises(ValueError):self.fit(z,main,target,self.prior,alpha)

    def test_half_overflow_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError,'overflows'):
            self.fit(torch.full_like(self.z,.1),torch.full_like(self.main,60000),torch.full_like(self.target,-60000),self.prior,1.)


class QKVProtocolTests(unittest.TestCase):
    def test_calibration_and_internal_sets_remain_disjoint(self):
        partition=helpers('partition').partition;rows=[dict(index=i) for i in range(40)]
        fit={r['index'] for r in partition(rows,True)};internal={r['index'] for r in partition(rows,False)}
        self.assertEqual((len(fit),len(internal)),(32,8));self.assertFalse(fit&internal)
        self.assertEqual(internal,{4,9,14,19,24,29,34,39})
        for bad in (rows[:-1],rows[::-1],[rows[0]]*40):
            with self.assertRaises(ValueError):partition(bad,True)

    def test_reused_evaluator_is_restored_on_success_and_error(self):
        names=('parameters','stack_for','identity','frozen_selection')
        old={name:object() for name in names};new={name:object() for name in names}
        previous=SimpleNamespace(**old)
        def run(value):
            for name in names:self.assertIs(getattr(previous,name),new[name])
            if value=='fail':raise RuntimeError('unexpected execution failure')
            return value
        previous.internal=run
        evaluate=helpers('evaluate',study=SimpleNamespace(previous=previous),**new).evaluate
        self.assertEqual(evaluate('internal','ok'),'ok')
        with self.assertRaisesRegex(RuntimeError,'unexpected'):evaluate('internal','fail')
        for name in names:self.assertIs(getattr(previous,name),old[name])

    def fixture(self,directory):
        down=np.ones((2,3),np.float16).tobytes();up=np.zeros((4,2),np.float16).tobytes()
        original={i:{name:LowRankCorrection((4,3),2,down,up) for name in NAMES} for i in range(30)}
        packed={i:{name:b'unchanged packed codes and scales' for name in NAMES} for i in range(30)}
        path=Path(directory)/'qkv-fit.npz'
        arrays={f'{p}/{i}/{n}/up':np.full((4,2),.25,np.float16) for p in PROFILES for i in range(30) for n in NAMES}
        np.savez_compressed(path,**arrays)
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        metadata=dict(status='passed',healthy=True,source_sha256='source',fitted_factors_sha256=sha(path))
        base=SimpleNamespace(mixed=SimpleNamespace(weights=lambda:packed),corrections=lambda rank:copy.deepcopy(original),
            read=lambda name:metadata,sha=sha)
        load=helpers('parameters',base=base,OUT=Path(directory),source_identity=lambda:dict(source_sha256='source')).parameters
        return load,original,packed,metadata,path,arrays

    def test_parameter_loading_changes_only_qkv_output_factors(self):
        with tempfile.TemporaryDirectory() as directory:
            load,original,packed,_,_,_=self.fixture(directory)
            weights,factors=load('qkv-0.50');self.assertIs(weights,packed)
            for i in range(30):
                for name in NAMES:
                    self.assertEqual(factors[i][name].down,original[i][name].down)
                    self.assertEqual(factors[i][name].rank,2)
                    self.assertTrue((factors[i][name].tensors()[1]==.25).all())
            self.assertEqual(load('baseline')[1],original)

    def test_parameter_loading_rejects_failed_identity_and_wrong_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            load,_,_,metadata,path,arrays=self.fixture(directory)
            with self.assertRaises(ValueError):load('unknown')
            metadata['healthy']=False
            with self.assertRaises(AssertionError):load('qkv-1.00')
            metadata['healthy']=True;metadata['source_sha256']='changed'
            with self.assertRaises(AssertionError):load('qkv-1.00')
            metadata['source_sha256']='source';arrays[next(iter(arrays))]=np.ones((4,2),np.float32)
            np.savez_compressed(path,**arrays);metadata['fitted_factors_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(AssertionError):load('qkv-0.25')


if __name__=='__main__':unittest.main()

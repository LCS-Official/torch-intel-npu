"""Offline contracts for whole-model output fitting with Half forward factors."""
import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F


def helpers(*names,**extra):
    path=Path(__file__).parent/'npunlock-probe/research/w8_logits_quality.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('FRESH','PROFILES')}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
    env=dict(torch=torch,F=F,np=np,Path=Path,__file__=str(path),NAMES=tuple('self_attn.'+n+'_proj.weight' for n in ('q','k','v')),**constants,**extra)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return SimpleNamespace(**{name:env[name] for name in names})


class LogitsLossTests(unittest.TestCase):
    def setUp(self):self.loss=helpers('logits_loss').logits_loss

    def test_exact_logits_have_zero_loss_and_gradient(self):
        x=torch.tensor([[1.,2.,3.]],requires_grad=True);loss=self.loss(x,x.detach())
        loss.backward();self.assertEqual(float(loss.detach()),0.);torch.testing.assert_close(x.grad,torch.zeros_like(x))

    def test_matches_relative_l2_teacher_kl_and_margin_formula(self):
        target=torch.tensor([[1.,2.,3.],[3.,1.,2.]]);actual=(target+torch.tensor([[.1,0.,0.],[0.,0.,.02]])).requires_grad_()
        square=(actual-target).square().sum(-1)/target.square().sum(-1)
        p=target.double().softmax(-1);q=actual.double().softmax(-1)
        kl=(p*(p.log()-q.log())).sum(-1).float()
        expected=(square+.1*kl+2*torch.relu(square-.0009)).mean()
        torch.testing.assert_close(self.loss(actual,target),expected,rtol=1e-4,atol=1e-8)

    def test_uniform_shift_is_penalized_even_when_probabilities_match(self):
        target=torch.tensor([[1.,2.,3.]])
        self.assertGreater(float(self.loss(target+1,target)),.1)

    def test_teacher_is_detached_and_actual_gets_gradient(self):
        target=torch.tensor([[1.,2.,3.]],requires_grad=True);actual=torch.tensor([[1.1,2.,3.]],requires_grad=True)
        self.loss(actual,target).backward();self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(actual.grad).all());self.assertGreater(float(actual.grad.abs().sum()),0.)

    def test_invalid_logits_reject(self):
        good=torch.ones(2,4)
        for actual,target in ((good.half(),good),(good,good[:,:2]),(good[:0],good[:0]),(good[0],good[0]),(good*float('nan'),good)):
            with self.assertRaises(ValueError):self.loss(actual,target)


class OutputMasterTests(unittest.TestCase):
    def fixture(self):
        model=torch.nn.Module();model.original=torch.nn.Parameter(torch.ones(1));model.factors=torch.nn.ModuleList();modules=[]
        for i in range(30):
            for name in ('self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight'):
                module=torch.nn.Module();module.register_buffer('up',torch.full((4,32),.125).half());module.register_buffer('down',torch.ones(32,3).half())
                model.factors.append(module);modules.append((name,module))
        return model,modules

    def test_only_qkv_outputs_train_with_half_forward_and_master_gradient(self):
        model,modules=self.fixture();h=helpers('trainable_outputs','bind_outputs');tunables=h.trainable_outputs(model,modules);h.bind_outputs(tunables)
        self.assertEqual(len(tunables),90);self.assertFalse(model.original.requires_grad)
        _,_,module,master,prior=tunables[0]
        self.assertEqual(module.up.dtype,torch.float16);torch.testing.assert_close(module.up,prior.half())
        F.linear(torch.ones(1,32),module.up.float()).sum().backward()
        self.assertIsNotNone(master.grad);self.assertGreater(float(master.grad.abs().sum()),0.)
        self.assertFalse(modules[3][1].up.requires_grad);self.assertFalse(module.down.requires_grad)

    def test_updated_master_is_rebound_before_snapshot_or_forward(self):
        model,modules=self.fixture();h=helpers('trainable_outputs','bind_outputs');tunables=h.trainable_outputs(model,modules);h.bind_outputs(tunables)
        _,_,module,master,prior=tunables[0];down=module.down.clone()
        with torch.no_grad():master.add_(.001)
        h.bind_outputs(tunables);torch.testing.assert_close(module.up,master.half(),rtol=0,atol=0)
        self.assertFalse(torch.equal(module.up,prior.half()));torch.testing.assert_close(module.down,down,rtol=0,atol=0)

    def test_nonfinite_and_half_overflow_reject(self):
        for value in (float('nan'),1e10):
            model,modules=self.fixture();h=helpers('trainable_outputs','bind_outputs');tunables=h.trainable_outputs(model,modules)
            with torch.no_grad():tunables[0][3].fill_(value)
            with self.assertRaises(ValueError):h.bind_outputs(tunables)

    def test_incomplete_or_wrong_rank_reject(self):
        model,modules=self.fixture();train=helpers('trainable_outputs').trainable_outputs
        with self.assertRaises(ValueError):train(model,modules[:-7])
        modules[0][1].up=torch.ones(4,16).half()
        with self.assertRaises(ValueError):train(model,modules)


class LogitsProtocolTests(unittest.TestCase):
    def test_internal_runner_restores_globals_after_failure(self):
        old_stack,old_identity=object(),object();new_stack,new_identity=object(),object()
        runner=SimpleNamespace(stack_for=old_stack,identity=old_identity)
        def fail(profile):
            self.assertIs(runner.stack_for,new_stack);self.assertIs(runner.identity,new_identity);raise RuntimeError('device failure')
        runner.internal=fail
        call=helpers('internal',study=SimpleNamespace(previous=runner),stack_for=new_stack,identity=new_identity).internal
        with self.assertRaisesRegex(RuntimeError,'device failure'):call('logits-e1')
        self.assertIs(runner.stack_for,old_stack);self.assertIs(runner.identity,old_identity)

    def test_fresh_cases_freeze_in_owned_directory_despite_legacy_import(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);written=[]
            common=SimpleNamespace(model=lambda:(object(),object()),chat=lambda *a:torch.tensor([[1,2]]),trajectory=lambda *a:torch.ones(1,64,dtype=torch.int64))
            base=SimpleNamespace(OUT=out,mixed=SimpleNamespace(OUT=out,PROMPTS=(),HOLDOUT=(),FRESH=(),common=common),sha=lambda p:'sha')
            def write(name,data):
                self.assertEqual(base.OUT,out);self.assertEqual(base.mixed.OUT,out);written.append((name,data))
            base.write=write;original_import=builtins.__import__
            def legacy_import(name,*args,**kwargs):
                if name=='w8_guarded_quality':base.OUT=base.mixed.OUT=out/'wrong';return SimpleNamespace(FRESH=())
                return original_import(name,*args,**kwargs)
            call=helpers('fresh_data',base=base,OUT=out).fresh_data
            with patch('builtins.__import__',side_effect=legacy_import):call()
            self.assertEqual(written[0][0],'fresh');self.assertEqual(len(written[0][1]['cases']),8)
            self.assertEqual(written[0][1]['status'],'frozen')


if __name__=='__main__':unittest.main()

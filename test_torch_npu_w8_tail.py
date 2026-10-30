"""Offline upper-tail objective and frozen prompt partition contracts."""
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
    root=Path(__file__).parent;path=root/'npunlock-probe/research/w8_tail_quality.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('CALIBRATION','FRESH','PROFILES')}
    old=ast.parse((root/'npunlock-probe/research/w8_logits_quality.py').read_text(encoding='utf-8'))
    constants['REGRESSION']=next(ast.literal_eval(n.value) for n in old.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='FRESH')
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
    env=dict(torch=torch,F=F,np=np,Path=Path,__file__=str(path),**constants);env.update(extra)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return SimpleNamespace(**{name:env[name] for name in names})


class TailLossTests(unittest.TestCase):
    def setUp(self):self.h=helpers('components','tail_loss')

    def test_components_match_relative_l2_and_teacher_kl(self):
        target=torch.tensor([[1.,2.,3.],[3.,1.,2.]]);actual=target+torch.tensor([[.1,0.,0.],[0.,.02,0.]])
        values=self.h.components(actual,target);p=target.double().softmax(-1);q=actual.double().softmax(-1)
        torch.testing.assert_close(values[:,0],(actual-target).square().sum(-1)/target.square().sum(-1))
        torch.testing.assert_close(values[:,1],(p*(p.log()-q.log())).sum(-1).float(),rtol=1e-3,atol=1e-7)

    def test_full_formula_uses_ceiling_top_ten_percent_and_fixed_budgets(self):
        target=torch.arange(1,45).reshape(11,4).float()/10
        actual=target+torch.linspace(.01,.3,11)[:,None];values=self.h.components(actual,target)
        budget=values*.8;l2,kl=values.unbind(-1)
        expected=(l2+.1*kl+2*torch.relu(l2-.0009)).mean()
        expected+=.5*(l2.sort().values[-2:].mean()+kl.sort().values[-2:].mean())
        expected+=2*(values-budget).relu().mean(0).sum()
        torch.testing.assert_close(self.h.tail_loss(actual,target,budget),expected,rtol=0,atol=0)

    def test_short_sequence_has_at_least_one_tail_position(self):
        target=torch.tensor([[1.,2.,3.]]);actual=target+.1;v=self.h.components(actual,target);l2,kl=v[0]
        expected=l2+.1*kl+2*torch.relu(l2-.0009)+.5*(l2+kl)
        torch.testing.assert_close(self.h.tail_loss(actual,target,v),expected)

    def test_improvements_cannot_cancel_other_budget_excess(self):
        target=torch.tensor([[1.,2.,3.],[1.,2.,3.]]);actual=target+torch.tensor([[.2,0.,0.],[0.,.02,0.]])
        values=self.h.components(actual,target);budget=values.clone();budget[0]*=.5;budget[1]*=3
        difference=self.h.tail_loss(actual,target,budget)-self.h.tail_loss(actual,target,values*4)
        torch.testing.assert_close(difference,values[0].sum()*.5,rtol=1e-5,atol=1e-8)

    def test_high_error_fields_receive_tail_gradients(self):
        values=torch.tensor([[.0001,.0001],[.0002,.02],[.01,.0002]],requires_grad=True)
        loss=helpers('tail_loss',components=lambda a,b:a).tail_loss(values,None,values.detach())
        loss.backward();self.assertGreater(float(values.grad[2,0]),float(values.grad[0,0]))
        self.assertGreater(float(values.grad[1,1]),float(values.grad[0,1]))

    def test_budget_boundary_uses_the_symmetric_subgradient(self):
        values=torch.tensor([[.0001,.0001],[.0002,.02],[.01,.0002]],requires_grad=True)
        l2,kl=values.unbind(-1);count=1
        plain=(l2+.1*kl+2*torch.relu(l2-.0009)).mean()+.5*(l2.topk(count).values.mean()+kl.topk(count).values.mean())
        plain_gradient=torch.autograd.grad(plain,values)[0]
        loss=helpers('tail_loss',components=lambda a,b:a).tail_loss(values,None,values.detach())
        gradient=torch.autograd.grad(loss,values)[0]
        torch.testing.assert_close(gradient-plain_gradient,torch.ones_like(values)/len(values))

    def test_teacher_and_budget_are_detached(self):
        target=torch.tensor([[1.,2.,3.]],requires_grad=True);actual=torch.tensor([[1.1,2.,3.]],requires_grad=True)
        budget=torch.zeros(1,2,requires_grad=True)
        self.h.tail_loss(actual,target,budget).backward()
        self.assertIsNone(target.grad);self.assertIsNone(budget.grad);self.assertTrue(torch.isfinite(actual.grad).all())

    def test_exact_logits_have_zero_objective(self):
        x=torch.tensor([[1.,2.,3.]],requires_grad=True);loss=self.h.tail_loss(x,x.detach(),torch.zeros(1,2))
        loss.backward();self.assertEqual(float(loss.detach()),0.);torch.testing.assert_close(x.grad,torch.zeros_like(x))

    def test_invalid_logits_and_budget_reject(self):
        x=torch.ones(2,4)
        for actual in (x.half(),x[:0],x[:,0],x*float('nan')):
            with self.assertRaises(ValueError):self.h.components(actual,x)
        for budget in (torch.zeros(2,2).half(),torch.zeros(2,1),-torch.ones(2,2),torch.full((2,2),float('nan'))):
            with self.assertRaises(ValueError):self.h.tail_loss(x,x,budget)


class TailDataTests(unittest.TestCase):
    def test_all_existing_output_factors_are_trainable_but_inputs_stay_fixed(self):
        model=torch.nn.Module();model.original=torch.nn.Parameter(torch.ones(1));modules=[]
        names=('self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight')
        for i in range(210):
            module=torch.nn.Module();module.register_buffer('up',torch.ones(4,32).half());module.register_buffer('down',torch.ones(32,3).half())
            model.add_module('factor'+str(i),module);modules.append((names[i%7],module))
        train=helpers('trainable_outputs').trainable_outputs;rows=train(model,modules)
        self.assertEqual(len(rows),210);self.assertFalse(model.original.requires_grad)
        self.assertEqual({r[1] for r in rows},set(names));self.assertTrue(all(r[3].requires_grad for r in rows))
        self.assertTrue(all(not r[2].down.requires_grad for r in rows))
        with self.assertRaises(ValueError):train(model,modules[:-1])

    def test_fit_order_interleaves_disjoint_original_and_added_rows(self):
        original=[dict(index=i) for i in range(40) if i%5!=4];added=[dict(index=i) for i in range(40,56)]
        fit=helpers('fit_rows').fit_rows;rows=fit(original,added)
        self.assertEqual(rows[2::3],added);self.assertEqual(len({r['index'] for r in rows}),48)
        self.assertFalse({r['index'] for r in rows}&set(range(4,40,5)))
        for bad_original,bad_added in ((original[:-1],added),(original,added[::-1]),(original,[added[0]]*16)):
            with self.assertRaises(ValueError):fit(bad_original,bad_added)

    def test_calibration_regression_and_fresh_prompts_are_distinct(self):
        groups=helpers('prompt_groups').prompt_groups()
        self.assertEqual(tuple(map(len,groups)),(16,8,8))
        self.assertEqual(len({p for g in groups for _,p in g}),32)

    def test_overlap_and_duplicate_tags_reject(self):
        groups=helpers('prompt_groups').prompt_groups();fresh=list(groups[2]);fresh[0]=groups[0][0]
        with self.assertRaisesRegex(ValueError,'overlap'):helpers('prompt_groups',FRESH=tuple(fresh)).prompt_groups()
        fresh=list(groups[2]);fresh[0]=(fresh[1][0],fresh[0][1])
        with self.assertRaises(ValueError):helpers('prompt_groups',FRESH=tuple(fresh)).prompt_groups()

    def test_generation_freezes_owned_groups_and_recovers_legacy_output_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);written={};common=SimpleNamespace(model=lambda:(object(),object()),
                chat=lambda *a:torch.tensor([[1,2]]),trajectory=lambda *a:torch.ones(1,64,dtype=torch.int64))
            base=SimpleNamespace(OUT=out,mixed=SimpleNamespace(OUT=out,PROMPTS=(),HOLDOUT=(),FRESH=(),common=common),sha=lambda p:'sha')
            def write(name,value):
                self.assertEqual(base.OUT,out);self.assertEqual(base.mixed.OUT,out);written[name]=value
            base.write=write;original_import=builtins.__import__
            def legacy_import(name,*args,**kwargs):
                if name=='w8_guarded_quality':base.OUT=base.mixed.OUT=out/'wrong';return SimpleNamespace(FRESH=())
                return original_import(name,*args,**kwargs)
            call=helpers('data','prompt_groups',OUT=out,base=base).data
            with patch('builtins.__import__',side_effect=legacy_import):call()
            self.assertEqual({k:len(v['cases']) for k,v in written.items()},{'tail-data':16,'regression':8,'fresh':8})
            self.assertTrue(all(v['status']=='frozen' for v in written.values()))
            self.assertEqual([r['index'] for r in written['tail-data']['cases']],list(range(40,56)))


if __name__=='__main__':unittest.main()

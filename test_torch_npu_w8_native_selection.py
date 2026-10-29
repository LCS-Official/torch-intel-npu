"""Offline blending, native budget alignment and frozen selection contracts."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch_npu_lowbit_correction import LowRankCorrection


def helpers(*names,**extra):
    path=Path(__file__).parent/'npunlock-probe/research/w8_native_selection.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
    ranking=lambda s:(s['positions']-s['quality_passed'],s['max_relative_l2'],s['mean_relative_l2'])
    env=dict(torch=torch,math=math,LowRankCorrection=LowRankCorrection,base=SimpleNamespace(mixed=SimpleNamespace(common=SimpleNamespace(ranking=ranking))))
    env.update(extra);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return SimpleNamespace(**{name:env[name] for name in names})


def record(objective=.2,passed=1500):
    return dict(status='passed',healthy=True,objective=objective,summary=dict(positions=1657,quality_passed=passed,
        max_relative_l2=.1,mean_relative_l2=.02,max_kl=.01,next_token_equal=1600))


class BlendTests(unittest.TestCase):
    def setUp(self):
        down=torch.ones(2,3).half();up=torch.tensor([[.125,.25],[.5,1.]]).half()
        self.prior=LowRankCorrection((2,3),2,down.numpy().tobytes(),up.numpy().tobytes())
        self.fitted=LowRankCorrection((2,3),2,self.prior.down,(up+.125).numpy().tobytes())
        self.blend=helpers('blend_factor').blend_factor

    def test_full_blend_reproduces_nonzero_fitted_bytes_and_keeps_input_basis(self):
        result=self.blend(self.prior,self.fitted,1.)
        self.assertEqual(result,self.fitted);self.assertEqual(result.down,self.prior.down)

    def test_full_blend_is_value_equal_when_signed_zero_is_normalized(self):
        up=torch.tensor([[-0.,.25],[.5,-0.]]).half()
        fitted=LowRankCorrection((2,3),2,self.prior.down,up.numpy().tobytes())
        result=self.blend(self.prior,fitted,1.)
        torch.testing.assert_close(result.tensors()[1],up,rtol=0,atol=0)
        self.assertNotEqual(result.up,fitted.up);self.assertEqual(result.down,fitted.down)

    def test_partial_blends_round_to_half_without_mutating_inputs(self):
        for alpha in (.25,.5):
            result=self.blend(self.prior,self.fitted,alpha)
            expected=(self.prior.tensors()[1].float()+alpha*(self.fitted.tensors()[1].float()-self.prior.tensors()[1].float())).half()
            torch.testing.assert_close(result.tensors()[1],expected,rtol=0,atol=0)
            self.assertEqual(result.down,self.prior.down);self.assertEqual(result.rank,2)

    def test_changed_basis_shape_or_unlisted_blend_rejects(self):
        altered=LowRankCorrection((2,3),2,(torch.ones(2,3)*2).half().numpy().tobytes(),self.fitted.up)
        for prior,fitted,alpha in ((self.prior,altered,.5),(self.prior,self.fitted,.75),(self.prior,self.fitted,True)):
            with self.assertRaises(ValueError):self.blend(prior,fitted,alpha)


class NativeBudgetTests(unittest.TestCase):
    def test_budget_matches_prompt_and_absolute_position(self):
        baseline=dict(positions=[dict(tag='a',position=8,components=[.1,.2]),dict(tag='b',position=8,components=[9.,9.]),dict(tag='a',position=9,components=[.3,.4])])
        actual=helpers('budget_for').budget_for(baseline,'a',8,2)
        torch.testing.assert_close(actual,torch.tensor([[.1,.2],[.3,.4]]),rtol=0,atol=0)

    def test_missing_duplicate_or_invalid_native_budget_rejects(self):
        call=helpers('budget_for').budget_for;row=dict(tag='a',position=8,components=[.1,.2])
        for rows,count in (([row],2),([row,row],1),([dict(row,components=[-1.,0.])],1),([dict(row,components=[float('nan'),0.])],1)):
            with self.assertRaises(ValueError):call(dict(positions=rows),'a',8,count)


class NativeChoiceTests(unittest.TestCase):
    def setUp(self):self.choose=helpers('choose_calibration').choose_calibration

    def test_real_native_objective_selects_candidate(self):
        records=dict(baseline=record(.2),one=record(.1,1490),two=record(.15,1510))
        self.assertEqual(self.choose(records),('one','one',{}))

    def test_non_improving_candidate_does_not_promote(self):
        self.assertEqual(self.choose(dict(baseline=record(.2),one=record(.21))),('baseline','one',{}))

    def test_equal_native_losses_use_original_quality_ranking(self):
        self.assertEqual(self.choose(dict(baseline=record(.2),one=record(.1,1500),two=record(.1,1501)))[0],'two')

    def test_known_precision_rejection_is_preserved(self):
        failure=dict(status='failed',healthy=True,local_guard_failure=dict(relative_l2=.0005,max_scaled_error=.012))
        selected,candidate,rejected=self.choose(dict(baseline=record(),one=failure))
        self.assertEqual((selected,candidate),('baseline',None));self.assertEqual(rejected['one'],failure['local_guard_failure'])

    def test_unknown_failure_unhealthy_device_and_bad_baseline_are_fatal(self):
        for bad in (dict(status='failed',healthy=True),dict(status='passed',healthy=False)):
            with self.assertRaises(ValueError):self.choose(dict(baseline=record(),one=bad))
        with self.assertRaises(ValueError):self.choose(dict(baseline=dict(status='failed',healthy=True)))

    def test_incomplete_or_nonfinite_calibration_rejects(self):
        for bad in (record(float('nan')),record(float('inf')),record(-.1),dict(record(),summary=dict(positions=1))):
            with self.assertRaises(ValueError):self.choose(dict(baseline=record(),one=bad))

    def test_only_frozen_candidate_reaches_internal_runner(self):
        calls=[];call=helpers('internal',device_choice=lambda:dict(candidate='one'),parent=SimpleNamespace(internal=lambda p:calls.append(p))).internal
        call('baseline');call('one');self.assertEqual(calls,['baseline','one'])
        with self.assertRaises(ValueError):call('two')

    def test_final_frozen_selection_checks_native_choice_first(self):
        calls=[];call=helpers('frozen_selection',device_choice=lambda:calls.append('native'),PARENT_FROZEN=lambda:calls.append('internal')).frozen_selection
        call();self.assertEqual(calls,['native','internal'])
        def fail():raise ValueError('missing native choice')
        call=helpers('frozen_selection',device_choice=fail,PARENT_FROZEN=lambda:calls.append('bad')).frozen_selection
        with self.assertRaises(ValueError):call()
        self.assertNotIn('bad',calls)


if __name__=='__main__':unittest.main()

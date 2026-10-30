"""Offline contracts for S57 cumulative FFN replacements and peak replays."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import npunlock as symbolic
import torch
import torch.nn.functional as F
from test_torch_npu_w8_gate import helpers as gate_helpers,small_ffn


def helpers():
    path=Path(__file__).parent/'npunlock-probe/research/w8_final_layer_audit.py'
    names={'ffn_reference','replay_rows'}
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    def project(layer,name,x):return F.linear(x.float(),layer[name].float()).half(),None,None
    def compare(actual,reference):return dict(max_abs=float((actual.float()-reference.float()).abs().max()))
    env=dict(torch=torch,F=F,probe=SimpleNamespace(cpu_projection=project),device=SimpleNamespace(compare=compare))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return SimpleNamespace(**{name:env[name] for name in names})


class FfnReplayTests(unittest.TestCase):
    def setUp(self):
        self.call=helpers().ffn_reference
        self.layer={'mlp.gate_proj.weight':torch.tensor([[1.,.5],[-.5,1.],[1.,-1.]]).half(),
                    'mlp.up_proj.weight':torch.tensor([[.5,-1.],[1.,.25],[.5,1.]]).half(),
                    'mlp.down_proj.weight':torch.tensor([[1.,.5,-.5],[-1.,1.,.5]]).half()}
        self.residual=torch.tensor([[[1.,2.],[-.25,.5]]]).half();self.norm=(self.residual*.5).half()

    def test_staged_reference_matches_independent_dense_formula(self):
        result=self.call(self.layer,self.residual,self.norm);x=self.norm.reshape(2,2).float()
        gate=(x@self.layer['mlp.gate_proj.weight'].float().t()).half()
        up=(x@self.layer['mlp.up_proj.weight'].float().t()).half()
        product=(F.silu(gate.float()).half()*up).half()
        down=(product.float()@self.layer['mlp.down_proj.weight'].float().t()).half()
        torch.testing.assert_close(result['output'],(self.residual+down.view_as(self.residual)).half(),rtol=0,atol=0)

    def test_gate_replacement_propagates_and_leaves_up_and_inputs_unchanged(self):
        original=self.call(self.layer,self.residual,self.norm);saved=self.norm.clone();gate=torch.zeros_like(original['gate'])
        changed=self.call(self.layer,self.residual,self.norm,{'gate':gate})
        torch.testing.assert_close(changed['up'],original['up'],rtol=0,atol=0)
        torch.testing.assert_close(changed['output'],self.residual,rtol=0,atol=0)
        torch.testing.assert_close(self.norm,saved,rtol=0,atol=0)

    def test_cumulative_later_boundary_overrides_prior_activation(self):
        original=self.call(self.layer,self.residual,self.norm)
        overrides={'gate':torch.zeros_like(original['gate']),'silu':torch.ones_like(original['silu'])}
        first=self.call(self.layer,self.residual,self.norm,overrides)
        torch.testing.assert_close(first['gated'],original['up'],rtol=0,atol=0)
        overrides['gated']=torch.zeros_like(original['gated'])
        second=self.call(self.layer,self.residual,self.norm,overrides)
        torch.testing.assert_close(second['output'],self.residual,rtol=0,atol=0)

    def test_down_replacement_preserves_half_residual_addition(self):
        down=torch.tensor([[.25,.5],[1.,-2.]]).half()
        changed=self.call(self.layer,self.residual,self.norm,{'down':down})
        torch.testing.assert_close(changed['output'],(self.residual+down.view_as(self.residual)).half(),rtol=0,atol=0)

    def test_invalid_override_or_input_is_rejected(self):
        original=self.call(self.layer,self.residual,self.norm)
        for overrides in ({'output':self.residual},{'gate':original['gate'].float()},
                          {'gate':torch.ones(2,2).half()},{'gate':torch.full_like(original['gate'],float('nan'))}):
            with self.assertRaises(ValueError):self.call(self.layer,self.residual,self.norm,overrides)
        for residual,normalized in ((self.residual.float(),self.norm),(self.residual,self.norm[:,:1]),
                                    (self.residual[:,:0],self.norm[:,:0])):
            with self.assertRaises(ValueError):self.call(self.layer,residual,normalized)


class CoordinateReplayTests(unittest.TestCase):
    def setUp(self):self.call=helpers().replay_rows;self.reference=torch.tensor([[[1.,2.],[3.,4.]]]).half()

    def test_signed_differences_reconstruct_final_peak_without_absolute_attribution(self):
        first=self.reference.clone();first[0,1,0]=3.5
        second=self.reference.clone();second[0,1,0]=2.875
        rows=self.call(self.reference,[('one',first),('two',second)],(0,1,0))
        self.assertEqual([r['signed_delta'] for r in rows],[.5,-.625])
        self.assertEqual(sum(r['signed_delta'] for r in rows),-.125)
        self.assertEqual(rows[-1]['peak_value'],2.875)

    def test_invalid_coordinate_does_not_select_a_different_element(self):
        for coordinate in ((0,1),(0,-1,0),(0,2,0),(0,1,True)):
            with self.assertRaises(ValueError):self.call(self.reference,[('one',self.reference)],coordinate)

    def test_empty_duplicate_nonfinite_or_mismatched_stages_are_rejected(self):
        for stages in ([],[('a',self.reference),('a',self.reference)],[('',self.reference)],
                       [('a',self.reference.float())],[('a',torch.zeros(1,2,1).half())],
                       [('a',torch.full_like(self.reference,float('inf')))]):
            with self.assertRaises(ValueError):self.call(self.reference,stages,(0,1,0))


class DownProbeTests(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).parent/'npunlock-probe/research/w8_final_layer_down.py'
        nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='down_probe_graph']
        assert len(nodes)==1
        env=dict(probe=gate_helpers());exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
        self.call=env['down_probe_graph']

    def test_down_probe_retains_weights_and_exports_only_after_the_gated_boundary(self):
        graph,xml=small_ffn();before=bytes(xml);tail,modified,_,names=self.call(graph,xml)
        self.assertEqual(names,('main','reduced','delta','corrected','output'))
        self.assertEqual([v.name for v in tail.inputs],['residual','gated'])
        self.assertFalse(any(n.op=='Swish' or n.name and ('mlp_gate_' in n.name or 'mlp_up_' in n.name) for n in tail.nodes))
        self.assertEqual(len(ET.fromstring(modified).findall("./layers/layer/rt_info/attribute[@name='decompression']")),1)
        self.assertEqual(xml,before)

    def test_extra_normalization_dependency_is_not_silently_removed(self):
        graph,xml=small_ffn()
        output=symbolic.op('Add',graph.outputs[0],graph.inputs[1],auto_broadcast='numpy',_shape=(2,4),_dtype='f16')
        with self.assertRaisesRegex(ValueError,'normalization'):self.call(symbolic.Graph(graph.inputs,[output]),xml)


if __name__=='__main__':unittest.main()

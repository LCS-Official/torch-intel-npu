"""CPU graph contracts and signed error accounting for the S50 FFN probe."""
import ast
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import npunlock as symbolic
import torch
import torch.nn.functional as F


def helpers():
    path=Path(__file__).parent/'npunlock-probe/research/w8_gate_quality.py'
    names={'linear_chain','serialize_probe','ffn_probe_graph','down_tail_graph','gated_error_parts'}
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    env=dict(symbolic=symbolic,np=np,ET=ET,copy=copy,replace=replace,torch=torch,F=F)
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return SimpleNamespace(**{name:env[name] for name in names})


def small_ffn(*,swapped=False):
    residual=symbolic.input('x',shape=(2,4),dtype='f16');norm=symbolic.input('normalized',shape=(2,4),dtype='f16')
    def linear(x,name,n):
        prefix='mlp_'+name+'_proj_weight_';k=x.shape[1]
        packed=symbolic.constant(np.ones((n,k),dtype=np.int8),name=prefix+'packed')
        weight=symbolic.op('Convert',packed,destination_type='f16',_shape=(n,k),_dtype='f16',_name=prefix+'decode')
        def mm(a,b,count):return symbolic.op('MatMul',a,b,transpose_a=False,transpose_b=True,_shape=(2,count),_dtype='f16')
        main=mm(x,weight,n)
        reduced=mm(x,symbolic.constant(np.ones((2,k),dtype=np.float16),name=prefix+'correction_down'),2)
        delta=mm(reduced,symbolic.constant(np.ones((n,2),dtype=np.float16),name=prefix+'correction_up'),n)
        return symbolic.op('Add',main,delta,auto_broadcast='numpy',_shape=(2,n),_dtype='f16')
    gate=linear(norm,'gate',6);up=linear(norm,'up',6)
    silu=symbolic.op('Swish',gate,_shape=(2,6),_dtype='f16')
    gated=symbolic.op('Multiply',*( (up,silu) if swapped else (silu,up)),auto_broadcast='numpy',_shape=(2,6),_dtype='f16')
    down=linear(gated,'down',4)
    output=symbolic.op('Add',residual,down,auto_broadcast='numpy',_shape=(2,4),_dtype='f16')
    graph=symbolic.Graph([residual,norm],[output]);xml=ET.fromstring(symbolic.serialize_ir(graph).xml)
    for node in xml.findall('./layers/layer'):
        if node.attrib['type']=='Convert':
            info=ET.SubElement(node,'rt_info');ET.SubElement(info,'attribute',name='decompression',version='0')
    return graph,ET.tostring(xml)


def experiment_helpers():
    path=Path(__file__).parent/'npunlock-probe/research/w8_gate_experiment.py'
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in ('front_graph','promotable')]
    env=dict(study=helpers());assert len(nodes)==2
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return SimpleNamespace(front_graph=env['front_graph'],promotable=env['promotable'])


class GateProbeTests(unittest.TestCase):
    def test_full_probe_exports_all_three_chains_and_gate_boundary(self):
        graph,xml=small_ffn();probe,_,_,names=helpers().ffn_probe_graph(graph,xml)
        self.assertEqual(len(names),15)
        self.assertEqual(names[-3:],('silu','gated','output'))
        self.assertEqual([v.shape for v in probe.outputs[:4]],[(2,6),(2,2),(2,6),(2,6)])
        self.assertEqual([v.shape for v in probe.outputs[8:12]],[(2,4),(2,2),(2,4),(2,4)])
        self.assertEqual(probe.inputs,graph.inputs)

    def test_runtime_annotations_and_swish_version_are_preserved(self):
        graph,xml=small_ffn();before=bytes(xml);_,modified,_,_=helpers().ffn_probe_graph(graph,xml)
        tree=ET.fromstring(modified)
        self.assertEqual(len(tree.findall("./layers/layer/rt_info/attribute[@name='decompression']")),3)
        self.assertEqual(tree.find("./layers/layer[@type='Swish']").attrib['version'],'opset4')
        self.assertEqual(xml,before)

    def test_down_tail_has_only_residual_and_gated_inputs(self):
        graph,xml=small_ffn();tail,modified,_,names=helpers().down_tail_graph(graph,xml)
        self.assertEqual(names,('corrected','output'))
        self.assertEqual([t.name for t in tail.inputs],['residual','gated'])
        self.assertFalse(any(n.op=='Swish' or n.name and ('mlp_gate_' in n.name or 'mlp_up_' in n.name) for n in tail.nodes))
        self.assertEqual(len(ET.fromstring(modified).findall("./layers/layer/rt_info/attribute[@name='decompression']")),1)

    def test_unexpected_gate_multiply_topology_rejects(self):
        graph,xml=small_ffn(swapped=True)
        with self.assertRaisesRegex(ValueError,'topology'):helpers().ffn_probe_graph(graph,xml)

    def test_tail_rejects_remaining_normalization_dependency(self):
        graph,xml=small_ffn()
        output=symbolic.op('Add',graph.outputs[0],graph.inputs[1],auto_broadcast='numpy',_shape=(2,4),_dtype='f16')
        invalid=symbolic.Graph(graph.inputs,[output])
        with self.assertRaisesRegex(ValueError,'normalization'):helpers().down_tail_graph(invalid,xml)

    def test_identical_paths_have_zero_signed_components(self):
        gate=torch.ones(2,6).half();up=torch.ones_like(gate);silu=F.silu(gate.float()).half()
        parts=helpers().gated_error_parts(silu*up,gate,up,silu,gate,up)
        self.assertTrue(all(value==0 for value in parts.values()))

    def test_projection_activation_and_multiply_differences_reconstruct(self):
        cpu_gate=torch.ones(2,6).half();cpu_up=torch.ones_like(cpu_gate)
        gate=cpu_gate+.125;up=cpu_up+.25;silu=F.silu(gate.float()).half()+.0625
        actual=(silu*up).half()+.03125
        parts=helpers().gated_error_parts(actual,gate,up,silu,cpu_gate,cpu_up)
        self.assertGreater(parts['projection'],0);self.assertGreater(parts['activation'],0);self.assertGreater(parts['multiply'],0)
        self.assertLess(parts['reconstruction'],1e-7)

    def test_front_uses_only_normalized_input_and_gate_up_weights(self):
        graph,xml=small_ffn();front,modified,_,names=experiment_helpers().front_graph(graph,xml)
        self.assertEqual(names,('gate','up'));self.assertEqual(front.inputs,(graph.inputs[1],))
        self.assertFalse(any(n.op=='Swish' or n.name and 'mlp_down_' in n.name for n in front.nodes))
        self.assertEqual(len(ET.fromstring(modified).findall("./layers/layer/rt_info/attribute[@name='decompression']")),2)

    def test_promotion_rejects_a_tradeoff_despite_one_more_gate_pass(self):
        decide=experiment_helpers().promotable
        baseline=dict(positions=10,quality_passed=8,next_token_equal=9,mean_relative_l2=.02,max_relative_l2=.1,max_kl=.01)
        improved=dict(baseline,quality_passed=9,mean_relative_l2=.019)
        self.assertTrue(decide(baseline,improved))
        self.assertFalse(decide(baseline,dict(improved,max_relative_l2=.11)))
        self.assertFalse(decide(baseline,dict(improved,next_token_equal=8)))
        with self.assertRaises(ValueError):decide(baseline,dict(improved,positions=11))


if __name__=='__main__':unittest.main()

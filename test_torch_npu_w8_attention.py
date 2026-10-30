"""Offline graph and causal CPU-replay contracts for S52."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import npunlock as symbolic
import torch
from test_torch_npu_w8_gate import helpers as gate_helpers


def helpers():
    path=Path(__file__).parent/'npunlock-probe/research/w8_attention_quality.py'
    names={'attention_nodes','attention_probe_graph','attention_tail_graph','cpu_attention','compare_masked'}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    env=dict(np=np,torch=torch,symbolic=symbolic,replace=replace,probe=gate_helpers(),
        device=SimpleNamespace(compare=lambda a,b:dict(finite_count=a.numel(),equal=torch.equal(a,b))))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return SimpleNamespace(**{name:env[name] for name in names})


def tiny_graph(*,axis=3,scale=.125,causal=True,transpose_pv=False):
    x=symbolic.input('x',shape=(1,3,8),dtype='f16');norm=symbolic.input('normalized',shape=x.shape,dtype='f16')
    def op(name,*args,shape,**kwargs):return symbolic.op(name,*args,_shape=shape,_dtype='f16',**kwargs)
    def reshape(v,shape):return op('Reshape',v,symbolic.constant(np.array(shape,np.int64)),shape=shape,special_zero=False)
    def linear(v,name,n):
        k=v.shape[-1];v=reshape(v,(3,k));prefix='self_attn_'+name+'_proj_weight_'
        packed=symbolic.constant(np.ones((n,k),np.int8),name=prefix+'packed')
        w=op('Convert',packed,shape=(n,k),destination_type='f16',_name=prefix+'decode')
        main=op('MatMul',v,w,shape=(3,n),transpose_a=False,transpose_b=True)
        down=op('MatMul',v,symbolic.constant(np.ones((2,k),np.float16),name=prefix+'correction_down'),shape=(3,2),transpose_a=False,transpose_b=True)
        up=op('MatMul',down,symbolic.constant(np.ones((n,2),np.float16),name=prefix+'correction_up'),shape=(3,n),transpose_a=False,transpose_b=True)
        return op('Add',main,up,shape=(3,n),auto_broadcast='numpy')
    def heads(v,h):return op('Transpose',reshape(v,(1,3,h,4)),symbolic.constant(np.array([0,2,1,3],np.int64)),shape=(1,h,3,4))
    q=heads(linear(norm,'q',8),2);k=heads(linear(norm,'k',4),1);v=heads(linear(norm,'v',4),1)
    def repeat(t):return op('Gather',t,symbolic.constant(np.array([0,0],np.int64)),symbolic.constant(np.int64(1)),shape=(1,2,3,4),batch_dims=0)
    k,v=repeat(k),repeat(v)
    scores=op('MatMul',q,k,shape=(1,2,3,3),transpose_a=False,transpose_b=True)
    scaled=op('Multiply',scores,symbolic.constant(np.float16(scale)),shape=scores.shape,auto_broadcast='numpy')
    mask=np.where(np.arange(3)[None,:]>np.arange(3)[:,None],-np.inf,0).astype(np.float16) if causal else np.zeros((3,3),np.float16)
    masked=op('Add',scaled,symbolic.constant(mask),shape=scores.shape,auto_broadcast='numpy')
    probs=op('Softmax',masked,shape=scores.shape,axis=axis)
    pv=op('MatMul',probs,v,shape=(1,2,3,4),transpose_a=False,transpose_b=transpose_pv)
    attended=reshape(op('Transpose',pv,symbolic.constant(np.array([0,2,1,3],np.int64)),shape=(1,3,2,4)),(1,3,8))
    result=op('Add',x,reshape(linear(attended,'o',8),x.shape),shape=x.shape,auto_broadcast='numpy')
    graph=symbolic.Graph([x,norm],[result]);xml=ET.fromstring(symbolic.serialize_ir(graph).xml)
    for node in xml.findall('./layers/layer'):
        if node.attrib['type']=='Convert':
            info=ET.SubElement(node,'rt_info');ET.SubElement(info,'attribute',name='decompression',version='0')
    return graph,ET.tostring(xml)


class AttentionProbeTests(unittest.TestCase):
    def test_exports_projections_scores_probabilities_and_residual(self):
        graph,xml=tiny_graph();probe,modified,_,names=helpers().attention_probe_graph(graph,xml)
        self.assertEqual(len(names),26);self.assertIn('raw',names);self.assertIn('probs',names)
        self.assertEqual(probe.inputs,graph.inputs)
        self.assertEqual(probe.outputs[names.index('masked')].shape,(1,2,3,3))
        self.assertEqual(len(ET.fromstring(modified).findall("./layers/layer/rt_info/attribute[@name='decompression']")),4)

    def test_tail_cuts_remove_qkv_and_normalization_dependencies(self):
        graph,xml=tiny_graph()
        for stage in ('raw','masked','probs','attended'):
            with self.subTest(stage=stage):
                tail,modified,_,names=helpers().attention_tail_graph(graph,xml,stage)
                self.assertEqual(names,('output',));self.assertEqual(len(tail.inputs),2 if stage=='attended' else 3)
                self.assertFalse(any(n.name and any('self_attn_'+part+'_' in n.name for part in ('q','k','v')) for n in tail.nodes))
                self.assertEqual(len(ET.fromstring(modified).findall("./layers/layer/rt_info/attribute[@name='decompression']")),1)

    def test_changed_scale_axis_mask_or_product_are_rejected(self):
        for kwargs in (dict(scale=.5),dict(axis=2),dict(causal=False),dict(transpose_pv=True)):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):helpers().attention_nodes(tiny_graph(**kwargs)[0])

    def test_tail_rejects_an_extra_input_path(self):
        graph,xml=tiny_graph();result=symbolic.op('Add',graph.outputs[0],graph.inputs[1],auto_broadcast='numpy',_shape=(1,3,8),_dtype='f16')
        bad=symbolic.Graph(graph.inputs,[result])
        with self.assertRaisesRegex(ValueError,'normalization'):helpers().attention_tail_graph(bad,xml,'probs')
        with self.assertRaisesRegex(ValueError,'Unknown'):helpers().attention_tail_graph(graph,xml,'logits')


class AttentionReplayTests(unittest.TestCase):
    def setUp(self):
        self.q=(torch.arange(24).view(1,2,3,4)/16-.5).half()
        self.k=(torch.arange(12).view(1,1,3,4)/8-.5).half();self.v=(self.k*.5).half()
        self.cos=torch.ones(1,1,3,4).half();self.sin=torch.zeros_like(self.cos)

    def test_half_boundaries_match_independent_scalar_causal_attention(self):
        output=helpers().cpu_attention(self.q,self.k,self.v,self.cos,self.sin)
        expected=torch.empty(1,2,3,4).half()
        for h in range(2):
            for row in range(3):
                raw=torch.stack([torch.dot(self.q[0,h,row].float(),self.k[0,0,j].float()).half() for j in range(row+1)])
                logits=(raw*.125).half().double();e=torch.exp(logits-logits.max());prob=(e/e.sum()).half()
                expected[0,h,row]=(prob.double()@self.v[0,0,:row+1].double()).half()
        torch.testing.assert_close(output['attended'],expected.transpose(1,2).reshape(3,8),rtol=0,atol=0)
        self.assertTrue(torch.isneginf(output['masked'][0,0,0,1:]).all())
        self.assertTrue((output['probs'][0,0,0,1:]==0).all())

    def test_rope_uses_two_half_products_and_grouped_head_repetition(self):
        cos=self.cos*.5;sin=self.cos*.25;out=helpers().cpu_attention(self.q,self.k,self.v,cos,sin)
        rotated=torch.cat((-self.q[...,2:],self.q[...,:2]),-1)
        torch.testing.assert_close(out['q_rot'],(self.q*cos+rotated*sin).half(),rtol=0,atol=0)
        torch.testing.assert_close(out['v_repeat'],self.v.repeat_interleave(2,1),rtol=0,atol=0)

    def test_replacing_probabilities_controls_the_later_pv_output(self):
        probabilities=torch.eye(3).repeat(1,2,1,1).half()
        out=helpers().cpu_attention(self.q,self.k,self.v,self.cos,self.sin,dict(probs=probabilities))
        expected=self.v.repeat_interleave(2,1).transpose(1,2).reshape(3,8)
        torch.testing.assert_close(out['attended'],expected,rtol=0,atol=0)

    def test_unknown_or_invalid_boundary_and_mask_structure_reject(self):
        for overrides in (dict(unknown=self.q),dict(q=self.q.float()),dict(probs=torch.zeros(1,2,3,2).half()),dict(masked=torch.zeros(1,2,3,3).half())):
            with self.subTest(keys=list(overrides)),self.assertRaises(ValueError):
                helpers().cpu_attention(self.q,self.k,self.v,self.cos,self.sin,overrides)

    def test_masked_comparison_keeps_inf_structure_out_of_finite_metrics(self):
        compare=helpers().compare_masked;x=torch.tensor([1.,-float('inf'),2.]).half()
        self.assertEqual(compare(x,x),dict(finite_count=2,equal=True,nonfinite_entries=1))
        for y in (torch.tensor([1.,0.,2.]).half(),torch.tensor([1.,float('inf'),2.]).half(),torch.tensor([1.,float('nan'),2.]).half()):
            with self.assertRaises(ValueError):compare(x,y)


class AttentionExperimentTests(unittest.TestCase):
    def helpers(self):
        root=Path(__file__).parent;path=root/'npunlock-probe/research/w8_attention_experiment.py'
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('front_graph','choose')]
        prior=root/'npunlock-probe/research/w8_gate_experiment.py'
        promote=next(n for n in ast.parse(prior.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='promotable')
        env={};exec(compile(ast.Module(body=[promote],type_ignores=[]),str(prior),'exec'),env)
        study=helpers();study.probe=gate_helpers();study.previous=SimpleNamespace(previous=SimpleNamespace(promotable=env['promotable']))
        env['study']=study;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
        return SimpleNamespace(front_graph=env['front_graph'],choose=env['choose'])

    def test_front_graph_uses_only_normalized_input_and_exports_masked_scores_and_values(self):
        graph,xml=tiny_graph();front,modified,_,names=self.helpers().front_graph(graph,xml)
        self.assertEqual(front.inputs,(graph.inputs[1],));self.assertEqual(names,('masked','value'))
        self.assertFalse(any(n.op=='Softmax' or n.name and 'self_attn_o_' in n.name for n in front.nodes))
        self.assertEqual(len(ET.fromstring(modified).findall("./layers/layer/rt_info/attribute[@name='decompression']")),3)

    def test_promotion_keeps_original_quality_rule(self):
        choose=self.helpers().choose;summary=dict(positions=290,quality_passed=243,mean_relative_l2=.02,max_relative_l2=.13,max_kl=.004,next_token_equal=286)
        baseline=dict(status='passed',healthy=True,summary=summary)
        candidate=dict(status='passed',healthy=True,summary=dict(summary,quality_passed=244,mean_relative_l2=.019))
        self.assertEqual(choose(baseline,candidate),'softmax-f32')
        candidate['summary']['max_kl']=.005;self.assertEqual(choose(baseline,candidate),'baseline')

    def test_known_precision_rejection_is_distinguished_from_execution_failure(self):
        choose=self.helpers().choose;baseline=dict(status='passed',healthy=True,summary=dict(positions=290))
        rejected=dict(status='failed',healthy=True,local_guard_failure=dict(relative_l2=.0005,max_scaled_error=.012))
        self.assertEqual(choose(baseline,rejected),'baseline')
        with self.assertRaises(ValueError):choose(baseline,dict(status='failed',healthy=True))
        with self.assertRaises(ValueError):choose(baseline,dict(rejected,healthy=False))


if __name__=='__main__':unittest.main()

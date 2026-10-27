"""Offline optional-model tests: requires safetensors, no checkpoint or NPU."""
import hashlib
from dataclasses import replace
import importlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch
import torch.nn.functional as F

if importlib.util.find_spec('safetensors') is None:
    raise unittest.SkipTest('optional safetensors dependency is not installed')
from torch_npu_smollm import FrozenSmolLayer, FrozenSmolStack
import torch_npu_smollm as smol
import torch_npu_graph_cache as cache
from npunlock.npu4000 import compile_fused_rmsnorm


class ProjectionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dict(model_type='llama',hidden_size=576,intermediate_size=1536,num_attention_heads=9,
                          num_key_value_heads=3,hidden_act='silu',attention_bias=False,mlp_bias=False,
                          rms_norm_eps=1e-5,rope_theta=100000,rope_scaling=None,pretraining_tp=1)
        shapes = {'self_attn.q_proj.weight':(576,576),'self_attn.k_proj.weight':(192,576),
                  'self_attn.v_proj.weight':(192,576),'self_attn.o_proj.weight':(576,576),
                  'mlp.gate_proj.weight':(1536,576),'mlp.up_proj.weight':(1536,576),
                  'mlp.down_proj.weight':(576,1536),'input_layernorm.weight':(576,),
                  'post_attention_layernorm.weight':(576,)}
        generator = torch.Generator().manual_seed(414)
        cls.weights = {name:(torch.randn(shape,generator=generator)*.015).half() if len(shape)==2 else torch.ones(shape).half()
                       for name,shape in shapes.items()}
        cls.x = (torch.randn(1,7,576,generator=generator)*.1).half()

    def layer(self, **options):
        return FrozenSmolLayer(self.config,self.weights,**options)

    def test_policy_guards_before_checkpoint_io(self):
        names = tuple(name for name,value in self.weights.items() if value.ndim==2)
        for bad in (None,[], 'mlp.down_proj.weight',('unknown',),('input_layernorm.weight',),
                    (False,),(['mlp.down_proj.weight'],),('mlp.down_proj.weight',)*2,names):
            with self.subTest(policy=bad):
                for constructor in (FrozenSmolLayer.from_checkpoint,FrozenSmolStack.from_checkpoint):
                    with self.assertRaises(ValueError):
                        constructor('missing-checkpoint',fp16_projections=bad)
        with self.assertRaises(ValueError):
            self.layer(mode='fp16',fp16_projections=('mlp.down_proj.weight',))

    def test_immutable_policy_and_prepared_weights(self):
        baseline = self.layer()
        prepared = dict(baseline._packed)
        mixed = self.layer(packed_weights=prepared,fp16_projections=('mlp.down_proj.weight',))
        snapshot = mixed._packed['mlp.down_proj.weight']
        prepared.clear()
        self.assertIs(mixed._packed['mlp.down_proj.weight'],snapshot)
        self.assertEqual(mixed.fp16_projections,frozenset(('mlp.down_proj.weight',)))
        with self.assertRaises(AttributeError):
            mixed.fp16_projections=()

    def test_protected_down_projection_uses_original_fp16(self):
        baseline = self.layer()
        mixed = self.layer(fp16_projections=('mlp.down_proj.weight',))
        with torch.no_grad():
            _,before = baseline.reference_cpu(self.x)
            _,after = mixed.reference_cpu(self.x)
            expected = F.linear(after['gated'].float(),self.weights['mlp.down_proj.weight'].float()).half()
        self.assertTrue(torch.equal(before['gated'],after['gated']))
        self.assertTrue(torch.equal(after['ffn_output'],expected))
        self.assertFalse(torch.equal(before['ffn_output'],expected))

    def test_mixed_ir_constants_and_cache_identity(self):
        protected = ('mlp.up_proj.weight','mlp.down_proj.weight')
        normal = self.layer()
        mixed = self.layer(fp16_projections=protected)
        reversed_policy = self.layer(fp16_projections=protected[::-1])
        kwargs = dict(segment='layer',fused_norm=True,kv_cache='decode')
        _,xml,data = mixed.graph((1,1,576),**kwargs)
        _,reverse_xml,reverse_data = reversed_policy.graph((1,1,576),**kwargs)
        _,normal_xml,normal_data = normal.graph((1,1,576),**kwargs)
        self.assertEqual((xml,data),(reverse_xml,reverse_data))
        self.assertNotEqual(hashlib.sha256(xml+data).digest(),hashlib.sha256(normal_xml+normal_data).digest())
        tree = ET.fromstring(xml)
        constants = [node for node in tree.findall('./layers/layer') if node.attrib['name'].endswith('_protected_fp16')]
        decodes = [node for node in tree.findall('./layers/layer') if node.find("./rt_info/attribute[@name='decompression']") is not None]
        self.assertEqual(len(constants),2)
        self.assertEqual(len(decodes),5)
        for node in constants:
            self.assertEqual(node.find('data').attrib['element_type'],'f16')

    def test_empty_policy_preserves_both_default_modes(self):
        for mode in ('fp16','w8a16'):
            original,explicit = self.layer(mode=mode),self.layer(mode=mode,fp16_projections=())
            kwargs = dict(segment='layer',fused_norm=True,kv_cache='prefill')
            self.assertEqual(original.graph((1,7,576),**kwargs)[1:],explicit.graph((1,7,576),**kwargs)[1:])
            with torch.no_grad():
                a,_ = original.reference_cpu(self.x)
                b,_ = explicit.reference_cpu(self.x)
            self.assertTrue(torch.equal(a,b))

    def identity(self, layer, shape=(1,24,576)):
        with patch.dict(os.environ, NPUNLOCK_GRAPH_CACHE='identity-test-only'):
            return cache.prefill_cache_graph(layer,shape,b'kernel',compile_fused_rmsnorm)

    def test_lowbit_prefill_binding_matches_real_graph(self):
        for options in ({'mode':'fp16'},{'mode':'w8a16'},
                        {'fp16_projections':('mlp.up_proj.weight','mlp.down_proj.weight')}):
            layer=self.layer(**options)
            for shape in ((1,24,576),(4,1,576),(8,64,576)):
                with self.subTest(options=options,shape=shape):
                    graph,digest=self.identity(layer,shape)
                    actual,_,_=layer.graph(shape,segment='layer',kv_cache='prefill',fused_norm=True)
                    self.assertEqual(cache._contract(graph),cache._contract(actual))
                    self.assertEqual(digest,self.identity(layer,shape)[1])

    def test_lowbit_prefill_codes_scales_and_policy_invalidate(self):
        layer=self.layer();name='self_attn.q_proj.weight';packed=layer._packed[name]
        digest=self.identity(layer)[1]
        changed=bytes([packed.data[0]^1])+packed.data[1:]
        with patch.dict(layer._packed,{name:replace(packed,data=changed)}):
            self.assertNotEqual(self.identity(layer)[1],digest)
        scale=torch.frombuffer(bytearray(packed.scales),dtype=torch.float16)
        scale[0]*=1.125
        with patch.dict(layer._packed,{name:replace(packed,scales=scale.numpy().tobytes())}):
            self.assertNotEqual(self.identity(layer)[1],digest)
        policy=('mlp.up_proj.weight','mlp.down_proj.weight')
        mixed=self.layer(packed_weights=layer._packed,fp16_projections=policy)
        reversed_policy=self.layer(packed_weights=layer._packed,fp16_projections=policy[::-1])
        self.assertNotEqual(self.identity(mixed)[1],digest)
        self.assertEqual(self.identity(mixed)[1],self.identity(reversed_policy)[1])

    def test_protected_packed_bytes_are_not_graph_constants(self):
        name='mlp.down_proj.weight';layer=self.layer(fp16_projections=(name,))
        digest=self.identity(layer)[1];packed=layer._packed[name]
        kwargs=dict(segment='layer',kv_cache='prefill',fused_norm=True)
        original=layer.graph((1,24,576),**kwargs)[1:]
        with patch.dict(layer._packed,{name:replace(packed,data=bytes([packed.data[0]^1])+packed.data[1:])}):
            self.assertEqual(self.identity(layer)[1],digest)
            self.assertEqual(layer.graph((1,24,576),**kwargs)[1:],original)
        data,shape=layer._weights[name]
        with patch.dict(layer._weights,{name:(bytes([data[0]^1])+data[1:],shape)}):
            self.assertNotEqual(self.identity(layer)[1],digest)

    def test_lowbit_runtime_gamma_and_research_hooks(self):
        layer=self.layer();digest=self.identity(layer)[1]
        for name in ('input_layernorm.weight','post_attention_layernorm.weight'):
            data,shape=layer._weights[name]
            with patch.dict(layer._weights,{name:(b'\0'*len(data),shape)}):
                self.assertEqual(self.identity(layer)[1],digest)
        with patch.object(smol,'weight_graph',wraps=smol.weight_graph):
            self.assertIsNone(self.identity(layer))
        with patch.object(smol.FrozenSmolLayer,'graph',wraps=smol.FrozenSmolLayer.graph):
            self.assertIsNone(self.identity(layer))
        with patch.dict(os.environ,NPUNLOCK_GRAPH_CACHE=''):
            self.assertIsNone(cache.prefill_cache_graph(layer,(1,24,576),b'kernel',compile_fused_rmsnorm))

    def test_lowbit_graph_source_invalidates_only_lowbit_recipe(self):
        layer=self.layer();fp16=self.layer(mode='fp16')
        lowbit_digest=self.identity(layer)[1];fp16_digest=self.identity(fp16)[1]
        lowbit=importlib.import_module('torch_npu_lowbit')
        with tempfile.TemporaryDirectory(prefix='lowbit-source-key-') as directory:
            source=Path(directory)/'lowbit.py'
            source.write_bytes(Path(lowbit.__file__).read_bytes()+b'\n# changed graph implementation\n')
            with patch.object(lowbit,'__file__',str(source)):
                self.assertNotEqual(self.identity(layer)[1],lowbit_digest)
                self.assertEqual(self.identity(fp16)[1],fp16_digest)


if __name__=='__main__':
    torch.set_num_threads(8)
    unittest.main()

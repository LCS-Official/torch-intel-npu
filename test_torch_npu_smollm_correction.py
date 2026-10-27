"""Optional model graph/reference contracts; no checkpoint or NPU required."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch


@unittest.skipUnless(importlib.util.find_spec('safetensors'), 'optional model dependency')
class SmolCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from torch_npu_smollm import FrozenSmolLayer
        from torch_npu_lowbit import quantize_weight
        from torch_npu_lowbit_correction import LowRankCorrection
        cls.Layer=FrozenSmolLayer
        cls.config=dict(model_type='llama',hidden_size=576,intermediate_size=1536,num_attention_heads=9,
            num_key_value_heads=3,hidden_act='silu',attention_bias=False,mlp_bias=False,rms_norm_eps=1e-5,
            rope_theta=100000,rope_scaling=None,pretraining_tp=1)
        shapes={'self_attn.q_proj.weight':(576,576),'self_attn.k_proj.weight':(192,576),
            'self_attn.v_proj.weight':(192,576),'self_attn.o_proj.weight':(576,576),
            'mlp.gate_proj.weight':(1536,576),'mlp.up_proj.weight':(1536,576),'mlp.down_proj.weight':(576,1536),
            'input_layernorm.weight':(576,),'post_attention_layernorm.weight':(576,)}
        cls.weights={name:torch.full(shape,.01 if len(shape)==2 else 1.,dtype=torch.float16) for name,shape in shapes.items()}
        cls.packed={name:quantize_weight(w,format='int8') for name,w in cls.weights.items() if w.ndim==2}
        cls.correction=LowRankCorrection((576,576),8,np.full((8,576),.001,dtype=np.float16).tobytes(),
            np.full((576,8),.001,dtype=np.float16).tobytes())

    def test_graph_keeps_int8_decompression_and_adds_only_selected_factors(self):
        import xml.etree.ElementTree as ET
        base=self.Layer(self.config,self.weights,packed_weights=self.packed)
        mapping={'self_attn.q_proj.weight':self.correction}
        layer=self.Layer(self.config,self.weights,packed_weights=self.packed,corrections=mapping)
        mapping.clear()
        _,x,b=base.graph((1,3,576));_,y,c=layer.graph((1,3,576))
        def count(xml,kind):return sum(n.attrib['type']==kind for n in ET.fromstring(xml).findall('./layers/layer'))
        self.assertEqual(count(y,'MatMul'),count(x,'MatMul')+2)
        self.assertEqual(count(y,'Add'),count(x,'Add')+1)
        self.assertEqual(y.count(b'name="decompression"'),x.count(b'name="decompression"'))
        self.assertGreater(len(c),len(b));self.assertNotEqual(x,y)
        self.assertEqual(len(layer._corrections),1)

    def test_reference_matches_independent_factor_projection(self):
        torch.manual_seed(443)
        base=self.Layer(self.config,self.weights,packed_weights=self.packed)
        layer=self.Layer(self.config,self.weights,packed_weights=self.packed,
            corrections={'self_attn.q_proj.weight':self.correction})
        value=torch.randn(1,3,576).half()
        # Q changes but K/V remain unchanged; compare the affected branch explicitly.
        _,old=base.reference_cpu(value);_,actual=layer.reference_cpu(value)
        torch.testing.assert_close(actual['keys'],old['keys'],rtol=0,atol=0)
        torch.testing.assert_close(actual['values'],old['values'],rtol=0,atol=0)
        self.assertTrue(torch.isfinite(actual['output']).all())
        # A zero residual must preserve the complete staged reference bitwise.
        from torch_npu_lowbit_correction import LowRankCorrection
        zero=LowRankCorrection((576,576),8,bytes(8*576*2),bytes(576*8*2))
        exact=self.Layer(self.config,self.weights,packed_weights=self.packed,corrections={'self_attn.q_proj.weight':zero})
        a,_=exact.reference_cpu(value);b,_=base.reference_cpu(value)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_invalid_corrections_rejected_during_cpu_preparation(self):
        from torch_npu_lowbit_correction import LowRankCorrection
        bad_shape=LowRankCorrection((2,2),1,bytes(4),bytes(4))
        for options in (dict(corrections=[]),dict(corrections={'unknown':self.correction}),
            dict(corrections={'self_attn.k_proj.weight':self.correction}),
            dict(corrections={'self_attn.q_proj.weight':bad_shape}),
            dict(corrections={'self_attn.q_proj.weight':self.correction},fp16_projections=('self_attn.q_proj.weight',))):
            with self.assertRaises(ValueError):self.Layer(self.config,self.weights,packed_weights=self.packed,**options)
        with self.assertRaises(ValueError):self.Layer(self.config,self.weights,corrections={'self_attn.q_proj.weight':self.correction})
        self.assertNotIn('torch_intel_npu',sys.modules)

    def test_correction_is_applied_at_selected_projection_before_residual(self):
        from torch_npu_lowbit_correction import LowRankCorrection
        factor=LowRankCorrection((576,1536),8,np.full((8,1536),.001,dtype=np.float16).tobytes(),self.correction.up)
        base=self.Layer(self.config,self.weights,packed_weights=self.packed)
        layer=self.Layer(self.config,self.weights,packed_weights=self.packed,corrections={'mlp.down_proj.weight':factor})
        value=torch.ones(1,3,576,dtype=torch.float16)
        _,old=base.reference_cpu(value);_,actual=layer.reference_cpu(value)
        a,b=factor.tensors()
        delta=torch.nn.functional.linear(torch.nn.functional.linear(old['gated'].float(),a.float()).half().float(),b.float()).half()
        torch.testing.assert_close(actual['ffn_output'],(old['ffn_output']+delta).half(),rtol=0,atol=0)
        self.assertFalse(torch.equal(actual['ffn_output'],old['ffn_output']))
        torch.testing.assert_close(actual['gated'],old['gated'],rtol=0,atol=0)

    def test_default_serialized_graph_preserves_published_baseline(self):
        path=Path(__file__).with_name('source-release')/'torch_npu_smollm.py'
        if not path.is_file():self.skipTest('local published source comparison')
        spec=importlib.util.spec_from_file_location('_correction_baseline_smollm',path)
        old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
        for mode in ('fp16','w8a16'):
            a=self.Layer(self.config,self.weights,mode=mode);b=old.FrozenSmolLayer(self.config,self.weights,mode=mode)
            for shape,kind in (((1,3,576),None),((1,1,576),'prefill'),((1,1,576),'decode')):
                options={} if kind is None else dict(segment='layer',kv_cache=kind,fused_norm=True)
                aa=a.graph(shape,**options);bb=b.graph(shape,**options)
                self.assertEqual(aa[1:],bb[1:])
            x=torch.zeros(1,3,576,dtype=torch.float16)
            torch.testing.assert_close(a.reference_cpu(x)[0],b.reference_cpu(x)[0],rtol=0,atol=0)

    def test_prefill_cache_identity_includes_residual_data_and_rank(self):
        from torch_npu_graph_cache import prefill_cache_graph
        from torch_npu_smollm import compile_fused_rmsnorm
        from torch_npu_lowbit_correction import LowRankCorrection
        a=self.correction
        changed=LowRankCorrection(a.shape,a.rank,np.full((8,576),.002,dtype=np.float16).tobytes(),a.up)
        def key(corrections):
            layer=self.Layer(self.config,self.weights,packed_weights=self.packed,corrections=corrections)
            result=prefill_cache_graph(layer,(1,1,576),b'test-kernel',compile_fused_rmsnorm)
            self.assertIsNotNone(result)
            return result[1]
        with patch.dict('os.environ',{'NPUNLOCK_GRAPH_CACHE':'metadata-only-test'}):
            keys=[key(c) for c in (None,{'self_attn.q_proj.weight':a},
                {'self_attn.q_proj.weight':changed},{'self_attn.q_proj.weight':a.truncated(4)},
                {'self_attn.o_proj.weight':a})]
            self.assertEqual(len(set(keys)),len(keys))
            self.assertEqual(keys[1],key({'self_attn.q_proj.weight':a}))
            first={'self_attn.q_proj.weight':a,'self_attn.o_proj.weight':changed}
            self.assertEqual(key(first),key(dict(reversed(list(first.items())))))


if __name__=='__main__':unittest.main()

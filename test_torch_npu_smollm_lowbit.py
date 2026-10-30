"""Frozen INT4/NF4 decoder entry points, preparation guards and references."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch


@unittest.skipUnless(importlib.util.find_spec('safetensors'),'optional model dependency')
class SmolLowbitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_torch_npu_smollm_policy import ProjectionPolicyTests
        ProjectionPolicyTests.setUpClass()
        cls.config=ProjectionPolicyTests.config
        cls.weights=ProjectionPolicyTests.weights
        cls.x=ProjectionPolicyTests.x[:,:3]
        from torch_npu_smollm import FrozenSmolLayer
        cls.Layer=FrozenSmolLayer

    def test_modes_prepare_group32_weights_without_mutating_checkpoint(self):
        saved={k:v.clone() for k,v in self.weights.items()}
        for mode,fmt in (('w4a16','int4'),('nf4a16','nf4')):
            layer=self.Layer(self.config,self.weights,mode=mode)
            self.assertEqual(layer.mode,mode);self.assertEqual(len(layer._packed),7)
            self.assertTrue(all(q.format==fmt and q.group_size==32 for q in layer._packed.values()))
        for name,value in saved.items():torch.testing.assert_close(self.weights[name],value,rtol=0,atol=0)

    def test_ir_keeps_packed_four_bit_constants_and_seven_decode_annotations(self):
        sizes=[]
        for mode,fmt in (('w4a16','i4'),('nf4a16','nf4')):
            layer=self.Layer(self.config,self.weights,mode=mode);graph,xml,blob=layer.graph(tuple(self.x.shape))
            tree=ET.fromstring(xml)
            self.assertTrue(tree.findall("./layers/layer/data[@element_type='"+fmt+"']"))
            self.assertEqual(len(tree.findall("./layers/layer/rt_info/attribute[@name='decompression']")),7)
            self.assertEqual(tuple(graph.outputs[0].shape),tuple(self.x.shape));sizes.append(len(blob))
        _,_,w8=self.Layer(self.config,self.weights).graph(tuple(self.x.shape))
        self.assertTrue(all(size<len(w8) for size in sizes))

    def test_cpu_values_use_the_actual_four_bit_decode_not_original_fp16_weights(self):
        for mode in ('w4a16','nf4a16'):
            layer=self.Layer(self.config,self.weights,mode=mode);_,trace=layer.reference_cpu(self.x)
            _,plain=layer.reference_cpu(self.x,quantized=False)
            q=layer._packed['self_attn.v_proj.weight'];weight=torch.from_numpy(q.dequantize().copy()).float()
            expected=torch.nn.functional.linear(trace['norm1'].float(),weight).half().view(1,3,3,64).transpose(1,2)
            torch.testing.assert_close(trace['values'],expected,rtol=0,atol=0)
            self.assertFalse(torch.equal(trace['values'],plain['values']))

    def test_nf4_single_token_padding_stays_inside_projections(self):
        layer=self.Layer(self.config,self.weights,mode='nf4a16')
        graph,_,_=layer.graph((1,1,576),segment='layer',kv_cache='decode',fused_norm=True)
        projections=[n for n in graph.nodes if n.op=='MatMul' and n.inputs[1].shape[-1] in (576,1536)]
        self.assertEqual(len(projections),7)
        self.assertTrue(all(n.inputs[0].shape[0]==32 for n in projections))
        self.assertEqual(tuple(graph.outputs[0].shape),(1,1,576))
        self.assertEqual(tuple(graph.outputs[1].shape),(1,3,64,64))

    def test_mismatched_formats_and_w8_only_options_are_rejected(self):
        prepared=self.Layer(self.config,self.weights,mode='w4a16')._packed
        with self.assertRaises(ValueError):self.Layer(self.config,self.weights,mode='nf4a16',packed_weights=prepared)
        for mode in ('w4a16','nf4a16'):
            with self.assertRaises(ValueError):self.Layer(self.config,self.weights,mode=mode,fp16_projections=('mlp.down_proj.weight',))
            with self.assertRaises(ValueError):self.Layer(self.config,self.weights,mode=mode,packed_weights=prepared,corrections={})

    def test_stack_propagates_modes_and_prepared_layer_maps(self):
        from torch_npu_smollm import FrozenSmolStack
        names={f'model.layers.0.{name}':tensor for name,tensor in self.weights.items()}
        archive=SimpleNamespace(keys=lambda:list(names),get_tensor=lambda name:names[name])
        with tempfile.TemporaryDirectory(prefix='smol-lowbit-test-',dir=Path(__file__).parent) as directory:
            folder=Path(directory).resolve();self.assertEqual(folder.parent,Path(__file__).resolve().parent)
            (folder/'config.json').write_text(json.dumps(dict(self.config,num_hidden_layers=1)),encoding='utf-8')
            with patch('torch_npu_smollm.safe_open') as opened:
                opened.return_value.__enter__.return_value=archive
                for mode,fmt in (('w4a16','int4'),('nf4a16','nf4')):
                    stack=FrozenSmolStack.from_checkpoint(folder,num_layers=1,mode=mode)
                    self.assertEqual(stack.layers[0].mode,mode);self.assertTrue(all(q.format==fmt for q in stack.layers[0]._packed.values()))
                    prepared={0:dict(stack.layers[0]._packed)}
                    other=FrozenSmolStack.from_checkpoint(folder,num_layers=1,mode=mode,packed_weights=prepared)
                    self.assertEqual(other.layers[0]._packed,prepared[0])


class CorpusBoundaryTests(unittest.TestCase):
    def run_prepare(self,eos_position):
        path=Path(__file__).parent/'npunlock-probe/research/smollm_lowbit_coverage.py'
        nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='prepare']
        ids=torch.full((1,64),17,dtype=torch.int64);ids[0,0]=0
        if eos_position is not None:ids[0,eos_position]=0
        common=SimpleNamespace(model=lambda:(SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=0)),None),
            chat=lambda *a:torch.tensor([[0,11]]),trajectory=lambda *a:ids,sha=lambda p:'source')
        with tempfile.TemporaryDirectory(prefix='smol-corpus-test-',dir=Path(__file__).parent) as directory:
            folder=Path(directory).resolve();self.assertEqual(folder.parent,Path(__file__).resolve().parent)
            env=dict(Path=Path,ast=ast,json=json,common=common,OUT=folder,__file__=str(path))
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env);env['prepare']()
            return json.loads((folder/'corpus.json').read_text(encoding='utf-8'))['cases']

    def test_first_generated_eos_keeps_its_prediction_and_ignores_later_positions(self):
        for position in (2,5):
            cases=self.run_prepare(position)
            self.assertTrue(all(row['offset']==1 and row['end']==position for row in cases))

    def test_prompt_eos_does_not_truncate_generated_quality_positions(self):
        self.assertTrue(all(row['end']==64 for row in self.run_prepare(None)))


class MatrixEvidenceTests(unittest.TestCase):
    def inspect(self,mode,wmode):
        from torch_npu_transformer import FrozenTransformer
        elf=SimpleNamespace(named={'dpu':0},sections=[],content=lambda section:bytes(352))
        tasks=[dict(section='dpu',offset=0,amode=0,wmode=wmode,elementwise=False)]
        with patch('torch_npu_transformer.Elf',return_value=elf),patch('torch_npu_transformer.dpu_modes',return_value=tasks):
            return FrozenTransformer._inspect_blob(SimpleNamespace(mode=mode),b'fixture')

    def test_four_bit_decompression_requires_fp16_dpu_evidence(self):
        for mode in ('w4a16','nf4a16'):
            self.assertEqual(self.inspect(mode,0)['int8_matrix_tasks'],0)
            with self.assertRaisesRegex(ValueError,'FP16 DPU'):self.inspect(mode,2)

    def test_existing_w8_matrix_guard_still_rejects_fp16_only_tasks(self):
        self.assertEqual(self.inspect('w8a16',2)['int8_matrix_tasks'],1)
        for mode in ('w8a16','mixed_int8'):
            with self.assertRaisesRegex(ValueError,'INT8 weight DPU'):self.inspect(mode,0)


if __name__=='__main__':unittest.main()

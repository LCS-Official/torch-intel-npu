"""CPU-only checks of research error decomposition and layer rescue routing."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


NAMES=('self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight',
       'self_attn.o_proj.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight')


def helpers():
    path=Path(__file__).parent/'npunlock-probe/research/w8_layerwise_quality.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    names={'relative','error_parts','offset_detail','validated_policy','set_policy','extra_parameter_bytes','candidate_key'}
    common=SimpleNamespace(ranking=lambda r:(r['positions']-r['quality_passed'],r['max_relative_l2'],r['mean_relative_l2']))
    env=dict(torch=torch,NAMES=NAMES,base=SimpleNamespace(mixed=SimpleNamespace(common=common)))
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return SimpleNamespace(**{name:env[name] for name in names})


class LayerwiseTests(unittest.TestCase):
    def test_local_and_inherited_errors_can_cancel(self):
        h=helpers();reference=torch.tensor([[2.,-2.]])
        local=torch.tensor([[1.,-1.]])
        parts=h.error_parts(reference,reference,reference+local,reference-local)
        self.assertEqual(parts['total'],0)
        self.assertAlmostEqual(parts['local'],.5)
        self.assertAlmostEqual(parts['inherited'],.5)
        self.assertAlmostEqual(parts['local_inherited_cosine'],-1.,places=6)
        self.assertEqual(parts['interaction'],0)
        self.assertEqual(parts['reconstruction_error'],0)

    def test_nonlinear_interaction_is_not_attributed_to_one_source(self):
        h=helpers();reference=torch.tensor([[1.,2.]])
        parts=h.error_parts(reference+3,reference,reference+1,reference+1)
        self.assertAlmostEqual(parts['interaction'],float(torch.tensor(2.).sqrt()/reference.norm()))
        self.assertEqual(parts['reconstruction_error'],0)

    def test_constant_offset_and_centered_errors_are_distinct(self):
        h=helpers();reference=torch.tensor([[1.,2.,3.],[-2.,1.,4.]])
        pure=h.offset_detail(reference+2,reference)
        self.assertGreater(pure['relative_l2'],0)
        self.assertEqual(pure['centered_relative_l2'],0)
        self.assertEqual(pure['offset_energy_fraction'],1)
        centered=h.offset_detail(reference+torch.tensor([-1.,0.,1.]),reference)
        self.assertEqual(centered['offset_energy_fraction'],0)
        self.assertGreater(centered['centered_relative_l2'],0)

    def test_policy_rejects_duplicates_unknown_names_and_boolean_indices(self):
        h=helpers();pair=(2,NAMES[0])
        self.assertEqual(h.validated_policy([pair]),frozenset([pair]))
        for bad in ([pair,pair],[(True,NAMES[0])],[(-1,NAMES[0])],[(30,NAMES[0])],[(0,'weight')]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):h.validated_policy(bad)

    def test_protection_changes_only_the_requested_layer_and_resets(self):
        h=helpers();modules=[(name,SimpleNamespace()) for _ in range(30) for name in NAMES]
        h.set_policy(modules,[(2,NAMES[1]),(29,NAMES[6])])
        self.assertEqual([i for i,(_,m) in enumerate(modules) if m.protected],[15,209])
        self.assertTrue(all(m.rank==32 for _,m in modules))
        h.set_policy(modules,(),rank=0)
        self.assertFalse(any(m.protected for _,m in modules))
        self.assertTrue(all(m.rank==0 for _,m in modules))

    def test_parameter_delta_excludes_replaced_w8_and_factors(self):
        h=helpers();original=SimpleNamespace(weight=torch.empty(8,16),out_features=8)
        module=SimpleNamespace(original=original,down=torch.empty(2,16),up=torch.empty(8,2))
        modules=[(name,module) for _ in range(30) for name in NAMES]
        self.assertEqual(h.extra_parameter_bytes(modules,[(2,NAMES[0])]),16)
        self.assertEqual(h.extra_parameter_bytes(modules,()),0)

    def test_ranking_keeps_raw_gate_before_memory_tiebreak(self):
        h=helpers()
        def row(label,passed,maximum,mean,cost):
            return dict(label=label,extra_parameter_bytes=cost,summary=dict(
                positions=10,quality_passed=passed,max_relative_l2=maximum,mean_relative_l2=mean))
        rows=[row('small',8,.03,.02,1),row('accurate',9,.1,.03,999),row('lower-max',9,.09,.04,1000)]
        self.assertEqual(min(rows,key=h.candidate_key)['label'],'lower-max')
        self.assertLess(h.candidate_key(row('a',9,.09,.04,1)),h.candidate_key(row('b',9,.09,.04,2)))

    def build_rescue(self,policy):
        path=Path(__file__).parent/'npunlock-probe/research/w8_layerwise_quality.py'
        node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='build_stack')
        packed={i:object() for i in range(30)}
        factors={i:{name:object() for name in NAMES} for i in range(30)}
        layers=[object() for _ in range(30)];original=layers.copy()
        fake=SimpleNamespace(FrozenSmolStack=SimpleNamespace(from_checkpoint=lambda *a,**kw:SimpleNamespace(layers=layers)),
            FrozenSmolLayer=SimpleNamespace(from_checkpoint=lambda *a,**kw:kw))
        env=dict(NAMES=NAMES,base=SimpleNamespace(mixed=SimpleNamespace(weights=lambda:packed,common=SimpleNamespace(MODEL='model')),
            corrections=lambda rank:factors))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
        with patch.dict(sys.modules,{'torch_npu_smollm':fake}):stack=env['build_stack']('selected',policy)
        self.assertTrue(all(stack.layers[i] is original[i] for i in range(30) if i!=24))
        return stack.layers[24],packed,factors

    def test_whole_block_rescue_uses_fp16_mode(self):
        layer,_,_=self.build_rescue([(24,name) for name in NAMES])
        self.assertEqual(layer,dict(layer_index=24,mode='fp16'))

    def test_projection_rescue_removes_only_its_correction(self):
        layer,packed,factors=self.build_rescue([(24,NAMES[0])])
        self.assertIs(layer['packed_weights'],packed[24])
        self.assertEqual(layer['fp16_projections'],(NAMES[0],))
        self.assertEqual(layer['corrections'],{name:q for name,q in factors[24].items() if name!=NAMES[0]})


if __name__=='__main__':unittest.main()

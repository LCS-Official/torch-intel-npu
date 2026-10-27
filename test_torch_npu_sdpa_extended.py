"""CPU checks for SDPA capacity contracts and GQA head layout/gradients."""
import ast
from math import prod,isfinite
from pathlib import Path
import unittest
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from torch_npu_geometry import softmax_row_tiles
from npunlock.npu4000 import MAX_DMA_ELEMENTS


def environment():
    source=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
    names=('_sdpa_spec','_sdpa_repeat_heads','_sdpa_sum_heads')
    nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
    env=dict(torch=torch,prod=prod,isfinite=isfinite,MAX_RANK=64,MAX_DMA_ELEMENTS=MAX_DMA_ELEMENTS,
             _validate_float_read=lambda x:None,_validate_read=lambda x:None,softmax_row_tiles=softmax_row_tiles,
             _materialize=lambda x:x.contiguous(),_broadcast=lambda x,shape:x.expand(shape).contiguous(),
             _reduce=lambda x,dim,**kwargs:x.sum(dim,dtype=kwargs['dtype']))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<sdpa-extended>','exec'),env)
    return env


class ExtendedSDPATests(unittest.TestCase):
    def test_rank_and_broadcast_geometry_matches_torch(self):
        fn=environment()['_sdpa_spec']
        cases=(((3,4),(5,4),(5,2)),((2,1,3,4),(5,4),(1,3,5,2)),
               ((2,1,3,4,2),(1,4,1,5,2),(2,1,1,5,3)),
               ((1,)*62+(2,3),(3,3),(3,4)))
        for shapes in cases:
            values=[torch.randn(shape) for shape in shapes]
            scores,_=fn(*values)
            with sdpa_kernel(SDPBackend.MATH):actual=torch.nn.functional.scaled_dot_product_attention(*values)
            self.assertEqual(actual.shape,scores[:-1]+(shapes[-1][-1],))

    def test_score_capacity_and_kernel_header_guards(self):
        fn=environment()['_sdpa_spec'];tensor=lambda shape:torch.empty(shape,device='meta')
        self.assertEqual(fn(tensor((1,1,1025,1)),tensor((1,1,1024,1)),tensor((1,1,1024,1)))[0],(1,1,1025,1024))
        for shapes,options in (
            (((1,MAX_DMA_ELEMENTS//16384+1,1),(1,16384,1),(1,16384,1)),{}),
            (((1,1,1),(1,65537,1),(1,65537,1)),{}),
            (((1,1048577,1),(1,1,1),(1,1,1)),dict(is_causal=True)),
            (((1,1,1),(1,1,1),(1,1,MAX_DMA_ELEMENTS+1)),{}),
            (((1,)*63+(2,3),(3,3),(3,2)),{})):
            with self.assertRaises(ValueError):fn(*(tensor(s) for s in shapes),**options)

    def test_empty_and_zero_feature_geometry_matches_torch(self):
        fn=environment()['_sdpa_spec']
        cases=(((0,2,3,4),(1,2,5,4),(1,2,5,6)),((2,0,4),(1,5,4),(2,5,6)),
               ((2,3,4),(1,0,4),(2,0,6)),((2,3,0),(1,5,0),(2,5,6)),
               ((2,3,4),(1,5,4),(2,5,0)))
        for shapes in cases:
            values=[torch.randn(s,requires_grad=True) for s in shapes]
            shape,scale=fn(*values)
            self.assertTrue(isfinite(scale))
            with sdpa_kernel(SDPBackend.MATH):actual=torch.nn.functional.scaled_dot_product_attention(*values)
            self.assertEqual(tuple(actual.shape),shape[:-1]+(shapes[-1][-1],))
            actual.sum().backward()
            self.assertTrue(all(t.grad is not None for t in values))
        for qh,kh in ((0,2),(2,0),(0,0)):
            values=[torch.empty(s) for s in ((1,qh,3,4),(1,kh,5,4),(1,kh,5,6))]
            shape,_=fn(*values,enable_gqa=True)
            self.assertEqual(shape,(1,qh,3,5))

    def test_group_layout_and_inverse_sum_match_repeat_interleave(self):
        env=environment();repeat=env['_sdpa_repeat_heads'];merge=env['_sdpa_sum_heads']
        for shape in ((2,3,5),(2,2,3,5),(1,)*61+(2,3,5)):
            x=torch.arange(prod(shape),dtype=torch.float32).view(shape)
            actual=repeat(x,6);expected=x.repeat_interleave(3,dim=-3)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            g=torch.randn_like(expected)
            leaf=x.clone().requires_grad_()
            leaf.view(-1,*leaf.shape[-3:]).repeat_interleave(3,dim=-3).view_as(expected).backward(g)
            torch.testing.assert_close(merge(g,2),leaf.grad,rtol=0,atol=0)
        x=torch.arange(120.).view(2,3,4,5)[...,::2]
        torch.testing.assert_close(repeat(x,6),x.repeat_interleave(2,dim=-3),rtol=0,atol=0)

    def test_gqa_math_and_shared_kv_gradients_match_native_torch(self):
        env=environment()
        for shapes in (((2,4,3,2),(1,2,5,2),(2,2,5,3)),((4,3,2),(2,5,2),(2,5,3))):
            values=[torch.randn(shape,requires_grad=True) for shape in shapes]
            env['_sdpa_spec'](*values,enable_gqa=True)
            q,k,v=values
            with sdpa_kernel(SDPBackend.MATH):
                expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,enable_gqa=True,scale=-.7)
                actual=torch.nn.functional.scaled_dot_product_attention(q,env['_sdpa_repeat_heads'](k,4),
                    env['_sdpa_repeat_heads'](v,4),scale=-.7)
            torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
            seed=torch.randn_like(actual)
            for a,b in zip(torch.autograd.grad(actual,values,seed),torch.autograd.grad(expected,values,seed)):
                torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)

    def test_invalid_gqa_and_masks_fail_at_spec(self):
        fn=environment()['_sdpa_spec'];tensor=lambda shape:torch.empty(shape,device='meta')
        for shapes in (((3,4,2),(2,5,2),(2,5,3)),((4,4,2),(2,5,2),(1,5,3)),((4,2),(5,2),(5,3))):
            with self.assertRaises(ValueError):fn(*(tensor(s) for s in shapes),enable_gqa=True)
        q,k,v=[tensor(s) for s in ((4,3,2),(2,5,2),(2,5,3))]
        for mask,options in ((tensor((2,3)),{}),(tensor((3,5)),dict(is_causal=True))):
            with self.assertRaises((ValueError,RuntimeError)):fn(q,k,v,mask,enable_gqa=True,**options)


if __name__=='__main__':unittest.main()

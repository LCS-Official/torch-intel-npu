"""Offline contracts for bounded online attention and nested metadata safety."""
import ast
from contextlib import nullcontext
from itertools import product
from math import prod
import operator
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from test_torch_npu_sdpa_extended import environment as sdpa_environment


def functions(path,names,env):
    tree=ast.parse(Path(__file__).with_name(path).read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<flash-nested-contract>','exec'),env)
    return env


def flash_environment():
    npu=SimpleNamespace(_sdpa_spec=sdpa_environment()['_sdpa_spec'],_materialize=lambda x:x.contiguous(),
        _cast=lambda x,dtype:x.to(dtype),_binary_op=lambda x,y,kind:x+y if kind=='add' else x/y,
        _where=lambda mask,x,y:torch.where(mask,torch.as_tensor(x),torch.as_tensor(y)),
        _compare=lambda x,y,kind:x==y,
        _stream_concat=lambda values,shape:torch.cat([v.reshape(-1) for v in values]).reshape(shape))
    def frame(q,k,v,state,mask,qs,ks,causal,scale,first,go=None):
        a=q*abs(scale)**.5*(-1 if scale<0 else 1);b=k*abs(scale)**.5;s=a@b.T
        if mask is not None:s=s.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else s+mask
        if causal:s=s.masked_fill(torch.arange(ks,ks+len(k))[None,:]>torch.arange(qs,qs+len(q))[:,None],float('-inf'))
        if go is not None:
            den=state[:,1:2].masked_fill(state[:,1:2]==0,1);p=(s-state[:,:1]).exp()/den
            p=p.masked_fill(state[:,1:2]==0,0)
            ds=p*(go@v.T-(go*(state[:,2:]/den)).sum(-1,keepdim=True))
            return (ds@b)*(abs(scale)**.5*(-1 if scale<0 else 1)),(ds.T@a)*abs(scale)**.5,p.T@go
        oldm=torch.full_like(state[:,:1],float('-inf')) if first else state[:,:1]
        m=torch.maximum(oldm,s.amax(-1,keepdim=True));safe=m.masked_fill(m==float('-inf'),0)
        alpha=(oldm-safe).exp();p=(s-safe).exp()
        l=alpha*state[:,1:2]+p.sum(-1,keepdim=True);o=alpha*state[:,2:]+p@v
        return torch.cat((m,l,o),-1)
    env=dict(torch=torch,npu=npu,product=product,prod=prod,frame=frame)
    return functions('torch_npu_flash_attention.py',('coordinates','operand_coords','tile','mask_tile','packet_words','block_shape','validate','forward','backward','selected'),env)


class FlashNestedContracts(unittest.TestCase):
    def test_streaming_spec_removes_score_storage_and_row_width_allocation(self):
        fn=sdpa_environment()['_sdpa_spec']
        q=torch.empty(4097,4,device='meta');k=torch.empty(65537,4,device='meta');v=torch.empty(65537,5,device='meta')
        with self.assertRaises(ValueError):fn(q,k,v)
        shape,scale=fn(q,k,v,streaming=True)
        self.assertEqual(shape,(4097,65537));self.assertEqual(scale,.5)

    def test_chunked_broadcast_and_gqa_gradients_match_independent_cpu(self):
        env=flash_environment();torch.manual_seed(423)
        for shapes,gqa in ((((2,1,3,7),(3,5,7),(1,3,5,5)),False),(((2,4,3,7),(1,2,5,7),(2,5,5)),True)):
            values=[torch.randn(s).requires_grad_() for s in shapes]
            with sdpa_kernel(SDPBackend.MATH):expected=torch.nn.functional.scaled_dot_product_attention(*values,enable_gqa=gqa,scale=-.7)
            with torch.no_grad():a,states,layout=env['forward'](*values,None,False,-.7,gqa,2,3)
            torch.testing.assert_close(a,expected,rtol=5e-5,atol=3e-6)
            g=torch.randn_like(a);expected.backward(g)
            actual=env['backward'](*values,None,states,layout,g,False,-.7,gqa,2,3,(True,True,True))
            for actual,original in zip(actual,values):torch.testing.assert_close(actual,original.grad,rtol=5e-5,atol=4e-6)

    def test_masked_and_causal_tail_tiles(self):
        env=flash_environment();torch.manual_seed(424)
        values=[torch.randn(s) for s in ((3,7),(5,7),(5,5))]
        for mask,causal in ((torch.tensor([[False]*5,[True,False,True,False,True],[True]*5]),False),(None,True)):
            a,_,_=env['forward'](*values,mask,causal,.4,False,2,3)
            with sdpa_kernel(SDPBackend.MATH):b=torch.nn.functional.scaled_dot_product_attention(*values,attn_mask=mask,is_causal=causal,scale=.4)
            torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)

    def test_empty_scores_keep_exact_zero_gradients(self):
        env=flash_environment();values=[torch.empty(s) for s in ((2,0,3),(1,4,3),(2,4,5))]
        a,states,layout=env['forward'](*values,None,False,None,False,2,3)
        self.assertEqual(a.shape,(2,0,5));self.assertEqual(layout,[])
        actual=env['backward'](*values,None,states,layout,a,False,None,False,2,3,(True,True,False))
        torch.testing.assert_close(actual[1],torch.zeros_like(values[1]));self.assertIsNone(actual[2])

    def test_packet_bounds_and_backend_selection(self):
        env=flash_environment();q=torch.empty(2,7);k=torch.empty(3,7);v=torch.empty(3,5)
        for p,qblock,kblock in ((.1,2,3),(0,0,3),(0,2,129),(0,True,3)):
            with self.assertRaises(ValueError):env['validate'](q,k,v,None,p,False,None,False,qblock,kblock)
        with self.assertRaises(ValueError):env['validate'](torch.empty(32,128),torch.empty(128,128),torch.empty(128,128),None,0,False,None,False,32,128)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):self.assertTrue(env['selected']())
        with sdpa_kernel(SDPBackend.MATH):self.assertFalse(env['selected']())

    def test_automatic_blocks_cover_backward_and_float_mask_budget(self):
        env=flash_environment()
        for d,dv in ((16,16),(128,128),(128,1),(1,128)):
            q,k,v=torch.empty(128,d),torch.empty(256,d),torch.empty(256,dv)
            for mask in (None,torch.empty(128,256,dtype=torch.bool),torch.empty(128,256,dtype=torch.float16)):
                qb,kb=env['block_shape'](q,k,v,mask,None,None)
                self.assertLessEqual(env['packet_words'](qb,kb,d,dv,mask),32768)
                self.assertEqual((qb,kb),(32,64) if (d,dv)==(128,128) else (32,128))
                self.assertEqual(env['block_shape'](q,k,v,mask,2,3),(2,3))
        with self.assertRaises(ValueError):env['block_shape'](q,k,v,None,True,None)
        q,k,v=torch.empty(128,128),torch.empty(256,128),torch.empty(256,128)
        with self.assertRaises(ValueError):env['block_shape'](q,k,v,None,None,128)
        self.assertEqual(env['block_shape'](q,k,v,None,16,None),(16,64))
        # Empty and tail dimensions use their actual payload, without allocation.
        self.assertEqual(env['block_shape'](q[:0],k[:0],v[:0],None,None,None),(32,128))
        self.assertEqual(env['block_shape'](q[:3],k[:5],v[:5],None,32,128),(32,128))

    def test_native_nested_view_validates_storage_bounds_and_metadata(self):
        npu=SimpleNamespace(_validate_read=lambda x:None,_inference_only=lambda xs:None,MAX_RANK=64)
        env=functions('torch_npu_nested.py',('from_buffer',),dict(torch=torch,npu=npu,_cpu=torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)))
        buf=torch.arange(24.);sizes=torch.tensor([[2,8],[1,8]]);strides=torch.tensor([[8,1],[8,1]]);offsets=torch.tensor([0,16])
        x=env['from_buffer'](buf,sizes,strides,offsets)
        self.assertTrue(x.is_nested);self.assertEqual(x.untyped_storage()._cdata,buf.untyped_storage()._cdata)
        with self.assertRaises(RuntimeError):env['from_buffer'](buf,sizes,strides,torch.tensor([0,20]))
        with self.assertRaises(ValueError):env['from_buffer'](buf,sizes,strides,torch.tensor([0,-1]))

    def test_nested_training_rejects_before_no_grad_and_payload_access(self):
        def guard(values):
            if torch.is_grad_enabled() and any(t.requires_grad for t in values):raise ValueError('inference only')
        npu=SimpleNamespace(_inference_only=guard,_lock=nullcontext())
        env=functions('torch_npu_nested.py',('from_mask','padded'),dict(torch=torch,npu=npu,prod=prod,operator=operator))
        x=torch.randn(2,3,8,requires_grad=True)
        with self.assertRaisesRegex(ValueError,'inference only'):env['from_mask'](x,torch.ones(2,3,dtype=torch.bool))
        with self.assertRaisesRegex(ValueError,'inference only'):env['padded'](x,0.)


if __name__=='__main__':unittest.main()

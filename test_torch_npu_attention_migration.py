"""CPU contracts for sequence helpers and empty-cache concatenation."""
import ast
from contextlib import nullcontext
from math import prod
import operator
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch


def environment():
    path=Path(__file__).with_name('torch_npu_sequence.py')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    names=('arange_spec','cumsum','isin','integer_pad')
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    integers=(torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool)
    def logical(x,y=None,*,kind):return ~x if kind=='not' else x|y
    def check(out,shape,dtype,inputs):
        if out is not None and (tuple(out.shape)!=shape or out.dtype!=dtype):raise ValueError('out mismatch')
    npu=SimpleNamespace(_lock=nullcontext(),_validate_read=lambda x:None,_check_out=check,
        _STORAGE_DTYPES=integers+(torch.float16,torch.float32,torch.bfloat16),
        _storage_limit=lambda dtype:(1<<30)//torch.empty((),dtype=dtype).element_size(),
        _materialize=lambda x:x.contiguous(),_cast=lambda x,dtype:x.to(dtype),_clone=lambda x:x.clone(),
        _copy=lambda out,x:out.copy_(x),_binary_op=lambda x,y,**kw:x+y,_cat=lambda xs,dim:torch.cat(xs,dim),
        _scalar=lambda x,dtype:torch.tensor(x,dtype=dtype),_storage_slice=lambda x,start,count:x.view(-1)[start:start+count].clone(),
        _boolean_reduce=lambda x,dim=None,**kw:x.any(dim) if dim is not None else x.any(),
        _compare=lambda x,y,**kw:x.eq(y),_logical=logical,
        _concat=lambda xs,shape,dtype:torch.cat(xs).view(shape).to(dtype))
    env=dict(torch=torch,npu=npu,prod=prod,operator=operator,INTEGERS=integers)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<sequence>','exec'),env)
    return env


class SequenceContracts(unittest.TestCase):
    def test_partial_unary_checks_original_output_alias_before_materializing(self):
        tree=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_execute')
        events=[]
        def check(out,shape,dtype,inputs):
            events.append('check')
            if out is not None and any(out.untyped_storage()._cdata==x.untyped_storage()._cdata for x in inputs):
                raise ValueError('Output alias')
        def materialize(x):events.append('materialize');return x.contiguous()
        env=dict(torch=torch,_lock=nullcontext(),synchronize=lambda:None,_validate_float_read=lambda x:None,
                 _check_out=check,_materialize=materialize,MAX_ELEMENTS=65536,
                 bridge=SimpleNamespace(_DTYPES={torch.float32:('f32',None)}),
                 _run=lambda key,inputs,shape,dtype,out:-inputs[0] if out is None else torch.neg(inputs[0],out=out))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<partial-unary>','exec'),env)
        x=torch.arange(24.).view(2,3,4)[...,1::2]
        torch.testing.assert_close(env['_execute']('neg',x),-x,rtol=0,atol=0)
        self.assertEqual(events,['check','materialize']);events.clear()
        with self.assertRaises(ValueError):env['_execute']('neg',x,out=x)
        self.assertEqual(events,['check'])

    def test_arange_metadata_matches_torch_before_allocation(self):
        fn=environment()['arange_spec']
        for args in ((0,0,1,None),(0,5,1,None),(7,-3,-2,torch.int64),(.25,2.,.25,torch.float32)):
            actual=fn(*args,1<<30);expected=torch.arange(*args[:3],dtype=args[3])
            self.assertEqual(actual.shape,expected.shape);self.assertEqual(actual.dtype,expected.dtype)
        for args in ((0,1<<28,1,torch.int64),(0,4,1,torch.float64),(0,4,0,torch.int64)):
            with self.assertRaises((ValueError,RuntimeError)):fn(*args,1<<30)

    def test_integer_cumsum_axes_layout_and_overflow_match_cpu(self):
        fn=environment()['cumsum']
        for dtype in (torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool):
            x=torch.arange(60).reshape(3,4,5).to(dtype).transpose(0,2)
            for axis in range(-3,3):
                for target in (None,torch.int16):
                    torch.testing.assert_close(fn(x,axis,dtype=target),torch.cumsum(x,axis,dtype=target),rtol=0,atol=0)
        for x in (torch.tensor(1<<62),torch.tensor([1<<62,1<<62,-1,7]),torch.empty((2,0),dtype=torch.int64)):
            torch.testing.assert_close(fn(x,-1),torch.cumsum(x,-1),rtol=0,atol=0)
        with self.assertRaises(ValueError):fn(torch.ones(3),0)

    def test_isin_exact_integers_nan_empty_sets_and_scalar_overloads(self):
        fn=environment()['isin']
        cases=((torch.tensor([1<<62,(1<<62)+1,-3,8]),torch.tensor([(1<<62)+1,-3])),
               (torch.tensor([0.,float('nan'),float('inf'),-.0]),torch.tensor([-.0,float('nan')])),
               (torch.empty((2,0),dtype=torch.int64),torch.tensor([1,2])),
               (torch.tensor([1,2]),torch.empty(0,dtype=torch.int64)),
               (torch.tensor([1,2,3]),2),(3,torch.tensor([1,3,3])))
        for a,b in cases:
            for invert in (False,True):
                torch.testing.assert_close(fn(a,b,invert=invert),torch.isin(a,b,invert=invert),rtol=0,atol=0)
        with self.assertRaises(ValueError):fn(torch.empty(4097,dtype=torch.int64),torch.empty(4097,dtype=torch.int64))

    def test_integer_pad_preserves_bits_crop_and_empty(self):
        fn=environment()['integer_pad'];pad=torch.nn.functional.pad
        for dtype in (torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool):
            x=torch.arange(12).reshape(3,4).to(dtype)
            for padding in ((0,1),(1,2,-1,1),(-4,0),(0,0)):
                torch.testing.assert_close(fn(x,padding,1),pad(x,padding,value=1),rtol=0,atol=0)
        x=torch.tensor([1<<62,(1<<62)+1]);torch.testing.assert_close(fn(x,(1,2),-100),pad(x,(1,2),value=-100),rtol=0,atol=0)
        x=torch.empty((0,),dtype=torch.int64);torch.testing.assert_close(fn(x,(1,2),-100),pad(x,(1,2),value=-100),rtol=0,atol=0)

    def test_cat_spec_skips_rank_one_empty_and_preserves_input_list(self):
        tree=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_cat_spec')
        env=dict(torch=torch,prod=prod,operator=operator,MAX_RANK=64,_validate_read=lambda x:None,
                 _storage_limit=lambda dtype:(1<<30)//torch.empty((),dtype=dtype).element_size())
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<cat-spec>','exec'),env)
        for shapes,axis in ((((0,),(2,3,4)), -2),(((2,3,4),(0,)),-1),(((0,),(0,)),0)):
            values=[torch.empty(s) for s in shapes];tensors,dim,shape=env['_cat_spec'](values,axis)
            self.assertEqual(shape,tuple(torch.cat(values,axis).shape));self.assertEqual(len(tensors),len(values))
        # An ignored empty input still participates in dtype promotion and its byte budget.
        env['_storage_limit']=lambda dtype:1 if dtype==torch.float32 else 2
        with self.assertRaises(ValueError):env['_cat_spec']([torch.empty(0),torch.empty(2,dtype=torch.float16)],0)


if __name__=='__main__':unittest.main()

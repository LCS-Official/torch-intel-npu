"""Bounded streamed accumulation and high-rank MatMul metadata, CPU only."""
import ast
from contextlib import nullcontext
from math import prod
from pathlib import Path
import unittest
import weakref
import torch


def functions(*names):
    source = ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
    body = [n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
    return compile(ast.Module(body=body,type_ignores=[]),'<matmul-workspace>','exec')


def stream_environment():
    maximum = []
    def concat(values,shape,dtype,out=None):
        count = sum(v.numel() for v in values)
        if count > 512: raise AssertionError('Unbounded partial stack')
        assert count == prod(shape)
        result = torch.cat([v.view(-1) for v in values]).view(shape)
        return result if out is None else out.copy_(result)
    def run(key,inputs,shape,dtype):
        a,b = (v.view(-1) for v in inputs)
        assert key[0] == 'stream-sum' and a.numel() == b.numel() <= 64
        maximum.append(a.numel())
        count = a.numel()//2
        total,value = a[:count],b[:count]
        next_value = total+value
        error = torch.where(total.abs()>=value.abs(),(total-next_value)+value,(value-next_value)+total)
        correction = torch.where(torch.isfinite(next_value),a[count:]+error,0.)
        return torch.cat((next_value,correction)).view(shape)
    def check_out(out,shape,dtype,inputs):
        if out is not None and (tuple(out.shape)!=tuple(shape) or out.dtype!=dtype): raise ValueError('out mismatch')
    env = dict(torch=torch,prod=prod,MAX_ELEMENTS=64,_validate=lambda x:None,_check_out=check_out,
               _scalar=lambda x,dtype:torch.tensor(x,dtype=dtype),
               _broadcast=lambda x,shape:x.expand(shape).clone(),_concat=concat,
               _slice=lambda x,start,count:x.view(-1)[start:start+count],_run=run,
               _execute=lambda kind,x,y:x+y)
    exec(functions('_stream_sum'),env)
    return env['_stream_sum'],maximum


class MatmulWorkspaceTests(unittest.TestCase):
    def test_matmul_empty_and_zero_inner_dimensions(self):
        def check_out(out,shape,dtype,inputs):
            if out is not None and (tuple(out.shape)!=shape or out.dtype!=dtype):raise ValueError('out mismatch')
        from npunlock.npu4000 import MAX_DMA_ELEMENTS
        env=dict(torch=torch,prod=prod,MAX_DMA_ELEMENTS=MAX_DMA_ELEMENTS,MAX_RANK=64,_lock=nullcontext(),
                 _validate_float_read=lambda x:None,synchronize=lambda:None,_check_out=check_out,
                 _fill=lambda x,v:x.fill_(v))
        exec(functions('_matmul_promote','_matmul_nd_spec','_matmul_nd'),env)
        for left,right in (((2,3,0),(1,0,4)),((2,1,3,0),(1,4,0,5)),((0,),(0,)),
                           ((2,3,0),(0,)),((0,2,3),(1,3,4))):
            a,b=torch.empty(left),torch.empty(right)
            expected=torch.matmul(a,b);out=torch.full(expected.shape,float('nan'))
            self.assertIs(env['_matmul_nd'](a,b,out=out),out)
            torch.testing.assert_close(out,expected,rtol=0,atol=0)

    def test_bmm_zero_dimensions_preserve_out_and_zero_k_result(self):
        def check_out(out,shape,dtype,inputs):
            if out is not None and (tuple(out.shape)!=shape or out.dtype!=dtype):raise ValueError('out mismatch')
        from npunlock.npu4000 import MAX_DMA_ELEMENTS
        env=dict(torch=torch,prod=prod,MAX_DMA_ELEMENTS=MAX_DMA_ELEMENTS,_lock=nullcontext(),
                 _validate_float_read=lambda x:None,synchronize=lambda:None,_check_out=check_out,
                 _fill=lambda x,v:x.fill_(v))
        exec(functions('_bmm_spec','_bmm'),env)
        for left,right in (((2,3,0),(2,0,4)),((2,0,3),(2,3,4)),((2,3,4),(2,4,0)),((0,2,3),(0,3,4))):
            a,b=torch.empty(left),torch.empty(right)
            expected=torch.bmm(a,b);out=torch.full(expected.shape,float('nan'))
            self.assertIs(env['_bmm'](a,b,out=out),out)
            torch.testing.assert_close(out,expected,rtol=0,atol=0)

    def test_streamed_partials_do_not_retain_the_full_stack(self):
        stream,maximum = stream_environment()
        references=[];live=[];expected=torch.zeros(17,19,dtype=torch.float64)
        generator=torch.Generator().manual_seed(727)
        def values():
            for _ in range(257):
                value=torch.randn(17,19,generator=generator)*.01
                expected.add_(value.double())
                references.append(weakref.ref(value))
                live.append(sum(ref() is not None for ref in references))
                yield value
        out=torch.empty(17,19)
        result=stream(values(),(17,19),out=out)
        self.assertIs(result,out)
        self.assertLessEqual(max(live),2)
        self.assertEqual(max(maximum),64)
        self.assertGreater(257*17*19,512)
        torch.testing.assert_close(result.double(),expected,rtol=2e-6,atol=1e-7)

    def test_streamed_layout_preserves_correction_and_nonfinite_categories(self):
        stream,_=stream_environment()
        values=[torch.full((3,19),v) for v in (1e8,1.,-1e8)]
        self.assertTrue(torch.equal(stream(iter(values),(3,19)),torch.ones(3,19)))
        values=[torch.tensor([float('inf'),float('nan'),-0.]).view(1,3),
                torch.tensor([-float('inf'),1.,-0.]).view(1,3)]
        result=stream(iter(values),(1,3))
        self.assertTrue(torch.isnan(result[0,:2]).all())
        self.assertEqual(result[0,2].item(),0.)

    def test_streamed_invalid_geometry_is_rejected(self):
        stream,_=stream_environment()
        for values,shape in (([],(2,3)),([torch.ones(2,3,dtype=torch.float64)],(2,3)),
                             ([torch.ones(3,2)],(2,3)),([torch.empty(0,3)],(0,3))):
            with self.assertRaises(ValueError):stream(iter(values),shape)

    def test_matmul_geometry_high_rank_vectors_and_storage_guards(self):
        from npunlock.npu4000 import MAX_DMA_ELEMENTS
        env=dict(torch=torch,prod=prod,MAX_RANK=64,MAX_DMA_ELEMENTS=MAX_DMA_ELEMENTS,
                 _validate_float_read=lambda x:None)
        exec(functions('_matmul_promote','_matmul_nd_spec'),env)
        for left,right in (((2,1,3,2,3),(1,4,1,3,5)),
                           ((1,)*61+(2,2,3),(1,)*62+(3,4)),
                           ((1,)*61+(2,2,3),(3,))):
            a,b=torch.zeros(left),torch.zeros(right)
            _,_,shape=env['_matmul_nd_spec'](a,b)
            self.assertEqual(shape,tuple(torch.matmul(a,b).shape))
        for left,right in (((MAX_DMA_ELEMENTS//16384+1,1),(1,16384)),((2,3),(4,2)),((),(3,)),((1,)*65,(1,1))):
            with self.assertRaises(ValueError):env['_matmul_nd_spec'](torch.empty(left,device='meta'),torch.empty(right,device='meta'))


if __name__=='__main__':unittest.main()

"""Offline seed/state and RNG packet ABI contracts; no device import."""
import unittest
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
import torch
from torch_npu_rng import RNGState, integer_range, permutation_size
from npunlock.npu4000 import rng_packet_words, discrete_packet_words


class RNGTests(unittest.TestCase):
    def test_permutation_rejects_ties_and_never_returns_a_partial_out(self):
        source=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
        node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_randperm')
        proxy=SimpleNamespace(int64=torch.int64,strided=torch.strided,no_grad=torch.no_grad,
                              device=lambda x:SimpleNamespace(type='intel_npu',index=0),
                              empty=lambda s,**kw:torch.empty(s,dtype=kw['dtype']),
                              zeros=lambda s,**kw:torch.zeros(s,dtype=kw['dtype']))
        draws=[];mode=['once']
        def fill(x,*args,**kwargs):
            draws.append(1)
            return x.copy_(torch.tensor([7,7,1] if mode[0]=='always' or len(draws)==1 else [7,-3,1]))
        env=dict(torch=proxy,_lock=nullcontext(),synchronize=lambda:None,_check_generator=lambda g:None,
                 permutation_size=permutation_size,_check_out=lambda *args:None,_random_integer_fill=fill,
                 _materialize=lambda x:x.contiguous(),_compare=lambda x,y,**kw:x.eq(y),
                 _boolean_reduce=lambda x,**kw:x.any(),_cast=lambda x,dtype:x.to(dtype),_copy=lambda x,y:x.copy_(y))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<randperm>','exec'),env)
        fake=ModuleType('torch_npu_sort');fake.sort=torch.sort
        with patch.dict('sys.modules',{'torch_npu_sort':fake}):
            self.assertEqual(env['_randperm'](3).tolist(),[1,2,0])
            self.assertEqual(len(draws),2)
            mode[0]='always';draws.clear();out=torch.full((3,),-9,dtype=torch.int64)
            with self.assertRaises(RuntimeError):env['_randperm'](3,out=out)
            self.assertEqual(len(draws),8)
            self.assertEqual(out.tolist(),[-9]*3)

    def test_permutation_dtype_and_sort_bounds(self):
        for dtype,limit in ((torch.float16,2049),(torch.bfloat16,257),(torch.int8,128),
                            (torch.uint8,256),(torch.int16,32768),(torch.float32,16777216),
                            (torch.int32,16777216),(torch.int64,16777216)):
            for n in (0,1,limit):self.assertEqual(permutation_size(n,dtype),n)
            for n in (-1,limit+1):
                with self.assertRaises(ValueError):permutation_size(n,dtype)
        for dtype in (torch.bool,torch.float64,torch.complex64):
            with self.assertRaises(ValueError):permutation_size(3,dtype)
        with self.assertRaises(TypeError):permutation_size(3.5,torch.int64)

    def test_seed_and_reservation(self):
        rng=RNGState(-1)
        self.assertEqual(rng.reserve(3),((1<<64)-1,0))
        self.assertEqual(rng.reserve(7),((1<<64)-1,3))
        rng.manual_seed(47)
        self.assertEqual(rng.reserve(1),(47,0))
        for bad in (-(1<<63)-1,1<<64):
            with self.assertRaises(RuntimeError):rng.manual_seed(bad)
        self.assertEqual(rng.reserve(1),(47,1))

    def test_state_roundtrip(self):
        rng=RNGState((1<<63)+19)
        rng.reserve((1<<32)-2)
        state=rng.get_state()
        rng.reserve(9)
        rng.set_state(state)
        self.assertEqual(rng.reserve(9),((1<<63)+19,(1<<32)-2))
        state.zero_()
        self.assertNotEqual(rng.get_state().sum().item(),0)
        good=rng.get_state()
        for bad in (state,torch.ones(23,dtype=torch.uint8),torch.ones(24),torch.ones(2,12,dtype=torch.uint8),
                    torch.ones(48,dtype=torch.uint8)[::2],None):
            with self.assertRaises(ValueError):rng.set_state(bad)
            self.assertTrue(torch.equal(rng.get_state(),good))

    def test_exhaustion(self):
        rng=RNGState(1)
        rng.reserve((1<<64)-2)
        self.assertEqual(rng.reserve(1)[1],(1<<64)-2)
        for count in (1,0,-1):
            with self.assertRaises(RuntimeError):rng.reserve(count)

    def test_packet_bound(self):
        for count,words in ((1,16),(7,16),(8,16),(9,16),(17,18),(8192,8192),(65535,65536),(65536,65536)):
            self.assertEqual(rng_packet_words(count),words)
        for count in (0,-1,65537,True,1.5):
            with self.assertRaises(ValueError):rng_packet_words(count)

    def test_integer_bounds_preserve_all_bits(self):
        self.assertEqual(integer_range(), (0,1<<63,0))
        self.assertEqual(integer_range(-(1<<63)), (1<<63,0,0))
        self.assertEqual(integer_range((1<<63)-2,(1<<63)-1), ((1<<63)-2,1,0))
        self.assertEqual(integer_range(-7,12), ((1<<64)-7,19,(1<<64)%19))
        # About half the raw draws need rejection for this interval.
        self.assertEqual(integer_range(-1,(1<<63)-1), ((1<<64)-1,1<<63,0))
        self.assertEqual(integer_range(-2,(1<<63)-1)[2], (1<<63)-1)
        for low,high in ((1,1),(2,1),(-(1<<63)-1,0),(0,(1<<63)+1)):
            with self.assertRaises(ValueError):integer_range(low,high)
        with self.assertRaises(TypeError):integer_range(.5,9)

    def test_discrete_packet_has_room_for_header_output_scratch(self):
        for rows,columns,samples in ((1,0,1),(8192,0,1),(32767,0,1),(1,49152,1),
                                     (3,20000,1),(1,3,10000),(7,19,5)):
            words=discrete_packet_words(rows,columns,samples)
            self.assertEqual(words%2,0)
            self.assertGreaterEqual(words,16+rows*columns)
            self.assertGreaterEqual(words,rows*columns+2*rows*samples+2)
            self.assertLessEqual(words,65536)
        for args in ((32768,0,1),(4,20000,1),(1,65521,1),(1,1,32767),(1,0,2),
                     (0,3,1),(1,-1,1),(1,3,0),(True,3,1),(1,3,1.5)):
            with self.assertRaises(ValueError):discrete_packet_words(*args)

    def test_int64_oracle_known_philox_vector(self):
        from torch_npu_discrete_demo import integer_reference
        # Random123 Philox4x32-10 zero key/counter: c0=6627e8d5,c1=e169c58d.
        raw=(0xe169c58d<<32)|0x6627e8d5
        self.assertEqual(integer_reference(0,0,1,-(1<<63),1<<63).item(),raw-(1<<63))
        self.assertEqual(integer_reference(0,0,1,0,17).item(),raw%17)


if __name__=='__main__':unittest.main()

"""Offline streaming ownership, alignment and convolution workspace plans."""
import ast
from math import prod
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref
import numpy as np
import torch
from torch_npu_geometry import conv_geometry,conv_window_indices


def environment():
    tree=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
    names=('_stream_concat','_stream_write','_stream_finish','_conv_streaming','_conv_window_tiles');calls=[]
    def copy(dst,src):
        assert dst.storage_offset()%1024==src.storage_offset()%1024==0
        calls.append((dst.storage_offset(),src.numel()));return dst.copy_(src)
    env=dict(torch=SimpleNamespace(float32=torch.float32,
        empty=lambda shape,**kw:torch.empty(shape,dtype=kw['dtype'])),prod=prod,MAX_ELEMENTS=65536,
        MAX_DMA_ELEMENTS=268435456,MAX_RANK=64,_CONV_WORKSPACE_ELEMENTS=262144,
        _validate=lambda *a,**kw:None,_materialize=lambda x:x.contiguous(),_range_copy=copy,
        _slice=lambda x,start,count:x.reshape(-1).narrow(0,start,count).clone(),
        _concat=lambda xs,shape,dtype:torch.cat([x.reshape(-1) for x in xs]).to(dtype).view(shape),
        conv_geometry=conv_geometry,conv_window_indices=conv_window_indices,
        _gather_window=lambda x,ids,row_width:torch.from_numpy(np.concatenate(
            (x.numpy().reshape(-1),np.zeros(1,dtype=x.numpy().dtype)))[ids]))
    exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],
                            type_ignores=[]),'<conv-capacity>','exec'),env)
    return env,calls


class ConvCapacityTests(unittest.TestCase):
    def test_unaligned_parts_flush_aligned_ranges_and_preserve_bits(self):
        env,calls=environment();sizes=(17,1007,1,65535,33,2049,3)
        raw=torch.arange(sum(sizes),dtype=torch.int32)*7919;source=raw.view(torch.float32)
        pieces=list(source.split(sizes));result=env['_stream_concat'](iter(pieces),(sum(sizes),))
        self.assertTrue(torch.equal(result.view(torch.int32),raw))
        self.assertTrue(all(offset%1024==0 for offset,count in calls))
        self.assertEqual(sum(count for offset,count in calls),sum(sizes))

    def test_old_parts_are_not_retained_by_assembler(self):
        env,_=environment();refs=[]
        def pieces():
            for i in range(8):
                self.assertLessEqual(sum(ref() is not None for ref in refs),1)
                value=torch.full((1031,),float(i));refs.append(weakref.ref(value));yield value
        result=env['_stream_concat'](pieces(),(8,1031))
        self.assertTrue(all(ref() is None for ref in refs))
        torch.testing.assert_close(result,torch.arange(8.).view(8,1).expand(8,1031),rtol=0,atol=0)

    def test_assembly_rejects_wrong_count_dtype_and_storage_shape(self):
        env,_=environment();fn=env['_stream_concat']
        self.assertEqual(fn(iter(()),(0,)).shape,torch.Size([0]))
        self.assertEqual(fn(iter((torch.empty(0),torch.ones(3))),(3,)).tolist(),[1.,1.,1.])
        for parts,shape in (([torch.ones(3)],(4,)),([torch.ones(4)],(3,)),
                            ([torch.ones(3,dtype=torch.float16)],(3,)),([],(268435457,)),([],(True,))):
            with self.assertRaises(ValueError):fn(iter(parts),shape)

    def test_streaming_accounts_for_windows_fold_and_output_workspace(self):
        env,_=environment();fn=env['_conv_streaming'];params=((3,3),(1,1),(1,1),(1,1))
        self.assertFalse(fn((1,4,8,8),params,(64,36)))
        self.assertTrue(fn((1,4,96,97),params,(96*97,36)))
        self.assertFalse(fn((1,1,1,4097),((1,1),(1,1),(0,0),(1,1)),(4097,1)))
        self.assertTrue(fn((1,4,96,97),params,(1,36)))

    def test_row_tiles_bound_input_and_output_and_keep_short_tail(self):
        env,_=environment();fn=env['_conv_window_tiles'];params=((1,1),(1,1),(0,0),(1,1))
        host=torch.arange(2051,dtype=torch.float32)
        tiles=list(fn(host,(1,1,1,2051),params,result_width=1024))
        self.assertEqual(sum(count for start,count,window in tiles),2051)
        self.assertEqual(tiles[-1][1],3)
        for start,count,window in tiles:
            self.assertLessEqual(count*1024,65536);self.assertEqual(window.numel(),count)
            torch.testing.assert_close(window.flatten(),host[start:start+count],rtol=0,atol=0)
        backward=list(fn(torch.arange(131*4,dtype=torch.float32),(1,1024,1,131),params,
                         output_channels=4,result_width=1024))
        self.assertEqual([count for start,count,window in backward],[64,64,3])
        for start,count,window in backward:self.assertEqual(window.shape,(count,4))


if __name__=='__main__':unittest.main()

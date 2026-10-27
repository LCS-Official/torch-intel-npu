"""Dense pool source coverage, ordered gradient pruning and byte storage bounds."""
import ast
from math import prod
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch_npu_geometry import max_pool2d_geometry,max_pool_windows,pool_dense_tiles,pool_gradient_ranges


class PoolDenseTests(unittest.TestCase):
    def test_forward_tiles_cover_every_sample_across_rows_and_planes(self):
        for spatial,kernel,stride,padding,dilation,ceil in (
            ((13,17),3,1,1,1,False),((7,11),(2,3),(2,1),(1,1),(2,2),True),
            ((1,66049),(1,3),(1,1),(0,1),1,False),((257,257),3,1,1,1,False)):
            params=max_pool2d_geometry(spatial,kernel,stride,padding,dilation,ceil);per=prod(params[-1])
            for alignment in (1024,2048):
                cursor=0
                for start,count,base,span in pool_dense_tiles(spatial,params,3,alignment):
                    self.assertEqual(start,cursor);cursor+=count;self.assertLessEqual(3*count+1,65536)
                    locations=max_pool_windows(spatial,params,start,count)
                    absolute=locations+np.arange(start,start+count)[:,None]//per*prod(spatial)
                    valid=absolute[locations>=0]
                    self.assertEqual(base%alignment,0);self.assertLessEqual(span,65519)
                    self.assertGreaterEqual(valid.min(),base);self.assertLess(valid.max(),base+span)
                self.assertEqual(cursor,3*per)

    def test_wide_window_keeps_gather_fallback(self):
        params=max_pool2d_geometry((1,200001),(1,3),(1,100000),0,(1,100000),False)
        self.assertEqual(list(pool_dense_tiles((1,200001),params,1)),[(0,1,None,None)])

    def test_backward_ranges_include_all_reachable_windows_and_keep_order(self):
        for spatial in ((1,17),(7,11),(13,9)):
            for k in (1,2,3):
                for stride in (1,2,3):
                    for dilation in (1,2,3):
                        try:params=max_pool2d_geometry(spatial,(1,k) if spatial[0]==1 else k,stride,
                            (0,k//2) if spatial[0]==1 else k//2,dilation,True)
                        except ValueError:continue
                        windows=max_pool_windows(spatial,params,0,prod(params[-1]))
                        for base in range(0,prod(spatial),5):
                            span=min(5,prod(spatial)-base)
                            expected=np.where(np.any((windows>=base)&(windows<base+span),axis=1))[0].tolist()
                            actual=[i for start,count in pool_gradient_ranges(spatial,params,base,span) for i in range(start,start+count)]
                            self.assertEqual(actual,sorted(set(actual)));self.assertTrue(set(expected)<=set(actual))
                            self.assertTrue(all(0<=i<len(windows) for i in actual))

    def test_dense_scan_work_scales_with_local_windows(self):
        spatial=(4097,4097);params=max_pool2d_geometry(spatial,3,1,1,1,False);per=prod(params[-1]);count=0
        for base in range(0,prod(spatial),61440):
            count+=sum(n for start,n in pool_gradient_ranges(spatial,params,base,min(61440,prod(spatial)-base)))
        old=((prod(spatial)+64511)//64512)*per
        self.assertLess(count,old//100)

    def test_storage_policy_and_dma_limits_use_bytes(self):
        import npunlock.npu4000 as runtime
        tree=ast.parse(Path('torch_intel_npu.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_storage_limit')
        env=dict(MAX_ALLOCATION_BYTES=2**30);exec(compile(ast.Module(body=[node],type_ignores=[]),'<storage>','exec'),env)
        for dtype in (torch.float16,torch.float32,torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool,torch.bfloat16):
            self.assertEqual(env['_storage_limit'](dtype)*dtype.itemsize,2**30)
        with patch.object(runtime,'compile_precision_graph',side_effect=lambda graph,**kw:graph):
            for dtype,width in (('f16',2),('f32',4),('u8',1)):
                limit=2**30//width;graph=runtime.compile_slice(limit,limit-3,3,dtype=dtype)
                self.assertEqual(graph.inputs[0].shape,(limit,width))
                with self.assertRaises(ValueError):runtime.compile_slice(limit+1,0,3,dtype=dtype)

    def test_streaming_all_storage_dtypes_keeps_bits_and_byte_alignment(self):
        tree=ast.parse(Path('torch_intel_npu.py').read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_stream_concat','_stream_write','_stream_finish')]
        def copied(dst,src):
            self.assertEqual(dst.storage_offset()*dst.element_size()%4096,0)
            self.assertEqual(src.storage_offset()*src.element_size()%4096,0)
            return dst.copy_(src)
        env=dict(torch=SimpleNamespace(float32=torch.float32,empty=lambda shape,**kw:torch.empty(shape,dtype=kw['dtype'])),
            prod=prod,MAX_RANK=64,MAX_DMA_ELEMENTS=268435456,_validate=lambda *a,**k:None,
            _materialize=lambda x:x.contiguous(),_range_copy=copied,_slice=lambda x,s,n:x.view(-1)[s:s+n].clone(),
            _concat=lambda xs,shape,dtype:torch.cat([x.view(-1) for x in xs]).view(shape))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<streaming>','exec'),env)
        sizes=(17,2031,1,65535,33,2049,3);count=sum(sizes)
        for dt in (torch.float16,torch.float32,torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool,torch.bfloat16):
            raw=(torch.arange(count*dt.itemsize,dtype=torch.int32)*7919).to(torch.uint8)
            source=raw.view(dt) if dt!=torch.bool else torch.arange(count)%3==0
            actual=env['_stream_concat'](iter(source.split(sizes)),(count,),dtype=dt)
            self.assertTrue(torch.equal(actual.view(torch.uint8),source.view(torch.uint8)),dt)

    def test_explicit_backward_keeps_arbitrary_index_route(self):
        tree=ast.parse(Path('torch_intel_npu.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_max_pool2d_backward')
        x=torch.empty(1,1,257,257,device='meta');g=torch.empty_like(x);ids=torch.empty(x.shape,device='meta',dtype=torch.int64)
        params=max_pool2d_geometry((257,257),3,1,1,1,False);calls=[]
        env=dict(torch=torch,prod=prod,_lock=__import__('contextlib').nullcontext(),_max_pool2d_spec=lambda *a:(params,x.shape),
            _validate_read=lambda *a:None,_check_out=lambda *a:None,_selection_backward_planes=lambda *a:calls.append('generic') or g)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<explicit-backward>','exec'),env)
        trusted=SimpleNamespace(backward=lambda *a:calls.append('geometry') or g)
        with patch.dict(sys.modules,{'torch_npu_pool_dense':trusted}):
            env['_max_pool2d_backward'](g,x,3,1,1,1,False,ids)
            env['_max_pool2d_backward'](g,x,3,1,1,1,False,ids,known_windows=True)
        self.assertEqual(calls,['generic','geometry'])


if __name__=='__main__':unittest.main()

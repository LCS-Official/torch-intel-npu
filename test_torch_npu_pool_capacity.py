"""Offline plane geometry, raw gradient routing, and extrema index relocation."""
import ast
from contextlib import nullcontext
from math import prod
from pathlib import Path
from types import SimpleNamespace
import operator
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch_npu_geometry import (max_pool2d_geometry,max_pool_windows,selection_plane_tiles,reduction_geometry)


def environment():
    tree=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
    names=('_max_pool2d_spec','_selection_backward_planes','_extreme_reduce')
    def row(x,kind,offset=0,indices=None):
        assert offset+x.shape[-1]<=1048576
        value,idx=(torch.max if kind=='max' else torch.min)(x,dim=1)
        return value,(indices.gather(1,idx[:,None]).view(-1) if indices is not None else idx+offset)
    env=dict(torch=torch,operator=operator,prod=prod,_lock=nullcontext(),synchronize=lambda:None,
        _validate_read=lambda *a,**kw:None,_validate_float_read=lambda *a,**kw:None,
        _check_out=lambda *a:None,_storage_limit=lambda dt:268435456,MAX_TENSOR_ELEMENTS=16777216,
        MAX_INT64_ELEMENTS=134217728,max_pool2d_geometry=max_pool2d_geometry,reduction_geometry=reduction_geometry,
        bridge=SimpleNamespace(_DTYPES={torch.float32:('f32',None),torch.float16:('f16',None)}),
        _materialize=lambda x:x.contiguous(),_cast=lambda x,dt:x.to(dt),_slice=lambda x,s,n:x.reshape(-1)[s:s+n].clone(),
        _concat=lambda xs,shape,dt:torch.cat([x.view(-1) for x in xs]).to(dt).view(shape),
        _cat=lambda xs,dim:torch.cat(xs,dim),_row_extreme=row,_binary_op=lambda x,y,kind:x+y,
        _stream_concat=lambda xs,shape,dtype=torch.float32:torch.cat(list(xs)).to(dtype).view(shape))
    exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],
                            type_ignores=[]),'<pool-capacity>','exec'),env)
    return env


class PoolCapacityTests(unittest.TestCase):
    def test_large_spatial_geometry_does_not_allocate_output_axis_arrays(self):
        with patch('torch_npu_geometry.np.arange',wraps=np.arange) as array:
            params=max_pool2d_geometry((1,134217728),(1,1),(),0,1,False)
            self.assertEqual(params[-1],(1,134217728))
            self.assertTrue(all(call.args[0]<=0 for call in array.call_args_list))
        with self.assertRaises(ValueError):max_pool2d_geometry((1,134217729),1,(),0,1,False)

    def test_boundary_validation_matches_full_axis_reference(self):
        for n in range(1,13):
            for k in range(1,6):
                for stride in range(1,4):
                    for dilation in range(1,4):
                        for p in range(k//2+1):
                            for ceil in (False,True):
                                count=(n+2*p-dilation*(k-1)-1+(stride-1 if ceil else 0))//stride+1
                                if ceil and (count-1)*stride>=n+p:count-=1
                                starts=np.arange(max(0,count))*stride-p
                                first=starts+np.maximum(0,(-starts+dilation-1)//dilation)*dilation
                                valid=count>0 and np.all(first<np.minimum(starts+dilation*(k-1)+1,n))
                                try:params=max_pool2d_geometry((1,n),(1,k),(1,stride),(0,p),(1,dilation),ceil)
                                except ValueError:self.assertFalse(valid)
                                else:self.assertTrue(valid);self.assertEqual(params[-1],(1,count))

    def test_row_tiles_cross_planes_without_changing_relative_indices(self):
        params=max_pool2d_geometry((5,7),3,2,1,1,True);rows=prod(params[-1])
        expected=np.concatenate([max_pool_windows((5,7),params,0,rows)]*4)
        actual=max_pool_windows((5,7),params,rows-2,2*rows+3)
        np.testing.assert_array_equal(actual,expected[rows-2:3*rows+1])

    def test_group_plan_exceeds_global_metadata_without_exceeding_packets(self):
        plan=selection_plane_tiles(8193,512,2048)
        self.assertEqual(plan[0],(0,31));self.assertEqual(sum(n for s,n in plan),8193)
        for start,count in plan:
            self.assertLessEqual(count*2048,65535);self.assertLessEqual(count*512,21842)
        self.assertEqual(selection_plane_tiles(0,0+1,16777217),())
        for args in ((1,1,16777217),(1,16777217,1),(-1,1,1),(1,0,1)):
            with self.assertRaises(ValueError):selection_plane_tiles(*args)

    def test_pool_capacity_keeps_dtype_bounds_and_rejects_zero_channels(self):
        fn=environment()['_max_pool2d_spec']
        for shape in ((1,1,1,524289),(1,4,513,513),(8193,1,32,64),(0,2,5,7)):
            x=torch.empty(shape,device='meta');params,out=fn(x,3,1,1,1,False)
            self.assertEqual(out,shape)
        with self.assertRaises(ValueError):fn(torch.empty(1,0,3,4,device='meta'),2,2,0,1,False)
        with self.assertRaises(ValueError):fn(torch.empty(134217729,1,1,1,device='meta'),1,1,0,1,False)

    def test_grouped_assignment_preserves_negative_zero_and_duplicate_sums(self):
        env=environment();calls=[]
        def selected(g,ids,shape,per,width,accumulate=True):
            calls.append(shape);result=torch.zeros(shape,dtype=g.dtype).view(-1,width)
            if not g.numel():return result.view(shape)
            for row,(values,labels) in enumerate(zip(g.view(-1,per),ids.view(-1,per))):
                for value,index in zip(values,labels):
                    if not 0<=index<width:raise IndexError('bounds')
                    result[row,index]=result[row,index]+value if accumulate else value
            return result.view(shape)
        env['_selection_backward']=selected;fn=env['_selection_backward_planes']
        for dt in (torch.float32,torch.float16):
            g=torch.tensor([-.0,.125,-.25,.5]*16,dtype=dt).expand(17,64).clone()
            for accumulate in (False,True):
                ids=(torch.arange(64)%8 if accumulate else torch.arange(64)).expand(17,64).clone()
                expected=torch.zeros(17,4096,dtype=dt)
                (expected.scatter_add_ if accumulate else expected.scatter_)(1,ids,g)
                actual=fn(g,ids,(17,4096),64,4096,accumulate=accumulate)
                self.assertTrue(torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)))
            self.assertEqual(fn(g[:0],ids[:0],(17,4096),0,4096).count_nonzero(),0)
        self.assertTrue(any(shape==(15,4096) for shape in calls))

    def test_extrema_relocates_indices_beyond_carrier_offset_bound(self):
        env=environment();fn=env['_extreme_reduce'];x=torch.zeros(1048593);x[3]=-9;x[1048513]=7;x[-1]=7
        for kind in ('max','min'):
            expected=(torch.max if kind=='max' else torch.min)(x,0)
            actual,index=fn(x,0,kind=kind,pair=True)
            self.assertEqual(actual,expected.values);self.assertEqual(index,expected.indices)
        x[1048514]=float('nan');x[-1]=float('nan')
        self.assertEqual(fn(x,kind='max',arg=True),torch.argmax(x))

    def test_gather_tiles_align_ranges_and_preserve_padding_and_raw_bits(self):
        tree=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gather_window')
        starts=[]
        def sliced(x,start,count):
            starts.append(start);self.assertEqual(start*x.element_size()%4096,0)
            self.assertLessEqual(count,65536);return x.view(-1)[start:start+count].clone()
        def run(key,inputs,shape,dtype):
            ids=np.frombuffer(key[3],dtype=np.int64).copy();source=inputs[0]
            data=torch.cat((source,torch.zeros(1,dtype=dtype)))
            return data[torch.from_numpy(ids)].view(shape)
        env=dict(_GATHER_TILE_ELEMENTS=8192,MAX_ELEMENTS=65536,_slice=sliced,_run=run,
                 _dma_dtype=lambda dt:'raw',_concat=lambda xs,shape,dtype:torch.cat(xs).view(shape))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<gather-alignment>','exec'),env)
        for dt in (torch.float32,torch.float16):
            x=torch.arange(100000,dtype=torch.float32).remainder(97).to(dt)
            ids=np.array([[21845,21846,-1],[65534,65535,65536],[99997,99998,99999]],dtype=np.int64)
            actual=env['_gather_window'](x,ids,1)
            expected=torch.cat((x,torch.zeros(1,dtype=dt)))[torch.from_numpy(ids)]
            self.assertTrue(torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)))
        self.assertTrue(starts)


if __name__=='__main__':unittest.main()

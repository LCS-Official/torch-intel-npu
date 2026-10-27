"""Wide selection routing/storage guards without loading the native backend."""
import ast
from contextlib import nullcontext
from math import prod
from pathlib import Path
import struct
import unittest
import torch
from torch_npu_geometry import max_pool2d_geometry,selection_plane_tiles


def environment():
    source=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
    nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('_selection_backward','_selection_accumulate')]
    keys=[]
    def run(key,inputs,shape,dtype):
        keys.append(key[0]);packet=inputs[0].view(-1)
        raw=packet.numpy().tobytes()
        fields=struct.unpack_from('<8I',raw) if key[0].endswith('u32') else tuple(int(v) for v in packet[:8])
        count,start,per,width,base,span,flags,total=fields
        assert base+span<=total and (start+count-1)//per<total//width
        result=torch.zeros(shape,dtype=dtype).view(-1);result[:span]=inputs[1].view(-1)[:span]
        labels=struct.unpack_from('<'+'q'*count,raw,4*(8+count))
        for value,index,j in zip(packet[8:8+count],labels,range(start,start+count)):
            if flags&4 and index==-1:continue
            if not 0<=index<width:result[-1]=1;break
            pos=j//per*width+index-base
            if 0<=pos<span:
                value=value if flags&2 else result[pos]+value
                result[pos]=value.half().float() if flags&1 and not flags&2 else value
        return result.view(shape)
    env=dict(torch=torch,prod=prod,struct=struct,_lock=nullcontext(),MAX_TENSOR_ELEMENTS=16777216,
        _validate_float_read=lambda x:None,_validate_read=lambda x:None,_storage_limit=lambda dt:268435456,
        _cast=lambda x,dt:x.to(dt),_materialize=lambda x:x.contiguous(),_clone=lambda x:x.clone(),
        _slice=lambda x,s,n:x.reshape(-1)[s:s+n].clone(),_scalar=lambda v,dt:torch.tensor(v,dtype=dt),
        _broadcast=lambda x,shape:x.expand(shape).clone(),_run=run,
        _fill=lambda x,v:x.fill_(v),_stream_concat=lambda xs,shape,dtype=torch.float32:torch.cat(list(xs)).to(dtype).view(shape),
        _concat=lambda xs,shape,dt:torch.cat([x.view(-1) for x in xs]).to(dt).view(shape))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<wide-selection>', 'exec'),env)
    return env,keys


class SelectionCapacityTests(unittest.TestCase):
    def test_wide_plane_preserves_odd_indices_and_assignment_bits(self):
        env,keys=environment();fn=env['_selection_backward'];width=16777219
        for dtype in (torch.float32,torch.float16):
            gradient=torch.tensor([.125,-.0,-.25,.5],dtype=dtype)
            ids=torch.tensor([width-2,width-1,3,3])
            result=fn(gradient,ids,(width,),4,width,accumulate=False)
            expected=torch.zeros(width,dtype=dtype);expected[width-2]=.125;expected[-1]=-.0;expected[3]=.5
            self.assertTrue(torch.equal(result.view(torch.uint8),expected.view(torch.uint8)))
        self.assertEqual(set(keys),{'selection-backward-u32'})

    def test_old_path_rounding_padding_initial_value_and_invalid_ids(self):
        env,keys=environment();fn=env['_selection_backward']
        for dtype in (torch.float32,torch.float16):
            g=torch.tensor([2048.,1.,-2048.,.25],dtype=dtype);ids=torch.tensor([1,1,1,-1])
            initial=torch.tensor([-.0,.125,5.],dtype=dtype);expected=initial.clone()
            for value in g[:3]:expected[1]=(expected[1].float()+value).to(dtype)
            result=fn(g,ids,(3,),4,3,allow_padding=True,initial_value=initial)
            self.assertTrue(torch.equal(result.view(torch.uint8),expected.view(torch.uint8)))
            with self.assertRaises(IndexError):fn(g,ids,(3,),4,3)
        self.assertEqual(set(keys),{'selection-backward-f32'})

    def test_geometry_and_storage_errors_happen_before_work(self):
        env,keys=environment();fn=env['_selection_backward']
        g=torch.empty(2,device='meta');ids=torch.empty(2,dtype=torch.int64,device='meta')
        for shape,per,width in (((268435457,),2,268435457),((-2,),2,2),((2,),True,2),
                                ((3,),1,2),((3,),0,3),((3,),3,3)):
            with self.assertRaises(ValueError):fn(g,ids,shape,per,width)
        self.assertEqual(keys,[])

    def test_plans_and_spatial_outputs_accept_wide_planes_with_storage_caps(self):
        self.assertEqual(selection_plane_tiles(2,16777219,16777221,max_elements=268435456),((0,1),(1,1)))
        self.assertEqual(max_pool2d_geometry((1,16777219),1,(),0,1,False)[-1],(1,16777219))
        with self.assertRaises(ValueError):selection_plane_tiles(1,1,268435457,max_elements=268435456)


if __name__=='__main__':unittest.main()

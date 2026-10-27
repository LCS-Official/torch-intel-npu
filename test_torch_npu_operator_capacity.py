"""Metadata-only checks for storage-backed workspaces; no NPU import."""
import ast
from contextlib import nullcontext
from math import isfinite,prod,isqrt
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch_npu_geometry import conv_geometry,softmax_row_tiles
from npunlock.npu4000 import MAX_DMA_ELEMENTS


def environment():
    names=('_conv_spec','_native_mha_spec','_transformer_encoder_layer_fwd','_native_half_masked_softmax','_concat_many',
           '_index_storage_bound','_index_select_spec','_index_large','_index_packet','_topk_spec')
    tree=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
    import operator
    env=dict(torch=torch,operator=operator,isfinite=isfinite,prod=prod,MAX_DMA_ELEMENTS=MAX_DMA_ELEMENTS,MAX_TENSOR_ELEMENTS=16777216,
             _lock=nullcontext(),synchronize=lambda:None,_validate=lambda *a,**k:None,
             _validate_float_read=lambda x:None,_validate_read=lambda x:None,_inference_only=lambda x:None,
             conv_geometry=conv_geometry,softmax_row_tiles=softmax_row_tiles,
             _materialize=lambda x:x,_cast=lambda x,dt:x.to(dt),_mask_cast=lambda x,dt:x.to(dt),
             _check_out=lambda *a:None,
             _run_rowwise=lambda kind,x,y:x,_layer_norm=lambda x,*a:(x,),
             _native_multi_head_attention=lambda x,*a:(x,None),_binary_op=lambda x,y,kind:x+y,
             _linear=lambda x,w,b:torch.empty((*x.shape[:-1],w.shape[0]),device='meta',dtype=x.dtype),
             _execute=lambda kind,x:x,bridge=SimpleNamespace(_DTYPES={torch.float32:('f32',None)}),
             _run_many=lambda key,inputs,specs,outs:tuple(torch.empty(shape,device='meta',dtype=dt) for shape,dt in specs))
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],
                            type_ignores=[]),'<operator-capacity>','exec'),env)
    return env


def encoder_arguments(length=1025,width=16384):
    new=lambda *s:torch.empty(s,device='meta')
    return [new(1,length,2),2,1,new(6,2),new(6),new(2,2),new(2),False,True,1e-5,
            new(2),new(2),new(2),new(2),new(width,2),new(width),new(2,width),new(2)]


class OperatorCapacityTests(unittest.TestCase):
    def test_index_source_output_and_metadata_bound(self):
        env=environment();fn=env['_index_select_spec']
        for rows,width,count in ((65529,17,2),(4,1,32765),(131073,128,1),(1,128,131073),
                                 (1048576,256,1),(1,256,1048576)):
            x=torch.empty(rows,width,device='meta');ids=torch.empty(count,dtype=torch.int64,device='meta')
            self.assertEqual(fn(x,0,ids),(0,rows,width,(count,width)))
        for rows,width,count in ((0,3,1),(2,0,1),(2,3,0),(1048577,256,1),(1,256,1048577),
                                 (16777217,1,1),(1,1,16777217)):
            with self.assertRaises(ValueError):env['_index_storage_bound'](rows,width,count)

    def test_large_index_duplicate_padding_and_rounding_against_cpu(self):
        env=environment()
        def normalize(ids,rows):
            if torch.any((ids<0)|(ids>=rows)):raise IndexError('bounds')
            return ids
        module=SimpleNamespace(index_select=torch.index_select,normalize=normalize,
            integer_binary=lambda x,y,kind:x*y if kind=='mul' else x+y)
        def selection(g,ids,shape,per,width,initial_value=None):
            result=torch.zeros(shape,dtype=torch.float32) if initial_value is None else initial_value.float().clone()
            flat=result.view(-1)
            for value,index in zip(g,ids):
                flat[index]=(flat[index]+value).to(g.dtype).float()
            return result.to(g.dtype)
        env.update(_boolean_reduce=lambda x,kind:torch.any(x),
                   _compare=lambda x,y,kind:x<y if kind=='lt' else x==y,
                   _where=torch.where,_selection_backward=selection)
        fn=env['_index_large'];labels=torch.tensor([2,0,2,3,2,0])
        for dtype in (torch.float32,torch.float16):
            for embedding in (False,True):
                source=torch.arange(12,dtype=dtype).view(4,3).requires_grad_()
                grad=torch.tensor([[.3333,-.2,.125],[-.2,.125,-.1],[.125,-.1,.3333],
                                   [-.1,.3333,-.2],[.3333,-.2,.125],[float('nan'),.25,.5]],dtype=dtype)
                ref=(torch.nn.functional.embedding(labels,source,padding_idx=0) if embedding else
                     torch.index_select(source,0,labels))
                ref.backward(grad)
                with patch.dict('sys.modules',{'torch_npu_compat_extended':module}):
                    actual=fn(grad,labels,4,3,backward=True,padding_idx=0 if embedding else -1,round_half=embedding)
                    forward=fn(source.detach(),labels,4,3)
                    with self.assertRaises(IndexError):fn(grad,torch.tensor([2,0,2,3,4,0]),4,3,backward=True)
                    with self.assertRaises(IndexError):fn(grad,torch.tensor([2,0,2,3,-1,0]),4,3,backward=True)
                torch.testing.assert_close(actual.view(4,3),source.grad,rtol=0,atol=0,equal_nan=True)
                torch.testing.assert_close(forward.view(6,3),ref,rtol=0,atol=0)

    def test_topk_uses_metadata_capacity_for_input_and_output(self):
        fn=environment()['_topk_spec']
        for dtype in (torch.float16,torch.float32,torch.int64,torch.bool):
            x=torch.empty(8193,129,device='meta',dtype=dtype)
            self.assertEqual(fn(x,129,-1)[-1],(8193,129))
            full=torch.empty(16777216,device='meta',dtype=dtype)
            self.assertEqual(fn(full,16777216,0)[-1],(16777216,))
            with self.assertRaises(ValueError):fn(torch.empty(16777217,device='meta',dtype=dtype),1,0)

    def test_column_partitions_preserve_layout_beyond_packet_metadata(self):
        env=environment();env['MAX_TENSOR_ELEMENTS']=16
        def too_large(*args):raise ValueError('force metadata column tiling')
        env.update(index_tile_width=too_large,_materialize=lambda x:x.contiguous(),
            _slice=lambda x,start,count:x.reshape(-1)[start:start+count],
            _concat=lambda xs,shape,dtype:torch.cat([x.reshape(-1) for x in xs]).to(dtype).view(shape),
            _index_large=lambda x,ids,rows,width,**kw:torch.index_select(x.reshape(rows,width),0,ids).reshape(-1))
        source=torch.arange(13*7,dtype=torch.float32).view(13,7);ids=torch.tensor([12,0,12])
        result=env['_index_packet'](source,ids,13,7)
        torch.testing.assert_close(result.view(3,7),source[ids],rtol=0,atol=0)

    def test_convolution_output_uses_storage_bound(self):
        fn=environment()['_conv_spec'];x=torch.empty(1,1,1,16385,device='meta')
        args=((1,1),(0,0),(1,1),False,(0,0),1)
        spatial,params,shape=fn(x,torch.empty(1024,1,1,1,device='meta'),None,*args)
        self.assertEqual(shape,(16385,1));self.assertEqual(spatial,(1,16385))
        with self.assertRaises(ValueError):fn(x,torch.empty(MAX_DMA_ELEMENTS//16385+1,1,1,1,device='meta'),None,*args)

    def test_native_mha_score_uses_storage_bound_and_retains_shape_checks(self):
        fn=environment()['_native_mha_spec']
        args=encoder_arguments();weights=args[3:7]
        for length in (4097,isqrt(MAX_DMA_ELEMENTS)+1):
            q=torch.empty(1,length,2,device='meta')
            call=lambda:fn(q,q,q,2,1,*weights,None,None)
            if length==4097:call()
            else:
                with self.assertRaises(ValueError):call()

    def test_encoder_ffn_uses_storage_bound_before_execution(self):
        env=environment();args=encoder_arguments()
        self.assertEqual(env['_transformer_encoder_layer_fwd'](*args).shape,args[0].shape)
        with self.assertRaises(ValueError):env['_transformer_encoder_layer_fwd'](*encoder_arguments(width=MAX_DMA_ELEMENTS//1025+1))

    def test_native_half_masked_softmax_no_longer_uses_legacy_million_limit(self):
        env=environment();x=torch.empty(256,4100,dtype=torch.float16,device='meta');mask=torch.empty_like(x,dtype=torch.bool)
        self.assertEqual(env['_native_half_masked_softmax'](x,mask).shape,x.shape)
        wide=torch.empty(1,65537,dtype=torch.float16,device='meta')
        with self.assertRaises(ValueError):env['_native_half_masked_softmax'](wide,torch.empty_like(wide,dtype=torch.bool))

    def test_multi_concat_uses_dma_capacity_without_numeric_metadata(self):
        fn=environment()['_concat_many'];a=torch.empty(16777216,device='meta');b=torch.empty(17,device='meta')
        outputs=fn(((a,b),(b,a)),((16777233,),(16777233,)),torch.float32)
        self.assertEqual([x.numel() for x in outputs],[16777233,16777233])
        large=torch.empty(MAX_DMA_ELEMENTS,device='meta')
        with self.assertRaises(ValueError):fn(((large,b),(b,large)),((MAX_DMA_ELEMENTS+17,),(MAX_DMA_ELEMENTS+17,)),torch.float32)


if __name__=='__main__':unittest.main()

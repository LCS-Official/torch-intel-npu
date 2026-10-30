"""CPU oracle checks for loss geometry, soft targets and weighted smoothing."""
import ast
from contextlib import nullcontext
from math import prod
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch.autograd.function import once_differentiable
from torch_npu_geometry import nll_row_tiles
from npunlock.npu4000 import MAX_DMA_ELEMENTS


def environment():
    tree=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
    names=('_mse_spec','_mse_loss','_mse_loss_backward','_classification_weight','_nll_spec',
           '_nll_row_weights','_nll_packet','_nll_forward','_nll_backward','_classification_input','_cross_entropy','_smooth_classification_loss')
    def validate(x):
        if x.dtype not in (torch.float16,torch.float32):raise ValueError('floating dtype required')
    def outcheck(out,shape,dtype,inputs):
        if out is not None and (tuple(out.shape)!=tuple(shape) or out.dtype!=dtype or
                               any(out.untyped_storage()._cdata==v.untyped_storage()._cdata for v in inputs)):
            raise ValueError('invalid out')
    def cast(x,dtype,*,out=None):
        result=x.to(dtype)
        return out.copy_(result) if out is not None else result
    def reduce(x,dim=None,*,kind):
        return getattr(x,kind)(dim=dim) if dim is not None else getattr(x,kind)()
    def run(key,inputs,shape,dtype):
        if key[0]=='nll-packet':return inputs[0]
        if key[0]=='nll-backward':return torch.where(inputs[0]==1,-inputs[1],0.)
        assert key[0]=='class-pack'
        x,labels,ignored=inputs;rows,classes=x.shape
        labels,ignored=(v.contiguous().view(torch.int64).reshape(-1) for v in (labels,ignored))
        selected=(labels>=0)&(labels<classes)&(labels!=ignored);r=torch.arange(rows)[selected];t=labels[selected]
        packet=torch.zeros(shape,dtype=dtype);packet[r,t]=1;packet[r,classes]=-x[r,t]
        packet[:,classes+1]=(~selected)&(labels!=ignored);packet[:,classes+2]=selected
        return packet
    env=dict(torch=torch,prod=prod,MAX_RANK=64,MAX_DMA_ELEMENTS=MAX_DMA_ELEMENTS,MAX_ELEMENTS=65536,
             _validate_float_read=validate,_validate_read=lambda x:None,_lock=nullcontext(),synchronize=lambda:None,
             _check_out=outcheck,_cast=cast,_materialize=lambda x:x.contiguous(),
             _binary_op=lambda x,y,kind:getattr(torch,kind)(x,y),_execute=lambda kind,x,y:getattr(torch,kind)(x,y),
             _reduce=reduce,_class_columns=lambda x,start,count:x[:,start:start+count],
             _compare=lambda x,y,kind:getattr(torch,kind)(x,y),_where=torch.where,_run=run,
             _scalar=lambda x,dtype:torch.tensor(x,dtype=dtype),_broadcast=lambda x,shape:x.expand(shape).contiguous())
    env.update(nll_row_tiles=nll_row_tiles,_slice=lambda x,start,count:x.view(-1)[start:start+count].clone(),
               _concat=lambda xs,shape,dtype:torch.cat([v.reshape(-1) for v in xs]).reshape(shape))
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],
                            type_ignores=[]),'<losses>','exec'),env)
    source=ast.parse(Path(__file__).with_name('torch_npu_autograd.py').read_text(encoding='utf-8'))
    functions={'_fresh_output','_active','nll','nll_nd','nll2d_forward','_guard_loss_sources'}
    scope=dict(torch=torch,npu=SimpleNamespace(**env),once_differentiable=once_differentiable)
    exec(compile(ast.Module(body=[n for n in source.body if
        isinstance(n,ast.FunctionDef) and n.name in functions or
        isinstance(n,ast.ClassDef) and n.name in {'NLL','LossSourceGuard'}],type_ignores=[]),'<loss-autograd>','exec'),scope)
    env['torch_npu_autograd']=SimpleNamespace(**scope)
    return env


class LossTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(429);torch.set_num_threads(1)

    def test_tiled_nll_matches_native_and_keeps_only_labels_for_backward(self):
        env=environment()
        for dtype in (torch.float32,torch.float16):
            x=torch.randn(513,257,dtype=dtype);target=torch.arange(513)%257;target[250]=-100
            weight=torch.linspace(.5,1.5,257,dtype=dtype)
            for red in (0,1,2):
                actual,total,packet=env['_nll_forward'](x,target,weight,red,-100)
                expected,ref_total=torch.ops.aten.nll_loss_forward(x,target,weight,red,-100)
                torch.testing.assert_close(actual,expected,rtol=.004 if dtype==torch.float16 else 4e-5,atol=.001 if dtype==torch.float16 else 2e-6)
                torch.testing.assert_close(total,ref_total)
                self.assertEqual(packet.numel(),0)
                g=torch.ones_like(actual)
                torch.testing.assert_close(env['_nll_backward'](g,x,packet,total,red,weight,target=target,validate_target=True),
                                           torch.ops.aten.nll_loss_backward(g,x,target,weight,red,-100,ref_total),
                                           rtol=.004 if dtype==torch.float16 else 4e-5,atol=.001 if dtype==torch.float16 else 2e-6)
            target[-1]=257
            with self.assertRaises(IndexError):env['_nll_forward'](x,target,weight,1,-100)
            with self.assertRaises(IndexError):env['_nll_backward'](torch.tensor(1.,dtype=dtype),x,packet,total,1,weight,target=target,validate_target=True)

    def test_nll_tiles_cover_storage_and_preserve_complete_rows(self):
        self.assertEqual(nll_row_tiles(0,100000,max_elements=134217728),())
        self.assertEqual(nll_row_tiles(3,65532,max_elements=134217728),((0,1),(1,1),(2,1)))
        self.assertEqual(nll_row_tiles(9000,3,max_elements=134217728),((0,9000),))
        aligned=nll_row_tiles(1048593,3,max_elements=134217728)
        self.assertTrue(all(start%1024==0 for start,count in aligned))
        self.assertEqual(aligned[-1],(1041408,7185))
        for rows,columns in ((513,257),(32769,3),(4096,1024)):
            tiles=nll_row_tiles(rows,columns,max_elements=134217728)
            self.assertEqual(sum(count for start,count in tiles),rows)
            self.assertTrue(all(count*(columns+4)<=65536 for start,count in tiles))
        for args in ((1,65533),(2,67108865),(-1,3),(1,0)):
            with self.assertRaises(ValueError):nll_row_tiles(*args,max_elements=134217728)

    def test_contiguous_spatial_labels_keep_the_original_version(self):
        env=environment();x=torch.randn(2,3,2,2,requires_grad=True);target=torch.zeros(2,2,2,dtype=torch.int64)
        loss=env['_cross_entropy'](x,target)
        target.add_(1)
        with self.assertRaisesRegex(RuntimeError,'version|inplace'):loss.backward()

    @staticmethod
    def version_error(loss,owner):
        with torch.no_grad():owner.add_(0)
        try:loss.backward()
        except RuntimeError as error:
            if 'version' in str(error) or 'inplace' in str(error):return True
            raise
        return False

    def test_strided_spatial_label_versions_follow_native_rank_policy(self):
        env=environment();auto=env['torch_npu_autograd']
        for shape in ((2,3,2),(2,3,2,2),(2,3,2,2,2)):
            for kind in ('ce','smooth','nll','native2d'):
                if kind=='native2d' and len(shape)!=4:continue
                with self.subTest(shape=shape,kind=kind):
                    size=(shape[0],shape[2]*2,*shape[3:])
                    owner=torch.zeros(size,dtype=torch.int64);reference=owner.clone()
                    target,labels=owner[:,::2],reference[:,::2]
                    x=torch.randn(shape,requires_grad=True);y=x.detach().clone().requires_grad_()
                    if kind in ('ce','smooth'):
                        smoothing=.2 if kind=='smooth' else 0.
                        actual=env['_cross_entropy'](x,target,label_smoothing=smoothing)
                        expected=torch.nn.functional.cross_entropy(y,labels,label_smoothing=smoothing)
                    elif kind=='nll':
                        actual=auto.nll_nd(x,target);expected=torch.nn.functional.nll_loss(y,labels)
                    else:
                        actual=auto.nll2d_forward(x,target,None,1,-100)[0]
                        expected=torch.ops.aten.nll_loss2d_forward(y,labels,None,1,-100)[0]
                    torch.testing.assert_close(actual,expected)
                    wanted=self.version_error(expected,reference)
                    self.assertEqual(self.version_error(actual,owner),wanted)
                    if not wanted:torch.testing.assert_close(x.grad,y.grad)

    def test_spatial_nll_input_versions_follow_native_copy_policy(self):
        auto=environment()['torch_npu_autograd']
        for shape in ((2,3,2),(2,3,2,2),(2,3,2,2,2)):
            for strided in (False,True):
                with self.subTest(shape=shape,strided=strided):
                    size=(*shape[:-1],shape[-1]*(2 if strided else 1))
                    owner=torch.randn(size);reference=owner.clone()
                    x=(owner[...,::2] if strided else owner).requires_grad_()
                    y=(reference[...,::2] if strided else reference).requires_grad_()
                    target=torch.zeros((shape[0],*shape[2:]),dtype=torch.int64)
                    actual=auto.nll_nd(x,target);expected=torch.nn.functional.nll_loss(y,target)
                    torch.testing.assert_close(actual,expected)
                    wanted=self.version_error(expected,reference)
                    self.assertEqual(self.version_error(actual,owner),wanted)
                    if not wanted:torch.testing.assert_close(x.grad,y.grad)

    def test_spatial_label_version_is_checked_again_after_retained_backward(self):
        fn=environment()['_cross_entropy'];x=torch.randn(2,3,2,2,requires_grad=True)
        owner=torch.zeros(2,4,2,dtype=torch.int64);loss=fn(x,owner[:,::2])
        loss.backward(retain_graph=True)
        self.assertTrue(self.version_error(loss,owner))
        with torch.no_grad():
            output=fn(x,owner[:,::2]);self.assertFalse(output.requires_grad)

    def test_source_guards_preserve_all_reductions_and_gradients(self):
        env=environment();auto=env['torch_npu_autograd']
        for kind in ('ce','nll','native2d'):
            for red in (0,1,2):
                with self.subTest(kind=kind,reduction=red):
                    x=torch.randn(2,3,2,2,requires_grad=True);y=x.detach().clone().requires_grad_()
                    owner=torch.arange(16).reshape(2,2,4)%3;reference=owner.clone()
                    target,labels=owner[...,::2],reference[...,::2];w=torch.tensor([.5,1.25,2.])
                    if kind=='ce':
                        actual=env['_cross_entropy'](x,target,w,red,-100,.2)
                        expected=torch.nn.functional.cross_entropy(y,labels,weight=w,reduction=('none','mean','sum')[red],label_smoothing=.2)
                    elif kind=='nll':
                        actual=auto.nll_nd(x,target,w,red)
                        expected=torch.nn.functional.nll_loss(y,labels,weight=w,reduction=('none','mean','sum')[red])
                    else:
                        actual=auto.nll2d_forward(x,target,w,red,-100)[0]
                        expected=torch.ops.aten.nll_loss2d_forward(y,labels,w,red,-100)[0]
                    torch.testing.assert_close(actual,expected)
                    g=torch.randn_like(expected);actual.backward(g,retain_graph=True);expected.backward(g,retain_graph=True)
                    torch.testing.assert_close(x.grad,y.grad)
                    owner.add_(0);reference.add_(0)
                    for loss in (actual,expected):
                        with self.assertRaisesRegex(RuntimeError,'version|inplace'):loss.backward(g)

    def test_mse_forward_backward_native_oracle(self):
        env=environment()
        for dtype in (torch.float16,torch.float32):
            for shapes in (((2,1),(1,3)),((),(2,3)),((0,1),(1,3)),((1,)*63+(2,),(2,))):
                x,y=(torch.randn(s,dtype=dtype) for s in shapes)
                for reduction in (0,1,2):
                    with self.subTest(dtype=dtype,shapes=shapes,reduction=reduction):
                        expected=torch.ops.aten.mse_loss(x,y,reduction)
                        actual=env['_mse_loss'](x,y,reduction)
                        torch.testing.assert_close(actual,expected,equal_nan=True)
                        g=torch.ones_like(expected)
                        torch.testing.assert_close(env['_mse_loss_backward'](g,x,y,reduction),
                                                   torch.ops.aten.mse_loss_backward.grad_input(g,x,y,reduction,
                                                       grad_input=torch.empty(torch.broadcast_shapes(x.shape,y.shape),dtype=x.dtype)),equal_nan=True)

    def test_mse_capacity_and_out_contract(self):
        env=environment();x=torch.ones(2,1);y=torch.zeros(1,3)
        out=torch.empty(2,3);self.assertIs(env['_mse_loss'](x,y,0,out=out),out)
        with self.assertRaises(ValueError):env['_mse_loss'](x,y,0,out=x)
        with self.assertRaises(ValueError):env['_mse_spec'](torch.empty(MAX_DMA_ELEMENTS//16384+1,1,device='meta'),torch.empty(1,16384,device='meta'),1)
        with self.assertRaises(ValueError):env['_mse_spec'](x,y,3)
        self.assertEqual(env['_mse_spec'](x.half(),y,1)[1],torch.float32)

    def test_soft_target_outputs_and_all_three_gradients(self):
        fn=environment()['_cross_entropy']
        for shape in ((3,),(2,3),(2,3,2,1),(0,3,2)):
            for dtype in (torch.float16,torch.float32):
                for reduction in (0,1,2):
                    for smooth in (0.,.2,1.):
                        inputs=[torch.randn(shape,dtype=dtype,requires_grad=True),
                                torch.randn(shape,requires_grad=True),
                                torch.randn(shape[0] if len(shape)==1 else shape[1],requires_grad=True)]
                        refs=[v.detach().clone().requires_grad_() for v in inputs]
                        a=fn(*inputs[:2],inputs[2],reduction,-2,smooth)
                        b=torch.nn.functional.cross_entropy(*refs[:2],weight=refs[2],reduction=('none','mean','sum')[reduction],
                                                           ignore_index=-2,label_smoothing=smooth)
                        torch.testing.assert_close(a,b,rtol=.004 if dtype==torch.float16 else 4e-5,
                                                   atol=.001 if dtype==torch.float16 else 2e-6,equal_nan=True)
                        g=torch.randn_like(a);a.backward(g);b.backward(g)
                        for x,y in zip(inputs,refs):torch.testing.assert_close(x.grad,y.grad,equal_nan=True)

    def test_hard_weighted_smoothing_matches_native(self):
        fn=environment()['_cross_entropy']
        for dtype in (torch.float16,torch.float32):
            for shape in ((3,),(2,3),(2,3,2,1),(0,3,2)):
                x=torch.randn(shape,dtype=dtype,requires_grad=True)
                target=torch.zeros(() if len(shape)==1 else (shape[0],*shape[2:]),dtype=torch.int64)
                w=torch.tensor([.5,0.,-1.],dtype=dtype)
                for red in (0,1,2):
                    for smooth in (0.,.2,1.):
                        a=fn(x,target,w,red,-100,smooth)
                        b=torch.nn.functional.cross_entropy(x,target,weight=w,reduction=('none','mean','sum')[red],label_smoothing=smooth)
                        torch.testing.assert_close(a,b,rtol=.004 if dtype==torch.float16 else 4e-5,
                                                   atol=.001 if dtype==torch.float16 else 2e-6,equal_nan=True)

    def test_selection_skips_unselected_nonfinite_weights(self):
        fn=environment()['_nll_row_weights']
        packet=torch.zeros(2,7);packet[0,0]=1
        actual=fn(packet,3,torch.tensor([2.,float('nan'),float('inf')]))
        torch.testing.assert_close(actual,torch.tensor([2.,0.]))

    def test_half_weight_extremes_and_native_backward_rounding(self):
        env=environment()
        for shape in ((3,),(2,3)):
            for scale in (60000.,1.e-7,0.,-.5):
                x=torch.full(shape,-1.25,dtype=torch.float16);t=torch.zeros(shape[:-1],dtype=torch.int64)
                w=torch.full((3,),scale,dtype=torch.float16)
                for red in (0,1,2):
                    a,total,packet=env['_nll_forward'](x,t,w,red,-100)
                    b,ref_total=torch.ops.aten.nll_loss_forward(x,t,w,red,-100)
                    torch.testing.assert_close(a,b,equal_nan=True);torch.testing.assert_close(total,ref_total)
                    g=torch.ones_like(a)
                    torch.testing.assert_close(env['_nll_backward'](g,x,packet,total,red,w),
                                               torch.ops.aten.nll_loss_backward(g,x,t,w,red,-100,ref_total),equal_nan=True)

    def test_class_weight_metadata_and_soft_target_errors(self):
        env=environment();fn=env['_classification_weight'];x=torch.ones(2,3)
        for w in (torch.ones(2),torch.ones(3,dtype=torch.float16),torch.ones(3,requires_grad=True)):
            with self.assertRaises((ValueError,RuntimeError)):fn(x,w,3)
        fn(x,torch.ones(3,dtype=torch.float16,requires_grad=True),3,soft=True)
        for target,kw in ((x,dict(ignore_index=0)),(x.long(),{}),(x,dict(label_smoothing=1.1))):
            with self.assertRaises(ValueError):env['_cross_entropy'](x,target,**kw)


if __name__=='__main__':unittest.main()

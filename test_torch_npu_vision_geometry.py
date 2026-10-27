"""Offline geometry checks against independent CPU PyTorch operators."""
import unittest
from math import prod
import numpy as np
import torch
import torch.nn.functional as F
from torch_npu_geometry import (resample_axis_geometry,transpose_resample_geometry,transpose_conv_windows,
                               validate_resample_backward_geometry)


class VisionGeometry(unittest.TestCase):
    def test_explicit_1d_scale_including_unchanged_size(self):
        for mode in ('nearest','nearest-exact','linear'):
            for size,scale in ((4,1.24),(5,2.1),(7,.7)):
                x=torch.randn(1,1,size,requires_grad=True);args=dict(mode=mode,scale_factor=scale)
                if mode=='linear':args['align_corners']=False
                y=F.interpolate(x,**args);positions,weights=resample_axis_geometry(size,y.shape[-1],mode,False,scale,legacy_nearest=False)
                actual=(x.detach().numpy().reshape(-1)[positions]*weights).sum(1)
                np.testing.assert_allclose(actual,y.detach().numpy().reshape(-1),rtol=2e-5,atol=2e-6)
                g=torch.randn_like(y);y.backward(g);indices,coeff=transpose_resample_geometry(size,positions,weights)
                actual=(np.append(g.numpy().reshape(-1),0)[indices]*coeff).sum(1)
                try:validate_resample_backward_geometry((size,),(y.shape[-1],),mode,False,(scale,))
                except ValueError:
                    self.assertFalse(np.allclose(actual,x.grad.numpy().reshape(-1),rtol=2e-5,atol=2e-6))
                    continue
                np.testing.assert_allclose(actual,x.grad.numpy().reshape(-1),rtol=2e-5,atol=2e-6)

    def test_interpolation_and_transpose(self):
        torch.manual_seed(34)
        for mode in ('nearest','nearest-exact','linear','cubic'):
            for size,output,scale in ((5,8,None),(7,3,None),(1,9,None),(9,1,None),(5,8,1.7),(4,4,1.1),(4,4,1.24),(5,10,2.1)):
                for align in ((False,True) if mode in ('linear','cubic') else (False,)):
                    with self.subTest(mode=mode,size=size,output=output,scale=scale,align=align):
                        indices,weights=resample_axis_geometry(size,output,mode,align,scale)
                        x=torch.randn(1,1,1,size,requires_grad=True)
                        args=dict(mode={'linear':'bilinear','cubic':'bicubic'}.get(mode,mode))
                        if mode in ('linear','cubic'):args['align_corners']=align
                        actual=(x.detach().numpy().reshape(-1)[indices]*weights).sum(1)
                        # Use height 2 -> 3 when width is unchanged: avoid Torch's whole-tensor copy shortcut.
                        x=x.expand(1,1,2,size).clone().detach().requires_grad_()
                        if scale is None:args['size']=(3,output)
                        else:args['scale_factor']=(1.5,scale)
                        y=F.interpolate(x,**args);expected=y.detach().numpy()[0,0,1]
                        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
                        back_indices,back_weights=transpose_resample_geometry(size,indices,weights)
                        gradient=torch.randn(output);y[0,0].sum(0).backward(gradient)
                        padded=np.append(gradient.numpy(),0)
                        backward=(padded[back_indices]*back_weights).sum(1)
                        np.testing.assert_allclose(backward,x.grad.numpy().sum((0,1,2))/3,rtol=2e-5,atol=2e-6)

    def test_conv_transpose_windows_and_both_gradients(self):
        torch.manual_seed(41)
        for kernel,stride,padding,dilation,output_padding in (
                ((3,3),(1,1),(0,0),(1,1),(0,0)),((2,3),(2,1),(0,1),(1,1),(1,0)),
                ((3,3),(1,1),(2,2),(3,3),(2,2)),((1,1),(2,3),(1,0),(1,1),(1,2))):
            x=torch.randn(2,2,3,4,dtype=torch.float64,requires_grad=True)
            w=torch.randn(2,3,*kernel,dtype=torch.float64,requires_grad=True)
            y=F.conv_transpose2d(x,w,stride=stride,padding=padding,dilation=dilation,output_padding=output_padding)
            params=kernel,stride,padding,dilation;rows=y.numel()//y.shape[1]
            source=np.append(x.detach().permute(0,2,3,1).numpy().reshape(-1),0)
            indices=transpose_conv_windows(tuple(x.shape),tuple(y.shape),*params,0,rows)
            output=torch.from_numpy(source[indices])@w.detach().permute(2,3,0,1).reshape(-1,3)
            torch.testing.assert_close(output.view(2,*y.shape[2:],3).permute(0,3,1,2),y)
            g=torch.randn_like(y);y.backward(g)
            source=np.append(g.permute(0,2,3,1).numpy().reshape(-1),0);rows=x.numel()//2
            indices=transpose_conv_windows(tuple(x.shape),tuple(y.shape),*params,0,rows,backward=True)
            columns=torch.from_numpy(source[indices]);a=x.detach().permute(0,2,3,1).reshape(-1,2)
            gx=(columns@w.detach().permute(2,3,1,0).reshape(-1,2)).view(2,*x.shape[2:],2).permute(0,3,1,2)
            gw=(a.t()@columns).view(2,*kernel,3).permute(0,3,1,2)
            torch.testing.assert_close(gx,x.grad);torch.testing.assert_close(gw,w.grad)

    def test_rejected_geometry(self):
        for values in ((0,3,'nearest',False,None),(2,0,'linear',False,None),(2,3,'invalid',False,None),(2,3,'linear',False,0),(2,3,'linear',False,float('nan'))):
            with self.assertRaises(ValueError):resample_axis_geometry(*values)


if __name__=='__main__':unittest.main()

"""CPU-only tests of new visual metadata and exact adjoint construction."""
import unittest
from math import prod
import numpy as np
import torch
import torch.nn.functional as F
from torch_npu_geometry import (convolution_windows_nd,antialias_axis_geometry,
                               padding_axis_geometry,transpose_resample_geometry)


class ExtendedGeometry(unittest.TestCase):
    def test_volume_convolution_windows_and_adjoint(self):
        torch.manual_seed(824)
        for transpose in (False,True):
            for stride,padding,dilation in (((1,1,1),(0,0,0),(1,1,1)),((2,1,2),(1,0,1),(2,1,1))):
                x=torch.randn(2,2,3,4,5,dtype=torch.float64,requires_grad=True)
                w=torch.randn((2,3,2,2,2) if transpose else (3,2,2,2,2),dtype=torch.float64,requires_grad=True)
                output=(F.conv_transpose3d if transpose else F.conv3d)(x,w,stride=stride,padding=padding,dilation=dilation)
                params=tuple(w.shape[2:]),stride,padding,dilation;rows=output.numel()//3
                pos=convolution_windows_nd(tuple(x.shape),tuple(output.shape),*params,0,rows,transpose=transpose)
                source=np.append(x.detach().permute(0,2,3,4,1).numpy().reshape(-1),0)
                matrix=w.detach().permute((2,3,4,0,1) if transpose else (2,3,4,1,0)).reshape(-1,3)
                actual=(torch.from_numpy(source[pos])@matrix).view(2,*output.shape[2:],3).permute(0,4,1,2,3)
                torch.testing.assert_close(actual,output)
                grad=torch.randn_like(output);output.backward(grad)
                pos=convolution_windows_nd(tuple(output.shape),tuple(x.shape),*params,0,x.numel()//2,transpose=not transpose)
                source=np.append(grad.permute(0,2,3,4,1).numpy().reshape(-1),0)
                matrix=w.detach().permute((2,3,4,1,0) if transpose else (2,3,4,0,1)).reshape(-1,2)
                actual=(torch.from_numpy(source[pos])@matrix).view(2,*x.shape[2:],2).permute(0,4,1,2,3)
                torch.testing.assert_close(actual,x.grad)

    def test_antialias_and_adjoint(self):
        torch.manual_seed(52)
        for mode in ('linear','cubic'):
            for align in (False,True):
                for size,output in ((7,3),(3,7),(9,5),(5,2),(2,1),(1,3)):
                    x=torch.randn(1,1,1,size,requires_grad=True)
                    y=F.interpolate(x,size=(1,output),mode='bilinear' if mode=='linear' else 'bicubic',align_corners=align,antialias=True)
                    p,w=antialias_axis_geometry(size,output,mode,align)
                    a=(np.append(x.detach().numpy().reshape(-1),0)[p]*w).sum(1)
                    np.testing.assert_allclose(a,y.detach().numpy().reshape(-1),rtol=2e-5,atol=2e-6)
                    grad=torch.randn_like(y);y.backward(grad);p,w=transpose_resample_geometry(size,p,w)
                    a=(np.append(grad.numpy().reshape(-1),0)[p]*w).sum(1)
                    np.testing.assert_allclose(a,x.grad.numpy().reshape(-1),rtol=2e-5,atol=2e-6)

    def test_padding_and_adjoint(self):
        for mode in ('constant','replicate','reflect','circular'):
            for left,right in ((2,1),(-1,2),(2,-1)):
                x=torch.randn(1,1,5,requires_grad=True);y=F.pad(x,(left,right),mode)
                p,w=padding_axis_geometry(5,left,right,mode)
                np.testing.assert_array_equal(np.append(x.detach().numpy().reshape(-1),0)[p].reshape(-1),y.detach().numpy().reshape(-1))
                g=torch.randn_like(y);y.backward(g);p,w=transpose_resample_geometry(5,p,w)
                np.testing.assert_allclose((np.append(g.numpy().reshape(-1),0)[p]*w).sum(1),x.grad.numpy().reshape(-1),rtol=1e-6,atol=1e-6)

    def test_invalid_geometry(self):
        for args in ((0,3,'linear'),(3,0,'cubic'),(3,2,'unknown')):
            with self.assertRaises(ValueError):antialias_axis_geometry(*args)
        with self.assertRaises(ValueError):padding_axis_geometry(2,2,0,'reflect')
        with self.assertRaises(ValueError):padding_axis_geometry(2,3,0,'circular')


if __name__=='__main__':unittest.main()

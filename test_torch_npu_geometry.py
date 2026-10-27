"""Offline index-geometry checks against CPU PyTorch convolution/autograd."""
import unittest
from math import prod
import numpy as np
import torch
from torch_npu_geometry import (conv_geometry, conv_window_indices, reduction_geometry, triangle_indices,
                               index_tile_width, softmax_row_tiles, indexed_rows_geometry, avg_pool2d_geometry, row_extrema_packet,
                               selection_packet,max_pool2d_geometry,max_pool_windows)


class ConvGeometryTests(unittest.TestCase):
    def test_selection_packet_capacity(self):
        self.assertEqual(selection_packet(1,65530,128),65536)
        self.assertEqual(selection_packet(1,32768,1,True),65536)
        self.assertEqual(selection_packet(21845,1,1),65535)
        for args in ((1,65531,1),(1,10,0),(1,130,129),(1,32769,1,True),(True,2,1)):
            with self.assertRaises(ValueError):selection_packet(*args)

    def test_max_pool_window_geometry_against_cpu(self):
        for shape in ((5,6),(7,9)):
            for kernel,stride,pad,dilation,ceil in ((2,2,0,1,False),(3,2,1,1,True),((2,3),(1,2),(1,1),(2,1),False)):
                params=max_pool2d_geometry(shape,kernel,stride,pad,dilation,ceil)
                data=torch.arange(prod(shape),dtype=torch.float64).view(1,1,*shape)
                values,indices=torch.nn.functional.max_pool2d(data,kernel,stride,pad,dilation,ceil,return_indices=True)
                self.assertEqual(params[-1],tuple(values.shape[-2:]))
                locations=max_pool_windows(shape,params,0,prod(params[-1]))
                patches=np.where(locations>=0,np.maximum(locations,0),-np.inf)
                winners=locations[np.arange(len(locations)),patches.argmax(1)]
                np.testing.assert_array_equal(winners,indices.flatten().numpy())
                start=min(3,len(locations)-1)
                np.testing.assert_array_equal(max_pool_windows(shape,params,start,len(locations)-start),locations[start:])
        for args in (((1,1),2,1,1,2,False),((3,4),0,(),0,1,False),((3,4),2,(),2,1,False)):
            with self.assertRaises(ValueError):max_pool2d_geometry(*args)

    def test_extrema_packet_covers_raw_int64_output_and_candidates(self):
        self.assertEqual(row_extrema_packet(1,65532),65536)
        self.assertEqual(row_extrema_packet(21845,1),65535)
        self.assertEqual(row_extrema_packet(1,16,True),32)
        for rows,width,candidates in ((1,65533,False),(21846,1,False),(1,32769,True),(0,3,False),(True,2,False)):
            with self.assertRaises(ValueError):row_extrema_packet(rows,width,candidates)

    def test_average_pool_geometry_divisors_against_cpu(self):
        for shape in ((3,4),(5,6),(1,7)):
            for kernel,stride,padding in (((2,2),(2,2),(1,1)),((3,2),(2,1),(1,1)),((1,3),(1,2),(0,1))):
                for ceil in (False,True):
                    for include in (False,True):
                        k,s,p,size,extra,divisor=avg_pool2d_geometry(shape,kernel,stride,padding,ceil,include,None)
                        ones=torch.ones(1,1,*shape,dtype=torch.float64)
                        expected=torch.nn.functional.avg_pool2d(ones,k,s,p,ceil,include)
                        sums=torch.nn.functional.avg_pool2d(ones,k,s,p,ceil,include,divisor_override=1)
                        self.assertEqual(tuple(expected.shape[-2:]),size)
                        np.testing.assert_allclose(divisor,(sums/expected)[0,0].numpy(),rtol=1e-12,atol=1e-12)
                        padded=torch.zeros(1,1,shape[0]+extra[0],shape[1]+extra[1])
                        actual=torch.nn.functional.conv2d(padded,torch.ones(1,1,*k),stride=s,padding=p)
                        self.assertEqual(tuple(actual.shape[-2:]),size)
        for kernel,stride,padding,divisor in (((0,2),(),0,None),((2,2),(),2,None),((2,2),(),0,0)):
            with self.assertRaises(ValueError):avg_pool2d_geometry((3,4),kernel,stride,padding,False,True,divisor)

    def test_dynamic_index_row_geometry_and_bounds(self):
        self.assertEqual(indexed_rows_geometry((2,3,4),(2,5,4),-2),(1,8,3,5))
        self.assertEqual(indexed_rows_geometry((128,4),(128,2),1),(1,128,4,2))
        for shape,indices,axis in (((2,3),(1,3),1),((129,3),(129,2),1),((3,),(0,),0),
                                    ((1025,),(1024,),0),((2,3),(2,3),2),((2,True),(2,2),1)):
            with self.assertRaises((ValueError,IndexError)):indexed_rows_geometry(shape,indices,axis)

    def test_softmax_tiles_preserve_rows_and_causal_positions(self):
        for rows,columns,query in ((1536,96,96),(1032,129,129),(257,257,257),(65537,1,65537),(256,4096,0),(16,65536,0)):
            plan=softmax_row_tiles(rows,columns);covered=[]
            for start,count in plan:
                self.assertLessEqual(count*columns,65536)
                covered.extend(range(start,start+count))
                if query:
                    actual=(np.arange(count)+start%query)%query
                    np.testing.assert_array_equal(actual,np.arange(start,start+count)%query)
            self.assertEqual(covered,list(range(rows)))
        self.assertEqual(softmax_row_tiles(65537,1),((0,65536),(65536,1)))
        for dimensions in ((0,1),(1,65537),(1025,1024),(True,3)):
            with self.assertRaises(ValueError):softmax_row_tiles(*dimensions)

    def test_index_column_tiles_keep_packet_and_work_bounds(self):
        for rows,width,count in ((7,3,4),(2048,35,32),(512,128,8),(65528,1,1),(3,65536,2)):
            tile=index_tile_width(rows,width,count)
            self.assertGreater(tile,0);self.assertLessEqual(tile,width)
            covered=0
            for start in range(0,width,tile):
                size=min(tile,width-start);covered+=size
                self.assertLessEqual(6+2*count+max(rows,count)*size,65536)
                self.assertLessEqual(rows*size*count,1048576)
            self.assertEqual(covered,width)
        self.assertEqual(index_tile_width(2048,35,32),16)
        for dimensions in ((0,3,2),(2,True,3),(65529,1,1),(1024,1,1025),(2,1048576,1),(1,1,32765)):
            with self.assertRaises(ValueError):index_tile_width(*dimensions)

    def test_reduction_shapes_and_errors(self):
        for shape in ((),(3,),(2,3,4)):
            host=torch.ones(shape,dtype=torch.bool)
            for dim in (None,(),-1,(0,),tuple(range(len(shape)))):
                for keep in (False,True):
                    _,result=reduction_geometry(shape,dim,keep)
                    self.assertEqual(result,tuple(torch.sum(host.float(),dim=dim,keepdim=keep).shape))
        with self.assertRaises(ValueError):reduction_geometry((2,3),(0,-2))
        with self.assertRaises(IndexError):reduction_geometry((2,3),2)

    def test_triangle_zero_sentinel_and_batch_indices(self):
        for shape in ((3,5),(2,4,3),(2,2,3,4)):
            source=np.arange(1,prod(shape)+1);padded=np.append(source,0)
            host=torch.from_numpy(source.reshape(shape))
            for diagonal in (-2**63,-2,0,1,2**63-1):
                for upper in (False,True):
                    indices=triangle_indices(shape,diagonal,upper=upper)
                    expected=(torch.triu if upper else torch.tril)(host,diagonal)
                    np.testing.assert_array_equal(padded[indices].reshape(shape),expected.numpy())

    def test_forward_and_input_gradient_tiles(self):
        torch.manual_seed(47)
        for shape, kernel, stride, padding, dilation, out_channels in (
                ((2, 3, 11, 13), (3, 2), (2, 3), (1, 2), (2, 1), 4),
                ((1, 2, 4, 5), (1, 1), (1, 1), (0, 0), (1, 1), 3),
                ((2, 1, 5, 7), (3, 3), (1, 2), (3, 3), (2, 2), 2)):
            with self.subTest(shape=shape, stride=stride, dilation=dilation):
                x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
                weight = torch.randn(out_channels, shape[1], *kernel, dtype=torch.float64)
                output = torch.nn.functional.conv2d(x, weight, stride=stride, padding=padding, dilation=dilation)
                spatial, (rows, width) = conv_geometry(shape, kernel, stride, padding, dilation)
                self.assertEqual(spatial, tuple(output.shape[2:]))
                host = x.detach().permute(0, 2, 3, 1).contiguous().numpy().reshape(-1)
                padded = np.append(host, 0)
                tiles = []
                for start in range(0, rows, 7):
                    indices = conv_window_indices(shape, kernel, stride, padding, dilation, start, min(7, rows-start))
                    self.assertLess(indices.max(), host.size)
                    tiles.append(torch.from_numpy(padded[indices]))
                columns = torch.cat(tiles)
                expected = torch.nn.functional.unfold(x.detach(), kernel, dilation=dilation,
                    padding=padding, stride=stride).permute(0, 2, 1).reshape(rows, width)
                torch.testing.assert_close(columns, expected, rtol=0, atol=0)
                gradient = torch.randn_like(output)
                output.backward(gradient)
                host_grad = gradient.permute(0, 2, 3, 1).contiguous().numpy().reshape(-1)
                padded_grad = np.append(host_grad, 0)
                tiles = []
                pixels = shape[0]*shape[2]*shape[3]
                matrix = weight.permute(2, 3, 0, 1).contiguous().reshape(-1, shape[1])
                for start in range(0, pixels, 7):
                    indices = conv_window_indices(shape, kernel, stride, padding, dilation,
                        start, min(7, pixels-start), output_channels=out_channels)
                    tiles.append(torch.from_numpy(padded_grad[indices]) @ matrix)
                actual = torch.cat(tiles).reshape(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2)
                torch.testing.assert_close(actual, x.grad, rtol=1e-12, atol=1e-12)

    def test_large_geometry_keeps_only_requested_rows(self):
        shape = (1, 16, 256, 256)
        spatial, (rows, width) = conv_geometry(shape, (3, 3), (1, 1), (1, 1), (1, 1))
        self.assertGreater(rows*width, 1048576)
        indices = conv_window_indices(shape, (3, 3), (1, 1), (1, 1), (1, 1), rows-17, 17)
        self.assertEqual(indices.shape, (17, 144))
        self.assertLess(indices.max(), prod(shape))
        self.assertEqual(spatial, (256, 256))

    def test_invalid_geometry_and_range(self):
        with self.assertRaises(ValueError):
            conv_geometry((1, 1, 2, 2), (5, 5), (1, 1), (0, 0), (1, 1))
        for start, count in ((-1, 1), (0, 0), (4, 1)):
            with self.assertRaises(ValueError):
                conv_window_indices((1, 1, 2, 2), (1, 1), (1, 1), (0, 0), (1, 1), start, count)


if __name__ == '__main__':
    unittest.main()

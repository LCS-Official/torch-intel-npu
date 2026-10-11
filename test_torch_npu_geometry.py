"""Offline index-geometry checks against CPU PyTorch convolution/autograd."""
import unittest
from math import prod
import numpy as np
import torch
from torch_npu_geometry import conv_geometry, conv_window_indices


class ConvGeometryTests(unittest.TestCase):
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

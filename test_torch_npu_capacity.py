"""Large storage DMA and row plans, without loading the NPU backend."""
import unittest
import importlib
import ast
from contextlib import nullcontext
from math import prod
from pathlib import Path
from types import SimpleNamespace
import torch
from unittest.mock import patch

npu4000 = importlib.import_module('npunlock.npu4000')
from torch_npu_geometry import softmax_row_tiles


class CapacityTests(unittest.TestCase):
    def test_scalar_square_and_identity_keep_values_and_storage_independent(self):
        source = ast.parse(Path(__file__).with_name('torch_npu_compat_extended.py').read_text(encoding='utf-8'))
        fn = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'math_forward')
        env = dict(torch=torch, INTEGERS=(torch.int64,torch.bool),
                   npu=SimpleNamespace(_validate_read=lambda x: None, _check_out=lambda *a: None,
                       _materialize=lambda x:x, _clone=lambda x:x.clone(),
                       _execute=lambda kind,x,y:torch.mul(x,y)),
                   copy=lambda out,value:out.copy_(value))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<scalar-powers>', 'exec'), env)
        for dtype in (torch.float32, torch.float16):
            x = torch.tensor([-float('inf'), -2., -0., 0., 1e-7, 2., float('inf'), float('nan')], dtype=dtype)
            for exponent in (1,2):
                actual = env['math_forward'](x, 'pow', exponent)
                torch.testing.assert_close(actual, torch.pow(x,exponent), rtol=0, atol=0, equal_nan=True)
                self.assertNotEqual(actual.data_ptr(), x.data_ptr())
                out = torch.empty_like(x)
                self.assertIs(env['math_forward'](x,'pow',exponent,out=out), out)

    def test_large_storage_cannot_bypass_selection_storage_bound(self):
        source = ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
        fn = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == '_selection_backward')
        env = dict(torch=torch, prod=prod, _lock=nullcontext(), MAX_TENSOR_ELEMENTS=16777216,
                   _validate_float_read=lambda x: None, _validate_read=lambda x: None,
                   _storage_limit=lambda dt:268435456)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<selection-bound>', 'exec'), env)
        gradient = torch.empty(1, device='meta')
        indices = torch.empty(1, dtype=torch.int64, device='meta')
        with self.assertRaisesRegex(ValueError, 'geometry/storage bounds'):
            env['_selection_backward'](gradient, indices, (268435457,), 1, 268435457)

    def test_large_dma_preserves_bytes_and_rejects_invalid_ranges(self):
        with patch.object(npu4000, 'compile_precision_graph', side_effect=lambda graph, **kw: graph):
            graph = npu4000.compile_slice(67108864, 67108861, 3, dtype='f32')
            self.assertEqual(graph.inputs[0].shape, (67108864, 4))
            self.assertEqual(graph.outputs[0].shape, (3, 4))
            graph = npu4000.compile_slice(npu4000.MAX_DMA_ELEMENTS,npu4000.MAX_DMA_ELEMENTS-3,3,dtype='f32')
            self.assertEqual(graph.inputs[0].shape,(npu4000.MAX_DMA_ELEMENTS,4))
            graph = npu4000.compile_matrix_concat((8388608, 8388609), 1, dtype='f32')
            self.assertEqual(graph.outputs[0].shape, (16777217, 1, 4))
            for elements, start, count in ((npu4000.MAX_DMA_ELEMENTS+1, 0, 1), (67108864, 67108864, 1), (10, 0, 65537)):
                with self.assertRaises(ValueError):
                    npu4000.compile_slice(elements, start, count, dtype='f32')

    def test_large_row_plan_keeps_complete_compute_tiles(self):
        plan = softmax_row_tiles(65537, 256, max_elements=67108864)
        self.assertEqual(plan[0], (0, 256))
        self.assertEqual(plan[-1], (65536, 1))
        self.assertEqual(sum(count for _, count in plan), 65537)
        self.assertTrue(all(count*256 <= 65536 for _, count in plan))
        for rows, columns, limit in ((65537, 1024, 67108864), (1, 65537, 67108864), (1, 1, True)):
            with self.assertRaises(ValueError):
                softmax_row_tiles(rows, columns, max_elements=limit)
        with self.assertRaises(ValueError):
            softmax_row_tiles(4097, 256)  # Existing SDPA callers retain their bound.


if __name__ == '__main__':
    unittest.main()

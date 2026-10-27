"""Offline guard against splitting an INT64 carry across SHAVE tasks."""
from dataclasses import dataclass
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import npunlock.npu4000 as kernels


@dataclass
class Program:
    graph_blob: bytes = b'carrier'
    patch_reports: tuple = ()

    @property
    def ir_provenance(self):
        return SimpleNamespace(vendor_id=0x8086, device_id=0x643e, compiler_version=(8, 3))


class Int64CarrierTests(unittest.TestCase):
    def test_complete_pair_requests_single_reference_schedule(self):
        evidence = {'ranges': [{}], 'invocations': [{'memrefs': [{'elements': 2}]*3}]}
        with patch.object(kernels, 'compile_precision_graph', return_value=Program()) as compiler, \
             patch.object(kernels, 'inspect', return_value=evidence), \
             patch.object(kernels, 'patch_binary_carrier', return_value=(b'patched', {})):
            result = kernels.compile_int64_counter(b'kernel')
        self.assertTrue(compiler.call_args.kwargs['reference_sw'])
        graph = compiler.call_args.args[0]
        self.assertEqual([x.shape for x in graph.inputs], [(2,), (2,)])
        self.assertTrue(all(x.dtype == 'f32' for x in graph.inputs))
        self.assertEqual(result.graph_blob, b'patched')
        self.assertEqual(json.loads(result.patch_reports[0])['scheduling'], 'single_complete_pair')

    def test_sharded_words_reject_before_patching(self):
        for evidence in (
            {'ranges': [{}]*8, 'invocations': [{'memrefs': [{'elements': 1}]*3}]*8},
            {'ranges': [{}], 'invocations': [{'memrefs': [{'elements': 1}]*3}]},
            {'ranges': [], 'invocations': []},
        ):
            with self.subTest(evidence=evidence), \
                 patch.object(kernels, 'compile_precision_graph', return_value=Program()), \
                 patch.object(kernels, 'inspect', return_value=evidence), \
                 patch.object(kernels, 'patch_binary_carrier') as patcher:
                with self.assertRaisesRegex(ValueError, 'both words'):
                    kernels.compile_int64_counter(b'kernel')
                patcher.assert_not_called()

    def test_kernel_type_rejects_before_compiling(self):
        with patch.object(kernels, 'compile_precision_graph') as compiler:
            with self.assertRaises(TypeError):
                kernels.compile_int64_counter('path.elf')
            compiler.assert_not_called()


if __name__ == '__main__':
    unittest.main()

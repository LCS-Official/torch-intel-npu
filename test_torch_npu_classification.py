"""Offline guards for complete-row scheduling and exact INT64 byte transport."""
import json
from unittest import TestCase, main
from unittest.mock import patch
from test_torch_npu_int64 import Program
import npunlock.npu4000 as kernels


def evidence(dims, elements, *, ranges=1, tasks=1):
    return {'ranges': [{}]*ranges,
            'invocations': [{'memrefs': [{'dims': dims, 'elements': elements}]*3}]*tasks}


class ClassificationCarrierTests(TestCase):
    def test_complete_rows_and_wide_single_row(self):
        for shape, dims in (((2, 3), (3, 2, 1, 1)), ((1, 4096), (1, 1, 4096, 1)),
                            ((17, 1), (1, 17, 1, 1))):
            with self.subTest(shape=shape), \
                 patch.object(kernels, 'compile_precision_graph', return_value=Program()) as compiler, \
                 patch.object(kernels, 'inspect', return_value=evidence(dims, shape[0]*shape[1])), \
                 patch.object(kernels, 'patch_binary_carrier', return_value=(b'patched', {})):
                result = kernels.compile_classification_rows(shape, b'kernel')
            self.assertTrue(compiler.call_args.kwargs['reference_sw'])
            self.assertEqual([x.shape for x in compiler.call_args.args[0].inputs], [shape, shape])
            self.assertEqual(json.loads(result.patch_reports[0])['scheduling'], 'single_complete_rows')

    def test_split_or_unrecognized_layout_rejects_before_patch(self):
        for data in (evidence((3, 1, 1, 1), 3, tasks=2), evidence((3, 2, 1, 1), 6, ranges=2),
                     evidence((1, 1, 3, 2), 6), evidence((3, 2, 1, 1), 5),
                     {'ranges': [{}], 'invocations': [{'memrefs': []}]}):
            with self.subTest(data=data), \
                 patch.object(kernels, 'compile_precision_graph', return_value=Program()), \
                 patch.object(kernels, 'inspect', return_value=data), \
                 patch.object(kernels, 'patch_binary_carrier') as patcher:
                with self.assertRaisesRegex(ValueError, 'splits a row'):
                    kernels.compile_classification_rows((2, 3), b'kernel')
                patcher.assert_not_called()

    def test_invalid_shape_and_type_reject_before_compiler(self):
        with patch.object(kernels, 'compile_precision_graph') as compiler:
            for shape, kernel in (((0, 3), b'x'), ((1, 65537), b'x'), ((2, True), b'x'),
                                  ((1, 2, 3), b'x'), ((2, 3), 'path.elf')):
                with self.subTest(shape=shape, kernel=kernel), self.assertRaises(ValueError):
                    kernels.compile_classification_rows(shape, kernel)
            compiler.assert_not_called()

    def test_pack_keeps_int64_words_as_bytes(self):
        with patch.object(kernels, 'compile_precision_graph', return_value=Program()) as compiler:
            kernels.compile_classification_pack(2, 3)
        self.assertTrue(compiler.call_args.kwargs['dma_only'])
        graph = compiler.call_args.args[0]
        self.assertEqual([x.shape for x in graph.inputs], [(2, 3, 4), (2, 2, 4), (2, 2, 4)])
        self.assertTrue(all(x.dtype == 'u8' for x in graph.inputs))
        with patch.object(kernels, 'compile_precision_graph') as compiler:
            for shape in ((0, 3), (True, 3), (1, 65533)):
                with self.assertRaises(ValueError):
                    kernels.compile_classification_pack(*shape)
            compiler.assert_not_called()


if __name__ == '__main__':
    main()

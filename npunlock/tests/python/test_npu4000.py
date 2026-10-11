"""Offline validation and malformed-input regression; no NPU/DLL required."""
import hashlib
from pathlib import Path
import struct
import sys
import unittest
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from npunlock.npu4000 import (Elf, compile_unary, inspect, patch_abs_carrier, patch_binary_carrier,
                             kernel_text, compile_reduce, compile_halfcode_slice, compile_broadcast, compile_permute,
                             compile_gather, compile_matrix_concat, compile_slice, compile_square_gemm,
                             compile_parallel_square_gemm, patch_parallel_square_gemm,
                             compile_four_tile_square_gemm, patch_four_tile_square_gemm,
                             compile_rect_gemm, rect_gemm_capacity,
                             MAX_TENSOR_ELEMENTS, _patch_carrier)

FIXTURE = ROOT / "tests/fixtures/npu4000"
PROFILE = dict(vendor_id=0x8086, device_id=0x643E, compiler_version=(8, 3))


class Npu4000Tests(unittest.TestCase):
    def test_eight_task_gemm_complete_replicas_and_dependencies(self):
        import json
        original = (FIXTURE/'gemm_eight/power-capacity-1152.blob').read_bytes()
        kernel = (FIXTURE/'gemm_eight/rect-f32.elf').read_bytes()
        class Driver:
            def compile_ir(self,xml,weights,**kwargs):
                return SimpleNamespace(graph_blob=original,**PROFILE)
        program = compile_rect_gemm((16,48,24),kernel,libraries=Driver(),replicas=8)
        report = json.loads(program.patch_reports[0])
        self.assertEqual(report['replicas'],8)
        self.assertEqual(program.graph.inputs[0].shape,(8,8,144))
        self.assertEqual(len(report['full_matrix_addresses']),8)
        self.assertEqual(len(report['dma_order_offsets']),12)
        start,extent = report['code_offset'],report['code_bytes']
        allowed={offset+3 for offset in report['dma_order_offsets']}
        self.assertTrue(all(a==b or start<=i<start+extent or i in allowed
            for i,(a,b) in enumerate(zip(original,program.graph_blob))))
        e=Elf(original); first=report['dma_order_offsets'][0]-16
        output=report['dma_order_offsets'][8]-16
        invocation=e.named['task.shave.invocation.0.0']['offset']
        for offset,fmt,value in ((first+24,'<I',4608),(first+48,'<Q',0x40219200),
                                (output+24,'<I',4608),(output+72,'<Q',0),
                                (invocation+40,'<Q',0)):
            corrupt=bytearray(original); struct.pack_into(fmt,corrupt,offset,value)
            class CorruptDriver:
                def compile_ir(self,xml,weights,**kwargs):
                    return SimpleNamespace(graph_blob=bytes(corrupt),**PROFILE)
            with self.subTest(offset=offset),self.assertRaises(ValueError):
                compile_rect_gemm((16,48,24),kernel,libraries=CorruptDriver(),replicas=8)
        for replicas in (0,2,16,True,8.0):
            with self.assertRaises(ValueError):
                compile_rect_gemm((16,48,24),kernel,libraries=Driver(),replicas=replicas)
        with self.assertRaises(ValueError):
            compile_rect_gemm((16,48,24),kernel,libraries=Driver(),replicas=4)
        manifest=json.loads((FIXTURE/'gemm_eight/manifest.json').read_text())
        for name,digest in manifest['source_sha256'].items():
            self.assertEqual(hashlib.sha256((FIXTURE/name).read_bytes()).hexdigest(),digest)
        self.assertEqual(hashlib.sha256(kernel).hexdigest(),manifest['elf_sha256'])

    def test_rect_gemm_packed_capacity_and_provenance(self):
        import json
        original = (FIXTURE/'gemm_rect/power-capacity-1152.blob').read_bytes()
        kernel = (FIXTURE/'gemm_rect/rect-f32.elf').read_bytes()

        class Driver:
            def compile_ir(self, xml, weights, **kwargs):
                return SimpleNamespace(graph_blob=original, **PROFILE)

        program = compile_rect_gemm((16,48,24),kernel,libraries=Driver())
        report = json.loads(program.patch_reports[0])
        self.assertEqual(program.graph.inputs[0].shape,(4,2,576))
        self.assertEqual(report['header'],[12345,16,48,24])
        self.assertEqual(report['capacity'],1152)
        self.assertEqual(rect_gemm_capacity((1,1,1)),64)
        self.assertEqual(rect_gemm_capacity((63,256,64)),16384)
        for shape in ((),(3,5),(3,5,7,9),(0,5,7),(3,257,7),(65,5,7),
                      (3,5,65),(True,5,7),(3,5.0,7),(64,256,64)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                compile_rect_gemm(shape,kernel,libraries=Driver())
        with self.assertRaises(ValueError):
            compile_rect_gemm((63,256,64),kernel,libraries=Driver())
        manifest = json.loads((FIXTURE/'gemm_rect/manifest.json').read_text())
        for name,digest in manifest['source_sha256'].items():
            self.assertEqual(hashlib.sha256((FIXTURE/name).read_bytes()).hexdigest(),digest)
        self.assertEqual(hashlib.sha256(kernel).hexdigest(),manifest['elf_sha256'])
        self.assertEqual(len(kernel_text(kernel)),manifest['text_bytes'])

    def test_four_tile_gemm_replica_dma_and_dependencies(self):
        import json
        original = (FIXTURE / 'gemm_parallel/power-four-tiles-8.blob').read_bytes()
        kernel = (FIXTURE / 'gemm_parallel/square-f32-four-tiles.elf').read_bytes()
        patched, report = patch_four_tile_square_gemm(original, kernel, size=8, **PROFILE)
        self.assertEqual(report['replicas'], 4)
        self.assertEqual(len(patched), len(original))
        self.assertEqual(len(report['dma_order_offsets']), 12)
        allowed = {offset+3 for offset in report['dma_order_offsets']}
        for offset in report['dma_order_offsets']:
            self.assertEqual(struct.unpack_from('<Q', patched, offset)[0],
                             struct.unpack_from('<Q', original, offset)[0] | (1 << 28))
        start, extent = report['code_offset'], report['code_bytes']
        self.assertTrue(all(a == b or start <= i < start+extent or i in allowed
                            for i, (a, b) in enumerate(zip(original, patched))))

        class Driver:
            def compile_ir(self, xml, weights, **kwargs):
                self.flags = kwargs['build_flags']
                return SimpleNamespace(graph_blob=original, **PROFILE)

        driver = Driver()
        program = compile_four_tile_square_gemm(8, kernel, libraries=driver)
        self.assertEqual(program.graph.inputs[0].shape, (4, 2, 32))
        self.assertEqual(program.graph.outputs[0].shape, (4, 2, 32))
        self.assertNotIn('NPU_TILES', driver.flags)
        elf = Elf(original)
        # Full uploads, both input completion barriers, whole output extents,
        # and the SHAVE wait must survive malformed-descriptor rejection.
        first = report['dma_order_offsets'][0]-16
        last = max(offset for offset in report['dma_order_offsets'][:8])-16
        output = report['dma_order_offsets'][8]-16
        invocation = elf.named['task.shave.invocation.0.0']['offset']
        for offset, fmt, value in ((first+24, '<I', 128), (first+40, '<Q', 4),
                                   (first+48, '<Q', 0x40218100), (last+64, '<Q', 0),
                                   (output+24, '<I', 128), (output+72, '<Q', 0),
                                   (invocation+40, '<Q', 0)):
            corrupt = bytearray(original)
            struct.pack_into(fmt, corrupt, offset, value)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                patch_four_tile_square_gemm(bytes(corrupt), kernel, size=8, **PROFILE)
        for size in (0, 4, 17, 128, True, 8.0):
            with self.assertRaises(ValueError):
                compile_four_tile_square_gemm(size, kernel, libraries=driver)
        with self.assertRaises(ValueError):
            patch_four_tile_square_gemm(original, kernel, size=16, **PROFILE)
        with self.assertRaises(ValueError):
            patch_four_tile_square_gemm((FIXTURE/'gemm/power-4-split.blob').read_bytes(),
                                        kernel, size=8, **PROFILE)
        manifest = json.loads((FIXTURE/'gemm_parallel/manifest.json').read_text())
        for name, record in manifest['kernels'].items():
            data = (FIXTURE/f'gemm_parallel/{name}.elf').read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record['elf_sha256'])
            self.assertEqual(len(kernel_text(data)), record['text_bytes'])

    def test_parallel_gemm_proves_full_dma_and_dependencies(self):
        import json
        original = (FIXTURE / 'gemm/power-4-split.blob').read_bytes()
        kernel = (FIXTURE / 'gemm_parallel/square-f32-lanes4.elf').read_bytes()
        patched, report = patch_parallel_square_gemm(original, kernel, size=4, **PROFILE)
        self.assertEqual(report['row_partitions'], [2, 2])
        self.assertEqual(len(patched), len(original))
        cfg_offset = report['dma_order_offset']
        self.assertEqual(struct.unpack_from('<Q', patched, cfg_offset)[0],
                         struct.unpack_from('<Q', original, cfg_offset)[0] | (1 << 28))
        start, size = report['code_offset'], report['code_bytes']
        changed = {i for i, (a, b) in enumerate(zip(original, patched)) if a != b}
        self.assertTrue(all(start <= i < start+size or i == cfg_offset+3 for i in changed))

        class Driver:
            def compile_ir(self, xml, weights, **kwargs):
                self.flags = kwargs['build_flags']
                return SimpleNamespace(graph_blob=original, **PROFILE)

        driver = Driver()
        program = compile_parallel_square_gemm(4, kernel, libraries=driver)
        self.assertIn('NPU_TILES="1"', driver.flags)
        self.assertNotIn('ReferenceSW', driver.flags)
        self.assertEqual(program.graph.outputs[0].shape, (4, 4))
        elf = Elf(original)
        dma_base = cfg_offset-16
        invocation = elf.named['task.shave.invocation.0.0']['offset']
        output = elf.named['task.dma.0.1']['offset']
        # Partial upload, wrong destination, missing ready/wait dependencies,
        # precision conversion and incomplete output must all fail closed.
        for offset, fmt, value in ((dma_base+24, '<I', 32), (dma_base+48, '<Q', 0x40218000),
                                   (dma_base+64, '<Q', 0), (invocation+40, '<Q', 1),
                                   (dma_base+16, '<Q', 0x10010affff000), (output+24, '<I', 32),
                                   (output+72, '<Q', 1)):
            corrupt = bytearray(original)
            struct.pack_into(fmt, corrupt, offset, value)
            with self.assertRaises(ValueError):
                patch_parallel_square_gemm(bytes(corrupt), kernel, size=4, **PROFILE)
        with self.assertRaises(ValueError):
            patch_parallel_square_gemm((FIXTURE/'gemm/power-4-reference.blob').read_bytes(), kernel, size=4, **PROFILE)
        manifest = json.loads((FIXTURE/'gemm_parallel/manifest.json').read_text())
        for name, digest in manifest['source_sha256'].items():
            self.assertEqual(hashlib.sha256((FIXTURE/name).read_bytes()).hexdigest(), digest)

    def test_square_gemm_requires_complete_fp32_carrier(self):
        import json
        kernel = (FIXTURE / "gemm/square-f32.elf").read_bytes()
        original = (FIXTURE / "gemm/power-4-reference.blob").read_bytes()

        class Driver:
            def __init__(self, blob):
                self.blob = blob

            def compile_ir(self, xml, weights, **kwargs):
                self.assert_flags = kwargs['build_flags']
                return SimpleNamespace(graph_blob=self.blob, **PROFILE)

        driver = Driver(original)
        program = compile_square_gemm(4, kernel, libraries=driver)
        report = json.loads(program.patch_reports[0])
        self.assertIn('NPU_COMPILATION_MODE="ReferenceSW"', driver.assert_flags)
        self.assertEqual(report['invocations'], 1)
        self.assertEqual(report['input_count'], 2)
        self.assertEqual(program.graph.outputs[0].shape, (4, 4))
        start, size = report['code_offset'], report['code_bytes']
        self.assertEqual(program.graph_blob[:start], original[:start])
        self.assertEqual(program.graph_blob[start + size:], original[start + size:])
        self.assertEqual(len(program.graph_blob), len(original))
        for size in (0, 1, 65, True, 4.0):
            with self.assertRaisesRegex(ValueError, '2..64'):
                compile_square_gemm(size, kernel, libraries=driver)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            compile_square_gemm(4, kernel,
                libraries=Driver((FIXTURE / "gemm/power-4-split.blob").read_bytes()))
        with self.assertRaisesRegex(ValueError, 'complete row-major'):
            compile_square_gemm(5, kernel, libraries=driver)
        corrupt = bytearray(original)
        corrupt[start] ^= 1
        with self.assertRaisesRegex(ValueError, 'verified'):
            compile_square_gemm(4, kernel, libraries=Driver(bytes(corrupt)))

    def test_compute_tiles_from_large_dma_storage(self):
        class Driver:
            def __init__(self, blob):
                self.blob = blob

            def compile_ir(self, xml, weights, **kwargs):
                return SimpleNamespace(graph_blob=self.blob, **PROFILE)

        for dtype, width in (("f16", 2), ("f32", 4)):
            dma = Driver((FIXTURE / "resident/halfcode-slice-17.blob").read_bytes())
            program = compile_slice(MAX_TENSOR_ELEMENTS, MAX_TENSOR_ELEMENTS-65536, 65536,
                                    dtype=dtype, libraries=dma)
            self.assertEqual(program.graph.inputs[0].shape, (MAX_TENSOR_ELEMENTS, width))
            self.assertEqual(program.graph.outputs[0].shape, (65536, width))
            for elements, start, count in ((0, 0, 1), (MAX_TENSOR_ELEMENTS+1, 0, 1),
                                          (MAX_TENSOR_ELEMENTS, -1, 1), (65537, 0, 65537),
                                          (65537, 65537, 1), (65537, True, 1)):
                with self.assertRaises(ValueError):
                    compile_slice(elements, start, count, dtype=dtype, libraries=dma)
            with self.assertRaisesRegex(ValueError, "DMA-only"):
                compile_slice(65537, 65536, 1, dtype=dtype,
                              libraries=Driver((FIXTURE / "abs-1x17.blob").read_bytes()))

    def test_matrix_concat_argument_and_size_bounds(self):
        class Driver:
            def __init__(self, blob):
                self.blob = blob

            def compile_ir(self, xml, weights, **kwargs):
                return SimpleNamespace(graph_blob=self.blob, **PROFILE)

        for dtype, width in (("f16", 2), ("f32", 4)):
            driver = Driver((FIXTURE / f"resident/matrix-concat-{dtype}.blob").read_bytes())
            program = compile_matrix_concat((3, 2, 1), 5, dtype=dtype, libraries=driver)
            self.assertEqual([t.shape for t in program.graph.inputs], [(3, 5, width), (2, 5, width), (1, 5, width)])
            self.assertEqual(program.graph.outputs[0].shape, (6, 5, width))
            with self.assertRaisesRegex(ValueError, "DMA-only"):
                compile_matrix_concat((3, 2, 1), 5, dtype=dtype,
                    libraries=Driver((FIXTURE / "abs-1x17.blob").read_bytes()))
            for rows, columns in (((1,), 5), ((1,)*64, 1), ((1, 0), 5), ((1, 1), 0),
                                  ((MAX_TENSOR_ELEMENTS//2, MAX_TENSOR_ELEMENTS//2+1), 1), ((1, True), 5)):
                with self.assertRaises(ValueError):
                    compile_matrix_concat(rows, columns, dtype=dtype, libraries=driver)

    def test_gather_transport_and_index_bounds(self):
        import numpy as np

        class Driver:
            def __init__(self, blob):
                self.blob = blob

            def compile_ir(self, xml, weights, **kwargs):
                return SimpleNamespace(graph_blob=self.blob, **PROFILE)

        indices = np.arange(144, dtype=np.int64).reshape(16, 9) % 36
        indices[0, 0] = -1
        for dtype, width in (("f16", 2), ("f32", 4)):
            blob = (FIXTURE / f"resident/gather-{dtype}-36.blob").read_bytes()
            driver = Driver(blob)
            program = compile_gather(36, indices, dtype=dtype, libraries=driver)
            self.assertEqual(program.graph.inputs[0].dtype, "u8")
            self.assertEqual(program.graph.inputs[0].shape, (36, width))
            self.assertEqual(program.graph.outputs[0].shape, (16, 9, width))
            with self.assertRaisesRegex(ValueError, "DMA-only"):
                compile_gather(36, indices, dtype=dtype,
                               libraries=Driver((FIXTURE / "abs-1x17.blob").read_bytes()))
            for elements, invalid in ((0, indices), (65537, indices), (36, [-2]), (36, [36]),
                                      (36, [0.5]), (36, []), (36, np.zeros((1, 1, 1, 1), dtype=np.int64)),
                                      (36, np.zeros(65537, dtype=np.int64))):
                with self.assertRaises(ValueError):
                    compile_gather(elements, invalid, dtype=dtype, libraries=driver)

    def test_permute_byte_transport_and_rejections(self):
        class Driver:
            def __init__(self, blob):
                self.blob = blob

            def compile_ir(self, xml, weights, **kwargs):
                return SimpleNamespace(graph_blob=self.blob, **PROFILE)

        for dtype, width in (("f16", 2), ("f32", 4)):
            blob = (FIXTURE / f"resident/permute-{dtype}-3x5.blob").read_bytes()
            program = compile_permute((3, 5), (1, 0), dtype=dtype, libraries=Driver(blob))
            self.assertEqual(program.graph.inputs[0].dtype, "u8")
            self.assertEqual(program.graph.inputs[0].shape, (3, 5, width))
            self.assertEqual(program.graph.outputs[0].shape, (5, 3, width))
            with self.assertRaisesRegex(ValueError, "DMA-only"):
                compile_permute((3, 5), (1, 0), dtype=dtype,
                    libraries=Driver((FIXTURE / "abs-1x17.blob").read_bytes()))
            for shape, axes in (((3, 5), (0, 0)), ((3, 5), (1,)), ((3, 5), (1, True)),
                                ((MAX_TENSOR_ELEMENTS+1,), (0,)), ((1, 1, 1, 1, 1), (0, 1, 2, 3, 4))):
                with self.assertRaises(ValueError):
                    compile_permute(shape, axes, dtype=dtype, libraries=Driver(blob))
        with self.assertRaises(ValueError):
            compile_permute((3, 5), (1, 0), dtype="u8", libraries=Driver(blob))

    def test_broadcast_transport_layout_and_rejections(self):
        class Driver:
            def __init__(self, blob):
                self.blob = blob

            def compile_ir(self, xml, weights, **kwargs):
                return SimpleNamespace(graph_blob=self.blob, **PROFILE)

        for dtype in ("f16", "f32"):
            blob = (FIXTURE / f"resident/broadcast-{dtype}-1x5-to-3x5.blob").read_bytes()
            program = compile_broadcast((1, 5), (3, 5), dtype=dtype, libraries=Driver(blob))
            # Logical FP32 is transported as pairs of 16-bit storage words.
            self.assertEqual(program.graph.inputs[0].dtype, "f16")
            self.assertEqual(program.graph.inputs[0].shape, (1, 5, 2) if dtype == "f32" else (1, 5))
            self.assertEqual(program.graph.outputs[0].shape, (3, 5, 2) if dtype == "f32" else (3, 5))
            with self.assertRaisesRegex(ValueError, "DMA-only"):
                compile_broadcast((1, 5), (3, 5), dtype=dtype,
                    libraries=Driver((FIXTURE / "abs-1x17.blob").read_bytes()))
            for shape, target in (((2, 5), (3, 5)), ((1,), (MAX_TENSOR_ELEMENTS+1,)), ((1,), (1, 1, 1, 1, 1))):
                with self.assertRaises(ValueError):
                    compile_broadcast(shape, target, dtype=dtype, libraries=Driver(blob))

    def test_resident_kernel_and_reduction_contracts(self):
        import json
        directory = FIXTURE / "resident"
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, digest in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest(), digest)
        for kind, record in manifest['kernels'].items():
            data = (directory / f'{kind}.elf').read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record['elf_sha256'])
            self.assertLessEqual(len(kernel_text(data)), 21216 if kind.startswith('relu_grad') else 5040)
        for kind, carrier in (("sum", "ReduceSum"), ("mean", "ReduceMean")):
            kernel = (directory / f"{kind}.elf").read_bytes()
            self.assertEqual(hashlib.sha256(kernel).hexdigest(), manifest["kernels"][kind]["elf_sha256"])
            self.assertLessEqual(len(kernel_text(kernel)), 5040)  # singleton Abs carrier also fits
            original = (directory / f"{carrier.lower()}-3x5.blob").read_bytes()
            args = dict(dtype="f32", carrier=carrier, input_count=1, reduction_count=5, **PROFILE)
            patched, report = _patch_carrier(original, kernel, **args)
            start, size = report["code_offset"], report["code_bytes"]
            self.assertEqual(patched[:start], original[:start])
            self.assertEqual(patched[start + size:], original[start + size:])
            self.assertEqual(len(patched), len(original))
            with self.assertRaisesRegex(ValueError, "partial reduction"):
                _patch_carrier(original, kernel, **{**args, "reduction_count": 10})
            corrupt = bytearray(original)
            corrupt[start] ^= 1
            with self.assertRaisesRegex(ValueError, "verified"):
                _patch_carrier(bytes(corrupt), kernel, **args)

            class Driver:
                def compile_ir(self, xml, weights, **kwargs):
                    return SimpleNamespace(graph_blob=original, **PROFILE)

            program = compile_reduce((3, 5), (1,), kernel, kind=kind, libraries=Driver())
            self.assertEqual(program.graph.outputs[0].shape, (3, 1))
            for shape, axes in (((3, 5), (1, 1)), ((3, 5), (2,)), ((65537,), (0,))):
                with self.assertRaises(ValueError):
                    compile_reduce(shape, axes, kernel, kind=kind, libraries=Driver())

    def test_halfcode_transport_rejects_compute_tasks(self):
        original = (FIXTURE / "resident/halfcode-slice-17.blob").read_bytes()

        class Driver:
            def __init__(self, blob):
                self.blob = blob

            def compile_ir(self, xml, weights, **kwargs):
                return SimpleNamespace(graph_blob=self.blob, **PROFILE)

        program = compile_halfcode_slice(17, libraries=Driver(original))
        self.assertEqual(program.graph.inputs[0].shape, (17, 2))
        self.assertEqual(program.graph.outputs[0].shape, (17, 1))
        with self.assertRaisesRegex(ValueError, "DMA-only"):
            compile_halfcode_slice(17, libraries=Driver((FIXTURE / "abs-1x17.blob").read_bytes()))
        for size in (0, 65537):
            with self.assertRaises(ValueError):
                compile_halfcode_slice(size, libraries=Driver(original))

    def test_fp32_precision_preservation_and_wrong_dtype(self):
        original = (FIXTURE / "abs-f32-1x17.blob").read_bytes()
        kernel = (FIXTURE / "precision/add1-f32.elf").read_bytes()
        patched, report = patch_abs_carrier(original, kernel, dtype="f32", **PROFILE)
        self.assertEqual({c["dtype"] for c in report["contracts"]}, {"f32"})
        self.assertEqual({c["span_bytes"] for c in report["contracts"]}, {4, 64})
        self.assertEqual(len(patched), len(original))
        with self.assertRaisesRegex(ValueError, "lowering"):
            patch_abs_carrier((FIXTURE / "abs-1x17.blob").read_bytes(), kernel, dtype="f32", **PROFILE)

        class Driver:
            def compile_ir(self, xml, weights, **kwargs):
                self.xml, self.kwargs = xml, kwargs
                return SimpleNamespace(graph_blob=original, **PROFILE)

        driver = Driver()
        program = compile_unary((1, 17), kernel, dtype="f32", libraries=driver)
        self.assertIn('EXECUTION_MODE_HINT="ACCURACY"', driver.kwargs["build_flags"])
        attribute = ET.fromstring(driver.xml).find("./layers/layer[@name='y']/rt_info/attribute")
        self.assertEqual(attribute.get("name"), "DisablePrecisionConversion")
        self.assertEqual(program.graph.inputs[0].dtype, "f32")

    def test_binary_three_records_and_output_alias_rejection(self):
        original = (FIXTURE / "power-f16-1x17.blob").read_bytes()
        kernel = (FIXTURE / "precision/mul-f16.elf").read_bytes()
        patched, report = patch_binary_carrier(original, kernel, dtype="f16", **PROFILE)
        self.assertEqual(report["input_count"], 2)
        self.assertEqual(len(patched), len(original))
        elf = Elf(original)
        refs = inspect(original, memref_count=3)["invocations"][0]["memrefs"]
        corrupt = bytearray(original)
        struct.pack_into("<Q", corrupt, elf.named["shave.params"]["offset"] + refs[-1]["offset"],
                         int(refs[1]["unrelocated_data"], 16))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            patch_binary_carrier(bytes(corrupt), kernel, dtype="f16", **PROFILE)

    def test_precision_kernel_manifest(self):
        import json
        root = FIXTURE / "precision"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(hashlib.sha256((FIXTURE / "arithmetic.c").read_bytes()).hexdigest(), manifest["source_sha256"])
        for stem, record in manifest["kernels"].items():
            data = (root / f"{stem}.elf").read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record["elf_sha256"])
            limit = 21216 if stem.split("-")[0] in ("add", "sub", "mul", "div") else 5040
            self.assertLessEqual(len(kernel_text(data)), limit)

    def test_unary_patch_and_rejections(self):
        original = (FIXTURE / "abs-1x17.blob").read_bytes()
        kernel = (FIXTURE / "add1.4000xx.elf").read_bytes()
        patched, report = patch_abs_carrier(original, kernel, **PROFILE)
        start, size = report["code_offset"], report["code_bytes"]
        self.assertEqual(len(patched), len(original))
        self.assertEqual(patched[:start], original[:start])
        self.assertEqual(patched[start + size:], original[start + size:])
        self.assertEqual(hashlib.sha256(patched).hexdigest(), report["patched_sha256"])
        self.assertEqual({c["elements"] for c in report["contracts"]}, {1, 16})
        self.assertEqual({c["tile"] for c in report["contracts"]}, {0, 1, 2, 3})
        self.assertEqual(report["changed_bytes"], 143)
        with self.assertRaisesRegex(ValueError, "profile"):
            patch_abs_carrier(original, kernel, **{**PROFILE, "device_id": 0x7D1D})
        with self.assertRaisesRegex(ValueError, "profile"):
            patch_abs_carrier(original, kernel, **{**PROFILE, "compiler_version": (99, 0)})
        for blob in (original[:63], original[:-1]):
            with self.assertRaises(ValueError):
                patch_abs_carrier(blob, kernel, **PROFILE)
        with self.assertRaises(ValueError):
            patch_abs_carrier(original, kernel[:51], **PROFILE)
        for shape in [(65537,), (1, 1, 1, 1, 1), ()]:
            with self.assertRaisesRegex(ValueError, "dimensions"):
                compile_unary(shape, kernel)
        elf = Elf(original)
        evidence = inspect(original)
        first = evidence["invocations"][0]["unary_memrefs"][0]
        dims = elf.relocation("shave.params", first["offset"] + 16)
        params = elf.named["shave.params"]["offset"]
        mutations = []
        corrupt = bytearray(original)
        corrupt[elf.named["shave.text"]["offset"]] ^= 1
        mutations.append((corrupt, "verified"))
        corrupt = bytearray(original)
        for ref in evidence["invocations"][0]["unary_memrefs"]:
            strides = elf.relocation("shave.params", ref["offset"] + 20)
            struct.pack_into("<Q", corrupt, params + strides["symbol_value"] + strides["addend"], 32)
        mutations.append((corrupt, "Non-dense"))
        corrupt = bytearray(original)
        struct.pack_into("<i", corrupt, params + dims["symbol_value"] + dims["addend"], -1)
        mutations.append((corrupt, "dimension"))
        corrupt = bytearray(original)
        struct.pack_into("<Q", corrupt, params + first["offset"], 0x40818000)  # tile 2 address in tile 0 task
        mutations.append((corrupt, "workspace"))
        corrupt = bytearray(original)
        section = elf.named["shave.params"]
        shoff = struct.unpack_from("<Q", original, 40)[0]
        struct.pack_into("<Q", corrupt, shoff + section["index"] * 64 + 32, len(original) + 1)
        mutations.append((corrupt, "bounds"))
        for corrupted, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                patch_abs_carrier(bytes(corrupted), kernel, **PROFILE)


if __name__ == "__main__":
    unittest.main()

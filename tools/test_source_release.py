"""Offline checks of source export exclusions, overwrite guards and integrity."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import export_source_release as exporter


class ExportTests(unittest.TestCase):
    def test_export_rejects_fine_grained_github_credentials_before_writing(self):
        with tempfile.TemporaryDirectory(prefix='source-release-secret-test-', dir=exporter.ROOT) as name:
            root=Path(name);(root/'code.py').write_text('github_pat_'+'a'*70,encoding='utf-8')
            target=root/'source-release-refused'
            with patch.object(exporter,'selection',return_value=(['code.py'],[])):
                with self.assertRaisesRegex(ValueError,'Possible credential'):exporter.export(root,target)
            self.assertFalse(target.exists())

    def test_patch_preserves_missing_final_newline(self):
        with tempfile.TemporaryDirectory(prefix='source-release-patch-test-', dir=exporter.ROOT) as name:
            root = Path(name).resolve()
            self.assertEqual(root.parent, exporter.ROOT.resolve())
            (root / 'npunlock').mkdir()
            (root / 'npunlock/new.txt').write_bytes(b'without final newline')
            with patch.object(exporter, 'git', side_effect=[b'base\n', b'']):
                record, text = exporter.upstream_record(root, ['npunlock/new.txt'], [])
            self.assertEqual(record['files'][0]['status'], 'added')
            self.assertIn('new file mode 100644\n--- /dev/null\n', text)
            self.assertTrue(text.endswith('+without final newline\n\\ No newline at end of file\n'))

    def test_fixture_and_workflow_policy(self):
        for name in ('src/graphinfer/session_win32.c', 'python/npunlock/npu4000.py',
                     'tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf', 'tests/fixtures/npu4000/rmsnorm/manifest.json'):
            self.assertTrue(exporter.fork_file_allowed(name), name)
        for name in ('.github/workflows/release.yml', 'tests/fixtures/npu4000/abs-1x17.blob',
                     'tests/fixtures/npu3720/add1-fp16.elf', 'build/npunlock.dll', 'python/npunlock/__pycache__/ir.pyc'):
            self.assertFalse(exporter.fork_file_allowed(name), name)

    def test_output_boundary_and_existing_directory(self):
        root = exporter.ROOT
        self.assertEqual(exporter.output_path(root, root / 'source-release-unit-new'), root / 'source-release-unit-new')
        for output in (root, root / 'npunlock', root.parent / 'source-release-outside', root / 'models/source-release-bad'):
            with self.assertRaises(ValueError):
                exporter.output_path(root, output)
        with tempfile.TemporaryDirectory(prefix='source-release-existing-', dir=root) as name:
            self.assertEqual(Path(name).resolve().parent, root.resolve())
            with self.assertRaises(FileExistsError):
                exporter.output_path(root, Path(name))

    def test_root_release_files_and_upstream_workflow_exclusion(self):
        with tempfile.TemporaryDirectory(prefix='source-release-selection-', dir=exporter.ROOT) as name:
            root = Path(name)
            workflow = root / '.github/workflows/offline.yml'
            workflow.parent.mkdir(parents=True)
            workflow.write_text('name: offline\n')
            with patch.object(exporter, 'git', return_value=b'.github/workflows/upstream.yml\0'):
                names, excluded = exporter.selection(root)
            self.assertIn('.github/workflows/offline.yml', names)
            self.assertIn('.gitattributes', names)
            self.assertNotIn('npunlock/.github/workflows/upstream.yml', names)
            self.assertIn('.github/workflows/upstream.yml', excluded)
            self.assertNotIn('npunlock-probe/pip-freeze.txt', names)
            self.assertNotIn('npunlock-probe/conda-explicit.txt', names)

    def test_corruption_and_unexpected_file(self):
        with tempfile.TemporaryDirectory(prefix='source-release-test-', dir=exporter.ROOT) as name:
            root = Path(name).resolve()
            self.assertEqual(root.parent, exporter.ROOT.resolve())
            source = root / 'code.py'
            source.write_bytes(b'pass\n')
            manifest = {'files': [dict(path='code.py', bytes=5, sha256=hashlib.sha256(b'pass\n').hexdigest())]}
            (root / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest))
            self.assertEqual(exporter.verify(root), (1, 5))
            (root / '.git').mkdir()
            (root / '.git/HEAD').write_text('ref: refs/heads/main\n')
            self.assertEqual(exporter.verify(root), (1, 5))
            source.write_bytes(b'fail\n')
            with self.assertRaisesRegex(ValueError, 'Checksum differs'):
                exporter.verify(root)
            source.write_bytes(b'pass\n')
            (root / 'unexpected').touch()
            with self.assertRaisesRegex(ValueError, 'File set changed'):
                exporter.verify(root)


if __name__ == '__main__':
    unittest.main()

"""Both documented native build routes must resolve their source records."""
import ast
from pathlib import Path
import unittest


class ReleaseProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parent
        tree=ast.parse((root/'npunlock-probe/research/w8_guarded_audit.py').read_text(encoding='utf-8'))
        helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='session_source_path')
        scope={'Path':Path};exec(compile(ast.Module(body=[helper],type_ignores=[]),'<provenance>','exec'),scope)
        cls.resolve=staticmethod(scope['session_source_path']);cls.root=root

    def test_gcc_and_msvc_sources_resolve_to_the_same_file(self):
        expected=self.root/'npunlock/src/graphinfer/session_win32.c'
        for name in ('src/graphinfer/session_win32.c','npunlock/src/graphinfer/session_win32.c',
                     'npunlock\\src\\graphinfer\\session_win32.c'):
            self.assertEqual(self.resolve(self.root,name),expected)

    def test_source_records_cannot_escape_the_export(self):
        for name in ('../outside.c','npunlock/../../outside.c',str(self.root/'outside.c')):
            with self.assertRaises(ValueError):self.resolve(self.root,name)


if __name__=='__main__':unittest.main()

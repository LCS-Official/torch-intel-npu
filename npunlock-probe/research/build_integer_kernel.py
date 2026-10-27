"""Build the bounded INT64 scalar-add kernel without rebuilding FP kernels."""
import hashlib
import json
from probe_movi_targets import PROJECT, stage


def main():
    root = PROJECT / 'npunlock/tests/fixtures/npu4000'
    source = (root/'resident_integer.c').read_bytes()
    assembly, ok = stage('compile', 'int64-add', 'moviCompile64.dll', 1,
        ['moviCompile.dll', '-cc1', '-triple', 'shave', '-target-cpu', '4000xx',
         '-S', '-O2', '-x', 'c', 'entry.c', '-o', '-'], source)
    assert ok, 'INT64 compile failed'
    obj, ok = stage('assemble', 'int64-add', 'moviAsm64.dll', 2,
                    ['moviAsm.dll', '--cv', '4000xx', '--noSPrefixing'], assembly)
    assert ok, 'INT64 assemble failed'
    elf, ok = stage('link', 'int64-add', 'moviLLD64.dll', 3,
        ['moviLLD.dll', '-flavor', 'gnu', '-EL', '-e', 'controlled_act',
         '-z', 'max-page-size=0x10', '--gc-sections'],
        (obj, (PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
    assert ok, 'INT64 link failed'
    dest = root/'resident'
    (dest/'int64-add.elf').write_bytes(elf)
    (dest/'int64-add.s').write_bytes(assembly)
    manifest = json.loads((dest/'manifest.json').read_text())
    manifest['source_sha256']['resident_integer.c'] = hashlib.sha256(source).hexdigest()
    manifest['kernels']['int64-add'] = {'elf_sha256': hashlib.sha256(elf).hexdigest(),
        'elf_bytes': len(elf), 'assembly_sha256': hashlib.sha256(assembly).hexdigest()}
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

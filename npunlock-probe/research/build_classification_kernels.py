"""Build four classification row kernels using the existing isolated Movi worker."""
import hashlib
import json
from probe_movi_targets import PROJECT, stage


def main():
    root = PROJECT/'npunlock/tests/fixtures/npu4000'
    helpers = (root/'arithmetic.c').read_bytes().split(b'void controlled_act(unsigned p) {', 1)[0]
    body = (root/'resident_classification.c').read_bytes()
    dest = root/'resident'
    manifest = json.loads((dest/'manifest.json').read_text())
    for mode, name in enumerate(('log-softmax', 'log-softmax-backward', 'nll-packet', 'nll-backward')):
        source = f'#define CLASS_MODE {mode}\n'.encode() + helpers + body
        asm, ok = stage('compile', name, 'moviCompile64.dll', 1,
            ['moviCompile.dll', '-cc1', '-triple', 'shave', '-target-cpu', '4000xx',
             '-S', '-O2', '-ffp-contract=off', '-x', 'c', 'entry.c', '-o', '-'], source)
        assert ok, (name, 'compile')
        obj, ok = stage('assemble', name, 'moviAsm64.dll', 2,
            ['moviAsm.dll', '--cv', '4000xx', '--noSPrefixing'], asm)
        assert ok, (name, 'assemble')
        elf, ok = stage('link', name, 'moviLLD64.dll', 3,
            ['moviLLD.dll', '-flavor', 'gnu', '-EL', '-e', 'controlled_act', '-z', 'max-page-size=0x10', '--gc-sections'],
            (obj, (PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
        assert ok, (name, 'link')
        (dest/f'{name}.elf').write_bytes(elf)
        (dest/f'{name}.s').write_bytes(asm)
        manifest['kernels'][name] = {'elf_sha256': hashlib.sha256(elf).hexdigest(), 'elf_bytes': len(elf),
                                    'assembly_sha256': hashlib.sha256(asm).hexdigest()}
    manifest['source_sha256']['resident_classification.c'] = hashlib.sha256(body).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

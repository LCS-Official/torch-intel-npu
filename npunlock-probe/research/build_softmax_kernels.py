"""Build bounded FP32 row softmax and backward with the isolated Movi worker."""
import hashlib
import json
from probe_movi_targets import PROJECT, stage


def main():
    root = PROJECT/'npunlock/tests/fixtures/npu4000'
    helpers = b''.join((root/name).read_bytes().split(b'void controlled_act(unsigned p) {', 1)[0]
                       for name in ('arithmetic.c', 'resident_classification.c'))
    body = root/'resident_softmax.c'
    dest = root/'resident'
    manifest = json.loads((dest/'manifest.json').read_text())
    for mode, name in enumerate(('softmax', 'softmax-backward')):
        source = f'#define SOFTMAX_BACKWARD {mode}\n'.encode()+helpers+body.read_bytes()
        assembly, ok = stage('compile', name, 'moviCompile64.dll', 1,
            ['moviCompile.dll', '-cc1', '-triple', 'shave', '-target-cpu', '4000xx',
             '-S', '-O2', '-ffp-contract=off', '-x', 'c', 'entry.c', '-o', '-'], source)
        if not ok:
            raise RuntimeError(f'Compile failed: {name}')
        obj, ok = stage('assemble', name, 'moviAsm64.dll', 2,
            ['moviAsm.dll', '--cv', '4000xx', '--noSPrefixing'], assembly)
        if not ok:
            raise RuntimeError(f'Assembly failed: {name}')
        elf, ok = stage('link', name, 'moviLLD64.dll', 3,
            ['moviLLD.dll', '-flavor', 'gnu', '-EL', '-e', 'controlled_act',
             '-z', 'max-page-size=0x10', '--gc-sections'],
            (obj, (PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
        if not ok:
            raise RuntimeError(f'Link failed: {name}')
        (dest/f'{name}.elf').write_bytes(elf)
        (dest/f'{name}.s').write_bytes(assembly)
        manifest['kernels'][name] = {'elf_sha256': hashlib.sha256(elf).hexdigest(), 'elf_bytes': len(elf),
                                   'assembly_sha256': hashlib.sha256(assembly).hexdigest()}
    manifest['source_sha256'][body.name] = hashlib.sha256(body.read_bytes()).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

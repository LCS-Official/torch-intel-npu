"""Build original erf/tanh GELU forward/backward kernels with the Movi worker."""
import hashlib
import json
from probe_movi_targets import PROJECT, stage


def main():
    root = PROJECT/'npunlock/tests/fixtures/npu4000'
    helpers = b''.join((root/name).read_bytes().split(b'void controlled_act(unsigned p) {', 1)[0]
                       for name in ('arithmetic.c', 'resident_classification.c'))
    # Share IEEE edge helpers inside each ELF instead of expanding them per polynomial term.
    for name in (b'add32', b'mul32', b'div32', b'encode64'):
        helpers = helpers.replace(b'static inline float '+name, b'static __attribute__((noinline)) float '+name)
    body = (root/'resident_gelu.c').read_bytes()
    dest = root/'resident'
    manifest = json.loads((dest/'manifest.json').read_text())
    for backward in (0, 1):
        for tanh, mode in enumerate(('none', 'tanh')):
            name = ('gelu-backward-' if backward else 'gelu-')+mode+'-f32'
            source = f'#define GELU_BACKWARD {backward}\n#define GELU_TANH {tanh}\n'.encode()+helpers+body
            asm, ok = stage('compile', name, 'moviCompile64.dll', 1,
                ['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx',
                 '-S','-Os','-ffp-contract=off','-x','c','entry.c','-o','-'], source)
            if not ok:
                raise RuntimeError(f'Compile failed: {name}')
            obj, ok = stage('assemble', name, 'moviAsm64.dll', 2,
                ['moviAsm.dll','--cv','4000xx','--noSPrefixing'], asm)
            if not ok:
                raise RuntimeError(f'Assembly failed: {name}')
            elf, ok = stage('link', name, 'moviLLD64.dll', 3,
                ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
                (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
            if not ok:
                raise RuntimeError(f'Link failed: {name}')
            (dest/f'{name}.elf').write_bytes(elf)
            (dest/f'{name}.s').write_bytes(asm)
            manifest['kernels'][name] = {'elf_sha256':hashlib.sha256(elf).hexdigest(),'elf_bytes':len(elf),
                                        'assembly_sha256':hashlib.sha256(asm).hexdigest()}
    manifest['source_sha256']['resident_gelu.c'] = hashlib.sha256(body).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()

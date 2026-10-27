"""Build ordered half-input LayerNorm statistics with an integer fused operation."""
import hashlib
import json
from probe_movi_targets import PROJECT, stage


def main():
    root = PROJECT/'npunlock/tests/fixtures/npu4000'
    arithmetic = (root/'arithmetic.c').read_bytes().split(b'void controlled_act(unsigned p) {', 1)[0]
    # CPU-compatible moments require correctly rounded ratios, not a hardware reciprocal approximation.
    branch = b'if (ea && eb && (int)ea - (int)eb > -125) return a / b;'
    assert arithmetic.count(branch) == 1
    arithmetic = arithmetic.replace(branch, b'/* exact integer division below */')
    for name in (b'add32', b'mul32', b'div32', b'encode64'):
        arithmetic = arithmetic.replace(b'static inline float '+name, b'static __attribute__((noinline)) float '+name)
    sqrt = (root/'resident_activation.c').read_bytes().split(b'void controlled_act(unsigned p) {', 1)[0]
    body = (root/'resident_layernorm.c').read_bytes()
    dest = root/'resident'
    manifest = json.loads((dest/'manifest.json').read_text())
    for mode, name in ((0, 'layernorm-stats-f32'), (1, 'layernorm-fma-test-f32')):
        source = f'#define LN_FMA_TEST {mode}\n#define RSQRT 1\n'.encode()+arithmetic+sqrt+body
        asm, ok = stage('compile', name, 'moviCompile64.dll', 1,
            ['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx',
             '-S','-Os','-ffp-contract=off','-x','c','entry.c','-o','-'], source)
        if not ok: raise RuntimeError(f'Compile failed: {name}')
        obj, ok = stage('assemble', name, 'moviAsm64.dll', 2,
            ['moviAsm.dll','--cv','4000xx','--noSPrefixing'], asm)
        if not ok: raise RuntimeError(f'Assembly failed: {name}')
        elf, ok = stage('link', name, 'moviLLD64.dll', 3,
            ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
            (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
        if not ok: raise RuntimeError(f'Link failed: {name}')
        (dest/f'{name}.elf').write_bytes(elf)
        (dest/f'{name}.s').write_bytes(asm)
        manifest['kernels'][name] = {'elf_sha256':hashlib.sha256(elf).hexdigest(),'elf_bytes':len(elf),
                                    'assembly_sha256':hashlib.sha256(asm).hexdigest()}
    manifest['source_sha256']['resident_layernorm.c'] = hashlib.sha256(body).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()

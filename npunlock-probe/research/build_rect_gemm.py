"""Build one bounded rectangular kernel with existing FP32 helpers and tools."""
import hashlib
import json
from probe_movi_targets import stage, PROJECT
from npunlock.npu4000 import kernel_text


def build(eight):
    root = PROJECT/'npunlock/tests/fixtures/npu4000'
    dest = root/('gemm_eight' if eight else 'gemm_rect')
    dest.mkdir(exist_ok=True)
    names = ('arithmetic.c', 'square_gemm_parallel.c', 'rect_gemm.c')
    helpers = (root/names[0]).read_bytes().split(b'void controlled_act(unsigned p) {', 1)[0]
    for name in ('add32', 'mul32'):
        helpers = helpers.replace(f'static inline float {name}('.encode(),
            f'static __attribute__((noinline)) float {name}('.encode())
    source = (f'#define GEMM_FAST 1\n#define GEMM_EIGHT {int(eight)}\n'.encode()+helpers+
              (root/names[1]).read_bytes().split(b'void controlled_act(unsigned p) {', 1)[0]+
              (root/names[2]).read_bytes())
    asm, ok = stage('compile', 'rect-f32', 'moviCompile64.dll', 1,
        ['moviCompile.dll', '-cc1', '-triple', 'shave', '-target-cpu', '4000xx',
         '-S', '-O2', '-ffp-contract=off', '-x', 'c', 'entry.c', '-o', '-'], source)
    assert ok, 'compile failed'
    obj, ok = stage('assemble', 'rect-f32', 'moviAsm64.dll', 2,
        ['moviAsm.dll', '--cv', '4000xx', '--noSPrefixing'], asm)
    assert ok, 'assembly failed'
    elf, ok = stage('link', 'rect-f32', 'moviLLD64.dll', 3,
        ['moviLLD.dll', '-flavor', 'gnu', '-EL', '-e', 'controlled_act',
         '-z', 'max-page-size=0x10', '--gc-sections'],
        (obj, (PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
    assert ok and len(kernel_text(elf)) <= 21216, 'link/window failed'
    (dest/'rect-f32.elf').write_bytes(elf)
    (dest/'rect-f32.s').write_bytes(asm)
    manifest = dict(source_sha256={n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
        generated_source_sha256=hashlib.sha256(source).hexdigest(), elf_sha256=hashlib.sha256(elf).hexdigest(),
        assembly_sha256=hashlib.sha256(asm).hexdigest(), text_bytes=len(kernel_text(elf)), target='4000xx',shave_tasks=8 if eight else 4)
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(manifest, flush=True)


if __name__ == '__main__':
    for eight in (False,True): build(eight)

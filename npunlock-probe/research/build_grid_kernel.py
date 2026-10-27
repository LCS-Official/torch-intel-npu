"""Build the source-backed grid sampling stencil carrier kernel."""
import hashlib
import json
from probe_movi_targets import PROJECT, stage


def main():
    root=PROJECT/'npunlock/tests/fixtures/npu4000';dest=root/'resident';name='grid-stencil-f32'
    helpers=(root/'arithmetic.c').read_bytes().split(b'void controlled_act(unsigned p) {',1)[0]
    body=(root/'resident_grid.c').read_bytes()
    helpers=helpers.replace(b'static inline ',b'static __attribute__((always_inline)) inline ')
    body=body.replace(b'static unsigned ',b'static __attribute__((always_inline)) inline unsigned ').replace(b'static float ',b'static __attribute__((always_inline)) inline float ')
    asm,ok=stage('compile',name,'moviCompile64.dll',1,
        ['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2','-fno-jump-tables','-ffp-contract=off','-x','c','entry.c','-o','-'],
        b'#define TYPE unsigned short\n#define DTYPE 2\n'+helpers+body)
    if not ok:raise RuntimeError('sort compile failed')
    if any(pattern in asm for pattern in (b'Function call',b'.LJTI',b'.rodata')):
        raise RuntimeError('Grid kernel contains unsafe absolute references: '+str([line for line in asm.splitlines() if any(p in line for p in (b'Function call',b'.LJTI',b'.rodata'))]))
    obj,ok=stage('assemble',name,'moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm)
    if not ok:raise RuntimeError('sort assemble failed')
    elf,ok=stage('link',name,'moviLLD64.dll',3,
        ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
        (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
    if not ok:raise RuntimeError('sort link failed')
    (dest/f'{name}.elf').write_bytes(elf);(dest/f'{name}.s').write_bytes(asm)
    manifest=json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
    manifest['kernels'][name]=dict(elf_bytes=len(elf),elf_sha256=hashlib.sha256(elf).hexdigest(),assembly_sha256=hashlib.sha256(asm).hexdigest())
    manifest['source_sha256']['resident_grid.c']=hashlib.sha256((root/'resident_grid.c').read_bytes()).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()

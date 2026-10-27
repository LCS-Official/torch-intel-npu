"""Build complete typed and dynamic-index packets for the Torch device backend."""
import hashlib
import json
from probe_movi_targets import PROJECT, stage


def main():
    root=PROJECT/'npunlock/tests/fixtures/npu4000'; dest=root/'resident'
    helpers=(root/'arithmetic.c').read_bytes().split(b'void controlled_act(unsigned p) {',1)[0]
    manifest=json.loads((dest/'manifest.json').read_text())
    math_helpers=b''.join((root/name).read_bytes().split(b'void controlled_act(unsigned p) {',1)[0]
                         for name in ('arithmetic.c','resident_classification.c','resident_gelu.c','resident_unary_math.c'))
    math_helpers=math_helpers.replace(b'static inline ',b'static inline __attribute__((always_inline)) ').replace(b'static float ',b'static inline __attribute__((always_inline)) float ')
    groups=(('resident_compat.c',('typed-cast','typed-math','typed-reduce'),'COMPAT_MODE',helpers),
            ('resident_dynamic_index.c',('dynamic-index','compact-index','index-offset'),'INDEX_MODE',b''),
            ('resident_scalar_math.c',tuple('math-'+name for name in ('sin','cos','pow','tan','floor','ceil','trunc','round','expm1','log1p')),'SCALAR_MODE',math_helpers))
    for filename,names,define,prefix in groups:
        body=(root/filename).read_bytes()
        for mode,name in enumerate(names):
            source=f'#define {define} {mode}\n#define TYPE unsigned short\n#define DTYPE 2\n'.encode()+prefix+body
            asm,ok=stage('compile',name,'moviCompile64.dll',1,
                ['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2','-ffp-contract=off','-fno-jump-tables','-x','c','entry.c','-o','-'],source)
            if not ok:raise RuntimeError((name,'compile'))
            if b'Function call' in asm or b'.LJTI' in asm or b'.rodata' in asm:raise RuntimeError((name,'unsafe absolute address'))
            obj,ok=stage('assemble',name,'moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm)
            if not ok:raise RuntimeError((name,'assemble'))
            elf,ok=stage('link',name,'moviLLD64.dll',3,
                ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
                (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
            if not ok:raise RuntimeError((name,'link'))
            (dest/f'{name}.elf').write_bytes(elf);(dest/f'{name}.s').write_bytes(asm)
            manifest['kernels'][name]=dict(elf_bytes=len(elf),elf_sha256=hashlib.sha256(elf).hexdigest(),assembly_sha256=hashlib.sha256(asm).hexdigest())
        manifest['source_sha256'][filename]=hashlib.sha256(body).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()

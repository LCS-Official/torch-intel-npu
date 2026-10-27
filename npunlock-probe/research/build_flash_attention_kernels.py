"""Build bounded FlashAttention-style forward/recomputation SHAVE packets."""
import hashlib
import json
from probe_movi_targets import PROJECT,stage


def main():
    root=PROJECT/'npunlock/tests/fixtures/npu4000';dest=root/'resident'
    helpers=b''.join((root/name).read_bytes().split(b'void controlled_act(unsigned p) {',1)[0]
                     for name in ('arithmetic.c','resident_classification.c'))
    body=(root/'resident_flash_attention.c').read_bytes()
    results={}
    for backward in (0,1):
        name='flash-backward-f32' if backward else 'flash-step-f32'
        source=f'#define TYPE unsigned short\n#define DTYPE 2\n#define FLASH_BACKWARD {backward}\n'.encode()+helpers+body
        # Keep IEEE edge helpers once in the linked code window, rather than inlining at every dot/PV site.
        for fn in (b'add32',b'mul32',b'div32'):
            source=source.replace(b'static inline float '+fn+b'(',b'static __attribute__((noinline)) float '+fn+b'(')
        source=source.replace(b'static float exp_negative(',b'static __attribute__((noinline)) float exp_negative(')
        asm,ok=stage('compile',name,'moviCompile64.dll',1,
            ['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2','-ffp-contract=off','-x','c','entry.c','-o','-'],source)
        if not ok:raise RuntimeError(name+' compile failed')
        obj,ok=stage('assemble',name,'moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm)
        if not ok:raise RuntimeError(name+' assembly failed')
        elf,ok=stage('link',name,'moviLLD64.dll',3,
            ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
            (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
        if not ok:raise RuntimeError(name+' link failed')
        results[name]=(elf,asm)
    manifest=json.loads((dest/'manifest.json').read_text())
    for name,(elf,asm) in results.items():
        (dest/f'{name}.elf').write_bytes(elf);(dest/f'{name}.s').write_bytes(asm)
        manifest['kernels'][name]=dict(elf_bytes=len(elf),elf_sha256=hashlib.sha256(elf).hexdigest(),assembly_sha256=hashlib.sha256(asm).hexdigest())
    manifest['source_sha256']['resident_flash_attention.c']=hashlib.sha256(body).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()

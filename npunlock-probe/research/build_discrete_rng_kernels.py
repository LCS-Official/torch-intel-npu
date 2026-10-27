"""Build INT64 integer RNG and categorical sampling with the existing ABI."""
import hashlib
import json
from probe_movi_targets import PROJECT, stage


def main():
    root=PROJECT/'npunlock/tests/fixtures/npu4000'; dest=root/'resident'
    helpers=(root/'arithmetic.c').read_bytes().split(b'void controlled_act(unsigned p) {',1)[0]
    for name in (b'add32',b'mul32',b'div32'):
        helpers=helpers.replace(b'static inline float '+name,b'static __attribute__((noinline)) float '+name)
    body=(root/'resident_discrete_rng.c').read_bytes()
    manifest=json.loads((dest/'manifest.json').read_text())
    for integer,name in ((1,'integer-rng-packet'),(0,'multinomial-packet')):
        source=f'#define INTEGER_RNG {integer}\n'.encode()+helpers+body
        asm,ok=stage('compile',name,'moviCompile64.dll',1,
            ['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-Os','-ffp-contract=off',
             '-fno-jump-tables','-x','c','entry.c','-o','-'],source)
        if not ok:raise RuntimeError(f'Compile failed: {name}')
        if b'.LJTI' in asm or b'.rodata' in asm:raise RuntimeError('Patched kernels cannot use absolute-address tables')
        obj,ok=stage('assemble',name,'moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm)
        if not ok:raise RuntimeError(f'Assembly failed: {name}')
        elf,ok=stage('link',name,'moviLLD64.dll',3,
            ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
            (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
        if not ok:raise RuntimeError(f'Link failed: {name}')
        (dest/f'{name}.elf').write_bytes(elf);(dest/f'{name}.s').write_bytes(asm)
        manifest['kernels'][name]=dict(elf_sha256=hashlib.sha256(elf).hexdigest(),elf_bytes=len(elf),
                                      assembly_sha256=hashlib.sha256(asm).hexdigest())
    manifest['source_sha256']['resident_discrete_rng.c']=hashlib.sha256(body).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()

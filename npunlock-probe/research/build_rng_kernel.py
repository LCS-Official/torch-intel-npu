"""Build the on-device Philox uniform/normal kernel with the validated carrier ABI."""
import hashlib
import json
from probe_movi_targets import PROJECT, stage


def main():
    root=PROJECT/'npunlock/tests/fixtures/npu4000'
    source=(root/'resident_rng.c').read_bytes()
    name='rng-packet'
    asm,ok=stage('compile',name,'moviCompile64.dll',1,
        ['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2','-ffp-contract=off',
         '-fno-jump-tables','-x','c','entry.c','-o','-'],source)
    assert ok,'RNG compile failed'
    assert b'Function call' not in asm,'RNG helpers must remain inline in the patched window'
    assert b'.LJTI' not in asm and b'.rodata' not in asm,'RNG must not use absolute-address tables'
    obj,ok=stage('assemble',name,'moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm)
    assert ok,'RNG assembly failed'
    elf,ok=stage('link',name,'moviLLD64.dll',3,
        ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
        (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
    assert ok,'RNG link failed'
    dest=root/'resident'
    (dest/f'{name}.elf').write_bytes(elf)
    (dest/f'{name}.s').write_bytes(asm)
    manifest=json.loads((dest/'manifest.json').read_text())
    manifest['source_sha256']['resident_rng.c']=hashlib.sha256(source).hexdigest()
    manifest['kernels'][name]=dict(elf_sha256=hashlib.sha256(elf).hexdigest(),elf_bytes=len(elf),
                                  assembly_sha256=hashlib.sha256(asm).hexdigest())
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()

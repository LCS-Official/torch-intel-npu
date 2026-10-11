"""Build fused FP32 RMSNorm with existing exact half/FP32/sqrt helpers."""
import hashlib
import json
from probe_movi_targets import stage, PROJECT
from npunlock.npu4000 import kernel_text


def main():
    root=PROJECT/'npunlock/tests/fixtures/npu4000';dest=root/'rmsnorm';dest.mkdir(exist_ok=True)
    names=('arithmetic.c','resident_activation.c','rmsnorm.c')
    prefix=b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n'
    source=prefix+b'\n'.join((root/name).read_bytes().split(b'void controlled_act(unsigned p) {',1)[0]
        if name!='rmsnorm.c' else (root/name).read_bytes() for name in names)
    asm,ok=stage('compile','rmsnorm-f16','moviCompile64.dll',1,
        ['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2',
         '-ffp-contract=off','-x','c','entry.c','-o','-'],source)
    assert ok,'RMSNorm compilation failed'
    obj,ok=stage('assemble','rmsnorm-f16','moviAsm64.dll',2,
        ['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm)
    assert ok,'RMSNorm assembly failed'
    elf,ok=stage('link','rmsnorm-f16','moviLLD64.dll',3,
        ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
        (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
    assert ok and len(kernel_text(elf))<=21216,'RMSNorm link/window failed'
    (dest/'rmsnorm-f16.elf').write_bytes(elf);(dest/'rmsnorm-f16.s').write_bytes(asm)
    manifest=dict(target='4000xx',width=576,epsilon=1e-5,storage='f16',arithmetic='f32',
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
        generated_source_sha256=hashlib.sha256(source).hexdigest(),elf_sha256=hashlib.sha256(elf).hexdigest(),
        assembly_sha256=hashlib.sha256(asm).hexdigest(),text_bytes=len(kernel_text(elf)))
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest,flush=True)


if __name__=='__main__':main()

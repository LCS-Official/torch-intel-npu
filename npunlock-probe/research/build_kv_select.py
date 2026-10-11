"""Build the validated FP16 KV/mask Select using the existing SHAVE tools."""
import hashlib
import json
from probe_movi_targets import stage, PROJECT
from npunlock.npu4000 import kernel_text


def main():
    root=PROJECT/'npunlock/tests/fixtures/npu4000';dest=root/'kv_select';dest.mkdir(exist_ok=True)
    source=(root/'kv_select.c').read_bytes()
    args=['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2',
          '-ffp-contract=off','-x','c','entry.c','-o','-']
    asm,ok=stage('compile','kv-select-f16','moviCompile64.dll',1,args,source);assert ok,'KV Select compilation failed'
    obj,ok=stage('assemble','kv-select-f16','moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm)
    assert ok,'KV Select assembly failed'
    elf,ok=stage('link','kv-select-f16','moviLLD64.dll',3,
        ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
        (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
    assert ok and len(kernel_text(elf))<=23088,'KV Select link/window failed'
    (dest/'kv-select-f16.elf').write_bytes(elf);(dest/'kv-select-f16.s').write_bytes(asm)
    manifest=dict(target='4000xx',storage='f16',operation='kv_row_and_mask_select',
        source_sha256={'kv_select.c':hashlib.sha256(source).hexdigest()},elf_sha256=hashlib.sha256(elf).hexdigest(),
        assembly_sha256=hashlib.sha256(asm).hexdigest(),text_bytes=len(kernel_text(elf)),compile_arguments=args)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(manifest,flush=True)


if __name__=='__main__':main()

"""Isolated reduction vectorization and copy control; preserve FP32 sum order."""
import hashlib
import json
from pathlib import Path
from probe_movi_targets import stage,PROJECT
from npunlock.npu4000 import kernel_text

OUT=PROJECT/'npunlock-probe/research/results/rmsnorm-small'
FIXTURES=PROJECT/'npunlock/tests/fixtures/npu4000'


def main():
    original=(OUT/'before/rmsnorm.c').read_text()
    start=original.index('        for (unsigned j = 0; j < 576; ++j) {')
    end=original.index('        float variance =',start)
    vector='''        for (unsigned j=0;j<576;j+=4) {
            Float4 values=decode4(x+base+j),squares=values*values;
'''
    for lane in range(4):
        vector+=f'''            {{
                float square=squares[{lane}],next=total+square;
                float error=total>=square ? (total-next)+square : (square-next)+total;
                correction=correction+error;total=next;
            }}
'''
    vector+='        }\n'
    variants=dict(reduce4=original[:start]+vector+original[end:])
    start=original.index('    for (unsigned base =')
    variants['copy']=original[:start]+'''    for (unsigned i=0;i<n;++i) out[i]=x[i];
}
'''
    helpers=b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n'+b'\n'.join(
        (FIXTURES/name).read_bytes().split(b'void controlled_act(unsigned p) {',1)[0]
        for name in ['arithmetic.c','resident_activation.c'])
    manifest=dict(status='building',baseline_manifest=json.loads((OUT/'before/manifest.json').read_text()),kernels={})
    for name,code in variants.items():
        source=helpers+code.encode();asm,ok=stage('small-'+name+'-compile',name,'moviCompile64.dll',1,
            ['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2','-ffp-contract=off','-x','c','entry.c','-o','-'],source)
        assert ok
        obj,ok=stage('small-'+name+'-assemble',name,'moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm);assert ok
        elf,ok=stage('small-'+name+'-link',name,'moviLLD64.dll',3,
            ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
            (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
        assert ok and len(kernel_text(elf))<=21216
        for suffix,data in [('.c',code.encode()),('.s',asm),('.elf',elf)]:(OUT/(name+suffix)).write_bytes(data)
        manifest['kernels'][name]=dict(source_sha256=hashlib.sha256(code.encode()).hexdigest(),generated_source_sha256=hashlib.sha256(source).hexdigest(),
            assembly_sha256=hashlib.sha256(asm).hexdigest(),elf_sha256=hashlib.sha256(elf).hexdigest(),text_bytes=len(kernel_text(elf)))
    manifest.update(status='built',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'build-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print({k:v['text_bytes'] for k,v in manifest['kernels'].items()},flush=True)


if __name__=='__main__':main()

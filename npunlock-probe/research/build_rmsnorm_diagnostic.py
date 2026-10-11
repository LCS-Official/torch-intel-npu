"""Build isolated RMSNorm ablations; never overwrite production source/ELF."""
import hashlib
import json
from probe_movi_targets import stage, PROJECT
from npunlock.npu4000 import kernel_text

OUT=PROJECT/'npunlock-probe/research/results/rmsnorm-diagnostic'
FIXTURES=PROJECT/'npunlock/tests/fixtures/npu4000'


def main():
    OUT.mkdir(exist_ok=True)
    original=(FIXTURES/'rmsnorm.c').read_text()
    start=original.index('    for (unsigned base = 0; base < n; base += 576) {')
    mean_end=original.index('        float inverse =',start)
    reduction=original[start:mean_end]
    bodies={
        'fill':'    for (unsigned i=0;i<n;++i) out[i]=0;\n',
        'copy':'    for (unsigned i=0;i<n;++i) out[i]=x[i];\n',
        'convert':'    for (unsigned i=0;i<n;++i) out[i]=round_half(load_half(x,i));\n',
        'affine':'    for (unsigned i=0;i<n;++i) out[i]=round_half(load_half(x,i)*load_half(gamma,i));\n',
        # Finite gamma makes this one. Keep the factor dynamic without adding
        # a per-element volatile load; normal multiplication remains observable.
        'epilogue':'''    for (unsigned base=0;base<n;base+=576) {
        float inverse=1.0f+load_half(gamma,base)*0.0f;
        for (unsigned j=0;j<576;++j) {
            unsigned short rounded=round_half(load_half(x,base+j)*inverse);
            out[base+j]=round_half(load_half(&rounded,0)*load_half(gamma,base+j));
        }
    }
''',
        'mean':reduction+'''        unsigned short value=round_half(variance);
        for (unsigned j=0;j<576;++j) out[base+j]=value;
    }
''',
        'inverse':reduction+'''        unsigned short value=round_half(1.0f/sqrt32(variance));
        for (unsigned j=0;j<576;++j) out[base+j]=value;
    }
'''}
    manifest=dict(status='building',sources_sha256={name:hashlib.sha256((FIXTURES/name).read_bytes()).hexdigest()
        for name in ('arithmetic.c','resident_activation.c','rmsnorm.c')},kernels={})
    prefix=b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n'
    helpers=b'\n'.join((FIXTURES/name).read_bytes().split(b'void controlled_act(unsigned p) {',1)[0]
        for name in ('arithmetic.c','resident_activation.c'))
    for name,body in bodies.items():
        source=(original[:start]+body+'}\n').encode()
        generated=prefix+helpers+b'\n'+source
        target='rmsnorm-diag-'+name
        asm,ok=stage('compile',target,'moviCompile64.dll',1,
            ['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2',
             '-ffp-contract=off','-x','c','entry.c','-o','-'],generated)
        assert ok,(name,'compile')
        obj,ok=stage('assemble',target,'moviAsm64.dll',2,
            ['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm)
        assert ok,(name,'assemble')
        elf,ok=stage('link',target,'moviLLD64.dll',3,
            ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
            (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
        assert ok and len(kernel_text(elf))<=21216,(name,'link/window')
        for suffix,data in (('.c',source),('.s',asm),('.elf',elf)):(OUT/(name+suffix)).write_bytes(data)
        manifest['kernels'][name]=dict(source_sha256=hashlib.sha256(source).hexdigest(),
            generated_source_sha256=hashlib.sha256(generated).hexdigest(),
            elf_sha256=hashlib.sha256(elf).hexdigest(),assembly_sha256=hashlib.sha256(asm).hexdigest(),
            text_bytes=len(kernel_text(elf)))
    full=(FIXTURES/'rmsnorm/rmsnorm-f16.elf').read_bytes()
    (OUT/'full.elf').write_bytes(full)
    manifest['kernels']['full']=dict(elf_sha256=hashlib.sha256(full).hexdigest(),text_bytes=len(kernel_text(full)))
    manifest['status']='built'
    (OUT/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print({name:data['text_bytes'] for name,data in manifest['kernels'].items()},flush=True)


if __name__=='__main__':main()

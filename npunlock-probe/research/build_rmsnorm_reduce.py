"""Build an isolated SIMD FP16 codec; keep ordered FP32 reduction unchanged."""
import hashlib
import json
from pathlib import Path
import shutil

from probe_movi_targets import stage,PROJECT
from npunlock.npu4000 import kernel_text

OUT=PROJECT/'npunlock-probe/research/results/rmsnorm-reduce'
FIXTURES=PROJECT/'npunlock/tests/fixtures/npu4000'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    before=OUT/'before';before.mkdir(exist_ok=True)
    if not (before/'rmsnorm.c').exists():
        for source in (FIXTURES/'rmsnorm.c',*(FIXTURES/'rmsnorm').glob('*')):shutil.copy2(source,before/source.name)
    original=(before/'rmsnorm.c').read_text(encoding='utf-8')
    start=original.index('static inline Float4 decode4(');end=original.index('static inline HalfBits4 encode4(',start)
    decode='''/* SIMD adaptation of FP16's IEEE conversion algorithm (MIT).
 * Copyright (c) 2017 Facebook Inc., 2017 Georgia Institute of Technology,
 * Copyright 2019 Google LLC. Full terms: rmsnorm/FP16_LICENSE.
 * https://github.com/Maratyszcza/FP16/blob/782eea126dc5c755827be751a099eb01826175cf/include/fp16/fp16.h
 * Integer extension avoids this toolchain's broken half->float convertvector.
 * The subnormal path uses normal FP32 values, so hardware FTZ cannot erase it. */
static inline Float4 decode_bits4(HalfBits4 raw) {
    UInt4 h=__builtin_convertvector(raw,UInt4),two=h<<17;
    union { UInt4 u;Float4 f; } normal,small,result;
    normal.u=(two>>4)+0x70000000u;
    normal.f=normal.f*0x1.0p-112f;
    small.u=(two>>17)|0x3f000000u;
    small.f=small.f-0.5f;
    UInt4 mask=(UInt4)(two<0x08000000u);
    result.u=((h&0x8000u)<<16)|(small.u&mask)|(normal.u&~mask);
    return result.f;
}
static inline Float4 decode4(const unsigned short *p) {
    return decode_bits4(*(const HalfBits4*)p);
}
'''
    codec=original[:start]+decode+original[end:]
    for lane in range(4):
        old=f'magnitude[{lane}]>=0x38800000u && magnitude[{lane}]<=0x477fe000u'
        new=f'(magnitude[{lane}]==0 || (magnitude[{lane}]>=0x38800000u && magnitude[{lane}]<=0x477fe000u))'
        assert codec.count(old)==1;codec=codec.replace(old,new)
    codec=codec.replace('return __builtin_convertvector(rounded,HalfBits4);',
        'rounded=(rounded&(UInt4)(magnitude!=0))|((bits.u>>16)&0x8000u);\n        return __builtin_convertvector(rounded,HalfBits4);')
    codec=codec.replace('decode4((const unsigned short*)&rounded)','decode_bits4(rounded)')
    roundtrip=codec[:codec.index('        float total =')]+'''        for (unsigned j=0;j<576;j+=4) {
            unsigned i=base+j;*(HalfBits4*)(out+i)=encode4(decode4(x+i));
        }
    }
}
'''
    prefix=b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n'
    helpers=b'\n'.join((FIXTURES/name).read_bytes().split(b'void controlled_act(unsigned p) {',1)[0]
        for name in ('arithmetic.c','resident_activation.c'))
    manifest=dict(status='building',builder_sha256=sha(Path(__file__).read_bytes()),baseline_manifest=json.loads((before/'manifest.json').read_text()),
        helper_sha256={name:sha((FIXTURES/name).read_bytes()) for name in ('arithmetic.c','resident_activation.c')},kernels={})
    shutil.copy2(before/'rmsnorm-f16.elf',OUT/'baseline.elf')
    manifest['kernels']['baseline']=dict(source_sha256=sha(original.encode()),elf_sha256=sha((OUT/'baseline.elf').read_bytes()),text_bytes=len(kernel_text((OUT/'baseline.elf').read_bytes())))
    for name,body in (('codec',codec),('roundtrip',roundtrip)):
        source=body.encode()
        generated=prefix+helpers+b'\n'+source
        target='rmsnorm-reduce-'+name
        arguments=['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2',
            '-ffp-contract=off','-x','c','entry.c','-o','-']
        asm,ok=stage('compile',target,'moviCompile64.dll',1,arguments,generated);assert ok,(name,'compile')
        obj,ok=stage('assemble',target,'moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm);assert ok,(name,'assemble')
        elf,ok=stage('link',target,'moviLLD64.dll',3,
            ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
            (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
        assert ok and len(kernel_text(elf))<=21216,(name,'link/window')
        for suffix,data in (('.c',source),('.s',asm),('.elf',elf)):(OUT/(name+suffix)).write_bytes(data)
        manifest['kernels'][name]=dict(source_sha256=sha(source),generated_source_sha256=sha(generated),
            elf_sha256=sha(elf),assembly_sha256=sha(asm),text_bytes=len(kernel_text(elf)),compile_arguments=arguments)
    manifest['status']='built';(OUT/'build.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print({name:record['text_bytes'] for name,record in manifest['kernels'].items()},flush=True)


if __name__=='__main__':main()

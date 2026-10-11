"""Build isolated four-element output loops; leave production artifacts alone."""
import hashlib
import json
from probe_movi_targets import stage, PROJECT
from npunlock.npu4000 import kernel_text

OUT=PROJECT/'npunlock-probe/research/results/rmsnorm-output'
FIXTURES=PROJECT/'npunlock/tests/fixtures/npu4000'

VECTOR_HELPERS='''
typedef float Float4 __attribute__((ext_vector_type(4)));
typedef unsigned UInt4 __attribute__((ext_vector_type(4)));
typedef unsigned short HalfBits4 __attribute__((ext_vector_type(4)));

static inline Float4 decode4(const unsigned short *p) {
    UInt4 h=__builtin_convertvector(*(const HalfBits4*)p,UInt4);
    UInt4 e=h&0x7c00u;
    if (e[0] && e[1] && e[2] && e[3] &&
        e[0]!=0x7c00u && e[1]!=0x7c00u && e[2]!=0x7c00u && e[3]!=0x7c00u) {
        /* Integer extension and bit reinterpretation, never half->float
         * convertvector: that builtin miscompiles on the current toolchain. */
        union { UInt4 u; Float4 f; } result;
        result.u=((h&0x8000u)<<16)|(((h&0x7fffu)+0x1c000u)<<13);
        return result.f;
    }
    return (Float4){load_half(p,0),load_half(p,1),load_half(p,2),load_half(p,3)};
}

static inline HalfBits4 encode4(Float4 value) {
    union { UInt4 u; Float4 f; } bits;
    bits.f=value;
    UInt4 magnitude=bits.u&0x7fffffffu;
    if (magnitude[0]>=0x38800000u && magnitude[0]<=0x477fe000u &&
        magnitude[1]>=0x38800000u && magnitude[1]<=0x477fe000u &&
        magnitude[2]>=0x38800000u && magnitude[2]<=0x477fe000u &&
        magnitude[3]>=0x38800000u && magnitude[3]<=0x477fe000u) {
        UInt4 rounded=((magnitude+0xfffu+((magnitude>>13)&1u)-0x38000000u)>>13)
            |((bits.u>>16)&0x8000u);
        return __builtin_convertvector(rounded,HalfBits4);
    }
    return (HalfBits4){round_half(value[0]),round_half(value[1]),
                      round_half(value[2]),round_half(value[3])};
}
'''


def main():
    OUT.mkdir(exist_ok=True)
    # Frozen baseline saved before promotion; rebuilding must not derive the
    # scalar control from the optimized production source.
    original=(OUT/'before/rmsnorm.c').read_text()
    start=original.index('        for (unsigned j = 0; j < 576; ++j) {',original.index('        float inverse ='))
    end=original.index('\n    }',start)
    prefix=b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n'
    helpers=b'\n'.join((FIXTURES/name).read_bytes().split(b'void controlled_act(unsigned p) {',1)[0]
        for name in ('arithmetic.c','resident_activation.c'))
    manifest=dict(status='building',sources_sha256={name:hashlib.sha256(
        (OUT/'before/rmsnorm.c' if name=='rmsnorm.c' else FIXTURES/name).read_bytes()).hexdigest()
        for name in ('arithmetic.c','resident_activation.c','rmsnorm.c')},kernels={})
    for name in ('unroll4','bits4','codec4','affine4'):
        if name=='unroll4':
            body='''        for (unsigned j=0;j<576;j+=4) {
            unsigned i=base+j;
            Float4 value={load_half(x,i),load_half(x,i+1),load_half(x,i+2),load_half(x,i+3)};
            Float4 normalized=value*inverse;
            unsigned short rounded[4]={round_half(normalized[0]),round_half(normalized[1]),
                                       round_half(normalized[2]),round_half(normalized[3])};
            Float4 half_value={load_half(rounded,0),load_half(rounded,1),load_half(rounded,2),load_half(rounded,3)};
            Float4 weight={load_half(gamma,i),load_half(gamma,i+1),load_half(gamma,i+2),load_half(gamma,i+3)};
            Float4 product=half_value*weight;
            out[i]=round_half(product[0]);out[i+1]=round_half(product[1]);
            out[i+2]=round_half(product[2]);out[i+3]=round_half(product[3]);
        }'''
        elif name=='bits4':
            body='''        for (unsigned j=0;j<576;j+=4) {
            unsigned i=base+j;
            HalfBits4 rounded=encode4(decode4(x+i)*inverse);
            *(HalfBits4*)(out+i)=encode4(decode4((const unsigned short*)&rounded)*decode4(gamma+i));
        }'''
        else:
            expression='decode4(x+i)' if name=='codec4' else 'decode4(x+i)*decode4(gamma+i)'
            body=f'        for (unsigned j=0;j<576;j+=4) {{ unsigned i=base+j; *(HalfBits4*)(out+i)=encode4({expression}); }}'
        source=original[:start]+body+original[end:]
        if name in ('codec4','affine4'):
            begin=source.index('        float total =')
            stop=source.index('        for (unsigned j=0;',begin)
            source=source[:begin]+source[stop:]
        source=source.encode()
        # Helpers use the scalar routines defined in the original source.
        entry=source.index(b'void controlled_act')
        source=source[:entry]+VECTOR_HELPERS.encode()+source[entry:]
        generated=prefix+helpers+b'\n'+source
        target='rmsnorm-output-'+name
        asm,ok=stage('compile',target,'moviCompile64.dll',1,
            ['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2',
             '-ffp-contract=off','-x','c','entry.c','-o','-'],generated)
        assert ok,(name,'compile')
        obj,ok=stage('assemble',target,'moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm)
        assert ok,(name,'assemble')
        elf,ok=stage('link',target,'moviLLD64.dll',3,
            ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
            (obj,(PROJECT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
        assert ok and len(kernel_text(elf))<=21216,(name,'link/window')
        for suffix,data in (('.c',source),('.s',asm),('.elf',elf)):(OUT/(name+suffix)).write_bytes(data)
        manifest['kernels'][name]=dict(source_sha256=hashlib.sha256(source).hexdigest(),
            generated_source_sha256=hashlib.sha256(generated).hexdigest(),elf_sha256=hashlib.sha256(elf).hexdigest(),
            assembly_sha256=hashlib.sha256(asm).hexdigest(),text_bytes=len(kernel_text(elf)))
    full=(OUT/'before/rmsnorm-f16.elf').read_bytes()
    (OUT/'scalar.elf').write_bytes(full)
    manifest['kernels']['scalar']=dict(elf_sha256=hashlib.sha256(full).hexdigest(),text_bytes=len(kernel_text(full)))
    manifest['status']='built'
    (OUT/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print({name:data['text_bytes'] for name,data in manifest['kernels'].items()},flush=True)


if __name__=='__main__':main()

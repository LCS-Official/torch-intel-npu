"""Same-carrier candidates for FP16 output-path range checks."""
import argparse
import hashlib
import json
from pathlib import Path
from w8_performance import ROOT, metadata, read, sha, load_weights

OUT=ROOT/'npunlock-probe/research/results/rmsnorm-encode'


def build():
    from probe_movi_targets import stage
    from npunlock.npu4000 import kernel_text
    fixtures=ROOT/'npunlock/tests/fixtures/npu4000';original=(fixtures/'rmsnorm.c').read_text(encoding='utf-8')
    old_start=original.index('static inline HalfBits4 encode4(');old_end=original.index('void controlled_act(',old_start)
    header='''static inline HalfBits4 encode4(Float4 value) {
    union { UInt4 u; Float4 f; } bits;
    bits.f=value;
    UInt4 magnitude=bits.u&0x7fffffffu;
    /* Unsigned subtraction combines the normal-value lower/upper tests. */
    UInt4 valid=(UInt4)((magnitude-0x38800000u)<=0x0effe000u)|(UInt4)(magnitude==0);
'''
    rounded='''    UInt4 rounded=((magnitude+0xfffu+((magnitude>>13)&1u)-0x38000000u)>>13)
        |((bits.u>>16)&0x8000u);
    rounded=(rounded&(UInt4)(magnitude!=0))|((bits.u>>16)&0x8000u);
'''
    mask=header+'''    if (valid[0]&valid[1]&valid[2]&valid[3]) {
'''+rounded+'''        return __builtin_convertvector(rounded,HalfBits4);
    }
    return (HalfBits4){round_half(value[0]),round_half(value[1]),
                      round_half(value[2]),round_half(value[3])};
}
'''
    lanes=header+rounded+'''    HalfBits4 result=__builtin_convertvector(rounded,HalfBits4);
    if (valid[0]&valid[1]&valid[2]&valid[3]) return result;
    if (!valid[0]) result[0]=round_half(value[0]);
    if (!valid[1]) result[1]=round_half(value[1]);
    if (!valid[2]) result[2]=round_half(value[2]);
    if (!valid[3]) result[3]=round_half(value[3]);
    return result;
}
'''
    helpers=b'\n'.join((fixtures/name).read_bytes().split(b'void controlled_act(unsigned p) {',1)[0] for name in ('arithmetic.c','resident_activation.c'))
    prefix=b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n'
    for candidate,replacement in (('mask',mask),('lanes',lanes)):
        directory=OUT/candidate;directory.mkdir(parents=True,exist_ok=True)
        body=(original[:old_start]+replacement+original[old_end:]).encode()
        roundtrip=body[:body.index(b'        float total =')]+b'''        for (unsigned j=0;j<576;j+=4) {
            unsigned i=base+j;*(HalfBits4*)(out+i)=encode4(decode4(x+i));
        }
    }
}
'''
        baseline=(fixtures/'rmsnorm/rmsnorm-f16.elf').read_bytes();(directory/'baseline.elf').write_bytes(baseline)
        (directory/'baseline.c').write_text(original,encoding='utf-8')
        manifest=dict(status='building',candidate=candidate,builder_sha256=sha(Path(__file__)),production=metadata(),
            baseline_manifest=read(fixtures/'rmsnorm/manifest.json'),kernels={'baseline':dict(elf_sha256=hashlib.sha256(baseline).hexdigest(),text_bytes=len(kernel_text(baseline)))})
        for name,source in (('codec',body),('roundtrip',roundtrip)):
            generated=prefix+helpers+b'\n'+source;target=f'rmsnorm-encode-{candidate}-{name}'
            args=['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2','-ffp-contract=off','-x','c','entry.c','-o','-']
            asm,ok=stage('compile',target,'moviCompile64.dll',1,args,generated);assert ok,(candidate,name,'compile')
            obj,ok=stage('assemble',target,'moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm);assert ok,(candidate,name,'assemble')
            elf,ok=stage('link',target,'moviLLD64.dll',3,['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
                (obj,(ROOT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()));assert ok and len(kernel_text(elf))<=21216
            for suffix,data in (('.c',source),('.s',asm),('.elf',elf)):(directory/(name+suffix)).write_bytes(data)
            manifest['kernels'][name]=dict(source_sha256=hashlib.sha256(source).hexdigest(),generated_source_sha256=hashlib.sha256(generated).hexdigest(),
                elf_sha256=hashlib.sha256(elf).hexdigest(),assembly_sha256=hashlib.sha256(asm).hexdigest(),text_bytes=len(kernel_text(elf)),compile_arguments=args)
        manifest['status']='built';(directory/'build.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(candidate,{name:r['text_bytes'] for name,r in manifest['kernels'].items()},flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('build','numeric','model'),required=True)
    parser.add_argument('--candidate',choices=('mask','lanes'),default='mask');parser.add_argument('--label',default='probe');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.case=='build':build();return 0
    import rmsnorm_reduce_probe as probe
    from torch_npu_smollm import FrozenSmolStack
    probe.OUT=OUT/args.candidate;original=FrozenSmolStack.from_checkpoint.__func__
    prepared=load_weights() if args.case=='model' else None
    def construct(cls,directory,**options):return original(cls,directory,**options,packed_weights=prepared if options.get('mode')=='w8a16' else None)
    FrozenSmolStack.from_checkpoint=classmethod(construct)
    try:code=probe.execute(args.case,31 if args.case=='model' else 21,args.label)
    finally:FrozenSmolStack.from_checkpoint=classmethod(original)
    path=probe.OUT/f'{args.case}-{args.label}.json';report=read(path)
    report.update(wrapper_sha256=sha(Path(__file__)),production=metadata(),w8_weights='activation_calibrated')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8');return code


if __name__=='__main__':raise SystemExit(main())

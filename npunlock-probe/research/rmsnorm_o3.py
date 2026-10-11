"""Isolated -O3 candidate; FP32 order and -ffp-contract=off are retained."""
import argparse
import json
from pathlib import Path
import shutil
from w8_performance import OUT as PERFORMANCE, metadata, read, sha, ROOT, load_weights

OUT=PERFORMANCE/'rmsnorm-o3'


def build():
    from probe_movi_targets import stage
    from npunlock.npu4000 import kernel_text
    import hashlib
    fixtures=ROOT/'npunlock/tests/fixtures/npu4000';body=(fixtures/'rmsnorm.c').read_bytes()
    prefix=b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n'
    helpers=b'\n'.join((fixtures/name).read_bytes().split(b'void controlled_act(unsigned p) {',1)[0] for name in ('arithmetic.c','resident_activation.c'))
    baseline=(fixtures/'rmsnorm/rmsnorm-f16.elf').read_bytes();(OUT/'baseline.elf').write_bytes(baseline)
    (OUT/'baseline.c').write_bytes(body)
    manifest=dict(status='building',builder_sha256=sha(Path(__file__)),production=metadata(),
        baseline_manifest=read(fixtures/'rmsnorm/manifest.json'),kernels={'baseline':dict(elf_sha256=hashlib.sha256(baseline).hexdigest(),text_bytes=len(kernel_text(baseline)))})
    roundtrip=body[:body.index(b'        float total =')]+b'''        for (unsigned j=0;j<576;j+=4) {
            unsigned i=base+j;*(HalfBits4*)(out+i)=encode4(decode4(x+i));
        }
    }
}
'''
    for name,source in (('codec',body),('roundtrip',roundtrip)):
        generated=prefix+helpers+b'\n'+source;target='rmsnorm-o3-'+name
        args=['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O3','-ffp-contract=off','-x','c','entry.c','-o','-']
        asm,ok=stage('compile',target,'moviCompile64.dll',1,args,generated);assert ok,(name,'compile')
        obj,ok=stage('assemble',target,'moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm);assert ok,(name,'assemble')
        elf,ok=stage('link',target,'moviLLD64.dll',3,['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
            (obj,(ROOT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()));assert ok and len(kernel_text(elf))<=21216
        for suffix,data in (('.c',source),('.s',asm),('.elf',elf)):(OUT/(name+suffix)).write_bytes(data)
        manifest['kernels'][name]=dict(source_sha256=hashlib.sha256(source).hexdigest(),generated_source_sha256=hashlib.sha256(generated).hexdigest(),
            elf_sha256=hashlib.sha256(elf).hexdigest(),assembly_sha256=hashlib.sha256(asm).hexdigest(),text_bytes=len(kernel_text(elf)),compile_arguments=args)
    manifest.update(status='built',code_identical=kernel_text(baseline)==kernel_text((OUT/'codec.elf').read_bytes()))
    (OUT/'build.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print({name:record['text_bytes'] for name,record in manifest['kernels'].items()},'identical',manifest['code_identical'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('build','numeric','model'),required=True)
    parser.add_argument('--label',default='probe');args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.case=='build':build();return 0
    import rmsnorm_reduce_probe as probe
    from torch_npu_smollm import FrozenSmolStack
    probe.OUT=OUT;original=FrozenSmolStack.from_checkpoint.__func__;prepared=load_weights()
    def construct(cls,directory,**options):return original(cls,directory,**options,packed_weights=prepared if options.get('mode')=='w8a16' else None)
    FrozenSmolStack.from_checkpoint=classmethod(construct)
    try:code=probe.execute(args.case,31 if args.case=='model' else 21,args.label)
    finally:FrozenSmolStack.from_checkpoint=classmethod(original)
    path=OUT/f'{args.case}-{args.label}.json';report=read(path)
    report.update(wrapper_sha256=sha(Path(__file__)),production=metadata(),w8_weights='activation_calibrated')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8');return code


if __name__=='__main__':raise SystemExit(main())

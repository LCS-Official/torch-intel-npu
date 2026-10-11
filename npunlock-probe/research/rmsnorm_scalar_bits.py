"""Compare nonnegative FP32 bit order in the single-row reduction path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from w8_quality_probe import ROOT, sha

OUT = ROOT / 'npunlock-probe/research/results/decoder-native/scalar-bits'


def build():
    import numpy as np
    import probe_movi_targets as compiler
    from npunlock.npu4000 import kernel_text
    fixtures = ROOT / 'npunlock/tests/fixtures/npu4000'
    source = (fixtures / 'rmsnorm.c').read_bytes()
    old = b'                float error=total>=square ? (total-next)+square : (square-next)+total;'
    new = b'''                union { unsigned u;float f; } total_bits,square_bits;
                total_bits.f=total;square_bits.f=square;
                unsigned choose=(total_bits.u>=square_bits.u)
                    &(total_bits.u<=0x7f800000u)&(square_bits.u<=0x7f800000u);
                float error=choose ? (total-next)+square : (square-next)+total;'''
    assert source.count(old) == 4
    source = source.replace(old, new)
    codes = np.arange(65536, dtype=np.uint16)
    with np.errstate(invalid='ignore', over='ignore'):squares = codes.view(np.float16).astype(np.float32)**2
    totals = np.random.default_rng(92311).integers(0, 0x7f800001, len(codes), dtype=np.uint32)
    totals[:6] = [0, 0x7f800000, 0x7fc00000, 0xffc00000, 0x7f800001, 0xffffffff]
    bits = squares.view(np.uint32)
    with np.errstate(invalid='ignore'):
        np.testing.assert_array_equal((totals>=bits)&(totals<=0x7f800000)&(bits<=0x7f800000), totals.view(np.float32)>=squares)
    OUT.mkdir(parents=True, exist_ok=True)
    compiler.OUT = OUT / 'worker';compiler.OUT.mkdir(exist_ok=True)
    previous = OUT.parent / 'duplicate-rows'
    prior = json.loads((previous / 'build.json').read_text())
    manifest = dict(status='building', builder_sha256=sha(Path(__file__)), comparison_checks=len(codes),
                    policy='Preserve each FP32 Neumaier operation; replace scalar nonnegative comparison with guarded unsigned bit order', kernels={})
    for name in ('baseline', 'roundtrip'):
        for suffix in ('.c', '.s', '.elf'):shutil.copyfile(previous / (name+suffix), OUT / (name+suffix))
        manifest['kernels'][name] = prior['kernels'][name]
        assert sha(OUT / (name+'.elf')) == manifest['kernels'][name]['elf_sha256']
    assert sha(fixtures / 'rmsnorm.c') == manifest['kernels']['baseline']['source_sha256']
    helpers = b'\n'.join((fixtures / name).read_bytes().split(b'void controlled_act(unsigned p) {', 1)[0]
                         for name in ('arithmetic.c', 'resident_activation.c'))
    generated = b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n' + helpers + b'\n' + source
    args = ['moviCompile.dll', '-cc1', '-triple', 'shave', '-target-cpu', '4000xx', '-S', '-O2', '-ffp-contract=off', '-x', 'c', 'entry.c', '-o', '-']
    asm, ok = compiler.stage('compile', 'scalar-bits', 'moviCompile64.dll', 1, args, generated);assert ok, 'compile'
    obj, ok = compiler.stage('assemble', 'scalar-bits', 'moviAsm64.dll', 2, ['moviAsm.dll', '--cv', '4000xx', '--noSPrefixing'], asm);assert ok, 'assemble'
    elf, ok = compiler.stage('link', 'scalar-bits', 'moviLLD64.dll', 3,
        ['moviLLD.dll', '-flavor', 'gnu', '-EL', '-e', 'controlled_act', '-z', 'max-page-size=0x10', '--gc-sections'],
        (obj, (ROOT / 'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
    assert ok and len(kernel_text(elf)) <= 21216
    for suffix, data in (('.c', source), ('.s', asm), ('.elf', elf)):(OUT / ('codec'+suffix)).write_bytes(data)
    manifest['kernels']['codec'] = dict(source_sha256=hashlib.sha256(source).hexdigest(),
        generated_source_sha256=hashlib.sha256(generated).hexdigest(), assembly_sha256=sha(OUT/'codec.s'),
        elf_sha256=sha(OUT/'codec.elf'), text_bytes=len(kernel_text(elf)), compile_arguments=args)
    manifest['status'] = 'built'
    (OUT/'build.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    shutil.copyfile(Path(__file__), OUT/'builder.py')
    print({name: row['text_bytes'] for name, row in manifest['kernels'].items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--case', choices=('build', 'numeric', 'model'), required=True)
    args = parser.parse_args();os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.case == 'build':build()
    else:
        import rmsnorm_row4 as row4
        row4.OUT=OUT
        if args.case == 'numeric':raise SystemExit(row4.numeric('scalar'))
        row4.model(1, 24, 'full30', save_graph_blobs=False)

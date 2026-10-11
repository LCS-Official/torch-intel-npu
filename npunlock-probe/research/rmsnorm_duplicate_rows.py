"""Isolated exact duplicate-row RMSNorm reuse; retain the production fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from w8_quality_probe import ROOT, sha

OUT = ROOT / 'npunlock-probe/research/results/decoder-native/duplicate-rows'


def build():
    import probe_movi_targets as compiler
    from npunlock.npu4000 import kernel_text
    fixtures = ROOT / 'npunlock/tests/fixtures/npu4000'
    original = (fixtures / 'rmsnorm.c').read_bytes()
    marker = b'    unsigned base=0;'
    replacement = b'''    /* B1 carriers duplicate a row; compare raw bits before any output write. */
    unsigned repeated = n == 2u*576u;
    if (repeated) {
        for (unsigned j=0;j<576;++j) {
            if (x[j]!=x[576+j] || gamma[j]!=gamma[576+j]) { repeated=0;break; }
        }
    }
    unsigned limit = repeated ? 576u : n;
    unsigned base=0;'''
    assert original.count(marker) == 1
    source = original.replace(marker, replacement)
    marker = b'    for (; base < n; base += 576) {'
    assert source.count(marker) == 1
    source = source.replace(marker, b'    for (; base < limit; base += 576) {')
    assert source.endswith(b'    }\n}\n')
    source = source[:-2] + b'''    if (repeated) {
        for (unsigned j=0;j<576;j+=4)
            *(HalfBits4*)(out+576+j)=*(HalfBits4*)(out+j);
    }
}
'''
    OUT.mkdir(parents=True, exist_ok=True)
    compiler.OUT = OUT / 'worker'
    compiler.OUT.mkdir(exist_ok=True)
    manifest = dict(status='building', builder_sha256=sha(Path(__file__)),
                    production_source_sha256=sha(fixtures / 'rmsnorm.c'),
                    policy='Only two bit-identical input and gamma rows reuse the first output; arithmetic order unchanged', kernels={})
    for suffix in ('.c', '.s', '.elf'):
        path = fixtures / ('rmsnorm.c' if suffix == '.c' else 'rmsnorm/rmsnorm-f16' + suffix)
        shutil.copyfile(path, OUT / ('baseline' + suffix))
    previous = ROOT / 'npunlock-probe/research/results/rmsnorm-row4-bits'
    prior = json.loads((previous / 'build.json').read_text())
    for suffix in ('.c', '.s', '.elf'):
        shutil.copyfile(previous / ('roundtrip' + suffix), OUT / ('roundtrip' + suffix))
    assert sha(OUT / 'roundtrip.elf') == prior['kernels']['roundtrip']['elf_sha256']
    for name in ('baseline', 'roundtrip'):
        manifest['kernels'][name] = dict(elf_sha256=sha(OUT / (name + '.elf')),
            source_sha256=sha(OUT / (name + '.c')), assembly_sha256=sha(OUT / (name + '.s')),
            text_bytes=len(kernel_text((OUT / (name + '.elf')).read_bytes())))
    helpers = b'\n'.join((fixtures / name).read_bytes().split(b'void controlled_act(unsigned p) {', 1)[0]
                         for name in ('arithmetic.c', 'resident_activation.c'))
    generated = b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n' + helpers + b'\n' + source
    args = ['moviCompile.dll', '-cc1', '-triple', 'shave', '-target-cpu', '4000xx', '-S', '-O2', '-ffp-contract=off', '-x', 'c', 'entry.c', '-o', '-']
    asm, ok = compiler.stage('compile', 'duplicate-rows', 'moviCompile64.dll', 1, args, generated)
    assert ok, 'compile'
    obj, ok = compiler.stage('assemble', 'duplicate-rows', 'moviAsm64.dll', 2, ['moviAsm.dll', '--cv', '4000xx', '--noSPrefixing'], asm)
    assert ok, 'assemble'
    elf, ok = compiler.stage('link', 'duplicate-rows', 'moviLLD64.dll', 3,
        ['moviLLD.dll', '-flavor', 'gnu', '-EL', '-e', 'controlled_act', '-z', 'max-page-size=0x10', '--gc-sections'],
        (obj, (ROOT / 'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
    assert ok and len(kernel_text(elf)) <= 21216
    for suffix, value in (('.c', source), ('.s', asm), ('.elf', elf)):
        (OUT / ('codec' + suffix)).write_bytes(value)
    manifest['kernels']['codec'] = dict(source_sha256=hashlib.sha256(source).hexdigest(),
        generated_source_sha256=hashlib.sha256(generated).hexdigest(), assembly_sha256=sha(OUT / 'codec.s'),
        elf_sha256=sha(OUT / 'codec.elf'), text_bytes=len(kernel_text(elf)), compile_arguments=args)
    manifest['status'] = 'built'
    (OUT / 'build.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    shutil.copyfile(Path(__file__), OUT / 'builder.py')
    print({name: record['text_bytes'] for name, record in manifest['kernels'].items()}, flush=True)


def numeric():
    import rmsnorm_row4 as row4
    import rmsnorm_reduce_probe as probe
    row4.OUT = probe.OUT = OUT
    original = probe.numeric

    def check(npu, torch, report, rounds):
        original(npu, torch, report, rounds)
        torch.manual_seed(92311)
        base = torch.randn(1, 576).half().repeat(2, 1)
        weight = torch.randn(1, 576).half().repeat(2, 1)
        cases = [('duplicate-random', base.clone(), weight.clone())]
        for tag, value in (('signed-zero', -0.), ('subnormal', 2**-24), ('large', 65504.), ('inf', float('inf')), ('nan', float('nan'))):
            cases.append(('duplicate-' + tag, torch.full_like(base, value), weight.clone()))
        for tag, value in (('gamma-subnormal', 2**-24), ('gamma-inf', float('inf')), ('gamma-nan', float('nan'))):
            cases.append(('duplicate-' + tag, base.clone(), torch.full_like(weight, value)))
        for target in ('input', 'gamma'):
            host, gamma = base.clone(), weight.clone()
            (host if target == 'input' else gamma).view(torch.int16)[1, -1] ^= 1
            cases.append(('one-bit-different-' + target, host, gamma))
        host = base.clone();host[1] *= -1
        cases.append(('different-input', host, weight.clone()))
        gamma = weight.clone();gamma[1] *= -1
        cases.append(('different-gamma', base.clone(), gamma))
        for tag, host, gamma in cases:
            x, g = host.to('intel_npu'), gamma.to('intel_npu')
            baseline = None
            for name in probe.NAMES:
                value = npu._run(('rmsnorm_reduce_probe', 2, name), [x, g], (2, 576), torch.float16).cpu()
                if baseline is None:baseline = value
                equal = (value.view(torch.int16) == baseline.view(torch.int16)) | (value.isnan() & baseline.isnan())
                assert equal.all(), (tag, name)
                report['checks'].append(dict(tag=tag, kernel=name, rows=2, bit_differences=0,
                    nan_policy='NaN-equivalent; all finite encodings and signed zero bit-equal'))
    probe.numeric = check
    try:return row4.numeric('duplicate')
    finally:probe.numeric = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('build', 'numeric', 'model'), required=True)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    if args.case == 'build':build()
    elif args.case == 'numeric':raise SystemExit(numeric())
    else:
        import rmsnorm_row4 as row4
        row4.OUT = OUT
        row4.model(1, 24, 'full30', save_graph_blobs=False)

"""Preserve row4 Neumaier order; avoid scalarized FP32 vector comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
import rmsnorm_row4 as row4
from w8_quality_probe import ROOT, sha
from shave_fp32_pipeline import metadata

OUT = ROOT / 'npunlock-probe/research/results/rmsnorm-row4-bits'
PREVIOUS = ROOT / 'npunlock-probe/research/results/shave-fp32-pipeline/row4'


def build(ordered=False, lanes=4, absolute=False, min_rows=16):
    import probe_movi_targets as compiler
    from npunlock.npu4000 import kernel_text
    compiler.OUT = OUT / 'worker'
    compiler.OUT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((PREVIOUS / 'build.json').read_text(encoding='utf-8'))
    source = (PREVIOUS / 'codec.c').read_bytes()
    assert hashlib.sha256(source).hexdigest() == previous['kernels']['codec']['source_sha256']
    old = b'            UInt4 choose=(UInt4)(total>=square);'
    new = b'''            union { UInt4 u;Float4 f; } total_bits,square_bits;
            total_bits.f=total;square_bits.f=square;
            // Nonnegative IEEE values sort by bits; unordered NaN still chooses false.
            UInt4 choose=(UInt4)(total_bits.u>=square_bits.u)
                &(UInt4)(total_bits.u<=0x7f800000u)
                &(UInt4)(square_bits.u<=0x7f800000u);'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    if ordered:
        old = b'''            union { UInt4 u;Float4 f; } left,right,error;
            left.f=(total-next)+square;right.f=(square-next)+total;
            error.u=(left.u&choose)|(right.u&~choose);
            correction=correction+error.f;total=next;'''
        new = b'''            union { UInt4 u;Float4 f; } larger,smaller;
            UInt4 swap=(total_bits.u^square_bits.u)&choose;
            larger.u=square_bits.u^swap;smaller.u=total_bits.u^swap;
            correction=correction+((larger.f-next)+smaller.f);total=next;'''
        assert source.count(old) == 1
        source = source.replace(old, new)
    if lanes == 16:
        helper = source[source.index(b'static inline Float4 decode_bits4'):source.index(b'static inline Float4 decode4')]
        helper = helper.replace(b'Float4', b'Float16').replace(b'UInt4', b'UInt16').replace(b'HalfBits4', b'HalfBits16').replace(b'decode_bits4', b'decode_bits16')
        if absolute:
            old = b'result.u=((h&0x8000u)<<16)|(small.u&mask)|(normal.u&~mask);'
            assert helper.count(old) == 1
            # Only the square-sum path discards sign; signed output decoding is untouched.
            helper = helper.replace(old, b'result.u=(small.u&mask)|(normal.u&~mask);')
        helper = b'''typedef float Float16 __attribute__((ext_vector_type(16)));
typedef unsigned UInt16 __attribute__((ext_vector_type(16)));
typedef unsigned short HalfBits16 __attribute__((ext_vector_type(16)));
''' + helper
        marker = b'void controlled_act(unsigned p) {'
        assert source.count(marker) == 1
        source = source.replace(marker, helper + marker)
        start, end = source.index(b'    unsigned base=0;'), source.index(b'        for (unsigned lane=0;lane<4;++lane)')
        block = source[start:end].replace(b'4u*576u', b'16u*576u')
        block = block.replace(b'Float4', b'Float16').replace(b'UInt4', b'UInt16').replace(b'HalfBits4', b'HalfBits16').replace(b'decode_bits4', b'decode_bits16')
        block = block.replace(b'(Float16){0,0,0,0}', b'(Float16){0}')
        raw = b'x[base+j],x[base+576+j],x[base+1152+j],x[base+1728+j]'
        assert block.count(raw) == 1
        block = block.replace(raw, ','.join(f'x[base+{i * 576}+j]' for i in range(16)).encode())
        source = source[:start] + block + source[end:].replace(b'for (unsigned lane=0;lane<4;++lane)', b'for (unsigned lane=0;lane<16;++lane)', 1)
        if min_rows in (32, 256):
            old = b'for (; base+16u*576u<=n; base+=16u*576u)'
            assert source.count(old) == 1
            source = source.replace(old, f'for (; n>={min_rows}u*576u && base+16u*576u<=n; base+=16u*576u)'.encode())
    # Exhaust half squares and sample reachable nonnegative FP32 sums, plus NaNs.
    codes = np.arange(65536, dtype=np.uint16)
    with np.errstate(invalid='ignore', over='ignore'):
        squares = codes.view(np.float16).astype(np.float32) ** 2
    rng = np.random.default_rng(92311)
    totals = rng.integers(0, 0x7f800001, len(codes), dtype=np.uint32)
    totals[:6] = [0, 0x7f800000, 0x7fc00000, 0xffc00000, 0x7f800001, 0xffffffff]
    bits = squares.view(np.uint32)
    choose = (totals >= bits) & (totals <= 0x7f800000) & (bits <= 0x7f800000)
    with np.errstate(invalid='ignore'):
        np.testing.assert_array_equal(choose, totals.view(np.float32) >= squares)
    manifest = dict(status='building', builder_sha256=sha(Path(__file__)), production=metadata(),
                    previous_build_sha256=sha(PREVIOUS / 'build.json'), comparison_checks=len(codes),
                    ordered_operands=ordered, lanes=lanes, absolute_square_decode=absolute, min_rows=min_rows,
                    policy='Unchanged FP32 per-row Neumaier order; integer ordering of nonnegative operands with NaN guards', kernels={})
    fixtures = ROOT / 'npunlock/tests/fixtures/npu4000'
    helpers = b'\n'.join((fixtures / name).read_bytes().split(b'void controlled_act(unsigned p) {', 1)[0]
                         for name in ('arithmetic.c', 'resident_activation.c'))
    for name in ('baseline', 'roundtrip'):
        for suffix in ('.c', '.s', '.elf'):
            shutil.copyfile(PREVIOUS / (name + suffix), OUT / (name + suffix))
        assert sha(OUT / (name + '.elf')) == previous['kernels'][name]['elf_sha256']
        manifest['kernels'][name] = previous['kernels'][name]
    generated = b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n' + helpers + b'\n' + source
    args = ['moviCompile.dll', '-cc1', '-triple', 'shave', '-target-cpu', '4000xx', '-S', '-O2', '-ffp-contract=off', '-x', 'c', 'entry.c', '-o', '-']
    asm, ok = compiler.stage('compile', 'row4-bits', 'moviCompile64.dll', 1, args, generated)
    assert ok, 'compile'
    obj, ok = compiler.stage('assemble', 'row4-bits', 'moviAsm64.dll', 2, ['moviAsm.dll', '--cv', '4000xx', '--noSPrefixing'], asm)
    assert ok, 'assemble'
    elf, ok = compiler.stage('link', 'row4-bits', 'moviLLD64.dll', 3,
                             ['moviLLD.dll', '-flavor', 'gnu', '-EL', '-e', 'controlled_act', '-z', 'max-page-size=0x10', '--gc-sections'],
                             (obj, (ROOT / 'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
    assert ok and len(kernel_text(elf)) <= 21216
    for suffix, data in (('.c', source), ('.s', asm), ('.elf', elf)):
        (OUT / ('codec' + suffix)).write_bytes(data)
    manifest['kernels']['codec'] = dict(source_sha256=hashlib.sha256(source).hexdigest(),
        generated_source_sha256=hashlib.sha256(generated).hexdigest(), elf_sha256=hashlib.sha256(elf).hexdigest(),
        assembly_sha256=hashlib.sha256(asm).hexdigest(), text_bytes=len(kernel_text(elf)), compile_arguments=args)
    manifest['status'] = 'built'
    shutil.copyfile(Path(__file__), OUT / 'builder.py')
    manifest['helpers_sha256'] = {}
    for name in ('rmsnorm_row4.py', 'rmsnorm_reduce_probe.py', 'shave_fp32_pipeline.py'):
        path = ROOT / 'npunlock-probe/research' / name
        shutil.copyfile(path, OUT / name)
        manifest['helpers_sha256'][name] = sha(path)
    (OUT / 'build.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print({name: data['text_bytes'] for name, data in manifest['kernels'].items()}, flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('build', 'numeric', 'model', 'promote', 'regression', 'audit'), required=True)
    parser.add_argument('--label', default='probe')
    parser.add_argument('--ordered', action='store_true')
    parser.add_argument('--lanes', choices=(4, 16), type=int, default=4)
    parser.add_argument('--absolute', action='store_true')
    parser.add_argument('--min-rows', type=int, choices=(16, 32, 256), default=16)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--prefix', type=int, default=32)
    args = parser.parse_args()
    if args.ordered:
        OUT = OUT.with_name('rmsnorm-row4-ordered')
    if args.lanes == 16:
        OUT = OUT.with_name('rmsnorm-row16-bits')
    if args.absolute:
        assert args.lanes == 16
        OUT = OUT.with_name('rmsnorm-row16-absolute')
    if args.min_rows == 256:
        assert args.lanes == 16 and not args.absolute and not args.ordered
        OUT = OUT.with_name('rmsnorm-row16-gated')
    if args.min_rows == 32:
        assert args.lanes == 16 and not args.absolute and not args.ordered
        OUT = OUT.with_name('rmsnorm-row16-gated32')
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    if args.case == 'audit':
        return audit()
    if args.case == 'build':
        build(args.ordered, args.lanes, args.absolute, args.min_rows)
        return 0
    if args.case == 'promote':
        return promote()
    if args.case == 'regression':
        import rmsnorm_reduce_regression as regression
        regression.OUT = OUT / 'regressions'
        return regression.main()
    row4.OUT = OUT
    if args.case == 'model':
        row4.model(args.batch, args.prefix, args.label, save_graph_blobs=False)
        return 0
    return row4.numeric(args.label)


def audit():
    from npunlock.npu4000 import kernel_text
    stages = ('rmsnorm-row4-bits', 'rmsnorm-row4-ordered', 'rmsnorm-row16-bits', 'rmsnorm-row16-absolute', 'rmsnorm-row16-gated', 'rmsnorm-row16-gated32')
    report = dict(status='running', files={}, numeric_checks=0, graph_checks=0, array_comparisons=0, stages={})
    for name in stages:
        directory = OUT.parent / name
        build = json.loads((directory / 'build.json').read_text(encoding='utf-8'))
        assert build['status'] == 'built' and sha(directory / 'builder.py') == build['builder_sha256']
        for helper, digest in build.get('helpers_sha256', {}).items():
            assert sha(directory / helper) == digest
        for relative, digest in build['production']['sources'].items():
            if '/research/' not in relative:
                promoted = OUT.parent / 'rmsnorm-row16-gated32/promotion.json'
                if promoted.exists() and relative in ('npunlock/tests/fixtures/npu4000/rmsnorm.c', 'npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf'):
                    info = json.loads(promoted.read_text(encoding='utf-8'))
                    assert sha(ROOT / relative) == info['after'][relative], relative
                else:
                    assert sha(ROOT / relative) == digest, relative
        for kernel, info in build['kernels'].items():
            assert sha(directory / (kernel + '.elf')) == info['elf_sha256']
            assert len(kernel_text((directory / (kernel + '.elf')).read_bytes())) == info['text_bytes']
            if 'source_sha256' in info:
                assert sha(directory / (kernel + '.c')) == info['source_sha256']
            if 'assembly_sha256' in info:
                assert sha(directory / (kernel + '.s')) == info['assembly_sha256']
        records = []
        for path in sorted(directory.glob('numeric-*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            assert data['status'] == 'passed', path
            assert data['build']['kernels']['codec']['elf_sha256'] == build['kernels']['codec']['elf_sha256']
            report['numeric_checks'] += len(data['checks'])
            for row in data['checks']:
                assert row['bit_differences'] == 0
            for rows, patches in data['carrier_evidence'].items():
                layouts = set()
                for patch in patches:
                    blob = (directory / f"norm-{rows}-{patch['kernel']}.blob").read_bytes()
                    assert hashlib.sha256(blob).hexdigest() == patch['patched_sha256']
                    start = patch['code_offset']; end = start + patch['window_bytes']
                    elf = (directory / (patch['kernel'] + '.elf')).read_bytes()
                    text = kernel_text(elf)
                    assert blob[start:start + len(text)] == text
                    digest = hashlib.sha256(blob[:start] + bytes(end-start) + blob[end:]).hexdigest()
                    assert digest == patch['layout_sha256']; layouts.add(digest)
                    report['graph_checks'] += 1
                assert len(layouts) == 1
            records.append(dict(path=path.name, checks=len(data['checks']), timings=data['timings']))
            stats = data['final_stats']
            assert stats['closed'] and not stats['context_failed']
            assert all(stats[key] == 0 for key in ('live_buffers', 'cached_buffer_bytes', 'quarantined_buffer_bytes'))
        models = []
        for path in sorted(directory.glob('model-*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            assert data['status'] == 'passed' and data['immutable_seed_arrays'] == 90
            assert any(sha(source) == data['wrapper_sha256'] for source in (directory / 'rmsnorm_row4.py', directory / 'model-runner.py') if source.exists())
            assert len(data['graphs']) == 60 and data['checks'][0]['bit_equal_arrays'] == 182
            output = directory / (data['label'] + '-outputs.npz')
            assert sha(output) == data['outputs_sha256']
            with np.load(output) as arrays:
                for i in range(182):
                    np.testing.assert_array_equal(arrays[f'baseline/{i}'].view(np.uint8), arrays[f'codec/{i}'].view(np.uint8))
                    report['array_comparisons'] += 1
                for i in range(90):
                    np.testing.assert_array_equal(arrays[f'baseline/{1+i}'].view(np.uint8), arrays[f'preserved/{i}'].view(np.uint8))
                    report['array_comparisons'] += 1
            for row in data['graphs']:
                assert row['baseline_blob_sha256'] != row['codec_blob_sha256']
                assert len(row['codec']) == (2 if row['operation'] == 'decode' else 1)
            stats = data['final_stats']
            assert stats['closed'] and not stats['context_failed']
            assert all(stats[key] == 0 for key in ('live_buffers', 'cached_buffer_bytes', 'quarantined_buffer_bytes'))
            models.append(dict(path=path.name, batch=data['batch'], prefix=data['prefix'], timings=data['timings']))
        report['stages'][name] = dict(numeric=records, models=models)
        for path in directory.rglob('*'):
            if path.is_file() and path.name not in ('audit.json', 'REPORT.md'):
                report['files'][str(path.relative_to(OUT.parent))] = sha(path)
    selected = OUT.parent / 'rmsnorm-row16-gated32'
    if (selected / 'promotion.json').exists():
        promotion = json.loads((selected / 'promotion.json').read_text(encoding='utf-8'))
        assert promotion['status'] == 'passed'
        for version in ('before', 'after'):
            for relative, digest in promotion[version].items():
                assert sha(selected / version / relative) == digest
                if version == 'after':
                    assert sha(ROOT / relative) == digest
        old = (selected / 'baseline.c').read_bytes().replace(b'\r\n', b'\n')
        current = (ROOT / 'npunlock/tests/fixtures/npu4000/rmsnorm.c').read_bytes().replace(b'\r\n', b'\n')
        assert old[old.index(b'        float total ='):] == current[current.index(b'        float total ='):]
        report['regressions'] = []
        for module, cases in (('torch_npu_smollm_demo', ('norm','numeric','boundary','guards','causality','hybrid')),
                              ('torch_npu_smollm_stack_demo', ('numeric','boundary'))):
            for case in cases:
                path = selected / 'regressions' / module / (case + '.json')
                data = json.loads(path.read_text(encoding='utf-8'))
                assert data['status'] == 'passed'
                assert data['kernel_manifest']['elf_sha256'] == sha(ROOT / 'npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf')
                stats = data['final_stats']
                assert not stats['context_failed']
                assert all(stats[key] == 0 for key in ('live_buffers', 'cached_buffer_bytes', 'quarantined_buffer_bytes'))
                report['regressions'].append(dict(module=module, case=case, checks=len(data['checks']), npu_calls=stats['npu_calls']))
    report['status'] = 'passed'
    (OUT.parent / 'rmsnorm-row16-gated32/audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({key: report[key] for key in ('status', 'numeric_checks', 'graph_checks', 'array_comparisons')}, flush=True)
    return 0


def promote():
    assert OUT.name == 'rmsnorm-row16-gated32'
    numeric = json.loads((OUT / 'numeric-gated32.json').read_text(encoding='utf-8'))
    assert numeric['status'] == 'passed' and len(numeric['checks']) == 158
    for batch in (1, 4, 8):
        data = json.loads((OUT / f'model-b{batch}.json').read_text(encoding='utf-8'))
        assert data['status'] == 'passed' and data['checks'][0]['bit_equal_arrays'] == 182
    build = json.loads((OUT / 'build.json').read_text(encoding='utf-8'))
    candidate = build['kernels']['codec']
    assert sha(OUT / 'codec.c') == candidate['source_sha256']
    names = ('npunlock/tests/fixtures/npu4000/rmsnorm.c',
             'npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf',
             'npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.s',
             'npunlock/tests/fixtures/npu4000/rmsnorm/manifest.json')
    report = dict(status='running', before={}, after={})
    assert not (OUT / 'promotion.json').exists(), 'Promotion already recorded'
    for relative in names:
        path = OUT / 'before' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
        report['before'][relative] = sha(path)
    import probe_movi_targets as compiler
    import build_rmsnorm
    compiler.OUT = OUT / 'formal-build'
    compiler.OUT.mkdir(exist_ok=True)
    try:
        shutil.copyfile(OUT / 'codec.c', ROOT / names[0])
        build_rmsnorm.main()
        assert sha(ROOT / names[1]) == candidate['elf_sha256']
        assert sha(ROOT / names[2]) == candidate['assembly_sha256']
    except Exception:
        for relative in names:
            shutil.copyfile(OUT / 'before' / relative, ROOT / relative)
        raise
    for relative in names:
        path = OUT / 'after' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
        report['after'][relative] = sha(path)
    report['status'] = 'passed'
    (OUT / 'promotion.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Promoted exact tested C/ELF/assembly with standard build manifest', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

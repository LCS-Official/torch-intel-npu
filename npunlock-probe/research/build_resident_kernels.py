"""Build resident reduction/activation kernels with the isolated Movi worker."""
import hashlib
import json
from probe_movi_targets import stage, PROJECT

ROOT = PROJECT / 'npunlock/tests/fixtures/npu4000'
DEST = ROOT / 'resident'
SOURCES = [ROOT / name for name in ('arithmetic.c', 'resident_reduce.c', 'resident_activation.c', 'resident_integer.c')]


def main():
    DEST.mkdir(exist_ok=True)
    manifest = {'target': '4000xx', 'source_sha256': {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}, 'kernels': {}}
    kinds = ('sum', 'mean', 'relu-f16', 'relu-f32', 'relu_grad-f16', 'relu_grad-f32', 'rsqrt-f32', 'int64-add')
    for kind in kinds:
        if kind in ('sum', 'mean'):
            prefix = f'#define REDUCE_MEAN {int(kind == "mean")}\n'
            body = SOURCES[1].read_bytes()
        else:
            prefix = f'#define DTYPE {1 if kind.endswith("f32") else 2}\n'
            if kind == 'rsqrt-f32':
                prefix += '#define RSQRT 1\n'
            body = (f'\n#undef INPUTS\n#define INPUTS {2 if kind.startswith("relu_grad") else 1}\n'.encode() +
                    SOURCES[2].read_bytes())
        helpers = SOURCES[0].read_bytes().split(b'void controlled_act(unsigned p) {', 1)[0]
        source = SOURCES[3].read_bytes() if kind == 'int64-add' else prefix.encode() + helpers + body
        assembly, ok = stage('compile', 'resident-' + kind, 'moviCompile64.dll', 1,
            ['moviCompile.dll', '-cc1', '-triple', 'shave', '-target-cpu', '4000xx',
             '-S', '-O2', '-ffp-contract=off', '-x', 'c', 'entry.c', '-o', '-'], source)
        if not ok:
            raise RuntimeError(f'Compile failed: {kind}')
        obj, ok = stage('assemble', 'resident-' + kind, 'moviAsm64.dll', 2,
            ['moviAsm.dll', '--cv', '4000xx', '--noSPrefixing'], assembly)
        if not ok:
            raise RuntimeError(f'Assembly failed: {kind}')
        elf, ok = stage('link', 'resident-' + kind, 'moviLLD64.dll', 3,
            ['moviLLD.dll', '-flavor', 'gnu', '-EL', '-e', 'controlled_act',
             '-z', 'max-page-size=0x10', '--gc-sections'],
            (obj, (PROJECT / 'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
        if not ok:
            raise RuntimeError(f'Link failed: {kind}')
        (DEST / f'{kind}.elf').write_bytes(elf)
        (DEST / f'{kind}.s').write_bytes(assembly)
        manifest['kernels'][kind] = {'elf_sha256': hashlib.sha256(elf).hexdigest(), 'elf_bytes': len(elf),
            'assembly_sha256': hashlib.sha256(assembly).hexdigest()}
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    from build_classification_kernels import main as build_classification
    build_classification()
    from build_optimizer_kernels import main as build_optimizer
    build_optimizer()
    from build_gelu_kernels import main as build_gelu
    build_gelu()
    from build_layernorm_kernel import main as build_layernorm
    build_layernorm()
    from build_softmax_kernels import main as build_softmax
    build_softmax()
    from build_sdpa_softmax_kernel import main as build_sdpa
    build_sdpa()
    from build_native_half_softmax_kernel import main as build_native_half
    build_native_half()
    from build_unary_math_kernels import main as build_unary_math
    build_unary_math()
    from build_discrete_rng_kernels import main as build_discrete_rng
    build_discrete_rng()
    from build_index_kernels import main as build_index
    build_index()
    from build_row_extrema_kernels import main as build_extrema
    build_extrema()
    from build_selection_kernels import main as build_selection
    build_selection()
    from build_sort_kernel import main as build_sort
    build_sort()
    from build_order_stats_kernel import main as build_order_stats
    build_order_stats()
    from build_grid_kernel import main as build_grid
    build_grid()
    from build_compat_kernels import main as build_compat
    build_compat()
    from build_stream_sum_kernel import main as build_stream_sum
    build_stream_sum()
    from build_flash_attention_kernels import main as build_flash
    build_flash()


if __name__ == '__main__':
    main()

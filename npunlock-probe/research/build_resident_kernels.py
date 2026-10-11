"""Build resident reduction/activation kernels with the isolated Movi worker."""
import hashlib
import json
from probe_movi_targets import stage, PROJECT

ROOT = PROJECT / 'npunlock/tests/fixtures/npu4000'
DEST = ROOT / 'resident'
SOURCES = [ROOT / 'arithmetic.c', ROOT / 'resident_reduce.c', ROOT / 'resident_activation.c']


def main():
    DEST.mkdir(exist_ok=True)
    manifest = {'target': '4000xx', 'source_sha256': {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}, 'kernels': {}}
    kinds = ('sum', 'mean', 'relu-f16', 'relu-f32', 'relu_grad-f16', 'relu_grad-f32', 'rsqrt-f32')
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
        source = prefix.encode() + helpers + body
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


if __name__ == '__main__':
    main()

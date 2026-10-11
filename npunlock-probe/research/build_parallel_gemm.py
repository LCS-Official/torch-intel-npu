"""Build one-tile/two-task and four-tile FP32 GEMM variants with existing tools."""
import hashlib
import json
from probe_movi_targets import stage, PROJECT
from npunlock.npu4000 import kernel_text

ROOT = PROJECT / "npunlock/tests/fixtures/npu4000"
DEST = ROOT / "gemm_parallel"


def main():
    DEST.mkdir(exist_ok=True)
    sources = [ROOT / "arithmetic.c", ROOT / "square_gemm_parallel.c"]
    manifest = {"target": "4000xx", "source_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, "kernels": {}}
    for lanes, fast, four in ((1, False, False), (4, False, False), (4, True, False), (4, True, True)):
        kind = 'square-f32-four-tiles' if four else ("square-f32-fast" if fast else f"square-f32-lanes{lanes}")
        helpers = sources[0].read_bytes().split(b"void controlled_act(unsigned p) {", 1)[0]
        if fast:
            for name in ('add32', 'mul32'):
                helpers = helpers.replace(f'static inline float {name}('.encode(),
                    f'static __attribute__((noinline)) float {name}('.encode())
        source = (f"#define GEMM_BLOCK4 {int(lanes == 4)}\n#define GEMM_FAST {int(fast)}\n#define GEMM_FOUR_TILE {int(four)}\n".encode() +
                  helpers + sources[1].read_bytes())
        assembly, ok = stage("compile", "parallel-"+kind, "moviCompile64.dll", 1,
            ["moviCompile.dll", "-cc1", "-triple", "shave", "-target-cpu", "4000xx",
             "-S", "-O2", "-ffp-contract=off", "-x", "c", "entry.c", "-o", "-"], source)
        assert ok, kind+" compile failed"
        obj, ok = stage("assemble", "parallel-"+kind, "moviAsm64.dll", 2,
            ["moviAsm.dll", "--cv", "4000xx", "--noSPrefixing"], assembly)
        assert ok, kind+" assembly failed"
        elf, ok = stage("link", "parallel-"+kind, "moviLLD64.dll", 3,
            ["moviLLD.dll", "-flavor", "gnu", "-EL", "-e", "controlled_act",
             "-z", "max-page-size=0x10", "--gc-sections"],
            (obj, (PROJECT / "npunlock/src/shavecc/shave_kernel.ld").read_bytes()))
        assert ok, kind+" link failed"
        code = kernel_text(elf)
        assert len(code) <= 21216, "Kernel exceeds Power window"
        (DEST / f"{kind}.elf").write_bytes(elf)
        (DEST / f"{kind}.s").write_bytes(assembly)
        manifest["kernels"][kind] = {"elf_sha256": hashlib.sha256(elf).hexdigest(),
            "elf_bytes": len(elf), "text_bytes": len(code),
            "generated_source_sha256": hashlib.sha256(source).hexdigest(),
            "normal_ops_inline": fast, "output_lanes": lanes,
            "assembly_sha256": hashlib.sha256(assembly).hexdigest()}
        print({"status": "built", "kind": kind, "text_bytes": len(code)}, flush=True)
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Build the bounded FP32 square GEMM prototype with the existing toolchain."""
import hashlib
import json
from probe_movi_targets import stage, PROJECT
from npunlock.npu4000 import kernel_text

ROOT = PROJECT / "npunlock/tests/fixtures/npu4000"
DEST = ROOT / "gemm"


def main():
    DEST.mkdir(exist_ok=True)
    sources = [ROOT / "arithmetic.c", ROOT / "square_gemm.c"]
    source = sources[0].read_bytes().split(b"void controlled_act(unsigned p) {", 1)[0] + sources[1].read_bytes()
    assembly, ok = stage("compile", "square-gemm-f32", "moviCompile64.dll", 1,
        ["moviCompile.dll", "-cc1", "-triple", "shave", "-target-cpu", "4000xx",
         "-S", "-O2", "-ffp-contract=off", "-x", "c", "entry.c", "-o", "-"], source)
    assert ok, "GEMM compilation failed"
    obj, ok = stage("assemble", "square-gemm-f32", "moviAsm64.dll", 2,
        ["moviAsm.dll", "--cv", "4000xx", "--noSPrefixing"], assembly)
    assert ok, "GEMM assembly failed"
    elf, ok = stage("link", "square-gemm-f32", "moviLLD64.dll", 3,
        ["moviLLD.dll", "-flavor", "gnu", "-EL", "-e", "controlled_act",
         "-z", "max-page-size=0x10", "--gc-sections"],
        (obj, (PROJECT / "npunlock/src/shavecc/shave_kernel.ld").read_bytes()))
    assert ok, "GEMM link failed"
    text = kernel_text(elf)
    assert len(text) <= 21216, "GEMM does not fit the verified Power code window"
    (DEST / "square-f32.elf").write_bytes(elf)
    (DEST / "square-f32.s").write_bytes(assembly)
    manifest = {"target": "4000xx", "source_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "kernels": {"square-f32": {"elf_sha256": hashlib.sha256(elf).hexdigest(),
                                    "elf_bytes": len(elf), "text_bytes": len(text),
                                    "assembly_sha256": hashlib.sha256(assembly).hexdigest()}}}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print({"status": "built", "text_bytes": len(text)}, flush=True)


if __name__ == "__main__":
    main()

"""Rebuild bounded 4000xx arithmetic kernels via the existing isolated Movi worker."""
import hashlib
import json
from pathlib import Path
import sys
from probe_movi_targets import stage, PROJECT

DEST = PROJECT / "npunlock/tests/fixtures/npu4000/precision"
SOURCE = PROJECT / "npunlock/tests/fixtures/npu4000/arithmetic.c"
OPS = ("add1", "neg", "abs", "add", "sub", "mul", "div")


def main():
    DEST.mkdir(exist_ok=True)
    manifest = {"target": "4000xx", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "kernels": {}}
    for dtype in ("f16", "f32"):
        operations = list(enumerate(OPS)) + [(7 if dtype == "f16" else 8, "cast"), (9, "sign")]
        for operation, name in operations:
            stem = f"{name}-{dtype}"
            prefix = f"#define TYPE {'float' if dtype == 'f32' else 'unsigned short'}\n#define DTYPE {1 if dtype == 'f32' else 2}\n#define OP {operation}\n"
            source = prefix.encode() + SOURCE.read_bytes()
            assembly, ok = stage("compile", stem, "moviCompile64.dll", 1,
                ["moviCompile.dll", "-cc1", "-triple", "shave", "-target-cpu", "4000xx",
                 "-S", "-O2", "-ffp-contract=off", "-x", "c", "entry.c", "-o", "-"], source)
            if not ok:
                raise RuntimeError(f"Compile failed: {stem}")
            (DEST / f"{stem}.s").write_bytes(assembly)
            obj, ok = stage("assemble", stem, "moviAsm64.dll", 2,
                            ["moviAsm.dll", "--cv", "4000xx", "--noSPrefixing"], assembly)
            if not ok:
                raise RuntimeError(f"Assembly failed: {stem}")
            script = (PROJECT / "npunlock/src/shavecc/shave_kernel.ld").read_bytes()
            elf, ok = stage("link", stem, "moviLLD64.dll", 3,
                ["moviLLD.dll", "-flavor", "gnu", "-EL", "-e", "controlled_act",
                 "-z", "max-page-size=0x10", "--gc-sections"], (obj, script))
            if not ok:
                raise RuntimeError(f"Link failed: {stem}")
            (DEST / f"{stem}.elf").write_bytes(elf)
            manifest["kernels"][stem] = {"elf_sha256": hashlib.sha256(elf).hexdigest(),
                "assembly_sha256": hashlib.sha256(assembly).hexdigest(), "elf_bytes": len(elf)}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

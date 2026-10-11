"""Compile/assemble minimal C offline through npunlock's isolated worker."""
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
PROJECT = ROOT.parent.parent
TOOLS = PROJECT / "npunlock-probe/toolchain/Driver/MVC_DEPEND/bin"
WORKER = PROJECT / ".conda/npunlock/Lib/site-packages/npunlock/_bin/npunlock_worker.exe"


def stage(name, target, dll, stage_number, arguments, source):
    dll_path = str(TOOLS / dll).encode()
    args = [arg.encode() for arg in arguments]
    inputs = list(source) if isinstance(source, tuple) else [source]
    request = struct.pack("<8I", 0x4B57504E, 1, stage_number, len(args), len(inputs), len(dll_path), 0, 0)
    request += dll_path + b"".join(struct.pack("<I", len(arg)) + arg for arg in args)
    request += b"".join(struct.pack("<Q", len(part)) + part for part in inputs)
    with (OUT / f"{target}-{name}-response.bin").open("w+b") as response:
        handle = msvcrt.get_osfhandle(response.fileno())
        os.set_handle_inheritable(handle, True)
        startup = subprocess.STARTUPINFO()
        startup.lpAttributeList = {"handle_list": [handle]}
        try:
            process = subprocess.run([str(WORKER), "movi", "--response-handle", str(handle)],
                                     input=request, capture_output=True, timeout=20,
                                     startupinfo=startup, close_fds=True)
        finally:
            os.set_handle_inheritable(handle, False)
        response.seek(0)
        raw = response.read()
    result = {"target": target, "stage": name, "exit_code": process.returncode,
              "stdout": process.stdout.decode(errors="replace"),
              "stderr": process.stderr.decode(errors="replace")}
    if len(raw) >= 32:
        magic, version, status, tool_return, size, diag_size = struct.unpack_from("<IIIiQQ", raw)
        assert magic == 0x5257504E and version == 1 and len(raw) == 32 + size + diag_size
        output = raw[32:32 + size]
        result.update(worker_status=status, tool_return=tool_return, output_bytes=size,
                      diagnostic=raw[32 + size:].decode(errors="replace"),
                      output_sha256=hashlib.sha256(output).hexdigest())
    else:
        output = b""
        result["error"] = "Worker produced no valid response header"
    (OUT / f"{target}-{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return output, result.get("worker_status") == 0 and result.get("tool_return") == 0


if __name__ == "__main__":
    source = b"""
typedef struct { unsigned lo, hi, isStatic, rank, dims, strides, dtype, orderLo, orderHi, location; } MemRef;
void controlled_act(unsigned p) {
    MemRef *m = (MemRef*)p;
    if (!p || !m[0].rank || m[0].rank > 15) return;
    unsigned *dims = (unsigned*)m[0].dims;
    unsigned n = 1;
    for (unsigned i = 0; i < m[0].rank; ++i) n *= dims[i];
    const __fp16 *in = (const __fp16*)m[0].lo;
    __fp16 *out = (__fp16*)m[1].lo;
    for (unsigned i = 0; i < n; ++i) out[i] = (__fp16)((float)in[i] + 1.0f);
}
"""
    (OUT / "add1-memref.c").write_bytes(source)
    for target in ("3720xx", "4000xx", "invalid_target_for_probe"):
        assembly, ok = stage("compile", target, "moviCompile64.dll", 1,
                             ["moviCompile.dll", "-cc1", "-triple", "shave", "-target-cpu", target,
                              "-S", "-O2", "-x", "c", "entry.c", "-o", "-"], source)
        if ok:
            (OUT / f"{target}.s").write_bytes(assembly)
            obj, ok = stage("assemble", target, "moviAsm64.dll", 2,
                            ["moviAsm.dll", "--cv", target, "--noSPrefixing"], assembly)
            if ok:
                assert obj.startswith(b"\x7fELF"), "Assembler output must be ELF"
                (OUT / f"{target}.o").write_bytes(obj)
                script = (PROJECT / "npunlock/src/shavecc/shave_kernel.ld").read_bytes()
                elf, ok = stage("link", target, "moviLLD64.dll", 3,
                                ["moviLLD.dll", "-flavor", "gnu", "-EL", "-e", "controlled_act",
                                 "-z", "max-page-size=0x10", "--gc-sections"], (obj, script))
                if ok:
                    assert elf.startswith(b"\x7fELF") and struct.unpack_from("<H", elf, 16)[0] == 2
                    (OUT / f"{target}.elf").write_bytes(elf)
        if target == "3720xx":
            assert ok, "Known-good target must pass; otherwise the probe is inconclusive"

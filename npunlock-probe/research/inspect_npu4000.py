"""Read-only CLI for the reusable npunlock NPU4000 parser."""
import argparse
import json
from pathlib import Path
import struct
from npunlock.npu4000 import Elf, bounded, inspect, string_at

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("blob", nargs="?", type=Path, default=ROOT.parent / "results/custom.blob")
    args = parser.parse_args()
    data = args.blob.read_bytes()
    result = inspect(data)
    try:
        inspect(data[:63])
    except ValueError:
        pass
    else:
        raise AssertionError("Truncated ELF must be rejected")
    corrupted = bytearray(data)
    shoff = struct.unpack_from("<Q", data, 40)[0]
    # The valid string table is file-backed: extending it beyond EOF must fail.
    struct.pack_into("<Q", corrupted, shoff + 64 + 32, len(data) + 1)
    try:
        Elf(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-bounds file-backed section must be rejected")
    output = ROOT / "results" / "npu4000-inspection.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Ranges: {len(result['ranges'])}; invocations: {len(result['invocations'])}")
    for invocation in result["invocations"]:
        refs = invocation.get("unary_memrefs", [])
        print(invocation["section"], invocation["index"], "tile", invocation["tile"],
              "range", invocation["range_index"], "dims", [r["dims"] for r in refs],
              "elements", [r["elements"] for r in refs])

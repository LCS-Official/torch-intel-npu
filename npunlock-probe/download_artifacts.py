"""Fetch the project's Windows release and the documented MoviTools payload."""
from pathlib import Path
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parent
DOWNLOADS = ROOT / "downloads"
DOWNLOADS.mkdir(exist_ok=True)
RELEASE = "https://github.com/hsfzxjy/npunlock/releases/download/2026-09-28.00/"


def fetch(url, filename, expected=None):
    path = DOWNLOADS / filename
    if not path.exists():
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with path.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    output.write(block)
        print(f"Downloaded {filename}: {path.stat().st_size} bytes", flush=True)
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    if expected is not None and digest != expected.lower():
        raise RuntimeError(f"SHA256 mismatch for {filename}: {digest}")
    return {"file": filename, "url": url, "sha256": digest, "bytes": path.stat().st_size}


records = [fetch(RELEASE + "SHA256SUMS", "SHA256SUMS")]
checksums = {}
for line in (DOWNLOADS / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    checksums[name.lstrip("*")] = digest
for name in ("npunlock-0.1.0-py3-none-win_amd64.whl", "npunlock-windows-x64.zip"):
    records.append(fetch(RELEASE + name, name, checksums[name]))
records.append(fetch(
    "https://download.lenovo.com/pccbbs/mobiles/r2gvp04w_v2.exe",
    "r2gvp04w_v2.exe",
    "cf767eea31debaa199b084cce06eb82443aa251d5963ef97030f5219b9fe28db",
))
(ROOT / "downloads.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
print("All artifact hashes verified.", flush=True)

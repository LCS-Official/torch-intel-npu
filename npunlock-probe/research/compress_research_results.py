"""Lossless Windows LZX compression of research evidence, with byte-hash checks.

Only regular files >=1 MiB beneath this workspace's results directory are used.
Junctions are skipped; active runtime and installed environments are untouched.
"""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'npunlock-probe/research/results'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);size=kernel.GetCompressedFileSizeW
    size.argtypes=(wintypes.LPCWSTR,ctypes.POINTER(wintypes.DWORD));size.restype=wintypes.DWORD
    def allocated(path):
        high=wintypes.DWORD();ctypes.set_last_error(0);low=size(str(path),ctypes.byref(high))
        if low==0xffffffff and ctypes.get_last_error():raise ctypes.WinError(ctypes.get_last_error())
        return (high.value<<32)|low
    paths=[]
    for directory,dirs,files in os.walk(RESULTS,followlinks=False):
        dirs[:]=[name for name in dirs if not (Path(directory)/name).lstat().st_file_attributes&0x400]
        for name in files:
            path=Path(directory)/name;stat=path.lstat()
            if stat.st_size>=1024*1024 and not stat.st_file_attributes&0x400:
                assert path.resolve().is_relative_to(RESULTS.resolve())
                paths.append(path)
    start=time.perf_counter();records=[];before=shutil.disk_usage(ROOT).free
    journal=ROOT/'npunlock-probe/research/workspace-compression.jsonl'
    with journal.open('w',encoding='utf-8') as stream:
        for i,path in enumerate(sorted(paths)):
            physical=allocated(path);digest=sha(path)
            result=subprocess.run(['compact.exe','/C','/EXE:LZX','/I','/Q',str(path)],capture_output=True)
            assert result.returncode==0,(str(path),result.stdout.decode(errors='replace'),result.stderr.decode(errors='replace'))
            assert sha(path)==digest,('Evidence bytes changed',str(path))
            row=dict(path=str(path.relative_to(ROOT)),logical_bytes=path.stat().st_size,
                before_allocated_bytes=physical,after_allocated_bytes=allocated(path),sha256=digest)
            records.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if (i+1)%25==0 or i+1==len(paths):print('verified',i+1,'/',len(paths),'saved GiB',
                round(sum(r['before_allocated_bytes']-r['after_allocated_bytes'] for r in records)/1024**3,3),flush=True)
    report=dict(status='passed',method='Windows compact /C /EXE:LZX',files=len(records),
        saved_allocated_bytes=sum(r['before_allocated_bytes']-r['after_allocated_bytes'] for r in records),
        sha256_checks=len(records),disk_free_before=before,disk_free_after=shutil.disk_usage(ROOT).free,
        elapsed_seconds=time.perf_counter()-start,journal_sha256=sha(journal),npu_calls=0)
    (ROOT/'npunlock-probe/research/workspace-compression.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

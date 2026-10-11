"""Read-only Windows workspace size audit; skip junctions and count hardlinks once."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
from collections import defaultdict
import shutil

ROOT=Path(__file__).resolve().parents[2]


def main():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    size=kernel.GetCompressedFileSizeW;size.argtypes=(wintypes.LPCWSTR,ctypes.POINTER(wintypes.DWORD));size.restype=wintypes.DWORD
    def allocated(path):
        high=wintypes.DWORD();ctypes.set_last_error(0);low=size(str(path),ctypes.byref(high))
        if low==0xffffffff and ctypes.get_last_error():raise ctypes.WinError(ctypes.get_last_error())
        return (high.value<<32)|low
    groups=defaultdict(lambda:dict(files=0,logical_bytes=0,allocated_bytes=0));seen=set();large=[];links=[];errors=[]
    unique=0;hardlinked=0
    for directory,dirs,files in os.walk(ROOT,followlinks=False):
        dirs.sort();kept=[]
        for name in dirs:
            path=Path(directory)/name
            if path.lstat().st_file_attributes&0x400:links.append(str(path))
            else:kept.append(name)
        dirs[:]=kept
        for name in files:
            path=Path(directory)/name
            try:
                stat=path.stat();physical=allocated(path);identity=(stat.st_dev,stat.st_ino)
                if identity not in seen:unique+=physical;seen.add(identity)
                if stat.st_nlink>1:hardlinked+=1
                relative=path.relative_to(ROOT);parts=relative.parts
                for depth in range(1,min(3,len(parts)-1)+1):
                    row=groups['/'.join(parts[:depth])];row['files']+=1;row['logical_bytes']+=stat.st_size;row['allocated_bytes']+=physical
                large.append(dict(path=str(relative),logical_bytes=stat.st_size,allocated_bytes=physical,hardlinks=stat.st_nlink))
            except OSError as error:errors.append(dict(path=str(path),error=str(error)))
    disk=shutil.disk_usage(ROOT)
    report=dict(root=str(ROOT),files=len(large),unique_allocated_bytes=unique,hardlinked_file_entries=hardlinked,
        disk_total_bytes=disk.total,disk_used_bytes=disk.used,disk_free_bytes=disk.free,
        logical_bytes=sum(r['logical_bytes'] for r in large),groups=dict(sorted(groups.items(),key=lambda p:p[1]['logical_bytes'],reverse=True)),
        largest_files=sorted(large,key=lambda p:p['logical_bytes'],reverse=True)[:35],skipped_reparse_points=links,errors=errors)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

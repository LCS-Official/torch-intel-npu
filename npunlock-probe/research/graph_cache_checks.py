"""No-device checks of cache identity, corruption rejection and bounded writes."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
from types import SimpleNamespace
import torch_npu_graph_cache as cache


def main():
    checks=[]
    profile=dict(vendor=0x8086,device=0x643e,driver=1005540,compiler=[8,3],graph_extension=1)
    key='a'*64;blob=b'ELF-test-payload'*100;reports=(b'{"operation":"test"}',)
    fingerprint=cache._key(b'graph',profile,True)
    assert fingerprint!=cache._key(b'changed',profile,True)
    assert fingerprint!=cache._key(b'graph',dict(profile,driver=1005541),True)
    assert fingerprint!=cache._key(b'graph',dict(profile,compiler=[8,4]),True)
    assert fingerprint!=cache._key(b'graph',profile,False)
    checks.extend(['graph fingerprint','driver invalidation','compiler invalidation','patch mode invalidation'])
    with tempfile.TemporaryDirectory(prefix='graph-cache-check-',dir=Path(__file__).parent) as name:
        directory=Path(name);path=directory/(key+'.npc')
        assert cache._write(directory,key,profile,blob,reports)
        assert cache._read(path,key,profile)==(blob,reports);checks.append('cache roundtrip')
        with zipfile.ZipFile(path) as archive:
            assert all(member.compress_type==zipfile.ZIP_STORED for member in archive.infolist())
        checks.append('uncompressed write format')
        assert not (directory/'.writer.lock').exists() and not list(directory.glob('*.tmp'))
        checks.append('atomic write cleanup')
        def rejected(label):
            try:cache._read(path,key,profile)
            except (ValueError,KeyError,TypeError,zipfile.BadZipFile):checks.append(label)
            else:raise AssertionError(label)
        with zipfile.ZipFile(path) as archive:manifest=json.loads(archive.read('manifest.json'))
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            archive.writestr('manifest.json',json.dumps(manifest));archive.writestr('graph.blob',blob)
        assert cache._read(path,key,profile)==(blob,reports);checks.append('legacy deflate read compatible')
        def entry(value,payload=blob,extra=False):
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('manifest.json',json.dumps(value));archive.writestr('graph.blob',payload)
                if extra:archive.writestr('unexpected','x')
        for field,value in [('key','b'*64),('profile',dict(profile,driver=1005541)),('version',2),('patch_reports',[]),('patch_reports',['[]'])]:
            entry(dict(manifest,**{field:value}));rejected(field+' mismatch '+str(value))
        entry(manifest,blob+b'bad');rejected('checksum mismatch')
        entry(manifest,extra=True);rejected('unexpected ZIP member')
        path.write_bytes(b'truncated');rejected('truncated entry')
        old=cache._LIMIT;cache._LIMIT=1
        try:assert not cache._write(directory,key,profile,blob,reports)
        finally:cache._LIMIT=old
        checks.append('capacity bound')
        assert not (directory/'.writer.lock').exists()
        (directory/'.writer.lock').write_text('held')
        assert not cache._write(directory,key,profile,blob,reports);checks.append('concurrent writer skipped')
        (directory/'.writer.lock').unlink()
        sentinel=object();calls=[]
        def custom(*args,**kwargs):calls.append((args,kwargs));return sentinel
        fake=SimpleNamespace(_native=object(),_driver_version=1005540,
            _info=SimpleNamespace(device_vendor_id=0x8086,device_id=0x643e),
            _bootstrap_program=SimpleNamespace(ir_provenance=SimpleNamespace(compiler_version=(8,3),graph_extension_version=1)))
        previous=os.environ.get('NPUNLOCK_GRAPH_CACHE');os.environ['NPUNLOCK_GRAPH_CACHE']=str(directory/'hook')
        try:
            assert cache.compile_fused_cached(fake,None,b'xml',b'weights',b'kernel',graph_digest=b'graph',
                whole_layer=True,select_kernel_elf=None,compiler_factory=custom) is sentinel
        finally:
            if previous is None:os.environ.pop('NPUNLOCK_GRAPH_CACHE',None)
            else:os.environ['NPUNLOCK_GRAPH_CACHE']=previous
        assert len(calls)==1 and calls[0][1]['libraries'] is fake._native and not (directory/'hook').exists()
        checks.append('custom compiler bypass preserves research hooks')
    output=Path(__file__).parent/'results/model-cross-device-followup/cache-checks.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(status='passed',checks=checks,source_sha256=hashlib.sha256(Path(cache.__file__).read_bytes()).hexdigest()),indent=2))
    print(len(checks),'cache checks passed',flush=True)


if __name__=='__main__':main()

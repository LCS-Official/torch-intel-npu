"""Compare ZIP storage formats with identical validated graph blobs and identities."""
import json
import shutil
import zipfile
import torch_npu_graph_cache as cache
from startup_first_token import OUT


if __name__=='__main__':
    source=OUT/'cache';target=OUT/'cache-stored'
    assert not target.exists(),'Use a fresh experimental directory'
    files=sorted(source.glob('*.npc'));assert len(files)==30
    target.mkdir()
    for path in files:
        with zipfile.ZipFile(path) as archive:manifest=json.loads(archive.read('manifest.json'))
        key=manifest['key'];profile=manifest['profile']
        blob,reports=cache._read(path,key,profile)
        assert shutil.disk_usage(target).free>=cache._RESERVE+len(blob)+1024*1024
        destination=target/path.name
        with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('manifest.json',json.dumps(manifest));archive.writestr('graph.blob',blob)
        assert cache._read(destination,key,profile)==(blob,reports)
    size=sum(p.stat().st_size for p in target.glob('*.npc'));assert size<=cache._LIMIT
    print(json.dumps(dict(entries=len(files),compressed_bytes=sum(p.stat().st_size for p in files),stored_bytes=size,all_blobs_equal=True)))

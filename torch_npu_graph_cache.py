"""Opt-in, bounded disk cache for the verified frozen SmolLM fused graphs."""
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

_LIMIT=512*1024*1024
_RESERVE=256*1024*1024
_BLOB_LIMIT=64*1024*1024
_counts=dict(hits=0,misses=0,compiles=0,rejected=0,writes=0,write_skipped=0,ir_skipped=0,read_ms=0.,write_ms=0.,bytes_loaded=0,bytes_written=0)


def stats():
    return dict(_counts, directory=os.environ.get('NPUNLOCK_GRAPH_CACHE') or None,
                limit_bytes=_LIMIT, reserve_bytes=_RESERVE)


def _key(graph_digest,profile,whole_layer):
    npu4000=importlib.import_module('npunlock.npu4000')
    recipe=hashlib.sha256(Path(__file__).read_bytes()+Path(npu4000.__file__).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(dict(version=1,graph=graph_digest.hex(),profile=profile,
        recipe=recipe,whole_layer=whole_layer),sort_keys=True).encode()).hexdigest()


def _contract(graph):
    return {name:[dict(name=t.name,shape=list(t.shape),dtype=t.dtype) for t in getattr(graph,name)]
            for name in ('inputs','outputs')}


def prefill_cache_graph(layer,shape,kernel,compiler_factory):
    """Binding metadata only; a cache miss must build the real graph before compiling."""
    import npunlock as s
    import torch_npu_smollm as smol
    from npunlock.npu4000 import compile_fused_rmsnorm
    if (not os.environ.get('NPUNLOCK_GRAPH_CACHE') or layer.mode!='fp16'
        or type(layer) is not smol.FrozenSmolLayer or compiler_factory is not compile_fused_rmsnorm
        or getattr(layer.graph,'__func__',None) is not smol._PREFILL_GRAPH
        or s.serialize_ir is not smol._PREFILL_SERIALIZER):return None
    shape=layer._shape(shape);rows=shape[0]*shape[1]
    inputs=[s.input('x',shape=shape,dtype='f16'),
            *[s.input(name,shape=(max(2,rows),576),dtype='f16') for name in ('gamma','ffn_gamma')]]
    outputs=[s.op('CachedNativeBinding',*inputs,_shape=out_shape,_dtype='f16',_name=name)
             for name,out_shape in [('y',shape),('cache_k',(shape[0],3,64,64)),('cache_v',(shape[0],3,64,64))]]
    graph=s.Graph(inputs,outputs,name='cached_smollm_prefill_binding')
    digest=hashlib.sha256(json.dumps(dict(route='fp16-layer-prefill-v1',contract=_contract(graph)),sort_keys=True).encode())
    # Invalidate for graph/serializer changes, without serializing model constants.
    for module in (smol,s,*(importlib.import_module('npunlock.'+name) for name in ('graph','ir','tensor'))):
        digest.update(hashlib.sha256(Path(module.__file__).read_bytes()).digest())
    for name,(data,wshape) in sorted(layer._weights.items()):
        # Norm weights are runtime inputs, so different gamma values may share a graph.
        if len(wshape)!=2:continue
        digest.update(json.dumps((name,wshape,len(data))).encode());digest.update(data)
    for data in (layer._cos,layer._sin,kernel):
        digest.update(len(data).to_bytes(8,'little'));digest.update(data)
    return graph,digest.digest()


def _read(path,key,profile,*,contract=None):
    if path.stat().st_size>_BLOB_LIMIT:raise ValueError('Oversized graph cache entry')
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist())!=['graph.blob','manifest.json']:raise ValueError('Unexpected graph cache members')
        if archive.getinfo('manifest.json').file_size>1024*1024 or archive.getinfo('graph.blob').file_size>_BLOB_LIMIT:
            raise ValueError('Oversized decompressed graph cache entry')
        manifest=json.loads(archive.read('manifest.json'))
        if manifest['version']!=1 or manifest['key']!=key or manifest['profile']!=profile:
            raise ValueError('Graph cache identity mismatch')
        if contract is not None and manifest.get('io_contract')!=contract:
            raise ValueError('Graph cache binding contract mismatch')
        blob=archive.read('graph.blob')
        if not blob or hashlib.sha256(blob).hexdigest()!=manifest['blob_sha256']:
            raise ValueError('Graph cache checksum mismatch')
        reports=manifest['patch_reports']
        if not isinstance(reports,list) or not reports or not all(isinstance(v,str) for v in reports):
            raise ValueError('Invalid graph patch reports')
        for report in reports:
            if not isinstance(json.loads(report),dict):raise ValueError('Invalid graph patch report')
    return blob,tuple(report.encode() for report in reports)


def _write(directory,key,profile,blob,reports,*,contract=None):
    if len(blob)>_BLOB_LIMIT:return False
    directory.mkdir(parents=True,exist_ok=True)
    lock=directory/'.writer.lock'
    # shortcut: after a writer crash, remove its stale lock with no writers running; reads still work.
    try:descriptor=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:return False
    temporary=None
    try:
        os.close(descriptor)
        path=directory/(key+'.npc')
        manifest=dict(version=1,key=key,profile=profile,blob_sha256=hashlib.sha256(blob).hexdigest(),
                      patch_reports=[report.decode() for report in reports])
        if contract is not None:manifest['io_contract']=contract
        # Reserve space before creating a temporary file; optional caching must not fill the drive.
        used=sum(p.stat().st_size for p in directory.glob('*.npc'))
        previous=path.stat().st_size if path.exists() else 0
        if used-previous+len(blob)+1024*1024>_LIMIT or shutil.disk_usage(directory).free<_RESERVE+len(blob)+1024*1024:
            return False
        with tempfile.NamedTemporaryFile(dir=directory,suffix='.tmp',delete=False) as file:
            temporary=Path(file.name)
        # Avoid startup decompression; this opt-in cache remains bounded by _LIMIT.
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('manifest.json',json.dumps(manifest))
            archive.writestr('graph.blob',blob)
        size=temporary.stat().st_size
        if used-previous+size>_LIMIT:return False
        os.replace(temporary,path);temporary=None
        _counts['bytes_written']+=size
        return True
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def compile_fused_cached(npu,graph,xml,weights,kernel,*,graph_digest,whole_layer,select_kernel_elf,compiler_factory,graph_factory=None):
    from npunlock import load_native
    from npunlock.npu4000 import compile_fused_rmsnorm
    # Research hooks replace the compiler; preserve their behavior and provenance.
    configured=os.environ.get('NPUNLOCK_GRAPH_CACHE') if compiler_factory is compile_fused_rmsnorm else None
    profile=dict(vendor=npu._info.device_vendor_id,device=npu._info.device_id,driver=npu._driver_version,
        compiler=list(npu._bootstrap_program.ir_provenance.compiler_version),
        graph_extension=npu._bootstrap_program.ir_provenance.graph_extension_version)
    contract=_contract(graph) if graph_factory is not None else None
    if configured:
        directory=Path(configured).expanduser().resolve();key=_key(graph_digest,profile,whole_layer)
        path=directory/(key+'.npc');start=time.perf_counter_ns()
        try:
            blob,reports=_read(path,key,profile,contract=contract)
            program=replace(load_native(blob,graph=graph,libraries=npu._native),patch_reports=reports)
            _counts['hits']+=1;_counts['bytes_loaded']+=len(blob)
            if graph_factory is not None:_counts['ir_skipped']+=1
            return program
        except FileNotFoundError:_counts['misses']+=1
        # This block only reads cache data; native session errors are never retried here.
        except Exception:_counts['rejected']+=1
        finally:_counts['read_ms']+=(time.perf_counter_ns()-start)/1e6
    if graph_factory is not None:
        graph,xml,weights=graph_factory()
        if _contract(graph)!=contract:raise ValueError('Built graph differs from cached binding contract')
    _counts['compiles']+=1
    program=compiler_factory(graph,xml,weights,kernel,libraries=npu._native,whole_layer=whole_layer,
                                  select_kernel_elf=select_kernel_elf)
    if configured:
        start=time.perf_counter_ns()
        try:
            saved=_write(directory,key,profile,program.graph_blob,program.patch_reports,contract=contract)
            _counts['writes' if saved else 'write_skipped']+=1
        except OSError:_counts['write_skipped']+=1
        finally:_counts['write_ms']+=(time.perf_counter_ns()-start)/1e6
    return program

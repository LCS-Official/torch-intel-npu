"""Isolated 64/128-block allocator controls for the resident 30-layer decoder."""
import argparse
import ctypes as ct
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import time
import types
import numpy as np
from shave_fp32_pipeline import metadata, snapshots, final, SHAPES, profiling_directory
from w8_quality_probe import ROOT, MODEL, sha

OUT=ROOT/'npunlock-probe/research/results/cache-blocks'
COUNTERS=('npu_calls','buffer_allocations','driver_allocations','allocation_reuses','host_upload_copies',
    'host_download_copies','host_device_copies','host_pack_copies','session_creations')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def delta(before,after):return {key:after[key]-before[key] for key in COUNTERS}


def build():
    before=OUT/'before';before.mkdir(parents=True,exist_ok=True)
    names=['torch_intel_npu.py','torch_npu_warm_profile.py','npunlock-probe/torch-device/backend.cpp',
        'npunlock-probe/torch-device/build.json','npunlock-probe/torch-device/_torch_intel_npu.pyd','build_torch_npu_device.ps1']
    if not (before/'sources.json').exists():
        state=metadata();state['sources'].update({name:sha(ROOT/name) for name in names})
        for name in state['sources']:
            path=before/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
        write(before/'sources.json',state)
    state=read(before/'sources.json')
    for name,digest in state['sources'].items():assert sha(before/name)==digest
    source=(before/'npunlock-probe/torch-device/backend.cpp').read_text(encoding='utf-8')
    marker='constexpr size_t max_cached_buffers = 64;';assert source.count(marker)==1
    source=source.replace(marker,'constexpr size_t max_cached_buffers = 128;\nsize_t diagnostic_cache_blocks = 64;')
    marker='s.cached.size() >= max_cached_buffers';assert source.count(marker)==1
    source=source.replace(marker,'s.cached.size() >= diagnostic_cache_blocks')
    marker='  m.attr("max_tensor_elements") = max_tensor_elements;';assert source.count(marker)==1
    source=source.replace(marker,marker+'''
  m.def("_diagnostic_cache_blocks", [](int64_t blocks) {
    std::lock_guard<std::recursive_mutex> lock(state().mutex);
    if (blocks != 0) {
      ready();
      TORCH_CHECK(blocks == 64 || blocks == 128, "Diagnostic capacity must be 64 or 128");
      empty_cache();
      diagnostic_cache_blocks = static_cast<size_t>(blocks);
    }
    return diagnostic_cache_blocks;
  }, pybind11::arg("blocks") = 0, pybind11::call_guard<pybind11::gil_scoped_release>());
''')
    directory=OUT/'diagnostic';directory.mkdir(exist_ok=True)
    (directory/'backend.cpp').write_text(source,encoding='utf-8')
    script=(before/'build_torch_npu_device.ps1').read_text(encoding='utf-8-sig')
    script=script.replace('$root = $PSScriptRoot',"$root = '"+str(ROOT).replace("'","''")+"'")
    marker="$output = Join-Path $root 'npunlock-probe\\torch-device'";assert script.count(marker)==1
    script=script.replace(marker,"$output = '"+str(directory).replace("'","''")+"'")
    path=directory/'build.ps1';path.write_text(script,encoding='utf-8')
    with (directory/'build.log').open('w',encoding='utf-8') as log:
        subprocess.run(['pwsh','-NoProfile','-File',str(path),'-Toolchain',str(ROOT/'npunlock-probe/toolchain/msvc')],stdout=log,stderr=log,timeout=180,check=True)
    info=read(directory/'build.json')
    assert info['source_sha256']==sha(directory/'backend.cpp') and info['extension_sha256']==sha(directory/'_torch_intel_npu.pyd')
    profile=profiling_directory(OUT/'host-runtime')
    bound=dict(info,session_sha256=profile['dll_sha256']);write(directory/'bound-build.json',bound)
    write(OUT/'build.json',dict(status='built',builder_sha256=sha(Path(__file__)),diagnostic=info,profiling=profile,
        policy='Only research PYD has a mutable 64/128 idle block limit; 16 MiB byte budget is unchanged.'))
    print('Isolated allocator extension built',flush=True)


def bind(host):
    import torch_npu_runtime
    directory=OUT/'diagnostic';info=read(OUT/'build.json')
    manifest=directory/('bound-build.json' if host else 'build.json')
    torch_npu_runtime.SESSION_DLL=OUT/'host-runtime/session_task_profile.dll' if host else ROOT/'npunlock-probe/session-runtime/npunlock_session.dll'
    source=(OUT/'before/torch_intel_npu.py').read_text(encoding='utf-8')
    source=source.replace('_EXT = _ROOT / "npunlock-probe/torch-device/_torch_intel_npu.pyd"',f'_EXT = Path({str(directory/"_torch_intel_npu.pyd")!r})')
    source=source.replace('_BUILD = _EXT.with_name("build.json")',f'_BUILD = Path({str(manifest)!r})')
    source=source.replace('"allocation_cache_limit_blocks": 64','"allocation_cache_limit_blocks": _C._diagnostic_cache_blocks()')
    module=types.ModuleType('torch_intel_npu');module.__file__=str(ROOT/'torch_intel_npu.py')
    sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
    assert read(manifest)['extension_sha256']==info['diagnostic']['extension_sha256']
    driver=None
    if host:
        driver=module._native.infer
        driver.warm_diag_reset.argtypes=[];driver.warm_diag_reset.restype=None
        driver.warm_diag_read.argtypes=[ct.POINTER(ct.c_uint64)]*2;driver.warm_diag_read.restype=None
    return module,driver


def probe(label,host):
    import torch
    npu,driver=bind(host)
    from torch_npu_smollm import FrozenSmolStack
    from transformers import AutoTokenizer
    from safetensors import safe_open
    from torch_npu_warm_profile import STAGES
    torch.set_num_threads(8)
    report=dict(status='running',label=label,host_instrumented=host,source_sha256=sha(Path(__file__)),
        build=read(OUT/'build.json'),frozen=read(OUT/'before/sources.json'),rows=[],checks=[])
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as file:embedding=file.get_tensor('model.embed_tokens.weight')
    arrays={}
    def work(batch,prefix):
        tag=f'b{batch}-s{prefix}';tokens=ids.repeat(1,(prefix+ids.shape[1])//ids.shape[1])[:,:prefix+1].repeat(batch,1)
        hidden=embedding[tokens].half().contiguous();x=hidden[:,:prefix].contiguous().to('intel_npu');one=hidden[:,prefix:].contiguous().to('intel_npu')
        stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=30,mode='fp16')
        print(label,tag,'preparing 30-layer graphs',flush=True)
        npu._C._diagnostic_cache_blocks(64)
        initial=stack.prefill(x);seed=initial[1];decoded=stack.decode(one,seed)
        anchor=[v.cpu().numpy().copy() for value in (initial,decoded) for v in snapshots(value)]
        for i,array in enumerate(anchor):arrays[f'{tag}/64/{i}']=array
        del initial,decoded
        npu._C._diagnostic_cache_blocks(128)
        initial=stack.prefill(x);decoded=stack.decode(one,seed)
        for i,value in enumerate(v for result in (initial,decoded) for v in snapshots(result)):
            array=value.cpu().numpy().copy();np.testing.assert_array_equal(array.view(np.uint8),anchor[i].view(np.uint8));arrays[f'{tag}/128/{i}']=array
        del initial,decoded,value
        graph_hashes={f'{i}-{local[-1]}':hashlib.sha256(npu._sessions[key][1].graph_blob).hexdigest()
            for i,layer in enumerate(stack.layers) for local,key in layer._keys.items()}
        assert len(graph_hashes)==60
        for operation in ('prefill','decode'):
            fn=(lambda:stack.prefill(x)) if operation=='prefill' else (lambda:stack.decode(one,seed))
            blocks=[];pairs=[]
            for round_index in range(9):
                order=(64,128,128,64) if round_index%2==0 else (128,64,64,128)
                pair={64:[],128:[]}
                for capacity in order:
                    npu._C._diagnostic_cache_blocks(capacity)
                    for _ in range(3):value=fn();del value
                    before=npu.stats();samples=[]
                    for _ in range(5):
                        if driver:driver.warm_diag_reset()
                        start=time.perf_counter_ns();value=fn();returned=time.perf_counter_ns();del value;freed=time.perf_counter_ns()
                        sample=dict(api_ms=(returned-start)/1e6,cleanup_ms=(freed-returned)/1e6,total_ms=(freed-start)/1e6)
                        if driver:
                            ns=(ct.c_uint64*9)();counts=(ct.c_uint64*9)();driver.warm_diag_read(ns,counts)
                            sample.update(stages_ms=dict(zip(STAGES,[n/1e6 for n in ns])),stage_counts=dict(zip(STAGES,list(counts))))
                            assert all(sample['stage_counts'][stage]==30 for stage in STAGES[3:])
                        samples.append(sample);pair[capacity].append(sample['total_ms'])
                    after=npu.stats();change=delta(before,after)
                    assert change['npu_calls']==5*30
                    assert all(change[k]==0 for k in ('host_upload_copies','host_download_copies','host_device_copies','host_pack_copies','session_creations'))
                    assert after['cached_buffer_bytes']<=16*1024*1024 and after['cached_buffer_blocks']<=capacity
                    blocks.append(dict(round=round_index,capacity=capacity,samples=samples,delta=change,cached_blocks=after['cached_buffer_blocks'],cached_bytes=after['cached_buffer_bytes']))
                pairs.append({str(k):st.mean(v) for k,v in pair.items()})
            medians={str(capacity):{field:st.median(s[field] for b in blocks if b['capacity']==capacity for s in b['samples'])
                for field in ('api_ms','cleanup_ms','total_ms')} for capacity in (64,128)}
            allocations={str(capacity):sum(b['delta']['driver_allocations'] for b in blocks if b['capacity']==capacity) for capacity in (64,128)}
            assert allocations['128']==0 and allocations['64']==(28*90 if operation=='decode' else 0)
            row=dict(tag=tag,operation=operation,blocks=blocks,pairs_total_ms=pairs,median_ms=medians,driver_allocations=allocations,
                positive_rounds=sum(p['128']<p['64'] for p in pairs),total_reduction_percent=100*(1-medians['128']['total_ms']/medians['64']['total_ms']))
            report['rows'].append(row);print(tag,operation,medians,allocations,'wins',row['positive_rounds'],flush=True)
        for i,cache in enumerate(seed):
            for j,value in enumerate((cache.keys,cache.values,cache._position)):
                array=value.cpu().numpy().copy();np.testing.assert_array_equal(array.view(np.uint8),anchor[1+i*3+j].view(np.uint8));arrays[f'{tag}/preserved/{i*3+j}']=array
        assert graph_hashes=={f'{i}-{local[-1]}':hashlib.sha256(npu._sessions[key][1].graph_blob).hexdigest()
            for i,layer in enumerate(stack.layers) for local,key in layer._keys.items()}
        report['checks'].append(dict(tag=tag,bit_equal_arrays=182,preserved_seed_arrays=90,graph_sha256=graph_hashes))
    with torch.no_grad():
        for batch,prefix in SHAPES:work(batch,prefix)
        final(npu,report)
    path=OUT/(label+'-outputs.npz');np.savez_compressed(path,**arrays)
    report.update(status='passed',outputs_sha256=sha(path));write(OUT/(label+'.json'),report)


def confirm(label):
    import shave_fp32_pipeline as common
    common.OUT=OUT/'production';common.OUT.mkdir(exist_ok=True);common.benchmark(label)
    report=read(common.OUT/(label+'.json'))
    assert report['final_stats']['allocation_cache_limit_blocks']==128
    assert all(row['warm_delta']['driver_allocations']==0 for row in report['rows'])
    with np.load(common.OUT/(label+'-outputs.npz')) as actual,np.load(OUT/'plain-outputs.npz') as expected:
        for batch,prefix in SHAPES:
            tag=f'b{batch}-s{prefix}'
            for i in range(182):np.testing.assert_array_equal(actual[f'{tag}/{i}'].view(np.uint8),expected[f'{tag}/128/{i}'].view(np.uint8))
    print('Actual fixed 128-block production PYD: zero warm allocations, 364 arrays bit-equal',flush=True)


def check(kind):
    path=OUT/'checks';path.mkdir(exist_ok=True)
    if kind=='guards':
        import torch_npu_warm_profile as common
        common.OUT=path;common.guards()
    elif kind=='order':
        import decode_cache_checks as common
        common.OUT=path;common.order()
    elif kind=='failure':
        import allocator_failure_probe as failure
        previous=ROOT/'npunlock-probe/research/results/warm-allocation'
        build_info=read(previous/'failure-build.json')
        assert sha(previous/'session_failure.dll')==build_info['dll_sha256']
        assert sha(previous/'session_failure.c')==build_info['source_sha256']
        shutil.copyfile(previous/'session_failure.dll',path/'session_failure.dll');write(path/'failure-build.json',build_info)
        failure.OUT=path;failure.run()
    else:
        import importlib
        demo=importlib.import_module('torch_npu_device_demo' if kind=='device' else 'torch_npu_training_demo')
        demo.OUT=path
        if kind=='device':demo.DESTINATION=path/'device.json';assert demo.execute(True)==0
        else:demo.REPORT=path/'training.json';assert demo.execute(True,False)==0


def audit():
    files={};comparisons=0
    def checked(path,digest=None):
        value=sha(path)
        if digest is not None:assert value==digest,path
        files[str(path.relative_to(OUT))]=value
        return path
    frozen=read(checked(OUT/'before/sources.json'))
    for name,digest in frozen['sources'].items():checked(OUT/'before'/name,digest)
    build_info=read(checked(OUT/'build.json'));assert build_info['builder_sha256']==sha(Path(__file__))
    info=read(checked(OUT/'diagnostic/build.json'));assert info==build_info['diagnostic']
    checked(OUT/'diagnostic/backend.cpp',info['source_sha256']);checked(OUT/'diagnostic/_torch_intel_npu.pyd',info['extension_sha256'])
    checked(OUT/'host-runtime/session_task_profile.dll',build_info['profiling']['dll_sha256'])
    rows=[]
    for label in ('plain','host'):
        report=read(checked(OUT/(label+'.json')));assert report['status']=='passed' and report['source_sha256']==sha(Path(__file__))
        assert report['build']==build_info and report['frozen']==frozen
        stats=report['final_stats'];assert stats['closed'] and not stats['context_failed']
        assert all(stats[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
        with np.load(checked(OUT/(label+'-outputs.npz'),report['outputs_sha256'])) as values:
            assert len(values.files)==908
            for batch,prefix in SHAPES:
                tag=f'b{batch}-s{prefix}'
                for i in range(182):np.testing.assert_array_equal(values[f'{tag}/64/{i}'].view(np.uint8),values[f'{tag}/128/{i}'].view(np.uint8));comparisons+=1
                for i in range(90):np.testing.assert_array_equal(values[f'{tag}/64/{i+1}'].view(np.uint8),values[f'{tag}/preserved/{i}'].view(np.uint8));comparisons+=1
        for row in report['rows']:
            assert len(row['blocks'])==36 and len(row['pairs_total_ms'])==9
            for capacity in (64,128):
                blocks=[b for b in row['blocks'] if b['capacity']==capacity];assert len(blocks)==18
                for field,value in row['median_ms'][str(capacity)].items():assert value==st.median(s[field] for b in blocks for s in b['samples'])
                count=sum(b['delta']['driver_allocations'] for b in blocks)
                assert count==row['driver_allocations'][str(capacity)]==(2520 if capacity==64 and row['operation']=='decode' else 0)
                for b in blocks:
                    assert len(b['samples'])==5 and b['delta']['npu_calls']==150
                    assert b['cached_blocks']<=capacity and b['cached_bytes']<=16*1024*1024
                    assert all(b['delta'][k]==0 for k in ('host_upload_copies','host_download_copies','host_device_copies','host_pack_copies','session_creations'))
                    if label=='host':
                        for s in b['samples']:assert s['stage_counts']['shared_alloc']==s['stage_counts']['shared_free']==(28 if capacity==64 and row['operation']=='decode' else 0)
            rows.append(dict(label=label,tag=row['tag'],operation=row['operation'],median_ms=row['median_ms'],total_reduction_percent=row['total_reduction_percent'],positive_rounds=row['positive_rounds']))
    with np.load(OUT/'plain-outputs.npz') as ordinary,np.load(OUT/'host-outputs.npz') as instrumented:
        for key in ordinary.files:np.testing.assert_array_equal(ordinary[key].view(np.uint8),instrumented[key].view(np.uint8));comparisons+=1
    current=read(ROOT/'npunlock-probe/torch-device/build.json')
    if current['extension_sha256']!=read(OUT/'before/npunlock-probe/torch-device/build.json')['extension_sha256']:
        expected=(OUT/'before/npunlock-probe/torch-device/backend.cpp').read_text(encoding='utf-8').replace('constexpr size_t max_cached_buffers = 64;','constexpr size_t max_cached_buffers = 128;')
        assert (ROOT/'npunlock-probe/torch-device/backend.cpp').read_text(encoding='utf-8')==expected
        assert current['source_sha256']==sha(ROOT/'npunlock-probe/torch-device/backend.cpp') and current['extension_sha256']==sha(ROOT/'npunlock-probe/torch-device/_torch_intel_npu.pyd')
        assert current['session_sha256']==read(OUT/'before/npunlock-probe/torch-device/build.json')['session_sha256']
        confirmation=read(checked(OUT/'production/confirmed.json'));assert confirmation['status']=='passed'
        assert all(r['warm_delta']['driver_allocations']==0 for r in confirmation['rows'])
        with np.load(checked(OUT/'production/confirmed-outputs.npz',confirmation['outputs_sha256'])) as values,np.load(OUT/'plain-outputs.npz') as expected:
            for key in values.files:
                tag,index=key.split('/');np.testing.assert_array_equal(values[key].view(np.uint8),expected[f'{tag}/128/{index}'].view(np.uint8));comparisons+=1
        for name in ('allocator-guards.json','idle-order.json','allocator-failure.json','device.json','training.json'):
            report=read(checked(OUT/'checks'/name));assert report['status']=='passed'
        for name in ('npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/build.json','npunlock-probe/torch-device/_torch_intel_npu.pyd',
            'torch_intel_npu.py','torch_npu_warm_profile.py','npunlock-probe/research/decode_cache_checks.py','npunlock-probe/research/decode_cache_probe.py'):
            path=OUT/'after'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path);checked(path)
    path=OUT/'cache_blocks_probe.py';shutil.copyfile(Path(__file__),path);checked(path)
    write(OUT/'audit.json',dict(status='passed',rows=rows,bit_equal_comparisons=comparisons,files_sha256=files,current_build=current))
    print('Passed:',comparisons,'array comparisons;',len(files),'files verified',flush=True)
    for row in rows:print(row['label'],row['tag'],row['operation'],row['total_reduction_percent'],'percent; wins',row['positive_rounds'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('build','probe','confirm','check','audit'),required=True)
    parser.add_argument('--label',default='plain');parser.add_argument('--host',action='store_true')
    parser.add_argument('--check',choices=('guards','order','failure','device','training'));args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.case=='build':build()
    elif args.case=='probe':probe(args.label,args.host)
    elif args.case=='confirm':confirm(args.label)
    elif args.case=='check':check(args.check)
    else:audit()


if __name__=='__main__':main()

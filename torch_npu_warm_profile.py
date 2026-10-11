"""Host API profiling in an isolated DLL; production code remains untouched."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/warm-allocation'
STAGES=('shared_alloc','shared_free','set_argument','append_graph','close_list',
        'submit','wait','reset_list','reset_fence')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    source=ROOT/'npunlock/src/graphinfer/session_win32.c'
    text=source.read_text(encoding='utf-8')
    extra=r'''
static volatile LONG64 diag_ticks[9], diag_counts[9];
static uint64_t diag_clock(void) {
  LARGE_INTEGER now; QueryPerformanceCounter(&now); return (uint64_t)now.QuadPart;
}
#define DIAG_TIME(slot, statement) do { \
  uint64_t begin = diag_clock(); statement; \
  InterlockedAdd64(&diag_ticks[slot], (LONG64)(diag_clock() - begin)); \
  InterlockedIncrement64(&diag_counts[slot]); \
} while (0)
__declspec(dllexport) void warm_diag_reset(void) {
  for (uint32_t i=0;i<9;i++) { InterlockedExchange64(&diag_ticks[i],0); InterlockedExchange64(&diag_counts[i],0); }
}
__declspec(dllexport) void warm_diag_read(uint64_t *ns, uint64_t *counts) {
  LARGE_INTEGER frequency; QueryPerformanceFrequency(&frequency);
  for (uint32_t i=0;i<9;i++) {
    ns[i]=(uint64_t)diag_ticks[i]*1000000000ull/(uint64_t)frequency.QuadPart;
    counts[i]=(uint64_t)diag_counts[i];
  }
}
'''
    assert text.count('struct graphinfer_session {')==1
    text=text.replace('struct graphinfer_session {',extra+'\nstruct graphinfer_session {')
    calls=[('code =\n      session->ze.mem_alloc_shared(session->context, &device_desc, &host_desc, size,\n'
            '                                   SESSION_PAGE_SIZE, session->device, &implementation->allocation);',1),
           ('session->ze.mem_free(session->context, implementation->allocation);',1),
           ('code = session->graph.set_argument(session->handle, (uint32_t)index, allocations[index]);',2),
           ('code = initialize ? session->graph.append_initialize(session->list, session->handle, NULL, 0, NULL)\n'
            '                    : session->graph.append_execute(session->list, session->handle, NULL, NULL, 0, NULL);',1),
           ('code = session->ze.command_list_close(session->list);',1),
           ('code = session->ze.command_queue_execute(queue, 1, &session->list, session->fence);',1),
           ('code = session->ze.fence_synchronize(session->fence, session->fence_timeout_ns);',1),
           ('code = session->ze.command_list_reset(session->list);',1),
           ('code = session->ze.fence_reset(session->fence);',1)]
    for slot,(call,count) in enumerate(calls):
        assert text.count(call)==count,(slot,text.count(call))
        text=text.replace(call,f'DIAG_TIME({slot}, {call[:-1]});')
    chain_calls={
        2:'code = session->graph.set_argument(session->handle, (uint32_t)j, allocations[i][j]);',
        3:'code = first->graph.append_execute(first->list, entries[i].session->handle, NULL, NULL, 0, NULL);',
        4:'code = first->ze.command_list_close(first->list);',
        5:'code = first->ze.command_queue_execute(first->context_owner->queue, 1, &first->list, first->fence);',
        6:'code = first->ze.fence_synchronize(first->fence, first->fence_timeout_ns);',
        7:'code = first->ze.command_list_reset(first->list);',
        8:'code = first->ze.fence_reset(first->fence);'}
    for slot,call in chain_calls.items():
        assert text.count(call)==1,(slot,call)
        text=text.replace(call,f'DIAG_TIME({slot}, {call[:-1]});')
    generated=OUT/'session_profile.c';generated.write_text(text,encoding='utf-8')
    compiler=Path('C:/Program Files/mingw64/bin/gcc.exe')
    sources=[ROOT/'npunlock'/p for p in ('src/common/buffer.c','src/common/checked.c',
             'src/common/diagnostic.c','src/common/status.c','src/graphinfer/graphinfer.c')]+[generated]
    args=[str(compiler),'-shared','-O2','-std=c17','-Wall','-Wextra','-Werror','-Wpedantic',
          '-static-libgcc','-DNPUNLOCK_BUILDING','-DNPUNLOCK_SHARED']
    for path in ('include','src','src/common','src/graphinfer'):args+=['-I',str(ROOT/'npunlock'/path)]
    args += [str(p) for p in sources]+['-o',str(OUT/'session_profile.dll')]
    result=subprocess.run(args,env=dict(os.environ,PATH=str(compiler.parent)+';'+os.environ['PATH']),
                          capture_output=True,text=True)
    (OUT/'profile-build.log').write_text(result.stdout+result.stderr,encoding='utf-8');result.check_returncode()
    manifest=dict(dll_sha256=sha(OUT/'session_profile.dll'),sources={str(p.relative_to(ROOT)):sha(p)
                  for p in (*sources,source,Path(__file__))},arguments=args)
    (OUT/'profile-build.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Isolated profile DLL built')


def bind(dll,manifest_path):
    import torch_npu_runtime
    torch_npu_runtime.SESSION_DLL=dll
    source=(ROOT/'torch_intel_npu.py').read_text(encoding='utf-8')
    original='_BUILD = _EXT.with_name("build.json")';assert source.count(original)==1
    source=source.replace(original,f'_BUILD = Path({str(manifest_path)!r})')
    module=types.ModuleType('torch_intel_npu');module.__file__=str(ROOT/'torch_intel_npu.py')
    sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)


def profile():
    import torch_npu_session_reuse_benchmark as common
    dll=OUT/'session_profile.dll'
    manifest=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    manifest['session_sha256']=sha(dll)
    path=OUT/'profile-bound-build.json';path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    bind(dll,path)
    driver=ct.CDLL(str(dll));driver.warm_diag_reset.argtypes=[];driver.warm_diag_reset.restype=None
    driver.warm_diag_read.argtypes=[ct.POINTER(ct.c_uint64)]*2;driver.warm_diag_read.restype=None
    profiles=[]
    def measure(runs,rounds=31):
        for run in runs.values():
            for _ in range(5):value=run();del value
        samples={name:[] for name in runs};stages={name:[] for name in runs};cleanup={name:[] for name in runs}
        for index in range(rounds):
            for name in (tuple(runs) if index%2==0 else tuple(reversed(runs))):
                driver.warm_diag_reset();start=time.perf_counter_ns();value=runs[name]()
                returned=time.perf_counter_ns();del value;freed=time.perf_counter_ns()
                ns=(ct.c_uint64*9)();counts=(ct.c_uint64*9)();driver.warm_diag_read(ns,counts)
                samples[name].append((returned-start)/1e6);cleanup[name].append((freed-returned)/1e6)
                stages[name].append(dict(ns=dict(zip(STAGES,ns)),counts=dict(zip(STAGES,counts))))
        result={name:dict(median_ms=statistics.median(values),samples_ms=values) for name,values in samples.items()}
        profiles.append({name:dict(api_median_ms=result[name]['median_ms'],cleanup_median_ms=statistics.median(cleanup[name]),
                        stage_median_ms={stage:statistics.median([s['ns'][stage] for s in stages[name]])/1e6 for stage in STAGES},
                        samples=stages[name]) for name in runs})
        print(json.dumps({name:{key:value for key,value in row.items() if key!='samples'}
                          for name,row in profiles[-1].items()}),flush=True)
        return result
    common.OUT=OUT;common.NEW_DLL=dll;common.measure=measure
    common.model('after','profile-model')
    (OUT/'host-profile.json').write_text(json.dumps(dict(status='passed',stages=STAGES,rows=profiles,
        profile_script_sha256=sha(Path(__file__)),scope='QPC host API costs; wait includes device work; '
        'output deletion outside API time; instrumented DLL, no device task profiling'),indent=2),encoding='utf-8')


def guards():
    import gc
    import torch
    import torch_intel_npu as npu
    from concurrent.futures import ThreadPoolExecutor
    checks=[]
    def clean():
        gc.collect();torch.intel_npu.empty_cache()
        stats=npu.stats();assert stats['cached_buffer_bytes']==stats['cached_buffer_blocks']==0
        return stats
    with torch.no_grad():
        before=clean()
        for _ in range(20):
            value=torch.empty(4096,dtype=torch.float16,device='intel_npu');del value
        after=npu.stats()
        assert after['buffer_allocations']-before['buffer_allocations']==20
        assert after['driver_allocations']-before['driver_allocations']==1
        assert after['allocation_reuses']-before['allocation_reuses']==19
        assert after['cached_buffer_bytes']==8192 and after['cached_buffer_blocks']==1
        checks.append(dict(name='exact_size_reuse',status='passed',logical_allocations=20,driver_allocations=1,reuses=19))
        clean()
        for dtype,bits in ((torch.float16,torch.int16),(torch.float32,torch.int32)):
            host=torch.tensor([1.,-0.,float('inf'),float('nan'),-3.,.25,2.,-1.],dtype=dtype)
            value=host.to('intel_npu');view=value.view(2,4);address=value.data_ptr();del value
            other=(host*2).to('intel_npu');assert other.data_ptr()!=address
            torch.intel_npu.empty_cache()
            assert torch.equal(view.cpu().view(bits).view(-1),host.view(bits))
            del other,view;clean()
        checks.append(dict(name='last_storage_view_and_live_empty_cache',status='passed',precisions=['float16','float32']))
        blocks=[torch.empty(1048576,dtype=torch.float32,device='intel_npu') for _ in range(5)]
        del blocks;stats=npu.stats()
        assert stats['cached_buffer_bytes']==stats['allocation_cache_limit_bytes']==16*1024*1024
        assert stats['cached_buffer_blocks']==4
        clean()
        limit=npu.stats()['allocation_cache_limit_blocks']
        blocks=[torch.empty(1,dtype=torch.float32,device='intel_npu') for _ in range(limit+3)]
        del blocks;stats=npu.stats()
        assert stats['cached_buffer_blocks']==stats['allocation_cache_limit_blocks']==limit
        assert stats['cached_buffer_bytes']==limit*4
        checks.append(dict(name='byte_and_block_limits',status='passed',bytes=16*1024*1024,blocks=limit))
        clean()
        def worker(index):
            with torch.no_grad():
                host=torch.full((32,),index+.25)
                for _ in range(20):
                    value=host.to('intel_npu');view=value.view(4,8);del value
                    assert torch.equal(view.cpu().view(-1),host);del view
        with ThreadPoolExecutor(max_workers=4) as threads:list(threads.map(worker,range(4)))
        checks.append(dict(name='concurrent_storage_recycling',status='passed',workers=4,copies=80))
        clean()
        value=torch.ones(32).to('intel_npu');view=value.view(4,8);del value
        spare=torch.empty(4096,dtype=torch.float16,device='intel_npu');del spare
        assert npu.stats()['cached_buffer_bytes']>0
        npu.shutdown();stats=npu.stats()
        assert stats['cached_buffer_bytes']==stats['cached_buffer_blocks']==0
        assert stats['live_buffers']==1
        del view;gc.collect();torch.intel_npu.empty_cache();stats=npu.stats()
        assert stats['live_buffers']==stats['live_buffer_bytes']==0 and not stats['context_failed']
        checks.append(dict(name='shutdown_with_held_storage',status='passed'))
    (OUT/'allocator-guards.json').write_text(json.dumps(dict(status='passed',checks=checks,final_stats=stats,
        device_build=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))),indent=2),encoding='utf-8')
    print('Allocator reuse, limits, live views, threads and shutdown: passed')


def worker(kind,label):
    import torch_npu_session_reuse_benchmark as common
    if kind=='before':
        # A separate process loads the actual saved PYD, C++ source and manifest.
        source=(OUT/'before/torch_intel_npu.py').read_text(encoding='utf-8')
        old='_EXT = _ROOT / "npunlock-probe/torch-device/_torch_intel_npu.pyd"';assert source.count(old)==1
        source=source.replace(old,f'_EXT = Path({str(OUT/"before/_torch_intel_npu.pyd")!r})')
        module=types.ModuleType('torch_intel_npu');module.__file__=str(ROOT/'torch_intel_npu.py')
        sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
    import torch_intel_npu as npu
    original_measure=common.measure;deltas=[]
    def measure(runs,rounds=31):
        before=npu.stats();result=original_measure(runs,rounds);after=npu.stats()
        deltas.append(dict(logical_allocations=after['buffer_allocations']-before['buffer_allocations'],
            driver_allocations=after.get('driver_allocations',after['buffer_allocations'])-
                               before.get('driver_allocations',before['buffer_allocations']),
            allocation_reuses=after.get('allocation_reuses',0)-before.get('allocation_reuses',0)))
        return result
    common.measure=measure
    common.OUT=OUT;common.model('after',label)
    path=OUT/f'{label}.json';report=json.loads(path.read_text(encoding='utf-8'))
    build_path=OUT/'before/build.json' if kind=='before' else ROOT/'npunlock-probe/torch-device/build.json'
    report.update(runtime=kind,device_build=json.loads(build_path.read_text(encoding='utf-8-sig')),
                  allocator_benchmark_sha256=sha(Path(__file__)))
    for row,delta in zip(report['rows'],deltas):row['allocator_counter_delta']=delta
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')


def control():
    import numpy as np
    labels=[]
    for index,kind in enumerate(('before','after','after','before')):
        label=f'allocator-{index}-{kind}';labels.append(label)
        with (OUT/f'{label}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,__file__,'--case','worker','--runtime',kind,'--label',label],
                           stdout=log,stderr=log,timeout=300,check=True)
        print(label,'passed',flush=True)
    with np.load(OUT/f'{labels[0]}-outputs.npz',allow_pickle=False) as reference:
        count=len(reference.files)
        for label in labels[1:]:
            with np.load(OUT/f'{label}-outputs.npz',allow_pickle=False) as actual:
                assert actual.files==reference.files
                for key in reference.files:np.testing.assert_array_equal(actual[key].view(np.uint8),reference[key].view(np.uint8))
    reports=[json.loads((OUT/f'{label}.json').read_text(encoding='utf-8')) for label in labels]
    rows=[]
    for index in range(4):
        for operation in ('prefill','decode'):
            values={kind:[r['rows'][index]['timings'][operation]['median_ms'] for r in reports if r['runtime']==kind]
                    for kind in ('before','after')}
            medians={kind:statistics.mean(samples) for kind,samples in values.items()}
            rows.append(dict(**{key:reports[0]['rows'][index][key] for key in ('batch','prefix_tokens','mode')},
                operation=operation,process_medians_ms=values,mean_process_median_ms=medians,
                latency_reduction_percent=100*(1-medians['after']/medians['before'])))
    (OUT/'allocator-control.json').write_text(json.dumps(dict(status='passed',process_order=labels,rows=rows,
        all_output_cache_bits_equal=True,arrays_per_process=count),indent=2),encoding='utf-8')
    print('ABBA model outputs, KV snapshots and lifetimes: passed')


def audit():
    import math
    import numpy as np
    build=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    for directory in ('before','after'):
        manifest=json.loads((OUT/directory/'build.json').read_text(encoding='utf-8-sig'))
        assert manifest['extension_sha256']==sha(OUT/directory/'_torch_intel_npu.pyd')
        assert manifest['source_sha256']==sha(OUT/directory/'backend.cpp')
        assert manifest['session_sha256']==sha(ROOT/'npunlock-probe/session-runtime/npunlock_session.dll')
    assert build==json.loads((OUT/'after/build.json').read_text(encoding='utf-8-sig'))
    assert build['source_sha256']==sha(ROOT/'npunlock-probe/torch-device/backend.cpp')
    assert build['extension_sha256']==sha(ROOT/'npunlock-probe/torch-device/_torch_intel_npu.pyd')
    assert sha(ROOT/'torch_npu_smollm.py')==sha(OUT/'before/torch_npu_smollm.py')==sha(OUT/'after/torch_npu_smollm.py')
    session=json.loads((ROOT/'npunlock-probe/session-runtime/build.json').read_text(encoding='utf-8-sig'))
    assert session['dllSha256'].lower()==build['session_sha256']
    for record in (*session['sources'],*session['headers']):
        assert sha(ROOT/'npunlock'/record['path'])==record['sha256'].lower()
    control=json.loads((OUT/'allocator-control.json').read_text(encoding='utf-8'));assert control['status']=='passed'
    assert control['arrays_per_process']==104 and control['all_output_cache_bits_equal']
    reports=[]
    for label in control['process_order']:
        report=json.loads((OUT/f'{label}.json').read_text(encoding='utf-8'));reports.append(report)
        assert report['status']=='passed' and report['allocator_benchmark_sha256']==sha(OUT/'measurement-script.py')
        assert report['script_sha256']==sha(ROOT/'torch_npu_session_reuse_benchmark.py')
        assert report['outputs_sha256']==sha(OUT/f'{label}-outputs.npz')
        expected=json.loads((OUT/report['runtime']/'build.json').read_text(encoding='utf-8-sig'))
        assert expected==report['device_build'] and report['session_sha256']==build['session_sha256']
        final=report['final_stats'];assert not final['context_failed'] and final['npu_calls']==1248
        assert final['live_buffers']==final['live_buffer_bytes']==final.get('cached_buffer_bytes',0)==0
        for row in report['rows']:
            assert row['previous_outputs_and_seed_preserved'] and row['warm_counter_delta']['npu_calls']==288
            assert all(row['warm_counter_delta'][key]==0 for key in ('host_upload_copies','host_download_copies','session_creations'))
            delta=row['allocator_counter_delta'];assert delta['logical_allocations']==1008
            assert delta['driver_allocations']+delta['allocation_reuses']==1008
            if report['runtime']=='after':assert delta['driver_allocations']<=2
            else:assert delta['driver_allocations']==1008
            for value in row['timings'].values():
                assert len(value['samples_ms'])==31 and statistics.median(value['samples_ms'])==value['median_ms']
    for row in control['rows']:
        for kind in ('before','after'):
            values=[next(r for r in report['rows'] if all(r[k]==row[k] for k in ('batch','prefix_tokens','mode')))
                    ['timings'][row['operation']]['median_ms'] for report in reports if report['runtime']==kind]
            assert values==row['process_medians_ms'][kind] and statistics.mean(values)==row['mean_process_median_ms'][kind]
        a=row['mean_process_median_ms'];assert math.isclose(row['latency_reduction_percent'],100*(1-a['after']/a['before']))
    with np.load(OUT/f"{control['process_order'][0]}-outputs.npz",allow_pickle=False) as reference:
        for label in control['process_order'][1:]:
            with np.load(OUT/f'{label}-outputs.npz',allow_pickle=False) as actual:
                assert reference.files==actual.files
                for key in reference.files:np.testing.assert_array_equal(reference[key].view(np.uint8),actual[key].view(np.uint8))
    for directory in (OUT/'profile-before',OUT):
        report=json.loads((directory/'profile-model.json').read_text(encoding='utf-8'))
        assert report['status']=='passed' and report['outputs_sha256']==sha(directory/'profile-model-outputs.npz')
        assert report['final_stats']['live_buffers']==0 and not report['final_stats']['context_failed']
        assert report['session_sha256']==sha(directory/'session_profile.dll')
        manifest=json.loads((directory/'profile-bound-build.json').read_text(encoding='utf-8'))
        assert manifest['extension_sha256']==report['device_build']['extension_sha256']
    profile=json.loads((OUT/'host-profile.json').read_text(encoding='utf-8'))
    assert profile['status']=='passed' and profile['profile_script_sha256']==sha(OUT/'measurement-script.py')
    for row in profile['rows']:
        for value in row.values():
            assert len(value['samples'])==31
            for sample in value['samples']:
                assert sample['counts']['shared_alloc']==sample['counts']['shared_free']==0
                assert all(sample['counts'][key]==4 for key in ('append_graph','close_list','submit','wait','reset_list','reset_fence'))
    with np.load(OUT/'profile-before/profile-model-outputs.npz',allow_pickle=False) as a,np.load(OUT/'profile-model-outputs.npz',allow_pickle=False) as b:
        assert a.files==b.files
        for key in a.files:np.testing.assert_array_equal(a[key].view(np.uint8),b[key].view(np.uint8))
    guards=json.loads((OUT/'allocator-guards.json').read_text(encoding='utf-8'))
    assert guards['status']=='passed' and len(guards['checks'])==5 and guards['device_build']==build
    failure=json.loads((OUT/'allocator-failure.json').read_text(encoding='utf-8'))
    assert failure['status']=='passed' and failure['synthetic_failure'] and not failure['actual_device_submission']
    assert failure['driver_free_calls']==failure['driver_reset_calls']==0
    assert failure['failed']['context_failed'] and failure['failed']['quarantined_buffer_bytes']>=failure['before']['live_buffer_bytes']+failure['before']['cached_buffer_bytes']
    regression={}
    for path in sorted((OUT/'regressions').glob('*.json')):
        report=json.loads(path.read_text(encoding='utf-8'));assert report['status']=='passed'
        assert not report['openvino_installed'] and all(c['status']=='passed' for c in report['checks'])
        stats=report['final_stats'];assert not stats['context_failed']
        assert stats['live_buffers']==stats['live_buffer_bytes']==stats['cached_buffer_blocks']==stats['cached_buffer_bytes']==0
        regression[path.name]=len(report['checks'])
    assert len(regression)==2
    sources={name:sha(ROOT/name) for name in ('npunlock-probe/torch-device/backend.cpp','torch_intel_npu.py',
             'torch_npu_smollm.py','torch_npu_warm_profile.py','npunlock-probe/research/allocator_failure_probe.py')}
    (OUT/'audit.json').write_text(json.dumps(dict(status='passed',device_build=build,source_sha256=sources,
        model_graph_calls=4992,profile_graph_calls=2496,model_arrays=104,timed_profile_operations_without_alloc_free=248,
        regressions=regression,synthetic_cached_buffer_quarantine_passed=True),indent=2),encoding='utf-8')
    print('Build provenance, raw timings, allocations, output bits, quarantine and regressions: passed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('build','profile','guards','worker','control','audit'),required=True)
    parser.add_argument('--runtime',choices=('before','after'),default='after');parser.add_argument('--label',default='allocator-after')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.case=='worker':worker(args.runtime,args.label)
    else:{'build':build,'profile':profile,'guards':guards,'control':control,'audit':audit}[args.case]()


if __name__=='__main__':main()

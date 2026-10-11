"""Offline validation, interval statistics and Perfetto export for task samples."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/smollm-device-profile'
sys.path.insert(0,str(ROOT/'npunlock-probe/research'))
from dpu_diagnostic import Task,decode
from torch_npu_device_profile import union_ns


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def rmsnorm(task):return task['exec_type']==2 and task['name'].split('?')[0] in ('rmsnorm_carrier','rmsnorm_ffn')


def kernel_patches(blob,reports,norm_manifest):
    """Validate either RMSNorm alone or its composition with KV Select."""
    from npunlock.npu4000 import kernel_text
    assert [p['carrier'] for p in reports] in (['Power'],['Power','Select'])
    root=ROOT/'npunlock/tests/fixtures/npu4000';windows=[]
    for i,patch in enumerate(reports):
        manifest=norm_manifest if i==0 else load(root/'kv_select/manifest.json')
        path=root/('rmsnorm/rmsnorm-f16.elf' if i==0 else 'kv_select/kv-select-f16.elf')
        assert patch['kernel_elf_sha256']==manifest['elf_sha256']==sha(path)
        code=kernel_text(path.read_bytes());start=patch['code_offset'];end=start+len(code)
        assert 0<=start<end<=len(blob) and patch['code_bytes']==len(code)==manifest['text_bytes']
        assert blob[start:end]==code
        assert patch['device_id']=='0x643e' and patch['compiler_version']==[8,3]
        assert patch['dtype']==patch['output_dtype']=='f16'
        assert all(max(start,a)>=min(end,b) for a,b in windows);windows.append((start,end))
        if i:assert reports[i-1]['patched_sha256']==patch['original_sha256']
    assert reports[-1]['patched_sha256']==hashlib.sha256(blob).hexdigest()
    return reports[0]


def trace(graphs,sample_index,path):
    # Lanes prevent visual overlaps; they are not physical engine identifiers.
    events=[]
    for pid,graph in enumerate(graphs,1):
        events.append(dict(ph='M',name='process_name',pid=pid,args=dict(name=graph['tag'])))
        tasks=graph['samples'][sample_index]['tasks'];origin=min(t['start_time_ns'] for t in tasks if t['duration_ns'])
        for kind,name in ((1,'DPU'),(2,'SHAVE'),(3,'DMA')):
            ends=[]
            for task in sorted((t for t in tasks if t['exec_type']==kind and t['duration_ns']),key=lambda t:t['start_time_ns']):
                lane=next((i for i,end in enumerate(ends) if end<=task['start_time_ns']),len(ends))
                if lane==len(ends):
                    ends.append(0)
                    events.append(dict(ph='M',name='thread_name',pid=pid,tid=kind*100+lane,
                        args=dict(name=f'{name} visual lane {lane} (not hardware ID)')))
                ends[lane]=task['start_time_ns']+task['duration_ns']
                events.append(dict(ph='X',name=task['name'],cat=name,pid=pid,tid=kind*100+lane,
                    ts=(task['start_time_ns']-origin)/1000,dur=task['duration_ns']/1000,
                    args={k:task[k] for k in ('layer_type','task_id','active_cycles','stall_cycles')}))
    path.write_text(json.dumps(dict(traceEvents=events,displayTimeUnit='ms',
        description='Per-graph relative time; graphs are separate panels, lanes are visual only; DMA unavailable on this NPU40xx driver.')),encoding='utf-8')


def main():
    build=load(OUT/'build.json');production=load(ROOT/'npunlock-probe/torch-device/build.json')
    assert build['dll_sha256']==sha(OUT/'session_task_profile.dll')
    assert build['generated_source_sha256']==sha(OUT/'session_task_profile.c')
    assert build['builder_sha256']==sha(OUT/'measurement.py')==sha(ROOT/'torch_npu_device_profile.py')
    assert build['production_session_sha256']==sha(ROOT/'npunlock/src/graphinfer/session_win32.c')
    host_build=load(ROOT/'npunlock-probe/research/results/warm-allocation/profile-build.json')
    for name,digest in host_build['sources'].items():
        if name!='torch_npu_warm_profile.py':assert sha(ROOT/name)==digest,(name,digest)
    previous=load(ROOT/'npunlock-probe/research/results/warm-allocation/audit.json')
    assert previous['device_build']==production
    for name,digest in previous['source_sha256'].items():assert sha(ROOT/name)==digest,(name,digest)
    assert production['session_sha256']==sha(ROOT/'npunlock-probe/session-runtime/npunlock_session.dll')
    assert production['extension_sha256']==sha(ROOT/'npunlock-probe/torch-device/_torch_intel_npu.pyd')
    assert production['source_sha256']==sha(ROOT/'npunlock-probe/torch-device/backend.cpp')
    bound=load(OUT/'bound-build.json');assert bound['session_sha256']==build['dll_sha256']
    assert {k:v for k,v in bound.items() if k!='session_sha256'}=={k:v for k,v in production.items() if k!='session_sha256'}
    kernel_root=ROOT/'npunlock/tests/fixtures/npu4000'
    kernel=load(kernel_root/'rmsnorm/manifest.json')
    for name,digest in kernel['source_sha256'].items():assert sha(kernel_root/name)==digest
    assert kernel['elf_sha256']==sha(kernel_root/'rmsnorm/rmsnorm-f16.elf')
    assert kernel['assembly_sha256']==sha(kernel_root/'rmsnorm/rmsnorm-f16.s')
    rows=[];task_records=0;calls=0;graph_count=0;comparisons=0;timed_calls=0;files={}
    for tag in ('b1-s32-fp16','b1-s32-w8a16','b4-s16-fp16','b4-s16-w8a16'):
        path=OUT/f'{tag}.json';report=load(path);files[path.name]=sha(path)
        assert report['status']=='passed' and report['device_build']==production
        assert report['source_sha256']==sha(OUT/'measurement.py')
        assert report['profiling_dll_sha256']==build['dll_sha256']
        for name,digest in report['production_sha256'].items():assert sha(ROOT/name)==digest,(name,digest)
        assert len(report['build_flags'])==len(report['graphs'])==8
        assert all(flag.endswith(' PERF_COUNT="YES"') and flag.count('PERF_COUNT')==1 for flag in report['build_flags'])
        final=report['final_stats'];assert not final['context_failed'] and final['closed']
        assert final['live_buffers']==final['live_buffer_bytes']==final['cached_buffer_bytes']==final['cached_buffer_blocks']==0
        calls+=final['npu_calls'];graph_count+=8
        path=OUT/f'{tag}-outputs.npz';assert report['outputs_sha256']==sha(path);files[path.name]=sha(path)
        with np.load(path,allow_pickle=False) as arrays:
            assert len(arrays.files)==64
            for i in range(26):
                np.testing.assert_array_equal(arrays[f'plain/{i}'].view(np.uint8),arrays[f'profile/{i}'].view(np.uint8));comparisons+=1
            for i in range(2,14):np.testing.assert_array_equal(arrays[f'plain/{i}'].view(np.uint8),arrays[f'seed/{i}'].view(np.uint8))
        for graph in report['graphs']:
            path=OUT/f"{graph['tag']}.blob";assert graph['blob_sha256']==sha(path);files[path.name]=sha(path)
            patch=kernel_patches(path.read_bytes(),graph['patch_reports'],kernel)
            assert patch['carrier']=='Power' and patch['device_id']=='0x643e' and patch['compiler_version']==[8,3]
            assert patch['kernel_elf_sha256']==kernel['elf_sha256']
            assert patch['dtype']==patch['output_dtype']=='f16' and patch['code_bytes']==kernel['text_bytes']
            assert patch['invocations']==len(patch['contracts'])==(8 if report['batch']==4 and graph['operation']=='decode' else 16)
            assert len(graph['samples'])==3 and len({s['raw_sha256'] for s in graph['samples']})==3
            for i,sample in enumerate(graph['samples']):
                path=OUT/f"{graph['tag']}-{i}-tasks.bin";data=path.read_bytes();files[path.name]=sha(path)
                assert sample['raw_sha256']==sha(path) and sample['task_record_bytes']==ct.sizeof(Task)==344
                assert len(data)==344*len(sample['tasks'])
                for offset,task in enumerate(sample['tasks']):
                    value=Task.from_buffer_copy(data[offset*344:(offset+1)*344])
                    for key,_ in Task._fields_:
                        actual=getattr(value,key)
                        if key in ('name','layer_type'):actual=decode(actual)
                        assert task[key]==actual,(key,task,actual)
                tasks=sample['tasks'];task_records+=len(tasks)
                assert all(0<=t['exec_type']<=3 and 0<t['duration_ns']<1_000_000_000 for t in tasks)
                span=max(t['start_time_ns']+t['duration_ns'] for t in tasks)-min(t['start_time_ns'] for t in tasks)
                assert 0<span<1_000_000_000 and sample['timeline_span_ms']==span/1e6
                assert sample['observed_union_ms']==union_ns(tasks)/1e6
                assert sample['unobserved_gaps_ms']==(span-union_ns(tasks))/1e6
                for kind,name in ((1,'dpu'),(2,'shave'),(3,'dma')):
                    selected=[t for t in tasks if t['exec_type']==kind]
                    assert sample['engines'][name]==dict(tasks=len(selected),sum_ms=sum(t['duration_ns'] for t in selected)/1e6,union_ms=union_ns(selected)/1e6)
                assert not sample['dma_duration_available'] and sample['engines']['dma']['tasks']==0
                assert len([t for t in tasks if rmsnorm(t)])==patch['invocations']
        for operation in ('prefill','decode'):
            graphs=sorted((g for g in report['graphs'] if g['operation']==operation),key=lambda g:g['layer'])
            assert [g['layer'] for g in graphs]==list(range(4))
            samples=[]
            for i,host in enumerate(report[operation]['samples']):
                assert host['shared_alloc_calls']==host['shared_free_calls']==0
                assert host['native_call_counts'][3:]==[4]*6
                assert host['native_times_ms'][6]==host['host_wait_ms'] and host['ms']>=host['host_wait_ms']>0
                timed_calls+=4
                device=[g['samples'][i] for g in graphs]
                row=dict(api_ms=host['ms'],host_wait_ms=host['host_wait_ms'],
                    span_ms=sum(d['timeline_span_ms'] for d in device),
                    observed_union_ms=sum(d['observed_union_ms'] for d in device),
                    unobserved_gap_ms=sum(d['unobserved_gaps_ms'] for d in device),
                    rmsnorm_union_ms=sum(union_ns([t for t in d['tasks'] if rmsnorm(t)])/1e6 for d in device),
                    select_union_ms=sum(union_ns([t for t in d['tasks'] if t['exec_type']==2 and t['layer_type']=='Select'])/1e6 for d in device),
                    softmax_union_ms=sum(union_ns([t for t in d['tasks'] if t['exec_type']==2 and '/softmax/' in t['name']])/1e6 for d in device),
                    other_shave_union_ms=sum(union_ns([t for t in d['tasks'] if t['exec_type']==2 and not rmsnorm(t)])/1e6 for d in device))
                for name in ('shave','dpu'):
                    for field in ('union_ms','sum_ms'):row[f'{name}_{field}']=sum(d['engines'][name][field] for d in device)
                row['rmsnorm_span_percent']=100*row['rmsnorm_union_ms']/row['span_ms']
                row['rmsnorm_shave_percent']=100*row['rmsnorm_union_ms']/row['shave_union_ms']
                row['dpu_span_percent']=100*row['dpu_union_ms']/row['span_ms']
                samples.append(row)
            assert len(samples)==3 and statistics.median(s['api_ms'] for s in samples)==report[operation]['median_ms']
            types={}
            for i in range(3):
                for graph in graphs:
                    for task in graph['samples'][i]['tasks']:
                        if task['exec_type']!=2:continue
                        entry=types.setdefault(task['layer_type'],dict(count=0,total_ms=0.0))
                        entry['count']+=1;entry['total_ms']+=task['duration_ns']/1e6
            medians={k:statistics.median(s[k] for s in samples) for k in samples[0]}
            rows.append(dict(tag=tag,operation=operation,samples=samples,medians=medians,
                software_task_types=types,range_ms={k:[min(s[k] for s in samples),max(s[k] for s in samples)] for k in ('api_ms','host_wait_ms','span_ms')}))
            middle=min(range(3),key=lambda i:abs(samples[i]['api_ms']-medians['api_ms']))
            path=OUT/f'{tag}-{operation}-trace.json';trace(graphs,middle,path);files[path.name]=sha(path)
    summary=dict(status='passed',rows=rows,dma_duration_available=False,
        timing_scope='Sum per-graph interval unions over four synchronous layer graphs. Medians are per metric; engine unions can overlap. Profiling changes graph schedule. Gaps are not DMA measurements.')
    path=OUT/'summary.json';path.write_text(json.dumps(summary,indent=2),encoding='utf-8');files[path.name]=sha(path)
    audit=dict(status='passed',graphs=graph_count,fresh_task_datasets=graph_count*3,task_records=task_records,
        total_npu_calls=calls,timed_graph_calls=timed_calls,bit_equal_arrays=comparisons,
        preserved_seed_arrays=48,production_build=production,rmsnorm_kernel_manifest=kernel,
        production_sha256=report['production_sha256'],files_sha256=files,auditor_sha256=sha(Path(__file__)))
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(f'Passed: {graph_count} graphs, {graph_count*3} fresh task datasets, {task_records} task records, {calls} NPU calls; {comparisons} bit-equal arrays.')
    for row in rows:
        m=row['medians'];print(row['tag'],row['operation'],json.dumps({k:round(m[k],4) for k in ('api_ms','host_wait_ms','span_ms','shave_union_ms','rmsnorm_union_ms','dpu_union_ms','unobserved_gap_ms','rmsnorm_span_percent')}))


if __name__=='__main__':main()

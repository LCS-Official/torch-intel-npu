"""Recheck saved 30-layer measurements, task ABI, patches and row4 controls offline."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics as st
import numpy as np
from shave_fp32_pipeline import OUT, SHAPES, metadata, read
from w8_quality_probe import ROOT, sha
from torch_npu_device_profile_audit import kernel_patches, rmsnorm
from torch_npu_device_profile import union_ns
from dpu_diagnostic import Task, decode


def healthy(report):
    assert report['status']=='passed'
    value=report['final_stats'];assert value['closed'] and not value['context_failed']
    assert all(value[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))


def timing(row, calls):
    for name, samples in row['samples_ms'].items():
        assert len(samples)==2*len(row['pairs_ms']) and all(x>0 for x in samples)
        assert st.median(samples)==row['median_ms'][name]
    assert row['warm_delta']['npu_calls']==calls
    if 'positive_rounds' in row:
        for name, count in row['positive_rounds'].items():
            assert count==sum(pair[name]<pair['baseline'] for pair in row['pairs_ms'])


def main():
    current=metadata();files={};comparisons=0;task_records=0;rows=[];cycles={}
    from w8_performance import profile_binary
    _,profile_build=profile_binary()
    def checked(path, expected=None):
        digest=sha(path)
        if expected is not None:assert digest==expected,path
        files[str(path.relative_to(OUT))]=digest
        return path
    baseline=read(checked(OUT/'baseline-0.json'));host=read(checked(OUT/'host.json'))
    for report in (baseline,host):
        healthy(report);assert {k:report[k] for k in current}==current
        for row in report['rows']:timing(row,62*30)
    production=read(ROOT/'npunlock-probe/torch-device/build.json')
    norm=read(ROOT/'npunlock/tests/fixtures/npu4000/rmsnorm/manifest.json')
    def bound(path, build):
        assert build==profile_build
        value=read(checked(path/'bound-build.json'))
        checked(path/'session_task_profile.dll',build['dll_sha256'])
        assert value['session_sha256']==build['dll_sha256']
        assert {k:v for k,v in value.items() if k!='session_sha256'}=={k:v for k,v in production.items() if k!='session_sha256'}
    bound(OUT/'host-runtime',host['profiling_build'])
    with np.load(checked(OUT/'baseline-0-outputs.npz',baseline['outputs_sha256'])) as ordinary, np.load(checked(OUT/'host-outputs.npz',host['outputs_sha256'])) as instrumented:
        assert len(ordinary.files)==len(instrumented.files)==364
        for key in ordinary.files:
            np.testing.assert_array_equal(ordinary[key].view(np.uint8),instrumented[key].view(np.uint8));comparisons+=1
        for batch,prefix in SHAPES:
            tag=f'b{batch}-s{prefix}';path=OUT/f'device-{tag}';report=read(checked(path/f'{tag}-fp16.json'))
            healthy(report);assert report['experiment']==current and report['device_build']==production
            assert len(report['graphs'])==len(report['build_flags'])==60
            assert all(flag.endswith(' PERF_COUNT="YES"') and flag.count('PERF_COUNT')==1 for flag in report['build_flags'])
            bound(path,report['profiling_build'])
            with np.load(checked(path/f'{tag}-fp16-outputs.npz',report['outputs_sha256'])) as values:
                assert len(values.files)==454
                for i in range(182):
                    np.testing.assert_array_equal(values[f'plain/{i}'].view(np.uint8),values[f'profile/{i}'].view(np.uint8));comparisons+=1
                    old=0 if i==0 else 91 if i==1 else i-1 if i<92 else i
                    np.testing.assert_array_equal(values[f'plain/{i}'].view(np.uint8),ordinary[f'{tag}/{old}'].view(np.uint8));comparisons+=1
                for i in range(2,92):
                    np.testing.assert_array_equal(values[f'plain/{i}'].view(np.uint8),values[f'seed/{i}'].view(np.uint8));comparisons+=1
            for graph in report['graphs']:
                blob=checked(path/(graph['tag']+'.blob'),graph['blob_sha256']).read_bytes()
                patch=kernel_patches(blob,graph['patch_reports'],norm)
                assert len(graph['samples'])==3 and len({s['raw_sha256'] for s in graph['samples']})==3
                for index,sample in enumerate(graph['samples']):
                    data=checked(path/f"{graph['tag']}-{index}-tasks.bin",sample['raw_sha256']).read_bytes()
                    assert sample['task_record_bytes']==ct.sizeof(Task)==344 and len(data)==344*len(sample['tasks'])
                    for offset,task in enumerate(sample['tasks']):
                        raw=Task.from_buffer_copy(data[offset*344:(offset+1)*344])
                        for key,_ in Task._fields_:
                            value=getattr(raw,key)
                            if key in ('name','layer_type'):value=decode(value)
                            assert task[key]==value
                        stat=cycles.setdefault(str(task['exec_type']),dict(tasks=0,nonzero_active=0,nonzero_stall=0))
                        stat['tasks']+=1
                        for key in ('active','stall'):stat['nonzero_'+key]+=int(task[key+'_cycles']!=0)
                    tasks=sample['tasks'];task_records+=len(tasks)
                    valid=[t for t in tasks if t['duration_ns']]
                    span=max(t['start_time_ns']+t['duration_ns'] for t in valid)-min(t['start_time_ns'] for t in valid)
                    assert span/1e6==sample['timeline_span_ms'] and union_ns(tasks)/1e6==sample['observed_union_ms']
                    assert (span-union_ns(tasks))/1e6==sample['unobserved_gaps_ms']
                    assert len([t for t in tasks if rmsnorm(t)])==patch['invocations']
                    for kind,name in ((1,'dpu'),(2,'shave'),(3,'dma')):
                        chosen=[t for t in tasks if t['exec_type']==kind]
                        assert sample['engines'][name]==dict(tasks=len(chosen),sum_ms=sum(t['duration_ns'] for t in chosen)/1e6,union_ms=union_ns(chosen)/1e6)
                    assert not sample['dma_duration_available']
            for operation in ('prefill','decode'):
                graphs=[g for g in report['graphs'] if g['operation']==operation];assert sorted(g['layer'] for g in graphs)==list(range(30))
                samples=[]
                for i,sample in enumerate(report[operation]['samples']):
                    assert sample['native_call_counts'][3:]==[30]*6
                    value=dict(api_ms=sample['ms'],wait_ms=sample['host_wait_ms'],shared_alloc_calls=sample['shared_alloc_calls'],shared_free_calls=sample['shared_free_calls'],span_ms=sum(g['samples'][i]['timeline_span_ms'] for g in graphs))
                    selectors=dict(rmsnorm=rmsnorm,dpu=lambda t:t['exec_type']==1,shave=lambda t:t['exec_type']==2,
                        select=lambda t:t['exec_type']==2 and t['layer_type']=='Select',softmax=lambda t:t['exec_type']==2 and '/softmax/' in t['name'])
                    for name,selector in selectors.items():value[name+'_ms']=sum(union_ns([t for t in g['samples'][i]['tasks'] if selector(t)])/1e6 for g in graphs)
                    for field in ('active_cycles','stall_cycles'):
                        value['rmsnorm_'+field]=sum(t[field] for g in graphs for t in g['samples'][i]['tasks'] if rmsnorm(t))
                    value['rmsnorm_span_percent']=100*value['rmsnorm_ms']/value['span_ms'];samples.append(value)
                assert len(samples)==3 and st.median(s['api_ms'] for s in samples)==report[operation]['median_ms']
                plain=next(r for r in baseline['rows'] if (r['batch'],r['prefix'],r['operation'])==(batch,prefix,operation))
                host_row=next(r for r in host['rows'] if (r['batch'],r['prefix'],r['operation'])==(batch,prefix,operation))
                assert len(host_row['host_samples'])==62
                rows.append(dict(tag=tag,operation=operation,plain_api_ms=plain['median_ms']['baseline'],host_api_ms=host_row['median_ms']['baseline'],
                    plain_warm_delta=plain['warm_delta'],host_stages_ms=host_row['median_host_stage_ms'],task_samples=samples,task_medians={k:st.median(s[k] for s in samples) for k in samples[0]}))
    candidate=OUT/'row4';experiments=[]
    numeric=candidate/'numeric-probe.json'
    if numeric.exists():
        report=read(checked(numeric));healthy(report);assert len(report['checks'])==112 and all(c['bit_differences']==0 for c in report['checks'])
        assert report['build']==read(checked(candidate/'build.json'))
        assert report['build']['production']==current and report['build']['builder_sha256']==sha(ROOT/'npunlock-probe/research/rmsnorm_row4.py')
        for name,info in report['build']['kernels'].items():
            checked(candidate/(name+'.elf'),info['elf_sha256'])
            if 'source_sha256' in info:checked(candidate/(name+'.c'),info['source_sha256']);checked(candidate/(name+'.s'),info['assembly_sha256'])
        for row in report['timings']:timing(row,21*4)
        experiments.append(dict(case='numeric',checks=112,timings=report['timings']))
    for path in sorted(candidate.glob('model-*.json')):
        report=read(checked(path));healthy(report);assert report['experiment']==current and len(report['graphs'])==60
        from npunlock.npu4000 import kernel_text
        code=kernel_text((candidate/'codec.elf').read_bytes())
        for graph in report['graphs']:
            blobs={name:checked(candidate/(graph['tag']+'-'+name+'.blob')).read_bytes() for name in ('baseline','codec')}
            old=kernel_patches(blobs['baseline'],graph['baseline'],norm)
            patches=graph['codec'];assert [p['carrier'] for p in patches]==[p['carrier'] for p in graph['baseline']]
            patch=patches[0];start=patch['code_offset'];end=start+patch['window_bytes']
            assert patch['kernel_elf_sha256']==report['build']['kernels']['codec']['elf_sha256']
            assert blobs['codec'][start:start+len(code)]==code and patch['code_bytes']==len(code)
            assert blobs['codec'][:start]+blobs['codec'][end:]==blobs['baseline'][:start]+blobs['baseline'][end:]
            assert sha(candidate/(graph['tag']+'-codec.blob'))==patches[-1]['patched_sha256']
            assert (start,end)==(old['code_offset'],old['code_offset']+old['window_bytes'])
            assert graph['layout_sha256']==hashlib.sha256(blobs['codec'][:start]+bytes(end-start)+blobs['codec'][end:]).hexdigest()
        with np.load(checked(candidate/(report['label']+'-outputs.npz'),report['outputs_sha256'])) as values:
            assert len(values.files)==454
            for i in range(182):np.testing.assert_array_equal(values[f'baseline/{i}'].view(np.uint8),values[f'codec/{i}'].view(np.uint8));comparisons+=1
            for i in range(90):np.testing.assert_array_equal(values[f'baseline/{i+1}'].view(np.uint8),values[f'preserved/{i}'].view(np.uint8));comparisons+=1
        for row in report['timings']:timing(row,31*4*30)
        experiments.append(dict(case=report['label'],timings=report['timings']))
    sources=dict(current['sources'])
    for name in ('npunlock-probe/research/rmsnorm_row4.py','npunlock-probe/research/shave_fp32_audit.py'):sources[name]=sha(ROOT/name)
    for name,digest in sources.items():
        path=OUT/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path);checked(path,digest)
    if (OUT/'compression.json').exists():
        compression=read(checked(OUT/'compression.json'));assert compression['status']=='passed' and len(compression['files'])==165
        for entry in compression['files']:
            path=OUT/entry['path'];assert path.resolve().is_relative_to(OUT.resolve()) and entry['hash_preserved']
            checked(path,entry['sha256'])
    if (OUT/'REPORT.md').exists():checked(OUT/'REPORT.md')
    assembly={}
    for name in ('baseline','codec'):
        text=checked(candidate/(name+'.s')).read_text(encoding='utf-8')
        loop=text.split('.LBB0_37:\n',1)[1].split('\n.LBB0_',1)[0]
        loop=loop.split('BRU.BRA .LBB0_37',1)[0]+'BRU.BRA .LBB0_37\n'
        (candidate/(name+'-reduction-loop.txt')).write_text(loop,encoding='utf-8')
        assembly[name]=dict(loop_label='.LBB0_37',counted_scope='Static prefix through backward branch, excluding subsequent delay-slot instructions; not dynamic cycles',vector_multiply_instructions=len(re.findall(r'VAU.MUL.f32',loop)),
            vector_add_sub_instructions=len(re.findall(r'VAU.(?:ADD|SUB).f32',loop)),
            scalar_add_sub_instructions=len(re.findall(r'SAU.(?:ADD|SUB).f32',loop)),
            stack_load_store_instructions=sum(bool(re.search(r'LSU\d.\w*\.(?:32|64|128).*\bi19\b',line)) for line in loop.splitlines()),
            folded_spill_reload_comments=loop.count('Folded Spill')+loop.count('Folded Reload'))
    summary=dict(status='passed',production=current,rows=rows,experiments=experiments,cycle_fields=cycles,assembly=assembly,
        task_records=task_records,fresh_task_datasets=360,bit_equal_comparisons=comparisons,files_sha256=files,
        limitations='Resident decoder only. Task profiles change scheduling; per-layer interval unions may overlap across categories. DMA timing unavailable. Static instructions are not hardware pipeline utilization.')
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Passed',task_records,'task records;',comparisons,'array comparisons; cycles',cycles)
    for row in rows:print(row['tag'],row['operation'],row['task_medians'])
    for row in experiments:
        print(row['case'],[(x.get('rows',x.get('operation')),x['median_ms']) for x in row['timings']])


if __name__=='__main__':main()

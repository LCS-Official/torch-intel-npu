"""Recompute timing summaries and validate saved profiling/output evidence."""
import ctypes as ct
import json
from pathlib import Path
import statistics as stats
import numpy as np
from w8_performance import OUT, ROOT, SHAPES, MODES, metadata, read, sha
from torch_npu_device_profile import union_ns
from torch_npu_device_profile_audit import rmsnorm, trace
from torch_npu_lowbit import dpu_modes
from dpu_diagnostic import Task, decode


def main():
    expected=metadata();benchmarks=[];hosts=[];devices=[];comparisons=0;calls=0;task_records=0;graph_count=0
    def final(report):
        nonlocal calls
        assert report['status']=='passed';s=report['final_stats'];calls+=s['npu_calls']
        assert s['closed'] and not s['context_failed']
        assert all(s[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
    original=np.load(OUT/'benchmark-0-outputs.npz',allow_pickle=False)
    for label in ('benchmark-0','benchmark-1','host'):
        report=read(OUT/f'{label}.json');final(report)
        assert all(report[k]==v for k,v in expected.items())
        path=OUT/f'{label}-outputs.npz';assert sha(path)==report['outputs_sha256']
        with np.load(path,allow_pickle=False) as arrays:
            assert set(arrays.files)==set(original.files)
            for key in arrays.files:
                np.testing.assert_array_equal(arrays[key].view(np.uint8),original[key].view(np.uint8));comparisons+=1
        assert len(report['rows'])==4
        for row in report['rows']:
            assert (row['batch'],row['prefix']) in SHAPES and row['operation'] in ('prefill','decode')
            assert row['warm_delta']['npu_calls']==744
            assert all(len(row['samples_ms'][name])==62 for name in MODES) and len(row['pairs_ms'])==31
            assert row['median_ms']=={name:stats.median(row['samples_ms'][name]) for name in MODES}
            compact={k:row[k] for k in ('batch','prefix','operation','median_ms','positive_rounds','warm_delta')}
            compact['label']=label
            compact['calibrated_latency_change_percent']=100*(row['median_ms']['calibrated']/row['median_ms']['baseline']-1)
            if label!='host':benchmarks.append(compact);continue
            for name in MODES:
                values=row['host'][name]['samples'];assert len(values)==62
                stages=row['host'][name]['median_stage_ms']
                assert stages=={stage:stats.median(v['times_ms'][stage] for v in values) for stage in stages}
                assert all(v['counts'][stage]==4 for v in values for stage in ('append_graph','close_list','submit','wait','reset_list','reset_fence'))
                assert all(v['counts'][stage]==0 for v in values for stage in ('shared_alloc','shared_free'))
                command_stages=('set_argument','append_graph','close_list','submit','reset_list','reset_fence')
                hosts.append(dict(batch=row['batch'],prefix=row['prefix'],operation=row['operation'],mode=name,
                    api_median_ms=row['median_ms'][name],stage_median_ms=stages,
                    command_median_ms=stats.median(sum(v['times_ms'][s] for s in command_stages) for v in values),
                    wait_api_mean_percent=100*stats.mean(v['times_ms']['wait'] for v in values)/stats.mean(row['samples_ms'][name])))
    kernel=sha(ROOT/'npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf')
    for batch,prefix in SHAPES:
        for mode in ('fp16','calibrated'):
            directory=OUT/f'device-{mode}';tag=f'b{batch}-s{prefix}-{"fp16" if mode=="fp16" else "w8a16"}'
            report=read(directory/f'{tag}.json');final(report)
            assert report['performance_metadata']==expected
            assert report['source_sha256']==sha(ROOT/'torch_npu_device_profile.py')
            assert report['profiling_dll_sha256']==sha(directory/'session_task_profile.dll')==report['profile_build']['dll_sha256']
            for name,digest in report['production_sha256'].items():assert sha(ROOT/name)==digest
            path=directory/f'{tag}-outputs.npz';assert sha(path)==report['outputs_sha256']
            with np.load(path,allow_pickle=False) as arrays:
                for i in range(26):
                    reference=original[f'b{batch}-s{prefix}-{mode}/{i}']
                    for variant in ('plain','profile'):
                        np.testing.assert_array_equal(arrays[f'{variant}/{i}'].view(np.uint8),reference.view(np.uint8));comparisons+=1
                for i in range(2,14):
                    np.testing.assert_array_equal(arrays[f'seed/{i}'].view(np.uint8),arrays[f'plain/{i}'].view(np.uint8));comparisons+=1
            assert len(report['graphs'])==8 and len(report['build_flags'])==8
            for graph in report['graphs']:
                graph_count+=1;blob=(directory/f"{graph['tag']}.blob").read_bytes();assert sha(directory/f"{graph['tag']}.blob")==graph['blob_sha256']
                assert all(p['kernel_elf_sha256']==kernel and p['patched_sha256']==graph['blob_sha256'] for p in graph['patch_reports'])
                if mode=='calibrated':assert any(t['wmode']==2 and t['amode']==0 and not t['elementwise'] for t in dpu_modes(blob))
                assert len(graph['samples'])==3 and len({s['raw_sha256'] for s in graph['samples']})==3
                for i,sample in enumerate(graph['samples']):
                    path=directory/f"{graph['tag']}-{i}-tasks.bin";data=path.read_bytes();assert sha(path)==sample['raw_sha256']
                    assert len(data)==ct.sizeof(Task)*len(sample['tasks']);task_records+=len(sample['tasks'])
                    for j,task in enumerate(sample['tasks']):
                        raw=Task.from_buffer_copy(data[j*ct.sizeof(Task):(j+1)*ct.sizeof(Task)])
                        for name,_ in Task._fields_:
                            value=getattr(raw,name);assert task[name]==(decode(value) if name in ('name','layer_type') else value)
                    assert not sample['dma_duration_available']
                    assert sample['observed_union_ms']==union_ns(sample['tasks'])/1e6
            for operation in ('prefill','decode'):
                graphs=sorted((g for g in report['graphs'] if g['operation']==operation),key=lambda g:g['layer']);assert [g['layer'] for g in graphs]==list(range(4))
                samples=[]
                for i,host in enumerate(report[operation]['samples']):
                    assert host['native_call_counts'][3:]==[4]*6 and host['shared_alloc_calls']==host['shared_free_calls']==0
                    selected=[g['samples'][i] for g in graphs]
                    values=dict(api_ms=host['ms'],wait_ms=host['host_wait_ms'],span_ms=sum(s['timeline_span_ms'] for s in selected))
                    categories={'rmsnorm':rmsnorm,'dpu':lambda t:t['exec_type']==1,'shave':lambda t:t['exec_type']==2,
                        'select':lambda t:t['exec_type']==2 and t['layer_type']=='Select',
                        'softmax':lambda t:t['exec_type']==2 and '/softmax/' in t['name']}
                    for category,select in categories.items():values[category+'_ms']=sum(union_ns([t for t in s['tasks'] if select(t)])/1e6 for s in selected)
                    values['gaps_ms']=sum(s['unobserved_gaps_ms'] for s in selected)
                    for category in ('rmsnorm','dpu','select','softmax'):values[category+'_span_percent']=100*values[category+'_ms']/values['span_ms']
                    samples.append(values)
                medians={k:stats.median(s[k] for s in samples) for k in samples[0]}
                assert medians['api_ms']==report[operation]['median_ms']
                devices.append(dict(batch=batch,prefix=prefix,mode=mode,operation=operation,medians=medians,samples=samples))
                trace(graphs,1,directory/f'{tag}-{operation}-trace.json')
    measurement_calls=calls
    candidate_dir=OUT/'rmsnorm-o3';build=read(candidate_dir/'build.json')
    from npunlock.npu4000 import kernel_text
    assert build['status']=='built' and build['production']==expected and not build['code_identical']
    assert build['builder_sha256']==sha(ROOT/'npunlock-probe/research/rmsnorm_o3.py')
    assert (candidate_dir/'codec.c').read_bytes()==(ROOT/'npunlock/tests/fixtures/npu4000/rmsnorm.c').read_bytes()
    for name,record in build['kernels'].items():
        data=(candidate_dir/f'{name}.elf').read_bytes()
        assert sha(candidate_dir/f'{name}.elf')==record['elf_sha256'] and len(kernel_text(data))==record['text_bytes']
    assert build['kernels']['baseline']['elf_sha256']==kernel
    candidates={}
    for case,label in (('numeric','probe'),('model','process0')):
        report=read(candidate_dir/f'{case}-{label}.json');final(report)
        assert report['production']==expected and report['wrapper_sha256']==build['builder_sha256']
        assert report['source_sha256']==sha(ROOT/'npunlock-probe/research/rmsnorm_reduce_probe.py')
        assert report['build']==build
        if case=='numeric':assert len(report['checks'])==104 and all(r['bit_differences']==0 for r in report['checks'])
        else:
            assert len(report['checks'])==4
            for check in report['checks']:
                assert check['bit_equal_arrays']==26 and check['seed_preserved']
                path=candidate_dir/f"{check['tag']}-outputs.npz";assert sha(path)==check['outputs_sha256']
                with np.load(path,allow_pickle=False) as arrays:
                    for i in range(26):
                        np.testing.assert_array_equal(arrays[f"{check['tag']}/baseline/{i}"].view(np.uint8),arrays[f"{check['tag']}/codec/{i}"].view(np.uint8));comparisons+=1
                    for i in range(12):
                        np.testing.assert_array_equal(arrays[f"{check['tag']}/baseline/{i+1}"].view(np.uint8),arrays[f"{check['tag']}/preserved/{i}"].view(np.uint8));comparisons+=1
            for evidence in report['carrier_evidence']:
                old=evidence['baseline'];a=old['code_offset'];b=a+old['window_bytes']
                original=(candidate_dir/f"{evidence['tag']}-baseline.blob").read_bytes()
                proposed=(candidate_dir/f"{evidence['tag']}-codec.blob").read_bytes()
                assert original[:a]+original[b:]==proposed[:a]+proposed[b:]
        rows=[]
        for row in report['timings']:
            assert row['median_ms']=={name:stats.median(values) for name,values in row['samples_ms'].items()}
            rows.append(dict(**{k:row[k] for k in ('rows','tag','operation') if k in row},median_ms=row['median_ms'],
                latency_change_percent=100*(row['median_ms']['codec']/row['median_ms']['baseline']-1),positive_rounds=row['positive_rounds']['codec']))
        candidates[case]=dict(checks=len(report['checks']),rows=rows)
    summary=dict(status='passed',metadata=expected,benchmarks=benchmarks,hosts=hosts,devices=devices,candidate=candidates,
        measurement_npu_calls=measurement_calls,candidate_npu_calls=calls-measurement_calls,
        total_npu_calls=calls,bit_equal_comparisons=comparisons,graphs=graph_count,fresh_task_datasets=graph_count*3,
        task_records=task_records,dma_duration_available=False,auditor_sha256=sha(Path(__file__)))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Passed',calls,'NPU calls;',comparisons,'bit comparisons;',graph_count,'graphs;',task_records,'task records')
    for row in benchmarks:print(row)
    for row in devices:print('device',row['batch'],row['mode'],row['operation'],row['medians'])


if __name__=='__main__':main()

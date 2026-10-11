"""Verify KV candidates and raw task samples without executing the NPU."""
from collections import Counter
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics as stats
import xml.etree.ElementTree as ET
import numpy as np
from kv_gather_probe import OUT, ROOT, SOURCES, sha
from torch_npu_device_profile import union_ns
from torch_npu_device_profile_audit import rmsnorm, trace
from dpu_diagnostic import Task, decode
from w8_performance import profile_binary


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    sources={name:sha(ROOT/name) for name in SOURCES}
    assert sha(OUT/'before/torch_npu_smollm.py')==sources['torch_npu_smollm.py']
    scripts={sha(ROOT/'npunlock-probe/research/kv_gather_probe.py'),sha(OUT/'measurement-v1.py')}
    calls=0;comparisons=0;cross=0;timings=[];devices=[];records=0;graph_count=0;profiles=0
    def final(report):
        nonlocal calls
        assert report['status']=='passed';s=report['final_stats'];calls+=s['npu_calls']
        assert s['closed'] and not s['context_failed']
        assert all(s[key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
    with np.load(OUT/'gather-0/outputs.npz',allow_pickle=False) as original:
        for label in ('gather-0','gather-1','flat-0','i32-0'):
            directory=OUT/label;report=read(directory/'probe.json');final(report);variant=report['candidate']
            assert report['sources']==sources and report['source_sha256'] in scripts
            assert sha(directory/'candidate.py')==report['candidate_sha256']
            assert report['calibration_weights_sha256']==sha(ROOT/'npunlock-probe/research/results/rmsnorm-encode/calibrated-current/weights.npz')
            path=directory/'outputs.npz';assert sha(path)==report['outputs_sha256']
            assert len(report['checks'])==172 and len({c['tag'] for c in report['checks']})==172
            with np.load(path,allow_pickle=False) as arrays:
                assert len(arrays.files)==report['output_array_count']==4472
                for check in report['checks']:
                    assert check['bit_equal_arrays']==13
                    for i in range(13):
                        key=f"{check['tag']}/baseline/{i}";a=arrays[key]
                        b=arrays[f"{check['tag']}/{variant}/{i}"]
                        np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8));comparisons+=1
                        if label!='gather-0':
                            np.testing.assert_array_equal(a.view(np.uint8),original[key].view(np.uint8));cross+=1
                for batch,prefix,mode in ((1,32,'fp16'),(1,32,'w8a16'),(4,16,'fp16'),(4,16,'w8a16')):
                    tag=f'b{batch}-s{prefix}-{mode}'
                    for position in range(prefix+1,65):assert any(c['tag']==f'{tag}/chain-{position}' for c in report['checks'])
                    for name in ('baseline',variant):
                        for i in range(13):
                            np.testing.assert_array_equal(arrays[f'{tag}/repeat/{name}/{i}'].view(np.uint8),arrays[f'{tag}/decode/{name}/{i}'].view(np.uint8))
            assert len(report['timings'])==4 and len(report['graphs'])==32
            for row in report['timings']:
                rounds=report['rounds'];samples=row['samples_ms'];pairs=row['pairs_ms']
                assert rounds==31 and len(pairs)==rounds and all(len(v)==2*rounds for v in samples.values())
                assert row['median_ms']=={name:stats.median(v) for name,v in samples.items()}
                assert row['warm_delta']['npu_calls']==496 and row['warm_delta']['driver_allocations']==0
                for name,values in samples.items():assert all(stats.mean(values[2*i:2*i+2])==pairs[i][name] for i in range(rounds))
                wins=sum(p[variant]<p['baseline'] for p in pairs);assert row['positive_rounds'][variant]==wins
                timings.append(dict(label=label,candidate=variant,tag=row['tag'],median_ms=row['median_ms'],positive_rounds=wins,rounds=rounds,
                    latency_decrease_percent=100*(1-row['median_ms'][variant]/row['median_ms']['baseline']),
                    paired_median_decrease_percent=stats.median(100*(1-p[variant]/p['baseline']) for p in pairs)))
            for graph in report['graphs']:
                graph_count+=1
                assert sha(directory/f"{graph['stem']}.xml")==graph['xml_sha256']
                assert sha(directory/f"{graph['stem']}.blob")==graph['blob_sha256']==graph['evidence']['blob_sha256']
                assert graph['evidence']['rmsnorm']['kernel_elf_sha256']==sources['npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf']
                types=Counter(n.attrib['type'] for n in ET.parse(directory/f"{graph['stem']}.xml").findall('./layers/layer'))
                if '-baseline-' in graph['stem']:assert types['Select']==3 and types['Gather']==4
                elif variant.startswith('gather'):assert types['Select']==2 and types['Gather']==6
                else:assert types['Select']==1 and types['ScatterUpdate']==2

    plain={};_,profile_build=profile_binary()
    for variant in ('baseline','gather'):
        directory=OUT/f'device-{variant}';report=read(directory/'b4-s16-w8a16.json');final(report)
        assert report['sources']==sources and report['entry_sha256']==sha(ROOT/'npunlock-probe/research/kv_gather_profile.py')
        assert report['source_sha256']==sha(ROOT/'torch_npu_device_profile.py')
        assert report['profile_build']==profile_build
        assert sha(directory/'session_task_profile.dll')==report['profiling_dll_sha256']==report['profile_build']['dll_sha256']
        for name,digest in report['production_sha256'].items():assert sha(ROOT/name)==digest
        if variant=='gather':assert sha(directory/'candidate.py')==report['candidate_sha256']==read(OUT/'gather-0/probe.json')['candidate_sha256']
        path=directory/'b4-s16-w8a16-outputs.npz';assert sha(path)==report['outputs_sha256']
        with np.load(path,allow_pickle=False) as arrays:
            for i in range(26):
                a=arrays[f'plain/{i}'];np.testing.assert_array_equal(a.view(np.uint8),arrays[f'profile/{i}'].view(np.uint8));profiles+=1
                if variant=='baseline':plain[i]=a.copy()
                else:np.testing.assert_array_equal(a.view(np.uint8),plain[i].view(np.uint8));profiles+=1
            for i in range(2,14):np.testing.assert_array_equal(arrays[f'seed/{i}'].view(np.uint8),arrays[f'plain/{i}'].view(np.uint8));profiles+=1
        assert len(report['graphs'])==8
        for graph in report['graphs']:
            graph_count+=1;assert sha(directory/f"{graph['tag']}.blob")==graph['blob_sha256']
            assert len(graph['samples'])==3 and len({s['raw_sha256'] for s in graph['samples']})==3
            assert all(p['kernel_elf_sha256']==sources['npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf'] and p['patched_sha256']==graph['blob_sha256'] for p in graph['patch_reports'])
            for i,sample in enumerate(graph['samples']):
                path=directory/f"{graph['tag']}-{i}-tasks.bin";data=path.read_bytes();assert sha(path)==sample['raw_sha256']
                assert len(data)==ct.sizeof(Task)*len(sample['tasks']);records+=len(sample['tasks'])
                for j,task in enumerate(sample['tasks']):
                    raw=Task.from_buffer_copy(data[j*ct.sizeof(Task):(j+1)*ct.sizeof(Task)])
                    for name,_ in Task._fields_:
                        value=getattr(raw,name);assert task[name]==(decode(value) if name in ('name','layer_type') else value)
                assert not sample['dma_duration_available'] and sample['observed_union_ms']==union_ns(sample['tasks'])/1e6
        for operation in ('prefill','decode'):
            graphs=[g for g in report['graphs'] if g['operation']==operation];rows=[]
            for i,host in enumerate(report[operation]['samples']):
                selected=[g['samples'][i] for g in graphs]
                row=dict(api_ms=host['ms'],wait_ms=host['host_wait_ms'],span_ms=sum(s['timeline_span_ms'] for s in selected),gaps_ms=sum(s['unobserved_gaps_ms'] for s in selected))
                categories={'norm':rmsnorm,'shave':lambda t:t['exec_type']==2,'dpu':lambda t:t['exec_type']==1,
                    'select':lambda t:t['exec_type']==2 and t['layer_type']=='Select',
                    'gather':lambda t:t['exec_type']==2 and t['layer_type']=='Gather'}
                for category,predicate in categories.items():row[category+'_ms']=sum(union_ns([t for t in s['tasks'] if predicate(t)])/1e6 for s in selected)
                rows.append(row)
            medians={key:stats.median(row[key] for row in rows) for key in rows[0]}
            assert medians['api_ms']==report[operation]['median_ms']
            counts=Counter(t['layer_type'] for g in graphs for t in g['samples'][0]['tasks'] if t['exec_type']==2)
            devices.append(dict(variant=variant,operation=operation,medians=medians,samples=rows,shave_task_types=dict(counts)))
            trace(graphs,1,directory/f'b4-s16-w8a16-{operation}-trace.json')
    rejected=read(OUT/'nd-0/probe.json')
    assert rejected['status']=='failed' and 'ir2blob.compile' in rejected['traceback'] and not rejected['failure_stats']['context_failed']
    assert rejected['candidate']=='scatter_nd' and rejected['source_sha256'] in scripts
    snapshots=read(OUT/'snapshots.json')
    for name,digest in snapshots.items():assert sha(ROOT/name)==digest==sha(OUT/'sources'/name)
    result=dict(status='passed',production_changed=False,total_successful_npu_calls=calls,candidate_array_comparisons=comparisons,
        cross_process_baseline_array_comparisons=cross,profile_array_comparisons=profiles,graphs=graph_count,task_records=records,
        sources=sources,timings=timings,devices=devices,rejected_decode_candidate='scatter_nd',snapshot_files=len(snapshots),auditor_sha256=sha(Path(__file__)))
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','timings')},indent=2))


if __name__=='__main__':main()

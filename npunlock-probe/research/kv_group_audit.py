"""Offline audit of graph-layout and same-blob Select kernel experiments."""
from collections import Counter
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics as stats
import xml.etree.ElementTree as ET
import numpy as np
from kv_gather_probe import ROOT, SOURCES, sha
from kv_select_kernel import OUT as KOUT, patch
from npunlock.npu4000 import kernel_text
from torch_npu_device_profile import union_ns
from torch_npu_device_profile_audit import rmsnorm, trace
from dpu_diagnostic import Task, decode
from w8_performance import profile_binary

OUT=KOUT.parent


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    sources={name:sha(ROOT/name) for name in SOURCES}
    assert sha(OUT/'before/torch_npu_smollm.py')==sources['torch_npu_smollm.py']
    kernel=(KOUT/'select.elf').read_bytes();manifest=read(KOUT/'build.json')
    assert manifest['elf_sha256']==sha(KOUT/'select.elf')
    assert manifest['source_sha256']==sha(KOUT/'select.c')==sha(ROOT/'npunlock-probe/research/kv_select.c')
    assert manifest['assembly_sha256']==sha(KOUT/'select.s') and manifest['text_bytes']==len(kernel_text(kernel))==1984
    assert manifest['builder_sha256']==sha(ROOT/'npunlock-probe/research/kv_select_kernel.py')
    old=read(KOUT/'initial-generic/build.json')
    assert old['elf_sha256']==sha(KOUT/'initial-generic/select.elf') and old['builder_sha256']==sha(KOUT/'initial-generic/builder.py')
    assert old['source_sha256']==sha(KOUT/'initial-generic/select.c') and old['assembly_sha256']==sha(KOUT/'initial-generic/select.s')
    totals=Counter();timings=[];devices=[];proofs=[]
    def final(report):
        assert report['status']=='passed';s=report['final_stats'];totals['successful_npu_calls']+=s['npu_calls']
        assert s['closed'] and not s['context_failed']
        assert all(s[key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
    def equal(a,b,category):
        np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8));totals[category]+=1
    cases=((OUT/'group-0','grouped_gqa',OUT/'measurement-group.py'),(OUT/'joint-0','joint_select',ROOT/'npunlock-probe/research/kv_group_probe.py'),
           (KOUT/'copy-0','select_shave_copy',None),(KOUT/'special-0','select_shave_copy',None),(KOUT/'special-1','select_shave_copy',None))
    with np.load(OUT/'group-0/outputs.npz',allow_pickle=False) as original:
        for directory,implementation,wrapper in cases:
            report=read(directory/'probe.json');final(report);variant=report['candidate']
            assert report['sources']==sources and report['source_sha256']==sha(ROOT/'npunlock-probe/research/kv_gather_probe.py')
            assert report['implementation']==implementation and sha(directory/'candidate.py')==report['candidate_sha256']
            assert report['wrapper_sha256']==sha(wrapper or ROOT/'npunlock-probe/research/kv_group_probe.py')
            assert report['calibration_weights_sha256']==sha(ROOT/'npunlock-probe/research/results/rmsnorm-encode/calibrated-current/weights.npz')
            if wrapper is None:
                assert report['entry_sha256']==sha(ROOT/'npunlock-probe/research/kv_select_probe.py')
                assert report['kernel_build']==(old if directory.name=='copy-0' else manifest)
            assert len(report['checks'])==172 and len({c['tag'] for c in report['checks']})==172
            assert sha(directory/'outputs.npz')==report['outputs_sha256']
            with np.load(directory/'outputs.npz',allow_pickle=False) as arrays:
                assert len(arrays.files)==report['output_array_count']==4472
                for check in report['checks']:
                    assert check['bit_equal_arrays']==13
                    for i in range(13):
                        key=f"{check['tag']}/baseline/{i}";a=arrays[key]
                        equal(a,arrays[f"{check['tag']}/{variant}/{i}"],'candidate_array_comparisons')
                        if directory.name!='group-0':equal(a,original[key],'cross_process_array_comparisons')
                for batch,prefix,mode in ((1,32,'fp16'),(1,32,'w8a16'),(4,16,'fp16'),(4,16,'w8a16')):
                    tag=f'b{batch}-s{prefix}-{mode}'
                    for position in range(prefix+1,65):assert any(c['tag']==f'{tag}/chain-{position}' for c in report['checks'])
                    for name in ('baseline',variant):
                        for i in range(13):equal(arrays[f'{tag}/repeat/{name}/{i}'],arrays[f'{tag}/decode/{name}/{i}'],'seed_reuse_array_comparisons')
            assert len(report['timings'])==4 and len(report['graphs'])==32
            for row in report['timings']:
                rounds=report['rounds'];samples=row['samples_ms'];pairs=row['pairs_ms']
                assert len(pairs)==rounds and all(len(v)==2*rounds for v in samples.values())
                assert row['median_ms']=={name:stats.median(v) for name,v in samples.items()}
                assert row['warm_delta']['npu_calls']==rounds*16 and row['warm_delta']['driver_allocations']==0
                for name,values in samples.items():assert all(stats.mean(values[2*i:2*i+2])==pairs[i][name] for i in range(rounds))
                wins=sum(p[variant]<p['baseline'] for p in pairs);assert row['positive_rounds'][variant]==wins
                timings.append(dict(label=directory.name,implementation=implementation,tag=row['tag'],median_ms=row['median_ms'],wins=wins,rounds=rounds,
                    latency_decrease_percent=100*(1-row['median_ms'][variant]/row['median_ms']['baseline']),
                    paired_median_decrease_percent=stats.median(100*(1-p[variant]/p['baseline']) for p in pairs)))
            for graph in report['graphs']:
                totals['graphs']+=1;stem=graph['stem'];xml=directory/f'{stem}.xml';blob=directory/f'{stem}.blob'
                assert sha(xml)==graph['xml_sha256'] and sha(blob)==graph['blob_sha256']==graph['evidence']['blob_sha256']
                assert graph['evidence']['rmsnorm']['kernel_elf_sha256']==sources['npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf']
                if wrapper is None and f'-{variant}-' in stem:
                    baseline=directory/stem.replace(f'-{variant}-','-baseline-')
                    assert xml.read_bytes()==baseline.with_suffix('.xml').read_bytes()
                    chosen=(KOUT/'initial-generic/select.elf').read_bytes() if directory.name=='copy-0' else kernel
                    rebuilt,evidence=patch(baseline.with_suffix('.blob').read_bytes(),chosen)
                    assert rebuilt==blob.read_bytes() and evidence['invocations']==20
                    proofs.append(dict(label=directory.name,stem=stem,code_offset=evidence['code_offset'],code_bytes=evidence['code_bytes'],invocations=20))
                types=Counter(n.attrib['type'] for n in ET.parse(xml).findall('./layers/layer'))
                if '-baseline-' in stem or wrapper is None:assert types['Select']==3 and types['Gather']==4
                elif implementation=='grouped_gqa':assert types['Select']==3 and types['Gather']==2
                else:assert types['Select']==2 and types['Concat']>=1

    plain={};_,build=profile_binary()
    for variant in ('baseline','gather'):
        directory=KOUT/f'device-{variant}';report=read(directory/'b4-s16-w8a16.json');final(report)
        assert report['sources']==sources and report['profile_build']==build
        assert report['entry_sha256']==sha(ROOT/'npunlock-probe/research/kv_gather_profile.py')
        assert report['wrapper_sha256']==sha(ROOT/'npunlock-probe/research/kv_select_profile.py')
        assert report['source_sha256']==sha(ROOT/'torch_npu_device_profile.py')
        assert sha(directory/'session_task_profile.dll')==report['profiling_dll_sha256']==build['dll_sha256']
        for name,digest in report['production_sha256'].items():assert sha(ROOT/name)==digest
        if variant=='gather':assert sha(directory/'candidate.py')==report['candidate_sha256']==read(KOUT/'special-1/probe.json')['candidate_sha256']
        path=directory/'b4-s16-w8a16-outputs.npz';assert sha(path)==report['outputs_sha256']
        with np.load(path,allow_pickle=False) as arrays:
            for i in range(26):
                a=arrays[f'plain/{i}'];equal(a,arrays[f'profile/{i}'],'profile_array_comparisons')
                if variant=='baseline':plain[i]=a.copy()
                else:equal(a,plain[i],'profile_array_comparisons')
            for i in range(2,14):equal(arrays[f'seed/{i}'],arrays[f'plain/{i}'],'profile_array_comparisons')
        assert len(report['graphs'])==8
        for graph in report['graphs']:
            totals['graphs']+=1;path=directory/f"{graph['tag']}.blob";assert sha(path)==graph['blob_sha256']
            rms=graph['patch_reports'][0];assert rms['kernel_elf_sha256']==sources['npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf']
            if variant=='baseline':assert rms['patched_sha256']==graph['blob_sha256']
            else:
                rebuilt,select=patch((KOUT/'device-baseline'/path.name).read_bytes(),kernel)
                assert rebuilt==path.read_bytes() and graph['patch_reports'][1]==json.loads(json.dumps(select))
                if select['invocations']:assert select['original_sha256']==rms['patched_sha256'] and select['patched_sha256']==graph['blob_sha256']
                else:assert graph['operation']=='prefill' and rms['patched_sha256']==graph['blob_sha256']
            assert len(graph['samples'])==3 and len({s['raw_sha256'] for s in graph['samples']})==3
            for i,sample in enumerate(graph['samples']):
                path=directory/f"{graph['tag']}-{i}-tasks.bin";data=path.read_bytes();assert sha(path)==sample['raw_sha256']
                assert len(data)==ct.sizeof(Task)*len(sample['tasks']);totals['task_records']+=len(sample['tasks'])
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
                    'select':lambda t:t['exec_type']==2 and t['layer_type']=='Select','gather':lambda t:t['exec_type']==2 and t['layer_type']=='Gather'}
                for category,predicate in categories.items():row[category+'_ms']=sum(union_ns([t for t in s['tasks'] if predicate(t)])/1e6 for s in selected)
                rows.append(row)
            medians={key:stats.median(row[key] for row in rows) for key in rows[0]};assert medians['api_ms']==report[operation]['median_ms']
            counts=Counter(t['layer_type'] for g in graphs for t in g['samples'][0]['tasks'] if t['exec_type']==2)
            devices.append(dict(variant=variant,operation=operation,medians=medians,samples=rows,shave_task_types=dict(counts)))
            trace(graphs,1,directory/f'b4-s16-w8a16-{operation}-trace.json')

    for label in ('numeric','numeric-special'):
        directory=KOUT/label;numeric=read(directory/'numeric.json');final(numeric)
        script=directory/'measurement.py' if label=='numeric' else ROOT/'npunlock-probe/research/kv_select_numeric.py'
        assert numeric['source_sha256']==sha(script) and numeric['kernel_sha256']==manifest['elf_sha256']
        assert len(numeric['checks'])==40 and len(numeric['graphs'])==5 and sha(directory/'outputs.npz')==numeric['outputs_sha256']
        with np.load(directory/'outputs.npz',allow_pickle=False) as arrays:
            assert len(arrays.files)==numeric['output_array_count']==(240 if label=='numeric' else 245)
            for check in numeric['checks']:
                for name in ('baseline','copy'):
                    for i in (0,1):equal(arrays[f"{check['tag']}/{name}/{i}"],arrays[f"{check['tag']}/expected/{i}"],'raw_fp16_array_comparisons')
                wanted=arrays[f"{check['tag']}/expected/0"]
                if check['all_fp16_patterns']:assert check['elements']>=65536 and wanted.dtype==np.uint16
                if label=='numeric-special':
                    head,position=check['tag'].split('-p');position=int(position)
                    np.testing.assert_array_equal(arrays[f"{check['tag']}/copy/0"][:,:,position:position+1,:],arrays[f'{head}/current'])
            domain=np.unique(np.concatenate([arrays[f'h16-p{p}/copy/0'].reshape(-1) for p in (0,1,7,8,15,31,32,63)]))
            np.testing.assert_array_equal(domain,np.arange(65536,dtype=np.uint16))
        for graph in numeric['graphs']:
            stem=f"h{graph['heads']}";assert sha(directory/f'{stem}.xml')==graph['xml_sha256']
            rebuilt,evidence=patch((directory/f'{stem}-baseline.blob').read_bytes(),kernel)
            assert rebuilt==(directory/f'{stem}-copy.blob').read_bytes() and json.loads(json.dumps(evidence))==graph['patch']
            totals['graphs']+=2
    snapshots=read(OUT/'snapshots.json')
    for name,digest in snapshots.items():assert sha(ROOT/name)==digest==sha(OUT/'sources'/name)
    result=dict(status='passed',production_changed=False,default_promoted=False,totals=dict(totals),sources=sources,
        timings=timings,devices=devices,same_blob_replacement_proofs=proofs,kernel=manifest,snapshot_files=len(snapshots),auditor_sha256=sha(Path(__file__)))
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','production_changed','default_promoted','totals','devices')},indent=2))


if __name__=='__main__':main()

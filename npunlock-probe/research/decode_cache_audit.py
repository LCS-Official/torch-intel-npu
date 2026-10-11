"""Offline provenance, tensor bytes, counters and process-control audit."""
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from decode_cache_probe import ROOT,OUT,read,sha


def main():
    builds={name:read(OUT/name/'build.json') for name in ('before','after')}
    for name,build in builds.items():
        assert build['extension_sha256']==sha(OUT/name/'_torch_intel_npu.pyd')
        assert build['source_sha256']==sha(OUT/name/'backend.cpp')
        assert build['session_sha256']==sha(ROOT/'npunlock-probe/session-runtime/npunlock_session.dll')
    assert builds['after']==read(ROOT/'npunlock-probe/torch-device/build.json')
    assert builds['after']['source_sha256']==sha(ROOT/'npunlock-probe/torch-device/backend.cpp')
    assert builds['after']['extension_sha256']==sha(ROOT/'npunlock-probe/torch-device/_torch_intel_npu.pyd')
    assert sha(ROOT/'torch_intel_npu.py')==sha(OUT/'before/torch_intel_npu.py')==sha(OUT/'after/torch_intel_npu.py')
    for name,digest in read(OUT/'unchanged.json').items():assert sha(ROOT/name)==digest,name
    assert sha(OUT/'measurement-script.py')==sha(ROOT/'npunlock-probe/research/decode_cache_probe.py')
    assert sha(OUT/'generation-script.py')==sha(ROOT/'npunlock-probe/research/decode_cache_generate.py')
    result=dict(status='passed',builds=builds,model_rows=[],generation_rows=[],regressions={})
    def healthy(stats):
        assert stats['closed'] and not stats['context_failed']
        for field in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'):assert stats[field]==0,field
    for runtime in ('before','after'):
        report=read(OUT/f'allocation-{runtime}.json');assert report['status']=='passed' and report['build']==builds[runtime]
        assert report['source_sha256']==sha(OUT/'measurement-script.py');healthy(report['final_stats'])
        d=report['checks'][0]['counter_delta']
        assert d['buffer_allocations']==84 and d['npu_calls']==0 and d['driver_allocations']==(63 if runtime=='before' else 3)
        assert d['driver_allocations']+d['allocation_reuses']==84
        t=report['timings'][0];assert len(t['samples_ms'])==21 and statistics.median(t['samples_ms'])==t['median_ms']
    labels=('model-0-before','model-1-after','model-2-after','model-3-before');reports=[]
    for label in labels:
        report=read(OUT/f'{label}.json');reports.append(report)
        assert report['status']=='passed' and report['build']==builds[report['runtime']]
        assert report['source_sha256']==sha(OUT/'measurement-script.py') and report['outputs_sha256']==sha(OUT/f'{label}-outputs.npz')
        healthy(report['final_stats']);assert report['final_stats']['npu_calls']==1760
        assert len(report['checks'])==4 and all(c['seed_preserved'] and c['chain_length']==32 for c in report['checks'])
        for timing in report['timings']:
            count=31 if timing['operation']=='single-decode' else 32
            assert len(timing['samples_ms'])==count and statistics.median(timing['samples_ms'])==timing['median_ms']
            d=timing['counter_delta'];assert d['npu_calls']==count*4 and d['session_creations']==d['host_upload_copies']==0
            assert d['buffer_allocations']==d['driver_allocations']+d['allocation_reuses']
            if timing['operation']=='single-decode':
                assert d['host_download_copies']==0
                assert len(timing['cleanup_ms'])==31 and statistics.median(timing['cleanup_ms'])==timing['cleanup_median_ms']
                if timing['scope']=='shape-switch':assert d['driver_allocations']==(403 if report['runtime']=='before' else 0)
    with np.load(OUT/f'{labels[0]}-outputs.npz') as reference:
        assert len(reference.files)==312
        for label,report in zip(labels[1:],reports[1:]):
            assert [c['graph_sha256'] for c in report['checks']]==[c['graph_sha256'] for c in reports[0]['checks']]
            with np.load(OUT/f'{label}-outputs.npz') as actual:
                assert reference.files==actual.files
                for name in reference.files:np.testing.assert_array_equal(actual[name].view(np.uint8),reference[name].view(np.uint8))
    for timing in reports[0]['timings']:
        values={runtime:[next(t for t in report['timings'] if (t['tag'],t['scope'],t['operation'])==(timing['tag'],timing['scope'],timing['operation']))
                         for report in reports if report['runtime']==runtime] for runtime in ('before','after')}
        medians={runtime:statistics.mean(t['median_ms'] for t in rows) for runtime,rows in values.items()}
        result['model_rows'].append(dict(tag=timing['tag'],scope=timing['scope'],operation=timing['operation'],
            process_median_ms={runtime:[t['median_ms'] for t in rows] for runtime,rows in values.items()},mean_process_median_ms=medians,
            latency_reduction_percent=100*(1-medians['after']/medians['before']),
            driver_allocations={runtime:[t['counter_delta']['driver_allocations'] for t in rows] for runtime,rows in values.items()}))
    gen_labels=('generate-0-before','generate-1-after','generate-2-after','generate-3-before');generations=[]
    for label in gen_labels:
        report=read(OUT/f'{label}.json');generations.append(report)
        assert report['status']=='passed' and report['source_sha256']==sha(OUT/'generation-script.py')
        assert report['outputs_sha256']==sha(OUT/f'{label}-outputs.npz');healthy(report['final_stats'])
        assert report['npu_layers']==4 and report['cpu_layers']==26 and report['new_tokens']==32 and not report['stop_at_eos']
        assert len(report['rows'])==1 and report['checks'][0]['first_logits_relative_l2_cpu']<.03
        row=report['rows'][0];assert len(row['samples'])==3
        for sample in row['samples']:
            assert len(sample['steps_ms'])==len(sample['npu_with_transfers_ms'])==32
            assert sample['counter_delta']['session_creations']==0 and sample['counter_delta']['npu_calls']==128
        assert statistics.median(s['elapsed_ms'] for s in row['samples'])==row['median_elapsed_ms']
        assert 32000/row['median_elapsed_ms']==row['end_to_end_tokens_per_second']
        assert statistics.median(sum(s['steps_ms'][1:]) for s in row['samples'])==row['median_decode_ms']
        assert 31000/row['median_decode_ms']==row['decode_tokens_per_second']
    with np.load(OUT/f'{gen_labels[0]}-outputs.npz') as reference:
        assert len(reference.files)==12
        for label in gen_labels[1:]:
            with np.load(OUT/f'{label}-outputs.npz') as actual:
                assert reference.files==actual.files
                for name in reference.files:np.testing.assert_array_equal(actual[name].view(np.uint8),reference[name].view(np.uint8))
    for operation,field in (('end-to-end','median_elapsed_ms'),('decode','median_decode_ms')):
        medians={runtime:[g['rows'][0][field] for g in generations if g['runtime']==runtime] for runtime in ('before','after')}
        mean={runtime:statistics.mean(values) for runtime,values in medians.items()}
        result['generation_rows'].append(dict(operation=operation,process_median_ms=medians,mean_process_median_ms=mean,
            latency_reduction_percent=100*(1-mean['after']/mean['before']),
            tokens_per_second={runtime:(32000 if operation=='end-to-end' else 31000)/value for runtime,value in mean.items()}))
    for name in ('allocator-guards','idle-order','allocator-failure'):
        report=read(OUT/f'{name}.json');assert report['status']=='passed'
        if name=='allocator-failure':
            assert report['synthetic_failure'] and not report['actual_device_submission'] and report['driver_free_calls']==report['driver_reset_calls']==0
            assert report['final_stats']['context_failed'] and report['final_stats']['live_buffers']==report['final_stats']['cached_buffer_bytes']==0
        else:healthy(report['final_stats'])
        result['regressions'][name]='passed'
    for name,path in [('kv-guards','kv/guards.json'),('kv-boundary','kv/boundary.json'),('device','device.json'),('training','training.json')]:
        report=read(OUT/'regressions'/path);assert report['status'] in ('passed','passed_with_limitations');healthy(report['final_stats'])
        for file,digest in report.get('sources',{}).items():assert sha(ROOT/file)==digest,file
        result['regressions'][name]=dict(status=report['status'],checks=len(report['checks']),npu_calls=report['final_stats']['npu_calls'])
    scatter=read(OUT/'scatter/probe.json');assert scatter['status']=='passed';healthy(scatter['final_stats'])
    assert scatter['source_sha256']==sha(ROOT/'npunlock-probe/research/kv_scatter_probe.py')
    assert scatter['candidate_sha256']==sha(OUT/'scatter/candidate.py')
    assert len(scatter['checks'])==4 and all(c['compared_arrays']==13 and c['byte_differences']==0 for c in scatter['checks'])
    for timing in scatter['timings']:
        assert timing['npu_calls']==336
        for name,values in timing['samples_ms'].items():assert len(values)==42 and statistics.median(values)==timing['median_ms'][name]
    result['scatter_experiment']=dict(status='passed',promoted=False,bit_equal_arrays=52,npu_calls=scatter['final_stats']['npu_calls'],
        rows=[dict(tag=t['tag'],median_ms=t['median_ms'],latency_reduction_percent=100*(1-t['median_ms']['scatter']/t['median_ms']['select'])) for t in scatter['timings']])
    result.update(model_arrays_per_process=312,generation_arrays_per_process=12,audit_source_sha256=sha(Path(__file__)))
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='builds'},indent=2))


if __name__=='__main__':main()

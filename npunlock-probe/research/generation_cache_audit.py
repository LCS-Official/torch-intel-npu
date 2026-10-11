"""Offline verification of cache traces, paired benchmarks and actual production."""
from collections import Counter
from pathlib import Path
import statistics as st

import numpy as np
from generation_cache_probe import OUT, read, write
from w8_quality_probe import ROOT, MODEL, sha


def released(report,limited=False):
    assert report['status'] in (('passed','passed_with_limitations') if limited else ('passed',))
    stats=report['final_stats'];assert not stats['context_failed']
    assert all(stats[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_blocks','cached_buffer_bytes','quarantined_buffer_bytes'))


def main():
    frozen=read(OUT/'before/sources.json');build=read(OUT/'build.json')
    assert build['sources']==frozen and build['builder_sha256']==sha(OUT/'build-probe.py')
    diag=read(OUT/'diagnostic/build.json');assert diag==build['diagnostic']
    assert sha(OUT/'diagnostic/backend.cpp')==diag['source_sha256']
    assert sha(OUT/'diagnostic/_torch_intel_npu.pyd')==diag['extension_sha256']
    source=OUT/'before/npunlock-probe/torch-device/backend.cpp'
    expected=source.read_text().replace('constexpr size_t max_cached_buffers = 128;','constexpr size_t max_cached_buffers = 256;')
    promoted=(ROOT/'npunlock-probe/torch-device/backend.cpp').read_text()==expected
    assert promoted or sha(ROOT/'npunlock-probe/torch-device/backend.cpp')==frozen['npunlock-probe/torch-device/backend.cpp']
    changes={'torch_intel_npu.py','npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/_torch_intel_npu.pyd','npunlock-probe/torch-device/build.json'}
    for name,digest in frozen.items():
        if name in changes:
            assert sha(OUT/'before'/name)==digest
            if not promoted:assert sha(ROOT/name)==digest
        else:assert sha(ROOT/name)==digest,name
    if promoted:
        expected_py=(OUT/'before/torch_intel_npu.py').read_text().replace('"allocation_cache_limit_blocks": 128','"allocation_cache_limit_blocks": 256')
        assert (ROOT/'torch_intel_npu.py').read_text()==expected_py
        current=read(ROOT/'npunlock-probe/torch-device/build.json')
        for name,key in (('backend.cpp','source_sha256'),('_torch_intel_npu.pyd','extension_sha256')):
            assert sha(ROOT/'npunlock-probe/torch-device'/name)==current[key]
        assert current['session_sha256']==frozen['npunlock-probe/session-runtime/npunlock_session.dll']
        promotion=read(OUT/'promotion.json')
        assert promotion['status']=='passed' and promotion['production_build']==current
        assert set(promotion['before'])==set(promotion['after'])==changes
        assert sha(OUT/'pre-promotion-audit.json')==promotion['pre_promotion_audit_sha256']
        assert sha(OUT/'pre-promotion-audit.py')==promotion['pre_promotion_audit_source_sha256']
        preaudit=read(OUT/'pre-promotion-audit.json')
        assert preaudit['status']=='passed' and not preaudit['promoted']
        assert preaudit['audit_source_sha256']==promotion['pre_promotion_audit_source_sha256']
        for name in changes:
            assert promotion['before'][name]==frozen[name]
            assert sha(OUT/'after'/name)==sha(ROOT/name)==promotion['after'][name]
    assert sha(MODEL/'model.safetensors')=='5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c'
    trace=read(OUT/'trace-fp16.json');released(trace)
    assert trace['source_sha256']==sha(OUT/'probe.py')
    assert sha(OUT/'trace-fp16-outputs.npz')==trace['outputs_sha256']
    report=dict(status='running',promoted=promoted,traces={},benchmarks={},production={},checks={})
    with np.load(OUT/'trace-fp16-outputs.npz') as arrays:
        assert arrays['logits'].shape[0]==32
        for capacity in (192,256):
            for i in range(90):np.testing.assert_array_equal(arrays[f'cache-128/{i}'].view(np.uint8),arrays[f'cache-{capacity}/{i}'].view(np.uint8))
    for sample in trace['trace']:
        capacity=sample['capacity'];expected_count={128:86,192:22,256:0}[capacity]
        assert sample['all_32_logits_and_tokens_bit_equal'] and sample['delta']['driver_allocations']==expected_count
        events=[e for stage in sample['events'] for e in stage['events']]+sample['cleanup_events']
        assert sum(e['creates'] for e in events)==sum(e['evictions'] for e in events)==expected_count
        assert sum(e['count_limit'] for e in events)==expected_count and sum(e['byte_limit'] for e in events)==0
        phases=[dict(step=s['step'],phase=s['phase'],creates=sum(e['creates'] for e in s['events'])) for s in sample['events'] if any(e['creates'] for e in s['events'])]
        if capacity==128:assert phases==[dict(step=0,phase='upload',creates=1),dict(step=0,phase='decoder',creates=2),dict(step=1,phase='decoder',creates=65),dict(step=2,phase='decoder',creates=18)]
        if capacity==256:assert sample['after_cleanup']['cached_buffer_blocks']==214
        report['traces'][str(capacity)]=dict(creates=expected_count,phases=phases,after_cleanup=sample['after_cleanup'])
    for mode in ('fp16','w8a16'):
        benchmark=read(OUT/f'benchmark-{mode}.json');released(benchmark)
        assert benchmark['source_sha256']==sha(OUT/'probe.py') and len(benchmark['samples'])==28 and len(benchmark['pairs'])==7
        assert sha(OUT/f'benchmark-{mode}-outputs.npz')==benchmark['outputs_sha256']
        assert len(benchmark['graphs'])==60 and {(g['layer'],g['operation']) for g in benchmark['graphs']}=={(g['layer'],g['operation']) for g in trace['graphs']}
        for sample in benchmark['samples']:
            assert not sample['events'] and sample['all_32_logits_and_tokens_bit_equal']
            change=sample['delta'];assert change['driver_allocations']==(86 if sample['capacity']==128 else 0)
            assert change['npu_calls']==960 and change['session_creations']==0 and change['chain_graphs']==930 and change['chain_groups']==248
            assert change['host_upload_copies']==change['host_download_copies']==32
        for capacity in (128,256):
            rows=[s for s in benchmark['samples'] if s['capacity']==capacity];assert len(rows)==14
            assert st.median(s['elapsed_ms'] for s in rows)==benchmark['summary'][str(capacity)]['elapsed_ms']
        assert benchmark['positive_rounds']==sum(p['256']<p['128'] for p in benchmark['pairs'])
        report['benchmarks'][mode]=dict(summary=benchmark['summary'],wins=benchmark['positive_rounds'],first_logits_vs_cpu=benchmark['first_logits_vs_cpu'])
        if not promoted:continue
        production=read(OUT/'production'/f'{mode}.json');released(production)
        assert production['all_32_logits_vs_isolated_baseline_bit_equal'] and len(production['samples'])==8
        assert production['wrapper_sha256']==sha(OUT/'validation.py')
        assert production['final_stats']['closed'] and production['final_stats']['allocation_cache_limit_blocks']==256
        assert sha(OUT/'production'/f'{mode}-outputs.npz')==production['outputs_sha256']
        for sample in production['samples']:
            assert sample['all_32_logits_and_tokens_bit_equal'] and sample['delta']['driver_allocations']==0
            assert sample['delta']['session_creations']==0 and sample['delta']['chain_graphs']==930
        reference=OUT/('trace-fp16-outputs.npz' if mode=='fp16' else 'benchmark-w8a16-outputs.npz')
        assert sha(reference)==production['reference_outputs_sha256']
        with np.load(reference) as old,np.load(OUT/'production'/f'{mode}-outputs.npz') as actual:
            np.testing.assert_array_equal(old['tokens'],actual['tokens'])
            for i in range(32):np.testing.assert_array_equal(old['logits'][i].view(np.uint8),actual[f'logits-{i}'].view(np.uint8))
        report['production'][mode]=dict(samples=8,warm_driver_allocations=0,all_32_logits_bit_equal=True)
    if promoted:
        for filename in ('allocator-guards','idle-order','device','training'):
            value=read(OUT/'checks'/f'{filename}.json');released(value,limited=filename=='training')
            report['checks'][filename]=dict(status=value['status'],checks=len(value.get('checks',[])))
        failure=read(OUT/'checks/allocator-failure.json')
        assert failure['synthetic_failure'] and not failure['actual_device_submission'] and not failure['driver_free_calls'] and not failure['driver_reset_calls']
        assert failure['status']=='passed' and failure['final_stats']['context_failed']
        assert all(failure['final_stats'][k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_blocks','cached_buffer_bytes'))
        assert failure['failed']['quarantined_buffer_bytes']>=failure['before']['live_buffer_bytes']+failure['before']['cached_buffer_bytes']
        report['checks']['allocator-failure']=dict(status='passed',synthetic=True,driver_frees=0,driver_resets=0)
    report.update(status='passed',audit_source_sha256=sha(Path(__file__)))
    if promoted:report['promotion_sha256']=sha(OUT/'promotion.json')
    write(OUT/'audit.json',report)
    print({k:report[k] for k in ('status','promoted','traces','benchmarks','production','checks')},flush=True)


if __name__=='__main__':main()

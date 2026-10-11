"""Offline audit of frozen controls and the promoted SHAVE RMSNorm."""
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from npunlock.npu4000 import kernel_text

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-reduce'
FIX=ROOT/'npunlock/tests/fixtures/npu4000'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    build=read(OUT/'build.json');manifest=read(FIX/'rmsnorm/manifest.json')
    assert build['status']=='built'
    assert sha(OUT/'measurement-builder.py')==build['builder_sha256']==sha(ROOT/'npunlock-probe/research/build_rmsnorm_reduce.py')
    for name,digest in read(OUT/'production-before.json').items():assert sha(ROOT/name)==digest,name
    for name,digest in manifest['source_sha256'].items():assert sha(FIX/name)==digest,name
    winner=build['kernels']['codec']
    assert manifest['source_sha256']['rmsnorm.c']==winner['source_sha256']==sha(OUT/'codec.c')
    for file,field in [('rmsnorm-f16.elf','elf_sha256'),('rmsnorm-f16.s','assembly_sha256')]:
        assert sha(FIX/'rmsnorm'/file)==manifest[field]==winner[field]==sha(OUT/'after'/file),file
    assert manifest['text_bytes']==winner['text_bytes']==4592
    for name,digest in build['helper_sha256'].items():assert sha(FIX/name)==digest
    for name,info in build['kernels'].items():
        assert sha(OUT/f'{name}.elf')==info['elf_sha256']
        assert len(kernel_text((OUT/f'{name}.elf').read_bytes()))==info['text_bytes']
    assert sha(OUT/'before/rmsnorm-f16.elf')==build['kernels']['baseline']['elf_sha256']
    old=(OUT/'before/rmsnorm.c').read_text(encoding='utf-8');new=(FIX/'rmsnorm.c').read_text(encoding='utf-8')
    begin='        float total = 0.0f, correction = 0.0f;';end='        for (unsigned j=0;j<576;j+=4) {\n            unsigned i=base+j;'
    assert old[old.index(begin):old.index(end)]==new[new.index(begin):new.index(end)],'ordered FP32 reduction changed'
    begin='void controlled_act(unsigned p) {';end='    for (unsigned base = 0; base < n; base += 576) {'
    assert old[old.index(begin):old.index(end)]==new[new.index(begin):new.index(end)],'MemRef guards changed'
    refs=read(OUT/'references/sources.json')
    for item in refs['files']:assert sha(OUT/'references'/item['file'])==item['sha256']
    license_sha=sha(OUT/'references/FP16_LICENSE')
    assert sha(FIX/'rmsnorm/FP16_LICENSE')==sha(OUT/'after/FP16_LICENSE')==license_sha
    measured=(OUT/'measurement-probe.py').read_text(encoding='utf-8')
    hook="    # Use the frozen old ELF even after the candidate becomes the default.\n    current_kernel=smol._rmsnorm_kernel\n    smol._rmsnorm_kernel=lambda:elf['baseline']\n    try:work()\n    finally:smol._rmsnorm_kernel=current_kernel\n\n\ndef execute(case,rounds,label):"
    current=(ROOT/'npunlock-probe/research/rmsnorm_reduce_probe.py').read_text(encoding='utf-8')
    assert current==measured.replace('    work()\n\n\ndef execute(case,rounds,label):',hook)

    def blob(path,patch,layout):
        data=path.read_bytes();a=patch['code_offset'];b=a+patch['window_bytes']
        assert sha(path)==patch['patched_sha256'],str(path)
        assert patch['device_id']=='0x643e' and patch['compiler_version']==[8,3]
        assert patch['window_bytes']==21216 and patch['dtype']=='f16' and patch['input_count']==2
        name=next(n for n,k in build['kernels'].items() if k['elf_sha256']==patch['kernel_elf_sha256'])
        text=kernel_text((OUT/f'{name}.elf').read_bytes())
        assert data[a:a+len(text)]==text
        outside=data[:a]+bytes(b-a)+data[b:]
        assert hashlib.sha256(outside).hexdigest()==layout
        return outside

    result=dict(status='passed',production_kernel_elf_sha256=manifest['elf_sha256'],license_sha256=license_sha,
                reports={},regressions={},timings=[],unchanged_runtime_files=7)
    for label in ('numeric-codec','model-codec','model-confirm'):
        report=read(OUT/f'{label}.json');assert report['status']=='passed'
        assert report['source_sha256']==sha(OUT/'measurement-probe.py') and report['build']==build
        stats=report['final_stats']
        assert stats['closed'] and not stats['context_failed']
        for field in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','cached_sessions','quarantined_buffer_bytes'):assert stats[field]==0,field
        if report['case']=='numeric':
            assert len(report['checks'])==104 and all(c['bit_differences']==0 for c in report['checks'])
            for rows,records in report['carrier_evidence'].items():
                for patch in records:blob(OUT/f"norm-{rows}-{patch['kernel']}.blob",patch,patch['layout_sha256'])
            arrays=0
        else:
            assert len(report['checks'])==4 and len(report['carrier_evidence'])==32
            arrays=0
            for check in report['checks']:
                tag=check['tag'];path=OUT/f'{tag}-outputs.npz'
                assert check['bit_equal_arrays']==26 and check['seed_preserved'] and sha(path)==check['outputs_sha256']
                with np.load(path) as data:
                    assert len(data.files)==64
                    for i in range(26):np.testing.assert_array_equal(data[f'{tag}/baseline/{i}'].view(np.uint8),data[f'{tag}/codec/{i}'].view(np.uint8))
                    for i in range(12):np.testing.assert_array_equal(data[f'{tag}/baseline/{i+1}'].view(np.uint8),data[f'{tag}/preserved/{i}'].view(np.uint8))
                arrays+=26
            for record in report['carrier_evidence']:
                outside=[blob(OUT/f"{record['tag']}-{name}.blob",record[name],record['layout_sha256']) for name in ('baseline','codec')]
                assert outside[0]==outside[1]
        for timing in report['timings']:
            rounds=report['rounds'];samples=timing['samples_ms'];pairs=timing['pairs_ms']
            assert len(pairs)==rounds
            for name,values in samples.items():
                assert len(values)==rounds*2 and all(v>0 for v in values)
                assert statistics.median(values)==timing['median_ms'][name]
                assert all(statistics.mean(values[2*i:2*i+2])==pairs[i][name] for i in range(rounds))
            assert timing['positive_rounds']['codec']==sum(p['codec']<p['baseline'] for p in pairs)
            assert timing['warm_delta']['npu_calls']==rounds*(4 if report['case']=='numeric' else 16)
            delta=timing['warm_delta']
            assert delta['buffer_allocations']==delta['driver_allocations']+delta['allocation_reuses']
            if report['case']=='numeric':assert delta['driver_allocations']==0
            a,b=[timing['median_ms'][name] for name in ('baseline','codec')]
            result['timings'].append(dict(report=label,tag=timing.get('tag',f"rows-{timing.get('rows')}"),operation=timing.get('operation','rmsnorm'),
                baseline_ms=a,codec_ms=b,latency_reduction_percent=100*(a-b)/a,positive_rounds=timing['positive_rounds']['codec'],rounds=rounds,
                warm_delta=delta))
        result['reports'][label]=dict(checks=len(report['checks']),bit_equal_model_arrays=arrays,npu_calls=stats['npu_calls'])
    for module,cases in (('torch_npu_smollm_demo',('norm','numeric','boundary','guards','causality','hybrid')),
                         ('torch_npu_smollm_stack_demo',('numeric','boundary'))):
        for case in cases:
            report=read(OUT/'regressions'/module/f'{case}.json')
            assert report['status']=='passed' and report['kernel_manifest']==manifest
            for name,digest in report['sources'].items():assert sha(ROOT/name)==digest,name
            stats=report['final_stats'];assert stats['closed'] and not stats['context_failed']
            for field in ('live_buffers','cached_buffer_bytes','quarantined_buffer_bytes'):assert stats[field]==0,field
            for check in report['checks']:
                if 'warm_graph_calls' in check:assert check['warm_graph_calls']==check.get('num_layers',1)
            result['regressions'][f'{module}/{case}']=dict(checks=len(report['checks']),npu_calls=stats['npu_calls'])
    smoke_dir=OUT/'post-promotion';smoke=read(smoke_dir/'model-repro-smoke.json')
    assert smoke['status']=='passed' and smoke['build']==build and smoke['rounds']==3
    assert smoke['source_sha256']==sha(ROOT/'npunlock-probe/research/rmsnorm_reduce_probe.py')
    assert len(smoke['checks'])==4 and len(smoke['carrier_evidence'])==32
    for check in smoke['checks']:
        path=smoke_dir/f"{check['tag']}-outputs.npz"
        assert check['bit_equal_arrays']==26 and check['seed_preserved']
        assert sha(path)==check['outputs_sha256']==sha(OUT/path.name)
    for record in smoke['carrier_evidence']:
        outside=[blob(smoke_dir/f"{record['tag']}-{name}.blob",record[name],record['layout_sha256']) for name in ('baseline','codec')]
        assert outside[0]==outside[1]
    stats=smoke['final_stats']
    assert stats['closed'] and not stats['context_failed']
    for field in ('live_buffers','cached_buffer_bytes','quarantined_buffer_bytes'):assert stats[field]==0,field
    result['post_promotion_repro']=dict(checks=4,bit_equal_model_arrays=104,npu_calls=stats['npu_calls'],rounds=3)
    result['audit_source_sha256']=sha(Path(__file__))
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='timings'},indent=2))


if __name__=='__main__':main()

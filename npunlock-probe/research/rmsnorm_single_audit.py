"""Verify single-row ablation, exact encoding and complete model controls."""
import json
from pathlib import Path
import statistics as st

import numpy as np
from rmsnorm_single_row import OUT,FIXTURES
from w8_quality_probe import ROOT,sha
from decoder_native_audit import released


def read(path):return json.loads(path.read_text())


def main():
    candidate=OUT/'half-slow';ablation=read(OUT/'ablation.json');released(ablation)
    assert ablation['status']=='passed' and len(ablation['checks'])==60
    report=dict(status='running',ablation={},models={},generation={},source_sha256=sha(Path(__file__)))
    build=read(OUT/'build.json')
    assert sha(OUT/'builder.py')==build['builder_sha256']
    for directory in (OUT,candidate):
        manifest=read(directory/'build.json');assert manifest['status']=='built'
        assert sha(directory/'builder.py')==manifest['builder_sha256']
        for kernel,fields in manifest['kernels'].items():
            for suffix,field in (('.c','source_sha256'),('.s','assembly_sha256'),('.elf','elf_sha256')):
                assert sha(directory/(kernel+suffix))==fields[field]
    assert sha(candidate/'baseline.c')==build['sources_sha256']['rmsnorm.c']
    assert sha(candidate/'baseline.elf')==build['kernels']['baseline']['elf_sha256']
    assert sha(OUT/'outputs.npz')==ablation['outputs_sha256']
    with np.load(OUT/'outputs.npz') as arrays:
        for row in ablation['checks']:
            left=arrays[f"{row['dataset']}/plain/{row['kernel']}"]
            right=arrays[f"{row['dataset']}/profile/{row['kernel']}"]
            np.testing.assert_array_equal(left.view(np.uint8),right.view(np.uint8))
    for row in ablation['profiles']:
        dataset,kernel=row['dataset'],row['kernel'];samples=row['samples']
        assert len({s['raw_sha256'] for s in samples})==3
        for i,sample in enumerate(samples):assert sha(OUT/'tasks'/f'{dataset}-{kernel}-{i}-tasks.bin')==sample['raw_sha256']
        assert row['median_shave_union_ms']==st.median(s['engines']['shave']['union_ms'] for s in samples)
        report['ablation'].setdefault(dataset,{})[kernel]=row['median_shave_union_ms']*1000
    for row in ablation['carriers']:assert all(c['elements']==576 for c in row['patch']['contracts'])
    encoding=read(candidate/'encodings.json');released(encoding)
    assert encoding['status']=='passed' and sha(candidate/'encoding-outputs.npz')==encoding['outputs_sha256']
    with np.load(candidate/'encoding-outputs.npz') as values:
        assert len(values['input'])==319494
        for name in ('encode-baseline','encode-codec'):
            a,b=values[name],values['reference']
            assert ((a.view(np.uint16)==b.view(np.uint16))|(np.isnan(a)&np.isnan(b))).all()
        np.testing.assert_array_equal(values['encode-baseline'].view(np.uint16),values['encode-codec'].view(np.uint16))
    report['encoding_values_per_kernel']=319494
    numeric=read(candidate/'numeric-half-slow.json');released(numeric)
    assert numeric['status']=='passed' and len(numeric['checks'])==158
    assert all(c['bit_differences']==0 for c in numeric['checks'])
    for name in ('full30','model-b4-s24','model-b8-s63'):
        model=read(candidate/(name+'.json'));released(model)
        assert model['status']=='passed' and model['checks'][0]['bit_equal_arrays']==182 and model['immutable_seed_arrays']==90
        assert sha(candidate/(name+'-outputs.npz'))==model['outputs_sha256']
        assert len(model['graphs'])==60
        with np.load(candidate/(name+'-outputs.npz')) as arrays:
            for i in range(182):np.testing.assert_array_equal(arrays[f'baseline/{i}'].view(np.uint8),arrays[f'codec/{i}'].view(np.uint8))
            for i in range(90):np.testing.assert_array_equal(arrays[f'baseline/{i+1}'].view(np.uint8),arrays[f'preserved/{i}'].view(np.uint8))
        timings={}
        for row in model['timings']:
            assert len(row['pairs_ms'])==31 and row['warm_delta']['driver_allocations']==0
            for kernel in ('baseline','codec'):
                assert len(row['samples_ms'][kernel])==62 and st.median(row['samples_ms'][kernel])==row['median_ms'][kernel]
            timings[row['operation']]=dict(median_ms=row['median_ms'],wins=row['positive_rounds']['codec'])
        report['models'][name]=timings
    for mode in ('fp16','w8a16'):
        generation=read(candidate/'generation'/(mode+'.json'));released(generation)
        assert generation['status']=='passed' and generation['comparison']=='RMSNorm baseline vs SIMD slow rounding, both group4'
        assert generation['decoder_layers']==30 and generation['cpu_decoder_layers']==0
        assert len(generation['graphs'])==60 and len(generation['samples'])==28
        assert sha(candidate/'generation'/(mode+'-outputs.npz'))==generation['outputs_sha256']
        for sample in generation['samples']:
            assert sample['all_32_logits_and_tokens_bit_equal'] and sample['delta']['chain_graphs']==930
            assert sample['delta']['chain_groups']==248 and sample['delta']['session_creations']==0
        for name,digest in generation['sources'].items():
            if name.endswith('/rmsnorm.c') or name.endswith('/rmsnorm-f16.elf'):continue
            assert sha(ROOT/name)==digest,name
        for label in ('1','4'):
            rows=[s for s in generation['samples'] if ('1' if s['group']==1 else '4')==label]
            assert st.median(s['elapsed_ms'] for s in rows)==generation['summary'][label]['elapsed_ms']
        report['generation'][mode]=dict(summary=generation['summary'],wins=generation['positive_rounds'],
            first_logits_vs_cpu=generation['first_logits_vs_cpu'])
    promotion=candidate/'promotion.json'
    if promotion.exists():
        data=read(promotion);assert data['status']=='passed'
        assert sha(candidate/'pre-promotion-audit.json')==data['audit_sha256']
        for version in ('before','after'):
            for name,digest in data[version].items():
                assert sha(candidate/version/name)==digest
                if version=='after':assert sha(ROOT/name)==digest
        report['promoted']=True
        manifest=read(FIXTURES/'rmsnorm/manifest.json')
        assert manifest['elf_sha256']==read(candidate/'build.json')['kernels']['codec']['elf_sha256']
        assert sha(FIXTURES/'rmsnorm/rmsnorm-f16.elf')==manifest['elf_sha256']
        for name,digest in manifest['source_sha256'].items():assert sha(FIXTURES/name)==digest
        report['regressions']=[]
        for module,cases in (('torch_npu_smollm_demo',('norm','numeric','boundary','guards','causality','hybrid')),
                             ('torch_npu_smollm_stack_demo',('numeric','boundary'))):
            for case in cases:
                value=read(candidate/'regressions'/module/(case+'.json'));released(value)
                assert value['status']=='passed' and value['kernel_manifest']['elf_sha256']==manifest['elf_sha256']
                report['regressions'].append(dict(module=module,case=case,checks=len(value['checks'])))
    else:
        assert sha(FIXTURES/'rmsnorm.c')==sha(candidate/'baseline.c')
        assert sha(FIXTURES/'rmsnorm/rmsnorm-f16.elf')==sha(candidate/'baseline.elf')
        report['promoted']=False
    report['files']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('audit.json','REPORT.md')}
    report['status']='passed';(OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','promoted','models','generation')},indent=2),flush=True)


if __name__=='__main__':main()

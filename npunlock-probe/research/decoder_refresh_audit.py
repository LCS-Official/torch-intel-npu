"""Offline verification of refreshed hotspots and the shared KV flag candidate."""
import ast
from collections import Counter
from pathlib import Path
import statistics as st

import numpy as np
from decode_mask_audit import graph_node, released
from decoder_refresh import OUT, read, write
from w8_quality_probe import ROOT, MODEL, sha


def profile(directory,mode,layers):
    path=directory/f'b1-s24-{mode}.json';value=read(path);released(value)
    assert len(value['graphs'])==layers*2 and sha(path.with_name(path.stem+'-outputs.npz'))==value['outputs_sha256']
    assert value['source_sha256']==sha(ROOT/'torch_npu_device_profile.py')
    assert sha(directory/'session_task_profile.dll')==value['profiling_dll_sha256']
    counts={}
    for graph in value['graphs']:
        assert len(graph['samples'])==len({s['raw_sha256'] for s in graph['samples']})==3
        for i,sample in enumerate(graph['samples']):
            assert sha(directory/f"{graph['tag']}-{i}-tasks.bin")==sample['raw_sha256']
            if graph['operation']=='decode':
                count=dict(Counter(t['layer_type'] for t in sample['tasks'] if t['exec_type']==2))
                assert counts.setdefault('shave',count)==count
                assert counts.setdefault('dpu_tasks',sample['engines']['dpu']['tasks'])==sample['engines']['dpu']['tasks']
    with np.load(path.with_name(path.stem+'-outputs.npz')) as values:
        for i in range(2+6*layers):np.testing.assert_array_equal(values[f'plain/{i}'].view(np.uint8),values[f'profile/{i}'].view(np.uint8))
        for i in range(2,2+3*layers):np.testing.assert_array_equal(values[f'plain/{i}'].view(np.uint8),values[f'seed/{i}'].view(np.uint8))
    return counts


def main():
    frozen=read(OUT/'before.json');directory=OUT/'position-flags'
    current=ast.parse((ROOT/'torch_npu_smollm.py').read_text())
    before=ast.parse((OUT/'before/torch_npu_smollm.py').read_text())
    actual,original=graph_node(current),graph_node(before);assert ast.dump(current)==ast.dump(before)
    baseline=ast.parse((directory/'baseline-graph.py').read_text()).body[0]
    candidate=ast.parse((directory/'flags-graph.py').read_text()).body[0]
    assert ast.dump(original)==ast.dump(baseline)
    promoted=ast.dump(actual)==ast.dump(candidate);assert promoted or ast.dump(actual)==ast.dump(baseline)
    for name,digest in frozen['sources'].items():
        assert sha(OUT/'before'/name)==digest,name
        if name!='torch_npu_smollm.py':assert sha(ROOT/name)==digest,name
    assert sha(OUT/'profile-runner.py')==frozen['runner_sha256']
    build=read(OUT/'profile/build.json');assert build==frozen['profiling_build']
    assert sha(OUT/'profile/session_task_profile.c')==build['generated_source_sha256']
    assert sha(OUT/'profile/session_task_profile.dll')==build['dll_sha256']
    assert sha(MODEL/'model.safetensors')=='5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c'
    host=read(OUT/'host.json');released(host)
    assert sha(OUT/'host-outputs.npz')==host['outputs_sha256']
    for row in host['rows']:
        assert len(row['host_samples'])==62 and row['warm_delta']['driver_allocations']==0
        assert st.median(row['samples_ms']['baseline'])==row['median_ms']['baseline']
    report=dict(status='running',promoted=promoted,profiles={},models={},generation={})
    for mode in ('fp16','w8a16'):report['profiles'][mode]=profile(OUT/'profile',mode,30)
    with np.load(OUT/'host-outputs.npz') as h,np.load(OUT/'profile/b1-s24-fp16-outputs.npz') as d:
        for i in range(182):
            plain=0 if i==0 else 1 if i==91 else i+1 if i<=90 else i
            np.testing.assert_array_equal(h[f'b1-s24/{i}'].view(np.uint8),d[f'plain/{plain}'].view(np.uint8))
    tags=('l4-b1-s24-fp16','l30-b1-s24-fp16','l30-b1-s24-w8a16',
          'l4-b4-s63-fp16','l4-b8-s24-fp16','l4-b8-s24-w8a16')
    for tag in tags:
        data=read(directory/(tag+'.json'));released(data);layers=data['layers'];count=1+3*layers
        assert data['wrapper_sha256']==sha(ROOT/'npunlock-probe/research/decode_mask_table.py')
        assert data['candidate_wrapper_sha256']==sha(ROOT/'npunlock-probe/research/decode_position_flags.py')
        for name in ('baseline','flags'):assert data['graph_source_sha256'][name]==sha(directory/(name+'-graph.py'))
        path=directory/(tag+'-outputs.npz');assert sha(path)==data['outputs_sha256']
        with np.load(path) as values:
            for i in range(count):np.testing.assert_array_equal(values[f'baseline/{i}'].view(np.uint8),values[f'flags/{i}'].view(np.uint8))
            for i in range(layers):
                for j in (0,1):
                    old,new=values[f'seed/{3*i+j}'],values[f'baseline/{1+3*i+j}']
                    for part in (slice(0,data['prefix']),slice(data['prefix']+1,64)):
                        np.testing.assert_array_equal(old[:,:,part,:].view(np.uint8),new[:,:,part,:].view(np.uint8))
                assert values[f'seed/{3*i+2}'][0]==data['prefix']
                assert values[f'baseline/{3*i+3}'][0]==data['prefix']+1
        assert data['immutable_seed_arrays']==3*layers
        if layers==4:
            for name in ('baseline','flags'):assert {c['position'] for c in data['checks'] if c['name']==name}==set(range(1,64))
        timing=data['timing'];assert len(timing['pairs_ms'])==31 and timing['warm_delta']['driver_allocations']==0
        for name in ('baseline','flags'):
            assert len(timing['samples_ms'][name])==62
            assert st.median(timing['samples_ms'][name])==timing['median_ms'][name]
        for graph in data['graphs']:assert graph['ir_nodes'].get('Equal',0)==(1 if graph['name']=='baseline' else 0)
        report['models'][tag]=dict(median_ms=timing['median_ms'],wins=timing['positive_rounds'])
    for name in ('baseline','flags'):
        report['profiles'][name]=profile(directory/'profile'/name,'fp16',4)
        value=read(directory/'profile'/name/'b1-s24-fp16.json')
        assert value['wrapper_sha256']==sha(ROOT/'npunlock-probe/research/decode_position_profile.py')
        assert value['graph_source_sha256']==sha(directory/(name+'-graph.py'))
    with np.load(directory/'profile/baseline/b1-s24-fp16-outputs.npz') as a,np.load(directory/'profile/flags/b1-s24-fp16-outputs.npz') as b:
        assert a.files==b.files
        for key in a.files:np.testing.assert_array_equal(a[key].view(np.uint8),b[key].view(np.uint8))
    for mode in ('fp16','w8a16'):
        path=directory/'generation'/(mode+'.json');value=read(path);released(value)
        assert len(value['samples'])==28 and len(value['pairs'])==7 and len(value['graphs'])==30
        assert sha(path.with_name(mode+'-outputs.npz'))==value['outputs_sha256']
        assert value['factory_sha256']==sha(ROOT/'npunlock-probe/research/decode_position_generation.py')
        assert value['wrapper_sha256']==sha(directory/'generation/graph-wrapper.py')
        assert value['graph_source_sha256']==sha(directory/'flags-graph.py')
        assert value['baseline_graph_source_sha256']==sha(directory/'baseline-graph.py')
        assert value['graph_wrapper_template_sha256']==sha(ROOT/'npunlock-probe/research/decode_mask_generation.py')
        for sample in value['samples']:
            assert sample['all_32_logits_and_tokens_bit_equal']
            change=sample['delta'];assert change['driver_allocations']==change['session_creations']==0
            assert change['npu_calls']==960 and change['chain_graphs']==930 and change['chain_groups']==248
            assert change['host_upload_copies']==change['host_download_copies']==32
        for label in (1,4):
            rows=[s for s in value['samples'] if (1 if s['group']==1 else 4)==label];assert len(rows)==14
            assert st.median(s['elapsed_ms'] for s in rows)==value['summary'][str(label)]['elapsed_ms']
        assert value['positive_rounds']==sum(p['4']<p['1'] for p in value['pairs'])
        with np.load(path.with_name(mode+'-outputs.npz')) as values:
            for i in range(32):assert np.isfinite(values[f'logits-{i}']).all()
        report['generation'][mode]=dict(summary=value['summary'],wins=value['positive_rounds'],quality=value['first_logits_vs_cpu'])
    if promoted:
        promotion=read(OUT/'promotion.json');assert promotion['status']=='passed'
        assert sha(ROOT/'torch_npu_smollm.py')==sha(OUT/'torch_npu_smollm-after.py')==promotion['after_sha256']
        assert sha(OUT/'before/torch_npu_smollm.py')==promotion['before_sha256']
        assert sha(OUT/'pre-promotion-audit.json')==promotion['pre_promotion_audit_sha256']
        assert read(OUT/'pre-promotion-audit.json')['audit_source_sha256']==sha(OUT/'pre-promotion-audit.py')
        report['production']={}
        for mode in ('fp16','w8a16'):
            path=OUT/'production'/(mode+'.json');value=read(path);released(value)
            assert value['sources']['torch_npu_smollm.py']==promotion['after_sha256']
            assert value['wrapper_sha256']==sha(OUT/'production-validator.py')
            assert value['factory_sha256']==sha(ROOT/'npunlock-probe/research/decoder_refresh_validate.py')
            assert value['validator_template_sha256']==sha(ROOT/'npunlock-probe/research/generation_cache_validate.py')
            assert len(value['samples'])==8 and value['all_32_logits_vs_isolated_baseline_bit_equal']
            assert all(s['delta']['driver_allocations']==0 and s['all_32_logits_and_tokens_bit_equal'] for s in value['samples'])
            reference=directory/'generation'/(mode+'-outputs.npz')
            assert sha(reference)==value['reference_outputs_sha256']
            assert sha(path.with_name(mode+'-outputs.npz'))==value['outputs_sha256']
            with np.load(reference) as a,np.load(path.with_name(mode+'-outputs.npz')) as b:
                np.testing.assert_array_equal(a['tokens'],b['tokens'])
                for i in range(32):np.testing.assert_array_equal(a[f'logits-{i}'].view(np.uint8),b[f'logits-{i}'].view(np.uint8))
            report['production'][mode]=dict(samples=8,driver_allocations=0,all_32_logits_bit_equal=True)
        checks=[]
        for module,cases in (('torch_npu_smollm_demo',('norm','numeric','boundary','guards','causality','hybrid')),
                             ('torch_npu_smollm_stack_demo',('numeric','boundary'))):
            for case in cases:
                value=read(OUT/'regressions'/module/(case+'.json'));released(value)
                checks.append(dict(module=module,case=case,count=len(value['checks'])))
        assert sum(c['count'] for c in checks)==74;report['regressions']=checks
    for path in (OUT/'runners').glob('*.py'):assert sha(path)==sha(ROOT/'npunlock-probe/research'/path.name)
    report.update(status='passed',audit_source_sha256=sha(Path(__file__)))
    write(OUT/'audit.json',report)
    print(dict(status='passed',promoted=promoted,models=report['models'],profiles=report['profiles'],
        generation={m:dict(wins=v['wins'],tokens_per_second={k:s['overall_tokens_per_second'] for k,s in v['summary'].items()}) for m,v in report['generation'].items()}),flush=True)


if __name__=='__main__':main()

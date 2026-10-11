"""Offline audit of graph-table candidates, task records and full generation."""
import ast
from collections import Counter
import json
from pathlib import Path
import statistics as st

import numpy as np
from decode_mask_table import OUT
from w8_quality_probe import ROOT, MODEL, sha


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def released(report):
    assert report['status']=='passed'
    stats=report['final_stats'];assert stats['closed'] and not stats['context_failed']
    assert all(stats[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes',
        'quarantined_buffer_bytes','cached_sessions'))


def graph_node(tree):
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FrozenSmolLayer')
    graph=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='graph')
    cls.body.remove(graph);return graph


def main():
    current=ast.parse((ROOT/'torch_npu_smollm.py').read_text())
    original=ast.parse((OUT/'torch_npu_smollm-before.py').read_text())
    actual,before=graph_node(current),graph_node(original)
    assert ast.dump(current)==ast.dump(original),'Changes outside graph()'
    baseline=ast.parse((OUT/'joint/baseline-graph.py').read_text()).body[0]
    joined=ast.parse((OUT/'joint/joined-graph.py').read_text()).body[0]
    assert ast.dump(before)==ast.dump(baseline)
    promoted=ast.dump(actual)==ast.dump(joined)
    assert promoted or ast.dump(actual)==ast.dump(baseline)
    report=dict(status='running',promoted=promoted,models={},profiles={},generation={})
    frozen=read(OUT/'l4-b1-s24-fp16.json')['sources']
    for name,digest in frozen.items():
        if name=='torch_npu_smollm.py':assert sha(OUT/'torch_npu_smollm-before.py')==digest
        else:assert sha(ROOT/name)==digest,name
    # Model file is unchanged from the preceding RMSNorm experiments.
    assert sha(MODEL/'model.safetensors')=='5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c'
    for directory,tag in ((OUT,'l4-b1-s24-fp16'),(OUT,'l30-b1-s24-fp16'),
        (OUT/'joint','l4-b1-s24-fp16'),(OUT/'joint','l4-b8-s24-fp16'),
        (OUT/'joint','l4-b4-s63-fp16'),(OUT/'joint','l30-b1-s24-fp16')):
        model=read(directory/(tag+'.json'));released(model);layers=model['layers'];count=1+3*layers
        assert model['wrapper_sha256']==sha(OUT/'runner.py')
        if directory.name=='joint':
            assert model['joint_wrapper_sha256'] in {sha(directory/'wrapper-v1.py'),sha(directory/'wrapper-v2.py')}
            assert model['joint_graph_sha256']==sha(directory/'joined-graph.py')
        path=directory/(tag+'-outputs.npz');assert sha(path)==model['outputs_sha256']
        with np.load(path) as arrays:
            for name in model['names'][1:]:
                for i in range(count):np.testing.assert_array_equal(arrays[f'{name}/{i}'].view(np.uint8),arrays[f'baseline/{i}'].view(np.uint8))
            for i in range(layers):
                for j in (0,1):
                    old,new=arrays[f'seed/{3*i+j}'],arrays[f'baseline/{1+3*i+j}']
                    for part in (slice(0,model['prefix']),slice(model['prefix']+1,64)):
                        np.testing.assert_array_equal(old[:,:,part,:].view(np.uint8),new[:,:,part,:].view(np.uint8))
                assert arrays[f'seed/{3*i+2}'][0]==model['prefix']
                assert arrays[f'baseline/{1+3*i+2}'][0]==model['prefix']+1
        if len(model['checks'])>len(model['names']):
            for name in model['names']:
                assert set(c['position'] for c in model['checks'] if c['name']==name)==set(range(1,64))
        assert all(c['bit_equal_arrays']==count for c in model['checks'])
        assert model['immutable_seed_arrays']==layers*3
        timing=model['timing'];assert len(timing['pairs_ms'])==31
        assert timing['warm_delta']['driver_allocations']==0
        for name in model['names']:
            assert len(timing['samples_ms'][name])==62
            assert timing['median_ms'][name]==st.median(timing['samples_ms'][name])
        report['models'][str((directory/tag).relative_to(OUT))]=dict(median_ms=timing['median_ms'],wins=timing['positive_rounds'])
    for name in ('baseline','joined'):
        directory=OUT/'profile'/name;profile=read(directory/'b1-s24-fp16.json');released(profile)
        assert sha(directory/'runner.py')==profile['wrapper_sha256']
        assert sha(directory/'session_task_profile.dll')==profile['profiling_dll_sha256']==read(directory/'build.json')['dll_sha256']
        assert sha(directory/'b1-s24-fp16-outputs.npz')==profile['outputs_sha256']
        task_counts=[]
        for graph in profile['graphs']:
            assert len(graph['samples'])==len({s['raw_sha256'] for s in graph['samples']})==3
            for i,sample in enumerate(graph['samples']):
                assert sha(directory/f"{graph['tag']}-{i}-tasks.bin")==sample['raw_sha256']
                if graph['operation']=='decode':
                    counts=Counter(t['layer_type'] for t in sample['tasks'] if t['exec_type']==2)
                    assert sum(counts.values())==(106 if name=='baseline' else 86)
                    assert sample['engines']['dpu']['tasks']==96
                    task_counts.append(dict(counts))
        assert len(task_counts)==12 and all(c==task_counts[0] for c in task_counts)
        report['profiles'][name]=task_counts[0]
    with np.load(OUT/'profile/baseline/b1-s24-fp16-outputs.npz') as baseline_arrays, np.load(OUT/'profile/joined/b1-s24-fp16-outputs.npz') as joined_arrays:
        assert baseline_arrays.files==joined_arrays.files
        for key in baseline_arrays.files:
            np.testing.assert_array_equal(baseline_arrays[key].view(np.uint8),joined_arrays[key].view(np.uint8))
    for mode in ('fp16','w8a16'):
        directory=OUT/'generation';generation=read(directory/(mode+'.json'));released(generation)
        assert generation['wrapper_sha256']==sha(directory/'wrapper.py')
        assert generation['graph_source_sha256']==sha(OUT/'joint/joined-graph.py')
        assert generation['baseline_graph_source_sha256']==sha(OUT/'joint/baseline-graph.py')
        assert len(generation['graphs'])==30 and len(generation['samples'])==28
        assert generation['decoder_layers']==30 and generation['cpu_decoder_layers']==0 and generation['new_tokens']==32
        assert sha(directory/(mode+'-outputs.npz'))==generation['outputs_sha256']
        allocations=generation['driver_allocations_per_generation'];assert allocations['1']==allocations['4']==[86]*14
        for sample in generation['samples']:
            assert sample['all_32_logits_and_tokens_bit_equal']
            change=sample['delta'];assert change['npu_calls']==960 and change['session_creations']==0
            assert change['chain_graphs']==930 and change['chain_groups']==248
        report['generation'][mode]=dict(summary=generation['summary'],wins=generation['positive_rounds'],
            driver_allocations_per_generation=86,first_logits_vs_cpu=generation['first_logits_vs_cpu'])
    if promoted:
        promotion=read(OUT/'promotion.json');assert promotion['status']=='passed'
        assert sha(ROOT/'torch_npu_smollm.py')==sha(OUT/'torch_npu_smollm-after.py')==promotion['after_sha256']
        assert sha(OUT/'torch_npu_smollm-before.py')==promotion['before_sha256']
        assert sha(OUT/'pre-promotion-audit.json')==promotion['pre_promotion_audit_sha256']
        assert read(OUT/'pre-promotion-audit.json')['source_sha256']==sha(OUT/'pre-promotion-audit.py')
        report['regressions']=[]
        for module,cases in (('torch_npu_smollm_demo',('norm','numeric','boundary','guards','causality','hybrid')),
                             ('torch_npu_smollm_stack_demo',('numeric','boundary'))):
            for case in cases:
                value=read(OUT/'regressions'/module/(case+'.json'));released(value)
                report['regressions'].append(dict(module=module,case=case,checks=len(value['checks'])))
        assert sum(r['checks'] for r in report['regressions'])==74
    report.update(status='passed',source_sha256=sha(Path(__file__)),current_graph_sha256=sha(ROOT/'torch_npu_smollm.py'))
    (OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],promoted=promoted,models=report['models'],
        generation={m:dict(wins=g['wins'],tokens_per_second={k:v['overall_tokens_per_second'] for k,v in g['summary'].items()})
                    for m,g in report['generation'].items()}),indent=2),flush=True)


if __name__=='__main__':main()

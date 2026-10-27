"""Audit frozen selection, scalar quality, NLL targets and compiled NPU evidence."""
import ast
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import w8_lowrank_quality as study


def main():
    source=study.sha(Path(study.__file__))
    tree=ast.parse(Path(study.__file__).read_text())
    maths={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name not in ('npu_run','main')}
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='npu_run')
    maths['npu_run.work']=ast.dump(next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=='work'),include_attributes=False)
    assert hashlib.sha256(json.dumps(maths,sort_keys=True).encode()).hexdigest()=='7ed067ff2de58826f267a784161552b62b91241d166626dc43004cf155f9e46c'
    prepare,factors,selection,independent,cpu=[study.read(n) for n in ('prepare','factors','selection','independent','cpu-language')]
    assert prepare['sources']==study.mixed.sources()
    assert study.sha(study.OUT/'weights.npz')==prepare['weights_sha256']==factors['weights_sha256']
    assert study.sha(study.OUT/'corrections.npz')==factors['factor_sha256']
    assert study.sha(study.OUT/'factors.json')==selection['factors_sha256']
    assert selection['ranks']==list(study.RANKS) and selection['selected_rank']==32
    assert independent['selection_sha256']==study.sha(study.OUT/'selection.json')
    assert len(prepare['trajectories'])==40 and len(prepare['matrices'])==210
    assert all(r['calibrated_sse']<=r['rtn_sse']+1e-15 for r in prepare['matrices'])
    assert sum(len(r['token_ids'][0]) for r in prepare['trajectories'] if r['index']%5!=4)==2048
    assert {tuple((r['system'],r['prompt'])) for r in prepare['trajectories']}.isdisjoint(
        (s,p) for _,s,p in study.mixed.HOLDOUT+study.mixed.FRESH)
    packed=study.mixed.weights();corrections=study.corrections(selection['selected_rank'])
    payload=sum(len(q.data)+len(q.scales) for layer in packed.values() for q in layer.values())
    correction_bytes=sum(len(q.down)+len(q.up) for layer in corrections.values() for q in layer.values())
    fp16_bytes=sum(np.prod(q.shape)*2 for layer in packed.values() for q in layer.values())
    scalar_checks=0
    def check(record,count):
        nonlocal scalar_checks
        rows=record['positions'];summary=study.mixed.common.summary(rows)
        assert len(rows)==count
        for key,value in summary.items():
            assert math.isclose(value,record['summary'][key],abs_tol=1e-14,rel_tol=1e-12),(key,value)
        assert all(r['quality_gate_passed']==(r['relative_l2']<.03 and r['next_token_kl']<.02) for r in rows)
        scalar_checks+=len(rows)
    for record in selection['profiles'].values():check(record,290)
    best=min(selection['profiles'],key=lambda r:(*study.mixed.common.ranking(selection['profiles'][r]['summary']),selection['profiles'][r]['factor_bytes']))
    assert int(best)==selection['selected_rank']
    for record in independent['variants'].values():check(record,142)
    data=study.read('language-data');language=study.language_helpers()
    assert cpu['language_data_sha256']==study.sha(study.OUT/'language-data.json')
    assert data['selection_sha256']==independent['selection_sha256'] and data['corpus_sha256']==study.sha(study.OUT/'corpus.json')
    def check_language(record):
        expected=[(case['offset'],w['start'],64-w['first']) for case in data['cases'] for w in language.windows(case['tokens'])]
        actual=[(w['offset'],w['start'],w['count']) for w in record['windows']]
        assert actual==expected and sum(r[2] for r in actual)==2048
        assert language.aggregate(record['windows'])==record['summary']
        assert len(record['generation'])==8
        for row,prompt in zip(record['generation'],data['generation']):
            assert row['tag']==prompt['tag'] and row['consumed']==len(prompt['tokens'])+len(row['token_ids'])-1<=64
            assert 1<=len(row['token_ids'])<=32
            assert (row['token_ids'][-1] in data['eos_ids'])==(row['stop_reason']=='eos')
    for record in cpu['variants'].values():check_language(record)
    npu={};calls=0
    for label in ('corrected','w8','fp16','probe'):
        d=study.read('npu-'+label);final=d['final_stats'];calls+=final['npu_calls']
        # Historical runs precede failure-capture/cache-budget CLI only; the
        # preparation, selection, evaluation and nested work ASTs are unchanged.
        assert d['status']=='passed' and d['source_sha256'] in (source,'85e20ad509c8afbee792ffe2a099a066cb24c2fc799d75790ad819943b7e45bb')
        assert d['adapter_sha256']==study.sha(study.mixed.common.ROOT/'torch_npu_smollm.py')
        assert d['selection_sha256']==study.sha(study.OUT/'selection.json')
        assert d['weights_sha256']==prepare['weights_sha256'] and d['factors_sha256']==study.sha(study.OUT/'factors.json')
        assert final['closed'] and not final['context_failed']
        assert all(final[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','cached_sessions','quarantined_buffer_bytes'))
        if label=='probe':
            assert len(d['guards'])==3 and len(d['graphs'])==3 and final['npu_calls']==9
            continue
        check(d,142);check_language(d['language'])
        assert len(d['graphs'])==90 and len(d['language']['local_checks'])==30
        assert all(r['relative_l2']<.003 and r['max_scaled_error']<.01 for r in d['language']['local_checks'])
        assert len(d['warm'])==17 and len(d['language']['warm'])==8
        expected_calls=30*(592+17+56+sum(r['consumed'] for r in d['language']['generation'])+8)+120
        assert final['npu_calls']==expected_calls
        decode=[]
        for graph in d['graphs']:
            evidence=graph['evidence'];integer=[m for m in evidence['modes'] if not m['elementwise'] and m['wmode'] in (1,2)]
            assert bool(integer)==(label!='fp16') and all(m['amode']==0 and m['wmode']==2 for m in integer)
            if graph['key'][-1]=='decode':decode.append(evidence)
        assert len(decode)==30
        npu[label]=dict(summary=d['summary'],language=d['language']['summary'],
            constant_bytes=sum(e['constant_bytes'] for e in decode),int8_tasks=sum(e['int8_matrix_tasks'] for e in decode),
            max_local_l2=max(r['relative_l2'] for r in d['language']['local_checks']),
            max_local_scaled=max(r['max_scaled_error'] for r in d['language']['local_checks']),
            generation_equal_cpu_fp16=sum(a['token_ids']==b['token_ids'] for a,b in zip(d['language']['generation'],cpu['variants']['fp16']['generation'])),
            generation_equal_cpu_same=sum(a['token_ids']==b['token_ids'] for a,b in zip(d['language']['generation'],cpu['variants'][label]['generation'])))
    reference=study.read('npu-fp16')['language']['generation']
    for label,row in npu.items():
        row['generation_equal_npu_fp16']=sum(a['token_ids']==b['token_ids'] for a,b in zip(study.read('npu-'+label)['language']['generation'],reference))
    assert np.isfinite([r['summary']['mean_relative_l2'] for r in npu.values()]).all()
    all_gates=all(r['quality_gate_passed'] for r in selection['profiles'][best]['positions']+
        independent['variants']['corrected']['positions']+study.read('npu-corrected')['positions'])
    result=dict(status='passed',records=4,npu_calls=calls,scalar_position_checks=scalar_checks,
        packed_weight_bytes=int(payload),correction_bytes=int(correction_bytes),original_fp16_projection_bytes=int(fp16_bytes),
        npu=npu,selected_rank=selection['selected_rank'],candidate_all_quality_gates_passed=all_gates,
        source_sha256=study.sha(Path(__file__)))
    study.write('audit',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Check selection, diagnostic accounting and real-NPU evidence before cleanup."""
import json
import math

import w8_layer_sensitivity as experiment


def ids_consistent(row):
    # topk may pick a different representative than argmax for tied maxima.
    return row['reference_top2_margin']==0 or row['next_token_equal']==(row['reference_top1_id']==row['actual_top1_id'])


def main():
    tied=experiment.diagnostics(experiment.torch.tensor([[1.,2.,2.,0.]]),experiment.torch.tensor([[1.,2.,2.,0.]]))
    assert tied['next_token_equal'] and tied['reference_top2_margin']==0 and ids_consistent(tied)
    assert not ids_consistent(dict(tied,reference_top2_margin=1,reference_top1_id=-1))
    prepare,selection,data=(experiment.read(name) for name in ('prepare','selection','independent'))
    provenance=experiment.provenance()
    for record in (selection,data):
        assert record['status']=='passed' and record['npu_calls']==0
        assert all(record[key]==value for key,value in provenance.items())
    assert prepare['sources']==experiment.mixed.sources()
    assert len(prepare['matrices'])==210 and len(prepare['trajectories'])==40
    assert all(row['calibrated_sse']<=row['rtn_sse']+1e-15 for row in prepare['matrices'])
    assert len(selection['ablations'])==60
    assert {tuple(row['removed']) for row in selection['ablations']}==experiment.GLOBAL
    def check(record,count):
        rows=record['positions']
        assert len(rows)==count and record['summary']==experiment.summary(rows)
        for row in rows:
            assert row['quality_gate_passed']==(row['relative_l2']<.03 and row['next_token_kl']<.02)
            assert -.0000000001<=row['kl_fp64'] and 0<=row['total_variation']<=1.0000000001
            assert 0<=row['offset_energy_fraction']<=1.0000000001
            assert ids_consistent(row)
            assert math.isclose(row['centered_delta_relative_l2']**2,
                  row['relative_l2']**2*(1-row['offset_energy_fraction']),rel_tol=1e-4,abs_tol=1e-12)
    for record in selection['candidates']:
        check(record,290)
        assert record['fp16_weight_fraction']<=.5
        assert len(record['protected'])*884736==record['fp16_elements']
    best=min(selection['candidates'],key=lambda row:(*experiment.mixed.common.ranking(row['summary']),row['fp16_elements']))
    assert best['protected']==selection['selected'] and best['label']==selection['selected_label']
    assert data['selection_sha256']==experiment.mixed.common.sha(experiment.OUT/'selection.json')
    composition=experiment.read('composition-checks')
    assert composition['status']=='passed' and all(composition[key]==value for key,value in provenance.items())
    assert composition['identical_global_graphs']==composition['selected_graphs']==30
    assert composition['protected_projections']==len(selection['selected'])
    assert composition['selection_sha256']==data['selection_sha256']
    expected=[(row['tag'],p) for row in data['cases'] for p in range(row['offset'],len(row['token_ids'][0]))]
    assert len(data['cases'])==17 and len(expected)==142
    for record in data['variants'].values():
        check(record,142)
        assert [(row['tag'],row['position']) for row in record['positions']]==expected
    totals={}
    for label in data['variants']:
        record=experiment.read('npu-'+label)
        assert record['status']=='passed' and all(record[key]==value for key,value in provenance.items())
        assert record['selection_sha256']==data['selection_sha256']
        check(record,142)
        assert [(row['tag'],row['position']) for row in record['positions']]==expected
        assert len(record['graphs'])==30 and len(record['warm'])==17
        assert all(row['calls']==30 and row['repeated_arrays']==91 and row['seed_arrays']==90
                   and row['payload_copies_and_compiles']==0 for row in record['warm'])
        final=record['final_stats']
        assert final['npu_calls']==18330 and final['closed'] and not final['context_failed']
        assert all(final[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
        for graph in record['graphs']:
            integer=[m for m in graph['evidence']['modes'] if not m['elementwise'] and m['wmode'] in (1,2)]
            assert bool(integer)==(label!='fp16') and all(m['amode']==0 and m['wmode']==2 for m in integer)
        if label in ('mixed','selected'):
            target=experiment.GLOBAL if label=='mixed' else frozenset(tuple(key) for key in selection['selected'])
            assert frozenset(tuple(key) for key in record['protected'])==target
        totals[label]=dict(**record['summary'],constant_bytes=sum(row['evidence']['constant_bytes'] for row in record['graphs']))
    result=dict(status='passed',selected_label=selection['selected_label'],npu=totals,
                total_npu_calls=18330*len(totals),**provenance)
    experiment.write('audit',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

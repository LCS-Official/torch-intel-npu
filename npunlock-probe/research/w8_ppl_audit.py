"""Audit fixed-corpus losses, free-generation accounting and NPU evidence."""
import json
import math

import w8_ppl_generation as experiment


def generation_comparison(rows, baseline):
    result=[]
    for row,ref in zip(rows,baseline):
        assert row['tag']==ref['tag']
        a,b=row['token_ids'],ref['token_ids']
        first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
        result.append(dict(tag=row['tag'],exact=a==b,matching_prefix=first,
                           first_divergence=None if a==b else first,actual_length=len(a),reference_length=len(b)))
    return dict(exact=sum(row['exact'] for row in result),cases=len(result),details=result)


def main():
    data=experiment.frozen()
    provenance=experiment.provenance()
    cpu=experiment.read('cpu')
    assert cpu['status']=='passed' and cpu['npu_calls']==0
    expected=[(row['offset'],window['start'],64-window['first']) for row in data['cases'] for window in experiment.windows(row['tokens'])]
    assert len(expected)==56 and sum(count for _,_,count in expected)==2048
    def check(record):
        assert [(row['offset'],row['start'],row['count']) for row in record['windows']]==expected
        assert record['summary']==experiment.aggregate(record['windows'])
        assert all(math.isfinite(row['nll_sum']) and row['nll_sum']>=0 for row in record['windows'])
        assert len(record['generation'])==8
        for row,prompt in zip(record['generation'],data['generation']):
            tokens=row['token_ids']
            assert row['tag']==prompt['tag'] and 1<=len(tokens)<=32
            assert row['consumed']==len(prompt['tokens'])+len(tokens)-1<=64
            assert not any(token in data['eos_ids'] for token in tokens[:-1])
            assert (row['stop_reason']=='eos')==(tokens[-1] in data['eos_ids'])
            assert row['stop_reason']=='eos' or len(tokens)==32
    for key,value in provenance.items():assert cpu[key]==value
    for record in cpu['variants'].values():check(record)
    assert all(row['nll_sum']==row['reference_nll_sum'] for row in cpu['variants']['fp16']['windows'])
    # Independent alignment check against the library's shifted-label loss.
    with experiment.torch.no_grad():
        official,tokenizer=experiment.layers.mixed.common.model()
        ids=experiment.torch.tensor([data['cases'][0]['tokens'][:65]])
        builtin=float(official(ids,labels=ids,use_cache=False).loss)
        ours=cpu['variants']['fp16']['windows'][0]['nll_sum']/64
        assert abs(builtin-ours)<.0003,(builtin,ours)
        original=official(ids[:,:64],use_cache=False).logits[:,:63]
        changed=ids[:,:64].clone();changed[0,-1]=(int(changed[0,-1])+1)%official.config.vocab_size
        assert experiment.torch.equal(original,official(changed,use_cache=False).logits[:,:63])
        ordinary=[]
        for prompt in data['generation'][:3]:
            ids=experiment.torch.tensor([prompt['tokens']])
            generated=official.generate(ids,attention_mask=experiment.torch.ones_like(ids),do_sample=False,
                max_new_tokens=32,eos_token_id=data['eos_ids'],pad_token_id=data['eos_ids'][0])
            tokens=generated[0,len(prompt['tokens']):].tolist()
            baseline=next(row for row in cpu['variants']['fp16']['generation'] if row['tag']==prompt['tag'])
            ordinary.append(dict(tag=prompt['tag'],token_ids=tokens,text=tokenizer.decode(tokens,skip_special_tokens=True),
                                 matches_sequential_cpu=tokens==baseline['token_ids']))
        del official,original,ids,changed
    all_npu={label:experiment.read('npu-'+label) for label in ('fp16','w8','mixed','selected')}
    output={}
    for label,record in all_npu.items():
        assert record['status']=='passed'
        assert all(record[key]==value for key,value in provenance.items())
        check(record)
        assert len(record['local_checks'])==30 and len(record['warm'])==8 and len(record['graphs'])==90
        assert all(row['relative_l2']<.003 and row['max_scaled_error']<.01 for row in record['local_checks'])
        assert all(row['calls']==30 and row['repeated_arrays']==91 and row['seed_arrays']==90
                   and row['payload_copies_and_compiles']==0 for row in record['warm'])
        final=record['final_stats']
        calls=30*(56+sum(row['consumed'] for row in record['generation'])+8)+120
        assert final['npu_calls']==calls and final['closed'] and not final['context_failed']
        assert all(final[key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
        constants={}
        for graph in record['graphs']:
            integer=[m for m in graph['evidence']['modes'] if not m['elementwise'] and m['wmode'] in (1,2)]
            assert bool(integer)==(label!='fp16') and all(m['amode']==0 and m['wmode']==2 for m in integer)
            kind=graph['key'][-1] or 'forward64'
            constants[kind]=constants.get(kind,0)+graph['evidence']['constant_bytes']
        wanted=experiment.layers.GLOBAL if label=='mixed' else experiment.POLICY if label=='selected' else frozenset()
        assert frozenset(tuple(key) for key in record['protected'])==wanted
        comparison=all_npu['fp16']['summary']
        own=cpu['variants'][label]['summary']
        streams=[]
        for case in data['cases']:
            current=experiment.aggregate([row for row in record['windows'] if row['offset']==case['offset']])
            baseline=experiment.aggregate([row for row in all_npu['fp16']['windows'] if row['offset']==case['offset']])
            streams.append(dict(offset=case['offset'],**current,delta_nll_vs_npu_fp16=current['nll']-baseline['nll']))
        output[label]=dict(**record['summary'],npu_calls=calls,constant_bytes=constants,streams=streams,
            delta_nll_vs_cpu_fp16=record['summary']['nll']-cpu['variants']['fp16']['summary']['nll'],
            delta_nll_vs_npu_fp16=record['summary']['nll']-comparison['nll'],
            ppl_ratio_vs_npu_fp16=record['summary']['ppl']/comparison['ppl'],
            delta_nll_vs_cpu_same_policy=record['summary']['nll']-own['nll'],
            generation_vs_cpu_fp16=generation_comparison(record['generation'],cpu['variants']['fp16']['generation']),
            generation_vs_npu_fp16=generation_comparison(record['generation'],all_npu['fp16']['generation']),
            generation_vs_cpu_same_policy=generation_comparison(record['generation'],cpu['variants'][label]['generation']))
    result=dict(status='passed',**provenance,npu=output,total_npu_calls=sum(row['npu_calls'] for row in output.values()),
                builtin_loss_delta=builtin-ours,causal_prefix_unchanged=True)
    result['ordinary_generate_checks']=ordinary
    experiment.write('audit',result)
    compact={label:{key:value for key,value in row.items() if key not in ('streams','constant_bytes') and not key.startswith('generation_')}
             for label,row in output.items()}
    print(json.dumps(dict(status='passed',npu=compact,total_npu_calls=result['total_npu_calls']),indent=2))


if __name__=='__main__':main()

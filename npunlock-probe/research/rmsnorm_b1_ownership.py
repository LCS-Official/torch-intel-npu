"""Fused normalization: batch isolation, resident gamma and output ownership."""
import gc
import hashlib
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-b1-fusion'


def main():
    import torch
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolLayer
    from torch_npu_smollm_stack_demo import errors
    report=dict(status='running',checks=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['torch_npu_smollm.py','torch_intel_npu.py','npunlock/python/npunlock/npu4000.py','npunlock-probe/research/rmsnorm_b1_ownership.py']})
    try:
        def work():
            torch.set_num_threads(8);torch.manual_seed(1091)
            for mode in ['fp16','w8a16']:
                for batch,prefix in [(1,1),(1,7),(1,32),(2,1),(3,1),(8,63),(1,63)]:
                    model=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode)
                    solo=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode)
                    host=torch.randn(batch,prefix+1,576).half()*.1
                    prefix_input=host[:,:prefix].contiguous().to('intel_npu')
                    prefill,state=model.prefill(prefix_input)
                    assert torch.equal(prefill.cpu().view(torch.int16),model(prefix_input).cpu().view(torch.int16))
                    prefix_reference,_=model.reference_cpu(host[:,:prefix].contiguous())
                    prefix_error=errors(prefill.cpu(),prefix_reference)
                    assert prefix_error['relative_l2']<.003 and prefix_error['max_scaled_error']<.01,prefix_error
                    old_keys=state.keys.cpu();old_values=state.values.cpu()
                    one=host[:,-1:].contiguous().to('intel_npu');cold_before=npu.stats();model.decode(one,state);cold_after=npu.stats()
                    for field in ['host_upload_copies','host_download_copies']:assert cold_before[field]==cold_after[field]
                    assert all(name in model._norm_parameters for name in ['input_layernorm.weight','post_attention_layernorm.weight'])
                    before=npu.stats();value,cache=model.decode(one,state);after=npu.stats()
                    for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                    assert after['npu_calls']-before['npu_calls']==2
                    actual=value.cpu();keys=cache.keys.cpu();values=cache.values.cpu();reference,_=model.reference_cpu(host)
                    error=errors(actual,reference[:,-1:]);assert error['relative_l2']<.003 and error['max_scaled_error']<.01,error
                    parity=[];comparisons=[]
                    for i in range(batch):
                        _,single=solo.prefill(host[i:i+1,:prefix].contiguous().to('intel_npu'))
                        result,single=solo.decode(host[i:i+1,-1:].contiguous().to('intel_npu'),single)
                        pairs=[(actual[i:i+1],result.cpu()),(keys[i:i+1,:,:prefix+1],single.keys.cpu()[:,:,:prefix+1]),
                               (values[i:i+1,:,:prefix+1],single.values.cpu()[:,:,:prefix+1])]
                        sample_errors=[errors(a,b) for a,b in pairs]
                        assert all(e['relative_l2']<.003 and e['max_scaled_error']<.01 for e in sample_errors),sample_errors
                        parity.append(all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in pairs))
                        comparisons.append(sample_errors)
                    changed=host[:,-1:].clone();index=batch//2;changed[index]*=-1
                    branch,branch_cache=model.decode(changed.contiguous().to('intel_npu'),state)
                    mask=torch.arange(batch)!=index;branch=branch.cpu()
                    assert torch.equal(actual[mask].view(torch.int16),branch[mask].view(torch.int16))
                    assert not torch.equal(actual[index].view(torch.int16),branch[index].view(torch.int16))
                    for tensor,other in [(keys,branch_cache.keys),(values,branch_cache.values)]:
                        assert torch.equal(tensor[mask].view(torch.int16),other.cpu()[mask].view(torch.int16))
                    assert torch.equal(value.cpu().view(torch.int16),actual.view(torch.int16))
                    assert torch.equal(old_keys.view(torch.int16),state.keys.cpu().view(torch.int16))
                    assert torch.equal(old_values.view(torch.int16),state.values.cpu().view(torch.int16))
                    assert torch.equal(keys[:,:,:prefix].view(torch.int16),old_keys[:,:,:prefix].view(torch.int16))
                    assert torch.equal(values[:,:,:prefix].view(torch.int16),old_values[:,:,:prefix].view(torch.int16))
                    assert not keys[:,:,prefix+1:].count_nonzero() and not values[:,:,prefix+1:].count_nonzero()
                    assert float(cache._position.cpu()[0])==prefix+1 and cache.length==prefix+1
                    entry=dict(mode=mode,batch=batch,prefix_tokens=prefix,execution=error,prefill_execution=prefix_error,prefill_forward_bits_equal=True,
                        graph_calls=2,cold_decode_host_copies=0,held_output_unchanged=True,solo_bits_equal_samples=parity,solo_comparisons=comparisons,
                        perturbation_isolated=True,old_snapshot_unchanged=True,warm_host_copies_and_compiles=0)
                    report['checks'].append(entry);print(entry,flush=True)
            # Changing batch shapes on the same layer reuses the two-row gamma
            # between B1 and B2, while public shapes and held results stay exact.
            held=[];gamma_addresses=None
            for batch in [1,2,1,8,3,1]:
                x=torch.randn(batch,1,576).half().to('intel_npu');output,cache=model.prefill(x)
                if batch in [1,2]:
                    addresses=tuple(model._norm_parameters[(name,2)].data_ptr() for name in ['input_layernorm.weight','post_attention_layernorm.weight'])
                    if gamma_addresses is None:gamma_addresses=addresses
                    assert addresses==gamma_addresses
                assert tuple(output.shape)==(batch,1,576) and tuple(cache.keys.shape)==(batch,3,64,64)
                assert torch.equal(output.cpu().view(torch.int16),model(x).cpu().view(torch.int16))
                changed=(x.cpu()*-2).contiguous().to('intel_npu');cold_before=npu.stats();model.decode(changed,cache);cold_after=npu.stats()
                for field in ['host_upload_copies','host_download_copies']:assert cold_before[field]==cold_after[field]
                result,next_cache=model.decode(changed,cache)
                held.append([(output,output.cpu()),(cache.keys,cache.keys.cpu()),(cache.values,cache.values.cpu()),
                    (result,result.cpu()),(next_cache.keys,next_cache.keys.cpu()),(next_cache.values,next_cache.values.cpu())])
            for group in held:
                for output,saved in group:assert torch.equal(output.cpu().view(torch.int16),saved.view(torch.int16))
            report['checks'].append(dict(batch_sequence=[1,2,1,8,3,1],held_output_cache_groups=len(held),all_bits_unchanged=True,gamma_shared_between_b1_b2=True))
            # Concurrent callers share cached sessions/parameters, while public
            # outputs and KV stay independently owned. The existing NPU lock
            # serializes graph execution; this checks correctness, not scaling.
            from concurrent.futures import ThreadPoolExecutor
            def decode(_):
                with torch.no_grad():return model.decode(one,state)
            reference,reference_cache=decode(0);saved=reference.cpu();before=npu.stats()
            with ThreadPoolExecutor(max_workers=2) as workers:outputs=list(workers.map(decode,range(6)))
            after=npu.stats();assert after['npu_calls']-before['npu_calls']==12
            for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
            for output,cache in outputs:
                assert torch.equal(output.cpu().view(torch.int16),saved.view(torch.int16))
                for name in ['keys','values','_position']:
                    a=getattr(cache,name);b=getattr(reference_cache,name);bits=torch.int16 if a.dtype==torch.float16 else torch.int32
                    assert torch.equal(a.cpu().view(bits),b.cpu().view(bits))
            groups=[[item[0] for item in outputs]]+[[getattr(item[1],name) for item in outputs] for name in ['keys','values','_position']]
            for tensors in groups:assert len({t.untyped_storage()._cdata for t in tensors})==6
            assert torch.equal(reference.cpu().view(torch.int16),saved.view(torch.int16))
            report['checks'].append(dict(workers=2,decode_calls=6,graph_calls=12,public_outputs_and_kv_independent=True,all_bits_equal=True,warm_host_copies_and_compiles=0))
        with torch.no_grad():work()
        gc.collect();assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown();report['final_stats']=npu.stats();assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'ownership.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

"""Distinct-sample decode parity and perturbation checks, including odd batches."""
import gc
import hashlib
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-kv-batch'


def main():
    import torch
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolLayer
    from torch_npu_smollm_stack_demo import errors
    report=dict(status='running',checks=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['torch_npu_smollm.py','npunlock-probe/research/smollm_batch_isolation.py']})
    try:
        def work():
            torch.set_num_threads(8);torch.manual_seed(1091)
            for mode in ['fp16','w8a16']:
                for batch,prefix in [(2,1),(3,1),(4,1),(5,7),(6,1),(7,32),(8,1),(8,63)]:
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
                    one=host[:,-1:].contiguous().to('intel_npu');model.decode(one,state)
                    before=npu.stats();value,cache=model.decode(one,state);after=npu.stats()
                    for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                    assert after['npu_calls']-before['npu_calls']==4
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
                    assert torch.equal(old_keys.view(torch.int16),state.keys.cpu().view(torch.int16))
                    assert torch.equal(old_values.view(torch.int16),state.values.cpu().view(torch.int16))
                    assert torch.equal(keys[:,:,:prefix].view(torch.int16),old_keys[:,:,:prefix].view(torch.int16))
                    assert torch.equal(values[:,:,:prefix].view(torch.int16),old_values[:,:,:prefix].view(torch.int16))
                    assert not keys[:,:,prefix+1:].count_nonzero() and not values[:,:,prefix+1:].count_nonzero()
                    assert float(cache._position.cpu()[0])==prefix+1 and cache.length==prefix+1
                    entry=dict(mode=mode,batch=batch,prefix_tokens=prefix,execution=error,prefill_execution=prefix_error,prefill_forward_bits_equal=True,
                        graph_calls=4,solo_bits_equal_samples=parity,solo_comparisons=comparisons,
                        perturbation_isolated=True,old_snapshot_unchanged=True,warm_host_copies_and_compiles=0)
                    report['checks'].append(entry);print(entry,flush=True)
        with torch.no_grad():work()
        gc.collect();assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown();report['final_stats']=npu.stats();assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'isolation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

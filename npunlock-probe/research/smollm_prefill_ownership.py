"""Check causal batch isolation and retained multi-token prefill snapshots."""
import gc
import hashlib
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-prefill-fusion'


def main():
    import torch
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolLayer
    report=dict(status='running',checks=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['npunlock-probe/research/smollm_prefill_ownership.py','torch_npu_smollm.py','torch_intel_npu.py']})
    try:
        def work():
            torch.set_num_threads(8);torch.manual_seed(8921)
            for mode in ['fp16','w8a16']:
                model=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode);held=[]
                for batch,length in [(1,2),(3,7),(2,17),(8,63),(1,64),(4,32),(1,2)]:
                    host=torch.randn(batch,length,576).half()*.1;x=host.to('intel_npu')
                    # A new length uploads one immutable FP32 metadata scalar.
                    # Exclude that initialization when checking resident gamma.
                    position_before=npu.stats();npu._scalar(length,torch.float32);position_after=npu.stats()
                    position_uploads=position_after['host_upload_copies']-position_before['host_upload_copies']
                    cold_before=npu.stats();model.prefill(x);cold_after=npu.stats()
                    # Only the first shape may upload gamma. Later shapes
                    # derive their broadcasts from the resident parameters.
                    if held:
                        for field in ['host_upload_copies','host_download_copies']:assert cold_before[field]==cold_after[field],(field,cold_before[field],cold_after[field])
                    before=npu.stats();out,state=model.prefill(x);after=npu.stats()
                    assert after['npu_calls']-before['npu_calls']==1 and after['buffer_allocations']-before['buffer_allocations']==3
                    for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                    saved=[out.cpu(),state.keys.cpu(),state.values.cpu(),state._position.cpu()]
                    index=batch//2;position=length//2;changed=host.clone();changed[index,position]*=-2
                    branch,cache=model.prefill(changed.to('intel_npu'))
                    mask=torch.arange(batch)!=index
                    assert torch.equal(saved[0][mask].view(torch.int16),branch.cpu()[mask].view(torch.int16))
                    assert torch.equal(saved[0][index,:position].view(torch.int16),branch.cpu()[index,:position].view(torch.int16))
                    assert not torch.equal(saved[0][index,position:].view(torch.int16),branch.cpu()[index,position:].view(torch.int16))
                    for old,new in zip(saved[1:3],[cache.keys.cpu(),cache.values.cpu()]):
                        assert torch.equal(old[mask].view(torch.int16),new[mask].view(torch.int16))
                        assert torch.equal(old[index,:,:position].view(torch.int16),new[index,:,:position].view(torch.int16))
                        assert not old[:,:,length:].count_nonzero()
                    assert state.length==length and float(saved[-1][0])==length
                    for a,b in [(out,branch),(state.keys,cache.keys),(state.values,cache.values)]:
                        assert a.untyped_storage()._cdata!=b.untyped_storage()._cdata
                    held.append(list(zip([out,state.keys,state.values,state._position],saved)))
                    report['checks'].append(dict(mode=mode,batch=batch,length=length,graph_calls=1,warm_allocations=3,
                        warm_host_copies_and_compiles=0,causal_batch_isolation=True,independent_output_kv=True,position_and_padding_correct=True,
                        position_metadata_prewarmed=True,initial_position_metadata_uploads=position_uploads))
                for group in held:
                    for tensor,old in group:assert torch.equal(tensor.cpu(),old)
                report['checks'].append(dict(mode=mode,held_groups=len(held),old_snapshots_unchanged=True,
                    changing_shape_cold_gamma_uploads_after_first=0))
        with torch.no_grad():work()
        gc.collect();assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers'],npu.stats()
        npu.shutdown();report.update(status='passed',final_stats=npu.stats());assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'ownership.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report['status'],len(report['checks']));return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

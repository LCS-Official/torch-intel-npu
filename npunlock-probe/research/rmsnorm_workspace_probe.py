"""Check changing inputs/shapes, fresh helper results and scratch ownership."""
import gc
import hashlib
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-small'


def main():
    import torch
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolLayer
    report=dict(status='running',checks=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['npunlock-probe/research/rmsnorm_workspace_probe.py','torch_npu_smollm.py','torch_intel_npu.py']})
    try:
        def work():
            torch.set_num_threads(8);torch.manual_seed(6141)
            model=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16')
            names=['input_layernorm.weight','post_attention_layernorm.weight']
            for batch,length in [(1,1),(2,1),(4,1),(8,1),(1,32),(4,64),(8,64),(1,1),(8,1)]:
                shape=(batch,length,576);x=torch.randn(shape).half().to('intel_npu');changed=(torch.randn(shape)*3).half().to('intel_npu')
                for name in names:
                    fresh=model._norm_device(x,name);saved=fresh.cpu().view(torch.int16)
                    expected=model._norm_device(changed,name).cpu()
                    first=model._norm_device(x,name,reuse=True);address=first.data_ptr();before=npu.stats()
                    actual=model._norm_device(changed,name,reuse=True);after=npu.stats()
                    assert actual.data_ptr()==address and fresh.data_ptr()!=address
                    assert torch.equal(actual.cpu().view(torch.int16),expected.view(torch.int16))
                    assert torch.equal(fresh.cpu().view(torch.int16),saved)
                    assert after['buffer_allocations']==before['buffer_allocations'] and after['npu_calls']-before['npu_calls']==1
                    for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                    assert len(model._norm_workspaces)<=2
                    assert sum(t.numel()*t.element_size() for t in model._norm_workspaces.values())<=2*512*576*2
                    report['checks'].append(dict(shape=shape,name=name,reused_pointer=True,changed_input_bits_equal=True,
                        held_fresh_output_unchanged=True,warm_allocations=0,graph_calls=1))
            # Serial 512 rows must retain independent chunk outputs; its private
            # fallback intentionally does not reuse the one-buffer workspace.
            x=torch.randn((8,64,576)).half().to('intel_npu')
            for name in names:
                parallel=model._norm_device(x,name).cpu();serial=model._norm_device(x,name,parallel=False,reuse=True).cpu()
                assert torch.equal(parallel.view(torch.int16),serial.view(torch.int16))
                report['checks'].append(dict(name=name,rows=512,serial_chunks_bits_equal=True))
            # Input aliases are rejected, including a full dense permutation.
            x=torch.randn((2,2,576)).half().to('intel_npu');scratch=model._norm_device(x,names[0],reuse=True);saved=scratch.cpu()
            for tag,value in [('contiguous',scratch),('dense_permutation',scratch.permute(1,0,2))]:
                before=npu.stats()
                try:model._norm_device(value,names[0],reuse=True)
                except ValueError as error:assert 'alias' in str(error)
                else:raise AssertionError('scratch alias was accepted')
                after=npu.stats();assert not after['context_failed'] and torch.equal(scratch.cpu().view(torch.int16),saved.view(torch.int16))
                report['checks'].append(dict(alias=tag,rejected=True,scratch_unchanged=True,graph_calls=after['npu_calls']-before['npu_calls']))
            model._norm_device(x,names[0],reuse=True)
            # Keep public outputs and all prior KV snapshots alive while scratch
            # is overwritten, resized by forward, and used from two callers.
            from concurrent.futures import ThreadPoolExecutor
            prefix=torch.randn((2,3,576)).half().to('intel_npu');one=torch.randn((2,1,576)).half().to('intel_npu')
            output,state=model.prefill(prefix);held=[(output,output.cpu())];snapshots=[]
            for step in range(3):
                snapshots.append((state,state.keys.cpu(),state.values.cpu(),state._position.cpu()))
                output,state=model.decode(one,state);held.append((output,output.cpu()));model(prefix)
            for output,saved in held:assert torch.equal(output.cpu().view(torch.int16),saved.view(torch.int16))
            for state,keys,values,position in snapshots:
                for actual,saved in [(state.keys,keys),(state.values,values),(state._position,position)]:
                    bits=torch.int16 if actual.dtype==torch.float16 else torch.int32
                    assert torch.equal(actual.cpu().view(bits),saved.view(bits))
            report['checks'].append(dict(held_public_outputs=len(held),held_kv_snapshots=len(snapshots),all_bits_unchanged=True))
            def decode(_):
                with torch.no_grad():return model.decode(one,state)[0]
            reference=decode(0).cpu();before=npu.stats()
            with ThreadPoolExecutor(max_workers=2) as workers:outputs=list(workers.map(decode,range(6)))
            after=npu.stats();assert after['npu_calls']-before['npu_calls']==24
            for output in outputs:assert torch.equal(output.cpu().view(torch.int16),reference.view(torch.int16))
            assert len({t.untyped_storage()._cdata for t in outputs})==6
            report['checks'].append(dict(workers=2,decode_calls=6,public_outputs_independent=True,all_bits_equal=True))
        with torch.no_grad():work()
        gc.collect();assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown();report.update(status='passed',final_stats=npu.stats());assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'workspace.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

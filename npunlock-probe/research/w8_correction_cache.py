"""Real NPU miss/hit checks for distinct W8 prefill residual factor identities."""
import gc
import json
import os
from pathlib import Path
import traceback

import torch
import w8_guarded_quality as study
from torch_npu_lowbit_correction import LowRankCorrection


@torch.no_grad()
def main():
    import torch_intel_npu as npu
    import torch_npu_graph_cache as cache
    from torch_npu_smollm import FrozenSmolLayer
    folder=study.OUT/'graph-cache'
    folder.mkdir(exist_ok=False)
    assert not os.environ.get('NPUNLOCK_GRAPH_CACHE')
    os.environ['NPUNLOCK_GRAPH_CACHE']=str(folder)
    packed=study.base.mixed.weights()[0];factor=study.base.corrections(32)[0]['mlp.down_proj.weight'];a,b=factor.tensors()
    half=LowRankCorrection(factor.shape,32,a.numpy().tobytes(),(b*.5).half().numpy().tobytes())
    variants=(('plain',None),('corrected',factor),('half',half),('rank16',factor.truncated(16)))
    host=(torch.randn(1,2,576,generator=torch.Generator().manual_seed(452))/4).half()
    results={};report=dict(status='running',runs=[]);before=cache.stats()
    def work():
        for repetition in range(2):
            for name,factors in variants:
                model=FrozenSmolLayer.from_checkpoint(study.base.mixed.common.MODEL,packed_weights=packed,
                    corrections=None if factors is None else {'mlp.down_proj.weight':factors})
                expected,_=model.reference_cpu(host)
                value,state=model.prefill(host.to('intel_npu'));actual=value.cpu()
                error=study.base.mixed.common.errors(actual,expected)
                assert error['relative_l2']<.003 and error['max_scaled_error']<.01,(name,error)
                if repetition:torch.testing.assert_close(actual,results[name],rtol=0,atol=0)
                else:results[name]=actual
                key=next(key for key in model._keys.values() if key[0]=='smollm_prefill_recipe')
                report['runs'].append(dict(name=name,repetition=repetition,graph_digest=key[1].hex(),errors=error))
                del model,value,state;gc.collect()
                # Drop all rebuildable in-memory graphs to require a disk load next time.
                limit=npu._MAX_SESSIONS;npu._MAX_SESSIONS=1
                try:npu._make_session_room()
                finally:npu._MAX_SESSIONS=limit
        assert len({r['graph_digest'] for r in report['runs']})==4
        assert len(list(folder.glob('*.npc')))==4
        assert all(not torch.equal(results[a],results[b]) for i,(a,_) in enumerate(variants) for b,_ in variants[i+1:])
        after=cache.stats();report['cache_delta']={k:after[k]-before[k] for k in ('hits','misses','compiles','writes','rejected','ir_skipped')}
        assert report['cache_delta']==dict(hits=4,misses=4,compiles=4,writes=4,rejected=0,ir_skipped=4)
    try:work();report['status']='passed'
    except Exception:report.update(status='failed',traceback=traceback.format_exc())
    report['before_close']=npu.stats();npu.shutdown();report['shutdown']=npu.stats()
    report['healthy']=report['shutdown']['closed'] and not report['shutdown']['context_failed'] and all(
        report['shutdown'][k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_sessions','quarantined_buffer_bytes'))
    os.environ.pop('NPUNLOCK_GRAPH_CACHE',None)
    study.base.write('cache-check',report)
    print(json.dumps(report,indent=2),flush=True)
    return int(report['status']!='passed' or not report['healthy'])


if __name__=='__main__':raise SystemExit(main())

"""First isolated static-KV compilation/numerical probe."""
import gc
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-kv'


def main():
    import torch
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolLayer
    from torch_npu_smollm_stack_demo import errors
    torch.set_num_threads(8);torch.manual_seed(1091)
    report=dict(status='running',checks=[])
    try:
        with torch.no_grad():
            def work():
                layer=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='w8a16')
                host=torch.randn(1,11,576).half()*.1
                prefix=host[:,:8].contiguous();x=prefix.to('intel_npu')
                out,cache=layer.prefill(x)
                comparison=errors(out.cpu(),layer(x).cpu())
                report['checks'].append(dict(stage='prefill',comparison=comparison,length=cache.length))
                print(report['checks'][-1],flush=True)
                assert comparison['relative_l2']<.003 and comparison['max_scaled_error']<.01
                for length in range(9,12):
                    one=host[:,length-1:length].contiguous().to('intel_npu')
                    decoded,cache=layer.decode(one,cache)
                    full=layer(host[:,:length].contiguous().to('intel_npu')).cpu()[:,-1:]
                    comparison=errors(decoded.cpu(),full)
                    entry=dict(stage='decode',comparison=comparison,length=cache.length)
                    report['checks'].append(entry);print(entry,flush=True)
                    assert comparison['relative_l2']<.003 and comparison['max_scaled_error']<.01
            work();gc.collect();report['stats_before_shutdown']=npu.stats()
            assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
            npu.shutdown();report['final_stats']=npu.stats();assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),stats=npu.stats())
        print(report['traceback'],flush=True)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'static-probe.json').write_text(json.dumps(report,indent=2))
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

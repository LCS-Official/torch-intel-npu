"""Run allocator lifetime/limits, failed-context and device regressions."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/decode-cache'


def order():
    import torch
    import torch_intel_npu as npu
    with torch.no_grad():
        npu.empty_cache()
        limit=npu.stats()['allocation_cache_limit_blocks']
        blocks=[torch.empty(i+1,dtype=torch.float32,device='intel_npu') for i in range(limit)]
        while blocks:
            value=blocks.pop(0);del value
        # Reuse a middle entry and return it as the newest idle entry.
        before=npu.stats();value=torch.empty(33,dtype=torch.float32,device='intel_npu');del value
        assert npu.stats()['driver_allocations']==before['driver_allocations']
        value=torch.empty(10000,dtype=torch.float32,device='intel_npu');del value
        before=npu.stats();value=torch.empty(1,dtype=torch.float32,device='intel_npu');del value
        assert npu.stats()['driver_allocations']==before['driver_allocations']+1,'oldest block was not evicted'
        before=npu.stats()
        for size in (33,10000):
            value=torch.empty(size,dtype=torch.float32,device='intel_npu');del value
        after=npu.stats();assert after['driver_allocations']==before['driver_allocations']
        assert after['cached_buffer_blocks']==limit and after['cached_buffer_bytes']<=16*1024*1024
        gc.collect();npu.shutdown();final=npu.stats()
        assert not final['context_failed'] and final['live_buffers']==final['cached_buffer_bytes']==0
    (OUT/'idle-order.json').write_text(json.dumps(dict(status='passed',checks=['oldest_release_evicted','middle_hit_preserves_order','recent_sizes_reused'],
        final_stats=final),indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',choices=('guards','order','failure','kv-guards','kv-boundary','device','training'))
    args=parser.parse_args()
    if args.child:
        if args.child=='guards':
            import torch_npu_warm_profile as common
            common.OUT=OUT;common.guards()
        elif args.child=='order':order()
        elif args.child=='failure':
            import torch_npu_warm_profile as common
            import allocator_failure_probe as failure
            # The diagnostic DLL comes from the unchanged native runtime source.
            import shutil
            shutil.copy2(common.OUT/'session_failure.dll',OUT/'session_failure.dll')
            common.OUT=OUT;failure.OUT=OUT;failure.run()
        elif args.child in ('kv-guards','kv-boundary'):
            import torch_npu_smollm_kv_demo as demo
            demo.OUT=OUT/'regressions/kv';demo.OUT.mkdir(parents=True,exist_ok=True)
            return demo.execute(args.child.removeprefix('kv-'))
        else:
            module='torch_npu_device_demo' if args.child=='device' else 'torch_npu_training_demo'
            import importlib
            demo=importlib.import_module(module)
            demo.OUT=OUT/'regressions';demo.OUT.mkdir(exist_ok=True)
            if args.child=='device':
                demo.DESTINATION=demo.OUT/'device.json';return demo.execute(True)
            demo.REPORT=demo.OUT/'training.json';return demo.execute(True,False)
        return 0
    for case in ('guards','order','failure','kv-guards','kv-boundary','device','training'):
        with (OUT/f'check-{case}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--child',case],cwd=ROOT,stdout=log,stderr=log,timeout=600)
        if result.returncode:
            print((OUT/f'check-{case}.log').read_text(encoding='utf-8'),flush=True);return result.returncode
        print(case,'passed',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

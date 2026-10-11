"""Warm W8 calibration controls and current host/device profiles.

Reuses the existing interval profiler and same-process timing helpers.
No production implementation or historical result is modified.
"""
import argparse
import ctypes as ct
import gc
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np
from w8_calibration import SOURCES, load_weights
from w8_quality_probe import ROOT, MODEL, sha

OUT=ROOT/'npunlock-probe/research/results/w8-performance'
HELPERS=('torch_npu_device_profile.py','torch_npu_warm_profile.py','npunlock-probe/research/rmsnorm_reduce_probe.py',
         'npunlock-probe/research/w8_calibration.py','npunlock/src/graphinfer/session_win32.c')
MODES=('baseline','calibrated','fp16')
SHAPES=((1,32),(4,16))


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def profile_binary():
    previous=ROOT/'npunlock-probe/research/results/smollm-device-profile'
    build=read(previous/'build.json')
    assert sha(previous/'session_task_profile.dll')==build['dll_sha256']
    assert sha(previous/'session_task_profile.c')==build['generated_source_sha256']
    assert sha(ROOT/'npunlock/src/graphinfer/session_win32.c')==build['production_session_sha256']
    host=read(ROOT/'npunlock-probe/research/results/warm-allocation/profile-build.json')
    for name,digest in host['sources'].items():
        if name.endswith('.c'):assert sha(ROOT/name)==digest,name
    return previous,build


def metadata():
    return dict(script_sha256=sha(Path(__file__)),sources={name:sha(ROOT/name) for name in (*SOURCES,*HELPERS)},
        weights_sha256=sha(ROOT/'npunlock-probe/research/results/w8-calibration/weights.npz'),
        checkpoint_sha256=sha(MODEL/'model.safetensors'))


def benchmark(label,host_profile=False):
    import torch
    driver=None
    if host_profile:
        from torch_npu_warm_profile import bind, STAGES
        directory,build=profile_binary();dll=directory/'session_task_profile.dll'
        manifest=read(ROOT/'npunlock-probe/torch-device/build.json');manifest['session_sha256']=build['dll_sha256']
        binding=OUT/'host-bound-build.json';binding.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        bind(dll,binding);driver=ct.CDLL(str(dll))
        driver.warm_diag_reset.argtypes=[];driver.warm_diag_reset.restype=None
        driver.warm_diag_read.argtypes=[ct.POINTER(ct.c_uint64)]*2;driver.warm_diag_read.restype=None
    import torch_intel_npu as npu
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from torch_npu_smollm import FrozenSmolStack
    from rmsnorm_reduce_probe import times
    torch.set_num_threads(8)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as file:embedding=file.get_tensor('model.embed_tokens.weight')
    prepared=load_weights();report=dict(status='running',label=label,host_instrumented=host_profile,rows=[],**metadata());arrays={}
    def snapshots(result):
        output,caches=result
        return [output,*[t for cache in caches for t in (cache.keys,cache.values,cache._position)]]
    with torch.no_grad():
        for batch,prefix in SHAPES:
            tokens=ids.repeat(1,(prefix+ids.shape[1])//ids.shape[1])[:,:prefix+1].repeat(batch,1)
            host=embedding[tokens].half().contiguous();x=host[:,:prefix].contiguous().to('intel_npu');one=host[:,prefix:].contiguous().to('intel_npu')
            stacks={name:FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,mode='fp16' if name=='fp16' else 'w8a16',
                packed_weights=prepared if name=='calibrated' else None) for name in MODES}
            seeds={};anchors={}
            for name,stack in stacks.items():
                initial=stack.prefill(x);decoded=stack.decode(one,initial[1]);seeds[name]=initial[1]
                anchor=[initial[0],decoded[0],*[t for result in (initial,decoded) for t in snapshots(result)[1:]]]
                anchors[name]=[v.cpu().numpy().copy() for v in anchor]
                for i,value in enumerate(anchors[name]):arrays[f'b{batch}-s{prefix}-{name}/{i}']=value
                del value,anchor,initial,decoded
            for operation in ('prefill','decode'):
                runs={name:(lambda stack=stack,name=name:stack.prefill(x) if operation=='prefill' else stack.decode(one,seeds[name])) for name,stack in stacks.items()}
                host_rows={name:[] for name in MODES}
                if driver is not None:
                    def wrap(name,run):
                        def measured():
                            driver.warm_diag_reset();result=run();ns=(ct.c_uint64*9)();counts=(ct.c_uint64*9)();driver.warm_diag_read(ns,counts)
                            host_rows[name].append(dict(times_ms={k:v/1e6 for k,v in zip(STAGES,ns)},counts=dict(zip(STAGES,counts))))
                            return result
                        return measured
                    runs={name:wrap(name,run) for name,run in runs.items()}
                timing=times(runs,31,npu)
                assert timing['warm_delta']['npu_calls']==31*2*3*4
                row=dict(batch=batch,prefix=prefix,operation=operation,**timing)
                if driver is not None:
                    row['host']={name:dict(samples=values[3:],median_stage_ms={stage:statistics.median(v['times_ms'][stage] for v in values[3:]) for stage in STAGES}) for name,values in host_rows.items()}
                    assert all(v['counts'][stage]==4 for values in host_rows.values() for v in values for stage in STAGES[3:])
                report['rows'].append(row);print(label,batch,prefix,operation,timing['median_ms'],flush=True)
                # Timed calls must leave the input snapshot and old outputs intact.
                for name,stack in stacks.items():
                    repeated=stack.prefill(x);decoded=stack.decode(one,seeds[name])
                    current=[repeated[0],decoded[0],*[t for result in (repeated,decoded) for t in snapshots(result)[1:]]]
                    for i,value in enumerate(current):np.testing.assert_array_equal(value.cpu().numpy().view(np.uint8),anchors[name][i].view(np.uint8))
                    for i,cache in enumerate(seeds[name]):
                        for j,value in enumerate((cache.keys,cache.values,cache._position)):
                            np.testing.assert_array_equal(value.cpu().numpy().view(np.uint8),anchors[name][2+i*3+j].view(np.uint8))
                    del value,cache,current,repeated,decoded
            del runs,seeds,stacks,stack,x,one
        gc.collect();npu.shutdown();report['final_stats']=npu.stats()
        assert report['final_stats']['closed'] and not report['final_stats']['context_failed']
        assert all(report['final_stats'][key]==0 for key in ('live_buffers','cached_buffer_bytes','quarantined_buffer_bytes'))
    path=OUT/f'{label}-outputs.npz';np.savez(path,**arrays);report.update(status='passed',outputs_sha256=sha(path))
    (OUT/f'{label}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


def device(batch,prefix,mode):
    import torch_npu_device_profile as profile
    from torch_npu_smollm import FrozenSmolStack
    previous,build=profile_binary();profile.OUT=OUT/f'device-{mode}';profile.OUT.mkdir(exist_ok=True)
    shutil.copy2(previous/'session_task_profile.dll',profile.OUT/'session_task_profile.dll')
    original=FrozenSmolStack.from_checkpoint.__func__;prepared=load_weights() if mode=='calibrated' else None
    def construct(cls,directory,**options):return original(cls,directory,**options,packed_weights=prepared)
    FrozenSmolStack.from_checkpoint=classmethod(construct)
    try:profile.execute(batch,prefix,'fp16' if mode=='fp16' else 'w8a16')
    finally:FrozenSmolStack.from_checkpoint=classmethod(original)
    path=profile.OUT/f'b{batch}-s{prefix}-{"fp16" if mode=="fp16" else "w8a16"}.json'
    report=read(path);report.update(performance_metadata=metadata(),calibration_mode=mode,profile_build=build)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')


def control():
    jobs=[['--case','benchmark','--label',f'benchmark-{i}'] for i in range(2)]
    jobs += [['--case','host','--label','host']]
    jobs += [['--case','device','--batch',str(b),'--prefix',str(s),'--mode',m] for b,s in SHAPES for m in ('fp16','calibrated')]
    for i,job in enumerate(jobs):
        path=OUT/f'worker-{i}.log'
        with path.open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,*job],cwd=ROOT,stdout=log,stderr=log,timeout=600)
        if result.returncode:
            print(path.read_text(encoding='utf-8')[-6000:],flush=True);raise RuntimeError(f'Worker failed: {job}')
        print('passed',job,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('control','benchmark','host','device'),required=True)
    parser.add_argument('--label',default='benchmark');parser.add_argument('--batch',type=int,default=1);parser.add_argument('--prefix',type=int,default=32)
    parser.add_argument('--mode',choices=('fp16','calibrated'),default='calibrated');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.case=='control':control()
    elif args.case=='device':device(args.batch,args.prefix,args.mode)
    else:benchmark(args.label,args.case=='host')


if __name__=='__main__':main()

"""Current 30-layer FP16 decoder baseline and isolated host/task profiling."""
import argparse
import ctypes as ct
import gc
import json
import os
from pathlib import Path
import shutil
import statistics
import numpy as np
from w8_full_calibration import SOURCES as FROZEN_SOURCES
from w8_quality_probe import ROOT,MODEL,sha

OUT=ROOT/'npunlock-probe/research/results/shave-fp32-pipeline'
SOURCES=list(dict.fromkeys(FROZEN_SOURCES+['torch_npu_device_profile.py','torch_npu_warm_profile.py',
    'npunlock-probe/research/rmsnorm_reduce_probe.py','npunlock-probe/research/shave_fp32_pipeline.py']))
SHAPES=((1,32),(4,16))


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def metadata():return dict(sources={name:sha(ROOT/name) for name in SOURCES},checkpoint_sha256=sha(MODEL/'model.safetensors'),
    decoder_layers=30,mode='fp16',scope='Resident 30-layer decoder; CPU embedding/final norm/lm_head excluded')


def profiling_directory(path):
    from w8_performance import profile_binary
    original,build=profile_binary();path.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(original/'session_task_profile.dll',path/'session_task_profile.dll')
    assert sha(path/'session_task_profile.dll')==build['dll_sha256']
    return build


def snapshots(result):
    output,caches=result
    return [output,*[value for cache in caches for value in (cache.keys,cache.values,cache._position)]]


def final(npu,report):
    gc.collect();npu.shutdown();report['final_stats']=npu.stats();stats=report['final_stats']
    assert stats['closed'] and not stats['context_failed']
    assert all(stats[name]==0 for name in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))


def benchmark(label,host_profile=False):
    import torch
    driver=None
    if host_profile:
        import torch_npu_device_profile as profile
        profile.OUT=OUT/'host-runtime';build=profiling_directory(profile.OUT);npu,driver=profile.setup()
    else:import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    from transformers import AutoTokenizer
    from safetensors import safe_open
    from rmsnorm_reduce_probe import times
    from torch_npu_warm_profile import STAGES
    torch.set_num_threads(8)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as file:embedding=file.get_tensor('model.embed_tokens.weight')
    report=dict(status='running',label=label,host_instrumented=host_profile,rows=[],**metadata());saved={}
    if host_profile:report['profiling_build']=build
    def work(batch,prefix):
        tag=f'b{batch}-s{prefix}';tokens=ids.repeat(1,(prefix+ids.shape[1])//ids.shape[1])[:,:prefix+1].repeat(batch,1)
        host=embedding[tokens].half().contiguous();x=host[:,:prefix].contiguous().to('intel_npu');one=host[:,prefix:].contiguous().to('intel_npu')
        stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=30,mode='fp16')
        print(label,tag,'preparing 30-layer prefill/decode graphs',flush=True)
        initial=stack.prefill(x);seed=initial[1];decoded=stack.decode(one,seed)
        anchors=[v.cpu().numpy().copy() for result in (initial,decoded) for v in snapshots(result)]
        for i,array in enumerate(anchors):saved[f'{tag}/{i}']=array
        del initial,decoded
        for operation in ('prefill','decode'):
            fn=(lambda:stack.prefill(x)) if operation=='prefill' else (lambda:stack.decode(one,seed));host_rows=[]
            def measured():
                driver.warm_diag_reset();result=fn();ns=(ct.c_uint64*9)();counts=(ct.c_uint64*9)();driver.warm_diag_read(ns,counts)
                host_rows.append(dict(times_ms=dict(zip(STAGES,[v/1e6 for v in ns])),counts=dict(zip(STAGES,list(counts)))))
                return result
            timing=times({'baseline':measured if driver else fn},31,npu)
            assert timing['warm_delta']['npu_calls']==62*30
            row=dict(batch=batch,prefix=prefix,operation=operation,**timing)
            if driver:
                row['host_samples']=host_rows[3:]
                submits=8 if operation=='decode' and batch in (1,4,8) else 30
                assert len(row['host_samples'])==62 and all(r['counts']['append_graph']==30 for r in host_rows)
                assert all(r['counts'][stage]==submits for r in host_rows for stage in STAGES[4:])
                row['median_host_stage_ms']={stage:statistics.median(r['times_ms'][stage] for r in host_rows[3:]) for stage in STAGES}
            report['rows'].append(row);print(label,tag,operation,timing['median_ms'],flush=True)
        repeated=stack.prefill(x);decoded=stack.decode(one,seed)
        values=[v for result in (repeated,decoded) for v in snapshots(result)];assert len(values)==len(anchors)==182
        for actual,expected in zip(values,anchors):np.testing.assert_array_equal(actual.cpu().numpy().view(np.uint8),expected.view(np.uint8))
        for i,cache in enumerate(seed):
            for j,value in enumerate((cache.keys,cache.values,cache._position)):
                np.testing.assert_array_equal(value.cpu().numpy().view(np.uint8),anchors[1+i*3+j].view(np.uint8))
        report.setdefault('checks',[]).append(dict(tag=tag,repeated_bit_equal_arrays=182,immutable_seed_arrays=90))
    with torch.no_grad():
        for batch,prefix in SHAPES:work(batch,prefix)
        final(npu,report)
    path=OUT/(label+'-outputs.npz');np.savez_compressed(path,**saved)
    report.update(status='passed',outputs_sha256=sha(path));(OUT/(label+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')


def device(batch,prefix):
    import torch_npu_device_profile as profile
    from torch_npu_smollm import FrozenSmolStack
    path=OUT/f'device-b{batch}-s{prefix}';build=profiling_directory(path);profile.OUT=path
    original=FrozenSmolStack.from_checkpoint.__func__
    def construct(cls,directory,**options):options['num_layers']=30;return original(cls,directory,**options)
    FrozenSmolStack.from_checkpoint=classmethod(construct)
    try:profile.execute(batch,prefix,'fp16')
    finally:FrozenSmolStack.from_checkpoint=classmethod(original)
    report_path=path/f'b{batch}-s{prefix}-fp16.json';report=read(report_path)
    assert report['status']=='passed' and len(report['graphs'])==60
    assert all(sorted(g['layer'] for g in report['graphs'] if g['operation']==op)==list(range(30)) for op in ('prefill','decode'))
    report.update(experiment=metadata(),profiling_build=build)
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('benchmark','host','device'),required=True)
    parser.add_argument('--label',default='baseline');parser.add_argument('--batch',type=int,default=1);parser.add_argument('--prefix',type=int,default=32);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.case=='device':device(args.batch,args.prefix)
    else:benchmark(args.label,args.case=='host')


if __name__=='__main__':main()

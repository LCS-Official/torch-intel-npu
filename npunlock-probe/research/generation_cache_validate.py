"""Actual production-PYD generation and existing allocator/device regressions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from generation_cache_probe import OUT, read, write
from w8_quality_probe import ROOT, sha


def production(mode):
    import torch_intel_npu as npu
    assert npu.stats()['allocation_cache_limit_blocks']==256
    assert not hasattr(npu._C,'_diagnostic_capacity')
    source=ROOT/'npunlock-probe/research/smollm_generation_profile.py';text=source.read_text()
    replacements={
        '            actual_group=None if baseline_prepare else group':'            actual_group=None',
        "    report['comparison']='output preparation, both group4' if baseline_prepare else 'group1 vs default group4'":
            "    report['comparison']='actual fixed 256-block production extension, both default group4'",
        'rounds=7,samples=[]':'rounds=2,samples=[]',
        '        for index in range(7):':'        for index in range(2):',
        '        for step in (0,15,31):':'        for step in range(32):',
        "'/ 7 wins'":"'/ 2 confirmation rounds'"}
    for old,new in replacements.items():assert text.count(old)==1,old;text=text.replace(old,new)
    directory=OUT/'production';directory.mkdir(exist_ok=True)
    runner=directory/'runner.py';runner.write_text(text,encoding='utf-8')
    scope=dict(__name__='cache_production_runner',__file__=str(runner))
    exec(compile(text,str(runner),'exec'),scope);scope['main'](mode,output=directory)
    path=directory/(mode+'.json');report=read(path)
    assert len(report['samples'])==8 and all(s['delta']['driver_allocations']==0 for s in report['samples'])
    reference=OUT/('trace-fp16-outputs.npz' if mode=='fp16' else 'benchmark-w8a16-outputs.npz')
    with np.load(reference) as baseline,np.load(directory/(mode+'-outputs.npz')) as actual:
        np.testing.assert_array_equal(actual['tokens'],baseline['tokens'])
        for step in range(32):np.testing.assert_array_equal(actual[f'logits-{step}'].view(np.uint8),baseline['logits'][step].view(np.uint8))
    report.update(wrapper_sha256=sha(Path(__file__)),template_sha256=sha(source),
        all_32_logits_vs_isolated_baseline_bit_equal=True,reference_outputs_sha256=sha(reference))
    write(path,report);print(mode,'actual production: zero warm allocations; all 32 logits match isolated baseline',flush=True)


def check(kind):
    directory=OUT/'checks';directory.mkdir(exist_ok=True)
    if kind=='guards':
        import torch_npu_warm_profile as common
        common.OUT=directory;common.guards()
    elif kind=='order':
        import decode_cache_checks as common
        common.OUT=directory;common.order()
    elif kind=='failure':
        import torch_npu_warm_profile as warm
        import allocator_failure_probe as failure
        warm.OUT=directory;warm.build();failure.OUT=directory;failure.build();failure.run()
    else:
        import importlib
        common=importlib.import_module('torch_npu_device_demo' if kind=='device' else 'torch_npu_training_demo')
        common.OUT=directory
        if kind=='device':common.DESTINATION=directory/'device.json';assert common.execute(True)==0
        else:common.REPORT=directory/'training.json';assert common.execute(True,False)==0


def regressions():
    directory=OUT/'checks';directory.mkdir(exist_ok=True)
    for kind in ('guards','order','failure','device','training'):
        with (directory/(kind+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--case',kind],cwd=ROOT,stdout=log,stderr=log,timeout=360)
        if result.returncode:
            print((directory/(kind+'.log')).read_text(),flush=True);raise SystemExit(result.returncode)
        print(kind,'passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('production','regressions','guards','order','failure','device','training'),required=True)
    parser.add_argument('--mode',choices=('fp16','w8a16'),default='fp16');args=parser.parse_args()
    if args.case=='production':production(args.mode)
    elif args.case=='regressions':regressions()
    else:check(args.case)

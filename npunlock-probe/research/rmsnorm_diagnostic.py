"""Same-layout SHAVE ablation and allocation controls, isolated subprocess."""
import argparse
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-diagnostic'
NAMES=('fill','copy','convert','affine','epilogue','mean','inverse','full')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def execute():
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from npunlock.npu4000 import compile_rmsnorm, patch_binary_carrier
    from torch_npu_smollm import FrozenSmolLayer, _rmsnorm_kernel
    from torch_npu_runtime import SESSION_DLL
    torch.set_num_threads(8);torch.manual_seed(9173)
    sources=['torch_intel_npu.py','torch_npu_smollm.py','npunlock/python/npunlock/npu4000.py',
        'npunlock-probe/torch-device/backend.cpp','npunlock/src/graphinfer/session_win32.c',
        'npunlock-probe/research/build_rmsnorm_diagnostic.py','npunlock-probe/research/rmsnorm_diagnostic.py']
    report=dict(status='running',checks=[],timings=[],carrier_evidence={},sources_sha256={p:sha(ROOT/p) for p in sources},
        packages={name:importlib.metadata.version(name) for name in ('torch','numpy','safetensors')},
        extension_build=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig')),
        session_dll_sha256=sha(SESSION_DLL),rounds=31,warmups=5,cpu_threads=8)
    try:
        manifest=json.loads((OUT/'build-manifest.json').read_text());assert manifest['status']=='built'
        kernels={name:(OUT/(name+'.elf')).read_bytes() for name in NAMES}
        for name,data in kernels.items():assert hashlib.sha256(data).hexdigest()==manifest['kernels'][name]['elf_sha256']
        assert kernels['full']==_rmsnorm_kernel()
        report['build_manifest']=manifest
        def key(rows,name):return ('rmsnorm_diag',rows,name)
        def prepare(rows):
            if key(rows,'full') in npu._sessions:return
            full=compile_rmsnorm(rows,kernels['full'],libraries=npu._native)
            info=full.ir_provenance;original=info.graph_blob;layout=None
            evidence=[]
            for name in NAMES:
                blob,patch=patch_binary_carrier(original,kernels[name],vendor_id=info.vendor_id,
                    device_id=info.device_id,compiler_version=info.compiler_version,dtype='f16')
                begin=patch['code_offset'];end=begin+patch['window_bytes']
                normalized=blob[:begin]+bytes(patch['window_bytes'])+blob[end:]
                digest=hashlib.sha256(normalized).hexdigest()
                if layout is None:layout=digest
                assert layout==digest,'Ablation changed bytes outside the shared code window'
                patch['diagnostic_operation']=name;patch['shape']=[rows,576];patch['layout_sha256']=digest
                program=replace(full,graph_blob=blob,patch_reports=(json.dumps(patch).encode(),))
                npu._sessions[key(rows,name)]=npu._native.create_inference_session(blob,parent=npu._root_session),program
                npu._session_creations+=1;evidence.append(patch)
                (OUT/f'carrier-{rows}-{name}.blob').write_bytes(blob)
            report['carrier_evidence'][str(rows)]=evidence
        def run(rows,name,x,gamma,output=None):return npu._run(key(rows,name),[x,gamma],(rows,576),torch.float16,out=output)
        def references(host,weight):
            value=host.float();variance=value.square().mean(-1,keepdim=True)+1e-5
            inverse=torch.rsqrt(variance)
            return dict(fill=torch.zeros_like(host),copy=host,convert=host,
                affine=host*weight,epilogue=host*weight,
                mean=variance.half().expand_as(host),inverse=inverse.half().expand_as(host),
                full=(value*inverse).half()*weight)
        def check(tag,host,weight):
            rows=host.shape[0];prepare(rows);refs=references(host,weight)
            x=host.to('intel_npu');gamma=weight.repeat(rows,1).to('intel_npu')
            for name in NAMES:
                output=torch.full_like(host,7.).to('intel_npu')
                actual=run(rows,name,x,gamma,output).cpu();expected=refs[name]
                if name in ('fill','copy','convert','affine','epilogue'):
                    assert torch.equal(actual.view(torch.int16),expected.contiguous().view(torch.int16)),(tag,name,'bits')
                else:torch.testing.assert_close(actual,expected,rtol=.002,atol=.001)
                report['checks'].append(dict(tag=tag,kernel=name,rows=rows,status='passed',
                    bitwise_equal=bool(torch.equal(actual.view(torch.int16),expected.contiguous().view(torch.int16)))))
                del output
            del x,gamma
        with torch.no_grad():
            model=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16')
            weights={name:model._cpu_weight(name) for name in ('input_layernorm.weight','post_attention_layernorm.weight')}
            for rows in (1,7,8,9,17,32,256,259,512):
                host=torch.randn(rows,576).half()
                for name,weight in weights.items():check('random-'+name,host,weight)
            bits=np.resize(np.array([0,1,3,1023,0x8000,0x8001,0x8003,0x83ff],dtype=np.uint16),576)
            special=[('zero',torch.zeros(3,576).half()),
                ('subnormal',torch.from_numpy(bits.view(np.float16).copy()).view(1,576)),
                ('large',torch.full((3,576),65504.).half())]
            special[-1][1][:,1::2]*=-1
            for tag,host in special:
                for name,weight in weights.items():check(tag+'-'+name,host,weight)
            real_file=ROOT/'npunlock-probe/research/results/rmsnorm-vector/performance.json'
            sample_file=ROOT/'npunlock-probe/research/results/smollm/perf-b4-s64-w8a16.npz'
            real=torch.from_numpy(np.load(sample_file)['host'].copy()).reshape(256,576)
            assert real.dtype==torch.float16 and torch.isfinite(real).all()
            report['real_input']=dict(path=str(sample_file),sha256=sha(sample_file),
                prior_performance_sha256=sha(real_file),rms=float(real.float().square().mean().sqrt()))
            weight=weights['input_layernorm.weight']
            for rows in (1,32,256,512):
                for dataset in ('unit_random','real_embedding'):
                    host=torch.randn(rows,576).half() if dataset=='unit_random' else real.repeat((rows+255)//256,1)[:rows].contiguous()
                    check(dataset,host,weight)
                    x=host.to('intel_npu');gamma=weight.repeat(rows,1).to('intel_npu')
                    outputs={name:torch.empty_like(host,device='intel_npu') for name in NAMES}
                    calls=[(name,allocation) for name in NAMES for allocation in ('fresh','reuse')]
                    for _ in range(5):
                        for name,allocation in calls:run(rows,name,x,gamma,outputs[name] if allocation=='reuse' else None)
                    before=npu.stats();samples={name+'-'+allocation:[] for name,allocation in calls}
                    for index in range(31):
                        position=index%len(calls)
                        for name,allocation in calls[position:]+calls[:position]:
                            start=time.perf_counter();run(rows,name,x,gamma,outputs[name] if allocation=='reuse' else None)
                            samples[name+'-'+allocation].append((time.perf_counter()-start)*1000)
                    after=npu.stats()
                    for field in ('host_upload_copies','host_download_copies','host_device_copies','host_pack_copies','session_creations'):
                        assert before[field]==after[field],(dataset,rows,field)
                    assert after['npu_calls']-before['npu_calls']==31*len(calls)
                    assert after['buffer_allocations']-before['buffer_allocations']==31*len(NAMES)
                    entry=dict(rows=rows,dataset=dataset,input_rms=float(host.float().square().mean().sqrt()),
                        median_ms={name:statistics.median(v) for name,v in samples.items()},
                        p10_ms={name:sorted(v)[3] for name,v in samples.items()},p90_ms={name:sorted(v)[27] for name,v in samples.items()},
                        samples_ms=samples,warm_npu_calls=after['npu_calls']-before['npu_calls'],
                        warm_allocations=after['buffer_allocations']-before['buffer_allocations'],warm_host_copies_and_compiles=0)
                    report['timings'].append(entry);print({k:v for k,v in entry.items() if k not in ('samples_ms','p10_ms','p90_ms')},flush=True)
                    outputs.clear();del x,gamma
            del model,weights,weight;gc.collect();npu.shutdown();report['final_stats']=npu.stats()
            assert report['final_stats']['live_buffers']==report['final_stats']['cached_sessions']==0
            assert not report['final_stats']['context_failed'] and report['final_stats']['quarantined_buffer_bytes']==0
        assert all(sha(ROOT/p)==h for p,h in report['sources_sha256'].items()),'Source changed during diagnosis'
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats())
        print(report['traceback'],flush=True)
    (OUT/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return int(report['status']!='passed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.child:return execute()
    with (OUT/'diagnostic.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--child'],stdout=log,stderr=log,timeout=300)
    print((OUT/'diagnostic.log').read_text(encoding='utf-8'),flush=True)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())

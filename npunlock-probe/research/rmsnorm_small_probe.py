"""Same-carrier reduction/copy and fresh/reused output diagnostics."""
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-small'


def main():
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from npunlock.npu4000 import compile_rmsnorm,patch_binary_carrier
    from torch_npu_smollm import FrozenSmolLayer,_rmsnorm_kernel
    report=dict(status='running',checks=[],timings=[],carriers={},sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['npunlock-probe/research/rmsnorm_small_probe.py','torch_npu_smollm.py','torch_intel_npu.py','npunlock/python/npunlock/npu4000.py']})
    try:
        build=json.loads((OUT/'build-manifest.json').read_text());report['build_manifest']=build
        kernels=dict(baseline=(OUT/'before/rmsnorm-f16.elf').read_bytes(),**{name:(OUT/f'{name}.elf').read_bytes() for name in ['reduce4','copy']})
        assert _rmsnorm_kernel() in (kernels['baseline'],kernels['reduce4'])
        for name in ['reduce4','copy']:assert hashlib.sha256(kernels[name]).hexdigest()==build['kernels'][name]['elf_sha256']
        def key(rows,name):return ('rmsnorm_small_probe',rows,name)
        def prepare(rows):
            if key(rows,'baseline') in npu._sessions:return
            carrier=compile_rmsnorm(rows,kernels['baseline'],libraries=npu._native);info=carrier.ir_provenance;layout=None;records=[]
            for name,kernel in kernels.items():
                blob,evidence=patch_binary_carrier(info.graph_blob,kernel,vendor_id=info.vendor_id,device_id=info.device_id,compiler_version=info.compiler_version,dtype='f16')
                a=evidence['code_offset'];b=a+evidence['window_bytes'];digest=hashlib.sha256(blob[:a]+bytes(b-a)+blob[b:]).hexdigest()
                if layout is None:layout=digest
                assert digest==layout
                evidence.update(kernel=name,layout_sha256=digest)
                program=replace(carrier,graph_blob=blob,patch_reports=(json.dumps(evidence).encode(),))
                npu._sessions[key(rows,name)]=npu._native.create_inference_session(blob,parent=npu._root_session),program;npu._session_creations+=1
                (OUT/f'carrier-{rows}-{name}.blob').write_bytes(blob);records.append(evidence)
            report['carriers'][str(rows)]=records
        def run(rows,name,x,gamma,out=None):return npu._run(key(rows,name),[x,gamma],(rows,576),torch.float16,out)
        def check(tag,host,weight):
            rows=host.shape[0];x=host.to('intel_npu');gamma=weight.expand(rows,576).contiguous().to('intel_npu');prepare(rows)
            baseline=run(rows,'baseline',x,gamma).cpu();got=run(rows,'reduce4',x,gamma).cpu();copied=run(rows,'copy',x,gamma).cpu()
            bits=int((baseline.view(torch.int16)!=got.view(torch.int16)).sum());assert bits==0,(tag,bits)
            assert torch.equal(copied.view(torch.int16),host.view(torch.int16))
            value=host.float();reference=(value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-5)).half()*weight
            torch.testing.assert_close(got,reference,rtol=.002,atol=.001)
            report['checks'].append(dict(tag=tag,rows=rows,old_new_bit_differences=bits,copy_bits_equal=True))
        def work():
            torch.set_num_threads(8);torch.manual_seed(9271)
            model=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16')
            weights=[model._cpu_weight(name) for name in ['input_layernorm.weight','post_attention_layernorm.weight']]
            for rows in [1,2,3,4,7,8,9,17,32,64,256,512]:
                host=torch.randn(rows,576).half()
                for i,weight in enumerate(weights):check(f'random-gamma{i}',host,weight)
            codes=np.arange(65536,dtype=np.uint16);codes=codes[(codes&0x7c00)!=0x7c00];np.random.default_rng(9271).shuffle(codes)
            mixed=torch.from_numpy(np.resize(codes,((len(codes)+575)//576)*576).view(np.float16).copy()).reshape(-1,576)
            for i,weight in enumerate(weights):check(f'all-finite-mixed-gamma{i}',mixed,weight)
            tiny=torch.from_numpy(np.resize(np.array([0,1,3,1023,0x8000,0x8001,0x8003,0x83ff],dtype=np.uint16),576).view(np.float16).copy()).view(1,576)
            large=torch.full((3,576),65504.).half();large[:,1::2]*=-1
            for tag,host in [('zeros',torch.zeros(3,576).half()),('subnormal',tiny),('large',large)]:
                for i,weight in enumerate(weights):check(tag+f'-gamma{i}',host,weight)
            path=ROOT/'npunlock-probe/research/results/rmsnorm-diagnostic/real-input.npz'
            real=torch.from_numpy(np.load(path)['host'].copy()).reshape(-1,576);report['real_input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            for rows in [1,2,4,8,32,256,512]:
                host=real.repeat((rows+real.shape[0]-1)//real.shape[0],1)[:rows].contiguous();weight=weights[0];check('real_embedding',host,weight)
                x=host.to('intel_npu');gamma=weight.expand(rows,576).contiguous().to('intel_npu')
                outputs={name:torch.empty_like(x) for name in kernels};variants=[(name,allocation) for name in kernels for allocation in ['fresh','reuse']]
                for _ in range(5):
                    for name,allocation in variants:run(rows,name,x,gamma,outputs[name] if allocation=='reuse' else None)
                samples={name+'-'+allocation:[] for name,allocation in variants};native={name:[] for name in samples};original=npu._C.run;durations=[]
                def native_timer(*args,**kwargs):
                    start=time.perf_counter()
                    try:return original(*args,**kwargs)
                    finally:durations.append((time.perf_counter()-start)*1000)
                before=npu.stats()
                with patch.object(npu._C,'run',native_timer):
                    for index in range(41):
                        order=variants[index%len(variants):]+variants[:index%len(variants)]
                        for name,allocation in order:
                            label=name+'-'+allocation;durations.clear();start=time.perf_counter()
                            run(rows,name,x,gamma,outputs[name] if allocation=='reuse' else None)
                            samples[label].append((time.perf_counter()-start)*1000);assert len(durations)==1;native[label].append(durations[0])
                after=npu.stats()
                for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                assert after['npu_calls']-before['npu_calls']==41*len(variants)
                assert after['buffer_allocations']-before['buffer_allocations']==41*len(kernels)
                entry=dict(rows=rows,rounds=41,median_ms={name:statistics.median(values) for name,values in samples.items()},
                    median_native_ms={name:statistics.median(values) for name,values in native.items()},samples_ms=samples,native_samples_ms=native,
                    fresh_allocations=after['buffer_allocations']-before['buffer_allocations'],warm_host_copies_and_compiles=0)
                report['timings'].append(entry);print({k:v for k,v in entry.items() if k not in ['samples_ms','native_samples_ms']},flush=True)
        with torch.no_grad():work()
        gc.collect();assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown();report.update(status='passed',final_stats=npu.stats());assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

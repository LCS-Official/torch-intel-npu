"""Compare isolated output loops on identical carrier bytes and old NPU bits."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-output'
NAMES=('scalar','unroll4','bits4','codec4','affine4')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(label,rounds):
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from npunlock.npu4000 import compile_rmsnorm,patch_binary_carrier
    from torch_npu_smollm import FrozenSmolLayer,_rmsnorm_kernel
    torch.set_num_threads(8);torch.manual_seed(9271)
    sources=['torch_intel_npu.py','torch_npu_smollm.py','npunlock/python/npunlock/npu4000.py',
        'npunlock/tests/fixtures/npu4000/rmsnorm.c','npunlock-probe/research/build_rmsnorm_output.py',
        'npunlock-probe/research/rmsnorm_output_probe.py']
    report=dict(status='running',checks=[],timings=[],carrier_evidence={},rounds=rounds,warmups=5,
        sources_sha256={p:sha(ROOT/p) for p in sources})
    try:
        manifest=json.loads((OUT/'build-manifest.json').read_text())
        kernels={name:(OUT/(name+'.elf')).read_bytes() for name in NAMES}
        assert manifest['status']=='built' and _rmsnorm_kernel() in (kernels['scalar'],kernels['bits4'])
        for name,data in kernels.items():assert hashlib.sha256(data).hexdigest()==manifest['kernels'][name]['elf_sha256']
        report['build_manifest']=manifest
        def key(rows,name):return ('rmsnorm_output_probe',rows,name)
        def prepare(rows):
            if key(rows,'scalar') in npu._sessions:return
            program=compile_rmsnorm(rows,kernels['scalar'],libraries=npu._native)
            info=program.ir_provenance;records=[];layout=None
            for name in NAMES:
                blob,patch=patch_binary_carrier(info.graph_blob,kernels[name],vendor_id=info.vendor_id,
                    device_id=info.device_id,compiler_version=info.compiler_version,dtype='f16')
                a=patch['code_offset'];b=a+patch['window_bytes']
                digest=hashlib.sha256(blob[:a]+bytes(b-a)+blob[b:]).hexdigest()
                if layout is None:layout=digest
                assert digest==layout
                patch.update(operation=name,layout_sha256=digest)
                prepared=replace(program,graph_blob=blob,patch_reports=(json.dumps(patch).encode(),))
                npu._sessions[key(rows,name)]=npu._native.create_inference_session(blob,parent=npu._root_session),prepared
                npu._session_creations+=1;records.append(patch)
                (OUT/f'carrier-{rows}-{name}.blob').write_bytes(blob)
            report['carrier_evidence'][str(rows)]=records
        def run(rows,name,x,gamma,output=None):return npu._run(key(rows,name),[x,gamma],(rows,576),torch.float16,out=output)
        def check(tag,host,weight,names=('scalar','unroll4','bits4')):
            rows=host.shape[0];prepare(rows)
            x=host.to('intel_npu');gamma=weight.expand(rows,576).contiguous().to('intel_npu')
            value=host.float();reference=(value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-5)).half()*weight
            old=None
            for name in names:
                output=torch.full_like(host,7.).to('intel_npu')
                got=run(rows,name,x,gamma,output).cpu()
                expected=host if name=='codec4' else (host.float()*weight.float()).half() if name=='affine4' else reference
                if name=='scalar':old=got
                baseline=expected if name in ('codec4','affine4') else old
                differences=int((got.view(torch.int16)!=baseline.contiguous().view(torch.int16)).sum())
                entry=dict(tag=tag,kernel=name,rows=rows,bit_differences=differences,
                    first_output=got.reshape(-1)[:8].tolist(),first_expected=baseline.reshape(-1)[:8].tolist())
                report['checks'].append(entry)
                assert differences==0,entry
                torch.testing.assert_close(got,expected,rtol=.002,atol=.001)
                del output
            del x,gamma
        with torch.no_grad():
            model=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16')
            weights={name:model._cpu_weight(name) for name in ('input_layernorm.weight','post_attention_layernorm.weight')}
            # Every finite binary16 encoding, including both zeros and all
            # subnormals. Repeating to a complete row tests every vector lane.
            codes=np.arange(65536,dtype=np.uint16)
            codes=codes[(codes&0x7c00)!=0x7c00]
            count=(len(codes)+575)//576
            host=torch.from_numpy(np.resize(codes,count*576).view(np.float16).copy()).reshape(count,576)
            check('all-finite-half-roundtrip',host,torch.ones(576).half(),('codec4',))
            if label=='confirm':
                shuffled=codes.copy();np.random.default_rng(9271).shuffle(shuffled)
                mixed=torch.from_numpy(np.resize(shuffled,count*576).view(np.float16).copy()).reshape(count,576)
                check('all-finite-half-shuffled',mixed,torch.ones(576).half(),('codec4',))
                for scale in (.5,1.5,2**-14):
                    check('all-finite-half-affine-'+str(scale),mixed,torch.full((576,),scale).half(),('affine4',))
                check('all-finite-half-real-gamma',mixed,weights['input_layernorm.weight'],('affine4',))
                for name,weight in weights.items():check('mixed-half-exponents-'+name,mixed[:9],weight)
            pairs=[(2**-14,.5),(2**-14,1-2**-11),(2**-24,.5),(3*2**-24,.5),
                   (1+2**-10,1.5),(65504,1+2**-10),(65504,2),(-0.,1),
                   (-2**-14,.5),(-3*2**-24,.5),(-65504,2),(1,-0.)]
            host=torch.tensor([x for x,w in pairs]).half().repeat(48).reshape(1,576)
            weight=torch.tensor([w for x,w in pairs]).half().repeat(48)
            check('rounding-underflow-overflow-ties',host,weight,('affine4',))
            for rows in (1,3,7,8,9,17,32,64,256,259,512):
                host=torch.randn(rows,576).half()
                for name,weight in weights.items():check('random-'+name,host,weight)
            bits=np.resize(np.array([0,1,3,1023,0x8000,0x8001,0x8003,0x83ff],dtype=np.uint16),576)
            special=[('zeros',torch.zeros(3,576).half()),
                ('subnormal',torch.from_numpy(bits.view(np.float16).copy()).view(1,576)),
                ('large',torch.full((3,576),65504.).half())]
            special[-1][1][:,1::2]*=-1
            for tag,host in special:
                for name,weight in weights.items():check(tag+'-'+name,host,weight)
            real_path=ROOT/'npunlock-probe/research/results/rmsnorm-diagnostic/real-input.npz'
            real=torch.from_numpy(np.load(real_path)['host'].copy()).reshape(256,576)
            report['real_input']=dict(path=str(real_path),sha256=sha(real_path))
            weight=weights['input_layernorm.weight']
            for rows in (1,32,256,512):
                for dataset in ('unit_random','real_embedding'):
                    host=torch.randn(rows,576).half() if dataset=='unit_random' else real.repeat((rows+255)//256,1)[:rows].contiguous()
                    check(dataset,host,weight)
                    x=host.to('intel_npu');gamma=weight.repeat(rows,1).to('intel_npu')
                    names=('scalar','unroll4','bits4');outputs={name:torch.empty_like(host,device='intel_npu') for name in names}
                    calls=[(name,allocation) for name in names for allocation in ('fresh','reuse')]
                    for _ in range(5):
                        for name,allocation in calls:run(rows,name,x,gamma,outputs[name] if allocation=='reuse' else None)
                    before=npu.stats();samples={name+'-'+allocation:[] for name,allocation in calls}
                    for index in range(rounds):
                        position=index%len(calls)
                        for name,allocation in calls[position:]+calls[:position]:
                            start=time.perf_counter();run(rows,name,x,gamma,outputs[name] if allocation=='reuse' else None)
                            samples[name+'-'+allocation].append((time.perf_counter()-start)*1000)
                    after=npu.stats()
                    for field in ('host_upload_copies','host_download_copies','host_device_copies','host_pack_copies','session_creations'):
                        assert before[field]==after[field],field
                    assert after['npu_calls']-before['npu_calls']==rounds*len(calls)
                    assert after['buffer_allocations']-before['buffer_allocations']==rounds*len(names)
                    entry=dict(rows=rows,dataset=dataset,median_ms={k:statistics.median(v) for k,v in samples.items()},
                        p10_ms={k:sorted(v)[len(v)//10] for k,v in samples.items()},
                        p90_ms={k:sorted(v)[len(v)*9//10] for k,v in samples.items()},samples_ms=samples,
                        timed_npu_calls=after['npu_calls']-before['npu_calls'],host_copies_and_compiles=0)
                    report['timings'].append(entry);print({k:v for k,v in entry.items() if k!='samples_ms'},flush=True)
                    outputs.clear();del x,gamma
            if label=='confirm':
                import torch_npu_smollm as smol
                class VariantLayer(FrozenSmolLayer):
                    def _norm_device(self,x,name,**kwargs):
                        original=smol._rmsnorm_kernel
                        smol._rmsnorm_kernel=lambda:kernels[self.kernel_name]
                        try:return super()._norm_device(x,name,**kwargs)
                        finally:smol._rmsnorm_kernel=original
                report['layer_timings']=[]
                for batch,length in ((1,1),(1,32),(4,64)):
                    host=real[:batch*length].reshape(batch,length,576)
                    x=host.to('intel_npu')
                    for mode in ('fp16','w8a16'):
                        models={name:VariantLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode)
                            for name in ('scalar','bits4')}
                        for name,layer in models.items():layer.kernel_name=name
                        old=models['scalar'](x).cpu();new=models['bits4'](x).cpu()
                        assert torch.equal(old.view(torch.int16),new.view(torch.int16)),(mode,batch,length,'layer bits')
                        for _ in range(3):
                            for layer in models.values():layer(x)
                        before=npu.stats();samples={name:[] for name in models}
                        for index in range(21):
                            for name in (('scalar','bits4') if index%2==0 else ('bits4','scalar')):
                                start=time.perf_counter();models[name](x);samples[name].append((time.perf_counter()-start)*1000)
                        after=npu.stats()
                        for field in ('host_upload_copies','host_download_copies','host_device_copies','host_pack_copies','session_creations'):
                            assert before[field]==after[field],field
                        assert after['npu_calls']-before['npu_calls']==21*2*4
                        entry=dict(shape=list(host.shape),mode=mode,old_npu_bitwise_equal=True,
                            median_ms={name:statistics.median(v) for name,v in samples.items()},samples_ms=samples,
                            warm_graph_calls_per_layer=4,host_copies_and_compiles=0)
                        report['layer_timings'].append(entry);print({k:v for k,v in entry.items() if k!='samples_ms'},flush=True)
                        models.clear();del layer
                    del x
            del model,weights,weight;gc.collect();npu.shutdown();report['final_stats']=npu.stats()
            assert report['final_stats']['live_buffers']==report['final_stats']['cached_sessions']==0
            assert not report['final_stats']['context_failed'] and report['final_stats']['quarantined_buffer_bytes']==0
        assert all(sha(ROOT/p)==h for p,h in report['sources_sha256'].items())
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats())
        print(report['traceback'],flush=True)
    (OUT/(label+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    return int(report['status']!='passed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true')
    p.add_argument('--label',default='probe');p.add_argument('--rounds',type=int,default=31);args=p.parse_args()
    assert args.label in ('probe','confirm') and args.rounds>=11
    OUT.mkdir(exist_ok=True)
    if args.child:return execute(args.label,args.rounds)
    log=OUT/(args.label+'.log')
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,__file__,'--child','--label',args.label,'--rounds',str(args.rounds)],
            stdout=stream,stderr=stream,timeout=300)
    print(log.read_text(encoding='utf-8'),flush=True)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())

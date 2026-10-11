"""Same-carrier RMSNorm controls and same-model four-layer controls."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-reduce'
NAMES=('baseline','codec')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def kernels():
    manifest=json.loads((OUT/'build.json').read_text(encoding='utf-8'));assert manifest['status']=='built'
    values={name:(OUT/f'{name}.elf').read_bytes() for name in manifest['kernels']}
    for name,data in values.items():assert hashlib.sha256(data).hexdigest()==manifest['kernels'][name]['elf_sha256']
    return manifest,values


def times(run,rounds,npu):
    for _ in range(3):
        for fn in run.values():value=fn();del value
    samples={name:[] for name in run};pairs=[];names=list(run)
    before=npu.stats()
    for i in range(rounds):
        rotated=names[i%len(names):]+names[:i%len(names)];order=rotated+list(reversed(rotated))
        row={name:[] for name in names}
        for name in order:
            start=time.perf_counter_ns();value=run[name]();elapsed=(time.perf_counter_ns()-start)/1e6
            del value;samples[name].append(elapsed);row[name].append(elapsed)
        pairs.append({name:statistics.mean(values) for name,values in row.items()})
    after=npu.stats()
    for key in ('host_upload_copies','host_download_copies','host_device_copies','host_pack_copies','session_creations'):
        assert after[key]==before[key],key
    return dict(samples_ms=samples,pairs_ms=pairs,median_ms={name:statistics.median(v) for name,v in samples.items()},
        positive_rounds={name:sum(p[name]<p['baseline'] for p in pairs) for name in names if name!='baseline'},
        warm_delta={key:after[key]-before[key] for key in ('npu_calls','buffer_allocations','driver_allocations','allocation_reuses')})


def numeric(npu,torch,report,rounds):
    import numpy as np
    from npunlock.npu4000 import compile_rmsnorm,patch_binary_carrier
    from torch_npu_smollm import FrozenSmolLayer
    manifest,elf=kernels();report['build']=manifest
    report.update(checks=[],timings=[],carrier_evidence={})
    def key(rows,name):return ('rmsnorm_reduce_probe',rows,name)
    def prepare(rows):
        if key(rows,'baseline') in npu._sessions:return
        original=compile_rmsnorm(rows,elf['baseline'],libraries=npu._native)
        info=original.ir_provenance;records=[];layout=None
        for name in (*NAMES,'roundtrip'):
            blob,patch=patch_binary_carrier(info.graph_blob,elf[name],vendor_id=info.vendor_id,
                device_id=info.device_id,compiler_version=info.compiler_version,dtype='f16')
            a=patch['code_offset'];b=a+patch['window_bytes'];digest=hashlib.sha256(blob[:a]+bytes(b-a)+blob[b:]).hexdigest()
            if layout is None:layout=digest
            assert digest==layout
            patch.update(kernel=name,layout_sha256=digest)
            program=replace(original,graph_blob=blob,patch_reports=(json.dumps(patch).encode(),))
            npu._sessions[key(rows,name)]=npu._native.create_inference_session(blob,parent=npu._root_session),program
            npu._session_creations+=1;records.append(patch);(OUT/f'norm-{rows}-{name}.blob').write_bytes(blob)
        report['carrier_evidence'][str(rows)]=records
    def check(tag,host,weight):
        rows=host.shape[0];prepare(rows);x=host.to('intel_npu');gamma=weight.expand(rows,576).contiguous().to('intel_npu')
        baseline=None
        for name in NAMES:
            got=npu._run(key(rows,name),[x,gamma],(rows,576),torch.float16).cpu()
            if name=='baseline':baseline=got
            equal=(got.view(torch.int16)==baseline.view(torch.int16))|(got.isnan()&baseline.isnan())
            different=int((~equal).sum());max_error=float(torch.nan_to_num(got.float()-baseline.float()).abs().max())
            report['checks'].append(dict(tag=tag,kernel=name,rows=rows,bit_differences=different,max_abs_error=max_error))
            assert different==0,(tag,name,different,max_error)
            if torch.isfinite(host).all() and torch.isfinite(weight).all():
                value=host.float();reference=(value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-5)).half()*weight
                torch.testing.assert_close(got,reference,rtol=.002,atol=.001)
        del x,gamma
    def work():
        model=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16')
        weights=[model._cpu_weight(name) for name in ('input_layernorm.weight','post_attention_layernorm.weight')]
        for rows in (1,2,3,7,8,9,16,17,31,32,64,128,129,255,256,257,259,511,512):
            host=torch.randn(rows,576).half()
            for i,weight in enumerate(weights):check(f'random-gamma{i}',host,weight)
        codes=np.arange(65536,dtype=np.uint16);finite=codes[(codes&0x7c00)!=0x7c00];np.random.default_rng(92311).shuffle(finite)
        # At 512 rows all eight SHAVE shards enter the wide reduction path.
        host=torch.from_numpy(np.resize(finite,512*576).view(np.float16).copy()).reshape(512,576)
        for i,weight in enumerate(weights):check(f'all-finite-half-gamma{i}',host,weight)
        for tag,codes in (('all-finite',finite),('all-encodings',np.arange(65536,dtype=np.uint16))):
            count=(len(codes)+575)//576;prepare(count)
            values=torch.from_numpy(np.resize(codes,count*576).view(np.float16).copy()).reshape(count,576)
            x=values.to('intel_npu');gamma=torch.ones_like(values).to('intel_npu')
            got=npu._run(key(count,'roundtrip'),[x,gamma],tuple(values.shape),torch.float16).cpu()
            same=(got.view(torch.int16)==values.view(torch.int16))|(got.isnan()&values.isnan())
            assert same.all(),tag
            report['checks'].append(dict(tag=tag,kernel='roundtrip',rows=count,bit_differences=0,nan_policy='NaN-equivalent; signed zero and all finite encodings bit-equal'))
            del x,gamma
        special=[('zero',torch.zeros(3,576).half()),('subnormal',torch.from_numpy(np.resize(np.array([0,1,3,1023,0x8000,0x8001,0x8003,0x83ff],dtype=np.uint16),576).view(np.float16).copy()).reshape(1,576)),
            ('large',torch.full((3,576),65504.).half()),('mixed-exponents',host[:9]),
            ('inf',torch.zeros(3,576).half()),('nan',torch.zeros(3,576).half())]
        special[2][1][:,1::2]*=-1;special[-2][1][:,17]=float('inf');special[-1][1][:,23]=float('nan')
        for tag,host in special:
            for i,weight in enumerate(weights):check(f'{tag}-gamma{i}',host,weight)
        # Many independently generated rows expose half rounding boundaries.
        for i in range(8):
            host=torch.randn(512,576).half()*2**((i-4)*3)
            check(f'random-rounding-stress-{i}',host,weights[i%2])
        path=ROOT/'npunlock-probe/research/results/rmsnorm-diagnostic/real-input.npz'
        real=torch.from_numpy(np.load(path)['host'].copy()).reshape(256,576);report['real_input_sha256']=sha(path)
        for rows in (1,4,32,64,128,256,512):
            host=real.repeat((rows+255)//256,1)[:rows].contiguous();weight=weights[0]
            check('real-embedding',host,weight);x=host.to('intel_npu');gamma=weight.expand(rows,576).contiguous().to('intel_npu')
            outputs={name:torch.empty_like(host,device='intel_npu') for name in NAMES}
            run={name:(lambda name=name:npu._run(key(rows,name),[x,gamma],(rows,576),torch.float16,out=outputs[name])) for name in NAMES}
            record=dict(rows=rows,**times(run,rounds,npu));report['timings'].append(record)
            print(rows,record['median_ms'],record['positive_rounds'],flush=True)
            run.clear();outputs.clear();del x,gamma
    work()


def model(npu,torch,report,rounds):
    import numpy as np
    from safetensors import safe_open
    from transformers import AutoTokenizer
    import torch_npu_smollm as smol
    from npunlock.npu4000 import _patch_carrier
    manifest,elf=kernels();report.update(build=manifest,checks=[],timings=[],carrier_evidence=[])
    # Preserve all existing model/precision behavior; vary only the Power code.
    directory=ROOT/'models/SmolLM2-135M-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as file:embedding=file.get_tensor('model.embed_tokens.weight')
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    def work():
        for batch,prefix,mode in ((1,32,'fp16'),(1,32,'w8a16'),(4,16,'fp16'),(4,16,'w8a16')):
            tag=f'b{batch}-s{prefix}-{mode}';tokens=ids.repeat(1,(prefix+ids.shape[1])//ids.shape[1])[:,:prefix+1].repeat(batch,1)
            host=embedding[tokens].half().contiguous();x=host[:,:prefix].contiguous().to('intel_npu');one=host[:,prefix:].contiguous().to('intel_npu')
            stack=smol.FrozenSmolStack.from_checkpoint(directory,num_layers=4,mode=mode)
            output,seed=stack.prefill(x);decoded,updated=stack.decode(one,seed)
            def arrays(value,cache):return [value.cpu().numpy().copy(),*[t.cpu().numpy().copy() for c in cache for t in (c.keys,c.values,c._position)]]
            anchors=arrays(output,seed)+arrays(decoded,updated);del output,decoded,updated
            strategies={name:[{} for _ in stack.layers] for name in NAMES}
            for index,layer in enumerate(stack.layers):
                strategies['baseline'][index]=dict(layer._keys)
                for local,key in layer._keys.items():
                    session,program=npu._sessions[key];info=program.ir_provenance
                    blob,patch=_patch_carrier(info.graph_blob,elf['codec'],vendor_id=info.vendor_id,device_id=info.device_id,
                        compiler_version=info.compiler_version,dtype='f16',carrier='Power',input_count=2,select_kernel=True)
                    old=json.loads(program.patch_reports[0]);a=old['code_offset'];b=a+old['window_bytes']
                    assert blob[:a]+blob[b:]==program.graph_blob[:a]+program.graph_blob[b:]
                    candidate=replace(program,graph_blob=blob,patch_reports=(json.dumps(patch).encode(),))
                    candidate_key=('rmsnorm_reduce_model',tag,index,local[-1]);strategies['codec'][index][local]=candidate_key
                    npu._sessions[candidate_key]=npu._native.create_inference_session(blob,parent=npu._root_session),candidate;npu._session_creations+=1
                    graph_tag=f'{tag}-l{index}-{local[-1]}'
                    for name,data in (('baseline',program.graph_blob),('codec',blob)):(OUT/f'{graph_tag}-{name}.blob').write_bytes(data)
                    report['carrier_evidence'].append(dict(tag=graph_tag,baseline=old,codec=patch,layout_sha256=hashlib.sha256(blob[:a]+bytes(b-a)+blob[b:]).hexdigest()))
            def select(name):
                for index,layer in enumerate(stack.layers):layer._keys=strategies[name][index]
            snapshots={f'{tag}/baseline/{i}':v for i,v in enumerate(anchors)}
            select('codec');output,new_seed=stack.prefill(x);decoded,updated=stack.decode(one,seed)
            actual=arrays(output,new_seed)+arrays(decoded,updated)
            for i,(a,b) in enumerate(zip(anchors,actual)):
                np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8));snapshots[f'{tag}/codec/{i}']=b
            del output,new_seed,decoded,updated,a,b
            report['checks'].append(dict(tag=tag,bit_equal_arrays=len(actual)))
            for operation,fn in (('prefill',lambda:stack.prefill(x)),('decode',lambda:stack.decode(one,seed))):
                def run(name):select(name);return fn()
                record=dict(tag=tag,operation=operation,**times({name:(lambda name=name:run(name)) for name in strategies},rounds,npu))
                assert record['warm_delta']['npu_calls']==rounds*4*4
                report['timings'].append(record);print(tag,operation,record['median_ms'],record['positive_rounds'],flush=True)
            # Repeated decode must leave the original prefix intact.
            old_seed=anchors[1:13]
            for i,tensor in enumerate(t for c in seed for t in (c.keys,c.values,c._position)):
                value=tensor.cpu().numpy().copy();np.testing.assert_array_equal(value.view(np.uint8),old_seed[i].view(np.uint8))
                snapshots[f'{tag}/preserved/{i}']=value
            path=OUT/f'{tag}-outputs.npz';np.savez(path,**snapshots)
            report['checks'][-1].update(outputs_sha256=sha(path),seed_preserved=True)
            del value,tensor,stack,seed,x,one,layer
    # Use the frozen old ELF even after the candidate becomes the default.
    current_kernel=smol._rmsnorm_kernel
    smol._rmsnorm_kernel=lambda:elf['baseline']
    try:work()
    finally:smol._rmsnorm_kernel=current_kernel


def execute(case,rounds,label):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(8);torch.manual_seed(92311)
    report=dict(status='running',case=case,rounds=rounds,source_sha256=sha(Path(__file__)))
    try:
        with torch.no_grad():{'numeric':numeric,'model':model}[case](npu,torch,report,rounds)
        gc.collect();npu.shutdown();report['final_stats']=npu.stats()
        assert not report['final_stats']['context_failed'] and report['final_stats']['live_buffers']==report['final_stats']['cached_buffer_bytes']==0,report['final_stats']
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/f'{case}-{label}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return int(report['status']!='passed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('numeric','model'),required=True)
    parser.add_argument('--rounds',type=int,default=21);parser.add_argument('--label',default='probe');parser.add_argument('--child',action='store_true')
    args=parser.parse_args();os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.child:return execute(args.case,args.rounds,args.label)
    with (OUT/f'{args.case}-{args.label}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--case',args.case,'--rounds',str(args.rounds),'--label',args.label,'--child'],stdout=log,stderr=log,timeout=360)
    print((OUT/f'{args.case}-{args.label}.log').read_text(encoding='utf-8'),flush=True);return result.returncode


if __name__=='__main__':raise SystemExit(main())

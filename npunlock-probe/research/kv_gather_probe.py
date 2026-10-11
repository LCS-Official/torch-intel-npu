"""Copy-only KV update candidate; keep the default model untouched."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import types

import numpy as np
import torch
from rmsnorm_reduce_probe import times
from w8_quality_probe import ROOT, MODEL, sha
import w8_calibration as calibration

OUT=ROOT/'npunlock-probe/research/results/kv-gather'
SOURCES=('torch_npu_smollm.py','torch_intel_npu.py','torch_npu_lowbit.py',
         'npunlock-probe/torch-device/_torch_intel_npu.pyd','npunlock-probe/session-runtime/npunlock_session.dll',
         'npunlock/tests/fixtures/npu4000/rmsnorm.c','npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf',
         'npunlock-probe/research/rmsnorm_reduce_probe.py')


def candidate(directory,kind):
    source=(ROOT/'torch_npu_smollm.py').read_text(encoding='utf-8')
    old="""            selected=reshape(op('Equal',slots,p,shape=(64,),dtype='boolean',auto_broadcast='numpy'),(1,1,64,1))
            k_cache=op('Select',selected,k_cache,past_k,shape=(1,batch*3,64,64) if folded else (b,3,64,64),auto_broadcast='numpy')
            v_cache=op('Select',selected,v_cache,past_v,shape=(1,batch*3,64,64) if folded else (b,3,64,64),auto_broadcast='numpy')"""
    new="""            selected=op('Equal',slots,p,shape=(64,),dtype='boolean',auto_broadcast='numpy')
            update_indices=op('Select',selected,s.constant(np.int64(64)),s.constant(np.arange(64,dtype=np.int64)),shape=(64,),dtype='i64',auto_broadcast='numpy')
            def update(past,current):
                joined=op('Concat',past,current,shape=(*past.shape[:2],65,64),axis=2)
                return op('Gather',joined,update_indices,s.constant(np.int64(2)),shape=past.shape,batch_dims=0)
            k_cache=update(past_k,k_cache);v_cache=update(past_v,v_cache)"""
    if kind in ('scatter_nd','scatter_flat'):
        new="""            offsets=s.constant((np.arange(batch*3,dtype=np.float16)*64).reshape(batch*3,1))
            flat_indices=op('Convert',op('Add',offsets,p,shape=(batch*3,1),auto_broadcast='numpy'),dtype='i64',destination_type='i64')
            def update(past,current):
                flat=reshape(past,(batch*3*64,64));updates=reshape(current,(batch*3,64))
                return reshape(op('ScatterNDUpdate',flat,flat_indices,updates,shape=flat.shape),past.shape)
            k_cache=update(past_k,k_cache);v_cache=update(past_v,v_cache)"""
        if kind=='scatter_flat':
            new=new.replace('reshape(batch*3,1)','reshape(batch*3)').replace('shape=(batch*3,1)','shape=(batch*3,)')
            new=new.replace("op('ScatterNDUpdate',flat,flat_indices,updates,shape=flat.shape)","op('ScatterUpdate',flat,flat_indices,updates,s.constant(np.int64(0)),shape=flat.shape)")
        line="            if layer.attrib['type']=='Swish':layer.set('version','opset4')"
        assert source.count(line)==1
        source=source.replace(line,line+"\n            if layer.attrib['type'] in ('ScatterNDUpdate','ScatterUpdate'):layer.set('version','opset3')")
    if kind=='gather_i32':
        new=new.replace('np.int64(64)','np.int32(64)').replace('np.arange(64,dtype=np.int64)','np.arange(64,dtype=np.int32)').replace("dtype='i64'","dtype='i32'")
    assert source.count(old)==1
    source=source.replace(old,new);path=directory/'candidate.py';path.write_text(source,encoding='utf-8')
    module=types.ModuleType('_smol_kv_gather');module.__file__=str(ROOT/'torch_npu_smollm.py')
    sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module


def arrays(result):
    value,caches=result
    return [value.cpu().numpy().copy(),*[getattr(c,field).cpu().numpy().copy()
        for c in caches for field in ('keys','values','_position')]]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',default='process0');parser.add_argument('--rounds',type=int,default=31)
    parser.add_argument('--candidate',choices=('gather','gather_i32','scatter_nd','scatter_flat'),default='gather')
    args=parser.parse_args();assert 1<=args.rounds<=101
    directory=OUT/args.label;directory.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    import torch_intel_npu as npu
    import torch_npu_smollm as baseline
    from safetensors import safe_open
    from transformers import AutoTokenizer
    torch.set_num_threads(8)
    other=candidate(directory,args.candidate)
    calibration.OUT=ROOT/'npunlock-probe/research/results/rmsnorm-encode/calibrated-current'
    packed=calibration.load_weights()
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as file:embedding=file.get_tensor('model.embed_tokens.weight')
    ids=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    report=dict(status='running',candidate=args.candidate,rounds=args.rounds,source_sha256=sha(Path(__file__)),candidate_sha256=sha(directory/'candidate.py'),
        sources={name:sha(ROOT/name) for name in SOURCES},checks=[],timings=[],graphs=[],calibration_weights_sha256=sha(calibration.OUT/'weights.npz'))
    saved={}
    def compare(tag,first,second):
        a,b=arrays(first),arrays(second);assert len(a)==len(b)
        for i,(x,y) in enumerate(zip(a,b)):
            np.testing.assert_array_equal(x.view(np.uint8),y.view(np.uint8),err_msg=f'{tag}/{i}')
            saved[f'{tag}/baseline/{i}']=x;saved[f'{tag}/{args.candidate}/{i}']=y
        report['checks'].append(dict(tag=tag,bit_equal_arrays=len(a)))
    def work():
        for batch,prefix,mode in ((1,32,'fp16'),(1,32,'w8a16'),(4,16,'fp16'),(4,16,'w8a16')):
            tag=f'b{batch}-s{prefix}-{mode}'
            tokens=ids.repeat(1,(64+ids.shape[1])//ids.shape[1])[:,:64].repeat(batch,1)
            host=embedding[tokens].half().contiguous()
            x=host[:,:prefix].contiguous().to('intel_npu')
            inputs=[host[:,i:i+1].contiguous().to('intel_npu') for i in range(prefix,64)]
            models={name:module.FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,mode=mode,
                packed_weights=packed if mode=='w8a16' else None) for name,module in (('baseline',baseline),(args.candidate,other))}
            seeds={name:stack.prefill(x) for name,stack in models.items()}
            compare(tag+'/prefill',*seeds.values())
            decoded={name:stack.decode(inputs[0],seeds[name][1]) for name,stack in models.items()}
            compare(tag+'/decode',*decoded.values());del decoded
            timing=times({name:(lambda name=name:models[name].decode(inputs[0],seeds[name][1])) for name in models},args.rounds,npu)
            timing.update(tag=tag,operation='decode');report['timings'].append(timing)
            assert timing['warm_delta']['npu_calls']==args.rounds*16
            print(tag,timing['median_ms'],timing['positive_rounds'],flush=True)
            states={name:seed[1] for name,seed in seeds.items()}
            for i,item in enumerate(inputs):
                decoded={name:stack.decode(item,states[name]) for name,stack in models.items()}
                compare(f'{tag}/chain-{prefix+i+1}',*decoded.values())
                states={name:result[1] for name,result in decoded.items()}
                assert all(c.length==prefix+i+1 for state in states.values() for c in state)
            # Same old seed remains reusable after advancing both independent chains.
            again={name:stack.decode(inputs[0],seeds[name][1]) for name,stack in models.items()}
            compare(tag+'/repeat',*again.values())
            for name in models:
                for i,value in enumerate(arrays(seeds[name])):
                    np.testing.assert_array_equal(value.view(np.uint8),saved[f'{tag}/prefill/{name}/{i}'].view(np.uint8))
                for i,value in enumerate(arrays(again[name])):
                    np.testing.assert_array_equal(value.view(np.uint8),saved[f'{tag}/decode/{name}/{i}'].view(np.uint8))
                for index,layer in enumerate(models[name].layers):
                    key=next(key for key in layer._keys if key[-1]=='decode');session=layer._keys[key]
                    graph,xml,data=layer.graph(key[0],segment='layer',kv_cache='decode',fused_norm=True)
                    blob=npu._sessions[session][1].graph_blob;stem=f'{tag}-{name}-l{index}-decode'
                    (directory/f'{stem}.xml').write_bytes(xml);(directory/f'{stem}.blob').write_bytes(blob)
                    report['graphs'].append(dict(stem=stem,xml_sha256=hashlib.sha256(xml).hexdigest(),blob_sha256=hashlib.sha256(blob).hexdigest(),evidence=layer._evidence[key]))
            del inputs,item,x,models,seeds,states,decoded,again,layer
    try:
        with torch.no_grad():work()
        gc.collect();npu.shutdown();report['final_stats']=npu.stats()
        assert report['final_stats']['closed'] and not report['final_stats']['context_failed']
        assert all(report['final_stats'][key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
        path=directory/'outputs.npz';np.savez(path,**saved)
        report.update(status='passed',outputs_sha256=sha(path),output_array_count=len(saved))
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (directory/'probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

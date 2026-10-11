"""Isolate a standard ScatterUpdate replacement; keep production model intact."""
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import types

from decode_cache_probe import ROOT,OUT,sha


def main():
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from safetensors import safe_open
    from transformers import AutoTokenizer
    import torch_npu_smollm as baseline
    directory=OUT/'scatter';directory.mkdir(exist_ok=True)
    source=(ROOT/'torch_npu_smollm.py').read_text(encoding='utf-8')
    old="""            selected=reshape(op('Equal',slots,p,shape=(64,),dtype='boolean',auto_broadcast='numpy'),(1,1,64,1))
            k_cache=op('Select',selected,k_cache,past_k,shape=(1,batch*3,64,64) if folded else (b,3,64,64),auto_broadcast='numpy')
            v_cache=op('Select',selected,v_cache,past_v,shape=(1,batch*3,64,64) if folded else (b,3,64,64),auto_broadcast='numpy')"""
    new="""            axis=s.constant(np.int64(2))
            k_cache=op('ScatterUpdate',past_k,indices,k_cache,axis,shape=past_k.shape)
            v_cache=op('ScatterUpdate',past_v,indices,v_cache,axis,shape=past_v.shape)"""
    assert source.count(old)==1;source=source.replace(old,new)
    line="            if layer.attrib['type']=='Swish':layer.set('version','opset4')"
    assert source.count(line)==1;source=source.replace(line,line+"\n            if layer.attrib['type']=='ScatterUpdate':layer.set('version','opset3')")
    (directory/'candidate.py').write_text(source,encoding='utf-8')
    module=types.ModuleType('_smol_scatter_probe');module.__file__=str(ROOT/'torch_npu_smollm.py')
    sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
    torch.set_num_threads(8);model_dir=ROOT/'models/SmolLM2-135M-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    with safe_open(model_dir/'model.safetensors',framework='pt',device='cpu') as file:embedding=file.get_tensor('model.embed_tokens.weight')
    report=dict(status='running',checks=[],timings=[],source_sha256=sha(Path(__file__)),candidate_sha256=sha(directory/'candidate.py'))
    def work():
        for batch,prefix,mode in ((1,32,'fp16'),(1,32,'w8a16'),(4,16,'fp16'),(4,16,'w8a16')):
            tag=f'b{batch}-s{prefix}-{mode}';tokens=ids.repeat(1,(prefix+ids.shape[1])//ids.shape[1])[:,:prefix+1].repeat(batch,1)
            host=embedding[tokens].half().contiguous();x=host[:,:prefix].contiguous().to('intel_npu');one=host[:,prefix:].contiguous().to('intel_npu')
            models={name:cls.from_checkpoint(model_dir,num_layers=4,mode=mode) for name,cls in
                (('select',baseline.FrozenSmolStack),('scatter',module.FrozenSmolStack))}
            seeds={};outputs={}
            for name,stack in models.items():
                output,seeds[name]=stack.prefill(x);output,state=stack.decode(one,seeds[name])
                values=[output,*[t for c in state for t in (c.keys,c.values,c._position)]]
                outputs[name]=[t.cpu().numpy().copy() for t in values]
                del output,state,values
            differences=sum(int(np.count_nonzero(a.view(np.uint8)!=b.view(np.uint8))) for a,b in zip(outputs['select'],outputs['scatter']))
            report['checks'].append(dict(tag=tag,compared_arrays=13,byte_differences=differences));assert differences==0,(tag,differences)
            samples={name:[] for name in models}
            for _ in range(5):
                for name in models:value=models[name].decode(one,seeds[name]);del value
            before=npu.stats()
            for i in range(21):
                order=('select','scatter','scatter','select') if i%2==0 else ('scatter','select','select','scatter')
                for name in order:
                    start=time.perf_counter_ns();value=models[name].decode(one,seeds[name]);samples[name].append((time.perf_counter_ns()-start)/1e6);del value
            after=npu.stats()
            for field in ('host_upload_copies','host_download_copies','session_creations'):assert after[field]==before[field]
            row=dict(tag=tag,samples_ms=samples,median_ms={name:statistics.median(v) for name,v in samples.items()},
                npu_calls=after['npu_calls']-before['npu_calls']);report['timings'].append(row);print(tag,row['median_ms'],flush=True)
            for i,layer in enumerate(models['scatter'].layers):
                for key,session in layer._keys.items():
                    graph,xml,data=layer.graph(key[0],segment='layer',kv_cache=key[-1],fused_norm=True)
                    (directory/f'{tag}-l{i}-{key[-1]}.xml').write_bytes(xml)
                    (directory/f'{tag}-l{i}-{key[-1]}.blob').write_bytes(npu._sessions[session][1].graph_blob)
            del x,one,models,seeds,stack,layer
    try:
        with torch.no_grad():work()
        gc.collect();npu.shutdown();report['final_stats']=npu.stats()
        assert not report['final_stats']['context_failed'] and report['final_stats']['live_buffers']==0
        report['status']='passed'
    except Exception:
        import traceback
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (directory/'probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    raise SystemExit(main())

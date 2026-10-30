"""S60: real frozen INT4/NF4 decoder forward and bounded KV validation."""
import argparse
import ast
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import traceback

import torch
from torch_npu_smollm import FrozenSmolStack
from torch_npu_smollm_stack_demo import errors
import w8_sequential_calibration as common
from w8_calibration_probe import metric

ROOT=common.ROOT
OUT=ROOT/'npunlock-probe/research/results/smollm-lowbit-coverage'


def prepare():
    path=OUT/'corpus.json'
    if path.exists():raise FileExistsError('Corpus already frozen')
    tree=ast.parse((Path(__file__).parent/'w8_native_down_check.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FRESH' for t in n.targets))
    prompts=ast.literal_eval(node.value);cpu,tokenizer=common.model();cases=[]
    eos=cpu.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [])
    for tag,prompt in prompts:
        ids=common.chat(tokenizer,'You are a helpful assistant.',prompt);prefix=ids.shape[1]
        ids=common.trajectory(cpu,ids,64);end=64
        for i in range(prefix,64):
            if int(ids[0,i]) in eos:end=i;break
        assert prefix-1<end<=64
        cases.append(dict(tag=tag,prompt=prompt,token_ids=ids.tolist(),offset=prefix-1,end=end))
    path.write_text(json.dumps(dict(cases=cases,policy='Eight fixed prompts, no calibration. Score through the first EOS prediction; ignore later continuation.',
        source_sha256=common.sha(Path(__file__))),indent=2),encoding='utf-8')


def run(mode,layers,graph_cache_mib=512):
    path=OUT/f'{mode}-{layers}.json'
    if path.exists():raise FileExistsError('Refusing to overwrite hardware record')
    import torch_intel_npu as npu
    npu.set_graph_cache_budget(graph_cache_mib*1024**2)
    report=dict(status='running',mode=mode,layers=layers,graph_cache_mib=graph_cache_mib,local_checks=[],positions=[],
        sources={n:common.sha(ROOT/n) for n in ('torch_npu_smollm.py','torch_npu_transformer.py','torch_npu_lowbit.py','npunlock-probe/research/smollm_lowbit_coverage.py')},
        corpus_sha256=common.sha(OUT/'corpus.json'),checkpoint_sha256=common.sha(common.MODEL/'model.safetensors'),
        packages={name:importlib.metadata.version(name) for name in ('torch','numpy','transformers','safetensors')})
    def check(actual,reference,stage,layer):
        result=errors(actual.cpu(),reference)
        report['local_checks'].append(dict(stage=stage,layer=layer,**result))
        if result['relative_l2']>=.003 or result['max_scaled_error']>=.01:
            report['local_guard_failure']=report['local_checks'][-1]
            raise ValueError(('Unchanged local execution guard',mode,stage,layer,result))
    def work():
        cpu,_=common.model();stack=FrozenSmolStack.from_checkpoint(common.MODEL,num_layers=layers,mode=mode)
        cases=json.loads((OUT/'corpus.json').read_text(encoding='utf-8'))['cases']
        selected=cases if layers==30 else cases[:1]
        for index,row in enumerate(selected):
            ids=torch.tensor(row['token_ids']);x=cpu.model.embed_tokens(ids).contiguous().to('intel_npu');inputs=x
            if index==0:
                for i,layer in enumerate(stack.layers):
                    reference,_=layer.reference_cpu(x.cpu());x=layer(x);check(x,reference,'forward',i)
            else:x=stack(x)
            if layers==30:
                target=cpu(ids,use_cache=False).logits.float();actual=cpu.lm_head(cpu.model.norm(x.cpu())).float()
                for j in range(row['offset'],row['end']):report['positions'].append(dict(tag=row['tag'],position=j,**metric(actual[:,j],target[:,j])))
            print('lowbit forward',mode,layers,row['tag'],flush=True)
        before=npu.stats();repeat=stack(inputs);after=npu.stats()
        counters=('npu_calls','session_creations','host_upload_copies','host_download_copies','graph_cache_evictions','cached_graph_blob_bytes')
        report['warm_forward_stats']={label:{k:stats[k] for k in counters} for label,stats in (('before',before),('after',after))}
        torch.testing.assert_close(repeat.cpu(),x.cpu(),rtol=0,atol=0)
        assert after['npu_calls']-before['npu_calls']==layers
        assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
        report['warm_forward_calls']=layers
        ids=torch.tensor(cases[0]['token_ids'])[:,:9]
        x=cpu.model.embed_tokens(ids[:,:8]).contiguous().to('intel_npu');cache_inputs=[];caches=[];sequence=object()
        for i,layer in enumerate(stack.layers):
            host=x.cpu();cache_inputs.append(host);reference,_=layer.reference_cpu(host)
            x,cache=layer.prefill(x);caches.append(replace(cache,_sequence=sequence));check(x,reference,'prefill',i)
        def cache_hash(cache):
            return hashlib.sha256(cache.keys.cpu().numpy().tobytes()+cache.values.cpu().numpy().tobytes()).hexdigest()
        snapshots=[cache_hash(c) for c in caches]
        x=cpu.model.embed_tokens(ids[:,8:]).contiguous().to('intel_npu');decode_input=x;updated=[]
        for i,(layer,cache) in enumerate(zip(stack.layers,caches,strict=True)):
            host=torch.cat((cache_inputs[i],x.cpu()),dim=1);reference,_=layer.reference_cpu(host)
            x,new_cache=layer.decode(x,cache);updated.append(new_cache);check(x,reference[:,-1:],'decode',i)
            assert cache.length==8 and new_cache.length==9
        assert [cache_hash(c) for c in caches]==snapshots
        before=npu.stats();again,_=stack.decode(decode_input,tuple(caches),group_size=1);after=npu.stats()
        report['warm_decode_stats']={label:{k:stats[k] for k in counters} for label,stats in (('before',before),('after',after))}
        torch.testing.assert_close(again.cpu(),x.cpu(),rtol=0,atol=0)
        assert after['npu_calls']-before['npu_calls']==layers
        assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
        report.update(warm_decode_calls=layers,cache_snapshots_unchanged=True,
            packed_projection_bytes=sum(len(q.data)+len(q.scales) for layer in stack.layers for q in layer._packed.values()),
            fp16_projection_bytes=sum(len(data) for layer in stack.layers for data,shape in layer._weights.values() if len(shape)==2),
            first_layer_evidence=[dict(route=str(key),profile=value) for key,value in stack.layers[0]._evidence.items()],stats=npu.stats())
        if layers==30:
            report['summary']=common.summary(report['positions'])
            report['model_quality_passed']=report['summary']['quality_passed']==report['summary']['positions']
        print('lowbit result',mode,layers,report.get('summary'),flush=True)
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    gc.collect();npu.shutdown();report['shutdown']=npu.stats()
    report['healthy']=report['shutdown']['closed'] and not report['shutdown']['context_failed'] and all(
        report['shutdown'][k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_sessions','quarantined_buffer_bytes'))
    if not report['healthy']:report['status']='failed'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return int(report['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run'))
    parser.add_argument('--mode',choices=('fp16','w8a16','w4a16','nf4a16'),default='w4a16')
    parser.add_argument('--layers',type=int,choices=(1,30),default=30)
    parser.add_argument('--graph-cache-mib',type=int,choices=(512,1024),default=512)
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():result=prepare() if args.stage=='prepare' else run(args.mode,args.layers,args.graph_cache_mib)
    raise SystemExit(result or 0)

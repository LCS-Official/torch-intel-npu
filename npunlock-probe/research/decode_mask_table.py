"""Isolated bounded decode mask tables; baseline and candidates share weights/state."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import numpy as np
from decode_chain_api import equal, provenance
from rmsnorm_reduce_probe import times
from shave_fp32_pipeline import final, snapshots
from w8_quality_probe import ROOT, MODEL, sha

OUT=ROOT/'npunlock-probe/research/results/decode-mask-table'
OLD_MASK="""        forbidden=op('Greater',slots,p,shape=(64,),dtype='boolean',auto_broadcast='numpy')
        mask=reshape(op('Select',forbidden,s.constant(np.float16(-np.inf)),s.constant(np.float16(0)),shape=(64,),auto_broadcast='numpy'),(1,1,1,64))"""
NEW_MASK="""        mask_table=s.constant(np.where(np.arange(64)[None,:]>np.arange(64)[:,None],-np.inf,0).astype(np.float16))
        mask=reshape(op('Gather',mask_table,indices,s.constant(np.int64(0)),shape=(1,64),batch_dims=0),(1,1,1,64))"""
OLD_SELECTED="        selected=reshape(op('Equal',slots,p,shape=(64,),dtype='boolean',auto_broadcast='numpy'),(1,1,64,1))"
NEW_SELECTED="""        selected_table=s.constant(np.eye(64,dtype=np.bool_))
        selected=reshape(op('Gather',selected_table,indices,s.constant(np.int64(0)),shape=(1,64),dtype='boolean',batch_dims=0),(1,1,64,1))"""


def variants():
    import torch_npu_smollm as smol
    baseline=OUT/'baseline-graph.py'
    source=baseline.read_text() if baseline.exists() else textwrap.dedent(inspect.getsource(smol.FrozenSmolLayer.graph))
    assert source.count(OLD_MASK)==source.count(OLD_SELECTED)==1
    versions=dict(baseline=source,mask=source.replace(OLD_MASK,NEW_MASK),
                  both=source.replace(OLD_MASK,NEW_MASK).replace(OLD_SELECTED,NEW_SELECTED))
    result={}
    for name,text in versions.items():
        path=OUT/(name+'-graph.py');path.write_text(text,encoding='utf-8')
        scope=dict(vars(smol));exec(compile(text,str(path),'exec'),scope)
        result[name]=scope['graph']
    return result


def model(batch,prefix,layers,mode,names,all_positions):
    import torch
    import torch_intel_npu as npu
    import torch_npu_smollm as smol
    from safetensors import safe_open
    from transformers import AutoTokenizer
    torch.set_num_threads(8);torch.manual_seed(90210)
    functions=variants();original=smol.FrozenSmolLayer.graph
    tag=f'l{layers}-b{batch}-s{prefix}-{mode}'
    report=dict(status='running',tag=tag,layers=layers,batch=batch,prefix=prefix,mode=mode,
        sources=provenance(),wrapper_sha256=sha(Path(__file__)),graphs=[],checks=[],names=names)
    saved={}
    def work():
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
        tokens=ids.repeat(1,(prefix+ids.shape[1])//ids.shape[1])[:,:prefix+1].repeat(batch,1)
        with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as file:embedding=file.get_tensor('model.embed_tokens.weight')
        host=embedding[tokens].half().contiguous();x=host[:,:prefix].contiguous().to('intel_npu');one=host[:,prefix:].contiguous().to('intel_npu')
        stack=smol.FrozenSmolStack.from_checkpoint(MODEL,num_layers=layers,mode=mode)
        print(tag,'preparing baseline',flush=True)
        initial=stack.prefill(x);seed=initial[1];seed_arrays=[v.cpu().numpy().copy() for v in snapshots(initial)[1:]]
        decoded=stack.decode(one,seed);anchors=[v.cpu().numpy().copy() for v in snapshots(decoded)]
        for i,array in enumerate(anchors):saved[f'baseline/{i}']=array
        del initial,decoded
        strategies={'baseline':[dict(layer._keys) for layer in stack.layers]}
        for name in names:
            if name!='baseline':
                smol.FrozenSmolLayer.graph=functions[name]
                for layer,keys in zip(stack.layers,strategies['baseline']):layer._keys={local:key for local,key in keys.items() if local[-1]!='decode'}
                print(tag,'preparing',name,flush=True)
                result=stack.decode(one,seed)
                values=[v.cpu().numpy().copy() for v in snapshots(result)];equal(values,anchors)
                for i,array in enumerate(values):saved[f'{name}/{i}']=array
                report['checks'].append(dict(name=name,position=prefix,bit_equal_arrays=len(values)))
                del result,values
                strategies[name]=[dict(layer._keys) for layer in stack.layers]
            for i,layer in enumerate(stack.layers):
                local=next(k for k in strategies[name][i] if k[-1]=='decode');program=npu._sessions[strategies[name][i][local]][1]
                graph,xml,weights=functions[name](layer,(batch,1,576),segment='layer',kv_cache='decode',fused_norm=True)
                report['graphs'].append(dict(name=name,layer=i,xml_sha256=hashlib.sha256(xml).hexdigest(),
                    weight_bytes=len(weights),ir_nodes=dict(Counter(node.op for node in graph.nodes)),
                    blob_sha256=hashlib.sha256(program.graph_blob).hexdigest(),blob_bytes=len(program.graph_blob),
                    patch_reports=[json.loads(p) for p in program.patch_reports]))
                del graph,xml,weights
        def select(name):
            for layer,keys in zip(stack.layers,strategies[name]):layer._keys=keys
        if all_positions:
            for pos in range(1,64):
                position=torch.tensor([pos],dtype=torch.float32).to('intel_npu')
                test_seed=tuple(replace(cache,length=pos,_position=position) for cache in seed)
                expected=None
                for name in names:
                    select(name);result=stack.decode(one,test_seed);actual=[v.cpu().numpy().copy() for v in snapshots(result)]
                    if expected is None:expected=actual
                    else:equal(actual,expected)
                    assert all(c.length==pos+1 for c in result[1])
                    report['checks'].append(dict(name=name,position=pos,bit_equal_arrays=len(actual)))
                del result,test_seed,position,actual,expected
            print(tag,'all 63 decode positions bit equal',flush=True)
        def run(name):select(name);return stack.decode(one,seed)
        report['timing']=times({name:(lambda name=name:run(name)) for name in names},31,npu)
        assert report['timing']['warm_delta']['driver_allocations']==0
        assert report['timing']['warm_delta']['npu_calls']==31*2*len(names)*layers
        for i,cache in enumerate(seed):
            for j,value in enumerate((cache.keys,cache.values,cache._position)):
                actual=value.cpu().numpy().copy();equal([actual],[seed_arrays[3*i+j]]);saved[f'seed/{3*i+j}']=actual
        report['immutable_seed_arrays']=layers*3
        print(tag,report['timing']['median_ms'],report['timing']['positive_rounds'],flush=True)
    try:
        smol.FrozenSmolLayer.graph=functions['baseline']
        with torch.no_grad():work();final(npu,report)
    finally:smol.FrozenSmolLayer.graph=original
    path=OUT/(tag+'-outputs.npz');np.savez_compressed(path,**saved)
    report.update(status='passed',outputs_sha256=sha(path))
    (OUT/(tag+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=int,choices=(1,4,8),default=1)
    parser.add_argument('--prefix',type=int,choices=(24,63),default=24);parser.add_argument('--layers',type=int,choices=(4,30),default=4)
    parser.add_argument('--mode',choices=('fp16','w8a16'),default='fp16');parser.add_argument('--names',nargs='+',choices=('baseline','mask','both'),default=['baseline','mask','both'])
    parser.add_argument('--all-positions',action='store_true');args=parser.parse_args()
    assert args.names[0]=='baseline' and len(set(args.names))==len(args.names)>1
    OUT.mkdir(parents=True,exist_ok=True)
    model(args.batch,args.prefix,args.layers,args.mode,args.names,args.all_positions)

"""Paired fresh-process prefill startup with identical compiled blobs.

Four frozen policies, existing local model/calibration, bounded temporary
caches and no new dependencies. Full-IR is the prior W8 execution path.
"""
import time
ENTRY=time.perf_counter_ns()
import argparse
from functools import wraps
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import w8_efficiency as common

ROOT=common.ROOT
OUT=ROOT/'npunlock-probe/research/results/w8-prefill-cache'
LABELS=common.LABELS
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'


def helpers():
    ppl=common.research();ppl.layers.OUT=OUT;ppl.layers.mixed.OUT=OUT
    return ppl


def write(name,data):
    (OUT/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def prepare():
    import torch
    ppl=helpers();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    with torch.no_grad():ppl.layers.mixed.prepare()
    assert ppl.layers.mixed.read('prepare')['weights_sha256']=='f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62'
    write('selection',dict(selected=[list(key) for key in sorted(ppl.POLICY)]))
    print('Prepared the unchanged calibrated weights and frozen 45-item policy',flush=True)


def worker(args):
    import numpy as np
    import torch
    ppl=helpers()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    import torch_intel_npu as npu
    import torch_npu_smollm as smol
    import torch_npu_graph_cache as cache
    from torch_npu_smollm_stack_demo import errors
    totals={}
    def hook(owner,name,label):
        original=getattr(owner,name)
        @wraps(original)
        def wrapped(*a,**kw):
            begin=time.perf_counter_ns()
            try:return original(*a,**kw)
            finally:
                row=totals.setdefault(label,dict(calls=0,ms=0.))
                row['calls']+=1;row['ms']+=(time.perf_counter_ns()-begin)/1e6
        setattr(owner,name,wrapped)
    if args.route=='full':hook(smol.FrozenSmolLayer,'graph','full_ir')
    for owner,name,label in ((smol,'prefill_cache_graph','identity'),(smol,'compile_fused_cached','cache_or_compile'),
        (smol.FrozenSmolLayer,'_inspect_blob','inspect_blob'),(smol.FrozenSmolLayer,'_norm_parameter','gamma'),
        (cache,'_read','read'),(cache,'_key','key'),(npu._native,'create_inference_session','session')):
        hook(owner,name,label)
    began=time.perf_counter_ns()
    model,tokenizer=ppl.layers.mixed.common.model()
    ids=ppl.layers.mixed.common.chat(tokenizer,'Tell stories.','Tell a story about a bird.')
    assert ids.shape==(1,24)
    stack=ppl.layers.make_stack(args.label,None if args.label=='fp16' else ppl.layers.mixed.weights())
    setup_ms=(time.perf_counter_ns()-began)/1e6
    def first():
        previous={key:dict(value) for key,value in totals.items()};stats=npu.stats()
        begin=time.perf_counter_ns();host=model.model.embed_tokens(ids);embedded=time.perf_counter_ns()
        x=host.to('intel_npu');uploaded=time.perf_counter_ns()
        hidden,state=stack.prefill(x);prefilled=time.perf_counter_ns()
        output=hidden.cpu();downloaded=time.perf_counter_ns()
        logits=model.lm_head(model.model.norm(output))[:,-1].float();headed=time.perf_counter_ns()
        token=int(logits.argmax(-1));end=time.perf_counter_ns()
        changes={key:npu.stats()[key]-stats[key] for key in common.FIELDS}
        hooks={key:dict(calls=value['calls']-previous.get(key,{}).get('calls',0),
                       ms=value['ms']-previous.get(key,{}).get('ms',0.)) for key,value in totals.items()}
        assert len(state)==30 and all(value.length==24 for value in state)
        def digest(value):return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        arrays=[output]+[value.cpu() for entry in state for value in (entry.keys,entry.values,entry._position)]
        record=dict(ttft_ms=(end-begin)/1e6,entry_to_token_ms=(end-ENTRY)/1e6,
            embed_ms=(embedded-begin)/1e6,upload_ms=(uploaded-embedded)/1e6,prefill_ms=(prefilled-uploaded)/1e6,
            download_ms=(downloaded-prefilled)/1e6,head_ms=(headed-downloaded)/1e6,argmax_ms=(end-headed)/1e6,
            hooks=hooks,delta=changes,token=token,array_sha256=[digest(value) for value in arrays],logits_sha256=digest(logits))
        return record,logits
    with torch.no_grad():
        cold,anchor=first();cold_cache=cache.stats();samples=[]
        for index in range(9):
            row,logits=first()
            assert row['array_sha256']==cold['array_sha256'] and row['logits_sha256']==cold['logits_sha256']
            assert row['delta']['npu_calls']==30 and row['delta']['session_creations']==row['delta']['driver_allocations']==0
            assert row['delta']['host_upload_copies']==row['delta']['host_download_copies']==1
            if index>=2:samples.append(row)
        reference=model(ids,use_cache=False).logits[:,-1].float()
        quality=errors(anchor,reference)
        if args.label=='fp16':assert quality['relative_l2']<.03
        graphs=[]
        for i,layer in enumerate(stack.layers):
            assert len(layer._evidence)==1
            evidence=next(iter(layer._evidence.values()))
            integer=[m for m in evidence['modes'] if not m['elementwise'] and m['wmode'] in (1,2)]
            assert bool(integer)==(args.label!='fp16') and all(m['amode']==0 and m['wmode']==2 for m in integer)
            graphs.append(dict(layer=i,blob_sha256=evidence['blob_sha256'],constant_bytes=evidence['constant_bytes'],
                               integer_tasks=len(integer),integer_modes=sorted(set((m['amode'],m['wmode']) for m in integer))))
        del layer
    report=dict(status='passed',label=args.label,route=args.route,tag=args.tag,setup_ms=setup_ms,
        cold=cold,cold_cache=cold_cache,samples=samples,quality=quality,graphs=graphs,cache=cache.stats(),
        sources={name:common.sha(ROOT/name) for name in ('torch_npu_graph_cache.py','torch_npu_smollm.py',
            'torch_npu_lowbit.py','torch_intel_npu.py','npunlock-probe/session-runtime/npunlock_session.dll',
            'npunlock-probe/torch-device/_torch_intel_npu.pyd','npunlock-probe/research/w8_prefill_cache.py')})
    del stack,model;gc.collect();npu.shutdown();final=npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
    report['final_stats']=final;write(args.tag,report)
    print(args.tag,'TTFT ms',round(cold['ttft_ms'],3),'entry ms',round(cold['entry_to_token_ms'],3),
          'hits/compiles/IR skipped',cold_cache['hits'],cold_cache['compiles'],cold_cache['ir_skipped'],flush=True)


def launch(label,route,tag,directory):
    environment=dict(os.environ,NPUNLOCK_GRAPH_CACHE=str(directory))
    subprocess.run([sys.executable,'-B',__file__,'--case','worker','--label',label,'--route',route,'--tag',tag],
        env=environment,cwd=ROOT,check=True)
    return json.loads((OUT/(tag+'.json')).read_text(encoding='utf-8'))


def paired_cache(label,fast,full):
    import torch_npu_smollm as smol
    import torch_npu_graph_cache as cache
    from npunlock.npu4000 import compile_fused_rmsnorm
    ppl=helpers()
    stack=ppl.layers.make_stack(label,None if label=='fp16' else ppl.layers.mixed.weights())
    with zipfile.ZipFile(next(fast.glob('*.npc'))) as archive:profile=json.loads(archive.read('manifest.json'))['profile']
    previous=os.environ.get('NPUNLOCK_GRAPH_CACHE');os.environ['NPUNLOCK_GRAPH_CACHE']=str(fast)
    rows=[];kernel=smol._rmsnorm_kernel()
    try:
        for i,layer in enumerate(stack.layers):
            stub,recipe=cache.prefill_cache_graph(layer,(1,24,576),kernel,compile_fused_rmsnorm)
            fast_key=cache._key(recipe,profile,True)
            blob,reports=cache._read(fast/(fast_key+'.npc'),fast_key,profile,contract=cache._contract(stub))
            graph,xml,weights=layer.graph((1,24,576),segment='layer',kv_cache='prefill',fused_norm=True)
            assert cache._contract(graph)==cache._contract(stub)
            full_key=cache._key(hashlib.sha256(xml+weights+kernel).digest(),profile,True)
            assert cache._write(full,full_key,profile,blob,reports)
            assert cache._read(full/(full_key+'.npc'),full_key,profile)==(blob,reports)
            rows.append(dict(layer=i,fast_key=fast_key,full_key=full_key,blob_sha256=hashlib.sha256(blob).hexdigest()))
    finally:
        if previous is None:os.environ.pop('NPUNLOCK_GRAPH_CACHE',None)
        else:os.environ['NPUNLOCK_GRAPH_CACHE']=previous
    assert len(rows)==30
    del stack,layer;gc.collect()
    return rows


def study():
    report=dict(status='running',variants={})
    temp_root=Path(tempfile.gettempdir()).resolve()
    for label in LABELS:
        with tempfile.TemporaryDirectory(prefix='torch-intel-npu-w8-prefill-') as name:
            directory=Path(name).resolve()
            assert directory.parent==temp_root and not directory.is_symlink()
            fast=directory/'fast';full=directory/'full';fast.mkdir();full.mkdir()
            fill=launch(label,'fast',label+'-fill',fast)
            assert fill['cold_cache']['compiles']==fill['cold_cache']['writes']==30 and fill['cold_cache']['hits']==0
            mapping=paired_cache(label,fast,full) if label!='fp16' else []
            rows=[]
            for index in range(3):
                order=('full','fast') if index%2==0 else ('fast','full')
                if label=='fp16':order=('fast',)
                for route in order:
                    current=launch(label,route,f'{label}-{route}-{index+1}',fast if route=='fast' else full)
                    counts=current['cold_cache']
                    assert counts['hits']==30 and counts['compiles']==counts['rejected']==0
                    assert counts['ir_skipped']==(30 if route=='fast' else 0)
                    assert current['cold']['array_sha256']==fill['cold']['array_sha256']
                    assert current['cold']['logits_sha256']==fill['cold']['logits_sha256']
                    assert [g['blob_sha256'] for g in current['graphs']]==[g['blob_sha256'] for g in fill['graphs']]
                    rows.append(current['tag'])
            corruption=None
            if label=='w8':
                entry=next(fast.glob('*.npc'))
                with zipfile.ZipFile(entry) as archive:manifest=json.loads(archive.read('manifest.json'));blob=archive.read('graph.blob')
                manifest['io_contract']['outputs'][0]['shape']=[1,1,576]
                with zipfile.ZipFile(entry,'w') as archive:
                    archive.writestr('manifest.json',json.dumps(manifest));archive.writestr('graph.blob',blob)
                corruption=launch(label,'fast','w8-corrupt',fast)
                counts=corruption['cold_cache']
                assert {k:counts[k] for k in ('hits','rejected','compiles','writes','ir_skipped')}==dict(hits=29,rejected=1,compiles=1,writes=1,ir_skipped=29)
                assert corruption['cold']['array_sha256']==fill['cold']['array_sha256']
                assert corruption['cold']['logits_sha256']==fill['cold']['logits_sha256']
            entries=list(directory.rglob('*'))
            assert all(not p.is_symlink() for p in entries)
            files=[p for p in entries if p.is_file()]
            assert all(p.suffix=='.npc' for p in files) and len(files)==(30 if label=='fp16' else 60)
            report['variants'][label]=dict(mapping=mapping,rows=rows,cache_files=len(files),
                cache_bytes=sum(p.stat().st_size for p in files),corruption=corruption['tag'] if corruption else None)
            assert directory.resolve().parent==temp_root and not directory.is_symlink()
        assert not directory.exists()
    report.update(status='passed',caches_removed=True);write('study',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('prepare','study','worker'),required=True)
    parser.add_argument('--label',choices=LABELS);parser.add_argument('--route',choices=('fast','full'),default='fast')
    parser.add_argument('--tag');args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.case=='worker':
        assert args.label and args.tag and os.environ.get('NPUNLOCK_GRAPH_CACHE');worker(args)
    else:
        assert not os.environ.get('NPUNLOCK_GRAPH_CACHE')
        {'prepare':prepare,'study':study}[args.case]()

"""CPU startup preparation profile and paired NPU startup with identical blobs."""
import time
ENTRY=time.perf_counter_ns()
import argparse
from functools import wraps
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import w8_prefill_cache as previous

ROOT=previous.ROOT
OUT=ROOT/'npunlock-probe/research/results/smollm-startup-prepare'


def helpers():
    previous.OUT=OUT
    return previous.helpers()


def profile(label):
    import torch
    ppl=helpers();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import torch_npu_smollm as smol
    totals={}
    def hook(owner,name,key):
        original=getattr(owner,name)
        @wraps(original)
        def wrapped(*a,**kw):
            begin=time.perf_counter_ns()
            try:return original(*a,**kw)
            finally:
                row=totals.setdefault(key,dict(calls=0,ms=0.))
                row['calls']+=1;row['ms']+=(time.perf_counter_ns()-begin)/1e6
        setattr(owner,name,wrapped)
    hook(smol,'_half_bytes','half_bytes')
    hook(smol.FrozenSmolLayer,'__init__','layer_init')
    marks=[time.perf_counter_ns()]
    cpu=AutoModelForCausalLM.from_pretrained(ppl.layers.mixed.common.MODEL,dtype=torch.float16,
        attn_implementation='eager',local_files_only=True).eval();marks.append(time.perf_counter_ns())
    tokenizer=AutoTokenizer.from_pretrained(ppl.layers.mixed.common.MODEL,local_files_only=True)
    ids=ppl.layers.mixed.common.chat(tokenizer,'Tell stories.','Tell a story about a bird.');marks.append(time.perf_counter_ns())
    packed=None if label=='fp16' else ppl.layers.mixed.weights();marks.append(time.perf_counter_ns())
    stack=ppl.layers.make_stack(label,packed);marks.append(time.perf_counter_ns())
    assert ids.shape==(1,24) and len(stack.layers)==30
    report=dict(status='passed',label=label,stages_ms=dict(zip(('full_cpu_model','tokenizer','packed_load_verify','stack'),
        [(b-a)/1e6 for a,b in zip(marks,marks[1:])])),hooks=totals)
    (OUT/('profile-'+label+'.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(label,json.dumps(report),flush=True)
    del cpu,stack,packed;gc.collect()


def legacy_stack(label,packed):
    """Prior mixed research composition, retained only as a measured baseline."""
    ppl=helpers()
    if label!='mixed':return ppl.layers.make_stack(label,packed)
    from torch_npu_smollm import FrozenSmolLayer,FrozenSmolStack
    stack=FrozenSmolStack.from_checkpoint(ppl.layers.mixed.common.MODEL,num_layers=30,mode='w8a16',packed_weights=packed)
    config=json.loads((ppl.layers.mixed.common.MODEL/'config.json').read_text())
    for i,layer in enumerate(stack.layers):
        weights={name:layer._cpu_weight(name) for name in layer._weights}
        stack.layers[i]=FrozenSmolLayer(config,weights,mode='w8a16',packed_weights=packed[i],
            fp16_projections=('mlp.up_proj.weight','mlp.down_proj.weight'))
    return stack


def io_check():
    import torch
    from torch_npu_smollm_io import FrozenSmolIO
    ppl=helpers();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    cpu,_=ppl.layers.mixed.common.model();io=FrozenSmolIO.from_checkpoint(ppl.layers.mixed.common.MODEL)
    checks=[];generator=torch.Generator().manual_seed(711)
    with torch.no_grad():
        for batch,length in ((1,1),(1,24),(4,1),(8,64)):
            ids=torch.randint(0,49152,(batch,length),generator=generator)
            assert torch.equal(io.embed_tokens(ids),cpu.model.embed_tokens(ids))
            for kind in ('random','zero','maximum'):
                hidden=(torch.randn(batch,length,576,generator=generator)*.5).half()
                if kind=='zero':hidden.zero_()
                if kind=='maximum':hidden.fill_(65504);hidden[...,::2].neg_()
                expected=cpu.lm_head(cpu.model.norm(hidden))
                actual=io.logits(hidden)
                assert torch.equal(expected.view(torch.int16),actual.view(torch.int16))
                checks.append(dict(batch=batch,length=length,kind=kind,logits_bit_equal=True))
    assert 'torch_intel_npu' not in sys.modules
    write('io-check',dict(status='passed',checks=checks,checkpoint_sha256=previous.common.sha(ppl.layers.mixed.common.MODEL/'model.safetensors')))
    print('Real checkpoint: 4 embedding shapes and 12 output-head cases bit-identical to Transformers',flush=True)


def write(tag,data):
    (OUT/(tag+'.json')).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


def read(tag):return json.loads((OUT/(tag+'.json')).read_text(encoding='utf-8'))


def worker(args):
    import hashlib
    import torch
    import torch_npu_graph_cache as cache
    import torch_intel_npu as npu
    from torch_npu_smollm_io import FrozenSmolIO
    ppl=helpers();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    begin=time.perf_counter_ns();marks=[begin]
    if args.route=='before':
        model,tokenizer=ppl.layers.mixed.common.model()
        embed=model.model.embed_tokens
        head=lambda value:model.lm_head(model.model.norm(value))
    else:
        from transformers import AutoTokenizer
        model=FrozenSmolIO.from_checkpoint(ppl.layers.mixed.common.MODEL)
        tokenizer=AutoTokenizer.from_pretrained(ppl.layers.mixed.common.MODEL,local_files_only=True)
        embed=model.embed_tokens;head=model.logits
    ids=ppl.layers.mixed.common.chat(tokenizer,'Tell stories.','Tell a story about a bird.')
    assert ids.shape==(1,24);marks.append(time.perf_counter_ns())
    packed=None if args.label=='fp16' else ppl.layers.mixed.weights();marks.append(time.perf_counter_ns())
    stack=legacy_stack(args.label,packed) if args.route=='before' else ppl.layers.make_stack(args.label,packed)
    marks.append(time.perf_counter_ns());gc.collect()
    setup_ms=(time.perf_counter_ns()-begin)/1e6
    prepared_memory=previous.common.memory()
    def digest(value):return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
    def state_hash(output,state):
        return [digest(output)]+[digest(value) for entry in state for value in (entry.keys,entry.values,entry._position)]
    def first():
        stats=npu.stats();start=time.perf_counter_ns()
        host=embed(ids);x=host.to('intel_npu');hidden,state=stack.prefill(x)
        downloaded=hidden.cpu();logits=head(downloaded)[:,-1].float();token=int(logits.argmax(-1))
        end=time.perf_counter_ns()
        delta={key:npu.stats()[key]-stats[key] for key in previous.common.FIELDS}
        assert all(entry.length==24 for entry in state)
        return dict(ttft_ms=(end-start)/1e6,entry_to_token_ms=(end-ENTRY)/1e6,delta=delta,token=token,
            embedding_sha256=digest(host),arrays=state_hash(downloaded,state),logits_sha256=digest(logits))
    def generation():
        stats=npu.stats();tokens=ids.clone();state=None;hashes=[]
        for index in range(32):
            host=embed(ids if index==0 else tokens[:,-1:]);x=host.to('intel_npu')
            hidden,state=stack.prefill(x) if index==0 else stack.decode(x,state)
            logits=head(hidden.cpu())[:,-1].float();hashes.append(digest(logits))
            tokens=torch.cat((tokens,logits.argmax(-1,keepdim=True)),dim=1)
        assert all(entry.length==55 for entry in state)
        return dict(tokens=tokens.tolist(),logits_sha256=hashes,arrays=state_hash(hidden,state),
            npu_calls=npu.stats()['npu_calls']-stats['npu_calls'])
    with torch.no_grad():
        cold=first();cold_cache=cache.stats();samples=[]
        for index in range(9):
            row=first()
            assert row['arrays']==cold['arrays'] and row['logits_sha256']==cold['logits_sha256']
            assert row['delta']['npu_calls']==30 and row['delta']['session_creations']==row['delta']['driver_allocations']==0
            if index>=2:samples.append(row)
        startup_memory=previous.common.memory()
        graphs=[]
        for i,layer in enumerate(stack.layers):
            evidence=next(iter(layer._evidence.values()))
            integer=[m for m in evidence['modes'] if not m['elementwise'] and m['wmode'] in (1,2)]
            assert bool(integer)==(args.label!='fp16') and all(m['amode']==0 and m['wmode']==2 for m in integer)
            graphs.append(dict(layer=i,blob_sha256=evidence['blob_sha256']))
        del layer
        generated=generation() if args.generate else None
    report=dict(status='passed',label=args.label,route=args.route,tag=args.tag,setup_ms=setup_ms,
        stages_ms=dict(zip(('cpu_endpoints_and_tokenizer','packed_load_verify','stack'),[(b-a)/1e6 for a,b in zip(marks,marks[1:])])),
        cold=cold,cold_cache=cold_cache,samples=samples,prepared_memory=prepared_memory,startup_memory=startup_memory,
        graphs=graphs,generation=generated,cache=cache.stats(),sources={name:previous.common.sha(ROOT/name) for name in (
            'torch_npu_smollm.py','torch_npu_graph_cache.py','torch_npu_smollm_io.py','torch_npu_lowbit.py','torch_intel_npu.py',
            'npunlock-probe/research/w8_layer_sensitivity.py','npunlock-probe/research/smollm_startup_prepare.py',
            'npunlock-probe/session-runtime/npunlock_session.dll','npunlock-probe/torch-device/_torch_intel_npu.pyd')})
    del stack,model,embed,head,packed;gc.collect();npu.shutdown();final=npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
    report['final_stats']=final;write(args.tag,report)
    print(args.tag,'setup/TTFT/entry ms',round(setup_ms,3),round(cold['ttft_ms'],3),round(cold['entry_to_token_ms'],3),flush=True)


def study():
    report=dict(status='running',variants={});temp_root=Path(tempfile.gettempdir()).resolve()
    def launch(label,route,tag,directory,generate=False):
        argv=[sys.executable,'-B',__file__,'--case','worker','--label',label,'--route',route,'--tag',tag]
        if generate:argv.append('--generate')
        subprocess.run(argv,cwd=ROOT,env=dict(os.environ,NPUNLOCK_GRAPH_CACHE=str(directory)),check=True)
        return read(tag)
    for label in previous.LABELS:
        with tempfile.TemporaryDirectory(prefix='torch-intel-npu-startup-prepare-') as name:
            directory=Path(name).resolve();assert directory.parent==temp_root and not directory.is_symlink()
            fill=launch(label,'after',label+'-fill',directory,True)
            assert fill['cold_cache']['compiles']==fill['cold_cache']['writes']==30
            rows=[]
            for index in range(3):
                for route in (('before','after') if index%2==0 else ('after','before')):
                    row=launch(label,route,f'{label}-{route}-{index+1}',directory,index==0)
                    assert row['cold_cache']['hits']==row['cold_cache']['ir_skipped']==30 and row['cold_cache']['compiles']==0
                    assert row['cold']['arrays']==fill['cold']['arrays'] and row['cold']['logits_sha256']==fill['cold']['logits_sha256']
                    assert row['cold']['embedding_sha256']==fill['cold']['embedding_sha256']
                    assert row['graphs']==fill['graphs']
                    if index==0:assert row['generation']==fill['generation']
                    rows.append(row['tag'])
            entries=list(directory.iterdir())
            assert len(entries)==60 and all(p.suffix=='.npc' and p.is_file() and not p.is_symlink() for p in entries)
            report['variants'][label]=dict(rows=rows,cache_files=len(entries),cache_bytes=sum(p.stat().st_size for p in entries))
            assert directory.resolve().parent==temp_root and not directory.is_symlink()
        assert not directory.exists()
    report.update(status='passed',caches_removed=True);write('study',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('prepare','profile','profile-worker','io-check','study','worker'),required=True)
    parser.add_argument('--label',choices=previous.LABELS);parser.add_argument('--route',choices=('before','after'),default='after')
    parser.add_argument('--tag');parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.case=='worker':
        assert args.tag and args.label and os.environ.get('NPUNLOCK_GRAPH_CACHE');worker(args)
    elif args.case=='prepare':
        assert not os.environ.get('NPUNLOCK_GRAPH_CACHE')
        previous.OUT=OUT;previous.prepare()
    elif args.case=='profile-worker':
        assert args.label;profile(args.label)
    elif args.case=='profile':
        for label in previous.LABELS:
            subprocess.run([sys.executable,'-B',__file__,'--case','profile-worker','--label',label],cwd=ROOT,check=True)
    elif args.case=='io-check':io_check()
    else:
        assert not os.environ.get('NPUNLOCK_GRAPH_CACHE');study()

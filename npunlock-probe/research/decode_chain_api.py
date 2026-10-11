"""Hardware checks and paired benchmarks of the production synchronous chain."""
import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from w8_quality_probe import ROOT, MODEL, sha
from shave_fp32_pipeline import snapshots, final

OUT = ROOT / 'npunlock-probe/research/results/decode-chain-api'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
FIELDS = ('npu_calls','chain_groups','chain_graphs','driver_allocations','session_creations',
          'host_upload_copies','host_download_copies','host_device_copies','host_pack_copies')


def write(name, report): (OUT/name).write_text(json.dumps(report,indent=2),encoding='utf-8')
def delta(before, after): return {k:after[k]-before[k] for k in FIELDS}
def arrays(result): return [t.cpu().numpy().copy() for t in snapshots(result)]
def equal(a, b):
    assert len(a) == len(b)
    for x,y in zip(a,b):
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_array_equal(x.view(np.uint8), y.view(np.uint8))


def provenance():
    names = ['torch_intel_npu.py','torch_npu_smollm.py','npunlock/include/npunlock/graphinfer.h',
             'npunlock/src/graphinfer/session_win32.c','npunlock/src/workers/level_zero_min.h',
             'npunlock-probe/torch-device/backend.cpp','npunlock-probe/torch-device/_torch_intel_npu.pyd',
             'npunlock-probe/torch-device/build.json','npunlock-probe/session-runtime/npunlock_session.dll',
             'npunlock-probe/session-runtime/build.json','npunlock/tests/unit/test_session_chain.c',
             'npunlock/tests/fixtures/npu4000/rmsnorm.c','npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf',
             'npunlock/tests/fixtures/npu4000/kv_select/kv-select-f16.elf']
    return {name:sha(ROOT/name) for name in names}


def checks(profile=False):
    import torch
    if profile:
        import ctypes as ct
        import torch_npu_warm_profile as warm
        warm.OUT=OUT/'host-profile';warm.OUT.mkdir(exist_ok=True);warm.build()
        manifest=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
        dll=warm.OUT/'session_profile.dll';manifest['session_sha256']=sha(dll)
        path=warm.OUT/'bound.json';path.write_text(json.dumps(manifest),encoding='utf-8');warm.bind(dll,path)
        driver=ct.CDLL(str(dll));driver.warm_diag_reset.argtypes=[];driver.warm_diag_reset.restype=None
        driver.warm_diag_read.argtypes=[ct.POINTER(ct.c_uint64)]*2;driver.warm_diag_read.restype=None
        from torch_npu_warm_profile import STAGES
    import torch_intel_npu as npu
    torch.set_num_threads(8)
    report = dict(status='running',sources=provenance(),checks=[])
    def rejected(name, fn):
        before=npu.stats()
        try: fn()
        except (ValueError,RuntimeError) as error:
            change=delta(before,npu.stats()); assert all(v==0 for v in change.values()), (name,change)
            assert not npu.stats()['context_failed']
            report['checks'].append(dict(name=name,rejected=str(error)))
        else: raise AssertionError(('accepted invalid chain',name))
    def work():
        for dtype,code in ((torch.float16,'f16'),(torch.float32,'f32')):
            program=npu.bridge._program('add1',code,(16,)); keys=[]
            for i in range(7):
                key=('chain_api',code,i)
                session=npu._native.create_inference_session(program.graph_blob,parent=npu._root_session)
                npu._sessions[key]=session,program; npu._session_creations+=1; keys.append(key)
            host=torch.arange(16,dtype=dtype); x=host.to('intel_npu')
            def plan():
                entries=[]; value=x
                for key in keys:
                    entry=npu._prepare_many(key,[value],[((16,),dtype)])
                    entries.append(entry); value=entry[-1][0]
                return entries
            entries=plan(); npu._run_chain(entries,4)
            torch.testing.assert_close(entries[-1][-1][0].cpu(),host+7,rtol=0,atol=0)
            anchor=entries[-1][-1][0].cpu().numpy().copy()
            report['checks'].append(dict(dtype=code,cold_group4_tail3='passed'))
            for group in (1,2,4):
                fresh=plan(); before=npu.stats()
                if profile: driver.warm_diag_reset()
                npu._run_chain(fresh,group); change=delta(before,npu.stats())
                assert change['npu_calls']==change['chain_graphs']==7 and change['chain_groups']==(7+group-1)//group
                assert all(v==0 for k,v in change.items() if k not in ('npu_calls','chain_graphs','chain_groups'))
                torch.testing.assert_close(fresh[-1][-1][0].cpu(),host+7,rtol=0,atol=0)
                equal([entries[-1][-1][0].cpu().numpy()],[anchor])
                record=dict(dtype=code,warm_group=group,delta=change)
                if profile:
                    ns=(ct.c_uint64*9)();counts=(ct.c_uint64*9)();driver.warm_diag_read(ns,counts)
                    record['stage_counts']=dict(zip(STAGES,list(counts)))
                    assert record['stage_counts']['append_graph']==7
                    assert all(record['stage_counts'][k]==(7+group-1)//group for k in STAGES[4:])
                report['checks'].append(record)
            rejected(code+'/empty',lambda:npu._run_chain([],4))
            rejected(code+'/bool_group',lambda:npu._run_chain(entries,True))
            rejected(code+'/invalid_group',lambda:npu._run_chain(entries,3))
            duplicate=[entries[0],(entries[0][0],program,entries[1][2],entries[1][3])]
            rejected(code+'/duplicate_session',lambda:npu._run_chain(duplicate,4))
            backward=[entries[0],(entries[1][0],program,entries[1][2],(x,))]
            rejected(code+'/overwrite_earlier_input',lambda:npu._run_chain(backward,4))
            npu._run_chain(entries,4)
            torch.testing.assert_close(entries[-1][-1][0].cpu(),host+7,rtol=0,atol=0)
            def thread(_):
                with torch.no_grad():
                    for i in range(20):
                        local=plan(); npu._run_chain(local,4)
                        torch.testing.assert_close(local[-1][-1][0].cpu(),host+7,rtol=0,atol=0)
            with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(thread,range(4)))
            report['checks'].append(dict(dtype=code,threads=4,iterations_per_thread=20,passed=True))
    with torch.no_grad(): work(); final(npu,report)
    report['status']='passed'; write('checks-profile.json' if profile else 'checks.json',report)
    print('production chain checks passed',len(report['checks']),flush=True)


def models(configs=None, *, steps=4, boundary=False):
    import torch
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    from transformers import AutoTokenizer
    from safetensors import safe_open
    torch.set_num_threads(8)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as file: embedding=file.get_tensor('model.embed_tokens.weight')
    tokens=ids.repeat(1,(64+ids.shape[1])//ids.shape[1])[:,:64]
    hidden=embedding[tokens].half().contiguous()
    report=dict(status='running',sources=provenance(),checkpoint_sha256=sha(MODEL/'model.safetensors'),cases=[])
    saved={}
    def work():
        cases=configs if configs is not None else [(mode,batch,7,8) for mode in ('fp16','w8a16') for batch in (1,4,8)]+[('fp16',1,30,32)]
        for mode,batch,layers,prefix in cases:
            tag=f'{mode}-b{batch}-l{layers}'; print(tag,'preparing',flush=True)
            stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=layers,mode=mode)
            host=hidden.repeat(batch,1,1); x=host[:,:prefix].contiguous().to('intel_npu'); one=host[:,prefix:prefix+1].contiguous().to('intel_npu')
            initial=stack.prefill(x); seed=initial[1]; seed_before=arrays(initial)[1:]
            # This is the first decode of these graphs: exercise cold fallback.
            cold=stack.decode(one,seed,group_size=4); candidate=arrays(cold); del cold
            reference=stack.decode(one,seed,group_size=1); anchor=arrays(reference); equal(anchor,candidate); del reference
            case=dict(tag=tag,cold_bit_equal_arrays=len(anchor),checks=[],timing={})
            for group in (2,4,None):
                before=npu.stats(); value=stack.decode(one,seed,group_size=group); change=delta(before,npu.stats())
                equal(arrays(value),anchor); del value
                expected=layers if group is not None or (layers>1 and batch in (1,4,8)) else 0
                assert change['chain_graphs']==expected
                case['checks'].append(dict(group=group,bit_equal_arrays=len(anchor),delta=change))
            if (mode,batch,layers,prefix)==('fp16',1,30,32):
                old=ROOT/'npunlock-probe/research/results/decode-chain/plain-outputs.npz'
                with np.load(old,allow_pickle=False) as previous:
                    equal(anchor,[previous[f'baseline/{i}'] for i in range(91)])
                    equal(seed_before,[previous[f'seed-before/{i}'] for i in range(90)])
                case['previous_binary_bit_equal_arrays']=181
            # Pair the production group4 with the retained original API.
            for group in (1,4):
                for _ in range(3): value=stack.decode(one,seed,group_size=group); del value
            samples=[]; pairs=[]; rounds=21 if layers==30 else 15; before=npu.stats()
            for index in range(rounds):
                pair={1:[],4:[]}
                for group in ((1,4,4,1) if index%2==0 else (4,1,1,4)):
                    start=time.perf_counter_ns(); value=stack.decode(one,seed,group_size=group); returned=time.perf_counter_ns(); del value; freed=time.perf_counter_ns()
                    row=dict(group=group,api_ms=(returned-start)/1e6,cleanup_ms=(freed-returned)/1e6,total_ms=(freed-start)/1e6)
                    samples.append(row); pair[group].append(row['total_ms'])
                pairs.append({group:st.mean(values) for group,values in pair.items()})
            change=delta(before,npu.stats())
            assert change['npu_calls']==rounds*4*layers and change['chain_graphs']==rounds*2*layers
            assert change['chain_groups']==rounds*2*((layers+3)//4)
            assert all(v==0 for k,v in change.items() if k not in ('npu_calls','chain_graphs','chain_groups')),change
            medians={str(group):st.median(r['total_ms'] for r in samples if r['group']==group) for group in (1,4)}
            case['timing']=dict(samples=samples,pairs=pairs,median_total_ms=medians,positive_rounds=sum(p[4]<p[1] for p in pairs),warm_delta=change)
            print(tag,medians,'wins',case['timing']['positive_rounds'],'/',rounds,flush=True)
            left=right=seed
            for step in range(steps):
                baseline=stack.decode(one,left,group_size=1); grouped=stack.decode(one,right,group_size=4)
                equal(arrays(baseline),arrays(grouped)); left=baseline[1]; right=grouped[1]
                assert all(c.length==prefix+step+1 and int(c._position.cpu().item())==c.length for c in right)
            case['continuous_decode_steps']=steps
            case['continuous_final_position']=right[0].length
            if boundary:
                for name,result in (('baseline',baseline),('group4',grouped)):
                    for i,value in enumerate(arrays(result)):saved[f'final/{tag}/{name}/{i}']=value
            equal(seed_before,[t.cpu().numpy().copy() for c in seed for t in (c.keys,c.values,c._position)])
            # Actual 63-token prefill, final legal step, then reject length64.
            if layers==7 or boundary:
                full=stack.prefill(host[:,:63].contiguous().to('intel_npu'))
                a=stack.decode(one,full[1],group_size=1); b=stack.decode(one,full[1],group_size=4)
                equal(arrays(a),arrays(b)); assert all(c.length==64 and c._position.cpu().item()==64 for c in b[1])
                before=npu.stats()
                try: stack.decode(one,b[1],group_size=4)
                except ValueError: assert all(v==0 for v in delta(before,npu.stats()).values())
                else: raise AssertionError('full cache accepted')
                case['kv_boundary']='actual prefill63 -> decode64 -> reject65'
                del full,a,b
            for i,value in enumerate(anchor): saved[f'{tag}/{i}']=value
            report['cases'].append(case)
            write('models-progress.json',report)
            del stack,initial,seed,x,one,left,right,baseline,grouped; gc.collect()
    with torch.no_grad(): work(); final(npu,report)
    np.savez_compressed(OUT/'models-outputs.npz',**saved)
    report.update(status='passed',outputs_sha256=sha(OUT/'models-outputs.npz')); write('models.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--case',choices=('checks','models','profile'),required=True)
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.case in ('checks','profile'): checks(args.case=='profile')
    else: models()

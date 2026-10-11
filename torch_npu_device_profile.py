"""Isolated task profiling for the actual patched SmolLM fused graphs."""
import argparse
import ctypes as ct
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/smollm-device-profile'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    # Reuse the working host counters and profiling extension implementation.
    import torch_npu_warm_profile as warm
    warm.OUT=OUT/'host-source';warm.OUT.mkdir(parents=True,exist_ok=True);warm.build()
    original=ROOT/'npunlock/src/graphinfer/session_win32.c'
    text=(warm.OUT/'session_profile.c').read_text(encoding='utf-8')
    current='  uint32_t device_id;\n};';assert text.count(current)==1
    text=text.replace(current,'''  uint32_t device_id;
  void *diag_pool;
  npunlock_ze_graph_profiling_query_handle diag_query;
  uint32_t(__cdecl *diag_pool_destroy)(void *);
  uint32_t(__cdecl *diag_query_destroy)(void *);
  uint32_t(__cdecl *diag_get_data)(void *, uint32_t, uint32_t *, uint8_t *);
};''')
    marker='static void release_submission(graphinfer_session *session) {';assert text.count(marker)==1
    text=text.replace(marker,'''static void release_profile(graphinfer_session *session) {
  if (!session->context_owner->failed) {
    if (session->diag_query) { session->diag_query_destroy(session->diag_query); session->diag_query=NULL; }
    if (session->diag_pool) { session->diag_pool_destroy(session->diag_pool); session->diag_pool=NULL; }
  }
}
'''+marker)
    marker='  release_submission(session);\n  free(session->arguments);';assert text.count(marker)==1
    text=text.replace(marker,'  release_submission(session);\n  release_profile(session);\n  free(session->arguments);')
    marker='    release_submission(session);\n    if (!session->context_owner->failed && session->handle != NULL)';assert text.count(marker)==1
    text=text.replace(marker,'    release_submission(session);\n    release_profile(session);\n    if (!session->context_owner->failed && session->handle != NULL)')
    old='session->graph.append_execute(session->list, session->handle, NULL, NULL, 0, NULL)';assert text.count(old)==1
    text=text.replace(old,'session->graph.append_execute(session->list, session->handle, session->diag_query, NULL, 0, NULL)')
    old='first->graph.append_execute(first->list, entries[i].session->handle, NULL, NULL, 0, NULL)';assert text.count(old)==1
    text=text.replace(old,'first->graph.append_execute(first->list, entries[i].session->handle, entries[i].session->diag_query, NULL, 0, NULL)')
    extra=(ROOT/'npunlock-probe/research/dpu_diagnostic_extra.c').read_text(encoding='utf-8').split('/* ponytail:')[0]
    start=extra.index('__declspec(dllexport) uint32_t diag_times(')
    end=extra.index('__declspec(dllexport) uint32_t diag_profile_enable(',start)
    extra=extra[:start]+extra[end:];text+='\n'+extra+'''
/* Decoded task data can be cached by the query. Recreate it only after the
 * synchronous submission completed and its command list was reset. */
__declspec(dllexport) uint32_t diag_profile_refresh(graphinfer_session *s) {
  uint32_t code = 0;
  if (!s) return 0x78000004u;
  EnterCriticalSection(session_lock(s));
  if (!s->active || !s->initialized || s->context_owner->failed ||
      !s->diag_query || !s->diag_pool) { code=0x78000004u; goto done; }
  code=s->diag_query_destroy(s->diag_query);
  if (code) { s->context_owner->failed=true; goto done; }
  s->diag_query=NULL;
  code=s->diag_pool_destroy(s->diag_pool);
  if (code) { s->context_owner->failed=true; goto done; }
  s->diag_pool=NULL;
  code=diag_profile_enable(s);
done:
  LeaveCriticalSection(session_lock(s));
  return code;
}
'''
    generated=OUT/'session_task_profile.c';generated.write_text(text,encoding='utf-8')
    before=json.loads((warm.OUT/'profile-build.json').read_text(encoding='utf-8'))
    args=[str(generated) if p==str(warm.OUT/'session_profile.c')
          else str(OUT/'session_task_profile.dll') if p==str(warm.OUT/'session_profile.dll')
          else p for p in before['arguments']]
    result=subprocess.run(args,env=dict(os.environ,PATH=str(Path(args[0]).parent)+';'+os.environ['PATH']),capture_output=True,text=True)
    (OUT/'build.log').write_text(result.stdout+result.stderr,encoding='utf-8');result.check_returncode()
    (OUT/'build.json').write_text(json.dumps(dict(arguments=args,dll_sha256=sha(OUT/'session_task_profile.dll'),
        production_session_sha256=sha(original),generated_source_sha256=sha(generated),
        builder_sha256=sha(Path(__file__))),indent=2),encoding='utf-8')
    print('Isolated task profiling DLL built')


def setup():
    from torch_npu_warm_profile import bind
    dll=OUT/'session_task_profile.dll'
    manifest=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    manifest['session_sha256']=sha(dll)
    path=OUT/'bound-build.json';path.write_text(json.dumps(manifest,indent=2),encoding='utf-8');bind(dll,path)
    import torch_intel_npu as npu
    driver=npu._native.infer
    driver.diag_profile_enable.argtypes=[ct.c_void_p];driver.diag_profile_enable.restype=ct.c_uint32
    driver.diag_profile_refresh.argtypes=[ct.c_void_p];driver.diag_profile_refresh.restype=ct.c_uint32
    driver.diag_profile_data.argtypes=[ct.c_void_p,ct.c_uint32,ct.POINTER(ct.c_uint32),ct.POINTER(ct.c_uint8)]
    driver.diag_profile_data.restype=ct.c_uint32
    driver.warm_diag_reset.argtypes=[];driver.warm_diag_reset.restype=None
    driver.warm_diag_read.argtypes=[ct.POINTER(ct.c_uint64)]*2;driver.warm_diag_read.restype=None
    return npu,driver


def union_ns(tasks):
    edges=sorted((t['start_time_ns'],t['start_time_ns']+t['duration_ns']) for t in tasks if t['duration_ns'])
    total=0;end=0
    for a,b in edges:
        total+=max(0,b-max(a,end));end=max(end,b)
    return total


def read_tasks(driver,session,tag):
    sys.path.insert(0,str(ROOT/'npunlock-probe/research'))
    from dpu_diagnostic import Task,decode
    size=ct.c_uint32();handle=session._result.session
    code=driver.diag_profile_data(handle,2,ct.byref(size),None)
    assert code==0 and 0<size.value<=16*1024*1024,(hex(code),size.value)
    raw=(ct.c_uint8*size.value)();code=driver.diag_profile_data(handle,2,ct.byref(size),raw)
    assert code==0 and size.value%ct.sizeof(Task)==0,(hex(code),size.value,ct.sizeof(Task))
    data=bytes(raw)[:size.value];(OUT/f'{tag}-tasks.bin').write_bytes(data)
    tasks=[]
    for offset in range(0,len(data),ct.sizeof(Task)):
        value=Task.from_buffer_copy(data[offset:offset+ct.sizeof(Task)])
        tasks.append(dict(name=decode(value.name),layer_type=decode(value.layer_type),exec_type=value.exec_type,
            start_time_ns=value.start_time_ns,duration_ns=value.duration_ns,active_cycles=value.active_cycles,
            stall_cycles=value.stall_cycles,task_id=value.task_id,parent_layer_id=value.parent_layer_id))
    assert tasks and all(0<=t['exec_type']<=3 and t['duration_ns']<1_000_000_000 for t in tasks)
    valid=[t for t in tasks if t['duration_ns']]
    span=max(t['start_time_ns']+t['duration_ns'] for t in valid)-min(t['start_time_ns'] for t in valid)
    assert 0<span<1_000_000_000,span
    engines={name:dict(tasks=sum(t['exec_type']==kind for t in tasks),
             sum_ms=sum(t['duration_ns'] for t in tasks if t['exec_type']==kind)/1e6,
             union_ms=union_ns([t for t in tasks if t['exec_type']==kind])/1e6)
             for kind,name in ((1,'dpu'),(2,'shave'),(3,'dma'))}
    return dict(raw_sha256=hashlib.sha256(data).hexdigest(),task_record_bytes=ct.sizeof(Task),tasks=tasks,engines=engines,timeline_span_ms=span/1e6,
                observed_union_ms=union_ns(tasks)/1e6,unobserved_gaps_ms=(span-union_ns(tasks))/1e6,
                dma_duration_available=engines['dma']['tasks']>0)


def execute(batch,prefix,mode,*,num_layers=4,save_blobs=True):
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from torch_npu_smollm import FrozenSmolStack
    npu,driver=setup();torch.set_num_threads(8)
    tag=f'b{batch}-s{prefix}-{mode}'
    report=dict(status='running',tag=tag,batch=batch,prefix=prefix,mode=mode,graphs=[],checks=[],
        source_sha256=sha(Path(__file__)),profiling_dll_sha256=sha(OUT/'session_task_profile.dll'),
        production_sha256={p:sha(ROOT/p) for p in ('npunlock/src/graphinfer/session_win32.c',
            'npunlock-probe/session-runtime/npunlock_session.dll','npunlock-probe/torch-device/backend.cpp',
            'npunlock-probe/torch-device/_torch_intel_npu.pyd','torch_intel_npu.py','torch_npu_smollm.py')},
        device_build=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig')))
    directory=ROOT/'models/SmolLM2-135M-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as file:embedding=file.get_tensor('model.embed_tokens.weight')
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    tokens=ids.repeat(1,(prefix+ids.shape[1])//ids.shape[1])[:,:prefix+1].repeat(batch,1)
    host=embedding[tokens].half().contiguous()
    with torch.no_grad():
        x=host[:,:prefix].contiguous().to('intel_npu');one=host[:,prefix:].contiguous().to('intel_npu')
        stack=FrozenSmolStack.from_checkpoint(directory,num_layers=num_layers,mode=mode)
        output,seed=stack.prefill(x);decoded,updated=stack.decode(one,seed)
        anchors=[output,decoded,*[t for cache in (*seed,*updated) for t in (cache.keys,cache.values,cache._position)]]
        arrays={str(i):value.cpu().numpy().copy() for i,value in enumerate(anchors)}
        snapshots={f'plain/{i}':array for i,array in arrays.items()}
        del anchors,output,decoded,updated
        plain={key:(session,program) for key,(session,program) in npu._sessions.items()}
        original_compile=npu._native.compile_ir
        original_create=npu._native.create_inference_session
        def compile_ir(xml,weights,**options):
            assert 'PERF_COUNT' not in options.get('build_flags','')
            options['build_flags']=options.get('build_flags','')+' PERF_COUNT="YES"'
            report.setdefault('build_flags',[]).append(options['build_flags'])
            return original_compile(xml,weights,**options)
        npu._native.compile_ir=compile_ir
        def create_session(blob,**options):
            session=original_create(blob,**options)
            code=driver.diag_profile_enable(session._result.session)
            if code:session.close();raise RuntimeError(f'Profiling query creation: {hex(code)}')
            return session
        # This driver requires a query on the very first profiled graph execute.
        npu._native.create_inference_session=create_session
        # Separate cache keys keep plain/profile variants alive for controls.
        for layer in stack.layers:layer._keys.clear()
        npu._sessions.clear()
        try:
            output,pseed=stack.prefill(x);decoded,pupdated=stack.decode(one,pseed)
            anchors=[output,decoded,*[t for cache in (*pseed,*pupdated) for t in (cache.keys,cache.values,cache._position)]]
            for i,value in enumerate(anchors):
                actual=value.cpu().numpy().copy();snapshots[f'profile/{i}']=actual
                np.testing.assert_array_equal(actual.view(np.uint8),arrays[str(i)].view(np.uint8))
            del actual
            report['checks'].append('profile graph outputs/KV match plain graph bits')
            del value,anchors,output,decoded,pupdated
            for layer_index,layer in enumerate(stack.layers):
                for local,key in layer._keys.items():
                    session,program=npu._sessions[key]
                    graph_tag=f'{tag}-l{layer_index}-{local[-1]}'
                    if save_blobs:(OUT/f'{graph_tag}.blob').write_bytes(program.graph_blob)
                    report['graphs'].append(dict(tag=graph_tag,layer=layer_index,operation=local[-1],
                        blob_sha256=hashlib.sha256(program.graph_blob).hexdigest(),patch_reports=[json.loads(r) for r in program.patch_reports]))
            for operation,fn in (('prefill',lambda:stack.prefill(x)),('decode',lambda:stack.decode(one,pseed))):
                for _ in range(3):value=fn();del value
                samples=[]
                for sample in range(3):
                    for graph in report['graphs']:
                        if graph['operation']!=operation:continue
                        layer=stack.layers[graph['layer']]
                        key=next(k for local,k in layer._keys.items() if local[-1]==operation)
                        code=driver.diag_profile_refresh(npu._sessions[key][0]._result.session)
                        assert code==0,hex(code)
                    driver.warm_diag_reset();start=time.perf_counter_ns();value=fn();elapsed=(time.perf_counter_ns()-start)/1e6
                    times=(ct.c_uint64*9)();counts=(ct.c_uint64*9)();driver.warm_diag_read(times,counts)
                    samples.append(dict(ms=elapsed,host_wait_ms=times[6]/1e6,shared_alloc_calls=int(counts[0]),shared_free_calls=int(counts[1]),
                        native_call_counts=list(counts),native_times_ms=[t/1e6 for t in times]))
                    for graph in report['graphs']:
                        if graph['operation']!=operation:continue
                        layer=stack.layers[graph['layer']]
                        key=next(k for local,k in layer._keys.items() if local[-1]==operation)
                        session=npu._sessions[key][0]
                        graph.setdefault('samples',[]).append(read_tasks(driver,session,f"{graph['tag']}-{sample}"))
                    del value
                report[operation]=dict(median_ms=statistics.median(r['ms'] for r in samples),samples=samples)
            assert all(len({s['raw_sha256'] for s in graph['samples']})==3 for graph in report['graphs']), 'Task results must be fresh per execution'
            report['checks'].append('three distinct task records per graph after query recreation')
            for i,cache in enumerate(pseed):
                for j,value in enumerate((cache.keys,cache.values,cache._position)):
                    actual=value.cpu().numpy().copy();snapshots[f'seed/{2+i*3+j}']=actual
                    np.testing.assert_array_equal(actual.view(np.uint8),arrays[str(2+i*3+j)].view(np.uint8))
            del actual
            del value,cache,layer,stack,seed,pseed,x,one
        finally:
            npu._native.compile_ir=original_compile
            npu._native.create_inference_session=original_create
            for session,_ in plain.values():session.close()
        gc.collect();report['before_shutdown']=npu.stats();npu.shutdown();report['final_stats']=npu.stats()
        assert report['final_stats']['live_buffers']==report['final_stats']['cached_buffer_bytes']==0,report['final_stats']
        assert not report['final_stats']['context_failed']
        report['status']='passed';report['checks'].append('seed KV preserved and all live/cache allocations released')
    path=OUT/f'{tag}-outputs.npz';np.savez(path,**snapshots);report['outputs_sha256']=sha(path)
    (OUT/f'{tag}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(tag,'passed',json.dumps({g['tag']:g['samples'][0]['engines'] for g in report['graphs']}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('build','probe','all'),required=True)
    parser.add_argument('--batch',type=int,default=1);parser.add_argument('--prefix',type=int,default=32)
    parser.add_argument('--mode',choices=('fp16','w8a16'),default='fp16')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.case=='build':build()
    elif args.case=='probe':execute(args.batch,args.prefix,args.mode)
    else:
        for batch,prefix,mode in ((1,32,'fp16'),(1,32,'w8a16'),(4,16,'fp16'),(4,16,'w8a16')):
            tag=f'b{batch}-s{prefix}-{mode}'
            with (OUT/f'{tag}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,__file__,'--case','probe','--batch',str(batch),'--prefix',str(prefix),'--mode',mode],
                               stdout=log,stderr=log,timeout=300,check=True)
            print(tag,'completed',flush=True)


if __name__=='__main__':main()

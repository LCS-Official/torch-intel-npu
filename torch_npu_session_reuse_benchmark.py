"""Same-blob runtime control and isolated same-model before/after checks."""
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
import types

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/session-reuse'
FIXTURES = OUT.parent/'dpu-diagnostic'
OLD_DLL = OUT/'before/npunlock_session.dll'
NEW_DLL = ROOT/'npunlock-probe/session-runtime/npunlock_session.dll'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(runs,rounds=31):
    samples = {name:[] for name in runs}
    for run in runs.values():
        for _ in range(5): value=run(); del value
    keys = tuple(runs)
    for index in range(rounds):
        order = keys[index%len(keys):]+keys[:index%len(keys)]
        if index%2: order = tuple(reversed(order))
        for name in order:
            start = time.perf_counter_ns(); value = runs[name]()
            samples[name].append((time.perf_counter_ns()-start)/1e6)
            del value
    return {name:dict(median_ms=statistics.median(values),samples_ms=values) for name,values in samples.items()}


def native():
    import numpy as np
    import npunlock as npu
    report = dict(status='running',rows=[],successful_graph_calls=0,
                  dll_sha256={'before':sha(OLD_DLL),'after':sha(NEW_DLL)},
                  script_sha256=sha(Path(__file__)),scope='same saved ELF blob and resident buffers; '
                  'compile/allocation/upload excluded; 31 alternating rounds, 5 warmups')
    libraries = {name:npu.NativeLibraries(session_library=path) for name,path in (('before',OLD_DLL),('after',NEW_DLL))}
    assert report['dll_sha256']['before'] == '5d145c0f1dd455994d47b4ca0433afb4812347075e673bf3fa653ca75b5d1e44'
    sessions=[];buffers=[];calls=0
    try:
        for m,k,n in ((1,1024,1024),(128,1024,1024),(1024,1024,1024),(2048,512,256)):
            tag = '-'.join(map(str,(m,k,n)))
            with np.load(FIXTURES/f'fixture-{tag}.npz',allow_pickle=False) as fixture:
                expected=fixture['expected'].copy()
                hosts={'fp16-fq':fixture['fp16'].copy(),'prequant-u8':fixture['u8'].copy()}
            for path,host in hosts.items():
                blob = FIXTURES/f'{tag}-{path}.blob'
                prepared={}
                for name,lib in libraries.items():
                    session=lib.create_inference_session(blob.read_bytes(),timeout_ms=10000);sessions.append(session)
                    ib=session.create_buffer(host.nbytes);ob=session.create_buffer(expected.nbytes);buffers.extend((ib,ob))
                    ct.memmove(ib.address,host.ctypes.data,host.nbytes)
                    prepared[name]=(session,ib,ob)
                def run(name):
                    nonlocal calls
                    session,ib,ob=prepared[name]
                    session.infer([('x',ib)],[('y',ob)]);calls+=1
                for name,(session,ib,ob) in prepared.items():
                    run(name)
                    actual=np.empty_like(expected);ct.memmove(actual.ctypes.data,ob.address,actual.nbytes)
                    np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
                    zeros=np.full_like(host,128 if path=='prequant-u8' else 0)
                    ct.memmove(ib.address,zeros.ctypes.data,zeros.nbytes);run(name)
                    ct.memmove(actual.ctypes.data,ob.address,actual.nbytes)
                    np.testing.assert_array_equal(actual,np.zeros_like(expected))
                    ct.memmove(ib.address,host.ctypes.data,host.nbytes);run(name)
                timings=measure({name:lambda name=name:run(name) for name in prepared})
                for value in timings.values(): value['effective_tops']=2*m*k*n/(value['median_ms']*1e9)
                for name,(session,ib,ob) in prepared.items():
                    actual=np.empty_like(expected);ct.memmove(actual.ctypes.data,ob.address,actual.nbytes)
                    np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
                    ib.release();ob.release();session.close()
                row=dict(shape_mkn=[m,k,n],path=path,blob_sha256=sha(blob),all_output_bits_equal=True,
                         changed_input_passed=True,timings=timings,
                         latency_reduction_percent=100*(1-timings['after']['median_ms']/timings['before']['median_ms']))
                report['rows'].append(row)
                print(json.dumps(dict(shape=row['shape_mkn'],path=path,ms={k:v['median_ms'] for k,v in timings.items()},
                                      latency_reduction_percent=row['latency_reduction_percent'])),flush=True)
        report['status']='passed'
    finally:
        for buffer in reversed(buffers):buffer.release()
        for session in reversed(sessions):session.close()
        report.update(successful_graph_calls=calls,all_buffers_released=all(b._released for b in buffers),
                      all_sessions_closed=all(s._closed for s in sessions))
        (OUT/'native.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


def model(kind,label):
    import numpy as np
    import torch
    import torch_npu_runtime
    if kind=='before':
        # Bind the unchanged C++ backend to the saved C runtime with a manifest
        # containing both actual hashes. Keep every normal provenance check.
        manifest=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
        manifest['session_sha256']=sha(OLD_DLL)
        binding=OUT/'before-bound-build.json';binding.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        torch_npu_runtime.SESSION_DLL=OLD_DLL
        source=(ROOT/'torch_intel_npu.py').read_text(encoding='utf-8')
        old='_BUILD = _EXT.with_name("build.json")';assert source.count(old)==1
        source=source.replace(old,f'_BUILD = Path({str(binding)!r})')
        source=source.replace('one list/fence per graph; re-record bindings and reset after completion','list/fence per submission (saved baseline)')
        (OUT/'before-bound-device.py').write_text(source,encoding='utf-8')
        module=types.ModuleType('torch_intel_npu');module.__file__=str(ROOT/'torch_intel_npu.py')
        sys.modules[module.__name__]=module
        exec(compile(source,module.__file__,'exec'),module.__dict__)
    import torch_intel_npu as npu
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from torch_npu_smollm import FrozenSmolStack
    torch.set_num_threads(8)
    directory=ROOT/'models/SmolLM2-135M-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as file:
        embedding=file.get_tensor('model.embed_tokens.weight')
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    report=dict(status='running',runtime=kind,label=label,rows=[],script_sha256=sha(Path(__file__)),
                model_adapter_sha256=sha(ROOT/'torch_npu_smollm.py'),session_sha256=sha(OLD_DLL if kind=='before' else NEW_DLL),
                device_build=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig')),
                scope='Four consecutive SmolLM2-135M decoder layers, real embedding inputs; warm prefill and one-token '
                      'decode; fresh output/KV snapshots; 31 samples; no full LLM generation claim')
    arrays={}
    try:
        with torch.no_grad():
            for batch,prefix,mode in ((1,32,'fp16'),(1,32,'w8a16'),(4,16,'fp16'),(4,16,'w8a16')):
                tokens=ids.repeat(1,(prefix+ids.shape[1])//ids.shape[1])[:,:prefix+1].repeat(batch,1)
                host=embedding[tokens].half().contiguous()
                x=host[:,:prefix].contiguous().to('intel_npu');one=host[:,prefix:].contiguous().to('intel_npu')
                stack=FrozenSmolStack.from_checkpoint(directory,num_layers=4,mode=mode)
                output,seed=stack.prefill(x);decoded,updated=stack.decode(one,seed)
                tag=f'b{batch}-s{prefix}-{mode}'
                snapshots=[output,decoded]
                for cache in (*seed,*updated): snapshots.extend((cache.keys,cache.values,cache._position))
                for index,value in enumerate(snapshots): arrays[f'{tag}-{index}']=value.cpu().numpy().copy()
                del cache,value,snapshots,updated
                before=npu.stats()
                timings=measure({'prefill':lambda:stack.prefill(x),'decode':lambda:stack.decode(one,seed)})
                after=npu.stats()
                for key in ('host_upload_copies','host_download_copies','session_creations'):
                    assert after[key]==before[key],(key,before[key],after[key])
                assert after['npu_calls']-before['npu_calls']==(31+5)*2*4
                preserved=[output,decoded]
                for cache in seed:preserved.extend((cache.keys,cache.values,cache._position))
                del cache
                for index,value in enumerate(preserved):
                    np.testing.assert_array_equal(value.cpu().numpy().view(np.uint8),arrays[f'{tag}-{index}'].view(np.uint8))
                del value,preserved
                row=dict(batch=batch,prefix_tokens=prefix,mode=mode,num_layers=4,timings=timings,
                         calls_per_operation=4,fresh_output_and_kv=True,previous_outputs_and_seed_preserved=True,
                         warm_counter_delta={key:after[key]-before[key]
                         for key in ('npu_calls','buffer_allocations','host_upload_copies','host_download_copies','session_creations')})
                report['rows'].append(row)
                print(json.dumps(dict(runtime=kind,batch=batch,prefix_tokens=prefix,mode=mode,
                                      median_ms={key:value['median_ms'] for key,value in timings.items()})),flush=True)
                del x,one,stack,seed,output,decoded
            gc.collect();assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers'],npu.stats()
            report['stats_before_shutdown']=npu.stats();npu.shutdown();report['final_stats']=npu.stats()
            assert not npu.stats()['context_failed'] and npu.stats()['live_buffer_bytes']==0
        np.savez(OUT/f'{label}-outputs.npz',**arrays)
        report['outputs_sha256']=sha(OUT/f'{label}-outputs.npz');report['status']='passed'
    finally:
        (OUT/f'{label}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


def audit():
    import math
    import numpy as np
    manifest=json.loads((ROOT/'npunlock-probe/session-runtime/build.json').read_text(encoding='utf-8-sig'))
    assert manifest['dllSha256'].lower()==sha(NEW_DLL)
    for record in (*manifest['sources'],*manifest['headers']):
        assert sha(ROOT/'npunlock'/record['path'])==record['sha256'].lower()
    device=json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    assert device['session_sha256']==sha(NEW_DLL)
    assert device['extension_sha256']==sha(ROOT/'npunlock-probe/torch-device/_torch_intel_npu.pyd')
    assert device['source_sha256']==sha(ROOT/'npunlock-probe/torch-device/backend.cpp')
    original=json.loads((OUT/'before/torch-build.json').read_text(encoding='utf-8-sig'))
    assert original['source_sha256']==device['source_sha256'] and original['torch_version']==device['torch_version']
    prior_session=json.loads((OUT/'before/session-build.json').read_text(encoding='utf-8-sig'))
    public_header='include/npunlock/graphinfer.h'
    assert next(r['sha256'].lower() for r in prior_session['headers'] if r['path']==public_header)==sha(ROOT/'npunlock'/public_header)
    binding=json.loads((OUT/'before-bound-build.json').read_text(encoding='utf-8'))
    assert binding['session_sha256']==sha(OLD_DLL) and binding['extension_sha256']==device['extension_sha256']
    raw=json.loads((OUT/'native.json').read_text(encoding='utf-8'))
    assert raw['status']=='passed' and raw['successful_graph_calls']==624
    assert raw['all_buffers_released'] and raw['all_sessions_closed']
    assert raw['script_sha256']==sha(OUT/'native-script.py')
    assert raw['dll_sha256']=={'before':sha(OLD_DLL),'after':sha(NEW_DLL)}
    for row in raw['rows']:
        assert row['all_output_bits_equal'] and row['changed_input_passed']
        tag='-'.join(map(str,row['shape_mkn']))
        assert row['blob_sha256']==sha(FIXTURES/f"{tag}-{row['path']}.blob")
        for value in row['timings'].values():
            assert len(value['samples_ms'])==31 and statistics.median(value['samples_ms'])==value['median_ms']
            assert math.isclose(value['effective_tops'],2*math.prod(row['shape_mkn'])/(value['median_ms']*1e9),rel_tol=1e-12)
    control=json.loads((OUT/'model-control.json').read_text(encoding='utf-8'))
    assert control['status']=='passed' and control['all_output_cache_bits_equal']
    reports=[]
    for label in control['process_order']:
        report=json.loads((OUT/f'{label}.json').read_text(encoding='utf-8'));reports.append(report)
        assert report['status']=='passed' and report['script_sha256']==sha(OUT/'model-script.py')
        assert report['outputs_sha256']==sha(OUT/f'{label}-outputs.npz')
        assert report['session_sha256']==sha(OLD_DLL if report['runtime']=='before' else NEW_DLL)
        assert report['model_adapter_sha256']==sha(ROOT/'torch_npu_smollm.py')
        assert report['final_stats']['live_buffers']==report['final_stats']['live_buffer_bytes']==0
        assert not report['final_stats']['context_failed'] and report['final_stats']['npu_calls']==1248
        for row in report['rows']:
            assert row['previous_outputs_and_seed_preserved'] and row['warm_counter_delta']['npu_calls']==288
            assert all(row['warm_counter_delta'][key]==0 for key in ('host_upload_copies','host_download_copies','session_creations'))
            for value in row['timings'].values():assert statistics.median(value['samples_ms'])==value['median_ms']
    with np.load(OUT/f"{control['process_order'][0]}-outputs.npz",allow_pickle=False) as reference:
        assert len(reference.files)==control['arrays_per_process']==104
        for label in control['process_order'][1:]:
            with np.load(OUT/f'{label}-outputs.npz',allow_pickle=False) as actual:
                assert reference.files==actual.files
                for key in reference.files:np.testing.assert_array_equal(reference[key].view(np.uint8),actual[key].view(np.uint8))
    for row in control['rows']:
        for kind in ('before','after'):
            values=[]
            for report in reports:
                if report['runtime']!=kind:continue
                match=next(r for r in report['rows'] if all(r[key]==row[key]
                           for key in ('batch','prefix_tokens','mode','num_layers')))
                values.append(match['timings'][row['operation']]['median_ms'])
            assert values==row['process_medians_ms'][kind]
            assert statistics.mean(values)==row['mean_process_median_ms'][kind]
        before=row['mean_process_median_ms']['before'];after=row['mean_process_median_ms']['after']
        assert math.isclose(row['latency_reduction_percent'],100*(1-after/before),rel_tol=1e-12)
    regression={}
    for path in sorted((OUT/'regressions').glob('*.json')):
        report=json.loads(path.read_text(encoding='utf-8'));assert report['status']=='passed'
        assert not report['openvino_installed'] and all(c['status']=='passed' for c in report['checks'])
        if 'session_dll_sha256' in report:assert report['session_dll_sha256'].lower()==sha(NEW_DLL)
        if 'extension_sha256' in report:assert report['extension_sha256'].lower()==device['extension_sha256']
        if 'final_stats' in report:
            assert not report['final_stats']['context_failed'] and report['final_stats']['live_buffers']==0
        regression[path.name]=dict(status=report['status'],checks=len(report['checks']))
    assert len(regression)==3
    unit=(OUT/'regressions/unit-session-reuse.log').read_text(encoding='utf-8-sig')
    assert 'session reuse, rebinding, safe failures and failure quarantine: passed' in unit
    snapshot={name:sha(ROOT/name) for name in ('npunlock/src/graphinfer/session_win32.c',
        'npunlock/src/workers/level_zero_min.h','npunlock/tests/unit/test_session_failure.c',
        'torch_npu_runtime.py','torch_intel_npu.py','torch_npu_smollm.py','torch_npu_session_reuse_benchmark.py')}
    (OUT/'audit.json').write_text(json.dumps(dict(status='passed',native_graph_calls=624,model_graph_calls=4992,
        model_output_arrays=104,regressions=regression,dll_sha256=raw['dll_sha256'],source_sha256=snapshot,
        compiled_backend_source_unchanged=True,all_model_output_and_cache_bits_equal=True),indent=2),encoding='utf-8')
    print('provenance, native rates, model bits, lifetimes and regressions: passed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('native','model','model-control','audit'),default='native')
    parser.add_argument('--runtime',choices=('before','after'),default='after');parser.add_argument('--label',default='model-after')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.case=='native':native()
    elif args.case=='audit':audit()
    elif args.case=='model':model(args.runtime,args.label)
    else:
        import numpy as np
        labels=[]
        for index,kind in enumerate(('before','after','after','before')):
            label=f'model-{index}-{kind}';labels.append(label)
            with (OUT/f'{label}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,__file__,'--case','model','--runtime',kind,'--label',label],
                               stdout=log,stderr=log,timeout=300,check=True)
            print((OUT/f'{label}.log').read_text(encoding='utf-8'),flush=True)
        with np.load(OUT/f'{labels[0]}-outputs.npz',allow_pickle=False) as reference:
            for label in labels[1:]:
                with np.load(OUT/f'{label}-outputs.npz',allow_pickle=False) as current:
                    assert current.files==reference.files
                    for name in reference.files:np.testing.assert_array_equal(current[name].view(np.uint8),reference[name].view(np.uint8))
        reports=[json.loads((OUT/f'{label}.json').read_text(encoding='utf-8')) for label in labels]
        rows=[]
        for index in range(4):
            for operation in ('prefill','decode'):
                values={kind:[r['rows'][index]['timings'][operation]['median_ms'] for r in reports if r['runtime']==kind]
                        for kind in ('before','after')}
                medians={kind:statistics.mean(samples) for kind,samples in values.items()}
                row=dict(**{key:reports[0]['rows'][index][key] for key in ('batch','prefix_tokens','mode','num_layers')},
                         operation=operation,process_medians_ms=values,mean_process_median_ms=medians,
                         latency_reduction_percent=100*(1-medians['after']/medians['before']))
                rows.append(row)
        (OUT/'model-control.json').write_text(json.dumps(dict(status='passed',process_order=labels,
            all_output_cache_bits_equal=True,arrays_per_process=len(reference.files),rows=rows,
            scope='ABBA isolated processes; mean of the two per-process medians, not same-process runtime switching'),indent=2),encoding='utf-8')


if __name__=='__main__':main()

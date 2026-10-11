"""Same-graph synchronous fixed-buffer replay experiment; diagnostic DLL only."""
import ctypes as ct
import hashlib
import json
import statistics
import time
from pathlib import Path

from dpu_diagnostic import ROOT, OUT, STAGES, native_setup, compile_graph, metadata


def execute():
    import numpy as np
    import torch
    from npunlock._native import _InferSessionRunResult
    from torch_npu_lowbit import PackedWeight
    from torch_npu_throughput_benchmark import timing
    native=native_setup();dll=native.infer
    dll.diag_replay_create.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t,ct.POINTER(ct.c_void_p),ct.POINTER(_InferSessionRunResult)]
    dll.diag_replay_run.argtypes=[ct.c_void_p,ct.c_uint32,ct.POINTER(_InferSessionRunResult)]
    dll.diag_replay_destroy.argtypes=[ct.c_void_p]
    for name in ('diag_replay_create','diag_replay_run','diag_replay_destroy'):getattr(dll,name).restype=ct.c_uint32
    torch.set_num_threads(8)
    report=dict(status='running',rows=[],source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),ROOT/'npunlock-probe/research/dpu_diagnostic.py',ROOT/'npunlock-probe/research/dpu_diagnostic_extra.c',OUT/'session_diagnostic.dll')},
        scope='same graph and buffers; standard native infer vs cached list/fence; replay16 synchronizes after each of 16 sequential submissions; not parallel batching')
    sessions=[];buffers=[];replays=[];calls=0
    def checked(code,result):
        text=bytes(result.diagnostic.json.data[:result.diagnostic.json.size]).decode('utf-8',errors='replace') if code else ''
        dll.graphinfer_session_infer_result_release(ct.byref(result))
        assert code==0,(code,text)
    try:
        for m,k,n in ((1,1024,1024),(128,1024,1024),(1024,1024,1024),(2048,512,256)):
            torch.manual_seed(731)
            ac=torch.randint(-64,65,(m,k),dtype=torch.int32);wc=torch.randint(-63,64,(n,k),dtype=torch.int32)
            q=PackedWeight((n,k),'int8',k,wc.to(torch.int8).numpy().tobytes(),np.full(n,1/256,dtype=np.float16).tobytes())
            expected=((ac.float()@wc.float().t())/32768).half().numpy()
            host_half=(ac.numpy()/128).astype(np.float16);host_u8=(ac.numpy()+128).astype(np.uint8)
            tag='-'.join(map(str,(m,k,n)))
            np.savez(OUT/f'fixture-{tag}.npz',fp16=host_half,u8=host_u8,expected=expected)
            for prequant in (False,True):
                name='prequant-u8' if prequant else 'fp16-fq'
                program,evidence,xml,data=compile_graph(native,q,m,prequant,False)
                session=native.create_inference_session(program.graph_blob,timeout_ms=10000)
                sessions.append(session)
                args=metadata(native,session)
                ib=session.create_buffer(args[0]['bytes']);ob=session.create_buffer(args[1]['bytes']);buffers.extend((ib,ob))
                host=host_u8 if prequant else host_half
                ct.memmove(ib.address,host.ctypes.data,host.nbytes)
                ni,ni_owners=session._binding_array([('x',ib)]);no,no_owners=session._binding_array([('y',ob)])
                result=_InferSessionRunResult();result.struct_size=ct.sizeof(result)
                def normal():
                    nonlocal calls
                    checked(dll.graphinfer_session_infer(session._result.session,ni,1,no,1,ct.byref(result)),result);calls+=1
                normal()
                replay=ct.c_void_p()
                checked(dll.diag_replay_create(session._result.session,ni,1,no,1,ct.byref(replay),ct.byref(result)),result)
                replays.append(replay)
                def cached(count=1):
                    nonlocal calls
                    checked(dll.diag_replay_run(replay,count,ct.byref(result)),result);calls+=count
                def validate():
                    actual=np.empty_like(expected);ct.memmove(actual.ctypes.data,ob.address,actual.nbytes)
                    np.testing.assert_array_equal(actual,expected)
                cached();validate()
                zeros=np.full_like(host,128 if prequant else 0)
                ct.memmove(ib.address,zeros.ctypes.data,zeros.nbytes);cached()
                actual=np.empty_like(expected);ct.memmove(actual.ctypes.data,ob.address,actual.nbytes)
                np.testing.assert_array_equal(actual,np.zeros_like(expected))
                ct.memmove(ib.address,host.ctypes.data,host.nbytes);cached();validate()
                runs={'standard':normal,'replay1':cached,'replay16':lambda:cached(16)}
                samples={key:[] for key in runs};stages={key:[] for key in runs}
                for run in runs.values():
                    for _ in range(5):run()
                for round in range(31):
                    keys=tuple(runs);order=keys[round%3:]+keys[:round%3]
                    if round%2:order=tuple(reversed(order))
                    for key in order:
                        start=time.perf_counter_ns();runs[key]();elapsed=(time.perf_counter_ns()-start)/1e6
                        count=16 if key=='replay16' else 1;samples[key].append(elapsed/count)
                        times=(ct.c_uint64*10)();assert dll.diag_times(session._result.session,times,10)==0 and not times[9]
                        stages[key].append({name:times[i]/count for i,name in enumerate(STAGES)})
                validate()
                row=dict(shape_mkn=[m,k,n],path=name,evidence=evidence,arguments=args,changed_input_and_restore_passed=True,
                    timings={key:dict(**timing(values,2*m*k*n),stage_median_ms={stage:statistics.median(v[stage] for v in stages[key])/1e6 for stage in STAGES}) for key,values in samples.items()})
                ib.release()
                code=dll.diag_replay_run(replay,1,ct.byref(result));assert code!=0
                dll.graphinfer_session_infer_result_release(ct.byref(result));row['released_input_rejected']=True
                assert dll.diag_replay_destroy(replay)==0;replays.pop()
                ob.release();session.close()
                report['rows'].append(row)
                print(json.dumps(dict(shape=[m,k,n],path=name,ms={key:value['median_ms'] for key,value in row['timings'].items()},tops={key:value['effective_teraops_per_second'] for key,value in row['timings'].items()})),flush=True)
        report['status']='passed'
    finally:
        for replay in replays:dll.diag_replay_destroy(replay)
        for buffer in reversed(buffers):buffer.release()
        for session in reversed(sessions):session.close()
        report['successful_graph_calls']=calls
        report['all_buffers_released']=all(v._released for v in buffers)
        report['all_sessions_closed']=all(v._closed for v in sessions)
        (OUT/'replay.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return 0


if __name__=='__main__':raise SystemExit(execute())

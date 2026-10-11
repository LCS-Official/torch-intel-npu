"""Isolated DPU quantization, host-stage and device profiling diagnostics."""
import argparse
import ctypes as ct
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/dpu-diagnostic'
STAGES = ('set_arguments','list_create','fence_create','append_graph','list_close','queue_submit','fence_wait','list_fence_destroy','execute_total')
SHAPES = ((1,1024,1024),(128,1024,1024),(512,1024,1024),(1024,1024,1024),
          (256,2048,512),(512,2048,512),(2048,512,512),(2048,512,256))


class Extension(ct.Structure):
    _fields_=[('name',ct.c_char*256),('version',ct.c_uint32)]


class Argument(ct.Structure):
    _fields_=[('index',ct.c_uint32),('type',ct.c_uint32),('precision',ct.c_uint32),('dims_count',ct.c_uint32),
              ('dims',ct.c_uint32*5),('name',ct.c_char*256),('name_size',ct.c_size_t),('data_size',ct.c_size_t)]


class Task(ct.Structure):
    _fields_=[('name',ct.c_char*256),('layer_type',ct.c_char*50),('exec_type',ct.c_uint32),
              ('start_time_ns',ct.c_uint64),('duration_ns',ct.c_uint64),('active_cycles',ct.c_uint32),
              ('stall_cycles',ct.c_uint32),('task_id',ct.c_uint32),('parent_layer_id',ct.c_uint32)]


def decode(value):
    return value.split(b'\0')[0].decode('utf-8',errors='replace')


def native_setup():
    import npunlock as s
    native=s.NativeLibraries(session_library=OUT/'session_diagnostic.dll')
    dll=native.infer
    dll.diag_extensions.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint32),ct.POINTER(Extension)]
    dll.diag_argument.argtypes=[ct.c_void_p,ct.c_uint32,ct.POINTER(Argument)]
    dll.diag_times.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint64),ct.c_uint32]
    dll.diag_profile_enable.argtypes=[ct.c_void_p]
    dll.diag_profile_data.argtypes=[ct.c_void_p,ct.c_uint32,ct.POINTER(ct.c_uint32),ct.POINTER(ct.c_uint8)]
    for name in ('diag_extensions','diag_argument','diag_times','diag_profile_enable','diag_profile_data'):
        getattr(dll,name).restype=ct.c_uint32
    return native


def make_graph(q,m,prequant=False):
    import numpy as np
    import npunlock as s
    from torch_npu_lowbit import weight_graph
    graph,xml,data=weight_graph(q,m,activation_scale=1/128)
    if not prequant:
        return graph,xml,data
    # Keep the exact existing quantized weight and FQ contract. Replace only
    # the activation FQ output with a U8 dequantization graph. The compiler's
    # DPU descriptor, actual input metadata and numeric output are checked.
    n,k=q.shape
    x=s.input('x',shape=(m,k),dtype='u8')
    cv=s.op('Convert',x,destination_type='f16',_shape=x.shape,_dtype='f16',_name='activation_codes_to_half')
    zero=s.constant(np.float16(128))
    cv=s.op('Subtract',cv,zero,auto_broadcast='numpy',_shape=x.shape,_dtype='f16')
    cv=s.op('Multiply',cv,s.constant(np.float16(1/128)),auto_broadcast='numpy',_shape=x.shape,_dtype='f16',_name='activation_dequantize')
    y=graph.outputs[0]
    assert y.producer.op=='MatMul' and y.producer.inputs[0].producer.op=='FakeQuantize'
    y.producer.inputs=(cv,y.producer.inputs[1])
    graph=s.Graph([x],[y],name='prequant_u8_linear')
    ir=s.serialize_ir(graph)
    return graph,ir.xml,ir.weights


def compile_graph(native,q,m,prequant=False,profile=False):
    import npunlock as s
    from torch_npu_lowbit import dpu_modes
    from npunlock.npu4000 import Elf
    graph,xml,data=make_graph(q,m,prequant)
    flags='--config EXECUTION_MODE_HINT="ACCURACY"'
    if profile:flags+=' PERF_COUNT="YES"'
    result=native.compile_ir(xml,data,build_flags=flags,timeout_ms=20000)
    assert (result.vendor_id,result.device_id,tuple(result.compiler_version))==(0x8086,0x643e,(8,3))
    modes=dpu_modes(result.graph_blob)
    matrices=[v for v in modes if not v['elementwise']]
    assert matrices and all((v['amode'],v['wmode'])==(1,1) for v in matrices),matrices
    program=s.load_native(result.graph_blob,graph=graph,libraries=native)
    elf=Elf(result.graph_blob)
    detail=dict(flags=flags,blob_sha256=hashlib.sha256(result.graph_blob).hexdigest(),dpu_modes=modes,
                sections={prefix:[dict(name=z['name'],size=z['size']) for z in elf.sections if z['size'] and z['name'].startswith(prefix)]
                          for prefix in ('task.dpu.invariant.','task.dpu.variant.','task.shave.invocation.','task.dma.','profiling')})
    return program,detail,xml,data


def metadata(native,session):
    values=[]
    for index in range(2):
        value=Argument()
        assert native.infer.diag_argument(session._result.session,index,ct.byref(value))==0
        values.append(dict(index=value.index,type=value.type,precision=value.precision,dims=list(value.dims)[:value.dims_count],
                           name=decode(value.name),bytes=value.data_size))
    return values


def execute(shape,profile=False):
    import numpy as np
    import torch
    from torch_npu_lowbit import PackedWeight
    from torch_npu_throughput_benchmark import timing
    native=native_setup()
    m,k,n=shape
    tag=('profile-' if profile else '')+'-'.join(map(str,shape))
    report=dict(status='running',shape_mkn=list(shape),profile=profile,checks=[],timings={},
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                  ('npunlock-probe/research/dpu_diagnostic.py','npunlock-probe/research/build_dpu_diagnostic.py',
                   'npunlock-probe/research/dpu_diagnostic_extra.c','npunlock/src/graphinfer/session_win32.c',
                   'npunlock-probe/research/results/dpu-diagnostic/session_diagnostic.dll','torch_npu_lowbit.py')},
                host_stage_scope='QueryPerformanceCounter inside copied runtime; fence wait includes driver/device/OS wait, not pure device execution',
                counter_scope='diagnostic session API bypasses Torch counters; successful graph executions counted by this script')
    sessions=[];buffers=[];calls=0
    torch.set_num_threads(8);torch.manual_seed(731)
    ac=torch.randint(-64,65,(m,k),dtype=torch.int32)
    wc=torch.randint(-63,64,(n,k),dtype=torch.int32)
    q=PackedWeight((n,k),'int8',k,wc.to(torch.int8).numpy().tobytes(),np.full(n,1/256,dtype=np.float16).tobytes())
    expected=((ac.float()@wc.float().t())/32768).half().numpy()
    selected=sorted(set((0,m//2,m-1)))
    independent=ac.numpy()[selected].astype(np.int64)@wc.numpy().astype(np.int64).T
    np.testing.assert_array_equal(expected[selected],(independent/32768).astype(np.float16))

    def run(session,ib,ob):
        nonlocal calls
        session.infer([('x',ib)],[('y',ob)])
        calls+=1

    try:
        for prequant in (False,True):
            name='prequant-u8' if prequant else 'fp16-fq'
            program,detail,xml,data=compile_graph(native,q,m,prequant,profile)
            (OUT/f'{tag}-{name}.blob').write_bytes(program.graph_blob)
            (OUT/f'{tag}-{name}.xml').write_bytes(xml.encode() if isinstance(xml,str) else xml)
            (OUT/f'{tag}-{name}.bin').write_bytes(data)
            session=native.create_inference_session(program.graph_blob,timeout_ms=10000,parent=sessions[0] if sessions else None)
            sessions.append(session)
            arguments=metadata(native,session)
            assert arguments[0]['precision']==(4 if prequant else 2) and arguments[0]['bytes']==m*k*(1 if prequant else 2),arguments
            assert arguments[1]['precision']==2 and arguments[1]['bytes']==m*n*2,arguments
            if len(sessions)==1:
                count=ct.c_uint32()
                assert native.infer.diag_extensions(session._result.session,ct.byref(count),None)==0
                values=(Extension*count.value)()
                assert native.infer.diag_extensions(session._result.session,ct.byref(count),values)==0
                report['extensions']=[dict(name=decode(v.name),version=v.version) for v in values]
            ib=session.create_buffer(arguments[0]['bytes']);ob=session.create_buffer(arguments[1]['bytes'])
            buffers.extend((ib,ob))
            host=(ac.numpy()+128).astype(np.uint8) if prequant else (ac.numpy()/128).astype(np.float16)
            ct.memmove(ib.address,host.ctypes.data,host.nbytes)
            if profile:
                code=native.infer.diag_profile_enable(session._result.session)
                detail['profile_enable_result']=hex(code)
                assert code==0,detail
            run(session,ib,ob)
            actual=np.empty((m,n),dtype=np.float16)
            ct.memmove(actual.ctypes.data,ob.address,actual.nbytes)
            np.testing.assert_allclose(actual,expected,rtol=3e-3,atol=3e-4)
            np.testing.assert_array_equal(actual,expected)
            detail.update(validation_passed=True,bit_exact_grid=True,arguments=arguments)
            report['checks'].append(dict(path=name,**detail))
            report.setdefault('prepared',{})[name]=(session,ib,ob)
        runs=report.pop('prepared')
        wall={name:[] for name in runs};stages={name:[] for name in runs}
        for session,ib,ob in runs.values():
            for _ in range(5):run(session,ib,ob)
        for round in range(31 if not profile else 3):
            names=tuple(runs) if round%2==0 else tuple(reversed(runs))
            for name in names:
                session,ib,ob=runs[name]
                start=time.perf_counter_ns();run(session,ib,ob);wall[name].append((time.perf_counter_ns()-start)/1e6)
                times=(ct.c_uint64*10)()
                assert native.infer.diag_times(session._result.session,times,10)==0 and not times[9]
                assert sum(times[:8])<=times[8]+1000,list(times)
                stages[name].append(dict(zip(STAGES,times[:9])))
        report['timings']={name:dict(**timing(samples,2*m*k*n),host_stage_median_ms={key:statistics.median(v[key] for v in stages[name])/1e6 for key in STAGES},stage_samples_ns=stages[name]) for name,samples in wall.items()}
        if profile:
            for name,(session,ib,ob) in runs.items():
                size=ct.c_uint32()
                code=native.infer.diag_profile_data(session._result.session,2,ct.byref(size),None)
                entry=dict(size_result=hex(code),bytes=size.value,task_record_bytes=ct.sizeof(Task))
                if code==0 and 0<size.value<=16*1024*1024:
                    raw=(ct.c_uint8*size.value)()
                    code=native.infer.diag_profile_data(session._result.session,2,ct.byref(size),raw)
                    entry['data_result']=hex(code)
                    data=bytes(raw)[:size.value];(OUT/f'{tag}-{name}-tasks.bin').write_bytes(data)
                    if code==0 and len(data)%ct.sizeof(Task)==0:
                        tasks=[]
                        for offset in range(0,len(data),ct.sizeof(Task)):
                            v=Task.from_buffer_copy(data[offset:offset+ct.sizeof(Task)])
                            tasks.append(dict(name=decode(v.name),layer_type=decode(v.layer_type),exec_type=v.exec_type,start_time_ns=v.start_time_ns,duration_ns=v.duration_ns,active_cycles=v.active_cycles,stall_cycles=v.stall_cycles,task_id=v.task_id))
                        entry['tasks']=tasks
                        entry['aggregate_ms']={str(kind):sum(v['duration_ns'] for v in tasks if v['exec_type']==kind)/1e6 for kind in (1,2,3)}
                        entry['timeline_span_ms']=(max(v['start_time_ns']+v['duration_ns'] for v in tasks)-min(v['start_time_ns'] for v in tasks))/1e6 if tasks else 0
                report.setdefault('device_profiling',{})[name]=entry
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        report.pop('prepared',None)
        print(report['traceback'],flush=True)
    finally:
        for buffer in reversed(buffers):buffer.release()
        for session in reversed(sessions):session.close()
        report['successful_graph_calls']=calls
        report['released_buffers']=len(buffers);report['closed_sessions']=len(sessions)
        (OUT/f'{tag}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(tag=tag,status=report['status'],timings={name:dict(ms=value['median_ms'],tops=value['effective_teraops_per_second'],stages=value['host_stage_median_ms']) for name,value in report['timings'].items()},calls=calls)),flush=True)
    return int(report['status']!='passed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');parser.add_argument('--profile',action='store_true');parser.add_argument('--shape',type=int,nargs=3)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.child:return execute(tuple(args.shape),args.profile)
    shapes=(tuple(args.shape),) if args.shape else ((128,1024,1024),) if args.profile else SHAPES
    for shape in shapes:
        tag=('profile-' if args.profile else '')+'-'.join(map(str,shape));print('starting '+tag,flush=True)
        with (OUT/f'{tag}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--child','--shape',*map(str,shape),*(['--profile'] if args.profile else [])],stdout=log,stderr=log,timeout=240)
        print((OUT/f'{tag}.log').read_text(encoding='utf-8'),flush=True)
        if result.returncode:return result.returncode
    return 0


if __name__=='__main__':raise SystemExit(main())

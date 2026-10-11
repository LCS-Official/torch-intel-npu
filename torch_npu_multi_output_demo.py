"""Isolated multi-output binding, alias/lifetime and legacy single-output checks."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/smollm-kv-multi'

def execute():
    import numpy as np
    import torch
    import npunlock as s
    import torch_intel_npu as npu
    report=dict(status='running',checks=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ['torch_npu_multi_output_demo.py','torch_intel_npu.py','npunlock-probe/torch-device/backend.cpp',
         'npunlock-probe/torch-device/_torch_intel_npu.pyd','npunlock-probe/session-runtime/npunlock_session.dll']})
    try:
        def work():
            shape=(2,17);half=s.input('half_input',shape=shape,dtype='f16');single=s.input('float_input',shape=shape,dtype='f32')
            def output(value,shape,name):
                return s.op('Reshape',value,s.constant(np.array(shape,dtype=np.int64)),special_zero=False,
                            _shape=shape,_dtype=value.dtype,_name=name)
            graph=s.Graph([half,single],[output(half,shape,'half_output'),output(single,shape,'float_output'),
                output(half,(34,),'half_copy')],name='multi_output_binding')
            ir=s.serialize_ir(graph);result=npu._native.compile_ir(ir.xml,ir.weights,timeout_ms=20000)
            program=s.load_native(result.graph_blob,graph=graph,libraries=npu._native)
            session=npu._native.create_inference_session(result.graph_blob,timeout_ms=10000,parent=npu._root_session)
            key=('multi_output_binding',hashlib.sha256(ir.xml+ir.weights).digest());npu._sessions[key]=session,program;npu._session_creations+=1
            specs=[(shape,torch.float16),(shape,torch.float32),((34,),torch.float16)]
            host16=torch.linspace(-2,2,34).half().view(shape)
            host32=torch.tensor([70000.,1.0000001192092896,2**-140,-2**-149,0.,-0.,float('inf'),-float('inf')]*4+[1.,-1.]).view(shape)
            inputs=[host16.to('intel_npu'),host32.to('intel_npu')]
            outputs=npu._run_many(key,inputs,specs)
            def check_values(values):
                for actual,expected in zip(values,[host16,host32,host16.reshape(34)]):
                    integer=torch.int16 if expected.dtype==torch.float16 else torch.int32
                    assert torch.equal(actual.cpu().view(integer),expected.view(integer))
            check_values(outputs)
            report['checks'].append(dict(name='three mixed shape/dtype outputs',status='passed',fp32_bits_preserved=True))
            addresses=[value.data_ptr() for value in outputs];before=npu.stats()
            for _ in range(50):assert npu._run_many(key,inputs,specs,outputs)==outputs
            after=npu.stats()
            for field in ['buffer_allocations','host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
            assert after['npu_calls']-before['npu_calls']==50 and addresses==[value.data_ptr() for value in outputs]
            check_values(outputs);report['checks'].append(dict(name='50 resident reused-output calls',status='passed',allocations=0,host_copies=0))
            host16=host16.flip(-1).contiguous();inputs[0].copy_(host16);npu._run_many(key,inputs,specs,outputs);check_values(outputs)
            report['checks'].append(dict(name='updated input is observed',status='passed'))
            cpu_out=outputs[0].cpu();bad_dtype=torch.empty(shape,dtype=torch.float16,device='intel_npu')
            noncontiguous=torch.empty((17,2),dtype=torch.float16,device='intel_npu').t()
            def reject(name,fn):
                a=npu.stats()
                try:fn()
                except (ValueError,RuntimeError) as error:
                    b=npu.stats()
                    for field in ['npu_calls','session_creations','host_upload_copies','host_download_copies']:assert a[field]==b[field],(name,field)
                    assert not b['context_failed'];report['checks'].append(dict(name=name,status='rejected_before_execution',error=str(error)))
                else:raise AssertionError(('missing rejection',name))
            reject('empty output specs',lambda:npu._run_many(key,inputs,[]))
            reject('wrong output count',lambda:npu._run_many(key,inputs,specs[:2]))
            reject('wrong outs count',lambda:npu._run_many(key,inputs,specs,outputs[:2]))
            reject('CPU output',lambda:npu._run_many(key,inputs,specs,(cpu_out,*outputs[1:])))
            reject('wrong output dtype',lambda:npu._run_many(key,inputs,specs,(outputs[0],bad_dtype,outputs[2])))
            reject('noncontiguous output',lambda:npu._run_many(key,inputs,specs,(noncontiguous,*outputs[1:])))
            reject('output/input alias',lambda:npu._run_many(key,inputs,specs,(inputs[0],*outputs[1:])))
            reject('output/output alias',lambda:npu._run_many(key,inputs,specs,(outputs[0],outputs[1],outputs[0].view(34))))
            names=[value.name for value in graph.inputs];out_names=[value.name for value in graph.outputs];pointer=session._result.session
            reject('native outs/name count',lambda:npu._C.run_many(pointer,inputs,names,outputs,out_names[:2]))
            reject('native output/input alias',lambda:npu._C.run_many(pointer,inputs,names,(inputs[0],*outputs[1:]),out_names))
            reject('native output/output alias',lambda:npu._C.run_many(pointer,inputs,names,(outputs[0],outputs[1],outputs[0].view(34)),out_names))
            reject('native duplicate output name',lambda:npu._C.run_many(pointer,inputs,names,outputs,[out_names[0],out_names[1],out_names[0]]))
            reject('legacy single binding for multi graph',lambda:npu._C.run(pointer,inputs,names,outputs[0],out_names[0]))
            check_values(npu._run_many(key,inputs,specs,outputs))
            # The shared executor also serves the unchanged single-output API.
            addend=torch.full(shape,.25).to('intel_npu');actual=npu._run(('add','f32',shape),[inputs[1],addend],shape,torch.float32).cpu()
            torch.testing.assert_close(actual,host32+.25,rtol=0,atol=0)
            report['checks'].append(dict(name='legacy genuine FP32 add',status='passed'))
        with torch.inference_mode():work()
        gc.collect();report['stats_before_shutdown']=npu.stats();assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown();report['final_stats']=npu.stats();assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'bindings.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2),flush=True)
    return int(report['status']!='passed')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.child:return execute()
    with (OUT/'bindings.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--child'],stdout=log,stderr=log,timeout=300)
    print((OUT/'bindings.log').read_text(encoding='utf-8'),flush=True);return result.returncode

if __name__=='__main__':raise SystemExit(main())

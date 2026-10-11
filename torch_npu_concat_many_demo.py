"""Byte-exact multi-concat, output reuse, binding bounds and recovery checks."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/smollm-kv-profile'

def execute():
    import numpy as np
    import torch
    import torch_intel_npu as npu
    report=dict(status='running',checks=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ['torch_npu_concat_many_demo.py','torch_intel_npu.py','npunlock/python/npunlock/npu4000.py',
         'npunlock-probe/torch-device/_torch_intel_npu.pyd','npunlock-probe/session-runtime/npunlock_session.dll']})
    try:
        def work():
            for dtype,integer in [(torch.float16,torch.int16),(torch.float32,torch.int32)]:
                bits=([0x0000,0x8000,0x0001,0x83ff,0x3c01,0x7c00,0xfc00,0x7e15] if dtype==torch.float16 else
                    [0x00000000,0x80000000,0x00000001,0x807fffff,0x3f800001,0x4788b800,0x7f800000,0xff800000,0x7fc01234])
                host=torch.from_numpy(np.array(bits,dtype=np.uint16 if dtype==torch.float16 else np.uint32).view(np.float16 if dtype==torch.float16 else np.float32).copy())
                cpu=[[host[:2],host[2:5]],[host[5:6],host[6:],host[:1]]]
                groups=[[t.to('intel_npu') for t in group] for group in cpu];shapes=[(sum(t.numel() for t in group),) for group in groups]
                outputs=npu._concat_many(groups,shapes,dtype)
                def check():
                    for actual,expected in zip(outputs,[torch.cat(group) for group in cpu]):
                        assert torch.equal(actual.cpu().view(integer),expected.view(integer))
                check();addresses=[t.data_ptr() for t in outputs];before=npu.stats()
                for _ in range(50):npu._concat_many(groups,shapes,dtype,outputs)
                after=npu.stats()
                for field in ['buffer_allocations','host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                assert after['npu_calls']-before['npu_calls']==50 and addresses==[t.data_ptr() for t in outputs];check()
                cpu[0][0]=host[:2].flip(0);groups[0][0].copy_(cpu[0][0]);npu._concat_many(groups,shapes,dtype,outputs);check()
                report['checks'].append(dict(dtype=str(dtype),name='bit patterns and 50 reused-output calls',status='passed',allocations=0,host_copies=0))
            # Real B8 cache dimensions exceed a compute tile; DMA retains the
            # existing 1,048,576-element storage bound rather than numeric tiling.
            cpu=[[torch.arange(size,dtype=torch.float32).remainder(31).half().add(i) for i in range(8)] for size in [576,12288,12288]]
            groups=[[value.to('intel_npu') for value in group] for group in cpu];shapes=[(8,1,576),(8,3,64,64),(8,3,64,64)]
            outputs=npu._concat_many(groups,shapes,torch.float16)
            for actual,expected in zip(outputs,[torch.cat(group).view(shape) for group,shape in zip(cpu,shapes)]):
                assert torch.equal(actual.cpu().view(torch.int16),expected.view(torch.int16))
            report['checks'].append(dict(name='B8 three independent outputs',status='passed',bindings=27))
            def reject(name,fn):
                before=npu.stats()
                try:fn()
                except (ValueError,RuntimeError) as error:
                    after=npu.stats()
                    for field in ['npu_calls','session_creations','host_upload_copies','host_download_copies']:assert before[field]==after[field],(name,field)
                    assert not after['context_failed'];report['checks'].append(dict(name=name,status='rejected_before_execution',error=str(error)))
                else:raise AssertionError(('missing rejection',name))
            reject('empty groups',lambda:npu._concat_many([],[],torch.float16))
            reject('one group',lambda:npu._concat_many(groups[:1],shapes[:1],torch.float16))
            reject('one tile',lambda:npu._concat_many([[groups[0][0]],*groups[1:]],shapes,torch.float16))
            reject('wrong output count',lambda:npu._concat_many(groups,shapes[:2],torch.float16))
            reject('wrong shape total',lambda:npu._concat_many(groups,[(1,),*shapes[1:]],torch.float16))
            reject('bool dimension',lambda:npu._concat_many(groups,[(True,1,4608),*shapes[1:]],torch.float16))
            reject('rank five',lambda:npu._concat_many(groups,[(1,1,1,1,4608),*shapes[1:]],torch.float16))
            reject('binding limit',lambda:npu._concat_many([groups[0]*4]*2,[(32*576,)]*2,torch.float16))
            reject('storage limit',lambda:npu._concat_many([[outputs[1]]*12,groups[0][:2]],[(12*98304,),(1152,)],torch.float16))
            reject('unsupported dtype',lambda:npu._concat_many(groups,shapes,torch.int32))
            reject('wrong input dtype',lambda:npu._concat_many(groups,shapes,torch.float32))
            reject('CPU input',lambda:npu._concat_many([[cpu[0][0],*groups[0][1:]],*groups[1:]],shapes,torch.float16))
            partial=groups[0][0][:32];reject('partial allocation',lambda:npu._concat_many([[partial,*groups[0][1:]],*groups[1:]],shapes,torch.float16))
            permutation=groups[0][0].view(24,24).t();reject('noncontiguous input',lambda:npu._concat_many([[permutation,*groups[0][1:]],*groups[1:]],shapes,torch.float16))
            reject('wrong outs count',lambda:npu._concat_many(groups,shapes,torch.float16,outputs[:2]))
            reject('output alias',lambda:npu._concat_many(groups,shapes,torch.float16,(outputs[0],outputs[1],outputs[1])))
            npu._concat_many(groups,shapes,torch.float16,outputs)
            x=torch.tensor([1.0000001192092896,70000.,2**-140]).to('intel_npu');y=torch.tensor([.25,1.,2**-140]).to('intel_npu')
            actual=npu._run(('add','f32',(3,)),[x,y],(3,),torch.float32).cpu()
            torch.testing.assert_close(actual,x.cpu()+y.cpu(),rtol=0,atol=0)
            report['checks'].append(dict(name='recovery and legacy genuine FP32 add',status='passed'))
        with torch.inference_mode():work()
        gc.collect();report['stats_before_shutdown']=npu.stats();assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown();report['final_stats']=npu.stats();assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'concat.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report['status'],flush=True)
    return int(report['status']!='passed')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.child:return execute()
    with (OUT/'concat.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--child'],stdout=log,stderr=log,timeout=300)
    print((OUT/'concat.log').read_text(encoding='utf-8'),flush=True);return result.returncode

if __name__=='__main__':raise SystemExit(main())

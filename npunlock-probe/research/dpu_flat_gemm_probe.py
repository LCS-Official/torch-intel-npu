"""Fairness check: four bounded I/O tensors around one internal 4096-row GEMM."""
import gc
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch_npu_gemm_energy as bench


def execute():
    import numpy as np
    import torch
    import torch_intel_npu as npu
    import npunlock as s
    from torch_npu_lowbit import dpu_modes
    torch.set_num_threads(8)
    data=np.load(bench.OUT/'fixture.npz')
    a,w,expected=(data[k] for k in ('a','w','expected'))
    report=dict(status='running',rows=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    run=validate=None
    try:
        with torch.inference_mode():
            for precision in ('float16','int8'):
                xs=[s.input(f'x{i}',shape=(1024,1024),dtype='f16') for i in range(4)]
                joined=s.op('Concat',*xs,axis=0,_shape=(4096,1024),_dtype='f16')
                weight=s.constant((w.astype(np.float32)/256).astype(np.float16))
                def fq(t,low,high):
                    lo,hi=(s.constant(np.float16(v)) for v in (low,high))
                    return s.op('FakeQuantize',t,lo,hi,lo,hi,levels=256,auto_broadcast='numpy',_shape=t.shape,_dtype='f16')
                if precision=='int8':
                    joined=fq(joined,-1,127/128)
                    weight=fq(weight,-128/256,127/256)
                result=s.op('MatMul',joined,weight,transpose_a=False,transpose_b=True,_shape=(4096,1024),_dtype='f16')
                ys=s.op('Split',result,s.constant(np.int64(0)),num_splits=4,
                        _outputs=[((1024,1024),'f16')]*4,_name='split_result')
                ys=[s.op('Reshape',t,s.constant(np.array([1024,1024],dtype=np.int64)),special_zero=False,
                         _shape=(1024,1024),_dtype='f16',_name=f'y{i}') for i,t in enumerate(ys)]
                graph=s.Graph(xs,ys,name='flat_gemm_probe')
                ir=s.serialize_ir(graph)
                compiled=npu._native.compile_ir(ir.xml.encode() if isinstance(ir.xml,str) else ir.xml,ir.weights,
                    build_flags=bench.FLAGS['tiles6'],timeout_ms=30000)
                modes=[v for v in dpu_modes(compiled.graph_blob) if not v['elementwise']]
                wanted=(1,1) if precision=='int8' else (0,0)
                assert modes and all((v['amode'],v['wmode'])==wanted for v in modes)
                program=s.load_native(compiled.graph_blob,graph=graph,libraries=npu._native)
                session=npu._native.create_inference_session(program.graph_blob,timeout_ms=10000,parent=npu._root_session)
                npu._sessions['flat-probe',precision]=session,program
                npu._session_creations+=1
                x=[torch.from_numpy(v.copy()).float().div(128).half().to('intel_npu') for v in a]
                y=[torch.empty((1024,1024),device='intel_npu',dtype=torch.float16) for _ in range(4)]
                sessions=[session._result.session]*16
                inputs,outputs=[x]*16,[y]*16
                names=[[t.name for t in graph.inputs]]*16
                outnames=[[t.name for t in graph.outputs]]*16
                def run():
                    with npu._lock:
                        npu._C.run_chain(sessions,inputs,names,outputs,outnames,1)
                def validate():
                    errors=[]
                    for i,t in enumerate(y):
                        actual=t.cpu();reference=torch.from_numpy(expected[i].copy())
                        torch.testing.assert_close(actual,reference,rtol=.003,atol=.0003)
                        errors.append(float((actual.float()-reference.float()).norm()/reference.float().norm()))
                    assert max(errors)<.001
                    return dict(relative_l2_max=max(errors))
                meta=dict(m=1024,k=1024,n=1024,groups=4,layout='fused4',profile='tiles6',precision=precision,
                          graph_structure='Concat-MatMul-Split',dpu_modes=modes)
                row=bench.measured(run,validate,64,meta,1,npu)
                report['rows'].append(row)
                print(precision,row['teraops'],flush=True)
                run=validate=None
                x=y=inputs=outputs=None
                gc.collect()
        report['status']='passed'
    finally:
        run=validate=None
        x=y=inputs=outputs=None
        gc.collect()
        npu.shutdown()
        report['final_stats']=npu.stats()
        bench.save('flat-probe',report)
        assert report['final_stats']['closed'] and not report['final_stats']['context_failed']
        assert all(report['final_stats'][k]==0 for k in ('live_buffers','live_buffer_bytes','cached_sessions',
            'cached_scalar_buffers','cached_buffer_blocks','quarantined_buffer_bytes'))


if __name__=='__main__':
    execute()

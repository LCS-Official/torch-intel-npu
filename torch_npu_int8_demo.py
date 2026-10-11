"""Isolated INT8 driver probes and validation; no OpenVINO runtime."""
import argparse
import gc
import hashlib
import json
import statistics
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/int8'


def execute(case):
    if case.startswith('lowbit-'):
        import torch_npu_lowbit_demo as previous
        previous.OUT=OUT/'regression';previous.OUT.mkdir(parents=True,exist_ok=True)
        return previous.execute(case.removeprefix('lowbit-'))
    import numpy as np
    import torch
    import torch_intel_npu as npu
    import npunlock as s
    from torch_npu_lowbit import PackedWeight,weight_graph,compile_weight,dpu_modes,QuantizedLinear,quantize_weight
    report=dict(case=case,status='running',checks=[],sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        for p in ['torch_npu_lowbit.py','torch_npu_int8_demo.py']})
    torch.manual_seed(841)
    def fq_reference(values,scales):
        low=(-128*scales).astype(np.float16).astype(np.float64)
        high=(127*scales).astype(np.float16).astype(np.float64)
        grid=(high-low)/255
        codes=np.clip(np.rint((values-low)/grid),0,255)
        return low+codes*grid
    def linear_reference(q,host,scale,bias=None):
        if scale is None:
            codes=np.frombuffer(q.data,dtype=np.int8).reshape(q.grouped_shape).astype(np.float64)
            scales=np.frombuffer(q.scales,dtype=np.float16).astype(np.float64).reshape(*q.grouped_shape[:2],1)
            result=host.double()@torch.from_numpy((codes*scales).reshape(q.shape)).t()
        else:
            # DPU uses an integer zero point (128 for U8), so weight scales are
            # (FP16(high)-FP16(low))/255 with zero aligned to code 128. The
            # ideal floating FQ formula may put zero between codes instead.
            scales=np.frombuffer(q.scales,dtype=np.float16).astype(np.float64)
            low=(-128*scales).astype(np.float16).astype(np.float64)
            high=(127*scales).astype(np.float16).astype(np.float64)
            inp=np.clip(np.rint(host.numpy().astype(np.float64)/scale),-128,127).astype(np.int64)
            wt=np.frombuffer(q.data,dtype=np.int8).reshape(q.shape).astype(np.int64)
            accum=np.einsum('mk,nk->mn',inp,wt,optimize=False)
            result=torch.from_numpy(accum.astype(np.float64)*scale*((high-low)/255))
        if bias is not None:result=result.half().double()+bias.half().double()
        return result
    def check(q,host,*,scale=None,bias=None,compact=False,transpose=False,exact=False,label='random'):
        layer=QuantizedLinear(q,bias=bias,compact_groups=compact,activation_scale=scale)
        x=host.t().contiguous().to('intel_npu').t() if transpose else host.to('intel_npu')
        before=npu.stats();actual=layer(x).cpu();after=npu.stats()
        reference=linear_reference(q,host,layer.activation_scale,bias)
        if exact:np.testing.assert_array_equal(actual.numpy().view(np.uint16),reference.half().numpy().view(np.uint16))
        else:
            torch.testing.assert_close(actual,reference.half(),rtol=3e-3,atol=3e-4)
            if reference.any():assert float(torch.linalg.vector_norm(actual.double()-reference)/torch.linalg.vector_norm(reference))<1e-3
        before=npu.stats();out=layer(x);end=npu.stats()
        for field in ('host_upload_copies','host_download_copies','session_creations'):assert before[field]==end[field],field
        program=npu._sessions[layer._keys[host.shape[0]]][1]
        entry=dict(status='passed',label=label,shape=[host.shape[0],*reversed(q.shape)],activation_scale=scale,
            compact=compact,transpose=transpose,exact=exact,max_abs_error=float((actual.double()-reference).abs().max()),
            relative_l2=float(torch.linalg.vector_norm(actual.double()-reference)/torch.linalg.vector_norm(reference)) if reference.any() else 0,
            warm_graph_calls=end['npu_calls']-before['npu_calls'],warm_host_copies_and_compiles=0,modes=dpu_modes(program.graph_blob))
        report['checks'].append(entry);print({k:v for k,v in entry.items() if k!='modes'},flush=True)
        np.savez(OUT/f'{case}-{len(report["checks"])}.npz',host=host.numpy(),actual=actual.numpy(),reference=reference.numpy())
    try:
        with torch.no_grad():
            if case=='probe':
                m,k,n=5,64,32
                codes=torch.randint(-127,128,(n,k),dtype=torch.int8).numpy()
                q=PackedWeight((n,k),'int8',k,codes.tobytes(),np.full(n,1/64,dtype=np.float16).tobytes())
                host=(torch.randn(m,k)/4).half();x=host.to('intel_npu')
                def fq(value,low,high,name,levels=256):
                    constants=[s.constant(np.asarray(v,dtype=np.float16)) for v in (low,high,low,high)]
                    return s.op('FakeQuantize',value,*constants,levels=levels,auto_broadcast='numpy',
                                _shape=value.shape,_dtype='f16',_name=name)
                for variant in ('fp16','w8a16','activation_fq','both_fq','output_fq'):
                    print('building '+variant,flush=True)
                    if variant in ('fp16','w8a16'):
                        graph,xml,data=weight_graph(q,m,reference_fp16=variant=='fp16')
                    else:
                        inp=s.input('x',shape=(m,k),dtype='f16')
                        v=fq(inp,-1,127/128,'activation')
                        if variant=='activation_fq':
                            w=s.op('Convert',s.constant(codes),destination_type='f16',_shape=(n,k),_dtype='f16')
                            w=s.op('Multiply',w,s.constant(np.float16(1/64)),auto_broadcast='numpy',_shape=(n,k),_dtype='f16')
                        else:w=fq(s.constant(q.dequantize()),-2,127/64,'weight')
                        y=s.op('MatMul',v,w,transpose_a=False,transpose_b=True,_shape=(m,n),_dtype='f16',_name='y')
                        if variant=='output_fq':y=fq(y,-32,127/4,'output')
                        graph=s.Graph([inp],[y],name='int8_probe');ir=s.serialize_ir(graph);xml,data=ir.xml,ir.weights
                    if isinstance(xml,str):xml=xml.encode()
                    print('IR built '+variant,flush=True)
                    (OUT/f'probe-{variant}.xml').write_bytes(xml);(OUT/f'probe-{variant}.bin').write_bytes(data)
                    compiled=npu._native.compile_ir(xml,data,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
                    print('compiled '+variant,flush=True)
                    program=s.load_native(compiled.graph_blob,graph=graph,libraries=npu._native)
                    (OUT/f'probe-{variant}.blob').write_bytes(program.graph_blob)
                    key=('int8_probe',variant)
                    session=npu._native.create_inference_session(program.graph_blob,timeout_ms=10000,parent=npu._root_session)
                    npu._sessions[key]=session,program;npu._session_creations+=1
                    actual=npu._run(key,[x],(m,n),torch.float16).cpu().numpy()
                    print('executed '+variant,flush=True)
                    inp_ref=host.numpy().astype(np.float64)
                    if variant not in ('fp16','w8a16'):inp_ref=np.clip(np.rint(inp_ref*128),-128,127)/128
                    # NumPy's conda MKL uses libomp, while torch uses libiomp5;
                    # non-BLAS einsum avoids initializing conflicting runtimes.
                    reference=np.einsum('mk,nk->mn',inp_ref,q.dequantize().astype(np.float64),optimize=False)
                    if variant=='output_fq':reference=np.clip(np.rint(reference*4),-128,127)/4
                    error=float(np.max(np.abs(actual-reference)))
                    entry=dict(variant=variant,max_abs_error=error,modes=dpu_modes(program.graph_blob))
                    report['checks'].append(entry);print(entry,flush=True)
                    np.savez(OUT/f'probe-{variant}.npz',host=host.numpy(),codes=codes,actual=actual,reference=reference)
                del x
            if case=='numeric':
                for sc in (.125,1.,16.):
                    q=PackedWeight((1,256),'int8',256,np.arange(-128,128,dtype=np.int8).tobytes(),np.float16(sc).tobytes())
                    check(q,torch.eye(256,dtype=torch.float16),exact=True,label='all_256_weight_codes')
                    check(q,torch.eye(256,dtype=torch.float16),scale=1/64,exact=True,label='w8a8_all_weight_codes')
                for m,k,n,g in [(1,17,3,None),(5,64,32,None),(65,256,67,None),(5,64,32,32),(7,9,5,3)]:
                    q=quantize_weight(torch.randn(n,k)/8,format='int8',group_size=g)
                    host=(torch.randn(m,k)/8).half()
                    check(q,host,label='w8a16')
                    if g is None:
                        for scale in (1/128,1/256):check(q,host,scale=scale,label='w8a8')
                    else:check(q,host,compact=True,label='compact_w8a16')
                q=quantize_weight(torch.randn(13,64)/8,format='int8')
                host=(torch.randn(7,64)/8).half();bias=torch.randn(13).half()/8
                for scale in (None,1/128):check(q,host,scale=scale,bias=bias,transpose=True,label='bias_transpose')
                q=quantize_weight(torch.zeros(3,17),format='int8')
                for scale in (None,1/128):check(q,torch.ones(2,17,dtype=torch.float16),scale=scale,exact=True,label='zero')
                q=PackedWeight((256,256),'int8',256,np.eye(256,dtype=np.int8).tobytes(),np.ones(256,dtype=np.float16).tobytes())
                grid=torch.arange(-128,128,dtype=torch.float16)/128
                host=torch.stack((grid,grid+1/256,grid-1/256,grid*2))
                check(q,host,scale=1/128,exact=True,label='all_activation_codes_ties_clipping')
                codes=np.stack((np.full(4096,127,dtype=np.int8),np.tile([127,-127],2048).astype(np.int8)))
                q=PackedWeight((2,4096),'int8',4096,codes.tobytes(),np.full(2,1/128,dtype=np.float16).tobytes())
                check(q,torch.full((1,4096),127/128,dtype=torch.float16),scale=1/128,exact=True,label='large_integer_accumulation_cancellation')
            if case=='edges':
                q=PackedWeight((16,16),'int8',16,np.eye(16,dtype=np.int8).tobytes(),np.ones(16,dtype=np.float16).tobytes())
                for scale in (2**-24,2**-14,256):
                    host=torch.arange(-8,8,dtype=torch.float16).reshape(1,16)*scale
                    check(q,host,scale=scale,exact=True,label='activation_scale_boundary')
            if case=='diagnose-weight':
                host=(torch.randn(1,1024)/8).half();q=quantize_weight(torch.randn(1024,1024)/8,format='int8')
                layer=QuantizedLinear(q,activation_scale=1/128)
                basis=torch.eye(1024,dtype=torch.float16)/2;x=basis.to('intel_npu')
                actual=layer(x).cpu().double().t()*2
                scales=np.frombuffer(q.scales,dtype=np.float16).astype(np.float64).reshape(-1,1)
                low=(-128*scales).astype(np.float16).astype(np.float64);high=(127*scales).astype(np.float16).astype(np.float64);grid=(high-low)/255
                codes=np.frombuffer(q.data,dtype=np.int8).reshape(q.shape).astype(np.float64)
                refs={'fp16_dequant':q.dequantize().astype(np.float64),'original_scale':codes*scales,
                      'fq_endpoints':fq_reference(q.dequantize().astype(np.float64),scales),
                      'aligned_zero':codes*grid}
                diagnostics={tag:dict(max_abs=float((actual-torch.from_numpy(ref)).abs().max()),
                    relative_l2=float(torch.linalg.vector_norm(actual-torch.from_numpy(ref))/torch.linalg.vector_norm(torch.from_numpy(ref)))) for tag,ref in refs.items()}
                print(diagnostics,flush=True);report['checks'].append(diagnostics)
                np.savez(OUT/'diagnose-weight.npz',codes=codes,scales=scales,actual=actual.numpy(),**refs)
                del x
            if case=='mlp':
                first=torch.nn.Linear(64,32).eval();second=torch.nn.Linear(32,7).eval()
                host=(torch.randn(7,64)/8).half()
                for scale in (None,1/128):
                    q1=QuantizedLinear.from_float(first,format='int8',activation_scale=scale)
                    q2=QuantizedLinear.from_float(second,format='int8',activation_scale=scale)
                    model=torch.nn.Sequential(q1,torch.nn.ReLU(),q2).eval();x=host.to('intel_npu')
                    reference=linear_reference(q1.weight,host,scale,first.bias).half().relu()
                    reference=linear_reference(q2.weight,reference,scale,second.bias).half()
                    actual=model(x).cpu();torch.testing.assert_close(actual,reference,rtol=3e-3,atol=3e-4)
                    original=second(first(host.float()).relu()).half()
                    before=npu.stats()
                    for _ in range(3):out=model(x)
                    after=npu.stats()
                    for field in ('host_upload_copies','host_download_copies','session_creations'):assert before[field]==after[field],field
                    assert after['npu_calls']-before['npu_calls']==9
                    entry=dict(status='passed',network='64->32->7 ReLU MLP batch7',activation_scale=scale,
                        max_abs_error_reference=float((actual.float()-reference.float()).abs().max()),
                        relative_l2_original=float(torch.linalg.vector_norm(actual.float()-original.float())/torch.linalg.vector_norm(original.float())),
                        warm_graph_calls=9,warm_host_copies_and_compiles=0)
                    report['checks'].append(entry);print(entry,flush=True);del x,out
            if case=='guards':
                layer=QuantizedLinear.from_float(torch.nn.Linear(16,3),format='int8',activation_scale=1/128)
                inputs=[torch.ones(1,16,dtype=torch.float16),torch.ones(1,16).to('intel_npu'),
                        torch.ones(1,15,dtype=torch.float16).to('intel_npu'),
                        torch.ones(2,16,dtype=torch.float16).to('intel_npu')[1:],
                        torch.ones(1,16,dtype=torch.float16).to('intel_npu').requires_grad_()]
                before=npu.stats()
                # requires_grad rejection is active only while grad mode is on.
                with torch.enable_grad():
                    for bad in inputs:
                        try:layer(bad)
                        except (ValueError,RuntimeError):pass
                        else:raise AssertionError('Unsupported input accepted')
                assert npu.stats()['npu_calls']==before['npu_calls']
                inputs.clear();del bad
                x=torch.ones(1,16,dtype=torch.float16).to('intel_npu')
                limit=npu._MAX_SESSIONS
                try:
                    npu._MAX_SESSIONS=len(npu._sessions)
                    try:layer(x)
                    except RuntimeError as error:assert 'cache is full' in str(error)
                    else:raise AssertionError('Full cache did not reject')
                finally:npu._MAX_SESSIONS=limit
                out=layer(x)
                report['checks'].append(dict(status='passed',input_rejections=5,cache_limit_rejection=1,rejected_graph_calls=0,recovery=True))
                del x,out
            if case=='performance':
                report['timings']=[]
                for m in (1,32,128,512):
                    k=n=1024;host=(torch.randn(m,k)/8).half();x=host.to('intel_npu')
                    original=torch.randn(n,k)/8;q=quantize_weight(original,format='int8')
                    layers={tag:QuantizedLinear(q,activation_scale=sc) for tag,sc in [('w8a16',None),('w8a8',1/128)]}
                    program,evidence=compile_weight(q,m,npu._native,reference_fp16=True)
                    key=('int8_fp16_reference',m)
                    session=npu._native.create_inference_session(program.graph_blob,timeout_ms=10000,parent=npu._root_session)
                    npu._sessions[key]=session,program;npu._session_creations+=1
                    run={tag:lambda tag=tag:layers[tag](x) for tag in layers}
                    run['fp16']=lambda:npu._run(key,[x],(m,n),torch.float16)
                    cpu_weight=torch.from_numpy(q.dequantize()).t().contiguous()
                    run['cpu_fp16']=lambda:host@cpu_weight
                    samples={tag:[] for tag in run};versions=tuple(run)
                    for tag in run:
                        for _ in range(3):run[tag]()
                    before=npu.stats()
                    for i in range(21):
                        order=versions[i%4:]+versions[:i%4]
                        for tag in order if i%2==0 else tuple(reversed(order)):
                            start=time.perf_counter();run[tag]();samples[tag].append(1000*(time.perf_counter()-start))
                    after=npu.stats()
                    for field in ('host_upload_copies','host_download_copies','session_creations'):assert before[field]==after[field],field
                    from npunlock.npu4000 import Elf
                    constants={tag:sum(s['size'] for s in Elf(npu._sessions[layer._keys[m]][1].graph_blob).sections if s['name'].startswith('buffer.Constant.')) for tag,layer in layers.items()}
                    calls={}
                    for tag in layers:
                        b=npu.stats();run[tag]();calls[tag]=npu.stats()['npu_calls']-b['npu_calls']
                    entry=dict(shape=[m,k,n],median_ms={tag:statistics.median(v) for tag,v in samples.items()},samples_ms=samples,
                        constant_bytes={**constants,'fp16':evidence['constant_section_bytes']},warm_graph_calls=calls,
                        warm_host_copies_and_compiles=0,cpu_threads=torch.get_num_threads(),input_weight_scale=1/8)
                    report['timings'].append(entry);print({k:v for k,v in entry.items() if k!='samples_ms'},flush=True)
                    for scale in (None,1/128):check(q,host,scale=scale,label='performance_numerical_gate')
                    del x
        gc.collect();report['stats_before_shutdown']=npu.stats()
        assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown();report['final_stats']=npu.stats()
        assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='completed_diagnostic' if case in ('probe','diagnose-weight') else 'passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats())
        print(report['traceback'],flush=True)
    (OUT/f'{case}.json').write_text(json.dumps(report,indent=2))
    return int(report['status'] not in ('passed','completed_diagnostic'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['probe','numeric','edges','performance','mlp','guards','diagnose-weight','lowbit-numeric','lowbit-mlp'],default='probe');parser.add_argument('--child',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.child:return execute(args.case)
    with (OUT/f'{args.case}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--child','--case',args.case],stdout=log,stderr=log,timeout=300)
    print((OUT/f'{args.case}.log').read_text(encoding='utf-8'),flush=True)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())

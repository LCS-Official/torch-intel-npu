"""Serial direct-driver grouped lowbit graph experiments and comparisons."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/lowbit-group'


def experiment_graph(q,batch,variant):
    import numpy as np
    import npunlock as s
    from torch_npu_lowbit import weight_graph,pack_codes
    if variant=='original':return weight_graph(q,batch)
    n,k=q.shape;g=q.group_size;groups=k//g
    x=s.input('x',shape=(batch,k),dtype='f16');value=x
    if variant=='pad':
        physical=max(batch,32)
        if physical!=batch:
            value=s.op('Concat',x,s.constant(np.zeros((physical-batch,k),dtype=np.float16)),axis=0,
                       _shape=(physical,k),_dtype='f16')
    else:physical=batch
    scales=np.frombuffer(q.scales,dtype=np.float16).reshape(n,groups,1)
    raw=np.frombuffer(q.data,dtype=np.uint8)
    codes=np.column_stack((raw&15,raw>>4)).reshape(-1)[:n*k].reshape(n,groups,g)
    def decoded(data,shape,scale,name):
        packed=s.constant_packed(data,shape=shape,dtype=q.dtype,name=name)
        dec=s.op('Convert',packed,destination_type='f16',_shape=shape,_dtype='f16',_name='decode_'+name)
        return s.op('Multiply',dec,s.constant(scale),auto_broadcast='numpy',_shape=shape,_dtype='f16')
    def reshape(v,shape):
        return s.op('Reshape',v,s.constant(np.array(shape,dtype=np.int64)),special_zero=False,_shape=shape,_dtype='f16')
    def slice_columns(start,count):
        args=[s.constant(np.array(a,dtype=np.int64)) for a in ([0,start],[batch,start+count],[1,1])]
        return s.op('StridedSlice',x,*args,begin_mask=[0,0],end_mask=[0,0],new_axis_mask=[0,0],
                    shrink_axis_mask=[0,0],ellipsis_mask=[0,0],_shape=(batch,count),_dtype='f16')
    if variant=='split':
        pieces=[]
        for j in range(groups):
            w=decoded(pack_codes(codes[:,j,:]),(n,g),scales[:,j,:],str(j))
            pieces.append(s.op('MatMul',slice_columns(j*g,g),w,transpose_a=False,transpose_b=True,_shape=(batch,n),_dtype='f16'))
        y=pieces[0]
        for p in pieces[1:]:y=s.op('Add',y,p,auto_broadcast='numpy',_shape=y.shape,_dtype='f16')
    else:
        shape=(n*groups,g) if variant=='flat' else q.grouped_shape
        scale=scales.reshape(n*groups,1) if variant=='flat' else scales
        w=reshape(decoded(q.data,shape,scale,'weight'),(n,k))
        y=s.op('MatMul',value,w,transpose_a=False,transpose_b=True,_shape=(physical,n),_dtype='f16')
        if variant=='pad' and physical!=batch:
            args=[s.constant(np.array(a,dtype=np.int64)) for a in ([0,0],[batch,n],[1,1])]
            y=s.op('StridedSlice',y,*args,begin_mask=[0,0],end_mask=[0,0],new_axis_mask=[0,0],
                   shrink_axis_mask=[0,0],ellipsis_mask=[0,0],_shape=(batch,n),_dtype='f16')
    y=s.op('Reshape',y,s.constant(np.array(y.shape,dtype=np.int64)),special_zero=False,_shape=y.shape,_dtype='f16',_name='y')
    graph=s.Graph([x],[y],name='lowbit_group_'+variant);ir=s.serialize_ir(graph);xml=ET.fromstring(ir.xml)
    for layer in xml.findall('./layers/layer'):
        if layer.attrib['name'].startswith('decode_'):
            ET.SubElement(ET.SubElement(layer,'rt_info'),'attribute',name='decompression',version='0')
    return graph,ET.tostring(xml),ir.weights


def execute(case):
    import numpy as np
    import torch
    import torch_intel_npu as npu
    import npunlock as s
    from npunlock.npu4000 import Elf
    from torch_npu_lowbit import quantize_weight,QuantizedLinear,NF4
    torch.manual_seed(418)
    report=dict(case=case,status='running',checks=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('torch_npu_lowbit.py','torch_npu_lowbit_group_demo.py')})
    try:
        with torch.no_grad():
            configurations=[(5,64,32,32),(1,256,256,128),(128,1024,1024,128)]
            for batch,k,n,size in configurations:
                for fmt in ('int4','nf4'):
                    q=quantize_weight(torch.randn(n,k)/8,format=fmt,group_size=size)
                    host=(torch.randn(batch,k)/8).half();x=host.to('intel_npu')
                    raw=np.frombuffer(q.data,dtype=np.uint8)
                    codes=np.column_stack((raw&15,raw>>4)).reshape(-1)[:n*k].reshape(q.grouped_shape)
                    dec=((codes.astype(np.int8)^8)-8).astype(np.float64) if fmt=='int4' else NF4[codes].astype(np.float64)
                    real=dec*np.frombuffer(q.scales,dtype=np.float16).astype(np.float64).reshape(n,k//size,1)
                    ideal=host.double()@torch.from_numpy(real.reshape(n,k)).t()
                    variants=('original','flat','split') if case=='group' else ('original','pad')
                    for variant in variants:
                        graph,xml,data=experiment_graph(q,batch,variant)
                        result=npu._native.compile_ir(xml,data,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
                        assert (result.vendor_id,result.device_id,tuple(result.compiler_version))==(0x8086,0x643e,(8,3))
                        elf=Elf(result.graph_blob)
                        dpu=[a['name'] for a in elf.sections if a['size'] and a['name'].startswith('task.dpu.')]
                        assert dpu
                        tag=f'{batch}-{k}-{n}-{size}-{fmt}-{variant}'
                        (OUT/f'{tag}.xml').write_bytes(xml);(OUT/f'{tag}.bin').write_bytes(data);(OUT/f'{tag}.blob').write_bytes(result.graph_blob)
                        program=s.load_native(result.graph_blob,graph=graph,libraries=npu._native)
                        key=('lowbit_group_experiment',tag)
                        session=npu._native.create_inference_session(program.graph_blob,timeout_ms=10000,parent=npu._root_session)
                        npu._sessions[key]=session,program;npu._session_creations+=1
                        actual=npu._run(key,[x],(batch,n),torch.float16).cpu()
                        rel=float(torch.linalg.vector_norm(actual.double()-ideal)/torch.linalg.vector_norm(ideal))
                        maximum=float((actual.double()-ideal).abs().max())
                        for _ in range(3):npu._run(key,[x],(batch,n),torch.float16)
                        samples=[];before=npu.stats()
                        for _ in range(9):
                            start=time.perf_counter();npu._run(key,[x],(batch,n),torch.float16);samples.append(1000*(time.perf_counter()-start))
                        after=npu.stats()
                        assert all(before[f]==after[f] for f in ('host_upload_copies','host_download_copies','session_creations'))
                        entry=dict(tag=tag,variant=variant,shape=[batch,k,n],format=fmt,group_size=size,
                            max_abs_error=maximum,relative_l2=rel,numeric_passed=rel<1e-3,
                            ir_weight_bytes=len(data),constant_bytes=sum(a['size'] for a in elf.sections if a['name'].startswith('buffer.Constant.')),
                            dpu_sections=dpu,shave_sections=[a['name'] for a in elf.sections if a['size'] and a['name'].startswith('task.shave.invocation.')],
                            median_ms=statistics.median(samples),samples_ms=samples,graph_calls=after['npu_calls']-before['npu_calls'])
                        report['checks'].append(entry);print({k:v for k,v in entry.items() if k not in ('dpu_sections','samples_ms')},flush=True)
                    del x,actual
        gc.collect();report['stats_before_shutdown']=npu.stats()
        assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown();report['final_stats']=npu.stats()
        assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='completed_diagnostic'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/f'{case}.json').write_text(json.dumps(report,indent=2)+'\n')
    return int(report['status']=='failed')


def integration():
    import importlib.util
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from npunlock.npu4000 import Elf
    from torch_npu_lowbit import quantize_weight,QuantizedLinear,NF4
    previous_path=OUT/'before/torch_npu_lowbit.py'
    spec=importlib.util.spec_from_file_location('_lowbit_before_group',previous_path)
    previous=importlib.util.module_from_spec(spec);sys.modules[spec.name]=previous;spec.loader.exec_module(previous)
    report=dict(case='integration',status='running',checks=[],timings=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('torch_npu_lowbit.py','torch_npu_lowbit_group_demo.py')},previous_source_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest())
    torch.manual_seed(419)
    try:
        with torch.no_grad():
            configs=[(1,64,32,32),(7,9,5,3),(31,128,64,64),(32,128,64,64),
                     (1,1024,1024,128),(128,1024,1024,128),(4,1024,32,32)]
            for fmt in ('int4','nf4'):
                for batch,k,n,size in configs:
                    q=quantize_weight(torch.randn(n,k)/8,format=fmt,group_size=size)
                    host=(torch.randn(batch,k)/8).half();transpose=k==9
                    x=host.t().contiguous().to('intel_npu').t() if transpose else host.to('intel_npu')
                    bias=torch.randn(n).half()/8 if batch in (7,31) else None
                    old_q=previous.PackedWeight(q.shape,q.format,q.group_size,q.data,q.scales)
                    layers={'previous':previous.QuantizedLinear(old_q,bias=bias),
                            'fused':QuantizedLinear(q,bias=bias),'compact':QuantizedLinear(q,bias=bias,compact_groups=True)}
                    raw=np.frombuffer(q.data,dtype=np.uint8)
                    codes=np.column_stack((raw&15,raw>>4)).reshape(-1)[:n*k].reshape(q.grouped_shape)
                    dec=((codes.astype(np.int8)^8)-8).astype(np.float64) if fmt=='int4' else NF4[codes].astype(np.float64)
                    real=(dec*np.frombuffer(q.scales,dtype=np.float16).astype(np.float64).reshape(n,k//size,1)).reshape(n,k)
                    ideal=host.double()@torch.from_numpy(real).t()
                    if bias is not None:ideal=ideal+bias.double()
                    # An independent full product, plus the absolute dot product
                    # rounding bound, keeps format error separate from DPU error.
                    depth=(k//size-1).bit_length()
                    budget=(depth+3)*torch.finfo(torch.float16).eps*(host.double().abs()@torch.from_numpy(np.abs(real)).t())+3e-4
                    for tag,layer in layers.items():
                        actual=layer(x).cpu()
                        error=actual.double()-ideal
                        rel=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(ideal))
                        assert rel<1e-3,(fmt,batch,k,n,size,tag,rel)
                        assert torch.all(error.abs()<=budget),(fmt,batch,k,n,size,tag)
                        before=npu.stats();warm=layer(x);after=npu.stats()
                        assert all(before[f]==after[f] for f in ('host_upload_copies','host_download_copies','session_creations'))
                        expected_calls=(int(transpose)+(4 if tag=='previous' and fmt=='int4' and batch<32 else 1))
                        assert after['npu_calls']-before['npu_calls']==expected_calls
                        elf=Elf(npu._sessions[layer._keys[batch]][1].graph_blob)
                        entry=dict(format=fmt,shape=[batch,k,n],group_size=size,variant=tag,bias=bias is not None,transpose=transpose,status='passed',
                            max_abs_error=float(error.abs().max()),relative_l2=rel,rounding_budget_max=float(budget.max()),
                            graph_calls=after['npu_calls']-before['npu_calls'],constant_bytes=sum(a['size'] for a in elf.sections if a['name'].startswith('buffer.Constant.')),
                            shave_sections=[a['name'] for a in elf.sections if a['size'] and a['name'].startswith('task.shave.invocation.')])
                        assert any(a['size'] and a['name'].startswith('task.dpu.') for a in elf.sections)
                        report['checks'].append(entry);print(entry,flush=True)
                        del warm,actual
                    if (batch,k,n) in ((1,64,32),(1,1024,1024),(128,1024,1024)):
                        for layer in layers.values():
                            for _ in range(3):layer(x)
                        samples={tag:[] for tag in layers};tags=tuple(layers);before=npu.stats()
                        for i in range(21):
                            order=tags[i%3:]+tags[:i%3]
                            for tag in order if i%2==0 else tuple(reversed(order)):
                                start=time.perf_counter();layers[tag](x);samples[tag].append(1000*(time.perf_counter()-start))
                        after=npu.stats();assert all(before[f]==after[f] for f in ('host_upload_copies','host_download_copies','session_creations'))
                        entry=dict(format=fmt,shape=[batch,k,n],group_size=size,median_ms={tag:statistics.median(v) for tag,v in samples.items()},samples_ms=samples,warm_host_copies_and_compiles=0)
                        report['timings'].append(entry);print({k:v for k,v in entry.items() if k!='samples_ms'},flush=True)
                    del x
            for fmt in ('int4','nf4'):
                source1=torch.nn.Linear(64,32).half();source2=torch.nn.Linear(32,7).half()
                a=QuantizedLinear.from_float(source1,format=fmt,group_size=32,compact_groups=True)
                b=QuantizedLinear.from_float(source2,format=fmt,compact_groups=True)
                host=(torch.randn(7,64)/8).half();x=host.to('intel_npu')
                expected=torch.nn.functional.linear(host,torch.from_numpy(a.weight.dequantize()),source1.bias).relu()
                expected=torch.nn.functional.linear(expected,torch.from_numpy(b.weight.dequantize()),source2.bias)
                actual=b(a(x).relu()).cpu();torch.testing.assert_close(actual,expected,rtol=3e-3,atol=3e-4)
                entry=dict(format=fmt,variant='compact',network='64->32->7 ReLU MLP, batch7',status='passed',max_abs_error=float((actual.float()-expected.float()).abs().max()))
                report['checks'].append(entry);print(entry,flush=True);del x,actual
        gc.collect();report['stats_before_shutdown']=npu.stats()
        assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown();report['final_stats']=npu.stats()
        assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'integration.json').write_text(json.dumps(report,indent=2)+'\n')
    return int(report['status']!='passed')


def edges():
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from torch_npu_lowbit import QuantizedLinear,PackedWeight,pack_codes,quantize_weight
    report=dict(case='edges',status='running',checks=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('torch_npu_lowbit.py','torch_npu_lowbit_group_demo.py')})
    try:
        with torch.no_grad():
            for fmt in ('int4','nf4'):
                for scale in (.125,1.,16.):
                    q=PackedWeight((1,16),fmt,8,pack_codes(np.arange(16,dtype=np.uint8)),np.full(2,scale,dtype=np.float16).tobytes())
                    x=torch.eye(16,dtype=torch.float16).to('intel_npu')
                    actual=QuantizedLinear(q,compact_groups=True)(x).cpu()
                    np.testing.assert_array_equal(actual.numpy().view(np.uint16),q.dequantize().T.copy().view(np.uint16))
                    report['checks'].append(dict(format=fmt,name='all_codes_two_groups',scale=scale,status='passed',bitwise_exact=True));del x,actual
                q=quantize_weight(torch.zeros(5,9),format=fmt,group_size=3)
                x=torch.ones(7,9,dtype=torch.float16).to('intel_npu')
                actual=QuantizedLinear(q,compact_groups=True)(x).cpu()
                assert torch.count_nonzero(actual)==0
                report['checks'].append(dict(format=fmt,name='odd_group_zero',status='passed'));del x,actual
                # Compact graphs expose FP16 partial sums. A finite full dot
                # product can have overflowing partials; record this limitation.
                positive,negative,activation=(7,13,1024.) if fmt=='int4' else (15,0,4096.)
                codes=np.array([positive]*16+[negative]*16,dtype=np.uint8)
                scales=np.array([1.,2.] if fmt=='int4' else [1.,.5],dtype=np.float16)
                q=PackedWeight((1,32),fmt,16,pack_codes(codes),scales.tobytes())
                x=torch.full((1,32),activation,dtype=torch.float16).to('intel_npu')
                full=QuantizedLinear(q)(x).cpu();compact=QuantizedLinear(q,compact_groups=True)(x).cpu()
                expected=float(torch.full((1,32),activation).double()@torch.from_numpy(q.dequantize().astype(np.float64)).t())
                entry=dict(format=fmt,name='partial_overflow_limitation',full_result=full.item(),compact_result=compact.item(),expected_full=expected,expected_limitation=True)
                print(entry,flush=True)
                assert full.item()==expected and compact.item()!=expected
                report['checks'].append(dict(**entry,status='passed'));del x
            q=quantize_weight(torch.ones(1,65536),group_size=32768)
            x=torch.ones(1,65536,dtype=torch.float16).to('intel_npu');before=npu.stats()['npu_calls']
            try:QuantizedLinear(q)(x)
            except ValueError as error:assert 'storage bound' in str(error)
            else:raise AssertionError('Physical padding bound was not rejected')
            assert npu.stats()['npu_calls']==before
            report['checks'].append(dict(name='physical_bound_before_submission',status='passed'));del x
        gc.collect();report['stats_before_shutdown']=npu.stats()
        assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown();report['final_stats']=npu.stats()
        assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'edges.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['checks']),flush=True)
    return int(report['status']!='passed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['group','pad','integration','edges','numeric','guards','mlp','performance','diagnose'],default='pad');parser.add_argument('--child',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.child:
        if args.case=='integration':return integration()
        if args.case=='edges':return edges()
        if args.case in ('group','pad'):return execute(args.case)
        import torch_npu_lowbit_demo as existing
        existing.OUT=OUT
        return existing.execute(args.case)
    with (OUT/f'{args.case}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--child','--case',args.case],stdout=log,stderr=log,timeout=300)
    print((OUT/f'{args.case}.log').read_text(encoding='utf-8'),flush=True);return result.returncode


if __name__=='__main__':raise SystemExit(main())

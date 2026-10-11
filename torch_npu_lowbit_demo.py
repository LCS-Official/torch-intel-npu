"""Isolated direct-driver compressed-weight DPU tests; no OpenVINO runtime."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/lowbit'


def execute(case):
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from torch_npu_lowbit import quantize_weight,compile_weight,weight_graph,QuantizedLinear,PackedWeight,pack_codes
    from npunlock.npu4000 import Elf
    torch.manual_seed(417)
    report=dict(case=case,status='running',checks=[],timings=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['torch_npu_lowbit.py','torch_npu_lowbit_demo.py','npunlock/python/npunlock/graph.py','npunlock/python/npunlock/ir.py','npunlock/python/npunlock/tensor.py']})
    def baseline(q,batch,tag):
        program,evidence=compile_weight(q,batch,npu._native,reference_fp16=True)
        key=('lowbit_reference',tag)
        session=npu._native.create_inference_session(program.graph_blob,timeout_ms=10000,parent=npu._root_session)
        npu._sessions[key]=session,program; npu._session_creations+=1
        return key,evidence

    def check(q,host,*,bias=None,transpose=False,exact=False,label='random'):
        layer=QuantizedLinear(q,bias=bias)
        x=host.t().contiguous().to('intel_npu').t() if transpose else host.to('intel_npu')
        before=npu.stats()
        actual=layer(x).cpu(); after=npu.stats()
        reference=host.float()@torch.from_numpy(q.dequantize()).t().float()
        if bias is not None:reference=reference.half().float()+bias.half().float()
        if exact:
            np.testing.assert_array_equal(actual.numpy().view(np.uint16),reference.half().numpy().view(np.uint16))
        else:torch.testing.assert_close(actual,reference.half(),rtol=3e-3,atol=3e-4)
        assert after['host_upload_copies']-before['host_upload_copies']==after['cached_scalar_buffers']-before['cached_scalar_buffers']
        before=npu.stats(); result=layer(x); end=npu.stats()
        for field in ('host_upload_copies','host_download_copies','session_creations'):assert before[field]==end[field],field
        evidence=npu._sessions[layer._keys[host.shape[0]]][1]
        from npunlock.npu4000 import Elf
        elf=Elf(evidence.graph_blob)
        constants=sum(s['size'] for s in elf.sections if s['name'].startswith('buffer.Constant.'))
        entry=dict(format=q.format,shape=[host.shape[0],q.shape[1],q.shape[0]],group_size=q.group_size,label=label,
            transpose=transpose,exact=exact,status='passed',max_abs_error=(actual.float()-reference).abs().max().item(),
            graph_calls=end['npu_calls']-before['npu_calls'],constant_section_bytes=constants,weight_data_bytes=len(q.data),scale_bytes=len(q.scales))
        report['checks'].append(entry);print(entry,flush=True)

    def measure(batch,k,n,group_size):
        original=torch.randn(n,k)/8;host=(torch.randn(batch,k)/8).half();x=host.to('intel_npu')
        layers={fmt:QuantizedLinear(quantize_weight(original,format=fmt,group_size=group_size)) for fmt in ('int4','nf4')}
        refs={fmt:baseline(layer.weight,batch,(batch,k,n,fmt,group_size)) for fmt,layer in layers.items()}
        run={fmt:lambda fmt=fmt:layers[fmt](x) for fmt in layers}
        run.update({f'fp16_{fmt}':lambda fmt=fmt:npu._run(refs[fmt][0],[x],(batch,n),torch.float16) for fmt in refs})
        calls={}
        for tag in run:
            for _ in range(3):run[tag]()
            before=npu.stats();run[tag]();calls[tag]=npu.stats()['npu_calls']-before['npu_calls']
        cpu_weights={fmt:torch.from_numpy(layer.weight.dequantize()).t().contiguous() for fmt,layer in layers.items()}
        samples={tag:[] for tag in [*run,'cpu']};versions=tuple(run);before=npu.stats()
        for i in range(15):
            order=versions[i%4:]+versions[:i%4]
            for tag in order if i%2==0 else tuple(reversed(order)):
                start=time.perf_counter();run[tag]();samples[tag].append(1000*(time.perf_counter()-start))
            start=time.perf_counter();host@cpu_weights['int4'];samples['cpu'].append(1000*(time.perf_counter()-start))
        after=npu.stats()
        for field in ('host_upload_copies','host_download_copies','session_creations'):assert before[field]==after[field],field
        precision=[]
        for fmt in layers:
            compressed=run[fmt]().cpu();plain=run[f'fp16_{fmt}']().cpu()
            q=layers[fmt].weight;raw=np.frombuffer(q.data,dtype=np.uint8)
            codes=np.column_stack((raw&15,raw>>4)).reshape(-1)[:n*k].reshape(q.grouped_shape)
            from torch_npu_lowbit import NF4
            decoded=((codes.astype(np.int8)^8)-8).astype(np.float64) if fmt=='int4' else NF4[codes].astype(np.float64)
            real=decoded*np.frombuffer(q.scales,dtype=np.float16).astype(np.float64).reshape(*q.grouped_shape[:2],1)
            ideal=host.double()@torch.from_numpy(real.reshape(n,k)).t()
            fused_error=float((compressed.double()-ideal).abs().max())
            ordinary_error=float((plain.double()-ideal).abs().max())
            rel_l2=float(torch.linalg.vector_norm(compressed.double()-ideal)/torch.linalg.vector_norm(ideal))
            print(dict(format=fmt,shape=[batch,k,n],group_size=group_size,fused_error=fused_error,ordinary_error=ordinary_error,relative_l2=rel_l2),flush=True)
            np.savez(OUT/f'precision-{batch}-{k}-{group_size}-{fmt}.npz',actual=compressed.numpy(),plain=plain.numpy(),ideal=ideal.numpy())
            # The compiler can fuse scale application instead of rounding each
            # reconstructed weight to FP16 first. Compare independent FP64
            # scaled codes, and budget the known weight-rounding difference.
            rounded=q.dequantize().astype(np.float64)
            weight_rounding=host.double().abs()@torch.from_numpy(np.abs(real.reshape(n,k)-rounded)).t()
            budget=weight_rounding+2*torch.finfo(torch.float16).eps*ideal.abs()+2e-4
            assert rel_l2<1e-3,('DPU error exceeds 0.1% relative L2',fmt,rel_l2)
            assert torch.all((compressed.double()-plain.double()).abs()<=budget),fmt
            precision.append(dict(format=fmt,max_abs_error_ideal=fused_error,relative_l2_ideal=rel_l2,
                max_abs_error_plain=float((compressed.float()-plain.float()).abs().max()),
                rounding_budget_max=float(budget.max()),rounding_budget_passed=True))
        entry=dict(shape=[batch,k,n],group_size=group_size,median_ms={tag:statistics.median(v) for tag,v in samples.items()},samples_ms=samples,
            graph_calls=calls,warm_host_copies_and_compiles=0,reference_evidence={fmt:refs[fmt][1] for fmt in refs},
            compressed_constant_bytes={fmt:sum(s['size'] for s in Elf(npu._sessions[layers[fmt]._keys[batch]][1].graph_blob).sections if s['name'].startswith('buffer.Constant.')) for fmt in layers},
            weight_relative_l2={fmt:float(torch.linalg.vector_norm(torch.from_numpy(layer.weight.dequantize()).float()-original)/torch.linalg.vector_norm(original)) for fmt,layer in layers.items()},precision=precision)
        report['timings'].append(entry);print({key:value for key,value in entry.items() if key!='samples_ms'},flush=True)

    try:
        with torch.no_grad():
            if case=='probe':
                x=(torch.randn(3,32)/8).half().to('intel_npu')
                for fmt in ('fp16','int4','nf4'):
                    q=quantize_weight(torch.randn(16,32)/8,format='nf4' if fmt=='nf4' else 'int4')
                    graph,xml,data=weight_graph(q,3,reference_fp16=fmt=='fp16')
                    (OUT/f'probe-{fmt}.xml').write_bytes(xml); (OUT/f'probe-{fmt}.bin').write_bytes(data)
                    program,evidence=compile_weight(q,3,npu._native,reference_fp16=fmt=='fp16')
                    (OUT/f'probe-{fmt}.blob').write_bytes(program.graph_blob)
                    key=('lowbit_probe',fmt)
                    session=npu._native.create_inference_session(program.graph_blob,timeout_ms=10000,parent=npu._root_session)
                    npu._sessions[key]=session,program; npu._session_creations+=1
                    actual=npu._run(key,[x],(3,16),torch.float16).cpu()
                    host=x.cpu().float()@torch.from_numpy(q.dequantize()).t().float()
                    torch.testing.assert_close(actual,host.half(),rtol=2e-3,atol=2e-4)
                    entry=dict(format=fmt,status='passed',max_abs_error=(actual.float()-host).abs().max().item(),**evidence)
                    report['checks'].append(entry); print(entry,flush=True)
                del x,actual
            if case=='numeric':
                for fmt in ('int4','nf4'):
                    for scale in (0.125,1.,16.):
                        q=PackedWeight((1,16),fmt,16,pack_codes(np.arange(16,dtype=np.uint8)),np.float16(scale).tobytes())
                        check(q,torch.eye(16,dtype=torch.float16),exact=True,label='all_codes')
                    for shape,size in [((1,3,17),None),((7,17,13),None),((5,64,32),32),((16,128,64),64),((65,256,67),128)]:
                        batch,k,n=shape
                        q=quantize_weight(torch.randn(n,k)/8,format=fmt,group_size=size)
                        check(q,(torch.randn(batch,k)/8).half(),label='grouped')
                    q=quantize_weight(torch.randn(13,64)/8,format=fmt,group_size=32)
                    check(q,(torch.randn(7,64)/8).half(),bias=torch.randn(13).half()/8,transpose=True,label='bias_transpose')
                    check(quantize_weight(torch.zeros(3,17),format=fmt),torch.ones(2,17,dtype=torch.float16),exact=True,label='zero')
            if case=='diagnose':
                from npunlock.npu4000 import Elf
                import xml.etree.ElementTree as ET
                for fmt in ('int4','nf4'):
                    for k,size in [(64,32),(128,64),(256,128)]:
                        q=quantize_weight(torch.randn(16,k)/8,format=fmt,group_size=size)
                        host=torch.eye(k,dtype=torch.float16);x=host.to('intel_npu')
                        for marked in (True,False):
                            graph,xml,data=weight_graph(q,k)
                            if not marked:
                                tree=ET.fromstring(xml)
                                for layer in tree.findall('./layers/layer'):
                                    if layer.find("./rt_info/attribute[@name='decompression']") is not None:
                                        layer.remove(layer.find('rt_info'))
                                xml=ET.tostring(tree)
                            result=npu._native.compile_ir(xml,data,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
                            program=__import__('npunlock').load_native(result.graph_blob,graph=graph,libraries=npu._native)
                            key=('lowbit_diagnose',fmt,k,size,marked)
                            session=npu._native.create_inference_session(program.graph_blob,timeout_ms=10000,parent=npu._root_session)
                            npu._sessions[key]=session,program;npu._session_creations+=1
                            actual=npu._run(key,[x],(k,16),torch.float16).cpu().numpy().T
                            expected=q.dequantize();error=np.max(np.abs(actual.astype(np.float32)-expected))
                            np.savez(OUT/f'diagnose-{fmt}-{k}-{marked}.npz',actual=actual,expected=expected)
                            entry=dict(format=fmt,k=k,group_size=size,decompression_mark=marked,max_abs_error=float(error),
                                bitwise_equal=bool(np.array_equal(actual.view(np.uint16),expected.view(np.uint16))),
                                constant_bytes=sum(s['size'] for s in Elf(result.graph_blob).sections if s['name'].startswith('buffer.Constant.')))
                            report['checks'].append(entry);print(entry,flush=True)
                        del x,actual
            if case=='performance':
                for size in (256,1024):
                    for batch in (1,128):
                        for group in (None,128):measure(batch,size,size,group)
            if case=='guards':
                module=QuantizedLinear.from_float(torch.nn.Linear(32,16),format='nf4')
                for host in (torch.ones(2,32),torch.ones(2,31).half(),torch.ones(2,2,32).half()):
                    value=host.to('intel_npu');before=npu.stats()['npu_calls']
                    try:module(value);raise AssertionError('Invalid activation accepted')
                    except ValueError:pass
                    assert npu.stats()['npu_calls']==before
                value=torch.ones(2,32,dtype=torch.float16,device='intel_npu',requires_grad=True)
                with torch.enable_grad():
                    try:module(value);raise AssertionError('Active autograd accepted')
                    except ValueError:pass
                report['checks'].append(dict(status='passed',name='dtype_shape_autograd_guards'))
                del value
            if case=='mlp':
                for fmt in ('int4','nf4'):
                    first=torch.nn.Linear(64,32,dtype=torch.float16)
                    second=torch.nn.Linear(32,7,dtype=torch.float16)
                    q1=QuantizedLinear.from_float(first,format=fmt,group_size=32)
                    q2=QuantizedLinear.from_float(second,format=fmt)
                    model=torch.nn.Sequential(q1,torch.nn.ReLU(),q2).eval()
                    host=(torch.randn(7,64)/8).half();x=host.to('intel_npu')
                    ref=torch.nn.functional.linear(host.float(),torch.from_numpy(q1.weight.dequantize()).float(),first.bias.float()).half().relu()
                    ref=torch.nn.functional.linear(ref.float(),torch.from_numpy(q2.weight.dequantize()).float(),second.bias.float()).half()
                    actual=model(x).cpu();torch.testing.assert_close(actual,ref,rtol=3e-3,atol=3e-4)
                    before=npu.stats()
                    for _ in range(3):result=model(x)
                    after=npu.stats()
                    for field in ('host_upload_copies','host_download_copies','session_creations'):assert before[field]==after[field],field
                    entry=dict(format=fmt,status='passed',network='64->32->7 ReLU MLP, batch7',warm_repeats=3,
                        max_abs_error_dequantized_cpu=float((actual.float()-ref.float()).abs().max()),
                        warm_graph_calls=after['npu_calls']-before['npu_calls'],warm_host_copies_and_compiles=0)
                    report['checks'].append(entry);print(entry,flush=True)
                    del x,result
            if case=='diagnose-small':
                q=quantize_weight(torch.randn(32,128)/8,format='int4',group_size=64)
                for batch in (32,63,64,65,128,129):
                    for pattern in ('random','onehot'):
                        host=(torch.randn(batch,128)/8).half() if pattern=='random' else torch.eye(128,dtype=torch.float16)[np.arange(batch)%128]
                        x=host.to('intel_npu');module=QuantizedLinear(q)
                        outputs=[module(x).cpu() for _ in range(3)]
                        key,evidence=baseline(q,batch,('small',batch,pattern))
                        plain=npu._run(key,[x],(batch,32),torch.float16).cpu()
                        reference=host.float()@torch.from_numpy(q.dequantize()).t().float()
                        entry=dict(batch=batch,pattern=pattern,errors=[float((a.float()-reference).abs().max()) for a in outputs],
                            plain_error=float((plain.float()-reference).abs().max()))
                        np.savez(OUT/f'small-{batch}-{pattern}.npz',host=host.numpy(),actual=outputs[0].numpy(),plain=plain.numpy(),reference=reference.numpy())
                        report['checks'].append(entry);print(entry,flush=True)
                        del x
                del outputs,plain
            if case=='diagnose-pad':
                for fmt in ('int4','nf4'):
                    for k,n,size in [(64,32,32),(128,16,64),(256,67,128),(1024,256,128)]:
                        q=quantize_weight(torch.randn(n,k)/8,format=fmt,group_size=size)
                        for batch in (1,16,31,32):
                            host=(torch.randn(batch,k)/8).half();x=host.to('intel_npu');module=QuantizedLinear(q)
                            actual=module(x).cpu();reference=host.float()@torch.from_numpy(q.dequantize()).t().float()
                            entry=dict(format=fmt,k=k,n=n,batch=batch,group_size=size,error=float((actual.float()-reference).abs().max()))
                            report['checks'].append(entry);print(entry,flush=True);del x,actual
            if case=='diagnose-tiles':
                import npunlock
                q=quantize_weight(torch.randn(32,64)/8,format='int4',group_size=32)
                host=(torch.randn(5,64)/8).half();x=host.to('intel_npu')
                for tiles in (1,2,4):
                    graph,xml,data=weight_graph(q,5)
                    result=npu._native.compile_ir(xml,data,build_flags=f'--config EXECUTION_MODE_HINT="ACCURACY" NPU_TILES="{tiles}"',timeout_ms=20000)
                    program=npunlock.load_native(result.graph_blob,graph=graph,libraries=npu._native)
                    key=('lowbit_tiles',tiles);session=npu._native.create_inference_session(program.graph_blob,timeout_ms=10000,parent=npu._root_session)
                    npu._sessions[key]=session,program;npu._session_creations+=1
                    actual=npu._run(key,[x],(5,32),torch.float16).cpu();reference=host.float()@torch.from_numpy(q.dequantize()).t().float()
                    entry=dict(tiles=tiles,error=float((actual.float()-reference).abs().max()));report['checks'].append(entry);print(entry,flush=True)
                del x,actual
        gc.collect(); report['stats_before_shutdown']=npu.stats()
        assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown(); report['final_stats']=npu.stats()
        assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats())
        print(report['traceback'],flush=True)
    (OUT/f'{case}.json').write_text(json.dumps(report,indent=2))
    return int(report['status']!='passed')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--case',choices=['probe','numeric','performance','guards','mlp','diagnose','diagnose-small','diagnose-pad','diagnose-tiles'],default='probe'); parser.add_argument('--child',action='store_true')
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.child:return execute(args.case)
    with (OUT/f'{args.case}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--child','--case',args.case],stdout=log,stderr=log,timeout=300)
    print((OUT/f'{args.case}.log').read_text(encoding='utf-8'),flush=True)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())

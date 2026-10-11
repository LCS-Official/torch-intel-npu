"""Isolated causal Transformer calibration/numerical/performance checks."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/transformer'


def execute(case):
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from torch_npu_transformer import FrozenTransformer
    torch.manual_seed(1041)
    report=dict(case=case,status='running',checks=[],timings=[],sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        for p in ['torch_npu_transformer.py','torch_npu_transformer_demo.py','torch_npu_lowbit.py','torch_intel_npu.py']})
    def source(d=128,layers=1,affine=False):
        block=torch.nn.TransformerEncoderLayer(d,d//32,dim_feedforward=4*d,dropout=0,activation='gelu',
                                               batch_first=True,norm_first=True)
        src=torch.nn.TransformerEncoder(block,layers,enable_nested_tensor=False).eval()
        if affine:
            with torch.no_grad():
                for layer in src.layers:
                    for norm in (layer.norm1,layer.norm2):
                        norm.weight.uniform_(.8,1.2);norm.bias.uniform_(-.03,.03)
        return src
    def prepare(src,mode):
        samples=[torch.randn(b,t,src.layers[0].self_attn.embed_dim)*scale for b,t,scale in [(1,16,.5),(2,32,1.),(1,64,2.)]]
        return FrozenTransformer.from_float(src,mode=mode,calibration_samples=samples if mode=='mixed_int8' else ())
    def compare(model,host,tag):
        ref,trace=model.reference_cpu(host)
        original,_=model.reference_cpu(host,quantized=False)
        x=host.to('intel_npu');before=npu.stats();actual=model(x).cpu();after=npu.stats()
        np.savez(OUT/f'{tag}.npz',host=host.numpy(),actual=actual.numpy(),reference=ref.numpy(),original=original.numpy())
        error=(actual.float()-ref.float()).abs();l2=float(torch.linalg.vector_norm(actual.float()-ref.float())/torch.linalg.vector_norm(ref.float()))
        print(dict(tag=tag,max_abs=float(error.max()),relative_l2=l2),flush=True)
        segments=[]
        if model.mode=='mixed_int8':
            value=x
            for i,block in enumerate(model._blocks):
                for j,segment in enumerate(('attention','ffn')):
                    normalized=model._norm_device(value,block['norm'][j],i,j)
                    host_value=value.cpu();host_norm=normalized.cpu()
                    segment_ref,local=model.reference_cpu(host_value,segment=segment,block_index=i,normalized=host_norm)
                    segment_out=model._run_graph(value,normalized,segment=segment,block_index=i)
                    segment_actual=segment_out.cpu()
                    torch.testing.assert_close(segment_actual,segment_ref,rtol=4e-3,atol=2e-3)
                    segment_l2=float(torch.linalg.vector_norm(segment_actual.float()-segment_ref.float())/torch.linalg.vector_norm(segment_ref.float()))
                    assert segment_l2<1e-3
                    gamma,beta,eps=block['norm'][j]
                    cpu_norm=torch.nn.functional.layer_norm(host_value.float(),(model._d,),
                        torch.from_numpy(np.frombuffer(gamma,dtype=np.float16).copy()).float(),
                        torch.from_numpy(np.frombuffer(beta,dtype=np.float16).copy()).float(),eps).half()
                    step=model.activation_scales[i][0 if j==0 else 2]
                    changes=int((torch.round(host_norm.float()/step)!=torch.round(cpu_norm.float()/step)).sum())
                    trajectory_norm=trace[f'b{i}_{"qkv" if j==0 else "ff1"}_input'].reshape_as(host_norm)
                    trajectory_changes=int((torch.round(host_norm.float()/step)!=torch.round(trajectory_norm.float()/step)).sum())
                    segments.append(dict(block=i,segment=segment,relative_l2=segment_l2,norm_code_changes=changes,
                                         trajectory_code_changes=trajectory_changes))
                    value=segment_out
            print(dict(tag=tag,fixed_input_segments=segments),flush=True)
        # Whole mixed trajectories can diverge by a quantization bin after a
        # rounded residual. Check L2 here and fixed-input segments separately.
        if model.mode!='mixed_int8':torch.testing.assert_close(actual,ref,rtol=4e-3,atol=2e-3*len(model._blocks))
        # Each fixed-input segment remains below 0.1%. Across up to two
        # quantized blocks, bin-boundary changes need a separate 0.5% gate.
        assert l2<(5e-3 if model.mode=='mixed_int8' else 2e-3),('execution relative L2',l2)
        quality=float(torch.linalg.vector_norm(actual.float()-original.float())/torch.linalg.vector_norm(original.float()))
        assert quality<.02,('quality relative L2',quality)
        clipping={}
        if model.activation_scales is not None:
            for i,scales in enumerate(model.activation_scales):
                for name,step in [('qkv',scales[0]),('ff1',scales[2])]:
                    activation=trace[f'b{i}_{name}_input']
                    clipping[f'b{i}_{name}']=int(((activation < -128*step)|(activation > 127*step)).sum())
            assert not any(clipping.values()),('held-out activation clipping',clipping)
        before=npu.stats();out=model(x);end=npu.stats()
        for field in ('host_upload_copies','host_download_copies','session_creations'):assert before[field]==end[field],field
        calls=end['npu_calls']-before['npu_calls']
        assert calls>1 if model.mode=='mixed_int8' else calls==1
        entry=dict(tag=tag,mode=model.mode,shape=list(host.shape),status='passed',max_abs_error=float(error.max()),
            execution_relative_l2=l2,quality_relative_l2_original=quality,
            warm_graph_calls=calls,warm_host_copies_and_compiles=0,scales=model.activation_scales,activation_clipped_elements=clipping,
            fixed_input_segments=segments,evidence=list(model._evidence.values()))
        report['checks'].append(entry);print({k:v for k,v in entry.items() if k!='evidence'},flush=True)
        return actual
    try:
        with torch.no_grad():
            if case=='probe':
                src=source();host=(torch.randn(1,16,128)/2).half()
                for mode in ('fp16','w8a16','mixed_int8'):
                    model=prepare(src,mode)
                    compare(model,host,'probe-'+mode)
                    for cache,key in model._keys.items():
                        shape,segment,block_index=cache
                        graph,xml,data=model.graph(shape,segment=segment,block_index=block_index)
                        tag=f'probe-{mode}-{block_index}-{segment}'
                        (OUT/f'{tag}.xml').write_bytes(xml);(OUT/f'{tag}.bin').write_bytes(data)
                        (OUT/f'{tag}.blob').write_bytes(npu._sessions[key][1].graph_blob)
            if case=='diagnose':
                import npunlock as s
                from npunlock.npu4000 import Elf
                src=source();model=prepare(src,'fp16');host=(torch.randn(1,16,128)/2).half()
                print('scales',model.activation_scales,flush=True)
                _,refs=model.reference_cpu(host);x=host.to('intel_npu')
                for stage in ['b0_norm1','b0_qkv','b0_attention','b0_residual1','b0_norm2','b0_ff1','b0_gelu','b0_ff2','b0_residual2']:
                    graph,xml,data=model.graph(tuple(host.shape),stage=stage)
                    result=npu._native.compile_ir(xml,data,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
                    program=s.load_native(result.graph_blob,graph=graph,libraries=npu._native);key=('transformer_prefix',stage)
                    session=npu._native.create_inference_session(program.graph_blob,timeout_ms=10000,parent=npu._root_session)
                    npu._sessions[key]=session,program;npu._session_creations+=1
                    actual=npu._run(key,[x],graph.outputs[0].shape,torch.float16).cpu()
                    ref=refs[stage];error=(actual.float()-ref.float()).abs()
                    entry=dict(stage=stage,max_abs=float(error.max()),relative_l2=float(torch.linalg.vector_norm(actual.float()-ref.float())/torch.linalg.vector_norm(ref.float())))
                    report['checks'].append(entry);print(entry,flush=True)
                    np.savez(OUT/f'{case}-{stage}.npz',actual=actual.numpy(),reference=ref.numpy())
                del x
            if case=='norm-fp32':
                host=(torch.randn(1,16,128)/2).half();x=host.to('intel_npu');v=x.float()
                centered=v-v.mean(-1,keepdim=True)
                variance=(centered*centered).mean(-1,keepdim=True)
                inverse=npu._execute('rsqrt',variance+1e-5)
                out=(centered*inverse).half();actual=out.cpu();ref=torch.nn.functional.layer_norm(host.float(),(128,)).half()
                entry=dict(max_abs=float((actual.float()-ref.float()).abs().max()),code_changes=int((torch.round(actual.float()/.0625)!=torch.round(ref.float()/.0625)).sum()))
                print(entry,flush=True);report['checks'].append(entry)
                torch.testing.assert_close(actual,ref,rtol=1e-3,atol=1e-4)
                del x,v,centered,variance,inverse,out
            if case=='segments':
                model=prepare(source(layers=2,affine=True),'mixed_int8')
                host=(torch.randn(2,16,128)/2).half();x=host.to('intel_npu');value=x
                for i,block in enumerate(model._blocks):
                    for j,segment in enumerate(('attention','ffn')):
                        normalized=model._norm_device(value,block['norm'][j],i,j)
                        ref,_=model.reference_cpu(value.cpu(),segment=segment,block_index=i,normalized=normalized.cpu())
                        out=model._run_graph(value,normalized,segment=segment,block_index=i);actual=out.cpu()
                        torch.testing.assert_close(actual,ref,rtol=4e-3,atol=2e-3)
                        l2=float(torch.linalg.vector_norm(actual.float()-ref.float())/torch.linalg.vector_norm(ref.float()))
                        assert l2<1e-3
                        entry=dict(status='passed',block=i,segment=segment,max_abs=float((actual.float()-ref.float()).abs().max()),relative_l2=l2)
                        report['checks'].append(entry);print(entry,flush=True);value=out
                del x,value,normalized,out,model
            if case=='numeric':
                for d,layers,b,length,scale in [(128,1,1,1,.5),(128,2,2,17,1.),(256,2,1,64,2.),
                                              (128,2,1,32,.5),(256,1,2,17,1.)]:
                    src=source(d,layers,affine=layers==2)
                    if layers==2:
                        for name,param in src.layers[1].named_parameters():
                            if 'norm' not in name:param.add_(torch.randn_like(param)*.01)
                    host=(torch.randn(b,length,d)*scale).half()
                    cpu=copy.deepcopy(src).half()
                    standard=cpu(host,mask=torch.triu(torch.ones(length,length,dtype=torch.bool),diagonal=1))
                    for mode in ('fp16','w8a16','mixed_int8'):
                        model=prepare(src,mode)
                        original,_=model.reference_cpu(host,quantized=False)
                        semantic=float(torch.linalg.vector_norm(original.float()-standard.float())/torch.linalg.vector_norm(standard.float()))
                        assert semantic<2e-3,('staged vs PyTorch architecture',semantic)
                        tag=f'numeric-d{d}-l{layers}-b{b}-s{length}-{mode}'
                        compare(model,host,tag);report['checks'][-1]['staged_vs_pytorch_relative_l2']=semantic
                        del model
                    del cpu
            if case=='causality':
                src=source(layers=2);host=torch.randn(2,17,128).half()
                changed=host.clone();changed[:,9:]*=-2
                for mode in ('fp16','w8a16','mixed_int8'):
                    model=prepare(src,mode);x=host.to('intel_npu');future=changed.to('intel_npu')
                    a=model(x).cpu();b=model(future).cpu()
                    assert torch.equal(a[:,:9],b[:,:9]),'future tokens affected earlier outputs'
                    assert not torch.equal(a[:,9:],b[:,9:])
                    report['checks'].append(dict(mode=mode,status='passed',prefix_tokens=9,future_tokens=8))
                    del x,future,model
            if case=='guards':
                src=source();calibration=[torch.randn(2,32,128)]
                model=FrozenTransformer(src,mode='mixed_int8',calibration_samples=calibration)
                def rejects(name,fn):
                    before=npu.stats()
                    try:fn()
                    except (ValueError,RuntimeError,NotImplementedError) as error:
                        assert npu.stats()['npu_calls']==before['npu_calls'],name
                        report['checks'].append(dict(name=name,status='rejected_before_execution',reason=str(error)))
                    else:raise AssertionError('Missing guard: '+name)
                rejects('unverified full W8A8',lambda:FrozenTransformer.from_float(src,mode='w8a8'))
                rejects('missing calibration',lambda:FrozenTransformer.from_float(src,mode='mixed_int8'))
                bad=source();bad.train();rejects('training source',lambda:prepare(bad,'fp16'))
                bad=source();bad.layers[0].activation_relu_or_gelu=1;rejects('ReLU',lambda:prepare(bad,'fp16'))
                bad=source();bad.layers[0].dropout.p=.1;rejects('dropout',lambda:prepare(bad,'fp16'))
                bad=source();bad.layers[0].norm1.weight[0]=float('nan');rejects('nonfinite weights',lambda:prepare(bad,'fp16'))
                rejects('nonfinite calibration',lambda:FrozenTransformer(src,calibration_samples=[torch.full((1,16,128),float('inf'))]))
                rejects('CPU forward',lambda:model(torch.randn(1,16,128).half()))
                x=torch.randn(1,16,128).half().to('intel_npu');x.requires_grad_(True)
                with torch.enable_grad():rejects('active autograd',lambda:model(x))
                model.train();rejects('training adapter',lambda:model(x));model.eval();del x
                for name,host in [('FP32 forward',torch.randn(1,16,128)),('wrong width',torch.randn(1,16,64).half()),('sequence ceiling',torch.randn(1,65,128).half())]:
                    x=host.to('intel_npu');rejects(name,lambda:model(x));del x
                x=torch.randn(2,16,128).half().to('intel_npu');part=x[:1]
                rejects('partial storage',lambda:model(part));del part,x
                host=torch.randn(2,17,128).half();dense=host.transpose(0,1).contiguous().to('intel_npu').transpose(0,1)
                actual=model(dense).cpu();contig=host.to('intel_npu');expected=model(contig).cpu()
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                report['checks'].append(dict(name='full dense permutation',status='passed'))
                # Identical snapshots reuse sessions and still expose evidence.
                clone=FrozenTransformer(src,mode='mixed_int8',calibration_samples=calibration)
                before=npu.stats();out=clone(contig);after=npu.stats()
                assert before['session_creations']==after['session_creations'] and clone._evidence
                torch.testing.assert_close(out.cpu(),expected,rtol=0,atol=0)
                with torch.no_grad():src.layers[0].linear1.weight.add_(1)
                assert torch.equal(model(contig).cpu(),expected),'source mutation changed frozen snapshot'
                report['checks'].append(dict(name='snapshot immutability and shared cache evidence',status='passed'))
                del model,clone,dense,contig,out
            if case=='performance':
                torch.set_num_threads(8)
                for d,layers,b,length in [(128,1,1,1),(128,1,1,32),(256,2,1,64),(256,2,8,64)]:
                    src=source(d,layers);host=(torch.randn(b,length,d)/2).half()
                    models={mode:prepare(src,mode) for mode in ('fp16','w8a16','mixed_int8')}
                    for mode,model in models.items():compare(model,host,f'performance-d{d}-l{layers}-b{b}-s{length}-{mode}')
                    cpu=copy.deepcopy(src).half();cpu32=copy.deepcopy(cpu).float();host32=host.float()
                    mask=torch.triu(torch.ones(length,length,dtype=torch.bool),diagonal=1)
                    x=host.to('intel_npu')
                    run={mode:(lambda m=model:m(x)) for mode,model in models.items()}
                    run['cpu_fp16']=lambda:cpu(host,mask=mask)
                    run['cpu_fp32']=lambda:cpu32(host32,mask=mask)
                    for _ in range(3):
                        for fn in run.values():fn()
                    before=npu.stats();samples={mode:[] for mode in run};tags=list(run)
                    for round_index in range(21):
                        index=round_index%len(tags)
                        for tag in tags[index:]+tags[:index]:
                            start=time.perf_counter();run[tag]();samples[tag].append((time.perf_counter()-start)*1000)
                    after=npu.stats()
                    for field in ('host_upload_copies','host_download_copies','session_creations'):assert before[field]==after[field],field
                    entry=dict(shape=[b,length,d],layers=layers,median_ms={k:statistics.median(v) for k,v in samples.items()},
                        samples_ms=samples,cpu_threads=torch.get_num_threads(),warm_host_copies_and_compiles=0,
                        graph_constant_bytes={mode:sum(e['constant_bytes'] for e in model._evidence.values()) for mode,model in models.items()})
                    report['timings'].append(entry);print({k:v for k,v in entry.items() if k!='samples_ms'},flush=True)
                    run.clear();models.clear();del model,x,cpu,cpu32,fn
        if 'model' in locals():del model
        gc.collect();report['stats_before_shutdown']=npu.stats()
        assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
        npu.shutdown();report['final_stats']=npu.stats()
        assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
        report['status']='completed_diagnostic' if case.startswith('diagnose') else 'passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats())
        print(report['traceback'],flush=True)
    (OUT/f'{case}.json').write_text(json.dumps(report,indent=2))
    return int(report['status'] not in ('passed','completed_diagnostic'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['probe','diagnose','norm-fp32','segments','numeric','causality','guards','performance'],default='probe');parser.add_argument('--child',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.child:return execute(args.case)
    with (OUT/f'{args.case}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--child','--case',args.case],stdout=log,stderr=log,timeout=300)
    print((OUT/f'{args.case}.log').read_text(encoding='utf-8'),flush=True)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())

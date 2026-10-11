"""Offline checkpoint, official CPU reference, isolated SmolLM NPU checks."""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'models/SmolLM2-135M-Instruct'
OUT=ROOT/'npunlock-probe/research/results/smollm'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'


def execute(case):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch_npu_smollm import FrozenSmolLayer, _rmsnorm_kernel
    from npunlock.npu4000 import compile_rmsnorm
    torch.set_num_threads(8);torch.manual_seed(1091)
    report=dict(case=case,status='running',checks=[],timings=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['torch_npu_smollm.py','torch_npu_smollm_demo.py','torch_npu_transformer.py','torch_npu_lowbit.py','torch_intel_npu.py',
                     'npunlock/python/npunlock/npu4000.py','npunlock/tests/fixtures/npu4000/rmsnorm.c']},
        packages={name:importlib.metadata.version(name) for name in ['torch','numpy','safetensors','tokenizers','transformers','huggingface_hub']},
        checkpoint=json.loads((MODEL/'download-manifest.json').read_text()),
        kernel_manifest=json.loads((ROOT/'npunlock/tests/fixtures/npu4000/rmsnorm/manifest.json').read_text()))
    npu=None
    try:
        with torch.no_grad():
            tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
            cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32 if case=='cpu' else torch.float16,
                attn_implementation='eager',local_files_only=True).eval()
            if case=='cpu':
                text=tokenizer.apply_chat_template([dict(role='user',content='Explain why the sky is blue in one sentence.')],tokenize=False,add_generation_prompt=True)
                inputs=tokenizer(text,return_tensors='pt');ids=inputs.input_ids
                start=time.perf_counter();generated=cpu.generate(**inputs,max_new_tokens=12,do_sample=False)
                entry=dict(input_tokens=ids.shape[1],output_tokens=generated.shape[1]-ids.shape[1],elapsed_ms=(time.perf_counter()-start)*1000,
                    prompt=text,generated_text=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True),dtype=str(next(cpu.parameters()).dtype))
                print(entry,flush=True);report['checks'].append(entry)
            else:
                import torch_intel_npu as npu
                def real_input(text,index=0,length=None,batch=1):
                    ids=tokenizer(text,return_tensors='pt').input_ids
                    if length is not None:
                        ids=ids.repeat(1,(length+ids.shape[1]-1)//ids.shape[1])[:,:length]
                    ids=ids.repeat(batch,1)
                    result=cpu(ids,use_cache=False,output_hidden_states=True)
                    return result.hidden_states[index].contiguous(),ids
                def prepare(host):
                    b,length,d=host.shape;positions=torch.arange(length)[None].expand(b,-1)
                    rope=cpu.model.rotary_emb(host,positions)
                    mask=torch.triu(torch.full((length,length),torch.finfo(host.dtype).min,dtype=host.dtype),diagonal=1)[None,None]
                    return dict(attention_mask=mask,position_ids=positions,position_embeddings=rope,use_cache=False)
                def official(layer,host,prepared=None):
                    return layer(host,**(prepare(host) if prepared is None else prepared))
                def errors(actual,ref):
                    delta=(actual.float()-ref.float()).abs();rms=ref.float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-5)
                    return dict(relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref.float()).clamp_min(1e-5)),
                        max_abs=float(delta.max()),max_scaled_error=float((delta/(ref.float().abs()+rms)).max()),
                        original_fixed_atol_mismatches=int((delta>(.005+.01*ref.float().abs())).sum()))
                def check(host,index,mode,tag):
                    model=FrozenSmolLayer.from_checkpoint(MODEL,layer_index=index,mode=mode)
                    ref,trace=model.reference_cpu(host);original=official(cpu.model.layers[index],host)
                    staged,_=model.reference_cpu(host,quantized=False)
                    semantic=float(torch.linalg.vector_norm(staged.float()-original.float())/torch.linalg.vector_norm(original.float()))
                    assert semantic<.001,('staged vs official CPU',semantic)
                    x=host.to('intel_npu');start=time.perf_counter();out=model(x);cold_ms=(time.perf_counter()-start)*1000;actual=out.cpu()
                    np.savez(OUT/f'{tag}.npz',host=host.numpy(),actual=actual.numpy(),reference=ref.numpy(),official=original.numpy())
                    execution=errors(actual,ref);quality=errors(actual,original)
                    print(dict(tag=tag,execution=execution,quality_l2=quality['relative_l2'],semantic_l2=semantic),flush=True)
                    value=x;segments=[]
                    for segment,name in [('attention','input_layernorm.weight'),('ffn','post_attention_layernorm.weight')]:
                        normalized=model._norm_device(value,name)
                        host_value=value.cpu();host_norm=normalized.cpu()
                        norm_ref=(host_value.float()*torch.rsqrt(host_value.float().square().mean(-1,keepdim=True)+1e-5)).half()*model._cpu_weight(name)
                        torch.testing.assert_close(host_norm,norm_ref,rtol=.002,atol=.001)
                        segment_ref,local=model.reference_cpu(host_value,segment=segment,normalized=host_norm)
                        segment_out=model._run_graph(value,normalized,segment=segment);got=segment_out.cpu()
                        error=errors(got,segment_ref)
                        assert error['relative_l2']<.001 and error['max_scaled_error']<.01,(segment,error)
                        segments.append(dict(segment=segment,**error))
                        value=segment_out
                    print(dict(tag=tag,segments=segments),flush=True)
                    # The original fixed-atol result stays visible. Trained layers
                    # have different activation scales; also bound every element
                    # relative to its token RMS, including values near zero.
                    assert execution['relative_l2']<.003 and execution['max_scaled_error']<.01,execution
                    assert quality['relative_l2']<.03,quality
                    before=npu.stats();warm=model(x);after=npu.stats()
                    for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field],field
                    # The default whole-layer graph includes both RMSNorms.
                    calls=after['npu_calls']-before['npu_calls'];assert calls==1,(tag,calls)
                    entry=dict(tag=tag,index=index,mode=mode,shape=list(host.shape),status='passed',execution_relative_l2=execution['relative_l2'],
                        quality_relative_l2_official=quality['relative_l2'],staged_relative_l2_official=semantic,cold_compile_run_ms=cold_ms,
                        execution=execution,segments=segments,warm_graph_calls=calls,warm_host_copies_and_compiles=0,evidence=list(model._evidence.values()))
                    report['checks'].append(entry)
                    return model
                if case in ('probe','numeric','boundary'):
                    configs={'probe':[(0,1,16)],'numeric':[(0,1,1),(0,2,17),(10,1,32),(20,1,64)],'boundary':[(29,8,64)]}[case]
                    for index,batch,length in configs:
                        host,ids=real_input('Once upon a time, a curious child asked how birds fly.',index,length,batch)
                        for mode in ['fp16','w8a16']:
                            tag=f'{case}-l{index}-b{batch}-s{length}-{mode}'
                            model=check(host,index,mode,tag)
                            if case=='probe':
                                for cache,key in model._keys.items():
                                    graph,xml,data=model.graph(tuple(host.shape),segment=cache[1])
                                    (OUT/f'{tag}-{cache[1]}.xml').write_bytes(xml);(OUT/f'{tag}-{cache[1]}.bin').write_bytes(data)
                                    (OUT/f'{tag}-{cache[1]}.blob').write_bytes(npu._sessions[key][1].graph_blob)
                            del model
                if case=='diagnose':
                    import npunlock as s
                    host,ids=real_input('Once upon a time, a curious child asked how birds fly.',length=16)
                    model=FrozenSmolLayer.from_checkpoint(MODEL,mode='fp16');_,trace=model.reference_cpu(host);x=host.to('intel_npu')
                    for stage in ['norm1','attention','residual1','norm2','silu','gated','output']:
                        graph,xml,data=model.graph(tuple(host.shape),stage=stage)
                        result=npu._native.compile_ir(xml,data,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
                        program=s.load_native(result.graph_blob,graph=graph,libraries=npu._native)
                        session=npu._native.create_inference_session(program.graph_blob,timeout_ms=10000,parent=npu._root_session)
                        key=('smol_prefix',stage);npu._sessions[key]=session,program;npu._session_creations+=1
                        actual=npu._run(key,[x],graph.outputs[0].shape,torch.float16).cpu();ref=trace[stage]
                        entry=dict(stage=stage,max_abs=float((actual.float()-ref.float()).abs().max()),relative_l2=float(torch.linalg.vector_norm(actual.float()-ref.float())/torch.linalg.vector_norm(ref.float())))
                        print(entry,flush=True);report['checks'].append(entry)
                        np.savez(OUT/f'diagnose-{stage}.npz',actual=actual.numpy(),reference=ref.numpy())
                    del x,model
                if case=='guards':
                    host,ids=real_input('A bicycle has two wheels.',length=17,batch=2)
                    model=FrozenSmolLayer.from_checkpoint(MODEL)
                    def rejects(name,fn):
                        before=npu.stats()
                        try:fn()
                        except (ValueError,RuntimeError,NotImplementedError) as error:
                            after=npu.stats()
                            assert before['npu_calls']==after['npu_calls'] and before['session_creations']==after['session_creations'],name
                            report['checks'].append(dict(name=name,status='rejected_before_execution',reason=str(error)))
                        else:raise AssertionError('Missing guard: '+name)
                    rejects('CPU forward',lambda:model(host))
                    rejects('unverified W8A8',lambda:FrozenSmolLayer.from_checkpoint(MODEL,mode='w8a8'))
                    rejects('invalid layer index',lambda:FrozenSmolLayer.from_checkpoint(MODEL,layer_index=-1))
                    config=json.loads((MODEL/'config.json').read_text());weights={name:model._cpu_weight(name) for name in model._weights}
                    bad=dict(config,rope_theta=10000);rejects('different RoPE',lambda:FrozenSmolLayer(bad,weights))
                    weights['input_layernorm.weight'][0]=float('nan');rejects('nonfinite weight',lambda:FrozenSmolLayer(config,weights))
                    weights['input_layernorm.weight'][0]=1
                    weights['input_layernorm.weight']=weights['input_layernorm.weight'].float();weights['input_layernorm.weight'][0]=1e10
                    rejects('FP16 weight overflow',lambda:FrozenSmolLayer(config,weights))
                    rejects('nonfinite reference',lambda:model.reference_cpu(torch.full_like(host,float('inf'))))
                    for name,value in [('FP32 forward',host.float()),('sequence ceiling',torch.zeros(1,65,576).half()),
                                       ('batch ceiling',torch.zeros(9,1,576).half()),('wrong width',torch.zeros(1,17,512).half())]:
                        x=value.to('intel_npu');rejects(name,lambda:model(x));del x
                    x=host.to('intel_npu');part=x[:1];rejects('partial storage',lambda:model(part));del part
                    x.requires_grad_(True)
                    with torch.enable_grad():rejects('active autograd',lambda:model(x))
                    x.requires_grad_(False);model.train();rejects('training adapter',lambda:model(x));model.eval()
                    expected=model(x).cpu();dense=host.transpose(0,1).contiguous().to('intel_npu').transpose(0,1)
                    actual=model(dense).cpu();assert torch.equal(actual,expected)
                    report['checks'].append(dict(name='dense permutation',status='passed'))
                    clone=FrozenSmolLayer.from_checkpoint(MODEL);before=npu.stats();out=clone(x)
                    assert before['session_creations']==npu.stats()['session_creations'] and clone._evidence
                    assert torch.equal(out.cpu(),expected)
                    with torch.no_grad():cpu.model.layers[0].mlp.gate_proj.weight.add_(1)
                    assert torch.equal(model(x).cpu(),expected)
                    report['checks'].append(dict(name='snapshot and cache evidence',status='passed'))
                    del model,clone,x,dense,out
                if case=='causality':
                    host,ids=real_input('A curious child asked how birds fly.',length=17)
                    changed=host.clone();changed[:,9:]*=-2
                    for mode in ['fp16','w8a16']:
                        model=FrozenSmolLayer.from_checkpoint(MODEL,mode=mode);x=host.to('intel_npu');future=changed.to('intel_npu')
                        a=model(x).cpu();b=model(future).cpu()
                        assert torch.equal(a[:,:9],b[:,:9]) and not torch.equal(a[:,9:],b[:,9:])
                        report['checks'].append(dict(mode=mode,status='passed',prefix_tokens=9,future_tokens=8))
                        del x,future,model
                if case=='hybrid':
                    # Demo only: one decoder layer offloaded; all other layers,
                    # embedding, final norm and logits explicitly remain on CPU.
                    class Bridge(torch.nn.Module):
                        def __init__(self,adapter):super().__init__();self.adapter=adapter;self.calls=0
                        def forward(self,hidden_states,*,position_ids=None,past_key_values=None,use_cache=False,attention_mask=None,**kwargs):
                            length=hidden_states.shape[1]
                            if use_cache or past_key_values is not None:raise ValueError('Bridge has no KV cache')
                            if position_ids is None or not torch.equal(position_ids,torch.arange(length)[None]):
                                raise ValueError('Bridge requires zero-based full-sequence positions')
                            if attention_mask is not None:
                                if attention_mask.shape!=(1,1,length,length):raise ValueError('Bridge requires single unpadded sequence')
                                forbidden=torch.triu(torch.ones(length,length,dtype=torch.bool),diagonal=1)[None,None]
                                if not torch.equal(attention_mask< -1e4,forbidden) or not torch.all(attention_mask[~forbidden]==0):
                                    raise ValueError('Bridge requires ordinary causal mask')
                            self.calls+=1
                            return self.adapter(hidden_states.to('intel_npu')).cpu()
                    original_layer=cpu.model.layers[0]
                    for prompt in ['What is two plus two?','Complete: Paris is the capital of']:
                        text=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
                        inputs=tokenizer(text,return_tensors='pt');assert inputs.input_ids.shape[1]+8<=64
                        baseline=cpu(**inputs,use_cache=False).logits[:,-1].float()
                        expected=cpu.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=False)
                        for mode in ['fp16','w8a16']:
                            adapter=FrozenSmolLayer.from_checkpoint(MODEL,mode=mode);bridge=Bridge(adapter)
                            cpu.model.layers[0]=bridge
                            try:
                                actual=cpu(**inputs,use_cache=False).logits[:,-1].float()
                                generated=cpu.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=False)
                            finally:cpu.model.layers[0]=original_layer
                            logits_l2=float(torch.linalg.vector_norm(actual-baseline)/torch.linalg.vector_norm(baseline))
                            logp=baseline.log_softmax(-1);logq=actual.log_softmax(-1)
                            kl=float((logp.exp()*(logp-logq)).sum())
                            assert torch.isfinite(actual).all() and logits_l2<.03 and kl<.02
                            entry=dict(prompt=prompt,mode=mode,input_tokens=inputs.input_ids.shape[1],npu_layers=1,cpu_layers=29,
                                use_cache=False,logits_relative_l2=logits_l2,next_token_kl=kl,
                                next_token_equal=bool(actual.argmax(-1)==baseline.argmax(-1)),generated_tokens_equal=bool(torch.equal(generated,expected)),
                                reference_text=tokenizer.decode(expected[0,inputs.input_ids.shape[1]:],skip_special_tokens=True),
                                generated_text=tokenizer.decode(generated[0,inputs.input_ids.shape[1]:],skip_special_tokens=True),offloaded_forwards=bridge.calls)
                            print(entry,flush=True);report['checks'].append(entry)
                            del adapter,bridge
                if case=='norm':
                    model=FrozenSmolLayer.from_checkpoint(MODEL,mode='fp16')
                    legacy=FrozenSmolLayer.from_checkpoint(MODEL,mode='fp16')
                    patterns=[('single',torch.randn(1,1,576).half()),
                        ('serial7',torch.randn(1,7,576).half()),('parallel8',torch.randn(1,8,576).half()),
                        ('parallel9',torch.randn(1,9,576).half()),('tail17',torch.randn(1,17,576).half()),
                        ('large',torch.randn(4,64,576).half()),('maximum',torch.randn(8,64,576).half()),
                        ('chunk-tail',torch.randn(7,37,576).half()),('zero',torch.zeros(1,3,576).half()),
                        ('half-square-overflow',torch.full((1,3,576),65504.).half())]
                    bits=np.resize(np.array([0,1,3,1023,0x8000,0x8001,0x8003,0x83ff],dtype=np.uint16),576)
                    patterns.append(('subnormal',torch.from_numpy(bits.view(np.float16).copy()).view(1,1,576)))
                    overflow=patterns[-2][1];overflow[...,1::2]*=-1
                    rounded_affine_differences=0
                    for tag,host in patterns:
                        x=host.to('intel_npu')
                        for name in ['input_layernorm.weight','post_attention_layernorm.weight']:
                            before=npu.stats();actual=model._norm_device(x,name);after=npu.stats();got=actual.cpu()
                            old=legacy._norm_device_legacy(x,name).cpu()
                            assert torch.equal(got.view(torch.int16),old.view(torch.int16)),(tag,name,'legacy bits differ')
                            value=host.float();normalized=value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-5)
                            gamma=model._cpu_weight(name);ref=normalized.half()*gamma
                            torch.testing.assert_close(got,ref,rtol=.002,atol=.001)
                            rounded_affine_differences+=int((ref.view(torch.int16)!=(normalized*gamma.float()).half().view(torch.int16)).sum())
                            warm_before=npu.stats();warm=model._norm_device(x,name);warm_after=npu.stats()
                            for field in ['host_upload_copies','host_download_copies','session_creations']:
                                assert warm_before[field]==warm_after[field],field
                            calls=warm_after['npu_calls']-warm_before['npu_calls'];assert calls==1
                            entry=dict(tag=tag,weight=name,shape=list(host.shape),legacy_bitwise_equal=True,
                                cpu_half_code_changes=int((got.view(torch.int16)!=ref.view(torch.int16)).sum()),
                                warm_graph_calls=calls,warm_host_copies_and_compiles=0,**errors(got,ref))
                            print(entry,flush=True);report['checks'].append(entry);del actual,warm
                        del x
                    assert rounded_affine_differences>0,'fixture must detect skipped half rounding before gamma'
                    report['rounded_affine_differences']=rounded_affine_differences
                    report['kernel_manifest']=json.loads((ROOT/'npunlock/tests/fixtures/npu4000/rmsnorm/manifest.json').read_text())
                    report['kernel_evidence']=[json.loads(program.patch_reports[0]) for key,(session,program) in npu._sessions.items() if key[0]=='smollm_rmsnorm']
                    for key,(session,program) in npu._sessions.items():
                        if key[0]=='smollm_rmsnorm':(OUT/f'rmsnorm-{key[1]}.blob').write_bytes(program.graph_blob)
                    report['carrier_guards']=[]
                    for rows,parallel in [(0,True),(513,True),(1024,True),(257,False),(512,False)]:
                        before=npu.stats()
                        try:compile_rmsnorm(rows,_rmsnorm_kernel(),libraries=npu._native,parallel=parallel)
                        except ValueError:
                            assert before['session_creations']==npu.stats()['session_creations']
                            report['carrier_guards'].append(dict(rows=rows,parallel=parallel,status='rejected_before_execution'))
                        else:raise AssertionError('missing rows guard')
                    del model,legacy
                if case=='performance':
                    class LegacyLayer(FrozenSmolLayer):_norm_device=FrozenSmolLayer._norm_device_legacy
                    class SerialLayer(FrozenSmolLayer):
                        def _norm_device(self,x,name,**kwargs):return super()._norm_device(x,name,parallel=False,**kwargs)
                    for batch,length in [(1,1),(1,32),(4,64)]:
                        host,ids=real_input('Explain how a bicycle works in simple words.',length=length,batch=batch)
                        models={mode:check(host,0,mode,f'perf-b{batch}-s{length}-{mode}') for mode in ['fp16','w8a16']}
                        layer16=cpu.model.layers[0];layer32=copy.deepcopy(layer16).float();host32=host.float();x=host.to('intel_npu')
                        legacy=LegacyLayer.from_checkpoint(MODEL,mode='w8a16');models['w8a16_legacy']=legacy
                        assert torch.equal(models['w8a16'](x).cpu().view(torch.int16),legacy(x).cpu().view(torch.int16))
                        serial=SerialLayer.from_checkpoint(MODEL,mode='w8a16');models['w8a16_serial']=serial
                        assert torch.equal(models['w8a16'](x).cpu().view(torch.int16),serial(x).cpu().view(torch.int16))
                        run={mode:(lambda m=model:m(x)) for mode,model in models.items()}
                        prepared16=prepare(host);prepared32=prepare(host32)
                        run['cpu_fp16']=lambda:official(layer16,host,prepared16);run['cpu_fp32']=lambda:official(layer32,host32,prepared32)
                        for _ in range(3):
                            for fn in run.values():fn()
                        before=npu.stats();samples={name:[] for name in run};names=list(run)
                        for round_index in range(21):
                            order=round_index%len(names)
                            for name in names[order:]+names[:order]:
                                start=time.perf_counter();run[name]();samples[name].append((time.perf_counter()-start)*1000)
                        after=npu.stats()
                        for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field],field
                        entry=dict(shape=list(host.shape),cpu_threads=torch.get_num_threads(),median_ms={k:statistics.median(v) for k,v in samples.items()},
                            samples_ms=samples,constant_bytes={mode:sum(e['constant_bytes'] for e in m._evidence.values()) for mode,m in models.items()},warm_host_copies_and_compiles=0)
                        print({k:v for k,v in entry.items() if k!='samples_ms'},flush=True);report['timings'].append(entry)
                        run.clear();models.clear();del x,layer32,fn,legacy,serial
                if case=='profile':
                    host,ids=real_input('Explain how a bicycle works in simple words.',length=64,batch=4)
                    model=check(host,0,'w8a16','profile-b4-s64-w8a16');x=host.to('intel_npu')
                    norm1=model._norm_device(x,'input_layernorm.weight');residual=model._run_graph(x,norm1,segment='attention')
                    norm2=model._norm_device(residual,'post_attention_layernorm.weight')
                    run={'norm1':lambda:model._norm_device(x,'input_layernorm.weight'),
                         'attention':lambda:model._run_graph(x,norm1,segment='attention'),
                         'norm2':lambda:model._norm_device(residual,'post_attention_layernorm.weight'),
                         'ffn':lambda:model._run_graph(residual,norm2,segment='ffn')}
                    samples={name:[] for name in run};calls={}
                    for name,fn in run.items():
                        before=npu.stats();fn();after=npu.stats();calls[name]=after['npu_calls']-before['npu_calls']
                    before=npu.stats()
                    for _ in range(9):
                        for name,fn in run.items():
                            start=time.perf_counter();fn();samples[name].append((time.perf_counter()-start)*1000)
                    after=npu.stats()
                    for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field],field
                    entry=dict(shape=list(host.shape),mode='w8a16',median_ms={name:statistics.median(values) for name,values in samples.items()},
                        samples_ms=samples,graph_calls=calls,warm_host_copies_and_compiles=0)
                    print(entry,flush=True);report['timings'].append(entry)
                    run.clear();del fn,model,x,norm1,norm2,residual
                del cpu
                gc.collect();report['stats_before_shutdown']=npu.stats()
                assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers']
                npu.shutdown();report['final_stats']=npu.stats()
                assert npu.stats()['live_buffers']==0 and not npu.stats()['context_failed']
        report['status']='completed_diagnostic' if case=='diagnose' else 'passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if npu is not None:report['failure_stats']=npu.stats()
        print(report['traceback'],flush=True)
    (OUT/f'{case}.json').write_text(json.dumps(report,indent=2))
    return int(report['status'] not in ['passed','completed_diagnostic'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['cpu','probe','numeric','boundary','guards','causality','hybrid','norm','diagnose','performance','profile'],default='probe')
    parser.add_argument('--child',action='store_true');args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.child:return execute(args.case)
    with (OUT/f'{args.case}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--case',args.case,'--child'],stdout=log,stderr=log,timeout=300)
    print((OUT/f'{args.case}.log').read_text(encoding='utf-8'),flush=True)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())

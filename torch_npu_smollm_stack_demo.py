"""Offline, isolated checks for resident consecutive SmolLM2 NPU layers."""
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
OUT=ROOT/'npunlock-probe/research/results/smollm-stack'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'


def errors(actual,reference):
    import torch
    delta=(actual.float()-reference.float()).abs()
    rms=reference.float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-5)
    return dict(relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(reference.float()).clamp_min(1e-5)),
        max_abs=float(delta.max()),max_scaled_error=float((delta/(reference.float().abs()+rms)).max()),
        original_fixed_atol_mismatches=int((delta>(.005+.01*reference.float().abs())).sum()))


def execute(case):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch_npu_smollm import FrozenSmolStack
    torch.set_num_threads(8);torch.manual_seed(1091)
    report=dict(case=case,status='running',checks=[],timings=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['torch_npu_smollm.py','torch_npu_smollm_stack_demo.py','torch_npu_transformer.py','torch_npu_lowbit.py',
                     'torch_intel_npu.py','npunlock/python/npunlock/npu4000.py','npunlock/tests/fixtures/npu4000/rmsnorm.c',
                     'npunlock-probe/session-runtime/npunlock_session.dll','npunlock-probe/torch-device/_torch_intel_npu.pyd']},
        packages={name:importlib.metadata.version(name) for name in ['torch','numpy','safetensors','transformers']},
        checkpoint=json.loads((MODEL/'download-manifest.json').read_text()),
        kernel_manifest=json.loads((ROOT/'npunlock/tests/fixtures/npu4000/rmsnorm/manifest.json').read_text()))
    try:
        with torch.no_grad():
            import torch_intel_npu as npu
            def work():
                tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
                cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
                def real_input(index,length,batch):
                    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
                    ids=ids.repeat(1,(length+ids.shape[1]-1)//ids.shape[1])[:,:length].repeat(batch,1)
                    return cpu(ids,use_cache=False,output_hidden_states=True).hidden_states[index].contiguous()
                def prepare(host):
                    batch,length,_=host.shape;positions=torch.arange(length)[None].expand(batch,-1)
                    mask=torch.triu(torch.full((length,length),torch.finfo(host.dtype).min,dtype=host.dtype),diagonal=1)[None,None]
                    return dict(attention_mask=mask,position_ids=positions,position_embeddings=cpu.model.rotary_emb(host,positions),use_cache=False)
                def official(layers,host,prepared):
                    for layer in layers:host=layer(host,**prepared)
                    return host
                def check(index,count,batch,length,mode):
                    tag=f'l{index}-n{count}-b{batch}-s{length}-{mode}'
                    host=real_input(index,length,batch);prepared=prepare(host)
                    stack=FrozenSmolStack.from_checkpoint(MODEL,start_layer=index,num_layers=count,mode=mode)
                    reference,traces=stack.reference_cpu(host)
                    staged,_=stack.reference_cpu(host,quantized=False)
                    baseline=official(cpu.model.layers[index:index+count],host,prepared)
                    x=host.to('intel_npu');start=time.perf_counter();out=stack(x);cold_ms=(time.perf_counter()-start)*1000
                    actual=out.cpu();execution=errors(actual,reference);quality=errors(actual,baseline);semantic=errors(staged,baseline)
                    entry=dict(tag=tag,start_layer=index,num_layers=count,mode=mode,shape=list(host.shape),status='running',
                        execution=execution,quality_official=quality,staged_official=semantic,cold_compile_run_ms=cold_ms)
                    report['checks'].append(entry)
                    print(dict(tag=tag,execution=execution,quality_l2=quality['relative_l2'],semantic_l2=semantic['relative_l2']),flush=True)
                    np.savez(OUT/f'{case}-{tag}.npz',host=host.numpy(),actual=actual.numpy(),reference=reference.numpy(),official=baseline.numpy())
                    assert torch.isfinite(actual).all(),tag
                    assert semantic['relative_l2']<.001,('staged vs official',entry)
                    assert execution['relative_l2']<.003 and execution['max_scaled_error']<.01,('execution',entry)
                    assert quality['relative_l2']<.03,('quality',entry)
                    before=npu.stats();warm=stack(x);after=npu.stats()
                    assert torch.equal(warm.cpu().view(torch.int16),actual.view(torch.int16)),tag
                    for field in ['host_upload_copies','host_download_copies','session_creations']:
                        assert before[field]==after[field],(tag,field)
                    # The default path submits one fused graph per layer.
                    calls=after['npu_calls']-before['npu_calls'];assert calls==count,(tag,calls)
                    # Check each layer on its actual predecessor's output. These
                    # diagnostic downloads are outside the resident warm call.
                    local=[];value=x
                    for offset,layer in enumerate(stack.layers):
                        layer_ref,_=layer.reference_cpu(value.cpu());value=layer(value)
                        error=errors(value.cpu(),layer_ref)
                        assert error['relative_l2']<.003 and error['max_scaled_error']<.01,('local layer',index+offset,error)
                        local.append(dict(layer=index+offset,**error))
                    assert torch.equal(value.cpu().view(torch.int16),actual.view(torch.int16))
                    evidence=[dict(layer=index+offset,shape=list(key[0]),segment=key[1],**item)
                        for offset,layer in enumerate(stack.layers) for key,item in layer._evidence.items()]
                    entry.update(status='passed',local_layers=local,warm_graph_calls=calls,warm_host_copies_and_compiles=0,
                        evidence=evidence,constant_bytes=sum(e['constant_bytes'] for e in evidence),
                        allocator_peak_bytes=after['peak_buffer_bytes'],allocator_live_bytes=after['live_buffer_bytes'])
                    return stack,host,prepared
                if case in ('probe','numeric','boundary'):
                    configs={'probe':[(0,1,16)],'numeric':[(0,1,1),(0,1,32),(10,2,17),(26,4,64)],'boundary':[(0,8,64)]}[case]
                    for index,batch,length in configs:
                        for count in [2,4]:
                            for mode in ['fp16','w8a16']:
                                stack,host,prepared=check(index,count,batch,length,mode)
                                del stack
                if case=='guards':
                    def reject(name,fn):
                        before=npu.stats()
                        try:fn()
                        except ValueError as error:
                            after=npu.stats()
                            for field in ['npu_calls','host_upload_copies','host_download_copies','session_creations']:
                                assert before[field]==after[field],(name,field)
                            report['checks'].append(dict(name=name,status='rejected_before_execution',error=str(error)))
                        else:raise AssertionError(('missing rejection',name))
                    for options in [dict(start_layer=-1),dict(start_layer=True),dict(start_layer=1.),dict(num_layers=0),
                                    dict(num_layers=-1),dict(num_layers=True),dict(num_layers=2.),dict(start_layer=29,num_layers=2),
                                    dict(num_layers=31),dict(mode='nf4')]:
                        reject(str(options),lambda options=options:FrozenSmolStack.from_checkpoint(MODEL,**options))
                    stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=4);host=real_input(0,8,1);x=host.to('intel_npu')
                    reject('CPU input',lambda:stack(host))
                    x32=x.float();reject('FP32 input',lambda:stack(x32))
                    for shape in [(1,65,576),(9,1,576),(1,8,512)]:
                        bad=torch.zeros(shape,dtype=torch.float16).to('intel_npu');reject(str(shape),lambda:stack(bad))
                    reject('partial storage',lambda:stack(x[:,:4]))
                    stack.train();reject('parent training',lambda:stack(x));stack.eval()
                    stack.layers[-1].train();reject('last child training',lambda:stack(x));stack.eval()
                    grad=host.to('intel_npu').requires_grad_()
                    with torch.enable_grad():reject('active autograd',lambda:stack(grad))
                    # Complete dense permutations are materialized on NPU.
                    a=stack(x);permuted=host.transpose(1,2).contiguous().to('intel_npu').transpose(1,2)
                    before=npu.stats();b=stack(permuted);after=npu.stats()
                    assert torch.equal(a.cpu().view(torch.int16),b.cpu().view(torch.int16))
                    for field in ['host_upload_copies','host_download_copies']:assert before[field]==after[field]
                    report['checks'].append(dict(name='dense permutation',status='passed'))
                    # Separate modules reuse compiled graphs, but own their gamma.
                    clone=FrozenSmolStack.from_checkpoint(MODEL,num_layers=4)
                    clone(x);before=npu.stats();cloned=clone(x);after=npu.stats()
                    assert torch.equal(a.cpu().view(torch.int16),cloned.cpu().view(torch.int16))
                    for field in ['session_creations','host_upload_copies','host_download_copies']:assert before[field]==after[field]
                    report['checks'].append(dict(name='snapshot cache reuse',status='passed',warm_graph_calls=after['npu_calls']-before['npu_calls']))
                if case=='causality':
                    host=real_input(0,17,1);changed=host.clone();changed[:,9:]=torch.randn_like(changed[:,9:])
                    for count in [2,4]:
                        for mode in ['fp16','w8a16']:
                            stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=count,mode=mode)
                            actual=stack(host.to('intel_npu')).cpu();future=stack(changed.to('intel_npu')).cpu()
                            assert torch.equal(actual[:,:9],future[:,:9]) and not torch.equal(actual[:,9:],future[:,9:])
                            report['checks'].append(dict(num_layers=count,mode=mode,status='passed',prefix_tokens=9,future_tokens=8))
                if case=='performance':
                    for batch,length in [(1,1),(1,32),(4,64)]:
                        for count in [2,4]:
                            stacks={}
                            for mode in ['fp16','w8a16']:
                                stack,host,prepared=check(0,count,batch,length,mode);stacks[mode]=stack
                            layers16=list(cpu.model.layers[:count]);layers32=copy.deepcopy(torch.nn.ModuleList(layers16)).float()
                            host32=host.float();prepared32=prepare(host32);x=host.to('intel_npu')
                            def transferred(stack):
                                value=host
                                for layer in stack.layers:value=layer(value.to('intel_npu')).cpu()
                                return value
                            run={mode:(lambda stack=stack:stack(x)) for mode,stack in stacks.items()}
                            run.update({mode+'_stack_copies':(lambda stack=stack:stack(host.to('intel_npu')).cpu()) for mode,stack in stacks.items()})
                            run.update({mode+'_per_layer_copies':(lambda stack=stack:transferred(stack)) for mode,stack in stacks.items()})
                            run.update(cpu_fp16=lambda:official(layers16,host,prepared),cpu_fp32=lambda:official(layers32,host32,prepared32))
                            expected_calls={};expected_copies={}
                            for name,fn in run.items():
                                fn();before=npu.stats();fn();after=npu.stats()
                                expected_calls[name]=after['npu_calls']-before['npu_calls']
                                expected_copies[name]={field:after[field]-before[field] for field in ['host_upload_copies','host_download_copies']}
                                assert after['session_creations']==before['session_creations']
                                if name in stacks:
                                    assert expected_calls[name]==count and not any(expected_copies[name].values())
                                if name.endswith('_per_layer_copies'):
                                    assert expected_calls[name]==count and set(expected_copies[name].values())=={count}
                                if name.endswith('_stack_copies'):
                                    assert expected_calls[name]==count and set(expected_copies[name].values())=={1}
                            samples={name:[] for name in run};names=list(run);before=npu.stats()
                            for round_index in range(21):
                                offset=round_index%len(names)
                                for name in names[offset:]+names[:offset]:
                                    start=time.perf_counter();run[name]();samples[name].append((time.perf_counter()-start)*1000)
                            after=npu.stats()
                            assert after['session_creations']==before['session_creations']
                            for field in ['host_upload_copies','host_download_copies']:
                                assert after[field]-before[field]==21*sum(v[field] for v in expected_copies.values())
                            entry=dict(shape=list(host.shape),num_layers=count,cpu_threads=torch.get_num_threads(),
                                median_ms={name:statistics.median(values) for name,values in samples.items()},samples_ms=samples,
                                graph_calls_per_forward=expected_calls,host_copies_per_forward=expected_copies,
                                constant_bytes={mode:sum(e['constant_bytes'] for layer in stack.layers for e in layer._evidence.values()) for mode,stack in stacks.items()},
                                allocator_process_peak_bytes=after['peak_buffer_bytes'],allocator_scope='cumulative process peak of explicit shared tensors; excludes compiler constants/CMX/driver allocations')
                            print({k:v for k,v in entry.items() if k!='samples_ms'},flush=True);report['timings'].append(entry)
                            run.clear();stacks.clear();del stack,fn,x,layers32
                if case=='hybrid':
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
                    original=cpu.model.layers
                    for prompt in ['What is two plus two?','Complete: Paris is the capital of']:
                        text=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
                        inputs=tokenizer(text,return_tensors='pt');assert inputs.input_ids.shape[1]+8<=64
                        baseline=cpu(**inputs,use_cache=False).logits[:,-1].float()
                        expected=cpu.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=False)
                        for count in [2,4]:
                            for mode in ['fp16','w8a16']:
                                stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=count,mode=mode);bridge=Bridge(stack)
                                cpu.model.layers=torch.nn.ModuleList([bridge,*list(original)[count:]])
                                try:
                                    actual=cpu(**inputs,use_cache=False).logits[:,-1].float()
                                    generated=cpu.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=False)
                                finally:cpu.model.layers=original
                                logits=errors(actual,baseline);logp=baseline.log_softmax(-1);logq=actual.log_softmax(-1)
                                kl=float((logp.exp()*(logp-logq)).sum())
                                entry=dict(prompt=prompt,mode=mode,npu_layers=count,cpu_layers=30-count,use_cache=False,
                                    logits_relative_l2=logits['relative_l2'],next_token_kl=kl,next_token_equal=bool(actual.argmax(-1)==baseline.argmax(-1)),
                                    generated_tokens_equal=bool(torch.equal(generated,expected)),offloaded_forwards=bridge.calls,
                                    generated_text=tokenizer.decode(generated[0,inputs.input_ids.shape[1]:],skip_special_tokens=True),
                                    reference_text=tokenizer.decode(expected[0,inputs.input_ids.shape[1]:],skip_special_tokens=True))
                                report['checks'].append(entry);print(entry,flush=True)
                                assert torch.isfinite(actual).all() and logits['relative_l2']<.03 and kl<.02
                # All locals, closures and adapters die before the cleanup check.
            work();gc.collect();report['stats_before_shutdown']=npu.stats()
            assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers'],npu.stats()
            npu.shutdown();report['final_stats']=npu.stats()
            assert npu.stats()['live_buffers']==0 and not npu.stats()['context_failed']
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if 'npu' in locals():report['failure_stats']=npu.stats()
        print(report['traceback'],flush=True)
    (OUT/f'{case}.json').write_text(json.dumps(report,indent=2))
    return int(report['status']!='passed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=['probe','numeric','boundary','guards','causality','performance','hybrid'],default='probe')
    parser.add_argument('--child',action='store_true');args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.child:return execute(args.case)
    with (OUT/f'{args.case}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--case',args.case,'--child'],stdout=log,stderr=log,timeout=600)
    print((OUT/f'{args.case}.log').read_text(encoding='utf-8'),flush=True)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())

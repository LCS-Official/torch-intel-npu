"""Offline static-KV checks, explicit hybrid generation and warm benchmarks."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from torch_npu_smollm_stack_demo import errors

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'models/SmolLM2-135M-Instruct'
OUT=ROOT/'npunlock-probe/research/results/smollm-kv-batch'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'


def execute(case):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from torch_npu_smollm import FrozenSmolLayer, FrozenSmolStack, SmolKVCache
    torch.set_num_threads(8);torch.manual_seed(1091)
    report=dict(case=case,status='running',checks=[],timings=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['torch_npu_smollm.py','torch_npu_smollm_kv_demo.py','torch_npu_smollm_stack_demo.py','torch_npu_transformer.py',
                     'torch_npu_lowbit.py','torch_intel_npu.py','npunlock/python/npunlock/npu4000.py',
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
                def make(count,mode,index=0):
                    return (FrozenSmolLayer.from_checkpoint(MODEL,layer_index=index,mode=mode) if count==1 else
                        FrozenSmolStack.from_checkpoint(MODEL,start_layer=index,num_layers=count,mode=mode))
                def before_module():
                    path=OUT/'before/torch_npu_smollm.py';spec=importlib.util.spec_from_file_location('smollm_before_batch',path)
                    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
                    old.__file__=str(ROOT/'torch_npu_smollm.py')
                    kernel_path=OUT/'before/rmsnorm-f16.elf'
                    if kernel_path.exists():
                        kernel=kernel_path.read_bytes();old._rmsnorm_kernel=lambda:kernel
                    return old
                def states(state):return state if type(state) is tuple else (state,)
                def real_input(index,length,batch=1):
                    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
                    # Distinct sequences expose cache slicing/assembly mistakes
                    # which identical samples would hide.
                    stream=ids.repeat(1,(length+batch-1+ids.shape[1]-1)//ids.shape[1])
                    ids=torch.cat([stream[:,i:i+length] for i in range(batch)])
                    return cpu(ids,use_cache=False,output_hidden_states=True).hidden_states[index].contiguous()
                def prepare(host,start=0):
                    batch,length,_=host.shape;positions=torch.arange(start,start+length)[None].expand(batch,-1)
                    mask=torch.where(torch.arange(start+length)[None,:]>torch.arange(start,start+length)[:,None],
                        torch.finfo(host.dtype).min,0).to(host.dtype)[None,None]
                    return dict(attention_mask=mask,position_ids=positions,position_embeddings=cpu.model.rotary_emb(host,positions))
                def official(index,count,host,cache=None,start=0):
                    prepared=prepare(host,start)
                    for layer in cpu.model.layers[index:index+count]:host=layer(host,past_key_values=cache,use_cache=cache is not None,**prepared)
                    return host
                def gate(error,tag):
                    assert error['relative_l2']<.003 and error['max_scaled_error']<.01,(tag,error)
                def check_state(state,reference_traces,length,*,local=False):
                    checks=[]
                    for offset,(cache,trace) in enumerate(zip(states(state),reference_traces)):
                        assert cache.length==length and float(cache._position.cpu()[0])==length
                        for name,tensor in [('keys',cache.keys),('values',cache.values)]:
                            host=tensor.cpu();assert not host[:,:,length:].count_nonzero()
                            error=errors(host[:,:,:length],trace[name])
                            if local:gate(error,('fixed-input KV reference',offset,name))
                            checks.append(dict(layer=offset,tensor=name,**error))
                    return checks
                def local_cache_check(model,x,state,previous=None,history=None):
                    # Replay outside timing, recording each layer's actual
                    # predecessor. Cache projections must be checked on those
                    # fixed inputs; CPU-stack inputs accumulate different rounding.
                    inputs=[];traces=[];layers=[model] if isinstance(model,FrozenSmolLayer) else model.layers
                    for i,(layer,cache) in enumerate(zip(layers,states(state))):
                        host=x.cpu();host=host if history is None else torch.cat((history[i],host),dim=1)
                        inputs.append(host);_,trace=layer.reference_cpu(host);traces.append(trace)
                        x,replay=layer.prefill(x) if previous is None else layer.decode(x,states(previous)[i])
                        for name in ['keys','values']:
                            assert torch.equal(getattr(cache,name).cpu().view(torch.int16),getattr(replay,name).cpu().view(torch.int16))
                    return inputs,check_state(state,traces,states(state)[0].length,local=True)
                def check(index,count,batch,prefix,steps,mode):
                    tag=f'l{index}-n{count}-b{batch}-p{prefix}-t{steps}-{mode}'
                    host=real_input(index,prefix+steps,batch);model=make(count,mode,index)
                    x=host[:,:prefix].contiguous().to('intel_npu');out,state=model.prefill(x)
                    full_input=host[:,:prefix].contiguous()
                    reference,traces=model.reference_cpu(full_input);traces=[traces] if count==1 else traces
                    baseline=official(index,count,full_input);staged,_=model.reference_cpu(full_input,quantized=False)
                    actual=out.cpu();execution=errors(actual,reference);quality=errors(actual,baseline)
                    history,local_checks=local_cache_check(model,x,state)
                    entry=dict(tag=tag,stage='prefill',length=prefix,execution=execution,quality_official=quality,
                        staged_official=errors(staged,baseline),cache_reference=check_state(state,traces,prefix),
                        cache_local_reference=local_checks,status='running')
                    report['checks'].append(entry);print({k:v for k,v in entry.items() if k!='cache_reference'},flush=True)
                    gate(execution,tag);assert quality['relative_l2']<.03 and entry['staged_official']['relative_l2']<.001
                    uncached=model(x).cpu();gate(errors(actual,uncached),('prefill vs NPU full',tag))
                    entry['status']='passed'
                    for length in range(prefix+1,prefix+steps+1):
                        previous=state;old=[(s.keys.cpu().view(torch.int16),s.values.cpu().view(torch.int16)) for s in states(previous)]
                        one=host[:,length-1:length].contiguous().to('intel_npu')
                        before=npu.stats();out,state=model.decode(one,state);after=npu.stats()
                        for field in ['host_upload_copies','host_download_copies']:assert before[field]==after[field],field
                        actual=out.cpu();full_input=host[:,:length].contiguous()
                        history,local_checks=local_cache_check(model,one,state,previous,history)
                        reference,traces=model.reference_cpu(full_input);traces=[traces] if count==1 else traces
                        baseline_all=official(index,count,full_input);baseline=baseline_all[:,-1:]
                        staged,_=model.reference_cpu(full_input,quantized=False)
                        execution=errors(actual,reference[:,-1:]);quality=errors(actual,baseline)
                        uncached=model(full_input.to('intel_npu')).cpu()[:,-1:]
                        entry=dict(tag=tag,stage='decode',length=length,execution=execution,quality_official=quality,
                            staged_official=errors(staged,baseline_all),staged_official_token=errors(staged[:,-1:],baseline),uncached_npu=errors(actual,uncached),
                            cache_reference=check_state(state,traces,length),cache_local_reference=local_checks,
                            graph_calls=after['npu_calls']-before['npu_calls'],
                            new_sessions=after['session_creations']-before['session_creations'],host_copies=0,status='running')
                        report['checks'].append(entry);print({k:v for k,v in entry.items() if k!='cache_reference'},flush=True)
                        gate(execution,tag);gate(entry['uncached_npu'],('cached vs NPU full',tag))
                        assert quality['relative_l2']<.03 and entry['staged_official']['relative_l2']<.001
                        for cache,(keys,values),new in zip(states(previous),old,states(state)):
                            assert torch.equal(cache.keys.cpu().view(torch.int16),keys) and torch.equal(cache.values.cpu().view(torch.int16),values)
                            assert torch.equal(new.keys.cpu()[:,:,:length-1].view(torch.int16),keys[:,:,:length-1])
                            assert torch.equal(new.values.cpu()[:,:,:length-1].view(torch.int16),values[:,:,:length-1])
                        before=npu.stats();warm,warm_state=model.decode(one,previous);after=npu.stats()
                        assert torch.equal(warm.cpu().view(torch.int16),actual.view(torch.int16))
                        for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                        entry.update(status='passed',snapshot_unchanged=True,warm_new_compiles=0)
                    np.savez(OUT/f'{case}-{tag}.npz',host=host.numpy(),actual=actual.numpy(),reference=reference.numpy(),official=baseline.numpy())
                    return model,host,state
                if case in ['probe','numeric','boundary']:
                    configs={'probe':[(0,1,8,3,[1])],
                        'numeric':[(0,1,1,3,[2,4]),(10,2,7,2,[2,4]),(26,1,32,2,[2,4])],
                        'boundary':[(0,8,63,1,[1,4]),(26,1,63,1,[4]),(0,1,64,0,[2])]}[case]
                    for index,batch,prefix,steps,counts in configs:
                        for count in counts:
                            for mode in ['fp16','w8a16']:
                                model,host,state=check(index,count,batch,prefix,steps,mode)
                                del model,state
                if case=='guards':
                    def reject(name,fn):
                        before=npu.stats()
                        try:fn()
                        except ValueError as error:
                            after=npu.stats()
                            for field in ['npu_calls','session_creations','host_upload_copies','host_download_copies']:
                                assert before[field]==after[field],(name,field)
                            report['checks'].append(dict(name=name,status='rejected_before_execution',error=str(error)))
                        else:raise AssertionError(('missing rejection',name))
                    model=make(4,'w8a16');host=real_input(0,8);x=host.to('intel_npu');one=host[:,:1].contiguous().to('intel_npu')
                    _,state=model.prefill(x)
                    reject('missing cache',lambda:model.decode(one,None));reject('wrong container',lambda:model.decode(one,list(state)))
                    reject('wrong layer count',lambda:model.decode(one,state[:-1]));reject('last cache wrong type',lambda:model.decode(one,(*state[:-1],None)))
                    reject('CPU input',lambda:model.decode(host[:,:1],state));fp32=one.float();reject('FP32 input',lambda:model.decode(fp32,state))
                    reject('multiple new tokens',lambda:model.decode(x,state))
                    other=make(4,'w8a16');_,foreign=other.prefill(x)
                    reject('foreign model',lambda:model.decode(one,foreign));reject('reordered layers',lambda:model.decode(one,tuple(reversed(state))))
                    _,second=model.prefill(x)
                    reject('mixed prefill snapshots',lambda:model.decode(one,(*state[:-1],second[-1])))
                    _,branch_a=model.decode(one,state);_,branch_b=model.decode(one*.5,state)
                    reject('mixed decode branches',lambda:model.decode(one,(*branch_a[:-1],branch_b[-1])))
                    changed=replace(state[-1],length=7)
                    reject('unequal lengths',lambda:model.decode(one,(*state[:-1],changed)))
                    changed=replace(state[-1],length=True)
                    reject('bool length',lambda:model.decode(one,(*state[:-1],changed)))
                    changed=replace(state[-1],length=64)
                    reject('full cache',lambda:model.decode(one,(*state[:-1],changed)))
                    bad_batch=torch.zeros(2,1,576).half().to('intel_npu')
                    reject('different batch',lambda:model.decode(bad_batch,state))
                    changed=replace(state[-1],keys=state[-1].keys.cpu())
                    reject('CPU keys',lambda:model.decode(one,(*state[:-1],changed)))
                    changed=replace(state[-1],values=state[-1].values.float())
                    reject('FP32 values',lambda:model.decode(one,(*state[:-1],changed)))
                    changed=replace(state[-1],keys=state[-1].keys[:,:,:32])
                    reject('partial cache storage',lambda:model.decode(one,(*state[:-1],changed)))
                    model.train();reject('training prefill',lambda:model.prefill(x));reject('training decode',lambda:model.decode(one,state));model.eval()
                    model.layers[-1].train();reject('last child training',lambda:model.decode(one,state));model.eval()
                    grad=one.detach().requires_grad_()
                    with torch.enable_grad():reject('active autograd',lambda:model.decode(grad,state))
                    with torch.inference_mode():
                        actual,new=model.decode(one,state)
                        assert states(new)[0].length==9 and torch.isfinite(actual.cpu()).all()
                        report['checks'].append(dict(name='inference_mode',status='passed'))
                if case=='positions':
                    for count in [1,4]:
                        for mode in ['fp16','w8a16']:
                            model=make(count,mode);host=real_input(0,64);x=host[:,:1].contiguous().to('intel_npu')
                            out,state=model.prefill(x);model.decode(host[:,1:2].contiguous().to('intel_npu'),state)
                            before=npu.stats();calls=[]
                            # New positions reuse one fixed attention graph; input
                            # token uploads are outside the checked call interval.
                            for length in range(2,65):
                                one=host[:,length-1:length].contiguous().to('intel_npu');a=npu.stats()
                                out,state=model.decode(one,state);b=npu.stats()
                                for field in ['host_upload_copies','host_download_copies','session_creations']:assert a[field]==b[field]
                                calls.append(b['npu_calls']-a['npu_calls'])
                                if length in [2,7,8,9,16,17,32,33,63,64]:
                                    reference,_=model.reference_cpu(host[:,:length].contiguous());error=errors(out.cpu(),reference[:,-1:]);gate(error,(count,mode,length))
                                    report['checks'].append(dict(num_layers=count,mode=mode,length=length,status='passed',execution=error))
                            assert len(set(calls))==1
                            report['checks'].append(dict(num_layers=count,mode=mode,status='passed',positions_checked=63,
                                graph_calls_per_decode=calls[0],new_sessions=npu.stats()['session_creations']-before['session_creations'],
                                cache_bytes=sum(s.keys.numel()*2+s.values.numel()*2 for s in states(state))))
                            del model,state,out
                if case=='hybrid':
                    # Explicit pipeline: CPU embedding, one NPU resident segment,
                    # then CPU layers with their own standard DynamicCache.
                    for prompt in ['What is two plus two?','Complete: Paris is the capital of']:
                        text=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
                        inputs=tokenizer(text,return_tensors='pt');ids=inputs.input_ids;assert ids.shape[1]+8<=64
                        baseline=cpu(**inputs,use_cache=True).logits[:,-1].float()
                        expected=cpu.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=True)
                        for count in [2,4]:
                            for mode in ['fp16','w8a16']:
                                model=make(count,mode);state=None;cpu_cache=DynamicCache(config=cpu.config);generated=ids.clone();samples=[]
                                for step in range(8):
                                    token_ids=ids if step==0 else generated[:,-1:]
                                    start=0 if state is None else states(state)[0].length
                                    host=cpu.model.embed_tokens(token_ids);prepared=prepare(host,start)
                                    begin=time.perf_counter()
                                    hidden,state=model.prefill(host.to('intel_npu')) if state is None else model.decode(host.to('intel_npu'),state)
                                    samples.append((time.perf_counter()-begin)*1000)
                                    hidden=hidden.cpu()
                                    for layer in cpu.model.layers[count:]:hidden=layer(hidden,past_key_values=cpu_cache,use_cache=True,**prepared)
                                    logits=cpu.lm_head(cpu.model.norm(hidden))[:,-1].float()
                                    if step==0:
                                        quality=errors(logits,baseline);logp=baseline.log_softmax(-1);logq=logits.log_softmax(-1)
                                        kl=float((logp.exp()*(logp-logq)).sum())
                                        assert quality['relative_l2']<.03 and kl<.02
                                    next_token=logits.argmax(-1,keepdim=True);generated=torch.cat((generated,next_token),dim=1)
                                    if int(next_token[0,0])==cpu.config.eos_token_id:break
                                assert all(cpu_cache.get_seq_length(i)==states(state)[0].length for i in range(count,30))
                                entry=dict(prompt=prompt,mode=mode,npu_layers=count,cpu_layers=30-count,use_cache=True,
                                    logits_relative_l2=quality['relative_l2'],next_token_kl=kl,generated_tokens_equal=bool(torch.equal(generated,expected)),
                                    generated_text=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True),
                                    reference_text=tokenizer.decode(expected[0,ids.shape[1]:],skip_special_tokens=True),
                                    npu_segment_ms=samples,cache_length=states(state)[0].length)
                                report['checks'].append(entry);print(entry,flush=True)
                                assert torch.isfinite(logits).all()
                                del model,state,hidden
                if case in ('profile','profile_before'):
                    # Inclusive stage times and their nested native envelopes;
                    # no device timestamps, so do not call these kernel times.
                    for count,batch,prefix in [(1,1,32),(4,1,32),(4,4,63),(1,8,63)]:
                        host=real_input(0,prefix+1,batch);one=host[:,-1:].contiguous().to('intel_npu')
                        for mode in ['fp16','w8a16']:
                            if case=='profile_before':
                                old=before_module();model=(old.FrozenSmolLayer.from_checkpoint(MODEL,mode=mode) if count==1 else
                                    old.FrozenSmolStack.from_checkpoint(MODEL,num_layers=count,mode=mode))
                                report['baseline_adapter_sha256']=hashlib.sha256((OUT/'before/torch_npu_smollm.py').read_bytes()).hexdigest()
                            else:model=make(count,mode)
                            layer_type=type(model) if count==1 else type(model.layers[0])
                            _,state=model.prefill(host[:,:prefix].contiguous().to('intel_npu'))
                            model.decode(one,state);model.decode(one,state)
                            reference,_=model.reference_cpu(host);gate(errors(model.decode(one,state)[0].cpu(),reference[:,-1:]),('profile',count,batch,mode))
                            before=npu.stats();model.decode(one,state);expected=npu.stats()['npu_calls']-before['npu_calls']
                            plain=[];traced=[];stages=[];native=[];health=[];stage='other'
                            for round_index in range(21):
                                bucket={name:0. for name in ['rmsnorm1','projection','attention','assembly','rmsnorm2','ffn','norm_attention','norm_ffn','norm_layer']}
                                inner={name:0. for name in [*bucket,'other']};calls={name:0 for name in inner};sync=[0.,0]
                                def timer(fn,name):
                                    def wrapped(*args,**kwargs):
                                        nonlocal stage
                                        label=('rmsnorm1' if args[2]=='input_layernorm.weight' else 'rmsnorm2') if name=='norm' else name
                                        if name=='attention' and kwargs.get('kind')=='project':label='projection'
                                        saved=bucket[label];nested=bucket['projection']+bucket['assembly']
                                        previous=stage;stage=label;begin=time.perf_counter()
                                        try:return fn(*args,**kwargs)
                                        finally:
                                            elapsed=(time.perf_counter()-begin)*1000
                                            bucket[label]=saved+elapsed-(bucket['projection']+bucket['assembly']-nested if label=='attention' else 0)
                                            stage=previous
                                    return wrapped
                                def concat(fn):
                                    def wrapped(*args,**kwargs):
                                        # Only attention batch assembly has >1
                                        # sample tensors; norm's one-view is in norm.
                                        if len(args[0])==1:return fn(*args,**kwargs)
                                        return timer(fn,'assembly')(*args,**kwargs)
                                    return wrapped
                                def native_timer(fn):
                                    def wrapped(*args,**kwargs):
                                        begin=time.perf_counter()
                                        try:return fn(*args,**kwargs)
                                        finally:inner[stage]+=(time.perf_counter()-begin)*1000;calls[stage]+=1
                                    return wrapped
                                def sync_timer(fn):
                                    def wrapped(*args,**kwargs):
                                        begin=time.perf_counter()
                                        try:return fn(*args,**kwargs)
                                        finally:sync[0]+=(time.perf_counter()-begin)*1000;sync[1]+=1
                                    return wrapped
                                def run_plain():
                                    begin=time.perf_counter();model.decode(one,state);plain.append((time.perf_counter()-begin)*1000)
                                def run_traced():
                                    with ExitStack() as stack:
                                        if hasattr(layer_type,'_run_fused'):
                                            original_fused=layer_type._run_fused
                                            def fused_timer(*args,**kwargs):return timer(original_fused,'norm_'+kwargs['segment'])(*args,**kwargs)
                                            stack.enter_context(patch.object(layer_type,'_run_fused',fused_timer))
                                        for obj,name,wrapper in [(layer_type,'_norm_device',lambda fn:timer(fn,'norm')),
                                            (layer_type,'_run_cached_attention',lambda fn:timer(fn,'attention')),
                                            (layer_type,'_run_graph',lambda fn:timer(fn,'ffn')),
                                            (npu,'_concat',concat),(npu,'_concat_many',concat),(npu._C,'run',native_timer),(npu._C,'run_many',native_timer),
                                            (npu,'synchronize',sync_timer)]:
                                            stack.enter_context(patch.object(obj,name,wrapper(getattr(obj,name))))
                                        begin=time.perf_counter();model.decode(one,state);traced.append((time.perf_counter()-begin)*1000)
                                before=npu.stats()
                                for fn in ([run_plain,run_traced] if round_index%2==0 else [run_traced,run_plain]):fn()
                                after=npu.stats()
                                for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                                assert sum(calls.values())==expected and after['npu_calls']-before['npu_calls']==expected*2
                                stages.append(bucket);native.append(dict(ms=inner,calls=calls));health.append(dict(ms=sync[0],calls=sync[1]))
                            averages={name:sum(s[name] for s in stages)/21 for name in stages[0]}
                            avg_total=sum(traced)/21;native_averages={name:sum(s['ms'][name] for s in native)/21 for name in native[0]['ms']}
                            entry=dict(mode=mode,num_layers=count,batch=batch,prefix_tokens=prefix,rounds=21,
                                median_plain_ms=statistics.median(plain),median_traced_ms=statistics.median(traced),
                                mean_traced_ms=avg_total,mean_stage_ms=averages,mean_stage_percent={name:value/avg_total*100 for name,value in averages.items()},
                                mean_native_ms=native_averages,mean_native_total_ms=sum(native_averages.values()),
                                mean_outside_run_stage_ms={name:value-native_averages[name] for name,value in averages.items()},
                                mean_other_ms=avg_total-sum(averages.values()),native_calls=native[0]['calls'],
                                mean_health_check_ms=sum(s['ms'] for s in health)/21,health_check_calls=health[0]['calls'],
                                samples_plain_ms=plain,samples_traced_ms=traced,samples_stages=stages,samples_native=native,samples_health=health,
                                warm_host_copies_and_compiles=0)
                            report['timings'].append(entry);print({k:v for k,v in entry.items() if not k.startswith('samples')},flush=True)
                            del model,state
                if case=='performance':
                    old=before_module()
                    for count in [1,2,4]:
                        for batch,prefix in [(1,32),(4,63)]:
                            host=real_input(0,prefix+1,batch);last=host[:,-1:].contiguous();x=host.to('intel_npu');one=last.to('intel_npu')
                            models={mode:make(count,mode) for mode in ['fp16','w8a16']};cache={}
                            for mode in ['fp16','w8a16']:
                                models['previous_'+mode]=(old.FrozenSmolLayer.from_checkpoint(MODEL,mode=mode) if count==1 else
                                    old.FrozenSmolStack.from_checkpoint(MODEL,num_layers=count,mode=mode))
                            for mode,model in models.items():_,cache[mode]=model.prefill(host[:,:prefix].contiguous().to('intel_npu'))
                            prepared=prepare(host);prepared_step=prepare(last,prefix)
                            cpu_cache=DynamicCache(config=cpu.config);official(0,count,host[:,:prefix].contiguous(),cpu_cache)
                            original_layers=list(cpu.model.layers[:count]);position_cache=cpu_cache
                            run={mode+'_decode':(lambda model=model,state=cache[mode]:model.decode(one,state)) for mode,model in models.items()}
                            run.update({mode+'_full_prefix':(lambda model=model:model(x)) for mode,model in models.items() if not mode.startswith('previous_')})
                            def cpu_step():
                                # Repeating the same step must retain the original
                                # prefix. Trim CPU DynamicCache after each call.
                                value=last
                                for layer in original_layers:value=layer(value,past_key_values=position_cache,use_cache=True,**prepared_step)
                                position_cache.crop(prefix)
                                return value
                            def cpu_full():
                                value=host
                                for layer in original_layers:value=layer(value,use_cache=False,**prepared)
                                return value
                            run.update(cpu_fp16_decode=cpu_step,cpu_fp16_full_prefix=cpu_full)
                            calls={}
                            for name,fn in run.items():
                                fn();a=npu.stats();result=fn();b=npu.stats();calls[name]=b['npu_calls']-a['npu_calls']
                                if name.endswith('_decode') and not name.startswith('cpu'):
                                    actual=result[0].cpu();reference,_=models[name.removesuffix('_decode')].reference_cpu(host)
                                    gate(errors(actual,reference[:,-1:]),('benchmark',name))
                                for field in ['host_upload_copies','host_download_copies','session_creations']:assert a[field]==b[field]
                            samples={name:[] for name in run};names=list(run);before=npu.stats()
                            for round_index in range(21):
                                offset=round_index%len(names)
                                for name in names[offset:]+names[:offset]:
                                    begin=time.perf_counter();run[name]();samples[name].append((time.perf_counter()-begin)*1000)
                            after=npu.stats()
                            for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                            entry=dict(shape=list(host.shape),prefix_tokens=prefix,num_layers=count,cpu_threads=8,
                                median_ms={name:statistics.median(values) for name,values in samples.items()},samples_ms=samples,
                                graph_calls=calls,warm_host_copies_and_compiles=0,
                                cache_bytes={mode:sum(s.keys.numel()*2+s.values.numel()*2 for s in states(state)) for mode,state in cache.items()})
                            report['timings'].append(entry);print({k:v for k,v in entry.items() if k!='samples_ms'},flush=True)
                            run.clear();models.clear();cache.clear();del model,fn,result,x,one
                if case=='paired':
                    old=before_module()
                    for count,batch,prefix in [(1,1,1),(1,2,32),(4,4,61),(1,8,61)]:
                        host=real_input(0,prefix+3,batch);x=host[:,:prefix].contiguous().to('intel_npu')
                        for mode in ['fp16','w8a16']:
                            previous=(old.FrozenSmolLayer.from_checkpoint(MODEL,mode=mode) if count==1 else
                                old.FrozenSmolStack.from_checkpoint(MODEL,num_layers=count,mode=mode))
                            current=make(count,mode);a,old_state=previous.prefill(x);b,new_state=current.prefill(x)
                            assert torch.equal(a.cpu().view(torch.int16),b.cpu().view(torch.int16))
                            for step in range(3):
                                one=host[:,prefix+step:prefix+step+1].contiguous().to('intel_npu')
                                before_old=npu.stats();a,old_state=previous.decode(one,old_state);after_old=npu.stats()
                                before=npu.stats();b,new_state=current.decode(one,new_state);after=npu.stats()
                                for field in ['host_upload_copies','host_download_copies']:assert before[field]==after[field]
                                if step:
                                    assert before['session_creations']==after['session_creations']
                                delta=errors(b.cpu(),a.cpu());gate(delta,('previous vs fused',count,batch,prefix,mode,step))
                                bits_equal=bool(torch.equal(a.cpu().view(torch.int16),b.cpu().view(torch.int16)))
                                for old_cache,new_cache in zip(states(old_state),states(new_state)):
                                    assert old_cache.length==new_cache.length==prefix+step+1
                                    assert torch.equal(old_cache._position.cpu().view(torch.int32),new_cache._position.cpu().view(torch.int32))
                                    for name in ['keys','values']:
                                        old_value=getattr(old_cache,name).cpu();new_value=getattr(new_cache,name).cpu()
                                        gate(errors(new_value,old_value),('previous vs fused KV',name))
                                        bits_equal &= bool(torch.equal(old_value.view(torch.int16),new_value.view(torch.int16)))
                                entry=dict(mode=mode,num_layers=count,batch=batch,length=prefix+step+1,status='passed',
                                    previous_new=delta,all_output_cache_bits_equal=bits_equal,
                                    graph_calls=after['npu_calls']-before['npu_calls'],previous_graph_calls=after_old['npu_calls']-before_old['npu_calls'],
                                    warmed=step>0)
                                if step:
                                    assert entry['graph_calls']==count
                                    expected_previous=(1 if OUT.name=='smollm-prefill-fusion' else
                                        2 if OUT.name=='smollm-whole-layer' or OUT.name=='rmsnorm-b1-fusion' and batch>=2 else
                                        4 if OUT.name in ['rmsnorm-small','rmsnorm-fusion','rmsnorm-b1-fusion'] or batch==1 else batch+4)
                                    assert entry['previous_graph_calls']==count*expected_previous
                                report['checks'].append(entry);print(entry,flush=True)
                            del previous,current,old_state,new_state,a,b
                if case=='regression':
                    old=before_module()
                    for index,batch,length in [(0,1,1),(0,2,17),(26,4,64),(29,8,64)]:
                        host=real_input(index,length,batch);x=host.to('intel_npu')
                        for mode in ['fp16','w8a16']:
                            previous=old.FrozenSmolLayer.from_checkpoint(MODEL,layer_index=index,mode=mode)
                            current=FrozenSmolLayer.from_checkpoint(MODEL,layer_index=index,mode=mode)
                            a=previous(x).cpu();b=current(x).cpu();assert torch.equal(a.view(torch.int16),b.view(torch.int16))
                            report['checks'].append(dict(index=index,shape=list(host.shape),mode=mode,status='passed',old_new_bits_equal=True))
                            del previous,current
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
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=['probe','numeric','boundary','guards','positions','hybrid','performance','regression','paired','profile','profile_before'],default='probe')
    parser.add_argument('--phase',choices=['smollm-kv-batch','rmsnorm-small','rmsnorm-fusion','rmsnorm-b1-fusion','smollm-whole-layer','smollm-prefill-fusion'],default='smollm-kv-batch')
    parser.add_argument('--child',action='store_true');args=parser.parse_args()
    OUT=OUT.parent/args.phase;OUT.mkdir(parents=True,exist_ok=True)
    if args.child:return execute(args.case)
    with (OUT/f'{args.case}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,__file__,'--case',args.case,'--phase',args.phase,'--child'],stdout=log,stderr=log,timeout=600)
    print((OUT/f'{args.case}.log').read_text(encoding='utf-8'),flush=True)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())

"""All-30-layer calibration and independent frozen-decoder quality checks.

Reuses the public INT8 calibrator and historical prompts/quality thresholds.
CPU embedding/final norm/head are explicit; every decoder layer runs on NPU.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import w8_calibration as calibration
from w8_calibration import CALIBRATION, input_name, unpack
from w8_quality_probe import ROOT, MODEL, sha

OUT=ROOT/'npunlock-probe/research/results/w8-full-calibration'
SOURCES=calibration.SOURCES+['npunlock/python/npunlock/npu4000.py',
    'npunlock/tests/fixtures/npu4000/kv_select.c','npunlock/tests/fixtures/npu4000/kv_select/kv-select-f16.elf',
    'npunlock/tests/fixtures/npu4000/kv_select/manifest.json',
    'npunlock-probe/research/w8_full_calibration.py','npunlock-probe/research/w8_calibration.py',
    'npunlock-probe/research/w8_calibration_probe.py','npunlock-probe/research/w8_calibration_decode.py',
    'torch_npu_smollm_stack_demo.py']


def write(name,report):
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')


def chat(tokenizer,system,prompt):
    text=tokenizer.apply_chat_template([dict(role='system',content=system),dict(role='user',content=prompt)],
        tokenize=False,add_generation_prompt=True)
    ids=tokenizer(text,return_tensors='pt').input_ids
    assert 1<ids.shape[1]<=64
    return ids


def prepare():
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from torch_npu_lowbit import quantize_weight,calibrate_int8
    start=time.perf_counter()
    report=dict(status='running',method='gptq',num_layers=30,calibration_texts=CALIBRATION,
        sources={name:sha(ROOT/name) for name in SOURCES},checkpoint_sha256=sha(MODEL/'model.safetensors'),
        calibrator='torch_npu_lowbit.calibrate_int8',
        gptq_parameters=dict(damping=.01,block_size=128,activation_order=True,solver_dtype='float64',row_scale='original_fp16'),
        inputs=[],weights=[])
    cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    assert len(cpu.model.layers)==30
    captured={f'{i}/{key}':[] for i in range(30) for key in ('norm1','attention','norm2','gated')};hooks=[]
    for i,layer in enumerate(cpu.model.layers):
        for name in ('self_attn.q_proj','self_attn.o_proj','mlp.gate_proj','mlp.down_proj'):
            key=f'{i}/{input_name(name+".weight")}'
            def collect(module,args,key=key):captured[key].append(args[0].detach().reshape(-1,args[0].shape[-1]).clone())
            hooks.append(layer.get_submodule(name).register_forward_pre_hook(collect))
    try:
        for system,prompt in CALIBRATION:
            ids=chat(tokenizer,system,prompt);host=cpu.model.embed_tokens(ids);length=ids.shape[1];positions=torch.arange(length)[None]
            kwargs=dict(position_ids=positions,attention_mask=torch.triu(torch.full((length,length),-float('inf'),dtype=torch.float16),diagonal=1)[None,None],
                position_embeddings=cpu.model.rotary_emb(host,positions),use_cache=False)
            for layer in cpu.model.layers:host=layer(host,**kwargs)
            report['inputs'].append(dict(system=system,prompt=prompt,input_tokens=length,token_ids=ids.tolist()))
    finally:
        for hook in hooks:hook.remove()
    inputs={key:torch.cat(value) for key,value in captured.items()};del captured
    assert len({value.shape[0] for value in inputs.values()})==1
    path=OUT/'calibration-inputs.npz';np.savez(path,**{key:value.numpy() for key,value in inputs.items()});report['inputs_sha256']=sha(path)
    arrays={}
    for i,layer in enumerate(cpu.model.layers):
        for name,parameter in layer.named_parameters():
            if parameter.ndim!=2:continue
            weight=parameter.detach().half().contiguous();x=inputs[f'{i}/{input_name(name)}'].double()
            original=quantize_weight(weight,format='int8');q=calibrate_int8(weight,inputs[f'{i}/{input_name(name)}'])
            target=x@weight.double().t()
            def reconstruction(packed):
                c,s=unpack(packed);prediction=(x@torch.from_numpy(c.astype(np.float64)).t())*torch.from_numpy(s.astype(np.float64))[None]
                return (prediction-target).square().sum(0)
            before=reconstruction(original);after=reconstruction(q);assert torch.all(after<=before+1e-18)
            c0,s0=unpack(original);c,s=unpack(q);assert np.array_equal(s,s0) and c.min()>=-127
            changed=np.any(c0!=c,axis=1);key=f'{i}/{name}'
            arrays[key+'/codes']=c;arrays[key+'/scales']=s
            report['weights'].append(dict(layer=i,name=name,rows=len(s),activation_rows=len(x),
                original_output_squared_error=float(before.sum()),calibrated_output_squared_error=float(after.sum()),
                changed_rows=int(changed.sum()),original_codes_sha256=__import__('hashlib').sha256(original.data).hexdigest(),
                calibrated_codes_sha256=__import__('hashlib').sha256(q.data).hexdigest()))
        print('calibrated layer',i+1,'/30',flush=True)
    path=OUT/'weights.npz';np.savez(path,**arrays);report['weights_sha256']=sha(path)
    assert len(report['weights'])==210 and len(arrays)==420
    previous=ROOT/'npunlock-probe/research/results/kv-select-integration/calibrated-current'
    with np.load(previous/'weights.npz',allow_pickle=False) as old:
        for key in old.files:np.testing.assert_array_equal(arrays[key].view(np.uint8),old[key].view(np.uint8))
        report['first_four_layers_bit_equal_arrays']=len(old.files)
    report.update(status='passed',elapsed_seconds=time.perf_counter()-start);write('prepare',report)
    print('prepared',report['elapsed_seconds'],'seconds',flush=True)


def weights():
    calibration.OUT=OUT
    result=calibration.load_weights() # Includes strict current-source and artifact fingerprint checks.
    assert set(result)==set(range(30)) and all(len(layer)==7 for layer in result.values())
    metadata=json.loads((OUT/'prepare.json').read_text())
    assert sha(MODEL/'model.safetensors')==metadata['checkpoint_sha256']
    return result


def evaluate(case):
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    from torch_npu_smollm_stack_demo import errors
    from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
    from w8_calibration_probe import HOLDOUT,FRESH,metric
    from w8_calibration_decode import PROMPTS
    report=dict(status='running',case=case,sources={name:sha(ROOT/name) for name in SOURCES},
        weights_sha256=sha(OUT/'weights.npz'),npu_decoder_layers=30,embedding_and_final_norm_and_lm_head='explicit_cpu',
        criteria=dict(logits_relative_l2=.03,next_token_kl=.02,local_relative_l2=.003,local_max_scaled_error=.01),positions=[],warm_checks=[])
    saved={};start=time.perf_counter()
    def work():
        packed=weights()
        cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        stacks={label:FrozenSmolStack.from_checkpoint(MODEL,num_layers=30,mode='fp16' if label=='fp16' else 'w8a16',
            packed_weights=packed if label=='calibrated' else None) for label in ('fp16','original','calibrated')}
        def quality(tag,outputs,baseline,**details):
            entry=dict(tag=tag,**details,variants={});report['positions'].append(entry)
            saved[tag+'/baseline-logits']=baseline.numpy().copy()
            for label,(hidden,state) in outputs.items():
                actual=cpu.lm_head(cpu.model.norm(hidden.cpu()))[:,-1].float()
                assert torch.isfinite(actual).all()
                saved[f'{tag}/{label}-logits']=actual.numpy().copy();entry['variants'][label]=metric(actual,baseline)
            print(tag,{label:round(row['relative_l2'],6) for label,row in entry['variants'].items()},flush=True)
        def warm(tag,x,previous,outputs):
            for label,stack in stacks.items():
                before=npu.stats();again=stack.decode(x,previous[label]);after=npu.stats()
                assert after['npu_calls']-before['npu_calls']==30
                assert all(after[key]==before[key] for key in ('session_creations','host_upload_copies','host_download_copies'))
                first=[outputs[label][0]]+[getattr(c,field) for c in outputs[label][1] for field in ('keys','values','_position')]
                second=[again[0]]+[getattr(c,field) for c in again[1] for field in ('keys','values','_position')]
                for a,b in zip(first,second):assert torch.equal(a.cpu().view(torch.uint8),b.cpu().view(torch.uint8))
            report['warm_checks'].append(dict(tag=tag,graph_calls_per_variant=30,bit_equal_arrays_per_variant=91,host_copies_and_compiles=0))
        if case=='holdout':
            assert set((system,prompt) for _,system,prompt in HOLDOUT+FRESH).isdisjoint(CALIBRATION)
            report.update(calibration_prompts_disjoint=True,input_policy='one-token prefill then forced prompt tokens; direct prefill controls for bird/math')
            for tag,system,prompt in HOLDOUT+FRESH:
                ids=chat(tokenizer,system,prompt);saved[tag+'/ids']=ids.numpy().copy();host=cpu.model.embed_tokens(ids)
                baseline=cpu(ids,use_cache=False).logits[:,-1].float();states=None
                for position in range(ids.shape[1]):
                    x=host[:,position:position+1].contiguous().to('intel_npu');previous=states
                    outputs={label:stack.prefill(x) if states is None else stack.decode(x,states[label]) for label,stack in stacks.items()}
                    states={label:out[1] for label,out in outputs.items()}
                assert all(c.length==ids.shape[1] for state in states.values() for c in state)
                quality(tag,outputs,baseline,system=system,prompt=prompt,input_tokens=ids.shape[1],execution='incremental_prompt')
                if tag in ('bird','math'):
                    warm(tag,x,previous,outputs)
                    full=host.to('intel_npu');direct={label:stack.prefill(full) for label,stack in stacks.items()}
                    quality(tag+'-direct',direct,baseline,input_tokens=ids.shape[1],execution='direct_prefill')
                    report.setdefault('incremental_vs_direct',[]).append(dict(tag=tag,variants={label:errors(out[0].cpu(),direct[label][0].cpu()) for label,out in outputs.items()}))
                    del full,direct
        elif case=='decode':
            cases=[(tag,chat(tokenizer,'Tell stories.',prompt),32) for tag,prompt in PROMPTS]
            ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
            ids=ids.repeat(1,(32+ids.shape[1])//ids.shape[1])[:,:32];cases.append(('repeat32',ids,33))
            report['input_policy']='natural full prefill, then identical CPU greedy continuation; no EOS stopping'
            for tag,ids,count in cases:
                assert ids.shape[1]+count-1<=64;cache=DynamicCache(config=cpu.config);states=None;tokens=ids
                generated=ids.clone();seed_snapshots=None
                for step in range(count):
                    baseline=cpu(tokens,past_key_values=cache,use_cache=True).logits[:,-1].float()
                    x=cpu.model.embed_tokens(tokens).to('intel_npu');previous=states
                    outputs={label:stack.prefill(x) if states is None else stack.decode(x,states[label]) for label,stack in stacks.items()}
                    states={label:out[1] for label,out in outputs.items()};length=ids.shape[1]+step
                    assert all(c.length==length for state in states.values() for c in state)
                    quality(f'{tag}/{step}',outputs,baseline,input_tokens=ids.shape[1],cache_length=length,execution='prefill' if step==0 else 'decode')
                    if step==0:
                        seeds=states;seed_snapshots={label:[v.cpu().clone() for c in state for v in (c.keys,c.values,c._position)] for label,state in seeds.items()}
                    if step==count-1:warm(tag,x,previous,outputs)
                    tokens=baseline.argmax(-1,keepdim=True);generated=torch.cat((generated,tokens),dim=1)
                for label,state in seeds.items():
                    for v,old in zip([v for c in state for v in (c.keys,c.values,c._position)],seed_snapshots[label]):assert torch.equal(v.cpu().view(torch.uint8),old.view(torch.uint8))
                report.setdefault('immutable_seed_checks',[]).append(dict(tag=tag,arrays=270))
                saved[tag+'/tokens']=generated.numpy().copy()
        else: # Per-layer diagnosis separates compressed-weight error from kernel execution.
            holdout=json.loads((OUT/'holdout.json').read_text())
            worst=max((row for row in holdout['positions'] if row['execution']=='incremental_prompt'),key=lambda row:row['variants']['calibrated']['relative_l2'])
            tag=worst['tag'];ids=chat(tokenizer,worst['system'],worst['prompt']);host=cpu.model.embed_tokens(ids)
            baseline=cpu(ids,use_cache=False).logits[:,-1].float();saved[tag+'/ids']=ids.numpy().copy();report['diagnostic_prompt']=worst
            outputs={};report['local_layers']=[]
            for label,stack in stacks.items():
                device=host.to('intel_npu');staged=host
                for i,layer in enumerate(stack.layers):
                    reference,_=layer.reference_cpu(device.cpu());staged,_=layer.reference_cpu(staged)
                    device,state=layer.prefill(device);actual=device.cpu();error=errors(actual,reference)
                    assert error['relative_l2']<.003 and error['max_scaled_error']<.01,(label,i,error)
                    report['local_layers'].append(dict(variant=label,layer=i,**error))
                    saved[f'{tag}/{label}/layer-{i}-actual']=actual.numpy().copy();saved[f'{tag}/{label}/layer-{i}-reference']=reference.numpy().copy()
                logits=cpu.lm_head(cpu.model.norm(staged))[:,-1].float();saved[f'{tag}/{label}-staged-logits']=logits.numpy().copy()
                report.setdefault('cpu_staged_quality',{})[label]=metric(logits,baseline);outputs[label]=(device,None)
            quality(tag,outputs,baseline,execution='direct_prefill_diagnosis')
        report['decode_graph_evidence']=[dict(variant=label,layer=i,evidence=[value for key,value in layer._evidence.items() if key[-1]=='decode'])
            for label,stack in stacks.items() for i,layer in enumerate(stack.layers)] if case!='diagnostic' else []
    try:
        work();gc.collect();npu.shutdown();stats=npu.stats();report['final_stats']=stats
        assert stats['closed'] and not stats['context_failed']
        assert all(stats[key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
        path=OUT/(case+'-outputs.npz');np.savez(path,**saved)
        report.update(status='passed',outputs_sha256=sha(path),output_array_count=len(saved),elapsed_seconds=time.perf_counter()-start,
            candidate_all_quality_gates_passed=all(row['variants']['calibrated']['quality_gate_passed'] for row in report['positions']))
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    write(case,report);return int(report['status']!='passed')


def audit():
    from w8_calibration_probe import metric
    from torch_npu_smollm_stack_demo import errors
    packed=weights();metadata=json.loads((OUT/'prepare.json').read_text());assert sha(OUT/'calibration-inputs.npz')==metadata['inputs_sha256']
    assert len(packed)==30 and len(metadata['weights'])==210 and metadata['first_four_layers_bit_equal_arrays']==56
    assert all(row['calibrated_output_squared_error']<=row['original_output_squared_error'] for row in metadata['weights'])
    summary=dict(status='passed',sources=metadata['sources'],checkpoint_sha256=metadata['checkpoint_sha256'],weights_sha256=metadata['weights_sha256'],
        calibrated_layers=30,calibrated_matrices=210,first_four_layers_bit_equal_arrays=56,runs={},quality_metric_rechecks=0,local_execution_rechecks=0,npu_calls=0)
    for case in ('holdout','decode','diagnostic'):
        report=json.loads((OUT/(case+'.json')).read_text());assert report['status']=='passed' and report['sources']==metadata['sources']
        assert report['weights_sha256']==metadata['weights_sha256'] and report['npu_decoder_layers']==30
        stats=report['final_stats'];assert stats['closed'] and not stats['context_failed']
        assert all(stats[key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
        summary['npu_calls']+=stats['npu_calls'];path=OUT/(case+'-outputs.npz');assert sha(path)==report['outputs_sha256']
        with np.load(path,allow_pickle=False) as arrays:
            assert len(arrays.files)==report['output_array_count']
            for row in report['positions']:
                baseline=torch.from_numpy(arrays[row['tag']+'/baseline-logits'])
                for label,result in row['variants'].items():
                    actual=torch.from_numpy(arrays[f'{row["tag"]}/{label}-logits']);recomputed=metric(actual,baseline)
                    for key,value in recomputed.items():
                        assert value==result[key] if isinstance(value,(bool,int)) else np.isclose(value,result[key],rtol=1e-6,atol=1e-9),(case,row['tag'],label,key)
                    summary['quality_metric_rechecks']+=1
            for row in report.get('local_layers',[]):
                key=f'{report["diagnostic_prompt"]["tag"]}/{row["variant"]}/layer-{row["layer"]}'
                recomputed=errors(torch.from_numpy(arrays[key+'-actual']),torch.from_numpy(arrays[key+'-reference']))
                for name,value in recomputed.items():assert np.isclose(value,row[name],rtol=1e-6,atol=1e-9)
                assert recomputed['relative_l2']<.003 and recomputed['max_scaled_error']<.01;summary['local_execution_rechecks']+=1
            for label,result in report.get('cpu_staged_quality',{}).items():
                tag=report['diagnostic_prompt']['tag'];actual=torch.from_numpy(arrays[f'{tag}/{label}-staged-logits'])
                recomputed=metric(actual,torch.from_numpy(arrays[tag+'/baseline-logits']))
                for key,value in recomputed.items():assert np.isclose(value,result[key],rtol=1e-6,atol=1e-9)
                summary['quality_metric_rechecks']+=1
        for warm in report['warm_checks']:assert warm['graph_calls_per_variant']==30 and warm['host_copies_and_compiles']==0 and warm['bit_equal_arrays_per_variant']==91
        summary['runs'][case]=dict(positions=len(report['positions']),npu_calls=stats['npu_calls'],variants={label:dict(
            max_relative_l2=max(row['variants'][label]['relative_l2'] for row in report['positions']),
            max_kl=max(row['variants'][label]['next_token_kl'] for row in report['positions']),
            quality_passed=sum(row['variants'][label]['quality_gate_passed'] for row in report['positions']),
            next_token_equal=sum(row['variants'][label]['next_token_equal'] for row in report['positions'])) for label in ('fp16','original','calibrated')})
    summary['candidate_recommended']=all(summary['runs'][case]['variants']['calibrated']['quality_passed']==summary['runs'][case]['positions'] for case in ('holdout','decode'))
    write('audit',summary);print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('prepare','holdout','decode','diagnostic','audit'),required=True);args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';torch.set_num_threads(8);OUT.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        if args.case=='prepare':prepare();return 0
        if args.case=='audit':audit();return 0
        return evaluate(args.case)


if __name__=='__main__':raise SystemExit(main())

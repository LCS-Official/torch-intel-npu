"""Separate frozen weight loss, staged arithmetic, and NPU execution errors."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/w8-quality'
MODEL=ROOT/'models/SmolLM2-135M-Instruct'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quantizer',choices=('baseline','refined'),default='baseline')
    args=parser.parse_args()
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from torch_npu_smollm import FrozenSmolStack
    if args.quantizer=='refined':
        import torch_npu_smollm
        from w8_scale_candidate import refine_weight
        torch_npu_smollm.quantize_weight=refine_weight
    from torch_npu_smollm_stack_demo import errors
    torch.set_num_threads(8);OUT.mkdir(parents=True,exist_ok=True)
    sources=['torch_npu_smollm.py','torch_npu_lowbit.py','torch_npu_transformer.py','torch_intel_npu.py',
        'npunlock-probe/torch-device/_torch_intel_npu.pyd','npunlock-probe/session-runtime/npunlock_session.dll',
        'npunlock/tests/fixtures/npu4000/rmsnorm.c','npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf']
    report=dict(status='running',quantizer=args.quantizer,sources={name:sha(ROOT/name) for name in sources},source_sha256=sha(Path(__file__)),
        quality_gate_relative_l2=.03,execution_gate_relative_l2=.003,execution_gate_max_scaled=.01,prompts=[],weights=[])
    arrays={}
    try:
        with torch.no_grad():
            cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
            tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
            stacks={mode:FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,mode=mode) for mode in ('fp16','w8a16')}
            original={(i,name):cpu.model.layers[i].get_parameter(name).clone() for i,layer in enumerate(stacks['w8a16'].layers) for name in layer._packed}
            for (i,name),weight in original.items():
                packed=stacks['w8a16'].layers[i]._packed[name]
                restored=torch.from_numpy(packed.dequantize().copy())
                report['weights'].append(dict(layer=i,name=name,group_size=packed.group_size,**errors(restored,weight)))
            prompts=[('bird','Tell stories.','Tell a story about a bird.'),
                ('math','You are a helpful assistant.','What is two plus two?'),
                ('paris','You are a helpful assistant.','Complete: Paris is the capital of')]
            for tag,system,prompt in prompts:
                text=tokenizer.apply_chat_template([dict(role='system',content=system),dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
                ids=tokenizer(text,return_tensors='pt').input_ids;assert ids.shape[1]<=64
                baseline=cpu(ids,use_cache=False).logits[:,-1].float()
                host=cpu.model.embed_tokens(ids);length=host.shape[1];positions=torch.arange(length)[None]
                prepared=dict(position_ids=positions,attention_mask=torch.triu(torch.full((length,length),-float('inf'),dtype=torch.float16),diagonal=1)[None,None],
                    position_embeddings=cpu.model.rotary_emb(host,positions),use_cache=False)
                def tail(value):
                    for module in cpu.model.layers[4:]:value=module(value,**prepared)
                    return cpu.lm_head(cpu.model.norm(value))[:,-1].float()
                def metric(logits):
                    logp=baseline.log_softmax(-1);logq=logits.log_softmax(-1)
                    error=errors(logits,baseline)
                    return dict(**error,next_token_kl=float((logp.exp()*(logp-logq)).sum()),
                        next_token_equal=bool(logits.argmax(-1)==baseline.argmax(-1)),
                        next_token=tokenizer.decode(logits.argmax(-1)),quality_gate_passed=error['relative_l2']<.03)
                entry=dict(tag=tag,prompt=prompt,system=system,input_tokens=length,variants={},local_layers=[])
                report['prompts'].append(entry);arrays[f'{tag}/ids']=ids.numpy().copy();arrays[f'{tag}/baseline']=baseline.numpy().copy()
                official=host
                for i,module in enumerate(cpu.model.layers[:4]):
                    official=module(official,**prepared);arrays[f'{tag}/official-layer-{i}']=official.numpy().copy()
                manual=tail(official)
                assert torch.equal(manual,baseline),('manual CPU bridge differs',tag,errors(manual,baseline))
                entry['manual_cpu_bridge_bit_equal']=True
                # Official Torch layers with exactly the stored quantized weights.
                try:
                    for (i,name),weight in original.items():
                        q=stacks['w8a16'].layers[i]._packed[name]
                        cpu.model.layers[i].get_parameter(name).copy_(torch.from_numpy(q.dequantize().copy()))
                    quantized=host
                    for i,module in enumerate(cpu.model.layers[:4]):
                        quantized=module(quantized,**prepared);arrays[f'{tag}/quantized-official-layer-{i}']=quantized.numpy().copy()
                    quantized_logits=tail(quantized)
                finally:
                    for (i,name),weight in original.items():cpu.model.layers[i].get_parameter(name).copy_(weight)
                entry['variants']['cpu_official_quantized']=metric(quantized_logits)
                arrays[f'{tag}/cpu-official-quantized-logits']=quantized_logits.numpy().copy()
                for mode,stack in stacks.items():
                    reference=host;device=host.to('intel_npu')
                    for i,layer in enumerate(stack.layers):
                        previous=device.cpu();local_ref,trace=layer.reference_cpu(previous)
                        device,cache=layer.prefill(device);actual=device.cpu()
                        local=errors(actual,local_ref)
                        assert local['relative_l2']<.003 and local['max_scaled_error']<.01,(tag,mode,i,local)
                        keys=cache.keys.cpu()[:,:,:length];values=cache.values.cpu()[:,:,:length]
                        entry['local_layers'].append(dict(mode=mode,layer=i,output=local,keys=errors(keys,trace['keys']),values=errors(values,trace['values'])))
                        reference,_=layer.reference_cpu(reference)
                        arrays[f'{tag}/{mode}/npu-layer-{i}']=actual.numpy().copy()
                        arrays[f'{tag}/{mode}/cpu-staged-layer-{i}']=reference.numpy().copy()
                        arrays[f'{tag}/{mode}/local-reference-{i}']=local_ref.numpy().copy()
                        del cache,keys,values,trace,local_ref
                    actual_logits=tail(actual);reference_logits=tail(reference)
                    entry['variants'][f'npu_{mode}']=metric(actual_logits)
                    entry['variants'][f'cpu_staged_{mode}']=metric(reference_logits)
                    entry['variants'][f'npu_{mode}']['relative_l2_staged_logits']=errors(actual_logits,reference_logits)['relative_l2']
                    if mode=='w8a16':
                        entry['variants'][f'npu_{mode}']['relative_l2_quantized_official_logits']=errors(actual_logits,quantized_logits)['relative_l2']
                    arrays[f'{tag}/{mode}/npu-logits']=actual_logits.numpy().copy()
                    arrays[f'{tag}/{mode}/cpu-staged-logits']=reference_logits.numpy().copy()
                    del device,layer
                print(tag,{k:round(v['relative_l2'],7) for k,v in entry['variants'].items()},flush=True)
            del stack,stacks,original,cpu;gc.collect();npu.shutdown();report['final_stats']=npu.stats()
            assert not report['final_stats']['context_failed'] and report['final_stats']['live_buffers']==report['final_stats']['cached_buffer_bytes']==0
            path=OUT/f'{args.quantizer}-outputs.npz';np.savez(path,**arrays)
            report.update(status='passed',outputs_sha256=sha(path),output_array_count=len(arrays))
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    if args.quantizer=='refined':report['candidate_source_sha256']=sha(Path(__file__).with_name('w8_scale_candidate.py'))
    (OUT/f'{args.quantizer}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

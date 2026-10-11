"""Offline W8 calibration using the public calibrate_int8 API by default.

The unpromoted scale-only option retains the earlier output-reconstruction
experiment. Both methods prepare weights on CPU without runtime fallback.
GPTQ source/license: results/w8-calibration/references/sources.json.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch_npu_lowbit import PackedWeight,quantize_weight,calibrate_int8
from w8_scale_candidate import refine_weight
from w8_quality_probe import ROOT,MODEL,sha

OUT=ROOT/'npunlock-probe/research/results/w8-calibration'
CALIBRATION=[
    ('You are a helpful assistant.','Explain how a thermometer measures temperature.'),
    ('You are a helpful assistant.','Give three ways to keep a kitchen organized.'),
    ('You are a helpful assistant.','Why does salt dissolve in water?'),
    ('You are a helpful assistant.','Calculate the total cost of three notebooks at five dollars each.'),
    ('You are a helpful assistant.','Describe a safe way to cross a busy street.'),
    ('You are a helpful assistant.','Explain the difference between a mountain and a hill.'),
    ('You are a helpful assistant.','Translate good evening into Spanish.'),
    ('You are a helpful assistant.','List the steps for planting a seed in a pot.'),
    ('Tell stories.','Write a short story about a baker who loses a recipe.'),
    ('Tell stories.','Tell a story about an old clock in a quiet shop.'),
    ('Tell stories.','Write a story about a child finding a painted stone.'),
    ('Tell stories.','Describe a sailor arriving at a small island after a long journey.'),
]
SOURCES=['torch_npu_lowbit.py','torch_npu_smollm.py','torch_npu_transformer.py','torch_intel_npu.py',
    'npunlock-probe/torch-device/_torch_intel_npu.pyd','npunlock-probe/session-runtime/npunlock_session.dll',
    'npunlock/tests/fixtures/npu4000/rmsnorm.c','npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf']


def input_name(name):
    if name.startswith('self_attn.o_'):return 'attention'
    if name.startswith('self_attn.'):return 'norm1'
    return 'gated' if name.startswith('mlp.down_') else 'norm2'


def unpack(q):
    return (np.frombuffer(q.data,dtype=np.int8).reshape(q.shape).copy(),
        np.frombuffer(q.scales,dtype=np.float16).copy())


def fit_weight(weight,inputs):
    original=quantize_weight(weight,format='int8')
    if (not isinstance(inputs,torch.Tensor) or inputs.device.type!='cpu' or inputs.dtype not in (torch.float16,torch.float32)
        or inputs.ndim!=2 or inputs.shape[1]!=weight.shape[1] or not 1<=inputs.shape[0]<=4096 or not torch.isfinite(inputs).all()):
        raise ValueError('Calibration requires 1..4096 finite CPU FP16/FP32 activation rows with matching K')
    values=weight.detach().double().numpy();activation=inputs.detach().double();importance=activation.square().mean(0).numpy()
    codes,scales=unpack(original)
    def diagonal_loss(c,s):return (np.square(values-c*s.astype(np.float64)[:,None])*importance).sum(-1)
    best=diagonal_loss(codes,scales);proposal_codes=codes.copy();proposal_scales=scales.copy()
    # ponytail: one bounded grid and four refinement iterations. Wider search
    # needs calibration evidence; held-out prompts never choose these settings.
    for factor in (.9,.95,.98,1.,1.02,1.05):
        s=np.clip(scales.astype(np.float64)*factor,2**-24,np.finfo(np.float16).max).astype(np.float16)
        for _ in range(4):
            c=np.clip(np.rint(values/s.astype(np.float64)[:,None]),-127,127).astype(np.int8)
            loss=diagonal_loss(c,s);improved=loss<best
            proposal_codes[improved]=c[improved];proposal_scales[improved]=s[improved];best[improved]=loss[improved]
            q=c.astype(np.float64);denominator=(q*q*importance).sum(-1)
            s=np.divide((values*q*importance).sum(-1),denominator,out=s.astype(np.float64),where=denominator>0)
            s=np.clip(s,2**-24,np.finfo(np.float16).max).astype(np.float16)
    weighted=PackedWeight(original.shape,'int8',original.group_size,proposal_codes.tobytes(),proposal_scales.tobytes())
    target=activation@torch.from_numpy(values).t()
    def reconstruction(q):
        c,s=unpack(q)
        prediction=(activation@torch.from_numpy(c.astype(np.float64)).t())*torch.from_numpy(s.astype(np.float64))[None]
        return (prediction-target).square().sum(0).numpy()
    before=reconstruction(original);best=before.copy();chosen=np.zeros(len(scales),dtype=np.int8)
    for index,candidate in enumerate((refine_weight(weight),weighted),1):
        loss=reconstruction(candidate);improved=loss<best
        c,s=unpack(candidate);codes[improved]=c[improved];scales[improved]=s[improved];best[improved]=loss[improved];chosen[improved]=index
    result=PackedWeight(original.shape,'int8',original.group_size,codes.tobytes(),scales.tobytes())
    after=reconstruction(result);assert np.all(after<=before+1e-18)
    return result,dict(rows=len(scales),activation_rows=len(inputs),original_output_squared_error=float(before.sum()),
        calibrated_output_squared_error=float(after.sum()),row_selection_counts={name:int((chosen==i).sum())
            for i,name in enumerate(('original','data_free','activation_weighted'))},
        original_codes_sha256=__import__('hashlib').sha256(original.data).hexdigest(),
        calibrated_codes_sha256=__import__('hashlib').sha256(result.data).hexdigest())


def load_weights():
    """Load this experiment's manifest-checked map for the public constructor."""
    metadata=json.loads((OUT/'prepare.json').read_text(encoding='utf-8'))
    if metadata['status']!='passed' or sha(OUT/'weights.npz')!=metadata['weights_sha256']:
        raise ValueError('Calibrated weights do not match their manifest')
    for name,digest in metadata['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Calibration profile changed: '+name)
    weights={}
    with np.load(OUT/'weights.npz',allow_pickle=False) as file:
        for row in metadata['weights']:
            i,name=row['layer'],row['name'];key=f'{i}/{name}';codes=file[key+'/codes'];scales=file[key+'/scales']
            if codes.dtype!=np.int8 or scales.dtype!=np.float16 or codes.ndim!=2:
                raise ValueError('Calibration artifact dtype/shape differs')
            q=PackedWeight(tuple(codes.shape),'int8',codes.shape[1],codes.tobytes(),scales.tobytes())
            weights.setdefault(i,{})[name]=q
    return weights


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=('scale-only','gptq'),default='gptq');args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from torch_npu_smollm import FrozenSmolStack
    torch.set_num_threads(8);OUT.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',method=args.method,calibration_texts=CALIBRATION,source_sha256=sha(Path(__file__)),
        sources={name:sha(ROOT/name) for name in SOURCES},weights=[],inputs=[])
    start=time.perf_counter()
    with torch.no_grad():
        cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        captured={f'{i}/{key}':[] for i in range(4) for key in ('norm1','attention','norm2','gated')};hooks=[]
        for i,layer in enumerate(cpu.model.layers[:4]):
            for name in ('self_attn.q_proj','self_attn.o_proj','mlp.gate_proj','mlp.down_proj'):
                key=f'{i}/{input_name(name+".weight")}'
                def collect(module,args,key=key):captured[key].append(args[0].detach().reshape(-1,args[0].shape[-1]).clone())
                hooks.append(layer.get_submodule(name).register_forward_pre_hook(collect))
        try:
            for system,prompt in CALIBRATION:
                text=tokenizer.apply_chat_template([dict(role='system',content=system),dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
                ids=tokenizer(text,return_tensors='pt').input_ids;assert ids.shape[1]<=64
                host=cpu.model.embed_tokens(ids);length=host.shape[1];positions=torch.arange(length)[None]
                prepared=dict(position_ids=positions,attention_mask=torch.triu(torch.full((length,length),-float('inf'),dtype=torch.float16),diagonal=1)[None,None],
                    position_embeddings=cpu.model.rotary_emb(host,positions),use_cache=False)
                for layer in cpu.model.layers[:4]:host=layer(host,**prepared)
                report['inputs'].append(dict(system=system,prompt=prompt,input_tokens=length,token_ids=ids.tolist()))
        finally:
            for hook in hooks:hook.remove()
        inputs={key:torch.cat(value) for key,value in captured.items()}
        assert len({value.shape[0] for value in inputs.values()})==1
        path=OUT/'calibration-inputs.npz';np.savez(path,**{key:value.numpy() for key,value in inputs.items()});report['inputs_sha256']=sha(path)
        stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,mode='w8a16');arrays={}
        if args.method=='gptq':
            report.update(calibrator='torch_npu_lowbit.calibrate_int8',
                gptq_parameters=dict(damping=.01,block_size=128,activation_order=True,solver_dtype='float64',row_scale='original_fp16'))
        for i,layer in enumerate(stack.layers):
            for name in layer._packed:
                weight=layer._cpu_weight(name);input_key=f'{i}/{input_name(name)}'
                if args.method=='gptq':
                    original=layer._packed[name];q=calibrate_int8(weight,inputs[input_key]);x=inputs[input_key].double();target=x@weight.double().t()
                    def reconstruction(packed):
                        c,s=unpack(packed);prediction=(x@torch.from_numpy(c.astype(np.float64)).t())*torch.from_numpy(s.astype(np.float64))[None]
                        return (prediction-target).square().sum(0)
                    before=reconstruction(original);after=reconstruction(q);assert torch.all(after<=before+1e-18)
                    c0,_=unpack(original);c,_=unpack(q);changed=np.any(c0!=c,axis=1)
                    row=dict(rows=weight.shape[0],activation_rows=len(x),original_output_squared_error=float(before.sum()),
                        calibrated_output_squared_error=float(after.sum()),row_selection_counts=dict(original=int((~changed).sum()),gptq=int(changed.sum())),
                        original_codes_sha256=__import__('hashlib').sha256(original.data).hexdigest(),
                        calibrated_codes_sha256=__import__('hashlib').sha256(q.data).hexdigest())
                else:q,row=fit_weight(weight,inputs[input_key])
                key=f'{i}/{name}';codes,scales=unpack(q)
                arrays[key+'/codes']=codes;arrays[key+'/scales']=scales
                report['weights'].append(dict(layer=i,name=name,**row))
                print(key,row['calibrated_output_squared_error']/max(row['original_output_squared_error'],1e-30),flush=True)
        path=OUT/'weights.npz';np.savez(path,**arrays);report['weights_sha256']=sha(path)
        del cpu,stack;gc.collect()
    report.update(status='passed',elapsed_seconds=time.perf_counter()-start,checkpoint_sha256=sha(MODEL/'model.safetensors'))
    (OUT/'prepare.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('prepared',report['elapsed_seconds'],flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

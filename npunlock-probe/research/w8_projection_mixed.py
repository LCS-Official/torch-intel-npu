"""Budgeted FP16 projection protection, selected before independent checks.

Research CPU Linear references use exact code*FP16-scale weights in FP64.
The surrounding network is the official FP16 Transformers model. This is
distinct from the NPU staged reference; final acceptance uses real hardware.
No model, dependency, graph cache, activations or raw logits are downloaded/saved.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

import w8_sequential_calibration as common
from torch_npu_lowbit import PackedWeight, calibrate_int8, quantize_weight
from w8_calibration import input_name
from w8_calibration_probe import FRESH, HOLDOUT, metric
from w8_continuation_calibration import PROMPTS

OUT = common.ROOT / 'npunlock-probe/research/results/w8-projection-mixed'
NAMES = tuple(name for stage in common.STAGES for name in stage)
FIT_SOURCES = ('torch_npu_lowbit.py', 'npunlock-probe/research/w8_projection_mixed.py',
               'npunlock-probe/research/w8_sequential_calibration.py',
               'npunlock-probe/research/w8_continuation_calibration.py',
               'npunlock-probe/research/w8_calibration_probe.py',
               'models/SmolLM2-135M-Instruct/config.json',
               'models/SmolLM2-135M-Instruct/tokenizer.json',
               'models/SmolLM2-135M-Instruct/tokenizer_config.json')


def write(name, value):
    (OUT / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def read(name):
    return json.loads((OUT / (name + '.json')).read_text(encoding='utf-8'))


def sources():
    return {name: common.sha(common.ROOT / name) for name in FIT_SOURCES}


def prepare():
    assert 'torch_intel_npu' not in sys.modules
    assert len(PROMPTS) == len(set(PROMPTS)) == 40
    assert set(PROMPTS).isdisjoint((system, prompt) for _, system, prompt in HOLDOUT + FRESH)
    started = time.perf_counter()
    cpu, tokenizer = common.model()
    report = dict(status='running', sources=sources(), checkpoint_sha256=common.sha(common.MODEL / 'model.safetensors'),
                  trajectories=[], matrices=[], criteria=dict(logits_relative_l2=.03, next_token_kl=.02),
                  policy='Fit i%5!=4 (2048 rows), internal selection i%5==4 (290 positions). '
                         'Public GPTQ 1% damping and original FP16 per-row scales unchanged.')
    for i, (system, prompt) in enumerate(PROMPTS):
        ids = common.chat(tokenizer, system, prompt)
        tokens = common.trajectory(cpu, ids, 64)
        report['trajectories'].append(dict(index=i, system=system, prompt=prompt,
                                          prefix=ids.shape[1], token_ids=tokens.tolist()))
        if (i+1) % 8 == 0:
            print('calibration trajectories', i+1, '/40', flush=True)
    ids = torch.cat([torch.tensor(row['token_ids']) for row in report['trajectories'] if row['index'] % 5 != 4])
    hidden = cpu.model.embed_tokens(ids)
    positions = torch.arange(64)[None].expand(8, -1)
    mask = torch.triu(torch.full((64,64), -float('inf'), dtype=torch.float16), diagonal=1)[None,None]
    arrays = {}
    for i, layer in enumerate(cpu.model.layers):
        captured = {key: [] for key in ('norm1','attention','norm2','gated')}
        hooks = []
        for name in ('self_attn.q_proj','self_attn.o_proj','mlp.gate_proj','mlp.down_proj'):
            key = input_name(name+'.weight')
            def capture(module, args, key=key):
                captured[key].append(args[0].detach().reshape(-1,args[0].shape[-1]).clone())
            hooks.append(layer.get_submodule(name).register_forward_pre_hook(capture))
        try:
            hidden = torch.cat([layer(chunk, position_ids=positions, attention_mask=mask,
                                      position_embeddings=cpu.model.rotary_emb(chunk, positions), use_cache=False)
                                for chunk in hidden.split(8)])
        finally:
            for hook in hooks:
                hook.remove()
        observed = {key: torch.cat(values) for key,values in captured.items()}
        for name in NAMES:
            weight = layer.get_parameter(name).detach().half().contiguous()
            x = observed[input_name(name)]
            assert x.shape[0] == 2048
            original = quantize_weight(weight, format='int8')
            q = calibrate_int8(weight, x)
            before, after = common.reconstruct(original,x,weight), common.reconstruct(q,x,weight)
            assert torch.all(after <= before+1e-18) and q.scales == original.scales
            arrays[f'{i}/{name}/codes'] = np.frombuffer(q.data,dtype=np.int8).reshape(q.shape)
            arrays[f'{i}/{name}/scales'] = np.frombuffer(q.scales,dtype=np.float16)
            report['matrices'].append(dict(layer=i,name=name,rtn_sse=float(before.sum()),calibrated_sse=float(after.sum())))
        print('prepared W8 layer',i+1,'/30',flush=True)
    assert len(arrays) == 420 and len(report['matrices']) == 210
    path = OUT/'weights.npz'
    np.savez_compressed(path,**arrays)
    report.update(status='passed',weights_sha256=common.sha(path),elapsed_seconds=time.perf_counter()-started,npu_calls=0)
    write('prepare',report)


def weights():
    metadata = read('prepare')
    assert metadata['sources'] == sources()
    assert metadata['checkpoint_sha256'] == common.sha(common.MODEL/'model.safetensors')
    path = OUT/'weights.npz'
    assert common.sha(path) == metadata['weights_sha256']
    result = {i:{} for i in range(30)}
    with np.load(path,allow_pickle=False) as arrays:
        assert len(arrays.files) == 420
        for key in arrays.files:
            if not key.endswith('/codes'):
                continue
            i,name,_ = key.split('/')
            codes,scales = arrays[key],arrays[key[:-5]+'scales']
            assert codes.dtype == np.int8 and scales.dtype == np.float16
            result[int(i)][name] = PackedWeight(codes.shape,'int8',codes.shape[1],codes.tobytes(),scales.tobytes())
    assert all(set(layer)==set(NAMES) for layer in result.values())
    return result


class ReferenceLinear(torch.nn.Module):
    def __init__(self, original, packed):
        super().__init__()
        self.original = original
        c = np.frombuffer(packed.data,dtype=np.int8).reshape(packed.shape).astype(np.float64)
        s = np.frombuffer(packed.scales,dtype=np.float16).astype(np.float64)
        self.register_buffer('decoded',torch.from_numpy(c*s[:,None]))
        self.protected = False

    def forward(self, value):
        if self.protected:
            return self.original(value)
        return (value.double() @ self.decoded.t()).half()


def install_reference(cpu, packed):
    modules = []
    counts = {}
    for i,layer in enumerate(cpu.model.layers):
        for name in NAMES:
            parent,field = name[:-7].rsplit('.',1)
            original = layer.get_submodule(name[:-7])
            assert original.bias is None
            reference = ReferenceLinear(original,packed[i][name])
            setattr(layer.get_submodule(parent),field,reference)
            modules.append((name,reference))
            counts[name] = counts.get(name,0)+original.weight.numel()
    return modules,counts


def set_policy(modules, names):
    protected = frozenset(names)
    assert protected <= set(NAMES)
    for name,module in modules:
        module.protected = name in protected


def evaluate(cpu, modules, names, cases):
    set_policy(modules,names)
    rows = []
    for tag,offset,ids,baseline in cases:
        logits = cpu(ids,use_cache=False).logits[:,offset:].float()
        assert logits.shape == baseline.shape and torch.isfinite(logits).all()
        for j in range(logits.shape[1]):
            rows.append(dict(tag=tag,position=offset+j,**metric(logits[:,j],baseline[:,j])))
    return dict(protected=list(sorted(names)),summary=common.summary(rows),positions=rows)


def select():
    metadata = read('prepare')
    cpu,_ = common.model()
    cases = []
    for row in metadata['trajectories']:
        if row['index'] % 5 == 4:
            ids = torch.tensor(row['token_ids'])
            offset = row['prefix']-1
            cases.append((str(row['index']),offset,ids,cpu(ids,use_cache=False).logits[:,offset:].float()))
    packed = weights()
    modules,counts = install_reference(cpu,packed)
    total = sum(counts.values())
    report = dict(status='running',sources=sources(),weights_sha256=metadata['weights_sha256'],
                  criteria=metadata['criteria'],projection_elements=counts,projection_total_elements=total,
                  max_fp16_weight_fraction=.5,profiles=[],policy='Singles, attention bundle and MLP pairs, then two greedy additions; '
                  'rank failed positions/max L2/mean L2, then protected elements. Only internal trajectories used.')
    seen = {}
    def check(names):
        key = tuple(sorted(names))
        if key not in seen:
            record = evaluate(cpu,modules,key,cases)
            record['fp16_elements'] = sum(counts[name] for name in key)
            report['profiles'].append(record)
            seen[key] = record
            print('internal',','.join(name.split('.')[-2] for name in key) or 'W8',record['summary'],flush=True)
        return seen[key]
    def rank(record):
        return (*common.ranking(record['summary']),record['fp16_elements'])
    proposals = [(),*((name,) for name in NAMES),NAMES[:4],
                 (NAMES[4],NAMES[5]),(NAMES[4],NAMES[6]),(NAMES[5],NAMES[6])]
    for names in proposals:
        assert sum(counts[name] for name in names) <= total//2
        check(names)
    best = min(report['profiles'],key=rank)
    for _ in range(2):
        current = tuple(best['protected'])
        for name in NAMES:
            candidate = tuple(sorted((*current,name)))
            if name not in current and sum(counts[n] for n in candidate)<=total//2:
                check(candidate)
        new = min(report['profiles'],key=rank)
        if new is best:
            break
        best = new
    # FP16 control is never eligible for the budgeted candidate selection.
    control = evaluate(cpu,modules,NAMES,cases)
    assert control['summary']['quality_passed'] == control['summary']['positions'] == 290
    assert control['summary']['max_relative_l2'] == 0
    report.update(status='passed',selected=best['protected'],selected_summary=best['summary'],
                  selected_fp16_weight_fraction=best['fp16_elements']/total,fp16_control=control['summary'],npu_calls=0)
    assert 'torch_intel_npu' not in sys.modules
    write('selection',report)
    print('SELECTED',report['selected'],report['selected_summary'],'FP16 fraction',report['selected_fp16_weight_fraction'],flush=True)


def independent():
    selection = read('selection')
    metadata = read('prepare')
    cpu,tokenizer = common.model()
    cases = []
    for tag,system,prompt in HOLDOUT+FRESH:
        ids = common.chat(tokenizer,system,prompt)
        count = 32 if tag in ('bird','garden','cat') else 1
        tokens = common.trajectory(cpu,ids,ids.shape[1]+count-1)
        cases.append((tag,ids.shape[1]-1,tokens,cpu(tokens,use_cache=False).logits[:,ids.shape[1]-1:].float()))
    ids = tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    ids = ids.repeat(1,(32+ids.shape[1])//ids.shape[1])[:,:32]
    tokens = common.trajectory(cpu,ids,64)
    cases.append(('repeat32',31,tokens,cpu(tokens,use_cache=False).logits[:,31:].float()))
    modules,_ = install_reference(cpu,weights())
    report = dict(status='running',sources=sources(),selection_sha256=common.sha(OUT/'selection.json'),
                  selected=selection['selected'],criteria=metadata['criteria'],
                  cases=[dict(tag=tag,offset=offset,token_ids=ids.tolist()) for tag,offset,ids,_ in cases],variants={})
    for label,names in (('w8',()),('mixed',selection['selected']),('fp16',NAMES)):
        result = evaluate(cpu,modules,names,cases)
        report['variants'][label] = result
        print('independent',label,result['summary'],flush=True)
    assert report['variants']['fp16']['summary']['max_relative_l2']==0
    report.update(status='passed',npu_calls=0)
    write('independent',report)


def npu_run(label):
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    metadata,selection,data = read('prepare'),read('selection'),read('independent')
    assert data['selection_sha256']==common.sha(OUT/'selection.json')
    policy = tuple(selection['selected']) if label=='mixed' else ()
    report = dict(status='running',label=label,source_sha256=common.sha(Path(__file__)),sources=sources(),
                  adapter_sha256=common.sha(common.ROOT/'torch_npu_smollm.py'),weights_sha256=metadata['weights_sha256'],
                  selection_sha256=common.sha(OUT/'selection.json'),protected=list(policy),criteria=metadata['criteria'],
                  positions=[],graphs=[],warm=[],initial_stats=npu.stats())
    def digest(value):
        return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
    def work():
        cpu,_ = common.model()
        stack = FrozenSmolStack.from_checkpoint(common.MODEL,num_layers=30,mode='fp16' if label=='fp16' else 'w8a16',
                    packed_weights=None if label=='fp16' else weights(),fp16_projections=policy)
        for case in data['cases']:
            ids = torch.tensor(case['token_ids'])
            offset = case['offset']
            host = cpu.model.embed_tokens(ids)
            baseline = cpu(ids,use_cache=False).logits[:,offset:].float()
            state = None
            for position in range(ids.shape[1]):
                x = host[:,position:position+1].contiguous().to('intel_npu')
                previous = state
                output,state = stack.prefill(x) if state is None else stack.decode(x,state)
                assert all(cache.length==position+1 for cache in state)
                if position==0:
                    seed = state
                    seed_hash = [digest(v) for c in seed for v in (c.keys,c.values,c._position)]
                if position>=offset:
                    logits = cpu.lm_head(cpu.model.norm(output.cpu()))[:,-1].float()
                    assert torch.isfinite(logits).all()
                    report['positions'].append(dict(tag=case['tag'],position=position,**metric(logits,baseline[:,position-offset])))
            before = npu.stats()
            again,again_state = stack.decode(x,previous)
            after = npu.stats()
            assert after['npu_calls']-before['npu_calls']==30
            assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
            first = [output]+[v for c in state for v in (c.keys,c.values,c._position)]
            second = [again]+[v for c in again_state for v in (c.keys,c.values,c._position)]
            assert [digest(v) for v in first]==[digest(v) for v in second]
            assert seed_hash==[digest(v) for c in seed for v in (c.keys,c.values,c._position)]
            report['warm'].append(dict(tag=case['tag'],calls=30,repeated_arrays=91,seed_arrays=90,payload_copies_and_compiles=0))
            print('NPU',label,case['tag'],common.summary([r for r in report['positions'] if r['tag']==case['tag']]),flush=True)
        for i,layer in enumerate(stack.layers):
            evidence = [value for key,value in layer._evidence.items() if key[-1]=='decode']
            assert len(evidence)==1
            integer = [m for m in evidence[0]['modes'] if not m['elementwise'] and m['wmode'] in (1,2)]
            if label=='fp16':
                assert not integer
            else:
                assert integer and all(m['amode']==0 and m['wmode']==2 for m in integer)
            report['graphs'].append(dict(layer=i,evidence=evidence[0]))
        report['summary'] = common.summary(report['positions'])
    try:
        work()
        report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    gc.collect()
    npu.shutdown()
    final = npu.stats()
    report['final_stats']=final
    assert final['closed'] and not final['context_failed']
    assert all(final[k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
    write('npu-'+label,report)
    print('NPU final',label,report.get('summary'),final['npu_calls'],flush=True)
    return int(report['status']!='passed')


def selfcheck():
    original = torch.nn.Linear(3,2,bias=False).half()
    original.weight.data.copy_(torch.tensor([[.1,.2,.3],[-.2,.7,.01]]).half())
    packed = quantize_weight(original.weight,format='int8')
    module = ReferenceLinear(original,packed)
    x = torch.tensor([[.3,-.2,.1]],dtype=torch.float16)
    expected = (x.double() @ module.decoded.t()).half()
    assert torch.equal(module(x),expected)
    module.protected=True
    assert torch.equal(module(x),original(x))
    assert len(set(NAMES))==7
    print('PASS: quantized-reference and protected-FP16 branches',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case',choices=('prepare','select','independent','selfcheck','npu-fp16','npu-w8','npu-mixed'),required=True)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    assert not os.environ.get('NPUNLOCK_GRAPH_CACHE')
    torch.set_num_threads(8)
    OUT.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        selfcheck()
        if args.case.startswith('npu-'):
            return npu_run(args.case[4:])
        if args.case!='selfcheck':
            {'prepare':prepare,'select':select,'independent':independent}[args.case]()
    return 0


if __name__=='__main__':
    raise SystemExit(main())

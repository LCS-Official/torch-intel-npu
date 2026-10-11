"""Extend the frozen generation protocol without changing its first-round evidence."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from model_cross_device import ROOT, sha, write, metric, distribution

OUT=ROOT/'npunlock-probe/research/results/model-cross-device-followup'
BASE=ROOT/'npunlock-probe/research/results/model-cross-device'
TEMPLATE=ROOT/'npunlock-probe/research/model_cross_device.py'


def run(args):
    assert args.device!='npu' or (args.dtype=='float16' and args.attention=='eager')
    args.label=args.label or f'{args.device}-b{args.batch}-{args.dtype}-{args.attention}-t{args.threads}'
    text=TEMPLATE.read_text()
    changes={
        'ROOT = Path(__file__).resolve().parents[2]':f'ROOT = Path({str(ROOT)!r})\nBASE = Path({str(BASE)!r})',
        "OUT = ROOT / 'npunlock-probe/research/results/model-cross-device'":f'OUT = Path({str(OUT)!r})',
        'sources = [Path(__file__),':f'sources = [Path(__file__), Path({str(Path(__file__).resolve())!r}), Path({str(TEMPLATE)!r}), ROOT/"torch_npu_graph_cache.py", ROOT/"npunlock/python/npunlock/npu4000.py", ROOT/"npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf", ROOT/"npunlock/tests/fixtures/npu4000/kv_select/kv-select-f16.elf",',
        "dtype='float16', decoder_layers=30":"dtype=args.dtype, decoder_layers=30",
        'batch=1, new_tokens=TOKENS':'batch=args.batch, new_tokens=TOKENS',
        'dtype=torch.float16,':'dtype=getattr(torch,args.dtype),',
        "attn_implementation='eager',":"attn_implementation=args.attention,",
        "ids = tokenizer(text, return_tensors='pt').input_ids":"ids = tokenizer(text, return_tensors='pt').input_ids.repeat(args.batch,1)",
        'ids.shape == (1,24)':'ids.shape == (args.batch,24)',
        "attention='eager HF; fixed-capacity 64-token graph for NPU'":"attention=args.attention+' HF; fixed-capacity 64-token graph for NPU'",
        "OUT/'fixture.npz'":"BASE/'fixture.npz'",
        "fixture = {k:saved[k].copy() for k in saved.files}":"fixture = {k:np.repeat(saved[k],args.batch,axis=1 if k=='logits' else 0) for k in saved.files}",
        'overall_tokens_per_second=32000/elapsed':'overall_tokens_per_second=args.batch*32000/elapsed',
        'decode_tokens_per_second=31000/sum(step_ms[1:])':'decode_tokens_per_second=args.batch*31000/sum(step_ms[1:])',
        'generated_cpu.shape == (1,56)':'generated_cpu.shape == (args.batch,56)',
        "('driver_allocations','session_creations','host_device_copies','host_pack_copies')":"('session_creations','host_device_copies','host_pack_copies')",
        "[metric(a,b) for a,b in zip(actual,fixture['logits'])]":"[metric(a,b) for aa,bb in zip(actual,fixture['logits']) for a,b in zip(aa,bb)]",
        'from shave_fp32_pipeline import final':"from shave_fp32_pipeline import final\n            from torch_npu_graph_cache import stats as cache_stats\n            report['graph_cache'] = cache_stats()",
        "report['xpu_peak_memory_allocated_bytes'] = torch.xpu.max_memory_allocated()":
            """report['xpu_peak_memory_allocated_bytes'] = torch.xpu.max_memory_allocated()
                if args.attention=='sdpa':
                    seed=model.model(ids.to(device),use_cache=True)
                    one=torch.from_numpy(anchor_tokens[:,24:25].copy()).to(device)
                    sync()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                        checked=model.model(one,past_key_values=seed.past_key_values,use_cache=True)
                        sync()
                    report['sdpa_profile']={event.key:event.count for event in profiler.key_averages() if 'scaled_dot_product' in event.key}
                    assert report['sdpa_profile'].get('aten::scaled_dot_product_attention')==30
                    del checked,seed""",
    }
    for old,new in changes.items():
        count=text.count(old)
        assert count==(6 if old=="OUT/'fixture.npz'" else 1),(old,count)
        text=text.replace(old,new)
    runners=OUT/'runners';runners.mkdir(parents=True,exist_ok=True)
    runner=runners/(args.label+'.py');runner.write_text(text,encoding='utf-8')
    scope=dict(__name__='followup_case',__file__=str(runner))
    exec(compile(text,str(runner),'exec'),scope)
    sources=[runner,Path(__file__).resolve(),TEMPLATE,ROOT/'torch_npu_smollm.py',ROOT/'torch_intel_npu.py',
             ROOT/'npunlock-probe/torch-device/_torch_intel_npu.pyd',ROOT/'npunlock-probe/session-runtime/npunlock_session.dll',
             ROOT/'torch_npu_graph_cache.py',ROOT/'npunlock/python/npunlock/npu4000.py',
             ROOT/'npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf',ROOT/'npunlock/tests/fixtures/npu4000/kv_select/kv-select-f16.elf']
    snapshots=OUT/'sources';snapshots.mkdir(exist_ok=True)
    frozen={}
    for source in sources:
        digest=sha(source);destination=snapshots/digest
        if not destination.exists():shutil.copyfile(source,destination)
        frozen[str(source.relative_to(ROOT))]=digest
    scope['execute'](args)
    path=OUT/(args.label+'.json');report=json.loads(path.read_text())
    assert report['sources']==frozen
    report['reference_scope']='first-round CPU FP16 logits, identical prompt repeated across batch'
    report['throughput_scope']='aggregate tokens across all batch members'
    write(path,report)


def audit():
    rows=[]
    with np.load(BASE/'fixture.npz',allow_pickle=False) as data:reference=data['logits'].copy();tokens=data['tokens'].copy()
    for path in sorted(OUT.glob('*.json')):
        if path.name=='summary.json':continue
        value=json.loads(path.read_text())
        if 'samples' not in value:continue
        assert value['status']=='passed' and len(value['samples'])==7,path.name
        for name,digest in value['sources'].items():assert sha(OUT/'sources'/digest)==digest,name
        assert value['checkpoint_sha256']==sha(ROOT/'models/SmolLM2-135M-Instruct/model.safetensors')
        assert value['fixture_sha256']==sha(BASE/'fixture.npz')
        evidence=OUT/(value['label']+'-outputs.npz');assert sha(evidence)==value['outputs_sha256']
        batch=value['batch']
        with np.load(evidence,allow_pickle=False) as data:
            expected=np.repeat(reference,batch,axis=1)
            errors=[metric(a,b) for aa,bb in zip(data['replay_logits'],expected) for a,b in zip(aa,bb)]
            assert errors==value['replay_errors']
            assert value['max_replay_relative_l2']==max(r['relative_l2'] for r in errors)
            assert value['replay_argmax_matches']==sum(r['argmax_equal'] for r in errors)
            assert value['greedy_tokens_equal_reference']==bool(np.array_equal(data['greedy_tokens'],np.repeat(tokens,batch,axis=0)))
        for key,summary in value['summary'].items():
            numbers=[sum(r['decoder_ms']) if key=='decoder_total_ms' else r[key] for r in value['samples']]
            assert distribution(numbers)==summary
        rows.append(dict(label=value['label'],device=value['device'],dtype=value['dtype'],batch=batch,threads=value['threads'],
            attention=value['attention'],elapsed_ms=value['summary']['elapsed_ms']['median'],
            ttft_ms=value['summary']['first_token_ms']['median'],tokens_per_second=value['summary']['overall_tokens_per_second']['median'],
            max_relative_l2=value['max_replay_relative_l2'],argmax_matches=value['replay_argmax_matches'],
            quality_gate_passed=value['quality_gate_passed'],greedy_equal=value['greedy_tokens_equal_reference']))
    assert rows
    write(OUT/'summary.json',dict(status='passed',rows=rows))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=('cpu','xpu','npu','audit'),required=True)
    parser.add_argument('--dtype',choices=('float16','float32','bfloat16'),default='float16')
    parser.add_argument('--attention',choices=('eager','sdpa'),default='eager')
    parser.add_argument('--batch',type=int,choices=(1,4,8),default=1)
    parser.add_argument('--threads',type=int,choices=(1,8),default=8)
    parser.add_argument('--label')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.device=='audit':audit()
    else:run(args)

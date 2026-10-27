"""Audit S45 scalar gates, frozen calibration choices and native cache evidence."""
import json
import math
from pathlib import Path
import numpy as np
import w8_guarded_quality as study


def session_source_path(root,relative):
    path=Path(relative.replace('\\','/'))
    if path.is_absolute() or '..' in path.parts:raise ValueError('Session provenance must use a safe relative source path')
    return root/path if path.parts[0]=='npunlock' else root/'npunlock'/path


def main():
    base=study.base;count=0
    base.mixed.weights();base.corrections(32);study.corrections()
    def check(record,positions):
        nonlocal count
        rows=record['positions'];assert len(rows)==positions
        expected=base.mixed.common.summary(rows)
        if 'mean_centered_relative_l2' in record['summary']:
            expected['mean_centered_relative_l2']=float(np.mean([r['centered_relative_l2'] for r in rows]))
        for key,value in expected.items():assert math.isclose(value,record['summary'][key],rel_tol=1e-12,abs_tol=1e-14),(key,value)
        assert all(r['quality_gate_passed']==(r['relative_l2']<.03 and r['next_token_kl']<.02) for r in rows)
        assert all(math.isfinite(r[k]) for r in rows for k in ('relative_l2','next_token_kl'))
        count+=len(rows)
    guard=base.read('guard-evaluation');strength=base.read('strength-selection');evaluation=base.read('strength-evaluation')
    assert guard['status']==evaluation['status']=='passed'
    for group,n in (('internal',290),('regression',142),('fresh',16)):
        for result in guard[group].values():check(result,n)
    rank=base.mixed.common.ranking
    assert guard['selected']==('guarded' if rank(guard['internal']['guarded']['summary'])<rank(guard['internal']['corrected']['summary']) else 'corrected')
    assert strength['alphas']==[.5,.75,1.,1.25]
    for result in strength['internal'].values():check(result,290)
    chosen=min(strength['alphas'],key=lambda a:(*rank(strength['internal'][str(a)]['summary']),abs(a-1)))
    assert chosen==strength['selected_alpha']==evaluation['selected_alpha']
    assert evaluation['selection_sha256']==base.sha(study.OUT/'strength-selection.json')
    for variant in evaluation['variants'].values():
        for group,n in (('regression',142),('fresh',128)):check(variant[group],n)
    for row in base.read('guard')['matrices']:
        assert row['off']+row['half']+row['full']==row['rows']
        assert row['guarded_sse']<=row['plain_sse']+1e-14
    cache=base.read('cache-check');assert cache['status']=='passed' and cache['healthy']
    assert cache['cache_delta']==dict(hits=4,misses=4,compiles=4,writes=4,rejected=0,ir_skipped=4)
    assert len({r['graph_digest'] for r in cache['runs']})==4 and len(cache['runs'])==8
    root=base.mixed.common.ROOT/'npunlock-probe/research/results/smollm-cache-lifetime'
    processes=[]
    for name in ('30-512-normal-family-production','30-256-normal-safe-budget-retry'):
        result=json.loads((root/(name+'.json')).read_text());assert result['status']=='passed' and result['healthy']
        assert result['before_close']['graph_cache_evictions']>0
        assert all(p['held_graphs']==0 and p['stats']['cached_graph_blob_bytes']<=p['stats']['graph_cache_limit_bytes'] for p in result['phases'])
        processes.append(dict(name=name,calls=result['before_close']['npu_calls'],evictions=result['before_close']['graph_cache_evictions']))
    result=json.loads((root/'30-256-normal-safe-budget-retry.json').read_text())
    assert 'prepared/pinned family' in result['guard']['message'] and not result['guard']['stats']['context_failed']
    assert result['guard']['stats']['graph_cache_evictions']==0
    kv=base.read('hardware-kv');assert kv['status']=='passed' and kv['healthy']
    check(kv,98);assert len(kv['warm'])==4
    forward=base.read('hardware-forward');assert forward['status']=='passed' and forward['healthy']
    for label,variant in forward['variants'].items():
        for group,n,fresh in (('regression',142,False),('fresh',128,True)):
            rows=[r for r in variant['positions'] if r['tag'].startswith('s45-')==fresh]
            check(dict(positions=rows,summary=variant['groups'][group]),n)
        assert len(variant['local_checks'])==30 and all(r['relative_l2']<.003 and r['max_scaled_error']<.01 for r in variant['local_checks'])
    assert all(r['quality_gate_passed'] for r in forward['variants']['fp16']['positions'])
    fixture=base.mixed.common.ROOT/'npunlock/tests/fixtures/npu4000'
    manifest=json.loads((fixture/'resident/manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():assert base.sha(fixture/name)==digest
    for name,row in manifest['kernels'].items():
        assert base.sha(fixture/'resident'/(name+'.elf'))==row['elf_sha256']
        assert base.sha(fixture/'resident'/(name+'.s'))==row['assembly_sha256']
    build=json.loads((base.mixed.common.ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
    for name,key in (('npunlock-probe/torch-device/backend.cpp','source_sha256'),
        ('npunlock-probe/torch-device/_torch_intel_npu.pyd','extension_sha256'),
        ('npunlock-probe/session-runtime/npunlock_session.dll','session_sha256')):
        assert base.sha(base.mixed.common.ROOT/name)==build[key]
    session=json.loads((base.mixed.common.ROOT/'npunlock-probe/session-runtime/build.json').read_text(encoding='utf-8-sig'))
    session_sources=session['sources']+session.get('headers',[])
    for row in session_sources:assert base.sha(session_source_path(base.mixed.common.ROOT,row['path']))==row['sha256'].lower()
    expected_failures={('bird',p) for p in (37,38,41,54)}|{('cat',p) for p in (34,35,38,39,48,51,53)}|{('fox',23)}|{('repeat32',p) for p in (32,41,52,54)}
    assert {(r['tag'],r['position']) for r in kv['failed']}==expected_failures
    report=dict(status='passed',scalar_positions=count,cache_processes=processes,
        integrity=dict(resident_sources=len(manifest['source_sha256']),elf=len(manifest['kernels']),assembly=len(manifest['kernels']),
            native=3,session_build=len(session_sources)),
        source_sha256=base.sha(Path(__file__)),quality_source_sha256=base.sha(Path(study.__file__)))
    base.write('audit',report);print(report,flush=True)


if __name__=='__main__':main()

"""Validate fused Power/DPU schedules before running real SmolLM segments."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-b1-fusion'


def main():
    import torch
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolLayer,_rmsnorm_kernel
    from torch_npu_smollm_stack_demo import errors
    from npunlock.npu4000 import compile_fused_rmsnorm
    report=dict(status='running',checks=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['npunlock-probe/research/rmsnorm_b1_model_probe.py','torch_npu_smollm.py','npunlock/python/npunlock/npu4000.py']},
        candidate_sha256=hashlib.sha256((OUT/'candidate.py').read_bytes()).hexdigest())
    try:
        path=OUT/'candidate.py';spec=importlib.util.spec_from_file_location('rmsnorm_fused_candidate',path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);module.__file__=str(ROOT/'torch_npu_smollm.py')
        def work():
            torch.set_num_threads(8);torch.manual_seed(6731)
            for mode in ['fp16','w8a16']:
                model=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode)
                candidate=module.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode)
                for batch in [1]:
                    host=torch.randn((batch,4,576)).half();x=host[:,-1:].contiguous().to('intel_npu')
                    _,state=model.prefill(host[:,:3].contiguous().to('intel_npu'))
                    for segment,kind in [('attention',None),('attention','prefill'),('attention','decode'),('ffn',None)]:
                        tag=f'{mode}-b{batch}-{kind or segment}';entry=dict(tag=tag,mode=mode,batch=batch,segment=segment,kind=kind,status='running')
                        report['checks'].append(entry)
                        graph,xml,data=candidate.graph(tuple(x.shape),segment=segment,kv_cache=kind,fused_norm=True)
                        try:program=compile_fused_rmsnorm(graph,xml,data,_rmsnorm_kernel(),libraries=npu._native)
                        except ValueError as error:
                            entry.update(status='layout_rejected',error=str(error));print(entry,flush=True);continue
                        evidence=json.loads(program.patch_reports[0]);entry['patch']=evidence;entry['dpu']=candidate._inspect_blob(program.graph_blob)
                        (OUT/f'{tag}.xml').write_bytes(xml);(OUT/f'{tag}-original.blob').write_bytes(program.ir_provenance.graph_blob)
                        (OUT/f'{tag}-patched.blob').write_bytes(program.graph_blob)
                        name='input_layernorm.weight' if segment=='attention' else 'post_attention_layernorm.weight'
                        normalized=model._norm_device(x,name);gamma=model._norm_parameter(name,max(2,batch))
                        if segment=='attention' and kind is not None:
                            baseline=model._run_cached_attention(x,normalized,state if kind=='decode' else None)
                        else:baseline=(model._run_graph(x,normalized,segment=segment),)
                        local=(tuple(x.shape),'kv_'+kind,0) if kind is not None else (tuple(x.shape),segment,0)
                        entry['baseline_dpu']=model._evidence[local]
                        old_modes=[m for m in entry['baseline_dpu']['modes'] if not m['elementwise']]
                        new_modes=[m for m in entry['dpu']['modes'] if not m['elementwise']]
                        assert len(old_modes)==len(new_modes),(tag,len(old_modes),len(new_modes))
                        key=('rmsnorm_fused_model_probe',tag)
                        npu._sessions[key]=npu._native.create_inference_session(program.graph_blob,parent=npu._root_session),program;npu._session_creations+=1
                        inputs=[x,gamma]+([state.keys,state.values,state._position] if kind=='decode' else [])
                        specs=[(tuple(v.shape),v.dtype) for v in baseline]
                        before=npu.stats();actual=npu._run_many(key,inputs,specs);after=npu.stats()
                        assert after['npu_calls']-before['npu_calls']==1
                        for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                        old=[v.cpu() for v in baseline];got=[v.cpu() for v in actual]
                        comparisons=[errors(b,a) for a,b in zip(old,got)]
                        equal=[torch.equal(a.view(torch.int16 if a.dtype==torch.float16 else torch.int32),b.view(torch.int16 if b.dtype==torch.float16 else torch.int32)) for a,b in zip(old,got)]
                        assert all(e['relative_l2']<.003 and e['max_scaled_error']<.01 for e in comparisons),(tag,comparisons)
                        entry.update(status='passed',comparisons=comparisons,bits_equal=equal,graph_calls=1,warm_host_copies_and_compiles=0)
                        print({k:v for k,v in entry.items() if k not in ['patch','dpu','baseline_dpu']},flush=True)
        with torch.no_grad():work()
        assert all(e['status']=='passed' and all(e['bits_equal']) for e in report['checks']),report['checks']
        gc.collect();assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers'],npu.stats()
        npu.shutdown();report.update(status='passed',final_stats=npu.stats());assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'model-probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

"""Verify both carriers before executing a complete frozen SmolLM layer."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-whole-layer'


def main():
    import torch
    import torch_intel_npu as npu
    from torch_npu_smollm import _rmsnorm_kernel
    from torch_npu_smollm_stack_demo import errors
    from npunlock.npu4000 import compile_fused_rmsnorm
    path=OUT/'candidate.py';spec=importlib.util.spec_from_file_location('whole_layer_candidate',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);module.__file__=str(ROOT/'torch_npu_smollm.py')
    old_path=OUT/'before/torch_npu_smollm.py';old_spec=importlib.util.spec_from_file_location('whole_probe_before',old_path)
    old=importlib.util.module_from_spec(old_spec);sys.modules[old_spec.name]=old;old_spec.loader.exec_module(old);old.__file__=str(ROOT/'torch_npu_smollm.py')
    FrozenSmolLayer=old.FrozenSmolLayer
    report=dict(status='running',checks=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['npunlock-probe/research/smollm_whole_layer_probe.py','torch_npu_smollm.py','npunlock/python/npunlock/npu4000.py']},
        candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),baseline_adapter_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest())
    try:
        def work():
            torch.set_num_threads(8);torch.manual_seed(7819)
            for mode in ['fp16','w8a16']:
                baseline=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode)
                candidate=module.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode=mode)
                for batch in range(1,9):
                    host=torch.randn(batch,4,576).half();x=host[:,-1:].contiguous().to('intel_npu')
                    _,state=baseline.prefill(host[:,:3].contiguous().to('intel_npu'))
                    for kind in [None,'prefill','decode']:
                        tag=f'{mode}-b{batch}-{kind or "forward"}';entry=dict(tag=tag,mode=mode,batch=batch,kind=kind,status='running');report['checks'].append(entry)
                        graph,xml,weights=candidate.graph(tuple(x.shape),segment='layer',kv_cache=kind,fused_norm=True)
                        (OUT/f'{tag}.xml').write_bytes(xml)
                        class Compiler:
                            def compile_ir(self,*args,**kwargs):
                                result=npu._native.compile_ir(*args,**kwargs)
                                (OUT/f'{tag}-original.blob').write_bytes(result.graph_blob);return result
                        try:program=compile_fused_rmsnorm(graph,xml,weights,_rmsnorm_kernel(),libraries=Compiler(),whole_layer=True)
                        except ValueError as error:
                            entry.update(status='layout_rejected',error=str(error));print(entry,flush=True);continue
                        entry['patch']=json.loads(program.patch_reports[0]);entry['dpu']=candidate._inspect_blob(program.graph_blob)
                        (OUT/f'{tag}-patched.blob').write_bytes(program.graph_blob)
                        if kind is None:old=(baseline(x),)
                        else:
                            value,cache=baseline.prefill(x) if kind=='prefill' else baseline.decode(x,state)
                            old=(value,cache.keys,cache.values)+((cache._position,) if kind=='decode' else ())
                        inputs=[x,*[baseline._norm_parameter(name,max(2,batch)) for name in ['input_layernorm.weight','post_attention_layernorm.weight']]]
                        if kind=='decode':inputs.extend([state.keys,state.values,state._position])
                        key=('whole_layer_probe',tag)
                        npu._sessions[key]=npu._native.create_inference_session(program.graph_blob,parent=npu._root_session),program;npu._session_creations+=1
                        before=npu.stats();actual=npu._run_many(key,inputs,[(tuple(v.shape),v.dtype) for v in old]);after=npu.stats()
                        assert after['npu_calls']-before['npu_calls']==1
                        for field in ['host_upload_copies','host_download_copies','session_creations']:assert before[field]==after[field]
                        previous=[v.cpu() for v in old];current=[v.cpu() for v in actual]
                        comparisons=[errors(b,a) for a,b in zip(previous,current)]
                        equal=[torch.equal(a.view(torch.int16 if a.dtype==torch.float16 else torch.int32),b.view(torch.int16 if b.dtype==torch.float16 else torch.int32)) for a,b in zip(previous,current)]
                        assert all(e['relative_l2']<.003 and e['max_scaled_error']<.01 for e in comparisons),(tag,comparisons)
                        entry.update(status='passed',comparisons=comparisons,bits_equal=equal,graph_calls=1,warm_host_copies_and_compiles=0)
                        print({k:v for k,v in entry.items() if k not in ['patch','dpu']},flush=True)
        with torch.no_grad():work()
        bad=[e['tag'] for e in report['checks'] if e['status']!='passed' or not all(e['bits_equal'])]
        assert not bad,bad
        gc.collect();assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers'],npu.stats()
        npu.shutdown();report.update(status='passed',final_stats=npu.stats());assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'model-probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

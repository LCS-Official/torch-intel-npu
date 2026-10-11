"""Default-path parity: B1..B8 boundaries and all 30 decoder layers."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import types
import numpy as np
import torch
from kv_gather_probe import arrays
from kv_select_checks import OUT
from w8_quality_probe import ROOT, MODEL, sha


def baseline():
    path=OUT/'before/torch_npu_smollm.py';source=path.read_bytes()
    expected=json.loads((OUT/'before.json').read_text())['torch_npu_smollm.py']
    if hashlib.sha256(source).hexdigest()!=expected:raise RuntimeError('Baseline source changed')
    module=types.ModuleType('_smol_before_select');module.__file__=str(ROOT/'torch_npu_smollm.py')
    sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__);return module


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('batch','full'),required=True)
    parser.add_argument('--mode',choices=('fp16','w8a16'),default='fp16');args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    import torch_intel_npu as npu
    import torch_npu_smollm as current
    from safetensors import safe_open
    from transformers import AutoTokenizer
    torch.set_num_threads(8);before=baseline();directory=OUT/('batch' if args.case=='batch' else f'full-{args.mode}')
    directory.mkdir(parents=True,exist_ok=True);saved={};graph_tags=set()
    report=dict(status='running',case=args.case,mode=args.mode,source_sha256=sha(Path(__file__)),
        sources={name:sha(ROOT/name) for name in ('torch_npu_smollm.py','npunlock/python/npunlock/npu4000.py',
            'torch_intel_npu.py','npunlock/tests/fixtures/npu4000/kv_select.c','npunlock/tests/fixtures/npu4000/kv_select/kv-select-f16.elf')},
        checks=[],graphs=[],cpu_quality=[])
    def compare(tag,first,second):
        a,b=arrays(first),arrays(second)
        for i,(x,y) in enumerate(zip(a,b)):
            np.testing.assert_array_equal(x.view(np.uint8),y.view(np.uint8),err_msg=f'{tag}/{i}')
            saved[f'{tag}/baseline/{i}']=x;saved[f'{tag}/current/{i}']=y
        report['checks'].append(dict(tag=tag,bit_equal_arrays=len(a)))
    def graphs(tag,models):
        if tag in graph_tags:return
        graph_tags.add(tag)
        for name,model in models.items():
            for index,layer in enumerate(model.layers):
                local=next(k for k in layer._keys if k[-1]=='decode');program=npu._sessions[layer._keys[local]][1]
                graph,xml,_=layer.graph(local[0],segment='layer',kv_cache='decode',fused_norm=True)
                stem=f'{tag}-{name}-l{index}';(directory/f'{stem}.blob').write_bytes(program.graph_blob);(directory/f'{stem}.xml').write_bytes(xml)
                reports=[json.loads(r) for r in program.patch_reports]
                assert len(reports)==(2 if name=='current' else 1)
                if name=='current':assert reports[1]['carrier']=='Select' and reports[1]['invocations']>0
                report['graphs'].append(dict(stem=stem,blob_sha256=sha(directory/f'{stem}.blob'),
                    xml_sha256=sha(directory/f'{stem}.xml'),patch_reports=reports))
    def warm(tag,models,one,seeds,expected):
        for name,model in models.items():
            start=npu.stats();again=model.decode(one,seeds[name]);end=npu.stats()
            assert end['npu_calls']-start['npu_calls']==model.num_layers
            assert all(end[k]==start[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
            for i,value in enumerate(arrays(again)):
                np.testing.assert_array_equal(value.view(np.uint8),saved[f'{expected}/{name}/{i}'].view(np.uint8))
        report.setdefault('warm_checks',[]).append(dict(tag=tag,graph_calls_per_variant=model.num_layers,host_copies_and_compiles=0))
    def batch_check():
        import w8_calibration as calibration
        calibration.OUT=OUT/'calibrated-current';packed=calibration.load_weights()
        report['calibration_weights_sha256']=sha(calibration.OUT/'weights.npz')
        with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as file:embedding=file.get_tensor('model.embed_tokens.weight')
        ids=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
        ids=ids.repeat(1,(64+ids.shape[1])//ids.shape[1])[:,:64]
        for mode in ('fp16','w8a16'):
            models={name:module.FrozenSmolStack.from_checkpoint(MODEL,num_layers=1,mode=mode,
                packed_weights={0:packed[0]} if mode=='w8a16' else None) for name,module in (('baseline',before),('current',current))}
            for batch in range(1,9):
                host=embedding[ids.repeat(batch,1)].half().contiguous()
                for prefix in (1,7,32,63):
                    tag=f'b{batch}-s{prefix}-{mode}';x=host[:,:prefix].contiguous().to('intel_npu')
                    one=host[:,prefix:prefix+1].contiguous().to('intel_npu')
                    seeds={name:model.prefill(x) for name,model in models.items()};compare(tag+'/prefill',*seeds.values())
                    outputs={name:model.decode(one,seeds[name][1]) for name,model in models.items()};compare(tag+'/decode',*outputs.values())
                    assert all(c.length==prefix+1 for out in outputs.values() for c in out[1])
                    if prefix<63:
                        two=host[:,prefix+1:prefix+2].contiguous().to('intel_npu')
                        advanced={name:model.decode(two,outputs[name][1]) for name,model in models.items()}
                        compare(tag+'/advance',*advanced.values());del two,advanced
                    warm(tag,models,one,{name:seed[1] for name,seed in seeds.items()},tag+'/decode')
                    for name,seed in seeds.items():
                        for i,value in enumerate(arrays(seed)):
                            np.testing.assert_array_equal(value.view(np.uint8),saved[f'{tag}/prefill/{name}/{i}'].view(np.uint8))
                    graphs(f'b{batch}-{mode}',models)
                    del x,one,seeds,outputs
                print(mode,'batch',batch,'4 cache boundaries passed',flush=True)
    def full_check():
        from transformers import AutoModelForCausalLM,DynamicCache
        from torch_npu_smollm_stack_demo import errors
        cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
        total=cpu.config.num_hidden_layers;assert total==30
        models={name:module.FrozenSmolStack.from_checkpoint(MODEL,num_layers=total,mode=args.mode)
            for name,module in (('baseline',before),('current',current))}
        report.update(npu_decoder_layers=total,embedding_and_final_norm_and_lm_head='explicit_cpu',weight_mode=args.mode)
        ids=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
        ids=ids.repeat(1,(32+ids.shape[1])//ids.shape[1])[:,:32];cache=DynamicCache(config=cpu.config)
        reference=cpu(ids,past_key_values=cache,use_cache=True);logits=reference.logits[:,-1].float()
        x=cpu.model.embed_tokens(ids).to('intel_npu');seeds={name:model.prefill(x) for name,model in models.items()}
        compare('prefill',*seeds.values());states={name:seed[1] for name,seed in seeds.items()}
        def quality(tag,outputs):
            values={name:cpu.lm_head(cpu.model.norm(value.cpu()))[:,-1].float() for name,(value,_) in outputs.items()}
            assert torch.equal(values['baseline'],values['current'])
            for name,value in values.items():saved[f'{tag}/{name}/logits']=value.numpy().copy()
            saved[f'{tag}/cpu/logits']=logits.numpy().copy()
            result=errors(values['current'],logits)
            report['cpu_quality'].append(dict(tag=tag,**result,next_token_equal=bool(values['current'].argmax(-1)==logits.argmax(-1))))
        quality('prefill',seeds);saved['input_ids']=ids.numpy().copy()
        for step in range(32):
            token=logits.argmax(-1,keepdim=True);saved[f'step-{step}/forced-token']=token.numpy().copy()
            reference=cpu(token,past_key_values=cache,use_cache=True);logits=reference.logits[:,-1].float()
            one=cpu.model.embed_tokens(token).to('intel_npu');previous=states
            outputs={name:model.decode(one,previous[name]) for name,model in models.items()}
            tag=f'chain-{33+step}';compare(tag,*outputs.values());quality(tag,outputs)
            states={name:value[1] for name,value in outputs.items()}
            assert all(c.length==33+step for state in states.values() for c in state)
            if step==31:warm(tag,models,one,previous,tag)
            print(args.mode,'all 30 layers, cache position',33+step,'exact',flush=True)
        for name,seed in seeds.items():
            for i,value in enumerate(arrays(seed)):
                np.testing.assert_array_equal(value.view(np.uint8),saved[f'prefill/{name}/{i}'].view(np.uint8))
        graphs('all30',models)
    try:
        with torch.no_grad():batch_check() if args.case=='batch' else full_check()
        gc.collect();npu.shutdown();report['final_stats']=npu.stats()
        assert report['final_stats']['closed'] and not report['final_stats']['context_failed']
        assert all(report['final_stats'][key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
        path=directory/'outputs.npz';np.savez(path,**saved);report.update(status='passed',outputs_sha256=sha(path),output_array_count=len(saved))
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (directory/'integration.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

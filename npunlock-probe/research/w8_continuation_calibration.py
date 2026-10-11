"""Calibrate 40 CPU greedy trajectories, reuse the frozen all-layer NPU checks.

2560 activation rows fit the existing 4M-element calibration bounds even for
1536-wide projections. Historical profiles and the runtime remain unchanged.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import w8_full_calibration as full
from w8_calibration import CALIBRATION,input_name,unpack
from w8_quality_probe import ROOT,MODEL,sha

OUT=ROOT/'npunlock-probe/research/results/w8-continuation-calibration'
BEFORE=full.OUT
PROMPTS=CALIBRATION+[
    ('You are a helpful assistant.',prompt) for prompt in (
        'Explain why metal expands when heated.',
        'How does a battery power a flashlight?',
        'Describe how clouds form.',
        'Explain the fraction one half to a child.',
        'A box has six rows of four oranges. How many oranges are in the box?',
        'If you spend twelve out of twenty dollars, how much money remains?',
        'What is the difference between a noun and a verb?',
        'Give an example of a sentence in the past tense.',
        'Write a Python function that counts vowels in a string.',
        'Explain what a loop does in programming.',
        'Write a Python function that adds all the numbers in a list.',
        'Translate see you tomorrow into French.',
        'Give short instructions for recycling paper.',
        'How do muscles help us move?',
        'Explain what a magnet attracts.',
        'Describe how bread dough rises.',
        'Give advice to a beginner learning the piano.',
        'What are the benefits of regular sleep?',
    )
]+[('Tell stories.',prompt) for prompt in (
    'Write a story about a young potter repairing a cracked bowl.',
    'Tell a story about two friends building a wooden boat.',
    'Write a story about a librarian discovering a room behind a bookshelf.',
    'Tell a story about a shoemaker making tiny boots for a toy.',
    'Write a story about a traveler sharing dinner at a remote inn.',
    'Tell a story about a musician helping a village prepare a festival.',
    'Write a story about a lighthouse keeper noticing a missing boat.',
    'Tell a story about an inventor building an unexpected useful machine.',
    'Write a story about a runner learning patience during practice.',
    'Tell a story about two sisters solving a puzzle in an attic.',
)]
SOURCES=full.SOURCES+['npunlock-probe/research/w8_continuation_calibration.py',
    'npunlock-probe/research/w8_full_cpu_diagnostic.py','npunlock-probe/research/w8_quality_audit.py',
    'models/SmolLM2-135M-Instruct/config.json','models/SmolLM2-135M-Instruct/tokenizer.json',
    'models/SmolLM2-135M-Instruct/tokenizer_config.json']


def configure():
    full.OUT=OUT;full.SOURCES=SOURCES;full.CALIBRATION=PROMPTS


def prepare():
    from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
    from torch_npu_lowbit import calibrate_int8,quantize_weight
    from w8_calibration_probe import HOLDOUT,FRESH
    assert len(PROMPTS)==len(set(PROMPTS))==40
    assert set(PROMPTS).isdisjoint((system,prompt) for _,system,prompt in HOLDOUT+FRESH)
    start=time.perf_counter()
    report=dict(status='running',method='gptq',num_layers=30,calibration_texts=PROMPTS,
        sources={name:sha(ROOT/name) for name in SOURCES},checkpoint_sha256=sha(MODEL/'model.safetensors'),
        calibration_prompts_disjoint=True,calibrator='torch_npu_lowbit.calibrate_int8',
        gptq_parameters=dict(damping=.01,block_size=128,activation_order=True,solver_dtype='float64',row_scale='original_fp16'),
        policy='CPU FP16 full-prompt prefill then greedy cached decode to 64 consumed tokens; ignore EOS',inputs=[],weights=[])
    cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);assert len(cpu.model.layers)==30
    captured={f'{i}/{key}':[] for i in range(30) for key in ('norm1','attention','norm2','gated')};hooks=[]
    for i,layer in enumerate(cpu.model.layers):
        for name in ('self_attn.q_proj','self_attn.o_proj','mlp.gate_proj','mlp.down_proj'):
            key=f'{i}/{input_name(name+".weight")}'
            def collect(module,args,key=key):captured[key].append(args[0].detach().reshape(-1,args[0].shape[-1]).clone())
            hooks.append(layer.get_submodule(name).register_forward_pre_hook(collect))
    try:
        for index,(system,prompt) in enumerate(PROMPTS):
            ids=full.chat(tokenizer,system,prompt);prefix=ids.shape[1];cache=DynamicCache(config=cpu.config)
            generated=ids.clone();output=cpu(ids,past_key_values=cache,use_cache=True)
            while generated.shape[1]<64:
                token=output.logits[:,-1].argmax(-1,keepdim=True);generated=torch.cat((generated,token),1)
                output=cpu(token,past_key_values=cache,use_cache=True)
            assert generated.shape==(1,64) and cache.get_seq_length()==64
            report['inputs'].append(dict(system=system,prompt=prompt,input_tokens=prefix,decode_tokens=64-prefix,
                token_ids=generated.tolist(),continuation=tokenizer.decode(generated[0,prefix:],skip_special_tokens=False)))
            print('captured trajectory',index+1,'/40, prefill',prefix,'decode',64-prefix,flush=True)
    finally:
        for hook in hooks:hook.remove()
    inputs={key:torch.cat(value) for key,value in captured.items()};del captured,output,cache
    assert {value.shape[0] for value in inputs.values()}=={2560}
    path=OUT/'calibration-inputs.npz';np.savez(path,**{key:value.numpy() for key,value in inputs.items()})
    report.update(inputs_sha256=sha(path),activation_rows=2560,
        prefill_token_rows=sum(row['input_tokens'] for row in report['inputs']),
        decode_token_rows=sum(row['decode_tokens'] for row in report['inputs']))
    arrays={}
    for i,layer in enumerate(cpu.model.layers):
        for name,parameter in layer.named_parameters():
            if parameter.ndim!=2:continue
            weight=parameter.detach().half().contiguous();observed=inputs[f'{i}/{input_name(name)}'];x=observed.double()
            original=quantize_weight(weight,format='int8');q=calibrate_int8(weight,observed);target=x@weight.double().t()
            def reconstruction(packed):
                c,s=unpack(packed);prediction=(x@torch.from_numpy(c.astype(np.float64)).t())*torch.from_numpy(s.astype(np.float64))[None]
                return (prediction-target).square().sum(0)
            before=reconstruction(original);after=reconstruction(q);assert torch.all(after<=before+1e-18)
            c0,s0=unpack(original);c,s=unpack(q);assert np.array_equal(s,s0) and c.min()>=-127
            key=f'{i}/{name}';arrays[key+'/codes']=c;arrays[key+'/scales']=s
            report['weights'].append(dict(layer=i,name=name,rows=len(s),activation_rows=len(x),changed_rows=int(np.any(c0!=c,axis=1).sum()),
                original_output_squared_error=float(before.sum()),calibrated_output_squared_error=float(after.sum()),
                original_codes_sha256=hashlib.sha256(original.data).hexdigest(),calibrated_codes_sha256=hashlib.sha256(q.data).hexdigest()))
        print('calibrated layer',i+1,'/30',flush=True)
    assert len(report['weights'])==210 and len(arrays)==420
    path=OUT/'weights.npz';np.savez(path,**arrays);report['weights_sha256']=sha(path)
    snapshots=[]
    for name,digest in report['sources'].items():
        assert sha(ROOT/name)==digest
        target=OUT/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        snapshots.append(dict(path=name,sha256=digest))
    (OUT/'snapshots.json').write_text(json.dumps(snapshots,indent=2),encoding='utf-8')
    report.update(status='passed',elapsed_seconds=time.perf_counter()-start);full.write('prepare',report)
    print('prepared',report['activation_rows'],'rows in',report['elapsed_seconds'],'seconds',flush=True)
    del inputs,cpu;gc.collect()


def audit():
    from w8_quality_audit import logits_same,same
    from w8_calibration_probe import metric
    from torch_npu_smollm_stack_demo import errors
    def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
    metadata=read('prepare.json');packed=full.weights();assert len(packed)==30
    assert metadata['activation_rows']==2560 and metadata['prefill_token_rows']+metadata['decode_token_rows']==2560
    assert len(metadata['inputs'])==40 and len(metadata['weights'])==210 and metadata['calibration_prompts_disjoint']
    assert sha(OUT/'calibration-inputs.npz')==metadata['inputs_sha256']
    assert all(row['calibrated_output_squared_error']<=row['original_output_squared_error'] for row in metadata['weights'])
    for name,digest in metadata['sources'].items():assert sha(ROOT/name)==sha(OUT/'sources'/name)==digest
    result=dict(status='passed',sources=metadata['sources'],weights_sha256=metadata['weights_sha256'],activation_rows=2560,
        calibrated_matrices=210,npu_calls=0,quality_metric_rechecks=0,local_execution_rechecks=0,decode_graphs=0,
        historical_cpu_arrays_bit_equal=0,historical_npu_control_arrays_bit_equal=0,runs={})
    for case,count in (('holdout',18),('decode',129),('diagnostic',1)):
        report=read(case+'.json');assert report['status']=='passed' and report['sources']==metadata['sources']
        assert report['weights_sha256']==metadata['weights_sha256'] and report['npu_decoder_layers']==30 and len(report['positions'])==count
        assert report['criteria']==dict(logits_relative_l2=.03,next_token_kl=.02,local_relative_l2=.003,local_max_scaled_error=.01)
        final=report['final_stats'];assert final['closed'] and not final['context_failed']
        assert all(final[key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
        result['npu_calls']+=final['npu_calls'];path=OUT/(case+'-outputs.npz');assert sha(path)==report['outputs_sha256']
        with np.load(path,allow_pickle=False) as arrays:
            assert len(arrays.files)==report['output_array_count']
            for row in report['positions']:
                baseline=torch.from_numpy(arrays[row['tag']+'/baseline-logits'])
                for label,record in row['variants'].items():
                    logits_same(torch.from_numpy(arrays[f'{row["tag"]}/{label}-logits']),baseline,record)
                    assert record['quality_gate_passed']==(record['relative_l2']<.03 and record['next_token_kl']<.02)
                    result['quality_metric_rechecks']+=1
            for local in report.get('local_layers',[]):
                key=f'{report["diagnostic_prompt"]["tag"]}/{local["variant"]}/layer-{local["layer"]}'
                error=errors(torch.from_numpy(arrays[key+'-actual']),torch.from_numpy(arrays[key+'-reference']));same(error,local)
                assert error['relative_l2']<.003 and error['max_scaled_error']<.01;result['local_execution_rechecks']+=1
            for label,record in report.get('cpu_staged_quality',{}).items():
                tag=report['diagnostic_prompt']['tag'];logits_same(torch.from_numpy(arrays[f'{tag}/{label}-staged-logits']),torch.from_numpy(arrays[tag+'/baseline-logits']),record)
                result['quality_metric_rechecks']+=1
            if case!='diagnostic':
                with np.load(BEFORE/(case+'-outputs.npz'),allow_pickle=False) as old:
                    for key in arrays.files:
                        if key.endswith(('/ids','/tokens','/baseline-logits')):
                            np.testing.assert_array_equal(arrays[key].view(np.uint8),old[key].view(np.uint8));result['historical_cpu_arrays_bit_equal']+=1
                        elif key.endswith(('/fp16-logits','/original-logits')):
                            np.testing.assert_array_equal(arrays[key].view(np.uint8),old[key].view(np.uint8));result['historical_npu_control_arrays_bit_equal']+=1
        for graph in report['decode_graph_evidence']:
            assert len(graph['evidence'])==1;evidence=graph['evidence'][0];selection=evidence['kv_select'];norm=evidence['rmsnorm']
            assert selection['carrier']=='Select' and selection['invocations']>0
            assert selection['kernel_elf_sha256']==metadata['sources']['npunlock/tests/fixtures/npu4000/kv_select/kv-select-f16.elf']
            assert norm['patched_sha256']==selection['original_sha256'] and selection['patched_sha256']==evidence['blob_sha256']
            integer=[mode for mode in evidence['modes'] if not mode['elementwise'] and mode['wmode'] in (1,2)]
            assert not integer if graph['variant']=='fp16' else integer and all(mode['amode']==0 and mode['wmode']==2 for mode in integer)
            result['decode_graphs']+=1
        for warm in report['warm_checks']:assert warm['graph_calls_per_variant']==30 and warm['host_copies_and_compiles']==0
        result['runs'][case]=dict(positions=count,npu_calls=final['npu_calls'],variants={label:dict(
            max_relative_l2=max(row['variants'][label]['relative_l2'] for row in report['positions']),
            mean_relative_l2=float(np.mean([row['variants'][label]['relative_l2'] for row in report['positions']])),
            max_kl=max(row['variants'][label]['next_token_kl'] for row in report['positions']),
            quality_passed=sum(row['variants'][label]['quality_gate_passed'] for row in report['positions']),
            next_token_equal=sum(row['variants'][label]['next_token_equal'] for row in report['positions'])) for label in ('fp16','original','calibrated')})
    cpu=read('cpu-reproduction.json');assert cpu['status']=='passed' and not cpu['npu_backend_imported'] and cpu['npu_calls']==0
    assert cpu['source_sha256']==sha(ROOT/'npunlock-probe/research/w8_full_cpu_diagnostic.py') and cpu['sources']==metadata['sources']
    assert cpu['decode_report_sha256']==sha(OUT/'decode.json') and cpu['weights_sha256']==metadata['weights_sha256']
    assert cpu['outputs_sha256']==sha(OUT/'cpu-reproduction-outputs.npz')
    with np.load(OUT/'cpu-reproduction-outputs.npz',allow_pickle=False) as arrays:
        for row in cpu['cases']:
            for label,record in row['cpu_staged_variants'].items():
                logits_same(torch.from_numpy(arrays[f'{row["tag"]}/{label}-staged-logits']),torch.from_numpy(arrays[row['tag']+'/baseline-logits']),record)
                result['quality_metric_rechecks']+=1
    assert result['decode_graphs']==180 and result['historical_cpu_arrays_bit_equal']==167 and result['historical_npu_control_arrays_bit_equal']==294
    result['candidate_recommended']=all(result['runs'][case]['variants']['calibrated']['quality_passed']==result['runs'][case]['positions'] for case in ('holdout','decode','diagnostic'))
    full.write('audit',result);print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('prepare','holdout','decode','diagnostic','cpu','audit'),required=True);args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';torch.set_num_threads(8);OUT.mkdir(parents=True,exist_ok=True);configure()
    with torch.no_grad():
        if args.case=='prepare':prepare();return 0
        if args.case=='audit':audit();return 0
        if args.case=='cpu':
            import w8_full_cpu_diagnostic as check
            check.OUT=OUT;check.SOURCES=SOURCES;check.weights=full.weights;check.main();return 0
        return full.evaluate(args.case)


if __name__=='__main__':raise SystemExit(main())

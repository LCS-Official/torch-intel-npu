"""Small HF architecture migration without weight/tokenizer downloads."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')


def hf_model(kind):
    from transformers import LlamaConfig,LlamaForCausalLM,GPT2Config,GPT2LMHeadModel
    if kind=='llama':
        return LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=32,
            attention_dropout=0.,bos_token_id=1,eos_token_id=2,pad_token_id=0))
    return GPT2LMHeadModel(GPT2Config(vocab_size=32,n_embd=16,n_layer=2,n_head=4,n_inner=32,n_positions=32,n_ctx=32,
        resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,bos_token_id=1,eos_token_id=2,pad_token_id=0))


def migrate_to_npu(model):
    import torch
    # PrivateUse1 conversion otherwise replaces Parameters and can split tied weights.
    original=torch.__future__.get_swap_module_params_on_conversion()
    torch.__future__.set_swap_module_params_on_conversion(True)
    try:return model.to('intel_npu')
    finally:torch.__future__.set_swap_module_params_on_conversion(original)


def execute(kind,case,precision,backend='math'):
    import torch
    import torch_intel_npu as npu
    from torch.nn.attention import sdpa_kernel,SDPBackend
    torch.set_num_threads(1);torch.manual_seed(419);started=time.perf_counter()
    dtype=getattr(torch,precision);row=dict(kind=kind,case=case,precision=precision,backend=backend)
    def device_call(fn,*args,**kwargs):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION if backend=='flash' else SDPBackend.MATH):return fn(*args,**kwargs)
    def close(actual,expected):
        torch.testing.assert_close(actual.detach().cpu(),expected.detach(),
            rtol=.005 if dtype==torch.float16 else 5e-5,atol=.001 if dtype==torch.float16 else 3e-6)
    def probe():
        cpu=hf_model(kind).to(dtype).eval();model=migrate_to_npu(copy.deepcopy(cpu))
        ids=torch.tensor([[1,4,7,9]]);mask=torch.ones_like(ids)
        x,m=ids.to('intel_npu'),mask.to('intel_npu')
        stage='forward';observations=[]
        try:
            with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
                expected=cpu(ids,attention_mask=mask,use_cache=True)
                actual=device_call(model,x,attention_mask=m,use_cache=True);close(actual.logits,expected.logits)
                observations.append(dict(stage=stage,status='passed'))
                stage='decode'
                token=torch.tensor([[3]]);next_mask=torch.ones((1,5),dtype=torch.int64)
                expected=cpu(token,attention_mask=next_mask,past_key_values=expected.past_key_values,use_cache=True)
                actual=device_call(model,token.to('intel_npu'),attention_mask=next_mask.to('intel_npu'),
                             past_key_values=actual.past_key_values,use_cache=True)
                close(actual.logits,expected.logits);observations.append(dict(stage=stage,status='passed'))
                stage='generate'
                expected=cpu.generate(ids,attention_mask=mask,max_new_tokens=3,do_sample=False)
                actual=device_call(model.generate,x,attention_mask=m,max_new_tokens=3,do_sample=False)
                torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
                observations.append(dict(stage=stage,status='passed',tokens=expected.tolist()))
        except Exception as error:
            observations.append(dict(stage=stage,status='blocked',error=str(error),traceback=traceback.format_exc()))
        return dict(attention=cpu.config._attn_implementation,parameters=sum(p.numel() for p in model.parameters()),
                    observations=observations,all_stages_passed=all(r['status']=='passed' for r in observations))
    def models():
        print(kind,case,precision,'constructing two-layer models',flush=True)
        cpu=hf_model(kind).to(dtype).eval();model=migrate_to_npu(copy.deepcopy(cpu))
        assert all(p.device.type=='intel_npu' for p in model.parameters())
        assert dict(model.named_parameters()).keys()==dict(cpu.named_parameters()).keys()
        names={id(p):name for name,p in cpu.named_parameters()}
        actual=dict(model.named_parameters(remove_duplicate=False))
        for name,p in cpu.named_parameters(remove_duplicate=False):assert actual[name] is actual[names[id(p)]],name
        return cpu,model
    def inference():
        cpu,model=models();records=[]
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            for ids in (torch.tensor([[1,4,7,9]]),torch.tensor([[0,0,1,4],[1,3,7,9]])):
                print('inference: batch',ids.shape[0],'prefill and cached decode',flush=True)
                mask=ids.ne(0).long();x,m=ids.to('intel_npu'),mask.to('intel_npu')
                expected=cpu(ids,attention_mask=mask,use_cache=True);actual=device_call(model,x,attention_mask=m,use_cache=True)
                close(actual.logits,expected.logits)
                for step in range(3):
                    token=torch.full((ids.shape[0],1),3+step,dtype=torch.int64)
                    ids=torch.cat((ids,token),1);mask=torch.cat((mask,torch.ones_like(token)),1)
                    expected=cpu(token,attention_mask=mask,past_key_values=expected.past_key_values,use_cache=True)
                    actual=device_call(model,token.to('intel_npu'),attention_mask=mask.to('intel_npu'),past_key_values=actual.past_key_values,use_cache=True)
                    close(actual.logits,expected.logits)
                    full=cpu(ids,attention_mask=mask,use_cache=False);close(actual.logits,full.logits[:,-1:])
                    assert actual.past_key_values.get_seq_length()==ids.shape[1]
                    for layer in actual.past_key_values.layers:
                        assert layer.keys.device.type=='intel_npu' and layer.values.device.type=='intel_npu'
                prompt=ids[:,:4].contiguous();mask=prompt.ne(0).long()
                for cache in (True,False):
                    print('inference: generate batch',prompt.shape[0],'use_cache',cache,flush=True)
                    options=dict(attention_mask=mask,max_new_tokens=3,do_sample=False,use_cache=cache,
                                 return_dict_in_generate=True,output_scores=True)
                    expected=cpu.generate(prompt,**options)
                    options['attention_mask']=mask.to('intel_npu');actual=device_call(model.generate,prompt.to('intel_npu'),**options)
                    torch.testing.assert_close(actual.sequences.cpu(),expected.sequences,rtol=0,atol=0)
                    assert len(actual.scores)==len(expected.scores)
                    for a,b in zip(actual.scores,expected.scores):close(a,b)
                    records.append(dict(batch=prompt.shape[0],left_padding=bool((mask==0).any()),
                                        use_cache=cache,tokens=expected.sequences.tolist(),all_scores_match=True))
        return dict(parameters=sum(p.numel() for p in model.parameters()),layers=2,attention=cpu.config._attn_implementation,
                    three_cached_steps_and_full_prefix_reference=True,generation=records,kv_stays_on_npu=True)
    def sampling():
        cpu,model=models();ids=torch.tensor([[1,4,7,9]]);x=ids.to('intel_npu');mask=torch.ones_like(x)
        saved=torch.intel_npu.get_rng_state();cpu_state=torch.get_rng_state()
        options=dict(attention_mask=mask,max_new_tokens=3,do_sample=True,top_k=4,top_p=1.,temperature=1.,
                     return_dict_in_generate=True,output_scores=True)
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            print('sampling: first draw and state replay',flush=True)
            first=device_call(model.generate,x,**options);advanced=torch.intel_npu.get_rng_state()
            torch.intel_npu.set_rng_state(saved);second=device_call(model.generate,x,**options)
            torch.testing.assert_close(first.sequences.cpu(),second.sequences.cpu(),rtol=0,atol=0)
            assert torch.equal(advanced,torch.intel_npu.get_rng_state())
            sequence=first.sequences.cpu()
            for step,(a,b) in enumerate(zip(first.scores,second.scores)):
                torch.testing.assert_close(a.cpu(),b.cpu(),rtol=0,atol=0)
                expected=cpu(sequence[:,:4+step],attention_mask=torch.ones_like(sequence[:,:4+step])).logits[:,-1].float()
                threshold=torch.topk(expected,4).values[:,-1,None]
                close(a,expected.masked_fill(expected<threshold,float('-inf')))
        assert torch.equal(cpu_state,torch.get_rng_state())
        torch.intel_npu.set_rng_state(saved)
        return dict(top_k=4,tokens=sequence.tolist(),state_and_scores_bitwise_replay=True,
                    every_sample_score_matches_cpu_teacher_forcing=True,cpu_rng_unchanged=True)
    def training():
        if dtype!=torch.float32:raise ValueError('This strict training acceptance uses FP32')
        from torch_npu_transformer_demo import cpu_tree
        cpu,model=models();cpu.train();model.train()
        ids=torch.tensor([[1,4,7,9],[1,3,6,8]]);mask=torch.ones_like(ids)
        x,m=ids.to('intel_npu'),mask.to('intel_npu')
        opts=[torch.optim.SGD(net.parameters(),lr=.03,momentum=.5,foreach=False,fused=False) for net in (model,cpu)]
        losses=[];checkpoint=None;final=None
        for step in range(2):
            print('training: CPU/NPU step',step+1,flush=True)
            for opt in opts:opt.zero_grad(set_to_none=True)
            with sdpa_kernel(SDPBackend.MATH):
                expected=cpu(ids,attention_mask=mask,labels=ids,use_cache=False).loss
                actual=device_call(model,x,attention_mask=m,labels=x,use_cache=False).loss
                actual.backward();expected.backward()
            close(actual,expected)
            for (name,p),(_,q) in zip(model.named_parameters(),cpu.named_parameters()):
                assert (p.grad is None)==(q.grad is None),name
                if p.grad is not None:
                    try:close(p.grad,q.grad)
                    except AssertionError as error:raise AssertionError('Gradient mismatch: '+name) from error
            for opt in opts:opt.step()
            for p,q in zip(model.parameters(),cpu.parameters()):close(p,q)
            losses.append(actual.item())
            state=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opts[0].state_dict()))
            if step==0:checkpoint=state
            else:final=state
        restored=migrate_to_npu(hf_model(kind));restored.load_state_dict(checkpoint['model'])
        print('training: checkpoint replay',flush=True)
        opt=torch.optim.SGD(restored.parameters(),lr=.03,momentum=.5,foreach=False,fused=False);opt.load_state_dict(checkpoint['optimizer'])
        with sdpa_kernel(SDPBackend.MATH):loss=device_call(restored,x,attention_mask=m,labels=x,use_cache=False).loss
        loss.backward();opt.step();assert loss.item()==losses[-1]
        for name,value in cpu_tree(restored.state_dict()).items():torch.testing.assert_close(value,final['model'][name],rtol=0,atol=0)
        for key,value in cpu_tree(opt.state_dict())['state'].items():
            torch.testing.assert_close(value['momentum_buffer'],final['optimizer']['state'][key]['momentum_buffer'],rtol=0,atol=0)
        return dict(steps=2,losses=losses,all_gradients_and_parameters_match_cpu=True,cpu_checkpoint_replay_bitwise=True,
                    tied_parameters_preserved=True,parameters=sum(p.numel() for p in model.parameters()))
    def primitives():
        print('primitives: factories and layout gradients',flush=True)
        for args in ((5,), (7,-3,-2),(.25,2.,.25)):
            actual=torch.arange(*args,device='intel_npu');close(actual,torch.arange(*args))
        host=torch.arange(24.,dtype=dtype).view(2,3,4).requires_grad_();base=host.detach().to('intel_npu').requires_grad_()
        actual=-base[...,1::2];expected=-host[...,1::2];actual.sum().backward();expected.sum().backward()
        close(actual,expected);close(base.grad,host.grad)
        empty=torch.empty(0,device='intel_npu',dtype=torch.float16,requires_grad=True)
        base.grad=None;actual=torch.cat((empty,base),dim=-2);close(actual,host)
        actual.sum().backward();assert empty.grad is not None and empty.grad.numel()==0
        close(base.grad,torch.ones_like(host))
        for host,axis,target in ((torch.tensor([1<<62,1<<62,-1,7]),0,None),
                                 (torch.arange(24).view(2,3,4).transpose(0,2),1,torch.int32),
                                 (torch.tensor([[True,False,True]]),-1,None),
                                 (torch.empty((2,0),dtype=torch.int64),1,None)):
            device=(host.transpose(0,2).contiguous().to('intel_npu').transpose(0,2) if host.ndim==3 else host.to('intel_npu'))
            print('primitives: cumsum',tuple(host.shape),host.dtype,axis,target,flush=True)
            actual=torch.cumsum(device,axis,dtype=target)
            close(actual,torch.cumsum(host,axis,dtype=target))
        for a,b in ((torch.tensor([1<<62,(1<<62)+1,-3]),torch.tensor([(1<<62)+1])),
                    (torch.tensor([0.,float('nan'),float('inf')]),torch.tensor([0.,float('nan')])),
                    (torch.tensor([1,2]),torch.empty(0,dtype=torch.int64))):
            for inverse in (False,True):
                print('primitives: isin',a.dtype,inverse,flush=True)
                aa,bb=a.to('intel_npu'),b.to('intel_npu');out=torch.empty(a.shape,dtype=torch.bool,device='intel_npu')
                torch.isin(aa,bb,invert=inverse,out=out)
                torch.testing.assert_close(out.cpu(),torch.isin(a,b,invert=inverse),rtol=0,atol=0)
        host=torch.tensor([[1<<62,(1<<62)+1,-3]])
        print('primitives: integer padding',flush=True)
        close(torch.nn.functional.pad(host.to('intel_npu'),(1,2),value=-100),torch.nn.functional.pad(host,(1,2),value=-100))
        return dict(arange_factories=True,rotary_partial_neg_gradient=True,empty_cache_cat_gradient=True,
                    exact_integer_cumsum=True,isin_out_nan_and_int64_bits=True,integer_label_padding=True)
    def native():
        records=[]
        for shape in ((0,3,8),(2,0,8)):
            print('native: empty MHA and Encoder',shape,flush=True)
            for masked in (False,True):
                cpu=torch.nn.MultiheadAttention(8,2,batch_first=True,dtype=dtype).eval();model=copy.deepcopy(cpu).to('intel_npu')
                host=torch.empty(shape,dtype=dtype);x=host.to('intel_npu')
                mask=torch.zeros(shape[:2],dtype=torch.bool) if masked else None
                with torch.no_grad():
                    expected,ew=cpu(host,host,host,key_padding_mask=mask)
                    actual,aw=model(x,x,x,key_padding_mask=mask.to('intel_npu') if masked else None)
                close(actual,expected);assert aw is None and ew is None
                records.append(dict(shape=shape,masked=masked))
            for norm_first in (False,True):
                cpu=torch.nn.TransformerEncoderLayer(8,2,dim_feedforward=16,batch_first=True,dtype=dtype,norm_first=norm_first).eval()
                model=copy.deepcopy(cpu).to('intel_npu')
                with torch.no_grad():close(model(x),cpu(host))
        return dict(default_fastpath=torch.backends.mha.get_fastpath_enabled(),mha=records,pre_and_post_norm_encoder_empty=True)
    actions=dict(probe=probe,inference=inference,sampling=sampling,training=training,primitives=primitives,native=native)
    try:
        row.update(actions[case]());row['status']='blocked' if case=='probe' and not row['all_stages_passed'] else 'passed'
    except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
    print(kind,case,precision,row['status'],row.get('error','')[:250],flush=True)
    for entry in row.get('observations',[]):print(entry['stage'],entry['status'],entry.get('error','')[:250],flush=True)
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes'))
    root=Path(__file__).resolve().parent
    sources=('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_sequence.py',
             'torch_npu_compat_extended.py','torch_npu_vision_extended.py',
             'test_torch_npu_attention_migration.py','torch_npu_attention_migration_demo.py')
    if backend=='flash':sources+=('torch_npu_flash_attention.py','torch_npu_nested.py','npunlock/tests/fixtures/npu4000/resident/manifest.json')
    result=dict(status=row['status'] if healthy else 'failed',checks=[row],stats=before,shutdown=final,
                elapsed_seconds=time.perf_counter()-started,
                source_sha256={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources})
    target=root/('npunlock-probe/research/results/flash-nested' if backend=='flash' else 'npunlock-probe/research/results/attention-migration');target.mkdir(exist_ok=True)
    (target/f'{"hf-" if backend=="flash" else ""}{kind}-{case}-{precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status']=='failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('llama','gpt2'),default='llama')
    parser.add_argument('--case',choices=('probe','inference','sampling','training','primitives','native'),default='inference')
    parser.add_argument('--precision',choices=('float16','float32'),default='float32')
    parser.add_argument('--backend',choices=('math','flash'),default='math')
    args=parser.parse_args();raise SystemExit(execute(args.kind,args.case,args.precision,args.backend))

"""Scale the standard token classifier; compare gradients, updates and checkpoint replay."""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import time
import traceback
from torch_npu_standard_modules_demo import cpu_rng_reference
from torch_npu_transformer_demo import token_classifier,cpu_tree
from torch_npu_geometry import index_tile_width
from torch_npu_sustained_train import process_memory

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'npunlock-probe/research/results/transformer-scale'
PROFILES={
    'small':dict(vocabulary=512,width=32,heads=4,length=8,layers=2,feedforward=128),
    'medium':dict(vocabulary=2048,width=64,heads=4,length=16,layers=2,feedforward=256),
    'long':dict(vocabulary=256,width=32,heads=4,length=96,layers=2,feedforward=128),
}


def execute(profile,precision,probe=False,batch=2):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1);torch.manual_seed(227)
    dtype=getattr(torch,precision);config=PROFILES[profile]
    result=dict(status='running',profile=profile,precision=precision,config=config,batch=batch,steps=[])
    start=time.perf_counter();initial_fastpath=torch.backends.mha.get_fastpath_enabled()
    def progress(phase):
        result['phase']=phase;print(phase,f'{time.perf_counter()-start:.3f}s',flush=True)
    def close(a,b):
        torch.testing.assert_close(a.detach().cpu(),b.detach(),rtol=.003 if dtype==torch.float16 else 3e-5,
                                   atol=.0005 if dtype==torch.float16 else 3e-6)
    def optimizer(model):return torch.optim.SGD(model.parameters(),lr=.02,momentum=.5,foreach=False,fused=False)
    def same_tree(actual,expected):
        if isinstance(expected,torch.Tensor):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        elif isinstance(expected,dict):
            assert actual.keys()==expected.keys()
            for key in expected:same_tree(actual[key],expected[key])
        elif isinstance(expected,(tuple,list)):
            assert type(actual)==type(expected) and len(actual)==len(expected)
            for a,b in zip(actual,expected):same_tree(a,b)
        else:assert actual==expected
    def workload():
        cpu=token_classifier(dtype,**config);model=copy.deepcopy(cpu).to('intel_npu')
        result['parameters']=sum(p.numel() for p in model.parameters())
        result['parameter_bytes']=sum(p.numel()*p.element_size() for p in model.parameters())
        host=torch.randint(1,config['vocabulary'],(batch,config['length']))
        host[:,-2:]=0;host[0,1]=host[0,0];host[1,0]=config['vocabulary']-1
        target=torch.tensor([0,2]*(batch//2));x=host.to('intel_npu');labels=target.to('intel_npu')
        progress('uploaded')
        if probe:
            progress('probe-forward');model(x)
        else:
            reference,opt=optimizer(cpu),optimizer(model);checkpoint=io.BytesIO()
            for step in range(2):
                reference.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
                with torch.backends.mkldnn.flags(enabled=False),cpu_rng_reference(torch.intel_npu.get_rng_state()) as tracker:
                    expected_logits=cpu(host);expected=torch.nn.functional.cross_entropy(expected_logits.float(),target)
                    expected.backward()
                cpu_rng=torch.get_rng_state();before=npu.stats();begin=time.perf_counter();progress(f'train-{step+1}')
                logits=model(x);loss=torch.nn.functional.cross_entropy(logits.float(),labels)
                progress(f'backward-{step+1}');loss.backward()
                elapsed=time.perf_counter()-begin
                computed=npu.stats();tiles=(config['width']+index_tile_width(config['vocabulary'],config['width'],host.numel())-1)//index_tile_width(config['vocabulary'],config['width'],host.numel())
                downloads=computed['host_download_copies']-before['host_download_copies']
                assert downloads==2*tiles+2,(downloads,tiles)
                sites=(batch*config['heads']*config['length']**2,batch*config['length']*config['width'],
                       batch*config['length']*config['feedforward'],batch*config['length']*config['width'])
                headers=config['layers']*sum((n+8191)//8192 for n in sites)
                assert computed['host_upload_copies']-before['host_upload_copies']==computed['cached_scalar_buffers']-before['cached_scalar_buffers']+headers
                assert computed['host_device_copies']==before['host_device_copies']
                assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(tracker.get_state(),torch.intel_npu.get_rng_state())
                close(logits,expected_logits);close(loss,expected)
                for (name,a),(_,b) in zip(model.named_parameters(),cpu.named_parameters()):
                    try:close(a.grad,b.grad)
                    except AssertionError as error:raise AssertionError(f'gradient {name}: {error}') from error
                reference.step();opt.step()
                for a,b in zip(model.parameters(),cpu.parameters()):close(a,b)
                assert not torch.count_nonzero(model.embedding.weight.grad.cpu()[0])
                after=npu.stats();row=dict(step=step+1,loss=loss.item(),forward_backward_seconds=elapsed,
                    npu_calls=after['npu_calls']-before['npu_calls'],sessions=after['cached_sessions'],
                    compute_host_downloads=downloads,embedding_column_tiles=tiles,rng_metadata_headers=headers)
                result['steps'].append(row);print(row,flush=True)
                state=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=torch.intel_npu.get_rng_state())
                if step==0:torch.save(state,checkpoint)
                else:second=state;second_loss=loss.item()
            progress('checkpoint-replay');checkpoint.seek(0);saved=torch.load(checkpoint,weights_only=True)
            restored=token_classifier(dtype,**config).to('intel_npu');restored.load_state_dict(saved['model'])
            restored_opt=optimizer(restored);restored_opt.load_state_dict(saved['optimizer']);torch.intel_npu.set_rng_state(saved['rng'])
            restored_opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(restored(x).float(),labels);loss.backward();restored_opt.step()
            assert loss.item()==second_loss
            same_tree(dict(model=cpu_tree(restored.state_dict()),optimizer=cpu_tree(restored_opt.state_dict()),rng=torch.intel_npu.get_rng_state()),second)
            result['checkpoint_bitwise']=True
            progress('eval');cpu.load_state_dict(second['model']);cpu.eval();restored.eval();rng=torch.intel_npu.get_rng_state()
            with torch.no_grad():close(restored(x),cpu(host))
            assert torch.equal(rng,torch.intel_npu.get_rng_state());result['eval_cpu_reference']=True
        result['status']='passed'
    try:workload()
    except Exception as error:result.update(status='failed',error=str(error),traceback=traceback.format_exc())
    finally:
        gc.collect();result['before_shutdown']=npu.stats();result['process_memory']=process_memory()
        npu.shutdown();result['final_stats']=npu.stats()
        final=result['final_stats']
        if final['context_failed'] or not final['closed'] or any(final[key] for key in (
                'live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions')):
            result.update(status='failed',lifecycle_error='Unhealthy final device state')
        assert initial_fastpath==torch.backends.mha.get_fastpath_enabled()
        result['elapsed_seconds']=time.perf_counter()-start
        sources=('torch_intel_npu.py','torch_npu_geometry.py','torch_npu_transformer_demo.py',
                 'torch_npu_transformer_scale.py','npunlock/tests/fixtures/npu4000/resident_index.c',
                 'npunlock/tests/fixtures/npu4000/resident/manifest.json')
        result['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=PROFILES,default='small')
    parser.add_argument('--precision',choices=('float16','float32'),default='float32')
    parser.add_argument('--batch',type=int,choices=(2,4),default=2)
    parser.add_argument('--probe',action='store_true');parser.add_argument('--label',default='after');args=parser.parse_args()
    result=execute(args.profile,args.precision,args.probe,args.batch);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{args.label}-{args.profile}-b{args.batch}-{args.precision}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result['status'],result.get('error','')[:800],flush=True)
    return int(result['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

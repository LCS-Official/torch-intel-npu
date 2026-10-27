"""Strict mixed LayerNorm replay, broader precision checks and small-network training."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/layernorm-mixed'


def execute(case, fp32_loss=False):
    import torch
    import torch_intel_npu as npu
    from layernorm_mixed_diagnose import sample
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.manual_seed(127)
    checks = []

    def close(a,b):
        a,b=a.detach().cpu(),b.detach().cpu()
        torch.testing.assert_close(a,b,rtol=.003 if a.dtype==torch.float16 else 3e-5,
                                   atol=.0005 if a.dtype==torch.float16 else 3e-6,equal_nan=True)

    def check(name, function):
        before=npu.stats()
        row={'name':name}
        try:
            row.update(function() or {})
            row['status']='passed'
        except Exception as error:
            row.update(status='failed',error=str(error),traceback=traceback.format_exc())
        gc.collect()
        after=npu.stats()
        row['npu_calls']=after['npu_calls']-before['npu_calls']
        row['live_buffer_delta']=after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
        if row['live_buffer_delta'] or after['context_failed']:
            row.update(status='failed',lifecycle_error='Buffer leak or context failure')
        checks.append(row)
        print(name,row['status'],row.get('error','')[:260],flush=True)

    def compare(host, shape, w, b, seed, eps=1e-5):
        host=host.detach().requires_grad_()
        w=w.detach().requires_grad_()
        b=b.detach().requires_grad_() if b is not None else None
        x,wn,bn=(v.detach().to('intel_npu').requires_grad_() if v is not None else None for v in (host,w,b))
        expected=torch.native_layer_norm(host,shape,w,b,eps)
        actual=torch.native_layer_norm(x,shape,wn,bn,eps)
        for a,e in zip(actual,expected): close(a,e)
        for a,e in zip(actual[1:],expected[1:]):
            assert torch.equal(a.cpu().view(torch.int32),e.view(torch.int32)), 'Statistics differ in bits'
        expected[0].backward(seed)
        actual[0].backward(seed.to('intel_npu'))
        errors=[]
        for a,e in zip((x,wn,bn),(host,w,b)):
            if a is not None:
                close(a.grad,e.grad)
                errors.append(float((a.grad.cpu()-e.grad).abs().max()))
        return errors

    def replay():
        host,w,b,g=sample()
        errors=compare(host,(4096,),w,b,g)
        return {'gradient_max_abs':errors,'input_sha256':hashlib.sha256(host.numpy().tobytes()+g.numpy().tobytes()).hexdigest()}

    def stress(width,rows,seed):
        torch.manual_seed(seed)
        shape=(rows,width)
        x=torch.randn(shape,dtype=torch.float16)
        w=torch.randn(width,dtype=torch.float32)
        b=torch.randn(width,dtype=torch.float32) if seed % 2 else None
        g=torch.randn(shape,dtype=torch.float16)/4
        return {'shape':list(shape),'seed':seed,'gradient_max_abs':compare(x,(width,),w,b,g)}

    def training(mode):
        host=torch.nn.Sequential(torch.nn.Linear(16,16),torch.nn.LayerNorm(16),torch.nn.GELU(approximate=mode),torch.nn.Linear(16,3)).half()
        host[1].float()
        model=copy.deepcopy(host).to('intel_npu')
        reference,actual=(torch.optim.SGD(v.parameters(),lr=.01,momentum=.8,foreach=False,fused=False) for v in (host,model))
        data=torch.randn(4,16,dtype=torch.float16)
        labels=torch.tensor([0,1,2,1])
        x,target=data.to('intel_npu'),labels.to('intel_npu')
        losses=[]
        for step in range(4):
            reference.zero_grad(set_to_none=True); actual.zero_grad(set_to_none=True)
            cpu_logits, logits=host(data), model(x)
            expected=torch.nn.functional.cross_entropy(cpu_logits.float() if fp32_loss else cpu_logits,labels)
            result=torch.nn.functional.cross_entropy(logits.float() if fp32_loss else logits,target)
            close(result,expected)
            expected.backward(); result.backward()
            for (name,a),(_,e) in zip(model.named_parameters(),host.named_parameters()):
                try: close(a.grad,e.grad)
                except AssertionError as error:
                    raise AssertionError(f'step={step}, parameter={name}, fp32_loss={fp32_loss}: {error}') from error
            reference.step(); actual.step()
            for a,e in zip(model.parameters(),host.parameters()): close(a,e)
            losses.append(float(result.detach().cpu()))
        assert losses[-1]<losses[0]
        return {'steps':4,'losses':losses,'layernorm_affine_fp32_rest_fp16':True,'fp32_loss':fp32_loss}

    def contracts():
        x=torch.ones(1,65537,dtype=torch.float16,device='intel_npu')
        w=torch.ones(65537,device='intel_npu')
        before=npu.stats()['npu_calls']
        try: npu._layer_norm(x,(65537,),w,None)
        except ValueError as error: assert 'normalized width' in str(error)
        else: raise AssertionError('Wide mixed LayerNorm accepted')
        assert npu.stats()['npu_calls']==before
        return {'wide_mixed_rejected_before_submit':True}

    if case=='replay': check('original-4096-replay',replay)
    elif case=='stress':
        for width,rows in ((2,3),(7,3),(15,3),(16,3),(17,3),(33,3),(257,3),(576,3),(4096,1),(4096,17),(65536,1)):
            for seed in (127,128): check(f'mixed-{rows}x{width}-seed{seed}',lambda w=width,r=rows,s=seed:stress(w,r,s))
    elif case=='training':
        for mode in ('none','tanh'): check(f'mixed-training-{mode}',lambda m=mode:training(m))
    else: check('mixed-width-contract',contracts)
    gc.collect()
    npu.shutdown()
    final=npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    return {'case':case,'fp32_loss':fp32_loss,'status':'failed' if any(c['status']!='passed' for c in checks) else 'passed',
        'checks':checks,'final_stats':final,'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
         ('torch_intel_npu.py','torch_npu_autograd.py','npunlock-probe/research/layernorm_mixed_acceptance.py',
          'npunlock-probe/research/layernorm_mixed_diagnose.py','npunlock/tests/fixtures/npu4000/resident_layernorm.c',
          'npunlock/tests/fixtures/npu4000/resident/manifest.json')}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('replay','stress','training','contracts'),required=True)
    parser.add_argument('--fp32-loss',action='store_true',help='Explicit FP32 CrossEntropy; preserves the original FP16-loss comparison')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    result=execute(args.case,args.fp32_loss)
    suffix='-fp32-loss' if args.fp32_loss else ''
    (OUT/f'{args.case}{suffix}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(result['status']!='passed')

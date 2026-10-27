"""Bounded AdamW pointwise, optimizer-state and rejection checks on intel_npu."""
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/adamw'


def execute(case):
    import numpy as np
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(61)
    checks = []

    def close(a, b):
        torch.testing.assert_close(a.detach().cpu(), b.detach().cpu(),
            rtol=.003 if a.dtype == torch.float16 else 3e-5,
            atol=.0005 if a.dtype == torch.float16 else 3e-6, equal_nan=True)

    def check(name, function):
        before = npu.stats()
        row = {'name':name}
        try:
            row.update(function() or {})
            row['status'] = 'passed'
        except Exception as error:
            row.update(status='failed', error=str(error), traceback=traceback.format_exc())
        gc.collect()
        after = npu.stats()
        row['npu_calls'] = after['npu_calls']-before['npu_calls']
        row['live_buffer_delta'] = after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
        if row['live_buffer_delta'] or after['context_failed']:
            row.update(status='failed', lifecycle_error='Live storage leak or failed context')
        checks.append(row)
        print(name, row['status'], row.get('error','')[:350], flush=True)

    def primitives(dtype):
        comparisons = 0
        with torch.no_grad():
            for transposed in (False, True):
                original = torch.tensor([[.25,-.5,1.],[-.125,.75,-1.5]],dtype=dtype)
                a = torch.tensor([[.5,-.75,.125],[.2,-.4,.6]],dtype=dtype)
                b = torch.tensor([[1.,2.,-.5],[.3,-.8,1.1]],dtype=dtype)
                if transposed:
                    original, a, b = (t.t() for t in (original,a,b))
                for value in (0.,.001,.1,.5,.9,1.,-2.,2.,float('inf'),float('nan')):
                    for operation in ('lerp','addcmul','addcdiv'):
                        host = original.clone()
                        data = host.contiguous().to('intel_npu')
                        if transposed:
                            data = host.t().contiguous().to('intel_npu').t()
                        left, right = (t.contiguous().to('intel_npu') for t in (a,b))
                        version = data._version
                        if operation == 'lerp':
                            host.lerp_(a,value)
                            result = data.lerp_(left,value)
                        else:
                            getattr(host,operation+'_')(a,b,value=value)
                            result = getattr(data,operation+'_')(left,right,value=value)
                        assert result is data and data._version == version+1
                        close(data,host)
                        comparisons += 1
                host, data = original.clone(), original.contiguous().to('intel_npu')
                host.add_(1e-8)
                assert data.add_(1e-8) is data
                close(data,host)
                for operation in ('lerp','addcmul','addcdiv'):
                    host, data = original.clone(), original.contiguous().to('intel_npu')
                    if operation == 'lerp':
                        host.lerp_(host,.2)
                        data.lerp_(data,.2)
                    else:
                        getattr(host,operation+'_')(host,host,value=.2)
                        getattr(data,operation+'_')(data,data,value=.2)
                    close(data,host)
                comparisons += 4
            extreme = torch.tensor([-torch.finfo(dtype).max,1.,torch.finfo(dtype).max],dtype=dtype)
            endpoint = extreme.flip(0).contiguous()
            for weight in (.49999999,.50000001,.5,-.49999999):
                host, data = extreme.clone(), extreme.to('intel_npu')
                host.lerp_(endpoint,weight)
                data.lerp_(endpoint.to('intel_npu'),weight)
                close(data,host)
                comparisons += 1
            if dtype == torch.float16:
                values = torch.from_numpy(np.arange(65536,dtype=np.uint16).view(np.float16).copy())
            else:
                words = np.random.default_rng(61).integers(0,2**32,8192,dtype=np.uint32)
                words[:16] = [0,0x80000000,1,0x80000001,0x7fffff,0x800000,0x3f800000,0x40800000,
                    0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,0xffc00001,0x00800001,0x80800001]
                values = torch.from_numpy(words.view(np.float32).copy())
            data = values.to('intel_npu')
            cpu_sqrt, actual = values.sqrt(), data.sqrt().cpu()
            with np.errstate(invalid='ignore'):
                expected = torch.from_numpy(np.sqrt(values.numpy().astype(np.float64)).astype(
                    np.float16 if dtype == torch.float16 else np.float32))
            torch.testing.assert_close(actual,expected,rtol=0,atol=0,equal_nan=True)
            finite = torch.isfinite(expected)
            assert np.array_equal(actual.numpy()[finite].view(np.uint16 if dtype == torch.float16 else np.uint32),
                                  expected.numpy()[finite].view(np.uint16 if dtype == torch.float16 else np.uint32))
            integer_dtype = torch.int16 if dtype == torch.float16 else torch.int32
            ulps = (cpu_sqrt[finite].view(integer_dtype).long()-expected[finite].view(integer_dtype).long()).abs()
            assert int(ulps.max()) <= 1
            out = torch.empty_like(data)
            assert torch.sqrt(data,out=out) is out
            close(out,expected)
            other = values.flip(0).contiguous()
            uploaded = other.to('intel_npu')
            close(torch.maximum(data,uploaded),torch.maximum(values,other))
            assert torch.maximum(data,uploaded,out=data) is data
            close(data,torch.maximum(values,other))
            big = torch.full((65537,), .25, dtype=dtype).to('intel_npu')
            close(big.sqrt(),torch.full((65537,),.5,dtype=dtype))
            expected = torch.tensor([float('nan'),float('inf'),-float('inf'),0.,-0.,1.],dtype=dtype)
            a = expected.to('intel_npu')
            b = torch.tensor([1.,float('inf'),1.,1.,1.,0.],dtype=dtype).to('intel_npu')
            host = expected.clone()
            host.addcdiv_(expected,b.cpu(),value=0.)
            a.addcdiv_(a,b,value=0.)
            close(a,host)
        return {'dtype':str(dtype),'pointwise_comparisons':comparisons+6,
                'sqrt_bit_patterns':values.numel(),'sqrt_finite_bits_exact':True,'tiled_65537':True,
                'cpu_sqrt_ulp_differences':int((ulps != 0).sum()), 'cpu_sqrt_max_ulp':int(ulps.max())}

    def state_equal(param, host, opt, reference):
        close(param,host)
        assert opt.state[param].keys() == reference.state[host].keys()
        for key,value in opt.state[param].items():
            close(value,reference.state[host][key])
            assert value.device.type == ('cpu' if key == 'step' else 'intel_npu')

    def optimizer(dtype, options):
        host = torch.nn.Parameter(torch.tensor([[.2,-.4,.6],[.8,-1.,1.2]],dtype=dtype).t())
        param = torch.nn.Parameter(host.detach().t().contiguous().to('intel_npu').t())
        untouched = torch.nn.Parameter(torch.ones(1,dtype=dtype).to('intel_npu'))
        settings = dict(lr=.002, eps=1e-4 if dtype == torch.float16 else 1e-8,foreach=False,fused=False)
        settings.update(options)
        reference = torch.optim.AdamW([host],**settings)
        opt = torch.optim.AdamW([param,untouched],**settings)
        copies, uploads = [], []
        for index in range(5):
            gradient = torch.tensor([[.5,-.3,.2],[.4,-.6,.8]],dtype=dtype).t()*(1.+index/8)
            if index == 1:
                gradient.zero_()
            elif index == 2:
                host.grad = param.grad = None
                snapshot = param.detach().cpu().clone()
                step = opt.state[param]['step'].item()
                before = npu.stats()['npu_calls']
                reference.step()
                opt.step()
                assert npu.stats()['npu_calls'] == before and opt.state[param]['step'].item() == step
                close(param,snapshot)
                continue
            host.grad, param.grad = gradient, gradient.t().contiguous().to('intel_npu').t()
            reference.step()
            before = npu.stats()
            opt.step()
            after = npu.stats()
            copied = sum(after[k]-before[k] for k in ('host_upload_copies','host_download_copies','host_device_copies'))
            scalars = after['cached_scalar_buffers']-before['cached_scalar_buffers']
            assert copied == scalars, (copied,scalars)
            copies.append(copied)
            uploads.append(scalars)
            state_equal(param,host,opt,reference)
            assert param.stride() == host.stride() and untouched not in opt.state
            close(untouched,torch.ones(1,dtype=dtype))
            if index == 3:
                saved = opt.state_dict()
                saved['state'] = {k:{name:v.detach().cpu() for name,v in state.items()} for k,state in saved['state'].items()}
                stream = io.BytesIO()
                torch.save(saved,stream)
                stream.seek(0)
                saved = torch.load(stream,map_location='cpu',weights_only=True)
                restored = torch.nn.Parameter(param.detach().cpu().t().contiguous().to('intel_npu').t())
                restored_opt = torch.optim.AdamW([restored,untouched],**settings)
                restored_opt.load_state_dict(saved)
                param,opt = restored,restored_opt
        return {'dtype':str(dtype),'options':settings,'step_host_copies':copies,'scalar_uploads':uploads,
                'state_and_dense_transpose':True,'zero_and_none_gradients':True,'checkpoint_resume':True}

    def contracts():
        x = torch.ones(2,2).to('intel_npu')
        other = torch.ones_like(x)
        small = torch.ones(1).to('intel_npu')
        incompatible = torch.ones(3).to('intel_npu')
        half = x.half()
        partial = x.view(-1)[1:]
        actions = {
            'lerp-broadcast':lambda:x.lerp_(small,.1),
            'lerp-dtype':lambda:x.lerp_(half,.1),
            'lerp-cpu':lambda:x.lerp_(torch.ones(2,2),.1),
            'lerp-bool':lambda:x.lerp_(other,True),
            'lerp-tensor-weight':lambda:x.lerp_(other,small),
            'lerp-partial':lambda:partial.lerp_(partial,.1),
            'lerp-overlap':lambda:x.lerp_(x.t(),.1),
            'addcmul-broadcast':lambda:x.addcmul_(small,other),
            'addcmul-coefficient':lambda:x.addcmul_(other,other,value=1e50),
            'addcdiv-overlap':lambda:x.addcdiv_(x.t(),other),
            'add-scalar-bool':lambda:x.add_(True),
            'maximum-broadcast':lambda:torch.maximum(x,incompatible),
            'maximum-overlap':lambda:torch.maximum(x,other,out=x.t()),
            'sqrt-output-alias':lambda:torch.sqrt(x,out=x),
        }
        rejected = []
        before = npu.stats()['npu_calls']
        with torch.no_grad():
            for name,function in actions.items():
                try:
                    function()
                except (ValueError,RuntimeError,NotImplementedError,TypeError):
                    rejected.append(name)
                else:
                    raise AssertionError(f'Accepted {name}')
        for name,function in {'lerp-grad-mode':lambda:x.lerp_(other,.1),
                'addcmul-grad-mode':lambda:x.addcmul_(other,other),
                'addcdiv-grad-mode':lambda:x.addcdiv_(other,other),
                'add-scalar-grad-mode':lambda:x.add_(.1)}.items():
            try:
                function()
            except (ValueError,RuntimeError,NotImplementedError):
                rejected.append(name)
            else:
                raise AssertionError(f'Accepted {name}')
        assert npu.stats()['npu_calls'] == before
        leaf = x.detach().requires_grad_()
        loss = (leaf*leaf).sum()
        with torch.no_grad():
            leaf.lerp_(other,.1)
        try:
            loss.backward()
        except RuntimeError as error:
            assert 'version' in str(error) or 'inplace' in str(error)
        else:
            raise AssertionError('Saved-tensor version guard bypassed')
        return {'rejected_without_graph_submit':rejected,'saved_tensor_version_guard':True}

    def half_default_epsilon():
        host = torch.nn.Parameter(torch.tensor([0.,1.],dtype=torch.float16))
        param = torch.nn.Parameter(host.detach().to('intel_npu'))
        host.grad, param.grad = torch.zeros_like(host), torch.zeros_like(param)
        reference, opt = (torch.optim.AdamW([p],foreach=False,fused=False) for p in (host,param))
        reference.step()
        opt.step()
        assert torch.isnan(host).all() and torch.isnan(param.detach().cpu()).all()
        close(param,host)
        return {'default_eps':1e-8,'cpu_and_npu_zero_gradient_nonfinite':True,
                'epsilon_not_silently_modified':True}

    if case == 'primitives':
        for dtype in (torch.float16,torch.float32):
            check(f'primitives-{dtype}',lambda d=dtype:primitives(d))
    elif case == 'optimizer':
        for name,options in (('default-betas',{}),('no-decay',{'weight_decay':0.}),
                ('betas',{'betas':(.1,.95)}),('maximize',{'maximize':True}),
                ('amsgrad',{'amsgrad':True}),('combined',{'amsgrad':True,'maximize':True,'weight_decay':.1})):
            for dtype in (torch.float16,torch.float32):
                check(f'adamw-{name}-{dtype}',lambda d=dtype,o=options:optimizer(d,o))
        check('fp16-default-epsilon-boundary',half_default_epsilon)
    else:
        check('update-contracts',contracts)
    gc.collect()
    before = npu.stats()
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    return {'case':case,'status':'failed' if any(c['status'] != 'passed' for c in checks) else 'passed',
        'checks':checks,'stats_before_shutdown':before,'final_stats':final,
        'reference_adam_sha256':hashlib.sha256(Path(sys.modules[torch.optim.Adam.__module__].__file__).read_bytes()).hexdigest(),
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('torch_intel_npu.py','torch_npu_autograd.py','torch_npu_adamw_demo.py',
             'npunlock/tests/fixtures/npu4000/resident_optimizer.c','npunlock/tests/fixtures/npu4000/resident/manifest.json',
             'npunlock-probe/torch-device/_torch_intel_npu.pyd')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('primitives','optimizer','contracts'),required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    result = execute(args.case)
    (OUT/f'{args.case}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    raise SystemExit(result['status'] != 'passed')

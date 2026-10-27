"""Standard GradScaler aliases, scale promotion/iterables, and Tensor control updates."""
import argparse
import gc
import hashlib
import json
import pickle
from pathlib import Path
import time
import traceback

import torch
from torch.amp import GradScaler as PreImportScaler
from torch.amp.grad_scaler import GradScaler as ModuleScaler
from torch import GradScaler as TopLevelScaler


def execute(case):
    import torch_intel_npu as npu
    from torch_npu_amp import GradScaler as LegacyScaler
    torch.set_num_threads(1);started=time.perf_counter()
    def close(actual,expected):
        a=actual.detach().cpu()
        torch.testing.assert_close(a,expected.detach(),rtol=0,atol=0,equal_nan=True)
        if a.dtype.is_floating_point:assert torch.equal(torch.signbit(a),torch.signbit(expected))
    def api():
        assert PreImportScaler is ModuleScaler is TopLevelScaler is torch.amp.GradScaler
        for factory in (PreImportScaler,ModuleScaler,TopLevelScaler):
            scaler=factory('intel_npu',128.,2.,.5,3)
            assert isinstance(scaler,PreImportScaler) and isinstance(scaler,LegacyScaler)
            assert scaler._scale is None
            loss=torch.ones((),device='intel_npu',requires_grad=True)
            scaled=scaler.scale(loss);scaled.backward()
            close(loss.grad,torch.tensor(128.))
            assert (scaler._scale.device.type,scaler._scale.dtype)==('cpu',torch.float32)
            assert (scaler._growth_tracker.device.type,scaler._growth_tracker.dtype)==('cpu',torch.int32)
            scaler.update(64.)
            restored=factory(device='intel_npu');restored.load_state_dict(scaler.state_dict())
            assert restored.state_dict()==scaler.state_dict()
            assert pickle.loads(pickle.dumps(scaler)).state_dict()==scaler.state_dict()
        assert type(PreImportScaler('cpu')) is PreImportScaler
        assert type(PreImportScaler('xpu',enabled=False)) is PreImportScaler
        disabled=PreImportScaler('intel_npu',enabled=False)
        value=torch.tensor(2.,requires_grad=True)
        assert disabled.scale(value) is value and disabled.state_dict()=={}
        optimizer=torch.optim.SGD([value],lr=.1)
        disabled.scale(value.square()).backward();disabled.step(optimizer);disabled.update()
        torch.testing.assert_close(value,torch.tensor(1.6))
        return dict(preimport_module_and_top_level_aliases=True,class_identity_preserved=True,
                    cpu_scaler_type_unchanged=True,fp32_int32_cpu_controls=True,
                    scaler_pickle_and_state_dict=True,disabled_cpu_workflow=True)
    def outputs():
        records=[]
        for scale in (8.,65536.):
            for dtype in (torch.float16,torch.float32):
                for shape in ((),(4,),(0,)):
                    host=(torch.tensor(.125,dtype=dtype) if not shape else
                          torch.tensor([0.,-.0,.125,-.25],dtype=dtype) if shape==(4,) else torch.empty(0,dtype=dtype))
                    host.requires_grad_();device=host.detach().to('intel_npu').requires_grad_()
                    scaler=PreImportScaler('intel_npu',init_scale=scale)
                    reference=PreImportScaler('cpu',init_scale=scale)
                    cpu_result=reference.scale(host);before=npu.stats();actual=scaler.scale(device)
                    assert actual.dtype==cpu_result.dtype
                    actual.backward(torch.ones_like(actual));cpu_result.backward(torch.ones_like(cpu_result))
                    assert npu.stats()['host_download_copies']==before['host_download_copies']
                    close(actual,cpu_result);close(device.grad,host.grad)
                    records.append(dict(scale=scale,dtype=str(dtype),shape=list(shape),result_dtype=str(actual.dtype)))
        host=[torch.tensor(.25,requires_grad=True),torch.tensor([.125,-.25],requires_grad=True)]
        device=[v.detach().to('intel_npu').requires_grad_() for v in host]
        scaler=PreImportScaler('intel_npu',init_scale=8.)
        nested=scaler.scale([device[0],(device[1],[])])
        assert isinstance(nested,list) and isinstance(nested[1],tuple) and nested[1][1]==[]
        (nested[0]+nested[1][0].sum()).backward()
        for leaf in device:close(leaf.grad,torch.full(leaf.shape,8.))
        lazy=PreImportScaler('intel_npu',init_scale=16.)
        mapped=lazy.scale(iter([v.detach() for v in device]));assert lazy._scale is None
        for actual,expected in zip(mapped,host):close(actual,expected.detach()*16.)
        empty=PreImportScaler('intel_npu');assert empty.scale([])==[] and empty._scale is None
        return dict(cases=records,nested_lists_tuples=True,lazy_generator=True,empty_iterable=True,
                    matches_native_cpu_result_dtype_values_signs_and_gradients=True)
    def updates():
        scaler=PreImportScaler('intel_npu',init_scale=8.);scaler.scale(torch.ones((),device='intel_npu'))
        for shape in ((),(1,)):
            value=torch.full(shape,128.,dtype=torch.float32,device='intel_npu')
            before=npu.stats();scaler.update(value)
            assert npu.stats()['host_download_copies']-before['host_download_copies']==1
            assert scaler.get_scale()==128.
            value.fill_(256.);assert scaler.get_scale()==128.
        bad_values=[torch.ones((),device='cpu'),torch.ones((),device='intel_npu',dtype=torch.float16),
                    torch.ones(2,device='intel_npu'),torch.ones((),device='intel_npu',requires_grad=True)]
        for value in bad_values:
            saved=scaler.state_dict();before=npu.stats()
            try:scaler.update(value)
            except ValueError:pass
            else:raise AssertionError('Invalid new_scale metadata accepted')
            assert scaler.state_dict()==saved and npu.stats()['npu_calls']==before['npu_calls']
        for bad in (0.,-1.,float('inf'),float('nan')):
            value=torch.tensor(bad,device='intel_npu');saved=scaler.state_dict()
            try:scaler.update(value)
            except ValueError:pass
            else:raise AssertionError('Invalid new_scale value accepted')
            assert scaler.state_dict()==saved
        scaler.update(64.);saved=scaler.state_dict()
        uninitialized=PreImportScaler('intel_npu');value=torch.ones((),device='intel_npu');before=npu.stats()
        try:uninitialized.update(value)
        except AssertionError:pass
        else:raise AssertionError('Update before scale accepted')
        assert scaler.state_dict()==saved and npu.stats()['npu_calls']==before['npu_calls']
        return dict(device_scalar_and_one_element=True,control_copied_without_alias=True,
                    metadata_rejections=4,data_rejections=4,state_unchanged_on_rejection=True,
                    explicit_python_scale=True,update_before_initialization_rejected=True)
    def adamw():
        p=torch.nn.Parameter(torch.tensor([.5,-1.,2.,.25],device='intel_npu'))
        q=torch.nn.Parameter(p.detach().cpu())
        opts=[torch.optim.AdamW([t],lr=.02,weight_decay=.05,amsgrad=True,foreach=False,fused=False) for t in (p,q)]
        scalers=[PreImportScaler('intel_npu',init_scale=8.,growth_interval=2),PreImportScaler('cpu',init_scale=8.,growth_interval=2)]
        history=[]
        for special in (None,None,float('inf'),None,float('nan'),None,None):
            for scaler,t in zip(scalers,(p,q)):scaler.scale(t.sum())
            grad=torch.tensor([2.,-4.,.25,0.])*scalers[0].get_scale()
            if special is not None:grad[1]=special
            p.grad=grad.to('intel_npu');q.grad=grad.clone()
            before=npu.stats();scalers[0].step(opts[0]);scalers[0].update()
            assert npu.stats()['host_download_copies']-before['host_download_copies']==1
            scalers[1].step(opts[1]);scalers[1].update()
            assert scalers[0].state_dict()==scalers[1].state_dict()
            torch.testing.assert_close(p.detach().cpu(),q.detach(),rtol=3e-5,atol=3e-6)
            for name,value in opts[0].state[p].items():
                actual=value.cpu() if value.device.type=='intel_npu' else value
                torch.testing.assert_close(actual,opts[1].state[q][name],rtol=3e-5,atol=3e-6)
            history.append(scalers[0].get_scale())
        assert history==[8.,16.,8.,8.,4.,4.,8.]
        return dict(scale_history=history,amsgrad_and_weight_decay=True,
                    parameters_and_all_adamw_states_match_cpu=True,overflow_does_not_advance_optimizer=True)
    row=dict(name=case)
    try:row.update(dict(api=api,outputs=outputs,updates=updates,adamw=adamw)[case]());row['status']='passed'
    except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
    print(case,row['status'],row.get('error','')[:450],flush=True)
    gc.collect();before=npu.stats();npu.shutdown();gc.collect();final=npu.stats()
    healthy=final['closed'] and not final['context_failed'] and all(final[k]==0 for k in (
        'live_buffers','cached_buffer_blocks','cached_scalar_buffers','cached_sessions','quarantined_buffer_bytes'))
    root=Path(__file__).resolve().parent
    sources=('torch_intel_npu.py','torch_npu_amp.py','torch_npu_amp_standard_demo.py','test_torch_npu_amp.py')
    result=dict(status='passed' if healthy and row['status']=='passed' else 'failed',case=case,
                checks=[row],stats=before,shutdown=final,elapsed_seconds=time.perf_counter()-started,
                source_sha256={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources})
    target=root/'npunlock-probe/research/results/amp-standard';target.mkdir(exist_ok=True)
    (target/f'{case}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return int(result['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=('api','outputs','updates','adamw'),required=True)
    args=parser.parse_args();raise SystemExit(execute(args.case))

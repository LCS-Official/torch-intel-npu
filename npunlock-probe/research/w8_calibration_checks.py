"""Prepared-weight API guards and calibrated generic Linear on real NPU."""
import gc
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from w8_calibration import OUT,SOURCES,load_weights
from w8_quality_probe import ROOT,MODEL,sha
from torch_npu_smollm_stack_demo import errors


def main():
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolLayer,FrozenSmolStack
    from torch_npu_lowbit import calibrate_int8,quantize_weight,QuantizedLinear,dpu_modes
    torch.set_num_threads(8);report=dict(status='running',source_sha256=sha(Path(__file__)),
        sources={name:sha(ROOT/name) for name in SOURCES},checks=[])
    def work():
        prepared=load_weights();config=json.loads((MODEL/'config.json').read_text())
        seed=FrozenSmolLayer.from_checkpoint(MODEL);weights={name:seed._cpu_weight(name) for name in seed._weights}
        def reject(name,fn):
            before=npu.stats()
            try:fn()
            except ValueError as error:
                after=npu.stats();assert all(before[key]==after[key] for key in ('npu_calls','session_creations','host_upload_copies','driver_allocations'))
                report['checks'].append(dict(name=name,status='rejected_before_device_execution',reason=str(error)))
            else:raise AssertionError('Missing rejection: '+name)
        for name,bad in [('empty',{}),('partial',{i:prepared[i] for i in (0,1,2)}),('extra',dict(prepared,**{'4':prepared[0]})),
            ('bool key',{False:prepared[0],**{i:prepared[i] for i in (1,2,3)}}),('none layer',{**prepared,3:None})]:
            reject('stack '+name,lambda bad=bad:FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,packed_weights=bad))
        reject('stack fp16',lambda:FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,mode='fp16',packed_weights=prepared))
        reject('stack wrong range',lambda:FrozenSmolStack.from_checkpoint(MODEL,start_layer=1,num_layers=2,packed_weights=prepared))
        name='self_attn.q_proj.weight'
        for label,bad in [('empty',{}),('partial',{key:value for key,value in prepared[0].items() if key!=name}),
            ('nonpacked',{**prepared[0],name:torch.zeros(576,576)}),
            ('shape',{**prepared[0],name:quantize_weight(torch.ones(576,575),format='int8')}),
            ('format',{**prepared[0],name:quantize_weight(torch.ones(576,576),format='int4')}),
            ('grouped',{**prepared[0],name:quantize_weight(torch.ones(576,576),format='int8',group_size=64)})]:
            reject('layer '+label,lambda bad=bad:FrozenSmolLayer(config,weights,packed_weights=bad))
        reject('layer fp16',lambda:FrozenSmolLayer(config,weights,mode='fp16',packed_weights=prepared[0]))
        supplied={i:dict(value) for i,value in prepared.items()}
        model=FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,packed_weights=supplied)
        snapshot=model.layers[0]._packed[name];supplied[0].clear();supplied.clear()
        assert model.layers[0]._packed[name] is snapshot
        ranged=FrozenSmolStack.from_checkpoint(MODEL,start_layer=1,num_layers=2,packed_weights={i:prepared[i] for i in (1,2)})
        assert ranged.start_layer==1 and ranged.num_layers==2
        report['checks'].append(dict(name='snapshot ownership and checkpoint indices',status='passed'))
        torch.manual_seed(491);linear=torch.nn.Linear(64,16).half().eval()
        calibration=torch.randn(64,64).half();calibration[:,1]=calibration[:,0]
        q=calibrate_int8(linear.weight,calibration);module=QuantizedLinear(q,bias=linear.bias)
        host=torch.randn(3,64).half();x=host.to('intel_npu')
        c=np.frombuffer(q.data,dtype=np.int8).reshape(q.shape).astype(np.float64);s=np.frombuffer(q.scales,dtype=np.float16).astype(np.float64)
        reference=((host.double()@torch.from_numpy(c).t())*torch.from_numpy(s)[None]).half()
        reference=(reference+linear.bias.detach()).half()
        actual=module(x).cpu();error=errors(actual,reference)
        assert error['relative_l2']<.003 and error['max_scaled_error']<.01,error
        before=npu.stats();warm=module(x);after=npu.stats()
        assert after['npu_calls']-before['npu_calls']==1
        assert all(before[key]==after[key] for key in ('session_creations','host_upload_copies','host_download_copies'))
        assert torch.equal(warm.cpu().view(torch.int16),actual.view(torch.int16))
        modes=dpu_modes(npu._sessions[module._keys[3]][1].graph_blob)
        assert any(mode['amode']==0 and mode['wmode']==2 and not mode['elementwise'] for mode in modes)
        report['checks'].append(dict(name='public calibrate_int8 + QuantizedLinear',status='passed',execution=error,warm_graph_calls=1,modes=modes))
    try:
        with torch.no_grad():work()
        gc.collect();npu.shutdown();report['final_stats']=npu.stats()
        assert not report['final_stats']['context_failed'] and report['final_stats']['live_buffers']==report['final_stats']['cached_buffer_bytes']==0
        report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report['status'],len(report['checks']),flush=True)
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

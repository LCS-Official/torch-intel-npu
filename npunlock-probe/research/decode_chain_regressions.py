"""Run existing regressions into this change's result directory."""
import argparse
from dataclasses import replace
from decode_chain_api import OUT, MODEL, write, final, delta, arrays, equal


def guards():
    import torch
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    torch.set_num_threads(8)
    report=dict(status='running',checks=[])
    def reject(name,fn):
        before=npu.stats()
        try:fn()
        except ValueError as error:
            assert all(v==0 for v in delta(before,npu.stats()).values()),name
            report['checks'].append(dict(name=name,error=str(error)))
        else:raise AssertionError(('accepted',name))
    def work():
        stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=4,mode='fp16')
        host=torch.randn(1,8,576).half()
        x=host.to('intel_npu');one=host[:,:1].contiguous().to('intel_npu')
        _,seed=stack.prefill(x);_,second=stack.prefill(x)
        for group in (True,False,0,3,-1):reject('group_'+str(group),lambda g=group:stack.decode(one,seed,group_size=g))
        reject('missing_cache',lambda:stack.decode(one,None))
        reject('wrong_container',lambda:stack.decode(one,list(seed)))
        reject('missing_layer',lambda:stack.decode(one,seed[:-1]))
        reject('wrong_last_type',lambda:stack.decode(one,(*seed[:-1],None)))
        reject('mixed_snapshots',lambda:stack.decode(one,(*seed[:-1],second[-1])))
        reject('reordered_layers',lambda:stack.decode(one,tuple(reversed(seed))))
        reject('unequal_positions',lambda:stack.decode(one,(*seed[:-1],replace(seed[-1],length=7))))
        reject('full_cache',lambda:stack.decode(one,tuple(replace(c,length=64) for c in seed)))
        reject('multiple_tokens',lambda:stack.decode(x,seed))
        cpu=one.cpu();reject('CPU_input',lambda:stack.decode(cpu,seed))
        fp32=one.float();reject('FP32_input',lambda:stack.decode(fp32,seed))
        stack.layers[-1].train();reject('last_layer_training',lambda:stack.decode(one,seed));stack.eval()
        stack.layers[-1]._whole_layer=False
        reject('unfused_explicit_group',lambda:stack.decode(one,seed,group_size=4))
        stack.layers[-1]._whole_layer=True
        grad=one.detach().requires_grad_()
        with torch.enable_grad():reject('active_autograd',lambda:stack.decode(grad,seed))
        with torch.inference_mode():
            value,updated=stack.decode(one,seed)
            assert torch.isfinite(value.cpu()).all() and all(c.length==9 for c in updated)
        report['checks'].append(dict(name='inference_mode',passed=True))
        single=FrozenSmolStack.from_checkpoint(MODEL,num_layers=1,mode='fp16')
        _,single_seed=single.prefill(x)
        expected=single.decode(one,single_seed,group_size=1)
        before=npu.stats();actual=single.decode(one,single_seed)
        assert delta(before,npu.stats())['chain_graphs']==0
        equal(arrays(expected),arrays(actual))
        report['checks'].append(dict(name='single_layer_default_fallback',bit_equal_arrays=4))
    with torch.no_grad():work();final(npu,report)
    report['status']='passed';write('model-guards.json',report);print('model guards',len(report['checks']),'passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('device','training','guards','task-build','task-probe'),required=True)
    args=parser.parse_args()
    if args.case=='device':
        import torch_npu_device_demo as demo
        demo.OUT=OUT;demo.DESTINATION=OUT/'device.json';raise SystemExit(demo.execute(True))
    elif args.case=='training':
        import torch_npu_training_demo as demo
        demo.OUT=OUT;demo.REPORT=OUT/'training.json';raise SystemExit(demo.execute(True,False))
    elif args.case=='guards':guards()
    else:
        import torch_npu_device_profile as profile
        profile.OUT=OUT/'task-profile';profile.OUT.mkdir(exist_ok=True)
        if args.case=='task-build':profile.build()
        else:profile.execute(1,32,'fp16')

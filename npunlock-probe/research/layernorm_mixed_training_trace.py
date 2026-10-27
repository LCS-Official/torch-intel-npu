"""Separate mixed LayerNorm local derivatives from independent FP16 network drift."""
import copy
import gc
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/layernorm-mixed'


def main():
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.manual_seed(127)
    torch.backends.mkldnn.enabled=False
    host=torch.nn.Sequential(torch.nn.Linear(16,16),torch.nn.LayerNorm(16),torch.nn.GELU(),torch.nn.Linear(16,3)).half()
    host[1].float()
    model=copy.deepcopy(host).to('intel_npu')
    data=torch.randn(4,16,dtype=torch.float16)
    target=torch.tensor([0,1,2,1])
    cpu_values=[]; npu_values=[]
    a,b=data,data.to('intel_npu')
    for cpu_layer,npu_layer in zip(host,model):
        a,b=cpu_layer(a),npu_layer(b)
        a.retain_grad(); b.retain_grad()
        cpu_values.append(a); npu_values.append(b)
    torch.nn.functional.cross_entropy(a,target).backward()
    torch.nn.functional.cross_entropy(b,target.to('intel_npu')).backward()
    def difference(a,b):
        a,b=a.detach().cpu(),b.detach().cpu()
        return {'max_abs':float((a-b).abs().max()),'unequal_elements':int((a!=b).sum()),'elements':a.numel()}
    layers=[]
    for i,(a,b) in enumerate(zip(cpu_values,npu_values)):
        layers.append({'layer':str(host[i]),'forward':difference(a,b),'upstream_gradient':difference(a.grad,b.grad)})
    values=npu_values[0].detach().cpu().requires_grad_()
    seed=npu_values[1].grad.detach().cpu()
    weight=model[1].weight.detach().cpu().requires_grad_()
    bias=model[1].bias.detach().cpu().requires_grad_()
    shadow=torch.nn.functional.layer_norm(values,(16,),weight,bias)
    shadow.backward(seed)
    local={'input':difference(values.grad,npu_values[0].grad),
           'weight':difference(weight.grad,model[1].weight.grad),'bias':difference(bias.grad,model[1].bias.grad)}
    for actual,expected in ((model[1].weight.grad,weight.grad),(model[1].bias.grad,bias.grad)):
        torch.testing.assert_close(actual.detach().cpu(),expected,rtol=3e-5,atol=3e-6)
    full={'weight':difference(host[1].weight.grad,model[1].weight.grad),
          'bias':difference(host[1].bias.grad,model[1].bias.grad)}
    cpu_log=a.detach().log_softmax(-1)
    device_log=npu._log_softmax(b.detach(),-1)
    host_device_log=device_log.cpu()
    logits=a.detach().clone().requires_grad_()
    torch.nn.functional.cross_entropy(logits,target).backward()
    nll_seed=torch.zeros_like(logits)
    nll_seed[torch.arange(4),target]=-.25
    with torch.no_grad():
        same_saved=torch.ops.aten._log_softmax_backward_data(nll_seed,host_device_log,-1,torch.float16)
        g=nll_seed.to('intel_npu')
        got=npu._log_softmax_backward(g,device_log,-1,torch.float16).cpu()
    components={'saved_log_softmax':difference(cpu_log,host_device_log),
                'backward_on_identical_saved_output':difference(same_saved,got),
                'full_ce_gradient':difference(logits.grad,npu_values[-1].grad)}
    result={'layers':layers,'same_input_and_seed_layernorm':local,'independent_cpu_layernorm_gradients':full,
            'same_input_and_seed_strict_affine_passed':True,'classification_components':components,'status':'passed'}
    del host,model,a,b,cpu_values,npu_values,cpu_layer,npu_layer,actual,device_log,g
    gc.collect()
    npu.shutdown()
    result['final_stats']=npu.stats()
    assert result['final_stats']['closed'] and not result['final_stats']['context_failed']
    assert all(result['final_stats'][k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_scalar_buffers','quarantined_buffer_bytes','cached_sessions'))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'training-trace.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()

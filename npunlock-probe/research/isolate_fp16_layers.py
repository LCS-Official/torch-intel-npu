"""Replay each saved layer with identical input/gradient, plus a CPU FP64 reference."""
import copy
import gc
import json
import torch
import torch_intel_npu as npu
from diagnose_fp16_training import OUT, metrics
from torch_npu_training_demo import make_small_resnet


def run(module, x, gradient, device, precise=False):
    dtype = torch.float64 if precise else x.dtype
    model = copy.deepcopy(module)
    if precise:
        model = model.double()
    model = model.to(device)
    data = x.detach().to(dtype=dtype, device=device).requires_grad_()
    output = model(data)
    output.backward(gradient.to(dtype=output.dtype, device=device))
    return {'output': output.detach().cpu(), 'input_gradient': data.grad.cpu(),
            **{name: p.grad.cpu() for name, p in model.named_parameters()}}


def main():
    torch.backends.mkldnn.enabled = False
    saved = torch.load(OUT / 'fp16-residual-layer-tensors.pt', weights_only=True)
    network = make_small_resnet(torch.float16)
    network.load_state_dict(saved['state'])
    layers = dict(network.named_modules())
    report = []
    for name, tensors in saved['records']['npu'].items():
        module = layers[name]
        if isinstance(module, torch.nn.ReLU):
            continue
        x, gradient = tensors['input'], tensors['gradient']
        cpu = run(module, x, gradient, 'cpu')
        precise = run(module, x, gradient, 'cpu', precise=True)
        actual = run(module, x, gradient, 'intel_npu')
        row = {'name': name, 'type': type(module).__name__, 'comparisons': {}}
        for key, value in actual.items():
            rounded = precise[key].to(cpu[key].dtype)
            row['comparisons'][key] = {'npu_vs_cpu': metrics(value, cpu[key]),
                                       'npu_vs_fp64_rounded': metrics(value, rounded),
                                       'cpu_vs_fp64_rounded': metrics(cpu[key], rounded)}
        report.append(row)
        print(row, flush=True)
        del actual, cpu, precise
        gc.collect()
    final = npu.stats()
    assert not final['context_failed']
    assert final['live_buffers'] == final['cached_scalar_buffers'], final
    npu.shutdown()
    (OUT / 'fp16-isolated-layers.json').write_text(json.dumps(report, indent=2))
    print('isolation complete', flush=True)


if __name__ == '__main__':
    main()

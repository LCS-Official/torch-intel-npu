"""Capture/replay the original FP16 residual failure, with explicit CPU validation copies."""
import copy
import json
from pathlib import Path
import torch
import torch_npu_training_demo as training

OUT = training.OUT
CHECKPOINT = OUT / 'fp16-residual-step5.pt'
REPORT = OUT / 'fp16-residual-diagnosis.json'


def capture():
    original = training.make_small_resnet

    def factory(dtype):
        model = original(dtype)
        calls = 0

        def snapshot(module, inputs):
            nonlocal calls
            if dtype != torch.float16 or inputs[0].device.type != 'cpu' or not torch.is_grad_enabled():
                return
            calls += 1
            if calls == 6:
                torch.save({'state': {k: v.detach().clone() for k, v in module.state_dict().items()},
                            'input': inputs[0].detach().clone(), 'step': 5}, CHECKPOINT)
                print('Captured original step 5:', CHECKPOINT, flush=True)
        model.register_forward_pre_hook(snapshot)
        return model

    training.make_small_resnet = factory
    training.REPORT = OUT / 'fp16-capture-training.json'
    return training.execute(False, True)


def metrics(actual, expected):
    a, b = actual.float(), expected.float()
    error = (a-b).abs()
    return dict(max_absolute=float(error.max()), relative_l2=float(error.norm()/b.norm().clamp_min(1e-30)),
                unequal=int((actual != expected).sum()), elements=actual.numel())


def replay(cpu_relu_mask=False, checkpoint=CHECKPOINT, prefix=None):
    import torch_intel_npu as npu
    torch.backends.mkldnn.enabled = False
    saved = torch.load(checkpoint, weights_only=True)
    cpu = training.make_small_resnet(torch.float16)
    cpu.load_state_dict(saved['state'])
    model = copy.deepcopy(cpu).to('intel_npu')
    records = {'cpu': {}, 'npu': {}}
    handles = []

    def hook(name):
        def record(module, inputs, output):
            device = 'cpu' if output.device.type == 'cpu' else 'npu'
            row = records[device].setdefault(name, {})
            row['input'] = inputs[0].detach().cpu().clone()
            row['output'] = output.detach().cpu().clone()
            mask = None
            if cpu_relu_mask and device == 'npu' and isinstance(module, torch.nn.ReLU):
                # Diagnostic intervention only: suppress entries the CPU reference
                # considers inactive, keeping all forward values/NPU weights intact.
                mask = (records['cpu'][name]['input'] > 0).to(output.dtype).to(output.device)

            def gradient(g):
                row['gradient'] = g.detach().cpu().clone()
                row['gradient_stride'] = list(g.stride())
                row['gradient_contiguous'] = g.is_contiguous()
                return g * mask if mask is not None else g
            output.register_hook(gradient)
        return record

    for network in (cpu, model):
        for name, module in network.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU, torch.nn.AdaptiveAvgPool2d, torch.nn.Linear)):
                handles.append(module.register_forward_hook(hook(name)))
    host = saved['input']
    data = host.to('intel_npu')
    reference, actual = cpu(host), model(data)
    target = torch.full_like(reference, .2)
    ((reference.float()-target.float()).square().mean().half()).backward()
    torch.nn.functional.mse_loss(actual, target.to('intel_npu')).backward()
    report = {'step': saved['step'], 'checkpoint': str(checkpoint),
              'diagnostic_cpu_relu_mask': cpu_relu_mask,
              'cpu_predictions': reference.detach().tolist(), 'npu_predictions': actual.detach().cpu().tolist(),
              'layers': [], 'parameter_gradients': []}
    for name, a in records['npu'].items():
        b = records['cpu'][name]
        row = {'name': name, 'forward': metrics(a['output'], b['output']),
               'incoming_gradient': metrics(a['gradient'], b['gradient']),
               'cpu_gradient_stride': b['gradient_stride'], 'cpu_gradient_contiguous': b['gradient_contiguous']}
        if isinstance(dict(cpu.named_modules())[name], torch.nn.ReLU):
            mask = (a['input'] > 0) != (b['input'] > 0)
            row['relu_mask_mismatches'] = int(mask.sum())
            row['relu_mismatch_cpu_values'] = b['input'][mask].tolist()
            row['relu_mismatch_npu_values'] = a['input'][mask].tolist()
        report['layers'].append(row)
        print(row, flush=True)
    for (name, p), q in zip(model.named_parameters(), cpu.parameters()):
        report['parameter_gradients'].append({'name': name, **metrics(p.grad.cpu(), q.grad)})
    # Save tensors for same-input/same-seed operator replay in a fresh process.
    if not cpu_relu_mask and prefix is None:
        torch.save({'records': records, 'state': saved['state'], 'input': host}, OUT / 'fp16-residual-layer-tensors.pt')
    path = OUT / f'{prefix}.json' if prefix else OUT / 'fp16-relu-boundary-intervention.json' if cpu_relu_mask else REPORT
    path.write_text(json.dumps(report, indent=2))
    for h in handles:
        h.remove()
    del actual, reference, cpu, model, data, p, q, module, network
    import gc
    gc.collect()
    assert not npu.stats()['context_failed']
    assert npu.stats()['live_buffers'] == npu.stats()['cached_scalar_buffers'], npu.stats()
    npu.shutdown()
    print('diagnosis complete', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(capture() if not CHECKPOINT.exists() else replay())

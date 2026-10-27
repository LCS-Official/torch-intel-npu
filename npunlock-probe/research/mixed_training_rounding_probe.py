"""Trace independent mixed training and replay each layer on identical inputs/gradients."""
import argparse
import copy
import ctypes
import gc
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/mixed-training-rounding'


def main(baseline=False, steps_count=4, unfused_reference=False):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.manual_seed(127)
    assert not baseline or steps_count == 4
    assert not unfused_reference or steps_count == 6 and not baseline
    fma = ctypes.CDLL('ucrtbase').fmaf
    fma.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fma.restype = ctypes.c_float

    def compare(expected, actual):
        expected, actual = expected.detach().cpu(), actual.detach().cpu()
        rtol, atol = (.003, .0005) if expected.dtype == torch.float16 else (3e-5, 3e-6)
        bad = ~torch.isclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True)
        unequal = expected != actual
        flat = unequal.flatten().nonzero().flatten()[:8]
        return {'max_abs': float((expected.float()-actual.float()).abs().max()),
                'unequal': int(unequal.sum()), 'outside_tolerance': int(bad.sum()),
                'examples': [{'index': int(i), 'cpu': float(expected.flatten()[i]),
                              'npu': float(actual.flatten()[i])} for i in flat]}

    runs = []
    first_failure = None
    for mode in ('none', 'tanh'):
        host = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.LayerNorm(16),
                                   torch.nn.GELU(approximate=mode), torch.nn.Linear(16, 3)).half()
        host[1].float()
        model = copy.deepcopy(host).to('intel_npu')
        reference, actual = (torch.optim.SGD(v.parameters(), lr=.01, momentum=.8,
                                           foreach=False, fused=False) for v in (host, model))
        data = torch.randn(4, 16, dtype=torch.float16)
        labels = torch.tensor([0, 1, 2, 1])
        x, target = data.to('intel_npu'), labels.to('intel_npu')
        digest = hashlib.sha256(data.numpy().tobytes())
        for value in host.parameters(): digest.update(value.detach().numpy().tobytes())
        steps = []
        for step in range(steps_count):
            before = npu.stats()['npu_calls']
            reference.zero_grad(set_to_none=True)
            actual.zero_grad(set_to_none=True)
            cpu_values, npu_values = [], []
            a, b = data, x
            for cpu_layer, npu_layer in zip(host, model):
                a, b = cpu_layer(a), npu_layer(b)
                a.retain_grad(); b.retain_grad()
                cpu_values.append(a); npu_values.append(b)
            expected = torch.nn.functional.cross_entropy(a.float(), labels)
            result = torch.nn.functional.cross_entropy(b.float(), target)
            expected.backward(); result.backward()
            parameters = {}
            for (name, e), (_, v) in zip(host.named_parameters(), model.named_parameters()):
                parameters[name] = compare(e.grad, v.grad)
                if parameters[name]['outside_tolerance'] and first_failure is None:
                    first_failure = {'mode': mode, 'step': step, 'parameter': name}
            layers = []
            for index, (cpu_value, npu_value) in enumerate(zip(cpu_values, npu_values)):
                local_input = (x if index == 0 else npu_values[index-1]).detach().cpu().requires_grad_()
                shadow = copy.deepcopy(host[index])
                shadow.load_state_dict({k: v.detach().cpu() for k, v in model[index].state_dict().items()})
                local_output = shadow(local_input)
                local_output.backward(npu_value.grad.detach().cpu())
                local = {'forward': compare(local_output, npu_value)}
                if index:
                    local['input_gradient'] = compare(local_input.grad, npu_values[index-1].grad)
                for name, value in model[index].named_parameters():
                    local[name+'_gradient'] = compare(dict(shadow.named_parameters())[name].grad, value.grad)
                layer = {'layer': str(host[index]), 'forward': compare(cpu_value, npu_value),
                         'upstream_gradient': compare(cpu_value.grad, npu_value.grad), 'identical_input_and_seed': local}
                if index == 2:
                    # Keep just discrepant activation samples, never entire experiment tensors.
                    indices = (local_output.detach() != npu_value.detach().cpu()).flatten().nonzero().flatten()
                    indices = sorted(set(indices.tolist()+
                        (local_input.grad != npu_values[index-1].grad.detach().cpu()).flatten().nonzero().flatten().tolist()))
                    layer['local_samples'] = [dict(index=i, x=float(local_input.detach().flatten()[i]),
                        seed=float(npu_value.grad.detach().cpu().flatten()[i]),
                        cpu_y=float(local_output.detach().flatten()[i]), npu_y=float(npu_value.detach().cpu().flatten()[i]),
                        cpu_dx=float(local_input.grad.flatten()[i]), npu_dx=float(npu_values[index-1].grad.detach().cpu().flatten()[i]))
                        for i in indices]
                layers.append(layer)
            optimizer = []
            if steps_count == 6:
                # Replay the optimizer on the exact NPU weights, gradients and resident momentum.
                shadow_parameters = [torch.nn.Parameter(p.detach().cpu()) for p in model.parameters()]
                shadow_optimizer = torch.optim.SGD(shadow_parameters, lr=.01, momentum=.8, foreach=False, fused=False)
                snapshots = []
                for (name, p), q in zip(model.named_parameters(), shadow_parameters):
                    q.grad = p.grad.detach().cpu()
                    snapshots.append(q.detach().clone())
                    if 'momentum_buffer' in actual.state[p]:
                        shadow_optimizer.state[q]['momentum_buffer'] = actual.state[p]['momentum_buffer'].detach().cpu()
                    state = {'parameter': name, 'before_weight': compare(dict(host.named_parameters())[name], p)}
                    if 'momentum_buffer' in actual.state[p]:
                        state['before_momentum'] = compare(reference.state[dict(host.named_parameters())[name]]['momentum_buffer'], actual.state[p]['momentum_buffer'])
                    optimizer.append(state)
                shadow_optimizer.step()
            if unfused_reference:
                # Diagnostic CPU control only; production NPU still uses unchanged torch.optim.SGD.
                with torch.no_grad():
                    for p in host.parameters():
                        if 'momentum_buffer' not in reference.state[p]:
                            reference.state[p]['momentum_buffer'] = p.grad.detach().clone()
                        else:
                            reference.state[p]['momentum_buffer'].mul_(.8).add_(p.grad)
                        momentum = reference.state[p]['momentum_buffer']
                        alpha = float(torch.tensor(-.01, dtype=p.dtype))
                        p.copy_((p.float()+momentum.float()*alpha).to(p.dtype))
            else:
                reference.step()
            actual.step()
            if steps_count == 6:
                for state, old, (name, p), q in zip(optimizer, snapshots, model.named_parameters(), shadow_parameters):
                    state['identical_state_weight'] = compare(q, p)
                    momentum = actual.state[p]['momentum_buffer'].detach().cpu()
                    state['identical_state_momentum'] = compare(shadow_optimizer.state[q]['momentum_buffer'], momentum)
                    alpha = float(torch.tensor(-.01, dtype=p.dtype))
                    separate = (old.float()+momentum.float()*alpha).to(p.dtype)
                    fused = torch.tensor([fma(alpha, float(g), float(w)) for w, g in zip(old.flatten(), momentum.flatten())], dtype=p.dtype).reshape(old.shape)
                    state['npu_vs_separate_fp32'] = compare(separate, p)
                    state['cpu_vs_fused_fp32'] = compare(fused, q)
                    state['npu_vs_fused_fp32'] = compare(fused, p)
                    if p.dtype == torch.float16:
                        half_product = old+(momentum*torch.tensor(alpha, dtype=torch.float16))
                        state['cpu_vs_half_product'] = compare(half_product, q)
                        split = old.numel()//32*32
                        hybrid = fused.flatten().clone()
                        hybrid[split:] = half_product.flatten()[split:]
                        state['cpu_vs_vector_and_scalar_tail'] = compare(hybrid.reshape(old.shape), q)
                        assert state['cpu_vs_vector_and_scalar_tail']['unequal'] == 0, state
                    assert state['npu_vs_separate_fp32']['unequal'] == 0, state
                del p, q, shadow_parameters, shadow_optimizer
            weights = {name: compare(e, v) for (name, e), (_, v) in zip(host.named_parameters(), model.named_parameters())}
            entry = {'step': step, 'loss': compare(expected, result), 'parameters': parameters,
                     'layers': layers, 'updated_weights': weights, 'npu_calls': npu.stats()['npu_calls']-before}
            if optimizer: entry['optimizer'] = optimizer
            steps.append(entry)
            print(mode, step, 'gradient failures', {k:v['outside_tolerance'] for k,v in parameters.items() if v['outside_tolerance']},
                  'local differences', [{k:v['unequal'] for k,v in layer['identical_input_and_seed'].items() if v['unequal']} for layer in layers], flush=True)
        runs.append({'mode': mode, 'initial_input_and_parameters_sha256': digest.hexdigest(), 'steps': steps})
        del host, model, reference, actual, x, target, a, b, result, cpu_values, npu_values, cpu_value, npu_value, cpu_layer, npu_layer, e, v, value
        gc.collect()
    # The independent CPU path is never rebased onto NPU weights or activations.
    expected_failure = None if unfused_reference else {'mode': 'none', 'step': 5, 'parameter': '1.weight'} if steps_count == 6 else ({'mode': 'tanh', 'step': 2, 'parameter': '1.weight'} if baseline else None)
    assert first_failure == expected_failure, first_failure
    if unfused_reference:
        assert all(v['unequal'] == 0 for run in runs for step in run['steps'] for v in (*step['parameters'].values(), *step['updated_weights'].values()))
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
    result = {'status': 'diagnosed' if baseline or steps_count == 6 else 'passed', 'unfused_cpu_reference': unfused_reference, 'first_independent_failure': first_failure, 'runs': runs, 'final_stats': final,
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
               ('torch_intel_npu.py', 'torch_npu_autograd.py', 'npunlock-probe/research/mixed_training_rounding_probe.py',
                'npunlock/tests/fixtures/npu4000/resident_gelu.c', 'npunlock/tests/fixtures/npu4000/resident/manifest.json')}}
    OUT.mkdir(parents=True, exist_ok=True)
    output = 'trajectory-unfused-control.json' if unfused_reference else 'trajectory-6step.json' if steps_count == 6 else 'trajectory-before.json' if baseline else 'trajectory-after.json'
    (OUT/output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('first independent failure', first_failure, 'NPU calls', final['npu_calls'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', action='store_true', help='Requires the historical pre-fix backend')
    parser.add_argument('--steps', type=int, choices=(4, 6), default=4, help='Six steps additionally replay optimizer state and rounding')
    parser.add_argument('--unfused-reference', action='store_true', help='Six-step diagnostic: CPU reference uses separate FP32 update arithmetic')
    args = parser.parse_args()
    main(args.baseline, args.steps, args.unfused_reference)

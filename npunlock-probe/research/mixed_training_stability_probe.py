"""Separate 32-step toy learning stability from strict independent CPU trajectory parity."""
import copy
import gc
import hashlib
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/classification-precision'


def main():
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.manual_seed(127)
    checks = []

    def training(mode):
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
        first_strict_failure = None

        def compare(cpu, device, where, failures):
            nonlocal first_strict_failure
            cpu, device = cpu.detach().cpu(), device.detach().cpu()
            assert torch.isfinite(cpu).all() and torch.isfinite(device).all(), where
            rtol, atol = (.003, .0005) if cpu.dtype == torch.float16 else (3e-5, 3e-6)
            different = ~torch.isclose(cpu, device, rtol=rtol, atol=atol)
            maximum = float((cpu.double()-device.double()).abs().max())
            if different.any():
                row = {'step': len(steps), 'where': where, 'mismatched': int(different.sum()), 'max_abs': maximum}
                failures.append(row)
                if first_strict_failure is None: first_strict_failure = row
            return maximum

        for step in range(32):
            reference.zero_grad(set_to_none=True); actual.zero_grad(set_to_none=True)
            cpu_logits, logits = host(data), model(x)
            expected = torch.nn.functional.cross_entropy(cpu_logits.float(), labels)
            result = torch.nn.functional.cross_entropy(logits.float(), target)
            failures = []
            compare(expected, result, 'loss', failures)
            expected.backward(); result.backward()
            gradient_error = {}
            for (name, cpu), (_, device) in zip(host.named_parameters(), model.named_parameters()):
                gradient_error[name] = compare(cpu.grad, device.grad, 'gradient:'+name, failures)
            reference.step(); actual.step()
            weight_error = {}
            for (name, cpu), (_, device) in zip(host.named_parameters(), model.named_parameters()):
                weight_error[name] = compare(cpu, device, 'weight:'+name, failures)
            steps.append({'step': step, 'cpu_loss': float(expected.detach()), 'npu_loss': float(result.detach().cpu()),
                          'gradient_max_abs': gradient_error, 'weight_max_abs': weight_error, 'strict_failures': failures})
            if step in (0, 3, 7, 15, 23, 31):
                print(mode, step+1, 'loss', steps[-1]['npu_loss'], 'strict first failure', first_strict_failure, flush=True)
        with torch.no_grad():
            cpu_output = host(data).float()
            device_output = model(x).float().cpu()
        final_cpu_loss = float(torch.nn.functional.cross_entropy(cpu_output, labels))
        final_npu_loss = float(torch.nn.functional.cross_entropy(device_output, labels))
        accuracy = float((device_output.argmax(-1) == labels).float().mean())
        assert final_npu_loss < steps[0]['npu_loss'] and accuracy == 1.
        return {'mode': mode, 'steps': steps, 'final_cpu_loss': final_cpu_loss, 'final_npu_loss': final_npu_loss,
                'training_accuracy': accuracy, 'initial_input_and_parameters_sha256': digest.hexdigest(),
                'first_strict_failure': first_strict_failure, 'strict_independent_trajectory_passed': first_strict_failure is None,
                'finite_learning_checks_passed': True, 'fixed_four_sample_toy_not_generalization': True}

    for mode in ('none', 'tanh'):
        before = npu.stats()
        row = {'name': 'mixed-32step-'+mode}
        try:
            row.update(training(mode))
            row['status'] = 'passed' if row['strict_independent_trajectory_passed'] else 'strict_failed'
        except Exception as error:
            row.update(status='failed', error=str(error), traceback=traceback.format_exc())
        gc.collect()
        after = npu.stats()
        row['npu_calls'] = after['npu_calls']-before['npu_calls']
        row['live_buffer_delta'] = after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
        if row['live_buffer_delta'] or after['context_failed']:
            row.update(status='failed', lifecycle_error='Live buffer leak or failed context')
        checks.append(row)
        print(row['name'], row['status'], row.get('error', '')[:240], flush=True)
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
    status = 'failed' if any(c['status'] == 'failed' for c in checks) else 'strict_failed' if any(c['status'] == 'strict_failed' for c in checks) else 'passed'
    result = {'status': status, 'checks': checks, 'final_stats': final,
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('torch_intel_npu.py', 'torch_npu_autograd.py', 'npunlock-probe/research/mixed_training_stability_probe.py',
                 'npunlock/tests/fixtures/npu4000/resident/manifest.json')}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'stability.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return status != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())

"""Independent CPU/NPU mixed training on synthetic train and held-out samples."""
import copy
import gc
import hashlib
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/mixed-step-diagnosis'


def main():
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.manual_seed(131)

    def experiment():
        generator = torch.Generator().manual_seed(601)
        teacher = torch.randn(16, 3, generator=generator)
        train = torch.randn(64, 16, generator=generator).half()
        heldout = torch.randn(64, 16, generator=generator).half()
        labels = (train.float()@teacher).argmax(-1)
        targets = (heldout.float()@teacher).argmax(-1)
        host = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.LayerNorm(16), torch.nn.GELU(), torch.nn.Linear(16, 3)).half()
        host[1].float()
        model = copy.deepcopy(host).to('intel_npu')
        reference, actual = (torch.optim.SGD(v.parameters(), lr=.03, momentum=.8, foreach=False, fused=False) for v in (host, model))
        model_digest = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in host.parameters())).hexdigest()

        def evaluate():
            with torch.no_grad():
                cpu_logits = host(heldout).float()
                logits = model(heldout.to('intel_npu')).float()
                loss = torch.nn.functional.cross_entropy(logits, targets.to('intel_npu')).cpu()
                device_logits = logits.cpu()
            assert torch.isfinite(loss) and torch.isfinite(device_logits).all()
            return {'cpu_loss': float(torch.nn.functional.cross_entropy(cpu_logits, targets)), 'npu_loss': float(loss),
                    'cpu_accuracy': float((cpu_logits.argmax(-1)==targets).float().mean()),
                    'npu_accuracy': float((device_logits.argmax(-1)==targets).float().mean()),
                    'prediction_disagreements': int((cpu_logits.argmax(-1)!=device_logits.argmax(-1)).sum()),
                    'logits_max_abs': float((cpu_logits-device_logits).abs().max())}

        initial = evaluate()
        steps, first_failure = [], None
        for epoch in range(4):
            indices = torch.randperm(64, generator=generator).reshape(8, 8)
            for batch in indices:
                reference.zero_grad(set_to_none=True); actual.zero_grad(set_to_none=True)
                data, target = train[batch], labels[batch]
                cpu_loss = torch.nn.functional.cross_entropy(host(data).float(), target)
                loss = torch.nn.functional.cross_entropy(model(data.to('intel_npu')).float(), target.to('intel_npu'))
                cpu_loss.backward(); loss.backward()
                failures = []

                def compare(cpu, device, name):
                    cpu, device = cpu.detach().cpu(), device.detach().cpu()
                    assert torch.isfinite(cpu).all() and torch.isfinite(device).all(), name
                    rtol, atol = (.003, .0005) if cpu.dtype == torch.float16 else (3e-5, 3e-6)
                    bad = ~torch.isclose(cpu, device, rtol=rtol, atol=atol)
                    if bad.any(): failures.append({'where': name, 'mismatched': int(bad.sum()), 'max_abs': float((cpu.double()-device.double()).abs().max())})

                compare(cpu_loss, loss, 'loss')
                for (name, cpu), (_, device) in zip(host.named_parameters(), model.named_parameters()): compare(cpu.grad, device.grad, 'gradient:'+name)
                reference.step(); actual.step()
                for (name, cpu), (_, device) in zip(host.named_parameters(), model.named_parameters()): compare(cpu, device, 'weight:'+name)
                row = {'step': len(steps), 'cpu_loss': float(cpu_loss.detach()), 'npu_loss': float(loss.detach().cpu()), 'strict_failures': failures}
                if failures and first_failure is None: first_failure = dict(step=row['step'], **failures[0])
                steps.append(row)
            print('epoch', epoch+1, 'last training loss', steps[-1]['npu_loss'], 'first strict failure', first_failure, flush=True)
        final = evaluate()
        assert final['npu_loss'] < initial['npu_loss'] and final['npu_accuracy'] > initial['npu_accuracy']
        data_hashes = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in
                       (('train', train), ('train_labels', labels), ('heldout', heldout), ('heldout_labels', targets), ('teacher', teacher))}
        return {'training_samples': 64, 'heldout_samples': 64, 'steps': steps, 'batch_size': 8, 'initial': initial, 'final': final,
                'first_strict_failure': first_failure, 'strict_trajectory_passed': first_failure is None,
                'heldout_improvement_passed': True, 'synthetic_task_not_real_dataset_certification': True,
                'data_sha256': data_hashes, 'initial_parameters_sha256': model_digest}

    before = npu.stats()
    row = {'name': 'default-gelu-heldout-64'}
    try:
        row.update(experiment())
        row['status'] = 'passed' if row['strict_trajectory_passed'] else 'strict_failed'
    except Exception as error:
        row.update(status='failed', error=str(error), traceback=traceback.format_exc())
    gc.collect()
    after = npu.stats()
    row['npu_calls'] = after['npu_calls']-before['npu_calls']
    row['live_buffer_delta'] = after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
    if row['live_buffer_delta'] or after['context_failed']: row.update(status='failed', lifecycle_error='Live buffer leak or failed context')
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
    result = {'status': row['status'], 'checks': [row], 'final_stats': final,
              'source_sha256': {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
               ('torch_intel_npu.py', 'torch_npu_autograd.py', 'npunlock-probe/research/mixed_validation_probe.py',
                'npunlock/tests/fixtures/npu4000/resident/manifest.json')}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: row[k] for k in ('status', 'initial', 'final', 'first_strict_failure', 'npu_calls') if k in row}), flush=True)
    return row['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())

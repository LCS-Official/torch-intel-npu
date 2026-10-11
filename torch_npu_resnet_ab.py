"""Interleaved old/new reduction in identical resident SmallResNet training."""
import copy
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch
import torch_intel_npu as npu
from torch_npu_benchmark import host_case

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'npunlock-probe/research/results/compatibility'
OLD = ROOT / 'npunlock-probe/research/compat-before/npunlock/tests/fixtures/npu4000/resident'


def main():
    torch.set_num_threads(8)
    fixture = torch.load(ROOT/'npunlock-probe/research/results/performance/fixtures/resnet_train-float32.pt', weights_only=True)
    host, _, cpu, _ = host_case('resnet_train', 'float32', fixture)
    models = {name: copy.deepcopy(cpu).to('intel_npu') for name in ('old', 'new')}
    inputs = {name: host.to('intel_npu').requires_grad_() for name in models}
    optimizers = {name: torch.optim.SGD(model.parameters(), lr=.001, foreach=False, fused=False) for name, model in models.items()}
    target = torch.zeros(2, 4).to('intel_npu')
    old_manifest = json.loads((OLD/'manifest.json').read_text())
    original_kernel, original_session = npu._resident_kernel, npu._session
    state = {'old': False}
    report = {'status': 'running', 'source_sha256': {
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('torch_npu_resnet_ab.py', 'torch_intel_npu.py', 'torch_npu_autograd.py', 'npunlock/tests/fixtures/npu4000/resident/manifest.json')},
        'fixture_sha256': hashlib.sha256((ROOT/'npunlock-probe/research/results/performance/fixtures/resnet_train-float32.pt').read_bytes()).hexdigest(),
        'old_manifest_sha256': hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest(),
        'scope': 'identical FP32 SmallResNet training, same process, old/new resident reduction only; compatibility fixes shared',
        'warmup_pairs': 3, 'measured_pairs': 9}

    def kernel(kind):
        if state['old'] and kind in ('sum', 'mean'):
            elf = (OLD/f'{kind}.elf').read_bytes()
            assert hashlib.sha256(elf).hexdigest() == old_manifest['kernels'][kind]['elf_sha256']
            return elf
        return original_kernel(kind)

    def session(key):
        # A separate retained cache key per old reduction avoids destroying or
        # re-importing graphs. The carrier's axes remain the first extra field.
        return original_session((*key, 'old_reduction') if state['old'] and key[0] in ('sum', 'mean') else key)

    def step(name):
        state['old'] = name == 'old'
        optimizers[name].zero_grad(set_to_none=True)
        inputs[name].grad = None
        output = models[name](inputs[name])
        error = output.float()-target
        loss = (error*error).mean()
        loss.backward()
        optimizers[name].step()
        return output.detach(), loss.detach()

    def equal():
        np.testing.assert_array_equal(inputs['old'].grad.cpu().numpy().view(np.uint32),
                                      inputs['new'].grad.cpu().numpy().view(np.uint32))
        for a, b in zip(models['old'].parameters(), models['new'].parameters()):
            np.testing.assert_array_equal(a.detach().cpu().numpy().view(np.uint32), b.detach().cpu().numpy().view(np.uint32))
            np.testing.assert_array_equal(a.grad.cpu().numpy().view(np.uint32), b.grad.cpu().numpy().view(np.uint32))

    try:
        npu._resident_kernel, npu._session = kernel, session
        for _ in range(3):
            for name in ('old', 'new'):
                value = step(name)
                del value
            equal()
        samples = {'old': [], 'new': []}
        for i in range(9):
            for name in (('old', 'new') if i % 2 == 0 else ('new', 'old')):
                before = npu.stats()
                started = time.perf_counter()
                value = step(name)
                samples[name].append((time.perf_counter()-started)*1000)
                after = npu.stats()
                for field in ('host_upload_copies', 'host_download_copies', 'host_device_copies', 'session_creations'):
                    assert after[field] == before[field], field
                del value
            equal()
            print('pair', i, samples['old'][-1], samples['new'][-1], flush=True)
        report['samples_ms'] = samples
        report['median_ms'] = {name: statistics.median(values) for name, values in samples.items()}
        report['speedup'] = report['median_ms']['old']/report['median_ms']['new']
        report['all_steps_input_gradients_parameters_gradients_and_updates_bitwise_equal'] = True
        report['warm_host_copies_and_compiles'] = 0
        del models, inputs, optimizers, target
        gc.collect()
        report['stats_before_shutdown'] = npu.stats()
        assert report['stats_before_shutdown']['live_buffers'] == report['stats_before_shutdown']['cached_scalar_buffers']
        npu.shutdown()
        report['final_stats'] = npu.stats()
        assert report['final_stats']['live_buffers'] == 0 and not report['final_stats']['context_failed']
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc(), failure_stats=npu.stats())
        print(report['traceback'], flush=True)
    finally:
        npu._resident_kernel, npu._session = original_kernel, original_session
    (OUT/'resnet-interleaved.json').write_text(json.dumps(report, indent=2))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())

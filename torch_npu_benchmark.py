"""CPU/NPU/XPU eager wall-time benchmark; no changes to device kernels."""
import argparse
import copy
import cProfile
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'npunlock-probe/research/results/performance'
FIXTURES = OUT / 'fixtures'
SOURCES = ('torch_npu_benchmark.py', 'torch_intel_npu.py', 'torch_npu_autograd.py', 'torch_npu_geometry.py',
           'npunlock/python/npunlock/npu4000.py', 'npunlock-probe/torch-device/backend.cpp',
           'npunlock-probe/torch-device/build.json', 'npunlock/tests/fixtures/npu4000/precision/manifest.json',
           'npunlock/tests/fixtures/npu4000/resident/manifest.json')
WORKLOADS = ('elementwise_1m', 'mm_64', 'mm_256', 'linear_forward',
             'linear_train_large_batch', 'conv_forward', 'conv_train', 'resnet_train')


def host_case(workload, precision, fixture=None):
    import torch
    from torch_npu_training_demo import make_small_resnet
    torch.manual_seed(61)
    dtype = getattr(torch, precision)
    model, right = None, None
    if workload.startswith('mm_'):
        size = int(workload.split('_')[1])
        host = torch.randn(size, size, dtype=dtype)/8 if fixture is None else fixture['host']
        right = torch.randn(size, size, dtype=dtype)/8 if fixture is None else fixture['right']
        meta = {'shape': [size, size, size], 'effective_flops': 2*size**3}
    elif workload == 'elementwise_1m':
        host = (torch.arange(1048576) % 17-8).to(dtype)/8 if fixture is None else fixture['host']
        meta = {'shape': [1048576], 'effective_flops': 2*1048576}
    else:
        if workload == 'linear_forward':
            model, shape = torch.nn.Linear(256, 256, dtype=dtype), (128, 256)
            meta = {'shape': list(shape), 'effective_flops': 2*128*256*256}
        elif workload == 'linear_train_large_batch':
            model, shape = torch.nn.Linear(2, 2, dtype=dtype), (65537, 2)
            meta = {'shape': list(shape), 'weight_gradient_k': 65537}
        elif workload.startswith('conv_'):
            model, shape = torch.nn.Conv2d(4, 4, 3, padding=1, dtype=dtype), (1, 4, 176, 176)
            meta = {'shape': list(shape), 'window_elements': 1115136}
            if workload == 'conv_forward':
                meta['effective_flops'] = 2*176*176*4*36
        else:
            model, shape = make_small_resnet(dtype), (2, 3, 8, 8)
            meta = {'shape': list(shape), 'residual_blocks': 4, 'parameters': sum(p.numel() for p in model.parameters())}
        host = torch.randn(shape, dtype=dtype)/3 if fixture is None else fixture['host']
        if fixture is not None:
            model.load_state_dict(fixture['state'])
    return host, right, model, meta


def summarize(samples):
    ordered = sorted(samples)
    quartiles = statistics.quantiles(ordered, n=4, method='inclusive')
    return {'median_ms': statistics.median(samples), 'min_ms': min(samples), 'max_ms': max(samples),
            'iqr_ms': quartiles[2]-quartiles[0], 'samples_ms': samples}


def execute(workload, precision, mode):
    import torch
    dtype = getattr(torch, precision)
    torch.set_float32_matmul_precision('highest')
    training = '_train' in workload
    report = {'status': 'running', 'workload': workload, 'precision': precision, 'mode': mode,
              'torch': torch.__version__, 'python': sys.version, 'results': [],
              'float32_matmul_precision': torch.get_float32_matmul_precision(),
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in SOURCES},
              'timing_scope': 'resident eager wall time; final accelerator sync included; setup/validation excluded',
              'training_step': 'zero_grad, input grad reset, forward, FP32 MSE, backward, SGD(lr=.001)',
              'warmup_calls': 3, 'measured_calls': 7 if training else 11}
    path = OUT / f'{mode}-{workload}-{precision}.json'
    npu = None

    def save(stage):
        report['stage'] = stage
        path.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)

    def payload_copies(stats, before):
        return sum(stats[k]-before[k] for k in ('host_upload_copies', 'host_download_copies', 'host_device_copies'))

    def prepare(device):
        fixture_path = FIXTURES / f'{workload}-{precision}.pt'
        fixture = torch.load(fixture_path, weights_only=True)
        host, right, model, meta = host_case(workload, precision, fixture)
        meta['fixture_sha256'] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        # All devices/versions load the exact same saved input and weights.
        reference_model = copy.deepcopy(model) if model is not None else None
        meta['input_sha256'] = hashlib.sha256(host.numpy().tobytes() +
            (right.numpy().tobytes() if right is not None else b'')).hexdigest()
        meta['model_sha256'] = (hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
                                if model is not None else None)
        cpu_input = host.detach().clone().requires_grad_(training)
        x = host.to(device).requires_grad_(training)
        y = right.to(device) if right is not None else None
        model = model.to(device) if model is not None else None
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, foreach=False, fused=False) if training else None
        target_host = torch.zeros_like(reference_model(cpu_input).detach()) if training else None
        target = target_host.to(device) if target_host is not None else None

        def operation():
            if training:
                optimizer.zero_grad(set_to_none=True)
                x.grad = None
                output = model(x)
                error = output.float()-target.float()
                loss = (error*error).mean()
                loss.backward()
                optimizer.step()
                return output.detach(), loss.detach()
            with torch.no_grad():
                if workload == 'elementwise_1m':
                    return (x+.25)*.5
                return x@y if y is not None else model(x)

        def validate_impl(result):
            if workload == 'elementwise_1m':
                expected = (host+.25)*.5
            elif y is not None:
                expected = host@right
            else:
                expected = reference_model(cpu_input)
            tol = {'rtol': 4e-3, 'atol': 5e-4} if dtype == torch.float16 else {'rtol': 7e-5, 'atol': 6e-6}
            actual = result[0] if training else result
            torch.testing.assert_close(actual.cpu(), expected.detach(), **tol)
            if not training:
                return {'forward_passed': True, 'tolerance': tol}
            difference = expected.float()-target_host.float()
            loss = (difference*difference).mean()
            loss.backward()
            torch.testing.assert_close(result[1].cpu(), loss.detach(), **tol)
            reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=.001, foreach=False, fused=False)
            reference_optimizer.step()
            errors, values = [], []
            for p, q in zip(model.parameters(), reference_model.parameters()):
                actual_gradient = p.grad.cpu()
                torch.testing.assert_close(actual_gradient, q.grad, **tol)
                torch.testing.assert_close(p.detach().cpu(), q.detach(), **tol)
                errors.append((actual_gradient.float()-q.grad.float()).flatten())
                values.append(q.grad.float().flatten())
            torch.testing.assert_close(x.grad.cpu(), cpu_input.grad, **tol)
            relative = float(torch.cat(errors).norm()/torch.cat(values).norm().clamp_min(1e-30))
            return {'forward_gradients_updates_passed': True, 'gradient_relative_l2': relative, 'tolerance': tol}

        def validate(result):
            context = (torch.backends.mkldnn.flags(enabled=False) if training and dtype == torch.float16
                       and workload.startswith('conv_') else nullcontext())
            with context:
                return validate_impl(result)

        return operation, validate, meta

    def measure(device, threads):
        torch.set_num_threads(threads)
        # The AVX2 CPU implementation has no oneDNN half Conv backward. Keep
        # oneDNN on for FP32/default inference and explicitly report the fallback.
        cpu_half_fallback = device == 'cpu' and training and dtype == torch.float16 and workload.startswith('conv_')
        torch.backends.mkldnn.enabled = not cpu_half_fallback
        sync = torch.xpu.synchronize if device == 'xpu' else (npu.synchronize if device == 'intel_npu' else lambda: None)
        sync()
        setup_start = time.perf_counter()
        operation, validate, meta = prepare(device)
        sync()
        setup_ms = (time.perf_counter()-setup_start)*1000
        row = {'device': device, 'cpu_threads': threads, 'mkldnn_enabled': torch.backends.mkldnn.enabled,
               'cpu_half_backward_fallback': cpu_half_fallback, 'setup_ms': setup_ms, **meta}
        before = npu.stats() if device == 'intel_npu' else None
        started = time.perf_counter_ns()
        first = operation()
        sync()
        row['first_call_ms'] = (time.perf_counter_ns()-started)/1e6
        if before is not None:
            after = npu.stats()
            row['first_call_compilations'] = after['session_creations']-before['session_creations']
            row['first_call_npu_calls'] = after['npu_calls']-before['npu_calls']
        row['validation'] = validate(first)
        del first
        for _ in range(3):
            result = operation()
            sync()
            del result
        before = npu.stats() if device == 'intel_npu' else None
        samples = []
        for _ in range(report['measured_calls']):
            sync()
            started = time.perf_counter_ns()
            result = operation()
            sync()
            samples.append((time.perf_counter_ns()-started)/1e6)
            del result
        row['warm'] = summarize(samples)
        if 'effective_flops' in row:
            row['effective_gflops'] = row['effective_flops']/(row['warm']['median_ms']*1e6)
        if before is not None:
            after = npu.stats()
            row['warm_host_payload_copies'] = payload_copies(after, before)
            row['warm_graph_compilations'] = after['session_creations']-before['session_creations']
            row['warm_npu_calls_per_operation'] = (after['npu_calls']-before['npu_calls'])/len(samples)
            row['cached_graphs'] = after['cached_sessions']
            assert row['warm_host_payload_copies'] == row['warm_graph_compilations'] == 0
            if workload in ('mm_256', 'conv_train', 'resnet_train'):
                # A separate profiled call; never mix instrumented samples into
                # the latency results. Native run includes compute/DMA/fences.
                profile = cProfile.Profile()
                original_run = npu._run
                kinds = {}

                def timed_run(key, *inputs, **options):
                    start = time.perf_counter_ns()
                    try:
                        return original_run(key, *inputs, **options)
                    finally:
                        count, elapsed = kinds.get(key[0], (0, 0))
                        kinds[key[0]] = count+1, elapsed+(time.perf_counter_ns()-start)/1e6

                npu._run = timed_run
                try:
                    profile.enable()
                    result = operation()
                    sync()
                finally:
                    profile.disable()
                    npu._run = original_run
                del result
                profile.dump_stats(str(OUT / f'{mode}-{workload}-{precision}.pstats'))
                entries = []
                for (filename, line, name), (_, calls, own, cumulative, _) in pstats.Stats(profile).stats.items():
                    entries.append({'function': name, 'file': filename, 'line': line, 'calls': calls,
                                    'self_ms': own*1000, 'cumulative_ms': cumulative*1000})
                row['profile'] = {'scope': 'separate warm call; profiling overhead applies; native run includes DMA/compute/fences',
                                  'graph_kinds': sorted([{'kind': k, 'calls': c, 'total_ms': t, 'mean_ms': t/c}
                                                        for k, (c, t) in kinds.items()], key=lambda e: e['total_ms'], reverse=True),
                                  'top_self': sorted(entries, key=lambda e: e['self_ms'], reverse=True)[:20],
                                  'top_cumulative': sorted(entries, key=lambda e: e['cumulative_ms'], reverse=True)[:20]}
        # Measuring the explicit input upload and output download separately;
        # these are not included in resident operation timings above.
        if device != 'cpu':
            size = meta['shape']
            input_shape = size[:2] if workload.startswith('mm_') else size
            host = torch.ones(input_shape, dtype=dtype)
            data = host.to(device)
            transfers = {}
            for name, function in (('upload', lambda: host.to(device)), ('download', lambda: data.cpu())):
                timings = []
                for _ in range(7):
                    sync()
                    started = time.perf_counter_ns()
                    value = function()
                    sync()
                    timings.append((time.perf_counter_ns()-started)/1e6)
                    del value
                transfers[name] = summarize(timings)
            row['input_transfer'] = {'shape': input_shape, 'bytes': host.numel()*host.element_size(), **transfers}
        del operation, validate
        gc.collect()
        return row

    try:
        if mode == 'npu':
            # Register PrivateUse1 before the first CPU backward initializes
            # PyTorch's per-device autograd queues.
            save('initializing_npu')
            started = time.perf_counter()
            import torch_intel_npu as npu
            report['backend_initialization_ms'] = (time.perf_counter()-started)*1000
        save('cpu_1_thread')
        if mode == 'npu':
            report['results'].append(measure('cpu', 1))
        save('cpu_8_threads')
        report['results'].append(measure('cpu', min(8, os.cpu_count())))
        if mode == 'npu':
            save('npu_cold_and_warm')
            report['results'].append(measure('intel_npu', 8))
            report['stats_before_shutdown'] = npu.stats()
            assert report['stats_before_shutdown']['live_buffers'] == report['stats_before_shutdown']['cached_scalar_buffers']
            npu.shutdown()
            report['final_stats'] = npu.stats()
            assert report['final_stats']['live_buffers'] == 0 and not report['final_stats']['context_failed']
        else:
            if not torch.xpu.is_available():
                raise RuntimeError('XPU unavailable in the selected comparison environment')
            report['xpu_device'] = torch.xpu.get_device_name(0)
            report['xpu_properties'] = str(torch.xpu.get_device_properties(0))
            save('xpu_cold_and_warm')
            report['results'].append(measure('xpu', 8))
        report['status'] = 'passed'
        save('complete')
        return 0
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        if npu is not None:
            report['failure_stats'] = npu.stats()
        save('failed')
        print(report['traceback'], flush=True)
        return 1


def main():
    global OUT, FIXTURES
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--workload', choices=('all', *WORKLOADS), default='all')
    parser.add_argument('--precision', choices=('all', 'float16', 'float32'), default='all')
    parser.add_argument('--mode', choices=('npu', 'xpu'), default='npu')
    parser.add_argument('--xpu-python')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--fixture-dir', type=Path, default=FIXTURES)
    args = parser.parse_args()
    OUT, FIXTURES = args.output_dir.resolve(), args.fixture_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.child:
        return execute(args.workload, args.precision, args.mode)
    summary = {'status': 'running', 'platform': platform.platform(),
               'started_at': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'timezone': 'Asia/Shanghai',
               'version_comparison': 'NPU/CPU use the extension build version; optional XPU has its own matched CPU baseline',
               'cases': []}
    path = OUT / 'benchmark.json'
    workloads = WORKLOADS if args.workload == 'all' else (args.workload,)
    precisions = ('float16', 'float32') if args.precision == 'all' else (args.precision,)
    modes = ('npu', 'xpu') if args.xpu_python else (args.mode,)
    for workload in workloads:
        for precision in precisions:
            if workload == 'resnet_train' and precision == 'float16':
                continue  # Known deep FP16 accuracy limits; benchmark validated FP32.
            fixture_path = FIXTURES / f'{workload}-{precision}.pt'
            if not fixture_path.is_file():
                import torch
                host, right, model, _ = host_case(workload, precision)
                fixture_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'host': host, 'right': right, 'state': model.state_dict() if model is not None else None}, fixture_path)
            for mode in modes:
                summary['stage'] = f'{mode}-{workload}-{precision}'
                path.write_text(json.dumps(summary, indent=2))
                python = args.xpu_python if mode == 'xpu' and args.xpu_python else sys.executable
                case_path = OUT / f'{mode}-{workload}-{precision}.json'
                if args.resume and case_path.is_file():
                    saved = json.loads(case_path.read_text())
                    fixture_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
                    sources_match = all(saved.get('source_sha256', {}).get(name) == hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                        for name in SOURCES)
                    if saved['status'] == 'passed' and sources_match and all(
                            row.get('fixture_sha256') == fixture_hash for row in saved['results']):
                        summary['cases'].append(saved)
                        print('reusing completed', summary['stage'], flush=True)
                        continue
                env = os.environ.copy()
                if mode == 'xpu' and args.xpu_python:
                    prefix = str(Path(python).parent)
                    env['PATH'] = os.pathsep.join((prefix, str(Path(prefix)/'Library/bin'), str(Path(prefix)/'Scripts'), env['PATH']))
                try:
                    completed = subprocess.run([python, __file__, '--child', '--mode', mode,
                        '--workload', workload, '--precision', precision, '--output-dir', str(OUT),
                        '--fixture-dir', str(FIXTURES)], env=env, capture_output=True, text=True, timeout=600)
                    (OUT / f'{mode}-{workload}-{precision}.log').write_text(completed.stdout+completed.stderr)
                    print(summary['stage'], completed.stdout+completed.stderr, flush=True)
                    case = json.loads((OUT / f'{mode}-{workload}-{precision}.json').read_text())
                    summary['cases'].append(case)
                    if completed.returncode:
                        summary['status'] = 'failed'
                        path.write_text(json.dumps(summary, indent=2))
                        return completed.returncode
                    if mode == 'xpu' and args.xpu_python:
                        baseline = summary['cases'][-2]['results'][0]
                        candidate = case['results'][0]
                        if (baseline['input_sha256'], baseline['model_sha256']) != (
                                candidate['input_sha256'], candidate['model_sha256']):
                            summary.update(status='failed', error='Inputs/model differ between Torch versions')
                            path.write_text(json.dumps(summary, indent=2))
                            return 1
                except subprocess.TimeoutExpired:
                    summary['status'] = 'timeout'
                    path.write_text(json.dumps(summary, indent=2))
                    return 1
    summary['status'] = 'passed'
    path.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

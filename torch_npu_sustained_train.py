"""Larger explicit-attention training with measured Windows NPU engine activity."""
import argparse
import copy
import ctypes as C
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/sustained-training'


def process_memory():
    class Counters(C.Structure):
        _fields_ = [('size', C.c_uint32), ('faults', C.c_uint32)] + [
            (name, C.c_size_t) for name in ('peak_rss', 'rss', 'peak_paged', 'paged',
                                          'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]
    value = Counters()
    value.size = C.sizeof(value)
    kernel, psapi = C.WinDLL('kernel32'), C.WinDLL('psapi')
    kernel.GetCurrentProcess.restype = C.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), C.byref(value), value.size):
        raise C.WinError()
    return {'rss': value.rss, 'peak_rss': value.peak_rss, 'private': value.private}


def engine_interval(first, last):
    elapsed = last['time']-first['time']
    assert elapsed > 0
    result = {}
    for engine, values in first['engines'].items():
        result[engine] = {}
        for name, begin in values.items():
            end = last['engines'][engine][name]
            if begin is not None and end is not None:
                delta = end-begin
                assert delta >= 0, (engine, name, begin, end)
                result[engine][name.replace('_us', '_percent')] = delta/(elapsed*1e6)*100
    return result


def execute(args, npu):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from tools.npu_dxcore import NpuCounters
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.mkldnn.enabled = False
    torch.manual_seed(103)
    dtype = getattr(torch, args.dtype)
    width, heads = args.width, args.heads

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1, self.norm2 = torch.nn.LayerNorm(width), torch.nn.LayerNorm(width)
            self.q, self.k, self.v, self.proj = (torch.nn.Linear(width, width) for _ in range(4))
            self.ff = torch.nn.Sequential(torch.nn.Linear(width, width*4), torch.nn.GELU(),
                                          torch.nn.Linear(width*4, width))

        def forward(self, x):
            h = self.norm1(x)
            q, k, v = [layer(h).view(args.batch, args.sequence, heads, width//heads).transpose(1, 2)
                       for layer in (self.q, self.k, self.v)]
            h = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h.transpose(1, 2).contiguous().view(x.shape)
            h = x+self.proj(h)
            return h+self.ff(self.norm2(h))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.Sequential(*(Block() for _ in range(args.blocks)))
            self.head = torch.nn.Linear(width, 4)

        def forward(self, x):
            return self.head(self.blocks(x).mean(1))

    cpu = Model().to(dtype)
    model = copy.deepcopy(cpu).to('intel_npu')
    host = torch.randn(args.batch, args.sequence, width, dtype=dtype)/3
    targets = torch.arange(args.batch, dtype=torch.int64) % 4
    data, labels = host.to('intel_npu'), targets.to('intel_npu')
    reference, optimizer = (torch.optim.SGD(m.parameters(), lr=.01, foreach=False, fused=False)
                            for m in (cpu, model))
    parameters = sum(p.numel() for p in model.parameters())
    print(f'PID={os.getpid()} model: blocks={args.blocks} width={width} heads={heads} '
          f'batch={args.batch} sequence={args.sequence} parameters={parameters} {args.dtype}', flush=True)
    config = vars(args).copy()
    config['parameters'] = parameters
    config['parameter_bytes'] = sum(p.numel()*p.element_size() for p in model.parameters())
    report = dict(config=config, pid=os.getpid(), warmup=[], steps=[], telemetry=[],
                  phase='warmup', status='running')
    counter = NpuCounters()
    report['counter'] = dict(npu=counter.identity, inventory=counter.inventory,
                             engines=counter.engines, supported=counter.supported)
    stopped = threading.Event()
    sampler = None

    def step(network, opt, x, y):
        opt.zero_grad(set_to_none=True)
        logits = network(x)
        loss = torch.nn.functional.cross_entropy(logits.float(), y)
        loss.backward()
        opt.step()
        return loss.detach()

    def compare(actual, expected):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach(),
                                   rtol=.003 if dtype == torch.float16 else 3e-5,
                                   atol=.0005 if dtype == torch.float16 else 3e-6)

    def sample():
        try:
            while not stopped.wait(1):
                row = counter.read()
                row.update(memory=process_memory(), npu_memory=counter.memory(), cpu_seconds=time.process_time())
                report['telemetry'].append(row)
        except Exception:
            report['telemetry_error'] = traceback.format_exc()

    try:
        for index in range(args.warmup):
            with sdpa_kernel(SDPBackend.MATH):
                expected = step(cpu, reference, host, targets)
            begin = time.perf_counter()
            before = npu.stats()
            actual = step(model, optimizer, data, labels)
            seconds = time.perf_counter()-begin
            compare(actual, expected)
            for a, b in zip(model.parameters(), cpu.parameters()):
                compare(a.grad, b.grad)
                compare(a, b)
            loss_value = actual.cpu().item()
            after = npu.stats()
            report['warmup'].append(dict(step=index+1, seconds=seconds, loss=loss_value,
                npu_calls=after['npu_calls']-before['npu_calls'], cached_sessions=after['cached_sessions']))
            print(f'warmup {index+1}: {seconds:.3f}s loss={loss_value:.8f} '
                  f'sessions={after["cached_sessions"]} CPU loss/gradients/updates agree', flush=True)
            del actual
        del cpu, reference, expected
        gc.collect()
        baseline = npu.stats()
        begin_counter = counter.read()
        begin_memory, begin_cpu = process_memory(), time.process_time()
        report['npu_memory_before'] = counter.memory()
        start = time.perf_counter()
        report['phase'] = 'steady'
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        print(f'STEADY TRAINING START: continuous >= {args.seconds:g}s, NPU={counter.identity["luid"]}', flush=True)
        last_progress = start
        while time.perf_counter()-start < args.seconds:
            before = npu.stats()
            begin = time.perf_counter()
            actual = step(model, optimizer, data, labels)
            npu.synchronize()
            compute_seconds = time.perf_counter()-begin
            after_compute = npu.stats()
            value = actual.cpu().item()
            assert math.isfinite(value), f'Nonfinite loss: {value}'
            del actual
            after = npu.stats()
            assert not after['context_failed']
            assert after['session_creations'] == baseline['session_creations'], 'Graph compilation during steady load'
            assert (after['live_buffers'], after['live_buffer_bytes']) == (
                baseline['live_buffers'], baseline['live_buffer_bytes']), 'Live training buffers drifted'
            scalars = after_compute['cached_scalar_buffers']-before['cached_scalar_buffers']
            copies = sum(after_compute[k]-before[k] for k in
                         ('host_upload_copies', 'host_download_copies', 'host_device_copies'))
            assert copies == scalars+1, (copies, scalars)  # CE reads one four-byte error flag.
            report['steps'].append(dict(step=len(report['steps'])+1, elapsed=time.perf_counter()-start,
                compute_seconds=compute_seconds, loss=value, npu_calls=after['npu_calls']-before['npu_calls'],
                compute_copies=copies, live_buffer_bytes=after['live_buffer_bytes'],
                cached_buffer_bytes=after['cached_buffer_bytes'], memory=process_memory()))
            if time.perf_counter()-last_progress >= 15:
                last_progress = time.perf_counter()
                current = counter.read()
                print(f'{last_progress-start:.1f}s steps={len(report["steps"])} loss={value:.6f} '
                      f'NPU={engine_interval(begin_counter, current)}', flush=True)
        end = time.perf_counter()
        end_counter = counter.read()
        report.update(phase='validated', elapsed_seconds=end-start, stats_before=baseline,
            stats_after=npu.stats(), engine_utilization=engine_interval(begin_counter, end_counter),
            begin_counter=begin_counter, end_counter=end_counter,
            process_cpu_seconds=time.process_time()-begin_cpu, memory_before=begin_memory,
            memory_after=process_memory(), npu_memory_after=counter.memory())
        stopped.set()
        sampler.join()
        assert not report.get('telemetry_error'), report.get('telemetry_error')
        assert report['steps'] and report['steps'][-1]['loss'] < report['warmup'][-1]['loss'], 'Loss did not decrease after warmup'
        for parameter in model.parameters():
            assert torch.isfinite(parameter.detach().cpu()).all(), 'Nonfinite parameter'
            assert torch.isfinite(parameter.grad.detach().cpu()).all(), 'Nonfinite final gradient'
        report['final_parameters_and_gradients_finite'] = True
        report['status'] = 'passed'
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        print(report['error'], flush=True)
    finally:
        stopped.set()
        if sampler is not None:
            sampler.join()
        counter.close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--sequence', type=int, default=16)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--dtype', choices=('float16', 'float32'), default='float32')
    parser.add_argument('--label', default='run')
    args = parser.parse_args()
    assert all(v > 0 for v in (args.width, args.heads, args.blocks, args.batch, args.sequence, args.seconds))
    assert args.width % args.heads == 0 and args.warmup >= 2
    assert args.batch*args.heads*args.sequence**2 <= 65536
    assert max(args.batch*args.sequence*args.width*4, args.width**2*4) <= 1048576
    assert args.label.replace('-', '').replace('_', '').isalnum()
    import torch_intel_npu as npu
    report = execute(args, npu)
    gc.collect()
    report['before_shutdown'] = npu.stats()
    npu.shutdown()
    report['final_stats'] = npu.stats()
    assert report['final_stats']['closed'] and not report['final_stats']['context_failed']
    assert all(report['final_stats'][key] == 0 for key in
               ('live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
    report.update(python=sys.version, source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('torch_npu_sustained_train.py', 'tools/npu_dxcore.py', 'torch_intel_npu.py', 'torch_npu_autograd.py')})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{args.label}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report.get(key) for key in ('status', 'config', 'elapsed_seconds', 'engine_utilization')}, indent=2), flush=True)
    raise SystemExit(report['status'] != 'passed')

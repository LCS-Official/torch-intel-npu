"""Matched resident GEMM throughput/energy workload; frozen DPU graph, no training."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/gemm-energy'
FLAGS = {'default': '--config EXECUTION_MODE_HINT="ACCURACY"',
         'tiles6': '--config EXECUTION_MODE_HINT="ACCURACY" PERFORMANCE_HINT="LATENCY" NPU_TILES="6"'}


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{name}.json').write_text(json.dumps(data, indent=2), encoding='utf-8')


def fixture():
    import numpy as np
    import torch
    torch.set_num_threads(8)
    random = np.random.default_rng(20261019)
    a = random.integers(-64, 65, (4, 1024, 1024), dtype=np.int8)
    w = random.integers(-63, 64, (1024, 1024), dtype=np.int8)
    ref = ((torch.from_numpy(a.reshape(-1, 1024)).float() @ torch.from_numpy(w).float().t())/32768).half().numpy().reshape(4, 1024, 1024)
    for group in range(4):
        exact = np.einsum('mk,nk->mn', a[group, [0, 512, 1023]].astype(np.int64), w.astype(np.int64), optimize=False)
        np.testing.assert_array_equal(ref[group, [0, 512, 1023]], (exact/32768).astype(np.float16))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/'fixture.npz', a=a, w=w, expected=ref)
    save('fixture', dict(seed=20261019, sha256=hashlib.sha256((OUT/'fixture.npz').read_bytes()).hexdigest(),
                         shape=[4, 1024, 1024], independent_int64_rows=[0, 512, 1023]))
    print('Fixture prepared and independent INT64 reference rows checked', flush=True)


def npu_case(torch, a, w, expected, m, layout, profile, precision):
    import numpy as np
    import npunlock as symbolic
    import torch_intel_npu as npu
    from torch_npu_lowbit import PackedWeight, weight_graph, dpu_modes
    groups = 1 if layout == 'single' else 4
    x = [torch.from_numpy(a[i, :m].copy()).float().div(128).half().to('intel_npu') for i in range(groups)]
    y = [torch.empty((m, 1024), dtype=torch.float16, device='intel_npu') for _ in range(groups)]
    if precision == 'int8' and layout != 'fused4':
        q = PackedWeight((1024, 1024), 'int8', 1024, w.tobytes(), np.full(1024, 1/256, dtype=np.float16).tobytes())
        graph, xml, weights = weight_graph(q, m, activation_scale=1/128)
    else:
        xs = [symbolic.input(f'x{i}', shape=(m, 1024), dtype='f16') for i in range(groups if layout == 'fused4' else 1)]
        constant = symbolic.constant((w.astype(np.float32)/256).astype(np.float16))
        def quantized(tensor, low, high):
            lo, hi = (symbolic.constant(np.asarray(v, dtype=np.float16)) for v in (low, high))
            return symbolic.op('FakeQuantize', tensor, lo, hi, lo, hi, levels=256,
                               auto_broadcast='numpy', _shape=tensor.shape, _dtype='f16')
        if precision == 'int8':
            constant = quantized(constant, -128/256, 127/256)
            xs_compute = [quantized(t, -1, 127/128) for t in xs]
        else:
            xs_compute = xs
        ys = [symbolic.op('MatMul', t, constant, transpose_a=False, transpose_b=True,
                         _shape=(m, 1024), _dtype='f16', _name=f'y{i}') for i, t in enumerate(xs_compute)]
        graph = symbolic.Graph(xs, ys, name='gemm_energy')
        ir = symbolic.serialize_ir(graph)
        xml, weights = ir.xml, ir.weights
    if isinstance(xml, str):
        xml = xml.encode()
    compiled = npu._native.compile_ir(xml, weights, build_flags=FLAGS[profile], timeout_ms=30000)
    assert (compiled.vendor_id, compiled.device_id, tuple(compiled.compiler_version)) == (0x8086, 0x643e, (8, 3))
    modes = [v for v in dpu_modes(compiled.graph_blob) if not v['elementwise']]
    wanted = (1, 1) if precision == 'int8' else (0, 0)
    assert modes and all((v['amode'], v['wmode']) == wanted for v in modes), modes
    program = symbolic.load_native(compiled.graph_blob, graph=graph, libraries=npu._native)
    key = ('energy', m, layout, profile, precision)
    copies = []
    for i in range(4 if layout == 'chain4' else 1):
        session = npu._native.create_inference_session(program.graph_blob, timeout_ms=10000, parent=npu._root_session)
        npu._sessions[key+(i,)] = session, program
        npu._session_creations += 1
        copies.append(session._result.session)
    names, output_names = [t.name for t in graph.inputs], [t.name for t in graph.outputs]
    repeats = 16
    if layout == 'fused4':
        sessions = copies*repeats
        inputs, outputs = [x]*repeats, [y]*repeats
    else:
        sessions = copies*repeats
        inputs = [[t] for _ in range(repeats) for t in x]
        outputs = [[t] for _ in range(repeats) for t in y]
    bindings = [names]*len(sessions)
    outbindings = [output_names]*len(sessions)
    group_size = 4 if layout == 'chain4' else 1

    def run():
        with npu._lock:
            npu._C.run_chain(sessions, inputs, bindings, outputs, outbindings, group_size)

    def validate():
        for i, tensor in enumerate(y):
            actual = tensor.cpu()
            reference = torch.from_numpy(expected[i, :m].copy())
            torch.testing.assert_close(actual, reference, rtol=.003, atol=.0003)
            relative = float((actual.float()-reference.float()).norm()/reference.float().norm())
            assert relative < .001, relative
        return dict(relative_l2_max=max(float((v.cpu().float()-torch.from_numpy(expected[i, :m].copy()).float()).norm()/
                                              torch.from_numpy(expected[i, :m].copy()).float().norm()) for i, v in enumerate(y)))

    return run, validate, groups*repeats, dict(m=m, k=1024, n=1024, groups=groups, layout=layout, profile=profile,
        precision=precision, dpu_modes=modes, flags=FLAGS[profile], blob_sha256=hashlib.sha256(compiled.graph_blob).hexdigest())


def host_case(torch, a, w, expected, args):
    device = args.device
    dtype = torch.float16 if args.precision == 'float16' else torch.float32
    groups = 1 if args.layout == 'single' else 4
    host = torch.from_numpy(a[:groups, :args.m].copy()).reshape(-1, 1024).float()/128
    weight = torch.from_numpy(w.copy()).float().t().contiguous()/256
    x, b = host.to(device=device, dtype=dtype), weight.to(device=device, dtype=dtype)
    raw = torch.empty((groups*args.m, 1024), device=device, dtype=dtype)
    output = torch.empty_like(raw, dtype=torch.float16) if dtype == torch.float32 else raw
    repeats = 16

    def run():
        for _ in range(repeats):
            torch.mm(x, b, out=raw)
            if output is not raw:
                output.copy_(raw)
        if device == 'xpu':
            torch.xpu.synchronize()

    def validate():
        reference = torch.from_numpy(expected[:groups, :args.m].copy()).reshape(-1, 1024)
        actual = output.cpu()
        torch.testing.assert_close(actual, reference, rtol=.003, atol=.0003)
        relative = float((actual.float()-reference.float()).norm()/reference.float().norm())
        assert relative < .001
        return dict(relative_l2_max=relative)

    return run, validate, groups*repeats, dict(m=args.m, k=1024, n=1024, groups=groups,
        layout='flat-equivalent-gemm', precision=args.precision, torch=str(torch.__version__), cpu_threads=8,
        device_name=torch.xpu.get_device_name(0) if device == 'xpu' else 'CPU',
        final_fp16_conversion_included=dtype == torch.float32)


def measured(run, validate, count, meta, seconds, npu=None):
    for _ in range(3):
        run()
    validation = validate()
    counter = None
    if npu:
        from tools.npu_dxcore import NpuCounters
        counter = NpuCounters()
    before = npu.stats() if npu else None
    first = counter.read() if counter else None
    start = time.time_ns()
    begin = time.perf_counter()
    chunks = []
    while time.perf_counter()-begin < seconds:
        tick = time.perf_counter()
        run()
        chunks.append(time.perf_counter()-tick)
    elapsed = time.perf_counter()-begin
    end = time.time_ns()
    last = counter.read() if counter else None
    after = npu.stats() if npu else None
    if counter:
        counter.close()
    validate()
    if npu:
        for name in ('host_upload_copies', 'host_download_copies', 'host_device_copies',
                     'session_creations', 'driver_allocations', 'live_buffers', 'live_buffer_bytes'):
            assert before[name] == after[name], (name, before[name], after[name])
        graphs_per_chunk = 16 if meta['layout'] == 'fused4' else count
        assert after['npu_calls']-before['npu_calls'] == len(chunks)*graphs_per_chunk
    ops = 2*meta['m']*1024*1024
    completed = len(chunks)*count
    return dict(status='passed', meta=meta, validation=validation, start_unix_ns=start, end_unix_ns=end,
                seconds=elapsed, completed_gemms=completed, gemms_per_second=completed/elapsed,
                teraops=ops*completed/elapsed/1e12, chunks=chunks, gemms_per_chunk=count,
                stats_before=before, stats_after=after, npu_counter_before=first, npu_counter_after=last)


def execute(args):
    import numpy as np
    import torch
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    payload = np.load(OUT/'fixture.npz')
    a, w, expected = (payload[key] for key in ('a', 'w', 'expected'))
    npu = None
    if args.device == 'npu':
        import torch_intel_npu as npu
    report = dict(status='running', label=args.label, rows=[], device=args.device,
                  fixture_sha256=hashlib.sha256((OUT/'fixture.npz').read_bytes()).hexdigest(),
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                      ('torch_npu_gemm_energy.py', 'torch_intel_npu.py', 'torch_npu_lowbit.py')})
    try:
        with torch.inference_mode():
            if args.calibrate:
                cases = [(m, layout, 'default', 'float16') for m in (128, 512, 1024) for layout in ('single','chain4','fused4')]
                cases += [(1024, layout, 'tiles6', precision) for precision in ('float16','int8') for layout in ('single','chain4','fused4')]
                cases += [(1024, 'chain4', 'default', 'int8')]
                cases += [(1024, 'fused4', 'default', 'int8')]
                for m, layout, profile, precision in cases:
                    print('calibrating', m, layout, profile, precision, flush=True)
                    try:
                        run, validate, count, meta = npu_case(torch, a, w, expected, m, layout, profile, precision)
                    except Exception as error:
                        if profile != 'tiles6':
                            raise
                        report['rows'].append(dict(status='unsupported-profile', m=m, layout=layout,
                            profile=profile, precision=precision, error=str(error)))
                        continue
                    row = measured(run, validate, count, meta, .25, npu)
                    report['rows'].append(row)
                    print('result', {k: row[k] for k in ('gemms_per_second','teraops')}, flush=True)
                    del run, validate
                    gc.collect()
                    save(args.label, report)
            else:
                print(f'prepare {args.device} {args.precision} {args.layout} {args.profile}', flush=True)
                spec = (npu_case(torch, a, w, expected, args.m, args.layout, args.profile, args.precision) if npu else
                        host_case(torch, a, w, expected, args))
                run, validate, count, meta = spec
                print('steady start', args.label, flush=True)
                row = measured(run, validate, count, meta, args.seconds, npu)
                report['rows'].append(row)
                print('result', {k: row[k] for k in ('gemms_per_second','teraops','seconds')}, flush=True)
                del run, validate, spec
        report['status'] = 'passed'
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        print(report['error'], flush=True)
    if npu:
        run = validate = spec = None
        gc.collect()
        report['before_shutdown'] = npu.stats()
        npu.shutdown()
        report['final_stats'] = npu.stats()
        assert report['final_stats']['closed'] and not report['final_stats']['context_failed']
        assert all(report['final_stats'][k] == 0 for k in ('live_buffers','live_buffer_bytes','cached_sessions',
            'cached_scalar_buffers','cached_buffer_blocks','quarantined_buffer_bytes'))
    save(args.label, report)
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--device', choices=('cpu','xpu','npu'), default='npu')
    parser.add_argument('--precision', choices=('float16','float32','int8'), default='float16')
    parser.add_argument('--layout', choices=('single','chain4','fused4'), default='chain4')
    parser.add_argument('--profile', choices=tuple(FLAGS), default='default')
    parser.add_argument('--m', type=int, choices=(128,512,1024), default=1024)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--label', default='calibration')
    args = parser.parse_args()
    assert args.seconds > 0 and args.label.replace('-', '').isalnum()
    assert args.device == 'npu' or args.precision in ('float16','float32')
    if args.prepare:
        fixture()
    else:
        raise SystemExit(execute(args))

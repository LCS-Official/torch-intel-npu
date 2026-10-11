"""Measured useful GEMM throughput, not a hardware peak or device timestamp."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'npunlock-probe/research/results/throughput-20261005'
SOURCES = ('torch_npu_throughput_benchmark.py', 'torch_intel_npu.py',
           'torch_npu_rect_gemm.py', 'torch_npu_lowbit.py',
           'npunlock/python/npunlock/npu4000.py',
           'npunlock/tests/fixtures/npu4000/rect_gemm.c',
           'npunlock/tests/fixtures/npu4000/gemm_eight/rect-f32.elf',
           'npunlock-probe/torch-device/_torch_intel_npu.pyd')


def timing(samples, ops):
    median = statistics.median(samples)
    return dict(median_ms=median, min_ms=min(samples), max_ms=max(samples),
                p25_ms=statistics.quantiles(samples, n=4, method='inclusive')[0],
                p75_ms=statistics.quantiles(samples, n=4, method='inclusive')[2],
                effective_teraops_per_second=ops / (median * 1e9), samples_ms=samples)


def execute(case, out):
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from torch_npu_lowbit import PackedWeight, compile_weight
    from torch_npu_rect_gemm import _kernel
    from npunlock.npu4000 import compile_rect_gemm, rect_gemm_capacity

    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(20261005)
    report = dict(status='running', case=case, started_at=datetime.now().astimezone().isoformat(),
                  rows=[], environment=npu.stats(), source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                  count_convention='2*M*K*N useful operations; excludes padding, replicas, quantization, compensation and packing',
                  timing_scope='synchronous warm wall time including graph DMA and fence; excludes compilation, input upload, validation and output download',
                  native_bound_scope='one graph, preallocated input/output, direct existing C++ binding; not device-only kernel time',
                  load_control='desktop workload, power mode and device clock not locked',
                  warmup_calls=5, rounds=31 if case != 'fp32-api' else 11)

    def save():
        (out / f'{case}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    def register(key, program):
        session = npu._native.create_inference_session(program.graph_blob, timeout_ms=10000, parent=npu._root_session)
        npu._sessions[key] = session, program
        npu._session_creations += 1
        (out / f'{case}-{key[-1]}.blob').write_bytes(program.graph_blob)
        return session

    def bound(session, program, inputs, output):
        names = [v.name for v in program.graph.inputs]
        def run():
            # Existing binding checks device, extents and aliasing and waits on
            # the completion fence. No asynchronous work escapes the timer.
            with npu._lock:
                npu._C.run(session._result.session, inputs, names, output, program.graph.outputs[0].name)
            return output
        return run

    def measure(runs, ops):
        for run in runs.values():
            for _ in range(5):
                value = run()
            del value
        before = npu.stats()
        samples = {name: [] for name in runs}
        names = tuple(runs)
        for i in range(report['rounds']):
            order = names[i % len(names):] + names[:i % len(names)]
            if i % 2:
                order = tuple(reversed(order))
            for name in order:
                start = time.perf_counter_ns()
                value = runs[name]()
                samples[name].append((time.perf_counter_ns()-start)/1e6)
                del value
        after = npu.stats()
        delta = {k: after[k]-before[k] for k in ('npu_calls', 'session_creations', 'host_upload_copies', 'host_download_copies', 'host_device_copies', 'buffer_allocations')}
        assert all(delta[k] == 0 for k in ('session_creations', 'host_upload_copies', 'host_download_copies', 'host_device_copies')), delta
        return {name: timing(values, ops) for name, values in samples.items()}, delta

    def dpu(m):
        k = n = 1024
        ac = torch.randint(-64, 65, (m, k), dtype=torch.int32)
        wc = torch.randint(-63, 64, (n, k), dtype=torch.int32)
        host = (ac.float()/128).half()
        q = PackedWeight((n, k), 'int8', k, wc.to(torch.int8).numpy().tobytes(), np.full(n, 1/256, dtype=np.float16).tobytes())
        expected = ((ac.float() @ wc.float().t())/32768).half()
        subset = sorted(set((0, m//2, m-1)))
        independent = ac.numpy()[subset].astype(np.int64) @ wc.numpy().astype(np.int64).T
        np.testing.assert_array_equal(expected[subset].numpy(), (independent/32768).astype(np.float16))
        x = host.to('intel_npu')
        runs, outputs, evidence, programs = {}, {}, {}, {}
        for tag in ('fp16-dpu', 'w8a16-dpu', 'int8-w8a8-dpu'):
            program, detail = compile_weight(q, m, npu._native, reference_fp16=tag=='fp16-dpu', activation_scale=1/128 if tag=='int8-w8a8-dpu' else None)
            modes = [v for v in detail['dpu_modes'] if not v['elementwise']]
            required = (0, 0) if tag=='fp16-dpu' else ((0, 2) if tag=='w8a16-dpu' else (1, 1))
            assert modes and all((v['amode'], v['wmode'])==required for v in modes), (tag, modes)
            key = ('throughput', m, tag)
            session = register(key, program)
            output = torch.empty((m, n), dtype=torch.float16, device='intel_npu')
            outputs[tag] = output
            runs[tag] = bound(session, program, [x], output)
            programs[tag] = key
            actual = runs[tag]().cpu()
            torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-4)
            error = float((actual.float()-expected.float()).norm()/expected.float().norm())
            assert error < 1e-3, error
            evidence[tag] = dict(**detail, validation_relative_l2=error, independent_int64_reference_rows=subset, validation_passed=True)
        stats, delta = measure(runs, 2*m*k*n)
        assert delta['npu_calls']==report['rounds']*len(runs) and delta['buffer_allocations']==0, delta
        eager = {tag: (lambda tag=tag: npu._run(programs[tag], [x], (m,n), torch.float16)) for tag in runs}
        eager_stats, eager_delta = measure(eager, 2*m*k*n)
        assert eager_delta['npu_calls']==report['rounds']*len(eager) and eager_delta['buffer_allocations']==report['rounds']*len(eager), eager_delta
        row = dict(shape_mkn=[m,k,n], useful_operations=2*m*k*n, native_bound=stats, resident_graph_api=eager_stats,
                   native_bound_counter_delta=delta, graph_api_counter_delta=eager_delta, evidence=evidence)
        report['rows'].append(row)
        print(json.dumps(dict(case=case, shape=row['shape_mkn'], bound_ms={t:v['median_ms'] for t,v in stats.items()}, bound_teraops={t:v['effective_teraops_per_second'] for t,v in stats.items()})), flush=True)

    def fp32_raw(shape):
        m,k,n = shape
        capacity = rect_gemm_capacity(shape)
        a, b = torch.randn(m,k)/8, torch.randn(k,n)/8
        packed_a = np.zeros((8,capacity), dtype=np.float32)
        packed_b = np.zeros_like(packed_a)
        packed_a[:,:4] = [12345,m,k,n]
        packed_a[:,4:4+m*k] = a.numpy().reshape(-1)
        packed_b[:,:k*n] = b.numpy().reshape(-1)
        layout = (8,8,capacity//8)
        inputs = [torch.from_numpy(v.reshape(layout)).to('intel_npu') for v in (packed_a,packed_b)]
        program = compile_rect_gemm(shape, _kernel(8), libraries=npu._native, replicas=8)
        key = ('throughput-fp32', '-'.join(map(str,shape)))
        session = register(key,program)
        output = torch.empty(layout, dtype=torch.float32, device='intel_npu')
        run = bound(session,program,inputs,output)
        raw = run().cpu().numpy().reshape(8,capacity)
        indices = np.arange(m*n).reshape(m,n)
        actual = torch.from_numpy(raw[np.arange(m)[:,None]%8, indices].copy())
        expected = a.double() @ b.double()
        torch.testing.assert_close(actual.double(),expected,rtol=2e-5,atol=2e-7)
        stats,delta = measure({'fp32-shave-packed-8':run},2*m*k*n)
        assert delta['npu_calls']==report['rounds'] and delta['buffer_allocations']==0,delta
        row = dict(shape_mkn=list(shape),useful_operations=2*m*k*n,native_bound=stats,counter_delta=delta,
                   validation_passed=True,validation_relative_l2=float((actual.double()-expected).norm()/expected.norm()),
                   output_scope='eight partial row replicas; row selection/reassembly excluded',packed_capacity=capacity,
                   patch_evidence=[json.loads(v) for v in program.patch_reports])
        report['rows'].append(row)
        print(json.dumps(dict(case=case,shape=list(shape),result=stats['fp32-shave-packed-8']['median_ms'],teraops=stats['fp32-shave-packed-8']['effective_teraops_per_second'])),flush=True)

    def fp32_api(size):
        a,b = torch.randn(size,size)/8,torch.randn(size,size)/8
        x,y = a.to('intel_npu'),b.to('intel_npu')
        actual=(x@y).cpu()
        expected=a.double()@b.double()
        torch.testing.assert_close(actual.double(),expected,rtol=2e-5,atol=2e-7)
        stats,delta=measure({'fp32-torch-mm':lambda:x@y},2*size**3)
        report['rows'].append(dict(shape_mkn=[size]*3,useful_operations=2*size**3,resident_graph_api=stats,
                                  counter_delta=delta,validation_passed=True,validation_relative_l2=float((actual.double()-expected).norm()/expected.norm())))
        print(json.dumps(dict(case=case,size=size,median_ms=stats['fp32-torch-mm']['median_ms'],teraops=stats['fp32-torch-mm']['effective_teraops_per_second'])),flush=True)

    try:
        with torch.no_grad():
            if case.startswith('dpu-'):
                dpu(int(case.split('-')[1]))
            elif case=='fp32-raw':
                for shape in ((32,64,32),(64,128,64),(56,256,64)):
                    fp32_raw(shape)
                    save()
            elif case=='fp32-api':
                for size in (128,256):
                    fp32_api(size)
                    save()
            else:
                raise ValueError(case)
        gc.collect()
        report['stats_before_shutdown']=npu.stats()
        assert npu.stats()['live_buffers']==npu.stats()['cached_scalar_buffers'],npu.stats()
        npu.shutdown()
        report['final_stats']=npu.stats()
        assert npu.stats()['live_buffers']==0 and not npu.stats()['context_failed'],npu.stats()
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats())
        print(report['traceback'],flush=True)
    save()
    return int(report['status']!='passed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--child',action='store_true')
    parser.add_argument('--case',default='all',choices=('all','fp32-raw','fp32-api',*(f'dpu-{m}' for m in (1,32,128,512,1024))))
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.child:
        return execute(args.case,out)
    cases=('fp32-raw','fp32-api',*(f'dpu-{m}' for m in (1,32,128,512,1024))) if args.case=='all' else (args.case,)
    summary=dict(status='running',cases=[])
    for case in cases:
        print('starting '+case,flush=True)
        with (out/f'{case}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--child','--case',case,'--output-dir',str(out)],stdout=log,stderr=log,timeout=300)
        print((out/f'{case}.log').read_text(encoding='utf-8'),flush=True)
        saved=json.loads((out/f'{case}.json').read_text(encoding='utf-8')) if (out/f'{case}.json').exists() else dict(status='failed',error='no report')
        summary['cases'].append(dict(exit_code=result.returncode,**saved))
        (out/'benchmark.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        if result.returncode:
            return result.returncode
    summary['status']='passed'
    (out/'benchmark.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return 0


if __name__=='__main__':
    raise SystemExit(main())

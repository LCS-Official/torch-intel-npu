"""Isolated rectangular GEMM precision, tiled sizes, training and timing checks."""
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

ROOT = Path(__file__).resolve().parent
EIGHT = '--four' not in sys.argv
OUT = ROOT/'npunlock-probe/research/results/gemm-eight'
if not EIGHT: OUT = OUT/'four'


def execute(case):
    import numpy as np
    import torch
    import torch_intel_npu as npu
    import torch_npu_rect_gemm as rect
    from functools import partial
    rectangle = partial(rect.matrix_fp32,replicas=8 if EIGHT else 4)
    import importlib.util
    previous_path = ROOT/'npunlock-probe/research/gemm-eight-before/torch_npu_rect_gemm.py'
    previous_spec = importlib.util.spec_from_file_location('previous_rect_gemm',previous_path)
    previous = importlib.util.module_from_spec(previous_spec)
    previous_spec.loader.exec_module(previous)
    torch.manual_seed(311)
    default_matrix = npu._matrix_fp32
    old_matrix = getattr(npu, '_matrix_fp32_expanded', default_matrix)
    report = dict(case=case, status='running', checks=[], timings=[], training=[],
        sources={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('torch_intel_npu.py', 'torch_npu_rect_gemm.py', 'torch_npu_rect_gemm_demo.py',
             'npunlock/python/npunlock/npu4000.py')},
        kernel_manifest=json.loads((ROOT/f'npunlock/tests/fixtures/npu4000/{"gemm_eight" if EIGHT else "gemm_rect"}/manifest.json').read_text()),
        previous_host_source_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),replicas=8 if EIGHT else 4,
        desktop_load='uncontrolled', timing_scope='resident synchronous end-to-end including NPU packing/tiling')

    def arrays_equal(a, b):
        np.testing.assert_array_equal(a.numpy().view(np.uint32 if a.dtype == torch.float32 else np.uint16),
                                      b.numpy().view(np.uint32 if b.dtype == torch.float32 else np.uint16))

    def check(shape, pattern='finite', dtype=torch.float32, transpose=False):
        m, k, n = shape
        a, b = torch.randn(m, k)/8, torch.randn(k, n)/8
        if pattern == 'precision':
            a.fill_(1.0003); b.fill_(1.0003)
        elif pattern == 'cancellation':
            a.fill_(1); a[:, :3] = torch.tensor([1e10, 1, -1e10]); b.fill_(1)
        elif pattern in ('subnormal', 'ieee_edges', 'random_bits'):
            if pattern == 'random_bits':
                rng = np.random.default_rng(313+k)
                a, b = (torch.from_numpy(rng.integers(0, 2**32, size=s, dtype=np.uint32).view(np.float32))
                        for s in ((m, k), (k, n)))
            else:
                bits = [1, 0x007fffff, 0x00800000, 0x80000001, 0x807fffff, 0x80800000]
                if pattern == 'ieee_edges':
                    bits += [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x7f7fffff]
                a = torch.from_numpy(np.resize(np.array(bits, dtype=np.uint32).view(np.float32), m*k).reshape(m,k).copy())
                b = torch.eye(k, n)
        a, b = a.to(dtype), b.to(dtype)
        if transpose:
            x, y = (v.t().contiguous().to('intel_npu').t() for v in (a, b))
        else:
            x, y = (v.to('intel_npu') for v in (a, b))
        expected = old_matrix(x, y).cpu()
        before = npu.stats()
        actual = rectangle(x, y).cpu()
        arrays_equal(expected, actual)
        after = npu.stats()
        metadata_uploads = after['host_upload_copies']-before['host_upload_copies']
        assert metadata_uploads == after['cached_scalar_buffers']-before['cached_scalar_buffers']
        entry = dict(shape=shape, pattern=pattern, dtype=str(dtype), transpose=transpose,
            status='passed', old_npu_bitwise_equal=True, graph_calls=after['npu_calls']-before['npu_calls'],
            cold_scalar_metadata_uploads=metadata_uploads)
        if pattern not in ('random_bits', 'ieee_edges'):
            reference = a.double()@b.double()
            np.testing.assert_allclose(actual.numpy(), reference.float().numpy(), rtol=2e-5, atol=2e-6)
            entry['max_abs_error_fp64_reference'] = (actual.double()-reference).abs().max().item()
        report['checks'].append(entry)
        print('check', entry, flush=True)

    def measure(shape):
        m, k, n = shape
        a, b = torch.randn(m, k)/8, torch.randn(k, n)/8
        x, y = a.to('intel_npu'), b.to('intel_npu')
        versions = ('old','previous_four','cached_four','eight')
        outputs = {v: torch.empty((m,n), device='intel_npu') for v in versions}
        run = {'old': lambda: old_matrix(x,y,out=outputs['old']),
               'previous_four': lambda: previous.matrix_fp32(x,y,out=outputs['previous_four']),
               'cached_four': lambda: rect.matrix_fp32(x,y,out=outputs['cached_four'],replicas=4),
               'eight': lambda: rect.matrix_fp32(x,y,out=outputs['eight'],replicas=8)}
        calls = {}
        for version in run:
            for _ in range(2): run[version]()
            before = npu.stats(); run[version]()
            calls[version] = npu.stats()['npu_calls']-before['npu_calls']
        before = npu.stats()
        samples = {v: [] for v in (*versions,'cpu')}
        cpu_out = torch.empty((m,n))
        for i in range(7):
            order = versions[i % len(versions):]+versions[:i % len(versions)]
            for v in (order if i % 2 == 0 else tuple(reversed(order))):
                start = time.perf_counter(); run[v]()
                samples[v].append(1000*(time.perf_counter()-start))
            start = time.perf_counter(); torch.mm(a,b,out=cpu_out)
            samples['cpu'].append(1000*(time.perf_counter()-start))
        after = npu.stats()
        for field in ('host_upload_copies','host_download_copies','host_device_copies','session_creations'):
            assert before[field] == after[field], field
        expected=outputs['old'].cpu()
        for v in versions[1:]: arrays_equal(expected,outputs[v].cpu())
        medians = {v: statistics.median(s) for v,s in samples.items()}
        entry = dict(shape=shape, median_ms=medians, samples_ms=samples, graph_calls=calls,
            speedup=medians['old']/medians['eight'],speedup_vs_previous_four=medians['previous_four']/medians['eight'],
            warm_host_copies_and_compiles=0)
        report['timings'].append(entry)
        print('timing', {key:value for key,value in entry.items() if key!='samples_ms'}, flush=True)

    def public_check(size, dtype):
        a,b = (torch.randn(size,size,dtype=dtype)/8 for _ in range(2))
        x,y = a.to('intel_npu'),b.to('intel_npu')
        expected = npu._cast(old_matrix(x,y),dtype).cpu()
        out = torch.empty_like(x)
        before = npu.stats()
        assert torch.mm(x,y,out=out) is out
        calls = npu.stats()['npu_calls']-before['npu_calls']
        arrays_equal(expected,out.cpu())
        arrays_equal(expected,(x@y).cpu())
        before = npu.stats()['npu_calls']
        try:
            torch.mm(x,y,out=x)
            raise AssertionError('Input/output alias accepted')
        except ValueError:
            pass
        assert npu.stats()['npu_calls']==before
        entry = dict(shape=[size]*3,dtype=str(dtype),status='passed',
            public_mm_matmul_out_bitwise_equal=True,graph_calls=calls,alias_rejected=True)
        report['checks'].append(entry); print('dispatch',entry,flush=True)

    def training(dtype, tiled=False, default=False):
        batch, features, hidden, outputs = (128,128,128,128) if default else ((65,96,67,13) if tiled else (16,48,24,7))
        def new_model():
            if default:
                return torch.nn.Sequential(torch.nn.Linear(features,outputs,dtype=dtype))
            return torch.nn.Sequential(torch.nn.Linear(features,hidden,dtype=dtype), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden,outputs,dtype=dtype))
        initial = new_model().state_dict()
        host, goal = torch.randn(batch,features,dtype=dtype)/8, torch.randn(batch,outputs,dtype=dtype)/8
        saved = []
        for version in ('cpu','old','default' if default else 'rect'):
            model = new_model(); model.load_state_dict(initial)
            device = 'cpu' if version=='cpu' else 'intel_npu'
            model.to(device)
            x = host.to(device).detach().requires_grad_(); target = goal.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            npu._matrix_fp32 = default_matrix if version=='default' else (rectangle if version=='rect' else old_matrix)
            states = []
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True); x.grad=None
                difference = model(x)-target
                loss = (difference*difference).mean(); loss.backward()
                state = [v.detach().cpu().clone() for v in
                         (loss, x.grad, *[p.grad for p in model.parameters()])]
                optimizer.step(); state += [p.detach().cpu().clone() for p in model.parameters()]
                states.append(state)
            saved.append(states)
        npu._matrix_fp32 = default_matrix
        max_error = 0
        for cpu_step, old_step, new_step in zip(*saved):
            for cpu, old, new in zip(cpu_step, old_step, new_step):
                arrays_equal(old,new)
                tol = 3e-3 if dtype==torch.float16 else 1e-5
                torch.testing.assert_close(new,cpu,rtol=tol,atol=tol)
                max_error = max(max_error,(new.float()-cpu.float()).abs().max().item())
        entry = dict(dtype=str(dtype), steps=3,
            network=f'{features}->{outputs} Linear, batch{batch}' if default else f'{features}->{hidden}->{outputs} ReLU MLP, batch{batch}',
            status='passed', default_dispatch=default,
            loss_gradients_updates_bitwise_equal=True, max_abs_error_cpu=max_error)
        report['training'].append(entry); print('training',entry,flush=True)

    try:
        with torch.no_grad():
            if case == 'smoke': check((3,5,7)); check((16,48,24))
            if case == 'numeric':
                for shape in ((1,1,1),(3,5,7),(7,17,5),(16,48,24),(24,16,48),(63,256,64),(65,33,67)):
                    for dtype in (torch.float32,torch.float16): check(shape,dtype=dtype)
                check((19,65,13),transpose=True)
                for shape in ((3,5,7),(16,48,24),(63,256,64)):
                    for pattern in ('precision','subnormal','cancellation','ieee_edges','random_bits'):
                        check(shape,pattern)
            if case == 'large':
                for size in (128,256):
                    for dtype in (torch.float32,torch.float16): check((size,size,size),dtype=dtype)
            if case == 'performance':
                for shape in ((16,48,24),(128,96,48),(128,128,128),(256,256,256)): measure(shape)
            if case == 'dispatch':
                for size in (128,256):
                    for dtype in (torch.float32,torch.float16): public_check(size,dtype)
        if case in ('training','training-tiled'):
            for dtype in (torch.float32,torch.float16): training(dtype,tiled=case=='training-tiled')
        if case == 'dispatch':
            for dtype in (torch.float32,torch.float16): training(dtype,default=True)
        gc.collect()
        report['stats_before_shutdown'] = npu.stats()
        assert report['stats_before_shutdown']['live_buffers']==report['stats_before_shutdown']['cached_scalar_buffers']
        npu.shutdown(); report['final_stats']=npu.stats()
        assert not report['final_stats']['context_failed'] and report['final_stats']['live_buffers']==0
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc(),failure_stats=npu.stats())
        print(report['traceback'],flush=True)
    finally:
        npu._matrix_fp32=default_matrix
    (OUT/f'rect-{case}.json').write_text(json.dumps(report,indent=2))
    return int(report['status']!='passed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('all','smoke','numeric','large','training','training-tiled','performance','dispatch'),default='all')
    parser.add_argument('--child',action='store_true')
    parser.add_argument('--eight',action='store_true')
    parser.add_argument('--four',action='store_true')
    args = parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.eight and args.four: parser.error('Choose --four or --eight')
    if args.child: return execute(args.case)
    cases = ('numeric','large','training','training-tiled','performance','dispatch') if args.case=='all' else (args.case,)
    for case in cases:
        with (OUT/f'rect-{case}.log').open('w',encoding='utf-8') as log:
            result = subprocess.run([sys.executable,__file__,'--child','--case',case,'--four' if args.four else '--eight'],stdout=log,stderr=log,timeout=900)
        print((OUT/f'rect-{case}.log').read_text(encoding='utf-8'),flush=True)
        if result.returncode: return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Four frozen precision policies: identical-token latency, memory and host costs.

Workers stay resident but only one executes at a time. No disk graph cache,
production changes, new dependencies or model downloads. Temporary evidence
is removed after the Markdown report is accepted.
"""
import time
ENTRY = time.perf_counter_ns()
import argparse
import ctypes as ct
from functools import wraps
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/w8-efficiency'
LABELS = ('fp16', 'w8', 'mixed', 'selected')
STAGES = ('embedding', 'upload', 'decoder', 'download', 'norm', 'head', 'greedy', 'bookkeeping')
FIELDS = ('npu_calls', 'session_creations', 'driver_allocations', 'chain_groups', 'chain_graphs',
          'host_upload_copies', 'host_download_copies', 'host_device_copies', 'host_pack_copies')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write(name, data):
    (OUT/(name+'.json')).write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


def read(name):
    return json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))


def research():
    import w8_ppl_generation as ppl
    ppl.layers.OUT = OUT
    ppl.layers.mixed.OUT = OUT
    return ppl


def prepare():
    import numpy as np
    import torch
    ppl = research()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    with torch.no_grad():
        ppl.layers.mixed.prepare()
        assert read('prepare')['weights_sha256'] == 'f883156ea4204d54cef056f90160ef81a98853c5b7672d2820fdbc8162ddba62'
        write('selection', dict(selected=[list(key) for key in sorted(ppl.POLICY)]))
        cpu, tokenizer = ppl.layers.mixed.common.model()
        ids = ppl.layers.mixed.common.chat(tokenizer, 'Tell stories.', 'Tell a story about a bird.')
        assert ids.shape == (1, 24)
        tokens = ppl.layers.mixed.common.trajectory(cpu, ids, 56)
        reference = cpu(tokens[:, :55], use_cache=False).logits[:, 23:55].float().numpy()
        assert reference.shape[1] == 32
        np.savez_compressed(OUT/'reference.npz', logits=reference)
        write('freeze', dict(tokens=tokens.tolist(), prompt=24, new_tokens=32, batch=1,
            weights_sha256=read('prepare')['weights_sha256'], reference_sha256=sha(OUT/'reference.npz'),
            selection_sha256=sha(OUT/'selection.json'), runner_sha256=sha(__file__),
            policy_source_sha256=sha(ppl.__file__), checkpoint_sha256=sha(ppl.layers.mixed.common.MODEL/'model.safetensors'),
            protocol='FP16 CPU greedy trajectory, teacher-forced identical work for every precision. '
                     'Compute argmax but feed frozen tokens; ignore EOS. Full prefill CPU norm/head retained. '
                     '2 warmups, 7 balanced alternating rounds; only one worker executes at a time.'))
    print('Prepared identical 24+32 token workload and unchanged calibrated W8', flush=True)


class Memory(ct.Structure):
    _fields_ = [('cb', ct.c_ulong), ('PageFaultCount', ct.c_ulong)] + [
        (name, ct.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
        'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
        'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]


def memory():
    kernel = ct.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ct.c_void_p
    api = ct.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    api.argtypes = (ct.c_void_p, ct.POINTER(Memory), ct.c_ulong)
    api.restype = ct.c_int
    value = Memory(); value.cb = ct.sizeof(value)
    if not api(kernel.GetCurrentProcess(), ct.byref(value), value.cb): raise ct.WinError(ct.get_last_error())
    return dict(private_bytes=value.PrivateUsage, peak_commit_bytes=value.PeakPagefileUsage,
        working_set_bytes=value.WorkingSetSize, peak_working_set_bytes=value.PeakWorkingSetSize,
        page_faults=value.PageFaultCount)


def distribution(values):
    return dict(median=st.median(values), minimum=min(values), maximum=max(values), samples=values)


def orders(rounds=7):
    for i in range(rounds):
        order = tuple(LABELS[(j+i) % 4] for j in (0, 1, 3, 2))
        yield order if i // 4 % 2 == 0 else order[::-1]


def emit(data):
    print('RESULT '+json.dumps(data), flush=True)


def worker(label, profile=False):
    import numpy as np
    import torch
    ppl = research()
    import torch_npu_smollm as smol
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    driver = None
    if profile:
        import torch_npu_warm_profile as warm
        warm.OUT = OUT/'host-profile'
        manifest = json.loads((ROOT/'npunlock-probe/torch-device/build.json').read_text(encoding='utf-8-sig'))
        manifest['session_sha256'] = sha(warm.OUT/'session_profile.dll')
        bound = warm.OUT/'bound.json'; bound.write_text(json.dumps(manifest), encoding='utf-8')
        warm.bind(warm.OUT/'session_profile.dll', bound)
        driver = ct.CDLL(str(warm.OUT/'session_profile.dll'))
        driver.warm_diag_reset.argtypes = []; driver.warm_diag_reset.restype = None
        driver.warm_diag_read.argtypes = [ct.POINTER(ct.c_uint64)]*2; driver.warm_diag_read.restype = None
    begin = time.perf_counter_ns()
    import torch_intel_npu as npu
    backend_ms = (time.perf_counter_ns()-begin)/1e6
    from torch_npu_smollm_stack_demo import errors
    frozen = read('freeze')
    assert frozen['runner_sha256'] == sha(__file__) and frozen['policy_source_sha256'] == sha(ppl.__file__)
    assert frozen['selection_sha256'] == sha(OUT/'selection.json')
    assert frozen['reference_sha256'] == sha(OUT/'reference.npz')
    hooks = {}
    def hook(owner, name, key):
        original = getattr(owner, name)
        @wraps(original)
        def wrapped(*a, **kw):
            start = time.perf_counter_ns()
            try: return original(*a, **kw)
            finally:
                row = hooks.setdefault(key, dict(ms=0., calls=0))
                row['ms'] += (time.perf_counter_ns()-start)/1e6; row['calls'] += 1
        setattr(owner, name, wrapped)
    hook(smol, 'compile_fused_cached', 'compile_inclusive')
    hook(npu._native, 'create_inference_session', 'session_create')
    baseline_memory = memory()
    begin = time.perf_counter_ns()
    cpu, _ = ppl.layers.mixed.common.model()
    stack = ppl.layers.make_stack(label, None if label == 'fp16' else ppl.layers.mixed.weights())
    gc.collect()
    setup_ms = (time.perf_counter_ns()-begin)/1e6
    tokens = torch.tensor(frozen['tokens']); ids = tokens[:, :24]
    with np.load(OUT/'reference.npz', allow_pickle=False) as archive: reference = archive['logits'].copy()
    anchor = None
    def run(cold=False):
        nonlocal anchor
        generated = ids.clone(); state = None; logits_rows = []; predictions = []; steps = []; host_rows = []
        before = npu.stats(); start = time.perf_counter_ns(); first_at = None
        for step in range(32):
            marks = [time.perf_counter_ns()]
            host = cpu.model.embed_tokens(ids if step == 0 else generated[:, -1:]); marks.append(time.perf_counter_ns())
            x = host.to('intel_npu'); marks.append(time.perf_counter_ns())
            if driver: driver.warm_diag_reset()
            hidden, state = stack.prefill(x) if step == 0 else stack.decode(x, state)
            marks.append(time.perf_counter_ns())
            if driver:
                ns = (ct.c_uint64*9)(); counts = (ct.c_uint64*9)(); driver.warm_diag_read(ns, counts)
                host_rows.append(dict(ms=dict(zip(warm.STAGES, [v/1e6 for v in ns])),
                                      counts=dict(zip(warm.STAGES, list(counts)))))
            hidden = hidden.cpu(); marks.append(time.perf_counter_ns())
            normalized = cpu.model.norm(hidden); marks.append(time.perf_counter_ns())
            logits = cpu.lm_head(normalized)[:, -1].float(); marks.append(time.perf_counter_ns())
            predicted = logits.argmax(-1, keepdim=True); marks.append(time.perf_counter_ns())
            generated = torch.cat((generated, tokens[:, 24+step:25+step]), dim=1)
            logits_rows.append(logits); predictions.append(predicted); marks.append(time.perf_counter_ns())
            if step == 0: first_at = marks[-1]
            steps.append(dict(zip(STAGES, [(b-a)/1e6 for a, b in zip(marks, marks[1:])])))
        end = time.perf_counter_ns()
        change = {k:npu.stats()[k]-before[k] for k in FIELDS}
        assert all(c.length == 55 for c in state) and torch.equal(generated, tokens)
        if not cold:
            assert change['npu_calls'] == 960 and change['session_creations'] == 0, change
            assert change['chain_graphs'] == 930 and change['chain_groups'] == 248, change
            assert change['host_upload_copies'] == change['host_download_copies'] == 32, change
            assert change['host_device_copies'] == change['host_pack_copies'] == 0, change
        array = torch.stack(logits_rows, dim=1).numpy()
        assert array.shape == reference.shape and np.isfinite(array).all()
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        if anchor is None: anchor = digest
        assert digest == anchor, (label, 'non-deterministic repeated logits')
        quality = [errors(torch.from_numpy(array[:, i]), torch.from_numpy(reference[:, i])) for i in range(32)]
        decode_ms = sum(sum(row.values()) for row in steps[1:])
        row = dict(elapsed_ms=(end-start)/1e6, first_token_ms=sum(steps[0].values()),
            entry_to_first_token_ms=(first_at-ENTRY)/1e6, decode_ms=decode_ms,
            decode_tokens_per_second=31000/decode_ms, steps=steps, delta=change, memory=memory(),
            logits_sha256=digest, argmax_ids=torch.cat(predictions, dim=1).tolist(),
            quality=dict(mean_relative_l2=st.mean(q['relative_l2'] for q in quality),
                max_relative_l2=max(q['relative_l2'] for q in quality),
                within_old_3_percent=sum(q['relative_l2'] < .03 for q in quality)), host=host_rows)
        if driver and not cold:
            for i, row_host in enumerate(host_rows):
                assert row_host['counts']['append_graph'] == 30
                assert row_host['counts']['wait'] == (30 if i == 0 else 8)
        return row
    ready = dict(label=label, profile=profile, pid=os.getpid(), backend_ms=backend_ms, setup_ms=setup_ms,
                 baseline_memory=baseline_memory, prepared_memory=memory(), device=npu.stats())
    try:
        with torch.no_grad():
            ready['cold'] = run(cold=True)
            ready['entry_to_first_token_ms'] = ready['cold']['entry_to_first_token_ms']
            ready['compile_hooks'] = hooks.copy()
            ready['warmups'] = [run(), run()]
            graphs = []
            for i, layer in enumerate(stack.layers):
                assert len(layer._evidence) == 2
                for key, evidence in layer._evidence.items():
                    integer = [m for m in evidence['modes'] if not m['elementwise'] and m['wmode'] in (1, 2)]
                    assert bool(integer) == (label != 'fp16')
                    assert all(m['amode'] == 0 and m['wmode'] == 2 for m in integer)
                    graphs.append(dict(layer=i, key=list(key), evidence=evidence))
            del layer
            ready.update(graphs=graphs, warmed_memory=memory(), protected=[list((i, name))
                for i, layer in enumerate(stack.layers) for name in sorted(layer.fp16_projections)])
            emit(dict(kind='ready', **ready))
            for line in sys.stdin:
                command = json.loads(line)
                if command['op'] == 'stop': break
                assert command['op'] == 'run'
                emit(dict(kind='sample', label=label, index=command['index'], result=run()))
        del stack, cpu
        gc.collect(); npu.shutdown(); stats = npu.stats()
        assert stats['closed'] and not stats['context_failed']
        assert all(stats[k] == 0 for k in ('live_buffers', 'live_buffer_bytes', 'cached_buffer_bytes', 'quarantined_buffer_bytes', 'cached_sessions'))
        emit(dict(kind='closed', label=label, stats=stats, memory=memory()))
    except Exception:
        emit(dict(kind='failed', label=label, traceback=traceback.format_exc()))
        raise


def response(process):
    while True:
        line = process.stdout.readline()
        if not line: raise RuntimeError(('worker exited', process.poll()))
        if line.startswith('RESULT '):
            value = json.loads(line[7:])
            assert value['kind'] != 'failed', value
            return value
        print(line.rstrip(), flush=True)


def benchmark(profile=False):
    frozen = read('freeze'); assert frozen['runner_sha256'] == sha(__file__)
    if profile:
        import torch_npu_warm_profile as warm
        warm.OUT = OUT/'host-profile'; warm.OUT.mkdir(exist_ok=True); warm.build()
    report = dict(status='running', profile=profile, freeze=frozen, ready={}, samples={}, closed={}, orders=[])
    processes = {}
    try:
        # Profile separately: its timers and DLL must not contaminate production latency numbers.
        for label in LABELS:
            args = [sys.executable, '-B', __file__, '--case', 'worker', '--label', label]
            if profile: args.append('--profile')
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, encoding='utf-8', bufsize=1, cwd=ROOT)
            processes[label] = process
            print('Initializing', label, 'profile' if profile else 'production', flush=True)
            row = response(process); assert row['kind'] == 'ready' and row['label'] == label
            report['ready'][label] = row; report['samples'][label] = []
            print(label, 'cold TTFT ms', round(row['cold']['first_token_ms'], 3),
                'warm decode tok/s', round(row['warmups'][-1]['decode_tokens_per_second'], 3), flush=True)
            if profile:
                for i in range(3):
                    process.stdin.write(json.dumps(dict(op='run', index=i))+'\n'); process.stdin.flush()
                    report['samples'][label].append(response(process)['result'])
                process.stdin.write('{"op":"stop"}\n'); process.stdin.flush()
                report['closed'][label] = response(process); assert process.wait(timeout=30) == 0
        if not profile:
            for i, order in enumerate(orders()):
                report['orders'].append(order)
                for label in order:
                    process = processes[label]
                    process.stdin.write(json.dumps(dict(op='run', index=i))+'\n'); process.stdin.flush()
                    row = response(process); assert row['kind'] == 'sample' and row['index'] == i and row['label'] == label
                    report['samples'][label].append(row['result'])
                    print('Round', i+1, label, 'TTFT/decode ms', round(row['result']['first_token_ms'], 3),
                          round(row['result']['decode_ms']/31, 3), flush=True)
            for label, process in processes.items():
                process.stdin.write('{"op":"stop"}\n'); process.stdin.flush()
                report['closed'][label] = response(process); assert process.wait(timeout=30) == 0
        report.update(status='passed', sources={name:sha(ROOT/name) for name in (
            'torch_npu_smollm.py', 'torch_npu_lowbit.py', 'torch_intel_npu.py',
            'npunlock-probe/session-runtime/npunlock_session.dll', 'npunlock-probe/torch-device/_torch_intel_npu.pyd',
            'npunlock-probe/research/w8_efficiency.py')})
        write('profile' if profile else 'benchmark', report)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.stdin.write('{"op":"stop"}\n'); process.stdin.flush()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: process.terminate(); process.wait(timeout=30)


def selfcheck():
    assert len(list(orders())) == 7
    assert all(set(order) == set(LABELS) and len(order) == 4 for order in orders())
    assert all(sum(order[position] == label for order in orders()) in (1, 2)
               for label in LABELS for position in range(4))
    assert distribution([3, 1, 2])['median'] == 2
    result = memory(); assert result['peak_commit_bytes'] >= result['private_bytes'] > 0
    assert result['peak_working_set_bytes'] >= result['working_set_bytes'] > 0
    print('Self-check passed: schedule, statistics, Windows process memory', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('selfcheck', 'prepare', 'benchmark', 'profile', 'worker'), required=True)
    parser.add_argument('--label', choices=LABELS)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    assert not os.environ.get('NPUNLOCK_GRAPH_CACHE')
    OUT.mkdir(parents=True, exist_ok=True)
    if args.case == 'worker':
        assert args.label; worker(args.label, args.profile)
    else:
        {'selfcheck': selfcheck, 'prepare': prepare, 'benchmark': benchmark,
         'profile': lambda: benchmark(True)}[args.case]()

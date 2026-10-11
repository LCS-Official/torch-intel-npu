"""Synchronous 1/2/4-layer submission experiment; production stays unchanged."""
import argparse
import ctypes as ct
import gc
import json
import os
from pathlib import Path
import shutil
import statistics as st
import subprocess
import time
import types
import sys
import numpy as np
from w8_quality_probe import ROOT, MODEL, sha
from shave_fp32_pipeline import metadata, snapshots, final

OUT = ROOT / 'npunlock-probe/research/results/decode-chain'
HERE = Path(__file__).parent


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path, value): path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def build():
    import torch_npu_warm_profile as warm
    before = OUT / 'before'; before.mkdir(exist_ok=True)
    state = metadata()
    for name in ('npunlock/src/graphinfer/session_win32.c', 'npunlock-probe/torch-device/backend.cpp',
                 'npunlock-probe/torch-device/build.json', 'build_torch_npu_device.ps1',
                 'npunlock-probe/research/decode_chain.py', 'npunlock-probe/research/decode_chain_extra.c',
                 'npunlock-probe/research/decode_chain_extra.cpp'):
        state['sources'][name] = sha(ROOT / name)
    if not (before / 'sources.json').exists():
        for name in state['sources']:
            path = before / name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
        write(before / 'sources.json', state)
    for name, digest in read(before / 'sources.json')['sources'].items(): assert sha(before / name) == digest
    warm.OUT = OUT / 'host'; warm.OUT.mkdir(exist_ok=True); warm.build()
    original = ROOT / 'npunlock/src/graphinfer/session_win32.c'
    extra = (HERE / 'decode_chain_extra.c').read_text(encoding='utf-8')
    compiler = read(warm.OUT / 'profile-build.json')['arguments']
    info = dict(status='building', production=metadata(), builder_sha256=sha(Path(__file__)), runtimes={})
    for name, source in (('plain', original.read_text(encoding='utf-8')),
                         ('profile', (warm.OUT / 'session_profile.c').read_text(encoding='utf-8'))):
        target = OUT / (name + '.c'); target.write_text(source + '\n' + extra, encoding='utf-8')
        args = [str(target) if p == str(warm.OUT / 'session_profile.c') else
                str(OUT / (name + '.dll')) if p == str(warm.OUT / 'session_profile.dll') else p for p in compiler]
        if name == 'profile': args.insert(1, '-DCHAIN_PROFILE')
        result = subprocess.run(args, env=dict(os.environ, PATH=str(Path(args[0]).parent)+';'+os.environ['PATH']), capture_output=True, text=True)
        (OUT / (name + '-build.log')).write_text(result.stdout + result.stderr, encoding='utf-8'); result.check_returncode()
        info['runtimes'][name] = dict(dll_sha256=sha(OUT / (name + '.dll')), source_sha256=sha(target), arguments=args)
    extension = OUT / 'extension'; extension.mkdir(exist_ok=True)
    source = (ROOT / 'npunlock-probe/torch-device/backend.cpp').read_text(encoding='utf-8')
    marker = 'void run(uintptr_t session,'; assert source.count(marker) == 1
    source = source.replace(marker, (HERE / 'decode_chain_extra.cpp').read_text(encoding='utf-8') + '\n' + marker)
    marker = '  m.attr("max_tensor_elements") = max_tensor_elements;'; assert source.count(marker) == 1
    source = source.replace(marker, marker + '\n  m.def("_diagnostic_chain", diagnostic_chain, pybind11::call_guard<pybind11::gil_scoped_release>());')
    (extension / 'backend.cpp').write_text(source, encoding='utf-8')
    script = (ROOT / 'build_torch_npu_device.ps1').read_text(encoding='utf-8-sig')
    script = script.replace('$root = $PSScriptRoot', "$root = '" + str(ROOT).replace("'", "''") + "'")
    marker = "$output = Join-Path $root 'npunlock-probe\\torch-device'"; assert script.count(marker) == 1
    script = script.replace(marker, "$output = '" + str(extension).replace("'", "''") + "'")
    (extension / 'build.ps1').write_text(script, encoding='utf-8')
    with (extension / 'build.log').open('w', encoding='utf-8') as log:
        subprocess.run(['pwsh', '-NoProfile', '-File', str(extension / 'build.ps1'), '-Toolchain', str(ROOT / 'npunlock-probe/toolchain/msvc')], stdout=log, stderr=log, timeout=240, check=True)
    info['extension'] = read(extension / 'build.json'); info['status'] = 'built'
    for name in ('plain', 'profile'):
        write(extension / (name + '-bound.json'), dict(info['extension'], session_sha256=info['runtimes'][name]['dll_sha256']))
    write(OUT / 'build.json', info); print('Isolated plain/profile runtimes and chain extension built', flush=True)


def bind(profile):
    import torch_npu_runtime
    name = 'profile' if profile else 'plain'; info = read(OUT / 'build.json')
    assert sha(OUT / (name + '.dll')) == info['runtimes'][name]['dll_sha256']
    torch_npu_runtime.SESSION_DLL = OUT / (name + '.dll')
    source = (ROOT / 'torch_intel_npu.py').read_text(encoding='utf-8')
    source = source.replace('_EXT = _ROOT / "npunlock-probe/torch-device/_torch_intel_npu.pyd"', f'_EXT = Path({str(OUT / "extension/_torch_intel_npu.pyd")!r})')
    source = source.replace('_BUILD = _EXT.with_name("build.json")', f'_BUILD = Path({str(OUT / "extension" / (name + "-bound.json"))!r})')
    module = types.ModuleType('torch_intel_npu'); module.__file__ = str(ROOT / 'torch_intel_npu.py')
    sys.modules[module.__name__] = module; exec(compile(source, module.__file__, 'exec'), module.__dict__)
    driver = module._native.infer
    if profile:
        driver.warm_diag_reset.argtypes = []; driver.warm_diag_reset.restype = None
        driver.warm_diag_read.argtypes = [ct.POINTER(ct.c_uint64)] * 2; driver.warm_diag_read.restype = None
    return module, driver


def execute_plan(npu, plan, group):
    npu._C._diagnostic_chain([s._result.session for s, p, xs, ys in plan],
        [xs for s, p, xs, ys in plan], [[t.name for t in p.graph.inputs] for s, p, xs, ys in plan],
        [ys for s, p, xs, ys in plan], [[t.name for t in p.graph.outputs] for s, p, xs, ys in plan], group)


def planned(npu, fn, group):
    import torch
    plan = []; original = npu._run_many
    def capture(key, inputs, specs, outs=None):
        assert outs is None
        npu.synchronize()
        for x in inputs: npu._validate(x)
        session, program = npu._session(key)
        assert len(specs) == len(program.graph.outputs)
        outputs = tuple(torch.empty(shape, dtype=dtype, device='intel_npu') for shape, dtype in specs)
        for out, (shape, dtype) in zip(outputs, specs): npu._check_out(out, tuple(shape), dtype, inputs)
        assert len({out.untyped_storage()._cdata for out in outputs}) == len(outputs)
        plan.append((session, program, list(inputs), list(outputs)))
        return outputs
    with npu._lock:
        npu._run_many = capture
        try: result = fn()
        finally: npu._run_many = original
        assert plan
        execute_plan(npu, plan, group)
        return result


def checks(profile):
    import torch
    npu, driver = bind(profile); torch.set_num_threads(8)
    report = dict(status='running', checks=[], build=read(OUT / 'build.json'), experiment=metadata())
    def work():
        for dtype, name in ((torch.float16, 'f16'), (torch.float32, 'f32')):
            program = npu.bridge._program('add1', name, (16,))
            sessions = [npu._native.create_inference_session(program.graph_blob, parent=npu._root_session) for _ in range(4)]
            for i, session in enumerate(sessions):
                npu._sessions[('diagnostic_add1', name, i)] = session, program; npu._session_creations += 1
            host = torch.arange(16, dtype=dtype); x = host.to('intel_npu')
            ys = [torch.empty_like(x) for _ in sessions]
            plan = [(s, program, [x if i == 0 else ys[i-1]], [ys[i]]) for i, s in enumerate(sessions)]
            # Cold sessions and malformed lengths are rejected before queue submission.
            before = npu.stats()['npu_calls']
            try: execute_plan(npu, plan[:2], 2)
            except RuntimeError as error: assert 'cold/closed/duplicate' in str(error)
            else: raise AssertionError('cold batch accepted')
            assert npu.stats()['npu_calls'] == before and not npu.stats()['context_failed']
            for i in range(4): npu._run(('diagnostic_add1', name, i), [x], (16,), dtype, out=ys[i])
            for group in (1, 2, 4):
                execute_plan(npu, plan, group)
                torch.testing.assert_close(ys[-1].cpu(), host + 4, rtol=0, atol=0)
                report['checks'].append(dict(dtype=name, group=group, dependent_add1='passed'))
            bad = [(sessions[0], program, [x], [x])]
            try: execute_plan(npu, bad, 1)
            except RuntimeError as error: assert 'aliased' in str(error)
            else: raise AssertionError('alias accepted')
            try: execute_plan(npu, [plan[0], (sessions[0], program, [ys[0]], [ys[1]])], 2)
            except RuntimeError as error: assert 'duplicate' in str(error)
            else: raise AssertionError('duplicate graph accepted')
            execute_plan(npu, plan, 4)
            torch.testing.assert_close(ys[-1].cpu(), host + 4, rtol=0, atol=0)
            report['checks'].append(dict(dtype=name, cold_alias_duplicate_and_recovery='passed'))
    with torch.no_grad(): work(); final(npu, report)
    report['status'] = 'passed'; write(OUT / 'checks.json', report)
    print('dependent chain checks passed', len(report['checks']), flush=True)


def model(profile, label):
    import torch
    npu, driver = bind(profile)
    from torch_npu_smollm import FrozenSmolStack
    from transformers import AutoTokenizer
    from safetensors import safe_open
    from torch_npu_warm_profile import STAGES
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    ids = tokenizer('Once upon a time, a curious child asked how birds fly.', return_tensors='pt').input_ids
    with safe_open(MODEL / 'model.safetensors', framework='pt', device='cpu') as file: embedding = file.get_tensor('model.embed_tokens.weight')
    report = dict(status='running', label=label, profile=profile, decoder_layers=30, build=read(OUT / 'build.json'),
                  experiment=metadata(), wrapper_sha256=sha(Path(__file__)), checks=[], timings=[], graphs={})
    shutil.copyfile(Path(__file__), OUT / (label+'-runner.py'))
    saved = {}
    def work():
        tokens = ids.repeat(1, (33 + ids.shape[1]) // ids.shape[1])[:, :33]
        host = embedding[tokens].half().contiguous(); x = host[:, :32].contiguous().to('intel_npu'); one = host[:, 32:].contiguous().to('intel_npu')
        stack = FrozenSmolStack.from_checkpoint(MODEL, num_layers=30, mode='fp16')
        print(label, 'preparing current B1/S32 prefill/decode controls', flush=True)
        initial = stack.prefill(x); seed = initial[1]; baseline = stack.decode(one, seed)
        anchors = [v.cpu().numpy().copy() for v in snapshots(baseline)]
        for i, value in enumerate(anchors): saved[f'baseline/{i}'] = value
        seed_arrays = [v.cpu().numpy().copy() for c in seed for v in (c.keys, c.values, c._position)]
        for i, value in enumerate(seed_arrays): saved[f'seed-before/{i}'] = value
        del initial, baseline
        report['graphs'] = {f'{i}/{local[-1]}': __import__('hashlib').sha256(npu._sessions[key][1].graph_blob).hexdigest()
                            for i, layer in enumerate(stack.layers) for local, key in layer._keys.items()}
        assert len(report['graphs']) == 60
        runs = {'baseline': lambda: stack.decode(one, seed)}
        runs.update({f'group{g}': (lambda g=g: planned(npu, lambda: stack.decode(one, seed), g)) for g in (1, 2, 4)})
        for name, fn in runs.items():
            value = fn()
            for i, tensor in enumerate(snapshots(value)):
                array = tensor.cpu().numpy().copy(); np.testing.assert_array_equal(array.view(np.uint8), anchors[i].view(np.uint8)); saved[f'{name}/{i}'] = array
            report['checks'].append(dict(strategy=name, bit_equal_arrays=91)); del value
            print(label, name, 'bit exact', flush=True)
        # Pair each candidate with the original API, ABBA/BAAB; report planning and cleanup too.
        for name in ('group1', 'group2', 'group4'):
            samples = []; pairs = []
            for fn in (runs['baseline'], runs[name]):
                for _ in range(3): value = fn(); del value
            before = npu.stats()
            for index in range(21):
                order = ('baseline', name, name, 'baseline') if index % 2 == 0 else (name, 'baseline', 'baseline', name)
                pair = {k: [] for k in ('baseline', name)}
                for key in order:
                    if profile: driver.warm_diag_reset()
                    start = time.perf_counter_ns(); value = runs[key](); returned = time.perf_counter_ns(); del value; freed = time.perf_counter_ns()
                    row = dict(strategy=key, api_ms=(returned-start)/1e6, cleanup_ms=(freed-returned)/1e6, total_ms=(freed-start)/1e6)
                    if profile:
                        ns = (ct.c_uint64*9)(); counts = (ct.c_uint64*9)(); driver.warm_diag_read(ns, counts)
                        row.update(stages_ms=dict(zip(STAGES, [v/1e6 for v in ns])), stage_counts=dict(zip(STAGES, list(counts))))
                        expected = 30 if key == 'baseline' else (30 + int(key[-1])-1) // int(key[-1])
                        assert row['stage_counts']['append_graph'] == 30
                        assert all(row['stage_counts'][stage] == expected for stage in STAGES[4:])
                    samples.append(row); pair[key].append(row['total_ms'])
                pairs.append({key: st.mean(values) for key, values in pair.items()})
            after = npu.stats()
            delta = {key: after[key]-before[key] for key in ('npu_calls','driver_allocations','host_upload_copies','host_download_copies','host_device_copies','host_pack_copies','session_creations')}
            assert delta['npu_calls'] == 21*4*30 and all(v == 0 for k, v in delta.items() if k != 'npu_calls'), delta
            medians = {key: {field: st.median(row[field] for row in samples if row['strategy']==key) for field in ('api_ms','cleanup_ms','total_ms')} for key in ('baseline', name)}
            wins = sum(pair[name] < pair['baseline'] for pair in pairs)
            report['timings'].append(dict(candidate=name, samples=samples, pairs_total_ms=pairs, median_ms=medians, positive_rounds=wins, warm_delta=delta))
            print(label, name, medians, 'wins', wins, flush=True)
        for i, tensor in enumerate(v for c in seed for v in (c.keys, c.values, c._position)):
            array = tensor.cpu().numpy().copy(); np.testing.assert_array_equal(array.view(np.uint8), seed_arrays[i].view(np.uint8)); saved[f'seed/{i}'] = array
        report['immutable_seed_arrays'] = 90
        # Advance the actual KV positions, rebinding fresh outputs at each step.
        reference_seed = candidate_seed = seed
        for step in range(8):
            reference = stack.decode(one, reference_seed)
            candidate = planned(npu, lambda: stack.decode(one, candidate_seed), 4)
            for i, (a, b) in enumerate(zip(snapshots(reference), snapshots(candidate))):
                expected = a.cpu().numpy().copy(); actual = b.cpu().numpy().copy()
                np.testing.assert_array_equal(expected.view(np.uint8), actual.view(np.uint8))
                saved[f'advance/{step}/baseline/{i}'] = expected; saved[f'advance/{step}/group4/{i}'] = actual
            reference_seed = reference[1]; candidate_seed = candidate[1]
            assert all(c.length == 33 + step and int(c._position.cpu().item()) == 33 + step for c in candidate_seed)
        for i, tensor in enumerate(v for c in seed for v in (c.keys, c.values, c._position)):
            np.testing.assert_array_equal(tensor.cpu().numpy().view(np.uint8), seed_arrays[i].view(np.uint8))
        report['advancing_decode'] = dict(steps=8, bit_equal_arrays=8*91, final_position=candidate_seed[0].length)
    with torch.no_grad(): work(); final(npu, report)
    output = OUT / (label+'-outputs.npz'); np.savez_compressed(output, **saved)
    report.update(status='passed', outputs_sha256=sha(output)); write(OUT / (label+'.json'), report)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--case', choices=('build','checks','model'), required=True)
    parser.add_argument('--profile', action='store_true'); parser.add_argument('--label', default='plain'); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); os.environ['HF_HUB_OFFLINE']='1'; os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.case == 'build': build()
    elif args.case == 'checks': checks(args.profile)
    else: model(args.profile, args.label)


if __name__ == '__main__': main()

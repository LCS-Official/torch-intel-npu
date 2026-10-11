"""Validate the direct square FP32 SHAVE GEMM against the resident NPU path."""
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
PARALLEL = "--parallel" in sys.argv
OUT = ROOT / f"npunlock-probe/research/results/{'gemm-parallel' if PARALLEL else 'gemm'}"


def execute():
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from npunlock.npu4000 import compile_square_gemm, compile_parallel_square_gemm, compile_four_tile_square_gemm
    directory = ROOT / "npunlock/tests/fixtures/npu4000"
    manifest = json.loads((directory / "gemm/manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    kernel = (directory / "gemm/square-f32.elf").read_bytes()
    assert hashlib.sha256(kernel).hexdigest() == manifest["kernels"]["square-f32"]["elf_sha256"]
    kernels = {"serial": kernel}
    parallel_manifest = None
    if PARALLEL:
        parallel_manifest = json.loads((directory / "gemm_parallel/manifest.json").read_text())
        for name, digest in parallel_manifest['source_sha256'].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
        for variant, kind in (('parallel1', 'square-f32-lanes1'), ('parallel4', 'square-f32-lanes4'),
                              ('parallel_fast', 'square-f32-fast'), ('four_tile', 'square-f32-four-tiles')):
            data = (directory / f'gemm_parallel/{kind}.elf').read_bytes()
            assert hashlib.sha256(data).hexdigest() == parallel_manifest['kernels'][kind]['elf_sha256']
            kernels[variant] = data
    variants = ('parallel1', 'parallel4', 'parallel_fast', 'four_tile') if PARALLEL else ('serial',)
    selected = variants[-1]
    report = {"status": "running", "kernel_manifest": manifest, "checks": [], "timings": [],
              "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                          ("torch_intel_npu.py", "npunlock/python/npunlock/npu4000.py", "torch_npu_gemm_demo.py")},
              "timing_scope": "same-process interleaved resident synchronous wall time; no uploads/downloads/compilation",
              "desktop_load": "not controlled; user recently reported video playback"}
    report['parallel_kernel_manifest'] = parallel_manifest
    torch.manual_seed(103)
    old_matrix = npu._matrix_fp32

    def direct(a, b, out=None, *, version=selected):
        size = a.shape[0]
        assert a.shape == b.shape == (size, size) and a.dtype == b.dtype == torch.float32
        with npu._lock:
            npu.synchronize()
            npu._validate(a)
            npu._validate(b)
            key = ("square_gemm_prototype", version, (size, size))
            if key not in npu._sessions:
                compiler = (compile_four_tile_square_gemm if version == 'four_tile' else
                            (compile_square_gemm if version == 'serial' else compile_parallel_square_gemm))
                program = compiler(size, kernels[version], libraries=npu._native)
                session = npu._native.create_inference_session(program.graph_blob,
                    timeout_ms=10000, parent=npu._root_session)
                assert len(npu._sessions) < npu._MAX_SESSIONS
                npu._sessions[key] = session, program
                npu._session_creations += 1
                (OUT / f"square-{size}-{version}-contract.json").write_bytes(program.patch_reports[0])
            if version == 'four_tile':
                shape = (4, 2, size*size//2)
                inputs = [npu._broadcast(v.view(1, size*size), (4, size*size)).view(shape) for v in (a, b)]
                partials = npu._run(key, inputs, shape, torch.float32).view(4, size, size)
                return npu._reduce(partials, 0, dtype=torch.float32, kind='sum', out=out)
            return npu._run(key, [a, b], (size, size), torch.float32, out=out)

    def candidate(a, b, *, out=None):
        size = a.shape[0]
        if a.shape == b.shape == (size, size) and 2 <= size <= 64:
            a, b = (npu._cast(npu._materialize(v), torch.float32) for v in (a, b))
            version = selected if size in (8, 16, 32, 64) or not PARALLEL else 'parallel_fast'
            return direct(a, b, out, version=version)
        return old_matrix(a, b, out=out)

    def check(size, pattern, version):
        a, b = torch.randn(size, size) / 4, torch.randn(size, size) / 4
        if pattern == "precision":
            a = torch.eye(size) * 1.0003
            b = torch.full((size, size), 1.0003)
        elif pattern == "subnormal":
            bits = np.array([1, 0x007fffff, 0x00800000, 0x80000001, 0x807fffff, 0x80800000], dtype=np.uint32)
            a = torch.from_numpy(np.resize(bits.view(np.float32), size*size).reshape(size, size).copy())
            b = torch.eye(size)
        elif pattern == "cancellation":
            a.fill_(1)
            a[:, :3] = torch.tensor([1e10, 1.0, -1e10])
            b.fill_(1)
        elif pattern == "ieee_edges":
            bits = np.array([0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x3f800000,
                             0xbf800000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc12345], dtype=np.uint32)
            a = torch.from_numpy(np.resize(bits.view(np.float32), size*size).reshape(size, size).copy())
            b = a.t().contiguous()
        elif pattern == 'random_bits':
            rng = np.random.default_rng(211+size)
            a, b = (torch.from_numpy(rng.integers(0, 2**32, size=(size, size), dtype=np.uint32).view(np.float32))
                    for _ in range(2))
        elif pattern in ('add_threshold', 'mul_threshold'):
            exponents = (24, 25, 26) if pattern == 'add_threshold' else (63, 64, 65)
            bits = np.array([(e << 23) | frac | sign for e in exponents
                             for frac in (0, 1, 0x7fffff) for sign in (0, 0x80000000)], dtype=np.uint32)
            a = torch.from_numpy(np.resize(bits.view(np.float32), size*size).reshape(size, size).copy())
            b = torch.ones(size, size) if pattern == 'add_threshold' else a.t().contiguous()
        x, y = a.to("intel_npu"), b.to("intel_npu")
        baseline = old_matrix(x, y).cpu()
        before = npu.stats()
        result = direct(x, y, version=version)
        after = npu.stats()
        actual = result.cpu()
        np.testing.assert_array_equal(actual.numpy().view(np.uint32), baseline.numpy().view(np.uint32))
        entry = {"size": size, "pattern": pattern, "version": version, "status": "passed", "old_npu_bitwise_equal": True,
                 "direct_graph_calls": after["npu_calls"] - before["npu_calls"]}
        assert entry["direct_graph_calls"] == (4 if version == 'four_tile' else 1)
        for field in ("host_upload_copies", "host_download_copies", "host_device_copies"):
            assert after[field] == before[field], field
        if pattern not in ("ieee_edges", 'random_bits'):
            reference = (a.double() @ b.double()).float()
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=3e-6)
            entry["max_abs_error_fp64_reference"] = (actual - reference).abs().max().item()
        if pattern == "precision":
            low = (a.half().float() @ b.half().float())
            assert not torch.equal(actual, low), "FP32 probe cannot distinguish FP16 input lowering"
            entry["differs_from_fp16_inputs"] = True
        report["checks"].append(entry)
        print("check", entry, flush=True)

    def measure(size):
        a, b = torch.randn(size, size) / 4, torch.randn(size, size) / 4
        x, y = a.to("intel_npu"), b.to("intel_npu")
        versions = ('old', 'serial', *variants) if PARALLEL else ('old', 'serial')
        outputs = {name: torch.empty((size, size), device="intel_npu") for name in versions}
        run = {"old": lambda: old_matrix(x, y, out=outputs["old"])}
        run.update({name: (lambda name=name: direct(x, y, out=outputs[name], version=name)) for name in versions[1:]})
        calls = {}
        for name in run:
            for _ in range(3):
                run[name]()
            before = npu.stats()
            run[name]()
            calls[name] = npu.stats()["npu_calls"] - before["npu_calls"]
        before = npu.stats()
        samples = {name: [] for name in (*versions, 'cpu')}
        cpu_out = torch.empty_like(a)
        for i in range(9):
            order = versions[i % len(versions):] + versions[:i % len(versions)]
            for name in order if i % 2 == 0 else tuple(reversed(order)):
                start = time.perf_counter()
                run[name]()
                samples[name].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            torch.mm(a, b, out=cpu_out)
            samples["cpu"].append((time.perf_counter() - start) * 1000)
        after = npu.stats()
        for field in ("host_upload_copies", "host_download_copies", "host_device_copies", "session_creations"):
            assert after[field] == before[field], field
        baseline = outputs['old'].cpu().numpy().view(np.uint32)
        for name in versions[1:]:
            np.testing.assert_array_equal(baseline, outputs[name].cpu().numpy().view(np.uint32))
        medians = {name: statistics.median(values) for name, values in samples.items()}
        entry = {"size": size, "median_ms": medians, "samples_ms": samples, "graph_calls": calls,
                 "speedup_vs_old_npu": medians["old"] / medians[selected], "warm_host_copies_and_compiles": 0,
                 "cpu_threads": torch.get_num_threads()}
        report["timings"].append(entry)
        print("timing", {k: v for k, v in entry.items() if k != "samples_ms"}, flush=True)

    def training(dtype):
        size = 16
        initial = torch.nn.Linear(size, size, bias=True, dtype=dtype)
        host = torch.randn(size, size, dtype=dtype) / 8
        target = torch.randn(size, size, dtype=dtype) / 8
        saved = []
        for version in ("old", selected):
            model = torch.nn.Linear(size, size, bias=True, dtype=dtype)
            model.load_state_dict(initial.state_dict())
            model.to("intel_npu")
            x = host.to("intel_npu").requires_grad_()
            goal = target.to("intel_npu")
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            npu._matrix_fp32 = old_matrix if version == "old" else candidate
            states = []
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                x.grad = None
                difference = model(x) - goal
                loss = (difference * difference).mean()
                loss.backward()
                state = [loss.detach().cpu(), x.grad.cpu(), *(p.grad.cpu() for p in model.parameters())]
                optimizer.step()
                state += [p.detach().cpu() for p in model.parameters()]
                states.append(state)
            saved.append(states)
        npu._matrix_fp32 = old_matrix
        for old_step, new_step in zip(*saved):
            for old, new in zip(old_step, new_step):
                integer = np.uint32 if dtype == torch.float32 else np.uint16
                np.testing.assert_array_equal(old.numpy().view(integer), new.numpy().view(integer))
        entry = {"dtype": str(dtype), "steps": 3, "loss_gradients_updates_bitwise_equal": True, "status": "passed"}
        report.setdefault("training", []).append(entry)
        print("training", entry, flush=True)

    try:
        with torch.no_grad():
            for size in (2, 3, 4, 5, 7, 8, 16, 17, 31, 32, 64):
                for version in variants:
                    if version == 'four_tile' and size not in (8, 16, 32, 64):
                        continue
                    check(size, "finite", version)
            for size in (4, 16, 64):
                patterns = ("precision", "subnormal", "cancellation", "ieee_edges")
                if PARALLEL:
                    patterns += ('random_bits', 'add_threshold', 'mul_threshold')
                for pattern in patterns:
                    for version in variants:
                        if version == 'four_tile' and size == 4:
                            continue
                        check(size, pattern, version)
            for size in (8, 16, 32, 64):
                measure(size)
        for dtype in (torch.float32, torch.float16):
            training(dtype)
        gc.collect()
        report["stats_before_shutdown"] = npu.stats()
        assert report["stats_before_shutdown"]["live_buffers"] == report["stats_before_shutdown"]["cached_scalar_buffers"]
        npu.shutdown()
        report["final_stats"] = npu.stats()
        assert not report["final_stats"]["context_failed"] and report["final_stats"]["live_buffers"] == 0
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc(), failure_stats=npu.stats())
        print(report["traceback"], flush=True)
    finally:
        npu._matrix_fp32 = old_matrix
    (OUT / "square-gemm-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return int(report["status"] != "passed")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--child" in sys.argv:
        return execute()
    with (OUT / "square-gemm-validation.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, __file__, "--child", *(['--parallel'] if PARALLEL else [])],
                                stdout=log, stderr=log, timeout=400)
    print((OUT / "square-gemm-validation.log").read_text(encoding="utf-8"))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

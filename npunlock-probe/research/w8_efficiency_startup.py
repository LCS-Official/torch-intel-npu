"""Fresh-process disk-cache startup, one temporary bounded cache at a time."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import w8_efficiency as bench


def child(label):
    original = bench.emit
    def emit(data):
        if data['kind'] == 'ready':
            import torch_npu_graph_cache
            data['cache'] = torch_npu_graph_cache.stats()
        original(data)
    bench.emit = emit
    bench.worker(label)


def startup():
    report = dict(status='running', variants={}, script_sha256=bench.sha(__file__))
    temp_root = Path(tempfile.gettempdir()).resolve()
    for label in bench.LABELS:
        with tempfile.TemporaryDirectory(prefix='torch-intel-npu-w8-startup-') as name:
            directory = Path(name).resolve()
            assert directory.parent == temp_root and directory.name.startswith('torch-intel-npu-w8-startup-')
            assert not directory.is_symlink()
            rows = []
            for index in range(3):
                environment = dict(os.environ, NPUNLOCK_GRAPH_CACHE=str(directory))
                started = time.perf_counter_ns()
                process = subprocess.Popen([sys.executable, '-B', __file__, '--label', label],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True,
                    encoding='utf-8', bufsize=1, cwd=bench.ROOT, env=environment)
                try:
                    ready = bench.response(process)
                    assert ready['kind'] == 'ready'
                    ready['parent_launch_to_ready_ms'] = (time.perf_counter_ns()-started)/1e6
                    process.stdin.write('{"op":"stop"}\n'); process.stdin.flush()
                    closed = bench.response(process)
                    assert closed['kind'] == 'closed' and process.wait(timeout=30) == 0
                    cache = ready['cache']
                    if index == 0:
                        assert cache['compiles'] == cache['misses'] == cache['writes'] == 60, cache
                        assert cache['hits'] == 0
                    else:
                        assert cache['hits'] == 60 and cache['compiles'] == cache['misses'] == 0, cache
                        assert cache['ir_skipped'] == 30, cache
                    assert cache['rejected'] == cache['write_skipped'] == 0
                    assert not rows or ready['cold']['logits_sha256'] == rows[0]['ready']['cold']['logits_sha256']
                    rows.append(dict(ready=ready, closed=closed))
                    print(label, 'cache populate' if index == 0 else 'cache hit '+str(index),
                        'TTFT ms', round(ready['cold']['first_token_ms'], 3),
                        'entry ms', round(ready['entry_to_first_token_ms'], 3), flush=True)
                finally:
                    if process.poll() is None:
                        process.terminate(); process.wait(timeout=30)
            entries = list(directory.iterdir())
            assert len(entries) == 60 and all(p.suffix == '.npc' and p.is_file() and not p.is_symlink() for p in entries)
            size = sum(p.stat().st_size for p in entries)
            assert size <= 512*1024*1024
            report['variants'][label] = dict(rows=rows, temporary_cache_files=len(entries), temporary_cache_bytes=size)
            # The path and every entry are checked before TemporaryDirectory removes them.
            assert directory.resolve().parent == temp_root and not directory.is_symlink()
        assert not directory.exists()
    report.update(status='passed', caches_removed=True)
    bench.write('startup', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--label', choices=bench.LABELS)
    args = parser.parse_args()
    if args.label: child(args.label)
    else:
        assert not os.environ.get('NPUNLOCK_GRAPH_CACHE')
        startup()

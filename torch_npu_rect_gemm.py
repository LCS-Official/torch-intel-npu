"""Bounded full-K FP32 GEMM; default dispatch selects measured 128/256 squares."""
from functools import lru_cache
import hashlib
import json
import torch
import torch_intel_npu as npu
from npunlock.npu4000 import compile_rect_gemm, rect_gemm_capacity


@lru_cache(maxsize=2)
def _kernel(replicas):
    root = npu._ROOT/'npunlock/tests/fixtures/npu4000'
    directory = 'gemm_rect' if replicas==4 else 'gemm_eight'
    manifest = json.loads((root/f'{directory}/manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Rectangular GEMM source changed; rebuild the kernel')
    data = (root/f'{directory}/rect-f32.elf').read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest['elf_sha256']:
        raise RuntimeError('Rectangular GEMM ELF differs from its manifest')
    return data


def _tile(a, b, cache, start, column, replicas):
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    capacity = rect_gemm_capacity((m, k, n))
    key = ('rect_gemm_prototype' if replicas==4 else 'rect_gemm_eight', 'f32', (capacity,))
    factory=lambda:compile_rect_gemm((m,k,n),_kernel(replicas),libraries=npu._native,replicas=replicas)
    npu._register_graph_factory(key,factory)
    packed = []
    for tag, parts in enumerate(([*[npu._scalar(v, torch.float32) for v in (12345, m, k, n)], a.view(-1)], [b.view(-1)])):
        cache_key = ('a', start, n, capacity) if tag == 0 else ('b', column, capacity)
        if cache_key not in cache:
            remaining = capacity-sum(v.numel() for v in parts)
            if remaining:
                parts.append(npu._broadcast(npu._scalar(0, torch.float32), (remaining,)))
            value = npu._concat(parts, (capacity,), torch.float32)
            rows = 2 if replicas==4 else 8
            cache[cache_key] = npu._broadcast(value.view(1, capacity), (replicas, capacity)).view(replicas, rows, capacity//rows)
        packed.append(cache[cache_key])
    rows = 2 if replicas==4 else 8
    partials = npu._run(key, packed, (replicas, rows, capacity//rows), torch.float32).view(replicas, capacity)
    # Each result has exactly one producing tile. DMA copies its row bits;
    # selecting <=4096 elements stays below the existing 8192 Gather ceiling.
    import numpy as np
    positions = np.arange(m*n,dtype=np.int64).reshape(m,n)
    owners = np.arange(m,dtype=np.int64)[:,None] % replicas
    if replicas==8:
        # Keep Gather's verified source bound: transpose full storage with DMA,
        # then trim to the <=32768 useful interleaved result elements.
        source = npu._slice(npu._materialize(partials.t()),0,m*n*8)
        indices = positions*8+owners
    else:
        source, indices = partials, owners*capacity+positions
    return npu._run(('gather_tile','f32',(source.numel(),),indices.tobytes()),
                    [source],(m,n),torch.float32)


def matrix_fp32(x, y, *, out=None, replicas=8):
    """M/N/K <=256, with M/N tiling and unchanged full-K compensated order."""
    with npu._lock:
        npu.synchronize()
        if type(replicas) is not int or replicas not in (4,8):
            raise ValueError('GEMM requires four or eight complete replicas')
        m, k, n = npu._matrix_shape(x, y)
        if max(m, k, n) > 256:
            raise ValueError('Experimental rectangular GEMM requires M/N/K <=256')
        npu._check_out(out, (m, n), torch.float32, (x, y))
        left, right = (npu._cast(npu._materialize(v), torch.float32) for v in (x, y))
        columns = min(n, 64)
        rows = min(m, 64, (16384-4)//k)
        column_major = npu._materialize(right.t()) if columns < n else None
        blocks = [right if columns == n else npu._materialize(
            npu._slice(column_major, col*k, min(columns, n-col)*k).view(min(columns, n-col), k).t())
            for col in range(0, n, columns)]
        tiles, cache = [], {}
        for start in range(0, m, rows):
            count = min(rows, m-start)
            a = npu._slice(left, start*k, count*k).view(count, k)
            pieces = [_tile(a, b, cache, start, column, replicas) for column, b in enumerate(blocks)]
            tiles.append(pieces[0] if len(pieces) == 1 else npu._materialize(
                npu._concat([npu._materialize(t.t()) for t in pieces], (n, count), torch.float32).t()))
        return npu._concat(tiles, (m, n), torch.float32, out)

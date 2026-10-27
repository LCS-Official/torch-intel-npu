"""Bounded rank 2-4 matmul, standard SDPA and multihead attention training."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/sdpa'
CASES = ('matmul', 'matmul-large', 'sdpa', 'sdpa-masks', 'layouts', 'contracts', 'training')


def execute(case):
    import torch
    import torch_intel_npu as npu
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.set_num_threads(1)
    torch.manual_seed(89)
    torch.backends.mkldnn.enabled = False
    checks = []

    def close(a, b):
        a, b = a.detach().cpu(), b.detach().cpu()
        torch.testing.assert_close(a, b, rtol=.003 if a.dtype == torch.float16 else 3e-5,
                                   atol=.0005 if a.dtype == torch.float16 else 3e-6, equal_nan=True)

    def upload(x):
        return x.detach().to('intel_npu', memory_format=torch.contiguous_format).requires_grad_(x.requires_grad)

    def copies_since(before, error_flags=0):
        after = npu.stats()
        copies = sum(after[k]-before[k] for k in ('host_upload_copies', 'host_download_copies', 'host_device_copies'))
        scalars = after['cached_scalar_buffers']-before['cached_scalar_buffers']
        assert copies == scalars+error_flags, (copies, scalars, error_flags)
        assert after['npu_calls'] > before['npu_calls']
        return {'copies': copies, 'scalar_uploads': scalars, 'error_flag_reads': error_flags}

    def check(name, function):
        before = npu.stats()
        row = {'name': name}
        try:
            row.update(function() or {})
            row['status'] = 'passed'
        except Exception as error:
            row.update(status='failed', error=str(error), traceback=traceback.format_exc())
        gc.collect()
        after = npu.stats()
        row['npu_calls'] = after['npu_calls']-before['npu_calls']
        row['live_buffer_delta'] = after['live_buffers']-after['cached_scalar_buffers']-before['live_buffers']+before['cached_scalar_buffers']
        if row['live_buffer_delta'] or after['context_failed']:
            row.update(status='failed', lifecycle_error='Buffer leak or failed context')
        checks.append(row)
        print(name, row['status'], row.get('error', '')[:400], flush=True)

    def matmul(dtype, large=False):
        specs = (((2, 1, 16, 257), (1, 2, 257, 17)), ((1, 1, 65537), (1, 65537, 1))) if large else (
            ((2, 3), (3, 4)), ((2, 2, 3), (1, 3, 4)), ((2, 3), (3, 3, 4)),
            ((2, 2, 3), (3, 4)), ((2, 1, 2, 3), (1, 3, 3, 4)),
            ((2, 1, 2, 3), (3, 3, 4)), ((2, 3), (2, 3, 3, 4)))
        records = []
        for ashape, bshape in specs:
            for need_a, need_b in (((True, True),) if large else ((True, True), (True, False), (False, True))):
                a, b = [(torch.randn(shape, dtype=dtype)/4).requires_grad_(need)
                        for shape, need in ((ashape, need_a), (bshape, need_b))]
                x, y = upload(a), upload(b)
                if not large:
                    x = x.detach().transpose(-2, -1).contiguous().transpose(-2, -1).requires_grad_(need_a)
                    y = y.detach().transpose(-2, -1).contiguous().transpose(-2, -1).requires_grad_(need_b)
                expected = torch.matmul(a, b)
                seed = torch.randn_like(expected)/16
                g = seed.to('intel_npu')
                before = npu.stats()
                actual = x@y
                actual.backward(g)
                records.append(copies_since(before))
                expected.backward(seed)
                close(actual, expected)
                if need_a:
                    close(x.grad, a.grad)
                if need_b:
                    close(y.grad, b.grad)
                with torch.no_grad():
                    out = torch.empty_like(actual)
                    assert torch.matmul(x, y, out=out) is out
                    close(out, expected)
        return {'operand_shapes': specs, 'forward_out_broadcast_gradients': True, 'compute_copies': records}

    def sdpa(dtype, masked=False):
        if masked:
            specifications = [('additive', (2, 2, 3, 4), (1, 2, 5, 4), (2, 1, 5, 2), False, None),
                              ('mixed-mask', (2, 2, 3, 4), (2, 2, 5, 4), (2, 2, 5, 2), False, .7),
                              ('fully-masked', (1, 2, 3, 4), (1, 2, 5, 4), (1, 2, 5, 2), False, None),
                              ('nonfinite', (1, 1, 3, 4), (1, 1, 5, 4), (1, 1, 5, 2), False, None)]
        else:
            specifications = [('rank3', (2, 3, 4), (2, 5, 4), (2, 5, 2), False, None),
                              ('rank4', (2, 2, 3, 4), (2, 2, 5, 4), (2, 2, 5, 2), False, None),
                              ('broadcast', (2, 1, 3, 4), (1, 2, 5, 4), (2, 2, 5, 2), False, .7),
                              ('causal-short', (1, 2, 3, 4), (1, 2, 5, 4), (1, 2, 5, 2), True, None),
                              ('causal-long', (1, 2, 5, 4), (1, 2, 3, 4), (1, 2, 3, 2), True, -.7),
                              ('zero-scale', (1, 2, 3, 4), (1, 2, 5, 4), (1, 2, 5, 2), False, 0.),
                              ('score-limit', (1, 1, 256, 1), (1, 1, 256, 1), (1, 1, 256, 1), True, None)]
            if dtype == torch.float32:
                specifications.append(('prescale-overflow', (1, 1, 1, 1), (1, 1, 2, 1), (1, 1, 2, 1), False, 1e-20))
        copies, results = [], []
        for name, qshape, kshape, vshape, causal, scale in specifications:
            values = [(torch.randn(shape, dtype=dtype)/3).requires_grad_() for shape in (qshape, kshape, vshape)]
            if name == 'prescale-overflow':
                with torch.no_grad():
                    values[0].fill_(1e20)
                    values[1].copy_(torch.tensor([1e20, 0.]).reshape(kshape))
            mask = None
            if masked:
                mask_dtype = torch.float32 if name == 'mixed-mask' else dtype
                mask = torch.randn(qshape[-2], kshape[-2], dtype=mask_dtype)/4
                mask[:, -1] = -float('inf')
                if name == 'fully-masked':
                    mask[0] = -float('inf')
                if name == 'nonfinite':
                    mask[0] = float('nan')
                    mask[1, 0] = float('inf')
                mask.requires_grad_()
            inputs = [upload(x) for x in values]
            # Exercise dense head/sequence permutations accepted by ordinary multihead models.
            if not masked and len(qshape) == 4:
                inputs = [x.detach().transpose(1, 2).contiguous().transpose(1, 2).requires_grad_() for x in inputs]
            nmask = upload(mask) if mask is not None else None
            options = dict(is_causal=causal, scale=scale)
            with sdpa_kernel(SDPBackend.MATH):
                expected = torch.nn.functional.scaled_dot_product_attention(*values, attn_mask=mask, **options)
            seed = torch.randn_like(expected)/4
            g = seed.to('intel_npu')
            before = npu.stats()
            actual = torch.nn.functional.scaled_dot_product_attention(*inputs, attn_mask=nmask, **options)
            actual.backward(g)
            copies.append(copies_since(before))
            expected.backward(seed)
            close(actual, expected)
            for x, ref in zip(inputs, values):
                close(x.grad, ref.grad)
            if mask is not None:
                close(nmask.grad, mask.grad)
            with torch.no_grad():
                close(torch.nn.functional.scaled_dot_product_attention(*inputs, attn_mask=nmask, **options), expected)
            if name == 'prescale-overflow':
                assert bool(torch.isfinite(actual.detach().cpu()).all())
            if name == 'fully-masked':
                assert torch.count_nonzero(actual.detach().cpu()[..., 0, :]) == 0
                assert torch.count_nonzero(inputs[0].grad.cpu()[..., 0, :]) == 0
                assert torch.count_nonzero(nmask.grad.cpu()[0]) == 0
            results.append(name)
        # Test one-sided gradients without imposing Q/K requires_grad for value-only training.
        for index in range(3):
            values = [(torch.randn(1, 2, 3, dtype=dtype)/3).requires_grad_(i == index) for i in range(3)]
            inputs = [upload(x) for x in values]
            with sdpa_kernel(SDPBackend.MATH):
                expected = torch.nn.functional.scaled_dot_product_attention(*values)
            actual = torch.nn.functional.scaled_dot_product_attention(*inputs)
            seed = torch.ones_like(expected)/7
            actual.backward(seed.to('intel_npu'))
            expected.backward(seed)
            close(actual, expected)
            close(inputs[index].grad, values[index].grad)
        return {'cases': results, 'compute_copies': copies, 'one_sided_gradients': True,
                'mask_gradients': masked, 'cpu_reference': 'Torch math SDPA'}

    def layouts(dtype):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        copies = []
        for shape in ((2, 3, 4), (2, 2, 3, 4)):
            for fmt in (None, torch.preserve_format, torch.contiguous_format):
                base = torch.randn(shape, dtype=dtype)
                host = base.transpose(1, 2).detach().requires_grad_()
                x = base.to('intel_npu').transpose(1, 2).detach().requires_grad_()
                seed = torch.randn_like(host).contiguous()/4
                g = seed.to('intel_npu')
                expected = torch.clone(host, memory_format=fmt)
                before = npu.stats()
                actual = torch.clone(x, memory_format=fmt)
                actual.backward(g)
                copies.append(copies_since(before))
                expected.backward(seed)
                assert actual.stride() == expected.stride() and actual.data_ptr() != x.data_ptr()
                close(actual, expected)
                close(x.grad, host.grad)
        base = torch.randn(2, 2, 3, 4, dtype=dtype).requires_grad_()
        x = upload(base)
        expected = base.transpose(1, 2).contiguous().view(2, 3, 8)
        before = npu.stats()
        actual = x.transpose(1, 2).contiguous().view(2, 3, 8)
        actual.sum().backward()
        copies.append(copies_since(before))
        expected.sum().backward()
        close(actual, expected)
        close(x.grad, base.grad)
        assert x.contiguous() is x
        clone = x.clone()
        assert clone.data_ptr() != x.data_ptr()
        close(clone, x)
        leaves = [torch.randn(2, 7, 5, dtype=dtype).requires_grad_() for _ in range(2)]
        seeds = [torch.randn(2, 5, 7, dtype=dtype) for _ in range(2)]
        tensors = [upload(t) for t in leaves]
        outputs = [t.clone() for t in tensors]
        gradients = [g.to('intel_npu').transpose(1, 2) for g in seeds]
        gate = Barrier(2)
        before = npu.stats()
        def backward(index):
            gate.wait(timeout=20)
            outputs[index].backward(gradients[index])
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(backward, index) for index in range(2)]
            for future in futures:
                future.result(timeout=60)
        copies.append(copies_since(before))
        created = npu.stats()['session_creations']-before['session_creations']
        assert created == 1, created
        for tensor, seed in zip(tensors, seeds):
            close(tensor.grad, seed.transpose(1, 2))
        return {'dense_clone_formats_and_gradients': True, 'head_merge_contiguous_view_backward': True,
                'distinct_clone_storage': True, 'concurrent_backward_threads': 2,
                'shared_permutation_session_creations': created, 'compute_copies': copies}

    def contracts():
        q = torch.ones(2, 2, 3, 4).to('intel_npu')
        k = torch.ones(2, 2, 5, 4).to('intel_npu')
        v = torch.ones(2, 2, 5, 2).to('intel_npu')
        mask, badmask = torch.zeros(3, 5).to('intel_npu'), torch.zeros(2, 3).to('intel_npu')
        half_k = k.half()
        wrong = torch.empty_like(q)
        bigq = torch.ones(1, 1, 8193, 1).to('intel_npu')
        bigk = torch.ones(1, 1, 16384, 1).to('intel_npu')
        gqa_bad = torch.ones(2,3,3,4).to('intel_npu')
        large_a = torch.ones(8193, 1, 1).to('intel_npu')
        large_b = torch.ones(1, 1, 16384).to('intel_npu')
        # Each input fits; broadcasting left over two batch axes would exceed the workspace bound.
        workspace_a = torch.ones(1, 65, 1, 65536).to('intel_npu')
        workspace_b = torch.ones(33, 1, 65536, 1).to('intel_npu')
        square = torch.ones(2, 2, 4, 4).to('intel_npu')
        cpu_mask = torch.ones(3, 5)
        scalar = torch.empty((), device='intel_npu')
        before, rejected = npu.stats()['npu_calls'], []
        F = torch.nn.functional.scaled_dot_product_attention
        actions = {
            'matmul-scalar': lambda: scalar@scalar,
            'mm-stays-2d': lambda: torch.mm(q.view(4, 3, 4), k.view(4, 4, 5)),
            'matmul-batch': lambda: q@k.view(1, 4, 5, 4).transpose(-2, -1),
            'matmul-inner': lambda: q@k,
            'matmul-dtype': lambda: q@half_k.transpose(-2, -1),
            'matmul-output-bound': lambda: large_a@large_b,
            'matmul-workspace-bound': lambda: workspace_a@workspace_b,
            'matmul-out-shape': lambda: torch.matmul(q, k.transpose(-2, -1), out=wrong),
            'matmul-alias': lambda: torch.matmul(q, square, out=q),
            'clone-channels-last': lambda: torch.clone(q, memory_format=torch.channels_last),
            'differentiable-cpu-download': lambda: q.detach().requires_grad_().to('cpu'),
            'sdpa-invalid-dropout': lambda: F(q, k, v, dropout_p=1.01),
            'sdpa-gqa-heads': lambda: F(gqa_bad, k, v, enable_gqa=True),
            'sdpa-mask-causal': lambda: F(q, k, v, attn_mask=mask, is_causal=True),
            'sdpa-cpu-float-mask': lambda: F(q, k, v, attn_mask=cpu_mask),
            'sdpa-bool-mask': lambda: F(q, k, v, attn_mask=cpu_mask.bool()),
            'sdpa-mask-shape': lambda: F(q, k, v, attn_mask=badmask),
            'sdpa-rank': lambda: F(q.view(-1), k.view(20, 4), v.view(20, 2)),
            'sdpa-dtype': lambda: F(q, half_k, v),
            'sdpa-inner': lambda: F(q, v, v),
            'sdpa-sequence': lambda: F(q, k, q),
            'sdpa-score-bound': lambda: F(bigq, bigk, bigk),
            'sdpa-nan-scale': lambda: F(q, k, v, scale=float('nan')),
            'sdpa-inf-scale': lambda: F(q, k, v, scale=float('inf')),
        }
        for name, function in actions.items():
            try:
                function()
            except (ValueError, RuntimeError, TypeError, NotImplementedError):
                rejected.append(name)
            else:
                raise AssertionError(f'Accepted {name}')
        assert npu.stats()['npu_calls'] == before
        for name in ('matmul', 'sdpa'):
            a = q.detach().requires_grad_()
            out = a@k.transpose(-2, -1) if name == 'matmul' else F(a, k, v)
            loss = out.sum()
            with torch.no_grad():
                a.add_(1)
            try:
                loss.backward()
            except RuntimeError as error:
                assert 'version' in str(error) or 'inplace' in str(error)
            else:
                raise AssertionError(f'{name} saved tensor guard bypassed')
        return {'rejected_without_submit': rejected, 'saved_version_guards': True}

    def train(dtype):
        class TinyMultihead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1, self.norm2 = torch.nn.LayerNorm(4), torch.nn.LayerNorm(4)
                self.q, self.k, self.v, self.proj = (torch.nn.Linear(4, 4) for _ in range(4))
                self.ff = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.GELU(), torch.nn.Linear(6, 4))
                self.head = torch.nn.Linear(4, 3)

            def forward(self, x):
                h = self.norm1(x)
                q, k, v = [layer(h).view(x.shape[0], x.shape[1], 2, 2).transpose(1, 2) for layer in (self.q, self.k, self.v)]
                h = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                h = h.transpose(1, 2).contiguous().view(x.shape)
                h = x+self.proj(h)
                h = h+self.ff(self.norm2(h))
                return self.head(h.mean(1))

        cpu = TinyMultihead().to(dtype)
        model = copy.deepcopy(cpu).to('intel_npu')
        host = torch.randn(2, 3, 4, dtype=dtype)/3
        target = torch.tensor([0, 2])
        data, labels = host.to('intel_npu'), target.to('intel_npu')
        reference, opt = (torch.optim.SGD(m.parameters(), lr=.025, foreach=False, fused=False) for m in (cpu, model))
        losses, copies = [], []
        for _ in range(4):
            reference.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            with sdpa_kernel(SDPBackend.MATH):
                expected = cpu(host)
            cpu_loss = torch.nn.functional.cross_entropy(expected.float(), target)
            cpu_loss.backward()
            reference.step()
            before = npu.stats()
            actual = model(data)
            loss = torch.nn.functional.cross_entropy(actual.float(), labels)
            loss.backward()
            opt.step()
            copies.append(copies_since(before, error_flags=1))
            close(actual, expected)
            close(loss, cpu_loss)
            for name, a in model.named_parameters():
                b = dict(cpu.named_parameters())[name]
                try:
                    close(a.grad, b.grad)
                    close(a, b)
                except AssertionError as error:
                    raise AssertionError(f'step={len(losses)+1}, parameter={name}: {error}') from error
            losses.append(float(loss.detach().cpu()))
        assert losses[-1] < losses[0]
        return {'steps': 4, 'losses': losses, 'heads': 2, 'causal': True, 'all_parameters_and_gradients': True,
                'explicit_fp32_loss': True, 'step_copies': copies}

    if case == 'contracts':
        check('contracts', contracts)
    else:
        for dtype in (torch.float16, torch.float32):
            check(f'{case}-{dtype}', lambda d=dtype: matmul(d, case == 'matmul-large') if case.startswith('matmul') else (
                sdpa(d, case == 'sdpa-masks') if case.startswith('sdpa') else layouts(d) if case == 'layouts' else train(d)))
    gc.collect()
    before = npu.stats()
    npu.shutdown()
    final = npu.stats()
    assert final['closed'] and not final['context_failed']
    assert all(final[k] == 0 for k in ('live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'quarantined_buffer_bytes', 'cached_sessions'))
    return {'case': case, 'checks': checks, 'status': 'failed' if any(c['status'] == 'failed' for c in checks) else 'passed',
            'stats_before_shutdown': before, 'final_stats': final, 'python': sys.version,
            'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                'torch_intel_npu.py', 'torch_npu_autograd.py', 'torch_npu_sdpa_demo.py',
                'npunlock/tests/fixtures/npu4000/resident_sdpa_softmax.c', 'npunlock/tests/fixtures/npu4000/resident/manifest.json')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result = execute(args.case)
    (OUT/f'{args.case}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    raise SystemExit(result['status'] == 'failed')

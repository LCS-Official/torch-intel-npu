"""Common activation/math migration, gradients and gated-model hardware checks."""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'npunlock-probe/research/results/unary-math'
KINDS = ('exp', 'log', 'tanh', 'sigmoid', 'silu', 'sqrt', 'rsqrt')


def gated_model(dtype):
    import torch
    class Gated(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.RMSNorm(8)
            self.gate = torch.nn.Linear(8, 12)
            self.up = torch.nn.Linear(8, 12)
            self.down = torch.nn.Linear(12, 8)
            self.head = torch.nn.Linear(8, 3)
            self.silu = torch.nn.SiLU(); self.tanh = torch.nn.Tanh(); self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            value = self.norm(x)
            hidden = self.silu(self.gate(value))*self.tanh(self.up(value))
            return self.head(self.sigmoid(x+self.down(hidden)).mean(1))
    return Gated().to(dtype=dtype)


def decoder_model(dtype):
    import torch
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Linear(5, 8); self.memory = torch.nn.Linear(5, 8)
            layer = torch.nn.TransformerDecoderLayer(8, 2, 16, .25, activation='gelu', batch_first=True)
            self.decoder = torch.nn.TransformerDecoder(layer, 2, norm=torch.nn.LayerNorm(8))
            self.head = torch.nn.Linear(8, 3)
            self.register_buffer('causal', torch.ones(3, 3, dtype=torch.bool).triu(1))
            self.register_buffer('padding', torch.tensor([[False, False, False, True], [False, False, True, True]]))
        def forward(self, x, memory):
            value = self.decoder(self.input(x), self.memory(memory), tgt_mask=self.causal,
                                 memory_key_padding_mask=self.padding, tgt_is_causal=True)
            return self.head(value.mean(1))
    return Decoder().to(dtype=dtype)


def execute(case, precision):
    import torch
    import torch_intel_npu as npu
    torch.set_num_threads(1); torch.manual_seed(251)
    dtype = getattr(torch, precision); checks = []
    function = lambda name, x: torch.nn.functional.silu(x) if name == 'silu' else getattr(torch, name)(x)
    def upload(host):
        return host.detach().to('intel_npu', memory_format=torch.contiguous_format).requires_grad_(host.requires_grad)
    def close(actual, expected, *, strict=False):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach(), equal_nan=True,
            rtol=3e-5 if strict or dtype == torch.float32 else .003,
            atol=3e-6 if strict or dtype == torch.float32 else .0005)
    def no_transfer(before):
        after = npu.stats()
        assert all(after[k] == before[k] for k in ('host_download_copies', 'host_device_copies'))
        assert after['host_upload_copies']-before['host_upload_copies'] == after['cached_scalar_buffers']-before['cached_scalar_buffers']
    def check(name, action):
        before = npu.stats(); row = dict(name=name)
        try: row.update(action() or {}); row['status'] = 'passed'
        except Exception as error: row.update(status='failed', error=str(error), traceback=traceback.format_exc())
        gc.collect(); after = npu.stats()
        row['npu_calls'] = after['npu_calls']-before['npu_calls']
        if after['live_buffers']-after['cached_scalar_buffers'] != before['live_buffers']-before['cached_scalar_buffers']:
            row.update(status='failed', lifecycle_error='Leaked tensor storage')
        checks.append(row); print(name, row['status'], row.get('error', '')[:200], flush=True)

    def probe():
        rows = []
        for name in KINDS:
            x = upload(torch.full((2, 3), .5, dtype=dtype, requires_grad=True))
            try:
                y = function(name, x); y.sum().backward()
                rows.append(dict(name=name, status='passed'))
            except Exception as error:
                rows.append(dict(name=name, status='blocked', error=str(error).splitlines()[0][:240]))
        return dict(observations=rows)

    def values():
        for name in KINDS:
            for shape in ((), (2, 3), (2, 3, 4), (1, 2, 3, 4)):
                host = (torch.rand(shape)+.25 if name in ('log', 'sqrt', 'rsqrt') else torch.randn(shape)*2).to(dtype).requires_grad_()
                x = upload(host); seed = torch.randn_like(host)/4; gradient = upload(seed)
                before = npu.stats(); actual = function(name, x); actual.backward(gradient); no_transfer(before)
                expected = function(name, host); expected.backward(seed)
                close(actual, expected); close(x.grad, host.grad)
            special = torch.tensor([float('-inf'), -4., -0., 0., 2.**-24, 2.**-149, 1., 4., float('inf'), float('nan')], dtype=dtype).requires_grad_()
            x = upload(special); seed = upload(torch.ones_like(special)); before = npu.stats()
            actual = function(name, x); actual.backward(seed); no_transfer(before)
            expected = function(name, special); expected.backward(torch.ones_like(special))
            close(actual, expected); close(x.grad, special.grad)
            for a, b in ((actual.detach().cpu(), expected.detach()), (x.grad.cpu(), special.grad)):
                zeros = (b == 0) & (a == 0)
                assert torch.equal(torch.signbit(a[zeros]), torch.signbit(b[zeros])), name
        return dict(operators=KINDS, scalar_to_rank4=True, first_order=True, nonfinite_and_signed_zero=True)

    def layouts():
        for name in KINDS:
            base = torch.linspace(.25, 1.25, 24, dtype=dtype).reshape(4, 6)
            x = upload(base)
            for v, host in ((x.t(), base.t()), (x[:, 1:5], base[:, 1:5]), (x[1:2, :].expand(3, 6), base[1:2, :].expand(3, 6))):
                out = torch.empty(v.shape, dtype=dtype, device='intel_npu')
                before = npu.stats(); actual = function(name, v)
                assert getattr(torch.ops.aten, name).out(v, out=out) is out
                no_transfer(before); close(actual, function(name, host)); close(out, function(name, host))
            big = upload(torch.full((65537,), .5, dtype=dtype)); before = npu.stats()
            result = function(name, big); no_transfer(before); close(result, function(name, torch.full((65537,), .5, dtype=dtype)))
        return dict(operators=KINDS, dense_partial_broadcast_reads=True, out=True, tiled_65537=True)

    def edges():
        import numpy as np
        if dtype == torch.float16:
            host = torch.from_numpy(np.arange(65536, dtype=np.uint16).view(np.float16))
        else:
            bits = np.random.default_rng(263).integers(0, 2**32, 32768, dtype=np.uint32)
            host = torch.cat((torch.from_numpy(bits.view(np.float32)), torch.tensor([
                -104., -103.97208, -88.72284, 88.72283, 88.72284, 89.,
                2.**-88, 2.**-87, 1e-4, 1e-6, 1.-2.**-24, 1., 1.+2.**-23])))
        host = host.requires_grad_(); source = upload(host); seed = torch.full_like(host, .125); g = upload(seed)
        maxima = {}
        for name in KINDS:
            host.grad = None; source.grad = None; before = npu.stats()
            actual = function(name, source); actual.backward(g); no_transfer(before)
            expected = function(name, host); expected.backward(seed)
            close(actual, expected); close(source.grad, host.grad)
            for a, b in ((actual.detach().cpu(), expected.detach()), (source.grad.cpu(), host.grad)):
                assert torch.equal(torch.isnan(a), torch.isnan(b)), name
                assert torch.equal(torch.isinf(a), torch.isinf(b)), name
            a, b = actual.detach().cpu().float(), expected.detach().float()
            normal = torch.isfinite(a) & torch.isfinite(b) & (b.abs() >= 1e-3)
            maxima[name] = float(((a[normal]-b[normal]).abs()/b[normal].abs()).max()) if normal.any() else 0.
        return dict(elements=host.numel(), all_half_encodings=dtype == torch.float16,
                    infinity_and_nan_masks_exact=True, max_relative_forward_above_1e_minus3=maxima)

    def norms():
        count = 0
        for shape, normalized in (((3, 4), (4,)), ((2, 3, 4), (3, 4)), ((2, 2, 3, 4), (4,))):
            for epsilon in (None, 0., 1e-5):
                for wd in (None, dtype, torch.float32) if dtype == torch.float16 else (None, dtype):
                    host = (torch.randn(shape)/4).to(dtype).requires_grad_()
                    weight = torch.randn(normalized, dtype=wd, requires_grad=True) if wd else None
                    x = upload(host); w = upload(weight) if weight is not None else None
                    seed = torch.randn_like(host)/8; g = upload(seed); before = npu.stats()
                    actual = torch.rms_norm(x, normalized, w, epsilon); actual.backward(g); no_transfer(before)
                    expected = torch.rms_norm(host, normalized, weight, epsilon); expected.backward(seed)
                    close(actual, expected); close(x.grad, host.grad)
                    if w is not None: close(w.grad, weight.grad, strict=wd == torch.float32)
                    assert actual.dtype == host.dtype; count += 1
        return dict(cases=count, default_epsilon_fp32=True, weight_and_input_gradients=True, mixed_weight=True)

    def contracts():
        x = upload(torch.full((2, 3), .5, dtype=dtype)); other = upload(torch.ones(3, dtype=dtype))
        bad = torch.empty(6, dtype=dtype, device='intel_npu'); alias = x
        before = npu.stats(); rejected = 0
        for name in KINDS:
            for action in (lambda: getattr(torch.ops.aten, name).out(x, out=bad),
                           lambda: getattr(torch.ops.aten, name).out(x, out=alias)):
                try: action()
                except (ValueError, RuntimeError): rejected += 1
                else: raise AssertionError('Invalid unary output accepted')
        for action in (lambda: torch.rms_norm(x, (2,), other), lambda: torch.rms_norm(x, (3,), bad),
                       lambda: torch.rms_norm(x, (3,), eps=-1.), lambda: torch.rms_norm(x, (3,), eps=float('nan')),
                       lambda: x.sigmoid_(), lambda: x.tanh_(), lambda: torch.nn.functional.silu(x, inplace=True)):
            try: action()
            except (ValueError, RuntimeError): rejected += 1
            else: raise AssertionError('Invalid RMSNorm accepted')
        assert npu.stats()['npu_calls'] == before['npu_calls']; no_transfer(before)
        for name in KINDS:
            host = torch.full((2, 3), .5, dtype=dtype, requires_grad=True); leaf = upload(host)
            actual = function(name, leaf)
            saved = leaf if name in ('log', 'silu') else actual
            with torch.no_grad(): saved.add_(torch.ones_like(saved))
            before = npu.stats()
            try: actual.backward(upload(torch.ones_like(host)))
            except RuntimeError as error: assert 'modified' in str(error) and 'inplace' in str(error)
            else: raise AssertionError('Missing saved-tensor version check')
            assert npu.stats()['npu_calls'] == before['npu_calls']
        return dict(metadata_rejections=rejected, version_guards=len(KINDS))

    def training():
        from torch_npu_transformer_demo import cpu_tree
        cpu = gated_model(dtype); model = copy.deepcopy(cpu).to('intel_npu')
        host = torch.randn(2, 3, 8, dtype=dtype)/4; target = torch.randn(2, 3)/4
        x, labels = upload(host), upload(target)
        reference, opt = [torch.optim.SGD(net.parameters(), lr=.02, momentum=.5, foreach=False, fused=False) for net in (cpu, model)]
        losses = []; checkpoint = io.BytesIO()
        for step in range(3):
            reference.zero_grad(set_to_none=True); opt.zero_grad(set_to_none=True)
            expected = torch.nn.functional.mse_loss(cpu(host).float(), target); expected.backward()
            before = npu.stats(); actual = torch.nn.functional.mse_loss(model(x).float(), labels); actual.backward(); no_transfer(before)
            close(actual, expected)
            for a, b in zip(model.parameters(), cpu.parameters()): close(a.grad, b.grad)
            reference.step(); opt.step()
            for a, b in zip(model.parameters(), cpu.parameters()): close(a, b)
            losses.append(actual.item())
            state = dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(opt.state_dict()))
            if step == 1: torch.save(state, checkpoint)
        checkpoint.seek(0); saved = torch.load(checkpoint, weights_only=True)
        restored = gated_model(dtype).to('intel_npu'); restored.load_state_dict(saved['model'])
        restored_opt = torch.optim.SGD(restored.parameters(), lr=.02, momentum=.5, foreach=False, fused=False)
        restored_opt.load_state_dict(saved['optimizer']); restored_opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(restored(x).float(), labels); loss.backward(); restored_opt.step()
        assert loss.item() == losses[-1]
        for name, tensor in cpu_tree(restored.state_dict()).items():
            torch.testing.assert_close(tensor, state['model'][name], rtol=0, atol=0)
        for key, value in cpu_tree(restored_opt.state_dict())['state'].items():
            torch.testing.assert_close(value['momentum_buffer'], state['optimizer']['state'][key]['momentum_buffer'], rtol=0, atol=0)
        model.eval(); cpu.eval()
        with torch.no_grad(): close(model(x), cpu(host))
        return dict(steps=3, parameters=sum(p.numel() for p in model.parameters()), losses=losses,
                    all_parameter_gradients=True, checkpoint_bitwise=True, eval=True)

    def decoder():
        from torch_npu_standard_modules_demo import cpu_rng_reference
        from torch_npu_transformer_demo import cpu_tree
        cpu = decoder_model(dtype); model = copy.deepcopy(cpu).to('intel_npu')
        hosts = [torch.randn(2, length, 5, dtype=dtype)/4 for length in (3, 4)]
        inputs = [upload(host) for host in hosts]; target = torch.randn(2, 3)/8; labels = upload(target)
        reference, opt = [torch.optim.SGD(net.parameters(), lr=.02, momentum=.5, foreach=False, fused=False) for net in (cpu, model)]
        checkpoint = io.BytesIO(); losses = []
        for step in range(2):
            reference.zero_grad(set_to_none=True); opt.zero_grad(set_to_none=True)
            with torch.backends.mkldnn.flags(enabled=False), cpu_rng_reference(torch.intel_npu.get_rng_state()) as tracker:
                expected = torch.nn.functional.mse_loss(cpu(*hosts).float(), target); expected.backward()
            cpu_rng = torch.get_rng_state(); before = npu.stats()
            actual = torch.nn.functional.mse_loss(model(*inputs).float(), labels); actual.backward()
            after = npu.stats()
            assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(tracker.get_state(), torch.intel_npu.get_rng_state())
            assert after['host_download_copies'] == before['host_download_copies']
            assert after['host_device_copies'] == before['host_device_copies']
            assert after['host_upload_copies']-before['host_upload_copies'] == 12+after['cached_scalar_buffers']-before['cached_scalar_buffers']
            close(actual, expected)
            for (name, a), (_, b) in zip(model.named_parameters(), cpu.named_parameters()):
                try: close(a.grad, b.grad)
                except AssertionError as error: raise AssertionError(f'decoder gradient {name}: {error}') from error
            reference.step(); opt.step()
            for a, b in zip(model.parameters(), cpu.parameters()): close(a, b)
            losses.append(actual.item())
            state = dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(opt.state_dict()), rng=torch.intel_npu.get_rng_state())
            if step == 0: torch.save(state, checkpoint)
        checkpoint.seek(0); saved = torch.load(checkpoint, weights_only=True)
        restored = decoder_model(dtype).to('intel_npu'); restored.load_state_dict(saved['model'])
        replay_opt = torch.optim.SGD(restored.parameters(), lr=.02, momentum=.5, foreach=False, fused=False)
        replay_opt.load_state_dict(saved['optimizer']); torch.intel_npu.set_rng_state(saved['rng'])
        loss = torch.nn.functional.mse_loss(restored(*inputs).float(), labels); loss.backward(); replay_opt.step()
        assert loss.item() == losses[-1] and torch.equal(state['rng'], torch.intel_npu.get_rng_state())
        for name, tensor in cpu_tree(restored.state_dict()).items():
            torch.testing.assert_close(tensor, state['model'][name], rtol=0, atol=0)
        for key, value in cpu_tree(replay_opt.state_dict())['state'].items():
            torch.testing.assert_close(value['momentum_buffer'], state['optimizer']['state'][key]['momentum_buffer'], rtol=0, atol=0)
        cpu.eval(); model.eval(); rng = torch.intel_npu.get_rng_state()
        with torch.no_grad(): close(model(*inputs), cpu(*hosts))
        assert torch.equal(rng, torch.intel_npu.get_rng_state())
        return dict(layers=2, parameters=sum(p.numel() for p in model.parameters()), losses=losses,
                    self_and_cross_attention=True, same_mask_dropout=.25, checkpoint_bitwise=True, eval=True)

    actions = dict(probe=probe, values=values, layouts=layouts, edges=edges, norms=norms, contracts=contracts, training=training, decoder=decoder)
    check(case, actions[case])
    gc.collect(); before = npu.stats(); npu.shutdown(); final = npu.stats()
    healthy = final['closed'] and not final['context_failed'] and all(final[k] == 0 for k in (
        'live_buffers', 'cached_buffer_blocks', 'cached_scalar_buffers', 'cached_sessions', 'quarantined_buffer_bytes'))
    sources = ('torch_intel_npu.py', 'torch_npu_autograd.py', 'torch_npu_unary_math_demo.py', 'npunlock/tests/fixtures/npu4000/resident/manifest.json')
    return dict(status='passed' if healthy and all(r['status'] == 'passed' for r in checks) else 'failed', checks=checks,
        before_shutdown=before, final_stats=final, source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all', 'probe', 'values', 'layouts', 'edges', 'norms', 'contracts', 'training', 'decoder'), default='all')
    parser.add_argument('--precision', choices=('all', 'float16', 'float32'), default='all')
    parser.add_argument('--label', default='after'); args = parser.parse_args()
    if args.case == 'all' or args.precision == 'all':
        for case in ('values', 'layouts', 'edges', 'norms', 'contracts', 'training', 'decoder') if args.case == 'all' else (args.case,):
            for precision in ('float16', 'float32') if args.precision == 'all' else (args.precision,):
                if subprocess.run([sys.executable, '-B', __file__, '--case', case, '--precision', precision, '--label', args.label]).returncode:
                    raise SystemExit(1)
        raise SystemExit(0)
    result = execute(args.case, args.precision); OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{args.label}-{args.case}-{args.precision}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    raise SystemExit(result['status'] != 'passed')

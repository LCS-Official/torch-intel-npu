"""Small real-device installation checks; use run_torch_npu.ps1 -Script to run."""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch_intel_npu as npu


def core_checks():
    torch.manual_seed(131)
    with torch.no_grad():
        host = torch.arange(17, dtype=torch.float32)
        torch.testing.assert_close((host.to('intel_npu') + 70000).cpu(), host + 70000, rtol=0, atol=0)
        matrix = torch.arange(16, dtype=torch.float16).reshape(4, 4) / 16
        actual = matrix.to('intel_npu') @ torch.eye(4, dtype=torch.float16).to('intel_npu')
        torch.testing.assert_close(actual.cpu(), matrix, rtol=0, atol=0)
    # One optimizer step checks forward, backward and parameter updates against CPU.
    for cpu, host in ((torch.nn.Linear(4, 3), torch.randn(2, 4) / 8),
                      (torch.nn.Conv2d(2, 2, 3, padding=1), torch.randn(1, 2, 5, 5) / 8)):
        device = copy.deepcopy(cpu).to('intel_npu')
        target = torch.zeros_like(cpu(host))
        cpu_optimizer = torch.optim.SGD(cpu.parameters(), lr=.01, foreach=False)
        optimizer = torch.optim.SGD(device.parameters(), lr=.01, foreach=False)
        expected = torch.nn.functional.mse_loss(cpu(host), target)
        actual = torch.nn.functional.mse_loss(device(host.to('intel_npu')), target.to('intel_npu'))
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual.detach().cpu(), expected, rtol=5e-5, atol=1e-6)
        for p, q in zip(device.parameters(), cpu.parameters()):
            torch.testing.assert_close(p.grad.cpu(), q.grad, rtol=5e-5, atol=1e-6)
        cpu_optimizer.step()
        optimizer.step()
        for p, q in zip(device.parameters(), cpu.parameters()):
            torch.testing.assert_close(p.detach().cpu(), q, rtol=5e-5, atol=1e-6)
    return ['strict FP32 tail arithmetic', 'FP16 matmul', 'Linear FP32 forward/backward/SGD',
            'Conv2d FP32 forward/backward/SGD']


def model_checks(model):
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from torch_npu_smollm import FrozenSmolStack
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    assert len(tokenizer('Hello')['input_ids']) > 0
    with safe_open(str(model / 'model.safetensors'), framework='pt', device='cpu') as file:
        assert len(file.keys()) > 0
    with torch.no_grad():
        stack = FrozenSmolStack.from_checkpoint(model, num_layers=1, mode='fp16')
        x = (torch.randn(1, 2, 576, dtype=torch.float16) / 8).to('intel_npu')
        y, state = stack.prefill(x)
        old_length = state[0].length
        token = (torch.randn(1, 1, 576, dtype=torch.float16) / 8).to('intel_npu')
        z, next_state = stack.decode(token, state)
        assert y.shape == (1, 2, 576) and z.shape == (1, 1, 576)
        assert state[0].length == old_length == 2 and next_state[0].length == 3
        assert torch.isfinite(y.cpu()).all() and torch.isfinite(z.cpu()).all()
    return ['local tokenizer/checkpoint', 'one-layer FP16 prefill/decode', 'old KV length preserved']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checks = core_checks()
    if args.model:
        checks.extend(model_checks(args.model.resolve()))
    gc.collect()
    npu.shutdown()
    stats = npu.stats()
    assert stats['closed'] and not stats['context_failed'] and stats['npu_calls'] > 0
    assert all(stats[key] == 0 for key in ('live_buffers', 'live_buffer_bytes', 'cached_buffer_bytes',
                                         'quarantined_buffer_bytes', 'cached_sessions'))
    assert not any(name == 'openvino' or name.startswith('openvino.') for name in sys.modules)
    print(json.dumps(dict(status='passed', checks=checks, python=sys.version.split()[0],
                          torch=torch.__version__, numpy=np.__version__, npunlock=str(Path(npu.npu.__file__).resolve()),
                          optional_models=bool(args.model), npu_calls=stats['npu_calls'],
                          live_buffer_bytes=stats['live_buffer_bytes'], context_failed=stats['context_failed'],
                          openvino_imported=False), indent=2))


if __name__ == '__main__':
    main()

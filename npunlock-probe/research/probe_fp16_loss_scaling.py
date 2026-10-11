"""Bounded static loss-scaling experiment; no CPU payload computation for NPU steps."""
import copy
import gc
import json
import torch
import torch_intel_npu as npu
from diagnose_fp16_training import OUT, CHECKPOINT
from torch_npu_training_demo import make_small_resnet


def train(saved, scale):
    cpu = make_small_resnet(torch.float16)
    cpu.load_state_dict(saved['state'])
    model = copy.deepcopy(cpu).to('intel_npu')
    data = saved['input'].to('intel_npu')
    target = torch.full((2, 4), .2, dtype=torch.float16)
    labels = target.to('intel_npu')
    optimizer = torch.optim.SGD(model.parameters(), lr=.05, foreach=False)
    rows = []
    for step in range(15):
        with torch.no_grad():
            for p, q in zip(model.parameters(), cpu.parameters()):
                q.copy_(p.detach().cpu())
        cpu.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        if step == 14:
            torch.save({'state': {k: v.detach().clone() for k, v in cpu.state_dict().items()},
                        'input': saved['input'], 'step': step, 'loss_scale': scale},
                       OUT / 'fp16-residual-late.pt')
        reference = ((cpu(saved['input']).float()-target.float()).square().mean().half())
        (reference*scale).backward()
        before = npu.stats()
        loss = torch.nn.functional.mse_loss(model(data), labels)
        (loss*scale).backward()
        with torch.no_grad():
            if scale != 1:
                for p in model.parameters():
                    p.grad = p.grad/scale
        optimizer.step()
        after = npu.stats()
        with torch.no_grad():
            a = torch.cat([p.grad.cpu().float().flatten() for p in model.parameters()])
            b = torch.cat([(p.grad/scale).float().flatten() for p in cpu.parameters()])
        row = {'step': step, 'loss': float(loss.detach().cpu()),
               'gradient_relative_l2': float((a-b).norm()/b.norm().clamp_min(1e-30)),
               'cosine': float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
               'reference_gradient_norm': float(b.norm()), 'npu_zero_gradient_elements': int((a == 0).sum()),
               'training_host_payload_copies': sum(after[k]-before[k] for k in
                  ('host_upload_copies', 'host_download_copies', 'host_device_copies'))}
        if step:
            assert row['training_host_payload_copies'] == 0
        rows.append(row)
        print(scale, row, flush=True)
    return {'scale': scale, 'steps': rows}


if __name__ == '__main__':
    torch.backends.mkldnn.enabled = False
    saved = torch.load(CHECKPOINT, weights_only=True)
    report = {'starting_checkpoint': str(CHECKPOINT), 'experiments': [train(saved, s) for s in (1, 1024)]}
    gc.collect()
    stats = npu.stats()
    assert not stats['context_failed']
    assert stats['live_buffers'] == stats['cached_scalar_buffers'], stats
    npu.shutdown()
    report['final_stats'] = npu.stats()
    (OUT / 'fp16-loss-scaling.json').write_text(json.dumps(report, indent=2))
    print('loss scaling experiment complete', flush=True)

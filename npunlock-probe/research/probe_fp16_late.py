"""Capture a near-converged scaled training state, then trace forward/backward differences."""
import torch
from diagnose_fp16_training import replay, OUT, CHECKPOINT
from probe_fp16_loss_scaling import train

if __name__ == '__main__':
    torch.backends.mkldnn.enabled = False
    train(torch.load(CHECKPOINT, weights_only=True), 1024)
    raise SystemExit(replay(checkpoint=OUT / 'fp16-residual-late.pt', prefix='fp16-late-diagnosis'))

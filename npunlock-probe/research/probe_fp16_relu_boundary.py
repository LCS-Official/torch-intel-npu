"""Causal diagnostic; CPU-derived ReLU masks are never used by the real backend."""
from diagnose_fp16_training import replay

if __name__ == '__main__':
    raise SystemExit(replay(cpu_relu_mask=True))

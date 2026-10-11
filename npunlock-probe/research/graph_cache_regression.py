"""Reuse frozen-stack input/training/ownership guards in a separate result directory."""
from model_cross_device_followup import OUT
import torch_npu_smollm_stack_demo as demo

if __name__=='__main__':
    demo.OUT=OUT/'regression';demo.OUT.mkdir(parents=True,exist_ok=True)
    raise SystemExit(demo.execute('guards'))

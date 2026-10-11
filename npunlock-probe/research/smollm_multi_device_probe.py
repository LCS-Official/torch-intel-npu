"""Reuse the existing full device quick checks, keeping previous reports."""
import torch_npu_device_demo as demo

demo.OUT=demo.ROOT/'npunlock-probe/research/results/smollm-kv-multi'
demo.DESTINATION=demo.OUT/'device.json'
demo.OUT.mkdir(parents=True,exist_ok=True)
raise SystemExit(demo.execute(True))

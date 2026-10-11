"""Reuse the existing quick device regression without overwriting old reports."""
from pathlib import Path
import torch_npu_device_demo as demo
demo.OUT=Path(__file__).resolve().parent/'results/smollm-kv-profile'
demo.OUT.mkdir(parents=True,exist_ok=True)
demo.DESTINATION=demo.OUT/'device.json'
raise SystemExit(demo.execute(True))

"""Reuse existing checks without overwriting their historical reports."""
import argparse
from smollm_generation_profile import OUT

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=('bindings','chain','defaults','guards','device','training'),required=True)
    case=parser.parse_args().case;directory=OUT/'regressions';directory.mkdir(exist_ok=True)
    if case=='bindings':
        import torch_npu_multi_output_demo as demo
        demo.OUT=directory;raise SystemExit(demo.execute())
    elif case=='device':
        import torch_npu_device_demo as demo
        demo.OUT=directory;demo.DESTINATION=directory/'device.json';raise SystemExit(demo.execute(True))
    elif case=='training':
        import torch_npu_training_demo as demo
        demo.OUT=directory;demo.REPORT=directory/'training.json';raise SystemExit(demo.execute(True,False))
    else:
        import decode_chain_api as common
        common.OUT=directory
        if case=='chain':common.checks()
        elif case=='defaults':
            import decode_chain_full as full
            full.OUT=directory;full.defaults()
        else:
            import decode_chain_regressions as regression
            regression.OUT=directory;regression.guards()

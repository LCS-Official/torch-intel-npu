"""Compare the frozen baseline and current preparation in one NPU context."""
import argparse
import ast
import json
from pathlib import Path
import torch_intel_npu as npu
from smollm_generation_profile import main, OUT
from w8_quality_probe import sha

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('fp16','w8a16'),required=True)
    mode=parser.parse_args().mode
    source=OUT/'before/torch_intel_npu.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    definition=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_prepare_many')
    scope={};exec(compile(ast.Module(body=[definition],type_ignores=[]),str(source),'exec'),npu.__dict__,scope)
    main(mode,baseline_prepare=scope['_prepare_many'],output=OUT/'preparation')
    path=OUT/'preparation'/(mode+'.json');report=json.loads(path.read_text())
    report.update(baseline_source_sha256=sha(source),wrapper_sha256=sha(Path(__file__)))
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')

"""Profile the same bounded Select replacement with the existing DLL."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import kv_gather_profile as control
from kv_select_kernel import OUT, candidate, patch, sha
from npunlock.npu4000 import compile_fused_rmsnorm


def profiled_candidate(directory,kind):
    module=candidate(directory,kind)
    def compile_profile(*args,**options):
        base=compile_fused_rmsnorm(*args,**options)
        blob,report=patch(base.graph_blob,module._SELECT_KERNEL)
        return replace(base,graph_blob=blob,patch_reports=base.patch_reports+(json.dumps(report).encode(),))
    module.compile_fused_rmsnorm=compile_profile
    return module


def main():
    control.OUT=OUT;control.candidate=profiled_candidate
    control.main()
    variant=sys.argv[sys.argv.index('--variant')+1]
    path=OUT/f'device-{variant}/b4-s16-w8a16.json';report=json.loads(path.read_text(encoding='utf-8'))
    report.update(implementation='select_shave_copy',wrapper_sha256=sha(Path(__file__)))
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()

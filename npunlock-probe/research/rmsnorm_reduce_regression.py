"""Run existing model/precision checks with the promoted kernel, serially."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-reduce/regressions'


def main():
    for module,cases in (('torch_npu_smollm_demo',('norm','numeric','boundary','guards','causality','hybrid')),
                         ('torch_npu_smollm_stack_demo',('numeric','boundary'))):
        directory=OUT/module;directory.mkdir(parents=True,exist_ok=True)
        command=f'import sys; from pathlib import Path; import {module} as d; d.OUT=Path(sys.argv[1]); raise SystemExit(d.execute(sys.argv[2]))'
        for case in cases:
            with (directory/f'{case}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-c',command,str(directory),case],cwd=ROOT,stdout=log,stderr=log,timeout=360)
            if result.returncode:
                print((directory/f'{case}.log').read_text(encoding='utf-8'),flush=True);return result.returncode
            report=json.loads((directory/f'{case}.json').read_text(encoding='utf-8'))
            assert report['status']=='passed'
            print(module,case,'passed',len(report['checks']),'checks',report['final_stats']['npu_calls'],'NPU calls',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

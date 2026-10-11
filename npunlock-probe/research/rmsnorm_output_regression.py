"""Run existing SmolLM checks serially, preserving their earlier result files."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-output/regression'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    command='import sys; from pathlib import Path; import torch_npu_smollm_demo as d; d.OUT=Path(sys.argv[1]); raise SystemExit(d.execute(sys.argv[2]))'
    for case in ('norm','numeric','boundary','guards','causality','hybrid','performance'):
        with (OUT/(case+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-c',command,str(OUT),case],cwd=ROOT,
                stdout=log,stderr=log,timeout=300)
        if result.returncode:
            print((OUT/(case+'.log')).read_text(encoding='utf-8'),flush=True)
            return result.returncode
        report=json.loads((OUT/(case+'.json')).read_text())
        assert report['status']=='passed'
        print(dict(case=case,status=report['status'],checks=len(report['checks']),
            npu_calls=report['final_stats']['npu_calls']),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

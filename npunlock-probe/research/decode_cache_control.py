"""Finish ABBA process controls serially after the saved baseline runs."""
import json
from pathlib import Path
import subprocess
import sys

from decode_cache_probe import ROOT,OUT


def main():
    for index,runtime in ((1,'after'),(2,'after'),(3,'before')):
        for case in ('model','generate'):
            label=f'{case}-{index}-{runtime}'
            script=Path(__file__).with_name('decode_cache_probe.py' if case=='model' else 'decode_cache_generate.py')
            command=[sys.executable,str(script),'--runtime',runtime,'--label',label,'--child']
            if case=='model':command+=['--case','model']
            with (OUT/f'{label}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=log,timeout=600)
            if result.returncode:
                print((OUT/f'{label}.log').read_text(encoding='utf-8'),flush=True);return result.returncode
            report=json.loads((OUT/f'{label}.json').read_text(encoding='utf-8'))
            assert report['status']=='passed'
            print(label,'passed',report['final_stats']['npu_calls'],'NPU calls',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

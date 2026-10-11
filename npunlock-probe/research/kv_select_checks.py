"""Reuse current model/quality checks without rewriting old calibration data."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from w8_quality_probe import ROOT, sha

OUT=ROOT/'npunlock-probe/research/results/kv-select-integration'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('prepare','holdout','decode','regression'),required=True)
    args=parser.parse_args()
    if args.case=='regression':
        import rmsnorm_reduce_regression as check
        check.OUT=OUT/'regressions';return check.main()
    import w8_calibration as calibration
    calibration.OUT=OUT/'calibrated-current'
    calibration.SOURCES=calibration.SOURCES+['npunlock/python/npunlock/npu4000.py',
        'npunlock/tests/fixtures/npu4000/kv_select.c','npunlock/tests/fixtures/npu4000/kv_select/kv-select-f16.elf',
        'npunlock/tests/fixtures/npu4000/kv_select/manifest.json']
    if args.case=='prepare':
        sys.argv=[calibration.__file__,'--method','gptq'];code=calibration.main()
    else:
        check=importlib.import_module('w8_calibration_probe' if args.case=='holdout' else 'w8_calibration_decode')
        code=check.main()
    path=calibration.OUT/f'{args.case}.json';report=json.loads(path.read_text(encoding='utf-8'))
    report['entry_runner_sha256']=sha(Path(__file__));path.write_text(json.dumps(report,indent=2),encoding='utf-8');return code


if __name__=='__main__':raise SystemExit(main())

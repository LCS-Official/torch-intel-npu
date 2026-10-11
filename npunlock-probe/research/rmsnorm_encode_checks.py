"""Current-kernel checks without overwriting historical calibration records."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
from rmsnorm_encode import OUT, ROOT
from w8_quality_probe import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('confirm','prepare','holdout','decode','regression'),required=True)
    args=parser.parse_args()
    if args.case=='confirm':
        import rmsnorm_encode as candidate
        directory=OUT/'confirmation/mask';directory.mkdir(parents=True,exist_ok=True)
        for name in ('build.json','baseline.elf','codec.elf','roundtrip.elf'):
            shutil.copy2(OUT/'mask'/name,directory/name)
        candidate.OUT=OUT/'confirmation'
        sys.argv=[candidate.__file__,'--case','model','--candidate','mask','--label','confirm']
        return candidate.main()
    if args.case=='regression':
        import rmsnorm_reduce_regression as regression
        regression.OUT=OUT/'regressions';return regression.main()
    import w8_calibration as calibration
    calibration.OUT=OUT/'calibrated-current'
    if args.case=='prepare':
        sys.argv=[calibration.__file__,'--method','gptq'];code=calibration.main();name='prepare'
    else:
        module=importlib.import_module('w8_calibration_probe' if args.case=='holdout' else 'w8_calibration_decode')
        code=module.main();name=args.case
    path=calibration.OUT/f'{name}.json';report=json.loads(path.read_text(encoding='utf-8'))
    report['entry_runner_sha256']=sha(Path(__file__));path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return code


if __name__=='__main__':raise SystemExit(main())

"""Confirm the actual default graph and run the existing model regressions."""
import argparse
from pathlib import Path

from decoder_refresh import OUT, read, write
from w8_quality_probe import ROOT, sha


def production(mode):
    template=ROOT/'npunlock-probe/research/generation_cache_validate.py'
    text=template.read_text()
    changes={
        "reference=OUT/('trace-fp16-outputs.npz' if mode=='fp16' else 'benchmark-w8a16-outputs.npz')":
            "reference=OUT/'position-flags/generation'/(mode+'-outputs.npz')",
        "baseline['logits'][step]":"baseline[f'logits-{step}']"}
    for old,new in changes.items():assert text.count(old)==1,old;text=text.replace(old,new)
    path=OUT/'production-validator.py';path.write_text(text,encoding='utf-8')
    scope=dict(__name__='decoder_refresh_production_runner',__file__=str(path))
    exec(compile(text,str(path),'exec'),scope);scope['OUT']=OUT;scope['production'](mode)
    path=OUT/'production'/(mode+'.json');report=read(path)
    report.update(factory_sha256=sha(Path(__file__)),validator_template_sha256=sha(template))
    write(path,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('production','regressions'),required=True)
    parser.add_argument('--mode',choices=('fp16','w8a16'),default='fp16');args=parser.parse_args()
    if args.case=='production':production(args.mode)
    else:
        import rmsnorm_reduce_regression as regression
        regression.OUT=OUT/'regressions';raise SystemExit(regression.main())

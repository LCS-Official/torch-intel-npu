"""S59: one bounded native-main fit; stop the single-layer branch if not promoted."""
import argparse
from pathlib import Path

import torch
import w8_decode_proxy as previous

base=previous.base
OUT=previous.ROOT/'npunlock-probe/research/results/w8-native-down-check'
for module in (previous,previous.previous,previous.parent,previous.basis,previous.attention,previous.runner,
               previous.runner.previous,previous.probe,previous.attention.device,
               previous.attention.device.previous,base,base.mixed):module.OUT=OUT
PROFILES=('exact-a50','native-a50')
original_identity=previous.source_identity
original_proxy=previous.proxy_main
original_run=previous.probe.run_graph
observed=[]
FRESH=(
    ('bottle','Explain why a reusable water bottle has a cap.'),
    ('feather','Why does a feather fall more slowly than a small stone?'),
    ('oranges','A bowl has seven oranges and five lemons. How many pieces of fruit are there?'),
    ('visit','Write a short polite message asking when a friend is free to visit.'),
    ('hinge','Describe how a hinge lets a door open.'),
    ('harbor','Use harbor, clouds, and morning in one descriptive sentence.'),
    ('market','How is a farmers market different from a supermarket?'),
    ('buttons','Suggest three ways to organize spare clothing buttons.'))


def aligned_native_main(rows,x):
    if not rows or any(len(row)!=2 for row in rows):raise ValueError('Missing native observations')
    gated=torch.cat([row[0] for row in rows]);main=torch.cat([row[1] for row in rows])
    if (gated.shape!=x.shape or not torch.equal(gated,x) or main.ndim!=2 or len(main)!=len(x)
        or any(t.device.type!='cpu' or t.dtype!=torch.float16 or not torch.isfinite(t).all() for t in (gated,main,x))):
        raise ValueError('Native observations do not match the fitting inputs')
    return main


def run_capture(npu,recipe,inputs):
    values,key=original_run(npu,recipe,inputs)
    if 'down_main' in values and 'gated' in values:
        observed.append((values['gated'].cpu(),values['down_main'].cpu()))
    return values,key


def proxy_main(weight,x,mode):
    native=aligned_native_main(observed,x)
    return original_proxy(weight,x,'exact') if mode=='exact' else native


def source_identity():
    return dict(**original_identity(),native_study_sha256=base.sha(Path(__file__)),
        native_policy='Last bounded alpha=.5 native-main control; no additional single-layer sweep.')


previous.PROFILES=PROFILES
previous.parent.PROFILES=PROFILES
previous.FRESH=FRESH
previous.source_identity=source_identity
previous.proxy_main=proxy_main

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','fit','internal','select'))
    parser.add_argument('--profile',choices=('baseline',*PROFILES),default='native-a50')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():
        if args.stage=='prepare':previous.basis.prepare();result=previous.data()
        elif args.stage=='fit':
            previous.probe.run_graph=run_capture
            try:result=previous.fit()
            finally:previous.probe.run_graph=original_run
        elif args.stage=='internal':result=previous.parent.internal(args.profile)
        else:result=previous.select()
    raise SystemExit(result or 0)

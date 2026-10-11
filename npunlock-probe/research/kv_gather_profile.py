"""Reuse the existing isolated profiler for a B4 calibrated-W8 control."""
import argparse
import json
from pathlib import Path
import shutil
import torch_npu_device_profile as profile
import torch_npu_smollm as baseline
from kv_gather_probe import OUT, ROOT, SOURCES, candidate, calibration, sha
from w8_performance import profile_binary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=('baseline','gather'),required=True)
    args=parser.parse_args();directory=OUT/f'device-{args.variant}';directory.mkdir(parents=True,exist_ok=True)
    previous,build=profile_binary();shutil.copy2(previous/'session_task_profile.dll',directory/'session_task_profile.dll')
    profile.OUT=directory
    calibration.OUT=ROOT/'npunlock-probe/research/results/rmsnorm-encode/calibrated-current'
    packed=calibration.load_weights()
    cls=baseline.FrozenSmolStack if args.variant=='baseline' else candidate(directory,'gather').FrozenSmolStack
    original=cls.from_checkpoint.__func__
    def construct(cls,directory,**options):return original(cls,directory,**options,packed_weights=packed)
    cls.from_checkpoint=classmethod(construct);baseline.FrozenSmolStack=cls
    profile.execute(4,16,'w8a16')
    path=directory/'b4-s16-w8a16.json';report=json.loads(path.read_text(encoding='utf-8'))
    report.update(variant=args.variant,entry_sha256=sha(Path(__file__)),profile_build=build,
        sources={name:sha(ROOT/name) for name in SOURCES},calibration_weights_sha256=sha(calibration.OUT/'weights.npz'))
    if args.variant=='gather':report['candidate_sha256']=sha(directory/'candidate.py')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()

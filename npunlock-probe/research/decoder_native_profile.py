"""Compact full-stack host and task profiles, reusing the isolated profilers."""
import argparse
import json
import shutil
from pathlib import Path
from w8_quality_probe import ROOT, sha

OUT=ROOT/'npunlock-probe/research/results/decoder-native'


def main(case,mode):
    import torch_npu_device_profile as profile
    profile.OUT=OUT/'profile';profile.OUT.mkdir(parents=True,exist_ok=True)
    if case=='build':profile.build();return
    build=json.loads((profile.OUT/'build.json').read_text())
    assert sha(profile.OUT/'session_task_profile.dll')==build['dll_sha256']
    assert sha(profile.OUT/'session_task_profile.c')==build['generated_source_sha256']
    assert sha(ROOT/'npunlock/src/graphinfer/session_win32.c')==build['production_session_sha256']
    assert sha(Path(profile.__file__))==build['builder_sha256']
    if case=='tasks':profile.execute(1,24,mode,num_layers=30,save_blobs=False)
    else:
        import shave_fp32_pipeline as pipeline
        source_dll=profile.OUT/'session_task_profile.dll'
        def bind_profile(directory):
            directory.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source_dll,directory/'session_task_profile.dll')
            assert sha(directory/'session_task_profile.dll')==build['dll_sha256']
            return build
        pipeline.OUT=OUT;pipeline.SHAPES=((1,24),);pipeline.profiling_directory=bind_profile
        pipeline.benchmark('host',True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('build','host','tasks'),required=True)
    parser.add_argument('--mode',choices=('fp16','w8a16'),default='fp16');args=parser.parse_args()
    main(args.case,args.mode)

"""Refresh the current decoder profiles without overwriting historical evidence."""
import argparse
import json
from pathlib import Path
import shutil

from decode_chain_api import provenance
from w8_quality_probe import ROOT, sha

OUT=ROOT/'npunlock-probe/research/results/decoder-refresh'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def prepare():
    source=ROOT/'npunlock-probe/research/results/decoder-native/profile'
    build=read(source/'build.json')
    for name,key in (('session_task_profile.dll','dll_sha256'),('session_task_profile.c','generated_source_sha256')):
        assert sha(source/name)==build[key]
    assert sha(ROOT/'torch_npu_device_profile.py')==build['builder_sha256']
    assert sha(ROOT/'npunlock/src/graphinfer/session_win32.c')==build['production_session_sha256']
    assert not (OUT/'before.json').exists(),'Preserve frozen evidence; use a new output directory'
    (OUT/'profile').mkdir(parents=True,exist_ok=True)
    for name in ('session_task_profile.dll','session_task_profile.c','build.json'):
        shutil.copyfile(source/name,OUT/'profile'/name)
    sources=provenance()
    for name in ('torch_npu_device_profile.py','torch_npu_warm_profile.py',
                 'npunlock-probe/research/decoder_native_profile.py','npunlock-probe/research/decoder_native_summary.py',
                 'npunlock/tests/fixtures/npu4000/kv_select.c','npunlock/tests/fixtures/npu4000/kv_select/manifest.json',
                 'npunlock/tests/fixtures/npu4000/rmsnorm/manifest.json'):
        sources[name]=sha(ROOT/name)
    for name in sources:
        path=OUT/'before'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    shutil.copyfile(Path(__file__),OUT/'profile-runner.py')
    write(OUT/'before.json',dict(sources=sources,runner_sha256=sha(Path(__file__)),profiling_build=build))
    print('Current production sources frozen; validated isolated profiler reused',flush=True)


def main(case,mode):
    if case=='prepare':prepare();return
    before=read(OUT/'before.json')
    for name,digest in before['sources'].items():assert sha(ROOT/name)==digest,name
    if case=='summary':
        import decoder_native_summary as summary
        summary.OUT=OUT;summary.summarize()
    else:
        import decoder_native_profile as profile
        profile.OUT=OUT;profile.main(case,mode)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('prepare','host','tasks','summary'),required=True)
    parser.add_argument('--mode',choices=('fp16','w8a16'),default='fp16');args=parser.parse_args()
    main(args.case,args.mode)

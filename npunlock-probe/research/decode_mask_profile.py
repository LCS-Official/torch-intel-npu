"""Reuse the validated task profiler for one isolated graph-table variant."""
import argparse
import json
from pathlib import Path
import shutil

from decode_mask_table import OUT, variants
from w8_quality_probe import ROOT, sha


def main(name):
    import torch_npu_device_profile as profile
    import torch_npu_smollm as smol
    source=ROOT/'npunlock-probe/research/results/decoder-native/profile'
    build=json.loads((source/'build.json').read_text())
    assert sha(source/'session_task_profile.dll')==build['dll_sha256']
    assert sha(source/'session_task_profile.c')==build['generated_source_sha256']
    assert sha(ROOT/'npunlock/src/graphinfer/session_win32.c')==build['production_session_sha256']
    assert sha(Path(profile.__file__))==build['builder_sha256']
    profile.OUT=OUT/'profile'/name;profile.OUT.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source/'session_task_profile.dll',profile.OUT/'session_task_profile.dll')
    (profile.OUT/'build.json').write_text(json.dumps(build,indent=2),encoding='utf-8')
    shutil.copyfile(Path(__file__),profile.OUT/'runner.py')
    if name=='joined':
        graph_source=OUT/'joint/joined-graph.py';namespace=dict(vars(smol))
        exec(compile(graph_source.read_text(),str(graph_source),'exec'),namespace);graph=namespace['graph']
    else:graph=variants()[name];graph_source=OUT/(name+'-graph.py')
    original=smol.FrozenSmolLayer.graph;smol.FrozenSmolLayer.graph=graph
    try:profile.execute(1,24,'fp16',num_layers=4,save_blobs=False)
    finally:smol.FrozenSmolLayer.graph=original
    path=profile.OUT/'b1-s24-fp16.json';report=json.loads(path.read_text())
    report.update(variant=name,wrapper_sha256=sha(Path(__file__)),graph_source_sha256=sha(graph_source))
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(name,'profiling passed; independent plain/profile controls and fresh task records',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',choices=('baseline','mask','both','joined'),required=True)
    main(parser.parse_args().name)

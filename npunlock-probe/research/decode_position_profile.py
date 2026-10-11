"""Profile the current graph and KV flag lookup with independent controls."""
import argparse
from pathlib import Path
import shutil

from decode_position_flags import OUT, variants
from decoder_refresh import OUT as REFRESH, read, write
from w8_quality_probe import sha


def main(name):
    import torch_npu_device_profile as profile
    import torch_npu_smollm as smol
    functions=variants();build=read(REFRESH/'profile/build.json')
    profile.OUT=OUT/'profile'/name;profile.OUT.mkdir(parents=True,exist_ok=True)
    source=REFRESH/'profile/session_task_profile.dll';assert sha(source)==build['dll_sha256']
    shutil.copyfile(source,profile.OUT/source.name);write(profile.OUT/'build.json',build)
    original=smol.FrozenSmolLayer.graph;smol.FrozenSmolLayer.graph=functions[name]
    try:profile.execute(1,24,'fp16',num_layers=4,save_blobs=False)
    finally:smol.FrozenSmolLayer.graph=original
    path=profile.OUT/'b1-s24-fp16.json';report=read(path)
    report.update(variant=name,wrapper_sha256=sha(Path(__file__)),graph_source_sha256=sha(OUT/(name+'-graph.py')))
    write(path,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',choices=('baseline','flags'),required=True)
    main(parser.parse_args().name)

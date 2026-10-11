"""Isolated shared position lookup including exact KV selection flags."""
import argparse
import inspect
from pathlib import Path
import textwrap

import decode_mask_table as probe
from decoder_refresh import OUT as REFRESH, read, write
from w8_quality_probe import sha

OUT=REFRESH/'position-flags'


def variants():
    import torch_npu_smollm as smol
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'baseline-graph.py'
    source=path.read_text() if path.exists() else textwrap.dedent(inspect.getsource(smol.FrozenSmolLayer.graph))
    changes={
        'mask_table),axis=1))':'mask_table,np.eye(64,dtype=np.float16)),axis=1))',
        'shape=(1,192),batch_dims=0)':'shape=(1,256),batch_dims=0)',
        probe.OLD_SELECTED:"        selected=reshape(op('Convert',slice_(positional,1,192,256),dtype='boolean',destination_type='boolean'),(1,1,64,1))"}
    candidate=source
    for old,new in changes.items():assert candidate.count(old)==1,old;candidate=candidate.replace(old,new)
    functions={}
    for name,text in (('baseline',source),('flags',candidate)):
        path=OUT/(name+'-graph.py');path.write_text(text,encoding='utf-8')
        scope=dict(vars(smol));exec(compile(text,str(path),'exec'),scope);functions[name]=scope['graph']
    return functions


def main(batch,prefix,layers,mode,all_positions):
    probe.OUT=OUT;probe.variants=variants
    probe.model(batch,prefix,layers,mode,['baseline','flags'],all_positions)
    path=OUT/f'l{layers}-b{batch}-s{prefix}-{mode}.json';report=read(path)
    report.update(candidate_wrapper_sha256=sha(Path(__file__)),
        graph_source_sha256={name:sha(OUT/(name+'-graph.py')) for name in ('baseline','flags')})
    write(path,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=int,choices=(1,4,8),default=1)
    parser.add_argument('--prefix',type=int,choices=(24,63),default=24)
    parser.add_argument('--layers',type=int,choices=(4,30),default=4)
    parser.add_argument('--mode',choices=('fp16','w8a16'),default='fp16');parser.add_argument('--all-positions',action='store_true')
    args=parser.parse_args();main(args.batch,args.prefix,args.layers,args.mode,args.all_positions)

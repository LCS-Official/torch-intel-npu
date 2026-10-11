"""Share one positional Gather for RoPE cosine, sine and causal mask."""
import argparse
import json
from pathlib import Path

import decode_mask_table as probe
from w8_quality_probe import sha

OLD_ROPE="""        def rope(data):
            table=s.constant(np.frombuffer(data,dtype=np.float16).reshape(64,64))
            return reshape(op('Gather',table,indices,s.constant(np.int64(0)),shape=(1,64),batch_dims=0),(1,1,1,64))
        cos=rope(self._cos);sin=rope(self._sin)"""
NEW_ROPE="""        mask_table=np.where(np.arange(64)[None,:]>np.arange(64)[:,None],-np.inf,0).astype(np.float16)
        table=s.constant(np.concatenate((np.frombuffer(self._cos,dtype=np.float16).reshape(64,64),
            np.frombuffer(self._sin,dtype=np.float16).reshape(64,64),mask_table),axis=1))
        positional=op('Gather',table,indices,s.constant(np.int64(0)),shape=(1,192),batch_dims=0)
        cos=reshape(slice_(positional,1,0,64),(1,1,1,64))
        sin=reshape(slice_(positional,1,64,128),(1,1,1,64))
        table_mask=reshape(slice_(positional,1,128,192),(1,1,1,64))"""


def main(batch,layers,all_positions,prefix=24):
    import torch_npu_smollm as smol
    probe.OUT=probe.OUT/'joint';probe.OUT.mkdir(parents=True,exist_ok=True)
    original=probe.variants
    def variants():
        functions=original();source=(probe.OUT/'baseline-graph.py').read_text()
        assert source.count(OLD_ROPE)==source.count(probe.OLD_MASK)==1
        source=source.replace(OLD_ROPE,NEW_ROPE).replace(probe.OLD_MASK,'        mask=table_mask')
        path=probe.OUT/'joined-graph.py';path.write_text(source,encoding='utf-8')
        namespace=dict(vars(smol));exec(compile(source,str(path),'exec'),namespace)
        return dict(baseline=functions['baseline'],joined=namespace['graph'])
    probe.variants=variants
    probe.model(batch,prefix,layers,'fp16',['baseline','joined'],all_positions)
    path=probe.OUT/f'l{layers}-b{batch}-s{prefix}-fp16.json';report=json.loads(path.read_text())
    report['joint_wrapper_sha256']=sha(Path(__file__))
    report['joint_graph_sha256']=sha(probe.OUT/'joined-graph.py')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=int,choices=(1,4,8),default=1)
    parser.add_argument('--layers',type=int,choices=(4,30),default=4);parser.add_argument('--all-positions',action='store_true')
    parser.add_argument('--prefix',type=int,choices=(24,63),default=24)
    args=parser.parse_args();main(args.batch,args.layers,args.all_positions,args.prefix)

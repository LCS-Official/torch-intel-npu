"""Reuse the existing boundary/KV/gamma checks with frozen calibrated policies."""
import argparse
import json

import torch
import torch_npu_smollm as smol
import w8_prefill_cache as study


def main(label,phase):
    import prefill_cache_hardware as hardware
    ppl=study.helpers()
    packed=None if label=='fp16' else ppl.layers.mixed.weights()[0]
    original=smol.FrozenSmolLayer.from_checkpoint
    def prepared(directory,**options):
        options.update(mode='fp16' if label=='fp16' else 'w8a16',packed_weights=packed,
            fp16_projections=('mlp.up_proj.weight','mlp.down_proj.weight') if label=='mixed' else ())
        return original(directory,**options)
    smol.FrozenSmolLayer.from_checkpoint=staticmethod(prepared)
    hardware.OUT=study.OUT/('hardware-'+label)
    hardware.OUT.mkdir(parents=True,exist_ok=True)
    hardware.main(phase)
    path=hardware.OUT/('shape-'+phase+'.json')
    report=json.loads(path.read_text(encoding='utf-8'));report['label']=label
    report['runner_sha256']=study.common.sha(__file__)
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--label',choices=('fp16','w8','mixed'),required=True)
    parser.add_argument('--phase',choices=('fill','load'),required=True);args=parser.parse_args()
    main(args.label,args.phase)

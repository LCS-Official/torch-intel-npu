"""Offline graph checks for research composition through existing layer APIs."""
import xml.etree.ElementTree as ET

import torch

import w8_layer_sensitivity as experiment
from torch_npu_smollm import FrozenSmolStack


def main():
    torch.set_num_threads(8)
    packed=experiment.mixed.weights()
    composed=experiment.make_stack('mixed',packed)
    public=FrozenSmolStack.from_checkpoint(experiment.mixed.common.MODEL,num_layers=30,packed_weights=packed,
        fp16_projections=('mlp.up_proj.weight','mlp.down_proj.weight'))
    options=dict(segment='layer',fused_norm=True,kv_cache='decode')
    for a,b in zip(composed.layers,public.layers):
        assert a._weights==b._weights and a.fp16_projections==b.fp16_projections
        assert a.graph((1,1,576),**options)[1:]==b.graph((1,1,576),**options)[1:]
    selected=experiment.make_stack('selected',packed)
    wanted=frozenset(tuple(key) for key in experiment.read('selection')['selected'])
    count=0
    for i,layer in enumerate(selected.layers):
        policy=frozenset(name for index,name in wanted if index==i)
        assert layer.fp16_projections==policy
        _,xml,_=layer.graph((1,1,576),**options)
        tree=ET.fromstring(xml)
        protected=[node for node in tree.findall('./layers/layer') if node.attrib['name'].endswith('_protected_fp16')]
        compressed=[node for node in tree.findall('./layers/layer') if node.find("./rt_info/attribute[@name='decompression']") is not None]
        assert len(protected)==len(policy) and len(compressed)==7-len(policy)
        assert {node.attrib['name'] for node in protected}=={name.replace('.','_')+'_protected_fp16' for name in policy}
        assert all(node.find('data').attrib['element_type']=='f16' for node in protected)
        count+=len(protected)
    assert count==len(wanted)
    result=dict(status='passed',**experiment.provenance(),identical_global_graphs=30,
                selected_graphs=30,protected_projections=count,
                selection_sha256=experiment.mixed.common.sha(experiment.OUT/'selection.json'))
    experiment.write('composition-checks',result)
    print('PASS: 30 default mixed graphs byte-identical; 30 selected graphs protect exactly',count,'projections')


if __name__=='__main__':
    with torch.no_grad():
        main()

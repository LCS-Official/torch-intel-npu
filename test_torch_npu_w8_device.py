"""CPU-only checks of diagnostic graph exports and frozen device selection."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import npunlock as symbolic


def helpers(previous=None):
    path=Path(__file__).parent/'npunlock-probe/research/w8_device_quality.py'
    names={'projection_probe_graph','device_rank','evaluate'}
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    ranking=lambda r:(r['positions']-r['quality_passed'],r['max_relative_l2'],r['mean_relative_l2'])
    env=dict(symbolic=symbolic,np=np,ET=ET,copy=copy,
        base=SimpleNamespace(mixed=SimpleNamespace(common=SimpleNamespace(ranking=ranking))),
        previous=previous,frozen_selection=lambda:'device-policy')
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return SimpleNamespace(**{name:env[name] for name in names})


def small_graph(*,up_name='mlp_down_proj_weight_correction_up'):
    residual=symbolic.input('residual',shape=(2,4),dtype='f16')
    norm=symbolic.input('normalized',shape=(2,5),dtype='f16')
    gated=symbolic.op('Swish',norm,_shape=(2,5),_dtype='f16')
    compressed=symbolic.constant(np.ones((4,5),dtype=np.int8),name='packed')
    weight=symbolic.op('Convert',compressed,destination_type='f16',_shape=(4,5),_dtype='f16',_name='decode')
    def mm(x,w,n):return symbolic.op('MatMul',x,w,transpose_a=False,transpose_b=True,_shape=(2,n),_dtype='f16')
    main=mm(gated,weight,4)
    down=mm(gated,symbolic.constant(np.ones((2,5),dtype=np.float16),name='mlp_down_proj_weight_correction_down'),2)
    up=mm(down,symbolic.constant(np.ones((4,2),dtype=np.float16),name=up_name),4)
    add=lambda a,b:symbolic.op('Add',a,b,auto_broadcast='numpy',_shape=(2,4),_dtype='f16')
    corrected=add(main,up);output=add(residual,corrected)
    graph=symbolic.Graph([residual,norm],[output]);xml=ET.fromstring(symbolic.serialize_ir(graph).xml)
    layer=xml.find("./layers/layer[@name='decode']");info=ET.SubElement(layer,'rt_info')
    ET.SubElement(info,'attribute',name='decompression',version='0')
    return graph,ET.tostring(xml),(gated,main,down,up,corrected,output)


class DeviceProbeTests(unittest.TestCase):
    def test_outputs_are_the_existing_chain_tensors_in_order(self):
        graph,xml,values=small_graph();probe,_,_,names=helpers().projection_probe_graph(graph,xml)
        self.assertEqual(names,('gated','main','reduced','delta','corrected','output'))
        self.assertEqual(probe.inputs,graph.inputs)
        self.assertTrue(all(out.producer.inputs[0] is value for out,value in zip(probe.outputs,values,strict=True)))
        self.assertEqual([v.shape for v in probe.outputs],[(2,5),(2,4),(2,2),(2,4),(2,4),(2,4)])

    def test_decompression_and_swish_contracts_survive_without_source_mutation(self):
        graph,xml,_=small_graph();before=ET.tostring(ET.fromstring(xml))
        _,modified,_,_=helpers().projection_probe_graph(graph,xml);tree=ET.fromstring(modified)
        self.assertIsNotNone(tree.find("./layers/layer[@name='decode']/rt_info/attribute[@name='decompression']"))
        self.assertTrue(all(n.attrib['version']=='opset4' for n in tree.findall("./layers/layer[@type='Swish']")))
        self.assertEqual(ET.tostring(ET.fromstring(xml)),before)

    def test_missing_named_correction_rejects(self):
        graph,xml,_=small_graph(up_name='different_weight')
        with self.assertRaisesRegex(ValueError,'correction-up'):helpers().projection_probe_graph(graph,xml)

    def test_ambiguous_correction_add_rejects(self):
        graph,xml,values=small_graph()
        extra=symbolic.op('Add',values[1],values[3],auto_broadcast='numpy',_shape=(2,4),_dtype='f16')
        ambiguous=symbolic.Graph(graph.inputs,[graph.outputs[0],extra])
        with self.assertRaisesRegex(ValueError,'correction add'):helpers().projection_probe_graph(ambiguous,xml)

    def test_device_ranking_uses_original_gates_and_baseline_tiebreak(self):
        records={p:dict(summary=dict(positions=10,quality_passed=8,max_relative_l2=.1,mean_relative_l2=.03))
            for p in ('baseline','candidate')}
        rank=helpers().device_rank
        self.assertEqual(min(records,key=lambda p:rank(p,records)),'baseline')
        records['candidate']['summary']['quality_passed']=9
        records['candidate']['summary']['max_relative_l2']=.2
        self.assertEqual(min(records,key=lambda p:rank(p,records)),'candidate')

    def test_reused_evaluator_gets_device_choice_and_restores_original_loader(self):
        original=lambda:'old-policy';previous=SimpleNamespace(frozen_selection=original)
        previous.hardware=lambda *args:(previous.frozen_selection(),args)
        result=helpers(previous).evaluate('hardware','candidate','kv')
        self.assertEqual(result,('device-policy',('candidate','kv')))
        self.assertIs(previous.frozen_selection,original)

    def test_loader_is_restored_even_if_evaluation_fails(self):
        original=lambda:'old-policy';previous=SimpleNamespace(frozen_selection=original)
        def fail(*args):raise RuntimeError('diagnostic failure')
        previous.hardware=fail
        with self.assertRaisesRegex(RuntimeError,'diagnostic failure'):helpers(previous).evaluate('hardware','candidate','kv')
        self.assertIs(previous.frozen_selection,original)


if __name__=='__main__':unittest.main()

"""Offline byte-packet ABI and graph validation, without NPU execution."""
import struct
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

import npunlock as npu
from npunlock.npu4000 import mask_packet_spec,compile_mask_packet


class MaskPacketTests(unittest.TestCase):
    def test_odd_mask_and_unaligned_payload(self):
        self.assertEqual(mask_packet_spec(5,6,2),((5,40,40),40,26))
        with patch('npunlock.npu4000.compile_precision_graph',side_effect=lambda graph,**kwargs:graph) as compile_graph:
            graph=compile_mask_packet(5,6,2)
        self.assertTrue(compile_graph.call_args.kwargs['dma_only'])
        self.assertEqual([v.shape for v in graph.inputs],[(1,5),(1,40),(1,40)])
        self.assertEqual(graph.outputs[0].shape,(1,104))
        ir=npu.serialize_ir(graph)
        self.assertEqual(struct.unpack('<4f',ir.weights[:16]),(5.,6.,2.,0.))
        self.assertEqual(set(layer.get('type') for layer in ET.fromstring(ir.xml).findall('./layers/layer')),
                         {'Parameter','Const','Concat','Result'})

    def test_output_padding_for_bool_cast(self):
        self.assertEqual(mask_packet_spec(5,8,2),((5,),40,10))
        self.assertEqual(mask_packet_spec(5,7,2),((40,),5,14))
        for code in range(4):
            for kind in range(13):
                if kind>=9 and code!=3:continue
                sizes,output,words=mask_packet_spec(8192,kind,code)
                self.assertGreaterEqual(words*4,16+sum(sizes))
                self.assertGreaterEqual(words*4,output)
                self.assertEqual(words%2,0)
                self.assertLessEqual(words,65536)

    def test_bounds_and_operator_rejection(self):
        for args in ((0,6,0),(65537,0,0),(32768,6,2),(1,13,0),(1,9,0),(1,0,4),(1.5,6,0)):
            with self.subTest(args=args),self.assertRaises(ValueError):mask_packet_spec(*args)


if __name__=='__main__':
    unittest.main()

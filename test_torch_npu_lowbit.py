"""Offline lowbit codec/IR checks; no driver or torch_intel_npu import."""
import unittest
import numpy as np
import torch
from torch_npu_lowbit import PackedWeight, NF4, pack_codes, quantize_weight, calibrate_int8, weight_graph,QuantizedLinear


class LowbitTests(unittest.TestCase):
    def test_calibration_reconstruction_and_guards(self):
        torch.manual_seed(647)
        weight=(torch.randn(7,129)/8).half().requires_grad_();snapshot=weight.detach().clone()
        inputs=torch.randn(48,129).half();inputs[:,1]=inputs[:,0];inputs[:,-1]=0
        inputs=inputs.t().contiguous().t()
        original=quantize_weight(weight,format='int8');prepared=calibrate_int8(weight,inputs)
        self.assertEqual(original.scales,prepared.scales);self.assertEqual(len(original.data),len(prepared.data))
        self.assertTrue(torch.equal(weight,snapshot));self.assertIsNone(weight.grad)
        def loss(q):
            codes=torch.from_numpy(np.frombuffer(q.data,dtype=np.int8).reshape(q.shape).astype(np.float64))
            scales=torch.from_numpy(np.frombuffer(q.scales,dtype=np.float16).astype(np.float64))
            return ((inputs.double()@codes.t())*scales[None]-inputs.double()@snapshot.double().t()).square().sum(0)
        self.assertTrue(torch.all(loss(prepared)<=loss(original)+1e-18))
        self.assertLess(float(loss(prepared).sum()),float(loss(original).sum()))
        self.assertEqual(prepared,calibrate_int8(weight,inputs))
        np.testing.assert_array_equal(np.frombuffer(prepared.data,dtype=np.int8).reshape(prepared.shape)[:,-1],
            np.frombuffer(original.data,dtype=np.int8).reshape(original.shape)[:,-1])
        self.assertEqual(original,calibrate_int8(weight,torch.zeros_like(inputs)))
        zero=torch.zeros_like(weight);self.assertEqual(quantize_weight(zero,format='int8'),calibrate_int8(zero,inputs))
        for bad in (None,torch.zeros(0,129),torch.zeros(2,128),torch.zeros(1,2,129),inputs.double(),
            torch.full_like(inputs,float('nan')),torch.full_like(inputs,float('inf')),torch.zeros(4097,129)):
            with self.assertRaises(ValueError):calibrate_int8(weight,bad)
        with self.assertRaises(ValueError):calibrate_int8(torch.ones(1,4097),torch.ones(1,4097))
        with self.assertRaises(ValueError):calibrate_int8(torch.ones(8192,1),torch.ones(513,1))

    def test_all_codes_and_odd_padding(self):
        codes=np.arange(16,dtype=np.uint8)
        self.assertEqual(pack_codes(codes),bytes([0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]))
        for fmt in ('int4','nf4'):
            weight=PackedWeight((1,16),fmt,16,pack_codes(codes),np.float16(1).tobytes())
            expected=np.arange(-8,8,dtype=np.float16)[(codes+8)%16] if fmt=='int4' else NF4.astype(np.float16)
            np.testing.assert_array_equal(weight.dequantize().ravel(),expected)
        self.assertEqual(pack_codes(codes[:3]),b'\x10\x02')

    def test_quantization_groups_zero_and_guards(self):
        torch.manual_seed(41)
        for shape,size in [((3,17),None),((8,64),32),((4,128),128)]:
            original=torch.randn(shape)/8
            for fmt in ('int4','nf4','int8'):
                weight=quantize_weight(original,format=fmt,group_size=size)
                self.assertEqual(weight.dequantize().shape,shape)
                self.assertEqual(len(weight.data),original.numel() if fmt=='int8' else (original.numel()+1)//2)
                graph,xml,data=weight_graph(weight,3)
                self.assertEqual(graph.outputs[0].dtype,'f16')
                self.assertIn(b'decompression',xml)
                self.assertIn(f'element_type="{weight.dtype}"'.encode(),xml)
                self.assertLess(len(data),original.numel()*2+32)
        for fmt in ('int4','nf4','int8'):
            zero=quantize_weight(torch.zeros(3,17),format=fmt)
            np.testing.assert_array_equal(zero.dequantize(),np.zeros((3,17),dtype=np.float16))
        for args in [dict(format='fp4'),dict(group_size=3),dict(group_size=True)]:
            with self.assertRaises(ValueError):quantize_weight(torch.ones(2,16),**args)
        for value in (float('inf'),float('nan')):
            with self.assertRaises(ValueError):quantize_weight(torch.full((2,16),value))
        with self.assertRaises(ValueError):
            PackedWeight((1,3),'int4',3,b'\x10\xf2',np.float16(1).tobytes())

    def test_group_graphs_and_physical_bounds(self):
        import xml.etree.ElementTree as ET
        q=quantize_weight(torch.randn(5,9),group_size=3)
        graph,xml,_=weight_graph(q,1,compact_groups=True)
        layers=ET.fromstring(xml).findall('./layers/layer')
        self.assertEqual(sum(l.attrib['type']=='MatMul' for l in layers),3)
        self.assertEqual(sum(l.find('./rt_info/attribute[@name="decompression"]') is not None for l in layers),3)
        self.assertEqual(graph.outputs[0].shape,(1,5))
        self.assertEqual(graph.inputs[0].shape,(1,9))
        graph,xml,_=weight_graph(q,1,pad_batch=True)
        self.assertEqual(graph.inputs[0].shape,(1,9))
        self.assertEqual(graph.outputs[0].shape,(1,5))
        self.assertIn(b'Concat',xml);self.assertIn(b'StridedSlice',xml)
        layer=QuantizedLinear(q,compact_groups=True)
        self.assertTrue(layer.compact_groups)
        with self.assertRaises(AttributeError):layer.compact_groups=False
        for options in ({'compact_groups':1},{'pad_batch':1}):
            with self.assertRaises(ValueError):weight_graph(q,1,**options)
        large=quantize_weight(torch.ones(1,65536),group_size=32768)
        with self.assertRaises(ValueError):weight_graph(large,1,pad_batch=True)
        self.assertEqual(weight_graph(large,1,compact_groups=True)[0].outputs[0].shape,(1,1))
        many=quantize_weight(torch.ones(1,33),group_size=1)
        with self.assertRaises(ValueError):QuantizedLinear(many,compact_groups=True)

    def test_int8_codec_and_activation_guards(self):
        codes=np.arange(-128,128,dtype=np.int8)
        q=PackedWeight((1,256),'int8',256,codes.tobytes(),np.float16(.125).tobytes())
        np.testing.assert_array_equal(q.dequantize().ravel(),codes.astype(np.float16)*.125)
        graph,xml,_=weight_graph(q,1,activation_scale=1/128)
        self.assertEqual(xml.count(b'type="FakeQuantize"'),2)
        self.assertEqual(graph.inputs[0].dtype,graph.outputs[0].dtype)
        layer=QuantizedLinear(q,activation_scale=1/128)
        self.assertEqual(layer.activation_scale,1/128)
        with self.assertRaises(AttributeError):layer.activation_scale=1/64
        for scale in (True,0,-1,2**-25,512,.0037,float('nan'),float('inf'),'0.125'):
            with self.assertRaises(ValueError):QuantizedLinear(q,activation_scale=scale)
        for fmt,size in [('int4',256),('nf4',256),('int8',128)]:
            weight=quantize_weight(torch.ones(1,256),format=fmt,group_size=size)
            with self.assertRaises(ValueError):QuantizedLinear(weight,activation_scale=1/128)
        with self.assertRaises(ValueError):QuantizedLinear(q,activation_scale=1/128,compact_groups=True)
        for data in (codes.tobytes()[:-1],codes.tobytes()+b'\0',bytearray(codes)):
            with self.assertRaises(ValueError):PackedWeight(q.shape,q.format,q.group_size,data,q.scales)
        large=PackedWeight((1,256),'int8',256,codes.tobytes(),np.float16(512).tobytes())
        with self.assertRaises(ValueError):QuantizedLinear(large,activation_scale=1/128)


if __name__=='__main__':unittest.main()

"""Offline packet validation and advanced-index layout reference checks."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import npunlock.npu4000 as kernels


def geometry():
    source=ast.parse(Path(__file__).with_name('torch_npu_compat_extended.py').read_text())
    selected=[node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in ('index_geometry','_address_metadata')]
    def integer_binary(a,b,kind):return {'add':torch.add,'mul':torch.mul}[kind](a,b)
    def normalize(ids,size):
        if ((ids < -size)|(ids >= size)).any():raise IndexError('index out of range')
        return ids.to(torch.int64)%size
    env=dict(torch=torch,npu=SimpleNamespace(_materialize=lambda x:x.contiguous(),MAX_RANK=8),
             nonzero=torch.nonzero,normalize=normalize,integer_binary=integer_binary,prod=__import__('math').prod)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<index geometry>','exec'),env)
    return env


class ExtendedCompatibilityTests(unittest.TestCase):
    def test_advanced_index_offsets_match_torch(self):
        fn=geometry()['index_geometry'];x=torch.arange(120).reshape(4,5,6)
        for indices in ([torch.tensor([3,1,-1])],
                        [torch.tensor([[0],[3]]),None,torch.tensor([[1,4,5]])],
                        [None,torch.tensor([4,1]),torch.tensor([5,0])],
                        [torch.tensor([True,False,True,False])],
                        [torch.zeros((4,5),dtype=torch.bool)],
                        [torch.tensor(True)],[torch.tensor(False)],
                        [None,None,torch.tensor(2)],
                        [torch.tensor([],dtype=torch.int64)]):
            with self.subTest(indices=indices):
                ids,shape=fn(x,indices);expected=x[tuple(slice(None) if v is None else v for v in indices)]
                self.assertEqual(shape,tuple(expected.shape));torch.testing.assert_close(ids,expected)

    def test_strided_metadata_matches_numpy(self):
        fn=geometry()['_address_metadata'];base=torch.arange(60).reshape(5,12)
        for x in (base[1:4,::3],base.t()[2:5],base.as_strided((4,4),(1,1))):
            np.testing.assert_array_equal(fn(x),x.numpy())

    def test_bad_index_fails_before_payload_work(self):
        fn=geometry()['index_geometry'];x=torch.ones(3,4)
        for ids in ([torch.tensor([3])],[torch.tensor([-4])],[torch.ones(2,dtype=torch.bool)]):
            with self.assertRaises(IndexError):fn(x,ids)

    def test_byte_packet_static_padding_matches_output_width(self):
        header=(3,0,4,2)+(0,)*11+(28,)
        with patch.object(kernels,'compile_precision_graph',return_value='compiled') as compiler:
            self.assertEqual(kernels.compile_byte_packet(header,(3,)),'compiled')
            graph=compiler.call_args.args[0];self.assertEqual(graph.outputs[0].shape,(1,72))
            self.assertTrue(compiler.call_args.kwargs['dma_only'])

    def test_packet_limits_reject_before_compilation(self):
        with patch.object(kernels,'compile_precision_graph') as compiler:
            for h,s in (((-1,), (4,)),((1,), (0,)),((1,), (262144,)),((1,),())):
                with self.assertRaises(ValueError):kernels.compile_byte_packet(h,s)
            compiler.assert_not_called()

    def test_sort_retains_its_own_capacity_and_dtype_guard(self):
        source=ast.parse(Path(__file__).with_name('torch_npu_sort.py').read_text())
        node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='sort')
        env=dict(torch=torch,npu=SimpleNamespace(_lock=nullcontext(),MAX_TENSOR_ELEMENTS=16777216),_axis=lambda x,dim:0)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<sort boundary>','exec'),env)
        with self.assertRaises(ValueError):env['sort'](torch.empty(16777217,device='meta'))
        with self.assertRaises(NotImplementedError):env['sort'](torch.empty(3,dtype=torch.int32,device='meta'))


if __name__=='__main__':unittest.main()

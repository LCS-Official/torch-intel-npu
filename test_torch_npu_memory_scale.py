"""Memory planning and constant-size streaming scratch, independent of NPU hardware."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from contextlib import nullcontext
import torch
from torch_npu_memory_scale_demo import training_plan,memory_status


class MemoryScaleTests(unittest.TestCase):
    def test_pointwise_graph_key_ignores_logical_rank_and_shape(self):
        tree=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_execute');keys=[]
        def run(key,inputs,shape,dtype,out):
            keys.append(key)
            return torch.neg(inputs[0],out=out) if out is not None else -inputs[0]
        env=dict(torch=torch,_lock=nullcontext(),synchronize=lambda:None,_validate_float_read=lambda *a:None,_materialize=lambda x:x,
                 _check_out=lambda *a:None,MAX_ELEMENTS=65536,bridge=SimpleNamespace(_DTYPES={torch.float32:('f32',None)}),_run=run)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<pointwise-key>','exec'),env)
        for shape in ((16,),(4,4),(1,)*63+(16,)):
            x=torch.arange(16,dtype=torch.float32).view(shape);out=torch.empty_like(x)
            self.assertIs(env['_execute']('neg',x,out=out),out)
            result=env['_execute']('neg',x)
            torch.testing.assert_close(result,-x,rtol=0,atol=0)
            self.assertIsNone(result._base)
        self.assertEqual(set(keys),{('neg','f32',(16,))})
        self.assertEqual(env['_execute']('neg',torch.tensor(2.)).shape,torch.Size([]))

    def test_large_arithmetic_binds_torch_ranges_without_staging_tiles(self):
        tree=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_execute');calls=[]
        def run(key,inputs,out):
            calls.append((key[2],inputs[0].storage_offset(),out.storage_offset()))
            return out.copy_(inputs[0]*inputs[1])
        env=dict(torch=torch,_lock=nullcontext(),synchronize=lambda:None,_validate_float_read=lambda *a:None,_materialize=lambda x:x,
                 _check_out=lambda *a:None,MAX_ELEMENTS=65536,bridge=SimpleNamespace(_DTYPES={torch.float32:('f32',None)}),
                 _run_ranges=run)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<range-arithmetic>','exec'),env)
        x=torch.linspace(-1,1,2*65536+17);y=torch.ones_like(x)*.25;out=torch.empty_like(x)
        self.assertIs(env['_execute']('mul',x,y,out=out),out)
        torch.testing.assert_close(out,x*y,rtol=0,atol=0)
        self.assertEqual(calls,[((65536,),0,0),((65536,),65536,65536),((17,),131072,131072)])

    def test_training_budget_counts_parameters_gradients_and_both_adam_moments(self):
        plan=training_plan(14.,4096)
        self.assertEqual(plan['layers'],55)
        self.assertEqual(plan['parameters'],922972160)
        self.assertEqual(plan['resident_state_bytes'],16*plan['parameters'])
        self.assertLess(plan['estimated_peak_bytes'],14.25*2**30)
        self.assertEqual(training_plan(1.,4096,2)['layers'],2)
        for args in ((15.,4096),(float('nan'),4096),(0.,4096),(14.,8193),(14.,4096,100),(14.,32,0)):
            with self.assertRaises(ValueError):training_plan(*args)

    def test_streaming_map_matches_torch_with_short_final_tile_and_out_identity(self):
        tree=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_map_tiles')
        sizes=[]
        def slices(x,start,count):sizes.append(count);return x[start:start+count].clone()
        env=dict(torch=SimpleNamespace(empty=lambda shape,**kw:torch.empty(shape,dtype=kw['dtype'])),prod=lambda s:s[0],
                 MAX_ELEMENTS=65536,_check_out=lambda *a:None,_slice=slices,_range_copy=lambda dst,src:dst.copy_(src))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<streaming-map>','exec'),env)
        x=torch.linspace(-1,1,3*65536+17);out=torch.empty_like(x)
        got=env['_map_tiles'](lambda tile:tile*.25+.125,[x],tuple(x.shape),x.dtype,out)
        self.assertIs(got,out);torch.testing.assert_close(got,x*.25+.125,rtol=0,atol=0)
        self.assertEqual(sizes,[65536,65536,65536,17])

    def test_windows_memory_probe_reports_physical_and_commit_headroom(self):
        state=memory_status()
        self.assertGreater(state['physical_total'],state['physical_available'])
        self.assertGreater(state['physical_available'],0);self.assertGreater(state['commit_available'],0)


if __name__=='__main__':unittest.main()

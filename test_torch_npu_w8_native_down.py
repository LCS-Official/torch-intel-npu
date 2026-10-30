"""Native-main fitting must use the exact observed input rows."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch


def helper():
    path=Path(__file__).parent/'npunlock-probe/research/w8_native_down_check.py'
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='aligned_native_main']
    assert len(nodes)==1;env=dict(torch=torch)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env['aligned_native_main']


class NativeInputTests(unittest.TestCase):
    def test_native_main_preserves_row_order_and_half_values(self):
        x=torch.tensor([[1.,2.],[3.,4.]]).half();y=torch.tensor([[.5],[1.]]).half()
        actual=helper()([(x[:1],y[:1]),(x[1:],y[1:])],x)
        torch.testing.assert_close(actual,y,rtol=0,atol=0)

    def test_reordered_or_changed_inputs_are_rejected(self):
        x=torch.tensor([[1.,2.],[3.,4.]]).half();y=torch.ones(2,1).half()
        for rows in ([(x.flip(0),y)],[(x+.5,y)],[(x[:1],y[:1])]):
            with self.assertRaises(ValueError):helper()(rows,x)

    def test_missing_nonfinite_or_nonhalf_observations_are_rejected(self):
        x=torch.ones(2,2).half();y=torch.ones(2,1).half()
        for rows in ([],[(x,y.float())],[(x,torch.full_like(y,float('nan')))],[(x,y[:1])]):
            with self.assertRaises(ValueError):helper()(rows,x)


if __name__=='__main__':unittest.main()

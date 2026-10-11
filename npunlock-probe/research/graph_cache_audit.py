"""Check cross-process cache equivalence and the production integration scope."""
import ast
import json
from pathlib import Path
import numpy as np
from model_cross_device import ROOT,sha,write
from model_cross_device_followup import OUT


def main():
    labels=('npu-cache-fill','npu-cache-read','npu-cache-corrupt')
    reports={label:json.loads((OUT/(label+'.json')).read_text()) for label in labels}
    for value in reports.values():
        assert value['status']=='passed' and value['quality_gate_passed']
        assert value['final_stats']['closed'] and not value['final_stats']['context_failed']
        assert all(value['final_stats'][k]==0 for k in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes','cached_sessions'))
    assert reports[labels[0]]['graph_cache']['compiles']==reports[labels[0]]['graph_cache']['writes']==60
    assert reports[labels[1]]['graph_cache']['hits']==60 and reports[labels[1]]['graph_cache']['compiles']==0
    assert reports[labels[2]]['graph_cache']['hits']==59 and reports[labels[2]]['graph_cache']['rejected']==reports[labels[2]]['graph_cache']['compiles']==1
    with np.load(OUT/(labels[0]+'-outputs.npz'),allow_pickle=False) as data:baseline={k:data[k].copy() for k in data.files}
    for label in labels[1:]:
        path=OUT/(label+'-outputs.npz');assert sha(path)==reports[label]['outputs_sha256']
        with np.load(path,allow_pickle=False) as data:
            assert set(data.files)==set(baseline)
            for key in baseline:np.testing.assert_array_equal(data[key].view(np.uint8),baseline[key].view(np.uint8))
    before=ast.parse((OUT/'before/torch_npu_smollm.py').read_text())
    after=ast.parse((ROOT/'torch_npu_smollm.py').read_text())
    after.body=[node for node in after.body if not (isinstance(node,ast.ImportFrom) and node.module=='torch_npu_graph_cache')]
    class Restore(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='compile_fused_cached':
                node.func.id='compile_fused_rmsnorm';node.args=node.args[1:]
                node.keywords=[k for k in node.keywords if k.arg not in ('graph_digest','compiler_factory')]
                node.keywords.insert(0,ast.keyword(arg='libraries',value=ast.Attribute(value=ast.Name(id='npu',ctx=ast.Load()),attr='_native',ctx=ast.Load())))
            return node
    assert ast.dump(Restore().visit(after))==ast.dump(before)
    checks=json.loads((OUT/'cache-checks.json').read_text());assert checks['status']=='passed'
    assert checks['source_sha256']==sha(ROOT/'torch_npu_graph_cache.py')
    directory=Path(reports[labels[2]]['graph_cache']['directory']);files=list(directory.glob('*.npc'));size=sum(p.stat().st_size for p in files)
    assert len(files)==60 and size<=512*1024*1024 and not (directory/'.writer.lock').exists() and not list(directory.glob('*.tmp'))
    result=dict(status='passed',cold_ms={k:v['cold']['elapsed_ms'] for k,v in reports.items()},
        cached_entries=len(files),cached_bytes=size,all_outputs_bit_equal=True,production_change='compiler wrapper only; graph AST unchanged',
        source_sha256=sha(ROOT/'torch_npu_graph_cache.py'),guard_checks=len(checks['checks']))
    write(OUT/'cache-audit.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

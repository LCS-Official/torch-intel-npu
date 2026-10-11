"""Compile-only B1 carrier layouts; no unverified kernel is executed."""
import hashlib
import json
from pathlib import Path
import traceback

from rmsnorm_fusion_probe import power_tasks
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-b1-fusion'


def main():
    import numpy as np
    import npunlock as s
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolLayer
    report=dict(status='running',checks=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['npunlock-probe/research/rmsnorm_b1_layout_probe.py','torch_npu_smollm.py','npunlock/python/npunlock/npu4000.py']})
    try:
        model=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16')
        weight=model._cpu_weight('self_attn.q_proj.weight').numpy()
        def first(v):
            return s.op('StridedSlice',v,*(s.constant(np.array(a,dtype=np.int64)) for a in [(0,0),(1,576),(1,1)]),
                _shape=(1,576),_dtype='f16',begin_mask=[0,0],end_mask=[0,0],new_axis_mask=[0,0],shrink_axis_mask=[0,0],ellipsis_mask=[0,0])
        for scheme,tiles in [(a,t) for a in ['plain','norm_dup','norm_aux','all_dup','matmul_aux'] for t in [4,1]]:
            tag=f'{scheme}-t{tiles}';entry=dict(tag=tag,scheme=scheme,tiles=tiles)
            try:
                rows=1 if scheme=='plain' else 2
                x=s.input('x',shape=(1,576),dtype='f16');gamma=s.input('gamma',shape=(rows,576),dtype='f16')
                value=x if rows==1 else s.op('Concat',x,x,_shape=(2,576),_dtype='f16',axis=0)
                norm=s.op('Power',value,gamma,_shape=(rows,576),_dtype='f16',_name='rmsnorm_carrier')
                all_rows=scheme in ['all_dup','matmul_aux'];left=norm if all_rows or rows==1 else first(norm)
                y=s.op('MatMul',left,s.constant(weight),_shape=left.shape,_dtype='f16',transpose_a=False,transpose_b=True)
                outputs=[first(y) if all_rows else y]
                if scheme=='norm_aux':outputs.append(norm)
                if scheme=='matmul_aux':outputs.append(y)
                graph=s.Graph([x,gamma],outputs,name='b1_complete_row_probe');ir=s.serialize_ir(graph)
                flags='--config EXECUTION_MODE_HINT="ACCURACY"'+(' NPU_TILES="1"' if tiles==1 else '')
                (OUT/f'{tag}.xml').write_bytes(ir.xml);(OUT/f'{tag}.bin').write_bytes(ir.weights)
                result=npu._native.compile_ir(ir.xml,ir.weights,build_flags=flags,timeout_ms=20000)
                assert (result.vendor_id,result.device_id,tuple(result.compiler_version))==(0x8086,0x643e,(8,3))
                (OUT/f'{tag}.blob').write_bytes(result.graph_blob);layout=power_tasks(result.graph_blob)
                (OUT/f'{tag}-layout.json').write_text(json.dumps(layout,indent=2),encoding='utf-8')
                refs=[ref for t in layout['invocations'] for ref in t['memrefs']]
                complete=bool(refs) and all(ref['rank']==4 and ref['order']=='0x4321' and
                    ref['dims']==(576,ref['elements']//576,1,1) and ref['strides_bits']==(16,9216,16*ref['elements'],16*ref['elements']) for ref in refs)
                entry.update(status='compiled',flags=flags,blob_sha256=hashlib.sha256(result.graph_blob).hexdigest(),
                    power_tasks=len(layout['invocations']),power_dims=[t['memrefs'][0]['dims'] for t in layout['invocations']],
                    complete_rows=complete,dpu_sections=len(layout['dpu_sections']),outputs=[list(o.shape) for o in outputs])
            except Exception as error:entry.update(status='compile_or_inspection_failed',error=str(error))
            report['checks'].append(entry);print(entry,flush=True)
        npu.shutdown();report.update(status='passed',final_stats=npu.stats());assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
    except Exception as error:report.update(status='failed',error=str(error),traceback=traceback.format_exc());print(report['traceback'],flush=True)
    (OUT/'layout-probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

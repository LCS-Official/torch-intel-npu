"""Compile-only layout checks for an RMSNorm Power carrier feeding DPU GEMM."""
import hashlib
import json
from pathlib import Path
import struct
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-fusion'


def power_tasks(blob):
    from npunlock.npu4000 import Elf,POWER_TEXT_SHA256,bounded
    elf=Elf(blob);text=elf.content(elf.named['shave.text']);ranges={};windows=set();tasks=[]
    for section in elf.sections:
        if not section['name'].startswith('task.shave.range.'):continue
        assert section['size']%40==0
        for offset in range(0,section['size'],40):
            raw=bounded(elf.content(section),offset,40)
            if raw[0]!=0:continue
            relocation=elf.relocation(section['name'],offset+16)
            assert relocation['symbol_section']=='shave.text' and relocation['type']==0
            start=relocation['symbol_value']+relocation['addend'];size=struct.unpack_from('<I',raw,24)[0]
            if hashlib.sha256(bounded(text,start,size)).hexdigest()!=POWER_TEXT_SHA256:continue
            windows.add((start,size));ranges[(section['name'],offset//40)]=dict(kernel_entry=hex(struct.unpack_from('<Q',raw,8)[0]))
    for section in elf.sections:
        if not section['name'].startswith('task.shave.invocation.'):continue
        assert section['size']%96==0
        tile,engine=map(int,section['name'].removeprefix('task.shave.invocation.').split('.'))
        for offset in range(0,section['size'],96):
            raw=bounded(elf.content(section),offset,96);index=struct.unpack_from('<I',raw,88)[0]
            identity=(f'task.shave.range.{tile}.{engine}',index)
            if identity not in ranges:continue
            params=elf.relocation(section['name'],offset+8);assert params['symbol_section']=='shave.params' and params['type']==0
            base=params['symbol_value']+params['addend'];refs=[elf.memref(base+i*40) for i in range(3)]
            tasks.append(dict(section=section['name'],index=offset//96,tile=tile,range_index=index,memrefs=refs))
    return dict(windows=sorted(windows),ranges=[dict(section=k[0],index=k[1],**v) for k,v in ranges.items()],invocations=tasks,
        dpu_sections=[dict(name=s['name'],size=s['size']) for s in elf.sections if s['name'].startswith('task.dpu.') and s['size']])


def main():
    import numpy as np
    import npunlock as s
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolLayer
    report=dict(status='running',checks=[],sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['npunlock-probe/research/rmsnorm_fusion_probe.py','torch_npu_smollm.py','npunlock/python/npunlock/npu4000.py']})
    try:
        model=FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16')
        weight=model._cpu_weight('self_attn.q_proj.weight').numpy()
        for rows,tiles,keep_norm in [(r,t,k) for r in [1,2,4,8,16] for t in [None,1] for k in [False,True]]:
            tag=f'r{rows}-t{tiles or 4}-aux{int(keep_norm)}';entry=dict(tag=tag,rows=rows,tiles=tiles or 4,keep_norm=keep_norm)
            try:
                x=s.input('x',shape=(rows,576),dtype='f16');gamma=s.input('gamma',shape=x.shape,dtype='f16')
                norm=s.op('Power',x,gamma,_shape=x.shape,_dtype='f16',_name='rmsnorm_carrier')
                y=s.op('MatMul',norm,s.constant(weight),_shape=(rows,576),_dtype='f16',transpose_a=False,transpose_b=True,_name='y')
                graph=s.Graph([x,gamma],[y,norm] if keep_norm else [y],name='rmsnorm_projection_probe');ir=s.serialize_ir(graph)
                flags='--config EXECUTION_MODE_HINT="ACCURACY"'+(' NPU_TILES="1"' if tiles==1 else '')
                (OUT/f'{tag}.xml').write_bytes(ir.xml);(OUT/f'{tag}.bin').write_bytes(ir.weights)
                result=npu._native.compile_ir(ir.xml,ir.weights,build_flags=flags,timeout_ms=20000)
                assert (result.vendor_id,result.device_id,tuple(result.compiler_version))==(0x8086,0x643e,(8,3))
                (OUT/f'{tag}.blob').write_bytes(result.graph_blob);layout=power_tasks(result.graph_blob)
                (OUT/f'{tag}-layout.json').write_text(json.dumps(layout,indent=2),encoding='utf-8')
                counts=[t['memrefs'][0]['elements'] for t in layout['invocations']]
                complete=bool(counts) and all(t['memrefs'][0]['dims'][0]==576 and t['memrefs'][0]['strides_bits'][0]==16 for t in layout['invocations'])
                entry.update(status='compiled',flags=flags,blob_sha256=hashlib.sha256(result.graph_blob).hexdigest(),
                    power_tasks=len(counts),power_elements=counts,power_dims=[t['memrefs'][0]['dims'] for t in layout['invocations']],
                    complete_rows=complete,dpu_sections=len(layout['dpu_sections']))
            except Exception as error:entry.update(status='compile_or_inspection_failed',error=str(error))
            report['checks'].append(entry);print(entry,flush=True)
        npu.shutdown();report.update(status='passed',final_stats=npu.stats());assert not npu.stats()['context_failed'] and npu.stats()['live_buffers']==0
    except Exception as error:report.update(status='failed',error=str(error),traceback=traceback.format_exc());print(report['traceback'],flush=True)
    (OUT/'layout-probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

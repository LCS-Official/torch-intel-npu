"""Offline malformed-IR/ELF checks for selected-kernel patching."""
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/rmsnorm-fusion'


def main():
    import numpy as np
    import npunlock as s
    from npunlock.npu4000 import Elf,compile_fused_rmsnorm,inspect,POWER_TEXT_SHA256
    from torch_npu_smollm import _rmsnorm_kernel
    x=s.input('x',shape=(2,576),dtype='f16');gamma=s.input('gamma',shape=x.shape,dtype='f16')
    norm=s.op('Power',x,gamma,_shape=x.shape,_dtype='f16',_name='rmsnorm_carrier')
    y=s.op('MatMul',norm,s.constant(np.ones((576,576),dtype=np.float16)),_shape=x.shape,_dtype='f16',transpose_a=False,transpose_b=True)
    graph=s.Graph([x,gamma],[y]);ir=s.serialize_ir(graph);original=(OUT/'r2-t4-aux0.blob').read_bytes();kernel=_rmsnorm_kernel()
    class Driver:
        def __init__(self,blob):self.blob=blob;self.calls=0
        def compile_ir(self,*args,**kwargs):
            self.calls+=1;return SimpleNamespace(graph_blob=self.blob,vendor_id=0x8086,device_id=0x643e,compiler_version=(8,3))
    valid=compile_fused_rmsnorm(graph,ir.xml,ir.weights,kernel,libraries=Driver(original));patch=json.loads(valid.patch_reports[0])
    start=patch['code_offset'];end=start+patch['code_bytes']
    assert original[:start]==valid.graph_blob[:start] and original[end:]==valid.graph_blob[end:]
    checks=[dict(name='valid_complete_rows',passed=True,outside_code_unchanged=True)]
    def reject(name,*,xml=ir.xml,blob=original):
        driver=Driver(blob)
        try:compile_fused_rmsnorm(graph,xml,ir.weights,kernel,libraries=driver)
        except ValueError as error:checks.append(dict(name=name,rejected=True,error=str(error),compile_calls=driver.calls))
        else:raise AssertionError(name+' was accepted')
    root=ET.fromstring(ir.xml);carrier=root.find("./layers/layer[@type='Power']")
    carrier.set('name','unrelated_power');reject('wrong_ir_name',xml=ET.tostring(root))
    root=ET.fromstring(ir.xml);carrier=root.find("./layers/layer[@type='Power']")
    carrier.find('./input/port').set('precision','FP32');reject('ir_precision_mismatch',xml=ET.tostring(root))
    root=ET.fromstring(ir.xml);carrier=root.find("./layers/layer[@type='Power']")
    root.find('layers').append(ET.fromstring(ET.tostring(carrier)));reject('additional_power',xml=ET.tostring(root))
    elf=Elf(original);evidence=inspect(original,memref_count=3,kernel_sha256=POWER_TEXT_SHA256)
    ref=evidence['invocations'][0]['memrefs'][0];params=elf.named['shave.params']['offset'];task=evidence['invocations'][0]
    changes=[('power_fingerprint',start,'<B',original[start]^1),
        ('input_dtype',params+ref['offset']+24,'<I',1),
        ('input_outside_cmx',params+ref['offset'],'<Q',0x40000000),
        ('input_output_alias',params+ref['offset'],'<Q',int(task['memrefs'][-1]['unrelocated_data'],16)),
        ('missing_power_invocation',elf.named[task['section']]['offset']+task['index']*96+88,'<I',999)]
    for name,offset,fmt,value in changes:
        corrupt=bytearray(original);struct.pack_into(fmt,corrupt,offset,value);reject(name,blob=bytes(corrupt))
    # All production pure-carrier inspections continue through their default
    # path; the existing offline suite covers those APIs separately.
    report=dict(status='passed',checks=checks,sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['npunlock-probe/research/rmsnorm_fusion_guards.py','npunlock/python/npunlock/npu4000.py','torch_npu_smollm.py']})
    (OUT/'patch-guards.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report['status'],len(checks))


if __name__=='__main__':main()

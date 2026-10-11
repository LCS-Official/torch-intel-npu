"""Reject unsafe whole-layer carrier contracts before device execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'npunlock-probe/research/results/smollm-whole-layer'


def main():
    from npunlock.npu4000 import Elf,compile_fused_rmsnorm,inspect,POWER_TEXT_SHA256
    from torch_npu_smollm import _rmsnorm_kernel
    path=OUT/'candidate.py';spec=importlib.util.spec_from_file_location('whole_guard_candidate',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.__file__=str(ROOT/'torch_npu_smollm.py')
    model=module.FrozenSmolLayer.from_checkpoint(ROOT/'models/SmolLM2-135M-Instruct',mode='fp16')
    graph,xml,weights=model.graph((2,1,576),segment='layer',kv_cache='decode',fused_norm=True)
    original=(OUT/'fp16-b2-decode-original.blob').read_bytes();kernel=_rmsnorm_kernel()
    class Driver:
        def __init__(self,blob):self.blob=blob;self.calls=0
        def compile_ir(self,*args,**kwargs):
            self.calls+=1;return SimpleNamespace(graph_blob=self.blob,vendor_id=0x8086,device_id=0x643e,compiler_version=(8,3))
    valid=compile_fused_rmsnorm(graph,xml,weights,kernel,libraries=Driver(original),whole_layer=True)
    patch=json.loads(valid.patch_reports[0]);start=patch['code_offset'];end=start+patch['code_bytes']
    assert original[:start]==valid.graph_blob[:start] and original[end:]==valid.graph_blob[end:]
    checks=[dict(name='both_complete_row_layouts',passed=True,outside_code_unchanged=True)]
    graph3,xml3,weights3=model.graph((3,1,576),segment='layer',kv_cache='decode',fused_norm=True)
    blob3=(OUT/'fp16-b3-decode-original.blob').read_bytes()
    compile_fused_rmsnorm(graph3,xml3,weights3,kernel,libraries=Driver(blob3),whole_layer=True)
    checks.append(dict(name='singleton_two_complete_rows',passed=True))
    def reject(name,*,ir=xml,blob=original,whole=True):
        driver=Driver(blob)
        try:compile_fused_rmsnorm(graph,ir,weights,kernel,libraries=driver,whole_layer=whole)
        except (ValueError,TypeError) as error:checks.append(dict(name=name,rejected=True,error=str(error),compile_calls=driver.calls))
        else:raise AssertionError(name+' was accepted')
    reject('two_carriers_in_single_carrier_api',whole=False)
    reject('non_bool_mode',whole=1)
    for name,change in [
        ('second_carrier_wrong_name',lambda c:c.set('name','unrelated_power')),
        ('second_carrier_wrong_dtype',lambda c:c.find('./output/port').set('precision','FP32')),
        ('second_carrier_wrong_width',lambda c:setattr(c.find('./input/port/dim[2]'),'text','288')),
    ]:
        root=ET.fromstring(xml);carrier=root.find("./layers/layer[@name='rmsnorm_ffn']");change(carrier)
        reject(name,ir=ET.tostring(root))
    root=ET.fromstring(xml);carrier=root.find("./layers/layer[@name='rmsnorm_ffn']")
    root.find('layers').remove(carrier);reject('second_carrier_missing',ir=ET.tostring(root))
    root=ET.fromstring(xml);root.find('layers').append(ET.fromstring(ET.tostring(root.find("./layers/layer[@name='rmsnorm_ffn']"))))
    reject('third_carrier',ir=ET.tostring(root))
    elf=Elf(original);evidence=inspect(original,memref_count=3,kernel_sha256=POWER_TEXT_SHA256)
    layouts={ref['order'] for task in evidence['invocations'] for ref in task['memrefs']}
    assert layouts=={'0x4321','0x2431'},layouts
    task=next(t for t in evidence['invocations'] if t['memrefs'][0]['order']=='0x2431')
    ref=task['memrefs'][0];params=elf.named['shave.params']['offset']
    def array_offset(field):
        reloc=elf.relocation('shave.params',ref['offset']+field)
        return params+reloc['symbol_value']+reloc['addend']
    changes=[('power_fingerprint',start,'<B',original[start]^1),
        ('singleton_bad_stride',array_offset(20)+8,'<q',32),
        ('singleton_split_row',array_offset(16),'<iiii',(2,288,1,1)),
        ('singleton_wrong_order',params+ref['offset']+28,'<Q',0x4321),
        ('singleton_input_dtype',params+ref['offset']+24,'<I',1),
        ('singleton_outside_cmx',params+ref['offset'],'<Q',0x40000000),
        ('singleton_input_output_alias',params+ref['offset'],'<Q',int(task['memrefs'][-1]['unrelocated_data'],16)),
        ('missing_power_invocation',elf.named[task['section']]['offset']+task['index']*96+88,'<I',999)]
    for name,offset,fmt,value in changes:
        corrupt=bytearray(original);struct.pack_into(fmt,corrupt,offset,*(value if isinstance(value,tuple) else (value,)))
        reject(name,blob=bytes(corrupt))
    report=dict(status='passed',checks=checks,sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['npunlock-probe/research/smollm_whole_layer_guards.py','npunlock/python/npunlock/npu4000.py','torch_npu_smollm.py']},
        candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),original_sha256=hashlib.sha256(original).hexdigest())
    (OUT/'patch-guards.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report['status'],len(checks))


if __name__=='__main__':main()

"""Guarded experimental replacement of this driver's FP16 Select code."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from math import prod
from npunlock.npu4000 import Elf, inspect, kernel_text, _dense_float
from probe_movi_targets import stage
from w8_quality_probe import ROOT, sha

OUT=ROOT/'npunlock-probe/research/results/kv-group/select-kernel'
FINGERPRINT='2bb6d89b2a4d94c392058f5b15e77418d68891e6e060e56feaf19a6040774af7'


def build():
    OUT.mkdir(parents=True,exist_ok=True);source=Path(__file__).with_suffix('.c')
    source=source.with_name('kv_select.c');data=source.read_bytes()
    args=['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2','-ffp-contract=off','-x','c','entry.c','-o','-']
    asm,ok=stage('compile','kv-select-f16','moviCompile64.dll',1,args,data);assert ok
    obj,ok=stage('assemble','kv-select-f16','moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm);assert ok
    elf,ok=stage('link','kv-select-f16','moviLLD64.dll',3,['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
        (obj,(ROOT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()));assert ok and len(kernel_text(elf))<=23088
    for name,value in (('select.c',data),('select.s',asm),('select.elf',elf)):(OUT/name).write_bytes(value)
    report=dict(source_sha256=sha(source),builder_sha256=sha(Path(__file__)),elf_sha256=sha(OUT/'select.elf'),
        assembly_sha256=sha(OUT/'select.s'),text_bytes=len(kernel_text(elf)),original_fingerprint=FINGERPRINT,arguments=args)
    (OUT/'build.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report,flush=True)


def patch(blob,kernel):
    elf=Elf(blob);evidence=inspect(blob,kernel_sha256=FINGERPRINT)
    if not evidence['ranges']:return blob,dict(operation='select',invocations=0)
    windows=set();counts={};contracts=[];pointer_bases={}
    for task in evidence['ranges']:
        rel=task['text_relocation'];assert task['workload_type']==0 and task['kernel_entry']=='0x1d000000'
        assert rel['type']==0 and rel['symbol_section']=='shave.text'
        windows.add((rel['symbol_value']+rel['addend'],task['code_size']))
        counts[task['section']]=elf.named[task['section']]['size']//40
        n=sum(t['section'].replace('invocation','range')==task['section'] and t['range_index']==task['index'] for t in evidence['invocations'])
        assert n and task['invocation_count'] in (0,n)
    assert len(windows)==1;offset,extent=windows.pop();assert extent==23088
    text=elf.named['shave.text'];assert hashlib.sha256(elf.content(text)[offset:offset+extent]).hexdigest()==FINGERPRINT
    for task in evidence['invocations']:
        tile,engine=map(int,task['section'].removeprefix('task.shave.invocation.').split('.'));section=f'task.shave.range.{tile}.{engine}'
        assert task['tile']==tile and 0<=task['range_index']<counts[section]
        rels=task['range_relocations']
        if rels:
            assert len(rels)==1;rel=rels[0]
            assert rel['type']==0 and rel['symbol_section']==section and rel['symbol_value']+rel['addend']==task['range_index']*40
        else:
            pointer=int(task['unrelocated_range_pointer'],16);base=pointer-task['range_index']*40
            assert 0<base<0x100000000 and pointer_bases.setdefault(section,base)==base
        rel=task['params_relocation'];assert rel['type']==0 and rel['symbol_section']=='shave.params'
        base=rel['symbol_value']+rel['addend'];refs=[elf.memref(base+i*40) for i in range(4)];out=refs[-1]
        assert all(r['rank']==4 and r['order']=='0x4321' and r['dtype']==2 for r in refs)
        assert out['elements']<=65536
        assert all(all(a in (1,b) for a,b in zip(r['dims'],out['dims'])) for r in refs[:-1])
        width,rows,heads,n=out['dims'];assert width==64 and 1<=rows<=64 and 1<=heads<=24 and n==1
        mask=rows==heads==1 and refs[0]['dims']==(64,1,1,1) and all(r['dims']==(1,1,1,1) for r in refs[1:3])
        kv=refs[0]['dims']==(1,rows,1,1) and refs[1]['dims']==(64,1,heads,1) and refs[2]['dims']==out['dims']
        assert mask or kv,'Only observed KV/mask broadcast contracts are supported'
        spans=[_dense_float(r,tile,elf,'f16') for r in refs]
        assert all(a%16==0 for a,_ in spans)
        address,size=spans[-1]
        assert all(max(a,address)>=min(a+n,address+size) for a,n in spans[:-1])
        assert all(all(s%16==0 and s//16<=65536 for s in r['strides_bits']) for r in refs)
        contracts.append(dict(tile=tile,memrefs=refs,spans=spans))
    code=kernel_text(kernel);assert 0<len(code)<=extent;start=text['offset']+offset
    updated=bytearray(blob);updated[start:start+len(code)]=code;updated=bytes(updated)
    assert len(updated)==len(blob) and updated[:start]==blob[:start] and updated[start+len(code):]==blob[start+len(code):]
    return updated,dict(operation='select_fp16_copy',original_sha256=hashlib.sha256(blob).hexdigest(),patched_sha256=hashlib.sha256(updated).hexdigest(),
        kernel_elf_sha256=hashlib.sha256(kernel).hexdigest(),code_offset=start,code_bytes=len(code),window_bytes=extent,invocations=len(contracts),contracts=contracts)


def candidate(directory,kind):
    import types,sys
    source=(ROOT/'torch_npu_smollm.py').read_text(encoding='utf-8')
    old="hashlib.sha256(xml+data+kernel).digest()";assert source.count(old)==1
    source=source.replace(old,'hashlib.sha256(xml+data+kernel+_SELECT_KERNEL).digest()')
    (directory/'candidate.py').write_text(source,encoding='utf-8')
    module=types.ModuleType('_smol_select_copy');module.__file__=str(ROOT/'torch_npu_smollm.py')
    sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
    manifest=json.loads((OUT/'build.json').read_text(encoding='utf-8'));kernel=(OUT/'select.elf').read_bytes()
    assert sha(Path(__file__).with_name('kv_select.c'))==manifest['source_sha256'] and sha(OUT/'select.elf')==manifest['elf_sha256']
    module._SELECT_KERNEL=kernel
    def compile_same(graph,xml,weights,rms,*,libraries,whole_layer,**options):
        import torch_intel_npu as npu
        key=('smollm_fused_norm',hashlib.sha256(xml+weights+rms).digest())
        assert key in npu._sessions,'Prepare identical baseline graph first'
        base=npu._sessions[key][1]
        assert (base.ir_provenance.vendor_id,base.ir_provenance.device_id,tuple(base.ir_provenance.compiler_version))==(0x8086,0x643e,(8,3))
        blob,report=patch(base.graph_blob,kernel)
        return replace(base,graph_blob=blob,patch_reports=base.patch_reports+(json.dumps(report).encode(),))
    module.compile_fused_rmsnorm=compile_same
    return module


if __name__=='__main__':build()

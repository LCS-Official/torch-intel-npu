"""Reject malformed Select layouts offline, including with Python -O."""
import hashlib
import json
from pathlib import Path
import struct
from npunlock.npu4000 import Elf, inspect, patch_kv_select, SELECT_TEXT_SHA256
from w8_quality_probe import ROOT, sha

OUT=ROOT/'npunlock-probe/research/results/kv-select-integration'


def main():
    fixtures=ROOT/'npunlock/tests/fixtures/npu4000';kernel=(fixtures/'kv_select/kv-select-f16.elf').read_bytes()
    old=ROOT/'npunlock-probe/research/results/kv-group/select-kernel/special-1'
    valid=(old/'b4-s16-w8a16-baseline-l0-decode.blob').read_bytes();checks=[]
    options=dict(vendor_id=0x8086,device_id=0x643e,compiler_version=(8,3))
    def reject(name,blob=valid,elf=kernel,**overrides):
        try:patch_kv_select(blob,elf,**(options|overrides))
        except ValueError as error:checks.append(dict(name=name,rejected=True,error=str(error)))
        else:raise RuntimeError(name+' was accepted')
    for batch,prefix in ((1,32),(4,16)):
        for mode in ('fp16','w8a16'):
            for layer in range(4):
                stem=f'b{batch}-s{prefix}-{mode}';baseline=(old/f'{stem}-baseline-l{layer}-decode.blob').read_bytes()
                blob,report=patch_kv_select(baseline,kernel,**options)
                if blob!=(old/f'{stem}-gather-l{layer}-decode.blob').read_bytes() or report['invocations']!=20:
                    raise RuntimeError('Production patch differs from the verified candidate')
                checks.append(dict(name=f'{stem}-l{layer}',bit_equal_to_candidate=True))
    reject('wrong_vendor',vendor_id=1);reject('wrong_device',device_id=0x7d1d);reject('wrong_compiler',compiler_version=(8,4))
    reject('invalid_kernel',elf=b'broken')
    reject('absent_carrier',blob=(old/'b4-s16-w8a16-baseline-l0-decode.blob').read_bytes()[:64])
    _,valid_report=patch_kv_select(valid,kernel,**options)
    image=Elf(valid);evidence=inspect(valid,memref_count=4,kernel_sha256=SELECT_TEXT_SHA256)
    kv=next(t for t in evidence['invocations'] if t['memrefs'][-1]['elements']>64)
    mask=next(t for t in evidence['invocations'] if t['memrefs'][-1]['elements']==64)
    params=image.named['shave.params']['offset'];out=kv['memrefs'][-1]
    def data_offset(ref):return params+ref['offset']
    def dim_offset(ref):
        r=image.relocation('shave.params',ref['offset']+16);return params+r['symbol_value']+r['addend']
    def stride_offset(ref):
        r=image.relocation('shave.params',ref['offset']+20);return params+r['symbol_value']+r['addend']
    changes=[('precision_lowered',data_offset(out)+24,'<I',1),('outside_cmx',data_offset(out),'<Q',0x40000000),
        ('output_alias',data_offset(out),'<Q',int(kv['memrefs'][2]['unrelocated_data'],16)),
        ('unaligned_vector',data_offset(out),'<Q',int(out['unrelocated_data'],16)+2),
        ('wrong_width',dim_offset(out),'<i',32),('wrong_head_bound',dim_offset(out)+8,'<i',25),
        ('wrong_order',data_offset(out)+28,'<Q',0x2431),('wrong_rank',data_offset(out)+12,'<I',3),
        ('nonstatic',data_offset(out)+8,'<I',0),('condition_stride',stride_offset(kv['memrefs'][0])+8,'<q',0),
        ('current_head_stride',stride_offset(kv['memrefs'][1])+16,'<q',0),
        ('mask_stride',stride_offset(mask['memrefs'][0]),'<q',32),
        ('bad_range_index',image.named[kv['section']]['offset']+kv['index']*96+88,'<I',999),
        ('carrier_fingerprint',valid_report['code_offset'],'<B',valid[valid_report['code_offset']]^1)]
    for name,offset,fmt,value in changes:
        damaged=bytearray(valid);struct.pack_into(fmt,damaged,offset,value);reject(name,blob=bytes(damaged))
    result=dict(status='passed',optimized_python=not __debug__,checks=checks,source_sha256=sha(Path(__file__)),
        production_sha256=sha(ROOT/'npunlock/python/npunlock/npu4000.py'),kernel_sha256=hashlib.sha256(kernel).hexdigest())
    (OUT/('guards-optimized.json' if not __debug__ else 'guards.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('passed',len(checks),'checks, optimized Python:',not __debug__)


if __name__=='__main__':
    main()
    if __debug__:
        import subprocess,sys
        subprocess.run([sys.executable,'-O',__file__],check=True)

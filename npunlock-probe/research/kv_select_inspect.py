"""Read four-MemRef Select candidates without launching patched code."""
import hashlib
import json
from npunlock.npu4000 import Elf, inspect
from w8_quality_probe import ROOT, sha


def main():
    records=[]
    for batch,prefix in ((1,32),(4,16)):
        path=ROOT/f'npunlock-probe/research/results/kv-gather/gather-0/b{batch}-s{prefix}-fp16-baseline-l0-decode.blob'
        data=path.read_bytes();elf=Elf(data);info=inspect(data)
        ranges={}
        for r in info['ranges']:
            relocation=r.get('text_relocation',{})
            if relocation.get('symbol_section')!='shave.text':continue
            offset=relocation['symbol_value']+relocation['addend'];code=elf.content(elf.named['shave.text'])[offset:offset+r['code_size']]
            ranges[(r['section'],r['index'])]=dict(code_offset=elf.named['shave.text']['offset']+offset,code_size=r['code_size'],sha256=hashlib.sha256(code).hexdigest())
        candidates=[]
        for task in info['invocations']:
            relocation=task.get('params_relocation')
            if relocation is None:continue
            base=relocation['symbol_value']+relocation['addend']
            try:refs=[elf.memref(base+40*i) for i in range(4)]
            except (ValueError,KeyError):continue
            if [r['dtype'] for r in refs[1:]]!=[2,2,2]:continue
            key=(task['section'].replace('invocation','range'),task['range_index'])
            candidates.append(dict(section=task['section'],index=task['index'],tile=task['tile'],code=ranges[key],memrefs=refs))
        records.append(dict(blob=str(path.relative_to(ROOT)),blob_sha256=sha(path),four_memref_candidates=candidates))
        print('batch',batch,'potential Select invocations',len(candidates),flush=True)
        if candidates:print(json.dumps(candidates[0],indent=2),flush=True)
    dest=ROOT/'npunlock-probe/research/results/kv-group/select-layout.json'
    dest.write_text(json.dumps(dict(inspector_sha256=sha(__import__('pathlib').Path(__file__)),records=records),indent=2),encoding='utf-8')


if __name__=='__main__':main()

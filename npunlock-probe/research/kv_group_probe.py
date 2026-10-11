"""Decode GQA with three query rows per KV head; reuse frozen controls."""
import json
from pathlib import Path
import sys
import types
import numpy as np
import kv_gather_probe as control
from w8_quality_probe import ROOT, sha

OUT=ROOT/'npunlock-probe/research/results/kv-group'
LAYOUT='group'


def candidate(directory,kind):
    source=(ROOT/'torch_npu_smollm.py').read_text(encoding='utf-8')
    if LAYOUT=='joint':
        old="""            k_cache=op('Select',selected,k_cache,past_k,shape=(1,batch*3,64,64) if folded else (b,3,64,64),auto_broadcast='numpy')
            v_cache=op('Select',selected,v_cache,past_v,shape=(1,batch*3,64,64) if folded else (b,3,64,64),auto_broadcast='numpy')"""
        new="""            channels=past_k.shape[1]
            together=op('Concat',k_cache,v_cache,shape=(past_k.shape[0],2*channels,1,64),axis=1)
            previous=op('Concat',past_k,past_v,shape=(past_k.shape[0],2*channels,64,64),axis=1)
            updated=op('Select',selected,together,previous,shape=previous.shape,auto_broadcast='numpy')
            k_cache=slice_(updated,1,0,channels);v_cache=slice_(updated,1,channels,2*channels)"""
        assert source.count(old)==1;source=source.replace(old,new)
        return load(source,directory)
    old="        k=repeat(k_cache);v=repeat(v_cache);keys=64 if kv_cache=='decode' else length"
    new="""        if kv_cache=='decode':
            # Three consecutive query heads share one KV head. Treat them as
            # three matrix rows rather than replicating the 64-slot KV tensor.
            q=reshape(q,(1,batch*3,3,64))
            k=reshape(k_cache,(1,batch*3,64,64));v=reshape(v_cache,(1,batch*3,64,64))
        else:k=repeat(k_cache);v=repeat(v_cache)
        keys=64 if kv_cache=='decode' else length"""
    assert source.count(old)==1;source=source.replace(old,new)
    old='(batch,q.shape[1],length,keys)';assert source.count(old)==1
    source=source.replace(old,'(batch,q.shape[1],q.shape[2],keys)')
    old='(batch,q.shape[1],length,64)';assert source.count(old)==1
    source=source.replace(old,'(batch,q.shape[1],q.shape[2],64)')
    old='        attended=attention(q,k,v,1 if folded else b)';assert source.count(old)==1
    source=source.replace(old,old+"\n        if kv_cache=='decode':attended=reshape(attended,(1,batch*9,1,64) if folded else (b,9,1,64))")
    return load(source,directory)


def load(source,directory):
    (directory/'candidate.py').write_text(source,encoding='utf-8')
    module=types.ModuleType('_smol_kv_group');module.__file__=str(ROOT/'torch_npu_smollm.py')
    sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module


def main():
    global LAYOUT
    if '--layout' in sys.argv:
        i=sys.argv.index('--layout');LAYOUT=sys.argv[i+1];del sys.argv[i:i+2]
    assert LAYOUT in ('group','joint')
    control.OUT=OUT;control.candidate=candidate
    previous=control.arrays;recent=[]
    def remember(result):
        values=previous(result);recent.append(values)
        if len(recent)>2:recent.pop(0)
        return values
    control.arrays=remember
    code=control.main()
    label=sys.argv[sys.argv.index('--label')+1] if '--label' in sys.argv else 'process0'
    directory=OUT/label;path=directory/'probe.json';report=json.loads(path.read_text(encoding='utf-8'))
    report.update(implementation='grouped_gqa' if LAYOUT=='group' else 'joint_select',wrapper_sha256=sha(Path(__file__)))
    if code and recent:
        failed=directory/'failure-outputs.npz'
        np.savez(failed,**{f'{j}/{i}':value for j,values in enumerate(recent) for i,value in enumerate(values)})
        report['failure_outputs_sha256']=sha(failed)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8');return code


if __name__=='__main__':raise SystemExit(main())

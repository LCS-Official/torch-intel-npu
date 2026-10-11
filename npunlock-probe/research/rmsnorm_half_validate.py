"""Additional model/generation controls using the frozen half-slow candidate."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import weakref

from rmsnorm_half_slow import OUT
from w8_quality_probe import ROOT,sha


def generation(mode):
    import torch_intel_npu as npu
    import torch_npu_smollm as smol
    from npunlock.npu4000 import _patch_carrier,patch_kv_select
    build=json.loads((OUT/'build.json').read_text())
    elf={name:(OUT/(name+'.elf')).read_bytes() for name in ('baseline','codec')}
    for name,data in elf.items():assert hashlib.sha256(data).hexdigest()==build['kernels'][name]['elf_sha256']
    source=ROOT/'npunlock-probe/research/smollm_generation_profile.py'
    text=source.read_text()
    old='            actual_group=None if baseline_prepare else group'
    assert text.count(old)==1
    text=text.replace(old,'            actual_group=None\n            select_kernel(group)')
    old="    report['comparison']='output preparation, both group4' if baseline_prepare else 'group1 vs default group4'"
    assert text.count(old)==1
    text=text.replace(old,"    report['comparison']='RMSNorm baseline vs SIMD slow rounding, both group4'")
    directory=OUT/'generation';directory.mkdir(exist_ok=True)
    runner=directory/'runner.py';runner.write_text(text,encoding='utf-8')
    state={};graphs=[]
    original_construct=smol.FrozenSmolStack.from_checkpoint.__func__
    original_kernel=smol._rmsnorm_kernel
    def construct(cls,*args,**options):
        stack=original_construct(cls,*args,**options);assert len(stack.layers)==30
        state['stack']=weakref.ref(stack);return stack
    def select(group):
        stack=state['stack']();assert stack is not None
        if group==1 and 'keys' not in state:return
        if 'keys' not in state:
            state['keys']={name:[] for name in elf}
            for i,layer in enumerate(stack.layers):
                state['keys']['baseline'].append(dict(layer._keys));keys={}
                assert len(layer._keys)==2
                for local,key in layer._keys.items():
                    _,program=npu._sessions[key];info=program.ir_provenance
                    options=dict(vendor_id=info.vendor_id,device_id=info.device_id,compiler_version=info.compiler_version)
                    blob,patch=_patch_carrier(info.graph_blob,elf['codec'],dtype='f16',carrier='Power',input_count=2,select_kernel=True,**options)
                    patches=[patch]
                    if local[-1]=='decode':
                        blob,selection=patch_kv_select(blob,(ROOT/'npunlock/tests/fixtures/npu4000/kv_select/kv-select-f16.elf').read_bytes(),**options)
                        patches.append(selection)
                    previous=json.loads(program.patch_reports[0]);a=previous['code_offset'];b=a+previous['window_bytes']
                    assert blob[:a]+blob[b:]==program.graph_blob[:a]+program.graph_blob[b:]
                    candidate=replace(program,graph_blob=blob,patch_reports=tuple(json.dumps(p).encode() for p in patches))
                    new_key=('half-slow-generation',mode,i,local[-1]);keys[local]=new_key
                    npu._sessions[new_key]=npu._native.create_inference_session(blob,parent=npu._root_session),candidate;npu._session_creations+=1
                    graphs.append(dict(layer=i,operation=local[-1],baseline_blob_sha256=hashlib.sha256(program.graph_blob).hexdigest(),
                        codec_blob_sha256=hashlib.sha256(blob).hexdigest(),layout_sha256=hashlib.sha256(blob[:a]+bytes(b-a)+blob[b:]).hexdigest(),patch=patch))
                state['keys']['codec'].append(keys)
        for layer,keys in zip(stack.layers,state['keys']['baseline' if group==1 else 'codec']):layer._keys=keys
    smol.FrozenSmolStack.from_checkpoint=classmethod(construct);smol._rmsnorm_kernel=lambda:elf['baseline']
    namespace=dict(__name__='half_slow_generation_runner',__file__=str(runner),select_kernel=select)
    try:
        exec(compile(text,str(runner),'exec'),namespace)
        namespace['main'](mode,output=directory)
    finally:smol.FrozenSmolStack.from_checkpoint=classmethod(original_construct);smol._rmsnorm_kernel=original_kernel
    path=directory/(mode+'.json');report=json.loads(path.read_text())
    report.update(template_sha256=sha(source),wrapper_sha256=sha(Path(__file__)),candidate_build_sha256=sha(OUT/'build.json'),graphs=graphs)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('model','generation'),required=True)
    parser.add_argument('--batch',type=int,choices=(1,4,8),default=1);parser.add_argument('--mode',choices=('fp16','w8a16'),default='fp16')
    parser.add_argument('--prefix',type=int,choices=(24,63),default=24)
    args=parser.parse_args()
    if args.case=='generation':generation(args.mode)
    else:
        import rmsnorm_row4 as row4
        row4.OUT=OUT;row4.model(args.batch,args.prefix,f'model-b{args.batch}-s{args.prefix}',save_graph_blobs=False)

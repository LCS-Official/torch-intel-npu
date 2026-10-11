"""Isolated row-interleaved FP32 Neumaier RMSNorm; same per-row operation order."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from shave_fp32_pipeline import OUT as PARENT,metadata,snapshots,final
from w8_quality_probe import ROOT,MODEL,sha

OUT=PARENT/'row4'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def build():
    import probe_movi_targets as compiler
    from npunlock.npu4000 import kernel_text
    compiler.OUT=OUT/'worker';compiler.OUT.mkdir(parents=True,exist_ok=True)
    fixtures=ROOT/'npunlock/tests/fixtures/npu4000';original=(fixtures/'rmsnorm.c').read_text(encoding='utf-8')
    marker='    for (unsigned base = 0; base < n; base += 576) {';assert original.count(marker)==1
    output='''        for (unsigned j=0;j<576;j+=4) {
            unsigned i=base+j;
            HalfBits4 rounded=encode4(decode4(x+i)*inverse);
            *(HalfBits4*)(out+i)=encode4(decode_bits4(rounded)*decode4(gamma+i));
        }'''
    assert original.count(output)==1
    # Each SIMD lane owns one independent row; K-order is never reassociated.
    replacement='''    unsigned base=0;
    for (; base+4u*576u<=n; base+=4u*576u) {
        Float4 total=(Float4){0,0,0,0},correction=(Float4){0,0,0,0};
        for (unsigned j=0;j<576;++j) {
            HalfBits4 raw=(HalfBits4){x[base+j],x[base+576+j],x[base+1152+j],x[base+1728+j]};
            Float4 value=decode_bits4(raw),square=value*value,next=total+square;
            UInt4 choose=(UInt4)(total>=square);
            union { UInt4 u;Float4 f; } left,right,error;
            left.f=(total-next)+square;right.f=(square-next)+total;
            error.u=(left.u&choose)|(right.u&~choose);
            correction=correction+error.f;total=next;
        }
        for (unsigned lane=0;lane<4;++lane) {
            float variance=(total[lane]+correction[lane])/576.0f+1e-5f;
            float inverse=1.0f/sqrt32(variance);
            for (unsigned j=0;j<576;j+=4) {
                unsigned i=base+lane*576+j;
                HalfBits4 rounded=encode4(decode4(x+i)*inverse);
                *(HalfBits4*)(out+i)=encode4(decode_bits4(rounded)*decode4(gamma+i));
            }
        }
    }
    for (; base < n; base += 576) {'''
    candidate=original.replace(marker,replacement).encode()
    # The codec is unchanged; retain the existing all-half round-trip check.
    roundtrip=original[:original.index('        float total =')].encode()+b'''        for (unsigned j=0;j<576;j+=4) {
            unsigned i=base+j;*(HalfBits4*)(out+i)=encode4(decode4(x+i));
        }
    }
}
'''
    baseline=(fixtures/'rmsnorm/rmsnorm-f16.elf').read_bytes();(OUT/'baseline.elf').write_bytes(baseline)
    (OUT/'baseline.c').write_text(original,encoding='utf-8');shutil.copyfile(fixtures/'rmsnorm/rmsnorm-f16.s',OUT/'baseline.s')
    helpers=b'\n'.join((fixtures/name).read_bytes().split(b'void controlled_act(unsigned p) {',1)[0] for name in ('arithmetic.c','resident_activation.c'))
    prefix=b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n'
    manifest=dict(status='building',builder_sha256=sha(Path(__file__)),production=metadata(),policy='SIMD lanes across four rows; unchanged column order, FP32 compensation and FP16 rounding',
        kernels={'baseline':dict(elf_sha256=hashlib.sha256(baseline).hexdigest(),text_bytes=len(kernel_text(baseline)))})
    for name,source in (('codec',candidate),('roundtrip',roundtrip)):
        generated=prefix+helpers+b'\n'+source;target='rmsnorm-row4-'+name
        args=['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2','-ffp-contract=off','-x','c','entry.c','-o','-']
        asm,ok=compiler.stage('compile',target,'moviCompile64.dll',1,args,generated);assert ok,(name,'compile')
        obj,ok=compiler.stage('assemble',target,'moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm);assert ok,(name,'assemble')
        elf,ok=compiler.stage('link',target,'moviLLD64.dll',3,['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
            (obj,(ROOT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()));assert ok and len(kernel_text(elf))<=21216
        for suffix,data in (('.c',source),('.s',asm),('.elf',elf)):(OUT/(name+suffix)).write_bytes(data)
        manifest['kernels'][name]=dict(source_sha256=hashlib.sha256(source).hexdigest(),generated_source_sha256=hashlib.sha256(generated).hexdigest(),
            elf_sha256=hashlib.sha256(elf).hexdigest(),assembly_sha256=hashlib.sha256(asm).hexdigest(),text_bytes=len(kernel_text(elf)),compile_arguments=args)
    manifest['status']='built';(OUT/'build.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print({name:r['text_bytes'] for name,r in manifest['kernels'].items()},flush=True)


def numeric(label):
    import rmsnorm_reduce_probe as probe
    probe.OUT=OUT
    original=probe.numeric
    def check(npu,torch,report,rounds):
        original(npu,torch,report,rounds)
        # The old three-row special cases take the unchanged tail path.
        # Exercise exceptional values in the new four-row loop as well.
        values=torch.zeros(8,576,dtype=torch.float16)
        values[0,23]=float('inf');values[1,23]=float('nan');values[2].fill_(-0.)
        values[3].fill_(2**-24);values[4].fill_(65504.);values[4,1::2]*=-1
        values[5,23]=-float('inf');values[6].fill_(1.);values[7,23]=float('nan')
        for rows in (4,7,8,16,17,31,32,256,257,511,512):
            host=values.repeat((rows+7)//8,1)[:rows].contiguous();x=host.to('intel_npu')
            weight=torch.ones_like(host);weight[:,1::2]*=-1;gamma=weight.to('intel_npu');baseline=None
            for name in probe.NAMES:
                actual=npu._run(('rmsnorm_reduce_probe',rows,name),[x,gamma],(rows,576),torch.float16).cpu()
                if baseline is None:baseline=actual
                equal=(actual.view(torch.int16)==baseline.view(torch.int16))|(actual.isnan()&baseline.isnan())
                different=int((~equal).sum());assert different==0,(rows,name,different)
                report['checks'].append(dict(tag='row4-mixed-specials',kernel=name,rows=rows,bit_differences=different))
    probe.numeric=check
    try:return probe.execute('numeric',21,label)
    finally:probe.numeric=original


def model(batch,prefix,label,save_graph_blobs=True):
    import torch
    import torch_intel_npu as npu
    import torch_npu_smollm as smol
    from torch_npu_smollm import FrozenSmolStack
    from npunlock.npu4000 import _patch_carrier,patch_kv_select
    from transformers import AutoTokenizer
    from safetensors import safe_open
    from rmsnorm_reduce_probe import times,kernels
    import rmsnorm_reduce_probe as probe
    probe.OUT=OUT;build_info,elf=kernels();torch.set_num_threads(8)
    report=dict(status='running',label=label,batch=batch,prefix=prefix,decoder_layers=30,build=build_info,experiment=metadata(),
        wrapper_sha256=sha(Path(__file__)),timings=[],graphs=[],checks=[])
    saved={};tag=f'b{batch}-s{prefix}';tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    ids=tokenizer('Once upon a time, a curious child asked how birds fly.',return_tensors='pt').input_ids
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as file:embedding=file.get_tensor('model.embed_tokens.weight')
    def work():
        tokens=ids.repeat(1,(prefix+ids.shape[1])//ids.shape[1])[:,:prefix+1].repeat(batch,1)
        host=embedding[tokens].half().contiguous();x=host[:,:prefix].contiguous().to('intel_npu');one=host[:,prefix:].contiguous().to('intel_npu')
        stack=FrozenSmolStack.from_checkpoint(MODEL,num_layers=30,mode='fp16')
        print(label,tag,'preparing same-model controls',flush=True)
        initial=stack.prefill(x);seed=initial[1];decoded=stack.decode(one,seed)
        anchors=[v.cpu().numpy().copy() for result in (initial,decoded) for v in snapshots(result)]
        for i,array in enumerate(anchors):saved[f'baseline/{i}']=array
        del initial,decoded
        strategies={name:[{} for _ in stack.layers] for name in ('baseline','codec')}
        for i,layer in enumerate(stack.layers):
            strategies['baseline'][i]=dict(layer._keys)
            for local,key in layer._keys.items():
                session,program=npu._sessions[key];info=program.ir_provenance
                options=dict(vendor_id=info.vendor_id,device_id=info.device_id,compiler_version=info.compiler_version)
                blob,patch=_patch_carrier(info.graph_blob,elf['codec'],dtype='f16',carrier='Power',input_count=2,select_kernel=True,**options)
                patches=[patch]
                if local[-1]=='decode':
                    blob,selection=patch_kv_select(blob,(ROOT/'npunlock/tests/fixtures/npu4000/kv_select/kv-select-f16.elf').read_bytes(),**options)
                    patches.append(selection)
                previous=[json.loads(p) for p in program.patch_reports];start=previous[0]['code_offset'];end=start+previous[0]['window_bytes']
                assert blob[:start]+blob[end:]==program.graph_blob[:start]+program.graph_blob[end:]
                candidate=replace(program,graph_blob=blob,patch_reports=tuple(json.dumps(p).encode() for p in patches))
                new_key=('rmsnorm_row4',tag,i,local[-1]);strategies['codec'][i][local]=new_key
                npu._sessions[new_key]=npu._native.create_inference_session(blob,parent=npu._root_session),candidate;npu._session_creations+=1
                graph_tag=f'{label}-l{i}-{local[-1]}'
                if save_graph_blobs:
                    for name,data in (('baseline',program.graph_blob),('codec',blob)):(OUT/(graph_tag+'-'+name+'.blob')).write_bytes(data)
                report['graphs'].append(dict(tag=graph_tag,layer=i,operation=local[-1],baseline=previous,codec=patches,
                    baseline_blob_sha256=hashlib.sha256(program.graph_blob).hexdigest(),codec_blob_sha256=hashlib.sha256(blob).hexdigest(),
                    layout_sha256=hashlib.sha256(blob[:start]+bytes(end-start)+blob[end:]).hexdigest()))
        def select(name):
            for i,layer in enumerate(stack.layers):layer._keys=strategies[name][i]
        select('codec');initial=stack.prefill(x);decoded=stack.decode(one,seed)
        values=[v for result in (initial,decoded) for v in snapshots(result)]
        for i,(actual,expected) in enumerate(zip(values,anchors)):
            array=actual.cpu().numpy().copy();np.testing.assert_array_equal(array.view(np.uint8),expected.view(np.uint8));saved[f'codec/{i}']=array
        report['checks'].append(dict(bit_equal_arrays=len(values)));del initial,decoded,values,actual
        for operation in ('prefill','decode'):
            def run(name):select(name);return stack.prefill(x) if operation=='prefill' else stack.decode(one,seed)
            timing=times({name:(lambda name=name:run(name)) for name in strategies},31,npu)
            assert timing['warm_delta']['npu_calls']==31*4*30
            report['timings'].append(dict(operation=operation,**timing));print(label,operation,timing['median_ms'],timing['positive_rounds'],flush=True)
        for i,cache in enumerate(seed):
            for j,value in enumerate((cache.keys,cache.values,cache._position)):
                array=value.cpu().numpy().copy();np.testing.assert_array_equal(array.view(np.uint8),anchors[1+i*3+j].view(np.uint8));saved[f'preserved/{i*3+j}']=array
        report['immutable_seed_arrays']=90
    current_kernel=smol._rmsnorm_kernel
    smol._rmsnorm_kernel=lambda:elf['baseline']
    try:
        with torch.no_grad():work();final(npu,report)
    finally:smol._rmsnorm_kernel=current_kernel
    path=OUT/(label+'-outputs.npz');np.savez_compressed(path,**saved)
    report.update(status='passed',outputs_sha256=sha(path));(OUT/(label+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('build','numeric','model'),required=True)
    parser.add_argument('--label',default='probe');parser.add_argument('--batch',type=int,default=1);parser.add_argument('--prefix',type=int,default=32);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.case=='build':build();return 0
    if args.case=='numeric':return numeric(args.label)
    model(args.batch,args.prefix,args.label);return 0


if __name__=='__main__':raise SystemExit(main())

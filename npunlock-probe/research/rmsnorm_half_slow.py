"""Keep the normal encode4 path; vectorize exact exceptional-lane rounding."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from w8_quality_probe import ROOT,sha
import rmsnorm_single_row as single

OUT=single.OUT/'half-slow'


def build():
    original=(single.FIXTURES/'rmsnorm.c').read_bytes()
    old=b'''    return (HalfBits4){round_half(value[0]),round_half(value[1]),
                      round_half(value[2]),round_half(value[3])};'''
    new=b'''    /* Valid shifts are 14..24; clamp inactive lanes before shifting. */
    UInt4 exponent=magnitude>>23;
    UInt4 small_mask=(UInt4)((exponent-102u)<=10u);
    UInt4 shift=((126u-exponent)&small_mask)|(14u&~small_mask);
    UInt4 mantissa=(magnitude&0x7fffffu)|0x800000u;
    UInt4 q=mantissa>>shift,remainder=mantissa&(((UInt4)1u<<shift)-1u);
    UInt4 halfway=(UInt4)1u<<(shift-1u);
    q=q+(((UInt4)(remainder>halfway)|((UInt4)(remainder==halfway)&(0u-(q&1u))))&1u);
    UInt4 normal_mask=(UInt4)(magnitude>=0x38800000u)&(UInt4)(magnitude<0x47800000u);
    UInt4 normal=(magnitude+0xfffu+((magnitude>>13)&1u)-0x38000000u)>>13;
    UInt4 overflow_mask=(UInt4)(magnitude>=0x47800000u);
    UInt4 nan_mask=(UInt4)(magnitude>0x7f800000u);
    UInt4 result=(normal&normal_mask)|(q&small_mask)|(0x7c00u&overflow_mask)
        |(((magnitude>>13)|0x200u)&0x3ffu&nan_mask)|((bits.u>>16)&0x8000u);
    return __builtin_convertvector(result,HalfBits4);'''
    assert original.count(old)==1
    source=original.replace(old,new)
    OUT.mkdir(parents=True,exist_ok=True);single.OUT=OUT
    manifest=dict(status='building',builder_sha256=sha(Path(__file__)),compiler_helper_sha256=sha(Path(single.__file__)),
        policy='Normal encode4 unchanged; exact integer SIMD subnormal/overflow/NaN fallback',kernels={})
    prior=json.loads((ROOT/'npunlock-probe/research/results/decoder-native/duplicate-rows/build.json').read_text())
    for suffix in ('.c','.s','.elf'):
        path=single.FIXTURES/('rmsnorm.c' if suffix=='.c' else 'rmsnorm/rmsnorm-f16'+suffix)
        shutil.copyfile(path,OUT/('baseline'+suffix))
    manifest['kernels']['baseline']=prior['kernels']['baseline']
    assert sha(OUT/'baseline.c')==manifest['kernels']['baseline']['source_sha256']
    manifest['kernels']['codec']=single.compile_kernel('codec',source)
    head=source[:source.index(b'    unsigned base=0;')]
    body=b'    for (unsigned i=0;i<n;i+=4) *(HalfBits4*)(out+i)=encode4(decode4(x+i));\n}\n'
    manifest['kernels']['roundtrip']=single.compile_kernel('roundtrip',head+body)
    encode=b'''    for (unsigned i=0;i<n/2;i+=4)
        *(HalfBits4*)(out+i)=encode4(*(const Float4*)((const unsigned*)x+i));
    for (unsigned i=n/2;i<n;++i) out[i]=0;
}
'''
    for name,text in (('encode-baseline',original),('encode-codec',source)):
        manifest['kernels'][name]=single.compile_kernel(name,text[:text.index(b'    unsigned base=0;')]+encode)
    manifest['status']='built';(OUT/'build.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    shutil.copyfile(Path(__file__),OUT/'builder.py')


def encodings():
    import gc
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from npunlock.npu4000 import compile_rmsnorm,patch_binary_carrier
    from dataclasses import replace
    from shave_fp32_pipeline import final
    build=json.loads((OUT/'build.json').read_text());torch.set_num_threads(8)
    report=dict(status='running',checks=[],build_sha256=sha(OUT/'build.json'),source_sha256=sha(Path(__file__)))
    saved={}
    def work():
        elf={name:(OUT/(name+'.elf')).read_bytes() for name in ('encode-baseline','encode-codec')}
        for name,data in elf.items():assert hashlib.sha256(data).hexdigest()==build['kernels'][name]['elf_sha256']
        original=compile_rmsnorm(1,(OUT/'baseline.elf').read_bytes(),libraries=npu._native);info=original.ir_provenance
        layout=None
        for name,data in elf.items():
            blob,patch=patch_binary_carrier(info.graph_blob,data,vendor_id=info.vendor_id,device_id=info.device_id,compiler_version=info.compiler_version,dtype='f16')
            assert all(c['elements']==576 for c in patch['contracts'])
            a=patch['code_offset'];b=a+patch['window_bytes'];digest=hashlib.sha256(blob[:a]+bytes(b-a)+blob[b:]).hexdigest()
            if layout is None:layout=digest
            assert digest==layout
            npu._sessions[name]=npu._native.create_inference_session(blob,parent=npu._root_session),replace(original,graph_blob=blob)
            npu._session_creations+=1
        gamma=torch.ones((1,576),dtype=torch.float16,device='intel_npu')
        half=np.arange(0x7c00,dtype=np.uint16).view(np.float16).astype(np.float32)
        mid=(half[:-1]+half[1:])*np.float32(.5)
        boundary=np.concatenate([np.nextafter(mid,-np.inf,dtype=np.float32),mid,np.nextafter(mid,np.inf,dtype=np.float32)])
        special=np.array([0.,-0.,65504.,65520.,65519.996,65520.004,65536.,2**-25,2**-24,np.inf,-np.inf,np.nan],dtype=np.float32)
        random=np.random.default_rng(92311).integers(0,2**32,65536,dtype=np.uint32).view(np.float32)
        values=np.concatenate([half,-half,boundary,-boundary,special,random]).astype(np.float32)
        with np.errstate(over='ignore',invalid='ignore'):expected=values.astype(np.float16)
        padded=np.pad(values,(0,(-len(values))%288));collected={name:[] for name in elf}
        for start in range(0,len(padded),288):
            words=padded[start:start+288].view(np.float16).copy().reshape(1,576)
            x=torch.from_numpy(words).to('intel_npu')
            for name in elf:
                got=npu._run(name,[x,gamma],(1,576),torch.float16).cpu().numpy().reshape(-1)
                assert np.all(got[288:].view(np.uint16)==0)
                collected[name].append(got[:288].copy())
        saved['input']=values;saved['reference']=expected
        for name,chunks in collected.items():
            got=np.concatenate(chunks)[:len(values)];saved[name]=got
            equal=(got.view(np.uint16)==expected.view(np.uint16))|(np.isnan(got)&np.isnan(expected))
            assert equal.all(),(name,int((~equal).sum()))
            report['checks'].append(dict(kernel=name,values=len(values),finite_and_signed_zero_bit_equal=True,nan_policy='NaN-equivalent'))
        a=saved['encode-baseline'];b=saved['encode-codec']
        assert ((a.view(np.uint16)==b.view(np.uint16))|(np.isnan(a)&np.isnan(b))).all()
        report['layout_sha256']=layout
    with torch.no_grad():work();final(npu,report)
    path=OUT/'encoding-outputs.npz';np.savez_compressed(path,**saved)
    report.update(status='passed',outputs_sha256=sha(path));(OUT/'encodings.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report['checks'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('build','encodings','numeric','model'),required=True)
    args=parser.parse_args();os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    if args.case=='build':build()
    elif args.case=='encodings':encodings()
    else:
        import rmsnorm_row4 as row4
        row4.OUT=OUT
        if args.case=='numeric':raise SystemExit(row4.numeric('half-slow'))
        row4.model(1,24,'full30',save_graph_blobs=False)

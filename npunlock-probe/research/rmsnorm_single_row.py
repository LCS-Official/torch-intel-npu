"""Current single-row RMSNorm ablations on the same plain/profile carriers."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics as st

from w8_quality_probe import ROOT, MODEL, sha

OUT=ROOT/'npunlock-probe/research/results/rmsnorm-single-row'
FIXTURES=ROOT/'npunlock/tests/fixtures/npu4000'
NAMES=('baseline','fill','copy','codec','square','affine','epilogue','mean','inverse','sqrt')


def compile_kernel(name,source):
    import probe_movi_targets as compiler
    from npunlock.npu4000 import kernel_text
    compiler.OUT=OUT/'worker';compiler.OUT.mkdir(parents=True,exist_ok=True)
    helpers=b'\n'.join((FIXTURES/name).read_bytes().split(b'void controlled_act(unsigned p) {',1)[0]
                       for name in ('arithmetic.c','resident_activation.c'))
    generated=b'#define DTYPE 2\n#define TYPE unsigned short\n#define RSQRT 1\n'+helpers+b'\n'+source
    args=['moviCompile.dll','-cc1','-triple','shave','-target-cpu','4000xx','-S','-O2','-ffp-contract=off','-x','c','entry.c','-o','-']
    asm,ok=compiler.stage('compile',name,'moviCompile64.dll',1,args,generated);assert ok,(name,'compile')
    obj,ok=compiler.stage('assemble',name,'moviAsm64.dll',2,['moviAsm.dll','--cv','4000xx','--noSPrefixing'],asm);assert ok,(name,'assemble')
    elf,ok=compiler.stage('link',name,'moviLLD64.dll',3,
        ['moviLLD.dll','-flavor','gnu','-EL','-e','controlled_act','-z','max-page-size=0x10','--gc-sections'],
        (obj,(ROOT/'npunlock/src/shavecc/shave_kernel.ld').read_bytes()))
    assert ok and len(kernel_text(elf))<=21216
    for suffix,data in (('.c',source),('.s',asm),('.elf',elf)):(OUT/(name+suffix)).write_bytes(data)
    return dict(source_sha256=hashlib.sha256(source).hexdigest(),generated_source_sha256=hashlib.sha256(generated).hexdigest(),
        elf_sha256=sha(OUT/(name+'.elf')),assembly_sha256=sha(OUT/(name+'.s')),text_bytes=len(kernel_text(elf)))


def build():
    from npunlock.npu4000 import kernel_text
    OUT.mkdir(parents=True,exist_ok=True)
    original=(FIXTURES/'rmsnorm.c').read_text()
    head=original[:original.index('    unsigned base=0;')]
    start=original.index('    for (; base < n; base += 576) {')
    inverse=original.index('        float inverse =',start)
    reduction=original[start:inverse].replace('for (; base < n;', 'for (unsigned base=0; base < n;',1)
    output=original[original.index('        for (unsigned j=0;j<576;j+=4) {',inverse):].rsplit('\n}',1)[0]
    bodies={
        'fill':'    for (unsigned i=0;i<n;i+=4) *(HalfBits4*)(out+i)=(HalfBits4){0};\n',
        'copy':'    for (unsigned i=0;i<n;i+=4) *(HalfBits4*)(out+i)=*(const HalfBits4*)(x+i);\n',
        'codec':'    for (unsigned i=0;i<n;i+=4) *(HalfBits4*)(out+i)=encode4(decode4(x+i));\n',
        'square':'    for (unsigned i=0;i<n;i+=4) { Float4 value=decode4(x+i);*(HalfBits4*)(out+i)=encode4(value*value); }\n',
        'affine':'    for (unsigned i=0;i<n;i+=4) *(HalfBits4*)(out+i)=encode4(decode4(x+i)*decode4(gamma+i));\n',
        'epilogue':'    for (unsigned base=0;base<n;base+=576) {\n        float inverse=1.0f+load_half(gamma,base)*0.0f;\n'+output+'\n',
        'mean':reduction+'        unsigned short value=round_half(variance);\n        for (unsigned j=0;j<576;++j) out[base+j]=value;\n    }\n',
        'inverse':reduction+'        unsigned short value=round_half(1.0f/sqrt32(variance));\n        for (unsigned j=0;j<576;++j) out[base+j]=value;\n    }\n',
        'sqrt':'    for (unsigned base=0;base<n;base+=576) {\n        float x0=load_half(x,base);\n        unsigned short value=round_half(1.0f/sqrt32(x0*x0/576.0f+1e-5f));\n        for (unsigned j=0;j<576;++j) out[base+j]=value;\n    }\n'}
    manifest=dict(status='building',builder_sha256=sha(Path(__file__)),sources_sha256={name:sha(FIXTURES/name)
        for name in ('rmsnorm.c','arithmetic.c','resident_activation.c')},kernels={})
    for suffix in ('.c','.s','.elf'):
        source=FIXTURES/('rmsnorm.c' if suffix=='.c' else 'rmsnorm/rmsnorm-f16'+suffix)
        shutil.copyfile(source,OUT/('baseline'+suffix))
    manifest['kernels']['baseline']=dict(source_sha256=sha(OUT/'baseline.c'),assembly_sha256=sha(OUT/'baseline.s'),
        elf_sha256=sha(OUT/'baseline.elf'),text_bytes=len(kernel_text((OUT/'baseline.elf').read_bytes())))
    for name,body in bodies.items():manifest['kernels'][name]=compile_kernel(name,(head+body+'}\n').encode())
    manifest['status']='built';(OUT/'build.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    shutil.copyfile(Path(__file__),OUT/'builder.py')
    print({name:r['text_bytes'] for name,r in manifest['kernels'].items()},flush=True)


def execute():
    import numpy as np
    import torch
    import torch_npu_device_profile as profile
    from npunlock.npu4000 import compile_rmsnorm,patch_binary_carrier
    from rmsnorm_reduce_probe import times
    from shave_fp32_pipeline import final
    build=json.loads((OUT/'build.json').read_text());assert build['status']=='built'
    for name,digest in build['sources_sha256'].items():assert sha(FIXTURES/name)==digest
    elf={name:(OUT/(name+'.elf')).read_bytes() for name in NAMES}
    for name,data in elf.items():assert hashlib.sha256(data).hexdigest()==build['kernels'][name]['elf_sha256']
    prior=ROOT/'npunlock-probe/research/results/decoder-native/profile'
    binary=json.loads((prior/'build.json').read_text());assert sha(prior/'session_task_profile.dll')==binary['dll_sha256']
    profile.OUT=OUT/'runtime';profile.OUT.mkdir(exist_ok=True)
    shutil.copyfile(prior/'session_task_profile.dll',profile.OUT/'session_task_profile.dll')
    npu,driver=profile.setup();profile.OUT=OUT/'tasks';profile.OUT.mkdir(exist_ok=True)
    torch.set_num_threads(8);torch.manual_seed(92311)
    report=dict(status='running',build=build,checks=[],timings=[],profiles=[],carriers=[],source_sha256=sha(Path(__file__)),
        profiling_dll_sha256=binary['dll_sha256'],interpretation='Ablations change outputs and compiler scheduling; time differences are diagnostic estimates, not additive production stages')
    saved={}
    def key(variant,name):return ('single-row-ablation',variant,name)
    def work():
        from torch_npu_smollm import FrozenSmolLayer
        model=FrozenSmolLayer.from_checkpoint(MODEL,mode='fp16')
        gamma_cpu=model._cpu_weight('input_layernorm.weight').view(1,576)
        original_compile=npu._native.compile_ir
        for variant in ('plain','profile'):
            def flagged(xml,weights,**options):
                options['build_flags']=options.get('build_flags','')+' PERF_COUNT="YES"'
                return original_compile(xml,weights,**options)
            if variant=='profile':npu._native.compile_ir=flagged
            try:base=compile_rmsnorm(1,elf['baseline'],libraries=npu._native)
            finally:npu._native.compile_ir=original_compile
            info=base.ir_provenance;layout=None
            for name in NAMES:
                blob,patch=patch_binary_carrier(info.graph_blob,elf[name],vendor_id=info.vendor_id,device_id=info.device_id,
                    compiler_version=info.compiler_version,dtype='f16')
                a=patch['code_offset'];b=a+patch['window_bytes'];digest=hashlib.sha256(blob[:a]+bytes(b-a)+blob[b:]).hexdigest()
                if layout is None:layout=digest
                assert digest==layout and all(c['elements']==576 for c in patch['contracts'])
                session=npu._native.create_inference_session(blob,parent=npu._root_session)
                if variant=='profile':assert driver.diag_profile_enable(session._result.session)==0
                program=replace(base,graph_blob=blob,patch_reports=(json.dumps(patch).encode(),))
                npu._sessions[key(variant,name)]=session,program;npu._session_creations+=1
                report['carriers'].append(dict(variant=variant,kernel=name,layout_sha256=digest,patch=patch))
        real_path=ROOT/'npunlock-probe/research/results/rmsnorm-diagnostic/real-input.npz'
        real=torch.from_numpy(np.load(real_path)['host'].copy()).reshape(256,576)
        report['real_input_sha256']=sha(real_path)
        sub=np.resize(np.array([0,1,3,1023,0x8000,0x8001,0x8003,0x83ff],dtype=np.uint16),576).view(np.float16).copy()
        datasets=[('random',torch.randn(1,576).half()),('real-embedding',real[:1].contiguous()),('subnormal',torch.from_numpy(sub).view(1,576))]
        for dataset,host in datasets:
            x=host.to('intel_npu');gamma=gamma_cpu.to('intel_npu');value=host.float()
            variance=value.square().mean(-1,keepdim=True)+1e-5
            inv=torch.rsqrt(variance)
            refs=dict(fill=torch.zeros_like(host),copy=host,codec=host,square=value.square().half(),affine=host*gamma_cpu,
                epilogue=host*gamma_cpu,mean=variance.half().expand_as(host),inverse=inv.half().expand_as(host),
                sqrt=torch.rsqrt(value[:,:1].square()/576+1e-5).half().expand_as(host),baseline=(value*inv).half()*gamma_cpu)
            plain={}
            for variant in ('plain','profile'):
                for name in NAMES:
                    got=npu._run(key(variant,name),[x,gamma],(1,576),torch.float16).cpu()
                    torch.testing.assert_close(got,refs[name],rtol=.002,atol=.001)
                    if variant=='plain':plain[name]=got.numpy().copy()
                    else:np.testing.assert_array_equal(got.numpy().view(np.uint8),plain[name].view(np.uint8))
                    saved[f'{dataset}/{variant}/{name}']=got.numpy().copy()
                    report['checks'].append(dict(dataset=dataset,variant=variant,kernel=name,status='passed'))
            outputs={name:torch.empty((1,576),dtype=torch.float16,device='intel_npu') for name in NAMES}
            run={name:(lambda name=name:npu._run(key('plain',name),[x,gamma],(1,576),torch.float16,out=outputs[name])) for name in NAMES}
            timing=times(run,31,npu);assert timing['warm_delta']['driver_allocations']==0
            report['timings'].append(dict(dataset=dataset,**timing));print(dataset,timing['median_ms'],flush=True)
            records={name:[] for name in NAMES}
            for i in range(3):
                names=NAMES[i:]+NAMES[:i]
                for name in names:
                    session=npu._sessions[key('profile',name)][0]
                    assert driver.diag_profile_refresh(session._result.session)==0
                    got=npu._run(key('profile',name),[x,gamma],(1,576),torch.float16)
                    record=profile.read_tasks(driver,session,f'{dataset}-{name}-{i}');records[name].append(record)
            for name,samples in records.items():
                assert len({s['raw_sha256'] for s in samples})==3
                report['profiles'].append(dict(dataset=dataset,kernel=name,samples=samples,
                    median_shave_union_ms=st.median(s['engines']['shave']['union_ms'] for s in samples)))
            print(dataset,'shave us',{name:round(st.median(s['engines']['shave']['union_ms'] for s in samples)*1000,2) for name,samples in records.items()},flush=True)
    with torch.no_grad():work();final(npu,report)
    path=OUT/'outputs.npz';np.savez_compressed(path,**saved)
    assert all(sha(FIXTURES/name)==digest for name,digest in build['sources_sha256'].items())
    report.update(status='passed',outputs_sha256=sha(path));(OUT/'ablation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=('build','ablate'),required=True)
    args=parser.parse_args();os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    {'build':build,'ablate':execute}[args.case]()

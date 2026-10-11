"""Independent raw-FP16 Select tests on the driver's real CMX layouts."""
import gc
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import torch
import npunlock as s
from kv_select_kernel import OUT, patch, sha


def main():
    import torch_intel_npu as npu
    directory=OUT/'numeric-special';directory.mkdir(parents=True,exist_ok=True)
    kernel=(OUT/'select.elf').read_bytes();saved={}
    report=dict(status='running',source_sha256=sha(Path(__file__)),kernel_sha256=sha(OUT/'select.elf'),graphs=[],checks=[])
    def op(name,*v,shape=None,dtype='f16',**attrs):return s.op(name,*v,_shape=shape or v[0].shape,_dtype=dtype,**attrs)
    def reshape(v,shape,name=None):return op('Reshape',v,s.constant(np.array(shape,dtype=np.int64)),shape=shape,dtype=v.dtype,special_zero=False,_name=name)
    try:
        with torch.no_grad():
            for heads in (3,1,12,16,24):
                shape=(1,heads,64,64);past=s.input('past',shape=shape,dtype='f16')
                current=s.input('current',shape=(1,heads,1,64),dtype='f16');position=s.input('position',shape=(1,),dtype='f32')
                p=op('Convert',position,destination_type='f16');slots=s.constant(np.arange(64,dtype=np.float16))
                cond=reshape(op('Equal',slots,p,shape=(64,),dtype='boolean',auto_broadcast='numpy'),(1,1,64,1))
                value=op('Select',cond,current,past,shape=shape,auto_broadcast='numpy')
                forbidden=op('Greater',slots,p,shape=(64,),dtype='boolean',auto_broadcast='numpy')
                mask=op('Select',forbidden,s.constant(np.float16(-np.inf)),s.constant(np.float16(0)),shape=(64,),auto_broadcast='numpy')
                graph=s.Graph([past,current,position],[reshape(value,shape,'updated'),reshape(mask,(64,),'mask')],name='kv_select_bits')
                ir=s.serialize_ir(graph);compiled=npu._native.compile_ir(ir.xml,ir.weights,build_flags='--config EXECUTION_MODE_HINT="ACCURACY"',timeout_ms=20000)
                assert (compiled.vendor_id,compiled.device_id,tuple(compiled.compiler_version))==(0x8086,0x643e,(8,3))
                blob,evidence=patch(compiled.graph_blob,kernel);assert evidence['invocations']>0
                stem=f'h{heads}';(directory/f'{stem}.xml').write_bytes(ir.xml)
                for name,data in (('baseline',compiled.graph_blob),('copy',blob)):
                    (directory/f'{stem}-{name}.blob').write_bytes(data)
                    program=s.load_native(data,graph=graph,libraries=npu._native)
                    key=('select_numeric',heads,name)
                    npu._sessions[key]=npu._native.create_inference_session(data,parent=npu._root_session),program;npu._session_creations+=1
                report['graphs'].append(dict(heads=heads,xml_sha256=hashlib.sha256(ir.xml).hexdigest(),patch=evidence))
                # For H>=16 every one of the 65,536 FP16 bit patterns occurs.
                bits=np.arange(heads*64*64,dtype=np.uint16).reshape(shape)
                fresh=(np.arange(heads*64,dtype=np.uint32)*73).astype(np.uint16).reshape(1,heads,1,64)
                special=np.array([0,0x8000,1,0x8001,0x3ff,0x83ff,0x400,0x8400,0x7bff,0xfbff,
                                  0x7c00,0xfc00,0x7c01,0xfc01,0x7e00,0xfe00,0x7fff,0xffff],dtype=np.uint16)
                fresh[:,:,:,:len(special)]=special
                saved[f'{stem}/current']=fresh.copy()
                old=torch.from_numpy(bits.view(np.float16).copy()).to('intel_npu')
                new=torch.from_numpy(fresh.view(np.float16).copy()).to('intel_npu')
                np.testing.assert_array_equal(old.cpu().numpy().view(np.uint16),bits)
                np.testing.assert_array_equal(new.cpu().numpy().view(np.uint16),fresh)
                for position in (0,1,7,8,15,31,32,63):
                    pos=torch.tensor([position],dtype=torch.float32,device='intel_npu');expected=bits.copy()
                    expected[:,:,position:position+1,:]=fresh
                    expected_mask=np.where(np.arange(64)>position,np.float16(-np.inf),np.float16(0)).astype(np.float16).view(np.uint16)
                    tag=f'{stem}-p{position}';saved[f'{tag}/expected/0']=expected;saved[f'{tag}/expected/1']=expected_mask
                    for name in ('baseline','copy'):
                        outputs=npu._run_many(('select_numeric',heads,name),[old,new,pos],[(shape,torch.float16),((64,),torch.float16)])
                        values=[v.cpu().numpy().view(np.uint16).copy() for v in outputs]
                        for i,(actual,wanted) in enumerate(zip(values,(expected,expected_mask))):
                            np.testing.assert_array_equal(actual,wanted,err_msg=f'{tag}/{name}/{i}');saved[f'{tag}/{name}/{i}']=actual
                        del outputs,values
                    report['checks'].append(dict(tag=tag,elements=int(expected.size),all_fp16_patterns=heads>=16));del pos
                np.testing.assert_array_equal(old.cpu().numpy().view(np.uint16),bits)
                np.testing.assert_array_equal(new.cpu().numpy().view(np.uint16),fresh)
                del old,new
                print(stem,'8 positions, raw FP16 + mask exact',flush=True)
        gc.collect();npu.shutdown();report['final_stats']=npu.stats()
        assert report['final_stats']['closed'] and not report['final_stats']['context_failed']
        assert all(report['final_stats'][key]==0 for key in ('live_buffers','live_buffer_bytes','cached_buffer_bytes','quarantined_buffer_bytes'))
        path=directory/'outputs.npz';np.savez(path,**saved)
        report.update(status='passed',outputs_sha256=sha(path),output_array_count=len(saved))
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (directory/'numeric.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

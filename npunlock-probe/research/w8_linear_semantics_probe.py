"""Identify W8A16 scale rounding on the failing prompt's real activations."""
import gc
import json
import os
from pathlib import Path
import traceback

from w8_quality_probe import ROOT,OUT,MODEL,sha
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'


def main():
    import numpy as np
    import torch
    import torch_intel_npu as npu
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from torch_npu_smollm import FrozenSmolLayer
    from torch_npu_lowbit import QuantizedLinear,compile_weight,dpu_modes
    from torch_npu_smollm_stack_demo import errors
    torch.set_num_threads(8);report=dict(status='running',source_sha256=sha(Path(__file__)),rows=[],sources={name:sha(ROOT/name)
        for name in ['torch_npu_lowbit.py','torch_npu_smollm.py','npunlock-probe/torch-device/_torch_intel_npu.pyd']})
    arrays={}
    try:
        with torch.no_grad():
            tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
            cpu=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,attn_implementation='eager',local_files_only=True).eval()
            text=tokenizer.apply_chat_template([dict(role='system',content='Tell stories.'),dict(role='user',content='Tell a story about a bird.')],tokenize=False,add_generation_prompt=True)
            ids=tokenizer(text,return_tensors='pt').input_ids
            layer=FrozenSmolLayer.from_checkpoint(MODEL,mode='w8a16')
            _,trace=layer.reference_cpu(cpu.model.embed_tokens(ids))
            for name,q in layer._packed.items():
                key=('norm1' if name.startswith('self_attn.') and not name.startswith('self_attn.o_') else
                    'attention' if name.startswith('self_attn.o_') else 'gated' if name.startswith('mlp.down_') else 'norm2')
                host=trace[key].reshape(-1,q.shape[1]).contiguous();x=host.to('intel_npu')
                codes=np.frombuffer(q.data,dtype=np.int8).reshape(q.shape).astype(np.float64)
                scales=np.frombuffer(q.scales,dtype=np.float16).astype(np.float64)[:,None]
                fused=(host.double()@torch.from_numpy(codes*scales).t()).half()
                rounded=(host.double()@torch.from_numpy(q.dequantize().astype(np.float64)).t()).half()
                quant=QuantizedLinear(q);actual=quant(x).cpu()
                program,evidence=compile_weight(q,host.shape[0],npu._native,reference_fp16=True)
                session=npu._native.create_inference_session(program.graph_blob,timeout_ms=10000,parent=npu._root_session)
                key=('w8_semantics_fp16',name);npu._sessions[key]=session,program;npu._session_creations+=1
                control=npu._run(key,[x],actual.shape,torch.float16).cpu()
                row=dict(name=name,shape=list(host.shape),npu_vs_unrounded_weight=errors(actual,fused),
                    npu_vs_fp16_dequantized_weight=errors(actual,rounded),npu_fp16_control_vs_rounded=errors(control,rounded),
                    npu_quant_vs_fp16_control=errors(actual,control),
                    quant_fp16_control_bit_equal=bool(torch.equal(actual.view(torch.int16),control.view(torch.int16))),
                    fp16_control_evidence=evidence,quant_dpu_modes=dpu_modes(npu._sessions[quant._keys[host.shape[0]]][1].graph_blob))
                report['rows'].append(row);print(name,row['npu_vs_unrounded_weight']['relative_l2'],row['npu_vs_fp16_dequantized_weight']['relative_l2'],flush=True)
                for label,value in [('input',host),('npu',actual),('npu-fp16-control',control),('cpu-unrounded',fused),('cpu-rounded',rounded)]:arrays[f'{name}/{label}']=value.numpy().copy()
                del x,quant
            del layer,cpu;gc.collect();npu.shutdown();report['final_stats']=npu.stats()
            assert not report['final_stats']['context_failed'] and report['final_stats']['live_buffers']==report['final_stats']['cached_buffer_bytes']==0
            path=OUT/'linear-semantics-outputs.npz';np.savez(path,**arrays);report.update(status='passed',outputs_sha256=sha(path))
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc(),failure_stats=npu.stats());print(report['traceback'],flush=True)
    (OUT/'linear-semantics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())

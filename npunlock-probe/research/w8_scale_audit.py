"""S51: diagnose the rejected final layer on one shared actual NPU prefix."""
from pathlib import Path
import hashlib
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import w8_scale_quality as study


def scaled_peak(actual,reference):
    if (actual.device.type!='cpu' or reference.device.type!='cpu' or actual.shape!=reference.shape
        or not actual.numel() or not torch.isfinite(actual).all() or not torch.isfinite(reference).all()):
        raise ValueError('Peak diagnostics require matching nonempty finite CPU outputs')
    rms=reference.float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-5)
    score=(actual.float()-reference.float()).abs()/(reference.float().abs()+rms)
    index=tuple(int(i) for i in np.unravel_index(int(score.argmax()),score.shape))
    return dict(index=index,actual=float(actual[index]),reference=float(reference[index]),
        scaled_error=float(score[index]),row_rms=float(rms[index[:-1]+(0,)]))


def main():
    import torch_intel_npu as npu
    base=study.base;probe=study.previous.study;compare=study.device.compare
    report=dict(status='running',profiles={},audit_source_sha256=base.sha(Path(__file__)),**study.identity())
    def work():
        cpu,_=base.mixed.common.model();case=next(r for r in base.read('prepare')['trajectories'] if r['index']==4)
        source=study.stack_for('ffn');x=cpu.model.embed_tokens(torch.tensor(case['token_ids'])).contiguous().to('intel_npu')
        for layer in source.layers[:29]:x=layer(x)
        shared=x.cpu();shared_hash=hashlib.sha256(shared.numpy().tobytes()).hexdigest();boundary=None
        for profile in ('baseline',*study.PROFILES):
            layer=study.stack_for(profile).layers[29]
            whole=layer(x);residual=layer._run_fused(x,segment='attention')[0]
            normalized=layer._norm_device(residual,'post_attention_layernorm.weight')
            signatures=[hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in (residual,normalized)]
            if boundary is None:boundary=signatures
            assert signatures==boundary
            normal=layer._run_graph(residual,normalized,segment='ffn');graph,xml,_=layer.graph(tuple(x.shape),segment='ffn')
            values,key=probe.run_graph(npu,probe.ffn_probe_graph(graph,xml),[residual,normalized])
            actual={name:value.cpu() for name,value in values.items()}
            reference,trace=layer.reference_cpu(shared);ffn_reference,_=layer.reference_cpu(residual.cpu(),segment='ffn',normalized=normalized.cpu())
            norm1=layer._norm_device(x,'input_layernorm.weight').cpu();host_residual=residual.cpu()
            def norm(value,name,*,fused_gain=False):
                value=value.float();result=value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-5)
                if not fused_gain:result=result.half()
                return (result*layer._cpu_weight(name)).half()
            norm2=norm(host_residual,'post_attention_layernorm.weight')
            attention_from_actual_norm1,_=layer.reference_cpu(shared,segment='attention',normalized=norm1)
            ffn_from_actual_r1,_=layer.reference_cpu(host_residual,segment='ffn',normalized=norm2)
            ffn_from_actual_norm1,_=layer.reference_cpu(attention_from_actual_norm1,segment='ffn',
                normalized=norm(attention_from_actual_norm1,'post_attention_layernorm.weight'))
            checks=dict(split_vs_whole=compare(normal.cpu(),whole.cpu()),exported_vs_normal=compare(actual['output'],normal.cpu()),
                whole_vs_staged_layer=compare(whole.cpu(),reference),normal_vs_staged_ffn=compare(normal.cpu(),ffn_reference),
                norm1_same_input=compare(norm1,trace['norm1']),norm1_fused_gain=compare(norm1,norm(shared,'input_layernorm.weight',fused_gain=True)),
                attention_actual_norm1=compare(host_residual,attention_from_actual_norm1),attention_boundary=compare(host_residual,trace['residual1']),
                norm2_same_input=compare(normalized.cpu(),norm2),
                norm2_fused_gain=compare(normalized.cpu(),norm(host_residual,'post_attention_layernorm.weight',fused_gain=True)))
            for name in ('gate','up','down'):
                inp=normalized.cpu().reshape(-1,576) if name!='down' else actual['gated']
                corrected,local,b=probe.cpu_projection(layer,'mlp.'+name+'_proj.weight',inp)
                for target,observed,expected in (
                    ('main_exact',actual[name+'_main'],local['main']),('main_rounded',actual[name+'_main'],local['rounded']),
                    ('reduced',actual[name+'_reduced'],local['reduced']),
                    ('delta_same_input',actual[name+'_delta'],F.linear(actual[name+'_reduced'].float(),b.float()).half()),
                    ('add_same_input',actual[name+'_corrected'],(actual[name+'_main']+actual[name+'_delta']).half()),
                    ('corrected',actual[name+'_corrected'],corrected)):
                    checks[name+'_'+target]=compare(observed,expected)
            gate=actual['gate_corrected'];up=actual['up_corrected'];silu=actual['silu'].reshape_as(gate)
            checks['silu_same_input']=compare(silu,F.silu(gate.float()).half())
            checks['multiply_same_input']=compare(actual['gated'],(silu*up).half())
            strict=F.silu(values['gate_corrected'].float()).half()*values['up_corrected'];tail=probe.down_tail_graph(graph,xml)
            control,_=probe.run_graph(npu,tail,[residual,values['gated']]);changed,_=probe.run_graph(npu,tail,[residual,strict])
            checks['tail_control']=compare(control['output'].cpu(),actual['output'])
            checks['strict_vs_staged_ffn']=compare(changed['output'].cpu(),ffn_reference)
            checks['strict_vs_staged_layer']=compare(changed['output'].cpu(),reference)
            assert all(checks[k]['exact_fraction']==1 for k in ('split_vs_whole','exported_vs_normal','tail_control'))
            peak=scaled_peak(whole.cpu(),reference)
            assert peak['scaled_error']==checks['whole_vs_staged_layer']['max_scaled_error']
            coordinate=peak['index']
            pieces=dict(norm1=float(ffn_from_actual_norm1[coordinate])-float(reference[coordinate]),
                attention=float(ffn_from_actual_r1[coordinate])-float(ffn_from_actual_norm1[coordinate]),
                norm2=float(ffn_reference[coordinate])-float(ffn_from_actual_r1[coordinate]),
                ffn=float(actual['output'][coordinate])-float(ffn_reference[coordinate]))
            assert sum(pieces.values())==peak['actual']-peak['reference']
            peak['signed_boundary_differences']=pieces
            report['profiles'][profile]=dict(checks=checks,peak=peak,attention_boundary_sha256=signatures,
                dpu_evidence=layer._inspect_blob(npu._sessions[key][1].graph_blob))
            print('shared-prefix final layer',profile,checks['whole_vs_staged_layer'],peak,flush=True)
        assert hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()==shared_hash
        report.update(prefix_profile='ffn',trajectory=4,layer=29,input_sha256=shared_hash,stats=npu.stats())
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return study.device.finish(report,'diagnostic',npu)


if __name__=='__main__':
    study.OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():result=main()
    raise SystemExit(result or 0)

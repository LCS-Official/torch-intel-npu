"""S57 follow-up: distinguish SHAVE activation rounding from the DPU down tail."""
import hashlib
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F
import w8_final_layer_audit as study
from w8_scale_audit import scaled_peak

base=study.base
probe=study.probe
OUT=study.OUT


def down_probe_graph(graph,xml):
    tail,xml,_,_=probe.down_tail_graph(graph,xml)
    chain=probe.linear_chain(tail,'mlp.down_proj.weight')
    values={name:chain[name] for name in ('main','reduced','delta','corrected')}
    values['output']=tail.outputs[-1]
    return probe.serialize_probe(tail.inputs,values,xml,'final_layer_down_stages')


def main(profile):
    if (OUT/('down-'+profile+'.json')).exists():raise FileExistsError('Refusing to overwrite down diagnostic')
    import torch_intel_npu as npu
    report=dict(status='running',profile=profile,variants={},down_source_sha256=base.sha(Path(__file__)),**study.identity())
    def work():
        cpu,_=base.mixed.common.model();row=base.read('fresh')['cases'][0];assert row['tag']=='zipper'
        x=cpu.model.embed_tokens(torch.tensor(row['token_ids'])).contiguous().to('intel_npu')
        stack=study.previous.parent.stack_for(profile)
        for layer in stack.layers[:29]:x=layer(x)
        layer=stack.layers[29];whole=layer(x);reference,_=layer.reference_cpu(x.cpu())
        residual=layer._run_fused(x,segment='attention')[0];host_residual=residual.cpu()
        normalized=layer._norm_device(residual,'post_attention_layernorm.weight')
        graph,xml,_=layer.graph(tuple(x.shape),segment='ffn')
        actual,_=probe.run_graph(npu,probe.ffn_probe_graph(graph,xml),[residual,normalized])
        strict_activation=F.silu(actual['gate_corrected'].float()).half()
        strict_product=(strict_activation*actual['up_corrected']).half()
        cpu_activation=F.silu(actual['gate_corrected'].cpu().float()).half()
        cpu_product=(cpu_activation*actual['up_corrected'].cpu()).half()
        compare=study.device.compare;peak=scaled_peak(whole.cpu(),reference);coordinate=tuple(peak['index']);site=(coordinate[1],coordinate[2])
        report.update(peak=peak,input_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
            activation=compare(strict_activation.cpu(),cpu_activation),product=compare(strict_product.cpu(),cpu_product))
        tail=down_probe_graph(graph,xml)
        for name,gated in (('original',actual['gated']),('shave-product',strict_product),('cpu-product',cpu_product.to('intel_npu'))):
            output,key=probe.run_graph(npu,tail,[residual,gated]);host={k:v.cpu() for k,v in output.items()}
            if name=='original':
                controls={k:compare(host[k],actual['down_'+k].cpu() if k!='output' else whole.cpu()) for k in host}
                assert all(c['exact_fraction']==1 for c in controls.values()),('Down observation changes arithmetic',controls)
                report['controls']=controls
            expected,local,b=probe.cpu_projection(layer,'mlp.down_proj.weight',gated.cpu())
            delta=F.linear(local['reduced'].float(),b.float()).half()
            delta_actual_basis=F.linear(host['reduced'].float(),b.float()).half()
            rounded=(local['rounded']+delta).half();cpu_output=(host_residual+expected.view_as(host_residual)).half()
            p={k:float(t[site]) for k,t in dict(main_npu=host['main'],main_exact=local['main'],main_rounded=local['rounded'],
                delta_npu=host['delta'],delta_cpu=delta,delta_actual_basis=delta_actual_basis,corrected_npu=host['corrected'],
                corrected_exact=expected,corrected_rounded=rounded).items()}
            p.update(residual=float(host_residual[coordinate]),output_npu=float(host['output'][coordinate]),
                output_cpu_same_input=float(cpu_output[coordinate]),full_reference=float(reference[coordinate]))
            checks=dict(main_exact=compare(host['main'],local['main']),main_rounded=compare(host['main'],local['rounded']),
                reduced=compare(host['reduced'],local['reduced']),delta_same_input=compare(host['delta'],delta_actual_basis),
                add_same_input=compare(host['corrected'],(host['main']+host['delta']).half()),
                residual_same_input=compare(host['output'],(host_residual+host['corrected'].view_as(host_residual)).half()))
            report['variants'][name]=dict(site=p,checks=checks,output_vs_full_reference=compare(host['output'],reference),
                output_vs_same_input_cpu=compare(host['output'],cpu_output))
            print('down same input',profile,name,p,flush=True)
        report.update(stats=npu.stats(),diagnostic_sha256=base.sha(OUT/('diagnostic-'+profile+'.json')))
        assert report['input_sha256']==base.read('diagnostic-'+profile)['cases'][0]['ffn']['input_sha256']
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return study.device.finish(report,'down-'+profile,npu)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('baseline','tail-e2-a100'),default='baseline')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():result=main(args.profile)
    raise SystemExit(result or 0)

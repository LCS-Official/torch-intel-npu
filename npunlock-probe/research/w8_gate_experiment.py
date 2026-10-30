"""S50 controlled SHAVE activation/product prototype; no production default."""
import argparse
from pathlib import Path
import time
import traceback

import torch
import torch.nn.functional as F
import w8_gate_quality as study

base=study.base
OUT=study.OUT


def front_graph(graph,xml):
    values={name:study.linear_chain(graph,'mlp.'+name+'_proj.weight')['corrected'] for name in ('gate','up')}
    return study.serialize_probe((graph.inputs[1],),values,xml,'w8_gate_front')


def promotable(baseline,candidate):
    if baseline['positions']!=candidate['positions']:raise ValueError('Different internal position counts')
    return (candidate['quality_passed']>baseline['quality_passed'] and candidate['next_token_equal']>=baseline['next_token_equal']
        and all(candidate[k]<=baseline[k] for k in ('mean_relative_l2','max_relative_l2','max_kl')))


class StrictStack:
    def __init__(self,stack):
        self.layers=stack.layers;self.recipes={};self.audit=[];self.audited=False

    def __call__(self,x):
        import torch_intel_npu as npu
        for i,layer in enumerate(self.layers):
            reference=None;ordinary=None
            if not self.audited:
                reference,_=layer.reference_cpu(x.cpu());ordinary=layer(x)
            residual=layer._run_fused(x,segment='attention')[0]
            normalized=layer._norm_device(residual,'post_attention_layernorm.weight')
            key=(i,tuple(x.shape))
            if key not in self.recipes:
                graph,xml,_=layer.graph(tuple(x.shape),segment='ffn')
                self.recipes[key]=(front_graph(graph,xml),study.down_tail_graph(graph,xml))
            front,tail=self.recipes[key];values,_=study.run_graph(npu,front,[normalized])
            if not self.audited:
                graph,xml,_=layer.graph(tuple(x.shape),segment='ffn')
                traced,_=study.run_graph(npu,study.ffn_probe_graph(graph,xml),[residual,normalized])
                control,_=study.run_graph(npu,tail,[residual,traced['gated']])
                checks={name:study.device.compare(a.cpu(),b.cpu()) for name,a,b in (
                    ('exported_vs_whole',traced['output'],ordinary),('front_gate',values['gate'],traced['gate_corrected']),
                    ('front_up',values['up'],traced['up_corrected']),('tail_control',control['output'],traced['output']))}
                self.audit.append(dict(layer=i,controls=checks))
                if not all(c['exact_fraction']==1 for c in checks.values()):
                    raise ValueError(('Graph partition changes baseline arithmetic',i,checks))
                del traced,control,ordinary
            activated=F.silu(values['gate'].float()).half();gated=activated*values['up']
            output,_=study.run_graph(npu,tail,[residual,gated]);x=output['output']
            if not self.audited:
                error=study.device.compare(x.cpu(),reference)
                self.audit[-1]['staged_cpu']=error
                assert error['relative_l2']<.003 and error['max_scaled_error']<.01,(i,error)
        self.audited=True
        return x


def internal(mode):
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    report=dict(status='running',mode=mode,positions=[],audit=[],source_sha256=base.sha(Path(__file__)),
        probe_source_sha256=base.sha(Path(study.__file__)),weights_sha256=base.read('prepare')['weights_sha256'],
        factors_sha256=base.read('factors')['factor_sha256'])
    def work():
        cpu,_=base.mixed.common.model()
        stack=FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,
            packed_weights=base.mixed.weights(),corrections=base.corrections(32))
        model=StrictStack(stack) if mode=='strict' else stack
        rows=[r for r in base.read('prepare')['trajectories'] if r['index']%5==4]
        assert [r['index'] for r in rows]==[4,9,14,19,24,29,34,39]
        for row in rows:
            ids=torch.tensor(row['token_ids']);offset=row['prefix']-1
            expected=cpu(ids,use_cache=False).logits[:,offset:].float();x=cpu.model.embed_tokens(ids).contiguous().to('intel_npu')
            try:output=model(x)
            finally:
                if mode=='strict':report['audit']=model.audit
            actual=cpu.lm_head(cpu.model.norm(output.cpu())).float()[:,offset:]
            for j in range(actual.shape[1]):
                report['positions'].append(dict(tag=str(row['index']),position=offset+j,**base.mixed.metric(actual[:,j],expected[:,j])))
            print('activation internal',mode,row['index'],flush=True)
        assert len(report['positions'])==290
        before=npu.stats();started=time.perf_counter();again=model(x);elapsed=time.perf_counter()-started;after=npu.stats()
        assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
        torch.testing.assert_close(again.cpu(),output.cpu(),rtol=0,atol=0)
        report.update(summary=base.mixed.common.summary(report['positions']),stats=npu.stats(),
            warm=dict(seconds=elapsed,calls=after['npu_calls']-before['npu_calls'],new_sessions=0,host_copies=0))
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return study.device.finish(report,'activation-internal-'+mode,npu)


def select():
    records={mode:base.read('activation-internal-'+mode) for mode in ('baseline','strict')}
    for r in records.values():assert r['healthy'] and r['status']=='passed' and r['source_sha256']==base.sha(Path(__file__))
    baseline,candidate=(records[m]['summary'] for m in ('baseline','strict'))
    choice='strict' if promotable(baseline,candidate) else 'baseline'
    base.write('activation-selection',dict(status='frozen',selected=choice,summaries={m:r['summary'] for m,r in records.items()},
        records_sha256={m:base.sha(OUT/('activation-internal-'+m+'.json')) for m in records},
        source_sha256=base.sha(Path(__file__)),
        policy='Eight internal trajectories only. Require more raw-gate passes, no worse mean/max L2 or max KL, '
        'and no fewer matching next tokens. Otherwise retain baseline. No external quality claim.'))
    print('ACTIVATION SELECTED',choice,baseline,candidate,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('internal','select'))
    parser.add_argument('--mode',choices=('baseline','strict'),default='strict')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():result=internal(args.mode) if args.stage=='internal' else select()
    raise SystemExit(result or 0)

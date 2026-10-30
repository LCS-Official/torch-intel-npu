"""S52: controlled full-decoder SHAVE FP32 softmax prototype."""
import argparse
from pathlib import Path
import time
import traceback

import torch
import w8_attention_quality as study

base=study.base
OUT=study.OUT


def front_graph(graph,xml):
    values=study.attention_nodes(graph)
    return study.probe.serialize_probe((graph.inputs[1],),dict(masked=values['masked'],value=values['v_repeat']),xml,'attention_front')


class StrictAttentionStack:
    def __init__(self,stack):
        self.layers=stack.layers;self.recipes={};self.audit=[];self.audited=False;self.failure=None

    def __call__(self,x):
        import torch_intel_npu as npu
        for i,layer in enumerate(self.layers):
            if not self.audited:reference,_=layer.reference_cpu(x.cpu());ordinary=layer(x)
            normalized=layer._norm_device(x,'input_layernorm.weight');key=(i,tuple(x.shape))
            if key not in self.recipes:
                graph,xml,_=layer.graph(tuple(x.shape),segment='attention')
                self.recipes[key]=(front_graph(graph,xml),study.attention_tail_graph(graph,xml,'probs'))
            front,tail=self.recipes[key];values,_=study.probe.run_graph(npu,front,[normalized])
            if not self.audited:
                residual=layer._run_fused(x,segment='attention')[0]
                graph,xml,_=layer.graph(tuple(x.shape),segment='attention')
                trace,_=study.probe.run_graph(npu,study.attention_probe_graph(graph,xml),[x,normalized])
                control,_=study.probe.run_graph(npu,tail,[x,trace['probs'],values['value']])
                split=layer._run_fused(residual,segment='ffn')[0]
                checks=dict(front_masked=study.compare_masked(values['masked'].cpu(),trace['masked'].cpu()),
                    front_value=study.device.compare(values['value'].cpu(),trace['v_repeat'].cpu()),
                    exported_attention=study.device.compare(trace['output'].cpu(),residual.cpu()),
                    tail_control=study.device.compare(control['output'].cpu(),residual.cpu()),
                    split_ffn=study.device.compare(split.cpu(),ordinary.cpu()))
                if not all(r['exact_fraction']==1 for r in checks.values()):raise ValueError(('Partition changes original arithmetic',i,checks))
                self.audit.append(dict(layer=i,controls=checks));del trace,control,split,residual,ordinary
            probabilities=values['masked'].float().softmax(-1).half()
            residual,_=study.probe.run_graph(npu,tail,[x,probabilities,values['value']])
            x=layer._run_fused(residual['output'],segment='ffn')[0]
            if not self.audited:
                error=study.device.compare(x.cpu(),reference);self.audit[-1]['local']=error
                if error['relative_l2']>=.003 or error['max_scaled_error']>=.01:
                    self.failure=dict(layer=i,**error);raise ValueError(('Candidate violates unchanged local gate',self.failure))
        self.audited=True
        return x


def identity():
    return dict(source_sha256=base.sha(Path(__file__)),diagnostic_source_sha256=base.sha(Path(study.__file__)),
        weights_sha256=base.read('prepare')['weights_sha256'],factors_sha256=base.read('factors')['factor_sha256'])


def internal(mode):
    import torch_intel_npu as npu
    report=dict(status='running',mode=mode,positions=[],local_checks=[],audit=[],**identity())
    def work():
        cpu,_=base.mixed.common.model();stack=study.previous.stack_for('baseline')
        model=StrictAttentionStack(stack) if mode=='softmax-f32' else stack
        rows=[r for r in base.read('prepare')['trajectories'] if r['index']%5==4]
        assert [r['index'] for r in rows]==[4,9,14,19,24,29,34,39]
        for index,row in enumerate(rows):
            ids=torch.tensor(row['token_ids']);offset=row['prefix']-1
            expected=cpu(ids,use_cache=False).logits[:,offset:].float();inputs=cpu.model.embed_tokens(ids).contiguous().to('intel_npu')
            try:
                if index==0 and mode=='baseline':
                    x=inputs
                    for i,layer in enumerate(stack.layers):
                        reference,_=layer.reference_cpu(x.cpu());x=layer(x);error=study.device.compare(x.cpu(),reference)
                        report['local_checks'].append(dict(layer=i,**error))
                        assert error['relative_l2']<.003 and error['max_scaled_error']<.01,(i,error)
                else:x=model(inputs)
            finally:
                if mode=='softmax-f32':
                    report['audit']=model.audit
                    if model.failure is not None:report['local_guard_failure']=model.failure
            actual=cpu.lm_head(cpu.model.norm(x.cpu())).float()[:,offset:]
            for j in range(actual.shape[1]):report['positions'].append(dict(tag=str(row['index']),position=offset+j,**base.mixed.metric(actual[:,j],expected[:,j])))
            print('attention internal',mode,row['index'],flush=True)
        assert len(report['positions'])==290
        before=npu.stats();started=time.perf_counter();again=model(inputs);elapsed=time.perf_counter()-started;after=npu.stats()
        assert all(after[k]==before[k] for k in ('session_creations','host_upload_copies','host_download_copies'))
        torch.testing.assert_close(again.cpu(),x.cpu(),rtol=0,atol=0)
        report.update(summary=base.mixed.common.summary(report['positions']),stats=npu.stats(),
            warm=dict(seconds=elapsed,calls=after['npu_calls']-before['npu_calls'],new_sessions=0,host_copies=0))
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    return study.device.finish(report,'internal-'+mode,npu)


def choose(baseline,candidate):
    if not baseline['healthy'] or baseline['status']!='passed' or baseline['summary']['positions']!=290:
        raise ValueError('Invalid baseline execution')
    if not candidate['healthy']:raise ValueError('Unhealthy candidate execution')
    if candidate['status']!='passed':
        guard=candidate.get('local_guard_failure')
        if not guard or not (guard['relative_l2']>=.003 or guard['max_scaled_error']>=.01):raise ValueError('Unexpected candidate failure')
        return 'baseline'
    return 'softmax-f32' if study.previous.previous.promotable(baseline['summary'],candidate['summary']) else 'baseline'


def select():
    records={mode:base.read('internal-'+mode) for mode in ('baseline','softmax-f32')};current=identity()
    assert all(all(row[k]==v for k,v in current.items()) for row in records.values())
    selected=choose(records['baseline'],records['softmax-f32'])
    base.write('attention-selection',dict(status='frozen',selected=selected,**current,
        summaries={m:r.get('summary') for m,r in records.items()},local_guard_failure=records['softmax-f32'].get('local_guard_failure'),
        records_sha256={m:base.sha(OUT/('internal-'+m+'.json')) for m in records},
        policy='Only softmax changes to the existing SHAVE FP32 path. Fixed 30 layers, W8 codes/scales/rank32 unchanged. '
        'Eight internal NPU trajectories, 290 positions; more passes and no worse mean/max L2, max KL or matching tokens. '
        'Original local precision guard and exact partition controls remain mandatory. No independent quality claim.'))
    print('ATTENTION SELECTED',selected,{m:r.get('summary',r.get('local_guard_failure')) for m,r in records.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('internal','select'))
    parser.add_argument('--mode',choices=('baseline','softmax-f32'),default='softmax-f32')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():result=internal(args.mode) if args.stage=='internal' else select()
    raise SystemExit(result or 0)

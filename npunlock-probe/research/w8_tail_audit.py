"""CPU-reference generalization audit after the S55 candidate is frozen."""
from pathlib import Path
import sys

import torch
import w8_tail_quality as study


def main():
    base=study.base;out=study.OUT;selection=study.previous.frozen_selection()
    if selection['candidate'] is None:raise ValueError('No internally valid candidate')
    current=study.source_identity();cpu,_=base.mixed.common.model();groups={}
    internal=[r for r in base.read('prepare')['trajectories'] if r['index']%5==4]
    assert [r['index'] for r in internal]==[4,9,14,19,24,29,34,39]
    groups['internal']=[dict(tag=str(r['index']),token_ids=r['token_ids'],offset=r['prefix']-1) for r in internal]
    for group in ('regression','fresh'):groups[group]=base.read(group)['cases']
    cases={group:[] for group in groups}
    for group,rows in groups.items():
        for row in rows:
            ids=torch.tensor(row['token_ids']);offset=row['offset']
            cases[group].append((row['tag'],ids,offset,cpu(ids,use_cache=False).logits[:,offset:].float()))
    modules=base.install(cpu);assert len(modules)==210
    report=dict(status='running',source_sha256=base.sha(Path(__file__)),fitting_identity=current,
        selection_sha256=base.sha(out/'selection.json'),candidate=selection['candidate'],variants={},npu_calls=0)
    for label,profile in (('baseline','baseline'),('candidate',selection['candidate'])):
        _,factors=study.parameters(profile)
        for index,(name,module) in enumerate(modules):
            down,up=factors[index//7][name].tensors()
            torch.testing.assert_close(module.down,down,rtol=0,atol=0)
            module.up.copy_(up);module.rank=32;module.protected=False
        report['variants'][label]={}
        for group,rows in cases.items():
            positions=[]
            for tag,ids,offset,reference in rows:
                actual=cpu(ids,use_cache=False).logits[:,offset:].float()
                assert actual.shape==reference.shape and torch.isfinite(actual).all()
                for j in range(actual.shape[1]):positions.append(dict(tag=tag,position=offset+j,
                    **base.mixed.metric(actual[:,j],reference[:,j])))
            assert len(positions)==sum(64-offset for _,_,offset,_ in rows)
            result=dict(positions=positions,summary=base.mixed.common.summary(positions))
            report['variants'][label][group]=result
            print('CPU generalization',label,group,result['summary'],flush=True)
    assert study.source_identity()==current and base.sha(out/'selection.json')==report['selection_sha256']
    assert 'torch_intel_npu' not in sys.modules
    report['status']='passed';base.write('cpu-generalization',report)


if __name__=='__main__':
    torch.set_num_threads(8)
    with torch.no_grad():main()

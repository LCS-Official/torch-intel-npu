"""S48 language checks reuse S44's pinned text, windows and generation protocol."""
import argparse
import gc
from pathlib import Path
import sys
import traceback

import torch
import w8_output_quality as study

base=study.base
OUT=study.OUT


def data():
    study.frozen_selection()
    base.data()


def identity():
    selection=study.frozen_selection();metadata=base.read('language-data')
    assert metadata['selection_sha256']==base.sha(OUT/'selection.json')
    assert metadata['corpus_sha256']==base.sha(OUT/'corpus.json')
    return selection,dict(selection_sha256=base.sha(OUT/'selection.json'),
        language_data_sha256=base.sha(OUT/'language-data.json'),source_sha256=base.sha(Path(__file__)),
        quality_source_sha256=base.sha(Path(study.__file__)),candidate=selection['candidate'])


def cpu():
    selection,report=identity();model,tokenizer=base.mixed.common.model()
    language=base.language_helpers();window=next(language.windows(base.read('language-data')['cases'][0]['tokens']))
    ids=torch.tensor([window['input_ids']]);reference=model(ids,use_cache=False).logits
    variants={'fp16':base.language_quality(model,tokenizer)}
    modules=base.install(model)
    for _,module in modules:module.protected=True
    torch.testing.assert_close(model(ids,use_cache=False).logits,reference,rtol=0,atol=0)
    for label,profile in (('baseline','baseline'),('candidate',selection['candidate'])):
        study.replace_down(modules,study.profile_factors(profile))
        for _,module in modules:module.rank=32;module.protected=False
        variants[label]=base.language_quality(model,tokenizer)
        print('CPU language',label,variants[label]['summary'],flush=True)
    assert 'torch_intel_npu' not in sys.modules
    report.update(status='passed',variants=variants,npu_calls=0)
    base.write('language-cpu',report)


def hardware(label):
    selection,report=identity()
    import torch_intel_npu as npu
    from torch_npu_smollm import FrozenSmolStack
    report.update(label=label,status='running')
    def work():
        model,tokenizer=base.mixed.common.model();profile=selection['candidate'] if label=='candidate' else 'baseline'
        stack=FrozenSmolStack.from_checkpoint(base.mixed.common.MODEL,num_layers=30,
            packed_weights=base.mixed.weights(),corrections=study.profile_factors(profile))
        result=base.language_quality(model,tokenizer,stack=stack)
        assert len(result['local_checks'])==30 and len(result['generation'])==len(result['warm'])==8
        report.update(result=result,stats=npu.stats())
    try:work();report['status']='passed'
    except Exception:
        report.update(status='failed',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    gc.collect();npu.shutdown();report['shutdown']=npu.stats()
    report['healthy']=report['shutdown']['closed'] and not report['shutdown']['context_failed'] and all(
        report['shutdown'][k]==0 for k in ('live_buffers','cached_buffer_blocks','cached_sessions','quarantined_buffer_bytes'))
    if not report['healthy']:report['status']='failed'
    base.write('language-npu-'+label,report)
    print('NPU language',label,report.get('result',{}).get('summary'),report['healthy'],flush=True)
    return int(report['status']!='passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('data','cpu','hardware'))
    parser.add_argument('--variant',choices=('baseline','candidate'),default='candidate')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(8)
    with torch.no_grad():result=hardware(args.variant) if args.stage=='hardware' else globals()[args.stage]()
    raise SystemExit(result or 0)

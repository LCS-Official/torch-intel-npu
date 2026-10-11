"""Reuse the paired graph-generation protocol, saving every logits row."""
import argparse
from pathlib import Path

from decode_position_flags import OUT
from decoder_refresh import read, write
from w8_quality_probe import ROOT, sha


def main(mode):
    template=ROOT/'npunlock-probe/research/decode_mask_generation.py'
    text=template.read_text()
    replacements={
        'from decode_mask_table import OUT':'from decode_position_flags import OUT',
        "OUT/'joint/joined-graph.py'":"OUT/'flags-graph.py'",
        "OUT/'joint/baseline-graph.py'":"OUT/'baseline-graph.py'",
        'original vs shared positional Gather, both group4':'current graph vs shared KV flag lookup, both group4',
        'text=source.read_text()':"text=source.read_text().replace('for step in (0,15,31):','for step in range(32):')"}
    for old,new in replacements.items():assert text.count(old)==1,old;text=text.replace(old,new)
    directory=OUT/'generation';directory.mkdir(exist_ok=True)
    runner=directory/'graph-wrapper.py';runner.write_text(text,encoding='utf-8')
    scope=dict(__name__='decode_position_generation_runner',__file__=str(runner))
    exec(compile(text,str(runner),'exec'),scope);scope['main'](mode)
    path=directory/(mode+'.json');report=read(path)
    assert all(s['delta']['driver_allocations']==0 for s in report['samples'])
    report.update(factory_sha256=sha(Path(__file__)),graph_wrapper_template_sha256=sha(template))
    write(path,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('fp16','w8a16'),required=True)
    main(parser.parse_args().mode)

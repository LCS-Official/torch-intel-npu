"""Promote the verified RMSNorm candidate and run existing model regressions."""
import json
from pathlib import Path
import shutil

from rmsnorm_half_slow import OUT
from w8_quality_probe import ROOT,sha


def main():
    audit=json.loads((OUT.parent/'audit.json').read_text())
    assert audit['status']=='passed' and not audit['promoted']
    for name,model in audit['models'].items():
        for row in model.values():
            assert row['median_ms']['codec']<row['median_ms']['baseline'] and row['wins']>=(24 if name=='full30' else 20)
    for data in audit['generation'].values():
        assert data['summary']['4']['elapsed_ms']<data['summary']['1']['elapsed_ms'] and data['wins']>=5
    build=json.loads((OUT/'build.json').read_text());candidate=build['kernels']['codec']
    assert sha(OUT/'codec.c')==candidate['source_sha256']
    names=('npunlock/tests/fixtures/npu4000/rmsnorm.c','npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.elf',
           'npunlock/tests/fixtures/npu4000/rmsnorm/rmsnorm-f16.s','npunlock/tests/fixtures/npu4000/rmsnorm/manifest.json')
    assert sha(ROOT/names[0])==build['kernels']['baseline']['source_sha256']
    assert sha(ROOT/names[1])==build['kernels']['baseline']['elf_sha256']
    assert not (OUT/'promotion.json').exists()
    shutil.copyfile(OUT.parent/'audit.json',OUT/'pre-promotion-audit.json')
    report=dict(status='running',before={},after={},audit_sha256=sha(OUT/'pre-promotion-audit.json'),source_sha256=sha(Path(__file__)))
    for name in names:
        path=OUT/'before'/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,path);report['before'][name]=sha(path)
    import probe_movi_targets as compiler
    import build_rmsnorm
    import rmsnorm_reduce_regression as regression
    compiler.OUT=OUT/'formal-build';compiler.OUT.mkdir(exist_ok=True)
    try:
        shutil.copyfile(OUT/'codec.c',ROOT/names[0]);build_rmsnorm.main()
        assert sha(ROOT/names[1])==candidate['elf_sha256'] and sha(ROOT/names[2])==candidate['assembly_sha256']
        regression.OUT=OUT/'regressions';assert regression.main()==0
        for name in names:
            path=OUT/'after'/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,path);report['after'][name]=sha(path)
        report['status']='passed'
    except Exception:
        for name in names:shutil.copyfile(OUT/'before'/name,ROOT/name)
        raise
    (OUT/'promotion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Promoted exact tested C/ELF/assembly; all existing RMSNorm/model regressions passed',flush=True)


if __name__=='__main__':main()

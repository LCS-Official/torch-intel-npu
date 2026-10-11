"""Same-blob Select copy kernel comparison using the existing full checks."""
import json
from pathlib import Path
import sys
import kv_group_probe as control
from kv_select_kernel import OUT, candidate, sha


def main():
    control.OUT=OUT;control.candidate=candidate
    code=control.main()
    label=sys.argv[sys.argv.index('--label')+1] if '--label' in sys.argv else 'process0'
    path=OUT/label/'probe.json';report=json.loads(path.read_text(encoding='utf-8'))
    report.update(implementation='select_shave_copy',entry_sha256=sha(Path(__file__)),kernel_build=json.loads((OUT/'build.json').read_text(encoding='utf-8')))
    path.write_text(json.dumps(report,indent=2),encoding='utf-8');return code


if __name__=='__main__':raise SystemExit(main())

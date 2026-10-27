"""Run normalization/activation acceptance and existing bounded device regressions."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'npunlock-probe/research/results/norm-activation'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commands = [(f'new-{case}', 'torch_npu_norm_activation_demo.py', '--case', case)
                for case in ('gelu', 'layernorm', 'contracts', 'training')]
    commands += [('migration', 'torch_npu_migration_demo.py')]
    commands += [(f'classification-{case}', 'torch_npu_classification_demo.py', '--case', case)
                 for case in ('logsoftmax', 'nll', 'contracts', 'training')]
    commands += [('device', 'torch_npu_device_demo.py', '--quick'),
                 ('training', 'torch_npu_training_demo.py', '--quick'),
                 ('compatibility', 'torch_npu_compat_demo.py')]
    commands += [(f'adamw-{case}', 'torch_npu_adamw_demo.py', '--case', case)
                 for case in ('primitives', 'optimizer', 'contracts')]
    runs = []
    for label, script, *arguments in commands:
        process = subprocess.run([sys.executable, str(ROOT/script), *arguments], capture_output=True,
                                 text=True, encoding='utf-8', timeout=1200)
        (OUT/f'regression-{label}.log').write_text(process.stdout+process.stderr, encoding='utf-8')
        runs.append({'name': label, 'returncode': process.returncode})
        (OUT/'regression.json').write_text(json.dumps(runs, indent=2), encoding='utf-8')
        print(label, 'exit', process.returncode, flush=True)
        if label.startswith('new-'):
            print(process.stdout, flush=True)
    return int(any(row['returncode'] for row in runs))


if __name__ == '__main__':
    raise SystemExit(main())

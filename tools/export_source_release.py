"""Export a reviewable source snapshot without environments, models or driver blobs."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'.blob', '.bin', '.dll', '.pyd', '.exe', '.obj', '.lib', '.exp',
             '.pyc', '.whl', '.npc', '.npz', '.npy', '.safetensors'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root / 'npunlock'), *args])


def fork_file_allowed(name):
    p = Path(name)
    if p.parts[0] == '.github' or p.suffix.lower() in FORBIDDEN:
        return False
    # Only this project's source-backed NPU4000 kernels accompany the source.
    return p.suffix != '.elf' or p.parts[:3] == ('tests', 'fixtures', 'npu4000')


def output_path(root, output):
    output = Path(output).absolute()
    if output.is_symlink() or output.resolve().parent != root.resolve() or not output.name.startswith('source-release'):
        raise ValueError('Output must be a new source-release* directory directly inside the workspace')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; choose a new source-release* name')
    return output


def selection(root):
    names = {p.relative_to(root).as_posix() for pattern in ('*.py', '*.ps1', '*.md', 'requirements*.txt') for p in root.glob(pattern)}
    names.update({'LICENSE', 'NOTICE', '.gitignore', '.gitattributes', 'references.bib'})
    names.update(p.relative_to(root).as_posix() for p in (root / '.github').rglob('*') if p.is_file())
    for directory in ('tools', 'LICENSES', 'provenance'):
        names.update(p.relative_to(root).as_posix() for p in (root / directory).glob('*') if p.is_file())
    fork = git(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0')
    excluded = sorted(name for name in fork if name and not fork_file_allowed(name))
    names.update('npunlock/' + name for name in fork if name and fork_file_allowed(name))
    research = root / 'npunlock-probe/research'
    names.update(p.relative_to(root).as_posix() for p in research.glob('*')
                 if p.is_file() and p.suffix in {'.py', '.ps1', '.c', '.cpp', '.md'}
                 and p.name != 'portable-msvc.py')
    names.update(p.relative_to(root).as_posix() for p in (research / 'results').glob('*/REPORT.md'))
    names.add('npunlock-probe/research/sources/manifest.json')
    names.add('npunlock-probe/torch-device/backend.cpp')
    for name in ('README.md', 'probe.py', 'run.ps1', 'download_artifacts.py'):
        names.add('npunlock-probe/' + name)
    return sorted(names), excluded


def upstream_record(root, names, excluded):
    base = git(root, 'rev-parse', 'HEAD').decode().strip()
    tracked = set(git(root, 'ls-tree', '-r', '--name-only', 'HEAD').decode().splitlines())
    records, patch = [], []
    for name in names:
        if not name.startswith('npunlock/'):
            continue
        rel = name[len('npunlock/'):]
        current = (root / name).read_bytes()
        original = git(root, 'show', f'HEAD:{rel}') if rel in tracked else b''
        if rel not in tracked:
            status = 'added'
        elif original == current:
            status = 'unchanged'
        elif original.replace(b'\r\n', b'\n') == current.replace(b'\r\n', b'\n'):
            status = 'line_endings_only'
        else:
            status = 'modified'
        records.append(dict(path=rel, status=status, source_sha256=sha(current),
                            upstream_sha256=sha(original) if rel in tracked else None))
        if status in {'modified', 'added'} and Path(rel).suffix != '.elf':
            # Text patch uses LF as Git does; byte-exact hashes remain in the manifest.
            old = original.decode('utf-8').replace('\r\n', '\n').splitlines(keepends=True)
            new = current.decode('utf-8').replace('\r\n', '\n').splitlines(keepends=True)
            patch.append(f'diff --git a/{rel} b/{rel}\n')
            if rel not in tracked:
                patch.append('new file mode 100644\n')
            for line in difflib.unified_diff(old, new, fromfile='a/' + rel if rel in tracked else '/dev/null', tofile='b/' + rel):
                patch.append(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n')
    record = dict(repository='https://github.com/hsfzxjy/npunlock', base_commit=base,
                  patch_scope='Included changed text files only; kernel ELF companions are covered by hashes.',
                  files=records, excluded_upstream_or_local_files=excluded)
    return record, ''.join(patch)


def verify(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    expected = {row['path'] for row in manifest['files']} | {'SOURCE_MANIFEST.json'}
    # A GitHub clone adds root Git metadata, which is not source payload.
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
              if p.is_file() and p.relative_to(directory).parts[0] != '.git'}
    if actual != expected:
        raise ValueError(f'File set changed: {sorted(actual ^ expected)[:8]}')
    for row in manifest['files']:
        path = directory / row['path']
        if not path.resolve().is_relative_to(directory) or path.is_symlink():
            raise ValueError(f'Unsafe export member: {row["path"]}')
        data = path.read_bytes()
        if len(data) != row['bytes'] or sha(data) != row['sha256']:
            raise ValueError(f'Checksum differs: {row["path"]}')
    return len(manifest['files']), sum(row['bytes'] for row in manifest['files'])


def export(root, output):
    output = output_path(root, output)
    names, excluded = selection(root)
    payload = {}
    for name in names:
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or path.suffix.lower() in FORBIDDEN:
            raise ValueError(f'Unsafe source member: {name}')
        data = path.read_bytes()
        if re.search(rb'-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}', data):
            raise ValueError(f'Possible credential in {name}; inspect before export')
        payload[name] = data
    record, patch = upstream_record(root, names, excluded)
    payload['provenance/npunlock-upstream.json'] = (json.dumps(record, indent=2) + '\n').encode()
    payload['provenance/npunlock-changes.patch'] = patch.encode()
    manifest = dict(schema=1, scope='Source, reports and source-backed NPU4000 kernel companions; no driver-generated graph blobs.',
                    files=[dict(path=name, bytes=len(data), sha256=sha(data)) for name, data in sorted(payload.items())])
    output.mkdir()
    for name, data in payload.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (output / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    count, size = verify(output)
    print(f'Exported and verified {count} files, {size / 1024**2:.3f} MiB: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'source-release')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        count, size = verify(args.verify)
        print(f'Verified {count} files, {size / 1024**2:.3f} MiB')
    else:
        export(ROOT, args.output)

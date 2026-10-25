#!/usr/bin/env python3
"""Prepare a fresh H unit from ROOT's complete assembled workspace and manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tarfile
import time
import tomllib

BASE = 'resume-all-five-hip-20261025-1'
PREDECESSOR_BASES = (
    'hip-multiset-source12-20261024-4',
    'hip-multiset-controls-v5-20261024-1',
    'hip-multiset-controls-v6-20261024-1',
    'hip-multiset-controls-v7-20261024-1',
    'hip-multiset-source20-20261024-2',
)
assert BASE not in PREDECESSOR_BASES
OWNER_SHA = '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'
CANCEL_SHA = '00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128'
# Immutable historical provenance only; never a current file list or authority.
BASELINE_PINS = {
    'assembled-build-source-19.json':
        {'bytes': 370276,
         'sha256': 'bf411c365550b6789144b06353c9a8754dcb3e117f85458e28aef4536b1d897d'},
    'assembled-build-source-20.json':
        {'bytes': 370334,
         'sha256': '30a79602c1812a921c61424da989f5b41aeb3a2bf3f06d2e38c28782320c02b2'},
}
SCOPE = ('complete ROOT-assembled all-five source/build closure, one hip_shared_models '
         'artifact, the three original finite shared lifecycle modes, and three bounded '
         'source-page query/wait/destroy injected-policy-unknown probe modes; no actual '
         'HIP driver query proof, recovery or general pending-event grant')
REQUIRED_CONFIG = ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml',
                   'scripts/check.sh', 'scripts/native-gpu.py', 'README.md')
# Match the unchanged runner/bootstrap.extract_archive bounds: one member and
# the complete source archive.
SOURCE_MEMBER_LIMIT = 64*1024**2
SOURCE_TOTAL_LIMIT = 192*1024**2
# The five active lanes required by the complete assembly: core, Qwen35,
# Gemma4, GPU and the shared engine. A H-only fixture tree is not it.
REQUIRED_UNITS = ('crates/zerogravity-core/',
                  'crates/zerogravity-model-qwen35/',
                  'crates/zerogravity-model-gemma4/',
                  'crates/zerogravity-gpu/',
                  'crates/zerogravity/')
OWNER_PATH = 'source/tools/qwen27b-full-window-reference/owned_container_v4.py'
TEMPLATE_ROOT = Path(__file__).resolve().parent
PAYLOAD_TEMPLATES = TEMPLATE_ROOT/'payload'
INCLUDE = re.compile(r'\binclude(?:_str|_bytes)?!\s*\(\s*"([^"]+)"')


def require(value, message):
    if not value:
        raise ValueError(message)


def pin_bytes(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def pin(path, limit=SOURCE_MEMBER_LIMIT):
    raw = Path(path).read_bytes()
    require(len(raw) <= limit, 'bounded regular input: '+str(path))
    return pin_bytes(raw)


def dump(path, value, exclusive=False):
    text = json.dumps(value, indent=2)+'\n'
    if exclusive:
        with Path(path).open('x') as file:
            file.write(text)
    else:
        Path(path).write_text(text)


def safe_rel(name):
    p = Path(name)
    require(isinstance(name, str) and not p.is_absolute() and p.parts
            and '..' not in p.parts and p.as_posix() == name,
            'safe contained relative path: '+str(name))
    return p


def mode_of(st):
    return stat.S_IMODE(st.st_mode)


def output_path(value, label, source_real, manifest_path, tool_real):
    path = Path(value)
    require(path.is_absolute() and not os.path.lexists(path),
            'fresh absolute output: '+label)
    parent = path.parent
    require(parent.is_dir() and not parent.is_symlink(),
            'existing real output parent: '+label)
    real_parent = Path(os.path.realpath(parent))
    require(real_parent == parent, 'symlink-free output parent: '+label)
    real = real_parent/path.name
    require(real != manifest_path, 'output cannot overwrite the source manifest: '+label)
    require(not real.is_relative_to(tool_real), 'no output inside the tracked tool tree: '+label)
    if real.is_relative_to(source_real):
        rel = real.relative_to(source_real)
        require(len(rel.parts) >= 4 and rel.parts[0] == '.agents'
                and rel.parts[1] == 'runs',
                'source-root output only under a fresh .agents/runs run: '+label)
    return path, real


def read_manifest(path):
    raw = Path(path).read_bytes()
    require(len(raw) <= 16*1024**2, 'bounded complete source manifest')
    value = json.loads(raw)
    require(isinstance(value, dict) and isinstance(value.get('files'), dict)
            and value['files'], 'manifest files dictionary required')
    if 'count' in value:
        require(type(value['count']) is int and value['count'] == len(value['files']),
                'manifest count equals actual records')
    for name, entry in value['files'].items():
        safe_rel(name)
        require(isinstance(entry, dict) and entry.get('kind') == 'file',
                'file record: '+name)
        require(type(entry.get('bytes')) is int
                and 0 <= entry['bytes'] <= SOURCE_MEMBER_LIMIT,
                'bounded record bytes: '+name)
        require(type(entry.get('mode')) is int and 0 <= entry['mode'] <= 0o7777,
                'record mode: '+name)
        sha = entry.get('sha256')
        require(isinstance(sha, str) and len(sha) == 64 and sha == sha.lower()
                and all(c in '0123456789abcdef' for c in sha),
                'record sha256: '+name)
    require(sum(entry['bytes'] for entry in value['files'].values())
            <= SOURCE_TOTAL_LIMIT,
            'bounded declared complete source bytes')
    return value, raw


def source_record(source_root, source_real, name, entry):
    rel = safe_rel(name)
    path = source_root/rel
    walked = source_root
    for part in rel.parts:
        walked = walked/part
        require(not walked.is_symlink(), 'no source symlink: '+name)
    real = Path(os.path.realpath(path))
    require(real.is_relative_to(source_real), 'contained source: '+name)
    st = os.lstat(real)
    require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode),
            'regular source file: '+name)
    require(st.st_size == entry['bytes'] and st.st_size <= SOURCE_MEMBER_LIMIT,
            'source bytes: '+name)
    require(mode_of(st) == entry['mode'], 'source mode: '+name)
    raw = real.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == entry['sha256'],
            'source sha256: '+name)
    return raw


def cargo_inputs(source_root, source_real, files):
    """Mechanical Cargo/path-dependency/vendor/build-script/include closure."""
    parsed = {}
    for name in files:
        if name == 'Cargo.toml' or name.endswith('/Cargo.toml'):
            raw = source_record(source_root, source_real, name, files[name])
            parsed[name] = tomllib.loads(raw.decode('utf-8'))
    require('Cargo.toml' in parsed, 'workspace Cargo.toml present')

    def need(rel, why):
        require(rel in files, 'complete source closure missing '+why+': '+rel)

    def contained(base, raw, why):
        value = os.path.normpath((base/Path(raw)).as_posix())
        require(not Path(value).is_absolute() and '..' not in Path(value).parts,
                'contained '+why+': '+str(raw))
        return value

    def crate_dir(name):
        return Path(name).parent

    def dep_paths(table, dirname):
        if not isinstance(table, dict):
            return
        for dep in table.values():
            if isinstance(dep, dict) and 'path' in dep:
                target = contained(dirname, dep['path'], 'path dependency')
                need(target+'/Cargo.toml', 'path dependency')

    def dep_tables(manifest):
        for section in ('dependencies', 'dev-dependencies', 'build-dependencies'):
            yield section, manifest.get(section, {})
        for cfg, table in manifest.get('target', {}).items():
            if isinstance(table, dict):
                for section in ('dependencies', 'dev-dependencies', 'build-dependencies'):
                    yield 'target.'+str(cfg)+'.'+section, table.get(section, {})

    for name, manifest in parsed.items():
        dirname = crate_dir(name)
        for sub in ('src', 'examples', 'tests', 'benches'):
            root = source_root/dirname/sub
            if root.is_dir():
                for path in root.rglob('*'):
                    require(not path.is_symlink(),
                            'no declared source symlink: '+path.as_posix())
                    if path.is_file():
                        need(path.relative_to(source_root).as_posix(),
                             'declared crate '+sub+' file')
        for _, table in dep_tables(manifest):
            dep_paths(table, dirname)
        for target in ('example', 'bin', 'test', 'bench'):
            for item in manifest.get(target, []):
                if isinstance(item, dict) and 'path' in item:
                    need(contained(dirname, item['path'], target+' path'), target+' path')
        lib = manifest.get('lib', {})
        if isinstance(lib, dict) and 'path' in lib:
            need(contained(dirname, lib['path'], 'library path'), 'library path')
        require(not Path(name).is_absolute(), 'contained Cargo.toml')
        package = manifest.get('package', {})
        build = package.get('build') if isinstance(package, dict) else None
        if build is False:
            pass  # [package] build = false disables the default script
        elif build is None or build is True:
            if (source_root/dirname/'build.rs').is_file():
                need(contained(dirname, 'build.rs', 'default build script'),
                     'default build script')
        elif isinstance(build, str):
            need(contained(dirname, build, 'declared build script'),
                 'declared build script')
        else:
            raise ValueError('[package] build must be bool or relative path: '+name)

    workspace = parsed['Cargo.toml'].get('workspace', {})
    for member in workspace.get('members', []):
        need(contained(Path('.'), member, 'workspace member')+'/Cargo.toml', 'workspace member')
    for member in workspace.get('default-members', []):
        need(contained(Path('.'), member, 'default workspace member')+'/Cargo.toml',
             'default workspace member')
    dep_paths(workspace.get('dependencies', {}), Path('.'))
    for _, table in parsed['Cargo.toml'].get('patch', {}).items():
        dep_paths(table, Path('.'))

    for name, entry in files.items():
        if not name.endswith('.rs'):
            continue
        raw = source_record(source_root, source_real, name, entry)
        text = raw.decode('utf-8', errors='strict')
        for target in INCLUDE.findall(text):
            resolved = contained(Path(name).parent, target, 'include input')
            need(resolved, 'literal include input')
    return parsed


def copy_templates(payload, now):
    require(PAYLOAD_TEMPLATES.is_dir() and not PAYLOAD_TEMPLATES.is_symlink(),
            'tracked static payload templates')
    for path in sorted(PAYLOAD_TEMPLATES.rglob('*')):
        rel = path.relative_to(PAYLOAD_TEMPLATES)
        require('__pycache__' not in rel.parts and rel.suffix != '.pyc',
                'no bytecode template: '+rel.as_posix())
        require(not path.is_symlink(), 'no template symlink: '+rel.as_posix())
        target = payload/rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        require(path.is_file(), 'regular template: '+rel.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), 'fresh template destination: '+rel.as_posix())
        raw = path.read_bytes()
        target.write_bytes(raw)
        os.chmod(target, mode_of(path.stat()))
        os.utime(target, ns=(now, now))
        require(pin(target) == pin_bytes(raw), 'template copy bytes: '+rel.as_posix())
        require(mode_of(target.stat()) == mode_of(path.stat()),
                'template copy mode: '+rel.as_posix())


def inventory(base):
    return {p.relative_to(base).as_posix(): p for p in sorted(base.rglob('*'))
            if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root
    manifest_path = args.source_manifest
    require(source_root.is_absolute() and source_root.is_dir(), 'explicit absolute source root')
    require(manifest_path.is_absolute() and manifest_path.is_file()
            and not manifest_path.is_symlink(), 'explicit absolute ROOT source manifest')
    source_real = Path(os.path.realpath(source_root))
    manifest_real = Path(os.path.realpath(manifest_path))
    tool_real = Path(os.path.realpath(TEMPLATE_ROOT))
    payload_arg, payload_real = output_path(args.payload, 'payload', source_real,
                                            manifest_real, tool_real)
    archive_arg, archive_real = output_path(args.archive, 'archive', source_real,
                                            manifest_real, tool_real)
    require(payload_real != archive_real and not archive_real.is_relative_to(payload_real)
            and not payload_real.is_relative_to(archive_real),
            'archive outside the fresh payload')
    prepared_real = Path(os.path.realpath(payload_arg.parent))/'PREPARED.json'
    require(prepared_real != manifest_real and prepared_real != payload_real
            and prepared_real != archive_real
            and not prepared_real.is_relative_to(payload_real)
            and not prepared_real.is_relative_to(tool_real)
            and not os.path.lexists(prepared_real),
            'fresh PREPARED.json outside inputs/tool tree')

    manifest, manifest_raw = read_manifest(manifest_path)
    files = manifest['files']
    manifest_pin = pin_bytes(manifest_raw)
    outputs = (payload_real, archive_real, prepared_real)
    for out in outputs:
        if out.is_relative_to(source_real):
            rel = out.relative_to(source_real)
            for name in files:
                require(not Path(name).is_relative_to(rel),
                        'declared output excluded from input discovery: '+out.as_posix())

    for name in REQUIRED_CONFIG:
        require(name in files, 'required workspace/build/config input: '+name)
    for name in files:
        require(name != '.git' and not name.startswith('.git/'),
                'staged source without .git')
    for name in ('.cargo/config.toml', '.cargo/config', 'rust-toolchain'):
        if (source_root/name).is_file():
            require(name in files, 'required Cargo/toolchain config input: '+name)
    if 'flake.nix' in files:
        require('flake.lock' in files, 'flake.lock with flake.nix')
    for prefix in REQUIRED_UNITS:
        require(any(name.startswith(prefix) for name in files),
                'complete five-lane unit closure: '+prefix)
        require(prefix+'Cargo.toml' in files,
                'required unit manifest: '+prefix+'Cargo.toml')
    cargo_inputs(source_root, source_real, files)

    now = time.time_ns()
    payload_arg.mkdir(parents=True, mode=0o755)
    payload = payload_arg
    copy_templates(payload, now)

    assembled = payload/'assembled-source.json'
    require(not assembled.exists(), 'fresh assembled manifest copy')
    assembled.write_bytes(manifest_raw)
    os.chmod(assembled, 0o644)
    os.utime(assembled, ns=(now, now))
    require(pin(assembled) == manifest_pin, 'exact supplied manifest bytes copied')

    total = 0
    source_out = payload/'source'
    require(not source_out.exists(), 'fresh source stage')
    for name, entry in files.items():
        raw = source_record(source_root, source_real, name, entry)
        staged = source_out/safe_rel(name)
        staged.parent.mkdir(parents=True, exist_ok=True)
        require(not staged.exists() and not staged.is_symlink(),
                'fresh staged source: '+name)
        staged.write_bytes(raw)
        os.chmod(staged, entry['mode'])
        os.utime(staged, ns=(now, now))
        require(pin(staged) == {k: entry[k] for k in ('bytes', 'sha256')},
                'staged source bytes: '+name)
        require(mode_of(staged.stat()) == entry['mode'], 'staged source mode: '+name)
        total += len(raw)
    require(total == sum(entry['bytes'] for entry in files.values()),
            'derived complete source bytes')
    require(total <= SOURCE_TOTAL_LIMIT, 'bounded complete source bytes')

    require(pin(payload/OWNER_PATH)['sha256'] == OWNER_SHA,
            'unchanged original owner helper in complete source closure')
    require(pin(payload/'runner/sigterm_cancellation.py')['sha256'] == CANCEL_SHA,
            'unchanged original cancellation helper')

    files_now = inventory(payload)
    for stale in ('runner/checks.py', 'runner/environment.py', 'runner/modes.py',
                  'runner/run.py', 'runner/source-expressions.py'):
        require(stale not in files_now, 'no stale predecessor controller: '+stale)
    require('closure.json' not in files_now, 'offline hydration closure.json not shipped')
    require('control.json' not in files_now and 'source-closure.json' not in files_now,
            'fresh control/closure packaging')

    control = {'schema': 'zg.hip-source20-x2-controller.v1',
               'base': BASE,
               'predecessor_bases': list(PREDECESSOR_BASES),
               'source_manifest': manifest_pin,
               'source_manifest_path': str(manifest_path),
               'source_root': str(source_root),
               'source_count': len(files),
               'changed_paths': {},
               'historical_baseline': BASELINE_PINS,
               'records': {name: pin(path) for name, path in files_now.items()}}
    dump(payload/'control.json', control)
    control_pin = pin(payload/'control.json')

    ship = inventory(payload)
    require(set(ship) == set(files_now) | {'control.json'},
            'control is the only new packaging member')
    closure = {'records': {name: pin(path) for name, path in ship.items()},
               'files': 0, 'bytes': 0,
               'source_count': len(files),
               'source_manifest': manifest_pin,
               'parent_frozen_source_manifest':
                   BASELINE_PINS['assembled-build-source-19.json'],
               'historical_baseline': BASELINE_PINS,
               'scope': SCOPE}
    closure['files'] = len(closure['records'])
    closure['bytes'] = sum(v['bytes'] for v in closure['records'].values())
    require(set(closure['records']) == set(ship) - {'source-closure.json'},
            'source closure records complete')
    dump(payload/'source-closure.json', closure)

    members = inventory(payload)
    require(set(members) == set(closure['records']) | {'source-closure.json'},
            'source closure archive members exact')
    sizes = {name: path.stat().st_size for name, path in members.items()}
    require(all(size <= SOURCE_MEMBER_LIMIT for size in sizes.values())
            and sum(sizes.values()) <= SOURCE_TOTAL_LIMIT,
            'archive members within the unchanged 64MiB/192MiB extraction bounds')
    with tarfile.open(archive_arg, 'x', format=tarfile.GNU_FORMAT) as tar:
        for name, path in sorted(members.items()):
            info = tar.gettarinfo(str(path), arcname=name)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            with path.open('rb') as file:
                tar.addfile(info, file)
    seen = set()
    with tarfile.open(archive_arg) as tar:
        for member in tar:
            require(member.isfile() and member.name in members
                    and member.name not in seen, 'fresh contained archive member')
            path = members[member.name]
            data = tar.extractfile(member).read()
            require(len(data) == path.stat().st_size
                    and hashlib.sha256(data).hexdigest() == pin(path)['sha256'],
                    'archive member bytes: '+member.name)
            require(member.mode == mode_of(path.stat()),
                    'archive member mode: '+member.name)
            seen.add(member.name)
    require(seen == set(members), 'complete archive member population')

    prepared = {'status': 'HIP_SHARED_MODELS_UNIT_PREPARED_REMOTE_HELD',
                'base': BASE,
                'predecessor_bases': list(PREDECESSOR_BASES),
                'source_root': str(source_root),
                'source_manifest_path': str(manifest_path),
                'source_files': len(files),
                'source_bytes': total,
                'source_manifest': manifest_pin,
                'parent_frozen_source_manifest':
                    BASELINE_PINS['assembled-build-source-19.json'],
                'historical_baseline': BASELINE_PINS,
                'changed_paths': {},
                'edited_payload_pins': {},
                'source_closure': pin(payload/'source-closure.json'),
                'controller': control_pin,
                'source_archive': pin(archive_arg),
                'tar_members': len(seen),
                'payload_files': len(members),
                'all_source_bytes_modes_verified': True,
                'all_payload_bytes_modes_verified': True,
                'no_closure_json_shipped': True,
                'no_payload_transfer_of_data_models_caches': True,
                'GPU_or_model': False, 'remote_effect': False}
    require(not os.path.lexists(prepared_real), 'fresh prepared record')
    dump(prepared_real, prepared, exclusive=True)
    print(json.dumps(prepared, indent=2))


if __name__ == '__main__':
    main()

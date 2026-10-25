#!/usr/bin/env python3
"""Freeze one explicit source root + ROOT manifest + the eleven-file natural
tool into a fresh controller payload and one fresh code archive.  The four
absolute operands are authoritative; there is no Git discovery, baseline,
overlay or default fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import time

BASE = 'hip-multiset-natural-x2-20261024-1'
PREDECESSOR_BASES = ('hip-multiset-source20-20261024-2', 'hip-multiset-source20-20261024-1',
                     'hip-multiset-source12-20261024-4', 'hip-multiset-controls-v5-20261024-1')
assert BASE not in PREDECESSOR_BASES
NATURAL_TOOL = 'tools/qwen27b-natural-full-window-reference'
NATURAL_RUNTIME = ('archive.py', 'capture.py', 'contract.py', 'inputs.json',
                   'owned_container_v4.py', 'run_reference.py', 'source-pin.json',
                   'source_owner.py', 'validate_reference.py')
NATURAL_EXTRA = ('README.md', 'test_contract.py')
NATURAL_PREFIX = NATURAL_TOOL+'/'
OWNER_SHA = '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'
CANCEL_SHA = '00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128'
# Static controller templates: the five top-level controls beside this script
# plus the nineteen payload members.  Nothing is assumed to be prepopulated.
CONTROL_TEMPLATES = ('prepare.py', 'bind-build.py', 'control-checks.py',
                     'check-boundary.py', 'caller-signal-controls.py')
PAYLOAD_TEMPLATES = ('hydrate.py', 'check-commands.json',
                     'runner/admin.py', 'runner/bootstrap.py', 'runner/bootstrap_receipt.py',
                     'runner/launch.py', 'runner/pins.py', 'runner/sigterm_cancellation.py',
                     'physical-runner/run.py', 'physical-runner/natural_data.py',
                     'physical-runner/natural_semantic.py', 'physical-runner/modes.py',
                     'physical-runner/joint_data.py', 'physical-runner/pins.py',
                     'physical-runner/qwen_data.py', 'physical-runner/gemma_data.py',
                     'physical-runner/planned-cost.json',
                     'qwen/data-closure.json', 'gemma/data-closure.json')
TEMPLATES = CONTROL_TEMPLATES + PAYLOAD_TEMPLATES
assert len(TEMPLATES) == 24
# Original approved helper/data authority definitions.  Bytes and package
# contracts are preserved; only the caller lifecycle controls are rewritten.
HELPER_PINS = {
    'hydrate.py': {'bytes': 4673,
                   'sha256': '34400794ad4a3da68f7f7032795302ccfc7114143937a436e37ba612ba3e25b0'},
    'runner/pins.py': {'bytes': 4026,
                       'sha256': '81ef68c133b2d532f915b73345e47652d8ef1c3397e7958d755227d556f57711'},
    'physical-runner/qwen_data.py': {'bytes': 5026,
                                     'sha256': '73bf91943fb6eb4efcefc1e5b3065d3235e8c86871e0ef440ff2e8fb37a1c2f4'},
    'physical-runner/gemma_data.py': {'bytes': 5723,
                                      'sha256': '9d6b16d1aa079987244d4528f0f634d52bfa92ed502288253f69eae13b20ea44'},
    'qwen/data-closure.json': {'bytes': 5990,
                               'sha256': 'a3748da811534109d5b48784ba3acc7a695e5278102fa585fa66eee73a3bf685'},
    'gemma/data-closure.json': {'bytes': 5509,
                                'sha256': '9f6296c2eaa18c531e8591b7e7b7bd7370e63a10102367ba8b72bae1a506b014'},
}
MAX_MEMBER = 64*1024**2
MAX_ARCHIVE = 192*1024**2
SCOPE = ('qwen27b natural x2: explicit complete native source root + the eleven-file natural tool, '
         'fresh actual build of release/examples/qwen27b_hip_natural, one Q-only natural '
         'two-position x2 case with the fresh natural full-window reference lease; complete '
         'unit HELD pending ROOT review')


def pin(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def canonical(path, label):
    path = Path(path)
    assert path.is_absolute(), label+' absolute'
    assert Path(os.path.realpath(path)) == path, label+' canonical, no symlink component'
    return path


def overlap(left, right):
    return left == right or left in right.parents or right in left.parents


def safe_rel(name):
    assert type(name) is str and name and not name.startswith('/') and '\\' not in name, \
        'safe relative source member'
    parts = PurePosixPath(name).parts
    assert parts and all(part not in ('', '.', '..') for part in parts), \
        'safe relative source member: '+repr(name)
    assert str(PurePosixPath(name)) == name, 'canonical relative source member: '+repr(name)


def input_collision(path, files, root, tool, manifest):
    path = Path(path)
    for name in files:
        source = root/name
        if path == source or path in source.parents or source in path.parents:
            return 'enumerated source member: '+name
    for candidate, label in ((tool, 'tracked tool template subtree'),
                             (root/'tools', 'source tool subtree'),
                             (root/NATURAL_TOOL, 'natural tool subtree'),
                             (manifest, 'source manifest')):
        if path == candidate or path in candidate.parents or candidate in path.parents:
            return label
    return None


def bounded_total(files, templates):
    source_bytes = sum(meta['bytes'] for meta in files.values())
    template_bytes = sum(templates[name].stat().st_size for name in PAYLOAD_TEMPLATES)
    assert source_bytes + template_bytes <= MAX_ARCHIVE, \
        'declared complete source/template byte total exceeds archive bound before writes'
    return source_bytes + template_bytes


def load_manifest(path):
    value = json.loads(Path(path).read_text())
    assert type(value) is dict, 'ROOT source manifest object'
    files = value.get('files', value)
    assert type(files) is dict and files, 'ROOT source manifest maps members'
    if 'source_count' in value:
        assert type(value['source_count']) is int and value['source_count'] == len(files), \
            'declared source count equals members'
    for name, meta in files.items():
        safe_rel(name)
        assert type(meta) is dict and set(meta) == {'kind', 'bytes', 'mode', 'sha256'}, \
            'source metadata kind/bytes/mode/sha256: '+name
        assert meta['kind'] == 'file', 'regular source member: '+name
        assert type(meta['bytes']) is int and 0 <= meta['bytes'] <= MAX_MEMBER, \
            'bounded source member (64 MiB): '+name
        assert type(meta['mode']) is int and 0 <= meta['mode'] <= 0o7777, 'source mode: '+name
        assert type(meta['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}', meta['sha256']), \
            'source SHA-256: '+name
    assert max(meta['bytes'] for meta in files.values()) <= MAX_MEMBER, '64 MiB member limit'
    natural_keys = {NATURAL_PREFIX+name for name in NATURAL_RUNTIME + NATURAL_EXTRA}
    assert set(name for name in files if name.startswith(NATURAL_PREFIX)) == natural_keys, \
        'complete ROOT manifest includes exactly the eleven natural-prefix members'
    natural = {name: files[name] for name in natural_keys}
    native = {name: meta for name, meta in files.items() if not name.startswith(NATURAL_PREFIX)}
    assert native and natural, 'complete native complement and natural subset'
    return value, files, native, natural


PAYLOAD_HELPERS = None


def accepted_metadata(path, expected_bytes, expected_sha, package):
    """Bounded nofollow same-FD extent/stamp/hash of the caller-pinned receipt."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        assert stat.S_ISREG(before.st_mode), 'regular accepted receipt'
        assert before.st_size == expected_bytes, 'external accepted receipt extent'
        assert 0 < expected_bytes <= 1024**2, 'bounded accepted receipt extent'
        digest = hashlib.sha256()
        raw = bytearray()
        remaining = expected_bytes
        with os.fdopen(fd, 'rb', closefd=False) as file:
            while remaining:
                chunk = file.read(min(1 << 20, remaining))
                assert chunk, 'accepted receipt EOF before pinned extent'
                digest.update(chunk)
                raw.extend(chunk)
                remaining -= len(chunk)
            assert file.read(1) == b'', 'accepted receipt EOF after pinned extent'
        after = os.fstat(fd)
        assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size,
                                        after.st_mtime_ns, after.st_ctime_ns), \
            'accepted receipt stamp stable across one held descriptor'
        assert digest.hexdigest() == expected_sha, 'external accepted receipt SHA-256'
    finally:
        os.close(fd)
    value = json.loads(bytes(raw))
    assert value.get('package') == package, \
        'receipt package is the same frozen nine-file natural authority'
    import sys
    sys.path.insert(0, str(PAYLOAD_HELPERS))
    sys.path.insert(0, str(PAYLOAD_HELPERS.parent/'runner'))
    import natural_data
    natural_data.validate_receipt(value, package)
    artifacts = {name: {k: value[name][k] for k in ('bytes', 'sha256')}
                 for name in natural_data.NATURAL_PINS}
    return value, artifacts, {'bytes': expected_bytes, 'sha256': expected_sha}


def authority_value(tool_pins, accepted=None):
    if accepted is None:
        return {'schema': 'zg.natural-reference-authority.v1',
                'status': 'AWAITING_ACCEPTED_ORIGINAL_CPU_CAPTURE',
                'source': NATURAL_TOOL+' validate --receipt',
                'package': {name: tool_pins[name] for name in NATURAL_RUNTIME},
                'receipt': None, 'artifacts': None}
    _, artifacts, receipt_pin = accepted
    return {'schema': 'zg.natural-reference-authority.v1',
            'status': 'ACCEPTED_ORIGINAL_CPU_CAPTURE',
            'source': NATURAL_TOOL+' validate --receipt',
            'package': {name: tool_pins[name] for name in NATURAL_RUNTIME},
            'receipt': receipt_pin, 'artifacts': artifacts}


def main():
    global PAYLOAD_HELPERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True,
                        help='absolute complete assembled source root')
    parser.add_argument('--source-manifest', type=Path, required=True,
                        help='absolute ROOT relpath->{kind,bytes,mode,sha256} manifest')
    parser.add_argument('--payload', type=Path, required=True,
                        help='absolute fresh output payload directory')
    parser.add_argument('--archive', type=Path, required=True,
                        help='absolute fresh output source archive')
    parser.add_argument('--accepted-receipt', type=Path,
                        help='absolute external accepted original natural receipt')
    parser.add_argument('--accepted-receipt-bytes', type=int,
                        help='caller-supplied accepted receipt extent')
    parser.add_argument('--accepted-receipt-sha256',
                        help='caller-supplied accepted receipt SHA-256')
    args = parser.parse_args()
    supplied = (args.accepted_receipt is not None,
                args.accepted_receipt_bytes is not None,
                args.accepted_receipt_sha256 is not None)
    assert len(set(supplied)) == 1, \
        'accepted receipt operand, extent and SHA are all absent or all present'
    require_receipt = supplied[0]
    tool = canonical(Path(__file__).resolve().parent, 'tracked tool template root')
    source_root = canonical(args.source_root, 'source root')
    manifest_path = canonical(args.source_manifest, 'source manifest')
    payload = canonical(args.payload, 'payload output')
    archive_path = canonical(args.archive, 'archive output')
    assert source_root.is_dir() and (source_root/'AGENTS.md').is_file(), \
        'explicit assembled source root, not a bare directory'
    assert manifest_path.is_file(), 'explicit source manifest file'
    def template(name):
        return tool/'payload'/name if name in PAYLOAD_TEMPLATES else tool/name
    templates = {name: template(name) for name in PAYLOAD_TEMPLATES}
    for name in TEMPLATES:
        assert template(name).is_file(), 'static tracked controller template: '+name
    assert (tool/'README.md').is_file(), 'tracked tool README'
    manifest, files, native_files, natural_files = load_manifest(manifest_path)
    manifest_pin = pin(manifest_path)
    # Validate every output path before the first write.  Fresh outputs under
    # an ignored source_root/.agents/runs directory are allowed; only actual
    # enumerated source/template/manifest collisions are refused.
    assert not payload.exists(), 'fresh payload output must not exist'
    assert not archive_path.exists(), 'fresh archive output must not exist'
    prepared_path = payload.parent/'PREPARED.json'
    assert not prepared_path.exists(), 'fresh PREPARED receipt'
    assert payload != prepared_path, 'payload is not the PREPARED receipt'
    assert archive_path != prepared_path, 'archive is not the PREPARED receipt'
    assert not overlap(payload, archive_path), 'non-overlapping payload/archive outputs'
    assert not overlap(payload, prepared_path) and not overlap(archive_path, prepared_path), \
        'outputs disjoint from the PREPARED receipt'
    for path, label in ((payload, 'payload output'), (archive_path, 'archive output'),
                        (prepared_path, 'PREPARED output')):
        hit = input_collision(path, files, source_root, tool, manifest_path)
        assert hit is None, label+' collides with '+str(hit)
    # The declared complete source+template bound is checked before first write.
    declared_total = bounded_total(files, templates)
    accepted_path = canonical(args.accepted_receipt, 'accepted receipt') if require_receipt else None
    if require_receipt:
        assert accepted_path.is_file(), 'external accepted receipt file'
        assert 0 < args.accepted_receipt_bytes <= 1024**2, 'bounded accepted receipt extent'
        assert type(args.accepted_receipt_sha256) is str and re.fullmatch(
            r'[0-9a-f]{64}', args.accepted_receipt_sha256), 'accepted receipt SHA-256 operand'
        hit = input_collision(accepted_path, files, source_root, tool, manifest_path)
        assert hit is None, 'accepted receipt collides with '+str(hit)
        assert not overlap(accepted_path, payload) and not overlap(accepted_path, archive_path), \
            'accepted receipt is external to outputs (no self-pin)'
        assert not overlap(accepted_path, prepared_path), 'accepted receipt is not the prepared receipt'
    # Validate actual source bytes AND modes against the SAME complete map,
    # before any output write.  Natural-prefix records are part of this map.
    for name, expected in files.items():
        source = source_root/name
        assert source.is_file() and not source.is_symlink(), 'regular source member: '+name
        actual = source.stat(follow_symlinks=False)
        assert stat.S_IMODE(actual.st_mode) == expected['mode'], \
            'source mode before any output write: '+name
        assert pin(source) == {'bytes': expected['bytes'], 'sha256': expected['sha256']}, \
            'source bytes before any output write: '+name
    tool_source = source_root/NATURAL_TOOL
    assert tool_source.is_dir() and not tool_source.is_symlink(), 'natural tool source root'
    for name in NATURAL_RUNTIME + NATURAL_EXTRA:
        assert (tool_source/name).is_file() and not (tool_source/name).is_symlink(), \
            'natural tool member: '+name
    source_tool_pins = {name: pin(tool_source/name) for name in NATURAL_RUNTIME + NATURAL_EXTRA}
    for name in NATURAL_RUNTIME + NATURAL_EXTRA:
        key = NATURAL_PREFIX+name
        assert natural_files[key] == {'kind': 'file', 'bytes': source_tool_pins[name]['bytes'],
                                      'mode': stat.S_IMODE((tool_source/name).stat(follow_symlinks=False).st_mode),
                                      'sha256': source_tool_pins[name]['sha256']}, \
            'natural source record is the same held byte/mode authority: '+key
    # The declared complete source+template bound is checked before first write.
    # Validate the external receipt against the source package before any write.
    accepted = None
    if require_receipt:
        PAYLOAD_HELPERS = tool/'payload/physical-runner'
        accepted = accepted_metadata(
            accepted_path, args.accepted_receipt_bytes, args.accepted_receipt_sha256,
            {name: source_tool_pins[name] for name in NATURAL_RUNTIME})

    now = time.time_ns()
    assert pin(manifest_path) == manifest_pin, 'ROOT source manifest stable across preparation'
    payload.mkdir(parents=True, mode=0o700)
    for name in PAYLOAD_TEMPLATES:
        source = templates[name]
        staged = payload/name
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, staged)
        os.chmod(staged, 0o644)
        os.utime(staged, ns=(now, now), follow_symlinks=False)
        assert pin(staged) == pin(source), 'static controller template bytes: '+name
        assert stat.S_IMODE(staged.stat(follow_symlinks=False).st_mode) == 0o644, \
            'static controller template mode: '+name
    # Copy the complete manifest once: native complement and natural prefix from
    # the SAME records.  No second generated authority or overwrite.
    for name, expected in files.items():
        source = source_root/name
        staged = payload/'source'/name
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, staged)
        os.chmod(staged, expected['mode'])
        os.utime(staged, ns=(now, now), follow_symlinks=False)
        assert pin(staged) == {'bytes': expected['bytes'], 'sha256': expected['sha256']}, \
            'source bytes: '+name
        assert stat.S_IMODE(staged.stat(follow_symlinks=False).st_mode) == expected['mode'], \
            'source mode: '+name
    tool_pins = {name: pin(payload/'source'/NATURAL_PREFIX/name)
                 for name in NATURAL_RUNTIME + NATURAL_EXTRA}
    assert len(NATURAL_RUNTIME) == 9 and len(NATURAL_EXTRA) == 2 and len(tool_pins) == 11
    assert set(tool_pins) == set(NATURAL_RUNTIME) | set(NATURAL_EXTRA)
    assert tool_pins == source_tool_pins, 'staged natural tool bytes equal source package'
    native_count = len(native_files)
    assert len(files) == native_count + len(tool_pins), 'derived native+natural union'
    assert len([k for k in files if not k.startswith(NATURAL_PREFIX)]) == native_count
    assert len([k for k in files if k.startswith(NATURAL_PREFIX)]) == len(tool_pins)
    assert files == manifest.get('files', manifest), 'prepared union equals complete ROOT map'
    assert declared_total == sum(meta['bytes'] for meta in files.values()) + \
        sum(templates[name].stat().st_size for name in PAYLOAD_TEMPLATES)
    for name, expected in HELPER_PINS.items():
        assert pin(payload/name) == expected, 'original approved helper/DataClosure: '+name
    assert pin(payload/'runner/sigterm_cancellation.py')['sha256'] == CANCEL_SHA, \
        'original approved cancellation guard'
    assert pin(payload/'source'/NATURAL_PREFIX/'owned_container_v4.py')['sha256'] == OWNER_SHA, \
        'original approved owner helper'

    natural_authority = authority_value(tool_pins, accepted)
    if require_receipt:
        natural_authority['external_operand'] = str(accepted_path)
    dump(payload/'natural-authority.json', natural_authority)
    root_manifest_pin = manifest_pin
    assembled = {'schema': 'zg.hip-source20-natural-x2-assembled-source.v1',
                 'files': files, 'source_count': len(files), 'implementation_qualified': False,
                 'native_source_count': native_count,
                 'natural_source_count': len(tool_pins),
                 'root_source_manifest': root_manifest_pin}
    dump(payload/'assembled-source.json', assembled)
    source_manifest_pin = pin(payload/'assembled-source.json')
    assembled_view = json.loads((payload/'assembled-source.json').read_text())
    assert assembled_view['source_count'] == len(assembled_view['files'])
    assert len([k for k in assembled_view['files'] if not k.startswith(NATURAL_PREFIX)]) == native_count
    assert len([k for k in assembled_view['files'] if k.startswith(NATURAL_PREFIX)]) == len(tool_pins)

    def inventory():
        return {str(p.relative_to(payload)): p for p in sorted(payload.rglob('*')) if p.is_file()}

    files_on_disk = inventory()
    assert (payload/'check-commands.json') in files_on_disk.values()
    assert not (payload/'source-closure.json').exists()
    for stale in ('runner/checks.py', 'runner/environment.py', 'runner/modes.py',
                  'runner/run.py', 'runner/source-expressions.py'):
        assert stale not in files_on_disk, 'no stale predecessor controller: '+stale
    control = {'schema': 'zg.hip-source20-natural-x2-controller.v1', 'base': BASE,
               'predecessor_bases': list(PREDECESSOR_BASES),
               'source_manifest': source_manifest_pin,
               'records': {name: pin(p) for name, p in files_on_disk.items() if name != 'control.json'}}
    dump(payload/'control.json', control)
    control_pin = pin(payload/'control.json')
    ship = inventory()
    assert set(ship) == set(files_on_disk) | {'control.json'}
    closure = {'records': {name: pin(p) for name, p in ship.items()},
               'files': 0, 'bytes': 0, 'source_count': len(files),
               'source_manifest': source_manifest_pin,
               'root_source_manifest': root_manifest_pin,
               'parent_frozen_source_manifest': root_manifest_pin,
               'native_source_count': native_count,
               'natural_source_count': len(tool_pins), 'scope': SCOPE}
    closure['files'] = len(closure['records'])
    closure['bytes'] = sum(v['bytes'] for v in closure['records'].values())
    assert set(closure['records']) == set(ship) - {'source-closure.json'}
    dump(payload/'source-closure.json', closure)
    members = inventory()
    assert set(members) == set(closure['records']) | {'source-closure.json'}
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, 'x', format=tarfile.GNU_FORMAT) as tar:
        for name, path in sorted(members.items()):
            info = tar.gettarinfo(str(path), arcname=name)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            with path.open('rb') as file:
                tar.addfile(info, file)
    seen = set()
    with tarfile.open(archive_path) as tar:
        for member in tar:
            assert member.isfile() and member.name in members and member.name not in seen
            path = members[member.name]
            data = tar.extractfile(member).read()
            assert len(data) == path.stat().st_size
            assert hashlib.sha256(data).hexdigest() == pin(path)['sha256']
            assert member.mode == stat.S_IMODE(path.stat().st_mode)
            seen.add(member.name)
    assert seen == set(members)
    assert sum(path.stat().st_size for path in members.values()) <= MAX_ARCHIVE
    prepared = {'status': 'QWEN27B_NATURAL_X2_UNIT_PREPARED_REMOTE_HELD',
                'base': BASE, 'predecessor_bases': list(PREDECESSOR_BASES),
                'source_root': str(payload/'source'),
                'source_root_operand': str(source_root),
                'source_files': len(files),
                'native_source_files': native_count,
                'natural_source_files': len(tool_pins),
                'source_bytes': sum(v['bytes'] for v in files.values()),
                'source_manifest': source_manifest_pin,
                'root_source_manifest': root_manifest_pin,
                'root_source_manifest_operand': str(manifest_path),
                'source_closure': pin(payload/'source-closure.json'),
                'controller': control_pin, 'source_archive': pin(archive_path),
                'source_archive_operand': str(archive_path),
                'payload_operand': str(payload),
                'tar_members': len(seen), 'payload_files': len(members),
                'all_source_bytes_modes_verified': True,
                'all_payload_bytes_modes_verified': True,
                'fresh_source_manifest_used': True,
                'source_union_basis': 'fresh_complete_root_manifest_native_plus_natural',
                'source_union_is_native_plus_natural': True,
                'source_union_derived_count': len(files),
                'root_source_manifest_complete': True,
                'root_source_manifest_natural_members': len(tool_pins),
                'declared_source_template_bytes_bound': declared_total,
                'archive_exclusive_create': True,
                'natural_tool_package_files': len(tool_pins),
                'natural_tool_package_pins': tool_pins,
                'natural_authority': pin(payload/'natural-authority.json'),
                'natural_authority_status': natural_authority['status'],
                'natural_capture_pins_absent': accepted is None,
                'accepted_receipt_operand': (str(accepted_path) if require_receipt else None),
                'no_payload_transfer_of_data_models_caches': True,
                'GPU_or_model': False, 'remote_effect': False}
    dump(payload.parent/'PREPARED.json', prepared)
    print(json.dumps({k: v for k, v in prepared.items() if k != 'natural_tool_package_pins'}, indent=2))


if __name__ == '__main__':
    main()

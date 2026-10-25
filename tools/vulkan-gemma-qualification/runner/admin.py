#!/usr/bin/env python3
"""Create one fresh local Vulkan Gemma source namespace from ROOT's manifest."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import sys
import time
import tomllib

sys.dont_write_bytecode = True

from pins import Held, alarm_until, check_closure, pin, require, stamp
from sigterm_cancellation import SigtermCancellation

BASE = 'resume-all-five-vulkan-20261025-2'
OWNER = Path(__file__).resolve().parent.parent
MANIFEST_LIMIT = 16 * 1024**2
SOURCE_MEMBER_LIMIT = 64 * 1024**2
SOURCE_TOTAL_LIMIT = 192 * 1024**2
WORK_SECONDS = 1830
FINAL_SECONDS = 120
RECEIPT_SECONDS = 5
REQUIRED_CONFIG = ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml',
                   'scripts/native-gpu.py', 'README.md')
IGNORE = ('.git', '.agents', 'target')
IGNORE_SUFFIX = ()
INCLUDE = re.compile(r'\binclude(?:_str|_bytes)?!\s*\(\s*"([^"]+)"')
PORTABLE_BASE = Path('/home/pvl/.cache/abird/zerogravity/qualification/portable-service-20261023-1')
PORTABLE_SHA = '42ac6bf6f82e61adfe1cff7c565076361f222c51792f02df09337ff1779a791a'
REGISTRY = 'index.crates.io-1949cf8c6b5b557f'
REGISTRY_MEMBER_LIMIT = 64 * 1024**2


class ManagedSignals:
    """Record TERM/INT/HUP/ALRM across local staging and every mandatory close."""
    SIGNALS = (signal.SIGHUP, signal.SIGALRM)

    def __init__(self, receipt, prefix):
        self.cancel = SigtermCancellation()
        self.receipt, self.prefix = receipt, prefix
        self.pending_alarm = None
        self.depth = 0
        self.previous = {}

    def __enter__(self):
        try:
            self.cancel.__enter__()
            for signum in self.SIGNALS:
                self.previous[signum] = signal.signal(signum, self._handle)
        except BaseException:
            # Partial guard setup restores both term signals and HUP/ALRM.
            try:
                self.restore()
            except BaseException:
                signal.setitimer(signal.ITIMER_REAL, 0)
            raise
        return self

    def __exit__(self, kind, value, traceback):
        self.restore()
        return False

    def _handle(self, signum, _frame):
        if signum == signal.SIGALRM:
            error = TimeoutError(self.prefix + ' phase deadline')
            if self.depth:
                self.pending_alarm = error
            else:
                raise error
        else:
            self.cancel.received = True

    @contextmanager
    def critical(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def check(self):
        if self.pending_alarm is not None:
            error = self.pending_alarm
            self.pending_alarm = None
            raise error
        if self.cancel.received:
            self.cancel.received = False
            raise self.cancel.exception()

    def note(self):
        if self.pending_alarm is not None:
            self.receipt['errors'].append(repr(self.pending_alarm))
            self.pending_alarm = None
        if self.cancel.received:
            self.receipt['errors'].append(repr(self.cancel.exception()))
            self.cancel.received = False

    def restore(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        first_error = None
        for signum, handler in list(self.previous.items()):
            try:
                signal.signal(signum, handler)
            except BaseException as error:
                if first_error is None:
                    first_error = error
            else:
                self.previous.pop(signum, None)
        try:
            self.cancel.__exit__(None, None, None)
        except BaseException as error:
            if first_error is None:
                first_error = error
        else:
            self.cancel.previous.clear()
        if first_error is not None:
            raise first_error


def write_receipt(path, receipt, deadline):
    """One receipt attempt inside the shared absolute finalization deadline."""
    alarm_until(deadline)
    path.write_text(json.dumps(receipt, indent=2) + '\n')


def pin_bytes(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def safe_rel(name):
    p = Path(name)
    require(isinstance(name, str) and not p.is_absolute() and p.parts
            and '..' not in p.parts and p.as_posix() == name,
            'safe contained relative path: '+str(name))
    return p


def mode_of(st):
    return stat.S_IMODE(st.st_mode)


def read_manifest(held):
    raw = held.file.read()
    require(len(raw) <= MANIFEST_LIMIT, 'bounded complete source manifest')
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


def payload_inventory(source_root, source_real):
    """Regular-file set under the same ignored-directory convention as staging."""
    seen = set()
    for path in sorted(source_root.rglob('*')):
        rel = path.relative_to(source_root)
        if any(part in IGNORE for part in rel.parts) \
                or rel.suffix in IGNORE_SUFFIX:
            continue
        require(not path.is_symlink(), 'no source symlink: '+rel.as_posix())
        real = Path(os.path.realpath(path))
        require(real.is_relative_to(source_real),
                'contained source: '+rel.as_posix())
        if path.is_dir():
            continue
        require(path.is_file(), 'regular source file: '+rel.as_posix())
        seen.add(rel.as_posix())
    return seen


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
    """Mechanical Cargo/path-dependency/build-script/include closure."""
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
        need(contained(Path('.'), member, 'workspace member')+'/Cargo.toml',
             'workspace member')
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


def index_relative(name):
    name = name.lower()
    if len(name) <= 2:
        return Path(str(len(name)))/name
    if len(name) == 3:
        return Path('3')/name[0]/name
    return Path(name[:2])/name[2:4]/name


def hydrate_registry(guard, base, source, closure, manifest_pin, closure_pin,
                     portable_held, lock_held):
    """Inline G hydrator over V's unprefixed staged source and own Cargo home."""
    require(portable_held.proof['sha256'] == PORTABLE_SHA,
            'original approved portable closure changed')
    portable_held.identity()
    old = json.load(portable_held.file)
    portable_held.file.seek(0)
    require(isinstance(old, dict) and isinstance(old.get('records'), dict)
            and old['records'], 'original portable closure records')
    require(not (base/'cargo').exists() and not (base/'cargo').is_symlink(),
            'fresh owned Cargo home')
    source_records = closure['records']
    for rel, expected in source_records.items():
        guard.check()
        require(pin(source/safe_rel(rel), limit=SOURCE_MEMBER_LIMIT) == expected,
                'source closure mismatch: '+rel)
    guard.check()
    lock_held.identity()
    lock_raw = lock_held.file.read()
    lock_held.file.seek(0)
    require(len(lock_raw) <= SOURCE_MEMBER_LIMIT, 'bounded staged Cargo.lock')
    require(lock_held.proof == source_records.get('Cargo.lock'),
            'staged Cargo.lock closure join')
    lock = tomllib.loads(lock_raw.decode('utf-8'))
    packages = lock.get('package')
    require(isinstance(packages, list), 'locked package table')
    keys = {}
    names = set()
    for pkg in packages:
        guard.check()
        if not pkg.get('source'):
            continue
        require(pkg['source'] == 'registry+https://github.com/rust-lang/crates.io-index',
                'original locked registry only')
        key = f"cargo/registry/cache/{REGISTRY}/{pkg['name']}-{pkg['version']}.crate"
        require(key in old['records'], 'approved registry archive missing: '+key)
        require(old['records'][key].get('sha256') == pkg.get('checksum'),
                'locked archive authority differs')
        keys[key] = old['records'][key]
        names.add(pkg['name'])
    for name in names:
        key = f'cargo/registry/index/{REGISTRY}/.cache/{index_relative(name)}'
        require(key in old['records'], 'approved registry index missing: '+key)
        keys[key] = old['records'][key]
    config = f'cargo/registry/index/{REGISTRY}/config.json'
    require(config in old['records'], 'approved registry config missing: '+config)
    keys[config] = old['records'][config]
    require(keys, 'non-empty locked registry selection')
    registry_records = {}
    for rel, expected in sorted(keys.items()):
        guard.check()
        require(isinstance(expected, dict)
                and set(expected) == {'bytes', 'sha256'}
                and type(expected['bytes']) is int
                and 0 <= expected['bytes'] <= REGISTRY_MEMBER_LIMIT,
                'bounded portable registry record: '+rel)
        source_path = portable_held.path.parent/safe_rel(rel)
        target = base/safe_rel(rel)
        require(not target.exists() and not target.is_symlink(),
                'fresh registry member: '+rel)
        before = source_path.stat(follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode)
                and 0 <= before.st_size <= REGISTRY_MEMBER_LIMIT,
                'bounded regular registry member: '+rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        fd = os.open(source_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            require(stamp(os.fstat(fd)) == stamp(before),
                    'registry input identity: '+rel)
            with os.fdopen(fd, 'rb') as stream:
                fd = None
                with target.open('xb') as out:
                    while True:
                        guard.check()
                        block = stream.read(1024**2)
                        if not block:
                            break
                        digest.update(block)
                        out.write(block)
                require(stamp(os.fstat(stream.fileno())) == stamp(before),
                        'registry input changed: '+rel)
        finally:
            if fd is not None:
                os.close(fd)
        require(stamp(source_path.stat(follow_symlinks=False)) == stamp(before),
                'registry input changed: '+rel)
        actual = {'bytes': before.st_size, 'sha256': digest.hexdigest()}
        require(actual == expected, 'approved registry bytes differ: '+rel)
        target.chmod(mode_of(before))
        require(pin(target, limit=REGISTRY_MEMBER_LIMIT) == actual,
                'copied registry member bytes: '+rel)
        registry_records[rel] = actual
    portable_held.after()
    lock_held.after()
    guard.check()
    registry = {'schema': 'zg.vulkan-gemma-registry-closure.v1',
                'files': len(registry_records),
                'bytes': sum(v['bytes'] for v in registry_records.values()),
                'packages': len(names),
                'source_manifest': manifest_pin,
                'source_closure': closure_pin,
                'source_count': closure.get('source_count'),
                'cargo_lock': lock_held.proof,
                'portable_authority': {'base': str(PORTABLE_BASE),
                                       'closure': portable_held.proof},
                'records': registry_records}
    path = base/'registry-closure.json'
    require(not path.exists() and not path.is_symlink(),
            'fresh registry closure receipt')
    with path.open('x') as file:
        json.dump(registry, file, indent=2)
        file.write('\n')
    return registry, pin(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.payload.is_absolute() and args.payload.is_dir(),
            'explicit absolute assembled source root')
    require(args.source_manifest.is_absolute() and args.source_manifest.is_file()
            and not args.source_manifest.is_symlink(),
            'explicit absolute ROOT source manifest')
    require(args.base.is_absolute() and args.output.is_absolute()
            and not args.output.exists(), 'absolute fresh admin output')
    require(args.base.parent == Path('/home/pvl/.cache/abird/zerogravity/qualification')
            and args.base.name == BASE and not args.base.exists(),
            'fresh exact owned Vulkan namespace')
    require(not args.base.is_relative_to(args.payload)
            and not args.output.is_relative_to(args.payload),
            'fresh output/base outside the source root')
    args.output.mkdir(parents=True, mode=0o700)
    summary = {'status': 'UNQUALIFIED', 'errors': [], 'steps': []}
    summary['local_work_seconds'] = WORK_SECONDS
    summary['local_finalization_seconds'] = FINAL_SECONDS
    summary['local_receipt_seconds'] = RECEIPT_SECONDS
    started = time.monotonic()
    work_end = started + WORK_SECONDS
    guard = ManagedSignals(summary, 'admin')
    held = []
    closure = control = registry = None
    closure_pin = control_pin = manifest_pin = registry_pin = None
    try:
        guard.__enter__()
        alarm_until(work_end)
        payload_root = args.payload
        payload_real = Path(os.path.realpath(payload_root))
        with guard.critical():
            # Acquire and register the held manifest inside the guarded section:
            # an ALRM cannot abandon an acquired FD before it is in `held`.
            manifest_held = Held(args.source_manifest, limit=MANIFEST_LIMIT)
            held.append(manifest_held)
            manifest, manifest_raw = read_manifest(manifest_held)
            files = manifest['files']
            manifest_pin = pin_bytes(manifest_raw)
            inventory = payload_inventory(payload_root, payload_real)
            require(inventory == set(files),
                    'payload source set differs from the supplied ROOT manifest')
            for name in REQUIRED_CONFIG:
                require(name in files, 'required workspace/build/config input: '+name)
            for name in files:
                require(name != '.git' and not name.startswith('.git/'),
                        'staged source without .git')
            for name in ('.cargo/config.toml', '.cargo/config', 'rust-toolchain'):
                if (payload_root/name).is_file():
                    require(name in files, 'required Cargo/toolchain config input: '+name)
            if 'flake.nix' in files:
                require('flake.lock' in files, 'flake.lock with flake.nix')
            cargo_inputs(payload_root, payload_real, files)
            guard.check()
            now = time.time_ns()
            args.base.mkdir(mode=0o700)
            source_out = args.base/'source'
            source_out.mkdir(mode=0o700)
            total = 0
            for name, entry in files.items():
                guard.check()
                rel = safe_rel(name)
                raw = source_record(payload_root, payload_real, name, entry)
                staged = source_out/rel
                staged.parent.mkdir(parents=True, exist_ok=True)
                require(not staged.exists() and not staged.is_symlink(),
                        'fresh staged source: '+name)
                shutil.copyfile(payload_root/rel, staged)
                os.chmod(staged, entry['mode'])
                os.utime(staged, ns=(now, now))
                require(pin(staged) == {k: entry[k] for k in ('bytes', 'sha256')},
                        'staged source bytes: '+name)
                require(mode_of(staged.stat()) == entry['mode'],
                        'staged source mode: '+name)
                total += len(raw)
            require(total == sum(entry['bytes'] for entry in files.values()),
                    'derived complete source bytes')
            require(total <= SOURCE_TOTAL_LIMIT, 'bounded complete source bytes')
            guard.check()
            staged_files = payload_inventory(
                source_out, Path(os.path.realpath(source_out)))
            require(staged_files == set(files),
                    'staged source set differs from the supplied ROOT manifest')
            assembled = args.base/'assembled-source.json'
            guard.check()
            require(not assembled.exists(), 'fresh assembled manifest copy')
            assembled.write_bytes(manifest_raw)
            os.chmod(assembled, 0o644)
            os.utime(assembled, ns=(now, now))
            require(pin(assembled) == manifest_pin,
                    'exact supplied manifest bytes copied')
            records = {name: pin(source_out/safe_rel(name)) for name in files}
            require(set(records) == set(files), 'source closure exact manifest union')
            for name, entry in files.items():
                require(records[name] == {k: entry[k] for k in ('bytes', 'sha256')},
                        'source closure pin: '+name)
            closure = {'schema': 'zg.vulkan-gemma-source-closure.v1',
                       'files': len(records),
                       'bytes': sum(v['bytes'] for v in records.values()),
                       'source_count': len(files),
                       'source_manifest': manifest_pin,
                       'records': records}
            (args.base/'source-closure.json').write_text(
                json.dumps(closure, indent=2) + '\n')
            closure_pin = pin(args.base/'source-closure.json')
            # The G hydrator's source records were base-relative; V's complete
            # source closure is unprefixed, so every record is checked at the
            # staged source root and the source receipt is never rewritten.
            portable_held = Held(PORTABLE_BASE/'closure.json',
                                 limit=MANIFEST_LIMIT)
            held.append(portable_held)
            lock_held = Held(source_out/'Cargo.lock',
                             limit=SOURCE_MEMBER_LIMIT)
            held.append(lock_held)
            registry, registry_pin = hydrate_registry(
                guard, args.base, source_out, closure, manifest_pin,
                closure_pin, portable_held, lock_held)
            control = {'schema': 'zg.vulkan-gemma-controller.v1',
                       'base': BASE, 'mode': 'vulkan-gemma-e2b-x2',
                       'source_manifest': manifest_pin,
                       'source_manifest_path': str(args.source_manifest),
                       'source_root': str(args.payload),
                       'source_count': len(files), 'changed_paths': {},
                       'source_closure': closure_pin,
                       'registry_closure': registry_pin,
                       'records': {'source-closure.json': closure_pin,
                                   'registry-closure.json': registry_pin}}
            with (args.base/'control.json').open('x') as file:
                json.dump(control, file, indent=2)
                file.write('\n')
            control_pin = pin(args.base/'control.json')
            guard.check()
            for name, source in (('gemma', OWNER/'data/gemma-data-closure.json'),
                                 ('qwen', OWNER/'data/qwen-data-closure.json')):
                guard.check()
                source_held = Held(source)
                held.append(source_held)
                target = args.base/name/'data-closure.json'
                target.parent.mkdir(mode=0o700)
                shutil.copy2(source, target)
                require(pin(target) == source_held.proof,
                        'exact accepted data authority copy')
                summary['steps'].append({'name': name,
                                         'data_closure': source_held.proof})
        guard.check()
        check_closure(args.base/'source', closure)
        summary.update(source_closure=closure_pin, control=control_pin,
                       source_manifest=manifest_pin, files=len(records),
                       source_count=len(files), bytes=closure['bytes'],
                       registry_closure=registry_pin,
                       registry_files=registry['files'],
                       registry_bytes=registry['bytes'])
        summary['status'] = 'CLOSED_PASS_SCOPED'
    except BaseException as error:
        summary['errors'].append(repr(error))
    finally:
        try:
            with guard.critical():
                signal.setitimer(signal.ITIMER_REAL, 0)
                final_end = min(time.monotonic(), work_end) + FINAL_SECONDS
                for item in held:
                    try:
                        alarm_until(final_end)
                        item.after()
                    except BaseException as error:
                        summary['errors'].append('held: '+repr(error))
                    finally:
                        try:
                            item.close()
                        except BaseException as error:
                            summary['errors'].append('held close: '+repr(error))
                try:
                    alarm_until(final_end)
                    require(control_pin is not None
                            and pin(args.base/'control.json') == control_pin,
                            'same local controller authority')
                    check_closure(args.base, control)
                except BaseException as error:
                    summary['errors'].append('finalization: '+repr(error))
        finally:
            guard.note()
            summary['seconds'] = time.monotonic() - started
            summary['status'] = ('CLOSED_PASS_SCOPED'
                                 if not summary['errors'] else 'FAILED_CLOSED')
            receipt_end = time.monotonic() + RECEIPT_SECONDS
            try:
                try:
                    write_receipt(args.output/'invocation.json', summary,
                                  receipt_end)
                except BaseException as error:
                    # Only the remaining part of the same absolute receipt budget.
                    summary['errors'].append('receipt: '+repr(error))
                    summary['status'] = 'FAILED_CLOSED'
                    try:
                        write_receipt(args.output/'invocation.json', summary,
                                      receipt_end)
                    except BaseException as retry:
                        summary['errors'].append('receipt retry: '+repr(retry))
                        summary['status'] = 'FAILED_CLOSED'
                try:
                    # Final stdout shares that deadline; handlers/timer remain
                    # installed through print/flush.
                    alarm_until(receipt_end)
                    print(json.dumps({'status': summary['status'],
                                      'errors': summary['errors']},
                                     indent=2), flush=True)
                except BaseException as error:
                    summary['errors'].append('stdout: '+repr(error))
                    summary['status'] = 'FAILED_CLOSED'
            finally:
                guard.restore()
    return (0 if summary['status'] == 'CLOSED_PASS_SCOPED'
            and not summary['errors'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())

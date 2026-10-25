"""Original read-only Gemma data FDs; reused canonical objects permit declared lock writes only."""
import json
from pathlib import Path

from pins import Held, relative, require

DATA_ROOT = Path('/home/pvl/.cache/abird/zerogravity/qualification/gemma4-hip-main314-v5-2/data')
DATA_PROOF = {'bytes': 5509, 'sha256': '9f6296c2eaa18c531e8591b7e7b7bd7370e63a10102367ba8b72bae1a506b014'}
SNAPSHOT = 'ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1'
INVOCATION = 'be800bc6e97015290b12bb8f1b756ff9490db34ee905b920937236a0b2f4dbe3'


def validate_authority(value):
    records = value['records']
    require(value['schema'] == 'zg.gemma4-exact-approved-data.v1'
            and value['snapshot'] == SNAPSHOT and value['expected_invocation_sha256'] == INVOCATION
            and value['source_file_count'] == 8 and value['expected_file_count'] == 26
            and value['new_download'] is False and len(records) == 35,
            'exact original Gemma35 source/corpus authority')
    require(records['source/manifest.json'] == {'bytes':1547, 'sha256':SNAPSHOT}
            and records['expected/invocation.json'] == {'bytes':10900, 'sha256':INVOCATION}
            and sum(v['bytes'] for k,v in records.items() if k.startswith('source/files/')) == 10278848001
            and sum(v['bytes'] for k,v in records.items() if k.startswith('expected/')) == 76721487,
            'same complete source/corpus extents')
    for name, proof in records.items():
        relative(name)
        require((name == 'source/manifest.json' or name.startswith(('source/files/', 'expected/')))
                and set(proof) == {'bytes','sha256'} and type(proof['bytes']) is int
                and 0 <= proof['bytes'] <= 16*1024**3 and len(proof['sha256']) == 64,
                'original data member proof')


def validate_tree(root, records):
    require(root.is_absolute() and root == DATA_ROOT and root.resolve(strict=True) == root,
            'fixed original read-only data root, no alias')
    seen = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'no data symlinks')
        if path.is_dir():
            require(path.relative_to(root).as_posix() in ('source','source/files','expected'),
                    'no extra data directories')
        else:
            require(path.is_file(), 'regular original data member')
            seen.add(path.relative_to(root).as_posix())
    require(seen == set(records), 'exact35 data member union')


class DataLease:
    def __init__(self, base, root):
        self.root = root
        self.authority = None
        self.held = []
        self.records = None
        try:
            self.authority = Held(base/'data-closure.json', DATA_PROOF, limit=1024**2)
            value = json.load(self.authority.file)
            self.authority.identity()
            validate_authority(value)
            self.records = value['records']
            validate_tree(root, self.records)
            for name, proof in sorted(self.records.items()):
                self.held.append(Held(root/relative(name), proof))
            require(len(self.held) == 35, 'all35 original FDs held')
            self.authority.after()
        except BaseException:
            self.close()
            raise

    def bindings(self):
        return [(item.file.fileno(), str(item.path)) for item in self.held]

    def authority_after(self):
        validate_tree(self.root, self.records)
        self.authority.after()

    def after(self):
        # Caller preserves each error and closes all original FDs independently.
        validate_tree(self.root, self.records)
        self.authority.after()
        for item in self.held:
            item.after()

    def close(self):
        for item in self.held:
            item.close()
        if self.authority is not None:
            self.authority.close()


def enclosed_build(base, source, command):
    """Same original GPU-free mount grammar; new offline source/target only."""
    require(base.is_absolute() and source == base/'source', 'owned source build root')
    return ['/run/current-system/sw/bin/bwrap', '--unshare-pid', '--die-with-parent',
            '--ro-bind', '/', '/', '--bind', str(base), str(base), '--dev', '/dev',
            '--proc', '/proc', '--tmpfs', '/tmp', '--chdir', str(source), '--', *command]


def enclosed(base, source, binary, data, command, gpu, output=None, cache_locks=None):
    """Only NEWbase/fresh owned report directory writable; bind original held data/binary FDs read-only."""
    argv = ['/run/current-system/sw/bin/bwrap']
    if not gpu:
        argv += ['--unshare-user', '--unshare-net']
    # GPU invocation already enters native-gpu.py's mandatory user/net namespace.
    argv += ['--unshare-pid', '--die-with-parent', '--ro-bind', '/', '/',
             '--bind', str(base), str(base)]
    if gpu:
        require(output is not None and output.parent == Path('/home/pvl/.local/state/abird/zerogravity')/base.name,
                'fresh original owned report directory')
        argv += ['--bind', str(output), str(output)]
    argv += ['--dev-bind', '/dev', '/dev'] if gpu else ['--dev', '/dev']
    if cache_locks is not None:
        require(gpu and cache_locks.is_absolute() and cache_locks.resolve(strict=True) == cache_locks
                and cache_locks.name == "locks", "only exact original cache locks writable")
        argv += ["--bind", str(cache_locks), str(cache_locks)]
    argv += ['--proc', '/proc', '--tmpfs', '/tmp', '--ro-bind-fd',
             str(binary.file.fileno()), '/tmp/zg-vulkan-gemma-held-binary']
    for fd, path in data.bindings():
        argv += ['--ro-bind-fd', str(fd), path]
    return argv + ['--chdir', str(source), '--', *command]

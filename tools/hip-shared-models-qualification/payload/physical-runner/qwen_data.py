"""Original read-only approved Qwen data FDs; new namespace only owns canonical import."""
import json
from pathlib import Path

from pins import Held, relative, require

DATA_ROOT = Path('/home/pvl/.cache/abird/zerogravity/qualification/qwen27b-hip-main400-v1-source466-20261024-1/data')
DATA_PROOF = {'bytes': 5990, 'sha256': 'a3748da811534109d5b48784ba3acc7a695e5278102fa585fa66eee73a3bf685'}
SNAPSHOT = '98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e'
ACCEPTED = '2a81216ede102c8b0c22c082332f62f38bf0bb2f326298e519f34ed20fc3e3d7'
RECEIVED_PROOF = {'bytes':403,'sha256':'50597c7809028a8be842cc31c663a558ade1ad6b05736a9a59b97e342a8d8e61'}


def validate_authority(value):
    records = value['records']
    require(value['schema'] == 'zg.qwen27b-exact-approved-data.v1'
            and value['snapshot'] == SNAPSHOT and value['expected_accepted_sha256'] == ACCEPTED
            and value['source_file_count'] == 31 and value['expected_file_count'] == 3
            and value['new_download'] is False and len(records) == 35,
            'exact original Qwen35 source/corpus authority')
    require(records['source/manifest.json'] == {'bytes':5849, 'sha256':SNAPSHOT}
            and records['expected/accepted-original-full-window-1.json'] == {'bytes':13214, 'sha256':ACCEPTED}
            and sum(v['bytes'] for k,v in records.items() if k.startswith('source/files/')) == 55586113293
            and sum(v['bytes'] for k,v in records.items() if k.startswith('expected/')) == 935824688,
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
        self.received = None
        try:
            self.received = Held(DATA_ROOT.parent/'data-received.json', RECEIVED_PROOF, limit=1024**2)
            received = json.load(self.received.file)
            require(received['status'] == 'EXACT_APPROVED_DATA_RECEIVED' and received['files'] == 35
                    and received['full_stdin_eof'] is True and received['data_closure'] == DATA_PROOF,
                    'SAME approved original receiver receipt; no new transfer claim')
            self.authority = Held(base/'data-closure.json', DATA_PROOF, limit=1024**2)
            value = json.load(self.authority.file)
            self.authority.identity()
            validate_authority(value)
            self.records = value['records']
            validate_tree(root, self.records)
            for name, proof in sorted(self.records.items()):
                self.held.append(Held(root/relative(name), proof))
            require(len(self.held) == 35
                    and len({item.before[:2] for item in self.held}) == 35, 'all35 distinct original FDs held')
            self.authority.after()
        except BaseException:
            self.close()
            raise

    def bindings(self):
        return [(item.file.fileno(), str(item.path)) for item in self.held]

    def authority_after(self):
        # Original run.py verifies/closes every data FD independently.
        validate_tree(self.root, self.records)
        self.authority.after()
        self.received.after()

    def close(self):
        for item in self.held:
            item.close()
        if self.authority is not None:
            self.authority.close()
        if self.received is not None:
            self.received.close()



def enclosed_import(base, source, binary, data, command):
    argv = ['/run/current-system/sw/bin/bwrap', '--unshare-user', '--unshare-net',
            '--unshare-pid', '--die-with-parent', '--ro-bind', '/', '/',
            '--bind', str(base), str(base), '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
            '--ro-bind-fd', str(binary.file.fileno()), '/tmp/zg-qwen-held-binary']
    for fd, path in data.bindings():
        argv += ['--ro-bind-fd', str(fd), path]
    return argv + ['--chdir', str(source), '--', *command]

#!/usr/bin/env python3
"""Reuse exact approved offline registry inputs into a NEW remote closure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tomllib

OLD_SHA = '42ac6bf6f82e61adfe1cff7c565076361f222c51792f02df09337ff1779a791a'
REGISTRY = 'index.crates.io-1949cf8c6b5b557f'


def require(value, message):
    if not value:
        raise ValueError(message)


def stamp(s):
    return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def pin(path):
    s = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(s.st_mode) and s.st_size <= 64*1024**2, 'bounded regular input')
    with path.open('rb') as f:
        require(stamp(os.fstat(f.fileno())) == stamp(s), 'input identity')
        sha = hashlib.file_digest(f,'sha256').hexdigest()
        require(stamp(os.fstat(f.fileno())) == stamp(s) and stamp(path.stat(follow_symlinks=False)) == stamp(s), 'input changed')
    return {'bytes':s.st_size,'sha256':sha}


def index_relative(name):
    name = name.lower()
    if len(name) <= 2:
        return Path(str(len(name)))/name
    if len(name) == 3:
        return Path('3')/name[0]/name
    return Path(name[:2])/name[2:4]/name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--cache-base',type=Path,required=True)
    a = p.parse_args()
    require(a.base.is_absolute() and a.cache_base.is_absolute() and a.base != a.cache_base, 'distinct absolute roots')
    require(not (a.base/'closure.json').exists() and not (a.base/'cargo').exists(), 'fresh cache/closure')
    old_path = a.cache_base/'closure.json'
    require(pin(old_path)['sha256'] == OLD_SHA, 'original approved cache closure changed')
    raw = old_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == OLD_SHA, 'parsed cache authority differs')
    old = json.loads(raw)
    source_path = a.base/'source-closure.json'
    source_pin = pin(source_path)
    raw = source_path.read_bytes()
    require(len(raw) == source_pin['bytes'] and hashlib.sha256(raw).hexdigest() == source_pin['sha256'], 'parsed source authority differs')
    report = json.loads(raw)
    records = report['records']
    for rel, expected in records.items():
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'safe source member')
        require(pin(a.base/rel) == expected, 'source closure mismatch: '+rel)
    lock = tomllib.loads((a.base/'source/Cargo.lock').read_text())
    keys = {}
    names = set()
    for pkg in lock['package']:
        if not pkg.get('source'):
            continue
        require(pkg['source'] == 'registry+https://github.com/rust-lang/crates.io-index', 'original locked registry only')
        key = f"cargo/registry/cache/{REGISTRY}/{pkg['name']}-{pkg['version']}.crate"
        require(old['records'][key]['sha256'] == pkg['checksum'], 'locked archive authority differs')
        keys[key] = old['records'][key]
        names.add(pkg['name'])
    for name in names:
        key = f'cargo/registry/index/{REGISTRY}/.cache/{index_relative(name)}'
        keys[key] = old['records'][key]
    key = f'cargo/registry/index/{REGISTRY}/config.json'
    keys[key] = old['records'][key]
    for rel, expected in sorted(keys.items()):
        source, target = a.cache_base/rel, a.base/rel
        before = source.stat(follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= 64*1024**2, 'bounded cache member')
        target.parent.mkdir(parents=True,exist_ok=True)
        digest = hashlib.sha256()
        with source.open('rb') as f, target.open('xb') as out:
            require(stamp(os.fstat(f.fileno())) == stamp(before), 'cache identity')
            while block := f.read(1024**2):
                digest.update(block)
                out.write(block)
            require(stamp(os.fstat(f.fileno())) == stamp(before) and stamp(source.stat(follow_symlinks=False)) == stamp(before), 'cache changed')
        actual = {'bytes':before.st_size,'sha256':digest.hexdigest()}
        require(actual == expected, 'approved cache bytes differ: '+rel)
        target.chmod(stat.S_IMODE(before.st_mode))
        records[rel] = actual
    require(pin(old_path)['sha256'] == OLD_SHA, 'old closure changed during copy')
    report.update(files=len(records),bytes=sum(v['bytes'] for v in records.values()),
                  reused_cache_closure_sha256=OLD_SHA,records=records)
    (a.base/'closure.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'},indent=2))


if __name__ == '__main__':
    main()

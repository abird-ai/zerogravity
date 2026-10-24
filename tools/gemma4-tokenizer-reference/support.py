"""Bounded metadata/source transport; no Torch/provider/model import."""
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

HERE = Path(__file__).resolve().parent
IMAGE = 'sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe'
IMAGE_DIGEST = 'sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634'
TORCH_GIT = 'cf30153c4c131c8164ee7798e5022d810682e2cb'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result

def read(path, cap=4*1024*1024, pin=None):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as file:
        before = os.fstat(file.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular metadata')
        raw = file.read(cap+1)
        after = os.fstat(file.fileno())
    require(len(raw) <= cap and fingerprint(before) == fingerprint(after), 'immutable metadata FD')
    if pin is not None:
        require(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'], 'exact metadata pin')
    return raw

def metadata(path, cap=4*1024*1024, pin=None):
    return json.loads(read(path, cap, pin), object_pairs_hook=unique)

def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    require(len(raw) <= 4*1024*1024, 'bounded output metadata')
    with Path(path).open('xb') as file:
        os.fchmod(file.fileno(),0o600)
        file.write(raw)

def fingerprint(meta):
    return [meta.st_dev,meta.st_ino,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns]

def fd_sha(fd, begin, count):
    value = hashlib.sha256()
    for offset in range(0,count,1024*1024):
        raw = os.pread(fd,min(1024*1024,count-offset),begin+offset)
        require(len(raw) == min(1024*1024,count-offset), 'short source FD read')
        value.update(raw)
    return value.hexdigest()


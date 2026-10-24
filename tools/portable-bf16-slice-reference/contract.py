"""Synthetic three-role source and exact original-reference protocol; no Torch."""
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
MODEL_SHA = '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b'
HELPER_SHA = '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d'
FILES = ('contract.py', 'capture.py', 'run_reference.py', 'validate_reference.py', 'owned_container_v4.py')
STAGES = ('embedding', 'raw_gamma', 'runtime_gamma', 'squares', 'mean', 'denominator',
          'inverse', 'normalized', 'weighted', 'rms_bf16', 'projection_bf16')
MAX = 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def read(path, cap=MAX):
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular file')
        raw = os.pread(fd, cap + 1, 0)
        require(len(raw) == before.st_size and fingerprint(before) == fingerprint(os.fstat(fd))
                == fingerprint(os.stat(path, follow_symlinks=False)), 'complete stable held file')
        return raw
    finally:
        os.close(fd)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON member')
        result[key] = value
    return result


def metadata(path):
    return json.loads(read(path), object_pairs_hook=unique)


def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    require(len(raw) <= MAX, 'bounded JSON output')
    with Path(path).open('xb') as output:
        output.write(raw)


def package():
    return {name: {'bytes': len(raw), 'sha256': digest(raw)}
            for name in FILES for raw in [read(HERE / name)]}


def fixture(width, rows=97):
    require(type(width) is int and width in (1024, 2048) and rows == 97, 'closed synthetic source geometry')
    lengths = [17 * width * 2, width * 2, rows * width * 2]
    header = json.dumps({
        'embedding': {'dtype': 'BF16', 'shape': [17, width], 'data_offsets': [0, lengths[0]]},
        'gamma': {'dtype': 'BF16', 'shape': [width], 'data_offsets': [lengths[0], sum(lengths[:2])]},
        'projection': {'dtype': 'BF16', 'shape': [rows, width], 'data_offsets': [sum(lengths[:2]), sum(lengths)]},
    }, separators=(',', ':')).encode()
    raw = bytearray(struct.pack('<Q', len(header)) + header)
    for member, length in enumerate(lengths):
        for i in range(length // 2):
            q = (i * 1664525 + member * 1013904223 + 123) & 0xffffffff
            exponent = 119 + ((q >> 8) % 5) if member == 1 else 0 if i % 37 == 0 else 112 + ((q >> 8) % 24)
            bits = ((q >> 31) << 15) | (exponent << 7) | ((q >> 16) & 127)
            raw.extend(struct.pack('<H', bits))
    require(len(raw) <= MAX, 'source cap before independent runtime')
    return bytes(raw)


def counts(width, rows):
    return (width, width, width, width, 1, 1, 1, width, width, width, rows)


def parse(raw, source_sha, width, rows=97, token=16):
    require(width in (1024, 2048) and rows == 97 and token == 16, 'closed original cohort')
    require(56 <= len(raw) <= MAX and raw[:8] == b'ZGBF16V1' and raw[8:40].hex() == source_sha,
            'original source/header')
    require(struct.unpack_from('<4I', raw, 40) == (width, rows, token, len(STAGES)), 'original geometry/token/stages')
    at = 56
    records = []
    for label, words in zip(STAGES, counts(width, rows), strict=True):
        require(at + 4 <= len(raw), 'whole length')
        length, = struct.unpack_from('<I', raw, at)
        at += 4
        require(length == words * 4 and at + length <= len(raw), 'full defined stage extent')
        payload = raw[at:at + length]
        for bits, in struct.iter_unpack('<I', payload):
            require(bits & 0x7f800000 != 0x7f800000, 'original finite words')
            if label in ('embedding', 'rms_bf16', 'projection_bf16'):
                require(bits & 0xffff == 0, 'original BF16 precision')
        records.append({'stage': label, 'words': words, 'offset': at, 'bytes': length, 'sha256': digest(payload)})
        at += length
    require(at == len(raw), 'no trailing original bytes')
    return records


def source_joins(raw, source, records, width, token=16):
    header_bytes, = struct.unpack_from('<Q', source)
    header = json.loads(source[8:8 + header_bytes], object_pairs_hook=unique)
    by_name = {row['stage']: raw[row['offset']:row['offset'] + row['bytes']] for row in records}
    for label, tensor, row in [('embedding', 'embedding', token), ('raw_gamma', 'gamma', 0)]:
        start = 8 + header_bytes + header[tensor]['data_offsets'][0] + row * width * 2
        expected = b''.join(struct.pack('<I', word << 16) for word, in struct.iter_unpack('<H', source[start:start + width * 2]))
        require(by_name[label] == expected, 'exact original source word widening: ' + label)

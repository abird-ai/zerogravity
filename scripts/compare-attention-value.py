#!/usr/bin/env python3
"""Strict native pre-gate attention values at six layers/four causal tokens.

Compare every binary32 bit, including zero signs. The optional self-test runs
only in memory and verifies rejection of incomplete or altered capture records.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


CHECKPOINTS = {
    '57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf',
    '91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6',
}
EXPECTED = {(pos, layer) for pos in range(4) for layer in (3, 7, 11, 15, 19, 23)}
WIDTH = 2048


def bits(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('nonfinite/non-numeric attention value')
    try:
        return struct.pack('<f', value)
    except (OverflowError, struct.error) as error:
        raise ValueError('attention value outside binary32 domain') from error


def record_key(row):
    if (not isinstance(row, dict) or type(row.get('position')) is not int
            or type(row.get('layer')) is not int):
        raise ValueError('invalid attention value record key')
    return row['position'], row['layer']


def values(row):
    vector = row.get('values')
    if not isinstance(vector, list) or len(vector) != WIDTH:
        raise ValueError('incomplete attention value vector')
    return [bits(value) for value in vector]


def compare(reference, candidate, stage="full_pregate"):
    table = {}
    for row in reference:
        key = record_key(row)
        if key not in EXPECTED or key in table:
            raise ValueError('duplicate/unexpected native attention value record')
        table[key] = values(row)
    if set(table) != EXPECTED:
        raise ValueError('incomplete four-token/six-layer native value coverage')
    seen = set()
    count = 0
    for row in candidate:
        if not isinstance(row, dict):
            raise ValueError('invalid Rust trace record')
        if row.get('stage') != stage:
            continue
        key = record_key(row)
        if key not in EXPECTED or key in seen:
            raise ValueError('duplicate/unexpected Rust attention value record')
        seen.add(key)
        actual = values(row)
        for offset, (a, b) in enumerate(zip(actual, table[key])):
            if a != b:
                raise ValueError(f'source attention value bits mismatch {key} offset={offset}: '
                                 f'{a.hex()} vs {b.hex()}')
        count += len(actual)
    if seen != EXPECTED:
        raise ValueError('missing Rust attention value records')
    return count


def self_test():
    reference = [dict(position=pos, layer=layer, values=[-0.0] + [1.0] * (WIDTH - 1))
                 for pos, layer in sorted(EXPECTED)]
    candidate = [dict(row, stage='full_pregate') for row in reference]
    if compare(reference, candidate) != 49_152:
        raise ValueError('positive comparator self-test failed')

    gated = [dict(row, stage='full_gated') for row in reference]
    if compare(reference, gated, 'full_gated') != 49_152:
        raise ValueError('gated comparator self-test failed')

    def rejected(source, actual):
        try:
            compare(source, actual)
        except ValueError:
            return
        raise ValueError('negative comparator self-test accepted malformed records')

    rejected(reference[:-1], candidate)
    rejected(reference + [reference[0]], candidate)
    rejected(reference, candidate[:-1])
    rejected(reference, candidate + [candidate[0]])
    rejected(reference, [dict(candidate[0], layer=0)] + candidate[1:])
    rejected(reference, [dict(candidate[0], values=[1.0])] + candidate[1:])
    for wrong in (0.0, 2.0, float('nan'), float('inf'), True):
        changed = [wrong] + candidate[0]['values'][1:]
        rejected(reference, [dict(candidate[0], values=changed)] + candidate[1:])
    print('ATTENTION_VALUE_COMPARATOR_SELF_TEST_PASS positive=2 negative=11 signed_zeros_strict=1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'candidate', 'model'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--stage', choices=('full_pregate', 'full_gated'), default='full_pregate')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    paths = (args.reference, args.candidate, args.model, args.sha256)
    if args.self_test:
        if any(value is not None for value in paths):
            parser.error('--self-test accepts no capture/model arguments')
        self_test()
        return
    if any(value is None for value in paths):
        parser.error('--reference, --candidate, --model and --sha256 are required')
    if args.sha256 not in CHECKPOINTS:
        raise ValueError('checkpoint pin is outside the two original qualified sources')
    with args.model.open('rb') as model:
        digest = hashlib.file_digest(model, 'sha256').hexdigest()
    if digest != args.sha256:
        raise ValueError('checkpoint hash mismatch')
    read = lambda path: [json.loads(line) for line in path.read_text().splitlines()]
    count = compare(read(args.reference), read(args.candidate), args.stage)
    print(f'INDEPENDENT_SOURCE_VALUE_BITS_PASS tokens=4 layers=6 native_values={count} '
          f'signed_zeros_strict=1 stage={args.stage} checkpoint={digest}')


if __name__ == '__main__':
    main()

"""Tiny protocol/ownership fixtures only; never executes an original runtime."""
import copy
import struct
import tempfile
import unittest
from pathlib import Path
from contract import (IMAGE, IMAGE_DIGEST, MODEL_SHA, STAGES, TORCH_GIT, counts, digest,
                      fixture, package, parse, source_joins, write)
from run_reference import command, environment, helper_load
from validate_reference import validate


def archive(width):
    source = fixture(width)
    header_bytes, = struct.unpack_from('<Q', source)
    import json
    header = json.loads(source[8:8 + header_bytes])
    raw = bytearray(b'ZGBF16V1' + bytes.fromhex(digest(source)) + struct.pack('<4I', width, 97, 16, 11))
    for label, words in zip(STAGES, counts(width, 97), strict=True):
        payload = b'\0' * (words * 4)
        if label in ('embedding', 'raw_gamma'):
            name = 'embedding' if label == 'embedding' else 'gamma'
            start = 8 + header_bytes + header[name]['data_offsets'][0] + (16 * width * 2 if name == 'embedding' else 0)
            payload = b''.join(struct.pack('<I', w << 16) for w, in struct.iter_unpack('<H', source[start:start + width * 2]))
        raw.extend(struct.pack('<I', len(payload)) + payload)
    return source, bytes(raw)


class Protocol(unittest.TestCase):
    def test_full_defined_extent_and_hostile_boundaries(self):
        for width in (1024, 2048):
            source, raw = archive(width)
            records = parse(raw, digest(source), width)
            source_joins(raw, source, records, width)
            for changed, sha, h in [(raw[:-1], digest(source), width), (raw + b'\0', digest(source), width),
                                    (raw, '00' * 32, width), (raw, digest(source), 3072)]:
                with self.assertRaises(ValueError):
                    parse(changed, sha, h)
            hostile = bytearray(raw)
            struct.pack_into('<I', hostile, 56, 4)
            with self.assertRaises(ValueError):
                parse(hostile, digest(source), width)
            hostile = bytearray(raw)
            struct.pack_into('<I', hostile, 60, 0x3f800001)
            with self.assertRaises(ValueError):
                parse(hostile, digest(source), width)
            hostile = bytearray(raw)
            struct.pack_into('<I', hostile, 60, 0x7f800000)
            with self.assertRaises(ValueError):
                parse(hostile, digest(source), width)
            hostile = bytearray(raw)
            hostile[60] ^= 0x80
            with self.assertRaises(ValueError):
                source_joins(hostile, source, records, width)

    def test_complete_fixture_and_real_owner_field_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'source').mkdir()
            (root / 'payload').mkdir()
            source, raw = archive(1024)
            (root / 'source/source.safetensors').write_bytes(source)
            (root / 'payload/reference.bin').write_bytes(raw)
            plan = {'width': 1024, 'rows': 97, 'token': 16, 'source_sha256': digest(source), 'source_bytes': len(source)}
            write(root / 'payload/input.json', plan)
            nonce = 'a' * 32
            inv = {'status': 'ORIGINAL_PORTABLE_BF16_REFERENCE_REQUIRES_VALIDATION', 'exit_code': 0, 'cleanup_errors': [],
                   'source_FDs_closed': True, 'source_before': [1, 2, len(source), 3, 4], 'source_after': [1, 2, len(source), 3, 4],
                   'package': package(), 'image_id': IMAGE, 'image_digest': IMAGE_DIGEST, 'owned_nonce': nonce,
                   'owned_name': 'zg-portable-bf16-original-' + nonce, 'expected_cpus': [0, 1, 2, 3],
                   'foreground_cleanup': {'all_nonzero_signals_before_reap': True, 'leader_reaped': True, 'group_absent': True,
                                          'kill_attempts_before_reap': 2, 'stage_errors': []},
                   'owned_container_absent_after_reap': True, 'plan': plan,
                   'log_cleanup': {'pipe_eof': True, 'all_closed': True, 'total_bytes': 32, 'retained_bytes': 32}}
            inv['argv'] = command(helper_load(), root, environment(inv['expected_cpus']), inv['owned_name'], nonce)
            write(root / 'image-inspect.json', [{'Id': IMAGE.removeprefix('sha256:'), 'Digest': IMAGE_DIGEST}])
            capture = {'schema': 'zg.portable-bf16-original.v1', 'status': 'ORIGINAL_REFERENCE_REQUIRES_VALIDATION',
                       'torch': '2.13.0+cu129', 'torch_git': TORCH_GIT, 'transformers': '5.17.0', 'modeling_sha256': MODEL_SHA,
                       'address_space_limit': [16*1024**3]*2, 'fd_limit': [1024,1024], 'cpu_capability': 'AVX2', 'affinity': [0, 1, 2, 3], 'threads': 4, 'interop_threads': 1,
                       'width': 1024, 'rows': 97, 'token': 16, 'source_sha256': digest(source),
                       'records': parse(raw, digest(source), 1024), 'bytes': len(raw), 'sha256': digest(raw),
                       'scope': {'synthetic_source': True, 'trained_model': False, 'portable_arithmetic_qualified': False,
                                 'runtime_rust_source_in_original': False, 'state_root': False, 'serving': False, 'performance': False}}
            write(root / 'payload/capture.json', capture)

            def check(candidate):
                path = root / 'invocation.json'
                path.unlink(missing_ok=True)
                write(path, candidate)
                return validate(root)

            self.assertFalse(check(inv)['portable_arithmetic_qualified'])
            for section, key, value in [('foreground_cleanup', 'all_nonzero_signals_before_reap', False),
                                         ('foreground_cleanup', 'kill_attempts_before_reap', 1),
                                         ('foreground_cleanup', 'group_absent', False),
                                         ('log_cleanup', 'pipe_eof', False), ('log_cleanup', 'retained_bytes', 31)]:
                bad = copy.deepcopy(inv)
                bad[section][key] = value
                with self.assertRaises(ValueError):
                    check(bad)
            bad = copy.deepcopy(inv)
            bad['argv'][bad['argv'].index('--cpus=4')] = '--cpus=8'
            with self.assertRaises(ValueError):
                check(bad)
            bad = copy.deepcopy(inv)
            bad['source_after'][3] += 1
            with self.assertRaises(ValueError):
                check(bad)


if __name__ == '__main__':
    unittest.main()

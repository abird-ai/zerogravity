"""Tiny natural protocol fixtures only; no framework, source scan or model execution."""
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from contract import (HERE, argmax_bf16, digest, extent, finite, record_metadata,
                      record_specs, role_view, roles, scope, specs, state_specs)
from validate_reference import Held, join_call_records, natural_inputs, validate_calls
from archive import Archive
from run_reference import source_after
from contract import fingerprint

SELECTIONS = {0: (220, 0x3f800000), 1: (2688, 0x40000000)}


def row_for(spec, stride, input_id):
    repeat, call, layer, label, shape, dtype = spec
    return dict(repeat=repeat, call=call, teacher=input_id, position=call, layer=layer,
                label=label, shape=shape.copy(), dtype=dtype, device='cpu', offset=0,
                bytes=extent(shape, dtype), original_stride=stride)


def calls():
    groups = ['embedding']+['layer.'+str(i) for i in range(64)]+['final', 'head']
    result = []
    for repeat in range(2):
        logits_sha = {}
        for call in range(2):
            selected, bits = SELECTIONS[call]
            # Byte-identical repetitions: the same call in both repeats holds the
            # same original fullV record identity.
            logits_sha[call] = digest(f'logits-{call}'.encode())
            moments = {}
            for moment in ('before', 'after'):
                flags = []
                for i in range(64):
                    initialized = moment == 'after' or call == 1
                    flag = dict(layer=i, kind='full_attention' if i % 4 == 3 else 'linear_attention')
                    for k in (('key', 'value') if i % 4 == 3 else ('conv', 'recurrent', 'previous')):
                        flag[k] = initialized
                    flags.append(flag)
                moments[moment] = flags
            result.append(dict(repeat=repeat, call=call,
                               teacher=11 if call == 0 else SELECTIONS[0][0],
                               position=call, cache_frontier=call+1,
                               natural_feedback_used=call == 1,
                               original_module_order=groups.copy(),
                               greedy_id_diagnostic_only=selected, selected_max_bits=bits,
                               logits_record_sha256=logits_sha[call],
                               predecessor_logits_sha256=logits_sha.get(0) if call else None,
                               **moments))
    return result


RUST = Path(os.environ.get('ZG_EXPECTED_RS')
            or HERE.parents[1]/'crates/zerogravity/examples/qwen27b_hip_matrix/expected.rs')
REFERENCE_TOOL = Path(os.environ.get('ZG_REFERENCE_TOOL')
                      or HERE.parent/'qwen27b-full-window-reference')


def declared_package(const):
    text = RUST.read_text()
    body = re.search(r'const '+const+r': \[\(&str, u64, &str\); 9\] = \[(.*?)\n\];',
                     text, re.S).group(1)
    rows = re.findall(r'"([^"]+)"\s*,\s*([\d_]+)\s*,\s*"([0-9a-f]{64})"', body)
    assert len(rows) == 9, const
    return {name: (int(size.replace('_', '')), sha) for name, size, sha in rows}


class Protocol(unittest.TestCase):
    def test_runtime_declared_pins_equal_these_actual_tool_bytes(self):
        """Declared Rust pins and the approved/natural package bytes cannot drift."""
        if not (RUST.is_file() and REFERENCE_TOOL.is_dir()):
            self.skipTest('candidate Rust source/reference tool roots absent in a frozen copy')
        for const, root in (('NATURAL_PACKAGE', HERE), ('REFERENCE_PACKAGE', REFERENCE_TOOL)):
            declared = declared_package(const)
            self.assertEqual(len(declared), 9)
            for name, (size, sha) in declared.items():
                raw = (root/name).read_bytes()
                self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (size, sha), name)
        natural, reference = declared_package('NATURAL_PACKAGE'), declared_package('REFERENCE_PACKAGE')
        self.assertNotEqual(natural, reference)
        self.assertEqual(natural['owned_container_v4.py'], reference['owned_container_v4.py'])
        self.assertEqual(natural['source-pin.json'], reference['source-pin.json'])
        self.assertEqual(natural['source_owner.py'], reference['source_owner.py'])
        self.assertEqual(natural['archive.py'], reference['archive.py'])

    def test_whole_source_and_complete_natural_four_call_record_geometry(self):
        plan = json.loads((HERE/'inputs.json').read_text())
        pins = json.loads((HERE/'source-pin.json').read_text())
        self.assertEqual(len(specs(plan, pins)), 31)
        actual = list(record_specs())
        self.assertEqual(len(actual), 1040)
        self.assertEqual(sum(extent(s[4], s[5]) for s in actual), 934776928)
        self.assertEqual(len(roles()), 851)
        self.assertEqual(len(state_specs(0, 'before')), 0)
        self.assertEqual(len(state_specs(1, 'before')), 128)
        self.assertEqual(sum('conv4' == s[1] for s in state_specs(1, 'after')), 48)
        self.assertTrue(all(s[2][-1] == 4 for s in state_specs(1, 'after') if s[1] == 'conv4'))
        self.assertTrue(scope()['natural_feedback_qualified'])
        self.assertFalse(scope()['teacher_forcing_only'])
        self.assertFalse(scope()['rust_numerical_parity_qualified'])
        self.assertFalse(scope()['mkl_exp_producer_qualified'])
        for change in ({'scope': {**plan['scope'], 'mkl_exp_producer_qualified': True}},
                       {'scope': {**plan['scope'], 'natural_feedback_qualified': False,
                                  'teacher_forcing_only': True}},
                       {'teachers': [11, SELECTIONS[0][0]]}, {'seed': 353}, {'calls': 1},
                       {'capacity': 3}):
            bad = copy.deepcopy(plan)
            bad.update(change)
            with self.assertRaises(ValueError):
                specs(bad, pins)

    def test_original_strides_are_elements_not_bytes_and_broadcast_positions_are_valid(self):
        s = (0, 0, None, 'positions', [3, 1, 1], 'I64')
        self.assertEqual(record_metadata(row_for(s, [0, 1, 1], 11), s, 0, 11), 24)
        s = (0, 0, 0, 'after.conv4', [1, 10240, 4], 'BF16')
        r = row_for(s, [40960, 4, 1], 220)
        self.assertEqual(record_metadata(r, s, 0, 220), 81920)
        for key, value in [('original_stride', [81920, 8, 1]), ('bytes', 81918),
                           ('call', False), ('shape', [True, 10240, 4]), ('layer', False),
                           ('teacher', 11)]:
            bad = copy.deepcopy(r)
            bad[key] = value
            with self.assertRaises(ValueError):
                record_metadata(bad, s, 0, 220)
        for bad_input in (248320, -1, True, None):
            with self.assertRaises(ValueError):
                record_metadata(r, s, 0, bad_input)

    def test_every_real_cache_kind_and_initialization_transition(self):
        rows = calls()
        self.assertEqual(natural_inputs(rows), [[11, 220], [11, 220]])
        validate_calls(rows)
        for mutate in ('drop_layer', 'false_prior', 'wrong_order', 'fake_frontier',
                       'teacher_forced', 'wrong_predecessor', 'no_feedback',
                       'nondeterministic_repeat'):
            bad = copy.deepcopy(rows)
            if mutate == 'drop_layer':
                bad[0]['after'].pop()
            elif mutate == 'false_prior':
                bad[1]['before'][0]['conv'] = False
            elif mutate == 'nondeterministic_repeat':
                bad[3]['greedy_id_diagnostic_only'] = (SELECTIONS[0][0] + 1) % 248320
                bad[3]['logits_record_sha256'] = 'e'*64
            elif mutate == 'wrong_order':
                bad[0]['original_module_order'][1:3] = ['layer.1', 'layer.0']
            elif mutate == 'fake_frontier':
                bad[1]['cache_frontier'] = 1
            elif mutate == 'teacher_forced':
                # Any call1 input that is not the independently decoded call0
                # argmax is teacher-forced, whatever its value.
                bad[1]['teacher'] = (SELECTIONS[0][0] + 1) % 248320
            elif mutate == 'wrong_predecessor':
                bad[1]['predecessor_logits_sha256'] = 'f'*64
            else:
                bad[1]['natural_feedback_used'] = False
            with self.assertRaises(ValueError):
                validate_calls(bad)
        # A call1 input that differs from the decoded call0 argmax is not a
        # natural cohort (the retired teacher-forcing trajectory used 353).
        forced = copy.deepcopy(rows)
        for r in range(2):
            forced[r*2]['teacher'] = forced[r*2+1]['teacher'] = (SELECTIONS[0][0] + 1) % 248320
        with self.assertRaises(ValueError):
            natural_inputs(forced)
        with self.assertRaises(ValueError):
            validate_calls(forced)

    def test_independent_natural_argmax_is_integer_total_order(self):
        def bf16(values):
            return b''.join(int(v).to_bytes(2, 'little') for v in values)
        self.assertEqual(argmax_bf16(bf16([0x3f80, 0x4000, 0x4000])), (1, 0x40000000))
        # Signed zeros are one numeric zero: first index wins and keeps -0 bits.
        self.assertEqual(argmax_bf16(bf16([0x8000, 0x0000])), (0, 0x80000000))
        self.assertEqual(argmax_bf16(bf16([0x0000, 0x8000])), (0, 0))
        self.assertEqual(argmax_bf16(bf16([0x8000, 0x8000])), (0, 0x80000000))
        # Subnormals keep source bits and order by signed encoding.
        self.assertEqual(argmax_bf16(bf16([0x0001, 0x0002])), (1, 0x00020000))
        self.assertEqual(argmax_bf16(bf16([0x8001, 0x8002])), (0, 0x80010000))
        self.assertEqual(argmax_bf16(bf16([0xc000, 0xbf80])), (1, 0xbf800000))
        # Exact numeric order: -2 < -1 < -0 == +0 < +1, ties keep the lowest id.
        self.assertEqual(argmax_bf16(bf16([0xc000, 0xbf80])), (1, 0xbf800000))
        self.assertEqual(argmax_bf16(bf16([0x8001, 0x0000])), (1, 0))
        self.assertEqual(argmax_bf16(bf16([0x0000, 0x8001])), (0, 0))
        self.assertEqual(argmax_bf16(bf16([0xbf80, 0x8001])), (1, 0x80010000))
        self.assertEqual(argmax_bf16(bf16([0x3f80, 0x3f80, 0x3f80])), (0, 0x3f800000))
        self.assertEqual(argmax_bf16(bf16([0xbf80, 0xbf80])), (0, 0xbf800000))
        with self.assertRaises(ValueError):
            argmax_bf16(bf16([0x7fc0, 0x3f80]))
        with self.assertRaises(ValueError):
            argmax_bf16(b'')
        with self.assertRaises(ValueError):
            argmax_bf16(b'\x00')

    def test_mixed_source_roles_keep_full_hf_conv_and_raw_f32(self):
        role = next(r for r in roles() if r['name'].endswith('layers.0.linear_attn.A_log'))
        header = {role['name']: dict(dtype='F32', shape=[48], data_offsets=[0, 192])}
        self.assertEqual(role_view(8, header, role, 200)['bytes'], 192)
        bad = copy.deepcopy(header)
        bad[role['name']]['shape'] = [16]
        with self.assertRaises(ValueError):
            role_view(8, bad, role, 200)
        conv = next(r for r in roles() if r['name'].endswith('layers.0.linear_attn.conv1d.weight'))
        self.assertEqual(conv['shape'], [10240, 1, 4])
        self.assertEqual(conv['allowed_dtypes'], ['BF16'])

    def test_record_cap_refuses_before_tensor_transport(self):
        class Tensor:
            device = type('Device', (), dict(type='cpu'))()
            layout = 'strided'
            dtype = 'bf16'
            shape = (1, 3)
            def detach(self):
                raise AssertionError('transport must not occur')
        torch = type('Torch', (), dict(Tensor=Tensor, strided='strided', bfloat16='bf16',
                                      float32='f32', int64='i64'))
        with tempfile.TemporaryDirectory() as tmp:
            archive = Archive(Path(tmp)/'raw', dict(record_cap=4, archive_cap=4), torch)
            archive.tag = dict(repeat=0, call=0, teacher=11, position=0)
            with self.assertRaises(ValueError):
                archive.add('test', None, Tensor())
            self.assertEqual(archive.total, 0)
            archive.close()

    def test_declared_natural_inputs_are_returned_and_record_identities_join(self):
        rows = calls()
        self.assertEqual(validate_calls(rows), natural_inputs(rows))
        self.assertEqual(validate_calls(rows), [[11, 220], [11, 220]])
        records = {(r, c): rows[r*2+c]['logits_record_sha256']
                   for r in range(2) for c in range(2)}
        self.assertTrue(join_call_records(rows, records))
        for change in ({'predecessor_logits_sha256': '1'*64},
                       {'predecessor_logits_sha256': None},
                       {'selected_max_bits': 0x3f800001},
                       {'selected_max_bits': 0x7fc00000},
                       {'natural_feedback_used': False}):
            bad = copy.deepcopy(rows)
            bad[1].update(change)
            with self.assertRaises(ValueError):
                validate_calls(bad)
        for change in ({'predecessor_logits_sha256': '2'*64},
                       {'logits_record_sha256': '0'*64}):
            bad = copy.deepcopy(rows)
            bad[1].update(change)
            with self.assertRaises(ValueError):
                join_call_records(bad, records)
        with self.assertRaises(ValueError):
            join_call_records(rows, {})
        # Call1's predecessor is its cohort's call0 record, never the call1 record
        # itself. Byte-identical repeats legitimately share call0's identity.
        bad = copy.deepcopy(rows)
        bad[3]['predecessor_logits_sha256'] = records[(1, 1)]
        with self.assertRaises(ValueError):
            join_call_records(bad, records)

    def test_whole_held_pin_refuses_replacement_and_mutation_during_final_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'raw'
            p.write_bytes(b'1234')
            with Held(p, 16) as held:
                q = Path(tmp)/'new'
                q.write_bytes(b'1234')
                q.replace(p)
                with self.assertRaises(ValueError):
                    held.pin()
            with Held(p, 16) as held:
                import validate_reference
                original = validate_reference.fd_sha
                def change_after_hash(fd, offset, count):
                    result = original(fd, offset, count)
                    p.write_bytes(b'5678')
                    return result
                with patch('validate_reference.fd_sha', change_after_hash):
                    with self.assertRaises(ValueError):
                        held.pin()

    def test_outer_original_source_close_does_not_accept_replaced_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'shard'
            p.write_bytes(b'raw!')
            fd = os.open(p, os.O_RDONLY)
            try:
                rows = [dict(name='shard', bytes=4, sha256=digest(b'raw!'),
                             fd_identity=fingerprint(os.fstat(fd)))]
                self.assertEqual(source_after({'shard': fd}, rows, tmp), rows)
                q = Path(tmp)/'replacement'
                q.write_bytes(b'raw!')
                q.replace(p)
                with self.assertRaises(ValueError):
                    source_after({'shard': fd}, rows, tmp)
            finally:
                os.close(fd)

    def test_codec_rejects_nonfinite_bits_and_import_has_no_model_side_effect(self):
        self.assertTrue(finite(b'\x00\x00\x00\x80', 'BF16'))
        self.assertFalse(finite(b'\x80\x7f', 'BF16'))
        self.assertFalse(finite(b'\x00\x00\x80\x7f', 'F32'))
        import capture
        self.assertIsNone(capture.OWNER)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)


if __name__ == '__main__':
    unittest.main()

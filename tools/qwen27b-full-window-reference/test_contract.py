"""Tiny protocol fixtures only; no framework, source scan or model execution."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from contract import (HERE, digest, extent, finite, record_metadata, record_specs,
                      role_view, roles, scope, specs, state_specs)
from validate_reference import Held, validate_calls
from archive import Archive
from run_reference import source_after
from contract import fingerprint


def row_for(spec, stride):
    repeat, call, teacher, layer, label, shape, dtype = spec
    return dict(repeat=repeat, call=call, teacher=teacher, position=call, layer=layer,
                label=label, shape=shape.copy(), dtype=dtype, device='cpu', offset=0,
                bytes=extent(shape, dtype), original_stride=stride)


def calls():
    groups = ['embedding']+['layer.'+str(i) for i in range(64)]+['final', 'head']
    result = []
    for repeat in range(2):
        for call, teacher in enumerate((11, 353)):
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
            result.append(dict(repeat=repeat, call=call, teacher=teacher, position=call,
                               cache_frontier=call+1, original_module_order=groups.copy(),
                               greedy_id_diagnostic_only=0, **moments))
    return result


class Protocol(unittest.TestCase):
    def test_whole_source_and_complete_four_forward_record_geometry(self):
        plan = json.loads((HERE/'inputs.json').read_text())
        pins = json.loads((HERE/'source-pin.json').read_text())
        self.assertEqual(len(specs(plan, pins)), 31)
        actual = list(record_specs())
        self.assertEqual(len(actual), 1040)
        self.assertEqual(sum(extent(s[5], s[6]) for s in actual), 934776928)
        self.assertEqual(len(roles()), 851)
        self.assertEqual(len(state_specs(0, 'before')), 0)
        self.assertEqual(len(state_specs(1, 'before')), 128)
        self.assertEqual(sum('conv4' == s[1] for s in state_specs(1, 'after')), 48)
        self.assertTrue(all(s[2][-1] == 4 for s in state_specs(1, 'after') if s[1] == 'conv4'))
        self.assertFalse(scope()['rust_numerical_parity_qualified'])
        self.assertFalse(scope()['mkl_exp_producer_qualified'])
        self.assertFalse(scope()['natural_feedback_qualified'])
        bad = copy.deepcopy(plan)
        bad['scope']['mkl_exp_producer_qualified'] = True
        with self.assertRaises(ValueError):
            specs(bad, pins)

    def test_original_strides_are_elements_not_bytes_and_broadcast_positions_are_valid(self):
        s = (0, 0, 11, None, 'positions', [3, 1, 1], 'I64')
        self.assertEqual(record_metadata(row_for(s, [0, 1, 1]), s, 0), 24)
        s = (0, 0, 11, 0, 'after.conv4', [1, 10240, 4], 'BF16')
        r = row_for(s, [40960, 4, 1])
        self.assertEqual(record_metadata(r, s, 0), 81920)
        for key, value in [('original_stride', [81920, 8, 1]), ('bytes', 81918),
                           ('call', False), ('shape', [True, 10240, 4]), ('layer', False)]:
            bad = copy.deepcopy(r)
            bad[key] = value
            with self.assertRaises(ValueError):
                record_metadata(bad, s, 0)

    def test_every_real_cache_kind_and_initialization_transition(self):
        rows = calls()
        validate_calls(rows)
        for mutate in ('drop_layer', 'false_prior', 'wrong_order', 'fake_frontier'):
            bad = copy.deepcopy(rows)
            if mutate == 'drop_layer':
                bad[0]['after'].pop()
            elif mutate == 'false_prior':
                bad[1]['before'][0]['conv'] = False
            elif mutate == 'wrong_order':
                bad[0]['original_module_order'][1:3] = ['layer.1', 'layer.0']
            else:
                bad[1]['cache_frontier'] = 1
            with self.assertRaises(ValueError):
                validate_calls(bad)

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

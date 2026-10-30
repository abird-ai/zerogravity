"""Real late-cancel report shape: full comparison, no published selection."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent / 'payload' / 'physical-runner'
spec = importlib.util.spec_from_file_location('pins', ROOT / 'pins.py')
pins = importlib.util.module_from_spec(spec)
sys.modules['pins'] = pins
spec.loader.exec_module(pins)
spec = importlib.util.spec_from_file_location('natural_semantic', ROOT / 'natural_semantic.py')
semantic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(semantic)


class LateComparison(unittest.TestCase):
    def scope(self):
        return dict(repeat=0, call=1, records=324, float_words=80114176,
                    positions=3, mismatches=0, exact=True)

    def test_complete_cancelled_comparison_needs_no_published_packet(self):
        value = self.scope()
        self.assertEqual(semantic.late_comparison(value), value)

    def test_every_scope_boundary_and_exactness_is_required(self):
        for key, wrong in [('repeat', 1), ('call', 0), ('records', 323),
                           ('float_words', 80114175), ('positions', 2),
                           ('mismatches', 1), ('exact', False)]:
            with self.subTest(key=key):
                value = self.scope()
                value[key] = wrong
                with self.assertRaises(ValueError):
                    semantic.late_comparison(value)
                value = self.scope()
                del value[key]
                with self.assertRaises(ValueError):
                    semantic.late_comparison(value)
        for key in ['repeat', 'call', 'records', 'float_words', 'positions', 'mismatches']:
            value = self.scope()
            value[key] = float(value[key])
            with self.assertRaises(ValueError):
                semantic.late_comparison(value)
        value = self.scope()
        value['repeat'] = False
        with self.assertRaises(ValueError):
            semantic.late_comparison(value)

    def test_fabricated_publication_cannot_replace_or_amend_comparison(self):
        packet = dict(selected_id=16, max_bits=1095827456)
        for value in [packet, {**self.scope(), **packet}, None, []]:
            with self.assertRaises(ValueError):
                semantic.late_comparison(value)


class ScopeAndFinalReader(unittest.TestCase):
    def test_declared_call_requires_its_complete_geometry(self):
        for call, records, words in [(0, 196, 40333824), (1, 324, 80114176)]:
            value = dict(declared_proposals=[[0, call]], records=records,
                         float_records=records-1, float_words=words,
                         positions=3, mismatches=0, exact=True)
            semantic.closure(value, 'control', call)
            for key in ['records', 'float_records', 'float_words', 'positions']:
                broken = {**value, key: value[key]-1}
                with self.assertRaises(ValueError):
                    semantic.closure(broken, 'control', call)
            with self.assertRaises(ValueError):
                semantic.closure(value, 'wrong call', 1-call)

    def test_sole_wrapper_has_exact_model_owners_not_a_shared_count(self):
        for wrapper, owners in [('Session', 2), ('Snapshot', 1)]:
            value = dict(wrapper=wrapper, fault=False, lease_bytes=12,
                         aliases_inside_wrapper=1, candidate_owners_before_drop=1,
                         model_owners_before_drop=owners, candidate_workspace_target_aliases=0,
                         candidate_retired=True, release_hits=1,
                         reader_hits_before_native_retirement=1, native_modules_before=121,
                         native_retirement_attempts=1, native_admission_retained_after=False,
                         native_modules_after=0, original_service_retained_after=False,
                         quarantined_handles_after=0)
            mode = wrapper.lower()+'-known'
            semantic.final_reader(value, mode)
            for field, bad in [('candidate_owners_before_drop', 2),
                               ('model_owners_before_drop', 3-owners),
                               ('aliases_inside_wrapper', 2)]:
                with self.assertRaises(ValueError):
                    semantic.final_reader({**value, field: bad}, mode)


if __name__ == '__main__':
    unittest.main()

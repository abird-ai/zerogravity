#!/usr/bin/env python3
"""Synthetic fail-closed controls; never trained-model qualification evidence."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "cpu_qualification_gate", Path(__file__).with_name("cpu-qualification-gate.py"))
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)
MODEL, SHA = next(iter(GATE.MODELS.items()))


def numeric():
    return {"sha256": SHA, "passed": True, "start_case": 0, "cases": [{
        "tokens": 32, "pass": True, "hard_top1_mismatches": 0,
        "missing_reference_top5_steps": 0, "unverified_fragment_positions": 0,
        "minimum_top20_overlap": 20, "max_shared_logprob_absolute_error": 0.0,
        "max_top1_probability_absolute_error": 0.0,
        "max_truncated_union_top20_mass_l1": 0.0,
    } for _ in range(8)]}


def context(length=2048):
    return {"source_sha256": SHA, "actual_reference_prompt_tokens": length,
        "qualified_length_type": "real full-model CPU token-sequential prefill",
        "pass": True, "next_token_exact": True, "top20_shared_ids": 20,
        "max_shared_logprob_error": 0.0, "rust_published_roots": 1}


def layers():
    return {"checkpoint_sha256": SHA,
        "independent_reference_provenance": "unit-only independent oracle control",
        "passed": True, "failed_stages": 0, "stage_count": 146,
        "all_element_count": 146 * 1024}


def write_phase1(root):
    for model, sha in GATE.MODELS.items():
        folder = root / model
        folder.mkdir()
        sample = numeric()
        sample["sha256"] = sha
        (folder / "numeric-32.json").write_text(json.dumps(sample))
        for length in GATE.PHASE1_CONTEXT_COUNTS:
            sample = context(length)
            sample["source_sha256"] = sha
            (folder / f"context-{length}.json").write_text(json.dumps(sample))
        sample = layers()
        sample["checkpoint_sha256"] = sha
        (folder / "independent-layers.json").write_text(json.dumps(sample))
        (folder / "state-invariants.json").write_text(json.dumps({
            "checkpoint_sha256": sha, "passed": True, "cross_session_rejection": True,
            "atomic_rollback": True, "replay_equivalence": True, "chunk_equivalence": True}))


class CpuQualificationGateTests(unittest.TestCase):
    def accepted(self, function, record, *args):
        problems = []
        self.assertTrue(function(record, MODEL, SHA, *args, problems), problems)
        self.assertEqual(problems, [])

    def rejected(self, function, record, *args):
        problems = []
        self.assertFalse(function(record, MODEL, SHA, *args, problems))
        self.assertTrue(problems)

    def test_original_thresholds_accept_boundaries_and_reject_crossings(self):
        for field, limit in (("max_shared_logprob_absolute_error", 0.45),
                             ("max_top1_probability_absolute_error", 0.05),
                             ("max_truncated_union_top20_mass_l1", 0.06)):
            with self.subTest(field=field):
                sample = numeric()
                sample["cases"][0][field] = limit
                self.accepted(GATE.numeric_ok, sample)
                sample["cases"][0][field] = math.nextafter(limit, math.inf)
                self.rejected(GATE.numeric_ok, sample)
        sample = context()
        sample["top20_shared_ids"] = 14
        sample["max_shared_logprob_error"] = 0.45
        self.accepted(GATE.context_ok, sample, 2048)
        sample["max_shared_logprob_error"] = math.nextafter(0.45, math.inf)
        self.rejected(GATE.context_ok, sample, 2048)

    def test_numerical_errors_require_finite_nonnegative_real_values(self):
        invalid = (float("nan"), float("inf"), -float("inf"), "0", True, False,
                   None, -0.001, 10**1000)
        for field in ("max_shared_logprob_absolute_error",
                      "max_top1_probability_absolute_error",
                      "max_truncated_union_top20_mass_l1"):
            for value in invalid:
                with self.subTest(field=field, value=repr(value)):
                    sample = numeric()
                    sample["cases"][7][field] = value
                    self.rejected(GATE.numeric_ok, sample)
            sample = numeric()
            del sample["cases"][7][field]
            self.rejected(GATE.numeric_ok, sample)

    def test_numeric_counts_are_integers_and_have_real_populations(self):
        for field in ("hard_top1_mismatches", "missing_reference_top5_steps",
                      "unverified_fragment_positions"):
            for value in (False, True, 0.0, "0", -1, 1, 33, None):
                with self.subTest(field=field, value=value):
                    sample = numeric()
                    sample["cases"][0][field] = value
                    self.rejected(GATE.numeric_ok, sample)
        for value in (False, True, 14.0, "20", -1, 13, 21, None):
            sample = numeric()
            sample["cases"][0]["minimum_top20_overlap"] = value
            self.rejected(GATE.numeric_ok, sample)
        sample = numeric()
        sample["cases"][0]["minimum_top20_overlap"] = 14
        self.accepted(GATE.numeric_ok, sample)

    def test_numeric_corpus_cannot_be_partial_or_mislabelled(self):
        for count in (0, 7, 9):
            sample = numeric()
            sample["cases"] = (sample["cases"] * 2)[:count]
            self.rejected(GATE.numeric_ok, sample)
        for value in (31, 33, 32.0, "32", True, None):
            sample = numeric()
            sample["cases"][3]["tokens"] = value
            self.rejected(GATE.numeric_ok, sample)
        for value in (False, True, 0.0, "0", 1, -1):
            sample = numeric()
            sample["start_case"] = value
            self.rejected(GATE.numeric_ok, sample)
        for value in (None, 1, [], "case"):
            sample = numeric()
            sample["cases"][3] = value
            self.rejected(GATE.numeric_ok, sample)

    def test_pass_flags_must_be_boolean_true(self):
        for value in (False, 1, "true", [], None):
            for field in ("passed", "case_pass"):
                sample = numeric()
                if field == "case_pass":
                    sample["cases"][0]["pass"] = value
                else:
                    sample[field] = value
                self.rejected(GATE.numeric_ok, sample)
            for field in ("pass", "next_token_exact"):
                sample = context()
                sample[field] = value
                self.rejected(GATE.context_ok, sample, 2048)
            sample = layers()
            sample["passed"] = value
            self.rejected(GATE.layer_ok, sample)

    def test_context_errors_and_top20_coverage_fail_closed(self):
        for value in (float("nan"), float("inf"), -float("inf"), "0", True,
                      False, None, -0.001, 10**1000):
            sample = context()
            sample["max_shared_logprob_error"] = value
            self.rejected(GATE.context_ok, sample, 2048)
        for value in (False, True, "20", 20.0, -1, 13, 21, None):
            sample = context()
            sample["top20_shared_ids"] = value
            self.rejected(GATE.context_ok, sample, 2048)

    def test_context_requires_actual_length_and_committed_roots(self):
        for value in (2047, 2049, 2048.0, "2048", True, None):
            sample = context()
            sample["actual_reference_prompt_tokens"] = value
            self.rejected(GATE.context_ok, sample, 2048)
        for value in (False, True, 0, 2049, 1.0, "1", None):
            sample = context()
            sample["rust_published_roots"] = value
            self.rejected(GATE.context_ok, sample, 2048)
        for marker in ("synthetic operator/position check", "synthetic 2048 operator steps", None):
            sample = context()
            sample["qualified_length_type"] = marker
            self.rejected(GATE.context_ok, sample, 2048)

    def test_layer_counts_are_strict_and_complete(self):
        self.accepted(GATE.layer_ok, layers())
        for field, invalid in (("failed_stages", (False, 0.0, "0", -1, 1, None)),
                ("stage_count", (True, 145, 146.0, "146", float("inf"), None)),
                ("all_element_count", (True, 146 * 1024 - 1, 146 * 1024.0, "149504", None))):
            for value in invalid:
                with self.subTest(field=field, value=value):
                    sample = layers()
                    sample[field] = value
                    self.rejected(GATE.layer_ok, sample)

    def test_wrong_or_malformed_checkpoint_identities_are_refused(self):
        for function, make, field, args in ((GATE.numeric_ok, numeric, "sha256", ()),
                (GATE.context_ok, context, "source_sha256", (2048,)),
                (GATE.layer_ok, layers, "checkpoint_sha256", ())):
            for value in (None, True, [], {}, "0" * 64):
                sample = make()
                sample[field] = value
                self.rejected(function, sample, *args)
        sample = numeric()
        sample["sha256"] = SHA.upper()
        self.accepted(GATE.numeric_ok, sample)

    def test_empty_and_phase1_populations_include_2048_for_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = GATE.check(root, "phase1")
            self.assertFalse(empty["ready_for_gpu"])
            self.assertEqual(len(empty["unmet_gates"]), 18)
            write_phase1(root)
            phase1 = GATE.check(root, "phase1")
            self.assertTrue(phase1["ready_for_gpu"], phase1["unmet_gates"])
            self.assertEqual(len(phase1["completed_gates"]), 18)
            self.assertEqual(len(phase1["deferred_gates"]), 4)
            self.assertFalse(phase1["extended_context_qualified"])
            extended = GATE.check(root, "extended")
            self.assertFalse(extended["ready_for_gpu"])
            self.assertEqual(len(extended["unmet_gates"]), 4)
            for model in GATE.MODELS:
                (root / model / "context-2048.json").unlink()
            legacy = GATE.check(root, "phase1")
            self.assertFalse(legacy["ready_for_gpu"])
            self.assertEqual(len(legacy["completed_gates"]), 16)
            self.assertEqual(len(legacy["unmet_gates"]), 2)
            self.assertTrue(all("context-2048.json" in x for x in legacy["unmet_gates"]))

    def test_nonobject_json_and_partial_files_never_clear_a_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_phase1(root)
            for name in ("numeric-32.json", "context-2048.json",
                         "independent-layers.json", "state-invariants.json"):
                path = root / MODEL / name
                original = path.read_text()
                for text in ("null", "[]", '"report"', "true", "{", ""):
                    with self.subTest(name=name, text=text):
                        path.write_text(text)
                        report = GATE.check(root, "phase1")
                        self.assertFalse(report["ready_for_gpu"])
                        self.assertTrue(report["unmet_gates"])
                path.write_text(original)

    def test_nonstandard_json_nan_cannot_bypass_full_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_phase1(root)
            path = root / MODEL / "numeric-32.json"
            sample = json.loads(path.read_text())
            sample["cases"][0]["max_shared_logprob_absolute_error"] = float("nan")
            path.write_text(json.dumps(sample))
            self.assertIn("NaN", path.read_text())
            self.assertFalse(GATE.check(root, "phase1")["ready_for_gpu"])

    def test_duplicate_json_keys_never_replace_failed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_phase1(root)
            for name, flag in (("numeric-32.json", "passed"), ("context-2048.json", "pass"),
                               ("independent-layers.json", "passed"),
                               ("state-invariants.json", "passed")):
                path = root / MODEL / name
                original = path.read_text()
                samples = [
                    '{"' + flag + '":false,' + original[1:],
                    '{"' + flag + '":true,' + original[1:],
                    original[:-1] + ',"unused":{"nested":1,"nested":2}}',
                ]
                if flag == "passed":
                    samples.append('{"pa\\u0073sed":false,' + original[1:])
                for text in samples:
                    with self.subTest(name=name, text=text[:60]):
                        path.write_text(text)
                        report = GATE.check(root, "phase1")
                        self.assertFalse(report["ready_for_gpu"])
                        self.assertTrue(any("duplicate JSON key" in item
                                            for item in report["unmet_gates"]))
                path.write_text(original)

    def test_unknown_metadata_cannot_hide_nonfinite_or_overflowing_json_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_phase1(root)
            for name in ("numeric-32.json", "context-2048.json",
                         "independent-layers.json", "state-invariants.json"):
                path = root / MODEL / name
                original = path.read_text()
                for number in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
                    with self.subTest(name=name, number=number):
                        path.write_text(original[:-1] + ',"unused":[{"value":' + number + '}]}')
                        report = GATE.check(root, "phase1")
                        self.assertFalse(report["ready_for_gpu"])
                        self.assertTrue(any("invalid JSON evidence" in item
                                            for item in report["unmet_gates"]))
                path.write_text(original[:-1] + ',"unused":[1e308,-1e308,"NaN",42]}')
                self.assertTrue(GATE.check(root, "phase1")["ready_for_gpu"])
                path.write_text(original)

    def test_state_flags_are_not_truthy_substitutes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_phase1(root)
            path = root / MODEL / "state-invariants.json"
            original = json.loads(path.read_text())
            for field in ("passed", "cross_session_rejection", "atomic_rollback",
                          "replay_equivalence", "chunk_equivalence"):
                for value in (False, 1, "true", None):
                    sample = copy.deepcopy(original)
                    sample[field] = value
                    path.write_text(json.dumps(sample))
                    self.assertFalse(GATE.check(root, "phase1")["ready_for_gpu"])


if __name__ == "__main__":
    unittest.main()

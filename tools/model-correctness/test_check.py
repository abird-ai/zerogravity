"""Source-only protocol counterexamples; no trained-model evidence."""
import unittest
from pathlib import Path
import tempfile

from check import compare, read, unique


def document():
    return {"schema": "zg.model-output.v2", "complete": True, "text_representation": "utf8",
            "owner_closure": {"complete": True, "errors": []},
            "model": {"source_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
                      "template_sha256": "c" * 64, "precision": "bf16", "quantization": "none"},
            "execution": {"backend": "cuda", "runtime_revision": "independent-original",
                          "numerical_profile": "fixture-only", "runtime_artifact_sha256": "d" * 64,
                          "runtime_artifact_kind": "executable",
                          "context_capacity_tokens": 512, "vocabulary_size": 8, "batch_invariant": False},
            "cases": [{"id": "decode_after_prompt", "input_ids": [1] * 48,
                       "phase": "decode", "prompt_tokens": 47, "request_batch_size": 1,
                       "input_tokens": 1, "cached_tokens": 47,
                       "generation": {"token_ids": [2, 3], "text": "fixture", "top_logprobs": None}}]}


class Protocol(unittest.TestCase):
    def test_all_model_labels_and_backends_share_comparison(self):
        for backend in ("cpu", "cuda", "hip", "vulkan"):
            ref, candidate = document(), document()
            candidate["execution"]["backend"] = backend
            candidate["execution"]["runtime_revision"] = "zerogravity-fixture"
            result = compare(ref, candidate)
            self.assertTrue(result["comparison_passed"])
            self.assertFalse(result["inference_qualified"])
            self.assertFalse(result["operator_state_physical_qualified"])

    def test_identity_and_precision_substitutions_refuse(self):
        for key, value in (("source_sha256", "e" * 64), ("precision", "f16"),
                           ("quantization", "q4"), ("tokenizer_sha256", "f" * 64)):
            other = document()
            other["model"][key] = value
            with self.assertRaises(ValueError):
                compare(document(), other)

    def test_missing_reordered_duplicate_and_interrupted_cases_refuse(self):
        other = document()
        other["complete"] = False
        with self.assertRaises(ValueError):
            compare(document(), other)
        for cases in ([], [document()["cases"][0]] * 2):
            other = document()
            other["cases"] = cases
            with self.assertRaises(ValueError):
                compare(document(), other)
        other = document()
        other["cases"][0]["id"] = "different"
        with self.assertRaises(ValueError):
            compare(document(), other)

    def test_fresh_prompt_cannot_substitute_cached_decode(self):
        other = document()
        other["cases"][0].update(phase="prefill", prompt_tokens=48, input_tokens=48, cached_tokens=0)
        with self.assertRaises(ValueError):
            compare(document(), other)

    def test_phase_and_counts_describe_actual_history(self):
        for edit in ({"phase": "unknown"}, {"phase": "prefill"},
                     {"prompt_tokens": 48}, {"prompt_tokens": True},
                     {"request_batch_size": 0}, {"request_batch_size": True},
                     {"request_batch_size": None}):
            other = document()
            other["cases"][0].update(edit)
            with self.assertRaises(ValueError):
                compare(document(), other)

    def test_batch_width_is_reported_without_imposing_bit_invariance(self):
        other = document()
        other["cases"][0]["request_batch_size"] = 4
        result = compare(document(), other)
        self.assertTrue(result["comparison_passed"])
        self.assertEqual(result["cases"][0]["reference_request_batch_size"], 1)
        self.assertEqual(result["cases"][0]["candidate_request_batch_size"], 4)

    def test_complete_prefill_and_final_prompt_chunk(self):
        for cached in (0, 24):
            ref, other = document(), document()
            for doc in (ref, other):
                doc["cases"][0].update(phase="prefill", prompt_tokens=48,
                                       cached_tokens=cached, input_tokens=48 - cached)
            self.assertTrue(compare(ref, other)["comparison_passed"])

    def test_owner_errors_context_truncation_and_invariance_change_refuse(self):
        for edit in (lambda d: d["owner_closure"].update(errors=["incomplete EOF"]),
                     lambda d: d["execution"].update(context_capacity_tokens=49),
                     lambda d: d["execution"].update(batch_invariant=True),
                     lambda d: d["cases"][0].update(cached_tokens=True)):
            other = document()
            edit(other)
            with self.assertRaises(ValueError):
                compare(document(), other)

    def test_full_vocab_required_and_metrics_do_not_widen_generation_acceptance(self):
        ref, candidate = document(), document()
        for d in (ref, candidate):
            d["cases"][0].update(full_logits=[0.0] * 8, target_id=2)
        result = compare(ref, candidate)
        self.assertIn("logits_metrics", result["cases"][0])
        candidate["cases"][0]["full_logits"] = [0.0] * 5
        with self.assertRaises(ValueError):
            compare(ref, candidate)

    def test_duplicate_json_keys_refuse(self):
        with self.assertRaises(ValueError):
            unique([("passed", True), ("passed", False)])

    def test_both_logit_targets_require_strict_bounded_integers(self):
        for side in (0, 1):
            for invalid in (True, 2.0, -1, 8, None):
                pair = [document(), document()]
                for d in pair:
                    d["cases"][0].update(full_logits=[0.0] * 8, target_id=2)
                pair[side]["cases"][0]["target_id"] = invalid
                with self.assertRaises(ValueError):
                    compare(*pair)

    def test_digest_only_observations_are_explicit_and_never_reconstructed_text(self):
        ref, candidate = document(), document()
        for d in (ref, candidate):
            d["text_representation"] = "sha256"
            d["cases"][0]["generation"]["text"] = "e" * 64
        result = compare(ref, candidate, policy="exact")
        self.assertTrue(result["comparison_passed"])
        self.assertEqual(result["text_equality_kind"], "sha256")
        candidate["text_representation"] = "utf8"
        with self.assertRaises(ValueError):
            compare(ref, candidate)

    def test_out_of_vocabulary_logprob_choices_refuse(self):
        other = document()
        other["cases"][0]["generation"]["top_logprobs"] = [[{"token_id": 8, "logprob": -1}]] * 2
        with self.assertRaises(ValueError):
            compare(document(), other)

    def test_overflow_json_numbers_in_unrecognized_metadata_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for text in ('{"extra": 1e999}', '{"extra": -1e999}', '{"extra": NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read(path)


if __name__ == "__main__":
    unittest.main()

"""Source-free comparator tests; no model, SDK, network or device access."""
import math
import unittest

from accuracy import backend_nmse, compare_generation, full_logits_metrics


def generation(ids, text="same", choices=None):
    return {"token_ids": ids, "text": text, "top_logprobs": choices}


def row(*tokens):
    return [{"token_id": token, "logprob": -float(rank + 1)} for rank, token in enumerate(tokens)]


class GenerationTests(unittest.TestCase):
    def test_exact_means_both_ids_and_text(self):
        original = generation([1, 2])
        self.assertTrue(compare_generation(original, original, policy="exact")["accepted"])
        changed = generation([1, 2], "different")
        result = compare_generation(original, changed, policy="exact")
        self.assertFalse(result["accepted"])
        self.assertTrue(result["exact_ids"])
        self.assertFalse(result["exact_text"])
        warning = compare_generation(original, changed)
        self.assertTrue(warning["accepted"])
        self.assertTrue(warning["text_warning"])

    def test_only_first_divergence_has_a_common_history(self):
        reference = generation([1, 2, 4], choices=[row(1, 2), row(2, 3), row(4, 5)])
        candidate = generation([1, 3, 9], "other", [row(1, 2), row(3, 2), row(9, 8)])
        result = compare_generation(reference, candidate, top_k=2)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["mismatch"])
        self.assertEqual(result["first_divergence"], 1)
        self.assertEqual(result["matched_prefix_tokens"], 1)
        self.assertEqual(result["compared_token_count"], 2)
        self.assertEqual(result["numerical_comparison_end"], 2)
        self.assertFalse(result["divergent_tail_compared"])
        self.assertFalse(compare_generation(reference, candidate, policy="exact", top_k=2)["accepted"])

    def test_mutual_membership_includes_returned_selected_extras(self):
        reference = generation([1], choices=[row(1, 2, 3)])
        candidate = generation([3], choices=[row(3, 1, 2)])
        self.assertTrue(compare_generation(reference, candidate, top_k=2)["accepted"])
        self.assertTrue(compare_generation(reference, candidate, top_k=3)["accepted"])
        unrelated = generation([8], choices=[row(8, 9, 10)])
        self.assertFalse(compare_generation(reference, unrelated, top_k=2)["accepted"])
        missing = compare_generation(generation([1]), generation([2]))
        self.assertFalse(missing["accepted"])
        self.assertIsNone(missing["numerical_comparison_end"])

    def test_ties_and_row_order_preserve_returned_map_membership(self):
        reference = generation([2], choices=[[{"token_id": t, "logprob": -1.0} for t in [3, 2, 1]]])
        candidate = generation([3], choices=[row(3, 2, 1)])
        self.assertTrue(compare_generation(reference, candidate, top_k=2)["accepted"])
        reference["top_logprobs"][0].reverse()
        self.assertTrue(compare_generation(reference, candidate, top_k=2)["accepted"])

    def test_pinned_vllm_selected_extra_counterexample(self):
        reference = generation([2], choices=[[{"token_id": token, "logprob": score}
                              for token, score in zip([1, 3, 4, 5, 6, 2], [0, -1, -2, -3, -4, -4])]])
        candidate = generation([6], choices=[row(6, 2, 3, 4, 5)])
        result = compare_generation(reference, candidate)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["top_logprobs_semantics"], "returned_top_k_plus_selected")
        reference["top_logprobs"][0].append({"token_id": 7, "logprob": -5})
        with self.assertRaises(ValueError):
            compare_generation(reference, candidate)

    def test_logprobs_are_nonpositive_with_zero_allowed(self):
        valid = generation([1], choices=[row(1, 2)])
        valid["top_logprobs"][0][0]["logprob"] = 0.0
        self.assertTrue(compare_generation(valid, valid, top_k=2)["accepted"])
        valid["top_logprobs"][0][0]["logprob"] = 0.01
        with self.assertRaises(ValueError):
            compare_generation(valid, valid, top_k=2)

    def test_rejects_truncation_malformed_data_and_bad_tail(self):
        good = generation([1], choices=[row(1, 2)])
        for bad in [generation([]), generation([True]), generation([-1]),
                    generation([1, 2]), generation([1], choices=[]),
                    generation([1], choices=[row(1)]), generation([1], choices=[row(1, 1)]),
                    generation([1], choices=[row(2, 3)]), {"token_ids": [1], "text": "same"}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compare_generation(good, bad, top_k=2)
        for nonfinite in [math.nan, math.inf, -math.inf]:
            bad = generation([3, 4], choices=[row(3, 1), row(4, 5)])
            bad["top_logprobs"][1][0]["logprob"] = nonfinite
            with self.assertRaises(ValueError):
                compare_generation(generation([1, 2], choices=[row(1, 3), row(2, 4)]), bad, top_k=2)


class MetricsTests(unittest.TestCase):
    def metrics(self, a, b, target=0):
        return full_logits_metrics(a, b, target, vocabulary_size=len(a))

    def test_identical_and_shifted_distributions(self):
        identity = self.metrics([0.0, 0.0], [0.0, 0.0])
        self.assertEqual(identity["KL_ref_candidate"], 0.0)
        self.assertEqual(identity["rms_probability_error"], 0.0)
        self.assertAlmostEqual(identity["reference_nll"], math.log(2))
        shifted = self.metrics([1.0, 2.0, 3.0], [1001.0, 1002.0, 1003.0])
        self.assertEqual(shifted["KL_ref_candidate"], 0.0)
        self.assertEqual(shifted["delta_nll"], 0.0)
        self.assertEqual(shifted["rms_probability_error"], 0.0)
        self.assertEqual(shifted["max_absolute_error"], 1000.0)
        self.assertEqual(shifted["rms_error"], 1000.0)
        self.assertNotIn("accepted", shifted)

    def test_kl_orientation_target_nll_and_errors(self):
        a, b = [math.log(0.9), math.log(0.1)], [math.log(0.5), math.log(0.5)]
        result = self.metrics(a, b, 1)
        expected = 0.9 * math.log(1.8) + 0.1 * math.log(0.2)
        self.assertAlmostEqual(result["KL_ref_candidate"], expected)
        self.assertNotAlmostEqual(result["KL_ref_candidate"], self.metrics(b, a)["KL_ref_candidate"])
        self.assertAlmostEqual(result["delta_nll"], math.log(0.2))
        self.assertAlmostEqual(result["rms_probability_error"], 0.4)
        self.assertTrue(result["top1_equal"])
        self.assertFalse(self.metrics([2.0, 1.0], [1.0, 2.0])["top1_equal"])

    def test_finite_extremes_do_not_overflow_softmax_or_squares(self):
        result = self.metrics([1000.0, -1000.0], [-1000.0, 1000.0])
        self.assertEqual(result["KL_ref_candidate"], 2000.0)
        self.assertEqual(result["rms_error"], 2000.0)
        huge = self.metrics([1e300, 1e300], [0.0, 0.0])
        self.assertEqual(huge["rms_error"], 1e300)
        self.assertEqual(huge["KL_ref_candidate"], 0.0)

    def test_missing_full_vocabulary_alignment_and_nonfinite_rejected(self):
        with self.assertRaises(TypeError):
            full_logits_metrics([0.0], [0.0], 0)
        for a, b, target, vocabulary in [([], [], 0, 0), ([0.0], [0.0, 1.0], 0, 1),
                                      ([0.0], [0.0], 0, 248320), ([0.0], [0.0], 1, 1),
                                      ([True], [0.0], 0, 1), ([0.0], [0.0], True, 1)]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                full_logits_metrics(a, b, target, vocabulary_size=vocabulary)
        for bad in [math.nan, math.inf, -math.inf]:
            with self.assertRaises(ValueError):
                self.metrics([bad], [0.0])
            with self.assertRaises(ValueError):
                self.metrics([0.0], [bad])

    def test_llama_nmse_candidate_denominator_and_zero_energy(self):
        self.assertEqual(backend_nmse([2.0], [1.0]), 1.0)
        self.assertEqual(backend_nmse([1.0], [2.0]), 0.25)
        self.assertEqual(backend_nmse([0.0, 0.0], [0.0, 0.0]), 0.0)
        self.assertEqual(backend_nmse([2e-300], [1e-300]), 1.0)
        self.assertEqual(backend_nmse([2e300], [1e300]), 1.0)
        self.assertEqual(backend_nmse([-1e308], [1e308]), 4.0)
        self.assertEqual(backend_nmse([0.0] * 999 + [5e-324], [0.0] * 999 + [5e-324]), 0.0)
        self.assertEqual(backend_nmse([0.0] * 999 + [0.0], [0.0] * 999 + [5e-324]), 1.0)
        for a, b in [([1.0], [0.0]), ([], []), ([0.0], [math.nan]), ([1.0], [1.0, 2.0])]:
            with self.assertRaises(ValueError):
                backend_nmse(a, b)


if __name__ == "__main__":
    unittest.main()

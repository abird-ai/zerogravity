import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import io

from contract import GENERATION, MODEL, REVISION, read_json, read_report_json, request
from run_reference import OriginalLogs, bind_command_evidence


def sample(tokens=47):
    return {"schema": "zerogravity.qwen35-cuda-reference.request.v1", "model": MODEL,
            "source_revision": REVISION, "phase": "prefill", "prompt_tokens": tokens,
            "cached_tokens": 0, "request_batch_size": 1, "prompt": "Actual prompt text",
            "generation": dict(GENERATION)}


class ContractTests(unittest.TestCase):
    def test_fresh_lengths_and_tokenized_ids(self):
        for count in range(47, 51):
            value = sample(count)
            self.assertIs(request(value), value)
            value["prompt_token_ids"] = list(range(count))
            self.assertIs(request(value, tokenized=True), value)

    def test_cached_batch_other_model_and_unknown_fields_refuse(self):
        for key, value in (("phase", "decode"), ("cached_tokens", 47),
                           ("request_batch_size", 4), ("model", "Qwen/Qwen3.5-2B"),
                           ("source_revision", "changed"), ("prompt_tokens", 51),
                           ("prompt_tokens", True), ("cached_tokens", False),
                           ("request_batch_size", True), ("prompt", ""), ("unknown", 0)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                request(dict(sample(), **{key: value}))

    def test_generation_substitutions_refuse(self):
        for key, value in (("max_new_tokens", True), ("max_new_tokens", 4),
                           ("eos_token_id", 248044), ("temperature", float("nan")),
                           ("temperature", False), ("allowed_token_ids", [1])):
            candidate = sample()
            candidate["generation"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                request(candidate)

    def test_incomplete_noninteger_or_foreign_ids_refuse(self):
        for ids in ([1] * 46, [False] * 47, [248320] * 47, [1.0] * 47):
            candidate = dict(sample(), prompt_token_ids=ids)
            with self.assertRaises(ValueError):
                request(candidate, tokenized=True)

    def test_json_duplicate_nonfinite_and_size_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            for body in ('{"a":1,"a":2}', '{"a":NaN}', " " * (1024 * 1024 + 1)):
                path.write_text(body)
                with self.assertRaises(ValueError):
                    read_json(path)

    def test_large_original_report_keeps_separate_bounded_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory-validation.json"
            value = {"original_report": "x" * (1024 * 1024 + 1)}
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                read_json(path)
            self.assertEqual(read_report_json(path), value)
            with path.open("r+b") as stream:
                stream.truncate(64 * 1024 * 1024 + 1)
            with self.assertRaises(ValueError):
                read_report_json(path)

    def test_original_contract_tail_is_unchanged(self):
        # Original validators keep complete source/state/fullV semantics.
        pins = read_json(Path(__file__).parent / "original-pins.json")
        self.assertIn("package/validate_trajectory.py", pins)
        self.assertIn("package/capture_matrix.py", pins)
        self.assertIn("package/collector.cpp", pins)

    def test_owned_command_failures_never_become_success(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            stdout, stderr = directory / "stdout", directory / "stderr"
            stdout.write_bytes(b"out")
            stderr.write_bytes(b"err")
            complete = {"evidence_complete": True, "result_persisted": True, "leader_reaped": True,
                "owned_process_group_gone": True, "status": "finished", "errors": [], "complete": True,
                "leader_reap_attempts": 1, "all_nonzero_signals_before_reap": True, "timed_out": False,
                "signals": [],
                "returncode": 1, "stdout": {"path": str(stdout)}, "stderr": {"path": str(stderr)}}
            current = dict(complete)
            lifecycle = SimpleNamespace(owned_command=lambda *args, **kwargs: current)
            helper = SimpleNamespace()
            bind_command_evidence(helper, lifecycle, directory / "commands")
            result = helper.command(["podman", "container", "exists", "owned"])
            self.assertEqual((result.returncode, result.stdout, result.stderr), (1, b"out", b"err"))
            for key, value in (("evidence_complete", False), ("result_persisted", False),
                               ("status", "command_error"), ("errors", [{"stage": "wait"}]),
                               ("complete", False), ("leader_reap_attempts", 2),
                               ("signals", [{"signal": "SIGTERM"}]),
                               ("all_nonzero_signals_before_reap", False), ("timed_out", True)):
                current = dict(complete, **{key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    helper.command(["podman", "stop", "owned"])

    def test_cleanup_command_still_runs_when_checkpoint_storage_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def owned(argv, label, cid, timeout, checkpoint, **kwargs):
                calls.append(list(argv))
                self.assertFalse(checkpoint(label + "_intent", {"status": "intent"}))
                return {"evidence_complete": False}

            helper = SimpleNamespace()
            bind_command_evidence(helper, SimpleNamespace(owned_command=owned), Path(directory) / "commands")
            with patch("run_reference.write", side_effect=OSError("evidence storage full")):
                with self.assertRaises(ValueError):
                    helper.command(["podman", "stop", "exact-owned-cid"])
            self.assertEqual(calls, [["podman", "stop", "exact-owned-cid"]])

    def test_partial_stream_open_closes_both_owned_pipes(self):
        class FirstLog:
            def __init__(self, pipe, path):
                if path.name == "stderr.log":
                    raise OSError("stderr open failed")
                self.pipe = pipe

            def finish(self):
                self.pipe.close()

        process = SimpleNamespace(stdout=io.BytesIO(), stderr=io.BytesIO())
        logs = OriginalLogs(SimpleNamespace(Log=FirstLog))
        with self.assertRaises(OSError):
            logs.open(process, Path("/owned"))
        logs.finish()
        self.assertTrue(process.stdout.closed and process.stderr.closed)

    def test_prepared_contract_keeps_real_prefix_and_defined_priors(self):
        # Import the successor contract against explicit test data, without a runtime.
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            value = dict(sample(50), prompt_token_ids=list(range(50)))
            (directory / "request.json").write_text(json.dumps(value))
            here = Path(__file__).parent
            (directory / "matrix_contract.py").write_bytes((here / "matrix_contract.py").read_bytes())
            spec = importlib.util.spec_from_file_location("reference_request", here / "contract.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            import sys
            old = sys.modules.get("reference_request")
            sys.modules["reference_request"] = module
            try:
                spec = importlib.util.spec_from_file_location("prompt_contract_test", directory / "matrix_contract.py")
                contract = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(contract)
                self.assertEqual(contract.rows_for(4), (50, 1, 1, 1))
                self.assertEqual(contract.sequences_for(4), (50, 51, 52, 53))
                priors = contract.expected_prior_records(3, 4)
                self.assertEqual(sum(r[0] == 1 for r in priors), 50)
                self.assertEqual(sum(r[0] == 2 for r in priors), 51)
                self.assertEqual(sum(r[0] == 3 for r in priors), 52)
                self.assertTrue(all(r[0] > 0 for r in priors))
            finally:
                if old is None:
                    sys.modules.pop("reference_request", None)
                else:
                    sys.modules["reference_request"] = old


if __name__ == "__main__":
    unittest.main()

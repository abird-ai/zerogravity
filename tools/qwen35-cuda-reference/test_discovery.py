"""Pure held-ancestry/ABI/byte-corruption controls; no provider or model execution."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from discovery import HeldFiles, archive, command_complete, compare_archives, compose_layouts, layout_key, streams
from run_reference import stage_pins


def row(image="a" * 64, params=None):
    return {"callback_name": "fixture", "name": "fixture", "module_sha256": image,
            "params": [[0, 8], [8, 4]] if params is None else params}


class DiscoveryTests(unittest.TestCase):
    def test_separate_frozen_stage_manifests(self):
        original, captured = stage_pins("discover"), stage_pins("capture")
        self.assertEqual((len(original), len(captured)), (42, 50))
        for relative in ("package/primary-runtime-pins.json", "package/capture_matrix.py",
                         "package/all42.py", "package/worker_observer.py", "package/primary_guard.py"):
            self.assertEqual(original[relative], captured[relative])
        self.assertNotEqual(original["package/collector.cpp"], captured["package/collector.cpp"])
        with self.assertRaises(ValueError):
            stage_pins("other")

    def test_held_raw_rejects_content_changes_path_swap_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mutation in ("content", "replacement", "symlink"):
                path = root / mutation
                path.write_bytes(b"actual retained fixture")
                with HeldFiles() as held:
                    held.hold(path)
                    if mutation == "content":
                        path.write_bytes(b"actual retained fixturE")
                    else:
                        path.unlink()
                        path.write_bytes(b"actual retained fixture") if mutation == "replacement" else path.symlink_to(root / "content")
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        held.check()
                if mutation == "symlink":
                    with HeldFiles() as held, self.assertRaises(OSError):
                        held.hold(path)

    def test_raw_metadata_must_cover_complete_extent(self):
        with tempfile.TemporaryDirectory() as directory, HeldFiles() as held:
            root = Path(directory)
            raw = b"01234567"
            (root / "raw.bin").write_bytes(raw)
            receipt = {"file": "raw.bin", "bytes": 8, "sha256": hashlib.sha256(raw).hexdigest(),
                       "records": [{"offset": 0, "bytes": 8}]}
            archive(held, root, receipt)
            for key, value in (("bytes", 9), ("sha256", "b" * 64)):
                changed = copy.deepcopy(receipt)
                changed[key] = value
                with self.subTest(field=key), self.assertRaises(ValueError):
                    archive(held, root, changed)
            receipt["records"][0]["offset"] = 1
            with self.assertRaises(ValueError):
                archive(held, root, receipt)

    def test_layout_union_preserves_actual_whole_image_and_refuses_conflict(self):
        def discovered(rows):
            return SimpleNamespace(root=Path("/pure-fixture"), layout={"layouts": rows},
                                   check=lambda: None, pins=lambda: [])
        catalog = compose_layouts([discovered([row()]), discovered([row(), row("b" * 64)])])
        self.assertEqual(len(catalog["layouts"]), 2)
        self.assertFalse(catalog["model_qualified"])
        with self.assertRaises(ValueError):
            compose_layouts([discovered([row()]), discovered([row(params=[[0, 16]])])])
        for changed in (dict(row(), guessed=True), row(params=[[0, 8], [4, 4]]),
                        row(params=[[4090, 8]]), row(image="invented")):
            with self.subTest(row=changed), self.assertRaises(ValueError):
                layout_key(changed)

    def test_full_raw_comparison_catches_a_changed_word_even_with_same_metadata(self):
        with tempfile.TemporaryDirectory() as directory, HeldFiles() as held:
            root = Path(directory)
            (root / "expected").write_bytes(b"\x00\x00\x80\x3f")
            (root / "actual").write_bytes(b"\x00\x00\x80\x3f")
            expected = {("plain", "fullV"): {"item": held.hold(root / "expected"), "records": [{"call": 0}]}}
            current = {("plain", "fullV"): {"item": held.hold(root / "actual"), "records": [{"call": 0}]}}
            self.assertEqual(compare_archives(expected, current), 4)
            # Preserve declared record metadata: equality must inspect every raw byte.
            (root / "actual").write_bytes(b"\x01\x00\x80\x3f")
            with self.assertRaises(ValueError):
                compare_archives(expected, current)

    def test_owned_validator_errors_cannot_be_promoted_by_exit_zero(self):
        record = {"status": "finished", "returncode": 0, "evidence_complete": True,
                  "result_persisted": True, "complete": True, "leader_reaped": True,
                  "owned_process_group_gone": True, "leader_reap_attempts": 1,
                  "all_nonzero_signals_before_reap": True, "signals": [], "errors": [], "timed_out": False}
        command_complete({"record": record})
        for key, value in (("signals", ["TERM"]), ("errors", ["capture failed"]),
                           ("error_type", "OSError"), ("leader_reap_attempts", 2), ("returncode", 1)):
            with self.subTest(field=key), self.assertRaises(ValueError):
                command_complete({"record": dict(record, **{key: value})})

    def test_retained_streams_bind_actual_bytes_path_eof_and_complete_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {}
            for name, raw in (("stdout", b"retained original output"), ("stderr", b"")):
                path = root / ("validation." + name + ".log")
                path.write_bytes(raw)
                records[name] = {"path": str(path), "bytes_seen": len(raw), "bytes_retained": len(raw),
                    "capture_complete": True, "sha256_retained": hashlib.sha256(raw).hexdigest(),
                    "eof": True, "reader_complete": True, "reader_error": None, "truncated": False,
                    "eof_observed": {"monotonic_ns": 1}}
            with HeldFiles() as held:
                self.assertEqual(streams(held, records, root, "validation.", True), 24)
                held.check()
            for key, value in (("capture_complete", False), ("eof", False), ("reader_complete", False),
                               ("reader_error", "lost pipe"), ("truncated", True), ("bytes_retained", 2),
                               ("eof_observed", {}), ("sha256_retained", "b" * 64),
                               ("path", str(root / "validation.stderr.log"))):
                changed = copy.deepcopy(records)
                changed["stdout"][key] = value
                with self.subTest(field=key), HeldFiles() as held, self.assertRaises(ValueError):
                    streams(held, changed, root, "validation.", True)
            with HeldFiles() as held:
                streams(held, records, root, "validation.", True)
                (root / "validation.stdout.log").write_bytes(b"truncated")
                with self.assertRaises(ValueError):
                    held.check()


if __name__ == "__main__":
    unittest.main()

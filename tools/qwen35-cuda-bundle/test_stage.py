"""Pure temporary-file controls; no real model/provider/image files are opened."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stage


def pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class Staging(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.legacy = self.root / "legacy"
        self.output = self.root / "staged"
        self.tool = self.root / "tool"
        self.repo = self.root / "repo"
        for path in (self.legacy, self.tool, self.repo):
            path.mkdir()
        rows = [
            dict(section="images", legacy_kind="Canonical47CutlassBf16Sm80", kind="FixedPromptCutlassBf16Sm80", path="model/first.cubin"),
            dict(section="rotary", legacy_kind="RotaryArangeSm86", kind="RotaryArangeSm86", path="rotary/arange.cubin"),
            dict(section="activation", path="activation/silu.cubin"),
            dict(section="cublas", path="providers/cublas.so"),
            dict(section="cublas_lt", path="providers/lt.so"),
        ]
        self.words = {}
        for i, row in enumerate(rows):
            raw = (f"synthetic resource {i}\n" * 3).encode()
            row.update(pin(raw))
            path = self.legacy / row["path"]
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
            self.words[row["path"]] = raw
        manifest = dict(schema="zerogravity-short-hf-native-v1")
        for section in ("images", "rotary"):
            manifest[section] = [{"kind": r["legacy_kind"], "path": r["path"]} for r in rows if r["section"] == section]
        for section in ("activation", "cublas", "cublas_lt"):
            manifest[section] = next(r["path"] for r in rows if r["section"] == section)
        raw = stage.encoded(manifest)
        (self.legacy / stage.LEGACY).write_bytes(raw)
        profile = dict(legacy_manifest=pin(raw), resources=rows, native_catalogue="synthetic", cohort_2b_extra_image_included=False)
        other = copy.deepcopy(profile)
        other["legacy_manifest"] = pin(b"different synthetic manifest")
        self.pins = dict(schema="zerogravity.cuda-bundle-migration-pins.v1", contract_sources={"contract.rs": pin(b"synthetic contract")},
                         profiles={"0.8b": profile, "2b": other})
        (self.repo / "contract.rs").write_bytes(b"synthetic contract")
        (self.tool / "pins.json").write_bytes(stage.encoded(self.pins))
        self.addCleanup(patch.stopall)
        patch.object(stage, "HERE", self.tool).start()
        patch.object(stage, "REPO", self.repo).start()

    def run_stage(self):
        return stage.stage(self.legacy, self.output)

    def test_copies_exact_words_and_emits_only_current_labels_and_false_grants(self):
        original = (self.legacy / stage.LEGACY).read_bytes()
        receipt = self.run_stage()
        self.assertTrue(receipt["complete"])
        self.assertFalse(receipt["native_execution_qualified"])
        self.assertFalse(receipt["admission_enabled"])
        self.assertEqual(original, (self.legacy / stage.LEGACY).read_bytes())
        current = json.loads((self.output / stage.CURRENT).read_bytes())
        self.assertEqual(current["schema"], "zerogravity.cuda-kernels.v2")
        self.assertEqual(current["images"][0]["kind"], "FixedPromptCutlassBf16Sm80")
        for name, raw in self.words.items():
            self.assertEqual(raw, (self.output / name).read_bytes())
            self.assertNotEqual((self.output / name).stat().st_ino, (self.legacy / name).stat().st_ino)
        self.assertFalse((self.output / stage.LEGACY).exists())

    def test_changed_same_extent_resource_refuses_before_output(self):
        path = self.legacy / "providers/cublas.so"
        raw = path.read_bytes()
        path.write_bytes(b"x" + raw[1:])
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            self.run_stage()
        self.assertFalse(self.output.exists())

    def test_missing_extra_and_empty_unknown_directory_refuse(self):
        for name, directory in (("extra", False), ("empty", True)):
            with self.subTest(name=name):
                path = self.legacy / name
                path.mkdir() if directory else path.write_bytes(b"extra")
                with self.assertRaisesRegex(ValueError, "extra legacy resource"):
                    self.run_stage()
                path.rmdir() if directory else path.unlink()
        (self.legacy / "model/first.cubin").unlink()
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            self.run_stage()

    def test_symlink_resource_and_directory_and_root_refuse(self):
        path = self.legacy / "model/first.cubin"
        original = self.root / "original"
        path.rename(original)
        path.symlink_to(original)
        with self.assertRaises(ValueError):
            self.run_stage()
        path.unlink()
        original.rename(path)
        model = self.legacy / "model"
        model.rename(self.root / "model")
        model.symlink_to(self.root / "model", target_is_directory=True)
        with self.assertRaises(ValueError):
            self.run_stage()
        model.unlink()
        (self.root / "model").rename(model)
        alias = self.root / "alias"
        alias.symlink_to(self.legacy, target_is_directory=True)
        with self.assertRaises(OSError):
            stage.stage(alias, self.output)

    def test_changed_legacy_or_current_contract_refuses(self):
        path = self.legacy / stage.LEGACY
        raw = path.read_bytes()
        path.write_bytes(raw + b"\n")
        with self.assertRaisesRegex(ValueError, "authenticated legacy"):
            self.run_stage()
        path.write_bytes(raw)
        (self.repo / "contract.rs").write_bytes(b"changed contract")
        with self.assertRaises(ValueError):
            self.run_stage()
        self.assertFalse(self.output.exists())

    def test_unknown_fields_duplicate_keys_missing_rows_and_traversal_refuse(self):
        raw = (self.legacy / stage.LEGACY).read_bytes()
        value = json.loads(raw)
        for kind in ("unknown", "missing", "traversal", "duplicate"):
            with self.subTest(kind=kind):
                current, pins = copy.deepcopy(value), copy.deepcopy(self.pins)
                if kind == "unknown":
                    current["authority"] = True
                elif kind == "missing":
                    current["images"] = []
                elif kind == "traversal":
                    current["images"][0]["path"] = "../outside"
                    pins["profiles"]["0.8b"]["resources"][0]["path"] = "../outside"
                candidate = stage.encoded(current)
                if kind == "duplicate":
                    candidate = candidate.replace(b'"images":', b'"images": [], "images":', 1)
                pins["profiles"]["0.8b"]["legacy_manifest"] = pin(candidate)
                with self.assertRaises(ValueError):
                    stage.plan(candidate, pins)

    def test_post_copy_source_mutation_retains_failure_and_no_success(self):
        original_write = stage.write_new

        def mutate(directory, name, raw):
            result = original_write(directory, name, raw)
            if name == stage.CURRENT:
                (self.legacy / "model/first.cubin").write_bytes(b"changed")
            return result
        with patch.object(stage, "write_new", side_effect=mutate):
            with self.assertRaises(ValueError):
                self.run_stage()
        self.assertFalse((self.output / "stage-result.json").exists())
        self.assertFalse(json.loads((self.output / "stage-failure.json").read_bytes())["complete"])

    def test_existing_output_and_symlink_output_parent_refuse(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):
            self.run_stage()
        self.output.rmdir()
        alias = self.root / "parent-alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(OSError):
            stage.stage(self.legacy, alias / "output")


if __name__ == "__main__":
    unittest.main()

"""Pure temporary-file controls; no real model/provider/image files are opened."""

import json
from pathlib import Path
import tempfile
import unittest

import stage


class Staging(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.legacy = self.root / "legacy"
        self.output = self.root / "staged"
        self.legacy.mkdir()

        self.rows = [
            ("images", "Canonical47CutlassBf16Sm80", "model/first.cubin"),
            ("images", "ShortHfFaCausalCombineSm80", "model/fa.cubin"),
            ("rotary", "RotaryArangeSm86", "rotary/arange.cubin"),
        ]
        self.words = {}
        for index, (_, _, name) in enumerate(self.rows):
            raw = (f"synthetic resource {index}\n" * 3).encode()
            path = self.legacy / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
            self.words[name] = raw

        support = {
            "activation": "activation/silu.cubin",
            "cublas": "providers/cublas.so",
            "cublas_lt": "providers/lt.so",
        }
        for index, name in enumerate(support.values(), start=10):
            raw = (f"synthetic support {index}\n" * 3).encode()
            path = self.legacy / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
            self.words[name] = raw

        manifest = {
            "schema": "zerogravity-short-hf-native-v1",
            "images": [
                {"kind": kind, "path": name}
                for section, kind, name in self.rows
                if section == "images"
            ],
            "rotary": [
                {"kind": kind, "path": name}
                for section, kind, name in self.rows
                if section == "rotary"
            ],
            **support,
        }
        (self.legacy / stage.LEGACY).write_bytes(stage.encoded(manifest))

    def test_check_legacy_is_read_only_and_translates_only_closed_labels(self):
        before = {
            path.relative_to(self.legacy): path.read_bytes()
            for path in self.legacy.rglob("*")
            if path.is_file()
        }
        checked = stage.check_legacy(self.legacy)
        self.assertTrue(checked["complete"])
        self.assertEqual(checked["schema"], "legacy")
        self.assertEqual(checked["resources"], len(self.words))
        kinds = [row["kind"] for row in checked["current_manifest"]["images"]]
        self.assertEqual(kinds, ["FixedPromptCutlassBf16Sm80", "TextFaCausalCombineSm80"])
        after = {
            path.relative_to(self.legacy): path.read_bytes()
            for path in self.legacy.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)

    def test_stage_copies_exact_bytes_without_pins_or_receipts(self):
        result = stage.stage(self.legacy, self.output)
        self.assertTrue(result["complete"])
        self.assertEqual(result["resources"], len(self.words))
        self.assertFalse((self.output / stage.LEGACY).exists())
        self.assertFalse((self.output / "stage-result.json").exists())
        current = json.loads((self.output / stage.CURRENT).read_bytes())
        self.assertEqual(current["schema"], "zerogravity.cuda-kernels.v2")
        for name, raw in self.words.items():
            self.assertEqual((self.output / name).read_bytes(), raw)
            self.assertNotEqual((self.output / name).stat().st_ino, (self.legacy / name).stat().st_ino)
        checked = stage.check_current(self.output)
        self.assertTrue(checked["complete"])

    def test_current_check_allows_old_audit_receipt_but_does_not_require_it(self):
        stage.stage(self.legacy, self.output)
        (self.output / "stage-result.json").write_text('{"historical":true}\n')
        self.assertTrue(stage.check_current(self.output)["complete"])
        (self.output / "stage-result.json").unlink()
        self.assertTrue(stage.check_current(self.output)["complete"])

    def test_extra_missing_symlink_and_bad_manifest_refuse(self):
        extra = self.legacy / "extra"
        extra.write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "extra"):
            stage.check_legacy(self.legacy)
        extra.unlink()

        target = self.legacy / "model/first.cubin"
        raw = target.read_bytes()
        target.unlink()
        target.symlink_to(self.legacy / "model/fa.cubin")
        with self.assertRaises(OSError):
            stage.check_legacy(self.legacy)
        target.unlink()
        target.write_bytes(raw)

        manifest = json.loads((self.legacy / stage.LEGACY).read_bytes())
        manifest["unexpected"] = True
        (self.legacy / stage.LEGACY).write_bytes(stage.encoded(manifest))
        with self.assertRaises(ValueError):
            stage.check_legacy(self.legacy)

    def test_failed_stage_removes_partial_output(self):
        original = stage.write_new

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        stage.write_new = fail
        try:
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                stage.stage(self.legacy, self.output)
        finally:
            stage.write_new = original
        self.assertFalse(self.output.exists())

    def test_existing_output_and_symlink_ancestors_refuse(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):
            stage.stage(self.legacy, self.output)
        self.output.rmdir()

        alias = self.root / "legacy-alias"
        alias.symlink_to(self.legacy, target_is_directory=True)
        with self.assertRaises(OSError):
            stage.check_legacy(alias)

    def test_duplicate_paths_kinds_and_traversal_refuse(self):
        manifest = json.loads((self.legacy / stage.LEGACY).read_bytes())
        for mutation in ("path", "kind", "traversal"):
            value = json.loads(json.dumps(manifest))
            if mutation == "path":
                value["images"][1]["path"] = value["images"][0]["path"]
            elif mutation == "kind":
                value["images"][1]["kind"] = value["images"][0]["kind"]
            else:
                value["images"][0]["path"] = "../outside"
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    stage.manifest(stage.encoded(value), legacy=True)


if __name__ == "__main__":
    unittest.main()

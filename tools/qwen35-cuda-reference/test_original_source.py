"""Optional pure ancestry/scope controls against the retained authenticated package."""
import ast
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from contract import GENERATION, MODEL, REVISION, require
from run_reference import prepare_package, verify_collector, verify_original


@unittest.skipUnless(os.environ.get("ZG_REFERENCE_SOURCE"), "set explicit retained original source path")
class OriginalSourceTests(unittest.TestCase):
    source_env = "ZG_REFERENCE_SOURCE"
    stage = "discover"

    def test_staging_preserves_ancestry_and_scope_checks(self):
        root = Path(os.environ[self.source_env])
        pins = verify_original(root, self.stage)
        for count in (47, 48, 49, 50):
            with self.subTest(prompt_tokens=count), tempfile.TemporaryDirectory() as output:
                value = {"schema": "zerogravity.qwen35-cuda-reference.request.v1", "model": MODEL,
                    "source_revision": REVISION, "phase": "prefill", "prompt_tokens": count,
                    "cached_tokens": 0, "request_batch_size": 1, "prompt": "text", "generation": GENERATION}
                package = prepare_package(root, Path(output), value, self.stage)
                for relative in pins:
                    if relative.startswith("package/") and relative not in (
                            "package/matrix_contract.py", "package/validate_native.py"):
                        self.assertEqual((root / relative).read_bytes(),
                                         (package / Path(relative).name).read_bytes()
                                         if len(Path(relative).parts) == 2 else
                                         package.joinpath(*Path(relative).parts[1:]).read_bytes())
                before = (root / "package/validate_native.py").read_text()
                after = (package / "validate_native.py").read_text()
                self.assertEqual(after.replace("prompt_rows == PROMPT_ROWS", "prompt_rows in (47,57,62,67,72)"), before)
                original_contract = (root / "package/matrix_contract.py").read_text()
                successor_contract = (package / "matrix_contract.py").read_text()
                self.assertEqual(original_contract[original_contract.index("def require("):
                    original_contract.index("\n\nrequire(len(PROMPT_IDS)")].rstrip("\n"),
                    successor_contract[successor_contract.index("def require("):].rstrip("\n"))
                function = next(node for node in ast.parse(after).body
                                if isinstance(node, ast.FunctionDef) and node.name == "validate_scopes")
                namespace = {"require": require, "PROMPT_ROWS": count}
                exec(compile(ast.Module(body=[function], type_ignores=[]), "<original-scope-control>", "exec"), namespace)
                scopes = []
                events = []
                for call in range(8):
                    rows = count if call == 0 else 1
                    scopes.append({"sequence": call, "begun": True, "closed": True, "call": call,
                        "role": 0, "tid": 1, "A": 1, "A_bytes": rows * 4,
                        "B": 2, "B_bytes": rows * 3 * 8, "C": 3, "C_bytes": 248320 * 2,
                        "workspace": 0, "workspace_bytes": 0, "first": call, "last": call + 1})
                    events.append({"sequence": call, "scope": call, "tid": 1})
                check = namespace["validate_scopes"]
                check(scopes, events, 8, count)
                with self.assertRaises(ValueError):
                    check(scopes, events, 8, count + 1)
                for field in ("A_bytes", "B_bytes", "C_bytes", "first", "tid"):
                    changed = [dict(scope) for scope in scopes]
                    changed[0][field] += 1
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        check(changed, events, 8, count)
        verify_original(root, self.stage)

    def test_wrong_stage_and_changed_inherited_source_refuse(self):
        root = Path(os.environ[self.source_env])
        pins = verify_original(root, self.stage)
        with self.assertRaises(ValueError):
            verify_original(root, "capture" if self.stage == "discover" else "discover")
        with tempfile.TemporaryDirectory() as output:
            copied = Path(output)
            for relative in pins:
                destination = copied / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / relative, destination)
            guard = copied / "package/primary-runtime-pins.json"
            guard.write_bytes(guard.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                verify_original(copied, self.stage)

    def test_stage_collector_source_and_headers_cannot_be_interchanged(self):
        root = Path(os.environ[self.source_env])
        matched = root.parent / ("original-build-5" if self.stage == "discover" else "original-build-6") / "build"
        wrong = root.parent / ("original-build-6" if self.stage == "discover" else "original-build-5") / "build"
        if not matched.is_dir() or not wrong.is_dir():
            self.skipTest("retained paired collector artifacts unavailable")
        verify_collector(root, matched)
        with self.assertRaises(ValueError):
            verify_collector(root, wrong)


@unittest.skipUnless(os.environ.get("ZG_CAPTURE_SOURCE"), "set explicit retained full-ABI capture source path")
class CaptureSourceTests(OriginalSourceTests):
    source_env = "ZG_CAPTURE_SOURCE"
    stage = "capture"


if __name__ == "__main__":
    unittest.main()

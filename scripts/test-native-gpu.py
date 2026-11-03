#!/usr/bin/env python3
"""Offline guard regressions; no SDK, real cache, command or GPU is required.

Actual user/network namespace isolation is a separate physical phase fixture.
These tests exercise real temporary-file verification before mocked execution.
"""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import types
import unittest
from unittest import mock
import tempfile


SCRIPT = Path(__file__).with_name("native-gpu.py")
# Load the script without importing it as __main__ or creating a bytecode cache.
NATIVE = types.ModuleType("native_gpu")
exec(compile(SCRIPT.read_bytes(), str(SCRIPT), "exec"), NATIVE.__dict__)


class TreeHashTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="native-gpu-tree-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        # Create in the reverse of hash order; include raw binary and empty data.
        (self.root / "z").write_bytes(b"")
        (self.root / "b").write_bytes(b"abc")
        (self.root / "a").write_bytes(b"\x00\xff")
        (self.root / NATIVE.SENTINEL).write_bytes(b"installation metadata")

    def test_independent_canonical_bytes_and_sentinel_exclusion(self):
        expected = hashlib.sha256(b"a\n2\n\x00\xffb\n3\nabcz\n0\n").hexdigest()
        self.assertEqual(NATIVE.tree_hash(self.root), (expected, 3))
        (self.root / NATIVE.SENTINEL).write_bytes(b"changed after installation")
        self.assertEqual(NATIVE.tree_hash(self.root), (expected, 3))

    @unittest.skipUnless(hasattr(os, "symlink"), "requires filesystem symlinks")
    def test_internal_and_dangling_links_are_excluded(self):
        before = NATIVE.tree_hash(self.root)
        (self.root / "regular-alias").symlink_to("a")
        (self.root / "dangling-alias").symlink_to("missing-tool")
        (self.root / "directory-alias").symlink_to(".", target_is_directory=True)
        self.assertEqual(NATIVE.tree_hash(self.root), before)

    @unittest.skipUnless(hasattr(os, "symlink"), "requires filesystem symlinks")
    def test_escaping_links_are_refused_even_when_dangling(self):
        escape = self.root / "escape"
        for target in ["../outside-missing-tool", str(self.root.parent)]:
            with self.subTest(target=target):
                escape.symlink_to(target)
                try:
                    with self.assertRaisesRegex(ValueError, "escapes verified tree"):
                        NATIVE.tree_hash(self.root)
                finally:
                    escape.unlink()


class ExecutionGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="native-gpu-exec-test-")
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name).resolve()
        self.root = self.home / ".local/share/tracel" / NATIVE.BUNDLE
        (self.root / "bin").mkdir(parents=True)
        (self.root / "bin/llvm-config").write_bytes(b"LLVM")
        self.sentinel = self.root / NATIVE.SENTINEL
        self.sentinel.touch()
        artifacts = self.home / ".cache/tracel"
        artifacts.mkdir(parents=True)
        self.archive = artifacts / (NATIVE.ARTIFACT + ".tar.xz")
        self.archive.write_bytes(b"existing offline archive fixture")
        self.archive_hash = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.content_hash = hashlib.sha256(b"bin/llvm-config\n4\nLLVM").hexdigest()
        self.sidecar = artifacts / (NATIVE.ARTIFACT + ".checksums.json")
        self.sidecar.write_text(
            json.dumps({
                "archive_sha256": self.archive_hash,
                "content_sha256": self.content_hash,
            }),
            encoding="utf-8",
        )
        patches = [
            mock.patch.object(NATIVE.Path, "home", return_value=self.home),
            mock.patch.object(NATIVE, "ARCHIVE_SHA256", self.archive_hash),
            mock.patch.object(NATIVE, "CONTENT_SHA256", self.content_hash),
            mock.patch.object(NATIVE.sys, "platform", "linux"),
            mock.patch.object(NATIVE.platform, "machine", return_value="x86_64"),
            # Deliberately unrelated: the upstream artifact path ignores it.
            mock.patch.dict(os.environ, {
                "XDG_DATA_HOME": str(self.home / ".local/share"),
                "XDG_CACHE_HOME": str(self.home / "unrelated-cache"),
            }),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def invoke(self, arguments):
        output, error = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(NATIVE.sys, "argv", [str(SCRIPT), *arguments]),
            redirect_stdout(output),
            redirect_stderr(error),
        ):
            try:
                result = NATIVE.main()
            finally:
                self.last_output = output.getvalue()
                self.last_error = error.getvalue()
        return result, output.getvalue(), error.getvalue()

    def assert_refused_before_execution(self, message):
        with (
            mock.patch.object(NATIVE.os, "execvpe") as execute,
            mock.patch.object(NATIVE.shutil, "which") as locate,
        ):
            result, output, error = self.invoke(["--require-llvm", "--", "must-not-run"])
        self.assertEqual(result, 1)
        self.assertEqual(output, "")
        self.assertIn(message, error)
        execute.assert_not_called()
        locate.assert_not_called()

    def test_missing_bundle_is_refused_before_execution(self):
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(self.home / "missing")}):
            self.assert_refused_before_execution("native-gpu:")

    def test_missing_sentinel_is_refused_before_execution(self):
        self.sentinel.unlink()
        self.assert_refused_before_execution(NATIVE.SENTINEL)

    def test_missing_sidecar_is_refused_before_execution(self):
        self.sidecar.unlink()
        self.assert_refused_before_execution("checksums.json")

    def test_sidecar_cannot_replace_pinned_hashes(self):
        self.sidecar.write_text(json.dumps({
            "archive_sha256": "0" * 64,
            "content_sha256": self.content_hash,
        }), encoding="utf-8")
        self.assert_refused_before_execution("sidecar differs from pinned identities")

    def test_archive_corruption_is_refused_before_execution(self):
        self.archive.write_bytes(b"changed archive")
        self.assert_refused_before_execution("archive SHA256 mismatch")

    def test_installed_content_corruption_is_refused_before_execution(self):
        (self.root / "bin/llvm-config").write_bytes(b"changed executable")
        self.assert_refused_before_execution("content SHA256 mismatch")

    def test_missing_unshare_has_no_unisolated_fallback(self):
        with (
            mock.patch.object(NATIVE.shutil, "which", return_value=None),
            mock.patch.object(NATIVE.os, "execvpe") as execute,
        ):
            result, output, error = self.invoke(["--", "must-not-run"])
        self.assertEqual(result, 1)
        self.assertEqual(output, "")
        self.assertIn("network isolation is mandatory", error)
        execute.assert_not_called()

    def test_verify_only_never_executes_or_requires_unshare(self):
        with (
            mock.patch.object(NATIVE.os, "execvpe") as execute,
            mock.patch.object(NATIVE.shutil, "which") as locate,
        ):
            result, output, error = self.invoke(["--verify-only"])
        self.assertEqual(result, 0)
        self.assertEqual(error, "")
        receipt = json.loads(output)
        self.assertEqual(receipt["archive"], str(self.archive))
        self.assertEqual(receipt["bundle"], str(self.root))
        self.assertEqual(receipt["archive_sha256"], self.archive_hash)
        self.assertEqual(receipt["content_sha256"], self.content_hash)
        execute.assert_not_called()
        locate.assert_not_called()

    def test_command_uses_mandatory_namespace_and_preserves_environment(self):
        before = os.environ.copy()
        current_directory = Path.cwd()
        with (
            mock.patch.object(NATIVE.shutil, "which", return_value="/test/unshare"),
            mock.patch.object(NATIVE.os, "execvpe", side_effect=SystemExit(73)) as execute,
            mock.patch.object(NATIVE, "verify") as verify,
        ):
            with self.assertRaises(SystemExit) as captured:
                self.invoke(["--", "cargo", "check", "--offline"])
        self.assertEqual(captured.exception.code, 73)
        executable, arguments, environment = execute.call_args.args
        self.assertEqual(executable, "/test/unshare")
        self.assertEqual(arguments, [
            "/test/unshare", "--user", "--map-root-user", "--net", "--",
            "cargo", "check", "--offline",
        ])
        self.assertEqual(environment, {**before, "CARGO_NET_OFFLINE": "true"})
        self.assertEqual(os.environ.copy(), before)
        self.assertEqual(Path.cwd(), current_directory)
        verify.assert_not_called()
        self.assertIn("LLVM verification not requested", self.last_error)
        self.assertNotIn("verified pinned LLVM", self.last_error)

    def test_default_command_does_not_require_an_unused_llvm_installation(self):
        missing = self.home / "absent-data-directory"
        with (
            mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(missing)}),
            mock.patch.object(NATIVE.shutil, "which", return_value="/test/unshare"),
            mock.patch.object(NATIVE.os, "execvpe", side_effect=SystemExit(73)) as execute,
            mock.patch.object(NATIVE, "verify") as verify,
        ):
            with self.assertRaises(SystemExit) as captured:
                self.invoke(["--", "cpp-only-command", "--require-llvm", ""])
        self.assertEqual(captured.exception.code, 73)
        self.assertEqual(execute.call_args.args[1][-3:], ["cpp-only-command", "--require-llvm", ""])
        self.assertEqual(execute.call_args.args[2]["XDG_DATA_HOME"], str(missing))
        self.assertEqual(execute.call_args.args[2]["CARGO_NET_OFFLINE"], "true")
        self.assertFalse(missing.exists())
        verify.assert_not_called()

    def test_required_llvm_is_verified_before_namespace_execution(self):
        order = []
        verify = NATIVE.verify

        def checked_verify():
            order.append("verify")
            return verify()

        def locate_unshare(name):
            self.assertEqual(name, "unshare")
            order.append("namespace")
            return "/test/unshare"

        def execute(executable, arguments, environment):
            order.append("execute")
            self.assertEqual(executable, "/test/unshare")
            self.assertEqual(arguments[-2:], ["llvm-command", "--verify-only"])
            self.assertEqual(environment["CARGO_NET_OFFLINE"], "true")
            raise SystemExit(73)

        with (
            mock.patch.object(NATIVE, "verify", side_effect=checked_verify),
            mock.patch.object(NATIVE.shutil, "which", side_effect=locate_unshare),
            mock.patch.object(NATIVE.os, "execvpe", side_effect=execute),
        ):
            with self.assertRaises(SystemExit) as captured:
                self.invoke(["--require-llvm", "--", "llvm-command", "--verify-only"])
        self.assertEqual(captured.exception.code, 73)
        self.assertEqual(order, ["verify", "namespace", "execute"])
        self.assertIn("verified pinned LLVM", self.last_error)
        self.assertNotIn("LLVM verification not requested", self.last_error)

    def test_verify_only_still_requires_the_existing_bundle(self):
        with (
            mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(self.home / "missing")}),
            mock.patch.object(NATIVE.os, "execvpe") as execute,
            mock.patch.object(NATIVE.shutil, "which") as locate,
        ):
            result, output, error = self.invoke(["--verify-only"])
        self.assertEqual(result, 1)
        self.assertEqual(output, "")
        self.assertIn("native-gpu:", error)
        execute.assert_not_called()
        locate.assert_not_called()

    def test_non_linux_command_is_refused_before_execution(self):
        for system in ["darwin", "win32"]:
            with (
                self.subTest(system=system),
                mock.patch.object(NATIVE.sys, "platform", system),
                mock.patch.object(NATIVE, "verify") as verify,
                mock.patch.object(NATIVE.shutil, "which") as locate,
                mock.patch.object(NATIVE.os, "execvpe") as execute,
            ):
                result, output, error = self.invoke(["--", "must-not-run"])
                self.assertEqual(result, 1)
                self.assertEqual(output, "")
                self.assertIn("Linux user/network namespaces", error)
                verify.assert_not_called()
                locate.assert_not_called()
                execute.assert_not_called()

    def test_exec_failure_has_no_retry_or_unisolated_fallback(self):
        with (
            mock.patch.object(NATIVE.shutil, "which", return_value="/test/unshare"),
            mock.patch.object(NATIVE.os, "execvpe", side_effect=OSError("cannot exec")) as execute,
        ):
            result, output, error = self.invoke(["--", "must-not-run"])
        self.assertEqual(result, 1)
        self.assertEqual(output, "")
        self.assertIn("cannot exec", error)
        execute.assert_called_once()

    def test_invalid_command_lines_are_refused_before_verification(self):
        for arguments in [
            [], ["--"], ["cargo", "check"], ["--verify-only", "--", "cargo"],
            ["--require-llvm"], ["--require-llvm", "--"],
            ["--verify-only", "--require-llvm"], ["--unknown", "--", "cargo"],
        ]:
            with self.subTest(arguments=arguments), mock.patch.object(NATIVE, "verify") as verify:
                with self.assertRaises(SystemExit) as captured:
                    self.invoke(arguments)
                self.assertEqual(captured.exception.code, 2)
                verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()

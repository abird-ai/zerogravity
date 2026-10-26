#!/usr/bin/env python3
"""Manifest-only architecture fixtures; no Cargo/build/model execution."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("check-architecture.py")
SPEC = importlib.util.spec_from_file_location("check_architecture", SCRIPT)
architecture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = architecture
SPEC.loader.exec_module(architecture)


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.workspace()

    def workspace(self, members='["crates/*"]', extra=""):
        (self.root / "Cargo.toml").write_text(
            f"[workspace]\nmembers = {members}\n{extra}", encoding="utf-8"
        )

    def package(self, directory, name, extra=""):
        path = self.root / "crates" / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / "src").mkdir(exist_ok=True)
        (path / "src/lib.rs").write_text("", encoding="utf-8")
        (path / "Cargo.toml").write_text(
            f'[package]\nname = "{name}"\nversion = "0.1.0"\n{extra}',
            encoding="utf-8",
        )

    def check(self):
        return architecture.check_workspace(self.root)

    def test_downward_dependencies_and_gpu_cpu_reuse(self):
        self.package("core", "zerogravity-core")
        self.package("artifacts", "zerogravity-artifacts",
                     '[dependencies]\nzerogravity-core = { path = "../core" }')
        self.package("cpu", "zerogravity-cpu",
                     '[dependencies]\nzerogravity-artifacts = { path = "../artifacts" }')
        self.package("gpu", "zerogravity-gpu",
                     '[dependencies]\nzerogravity-cpu = { path = "../cpu", optional = true }')
        self.package("family", "zerogravity-model-unlisted-family",
                     '[dependencies]\nzerogravity-gpu = { path = "../gpu" }')
        self.package("app", "zerogravity",
                     '[dependencies]\nmodel = { path = "../family", package = "zerogravity-model-unlisted-family" }')
        self.package("cli", "zerogravity-cli",
                     '[dependencies]\nzerogravity = { path = "../app" }')
        self.assertEqual(self.check(), [])

    def test_optional_alias_cannot_hide_future_family(self):
        self.package("family", "zerogravity-model-new-family")
        self.package("cpu", "zerogravity-cpu", '''
[dependencies]
innocent = { path = "../family", package = "zerogravity-model-new-family", optional = true }
''')
        errors = "\n".join(self.check())
        self.assertIn("zerogravity-cpu -> zerogravity-model-new-family", errors)
        self.assertIn("dependencies.innocent", errors)

    def test_target_normal_and_build_edges_are_checked_even_for_inactive_target(self):
        self.package("family", "zerogravity-model-future")
        for table in ("dependencies", "build-dependencies"):
            with self.subTest(table=table):
                self.package("gpu", "zerogravity-gpu", f'''
[target.'cfg(target_os = "never")'.{table}]
hidden = {{ path = "../family", package = "zerogravity-model-future", optional = true }}
''')
                errors = "\n".join(self.check())
                self.assertIn("zerogravity-gpu -> zerogravity-model-future", errors)
                self.assertIn(f"{table}.hidden", errors)

    def test_inherited_target_alias_uses_workspace_path_base(self):
        self.workspace(extra='''
[workspace.dependencies]
shared = { path = "crates/family", package = "zerogravity-model-next" }
''')
        self.package("family", "zerogravity-model-next")
        self.package("core", "zerogravity-core", '''
[target.'cfg(unix)'.build-dependencies]
shared = { workspace = true, optional = true }
''')
        self.assertIn("zerogravity-core -> zerogravity-model-next", "\n".join(self.check()))

    def test_inherited_registry_package_identity_is_checked(self):
        self.workspace(extra='''
[workspace.dependencies]
ordinary = { package = "zerogravity-model-next", version = "1" }
''')
        self.package("cpu", "zerogravity-cpu",
                     '[build-dependencies]\nordinary = { workspace = true }')
        self.assertIn("zerogravity-model-next", "\n".join(self.check()))

    def test_package_identity_cannot_disagree_with_local_path(self):
        self.package("family", "zerogravity-model-future")
        self.package("core", "zerogravity-core", '''
[dependencies]
hidden = { path = "../family", package = "ordinary-helper" }
''')
        with self.assertRaisesRegex(ValueError, "path contains zerogravity-model-future"):
            self.check()

    def test_dev_dependency_cycle_is_not_a_production_cycle(self):
        self.package("core", "zerogravity-core", '''
[dev-dependencies]
zerogravity-model-example = { path = "../family" }
[target.'cfg(unix)'.dev-dependencies]
also = { path = "../family", package = "zerogravity-model-example" }
''')
        self.package("family", "zerogravity-model-example",
                     '[dependencies]\nzerogravity-core = { path = "../core" }')
        self.assertEqual(self.check(), [])

    def test_production_cycle_is_reported(self):
        self.package("a", "zerogravity-model-a",
                     '[dependencies]\nzerogravity-model-b = { path = "../b", optional = true }')
        self.package("b", "zerogravity-model-b",
                     '[build-dependencies]\nzerogravity-model-a = { path = "../a" }')
        self.assertIn("production dependency cycle: zerogravity-model-a -> zerogravity-model-b -> zerogravity-model-a",
                      self.check())

    def test_in_workspace_path_dependency_is_an_automatic_member(self):
        self.workspace(members='["crates/core"]')
        self.package("family", "zerogravity-model-unlisted")
        self.package("core", "zerogravity-core", '''
[dependencies]
zerogravity-model-unlisted = { path = "../family" }
''')
        self.assertIn("zerogravity-model-unlisted", "\n".join(self.check()))

    def test_workspace_glob_excludes_are_not_read(self):
        self.workspace(extra='exclude = ["crates/excluded"]')
        self.package("core", "zerogravity-core")
        self.package("excluded", "not-a-member")
        (self.root / "crates/excluded/Cargo.toml").write_text("invalid TOML")
        self.assertEqual(self.check(), [])

    def test_registry_name_does_not_invent_local_cycle(self):
        self.package("a", "support-a", '[dependencies]\nsupport-b = "0.1"')
        self.package("b", "support-b",
                     '[dependencies]\nsupport-a = { path = "../a" }')
        self.assertEqual(self.check(), [])

    def test_transitive_helper_cannot_hide_upward_edge(self):
        self.package("family", "zerogravity-model-next")
        self.package("helper", "ordinary-helper", '''
[dependencies]
model = { path = "../family", package = "zerogravity-model-next" }
''')
        self.package("cpu", "zerogravity-cpu",
                     '[dependencies]\nordinary-helper = { path = "../helper" }')
        self.assertIn("zerogravity-cpu -> ordinary-helper -> zerogravity-model-next",
                      "\n".join(self.check()))

    def test_models_application_and_core_keep_their_boundaries(self):
        cases = [
            ("zerogravity-model-future", "zerogravity"),
            ("zerogravity", "zerogravity-cli"),
            ("zerogravity-core", "zerogravity-gpu"),
            ("zerogravity-artifacts", "zerogravity-cpu"),
        ]
        for source, target in cases:
            with self.subTest(source=source):
                self.package("target", target)
                self.package("source", source,
                             f'[dependencies]\nalias = {{ path = "../target", package = "{target}" }}')
                self.assertIn(f"{source} -> {target}", "\n".join(self.check()))

    def test_missing_or_overridden_workspace_identity_refuses(self):
        self.package("core", "zerogravity-core",
                     '[dependencies]\nmissing = { workspace = true }')
        with self.assertRaisesRegex(ValueError, "unresolved workspace dependency"):
            self.check()
        self.workspace(extra='[workspace.dependencies]\nshared = "1"')
        self.package("core", "zerogravity-core", '''
[dependencies]
shared = { workspace = true, package = "renamed" }
''')
        with self.assertRaisesRegex(ValueError, "overrides inherited identity"):
            self.check()


if __name__ == "__main__":
    unittest.main()

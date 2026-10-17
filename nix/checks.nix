{
  native,
  sourceRoot,
  module,
}: let
  inherit (native) pkgs;
  inherit (pkgs) lib;

  shellcheck =
    pkgs.runCommand "zerogravity-shellcheck" {
      nativeBuildInputs = [pkgs.shellcheck];
    } ''
      shellcheck ${sourceRoot}/install.sh ${sourceRoot}/scripts/build-release-artifacts.sh ${sourceRoot}/scripts/check.sh ${sourceRoot}/scripts/test-install.sh
      touch "$out"
    '';

  installer =
    pkgs.runCommand "zerogravity-install-fixture" {
      nativeBuildInputs = [pkgs.bash pkgs.coreutils pkgs.curl];
    } ''
      bash ${sourceRoot}/scripts/test-install.sh ${native.package}/bin/zerogravity
      touch "$out"
    '';

  # Check the actual filtered derivation source, not a filename heuristic.
  # Every reviewed vendor byte (including README, native inputs, licenses and
  # provenance JSON) must survive; no second Cargo.lock may enter the source.
  source-integrity =
    pkgs.runCommand "zerogravity-workspace-source-integrity" {
      nativeBuildInputs = [pkgs.python3];
    } ''
      python3 - "${sourceRoot}" "${native.src}" "$out" <<'PY'
      import hashlib, json, pathlib, sys, tomllib
      original, filtered, output = map(pathlib.Path, sys.argv[1:])
      vendor = pathlib.Path("crates/zerogravity-gpu/vendor")
      packages = ["cubecl-ir", "cubecl-cpp", "cubecl-cuda", "cubecl-hip"]
      manifest = tomllib.loads((filtered / "Cargo.toml").read_text())
      assert "package" not in manifest and manifest["workspace"]["package"]["version"]
      members = set(manifest["workspace"]["members"])
      assert members == {"crates/zerogravity-core", "crates/zerogravity-artifacts", "crates/zerogravity-cpu", "crates/zerogravity-model-qwen35", "crates/zerogravity-model-gemma4", "crates/zerogravity-gpu", "crates/zerogravity", "crates/zerogravity-cli", "tools/gpu-qualify"}, members
      for member in members:
          cargo_file = pathlib.Path(member) / "Cargo.toml"
          assert (filtered / cargo_file).read_bytes() == (original / cargo_file).read_bytes(), cargo_file
          member_manifest = tomllib.loads((filtered / cargo_file).read_text())
          assert member_manifest["package"]["version"] == {"workspace": True}, cargo_file
      excluded = set(manifest["workspace"].get("exclude", []))
      for package in packages:
          relative = vendor / package
          assert str(relative) in excluded, (relative, "vendor workspace member")
          assert (filtered / relative / "Cargo.toml").is_file(), relative
          cargo = tomllib.loads((original / relative / "Cargo.toml").read_text())
          for key in ["readme", "license-file"]:
              name = cargo["package"].get(key)
              if isinstance(name, str):
                  assert (filtered / relative / name).is_file(), (relative, key, name)
      files = []
      for path in sorted((original / vendor).rglob("*")):
          if not path.is_file():
              continue
          relative = path.relative_to(original)
          actual = filtered / relative
          assert actual.is_file() and actual.read_bytes() == path.read_bytes(), relative
          files.append({"path": str(relative), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
      assert files
      arm_notice = pathlib.Path("crates/zerogravity-gpu/LICENSE-ARM-MATH")
      assert (filtered / arm_notice).read_bytes() == (original / arm_notice).read_bytes()
      # A Rust module named target is source, not a Cargo output directory.
      target_module = vendor / "cubecl-cpp/src/target.rs"
      assert (filtered / target_module).read_bytes() == (original / target_module).read_bytes()
      assert sorted(str(p.relative_to(filtered)) for p in filtered.rglob("Cargo.lock")) == ["Cargo.lock"]
      assert (filtered / "README.md").read_bytes() == (original / "README.md").read_bytes()
      model_sources = []
      for member in ["crates/zerogravity-artifacts", "crates/zerogravity-model-qwen35", "crates/zerogravity-model-gemma4"]:
          for path in sorted((original / member).rglob("*.rs")):
              relative = path.relative_to(original)
              assert (filtered / relative).read_bytes() == path.read_bytes(), relative
              model_sources.append(str(relative))
      model_fixtures = []
      for name in ["official-e2b-it-config.json", "official-26b-it-config.json"]:
          relative = pathlib.Path("crates/zerogravity-model-gemma4/tests/fixtures") / name
          body = (original / relative).read_bytes()
          assert (filtered / relative).read_bytes() == body, relative
          model_fixtures.append({"path": str(relative), "sha256": hashlib.sha256(body).hexdigest()})
      # Synthetic localhost TLS material is test input, never production trust
      # or credentials. Keep the exact reviewed fixtures in optional builds.
      for name in ["ca.pem", "server.pem", "server.key", "README.md"]:
          relative = pathlib.Path("crates/zerogravity-artifacts/tests/fixtures/acquisition") / name
          assert (filtered / relative).read_bytes() == (original / relative).read_bytes(), relative
      # Independent synthetic Azure wire signature verification is also an
      # exact reviewed test input, not a general Python-source allowance.
      relative = pathlib.Path("crates/zerogravity-artifacts/tests/fixtures/azure_wire.py")
      assert (filtered / relative).read_bytes() == (original / relative).read_bytes(), relative
      output.mkdir()
      (output / "manifest.json").write_text(json.dumps({"vendor_files": files, "workspace_members": sorted(members), "workspace_version_inherited": True, "model_sources": model_sources, "model_fixtures": model_fixtures, "single_root_lock": True, "root_readme_unchanged": True, "arm_math_notice_sha256": hashlib.sha256((filtered / arm_notice).read_bytes()).hexdigest()}, indent=2) + "\n")
      PY
    '';

  actionlint =
    pkgs.runCommand "zerogravity-actionlint" {
      nativeBuildInputs = [pkgs.actionlint];
    } ''
      actionlint -no-color ${sourceRoot}/.github/workflows/build-binaries.yml
      touch "$out"
    '';

  nixfmt =
    pkgs.runCommand "zerogravity-nixfmt" {
      nativeBuildInputs = [pkgs.alejandra];
    } ''
      alejandra --check ${sourceRoot}/flake.nix ${sourceRoot}/nix/*.nix ${sourceRoot}/nix/tests/*.nix
      touch "$out"
    '';
in
  {
    inherit (native) package tests clippy fmt;
    inherit shellcheck actionlint nixfmt installer source-integrity;
  }
  // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
    nixos-module = import ./tests/module.nix {inherit pkgs module;};
  }

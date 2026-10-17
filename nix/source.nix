# Keep Cargo inputs and complete reviewed vendor trees. Crane's generic filter
# keeps Rust/TOML only; README/include/license/provenance files are also inputs.
{
  lib,
  craneLib,
  sourceRoot,
}: let
  cargo = builtins.fromTOML (builtins.readFile (sourceRoot + "/Cargo.toml"));
  packageRoots = [""] ++ (cargo.workspace.members or []) ++ (cargo.workspace.exclude or []);
  excluded = builtins.concatLists (map (root:
    map (name:
      if root == ""
      then name
      else "${root}/${name}") ["target" ".git" ".direnv"])
  packageRoots);
in
  lib.cleanSourceWith {
    src = sourceRoot;
    name = "zerogravity-workspace-source";
    filter = path: type: let
      relative = lib.removePrefix "${toString sourceRoot}/" (toString path);
      parts = lib.splitString "/" relative;
      top = builtins.head parts;
      base = baseNameOf path;
      sourceTree = builtins.elem top ["crates" "tools" "src" "examples" "tests" ".cargo"];
      vendorTree = lib.hasPrefix "crates/zerogravity-gpu/vendor/" relative;
      buildOutput = builtins.any (root: relative == root || lib.hasPrefix "${root}/" relative) excluded;
      packageNotice = builtins.elem base ["README.md" "LICENSE" "LICENSE-MIT" "LICENSE-APACHE" "LICENSE-ARM-MATH" "NOTICE"];
      modelFixture = builtins.elem relative [
        "crates/zerogravity-model-gemma4/tests/fixtures/official-e2b-it-config.json"
        "crates/zerogravity-model-gemma4/tests/fixtures/official-26b-it-config.json"
      ];
      acquisitionFixture = builtins.elem relative [
        "crates/zerogravity-artifacts/tests/fixtures/acquisition/ca.pem"
        "crates/zerogravity-artifacts/tests/fixtures/acquisition/server.pem"
        "crates/zerogravity-artifacts/tests/fixtures/acquisition/server.key"
      ];
      rootInput = builtins.elem relative ["Cargo.toml" "Cargo.lock" "rust-toolchain.toml" "README.md" "LICENSE" "LICENSE-MIT" "LICENSE-APACHE"];
    in
      !buildOutput
      && (rootInput || (sourceTree && (craneLib.filterCargoSources path type || vendorTree || packageNotice || modelFixture || acquisitionFixture)));
  }

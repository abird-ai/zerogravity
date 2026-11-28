# Keep complete declared package source/test trees and reviewed vendor inputs.
# Native kernels, embedded data and fixtures need not have Rust extensions.
{
  lib,
  craneLib,
  sourceRoot,
}: let
  cargo = builtins.fromTOML (builtins.readFile (sourceRoot + "/Cargo.toml"));
  members = cargo.workspace.members or [];
  packageRoots = [""] ++ members ++ (cargo.workspace.exclude or []);
  sourceRoots = lib.concatMap (root: map (dir: "${root}/${dir}") ["src" "examples" "tests" "benches"]) members;
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
      webTree = lib.hasPrefix "crates/zerogravity-cli/web/dist/" relative || relative == "crates/zerogravity-cli/web/dist";
      webNodeModules = lib.hasPrefix "crates/zerogravity-cli/web/node_modules/" relative || relative == "crates/zerogravity-cli/web/node_modules";
      vendorTree = lib.hasPrefix "crates/zerogravity-gpu/vendor/" relative || lib.hasPrefix "crates/zerogravity-serving/vendor/" relative;
      buildOutput = builtins.any (root: relative == root || lib.hasPrefix "${root}/" relative) excluded;
      packageNotice = builtins.elem base ["README.md" "LICENSE" "LICENSE-MIT" "LICENSE-APACHE" "LICENSE-ARM-MATH" "NOTICE"];
      packageSource = builtins.any (root: relative == root || lib.hasPrefix "${root}/" relative) sourceRoots;
      rootInput = builtins.elem relative ["Cargo.toml" "Cargo.lock" "rust-toolchain.toml" "README.md" "LICENSE" "LICENSE-MIT" "LICENSE-APACHE"];
    in
      !buildOutput
      && !webNodeModules
      && (rootInput || (sourceTree && (craneLib.filterCargoSources path type || vendorTree || webTree || packageNotice || packageSource)));
  }

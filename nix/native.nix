{
  pkgs,
  sourceRoot,
  version,
  mkCraneLib,
}: let
  craneLib = mkCraneLib pkgs null;
  src = import ./source.nix {
    inherit (pkgs) lib;
    inherit craneLib sourceRoot;
  };
  workspace = import ./workspace.nix {inherit (pkgs) lib;};
  commonArgs = {
    inherit src version;
    pname = "zerogravity";
    strictDeps = true;
    cargoExtraArgs = workspace.cliArgs;
    CARGO_NET_OFFLINE = "true";
  };

  # Crane caches dependency artifacts across source-only changes.
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  package = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
      doCheck = false;
      meta = {
        description = "Zerogravity CPU/Qwen3.5 text inference and source inspection";
        mainProgram = "zerogravity";
        platforms = pkgs.lib.platforms.unix;
      };
    });

  tests = craneLib.cargoTest (commonArgs
    // {
      inherit cargoArtifacts;
      cargoExtraArgs = workspace.qualityArgs;
    });
  clippy = craneLib.cargoClippy (commonArgs
    // {
      inherit cargoArtifacts;
      cargoExtraArgs = workspace.qualityArgs;
      cargoClippyExtraArgs = "--all-targets -- -D warnings";
    });
  fmt = craneLib.cargoFmt {
    inherit src version;
    pname = "zerogravity";
    cargoExtraArgs = "--all";
  };
in {
  inherit pkgs craneLib src commonArgs cargoArtifacts package tests clippy fmt;
}

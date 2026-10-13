{
  nixpkgs,
  overlay,
  sourceRoot,
  version,
  mkCraneLib,
}: let
  mkTarget = {
    pkgs,
    target,
    rustFlags ? null,
    nativeBuildInputs ? [],
    extraArgs ? {},
  }: let
    craneLib = mkCraneLib pkgs target;
    src = import ./source.nix {
      lib = nixpkgs.lib;
      inherit craneLib sourceRoot;
    };
    workspace = import ./workspace.nix {lib = nixpkgs.lib;};
    commonArgs =
      {
        inherit src version;
        pname = "zerogravity";
        strictDeps = true;
        doCheck = false;
        cargoExtraArgs = workspace.cliArgs;
        CARGO_NET_OFFLINE = "true";
        CARGO_BUILD_TARGET = target;
      }
      // nixpkgs.lib.optionalAttrs (rustFlags != null) {CARGO_BUILD_RUSTFLAGS = rustFlags;}
      // extraArgs;
    cargoArtifacts = craneLib.buildDepsOnly commonArgs;
    package = craneLib.buildPackage (commonArgs
      // {
        inherit cargoArtifacts nativeBuildInputs;
        doCheck = false;
        meta = {
          description = "Cross-compiled Zerogravity CPU text frontend";
          mainProgram = "zerogravity";
        };
      });
  in {
    inherit pkgs craneLib src commonArgs cargoArtifacts package;
  };

  mkCross = {
    localSystem,
    crossSystem,
    target,
    rustFlags ? null,
  }: let
    pkgs = import nixpkgs {
      inherit localSystem crossSystem;
      overlays = [overlay];
    };
  in
    mkTarget {inherit pkgs target rustFlags;};
in {
  inherit mkTarget mkCross;
}

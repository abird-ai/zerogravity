{
  pkgs,
  sourceRoot,
  version,
  mkCraneLib,
  runtime ? null,
}: let
  craneLib = mkCraneLib pkgs null;
  src = import ./source.nix {
    inherit (pkgs) lib;
    inherit craneLib sourceRoot;
  };
  workspace = import ./workspace.nix {inherit (pkgs) lib;};
  commonArgs =
    {
      inherit src version;
      pname = "zerogravity";
      strictDeps = true;
      cargoExtraArgs =
        if runtime == null
        then workspace.cliArgs
        else "--locked --offline -p zerogravity-cli --bin zerogravity";
      CARGO_NET_OFFLINE = "true";
    }
    // pkgs.lib.optionalAttrs (runtime != null) {
      CUDARC_CUDA_VERSION = runtime.cudaBuildVersion;
    };

  # Crane caches dependency artifacts across source-only changes.
  cargoArtifacts = craneLib.buildDepsOnly commonArgs;
  package = craneLib.buildPackage (commonArgs
    // {
      inherit cargoArtifacts;
      doCheck = false;
      meta = {
        description =
          if runtime == null
          then "ZeroGravity portable CPU text frontend"
          else "ZeroGravity native CPU, CUDA, ROCm and Vulkan runtime";
        mainProgram = "zerogravity";
        platforms = pkgs.lib.platforms.unix;
      };
    }
    // pkgs.lib.optionalAttrs (runtime != null) {
      nativeBuildInputs = [pkgs.makeWrapper];
      passthru.gpuRuntime = runtime;
      postInstall = ''
        wrapProgram "$out/bin/zerogravity" ${runtime.wrapperArgs}
        # Native package: verify that the wrapped executable itself starts
        # without a dev shell, attached device or caller SDK environment.
        env -u CUDA_PATH -u HIP_PATH -u ROCM_PATH -u ROCM_HOME \
          -u HIP_DEVICE_LIB_PATH -u LD_LIBRARY_PATH "$out/bin/zerogravity" --version
      '';
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

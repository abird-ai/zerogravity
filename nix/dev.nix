{
  native,
  sourceRoot,
  gpuPkgs ? null,
}: let
  inherit (native) pkgs;
  commonPackages = [
    pkgs.alejandra
    pkgs.shellcheck
    pkgs.actionlint
    pkgs.python3
    pkgs.curl
  ];
  gpuPackages = [
    gpuPkgs.cudaPackages.cuda_cudart
    gpuPkgs.cudaPackages.cuda_nvrtc
    gpuPkgs.cudaPackages.cccl
    gpuPkgs.cudaPackages.cuda_nvcc
    gpuPkgs.cudaPackages.libcublas
    gpuPkgs.rocmPackages.clr
    gpuPkgs.rocmPackages."rocm-runtime"
    gpuPkgs.rocmPackages.hipblas
    gpuPkgs.rocmPackages.hipblaslt
    gpuPkgs.rocmPackages.rocblas
  ];
  gpuLibraryPath = gpuPkgs.lib.makeLibraryPath (gpuPackages ++ [gpuPkgs.stdenv.cc.cc]);
in {
  shell = native.craneLib.devShell {
    checks = {inherit (native) tests clippy;};
    packages = commonPackages;
    RUST_BACKTRACE = "1";
  };

  gpu =
    if gpuPkgs == null
    then null
    else
      native.craneLib.devShell {
        checks = {};
        packages = commonPackages ++ gpuPackages;
        RUST_BACKTRACE = "1";
        shellHook = ''
          export HIP_PATH="${gpuPkgs.rocmPackages.clr}"
          export ROCM_HOME="$HIP_PATH"
          export ROCM_PATH="$HIP_PATH"
          export CUDA_PATH="''${TMPDIR:-/tmp}/zerogravity-cuda-$UID"
          "${sourceRoot}/scripts/cuda-nix-include-view.sh" \
            "${gpuPkgs.cudaPackages.cuda_cudart}/include" \
            "${gpuPkgs.cudaPackages.cccl}/include" \
            "$CUDA_PATH" >/dev/null
          export LD_LIBRARY_PATH="${gpuLibraryPath}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };

  formatter = pkgs.alejandra;
}

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
  flashAttentionSrc =
    if gpuPkgs == null
    then null
    else
      gpuPkgs.fetchFromGitHub {
        owner = "vllm-project";
        repo = "flash-attention";
        rev = "b273a02621d3c94bc747df67204f591b67f38fae";
        hash = "sha256-8qGtJl18nIo8Kd/jPYlDK8K83uuu4W1mZ0x0oTvXU8w=";
      };
  flashAttentionCutlass =
    if gpuPkgs == null
    then null
    else
      gpuPkgs.fetchFromGitHub {
        owner = "NVIDIA";
        repo = "cutlass";
        rev = "62750a2b75c802660e4894434dc55e839f322277";
        hash = "sha256-eL64JYiFAqrn6OJDzijpMR4DbEB4El/8yT7++iCZBlE=";
      };
  flashInferSrc =
    if gpuPkgs == null
    then null
    else
      gpuPkgs.fetchzip {
        url = "https://files.pythonhosted.org/packages/05/76/1a42eafe2a3cd26924e485fb85bc635a2a41fcb14f453a8c037214b32940/flashinfer_python-0.7.2.post1.tar.gz";
        hash = "sha256-VOR0Wn/rbIyQDcR6sXE0BrvSK6dKP0oEY3vOEuv4v2c=";
      };
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
          export ZEROGRAVITY_FLASH_ATTN_SRC="${flashAttentionSrc}"
          export ZEROGRAVITY_FLASH_ATTN_CUTLASS="${flashAttentionCutlass}"
          export ZEROGRAVITY_FLASHINFER_SRC="${flashInferSrc}"
          export ZEROGRAVITY_CUDART_LIB="${gpuPkgs.cudaPackages.cuda_cudart}/lib"
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

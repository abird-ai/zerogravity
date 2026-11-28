{
  native,
  sourceRoot,
  gpuPkgs ? null,
  runtime ? null,
}: let
  inherit (native) pkgs;
  commonPackages = [
    pkgs.alejandra
    pkgs.shellcheck
    pkgs.actionlint
    pkgs.python3
    pkgs.curl
  ];
  gpuPackages =
    runtime.packages
    ++ [
      gpuPkgs.cudaPackages.cuda_nvcc
      gpuPkgs.rocmPackages.hipblas
      gpuPkgs.rocmPackages.hipblaslt
    ];
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
  commonShell = {
    checks = {inherit (native) tests clippy;};
    packages = commonPackages;
    RUST_BACKTRACE = "1";
  };
in {
  portable = native.craneLib.devShell commonShell;

  # Default Cargo enables all native backends. Use the package's same
  # runtime closure, without pulling optional benchmark source trees.
  shell = native.craneLib.devShell (commonShell
    // pkgs.lib.optionalAttrs (runtime != null) {
      packages = commonPackages ++ runtime.packages;
      inherit (runtime) shellHook;
    });

  gpu =
    if gpuPkgs == null
    then null
    else
      native.craneLib.devShell {
        checks = {};
        packages = commonPackages ++ gpuPackages;
        RUST_BACKTRACE = "1";
        shellHook = ''
          ${runtime.shellHook}
          export LD_LIBRARY_PATH="${gpuPkgs.lib.makeLibraryPath [gpuPkgs.rocmPackages.hipblas gpuPkgs.rocmPackages.hipblaslt]}:$LD_LIBRARY_PATH"
          export ZEROGRAVITY_FLASH_ATTN_SRC="${flashAttentionSrc}"
          export ZEROGRAVITY_FLASH_ATTN_CUTLASS="${flashAttentionCutlass}"
          export ZEROGRAVITY_FLASHINFER_SRC="${flashInferSrc}"
          export ZEROGRAVITY_CUDART_LIB="${gpuPkgs.cudaPackages.cuda_cudart}/lib"
        '';
      };

  formatter = pkgs.alejandra;
}

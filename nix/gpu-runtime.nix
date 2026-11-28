# One native Linux runtime closure for the ordinary package and development
# environment. Driver devices and vendor kernel modules remain host-owned.
{
  pkgs,
  sourceRoot,
}: let
  inherit (pkgs) lib;
  inherit (pkgs.cudaPackages) cuda_cudart cuda_nvrtc cccl libcublas;
  inherit (pkgs.rocmPackages) clr rocblas rocm-runtime rocm-device-libs;
  packages = [cuda_cudart cuda_nvrtc cccl libcublas clr rocblas rocm-runtime pkgs.vulkan-loader];
  # Match cudarc's build-time bindings to the runtime SDK rather than its
  # latest-version fallback when nvcc is absent from the build.
  cudaBuildVersion = "${lib.versions.major cuda_cudart.version}0${lib.versions.minor cuda_cudart.version}0";
  cudaPath =
    pkgs.runCommand "zerogravity-cuda-headers" {
      nativeBuildInputs = [pkgs.bash pkgs.coreutils];
    } ''
      bash ${sourceRoot + "/scripts/cuda-nix-include-view.sh"} \
        "${cuda_cudart}/include" "${cccl}/include" "$out"
    '';
  environment = {
    CUDA_PATH = toString cudaPath;
    HIP_PATH = toString clr;
    ROCM_HOME = toString clr;
    ROCM_PATH = toString clr;
    HIP_DEVICE_LIB_PATH = "${rocm-device-libs}/amdgcn/bitcode";
  };
  libraryPath = lib.makeLibraryPath (packages ++ [pkgs.stdenv.cc.cc.lib]) + ":/run/opengl-driver/lib";
  binPath = lib.makeBinPath [clr];
in {
  inherit packages environment libraryPath binPath cudaPath cudaBuildVersion;
  # No runtime header copying or SDK acquisition; every packaged reference
  # is a Nix closure edge. Explicit environment overrides remain possible.
  wrapperArgs =
    lib.concatMapStringsSep " " (
      name: "--set-default ${name} ${lib.escapeShellArg environment.${name}}"
    ) (builtins.attrNames environment)
    + " --prefix LD_LIBRARY_PATH : ${lib.escapeShellArg libraryPath}"
    + " --prefix PATH : ${lib.escapeShellArg binPath}";
  shellHook =
    lib.concatMapStringsSep "\n" (
      name: "export ${name}=${lib.escapeShellArg environment.${name}}"
    ) (builtins.attrNames environment)
    + "\n"
    + ''
      export CUDARC_CUDA_VERSION="${cudaBuildVersion}"
      export LD_LIBRARY_PATH="${libraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      export PATH="${binPath}:$PATH"
    '';
}

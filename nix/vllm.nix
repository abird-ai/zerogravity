{
  pkgs,
  sourceRoot,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
}: let
  mkProvider = {
    name,
    workspaceRoot,
    python,
    driverLibPath,
    tritonDriverVariable,
  }: let
    workspace = uv2nix.lib.workspace.loadWorkspace {
      inherit workspaceRoot;
    };
    overlay = workspace.mkPyprojectOverlay {
      sourcePreference = "wheel";
    };
    pythonBase = pkgs.callPackage pyproject-nix.build.packages {
      inherit python;
    };
    wheelFixups = final: prev: let
      withDeps = package: deps: ignored:
        prev.${package}.overrideAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ deps;
          preFixup =
            (old.preFixup or "")
            + pkgs.lib.optionalString (deps != []) ''
              for dependency in ${pkgs.lib.concatStringsSep " " (map toString deps)}; do
                while IFS= read -r library_dir; do
                  addAutoPatchelfSearchPath "$library_dir"
                done < <(find "$dependency" -type d -name lib -print)
              done
            '';
          autoPatchelfIgnoreMissingDeps =
            (old.autoPatchelfIgnoreMissingDeps or []) ++ ignored;
        });
      ffmpegCompatSonames = pkgs.lib.concatLists [
        (map (version: "libavcodec.so.${toString version}") (pkgs.lib.range 58 63))
        (map (version: "libavdevice.so.${toString version}") (pkgs.lib.range 58 63))
        (map (version: "libavfilter.so.${toString version}") (pkgs.lib.range 7 12))
        (map (version: "libavformat.so.${toString version}") (pkgs.lib.range 58 63))
        (map (version: "libavutil.so.${toString version}") (pkgs.lib.range 56 61))
        (map (version: "libswresample.so.${toString version}") (pkgs.lib.range 3 7))
        (map (version: "libswscale.so.${toString version}") (pkgs.lib.range 5 10))
      ];
      torchDeps = [
        final."nvidia-cublas"
        final."nvidia-cuda-cupti"
        final."nvidia-cuda-nvrtc"
        final."nvidia-cuda-runtime"
        final."nvidia-cudnn-cu13"
        final."nvidia-cufft"
        final."nvidia-cufile"
        final."nvidia-curand"
        final."nvidia-cusolver"
        final."nvidia-cusparse"
        final."nvidia-cusparselt-cu13"
        final."nvidia-nccl-cu13"
        final."nvidia-nvjitlink"
        final."nvidia-nvshmem-cu13"
        final."nvidia-nvtx"
      ];
    in {
      "humming-kernels" = withDeps "humming-kernels" [final.torch] ["libcuda.so.1"];
      "nccl-extensions" = withDeps "nccl-extensions" [final."nvidia-nccl-cu13"] ["libcuda.so.1"];
      nccl4py = withDeps "nccl4py" [final."nvidia-nccl-cu13"] ["libcuda.so.1"];
      numba = withDeps "numba" [pkgs.tbb] [];
      "nvidia-cublas" = withDeps "nvidia-cublas" [final."nvidia-cuda-nvrtc"] [];
      "cuda-bindings" = withDeps "cuda-bindings" [] ["libcuda.so.1"];
      "cuda-core" = withDeps "cuda-core" [] ["libcuda.so.1"];
      "nvidia-cuda-nvcc" = withDeps "nvidia-cuda-nvcc" [
        final."nvidia-cuda-crt"
        final."nvidia-cuda-runtime"
        final."nvidia-nvvm"
      ] [];
      "nvidia-cudnn-cu13" = withDeps "nvidia-cudnn-cu13" [final."nvidia-cublas"] [];
      "nvidia-cufft" = withDeps "nvidia-cufft" [final."nvidia-nvjitlink"] [];
      "nvidia-cufile" = withDeps "nvidia-cufile" [pkgs.rdma-core] [];
      "nvidia-cusolver" = withDeps "nvidia-cusolver" [
        final."nvidia-cublas"
        final."nvidia-cusparse"
        final."nvidia-nvjitlink"
      ] [];
      "nvidia-cusparse" = withDeps "nvidia-cusparse" [final."nvidia-nvjitlink"] [];
      "nvidia-cutlass-dsl" = withDeps "nvidia-cutlass-dsl" [] ["libcuda.so.1"];
      "nvidia-cutlass-dsl-libs-base" = withDeps "nvidia-cutlass-dsl-libs-base" [] ["libcuda.so.1"];
      "nvidia-cutlass-dsl-libs-core" = withDeps "nvidia-cutlass-dsl-libs-core" [] ["libcuda.so.1"];
      "nvidia-cutlass-dsl-libs-cu12" = withDeps "nvidia-cutlass-dsl-libs-cu12" [] ["libcuda.so.1"];
      "nvidia-cutlass-dsl-libs-cu13" = withDeps "nvidia-cutlass-dsl-libs-cu13" [] ["libcuda.so.1"];
      "nvidia-nvshmem-cu13" = withDeps "nvidia-nvshmem-cu13" [
        pkgs.libfabric
        pkgs.openmpi
        pkgs.pmix
        pkgs.rdma-core
        pkgs.ucx
      ] ["libcuda.so.1"];
      tilelang = withDeps "tilelang" [
        final."apache-tvm-ffi"
        final.torch
        final."z3-solver"
      ] ["libcuda.so.1"];
      "tokenspeed-triton" = withDeps "tokenspeed-triton" [] ["libcuda.so.1"];
      triton = withDeps "triton" [] ["libcuda.so.1"];
      torch = withDeps "torch" torchDeps ["libcuda.so.1"];
      "flashinfer-python" = withDeps "flashinfer-python" [final.torch] ["libcuda.so.1"];
      instanttensor = withDeps "instanttensor" [final.torch] ["libcuda.so.1"];
      "quack-kernels" = withDeps "quack-kernels" [final.torch] ["libcuda.so.1"];
      pynvvideocodec = withDeps "pynvvideocodec" [] ["libcuda.so.1"];
      "torch-c-dlpack-ext" = withDeps "torch-c-dlpack-ext" [final.torch] ["libcuda.so.1"];
      torchcodec =
        withDeps "torchcodec" [
          final.torch
          final."nvidia-cuda-nvrtc"
          final."nvidia-cuda-runtime"
          pkgs.libheif
        ]
        ffmpegCompatSonames;
      torchaudio = withDeps "torchaudio" [final.torch] ["libcuda.so.1"];
      torchvision = withDeps "torchvision" [final.torch] ["libcuda.so.1"];
      vllm = withDeps "vllm" [
        final.torch
        final."nvidia-cuda-nvrtc"
        final."nvidia-cuda-runtime"
      ] ["libcuda.so.1"];
      xgrammar = withDeps "xgrammar" [final."apache-tvm-ffi"] [];
    };
    pythonSet = pythonBase.overrideScope (
      pkgs.lib.composeManyExtensions [
        pyproject-build-systems.overlays.wheel
        overlay
        wheelFixups
      ]
    );
    environment =
      (pythonSet.mkVirtualEnv "zerogravity-vllm-${name}-env" {
        vllm = [];
      }).overrideAttrs (_: {
        venvIgnoreCollisions = [
          "lib/python${python.pythonVersion}/site-packages/build_backend.py"
        ];
      });
    provider = pkgs.writeShellApplication {
      name = "zerogravity-vllm-${name}";
      runtimeInputs = [
        pkgs.cacert
        pkgs.coreutils
        pkgs.curl
        pkgs.uv
        python
      ];
      text = ''
        set -euo pipefail

        cache_home="''${ZG_VLLM_CACHE_HOME:-''${XDG_CACHE_HOME:-$HOME/.cache}}"
        root="$cache_home/abird/zerogravity/vllm-${name}"
        kernel_env="$root/kernels"
        runtime_python="$kernel_env/bin/python"

        export VLLM_NO_USAGE_STATS=1
        export PYTHONNOUSERSITE=1
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export HF_HUB_DISABLE_TELEMETRY=1
        export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
        export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
        export CURL_CA_BUNDLE="$SSL_CERT_FILE"
        export PYTHONPATH="${environment}/${python.sitePackages}''${PYTHONPATH:+:$PYTHONPATH}"
        export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${driverLibPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export ${tritonDriverVariable}=${driverLibPath}

        command="''${1:-}"
        if [ -z "$command" ]; then
          echo "usage: zerogravity-vllm-${name} {prepare|check|serve} [args...]" >&2
          exit 2
        fi
        shift

        require_prepared() {
          if [ ! -x "$runtime_python" ]; then
            echo "vLLM host kernels are not prepared; run 'nix run .#vllm-${name} -- prepare' first" >&2
            exit 2
          fi
        }

        case "$command" in
          prepare)
            mkdir -p "$root"
            if [ ! -x "$runtime_python" ]; then
              ${python}/bin/python -m venv "$kernel_env"
            fi
            export UV_NO_CACHE=1
            "$runtime_python" -c 'import torch, vllm'
            "$runtime_python" -m vllm.entrypoints.cli.main download-kernels
            ;;
          check)
            require_prepared
            exec "$runtime_python" "$@"
            ;;
          serve)
            require_prepared
            exec "$runtime_python" -m vllm.entrypoints.cli.main serve "$@"
            ;;
          *)
            echo "unknown command: $command" >&2
            exit 2
            ;;
        esac
      '';
    };
  in {
    inherit environment provider;
    shell = pkgs.mkShell {
      packages = [
        environment
        pkgs.uv
        python
      ];
      VLLM_NO_USAGE_STATS = "1";
      UV_NO_SYNC = "1";
      UV_PYTHON_DOWNLOADS = "never";
    };
  };
in {
  cuda = mkProvider {
    name = "cuda";
    workspaceRoot = sourceRoot + "/benchmarks/vllm/cuda";
    python = pkgs.python312;
    driverLibPath = "/run/opengl-driver/lib";
    tritonDriverVariable = "TRITON_LIBCUDA_PATH";
  };
}

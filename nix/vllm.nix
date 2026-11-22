{
  pkgs,
  sourceRoot,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
}: let
  rocmHipSparseLt = pkgs.stdenvNoCC.mkDerivation {
    pname = "hipsparselt";
    version = "0.2.6.70203";
    src = pkgs.fetchurl {
      url = "https://repo.radeon.com/rocm/apt/7.2.3/pool/main/h/hipsparselt7.2.3/hipsparselt7.2.3_0.2.6.70203-90~24.04_amd64.deb";
      name = "hipsparselt-0.2.6.70203.deb";
      sha256 = "6a3e5aff56a1ad098668c7e66567709f1b367c0c8ba62fc18bbcedd03177849b";
    };
    dontUnpack = true;
    nativeBuildInputs = [
      pkgs.binutils
      pkgs.patchelf
      pkgs.xz
    ];
    installPhase = ''
      runHook preInstall
      mkdir -p root "$out"
      cd root
      ar p "$src" data.tar.xz | tar -xJ
      cp -a opt/rocm-7.2.3/lib "$out/"
      patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib:$ORIGIN/../llvm/lib:$ORIGIN/../lib/hipsparselt/lib:${pkgs.lib.makeLibraryPath [pkgs.rocmPackages.clr pkgs.rocmPackages.roctracer pkgs.stdenv.cc.cc.lib]}' \
        "$out/lib/libhipsparselt.so.0.2.70203"
      runHook postInstall
    '';
  };

  openMpi4Compat = pkgs.stdenvNoCC.mkDerivation {
    pname = "openmpi4-compat";
    version = "4.1.6";
    src = pkgs.fetchurl {
      url = "https://archive.ubuntu.com/ubuntu/pool/universe/o/openmpi/libopenmpi3t64_4.1.6-7ubuntu2_amd64.deb";
      name = "openmpi4-runtime-4.1.6.deb";
      sha256 = "4709aa89fd1064c6df3b8e7e460f530d8f2df24937a1d8df8fbf4948b391e99b";
    };
    dontUnpack = true;
    nativeBuildInputs = [
      pkgs.binutils
      pkgs.patchelf
      pkgs.zstd
    ];
    installPhase = ''
      runHook preInstall
      mkdir -p root "$out/lib"
      cd root
      ar p "$src" data.tar.zst | tar --zstd -x
      cp -a usr/lib/x86_64-linux-gnu/libmpi.so.40* "$out/lib/"
      cp -a usr/lib/x86_64-linux-gnu/libmpi_cxx.so.40* "$out/lib/"
      cp -a usr/lib/x86_64-linux-gnu/libopen-pal.so.40* "$out/lib/"
      cp -a usr/lib/x86_64-linux-gnu/libopen-rte.so.40* "$out/lib/"
      for library in "$out"/lib/*.so.*.*; do
        patchelf --set-rpath '$ORIGIN:${pkgs.lib.makeLibraryPath [pkgs.hwloc pkgs.libevent pkgs.stdenv.cc.cc.lib pkgs.zlib]}' "$library"
      done
      runHook postInstall
    '';
  };

  mkProvider = {
    name,
    backend,
    workspaceRoot,
    python,
    runtimePackages ? [],
    runtimeLibraryPaths ? [],
    runtimeEnvironment ? {},
    downloadKernels ? false,
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
      common = {
        numba = withDeps "numba" [pkgs.tbb] [];
        xgrammar = withDeps "xgrammar" [final."apache-tvm-ffi"] [];
      };
      cuda = {
        "humming-kernels" = withDeps "humming-kernels" [final.torch] ["libcuda.so.1"];
        "nccl-extensions" = withDeps "nccl-extensions" [final."nvidia-nccl-cu13"] ["libcuda.so.1"];
        nccl4py = withDeps "nccl4py" [final."nvidia-nccl-cu13"] ["libcuda.so.1"];
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
      };
      rocmDeps = [
        pkgs.libdrm
        pkgs.rocmPackages.clr
        pkgs.rocmPackages."rocm-runtime"
      ];
      rocmTorchDeps =
        rocmDeps
        ++ [
          rocmHipSparseLt
          openMpi4Compat
          pkgs.rocmPackages.hipblas
          pkgs.rocmPackages.hipblaslt
          pkgs.rocmPackages.hipfft
          pkgs.rocmPackages.hiprand
          pkgs.rocmPackages.hipsolver
          pkgs.rocmPackages.hipsparse
          pkgs.rocmPackages.miopen
          pkgs.rocmPackages.rccl
          pkgs.rocmPackages.rocblas
          pkgs.rocmPackages."rocprofiler-sdk"
          pkgs.rocmPackages.rocsolver
          pkgs.rocmPackages.roctracer
        ];
      rocm = {
        "amd-aiter" = withDeps "amd-aiter" [final.torch] [];
        "flash-attn" = withDeps "flash-attn" [final.torch] [];
        flydsl = withDeps "flydsl" [pkgs.rocmPackages.clr] [];
        "mooncake-transfer-engine-rocm" = withDeps "mooncake-transfer-engine-rocm" (rocmDeps ++ [pkgs.rdma-core pkgs.xz]) [];
        "torch-c-dlpack-ext" = withDeps "torch-c-dlpack-ext" [final.torch] [
          "libc10_cuda.so"
          "libtorch_cuda.so"
        ];
        torch = withDeps "torch" rocmTorchDeps [];
        torchaudio = withDeps "torchaudio" [final.torch] [];
        torchvision = withDeps "torchvision" [final.torch pkgs.libjpeg8] [];
        tilelang = withDeps "tilelang" [
          final."apache-tvm-ffi"
          final.torch
          final."z3-solver"
        ] [];
        triton = withDeps "triton" [pkgs.rocmPackages.clr] [];
        vllm = withDeps "vllm" [
          final."amd-aiter"
          final.torch
          pkgs.rocmPackages.clr
        ] [];
      };
      selected =
        if backend == "cuda"
        then cuda
        else if backend == "rocm"
        then rocm
        else throw "unsupported vLLM backend: ${backend}";
    in
      common // selected;
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
      runtimeInputs =
        [
          pkgs.cacert
          pkgs.coreutils
          pkgs.curl
          pkgs.uv
          python
        ]
        ++ runtimePackages;
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
        export LD_LIBRARY_PATH="${pkgs.lib.concatStringsSep ":" (["${pkgs.stdenv.cc.cc.lib}/lib"] ++ runtimeLibraryPaths)}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        ${pkgs.lib.concatStringsSep "\n" (pkgs.lib.mapAttrsToList (key: value: "export ${key}=${pkgs.lib.escapeShellArg (toString value)}") runtimeEnvironment)}

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
            ${pkgs.lib.optionalString downloadKernels ''
          "$runtime_python" -m vllm.entrypoints.cli.main download-kernels
        ''}
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
    backend = "cuda";
    workspaceRoot = sourceRoot + "/benchmarks/vllm/cuda";
    python = pkgs.python312;
    runtimeLibraryPaths = ["/run/opengl-driver/lib"];
    runtimeEnvironment.TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
    downloadKernels = true;
  };
  rocm = mkProvider {
    name = "rocm";
    backend = "rocm";
    workspaceRoot = sourceRoot + "/benchmarks/vllm/rocm";
    python = pkgs.python312;
    runtimePackages = [
      pkgs.rocmPackages.clr
      pkgs.stdenv.cc
    ];
    runtimeLibraryPaths = [
      "${pkgs.rocmPackages.clr}/lib"
      "${pkgs.rocmPackages."rocm-runtime"}/lib"
    ];
    runtimeEnvironment = {
      CC = "${pkgs.stdenv.cc}/bin/cc";
      HIP_PATH = pkgs.rocmPackages.clr;
      ROCM_HOME = pkgs.rocmPackages.clr;
      ROCM_PATH = pkgs.rocmPackages.clr;
    };
  };
}

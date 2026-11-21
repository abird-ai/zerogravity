{
  pkgs,
  sourceRoot,
}: let
  mkProvider = {
    name,
    project,
    python,
    torchBackend,
  }: let
    cacheName = "vllm-${name}";
    provider = pkgs.writeShellApplication {
      name = "zerogravity-vllm-${name}";
      runtimeInputs = [
        pkgs.cacert
        pkgs.coreutils
        pkgs.curl
        pkgs.uv
      ];
      text = ''
        set -euo pipefail

        project=${project}
        cache_home="''${ZG_VLLM_CACHE_HOME:-''${XDG_CACHE_HOME:-$HOME/.cache}}"
        root="$cache_home/abird/zerogravity/${cacheName}"
        unset VIRTUAL_ENV
        export UV_PROJECT_ENVIRONMENT="$root/venv"
        export UV_CACHE_DIR="$root/uv"
        export UV_TORCH_BACKEND=${torchBackend}
        export VLLM_NO_USAGE_STATS=1
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export HF_HUB_DISABLE_TELEMETRY=1
        export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
        export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
        export CURL_CA_BUNDLE="$SSL_CERT_FILE"
        export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib

        command="''${1:-}"
        if [ -z "$command" ]; then
          echo "usage: zerogravity-vllm-${name} {prepare|check|serve} [args...]" >&2
          exit 2
        fi
        shift

        case "$command" in
          prepare)
            mkdir -p "$root"
            unset UV_OFFLINE
            uv sync \
              --project "$project" \
              --frozen \
              --inexact \
              --no-dev \
              --python ${python}/bin/python3.12
            "$UV_PROJECT_ENVIRONMENT/bin/python" \
              -m vllm.entrypoints.cli.main download-kernels
            ;;
          check)
            export UV_OFFLINE=1
            exec "$UV_PROJECT_ENVIRONMENT/bin/python" "$@"
            ;;
          serve)
            export UV_OFFLINE=1
            exec "$UV_PROJECT_ENVIRONMENT/bin/python" \
              -m vllm.entrypoints.cli.main serve "$@"
            ;;
          *)
            echo "unknown command: $command" >&2
            exit 2
            ;;
        esac
      '';
    };
  in {
    inherit provider;
    shell = pkgs.mkShell {
      packages = [
        pkgs.uv
        python
      ];
      UV_TORCH_BACKEND = torchBackend;
      VLLM_NO_USAGE_STATS = "1";
    };
  };
in {
  cuda = mkProvider {
    name = "cuda";
    project = sourceRoot + "/benchmarks/vllm/cuda";
    python = pkgs.python312;
    torchBackend = "cu130";
  };
}

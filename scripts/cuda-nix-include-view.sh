#!/usr/bin/env bash
# Construct CubeCL CUDA_PATH from existing split Nix CUDA headers.
# No SDK acquisition, kernel invocation, driver configuration or copies.
set -euo pipefail

if (( $# != 3 )); then
  printf 'Usage: %s CUDART_INCLUDE_DIR CCCL_INCLUDE_DIR OUTPUT_DIR\n' "$0" >&2
  exit 2
fi

cudart=$(realpath -e -- "$1")
cccl=$(realpath -e -- "$2")
dest=$(realpath -m -- "$3")

if [[ ! -f "$cudart/cuda_runtime.h" || ! -f "$cudart/cuda_fp16.h" ]]; then
  printf 'Missing CUDA runtime headers in %s\n' "$cudart" >&2
  exit 2
fi
if [[ ! -d "$cccl/cuda/std" ]]; then
  printf 'Missing CUDA CCCL device headers in %s\n' "$cccl" >&2
  exit 2
fi
if [[ -e "$dest" && ! -d "$dest" ]]; then
  printf 'Output exists and is not a directory: %s\n' "$dest" >&2
  exit 2
fi
mkdir -p -- "$dest/include/cccl"

# Fail closed on existing conflicting paths; only this helper's links
# can be reused. Never rewrite user-managed files/directories.
link_once() {
  local src=$1 dst=$2
  if [[ -e "$dst" || -L "$dst" ]]; then
    if [[ ! -L "$dst" || "$(readlink -- "$dst")" != "$src" ]]; then
      printf 'Conflicting existing CUDA include path: %s\n' "$dst" >&2
      exit 2
    fi
  else
    ln -s -- "$src" "$dst"
  fi
}
for src in "$cudart"/*; do
  link_once "$src" "$dest/include/$(basename -- "$src")"
done
for src in "$cccl"/*; do
  link_once "$src" "$dest/include/cccl/$(basename -- "$src")"
done
test -f "$dest/include/cuda_runtime.h"
test -f "$dest/include/cuda_fp16.h"
test -d "$dest/include/cccl/cuda/std"
printf '%s\n' "$dest"

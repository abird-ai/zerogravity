#!/usr/bin/env bash
set -euo pipefail

# Run from any cwd; relative output is based on the invoking directory.
output_dir="${1:-dist}"
case "$output_dir" in
  /*) ;;
  *) output_dir="$PWD/$output_dir" ;;
esac

repo_dir="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

nix_args=(--extra-experimental-features "nix-command flakes")
echo "Building the six pinned CPU cross-release outputs via crane..."
release_path="$(nix "${nix_args[@]}" build .#release-all --no-link --print-out-paths)"

mkdir -p "$output_dir"
cp "$release_path"/zerogravity-* "$output_dir/"
cp "$release_path"/VERSION "$release_path"/PLATFORMS.txt "$output_dir/"

(
  cd "$output_dir"
  sha256sum -c ./*.sha256
)

printf '\nCPU text release outputs (GPU backends require separately qualified builds):\n'
ls -lh "$output_dir"

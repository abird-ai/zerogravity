#!/usr/bin/env bash
set -euo pipefail

# Purely local package-integrity regression test.
# Usage: scripts/test-install.sh [path-to-zerogravity-binary]
repo="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bin="${1:-$repo/target/release/zerogravity}"
[[ -f "$bin" ]] || { echo "installer fixture: no binary at $bin" >&2; exit 1; }
bin="$(realpath "$bin")"

case "$(uname -s):$(uname -m)" in
  Linux:x86_64|Linux:amd64) asset=zerogravity-linux-x86_64 ;;
  Linux:aarch64|Linux:arm64) asset=zerogravity-linux-aarch64 ;;
  Darwin:x86_64) asset=zerogravity-macos-x86_64 ;;
  Darwin:arm64) asset=zerogravity-macos-aarch64 ;;
  *) echo "installer fixture not supported for this machine; use an eligible native runner" >&2; exit 1 ;;
esac

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/releases" "$scratch/install"
cp "$bin" "$scratch/releases/$asset"

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

expected="$(hash_file "$scratch/releases/$asset")"
printf '%s  %s\n' "$expected" "$asset" > "$scratch/releases/$asset.sha256"

install_binary() {
  ZEROGRAVITY_RELEASE_BASE_URL="file://$scratch/releases" \
    ZEROGRAVITY_INSTALL_DIR="$scratch/install" \
    "$repo/install.sh"
}

install_binary
[[ -x "$scratch/install/zerogravity" ]]
[[ "$(hash_file "$scratch/install/zerogravity")" == "$expected" ]]
[[ "$("$scratch/install/zerogravity" --version)" == "$("$bin" --version)" ]]

# The second run must be idempotent.
install_binary
[[ "$(hash_file "$scratch/install/zerogravity")" == "$expected" ]]

# An attacker-corrupted checksum must be rejected without changing the installation.
printf '%064d  %s\n' 0 "$asset" > "$scratch/releases/$asset.sha256"
if install_binary >/dev/null 2>&1; then
  echo "installer fixture: corrupt checksum was wrongly accepted" >&2
  exit 1
fi
[[ "$(hash_file "$scratch/install/zerogravity")" == "$expected" ]]

# An incorrect requested version must be rejected before replacement.
printf '%s  %s\n' "$expected" "$asset" > "$scratch/releases/$asset.sha256"
if ZEROGRAVITY_VERSION=999.0.0 install_binary >/dev/null 2>&1; then
  echo "installer fixture: mismatched requested version was wrongly accepted" >&2
  exit 1
fi
[[ "$(hash_file "$scratch/install/zerogravity")" == "$expected" ]]

echo "POSIX installer fixture passed (install, idempotency, checksum, version)."

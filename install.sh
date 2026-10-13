#!/bin/sh
set -eu

die() {
  printf 'zerogravity installer: %s\n' "$*" >&2
  exit 1
}

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    die 'sha256sum or shasum is required'
  fi
}

VERSION="${ZEROGRAVITY_VERSION:-latest}"
REPO="${ZEROGRAVITY_REPO:-}"
BASE_URL="${ZEROGRAVITY_RELEASE_BASE_URL:-}"
INSTALL_DIR="${ZEROGRAVITY_INSTALL_DIR:-${HOME:-}/.local/bin}"
[ -n "$INSTALL_DIR" ] || die 'set HOME or ZEROGRAVITY_INSTALL_DIR'

case "$(uname -s)" in
  Linux)
    case "$(uname -m)" in
      x86_64|amd64) ASSET='zerogravity-linux-x86_64' ;;
      aarch64|arm64) ASSET='zerogravity-linux-aarch64' ;;
      *) die "unsupported Linux architecture: $(uname -m)" ;;
    esac
    ;;
  Darwin)
    case "$(uname -m)" in
      x86_64|amd64) ASSET='zerogravity-macos-x86_64' ;;
      arm64|aarch64) ASSET='zerogravity-macos-aarch64' ;;
      *) die "unsupported macOS architecture: $(uname -m)" ;;
    esac
    ;;
  MINGW*|MSYS*|CYGWIN*)
    case "$(uname -m)" in
      x86_64|amd64) ASSET='zerogravity-windows-x86_64.exe' ;;
      arm64|aarch64) ASSET='zerogravity-windows-aarch64.exe' ;;
      *) die "unsupported Windows architecture: $(uname -m)" ;;
    esac
    ;;
  *) die "unsupported OS: $(uname -s)" ;;
esac

if [ -z "$BASE_URL" ]; then
  [ -n "$REPO" ] || die 'set ZEROGRAVITY_REPO=OWNER/zerogravity after publishing a GitHub repository (or ZEROGRAVITY_RELEASE_BASE_URL)'
  case "$VERSION" in
    latest) BASE_URL="https://github.com/$REPO/releases/latest/download" ;;
    v*) BASE_URL="https://github.com/$REPO/releases/download/$VERSION" ;;
    *) BASE_URL="https://github.com/$REPO/releases/download/v$VERSION" ;;
  esac
fi

command -v curl >/dev/null 2>&1 || die 'curl is required'
workdir="$(mktemp -d "${TMPDIR:-/tmp}/zerogravity-install.XXXXXX")"
destination_temp=''
cleanup() {
  rm -rf "$workdir"
  [ -z "$destination_temp" ] || rm -f "$destination_temp"
}
trap cleanup EXIT HUP INT TERM

printf 'Downloading zerogravity %s...\n' "$ASSET"
curl --fail --silent --show-error --location "${BASE_URL%/}/$ASSET" --output "$workdir/$ASSET"
curl --fail --silent --show-error --location "${BASE_URL%/}/$ASSET.sha256" --output "$workdir/$ASSET.sha256"

expected="$(awk 'NR == 1 {print $1; exit}' "$workdir/$ASSET.sha256" | tr 'A-F' 'a-f')"
[ "${#expected}" -eq 64 ] || die 'invalid checksum sidecar'
case "$expected" in *[!0-9a-f]*) die 'invalid checksum sidecar' ;; esac
actual="$(hash_file "$workdir/$ASSET" | tr 'A-F' 'a-f')"
[ "$actual" = "$expected" ] || die 'SHA-256 mismatch, refusing to install'

chmod 0755 "$workdir/$ASSET"
reported="$("$workdir/$ASSET" --version)" || die 'downloaded asset fails --version'
case "$reported" in
  'zerogravity v'*) reported_version="${reported#zerogravity v}" ;;
  *) die 'downloaded asset reports an unexpected version string' ;;
esac

if [ "$VERSION" != latest ]; then
  requested="${VERSION#v}"
  [ "$reported_version" = "$requested" ] || die 'downloaded binary version differs from requested version'
fi

mkdir -p "$INSTALL_DIR"
case "$ASSET" in *.exe) binary='zerogravity.exe' ;; *) binary='zerogravity' ;; esac
destination="$INSTALL_DIR/$binary"

if [ -f "$destination" ] && [ "$(hash_file "$destination" | tr 'A-F' 'a-f')" = "$expected" ]; then
  printf 'zerogravity v%s is already installed at %s\n' "$reported_version" "$destination"
  exit 0
fi

destination_temp="$INSTALL_DIR/.$binary.tmp.$$"
cp "$workdir/$ASSET" "$destination_temp"
chmod 0755 "$destination_temp"
mv -f "$destination_temp" "$destination"
destination_temp=''

printf 'Installed zerogravity v%s at %s\n' "$reported_version" "$destination"
case ":${PATH:-}:" in
  *":$INSTALL_DIR:"*) ;;
  *) printf 'Add %s to PATH to use zerogravity.\n' "$INSTALL_DIR" ;;
esac

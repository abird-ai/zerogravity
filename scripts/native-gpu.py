#!/usr/bin/env python3
"""Run a command without network access; optionally verify pinned LLVM first.

Usage: python3 scripts/native-gpu.py -- COMMAND ARG...
       python3 scripts/native-gpu.py --require-llvm -- COMMAND ARG...
       python3 scripts/native-gpu.py --verify-only

Every command inherits the current directory and environment, with Cargo offline
mode enabled, inside a mandatory unshare user/network namespace. No command runs
if namespace setup fails. CPP-only commands need no LLVM installation. Callers
must select --require-llvm for commands using LLVM; --verify-only performs the
same full verification without executing a command or creating a namespace.

LLVM verification requires the existing tracel LLVM 23.1.0-3 bundle, installation
sentinel, archive and checksum sidecar. This script never downloads, installs,
copies or patches them. The locked bundler honors an absolute XDG_DATA_HOME for
installation, but stores artifacts under HOME/.cache/tracel regardless of
XDG_CACHE_HOME. CUDA/ROCm SDK and runtime setup remain the caller's responsibility.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import sys


BUNDLE = "tracel-llvm-23.1.0-3"
ARTIFACT = BUNDLE + "-linux-x64"
SENTINEL = ".tracel-llvm-installed"
ARCHIVE_SHA256 = "1a2b381da394078c82646bd118eb7dceb56320b700a7ef879468e6175a315c1a"
CONTENT_SHA256 = "d1b44f818d17fd882d938a625017b31234fc8b726c190b2a5c324e2be3551bbf"


def data_directory(variable, fallback):
    value = os.environ.get(variable)
    path = Path(value) if value else fallback
    # Match dirs6's Linux XDG rule: ignore empty or relative overrides.
    return path if path.is_absolute() else fallback


def regular_file(path):
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"required regular file is missing: {path}")


def file_hash(path):
    regular_file(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_hash(root):
    """Match upstream PATH LF SIZE LF BYTES ordering, excluding its sentinel.

    Upstream does not hash or follow symlinks. Check their targets separately
    so unmeasured links cannot point outside the verified tree.
    """
    files = []

    def walk_error(error):
        raise error

    for directory, directories, filenames in os.walk(
        root, followlinks=False, onerror=walk_error
    ):
        for name in directories + filenames:
            path = Path(directory) / name
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                # The pinned bundle contains ld64.lld -> lld without lld;
                # upstream ignores such internal links. Only containment is
                # required here, without following them during hashing.
                if not path.resolve(strict=False).is_relative_to(root):
                    raise ValueError(f"LLVM symlink escapes verified tree: {path}")
            elif stat.S_ISREG(metadata.st_mode):
                if relative != SENTINEL:
                    # This replacement also matches upstream on Linux names
                    # containing a literal backslash.
                    files.append((relative.replace("\\", "/"), path))
            elif not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"unsupported LLVM tree entry: {path}")
    digest = hashlib.sha256()
    for relative, path in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\n")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest(), len(files)


def verify():
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("this pinned LLVM bundle requires Linux x86_64")
    home = Path.home()
    if not home.is_absolute():
        raise ValueError("the existing home directory must be absolute")
    data = data_directory("XDG_DATA_HOME", home / ".local/share")
    # Locked bundler 23.1.0-3 honors XDG_DATA_HOME for the installation but
    # hardcodes HOME/.cache for artifacts, ignoring XDG_CACHE_HOME entirely.
    cache = home / ".cache"
    root = (data / "tracel" / BUNDLE).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"required LLVM bundle directory is missing: {root}")
    sentinel = root / SENTINEL
    if sentinel.is_symlink():
        raise ValueError("LLVM installation sentinel must not be a symlink")
    regular_file(sentinel)
    artifacts = (cache / "tracel").resolve(strict=True)
    archive = artifacts / (ARTIFACT + ".tar.xz")
    sidecar = artifacts / (ARTIFACT + ".checksums.json")
    regular_file(sidecar)
    if sidecar.stat().st_size > 64 * 1024:
        raise ValueError("LLVM checksum sidecar exceeds 64 KiB")
    expected = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    if not isinstance(expected, dict) or (
        expected.get("archive_sha256") != ARCHIVE_SHA256
        or expected.get("content_sha256") != CONTENT_SHA256
    ):
        raise ValueError("LLVM checksum sidecar differs from pinned identities")
    archive_digest = file_hash(archive)
    if archive_digest != ARCHIVE_SHA256:
        raise ValueError(f"LLVM archive SHA256 mismatch: {archive_digest}")
    content_digest, count = tree_hash(root)
    if content_digest != CONTENT_SHA256:
        raise ValueError(f"LLVM content SHA256 mismatch: {content_digest}")
    return {
        "bundle": str(root),
        "sentinel": str(sentinel),
        "archive": str(archive.resolve(strict=True)),
        "sidecar": str(sidecar.resolve(strict=True)),
        "archive_sha256": archive_digest,
        "content_sha256": content_digest,
        "regular_files": count,
        "cargo_net_offline": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--require-llvm", action="store_true")
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    options = parser.parse_args(arguments[:separator])
    command = arguments[separator + 1 :] if separator < len(arguments) else []
    if options.verify_only and separator < len(arguments):
        parser.error("--verify-only does not accept a command")
    if not options.verify_only and not command:
        parser.error("provide a command after --, or use --verify-only")
    try:
        if options.verify_only:
            print(json.dumps(verify(), sort_keys=True))
            return 0
        if sys.platform != "linux":
            raise ValueError("native commands require Linux user/network namespaces")
        if options.require_llvm:
            verify()
        unshare = shutil.which("unshare")
        if unshare is None:
            raise ValueError("unshare is unavailable; network isolation is mandatory")
        environment = os.environ.copy()
        environment["CARGO_NET_OFFLINE"] = "true"
        llvm_status = (
            "verified pinned LLVM"
            if options.require_llvm
            else "LLVM verification not requested"
        )
        print(
            f"native-gpu: {llvm_status}; mandatory user/network isolation "
            "via unshare (setup failure aborts the command)",
            file=sys.stderr,
            flush=True,
        )
        os.execvpe(
            unshare,
            [unshare, "--user", "--map-root-user", "--net", "--", *command],
            environment,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"native-gpu: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

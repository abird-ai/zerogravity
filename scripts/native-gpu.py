#!/usr/bin/env python3
"""Run a command without network access; optionally verify pinned LLVM first.

Usage: python3 scripts/native-gpu.py -- COMMAND ARG...
       python3 scripts/native-gpu.py --require-llvm -- COMMAND ARG...
       python3 scripts/native-gpu.py --require-private-network -- COMMAND ARG...
       python3 scripts/native-gpu.py --verify-only
       python3 scripts/native-gpu.py --isolation-info

Every command inherits the current directory and environment, with Cargo offline
mode enabled and mandatory user/network isolation. Inside an existing Bubblewrap
PID/user sandbox, reuse it instead of nesting another user namespace. If
Bubblewrap already has a private network namespace, execute directly; otherwise
deny new non-AF_UNIX sockets with seccomp. Outside Bubblewrap, create the original
private user/network namespaces with unshare. No command runs if the required
isolation cannot be established. Commands that require TCP/UDP loopback must use
--require-private-network; the socket-deny fallback cannot safely provide it.
CPP-only commands need no LLVM installation. Callers must select --require-llvm
for commands using LLVM. --verify-only checks the pinned LLVM bundle without
execution; --isolation-info reports the execution plan without changing process
isolation or launching a command.

LLVM verification requires the existing tracel LLVM 23.1.0-3 bundle, installation
sentinel, archive and checksum sidecar. This script never downloads, installs,
copies or patches them. The locked bundler honors an absolute XDG_DATA_HOME for
installation, but stores artifacts under HOME/.cache/tracel regardless of
XDG_CACHE_HOME. CUDA/ROCm SDK and runtime setup remain the caller's responsibility.
"""

import argparse
import ctypes
import errno
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
INITIAL_UID_MAP = [(0, 0, 4294967295)]


def user_namespace_map(path=Path("/proc/self/uid_map")):
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 3:
            raise ValueError("invalid Linux uid_map")
        rows.append(tuple(int(field) for field in fields))
    if not rows:
        raise ValueError("empty Linux uid_map")
    return rows


def bubblewrap_network_isolation(proc=Path("/proc")):
    """Return Bubblewrap network isolation, or None outside an existing sandbox."""
    try:
        if user_namespace_map(proc / "self/uid_map") == INITIAL_UID_MAP:
            return None
        arguments = [
            item.decode("utf-8", "surrogateescape")
            for item in (proc / "1/cmdline").read_bytes().split(b"\0")
            if item
        ]
    except (OSError, ValueError):
        return None
    if not arguments or Path(arguments[0]).name != "bwrap":
        return None
    try:
        separator = arguments.index("--")
    except ValueError:
        separator = len(arguments)
    options = arguments[1:separator]
    if "--unshare-pid" not in options:
        return None
    return "--unshare-net" in options


def isolation_plan(require_private_network=False):
    """Plan one fail-closed Linux isolation route without mutating the process."""
    if sys.platform != "linux":
        raise ValueError(
            "native commands require Linux user/network namespaces "
            "or equivalent sandbox isolation"
        )
    sandbox_network = bubblewrap_network_isolation()
    if sandbox_network is not None:
        if sandbox_network:
            return {
                "mode": "bubblewrap-network-namespace",
                "existing_bubblewrap": True,
                "private_network_namespace": True,
                "network_isolation": "namespace",
                "loopback_capable": True,
            }
        if require_private_network:
            raise ValueError(
                "existing Bubblewrap sandbox shares its network namespace; "
                "a private network namespace is required"
            )
        return {
            "mode": "bubblewrap-seccomp",
            "existing_bubblewrap": True,
            "private_network_namespace": False,
            "network_isolation": "seccomp-socket-create",
            "loopback_capable": False,
        }
    unshare = shutil.which("unshare")
    if unshare is None:
        raise ValueError("unshare is unavailable; network isolation is mandatory")
    return {
        "mode": "unshare-user-network",
        "existing_bubblewrap": False,
        "private_network_namespace": True,
        "network_isolation": "namespace",
        "loopback_capable": True,
        "unshare": unshare,
    }


def inherited_socket_fds(fd_root=Path("/proc/self/fd")):
    sockets = []
    for entry in fd_root.iterdir():
        try:
            descriptor = int(entry.name)
        except ValueError:
            continue
        try:
            mode = os.fstat(descriptor).st_mode
        except OSError as error:
            if error.errno == errno.EBADF:
                continue
            raise
        if stat.S_ISSOCK(mode):
            sockets.append(descriptor)
    return sorted(sockets)


def install_no_network_seccomp():
    """Deny non-AF_UNIX socket creation without creating another user namespace."""
    if platform.machine() != "x86_64":
        raise ValueError(
            "existing-sandbox seccomp network isolation currently requires x86_64"
        )
    sockets = inherited_socket_fds()
    if sockets:
        raise ValueError(
            "inherited socket descriptors prevent fail-closed network isolation: "
            + ",".join(str(descriptor) for descriptor in sockets)
        )

    class SockFilter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class SockFprog(ctypes.Structure):
        _fields_ = [
            ("len", ctypes.c_ushort),
            ("filter", ctypes.POINTER(SockFilter)),
        ]

    # Linux classic-BPF seccomp over struct seccomp_data:
    #   u32 nr @ 0, u32 arch @ 4, u64 args[0] @ 16.
    bpf_ld_w_abs = 0x20
    bpf_jmp_jeq_k = 0x15
    bpf_ret_k = 0x06
    seccomp_ret_kill_process = 0x80000000
    seccomp_ret_errno = 0x00050000
    seccomp_ret_allow = 0x7FFF0000
    audit_arch_x86_64 = 0xC000003E
    sys_socket = 41
    sys_socketpair = 53
    # io_uring can create sockets without invoking the socket syscall directly.
    # Deny the ring API in this fallback so socket creation cannot bypass seccomp.
    sys_io_uring_setup = 425
    sys_io_uring_enter = 426
    sys_io_uring_register = 427
    af_unix = 1
    instructions = [
        (bpf_ld_w_abs, 0, 0, 4),
        (bpf_jmp_jeq_k, 1, 0, audit_arch_x86_64),
        (bpf_ret_k, 0, 0, seccomp_ret_kill_process),
        (bpf_ld_w_abs, 0, 0, 0),
        (bpf_jmp_jeq_k, 6, 0, sys_socket),
        (bpf_jmp_jeq_k, 5, 0, sys_socketpair),
        (bpf_jmp_jeq_k, 3, 0, sys_io_uring_setup),
        (bpf_jmp_jeq_k, 2, 0, sys_io_uring_enter),
        (bpf_jmp_jeq_k, 1, 0, sys_io_uring_register),
        (bpf_ret_k, 0, 0, seccomp_ret_allow),
        (bpf_ret_k, 0, 0, seccomp_ret_errno | errno.EPERM),
        (bpf_ld_w_abs, 0, 0, 16),
        (bpf_jmp_jeq_k, 0, 1, af_unix),
        (bpf_ret_k, 0, 0, seccomp_ret_allow),
        (bpf_ret_k, 0, 0, seccomp_ret_errno | errno.EPERM),
    ]
    filters = (SockFilter * len(instructions))(
        *(SockFilter(*instruction) for instruction in instructions)
    )
    program = SockFprog(len(instructions), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    # no_new_privs is already set by Bubblewrap, but require it explicitly before
    # installing the filter so this path also fails closed in equivalent sandboxes.
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:  # PR_SET_SECCOMP/FILTER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


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
    mode.add_argument("--isolation-info", action="store_true")
    parser.add_argument("--require-llvm", action="store_true")
    parser.add_argument("--require-private-network", action="store_true")
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    options = parser.parse_args(arguments[:separator])
    command = arguments[separator + 1 :] if separator < len(arguments) else []
    if (options.verify_only or options.isolation_info) and separator < len(arguments):
        parser.error("inspection modes do not accept a command")
    if options.verify_only and (options.require_llvm or options.require_private_network):
        parser.error("--verify-only does not accept execution requirements")
    if options.isolation_info and options.require_llvm:
        parser.error("--isolation-info does not verify LLVM")
    if not (options.verify_only or options.isolation_info) and not command:
        parser.error("provide a command after --, or use an inspection mode")
    try:
        if options.verify_only:
            print(json.dumps(verify(), sort_keys=True))
            return 0
        if options.isolation_info:
            plan = isolation_plan(options.require_private_network)
            print(json.dumps(plan, sort_keys=True))
            return 0
        if options.require_llvm:
            verify()
        plan = isolation_plan(options.require_private_network)
        environment = os.environ.copy()
        environment["CARGO_NET_OFFLINE"] = "true"
        environment["ZG_NATIVE_GPU_ISOLATION"] = plan["mode"]
        llvm_status = (
            "verified pinned LLVM"
            if options.require_llvm
            else "LLVM verification not requested"
        )
        if plan["existing_bubblewrap"]:
            if plan["network_isolation"] == "namespace":
                isolation = "existing Bubblewrap user/network namespaces"
            else:
                install_no_network_seccomp()
                isolation = "existing Bubblewrap user namespace + seccomp network filter"
            print(
                f"native-gpu: {llvm_status}; mandatory user/network isolation "
                f"via {isolation} (setup failure aborts the command)",
                file=sys.stderr,
                flush=True,
            )
            os.execvpe(command[0], command, environment)

        unshare = plan["unshare"]
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

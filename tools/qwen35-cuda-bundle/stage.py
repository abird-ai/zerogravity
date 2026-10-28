#!/usr/bin/env python3
"""Structurally translate/check Qwen CUDA bundles; Rust authenticates payload bytes."""

from contextlib import ExitStack
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat

MIB = 1024 * 1024
GIB = 1024 * MIB
LEGACY = "short-hf-native.json"
CURRENT = "cuda-kernels.json"
OPTIONAL_CURRENT = {"stage-result.json"}
MAX_FILES = 64
MAX_DEPTH = 3
MAX_RESOURCE_BYTES = GIB
MAX_TOTAL_BYTES = 2 * GIB

KIND_RENAMES = {
    "Canonical47CutlassBf16Sm80": "FixedPromptCutlassBf16Sm80",
    "ShortHfCutlassBf16Sm80": "TextCutlassBf16Sm80",
    "ShortHfFaCausalCombineSm80": "TextFaCausalCombineSm80",
    "ShortHfFaSplitCombineSm80": "TextFaSplitCombineSm80",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stamp(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def decode(raw):
    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _: require(False, "nonfinite JSON"),
    )


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def relative(value):
    require(
        isinstance(value, str)
        and 0 < len(value) <= 256
        and "\\" not in value
        and "\0" not in value,
        "bounded POSIX resource path",
    )
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in ("", ".", "..") for part in value.split("/")),
        "canonical relative resource path",
    )
    return path.parts


def kind(value):
    require(
        isinstance(value, str)
        and 0 < len(value) <= 96
        and value.isascii()
        and value.isalnum(),
        "bounded native kind",
    )
    return value


class Directory:
    def __init__(self, path):
        self.path = Path(path)
        require(self.path.is_absolute(), "absolute directory required")
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for part in self.path.parts[1:]:
                require(part not in (".", ".."), "canonical directory required")
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=fd,
                )
                os.close(fd)
                fd = next_fd
            self.fd = fd
            self.before = stamp(os.fstat(fd))
        except BaseException:
            os.close(fd)
            raise

    def close(self):
        os.close(self.fd)

    def parent(self, name):
        parts = relative(name)
        fd = os.dup(self.fd)
        try:
            for part in parts[:-1]:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=fd,
                )
                os.close(fd)
                fd = next_fd
            return fd, parts[-1]
        except BaseException:
            os.close(fd)
            raise

    def open(self, name):
        parent, leaf = self.parent(name)
        try:
            return os.open(
                leaf,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=parent,
            )
        finally:
            os.close(parent)

    def inventory(self):
        found = set()

        def walk(fd, prefix="", depth=0):
            require(depth <= MAX_DEPTH, "bounded bundle depth")
            scan = os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC, dir_fd=fd)
            try:
                with os.scandir(scan) as entries:
                    for entry in entries:
                        require(len(found) < MAX_FILES * 2, "bounded bundle inventory")
                        name = prefix + entry.name
                        relative(name)
                        mode = entry.stat(follow_symlinks=False).st_mode
                        if stat.S_ISDIR(mode):
                            found.add(name + "/")
                            child = os.open(
                                entry.name,
                                os.O_RDONLY
                                | os.O_DIRECTORY
                                | os.O_NOFOLLOW
                                | os.O_CLOEXEC,
                                dir_fd=fd,
                            )
                            try:
                                walk(child, name + "/", depth + 1)
                            finally:
                                os.close(child)
                        else:
                            require(
                                stat.S_ISREG(mode),
                                "bundle resources must be regular, no symlinks",
                            )
                            found.add(name)
            finally:
                os.close(scan)

        walk(self.fd)
        return found

    def check(self):
        require(
            stamp(os.fstat(self.fd)) == self.before == stamp(self.path.lstat()),
            "held bundle directory identity changed",
        )

    def require_same_node(self):
        current = stamp(os.fstat(self.fd))
        path = stamp(self.path.lstat())
        require(
            current[:3] == self.before[:3] == path[:3],
            "held bundle directory node changed",
        )


class Held:
    def __init__(self, directory, name):
        self.directory = directory
        self.name = name
        self.fd = directory.open(name)
        try:
            self.before = stamp(os.fstat(self.fd))
            require(
                stat.S_ISREG(self.before[2])
                and 0 < self.before[3] <= MAX_RESOURCE_BYTES,
                "bounded regular resource: " + name,
            )
        except BaseException:
            os.close(self.fd)
            raise

    @property
    def size(self):
        return self.before[3]

    def close(self):
        os.close(self.fd)

    def chunks(self):
        offset = 0
        while offset < self.size:
            block = os.pread(self.fd, min(MIB, self.size - offset), offset)
            require(bool(block), "complete resource read: " + self.name)
            offset += len(block)
            yield block
        require(os.pread(self.fd, 1, offset) == b"", "resource exact EOF: " + self.name)

    def verify_unchanged(self):
        current = self.directory.open(self.name)
        try:
            require(
                self.before == stamp(os.fstat(self.fd)) == stamp(os.fstat(current)),
                "held resource changed: " + self.name,
            )
        finally:
            os.close(current)


def read_bounded(directory, name, cap):
    fd = directory.open(name)
    try:
        before = stamp(os.fstat(fd))
        require(
            stat.S_ISREG(before[2]) and 0 < before[3] <= cap,
            "bounded regular metadata: " + name,
        )
        chunks = []
        offset = 0
        while offset < before[3]:
            block = os.pread(fd, min(MIB, before[3] - offset), offset)
            require(bool(block), "complete held metadata read: " + name)
            chunks.append(block)
            offset += len(block)
        require(os.pread(fd, 1, offset) == b"", "held metadata exact EOF: " + name)
        raw = b"".join(chunks)
        require(
            len(raw) == before[3] and stamp(os.fstat(fd)) == before,
            "held metadata changed: " + name,
        )
        return raw
    finally:
        os.close(fd)


def row(value, *, translate):
    require(isinstance(value, dict) and set(value) == {"kind", "path"}, "native row fields")
    old = kind(value["kind"])
    return {
        "kind": KIND_RENAMES.get(old, old) if translate else old,
        "path": value["path"],
    }


def manifest(raw, *, legacy):
    value = decode(raw)
    schema = "zerogravity-short-hf-native-v1" if legacy else "zerogravity.cuda-kernels.v2"
    require(
        isinstance(value, dict)
        and set(value) == {"schema", "images", "activation", "rotary", "cublas", "cublas_lt"}
        and value["schema"] == schema,
        "closed native manifest fields/schema",
    )

    output = {"schema": "zerogravity.cuda-kernels.v2"}
    resources = []
    for section, maximum in (("images", 48), ("rotary", 16)):
        rows = value[section]
        require(isinstance(rows, list) and 1 <= len(rows) <= maximum, "bounded native rows")
        translated = [row(item, translate=legacy) for item in rows]
        require(
            len({item["kind"] for item in translated}) == len(translated),
            "distinct native kinds",
        )
        output[section] = translated
        resources.extend(item["path"] for item in translated)

    for section in ("activation", "cublas", "cublas_lt"):
        relative(value[section])
        output[section] = value[section]
        resources.append(value[section])

    for name in resources:
        relative(name)
    require(
        len(resources) <= MAX_FILES and len(set(resources)) == len(resources),
        "distinct bounded native resources",
    )
    return output, resources


def expected_inventory(manifest_name, resources, *, optional=()):
    result = {manifest_name}
    for name in resources:
        parts = relative(name)
        result.add(name)
        result.update("/".join(parts[:i]) + "/" for i in range(1, len(parts)))
    result.update(optional)
    return result


def inspect(root, manifest_name, *, legacy):
    with ExitStack() as stack:
        directory = Directory(root)
        stack.callback(directory.close)
        raw = read_bounded(directory, manifest_name, 64 * 1024)
        current, resources = manifest(raw, legacy=legacy)
        held = []
        total = 0
        for name in resources:
            item = Held(directory, name)
            stack.callback(item.close)
            held.append(item)
            total += item.size
            require(total <= MAX_TOTAL_BYTES, "bounded total native resource bytes")

        inventory = directory.inventory()
        expected = expected_inventory(
            manifest_name,
            resources,
            optional=OPTIONAL_CURRENT if not legacy else (),
        )
        if not legacy and "stage-result.json" not in inventory:
            expected.discard("stage-result.json")
        require(inventory == expected, "missing or extra native bundle resource")
        for item in held:
            item.verify_unchanged()
        directory.check()
        return current, resources, total


def check_legacy(root):
    current, resources, total = inspect(Path(root), LEGACY, legacy=True)
    return {
        "complete": True,
        "schema": "legacy",
        "resources": len(resources),
        "bytes": total,
        "current_manifest": current,
    }


def check_current(root):
    _, resources, total = inspect(Path(root), CURRENT, legacy=False)
    return {
        "complete": True,
        "schema": "current",
        "resources": len(resources),
        "bytes": total,
    }


def write_new(directory, name, raw):
    parent, leaf = directory.parent(name)
    try:
        fd = os.open(
            leaf,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=parent,
        )
    finally:
        os.close(parent)
    with os.fdopen(fd, "wb") as target:
        require(target.write(raw) == len(raw), "complete metadata write")
        target.flush()
        os.fsync(target.fileno())


def stage(legacy_root, output):
    legacy_root, output = Path(legacy_root), Path(output)
    require(
        output.is_absolute() and not output.exists() and not output.is_symlink(),
        "fresh absolute output required",
    )
    require(
        output.parent.is_dir()
        and not output.is_relative_to(legacy_root)
        and not legacy_root.is_relative_to(output),
        "disjoint output with existing parent",
    )

    created = False
    created_node = None
    try:
        with ExitStack() as stack:
            source = Directory(legacy_root)
            stack.callback(source.close)
            raw = read_bounded(source, LEGACY, 64 * 1024)
            current, resources = manifest(raw, legacy=True)

            held = []
            total = 0
            for name in resources:
                item = Held(source, name)
                stack.callback(item.close)
                held.append(item)
                total += item.size
                require(total <= MAX_TOTAL_BYTES, "bounded total native resource bytes")

            require(
                source.inventory() == expected_inventory(LEGACY, resources),
                "missing or extra legacy resource",
            )

            parent = Directory(output.parent)
            stack.callback(parent.close)
            os.mkdir(output.name, 0o700, dir_fd=parent.fd)
            created = True
            staged = Directory(output)
            stack.callback(staged.close)
            created_node = staged.before[:3]

            directories = sorted(
                {str(PurePosixPath(name).parent) for name in resources if "/" in name}
            )
            for name in directories:
                directory_fd = os.dup(staged.fd)
                try:
                    for part in relative(name):
                        try:
                            os.mkdir(part, 0o700, dir_fd=directory_fd)
                        except FileExistsError:
                            pass
                        next_fd = os.open(
                            part,
                            os.O_RDONLY
                            | os.O_DIRECTORY
                            | os.O_NOFOLLOW
                            | os.O_CLOEXEC,
                            dir_fd=directory_fd,
                        )
                        os.close(directory_fd)
                        directory_fd = next_fd
                finally:
                    os.close(directory_fd)

            for item in held:
                directory_fd, leaf = staged.parent(item.name)
                try:
                    fd = os.open(
                        leaf,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                        0o600,
                        dir_fd=directory_fd,
                    )
                finally:
                    os.close(directory_fd)
                with os.fdopen(fd, "wb") as target:
                    copied = 0
                    for block in item.chunks():
                        require(target.write(block) == len(block), "complete resource write")
                        copied += len(block)
                    require(copied == item.size, "complete resource extent")
                    target.flush()
                    os.fsync(target.fileno())
                item.verify_unchanged()

            write_new(staged, CURRENT, encoded(current))
            require(
                staged.inventory() == expected_inventory(CURRENT, resources),
                "staged current inventory differs",
            )
            staged.require_same_node()
            os.fsync(staged.fd)
            source.check()
            os.fsync(parent.fd)

        return {
            "complete": True,
            "resources": len(resources),
            "bytes": total,
        }
    except BaseException:
        if created and output.exists() and not output.is_symlink():
            current = output.lstat()
            if created_node is not None and (current.st_dev, current.st_ino, current.st_mode) == created_node:
                shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-legacy", type=Path)
    group.add_argument("--check-current", type=Path)
    group.add_argument("--legacy-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.check_legacy is not None:
        require(args.output is None, "--check-legacy takes no output")
        result = check_legacy(args.check_legacy)
    elif args.check_current is not None:
        require(args.output is None, "--check-current takes no output")
        result = check_current(args.check_current)
    else:
        require(args.output is not None, "--legacy-root requires --output")
        result = stage(args.legacy_root, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Copy authenticated legacy CUDA resources into a fresh current-schema bundle."""
from contextlib import ExitStack
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MIB = 1024 * 1024
LEGACY = "short-hf-native.json"
CURRENT = "cuda-kernels.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stamp(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "nonfinite JSON"))


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def relative(value):
    require(isinstance(value, str) and 0 < len(value) <= 256 and "\\" not in value
            and "\0" not in value, "bounded POSIX resource path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.as_posix() == value
            and all(part not in ("", ".", "..") for part in value.split("/")),
            "canonical relative resource path")
    return path.parts


class Directory:
    def __init__(self, path):
        self.path = Path(path)
        require(self.path.is_absolute(), "absolute directory required")
        # Refuse symlinks in every component, including an ancestor of the root.
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for part in self.path.parts[1:]:
                require(part not in (".", ".."), "canonical directory required")
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                  dir_fd=fd)
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
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                  dir_fd=fd)
                os.close(fd)
                fd = next_fd
            return fd, parts[-1]
        except BaseException:
            os.close(fd)
            raise

    def open(self, name):
        parent, leaf = self.parent(name)
        try:
            return os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                           dir_fd=parent)
        finally:
            os.close(parent)

    def inventory(self):
        found = set()

        def walk(fd, prefix="", depth=0):
            require(depth <= 3, "bounded bundle depth")
            scan = os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC, dir_fd=fd)
            try:
                with os.scandir(scan) as entries:
                    for entry in entries:
                        require(len(found) < 64, "bounded bundle inventory")
                        name = prefix + entry.name
                        relative(name)
                        mode = entry.stat(follow_symlinks=False).st_mode
                        if stat.S_ISDIR(mode):
                            found.add(name + "/")
                            child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                            dir_fd=fd)
                            try:
                                walk(child, name + "/", depth + 1)
                            finally:
                                os.close(child)
                        else:
                            require(stat.S_ISREG(mode), "bundle resources must be regular, no symlinks")
                            found.add(name)
            finally:
                os.close(scan)
        walk(self.fd)
        return found

    def check(self):
        require(stamp(os.fstat(self.fd)) == self.before == stamp(self.path.lstat()),
                "held source directory identity changed")


class Held:
    def __init__(self, directory, name, pin, cap=1024 * MIB):
        require(type(pin["bytes"]) is int and 0 < pin["bytes"] <= cap
                and isinstance(pin["sha256"], str) and len(pin["sha256"]) == 64,
                "bounded expected resource pin")
        self.directory, self.name, self.pin = directory, name, pin
        self.fd = directory.open(name)
        try:
            self.before = stamp(os.fstat(self.fd))
            require(stat.S_ISREG(self.before[2]) and self.before[3] == pin["bytes"],
                    "exact regular resource extent: " + name)
            self.verify()
        except BaseException:
            os.close(self.fd)
            raise

    def close(self):
        os.close(self.fd)

    def chunks(self):
        offset = 0
        while offset < self.pin["bytes"]:
            block = os.pread(self.fd, min(MIB, self.pin["bytes"] - offset), offset)
            require(bool(block), "complete resource read")
            offset += len(block)
            yield block
        require(os.pread(self.fd, 1, offset) == b"", "resource exact EOF")

    def verify(self):
        digest = hashlib.sha256()
        for block in self.chunks():
            digest.update(block)
        require(digest.hexdigest() == self.pin["sha256"], "resource SHA differs: " + self.name)
        current = self.directory.open(self.name)
        try:
            require(self.before == stamp(os.fstat(self.fd)) == stamp(os.fstat(current)),
                    "held FD/path resource changed: " + self.name)
        finally:
            os.close(current)

    def raw(self):
        return b"".join(self.chunks())


def plan(raw, pins):
    require(set(pins) == {"schema", "contract_sources", "profiles"}
            and pins["schema"] == "zerogravity.cuda-bundle-migration-pins.v1"
            and set(pins["profiles"]) == {"0.8b", "2b"}, "closed migration catalogue")
    digest = hashlib.sha256(raw).hexdigest()
    matched = [(name, row) for name, row in pins["profiles"].items()
               if row["legacy_manifest"] == {"bytes": len(raw), "sha256": digest}]
    require(len(matched) == 1, "exact authenticated legacy manifest required")
    family, profile = matched[0]
    value = decode(raw)
    require(set(value) == {"schema", "images", "activation", "rotary", "cublas", "cublas_lt"}
            and value["schema"] == "zerogravity-short-hf-native-v1", "closed legacy manifest fields")
    expected = profile["resources"]
    require(1 <= len(expected) <= 32 and len({r["path"] for r in expected}) == len(expected),
            "distinct complete pinned resources")
    output = {"schema": "zerogravity.cuda-kernels.v2"}
    for section in ("images", "rotary"):
        rows = [r for r in expected if r["section"] == section]
        require(value[section] == [{"kind": r["legacy_kind"], "path": r["path"]} for r in rows],
                "exact legacy kind/path inventory")
        require(len({r["kind"] for r in rows}) == len(rows), "distinct current kinds")
        output[section] = [{"kind": r["kind"], "path": r["path"]} for r in rows]
    for section in ("activation", "cublas", "cublas_lt"):
        rows = [r for r in expected if r["section"] == section]
        require(len(rows) == 1 and value[section] == rows[0]["path"], "exact support resource")
        output[section] = value[section]
    require(set(r["section"] for r in expected) == {"images", "rotary", "activation", "cublas", "cublas_lt"}
            and sum(r["bytes"] for r in expected) <= 1536 * MIB, "bounded closed resource cohort")
    for row in expected:
        relative(row["path"])
    return family, profile, output


def write_new(directory, name, raw):
    parent, leaf = directory.parent(name)
    try:
        fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=parent)
    finally:
        os.close(parent)
    with os.fdopen(fd, "wb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    return {"path": str(directory.path / name), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def inventory_names(resources):
    result = {LEGACY}
    for row in resources:
        parts = relative(row["path"])
        result.add(row["path"])
        result.update("/".join(parts[:i]) + "/" for i in range(1, len(parts)))
    return result


def stage(legacy_root, output):
    legacy_root, output = Path(legacy_root), Path(output)
    require(output.is_absolute() and not output.exists() and not output.is_symlink(), "fresh absolute output required")
    require(output.parent.is_dir() and not output.is_relative_to(legacy_root)
            and not legacy_root.is_relative_to(output), "disjoint output with existing parent")
    with ExitStack() as stack:
        source = Directory(legacy_root)
        stack.callback(source.close)
        tool = Directory(HERE)
        stack.callback(tool.close)
        pins_raw = (HERE / "pins.json").read_bytes()
        require(0 < len(pins_raw) <= 64 * 1024, "bounded migration catalogue")
        pins = decode(pins_raw)
        held_pins = Held(tool, "pins.json", {"bytes": len(pins_raw), "sha256": hashlib.sha256(pins_raw).hexdigest()}, MIB)
        stack.callback(held_pins.close)
        require(held_pins.raw() == pins_raw, "same catalogue bytes")
        repository = Directory(REPO)
        stack.callback(repository.close)
        contracts = []
        for name, pin in pins["contract_sources"].items():
            held = Held(repository, name, pin, MIB)
            stack.callback(held.close)
            contracts.append(held)
        manifest_fd = source.open(LEGACY)
        try:
            size = os.fstat(manifest_fd).st_size
            require(0 < size <= 64 * 1024, "bounded legacy manifest")
            raw = os.pread(manifest_fd, size + 1, 0)
        finally:
            os.close(manifest_fd)
        family, profile, current = plan(raw, pins)
        manifest = Held(source, LEGACY, profile["legacy_manifest"], MIB)
        stack.callback(manifest.close)
        require(manifest.raw() == raw, "same held authenticated legacy manifest")
        expected = inventory_names(profile["resources"])
        require(source.inventory() == expected, "missing or extra legacy resource")
        resources = []
        for row in profile["resources"]:
            held = Held(source, row["path"], row)
            stack.callback(held.close)
            resources.append(held)
        # All original resources are authenticated and held before any output.
        parent = Directory(output.parent)
        stack.callback(parent.close)
        os.mkdir(output.name, 0o700, dir_fd=parent.fd)
        staged = None
        try:
            staged = Directory(output)
            stack.callback(staged.close)
            for name in sorted({str(PurePosixPath(r.name).parent) for r in resources}):
                directory_fd = os.dup(staged.fd)
                try:
                    for part in relative(name):
                        try:
                            os.mkdir(part, 0o700, dir_fd=directory_fd)
                        except FileExistsError:
                            pass
                        next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                          dir_fd=directory_fd)
                        os.close(directory_fd)
                        directory_fd = next_fd
                finally:
                    os.close(directory_fd)
            copied = []
            for held in resources:
                directory_fd, leaf = staged.parent(held.name)
                try:
                    fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                                 0o600, dir_fd=directory_fd)
                finally:
                    os.close(directory_fd)
                digest = hashlib.sha256()
                with os.fdopen(fd, "wb") as stream:
                    for block in held.chunks():
                        require(stream.write(block) == len(block), "complete resource write")
                        digest.update(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                require(digest.hexdigest() == held.pin["sha256"], "copied original resource differs")
                copied.append({"path": held.name, "bytes": held.pin["bytes"], "sha256": digest.hexdigest()})
            current_pin = write_new(staged, CURRENT, encoded(current))
            # Reopen all output resources through NOFOLLOW directory capabilities.
            for row in copied:
                held = Held(staged, row["path"], row)
                stack.callback(held.close)
            held_manifest = Held(staged, CURRENT, current_pin, MIB)
            stack.callback(held_manifest.close)
            for held in [held_pins, manifest, *contracts, *resources]:
                held.verify()
            source.check()
            require(source.inventory() == expected, "source inventory changed during copy")
            require(stamp(os.fstat(staged.fd))[:3] == stamp(output.lstat())[:3],
                    "staged directory identity changed")
            receipt = dict(schema="zerogravity.cuda-bundle-stage.v1", complete=True, family=family,
                           legacy_root=str(legacy_root), legacy_manifest=profile["legacy_manifest"],
                           current_manifest=current_pin, resources=copied,
                           migration_pins_sha256=hashlib.sha256(pins_raw).hexdigest(),
                           driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           contract_sources=pins["contract_sources"],
                           transformations="manifest filename/schema and four closed kind labels only; resource bytes and relative paths unchanged",
                           native_catalogue=profile["native_catalogue"],
                           cohort_2b_extra_image_included=False,
                           admission_enabled=False, native_execution_qualified=False,
                           model_execution_qualified=False, performance_qualified=False)
            write_new(staged, "stage-result.json", encoded(receipt))
            os.fsync(staged.fd)
            os.fsync(parent.fd)
            return receipt
        except BaseException as error:
            if staged is not None:
                write_new(staged, "stage-failure.json", encoded(dict(complete=False, error_type=type(error).__name__,
                          message=str(error), native_execution_qualified=False)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = stage(args.legacy_root, args.output)
    print(json.dumps({key: result[key] for key in ("complete", "family", "current_manifest", "native_execution_qualified")}))


if __name__ == "__main__":
    main()

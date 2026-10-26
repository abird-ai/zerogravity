#!/usr/bin/env python3
"""Check Cargo workspace dependency boundaries and the production DAG.

Includes normal/build dependencies, every target table and optional dependency.
Dev dependencies are excluded. Uses manifests only; never invokes Cargo/network.
"""

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib


@dataclass(frozen=True)
class Dependency:
    source: str
    target: str
    path: Path | None
    manifest: Path
    table: str
    alias: str


def read_manifest(path):
    with path.open("rb") as stream:
        return tomllib.load(stream)


def dependency_tables(data):
    for kind in ("dependencies", "build-dependencies"):
        yield kind, data.get(kind, {})
    for target, tables in sorted(data.get("target", {}).items()):
        for kind in ("dependencies", "build-dependencies"):
            yield f"target.{target}.{kind}", tables.get(kind, {})


def resolve_dependency(alias, spec, inherited, manifest, root):
    directory = manifest.parent
    if isinstance(spec, dict) and "workspace" in spec:
        if spec["workspace"] is not True or alias not in inherited:
            raise ValueError(f"{manifest}: unresolved workspace dependency {alias}")
        if any(key in spec for key in ("package", "path", "version", "git", "registry")):
            raise ValueError(f"{manifest}: dependency {alias} overrides inherited identity")
        spec = inherited[alias]
        directory = root
    if isinstance(spec, str):
        return alias, None
    if not isinstance(spec, dict):
        raise ValueError(f"{manifest}: invalid dependency {alias}")
    name = spec.get("package", alias)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{manifest}: invalid package name for {alias}")
    path = spec.get("path")
    if path is not None and not isinstance(path, str):
        raise ValueError(f"{manifest}: invalid dependency path for {alias}")
    return name, (directory / path).resolve() if path is not None else None


def workspace_dependencies(root):
    root = root.resolve()
    document = read_manifest(root / "Cargo.toml")
    workspace = document.get("workspace")
    if not isinstance(workspace, dict):
        raise ValueError(f"{root}: expected a Cargo workspace")
    excluded = {
        path.resolve()
        for pattern in workspace.get("exclude", [])
        for path in root.glob(pattern)
    }

    def is_excluded(path):
        return any(path == item or path.is_relative_to(item) for item in excluded)

    pending = set()
    for pattern in workspace.get("members", []):
        matches = list(root.glob(pattern))
        if not matches:
            raise ValueError(f"{root}: workspace member pattern has no match: {pattern}")
        pending.update(path.resolve() / "Cargo.toml" for path in matches
                       if not is_excluded(path.resolve()))
    if "package" in document:
        pending.add(root / "Cargo.toml")
    packages = {}
    paths = {}
    dependencies = []
    inherited = workspace.get("dependencies", {})
    while pending:
        manifest = min(pending)
        pending.remove(manifest)
        if manifest.parent in paths:
            continue
        data = read_manifest(manifest)
        name = data.get("package", {}).get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{manifest}: missing package.name")
        if name in packages:
            raise ValueError(f"{manifest}: duplicate workspace package {name}")
        packages[name] = manifest
        paths[manifest.parent] = name
        for table, entries in dependency_tables(data):
            for alias, spec in sorted(entries.items()):
                target, path = resolve_dependency(alias, spec, inherited, manifest, root)
                dependencies.append(Dependency(name, target, path, manifest, table, alias))
                # Cargo automatically includes in-workspace path dependencies.
                if path is not None and path.is_relative_to(root) and not is_excluded(path):
                    pending.add(path / "Cargo.toml")
    if not packages:
        raise ValueError(f"{root}: workspace contains no packages")
    for edge in dependencies:
        if edge.path in paths and paths[edge.path] != edge.target:
            raise ValueError(
                f"{edge.manifest}: {edge.table}.{edge.alias} names {edge.target}, "
                f"but its path contains {paths[edge.path]}; use package for an alias"
            )
    return packages, paths, dependencies


def layer(package):
    if package.startswith("zerogravity-model-"):
        return 3
    return {
        "zerogravity-core": 0,
        "zerogravity-artifacts": 1,
        "zerogravity-cpu": 2,
        "zerogravity-gpu": 2,
        "zerogravity": 4,
        "zerogravity-cli": 5,
    }.get(package)


def check_workspace(root):
    packages, paths, dependencies = workspace_dependencies(root)
    outgoing = {name: [] for name in packages}
    local = {name: set() for name in packages}
    for edge in dependencies:
        outgoing[edge.source].append(edge)
        if edge.path in paths:
            local[edge.source].add(paths[edge.path])
    errors = []
    # Check reachable local helpers too, so an alias/helper cannot conceal an
    # upward dependency. External dependencies are leaves, identified by package.
    for source in sorted(packages):
        source_layer = layer(source)
        if source_layer is None:
            continue
        queue = deque([(source, [source])])
        visited = {source}
        while queue:
            node, chain = queue.popleft()
            for edge in outgoing.get(node, []):
                target_layer = layer(edge.target)
                if target_layer is not None and target_layer > source_layer:
                    errors.append(
                        f"forbidden dependency: {' -> '.join(chain + [edge.target])} "
                        f"({edge.manifest.relative_to(root.resolve())}: "
                        f"{edge.table}.{edge.alias})"
                    )
                if edge.path in paths and edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, chain + [edge.target]))
    active = []
    done = set()

    def cycle(node):
        if node in active:
            return active[active.index(node):] + [node]
        if node in done:
            return None
        active.append(node)
        for target in sorted(local[node]):
            found = cycle(target)
            if found:
                return found
        active.pop()
        done.add(node)
        return None

    for source in sorted(packages):
        found = cycle(source)
        if found:
            errors.append("production dependency cycle: " + " -> ".join(found))
            break
    return sorted(set(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        errors = check_workspace(args.workspace)
    except (OSError, ValueError, TypeError) as error:
        print(f"architecture: {error}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("architecture: dependency boundaries and production DAG pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())

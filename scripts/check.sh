#!/usr/bin/env bash
set -euo pipefail

repo="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required for workspace and qualification checks." >&2; exit 1; }

echo "Checking Rust formatting..."
cargo fmt --all -- --check

# Explicit CPU owners keep optional GPU compilers and qualification tools out
# of default CI. Cargo's --all-features would also enable diagnostic GPU paths.
cpu_packages=(
  -p zerogravity-core -p zerogravity-artifacts -p zerogravity-cpu
  -p zerogravity-model-qwen35 -p zerogravity-model-gemma4 -p zerogravity -p zerogravity-cli
)
cpu_features='zerogravity-cli/cpu,zerogravity-cli/model-qwen35,zerogravity-cli/text,zerogravity-cli/diagnostic-cpu-checkpoint,zerogravity-cli/diagnostic-gdn-trace,zerogravity/model-gemma4'
cpu_args=(--locked --offline --no-default-features "${cpu_packages[@]}" --features "$cpu_features")

echo "Checking workspace owners and default CPU dependency isolation..."
python3 - <<'PYWORKSPACE'
import json, subprocess
metadata = json.loads(subprocess.check_output([
    "cargo", "metadata", "--locked", "--offline", "--no-deps", "--format-version", "1"
]))
names = {p["id"]: p["name"] for p in metadata["packages"]}
expected = {"zerogravity-core", "zerogravity-artifacts", "zerogravity-cpu", "zerogravity-model-qwen35", "zerogravity-model-gemma4", "zerogravity", "zerogravity-cli", "zerogravity-gpu", "zerogravity-gpu-qualify"}
assert {names[m] for m in metadata["workspace_members"]} == expected
assert {names[m] for m in metadata["workspace_default_members"]} == {"zerogravity", "zerogravity-cli"}
tree = subprocess.check_output([
    "cargo", "tree", "--locked", "--offline", "-p", "zerogravity-cli", "--no-default-features",
    "--features", "cpu,model-qwen35,text", "--edges", "normal,build", "--prefix", "none", "--format", "{p}"
], text=True)
for line in tree.splitlines():
    name = line.split()[0]
    assert not name.startswith(("cubecl", "tracel-llvm")) and name != "zerogravity-gpu", line
PYWORKSPACE

echo "Checking CPU clippy (including explicit CPU diagnostics)..."
cargo clippy "${cpu_args[@]}" --all-targets -- -D warnings

echo "Running CPU workspace tests..."
cargo test "${cpu_args[@]}" --all-targets

echo "Checking minimal core, CPU-free family/checkpoint and inspection-only CLI..."
cargo check --locked --offline -p zerogravity-core --no-default-features
cargo check --locked --offline -p zerogravity-model-qwen35 --no-default-features --features diagnostic-cpu-checkpoint
cargo check --locked --offline -p zerogravity-model-gemma4 --no-default-features
cargo test --locked --offline -p zerogravity --no-default-features --features model-gemma4 --test gemma_contract
cargo check --locked --offline -p zerogravity-cli --no-default-features

echo "Checking optional local artifact-cache transactions and diagnostics..."
cargo clippy --locked --offline -p zerogravity-artifacts --no-default-features --features cache --all-targets -- -D warnings
cargo test --locked --offline -p zerogravity-artifacts --no-default-features --features cache --test cache_transactions
cargo clippy --locked --offline -p zerogravity-artifacts --no-default-features --features cache,diagnostic-cache --all-targets -- -D warnings
cargo test --locked --offline -p zerogravity-artifacts --no-default-features --features cache,diagnostic-cache --all-targets

echo "Building the default CPU release CLI..."
cargo build --locked --offline --release --no-default-features -p zerogravity-cli --bin zerogravity --features cpu,model-qwen35,text
./target/release/zerogravity --version

printf 'Running offline installer integrity fixture...\n'
./scripts/test-install.sh ./target/release/zerogravity

if command -v shellcheck >/dev/null 2>&1; then
  echo "Linting shell entry points..."
  shellcheck install.sh scripts/build-release-artifacts.sh scripts/check.sh scripts/test-install.sh
fi

if command -v alejandra >/dev/null 2>&1; then
  echo "Checking Nix formatting..."
  alejandra --check flake.nix nix/*.nix nix/tests/*.nix
fi

if command -v python3 >/dev/null 2>&1; then
  echo "Checking CPU qualification scripts without downloading model weights..."
  python3 - <<'PYCOMPILE'
import ast
from pathlib import Path
for path in sorted(Path('scripts').glob('*.py')):
    ast.parse(path.read_text(), filename=str(path))
PYCOMPILE
  python3 scripts/compare-qwen-layers.py --self-test
  python3 scripts/cpu-qualification-gate.py --self-test
  python3 scripts/compare-attention-value.py --self-test
  python3 scripts/test-numeric-resume.py
  python3 scripts/test-native-gpu.py
fi

echo "Zerogravity CPU workspace quality gates passed (GPU backend builds and physical qualification require their separate explicit gates)."

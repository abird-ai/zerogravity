#!/usr/bin/env bash
# Source-only compile/test gate for GPU backend feature boundaries.
# This does NOT load a model or certify any physical GPU behavior.
set -euo pipefail

repo="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export PYTHONDONTWRITEBYTECODE=1

printf 'Checking HIP-only combined original qualification compilation...\n'
cargo check --locked --offline -p zerogravity --example hip_shared_models \
  --features shared-hip-qualification,zerogravity-gpu/hardware-abort-probe

printf 'Checking Vulkan-only complete/final qualification compilation...\n'
cargo check --locked --offline -p zerogravity \
  --example gemma4_vulkan_matrix --example gemma4_vulkan_final_owner \
  --features gemma4-vulkan-peer-qualification,zerogravity-gpu/hardware-abort-probe

printf 'Checking production CUDA native transport without qualification...\n'
cargo check --locked --offline -p zerogravity-gpu --no-default-features \
  --features cuda-native

printf 'Checking ordinary Qwen CUDA serving/refusal feature boundary...\n'
ordinary_tree="$(cargo tree --locked --offline -p zerogravity-cli --no-default-features \
  --features qwen35-cuda-serving --edges features --invert zerogravity-gpu)"
if grep -Eq 'zerogravity-gpu feature "(qualification|cuda-native-qualification)"' <<<"$ordinary_tree"; then
  printf '%s\n' "$ordinary_tree" >&2
  printf 'ordinary Qwen CUDA serving leaked GPU qualification\n' >&2
  exit 1
fi
cargo test --locked --offline -p zerogravity-cli --no-default-features \
  --features qwen35-cuda-serving --bin zerogravity qwen_cuda
cargo test --locked --offline -p zerogravity-cli --no-default-features \
  --features qwen35-cuda-serving --bin zerogravity ordinary_cpu_and_gguf_routes

printf 'Checking explicit finite Qwen CUDA qualification tests...\n'
qualified_tree="$(cargo tree --locked --offline -p zerogravity-cli --no-default-features \
  --features qwen35-cuda-qualification --edges features --invert zerogravity-gpu)"
grep -q 'zerogravity-gpu feature "qualification"' <<<"$qualified_tree"
grep -q 'zerogravity-gpu feature "cuda-native-qualification"' <<<"$qualified_tree"
cargo test --locked --offline -p zerogravity --no-default-features \
  --features qwen35-cuda-qualification --lib qwen35_cuda::templates::tests
cargo test --locked --offline -p zerogravity-cli --no-default-features \
  --features qwen35-cuda-qualification --bin zerogravity qwen35_cuda_route_tests

printf 'Checking combined Qwen CUDA qualification and serving tests...\n'
cargo test --locked --offline -p zerogravity-cli --no-default-features \
  --features qwen35-cuda-qualification,qwen35-cuda-serving --bin zerogravity \
  route_rejects_budget_fusion_and_context_before_source_device_or_native_load

printf 'Checking source-only native qualification regressions...\n'
python3 -B -m unittest discover -s tools/gemma4-hip-qualification -p test_row_traffic.py
python3 -B -m unittest discover -s tools/vulkan-gemma-qualification -p test_report_contract.py
printf 'GPU feature boundary compile gates passed; physical qualification NOT inferred.\n'

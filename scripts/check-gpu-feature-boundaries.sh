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
  --features shared-hip-qualification,zerogravity-gpu/hardware-abort-probe,state-checkpoint

printf 'Checking Vulkan-only complete/final qualification compilation...\n'
cargo check --locked --offline -p zerogravity \
  --example gemma4_vulkan_matrix --example gemma4_vulkan_final_owner \
  --features gemma4-vulkan-peer-qualification,zerogravity-gpu/hardware-abort-probe

printf 'Checking production CUDA native transport without qualification...\n'
cargo check --locked --offline -p zerogravity-gpu --no-default-features \
  --features cuda-native

printf 'Checking default production CLI feature boundary and compilation...\n'
default_tree="$(cargo tree --locked --offline -p zerogravity-cli --edges features --invert zerogravity-cli)"
for feature in all all-models all-backends cpu cuda hip vulkan model-qwen35 model-gemma4 acquisition serve tui web; do
  grep -Fq "zerogravity-cli feature \"$feature\"" <<<"$default_tree" || {
    printf 'Default CLI is missing production feature %s\n' "$feature" >&2
    exit 1
  }
done
default_gpu_tree="$(cargo tree --locked --offline -p zerogravity-cli --edges features --invert zerogravity-gpu)"
if grep -Eq 'zerogravity-(cli|gpu|model-qwen35|model-gemma4|artifacts|core) feature "(.*qualification.*|diagnostic-.*|hardware-abort-probe|cuda-flash-attn|cuda-flashinfer)"' <<<"$default_gpu_tree"; then
  printf 'Default CLI selected a diagnostic/qualification or opt-in attention feature\n' >&2
  exit 1
fi
cargo check --locked --offline -p zerogravity-cli

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

cargo test --locked --offline -p zerogravity --no-default-features \
  --features qwen35-cuda --lib \
  diagnostics::text_gpu::qwen35_cuda::tests::ordinary_serve_capability_refusal_is_configuration_not_inference -- --exact

cargo test --locked --offline -p zerogravity --no-default-features \
  --features qwen35-cuda --lib builtins::text::gpu::load::tests -- --test-threads=1

printf 'Checking explicit finite Qwen CUDA qualification tests...\n'
qualified_tree="$(cargo tree --locked --offline -p zerogravity-cli --no-default-features \
  --features qwen35-cuda-qualification --edges features --invert zerogravity-gpu)"
grep -q 'zerogravity-gpu feature "qualification"' <<<"$qualified_tree"
grep -q 'zerogravity-gpu feature "cuda-native-qualification"' <<<"$qualified_tree"
cargo test --locked --offline -p zerogravity --no-default-features \
  --features qwen35-cuda-qualification --lib qwen35_cuda::templates::tests
cargo test --locked --offline -p zerogravity --no-default-features \
  --features qwen35-cuda-qualification --lib diagnostics::qwen_route::qwen35_cuda_route_tests

printf 'Checking combined Qwen CUDA qualification and serving tests...\n'
cargo test --locked --offline -p zerogravity --no-default-features \
  --features qwen35-cuda-qualification --lib \
  diagnostics::text_gpu::qwen35_cuda::tests::route_rejects_budget_fusion_and_context_before_source_device_or_native_load -- --exact

cargo test --locked --offline -p zerogravity --no-default-features \
  --features qwen35-cuda-qualification --lib \
  diagnostics::text_gpu::fixed_prompt::eight_session_profile_tests -- --test-threads=1

printf 'Checking source-only native qualification regressions...\n'
python3 -B -m unittest discover -s tools/gemma4-hip-qualification -p test_row_traffic.py
python3 -B -m unittest discover -s tools/vulkan-gemma-qualification -p test_report_contract.py
printf 'GPU feature boundary compile gates passed; physical qualification NOT inferred.\n'

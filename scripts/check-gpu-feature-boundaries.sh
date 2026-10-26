#!/usr/bin/env bash
# Source-only/compile-only qualification for GPU backend feature boundaries.
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

printf 'Checking CUDA-only native grouped transport compilation...\n'
cargo check --locked --offline -p zerogravity-gpu --no-default-features \
  --features cuda-native

printf 'Checking source-only native qualification regressions...\n'
python3 -B -m unittest discover -s tools/gemma4-hip-qualification -p test_row_traffic.py
python3 -B -m unittest discover -s tools/vulkan-gemma-qualification -p test_report_contract.py
printf 'GPU feature boundary compile gates passed; physical qualification NOT inferred.\n'

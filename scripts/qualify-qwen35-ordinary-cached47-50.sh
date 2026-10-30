#!/usr/bin/env bash
# Exact two-process ordinary Qwen3.5 .8B cached47->50 qualification.
#
# The checked-in capture path defines the reference procedure. Rust loaders
# authenticate model/kernel/provider bytes; this wrapper only validates paths,
# bundle structure, process isolation and the phase-1 evidence digest handoff.
set -euo pipefail

repo="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo"

usage() {
  cat >&2 <<'USAGE'
Usage (capture + compare, or preflight):
  ZG_QWEN35_MODEL_DIR=/abs/approved-qwen35-0.8b \
  ZG_QWEN35_CUDA_KERNEL_DIR=/abs/current-or-legacy-native-bundle \
  ZG_QWEN35_ORDINARY_PARITY_EVIDENCE=/abs/new-evidence.bin \
  CUDA_PATH=/abs/cuda-include-view \
  LD_LIBRARY_PATH=.../nvrtc/lib:... \
  bash scripts/qualify-qwen35-ordinary-cached47-50.sh [--preflight-only]

Usage (resume compare only; no historical bundle required):
  ZG_QWEN35_MODEL_DIR=/abs/approved-qwen35-0.8b \
  ZG_QWEN35_ORDINARY_PARITY_EVIDENCE=/abs/existing-evidence.bin \
  ZG_QWEN35_ORDINARY_PARITY_EXPECTED_SHA256=<trusted phase-1 digest> \
  CUDA_PATH=/abs/cuda-include-view \
  LD_LIBRARY_PATH=.../nvrtc/lib:... \
  bash scripts/qualify-qwen35-ordinary-cached47-50.sh --compare-existing

Default run and --preflight-only require a fresh evidence path.
--preflight-only is deliberately structural/environment-only; source, kernel
and provider byte authentication remains owned by the Rust phase loaders.
--compare-existing requires the digest emitted by the original phase-1 process.
Legacy bundles are checked read-only during preflight and copied only for the
normal phase-1 run into an ephemeral private directory that is removed afterward.
USAGE
  exit 2
}

case $# in
  0) mode=run ;;
  1) mode=$1 ;;
  *) usage ;;
esac
case "$mode" in
  run|--preflight-only|--compare-existing) ;;
  *) usage ;;
esac

require_absolute() {
  local name=$1 value=$2
  if [[ -z "$value" || "$value" != /* ]]; then
    printf 'Required absolute %s is missing/invalid\n' "$name" >&2
    exit 2
  fi
  printf '%s\n' "$value"
}

model_dir="$(require_absolute ZG_QWEN35_MODEL_DIR "$(printenv ZG_QWEN35_MODEL_DIR || true)")"
evidence="$(require_absolute ZG_QWEN35_ORDINARY_PARITY_EVIDENCE "$(printenv ZG_QWEN35_ORDINARY_PARITY_EVIDENCE || true)")"
cuda_path="$(require_absolute CUDA_PATH "$(printenv CUDA_PATH || true)")"

kernel_dir=
if [[ "$mode" != "--compare-existing" ]]; then
  kernel_dir="$(require_absolute ZG_QWEN35_CUDA_KERNEL_DIR "$(printenv ZG_QWEN35_CUDA_KERNEL_DIR || true)")"
fi

[[ -d "$model_dir" ]] || { printf 'Model directory absent: %s\n' "$model_dir" >&2; exit 2; }
[[ -d "$cuda_path" && -f "$cuda_path/include/cuda_runtime.h" ]] || {
  printf 'CUDA_PATH does not contain include/cuda_runtime.h: %s\n' "$cuda_path" >&2
  exit 2
}
evidence_parent="$(dirname -- "$evidence")"
[[ -d "$evidence_parent" ]] || {
  printf 'Evidence parent directory absent: %s\n' "$evidence_parent" >&2
  exit 2
}

if [[ "$mode" == "--compare-existing" ]]; then
  [[ -f "$evidence" && ! -L "$evidence" ]] || {
    printf 'Compare-existing requires regular non-symlink evidence: %s\n' "$evidence" >&2
    exit 2
  }
else
  if [[ -e "$evidence" || -L "$evidence" ]]; then
    printf 'Refusing stale/existing parity evidence path: %s\n' "$evidence" >&2
    exit 2
  fi
fi

expected_evidence_sha="$(printenv ZG_QWEN35_ORDINARY_PARITY_EXPECTED_SHA256 || true)"
if [[ "$mode" == "--compare-existing" && ! "$expected_evidence_sha" =~ ^[0-9A-Fa-f]{64}$ ]]; then
  printf 'Compare-existing requires trusted ZG_QWEN35_ORDINARY_PARITY_EXPECTED_SHA256 (64 hex digits)\n' >&2
  exit 2
fi

command -v python3 >/dev/null 2>&1 || { printf 'python3 required\n' >&2; exit 2; }
command -v cargo >/dev/null 2>&1 || { printf 'cargo required\n' >&2; exit 2; }
command -v nvidia-smi >/dev/null 2>&1 || { printf 'nvidia-smi required\n' >&2; exit 2; }

ld_library_path="$(printenv LD_LIBRARY_PATH || true)"
python3 - "$ld_library_path" <<'PY'
import os
import sys

dirs = [p for p in sys.argv[1].split(":") if p]
if not any(
    os.path.exists(os.path.join(directory, name))
    for directory in dirs
    for name in ("libnvrtc.so", "libnvrtc.so.12")
):
    raise SystemExit("LD_LIBRARY_PATH does not expose libnvrtc.so/libnvrtc.so.12")
print("QWEN35_ORDINARY_PARITY_NVRTC_PREFLIGHT_PASS")
PY

bundle_kind=
if [[ "$mode" != "--compare-existing" ]]; then
  [[ -d "$kernel_dir" ]] || { printf 'Native kernel directory absent: %s\n' "$kernel_dir" >&2; exit 2; }
  if [[ -f "$kernel_dir/cuda-kernels.json" ]]; then
    PYTHONDONTWRITEBYTECODE=1 python3 -B "$repo/tools/qwen35-cuda-bundle/stage.py" --check-current "$kernel_dir"
    bundle_kind=current
  elif [[ -f "$kernel_dir/short-hf-native.json" ]]; then
    PYTHONDONTWRITEBYTECODE=1 python3 -B "$repo/tools/qwen35-cuda-bundle/stage.py" --check-legacy "$kernel_dir"
    bundle_kind=legacy
  else
    printf 'Native bundle has neither current nor supported legacy manifest: %s\n' "$kernel_dir" >&2
    exit 2
  fi
fi

require_idle_gpu() {
  local pids
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "$pids" ]]; then
    printf 'Refusing qualification while CUDA compute process(es) are active: %s\n' "$pids" >&2
    exit 2
  fi
}

require_idle_gpu
printf 'QWEN35_ORDINARY_PARITY_STRUCTURAL_PREFLIGHT_PASS evidence=%s mode=%s\n' "$evidence" "$mode"

if [[ "$mode" == "--preflight-only" ]]; then
  exit 0
fi

if ! printenv CARGO_BUILD_JOBS >/dev/null 2>&1; then export CARGO_BUILD_JOBS=8; fi
if ! printenv RAYON_NUM_THREADS >/dev/null 2>&1; then export RAYON_NUM_THREADS=8; fi
export PYTHONDONTWRITEBYTECODE=1

stage_parent=
cleanup_stage() {
  if [[ -n "$stage_parent" && -d "$stage_parent" && ! -L "$stage_parent" ]]; then
    rm -rf -- "$stage_parent"
  fi
}
trap cleanup_stage EXIT

if [[ "$mode" != "--compare-existing" ]]; then
  if [[ "$bundle_kind" == legacy ]]; then
    stage_parent="$(mktemp -d "$evidence_parent/.zg-qwen35-native.XXXXXX")"
    staged_kernel_dir="$stage_parent/current"
    python3 -B "$repo/tools/qwen35-cuda-bundle/stage.py"       --legacy-root "$kernel_dir"       --output "$staged_kernel_dir"
    kernel_dir="$staged_kernel_dir"
    export ZG_QWEN35_CUDA_KERNEL_DIR="$kernel_dir"
  fi

  printf '\n== phase 1/2: historical current-source reference -> evidence ==\n'
  set +e
  phase1_output=$(cargo test --locked --offline --release --no-default-features     -p zerogravity --features qwen35-cuda-qualification     --test qwen35_cuda     ordinary_cached47_to50_capture_reference_evidence     -- --ignored --exact --nocapture 2>&1)
  phase1_status=$?
  set -e
  printf '%s\n' "$phase1_output"
  if (( phase1_status != 0 )); then
    exit "$phase1_status"
  fi

  phase1_receipts=$(
    printf '%s\n' "$phase1_output" | grep -Eo       'QWEN35_ORDINARY_CUDA_CACHED47_50_REFERENCE_EVIDENCE_PASS sha256=[0-9a-f]{64}'       || true
  )
  receipt_count=$(printf '%s\n' "$phase1_receipts" | sed '/^$/d' | wc -l)
  expected_evidence_sha=$(printf '%s\n' "$phase1_receipts" | sed -n '1s/.*sha256=//p')
  if [[ "$receipt_count" != 1 || ! "$expected_evidence_sha" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'Phase 1 did not emit exactly one trusted evidence SHA256 receipt\n' >&2
    exit 2
  fi
  export ZG_QWEN35_ORDINARY_PARITY_EXPECTED_SHA256="$expected_evidence_sha"
  [[ -f "$evidence" && ! -L "$evidence" ]] || {
    printf 'Phase 1 did not create regular evidence: %s\n' "$evidence" >&2
    exit 2
  }

  cleanup_stage
  stage_parent=
  require_idle_gpu
else
  export ZG_QWEN35_ORDINARY_PARITY_EXPECTED_SHA256="$expected_evidence_sha"
  printf '\n== resume: existing captured reference evidence -> ordinary comparison ==\n'
fi

printf '\n== phase 2/2: fresh ordinary process -> exact evidence comparison ==\n'
cargo test --locked --offline --release --no-default-features   -p zerogravity --features qwen35-cuda-qualification   --test qwen35_cuda   ordinary_cached47_to50_compare_reference_evidence   -- --ignored --exact --nocapture

require_idle_gpu
printf '\nQWEN35_ORDINARY_CUDA_CACHED47_50_SERIAL_PASS evidence=%s expected_sha256=%s\n'   "$evidence" "$ZG_QWEN35_ORDINARY_PARITY_EXPECTED_SHA256"

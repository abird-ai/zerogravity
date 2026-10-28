#!/usr/bin/env bash
# Exact two-process ordinary Qwen3.5 .8B cached47->50 qualification.
#
# NO acquisition, provider substitution, repinning, permission changes or GPU
# recovery. Every source/provider path is explicit and authenticated before
# either release-CUDA process is started.
set -euo pipefail

repo="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo"

usage() {
  cat >&2 <<'USAGE'
Usage:
  ZG_QWEN35_MODEL_DIR=/abs/approved-qwen35-0.8b \
  ZG_QWEN35_CUDA_KERNEL_DIR=/abs/pinned-native-kernel-dir \
  ZG_QWEN35_ORDINARY_PARITY_EVIDENCE=/abs/new-evidence.bin \
  ZG_QWEN35_CUBLAS=/abs/.../libcublas.so.12 \
  ZG_QWEN35_CUBLAS_LT=/abs/.../libcublasLt.so.12 \
  CUDA_PATH=/abs/cuda-include-view \
  LD_LIBRARY_PATH=.../nvrtc/lib:... \
  bash scripts/qualify-qwen35-ordinary-cached47-50.sh [--preflight-only|--compare-existing]

Default run and --preflight-only require a fresh evidence path.
--compare-existing requires an already-published regular evidence file and runs
only the ordinary comparison phase after full artifact/GPU preflight.
USAGE
  exit 2
}

if (( $# == 0 )); then
  mode=run
elif (( $# == 1 )); then
  mode=$1
else
  usage
fi
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
kernel_dir="$(require_absolute ZG_QWEN35_CUDA_KERNEL_DIR "$(printenv ZG_QWEN35_CUDA_KERNEL_DIR || true)")"
evidence="$(require_absolute ZG_QWEN35_ORDINARY_PARITY_EVIDENCE "$(printenv ZG_QWEN35_ORDINARY_PARITY_EVIDENCE || true)")"
cublas="$(require_absolute ZG_QWEN35_CUBLAS "$(printenv ZG_QWEN35_CUBLAS || true)")"
cublas_lt="$(require_absolute ZG_QWEN35_CUBLAS_LT "$(printenv ZG_QWEN35_CUBLAS_LT || true)")"
cuda_path="$(require_absolute CUDA_PATH "$(printenv CUDA_PATH || true)")"

[[ -d "$model_dir" ]] || { printf 'Model directory absent: %s\n' "$model_dir" >&2; exit 2; }
[[ -d "$kernel_dir" ]] || { printf 'Native kernel directory absent: %s\n' "$kernel_dir" >&2; exit 2; }
[[ -d "$cuda_path" && -f "$cuda_path/include/cuda_runtime.h" ]] || {
  printf 'CUDA_PATH does not contain include/cuda_runtime.h: %s\n' "$cuda_path" >&2
  exit 2
}
[[ -d "$(dirname -- "$evidence")" ]] || {
  printf 'Evidence parent directory absent: %s\n' "$(dirname -- "$evidence")" >&2
  exit 2
}
if [[ "$mode" == "--compare-existing" ]]; then
  if [[ ! -f "$evidence" || -L "$evidence" ]]; then
    printf 'Compare-existing requires regular non-symlink evidence: %s\n' "$evidence" >&2
    exit 2
  fi
else
  if [[ -e "$evidence" || -L "$evidence" ]]; then
    printf 'Refusing stale/existing parity evidence path: %s\n' "$evidence" >&2
    exit 2
  fi
fi

command -v python3 >/dev/null 2>&1 || { printf 'python3 required\n' >&2; exit 2; }
command -v cargo >/dev/null 2>&1 || { printf 'cargo required\n' >&2; exit 2; }
command -v nvidia-smi >/dev/null 2>&1 || { printf 'nvidia-smi required\n' >&2; exit 2; }

python3 - "$model_dir" "$repo/crates/zerogravity/src/qwen35_cuda/source08.json" <<'PY'
import hashlib
import json
import os
import stat
import sys

root, manifest_path = sys.argv[1:]
with open(manifest_path, "rb") as f:
    manifest = json.load(f)
if manifest.get("repo") != "Qwen/Qwen3.5-0.8B":
    raise SystemExit("unexpected checked-in .8B source repository")
if manifest.get("revision") != "2fc06364715b967f1860aea9cf38778875588b17":
    raise SystemExit("unexpected checked-in .8B source revision")

for item in manifest["files"]:
    path = os.path.join(root, item["name"])
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise SystemExit(f"approved source member absent: {path}")
    if not stat.S_ISREG(st.st_mode):
        raise SystemExit(f"approved source member is not a regular file: {path}")
    if st.st_size != item["bytes"]:
        raise SystemExit(
            f"approved source extent differs: {path}: {st.st_size} != {item['bytes']}"
        )
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest()
    if actual != item["sha256"]:
        raise SystemExit(
            f"approved source SHA256 differs: {path}: {actual} != {item['sha256']}"
        )
print(
    "QWEN35_ORDINARY_PARITY_SOURCE_PREFLIGHT_PASS "
    f"repo={manifest['repo']} revision={manifest['revision']} files={len(manifest['files'])}"
)
PY

python3 - "$kernel_dir" "$repo/tools/qwen35-cuda-bundle/pins.json" <<'PY'
import hashlib
import json
import os
import stat
import sys

root, pins_path = sys.argv[1:]
with open(pins_path, "rb") as f:
    pins = json.load(f)
profile = pins["profiles"]["0.8b"]
checked = 0
for item in profile["resources"]:
    if item["section"] in ("cublas", "cublas_lt"):
        continue
    path = os.path.join(root, item["path"])
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise SystemExit(f"pinned native kernel absent: {path}")
    if not stat.S_ISREG(st.st_mode):
        raise SystemExit(f"pinned native kernel is not a regular file: {path}")
    if st.st_size != item["bytes"]:
        raise SystemExit(
            f"pinned native kernel extent differs: {path}: {st.st_size} != {item['bytes']}"
        )
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest()
    if actual != item["sha256"]:
        raise SystemExit(
            f"pinned native kernel SHA256 differs: {path}: {actual} != {item['sha256']}"
        )
    checked += 1
if checked == 0:
    raise SystemExit("0.8B native kernel pin census is empty")
print(f"QWEN35_ORDINARY_PARITY_KERNEL_PREFLIGHT_PASS files={checked}")
PY

python3 - "$cublas" "$cublas_lt" <<'PY'
import hashlib
import os
import stat
import sys

expected = [
    ("libcublas.so.12", 105_140_976,
     "d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048"),
    ("libcublasLt.so.12", 749_205_904,
     "96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541"),
]
for supplied, (soname, size, digest) in zip(sys.argv[1:], expected):
    if os.path.basename(supplied) != soname:
        raise SystemExit(f"provider SONAME path differs: {supplied} (expected {soname})")
    resolved = os.path.realpath(supplied)
    try:
        st = os.stat(resolved)
    except FileNotFoundError:
        raise SystemExit(f"pinned provider absent: {supplied}")
    if not stat.S_ISREG(st.st_mode) or st.st_size != size:
        raise SystemExit(
            f"pinned provider extent differs: {supplied}: {st.st_size} != {size}"
        )
    h = hashlib.sha256()
    with open(resolved, "rb", buffering=0) as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest()
    if actual != digest:
        raise SystemExit(
            f"pinned provider SHA256 differs: {supplied}: {actual} != {digest}"
        )
print("QWEN35_ORDINARY_PARITY_PROVIDER_PREFLIGHT_PASS providers=2")
PY

ld_library_path="$(printenv LD_LIBRARY_PATH || true)"
python3 - "$ld_library_path" <<'PY'
import os
import sys

dirs = [p for p in sys.argv[1].split(":") if p]
if not any(
    os.path.exists(os.path.join(d, name))
    for d in dirs
    for name in ("libnvrtc.so", "libnvrtc.so.12")
):
    raise SystemExit("LD_LIBRARY_PATH does not expose libnvrtc.so/libnvrtc.so.12")
print("QWEN35_ORDINARY_PARITY_NVRTC_PREFLIGHT_PASS")
PY

require_idle_gpu() {
  local pids
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d '[:space:]')"
  if [[ -n "$pids" ]]; then
    printf 'Refusing qualification while CUDA compute process(es) are active: %s\n' "$pids" >&2
    exit 2
  fi
}

require_idle_gpu
printf 'QWEN35_ORDINARY_PARITY_PREFLIGHT_PASS evidence=%s\n' "$evidence"

if [[ "$mode" == "--preflight-only" ]]; then
  exit 0
fi

if ! printenv CARGO_BUILD_JOBS >/dev/null 2>&1; then export CARGO_BUILD_JOBS=8; fi
if ! printenv RAYON_NUM_THREADS >/dev/null 2>&1; then export RAYON_NUM_THREADS=8; fi
export PYTHONDONTWRITEBYTECODE=1

if [[ "$mode" != "--compare-existing" ]]; then
  printf '\n== phase 1/2: historical current-source reference -> evidence ==\n'
  cargo test --locked --offline --release --no-default-features \
    -p zerogravity --features qwen35-cuda-qualification \
    --test qwen35_cuda \
    ordinary_cached47_to50_capture_reference_evidence \
    -- --ignored --exact --nocapture

  [[ -f "$evidence" && ! -L "$evidence" ]] || {
    printf 'Phase 1 did not create regular evidence: %s\n' "$evidence" >&2
    exit 2
  }
  require_idle_gpu
else
  printf '\n== resume: existing reference evidence -> ordinary comparison ==\n'
fi

printf '\n== phase 2/2: fresh ordinary process -> exact evidence comparison ==\n'
cargo test --locked --offline --release --no-default-features \
  -p zerogravity --features qwen35-cuda-qualification \
  --test qwen35_cuda \
  ordinary_cached47_to50_compare_reference_evidence \
  -- --ignored --exact --nocapture

require_idle_gpu
printf '\nQWEN35_ORDINARY_CUDA_CACHED47_50_SERIAL_PASS evidence=%s\n' "$evidence"

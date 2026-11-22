# Current performance baseline — 22 Nov 2026

This is the active performance baseline for ZeroGravity work after 2a595be.
It replaces older benchmark/evidence chronology as engineering authority. Git
history preserves prior experiments; do not carry old benchmark artifacts or
historical comparison numbers into new implementation decisions.

## Workload

All rows use the same pre-framed 47-token integer-continuation prompt and greedy
3-token completion. Offered populations are 1/4/8/16/32/64. Each retained
matrix uses one warmup plus two timed waves unless explicitly noted. Exact token
IDs are required before a row is considered usable.

## L5 — RTX 3060 Laptop 6 GiB — CUDA

### vLLM 0.33.0, optimized text-only

| Offered | 0.8B tok/s | 2B tok/s |
| ---: | ---: | ---: |
| 1 | 55.062 | 42.264 |
| 4 | 125.637 | 100.917 |
| 8 | 227.948 | 157.678 |
| 16 | 257.028 | 170.158 |
| 32 | 277.783 | 178.899 |
| 64 | 292.064 | 182.560 |

Both models return exact [248068, 198, 90700] at every width.

### ZeroGravity current CUDA path

| Offered | 0.8B tok/s | 2B tok/s |
| ---: | ---: | ---: |
| 1 | 39.621 | 34.802 |
| 4 | 119.613 | 83.038 |
| 8 | 133.676 | 96.099 |
| 16 | 137.410 | 95.255 |
| 32 | 135.918 | 96.091 |
| 64 | 136.385 | 95.962 |

Only C1 is currently accepted for cross-provider performance comparison.
C4+ is deterministic but fails lane invariance for identical requests on the
fused CUDA path, so those rows are diagnostic until that correctness bug is
fixed.

### ZeroGravity 27B streamed from SSD

Current smart startup chooses weight_storage=Streamed under the normal L5
device envelope (fresh available 6,066,667,520 B, configured provider
4,831,838,208 B, 512 MiB device headroom), with total startup admission
3,542,415,185 B. Physical execution is singleton/no fused cohort.

- expected IDs: [248068, 198, 760]
- warmup: 311.557 s, exact
- timed wave 1: 305.902 s, exact
- timed wave 2: 306.003 s, exact
- timed median: 305.952 s for 3 output tokens (~0.009805 output tok/s)
- shutdown: clean; RTX 3060 returned to 37 MiB / 0% busy

## X2 — Radeon 8060S gfx1151 — ROCm

### Matched 24 GiB envelope — 0.8B / 2B

vLLM uses gpu-memory-utilization=0.25, max-model-len=1024,
max-num-seqs=8, BF16, text-only optimized execution.

| Offered | vLLM 0.8B | ZG HIP 0.8B | vLLM 2B | ZG HIP 2B |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 51.966 | 12.406 | 36.054 | 6.169 |
| 4 | 135.358 | 12.048 | 96.233 | 6.897 |
| 8 | 219.612 | 12.139 | 140.847 | 6.771 |
| 16 | 245.256 | 11.881 | 150.059 | 6.752 |
| 32 | 271.828 | 11.848 | 156.828 | 6.735 |
| 64 | 284.356 | 11.797 | 159.801 | 6.721 |

All outputs are exact [248068, 198, 90700]. These rows are the clean
pre-optimization baseline; that path was physical batch width 1.

### First HIP decode-cohort checkpoint — 0.8B

The first model-neutral batching cut shares only BF16 projection work across
aligned cached-decode rows. Stateful operators retain each session's private
state/position and execute per row; prefill and full-logit host greedy remain
unchanged. With admitted cohort width 8 on X2:

| Offered | Baseline tok/s | Projection-cohort tok/s | Change |
| ---: | ---: | ---: | ---: |
| 4 | 12.048 | 13.270 | +10.14% |
| 8 | 12.139 | 13.061 | +7.60% |

Both cells are exact [248068, 198, 90700], replay-stable and lane-invariant.
The runtime reports real fused width-4/8 cohort publications, zero cohort
admission refusals, and clean retirement to 274,452,480 B VRAM / 0% busy /
no KFD owner. This proves the shared cohort path without claiming full-model
physical fusion.

### Matched 75 GiB envelope — 27B

vLLM uses gpu-memory-utilization=0.78125 (80,530,636,800 configured bytes),
max-model-len=1024, max-num-seqs=8, BF16 and text-only optimized execution.
ZeroGravity uses the already-qualified explicit 75 GiB resident HIP path.

| Offered | vLLM tok/s | ZG HIP tok/s |
| ---: | ---: | ---: |
| 1 | 3.803 | 0.540921 |
| 4 | 9.917 | 0.536684 |
| 8 | 13.491 | 0.534072 |
| 16 | 13.533 | 0.530675 |
| 32 | 13.492 | 0.528366 |
| 64 | 13.461 | 0.529800 |

Every fresh vLLM request returns exact [248068, 198, 760]; the resident
ZeroGravity ladder is likewise exact/replay-stable/lane-invariant. X2 retires
to 274,452,480 B VRAM, 0% busy and no KFD owner after the vLLM run.

## What this baseline says

1. The first HIP batching primitive is now proven: aligned decode rows can
   share projection work while retaining independent state/root authority. The
   modest C4/C8 gain shows the remaining stateful per-row work is substantial.
2. Batching is not the whole HIP gap. C1 is already substantially behind
   vLLM on 0.8B, 2B and especially 27B, so singleton kernel/execution efficiency
   remains independently important.
3. CUDA fused correctness comes before CUDA scaling work. C4+ performance
   numbers cannot become accepted comparisons until identical lanes agree.
4. 27B L5 streaming is a distinct memory-constrained mode. It is expected
   to be orders slower than resident X2 execution; use it to optimize SSD/source
   streaming mechanics, not to judge resident GPU arithmetic.
5. Start new optimization work from these rows only. Do not resurrect historical
   eager-vLLM, Podman, old provider, or superseded microbenchmark numbers.

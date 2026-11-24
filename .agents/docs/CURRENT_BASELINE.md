# Current performance baseline — 22 Nov 2026

This is the active performance baseline for ZeroGravity work after 2a595be.
It replaces older benchmark/evidence chronology as engineering authority. Git
history preserves prior experiments; do not carry old benchmark artifacts or
historical comparison numbers into new implementation decisions.

## 24 Nov — native projection opportunity (not model throughput)

Production exact HIP remains unchanged from 909cf1e6. The retained matched
C8 comparison is 0.8B 18.701 versus 219.612 tok/s and 2B 9.481 versus
140.847 tok/s (11.7x/14.9x). Current vLLM provider closure is unavailable,
so do not describe these as fresh paired measurements.

Physical X2 synthetic finite BF16 projection timing, ms per GEMM; scalar includes
its projection status reduction, custom WMMA fuses status/output, rocBLAS writes
packed BF16. No activation pack/widen or whole-model timing in this table.

| Weight rows x columns | Input rows | Scalar hoisted | Custom WMMA | Direct rocBLAS |
| --- | ---: | ---: | ---: | ---: |
| 3584x1024 | 1 | .044465 | .040280 | .040289 |
| 3584x1024 | 47 | .382650 | .235214 | .036096 |
| 3584x1024 | 128 | 1.060866 | .668301 | .062983 |
| 6144x1024 | 1 | .043819 | .063958 | .073326 |
| 6144x1024 | 47 | .658727 | .755345 | .072908 |
| 6144x1024 | 128 | 1.934770 | 1.205674 | .085698 |
| 1024x3584 | 1 | .039322 | .053106 | .075483 |
| 1024x3584 | 47 | .488879 | .072773 | .075573 |
| 1024x3584 | 128 | 1.445303 | .203698 | .072723 |
| 5120x2048 | 1 | .062353 | .079488 | .104410 |
| 5120x2048 | 47 | 1.222097 | .362536 | .106822 |
| 5120x2048 | 128 | 3.446642 | .851763 | .098360 |

Custom WMMA passes the non-symmetric 19x21x64 integer mapping and strided tail
test; normalized RMSE <= 1.56e-6 versus scalar in these random finite cells.
This is not bit parity, a quality grant, or a model permission. Direct rocBLAS
7.2.4 uses BF16 A/B/C/D and FP32 compute, all-one integer oracle, preallocated
64 MiB workspace. Its 128-row direct GEMM speedups are ~16.8–35.0x;
B1 is mixed. The separate owned-actor rocBLAS qualification at 9b9e4996
uses 6144x1024 by 128 rows with varied signed integer rows and checks every
output in four prepare/enqueue/complete/retire cycles. Device means per cycle:
.095802/.084230/.085970/.111742 ms, mean .094436 ms (~20.5x versus the
independent scalar measurement). All cycles retire per-model ownership;
only persistent context envelope remains, and the process returns VRAM to
259768320 B/no KFD owner. Separate direct handle/library lifecycle shows
~187 MB context-resident cache after destruction, stable across two cycles.
No extra stream/allocator or inferred memory reclamation.

The provider ceiling is not a whole-model speedup. The following native
execution measurement supersedes this standalone opportunity as the active
optimization baseline.

## 24 Nov — native HIP model execution

`312bbccd` wires owned rocBLAS BF16/FP32 GEMM through the original resident
source and cohort executor; `58a10d3c` uses whole aligned cohort GEMM instead
of the scalar 128-row tile cap. Explicit `HipRocBlasBf16MixedV1` numerical
identity, SDK `ModelLoadRequest::qwen_hip_native`, CLI
`zerogravity run-rocm MODEL --numerical native`. Native selection requires
qualified gfx1151 and admitted resident finite weights. Default exact
CPU-reference profile and all alternate primitives remain available. The
64 MiB vendor scratch is priced once in the same resident envelope; no
second scheduler, stream or state-root owner.

Actual X2 public SDK, resident source, 47+3 workload, one warmup/two timed
waves, 24 GiB small-model/68 GiB large-model envelopes. Projection-only
checkpoints (warm mean):

| Model / offered | Previous exact ms | Native projection ms | Native tok/s | Retained vLLM tok/s |
| --- | ---: | ---: | ---: | ---: |
| 0.8B C1 | 191.113 | 149.064 | 20.126 | 51.966 |
| 0.8B C8 | 1283.333 | 774.749 | 30.978 | 219.612 |
| 2B C1 | 370.777 | 305.329 | 9.825 | 36.054 |
| 2B C8 | 2531.391 | 1011.093 | 23.737 | 140.847 |
| 27B C1 | 5301.120 | 1516.150 | 1.979 | 3.803 |

C1 small-model rows are from the first native projection commit; whole-cohort
recheck C1 was noisy (200.234/464.148 ms), no new C1 improvement inferred.
C8 is the whole-cohort projection selection. All three return expected greedy
IDs, repeat/lane invariant, no failed quanta, and return to 259768320 B idle
VRAM/no KFD owner. Distinct aligned multichunk prompts match native singleton
outputs. This establishes actual serving execution, not CPU bitwise parity or
unrestricted model-quality equivalence. No fresh matched vLLM run.

Transient C8 0.8B profiling after whole-cohort GEMM: 47+1 prefill summed
observed CubeCL device time 702.277 ms / 44774 launches; 47+3 814.568 ms /
53742 launches. Does not include external BLAS timing; observer inflates wall.
Four GDN chunk product stages (Gram, prior, core, state) sum 205.493 ms;
triangular solve 92.199 ms. Other leading costs: gated activation 62.165 ms,
RMS scalar 54.133 ms, convolution 42.196 ms. Prioritize replacing complete
dense products and solve with mature native primitives and amortizing large
dispatch sequences; no scalar-only tuning campaign.

`6c31363d` replaces the four GDN chunk dense product stages with six native
FP32 strided-batched GEMMs and the original pointwise math in four epilogues.
Same actor, source leases, 64 MiB workspace, independent head state and
publication. `HipRocBlasF32ChunkV1` explicitly identifies the changed GDN
arithmetic within the opt-in native mixed profile. Exact source kernels and
the reusable WMMA alternative remain intact. The physical owned-provider
qualifier checks non-square NN/NT/TN layouts across three independent batches.

| Model / offered | Native projection ms | Native projection + GDN ms | Native tok/s |
| --- | ---: | ---: | ---: |
| 0.8B C1 | 149.064 | 130.158 | 23.049 |
| 0.8B C8 | 774.749 | 603.350 | 39.778 |
| 2B C1 | 305.329 | 285.147 | 10.521 |
| 2B C8 | 1011.093 | 825.072 | 29.088 |
| 27B C1 | 1516.150 | 1362.961 | 2.201 |

Retained same-workload checkpoint comparisons, not fresh paired A/B.
Expected IDs, replay/lane invariance, zero failed quanta, and distinct
aligned multichunk native-vs-native singleton comparison all pass. All
three models retire native ownership; final physical observation returns to
259768320 B idle VRAM, zero busy and no KFD owner. GPU feature boundaries
and CPU Clippy pass; the full gate stops at the pre-existing CPU admission
assertion in cache_prepare.rs:142.
These are the new native performance cells; exact-reference performance
remains separate.

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

### Current HIP decode cohort — 0.8B

Aligned cached-decode rows now share BF16 projections, row-wise
RMS/residual-RMS/sigmoid/multiply/SILU-multiply, aligned Q/K normalization and
rotary preparation, plus the 18 GDN layers' Conv4 state update. Conv4 binds each
lane's original private prior/candidate buffers directly in one bounded 2/4/8-row
kernel; no state staging or publication authority was added. Recurrent GDN, KV
append and attention state remain per-session. Prefill and full-logit host greedy
remain unchanged.

| Offered | Clean baseline tok/s | Current tok/s | Change |
| ---: | ---: | ---: | ---: |
| 4 | 12.048 | 13.297 | +10.37% |
| 8 | 12.139 | 13.351 | +9.98% |

Both cells are exact [248068, 198, 90700], replay-stable and lane-invariant.
The runtime reports real fused width-4/8 cohort publications, zero cohort
admission refusals, and clean retirement to 274,452,480 B VRAM / 0% busy /
no KFD owner. Current physical forward dispatches are 32,086 per C4 wave and
61,278 per C8 wave. The stateful Conv4 cut improves C4 about 1.87% over the prior
checkpoint and leaves C8 effectively flat; the much larger recurrent GDN state
path is the next useful batching target.

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

### X2 recovery and native admission — 24 Nov

All three resident models run again on the installed CLR7.2.4/glibc2.44
closure (runtime/driver70253211,RTC9.0); see CURRENT_HANDOFF for the temporary
process-local loader recipe. The locked development shell and pinned vLLM
provider are not currently available on X2. The following uses the original
SDK population benchmark, optimized non-LTO/codegen16, one warmup + two
timed waves. Native catalogue probe `11a44d60` measured the original envelopes;
selected policy `38f20ca3` removes the probe and rechecks smaller envelopes.

| Model | Pre-hoist C1, original envelope (GiB) | Output tok/s | Retained matched vLLM tok/s | Native modules | Native prepare delta (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 24 | 12.508 | 51.966 | 93 | 168.116 |
| 2B | 24 | 6.977 | 36.054 | 90 | 168.059 |
| 27B | 75 | 0.5392 | 3.803 | 95 | 168.153 |

The four-cycle native lifecycle observes a maximum ~194MB incremental VRAM;
complete native sets contain 2.37–2.49MB of load segments. Qualified integrated
gfx1151 now has a 1GiB opaque allowance instead of 8GiB, plus the unchanged
1GiB upper code bound. Actual reduced-envelope validation:

| Model | Total admitted GiB | Warm mean (ms) | Output tok/s | Native device charge (B) | Dispatches/request |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2B | 14 | 434.714 | 6.901 | 1,076,111,858 | 9,673 |
| 27B | 68 | 5,556.005 | 0.5400 | 1,076,230,179 | 36,741 |

Actual VRAM peaks remain ~4.54GB and ~56.43GB respectively, unchanged within
measurement variation: this is admission headroom, not a physical allocation
or throughput reduction. All cells return exact expected IDs, replay-stable,
lane-invariant, with zero failed quanta and original 259,768,320B idle VRAM/no
KFD owner after retirement.

Pre-hoist small-model cohort recheck at the original matched 24GiB envelope,
`38f20ca3`, configured width8, actual fused decode cohorts (two publications
per wave), no failed quanta:

| Offered | Pre-hoist ZG 0.8B | Retained vLLM 0.8B | Pre-hoist ZG 2B | Retained vLLM 2B |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 13.743 | 135.358 | 7.842 | 96.233 |
| 8 | 13.482 | 219.612 | 7.641 | 140.847 |

Both models pass all identical lanes, exact greedy IDs and replay, and
retire to original idle VRAM/KFD. Dispatches/wave31,794 at C4 and60,790 at
C8 for both.

### HIP hoisted projection — qualified gfx1151

Physical localization (`533c72c`, temporary observer removed) with 47+1
versus 47+16 for C1 and 47+1 versus 47+3 for C8: each side twice in ABBA
order, exact greedy prefix/replay. Summed serialized device events, not
end-to-end latency. Cached decode is mean difference divided by extra outputs.

| Model / offered | Prefill + first output (ms) | Projection GEMM (ms) | Cached decode step (ms) | Projection GEMM (ms) |
| --- | ---: | ---: | ---: | ---: |
| 0.8B C1 | 199.62 | 85.44 | 23.65 | 10.21 |
| 0.8B C8 | 1,589.56 | 680.35 | 87.68 | 31.04 |
| 2B C1 | 332.48 | 214.79 | 41.14 | 25.98 |

C8 prefill remains eight independent invocations (51,352 launches versus
6,419 at C1); cached decode cohorts cut the eight-client step to4,719
launches instead of8×1,627. GDN triangular prefill is the next individual
hotspot (32.83ms/C1,262.65ms/C8,33.01ms/2B C1). These are pre-selection
localization results.

Same-binary real-model ABBA `c1cd04f3`, two fresh loads per variant/model,
one warmup + two timed waves at each C1/C4/C8, same original47+3 prompt,
24GiB total, active8/configured width8. `99eb7990` selects the existing
hoisted traversal for both qualified integrated gfx1150/gfx1151 resident
policies (other paths unchanged).

| Model / offered | Canonical mean (ms) | Hoisted mean (ms) | Hoisted ZG tok/s | Retained matched vLLM tok/s |
| --- | ---: | ---: | ---: | ---: |
| 0.8B C1 | 238.514 | 211.895 | 14.158 | 51.966 |
| 0.8B C4 | 872.431 | 765.040 | 15.685 | 135.358 |
| 0.8B C8 | 1,787.195 | 1,540.048 | 15.584 | 219.612 |
| 2B C1 | 420.655 | 388.041 | 7.731 | 36.054 |
| 2B C4 | 1,533.845 | 1,389.398 | 8.637 | 96.233 |
| 2B C8 | 3,153.334 | 2,848.838 | 8.424 | 140.847 |

All 48 timed waves plus warmups exact, lane-invariant and replay-stable;
unchanged dispatches9,673/31,794/60,790 per C1/4/8 wave, zero failed quanta,
clean original VRAM/KFD retirement. Every candidate timed wave beats the
fastest same-cell baseline. Warm mean latency falls7.75–13.83%.
Final source `99eb7990` (temporary switch/runner removed) physically passes
the unchanged original C1 benchmark:0.8B24GiB mean211.004ms,14.218tok/s,
and27B68GiB mean5,376.029ms,0.5580tok/s versus previous5,556.005ms.
The 27B single recheck is a correctness/residency canary, not a paired
performance claim. Exact expected IDs/replay, unchanged dispatches9,673/36,741,
zero failed quanta and original idle VRAM/no KFD owner.
No new vLLM execution or fresh paired comparison; at that checkpoint C8
gaps were~14.1x/~16.7x.

### Previous gfx1151 HIP checkpoint — guarded GDN triangular

`393d0f56` adds separately rounded guarded FP32 multiply-then-add with
the original widened fallback. Exact zero products preserve the stored
addend and IEEE zero sign. No change to the sequential 64-row solve,
four-way partial accumulation/reduction, status or scratch ownership.
Direct physical A/B (16 heads, six paired 16-launch windows per cell)
47 active rows padded to64:26.365→9.737ms (-63.1%);
64 active rows:17.217→9.162ms (-46.8%). Both complete output matrices match
finite bits and status, including padded zeros, signed zero/subnormal,
overflow/nonfinite cases and NaN classification. Existing scalar-body
regression passes both denormal-preserving and simulated flush modes.

Same-binary ABBA `cd86a965`, existing47+3 population benchmark,24GiB,
active8/configured cohort8, two fresh loads and four timed waves per
variant/cell. `4a5961da` selects guarded triangular only on qualified gfx1151;
other paths including the original reference qualifier retain canonical
arithmetic. Cold selection is immutable and shared by catalogue and launch.

| Model / offered | Previous mean (ms) | Guarded mean (ms) | Current ZG tok/s | Retained matched vLLM tok/s |
| --- | ---: | ---: | ---: | ---: |
| 0.8B C1 | 211.801 | 191.113 | 15.697 | 51.966 |
| 0.8B C4 | 765.845 | 681.951 | 17.597 | 135.358 |
| 0.8B C8 | 1,545.560 | 1,378.619 | 17.409 | 219.612 |
| 2B C1 | 389.243 | 370.777 | 8.091 | 36.054 |
| 2B C4 | 1,386.046 | 1,302.572 | 9.213 | 96.233 |
| 2B C8 | 2,841.161 | 2,677.231 | 8.964 | 140.847 |

All six means improve4.74–10.95%, exact expected IDs, replay, lane
invariance, unchanged9,673/31,794/60,790 dispatches, zero failed quanta
and original retirement. Final `4a5961da` unchanged C1 benchmark passes
0.8B24GiB warm mean190.322ms,15.763tok/s, and27B68GiB warm mean5,301.120ms,
0.5659tok/s (correctness/residency canary, not paired performance evidence).
Expected IDs, replay, unchanged9,673/36,741 dispatches/request, zero failed
quanta, original idle VRAM259,768,320B and no KFD owners.
No new vLLM execution or fresh paired comparison.

### Current gfx1151 HIP path — aligned shared prefill

`2ed9c10f` extends the existing HIP cohort traversal and completion owner to
equal-frontier/equal-length 2..64-token prefill chunks with the same output
policy. Configured cohort cap 4/8 admits widths 2/4/8 with private state roots.
One-token prefill tails remain singleton. Projection input rows flatten
across sessions, terminal heads gather their last rows, and activation
slots/scratch grow only for the admitted cohort extent. Capacity refusal
keeps singleton fallback. `282a3437` / `2eee25fa` share stateless RMS,
sigmoid, multiply and SILU-multiply and write single-matrix projections
directly when lease roles permit; positional and stateful prefill work
remains per session.

The first untiled shared build produced exact outputs but mixed wall time:
C4/C8 mean0.8B659.131/1363.181ms,2B1227.278/2887.715ms.
`fd914f63` caps resident projection token tiles at128 (streamed operands
retain one whole-cohort traversal), reducing temporary projection extents
and recovering C8 locality. Existing full-row primitives remain reusable.

X2 actual public SDK, same canonical47+3 workload,24GiB,active8/cohort8,
one warmup plus two timed waves per cell. Previous means are the retained
guarded-triangular checkpoint, not fresh paired measurements:

| Model / offered | Previous mean (ms) | Current mean (ms) | Current ZG tok/s | Retained matched vLLM tok/s |
| --- | ---: | ---: | ---: | ---: |
| 0.8B C4 | 681.951 | 657.981 | 18.238 | 135.358 |
| 0.8B C8 | 1378.619 | 1283.333 | 18.701 | 219.612 |
| 2B C4 | 1302.572 | 1249.184 | 9.606 | 96.233 |
| 2B C8 | 2677.231 | 2531.391 | 9.481 | 140.847 |

All expected IDs/replay/lane invariance exact; three actual fused cohorts
per wave (prefill + two decode),29,564/55,695 dispatches versus
31,794/60,790 previously, zero failed quanta. Mean reductions3.5–6.9%.
C1 remains the previous checkpoint; no fresh vLLM or C1 run here.
`47b9f6f6` differential repeated on selected tiled build: distinct
0.8B prompts,64 state-only +17 terminal tokens, discarded terminal cohort
then exact replay,22,922,240 FP32 words bit-exact against singleton
state/logits. `74619a0b` public four-session 81-token distinct-prompt
regression observes actual state-only fusion and singleton-equivalent
output. All physical runs retire to259,768,320B idle VRAM, no KFD owners.
Final feature-boundary/CPU Clippy gates pass; full isolated check.sh
stops at the existing CPU cache-admission assertion.
Current C8 gaps versus retained vLLM remain~11.7x/~14.9x.
Next localize the selected shared-prefill path, especially remaining
per-session stateful work and projection throughput; do not infer a
speedup from fewer launches alone.

## A1 — Radeon 890M gfx1150 — local resident HIP comparison, 24 Nov

Separate local optimization cell, not a matched vLLM baseline: Qwen3.5-0.8B,
C1, 47-token prompt +16 output, context1024,14GiB admission. Same-binary ABBA
at `224085b1`, two fresh loads/variant, one warmup + four timed requests/load.

| Traversal | Warm mean | Timed range |
| --- | ---: | ---: |
| Canonical full-K64 | 924.5 ms | 885.3–997.5 ms |
| Hoisted packed indices | 830.2 ms | 806.5–849.6 ms |

10.2% lower mean request latency; exact greedy IDs/replay, 30,824 dispatches
unchanged, no failed quanta, clean source/session retirement and no added KFD
owners. Hoisted traversal selected for the qualified gfx1150 resident HIP path.
The retained matched vLLM tables above are unchanged.

## What this baseline says

1. HIP decode cohorts now share projections, row-wise activation/QK/rotary work
   and the 18 GDN layers' Conv4 private-state update while retaining independent
   StateRoot/publication authority. Conv4 is exact but only modestly moves wall time;
   X2 now also shares aligned prefill projections and stateless operators,
   using 128-token resident projection tiles. Stateful prefill work remains
   private per session. Measure remaining shared-path costs before expanding
   the stateful consumers.
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

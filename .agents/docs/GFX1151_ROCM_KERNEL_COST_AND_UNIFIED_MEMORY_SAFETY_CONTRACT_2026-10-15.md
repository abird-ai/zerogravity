# Zerogravity — gfx1151 ROCm Kernel Specialization, MoE Scheduling and Unified-Memory Safety Contract

**2026-10-15 • Hardware-specific DESIGN REFINEMENT under v4.3 generic DeviceBackend/KIR, not a separate AMD inference engine.**
**Source:** published gfx1151 port notes, Qwen serving recipes, kernel notes, first-image autotune observations, and MiMo coherent-memory experience.

## 0. Governing invariants

Device ISA or APUs never change ModelRecipe/source semantics, accepted StateRoot or source tensor identity. Specialized kernels are guarded candidates in the shared ExecutableBundle, with CPU/source oracle, numerical policy and physical event/lease proof. Missing ISA/shape/numeric capability must select a **qualified** generic candidate or refuse honestly. All flags below from the prior-art study are study references, NOT Zerogravity flags or universal defaults.

## 1. Target hardware and capabilities

AMD Strix Halo Ryzen AI Max+ 395 / Radeon 8060S iGPU is gfx1151 RDNA3.5, wave32 kernel specialization, **40 compute units** and unified host-device LPDDR memory. The prior 20-CU statement was incorrect; [AMD’s hardware specification](https://rocm.docs.amd.com/en/docs-10.0.0/reference/gpu-specs.html) lists 40 CUs. Hardware also supports wave64; each kernel must declare its actual wave mode. The prior-art physical source tested gfx1151 with ROCm7 and specialized WMMA. Nvidia CUDA, gfx12 RDNA4, CDNA matrix cores and non-AMD NPU have different ISA/warp/register and memory contracts.

A fully qualified device receipt must identify actual PCI/driver/firmware/compiler, wave size, memory topology, supported WMMA/MFMA/DOT/SAD instructions, layout and numerical reduction capabilities, LDS bank geometry, VGPR allocation/occupancy, synchronization and physical event behavior. A CUDA-like mnemonic compiled to HIP is not evidence of equivalent instruction support or source output.

### Kernel candidate dimensions
- task regime: decode M1, verify M2–16/32, prefill M64..8K, short vision encoder and long AV;
- source tensor codebook/bit width, scale and value dtype, fused activation quant;
- logical dimensions M/N/K vs physical padded shapes, alignment and block pack;
- wave/warp/occupancy, LDS tile, subgroup precision/reduction and scratch;
- device ISA+driver+compiler and runtime graph capture/eager mode;
- co-resident expert rows, output format, source state snapshots and physical buffer lease;
- verified source equivalence/quality and measured whole-workload performance.

## 2. Wave32 WMMA fragment layout and LDS bank conflict

The prior-art source ported an RDNA3.5 WMMA GEMV for EXL3 decode, with source-oracle fragment layout, rather than using the fallback reconstruct+FP16 GEMM. A classic CUDA cp.async sequence is not available as-is on this target, so their INT8 activation GEMV uses register-staged B loads and AMD v_dot4_i32_iu8 path.

**Important measured optimization:** its half2/shared LDS staging array had physical innermost extent 4, causing repeated 32-bank LDS collisions. Padding that physical stride to 5 (coprime with 32) reduced conflicts, with reported ~10% overall mean decode gain at a particular pipeline stage. This is **a layout-tuning hypothesis scoped to hardware and kernel**, not a new semantic operator or unconditional global padding rule.

Correct specialization:
1. emit source logical tensor layout independent of LDS physical swizzle/padding;
2. choose an ISA-target-specific fragment map with verified lane→matrix element and grouped row indexing;
3. treat packed shared-memory stride as a compiler physical parameter, candidate 4/5/safe alternatives;
4. validate barriers, wave count and no out-of-bounds for ALL callers using the shared scratch, not only MoE (the prior-art source warns a 4-warp scratch used by 16-warp GEMV silently produced NaN logits);
5. run source tensor output golden vectors including ragged M/N, smallest and largest codebook K, zero/nonfinite values, padded rows and multiple model layers;
6. measure LDS bank conflicts, VGPR/spills, occupancy/byte throughput and **whole model decode with MTP**, not only GPU instruction counts.

A broken scratch layout may keep aggregate PPL apparently stable while corrupting individual logits. Must include full tensor/source validation and state/greedy prefix, not solely perplexity.

## 3. Dense/GDN/MLA/vision kernels: source-first closed regions

Hardware candidate list, promoted only after legal semantic guard:
- decode quant GEMV directly from packed low-bit codebook with WMMA wave32;
- prefill dense grouped GEMM and matrix tiles;
- **skinny GEMM** for GDN input projection b/a or other unusually narrow matrices where hipBLASLt split-K overhead dominates;
- fused GDN/KDA short convolution→gate/recurrence update, preserving causal state, numerical order and rollback;
- MLA/DSA compressed latent Q/K selection and sparse attention masked/position-correct;
- mHC MixNorm/residual chain fused only when stream stack/result numeric equivalence demonstrated;
- grouped MoE router, expert compact assignment, packed GEMV/GEMM, post-expert reduction and shared expert;
- source E8M0 scale reader and expert FP16 intermediate range guard (MiMo-specific);
- optional physical INT8 activation GEMV, with v_dot4 variant only where legal and measured;
- architecture-specific bitcodebook SAD trellis decode, no output bit reordering;
- image/video vision patch merge convolution lowered to logically identical GEMM if supported.

**Do not** automatically enable all fused kernels. The prior-art source reports CPU offloaded routed experts on one gfx1151 UMA test **slowed down** decode relative to GPU; A-ring prefetched register/GEMV variants and graph replay also provided null/negative improvements in given cases. Keep one simple source-exact baseline and only promote repeated whole-task wins.

## 4. MoE row-packing, verification and multi-session scheduling

A row count is a *physical kernel dimension* independent of logical token count, because each draft block generates multiple verifier rows. For B concurrent sequences with per-session proposed depths d_i:

R_total = sum_i (1 + d_i).

Actual MoE expert work depends on:
- unique expert IDs across all R rows and models, routes per row and stable compact worklist;
- expert-row histogram, uneven hot/cold placement, shared-expert gates and reduction order;
- logical rows vs padded per-expert tiles, source quant/codebook and kernel specific physical group size G;
- expert weight read bytes, warm/OS/SSD residency and host staging, physical memory bandwidth;
- verification acceptance by prompt/content class, target state branch COW and draft cost;
- source-compatible numerator of **committed** accepted tokens rather than proposed drafts;
- p95 tail and wait fairness, priority and deadline, guard memory and compiled variant cost.

Where a physical kernel rounds work to G-row chunks, padded cost resembles G*ceil(R/G) only as a rough bound: real grouped expert work can have many more per-expert partial tiles. G is recorded with a **KernelCandidate** and device profile, not a constant in TaskDriver/SIR.

### Prior-art measured counterexample (source, not Zerogravity result)

On a particular gfx1151 Qwen EXL3 pack the authors measured a single-session verify forward T_ms approximately 32.4 + 9.04R. At speculative depth 2, five concurrent requests produce 15 verifier rows, near one 16-row chunk. Six produce 18 rows, spilling into a second mostly-empty chunk and reported a strong total-throughput decline. Larger verifier width also reads *more distinct routed experts* and can reduce acceptance; a 16-row speculative tree was counterproductive under measured marginal row costs.

The source notes ~47 tok/s single session on a favorable prompt, versus ~74–75 tok/s aggregate for a realistic deeper-queued B5 batch in one configuration. **These are different objectives**. User-facing interactive latency and aggregate throughput need separate settings and reported metrics. The general scheduler should select one of a small bounded set of source-qualified batch/depth/shape candidates under active state and P95 fairness, not run graph search or retune every token.

### Shape-aware batch cost (conceptual)
- observed GPU launch + per-step fixed cost;
- expert unique-row bytes + packed codebook work;
- quant GEMV bandwidth and LDS spills;
- GPU state update/verification per row and selected attention indexer;
- queue delay and input/output physical DMA, speculative correction and held resources;
- valid accepted tokens forecast and fallback hysteresis.

An unexpected batch trough should cause candidate reselection after measured evidence and cooldown, not a silent change of quant/attention or unlimited queue depth.

## 5. Unified-memory admission and Linux page cache

On coherent-memory systems, GPU allocation, CPU heap, OS page cache, host pinned staging and GPU buffers compete for a finite physical pool. **Memory mapped/addressable GPU-visible bytes, driver free GTT, MemFree and MemAvailable are different quantities.** A large model can fit logically yet exhaust actual RAM/pinned pages or stall the driver during loading.

Prior-art MiMo notes report a near-fatal case on GB10 where buffered pread of about 86GiB model shards left an approximately equal-size clean file page cache while the GPU also held weights. MemFree-like driver reported free memory did not capture reclaimable cache properly, and the system entered severe reclaim thrash. Their opt-in external guard checked MemAvailable, reclaimed only known shard page cache via fadvise(DONTNEED), and killed the isolated serving process if a configured physical headroom floor was crossed. **Do not copy that kill policy as a default Zerogravity runtime behavior**; it is a valuable adversarial failure mode and user-supervised deployment consideration.

### Authoritative resource design adjustments
1. Keep **one PhysicalResourceArbiter** for physical card and model/session envelopes, but admission also checks appropriate **host/OS-safety** envelopes on UMA. Do not make OS file cache a second GPU owner.
2. Distinguish mandatory/evictable resources: packed target weights, draft, vision, recurrent/KV state, page-cache mapped shards, clean file cache, pinned DMA, local scratch, OS health reserve and unrelated system workloads.
3. At startup check **platform-aware** headroom: Linux MemAvailable and cgroup memory.current/memory.max/memory.stat where available, driver committed/resident and known pinned pages, memory pressure indicators/PSI. Driver-reported GTT maximum is NOT guaranteed free physical bytes. Other OSs use native APIs.
4. Estimate and reserve source+codec+model+draft+state+transfer+read-cache footprint before load. During multi-shard model loading, budget peak staging and likely reclaimable clean page-cache duplicates.
5. Use bounded streaming read or mmap with source-file-descriptor lifetime, version/hash validation and controlled prefetch; do not eagerly buffer 100GiB source *again* on CPU by accident. A host page-cache hint is an optimization, not permission to discard dirty files or another process's cache.
6. Optional, **user-authorized** clean-shard DONTNEED on only the manifest-verified model files after their bytes have been safely copied/retired; never blindly evict global page cache or require root.
7. On predicted headroom miss, reject/defers **before** model GPU allocations; on observed pressure suspend model loads, release **safe idle** caches and return typed admission error. No hidden SIGKILL or OOM-based throttling inside EngineCore.
8. A separate supervised daemon policy might terminate a disposable isolated model worker on impending system hang **only if explicitly configured** and after it preserves no false StateRoot receipt. This is a deployment choice, not a library inference default.
9. Keep physical GPU device buffer quarantine on unknown submitted work, and do not treat OS page cache freeing as proof GPU command quiescence.
10. Telemetry must report physical OS headroom, driver commit, page cache bytes, old memory pressure, true live StateRoot leases and per-session rings; never conflate logical source budget release with driver pool eviction.

### Atomic multi-model loading and cache safety
- Model aliases and exact source identity can share immutable weights; per-session StateRoot and mutable drafts/KV are independent.
- Concurrent loads use bounded per-source singleflight to avoid two simultaneous copies and cache thundering herd.
- A ModelCatalog can contain dozens of registered large models while only a few fit resident; TTL/pin/revoke and warm/cold resource receipts are distinct.
- No auto-up/downquantize to fit in 128GB; source/derived artifact choice is an explicit model binding.
- Optional systemd deployment controls ResourceMaxMemory/MemoryHigh and capabilities, but can cause driver/process pressure; verify expected behavior with actual Linux GPU.

## 6. Cold-start and first-image specialization

The prior-art source found GLM5.3 first vision image/resolution sometimes spent **many seconds/minutes in MIOpen convolution solver and dense GEMM autotune search**, triggered anew by subtly different input shape, while steady-state math was fast. Their source has a verified patch-merge convolution with kernel=stride=2 that can become a flattened GEMM with identical contraction and no costly first-use convolution tuning.

Recommended cost/tuning lifecycle:
- **install/compile:** emit known-good reference kernels and bounded guarded variants for shape bands/registered encoders;
- **model load:** compile/prime only finite representative shapes if budget permits, allow eager direct GEMM patch merge for legally verified shapes;
- **request:** choose guarded prequalified kernel cheaply; unfamiliar shapes execute generic known-good path or bounded safe limited compile, **never indefinite live solver search**;
- **offline/autotune:** evaluate alternatives asynchronously in a low-priority, resource-capped window, persist evidence cache keyed by model+shape+device/compiler/numeric mode;
- **future request:** reuse persisted validated cache; event/current driver version and shape must match.

Vision-source equivalence requires exact convolution stride, padding, dilation, groups, channel and image patch token order. A shape-only transformation is invalid if its source convolution has overlapping windows or nonlinear postprocessing. The generic slow path remains available and truthful.

No startup ETA or prefill speed should be invented without measured load history. Progress events may report deterministic counts of uploaded tensor bytes, backend compile stages and applied cache state.

## 7. Qualifying negative results and overhead ceilings

Take seriously the documented **non-wins**:
- CPU MoE expert offload on this UMA target decreased decode speed because shared LPDDR contention/transfer overhead dominated any CPU compute gain.
- Some INT8 activation GEMV paths performed worse than FP16 WMMA; activation encode and quant scale overhead matters.
- Graph replay was neutral at tested MTP workload; saved host launches were a small fraction of total wall time.
- Some candidate prefetch/register rings increased VGPR pressure and did not help.
- Uniform lower bitrate (2.05) performed much worse than 3.05 due to trellis K2 kernel tile throughput.
- Static draft depth 2 beat dynamic on one favorable prompt, but multi-prompt averages preferred default dynamic policy; never promote one cherry-picked prompt.

Experiments therefore require:
- repeated ABBA/BAAB paired runs on warmed and cold model, temperature/power clock stabilized, narrow confidence intervals and a control without change;
- source model/derived quant exactly pinned and same prompt/task/sampling/accepted token policy;
- kernel and end-to-end instrumentation separated (not every request measured under expensive profiler);
- no memory pressure/driver hang tests on an interactive system without explicit isolation and operator approval;
- measure target-only, MTP, DFlash, 1–8 sequences, expert entropy, long context and image shape changes;
- negative/neutral candidates remain in evidence database so later agents don't rediscover them.

## 8. Acceptance gates for P01/P07/P10/P12/P16/P20/P22

| ID | Scenario | Mandatory proof |
|---|---|---|
| GFX-01 | gfx1151 wave32 WMMA fragment/lane layout | scalar source/CPU oracle, full tile bounds and original model output |
| GFX-02 | LDS stride 4 vs 5, full shared-warp users | no bank-related regression, valid 16-warp caller, no NaN logits |
| GFX-03 | direct EXL3 K2/K3/K4 packed GEMV at M1,3,4 and 8/16/32 | measured kernel/full-model speed and exact derived-quant oracle |
| GFX-04 | int8 DOT4 activation against FP16 default | codebook/activation scale, source numerics, full wall benefit or reject |
| GFX-05 | GDN skinny GEMM + short conv fuse, QSA/DSA source attention | recurrence/sparse indexer/StateRoot source invariants |
| GFX-06 | mHC/gated residual fused region and source intermediate | source full logits, overflow/range and aligned stream state |
| GFX-07 | MoE batch five→six under 16-row candidate | measure shape cliff, no hard-coded G=16 in SIR |
| GFX-08 | B and draft width with different expert uniqueness | dynamic candidate selects measured best, no P95 starvation |
| GFX-09 | model + draft + 256K KV/SWA ring + 100GiB input on UMA | OS MemAvailable/cgroup/cache/GPU pool headroom, fails before hang |
| GFX-10 | load 80–110GiB shard files, intentionally warm page cache | safe source-backed fadvise policy opt-in; no global cache eviction |
| GFX-11 | two models concurrent loads and one session disconnect | correct singleflight model/StateRoot/quarantine, no memory double charge |
| GFX-12 | source-equivalent patch-conv lowering for first new image shape | no multi-minute tuner search; numerical/shape/stride parity |
| GFX-13 | cold/warm GEMM tune/shape cache | bounded startup, drivers invalidation, no JIT per audio/media frame |
| GFX-14 | graph capture on/off, CPU MoE offload, int8, warp layout variants | full-model gain or negative receipt retained, source quality |
| GFX-15 | physical GPU driver loss or canceled submitted Future | retain actual buffer handles/envelopes and refuse false recovery |
| GFX-16 | power/thermal and shared LPDDR interference | comparable held-clock ABBA and per-model P95/energy |
| GFX-17 | unsupported wave/ISA or component/source codec | guarded slow reference or typed Unsupported; no false GPU |
| GFX-18 | multiple active live speech/video plus extreme text MoE | bounded QoS, coherent-memory pressure and no hidden numerical degradation |

## 9. Existing roadmap mapping

- P01 and P07: ISA/fragment/WMMA/DOT4/LDS/skinny GEMM candidates with real hardware source proof.
- P05 and P20: compiler guarded shape buckets, no first-use unbounded search; empirical auto-tune database.
- P10: physical UMA, page-cache/allocator distinction and guarded large-model admission.
- P11–P14: source sparse attention, GDN/KDA/mHC, grouped MoE and variants.
- P15/P16: dynamic target verifier row count and expert-aware batch/route fairness.
- P22: cost-quality-matched original checkpoints, the source as external prior art, true reproducible target-specific performance.

**Architecture remains one Rust-owned semantic engine and physical resource authority.** No new AMD-specific executor, no required Python inference runner, no uncontrolled system watchdog and no default change to driver or OS configuration.

## U-048 — combined kernel, replay and NPU qualification

The [heterogeneous pipeline contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) requires shape/ISA fusions, replay transport
and cooperative banks to qualify separately and as the intended composition.
LDS slot reuse must retire all prior reads, including asynchronous loads, under
independent GPU co-tenants; no single-process benchmark can establish that. Native
Rust ISA/AIE code paths may avoid a full SDK at runtime but still need source,
program, ABI and physical device qualification. Charge mapped/pinned arenas,
resident panels, live prepared plans and shared LPDDR once; performance calibration
cannot substitute for SoC safety, exact numerical mode or kernel barrier proof.

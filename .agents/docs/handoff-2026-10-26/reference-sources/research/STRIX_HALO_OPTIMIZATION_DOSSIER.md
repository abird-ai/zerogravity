# AMD Strix Halo dossier — gfx1151 extreme inference targets

**15 Oct extreme-specialization research reconciliation:** [extreme-specialization source atlas](SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md), [gfx1151/UMA spec](../docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) and [EXTREME P03–P22 tests](EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) detail architecture-specific WMMA fragments/LDS bank padding, source trellis codebook, sparse MoE batching/target speculation and memory-safe 100GiB model admission. Quant 2.05 vs 3.05 bpw can invert decode speed; CPU expert offload and some graph optimizations proved negative on tested upstream. Host MemAvailable, reclaimable file cache, pinned allocations and GPU GTT are different. These are **external measured gfx1151/GB10 observations**, not tested Zerogravity performance. No hidden ROS daemon/process kill or universal tuning flags for that project.


> **Research/provenance only (v3, 2026-10-12).** Project names, URLs, specific implementation techniques and source-author benchmarks in this document are **attributed prior art, not normative zerogravity runtime or compiler types**. The independent governing architecture is [Modular AI systems](../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [Composition contracts](../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md), and [Prior-art license policy](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md). External projects never become a requirement to clone a model-specific inference engine; technical/measured evidence remains available for independent validation.


**Hardware:** Ryzen AI Max+ 395 / Radeon 8060S, RDNA 3.5 `gfx1151` APU, often 64/96/128 GB unified LPDDR configurations. Distinguish machine-specific firmware and Linux kernel GTT/TTM behavior from actual available physical memory. Our architecture must also run on non-Strix GPUs via other kernels or portable fallback.

## 1. Why this APU is special

Substantial unified memory can accommodate weights that do not fit consumer discrete VRAM. CPU/GPU share physical RAM, but bandwidth, cache coherency, page pinning and competing CPU activity determine practical performance. GPU ROCm/HIP may report large virtual GTT capacity even while physical `MemAvailable` is insufficient. Kernel launch overhead, wave32 occupancy, quantized weight streaming and sparse attention selection can dominate; generic GPU graph coverage is often uneven.

**Design principle:** `gfx1151` is an eligible optimization target, not a separate hard-coded engine. Specialized regions only when semantic op/shape/quant/capacity/numerical contracts match; other operations fall back without rewriting model definition.

## 2. Models and primary cost sources

### Qwen3.8-Flash-Next
Research: 48 backbone layers, 36 Gated DeltaNet and 12 sparse attention layers, 512 routed experts top 10, gated residual, n-gram lookup and trained MTP, ~125B language backbone and ~6B active/token. The advertised 51B n-gram parameters form a different access pattern than MoE; support disk-only read of original BF16 rows where checkpoint provides them.

Main bottlenecks:
- small decode rows repeatedly stream dense and selected expert weights;
- QSA indexer/top-k/sparse attention at long context; CPU fallback can cause drastic depth scaling;
- GDN recurrence and convolution require correct state and fast small-row kernels;
- n-gram disk lookup must avoid forced giant resident allocation;
- MTP verification and rollback state must be efficient and numerically valid.

External reference:
- **Specialized gfx1151 engine**: native model-specific quant, `gfx1151` highly tuned kernels, ~1584 pp at 8K, ~1517 pp at 131K, ~37.6 short-context serial tg and ~55–56 coding spec (project claims); core engine source not published. Its ~62 GiB native checkpoint and ~47.7 GiB paged lookup differ from Gufo/DwarfStar.
- **Gufo**: `UD-Q4_K_XL` 128 GiB APU, ~1700.52 pp/25.95 AR tg d0, ~1478.66 pp/24.92 AR tg d128K; MTP mixed ~31.32, repetitive ~60.39; GPU attention and state snapshot source inspectable. One warmed sample/point caveat.
- **DwarfStar**: Qwen Metal/CUDA, not ROCm as of 2026-10-10 inspected snapshot; Q2 main/MTP 41.73 GiB and original BF16 n-grams 95.37 GiB disk-only; cannot claim its Strix Qwen performance.

### GLM-5.3-Flash
Approx 321B total/18B active, 45 layers; 34 KDA recurrence + 11 sparse MLA/DSA, 42 routed MoE layers x288 experts top8, mHC, 1 MTP. `gfx1151` 128GB can hold aggressive Q2 expert checkpoint under certain resource policies; 92/96GB requires careful scratch/expert trade-off and may stream from NVMe.

External reference:
- Maya-S Q2 ~96.5 GB GGUF, reported ~210–217 pp and ~17.8 tg on 128GB Strix with ~99.9% expert hits and 12096 experts in pool; tuning hipBLASLt/pre-fill chunk changed speed drastically.
- DwarfStar GLM-5.3 Flash Q2 about 90GiB, Strix ROCm supported including SSD expert streaming; same model family, not necessarily same quant/layout or numerical fidelity.
- llama.cpp supplies useful portable kernel/operator references, but specific sparse top-k and host round-trip paths require benchmark verification.

### DeepSeek V4 Flash
DwarfStar's research: 284B total/13B active, 43 layers; raw 128-token sliding KV + compressed historical attention at 4:1/128:1 depending on layer; 512 top-k index selection on selective layers, mHC, MoE and trained DSpark. ROCm `gfx1151` model-specific kernel improvements: 4K warm prefill ~187.26→292.86 tok/s with IQ2 MMQ/rocWMMA/attention kernel changes. This demonstrates large gains from shape-specific kernels on the same checkpoint, not Qwen/GLM direct benchmark.

## 3. Physical resource profiles (illustrative, must measure)

| RAM tier | Likely constraints | Planning response |
|---|---|---|
| 64 GiB | large Qwen Q4/GLM may not fit beside state and OS | low-bit model if user selected; disk tables; smaller contexts; explicit refusal when mandatory bytes exceed safe capacity |
| 92/96 GiB | Qwen low-bit + context feasible in some configs; GLM Q2 likely needs careful residency/tier | safe physical headroom, bounded context/prefill, expert cache sized after mandatory allocations, prioritize dense resident |
| 128 GiB | more aggressive expert-resident Qwen/GLM feasible; still OS and prefill scratch pressure | maximize useful resident pool, avoid duplicate staging, tune memory borrowing |
| Larger | weight residency easier, device bandwidth/compute/attention remain limits | adapt to actual device and controller rather than linear speed claim |

Numbers are tiers, not guaranteed working model fits. Physical availability, model quant, context, vision, draft and other processes matter. Design must use real preflight measurements instead of fixed 1GB/16GB reserves from individual reference projects.

## 4. ROCm kernel feasibility tasks

1. `hipcc`/CubeCL compile for `gfx1151`, driver/ROCm versions and codegen ISA inspected.
2. Q8_0/Q4_K/Q2_K/IQ2_XXS decode GEMV with vectorized block loads and wave32 reductions; exact GGUF quant structure.
3. Pre-fill MMQ/GEMM with rocWMMA or directly emitted matrix instructions, correct tile padding/K block reads. Compare DwarfStar's tuned IQ2 x64/y64 and Q8/F16 WMMA shapes as independent references.
4. GDN recurrent `step` and `chunk` plus short conv; optimize state update without per-token host barriers, snapshot fast path.
5. KDA fine-grained per-channel gates and state layout; measure compact recurrent representation and chunked matrix operations.
6. Qwen QSA indexer/selection/attention entirely on device; compare bounded scratch, tile occupancy, sparsity scaling to >=128K.
7. GLM MLA latent NoPE attention, DSA top-2048 and dynamic indexer, FP16 latent shared tiles; separate absorbed/expanded correctness paths.
8. Grouped MoE route+compact list+quantized gate/up/down, small-row verify and large prefill; reused expert weights across rows.
9. MTP verification compact batched rows and full state rollback; exact sampler path, no stale cache on accept/reject.
10. N-gram and Engram disk service with bounded read buffers, partial reads/corruption handling and no huge mapping.

For each track measure occupancy, register/LDS, weight bytes, instruction mix, launch count, full-model critical path and quality. A faster microkernel is not automatically retained if full HTTP gets worse.

## 5. Resident routing and streaming expert model

```text
read-only GGUF tensor manifest -> expert source
    |                  |
  dense weights     device resident cache slabs
    |                  |
GPU selected expert IDs -> device pointer table hits
    |                        |
    |                 GPU grouped kernels
    v
missing IDs -> host-mapped bounded request ring -> async read/pinned stage
                                        -> lease-safe slot eviction/upload
                                        -> device completion publish
```

All slot pointers carry a generation and active-reader lease. Avoid 100s of `hipMalloc`/free per token. For 128GB all-expert-resident case, no host trip on hit path. For 92GB cold-miss use error/timeout and explicit latency. CPU expert computation may compete for the same LPDDR bandwidth, so do not enable by default just because discrete PCIe benchmarks benefit.

## 6. Unified-memory and OS safety

Inspect GPU-visible memory pool via `rocminfo`, actual `hipMemGetInfo`, Linux kernel/device `/dev/kfd`, `MemAvailable`, `CmaFree`, cgroup available, RSS, pinned and scratch. Do not assume `rocm-smi` VRAM count is physical budget. `amdgpu.gttsize`, TTM page limits and BIOS UMA reservation can affect reported addressability but tuning them is a system-specific user action, not a runtime default or proof of physical RAM.

Failure tests must not freeze user's machine: use simulated capacity, model-free GPU allocation probes under a watchdog, then gradual measured physical loads. No disabling IOMMU. Device failure should abort request and quarantine leases, not spin indefinitely.

## 7. Large model residency and bandwidth reasoning

Approximate lower bound if `active_params * stored_bits/8` bytes must be streamed per token: Qwen 6B active at 4 bits gives ~3 GB weights before metadata and other traffic; GLM 18B at 2 bits gives ~4.5 GB. These are rough illustrative active payloads and **not** measured bytes (shared/dense inclusion, repeated experts, working-set cache and quant overhead vary). Effective t/s upper bound depends on memory bandwidth plus compute, launch, attention and SSD misses. Speculation can accept multiple tokens per target verification, changing output tokens per pass.

Memory tuning must not select an illegally small sparse-attention top-k to achieve speed. Preserve semantic selection and quant quality.

## 8. Different operating modes on gfx1151

**Latency mode:** dense/experts resident as budget allows, decode1 lowest launches, minimal host sync, spec only if net benefit. **Throughput mode:** larger native batches and grouped MoE, fairness/P95 guard, prefix shares. **Memory-safe mode:** conservatively reduced context/quant chosen explicitly and expert SSD tier as available, strict OS reserve. **Reference mode:** portable operators, diagnostic precise trace, no undocumented approximation.

Kernel and plan variants selected at model load and scheduling boundaries; avoid constant recompile/repack under changing request mix.

## 9. Benchmark scenarios to qualify

- Exact checkpoint/quant and reference logit comparison against Gufo/Maya/DwarfStar or official model, not cross-family speed.
- Qwen AR pp2048/tg128 at 0/4K/32K/128K; MTP same depths mixed/repetitive/code, reported acceptance; concurrent 1/2/4/8 sessions and model quality.
- GLM 8K context warm/cold expert cache, Q2 vs Q4 quality, initial/continued 2K/4K/8K prompt, 128/512 decode; SSD misses measured.
- DeepSeek V4 Flash 2K/4K warm prefill and DSpark low/high acceptance, no speculative baseline.
- Physical RAM and latency at 64/92/96/128GB where hardware available; no hidden swap/host OOM.
- CPU lane test on UMA vs discrete; prefetch next-layer effectiveness and IO bandwidth.
- Captured graph vs eager: CPU launch overhead, graph pointer safety and device state.

## 10. Actual speed gap ownership

If performance lags:
- prefill -> inspect tile shapes/GEMM, expert grouping, staging/dequant, quant format, occupancy/launches;
- decode -> inspect actual streamed weight bytes, dense vs sparse layer time, GPU top-k, GDN/KDA state, host sync;
- deep context -> inspect selected attention complexity, indexer/latent cache, CPU fallback, scratch spills;
- large MoE -> expert hit/miss, disk random latency, pinned copy, slot churn and quant;
- speculation -> drafter overhead, accepted length, verifier state snapshots and target rollback;
- concurrency -> scheduler head-of-line blocking, too little grouped work, too much scratch or bandwidth contention.

Do not attribute all speed loss to Rust/IR if generated GPU ISA and critical-path memory operations are the problem.

## 11. Implementation priority for this hardware

After CPU/reference IR and general dense fallback, implement small `gfx1151` kernel bakeoff, then a Qwen/GLM representative hybrid op and routed MoE, and only then full model with tiering and speculation. This order is a **proposal**; user hasn't chosen first model release priority. Strix optimization must not force model semantics or memory layouts incompatible with CPU/CUDA/Metal fallbacks.

## 12. Uncertainties to maintain openly

Qualifying model quant/checkpoint and actual real-world code quality; CubeCL codegen competitive with hand HIP; full 128GB model fit after runtime/OS reserve; 92GB expert working set and SSD bandwidth; HIP graph capture practicality; state prefix sharing cost; MTP ROI under diverse agent workloads; distribution/accuracy trade-offs. Close by physical tests and pinned reference evidence, not narrative.

## 2026-10-10 integrated v2 addendum — hardware optimization under unified compiler

**New hardware/toolchain priority:** `gfx1151` Strix Halo remains a primary target of user intent, but no purchased or attached 128GB reference machine is assumed. The historical research plan's RTX3060/Cezanne/27GB "local hardware" is *not verified current inventory*; P00 must probe actual GPUs/ROCm versions, physical RAM, cgroups, UMA firmware and KFD gfx target before claiming tests. A 92/96GB host is not equivalent to published 128GB Qwen/GLM quant context. Never blindly change IOMMU, kernel parameters or reserve GTT as physically safe RAM.

**Rust GPU provider benchmark:** `CubeCL` gfx1151 and future Rust-native AMDGPU direct instruction paths evaluated for wave32/rocWMMA or MFMA, Q2/IQ2 low-bit GEMV/GEMM, fused GDN/KDA, sparse indexer/topk and grouped experts. New NVIDIA Rust Tile/SIMT compiler choices affect CUDA portability only; don't confuse CUDA sm100 FP4 support with AMD RDNA3.5 INT8/WMMA capabilities. KernelPlan ties generated ISA/compiler/driver and packed quant layout to verified model state/quality.

**CIR differentiator:** represent device resident expert table read as typed command, host-mapped miss doorbell, bounded `pread`/IO staging and GPU upload/publish events, followed by expert GEMV. A fully resident Qwen/GLM pass has **no CPU synchronization every routed MoE layer**; a cache miss can interrupt at legal graph region boundary and resume only after valid generation. Prefill scratch borrowing must have verified release fences; CPU/GPU LPDDR contention measured.

**Model-target matching:** Qwen3.8-Flash-Next compare Gufo with exact quant/state/draft acceptance or label incomparable. GLM5.3 use Maya/DwarfStar with different Q2 recipes/quality accounted; MTP may lose on one `gfx1151` and should not default on unconditionally. DeepSeek V4 learned time-compressed attention and SSD-only Engram distinct from Qwen sparse and GLM KDA. General Llama/Gemma4/Qwen3.5+ and other AMD APU/discrete devices remain supported by generic kernels and capability negotiation.

**Real performance proof:** same checkpoint/quant/hardware and expected context, single stream vs C8 grouped, cold/warm expert/IO, long context, p95 TTFT and memory watchdog, speculative acceptance/quality. A 20% kernel microbench improvement does not establish full request/agent task t/s dominance. See `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md` S0-01/10/12 and the expanded research note.


## 2026-10-12 integration — XDNA2 NPU + Radeon iGPU on the same Strix Halo SoC

**New source:** `mikealanni/gufo` `fork-enhancements` pinned at `8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`, user Reddit 2026-10-11 on EVO-X2 MAX+395 128GB. Reports Qwen3.8 Flash-Next **UD-IQ4_XS** prefill 16K GPU-only **1,698 tok/s** vs GPU+NPU **1,821** (+7.2%), 32K NPU 1,783 and 40K cold served 1,637; do not compare numbers against upstream Gufo Q4_K_XL without checking quant, model and flags. Source actually computes *row slices* of GDN z, attention query/gate, shared expert gate/up on XDNA2 while GPU computes other projection rows, routed experts and attention. The NPU **does not** hold a second model, and this is prefill not a 1,821 tok/s decode figure.

**Performance pipeline:** RIR splits independent 4096-token-multiple projection GEMMs; KIR selects 4096×2560×1024 BF8/NPU program with source-quant and numeric provenance; CIR schedules GPU input packing, XRT BO pin/import, NPU dispatch, concurrent iGPU work, wait at first consumer, ordered output join and all-state commit. Generic/optimized GPU-only remains legal for other shapes, numerical policies and other Ryzen hardware. Optimizer should measure pinned memory/LPDDR bandwidth, XRT launch/sync overhead and load-time prepacked weights; NPU offload is not a free performance multiplier.

**Severe hardware qualification issue:** author reports machine-check hard hangs if GPU fabric clock changes during NPU+GPU activity. Their workaround requires elevated FCLK hold, NPU power mode and runtime-suspend controls; a user may see extra idle watts and firmware dependency. **Do not automatically perform privileged system tuning or turn on NPU by default.** Detect XDNA2/XRT/firmware/clock profile and expose `unqualified/experimental/validated`; GPU-only for unknown hosts. A program timeout cannot guarantee recovery of a frozen SoC. No silent IOMMU disable (can break NPU). Test isolated, fully instrumented hardware only and abort automatic enablement after any hang. See [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

**Strix validation matrix:** 64/92/96/128/192GB if physically available; `gfx1151+XDNA2` generation, GPU driver ROCm, XRT, kernel/BIOS/AGESA, idle and held FCLK, cold boot/resume, CPU+GPU+NPU contended bandwidth, NPU disabled vs enabled under identical clocks, long context and C8. A host lacking such hardware cannot have a PASSED NPU performance or safety gate. No actual zerogravity NPU implementation/benchmark exists.

# zerogravity — Comprehensive cross-stack research and design selection

> **Research/provenance only (v3, 2026-10-12).** Project names, URLs, specific implementation techniques and source-author benchmarks in this document are **attributed prior art, not normative zerogravity runtime or compiler types**. The independent governing architecture is [Modular AI systems](../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [Composition contracts](../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md), and [Prior-art license policy](PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md). External projects never become a requirement to clone a model-specific inference engine; technical/measured evidence remains available for independent validation.


**Research cut:** 2026-10-10. **Nature:** engineering research, not an implementation or benchmark result. **Scope:** inference libraries, kernel compilers, optimization passes, async execution, serving, model state, formats, memory, distributed systems, evaluation, security and evolution of the entire pipeline.

**Local source corpus**: the external read-only research library (40 substantial reports) and its pinned upstream clone manifest. This research tree is read-only in this consolidation. **Target design corpus**: /home/pvl/spaces/abird/src/zerogravity/.agents. At implementation, inspect current source SHA; documented vendor functionality may change. The word "best" means *best under measured workload and hardware objectives*, not impossible universal dominance.

## 1. Method and independent evidence classes

Evaluate every project on **semantics**, **computation**, **data movement**, **control/scheduling**, **quality**, **operational trust**, **portability** and **extension cost**. Tag findings:
- **SOURCE** — direct pinned source or first-party architectural documentation.
- **REPORTED** — performance published by the project, not reproduced by zerogravity.
- **INFERRED** — deduction from math/dataflow; must test on real hardware.
- **HYPOTHESIS** — proposed zerogravity innovation with independent baseline/kill criterion.
- **OPEN** — policy choice requiring explicit product decision or technical feasibility proof.

No architectural design by itself guarantees beating all incumbents on every model+hardware+load. The proper goal is a **Pareto envelope**: for each supported workload, correctness/fidelity and stability constraints hold, and zerogravity can choose an optimal specialization that approaches or improves upon the best measured peer without surrendering portability elsewhere.

## 2. Inference research: transferable strengths and deliberate limits

| System | Evidence-backed strongest mechanism | Carry forward | Avoid |
|---|---|---|---|
| vLLM V1 | request step as token budget, paged KV, prefix hash and serving metrics, backend runners | one request scheduler, deterministic block leasing, batch/graph modes | runtime or KV representation hardwired as the model's math |
| SGLang | RadixAttention, HiCache, overlap, P/D split, EAGLE/EP | optional prefix index, transfer-aware scheduling, production batches | making one cache/allocator policy cover all state kinds |
| llama.cpp / GGML | GGUF, 100+ adapters, mmap/CPU SIMD, backend vtable, low-bit matvec and approachable service | artifact support, quant registry, CPU fallback, provenance, single-binary UX | per-token interpreted general graph for every extreme model |
| A closed tuned reference | tuned gfx1151 Qwen Flash-Next quant+KDA/GDN/QSA/sparse+MTP and ngram table | performance target, user-visible fit/capture strategies | copying/proprietary reverse engineering; treating unlike quant as matched |
| Gufo | inspectable gfx1151 fused Qwen model and prefix/MTP/long-context correctness, proven negative experiments | state rollback, sparse selector, fast GPU quant and complete benchmarks | full runtime duplicated per model |
| Strata/Maya | expert source→slot table→device hot hit/host miss, host pinned IO, CPU lane, profile-aware LFU | async expert tier, lease/generation, routed grouping | assuming CPU/GPU coexecution helps UMA bandwidth |
| DwarfStar (ds4) | model-native compressed attention, portable tiny native engine, SSD rows/expert slabs, frontier/CommitPlan, TP/PP, QA | command plans, native data access, trained speculation, genuine disk-only lookup tables | hardcoding one graph/back-end per model; relying on native batching where it is ordered fallback |
| Heterogeneous CPU/GPU expert systems | heterogeneous CPU/GPU expert evaluation, learned selection, deferral | topology-calibrated CPU fallback where copies hurt less than execution | blind cpu offload on bandwidth-shared APUs |
| Low-bit kernel families | low-bit arithmetic, decoupled GEMV small row vs grouped GEMM prefill | quant/matmul regime specialization and format-sensitivity | inferring model quality from compressed bytes alone |
| A 2026 serving-aware kernel library | unified MoE API, runners/tactics and warmup autotuner aware of eager vs graph execution | measured candidate registry with real serving-mode timing; stable public API for kernels | a Python/JIT dependency in the serving critical path |
| Vendor/edge graph runtimes | graph transformation, vendor lowering, cross-device/state models | graph legalization and optional accelerator extension seam | monolithic proprietary backend assumptions and universal same-plan numerics |
| Rust model/runtime libraries | real Rust model/runners, tensor ops, SDK and quant workflows | Rust ownership/FFI practices, parser/model implementation references | assume Rust language alone fixes operator performance |
| A Rust graph-rewrite project | tiny primitive IR, symbolic shapes, graph rewrites/search, Rust accelerator runtimes | bounded canonical low-level primitives and rewrite ideas | unconstrained search/rewrite explosion, pure-DAG-only view of hybrid inference |

**Evidence**: the local research corpus (synthesis architecture, runtime/kernel-dataflow innovation, Rust-ecosystem, Strata/Maya and DwarfStar notes, cross-cutting studies) and its source pins. Web documentation: a 2026 serving-aware autotuner report and a Rust graph-rewrite project.

## 3. Compiler research: new decisions not fully captured by either old architecture

### 3.1 A stream/resource layer provides the missing separation

A known compiler separates mathematical `flow`, explicit async device allocation/scheduling `stream`, and low-level hardware execution `hal`. Stream includes timepoints, concurrent region scheduling, device affinity, copies, transient allocation/reuse, and host/device transfer analysis. This distinction is directly relevant to GPU expert fetch, n-gram IO, prefill scratch borrowing and session-state commits.

**Adopt conceptually, not as a mandatory dependency:** introduce a typed **Async Command/Resource IR** between optimized region graph and executable device plan. It records CPU/GPU/NVMe/remote tasks, leases, event/timepoint edges and failure boundaries. An IR that only knows "fused GEMM" cannot tell whether a copy can safely overlap attention or when an expert slot may be recycled.

### 3.2 Separating operator math from schedules; search must be selective

A known tensor compiler separates model programs from hardware-level loop/thread tiling, with a measured schedule search over alternatives for a given operator shape. The separation validates a design where a model-family recipe does **not** own the low-level kernel implementation and hardware choices are made through a candidate registry. It also cautions against embedding every available tiling decision in a runtime algorithm.

**Adopt:** Tensor/Region IR that holds semantically legal fusion/layout alternatives, and a kernel-lowering interface that can choose CubeCL or target-specific Rust-authored kernels. Do not vendor an entire external compiler runtime or introduce Python into inference.

### 3.3 MLIR's effect system validates explicit non-tensor state semantics

MLIR documents MemoryEffectsOpInterface, resource scopes and conditional speculation: only pure SSA producers can be moved freely. Stateful op changes, recurrent updates, disk/remote fetch, RNG and session commit cannot be legally optimized as if they were ordinary pure tensors.

**Adopt:** typed `EffectDomain`, explicit read/write/fork/commit/abort and resource-conflict checking, algebraic legalizations guarded by numerical policy. Distinguish compiler speculation (executing potentially undefined ops early) from **token speculative decoding**, which has its own distribution-preserving semantics. A runtime state transaction is never merely an integer offset change when underlying buffers need materialization.

Source: https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/ .

### 3.4 Minimal-primitive rewriting illustrates both opportunity and risk

A published Rust graph-rewriting effort proposes minimal primitives + symbolic shapes + search/equivalence transformations and a Rust accelerator runtime. It creates a potentially huge optimization search space even for familiar models. But a public e-graph rewrite report shows one Qwen3 MoE rewrite setup improving compile time from ~15 minutes to <1 minute by staging four rules: **rewrite-system structure and compile budget are themselves performance constraints**.

**Adopt selectively:** canonical tiny *lower-level* tensor ops, shape-aware reuse of legal rewrite rules; exploration/e-graphs only for **pure, bounded regions and offline research**, with proof/validation oracle. Avoid representing high-level KDA/mHC/top-k as untyped 15 primitives at the semantic layer, because that loses model-specific state/fallback/quality contracts. Never require a graph-wide equality saturation pass for every model load.

### 3.5 A serving-aware autotuner is decisive

A published serving-aware autotuner (Sep 2026) shows naive device-event timing can select the wrong runner: eager host launch overhead and captured graph replay have different critical paths. The correct choice depends on the *actual execution mode*, not just isolated GPU arithmetic.

**Adopt:** two-stage tuning: offline microkernels find candidates; measured **full region and whole request** tests promote winners. Persist selected tactic and evidence by model geometry/quant/device/compiler/driver/context/batch/sampling/graph mode. Keep serving free of unbounded experimental tuning. Include latency p95, physical memory, cache miss, quality as part of evaluation. Warmup profiling must be bounded or opt-in.

Source: a 2026 serving-aware autotuner publication.

## 4. 2026 Rust GPU toolchain landscape materially changes the old CubeCL-only premise

Old research master pinned `cubecl=0.11.0`, `cudarc`, and a C++/HIP kernel escape tier. **Do not freeze compiler availability to that dated snapshot.** Current options to investigate independently:

| Toolchain | Potential role | Key risk/decision |
|---|---|---|
| CubeCL/CubeK | cross-platform Rust kernel language for CUDA, HIP, Metal, Vulkan/WebGPU, CPU; portable baseline and tuned paths | gfx1151 ISA/register occupancy and target intrinsics need physical proof; rapidly evolving APIs |
| NVIDIA stable-Rust tile DSL (Sep 2026) | stable-Rust tile DSL for CUDA, natural GEMM/attention/low-bit tensor loops; CUDA 13.3+ and compatible NVIDIA hardware | CUDA-only, JIT/Tile IR, requires careful runtime and compiler fingerprint |
| NVIDIA Rust SIMT→PTX frontend (Sep 2026) | Rust SIMT → PTX, explicit warp/memory control needed for extreme kernels | early alpha, pinned nightly/rustc-dev, limited validated device and OS matrix; do not mandate |
| Rust-native AMDGPU experimentation / CubeCL HIP target intrinsics | tuned AMD gfx1151 wave32/MFMA/WMMA and buffer ops | HIP/LLVM/ROCm codegen and practical performance unknown; validate S1 with source + physical device |
| Rust shader/portability toolchains / SPIR-V / Vulkan | generic portability and GPU fallback | matrix instruction and quantized kernel limits; not expected to beat HIP on all models |
| Rust CPU SIMD | deterministic canonical reference plus tuned AVX/NEON/VNNI/AMX where supported | floating reduction order and platform portability |
| Vendor optimized math via Rust FFI | potential market-best GEMM/attention for specific hardware | user Rust-only policy interpretation unresolved; never make non-Rust model runtime a requirement |

NVIDIA's Sep 2026 introduction documents two complementary Rust kernel tracks; the stable tile DSL needs CUDA 13.3+ and stable Rust 1.89+, while the SIMT→PTX frontend remains alpha/nightly. **That is a new option not covered in the original research council's pinned compiler decision.**

Sources: https://github.com/tracel-ai/cubecl .

**Ruling:** stable internal `KernelCapability` and `DeviceBackend` ABI, compiler providers selected *per target operation and device*; do not tie Semantic IR to CubeCL or to a single JIT source language. User's all-Rust-owned-source intent remains primary. A vendor driver binding does not automatically permit copying a vendored C++/HIP kernel (separate open policy).

## 5. Unified compiler plan: math, legality, resource schedule, concrete code

```
Untrusted weights/metadata/tokenizer
         |
         v
ModelArtifact -> Verified ModelRecipe / ArchDef / semantic operator registry
         |
         v
SIR: strongly typed model math + session state/effects + position/quant rules
         |
         v
RIR: legal pure-region rewrites, fusion, layout, quant repacking, shape variants
         |                                     \
         |                              KIR / kernel registry
         |                              Rust CPU/portable/target-specific code
         |                                     /
         v                                    /
CIR: async command/resources                /
     allocations + leases + residency + timepoint/IO/peer dependencies
         |
         v
ExecutableBundle: selected kernel handles, graph segments, scratch, state cuts
         |
         v
EngineCore -> StepPlan references already-compiled bundle variants
         |
         v
Device workers (direct launches / captured device graph; async IO only on misses)
```

- **SIR** answers *what precisely computes and which state changes*.
- **RIR** answers *which mathematical transformations and fusions are valid*.
- **KIR** is a *separate codegen branch*, not interpreted at inference; translates candidate region patterns to target kernels or looks up validated custom implementations.
- **CIR** answers *when/where resources are allocated, moved, synchronized and committed*. It need not be a fully general external compiler clone; first implement a small typed command DAG and event/lease checker.
- **ExecutableBundle** binds resource policy, compiled region handles and state-compatibility conversions with explicit fallback edges.
- **ExecutionPolicy** (from a pure capability/profile planner) is *not* another IR: it records physical budget, KV variant, speculation and admission choices.
- **StepPlan** is mutable per schedule cycle and contains IDs/token counts/batch rows and compiled handles; does not contain semantic graph interpreters.

**Hot-path guarantee:** none of SIR/RIR/KIR/CIR is interpreted node-by-node for each emitted token. Runtime can issue pre-built dispatch arrays or capture, with one small bounded host step; dynamic routing should remain device-side where hot resident path permits. Plan variant caches bounded by model, shape bands and numerical contract. Fallback changes at verified state/layout region boundary, not arbitrary raw pointer substitution.

## 6. First-class adaptability in *every pipeline stage*

A genuine technology-adaptive engine must have exactly one intentional extension seam for each fundamental axis, but no mandatory dynamic plugin ecosystem:

| Axis | Minimal extension contract | Proof and test |
|---|---|---|
| Weight format / source | `ArtifactReader + TensorManifest + QuantCodec` | malformed/overflow/shard tests; model SHA+license |
| Tokenizer/chat/tool syntax | `Tokenizer + PromptRenderer + ToolParser + GrammarFSM` | byte-exact official vectors and stop/role alignment |
| Model family / operator | declarative `ArchDef`, `SemanticOpSchema`, `StateContract` | reference layer logits, state transition and shape effects |
| Graph optimization | versioned `RewritePass` with legality/numeric preconditions | oracle differential and no illegal effect reorder |
| New hardware architecture | `DeviceCapabilities + Buffer/Stream/Event` | probe + physical parity + addressability/memory safety |
| Kernel compiler | `KernelProvider` keyed by op/shape/device/quant/precision | generated ISA/correctness/latency across modes |
| Quant method | `QuantCodec` + packing + sensitivity/quality manifest | exact decode, calibration and quality band |
| KV/recurrent/sparse state | `StateDomainCodec`, cache-compatibility/canonical snapshot | fork/rewind/reuse/serialize/restore tests |
| Expert / external storage | `ExpertSource + ResidencyPolicy + IoProvider` | miss/cancel/short read/use-after-evict tests |
| Scheduler | deterministic `SchedulePolicy` with resource-cost estimates | fairness, P95, multiuser behavior and throughput |
| Draft algorithm | `DraftSource + VerifyMode + CommitPlan` | exact target distribution, accepted-prefix and RNG |
| New modality | `MmProcessor + Encoder + Projector + MmSpan` | preprocessing/multimodal positions/encoder parity |
| Embeddings/rerank | `PoolingSpec + PoolingHead` | score/normalization/Matryoshka reference |
| Parallel topology | `Collective/Transport + PlacementSolver` | rank handshake, link cost, failure consistency |
| Storage/package | immutable manifest/content store, accelerator-sidecar pack schema | checksum, migration and safe GC |
| Service transport | edge adapters only, canonical `InferRequest`, `EngineMsg/EngineEvent` | streaming, auth, truncation, tool/grammar |
| Diagnostics/benchmark | typed mode receipt, probe block, per-region trace | replayable evidence and numerical regressions |

**Maintainability constraints**: one representation/owner per concept, no backdoor model→GPU target decisions, no N-version duplicate runtime, optional features do not explode crate count, stable internal capability contracts, no sprawling trait zoo. Prefer eight to twelve cohesive Rust crates initially; split only on measured compile time, distinct release/security ownership or independent public API need. Plugins are statically linked/built-in first; allow dynamic externally authored executable kernels only after a threat model/signature/sandbox plan.

## 7. Memory and state are a compiled resource graph, not just caches

Resource lifetimes:
- **Model lifetime**: immutable checkpoint tensors, optional packed/device-resident weights, expert slots whose residency changes with generation; large n-gram/Engram table source.
- **Session lifetime**: mutable KV, GDN/KDA recurrence, sparse/indexer compressed state, multimodal position/fingerprint, sampler/RNG, draft state, logical frontier.
- **Prefix snapshot lifetime**: immutable, versioned copies or COW views of **all required state domains**, not merely K/V; canonical format used for across-plan conversion.
- **Step lifetime**: prefill activations, quant activation, scratch, GPU events, I/O staging, provisional verifier rows.
- **Distributed peer lifetime**: collective buffers, rank-specific state versions, reliable commit epoch.

One physical-budget arbiter; distinct cache eviction/reconstruction policies; all allocation through validated leases with generation and event completion. Streaming routed experts uses GPU hit pointer table, bounded host-mapped miss request ring, fixed slabs, safe publication fence and timeout. SSD-backed n-gram rows have a different tiny lookup cache from multi-megabyte expert blobs.

**New compiler advantage:** CIR may place an async next-layer expert fetch concurrently with current-layer attention, borrow cold expert slots for large prompt scratch, or schedule cached prefix restoration while model load warms, but only when effects and lease/conflict analysis prove it legal. CPU/GPU/RAM/SSD are actual resources with topology/bandwidth, not free aliases. Unified physical memory limits admission even when GPU GTT appears huge.

## 8. Cache correctness beyond vLLM's current hybrid intersection limits

vLLM's documented hybrid KV cache coordinator uses group-specific prefix-hit rules and **intersects** their valid prefixes; the design document describes current implementation optimized for full+one other efficient-attention family and incomplete Mamba caching. New model families can include more than two state domains, plus recurrent and sparse indexer snapshots.

**Proposed generalization:** cache state reuse requires a *compatible resume frontier* across every state group. Each state domain exposes `CanResume(prefix, state_schema, positional_rules, stored_rows, numeric_contract)` and `RestoreCost`; the coordinator computes a common legal frontier, respecting compressor finalization and causal masks, or recomputes the missing domain. Heterogeneous group page sizes may coexist under one budget but not one universal cache-key algorithm.

This is a potentially differentiating **correctness and efficiency** capability; it needs a test corpus with 3+ mixed state types, not just a diagram. Store hybrid immutable prefix snapshot with state domain IDs and model/quant/template + modality fingerprints. Cache key: collision-resistant SHA256/BLAKE3-256 over canonical encoded token IDs/structural keys and model/version/tenant salt; avoid a bare SHA1 rendered-text key as authoritative identity. BPE merge across rendered bytes can invalidate naive text-to-token prefix mapping, so match token positions and boundary semantics.

Source: https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/ .

## 9. Service and scheduling: concrete vLLM/Strata mechanisms under SIR contracts

Adopt research's single-owner logical EngineCore and worker delta protocol (one dedicated OS thread, per-device worker thread and bounded channels), not a global `Arc<Mutex<Session>>` or per-token interpreted graph.

- `EngineMsg` incoming adds/cancels/priority/tools; `EngineEvent` emits committed tokens/finish/status; `StepPlan` maps selected rows to compiled region handles, required leases and state epoch; `WorkerDelta` carries logits/commit candidates, events and errors.
- Canonical mixed prefill/decode/verification request scheduler uses **two budgets** (compute/input or device/encoder where appropriate), persistent batch state and diff updates; asynchronous overlap with 1–2 in-flight step plans *only after* state-frontier and grammar constraints proved.
- Generic mode: paged `KvPool` + chained hash + optional radix index, token-budget admission, recompute-only preemption when explicitly safe. Extreme low-concurrency: contiguous `KvPool`, reserve-then-admit and no-preemption if measured superior. The seam includes region movement, worker output unit and lifetime; not merely a Boolean flag.
- MoE mode: GPU route+count+group; expert residency and batch cost co-planned, but cannot predict unknown router outputs as correctness fact. Agent-lane fairness/ITL deadlines prevent speculative thrash.
- Spec controller can use trained MTP/DSpark/EAGLE/draft model or zero-weight prompt lookup; exact target distribution default in supported sampling modes; opt-in approximate semantics. Draft weights and verification scratch are globally budgeted.
- Mode receipt per response describes actual backend, quant/KV dtype, graphics path, spec mode/exactness, prefix hit, capture/eager and resource degradation. Strict reference path independent of the numerical plan.
- Advanced cross-device P/D disaggregation is optional and topology/effect-cost gated; transferring only KV for GDN/KDA hybrid models is insufficient.

## 10. Data and format pipeline: research specificity, project semantic ownership

The research implementation blueprint has concrete GGUF reader rules, quant format dialects, tokenizer GGUF pre-types, Jinja tooling, grammar-FSM/GBNF, shared adapter semantics, model store and derived pack. **Adopt these as product implementation references**, not as proof that every GGUF type or model architecture is supported.

Specific mechanisms worth retaining:
- safe, bounded GGUF/safetensors parser and per-file quant dialect registry; no arbitrary Hugging Face Python execution;
- runtime model architecture `ArchDef` can be declarative for repetitive dense/MoE, with a tested `ModelSpec` recipe built into SIR for novel stateful ops;
- GGUF is a canonical *input interoperability* path; internal accelerated packed layouts optional and regenerable from original checkpoint. Research's "APK" should be treated as a working acceleration-pack label, not an automatically chosen public file extension (conflicts with Android APK).
- quant codec registry + routed expert-aware imatrix/calibration, mixed-precision per tensor role with distinct model artifact identity and NLL/KL/NIAH quality band, optional overlay patch.
- tokenizer byte-exact first, minijinja template safety, explicit tool parsers, constrained grammar FSM separate from sampling RNG, with rollback state in speculative transaction.
- content-addressed store (immutable blobs/manifests/sidecars), safe atomic downloads, offline mode and GC; one binary with run/serve/inspect/doctor/bench/convert tools; no mandatory web UI.

**Two research defects needing correction:** (a) the alleged 48-byte disk-KV header's listed field widths sum to 54 bytes, while the 5-character magic cannot be stored in u16; design a new consistent versioned serialization before any code. (b) research claims "any model any reference can run must run" but rejects unknown mathematical architecture. Actual guarantee = any model whose **full semantic operator set is registered and lowered to a capable backend and fits resource budget** can run on portable fallback. Fail clearly otherwise.

## 11. Quantization, exactness and model performance are independent

For a given checkpoint, recognize:
- semantic formula correctness;
- numeric tensor parity versus a stable reference;
- model quality at the chosen **stored** quantization, evaluated with official and held-out tasks;
- same-plan strict greedy replay versus stochastic distribution-preserving decoding;
- cross-backend tolerance, not unconditional bit-identical tokens.

A faster Q2 model vs BF16 or different Q4 imatrix changes fidelity. No quiet quant/sparse top-k/sampling changes. A region/whole-model match must include selected sparse IDs, hybrid recurrent index state, tokenizer/template, mutable KV, full logits, nonzero-temperature exact verifier and final user output.

Research's bitwise ABBA full-logit tests are useful for controlled fixed-precision paths and zeroing hazards, but should not block all *numerically correct* GPU plans simply because floating reductions changed order. Distinguish bitwise deterministic mode from validated tolerance/quality mode.

## 12. Performance search objectives and adaptation

**Off-line**:
1. establish CPU semantic/reference and exact data layout;
2. sample candidate pure-graph fusions and kernel tensor schedules, reject illegal effects;
3. search kernel modes per shape/device for exact numeric and resource contract;
4. measure *whole region and representative full request* with graph/eager mode held constant;
5. persist passing winning plan with accuracy and perf evidence. Search only high-value hot regions.

**On-line**:
1. request admission uses current physical memory + prefix and cache leases;
2. scheduler chooses among prevalidated plan variants and safe batching, latency targets;
3. controller adjusts speculative depth/cache/IO prefetch only at legal epoch boundaries, with hysteresis and detailed receipt;
4. unsafe or slower policy automatically declines to simpler baseline, without changing model quant or hidden sampling policy.

**First projects to try beyond peers:**
- (I) state-aware full hybrid prefix sharing across 3+ cache groups and frequent agent forks;
- (II) expert-aware grouped batch × residency × speculation joint scheduler, on GPU routing;
- (III) compiler-time async command/stream layout to overlap SSD row/experts with compute and avoid scratch overcommit;
- (IV) graph-captured decode islands with dynamic model expert pointer table and typed rollback;
- (V) mixed sparse/dense architecture specialization automatically reused by novel model variants with same semantics and quant geometry.

Each proposal has a clearly defeasible experiment, not a required performance miracle; measure latency, aggregate throughput, tokens/J and quality on actual target hardware.

## 13. Targeted hardware and model priority

**Concrete user focus:** AMD Strix Halo `gfx1151` for Qwen3.8-Flash-Next and GLM-5.3-Flash, plus broader Llama/Gemma 4/Qwen3.5+/DeepSeek variants and CPU/NVIDIA/Apple/Vulkan portability. Research master described a 6GB RTX3060 + Cezanne/27GB "local box"; **this is historical research context, not a current inventory assumption.** Discover actual hardware/driver/memory by tool when implementing. No automatic purchase recommended as a prerequisite when a device may already be available.

For `gfx1151`: pin correct compiler/ROCm/kernel, measure physical `MemAvailable` and GPU addressability separately; test Q2/IQ2/Q4 decode, BF16/FP8 projections, grouped experts, QSA/DSA KDA/GDN, cache entry/IO, MTP rollback, graph capture and long context. An apparently available 128 GB virtual address range does not guarantee OS-safe resident capacity on 92/96GB RAM.

For NVIDIA: test a new stable Rust tile DSL alongside CubeCL, and a Rust SIMT→PTX frontend only in an experimental target-specific track; vendor math FFI remains user policy open. Apple and Vulkan fallback should not block initial CPU/AMD feasibility.

## 14. Maintainability scorecard (candidate designs)

Score is qualitative evidence/risk ranking, not a measured metric:
- **One hard-coded model runtime per hot model:** highest immediate performance potential, worst upkeep/model-port cost.
- **Pure generic tensor graph with one portable kernel implementation:** easiest initial compatibility, likely heavy peak-performance gap on sparse/recurrent/MoE.
- **Research original ModelSpec + KernelPlan design:** strong implementation detail, but model↔backend coupling and lack of explicit compiler effect legality may scale poorly as state types diversify.
- **Minimal pure Semantic IR compiler with no async resources:** clean model math but misses NVMe, copy overlap, cache ownership, graph-in-flight hazards.
- **Chosen hybrid Semantic/Region/Async IR + measured kernel providers + resource-aware event-driven scheduler:** more upfront design cost, best *potential* to preserve both generality and specialized critical-path execution under new model/device technologies. Must be kept small, demand-driven, bounded and validated to avoid becoming a heavy general-purpose compiler framework.

**Critical rule:** no abstraction should prevent a whole-layer or whole-model **Rust-authored** extreme kernel/graph replacing any closed region if numerical/effects/resource contracts match. The region's fallback is a correct smaller plan, not a second independent model runtime.

## 15. Research-to-implementation proof program

Before locking architecture:
- **C1 Hardware kernel bakeoff:** physical gfx1151 CubeCL vector/WMMA/IQ2 and Rust escape path, compare measured reference kernels and generated ISA; NVIDIA Rust tile/SIMT compiler candidates as a separate experimental branch.
- **C2 State-effects legality:** synthetic model with full attention, GDN, sparse indexer and RNG; generate legal/illegal rewrites, make illegal ones fail verifier.
- **C3 Minimal async CIR:** CPU/GPU event/IO mock + buffer-lease generations; prove no early free/miss/ABA and legal prefetch overlap under fault injection.
- **C4 Real end-to-end model:** GGUF tokenizer/config→SIR→RIR→CIR+kernel→HTTP output on CPU and one GPU; force missing specialized kernel and preserve output.
- **C5 Cost/tuning feasibility:** same kernels eager/captured and real traffic mixes, verify serving-mode-dependent winners and bounded warmup.
- **C6 Hybrid prefix generalized intersection:** 3+ different state domains, branch, compressed finalization, rendered/semantic keys, bitwise same-plan replay.
- **C7 MoE+spec joint test:** cross-request GPU-grouped route vs baseline, SSD and cache pressure, quantify throughput/TTFT/P95 and quality.
- **C8 Compiler maintenance exercise:** add new attention state family or GPU backend without modifying server, cache and compiler core except for genuinely new semantic/kernel capability; explicit effort count.
- **C9 Source governance:** review licenses/pins, security, incorrect old ABI; no unsupported claims or proprietary closed-reference reuse.

Exit acceptance: tests/logs/benchmark in repository **after implementation**, decision log and disabled unsafe candidates. Do not freeze a library version or public API before its functional feasibility is tested.

## 16. Distinct research sources and provenance

**Read-first local:**
- Historical three-plane architecture report: three-plane model, detailed types, core, KV, quant, performance and S1-S12.
- Historical implementation plan: bootstrap tasks, first 15, phase gates, binary formats (flag defects).
- Critique-and-roadmap report: 12 conflicts and 16 gaps.
- Runtime-innovation report: SCH/SPEC/KVM/TIER/AUTO/FAIL hypotheses with experimental criteria.
- Kernel-dataflow innovation report: K/P/D/H/A primitive experiments.
- Cross-cutting playbooks (discrete GPU, unified-memory consumer, multimodal embeddings): per-hardware and non-chat priorities.
- Hardware platform report (Strix Halo): KFD/SVM/IOMMU/ROCm memory constraints.
- Source pins manifest: pinned revisions, not a default checkout instruction; licences separate.

**Web/first-party compiler research captured on 2026-10-10:**
- https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/
- https://github.com/tracel-ai/cubecl
- https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/
- https://github.com/vllm-project/vllm
- https://github.com/sgl-project/sglang

**Final engineering recommendation:** preserve **semantic mathematics and all mutable effects** in Rust independent of hardware; use a bounded region optimizer, a separate typed async resource-command plan and pluggable Rust kernel toolchains; retain research's concrete single-owner server/KV/packaging infrastructure and fully source-backed experiments. Treat every performance claim as an optimization hypothesis until the physical model/device/reference measurement exists.

## 2026-10-12 Gufo/XDNA2 follow-up

**New experimental evidence absent in original compiler survey:** `mikealanni/gufo` fork schedules block-float XDNA2 AIE `4096×2560×1024` partial output-row GEMMs concurrently with Radeon GPU, through HIP mapped pinned/XRT-imported BOs and HIP host callbacks; GPU waits only where first consumer requires NPU output. Applied to GDN z, attention query/gate and shared expert gate/up in Qwen3.8 Flash-Next. Author reports **1698→1821 prefill tok/s on UD-IQ4_XS 16K** (+7.2%); 30 matching greedy prompts do not prove full numerical equality. **Safety:** fabric-clock changes during NPU/GPU concurrency reportedly caused hard host hangs, so firmware+driver qualification and opt-in are essential.

**Compiler synthesis improvement:** separate **per-output-row compute assignment** in RIR from **routed-expert placement** (a dGPU+iGPU asymmetric expert split) and **Async Command and Resource IR** operations `ImportMappedBO/Pack/NpuDispatch/GpuDispatch/ExternalCompletion/FirstUseJoin`. This is model-neutral and can generalize to future encoder GEMMs/other NPUs once Rust-owned AIE kernel and supported XRT FFI proven. Current source program is Python MLIR-AIE+C++ and cannot be unconditionally imported under the Rust-only user policy. Upstream Gufo's broader ASR/TTS/image/video work validates TaskProgram stage loops on shared SIR/RIR/KIR/CIR rather than one universal LM next-token loop.

Full pinned source study and benchmark/safety limitations: [GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md); runtime integration: [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

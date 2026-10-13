# Prior-art/historical archive — IMPLEMENTATION_ROADMAP.md

**Historical v2 planning snapshot, archived 2026-10-12. NON-NORMATIVE RESEARCH.** External named projects and benchmark figures are preserved as source research, while actual current design and phase gates use project-neutral contracts at `../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`, `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`. User-provided instructions and decisions remain separately in DECISION_REGISTER. Never use historical source-project APIs as the new runtime's type or module names.

---

# zerogravity — Implementation Roadmap and Closure Gates (v2 reconciled)

> **Current authority note, 2026-10-10:** keep all ZG-P00…P23 phase identifiers and detailed task coverage below, but use [INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md](../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md), [ZEROGRAVITY_MASTER_PLAN_2026-10-10.md](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) and [ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md](ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md) for **v2 sequencing and contracts**. Phase 0 now requires a *minimal* effect-aware SIR/RIR + independent KIR provider + async CIR event/lease proof alongside CPU model and physical GPU feasibility; the historical phase ordering is a dependency guide, not permission to build a large compiler before a working model. All phases remain NOT_STARTED.


**2026-10-10. PLAN ONLY.** This is an explicit, dependency-aware set of proposed implementation phases, not an assertion that anything was coded. All user decisions and unresolved items remain in `DECISION_REGISTER.md`. Prioritize verified feasibility and a small complete end-to-end engine before optimization breadth. Phase identifiers ZG-Pxx remain stable across handoffs. Each milestone requires implementation + independent tests + reference evidence + documentation updates; compilation alone is not closure.

## Governance of phases

Status enumeration: `NOT_STARTED`, `ACTIVE`, `VERIFIED`, `BLOCKED`, `DEFERRED`. As of 2026-10-10 all implementation phases `NOT_STARTED`; only the plan archive exists. Phase changes must append dated notes including SHA, exact commands, hardware/compiler, tests, outcomes, regressions and pending issues. Never claim performance without real hardware proof or functional model support solely because an adapter compiles.

Critical path **ZG-P00 → P01/P02 → P03/P04 → P05/P06 → P07/P08 → P09/P10 → P11/P12 → P13/P14 → P15/P16**. Advanced branches proceed only after foundational state/memory correctness and physical hardware validation. Parallel work permitted when interfaces, ownership and model sources are pinned, but do not introduce scaffolding from all future phases prematurely.

### Lane A: correctness and general portability
P00 baseline, P01 feasibility, P02 oracle, P03 artifacts, P04 semantic IR, P05 lowerer, P06 CPU runtime, P07 GPU basics, P08 first dense model.

### Lane B: production service and hybrid intelligence
P09 server, P10 global memory/prefix, P11 hybrid attention, P12 MoE, P13 Qwen specialization, P14 GLM specialization, P15 speculation, P16 advanced scheduler.

### Lane C: breadth and scale
P17 model family breadth, P18 multimodal, P19 distributed, P20 kernel autotuner, P21 security/releases, P22 final comparative qualification, P23 continued model generations.

---

## ZG-P00 — Repository baseline, reproducible toolchains and decision freeze

**Prerequisites:** user's later implementation authorization; decision register read. **Goal:** small Rust workspace with reproducible Nix and Cargo, pinned compiler/version, smoke tests, CI skeleton, modeling vocabulary and honest feature flags.

Deliverables:
- Git state and ownership documented; opt-in Git init if user chooses. Licensed artifact policy and third-party notices; project license remains OPEN until chosen.
- `rust-toolchain.toml`, Cargo lock, `flake.nix`/dev shell if requested, targets/build scripts (no blind system mutation).
- Workspace partition emphasizing semantic core, compiler/backend interface, runtime/service, test fixtures; no dozens of needless crates.
- Docs lint, cargo fmt/clippy/test, feature matrix, bench registry scaffolding and sample model-free fixtures.
- `capabilities` command or equivalent report; verify actual CPU/GPU/driver, not marketing names.
- Source-pin ledger for upstream benchmark/test cases; original URLs/commit SHAs, license and checksum manifest.
- Deterministic commit and release discipline; no supply-chain downloads during normal tests without explicit flag.

**Exit gate:** fresh checkout builds/tests CPU model-free on target dev OS; independent agent follows START_HERE without chat; no dependency licenses unknown in release artifacts. **Stop if:** compiler/kernel toolchain remains undecided; keep a minimal baseline.

## ZG-P01 — GPU/Rust toolchain feasibility bakeoff (high-risk critical gate)

**Goal:** prove Rust-authored GPU kernels can compile and approach high quality target-specific implementation for AMD `gfx1151` without requiring a C++ inference runtime. Candidate CubeCL versus Rust-targeted AMDGPU/LLVM pathway, with safe driver/HIP access. No assumption of victory.

Tasks:
- Matrix of `gfx1151` wave32, shared/LDS size, MFMA/WMMA access, compiler codegen, occupancy and register usage.
- Implement *independent* shape-parametric Rust kernels: Q8_0/Q4_K/IQ2_XXS GEMV decode; grouped MoE with 1/4/8 rows; 256/1024/2048-token matrix tiles; masked sparse attention selection; GDN one-step state update.
- Compare against legal reference HIP/Gufo/DwarfStar or reproduced source kernels with exact inputs and numerical tolerances; report prefill vs decode vs context.
- Verify shape tails, alignment, invalid IDs, odd K dimensions, exact pack layouts; GPU sanitizers where available.
- Measure launch overhead (eager vs captured), graph capture feasibility; read device clocks, power and frequency stability.
- Prototype CPU fallback for same operations and compiler IR extraction.

**Exit gate:** end-to-end kernel compilation and correct parity on *physical* Strix Halo, stable within a clearly recorded bounded performance gap; decision memo on toolchain path and identified missing intrinsics. **Stop/replan:** severe instruction/codegen inability that would preclude target performance; improve backend/IR or consider custom Rust-authored lowerer, rather than paper over with architecture.

## ZG-P02 — Canonical CPU numeric and state operator oracle

**Goal:** Rust CPU reference for all foundational semantics used in first model(s).

Tasks:
- Pure operators RMSNorm, RoPE/MRoPE, linear/GEMM/GEMV, SwiGLU/GELU, quant/dequant and router/scatter/gather.
- Attention full/GQA/local, causal masks, padded tails and numerically stable softmax; FP64 diagnostic path.
- GDN and KDA recurrent update/conv history, sparse-selection/indexer and mHC later by explicit operator family.
- Model-free state machine that supports `read`, `transition`, `checkpoint`, `fork`, `restore`, `commit`, `abort` for every operator.
- Reference logits for tiny synthetic models, deterministic teacher-forced sequences with preserved RNG state.

**Exit gate:** unit/property/golden tests for shapes, state rewinds, zero lengths, masks, invalid metadata; replay across chunk boundaries and speculative accept-prefix widths 0..k; no unsafe GPU dependencies.

## ZG-P03 — Artifact format, tokenizer, quantization and provenance

**Goal:** trustworthy GGUF, safetensors, tokenizer and template pipeline for known semantic models.

Tasks:
- Zero-copy bounded readers, 64-bit overflow checks, checked tensor names, offsets, dimensional product and quant layout.
- GGUF known tensor type table and adapter registration; safetensors index/shard validation, source hash, tensor manifest.
- Support native Q8_0, Q4_0, Q4_K, Q2_K, IQ2_XXS and FP8/BF16/F16 as *measured* needs; each additional type registered with reference quant/dequant and layout conversion.
- Tokenizer and chat template parity against model-specific official vectors, Unicode, special/control tokens and tools.
- Tensor-by-tensor manifest for missing/misnamed tensors, invalid quantized blocks and tokenizer mismatches; no speculative inference from arbitrary Python code.
- Optional offline packer/calibrator without changing original bytes by default; quantify loss when requantizing.

**Exit gate:** realistic model configs parse with bounded allocations; malformed corpus fails without crash/OOM; corpus token parity and known tensor golden vectors pass; model identity fingerprints stable.

## ZG-P04 — Semantic IR v1, verifier and state-effect model

**Goal:** typed graph program that declares math and state semantics independently of hardware.

Tasks:
- Strongly typed `TensorType`, `QuantType`, `ShapeExpr`, `Layout`, `StateType`, `Effect`, `Operator`, `ValueId` and `StateVersion`.
- Static semantics for read/write/fork/commit, causal position, token row/sampler RNG, image spans and sparse indexer state.
- Pure DAG value SSA plus explicit state tokens/effect edges; no state aliasing under speculative copies.
- Canonical op lowering into CPU reference; validated model structure adapters.
- Deterministic IR serialization/version/hash and diagnostic graph inspector; reject unrecognized operations and unsupported effect combinations.

**Exit gate:** synthetic hybrid transformer IR type checks; optimized and unoptimized executions produce matching reference traces and logits; illegal state reordering fails verification.

## ZG-P05 — Execution IR, optimizer and plan capabilities

**Goal:** legal graph rewrite and per-device plan selection that incurs no per-op interpreted IR cost during execution.

Tasks:
- Lower to execution regions respecting stateful edges and numerical policies; lifetime/alias analysis; alignment and layout descriptors.
- Fusion candidates norm+quant+projection, router+selection, MoE preparation, attention/post-op and recurrent update; analyze register/resource costs.
- Kernel candidate registry: exact supported op/shape/quant/device/numerical/capture contracts; typed fallback regions with conversions.
- `DevicePlan` with explicit allocations, stream events, kernel handles, captures, staging and output state versions.
- Plan variant cache for prefill/decode/verify, bounded shape variants and stable target/driver identity.
- Emit readable `inspect --explain-plan`, including why fallback occurred.

**Exit gate:** same synthetic model through portable and specialized-plan lowering, correct outputs and state; disabling any accelerated kernel falls back at safe boundary without reinterpreting incompatible state.

## ZG-P06 — Rust CPU portable executor

**Goal:** one usable general executor with feature-limited but correct model support on CPU.

Tasks:
- Row-major/strided tensor execution with bounded memory and quantized CPU matvec/matmul; SIMD multi-arch opt where measured.
- Token prefill/decode paths and single/multi-session state; deterministic sampling, cancellation/restart.
- Reference precision mode and optimized CPU mode, both with explicit trace parity.
- Provider-independent `Session` state and buffers using typed handles/arena contracts.

**Exit gate:** tiny real dense model runs CLI smoke end-to-end from GGUF/safetensors and agreed template; sustained prompts and close/reopen stable. No speed claims.

## ZG-P07 — First GPU backends and device resource API

**Goal:** safe buffer, stream, event, graph and kernel dispatch primitives.

Tasks:
- `DeviceCapabilities` reports instruction set, memory/limits, shared memory, supported dtypes and safe staging.
- `BufferLease` with device ownership, generation tags, events and alias tracking. Async upload/download with ordering.
- ROCm/HIP `gfx1151` first as proposed experimental priority; at least one second platform proof (CUDA or Metal) to avoid baked-in AMD assumptions.
- Graph-capture API only where backend actually supports required operations, no implicit malloc in captured region.
- Kernel launch descriptors with shapes, scratch, state pointers and explicit sync dependencies.

**Exit gate:** physical GPU backends run tensor/operator parity including midstream cancellation, invalid buffers, memory pressure, deterministic cleanup; detect and clearly report unsupported capabilities.

## ZG-P08 — First complete dense model, general fallback proof

**Goal:** one common Llama-style dense language model end to end on CPU and GPU.

Tasks:
- Standard causal/GQA attention, RoPE, norm, FFN, LM head, tokenizer, weights and basic KV.
- Prefill/decode and a baseline GGUF quant type with measured kernel implementations.
- Small CLI `run`, local server integration placeholder, deterministic test prompts, CPU-vs-GPU logits/tokens comparison.
- One GPU kernel missing forced during test and correctly lowered to generic alternative.
- Context length, aligned tails, prompt chunk and memory-fit guards.

**Exit gate:** real model outputs pass reference logit/continuation and basic quality checks with repeatable commands; user-facing message for mismatched/unsupported model metadata.

## ZG-P09 — Core local service and sessions

**Goal:** llama.cpp-like usability with vLLM-grade foundations but not premature distributed complexity.

Tasks:
- `run`, `serve`, `inspect --explain-plan`, `bench` CLI contract as approved.
- HTTP OpenAI chat/completions, streaming cancellation, tool-state parsing, reasoning and template semantics; Anthropic Responses planned behind individual conformance tests.
- Service session ledger, admission and typed model request; timer/metrics for queue, TTFT and inter-token.
- Async engine model: CPU control and GPU workloads staged without blocking an entire executor on one device event.
- Correct fairness for concurrent callers before native GPU batching optimization; bounded backpressure and graceful shutdown.

**Exit gate:** API integration tests with SDK clients, cancellation/reconnect/replay, malformed request fuzz, streaming consistency, no leaking hidden reasoning into visible text.

## ZG-P10 — Resource arbiter, KV/prefix caching and state snapshots

**Goal:** capacity-safe execution across multiple sessions and persistent context.

Tasks:
- Physical RAM/GPU/cgroup bounds, headroom, simulated pressure, counter checks, staging/OS reserves and preflight refusal.
- Paged KV and immutable prefix sharing with COW; versioned prefix key includes model/quant/template/positions/numeric mode.
- Recurrent and sparse-indexer state snapshot descriptors; disk snapshot integrity, format migrations and invalidation.
- Scratch borrowing and timed leases; bounded evictions and valid re-materialization.
- Nix/Linux unified memory emergency guard test on Strix Halo where applicable.

**Exit gate:** sessions branch/rewind/restart with exact state; all resources freed or intentionally cached; allocator never relies on GTT as physical free RAM; safe under forced quota exhaustion.

## ZG-P11 — Hybrid operators: GDN, KDA, sparse and compressed attention

**Goal:** model-family operator implementations beyond dense transformer.

Tasks:
- GDN chunk prefill versus one-step decode with conv/gate/state exactness.
- KDA per-channel gate and recurrence, model-specific lower bounds and causal semantics.
- MLA latent KV, sparse indexer and top-k masks; Qwen QSA and DeepSeek CSA/HCA as separate typed operators.
- Streaming attention and indexer state model, context-depth sweeps 2K..128K..long, if memory permits.
- Parity against CPU high precision and official model vectors, not just plausibility.

**Exit gate:** synthetic and at least one real hybrid model passes reference logits, long prompts, restart and mixed chunk boundaries; every new op has a working portable lowerer.

## ZG-P12 — MoE routing, quantized kernels and expert tiers

**Goal:** high-performance routed/shared expert execution within the global budget.

Tasks:
- Router scoring/top-k/scaling, support several expert layouts and quant types with no assumption GLM/Qwen routing semantics identical.
- CPU+GPU expert execution with measured benefits, separate discrete/UMA policies.
- Fixed-size device expert slabs partitioned or protected by leases; dynamic slot table and miss requests; pinned staging and SSD IO.
- GPU-side routing/grouping, small-row vs large-prompt kernel dispatch, expert overlap across requests.
- Cold/hot cache statistics, eviction/prefetch policy and scratch/cache budget joint admission.

**Exit gate:** 1000+ randomized route sequences with no overwrite hazard; forced miss/evict tests, no stale activations, identical logits on CPU/reference; real MoE runs with cold/warm metrics; safe cache resizing.

## ZG-P13 — Qwen3.8 Flash-Next extreme Strix Halo plan

**Goal:** coherent complete Qwen3.8-Flash-Next model path on `gfx1151`, against Gufo/llama.cpp under matched weights and policy.

Tasks:
- Canonical `qwen4exp` model adapters, GDN, QSA, gated residual, 512 experts top-10, n-gram row lookup, built-in MTP.
- Q2/Q4 candidate format support without claiming quality equivalence; disk-only 95 GiB BF16 n-gram when appropriate.
- Device-native sparse selector/attention with long-context stable complexity; fused residual and decode graph candidate.
- `gfx1151` wave32 quantized expert decode, sized prefill MMQ and grouped verifier; correct rollback/checkpoint.
- Compare independent scalar, batched AR and MTP at 0/4K/32K/128K contexts with official outputs and quality.

**Exit gate:** complete model correctness, longitudinal state replay, reference Qwen vectors, cold/warm stability and reproducible external engine comparisons. On smaller RAM devices allow a distinct quant/ctx plan with explicit quality/capacity tradeoffs. DO NOT claim 1,700 prefill or 60 tok/s unless independently measured.

## ZG-P14 — GLM-5.3-Flash extreme Strix Halo plan

**Goal:** 321B-class sparse MoE with KDA/DSA/MLA/mHC on unified and discrete hardware.

Tasks:
- 45-layer geometry and model-native quant formats; 34 KDA plus 11 sparse MLA, mHC, 42 routed layers of 288 experts x top8; model-native MTP.
- Mixed quant weight recipe and reference quality, density-vs-memory comparison.
- Full resident pool on 128 GB if actually fits physical budgets; tiered cache and NVMe miss handling for smaller RAM.
- KDA state checkpoint, MLA absorbed vs expanded correctness alternative, sparse indexer and attention kernel.
- Optional dual-GPU layer split with correct signal/transport semantics after single GPU verified.

**Exit gate:** real GLM state/logits and official continuations; SSD cold/warm tests, no hung device on IO errors, bounded host memory; benchmark Maya and DwarfStar at controlled quant/quality as available.

## ZG-P15 — Speculative decoding engine and transactional verification

**Goal:** consistent target verifier independent of drafter, tested state rollback.

Tasks:
- Speculative state machine, verification proposals and accepted-prefix commit, draft store and RNG tracking.
- Native MTP (Qwen/GLM), DSpark if weights/licensing support, EAGLE-style, smaller external draft, prompt/n-gram lookup.
- Distribution-preserving stochastic sampling and exact greedy target-per-plan control; explicit approximation mode.
- Dynamic drafter/length policy based on **measured net benefit**, acceptance, memory and validation cost; hysteresis and bypass.
- Tool boundary, EOS/stop/cancel, rewind/rollback, 1..N accepted tokens, batch/out-of-order failure tests.

**Exit gate:** golden logits, greedy and sampled replay with exact target distribution, stable state through rejection, duration/throughput improvements demonstrated without degrading unpredictable workloads.

## ZG-P16 — Native batching and joint memory/schedule planning

**Goal:** move from fair multi-client service to high-throughput continuous batching.

Tasks:
- Unified token budget for waiting/running, mixed chunked prefill and decoding, verification integration.
- GPU-grouped routing rows by expert across independent sessions with correct scatter/reduction.
- Shared weight and temporary buffer reuse, bounded batch sizes, queue latency and starvation constraints.
- Dynamic cost model considers decode wall time, cache miss risk, contention, scratch, speculation and prefix value.
- Physical stress at concurrency 1/2/4/8/16/32 where memory allows; compare plain vs fused group path in ABBA benchmarks.

**Exit gate:** measurable aggregate improvement, bounded P95/TTFT, identical per-session state semantics, no starvation or excess overcommit, forced fallback still correct.

## ZG-P17 — Broader model family readiness

**Goal:** Gemma 4, Qwen3.5+, Llama newer families, DeepSeek V4 compressed attention and future variants.

Tasks: implement semantic adapters with versioned operator registry and capability matrix; dense/MoE, local/global attention, hybrid recurrence, compressed KV, mHC, native tokenization and templates; import checkpoint manifest with provenance. Prioritize models by user choice and reproducible weights. Model QA per variant/quant. No blanket `--trust-remote-code`.

**Exit gate:** model-by-model acceptance table with usable contexts, formats, modalities, numerical errors and performance.

## ZG-P18 — Multimodal video/image/audio extensions

**Goal:** standardized modality-specific semantic operators and state lifecycle; image preprocessing/fingerprints, encoder weights, multimodal positions/MRoPE, vision attention, audio encoders and streaming. Explicit state reuse and invalidation across image edits, tool boundaries, snapshots.

**Exit gate:** official vision embedding parity where available, mixed image/text sessions and cancellation/resume, bounded input and no hidden network fetches.

## ZG-P19 — Parallel and distributed serving

**Goal:** topology-aware tensor/expert/pipeline parallel, heterogeneous GPUs, optional P/D split, recoverable errors.

Tasks: versioned wire protocol, authenticated transport, handshake hashes, per-rank graph identities, layout partitioning, remote state ownership, collective timeouts, retry discipline, frontier atomicity, fail-closed on ambiguous outputs; benchmarks include bandwidth and memory.

**Exit gate:** two real devices/hosts under network loss and process crash, no split-brain states, correct session restart from persisted prefix. Measure whether distributed mode beats capacity-only alternative.

## ZG-P20 — Autotuning and optimization lab

**Goal:** improve runtime without speculative complexity contaminating correctness.

Tasks: reproducible `bench kernel|model|serve`, ABBA control/candidate scheduling, pin kernel/driver build, device telemetry, kernel profiling and register/LDS occupancy, noise filtering, tuning DB provenance, tracked rejected experiments, safe rollback and invalidation.

**Exit gate:** multiple model/hardware wins reproducible by second run and regression suite; auto selects only proven candidates.

## ZG-P21 — Packaging, reliability and security

**Goal:** production-quality artifact.

Tasks: signed/checksummed builds when user selects signing policy, reproducible Cargo/Nix/release pipeline, per-platform feature flags, local-only default API bind, TLS/auth policy, resource quotas, fuzz malformed formats/templates/cache state, permission/sandbox design for optional agent tools, crash recovery, observability and docs.

**Exit gate:** release checklist in `SECURITY_AND_RELIABILITY.md`, no critical unexplained crashes/corruption, reference correctness and minimum viable model matrix pass.

## ZG-P22 — Comparative performance/quality release qualification

**Goal:** evidence-backed claim matrix, not promotional estimates.

Run like-for-like matrix: CPU and supported AMD/NVIDIA/Metal; at minimum one dense, one hybrid, one large MoE; compare llama.cpp, vLLM/SGLang, Gufo (documented only if binary legally available), Maya/DwarfStar, an independent Rust serving project, as feasible. Capture model identity, quant, context depths, concurrency, complete latency histograms, acceptance, P95, memory, power and model quality. Report **where zerogravity wins, ties and loses**, with confidence intervals.

**Exit gate:** reproducible public-style benchmark protocols and artifact checksums; explicit unsupported cells and performance uncertainties.

## ZG-P23 — Ongoing compatibility/autoresearch

**Goal:** new model family can run semantically correct first and then improve by adding optimized kernels; hardware target can be introduced without redefining models.

Tasks: model integration guide, kernel shape contract registry, target-specific compiler tests, quant schema revisions, rolling external diff reviews, continuous device perf telemetry where user-controlled. Never silently replace checkpoint/quant.

**Exit gate:** external contributor adds a new compatible model using semantic components without changes to HTTP/memory runtime except genuinely novel capabilities; device backend can skip extreme specialization and still run the tested general path.

---

## Cross-phase nonnegotiable checklists

### Any new model
- [ ] Weight config/architecture/tokenizer/checkpoint SHA pinned
- [ ] Semantic ops, shapes, quantization and chat template explicit
- [ ] Reference execution logits and state transitions
- [ ] Adapter rejects unsupported metadata clearly
- [ ] CPU fallback and GPU compatibility matrix revised
- [ ] Numerical/quality regression versus official/independent reference
- [ ] Prefill/decode, continued prefill, long context and cancellation
- [ ] Speculative path if applicable, otherwise explicitly disabled
- [ ] Human-friendly error and memory-fit report

### Any new kernel or rewrite
- [ ] Capability contract, preconditions, aligned/ragged tails
- [ ] CPU/reference parity, numerical tolerance and important near ties
- [ ] Effect/state ordering, scratch lifetimes, no alias hazards
- [ ] GPU sanitizer and repeated graph/eager tests
- [ ] ABBA end-to-end throughput; no silent long-context or batched regressions
- [ ] fallback path tested under forced failure
- [ ] tuning key/version update and source attribution

### Any new memory optimization
- [ ] Global reservation and OS headroom
- [ ] Lease/generation proof against use-after-evict
- [ ] Cold/hot miss, short and long prompts, budget resizing
- [ ] I/O errors return bounded failure; never consume stale data
- [ ] UMA and discrete transfer assumptions documented
- [ ] OOM safety monitored on physical host without dangerous stress defaults

### Any new speculative feature
- [ ] Same-plan greedy control, stochastic exact distribution check
- [ ] Every accept length, rollback and RNG restored
- [ ] Tool/sampling-mode boundaries, EOS, cancellation and restart
- [ ] Non-speculative fallback without prefill replay storms
- [ ] Workload-dependent draft benefit, worst-case overhead

## Milestone releases (not dates)

- **M0: Proven architecture** — P00–P05 contract + GPU feasibility.
- **M1: Useful portable local runner** — P06–P09 with a full dense model.
- **M2: Strong general runtime** — P10–P12 state/memory/hybrid/MoE.
- **M3: Strix Halo specialist** — P13–P15 and Qwen/GLM parity.
- **M4: High-throughput server** — P16 and selected P17/P18.
- **M5: Scaled and qualified** — P19–P22 on physical hardware.
- **M6: Extensible platform** — P23 compatibility maintenance.

## Dependency and scope rules

Do not attempt P13 on the strength of P01 synthetic kernels alone; real semantics, state, artifacts and memory must exist. Do not build elaborate autotuning in P05 before one end-to-end representative benchmark. Do not implement distributed inference before single-device state transactions are proven. Do not let the reference path become a slow unmaintained test-only code path: it is the correctness backstop of the project.

Every completed phase must update this roadmap, decision register if decisions changed, affected subsystem docs, handoff's recent status, and tests/bench/evidence. No progress statement can substitute for an actual repository inspection.

## 2026-10-10 integrated v2 delivery adjustments

**First executable artifact is a correct small end-to-end model, not a 20+ crate scaffold or a giant optimizer.** The original research program S1–S12 provides useful actual proof tasks, but its 93 engineer-days total is not equivalent to 4–6 weeks for two engineers; distinguish **mandatory hard-gate** vs optional research experiments. Consult `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md` for **S0-00…S0-14**, full dependency and physical gate details.

The primary hard path after P00: (1) P01 gfx1151 Rust GPU codegen vs reference plus optional CUDA Rust tile / SIMT-PTX compiler probe; (2) P02+P03 actual GGUF/tokenizer/CPU math/quality oracle; (3) **P04 minimal SIR with explicit effects**; (4) **P05 RIR pure fusion + KIR kernel providers + CIR async lease/event DAG**; (5) P06/P07 portable CPU/GPU execution; (6) **P08 real model SIR→RIR→CIR→ExecutableBundle→EngineCore→token** with forced partial fallback; (7) P09+P10 paged KV/service/prefix and historical research EngineCore/StepPlan/WorkerDelta; only then expand P11–P23.

**P12–P16 major performance work:** source-specific MoE cache and GPU route, Qwen/GLM specialty, full-state MTP/DSpark `CommitPlan` and joint scheduling; no performance claim without like-for-like model+quant/context+device/acceptance and p95. **P17** expands broad modern models and explicitly adds LoRA/embeddings/reranking runner capability. **P18** multimodal and video/audio future, scope independently qualified. **P19** topology-aware placement and rank-frontier safety. **P20** two-level offline/online tuning + kernel-dispatch eager/graph tactic research, bounded-search pure rewrites. **P21** all-Rust/vendor FFI open product policy and source licenses, trusted external kernel plugins optional only under secure controls.

### Major ownership contracts required at P00 interface review

- `ArchDef/ModelRecipe`: mathematical tensors/semantics only.
- `SemanticOpSchema/StateDomainCodec`: CPU oracle/effects/transaction.
- `Compiler/KernelProvider`: region legalized and device-tuned, no model-specific `kernel_plan(hw)` ownership.
- `CIR/ExecutableBundle`: typed BufferLease and GPU/host/SSD/timepoint/graph ordering.
- `HardwareProbe/PlanSource/ResourceArbiter`: real physical safe admission and device policy.
- `EngineCore/StepPlan/WorkerDelta`: canonical request/session owner and asynchronous worker protocol; direct bundle handles on hot path.
- `KvPool/ExpertStore/Frontier/CommitPlan`: paged or contiguous state/weight staging and all-state rollback.
- `InferRequest/ModeReceipt`: input/output fidelity, conformance and user-visible degradation.

Proposals are not user-approved binary ABI; freeze a *minimal* version only after G0-C complete vertical slice. All old per-phase "Exit gate" text still applies unless explicitly modified by `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`; source and context changes tracked with dated notes. Never mark a phase VERIFIED based solely on docs.


## 2026-10-12 NPU and Gufo upstream pipeline integration (U-016)

Keep all existing ZG-P00…P23 phase IDs and exits. Add **specific work items**, not a separate inference executable:
- **P00–P01:** source-pin `mikealanni/gufo` (fork-enhancements `8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`), upstream `gufo-org/gufo`, device safety/driver/XRT inventory; research Rust-authored AIE/XDNA2 kernel pipeline and permissible vendor runtime FFI (O-023).
- **P04–P05:** add generic RIR `ProjectionRowSlice` legalizer and explicit math/tensor row-map coverage; KIR kernel-provider device capabilities, CIR cross-API Host→GPU→NPU events/leases and first-use `AssembleSlices` join. Mock test no NPU required.
- **P07–P08:** optional `xdna` device backend, BO/import/driver/event/fence/capability/refusal; no core runtime dependency on XRT; verify GPU-only portable end-to-end path unchanged.
- **P10–P13:** resource arbiter accounts NPU prepacked immutable weights, pinned A/B/C and instruction/context, no GTT overcommit; Qwen3.8-Flash-Next GDN z, attention query/gate, shared expert gate/up optional NPU prefill row slices with GPU continuation, source model parity; no NPU default-on.
- **P16/P20:** shape/context/phase-aware NPU offload cost and bundled variant selection (GPU-only versus cooperative); controlled ABBA same FCLK/power baseline, P95 and energy, and full-model quality; no speculative policy change or trained MoE expert reduction.
- **P17/P18:** define `TaskProgram` typed orchestration for ASR speech encoder/transcription, TTS acoustic/vocoder audio streaming, image flow/diffusion editing and video temporal outputs, in addition to LLM decoder/drafter and embeddings/rerank. Reuse SIR/KIR/CIR/ResourceArbiter; independent model+task release cells.
- **P21/P22:** full SoC firmware/AGESA XRT/ROCm safety qualification, real-accelerator compiler/program license and Rust-source policy, DMA timeout/quarantine, boot/suspend/24h reliability, and matched UD-IQ4_XS 1698→1821 tok/s source report validation. Never require privileged automatic clock or kernel changes.
- **P23:** extension drill: add one new NPU kernel shape/provider and one new nontext TaskProgram stage without touching unrelated model/serving code; record source/quality and fallback.

**New integrated Stage-0 track S0-NPU-A…F** in `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`; it can be BLOCKED without blocking normal CPU/GPU G0/G1. Strong safety rule: GPU+NPU concurrent work is a *hardware-qualification feature*, not a generic supported device just because `xrt-smi` enumerates it; reporter observed hard machine hangs during fabric clock changes. Source detail and implementation boundaries in `GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md` and `../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md` (read before implementing this track).

All these are **plans, not implemented features or a promise of automatic enablement**.
# Zerogravity — architecture roadmap P00–P23

This is a design reference. Current work and completion live only in
[the plan TODO index](TODO.md); use [AGENTS.md](../../AGENTS.md) for the workflow.
The phase IDs and architectural requirements below remain unchanged.

## A. Program objective and hard invariants

Build one native Rust **AI TaskProgram + semantic compiler + compiled device executor** that accepts independently verified model/artifact types and replaces closed computation regions with optional specialized implementations. All phases respect one session/state owner, one physical resource arbiter and one async command/event graph, without creating a new model-specific engine. Source projects and research publications are not code architecture dependencies; MIT code reuse is optional after per-file audit, non-MIT sources stay prior-art.

**Phase complete only when:** actual source/test/model artefacts exist, independent reference and numerical quality pass, devices/IO/state are safe, and documentation records exact commands and model/source SHA. A phase may be functionally VERIFIED without being PERFORMANCE_QUALIFIED. No unvalidated kernel/project benchmark counts as zerogravity results.

### Critical minimum path

```text
P00 workspace/source & truth
  ├── P01 device/codegen feasibility (CPU/real GPU; optional NPU independent)
  ├── P02 portable mathematical/state oracle
  └── P03 artifacts/tokenizer/quant
             |
      P04 minimal effectful SIR + TaskProgram stub
             |
      P05 legal RIR + kernel registry + CIR event/resource proof
             |
      P06 CPU direct/reference executor + P07 one GPU backend
             |
      P08 first REAL model→task→generated output vertical slice
             |
      P09 serving/stages + P10 state/physical resources
             |
      P11 stateful/sparse semantics + P12 routed experts
             |
      P13/P14 selected heavy model/device specializations
             |
      P15 target-authoritative speculation + P16 task-aware batches
             |
      P17 operator/model family growth + P18 multimodal AI TaskPrograms
             |
      P19 heterogeneous/distributed topology
             |
      P20 tuning, P21 security/release, P22 qualification, P23 extensibility
```

Dependencies are minimum contracts, not proof that every future operator needs all preceding phases. Work can overlap only where interfaces and validation are isolated; avoid premature construction of all modules.

## ZG-P00 — Lean foundation, source policy and reproducible build

**Goal:** build a minimal Rust workspace that genuinely compiles and runs tests on CPU; pin toolchain and dependencies reproducibly, including optional device features. Prepare versioned data/semantic/lease identity types without adding 25 empty crates.

**Deliverables:** project metadata, Cargo lock, dev shell/Nix where appropriate, CPU-only smoke executable, test fixture policy, source/NOTICE/license audit, benchmark harness skeleton and hardware capability probe. A model/toolchain/source SHA ledger and exact trusted artifact path policy. Distinguish device execution source-language requirement from vendor runtime FFI; not all policy questions can be resolved by code.

**Gate:** fresh checkout builds CPU without GPU, no unknown dependencies or unreviewed code imports, static source policies and verified file constraints, measured exact local device inventory. **Out of scope:** implement full IR/backend architecture in one massive scaffold.

## ZG-P01 — Architecture-specific Rust kernel and hardware feasibility

**Goal:** demonstrate at least one real physical GPU kernel compiled from Rust-authored source with correct reference parity, and evaluate access to critical ISA/quant/tiling.

**Deliverables:** small shape matrix (dense decode GEMV, prefill GEMM, low-bit packed blocks, route+gather, simple recurrent update), device capability/driver/physical memory probe, generated ISA/register occupancy and reference benchmark. Multiple optional Rust codegen/kernel-provider backends may coexist; experimental target compilers cannot force the whole core to use unstable toolchain. Driver and vendor math FFI policy explicit.

**Gate:** correct physical GPU execution and plausible path to native instruction efficiency; if an extreme backend fails, portable GPU/CPU plan remains possible, and specialization is marked blocked. Optional cross-accelerator NPU programming and power-fabric qualification run in **separate Stage N gates**, never as release prerequisite for CPU/GPU.

## ZG-P02 — Independent mathematical and state-transition oracle

**Goal:** canonical Rust implementation of all operators needed for the first full model plus small synthetic stateful pipeline.

**Deliverables:** scalar/FP32/FP64 diagnostic norms, quant matmul, causal attention, gating/router/activations/positioning and reference KV/recurrent/indexer effects; `StateTransaction` begin/commit/abort/fork/replay; reference RNG/sampling where needed. Golden vectors and generated property tests for chunk/branch state and illegal sparse indices.

**Gate:** independent official/reference parity and blocked invalid state/order transformations. New semantic ops always require reference truth before optimized kernels.

## ZG-P03 — Trusted artifacts, quant codecs and rendering

**Goal:** read a real model checkpoint and produce canonical token/media input without executing external model repository code.

**Deliverables:** bounded GGUF/safetensors + shard integrity, quant codec/block manifest, tokenizer/pre-type and model-native template vectors, special tokens and structured input; content-addressed artifact manifest/store skeleton and safe offline resolution. Distinguish checkpoint storage quant, lossless physical packing and optional lossy compute format.

**Gate:** malformed corpus fails safely; real checkpoint tensor names/shapes and tokenizer/role rendering match official fixtures. No model support from a parser alone.

## ZG-P04 — Typed Semantic IR and minimal TaskProgram

**Goal:** represent pure tensor math and mutable state as explicit effects, with a minimal typed task stage/loop wrapper.

**Deliverables:** SSA values, symbolic logical dimensions, numeric policy, state-effect tokens and schema, one simple autoregressive task and one synthetic non-token bounded iterative task; IR dumper/verifier. Canonical CPU reference lowering.

**Gate:** illegal reorders/alias/position breaks rejected, full state frontier after speculative branch correct, same task operators can be interpreted in diagnostic mode and lowered without semantic difference. Keep grammar/media and novel model ops outside initial scope unless needed by vertical slice.

## ZG-P05 — Legal region optimizer, kernel registry and async CIR

**Goal:** minimal executable compiler that can substitute a whole region without a hot-path semantic interpreter and can enforce async memory/resource safety.

**Deliverables:** RIR fusion and quant/layout transformation legality, shape-band kernel candidate registry (KIR), optional static provider, CIR event/resource command DAG with buffers/transfers/dispatch/wait/commit/abort, liveness/alias/generation checks; tiny `ExecutableBundle` cache and safe fallback conversions.

**Gate:** reference parity for optimized vs nonoptimized synthetic graph, forced specialized-kernel miss, simulated IO failure/stale mapped allocation/cancel, no compiler AST walk per token. Do not implement an unlimited equality-saturation optimizer at this phase.

## ZG-P06 — Useful portable Rust execution backend

**Goal:** correct CPU model executor with direct prepared kernels, bounded allocations and no GPU dependency. Optional SIMD optimized path after reference tests.

**Deliverables:** valid tensor ops/quant matvec, one text task session with prefill/decode and sampler, CPU StateDomainCodec and task loop, cancellation, diagnostic state checksums, resource refusal. Clear unsupported-op errors and first minimal SDK.

**Gate:** one real small model can generate meaningful reference-checked output on CPU, including prompt continuation and stop/cancel. Source quality reference recorded.

## ZG-P07 — Accelerator providers, streams, leases and kernel ownership

**Goal:** one real GPU DeviceProvider capable of physical allocation/copy/launch/events, optional graph capture, persistent kernel/layout handles and safe failure; second GPU target smoke to guard portability.

**Deliverables:** `DeviceCapabilities`, `ResourceMemoryAdapter`, `EventBridge`, typed leases, async worker and graph replay where supported. Portable tensor kernels and optional fast kernel variants. Generic device capability negotiation and fallback.

**Gate:** CPU vs physical GPU parity on real operators and device-loss/cancel tests; no stale buffer free and no hidden global driver synchronization. NPU/other coaccelerators may remain unqualified.

## ZG-P08 — First complete real model and TaskProgram vertical slice

**Goal:** connect **artifact→model definition→SIR→RIR→KIR/CIR→CompiledBundle→TaskCoordinator→output** for one model/task on CPU and eligible GPU.

**Deliverables:** source-verified checkpoint+quant, tokenization/template, complete architecture operator implementation, state+sampler, CLI smoke (optional HTTP), `inspect --explain-plan`, forced disable of a specialized region and correct general fallback.

**Gate:** actual model logits/continuations and state parity, repeated prompts and chunk boundaries; memory-fit estimate; no duplicated model server or hidden external inference runtime. This is the primary architectural viability milestone.

## ZG-P09 — Native TaskCoordinator, worker and serving protocols

**Serving precision:** Three independent admission valves—bounded API/tenant request count+payload, queued prefill work+active model/session residency, and physical per-step active sequences/token+media rows/scratch—with typed overload and fair cancellation. Requests do not preallocate full KV while waiting. Source-valid decode/duplex work and aged chunked prefill share real event-backed Prepare/Dispatch/Retire, avoiding per-token router/catalog/JSON overhead. SERVOPT-01..11, 24..27, 32.


**Goal:** production-minded service skeleton, one logical session owner, event-driven heterogeneous device workers and stage-aware TaskProgram execution without forced LM-specific internals.

**Deliverables:** `TaskRequest/TaskResult`, typed `EngineMsg/EngineEvent`, `StepPlan/WorkerDelta`, persistent batch state, bounded channels, streaming/cancellation, structured model/tool/grammar output, local-only safe API/SDK, metrics and mode receipts.

**Gate:** canonical API/CLI share one engine path, no global per-token locks, no provisional output to clients, correct fairness/recovery under multiple sessions and simple concurrent tasks.

## ZG-P10 — Global physical memory and full-state prefix manager

**Serving precision:** Prefix reuse is the largest **source-complete frontier across every required state domain** (full KV, SWA, GDN/KDA/SSM, MLA/indexers, draft and media processors); exact artifact/tokenizer/template/numeric/tenant key, COW, real GPU reader leases and measured RAM/SSD restore-versus-recompute. Prefix reuse must never claim success from KV-only bytes when another required domain is absent. SERVOPT-12..18, 26.


**Goal:** unify memory reservation across all device/host/IO resources while allowing distinct policy per model weights, sparse experts, KV/state, prefix snapshots, adapters, encoders and scratch.

**Deliverables:** physical RAM/cgroup/UMA/addressability budget+OS margin, paged state pool, optional contiguous low-concurrency pool, immutable prefix sharing, 3+ StateDomain intersection/resume alignment, host/disk snapshot integrity and strict private cache identity. Per-request admission and safe cache evictions.

**Gate:** no OS-host OOM in controlled quota tests, state exact after snapshot/fork/replay, no double free/in-flight slot overwrite, graceful admission refusal. Huge virtual GPU address space cannot be counted as spare physical memory.

## ZG-P11 — Hybrid/stateful attention and complex operators

**Goal:** expand semantic reference and optimized device implementations for recurrent attention, latent/sparse/compressed context and multimodal position tracking.

**Deliverables:** GDN/KDA-style exact recurrence variants, QSA/MLA/indexer-like sparse selection, learned compressed history, hyperconnections/residual variants as required by selected model fixtures; chunk prefill vs step decode, fixed state layouts and checkpoint boundaries.

**Gate:** official full layer/logit/quality parity at representative deep contexts, safe acceptance/rewind/recompute, strict sparse IDs/position and fallback at legal layout boundary.

## ZG-P12 — Routed sparse parameters and tiered expert execution

**Serving precision:** Compile source/quant-compatible grouped decode/prefill/verify rows; account routed unique experts, padded physical row tiles, actual encoded GPU slot pins and scratch/transfer budget. Multiple models may interleave at one worker but incompatible models cannot be fused into one invalid GEMM. SERVOPT-21..26.


**Goal:** mathematically faithful routed experts across differing residency and compute tiers, no GPU↔host sync on resident hot path.

**Deliverables:** exact router scoring/top-k/shared expert; GPU compact assignment/grouping, low-row and large prefill kernels, leased resident expert slots/generation/prefetch, host bounded miss ring, IO staging and optional CPU execution lane only if measured favorable. Separate disk lookup-row service for massive immutable table.

**Gate:** synthetic randomized route sequences, cold/hot/SSD corruption/cancel tests, real MoE source reference parity; no expert ID changes for performance absent explicit changed-model policy.

## ZG-P13 — First extreme hybrid sparse model specialization

**Goal:** select a complex hybrid/recurrent/sparse/MoE model checkpoint and reach a fully correct specialized hardware plan while preserving portable fallback.

**Deliverables:** model-specific operator graph and source data, fused region candidates, GPU-resident sparse index/route, long-context stable attention, optional trained multi-token predictor, bounded lookup row IO and first measured critical-path profile. Optional GPU/NPU split-projection rows only when qualified.

**Gate:** complete model validation and matched source checkpoint/quant/context/power performance against source evidence, plus a forced generic region fallback and intact state transaction.

## ZG-P14 — Second, substantially different heavy model specialization

**Goal:** prove architecture reuse on a model with different recurrence, indexed sparse attention, hyperconnections, expert number and quant geometry—without rewriting first model's server.

**Deliverables:** new model recipe/operator implementations/reference; reuse same resource arbiter, state/TaskProgram and kernel registry; adaptive resident/streamed experts as physical fit allows and optional heterogeneous compute owners.

**Gate:** correct complete model+state and controlled resource/SSD/P95, including smaller vs larger RAM device tiers; documented delta code footprint demonstrates composability.

## ZG-P15 — Target-authoritative speculation and model/task transactions

**Serving precision:** Versioned MTP/DFlash/lookup/small-model DraftSources are provisional; only target verifies source-correct accepted prefix with exact sampled correction, grammar/stop state and full rollback. Benchmark proposed-vs-accepted tokens, state overhead, expert-cache churn and low-acceptance disable. SERVOPT-19..21.


**Goal:** support optional trained predictors, draft models, prompt/token lookup proposals and sampled distribution-preserving verification without state leakage.

**Deliverables:** `DraftSource`, exact target verifier/correction, state frontier snapshots/CommitPlan mechanism, arbitrary accepted prefix, per-task RNG/grammar, source artifact compatibility and dynamic draft policy backoff. Task loops outside LM use their own verifier semantics, not automatically borrowed.

**Gate:** exact target-distribution correctness where claimed; all-state rollback, EOS/tool boundaries, cancellation, moderate/long context and no hidden token stream drafts; fail/disable when net slower.

## ZG-P16 — Continuous batching and cross-resource cost optimization

**Serving precision:** Choose from finite prevalidated Pareto StepPlans jointly spanning active rows, legal prefill chunk, draft depth, MoE expert grouping, GPU/RAM/SSD residency, backend kernel, placement and soft deadlines. Optimize **committed SLO-goodput with all offered/rejected requests counted and per-tenant fairness**, not raw draft rates or GPU utilization alone. Protect decode and live audio without starving long prefills; do not optimize per token with global locks. SERVOPT-01..11/19..27/31/34.


**Goal:** high-concurrency and multi-task serving with stage-aware capacity/fairness and no quality regression; co-optimize resident weights, partial compute owners and speculative verifier work.

**Deliverables:** token/encoder/frame/timestep work budgets and physical resource reservations; chunk prefill, native grouped MoE, prefix-reuse admission, multi-session step-deltas, P95-aware scheduling and optional pipeline/graph variants.

**Gate:** C1/2/4/8/16/32 workload charts with TTFT/P95, aggregate useful outputs, cache hit/miss, memory/energy and no starvation/session cross-talk. Disable complexity that loses versus simple policy.

## ZG-P17 — Broader model families, adapters, embedding and reranking

**Goal:** widen registered semantics and useful models without creating another runtime; add per-request model adapters and noncausal encoder/ranking tasks.

**Deliverables:** model adapters for agreed dense/hybrid families and newer variants, encoder task/pooling, cross-encoder scoring, LoRA artifact/budget/snapshot identity, source checkpoint+quant quality matrix and text/multimodal tokenizer support.

**Gate:** independently qualified model/runner/quant/backend cells, no model-name branches in generic worker/server, correct adapter cache invalidation and source numeric agreement.

## ZG-P18 — Media TaskPrograms: audio, image, video and iterative AI

**Goal:** support independently certified non-token task loops on the same semantic/device/resource spine.

**Deliverables:** audio preprocess/ASR transcription, acoustic synthesis/vocoder, image/vision encoder, iterative diffusion/flow/timestep, video temporal and media output stages, resource/latency estimates, streaming codecs and cancellation/partial output semantics.

**Gate:** one real accepted model per claimed task type with official/independent task quality vectors; no retrofitting media into text token state. Other media programs remain planned until tested.

## ZG-P19 — Heterogeneous accelerators, collectives and remote placement

**Goal:** topology-aware compute and memory placement across local GPU/NPU/CPU and optionally remote nodes with one semantic/task/runtime protocol.

**Deliverables:** validated `SplitProjectionCompute`, `AsymmetricExpertOwnership`, layer/tensor/expert parallel and prefill/decode disaggregation where feasible, cross-driver event bridge, leases and peer session epoch handshake. Device power-fabric safety and memory bandwidth/copy cost measured.

**Gate:** local cross-accelerator safe parity with reference, shared SoC safety profile, two-device/host failure tests with no partial commit and performance where link speeds justify. Optional device/remote path cannot break single-device defaults.

## ZG-P20 — Bounded compiler/autotuner and negative experiments

**Serving precision:** Persist valid+negative hardware/model/quant/shape/driver measurements and offline candidate Pareto sets; unknown shape takes a source-qualified generic fallback, not unbounded request-time JIT or autotune. Paired repeated whole-task goodput, p95 tail and physical memory measurements authorize candidate promotion. SERVOPT-23/24/33/34.


**Goal:** improve measured kernel and whole-task performance without adding unbounded compile/search or hot-path plan churn.

**Deliverables:** bench kernel/region/task/server, offline shape/graph tactic selection, ISA/occupancy and numerical quality evidence, compiler/driver/quant keyed plan cache, guarded online policy hysteresis, repeated A/B and rejected experiment registry.

**Gate:** stable repeated whole-request latency/throughput gain on at least one true target regime and no regressions or memory faults; no "future" unbenchmarked plan auto-enabled.

## ZG-P21 — Security, trust, licensing and reproducible releases

**Goal:** audited Rust ownership, model/task/device security, safe filesystem/network inputs and installable CPU/GPU packages.

**Deliverables:** per-file license/NOTICE/SBOM policy, no proprietary copying/untrusted executable model code, signed/reproducible build decisions, fuzz parsers and snapshots, permissioned external actions, API auth, private cache, NixOS/device/driver feature isolation and GPU/NPU power safety controls.

**Gate:** release security checklist, no critical UB/resource/OS-hang behavior under supported profiles, truthful feature flags and legal artifact rights; licensed prior art remains an index, not a hidden runtime dependency.

## ZG-P22 — Full performance and quality release qualification

**Competitive evidence gate:** Pin current optimized vLLM V1 and vLLM-Omni baselines and compare equivalent source model/quant/tokenizer, numerical mode, GPU, power, prompt arrival distribution, cache/cold-warm state and tasks. Report offered/admitted/rejected/canceled requests, correct committed output meeting TTFT+ITL/media deadlines, p50/p95/p99, fairness, physical memory and energy with repeated uncertainty. No claim of universally outperforming vLLM from a single kernel/model/prompt. SERVOPT-35/36.


**Goal:** benchmark the available portable, specialized and heterogeneous paths against independently qualified external baselines, with fair source/model/task comparisons.

**Deliverables:** model/checkpoint/quant/hardware/driver/context/concurrency/sampling/power matched result matrix; logits/quality metrics and state integrity, TTFT/P95/committed outputs, energy, physical RAM, cache hit/miss, startup and cold/warm IO. Publish wins/ties/losses, and explicitly incomparable or untested cells.

**Gate:** reproducible commands and raw artifacts, statistical confidence, correct source attribution/license; no advertising universal performance superiority from mismatched models.

## ZG-P23 — Composability and future-technology evolution qualification

**Goal:** prove the primary product promise: new models, devices, tasks and storage/kernels do not require rewriting a specialized inference engine.

**Deliverables:** scripted architecture extension drills: one new SemanticOp/StateDomain, one kernel provider/device, one non-token TaskProgram, one quant/storage or API adaptation, with total touched modules, recompilation cost, source approval, release compatibility and full fallback/quality tests. Version operator/task/state/plan caches and migration or invalidation.

**Gate:** demonstrated independent extension with unchanged core service/allocator/session semantics; no nested model-specific servers. A future feature that needs pervasive unrelated changes requires interface revision before release.

---

## B. Milestones (capability, not dates)

| Milestone | Closures | Product result |
|---|---|---|
| **M0 — Feasible design** | P00–P05 slices and physical kernel checks | precise math/state, kernel and CIR resource contracts proved |
| **M1 — Usable reference/runtime** | P06–P09 | one real model direct CPU/GPU execution and basic service |
| **M2 — Stateful general serving** | P10–P12 | real hybrid-state context, expert residency and multi-session operation |
| **M3 — Extreme specialization** | P13–P15 | two contrasting complex model families, safe drafts, no independent engines |
| **M4 — Concurrency and task breadth** | P16–P18 | high-load responsive service and accepted non-token model tasks |
| **M5 — Heterogeneity and performance** | P19–P22 | qualified multi-device and Pareto/matched quality/performance evidence |
| **M6 — Extension-proof platform** | P23 | add new model/device/task without server rewrite |

## C. New model / kernel / task checklists

**Model:** checkpoint+manifest/tokenizer/quant and exact config; registered mathematics and state effects; CPU reference; tensor/logit and task quality; known backend capability; forced fallback; checked source rights.

**Kernel:** target ISA, shapes/quant, arithmetic and scratch, state effects, CIR liveness, async event fences, exact/tolerant numeric proof, complete task benchmarks, executable code provenance and fallback.

**Task:** input/output typed stages, iteration/stop/frontier, model/encoder/vocoder/denoiser as relevant, resource budgets, privacy/security, streaming/cancellation, independent source quality. Never assume all tasks use language KV.

**NPU/hardware:** optional feature under exact firmware/driver/power qualification, no automatic privileged tuning, Rust codegen feasibility and per-device compiler/source review; simulated proof not substitute for physical safety; keep GPU-only.

**Memory:** OS-safe physical reserves, all in-flight buffer generations, immutable persistent source and task state, SSD short reads and stale pointers, complete abort/quarantine under GPU/IO failures.

**Serving:** canonical task requests, authorization and bounded work queue, no hidden state leakage, p95/fairness, honest capability/mode receipt, correct output commit, independent device worker/session owner.

## D. Decision and progress discipline

Track implementation, required tests, results and remaining work in the relevant
`<plan>/TODO.md`, linked from [TODO.md](TODO.md). Update a design specification
only when the design changes. P00–P23 are architecture scope, not a completion ledger.

## v4 roadmap refinement — 2026-10-12

**Do not create new milestone numbers; preserve P00–P23.** These are **implementation clarifications** from `ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md`, not new "already implemented" features. Full protocol and review gates in `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`, `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md` and `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md` (S0-V4-A…J).

| Stable phase | v4 required change in actual acceptance |
|---|---|
| **P00** | **one Rust library + one CLI binary** to start, with modules/types not dozen scaffold crates; source/license and CPU tests. Split at 2–4 crates only if GPU compiler/unsafe build boundary proves useful |
| **P01/P07** | actual GPU ISA/source language and **async-first device-native input/worker APIs**, preallocated argument slots, safe event bridge; no host read of router IDs every token in performance-qualified hot plan |
| **P02/P04** | replace independently mutating StateDomain commit with **Prepare→PublishRoot**, source model math/effects, bounded ShapeEnv; TaskDriver stub only one minimal trusted control loop |
| **P03** | source/compute quant distinction, canonical artifact/adapter hashes and normalized source semantics, not ad hoc model code/format trust |
| **P05** | SIR + physical ExecutableBundle are the **only enduring model compiler data shapes**; RIR/KIR are transient passes; CIR is typed resource/timepoint graph inside bundle; guard ragged/dynamic shapes and avoid one variant per length |
| **P06/P08** | real model reference CPU, then GPU same TaskDriver and TaskSession; fast whole-region substitution + forced portable fallback, no interpreted IR/per-op allocations in hot path |
| **P09** | logical TaskCoordinator per session/shard, no single global queue/per-token lock, WorkerReply provisional, output after root publication and output_seq/backpressure |
| **P10** | PhysicalResourceArbiter coarse reserve and per-device local suballocators; OS-pressure fail closed, full hybrid state and prefix compatible root |
| **P11–P14** | exact source attention/recurrence/MoE and state semantics; optimized closed kernels can cover entire operator/layer while retaining portable reference; no model-specific runtime fork |
| **P15** | trained/prompt draft verifier returns PreparedRoot, sampled correction and RNG rollback before logical publish, not a kernel-side commit |
| **P16** | typed task budgets and p95/fairness/age; GPU-native StepProgram and persisted batch slots; prefill/encoder/vocoder/denoise stage classes distinct |
| **P17–P18** | **small trusted TaskDriver** for encoder/audio/media, stage-specific output/frontier; no unrestricted ExternalAction or general workflow interpreter in core |
| **P19** | epoch fencing, Prepared/Ready cohorts and safe coordinator publication with lost ACK/peer failure tests; optional durable replay must be separately designed |
| **P20** | bounded offline pure-region tuning and plan variant LRU; no context-length-by-length JIT; performance measured complete model/task under quality/power |
| **P21–P22** | source-license+Rust device program review, driver/SoC safety, resource/stream protocol, matched full-task quality/p95/energy; declare unsupported matrix honestly |
| **P23** | add new SemanticOp, KernelProvider, non-token TaskDriver, quant/IO tier **and synthetic optional ParameterVersion/TrainStep seam** without changing unrelated API/allocator/model code; training itself is future |

**Design minimality gate:** if a new model family demands a second service/session/allocator, or GPU plan compilation is mandatory per emitted token, halt that phase and correct architectural ownership. If a runtime path is correct but cannot yet avoid host synchronization, label `FUNCTIONAL` rather than `PERFORMANCE_QUALIFIED`. External source performance figures remain research baselines, not acceptance facts.


## v4.1 universal region-backend implementation crosswalk — 2026-10-12

**Keep P00–P23 unchanged, no implementation claimed.** The core bootstrap remains one CPU reference + one physically tested GPU; **do not block M0/M1 on an inaccessible TPU, wafer-scale/managed LPU or proprietary graph compiler**. Architecture accommodates these by widening the **single KernelProvider seam** into `RegionExecutionProvider` (native kernel provider is one form) accepting any semantically closed and exact state-compatible region. Candidate physical kinds: native tensor kernel, captured direct graph, AOT graph program, statically scheduled on-chip/fabric dataflow program. A completely opaque managed inference API is **TaskService interoperability**, not another kernel backend.

| Phase | Specific v4.1 addition / acceptance |
|---|---|
| **P00/P01** | identify real device capabilities and kernel/source compiler vs external vendor compiler runtime policy; local coherent ARM/LPDDR, discrete GDDR/HBM and rack fabric are distinct; no extra crates before physical need |
| **P04/P05** | `ClosedSemanticRegion` includes exact operator+state/numeric/source contract; `RegionExecutionProvider` returns region-scope/kind, lifecycle, shape, StateBridge, topology capability and fallback; RIR/KIR transient, CIR in ExecutableBundle |
| **P06/P07** | DeviceBackend can `PrepareProgram/SubmitProgram/Ready/Retire` with one AOT whole-region event rather than require CUDA-like per-op launch; native GPU remains fast and source-owned |
| **P08** | end-to-end CPU/GPU real model plus mock **graph-only AOT program** and static stateless fabric candidate exercising same SIR/TaskDriver/StateRoot; no real hardware compatibility claim from mock |
| **P09/P10** | coarse admission for compiled program SRAM/HBM/LPDDR/remote quotas, weight source/stream, graph/code cache and actual physical headroom; TaskCoordinator one logical root publication regardless of internal device micro-schedule |
| **P11–P15** | special model graphs must lower exact recurrence/sparse/MoE effects or partition safely, with real `StateBridge`; no opaque in-place state for drafts/forks |
| **P16** | phase-aware cost considers compile/load/warmup, graph region size, transfer and link critical path, shape recompile risk, whole-task quality and p95 |
| **P17/P18** | graph compiler may execute entire encoder/denoise/ASR/TTS task stage; no new model/task engine and no forced token-loop host GEMM simulation |
| **P19** | network/fabric compiler owner, actual collective topology and rank epoch fencing; no one-pointer unified rack assumption, opaque remote model APIs remain separate TaskServiceAdapter |
| **P20** | bounded graph/program cache keyed by compiler/device firmware/model/quant/shape/state schema and numeric mode; no shape explosion or per-token AOT recompilation |
| **P21** | user Rust-owned device code policy, vendor compiler/driver/runtime/IP/remote-data privacy separate; `ARCHITECTURE_COMPATIBLE` never presented as native executable `SUPPORTED` |
| **P22** | actual GB10, native Blackwell discrete/HBM, TPU AOT, wafer-static, cloud ASIC performances only where physical source/toolchain and matching checkpoint/quant/quality exist; unmeasured cells marked blocked |
| **P23** | add fake graph-only TPU-like and static-dataflow program with **same model/TaskDriver/state/allocator/server**, and independently choose another Arm64 coherent/native GPU kernel provider |

**New optional stage:** `S0-FABRIC-A…H`. Existing 24 phases, Stage0/NPU/V4 tests remain intact. New **H44…H55** research experiments cover actual hardware families and compositional failure cases; none are run. Precise normative [region-backend contract](../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md); source-specific research [platform evidence](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md).

## v4.2 distributed execution and tunneled-machine phase crosswalk (U-020)

**Keep ZG-P00…P23 unchanged; all NOT_STARTED.** Distributed execution is a *native option within the single TaskCoordinator/RegionExecutionProvider/CIR/ResourceArbiter spine*, not an independent cluster inference server. The full topology and remote protocol are [placement](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [peer/state handoff](../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md), and [updated distributed entrypoint](../docs/DISTRIBUTED_PARALLELISM.md). No node/tunnel installation or network changes are authorized by this design.

| Phase | Additional distributed responsibility |
|---|---|
| **P00–P05** | keep zero-network standalone and single SIR; define minimal compute/weight/mutable-state/cache placement view and state PartitionAlgebra/ShapeGuard without building a cluster |
| **P06–P08** | real CPU/GPU single-node vertical slice remains primary gate; mock multi-node closed region/layer with state co-location and prepared root before implementation |
| **P07/P09** | optional authenticated PeerAgent/TransportProvider using same compiled region worker and TaskCoordinator semantics, tiny checked wire protocol, explicit peer authorization; no arbitrary shell or public server |
| **P10** | remote StateShardOwner with lease/placement generation, immutable remote prefix/cache directory hints, state codec and source/adapter/model schema; coarse physical reservations per node |
| **P11–P15** | stateful attention/SSM/indexer correctness across layer/local remote, full hybrid P/D handoff and speculation target fork/replay, per-attention PartitionAlgebra for token-range shards |
| **P16** | costed placement/admission includes direct vs relayed RTT/P95/jitter, activation/KV/collective bytes, prefill/decode/encode/task shape, privacy, failure and egress; measured selection, no per-token placement optimizer |
| **P17/P18** | reuse same TaskDriver/RegionExecutionProvider on remote media/encoder/vocoder/denoiser stages, task-specific output frontier and trust; no model/task engine duplication |
| **P19** | **major distributed delivery phase**: (1) 2-node authorized layer-group+local KV split, (2) remote immutable prefix full-state hydration, (3) P/D, (4) attention+state vs FFN/experts, (5) sequence KV shards on verified operator, (6) optional tensor/head/collectives/rack, (7) safe topology reconfiguration at committed frontier; no quorum DB initially |
| **P20** | bounded measured placement search, transport link profiler, costed data transfer/compute overlap, mixed-task p95 and hysteresis, negative experiments on relayed tunnel/remote KV |
| **P21** | node identity/enrollment, tenant/model/weight permission, encrypted QUIC/TCP or approved overlay tunnel; proper physical RDMA/registered memory keys, transport framing and no automatic code/OS privilege |
| **P22** | test actual C1/C8 and long context capacity vs speed for LAN/direct overlay/relayed with source model/quant/quality, physical network link and coordinator fail/recovery |
| **P23** | add one new hardware peer, new TransportProvider and remote attention/state placement **without edits to ModelDefinition, TaskDriver, TaskCoordinator or API**; standalone startup still zero cluster cost |

**Proof gates:** new optional **S0-DIST-A…L** plus experiments **H56…H70**. CPU/GPU source reference and local fast model path cannot wait for remote access. Source prior art is in `../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md`, not an architectural dependency. User's tunneling requirement is capability support, not permission to touch networking credentials or machines.

## v4.3 integrated ModelCatalog, ModelRouter, ResidencyManager and API/UX crosswalk (U-021)

**Do not create another phase or engine**. Preserve ZG-P00…P23 and prior distributed/accelerator stage programs. Multi-model routing is a small **request/model-admission** layer using the same physical ResourceArbiter, compiled SemanticProgram/ExecutableBundle, TaskCoordinator and DeviceWorkers. The `run` CLI is simply another frontend to canonical TaskRequest, not an HTTP/gateway hop or an independent model-specific server.

| Stable phase | v4.3 model management and user-facing acceptance |
|---|---|
| **P00/P03** | clean CPU-only Rust CLI skeleton, content-addressed source/quant/tokenizer manifest and local model identity/alias registry, no arbitrary remote download; versioned safe config and source rights |
| **P04/P05** | TaskRequest validated source/TaskDriver/model profile and numeric contracts, no router logic/model-name branching in semantic IR or kernels |
| **P06–P08** | **one-command `run ./model.gguf` local in-process** with no config/daemon, actual reference/GPU model task and one committed output stream; loading progress/refusal meaningful |
| **P09** | single `serve` loopback native engine+router/API adapter, exact alias→artifact binding, GET models/chat SSE/usage/finish, authorized native model admin, Python+JS SDK conformance; no gateway process |
| **P10** | actual ResidentModel/Replica lifecycle, one global physical reserve, safe load/warm/keepalive/pin/unload, deduplicated concurrent model loads and full-session/graph/adapter/draft memory; no GTT overcommit or active KV eviction |
| **P11–P15** | multiple models with exact operators/quant/StateRoot, source-specific model profiles, trained draft+target as separate model roles, no incorrect load-balanced target model substitutions |
| **P16** | same-source replica balancing, sticky KV/prefix locality, per-model and per-tenant fairness/deadlines/queue, mixed resident model workloads and cold-load thrash prevention; router never on per-token hot path |
| **P17/P18** | embedding/rerank/ASR/TTS/image TaskDrivers and separate model roles available via truthful capability/endpoint subset; stateless Responses then qualified stateful retention, per-model adapters/config |
| **P19** | one distributed model deployment vs multiple replicas across explicitly authorized nodes; pooled/placement resource budget and StateRoot-compatible migration; remote model/source rights |
| **P20** | measured model-cold-load/warm/unload/caching and route cost, cache affinity, replica count/keepalive hysteresis and graph kernel plans, no quality or P95 regression |
| **P21** | local loopback default, external bind auth/TLS/admin separation, model catalog privacy, source pull consent, API/schema/stream security and source licensing |
| **P22** | real same-model direct `run` vs router `serve`, multi-resident/replica P95 throughput/TTFT/memory, SDK chat/Responses/embeddings conformance and source numeric fidelity |
| **P23** | add a new model profile, new exact model family and replica/router policy with **no edits to TaskCoordinator/device executor/SIR**, demonstrate minimal change radius and inspectable effective config |

**New optional Stage0 S0-SERVE-A…J and experiments H71…H86**: staged from one real model/CLI/API to multi-model physical loading and multi-model concurrent serving. Full specs [router](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) and [CLI/API](../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md). **All are design only.** User-visible command/port/profile syntax remains proposed, not a guarantee it is compiled.


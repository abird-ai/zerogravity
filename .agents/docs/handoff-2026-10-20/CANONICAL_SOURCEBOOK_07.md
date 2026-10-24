# Canonical Sourcebook — volume 7

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md

Original UTF-8 bytes: 30204; SHA256 `129bd53a896531551ef2693c9bc5b638717c4d38a751355468f67d6dc4a71df4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — Reconciled architecture and authority matrix

> **Research/provenance only (v3, 2026-10-12).** Project names, URLs, specific implementation techniques and source-author benchmarks in this document are **attributed prior art, not normative zerogravity runtime or compiler types**. The independent governing architecture is [Modular AI systems](../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [Composition contracts](../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md), and [Prior-art license policy](PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md). External projects never become a requirement to clone a model-specific inference engine; technical/measured evidence remains available for independent validation.


**Version:** 2026-10-10 consolidated baseline. **Status:** working *technical design* authorized by user request to combine the best of both programs; not independent user approval of library versions, license, numerical targets or vendor-kernel policy. Root: /home/pvl/spaces/abird/src/zerogravity. Read the detailed research note `COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md` first when refining a subsystem.

**Source evidence (strictly read-only):** /home/pvl/spaces/abird/research/inference/analysis and /home/pvl/spaces/abird/research/inference/refs/COMMITS.txt. No source directories from this evidence corpus were edited or copied as part of design consolidation.

## 0. Reconciliation charter

Retain (A) the project design's typed Semantic IR, mutable-state transaction safety, per-region fallback, global resource accounting, validated Rust kernel interfaces, modern model coverage and explicit decision provenance; and (B) the research program's concrete single-owner scheduler/worker types, KV free lists and prefix hashes, compact frontier/CommitPlan mechanics, hardware probe/ledger/refusals, quantified expert tiering, format/quant/grammar/model store, source pins, innovation experiment cards, real benchmark gates and failure/CI governance. Extend both with 2026 compiler research: stream/resource-style asynchronous resource scheduling, tensor-compiler-like region/kernel scheduling separation, bounded pure-graph rewrite lessons from a Rust graph-rewriting project, MLIR-like side-effect legality, serving-mode-aware tuning and new Rust CUDA compiler options.

**Non-negotiables**: no independent model-specific runtime fork; no semantic interpretation per hot operation; no unsafe OOM on UMA; no general fallback claim for unsupported operators; no unverified performance superiority; no unrequested precision or sampling change; no non-Rust engine dependency disguised as a Rust wrapper.

## 1. Six ownership planes (logical, NOT six server processes)

1. **Artifacts and rendering**: source manifests, GGUF/safetensors, tensor types/quant, tokenizers, templates/grammar, multimodal prep, content-addressed store. This plane defines artifact/protocol identity; no GPU plan decisions.
2. **Semantic model layer (SIR)**: shapes, operator formula, position, model state/effect transitions, sampler/latent/multimodal validity. Model adapters construct SIR; no vendor kernel pointer in model source.
3. **Compiler/optimizer (RIR + KIR + CIR)**: legal rewrites/fusions/layouts, target kernel selection and codegen, asynchronous buffer/task/timepoint scheduling, proof-of-compatibility and persistent executable bundle. Only this plane decides *how* a closed semantic region executes.
4. **Resource/placement control**: HardwareProfile, pure hardware policy/ledger and ResourceArbiter, per-domain KV/expert/prefix/n-gram/IO tiers and persistent validated layouts. Owns leases; no HTML or model math.
5. **Engine core and workers**: one logical owner for request queue, state frontiers, StepPlan, speculation and mode receipts; bounded event channels to device workers with persistent InputBatch, direct launch/capture; no global mutex per session.
6. **Service, trust and evidence**: CLI/SDK/HTTP, structured tools/streaming, API auth, metrics/benchmarks, release qualification, failure recovery.

These planes can be implemented in **~8–12 cohesive Rust crates initially**, not mandatory one crate per concept. Favor narrow type/module ownership and measure compilation/maintenance cost before splitting. External kernel compiler and GPU driver bindings remain at well-audited ports.

## 2. Canonical set of plans / names

**SIR = Semantic IR**: exact model operations and state effects (pure SSA values + linear/affine/versioned state tokens, separate resource conflict/alias model). **RIR = Region IR**: only legal operation fusion, tensor type/shape/quant/layout/precision alternatives, region cut points and backup reference equivalents. **KIR = Kernel authoring/lowering branch**: per-region tile/warp/instruction kernels in Rust or verified backend provider; *not* a second model IR. **CIR = Async Command and Resource IR**: stream/device/CPU/IO/remote command DAG, `Acquire/Release/Transfer/Dispatch/Wait/Signal/Commit` and explicit timepoint, mapping, ownership and failure edges. **ExecutableBundle = compiled payload**: region kernel handles, shape bands, graph captures, memory layouts, event schedule, state compatibility/fallback adapters and provenance.

**HardwarePolicy = pure placement/fit decision**, using exactly recorded probe and model facts; not an executable kernel graph. **KernelPlan = versioned registry/tuning artifact** of legal implementations/shape/target/numeric evidence used by RIR and CIR. **StepPlan = per-cycle scheduling command** linking concrete request rows to already-compiled executable variant and reserved leases. **ModeReceipt = truthfulness report** per request/response identifying actual fidelity/paths/degradations. The word `ExecutionPlan` alone is ambiguous; avoid introducing a monolithic object owning all semantics, compilation and live queues.

### Control-flow diagram

```
Artifact -> verified ModelRecipe -> SIR
                                -> RIR -> KIR / kernel candidates
                                        \
HardwareProfile -> HardwarePolicy ----> CIR (resource/timepoint/IO DAG)
                                              |
                                  executable immutable bundle
                                              |
Client -> EngineMsg -> EngineCore -> StepPlan -> WorkerBackend (GPU/CPU)
                                           |        |
                                       ResourceArbiter owns buffer leases
WorkerDelta/Events -> EngineCore -> commit valid state -> EngineEvent -> client
```

One critical question: can a policy change resource placement while preserving the plan's state layout? If yes, reserve/move with a new epoch and validated adapter. If not, transition at a checkpoint/replay boundary; never mutate a live device graph's captured raw pointer to a different stale type.

## 3. Model and optimization ownership

`ArchDef/ModelRecipe`: parse and validate architecture config/tensor map; declare semantic operator sequence, position/state/quant and supported modality; expose reference parity tests. `ModelSpec` is an optional helper to produce SIR for complex families, not responsible for GPU kernel selection. Thus **remove historical `ModelSpec::kernel_plan(hw)` as the final placement owner**; instead `KernelProvider::candidates(SemanticRegion, TargetCaps, NumericContract)` and `Compiler::compile()` perform lawful selection.

This is the primary long-term extensibility advantage: future recurrent/sparse/diffusion or MoE architecture adds op schema/reference/model adapters, not an entire serving/runtime fork; new GPU adds kernel/provider/driver caps, not model-name conditionals. A completely unknown mathematical op still requires a semantic implementation and cannot fall back by mere GGUF reading.

## 4. The general fallback: exact boundaries of the guarantee

**Can run:** model's tensor artifacts + tokenizer + position/sampler definitions validated; every semantic operator has a portable reference lowering and available device/CPU execution; enough physical RAM/time; correct state conversions. Missing an extreme kernel -> lower only the affected closed region to tuned/generic/reference, unless its state layout/IO policy requires a plan boundary. CPU fallback of one op may be very slow because of GPU→CPU transfers; compiler's memory estimate must expose that cost and an explicit capability/performance warning.

**Cannot promise:** arbitrary unknown HF `trust_remote_code` formulas, all GGUF quant IDs, any context length without memory, all hardware drivers, same speed for all device targets. Distinguish `RecognizedArtifact`, `SemanticallySupported`, `RunnableOnDevice`, `ValidatedModel`, `PerformanceQualified`.

## 5. Scheduler + worker seam adopted from research

- One logical owner engine-core (normally dedicated Rust OS thread) owns `PendingRequest`, `Waiting/Running`, scheduling budgets, logical Session/Frontier, runtime policy and commit decisions.
- Frontends use bounded `EngineMsg` and receive `EngineEvent`; worker owns GPU context/streams/graph/expert resident tables, applies `StepPlan` via persistent `InputBatch` diffs, returns `WorkerDelta` and event times, never mutates engine's logical request objects from foreign threads.
- One unified **token compute budget** plus separately accounted input/encoder/memory budgets. Chunked prefill, decode and speculative verification all charge actual target work. Preemption/recompute/priority fairness determined by `SchedulePolicy`; no arbitrary global serialization.
- Research two-in-flight-step overlap is a *candidate*, only legal where StepPlan_n does not depend on not-yet-committed tokens or grammar state from n-1. GPU pointer versions and speculative state drafts must be isolated; dynamic overlap is a proof-gated feature.
- Generic: paged block `KvPool` (intrusive free queue, chained hashes), optional radix prefix index, recompute-only preemption if safe; Extreme: contiguous slab/reserve-all/no-preempt when measured advantageous. These are two `KvPool` strategies under the SAME global ResourceArbiter, not two uncoordinated allocators.
- `BatchPolicy::Coalesce{window}` can group 1–2 serial/verify requests but is not assumed universally better; dispatch native grouped MoE where GPU-side router and device slot table support it. "Concurrent HTTP" is not inherently native batch.
- ModeReceipt accompanies metrics/protocol diagnostics; exact availability of user-visible fields separate from hiding private prompts.

Adopt from research `DraftSource`, `VerifyMode`, and `CommitPlan { FullDirect, Prefix(n), Replay(n) }`. Implement **inside** the project's typed state transaction across GDN/KDA/KV/indexer/ngram/RNG/modality rather than assuming O(1) counter repair always restores physical state. Frontier snapshots may be cheap references if immutable buffers/generations support it, or require actual copy/recompute.

## 6. Memory and caching ownership

One `ResourceArbiter` accounts mandatory weights, optional expert/cache pools, prefix state, recurrent/indexer, draft/MTP, multimodal, scratch, staging, IO, host OS/cgroup and physical memory. Each resource class gets its own policy; expert slots and KV pages have distinct identities but share a physical reservation pool. UMA GTT/addressability ≠ free physical LPDDR.

Research innovations to retain:
- paged block pool with intrusive metadata/free lists and cheap no-prefix-hit behavior; optional radix index over SAME allocation pool;
- grouped KVCacheSpecs by attention/recurrent/compressed type, prefix-hit intersection and finalization alignment;
- extreme contiguous KV allocator with reserve-all under low-concurrency workloads;
- expert slot slabs by size/layer, gen-tagged leases, device-side (layer,expert)->slot, host-mapped bounded miss ring, no host wait on hit;
- fill-only startup quotas, per-layer recency/LFU variants, hotlist seeding and prefetch hints with host publish fence;
- CPU expert lane calibrated by measured shared LPDDR/PCIe bandwidth;
- NVMe one-blob expert read, disk-only row lookup with dedup bounded staging;
- per-tier KV precision and source; quantized KV defaults only after model quality tests;
- scratch borrowing from optional cache when live references quiesced, not overlapping raw allocations.

Project improvement:
- **State-domain general resume coordinator**: each KV/recurrent/sparse/ngram/vision domain implements `can_resume(frontier)`, compatibility key and restore/recompute cost; select common legal prefix frontier and decompose misses safely, generalizing research/vLLM hybrid two-group assumptions. Cache identity includes rendered **tokens**, model/quant/schema/numeric/pos/modality and tenant salt with collision-resistant SHA256/BLAKE3-256.

## 7. Kernel authoring policy and new CUDA-Rust evidence

The user requested entire Rust-owned engine including kernels. Historical research recommends `cubecl=0.11.0` plus a `zg-vendor` module of manually ported or **vendored C++/HIP** quant kernels. **Not silently adopted**. Source Rust policy and license review precede selection.

Define `KernelCapability` interface independent of compiler:
`{semantic region id/hash, tensor dimensions, dtype/quant/layout, batch/context band, numeric/state effects, kernel scratch, supported target/capture, input/out leases, ISA code version, reference/quality proof IDs, measured performance}`.

Providers may include:
- Rust CPU ref/SIMD;
- CubeCL/CubeK portable cross-backend;
- Rust-targeted AMD wave32/WMMA/MFMA extensions or compiler backend;
- NVIDIA's **stable Rust tile DSL** (stable Rust 1.89+ and CUDA 13.3+ per Sep 2026 docs) for tile kernels;
- NVIDIA **Rust SIMT→PTX frontend** (pinned nightly alpha for SIMT PTX);
- Rust shader/portability toolchains for Vulkan fallback;
- **optional** vendor GPU math library FFI after explicit user policy decision, with no underlying foreign inference runtime.

Kernel plan can contain multiple alternative source/codegen providers for the **same** semantic operator. Compiler chooses by real target capabilities, shape, graph/eager mode and quality. Model code must not know which provider built a given region. Source attribution and compiler/driver ISA pins required. Adopt safe Rust wrappers and narrow unsafe GPU FFI boundary; a Rust-facing wrapper around an entire non-Rust inference runtime is prohibited.

## 8. Portable first, specialized immediately when justified

Each model's reference semantic graph is mandatory. Tiering:
- **General Reference**: minimal CPU kernels, correct state/format; foundation and debug oracle.
- **Optimized Generic**: backend-tuned GEMM/attention/quant, paged state, grouped MoE, native batching.
- **Extreme**: model-pattern x actual hardware shape specialized wave/intrinsic kernels, whole-layer or whole-model fused subgraphs, resident-route graph islands, disk/CPU expert hybrid/IO overlap as needed.

The progression is not three separate engine executables. Plans select legal regions; layout/state incompatibility can require a **whole plan** switch/replay, which is acceptable when clearly reported. No abstraction can forbid a Rust-authored device-specific whole-region kernel.

## 9. Tune kernel variants and joint online policy at different timescales

- **Preload:** semantically validate graph, compile/lookup legal candidate regions, get physical hardware probe, choose safe MemoryLedger. Cold compile constrained by startup policy.
- **Offline / explicit `tune`**: measure kernel/region alternatives and whole HTTP/request workloads, including eager versus graph replay, power/cache/thermal and model quality. Store versioned `KernelPlan` with provenance/evidence; avoid per-token timing selection.
- **Online**: scheduler chooses among approved variant bands, adjusts batch/prefill quantum, cache reservations/spec depth only at stable request/epoch boundaries with hysteresis. No unbounded JIT or e-graph rewriting while serving. Optional guarded JIT cache on shape miss compiles outside hot path then publishes after correctness gate.
- **Search**: tensor-compiler/graph-rewriter-style exploration on **pure regions only** and bounded compile budget, plus symbolic/dynamic shapes; never make equality-saturation a required dependency for ordinary model load. Stream/resource-style CIR schedules streams/event/memory rather than trusting raw GPU graph to infer all IO and state dependencies.
- **Cost objective**: critical-path wall time or tokens/J under physical budget, P95 and quality constraints; report Pareto frontier, not one global "fastest" number.
- **Performance claims**: follow research source pins and experimental matrix; changed checkpoint quant/ctx/draft/driver not matched comparison.

## 10. Operational product capabilities absorbed

The earlier project plans are under-detailed on these; the reconciled baseline includes:
1. GGUF canonical input and Hugging Face safetensors readers; per-file quant dialects/validation. An optional **derived acceleration pack** can bundle GPU-native tensor packing/expert blob index and KernelPlan; not required for correctness or load. Research label "APK" is internal and potentially confusing with Android packages; public extension remains OPEN.
2. ModelStore content-addressed blob/manifest list, atomic downloads/partial resumption/checksum and GC, optional offline mode; avoids repeated ad hoc model downloads.
3. Tokenizer GGUF pre-types, exact bytes, minijinja role/tool rendering, streaming parsing, grammar-FSM/GBNF or equivalent Rust engine; do not freeze external dependency versions until proof.
4. Embeddings/PoolingSpec and scoring/rerank endpoints are a **separate model runner mode** (no autoregressive state), not a fake chat route. LoRA/adapters require per-request generation and cache invalidation, not just an option flag.
5. Multimodal encoder state and span handling with independent encoder budget, image-content identity, M-RoPE and non-causal span chunk safety. Model generation modes outside text/embeddings are future semantic extensions, not v1 promise.
6. API OpenAI chat/completions/models, tokenization, tool/reasoning/JSON grammar, metrics/doctor; Anthropic/Responses only when conformance tests pass; no public unauthenticated bind default, no implicit code execution.
7. Real bug/security: historical research disk-KV schema labeled "48-byte" but fields sum **54 bytes**; its five-character magic typed `u16` is invalid. Do not implement verbatim. Define new versioned, checked serialization with cryptographic identity and no persisted raw pointers before implementing cache files.

## 11. Per-model/hardware specialization contract

**Model family descriptors:** Llama/Gemma4/Qwen3.5+/Qwen3.8-Flash-Next/GLM-5.3-Flash/DeepSeek V4 variants and later encoder-only/multimodal. Derive exact layer geometry and token/quant manifest from pinned official checkpoint, not a name template.

**Qwen3.8 Flash-Next gfx1151:** GDN, QSA, routed 512 top-10 MoE, ngram table as bounded disk rows, trained MTP, fused gated residual, attention long-context GPU selection. Validate optimized quant against Gufo and an independent tuned closed reference with matched model/precision where possible; different quantizations produce separate cells.

**GLM-5.3 Flash gfx1151:** KDA per-channel state, sparse MLA/DSA, mHC, 288 top-8 × 42 routed layers, expert resident/tier budget, MTP only on measured net-benefit workloads. Maya and DwarfStar quantities differ; 92/96 vs 128GB fit must be separately probed.

**DeepSeek V4 class:** learned time-compressed attention, local raw and 4×/128× historical rows, sparse indexer and DSpark when compatible, disk-only Engram. Do not use ordinary GQA for unknown compressed-KV semantics.

**Dense/Gemma4/Qwen3.5:** LoRA and vision as separate extensions, GQA + sliding window + GDN/full attention, optimized paged KV and high-throughput batching; do not starve ordinary model support by building only exotic kernels.

## 12. Evidence, correctness and feasibility gates

Research Stage 0, enhanced with IR proof:
- S0-H hardware/driver probe and memory safety; physical gfx1151 kernel feasibility (CubeCL versus target Rust variants) and CUDA modern compiler discovery.
- S0-C CPU reference tokenizer/GGUF/quant and first dense model; independent official/llama.cpp comparison.
- S0-I minimal SIR typed effects + RIR legality + CIR async lease/event simulation; synthetic mixed KV+GDN+indexer rollback.
- S0-S scheduler/KvPool/Frontier/CommitPlan simulator, fuzz >=10^6 generated transitions if runtime budget reasonable; strict mode and mode receipts.
- S0-K KernelPlan shape-candidate artifact and eager/graph parity with explicit failure retirement.
- S0-P model-specific portability test: force disabling only one specialized op region, retain other optimized ops and prove whole-model state.
- S0-Q quality/bench harness matching checkpoint/power/context and sampling; ABBA strict-bit where meaningful and tolerance/quality otherwise.
- S0-U usability/portability: tiny full vertical slice artifact→CPU/one GPU→token→API with clear typed error, not a 25-crate placeholder scaffold.

No compiler/pass/API contract is "frozen" before a green vertical slice proves the proposed abstraction isn't a performance/correctness liability. Performance targets from research are **conditional goals**, not observed zerogravity achievements.

## 13. Reconciliation ledger of historical research decisions D-01..D-20

| Research council decision | Reconciled status | Rationale |
|---|---|---|
| D-01 `zg`/namespace/store | Candidate product convention; zerogravity name confirmed; binary/paths/flags remain open | user's explicit choice only name/location |
| D-02 engine single logical owner | **Adopt as working implementation baseline** | isolates mutable scheduler state, avoids global locks |
| D-03 paged KV/optional radix/extreme contiguous | **Adopt as pluggable variants with full-state eligibility** | good measured mechanisms, need SIR state compatibility |
| D-04 real extreme reserve-all/no-preempt seam | **Adopt** | no pretend transparent scheduler flag |
| D-05 GGUF required, optional accelerated pack | **Adopt concept, extension open** | portability + specialized layouts |
| D-06 CubeCL + C++/HIP vendored kernel escape | **Modify**: CubeCL one candidate, Rust-owned target providers; vendor FFI separate policy | strict user Rust-language intent and 2026 CUDA-Rust tools |
| D-07 token budget/two-step overlap | **Adopt baseline**, overlap proof-gated | vLLM proven but state/grammar dependency not always safe |
| D-08 generic recompute and extreme reserve admission | **Adopt selectable policies**, no universal default | resource/latency model |
| D-09 Frontier/CommitPlan speculation | **Adopt inside typed state transaction** | O(1) logical frontier not necessarily physical state |
| D-10 pure plan + auto-extreme | **Modify**: pure resource policy, compiler and live StepPlan distinct; extreme automatic only after measured validation | coherent timescales |
| D-11 content-addressed artifact store | **Adopt concept** | product quality and reproducibility |
| D-12 single GPU/tier v1, TP later | **Adopt staged non-goal**, exact v1 not frozen | tractability |
| D-13 event transport / run=serve | **Adopt typed protocol**, whether local run speaks HTTP internally still open | avoid redundant network costs |
| D-14 MIT/Apache, pinned MSRV/crates | **Open** user license/build policy; source list retained as research | not explicit user choice; compiler landscape changed |
| D-15 loopback/fuzz/opt-in telemetry | **Adopt conservative security requirements** | safety |
| D-16 platform matrix | **Proposed ordering only** | user cares gfx1151 and general portability; actual hardware probe |
| D-17 model tiers | **Modify** include explicit Gemma4/Qwen3.5+ and user modern scope | old research largely Gemma3/Qwen3 |
| D-18 tokenizer/grammar stack | **Adopt fidelity requirements**; dependency versions subject to proof | serving correctness |
| D-19 quant/imatrix quality bands | **Adopt** | low-bit specialist quality |
| D-20 multi-lane CI + verified performance | **Adopt** | essential proof |

This is an **engineering council reconciliation**, not retroactive user signature on every table row. All future user policy choices are tracked in `DECISION_REGISTER.md`; tests/code override hypothetical mechanisms.

## 14. Defects or gaps found and disposition

- Research universal-fallback promise impossible for unknown semantics → constrain to registered ops and resources.
- Research binary disk-KV header 48 vs 54 bytes / magic width → do not port; redesign checked on-disk schema.
- Research dated local hardware claim (3060/Cezanne/27 GB) → re-probe real active development hardware; do not demand purchase automatically.
- Research 75–125 engineering-month scope with ~93 engineer-day "4–6 weeks pair/6–8 solo" Stage 0 → separate hard gates, optional experiments, critical path and staffing; numbers unverified budgets.
- Research 8×L40S DeepSeek DwarfStar TP benchmark shown in GLM comparison row → keep architecture evidence, **not a matched GLM performance baseline**.
- Research CUDA vendor escape conflict with Rust-only → open product policy, architect Rust-authored alternative provider.
- Old project research atlas missing latest Rust CUDA toolchains and stream/resource and rewrite research → add references to new synthesis.
- Original project lacks LoRA, grammar, ModelStore, embeddings and formal CIR → newly documented subsystem ownership and roadmap coverage.
- Both designs too optimistic about generic model coverage → named capability matrix and errors instead of "all any runtime models".
- Original research "auto-extreme for all eligible" without yet measured quality/cost → require validated plan, safe memory, performance and effect contracts.
- Old 24 phases assume first new kernel/states serialized later → put small semantics/effect/CIR/CPU/model and hardware feasibility side-by-side, no monolithic compiler first.

## 15. Open technical choices kept deliberately unfrozen

License, exact Rust driver/vendor math FFI policy, primary GPU compiler for gfx1151 and CUDA, initial quant/formats, concrete binary name/CLI, crate split, first end-to-end accepted model and hardware, public service conformance, remote auth, persistence defaults, target SLO and numerical tolerances. Decisions must be made from product requirements, source-license audit and physical proof, not because an older research table says "resolved".

## 16. Next-step success test for this design

Have an external agent add (A) a novel recurrent attention op using a fresh semantic reference and one GPU fallback, and (B) a new NVIDIA kernel provider compiled in Rust, **without editing** unrelated HTTP/engine-state/weight-cache code. It must pass mixed prefill/decode/verify, state snapshots, format/model parity, GPU memory leases and plan introspection. If the change touches every subsystem, our conceptual modularity failed even if a benchmark happens to look fast.

**Core rule:** semantics must remain correct as technology evolves; optimization mechanisms replace implementations, never redefine the model. When a measured specialized plan wins on one hardware/workload, reuse it there. When it loses or cannot be validated, the portable model should still run and report the degraded mode honestly.

## 2026-10-12 addition — Gufo cooperative NPU prefill reconciled without new runtime fork

The user has now explicitly selected two independent Gufo sources: the existing upstream `gufo-org/gufo`, and **`mikealanni/gufo` at pinned commit `8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`**, whose new work offloads selected projection row slices to XDNA2 NPU while Radeon iGPU continues work. This is **not** a dGPU+iGPU routed MoE expert split or a full NPU model runner. Both are legal **compute owner placement modes** in common SIR/RIR/KIR/CIR, the same EngineCore, and the same ResourceArbiter.

Update ownership: SIR/GDN/QSA/shared FFN exact math unchanged; RIR validates disjoint projection output rows; KIR advertises Rust-authored NPU AIE/XRT program candidate (blocked unless feasible), CIR imports pinned XRT BO memory, emits host/device timepoints and waits at first required consumer; DeviceWorker joins and EngineCore commits, GPU-only reference fallback everywhere. A future ASR/TTS/image/video `TaskProgram` shares same mathematics and memory but carries a different task-specific control/output loop rather than simply treating all inference as text token sampling.

**Policy and safety:** the Reddit author reports whole-machine hangs from concurrent NPU+GPU power/fabric transitions. NPU is experimental/opt-in and never auto-enables on unqualified hosts or changes system clocks/IOMMU/suspend privileges. Gufo's AIE/Python/C++ device kernel is not an all-Rust-owned implementation merely because a Rust wrapper calls XRT. Work on Rust AIE codegen or record blocked; no weakening of U-003 silently. These safety, model-precision and performance gates are in [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).


## Multimodal scope review — v5 proposal (14 October 2026)

The current v4.3 contracts remain authoritative. The independently researched [unified multimodal design proposal](../docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md) and its [source-model family atlas](../research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md) identify six small, orthogonal contract refinements: typed media timebases, source-verified compound models, non-token iteration/decision drivers, multi-track synchronized StateRoot, separate tensor source codecs/transforms, and explicit native-versus-external provider capability. They are **not additional enduring IRs or replacement schedulers**; they are proposed additions to ValueType, ModelDefinition, TaskDriver, StateDomain and region qualification. No external media model support or new model architecture code is claimed as implemented. Existing Qwen3.5 CPU 8K admission and P07 source-correct GPU region implementation continue unchanged.

````

## Source: .agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md

Original UTF-8 bytes: 29689; SHA256 `15dea9e967c4baafc3a6673df1f6bae7e0667585db18906592a3b0803ce5b88a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — v4 critical architectural review, challenges and resolution record

**Review date:** 2026-10-12. **Scope:** a fresh cross-cutting engineering review of the complete v3 Markdown architecture, subsystem specs, 24-phase implementation roadmap, 31 experiments, task/device API sketches, state/memory/safety contracts, and the prior-art research corpus. **Classification:** design review and adjudicated architecture proposals, **not code or benchmark results**. Execution rules live in the [minimal spine](../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler contract](../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md). Earlier v2/v3 document versions are available in Git. The [master](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) carries design context; [active TODOs](../plans/TODO.md) and the accepted checkpoint control current completion. Historical review proposals and open-policy labels below are not a new implementation mandate or a reversal of later decisions.

**User objective:** achieve the highest plausible performance/quality/productivity Pareto frontier for **composable AI execution**, retaining a very small mandatory core, clean future model/operator/device/quant/task extensibility and no new model-specific inference engine for specializations. Performance supremacy is *unverified*, not an architectural guarantee.

## 1. Research methodology and genuinely material new evidence

The review used the existing **read-only** `research/inference` source investigation, v3 live project specs and independent first-party public design/implementation documentation. Source names are **deliberately confined to this research record** and the existing prior-art/license atlas; normative v4 design uses independent mechanism names.

- **Asynchronous inference hot path:** [vLLM Model Runner V2, March 24 2026](https://vllm.ai/blog/2026-03-24-mrv2) separates generic model logic from GPU-native input preparation, device-side rejection sampling, CPU/GPU overlap, and asynchronous output stream. It explicitly calls out host synchronization and CPU-producer/GPU-consumer buffer race risks. The lesson is **avoid mandatory CPU round-trips between GPU steps while retaining a logical owner and causally ordered state**; do not blindly reproduce one library's runner API.
- **Explicit resource lifetimes:** a streaming-resource dialect of a known compiler IR models async work placement, timepoints, allocation/liveness and target-specific resources. The lesson is **model device work and resources separately from mathematical graph**. Do not implement the entire external compiler as zerogravity's runtime.
- **Portable semantic schema and shape dynamism:** [StableHLO compatibility](https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md), [dynamic dimensions](https://github.com/openxla/stablehlo/blob/main/docs/dynamism.md) and [spec](https://openxla.org/stablehlo/spec) demonstrate versioned operator contracts, static-rank/dynamic-dimension handling, runtime guards and effect tokens. The lesson is **explicit shape predicates, schema version compatibility and effect verification**, not source project dialect inheritance. This document does **not** propose embedding StableHLO C++ in the core or claiming its exact compatibility window for our own artifact.
- **Serving fairness:** [Fairness-aware chunked-prefill research (June 2026)](https://arxiv.org/abs/2606.09061) identifies latency jitter/head-of-line blocking from rigid token budgets and studies stage time constraints. The lesson is **scheduler budgets as target-time/resource/fairness tradeoffs** rather than unconditional 128/2048-token chunks or only aggregate throughput. The paper's author results are not independently reproduced here.
- **Technical risk:** wide model/inference source corpora show that per-model fused kernels, hetero expert ownership and GPU+NPU projection slicing can be fast under particular shapes/precision; they should be optional providers with references, typed state boundaries, XRT/GPU event ownership and physical hardware safety qualification, not the new core.

The review does **not** claim other systems lack some technique or that new 2026 software has been compiled/tested on our own devices.

## 2. Severity-ranked findings: concrete faults in v3

Severity indicates the historical *cost of freezing the wrong interface*, not a discovered executable bug. This review preceded the working engine; its findings explain the contract choices rather than assert today’s implementation status.

| ID / priority | v3 design problem | Consequence if coded literally | v4 correction |
|---|---|---|---|
| **R4-01 Critical** | `StateDomain::commit(candidate,...)` can mutate/publish individual domains before the promised all-domain atomic `StateTransaction`. | partial logical KV/recurrent/RNG state on failure; impossible true all-domain commit | domain **Prepare**/rollback, immutable state-root assembly, one **PublishRoot** by single session writer, retire old root after device-event quiescence |
| **R4-02 Critical** | CIR includes `Commit{transaction}`, while TaskCoordinator/EngineCore independently owns commit. | two commit authorities and ambiguous device/host visibility | CIR signals `Ready/Failed` provisional work; **only coordinator publishes logical state/output**; device command completion never means user-visible commit |
| **R4-03 High** | Logical single owner is sometimes conflated with physical owner for all request/session queues, KV ledger and allocations. | one process-global lock/queue bottleneck and synchronous allocations under C32+ | single writer **per session/shard**, independently concurrent model/device workers, hierarchical resource reservations; fast device-local allocator owns physical suballocations |
| **R4-04 High** | Four IR names SIR/RIR/KIR/CIR are presented as separate, always mandatory compiler data formats. | compiler framework complexity, version explosion and per-step overhead | only **two durable intermediate contracts** initially: semantic program and physical executable plan; RIR/KIR are **transient pass/candidate views**, CIR is the typed command/resource portion of executable plan; no general compiler framework dependency |
| **R4-05 High** | `TaskNode::{Sequence,Parallel,Select,Iterate,ExternalAction...}` effectively includes a generic app/agent workflow DSL in the hot runtime. | scope creep, memory/security complexity, task graph interpretation every token | **small trusted Rust TaskDriver** with `poll/prepare/step/commit/emit` and optional small bounded stage DAG; external tools and agent workflows in a **separate permissioned application layer** |
| **R4-06 High** | v3 says “no per-token IR interpreter” but does not define how dynamic routed experts, streaming loops, shape changes and draft accept/reject avoid CPU sync. | fast low-level kernel nullified by CPU copy/graph breaks and old-input races | prepared GPU-native **StepProgram** with index/generation buffers, mutable parameter slots, device routing/sampling where valid, host control only at true inter-stage/commit boundaries |
| **R4-07 High** | Multiple identities (`KernelPlan`, DevicePlan, `ExecutableBundle`, `ExecutionPolicy`, `StepPlan`) but unclear what persists/changes and what is in cache key. | recompile storms, stale device pointers, graph-cache explosion and misuse after driver/model upgrades | identity matrix + canonical plan fingerprints; immutable compiled variants, mutable bounded parameter slots; runtime step only binds IDs/shape guards/state epoch. LRU and cost of cold JIT explicitly budgeted |
| **R4-08 High** | Physical memory budget is described as a universal exact guarantee despite OS/cgroup/GTT changes and kernel/driver allocations. | false no-OOM promise, reserve bottleneck or unsafe peak scratch | global **admission reservations + safety headroom**, per-device memory pool, runtime pressure/reclaim/watchdog, fail-closed allocation; cannot guarantee OS never changes memory unexpectedly |
| **R4-09 High** | Dynamic dimension, ragged batch, sparse route and chunk shape bounds aren't a first-class typed guard. | treating `M=4096`/exact quant/shape as generic; wrong kernel call or endless recompilation | bounded symbolic `ShapeEnv` / runtime guard contract, segmented ragged rows and shape bands; decompose/fallback on shape miss, JIT off critical path |
| **R4-10 High** | Numerical `exact sampled distribution`, “bitwise full logits”, “reference” and “validated tolerant” mix distinct guarantees. | impossible cross-backend bit-identical promises and accepting numerically changed router top-k | explicit math, numerical/quality and stochastic sampling **separate contracts**; same-plan deterministic replay distinct from cross-plan tolerance; intentional approximation only opt-in |
| **R4-11 High** | Distributed “atomic same frontier” underspecifies partial network failure, lost ACK and external streaming. | dual ranks committing different state, duplicate/phantom output on retries | epoch fencing, prepare/ready/commit protocol at logical coordinator, idempotent retry/replay, outputs after durable or volatile commit according to declared guarantees; no magical exactly-once network semantics |
| **R4-12 High** | Output streaming tied implicitly to state publication with no explicit transport delivery contract. | duplicate token/audio chunk on disconnect/retry, exposed uncommitted media | ordered `(session,commit_epoch,output_seq)` emission with at-most-once live stream, optional durable replay/ACK if requested; exact-once end-to-end **not guaranteed** |
| **R4-13 High** | Training/backprop and mutable model weights absent despite “all AI execution”, while requirements could imply immediate support. | either unnecessary initial optimizer/training bloat or architectural barriers later | keep inference-first **read-only model core**; reserve `ParameterVersion + GradientProgram` extension seam for training/fine-tuning with separate task loop and versioned optimizer state, **not default release commitment** |
| **R4-14 High** | TaskProgram's `ExternalAction` appears inside core while security forbids arbitrary tools. | inference engine becomes privileged workflow executor | move actions to app/agent integration boundary with authenticated intent and sandbox; core only accepts trusted typed task/data requests |
| **R4-15 Medium** | Optional plugins and 8–12 crates appear mandatory in roadmap examples. | unnatural trait dispatch/many dependency seams and long compilation | start **one library + one binary** or 2–4 cohesive crates; statically linked built-in providers, expand only with measured compile/security/release boundary |
| **R4-16 Medium** | “one semantic op” can be a giant fused black box or tiny primitive without guidance. | can't optimize/fallback at useful layers; wrong effect hiding or variant explosion | two-level op vocabulary: high-level **semantic regions** retain attention/recurrent/task state identity; bounded low-level pure tensor primitives/transformations inside region; only required equations model-specific |
| **R4-17 Medium** | Specialized cache and state snapshot/effect checker don't specify immutable source artifact vs mutable LoRA/weight generation placement. | cross-adapter cache contamination, drafter mismatch and executable invalidation | separate immutable checkpoint/packing, adapter generation and parameter version; central `CompatibilityKey` includes source+task+state/numeric/geometry when needed; version changes invalidate unsafe state/plan reuse |
| **R4-18 Medium** | All optimizer candidates assumed to be searched offline before serving; cold model/shape support unclear. | long load time, surprise cold compile and missing all-purpose baseline | always provide safe portable kernel/reference; load only needed shape bands, deferred background/JIT only if explicitly requested/controlled; no runtime work waiting for expensive search |
| **R4-19 Medium** | Benchmark claims use “committed tokens” as universal unit, despite audio/image/video tasks. | false comparability of throughput | define `WorkUnit` and `OutputUnit` for each TaskDriver; normalize energy/latency per **meaningful committed task output** with optional device utilization, not fake token conversions |
| **R4-20 Medium** | Broad device/topology/NPU safety modeled, but qualification/unsafe transition has no invalidation and rollback semantics. | stale “safe” program after firmware/power change and dangerous auto run | device attestation fingerprint and runtime safety gate; on change disable/re-evaluate profile, preserve CPU/GPU-only; no automatic privileged mitigation |
| **R4-21 Medium** | Data/model representation has no clear stop at “inference runtime” boundary vs tool-rich agent platform. | duplicative security sandbox/state engines, misses specialization priorities | core provides typed programs/effects/outputs; external apps can orchestrate multiple calls and tools via independent permissioned adapters; no embedded unrestricted agent automation |
| **R4-22 Medium** | No mandatory "negative architecture test" when adding future model types. | creeping model-name conditionals and duplicated runtime | explicit change-radius budget, second model/new device/non-token task/fault injection, architecture acceptance must fail if new specialized server/allocator required |
| **R4-23 Medium** | Version policy lists many keys but lacks canonical wire/snapshot/compiled cache separation. | serializing raw pointers, breaking upgrades, stale implementation state | specify persisted artifact/semantic/state schemas separately; **never persist executable pointers**. Compiled bundle caches expendable and invalidated by driver/compiler; snapshot format versioned and verified independently |
| **R4-24 Medium** | Precision and performance optimization objective lacks explicit counterexample: reduced top-k is faster but changed trained math. | accidental quality/sampling change | mark optimization as `SemanticallyEquivalent`, `NumericallyDifferentButValidated` or `ModelChangingApproximation`; only first two auto-eligible under configured numeric policy; latter only explicit opt-in |

## 3. Primary adjudications — fewer concepts, not more mandatory infrastructure

**D4-A — Minimum runtime ownership:** three core authorities only: **Model/Task program ownership (immutable)**, **per-session mutable TaskCoordinator logical state**, **PhysicalResourceArbiter admission and per-device leases**. Compiler and backend providers are pure/adaptable; they cannot independently mutate session state. TaskDriver is a trusted control policy, not yet a DSL runtime.

**D4-B — IR minimization:** SIR persists as canonical mathematics/effects; RIR/KIR are *compiler-internal transient views* (no new universal serialized formats); CIR is a compiled **command and resource DAG within executable plan**. Runtime only consumes direct plan handles and bounded parameter slots. Keep terms to communicate phases but don't force the codebase into four separate interpreters or crate registries.

**D4-C — Safe session state publication:** StateDomains do **not** expose a public mutating `commit()`; instead return immutable candidate versions, with `prepare` and rollback/quarantine, and a single atomic root publication when all required device events are ready. Old root stays visible on aborted speculative branch, IO failure or model repair. Streaming output only after publication.

**D4-D — Async-first hot loops:** host TaskCoordinator must schedule and own commit logically; a kernel/graph can compute multiple dependent token steps or verify candidates entirely device-side if verified, yielding a *single prepared frontier summary and output slice* to host after event completion. No CPU per-token router/grammar/quantize barrier unless model exactness or user-visible output requirement demands it.

**D4-E — Executor modularity:** KernelProvider capabilities include op/region semantics/shape/quant/numeric/device regime. A provider can be a hand-tuned whole model region, but cannot own its own API, state authority, store, allocator or task loop. Device programs and XRT/GPU driver FFI are independently reviewed; GPU-only path unchanged without experimental NPU.

**D4-F — Hierarchical memory:** central budget and admissions; local fast memory pool/lease/liveness obey global reservations and pressure. Don't allocate persistent weights/device graph per token. Prefix cache and expert tier have specialized eviction, one shared physical ledger.

**D4-G — AI breadth only through typed tasks:** a small TaskDriver and task-specific stage transitions handle text, embedding, ASR, TTS, image/video and **optional future gradients/training**. The memory/compiler core does not become a model training optimizer, workflow orchestrator or arbitrary external-tool host in first release. External tool calls belong above core API.

**D4-H — Portable first / specialized always optional:** source checkpoint/ModelDefinition and reference mathematics are authoritative; new hardware or model attaches via narrow capability interface. Runtime can decline unqualified kernel/shape/power and still function on generic backend where all registered semantic ops and physical memory suffice.

## 4. Performance budget and hot-path invariants

Performance-oriented design must be stated as measurable *budget categories* rather than invented universal microsecond numbers:

- **Hot dispatch:** cached `ExecutableBundle` maps shape/phase to compiled code handles and device-resident argument/index buffers; cannot deserialize arbitrary model/task/IR graph on each token. Count CPU instructions/host sync, allocation and enqueue per step.
- **GPU native work:** router count/selected IDs, grouped experts, state update and trained draft verification occur device-side when model/provider permits; host does not stage tiny expert-ID tensors every token in a guaranteed resident fast path.
- **Scheduling:** bounded StepPlan/diff update, pipeline CPU prepare and GPU execute, output transfer asynchronously where exact state dependencies allow; StepProgram can batch multiple device-native microsteps but remains transactional and cancellation-safe.
- **Memory:** no steady per-token host/device allocation for KV/slabs, no per-operator global arbiter synchronization; pre-reserve, reuse typed slots, deferred retire.
- **Shape:** symbolic rank/dimension bounds plus validated shape guards; avoid one kernel variant for every integer context length, bucket by cost and verified numeric range.
- **Compilation:** on cold model, compile only immediately necessary portable kernel/shape plan; optimize/tune hot regions asynchronously/explicitly after validation; no guaranteed instant compilation of exotic models.
- **Fused regions:** huge register/LDS kernels allowed but promote by full-task elapsed time and quality, not source elegance/launch count.
- **Concurrency:** evaluate C1/2/4/8/16/32 and task mix including audio/media with P95 and fairness; no blanket faster-on-every-model claim.
- **Energy/physical safety:** GPU+NPU shared LPDDR/power and SSD latency must be measured under same-clock ABBA; hardware unsafe profile disables optimization, never modifies privileged settings.

## 5. Minimality budget / architectural complexity constraints

**Required in first real vertical slice:** one semantic program representation, bounded verified state-effect graph, reference execution, one task driver, one kernel registry for a handful of ops, one device provider, a small command DAG/event lease verifier, one resource-admission manager, and one session owner/worker protocol. The original **2–4-crate bootstrap** suggestion (or one library + binary) was a way to avoid speculative scaffolding, not a permanent cap on the subsequently tested workspace.

**Explicitly deferred until tests justify it:** general workflow/agent DSL and external-action runtime; dynamic library plugin ABI; arbitrary offline equality saturation; distributed transaction WAL; automatic topology co-optimizer; every quant format; media modalities without model test fixtures; online training engine; JIT per new token shape; persistent cross-driver graph cache serialization.

**Minimum schema count:** one SIR semantic schema; one versioned state snapshot schema; one compiled executable plan representation whose cache can be dropped/rebuilt; one canonical TaskRequest/Result schema. RIR/KIR/CIR are logical compiler stages/views, not necessarily persisted standalone formats. External model artifacts are their original formats plus separately versioned optional derived packs.

**Software boundary:** all portability lies in Rust logical semantics and device/IO backend contracts. Never require an upstream model server; a 1-off fastest kernel should be expressible as a region candidate under existing runtime.

**No dynamic plugin system by default:** built-in static providers first; capability registry can be a Rust enum/table with function pointers/traits resolved once at model load, not a per-kernel dynamic dispatch zoo.

## 6. New required proof/counterexample program

| Test | What it proves | Rejection signal |
|---|---|---|
| **T4-01 Atomic state root** | fake KV+GDN+RNG branches prepare then one PublishRoot; failures leave original root intact | one domain mutates visible root while another fails |
| **T4-02 Device-native second step** | StepProgram executes two dependent GPU-native iterations without host forcing sampled token roundtrip when legal | scheduler must transfer every sampled token just to continue |
| **T4-03 Hierarchical lease** | central model/session reserve with device-local fast suballocator and async retire; no global allocation lock in hot path | buffer ABA or 1 global lock per op |
| **T4-04 Shape guard** | ragged/chunk inputs choose valid compiled shape variant or portable safe fallback | invalid kernel receives unsupported shape, infinite JIT churn |
| **T4-05 Non-token task reuse** | one lightweight TaskDriver handles encoder/iterative stage using same executor; no giant DSL | every stage requires a separate server or must report fake tokens |
| **T4-06 Training future seam** | synthetic parameter-update loop uses mutable ParameterVersion and gradient/optimizer state without editing inference kernel/session ownership | adding parameter effects forces rewrite of entire inference engine |
| **T4-07 Distributed fencing** | coordinator fails after rank prepared and before commit; old root/outputs remain valid, recovery/epoch fencing | ranks emit divergent user-visible results or stale retries commit |
| **T4-08 Output reconnect** | disconnect/retry with optional output seq cursor and zero uncommitted media/text leakage | duplicate unidentifiable emitted chunks or leaked speculative tails |
| **T4-09 Unfused path dominance** | zero-specialization path is fully functional and one extreme fused kernel replaces legal region without owning server | fused kernel forces copying sampler/resource manager |
| **T4-10 Compilation/cache stability** | shape/toolchain upgrades invalidate only necessary bundles and don't blow cold-start memory | stale graph pointers, hundreds of unused variants |
| **T4-11 Cross-run numeric fidelity** | source quant/compute precision and deliberate approximation tracked independently | route/top-k silently reduced for t/s |
| **T4-12 External tool boundary** | untrusted task graph cannot issue arbitrary process/network/file action through inference kernel or IR op | model tensor can trigger privileged tool calls |

These are design proof obligations. Consult current source/tests and [active plans](../plans/TODO.md) for completed slices and remaining gates; do not infer passing results or reopen completed work from this historical table.

## 7. Risks and unresolved policy, retained deliberately

- **Model scope:** “all things related to AI execution” means *composable supported AI programs*, not unrestricted execution of arbitrary unknown mathematics. Training/autograd/fine-tuning is a **future optional task capability**; the inference-first optimizer/recurrent/quant core must not prepay its design cost. Check user priority before treating training as an MVP deliverable.
- **Rust source:** backend kernels must be Rust-authored under stated user intent; whether external CUDA/ROCm/XRT driver, vendor math or non-Rust machine-code generator is allowed is open. C++/Python AIE kernel source remains research, not silently imported.
- **Lifetime:** no universal strict “host never OOMs” proof against unrelated system allocations. Guarantee admission under a measured conservative snapshot and best-effort pressure fallback, with clean device allocation failure handling.
- **Distributed correctness:** volatile single-coordinator publication is not durable exactly-once network delivery; don't promise durable distributed transactions until implementing proper log/ack/recovery.
- **Numerical:** bitwise replay can be guaranteed only under a *particular fixed plan/mode* where verified. Cross-GPU reductions can differ; the numeric contract specifies acceptable tolerances and quality, not silent approximation.
- **Pipeline universality:** task-specific loops and live reactive workflows can require new semantic op/control effects; stable primitives reduce but don't eliminate needed new code.
- **Legal scope:** project-neutral design and provenance remain separate. Verified MIT source may be considered for concrete source reuse with file-level notices and Rust-source approval; non-MIT upstream code is cited research only by user's project policy.
- **Performance:** even the best designed modular architecture may trail a hand-coded specialist on one workload. Specialized-region substitution is the mechanism to compete **without duplicating engine ownership**, not a promise of guaranteed outperformance.

## 8. Retained migration rationale and current references

The v2/v3 archives are no longer required live documents. The useful rejected alternatives remain in R4-01…R4-24 above and in these technical contracts:

- **One publication authority:** domain-level commit and rank-local commit were rejected because failure could expose inconsistent KV/recurrent/RNG frontiers. [The execution spine](../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) defines Prepare/Ready and one logical PublishRoot; [peer handoff](../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md) adds fencing and fail-stop/replay rather than network exactly-once promises.
- **No duplicate compiler or task runtime:** four enduring IR ecosystems and a generic in-core workflow DSL were rejected for versioning, hot-loop and privilege cost. [Task/compiler refinement](../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) retains transient RIR/KIR views, one physical plan and a small trusted TaskDriver. External tools remain above the compute boundary.
- **Measured costs, not optimistic labels:** a cheap logical frontier does not prove O(1) physical rollback; multiple callers do not prove native batching; CPU expert work on UMA can compete for the same bandwidth. [State transactions](../docs/STATE_CACHE_AND_TRANSACTIONS.md), [memory/IO](../docs/MEMORY_RESIDENCY_AND_IO.md) and [joint serving](../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) retain those distinctions.
- **Correct sources before reuse:** the old disk-header arithmetic and mismatched-model benchmark errors are retained in [the reconciliation record](ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md). They justify fresh checked schemas and matched comparisons, not frozen legacy formats or performance floors.
- **Independent hardware strategies and source rights:** routed expert ownership, same-SoC projection-row slicing and remote collectives have distinct costs and failure boundaries. [Placement](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) and [GPU/NPU cooperation](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) own the contracts; the [pinned fork research](GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md) retains attribution, source-language and hard-hang caveats. A device-present probe or Rust host wrapper alone proves neither safety nor Rust-authored device code.

Current source/license decisions follow [the provenance policy](PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md) and decision register; existing qualified native profiles are not reopened by the historical FFI questions. Work tracking follows [AGENTS.md](../../AGENTS.md) and [TODO-SPEC.md](../plans/TODO-SPEC.md). No archive-retention, handoff-generation or SHA-manifest process is required.

## 9. Review verdict

The most consequential correction is **not another IR dialect or optimizer**. It is to make session state publication and CPU/device asynchronous execution *mechanically precise*, while reducing the number of **persistent abstractions** that runtime and future agents must maintain.

The architecture should ultimately have **one semantic program, one task driver, one executable plan with validated resource commands, one logical session-state publication authority, one physical budget authority, and many independent kernels/devices**. Fast paths are chosen at model/phase/shape boundaries and can cover entire fused regions, but cannot override mathematical or state ownership.

This increases the *chance* of achieving high performance **and** future modularity by allowing specialization to become a small replaceable component, not an independently administered runtime.

````

## Source: .agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md

Original UTF-8 bytes: 36543; SHA256 `a35cb596ad869dc67ac6b0589e52a17ae0d2a75955d8e1f5cb65045b6298a41c`.
Resolve embedded relative paths against the original path, not this volume.

````text
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

````

## Source: .agents/prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md

Original UTF-8 bytes: 10863; SHA256 `17c3c22741d59afbb93acc9c2d1a55abae1d6a934a2efdc1543d74230b39e9e8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Distributed execution, state disaggregation and tunnel transport: prior-art research

**Research snapshot: 2026-10-12. Non-normative, source-attributed.** Project-specific names are retained here with exact research URLs, and MUST NOT become zerogravity's architectural classes, duplicated inference engines or vendor-dependent type names. Normative distributed design: DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md. No source from prior-art projects was copied unless directly vendored (for example, CubeCL (vendored)), which retains its upstream license; the Zerogravity engine is a from-scratch rebuild informed by prior-art concepts, learnings and research. No network was connected, no cluster benchmark was run, and no external research repository was modified.

## 1. New axes beyond model/tensor sharding

Contemporary distributed AI execution includes **independently placing**: mathematical operations and task stages, immutable weight shards, stateful KV/recurrent/indexer data, full-prefix reusable snapshots, compressed historical token ranges, sparse routed experts, scratch/IO, and transport/program execution locations. The best partition depends on whether computation or memory is the bottleneck and whether moving **activations** or **state** across a link is less expensive. Do not assume a tunnel has datacenter RDMA latency.

Important combinations: colocated layer+own KV, model layers on separate workers, stateful attention at a KV owner with weight-heavy FFN on another, prefill/decode with full-state transfer, remote prefix-only cache hydrate, attention KV token-range sharding with mathematical softmax merge, and heterogeneous GPU/NPU/CPU expert ownership. All must share one semantic model and transaction/ResourceArbiter contract.

## 2. Prefill/decode, KV transfer, cache-aware routing

- [vLLM disaggregated prefilling docs](https://docs.vllm.ai/en/latest/features/disagg_prefill/) explicitly describe independent prefill and decode workers, async KV connectors (NIXL, Mooncake, and other independent connector implementations), and say disaggregated prefill is NOT inherently throughput-improving. Useful distinction: tail inter-token latency/TTFT and placement flexibility versus state handoff costs.
- [Practical source-specific guide published Sep 29, 2026](https://vllm-project.github.io/2026/09/29/disaggregated-serving-guide.html) discusses CPU frontend separation, P/D, remote KV and encoder/attention-FFN. One reported example is roughly 3 GB KV for 10k Llama-70B BF16 tokens; not every model/quant has this size.
- [SGLang P/D docs](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/pd_disaggregation.mdx) explicitly differentiate reserving decode KV slots before prefill versus only once prefill is complete; safe destination admission, deadlock avoidance and latency matter.
- First-party P/D serving architecture documentation documents KV-aware routing and nonblocking VRAM-to-VRAM transfers. An independent routing design describes cache-aware routing, shared prefix from a peer CPU tier and SLO-aware scaling.
- [Mooncake KV-centric paper](https://arxiv.org/abs/2407.00079) and independent multi-tier prefix-cache documentation motivate multi-tier remote prefix caches and cache-aware placement. Research claims are not our benchmarks.

**Transferable mechanisms:** independent prefill/decode or encoder pools; explicit whole-state Schema/StateCompatKey (not only simple KV for hybrid models); source/destination reservation handshake; cache presence hints and cost-aware routing; optional remote prefix hydration; measured P95 and state transfer.

## 3. Attention/FFN disaggregation and remote KV execution

- [vLLM AFD experimental plugin, Jul 23, 2026](https://vllm-project.github.io/2026/07/23/vllm-afd-plugin.html) separately places stateful attention/KV workers and FFN/MoE worker groups. Per-layer activations and execution metadata move, while FFN ranks can scale independent of KV capacity.
- [CMU CSD operator-level disaggregation, May 15, 2026](https://www.cs.cmu.edu/~csd-phd-blog/2026/operator-level-disaggregated-serving/) compares attention memory cost to compute-rich FFN and explains why fixed attention/FFN splits can LOSE efficiency if stage latencies mismatch or hybrid attention layers vary.
- [Attention/FFN provisioning theory, Jan 2026](https://arxiv.org/abs/2601.21351), and [disaggregation design space, May 2026](https://arxiv.org/abs/2605.28302) examine conditional cost/scaling. These are mathematical/simulator/source studies, not universal wins.

**Critical distinction:** locating KV on node B while running attention computation on node A requires continual *remote reads or cache transfers*, and can be disastrously expensive. An alternative is **compute attention on node B, beside KV**, transferring only queries and compact results to node A where projection/experts run. This difference must be explicit in the planner, not a single boolean named offload.

## 4. Token-range KV sharding and long-context elastic attention

- [KernelFlume, Jun 2026](https://arxiv.org/abs/2606.29207) studies weight-heavy projection/FFN nodes and **weightless attention nodes holding token-range KV shards**, query-first attention and inter-layer pipelining. Published physical P99 and projected larger simulations are author reports, not our results.
- [LoongServe elastic sequence parallelism](https://arxiv.org/abs/2404.09526) motivates varying sequence-attention partition degree by context and workload, with cost of moving KV shards and group communication.
- The attention math allows normalized softmax summaries (local max, exponential normalization denominator, weighted value numerator) to be merged across token-range shards using a global max. It requires correct masks, positions, numeric agreement and **special handling or refusal** for sparse top-k, compressed attention and recurrent GDN/KDA—none inherit ordinary full-softmax partition legality by default.

**Generalized mechanism:** separate query/FFN computation and KV state ownership, optionally shard sequence state over attention nodes and send compact query/partial outputs instead of K/V tensors every token. Route updates only at valid committed frontier; session state and compression/position remain typed.

## 5. LAN and consumer-machine distribution

- [llama.cpp RPC backend](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md) supports local+remote weight/KV distribution, tensor split, remote tensor cache and negotiated TCP/RDMA links when physical RDMA eligible. Exact source ABI/flags remain prior art; MIT-repo candidate still needs file-level license.
- An independent automatic-discovery and topology-aware sharding design claims heterogeneous peer discovery and topology-aware parallel distribution, RDMA over eligible Thunderbolt systems; an earlier revision includes ring-style contiguous layer groups across network machines.
- Collaborative internet-scale model block execution research and its [ACL 2023 paper](https://aclanthology.org/2023.acl-demo.54/) demonstrate model layers across independent network hosts, including privacy and peer trust challenges. Public, untrusted contributor swarms expose private inputs/hidden states; not a default product trust model.

**Transferable:** a small private node cluster with explicitly authorized peers and dynamic capability discovery, capacity-first layer partition where WAN latency tolerates few stage boundaries, and independent checkpoint/tensor rights. Auto discovery does NOT imply auto sharing user prompts/weights.

## 6. Transport and encrypted tunnels

- [NIXL architecture](https://docs.nvidia.com/nixl/getting-started/architecture/) and [transport matrix](https://docs.nvidia.com/nixl/user-guide/backend-selection/) expose typed registered VRAM/DRAM/File/Object/Block memory segments and nonblocking transfers using local peer links, IB/RoCE/GPUDirect/UCX/TCP/storage according to actual capability. Its implementation is not a required project dependency.
- Mesh-overlay connection-type documentation: direct UDP peer connections, peer relay and DERP relay differ greatly in latency/throughput; all are WireGuard-encrypted. Published performance guidance and hard-NAT warnings document relay throughput and possible head-of-line constraints.
- Rust QUIC library documentation describes encrypted, authenticated multiplexed reliable streams and optional datagrams. QUIC is a candidate wire implementation and not universally faster than tuned TCP or QUIC-inside-WireGuard.
- A VPN/WireGuard/tunnel exposes network connectivity and encrypted traffic, **not** real IB/RDMA/GPUDirect. Physical registered-memory access needs verified NIC/capability/trust; an overlay usually uses stream/buffered copies. Double encryption, relay hops, CPU crypto, MTU and packet loss count in actual effective latency/bandwidth.

**Transferable:** an explicitly trusted peer control plane (node identity/authorization/discovery/health), a data plane whose transport is chosen per physical link, and separate capability for direct, overlay-direct and overlay-relayed paths. Tunnel-only links may be suitable for **layer capacity, remote immutable/prefix caching or migration**, while per-layer per-token attention↔FFN over relayed WAN may be strictly refused.

## 7. Safety, numerical, economic and product boundaries

Pipeline parallel's few activation hops can fit limited links but may worsen serial decode ITL; tensor/attention/FFN per-layer all-to-all needs very fast fabric and workload batching. Remote cache hydration has large initial transfer cost and should be compared to recompute. Remote attention can reduce transferred state but adds query/return RTT. Mixed quant/compute must retain model sampling quality; reduction order can vary.

Failure modes: incorrect model/quant/position/adapter, remote rank loss, split-brain coordinator, expired capability token, canceled session with live RDMA buffers, packet duplication/reordering, stale peer cache directory, dangling device program, tail-latency outliers and secret leakage. Proper single logical StateRoot publication, epoch fencing and replan/recompute from durable checkpoint are needed. A network tunnel cannot guarantee remote machine won't read weights/prompts; use trusted peers and explicit authorization.

## 8. Conclusion

The architecture should represent (a) **logical model/task execution graph**, (b) **physical compute/weight/state/cache storage assignments**, and (c) **network topology and transfer plans**, with no new model-specific runtime. Source projects and performance claims above remain attributed prior art. The ability to *describe* a placement mode is not proof it outperforms colocation; physical link throughput, model/quant/full-state reference, reliability and security must qualify every plan.

````

## Source: .agents/prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md

Original UTF-8 bytes: 17174; SHA256 `bfb2d043c2a6837b378dc3a41bcef852252fbcce6d473102fd5ea5e89362c8f4`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — Final Architecture, Editorial and Agent-Handoff Review

**Review snapshot:** 12 October 2026 · Architecture v4.3 · Publication dossier 0.1.  
**Status:** **DESIGN-ONLY REVIEW; NO IMPLEMENTED RUNTIME OR BENCHMARKS.**  
**Purpose:** provide the final, honest engineering/editorial verdict for the requested publication-ready whitepaper and complete LLM-agent implementation specification. This file is an issue/authority and residual-risk ledger, **not a second architectural source of truth**.

**Canonical root:** `/home/pvl/spaces/abird/src/zerogravity/`. The independent `/home/pvl/spaces/abird/research/inference` tree is read-only prior art. The repository has not been initialized with Git and the source has no Rust engine as of this review.

## 1. Final deliverables and ownership

| Deliverable | Current path | Review purpose |
|---|---|---|
| Whitepaper manuscript | `../../docs/WHITEPAPER.md` | Externally readable proposal: motivation, novel *composition* argument, architecture, research context, evaluation, risks and limitations |
| Primary engineering spec | `../../docs/ENGINEERING_SPECIFICATION.md` | **Cross-cutting normative ES-00…ES-15** ownership, math/state, compiler/worker, physical memory, distributed and router/API rules |
| Bibliography | `../../docs/REFERENCES.md` | Numbered academic/first-party source references and attribution, no invented original-source credit |
| Publication checklist | `../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md` | Claims, independent results, code/whitepaper license, author signoff and editorial release gates |
| Public docs index | `../../docs/README.md` | Routes readers/reviewers/agents, classifies normative versus source history |
| Implementation playbook | `AGENT_IMPLEMENTATION_PLAYBOOK.md` | Every stable P00–P23 phase has bounded goal, owned modules, reference/negative tests and handoff rules |
| Requirements traceability | `REQUIREMENTS_AND_TEST_TRACEABILITY.md` | U-001…U-022 → ES + invariants + P phases and Stage0 proof requirements |
| Current concise agent entry | the agent playbook | Minimal mandatory reading order and existing project/hardware/privacy status |
| Full handoff | the retired master handoff | Latest publication takeover at top, all previous v4.3/v4.2/v4.1 history preserved |
| Core architecture detail | `ZEROGRAVITY_MASTER_PLAN_2026-10-10.md` and current `../docs` contracts | Full specialized technical source, subordinated on cross-cutting ambiguity to integrated engineering spec |
| Source/historical evidence | `../prior-art/PRIOR_ART_*`, `../prior-art/DESIGN_EVOLUTION_*`, platform investigations | Attributed research, legal provenance, old plans **not** another runtime interface authority |
| Document integrity manifest | the retired integrity manifest | Byte-exact SHA records of every other current Markdown file, regenerated **last** |

**Rationale:** The project had become a very large chronologically accumulated specification collection. These entrypoints **avoid** asking every implementation agent to reinterpret hundreds of pages of overlapping historical notes before making a small code change. The detailed original plans and research remain accessible; explicit authority and traceability prevent accidental silent loss of decisions.

## 2. Cross-cutting technical findings and dispositions

| Finding | Risk/failure if implemented literally | Final disposition and normative source |
|---|---|---|
| **F-01: Per-domain commit versus global transaction** | incomplete KV/GDN/RNG state published during speculative failure | **RESOLVED IN DESIGN:** StateDomains `Prepare`; TaskCoordinator alone publishes an immutable whole StateRoot after all physical readiness. [Engineering ES-06](../../docs/ENGINEERING_SPECIFICATION.md); runtime source test NOT RUN |
| **F-02: Device compiler abstraction too narrow** | TPU/wafer accelerator simulated with thousands of host kernel RPC calls | **RESOLVED IN DESIGN:** one RegionExecutionProvider can own kernel, captured graph, AOT stage, or static dataflow at legal granularity, with explicit StateBridge and lifecycle. [ES-04](../../docs/ENGINEERING_SPECIFICATION.md); vendor SDK support unverified |
| **F-03: Overly many enduring IR/runtime formats** | enormous code/ABI and per-token traversal overhead | **RESOLVED IN DESIGN:** SemanticProgram and ExecutableBundle are enduring concepts; RIR/KIR transient, CIR physical dependency DAG inside bundle; source-specific full-runtime adapters not required |
| **F-04: Global resource manager as hot per-kernel lock** | GPU idle, central contention at 32 concurrent tasks | **RESOLVED IN DESIGN:** coarse global admission and local fast pools with generation/event leases; must measure actual O(1) hot path and pressure failure |
| **F-05: TaskProgram tempted to become agent tool workflow** | arbitrary process/file/network side effects in model kernels | **RESOLVED IN DESIGN:** small trusted TaskDriver, external authorized application/agent layer; no unrestricted ExternalAction in inference |
| **F-06: Same source mathematical operators across modes** | silent top-k/quant/attention semantic changes, false throughput wins | **RESOLVED IN DESIGN:** distinct semantic, numerical, task quality and sampling-distribution evidence; source model artifact separate from temporary physical compute |
| **F-07: Compute and cache location conflation** | remote full KV reads every token over slow network, or source-invalid prefix hits | **RESOLVED IN DESIGN:** independent compute/parameters/mutable state/immutable cache maps plus costed TransferPlan and StateCompatKey; remote attention/hydrate/direct-read alternatives are distinct |
| **F-08: Multi-machine state authority** | split-brain coordinator, network timeout frees active DMA, duplicate output | **RESOLVED IN DESIGN:** session/placement epochs, remote shard leases, Prepare/Ready cohort, fail-stop/checkpoint replay; durable HA/consensus not implemented |
| **F-09: More than one model means physical infinity** | uncontrolled out-of-memory, cold model thrash, active session eviction | **RESOLVED IN DESIGN:** RegisteredModel vs aliases vs ResidentModel/Replicas vs TaskSession; safe load/warm/pin/evict under ResourceArbiter |
| **F-10: Model load balancing could change target** | wrong checkpoint/quant/adapter for a user-requested model | **RESOLVED IN DESIGN:** exact model identity pinned, equivalent replicas only; optional explicit multi-model policy transparent |
| **F-11: OpenAI API facade mistaken for cloud parity** | wrong SSE/Responses schema, unexpected tool execution, private stored data | **RESOLVED IN DESIGN:** endpoint/task/field/version SDK test matrix, model-generated tool-call data only, privacy-first store and strict unsupported semantics |
| **F-12: Publication text overclaims implementation** | misleading “fastest”, generic any-model, supported TPU/wafer/tunnel capability | **RESOLVED IN MANUSCRIPT:** all performance ambitions labeled hypotheses and physical tests absent; exact capabilities classified PLANNED/REFERENCE/COMPILING/FUNCTIONAL/QUALIFIED only from evidence |
| **F-13: Project-named source architecture dependencies** | incompatible licenses or fragile upstream API ownership | **RESOLVED IN DESIGN:** project-neutral normative types, named sources with dates/links preserved in separate prior art; exact-file license review before any reuse |
| **F-14: Agent takeover requires hidden conversation** | next agent renumbers phases, copies source examples or repeats decisions | **RESOLVED IN DOCUMENTATION:** public whitepaper+engineering spec, the agent playbook, Traceability, the retired master handoff and negative-test matrix |

**Technical outcome:** a coherent and falsifiable set of ownership contracts is specified. **Implementation correctness or performance has not been demonstrated**; nothing above should be promoted to physical VERIFIED until code/fixtures/benchmarks pass.

## 3. Editorial and citation quality review

- The **whitepaper** is separate from the engineering specification. It presents a stand-alone technical argument and does not require a reader to inspect private conversation history. It includes abstract, problem, design principles, mathematical/runtime/distributed/serving architecture, source-backed related work, evaluation hypotheses, security, limitations and conclusion.
- The **engineering spec** is implementation-focused and organized as ES-00 through ES-15; illustrative Rust interfaces are clearly marked as *not stable public ABI or existing source*.
- The **references** are numbered and source attributed. Peer-reviewed reference R07 was corrected to **Ansor at OSDI 2020**, R05 to **DistServe at OSDI 2024**, and the OpenAI Responses documentation link corrected to the dedicated Responses overview. Primary proceedings and current API references were independently checked. These support research context only, not zerogravity benchmark claims.
- The initial integrated diagram incorrectly suggested ModelRouter/admission happens **downstream of** physical worker execution; this was corrected to clarify request admission → compiled execution → StateRoot publication, while resource reservations remain cross-cutting. A structural review also found three accidentally concatenated section headings (`ES-05`, `ES-08` and `ES-11`) caused by interrupted incremental writing. These were separated into valid headings and included in the navigable ES-00…ES-15 contents; a compact ES-15 glossary was added.
- Historical design/version appendices are **preserved, not silently deleted**; active authority ordering resolves contradictions without rewriting prior evidence. External project references are acceptable in whitepaper/bibliography/prior-art dossiers as citations, but not core architectural type names.
- The manuscript intentionally omits author attribution, unapproved original project license, DOI, public Git URL and experimental graphs. Those require owner and independent reviewer inputs rather than invention.
- The public manuscript is **technical-review-ready**; an actual public distribution still requires user signoff, license/byline and external link QA. Its claims should not be marketed as software features until implemented.

## 4. Open research and implementation risks that documentation cannot close

| ID | Risk / unknown | Concrete blocking proof or policy |
|---|---|---|
| **RISK-A — real Rust GPU code generation** | Can the chosen Rust-authored device kernel path match mature low-bit kernels under register/shared-memory constraints? | One actual source-verified model, independent reference GEMV/prefill/decode/quant and profiling on target GPUs |
| **RISK-B — enough generic semantic ops** | Novel future models may require new operators/effects not representable by current registry | Second truly different recurrent/sparse model with portable reference and closed-region fallback |
| **RISK-C — GPU StepProgram control** | Exact grammar/RNG/cancellation may force host synchronization despite graph capture | Device-native multistep with exact acceptance/stop reference and measured host sync, or honest host-loop fallback |
| **RISK-D — economical full-state COW** | Logical root swap may need costly physical recurrent matrix copy and multi-domain snapshot | Three-domain failure replay/copy benchmark and bounded physical memory under C1/8/32 |
| **RISK-E — mixed model residency** | Cold model load/eviction and program memory could thrash unified memory | True physical memory admits, 100-load dedup, 1–8 model workload P95/energy, active KV lease safety |
| **RISK-F — distributed network performance** | WAN/tunnel RTT can overwhelm benefits of remote attention/FFN/sequence shards | Authorized actual two-peer direct and relayed tests at matched model/quant/long context; capacity-only wins labeled honestly |
| **RISK-G — remote failure/retention** | No durable coordinator quorum or exactly-once output mechanism | Explicit product durability target and implementation of lease fencing/checkpoints/ACK replay if needed, otherwise fail-stop |
| **RISK-H — vendor SDK/source language** | AOT TPU/wafer/ASIC may lack legal direct Rust device compiler or StateBridge | Specific vendor toolchain/license/SDK/physical source proof or unsupported mode; keep generic CPU/GPU |
| **RISK-I — OpenAI API evolution** | Cloud schema, models, streaming events and SDK behavior change over time | Pinned SDK/endpoint/parameter capability matrix, conformance tests, explicit unsupported fields |
| **RISK-J — model/training release breadth** | Broad task names are not runnable model/codec support and optional autograd may bloat inference | One real qualified checkpoint+TaskDriver per advertised modality; training optional ParameterVersion seam only |
| **RISK-K — publication/provenance** | No byline, document/code license, stable venue or independent review yet | Resolve O-067…O-071 and run external link/license/peer review before public release |
| **RISK-L — architecture complexity growth** | Extensive docs tempt coding agents to implement every proposed abstraction up front | Playbook P00–P08 real vertical slice first; measure crates, code churn, build time and hot-path overhead before expanding |

These are **not excuses to add hypothetical frameworks now**. Each risk should become a small code/test/evidence ticket only when relevant to the authorized phase.

## 5. Requirements and work preservation

- User decisions **U-001…U-022** remain in `DECISION_REGISTER.md` with all P working decisions and O open items retained.
- Stable original implementation phases **ZG-P00…P23** remain unchanged and NOT_STARTED.
- Experiments **H01…H86** remain a research/proof program, not a to-do list that must all be implemented in the first release.
- Original Stage0 S0-00…S0-14, optional NPU S0-NPU-A…F, V4 S0-V4-A…J, optional fabric S0-FABRIC-A…H, distributed S0-DIST-A…L and serving S0-SERVE-A…J remain traceable.
- The complete previous v4/v4.1/v4.2/v4.3 handoffs, v2/v3 historical snapshots and source-license/source-specific research remain intact.
- The **current intended first executable** remains one trusted checkpoint/tokenizer/quant CPU reference, one TaskDriver, direct compiled region and resource event lease, full Prepare→PublishRoot and local terminal output; then same-engine API and model router.

## 6. Verification ledger and final closure procedure

The final review should run:
1. existence and nonempty length of whitepaper, integrated engineering spec, bibliography, public index, publication checklist, agent playbook, traceability, status/handoff and decision/roadmap;
2. all Markdown relative links, internal section anchors, code fences/UTF-8 and H1 headings;
3. exact user decision IDs U-001…U-022, proposal P-001…P-091, open O-001…O-071 without lost historical IDs;
4. exact unique phase headings ZG-P00…P23 and experiment H01…H86 headings;
5. all expected Stage0 gate IDs and separated source-named research vs normative architecture (brand names cited as prior art are fine);
6. no new implementation source files, Git, installed network services, model downloads or changed read-only source corpus;
7. regenerate the integrity manifest **last** and verify exact SHA-256 for every other project Markdown file.

**Observed final structural review (2026-10-12, prior to manifest refresh):** **70 Markdown files**, **672 Markdown links**, **56 internal anchors**, **16 ES sections**, **25 integrated invariants**, **23 numbered references**, **24 stable roadmap phases**, **86 experiment IDs**, **61 Stage0 gates**, **U-001…U-022**, **P-001…P-091** and **O-001…O-071** all verified present with **zero structural/link/anchor/normative source-branding errors**. A full archive file-content SHA verification is the final follow-up, recorded in the manifest and last tool output. These tests establish documentation consistency only, not physical model execution. The absence of source files is verified on the available project filesystem but does not prove anything about external model/hardware systems.

## 7. Publication and engineering verdict

**Architecture coherence for design review:** ready to submit for peer technical review, with known unimplemented risks explicitly named. **Executable performance/resource/security correctness:** **NOT YET VERIFIED**. **External/public distribution of manuscript:** pending owner-selected authorship and licensing/peer signoff. **Agent implementation handoff:** complete as a design-to-code contract, provided the new agent respects phase scope/actual source state and independently validates physical source/math before promoting capabilities.

**Success after implementation will mean**: a supported model runs simply and correctly, the runtime admits independent specializations without multiplying engines, and compatible heterogeneous/distributed/model-router workloads improve measured practical quality/latency/memory/energy under the exact stated safety and fidelity constraints. The present files describe a route to that goal, not an achieved benchmark result.

````

## Source: .agents/prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md

Original UTF-8 bytes: 21647; SHA256 `089c60c27009c5c6177c35b7f648b72bfda64066eba4f425d45cc86d80aabe4b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Gufo XDNA2 cooperative-prefill fork and October 2026 Strix Halo report — primary-source research

> **Research/provenance only (v3, 2026-10-12).** Project names, URLs, specific implementation techniques and source-author benchmarks in this document are **attributed prior art, not normative zerogravity runtime or compiler types**. The independent governing architecture is [Modular AI systems](../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [Composition contracts](../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md), and [Prior-art license policy](PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md). External projects never become a requirement to clone a model-specific inference engine; technical/measured evidence remains available for independent validation.


**Research date:** 2026-10-12 (Asia/Singapore). **Scope:** the exact user-supplied [mikealanni/gufo](https://github.com/mikealanni/gufo) fork and [r/StrixHalo 1x3gtdh](https://www.reddit.com/r/StrixHalo/comments/1x3gtdh/qwen38_flashnext_prefill_past_1800_toks_on_strix/) post, compared with [upstream gufo-org/gufo](https://github.com/gufo-org/gufo). This is a READ/SOURCE research record, **not** a zerogravity benchmark or executed NPU test. GitHub was inspected via the connected GitHub app; Reddit was independently read via public search. Distinguish this fork from original Gufo and from a separate dGPU+iGPU asymmetric MoE path.

**Verified GitHub fork ref:** `mikealanni/gufo`, default branch `fork-enhancements`, commit **`8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`** when inspected. Its `README.md` blob SHA was `af75c7a7211a0514935d940ed2374e7a56e75845`. Upstream `gufo-org/gufo` has default branch `main`; the two should be independently source-pinned. GitHub files below were inspected at this exact fork commit. Upstream HEAD and fork may change: re-fetch and diff before code/benchmark integration.

## 1. What the user's sources actually establish

**Author's claim:** 128 GiB GMKtec EVO-X2 Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) with Qwen3.8 Flash-Next **UD-IQ4_XS GGUF**:
- 16,384-token prompt, GPU-only **1,698 prefill tok/s**, GPU + XDNA2 NPU **1,821 prefill tok/s**; approximately **+7.24%** throughput at matched reported benchmark shape.
- 32,768-token prefill **1,783 tok/s** with NPU (no GPU-only value in fork README).
- 40K cold served prompt, MTP+cache: **1,637 tok/s** prefill with NPU (different workload/flags).
- README reports 8,192-token prefill chunks, held maximum GPU fabric clock, NPU performance mode and no competing load.
- Post reports 30 longer greedy prompts + a copy task produced identical *generated tokens* with and without NPU. This **does not prove** bit-identical full logits, sampled distribution, long-context correctness, input rounding parity or corpus-wide quality.
- Post reports prior **hard machine freezes / machine-check failures** when GPU fabric clock changes during concurrent NPU and GPU work; 12 subsequent clean runs after holding fabric clock. This is author experience on a particular firmware/driver/power configuration, not a general hardware fix guarantee. Some replies discuss AGESA version requirements and idle power increases from clock hold.

**Do not misattribute to upstream Gufo:** older `gufo-org/gufo` engine includes `gfx1151` Qwen3.8 optimized GPU inference, MTP, sparse attention, concurrent sessions and other model/media pipelines. The new feature in `mikealanni/gufo` is **partial projection GEMM offload onto Strix Halo's XDNA2 NPU simultaneously with its Radeon iGPU**, using the same *model forward pass*. It is **not**:
- a separate NPU draft language model taking turns with the GPU;
- two discrete GPUs with MoE expert partitioning;
- general XDNA inference of the full transformer/attention/SSM model;
- a GPU decode speed claim of 1,821 tokens/s (the number is **prefill**);
- a guaranteed speedup at every chunk size or on CPU/NVIDIA/Apple.

Sources:
- [fork README](https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/README.md)
- [user's Reddit thread](https://www.reddit.com/r/StrixHalo/comments/1x3gtdh/qwen38_flashnext_prefill_past_1800_toks_on_strix/)
- [upstream README](https://github.com/gufo-org/gufo/blob/main/README.md)

## 2. Architecture in actual fork code

### 2.1 Hardware and programming interfaces

**Compute engines:** AMD RDNA 3.5 GPU and XDNA2 AIE NPU on same Ryzen AI Max+395 SoC. NPU accessed through **Xilinx Runtime (XRT)** and an `.xclbin` hardware program plus matching instructions `insts_*.txt`. NPU AIE source uses a **Python MLIR-AIE layout generator and C++ mixed block-float tile kernel**. The host bridge uses C++ XRT and HIP APIs, controlled by optional CMake `GUFO_ENABLE_NPU=ON`; without it the GPU-only path is available. Fork source components:
- NPU GEMM host-bridge header [blob `2911538b5a8313f0b731d2592e72a7ca859027b0`]
- NPU GEMM host-bridge implementation [blob `289c98808ad14f0ec52ecd96bcb8dd7c7291df91`]
- device model (weight slicing/capability/activation)
- executor (GPU-NPU scheduling, submits, waits and GPU continuation)
- HIP kernels (device-side NPU block-float activation/weight input staging and output decoding)
- CMake NPU optional integration
- AIE program inputs: placed-core generator, mixed block-float kernel, `.xclbin` binary and instructions, and fabric-clock helper.

The .xclbin is *not* written in Rust. Its presence is evidence of a feasible XDNA2 hardware strategy, **not** permission to embed non-Rust kernels into zerogravity. The Python AIE generator carries Apache-2.0 WITH LLVM-exception provenance, the fork overall says MIT; individual file/header/licensing and compiled artifacts need audit before reuse. User's Rust-owned kernels constraint remains explicit.

### 2.2 One fixed NPU GEMM geometry (initial implementation, NOT fundamental model identity)

`NpuGemm::Config` has defaults **M=4096 prompt tokens, K=2560 input channels, N=1024 output channels**. Tile geometry `128x64x128` as named by the binary. AIE network uses **8 columns × 4 rows = 32 cores**. Kernels use a 9-byte group for eight values: one shared exponent plus eight signed 8-bit mantissas, `v8bfp16ebs8` (not exactly E4M3 FP8, GGUF Q8_0 or BF16). Recorded decoding: `value = mantissa * 2^(exp-127)/64`. GPU packs dense activations and Q8_0 expert/projection weights into that NPU-specific layout, including padded tails.

**Model-load prepacking:** the device model extracts selected rows from known Q8_0 projection tensors into immutable, NPU-friendly pinned/XRT buffers. The NPU operator is then fed narrow slices rather than converting the entire model to an NPU graph.

**Shape gate:** most active paths require `n_tokens >= 1024` but also `n_tokens % npu->tokens() == 0`, so the particular fixed kernel actually fires for **4096-token multiples** within max batch and 1–4 active groups. Other batch sizes and unsupported projections keep their GPU path. This is a useful example of **shape- and phase-specific provider capability**, not a general GPU+NPU promise.

### 2.3 Exactly which mathematical projections are split

The NPU is used for three independent GEMM families while the GPU performs the rest of the same layer:

1. **Gated DeltaNet (GDN) linear-attention input projection — `z` rows.** The device model defaults to 6 slices × 1024 output rows of the last part of a 16384-row input matrix; model route checks Q8_0 and `K=2560`. The GPU executes remaining projection/conv/scan while the NPU computes `z`. The NPU output is consumed just before **GDN gated-norm/epilogue**, not before the whole layer.
2. **Qwen sparse/full attention query and gate projections.** Input `[13312,2560]`, `num_heads=24`, `num_kv_heads=2`, `head_dim=256`, `rotary_dim=64` in the fork's guarded path. Default `GUFO_NPU_GATE_SLICES=8`, each slice 1024 output rows = query+gate of two attention heads, or 16 heads for 8 slices. GPU computes the remaining heads/keys/values and can continue indexer work. First NPU-result consumer is the query/gate preparation/attention phase.
3. **Shared expert gate/up projections.** When eligible Q8_0, dimensions fit NPU slice, and `GUFO_NPU_SHEXP` not 0, NPU runs two GEMMs for gate and up while GPU computes routed experts. NPU gate/up outputs feed a GPU-side SwiGLU then GPU shared-expert down projection; the result joins the MoE residual. This is **not moving routed experts to NPU**.

These are legal because the parent projection can be partitioned by *output rows* and the two parts have no dependence until the known join. Correctness still depends on stored quant, model score/order, output scaling and kernel rounding. Such partitions must be modeled in RIR and the resource/timepoint graph, **not hard-coded in Qwen ModelRecipe**.

### 2.4 Asynchronous execution and memory sharing

- NPU weight buffers are prebuilt at model load.
- GPU produces input activations; a GPU kernel encodes them into block-float staged A.
- `NpuGemm::Submit` enqueues HIP `hipLaunchHostFunc` callback into the GPU stream. Callback sends work to dedicated CPU thread, which pushes XRT NPU commands. GPU stream can continue independent GPU kernels; compiler/human schedules the first required result read later.
- `NpuGemm::Wait` enqueues another HIP host callback that waits for worker completion, introducing the dependency at **first true consumer**.
- NPU has **up to four outstanding jobs**; it uses one long-lived XRT hardware context to avoid millisecond kernel/context switches.
- GPU-visible pinned host pages (`hipHostMallocMapped`) are **imported** as XRT user pointer BO. In an isolation mode the NPU has private BOs with explicit copy-in/out around runs. A GPU kernel can decode NPU C block-float streams directly or through a staging/decode kernel, in some cases fusing activation like SwiGLU.
- This is **shared mapped memory with explicit XRT synchronization**, not proof of universally coherent zero-overhead reads or an extra VRAM pool. Count all pinned pages and staged duplicate weights in global physical memory.
- On timeout/XRT failure the fork marks NPU backend unhealthy and aborts affected batch. Timeouts can leave DMA unsafe to deallocate; code retains/quarantines affected buffers rather than freeing them immediately. **Zerogravity must never recycle a possibly active DMA buffer.**

This is an excellent source-level proof that our planned **CIR Async Resource/Command IR** should express execution dependencies between two accelerators in *one host*, including `Pack, SubmitNpu, ContinueGpu, WaitNpu, DecodeOrConsume, Join, Commit`.

### 2.5 Runtime/safety limitations actually observed

Fork README and post require **held fabric clock** on the author's firmware/driver to avoid hard host freezes, NPU `performance` power mode, and disabling runtime suspend to avoid NPU power transition contention. Internally it runs a periodic zero-operand XRT heartbeat to keep the NPU awake (otherwise driver suspend/reinitialize interacts with GPU); code also supports `GUFO_NPU_KEEPALIVE=0`. Root-level clock-hold tooling changes `power_dpm_force_performance_level` and `pp_dpm_fclk`. Do **not** run or recommend automatic privileged tuning at zerogravity startup, and do **not** assume changing frequency is universally safe or permanent.

The Reddit conversation contains reports of increased idle power with NPU/fabric-locked setups and firmware/AGESA-specific behavior. **These are user reports**, not independent root-cause validation. A reported 12 clean repeated runs is far too little for a general-purpose "safe to enable automatically" guarantee. On an unsupported system, the default must be GPU-only without launching NPU jobs or changing firmware/clock/power state. If NPU is unqualified, opt-in experiment with explicit preflight and emergency disabled fallback **before** any simultaneous GPU+NPU command. A software timeout cannot necessarily recover a machine hard-locked by an upstream power-management race.

**Do not assume NPU use improves decoder t/s, serial batch=1, or even prefill after considering 24h power/thermal effects.** NPU offload is justified only where its real critical-path benefit exceeds packing, XRT queue, shared-memory bandwidth and join costs, and the device is proven stable.

## 3. Relation to upstream gufo-org/gufo model pipelines

The upstream Gufo project already demonstrates more than Qwen Flash-Next:
- Qwen3.8-27B, image input, autoregressive and **DFlash2** speculative drafting;
- Qwen3.8 Flash-Next, **GDN/QSA/MTP**, large memory and long-context prefill;
- DeepSeek V4 Flash, **DSpark**, compressed attention;
- Qwen3-ASR speech recognition, **Qwen3-TTS** voice generation;
- developing Qwen-Image-2.1 image generation/editing and MiniMax H3 text-to-video/audio, with separate task models and numerical evidence.
- Native multi-request streaming, cache/quality reports and Nix+ROCm deployment.

**Design lesson**: do not model all these tasks as decoder-only `StepPlan::next_token`. They share SIR tensor/kernel/event/resource core, but different **TaskProgram** loops: LM autoregression+MTP, block-diffusion verifier, encoder-only ASR/pooling, TTS acoustic stream/vocoder, image diffusion/denoise iterations, video temporal/frame steps and multimodal pre/postprocessing. Typed effects, task-local state and `CIR` allow these to be added without rewriting the text server or memory ownership. Media pipelines are future-capability targets, not already functional in zerogravity. Learn numerical tests and measured bottlenecks per task.

See upstream `gufo-org/gufo` [README](https://github.com/gufo-org/gufo/blob/main/README.md), its model guides, benchmark documentation, upstream [TESTING](https://github.com/gufo-org/gufo/blob/main/docs/TESTING.md). The specific fork's new feature remains **XDNA2 partial projection offload**, not a replacement for the entire upstream multi-modal roadmap.

## 4. Observed strengths, weaknesses and potential zerogravity improvements

| Dimension | Fork proof | Not established | Zerogravity opportunity |
|---|---|---|---|
| Math-domain partition | selected output-row GEMMs of three independent projections | arbitrary new model/operator/quant | RIR `SplitProjectionRows` transformation with exact producer/consumer effects |
| Accelerators | gfx1151 + XDNA2 | Intel NPU, NVIDIA NPU, multi-host or other shapes | DeviceCapabilities NPU + generic AcceleratorProvider, no model-specific hardware hooks |
| NPU layout | 9-byte BFP8 (8 mantissas + exponent), fixed 4096×2560×1024 | arbitrary FP8/block shapes or exact BF16 parity | explicit compute QuantEncoding and quality gate, shape-specialized candidate |
| Async handoff | XRT long-lived context, HIP callbacks, shared mapped BO | no host sync overhead, guaranteed safe coherence | CIR release/acquire timepoints, bounded queued ops, source/target resource domains |
| Efficiency | author's +7.2% prefill for one UD-IQ4_XS workload | causal gain independent of clock pin and thermal policy | ABBA same FCLK/temp/power with NPU active/inactive, idle energy and 24h soak |
| Correctness | same generated tokens in 30 greedy prompts | full logits/quality, sampled distribution, long context, mixed prefill/decode | official checkpoint oracle, KL/logit/quality, every partial-join boundary |
| Reliability | 12 clean runs after fabric hold | unattended workstation stability across firmware | OFF-by-default until hardware-qualified, reversible external tuning, no sudo in serving |
| Adaptability | model-specific hard-coded slices | automatic choice for new models | per-model semantic region traits; generic `ProjectionSplitCandidate` planner and fallback |
| Maintenance | fork of C++ engine with AIE C++/Python code | Rust-only compiled kernel path | independent Rust DeviceBackend/KernelProvider; ABI+licensing proof before using XRT/AIE artifacts |

## 5. Required experiments before enabling any NPU offload

- **CPU simulation:** partitioned matrix projection join equals unsplit SIR oracle (full logits and output state), including arbitrary row split, padding, tensor quant scales and fused activation, dozens of shapes.
- **XDNA2 capability probe:** exact XRT version, XDNA revision, compatible precompiled AIE kernel ABI, physical pinned memory/map limits, IOMMU requirement, host driver/power/BIOS reliability and device-specific safety profile.
- **Rust kernel feasibility:** can AIE/NPU program be authored or generated/compiled from Rust-owned source while respecting user requirement? A Rust wrapper around existing `mm_bfp_mixed.cc`/Python AIE kernel is **not** a Rust-source implementation. If no viable Rust-owned provider exists, record as `blocked-by-policy`; GPU generic fallback works.
- **Bridge semantics:** Stage A/B/C bytes and scale/rounding, GPU→XRT shared BO ownership, explicit DMA and fences, host callback effects, timeouts and quarantine; no stale generation, host pointer or swallowed completion error.
- **First-use overlap:** measure actual (T_{GPU remaining}), (T_{NPU pack+queue+compute+sync+decode}), and join/transfer; proof gain equals critical-path reduction, not a second device running in name only.
- **Workloads:** Qwen3.8 Flash-Next exact UD-IQ4_XS baseline 16K/32K prompt under 8192 chunks; C1/C4, cold/warm cache, context depth and verify; GDN projection, attention queries/gates and shared expert toggled independently. Test arbitrary unsupported prompt length (GPU fallback), short one-token decode (GPU), model variants and other quant types.
- **Quality:** layer activation/logit relative error and sparse top-k IDs; 30 prompt same greedy token match is one smoke test, not a release gate. Target sampled speculation distribution, tool/reasoning, NLL/KL and official held-out continuation.
- **Fair performance:** ABBA NPU ON/OFF with **same held clock, NPU power state, driver/quant, prompt, device memory, temperature and CPU load**; compare GPU-only same-clock versus GPU-only default-clock separately. Report throughput, TTFT/P95, energy, pinned memory and system idle draw.
- **Safety:** normal boot/suspend/resume, idle GPU, mixed concurrent GPU and NPU, driver loss/power transitions, shutdown, failure/cancel, and >24-hour unattended soak on an isolated qualified test machine **with no destructive clock or BIOS modifications**. If any hard hang occurs, quarantine configuration; a software watchdog may not recover a frozen SoC. Never automatically run privileged clock-hold tooling, disable IOMMU or install a systemd clock-hold service.
- **Correct fallback:** absent XRT, NPU allocation failure, unsafe clock, no suitable shape, unsupported quant, too much NPU memory, XRT sync failure before job -> GPU compiled plan (from valid committed frontier); error after partial execution -> abort/quarantine/replay, not return incorrect result.

## 6. Source-by-source references and provenance

**User supplied**
- https://github.com/mikealanni/gufo (branch `fork-enhancements`, inspected commit `8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`)
- https://www.reddit.com/r/StrixHalo/comments/1x3gtdh/qwen38_flashnext_prefill_past_1800_toks_on_strix/ (author report published 2026-10-11)

**Fork source (immutable commit links)**
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/src/models/qwen38_flash_next/kernels/rocm/npu_gemm.cpp
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/src/models/qwen38_flash_next/kernels/rocm/device_model.cpp
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/src/models/qwen38_flash_next/kernels/rocm/executor.cpp
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/src/models/qwen38_flash_next/CMakeLists.txt
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/tools/npu/n32_core_placed.py
- https://github.com/mikealanni/gufo/blob/8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7/tools/npu/mm_bfp_mixed.cc
- the fork's privileged clock-hold script at pinned commit `8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`.

**Upstream**
- https://github.com/gufo-org/gufo
- https://github.com/gufo-org/gufo/tree/main/docs/models

**Other complementary research**
- A model-agnostic GPU+GPU/CPU expert owner split — this is a *different* heterogeneous parallelism from the Gufo fork NPU partial GEMM.
- /home/pvl/spaces/abird/src/zerogravity/.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md — target-independent timepoints, allocations and first-use join.
- /home/pvl/spaces/abird/src/zerogravity/.agents/docs/DEVICE_KERNEL_AND_CODEGEN.md — Rust-owned provider and capability contract.
- /home/pvl/spaces/abird/src/zerogravity/.agents/research/STRIX_HALO_OPTIMIZATION_DOSSIER.md — APU physical memory, GTT, GPU kernels.
- /home/pvl/spaces/abird/src/zerogravity/.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md — source-author figures versus reproducible quality/throughput.

**Conclusion:** the user-supplied fork contributes a *specific and important missing specialization* to zerogravity: **heterogeneous operator-slice pipelining on one APU with XDNA2 NPU**, including its own data-packing, queue, dependency, safety and precision policies. It fits cleanly under SIR/RIR/KIR/CIR and global ResourceArbiter if these contracts are formalized. It is not yet implementable/automatic and must default OFF until Rust-source kernel feasibility and hardware reliability are demonstrated.

````

## Source: .agents/prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md

Original UTF-8 bytes: 11958; SHA256 `602044316a685b0b7b90a93e5cd9783f6db770ea4249069771eb2e05da354a6f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Multi-model local runtime, router, CLI and API prior-art research

**2026-10-12 · NON-NORMATIVE SOURCE RESEARCH.** Public projects and API providers are named here for attribution, licensing and test-comparison only; core zerogravity architecture is independently specified in `MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md` and `ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md`. The source research tree `/home/pvl/spaces/abird/research/inference` is **read-only**. No remote API request, model download, server install or source code import was performed.

## 1. User goal and immediately important prior art

The user wants one experience that is **as simple as one command to run a model** while also operating a built-in model router, load balancing and concurrent multi-model serving with per-model hardware/quant/context/sampler/configuration, exposing OpenAI-compatible APIs and preserving the distributed/heterogeneous compiler architecture.

That request should not be reduced to a conventional API gateway proxy:
- model identity and model artifact/configuration aliases are separate;
- immutable loaded model weights, physical replicas, device placements, draft/adapters and *mutable session state* have different lifecycles;
- limited physical RAM/GPU HBM imposes actual simultaneous-residency limits even if the registry contains thousands of model names;
- switching a model mid-session generally changes logits/tokenizer/state and cannot be silent;
- a service/router only adds value when using the **same TaskCoordinator, ResourceArbiter, RegionExecutionProvider and TaskDriver** that the local CLI uses.

### 1.1 Ollama CLI and lifecycle: simplicity baseline

- [Official Ollama CLI reference](https://docs.ollama.com/cli) (accessed Oct 12, 2026) documents `ollama run <model>`, `pull`, `ls`, `ps`, `rm`, `stop`, `serve`, and customization through Modelfile. It also describes multimodal prompting and embeddings from CLI.
- [Official Ollama FAQ](https://docs.ollama.com/faq) says it supports **multiple resident models when memory allows**, multiple parallel requests for a model, queueing when a new model cannot load until another idle one unloads, a configurable maximum resident model count and a `keep_alive` control. Its guidance explicitly ties parallel context to additional memory use. These are **product behavioral precedents**, not fixed counts/settings zerogravity must copy.
- [Modelfile reference](https://docs.ollama.com/modelfile) includes `FROM`, `PARAMETER`, `SYSTEM`, `TEMPLATE`, `ADAPTER`, `LICENSE` and other customization rules. Its parameter overlay and model variant alias are useful convenience concepts; copying its exact syntax/API without source/license review is **not necessary** for an original Rust design.
- [Ollama API OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) says its compatibility is a **subset** of OpenAI endpoints, including local `/v1/chat/completions` and `/v1/responses` examples, vision and stream concerns. Local SDK examples use a placeholder API key ignored by server; zerogravity should make **local-vs-remote authentication explicit** rather than assuming any token is secure.
- [Ollama native API](https://docs.ollama.com/api/introduction) separates its API from OpenAI compatibility; zerogravity likewise should have a small native diagnostics/model-control API distinct from interoperability endpoints.

**Generalized lessons:** stable single binary/CLI, free-standing run vs long-lived daemon behavior, model snapshots/inspect, content-addressed ModelStore, automatic on-demand load/unload under physical memory safety, configurable model profiles and optional keepalive pins, and one API endpoint over many resident/prepared models. A common anti-pattern is silently overcommitting parallel context allocations.

**License note:** Ollama is publicly MIT at its repository root historically, but verify the exact code/files/revision and notices before reusing any source; documentation and model artifacts have separate rights. Model code and remote registries are not user-approved implicit download permissions.

### 1.2 Lightweight native runner/router prior art

- [llama.cpp CLI docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md) show a simple model path / prompt CLI and optional connection to a local model server.
- [llama.cpp server main README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) documents an integrated **router mode** with model source cache/dir/presets, on-demand load/unload, a max concurrently loaded models limit and OpenAI-compatible chat/completion/responses/embeddings. The README distinguishes a router's model selection by API `model` from execution in an instance.
- [llama.cpp server developer docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md) describe multiple inference server instances behind one router. **Zerogravity's direction differs deliberately:** support many model handles/resident plans/replicas through one shared inference engine/ResourceArbiter/TaskCoordinator; a model-specific subprocess may be an optional isolation/fault-domain policy, not a required C++ or per-model inference engine clone.
- Other native runner products and Rust model serving libraries under the original prior-art catalog may support quant formats/tokenizer/hardware backends, but are not canonical engine owners.

**Generalized lesson:** multi-model registry, loadable resident instances and API routing are **distinct but composable interfaces**. Model-level routing should never preclude native multi-request batching *within* a model, or cross-model scheduling when safe.

## 2. OpenAI-compatible APIs must be pinned to an explicit tested subset

- [OpenAI Responses API reference (accessed Oct 2026)](https://developers.openai.com/api/reference/resources/responses/methods/create) includes structured inputs, streaming and optional store/background/sequence semantics; endpoint evolution is fast. [OpenAI Responses overview](https://developers.openai.com/api/reference/responses/overview) differentiates stateful conversations/tool requests. [Streaming events reference](https://developers.openai.com/api/reference/resources/responses/streaming-events) defines typed `response.*` events, not simply the same chat `delta` SSE format.
- [OpenAI Python client](https://developers.openai.com/api/reference/python) and [official streaming guide](https://developers.openai.com/api/docs/guides/streaming-responses) provide useful client-conformance targets. The first-party SDK may send an API-key header even against a localhost clone; do not interpret that as proof remote caller is authorized or leak in log.
- Compatibility contract has **endpoint-level, modality-level and parameter-level coverage**, not a single boolean "OpenAI-compatible". Chat Completions and Responses differ in schema, tool event handling, conversation store, usage and streaming events; multimodal ASR/TTS/image APIs have additional format/codec and task rules.
- Remote API compatibility is **not** permission for the runtime to execute tools, invoke built-in web search, retrieve private files, or store conversations; native source TaskDriver can generate structured tool calls but executing tools is an external permissioned application boundary.
- Some API parameters are model or backend capability dependent (e.g. response format/JSON schema, seed reproducibility, logprobs, parallel tool calls, audio output, context, requested sampling). Unsupported fields should receive **typed errors** or have explicitly documented safe defaults, never be silently reinterpreted.

**Generalized lesson:** canonical typed `TaskRequest/StreamContract` with protocol adapters that preserve semantics, streaming/usage/finish reason, auth and explicit per-endpoint conformance fixtures. Default privacy-first state retention differs from a cloud provider's current storage defaults.

## 3. Multi-model lifecycle and routing architecture research

**Resident model vs registered model:** even if thousands of manifests/aliases are visible in a catalog, only models admitted by physical ResourceArbiter can have weights, compiled programs and KV/SSM state resident simultaneously. An alias is not a physical replica.

**Routing is not one algorithm:**
- **Exact model resolution:** deterministic `model` alias → pinned checkpoint+quant/adapter/numeric identity, no silent substitution of another family, checkpoint or quant.
- **Replica balancing:** choose among semantically equivalent prepared model replicas using queue/SLO/cache affinity/admitted physical capacity and same source artifact; avoid naive round-robin migration of stateful sessions.
- **Task and phase routing:** embedding/reranking/ASR/TTS/vision roles select *task-compatible model definitions* and TaskDrivers, with explicit task/model policy; some compound tasks run multiple named models under one TaskProgram.
- **Alias/policy selection:** a special configured `auto` alias may select **different model identities** by cost/capability/quality if the caller explicitly permits it, with actual selected model and semantic change visible. This is not equivalent to replica balancing.
- **Device/machine placement:** once model selected, choose compatible executable/cost topology from existing placements; do not copy an inference request to a separate gateway runtime.

**Load balancing and capacity:** model warm/load JIT and quant/layout conversion cost, cache affinity and session stickiness, device power/firmware and memory, multiple concurrent model and model-state reservations, fairness and queue backpressure, on-demand load/unload (only idle/unpinned) and prewarming. Protect against thrashing from alternating large model choices. A scale-to-zero model may need an explicit saved/snapshot/replay guarantee before evicting its live session's mutable cache.

## 4. Security, model provenance and user experience

Simple local CLI and API defaults should be **loopback only**, with remote exposure only on explicit bind and authenticated access. Remote peer/tunnel enrollment is a **separate authorization** from serving the HTTP API. A model alias cannot escalate model license/data locality restrictions or select an unauthorized remote node. The local user might have downloaded proprietary/private weights: automatic model pulls and cross-host artifact copies need explicit authorization and verifiable hashes.

**Default usability** means:
- `run <model>` handles model resolution from local catalog/path and safe backend auto-selection; if remote artifact needed, prompt/explicit pull before any network download, with license and size exposed.
- all behavior adjustable through **profile/config/CLI/approved request fields**, and `inspect --explain` reveals actual effective settings, source and reason for fallback.
- avoid a mandatory admin dashboard, cluster scheduler or per-model `serve` process; any resident daemon/service is an optional session-sharing deployment form, not the definition of the runtime.
- keep CLI default concise; use `--verbose`, `--explain`, `config show --effective` for progressive disclosure.

## 5. Source distinction from requested deliverable

This document identifies **external references only**, including MIT source candidates. It does not make the project a wrapper around another runtime or API gateway, and published model-server behaviors are not internal zerogravity tests.

**Normative destination:**
- `MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md` — explicit catalog/model-profile/resident-replica/router/scheduler/memory/placement ownership.
- `ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md` — proposed one-command local UX, configuration, OpenAI endpoint subset and conformance/failure policy.

**Status:** research + designs only; no code, model downloads, API tests or benchmark runs executed.

````

## Source: .agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md

Original UTF-8 bytes: 7227; SHA256 `57ab09f4d823ad78c9704d5b81ce9decd7ba52f2e7e06c2f71177c7c4b08b05f`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Prior art, provenance and architecture neutrality

**15 October 2026 — LICENSE SELECTION:** The user selected [MIT for Zerogravity's
own source and documentation](../../LICENSE) and approved root and GPU package
metadata `license = "MIT"`, with founder/contributor copyright. This **does not**
automatically relicense externally sourced files, model weights, tokenizers,
codecs, datasets, precompiled GPU kernels, or dependencies. [Contribution and AI
development policy](../plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md)
records human-led design and AI-assisted implementation with a docs-first
contribution workflow.

**Date:** 2026-10-12. **Authority:** project policy for engineering design
documentation, reflecting explicit user directive U-017. **Not legal advice.**

## 1. From-scratch architecture rule

The design is an **original modular AI computation/runtime architecture**. It must
**not** depend conceptually, organizationally or programmatically on a named
external engine, library's internal data structures, model-specific server, or
proprietary project.

Zerogravity is a from-scratch rebuild informed by accumulated prior-art concepts,
learnings and research. **No source from any prior-art project was copied or
linked unless it is directly vendored (for example, CubeCL (vendored)), which
retains its upstream license.** Named project references are limited to the
acknowledged inspiration projects in the root README (llama.cpp, vLLM, SGLang,
Ollama, gufo, strata, maya, CubeCL, pi and dwarfstar4) and to the mechanism names
below.

- **Canonical normative design** (master architecture, composition spec,
  contracts, implementation roadmap, acceptance criteria, agent prompt) must use
  **mechanism/capability/interface names**, not "Project-X-style runtime" or
  `ProjectXPlan` as a type. Legitimate backend APIs and model/hardware
  identifiers (e.g., ROCm, CUDA, Vulkan, GGUF, MLIR as a source language, model
  checkpoint ID, XDNA2, gfx1151) identify actual protocols and deployment
  targets, not the architecture's parentage.
- **Research-only files** keep original technical analysis, exact numbers, limits
  and negative findings, written as our own ideas; external implementation
  specifics are not automatically the design to copy.
- **Code reuse**: any candidate source must be audited at the exact pinned
  revision/file for third-party notices (GPL/Apache/BSD/LLVM exceptions) before
  any import; model weights/device binaries have separate rights. No automatic
  copy/import.
- **Proprietary/nonpublic work**: black-box published performance/behavior can
  motivate experiments, but no illegal source access, cloning or fabricated
  knowledge. Patents/trademarks remain independent.
- **No vendor dependency surprise**: XRT, ROCm, CUDA driver/maths FFI and compiler
  toolchains are optional device adapters, never semantic or server dependencies.
  The requirement for Rust-owned kernel source remains in force.

## 2. Reuse and license review

A source repository being permissively licensed at its root is not sufficient to
import a specific kernel, transport/RDMA vendor binding, model checkpoint, device
program or tunnel client. Before reusing any source:

1. Confirm the exact upstream license file, `NOTICE` and any file-level header at
   the pinned revision.
2. Confirm the user's Rust source-ownership policy for any foreign device kernel.
3. Record attribution/NOTICE/SBOM; treat model weights, tokenizers and datasets as
   separately licensed.
4. Obtain explicit user approval where source-language/licensing constraints
   require it.

Vendored components retain their own license texts in place (for example, the
vendored CubeCL packages under `crates/zerogravity-gpu/vendor/`). The
Zerogravity-authored source remains MIT.

## 3. Mechanism taxonomy: use these instead of project names in design

| Prior-art mechanism | Canonical neutral architecture term | Component owner |
|---|---|---|
| serving-token scheduler / continuous batching | **BudgetedWorkScheduler** and `StepPlan` | Runtime policy |
| paged/prefix KV and ranked prefix index | **StatePagePool + PrefixIndex** | State/Memory |
| general graph math execution | **PortableSemanticExecutor** | Semantic/Kernels |
| single-model extreme fused kernel | **SpecializedRegionProvider** | Region optimizer / KernelProvider |
| GPU/RAM/SSD expert pool | **TieredSparseParameterStore** | Resource arbiter + IO |
| GPU+GPU routed experts | **AsymmetricExpertOwnership** | Placement/CIR |
| GPU+NPU partial GEMM | **SplitProjectionCompute** | RIR/KIR/CIR |
| GPU graph eager/capture | **CompiledDispatchBundle** | CIR device execution |
| high-level/low-level math compiler | **TypedSemanticGraph + RegionKernelLowering** | SIR/RIR/KIR |
| async stream/resource compiler | **ResourceCommandGraph** | CIR |
| bounded equality rewrite search | **PureRegionRewriteSearch** | Offline optimizer |
| speculative branches | **DraftVerifier + StateTransaction** | TaskProgram/runtime |
| hybrid state prompt reuse | **FullStatePrefixSnapshot** | State codec/coordinator |
| content-addressed model downloads | **ArtifactStore + TrustedManifest** | Product/artifact |
| multimodal ASR/TTS/image/video pipelines | **TypedTaskProgram** | Task orchestrator |

**Enforcement:** before implementation or design edits, a reviewer should be able
to delete the entire prior-art and research corpus without losing any needed
mathematical/semantic/device/runtime specification. If code depends on a named
project's proprietary type, source layout, cross-project environment variable or
process lifecycle, the architecture is not yet generalized.

## 4. Prior-art and research preservation

Research documents live under [`.agents/prior-art/`](.) and
[`.agents/research/`](../research/) and keep their technical analysis, numbers,
limits and negative findings. They are **non-normative**: a source
implementation's particular flags or defaults are not automatically the design to
copy. They are retained as searchable engineering context, written as our own
ideas, not as links to incorporated code.

## 5. Concrete developer checklist

1. Define **capability/mechanism** in project-neutral language and identify
   semantic/non-semantic responsibility.
2. Cite named prior art only where the acknowledgement policy permits it.
3. If reusing actual code, verify **that file's** license/NOTICE/derivative
   attribution; confirm Rust-owned source and user consent for foreign device
   kernels.
4. Implement behind our own versioned capability interfaces; do not create adapter
   wrappers around a whole external model server.
5. Check fallback and new-model/new-device/new-task extensibility tests.
6. Revalidate numerical quality, memory safety and performance against an
   independent baseline; a source author's results are hypothesis targets.
7. Update the decision register when user policy changes, benchmark records when
   evidence changes and the license table when actual source pins change.

**Final principle:** external work is prior art and a validation benchmark; the
executable architecture belongs to Zerogravity and must remain coherent if every
external project stops developing tomorrow.

````

## Source: .agents/prior-art/RESEARCH_SOURCE_ATLAS.md

Original UTF-8 bytes: 19294; SHA256 `fa8c73dcd109e72f02cc1d5af2781ac49d7ad0543ee08cc8ec63c31f7b3f1c96`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Source atlas — architectural research preserved for implementers

> **Research/provenance only (v3, 2026-10-12).** Project names, URLs, specific implementation techniques and source-author benchmarks in this document are **attributed prior art, not normative zerogravity runtime or compiler types**. The independent governing architecture is [Modular AI systems](../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [Composition contracts](../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md), and [Prior-art license policy](PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md). External projects never become a requirement to clone a model-specific inference engine; technical/measured evidence remains available for independent validation.


Research consolidated as of **2026-10-10**. Source URLs are starting points, not immutable references unless a SHA is given. At implementation time obtain a commit pin, review current code and license, record checked-out SHA and test data hash. External benchmark numbers here are project-author-reported, **not zerogravity results**. Never compare different model family/quant/context numbers as a ranking.

## A. Main engines — direct user selections

### vLLM — https://github.com/vllm-project/vllm
**Study:** V1 token-budget scheduler and request computed-token count, waiting/running admission, KV block/paged cache, prefix lookup, graph capture, per-shape kernel backend selection, tensor/pipeline/expert parallelism, concurrency. Relevant candidate components: the V1 scheduler, KV cache manager, worker/model runner and attention backend interfaces; verify paths after checkout. **Transfer:** token-aware unified request scheduling, typed memory ownership, execution variant selection, clean distributed resource accounting. **Avoid:** depending on Python/PyTorch as zerogravity engine and assuming all vLLM kernels optimized equally across AMD/NVIDIA/model. Core P09/P10/P16/P19.

### SGLang — https://github.com/sgl-project/sglang
**Study:** RadixAttention/radix prefix cache, HiCache GPU/RAM/storage tiers, overlap scheduler, speculative decode (MTP/EAGLE/DFlash...), P/D disaggregation, expert-parallel grouped GEMM and communication overlap. **Transfer:** prefix-aware admission, cache-aware balancing and async execution, independent prefill and decode resource decisions. **Avoid:** conflating KV/CPU prefix-cache tiers with Maya's *model expert* tiers, or believing P/D network split improves one APU automatically. P09/P10/P15/P16/P19.

### llama.cpp / GGML — https://github.com/ggml-org/llama.cpp
**Study:** GGUF format, tensor metadata and quant layouts, model registry and graph build, CPU SIMD, CUDA/HIP/Metal/Vulkan backends, partial GPU offload, mmap and CLI/HTTP ergonomics. Read the pinned GGML and llama model source and GGUF/GGML backend implementations. **Transfer:** common model packaging, many architecture forms, quantized block exactness, native UX, CPU reference/optimized variants. **Avoid:** copy-pasting unsupported backend kernels without licenses, using a general graph interpreter in hot path and assuming a GGUF loader validates mathematics. P03/P06/P08/P17.

### Specialized gfx1151 engine
**Study:** published benchmark/config/quant/quality docs for Qwen3.8-Flash-Next optimized for AMD Strix Halo `gfx1151`. Reported example: ~1584 prefill tok/s at 8K, ~1517 at 131K, ~37.6 short-context serial tok/s and ~55–56 coding-agent speculative tok/s on 128 GB Strix Halo; other published versions/configs differ. Native checkpoint ~62 GiB, n-gram lookup ~47.7 GiB paged, strong attention and mixed quant tuning. **Source access caveat:** published repository does not provide core engine source; proprietary engine/EULA must not be copied or reverse-engineered unlawfully. Use output/perf methodology as black-box reference and independently implement known public model mathematics. P01/P13/P15.

### Maya — https://github.com/mw00/project-maya
**Study:** the GLM fast path, AMD/GLM platform notes, expert source, cache/CPU expert implementation and tests. Based on Strata-derived concepts, adds GLM-5.3-Flash port with KDA, sparse MLA, mHC; GPU device-side (layer,expert)->slot lookup, host mapped route miss ring, pinned staging, async disk reads, per-layer slot partition/generation-safe eviction. Reported Strix 128 GB GLM Maya-S ~210–217 prefill and ~17.8 decode; 12,096 experts nearly all in pool on cited run. Dual AMD two-GPU ~33 tok/s with MTP; not equivalent to single GPU. **Transfer:** detailed cache safety/IO handoff and model-specific reference parity, selective CPU expert lane. **Limit:** not a general multi-client continuous-batching vLLM peer; AMD experimental text-only paths in inspected snapshot. GitHub connector inspection saw main blob SHA `56bbd3af0f14720294a8ff72046a7bf91860bf4e` for indexed results; capture current commit before writing code. P12/P14/P16.

### Project Strata — https://github.com/mach10x/strata
**Study:** originally Qwen3.8-Flash-Next specialized model with VRAM expert cache, pinned RAM and SSD, GPU/CPU expert overlap, large n-gram table, speculation and heterogeneous layer splitting. **Transfer:** dynamic placements depend on PCIe bandwidth, CPU SIMD and batch shape; n-gram lookup has distinct IO policy from MoE matrix weights. **Limit:** specialized runtime; older/newer engines may diverge. P10/P12/P13.

### Gufo — https://github.com/gufo-org/gufo
**Study:** published model benchmarks, quality and experiment documents, the ROCm executor, GDN/sparse attention, quant MoE, GPU state snapshots, MTP sampling, multisequence batch. October 8–9 author-reported `gfx1151` 128 GB Qwen3.8 UD-Q4_K_XL: AR pp2048 ~1700.52 tok/s, tg128 ~25.95 at d0; d128K pp ~1478.66, tg ~24.92; MTP repetitive ~60.39 vs mixed ~31.32; C8 AR aggregate ~108.68. Historical llama.cpp comparator pinned and one warmed sample/point; *not* independent verification. **Transfer:** session-private recurrent/indexer state, speculative rollback per accepted prefix, long context, real workload benchmarks and rejected-kernel experiment discipline. **Limit:** GPU kernels C++/HIP today; check license before code copying. P01/P11/P13/P15/P16.

### DwarfStar (antirez/ds4) — https://github.com/antirez/ds4
**Study:** `ds4.h`, `ds4.c`, `ds4_rocm_compat.cu`, `rocm/ds4_rocm_runtime.cuh`, `ds4_engram.c`, `ds4_ssd.c`, `ds4_tp.h`, platform/model/speculative-decoding/distributed notes. GitHub connector read main whose newest visible commit was `fc80bd695da76ee14bcb86820bf900a5cdfda806` (2026-10-07); current head may differ. MIT `LICENSE` with GGML attribution. Native C/ObjC/CUDA/HIP without full GGML runtime, custom GGUF packs and model graphs, Metal/CUDA/ROCm, DSpark, time-compressed DeepSeek attention (128 raw local window, 4:1 indexed /128:1 unindexed older history), large SSD n-gram/Engram row tables, TP+PP and tool-aware session snapshots. `gfx1151` DeepSeek V4 Q2 prefill tuned ~187.26 → ~292.86 tok/s warm 4K frontier, specific ROCm model and version. **Critical:** Qwen3.8 path documented Metal/CUDA only, **not ROCm** in inspected Qwen docs; `ds4_rocm_compat.cu` returns no support for decode-island graph capture; single-device ROCm eager. Server modes vary between native batches and ordered fallback; concurrency != native batching. **Transfer:** compact narrow Rust API, model-specific graph, SSD tiering and fixed slabs, state and distributed failure contracts. P03/P09/P10/P13/P14/P15/P19.

## B. Complementary engines and tools

### A CPU/GPU heterogeneous engine
CPU AVX-512 + GPU heterogeneous execution, grouped/layerwise prefill and native large-model weights; GLM-5.3 official FP8 fits machines with hundreds of GB RAM, not automatically Strix 128 GB. Integration into SGLang is an architectural lesson about specialization + generic serving. P12/P14.

### A quant-aware inference project
Quant-aware MoE separate decode GEMV and large-prefill grouped kernel paths; its Qwen Flash-Next report measured ~16% prefill gain on cited kernel change. Understand row-count thresholds, per-expert launch overhead, grouped padded GEMM and atomic reduction/numerical tradeoffs. P12/P16.

### A kernel-dispatch library
Kernel library: paged/sparse/MLA attention, MoE grouped kernels, sampling, JIT/autotune by shape. Primarily CUDA/NVIDIA reference for kernel-dispatch architecture. P05/P07/P20.

### A vendor graph compiler
Model graph extraction/recognition (AutoDeploy), fused attention/MoE transforms, device graph compiler, parallel/disaggregated serving. Study legalization and typed graph optimizations; do not require a vendor graph backend. P04/P05/P19.

### Apple unified-memory frameworks
Apple unified memory and lazy tensor execution, useful analogy for CPU/GPU shared addressability. AMD UMA physical behavior is different. P07/P10.

### An independent Rust serving project
Rust inference+serving, continuous batching, many models, a Rust tensor library; competitive dense examples but reported substantial gap on certain Gemma 4 MoE prefills. Lesson: Rust is not intrinsically slow, kernel/operator coverage remains crucial. P00/P03/P09/P17.

### A Rust tensor library
Rust tensors/models and CPU/CUDA/Metal paths. Useful numerical reference/portability ideas; not a substitute for whole server optimizer. P02/P06.

### CubeCL — https://github.com/tracel-ai/cubecl
Rust GPU kernel language + compiler/intermediate representation, CUDA/HIP/Metal/Vulkan/WebGPU backends, tile/reduction/matmul primitives. Alpha/evolving; judge `gfx1151` generated instructions before selecting as mandatory. Need potential low-level architecture intrinsics for peak kernels. P01/P07/P20.

## C. Research interpretation constraints

- **Do not reproduce proprietary engine code**. Public algorithm documentation and original model papers can inform independent implementation.
- Strong source-level examples include Gufo/Maya/DwarfStar; the performance figures are maintained by those projects and may be single-run/config-specific.
- Specialized-engine and Gufo model quantization likely differ; do not conclude engine-vs-engine win from unmatched checkpoint speed.
- Maya 321B total model routes only 18B active parameters/token; working set/residency and quantization drastically affect throughput.
- A CPU/GPU shared address is not free memory bandwidth; unified RAM, GTT and PCIe must be measured separately.
- vLLM/SGLang throughput on datacenter GPUs does not establish same throughput on Strix Halo without kernel parity.
- Model families and tensor formats evolve quickly. Pin sources when building against them. Check legal code/data licenses, distribution rights and trademarks separately.

## D. Minimum reproducibility manifest for future research additions

```text
name / url / checkout SHA / license reviewed on date
model checkpoint + quant SHA / tokenizer + template version
hardware GPU arch / VRAM + physical RAM / driver / power
algorithm source path + call graph / model semantics contracts
bench inputs / context / concurrency / sampling / acceptance
measured prefill, decode, TTFT, P95, memory, error/quality
what remains unverified / differences from other studies
```

On adding a new project, record **transferable mechanism**, **limitations**, **source status**, **performance evidence** and **license restrictions**, then map it to the relevant subsystem spec and experiment rather than simply adding to a long list of names.

## 2026-10-10 integrated v2 addendum — additional first-party compiler and source programs

The canonical **local research corpus** is the external read-only research library and its pinned-revision manifest, containing source-pinned vLLM, SGLang, llama.cpp, Strata/Maya, DwarfStar and Rust ecosystem evidence. Its historical master and implementation plan are **reference research**, not a second authoritative project implementation plan. The Gufo source, modern Gemma4 and Qwen3.5+ requirements in this project are important complementary additions not fully addressed by old council.

**New first-party compiler evidence, accessed 2026-10-10:**
- An ahead-of-time ML compiler's Stream/HAL (external reference removed) — explicit async resources, device affinity, command/timepoints and schedule-allocation, inspiration for small Rust `CIR`.
- A tensor compiler's Relax→TensorIR and MetaSchedule (external reference removed) — split semantic operator graph from target tiling/schedule and measured candidate search.
- MLIR effect/conditional speculation model, https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/ — pure SSA reordering versus mutable resource effects; optimizer legality model.
- A Rust graph-search project and an e-graph compile-search warning (external references removed) — minimal lower-level primitives, symbolic shapes, bounded pure-region rewriting, compilation time as a risk.
- A kernel-dispatch autotuner v2 (external reference removed) — 2026 eager-vs-captured execution can change best tactic; benchmark whole serving regime, not bare GPU device time.
- NVIDIA Rust GPU codegen announcement Sep 8 2026 (external references removed) — a Rust tile compiler (tile stable Rust) and a Rust SIMT/PTX compiler (SIMT PTX early alpha), supplement CubeCL rather than freezing a CUDA vendor C++ kernel escape.
- vLLM Hybrid KV cache docs https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/ — group-specific cache hit and intersection; documented implementation limitations inspire generalized 3+ state domain coordinator.

**Research defects flagged:** earlier "48 byte" disk-KV ABI internally sums to 54 and declares impossible 5-byte magic in u16; its "any model any engine runs" fallback scope is logically impossible for unknown ops; dated local box, v1 phase/release mismatches, compiler pin and unapproved license are not authoritative. Use `ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md` for exact D-01…D-20 ruling. Pin exact current compiler/project commits before implementation, verify licenses; the specialized engine source is not public.


## 2026-10-12 integration — distinct upstream Gufo and XDNA2 fork

New user sources: https://github.com/mikealanni/gufo (default fork-enhancements, commit 8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7) and https://www.reddit.com/r/StrixHalo/comments/1x3gtdh/qwen38_flashnext_prefill_past_1800_toks_on_strix/ (October 11, 2026). Upstream Gufo is https://github.com/gufo-org/gufo (main inspected f37fb012281bbd58054b1082933153bbec303106). Treat the fork and upstream as separately pinned sources. The upstream includes Qwen3.8/DeepSeek LLMs, ASR/TTS and emerging image/video models; the fork adds a **different, same-SoC GPU+XDNA2 cooperative prefill** capability.

Verified fork source: partial Qwen3.8-Flash-Next GDN z, attention query/gate, shared-expert gate/up projection rows run on an XRT/AIE NPU while remaining rows and consumer GPU kernels continue. Its NPU GEMM fixed geometry is 4096×2560×1024 using 9 bytes/eight block-float values; HIP-mapped pinned host/XRT userpointer BO, host submit/wait callbacks and first-use output join. Author reports UD-IQ4_XS 16K prefill GPU-only 1,698 vs NPU+GPU 1,821 tok/s (+7.2%), with only 30 greedy prompts checked. This is *not* a decode speed or zerogravity result.

**Safety:** author reports machine-check/host hard hangs during fabric-clock transitions; 12 clean subsequent runs with manual held FCLK do not prove general safety. Never enable automatically on unqualified host or run privileged clock/power tweaks. **Rust policy:** AIE kernel comes from C++/Python source and compiled xclbin, not a Rust-authored kernel; a Rust-owned device codegen path must be proven or stay blocked. Full source/ABI/quant/clock analysis: [GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md). Generalized design: [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

## 2026-10-12 addendum — heterogeneous accelerator execution families

**New comprehensive first-party research index:** [ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md](ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md), referencing DGX Spark/GB10, other workstation and datacenter Blackwell/B200/B300/GB200/GB300, Google's TPU6e/TPU7x and graph/PJRT stack, Cerebras wafer-scale CS-3/WSE-3, Groq deterministic LPU, AWS Trainium/Inferentia, AMD CDNA MI350/MI400, Intel Gaudi, Tenstorrent, SambaNova RDU and other graph/dataflow ASIC families. Distinguishes the **Cerberus** adaptive parallel decoding paper (not Cerebras hardware). All external source names, repository licenses, URLs, vendor claims and hardware specifics remain **research-only provenance**.

**Project-neutral adaptation:** [ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md](../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) sets one RegionExecutionProvider across NativeKernel/CapturedGraph/AOTDeviceGraph/StaticDataflow. Source analysis alone is not proof a particular vendor compiler or StateBridge can support the runtime. Opaque model API services are different from native model-region lowering. No SDK or source was installed/copied or tested during this documentation pass.


## 2026-10-12 new distributed state/cache/tunnel research

Full source-linked prior-art study: [DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md](DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md). It compares P/D and KV connector frameworks, remote stateful attention versus remote KV reads, attention/FFN and heterogeneous MoE split, token-range sharded softmax, cache-centric state tiering, consumer peer layer clusters, authenticated TCP/QUIC/direct/relayed encrypted overlays and physically validated RDMA. Source models and published speedups remain external author research, NOT zerogravity achievements.

**Normative derived design:** [DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) and [REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md](../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md). Single TaskCoordinator, Source SemanticProgram, Prepared StateRoot, four orthogonal compute/parameter/state/cache placements and TransferPlan. No mandatory network runtime for standalone.

## 2026-10-12 additional model-router/local CLI and API source research

Source-specific [multi-model router and API research](MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md) compares current local model CLIs, model keepalive/load policy, model-specific settings/presets, built-in multi-model router and official OpenAI Chat Completions/Responses/embeddings and streaming SDK schemas. **These are external prior art**, not source copied or capabilities tested by zerogravity.

Normative independent interface targets: [ModelRouter/ResidentModel lifecycle](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) and [one-command CLI/config/OpenAI API contract](../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md). The router handles coarse exact-model admission and equivalent replica selection; TaskCoordinator/DeviceWorker remains the only native model execution/spaces state authority. No model-specific inference engine or gateway required.

````

## Source: .agents/prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md

Original UTF-8 bytes: 16124; SHA256 `027703791f1604a822a7d3ee332e1f746bf9db1bece7f65b447c94be5103b466`.
Resolve embedded relative paths against the original path, not this volume.

````text
# SSD Streaming Research Dossier — DS4/DwarfStar and related offload prior art

**15 October 2026 • Source-grounded design research, no Zerogravity SSD model execution.**
**Question:** how to make SSD-backed weights and state *naturally* available inside one native Rust tensor/resource fabric without sacrificing the fully resident fast path or GPU lifetime correctness?
**Answer:** bind exact source tensor identity independently of physical residency; source-specific expert/lookup/session storage policies share asynchronous IO, device event/lease safety, bounded physical memory and compiler prepared execution. Resident decoding becomes a zero-extra-IO specialization, while SSD streaming is an explicitly costed and verifiable materialization path.
**Governing design:** .agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md.

## 1. Distinguish four source-observed techniques

1. **Dynamic routed expert weights**: DS4 pages missing experts from immutable GGUF source and caches hot experts in bounded GPU memory. It is not generic stream-every-weight prefill.
2. **File-backed sparse row lookup**: an independent file-backed Qwen 47.7GiB n-gram table and DS4 95GiB Qwen PLE/189GiB DeepSeek Engram may remain file-backed and fetch small selected rows; not a full tensor H2D copy.
3. **Persisted session/KV checkpoints**: both engines can save compatible committed conversation/prefix state to SSD and restore useful prefixes. Different mutability and tenant/lifetime rules from model weights.
4. **Unpinned dense trunk as capacity fallback**: the file-backed lookup design documents a model-trunk unpin option that frees memory but substantially lowers decode and adds cold reload latency. This must be opt-in/capacity tier, not a zero-cost equivalent.

These can be implemented over one SourceExtent/BackingTier/ReadyTensorLease and typed StateSnapshot substrate, but require **different caching and finality policies**.

## 2. Antirez DS4 / DwarfStar streaming

Official docs:
- https://github.com/antirez/ds4
- https://github.com/antirez/ds4/blob/main/docs/SSD_STREAMING.md
- https://github.com/antirez/ds4/blob/main/ds4_ssd.c
- https://github.com/antirez/ds4/blob/main/docs/SERVER.md

DS4 distinguishes fully resident from explicitly enabled SSD streaming, and says resident is faster when feasible. Its SSD mode reserves enough memory for nonrouted weights, attention KV/recurrent state, graph and prefill transient allocations, and spends only remaining budget on dynamic expert slots.

- Automatic cache byte budget and bounded expert count are requested capacities; effective cache may be lower because mandatory model/OS/graph memory comes first.
- Source backend/model coverage differs: Metal DeepSeek/GLM, CUDA routes and ROCm GLM5.2/5.3 (at documented version) are not proof every GGUF/quant streaming works.
- Optional full routed prefix layers are allocated first; dynamic cache otherwise may range across all layers.
- A larger cache can displace nonrouted mandatory weights or session KV on unified memory, slowing decode despite more expert hits.
- It preloads experts under supported paths; cold/preload controls are benchmarking tools, not universal speed defaults.
- Streaming speed is sensitive to cache hit rate, random SSD IOPS, verified expert byte group, staging and target verification width.

DS4's publicly reported M5 Max 128GB measurements (September 6 2026; author setup, **not Zerogravity**): GLM5.3 Flash Q4_K file 177.77GiB; its SSD-streamed initial prefill 121 tokens/s, continued prefill 104 tokens/s, generation after two frontiers 11.9/14.9 tokens/s. DeepSeek Flash Vision Exp 145.26GiB: initial 300, continued 263, generation 11.9/19.3 tokens/s in a single run. These are source workload-specific, not general SSD/target performance promises.

### 2.1 DS4 issue #384: encoded-but-not-executed GPU command uses evicted expert slab

https://github.com/antirez/ds4/issues/384

This is the most valuable correctness negative test for Zerogravity. DS4 recorded a deterministic argmax flip at constrained 16–20GiB cache in Metal layer-batched decode. The CPU producer encoded ALL layer commands first; a later layer's missing expert caused a prior layer's unsubmitted slot to be overwritten before its GPU command actually ran. The GPU pointer remained valid, source CPU cache directory consistent and GPU completion truthful, but **the bytes were a different expert**. Prefill didn't show the issue.

Required architecture:
- mark expert slab reader lease as soon as a GPU command records/reference/captures that slot, not just when it is submitted or executing;
- protect the full set referenced by **every command in the currently unexecuted batch**;
- combine slot generation and physical event lease, stable pointers or validated indirection table;
- do not recycle until GPU completion proves quiescence;
- when no eviction victim exists, safely flush/await current batch or return backpressure; never overwrite a queued GPU reader;
- this is a test with two layers, cache capacity 1, encoded batch + later miss, so same reuse bug cannot hide behind CPU source checks.

The DS4 issue was observed on one configuration; read its resolution/source revision when porting but keep the invariant regardless of vendor.

### 2.2 DS4 issue #503: mixed-quant cache reset and prefill evicting decode

https://github.com/antirez/ds4/issues/503

On a mixed Q2/Q4 model, an original CUDA expert cache used a single expert-size assumption and cleared all slots when gate/down bytes changed, leading to ~0.9 tokens/s and repeated expensive reset. Another prefill path released all warmed decode cache for every prompt, even short tool-result followups.

Reported patch proposal: per-size-class slabs keyed by gate/down bytes, try staging allocation before eviction, use existing warm cache for prefill, and avoid evicting decode working set in long prefill. A second hazard was discovered: per-class N slots can overcommit when **every** class independently requests N. Correct design is **one authoritative byte budget across heterogeneous classes** with reserved arena/scratch and per-class weights as secondary policy.

Additional performance ceiling: a 97% hit rate could still copy ~2GB/token device-to-device into a compact buffer. The goal should be direct source-valid resident-pointer consumption when legal, not merely high cache-hit count.

Do not claim the issue's proposed patch was merged or that the source author measurements are Zerogravity results.

### 2.3 DS4 issue #629: optional pinned host-RAM second tier

https://github.com/antirez/ds4/issues/629

Research/work-in-progress proposed copying source expert bytes into bounded host RAM slabs as a second-level cache. The published CUDA path used O_DIRECT for SSD reads, and on buffered fallback advised cache eviction, so ordinary OS page cache wasn't retaining useful expert hotness. On a host tier hit, stream from RAM to GPU without random SSD IO.

Lessons:
- host cache is distinct from OS page cache and requires real RAM/pinned byte admission;
- on Strix Halo UMA, RAM+GPU may alias DRAM, and a redundant pinned duplicate could hurt memory efficiency;
- on discrete GPUs or direct-IO systems, dedicated hot RAM slabs may be economically useful;
- size by observed hot routes and global physical byte credits, not every class allocating a full budget;
- host cache fill/read must never overwrite a pinned IO or GPU reader buffer prematurely.
- Treat issue status as a source idea/benchmark, not guaranteed merged functionality.

### 2.4 DS4 disk KV session cache and direct-file lookups

https://github.com/antirez/ds4/blob/main/docs/SERVER.md

DS4 uses disk prefix/KV to avoid expensive refilling when clients resend shared conversation history, and keeps source static lookups file-backed for models such as Qwen3.8 and DeepSeek Engram. This is **not identical to expert streaming**:
- KV state is private, mutable, session-versioned and compatibly snapshot after StateRoot publication.
- Immutable lookup rows are model-specific hashed addresses from a verified large table, indexed by source token history and row schema.
- On distributed execution, disk prefix restore may require recomputing/redistributing state on each rank rather than pointer restoration, and numerical mode can differ.
- Disk persistence requires explicit rights/tenant scope and source/processor/kernel state compatibility to prevent cross-user history disclosure.

## 3. File-backed disk-only lookup and optional dense reload, not a blanket SSD-expert engine

Source documentation:

A prior-art file-backed design emphasizes keeping the tuned Qwen3.8 Flash-Next model **trunk pinned/resident** on gfx1151 to preserve decode speed. Its distinct 47.7GiB FP8 n-gram source table remains file-backed and is read on demand with bounded indexed rows; a cold 32K prompt may read some rows from SSD when OS page cache is cold, then subsequent hits benefit from file cache.

A setting to unpin the model trunk deliberately exchanges capacity for dramatic decode slowdown and extra per-prompt layer reload. The documented example contrasts ~37 tokens/s pinned with ~6 tokens/s unpinned and cold prompt weight reloads 7s warm file cache versus 25–50s after disk eviction. This is a **capacity-over-latency** mode, not a universal transparent equivalent of resident GPU memory.

The same design also separately supports persistent disk prompt/KV cache with incremental committed state snapshots. It records model/weights/config-specific disk directories and limits (historically default 64GiB disk cache). This is a third lifecycle (session state), not another immutable tensor expert slab.

Its memory discussion notes that the n-gram source table relies on OS cache and that host reserve and KV pool compete with that cache. The correct global planner considers mandatory trunk, session pool, prefix state, reclaimed page cache, physical GPU allocations and OS reserve, not a simplistic GPU MemFree number.

Another optimization in that design: in-place prompt snapshots can avoid copying entire 20+GiB KV state when only small position-free state changes. This is an effect/compatibility optimization under committed StateRoot, not permission to make raw GPU pointer snapshots durable.

## 4. Prior offloading system: one system across GPU, RAM and disk, optimize task regime

Background paper: https://proceedings.mlr.press/v202/sheng23a.html

A 2023 offloading system used block scheduling/offloading and compression to run enormous dense models on limited GPUs, optimizing **throughput-oriented large-batch** workloads via LP policy across GPU/CPU/disk rather than low-latency single-user interaction. Its paper explicitly warns that SSD offloading under small-batch interactive workloads can be much slower than fully resident GPU.

Zerogravity should incorporate the **latency/throughput regime distinction and cost-based placement**, not replicate a new offload engine or giant per-token LP solver. Model load/plan may solve a coarse constrained placement; hot path follows precompiled resident/demand materialization candidates.

## 5. Activation-aware expert caching and source tracing

A known activation-aware design demonstrates routed expert offload across SSD and host RAM using activation-aware expert hotness, tracing/prefetching and multi-GPU per-device caches. Its current open-source edition differs from paper target and may expose unsupported/no-op features depending on checkpoint and backend. Treat its code as a policy experiment, not a guarantee every model/quant/prefix KV offload works.

Design lesson: source router computes expert IDs; cache hotness/traces are **hints** for physical residency only. Model math remains unchanged. Multi-GPU uses per-device expert slot maps with one physical-budget authority and real cross-device/peer transfer receipts; remote network storage is separate explicit authorization.

## 6. Source comparison and exact gaps closed

| Capability | DS4 | File-backed lookup | Offload project | Expert-offload project | Required Zerogravity |
|---|---|---|---|---|---|
| Fully resident fast path | yes | explicit primary | possible | possible | direct kernel pointer, no IO overhead |
| SSD expert demand | supported subsets | mostly unpin trunk (different tradeoff) | layer/weight offload | expert offload | same TensorView, typed asynchronous materialization |
| On-disk sparse lookup | Qwen/DeepSeek source tables | huge FP8 n-gram | model-dependent | model-dependent | source-specific row cache, small IO |
| Disk KV snapshot | server cache | persistent prompt state | KV offload | configuration-dependent | frozen versioned StateDomain, private |
| Mixed quant cache | bug case issue #503 | source 4-bit quant variant | low-bit compression | per-device policy | size class, shared bytes, no thrash |
| Reader protection | bug case issue #384 | pinned trunk | stage block schedule | async transfers | pin from encoding through GPU event |
| Host RAM tier | proposed issue #629 | OS file cache primary | planned multi-tier | yes | optional RAM cache for source hotness |
| Automatic capacity | cache constrained by other state | host reserve/KV pool | plan optimizer | dynamic experts | conservative physical+OS admission |
| Interactive service | cache-sensitive | optimized resident | weaker | backend/task specific | truthfully measured QoS and refusal |
| Dedicated engine | narrow C source | gfx1151-specific | Python research | Python tensor framework | NO; one Rust engine |

## 7. Numerical source and performance honesty

A fully resident model and its SSD-streamed twin should produce identical source-exact output given identical model/quant/numeric policy and state. But if device layouts/dtype/reduction order change with batch/route/IO mode, numerical equivalence requires separate source-based validation and ModeReceipt. GPU event Ready only proves command completion, not that an evicted expert slot contained the expected bytes (DS4 #384); original model logits/hidden-state tests under extreme cache pressure are mandatory.

Include cold/warm SSD, random vs sequential IO, page-cache hotness, host staged vs direct SSD, source extents, cache shape, active model count and session count in performance evidence. A model that fits on SSD may only generate a few tokens/s; that is a valid capacity feature with a truthful slowdown, not a failed abstraction.

## 8. Strictly not adopted

- No separate DS4 or other prior-art offload engine or Python wrapper in Zerogravity runtime.
- No default force of all model tensors onto SSD, no arbitrary GPU page fault inside unsupported kernel, no unbounded per-token IO worker or global source scans.
- No fixed magic 80% cache reservation from one ROCm result; memory/OS headroom is measured/conservative and workload specific.
- No automatic process killer or OS page-cache purge based on prior-art anecdote; optional known-source hints require explicit authorization.
- No assumption unpinning model trunk is performance-equivalent or that persisted KV can resume arbitrary different model/quant/session.
- No copying an upstream issue patch or library without license/source review and actual hardware/quality evidence.
- No claim implemented because old CIR spec mentions NVMe; each source tensor/backing/device combination must pass its own acceptance.

## 9. Upstream provenance and exact time

The publicly available DS4 docs and GitHub issues, the file-backed lookup project documentation, the offload paper and the expert-offload source were inspected on 2026-10-15. Source issues represent reproducible failure modes and improvement suggestions; whether a particular fix has been merged is not asserted from its issue alone. Author measured throughput, cache sizes, input contexts and device versions cannot be transferred to the user's Strix Halo or a Zerogravity GPU proof without independent reproduction.

This dossier supports the single unified virtual-residency design and its SSD-01..32 planned acceptance suite; **it is not an SSD-streamed Zerogravity model execution report**.

````

## Source: .agents/prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md

Original UTF-8 bytes: 12914; SHA256 `357b75cf8a85ed3b4709c232cd43d42a31b77568d9affbc48aabd4555b09a49d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# vLLM Concurrency Prior Art and Zerogravity Architecture Reconciliation

**Researched 15 October 2026.** Primary references are vLLM and vLLM-Omni's published design, scheduler configuration and API documentation; this file is source research, **not** a Zerogravity benchmark. Models/configurations in upstream products evolve and exact baseline must be pinned at implementation.

## 1. Verdict

vLLM's concurrency performance comes from its whole serving system, not just PagedAttention. The relevant baseline includes:
- unified step scheduling across prompt, cached prefix and speculative work,
- continuous batching and decode-prioritized chunked prefill,
- controlled active sequences, queued requests and queued prefill tokens,
- physical KV block allocation, refcounted prefix cache and hybrid KV cache groups,
- asynchronous worker execution and bounded device snapshots/graphs,
- speculation, expert and parallel model-aware scheduling,
- disaggregated prefill/decode for independently tuning TTFT and ITL,
- vLLM-Omni multimodal/flow/AV task stages and async serving.

Zerogravity's existing architecture already accounts for these conceptually. **It does not yet implement equivalent high-concurrency production serving.** Its potential advantage is to co-optimize *compatible* kernel, quant, model state, routed expert cache, speculative row shape, GPU/RAM/SSD tier and hardware placement under one source-verified cost model, while retaining a simple resident/no-draft baseline.

## 2. vLLM V1 unified scheduling — real source contract

Official scheduler source/API: https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/

Current V1 schedule implementation states that there is no intrinsic “prefill phase” or “decode phase” inside core scheduler. Each request has computed tokens and the total needed tokens including speculative proposals. A bounded new-token budget advances computed work, subject to cache/residency and stage constraints. This naturally expresses chunked prefill and target speculation.

**What Zerogravity should preserve:**
- same source token positions and committed StateRoot, including per-model decoder and tokenizer semantics;
- track needed/computed/reused/accepted/draft work independently;
- allow compatible requests to enter a batch at safe per-token/stage frontier without waiting for other requests to finish;
- soft decode-first QoS while protecting long prefill from starvation, with source-legal chunk boundaries and valid per-chunk state;
- account required cache pages/physical worker occupancy before finalizing a batch;
- permit media stages only with their own atomic units and rational clocks.

Do not implement a second prefill-specific GPU inference engine or per-token global router. One coherent StepPlan controls existing prepared device region execution.

## 3. Three separate overload valves: current vLLM configuration

Official config: https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/

Current documented controls include:
- max_num_queued_reqs: global API-server bounded total queued/running requests, rejects excess with HTTP 503.
- max_num_queued_tokens: total tokens in requests still prefilling (coarse upper estimate across DP workers), guards TTFT against huge backlog.
- max_num_seqs: active sequences per serving rank/engine scheduling.
- max_num_batched_tokens: actual target step work/token budget, including relevant encoder/prefill/verification planning constraints.
- optional priority/FCFS, long-prompt thresholds and policies.

This is stronger than a single queue-length cap. Zerogravity should additionally bind memory, tenant fairness, typed encoder/audio-video work and external provider jobs without duplicate allocation. A queue estimate may intentionally overestimate prefill cost when cache reuse is not yet known, which is the safe direction for QoS; document observed estimate vs actual backlog.

**Negative test:** server with 20 waiting 200K prompts and only 2 active KV sessions must reject/defer long work before OOM, not wait until physical GPU occupancy fails.

## 4. Hybrid KV and source-domain cache compatibility

vLLM public design: https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/
Source API: https://docs.vllm.ai/en/latest/api/vllm/v1/core/kv_cache_coordinator/
Prefix cache design: https://docs.vllm.ai/en/latest/design/prefix_caching/

The hybrid design supports cache classes with different logical lifetimes and prefix replay rules: full attention, sliding-window and Mamba-style recurrent state. Unlike standard full-attention prefix caching, an SWA prefix may be valid when only the trailing source window is retained, while recurrent state may require boundary snapshots. Source document revisions differ on the number/group compositions supported; don't assert fixed two-group limitation across future/current releases without pinning code.

**Zerogravity design extension:** a generic bounded source-domain frontier meet across full KV, SWA, GDN, KDA, MLA/DSA/QSA indexers, mHC, draft and source-dependent media features, with exact processor, model/quant, numeric, content and tenant access keys. Where one required domain lacks a valid frontier, reuse the earlier complete joint root or recompute/restore; don't report a fake prefix hit based only on KV bytes.

Important: generic multi-state-domain support as a mathematical contract does NOT mean generic interchangeability of unrelated model state. Each StateDomain must provide its own source proof, restoration shape, causal dependency and trusted physical conversion receipt.

## 5. Chunked prefill and real-time tail protection

vLLM guidance: https://docs.vllm.ai/en/stable/configuration/optimization/

Decode work is usually prioritized for ITL. Remaining token budget is filled with prefill work, chunking long requests. Larger max_num_batched_tokens can help TTFT/batch throughput while smaller favors shorter decode inter-token delays; optimal depends on hardware/model and load.

Zerogravity must account for **three QoS constraints together**:
1. max queue wait and TTFT for newly admitted long prompts, including aging/protected minimum prefill service;
2. inter-token latency and per-tenant fairness for already decoding text;
3. soft realtime TTFA/playback packet deadlines and long-running solver-step yields for audio/video.

A low-latency speech stream cannot be protected simply by labeling all prefills lower-priority forever; admission and legal step boundaries must be measured. A P/D split may isolate tail but carries additional state transfer and memory.

## 6. Disaggregated prefill/decode

vLLM source: https://docs.vllm.ai/en/latest/features/disagg_prefill/

Published docs describe P/D disaggregation as experimental, mainly to independently tune TTFT/ITL and improve tail ITL control; explicitly note it does **not intrinsically improve throughput**. State transfer connectors include NIXL, Mooncake, an independent KV-cache layer and others, with source/lifecycle limitations.

Zerogravity must retain an equally first-class optional plan without duplicating inference engine:
- prefill and decode owners can run on different devices/peers;
- transfer ALL required hybrid state domains, not just full KV, with exact source model/numeric/epoch/tenant receipt;
- IO, peer transfer and actual device completion require real event/lease readiness before new StateRoot;
- compare T_prefill + T_network/state transfer + T_decode vs collocated mixed scheduling under tail latency and memory headroom;
- wrong model/quant/processors, interrupted transfer and incompatible state layouts must safely refuse/recompute, not silently read stale state.

Transport topology/permissions and authentic remote events remain core physical placement contracts, not API-layer tricks.

## 7. vLLM-Omni source capabilities and fair comparison

Official overview: https://docs.vllm.ai/projects/vllm-omni/en/latest/
Serving endpoints: https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/
Diffusion batching: https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_continuous_batching/
Video streaming input: https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/video_stream_api/

vLLM-Omni has native support for many text/image/audio/video/action model families, typed stage orchestration, async streaming, realtime and duplex variants. Its published model-serving docs describe **one loaded model per server instance** in primary API deployment examples, with per-model stage orchestration. However, do not extrapolate to a claim that a deployment of multiple instances cannot route across models, or that no plugin/multiple-model pattern exists elsewhere.

Diffusion step batching supports request-batching and iterative solver-step batching where exact pipeline source permits it. Different source image/video models may not be safely batched together, and a simultaneous TTS pipeline is not a native joint audio/video generator.

Zerogravity target differentiation:
- one physical device serving several independently registered models under shared source weight memory and global admission, without one inference engine/OS process per model;
- source-bound joint optimization across model, quantization, expert routing, speculation, storage tier and ISA;
- source-correct native media/decision TaskDrivers coordinated by latency and resource planner.

**Still future, not implemented.** Run original models and tasks under comparable source input, numerical/solver, codec and output finality requirements before promising any advantage.

## 8. Metrics and honest goodput

vLLM metrics: https://docs.vllm.ai/en/latest/design/metrics/

Core events distinguish queued, first scheduled, preempted and new output tokens. User-visible timings include TTFT, queue, prefill, ITL and end-to-end latency. Zerogravity should expose these plus true physical queued/issued/completed GPU and source state commit, rejected offers, tenant fairness, cache state/bytes, draft accepted vs proposed, energy, cold/warm, physical OS+GPU residency and audio/video clocks.

**Service goodput** only counts valid committed result under declared latency objectives, but its reporting denominator and service availability must include ALL offered requests. A strategy that rejects 80% of long prompts may deliver low tail latency to the remaining users, but is not automatically a better system. Include offered input rate, admitted/completed/rejected/expired share, p50/p95/p99 time, per-tenant fairness and source fidelity.

Quantized pack quality vs original FP8/BF16, GPU candidate vs same pack, and target+MTP vs target-only must be independently tested. At most claim “better in workload X at hardware/driver/config Y under comparable quality” after meaningful repeated trials.

## 9. Specific research and software risks

- Many vLLM performance optimizations rely on validated CUDA/HIP kernel assumptions; a Strix Halo shared UMA deployment has different bandwidth, OS page-cache and quant GEMV tradeoffs.
- “More draft tokens” can increase MoE unique experts and padded target verify rows, making output slower despite high proposal acceptance (one measured prior-art example).
- A longer prefill chunk can improve throughput and hurt a live user's ITL/TTFA; P/D disagg or adaptive batching is beneficial only after observed tail/goodput results.
- A copied prefix from a mismatched recurrent KDA/GDN schema can yield plausible outputs while making neural state incorrect.
- Physical GPU slot reused after encoding but before actual completion can corrupt tensor outputs even if every hardware command reports Ready (DS4 #384); resource/cache plan must cover entire pending graph.
- Diverse model assemblies compete for resident weights, draft cache, SSD and OS headroom. “Supports N model aliases” is not proof N models fit or run concurrently with deadlines.
- A latency-optimizing policy can starve long prefills or reject too much of load. Explicit fairness/admission governance is required.
- Any runtime optimization should retain a simple source-correct single-request/no-draft/reference plan and record negative speed experiments.

## 10. Phase owner map and evidence boundary

P09: API queue and bounded model/task/tenant admission; P10: hybrid state and prefix; P12: batched compatible source kernels; P15: MTP/DFlash; P16: finite multi-dimensional candidate scheduling; P20: offline tuner; P22: pinned vLLM/V1 and Omni benchmark. v5.1 adds timed media and different QoS task dimensions; distributed plan adds source-compatible P/D state transfer.

Read normative joint scheduling contract and SERVOPT-01..36. **None of these new tests or measured comparison results exists solely by authoring the design.**

Research reviewed against official upstream docs on 2026-10-15; URLs and source versions can change and must be reverified before implementation.

````

## Source: docs/DEVELOPMENT.md

Original UTF-8 bytes: 7413; SHA256 `3427b2041dd19bc3053ef79a11d6bade5eefc493131216de0dc9810f082a63a5`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Development, builds and release definitions

Use [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution and AI-assisted plan
authorship, [AGENTS.md](../AGENTS.md) for repository boundaries, and the
[plan index](../.agents/plans/TODO.md) for current implementation and test work.
The architecture is described in the [engineering specification](ENGINEERING_SPECIFICATION.md).
This guide describes the checked-in build interfaces; it does not certify every
platform, model or release target.

## Local Rust development

The [toolchain](../rust-toolchain.toml) pins Rust 1.95.0 with rustfmt and Clippy.
The virtual [workspace](../Cargo.toml) has one Cargo.lock and one package-version
authority. Its default members are the library facade and CLI; default inference
features select CPU, Qwen3.5 and text without the GPU dependency closure.

From the repository root, with the locked crate sources already available:

```sh
cargo build --release --locked --offline -p zerogravity-cli
cargo run --locked --offline -p zerogravity-cli -- --help
cargo run --locked --offline -p zerogravity-cli -- --version
cargo fmt --all -- --check
```

Source installation uses `cargo install --locked --offline --path crates/zerogravity-cli`.
The sole product executable is `zerogravity`; argument handling and text output
belong to the [CLI package](../crates/zerogravity-cli/README.md).

After moving or copying a checkout, regenerate affected build/test outputs or
use a fresh Cargo target directory. Integration tests can embed an absolute
`CARGO_BIN_EXE_zerogravity` path from the old checkout even when the current
binary runs correctly. Such caches are disposable build inputs, not retained
test evidence; keep model and SDK directories outside any cache cleanup scope.

Run the complete repository gate with the approved cached inputs and actual
user/network isolation:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh
```

[check.sh](../scripts/check.sh) selects the CPU owners, optional cache tests and
feature-isolation checks explicitly. A workspace-wide `--all-features` check also
selects GPU diagnostics and is not the default CPU gate. Trained-model and
physical-device tests require their actual fixtures, permissions and qualified
scope; ordinary builds and source-free tests do not replace them.

GPU products are explicit builds, for example the Vulkan text frontend:

```sh
cargo build --release --locked --offline -p zerogravity-cli \
  --no-default-features --features model-qwen35,text,vulkan
```

CUDA and HIP use their explicit `cuda` or `hip` feature and the required existing
native runtime/toolchain setup. Follow the [GPU library runbook](../crates/zerogravity-gpu/README.md)
for backend-specific builds, source/physical limits and downstream Cargo patches.
The [qualification tool](../tools/gpu-qualify/README.md) is separate from the
product. There is no independent probe workspace or second lockfile. Optional
backends, native LLVM code generation and vendor SDKs have distinct requirements;
an offline Cargo flag alone cannot prevent a dependency build script from using
the network. Keep actual network isolation and applicable acquisition authority.

## Nix development and native checks

The [flake](../flake.nix) pins nixpkgs, crane and rust-overlay. Native dependency
builds, packages and checks are separate derivations. Its default package and
app select the CPU text frontend.

```sh
nix develop
nix build .#zerogravity
nix run .# -- --version
nix flake check
```

Nix can fetch declared inputs and toolchains that are not already present.
These commands are not an instruction to acquire unapproved models or SDKs.
The [source filter](../nix/source.nix) and Git flake inputs determine the included
files; local untracked changes are not automatically part of a Git flake source.
Local Cargo success does not establish a Nix, cross-target or installed-service
pass.

## CPU cross-release definitions

On an x86_64 Linux build host, [nix/release.nix](../nix/release.nix) defines six
individual CPU assets and SHA-256 sidecars:

| Asset | Rust target |
| --- | --- |
| `zerogravity-linux-x86_64` | `x86_64-unknown-linux-musl` |
| `zerogravity-linux-aarch64` | `aarch64-unknown-linux-musl` |
| `zerogravity-windows-x86_64.exe` | `x86_64-pc-windows-gnu` |
| `zerogravity-windows-aarch64.exe` | `aarch64-pc-windows-gnullvm` |
| `zerogravity-macos-x86_64` | `x86_64-apple-darwin` |
| `zerogravity-macos-aarch64` | `aarch64-apple-darwin` |

With the required platform inputs authorized and available:

```sh
nix build .#release-all
./scripts/build-release-artifacts.sh dist
```

The [helper](../scripts/build-release-artifacts.sh) materializes and checks the
assets plus VERSION and PLATFORMS.txt. Linux assets use static musl; Windows
uses MinGW/LLVM-MinGW, and macOS uses an explicitly fetched Apple SDK plus
LLVM/LLD. The macOS definitions produce unsigned binaries. GPU backends are not
bundled in this six-target set. Cross-compilation, native execution on each
target, model numerics and platform signing/notarization are separate gates;
their definitions are not a claim of completed qualification.

The [CI workflow](../.github/workflows/build-binaries.yml) runs native quality
checks on PRs and main pushes. The six-target build runs on manual dispatch or
version tags after quality checks. Publication runs only for a tag matching
`v<workspace.package.version>` and refuses to overwrite an existing release.
Publishing or creating tags still requires the applicable explicit user action.

## Release installers

[install.sh](../install.sh) and [install.ps1](../install.ps1) select an OS/CPU
asset, verify its SHA-256 and reported binary version, and replace the executable
under a user-selected directory. They install no service. They require an
explicit release repository or base URL; no remote repository is guessed.
For an authorized, existing release:

```sh
ZEROGRAVITY_REPO=OWNER/zerogravity ZEROGRAVITY_VERSION=0.1.0 \
  ZEROGRAVITY_INSTALL_DIR="$HOME/.local/bin" ./install.sh
```

```powershell
.\install.ps1 -Repository "OWNER/zerogravity" -Version "0.1.0"
```

The base URL override is `ZEROGRAVITY_RELEASE_BASE_URL` or PowerShell
`-ReleaseBaseUrl`. A checksum fetched from the same release host detects
corruption; it does not authenticate against a compromised host. Release signing
and Windows/macOS signing remain separately governed work. The offline installer
fixture is `./scripts/test-install.sh ./target/release/zerogravity` after building
the local executable.

## NixOS service definition

The flake exports `nixosModules.default` and `nixosModules.zerogravity`. After
importing the module, keep its service disabled until serving is implemented:

```nix
services.zerogravity.enable = false;
```

The [module](../nix/module.nix) defines package, absolute workingDirectory,
extraArgs, non-secret environment and optional runtime environmentFile settings.
An enabled unit invokes `zerogravity serve`, which currently refuses as
unimplemented; `Restart = "no"` prevents a restart loop. The unit uses
DynamicUser, private temporary storage, state/cache directories, journal output,
a restricted umask and no automatic firewall exposure. It is a service
definition, not a working server. Keep secrets out of the Nix store and ordinary
environment attributes; a future serving change needs its own integration tests.

````

## Source: docs/ENGINEERING_SPECIFICATION.md

Original UTF-8 bytes: 85270; SHA256 `581e597a0b007bf6af8297b4c16aca11a07b3906bd1cc999ae07bcfa029f4711`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — Integrated Engineering Specification
## Executable contracts for a modular AI runtime

**Specification edition 1 · Architecture v4.3 · 12 October 2026**  
**Classification:** authoritative *integrated design* for implementation agents, not working code or a frozen public ABI.  
**Audience:** Rust engineers, model/kernel/compiler engineers, security reviewers, release maintainers and autonomous coding agents.  
**Companion:** [Whitepaper](WHITEPAPER.md), [Implementation Playbook](../.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md), [Requirements Traceability](../.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md), [Reference Bibliography](REFERENCES.md).

> **Build rule:** Treat the mathematical contracts, state/resource ownership and rejection/fallback behavior below as requirements. Rust snippets and command syntax are **pseudocode to clarify interfaces**, not preapproved modules, crate organization or backward-compatible API commitments. Actual tests/physical measurements are necessary before status changes to FUNCTIONAL, STATE_SAFE or PERFORMANCE_QUALIFIED.

## Contents

- [ES-00. Scope, authority and document discipline](#es-00-scope-authority-and-document-discipline)
- [ES-01. Minimal dependency boundaries and identity model](#es-01-minimal-dependency-boundaries-and-identity-model)
- [ES-02. Trusted artifacts, registered mathematics and source quality](#es-02-trusted-artifacts-registered-mathematics-and-source-quality)
- [ES-03. Trusted TaskDrivers, typed outputs and application boundary](#es-03-trusted-taskdrivers-typed-outputs-and-application-boundary)
- [ES-04. Compiler, closed regions and qualified providers](#es-04-compiler-closed-regions-and-qualified-providers)
- [ES-05. Physical resources, worker protocol and device execution](#es-05-physical-resources-worker-protocol-and-device-execution)
- [ES-06. Exact logical state transactions and output publication](#es-06-exact-logical-state-transactions-and-output-publication)
- [ES-07. Model catalog, resident instances, router and live service](#es-07-model-catalog-resident-instances-router-and-live-service)
- [ES-08. Distributed placement and authenticated peer execution](#es-08-distributed-placement-and-authenticated-peer-execution)
- [ES-09. User-facing CLI, SDK, HTTP and configuration](#es-09-user-facing-cli-sdk-http-and-configuration)
- [ES-10. Security, licensing, error taxonomy and observability](#es-10-security-licensing-error-taxonomy-and-observability)
- [ES-11. Required invariants and reviewable implementation tests](#es-11-required-invariants-and-reviewable-implementation-tests)
- [ES-12. End-to-end execution walkthroughs](#es-12-end-to-end-execution-walkthroughs)
- [ES-13. Implementation and proof sequencing](#es-13-implementation-and-proof-sequencing)
- [ES-14. Acceptance and architectural sustainability](#es-14-acceptance-and-architectural-sustainability)
- [ES-15. References and formal source of truth](#es-15-references-and-formal-source-of-truth)

---

## ES-00. Scope, authority and document discipline

### ES-00.1 Purpose

This document connects the full source-of-truth architecture into an **implementable system specification**. It answers: what entities exist, who owns state, what data crosses boundaries, which operations are legal, how failures are handled, how models/devices/tasks/remote peers extend the system, and what tests establish completion.

The project root is `/home/pvl/spaces/abird/src/zerogravity/`. Its `.agents/docs` directory holds depth specifications, `.agents/plans` holds user decisions/roadmap/gates/source status, and `docs` holds the public publication/implementation dossier. The external `/home/pvl/spaces/abird/research/inference` corpus is **read-only** unless specifically authorized.

**Precedence when documents disagree:**

1. Latest explicit user requirements in [Decision Register](../.agents/plans/DECISION_REGISTER.md) and live verified code/tests (once source exists); a new user decision may revise design.
2. **This integration specification**, [Modular Reference Architecture](../.agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [Composition Contracts](../.agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md) and [current Master Plan](../.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md).
3. Detailed current normative subsystem specs for precise operators, numeric and hardware behavior; newer explicit v4.x corrections supersede earlier appended examples.
4. [Current implementation roadmap](../.agents/plans/IMPLEMENTATION_ROADMAP.md), [Stage0 gates](../.agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md), [experiments](../.agents/plans/EXPERIMENT_PROGRAM.md), [release criteria](../.agents/plans/RELEASE_ACCEPTANCE_GATES.md). They define work and **prove nothing until run**.
5. Retained design reviews and source-specific research: **attributed non-normative context, not implementation authority**. Superseded snapshot bodies remain in Git history.

In an apparent contradiction, **record it and choose the smallest contract-consistent implementation**, rather than silently inventing source semantics or conflating physical device execution with logical session commit. Research source names are not core Rust API names.

### ES-00.2 Release status and support ladder

As of the initial manuscript, there was no source runtime. **Repository update (12 October 2026):** a dependency-free **Rust packaging/bootstrap CLI** with `--help`, `--version` and explicit unimplemented model/server commands plus Cargo/Nix/release scaffolding now exists; **there is still no inference-engine implementation, model execution, GPU/NPU/TPU program, serving API, published CI artifact, benchmark or Git repository**. P00 is **PARTIAL** (packaging/tooling only); P01–P23 are NOT_STARTED. Future capability reporting is **per exact combination** of model checkpoint/quant, TaskKind, operator/shape/context, hardware/driver and numeric mode:

```text
PLANNED
  -> REFERENCE (source/reference mathematics implemented and tested)
  -> COMPILES (on a real target, if relevant)
  -> FUNCTIONAL (full task runs)
  -> STATE_SAFE (transactions/cancel/failure qualified)
  -> QUALITY_VALIDATED
  -> PERFORMANCE_QUALIFIED (matched real hardware/tasks)
  -> PRODUCTION_QUALIFIED (security/availability/release)
or BLOCKED / UNSUPPORTED / UNSAFE_DISABLED for concrete reasons.
```

A simulated graph-only provider or fake multi-node topology may pass architecture-interface tests but cannot be labeled a physical GPU/TPU/network implementation. Do not report an unmeasured source-author speedup as a zerogravity achievement.

### ES-00.3 In and out of the mandatory core

Mandatory first executable: **one library plus one CLI binary**, trusted parser/ModelDefinition, small SemanticProgram, reference math and effect/state protocols, one trusted TaskDriver, one RegionExecutionProvider form, compiled device execution and enough ResourceArbiter for safe admission, one model-to-user-output vertical slice. One CPU path must build without GPU compiler/model download.

**Optional after measured need:** additional vendor graph/wafer program providers; network peer transport and placement; per-model replica scaler; long-retention response store; high-availability quorum; public dynamic plugin ABI; entire task workflow language; full autograd/optimizer training stack. Existing upstream inference servers are *not* mandatory dependencies.

### ES-00.4 Public versus internal format stability

Only explicitly published service/SDK/serialization versions can be backward-compatible public contracts. SIR and ExecutableBundle are enduring **architectural concepts** but may begin as internal Rust types; no promise of externally serializable SIR before implementation. Region/Kernel/Command transforms may be transient compiler views and need not become separate dialect files. State snapshots and ModelArtifact hashes require versioning as soon as persisted. Compiled GPU/TPU driver handles/pointers are never persistent user artifacts.

---

## ES-01. Minimal dependency boundaries and identity model

### ES-01.1 Logical composition and control flow (not literal crate dependencies)

```text
                     CLI / SDK / HTTP adapter
                               |
                      validated TaskRequest
                               |
       ModelCatalog -> ModelRouter / exact binding / admission
                               |
    ModelDefinition + TaskDriver -> SemanticProgram / registered effects
                               |
                    compiler + region providers
                               |
               ExecutableBundle + physical placement
                               |
              TaskCoordinator / per-session StateRoot
                               |
        Device/IO/Remote RegionWorkers + local device pools
                               |
         prepared candidate states + real event readiness
                               |
              TaskCoordinator publishes one StateRoot
                               |
                  committed output -> adapter

  ResourceArbiter: coarse physical reservations at admission and
                   step boundaries; leases enforced by device pools.
```

This is a **logical request/control and ownership diagram**, not a Rust dependency graph: the ModelCatalog and model semantics are constructed/validated before route admission where necessary, and compiler variants may be prepared and cached at model load. The diagram's purpose is to distinguish **request-level routing**, compiled device dispatch and state publication; it does not imply a per-token traversal or compilation step. Avoid circular imports by putting shared IDs, effects and immutable metadata in small foundational modules. Compiler code must not depend on HTTP models, and model adapters must not choose a GPU by string name. ResourceArbiter may be used by runtime and router, but one physical budget owns reservations. The worker can receive compiled handles + lease IDs, not clone a full task history.

The executor can be statically linked first; optional backend feature flags avoid linking unavailable vendor runtimes into CPU-only builds.

### ES-01.2 Canonical identities — what changes when?

| Identity | Meaning | Change triggers |
|---|---|---|
| **ArtifactKey** | exact source tensor/config/shard/tokenizer/template/sidecar provenance and stored quant | new checkpoint, changed tensor bytes or adapter version |
| **SemanticKey** | exact registered op formula, versions, position/state effects and required numerical semantics | new operator formula/schema |
| **TaskKey** | TaskDriver stage/output/stream and task-specific state contract | new task program or stream schema |
| **ModelBinding** | immutable source + quant + adapter/profile revision and TaskKind/numeric policy | resolved at admission |
| **AliasRevision** | human name/profile mapping generation | user/admin alias update; **new routes only** |
| **ResidentModelId / DeploymentId** | physical loaded program/weight placement, perhaps sharded across peers | replica/load/placement lifecycle |
| **ReplicaGroupId** | group of equivalent source/model semantics, *not* arbitrary models | replica creation/removal |
| **SessionEpoch + StateRootId** | authoritative TaskSession writer generation and all-domain committed state | safe root publication |
| **StateShardGeneration** | remote/local physical per-domain version and owner lease | shard migration/reallocation |
| **PlacementEpoch** | frozen compute/parameter/state/cache topology used by current plan | committed migration/replan |
| **BundleCacheKey** | semantic region, codegen/compiler/driver, shape class/numeric and physical layout | code upgrade/shape/ISA change |
| **RuntimeBinding** | current row/parameter slots, SessionEpoch, live leases and events | **each step**; never a compiler cache key |

The same trained source can have multiple ModelProfiles and separate ResidentModel deployments, but cannot silently become a new model source when a request is already running. One distributed deployment spanning four GPUs is one **replica**. Two fully loaded copies of the same exact model are two replicas. A target+draft task binds **different logical model roles**, not replica copies.

### ES-01.3 Configuration provenance

Each setting has an explicit scope, type, valid range, mutability boundary and constraints. Priority of allowed user-facing inputs is documented from safe qualified defaults through system/user/profile/alias/CLI/request controls. **Security floors, exact immutable checkpoint/model math and physically safe admission are hard constraints regardless of precedence.** A user might change sampler temperature for a request, but an in-flight model quant, state layout or destination cluster peer is not changed by an unvalidated request field.

A `config show --effective`/inspect path must return value + source + disallowed reason without leaking passwords, API keys, peer addresses or private user data.

---

## ES-02. Trusted artifacts, registered mathematics and source quality

### ES-02.1 Trusted model source pipeline

```text
explicit local file or approved registry reference
  -> bounded ArtifactReader + content integrity/license
  -> config and tensor/quant/shard + tokenizer/template validation
  -> validated ModelDefinition and exact SemanticOpRegistry
  -> immutable SemanticProgram + reference vectors
  -> legal physical compile candidates
  -> admitted ResidentModel with stable ModelBinding
```

Unknown GGUF dialect, safetensors metadata, tensor name/shapes, quant block, trained routing formula, tokenizer pretype or model template must fail with `UnknownModelOp/UnsupportedQuant/BadArtifact` and exact diagnostic. Recognizing an archive and reading tensors is **not** proof it can generate correct tokens. Model registry downloads must be explicit/approved, atomic/resumable/size bounded with hashes and rights; no arbitrary executable model repository code.

Source formats, quant and model families are specified in [Formats and Quantization](../.agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md), [Model Semantics](../.agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md) and [Product Artifact Pipeline](../.agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md).

### ES-02.2 Semantic operator contract

A SemanticOp schema must specify:

- typed logical inputs/outputs and bound or symbolic shape constraints, ragged segments and strides;
- exact mathematical formula and trained parameter roles, position/attention mask and model-specific selected-indexer/router decisions;
- source stored dtype/quant, compute numeric policy, output dtype and explicitly tolerable nonassociative reductions;
- read/write *effects* on named StateDomains plus causal order and permitted region cuts;
- independent portable/reference implementation and source-model test vectors;
- whether a partition algebra exists (e.g., masked softmax shard reduction) and its legality tests;
- allowed fallback and state-layout conversion/replay requirements.

High-level semantics represent complex attention/recurrent/MoE/stateful operations; compiler may lower their **pure internals** to matmul/gather/pointwise/reduce without hiding effects. ModelVersion should not be switched through a server-level `if model_name == ...` branch. An unfamiliar model built from existing exact operators adds ModelDefinition/recipe and checkpoint tests; a new operator family first adds source mathematical reference.

### ES-02.3 Numeric correctness levels

These contracts must be tested independently:

1. **Semantic correctness:** trained math, attention mask, position, expert top-k, recurrence and source artifacts.
2. **Numerical tolerance:** actual data representation, accumulation/reassociation, block scaling and layer/logit agreement.
3. **Model/task quality:** checkpoint and quant against trusted independent prompts, long context, tool, audio, image/video or encoder tasks.
4. **Stochastic target distribution:** sampler/RNG/verifier state and accepted branch; exactness only for a mathematically valid target-authoritative algorithm.

Stored checkpoint quant, lossless on-device packing, temporary compute encoding and derived requantized source are **four different provenance states**. Never call a model changing its trained router selection merely a faster equivalent kernel.

### ES-02.4 Versioned mutable model parameters (future)

Default inference uses immutable ParameterVersion. Future GradientProgram/TrainStep may produce provisional gradient/optimizer state and publish a new ParameterVersion. Existing inference sessions remain pinned to previous source/adapter/compiled plan until explicit state-compatible cutover; **no default autograd engine or distributed training WAL** is required in the first release.

---

## ES-03. Trusted TaskDrivers, typed outputs and application boundary

### ES-03.1 TaskDriver scope

A TaskDriver controls trusted coarse stage transitions, not tensor arithmetic or an unrestricted agent workflow. It validates TaskRequest, binds one or more RegisteredModel roles, defines per-stage/loop required effects, output `StreamContract`, `TaskFrontier`, deadlines and cancellation, and advances only at semantic stage boundaries.

Illustrative contracts:
```rust
trait TaskDriver {
    fn validate(&self, req: &TaskRequest) -> Result<TaskSpec>;
    fn start(&self, req: TaskRequest, root: &StateRoot) -> Result<TaskCursor>;
    fn next(&self, cursor: &TaskCursor, completed: Option<&StageCompletion>)
        -> Result<StageIntent>;
    fn prepare_commit(&self, cursor: &TaskCursor, result: &ProvisionalResult)
        -> Result<TaskCommitCandidate>;
}
enum StageIntent {
    Run(CompiledStageId, StageInputs),
    ParallelKnown(Vec<CompiledStageId>, JoinContract),
    WaitForInput(InputWaitId),
    Done(FinishReason),
}
```

A device-native StepProgram may internally execute repeated stages/iterations where all needed semantic effects, sampling and shape guards are supported. **TaskDriver does not need to be called for every GEMM.** Control may still pass through the host when tools/grammar or unsupported GPU steps require it.

### ES-03.2 Task families and frontiers

| Task | Principal stages | Committed unit | Stateful effects |
|---|---|---|---|
| Text generation | render→prefill→decode/target verify→stream | verified token or structured output item | KV, recurrence, indexer, sampler/RNG, grammar |
| Embedding and rerank | tokenize/encode→pool/score | score/vector record | optional encoder cache |
| ASR | window/audio preproc→encoder→decode/align | verified transcript span/time range | acoustic and optional token state |
| TTS | text condition→acoustic model→vocoder | completed PCM/encoded audio chunk | voice/speaker/RNG/acoustic |
| Image generation/edit | condition→iterate denoiser/flow→decode | final image/compatible verified result | timestep, latent and RNG |
| Video/audio | temporal stages, codec, sync and mux | complete ordered frame/window/chunk | temporal/modality alignment |
| Composite compute | typed trusted stage graph across models | task-defined result | individually bound model roles |
| Future training | forward→backward→optimizer step | published new ParameterVersion | gradients, optimizer and model generation |

Not every type of output is a token; limits/fairness must use typed work units rather than pretend media frames are language tokens. **The presence of a TaskKind in design is not proof of a runnable model/codec.** Only qualify real source task instances with reference outputs and resource checks.

### ES-03.3 External actions and agent orchestration

Prompt-originated tool descriptions may be passed to a language model and generated function calls may be returned as structured data. A TaskDriver and SemanticOp **never execute filesystem/process/network tools merely because the model requested them**. RAG, repositories, web, external user accounts and multi-agent workflow side effects belong above TaskRequest in an **independent permissioned application layer**, with explicit authorization/tenant isolation. This keeps core model execution general yet minimal and avoids embedding a second workflow runtime.

### ES-03.4 Output validity

TaskDriver declares whether an intermediate step is publishable. A generated image's latent after step 4/50 is not necessarily a valid image; a completed transcript span or TTS PCM chunk may be emitted while later chunks compute. Only output corresponding to a successfully published logical StateRoot is emitted. Stop sequences, UTF-8 framing, structured tool messages and media chunk boundaries require codec-specific terminal logic.

---

## ES-04. Compiler, closed regions and qualified providers

### ES-04.1 One semantic program, one compiled plan

**Mandatory canonical source program:** typed SemanticProgram (SIR) and effect schema.

**Mandatory prepared execution object:** an immutable ExecutableBundle containing verified region implementations plus typed command, event, resource and fallback plans.

**Transient internal views:** region selection/rewrite (RIR), per-target kernel/program candidate (KIR), and a CIR builder whose resulting physical dependency DAG is held inside the bundle. Four permanently serialized IR formats, four interpreters and one trait factory per IR are **not** required.

Compiler passes may fuse pure math, create shape-band kernels, partition independent projection rows, group MoE routes, choose quant packing and place a whole region on AOT/dataflow hardware. They cannot silently drop state effects or change trained model selection.

### ES-04.2 ClosedSemanticRegion and provider choice

```rust
struct RegionContract {
    semantic_key: SemanticRegionKey,
    inputs: Vec<ValueType>,
    outputs: Vec<ValueType>,
    effects: StateEffectContract,
    numeric: NumericContract,
    runtime_shape_guard: ShapeConstraintSet,
    fallback: Option<ReferenceRegionId>,
}
enum ExecutableKind {
    NativeKernel,
    CapturedCommandGraph,
    AheadOfTimeDeviceGraph,
    StaticallyScheduledDataflow,
    ManagedDeviceSession, // only with proven state bridge
}
trait RegionExecutionProvider {
    fn capabilities(&self) -> ProviderCapabilities;
    fn candidates(&self, region: &RegionContract, topology: &TopologySnapshot)
        -> Vec<RegionCandidate>;
    fn prepare(&self, candidate: &RegionCandidate,
               budget: &CompileBudget) -> Result<PreparedExecutable>;
}
```

A provider chooses **minimum efficient closed-region granularity**: one operation, fused layer, full encoder/denoiser stage, whole model or multi-device programmed fabric. It reports supported op/quant/shape/control/snapshot/failure, target ISA/compiler runtime/firmware, topology, compile/load/warmup/program residency and numeric evidence.

The native-kernel form can generate a Rust-owned architecture-specific device function. An AOT graph form may call an approved vendor compiler consuming a validated *temporary interchange graph*; it does not turn the engine into a foreign inference server. A static dataflow compiler may own device-internal micro-scheduling and SRAM placement. A managed opaque model text API lacking credible operator and StateBridge access is **not** a RegionExecutionProvider—it is an external application TaskServiceAdapter with separately declared contract.

### ES-04.3 ShapeEnv and variant caching

Every executable candidate guards static rank, dynamic min/max/divisibility, strides/alignment, ragged route offsets, source stored quant and compute encoding, device program version, state schema, task regime (prefill/decode/verify/encoder/denoiser), graph capture eligibility and resource footprint.

Before launch:
1. bind concrete ShapeEnv/request values to a candidate;
2. verify numerical, state, topology/driver and source/quant compatibility;
3. reserve required buffers/graph slots;
4. choose a compiled eligible variant;
5. otherwise use verified generic implementation, convert/replay state at a legal boundary or return typed error.

Do not compile a fresh model graph for every context length, per-token expert ID or media frame. Expert IDs/token lengths are runtime data where the target provider supports them. Bound source compilation cost and plan cache, and never serialize raw pointers or driver handles into a persistent state snapshot.

### ES-04.4 Compiled device StepProgram

A StepProgram can be a prepared graph/region/loop with persistent parameter and index buffers, valid iteration bound, device-side routing/sampling/verification when supported, and a readiness/completion record. Host TaskCoordinator may prepare independent work while device executes, but **cannot advance authoritative state or silently assume token dependency** before the actual candidate is ready.

Potentially faster modes are **conditional**: if a grammar requires host parsing at each token, the correct implementation uses a host-controlled mode and reports it. Profiling must measure CPU↔GPU sync and queue cost alongside kernel throughput, not merely assume one graph equals zero overhead.

### ES-04.5 Resource/command dependencies (CIR)

```text
Acquire/Reserve (typed lease)
  -> Copy/Import/Prefetch/Pack
  -> SubmitNativeKernel / SubmitGraph / SubmitProgram
  -> WaitAtFirstUse (host/device/peer ready events)
  -> Join/Collective/PublishPhysicalResidency
  -> ReadyForPublication (candidate state and outputs)
  -> asynchronous Retire/Release only after last use
```

**CIR has no authority to publish semantic TaskSession state.** A device event marked Ready is necessary but not sufficient for all-domain PublishRoot. Transfers across GPU/NPU/storage/remote peers include resource lease generations and actual coherence/ordering, not just host completion of an enqueue call.

### ES-04.6 Fallback and implementation invariants

A fast whole-layer/model kernel can replace generic math when it produces compatible state and output at the declared boundary. If a physical layout differs, fall back at a checked conversion or committed replay, not unsafe pointer reinterpretation. A missing optimized provider never forces a second model server. No region may be enabled by default solely because its vendor claims speed or a microbenchmark is faster; measured full task quality, memory and SLO are required.

Relevant detailed specs: [Semantic Compiler](../.agents/docs/SEMANTIC_IR_AND_COMPILER.md), [Region Execution Families](../.agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md), [Device Kernels](../.agents/docs/DEVICE_KERNEL_AND_CODEGEN.md), [Async CIR](../.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md) and [Task/Compiler Refinements](../.agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md).

---

### ES-04.7 Proposed multimodal extension: typed media clocks, component assemblies and non-token task semantics (v5 review)

**Design only, not current implementation or a new enduring IR.** The existing TaskDriver/TypedValue/StateDomain abstractions support other task regimes conceptually, but source-faithful support for contemporary speech/image/video/decision/low-bit models additionally requires:

1. **ValueDescriptor/ClockDomain:** typed audio samples/mels/RVQ codebooks; image/video latents, encoded media and structured decision scores; rational clock units and stable sample/frame timelines.
2. **ModelAssembly/TypedPortEdge:** independently loadable verified conditioner, encoder, predictor, vocoder/VAE, decision head and media codec components, preserving one source model assembly identity, with explicit state effects and cross-component resource placement.
3. **IterativePolicy/TrackFrontier:** true task-specific token decoding, CTC/RNN-T/TDT ASR, nested acoustic codebook AR, source-timestep flow/denoise solver, temporal video reconstruction, non-token typed scoring and full-duplex input/output lanes. Correct sampler/RNG/expert/sparse-mask semantics are not optional implementation choices.
4. **AudioVideoSync/CommitGroup:** a StateRoot may contain more than one causal clock. The coordinator alone publishes complete, playable output groups after all model, codec, device and IO readiness events; no partially valid AV frames, revisable ASR hypothesis or incomplete acoustic-codebook payload may be mislabeled as final.
5. **TensorEncoding/ActivationTransform:** source quant codec and required activation transform (including PTQ1_0/PQ2_0 packed ternary plus block Hadamard) are orthogonal to semantic model topology and must be checked before compiled region specialization.
6. **ScoreFiniteSchema/ExternalTaskServiceAdapter:** local typed-decision model logits are not autoregressive JSON, and several proprietary hosted generative services are authorized external task services, not pretend native RegionExecutionProviders.

All additions must be statically testable against fake source-model/codec/clock fixtures before introducing real weights, GPUs or cloud credentials; retain source and numeric qualification across all mixed-model components. The architecture should still have two enduring representations (SemanticProgram and ExecutableBundle), one coordinator writer per task session, one physical resource authority and one user-facing model router. Supporting contracts and model-by-model evidence: [unified multimodal v5 proposal](../.agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md), [source model families](../.agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md). No image/audio/video/typed-decision/ternary model checkpoint is claimed operational in Zerogravity at this stage.

---

## ES-05. Physical resources, worker protocol and device execution

### ES-05.1 Two levels of resource ownership

**Global PhysicalResourceArbiter:** physical admission/reservations, per-model/session/task/tenant quotas, OS safety margin, caches with reclaim policies, multi-device memory and program residency. It is the **one logical resource authority**; it does not acquire a global lock for every kernel temporary buffer.

**DeviceBackend local pool:** fast slab/page allocations and preallocated slots under a reservation, with `BufferLease{allocation_id,generation,tier,access,last_use_events}`. It may manage GPU local VRAM, mapped/coherent UMA, pinned host pages, on-chip SRAM or approved remote device allocations. Leases are retired **after actual device and IO quiescence**, never merely when a request handler times out.

```text
TopologyProbe + physical OS/cgroup/device headroom
       -> Global Reserve(model/task/session/step ceiling)
       -> Device-local subpool / IO staging allocation
       -> generation-tagged BufferLease(s)
       -> compiled command uses, event dependencies
       -> Ready / Failed / potentially live DMA quarantine
       -> retire and release budget only when physically safe
```

All allocations of model weights, experts, token and modality state, snapshot COW, compile graph images, trained draft/adapters, SSD lookup, copy and scratch must be accounted for. Virtual GPU addressability/GTT is **not** a measure of free DRAM. Third-party OS/driver allocations can change pressure after admission; fail closed and replan/evict safely, rather than claiming an absolute no-OOM guarantee.

### ES-05.2 Device capability and safety record

Each DeviceBackend supplies a validated `DeviceCapabilities` including compute architecture/instruction formats, execution kind, physical memory/coherence/addressability, async stream/event/graph features, local pool capabilities, driver/firmware/compiler versions, topology/links, device safety qualification and observed performance bands. A device's existence does not imply it is safe for arbitrary GPU+NPU concurrent power use or that low-bit native arithmetic matches stored source quant.

Potential hardware-wide hangs mean the affected configuration is **UNSAFE_DISABLED** until isolated physical qualification; no automatic privileged fabric-clock, BIOS, IOMMU or kernel-driver alterations.

### ES-05.3 WorkEnvelope and completion protocol

Illustrative internal messages:

```rust
struct WorkEnvelope {
    session: SessionId,
    attempt_epoch: SessionEpoch,
    expected_root: StateRootId,
    bundle: ExecutableBundleId,
    shape_band: ShapeBandId,
    bindings: RowAndParameterSlots,
    resource_reservation: ReservationTicketId,
    input_events: Vec<EventId>,
    output_budget: WorkOutputBudget,
}
enum WorkerReply {
    Submitted { ticket: TicketId, event: EventId },
    Complete { ticket: TicketId, ready_events: Vec<EventId>,
               provisional: PreparedCandidateId, stats: DeviceStats },
    Failed { ticket: TicketId, reason: WorkerError,
             may_have_live_dma: bool },
}
```

Messages should be compact IDs and stable row/buffer tables, not a copy of model graphs or entire token histories. Workers own physical context/stream/compiled handles; TaskCoordinator owns session logical state. A duplicate or late `WorkerReply` with stale epoch is ignored for state publication, but its physical allocations stay leased until quiescent. Queue ingress, output buffers and remote transfer credits must be bounded.

**Per-device concurrency:** compatible TaskSessions may share resident weights and device-native batches, while per-model/TaskDriver StateRoot is independently versioned. Unrelated models cannot be combined into one mathematical batch just because tensors have the same shape. A long request must not monopolize the CPU task scheduler, GPU resource pool or HTTP output queue.

---

## ES-06. Exact logical state transactions and output publication

### ES-06.1 StateRoot semantics

One `TaskCoordinator` is the **single logical writer for each TaskSession or explicitly bounded shard of sessions**; it is not one mandatory process-global OS thread or physical allocator. It owns `SessionEpoch, StateRootId, TaskFrontier, output_seq` and cancellation/attempt epoch.

A `StateRoot` is an immutable manifest of all required domain-version handles: KV and recurrent matrices, sparse indexer/ngram/history, sampler/RNG, grammar and tool/message state, multimodal position/latent/task state, and (future) explicit optimizer parameter versions. A domain may be absent if the model/task genuinely doesn't use it; a prefix lookup must check *all required domains* for compatibility.

### ES-06.2 Four-step all-domain transaction

```text
Begin(old_root, attempt_epoch)
   1. pin immutable source root and physical reader leases
   2. execute into private/COW or validated reversible provisional state
Prepare(provisional, intended_frontier)
   3. validate every domain schema/position/numeric and same frontier
   4. require real device/IO/remote Ready cohort and lease generations
PublishRoot(expected_old_root, prepared)
   5. only TaskCoordinator verifies epoch and publishes one complete root
   6. generate committed output events (session, epoch, sequence)
Retire/Abort
   7. release stale candidate or old buffers *after* true last-use fences
```

If any `prepare` step fails, the original root remains visible and **no speculative/rejected output may be streamed**. One domain cannot call a public mutating `commit()` independently. CIR device commands may indicate ReadyForPublication but **never** publish the TaskSession root. An O(1) host metadata root swap does not mean O(1) GPU recurrent state copying: provisional implementations must have a real efficient COW/replay/copy contract and corresponding cost.

### ES-06.3 Speculative and asynchronous hazards

Speculative target verification may accept zero, one or several proposed tokens. Only those accepted and any target-corrected output contribute to committed sampler/grammar, recurrent, KV, indexer and token-history state. Device-native multi-token progress is legal only where the sampler/RNG/grammar and stop/position semantics have been validated; a general host-loop fallback remains correct.

Cancellation before Commit abandons provisional changes. Cancellation after successful PublishRoot retains previously committed results and stops future work. A late GPU event from old epoch cannot mutate a newer root. If a driver/remote DMA remains active after timeout, the resource must be quarantined or the relevant device reset safely rather than freed/reused.

### ES-06.4 Prefix sharing and state compatibility

A `StateCompatKey` must bind exact checkpoint, stored quant/semantic op schema, adapter generation, tokenizer/template and canonical token/position/media prefix, task/grammar/RNG semantics, snapshot layout/codec and tenant/security domain. A cache hit provides the **last common resumable frontier**, not just the longest matching text. Partial compressed/recurrent state groups may force a shorter prefix and replay suffix.

Immutable snapshots may be shared via COW or content-addressed references where authorized; mutable remote live StateShardRef is **not equivalent to an immutable prefix-cache hint**.

### ES-06.5 Output transport is a different guarantee

Output becomes eligible to emit only after PublishRoot. The service emits ordered `OutputEvent(session_id,epoch,seq,kind,payload)` and applies typed stage-specific completion/partial chunk policy.

On an HTTP/tunnel disconnect, the client may have received a chunk whose ACK was lost. The default defensible guarantee is **ordered, at-most-once live delivery attempt**, not exactly-once remote receipt. Optional authenticated resumable cursors require a bounded/durable output-log design and client deduplication. Remote rank prepare/commit is similarly not an automatic durable distributed transaction.

### ES-06.6 State transaction test oracles

Required unit/fault oracles:
- three independently prepared state domains where second preparation fails → root unchanged;
- accepted speculative prefix length 0..N and RNG/grammar equal to exact target oracle;
- cancellation before/after PublishRoot; no leaked uncommitted user output;
- repeated snapshot/fork/COW and full-context prefix partial reuse;
- device/remote event arrives after stale epoch; not reused state or premature buffer free;
- lost remote Ready ACK, wrong quant/adapter/schema, coordinator partition; fail-stop or safe replay rather than split-brain;
- client disconnect before and after event delivery; ordered output sequence and honest uncertainty.

Detailed current contracts: [State/worker protocol](../.agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), [Hybrid state cache](../.agents/docs/STATE_CACHE_AND_TRANSACTIONS.md), [Invariants](../.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md) and [Distributed protocol](../.agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md).

---

## ES-07. Model catalog, resident instances, router and live service

### ES-07.1 Four model-serving entities

1. `RegisteredModel`: exact trusted immutable source checkpoint/config/quant, ModelDefinition, tokenizer/template/sidecars, supported task and semantic capability. It may be listed while unloaded.
2. `ModelAlias/ModelProfile`: human-facing versioned name and typed defaults for context, sampler, task, hardware/placement and resident policy. Alias revision changes affect **future request resolutions only**.
3. `ResidentModel/Replica/Deployment`: physically admitted/warmed weight/program instance, perhaps itself sharded across several devices/peers. Several equivalent source instances form a replica group.
4. `TaskSession`: mutable logical StateRoot and TaskDriver per user/tenant/stream, bound to exact model and task identity for its lifetime.

Many registered models do **not** imply infinite simultaneously loaded resident instances. Memory fit is constrained by actual weight and context state, compiler programs, device scratch and OS margin. A distributed single model across four machines is **one deployment**; two copies of an equivalent model are two replicas; an embedding+rerank task or target+draft uses two distinct model roles, not alternative replicas.

### ES-07.2 Exact source route, then compatible replica

The ModelRouter operates on **request admission**, not on every token. It first resolves the exact model alias + revision and validates TaskKind, source quant, adapter, numeric policy, tenant rights and requested output. It then chooses an eligible *semantically equivalent* warm/new replica using queue, projected P95, cache affinity, existing StateRoot location, physical capacity, safety and distributed permissions.

**No silent different-target route** for an exact model request. If a configured `auto` policy explicitly allows selection among different models, actual target source/quality must be reported and session state cannot be naively shared across models. A new source quant is a different artifact even if its public name looks similar. Active stateful sessions remain sticky or migrate only with valid complete state snapshot/fencing at a committed frontier.

### ES-07.3 Model lifecycle and resource accountability

```text
REGISTERED -> VERIFYING -> ADMITTING -> PREPARING -> LOADING
        -> WARMING -> READY -> BUSY / IDLE -> DRAINING -> UNLOADING
                                   |                       |
                              retained lease             UNLOADED
          any stage -> FAILED / UNSUPPORTED / BLOCKED_BY_POLICY
```

When 100 requests arrive for one unloaded model, deduplicate model preparation into **one** verified resource reservation/load operation and a bounded set of waiting futures. Continue serving unrelated already warm models while cold model compile/load/IO proceeds.

Keepalive/TTL, preload/pin and scale-to-zero are configurable. Under pressure, unload safely idle/unused model and evict optional caches before refusing; **never evict live unsnapshotable state, active GPU graph or in-flight DMA**. A model not fitting exactly can be refused, queued, or placed on authorized peers—not silently converted to lower quality or replaced by another model.

### ES-07.4 Router and TaskCoordinator do not duplicate scheduling

Router selects model+replica+admission at coarse boundaries; TaskCoordinator schedules model/task rows, batching, speculation and state commits. Compatible model/adapter/TaskKind/numeric sessions may be physically batched, while unrelated models can run concurrently only when physical resources permit. Per-model queue budgets and fairness prevent giant cold loads from starving small interactive requests. Router cannot take a global per-token lock or re-encode each step through a second HTTP hop.

### ES-07.5 Hot alias reload and session consistency

Updating a profile or alias creates an immutable new revision; an admitted session holds the old `ModelBinding` and source/template/StateCompatKey through its stream unless deliberately migrated with exact semantics. An alias removed by an administrator must not free an active session's source or graph leases prematurely. Deleting blobs from a model store must respect aliases, active model deployments, cached derived packs and snapshot references.

Detailed [Model Router and Residency](../.agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md).

---

## ES-08. Distributed placement and authenticated peer execution

### ES-08.1 Four orthogonal placements in one physical plan

A `PlacementPlan` binds:
- `ComputePlacement`: semantic region/layer/attention/FFN/expert/task stage and qualified device program owner;
- `ParameterPlacement`: immutable source/packed weights, adapters, sparse routed experts and external large lookup rows;
- `StatePlacement`: authoritative versioned per-session mutable KV/recurrent/indexer/sampler/media state shards and owner leases;
- `CachePlacement`: immutable snapshots/prefix or encoder feature copies (advisory directory) and derived artifacts;
- derived `TransferPlan`: activation/state/weight/copy/partial reductions/collectives with physical transport and event dependencies.

These maps are a view within ExecutableBundle/CIR bound to a `TopologyEpoch` and `PlacementEpoch`, **not another enduring semantic IR**. ModelDefinition and TaskDriver do not change merely because the same closed region executes on a peer rather than locally. A distributed RegionExecutionProvider may own a whole rank/fabric internal program, but it cannot publish TaskSession state.

### ES-08.2 Supported classes of placement

| Mode | Compute and state location | Transfer behavior | Gate |
|---|---|---|---|
| Local colocated | full model/stage + live cache on one host | none | default reference performance |
| Layer pipeline | layer group and its own KV/SSM colocated on each peer | activation boundary per step/chunk | correct same-model logits/state, serial hop cost |
| Attention/FFN split | stateful attention beside KV; FFN/expert where weights/compute fit | query/activation/result per affected layer | exact op/state, P95 link and batch |
| Remote state hydrate | cache snapshot moved once to local compute | large initial state transfer | full hybrid StateCompatKey, transfer vs recompute |
| Remote direct state reads | compute on A, state bytes on B | potentially growing KV read per step | only on physically qualified fast fabric |
| Prefill/decode | full state changes owner at phase boundary | bulk KV+recurrent+indexer+task state | receiver reserve, complete schema and placement epoch |
| Token-range attention | ordinary softmax attention partial at each KV shard | query and compact summary + reduction | operator-specific algebra and numerics |
| Tensor/head/expert parallel | per-rank weights and state, or selected experts | collective/allreduce/scatter/gather | high-bandwidth links, same math and rank fencing |
| Mixed media/task | model stages or codecs on separate workers | stage outputs/latents/audio/frame | exact output frontier, privacy, latency |
| Whole graph/fabric | compiler-owned multi-device program | program submission and external ready events | state bridge and topology/safety |

These modes are **candidate options**, not a promise that any legal combination is fast. A high RTT relayed tunnel may make per-layer attention/FFN splitting unusable while allowing a few large layer groups as a capacity solution.

### ES-08.3 Exact token-range attention composition

An attention operator may expose an explicit `PartitionAlgebra` (not assumed by all attention types). For each query and independently masked KV token-range shard `i`, compute:

- `m_i` — maximum valid attention score in shard;
- `z_i` — sum of exponentials of local scores shifted by `m_i`;
- `n_i` — weighted value numerator under the same shift.

The reducer obtains `m = max_i(m_i)`, `Z = Σ_i z_i exp(m_i − m)`, `N = Σ_i n_i exp(m_i − m)`, and the attention result `N/Z` when `Z > 0`. Empty/all-masked shards contribute zero mass under carefully defined negative-infinity handling. This is a **real-arithmetic softmax identity**, not a guarantee of bitwise equivalence under different floating reductions. Positions, GQA/MQA, RoPE, causal/sliding masks, score biases and operator-specific source kernels must be validated.

Sparse global top-k, compressed attention, learned indexers and GDN/KDA recurrence do **not** automatically inherit the same partition legality. The compiler must reject unknown global selection/recurrence cases instead of returning superficially plausible output.

### ES-08.4 Topology measurement and cost-aware solver

A topology snapshot records for each authorized node/device: ISA/compiler/program capabilities, physical RAM/HBM/UMA/OS headroom, SSD/cache and program memory, and for each link: *directional effective bandwidth by transfer size*, RTT/P95 jitter, loss/MTU, data copy/crypto cost, direct/relay mode, physical RDMA/collective eligibility, tenant/privacy and egress restrictions.

An optimizer compares the **critical path** of full TaskProgram execution. Approximate one transfer's time as `payload_bytes/effective_rate + RTT + staging_and_crypto + remote_queue`, but calculate independent overlap and repeated per-token sync explicitly. It also accounts for compile/load/warmup and cache source-hydrate. **Never automatically distribute solely because another peer advertises more free memory.**

One model may be *capacity enabled* by a distributed placement while obtaining worse C1 decode tokens/s; reporting and SLO mode must distinguish these outcomes.

### ES-08.5 Peer protocol and tunnels

A remote machine runs the **same authorized closed-region worker** in PeerAgent mode, not a separate model server. It exposes narrowly scoped capability advertisement, resource reservation, trusted artifact/program preparation, state/activation transfer, region execution, health and lease fencing. It does not accept arbitrary shell/process/model-provided Python code.

Control plane is authenticated and bounded. A user may explicitly enroll already trusted machines with a private tunnel or direct authenticated link; network reachability does **not** grant permission to send model weights, prompts, activations or KV. The bulk data plane chooses actual physically qualified direct copy/RDMA or bounded encrypted streams. **A VPN/tunnel is not RDMA**, and relayed topology latency must be measured. Control and data planes may select different eligible links, but both have explicit peer identity, model/tenant access and versioned operation bounds.

Transfer source/destination allocations are leased until verified transfer/device completion. A host socket write or timeout does not imply RNIC/GPU DMA quiescence. Credit-based bounds protect KV/weight staging buffers; streaming output and device tasks have their own fairness limits.

### ES-08.6 Remote state and failure fencing

A remote mutable state shard handle contains `SessionEpoch, PlacementEpoch, StateShardGeneration, StateCompatKey, owner_identity, lease` and requested committed frontier. A worker validates all of these and returns provisional versions/Ready events; TaskCoordinator alone publishes the global root. An eventually consistent cache presence hint cannot substitute for an authoritative shard lease.

Migration:
1. freeze at legal **committed frontier**;
2. reserve/prepare destination and transfer/check entire compatible state;
3. wait for required events and verify model/position/numeric identity;
4. fence the old state owner, update placement root once under coordinator epoch;
5. retire old physical resources only after safe completion.

A network partition can stop progress; without explicitly built durable leader election/checkpoint/output ACK protocol, **fail-stop or replay** from known committed history rather than creating split-brain writers or claiming exactly-once client delivery. The same physical worker may be selected as coordinator for one session and be an ordinary remote worker for another; authority is scoped to each session epoch.

Detailed current specifications: [Distributed Placement](../.agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [Remote Peer Protocol](../.agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md) and [Distributed Overview](../.agents/docs/DISTRIBUTED_PARALLELISM.md).

---

## ES-09. User-facing CLI, SDK, HTTP and configuration

### ES-09.1 One engine behind every entry point

The *proposed* CLI is `zerogravity run <model>` for a local supported checkpoint or named alias; it instantiates the engine in process and emits interactive or piped output without mandatory daemon, account, cloud connection or manual device flags. `zerogravity serve` instantiates the **same engine** with a loopback model-aware router and HTTP adapters; the Rust SDK can embed that same library.

Optional reuse of a same-user verified local service socket is explicit and versioned; no sending private prompts to an unverified listener, silently installing system services, or modifying privileged network settings.

Other proposed commands include `models list/ps/import/pull/show/stop`, `inspect --explain-plan`, `config validate` and `config show --effective`. Command spelling, final port and profiles remain open product decisions until tested.

### ES-09.2 Scope of local profile and runtime settings

Versioned typed config resolves safe defaults, authorized service/user/global policy, model profile, alias/deployment, CLI flags and request-scoped allowed generation options. Semantic/source and security constraints are **non-overridable gates**:
- exact immutable checkpoint hash/quant/tokenizer/adapter and model equation;
- known task/grammar/sampling capability and session StateCompatKey;
- physical memory and device-safety qualification;
- permitted remote peers/data rights, listener auth and admin scopes.

Options with physical plan/model resident effects (quant pack, offload, context memory, graph capture, replica groups) require a safe new deployment or state migration; request sampler values can change only within validated per-request/session semantics. Every selected value is explainable with effective source, constraint and why a faster candidate was refused.

Config and JSON inputs are data: reject unknown security-affecting keys, oversized values, secrets in logs and out-of-scope profile aliases; do not run embedded config code.

### ES-09.3 OpenAI-compatible API is a conformance matrix

Initial intended endpoints:
- `GET /v1/models`: filtered authorized aliases/task capability, not leaked private paths/peer hardware inventory;
- `POST /v1/chat/completions`: exact model-native role template, task sampling/structured tool-call generation, correct ChatCompletions JSON and SSE deltas/finish/usage;
- `POST /v1/embeddings`: only when a real supported encoder/pooling TaskDriver passes quality/conformance;
- `POST /v1/completions`: optional raw completion after separate tests;
- `POST /v1/responses`: separately specified versioned subset, **stateless first**, typed Responses event stream and stored state only when actually implemented.

Endpoint presence does not imply that every cloud-provider feature works. An unimplemented `previous_response_id`, `store`, built-in web/file/computer tools, unsupported JSON schema/logprobs/audio or tool execution must return a **typed unsupported feature error**, not a successful response with changed semantics.

Tool call **generation** is structured model output. Authorization to execute a tool is separate application software above core inference; a model cannot grant itself network or shell permissions through a request field.

### ES-09.4 Streaming and client SDK tests

All compatible streams emit only **committed** output. ChatCompletions and Responses have **different SSE event shapes**, stop/finish and usage rules and must be tested with pinned official Python and JS SDK versions. UTF-8 split across tokens, stop markers across token boundaries, structured tool-call fragments, client backpressure, cancellation and failure after streaming begins are distinct conformance tests. An endpoint is “compatible” only for a documented **model × task × field × streaming × tested SDK** subset.

Status/error mapping should be stable for invalid source, unsupported model/quant/parameter/task, loading/deadline, physical budget, unauthorized model, overload, device failure and state incompatibility. Native administrator model/control endpoints require stronger authorization than an inference-only API key.

### ES-09.5 Bind, privacy and remote service policy

Default HTTP listener is **127.0.0.1/loopback**. External bind requires an explicit authentication/TLS or trusted reverse-proxy setup and per-tenant model alias visibility. Dummy SDK local API keys are not a remote security mechanism. Downloading models from the internet, sending to a peer, fetching arbitrary model-supplied URLs or persisting a conversation all require their own configured user policy.

CLI can expose simple progress for slow model warmup; expert `inspect` exposes source/quant/alias/model profile, device/remote placement, actual physical memory, cache availability, performance evidence and fallback reason without dumping private prompts or credentials.

Detailed [CLI, API and Config Contract](../.agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md), [Runtime Serving](../.agents/docs/RUNTIME_SERVING_AND_UX.md) and [Router and Residency](../.agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md).

---

## ES-10. Security, licensing, error taxonomy and observability

### ES-10.1 Security threat boundaries

| Boundary | Threats | Required defense |
|---|---|---|
| Model/quant/tokenizer/sidecar file | malicious metadata, overflow, hidden executable model code | bounded parsers, verified source/manifest, no arbitrary remote model code |
| Device compiler/program | untrusted binary, source-license violation, driver/fabric crash | trusted code provenance, explicit device/source-language policy, safety qualification |
| TaskDriver/agent app | model-generated request to execute shell/web/file actions | core emits data; external permissioned application performs actions |
| Physical allocations | stale pointer, device live DMA, page alias/cross-session state | generation-tagged lease, effect and event checks, quarantine |
| Public model HTTP API | unauthenticated model/tenant enumeration, media URL SSRF, unbounded requests | loopback default, auth/TLS/limits, admin separation, private logs |
| Multi-model router | substituted model/quant, cross-tenant cache, eviction of live state | immutable binding/alias revision, StateCompatKey, resource admission and pressure gates |
| Remote peer/tunnel | unauthorized tensor/weight disclosure, stale node, split brain | explicit enrollment, end-to-end peer authorization, trust scope, epoch fencing and no implied RDMA |
| Durable snapshots and caches | leaked previous prompts, stale full-state compatibility | immutable versioned identity/tenant, safe retention, integrity and encrypted storage where policy demands |

The engine's Rust source and device-kernel language policy remains strict. A Rust host calling a vendor compiler/driver is **not automatically** equivalent to a Rust-authored native accelerator program. No project-specific source code is copied merely because a public article documents the idea; direct reuse requires exact-file source license, notices, artifact rights and developer approval.

### ES-10.2 Canonical failure categories

At minimum define typed failures for:

```text
UnknownArtifact / UnknownSemanticOp / UnsupportedQuant / UnsupportedTask
UnsupportedShape / NoLegalKernel / CompilerUnavailable / DeviceUnsafe
UntrustedArtifact / BadInput / Unauthorized / Forbidden / ResourceOutOfBudget
ResourceLeaseInvalid / DeviceLost / IoShortRead / TransferCorrupt
IncompatibleState / StaleEpoch / StateBridgeUnsupported
ModelNotFound / ModelLoading / NoCompatibleReplica / RequestOverloaded
Cancelled / DeadlineExceeded / OutputSchemaMismatch / InternalFailure
```

Every internal error must document **which state was committed and which provisional work was discarded**. A retry policy can retry a failed *uncommitted* work envelope or select a compatible fallback, but cannot retroactively un-emit an already delivered output event. A `ModeReceipt` records actual model/quant/plan/replica/shape/task/quality/fallback and state/transport profile as appropriate, with secure redaction.

### ES-10.3 Observability without telemetry leakage

Metrics include request queue, cold model load, TTFT/total latency/P95/P99, typed committed output units, draft acceptance and verifier cost, GPU/CPU utilization and host synchronization, quant/graph candidate, physical RAM/GTT/VRAM/SSD/pinned allocations, expert/remote cache hits, remote activation/state transfer bytes and P95 link delay, reload/eviction, model/replica use, device failures, memory admits/refusals, and cancellation/abort.

A full report should carry anonymized, safely redactable source/model shape and runtime plan identity. Default telemetry **must not contain prompts, user identifiers, hidden reasoning, audio/image samples, API keys, model weight bytes or remote RDMA keys**. Deep profiling is explicit because instrumentation changes performance. Counter labels should avoid high-cardinality user/session identifiers in unauthenticated metrics.

### ES-10.4 Operational safety versus impossible guarantees

Avoid these overclaims:
- “GPU-visible virtual memory is guaranteed available physical memory.”
- “A process timeout proves the GPU/NPU/DMA stopped using a buffer.”
- “A distributed coordinator root swap physically atomically changes all peer memories.”
- “A private tunnel provides RDMA or low-latency interconnect.”
- “A hosted text API is the same as compiling the model to a device.”
- “Every model file recognized by the parser can execute.”
- “API compatibility means every proprietary cloud feature works.”
- “No user request can OOM the OS under any circumstances.”
- “An architecture can guarantee fastest throughput without running benchmarks.”
- “An MIT repository license automatically covers all source subfiles, weights, device programs and patented technologies.”

These statements may be convenient in marketing but are neither scientifically grounded nor appropriate implementation requirements. The system must detect and report when it cannot meet the intended capability.

Detailed security and failure specs: [Security/Reliability](../.agents/docs/SECURITY_AND_RELIABILITY.md), [Invariants](../.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md) and [Numeric Quality](../.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md).

---

## ES-11. Required invariants and reviewable implementation tests

The following concise IDs are the **integrated implementation acceptance contracts**. They are derived from detailed v4.x specs; not new assertions of test completion.

| ID | Nonnegotiable invariant | Minimum evidence |
|---|---|---|
| **INV-01** | Unknown model operator, source quant, tokenizer or position math never executes as if supported | malformed/unknown artifact and independent source reference |
| **INV-02** | Every optimized region preserves its declared values, effect order and numeric policy | portable vs accelerated layer/logit/state parity; forced fallback |
| **INV-03** | No device worker/domain publishes committed session state | fail second-domain Prepare with original root intact |
| **INV-04** | Only one TaskCoordinator root writer per SessionEpoch; no old peer/step commits | out-of-order/stale WorkerReply and epoch-fencing fault injection |
| **INV-05** | State snapshots/prefix hits require all causal effect domains | 3+ domains and partial compatible frontier/adapter/grammar mismatch |
| **INV-06** | Output follows logical publication and is sequenced; failed draft never streams | SSE/client abort and accepted-prefix verifier tests |
| **INV-07** | No free/recycle while any GPU/NPU/peer DMA/IO reader is live | async delayed completion, stale generation, cancellation/quarantine |
| **INV-08** | Global physical reserve includes every source/model/context/scratch/cache tier while hot local allocations avoid global mutex | quota stress + concurrent resident model loads |
| **INV-09** | Unsupported shape/quant/ISA/codegen picks legal fallback or typed refusal | ragged/odd sizes, driver upgrade, forced kernel disable |
| **INV-10** | Runtime device worker executes prepared handles; no SIR/CIR AST interpreted per token | trace CPU commands/counters, compare direct mode to reference |
| **INV-11** | TaskDriver is independent of model-specific server and arbitrary tool execution | text plus non-token task, denied ExternalAction request |
| **INV-12** | Exact model alias never silently changes checkpoint/quant/adapter | alias change mid-stream, overloaded fallback refusal |
| **INV-13** | Router, catalog and model residency do not create a second token scheduler/allocator | direct `run` vs `serve` instrumentation; extension drill |
| **INV-14** | Equivalent replica route preserves full session StateCompatKey | same-model replicas, sticky state, cross-replica migration fail/replay |
| **INV-15** | No unauthorized model pull, external listener or remote peer execution by default | clean offline CLI, HTTP bind/admin and tunnel authorization tests |
| **INV-16** | Cache directory hints are never authoritative live mutable state | stale remote prefix hint vs valid StateShardRef |
| **INV-17** | Placement/state migration occurs only at legal committed frontier, with fenced old owner | induced network partition and half-completed cache migration |
| **INV-18** | Dynamic device/network topology is measured; overlay tunnel is not assumed RDMA | direct/relay link cost and transport capability tests |
| **INV-19** | Distributed attention shard algebra requires operator-specific mathematical proof | masked softmax positive/empty shard and denied unknown recurrence |
| **INV-20** | One TaskProgram may use multiple model roles under a shared budget without confusing those with replicas | encoder+reranker and target+draft budget/correctness |
| **INV-21** | External compiler/native device program is qualified by source, shape, state bridge and observed hardware | mock graph-only vs real kernel qualification matrix |
| **INV-22** | Unsupported API features/parameter semantics don't return successful misleading responses | official pinned SDK endpoint/field/stream tests |
| **INV-23** | No trained routing/attention changes or lossy quant changes without explicit model/quality mode | expert top-k/source quant, numeric quality tests |
| **INV-24** | Optional future training parameter changes do not mutate live inference checkpoints | synthetic ParameterVersion update with old session intact |
| **INV-25** | Zero network/cluster/service overhead in single-model in-process CLI | cold local `run` with no socket/discovery threads or per-token HTTP |

For each, capture an executable test identifier, exact model/driver or deterministic simulated fixture, expected outcome and retained failure artifacts. A simulation may establish a **protocol invariant** but not physical performance/safety of a GPU or network stack.

### ES-11.1 Counterexamples are first-class test cases

Deliberately attempt unsafe behavior:
- model parser accepts an unknown GGUF tensor block or incompatible tokenizer;
- recurrent state is written in place before a second domain Prepare fails;
- an old cluster peer completes a kernel after its state slot generation has been reassigned;
- 100 concurrent cold-load requests each start a model compile/load;
- a router's exact model is unavailable and a lower quant of another checkpoint is warm;
- a slow SSE client pins unbounded KV memory;
- TCP write returns but GPU DMAs still reference transfer staging;
- a relayed WAN connection advertises false high-bandwidth RDMA eligibility;
- a graph-only accelerator compiles but cannot snapshot private recurrent state;
- a hot alias/profile update changes rendered token history mid generation.

**Correct rejection and safe fallback often matter more than a benchmark win.**

---

## ES-12. End-to-end execution walkthroughs

### ES-12.1 Supported local model, zero config

```text
CLI run local-file.gguf
  -> validate source path and immutable artifact hash
  -> ModelDefinition and supported TaskDriver checked
  -> hardware probe and physical admission
  -> CPU baseline plus optional qualified GPU region compiled/warmed
  -> create TaskSession, StateRoot R0, canonical rendered prompt
  -> SIR-derived prefill and StepProgram on DeviceWorker
  -> state domains Prepare R1, coordinator PublishRoot(R1)
  -> CLI outputs only committed text
  -> next step/cancel and event-safe resource retire
```

No public server, background system service, remote model download or peer enrollment. A missing executable GPU kernel may invoke a valid portable path; an unknown semantic model must be rejected rather than producing garbage.

### ES-12.2 Multi-model API with concurrent physical residency

```text
POST /v1/chat/completions {"model":"coding",...}
  -> authenticate + validate versioned API schema
  -> alias "coding" revision 7 -> exact ArtifactKey and TaskKind
  -> Router picks equivalent warmed Replica B by state/cache affinity
  -> if cold: deduplicated resident model reservation/prepare
  -> same TaskCoordinator as local CLI executes the task
  -> PublishRoot and Chat SSE/usage/finish response

Simultaneously:
POST /v1/embeddings {"model":"embedding",...}
  -> different supported encoder model and TaskDriver
  -> ResourceArbiter checks combined memory/queue
  -> independent stage result, own typed vector output
```

Changing alias revision while a stream is in flight does not change that stream's checkpoint/quant/tokenizer. A cold giant model is allowed to wait without monopolizing the warm model's device queue. Unsupported encoder API receives typed refusal if no real reference-qualified encoder exists.

### ES-12.3 Mixed-machine layer pipeline

```text
Node A: TaskCoordinator, layers 0..k, their KV/SSM
Node B: authorized RemoteRegionWorker, layers k+1..n and their KV/SSM
Control: authenticated peer + PlacementEpoch and SessionEpoch

For each staged step:
  A sends bounded hidden activation, shape/position and expected root
  B validates source/quant/adapter and executes closed layer region
  B returns next activation + prepared B-layer state and Ready
  A validates A+B required state domains, PublishRoot once
  client receives committed output sequence
```

Node B's physical KV remains local to B across steps; the entire state is **not copied over the network each token**. A lost peer aborts prepared work and may replay last valid checkpoint; no second owner independently commits tokens.

### ES-12.4 Remote attention and independently located state

A compute-rich worker A needs attention state held on cache-rich worker B. Candidate 1 migrates a compatible complete snapshot once to A, then computes locally. Candidate 2 executes attention on B beside KV and returns compact query/result with provisional B state, while A executes FFN. Candidate 3 repeatedly reads remote KV on A, typically expensive on ordinary tunneled WAN. The placement solver quantifies all three. It does not choose B merely because B reports more free RAM, and it rejects math/state/permission-incompatible options.

### ES-12.5 Device graph and future media task

An image TaskDriver performs bounded denoising iteration over a registered model SemanticProgram. A graph-capable provider prepares a large closed denoiser stage with a shape guard and full private latent/timestep effects. The same TaskCoordinator/admission/DeviceWorker submits graph handles, observes Ready and publishes a completed valid image only under its declared output contract. A fake TPU/static fabric provider can test interfaces; it does not qualify real physical AOT support.

### ES-12.6 Faulted speculative verifier with slow client

An optimized region proposes tokens and provisional recurrent/KV/RNG states. GPU reports completion but one sparse-indexer state Prepare fails. Coordinator retains prior committed root and does **not** send proposals to the client; physical buffers are quarantined if device work may remain. Correct retry can use a portable region or replay. A slow SSE output consumer receives only previously committed chunks and bounded credits, not unbounded GPU queues.

---

## ES-13. Implementation and proof sequencing

**Do not implement 86 experiments, 71 recommendations or all accelerator backends before one model runs.** This specification remains one coherent system despite a large research/history archive.

### ES-13.1 First executable vertical slice

The smallest valuable path is:

```text
trusted local model artifact + exact tokenizer
  -> ModelDefinition / source reference
  -> one real TaskDriver (text)
  -> SemanticProgram with 1–2 effects
  -> legal native CPU region + simple GPU candidate
  -> executable command/lease/event proof
  -> TaskCoordinator, per-session prepared root publication
  -> committed terminal output
```

Use one Rust library plus CLI binary initially; modularize internally before splitting crates. Choose a small source-verified checkpoint whose exact operator and quant have available reference vectors. Don't obscure unsupported semantics with fake random logits. Test a CPU source reference before optimizing GPU kernels. A native GPU may be blocked by toolchain feasibility; record it separately and still complete CPU reference.

**Minimum first gates:** physical codegen feasibility; source tokenizer/tensor/quant vector; effect-aware SIR; fake async buffer/events; exact StateRoot Prepare/PublishRoot; one real model task output; forced fast-kernel fallback; compiler shape guard; local CLI with no network/server and source/quant integrity.

### ES-13.2 Then prove composability

Add, in order, without duplicating core:
- a second source model using already supported ops, then one genuinely new StateDomain/op;
- another kernel candidate with the same RegionContract and forced disable/fallback;
- a mock non-token encoder or denoiser TaskDriver using same workers/resources;
- a catalog with two immutable exact model aliases, one resident lifecycle and bounded admission;
- loopback `serve`/chat/models API mapping through the same engine; official SDK stream checks;
- concurrent load/idle eviction and equivalent replica load-balancing tests;
- fake two-node layer+own-state remote worker, then physical authorized peer only with user approval.

Expand real MoE/recurrent/indexer/source quant, attention/FFN and heterogeneous devices by measured benefit and availability. Add AOT and static dataflow/future gradient task seams only after the minimal runtime survives extension drills.

### ES-13.3 Existing stable project phase program

The **24 existing IDs remain authoritative** in [Implementation Roadmap](../.agents/plans/IMPLEMENTATION_ROADMAP.md), from ZG-P00 workspace through ZG-P23 extension proof. Supporting [Stage0 Gates](../.agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md) and [Experiments](../.agents/plans/EXPERIMENT_PROGRAM.md) contain a large optional matrix. Every phase has an implementation/validation owner and a proof package.

The [Implementation Playbook](../.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md) groups the stable phases into dependencies, concrete file/trait/test ownership and source-of-truth checks to prevent agents from implementing an isolated advanced feature before basic execution is correct.

### ES-13.4 Definition of Done per capability/phase

A capability is **not done** because a Markdown file or empty crate exists. Completion evidence includes:
1. exact code patch/commit or source hash and which modules were modified;
2. documented model/checkpoint/quant/tokenizer/driver/OS/provider artifact used;
3. actual runnable reference/task result and expected logs/tests (or truthful BLOCKED state);
4. independent semantic and numeric/quality verification where applicable;
5. state/buffer/cancellation/tenant/resource failure tests;
6. source model/param/peer/transport fidelity and runtime fallback;
7. profiling for performance-qualified changes, matched baseline and raw uncertainty;
8. docs/decision/status/roadmap/spec updates; source license/NOTICE as applicable;
9. no unrelated hidden adapter/server/allocator forks or new security privileges.

**The status label must reflect the weakest unproven gate**. A function may be semantically correct on CPU and remain UNSUPPORTED on NPU/TPU; a graph may compile but be unsafe for stateful speculation; a simulated peer may prove protocol while providing no physical speed evidence.

---

## ES-14. Acceptance and architectural sustainability

The primary long-term product promise is **low change radius**:

| Extension | Intended owned modifications | Unrelated parts that should stay unchanged |
|---|---|---|
| New checkpoint built from known ops | Artifact mapping, ModelDefinition and source vectors | HTTP, TaskCoordinator, DeviceBackend |
| Novel trained recurrent/attention operator | SemanticOp/ref, state codec, legal generic lowering | other model families, public API |
| Fast fused model/layer kernel | RegionExecutionProvider candidate, shape/numeric/effects proof | model loader, session root, service router |
| New GPU/NPU/AOT/dataflow target | device/backend provider, program lifecycle, safe event bridge | semantic model and TaskDriver |
| New quant/packing format | QuantCodec and numeric oracle, optionally derived artifact | source model semantics unless intentionally requantized |
| New AI TaskDriver | trusted stages, stream frontier, required semantic ops | other tasks and physical allocator |
| New authorized network peer/transport | NodeCapabilities/TransportProvider/StateBridge and topology test | model formulas and serving endpoints |
| New attention cache topology | PartitionAlgebra, StatePlacement/TransferPlan, cost tests | unrelated checkpoint adapters |
| New replica/cold model policy | coarse ModelRouter/ResidencyManager policy and tests | per-token device execution engine |
| New OpenAI-compatible endpoint | request/output schema adapter and pinned SDK tests | model/DeviceWorker math |
| Future optimizer/fine-tuning | GradientProgram/ParameterVersion/optimizer effects | active immutable inference session semantics |

An extension is a **design failure** if it requires a new model-specific API server, independent session-state publisher, duplicate resource arbiter, hidden model-name branches in generic device execution, different semantic runtime, or a broad mutable global plan. When extensions reveal a real missing shared capability, change the smallest central interface with an independent regression suite, not add another engine.

### ES-14.1 Architecture review before implementation

Review each proposed optimization against:
- **Correctness:** does it preserve exact trained model math and task/state effects?
- **Resource:** can it reserve and safely retire actual physical memory/events/links?
- **Hot path:** does it avoid unintended host/device sync, per-token abstraction work and stale code capture?
- **Fallback:** can the same model/task continue without the specialization?
- **Versioning:** what schemas/artifacts/bundle cache/state compatibility change?
- **Security:** does it introduce tool privileges, executable model code, remote data export or unsafe device state?
- **Benefit:** will realistic full task quality, latency/P95, throughput/memory/energy and source-specific hardware justify it?
- **Minimality:** can it be implemented through an existing provider/policy rather than a new crate, process or serialized IR?

### ES-14.2 Historical traceability and open decisions

The original user decisions, proposals and unresolved product questions remain in [Decision Register](../.agents/plans/DECISION_REGISTER.md). The [critical review](../.agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md) retains rejected-design rationale, and the [research source atlas](../.agents/prior-art/RESEARCH_SOURCE_ATLAS.md) retains prior-art attribution. Superseded v2/v3 bodies remain in Git history. **The currently proposed Rust interfaces are not irreversible public types**, and future measured code may justify revisions. Frozen security, numerical and session guarantees require an explicit decision/test protocol before relaxation.

### ES-14.3 What makes the published documents usable independently?

A new implementing agent should be able to:
1. identify **one root** and latest architecture edition;
2. choose one bounded, dependency-valid phase rather than implement every possible device at once;
3. find the exact owner, input/output/state/effect, resource and security contracts;
4. see whether a requirement is USER, working recommendation or OPEN;
5. run the relevant reference and negative tests without this conversation;
6. report the result accurately, update plans/decision register and hand off;
7. reject a requested optimization that violates math/StateRoot/resource/auth even if a microbenchmark looks attractive.

The [Agent Implementation Playbook](../.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md), [Traceability Matrix](../.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md) and [Document Index](README.md) implement these navigation and proof obligations.

---

## ES-15. References and formal source of truth

The [Whitepaper](WHITEPAPER.md) presents the rationale and research context for public review. Its [References](REFERENCES.md) are attributed prior art. The actual normative design is specified here and in the current active source-neutral contracts under `.agents/docs`, with [Master Plan](../.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) for cross-domain integration, [Release Gates](../.agents/plans/RELEASE_ACCEPTANCE_GATES.md) for concrete acceptance and [Decision Register](../.agents/plans/DECISION_REGISTER.md) for exact user intent versus open policy.

**Document status: complete integrated design, unimplemented.** This is suitable as an implementation handoff and engineering design-review submission. Performance leadership, public API compatibility, physical accelerator breadth, distributed availability and safety certification remain **testable ambitions rather than achieved facts**.

### ES-15.1 Compact implementation glossary

| Term | Exact intended meaning |
|---|---|
| **ArtifactKey** | Versioned immutable checkpoint/tensor/quant/tokenizer/template/model-sidecar source identity; one change may invalidate state and model binding |
| **ModelDefinition** | Registered validated source-model recipe and semantic operator graph; not a checkpoint-specific HTTP server |
| **SemanticProgram / SIR** | Authoritative typed mathematical computation and explicit state effects, independent of device execution |
| **SemanticRegion / closed region** | Subgraph with declared external inputs/outputs/effects and numeric/state boundary; may be replaced by a specialized implementation |
| **RIR / KIR** | Optional *transient* region optimization and kernel/program-candidate views inside compilation |
| **CIR** | Typed asynchronous resource/timepoint/command dependencies carried inside a compiled ExecutableBundle, not a semantic session commit system |
| **ExecutableBundle** | Immutable prepared native/graph/fabric code handles plus shape guard, resource plan, fallback and device program provenance |
| **RegionExecutionProvider** | Qualified adapter that implements a closed semantic region on a kernel, captured graph, AOT program or static fabric under same ownership |
| **TaskDriver** | Trusted task-stage/loop/stream policy (language, encoder, audio, image/video); no implicit external tools |
| **TaskCoordinator** | Single logical writer per TaskSession/authorized shard, owns current committed StateRoot, attempts and ordered output |
| **StateRoot** | Versioned immutable manifest of all required local/remote domain versions at a common legal TaskFrontier |
| **PreparedDomainVersion** | Provisional candidate state with verified schema/frontier and required event readiness, not publicly committed |
| **BufferLease** | Generation-tagged physical allocation/access lifetime with event-backed retirement and resource admission |
| **ResourceArbiter** | Coarse global and per-node physical capacity/quota admission authority; device-local pools allocate hot buffers |
| **PlacementPlan** | Costed compute/parameter/live-state/immutable-cache assignment and TransferPlan bound to a topology/epoch |
| **StateShardRef** | Versioned authoritative remote/local mutable state allocation identity, owner, lease and schema for current root |
| **CacheLocationHint** | Revalidatable, possibly stale location advertisement for immutable prefix/model-derived data, never authority to publish live state |
| **RegisteredModel** | Verified immutable source/semantic identity available in catalog, not necessarily physically loaded |
| **ModelAlias / ModelProfile** | Versioned friendly name and allowed user settings bound to a RegisteredModel source/roles |
| **ResidentModel / Replica** | Actual physically admitted warmed model deployment; one deployment can itself be sharded across devices |
| **ModelRouter** | Coarse request-level exact model source and compatible replica selection; never a per-token second model engine |
| **ModeReceipt** | Truthful scoped record of actual source/numeric/shape/task/device/placement/fallback and evidence level |
| **TaskServiceAdapter** | Separate permissioned external model/API service integration, **not** native SIR-to-kernel/graph execution |
| **Qualify** | Demonstrate a specific model/task/quant/shape/device contract via tests and recorded evidence, not assume it from a capability string |

These names are **contract vocabulary, not frozen public Rust ABI**. A future code review may combine traits or rename internal structs if ownership, semantics, fallbacks and validation evidence remain intact. The strongest proof of simplicity is an executable vertical slice followed by successful *second-model, second-provider and non-token TaskDriver* extension drills.


````

## Source: docs/PUBLICATION_AND_REVIEW_CHECKLIST.md

Original UTF-8 bytes: 10283; SHA256 `da14176e79c2c7ec85f663dc16ac869bb5860a8ddbd84aacb39a367ca15cafb1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Whitepaper publication and final architecture review checklist

**Publication candidate:** [Zerogravity Whitepaper v0.1](WHITEPAPER.md), [Engineering Specification edition 1](ENGINEERING_SPECIFICATION.md).  
**Draft date:** 12 October 2026. **Status:** reviewable **design manuscript**, **NOT a results paper**, not a fully cleared/legal-public release.  
**Scope:** technical accuracy, architecture coherence, source attribution, reproducibility, editorial style, accessibility, software/hardware qualification and agent handoff.

This checklist prevents a polished design document from being mistaken for a proven implementation. Exact completed/pending status should only be marked after the checks are run, not because an earlier agent said a manuscript looked complete.

This dated manuscript checklist is not current runtime status. Use the [plan index](../.agents/plans/TODO.md) and [TODO spec](../.agents/plans/TODO-SPEC.md) for current work, passing gates and committed completion.

## 1. Technical claims and source honesty

- [x] Whitepaper title/abstract clearly identify the work as a **proposal**, not implemented runtime or measured fastest serving engine.
- [x] Distinct claims for mathematical semantics, numerical agreement, checkpoint/task quality and target sampling distribution.
- [x] No first-invention claim for prior-art paging, tiling, speculative verification, disaggregation, source-compiler layering or external topology technologies.
- [x] No universal tokens/s, throughput/energy or percentage superiority is claimed without reproducing source/model/task/hardware.
- [x] Compatibility status is per model checkpoint/quant/task/shape/device/driver and event/state safety.
- [x] Proprietary vendor compiler runtime and Rust-authored native device source remain distinct; no fake universal TPU/wafer kernel support.
- [x] Transparent distinction between architecture-compatible, simulated, functional, performance-qualified and production-qualified.
- [x] A recognized model file isn't claimed to execute unknown trained operators.
- [x] No source or training/fine-tuning functionality claimed to exist currently.
- [x] The spec disclaims global durable exactly-once streaming and automatic distributed leader failover.
- [ ] When experimental code exists, add a **separate reproducibility/results appendix** with real model/quant/driver/OS/link/thermal/workload, independent quality, p95/energy, matched baseline and raw data. *Pending by design: no engine implementation.*

## 2. Architecture consistency

- [x] **One SemanticProgram and one ExecutableBundle** as enduring compilation/execution concepts; region/kernel/command views are internal, not four mandatory public dialects.
- [x] Trusted TaskDriver controls computation stages; external tools/agent network/file actions belong to a separately authorized application boundary.
- [x] One TaskCoordinator writer per TaskSession/shard; StateDomains only **Prepare**, device/CIR only Ready/Failed, TaskCoordinator alone **PublishRoot**.
- [x] ResourceArbiter coarse physical budget and fast device-local pools with generation/event leases; no GPU virtual memory mistaken for available RAM.
- [x] One RegionExecutionProvider handles native kernel/captured graph/AOT/dataflow/limited managed-state program as qualified.
- [x] Distributed placement separates compute, parameters, live mutable state and immutable prefix/cache, with costed TransferPlan and trusted peers.
- [x] ModelCatalog/ModelRouter/ResidencyManager use same model/session/device engine as direct CLI and HTTP; no model-specific duplicate runtimes.
- [x] Exact model routing never silently substitutes checkpoint/quant/adapter; user-authored explicit model-choice policies separate from same-source replica balancing.
- [x] Local one-command run is independent of server, cloud account, mandatory cluster or remote model pull.
- [x] OpenAI compatibility explicitly defined as tested endpoint/field/SDK/SSE subset.
- [x] First source-verified CPU/GPU vertical slice is a prerequisite to ambitious multi-model/distributed qualification, not a framework-only launch.
- [ ] When real source exists, confirm architecture with working multi-model, non-token and second backend **extension drills**, profiling and resource-fault injection. *Pending by design.*

## 3. Publication manuscript and bibliography

- [x] Manuscript includes abstract, problem statement, design axioms, component architecture, state/physical resource/distributed/router descriptions, numerical/security boundaries, falsifiable evaluation and limitations.
- [x] Related prior work is cited in a separate numbered [References](REFERENCES.md) dossier with primary official links; non-MIT project names appear there/source research, not as core architecture types.
- [x] Useful design rationale and attribution remain in current research/references; retired document versions remain in Git history.
- [x] Diagram includes a textual equivalent explaining node/edge semantics; no image or copyrighted figure imported.
- [x] Tables have explicit headings and enough surrounding prose; Markdown renders in ordinary repository viewers.
- [ ] **Author credit/byline, institutional affiliation and contact** need the project's owner to select a public identity. No personal or organization attribution was invented.
- [ ] Confirm the publication artifact's licensing and third-party permissions before public release, following the owner's [MIT project policy](../.agents/plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md) and [LICENSE](../LICENSE). Separate third-party material and formal publication clearance still require review.
- [ ] **Trademark, third-party material and publication legal review** remain owner responsibility; prior-source code/weights/license rights must be reassessed if implementation incorporates them.
- [ ] **Persistent publication date/version/DOI/canonical URL** to be assigned when publicly submitted; internal paths are local developer references, not external URLs.
- [ ] **External link rot audit at actual publication time** and citation-style proofreading should be repeated; references were researched as of 12 October 2026.
- [ ] **PDF/HTML publication artifact, accessibility/print QA and figure export** optional for release; no PDF was requested, generated or claimed in this design pass.
- [ ] **Independent technical peer review** and copyedit/author signoff still required for a formal release.

## 4. Implementation planning and verification

- [x] Public [Engineering Specification](ENGINEERING_SPECIFICATION.md) includes ES-00…ES-15, identity, interfaces, state/lease/worker protocols, fallbacks, network and API, negative tests, six end-to-end traces and release conditions.
- [x] [Implementation Playbook](../.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md) lists stable P00–P23 with bounded deliverables/independent proof and stop/blocked instructions.
- [x] [Requirements Traceability](../.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md) maps explicit user decisions U-001…U-022 to technical source/phase/test requirements and integrated invariants.
- [x] [plan TODO index](../.agents/plans/TODO.md), root README and [Document Index](README.md) provide ordered navigation and authoritative/research separation.
- [x] No implementation phase marked complete merely because an acceptance test was described.
- [x] Remaining publication, release model/task/backend, public API and network-trust choices stay explicit. Follow the selected project license and qualified vendor-runtime scope rather than reopening settled decisions.
- [ ] Before implementation, inspect the live source revision, working changes and agent ownership; follow [AGENTS.md](../AGENTS.md).
- [ ] Record actual test commands/results, relevant model-content identities, fixtures/hardware, failures and unfinished work in the plan/run TODOs. Group completed work under its landing commit after its gates pass, following [TODO-SPEC.md](../.agents/plans/TODO-SPEC.md); no generated handoff or separate proof inventory is required.

## 5. Automated integrity checks

Current publication documents should pass:

- headings and balanced fenced code blocks across active Markdown;
- all relative Markdown file references (and internal section anchors when possible);
- no accidental stale root path or broken canonical architecture references;
- all stable phase IDs ZG-P00…P23, experiments H01…H86, required S0 gate groups, U/P/O decision coverage;
- no non-MIT source project names used as core Rust type/interface names in normative documents, while named sources remain in prior-art research and bibliography;
- zero source/runtime/Git/network changes unless separately authorized.

Record applicable checks in the owning plan TODO and commit verified changes together. Structural checks establish document integrity only; they do not establish runtime correctness, current external URLs or publication approval. No generated archive or documentation SHA manifest is required.

## 6. Publication release decision

**Current recommended classification:**

- **Design manuscript ready for external technical review:** YES, subject to the final integrity audit and editorial check.
- **Ready for public distribution with credited authors/license:** **PENDING** public byline, publication clearance, owner approval and final citation/link QA under the selected project license.
- **Ready to publish experimental performance/reliability results:** **NO** — each publication claim still needs its matched source/task/device measurements and review; accepted implementation slices do not pass every publication gate.
- **Ready for implementing LLM agents to start a bounded authorized phase:** YES within the user-authorized scope, following the live repository checks and [current plan TODOs](../.agents/plans/TODO.md).
- **Guaranteed highest-performing or supports every hardware/model:** NO; those remain engineering goals.

An eventual results edition should retain the design/limitations, add independent reproducibility, source/driver/test matrices, exact negative results and measured comparison against contemporaneous baselines. It should not retroactively treat this design document as already proving those results.

````

## Source: docs/README.md

Original UTF-8 bytes: 9185; SHA256 `a48301c57c61da0465e19de248c7452af749db88b08cc385231261dfda46c6b8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity documentation

The whitepaper and engineering specifications describe the architecture. Current
implementation, tests and remaining work are tracked in
[the plan TODO index](../.agents/plans/TODO.md). Agents use
[AGENTS.md](../AGENTS.md), their plan TODO and their own run TODO.

## Read according to your purpose

| Audience | Start here | Then |
|---|---|---|
| **Technical reader, collaborator or reviewer** | [Whitepaper](WHITEPAPER.md) | [References](REFERENCES.md) and [Publication/Review Checklist](PUBLICATION_AND_REVIEW_CHECKLIST.md) |
| **Systems/ML engineer implementing the runtime** | [Engineering Specification](ENGINEERING_SPECIFICATION.md) | [Agent Implementation Playbook](../.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md) and the relevant subsystem contract |
| **Coding LLM or autonomous agent** | [Plan TODO index](../.agents/plans/TODO.md) | [AGENTS.md](../AGENTS.md), the selected plan TODO and its linked specifications |
| **Architecture/product lead** | [Whitepaper](WHITEPAPER.md) + [Master Architecture](../.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) | [Open decisions](../.agents/plans/DECISION_REGISTER.md), [Critical Review](../.agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md), [Final Architecture/Publication Review](../.agents/prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md), [Release Gates](../.agents/plans/RELEASE_ACCEPTANCE_GATES.md) |
| **Performance or correctness reviewer** | [Engineering Spec ES-11…13](ENGINEERING_SPECIFICATION.md) | [Benchmark/Numerics](../.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md), [State/Failure Invariants](../.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md), [Experiments](../.agents/plans/EXPERIMENT_PROGRAM.md) |

## Documentation categories and authority

### A. Publication-facing

- [**Whitepaper**](WHITEPAPER.md): cohesive, externally readable technical manuscript. Research context, architectural contributions, falsifiable hypotheses, explicit current limitations. **Proposed system**, not benchmark results.
- [**Engineering Specification**](ENGINEERING_SPECIFICATION.md): integrated normative requirements **ES-00…ES-15**, including interfaces, state machine, physical memory, models, task control, routing, distributed placement, API and acceptance tests.
- [**References**](REFERENCES.md): numbered primary papers and official sources, with source author benchmarking and license caveats.
- [**Publication and Review Checklist**](PUBLICATION_AND_REVIEW_CHECKLIST.md): technical/claims/editorial/policy gates for a public whitepaper version and future results revision.

### B. Canonical current core design (project-neutral)

The public Engineering Specification governs cross-component consistency. For more exact detail, use these normative source files:

| Concern | Source |
|---|---|
| Minimal ontology and ownership | [Modular AI Architecture](../.agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md) |
| Typed Rust extension seams | [Composition and Extension Contracts](../.agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md) |
| SIR and legal compiler transforms | [Semantic IR and Compiler](../.agents/docs/SEMANTIC_IR_AND_COMPILER.md), [Task/Compiler Refinement](../.agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) |
| Device commands and lifecycle | [Async Resource Command IR](../.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md), [Device Kernels](../.agents/docs/DEVICE_KERNEL_AND_CODEGEN.md) |
| Native, graph and static fabric devices | [Accelerator Execution Families](../.agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) |
| Logical StateRoot, workers and failure | [Minimal Execution/State Protocol](../.agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), [State/Cache Transactions](../.agents/docs/STATE_CACHE_AND_TRANSACTIONS.md) |
| Physical resource and cost | [Memory Residency and IO](../.agents/docs/MEMORY_RESIDENCY_AND_IO.md), [Cost Model/Autotuning](../.agents/docs/COST_MODEL_AND_AUTOTUNING.md) |
| Scheduler, batching, speculation | [Scheduler and Speculation](../.agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md) |
| Mathematical model source/quant | [Model Semantics](../.agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md), [Formats and Quantization](../.agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md), [Operator Catalog](../.agents/docs/OPERATOR_AND_FUSION_CATALOG.md) |
| Multi-model routing/residency | [Model Router Contract](../.agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) |
| CLI, configuration and API | [Zero-Config CLI/API](../.agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md), [Runtime Serving](../.agents/docs/RUNTIME_SERVING_AND_UX.md) |
| Distributed topology, remote state | [Distributed Placement](../.agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [Remote Peer/Tunnel State Handoff](../.agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md) |
| Multimodal tasks and NPU split | [Product/Model Pipeline](../.agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md), [GPU/NPU Cooperative Tasks](../.agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) |
| Security and fidelity | [Security and Reliability](../.agents/docs/SECURITY_AND_RELIABILITY.md), [Numerical Quality](../.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md), [Invariants](../.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md) |
| Rust workspace minimality | [Rust Engineering](../.agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md) |

**Authority rule:** latest explicit user decision and actual verified code/tests outrank static plans; for design-only work, the [Engineering Specification](ENGINEERING_SPECIFICATION.md) and current [Master Plan](../.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) take precedence over old subsystem appendices when inconsistent. Register contradictions rather than silently creating incompatible interfaces.

### C. Implementation plans and tests

- [Plan TODO index](../.agents/plans/TODO.md): one line per plan.
- [Agent workflow](../AGENTS.md): per-plan and per-agent checklists, commit
  provenance, coordinator integration and carrying unfinished items forward.
- [Architecture roadmap](../.agents/plans/IMPLEMENTATION_ROADMAP.md),
  [requirements/test design](../.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md)
  and [decision register](../.agents/plans/DECISION_REGISTER.md): design references.
- [Release gates](../.agents/plans/RELEASE_ACCEPTANCE_GATES.md) and
  [experiments](../.agents/plans/EXPERIMENT_PROGRAM.md): planned checks; completion
  and actual results belong in the selected plan TODO.

### D. Non-normative historical design and cited prior art

Retained for context and legal/provenance review, **not as additional architectural authorities**:

- [Prior-art and License Policy](../.agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md) and [Original Source Atlas](../.agents/prior-art/RESEARCH_SOURCE_ATLAS.md).
- [Compiler/Runtime Research](../.agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md), [Historical Reconciliation](../.agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md), [Strix Halo Dossier](../.agents/research/STRIX_HALO_OPTIMIZATION_DOSSIER.md).
- [GPU/NPU Source Investigation](../.agents/prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md), [Accelerator Platform Research](../.agents/prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md), [Distributed/Tunnel Research](../.agents/prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md), [Router/API Research](../.agents/prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md).
- Retired design snapshots are available in Git history. Their useful rejected-design rationale is retained in the [critical review](../.agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md); current work is tracked in [plan TODOs](../.agents/plans/TODO.md).
- The external pre-existing research library is **READ-ONLY**.

## Invariants in one paragraph

The model's actual trained mathematics and source quantization are authoritative. Device kernels and compiled whole-regions may be specialized without creating another inference engine or silently changing the model. Device/IO workers produce **prepared** state; only one TaskCoordinator publishes each full StateRoot and its corresponding ordered output. Global physical admissions and fast local device pools share one budget; an unavailable device or invalid shape uses a legal fallback. A ModelRouter resolves exact source and compatible replicas, not an unrequested alternative target. Network peers require explicit authorization, real link measurement and epoch-fenced state handoff. API compatibility is an independently tested feature/endpoint subset, not a promise of all cloud tools.

## Publication and implementation status

Consult [the plan TODOs](../.agents/plans/TODO.md) for implemented/tested scope,
failures and remaining work. A design, proposed experiment or packaging definition
does not establish measured quality, physical backend support or a release.
Publication/tagging/release still require the owner's authorization.

````

## Source: docs/REFERENCES.md

Original UTF-8 bytes: 10251; SHA256 `6bba38cd43a9b013d9619c8ed755aab52cb566a43336125e6e63462a371dd20e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# References and source provenance

*Zerogravity whitepaper supporting bibliography · literature review snapshot: 12 October 2026.*

This bibliography supports the **design motivations and prior-art comparisons** in [the whitepaper](WHITEPAPER.md). Its links are to publications or primary documentation, not to incorporated code. **No claim of authorship or first invention is made for the cited techniques.** Source names appear here as attributed research, not as the names of zerogravity's normative interfaces. Public materials may change after the snapshot date; the exact reference should be revalidated at submission.

## Peer-reviewed and academic research

### R01
Kwon, W. et al. (2023). **Efficient Memory Management for Large Language Model Serving with PagedAttention.** *Proceedings of the ACM Symposium on Operating Systems Principles (SOSP).* https://doi.org/10.1145/3600006.3613165 — Dynamic state paging, efficient memory sharing and batching. Published performance improvements belong to the authors and their test setup.

### R02
Dao, T., Fu, D. Y., Ermon, S., Rudra, A. & Ré, C. (2022). **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.** *Advances in Neural Information Processing Systems.* https://arxiv.org/abs/2205.14135 — Exact attention tiling and hierarchy-aware memory traffic. Distinguish exact dense attention from its separately described approximate block-sparse extension.

### R03
Leviathan, Y., Kalman, M. & Matias, Y. (2023). **Fast Inference from Transformers via Speculative Decoding.** *International Conference on Machine Learning.* https://proceedings.mlr.press/v202/leviathan23a.html — Target-distribution-preserving speculative sampling under specified algorithmic conditions. The paper's speedups are not measurements for zerogravity.

### R04
Zheng, L. et al. (2024). **SGLang: Efficient Execution of Structured Language Model Programs.** *NeurIPS 2024.* https://arxiv.org/abs/2312.07104 — Structured language program execution and prefix-aware cache reuse. Distinct from zerogravity's deliberately minimal, trusted TaskDriver rather than an embedded arbitrary workflow language.

### R05
Zhong, Y. et al. (2024). **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.** *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’24)*, pp. 193–210. [Peer-reviewed proceedings](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin); [preprint](https://arxiv.org/abs/2401.09670). Prefill/decode interference, SLO-based placement and transfer costs. Published benefits depend on workload/interconnect and are not reproduced here.

### R06
Lattner, C. et al. (2020). **MLIR: A Compiler Infrastructure for the End of Moore's Law.** https://arxiv.org/abs/2002.11054 — Motivation for typed compiler layering and target extensibility. Zerogravity does not propose shipping that compiler as a required runtime component.

### R07
Zheng, L. et al. (2020). **Ansor: Generating High-Performance Tensor Programs for Deep Learning.** *14th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’20)*, pp. 863–879. [Peer-reviewed proceedings](https://www.usenix.org/conference/osdi20/presentation/zheng); [preprint](https://arxiv.org/abs/2006.06762). The later auto-scheduler introduction is an implementation background source, not the publication venue. Motivates search, real target measurement and bounded optimizer complexity; no claim a generated kernel always beats hand-written code.

### R08
**LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism.** https://arxiv.org/abs/2404.09526 — Elastic sequence parallelism and dynamic KV/state placement for long contexts; no automatic generalization to recurrence.

### R09
**Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.** https://arxiv.org/abs/2407.00079 — Distributed context caching and admission/transfer, with source-author benchmarks.

### R10
**KernelFlume: Elastic Core-Attention Scaling for Agentic Long-Context Decoding.** https://arxiv.org/abs/2606.29207 — Weight-vs-attention-state separation and token-range KV ownership. Reported experimental and simulated results must be distinguished.

### R11
**Cerberus: Efficient Inference with Adaptive Parallel Decoding and Sequential Knowledge Enhancement.** https://arxiv.org/abs/2410.13344 — A *parallel decoding algorithm*, not the similarly spelled wafer-scale hardware manufacturer. Any exact sampling-distribution claim requires algorithm-specific proof.

## Primary compiler and hardware documentation

### R12
**StableHLO compatibility and dynamism contracts.** https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md and https://github.com/openxla/stablehlo/blob/main/docs/dynamism.md — Versioned operator and bounded shape semantics, used for research into compiler interoperability. This is not a required zerogravity binary format.

### R13
**Stream-dialect compiler IR.** Research on resource lifetimes, asynchronous work and explicit timepoints. No source-dialect or runtime dependence implied.

### R14
**NVIDIA Inference Xfer Library (NIXL), overview and backend selection.** https://docs.nvidia.com/nixl/getting-started/overview/ and https://docs.nvidia.com/nixl/user-guide/backend-selection/ — Transfer abstractions across DRAM/VRAM/storage, transports and registered-resource constraints. The mere presence of an encrypted tunnel does not create physical RDMA capabilities.

### R15
**Google Cloud TPU7x (Ironwood), architecture and topology.** https://cloud.google.com/tpu/docs/tpu7x — Compiler, on-chip memory and interchip placement motivation. Hardware marketing specifications are not zerogravity performance results.

### R16
**Cerebras Software Documentation, How Cerebras Works and Weight Streaming.** https://training-api.cerebras.ai/en/1.7.1/cerebras-basics/how-cerebras-works.html and https://training-api.cerebras.ai/en/1.8.0/wsc/cerebras-basics/cerebras-execution-modes.html — Graph-compiled wafer-scale execution and weight-streaming prior art, with source caveats about training modes and hardware access.

### R17
**NVIDIA Blackwell native FP4 and inference precision.** https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ — Research motivation for distinct low-bit block-scaling and reference quality, not blanket compatibility of all FP4 variants.

### R18
**NVIDIA DGX Spark product specifications.** https://www.nvidia.com/en-us/products/workstations/dgx-spark/ — Coherent CPU/GPU LPDDR system profile, distinct from datacenter GPU HBM/rack topologies. Vendor headline throughput is not a measured zerogravity benchmark.

## Serving, APIs, local user experience and networking

### R19
**Ollama CLI, Modelfile and model residency FAQ.** https://docs.ollama.com/cli ; https://docs.ollama.com/modelfile ; https://docs.ollama.com/faq — Local one-command interaction, named model presets, load/unload and memory-bound multi-model residency as user-experience prior art. Exact source licenses/implementation revisions require review before reuse.

### R20
**llama.cpp native server and model router documentation.** https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md — Published router mode, model aliases/loaded models and compatibility APIs. Zerogravity's proposed design deliberately retains one shared model executor rather than using an existing runtime as a mandatory implementation dependency.

### R21
**OpenAI API official reference**, including [Chat Completions](https://developers.openai.com/api/reference/chat-completions/overview), [Responses](https://developers.openai.com/api/reference/responses/overview) and [Responses streaming](https://developers.openai.com/api/docs/guides/streaming-responses). Public API compatibility is a **versioned, tested endpoint-and-feature subset**; a locally built endpoint cannot claim every cloud-only feature or execute tools merely by naming itself compatible.

### R22
**Encrypted overlay network direct/relay distinctions.** Link latency and path-dependent throughput important for private tunneled peer planning. Overlay networking is not equivalent to GPU memory coherence or RDMA.

### R23
**Rust QUIC transport documentation.** Optional reliable stream/datagram implementation candidate for authenticated, bounded peer transport; it is not an architectural dependency or a guarantee of better performance on every link.

## Additional project research dossiers

The historical research corpus provides deeper contemporaneous implementation comparisons and license records that should be retained as **non-normative**:
- [Prior-art license and provenance policy](../.agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md)
- [Compiler/runtime comparison](../.agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md)
- [GPU/NPU cooperative compute source report](../.agents/prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md)
- [Accelerator family research](../.agents/prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md)
- [Distributed placement and tunnel research](../.agents/prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md)
- [Model router/API comparative research](../.agents/prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md)
- [Original source atlas](../.agents/prior-art/RESEARCH_SOURCE_ATLAS.md)

**License/attribution policy:** reference content is cited and analyzed, not copied into executable source. Actual use of third-party code, compiler/runtime programs, pretrained weights, model assets and datasets must be separately reviewed at an exact revision/file and comply with the user's Rust source-ownership policy. Patent and trademark questions are independent of repository root licenses.

**Literature limitations:** this is a selective engineering bibliography, not an exhaustive survey or a claim that the combined system is the first implementation of a technique. The underlying architecture is a proposed independent design and all performance hypotheses require reproduction.

````

## Source: docs/WHITEPAPER.md

Original UTF-8 bytes: 45354; SHA256 `3e54d96b6004c80c694f0d43bf0c40f902b2ad0ec3d943bc2f54a15805e17559`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity
## A composable execution architecture for AI models, heterogeneous accelerators, and distributed state

**Technical whitepaper · Design proposal · Version 0.1 · 12 October 2026**

**Status:** proposed architecture, not a report of an implemented or experimentally validated runtime. The concrete engineering contracts are specified in the accompanying [Engineering Specification](ENGINEERING_SPECIFICATION.md). The source-linked bibliography is in [References](REFERENCES.md).

---

## Abstract

AI execution systems increasingly optimize different layers of the same problem: model operator semantics, low-bit arithmetic, memory residency, batching, compilation, accelerator placement, state reuse, distributed processing and user-facing serving. Many optimizations are valuable only for particular model architectures, shapes, hardware or request regimes. Combining them often requires parallel serving engines, duplicated state managers, incompatible device runtimes or model-specific integrations. Generality, meanwhile, can impose unnecessary dispatch, conversion and synchronization costs.

**Zerogravity proposes a single Rust-owned, composition-oriented AI execution architecture** that separates mathematical semantics, task control, physical implementation, resource residency and session state. An immutable SemanticProgram expresses model computations and versioned effects; a compiled ExecutableBundle contains closed-region implementations and asynchronous resource commands. One RegionExecutionProvider boundary accommodates direct kernels, captured graphs, ahead-of-time accelerator graphs and statically scheduled fabric programs. A small TaskDriver controls different AI tasks without imposing an autoregressive token loop on encoders, speech, diffusion or future training extensions. A per-session coordinator publishes one consistent logical StateRoot only after every required state domain and physical event is prepared. A resource authority admits model, task and device memory under real physical budgets, while fast device-local pools avoid a central per-kernel allocation bottleneck.

The same execution spine also supports independently placed compute, model parameters, mutable session state and immutable caches across local or authenticated remote machines. Built-in multi-model routing separates model identity from warm physical replicas, enabling one-command local use and a versioned OpenAI-compatible API subset without another inference engine.

The proposal targets a measurable Pareto frontier of **semantic fidelity, latency, throughput, memory efficiency, energy, developer effort and composability**. It does **not** claim unprecedented performance, universal model support or functional accelerator backends today. We present the architecture, invariants, trade-offs, falsifiable hypotheses and staged validation needed to determine whether it can deliver those goals.

**Keywords:** AI inference; semantic compilation; model serving; operator specialization; heterogeneous accelerators; asynchronous execution; KV and recurrent state; distributed placement; model routing; Rust.

---

## Contents

- [1. Introduction: why a unified execution system?](#1-introduction-why-a-unified-execution-system)
- [2. Architecture principles](#2-architecture-principles)
- [3. The minimal execution spine](#3-the-minimal-execution-spine)
- [4. AI tasks as distinct control policies over one executor](#4-ai-tasks-as-distinct-control-policies-over-one-executor)
- [5. State consistency is a primary execution contract](#5-state-consistency-is-a-primary-execution-contract)
- [6. Resource ownership and efficient execution](#6-resource-ownership-and-efficient-execution)
- [7. Hardware universality without pretending devices are equivalent](#7-hardware-universality-without-pretending-devices-are-equivalent)
- [8. Distributed execution: place computation, parameters, state and caches independently](#8-distributed-execution-place-computation-parameters-state-and-caches-independently)
- [9. Built-in multi-model service and local UX](#9-built-in-multi-model-service-and-local-ux)
- [10. Quality and security model](#10-quality-and-security-model)
- [11. A falsifiable evaluation program](#11-a-falsifiable-evaluation-program)
- [12. Implementation and evolution strategy](#12-implementation-and-evolution-strategy)
- [13. Limitations and open engineering questions](#13-limitations-and-open-engineering-questions)
- [14. Conclusion](#14-conclusion)
- [References and reproducibility](#references-and-reproducibility)

---

## 1. Introduction: why a unified execution system?

A contemporary AI model may contain dense projections, multiple attention varieties, conditional experts, learned state compression, recurrent matrices, audio/vision encoders and external lookup structures. Its request may require millions of relatively regular prefill operations followed by many latency-sensitive incremental steps. Another model may execute an encoder pass, diffusion iteration or vocoder stage instead of autoregressive text generation.

Hardware diversity makes the challenge harder. A consumer APU shares physical RAM between CPU and GPU; a discrete accelerator has its own memory and expensive interconnect; a rack of GPUs coordinates collectives; a TPU-like or wafer-scale system compiles much larger regions into device-owned programs. A second machine might offer capacity but be separated by slow or relayed network links. A model may need dense compute on one processor while its attention state or sparse experts live elsewhere.

Existing research demonstrates benefits of attention tiling [R02](REFERENCES.md#r02), paged state management [R01](REFERENCES.md#r01), prefix reuse [R04](REFERENCES.md#r04), speculative execution [R03](REFERENCES.md#r03), compiler search [R07](REFERENCES.md#r07), phase disaggregation [R05](REFERENCES.md#r05) and remote tensor transfer [R14](REFERENCES.md#r14). **These are prior-art mechanisms, not original inventions claimed here.** Zerogravity asks a systems question: can their useful *classes of optimization* coexist under one set of mathematical and state/resource contracts, without paying the cost of a generalized interpreted workflow engine at execution time?

The resulting design has two outward-facing requirements. At the developer level, new models, mathematical operators, kernels, device types, media tasks and topologies should be added through narrow interfaces. At the user level, a supported local model should be usable with a single command, and many named models should be served through one predictable interface, with complex details visible only when requested.

### 1.1 Problem statement

We seek a runtime architecture capable of minimizing cost for a supported task over a given physical topology, subject to a correctness and security contract:

```text
Select an admissible execution plan P for model M, task T and topology H
that optimizes latency, useful throughput, memory and energy,
subject to:
  exact registered model/operator/state semantics;
  caller-selected numerical and sampling policy;
  physical resource capacity, data access and device qualification;
  valid state transitions, cancellation and output ordering;
  bounded compilation/runtime overhead and recovery behavior.
```

No single plan is optimal for all inputs. A speed-oriented GPU kernel may be slower at decode batch size one than at prefill. A distributed model might fit in aggregate memory but lose interactive speed to network round trips. The goal is **plan selection and specialization under explicit truth constraints**, not a universal assertion that more parallelism always helps.

### 1.2 Scope

**In scope, architecturally:** typed model semantics, source validation, reference execution, specialized compiled regions, task-specific loops, CPU/GPU/NPU and graph accelerators, multi-model routing, stateful serving, remote execution and storage, authenticated network transport, and resource/quality governance.

**In scope as later optional extensions:** training/fine-tuning through versioned parameter/gradient tasks, additional accelerator architectures, large cluster orchestration, specialized media pipelines and broader API standards.

**Not implied:** execution of arbitrary unknown checkpoint architectures, automatic compilation of proprietary device kernels in Rust, compatibility with all vendor cloud models, unrestricted agent tool execution, or demonstrated performance leadership.

---

## 2. Architecture principles

The design is guided by seven constraints.

**Axiom A — mathematical semantics precede hardware.** A trained operator must have an exact registered meaning, shape, position convention, source quantization and effect contract. A file parser recognizing a tensor type does not make an unknown model executable.

**Axiom B — task control is not model math.** Sampling one token, scoring a document, transcribing audio and iterating a denoiser have different stopping conditions, state frontiers and output validity. A shared executor should not require a fake next-token interface for every task.

**Axiom C — specialization replaces closed regions, not entire engines.** A provider may implement a GEMM tile, a fused layer, a model stage or a whole compiled graph if its externally visible values/effects match the original semantic region. It cannot introduce an independent model registry, session writer, allocator or HTTP server.

**Axiom D — state publication has one authority.** Workers can execute and prepare provisional state; a TaskCoordinator publishes a new logical session root only after the required domains and events agree. Output becomes visible only after the corresponding root publication.

**Axiom E — resources are physical and time-dependent.** Memory addresses are not necessarily free RAM, GPU allocations have asynchronous lifetimes, and data resident on another machine is not zero-cost local state. Admission and device-local allocation are distinct responsibilities.

**Axiom F — distributed placement is optional, measured and trusted.** More devices and machines can increase capacity without increasing single-request speed. Remote execution is chosen only when the workload, network, source rights and recovery contract permit it; a private tunnel is not equivalent to RDMA.

**Axiom G — simple experience, explainable decisions.** Users can run or serve a named local model without learning a compiler IR, manually distributing layers or starting a cluster. Every meaningful nonsecurity configuration can be examined and overridden at a valid lifecycle boundary.

These principles yield a specific thesis: **composability is achieved by fixing what each component may own, not by introducing a plugin abstraction for every internal implementation detail.**

---

## 3. The minimal execution spine

Figure 1 summarizes the proposed system. Arrows represent logical lowering and control; the diagram does not imply a host round trip for every operator or token.

```text
  Local CLI / Rust SDK / API
              |
  Request validation + ModelCatalog / exact alias / ModelRouter
              |
  Artifact / ModelDefinition ---------- Trusted TaskDriver
              |                               |
              +------ SemanticProgram --------+
                       (SIR: math + state effects)
                            |
             legal region transforms, provider selection
                     (temporary compiler views)
                            |
                  ExecutableBundle
    (compiled kernels/graphs + async commands + shape guards)
                            |
         Topology / PlacementPlan / ResourceArbiter
          (coarse admissions, local device subpools)
                            |
          TaskCoordinator / StepPlan / RegionWorkers
          (GPU, CPU, NPU, AOT graph, network peers)
                            |
       prepared domain versions + true ready events
                            |
                 one PublishRoot
                            |
       committed typed outputs + ModeReceipt
```

*Figure 1 — Zerogravity's proposed architectural ownership and compilation flow.*

### 3.1 Two enduring representations

The first representation, **SemanticProgram**, records *what* the model computes: typed tensor and structured values, precise mathematical operators, dimensions, positions, numerical policy and explicit state effects. It supports a reference evaluator and is the basis for correctness tests.

The second, **ExecutableBundle**, records *how and where* a legal closed semantic region executes: prepared kernel or accelerator program handles, shape guards, resource lifetimes, copies, events, collectives and allowed fallbacks. Drivers and topology may require several compiled variants; those are managed as bounded caches.

The conceptual optimizer stages remain useful but need not be separate enduring formats:
- **Region transformation** finds legal fusion, tiling, layout changes, expert grouping or independent output-row partitions.
- **Kernel/program selection** checks shape, precision, driver, ISA, topology, state and evidence for eligible implementations.
- **Physical command planning** lays out asynchronous copies, reservations, submissions, producer-consumer fences and completion events *within* the compiled bundle.

This separation draws general inspiration from extensible compiler research [R06](REFERENCES.md#r06), explicit asynchronous resource modeling [R13](REFERENCES.md#r13) and versioned semantic operations [R12](REFERENCES.md#r12). It does not require embedding an external compiler runtime. The concrete design intent is to remove interpreted semantic graph traversal from latency-sensitive steady-state execution.

### 3.2 Closed-region substitution

A **closed region** has declared logical inputs, outputs, mutable effects, numerical conditions and a compatible fallback boundary. The compiler can ask several implementation providers to prepare it. One may supply a portable CPU implementation; another may supply a fused GPU graph; another may consume a whole model stage in an AOT accelerator compiler.

An aggressively optimized provider is eligible only if it preserves the semantic contract, accepts its shape/dtype/quant, has a valid state bridge and resource plan, and meets safety/quality policy. Some alternative implementations produce slightly different floating-point reductions; equivalence of *mathematical operator* does not automatically mean bit-identical output across hardware. Numerically different but qualified execution is separate from changing the trained model itself.

**Fallback is a composability test.** Disabling a specialized implementation must not require a different model server. If physical state layouts are incompatible, a fallback may require snapshot conversion or replay at a committed boundary rather than an unsafe pointer cast.

### 3.3 Shape-bounded compilation

Accelerator code often specializes batch size, sequence length, quant block, tensor strides, route capacity and attention regime. ExecutableBundle therefore carries shape predicates and bands, not a new kernel for every observed context length. On a guard miss, the runtime selects a prepared generic region, compiles a legal variant outside the latency-critical path when allowed, or reports a precise unsupported shape. It never launches code on an unvalidated shape.

**Prediction:** a bounded set of common shape regimes plus direct device handles can reduce compilation churn and CPU dispatch overhead while retaining rare-shape correctness. This must be measured in a real model, not inferred from the representation alone.

---

## 4. AI tasks as distinct control policies over one executor

A **TaskDriver** owns coarse, trusted task control: which stage executes next, the loop boundary, an output unit, the currently committed frontier and cancellation. It does not implement a general permissionless workflow language.

For language generation, it renders and validates prompt tokens, performs prefill, then iterates decode or target-verified speculative steps. For embeddings it encodes and pools; for speech recognition it handles audio windows and transcript boundaries; for TTS it chains acoustic and vocoder stages; for diffusion it advances latent/timestep state until a complete output can be decoded. Video and multimodal tasks may interleave stages, maintaining their own temporal validity.

A task may use multiple different models, such as encoder plus reranker or target plus draft. Each logical model remains separately identified and admitted. **Multiple models in a task are not replicas of one model.**

**Multimodal model-family stress test (v5 design proposal, 14 October 2026):** The fundamental ownership contracts above accommodate text, speech, image, video, structured decisions and compound models, but the v4 prose alone does not fully specify their execution semantics. A proposed additive contract introduces typed media values and rational timebases, source-verified component assemblies, codec/quantization transforms, non-token decision heads, denoising/flow iteration, and multi-track synchronized StateRoot frontiers. These prevent incompatible assumptions such as assuming a large image-generation checkpoint's generation is either purely AR or purely DiT without auditing its checkpoint-specific source path, mapping CTC/Transducer speech to a text token loop, or treating synchronized audiovisual generation as independent video plus TTS. The same SemanticProgram, ExecutableBundle, TaskDriver and sole PublishRoot authority remain; the execution spine is **not** being replaced. See the [proposed unified multimodal execution contract](../.agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md) and [author-source model research atlas](../.agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md). Native execution of these additional named checkpoints is not implemented.

The TaskDriver calls compiled model regions through the common device/resource system. Where correct, a **device-native StepProgram** can perform multiple causally dependent operations internally—including routing, sampling or verification—before host inspection, avoiding a mandatory CPU readback at every token. This cannot be promised for every model, grammar or media loop. Unsupported modes retain a correct host-controlled path.

Externally authorized retrieval, tools and multi-agent workflows belong to an **application orchestration boundary above** TaskRequest, not to tensor operators or a hidden shell inside the TaskDriver. This preserves inference-engine safety and keeps the mandatory runtime small.

A future optional **GradientProgram** could express backward operators and transactional ParameterVersion updates. Ordinary inference sessions would remain bound to an immutable checkpoint/adapter generation. This is an extensibility seam, **not** a claim to support training today.

---

## 5. State consistency is a primary execution contract

Many serving optimizations depend on modifying state: KV pages, sliding caches, compressed attention, recurrent matrices, sparse indexers, token-history lookups, sampler/RNG and grammar state. Treating these as implicit mutable buffers makes speculative rejection, prefix forks, cancellation, migration and distributed execution difficult to reason about.

Zerogravity proposes a **logical StateRoot** containing a versioned reference to every required state domain at the same committed task frontier.

### 5.1 Prepare, then publish once

The transition is:

```text
Begin(epoch, committed_root)
  -> execute into private / COW / rollback-safe provisional state
  -> Prepare each required domain and validate common frontier
  -> wait for actual device, copy, IO and peer-ready events
  -> TaskCoordinator checks expected old root and session epoch
  -> PublishRoot(new complete root) exactly once
  -> release committed output events in order
  -> retire old physical resources after their true last-use events
```

**No state domain or device command publishes visible session state independently.** If the second domain fails preparation, the original root remains authoritative; private mutations are aborted or reconstructed. A physical buffer may still be active after a host timeout: its lease must remain held or quarantined until safely quiescent.

This is **atomic logical visibility**, not a claim that all GPU memory writes occur atomically across devices, nor a free distributed durability guarantee. An implementation needs private/COW tails, compatible checkpoint replay or carefully proven undo operations to ensure the published root never references half-mutated state.

### 5.2 Prefix sharing and speculative verification

A prefix-cache hit is valid only if the *entire required model state* can resume from the same causal frontier under an exact source/quant/adapter/template/position/tenant/grammar identity. Matching visible text alone is insufficient. Copy-on-write snapshots allow efficient compatible forks, but cache eviction cannot invalidate a reader holding a live lease.

Speculative decoding proposes candidate tokens, checks them against the authoritative target and commits only the accepted corrected prefix. When advertised as target-distribution preserving, sampling and RNG state must follow a valid verifier algorithm [R03](REFERENCES.md#r03). Approximate alternatives require a distinct explicit quality mode; better speed from altered trained top-k or reduced indexer semantics is **a changed model**, not a transparent optimization.

### 5.3 Streaming and remote failures

Committed output carries a monotonically ordered per-session sequence. A client may receive an event and disconnect before acknowledging it: the runtime can promise ordered **attempted** delivery and an optional authenticated replay cursor, but not exactly-once reception without a separate durable protocol.

Across remote machines, workers prepare versioned state shards under coordinator and placement epochs. The session coordinator alone publishes the new root after all required participants are ready. A lost node, peer lease or old coordinator must be fenced; without durable consensus and checkpoint recovery, the safe behavior may be to stop or replay. **Automatic transparent failover is not an inherent property of a private tunnel.**

---

## 6. Resource ownership and efficient execution

Physical admission and fast allocation operate at different timescales. A global **PhysicalResourceArbiter** accounts for mandatory model weights, adapter/draft/encoder models, KV/recurrent and other mutable state, expert slots, prefix caches, compiler program residency, scratch, DMA staging, SSD/remote IO, OS headroom and multiple concurrent tasks. It decides whether a model/session/placement can be admitted and coordinates evictable capacity.

A **DeviceBackend local pool** performs frequent allocation and reuse from approved reservations with generation-tagged BufferLeases. Retire events track outstanding GPU/NPU/IO readers. A single global mutex on every matmul scratch allocation or token would defeat the high-concurrency design. Conversely, allocating directly from each backend without a shared physical ledger would overcommit a multi-model or shared-memory system.

These ideas follow the general problem of dynamic attention state management described in [R01](REFERENCES.md#r01), but the architecture extends state accounting to hybrid recurrent domains, multiple models and heterogeneous physical memory classes.

**Resource caveat:** no userspace process can guarantee unrelated processes will never change available OS memory. The enforceable goals are conservative admission, pressure observation, bounded buffers, safe failure and recovery—not an absolute claim that the machine can never OOM.

---

## 7. Hardware universality without pretending devices are equivalent

The provider interface adapts to the **minimum efficient execution granularity of the hardware**:

| Executable kind | Native physical unit | Typical specialization scope |
|---|---|---|
| Direct kernel | CPU/GPU/NPU kernel or tile | one op or fused region |
| Captured command graph | prepared device submissions | repeated shapes, decoder or denoiser island |
| Ahead-of-time device graph | compiler-generated closed subgraph | whole layer/model stage |
| Statically scheduled dataflow | compiler-placed on-chip program/fabric | full pipeline or partitioned model |
| Managed device session | opaque physical state under a *verified* state bridge | only qualified programs; often restricted |

A CPU/GPU kernel can run within a simple device worker. A TPU-like graph backend may compile a whole semantic stage into a program with shape guards and its own device-internal schedule [R15](REFERENCES.md#r15). A wafer-scale system may place a dataflow pipeline across on-chip compute and stream weights from elsewhere [R16](REFERENCES.md#r16). Zerogravity should not emulate every microoperation of such hardware with a host RPC or claim native kernel source access when only a vendor compiler is available.

Supported compiler and driver APIs, executable source rights, device program state snapshots, compile/upload/warmup costs, internal memory and collectives are all **capabilities to verify**, not presumed from a product name. A remote text-generation service that exposes only input/output and no trustworthy mathematical state bridge is an **external TaskServiceAdapter**, not an equivalent native implementation of arbitrary closed semantic regions.

### 7.1 Hardware topology matters

A coherent CPU/GPU workstation shares physical RAM and bandwidth [R18](REFERENCES.md#r18); a datacenter GPU uses independent HBM and fast peer links; a TPU pod uses its own interchip topology [R15](REFERENCES.md#r15). GPU virtual addressability is not necessarily spare physical memory. Numerical formats must be identified precisely: differently scaled FP4 formats or temporary low-precision compute are **not** interchangeable model storage artifacts [R17](REFERENCES.md#r17).

The system therefore treats compute domains, memory tiers, links, compiler domains, driver versions and power/safety constraints as separate topology properties. Backend selection must be based on actual qualified capabilities and measured task costs.

---

## 8. Distributed execution: place computation, parameters, state and caches independently

Most distributed systems begin with a single model partition: layers, tensors, experts or pipeline stages. Zerogravity proposes a more general **PlacementPlan** with four orthogonal maps:

1. **ComputePlacement** locates semantic operators, layer ranges, attention cores, FFN/experts, encoder/denoiser stages or graph programs.
2. **ParameterPlacement** locates trained dense/shared/expert weights, adapters, immutable lookup tables and packed artifacts.
3. **StatePlacement** locates each mutable session/layer KV, recurrence, indexer, grammar or media state shard under versioned leases.
4. **CachePlacement** locates optional immutable reusable prefix snapshots and derived artifacts. A presence hint does not grant authority to mutate live session state.

A **TransferPlan** joins the maps with explicit activations, state snapshots, partial reductions, copies, collectives, memory reservations and authenticated transport events. These are views within ExecutableBundle—not another durable IR or independent cluster inference engine.

### 8.1 Representative topologies

**Colocated execution** keeps layers, weights and state on one machine and remains the latency baseline.

**Layer pipeline** partitions model layer groups over machines, typically with each layer's KV/recurrent state colocated with that layer's computation. It transfers activations at stage boundaries rather than the full cache every token. It may enable a model too large for one node, while adding serial per-token hop latency.

**Attention/FFN disaggregation** keeps attention and its potentially large KV state on a memory-rich worker, with projections/experts on another compute-rich worker. It exchanges compact queries, hidden activations and results; this can be better than moving large caches but may suffer repeated cross-network calls. Operator-level and P/D disaggregation are extensively explored in prior work [R05](REFERENCES.md#r05) [R10](REFERENCES.md#r10).

**Remote prefix cache** hydrates one immutable compatible snapshot from a cache worker, then computes locally. Recompute may be cheaper than transferring gigabytes over a constrained link.

**Sequence-sharded attention** places disjoint token-range KV shards on attention workers and merges partial softmax outputs. For ordinary softmax attention each shard returns a local maximum score, shifted exponential denominator and weighted-value numerator; global rescaling reconstructs the attention result in real arithmetic. Positions, causal masks, finite precision, global selection and state updates must be correct. The same algebra does **not** automatically partition recurrent or learned sparse indexer state [R08](REFERENCES.md#r08) [R10](REFERENCES.md#r10).

**Prefill/decode separation** places prompt processing and token generation on different workers, transferring the *full compatible hybrid state* at a safe committed frontier, not merely plain transformer KV [R05](REFERENCES.md#r05).

**Tensor/head/expert parallelism** uses collective communication or compact routed rows on physically suitable links. A bandwidth-starved tunnel usually cannot sustain per-layer allreduce at datacenter-like performance.

### 8.2 Network and trust

An opt-in **PeerAgent** on another authorized machine reuses the same compiled-region executor and physical admission system; it is not a separate public model server. The control plane handles node identity, capabilities, authorization, placement epochs and lease/failure events. The data plane chooses actual eligible local DMA/RDMA or bounded reliable encrypted transfers, with source/destination leases and flow control [R14](REFERENCES.md#r14).

An encrypted overlay can connect machines through direct peers or relays, but does not create GPU-RDMA, universal coherence or low-latency bandwidth. Effective link rate, RTT/P95 jitter, copy/crypto costs, loss and egress are inputs to placement [R22](REFERENCES.md#r22). Private prompt, weight and state export requires explicit user/tenant/model rights; the existence of a peer does not grant permission to send it all model inputs.

Relocation takes place only at a compatible committed TaskFrontier: reserve and warm the destination, transfer/validate all required state, fence the old writer and publish a new placement epoch. A lost coordinator or network partition may require fail-stop/replay rather than magical live failover.

### 8.3 The performance condition for distribution

Distribution is worthwhile only when increased compute capacity, memory availability or overlapped work outweighs additional communication and recovery costs. A useful *rough model*, not a scheduler formula, is:

```text
T_task(P) ≈ longest critical dependency path of:
    region compute
  + transfers (effective bytes / measured bandwidth)
  + link RTT and synchronization
  + queue, cache hydrate and program warmup
  + state publication and output delivery
subject to correctness, physical capacity, trust and latency constraints.
```

Legal parallel execution reduces the critical path only where independent device/transport work overlaps. A large activation transferred once at a layer boundary is different from a remote read of growing KV **on every token**. A slow-link distributed plan can be a useful *capacity mode* even when single-request generation speed decreases. This is an honest and important product distinction.

---

## 9. Built-in multi-model service and local UX

Users should not have to operate a cluster scheduler to use one local model. The proposed terminal surface begins with:

```sh
zerogravity run ./model.gguf
zerogravity run coding
zerogravity serve
zerogravity models list
zerogravity inspect coding --explain-plan
```

*Illustrative, not implemented or a frozen public interface.*

The built-in model router manages **four distinct identities**: an immutable RegisteredModel checkpoint/quant/adapter; a versioned alias/profile carrying a friendly name and configuration; one or more physically admitted ResidentModel replicas/deployments; and per-request mutable TaskSessions/StateRoots.

A catalog may contain many model definitions without allocating their tensors. Concurrent residency is bounded by real physical resource capacity. The manager deduplicates cold loads, reserves before loading, warms an executable when possible, serves other resident models concurrently and releases idle instances only after active sessions, device events and state leases are safe.

**Exact-model routing is the default.** Requesting a named model never silently selects another checkpoint, quantization or adapter because the original is cold or overloaded. Balancing among *equivalent* replicas is a different operation, preserving session/cache affinity and the selected numerical contract. A policy alias may intentionally select a different model only when explicitly configured and disclosed. One sharded deployment spanning several machines is not the same as several replicas.

The server exposes one model-aware API surface over the same executor as the CLI. An initial **OpenAI-compatible subset** would include model listing and chat completions with correct request/stream/finish/error semantics; embeddings need a validated encoder TaskDriver; Responses has distinct event and state-store requirements [R21](REFERENCES.md#r21). A local endpoint should not claim all cloud-specific features, built-in tools or conversation retention merely because a path exists.

Configuration is progressive: safe automatic defaults for `run`, model management and inspection for daily use, then typed global/model/alias/CLI/request profiles, placement, routing, numeric, hardware and security controls when needed. Hard model semantics, physical safety and tenant authorization always take precedence over ordinary overrides.

---

## 10. Quality and security model

The architecture distinguishes four kinds of claims:

| Claim | Required evidence |
|---|---|
| **Semantic equivalence** | exact source model formula, shapes, positions, attention/router and all state effects |
| **Numerical agreement** | kernel/source dtype and quant conditions, reference tensors/logits, justified error tolerances |
| **Task quality** | independent model/task fixtures under realistic prompts, contexts, audio, image/video or scores |
| **Sampling/distribution fidelity** | exact target verifier algorithm where promised; RNG/grammar state and rejected branch rollback |

A faster model with reduced expert top-k, altered sparse selection or new lossy checkpoint is not the same model merely because its responses look plausible. Alternative numeric quality modes must be visible and testable.

Security follows the same ownership architecture. Checkpoint files are untrusted data, not arbitrary executable code. Model-generated tool calls are output data, not shell permissions. Compiled device programs require source/toolchain trust. Remote workers require explicit enrollment and data-access rights. Local API listeners default to loopback; external exposure needs intentional authentication/TLS and restricted model-management controls. Prefix caches and metrics must not leak tenant context or hidden model inputs.

**No absolute hardware fault prevention is claimed.** Unsafe shared power/fabric configurations and unknown device program sources must be disabled until specifically qualified; device DMA can outlive a host timeout and requires proper quarantine or reset.

---

## 11. A falsifiable evaluation program

The architecture is a proposal. Demonstrating its value requires an implementation and independent measurements. In particular, **composability is a testable property**, not just a diagram.

### 11.1 Functional milestones

1. **Reference semantics:** parse one real trusted checkpoint and tokenizer; validate exact target logits/state and produce a correct CPU result.
2. **Compiled physical execution:** run the same model on a qualified GPU using a prepared executable, safe buffer/event lifetime and a forced generic fallback.
3. **Atomic state:** inject a failure in one of several state domains and prove no partially published root or speculative output.
4. **Extension drill:** add a second model operator and a new kernel provider without changing unrelated HTTP, model registry, allocator or session owner.
5. **Non-token task:** run an encoder or mocked iterative media driver through the same executor without fake autoregressive state.
6. **Multi-model product:** one-command direct run, same-engine router server, exact aliases, safe concurrent residency/replica balancing and official client compatibility tests.
7. **Distribution:** reproduce a single model across two authorized workers, per-layer state colocation, safe transfer/migration and peer failure behavior; separately test optional remote cache and attention partition.

### 11.2 Performance measurements

For each supported model/artifact/quant/device/task combination, capture:

- time to first **valid committed** output, total task completion and P50/P95/P99 tail latencies;
- useful throughput at concurrency 1, 2, 4, 8, 16 and beyond, with native batching and fairness;
- program compilation/warmup and cold versus warm model load;
- kernel/device occupancy, source/numeric agreement and device/host synchronization overhead;
- actual peak physical DRAM/VRAM/HBM, model+KV+scratch/graph cache, cache hit/miss, SSD/network bytes;
- energy, CPU host overhead, thermal/clock conditions, SDK/compiler/driver versions and exact topology;
- load-balancing behavior for multiple models, replicas, output streaming and queue contention;
- network direct, encrypted direct and relayed path performance, capacity gains **separate from** generation speed.

A benchmark claim should be associated with exact source checkpoint/hash, model/adapter, prompt/context, sampler/output type, execution plan, driver/hardware and full task quality. Baselines must use matched semantics. Model-specific work may outperform a generic path and still lose to an external specialist; publishing both positive and negative outcomes is essential.

### 11.3 Explicit hypotheses rather than results

**H-A:** prepared, device-native execution reduces CPU synchronization and dispatch overhead for eligible models.

**H-B:** closed-region substitution allows extreme kernel performance without additional model-serving engines.

**H-C:** coordinated memory admission plus device-local allocation improves multi-model concurrency and reduces preventable resource errors.

**H-D:** independent compute/state/cache placement enables some previously infeasible model/context sizes and improves SLOs on suitable fast networks.

**H-E:** a small, versioned task API can add another model, accelerator or task with limited change radius and without breaking prior workloads.

Each hypothesis can fail under particular hardware or models. Success is determined by reproduced end-to-end results, not chosen vocabulary or source-author benchmark numbers.

---

## 12. Implementation and evolution strategy

A usable system should precede a large framework. The initial implementation should begin as **one Rust library and one CLI binary**, split into additional crates only where target compiler, security or build constraints justify it. The first vertical slice is a real supported model, trusted artifact parser, CPU reference, minimal semantic/effect representation, one task driver, simple legal region/provider, asynchronous buffer/command checker, one physical GPU and a committed output.

The second slice proves **architecture reuse**: another model/op, a missing fast-kernel fallback, a non-token task and controlled physical resource sharing. Multi-model routing and one loopback API then use the same engine; distributed execution progresses from a simulated remote worker and an authorized two-node layer split to more demanding attention/state and collective topologies.

Full TPU/static-dataflow, advanced media pipelines, high-availability clustering, native accelerator compiler toolchains and gradient training remain **separately qualified extensions**. They should not inflate the startup or runtime cost of a single CPU/GPU text request.

The [Engineering Specification](ENGINEERING_SPECIFICATION.md) states exact component contracts, model/task/device/placement interfaces and failure tests. The [Implementation Playbook](../.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md) organizes the preserved 24-phase roadmap into bounded, independently verifiable work packages. Prior-art licenses and historical decisions remain auditable.

---

## 13. Limitations and open engineering questions

This proposal cannot guarantee universal inference superiority. Known risks include low-bit GPU/accelerator code-generation efficiency, dynamic shape compile cost, numerical behavior of fused kernels, high-overhead abstraction interfaces, failure recovery of opaque device-managed state, transient pressure in unified-memory systems, poor network/tunnel links, and inconsistent model-family/checkpoint semantics. A narrow hand-tuned implementation can outperform a general system on its favorite shape; our claim is that such an implementation *should be integrable as a candidate* without duplicating the rest of the serving runtime.

Some device vendors expose a compatible graph compiler or an opaque model API but no independently implementable Rust device kernel target. A Rust host cannot honestly claim a fully Rust-authored native accelerator compiler on that basis. Storage/snapshot and model weight redistribution rights can constrain distributed placement. Network-exactly-once results and automatic coordinator failover require additional durable protocols and are not included by default.

Training/fine-tuning is intentionally a future parameter-version/gradient task seam rather than a hidden default requirement. The project may later decide to specialize for selected AI modalities or accelerator families based on actual demand. Source registry choices, stable API levels, ABI/compiler policy, default model, model downloads, security deployment and service configuration are not all frozen.

Finally, publication of a design is **not** a substitute for implementation artifacts. This whitepaper is ready for technical review as a proposed system design; any future experimental-results edition must add real measurements, reproducible sources and independent quality evaluations.

---

## 14. Conclusion

Zerogravity proposes a **single composable AI execution spine** that spans model semantics, optimized regions, accelerator programs, asynchronous resources, state transactions, model serving and distributed placement. The organizing decision is to keep ownership narrow: one canonical mathematical program, one prepared physical execution plan, task-specific control, authoritative session-state publication, global resource admission with local fast allocators, and interchangeable qualified providers.

A new mathematical model changes its model definition and genuinely novel operators. A new accelerator changes region lowering and its physical capability adapter. A new task supplies a task driver and validated output state. A new placement changes a costed physical plan and transport. A new API changes the typed ingress/egress adapter. **None should require a new model-serving engine.**

The corresponding user promise is equally important: a supported model can be started with one command, while every meaningful specialization and distributed decision can be inspected and controlled when needed.

This is a concrete and falsifiable architecture for evolving AI execution systems. Whether it becomes a leading runtime will depend on carefully implemented kernels, trusted state/resource mechanics, benchmark discipline and the practical cost of adding the next generation of models and hardware.

---

## References and reproducibility

Full bibliography: [References](REFERENCES.md). The academic and primary-source material covers memory-paged serving, IO-aware attention, target-authoritative speculation, structured serving, phase disaggregation, typed compilers, accelerator and transport architectures, local model UX and API contracts. **Names and code licenses are retained solely as attributed prior art; zerogravity's component names and executable design are independent.**

**Documentation release:** architecture v4.3, manuscript version 0.1 (12 October 2026). **Implementation status:** a packaging-only Rust binary and release/CI scaffolding partially advance P00; the **inference engine and P01–P23 remain NOT_STARTED**, and no user hardware/model/tunnel benchmark is represented as performed here. Figures are architectural illustrations, not screenshots of running software.

````

## Source: AGENTS.md

Original UTF-8 bytes: 5949; SHA256 `86a2089819974a901bedf812ec6e911c2fb4a9b7fa73cc189ed56390771e3402`.
Resolve embedded relative paths against the original path, not this volume.

````text
# AGENTS.md

<!-- BEGIN common -->

## Shared conventions

These are byte-identical across all abird repositories. Repo-specific rules
follow the markers below.

- **Source of truth.** Live source, tests and executed results outrank dated
  prose. Start at `.agents/plans/README.md` (overview + spec), then work from
  `.agents/plans/TODO.md` (the queue).
- **XDG paths.** Where the project owns user config, data, cache or state,
  resolve it under `$XDG_*_HOME/abird/<project>` — an absolute XDG base, falling
  back to `HOME` when unset. No legacy or alternate root.
- **Shared `.gitignore`.** The block at the top of `.gitignore` is identical
  across repos; repo-specific entries follow it. Keep it mechanical.
- **Preserve other work.** Never revert or overwrite changes you did not make.
- **Owned changes only.** Commit only the paths you changed for the request.
  Before committing, inspect `git diff --cached --stat`; if it lists files you
  did not touch, stop and ask — unless the user explicitly asked you to include
  them.
- **Ownership.** A plan directory belongs to the `@owner` on the queue line, and
  the coordinator alone reconciles `.agents/plans/TODO.md`; do not edit another
  owner's plan or worktree unless the user explicitly asks.
- **Clean pushes.** Do not push from a dirty worktree unless the user explicitly
  asked to include those changes; otherwise commit/stash your own change first,
  or push from a clean temporary worktree.
- **Terse writing.** Keep code comments and docs succinct. Comments need not be
  full sentences; prefer symbols and fragments unless the user asked for an
  explanation or the topic is design-level or otherwise complex. Prefer bullets
  and engineering focus over prose in other docs.
- **Terse output.** Keep replies short and on point, preferably bulleted; avoid
  prose unless the user asked for an explanation. Be terse without dropping
  detail.
- **Git safety.** Commits are signed; never disable signing. Never push,
  force-push, tag, publish, deploy, or run destructive operations — history
  rewrite, `reset --hard`, branch deletion, or touching another agent's staged
  or uncommitted work — without explicit user confirmation for each action. A
  request to "commit" or "apply" is not authorization to push.

<!-- END common -->

<!-- BEGIN repo-specific -->

## Repository

Start with [.agents/plans/TODO.md](.agents/plans/TODO.md), choose the relevant
plan, and read its `TODO.md` plus the specifications it links. These TODO files
replace the old handoffs, status ledgers, evidence workflow and SHA inventories.
Older documents remain design/history references, not extra tracking duties.

Read [.agents/plans/README.md](.agents/plans/README.md) before opening a plan —
it is the rulebook for the layout. A plan is `.agents/plans/<plan>/` with an
optional `README.md` (what it is and why it exists, plus design, decisions and
references sections, opened with its `Author:`/`Co-authors:`) and a required,
extensive `TODO.md` whose completed work is grouped under the commit that landed
it and whose open work sits under `## Uncommitted`.

- **XDG project:** `abird/zerogravity`.

## Plans and parallel work

- `.agents/plans/TODO.md`: one checkbox and link per plan. Check it only when
  the whole plan is implemented and tested; partial milestones stay in its file.
- `.agents/runs/<plan>-<session-or-agent-id>/TODO.md`: each agent's own working
  checklist. Record its parent plan, owned scope, tests, results, failures and
  unfinished items, so another agent can continue an interrupted run. Do not
  edit another agent's checklist.
- The coordinator alone reconciles the shared plan: one writer, one serialized
  atomic update; parallel appends must not overwrite each other. A finished run
  file can be removed once its items are in the main plan; keep an interrupted
  run file until its work is recovered.
- Commit verified implementation, tests and plan updates together with explicit
  paths. After each commit, move completed items under that commit and carry
  unfinished items forward; do not create commits merely to record their own
  IDs. On a finished plan, check its one line in the plan index.
- When a plan grows long, move only completed sections into
  `.agents/plans/<plan>/archive/<date-or-milestone>/TODO.md` and link them from
  the live TODO.

No mandatory run receipts, source snapshots, per-file hashes, archives or
generated handoffs. Actual model/content hashes and numerical acceptance limits
remain part of the implementation wherever needed.

## Implementation boundaries

Preserve one Rust mathematical engine, immutable source identity, independent
StateRoot ownership, the physical ResourceArbiter, fair workers and actual
completion/quarantine. Run checks appropriate to the change and the complete
`scripts/check.sh` gate before accepting an implementation phase. Use
`CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
for actual user/network isolation with already approved inputs. GPU/model
changes also require their relevant real source, state, cancellation and
physical tests; compilation or mocks do not establish inference support. Inspect
changed code and failed/interrupted tests when resuming; Git IDs are source
references.

When merging source from a worktree, copy bytes with fresh modification times.
Preserving older timestamps can make Cargo reuse a stale library; unchanged
bytes may need a narrow timestamp refresh before rerunning the same affected
checks.

The root README is user-authored: leave its bytes unchanged. Do not download
models/SDKs or change device privileges or recovery without applicable user
authorization. The local Renoir/gfx90c AMD launch hold remains. Use subagents
for independent implementation/review/test work; the main agent coordinates,
reconciles, decides and merges. Preserve unrelated user work.

<!-- END repo-specific -->

````

## Source: README.md

Original UTF-8 bytes: 11745; SHA256 `6c00873ec0ac2c59bc73703b55a9bfa4d93703334a03fa16bcac2d204cda5b15`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity

**The blazingly fast AI execution fabric - built for composable multi-modal inference specialization at every layer, from individual kernels to distributed systems**

A single binary Rust execution fabric for **text, image, video, audio, real-time, and combined multimodal inference**. Model-specific mathematics, hardware kernels, memory placement, scheduling, and serving compose under one engine—not a separate runtime for every model or modality.

Run _models locally_ without compromise - *one engine to run them all*.

> **From scratch, informed by prior art.** No source from any prior-art project
> was copied or linked unless it is directly vendored (for example, CubeCL
> (vendored)) and therefore retains its upstream license. Zerogravity is an
> independent Rust rebuild that learns from the concepts, research and
> engineering of the projects acknowledged under [Related projects](#related-projects).

## Intro

- **Any model architecture.** Transformers, hybrid recurrent models, MoE, diffusion/flow, speech codecs, direct decision heads, and model assemblies—each with verified semantics.
- **Specialization at every layer.** Source-aware quantization, fused operators, ISA-specific kernels, batching, speculative decoding, and distributed execution.
- **One memory fabric.** GPU VRAM, unified memory, system RAM, and SSD-streamed tensors share a logical model identity and safe physical leases.
- **High-concurrency serving.** Continuous batching, chunked prefill, paged/hybrid state, prefix reuse, multi-model routing, and latency-aware admission.
- **Joint optimization.** Choose compatible batch, draft, kernel, residency, and placement strategies together—not as isolated optimizations.
- **Correct by construction.** Source-verified mathematics, explicit state effects, actual device completion, and one authoritative session-state publisher.
- **Simple interface.** Local CLI and Rust API, with HTTP/OpenAI-compatible, WebSocket, WebRTC, and distributed serving in the architecture.

## Quick start

Build on Linux with the Nix development environment:

~~~sh
nix develop
cargo build --release --locked
~~~

Run a local **Qwen3.5 0.8B or 2B GGUF** with its matching `config.json` and `tokenizer.json`:

~~~sh
./target/release/zerogravity inspect /path/to/model.gguf
./target/release/zerogravity run /path/to/model.gguf \
  --prompt "Explain Rust ownership in three sentences" --max-tokens 32
~~~

Run the repository's quality checks:

~~~sh
./scripts/check.sh
~~~

Only local files are used; no model weights are automatically downloaded or executed as third-party model code. See [Implementation status](#implementation-status-and-roadmap) for the currently supported execution paths.

## Architecture

~~~mermaid
flowchart TB
    I["CLI · SDK · HTTP · WebSocket · WebRTC"]
    R["Model catalog · Router · Admission"]
    S["Verified ModelAssembly + TaskDriver"]
    P["SemanticProgram → ExecutableBundle"]
    W["CPU · GPU · NPU · Peer workers"]
    M["GPU VRAM · Unified RAM · Host RAM · SSD"]
    C["Real device events → TaskCoordinator → StateRoot"]

    I --> R --> S --> P --> W --> C
    M <--> W
    C --> I
~~~

**Four independent concerns:**

| Concern | Contract |
|---|---|
| **Model** | Verified source, tensor encodings, components, operators, and task-specific state |
| **Execution** | Compiled, guarded regions and kernels; CPU/GPU/NPU/peer placement |
| **Resources** | Shared physical admission; leased weights, state, caches, and GPU/RAM/SSD materialization |
| **Serving** | Concurrent requests, stream events, routing, QoS, and single-authority state commitment |

A model is defined once. Its **compute, immutable weights, mutable session state, and reusable caches** can be placed independently when source semantics and physical completion allow it. Fully resident tensors take the direct device path; SSD misses use bounded asynchronous materialization.

### Joint optimization

~~~mermaid
flowchart LR
    B["Batch · Prefill"]
    D["Draft · Verify"]
    K["Kernel · Quant"]
    M["Cache · Residency"]
    L["Device · Placement"]
    P["Bounded, source-qualified planner"]
    O["SLO goodput · Latency · Fairness"]

    B --> P
    D --> P
    K --> P
    M --> P
    L --> P
    P --> O
~~~

**Choose the best validated combination**, subject to exact model semantics, numerical policy, resource budgets, and deadlines:

~~~text
best_plan = argmax SLO_goodput(plan)
            over source-qualified, physically feasible plans
~~~

A larger batch can waste padded MoE rows. A longer draft can touch more experts. A smaller quantization can decode more slowly. An SSD cache hit can still be expensive if it copies full weights each step. **The best plan optimizes these interactions together**, with a simple verified fallback.

### Design decisions

- **One execution engine**, not one backend server per architecture, quantization, device, or modality.
- **One authoritative StateRoot per session**; speculative output cannot publish before target verification.
- **One physical resource authority**; buffer reuse requires real IO/device completion.
- **Composable specialization** from tensor format and fused operator through task, model, hardware, and cluster.
- **Native and hosted models are distinct**; external APIs use explicit service adapters, not counterfeit local kernels.

## Engineering

- [Architecture and subsystem specifications](.agents/docs/)
- [Joint serving and optimization contract](.agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md)
- [Multimodal and real-time architecture](.agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md)
- [Unified GPU/RAM/SSD residency](.agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md)
- [Whitepaper](docs/WHITEPAPER.md) · [Engineering specification](docs/ENGINEERING_SPECIFICATION.md)
- [Build, releases, NixOS, and previous implementation details](.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md)

## Interfaces

| Surface | Purpose |
|---|---|
| **CLI / Rust** | Local model inspection, execution, and embedded workflows |
| **HTTP / OpenAI-compatible** | Text, embeddings, model routing, tools, and structured outputs |
| **WebSocket / WebRTC** | Streaming text, speech, native duplex audio, and live multimodal sessions |
| **Media tasks** | Image, video, speech synthesis/transcription, and asynchronous generation |
| **Distributed** | Qualified compute, model weights, session state, and cache placement across devices or peers |

These are **architectural targets**, not a claim that every interface or model family is implemented today.

## Implementation status and roadmap

| Available now | In progress / planned |
|---|---|
| Native Rust **Qwen3.5 0.8B / 2B** GGUF text CPU execution and inspection | Complete native accelerator model execution and additional model families |
| Original-model CPU state, snapshots, prefill, and source-reference checks | General compiled SemanticProgram / ExecutableBundle |
| Shared immutable model loading and model catalog foundations | Continuous batching, exact speculation, and OpenAI-compatible multi-model server |
| Source-qualified **selected** Vulkan GPU projections on NVIDIA and AMD | Whole-model GPU recurrent/attention/KV, CUDA/HIP, and distributed execution |
| Bounded physical GPU admission, fair worker primitives, and real-event safety probes | First-class SSD execution, multimodal media/real-time models, and full hardware tuning |

**Qualification:** the Qwen CPU Phase 1 acceptance gate passed through 8K; 16K/32K full-model cases remain deferred. The GPU evidence covers selected original-model operations—not a complete GPU model run. vLLM-class serving and multimodal specialization remain design/implementation work, not measured product capabilities.

**Roadmap:** [phases P00–P23](.agents/plans/IMPLEMENTATION_ROADMAP.md) · [current progress](.agents/plans/CURRENT_STATUS_AND_TRACKING.md) · [acceptance and evidence](.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md)

## FAQ

**Which models does zerogravity support?** The architecture is model-family agnostic, but execution requires verified source semantics, codecs, and kernels. See [Implementation status](#implementation-status-and-roadmap) for what runs today.

**Does it require a GPU or a Python inference engine?** No. The core is native Rust and supports CPU execution; accelerator providers are optional. Zerogravity does not wrap a separate Python/C++ model-serving engine.

**Can a model use SSD when it does not fit in GPU memory?** The design supports one logical tensor across GPU, unified RAM, host memory, and asynchronously streamed SSD. Resident access remains direct; actual SSD misses have measurable latency. Native SSD execution is still on the roadmap.

**How does it differ from vLLM?** Zerogravity targets the same concurrency fundamentals while jointly specializing model semantics, kernels, quantization, state, memory tier, hardware, and serving QoS. Performance superiority must be established by comparable benchmarks, not assumed.

**Can I run several models at once?** The model catalog and shared-weight foundation exist. Full multi-model inference serving and cross-model admission are planned; registering models does not imply they all fit in physical memory.

## Related projects

Zerogravity would not exist without the open-source inference and systems
community. We thank and acknowledge the following projects as inspiration and
prior art; their ideas, research and engineering shaped this work:

- **llama.cpp** — portable GGUF execution and the dead-simple local single-binary experience.
- **vLLM** — high-throughput serving, continuous batching and paged state.
- **SGLang** — structured program execution and prefix-aware cache reuse.
- **Ollama** — one-command local model UX and resident model management.
- **gufo** — model-family specialization, accelerator kernels and multimodal tasks.
- **strata** and **maya** — extreme model specialization and VRAM→RAM→SSD expert tiering.
- **CubeCL (vendored)** — the Rust GPU kernel IR, vendored with its upstream license.
- **pi** and **dwarfstar4** — agentic engineering and extreme-model execution research.

These names are acknowledged as inspiration and interoperability references only.
No source code from any prior-art project is copied or linked unless it is
directly vendored and retains its upstream license; Zerogravity is an
independent, from-scratch rebuild.

## Contributing

We build zerogravity in the open and would love your ideas, architectural
challenges, research, and focused improvements. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the best ways to take part.

## AI disclosure

**Human-centric design. AI-centric implementation.** Zerogravity is built with
extensive AI assistance for research, code, tests, and documentation. We believe
the best software comes from human-led design and AI-assisted implementation:
humans own the architecture, priorities, and acceptance criteria, while AI helps
implement, test, and iterate. We don't hand direction to autonomous models or
claim every generated line is manually reviewed — correctness rests on explicit
designs, reproducible tests, and continuous validation.

## License

[MIT](https://github.com/abird-ai/zerogravity/blob/main/LICENSE) © 2026 Prasanna Loganathar and zerogravity contributors. Zerogravity-authored
text and software are MIT-licensed. Vendored third-party components (for example,
CubeCL) and other assets retain their respective upstream licenses. Model
weights, third-party libraries, and other assets retain their respective licenses.

````


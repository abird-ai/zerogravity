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

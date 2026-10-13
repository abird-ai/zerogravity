# v3 design evolution snapshot — superseded but preserved

**Research/history only, archived 2026-10-12 as part of a v4 architectural review.** The source documents below were normative v3, prior to addressing atomic state publication, task-driver minimality, shape specialization, host/device async input races, hierarchical resource allocations, distributed fencing and training as future optional capability. They remain available to preserve design provenance but **do not supersede** current v4 normative plans. Consult `ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md` for findings, `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md` and `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md` for resolutions.

---

# Archived v3 source 1: `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`

# Modular AI systems — original, project-neutral reference architecture

**Version 3 · 2026-10-12 · authoritative normative architecture.** This specification supersedes source-name-derived design language in historical plans. It is intentionally independent of every previous engine's internal design, licensing, exact tensor library and release cycle. **The project is zerogravity**; its architecture is defined by typed contracts and measurable properties, not external brands. External work is indexed separately in `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`.

**Status:** researched design only. The following types, code examples, APIs, kernels and benchmarks are not implemented. Preserve explicit user intent for a Rust-authored core and owned device kernels, correct general fallback and maximal performance *where measured*. Exact external driver/maths FFI, public license, release targets and GPU compiler selection remain open.

## 0. Prime design principle

> **Compose capabilities, not engines.** An AI model is a semantic program. An AI application is a typed task program composed of models and processing stages. A device is a collection of capabilities and resource costs. A kernel is one possible implementation of a semantic region. The runtime owns state and scheduling. Any specialization can replace a proven execution region without requiring a new model runtime, scheduler or service.

Three independently useful outcomes:
1. A new **model/operator family** can be executed correctly using a reference implementation and then accelerated, without changing the API, artifact loader, session scheduler or device backends unless the semantics genuinely require a new capability.
2. A new **hardware architecture/NPU/accelerator** can be discovered and provide kernel and memory/IO capabilities, without adding device identifiers to model recipe definitions.
3. A new **task modality/control loop** can compose existing models, encoders, kernels, state and IO into a new `TaskProgram` without pretending every workload is autoregressive next-token decoding.

**Universality has boundaries:** unknown trained mathematical operators, missing output encoders, fundamentally different task control semantics or insufficient physical resources still require new reference code/validation; cannot be manufactured by a generic GGUF reader. The architecture is **semantics extensible**, not magically supports arbitrary model code.

## 1. Orthogonal primitives — no inheritance from named source engines

The architecture consists of a small, intentionally stable vocabulary:

| Primitive | Meaning and ownership | Must NOT own |
|---|---|---|
| `Artifact` | immutable bytes, manifest, tensor metadata, tokenizer/media/adapter provenance | GPU kernel choices or sessions |
| `ModelDefinition` | verified model architecture and semantic graph recipe | specific GPU vendor or HTTP router |
| `SemanticOp` | mathematical operation plus shape/dtype/position/numeric and effect contract | buffer addresses or timeouts |
| `StateDomain` | logical evolving session data, transitions, snapshots, fork/restore/commit | unversioned mutable memory |
| `TaskProgram` | task-dependent control flow, stages, iteration and output stream contract | particular matrix kernel / device brand |
| `Region` | semantic subgraph with checked inputs/outputs/state effects and fallback boundary | API requests or global memory budget |
| `KernelCandidate` | target-specific or generic implementation for one region/shape/precision/memory policy | changing model mathematics |
| `Device` | probed compute, memory, links, synchronization and power/isolation capabilities | model family rules |
| `ResourceLease` | globally budgeted buffer, lifetime, alias, generation and outstanding events | untracked GPU/host allocations |
| `CommandGraph` | asynchronous compute/IO/copy/collective/fence and commit dependency DAG | model semantic inference |
| `ExecutableBundle` | compiled region alternatives and direct command handles, keyed by provenance | interpreting whole model each token |
| `ExecutionPolicy` | pure resource/device/session/quality constraints for a workload | request mutable frontier |
| `StepPlan` | selected actual work rows/iterations and executable handles this scheduling cycle | mathematical model source |
| `TaskSession` | authoritative logical input/output frontier, model state and cancellation epoch | direct unsafe GPU pointer ownership |
| `ModeReceipt` | honest report of actual compute/precision/fallback/state/performance mode | hidden user data |
| `ValidationEvidence` | tests/quality/performance/driver/source proofs for promotion of candidate | speculative runtime guesses |

Each primitive has one owning module. Cross-cutting metadata comes from strongly typed IDs and versioned references. Don't build a mandatory dynamic plugin runtime just to separate crate names. Initially statically compiled Rust modules/crates are preferable; optional isolation/plugins only with trust/performance evidence.

## 2. Four dimensions of composability

**(A) Semantic composition:** tensor values, ragged/sequences/encoded objects, positional invariants, typed state effects. A task's operators form a graph. Pure graph fragments may legally fuse/reorder; stateful fragments require effect ordering.

**(B) Control composition:** `TaskProgram` loops and conditional branches decide *when* semantic regions run: prefill/decode, speculative verify, encoder/chunk stream, iterative denoising, vocoder steps, tool-call parsing, reranking, video temporal scheduling. Control is not "just tensor data" and need not be forced into a decoder loop.

**(C) Resource composition:** independent compute owners, weight/source residency, state owners and output consumers; memory tiers, storage, links and clocks; asynchronous event/timepoint dependencies and physical safety.

**(D) Service composition:** structured API, SDK, CLI, local/remote/generic orchestrator; user request validation and streaming do not determine computation backend or model graph. A new API transport translates to canonical task requests; no new kernel implementation.

If these dimensions are entangled, the system will accumulate per-model forks. Architecture reviews must test that an extension touches only its owned dimension plus unavoidable *new* semantic capabilities.

## 3. Layered compilation and execution — graph IR disappears from hot path

```
User/SDK/Protocol edge
      |
     TaskRequest + Model/Artifact identity
      |
  Trusted ArtifactStore, Decoder/Tokenizer/Preprocessors, TaskProgram
      |
  [SIR] Typed Semantic Graph + effects
      |  pure values, state domains, math and operator version
      v
  [RIR] Legal region formation and optimization
      |  fusion, tiling candidates, quant packing, row split, layout conversions
      +------------+
      |            \
      |           [KIR / Kernel Providers]
      |            portable CPU/GPU and architecture-specific Rust kernels
      v           /
  [CIR] Resource/Command Graph
      |   explicit allocation/lease, copy, device streams, SSD/host/NPU work,
      |   timepoints/fences, graph breaks, commit/abort, remote collectives
      v
  Immutable ExecutableBundle variants (+ independent fallback)
      |
  HardwareCapability/TopologyProbe -> PhysicalResourceArbiter
      |
  TaskCoordinator (control/state)  -> StepPlan
      |
  Device/IO Workers: direct launches / captured graphs / coded events
      |
  WorkerCompletion + provisional state -> StateTransaction commit
      |
  Structured output, stream and ModeReceipt
```

`SIR/RIR/KIR/CIR` are **design-stage representations**, not mandatory separate public crates, text files or long-lived runtime interpreters. The compiler may perform partial JIT at model load or reuse cached kernels. At steady execution, the engine submits already verified command handles/graphs; no per-operator string lookup, source rewriting, IR AST walk or dynamic memory allocation for each token/frame. Source model, hardware and runtime policy remain separate.

### 3.1 Semantic IR (SIR): mathematical truth
Value types carry shape constraints, logical dtype, quant *storage* vs compute contract, positions, token/media spans, device-independent state domains and effects. Operators include pure arithmetic and attention/recurrent/sparse/multimodal primitives with reference implementation. `ModelDefinition` validates exact checkpoint tensor names/config and creates a SIR graph from typed ops.

Effect examples: read/write KV, recurrent matrix, convolution history, sparse/compression index, n-gram history, RNG/grammar state, sampling, streaming. Compiler cannot move effects across causal and commit boundaries unless a proven transform is legal.

### 3.2 Region IR (RIR): legally better math
Choose a **closed semantic region** with explicit type/state inputs and outputs; pass legality checker and numeric contract; generate alternative fusion, quant repacking, grouped MoE, sparse gather, projection output-row partition, decode/prefill/verification tiles. Pure lower-level rewrite/exploration may be offline and bounded. A transformation must not silently shrink trained expert top-k, change sparse selection or modify model numerics without a user-visible validated quality mode.

RIR records `FallbackBoundary` and optional `StateLayoutAdapter`. A missing optimized kernel can be substituted at an op/region boundary only when physical layout/state versions and numeric contract match; otherwise compiler schedules conversion/checkpoint replay. A forced "no specialized region" test is mandatory for each new operator family.

### 3.3 Kernel IR/providers (KIR): best implementation is a capability
Each implementation registers:
- semantic region hash and precise supported op algebra, state effects and version;
- shape/strides, alignment, model tensor quant and temporary compute encoding;
- target instruction/architecture features and precision/reduction behavior;
- scratch/lease/IO, stream/capture support and lifetime constraints;
- proof IDs: independently measured numerical quality, stability, whole-request performance and source/license;
- explicit unavailable/fallback reason.

KIR is a **branch into code generation and implementation selection**, not a second semantic model. A Rust-owned whole-layer or whole-model fast kernel is welcome if it satisfies the **same region contract** as its portable decomposition and doesn't force a parallel model runtime. Kernel authors can use different Rust GPU DSL/target intrinsics; user policy on GPU vendor FFI and foreign device programs remains explicit and open.

### 3.4 Command/resource IR (CIR): data moves and dependencies
A typed DAG describes `Acquire,Copy,Import,Pack,Prefetch,Dispatch,Wait,Signal,Barrier,Join,PublishResidency,RemoteCollective,Commit,Abort,Release`. It knows host/GPU/NPU/SSD/remote device capabilities and alias/lease generations. GPU driver/host IO failures are effects and cannot be reordered past state commit.

Parallel work is legal only when resource and state effects allow it. Example: execute independent projection output rows on one GPU and an NPU, or route hot/cold experts to two GPUs. The **compute owner may differ from the weight resident owner**; the task's state owner is yet another entity. Shared addressability is not free bandwidth nor proof of coherence.

### 3.5 ExecutableBundle and direct runtime
A plan cache keyed by model/quant, semantic schema, state layout, kernel source/compiler/driver/ISA, shape/batch/context band, execution mode, numeric/quality policy and hardware topology stores proven candidates. Runtime selects among precompiled variants with bounded policy/hysteresis; no unbounded autotuning/JIT in latency-sensitive loops. A fully specialized execution island can capture GPU graphs. A cache miss or unqualified device triggers the safe generic plan.

## 4. TaskProgram: model-general AI stage execution

The task controller is **an intentional higher-level construct** separate from SIR pure math and service HTTP. Typed stage nodes consume/produce `ValueType`, `StateDomain`, `ResourceEstimate` and `StreamBoundary`; edges may be data dependencies, control branches, loop-carried state or asynchronous IO.

### 4.1 Control graph types
- `Stage`: typed semantic model/pipeline region or preprocessing/postprocessing operation.
- `Sequence`: compose stages in order with typed data and state handoff.
- `Parallel`: independent stages with explicit join and resource reservation.
- `Select`: conditional branch using validated runtime decision and reproducibility policy.
- `Iterate`: loop body, termination predicate, carried state and output frontier.
- `Window`: bounded streaming temporal/windowed processing with finalization alignment.
- `VerifyAndCommit`: draft candidates and authoritative target state transaction, if task uses speculation.
- `Stream`: emitted chunks only after task-specific commit/progress validation.
- `ExternalAction`: optional high-trust tool/workflow stage **outside** model computation (permissioned and sandboxed, never implied by a prompt).

This is **not a universal untrusted workflow language**: initially tasks are Rust-authored built-in definitions. Unknown external code requires explicit sandbox/provider trust; loop predicates and control effects are validated and bounded.

### 4.2 Common task recipes without separate engines

| Task class | Typical TaskProgram | State/frontier semantics | Output |
|---|---|---|---|
| Text generation | render→prefill→Iterate(decode/optional draft verify)→stream | token frontier, KV/recurrent/indexer/RNG | text, structured tool/reasoning |
| Encoder embeddings | tokenize/encode→pool/normalize | encoder spans, no autoregressive KV | vector scores/dimensions |
| Cross-encoder reranking | tokenize pair→encode→score | paired input span, no LM sampler | ranked/scored items |
| ASR | audio preprocess/window→encoder→CTC/seq2seq as model requires→align | acoustic windows/state and optional decoder tokens | transcript/timecodes |
| TTS / voice | text condition→acoustic model loop→vocoder→audio stream | speaker/voice and acoustic frame frontier | PCM or encoded speech |
| Image generation/edit | input/conditioning→Iterate(denoise/flow steps)→decode | latent, timestep, random noise and conditioning | image |
| Video/audio generation | multimodal conditioning→temporal/frame loops→postprocess/output mux | frame/temporal state, modality alignment | video+audio stream |
| Retrieval/RAG | embed→retrieve external allowed data→rerank→compose prompt→generate | retrieval provenance and tool boundary | text + grounded references |
| Multi-agent orchestration | separate permissioned TaskPrograms exchanging typed results | separate task sessions and provenance | agent artifacts/results |

These are **capability targets**, not assertions of support. Each model/task/control/quant/backend has independently measured reference/functional/performance statuses.

### 4.3 Streaming and cancellation contract
No task assumes a token == a frame == a diffusion timestep. Every TaskProgram declares its own `CommittedFrontier`, partial-output policy and cancellation granularity, plus stream codec and privacy requirements. A generated image latent is not externally committed merely because one denoiser step finished. A TTS stage may safely emit completed PCM chunks while later chunks are pending. A model tool call does not grant runtime arbitrary network/fs permissions.

## 5. Reference semantics and invariants across model families

The `SemanticOpRegistry` is reusable by **operator algebra**, not name/version hacks. High-level semantic families include:
- full/sliding/latent/compressed/sparse attention, GQA/MQA, position encodings;
- gated recurrent/delta attention and short convolution, model-specific recurrence equations;
- router-scored MoE with shared/routed expert weight rules and stable aggregation;
- hyperconnections/gated residual, normalization, packed quant linear/pointwise;
- sparse/time-indexed n-gram/Engram lookup and immutable disk row semantics;
- trained drafter, target verifier and distribution-preserving stochastic sampler;
- encoder, projector, vocoder, diffusion timestep, flow and media decoder;
- tokenizer, model-native role/grammar constraints, adapter overlays and multimodal positional spans.

An operator includes **exact mathematical geometry and equations** derived from a pinned model/config; a model name can appear as a test fixture or model adapter identifier but never in the core compiler/allocator/server API. Register new high-level op when an existing op's math is not equivalent. Do not approximate a trained sparse top-k, compressed context ratio or recurrent gate to reuse an existing kernel silently.

## 6. Topology- and quality-aware optimization

Discover CPU architecture/SIMD, integrated and discrete GPUs, optional NPU/AIE/other accelerators, physical RAM/VRAM/unified mapping/pinned allocations, storage, peer links, NUMA, power/fabric/shared-resource domains, firmware and driver. Capability probe exposes *verified available* from *advertised address space* and *quality/stability qualified*.

Selection maximizes a **measured Pareto objective** (committed outputs per wall time, TTFT/P95, throughput/concurrency, energy, physical memory, accuracy) subject to resource reservations and task correctness.

### 6.1 Two important heterogeneous region strategies
**SplitProjectionCompute:** RIR may divide a pure GEMM's output rows into disjoint, covering spans executed by different providers, with explicit assemble and first-use barrier. NPU block-float is an optional *compute encoding*, not a change to checkpoint source quant. If the NPU is unqualified or a requested numeric mode rejects its approximation, choose GPU-only without changing ModelDefinition. Hardware power/fabric instability can be a **safety veto**, and no privileged clock/sysfs/firmware tweaks are automatic.

**AsymmetricExpertOwnership:** a routed MoE's mathematical expert selection remains identical while compute and weight residency are distributed across faster/smaller and slower/larger accelerators or SSD. Send activations and partial results, not giant weight matrices every token. Cost compares transfer, dispatch, queued IO and kernels per phase, not just GPUs' reported memory.

Other strategies: layer pipeline, tensor/data/expert parallel, disaggregated prefill/decode, CPU/GPU hybrid, shared-memory zero-copy where actually coherent, scratch alias and cold expert prefetch; each must be opt-in/validated candidate with fallback and task-state schema.

### 6.2 Task-aware specialization
For prefill, decode1, target verification, encoder batch, diffusion denoiser, acoustic/vocoder and streaming media: different shapes, numerical mode and critical paths lead to different kernel/placement/graph strategies. The task stage declares its result and precision contract; runtime selects physical implementation. This prevents the false assumption that a fast prefill GEMM also accelerates batch1 decoding or a vocoder.

## 7. One global resource authority; different cache policy by data class

PhysicalResourceArbiter owns mandatory model weights, expert/cache slabs, KV/recurrent/indexer/prefix state, adapter weights, NPU+GPU pinned buffers, IO staging, scratch, compiler graph handles and all in-flight DMA/host buffer leases. Pools may differ by lifecycle and eviction strategy, but allocations all reserve against OS/cgroup/headroom. GPU virtual address/GTT is NOT physical free RAM.

Per-domain state supports `CanResume`, `Snapshot`, `Restore`, `Fork`, `Commit`, `Abort`, `Recompute`, schema/quant/layout identity and optional codec. The **full common prefix frontier** must be valid across all token/vision/attention/recurrent/indexer/grammar/RNG domains; a text prefix hash alone is not a valid state hit.

Model residency, session state and TaskProgram stage state are separate; no expert eviction can change mathematical tensor identity. Resource leases carry generation, device(s), access mode, last relevant event and drop/quarantine actions. Timed-out GPU/NPU DMA cannot be freed merely because host timeouts elapsed. Device lost/error aborts provisional task state; valid committed output remains valid.

## 8. Serving and concurrency independent of models

One canonical `TaskRequest`/result/stream contract and one logical session writer; `EngineMsg/EngineEvent` and device worker `StepPlan/WorkerDelta` transport small typed IDs/leases rather than full tensor values. A user-facing SDK/API maps chat, completions, embeddings, ASR/TTS or image/video result requests into TaskPrograms. DeviceWorkers do not own token sampler state; they execute compiled mathematical regions and return provisional effects.

Token-budget continuous batching is a **policy for compatible autoregressive text stages**; other tasks have their own unit budgets (audio frames, encoder rows, diffusion steps, GPU milliseconds and memory). Stage-aware scheduler fairly shares devices and slots without forcing a single universal token metric. Prefill/decoder/sampler/timestep loops can be batched when task/mode state and kernel shape allow, otherwise independently dispatched.

Event-driven overlap permits CPU preprocessing, SSD expert/table reads, NPU/GPU projection slices, GPU routes and remote sends where effects permit. `ModeReceipt` truthfully records selected backend/kernel/quality/caching/speculation, SLO profile, fallbacks, unqualified devices and task execution stages. Do not leak private prompts, audio, images or hidden reasoning in metrics.

## 9. Numeric fidelity, sampling and artifact identity

**Separate four truths:**
1. *Semantic correctness*: trained mathematical op and position, all state effects are right.
2. *Numerical agreement*: compute dtype/rounding/quant and tolerated output error are documented and oracle tested.
3. *Model/task quality*: checkpoint+quant vs official/independent quality tests at realistic prompts, context, modalities and tool scenarios.
4. *Target distribution/state*: speculative proposals do not perturb target sampling or rejected recurrent/KV/indexer/RNG/grammar state.

A source checkpoint's stored precision and derived device **packing** are different from a lossy **compute encoding** or a new **requantized checkpoint**. A fast kernel can only become default under the selected contract; strict reference mode rejects unqualified numerical change. Greedy token match on 30 prompts is not a quality proof; neither is a faster kernel alone a serving win.

Record artifact source SHA, tokenizer/template/model semantics/adapter/multimodal/quant/driver/kernel/compiler/plan/program IDs, safety profile, memory topology, context/batch/concurrency/acceptance, and full-task completion mode. Tests and benchmarks independently reference source code/checkpoints by provenance, never infer correctness from a named engine's published tok/s.

## 10. Security and code provenance

Only verified trusted **artifact data**, not executable custom Python/code from a model registry by default. Kernel/device programs are executable trusted code and subject to user Rust-owned source intent; upstream code may be studied as prior art under verified licenses, not imported because architecture wants its performance. A foreign language GPU/NPU kernel invoked via Rust FFI does not automatically satisfy all-Rust-source requirement. Driver/OS APIs and vendor maths binding interpretation remains an explicit user decision.

Privacy: session/cache/model-store separation, namespace/tenant salt and safe file permissions; authentication for remote protocols; external agent tool execution requires explicit permission sandbox; no automatic sudo/system clock/BIOS/IOMMU manipulation by inference runtime. A shared SoC fabric/accelerator combination with known machine hangs is **disabled until exact hardware-qualified**, never turned on by a generic "device exists" probe.

License and attribution detail is in the prior-art atlas. Project-independent interfaces make it possible to remove a non-MIT source entirely without requiring architectural changes.

## 11. Minimal Rust crate/module ownership (not mandated API)

Suggested **small coherent workspace**:
- `artifacts` — bytes, tensors, quant codecs, model and tokenizer manifests.
- `semantics` — typed values/ops/state/effects, reference evaluator, model recipes.
- `tasks` — TaskProgram control graph and task results, iteration/frontiers.
- `compiler` — region passes, KernelProvider registry and command lowering.
- `devices` — actual CPU/GPU/NPU/IO and stream/event capability adapters.
- `resources` — physical budget, typed leases, KV/expert/context pools.
- `runtime` — TaskCoordinator/EngineCore, StepPlan, transactions, scheduler.
- `service` — CLI/Rust SDK/HTTP/streaming, auth and observability.
- `validation` — official model/task oracle, fuzzing, performance and evidence.

Not one crate per line by default; group for clear layering, avoid cycles. Hot devices are statically linked Rust providers first; no mandatory dynamic plug-in ecosystem. The compiler is allowed to hand off a whole model/region to a direct specialized Rust kernel graph if inputs/outputs/effects/resources match; does **not** hand control to a separate model-specific server.

## 12. Extension acceptance: how we know modularity is genuine

**New trained model:** add exact tensor manifest/ModelDefinition+required SemanticOps/reference, optionally numeric/state schema. Existing generic backend runs it if all operations supported. New optimized kernel follows independent test. **No changes to HTTP, generic scheduler or physical allocator unless new capability actually required.**

**New accelerator:** implement capability probe, codegen/kernel provider, buffer/event/IO adapter and safety/quality qualification. Existing ModelDefinition and TaskProgram remain unchanged. Probe can refuse unsupported/unsafe device and choose existing CPU/GPU compiled regions.

**New quantization scheme:** add QuantCodec + packing/dequant oracle and numerical policy; semantically identical model except consciously derived checkpoint identity. No model-name-specific global switch or silently changed quality.

**New task modality:** add TaskProgram control stages, codecs, state loop/commit boundary and output adapter, reusing model ops and backends. No forced decoder loop or second standalone engine.

**New inference optimization:** implement legal rewrite or scheduling alternative behind region capability and source/benchmark proof; disable safely if quality/latency/resource regress.

**New storage/transport/remote topology:** extend ResourceTier/CommandProvider and security/error contract; do not modify every model adapter, tokenizer or API route.

**New tool/agent action:** add permissioned ExternalAction stage; never let a model math kernel acquire filesystem/network privileges merely because of a task graph node.

## 13. Anti-patterns explicitly prohibited

- `if model_name == ...` branching in generic device or server runtime for every specialized model.
- Model-specific clone of server, scheduler, resource manager, sampler, tokenizer registry or device graph runner.
- Project-named core Rust interface or mirroring third-party object layouts.
- One giant opaque `ExecutionPlan` containing model math, kernel code, memory placement and mutable request queue.
- Raw GPU/NPU pointer aliases without event, generation, memory-tier and failure contract.
- Per-token SIR/RIR/CIR AST interpretation, unbounded dynamic code search or graph recompilation on every shape.
- Forced conversion of non-token iterative tasks into `decode_next_token`.
- Using available GPU address space as physical RAM, silently reducing expert top-k or source quant to gain speed.
- Reusing incompatible state across different LoRA adapters, model versions, prefixes, images or sampled grammars.
- Automatic privileged device/firmware/power tuning to enable an experimental accelerator.
- Treating any source-author benchmark or unverified license assumption as a validated zerogravity result.

## 14. Proof-of-architecture program (project-neutral gates)

**G-A Structural:** one small semantic reference model and a synthetic hybrid-state model lower through SIR/RIR/KIR/CIR to CPU and one GPU. Forced removal of a target kernel selects a correct region fallback with full state. No token-time IR interpretation.

**G-B Compositional:** add second distinct model/operator algebra without editing runtime API/ResourceArbiter and add a second hardware backend without editing ModelDefinition. Code-change/line/module metrics document extension cost, including independent failures.

**G-C Resource/state:** fuzzer checks async leases/alias/IO and 3+ state domains' snapshot/accepted-prefix/cancel/backpressure, with exact causal frontier, media spans and tenant/model hashes.

**G-D Task reuse:** one non-token TaskProgram executes a reference encoder or mocked iterative denoising workflow through the same runtime workers, outputs stage-specific result and supports cancellation without language token state.

**G-E Specializations:** representative shape-specific quant kernels, sparse experts, projection-row GPU/NPU split and nonresident IO compete against generic baseline under matched models/precision/power and supervised hardware safety. A blocked optional NPU never blocks portable CPU/GPU.

**G-F Product/reliability:** legal model formats/quant, templates/tools, API conformance, numerical/quality and physical memory admission; cold/warm and real concurrency P95. A novel accelerator requiring uncontrolled privileged clocks is not eligible for automatic scheduling.

**G-G Sustainable evolution:** re-run device/compiler/IR/schema upgrade with model/task state migration or invalidation, source license attribution, compile budgets, error taxonomy and rollback capability. An optimization that cannot be removed without breaking unrelated model support failed modularity.

## 15. Performance strategy — optimize across regions and workloads, not for slogans

A combination of a compact, correct execution spine and aggressive optional specialized regions has the best **architectural potential** to reach multiple hardware/workload Pareto fronts. It does not promise universal speed superiority; every win requires benchmark under identical model checkpoint/quant/driver/power/mode/context/concurrency and quality tolerance.

Main potentially differentiating mechanisms:
- reusable **state-aware hybrid-prefix sharing** across 3+ state groups and modalities;
- **joint batching/expert residency/speculation** under one physical budget and measured critical path;
- **cross-accelerator operator slices** with explicit first-use barriers, not GPU-only graph assumptions;
- **heterogeneous expert ownership** across fast/small and slow/large GPU/CPU memory tiers;
- **compiled asynchronous command graphs** to overlap SSD/CPU/device work while preserving ownership and state;
- **task-aware scheduling** for LM, ASR, TTS, image/video, embeddings and future AI computations;
- ability to **override entire semantic regions** with Rust hand-tuned kernels while still retaining a portable reference.

A narrow hard-coded engine may remain faster on its one exact shape; this architecture's benefit is that adding that kernel/shape does not cost a second runtime. Measure this claim through full-model benches and the extension drill.

## 16. Reading and scope authority

**Canonical norms:** this file, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`, the rewritten master plan, `IMPLEMENTATION_ROADMAP.md`, `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`, `INVARIANTS_AND_FAILURE_SEMANTICS.md`, `SECURITY_AND_RELIABILITY.md` and `DECISION_REGISTER.md` (user-approved choices only). Older subsystem documents contribute operator/driver-specific detailed requirements but named upstream projects inside them are **research annotations, not normative architecture rules**.

**Prior art:** `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` and source-specific research files preserve exact source names/SHAs/benchmark provenance. **No source from prior-art projects was copied unless directly vendored (for example, CubeCL (vendored)), which retains its upstream license; the Zerogravity engine is a from-scratch rebuild informed by prior-art concepts, learnings and research.** No kernel was compiled by this documentation work. Implementation remains all NOT_STARTED.

---

# Archived v3 source 2: `COMPOSITION_AND_EXTENSION_CONTRACTS.md`

# Composition and extension contracts — modular AI without engine rewrites

**Normative design · 2026-10-12 · NOT IMPLEMENTED.** This specification operationalizes `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`. Types below are **illustrative Rust design sketches**, not frozen public API or crate names. Every interface represents a responsibility independent of named external projects. Prior-art implementations and licenses are indexed separately in `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`.

## 1. Product-independent contract rules

- **Semantic identity precedes execution choice:** an operator implementation is selected by `SemanticOpId + OpVersion + Geometry + StateEffects + NumericContract + ExecutionRegime`, not a string model name or an external project's kernel kind.
- **Operator = mathematical meaning; region = replaceable unit:** an implementation can replace pure matrix tiles, an attention+MLP fused layer, an entire decoder island or a whole validated model if and only if external values/state effects match.
- **Task = control:** a loop/stream condition belongs to a task controller, not a GPU kernel or a type-erased model entry. Token loop and denoise-step loop are different instances of the same typed stage orchestration, not the same semantic program.
- **Hardware = capability:** device differences are handled through supported tensor/codegen features, memory resources, event/IO behavior and power isolation. No model recipe references a device SKU.
- **Resource = lease:** every CPU/GPU/NPU/SSD/remote alias has an accountable lifetime and a result/event dependency. No free while in-flight.
- **Fidelity = contract:** output precision and sampling/quality properties must be validated. Fallback cannot silently alter trained math, MoE top-k, sparse attention or model artifact.
- **Failure = typed effect:** failed IO/device/collective invalidates provisional results and preserves the last committed TaskSession frontier, or clearly reports that recovery is impossible without replay.
- **Interface surface = small and versioned:** no giant dynamic trait hierarchy and no explosion of microcrates. Prefer Rust enums, typed IDs and capability registries until extension demands dynamic loading.

## 2. Universal data contract: values, streams and physical representations

### 2.1 Logical values

```rust
enum ValueType {
    Tensor(TensorType),
    Tokens(TokenBatchType),
    TextFragment(TextEncoding),
    Audio(AudioSpanType),
    Image(ImageTensorType),
    Video(VideoFrameType),
    Structured(StructureSchemaId),
    SparseRoute(RouteType),
    Scores(ScoreType),
    Unit,
}
struct TensorType {
    logical_dtype: LogicalDType,
    logical_dims: Vec<DimExpr>,
    quant: QuantSemantics,
    layout_constraints: LayoutConstraints,
}
struct ValueRef {
    id: ValueId,
    ty: ValueType,
    provenance: DataProvenanceId,
}
```

Logical types never embed GPU pointers. Device representations are separate `PhysicalValue` with layout, alignment, block-packed quant, buffer generation, memory tier, finalization and observed timepoint. A device implementation cannot reinterpret compressed attention cache as full KV or block-float compute data as FP32 merely because byte counts match.

### 2.2 Stream frontiers

`StreamContract` identifies event unit (text token, transcript/audio frame, video frame, diffusion completed image, score vector or structured document), legal partial output and finalization, ordering and backpressure. `CommittedFrontier` may be `TokenIndex(u64)`, `AudioSample(u64)`, `FrameIndex(u64)`, `DenoiseStep{index,total}`, `SequenceOfSegments` or task-specific state; it must not be an untyped `u64` assumed to represent one language token.

A contract must identify what is externally irrevocable and whether intermediate state is private/provisional. For image generation, 20/50 denoising steps completed does not imply an image emitted; TTS may emit one verified PCM chunk while preparing another.

## 3. ModelDefinition and exact semantic operators

```rust
trait ModelDefinition {
    fn identify(&self, artifact: &ArtifactManifest) -> Result<ModelIdentity>;
    fn validate(&self, artifact: &ArtifactManifest) -> Result<ModelRecipe>;
    fn build_program(&self, recipe: &ModelRecipe) -> Result<SemanticProgram>;
}
trait SemanticOp {
    fn schema(&self) -> &'static OpSchema;
    fn verify(&self, args: &OpArguments, effects: &EffectSet) -> Result<()>;
    fn reference_execute(
        &self,
        inputs: &ReferenceInputs,
        state: &mut ReferenceState,
    ) -> Result<ReferenceOutputs>;
}
```

The model recipe includes tokenizer/position/quant/semantic op/state family and exact layer/tensor names, not "compile for accelerator X" or "use vendor Y". Adding truly new model math adds a reference operator and validated state codec. A model file without these cannot be claimed runnable. A single model architecture may yield more than one task definition (e.g. encoder+decoder, retrieval score and generation), but shared weights/caches are versioned separately.

**Novel models don't automatically need new operators.** An architecture composed of registered exact semantics merely provides new tensor geometry and sequence/graph definition. Conversely two architectures with the same marketing label might need different operators due to indexer/position/gate changes.

## 4. StateDomain: make effectful inference composable

```rust
trait StateDomain {
    fn schema(&self) -> StateSchema;
    fn can_resume(&self, prefix: &ResumeQuery) -> ResumeCompatibility;
    fn snapshot(&self, committed: &StateHandle) -> Result<SnapshotHandle>;
    fn fork(&self, frozen: &SnapshotHandle) -> Result<StateHandle>;
    fn transition(&self, input: &StateHandle, value: &ValueRef)
        -> Result<ProvisionalState>;
    fn commit(&self, candidate: ProvisionalState, frontier: CommittedFrontier)
        -> Result<StateHandle>;
    fn abort(&self, candidate: ProvisionalState);
    fn restore_or_recompute(&self, source: &CanonicalHistory)
        -> Result<StateHandle>;
}
```

No `commit` is individually externally visible unless the task-level `StateTransaction` coordinates **all** domains (KV, recurrence, sparse index, n-gram, grammar/RNG, images and task-local state) and publishes a single consistent frontier.

A snapshot interface returns immutable state with explicit versioned layout. A backend can use COW, a frontier ring, deferred GPU copy or replay, but state version/quant/position semantics must remain correct. A failed proposed token can be discarded without changing target model distribution or polluting prefix cache. Prefix sharing requires compatible model/quant/adapter/template/position/modality/task identity and a common resumable frontier across all domains.

## 5. TaskProgram: control-flow algebra

```rust
enum TaskNode {
    Invoke { stage: StageId, inputs: Vec<ValueId> },
    Sequence { stages: Vec<TaskNodeId> },
    Parallel { branches: Vec<TaskNodeId>, join: JoinPolicy },
    Select { decision: PredicateId, branches: Vec<TaskNodeId> },
    Iterate { body: TaskNodeId, condition: PredicateId,
              carried_state: Vec<StateDomainId>, max_steps: u64 },
    Window { size: u32, finalization: WindowPolicy, body: TaskNodeId },
    VerifyAndCommit { target: TaskNodeId, drafts: Vec<TaskNodeId> },
    Emit { input: ValueId, contract: StreamContractId },
    ExternalAction { policy: ActionPolicyId, request: ValueId },
}
struct TaskProgram {
    task_kind: TaskKind,
    input: InputSchema,
    output: StreamContract,
    graph: Vec<TaskNode>,
    semantic_programs: Vec<SemanticProgramId>,
    resource_limits: TaskResourceLimits,
}
```

This is a **minimal typed built-in orchestration language**, not a user-supplied sandbox escape. `Iterate.max_steps`, timeouts, value dimension bounds, number of branches and resource policy must be checked. Task stages can contain callbacks to trusted Rust implementations, registered semantic subgraphs or authorized external actions. A task never creates a second model server; it calls the shared engine executor.

### Task-level guarantees

- **Text LM:** reference source target logits, token/grammar/sampler transaction; tool outputs are structured messages, not GPU kernels.
- **ASR:** samples/chunks and timestamps, model-specific encoder and optional decoder, sliding windows and finalized transcript spans.
- **TTS:** conditioning, acoustic tokens/latent and vocoder, stable PCM/codec partial chunks; a voice model's style/seed becomes TaskSession state.
- **Image edit/gen:** image/text conditioning, timestep loop, deterministic or sampled noise state; output image after validated decode.
- **Video/audio:** frame/temporal interleave, chunked resource budgets and output mux, repeated denoiser/decoder stages.
- **Embeddings/rerank:** encoder plus normalizing/pooling/score, no autoregressive KV or sampler.
- **Agent composition:** external tool execution and other task calls happen through **permissioned** ExternalAction, never by giving inference code filesystem/network privileges.

### Threading and ownership

One logical `TaskCoordinator` (EngineCore is an implementation name) owns task sessions/control/frontiers. DeviceWorkers process compiled regions and return provisional completions. Stage/loop control never lives inside HTTP handlers or device kernels. Different TaskPrograms can batch compatible stage kernels where model/shape/adapter/numeric contracts agree.

## 6. KernelProvider and closed-region substitution

```rust
trait KernelProvider {
    fn enumerate(
        &self,
        region: &SemanticRegion,
        device: &DeviceCapabilities,
        numeric: &NumericContract,
    ) -> Vec<KernelCandidate>;
    fn compile(&self, candidate: &KernelCandidate) -> Result<CompiledKernel>;
}
struct KernelCandidate {
    region_identity: RegionId,
    supported_shapes: ShapeConstraintSet,
    input_layouts: Vec<PhysicalLayoutId>,
    output_layouts: Vec<PhysicalLayoutId>,
    scratch: ResourceFootprint,
    effects: EffectContract,
    execution_modes: Vec<ExecutionMode>,
    numerical_contract: NumericContractId,
    target_capability: CapabilityExpr,
    evidence: EvidenceId,
}
```

A provider can be a portable CPU evaluator, target GPU kernel, NPU tile, special low-bit GEMM or whole-layer fused graph. It must register a reference/fallback region, and its `compile` must fail with a typed `UnsupportedShape`/`UnsupportedQuant`/`UnqualifiedDevice` rather than emitting wrong arithmetic. A solver chooses validated plan variants by numeric/resource constraints, not model-name if/else.

A kernel replacing a whole region is allowed to have a radically different implementation/layout, provided it can materialize the exact logical inputs/outputs, states and effects at a legal boundary. Some state conversions are too expensive or unsupported mid-token; fallback may need to replay from a canonical committed checkpoint.

## 7. DeviceProvider and accelerator topology

```rust
trait DeviceProvider {
    fn probe(&self) -> Result<DeviceCapabilities>;
    fn memory(&self) -> &dyn ResourceMemoryAdapter;
    fn submission(&self) -> &dyn CommandSubmitter;
    fn event_bridge(&self) -> &dyn EventBridge;
}
struct DeviceCapabilities {
    device_id: DeviceId,
    compute: ComputeFeatureSet,
    memory: PhysicalMemoryProfile,
    transfers: Vec<TransferCapability>,
    isolation_and_power: DeviceSafetyProfile,
    driver_and_kernel: VersionedIdentity,
    execution_modes: Vec<ExecutionMode>,
}
```

A device *is not* merely a quantized GEMM capability. It may support partial recurrent, BLAS GEMM or encoding only, require specific pinned memory or have hazardous shared power domain. The Runtime avoids invoking a device if it lacks capability/numerical/stability qualification. Default fallback to a known provider, with safe memory reservation and user-facing explanation.

**No universal manual configuration flags:** devices are dynamically probed, but controlled hardware profile use is explicit. A reported SoC fabric/clock hard hang means the NPU path is unqualified and cannot auto-execute. The engine never runs privileged sysfs/firmware changes itself.

## 8. ResourceArbiter and CIR event algebra

```rust
struct ResourceReservation {
    committed_bytes: u64,
    staging_bytes: u64,
    scratch_bytes: u64,
    device: DeviceId,
    safe_headroom: u64,
}
struct BufferLease {
    allocation_id: AllocationId,
    generation: Generation,
    memory_tier: MemoryTier,
    access: AccessMode,
    held_by: Vec<TimepointId>,
}
enum Command {
    Acquire(ResourceRequest),
    Transfer { src: BufferLeaseId, dst: BufferLeaseId },
    ImportMapped { external: ExternalHandle },
    Pack { source: BufferLeaseId, codec: CodecId },
    Dispatch { kernel: CompiledKernelId, deps: Vec<TimepointId> },
    Wait { point: TimepointId },
    Join { inputs: Vec<TimepointId>, output: BufferLeaseId },
    PublishResidency { table: ExpertTableId, generation: Generation },
    Snapshot { state: StateVersionId },
    Commit { transaction: TransactionId },
    Abort { transaction: TransactionId },
    Release { lease: BufferLeaseId },
}
```

CIR verifies alias/read-write conflicts, scratch liveness, graph capture and command/effect legality, and compiles static command sequences. The executor handles actual async GPU/NPU/CPU/SSD/remote APIs, not generic SIR interpretation. A host IO failure invalidates dependent state commit. An externally still-running DMA maps a potentially live buffer and must be quarantined rather than freed.

**SplitProjectionCompute example**: RIR proves disjoint covering output rows and source quant/numeric contract; CIR packs/dispatches eligible output rows to NPU while GPU computes others, waits at first actual consumer and assembles outputs; on unsupported NPU program/shape choose GPU-only. ModelDefinition must remain device-neutral. Weight/source residency and compute owner may differ, but all pinned/mapped physical bytes count.

**AsymmetricExpertOwnership example**: target router generates exact top-k, GPU-side table maps each selected expert to a compute owner/weight resident slot. CIR passes activations/partial results and stages misses under leases; it does not alter router scores or expert IDs to reduce SSD traffic without explicit changed-model semantics and quality policy.

## 9. Scheduling policy and control units

```rust
trait SchedulingPolicy {
    fn choose(
        &self,
        ready: &[ReadyStage],
        admission: &AdmissionState,
        resources: &ResourceSnapshot,
        profiled: &CostProfile,
    ) -> Result<StepPlan>;
}
```

Stage work units have a **typed** `WorkUnit`: target tokens, encoder tokens, audio samples/frames, GPU tile rows, denoiser iterations, video frames, route assignments or general device microseconds. A target LLM decoder's speculative accepted draft tokens are **not** counted as completed output until verifier commits.

A single global scalar "token budget" is insufficient for arbitrary AI workloads; use typed per-task budgets plus physical concurrency/memory constraints. Compatible stage batches can share kernels; incompatible stages remain separate but may share device asynchronously. The scheduler can optimize P95 responsiveness, throughput, energy or memory budget under policy. Uncontrolled dynamic autotuning is forbidden in the hot path.

## 10. Artifact and codec extension

New sources are supported by a bounded `ArtifactReader`, `ManifestResolver`, `TensorCodec` and `Tokenizer/PromptRenderer`/media processor. Reading a checkpoint format says nothing about semantic model support; ModelDefinition must validate tensors and reference results. ModelStore is content-addressed, atomic/resumable and has clear cache retention/security policy.

`QuantCodec` handles stored checkpoint encoding, checked block sizes, dequant and optional packing; a compute-only low precision (e.g. block-floating NPU tile) is a **KernelCandidate numerical policy**, not a new source-model identity unless actual checkpoint bytes/precision change. All lossy paths require full-model quality tests and default policy explicit.

## 11. Versioning and extension lifecycle

Every stable registry item carries:
- semantic operator schema version and reference oracle compatibility;
- task program/stream schema version and allowed control effects;
- buffer/state layout schema and canonical conversion/replay;
- physical device/compiler/codegen/driver target versions;
- kernel candidate and numeric policy/evidence hash;
- model artifact/tensor/quant/adapter/template/media fingerprint;
- `CapabilityStatus` (planned, reference, experimental, validated, performance-qualified, unsafe/disabled);
- cache/serialization version and integrity checksum where persisted.

Schema upgrades invalidate incompatible compiled kernels/prefix snapshots and can migrate *only* with tested explicit conversion. Source model mathematical semantics should not be changed by a kernel version upgrade. A provider or experimental mode may be deprecated without deleting ability to load/execute models via generic provider.

**Capability negotiation:** record fallback tiers and reasons. The engine may try a lower-acceleration plan if numerical mode/driver/shape fails, but it must not select a different model quant or silently alter stochastic distribution.

## 12. Benchmark and quality evidence gating

No provider automatically becomes default on "supported" shape alone. Evidence includes:
- independent CPU/high-precision oracle and layer/logit/state parity;
- kernel numeric limits and model quality under same checkpoint;
- correct state fork/rollback/cancel and error buffer lifetime;
- actual end-to-end stage/request latency and P50/P95, target throughput, energy and memory;
- warm/cold/cache/physical power/driver version; native batch vs ordered;
- shape band and eager/captured execution mode;
- old and new source versions and license/attribution.

We should capture candidate performance per actual workload and compare against a simpler general path with paired experiments. A faster kernel in isolation may lose at full-task latency. A GPU/NPU combination with risk of firmware hard hangs is disabled regardless of microbenchmark gain.

## 13. Cross-component extension test scenarios

1. **New recurrent architecture:** one new exact `SemanticOp+StateDomain`, all old task/service code unchanged; CPU reference/decoder proven; additional GPU kernel optional.
2. **New sparse indexer:** supports different selected positions and state serialization; cannot be implemented by silently retuning an old top-k mask.
3. **New NPU:** capabilities/program hash, KernelProvider and physical EventBridge; existing model math unchanged; unsupported quant/unsafe driver chooses GPU-only.
4. **New embedding model:** Artifact+ModelDefinition+TaskProgram pooling; no autoregressive tokenizer sampler or KV state required; same backend/memory execution.
5. **New diffusion or video pipeline:** typed loop and temporally aligned media stream with own frontier; reuses model kernels and resource leases.
6. **New expert tier/IO transport:** CIR CommandProvider and ResourceTier, no changes to model math or API; fault injection prevents stale data.
7. **New low-bit representation:** lossless device repack or explicit requantized checkpoint, not silent numerical downgrade.
8. **New server frontend:** canonical TaskRequest/TaskResult map plus exact auth/validation; engine and KIR untouched.
9. **New scheduling policy:** deterministic input and resource snapshot with testable goodput/P95/fairness, correctness invariant from StateTransaction unchanged.
10. **Disable all extreme providers:** the full semantically supported model/task still functions (subject to available capacity/CPU time) using portable execution or clear error, rather than a fork of another engine.

**Acceptance:** architecture fails the composability requirement when adding any of these requires duplicating or rewriting a model-specific server, scheduler, memory manager, artifact store or loop unrelated to the new capability. Count files/crates/lines changed and performance overhead; proof must be concrete.

## 14. Rust workspace dependency direction

Possible module DAG:

```
core_types <- semantics <- task_program
            \
             compiler_regions -> kernel_registry -> device_bindings
             state_codecs -> resources -> runtime_coordinator
             artifacts -> model_definition -> semantics
             runtime_coordinator -> service_frontends
             validation -> all prior layers (test-only tooling)
```

Actual Rust crate graph must be acyclic and minimal. The diagram conveys conceptual ownership, not literal Rust dependency order. For implementation, extract tiny core traits/IDs into a low-level module, use generics and static dispatch on hot paths, and inject backend providers through bounded registries. Resource and device interface signatures should avoid hard dependencies that force compiler to import HTTP or model adapters to import GPU runtime.

Retain one coherent error taxonomy: `UnknownOp`, `UnsupportedQuant`, `NoLegalKernel`, `IncompatibleState`, `PhysicalBudget`, `DeviceUnsafe`, `DeviceLost`, `BufferLeaseInvalid`, `BackendBusy`, `OutputSchemaMismatch`, `UntrustedArtifact`, `Cancelled`.

## 15. Closure

This document intentionally contains **no external engine project names** in its algorithm or Rust interface definitions. Hardware/protocol/model type names only appear when identifying actual device or source artifact requirements. Prior-art names/licenses and original source paths remain solely in research files. The product must remain understandable and implementable from these interfaces even if every upstream project disappears tomorrow.

---

# Archived v3 source 3: `ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`

# zerogravity — Comprehensive Master Plan v3: Modular Composable AI Systems

**Updated:** 2026-10-12. **Canonical root:** `/home/pvl/spaces/abird/src/zerogravity/`. **Status:** architectural research and design; *no engine implementation, compiled kernels, benchmarks, source repo initialization or release*. All ZG-P00…P23 implementation phases remain NOT_STARTED. This v3 supersedes earlier project-name-derived master architecture while retaining its verified mechanisms as prior art in an explicit research archive.

**Normative companions (read in full):** `../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md`, `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`, `DECISION_REGISTER.md`, `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`, `IMPLEMENTATION_ROADMAP.md`, `RELEASE_ACCEPTANCE_GATES.md`. The first two define original zerogravity architecture. The prior-art license catalog and `PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md` retain exact external project names/evidence, not architectural authority.

## 0. Mission and non-negotiable constraints

Create a **single Rust-authored, semantically correct, hardware- and workload-adaptive execution system for composable AI tasks**. Its design should allow each model/operator/task stage to reach specialized performance on eligible hardware—up to purpose-built fused whole-region kernel execution—without sacrificing a portable reference path or rewriting the rest of the runtime.

Primary qualities, in order:
1. **Math/state truth:** source checkpoint/model/tokenizer/position/state semantics preserved; correctness and user-selected numeric/sampling distribution transparent.
2. **Composability:** new model algorithms, kernels, hardware, memory/IO paths, task modalities and service integrations attach through narrow interfaces; no independent model-specific serving engines.
3. **Physical safety:** every session, kernel, mapped device buffer, cache and async IO reader is reserved/leased; no OS-wide OOM/GPU dangling buffer, unsafe automatic firmware tuning or unvalidated device configuration.
4. **Peak performance freedom:** compiler can lower canonical IR to tuned generic or radically specialized Rust implementations, direct device graphs and native batch/phase-specific plans; no persistent IR interpreter tax.
5. **Operational usefulness:** native local CLI/SDK and robust serving, verified source formats, structured chat/tools/grammar, full-state prefix cache, legitimate multimodal and non-token workloads, accurate observability.
6. **Reproducible optimization:** kernel and serving winners promoted only after matched model/artifact/quant/numeric/workload/driver/power measurements with quality/P95 and failure checks.
7. **Sustainable evolution:** fewer coherent modules, versioned semantic/device/task schemas, hardware drivers and compilers isolated, no large mandatory third-party model runtime or foreign-language kernel source.

**Vision:** maximize the *measured* Pareto frontier of latency, throughput, physical capacity, quality, energy, developer ergonomics and supported models. This is an engineering objective; performance supremacy across every competing implementation is **not guaranteed by architecture**.

## 1. Original independent architecture

```text
                   Application / Local CLI / SDK / API edge
                                  |
                         Canonical TaskRequest
                                  |
         Trusted Model/Artifact Store + Tokenizer / Media Processor
                                  |
                        Typed TaskProgram control
                 (stages, branches, loops, streaming)
                                  |
                          ModelDefinition
                                  |
        [SIR] Mathematical Semantic IR + versioned StateEffects
                                  |
        [RIR] Legal optimized regions + quant/layout/shape plans
                        |                  |
                        |        [KIR] Kernel capability providers
                        |       (portable and specialized Rust implementations)
                        |                  |
                        v                  v
        [CIR] Asynchronous command/resource DAG
             buffers, leases, GPU/NPU/CPU/SSD/remote, events, graph capture
                                  |
                Compiled immutable ExecutableBundles
                                  |
       DeviceCapabilities + Topology + PhysicalResourceArbiter
                                  |
      TaskCoordinator / EngineCore owns session + commit frontier
                                  |
             StepPlan  →  CPU/GPU/NPU/IO Workers
                                  |
            WorkerCompletion + provisional state + verifier
                                  |
              Atomic TaskSession state commit
                                  |
                  Structured result/stream + ModeReceipt
```

**Separations that must survive every implementation decision:**
- `ModelDefinition` = *what* trained math and state transitions are required, never a GPU/NPU selector.
- `TaskProgram` = *how the task proceeds through stages and emits valid outputs*, not a single forced token loop.
- `SIR` = typed semantic equations/effects, `RIR` = legal transformations, `KIR` = how target kernels are compiled, `CIR` = when physical resources/tasks execute and can overlap.
- `KernelPlan` = qualified candidate catalog, `ExecutionPolicy` = hardware/quality/physical-resource preferences, `ExecutableBundle` = immutable compiled operations/commands, `StepPlan` = a runtime work assignment. Never mix them into a single mutable global "execution plan."
- `TaskCoordinator` = authoritative mutable session/commit owner; device workers can only produce provisional completion.
- `ResourceArbiter` = authoritative physical memory reservation and buffer lease origin; no hidden allocations in model-specific modules.
- `ValidationEvidence` = evidence required for an optimized capability to be promoted to default. A source author's benchmark is not our measurement.

Rust crate boundaries are implementation details, not the architecture's definition.

## 2. Full AI model and task space

### 2.1 Semantic model families

Support **known** mathematical operators and model recipes for:
- decoder-only dense/full/sliding-window and attention variants; group-query/MQA/MLA/latent/compressed/selected sparse;
- trained gated recurrent/delta attention and short conv, recurrence with full checkpoint rollback;
- routed/shared expert networks and mathematically faithful top-k router, gated residual and multi-stream hyperconnections;
- independently indexed/token-history lookup, disk-backed immutable table/embedding rows and conditional sparse routing;
- trained MTP/independent speculative drafter/parallel verification, target sampling distribution correctness;
- text+image/audio/video encoders, projections and multimodal positional alignment;
- encoder-only pooling and paired cross-encoder ranking;
- vocoder/audio acoustic generation and audio codec stages;
- diffusion/flow/denoising/temporal video model operators where independently specified and tested.

**User-requested model examples for acceptance** (test-fixture/model-adapter names, *not architectural base classes*): Llama, Gemma 4, Qwen3.5+, Qwen3.8-Flash-Next, GLM-5.3-Flash, DeepSeek V4 family, later compatible model versions. Individual checkpoints may require new semantics, tokenizer versions and quant codecs. Unknown formula or private custom Python model code must be rejected clearly rather than guessed.

### 2.2 Task composition beyond language generation

`TaskProgram` defines typed stages, conditional control, bounded iteration, shared immutable model weights, separate mutable state, resource budgets, and output/commit semantics:
- Text LM: prompt render/prefill/decode or target-verified draft loop; streamed tokens and structured tools/grammar.
- Embedding/rerank: encoder+pool/normalize or query/document scoring without language decoder/sampler.
- ASR: acoustic preprocessing, encoder/chunk/time alignment, model-dependent transcription.
- TTS/voice: text conditioning, acoustic frames and vocoder/PCM/audio stream with speaker/style seed state.
- Image/edit: conditioning and denoise/flow iterative model loop with validated latent/timestep state and image output.
- Video/audio: frame/temporal/AV alignment and staged media output with bounded memory.
- Compound AI apps: retrieving and reranking documents before generation, multi-model or permissioned agent tool orchestrations; external actions are policy-gated rather than arbitrary model code.

Same model, kernel, resource and worker infrastructure; **different TaskPrograms may have different frontiers and stage budgets** (token, audio sample, frame or timestep). A task-type extension is not another inference server.

## 3. Mathematical model, state, numerical and artifact contracts

### 3.1 ModelDefinition/Operator

A validated immutable `ModelArtifact` contains checkpoint/config/tensors/quant/sidecars/license/source hashes. `ModelDefinition` emits SIR graph for exact formulas, typed logical dims, quant/padding and position/sampler, and a per-domain state schema. Novel op gets a portable reference evaluator, independent official fixture and explicit backend capability. Recognizing GGUF/safetensors alone does not make it executable.

### 3.2 Effects and transaction

SIR distinguishes pure SSA values from effects on KV/sliding/compressed caches, gated recurrence, sparse index, token history, RNG/sampler, grammar FSM, modality embeddings and task-local iterative state. A speculative/distributed/cancelled computation can create provisional versions but the authoritative coordinator publishes only when **all required state domains match the same committed frontier**.

Full-state prefix sharing uses immutable snapshot/fork/COW or deferred materialization, versioned model/quant/adapter/tokenized prompt/position/vision identity and a common valid resume frontier across any number of state groups. If compressor windows align differently, choose a legal shorter prefix and recompute missing suffix. A plain prompt-text hash is insufficient.

### 3.3 Precision and quantization

Stored model quant, lossless device packing, temporary compute precision and explicitly requantized checkpoint are **different concepts**. A fast lower precision kernel may be legal in `ValidatedTolerant` mode with layer logits/quality tested but not automatically equivalent to reference. Sampled speculation must preserve target distribution where advertised; approximate/opportunistic modes require explicit opt-in.

### 3.4 Source files

Safe bounded GGUF/safetensors parser, exact tokenizer pre-types/chat/role/streaming/tool and grammar constraints, content-addressed model store/manifest, optional derived acceleration packs and verification of all source bytes. No untrusted executable model code, no guessing unknown architectural tensor layout, no model downloads without policy/permission. If a format or quant unsupported, typed diagnostic and remaining model support retained.

## 4. Optimization, specialization and fallback

### 4.1 General→optimized→extreme tiers

**Reference**: portable Rust semantic operators, correct task loop and all state transitions; potentially slow but usable for source verification and fallback on sufficient resources.

**Optimized generic**: device-aware GEMM/GEMV, attention, quant, grouped MoE, paged KV, caching, batching and direct compiled region execution across supported devices.

**Extreme**: tuned model/operator geometry, shaped quant kernels, sparse selectors and recurrent fusion, entire selected layer/model region in one Rust-authored device kernel or captured graph, with no new engine. Device architecture and execution regime determine real applicability, not a model's marketing label.

An extreme provider is a replaceable `KernelCandidate` satisfying identical logical inputs/outputs/effects/numeric/resource contract; it may use hardware-specific intrinsics and packed physical layouts. If unavailable, lower a compatible region to generic/reference. If state layouts differ, use explicit conversion or replay from a committed canonical checkpoint. Never reinterpret pointers or partially use failed NPU/GPU results.

### 4.2 Compiling resource/timepoint scheduling

RIR can fuse and partition *pure* math, KIR emits kernels, and CIR places them with:
- allocation/lease/alias liveness;
- GPU/CPU/NPU and IO submit/wait/signal;
- pinned-buffer imports and safe cross-driver synchronization;
- SSD expert/n-gram reads and prefetch;
- coalescing and memory staging;
- event-backed state transaction and distributed collective barriers;
- graph capture where stable/memory safe, eager fallback if faster.

The result is an immutable `ExecutableBundle` with shape/dtype/driver/quant/numeric/regime validity and dynamic parameters bounded; the token hot path never walks SIR/RIR/KIR/CIR.

### 4.3 Heterogeneous compute patterns

**SplitProjectionCompute:** independently compute GEMM output-row spans on different accelerators (e.g. integrated GPU and NPU) while respecting output indexing, quant, precision and first-use join. NPU device program and pinned memory must be hardware/safety qualified. Works only in eligible phase/shape regimes; GPU-only elsewhere.

**AsymmetricExpertOwnership:** selected MoE experts executed across devices with different memory speed/capacity; route/weight semantics unchanged, small activations/results cross links, CPU/SSD stage optional. Distinct compute ownership from weight residency and session state ownership.

**Other placements:** contiguous/paged cache, layer/tensor/expert parallel, prefilling/decoding split, CPU/GPU hybrid, multi-host scheduling. The optimizer chooses by measured critical-path/memory/precision/fairness, not hardcoded assumptions that NPU is faster or two GPUs double throughput.

### 4.4 Runtime adaptation boundaries

Offline compile/tune kernel variants with independent source numerics, full-region and whole-request benchmarking; cache evidence. Online request scheduler chooses among approved variants and may adjust batch size, prefill quantum, expert cache or speculative length **only at safe boundaries** with hysteresis. No unbounded graph rewrite/JIT or arbitrary quantization changes during interactive token/frame output.

## 5. Physical resources and session/worker architecture

One `PhysicalResourceArbiter` inventories actual DRAM, GPU-visible addressability, pinned staging, device allocator limits, cgroup headroom, page cache, KV/SSM/indexer state, draft/adapter weights, expert slabs, SSD lookup/cache, vision/audio encoders, scratch and graph programs. GPU virtual GTT is **not** available physical RAM.

Specialized caches share a budget, not necessarily an eviction policy:
- immutable dense/shared weights: resident where profitable;
- sparse experts: fixed slabs, device pointer/generation table, GPU route hit path, async IO miss;
- giant lookup tables: disk-only row source, bounded staging/dedup;
- prefix state: versioned immutable full-state snapshots, paged or contiguous KV, physical COW;
- execution scratch: lease-backed reuse only after quiescence;
- media encoders/decoder weights and task frames: separate lifetime and admission accounting.

A single logical session writer owns `TaskSession` frontiers and service request queue. `EngineMsg/EngineEvent` and `StepPlan/WorkerDelta` are internal typed protocols; device workers own target context, kernel handles and stream events. Dataflow can overlap while effects allow, but no worker commits output to client without coordinator. GPU route results should stay device-side on resident hot path.

**Task-aware scheduling:** token budget is only one kind of budget. Audio frames, image denoiser iterations, video frames and encoder rows use typed stage work accounting, with real physical resource/cost fairness. Low-latency and throughput modes have different work aggregation. A multi-client HTTP queue is not necessarily native GPU batching, and mode receipt must report actual behavior.

## 6. Multi-device/host and failure model

Single-device path must remain lean. Multi-accelerator topology describes instruction ability, tensor memory, shared power/fabric domain, transport latency/bandwidth and verified safety; planner can choose stage-specific compute owners, storage residency, peer collectives and transfer/lease schedule. Remote/TP/EP/PP introduces handshake of model/quant/plan/state epoch and **atomic common frontier**, otherwise fail closed. No remote auth/transport exposure by default.

Async failures: IO short read, stale expert slot, device reset, cross-API callback timeout, memory exhaustion, partial speculative target or wrong peer hash all invalidate dependent provisional task work. Leases only released after quiescence or quarantined; a software timeout doesn't prove DMA has stopped. On irrecoverable host/hardware risk, refuse that accelerator plan instead of attempting runtime recovery.

Hardware safety: if a GPU+NPU fabric/power transition is known to cause potential machine hangs on unqualified firmware, default NPU OFF. No silent privileged `sudo`/clock/BIOS/IOMMU/systemd changes or benchmark runs on a critical workstation. This is a general `DeviceSafetyProfile` capability rule, not a model-specific special case.

## 7. Technology extension seams

| New requirement | Necessary additions | Code that must NOT need alteration |
|---|---|---|
| New model checkpoint with existing math | artifact/tensor mapping + ModelDefinition + source fixtures | device executor, HTTP, scheduler, caches |
| Novel recurrence/sparse operator | reference SemanticOp, StateDomain schema, generic lowerer and tests | unrelated models/API/device providers |
| Novel quant format | QuantCodec + packing/source provenance, numeric reference | task/orchestrator, unrelated models |
| New GPU/NPU architecture | DeviceCapabilities + KernelProvider + buffer/event backend + proof | ModelDefinition, tokenizer, service |
| New hardware transport/storage tier | CIR commands/ResourceTier/Topology + fault tests | model math and task semantics |
| New task type (ASR/TTS/image/video/other) | TaskProgram stages/control/outputs + needed new operators | existing LM server and math/quant |
| New speculative algorithm | DraftSource + Verify/Commit semantics + resource policy | base model math, decoder API |
| New kernel fusion/autotuner | Region rewrite/candidate with legality/quality/bench | model source or users' chat template |
| New API frontend | TaskRequest mapper, auth/schema/stream adapter | KernelProvider and execution plan |
| New adapter/LoRA | per-request weights and cache/plan key, tensor operator | full independent runtime fork |

**Mandatory extension drill:** add one novel attention/state operator, one target device kernel, and one non-token TaskProgram using the same backend without changes to unrelated modules. Record actual source changes and compile/perf overhead; if new specialization requires cloning server/kernel graph/allocator, the architecture failed its core purpose.

## 8. Product and SDK surface

Candidate commands (not final UX promises): `run`, `serve`, `models`, `inspect --explain-plan`, `diagnose`, `bench`, `convert`, `quantize`, `tune`. One Rust SDK and native binary, CPU-only build doesn't require GPU compiler or model download. API adapters for chat/completions, tool/grammar, embeddings/rerank and separate media tasks only when conformance and model checkpoints work. Text/audio/video/image output cannot be represented as one untyped string. Credentials, prompts, tool data and stored prefixes remain private by default.

`inspect` reports model artifact identity, semantic support, missing op/kernel, physical fit and headroom, selected task/plan/backend, numeric mode, kernel/capture variants, cache, readiness and fallbacks; `ModeReceipt` per task logs actual path without user secrets. Unsupported/unqualified device is a documented fallback, not a silent quality mode change.

## 9. Phases, evidence, licensing and authority

The original implementation phase IDs **ZG-P00…P23** remain in `IMPLEMENTATION_ROADMAP.md`; the integrated Stage-0 gate program retains optional hardware/composability experiments. The overarching dependency order:

**Foundations:** physical kernel compiler feasibility, source format/tokenizer/quant reference, typed SIR/effects, minimal legal RIR+KIR and CIR buffer/event simulation, engine worker protocol, one complete real model CPU+GPU vertical slice with forced specialization fallback.

**General runtime:** paged and full-state KV/prefix, async request scheduling, basic multi-session service, ordinary models and task-specific encoder/pooling.

**Specializations:** hybrid sparse/recurrent/MoE, SSD tiers, heterogeneous GPU/NPU, speculation and many-client/native grouped serving, calibrated full model quality. NPU risk/optional compiler policy cannot block portable path.

**Breadth and scale:** adapters, ASR/TTS/image/video TaskPrograms, distributed and prefill/decode split, configurable kernel-provider hardware targets, security/release and full comparisons.

**Long-term evolution:** versioned operator/device/task schemas, transparent migration/fallback, negative optimization experiments and extension drill repeated for each new model/device class.

Every gate requires actual commands/test fixtures, source/checkpoint hashes, numerical/quality/provenance, physical device/driver/memory safety, matched throughput/TTFT/P95 and known failures. One external source author's benchmark is evidence for an experiment, **never a zerogravity achievement**.

**Licensing/naming policy (user U-017):** core architecture must use **original mechanism/contract terminology** rather than name another inference project as its architectural base. MIT-licensed sources may be referenced as optional concrete implementations with exact file-level notices/attribution, but *not* as dependencies of model/runtime identity. Other licensed work remains explicit cited prior art in `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`, not normative project-branded layers. Source code reuse requires source-language policy and license audit. The old named master plan is preserved for provenance in `PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md`.

## 10. Non-goals and stop conditions

- No "accept any arbitrary model file" claim for unknown semantics.
- No separate model-specific graph engine or per-model HTTP/allocator/scheduler clones.
- No mandatory foreign inference server, unreviewed device program or hidden kernel license transition.
- No always-on online graph search, compile explosions, hot-path AST interpretation or implicit device sync.
- No silent low-bit quantization/selected expert count/attention topology/sampling distribution change.
- No universal claim that multiple accelerators or NPU yields faster or more energy-efficient execution.
- No automatic privileged power/firmware/driver tweaks or physical memory overcommit.
- No acceptance of unvalidated external benchmark numbers as internal performance milestones.
- No large plugin ecosystem or speculative 25-crate tree ahead of one tested end-to-end vertical slice.
- No system-wide task workflow sandbox beyond explicit model inference unless external-action authority granted.

A failure in Rust GPU/NPU codegen does **not** invalidate semantic/task architecture: pause that extreme device profile, keep portable execution and record feasibility proof. A kernel or scheduler policy that does not deliver repeatable whole-request gain loses to a simpler correct implementation.

## 11. Final authority

**Read-first:** this master; `../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`; `../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md`; `DECISION_REGISTER.md`; `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`; `IMPLEMENTATION_ROADMAP.md`; the relevant subsystem specs. Exact prior art/benchmarks/license tables reside in separate research files. The historical v2 master and old named handoff remain for traceability **not as independent architectural authorities**.

**Source of truth when implementation begins:** live code/tests and actual hardware/model benchmark evidence supersede stale assumptions. User decisions supersede proposals; this v3 design is the agreed direction for refinement, not a false claim that all implementation/library/licensing decisions have already been approved.

**Success criterion:** a developer can add a new model, task, accelerator, kernel, quantization or storage tier by implementing its own validated contracts and using the existing runtime spine—*without writing another inference engine*.

---


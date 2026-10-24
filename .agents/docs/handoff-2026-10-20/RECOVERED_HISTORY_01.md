# Recovered History — volume 1

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/prior-art/DESIGN_EVOLUTION_V3_HANDOFF_SNAPSHOT.md

Original UTF-8 bytes: 20500; SHA256 `afd369770e8a1c7f64c79b1574219b53190dc0c35b63325d1852bcdfb1ff3bb3`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Design evolution archive — complete v3 agent handoff

**Non-normative historical snapshot archived 2026-10-12**, preserving the comprehensive original v3 takeover text with decisions, prior-art research and older execution contracts before the fresh v4 critical review. Current authority: `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`, `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`, `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`, current v4 master and decision register. In particular v3 StateDomain::commit/TaskNode/4-enduring-IR and single-owner physical-allocation ideas are superseded by v4.

---

# ZEROGRAVITY — FULL COMPREHENSIVE TAKEOVER HANDOFF v3 (2026-10-12)

**Project:** zerogravity — an original Rust-authored **modular AI task compiler, execution engine and serving runtime**.  
**Canonical root:** `/home/pvl/spaces/abird/src/zerogravity/`.  
**Evidence corpus:** `/home/pvl/spaces/abird/research/inference/`, **read-only** unless the user separately authorizes edits.  
**State:** COMPLETE architectural research/planning only; **no Rust engine, Cargo workspace, compiled GPU/NPU device code, model execution, physical test, benchmark, server, Git repository/commit or release exists**. All ZG-P00…P23 implementation phases are NOT_STARTED.  
**The latest user directive U-017:** express all normative plans in original, project-neutral capability terms; prior projects are explicitly attributed **prior art**, with source-specific direct references outside research retained only as needed for verified MIT-licensed sources. Do not turn someone else's implementation into the engine's parent architecture.

**Read this handoff in full before editing code.** Historic source-labeled takeover handoff (including all original research decisions and summaries) is preserved in `PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md` and is *not* the normative v3 design.

---

## 1. Precise user requirements and their evolution

The conversation began with a request for a new Rust-native runtime that could combine **general model support and serving** with **extreme hardware/model specialization**, particularly on AMD Strix Halo `gfx1151`. User targeted broad families including Llama, Gemma 4, Qwen3.5+, Qwen3.8-Flash-Next, GLM-5.3-Flash and later models, and wanted to learn from leading inference engines without maintaining incompatible model-specific runtimes.

The user explicitly endorsed **Semantic IR** as an architecture direction, subject to reality: it enables transformation and portability, not automatic performance superiority. They named the project **zerogravity**, requested exhaustive plans/handoffs, corrected canonical path to `/home/pvl/spaces/abird/src/zerogravity/`, and asked us to compare and combine an extensive existing source-analysis corpus with the new designs.

Subsequently the user demanded comprehensive fresh research on **long-term adaptability at every pipeline stage** (models, codegen, memory, scheduling, runtime, application, hardware) and incorporated new source demonstrations of (a) asymmetric compute/weight ownership across discrete and integrated accelerators and (b) **independent same-SoC GPU+NPU output-row GEMM cooperation** with strict hardware-safety constraints. They also requested speech, media and iterative AI pipelines to compose without using fake text-decoder loops.

Finally, **U-017** directs us to generalize all plans away from external project names unless those projects are MIT licensed; source research must remain cited and properly licensed, and the **architectural expression must be ours**. They want a composable AI system that can add new specializations *without writing new engines each time*.

User decision register `DECISION_REGISTER.md` is authoritative for exact historic IDs (U-001…U-017); design recommendations are P-xxx and unresolved product/technological decisions O-xxx. Do not infer that an old source project's license, API, compiler version, speed threshold or kernel language was user-approved.

## 2. Required reading and source-of-truth order

1. the agent playbook and **`DECISION_REGISTER.md`**.
2. This handoff, the rewritten **`ZEROGRAVITY_MASTER_PLAN_2026-10-10.md` v3**.
3. **`../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`** and **`../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md`**.
4. `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`: verified repository root license/NOTICE caveats and prior-art taxonomy; names/data used only in research, not core types.
5. `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`, `IMPLEMENTATION_ROADMAP.md`, `EXPERIMENT_PROGRAM.md`, `RELEASE_ACCEPTANCE_GATES.md`.
6. Detailed subsystem specs for the current task in `../docs`, especially `SEMANTIC_IR_AND_COMPILER.md`, `ASYNC_RESOURCE_COMMAND_IR.md`, `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`, `STATE_CACHE_AND_TRANSACTIONS.md`, `MEMORY_RESIDENCY_AND_IO.md`, `DEVICE_KERNEL_AND_CODEGEN.md`, `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`, `SECURITY_AND_RELIABILITY.md`, `INVARIANTS_AND_FAILURE_SEMANTICS.md` and `NUMERICS_QUALITY_AND_BENCHMARKS.md`.
7. For an external algorithm/benchmark, search the **research-only prior-art** files and source revision URLs or the read-only `research/inference` tree, not the normative architecture for an external project's type names.

**If the current checked code and measured tests conflict with a plan**, validate the live code/model and update the plan—never assert a phase completed based on a historic roadmap sentence. Do not automatically run `git status` in a folder with no Git; check existence first.

## 3. Original, composable system architecture

**One execution spine**:

```text
Source Manifest / Artifact / Tokenizer / Media Encoder
                       |
                ModelDefinition
                       |
             Semantic Model IR (SIR)
              typed math + state effects
                       |
             Optimized Region IR (RIR)
           legal fusion/packing/row splitting
                       |              \
                       |             Kernel Providers (KIR)
                       |              portable to extreme Rust code
                       |             /
             Command/Resource IR (CIR)
            async buffers/leases/IO/streams/events
                       |
             Compiled ExecutableBundle
                       |
    Topology/Device Capabilities + Physical Resource Arbiter
                       |
  Typed TaskProgram → TaskCoordinator/EngineCore → StepPlan
                       |
               Device/IO Workers
                       |
             Provisional state/events
                       |
      Atomic TaskSession transaction and result/stream
```

**Independent dimensions of capability composition**:
- **Model mathematics:** typed source operators, tensor/quant/position, recurrence/attention/router semantics, full reference oracle.
- **Task control:** text autoregression/speculation, encoder-only scores, ASR, TTS, image/video iterative stages, multimodal/agent workflows with different committed frontiers.
- **Device kernels/codegen:** portable CPU/GPU and optional hardware-specialized whole-region kernels; selection by op/shape/precision/safety/evidence, not model names.
- **Resource placement/IO:** immutable weights, expert slots, KV/recurrent/latent state, n-gram lookup rows, GPU/NPU pinned buffers, SSD/remote; all leased under one physical budget.
- **Runtime/orchestration:** TaskCoordinator owns sessions and scheduling; StepPlan/WorkerDelta are work commands, not whole model data; output only after commit.
- **Product/service:** canonical TaskRequest/TaskResult/ModeReceipt, independent SDK/HTTP/CLI/media output adapters, model store and permissioned external actions.

An optimized implementation may replace any **closed semantically equivalent execution region** with a direct Rust-authored fused device kernel or captured graph if full values, state effects, numeric precision and resource dependencies are validated. A missing optimized kernel uses portable/generic region execution or canonical state conversion/replay; a new kernel must never require a copy of the whole server. Unknown mathematical operators are not implicitly executable from a checkpoint format.

**Do not conflate** `KernelPlan` (validated candidates), `ExecutionPolicy` (device/resource/quality constraints), `ExecutableBundle` (compiled commands) and `StepPlan` (actual mutable scheduling cycle). One massive object binding all four would impair maintainability.

## 4. Why task composition is a core requirement

AI does not mean only autoregressive next-token models:
- Language generation has causal token frontier, KV/recurrent/indexer state, sampler and potential draft verifier.
- Embeddings and rerank models have batched encoder/pooling/score and no language decoder.
- ASR has acoustic sample/window/stream state, preprocessing, encoder and transcription.
- TTS has text conditioning, acoustic frames/tokens and vocoder PCM with speaker/voice state.
- Image/editing has conditioning and iterative diffusion/flow/denoiser latent steps before image output.
- Video/audio uses temporal/frame state and codec/output-mux behavior, potentially multiple model programs.
- Compound RAG and agent tasks compose permitted retrieval/ranking/generation/other TaskPrograms, with a security boundary around external tools.

Every task exposes typed `TaskProgram` stages (`Sequence`, `Parallel`, `Iterate`, `Select`, `Window`, `VerifyAndCommit`, `Emit` and optional permissioned `ExternalAction`). They share model Semantics/KIR/CIR/memory/DeviceWorker but may use different scheduler **work units and stream commit frontiers**. Implementing ASR/TTS/image/video is optional until a model/task quality matrix passes; the design supports them independently without pretending they already work.

## 5. Specialized parallelism without engine forks

- **Region fusion/layout:** normal/quant/projection, sparse selector+attention, recurrent update, routed expert gate/up/down, LM head/sampler, encoder/vocoder/denoiser as legal.
- **Routed expert ownership:** tensor identity/route top-k unchanged; hot vs cold experts computed on different GPU/CPU/IO owners; activations and partial results transferred under bounded leases and event ordering.
- **Intra-operator row partition:** independent projection outputs can run on GPU+NPU/other eligible devices while the GPU progresses on unrelated work; CIR first-use join and mathematical row/quant coverage are mandatory. NPU block-float arithmetic may change numerical results, so source numeric/quality policy applies independently.
- **Model/device/phase placement:** single CPU/GPU, partial offload, layer/tensor/expert parallel, prefill/decode split, future heterogeneous remote topology with authentication and state consistency.
- **Speculative target execution:** trained/native or independent draft and prefix lookup, exact target-distribution correction by default where implemented; full state/RNG/grammar rollback for rejected proposals.
- **Prefix reuse:** full hybrid-domain state snapshots across 3+ different attention/recurrent/indexer/media state types; one common legal resume frontier and immutable sharing/COW.

Every physical accelerator profile includes compute capability, memory/links and **safety/firmware/fabric**. A potential GPU+NPU whole-host hang is a hard qualification veto until isolated physical testing; no automatic root/sysfs/clock/BIOS/IOMMU changes. GPU-only reference remains usable. Driver runtime FFI and owner Rust AIE/kernel source are separate policy questions; wrapping foreign source is not automatically a Rust kernel.

## 6. Source formats, provenance and licenses

Safe GGUF/safetensors, tokenizer/chats/template/structured tools/grammar, content-addressed immutable artifact store, optional packer/quant/calibration with separate source and compute-precision identity, adapters, sidecars, model/vision/audio manifests. Format support requires verified loader **and** registered semantics, sufficient resources and task output support.

**Source naming and reuse policy:** no architecture type or phase depends on an external engine's project name, even if its MIT license permits study/reuse. Verified MIT source is a *candidate* to reuse after per-file header/NOTICE, model data/device executable and Rust-source policy checks. Non-MIT source retains name only as prior-art evidence unless explicitly authorized. Proprietary/nonpublic code cannot be copied. Licenses for model weights, trained derivative drafters, metrics/input datasets and GPU/NPU device programs are distinct from engine repository root licenses.

**Source evidence preserved** (do not erase): `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` with verified license-text blob SHAs, `PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md`, `PRIOR_ART_HISTORICAL_ROADMAP_V2.md`, `PRIOR_ART_HISTORICAL_STAGE0_V2.md`, `PRIOR_ART_HISTORICAL_EXPERIMENTS_V2.md`, `PRIOR_ART_HISTORICAL_ACCEPTANCE_GATES_V2.md`, `PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md`, and dedicated source/hardware research analyses. External research tree stays read-only.

## 7. Physical memory, safety and async ownership

One `PhysicalResourceArbiter`, specialized subpools for immutable always-on weights, routed expert slots, paged or contiguous KV/state, frozen prefixes, lookup rows, NPU/GPU pinned staging, scratch, encoder/media and optional remote buffers. One global budget and OS/cgroup headroom, not a universal cache eviction policy.

Every async device/IO/peer operation reads/writes through a versioned BufferLease and timepoint, with explicit alias and ownership. A failed or timed-out DMA may still be accessing memory; quarantine until guaranteed quiescence or device reset. No reuse based only on a software timer. GPU addressability/unified virtual GTT is not free physical RAM.

One session owner commits all state effects atomically; device worker never publishes externally visible text/audio/image/video before TaskProgram's output validity rule. Cancellation leaves last complete emitted frontier and aborts uncommitted work. Distributed peer partial computations cannot masquerade as successful model state.

## 8. Correctness and performance expectations

Benchmark model/task-specific **semantics, numeric agreement, task quality and state transactions separately**. For source model checkpoint/quant/source hardware/driver and workload match:
- logs of teacher-forced layer activations/full-vocab logits and effects;
- target stochastic distribution and draft rollback where available;
- task-specific output quality (text, embedding/rerank scores, ASR audio alignment, TTS, image/video);
- physical resource headroom, cold/warm IO/expert cache and GPU/NPU event/overlap;
- TTFT, P50/P95 latency, committed useful outputs/sec (correct unit), energy and concurrency;
- compiler/driver/library/shape/quant/numeric plan provenance, actual ISA occupancy;
- ABBA/BAAB and raw uncertainty with a portable baseline and external peer when genuinely comparable.

**Never** claim universal speed lead based on architectural elegance, a microkernel, incomparable model quant, reduced trained MoE top-k, altered sampling, or unmatched power clocks. A complex scheduler/kernel candidate that loses whole-task latency or quality is disabled; negative experiments remain recorded.

## 9. Roadmap and change control

Stable project phases **P00–P23** in the rewritten `IMPLEMENTATION_ROADMAP.md`. Stage-0 S0-00…S0-14 plus optional NPU S0-NPU-A…F in `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`; experiments H01–H31 in `EXPERIMENT_PROGRAM.md`; detailed release criteria in `RELEASE_ACCEPTANCE_GATES.md`.

**Start smallest full vertical slice:**
1. Lean Rust CPU build, trusted artifact parser and tokenizer/quant reference;
2. basic math/state SIR and CPU oracle, effect verifier and TaskProgram loop;
3. minimal RIR region/legal fallback, simple KIR kernel provider;
4. CIR buffer/event/IO simulator proving leases and aborts;
5. one real physical GPU device backend, direct compiled bundle and actual model result through TaskCoordinator;
6. one synthetic non-token task and second operator/device extension drill proving no duplicate engine;
7. only then advanced MoE/quant/sparse/heterogeneous/spec/graph capture/distributed and media tasks as requested/feasible.

NPU hardware safety/source-language feasibility is optional; no delay to general CPU/GPU release if blocked. No 25 empty crates or general-purpose optimizer before the vertical slice. User specifically asked for **plans** here, not code or benchmarks.

Each code phase completion must append actual changed files/test commands/driver/device/quant/source SHA, numerical/task quality, physical memory, P95 and known failures to `CURRENT_STATUS_AND_TRACKING.md`, and update decision register, relevant specs and checksum manifest. Never rewrite historical decisions to make an experiment seem accepted.

## 10. Open policies that require future evidence/user input

Project license and binary name, exact Rust GPU/NPU source-language/vendor math FFI boundaries, mandatory first device/model/task/quant release matrix, public SDK/HTTP transport and authentication, distributed trust default, C ABI/plugin model, deployment OS/CI hardware, default SLO and numerical tolerance, device clock/fabric safety qualification, proprietary device program rights, model adapter DSL choice and artifact/cache serialization upgrades. Detailed O-xxx entries in `DECISION_REGISTER.md`; don't fill silently.

No need to re-ask the user about project name/location, Rust-owned core, semantic IR, broad modern model families, integration of prior art, or modular composition goal—they are explicit already.

## 11. Agent startup prompt for direct reuse

```text
You are taking over the zerogravity Rust modular AI execution engine at
/home/pvl/spaces/abird/src/zerogravity.

Read in full:
- .agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md
- .agents/plans/DECISION_REGISTER.md (especially U-017)
- .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md (v3)
- .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md
- .agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md
- .agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md
- .agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md
- .agents/plans/IMPLEMENTATION_ROADMAP.md
- .agents/plans/EXPERIMENT_PROGRAM.md
- .agents/plans/RELEASE_ACCEPTANCE_GATES.md
and all relevant state/kernel/CIR/task/resource/security/numeric subsystem specs.

Inspect the live project before relying on any completion status; at creation
there is no Rust implementation or Git repository. Treat the separate
/home/pvl/spaces/abird/research/inference directory as READ-ONLY prior art.

Architecture is ORIGINAL, project-neutral and capability-composable:
Artifact/ModelDefinition -> effect-aware Semantic IR -> legal Region IR ->
Rust KernelProviders -> async Command/Resource IR -> immutable direct
ExecutableBundle -> single TaskCoordinator/EngineCore + physical resource
arbiter -> typed TaskProgram/session transactions and streaming results.

Add a new model/operator/device/quant/media TaskProgram through its own
validated interface, never by creating another specialized inference server.
Preserve reference math, full mutable state, resource leases, numeric/sampling
quality and safe fallback. Names of non-MIT external projects appear only in
prior-art research/provenance, not architecture types; MIT code reuse still
needs file-level notice and Rust-source review.

If the current user request is design-only, modify docs/decisions/manifest
only. For implementation work start a tiny Stage-0 real model/task vertical
slice, test physical device+quality+memory and record actual evidence.
Never claim performance or a phase completed merely from documents.
```

## 12. Closure of this handoff

This is the complete operational context for any subsequent autonomous engineering agent. It identifies the original product goals, strict source/licensing distinction, model/task/semantic/kernel/resource/serving interfaces, prior research location, phase gates, physical memory and hardware safety, experiment methodology, pending decisions, and exact next-step discipline.

**Our defining promise is not "we copied the best engine"; it is "we can add the best verified implementation of any supported semantic region, on any qualified hardware, without duplicating the rest of the AI system."**

````

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/prior-art/DESIGN_EVOLUTION_V3_SNAPSHOT.md

Original UTF-8 bytes: 80189; SHA256 `01b06c809bb810b788cbf63a8fe45a58aae0fbcec3ab340f1ab7b2e9409d40d1`.
Resolve embedded relative paths against the original path, not this volume.

````text
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


````

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/prior-art/PRIOR_ART_HISTORICAL_ACCEPTANCE_GATES_V2.md

Original UTF-8 bytes: 11535; SHA256 `82c90febb499eb7d2d86bb553d3a41ac8a5df48df2526aa36dd7450b935755e8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Prior-art/historical archive — RELEASE_ACCEPTANCE_GATES.md

**Historical v2 planning snapshot, archived 2026-10-12. NON-NORMATIVE RESEARCH.** External named projects and benchmark figures are preserved as source research, while actual current design and phase gates use project-neutral contracts at `../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`, `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`. User-provided instructions and decisions remain separately in DECISION_REGISTER. Never use historical source-project APIs as the new runtime's type or module names.

---

# zerogravity — Acceptance, verification and release gates

**Scope:** binds every proposed roadmap phase to objective, independently repeatable evidence. This document is a proposed governance standard, not an assertion of passing tests.

## A. Project lifecycle gates

| Gate | Earliest phase | Required evidence | Failure |
|---|---|---|---|
| G0: source/toolchain | P00 | fresh checkout builds, pinned tools, attribution and decision log | cannot reproduce |
| G1: kernel feasibility | P01 | real physical `gfx1151`, numerical references, disassembly/profiling, documented performance gap | no competitive Rust codegen path |
| G2: semantics/state | P02–P04 | CPU reference with tensors+effects, shape errors, snapshots and rollback | failed oracle/replay |
| G3: compiler legality | P05 | differential graph transformations, no incorrect reorder, fallback transitions | changed semantics / bad alias |
| G4: real model | P06–P08 | GGUF/safetensors and tokenizer parity, end-to-end dense outputs, CPU/GPU comparisons | plausible but wrong tokens |
| G5: sessions/serving | P09–P10 | streaming/tool/integration tests, memory safe, persistent context | hidden state or OOM corruption |
| G6: hybrid/MoE | P11–P12 | long-context GDN/KDA/indexer parity, route/cache miss tests | state corruption/unsafe eviction |
| G7: Strix specialist | P13–P14 | real Qwen/GLM reference and matched comparison, cold/warm | no model/quality proof |
| G8: speculation | P15 | rejected-prefix/RNG distribution, latency and benefit, adversarial stop | target mismatch |
| G9: high throughput | P16 | concurrency, fairness, p95 and no sequence cross-talk | stall or state leak |
| G10: breadth/scale | P17–P19 | per-model feature matrix, multi-device failure recovery | claims unsupported |
| G11: release | P20–P22 | signed provenance (policy TBD), performance and quality reports, security audit, QA | regression/critical risk |

## B. Correctness corpus levels

**Level 0:** parse/graph/operator shape/type/canonical reference with unit + property tests. **Level 1:** GPU microkernels against CPU FP32/FP64 as appropriate, quantized block exact decode, output tail, masks and invalid selectors. **Level 2:** synthetic complete model with layer-wise tensors/logits and state transitions. **Level 3:** official checkpoint and known tokenizer/template, 100+ held-out continuations and long-context adversarial fixtures. **Level 4:** HTTP/agent lifecycle, tools/reasoning, concurrent sessions, serialization/restore/cancel/VRAM exhaustion. **Level 5:** distributed two-rank partial failure and integrity.

A source project may report exact-logit equivalence to its own previous version; that is not automatically parity to an official HF model. Test both. Use same quant/artifact and normalized template whenever comparing runtime. Never claim quality preserved by small one-token agreement tests alone.

## C. Mandatory validation for execution plans

For every plan region record: semantic op IDs, value/state input/output types, tensor shape/strides, dtype & quant zero point/scales, allowed alignment, causal position, scratch bytes, all device/host leases, graph-capture flag, in-flight events, numerical policy, supported max batch/context and fallback plan. Programmatically reject violations. Test forced operator fallback and interrupted graph rollout (host/device failures) at legal commit boundaries.

## D. Resource safety minimum

- preflight total resident/model dense bytes + KV/SSM/indexer + optimizer and graph scratch + pinned staging + page/memory headroom;
- hard reservation per admitted request; dynamic scratch borrowed only via lease;
- device code never accesses expert slot after its generation changed;
- detect physical memory underestimation and bail safely before OS OOM where possible;
- user is warned before opt-in kernel/firmware memory configuration changes;
- corrupted SSD read invalidates request/state frontier instead of returning stale data;
- model-unload and server restart free or intentionally reuse all allocations;
- no unauthenticated public network exposure default.

## E. Serving service-level verification

For each supported model/hardware report max context, maximum admitted concurrency, typical TTFT P50/P95, intertoken latency P50/P95, aggregate tokens/s, peak RSS/HIP allocated, startup warm/cold, memory fit and quality contract. No universal SLO numbers defined before physical benchmarks; they are *open decisions*. Fault tests must cover malformed requests, stream resets, cancellations, tools, reasoning boundaries, slow clients, token limits and invalid cache files.

## F. Report schema recommendation

```json
{
  "project": "zerogravity",
  "result": "pass|neutral|fail|inconclusive",
  "phase": "ZG-Pxx",
  "date_utc": "...",
  "commit": "...",
  "model": {"id": "...", "artifact_sha256": "...", "quant": "...", "tokenizer_sha256": "..."},
  "hardware": {"gpu_arch": "...", "ram_bytes": 0, "driver": "...", "power_limit": "..."},
  "runtime": {"rustc": "...", "compiler": "...", "backend": "...", "plan_hash": "..."},
  "workload": {"prefill_tokens": 0, "prefix_depth": 0, "output_tokens": 0, "concurrency": 1, "sampling": "..."},
  "performance": {"ttft_ms_p50": 0, "itlat_ms_p95": 0, "prefill_tps": 0, "decode_tps": 0, "aggregate_tps": 0},
  "quality": {"reference_id": "...", "logit_rms": 0, "greedy_match": "...", "sampling_test": "..."},
  "memory": {"rss_peak_bytes": 0, "gpu_peak_bytes": 0, "expert_hits": 0, "ssd_bytes": 0},
  "repetitions": 0,
  "raw_artifact_paths": [],
  "known_limitations": []
}
```

The JSON above is an illustrative proposed schema and not a filled result. Unknown fields must be null/omitted in real reports, not misleading zeros.

## G. Documentation handoff closure

At phase finish update `START_HERE` references if changed, roadmap status, decision register with new decisions, relevant `.agents/docs`, benchmark evidence, risk register, and `MASTER_HANDOFF`. Verify live tree state and final test commands; cite exact files and commits. Agents must never say phase done without an executable proof and list of known blockers.

## 2026-10-10 integrated v2 acceptance gates

**New G0 baseline:** `G0-A` source/CPU artifact/math/state reference; `G0-B` physical gfx1151 Rust kernel feasibility and CIR event/resource safety; `G0-C` complete checkpoint→SIR→RIR→KIR/CIR→EngineCore→correct token vertical slice with forced per-region fallback. All are documented in `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`. Historical pure G0 kernel and tokenizer tests remain necessary, but a giant precompiled crate skeleton alone does not pass.

**Stage-specific proofs:**
- Compiler: illegal effect rewrite rejected, quant+position typed, no hidden stateful op removal; compiled bundle must not interpret SIR per output token.
- Device: Rust-authored kernel and target ISA reference/quality, bounded compile/JIT, Capture/Eager numeric/scratch/event compatibility, safe fallback on unavailable specialized region.
- CIR: async buffer lease generation, IO/transfer timepoint, short read and device loss abort; no in-flight expert overwrite, resource budget violation or sample/recurrent partial commit.
- State: 3+ hybrid groups cached frontier, full canonical immutable prefix, disk snapshot schema/hash (not broken original 48/54 byte), opt-in privacy and model/quant/tokenizer/adapter/modalities/salt identity.
- Serving: EngineMsg/StepPlan/WorkerDelta epochs, native vs ordered batching, cancel/stop/tool/grammar, no output ahead of committed state, p95 under C1..C32.
- Product: tokenizer, Jinja template, grammar, ModelStore, embedding/rerank runner, LoRA and multimodal only marked supported where conformance passes.
- Kernel tuning: eager-vs-captured actual serving tactic, micro plus whole request and quality band; no unbounded search.
- Distributed: peer model/state epoch handshake, partial collective fail-closed, no unauthenticated public transport by default.
- Research claims: source pins verified and matched checkpoint/quant/power; no universal superiority promised.

**Two corrected legacy errors:** research disk-KV fields total 54 bytes despite a 48 byte label and magic cannot be five characters in u16. Research line for 8×L40S DwarfStar DeepSeek TP cannot be a measured GLM-class benchmark. These are release/source integrity issues, not optional style edits.

**Release phases:** G1 portable (known dense model), G2 generic serving, G3 Qwen/GLM/DeepSeek extreme, G4 breadth/parallel. SLO percentages from earlier research are historical proposed target floors, **not** user-approved achievements; use meaningful repeated A/B and caveats. No gate marked VERIFIED before repository evidence.


## 2026-10-12 acceptance: optional XDNA2 and TaskProgram device/semantic qualification

**XDNA2 always optional:** no GPU/CPU release phase depends on an NPU or XRT. A hardware-qualified NPU region must pass (A) Rust-owned AIE source/toolchain and license rule, (B) full mathematical projection row-map/quant/compute precision and state parity, (C) CIR GPU↔XRT BO mapped/pinned lease/event/quarantine under cancellation and timeout, (D) controlled physical SoC firmware/driver/FCLK/power stability and long soak without hard hang, (E) same-clock GPU-only vs GPU+NPU model quality and matched full-request performance win, (F) clean GPU-only fallback for absent/misaligned/unqualified accelerator. No amount of generated token speed waives A–D.

The Gufo fork reports real **whole-machine hangs** when fabric clock transitions during concurrent NPU/GPU; **automatic** hardware tuning/activation is **not permitted** based on a handful of successful author runs. Do not call root shell/sysfs/systemd/IOMMU/BIOS changes from runtime. On unqualified hardware mark `NPU unavailable/experimental`, not performance-qualified, and execute GPU-only. A software timeout cannot recover a SoC hardware freeze; physical safety trials isolated and manually consented.

**Numeric/model truthfulness:** source `UD-IQ4_XS` 16K **1698→1821 prefill tok/s** is an external reported result, not reproduced by zerogravity. It uses BFP8 NPU *compute* format, not necessarily bit-equal to BF16/F32 kernel arithmetic. Qualify full logits, GDN/QSA index state, MTP sampled distribution and quality with original checkpoint. Require real model+hardware evidence before `NpuKernelProvider` becomes automatically selected on any profile.

**TaskProgram capability:** ASR/TTS/image/video model families carry their own task state semantics, encoder/acoustic/vocoder/denoise/temporal references, output format and privacy/security; task may be `reference`, `experimental` or `qualified` per checkpoint/backend/quant. Upstream Gufo implementation is evidence only, not proof of zerogravity production support. See `../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md`. All original G0–G4 acceptance gates remain applicable.

````

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/prior-art/PRIOR_ART_HISTORICAL_DISTRIBUTED_PARALLELISM_V4_1.md

Original UTF-8 bytes: 16521; SHA256 `e17ffc8185c21d630e184fd45fd336efe79b29b834867845c257a581164adc86`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Historical distributed-parallelism design before topology v4.2

**NON-NORMATIVE ARCHIVE · 2026-10-12.** Preserves every detailed original layer/tensor/expert and P/D distributed design note from v2–v4.1 as prior art and historical context. The **current normative** specification is `DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md`, with wire/peer detail in `REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md` and source research at `DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md`. Do not adopt older participant-level `ranks commit` examples where inconsistent with TaskCoordinator's single logical PublishRoot and remote state leases.

---

# Heterogeneous devices, distributed execution, topology and recovery

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


All strategies must preserve one logical Semantic IR and one committed session frontier. Distribution is optional: a lone Strix Halo must never pay for network orchestration when it can run resident.

## 1. Placement modes

**Single device:** one CPU/GPU handles model and state; baseline, no collective tax. **Layer/pipeline parallel:** contiguous disjoint layer groups across devices/machines, one activation handoff per stage, good capacity and multi-chunk prefill; cannot automatically speed single-token serial generation. **Tensor parallel:** matvec/GEMM and possibly attention components split over peers, partial sums exchanged; useful when interconnect fast enough. **Expert parallel:** route selected token rows to expert owner GPUs and aggregate contributions; high benefits for MoE at throughput scale but has dispatch/comm cost. **Prefill/decode disaggregation:** separate prefill workers and decode workers with KV/recurrent state transfer, only when service economics justify.

Heterogeneous plans may combine pipeline across CPU/GPU/machines and tensor/expert within a stage. Composition constrained by state and expected communication cost. Not all variants supported at first release.

## 2. Cost and topology model

`TopologyGraph` devices and edges: device vendor/arch, free VRAM/unified RAM, memory speed, compute capacities by kernel shape, PCIe lanes/generation, P2P availability, NVLink/RDMA/TCP bandwidth and latency, CPU NUMA locality, physical host RAM/SSD. Benchmark representative transfers rather than trusting brochure bandwidth.

For each placement, estimate:
- each tensor/quant packed bytes and mandatory model/optimizer state;
- per-stage compute time for prefill/decode/verify, not just FLOPs;
- weights duplicated vs sharded, expert residency and cache misses;
- per-token activation/collective bytes and launch/stream synchronization;
- per-session KV/recurrent state mapping and concurrent memory pressure;
- context length and speculative states that grow per active session;
- failure/recovery and cold startup.

A larger GPU can receive more layers, but load balance requires actual layer complexity (dense vs sparse vs MoE), not equal layer count.

## 3. Layer pipeline

Assign model layers [0..k], [k+1..m] etc with exact model recipe and subgraph boundary. Each worker owns its slice and state for those layers; coordinator owns input tokens, sampling policy and API. Prefill chunks overlap stages when multiple chunks can be in flight and downstream state consistency allows. Decode one token must pass all layers before next target token sampled; pipeline latency often bounded by sum stage timings and links.

Wire data: model/quant hash, session ID+epoch, token frontier, shape and activation dtype, exact stage IDs, optional modality positions, checksums. Compressing transmitted activations F32→F16/INT8 changes numerical contract; explicit opt-in and quality test.

## 4. Tensor parallel

Two ranks may split routed experts by owner and exchange partial contributions every relevant layer; dense and shared tensors may remain replicated. A peer identity handshake validates checkpoint size/hash, tokenizer/config, quant, positional policy, layer/collective schedule, context size, draft and RNG protocol, hardware capability and runtime version. DwarfStar's `ds4_tp.h` is a useful example of a narrow command protocol and GPU-visible RDMA slab separate from frontend logic.

Some workloads favor sharding attention heads/vocab, others experts only; choose based on operation semantics and device link. Prohibit a rank from applying unacknowledged logits or speculative proposals as committed. All ranks agree on exact collective shape, epoch, sequence and model layer gate schedule.

## 5. Expert parallel

GPU-side router produces token-expert assignments; local/remote expert owners process compact row groups; reduce contributions deterministically per configured policy. Network AllToAll/dispatch overhead must be overlapped with on-device shared expert and independent work where safe. For small per-token batch on slow PCIe, routing remotely can be worse than local CPU or expert fetch; admission should choose measured regime. Keep expert source residency separate from session KV locality.

## 6. Prefill/decode disaggregation

Offload prefill to workers optimized for big GEMM, then transfer **all required session state**, not only KV (GDN/KDA/indexer/history/modality state), to decode workers. Transfer cost and incompatible layout conversions may erase benefit. Prefix cache state versions and ABI must agree. This is not guaranteed feature on every model or backend.

## 7. Wire protocol and transport

Candidate transports RDMA (verbs/RoCE/Thunderbolt if supported), authenticated TCP or local IPC. **No unauthenticated remote serving by default.** Even trusted RDMA systems should bind to allowed devices/networks and reject arbitrary ranks. Control frames use versioned types, checked lengths and content hash; data transfers register explicit memory regions with lifetime leases, not persistent unchecked raw pointers. Streaming failure/partial sends must be distinguishable from complete ACK.

Proposed frame identity:
```text
protocol_version, cluster_run_id, model/checkpoint hash, quant/layout hash
source/destination rank, session_id, state_epoch, operation_id
committed_token_frontier, graph_region_id, expected_payload_bytes
sequence_no, request_nonce, checksum/authentication
```

Validate before allocating network payload. RDMA memory keys/addresses are sensitive; protect logs and handle lifecycle. Use end-to-end optional integrity checks for partial sums/activation hashes during QA.

## 8. Distributed state transaction

```text
coordinator selects operation at committed p
ranks verify same model/plan/epoch and reserve buffers
execute subgraphs/collectives, produce provisional state
collect successful completion acknowledgments
coordinator samples/chooses accept-prefix a
ranks commit same p+a and ack
coordinator emits only committed output
```

If any collective fails, rank times out, memory lease invalidates or checksum mismatches, **the operation fails globally**. Peers must not keep generating independently. Recovery requires new consistent route and a canonical committed prefix snapshot or full verified replay. Leader-only restore cannot pretend the worker's recurrent state is restored.

### Idempotency

Each operation tagged by `session_epoch, request_id, op_seq`. Duplicate/reordered frames rejected or explicitly idempotent. No partially completed MoE allreduce can be reused after reconnect. Cancellation generation must propagate and await quiescence. Session locking prevents concurrent distributed writers.

## 9. Multi-GPU single-host handling

On one host, GPU topology can be heterogeneous. Layer splits often best for incompatible cards or thin PCIe; tensor parallel works better with fast peer access; allow CPU spill only when semantic and numeric kernel supports it and measured performance remains viable. A GPU's reported free VRAM is not enough: reserve per-device graphs, cached weights, temporary prefill and sessions before computing placement.

For AMD Strix Halo, no exposed multiple GPUs assumed unless hardware actually enumerates them; single-device path must remain simple.

## 10. Performance and quality evaluation

Bench C1 and C8+ throughput, single-token latency, 8K/32K/128K prompt prefill, network bytes/step, rank memory/OS pressure, cold reload, snapshots and failure recovery. Compare CPU/GPU partition vs two-GPU vs single-GPU where quant/capacity equal; if multi-host enables larger quantization, classify as **capacity improvement**, not fair raw engine speed comparison. Distributed stochastic sampler and quantized collective changes can alter numerical behavior; record per-rank logits and quality.

## 11. Fault-injection and security

Tests: wrong model/checkpoint, mismatched quant, incompatible draft, wrong collective order, interrupted TCP send, RDMA completion failure, slow worker, rank process crash, network partition, cancellation during exchange, stale registered slab, replay duplicated command, out-of-memory on one rank. In every case either all ranks have a valid committed token frontier or execution stops and safely reconstructs from checkpoint.

## 12. Open choices

Collective/RDMA crate, topology discovery stack, stable protocol version, remote credentials/transport security, per-GPU kernel compiler transport compatibility, TP+MTP combined state handling, P/D staging format, checkpoint restore semantics and which modes should be supported at first distributed release. Requires physical hardware evidence before promising support.

## 2026-10-10 integrated v2 addendum — distributed inference is CIR + model state protocol

Research's DS4 TP/RDMA and layer pipeline dataflow is kept, and project transaction rule controls correctness. A distributed model plan is composed of **SIR region partition**, RIR tensor/layout/quant conversion, **CIR** peer transfer/collective/timepoint commands and explicit resource leases; only rank/session identity and accepted frontier are maintained by the coordinator. The kernel/runtime may use RCCL/NCCL or direct RDMA/TCP as a backend policy, but must not require a non-Rust inference server.

Separate data-parallel **replicas** (independent sessions, no per-layer collective) from TP/EP/PP and prefill/decode disaggregation. v1 should retain single-device fast path unless actual hardware demand and tests justify distributed overhead; P19 is later. Research original v1 multi-tier/TP exclusions are scope recommendations, not product limitation forever. Compile topology-specific plan variants only after measuring PCIe/NVLink/RDMA activation traffic and exact buffer formats, not simply choosing "two GPUs = twice throughput".

**State across ranks:** hybrid model requires transfer/restore of KDA/GDN, sparse/compressed indexer, modality and RNG/grammar state—not ordinary KV alone. CIR collective `TimepointId` and rank epoch/generation must all agree before commit, or fail globally/replay from a common validated checkpoint. Buffer lease pin/overrun, stale RDMA keys, wrong quant/checkpoint/semantic hashes, partial capture failure and slow rank must abort, not publish inconsistent frontier. No public unauthenticated TP port.

A new transport (RDMA, PCIe peer, TCP, future shared-memory interconnect) should plug in as `Collective/TransportProvider` with physical cost and safety contracts, not modify model mathematics, tokenizer or HTTP. Add fault-injection and a 2-rank official model parity test before any parallel performance claim.


## 2026-10-12 integration — one-host heterogeneous computation beyond GPU expert owners

The placement contract must distinguish (A) **dGPU↔iGPU routed MoE expert owners** (activation/partial result transfer), (B) **iGPU↔XDNA2 NPU independent projection output-row slices** (the user's Gufo fork), and (C) multi-host TP/PP/EP/P-D collectives. (B) is **intra-op heterogeneous execution within one model layer on one SoC**, not distributed MPI-like tensor parallelism, and it should not require RDMA, a remote server or multiple model copies.

`TopologyGraph` now includes `Node::XDNA2` with AIE kernel/instruction capability, pinned shared-LPDDR BO import path and nonuniform compute shapes, and `PowerFabricDomain` shared with RDNA GPU. `RegionPlacementPlan` assigns compute owner by *projection slice*, separate from `WeightResidencyOwner`, `KvStateOwner` and the `EngineCore` logical session owner. Per-device `CIR Timepoint` and lease generations bridge HIP/XRT/CPU/SSD and any remote queue, ensuring first-consumer joins and state-commit only after all required partials arrive.

The portable fallback is GPU-only if NPU unavailable/unsupported/unsafe. Automatic NPU enablement cannot be inferred from `device present`; it requires physical power/driver/firmware qualification and numerical validation. A new model may expose a projector that the NPU can accelerate without changes to TCP/remote collective code, while a new NPU generation adds provider capabilities and AIE program schema rather than changing model SIR. See [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

## v4 refinement — epoch fencing and limits of atomicity

The protocol guarantees **logical publication consistency**: coordinator epoch identifies one session StateRoot, ranks execute into provisional private versions and report `Prepared/Ready` only after their device/transport fences complete; coordinator issues one logical publish when required cohort is ready. Late results from old epochs are **fenced**. A rank lost between prepare/commit may require recovery/replay from a retained canonical committed root; device memory and network delivery are **not physically atomic transactions**. Exactly-once delivery would require a separate durable log/ACK/consensus protocol, outside the initial distributed execution path.

TP/PP/EP/prefill-decode split and GPU/NPU on-host operators all use the same staged regions and CIR resource/event contracts. Cross-device KDA/GDN/indexer/adapter/media state must accompany true handoff, not only attention KV; resource transfers carry explicit layout/quant/position identity. Smaller portable single-device plan remains clean with zero remote transport dependency.

## v4.1 — distributed topology includes compiler-owned fabric programs

Separate **device/host execution control** from **chip-internal or collective micro-schedules**. GPU peer ring, NVLink switch, Infinity-style all-to-all, TPU interchip collective, statically mapped wafer dataflow and remote program submission have different topology, compiler capability, cost and failure semantics. A `PartitionedProgram` can own several ranks/chips inside **one compiled region candidate**. TaskCoordinator still owns session admission, attempt epoch, common model/StateRoot schema and single logical PublishRoot after every required rank reports actual Ready/Prepared. Stale/partial remote rank must be fenced/replayed, not considered a successful user-visible commit.

A graph compiler may schedule inside a pod/fabric more efficiently than a Rust host issuing tiny separate operations. The provider publishes a `TopologyId`, `CollectiveGroup`, `InternalScheduleOwner`, artifact/quant hashes, memory/link placement and external async event. **Do not model aggregate rack memory as one dereferenceable GPU pointer.** A remote inference API with opaque state/result is **TaskService interoperability**, not equivalent to native SIR→subgraph lowering or guaranteed compatible KV+RNG snapshots.

Source and platform comparison is contained in prior-art records; normative code uses resource/fabric/collective capabilities. See [backend contract](../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

````

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/prior-art/PRIOR_ART_HISTORICAL_EXPERIMENTS_V2.md

Original UTF-8 bytes: 21749; SHA256 `2f0dae66f99af229ebc7fa256815277918ee6014035274eaea9a13a68042644e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Prior-art/historical archive — EXPERIMENT_PROGRAM.md

**Historical v2 planning snapshot, archived 2026-10-12. NON-NORMATIVE RESEARCH.** External named projects and benchmark figures are preserved as source research, while actual current design and phase gates use project-neutral contracts at `../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`, `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`. User-provided instructions and decisions remain separately in DECISION_REGISTER. Never use historical source-project APIs as the new runtime's type or module names.

---

# zerogravity — Measured-performance and feasibility experiment program

**Status:** proposed. No zerogravity experiment has been run. Every numerical result from an external project is an attributed historical source claim and must not be used as an achieved result.

## Scientific protocol — same for every candidate

**Proposition:** a modification changes a specified independent variable; define the expected mechanism, affected critical-path stage, failure hazards and stop criteria *before* implementation. Evaluate A (locked baseline) and B (candidate) using identical tensor weights/checkpoints, tokenizer/templates, numerical policy, quantization/layout, runtime versions, context, prompt and output budget, cache warmness, power/thermals and exact measurement windows. If B alters precision or sampling distribution, classify as a different quality/speed trade-off rather than a speed-only improvement.

Artifacts per run: commit SHA, dirty diff or tree hash, Rust and kernel compiler SHA, OS/kernel and GPU driver/ROCm/CUDA/Metal versions, device/frequency/power cap, RAM/UMA carveout/GTT, quant file SHA and manifest, graph plan hash, response hashes/logits, workload corpus license, context depth/new tokens, concurrency, model format, prefill/AR/spec method, acceptance distribution, cold/warm state, expert residence/hits/misses, bytes SSD and memory transferred, queue and TTFT, per-token latency histogram (median/p95/p99), memory peak and resident set, errors, repetitions and raw JSON/CSV.

**Protocol:** warmup and stabilize; run mirrored experiments in alternated ABBA/BAAB order, multiple trials, measure noise and uncertainty; report total output tokens/time plus per-sequence latencies, not a lone maximum. Prefer wall-clock end-to-end HTTP/SDK benchmark plus isolated kernel attribution. Counter instrumentation can change scheduling; use separate diagnostic and non-instrumented performance runs. If data is inconclusive, label inconclusive and do not auto-enable.

**Safety:** never intentionally overcommit RAM to trigger Linux OOM/GPU reset on a valuable machine. Simulate quotas first; physical memory tests require watchdog, OS headroom, spare swap awareness and automatic abort. Do not run competing GPU work in the middle of a performance run.

## Baseline and comparison ledger (not apples-to-apples)

Research starting points from prior source documents:
- Gufo on 128 GB Strix Halo and Qwen3.8-Flash-Next UD-Q4_K_XL: single-user AR ~1700.52 tok/s prefill and ~25.95 tok/s decode at depth 0; long 128K AR ~1478.66 pp and ~24.92 tg; repetitive MTP ~60.39 tg, mixed MTP ~31.32 tg, C8 AR ~108.68 aggregate tg. Comparators pinned llama.cpp revisions; each table is typically one warmed sample.
- A specialized engine with native quant on 128 GB Strix Halo: ~1584 tok/s prefill at 8K, ~1517 at 131K, ~37.6 serial tg (short context) and ~55–56 coding-agent speculative tg (different workloads/versions); core engine source nonpublic.
- Maya on Strix Halo GLM-5.3-Flash Maya-S 2-bit: ~210–217 tok/s prefill and ~17.8 tok/s generation with all experts effectively resident; changes to chunking/hipBLASLt made substantial prefill gains; on discrete dual AMD ~33 tok/s with MTP, not a straight device-to-device comparison.
- DwarfStar ROCm on 128 GB Strix Halo DeepSeek V4 Flash Q2: tuned ~292.86 tok/s for warm 4K prefill (from ~187.26 prior); GLM/DeepSeek/other runs differ in weight quant and implementation. DwarfStar's Qwen3.8 has Metal/CUDA but no ROCm as of inspected snapshot.
- vLLM/SGLang dense serving throughput from per-model official benchmarks only; must remeasure with reproducible same device/models. An independent Rust serving project reports model-dependent gaps, especially Gemma 4 MoE.

**Important:** Qwen3.8 Flash-Next, GLM-5.3-Flash, DeepSeek V4 Flash and Gemma 4 are different computational models; their tokens/s cannot prove which engine is faster.

## Experiment H01 — Kernel compiler feasibility on gfx1151 (ZG-P01)

Hypothesis: Rust-authored GPU kernels can use wave32/MFMA/WMMA and implement Q8_0/Q4_K/IQ2_XXS decode and grouped-MoE efficiently enough to support model-family peak paths.

Test: pinned synthetic and real weight blocks for `M=2560/4096/8192, K=640/2048/4096/8192`, 1,2,4,8,16,128,256,1024,2048 rows where meaningful; baseline legal HIP implementations (own or licensable references), Rust CubeCL and optional direct backend. Compare generated ISA, VGPR/LDS, occupancy, instructions, alignment, wave32 vs wave64 and per-kernel latency. Hold numeric oracle stable; compare ragged K and tensor tail. Report whether a given backend loses because of memory traffic, compiler codegen or launch overhead. **Go/no-go:** correct operation + plausible direct path to parity without prohibitive compilation constraints.

## H02 — Semantic IR overhead removal (ZG-P04/P05)

Hypothesis: a lowered plan has no per-operation semantic interpretation overhead versus direct handwritten dispatch. Build same tiny model on CPU and GPU via (a) semantic interpreter, (b) optimized plan, (c) direct static control. Measure host launch overhead, GPU idle gaps, plans rebuilds and warmup; inspect callable/captured kernel sequences. Fail if IR effects enforce unnecessary global synchronization or numerous host callback barriers.

## H03 — Whole-region fusion (P05/P11/P13/P14)

Candidates: RMSNorm→quant→projection, GDN recurrence→state/conv, routing→selected-ID table, HC residual→gate, sparse attention select→compute, LM head/sampling. Check fusion's register spills, occupancy, scratch, gate stochastic ordering. Use synthetic shapes and full-model end-to-end. Some fusions intentionally rejected if quality or throughput regress.

## H04 — GPU route-aware batching of MoE (P12/P16)

Hypothesis: batching rows across requests/speculative candidates by selected expert reduces re-reading weights and improves grouped MMQ/GEMM without worse P95. Build deterministic workloads with uniformly random vs skewed route overlap: mean distinct experts per N assignments; batch rows 1/2/4/8/16/32; model-specific expert matrices. Compare ordered per-token, grouped-by-expert and scheduler-guided with fixed and pressure-limited cache. Measure router output transfer (must stay GPU-side), regroup/scatter cost, bandwidth and fairness. Abort experimental policy on meaningful worst-case latency or memory regressions.

## H05 — Residency × scheduling co-optimization (P10/P12/P16)

Compare fixed LFU/LRU/layer-partitioned cache, preloaded popularity, dynamic cache, GPU hit/no-miss and OS page-cache subsets; then jointly vary batch width and resident expert slots. Benchmark hit% is inadequate: measure miss latency critical path, transfer bytes, resident slab fragmentation and queue stalls. Test 128 GB `gfx1151` full-resident vs 92–96 GB with SSD fallback, separately from PCIe discrete. Disable CPU expert lane on UMA by default until contention/benefit proven; test PCIe variants with CPU lane separately.

## H06 — Disk-only n-gram/Engram (P03/P10/P13/P17)

Hypothesis: read-only row fetch from a huge lookup table can use a much smaller bounded RAM cache than mapping entire table. Test random/scattered/sequential/skewed row accesses, correct BF16 reconstruction, alignment, repeated IDs, out-of-order reads, corrupt/truncated source, cold and warm NVMe, OS page cache; compare `pread`, `io_uring`, suitable direct IO; quantify SSD throughput/latency and impact on main GPU inference. Never infer 95+ GiB disk table means 95+ GiB must be GPU resident.

## H07 — Hybrid prefix state sharing (P10/P11)

Hypothesis: shared immutable prefix includes recurrent GDN/KDA, indexer and token-history state, reducing prefill without expensive copies. Workload: 40K shared system/repo prefix with 8 diverging agents, branching in middle and tool-result rewinds. Measure full cache reconstruction vs immutable prefix COW vs eager copies vs deferred state materialization. Verify every state buffer, logits and RNG against fresh evaluation. Test eviction, file restoration, cancellation and mixed image/text layouts.

## H08 — Adaptive speculation economics (P15/P16)

Measure no speculation, MTP 1/2/3 depths, trained DSpark when supported, independent smaller drafter, prompt lookup and n-gram. Workloads: repetitive code, agent tool results, unpredictable prose, math/reasoning, nonzero temperature, long context and mixed concurrent sessions. Controller chooses by *net committed tokens per total draft + verification + rollback + idle cost*, not just raw acceptance. Compare switching and hysteresis; low-acceptance bypass vs always-on; correctness against exact target sampling. Publish acceptance distribution, rejection position, ratio of useful target forwards and quality differences.

## H09 — Prefill versus decode resource competition (P09/P16)

Workload matrix: C=1,2,4,8,16,32; requests arriving under Poisson/bursty schedules; 256, 2K, 8K, 32K prompts and 128/512 outputs. Compare pure FIFO, token-budget chunking, prefix-aware, mixed prefill/decode with 128/256/512/1024 quantum and bounded latency priority. Observe TTFT P95, inter-token P95, starvation, total tokens/s, GPU occupancy and expert residency. Tune for separate interactive vs throughput objectives.

## H10 — Compressed/sparse attention context depth (P11/P13/P14/P17)

Independent operator oracles and full models: dense GQA, sliding, QSA, KDA/MLA+DSA, CSA/HCA, compress boundary. Depths 0/4K/8K/16K/32K/64K/128K/256K and > where physically supported. Test masked index rows, selection ties, FP32 reductions and cache quantization. Compare GPU selectors vs CPU fallbacks; quality fixed-token logit/KL and retrieval. Discard alleged win if it relies on changing number of selected keys, precision or attention semantics without labeled policy.

## H11 — Unified-memory safety and throughput (P07/P10/P13/P14)

Profile Linux Strix Halo UMA carveout, HIP-reported allocatable VRAM/GTT, `MemAvailable`, `CmaFree`, physical working set, pinned stage limits, kernel/firmware and IOMMU settings. Capacity targets 64/92/96/128 GB as available; vary concurrent requests and prefill scratch. Require admission refusal before danger, stable cleanup, no system stalls, performance with/without redundant pinned RAM tier. Do not prescribe changing kernel boot parameters as a universal solution.

## H12 — Mixed quantization and model quality (P03/P12/P13/P14)

For same reference FP8/BF16 model, compare Q2/IQ2 and Q4/K plus Q8 dense/mHC choices, unchanged lossless layout repacking vs deliberate requantization. Quality gates include logit RMS/max error, KL divergence, next-token agreement, calibration case official continuations, coding/tool/long retrieval. Separate speed due to better layout, cache footprint and **changed numerical precision**. Refuse default quality upgrade/downgrade without recorded trade-off.

## H13 — Distributed topology and transport (P19)

Measurements: local 2/4/8 GPUs, heterogeneous PCIe, two 128 GB hosts with RDMA/TCP where available. Compare layer pipeline (prefill), tensor parallel and expert parallel (decode), and batching under constrained links. Quantify P2P/RDMA activation traffic, comm stalls, memory capacity, backpressure and workload scaling. Inject partial writes, timeout and wrong model/checkpoint handshake to test fail-closed. No unnecessary remote access/security downgrade.

## H14 — Long-lived agent workload (P09/P10/P15/P17)

Replay realistic coding agent conversations: shared tools/system instructions, full repo context, repeated tool results, JSON, reasoning boundaries, stream interruptions, multi-turn reuse and image attachment. Measure total task wall time and quality, not just kernel t/s. Identify repeated-prefix savings, hit rate, checkpoint storage and branch costs. Compare API semantics against native tokenizer/template references.

## Bench artifacts organization (future)

```
bench/
  corpora/         # hashes, origin, license (no secret transcripts)
  configs/         # exact comparator commands
  kernels/         # microbench kernel shapes
  models/          # same GGUF/safetensors checkpoints + hashes
  scenarios/       # arrival/concurrency/cache profiles
  results/YYYY-MM-DD/<machine>/<engine>/<model>/
     run.json  metrics.csv  logits.sha256  stdout.log  stderr.log
  reports/         # methodology, confidence, wins/losses
```

Scripts must never publish model weights or private prompts by default. End-to-end metrics originate from complete responses with EOS/stop semantics described, not a misleading concatenation of isolated calls. Any measurement with failed outputs, OOM, swap thrash or uncontrolled concurrent GPU activity is invalidated.

## Qualification decisions

- `PASS`: correctness gates pass; repeatable speed gain above noise on matched workload; safe memory; regression set green.
- `NEUTRAL`: correctness pass but benefit within noise or on only microbench; keep simplest existing path, or guarded opt-in.
- `FAIL`: numerical/state error, unsafe memory, latency regression beyond policy, quality decline or unsupported behavior; revert candidate.
- `INCONCLUSIVE`: insufficient trials, thermals/driver differences, incompatible quant/model, uncontrolled hardware; no performance claim.

Every optimization report links the source commit, test corpus, full benchmarks, reasons for retaining/rejecting and next investigation. That is part of zerogravity's intended transparent engineering discipline.

## 2026-10-10 integrated v2 research expansion (new experiments H15–H25)

**H15 — Rust compiler provider comparison:** CubeCL, direct target Rust HIP/AMDGPU option on gfx1151; CubeCL vs a Rust tile compiler and experimental a Rust SIMT/PTX compiler SIMT on suitable NVIDIA hardware. Compare generated ISA/LDS/register/tiles, Q8/Q4/IQ2 GEMV, grouped GEMM, attention and compile stability under exact same inputs/quality. Driver/FFI policy remains open; never pass external C++ engine as Rust-owned kernel result.

**H16 — Compiler effect legality:** synthetic full KV + GDN + sparse indexer + sampler RNG+grammar, randomly permute legal pure ops and deliberately illegal state reorders; SIR verifier and RIR pass reject illegal, reference state/logits remain stable. Compare full-model after chunk/snapshot/fork/rejected draft.

**H17 — CIR scheduling and resource reuse:** async simulated GPU/SSD/CPU timepoints, leases and alias check, forced stale slot generation, parallel copies and IO, fault injection; physical measure whether next-layer expert prefetch overlaps useful computation and whether scratch borrowing reduces resident pressure without p95 regression.

**H18 — Execution mode tactic selection:** same kernel algorithms under eager vs device graph capture (a 2026 kernel-dispatch lesson), measure full model and serving TTFT/P95, not just GPU event; persist mode-keyed candidate only after numerical and quality proof.

**H19 — Bounded equality search:** choose one pure norm/quant/projection or attention prelude and compare no rewrite vs heuristic vs bounded e-graph search (bounded graph-search lesson), report compile seconds, code size, memory, generated plan and actual end-to-end t/s. Kill if compile-time explosion or no meaningful gain.

**H20 — Three-plus domain prefix snapshots:** full/sliding KV + GDN/KDA+compressed indexer+token ngram/adapter/multimodal; compute exact common valid resume frontier under different alignments and storage tiers; compare replay vs state restoration, shared agent TTFT, byte identity in same fixed plan and full logits. Guard tenant salt and model/quant identity.

**H21 — Extensibility drill:** add a novel stateful attention op to SIR/reference, implement generic kernel and one target-tuned kernel, add a new device compiler provider. Count unrelated crates/files modified, run API/session unchanged; architecture fails if model-specific HTTP/allocator branching required.

**H22 — Full product correctness:** content-addressed model store/manifest/atomic partial download, tokenizer pre-type exact vectors, grammar mask/rollback, LoRA cache generation and embedding/rerank normalization, image encoder spans; fuzz inputs and staged release matrix, no unknown-op execution.

**H23 — Prevent misleading claimed improvements:** maintain separate tables of same-model same-quant benchmarks and unmatched source project literature. DwarfStar 8×L40S DeepSeek TP cannot stand in GLM matching cell; fix historical dataset attribution. Report uncertainties/incomparable rows as such.

**H24 — Unified-memory machine safety:** probe actual available devices instead of old 3060/Cezanne/27GB research assumption, validate physical RAM/GTT/cgroup/IO under 92/96/128GB (when present), no forced host OOM and full resident/streamed model fit. All RAM-heavy trials guarded; no blind firmware/kernel changes.

**H25 — Compile/hardware hot-path cost:** bench one real model as (a) CPU reference, (b) SIR interpreter for debugging, (c) RIR/CIR compiled bundle direct launches and (d) hand-coded direct dispatch baseline; demonstrate no meaningful per-token IR interpretation overhead. Include graph capture and worker diff overhead.

**Evidence discipline:** S0-00…S0-14 in integrated plan define Stage0 mandatory status. Every H15–H25 is a falsifiable proposal, not proof of performance. New benchmark cells include `semantic_schema, state_layout, cir_event_version, kernel_provider, execution_mode, model/quant/checkpoint/hash, context, new prefill tokens, power/driver, p95 and quality`. Do not automatically enable unbenchmarked heuristic or compile search in serving.


## 2026-10-12 research expansion: Gufo upstream and XDNA2 cooperative prefill (H26–H31)

**H26 — Same-model independent output-row projection partition**: synthetic CPU math for GEMM row split, GDN z, QSA query/gate, shared expert gate/up, odd widths, quant-block/bias/scales and downstream full state. RIR pass and CIR fake device first-consumer barrier. Proof: same semantics with GPU-only and mock 2-device variants, missing NPU fallback works. **Kill** if joint row map/state cannot be verified or compiler requires Qwen model-specific hardware code.

**H27 — XDNA2 program/toolchain and Rust source feasibility:** pinned fork's C++/Python AIE `v8bfp16ebs8` 4096×2560×1024 `.xclbin` is a research baseline **not a Rust-written kernel**. Evaluate a Rust-authored AIE/MLIR source generation and XRT driver FFI; document ABI, compile path, licenses, shape support, build reproducibility, whether compiled bitstream identity validated. **Blocked** if strict Rust source cannot be met; do not import source as a hidden wrapper.

**H28 — Cross-API memory/DMA safety and concurrency:** fake HIP mapped host BO/XRT import, generation-tagged buffers, NPU queue depth four, GPU stream host submit/wait callbacks (as source reference), first-use event join, short read/timeout/lost device/cancel and quarantine. Physical concurrent NPU/GPU firmware/AGESA/FCLK/XRT/ROCm soak on isolated device only. **No automatic privileged fabric/power changes or system hang attempts**; one crash invalidates auto-enable qualification.

**H29 — Source benchmark independent replication:** Qwen3.8-Flash-Next **UD-IQ4_XS** 16K 1698 GPU-only vs 1821 GPU+NPU prefill author claim, 32K 1783 NPU, 40K cold served 1637 NPU, 8192 chunk source conditions. ABBA same held fabric clock, XRT performance/NPU wake state and power/thermals in *both* arms, separate GPU-only default-clock comparison; measure TTFT/P95, actual overlap/stall, pinned memory and idle energy. Measure NPU projection subsets separately and zero-benefit modes, not just one headline. **No unqualified peer-engine ranking**.

**H30 — Mixed-compute numerical and quality:** v8bfp16ebs8 block-float nine-byte/eight-value encode/decode as separate ComputeEncoding, Q8_0 checkpoint stored quant unchanged. Compare source CPU/GPU outputs, per-layer full logits, GDN state/attention selected IDs, 30 greedy smoke + hundreds of held-out, exact sampler/MTP parity, longest context. Reject auto-enabled offload if quality changes outside approved validated numerical policy.

**H31 — General TaskProgram extension:** upstream Gufo Qwen3-ASR, Qwen3-TTS, Qwen-Image-2.1, MiniMax H3 research; model-independent task stage loop (ASR encoder, TTS vocoder, image denoise, video frame) compiles to SIR/KIR/CIR and ResourceArbiter, shares DeviceWorkers but has unique TaskProgram state/output. Test one fake nontext task plus one real accepted model reference when available; no claim all media models are supported before independent quality/latency tests.

**Evidence keys:** exact fork commit `8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`, Gufo upstream head `f37fb012281bbd58054b1082933153bbec303106`, source model/quant/compiled NPU program hashes, AIE toolchain, XRT/ROCm driver, BIOS/AGESA and held/default fabric clocks. Distinguish discrete GPU+iGPU expert ownership (a discrete-GPU+iGPU expert-ownership approach) from same-SoC NPU+iGPU projection slicing (Gufo fork). Sources in the two new `.agents/docs` files. Research and plan only, no H26–H31 runs completed.

````

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/prior-art/PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md

Original UTF-8 bytes: 51878; SHA256 `67bde37af2af6b14798e801276c048c73edb30dc0e6c0a7185450d0fcacdde8a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Prior-art archive — historical complete v2/v2.1 takeover handoff

**NON-NORMATIVE historic source and architecture provenance, archived 2026-10-12.** The prior original handoff names research projects, codepaths and source benchmarks; all are retained for traceability, but the current project-neutral architecture, live roadmap and next-agent prompt control design and implementation. Do not adopt its earlier source-project-labeled interfaces or benchmark parameters as user choices. See `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md` and `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` for the operative design and safe source usage rules.

---

# ZEROGRAVITY — LATEST TAKEOVER UPDATE (v2.1 · 2026-10-12)

**Canonical project root:** `/home/pvl/spaces/abird/src/zerogravity/`. **Scope of this update:** user explicitly requested architectural and execution-pipeline inclusion of [mikealanni/gufo](https://github.com/mikealanni/gufo) and the [Oct 11 Strix Halo NPU prefill report](https://www.reddit.com/r/StrixHalo/comments/1x3gtdh/qwen38_flashnext_prefill_past_1800_toks_on_strix/). Existing **upstream `gufo-org/gufo`** is a different source reference. This section supplements and takes precedence over the v2 historical takeover below if details conflict. No source from prior-art projects was copied unless directly vendored (for example, CubeCL (vendored)), which retains its upstream license; the Zerogravity engine is a from-scratch rebuild informed by prior-art concepts, learnings and research.

**Status:** all implementation phases P00–P23 are STILL **NOT_STARTED**; no functioning Rust engine, AIE program, XRT/NPU backend, model runs, performance or hardware safety validation exists. No Git repository was initialized. Nothing in these files means GPU+NPU is automatically enabled. The user's separate `research/inference` corpus is preserved read-only; modifications restricted to current `.agents/docs` and `.agents/plans`.

## Essential reading additions (must read before NPU/TaskProgram implementation)

- `GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md` — inspect exact fork and upstream; primary source at fork `fork-enhancements`, commit **`8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`** (GitHub app); Reddit author report date **2026-10-11**. Paths include NPU GEMM source, XRT BO bridge, HIP staging/first-use join, AIE source and compiled program.
- `../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md` — **authoritative integration design** for projection-row compute ownership across RDNA GPU+XDNA2 NPU, CIR events, resource manager, Rust source policy, safety/fallback and TaskProgram media stage variants.
- `DECISION_REGISTER.md`: user decision **U-016**, working choices **P-026..030**, open **O-023..026**. Older user decisions U-001..U-015 preserved; no unapproved source language, license or device policy overridden.
- `ZEROGRAVITY_MASTER_PLAN_2026-10-10.md` **§17**: latest v2.1 architecture and all domains.
- `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`: new **S0-NPU-A…F** optional track; doesn't block CPU/GPU vertical slice; `IMPLEMENTATION_ROADMAP.md`: phase mapping P01/P04/P05/P07/P10/P11/P13/P16/P17/P18/P20/P21/P22/P23.
- `EXPERIMENT_PROGRAM.md`: H26–H31 tests (source precision, Rust-owned XDNA2 codegen, cross-API event/BO safety, same-clock ABBA, whole-model quality, TaskProgram extension). `RELEASE_ACCEPTANCE_GATES.md` makes hardware safety and numerical qualification mandatory.

## Critical user-requested architecture — WHAT IS DIFFERENT

The specific `mikealanni/gufo` fork is **not** another ordinary Gufo benchmark nor a discrete-GPU+iGPU expert-ownership approach. It offloads pure projection **output row subsets** from `gfx1151` GPU to the **on-SoC XDNA2 NPU** while GPU computes independent rows/ops in parallel. Source-demonstrated cases:
1. Gated DeltaNet SSM input **z** rows;
2. Qwen Sparse Attention query/gate rows for selected heads;
3. shared expert gate/up GEMM (NOT routed experts themselves).

One fixed NPU GEMM candidate is **M=4096 prompt tokens, K=2560 input, N=1024 output** with **v8bfp16ebs8** (one exponent plus eight 8-bit mantissas per nine-byte group), AIE 32-core program and XRT long-lived hardware context. GPU encodes inputs, NPU consumes HIP-mapped pinned/XRT-imported BOs, C output joins at its *first actual consumer* through callback/event ordering. Its input is weight/activation slices of the *same model forward pass*, not a second NPU drafter or independent model.

Author reports on **128GB Strix Halo GMKtec EVO-X2, UD-IQ4_XS** at 16K prefill: **1698 GPU-only → 1821 GPU+NPU tok/s, +7.2%**, 32K NPU 1783, 40K cold served 1637; 30 greedy prompts identical is a limited smoke test, not full-logit parity or independent benchmark. Do not confuse with upstream Gufo UD-Q4_K_XL GPU-only source measurements, Qwen 27B DFlash2, or its DeepSeek/R9700 results.

**Main design:** no new Gufo model runtime. ModelRecipe/SIR stays exactly mathematical, RIR optionally creates `ProjectionRowSlice` with disjoint row maps and selected compute owners, KIR emits capable NPU/GPU provider kernels/encodings, CIR plans mapped BO leases, external XRT enqueue/wait, concurrent GPU progress and first-use joins, while **one EngineCore** commits all recurrent/attention/sampler state. `ResourceArbiter` accounts every pinned weight/input/output and clock/power domain capacity. Missing NPU, unsupported shapes, incompatible precision or unqualified hardware => **GPU-only fallback**. XRT midflight failure aborts provisional transaction and quarantines DMA buffers.

**Safety is a hard veto for automatic NPU activation:** the Reddit poster/fork author reports **hard system hangs/machine checks** when GPU fabric frequency changes while NPU is working. Holding fabric clock at max, NPU power mode and suspend/keepalive changes were the author's environment-specific workaround; 12 clean runs do NOT establish general reliability and pinned clocks can raise idle power. **Never automatically run sudo/sysfs/fabric-clock/systemd/IOMMU/firmware tuning.** An independent GPU-only path is mandatory; NPU stays OFF unless firmware/AGESA/ROCm/XRT/power/driver/compiled program is controlled, source-policy compatible and physically qualified on isolated hardware. A software watchdog cannot recover a frozen SoC.

**Rust-only caveat:** fork's AIE kernel and compiler generator are **C++ plus Python MLIR-AIE**, hosted by C++ XRT, with compiled `.xclbin` and instruction binaries. A Rust FFI wrapper does not satisfy the stated Rust-authored custom kernel requirement. S0-NPU-B must prove genuine Rust-owned AIE/MLIR kernel codegen or mark feature blocked without taking over Gufo source. Driver/runtime FFI remains an explicit user policy question, and NPU build is optional.

## Other upstream Gufo pipelines (future model/task categories)

The **upstream gufo-org/gufo** repository now includes model guides and runnable work for Qwen3.8-27B with DFlash2 and vision, Qwen3.8 Flash-Next MTP, DeepSeek V4 DSpark, Qwen3-ASR, Qwen3-TTS; Qwen-Image-2.1 and MiniMax H3 text/video/audio pipelines are in progress. These **do not mean zerogravity supports any one of them**. Implement task-specific stage/loop contracts:
- LM decode / draft verifier: causal token frontier;
- ASR: audio input encoder/chunks and transcript;
- TTS: acoustic generation + vocoder/media stream;
- image edit/generate: conditioning + repeated denoise/flow steps;
- video/audio: frame/temporal-stage and media output contract;
- embeddings/rerank: encoder+pool/score without LM KV.

All share ModelStore, exact SIR operator reference, RIR/KIR candidate backend, CIR state/async resources and EngineCore/DeviceWorkers, but **TaskProgram** owns stage control, iteration frontiers and output type so no media pipeline is shoehorned into next-token sampling. A new NPU encoder offload is *unproven* until actual kernel/quality/stability experiments.

## Handoff/final safety instruction

For **design-only** follow-up, edit docs/decisions/manifest only. For **authorized implementation**, start CPU reference+generic GPU and small RIR/CIR fake device slice test; NPU fork offload remains a separate experimental S0-NPU-A…F and can be blocked without blocking portable inference. Never announce "seamless NPU enabled" before real Rust AIE codegen, cross-XRT/HIP DMA correctness, matched full logits, controlled same-clock/energy A/B and safe repeated physical hardware qualification. Last accepted model semantics, numerical precision, sampling distribution and durable state always outrank peak t/s.

---

## PREVIOUS UNIFIED v2 TAKEOVER (preserved, subordinate to the latest update)

### Previous title: ZEROGRAVITY — CURRENT COMPREHENSIVE MASTER TAKEOVER (Integrated v2 · 2026-10-10)

**Authoritative root**: `/home/pvl/spaces/abird/src/zerogravity/`. Evidence and pinned source analysis: `/home/pvl/spaces/abird/research/inference/` (**read-only during this merger**). Tool: dotlink l5. **Status:** complete v2 architecture/research **documentation only**; no Rust engine/source implementation, Cargo workspace, Git repo, GPU benchmark, model downloads or release was created. Every ZG-P00…P23 implementation phase is NOT_STARTED.

**This section supersedes earlier v1 handoff text wherever inconsistent.** The earlier handoff remains below under a historical heading to preserve the original user decisions and research discussion. Latest user decisions U-014/U-015 explicitly request merging the best of both plans and expanding research for the strongest possible future-proof inference design, while preserving U-001…U-013 (project Rust/name/models/Strix/canonical path).

## Read-first authority

1. the agent playbook and `DECISION_REGISTER.md`: confirmed user decisions versus assistant/research working choices and open questions; **research council "RESOLVED" values are not automatically approved by user**.
2. `ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`: **rewritten v2** complete architecture.
3. `ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md`: explicit D-01…D-20 ruling and research defects.
4. `COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md`: source-backed comprehensive new research across compiler/serving/IO/kernel/model evolution.
5. `../docs/SEMANTIC_IR_AND_COMPILER.md`, `../docs/ASYNC_RESOURCE_COMMAND_IR.md`, `../docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`: runtime/IR foundations.
6. `../docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`: GGUF/safetensors/quant/tokenizer/grammar/model store/LoRA/embedding/multimodal.
7. `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`, `IMPLEMENTATION_ROADMAP.md`, `EXPERIMENT_PROGRAM.md`, `RELEASE_ACCEPTANCE_GATES.md`: real dependency, 24 phases, experimental/quality gates.
8. All other subsystem docs including state/KV/cache/MoE/scheduling/kernel/Strix/distributed/numerics/security/Rust engineering, status and refreshed SHA-256 manifest.
9. Read the relevant pinned-revision manifest, synthesis/innovation research and platform/machine docs; verify latest upstream before coding. **Do not modify research corpus in this task.**

## Current architecture (mandatory conceptual distinction)

```text
ModelStore / GGUF/safetensors / tokenizers + template + mm / adapters
    -> verified ArchDef/ModelRecipe -> Semantic Model IR (SIR)
       mathematical model operator graph + explicit SSA state effects
    -> legal optimized Region IR (RIR): fusion/layout/precision/shape
                          \
                           Kernel IR/compiler providers (KIR):
                           Rust CPU + CubeCL + target Rust HIP +
                           NVIDIA Rust tile / SIMT-PTX compiler candidates
                          /
    -> Async Command/Resource IR (CIR):
       Buffers+leases+timepoints+GPU streams+SSD+peer+commit edges
    -> immutable ExecutableBundle: direct kernels/captured regions
                      |
HardwareProbe -> pure HardwarePolicy + ResourceArbiter (physical budget)
                      |
Edge/API -> EngineCore(single logical session owner) -> StepPlan
           -> WorkerBackend GPU/CPU direct launch -> WorkerDelta
           -> target verification + exact all-state frontier commit
           -> streaming committed output / ModeReceipt / metrics
```

**A kernel plan is not a semantic model**: `KernelPlan` stores tested candidate implementations and provenance; `HardwarePolicy` sets safe resident/SSD/KV placement; `ExecutableBundle` holds compiled code+resource graph; `StepPlan` assigns real request rows to handles. Do not create a giant object called ExecutionPlan that conflates these. Model-specific `ModelSpec::kernel_plan(hw)` from historical research is superseded by model recipe with exact math plus compiler `KernelProvider` target selection.

**RIR/KIR/CIR build-time only**, no per-token IR graph interpreter. A specialized whole-layer/whole-model Rust GPU kernel may replace any semantically closed region if quality, state-effect, device capacity/alias and resource contracts validate. If a fast candidate unavailable, fall back to generic/ref region at a *compatible state/layout boundary*, or replay from canonical prefix if conversion required. Unknown new model op still requires a reference implementation; file format alone doesn't grant arbitrary model coverage.

## Best of the research preserved

- **vLLM/SGLang serving**: token-budget scheduling, paged KV, chained prefix hash, optional SGLang radix index over same allocation pool, async prefill/decode/verify, latency priority and per-user p95.
- **Detailed native engine**: single owner Rust `EngineCore`, bounded `EngineMsg/EngineEvent` and worker `StepPlan/WorkerDelta`, persistent `InputBatch` deltas, no global per-token lock; logical owner and device workers distinct.
- **KV choice by workload**: paged generic and contiguous extreme reserve-all, both under one physical resource arbiter; v1 design can choose only proven subset. Generalized full hybrid state-group prefix coordinator works across 3+ types, not only ordinary KV.
- **Maya/Strata/DwarfStar giant MoE**: SSD/RAM/UMA expert tiers, fixed safe slots, router device pointer table, asynchronous bounded miss request, pinned staging, hotlists/recency/LFU, CPU lane calibrated; n-gram/Engram huge table reads are separate tiny row-lookup workload.
- **Gufo performance path**: model-specific gfx1151 Qwen GDN/QSA/MTP and grouped low-bit MoE; GLM KDA/DSA/MLA/mHC and per-layer expert caches; attention selection and draft rollback must stay device native where feasible.
- **DwarfStar frontier mechanics**: `CommitPlan{FullDirect,Prefix(n),Replay(n)}`, frozen boundary snapshots, trained DSpark/MTP verification; wrapped inside full SIR state transaction (KV/GDN/KDA/indexer/token history/grammar/RNG/modality).
- **llama.cpp product**: GGUF parser, CPU SIMD, quant/dialect, tokenizer pretype, Jinja model prompt, grammar (a grammar library/GBNF), tool parser and easy single-binary CLI.
- **Ollama model store+calibration**: content-addressed SHA256 blobs, manifest list/GC/sidecars, offline download, optional acceleration pack and imatrix/quality overlays; adapters LoRA, embedding/rerank PoolingSpec and multimodal separate encoder scheduler.
- **2026 compiler innovation**: an ahead-of-time ML compiler's Flow/Stream/HAL explicit async resource, a tensor compiler's Relax/TIR/MetaSchedule, MLIR side effects, bounded graph-search, 2026 kernel-dispatch eager/graph aware tuning; NVIDIA Rust tile/SIMT-PTX compilers and CubeCL alternatives. These are research evidence, not bundled dependencies.
- **Quality and operations**: independent official/CPU numeric oracle, ABBA where exact applicable, target distribution correctness, bounded memory, source pinned experiments, compiler+model critical path profiling, source licenses/security and phase gates.

## Original programs' faults explicitly corrected

1. Old research "any model any runtime runs" is overbroad: **only supported model semantics and backend/resources** may run, otherwise typed refusal.
2. Disk-KV research calls header 48 bytes while specified fields total **54**; five-character magic erroneously `u16`. **New versioned binary schema required**, no original copy.
3. DwarfStar 8×L40S DeepSeek multi-session benchmark appears in a GLM table: not a matched GLM performance baseline.
4. Old research 6GB RTX 3060/Cezanne 27GB machine and purchase of 128GB Strix are historical assumptions: probe current reachable devices; user hardware info cannot be inferred reliably from old research.
5. `cubecl=0.11`, exact Rust MSRV, `zg`, dual MIT/Apache-2.0 and C++/HIP vendor escape are research council preferences; **remain open** until measured or user approved.
6. Research "two concurrent requests = native batch" and "frontier rewind O(1)" need implementation caveats; device grouped rows and physical state rollback measured.
7. Prior project 3-layer SIR/EIR/DPI omitted explicit async resource/command optimizer, and product lacked grammar/ModelStore/LoRA/embedding/rerank detail.
8. 93 engineer-days research Phase0 cannot honestly be equated to 4–6 weeks for two engineers without major scope cuts or added staffing. Stage0 now splits hard gate vs optional.
9. Historical source repositories/performance claims are dated and from authors, not zerogravity benchmarks. No source-license-free copying of specialized engine code.

## New Stage 0 and phases

**Mandatory earliest evidence** from `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`:
- S0-00 reproducible lean Rust workspace/source pins/decision log.
- S0-01 true gfx1151 Rust kernel ISA/quality and performance feasibility; do not infer all GPU support from CubeCL README.
- S0-03 GGUF/tokenizer + CPU quant/attention math oracle, first real dense model; optional CUDA Rust tile compiler / Rust SIMT-PTX compiler as second target.
- S0-04 minimal SIR explicit KV+GDN effects (full indexer soon).
- S0-05 RIR legal fusion + KIR kernel candidate and missing-specialization safe fallback.
- S0-06 CIR lease/timepoint/IO/error simulation, no alias/early release.
- S0-07 EngineCore/StepPlan/WorkerDelta and paged KV/frontier simulator.
- S0-08 speculative all-state/RNG rollback and reference quality.
- **S0-12** complete artifact → SIR→RIR→KIR/CIR→CPU & GPU direct executable→tokens through same EngineCore, with fallback disabled kernel tested.

Other experiments (grammar/embedding/LoRA/MTP/various GPUs and long-term async cache innovation) are independent tracks. **No need to implement a 25-crate scaffold or general search optimizer to complete the vertical slice**. All implementation phases ZG-P00…ZG-P23 remain NOT_STARTED now; this document update does not count as actual implementation.

Release goals G1 portable runner, G2 serving, G3 extreme Qwen/GLM/DeepSeek, G4 breadth/parallel all need real source-matched quality/latency/performance evidence. Huge model fit requires OS headroom and resource capacity; do not trust ROCm GTT as free RAM.

## Existing user decisions and open policies

Consult decision IDs **U-001…U-015**. User approved consolidation and comprehensive research, **not** specific libraries, license, vendor binding, benchmark percentages, release backend sequence or buying hardware. The reconciliation proposals are P-018…P-025. Open O-001…O-022 include FFI and non-Rust kernel source, compiler selection, product executable/license, v1 model/quant and OS matrix, disk snapshot schema, user-visible CLI and external plugin trust.

## Extension/maintainability acceptance

For every new model semantic op: official source/checkpoint geometry and tokenizer, Rust CPU independent oracle, typed operator/effect/snapshot/replay, generic provider, optional tuned KIR, full logits/quality at long context and forced fallback. Adding op shouldn't change HTTP/ModelStore/EngineCore except genuinely novel capability. New GPU should add DeviceCapabilities + KernelProvider with independent benchmarks, not alter every model recipe. New I/O tier extends CIR resources and ResourceArbiter, not model math.

**Release claim rule:** only matched model/checkpoint/quant/context/sampler/concurrency/device/driver/power, TTFT/p95 and benchmark repeats can establish superiority. Negative results and unsupported cells must be published. Every compiled optimization remains optional and can be disabled when worse.

## Next-agent operating instructions

1. Use the current live canonical root; dotlink l5 launch root `/home/pvl/spaces/abird`. The research corpus is read-only reference.
2. Read master+decision+reconciliation+research+all pertinent subsystem docs; verify actual files and Git if initialized (was absent at consolidation).
3. Obey **current user's requested scope**. For design request edit docs only. For implementation take one small Stage0/phase subset, don't decide unapproved policy choices silently.
4. Inspect exact source pins/hardware and independent official model reference. Preserve SIR effects and physical resource/event safety.
5. Test and record source SHA, model/quant/hardware, kernel generated ISA, numerical parity, prefix/spec rollback, quality, measured p95 and full-model throughput; mark incomplete GPU/model tests SKIPPED/BLOCKED honestly.
6. Update roadmap/decision/status/handoff/manifest on completion. Never claim faster than any engine without a controlled comparison.
7. No closed-source engine copying, no implicit target precision change, no arbitrary HF remote code execution, no unsafe memory overcommit.

**Current completion:** comprehensive research/plan merger; no engine code/release. The following historical snapshot is retained as provenance and is subordinate to this current v2 section.

---

## HISTORICAL INITIAL HANDOFF SNAPSHOT (v1, preserved for context)

### Prior title: ZEROGRAVITY — FULL MASTER HANDOFF AND TAKEOVER AUTHORITY (2026-10-10)

> **Purpose:** external LLMs and engineering agents must be able to take over zerogravity from disk alone, without the originating ChatGPT conversation. This is a **design/research handoff**, not a code completion report. Read the master plan and subsystem specs; this document makes the objectives, user decisions, research and next-action boundaries self-contained.

Project: **zerogravity**  
Creation date: **Saturday, 2026-10-10 (Asia/Singapore)**  
Working directory observed through **dotlink l5**: `/home/pvl/spaces/abird`  
Project directory (user-corrected canonical location): `/home/pvl/spaces/abird/src/zerogravity`  
Documentation: `zerogravity/.agents/docs`, `zerogravity/.agents/plans`.  
**Implementation state at handoff:** design/archive only. No Cargo workspace, Rust engine, GPU kernels, benchmarks, downloaded model weights, CI, Git repository, or releases were created in this request. Never mistake external source-project performance measurements for zerogravity results.

---

## 1. What the user explicitly wants

The user wants to build one entirely Rust-authored inference engine called **zerogravity** that runs:
- common models (Llama, Gemma 4, Qwen 3.5+ dense/MoE/hybrid and future compatible architecture families);
- exceptionally large modern models, especially **Qwen3.8-Flash-Next (~125B language backbone, auxiliary n-gram lookup/MTP)** and **GLM-5.3-Flash (~321B total)**;
- common CPU/GPU hardware as a normal portable engine, but **aggressively specializes for eligible hardware**, especially **AMD Ryzen AI Max+ 395 / Radeon 8060S / Strix Halo `gfx1151`**, to compete with highly tuned engines;
- high server concurrency/throughput like vLLM/SGLang; model support and usability like llama.cpp; model/hardware specialization like Gufo; expert memory tiering like Maya/Strata/DwarfStar;
- seamless correctness-preserving **fallback to a general model runtime** when a specialized kernel/whole region is absent, rather than refusing all other models or duplicating runtimes.

The user specifically requested a **Semantic IR-based architecture**, supported the idea of pushing it beyond existing inference runtimes when possible, and demanded that the docs preserve every design decision/unknown for external agents. They asked for research, then a refined design, then a comprehensive persistent documentation/handoff archive through **dotlink l5**. They have **not** asked in this turn for Rust implementation, repo initialization or model benchmarking.

User decisions are verbatim-intent summarized with IDs **U-001..U-012** in `DECISION_REGISTER.md`. **Do not upgrade any assistant proposal into an explicit user choice.**

### Key user questions that shaped the project

1. Research vLLM, SGLang, llama.cpp, specialized gfx1151 engines, two Strix Halo Reddit experiments (Maya GLM and Qwen Flash-Next) and additional optimized engines; first goal research only.
2. Include `https://github.com/antirez/ds4` (DwarfStar) in architectural research.
3. Combine the best concepts to build a **refined architecture reaching each model's peak possible performance**, potentially exceeding specialized projects, while **retaining a general runtime fallback**.
4. Ask whether Semantic IR gives an ability to outclass existing runtimes; answer was **credible architectural opportunity, not speed guarantee**. The key is mathematical/state definitions compiled to hardware-specialized direct GPU plans, not runtime IR interpretation.
5. Name project **zerogravity** and write exhaustive authoritative plans/docs/handoff at the initially requested relative `./zerogravity/.agents/{docs,plans}` via `dotlink l5`.
6. **Later explicit user correction (U-013):** move the complete project to `/home/pvl/spaces/abird/src/zerogravity/`, update documentation paths and revalidate. This corrects placement only, preserving all technical decisions.

This chronology matters: no claim that source research proved performance, and no implementation phase has been approved/started by this request.

---

## 2. Authority and reading order

1. `../plans/AGENT_IMPLEMENTATION_PLAYBOOK.md` — navigation/agent contract.
2. `../plans/DECISION_REGISTER.md` — user's choices versus assistant proposals/open decisions.
3. `../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md` — complete system charter and architecture.
4. This handoff — history, decisions and takeover steps.
5. `../plans/IMPLEMENTATION_ROADMAP.md` — 24 phases P00–P23 with gates and dependency order.
6. `../plans/EXPERIMENT_PROGRAM.md` and `RELEASE_ACCEPTANCE_GATES.md` — methodology/closure conditions.
7. The original per-file SHA-256 integrity manifest — to be refreshed after deliberate modifications.
7. All `docs/` specs that relate to the area of work, plus always `INVARIANTS_AND_FAILURE_SEMANTICS.md`, `NUMERICS_QUALITY_AND_BENCHMARKS.md` and `SECURITY_AND_RELIABILITY.md`.

**Live source beats historical prose:** before claiming any progress on future turns, inspect the actual files and Git state if repo exists. No project Git repository had been initialized at initial documentation creation or at directory relocation. Source upstream GitHub heads change; pin specific revisions before importing/benchmarking.

### Subsystem docs and what they must answer

- `VISION_SCOPE_AND_PRINCIPLES.md`: supported product experiences, goals/non-goals, layering, generality.
- `RESEARCH_SOURCE_ATLAS.md`: all studied upstream projects, concrete repo/source paths, author benchmarks, limitations and license caveats.
- `MODEL_SEMANTICS_AND_COMPATIBILITY.md`: model architectures, operator registry, adapter protocol, status matrix.
- `OPERATOR_AND_FUSION_CATALOG.md`: operator-level math/effects/quant/shape, region fusion and safe fallback.
- `SEMANTIC_IR_AND_COMPILER.md`: typed state-effect SSA, SIR→EIR→DPI, verifier, lowering, plan cache and legalization.
- `COST_MODEL_AND_AUTOTUNING.md`: measured GPU compute/memory/IO/queue cost, plan variants, runtime adaptation and hypothesis testing.
- `DEVICE_KERNEL_AND_CODEGEN.md`: Rust GPU compiler decision/FFI, kernel families, `gfx1151`, CUDA/Metal/Vulkan/CPU, graph capture.
- `STATE_CACHE_AND_TRANSACTIONS.md`: KV/GDN/KDA/indexer/lookup state, immutable prefixes, COW, speculative commit, disk snapshots.
- `MEMORY_RESIDENCY_AND_IO.md`: physical RAM/GTT, expert slots/leases, fixed slabs, RAM/SSD tiers, disk-only n-gram and scratch.
- `SCHEDULER_BATCHING_AND_SPECULATION.md`: unified token budget, prefill/decode/verify batching, GPU MoE grouping, draft policies and exact sampling.
- `DISTRIBUTED_PARALLELISM.md`: tensor/expert/layer/P-D parallelism, topology, handshake, transaction, fault recovery.
- `FORMATS_QUANTIZATION_AND_LOADING.md`: GGUF/HF safetensors, tokenizer/template, quant provenance, safe parsing, lossless packing.
- `RUNTIME_SERVING_AND_UX.md`: CLI/SDK/server APIs, streaming, tools, introspection, metrics and resource clarity.
- `NUMERICS_QUALITY_AND_BENCHMARKS.md`: reference/quant/quality differences, real benchmark protocols, statistics and tests.
- `RUST_WORKSPACE_AND_ENGINEERING.md`: Cargo workspace and Rust source/FFI, NixOS/toolchains, CI, safety/contributor rules.
- `SECURITY_AND_RELIABILITY.md`: trust boundaries, GPU memory, untrusted model files, HTTP auth, private snapshots, distributed failures.
- `STRIX_HALO_OPTIMIZATION_DOSSIER.md`: Qwen/GLM/DeepSeek on `gfx1151`, memory tiers and kernel experiments.
- `PERFORMANCE_HYPOTHESES_AND_RISKS.md`: why it could be better, why it could fail, alternatives and kill criteria.
- `EXAMPLE_EXECUTION_TRACES.md`: end-to-end traces of fallback, Qwen resident, GLM streamed, coding agents, remote failure.

The documents intentionally overlap where necessary so individual agents can work from a focused spec without losing critical correctness or context. In a conflict, user decisions and independent tested mathematical source override a prose recommendation; register the conflict in a new decision record.

---

## 3. Research history — each project's actual contribution

### vLLM (https://github.com/vllm-project/vllm)
Token-budget continuous batching V1 scheduler, paged KV, prefix reuse, optimized attention and GPU graphs, distributed parallelism. Understand scheduling from computed tokens and available memory. Its strongest quality is high-throughput service architecture. **Do not copy Python/Torch dependency architecture** just to gain V1 scheduling ideas.

### SGLang (https://github.com/sgl-project/sglang)
RadixAttention/prefix reuse, HiCache GPU/RAM/storage, overlap scheduling, speculation (MTP/EAGLE/DFlash etc.), expert parallel and P/D disaggregation. Key: scheduling must know cached work and overlap CPU/GPU/IO, but HiCache stores **context state** whereas Maya stores **model expert weights**. These require separate caches sharing one budget.

### llama.cpp / ggml (https://github.com/ggml-org/llama.cpp)
GGUF, CPU/GPU native graph model adapters, many quant types, partial offload/mmap, simple CLI/server and broad device support. Key: portability/compatibility and model assets. Pitfall: generic graph/kernel gaps on new GDN/DSA/MoE shapes and CPU fallback for some GPU operations can hurt long-context speed. The new engine should be easy to run, not depend on llama.cpp as a hidden runtime.

### Specialized gfx1151 engine
Specialized Qwen3.8-Flash-Next `gfx1151` engine with deeply tuned kernels, quant precision split, resident core weights and paged n-gram table, trained speculation. Reference author reports ~1584 prefill tokens/s 8K, ~1517 at 131K, ~37.6 short serial generation and ~55–56 coding speculation on 128GB Strix (different versions/conditions). Native checkpoint ~62.1 GiB plus ~47.7 GiB paged n-gram. **Core engine source not published; EULA limits source reuse.** Treat benchmarks and public descriptions as independent behavior research. Cannot copy closed source.

### Gufo (https://github.com/gufo-org/gufo)
Inspectable ROCm `gfx1151` Qwen3.8 Flash-Next extreme kernels and GPU session state/rollback, long-context sparse attention, native batching and transparent rejected experiments. Published Oct 8–9 2026 single-user UD-Q4_K_XL AR ~1700.52 prefill/~25.95 decode at depth 0 and ~1478.66/~24.92 at 128K; repetitive MTP ~60.39 while mixed MTP ~31.32; C8 AR aggregate ~108.68. One warmed sample/point, specific comparator llama.cpp revisions. Deep value: accurate per-session GDN/indexer/MTP replay and benchmarking. **Not apples-to-apples versus that engine's quant and model layout.**

### Project Maya (https://github.com/mw00/project-maya)
GLM-5.3-Flash specialized 321B (18B active) with `src/core/glm_fast_path.cu`, GPU device expert pointer table, host-mapped miss ring, pinned DMA staging, cache slots protected per-layer/in-flight, CPU lane, ROCm `gfx1151` experimental. GLM has 45 layers, 34 KDA+11 sparse MLA/DSA, 42 routed layers × 288 = 12,096 expert blobs, top8, 4-stream mHC and trained MTP. Maya-S quant ~96.5 GiB; Strix 128GB author reports ~210–217 prefill, ~17.8 generation and ~99.9% expert cache hits when pool holds virtually all experts. Two discrete AMD GPU+MTP result ~33 tok/s different hardware/acceptance. Core transferable insight: **cache migration and physical memory topology** are first-class execution decisions. Maya service is not vLLM-equivalent high-concurrency native batching.

### Strata (https://github.com/mach10x/strata)
Prior Qwen3.8-Flash-Next (qwen4exp) specialized giant MoE with GPU/RAM/NVMe expert tier, GPU/CPU overlap and n-gram lookups. Distinguish large lookup rows vs expert matrix blobs. Learn placement policies, pipeline/layer split and speculation. Not portable model runtime.

### DwarfStar / antirez/ds4 (https://github.com/antirez/ds4)
Native C/ObjC/CUDA/HIP model-specific runtime, permissive MIT source with GGML attribution. Inspect `ds4.h`, `ds4_rocm_compat.cu`, `rocm/ds4_rocm_runtime.cuh`, `ds4_ssd.c`, `ds4_engram.c`, `ds4_tp.h`, docs/QA. Visible GitHub connector head `fc80bd695da76ee14bcb86820bf900a5cdfda806` dated 2026-10-07; reverify actual checkout. Model focus DeepSeek V4/V4.1, GLM 5.2/5.3/5.3 Flash, Qwen3.8 Flash-Next. Compressed V4 attention uses 128-token raw window with ratio4/indexed and ratio128 compressed older history. Disk-only Qwen n-gram ~95.37 GiB with Q2 main/MTP ~41.73 GiB, V4.1 Engram ~189 GiB. Dynamic SSD expert slabs, safe memory guards, DSpark (5 candidate tokens), true/exact sampling option, TP RDMA and layer PP, agent/HTTP/integration QA. Strix DeepSeek Q2 author measured prefill 187.26→292.86 after tuned IQ2/rocWMMA kernels. **Important limitations:** Qwen3.8 path Metal/CUDA but not ROCm in inspected docs; ROCm decode-island graph capture stub returns unsupported in `ds4_rocm_compat.cu`; certain concurrency is ordered fallback, not native batching. Learn both the strengths and the limits.

### Others
- a CPU/GPU heterogeneous engine: x86 AVX-512 expert computation + GPUs and layerwise prefill, high-precision massive checkpoints.
- a quant-aware inference project: quantized expert decode GEMV vs grouped prefill GEMM by routing shape.
- a kernel-dispatch library: attention/MLA/MoE kernel dispatch and autotuning patterns.
- a vendor graph compiler: AutoDeploy/model graph transformation and device execution compiler.
- a unified-memory tensor framework: unified-memory semantics; Apple and AMD hardware not identical.
- An independent Rust serving project: broad Rust serving with a Rust tensor library, competitive dense results and some large MoE gaps, proving Rust language alone doesn't solve kernel quality.
- A Rust tensor library: Rust tensors/model reference.
- CubeCL: Rust-authored GPU source and codegen; alpha/evolving, `gfx1151` compiler performance must be proven.

## 4. Hardware/model facts to carry into planning

| Model | Research shape | State/weight distinctions | Must test |
|---|---|---|---|
| Llama dense | GQA/RoPE/FFN decoder | ordinary paged KV | CPU/GPU parity, launch/throughput |
| Gemma 4 | sliding/global and MoE for 26B-A4B | different attention windows and 128 top8+shared experts in cited variant | template/precision/attention and grouped MoE |
| Qwen 3.5+ | hybrid GDN/full attention, optionally MoE | recurrent state + KV | chunk parity, state snapshots |
| Qwen3.8 Flash-Next | 48 layers, GDN/QSA 36+12, 512 top10, MTP, large ngram | resident compute weights vs disk lookup | `gfx1151` deep attention, MTP, memory, quant quality |
| GLM5.3 Flash | 45 layers, 34 KDA+11 sparse MLA, 288 top8 on 42 MoE layers, mHC | massive expert pool and recurrent/latent state | 92 vs 128 GB memory, cold/warm cache, KDA/DSA |
| DeepSeek V4 Flash | compressed time-axis attention + MoE, DSpark | 128 local window, 4:1 vs 128:1 historical | correct compression/indexing, spec economics |

The user targets older and newer family variants too; exact geometry MUST come from the pinned model config/checkpoint, not this coarse model-family table. `Qwen 3.5+` is not synonym for `Qwen3.8-Flash-Next`.

Physical device support is **model/quant/operation dependent**. A `gfx1151` with 92/96GB is not guaranteed to fit a 128GB reference model profile; model dense/KV/KDA/scratch/draft plus OS headroom matters. GPU-visible GTT does not equal available RAM.

## 5. The architecture — authoritative synthesis

```
frontends: CLI / API / Rust SDK / agents
    -> artifact loader / tokenizer & template
    -> MODEL SEMANTIC IR (typed math + explicit state transitions)
    -> OPTIMIZED EXECUTION IR (fusion/shape/layout/quant/placement)
    -> DEVICE PLAN (kernels/streams/graph/events/leases)
    -> runtime scheduler: token budget + prefill/decode/verify
                |                |
       prefix-state cache    global ResourceArbiter
                |            / | experts, KV, lookup IO, scratch
                v           /
          model/device executor (CPU, ROCm, CUDA, Metal, Vulkan)
          + online measurements, offline tuning, numerical oracles
```

### How a generic model runs
Known adapter maps checkpoint to typed semantic operators; portable CPU reference exists, and generic GPU/backend lowers all supported ops where available. Runtime plans and executes direct kernels without interpreting semantic graph per token. Unsupported novel op gives explicit error. Availability of GGUF alone cannot guarantee model execution.

### How an extreme model runs
Compiler recognizes exact legal model operator regions, quant, shape and architecture (e.g. Qwen GDN/QSA/MoE on `gfx1151`), resolves specialized Rust-authored kernels, preallocates leases/scratch and captures graphs where viable. Scheduler chooses a plan based on batch, context, memory and numeric policy. Resident GPU expert routing avoids host round-trips; disk misses enter safe async handoff. Non-specialized regions use valid fallback at canonical tensor/state cut points. State correctness survives switching.

### Why semantic IR can help but does not guarantee speed
Semantic IR exposes optimization legality, shared model definition and backend independence. Peak performance still requires exceptional hardware kernels, correctly priced memory/IO, efficient scheduling, reduced launch overhead, controlled fusion and robust compilation. A handwritten specialized/Gufo kernel can outrun generic IR output. The architecture **must permit handwritten Rust target kernels/whole-layer graphs** without any abstraction performance ceiling.

## 6. Key subsystem decisions and invariants

### Semantic IR
Three levels SIR→EIR→DPI. SIR includes SSA tensor data **and ordered effects** for KV/GDN/KDA/indexer/history/RNG. Compiler reorders pure ops only when state/effects and numerical policy permit. Device plans directly execute precompiled kernels; no per-token interpreted IR. Repacking quant layout differs from requantizing model.

### State and prefix
Commit only a consistent token frontier across all mutable state. A speculative rejected token cannot leak changes to KV/SSM/indexer/lookup/sampler. Prefix snapshots must include all model-specific state, not just KV. COW/frozen immutable prefixes shared among many agents, mutable tails private. Disk snapshot schema versioned with exact model/quant/position/renderer/bytes and checksums. State fallback may require explicit conversion/replay.

### Memory
One global budget, different cache policies for dense weights, routed experts, n-gram rows, KV and recurrent state. Strix physical RAM/GTT separated; avoid unsafe allocations. Cache slabs/leases/generations and GPU route pointer table; host-mapped request ring only for misses, never miss on every hit. CPU expert lanes only when measured favorable; UMA CPU/GPU share bandwidth. Prefill scratch may borrow optional expert-cache capacity only through safe eviction/lease.

### Scheduler
vLLM-like unified token budgets; SGLang-like prefix-aware admission; native batching and route grouping for MoE where profitable; prefill/decode/verify distinct kernel regimes. Multiple HTTP clients are not synonymous with native batching. Fairness/TTFT/P95 and throughput objective separated; no starvation or silent session evictions.

### Speculation
Target verifier authority, MTP/DSpark/EAGLE/small-model/prompt-lookup options when compatible; automatic controller may select depth/backoff by committed tokens per **complete** draft/verify/rollback/memory cost. Proposed default is distribution-preserving stochastic sampling, approximate/opportunistic mode explicit, pending policy confirmation. Exact same-plan greedy may be tested, cross-plan bitwise identity not automatically possible.

### Device kernels
All own source Rust, through candidate CubeCL or custom backend intrinsics when peak ISA needs it. Vendor runtime FFI policy OPEN; don't commit to a full C++ inference engine. Separate prefill and decode kernels, target-specific WMMA/MFMA, sparse selector, GDN/KDA, grouped MoE and full model graph. Generated ISA and physical GPU measurements determine whether any compiler is adequate.

### Distributed
Single-device default. Pipeline, tensor/expert parallelism, optional P/D; topology/memory-aware placement and frontier atomicity. Wrong model/hash or partial collective fail closed, rebuild from durable common prefix. No unauthenticated remote exposure by default.

### Quant/quality
Input GGUF/safetensors, known tokenizer/template/architecture, strict manifest and hashes, no HF remote code runtime execution. Automatic kernel/packing change may be lossless; lower precision always a different explicitly requested artifact. Use independent CPU/FP64 oracle, official checkpoint-based model evaluation and controlled A/B benchmarks; no fabricated improvement percentages.

### UX
One clean local `run/serve/inspect/bench` experience (example commands, not decided), OpenAI/Anthropic compatible APIs where actually tested, tool/reasoning/template correctness, visible backend/fallback/memory plan, local-only bind by default, no surprise model download or code execution.

## 7. Proposed development phases, all still NOT_STARTED

The road map `IMPLEMENTATION_ROADMAP.md` defines P00 through P23:
- **P00–P05 foundations:** repo/toolchains; Rust GPU feasibility; independent CPU oracle; artifact formats; typed SIR; EIR compiler and lowering.
- **P06–P09 usable general engine:** CPU executor, GPU backends, first complete Llama-style dense model and local CLI/server.
- **P10–P12 state/memory/hybrid/MoE:** resource allocator and prefix-state cache, GDN/KDA/sparse attention, expert tiering and grouped MoE.
- **P13/P14 extreme hardware targets:** Qwen3.8 and GLM5.3 on `gfx1151`, full quality/bench evidence versus reference projects.
- **P15/P16 speculation and high-throughput server:** exact target verifier, draft algorithms, adaptive policy, continuous mixed batching, multiuser MoE grouping.
- **P17–P19 breadth:** Gemma 4/Qwen3.5+/DeepSeek and newer, multimodal, distributed parallelism.
- **P20–P23 finish:** autotuner and performance lab, security/release, benchmark qualification, next model compatibility.

Each phase lists prerequisites, tasks, exit criteria and failure/stop strategy. Do not implement all phases in one unreviewable change; begin with P00/P01 feasibility and P02 reference state correctness after explicit user instruction.

Milestones:
M0 architecture/ISA proof, M1 portable useful engine, M2 general hybrid/MoE, M3 Strix specialist, M4 throughput, M5 distributed+quality, M6 extensibility.

## 8. Research/benchmark discipline and important caveats

Performance is a *benchmarking claim*, not guaranteed by having Semantic IR. External measurements used different models, quantizations, ROCm, prefill chunk/context, CPU/GPU memory and speculation acceptance. Always match checkpoint hash/quant and quality, device/driver/power, new prefill tokens vs cached context depth, output length, warm/cold cache, concurrency, sampling and draft accept rate. Report TTFT and P95, aggregate committed tokens/s and resident bytes, not only peak t/s. Use several alternating A/B repeats, retain raw artifacts and both winning and losing results.

Examples of critical distinctions:
- A native-quant specialized engine versus Gufo Unsloth Q4 versus DwarfStar Q2 **not** same checkpoint/precision.
- ~60 tok/s repetitive MTP may drop to ~31 tok/s mixed text; not serial decode speed.
- Maya all-resident GLM performance on 128GB may degrade sharply when 92GB forces SSD expert reads.
- vLLM/SGLang enterprise throughput on H100 does not by itself predict Strix Halo on ROCm.
- `hipMemGetInfo` GTT capacity can be virtual, not usable RAM.
- Graph capture availability must be tested per backend; DwarfStar ROCm eager path is not proof HIP cannot graph capture.

## 9. What has actually been created, and what has not

**Created and relocated:** the full Markdown design corpus under `/home/pvl/spaces/abird/src/zerogravity/.agents/docs` and `.agents/plans`, plus project README. The user explicitly corrected the initial directory placement and the archive was moved. The files contain user decision IDs, original research source URLs, architecture and subsystem specs, risk/cost models, detailed phases, experiment and acceptance gates.

**Not created:** Rust code, Cargo workspace, kernel compiler implementation, release, benchmark output, CI, models, Git commit/init or tool modifications to unrelated repositories. No promise of a faster-than-specialized-engine result has been experimentally validated. If a future agent sees any implementation after this handoff, inspect and update status from **live tree**, do not claim it was part of the original documentation delivery.

## 10. How next agent should begin

1. Connect to the actual available dotlink connector; this creation used **dotlink l5**. Connector launch directory observed `/home/pvl/spaces/abird`.
2. Confirm `zerogravity/` exists, list files, read `START_HERE`, this handoff, `DECISION_REGISTER`, `ZEROGRAVITY_MASTER_PLAN`, roadmap and relevant subsystem docs **in full**.
3. Inspect `git status --short --branch`, `git rev-parse HEAD`, `git worktree list` **only if** a Git repository has since been initialized. At creation no project .git.
4. Decide the requested scope. If user asked design update, edit docs only. If coding/phase requested, take the next dependency-approved bounded phase and write validation plan first.
5. Pin and inspect live official source/repo snapshots before assuming current model geometry or upstream benchmarks. Honor source licenses.
6. For P00/P01, preserve future extensibility but keep code small. First meaningful GPU experiment is Rust-authored gfx1151 Q8/IQ2 MMVQ/WMMA and one stateful GDN step with CPU oracle. No giant serving framework before this proof.
7. After code, test CPU and physical relevant GPU, run performance and quality controls, update roadmap status with commit/test/evidence, decision register, affected specs and handoff.
8. Never report phase complete without fresh live-tree evidence.

## 11. High-priority open decisions

Explicitly unresolved, don't invent a user's selection:
- O-001 project license;
- O-002 production hardware target order and v1 scope;
- O-003 strict interpretation of Rust-only around vendor driver/math APIs;
- O-004 primary GPU Rust compiler/CubeCL versus custom low-level backend;
- O-005 first model release acceptance target and quant checkpoint;
- O-006 exact quant/format support matrix;
- O-007 cross-backend determinism policy;
- O-008 public SDK/API ABI;
- O-009 remote transport/auth policy;
- O-010 OS/toolchains/release CI support;
- O-011 model-quality thresholds;
- O-012 SLO and concurrency defaults;
- O-013 declarative vs Rust model adapter interface;
- O-014 model/kernel plugin trust;
- O-015 when to initialize Git/release;
- O-016 CLI branding and public flags.

Some can be resolved by benchmark before asking preference; explicit product and policy choices should be presented to user when they materially block implementation. Do not re-ask user facts already provided.

## 12. Source/decision preservation policy

New architecture decision record must include date, source/user instruction, old assumption, rejected alternatives, changed invariants, source/benchmark proof, next actions and superseded decision ID. Documentation can evolve, but historical reasoning and negative experiments should not disappear. External references must retain URL, commit or dated snapshot and source license. Model weights and private prompts are not checked into this project by default.

The long-term engine must remain both **maintainable and optimizable**: no abstraction may impose a hot-path performance ceiling, but optimized kernels must be lawful under the same model semantics and state contract. Code-level specialization can always replace a whole IR region if it is independently proven correct and efficient.

---

## 13. Self-contained next-agent starting prompt

```text
You are taking over the zerogravity Rust inference-engine project in
/home/pvl/spaces/abird/src/zerogravity using the available dotlink connector.

Before any implementation, read IN FULL:
- .agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md
- .agents/plans/DECISION_REGISTER.md
- .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md
- .agents/plans/IMPLEMENTATION_ROADMAP.md
- .agents/plans/EXPERIMENT_PROGRAM.md
- .agents/plans/RELEASE_ACCEPTANCE_GATES.md

Then read all relevant .agents/docs, especially:
SEMANTIC_IR_AND_COMPILER, STATE_CACHE_AND_TRANSACTIONS,
DEVICE_KERNEL_AND_CODEGEN, MEMORY_RESIDENCY_AND_IO,
SCHEDULER_BATCHING_AND_SPECULATION, MODEL_SEMANTICS_AND_COMPATIBILITY,
INVARIANTS_AND_FAILURE_SEMANTICS, NUMERICS_QUALITY_AND_BENCHMARKS,
RUST_WORKSPACE_AND_ENGINEERING, SECURITY_AND_RELIABILITY.

Inspect the live directory, repository state (if Git exists), and all source
before trusting any documented phase completion. All implementation phases
were NOT_STARTED at creation of these plans.

Preserve the user's requirements: name zerogravity; Rust-authored engine
and owned kernels; general semantic portable fallback; vLLM/SGLang-class
serving ambition; llama.cpp-like ease/model breadth; maximal eligible
hardware specialization for Qwen3.8-Flash-Next, GLM-5.3-Flash, Gemma 4,
Qwen3.5+, Llama and others, especially AMD gfx1151/Strix Halo.

Keep proposals/open decisions distinct from explicit user approval.
Do not claim benchmark wins without physical tests; do not silently change
precision or sampling; do not copy proprietary engine code.

Follow the narrow dependency-ordered roadmap and write rigorous numerical,
state, memory, security and end-to-end performance evidence. If the current
request is design-only, work on docs and do NOT begin Rust code.
```

## 14. Closure of THIS handoff

The right next action depends on the *next* user instruction: either review/refine these specs or begin P00/P01 once explicitly requested. Everything necessary to orient an engineering agent—the user goals, historical research, architecture, subsystem contracts, phase dependencies, risks, validation criteria and open decisions—has been deliberately preserved on disk.

````

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/prior-art/PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md

Original UTF-8 bytes: 32360; SHA256 `b4b4e54dcad093c1612c020040994e403978d01e8efb7b96ce4df7da76e6ff1a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Prior-art archive — Historical v2 unified master architecture

**Archived 2026-10-12 by user's project-neutral redesign directive. RESEARCH ONLY / NON-NORMATIVE.** This is the historical pre-v3 complete master text, preserved for design provenance, implementation heuristics, older decisions and external references. **Do not follow source-project-named design instructions from here as canonical**; new authoritative system-independent contracts live in `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md` and the rewritten `ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`. Research source and license evidence is separate under `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`.

---

# zerogravity — Comprehensive Unified Master Architecture and Product Plan (v2)

**Updated:** 2026-10-12 (v2.1: Gufo XDNA2 + upstream media task pipelines). **Path:** /home/pvl/spaces/abird/src/zerogravity/. **Status:** unified researched technical design; **no implementation or zerogravity benchmark exists yet**. Supersedes the original v1 interpretation where incompatible with the synthesis. Explicit user intent is authoritative in `DECISION_REGISTER.md`; unapproved product/license/library choices remain open.

**Essential sources:** `.agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md`, `.agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md`, `.agents/docs/SEMANTIC_IR_AND_COMPILER.md`, `.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md`; extensive pinned reference analysis at `/home/pvl/spaces/abird/research/inference/` (read-only source corpus). Implementation roadmap and handoff describe execution sequence, not completed work.

## 0. Mission

Build **one Rust-authored inference compiler + serving runtime** that can run broad known model semantics reliably through portable reference and generic optimized backends, while selecting extreme model/device/workload-specific kernels and dataflow when applicable. The intended combined strengths:

- **llama.cpp**: GGUF/safetensors interoperability, broad model adapters, native CPU/GPU deployment and approachable single-binary experience;
- **vLLM / SGLang**: token-aware continuous batching, prefix state sharing, asynchronous overlap, efficient KV and distributed serving;
- **Gufo / Maya / Strata / DwarfStar**: exceptional `gfx1151` hybrid/sparse/MoE computation, low-bit packed kernels, large model expert/lookup storage tiers, model-specific sampling and true state rollback;
- **An ahead-of-time ML compiler / a tensor compiler / MLIR / a Rust graph-search project / a kernel-dispatch library**: separation of semantic graph, legal tensor rewrites, async resource/stream plans, target-kernel tuning, model-load-time specialization and serving-mode-aware cost;
- **Rust ecosystem**: CubeCL, independent Rust tensor/serving projects, possible Rust GPU/CUDA Tile and SIMT compilers, Rust CPU SIMD and safe explicit device ownership.

**Ambition:** reach the Pareto-best performance/fidelity/latency/capacity frontier for each eligible model/hardware/workload and sometimes surpass specialized incumbents by co-optimizing kernel compute, request batching, expert/prefix residency, IO overlap and speculation. It is a hypothesis needing matched physical benchmarks, **not a guarantee that an IR magically outruns every runtime**.

**Foundational rule:** mathematical semantics once, state effects explicit, lower to specialized direct executable regions with no per-token graph interpretation; maintain correct fallback when specialization absent and openly refuse unknown semantics/insufficient resources.

## 1. User-confirmed scope versus working choices

**User-confirmed:** name zerogravity; code at exact root above; Rust engine and Rust-authored custom kernels intent; new general runtime fallback; Llama, Gemma 4, Qwen3.5+ and specialized Qwen3.8-Flash-Next/GLM-5.3-Flash and further modern families; `gfx1151` Strix Halo focus; target vLLM/SGLang throughput, llama.cpp convenience and Gufo/Maya/DwarfStar speed; maximize possibility of surpassing existing engines and maintain clean long-term adaptability; complete `.agents/{docs,plans}` handoff; combine prior `research/inference` and project plans.

**Working technical design, not user-approved product choices:** single logical engine owner, SIR/RIR/KIR/CIR, bounded compiler, typed state effects, general/exreme KV variants, global ResourceArbiter, route-aware batching, exact sampled speculation by default, model store and source manifest, async IO and per-response mode receipt, CPU oracle first. More details in reconciliation ledger.

**Unresolved:** license, binary name and public UX/paths, primary GPU kernel compiler and whether vendor math FFI is allowed, supported release backends/first models/quant, performance SLOs, release credentials/plugin security and distributed scope. These cannot be silently declared accepted solely because historical research labeled them RESOLVED.

## 2. Unified system architecture

```text
CLI / HTTP adapters / Rust SDK / model-management tools
     |
     +-> ModelStore / tokenizer / chat+grammar / multimodal processing
     |       |
     +-> Artifact parser & verified Architecture/ModelRecipe
                 |
                 v
        [SIR] Semantic Model IR
        pure SSA tensors + explicit versioned state/effect tokens
                 |
                 v
        [RIR] Legal Region Optimizer
        pure math rewrite/fusion, quant packing, symbolic shape/layout
                 |      \
                 |      [KIR] kernel providers:
                 |        Rust CPU, CubeCL, Rust HIP target, a Rust tile compiler,
                 |        a Rust SIMT/PTX compiler, Vulkan/Metal (as verified)
                 |       /
          [CIR] async resource/command planner
          memory leases + transfers + SSD rows/experts + GPU events
          graph capture/IO barriers + state commit points
                 |
                 v
         Immutable ExecutableBundles (verified variants)
                 |
                 v
 HardwareProbe -> pure HardwarePolicy+MemoryLedger <- ResourceArbiter
                 |                                    |
                 v                                    |
        EngineCore (single logical request owner) ----+
        token budget + admission + prefix cache + speculative policy
                 |
           StepPlan / typed WorkerDelta
                 |
                 v
        Device workers -> direct kernel/graph launches
                 |
        event-complete provisional state
                 |
       EngineCore target verification & atomic token frontier commit
                 |
       Stream committed tokens / metrics / mode receipt
```

**Important distinction:** KernelPlan records validated kernel options, HardwarePolicy sets machine-level placement/reservation, ExecutableBundle is the compiled region/command graph, StepPlan is actual requests/tokens. None is the semantic model. A model parser cannot choose vendor GPU kernels; a kernel cannot secretly alter routing/top-k/precision/sampling.

This architecture is based on an ahead-of-time ML compiler's Flow→Stream→HAL and a tensor compiler's Relax→TIR separation *conceptually* while staying a small Rust-native compiler, not embedding their C++/Python runtime or trying to implement a general MLIR clone.

## 3. Semantic IR and compiler legality

### 3.1 Model semantics

SIR represents: dense and sliding/local GQA/MQA, RoPE/M-RoPE, RMSNorm, quant projections and GLU, routed/shared MoE with exact route scoring/sum, Gated DeltaNet, Kimi Delta Attention, Qwen Sparse Attention, absorbed/noPE MLA+DSA, DeepSeek learned time-compressed attention, residual/hyperconnection/mHC, n-gram/Engram lookup, MTP/draft, vision/audio encoders, embedding/rerank pooling.

Each `SemanticOpSchema` declares:
- logical shapes and symbolic constraints, logical versus stored/packed dtype/quant;
- the precise formula and model version/position rules;
- state-domain reads/writes, causal ordering and initial/reset/transition/snapshot/fork/commit/abort;
- permitted pure algebraic rewrites and numerical policy;
- CPU reference and golden source logits/state; known backend lowerings and required resources.

Model recipes are built from validated config/tensor names and operator schemas, not hard-coded GPU targets. Novel model math without a reference op returns `UnsupportedOperator`. An entire model can fall back to generic execution **only when all semantic operators are implemented and resources suffice**.

### 3.2 Optimized Region IR

RIR marks pure subgraphs and mutable effect edges. Legal rewrites: projection/norm/quant fusion; no redundant materializations; local shared-activation quantization; attention/indexer fused prelude; layout packing; grouped routed expert execution; KDA/GDN chunk and decode form; state snapshot region cuts.

Optimization forbids hidden top-k changes, mHC approximation as generic gated residual, cross-quant prefix cache reuse and floating rearrangements outside numeric/quality policy. Fusion is chosen based on full-region critical path, not maximum operation count in one kernel.

### 3.3 Kernel authoring IR and registry

A target kernel provider advertises `KernelCapability`: region pattern/hash, hardware ISA, layout/quant/shape, numeric tolerance, state effects, scratch/leases, capture mode and performance evidence. A candidate may be a hand-tuned Rust `gfx1151` whole-layer implementation, generic CubeCL, Rust NVIDIA Tile/SIMT compiler or CPU reference; all satisfy same SIR contract.

No universal GPU kernel source required; **Rust-origin model and kernels** with backend-specific instruction control. External vendor driver/runtime/math FFI interpretation is an open user-policy detail; never quietly bundle another C++ model runtime.

### 3.4 Async Command/Resource IR

CIR records `Acquire,Release,Copy,Dispatch,PrefetchExpert,ReadLookup,PublishResidency,Wait,Signal,Commit,Abort,PeerCollective` with timepoint edges, buffer alias/lifetime, device affinity and error-state dependencies. It can legally overlap independent CPU/GPU/SSD work and reduce scratch peak before executable graph capture. Device pointer generations/leases guard against expert eviction and unsafe state/layout conversion.

Once compiled, executable variant is direct kernel/graph launch, **no semantic IR traversal during normal autoregressive steps**. Runtime policy picks among prevalidated variants; local JIT-on-miss is opt-in/bounded or runs off interactive critical path.

## 4. General path and extreme path — one implementation spine

Capability tiers:
- **Reference:** canonical Rust CPU tensor and state implementation; independent official model/llama.cpp fixture validation; can use scalar/FP64 diagnostics.
- **Optimized generic:** tuned CPU SIMD or GPU kernels for common projection/attention/MoE, paged KV, batching, compressed/latents when implemented.
- **Extreme:** compiled closed model-region on selected real hardware/quant/shape, fused recurrent/sparse/expert/pointer-table kernels, resident hot path, graph replay and nonresident IO if needed.

Fallback is **per-region where buffer layouts/state transitions are compatible**. Otherwise transition at an explicit canonical checkpoint with validated layout conversion/replay; runtime never silently reinterpret compressed MLA as expanded KV or advance GDN recurrence to rejected draft tokens. A CPU op fallback may be correct but too slow to claim peak performance; show it in `inspect` and mode receipt.

## 5. Physical resources and five+ lifetime classes

- model immutable weights/packed layouts;
- sparse expert slots with generation and source on SSD/RAM/GPU;
- session mutable KV/recurrent/indexer/ngram/sampler/multimodal;
- immutable shared full-prefix snapshots;
- execution scratch/provisional verify state/IO staging;
- distributed rank/transport buffers;
- compiler cached executable/capture graph and code artifacts.

One `ResourceArbiter` enforces real RAM/VRAM/UMA/cgroup/pinned/OS reserve and independent device addressability. **ROCm GPU-visible GTT capacity is not physical free memory.** Each cache has separate policy: always-used dense resident, routed MoE reuse tiers, ngram/Engram bounded disk row reads, paged attention or compressed indexer, state snapshots and scratch borrowing via leases.

Adopt research's stable expert slots/layer quotas, host-mapped ring, async pread/O_DIRECT when measured, hotlist hinting (not correctness), CPU lane only where measured, one-blob optional source pack and structured `ExpertStore` prefetch/publish. On one Strix Halo CPU and GPU contend for LPDDR; don't assume CPU expert lane helps.

**KV:** research's intrusive paged pool with chained hash and optional radix over same physical allocation, and extreme contiguous reserve-all variant, all under global reservation. For hybrid attention/recurrent models a general *state-group coordinator* computes common legal cached prefix frontier across any number of groups, including compressed finalization alignment and modality; vLLM hybrid state docs show incomplete generality for more-than-two groups. Cache identity binds canonical tokens/rendering/model/quant/position/state/schema/tenant+LoRA, not just rendered SHA1 bytes.

## 6. Runtime serving, scheduler and concurrency

Adopt one dedicated `EngineCore` logical owner and async `EngineMsg/EngineEvent`, persistent worker `InputBatch`, diff-based `StepPlan` and `WorkerDelta`. Worker owns device stream/alloc/captures; client never sees raw state. CPU pools handle tokenizer/tool grammar preprocessing; bounded host IO services handle SSD/prefix snapshots.

Scheduling uses vLLM-style unified target token budget plus input/encoder/physical memory/fairness budgets. Chunked prefill, 1-row decode and small speculative verification choose different kernels. Coalescing, 2-in-flight steps, native grouped MoE and expert affinity are measured optional policies, not permanent 2ms or 128/2048 hard constants. Queue priority and decode ITL protect user-facing latency.

**Cost-aware innovations**: plan admission weighs prefix reuse vs recomp/IO, expert residency and cache pressure, draft weights+verification cost, predicted route overlap, dynamic scratch and GPU/CPU+SSD transfers. Group selected routed experts on GPU where applicable; no host routing sync for every resident expert hit. Optimize aggregate committed output and P95 under quality, memory, fairness and power constraints. Peak performance may be different for single stream, concurrency 8, 32K cached context or disk misses.

## 7. Speculation, state transactions and cache correctness

`DraftSource` may be trained MTP, DSpark-compatible, EAGLE/draft model or zero-weight prompt/n-gram proposal. A dedicated `TargetVerifier` computes authoritative logits/states; `CommitPlan {FullDirect, Prefix(n), Replay(n)}` is a *physical execution mechanism* within a type-safe session transaction. Each verifier candidate has provisional multi-domain states. Commit publishes exactly accepted prefix and corrected sampler token, discarding rejected KV/GDN/KDA/indexer/ngram/modalities/RNG/grammar changes.

Sampled speculation should preserve target model distribution by default where implemented; approximation/opportunistic mode explicit. Exactness modes are distinct: fixed-plan greedy identity, reference numerical tolerance, cross-plan quality, stochastic distribution. No blanket bitwise identity across different GPU reduction orders. Acceptance policy must be measured per model/hardware and disabled if net worse.

Full hybrid-state immutable prefix caching and disk snapshots must be checkpoint/schema/version/quant and renderer compatible; copy-on-write or deferred materialization to avoid a full recurrence matrix copy per branch. A small frontier ring can be logical O(1) only if physical state representations allow it; otherwise copy/replay at bounded cost.

## 8. Formats, model store, grammar and new modalities

Input GGUF v2/v3 and supported safetensors/hub variants. Strict model adapter/quant dialect validation, no `trust_remote_code`, content-hash manifests, resumable atomic downloads and GC. Derived acceleration pack optional/regenerable with model fingerprint; research provisional public name "APK" is not automatically adopted (Android collision); own on-disk binary spec gets byte-checked sizes/version before writing code.

A complete convenience layer needs: byte-exact tokenizer and pre-type handling, Jinja-compatible model chat templates, incremental tool parse/reasoning, JSON-schema/grammar masking (a grammar library or verified alternative), sampled stop and grammar FSM transaction, one clean CLI/SDK, OpenAI API and Anthropic/Responses only with conformance, diagnostics `doctor/inspect`, mode receipts/metrics and no unsafe network defaults.

Dedicated `PoolingSpec` runner for embeddings/reranking (no decoder loop/KV), token/mean/CLS/LAST pooling, normalize and Matryoshka dims. LoRA adapters require per-request generation, distinct prefix cache keys, kernel shape and resource budget; defer high-concurrency multi-adapter fusion until tested. Multimodal encoder/processor/projector owns spans/content fingerprints and separate encoder budgets, avoids splitting noncausal spans; audio/video/diffusion can be future semantic extension with no automatic release promise.

## 9. Distributed and heterogeneous

Single physical GPU baseline lean. Optionally partition layers, tensor arithmetic, experts or prefill/decode service across devices/machines, chosen by interconnect/compute/state-transfer measurements. Worker rank must handshake model/quant/layout/semantic/schema/provisional frontier, protect buffers and commit same accepted token prefix or fail and rebuild. No implicit unauthenticated remote exposure. Copying only ordinary KV between hybrid prefill/decode ranks is insufficient.

## 10. Rust ownership, source policy and maintainability

Start with **few coherent Rust crates (~8–12)** for `artifact/model`, `ir/compiler`, `kernel/device`, `memory/state`, `runtime`, `service`, `validation`, plus binary; split as independent release/compile/security seams justify. Avoid historical 20–25 crate scaffold before one end-to-end vertical slice.

One authoritative trait/schema per axis: ArtifactReader, SemanticOp, CompilerPass/KernelProvider, DeviceBackend, ResourceArbiter/KvPool, ExpertSource, DraftSource/TargetVerifier, state codec, Encoder/Pooling, Placement/Transport, frontend adapter. No giant dynamic plugin host required: built-in statically linked Rust providers first, externally loaded executable kernels only after signed/trusted plugin security design.

Technology adaptation test: add a novel model family or target GPU compiler without editing unrelated HTTP/tokenizer/cache implementations. Plan/cache keys versioned by semantic/compiler/driver/quant/numeric/layout/shape. Document data-migration, deprecated operator version and change protocol. Compiler backend version upgrades require kernel parity and model-quality retests, not just Cargo build.

## 11. Physical benchmark + quality governance

Research comparisons must match exact checkpoint hash/quant/position/mode/device/driver/power/context/batch/acceptance and benchmark times. Source Gufo/Maya/DwarfStar numbers remain **external author reports**. For zerogravity, measure:
- kernel microbench and machine-instruction efficiency/occupancy;
- prefill (new tokens vs cached depth) and target AR/committed speculative t/s;
- TTFT, P95 ITL, aggregate throughput C=1/2/4/8/16/32;
- memory headroom, KV/SSM/indexer footprint, expert cache hits/miss critical path, NVMe bytes, power;
- CPU/reference tensor/logit parity, long contexts, tokenizer/tool/grammar, quality NLL/KL/retrieval;
- prefix snapshot, arbitrary speculative prefix acceptance, cancellations, OOM/fail-closed, rank failure.

Use ABBA/BAAB alternation and confidence bounds, full-model end-to-end validation and negative experimental records. One successful microkernel does not establish engine superiority. Dataflow cost model counts overlap and critical path, not naive sum. No silent quality upgrade/downgrade.

## 12. Correctness, security and failure invariants

1. No visible token without atomic publication of valid model and sampler state frontier.
2. No incompatible prefix hit across model/quant/adapter/modalities/position/state scheme.
3. No slot or buffer recycle while in-flight async work has a lease; generation checks prevent ABA.
4. No stateful rewrite/fusion across incompatible effects or unverified numeric assumptions.
5. No expert prefetch hint affects routed expert choice; cache miss aborts on IO error.
6. Physical memory reservation includes OS headroom and permanent/temporary allocations; GPU virtual address is not physical capacity.
7. Kernel fallback may only occur at type/state-conversion compatible boundary; otherwise replay from canonical committed prefix.
8. All model artifact and API input fields are bounded; unknown model ops cannot execute untrusted code.
9. Distributed rank mismatch or failure cannot publish half-committed target output.
10. Every throughput claim includes quality, model/quant/context/concurrency and verified source; no inflated draft tokens as completed output.
11. Serving auth defaults localhost, no unsafe tool execution or untrusted model extension code; snapshots private.
12. Model/kernel compiler/storage/HTTP adapter changes preserve independent owner boundaries and test evidence.

## 13. Research decisions reconciled

See `.agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md` for exact historical research council D-01…D-20 dispositions. Key decisions:
- adopt research `EngineCore/StepPlan/WorkerDelta/KvPool/ExpertStore/PlanSource/KernelPlan` as **implementation guidance**, not direct frozen Rust API;
- modify `ModelSpec::kernel_plan(hw)` into semantic model recipe and compiler/kernel provider ownership;
- adopt speculative `CommitPlan` only inside typed full-state transactions;
- retain GGUF canonical and derived acceleration pack but revise broken 48-versus-54-byte disk-KV schema before implementation;
- keep strict Rust source requirement and broaden kernel candidate research to CubeCL, a Rust tile compiler and a Rust SIMT-PTX compiler as of September 2026;
- distinguish hardware policy, compiled execution bundle and per-step runtime scheduling;
- correct overly broad "any model any engine runs" claim;
- do not treat outdated local-box assumptions, old release dates, 75–125 engineer-month estimate or an unapproved license as user facts.

## 14. Implementation dependency strategy

**Do not start by implementing 25 crates or an unbounded compiler.** Follow `IMPLEMENTATION_ROADMAP.md` with the integrated Stage 0 proof:

- Physical `gfx1151` kernel/ISA and alternative Rust compiler bakeoff, CUDA representative second target.
- CPU GGUF tokenizer/quant/mathematical oracle and tiny dense model.
- Minimal typed SIR including GDN and one sparse/indexer state effect; RIR legalization.
- Minimal CIR with leases, two async tasks/device events and simulated I/O errors.
- Research paged KV/core scheduler/frontier fuzzer and `CommitPlan` typed rollback.
- Complete vertical slice: checkpoint → exact adapter/SIR → compiled CPU + one GPU → correct tokens via same EngineCore; forced specialization miss to legal fallback.
- Serving-mode-aware kernel tuning and model quality harness.
- Only after these succeed: advanced MoE tier, rich model breadth, MTP, full async graph capture, distributed and multimodal.

The existing detailed 24-phase ZG-P00…ZG-P23 remain stable IDs, with new consolidated Stage 0 proof mapping and gates in `IMPLEMENTATION_ROADMAP.md` and `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`. **All phases currently NOT_STARTED**.

## 15. Risks and disconfirming tests

Highest risks: Rust `gfx1151` compiler codegen; source-language/vendor FFI policy; semantic/effect legality and full-state rollback; runtime C IR overhead; physical UMA overcommit; prefix/cache alias; unreasonable breadth; too many unproven optimizations; benchmark mismatches; external code license and parser security.

A verified 2026 claim that no Rust compiler emits adequate code for a key operator does not require abandoning SIR. It requires improving or extending the kernel provider backend while retaining Rust-authored source and independent fallback. If a proposed whole-system scheduling innovation adds complexity but shows no repeatable net gain at representative latency, disable it by default and retain the simpler plan.

## 16. Authority and handoff

**Canonical reading:** the agent playbook → decision register → this master → reconciliation and comparative research → `ASYNC_RESOURCE_COMMAND_IR.md` and EngineCore protocol → domain specs → integrated Phase 0/24-phase roadmap → handoff. The research corpus is an important read-only evidence/benchmark library, not a competing authoritative implementation plan. Update decision register/roadmap/handoff with actual proofs when coding.

**Current state:** this is **documentation and source research**, not a functioning runtime, compiled GPU kernel, installed toolchain, benchmark or Git commit. Performance remains an ambitious goal to validate. The design preserves peak specialization freedom without sacrificing known-model semantic portability or system maintainability.

## 17. October 12, 2026 — Gufo fork, XDNA2 and generalized task pipelines (v2.1)

**Latest explicit user decision U-016:** include the implementation techniques in [mikealanni/gufo](https://github.com/mikealanni/gufo) and the [October 11 Strix Halo GPU+NPU prefill report](https://www.reddit.com/r/StrixHalo/comments/1x3gtdh/qwen38_flashnext_prefill_past_1800_toks_on_strix/), ensuring they fit **natively into our compiler and execution pipeline** rather than adding a separate specialized server. The upstream [gufo-org/gufo](https://github.com/gufo-org/gufo) is a distinct existing reference with additional ASR, TTS, image and video task pipelines. Source-pinned technical investigation: [GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md). Full integration design: [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

### 17.1 Three heterogeneous execution strategies, one SIR

1. **Intra-operator projection-row partition:** on the same APU, GPU handles some output channels while XDNA2 NPU executes independent GEMM row groups. Source-demonstrated Qwen3.8 Flash-Next prefill partitions GDN z, attention Q/gate and shared expert gate/up; downstream values join at the first consumer. RIR proves exact row coverage and legality, KIR supplies device-specific candidates, CIR schedules NPU+GPU compute and cross-API fences. No XDNA name or hardware status appears in the trained model's semantic math.
2. **Routed-expert owner partition:** separately demonstrated by a discrete-GPU+iGPU expert-ownership approach with faster discrete GPU hot expert ownership, slower/larger UMA iGPU cold expert ownership and optional SSD tier. Sends selected activations and partial results across devices, not whole expert weights every token. Compute owner and residency/state owner are distinct; model's original MoE top-k and weights are unchanged.
3. **Layer/tensor/phase partition:** usual GPU/CPU/distributed pipeline, including prefill-vs-decode and multimodal encoder stages; requires explicit state and layout handoff and topology cost model. A plain GPU-only path remains correct and resource-efficient.

These are three **alternative/composable CIR/RIR placements**, not three independent runtimes. A qualified hardware profile may use several accelerators and storage tiers simultaneously, but the default never assumes availability or benefit.

### 17.2 Hardware support, numeric precision and safety

The supplied fork author reports Qwen3.8-Flash-Next **UD-IQ4_XS** on 128GB Strix Halo: **1,698 GPU-only → 1,821 GPU+XDNA2 prefill tokens/s at 16K** (+7.2%), 32K NPU 1,783 and 40K cold served 1,637. This is **author-reported prefill**, not decode, not a zerogravity benchmark and not proof of universal superiority. Limited 30 greedy-prompt matching outputs do not prove bitwise logits or stochastic distribution preservation. In the source, the NPU GEMM is a fixed **4096×2560×1024** block-float (`v8bfp16ebs8`) kernel with explicit XRT pinned BO and HIP callback dependencies; quantized compute error must be independently qualified. The AIE program source is Python MLIR-AIE/C++, not Rust. Rust source policy requires investigation of Rust-authored AIE lowering and separate XRT FFI license review before executable NPU support.

**Critical safety default:** the author reports hard whole-host machine-check hangs when the GPU fabric clock changes during NPU+GPU work. Manual held clock and power management are environment-specific and can increase idle power. **No automatic privileged tuning, no assumed BIOS/firmware safety, and no NPU offload enabled on unqualified machines.** DeviceCapabilities must represent SoC firmware/XRT/ROCm/safety state and support explicit experimental opt-in. If absent/unsafe/unsupported-shape, the compiler selects GPU-only region without user-visible model changes. A hardware hang is not recoverable by an ordinary inference timeout; physical tests belong on isolated approved machines.

### 17.3 TaskProgram for ASR/TTS/vision/diffusion/video

Upstream Gufo research demonstrates why we must not force every inference model into `EngineCore::next_token`. Introduce **`TaskProgram` with typed stage DAG + iteration + state + output contract** over the same SIR/RIR/KIR/CIR/ResourceArbiter:
- LM: prefill + causal tokens + optional MTP/DFlash2/DSpark speculation.
- ASR: audio preprocessing, speech encoder and model-specific decoder/transcription (potential NPU encoder GEMM, currently unproven).
- TTS: text conditioning, acoustic frame generation or flow, vocoder and audio streaming, speaker/style/voice state as required by model.
- Image generation/edit: conditioning + iterative denoise/flow/sample step + image decoder; separate iteration frontier from causal language tokens.
- Video/audio generation: temporal/frame state, multimodal encoder/denoiser/output-mux stages and bounded streaming resources.
- Embedding/rerank: encoder/pool/score without autoregressive KV/sampler.

These use a **common tensor/operator/kernel/async resource pipeline**, with different task loops and per-stage progress and media output; no media model becomes "supported" merely because upstream Gufo implements it. Each model/task/quant/device has its own acceptance and output fidelity. New accelerators and pipelines can be added without touching existing text server/ModelRecipe logic outside genuinely new operators.

### 17.4 Integration and acceptance

Maintain original P00–P23 identifiers; add an *optional but explicit* `S0-NPU-A…F` feasibility track in [INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md](../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md), model-specific `P07/P11/P13` NPU support and `P17/P18` TaskProgram extension gates. NPU hardware safety/source policy is **not a hard requirement to ship a portable CPU/GPU engine**. Tests include fake GPU/NPU execution graph with row partition/join, pinned buffer lifetime and XRT timeout, an actual Rust-owned AIE compiler/program feasibility trial, numerical source checkpoint parity, under-identical-clock ABBA performance, energy and long unattended safety qualification on isolated hardware. No automatic launch before validation.

### 17.5 Why this strengthens long-term flexibility

The accelerator's hardware details live in `DeviceCapabilities/KernelProvider`, the legal row split in RIR, cross-API resource and scheduling in CIR, and user/task workflow in EngineCore/TaskProgram. A new hardware NPU or model family may benefit from a previously verified projection-slice transformation **without adding another hard-coded model runtime**, while even a total NPU driver/compiler absence leaves the GPU-only model path intact. This is the concrete architectural meaning of "seamless", not a claim that the NPU program has already been implemented.

````

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/prior-art/PRIOR_ART_HISTORICAL_ROADMAP_V2.md

Original UTF-8 bytes: 33999; SHA256 `2e6559d784d83c2f7a1349b5138f7e51b27d66e712b224f11a52be8fcd2da52a`.
Resolve embedded relative paths against the original path, not this volume.

````text
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

````

## Source: 7e61cb63efd0c1429aa96044f2f7d28fda1e59aa:.agents/prior-art/PRIOR_ART_HISTORICAL_STAGE0_V2.md

Original UTF-8 bytes: 20730; SHA256 `5779560332a94b6993e2657bb9225ad10edfee9803de54181db5e86034f91e7c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Prior-art/historical archive — INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md

**Historical v2 planning snapshot, archived 2026-10-12. NON-NORMATIVE RESEARCH.** External named projects and benchmark figures are preserved as source research, while actual current design and phase gates use project-neutral contracts at `../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`, `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`. User-provided instructions and decisions remain separately in DECISION_REGISTER. Never use historical source-project APIs as the new runtime's type or module names.

---

# zerogravity — Integrated feasibility, critical path, stage gates and delivery contract

**2026-10-10 v2 planning.** All activities NOT_STARTED. This is the authoritative **gate crosswalk** between the research program's S1–S12/G0–G4 and the project's ZG-P00…P23 phases. It is a staged engineering plan, not a date promise, and must not be interpreted as user approval of untested performance constants.

## 1. The core program contradiction we avoid

**Research risk:** building 20+ frozen crates and choosing CubeCL/C++ escape before proving Semantic IR/math and Rust gfx1151 codegen. **Project risk:** building a large symbolic/effect compiler before one GGUF model runs or a kernel reaches competitive GPU ISA. **Solution:** execute small orthogonal feasibility probes in parallel, then a thin end-to-end vertical slice, freeze interfaces only *after* both semantic and physical proof. Maintain a safe general path while specialized kernels are optional.

Canonical root: /home/pvl/spaces/abird/src/zerogravity. Source evidence: /home/pvl/spaces/abird/research/inference (read-only). Implementation still absent.

## 2. Integrated Stage 0 experiments, with mandatory proof

| ID | Path from original plans | Goal and smallest deliverable | Hard go/no-go | Hardware |
|---|---|---|---|---|
| **S0-00** | P00, research task1 | Lean Cargo workspace, rust toolchain+Nix shell+CI0, decision and source-pin ledger, clean CPU smoke | reproducible source/build, no unreviewed license or vendor runtime | CPU |
| **S0-01** | P01, research S1 | Rust GPU kernel compiler bakeoff: CubeCL gfx1151 low-bit GEMV/WMMA, Rust native AMD target alternatives; inspect device ISA/perf, graph capability | correct `gfx1151` results and competitive path or honest pivot | physical gfx1151 |
| **S0-02** | P01/P07, NEW | CUDA provider bakeoff: CubeCL versus new Rust tile compiler and SIMT-PTX compiler (alpha), same simple shapes and reference | Linux/CUDA toolchain feasibility, numeric and source language policy | NVIDIA compatible GPU |
| **S0-03** | P02/P03, research S2 | Safe GGUF+dialect/safetensors slice, tokenizer byte parity, CPU quant/attention + scalar oracle | complete tiny/real dense checkpoint, independent logits/tokenizer | CPU |
| **S0-04** | P04, NEW | Minimal SIR typed tensor + state effect operator, GDN and KV effects, snapshot/commit rule; verifier rejects invalid reorder | independent CPU state trace and invalid-effect negative tests | CPU |
| **S0-05** | P05, NEW | Bounded RIR regions and one kernel candidate registry, forced missing specialization; no per-token interpreted graph | real reference vs lowerer output and safe region fallback | CPU + GPU eventually |
| **S0-06** | P05/P07, NEW | Minimal CIR resource/timepoint/IO simulation: lease, dispatch, copy, acquire/release, commit, one fault | event DAG, alias, stale generation, cancellation/partial read fail closed | CPU fake driver |
| **S0-07** | P09/P10, research S3/S5 | EngineMsg/Event, StepPlan/WorkerDelta, paged KV pool/frontier+pure resource planner simulator | invariants, deterministic ledger/refusals, no global hot mutex | CPU simulator |
| **S0-08** | P02/P15, research S8/S9 | Full-logit ABBA when exact, tolerance/quality when permitted; speculative accept/replay across KV/GDN/RNG | no invisible mutations, exact target sampled distribution | CPU + GPU modes |
| **S0-09** | P03/P09, research S4/S6 | token template, grammar FSM, tool stream and HTTP mock clients, cancellation | official token vectors, API conformance smoke, no tool injection | CPU |
| **S0-10** | P01/P05, research S1b/S10 | ExecutableBundle+KernelPlan artifact, eager/captured graph lifecycle, old driver/shape refusal | no stale pointers/alloc under capture, deterministic bundle key | GPU |
| **S0-11** | P12/P20, research S7 | Offline expert-cache trace simulator: per-layer slots, LFU/recency/hotlist, route overlap, concurrent pressure | reproduce evidence or label mismatch, no policy promotion without gain | CPU trace |
| **S0-12** | P08/P09, NEW | **End-to-end vertical slice**: model artifact → SIR → RIR → CIR + kernel → EngineCore → generated token via CLI/server | one dense model correct on CPU and at least one GPU; disable extreme region and remain functional | CPU + GPU |
| **S0-13** | P10/P11, NEW | 3+ different hybrid state domains shared prefix, compressor alignment and fork/rollback | same causal frontier; exact compatible state reuse, safe partial hit/recompute | CPU then GPU |
| **S0-14** | P20, research S11/S12 | Optional portable Vulkan and zero-weight prefix-derived draft index, A/B against baseline | correct fallback, opt-in experimental policy only after measured result | Vulkan/CPU |

**Mandatory before architectural interface freeze:** S0-00/01/03/04/05/06/07/08/12. S0-01 can fail without killing portable product; it kills the extreme gfx1151 guarantee until a new Rust toolchain or kernel path proves viable. S0-02/09/10/11/13/14 can be independently scheduled when prerequisites ready; select a subset to avoid pretending the full ~93 engineer-days research program fits 4–6 weeks with two engineers.

**Important:** S0-12 requires enough basic artifact/model/runtime support to run one model; do not delay it until a giant compiler with all future state ops exists. S0-04/05/06 start tiny and expand as real ops demand.

## 3. Gates and hard evidence

### G0-A: correctness substrate
- GGUF safe parsing, tokenizer exact vectors, Q4/Q8 CPU quant oracle and at least one real small dense checkpoint; official/independent per-layer/logit tests; semantic state effects verified on synthetic GDN.
- Source/build provenance and license review; user Rust-only source policy not violated.
- Deterministic boundchecked artifact/schema/version; unknown op/quant returns typed refusal.
- No silent precision/layout changes.

### G0-B: codegen and async execution
- Physical `gfx1151` proof with generated ISA, register/LDS occupancy and comparison to selected licensed reference kernels, same dtype/shape.
- Rust CUDA compiler options evaluated when target available; no global CubeCL lock-in.
- CIR fake-resource events, IO-short-read/quarantine/generation tests pass; capture fallback safe.
- Runtime hot path references compiled bundle handles, not an IR interpreter.

### G0-C: first functioning vertical slice
- Real dense model CPU+one GPU verified; forced specialized operator disabled but runnable fallback.
- Same session state/tokenizer/API handling through EngineCore and direct worker outputs, no duplicate independent serving driver.
- Versioned minimal plan introspection and physical memory-fit report.
- Source, phase and tests documented from actual run.

**Architecture freeze only after G0-A/B/C** and only for proven external interfaces; new operator capability registry remains evolvable. If physical GPU not available, state that extreme gates are blocked; proceed CPU-only feasibility without claiming GPU qualification.

### G1: portable product
- one useful binary/SDK path, GGUF/safetensors supported matrix, correct common dense checkpoint(s), tokenizer/template/tools, CPU + demonstrated available GPU, typed refusals, beginner usability;
- no guessed license or model coverage;
- compare same hardware/checkpoint with llama.cpp at TTFT, decode, prefill with quality. Research proposed thresholds ≥70% decode/≥60% prefill; **not user-approved** and should become explicit experimental gates with selected reference.

### G2: serving parity
- paged KV/prefix, native C1/2/4/8/16/32 batching, preemption/fairness, mixed prefill/decode, optional spec and structured output;
- p95 TTFT/ITL, aggregate committed tokens, memory pressure and 24h soak;
- compare pinned vLLM/SGLang on same model/quant/hardware. Historical proposal ≥80% vLLM decode BS1/≥60% BS32, requires confirmation as release floor.

### G3: extreme models
- full model reference and modern checkpoint quant/quality, `gfx1151` strong MMVQ, QSA/GDN and GLM KDA/DSA/mHC, resident+SSD MoE, full hybrid snapshots, MTP;
- Gufo/Maya/DwarfStar matched benchmarks or separate unmatched published context, no fictitious win;
- honest path/capability records for 92/96/128GB and driver variants; no OS OOM.

### G4: breadth/distribution/future
- modern Gemma4/Qwen3.5+/DeepSeek, LoRA, embeddings/rerank, vision and audio as separately qualified paths;
- topology-aware TP/EP/PP and fail-closed peer agreement, optional CUDA/Metal/Vulkan/NPU;
- multi-model hosting and upgrade/ABI/migration, clear security and quality.

## 4. Program phase mapping, unchanged IDs P00–P23

| Existing project phase | Consolidated required adjustment |
|---|---|
| P00 | lean workspace and pinned source/decision evidence; **no mandatory 25 crates**, no Git init unless approved |
| P01 | CubeCL + AMD Rust codegen and NVIDIA Rust tile/SIMT-PTX compiler feasibility; never assume toolchain works from docs |
| P02 | CPU model operators and complete mutable reference state; add grammar/sampler golden as separate test |
| P03 | GGUF/safetensors, tokenizer pretypes, source hashes, imatrix/quant and optional derived pack plan |
| P04 | minimal SIR SSA+effects+state group codec; explicit novel-op rejection, same-plan replay |
| P05 | RIR pure/fusion/legalization + **KIR registry + CIR async resource/events**, bounded specialization |
| P06 | CPU executor direct compiled plan, tested SIMD where possible, correct fallback |
| P07 | GPU device capabilities, validated leases/events, graph safety and per-target Rust kernel provider |
| P08 | first real dense vertical slice and forced per-region fallback; first true kernel/perf evidence |
| P09 | EngineCore/typed worker protocol, CLI/HTTP, tokenizer/grammar/tool, event loops and mode receipts |
| P10 | paged+contiguous KV under one arbiter, optional radix, full-state prefix snapshots and 3+ group valid intersection |
| P11 | typed GDN/KDA/QSA/MLA/DSA/CSA and precise state codecs, long-context and encoder span alignment |
| P12 | MoE route+GPU grouping, expert slabs/lease/miss tiers, CPU lane only after measured benefit |
| P13 | modern Qwen3.8 specialized gfx1151 with ngram SSD tables, MTP and Gufo comparisons |
| P14 | GLM-5.3-Flash state and expert residency, Maya/DwarfStar matched quant and real RAM safety |
| P15 | target-authoritative exact stochastic verifier, CommitPlan/Frontier mechanics, full state rollback |
| P16 | 1–32+ concurrency, route-aware batching, joint speculation/residency cost and measured p95 |
| P17 | model breadth Gemma4/Qwen3.5+/DeepSeek **plus explicit LoRA and embedding/rerank runner contracts** |
| P18 | multimodal and modality-specific encoder budgets, video/audio as qualified extensions |
| P19 | tensor/expert/pipeline/P-D distributed strategies with model/state identity and failure atomicity |
| P20 | bounded offline/online tuning, KernelPlan provenance, eager/graph-mode tactics, rejected experiments |
| P21 | license policy, Rust vendor FFI decision, auth/privacy/plugin trust, cargo-deny, Nix/CI/release |
| P22 | replayable peer-matched bench with quality, hardware/quant, Pareto/performance claim ledger |
| P23 | versioned SemanticOp and KernelProvider registry, architecture evolution exercise, deprecation/migrations |

P00–P05 may overlap only where interface ownership is settled, but P08 vertical slice must not be delayed by implementing hypothetical P17 operators first. P13/P14 use P11/P12/P10 correctness; high-concurrency P16 may evolve independently of two huge model support implementations and must not postpone useful generic serving.

## 5. Actual user device versus dated research machine

Historical `research/inference` plan assumed RTX3060 6GB/Cezanne 27GB + purchase of gfx1151. **Do not treat this as verified current hardware.** At P00 physical `DeviceProbe` must query the available machines and drivers. Users may have a Strix Halo 92/96GB device and a separate NVIDIA notebook; if present, use them and do not recommend purchase. Device class selection by capability, not marketing/model names.

If actual Strix not accessible through connector and hardware cannot be tested, mark physical G0-B and G3 BLOCKED, not VERIFIED.

## 6. Source/kernel and licensing policy

- User chose Rust-authored engine. CubeCL not automatically the only compiler.
- New public Sept 2026 Rust tile and SIMT-PTX compilers need distinct capability/ISA tests. Avoid committing to rustc nightly as required for whole project because one experimental kernel provider needs it.
- Vendored C++/HIP kernel source allowed only if user explicitly resolves language policy; documented external source licenses still required for legal study/porting.
- GGUF input and derived acceleration pack: library packaging names open; research's `APK` confusing with Android .apk, avoid codifying in public ABI.
- Follow threat model/crypto-correct model/prefix ID before writing a disk-KV schema; research 48-byte header is invalid by arithmetic, build fresh checked versioned schema.

## 7. Test and evidence matrix

| Area | Small proof | Long proof | Release blocker |
|---|---|---|---|
| Tensor and state | scalar ref and shape fuzz | full official model/long context | missing op, wrong frontiers |
| Compiler | legal/illegal effect rewrite | GPU direct plan and fallback | semantic change, hot-path IR overhead |
| CIR and cache | lease/event simulator + injected IO | 24h stress/fragmentation/pressure | stale/overwritten device slot |
| Tokenizer/grammar | golden byte/token vectors | full tools/structured output | hidden control tokens/injection |
| Quant/format | block exact decode and manifest | NLL/KL quality bands | implicit downgrade |
| Batching | C1/2/4 goodput | C8/16/32 p95/ITL + fairness | wrong session state/starvation |
| Spec | accepted-length 0..N exact state | distribution preserving sampled multiworkload | rejected draft leaks into state |
| Hardware | probe + one real kernel | multiple driver/quant/context matrix | unsafe UMA overcommit |
| Multi-modal/LoRA | encoder/adapters reference | image span + LoRA concurrent replay | cache cross-use |
| Distributed | two-rank handshake/failure | multi-rank/transport fault soak | partial commit |
| Performance | paired ABBA matched run | full request replay/power+quality CI | unverified marketing claim |

## 8. Engineering estimates and kill criteria

Do not use a misleading linear roadmap. The original research's `~93 engineer-days Stage 0` is **18.6 engineer-weeks of labor**; with two full-time engineers it is ~9.3 elapsed weeks **before overhead/dependencies**, not a blanket 4–6 weeks claim. Reduce Stage 0 to the hard-gate set and schedule optional experiments separately rather than inventing faster parallelism. Project phases are milestone identifiers, not chronology guarantees.

**Kill/pivot conditions:**
1. Compiler/effect model cannot represent target model state without hidden writes → redesign SIR immediately.
2. Rust GPU toolchain cannot generate competitive target instruction code and no Rust-authored escape provider feasible → defer extreme claim; general engine continues.
3. No independent known checkpoint token/logit parity → stop specialization before baseline correct.
4. CIR introduces measurable unavoidable per-token overhead → compile plan to direct command/graph handles and simplify.
5. Memory allocator cannot safely refuse unfit model on UMA → block public release.
6. Repeated policy complexity gives no end-to-end p95 or committed t/s win → keep simpler policy.
7. Quant/model quality weak or stochastic verifier alters target distribution without opt-in → disable changed plan.
8. Kernel plan/version/cache invalidation is unsafe across driver/model schema upgrades → invalidate/rebuild, not continue.
9. Security parser/network tool vulnerability unresolved → block service release.
10. Superiority cannot be matched fairly with reference quant/model → report incomparable, no claim.

## 9. Exit bookkeeping

For every stage record `phase_id, model_sha/quant, operator contract version, plan_hash, source commit, command/test, actual hardware/ROCm/CUDA/driver/power, CPU/GPU numerical parity, quality results, memory headroom, latency distribution, fallback outcome, known skip/failure, next decision`. Mark statuses `NOT_STARTED/ACTIVE/VERIFIED/BLOCKED/DEFERRED` in current status document. Research and design writing alone never marks a phase VERIFIED.

## 2026-10-12 additional optional feasibility track: GPU + XDNA2 NPU and TaskProgram

**User decision U-016:** include the Gufo fork's GPU/iGPU+XDNA2 cooperative prefill and upstream Gufo ASR/TTS/image/video pipelines in our clean extensibility design. Keep S0-00…S0-14 unchanged; add **S0-NPU-A…S0-NPU-F as an independent gated experimental track**. This track does **not** block CPU/GPU model vertical slice or G1 portable release. It requires physical NPU hardware for qualification and *a feasible Rust-owned AIE source toolchain*. Refer to [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) for exact invariant and source links.

| New gate | Dependency and artifact | Objective proof / closure |
|---|---|---|
| **S0-NPU-A** | P04/P05 SIR+RIR+CIR; mock GPU/NPU backend | Model-independent output-row `ProjectionRowSlice`, complete row coverage, first-use dependency DAG, effect legality and canonical GPU-only fallback; fake device test no XRT |
| **S0-NPU-B** | S0-NPU-A and P01 device API | **Rust-owned** XDNA2/AIE kernel/compiler feasibility and bounded XRT runtime bindings, program/insts hash, supported shape and numeric encoding; if only non-Rust AIE source exists, mark blocked-by-Rust-policy |
| **S0-NPU-C** | physical qualified isolated `gfx1151+XDNA2` host | Firmware/AGESA/ROCm/XRT/FCLK/power/suspend, shared memory and safety preflight; no machine-check hangs during controlled concurrent testing; ≥24h soak including restart/resume; user-controlled profile, no auto root tweaks |
| **S0-NPU-D** | P11/P13 real Qwen model + source checkpoint | Reference versus GPU-only versus XDNA2 sliced GEMM for GDN z, QSA query/gate, shared gate/up separately and combined, full logits/state/quality, strict numerical-mode refusal, MTP/grammar |
| **S0-NPU-E** | S0-NPU-C/D and P20 performance harness | Reproduce/qualify source report UD-IQ4_XS 16K `1698→1821` author results under **identical fabric clock** in A/B, also 8K/32K/40K cold, C1/C4, p95, energy/idle watts, fixed+auto clock control, confidence |
| **S0-NPU-F** | S0-NPU-A/E; optional P17/P18 stage model | Add a second compatible projection family / encoder stage without edits to model-core/HTTP; unsupported NPU defaults GPU-only; TaskProgram ASR/TTS/media stage outputs have independent task reference/quality |

**Exit conditions:** S0-NPU-A may pass on CPU fake executor; S0-NPU-B remains BLOCKED if no Rust-authored AIE path; S0-NPU-C/E cannot be marked verified on simulated hardware; S0-NPU-D needs real model weights and numerical results. A frozen host is an unsafe outcome that blocks auto-enable until actual platform problem resolved; a software timeout does not guarantee recovery. Never perform destructive power/BIOS changes on the user's everyday machine to force a demonstration.

**Release/capability state:** NPU absent/unqualified/experimental/qualified. Default GPU-only; no implicit sudo/fabric hold/NPU keepalive. A supported TaskProgram (ASR, TTS, image, video) must separately pass model/codec/iteration/output/quality and resource state acceptance, not simply instantiate a language model decoder. Each capability has model+hardware+precision+source-code qualification. Partial value is fine: safe GPU-only fallback remains a genuine deliverable.

**Phase placement:** P01 NPU compiler feasibility, P05 RIR row-split and CIR first-use events, P07 DeviceProvider and pinned XRT BO, P10 physical budget, P11 hybrid math/reference, P13 Qwen GPU+NPU, P16 cost-aware scheduling, P17–18 TaskProgram media, P20 same-clock A/B autotuning, P21 SoC hard-hang security/power gate, P22 matched quality/performance. Keep P00–P23 IDs stable.

````


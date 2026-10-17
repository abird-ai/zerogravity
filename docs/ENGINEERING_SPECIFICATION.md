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


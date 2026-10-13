# zerogravity — v4.1 typed composition and region execution contracts

**15 October 2026 — ZERO-ADDITIONAL-ENGINE STORAGE EXTENSION:** Model/SourceTensor semantics remain invariant as BackingTier changes among verified GPU/UMA/host/SSD; existing CIR Read/Transfer/Ready and physical ResourceArbiter/Lease own materialization. Source tensor identity, codec, shape and numeric plan are independent of device pointer, page cache state or file offset. Use one compiled direct-ready fast path and a bounded miss-capable asynchronous path, **not** a VirtualTensor trait call per multiply or a new SSD inference runtime. Encoded-but-unexecuted GPU graph reads pin slots until completion (DS4 #384). See [full source-neutral contract](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md).


**15 Oct extension-minimality decision proposal:** [quant](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md), [source model/DraftSource](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md), and [gfx kernel](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) extend existing source adapter + TensorEncoding + verified auxiliary ModelAssembly + TaskDriver + region candidate. Do **not** add EXL3, MiMo, AMD or DFlash inference engines/allocators. Optional residual-activation steering is a **behavior-changing, source-versioned semantic model variant**, independently quality/safety reviewed and OFF by default; it is never a numerical-preserving physical accelerator, automatically loaded from untrusted weights or silently used as a default.


**2026-10-12 · authoritative illustrative Rust interface design · NOT IMPLEMENTED.** These are *minimal capability sketches*, not code to paste verbatim or a public ABI freeze. The reference architecture is `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`; complete state/worker and task/compiler protocols in `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md` and `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`. Earlier v3 contracts are preserved in `DESIGN_EVOLUTION_V3_SNAPSHOT.md`.

## 1. Core type ownership — no oversized trait zoo

Required stable concepts:
- `SemanticProgram` and verified `SemanticOp` schemas;
- one trusted `TaskDriver` per **task kind**, with typed outputs/frontier;
- versioned `StateDomain` **prepare**, combined root **publish** owned only by TaskCoordinator;
- **`RegionExecutionProvider`** capability selection for one closed semantic region or whole graph/stage; native `KernelProvider` is one implementation form, not a separate model-serving engine;
- `DeviceBackend` owning streams/alloc/events plus `ResourceArbiter` admission;
- `ExecutableBundle` as compiled device command/resource plan with shape guard;
- `WorkEnvelope/WorkerReply` and `ModeReceipt`.

Each can initially be a Rust module in one library. Avoid public dynamic trait-object plugins, blanket cloning of model schedulers and separate serialized RIR/KIR dialects until tests justify them.

## 2. Model math and type/shape guards

```rust
struct SemanticOpSchema {
    id: SemanticOpId,
    version: OpVersion,
    inputs: Vec<ValueType>,
    outputs: Vec<ValueType>,
    shape_rules: ShapeConstraintSet,
    effects: EffectContract,
    numerical: NumericContract,
}
trait ModelDefinition {
    fn validate(&self, artifact: &ArtifactManifest) -> Result<ModelRecipe>;
    fn semantic_program(&self, recipe: &ModelRecipe)
        -> Result<SemanticProgram>;
}
trait ReferenceOp {
    fn schema(&self) -> &SemanticOpSchema;
    fn evaluate(&self, input: &RefValues, state: &RefState)
        -> Result<(RefValues, ProvisionalRefState)>;
}
```

`ReferenceOp` returns **provisional** state; it cannot directly publish a TaskSession state root. Novel mathematical ops require separate formula/version and source vector tests, not a model-name branch inside the GPU worker.

```rust
enum DimExpr { Static(u32), Symbol(DimId) }
struct RuntimeShapeGuard {
    constraints: ShapeConstraintSet, // rank, min/max/divisibility, strides, ragged
    numeric: NumericContractId,
    physical_layout: PhysicalLayoutId,
    state_schema: StateSchemaId,
}
enum GuardDecision { Eligible, Fallback(FallbackReason), Reject(ShapeError) }
```

A `ShapeEnv` binds logical symbolic dimensions and actual ragged row offsets at stage preparation. A candidate with shape/quant not supported **must never launch**; select a prevalidated portable kernel or typed refusal. Dynamic expert IDs stay data inside a device-resident route table, not one compiled variant per ID.

## 3. Region execution contract: native kernels through compiled accelerator programs

```rust
struct RegionContract {
    semantic_hash: SemanticRegionId,
    inputs: Vec<ValueType>,
    outputs: Vec<ValueType>,
    effects: EffectContract,
    numeric: NumericContract,
    fallback: ReferenceRegionId,
}
enum ExecutableKind {
    NativeKernel,
    CapturedCommandGraph,
    AheadOfTimeDeviceGraph,
    StaticallyScheduledDataflow,
    ManagedDeviceSession,
}
struct RegionCandidate {
    region: SemanticRegionId,
    scope: RegionScope,  // single op, fused region, task stage, full model, fabric
    kind: ExecutableKind,
    shape: RuntimeShapeGuard,
    compute_precision: ComputePrecision,
    source_quant: QuantIdentity,
    target: DeviceCapabilityQuery,
    resources: ResourceFootprint,
    state_bridge: StateBridgeContract,
    program_lifecycle: ProgramLifecycle,
    accuracy_and_quality_evidence: EvidenceId,
}
trait RegionExecutionProvider {
    fn candidates(&self, closed: &RegionContract,
        topology: &TopologyCapabilities) -> Vec<RegionCandidate>;
    fn prepare(&self, chosen: &RegionCandidate,
        budget: &CompileBudget) -> Result<PreparedExecutable>;
}
```

This is an illustrative **model-load/compile-time** interface, not called for every token. A provider may prepare a CPU tile, direct GPU kernel, captured GPU command graph, external accelerator **ahead-of-time device graph**, or fully statically scheduled dataflow program across one or many chips. Its prepared executable is submitted under the **same** StepPlan/ResourceArbiter/TaskCoordinator and never owns the serving session. An opaque text-in/text-out vendor service without verified model state semantics belongs at an external TaskService boundary, not as a native region backend. Model top-k/indexer and requested precision must not change silently.

Fallback: direct region replacement when state/numeric/layout contracts match, explicit conversion/checkpoint replay when they do not; a stateless subgraph can use an AOT provider without state bridge, but any stateful/partial graph region requires tested provisional versions/snapshot/replay. Shape-unsupported/compiled-program-unavailable candidates return a typed fallback. If no semantic reference for the op exists, **reject UnknownSemanticOp**, not a guessed vendor graph. See [backend families](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## 4. Prepared whole-state transaction (replaces v3 mutating `commit`)

```rust
trait StateDomain {
    fn schema(&self) -> StateSchemaId;
    fn can_resume(&self, snapshot: &SnapshotHandle,
                  request: &ResumeQuery) -> ResumeCompatibility;
    fn fork(&self, committed: &StateRoot) -> Result<ProvisionalDomain>;
    fn prepare(&self, provisional: &ProvisionalDomain,
               frontier: TaskFrontier) -> Result<PreparedDomainVersion>;
    fn abort(&self, provisional: ProvisionalDomain);
    fn recompute(&self, previous: &StateRoot,
                 continuation: &CanonicalHistory) -> Result<ProvisionalDomain>;
}
struct StateRoot {
    epoch: SessionEpoch,
    frontier: TaskFrontier,
    versions: Vec<DomainVersionHandle>,
    compatibility: StateCompatKey,
}
struct PreparedRoot {
    old_version: StateRootId,
    new_root: StateRoot,
    required_events: Vec<TimepointId>,
    provisional_output: Option<OutputRef>,
}
```

**Ownership law:** **no `StateDomain::commit()` in the public interface**, because per-domain mutation undermines whole-session atomic logical publication. The TaskCoordinator checks all prepared domain versions, epoch and readiness events and performs **one `PublishRoot`**. Underlying device buffer writes may be asynchronous but cannot corrupt old root; COW/undo/recompute must preserve it. An executable/device kernel only marks its provisional result Ready/Failed and can never directly commit externally visible state.

All token/attention/recurrent/indexer/grammar/RNG/audio latent/media/adapter state domains participate according to the exact task. The same API can support future parameter optimizer state, but normal inference weights are immutable versioned views.

## 5. TaskDriver is stage control, not an embedded general workflow engine

```rust
enum StageIntent {
    Run { bundle: CompiledStageId, inputs: StageInputs },
    ParallelKnown { bundles: Vec<CompiledStageId>, join: JoinPolicy },
    WaitForInput(InputWaitId),
    Done(FinishReason),
}
trait TaskDriver {
    fn validate(&self, request: &TaskRequest) -> Result<TaskSpec>;
    fn start(&self, request: TaskRequest, root: &StateRoot) -> Result<TaskCursor>;
    fn next(&self, state: &TaskCursor, completed: Option<&StageCompletion>)
        -> Result<StageIntent>;
    fn prepare_commit(&self, state: &TaskCursor,
                      provisional: &ProvisionalOutput)
        -> Result<TaskOutputCandidate>;
}
```

**Task boundary:** `TaskDriver` owns trusted coarse control over executable regions and optional device-native loops. It does **not** include `ExternalAction` or arbitrary user callbacks with filesystem/network/process privileges. Retrieval, agent workflows and external tools run above TaskRequest behind independent permissioned service adapters.

Different `TaskDriver` instances: text causal generation, encoder/pool/score, ASR, TTS/acoustic/vocoder, image denoising, video temporal frame/mux. Each defines typed `TaskFrontier` and output `StreamContract`, not one `token` scalar for every task. A TaskDriver remains compilable/statically linkable; no public plugin ABI required initially.

## 6. Runtime StepPlan and DeviceBackend

```rust
struct WorkEnvelope {
    session: SessionId,
    attempt_epoch: SessionEpoch,
    expected_root: StateRootId,
    bundle: ExecutableBundleId,
    shape_band: ShapeBandId,
    slots: RowBindingTableId,
    resources: ReservationTicketId,
    deps: Vec<TimepointId>,
}
enum WorkerReply {
    Complete { ticket: TicketId, ready: Vec<TimepointId>,
               provisional: PreparedCandidateId },
    Failed { ticket: TicketId, reason: WorkerError, possibly_live_dma: bool },
}
trait DeviceBackend {
    fn capabilities(&self) -> DeviceCapabilities;
    fn submit(&self, work: &WorkEnvelope) -> Result<TicketId>;
    fn poll_ready(&self, ticket: TicketId) -> Result<DeviceCompletion>;
    fn retire(&self, leases: LeaseBatchId) -> Result<RetirementEvent>;
}
```

Execution is direct already-compiled command/graph handles; `WorkEnvelope` has no per-token model graph or giant tensor clone. Normal hot batches use compact preallocated indices/slots rather than heap Vec where profiling requires. DeviceBackend's `poll_ready` may be async/event-driven, not busy spin. Host TaskCoordinator can overlap preparing independent steps, but must not violate dependencies on uncommitted tokens or sampling/grammar/RNG state.

`StepProgram` (compiled kernel/control island) may perform multiple successive device-native decode/verifier/denoiser microsteps and return an event plus a *prepared* state candidate; that is compatible with one host publication event and avoids unnecessary device↔host roundtrips. A task that requires host logic on every iteration uses honest slower fallback.

## 7. Resource admission, local allocators and CIR

```rust
trait PhysicalResourceArbiter {
    fn reserve(&self, req: &ResourceEstimate,
               tenant: TenantId) -> Result<ReservationTicket>;
    fn adjust(&self, ticket: ReservationTicketId,
              request: &ResourceDelta) -> Result<ReservationTicket>;
    fn release(&self, ticket: ReservationTicket);
}
struct BufferLease {
    id: AllocationId,
    generation: u64,
    memory_tier: MemoryTier,
    access: BufferAccess,
    last_use: TimepointId,
}
enum Command {
    Acquire(ResourceClaim),
    Copy(TransferSpec),
    ImportMapped(ExternalBufferSpec),
    Pack(CodecId),
    Dispatch(KernelHandle, ParamSlotId),
    Wait(TimepointId),
    Join(Vec<TimepointId>),
    PublishResidency(TableVersionId),
    Release(BufferLeaseId),
    AbortProvisional(PreparedCandidateId),
}
```

**There is deliberately no `Command::Commit` semantic publication**. CIR manages readiness/event and buffer lifetime. The coordinator performs state root publication. Each device-local allocator draws from reserved memory and returns generation-tagged leases, with async event-based retire; `PhysicalResourceArbiter` **does not take a global lock per GEMM tile or token**.

OS memory and device/GTT pages may change independently; global admission is conservative and responds to pressure/failure. Never promise 100% prevention of unrelated OS-wide OOM; instead refuse unfit work and prevent unsafe engine-generated unbounded allocations.

## 8. Task output and transport delivery

```rust
struct CommittedOutput {
    session: SessionId,
    commit_epoch: SessionEpoch,
    sequence: OutputSequence,
    frontier: TaskFrontier,
    media_or_text: OutputRef,
}
enum DeliveryPolicy {
    LiveOrderedAtMostOnceAttempt,
    ResumableWithReplayCursor { max_retention: RetentionLimit },
}
```

`PublishRoot` precedes `CommittedOutput`. Live network stream can lose the acknowledgement of a chunk the client received; **do not promise exactly-once end-to-end**. Optional retained cursor/idempotency allows a client to deduplicate replayed committed events. Output buffers/PCM/video frames have backpressure credits and retention limits. Crash durability across hosts requires a separate protocol/log and is not built into simple GPU inference.

## 9. Fingerprints and invalidation by responsibility

| Key | Includes | Excludes |
|---|---|---|
| **ArtifactKey** | immutable model tensors/config/tokenizer/sidecars/quant and adapter version | current request row IDs |
| **SemanticKey** | op schema, effects, mathematical position/numeric policy and model source identity | live memory pointers |
| **TaskKey** | task stage/frontier/grammar/sampler/media codec and output validity | GPU ISA |
| **StateCompatKey** | state schema/layout/positions, source/adapter/template/modality/tenant, compatible resume frontier | unrelated driver compilation time |
| **BundleCacheKey** | Region/KIR implementation, target ISA/compiler/driver/graph mode, shape guard/layout/numeric quality | active RequestId and fresh expert slot contents |
| **RuntimeBinding** | current session epoch/root, row slots, physical leases and active IO/event table | immutable compiler's cache key |

No persisted state snapshot may include raw GPU pointer or EventId. Compiled bundle caches are **expendable**, rebuilt/invalidate after driver or codegen changes. Serializable source/model/state schemas are separately versioned/validated, not one universal serialized C++/Rust object graph.

## 10. Future gradient and parameter update seam

A future `TrainStep` TaskDriver may invoke known forward and backward `SemanticProgram` regions (requiring separate adjoint/gradient operator schemas), with activation checkpoint/recompute, optimizer and RNG state and a **mutable ParameterVersion**. Inference tasks pin immutable checkpoint+adapter version until explicit cutover; no hidden model-weight mutation or cache reuse across incompatible versions. This is a **future optional task specialization**; core reference inference doesn't need a general autodiff tape, distributed training WAL, optimizer algorithm catalog or external training framework.

## 11. Lifecycle and test-before-freeze

Start with the most modest possible Rust representation/trait set. Before making public ABI stable or creating many crates:
1. One real small checkpoint and task output CPU and GPU, independent oracle.
2. One full `Begin→Prepare→PublishRoot` across multiple fake state domains and one failing device.
3. One legal fused region replacement and one forced fallback without copying serving code.
4. One shape guard with ragged input and one compiled StepProgram with no CPU sampled-token dependency when provider supports it.
5. One fake non-token TaskDriver using the same worker/budget/compiled plan.
6. One bad external DMA/IO event causing quarantine and no user-visible partial state.
7. One optional gradient/weight-version synthetic seam, without training code in default runtime.
8. One new hypothetical device provider without edits to ModelDefinition.

Only proven implementations become `FUNCTIONAL` or `PERFORMANCE_QUALIFIED`, as indicated by exact model/quant/hardware/quality test records. Design sketches are hypotheses until source exists.

**Core invariant:** if adding an optimized kernel still requires a model-specific scheduler, API, state root or memory manager, the architecture is too tightly coupled and must be simplified.

## v4.1 addendum — state/shape/compile qualification for graph-only backends

`RegionExecutionProvider` **replaces** the single-kernel assumption. Its `RegionCandidate` must carry `ExecutableKind`, `RegionScope`, bounded `RuntimeShapeGuard`, `StateBridgeContract` (provisional/snapshot/restore/recompute or stateless), `ProgramLifecycle` (compiler/firmware ABI, load/warmup, program resident memory and code cache), `TopologyCapabilities` and source/numeric evidence. The same TaskCoordinator/DeviceBackend submit/ready/state-root protocol handles all backend program kinds. A candidate lacking state rollback/fork cannot silently implement general stateful LM decoding or speculative inference.

**Strong rule:** an external model API returning text with no accessible mathematical region state is *not* interchangeable with a native compiled graph provider. Treat it as an external TaskServiceAdapter with explicit privacy/cost/user consent. Device-side static micro-scheduling belongs to the provider/compiler; don't manually interpret each TPU/WSE/LPU instruction in our host CIR.

**Extension drill:** add a fake whole-model graph backend with static shape bands and state-private candidate handles, and a fake stateless streaming-fabric backend without touching ModelDefinition/TaskDriver/StateRoot/ResourceArbiter or HTTP code. A failed shape/backend compiler must transparently fall back to a qualified smaller region/general provider. **No new serialized IR or extra server required.** See [accelerator execution families](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## 12. v4.2 — orthogonal distribution contracts and region execution

The runtime adds **no extra model server** for remote execution. One PlacementPlan within ExecutableBundle binds ComputePlacement, ParameterPlacement, StatePlacement, CachePlacement and TransferPlan against an authenticated TopologyEpoch and PlanEpoch. These are compile-time/placement decisions, not new durable semantic IRs or unbounded provider traits. Existing DeviceBackend and RegionExecutionProvider execute *the same* closed SIR region whether local or through an authorized RemoteRegionWorker.

**Required typed concepts** (illustrative, not frozen API): NodeCapabilityRecord, TransportLinkCapability (directional throughput/RTT/P95/direct-vs-relay/security), PlacementPlanId, StateShardRef (session/layer/domain/version/location/lease/epoch), RemoteRegionHandle (program/model/quant/shape/numeric/TaskFrontier), TransferDescriptor (artifact/session/source/destination/bytes/priority/hash/completion), PeerAuthorizationPolicy. A CacheLocationHint may be stale and always gets content/schema checked; a StateShardRef may **not** be stale/written by another epoch.

Region placement can choose (1) state colocated with its layer worker, (2) attention execute at remote KV owner and FFN elsewhere, (3) state snapshot hydrate and then local attention, (4) direct remote state read only with verified fabric, (5) attention token-range partial program whose mathematical reduction is explicitly registered. A K/V softmax partition **requires** correct per-shard normalized summaries and global reduction; recurrent GDN/KDA or unknown sparse selected operators cannot blindly use this algebra.

**Coordinator invariant:** remote node can Prepare versioned provisional state; one session TaskCoordinator PublishRoot checks the complete required local/remote Ready cohort. Migration happens at committed TaskFrontier with destination reserve/transfer/verify/fence, never mid-collective by gossip. Each peer has scoped identity and explicit permission to receive user content and weights. A tunnel does not grant RDMA; TransportProvider advertises actual capabilities. The full contract is in [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) and [peer transfer](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md).

## 13. v4.3 — exact route resolution and per-model admission contract

A single lightweight `ModelRouter` sits at **TaskRequest admission**, not inside SIR/RIR/KIR/CIR kernels or the inner TaskCoordinator token loop. Distinct typed identities: `RegisteredModelId(ArtifactKey+SemanticKey)`, `AliasId/AliasRevision`, `ModelProfileId`, `ResidentModelId/ReplicaGroupId`, `DeploymentId(PlacementPlan)`, and `TaskSessionStateRoot`. Alias resolution pins an immutable exact checkpoint/quant/adapter/tokenizer/TaskDriver and allowed numeric policy *for a request/session*; a later alias hot reload only changes future routes.

Illustrative Rust interfaces:
```rust
trait ModelCatalog {
    fn resolve(&self, alias: &AliasId, scope: &TenantScope)
        -> Result<ResolvedModelBinding>;
    fn enumerate(&self, scope: &TenantScope) -> Vec<ModelCapability>;
}
trait ModelRouter {
    fn route(&self, request: &ValidatedTaskRequest,
        binding: &ResolvedModelBinding,
        candidates: &ReplicaAndResourceSnapshot) -> Result<AdmittedRoute>;
}
trait ResidencyManager {
    fn prepare(&self, binding: &ResolvedModelBinding,
        placement: &PlacementCandidate,
        request_budget: &AdmissionBudget) -> Result<ResidentModelLease>;
    fn drain(&self, instance: ResidentModelId, policy: DrainPolicy)
        -> Result<RetireWhenQuiescent>;
}
```

These are **coarse control APIs** resolved before StepPlan/direct device execution, not permission to build a duplicate model serving engine or a separate physical GPU allocator. `ResidentModelLease` is tied to real ResourceArbiter admission and safe event retirement. One distributed model across several machines is a **single Deployment/Replica**; independent copies of the same exact source are **replicas** that may balance new/stateless requests; a TaskProgram with different target+draft/encoder models holds multiple separately bound model resources. Stateful session routing preserves full StateCompatKey and only moves after an explicit compatible committed state transfer.

Two separate routing decisions: **exact source identity (default)** and **equivalent replica selection**; an optional permitted `auto` target-choice policy selecting another source is not a load-balancing fallback and must reveal the model/quality change. Reject wrong models/quant/adapter for overloaded requests rather than silently substitute. Configuration is typed, versioned and explainable with mandatory immutable/security/physical limits; request fields cannot bypass them. Full details: [model routing](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) and [CLI/API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).

# Minimal execution spine v4 — state, asynchronous workers, resource ownership and hot-path performance

**Normative technical contract · 2026-10-12 · NOT IMPLEMENTED.** Read together with `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md` and the v4 master/reference architecture. This document resolves contradictions in the older `StateDomain::commit`, `Command::Commit`, one-owner/one-pool and worker protocol sketches. All proposed Rust signatures are illustrative **internal types**, not frozen public APIs or a promise of a full distributed database.

## 1. Exactly five ownership boundaries

**Immutable semantic owner:** trusted ModelDefinition + immutable source artifacts and typed semantic graph. It defines mathematics, state schemas and allowed layout conversions. It cannot allocate device cache on an unbounded basis, send service responses or choose architecture-specific code by model name.

**Compiled plan owner:** Compiler creates verified immutable `ExecutableBundle` from source semantics, task execution regime and `KernelCandidate`; it contains kernel handles, shape guards, resource command template, fallback edges and cache provenance. It does **not** own session frontiers or live devices.

**Logical session owner:** `TaskCoordinator` is the **single writer per TaskSession or per bounded session shard**, not a mandatory single global process-wide thread. It owns `TaskDriver` state, request queue/admission intent, current committed `StateRoot`, cancellation/attempt epoch, output sequence and policy. It alone publishes a new root and output.

**Physical resource authority:** `ResourceArbiter` decides globally admissible physical RAM/device memory/IO headroom for model sessions/tasks and outstanding draft/copy buffers. It issues **reservations at model/session/step boundaries**, rather than routing every GPU kernel temporary allocation through a central lock. Each DeviceWorker owns a **fast local subpool/suballocator** consuming a pre-reserved budget with generation-tagged leases; cache policies are independent clients.

**Device execution authority:** DeviceWorker/IOWorker owns context, code/graph handles, stream queues, buffer mapping and completion events. It executes direct physical work and **never mutates a user's committed state root or emits an unverified TaskResult**. Its reported completion cannot itself commit model semantics; it can only make a provisional candidate ready.

These boundaries are the minimal answer to the original question: *what changes when a new mathematical op, task, target device, memory tier or fused specialization is added?* There should be exactly one primary owner to extend for each.

## 2. State versioning — prepared, not independently committed

### 2.1 The core error to avoid

The v3 sketch allowed `StateDomain::commit(candidate)` returning a new handle while the surrounding narrative required an **atomic whole-session transaction** across KV, GDN/KDA, sparse index, n-gram, grammar, sampler/RNG and task state. Per-domain `commit` that mutates visible buffers immediately **cannot** provide such atomicity. Similarly, a `CIR Command::Commit` cannot publish logical session state while the host TaskCoordinator also claims that authority.

**v4 rule:** each domain **prepares** a candidate state; only the session owner **publishes one immutable versioned StateRoot**, after required device events/effects are ready and all domain versions agree on the same frontier.

### 2.2 Illustrative states, epochs and root ownership

```rust
struct SessionEpoch(u64);
struct StateVersionId(u128);
struct TaskFrontier {
    kind: FrontierKind,   // tokens, audio frames, denoise iteration, etc.
    value: u64,           // validated monotonic within this kind/epoch
}
struct StateRoot {
    version: StateVersionId,
    session_epoch: SessionEpoch,
    frontier: TaskFrontier,
    domains: Vec<DomainVersionHandle>,  // all required domains + generation
    provenance: SemanticAndNumericKey,
}
enum DomainWork {
    Prepared(PreparedDomainVersion),
    NeedsReplay(ReplayRequirement),
    Rejected(StateError),
}
struct PreparedCommit {
    epoch: SessionEpoch,
    from_root: StateVersionId,
    proposed_frontier: TaskFrontier,
    domain_versions: Vec<PreparedDomainVersion>,
    ready_events: Vec<EventId>,
    output: Option<ProvisionalOutput>,
}
trait StateDomain {
    fn snapshot(&self, root: &StateRoot) -> Result<SnapshotHandle>;
    fn fork(&self, root: &StateRoot) -> Result<ProvisionalDomain>;
    fn prepare(&self, provisional: &ProvisionalDomain,
               frontier: TaskFrontier) -> Result<DomainWork>;
    fn abort(&self, provisional: ProvisionalDomain);
    fn restore_or_recompute(&self, source: &CanonicalHistory) -> Result<ProvisionalDomain>;
}
```

**Logical operation:**
1. `Begin(epoch, read_root)` pins immutable snapshot/reference and reserves provisional buffers/state from global budget via local subpools.
2. Compiled kernel/TaskDriver stage executes against the snapshot/private COW tail, generating `ProvisionalDomain` and provisional output. GPU native speculative steps may generate many candidate frontier versions without publishing them.
3. `Prepare`: all required domain state versions have the same `frontier`, schema and valid numeric/position rules; CIR events, copy/IO and device writes are either **Ready** or subject to explicit compatible replay. If any invalid, `Abort`, keep committed root.
4. `PublishRoot`: the sole logical session writer checks epoch and expected old root (CAS-like **host logical** publication) then swaps the entire vector of prepared domain version handles as one root. **No independent domain becomes visible earlier.**
5. Emit committed outputs with monotonically increasing stream sequence. Retire old root/state buffers only when readers' leases and last physical device events are done.

This is **atomic visibility of logical state metadata**, not instantaneous physical atomicity of all GPU devices nor an automatically durable multi-host transaction. Physical mutable pages must use COW/private tail or an undo/recompute guarantee; *mutating the last committed page in place while GPU work can still fail is forbidden unless safe reconstruction proven*. Logical O(1) root swaps do not mean O(1) recurrent matrix copy.

### 2.3 On GPU-only decode, how to avoid per-token host barriers

A GPU-native `StepProgram` can compute multiple dependent row/token iterations in a **closed compiled execution island**:
- model forward, routing, penalties/grammar mask where GPU-qualified, sampler RNG, state update/verification, provisional token buffer;
- controls: max iterations, accepted frontier, device EOS/stop flag, legal interruption point and bounded output/step memory;
- device-side input buffers and index maps; run captured graph/loops without host fetching each router or draft token;
- explicit final `CompletionRecord` and `PreparedStateRoot` metadata sent to host when necessary for streaming/admission/next plan choice.

Not every model or sampling/grammar implementation can execute multiple steps on GPU. On those, the correct plan uses smaller islands or host transitions and reports mode; it must **never** speculate past incompatible grammar/RNG/cancellation state. A single target sample visible to user still requires a logical publication/stream decision, but this need not block the next independent GPU stream if device supports safe double-buffering and epochs.

A two-in-flight host scheduler is *optional* and only legal where step `N+1` inputs don't secretly depend on state or tokens not committed from step `N`, or the execution island handles that provisional dependency internally.

## 3. Worker step protocol and backpressure

```rust
struct WorkEnvelope {
    session: SessionId,
    attempt_epoch: SessionEpoch,
    expected_root: StateVersionId,
    bundle: ExecutableBundleId,
    shape_class: ShapeBandId,
    rows: RowBindings,            // stable slot indices + generations
    leases: LeaseBatchId,         // globally reserved/locally realized
    dependencies: Vec<EventId>,
    output_quota: WorkOutputBudget,
}
enum WorkerReply {
    Submitted { ticket: TicketId, event: EventId },
    Complete { ticket: TicketId, events: Vec<EventId>,
               candidate: ProvisionalCandidateId, stats: StepStats },
    Failed { ticket: TicketId, reason: WorkerError,
             may_have_live_dma: bool },
}
```

These types can be compact, pooled fixed-capacity tables rather than dynamic Vec in hot code. WorkEnvelope represents one precompiled region/stage assignment, **not an IR graph copy**. DeviceWorker maintains persistent batch slots and updates only deltas. Every slot/lease/ticket carries generation and session epoch; late duplicate worker completions are discarded, but buffers remain alive until actual device event/cancellation drain.

**Backpressure:** ingress bounded by request memory and timeout budget; event queue separates control/cancel from output chunks; output buffers bound by client acknowledgments or credits; device memory quotas allocated ahead of submission. A slow HTTP client cannot pin infinite GPU KV or an unbounded PCM/video queue. TaskCoordinator can continue unrelated sessions while one session is waiting on SSD/NPU/remote IO.

**Errors:** typed `UnsupportedShape`, `UnqualifiedDevice`, `StaleEpoch`, `BufferLeaseInvalid`, `DeviceLost`, `IoShortRead`, `StateIncompatible`, `OutOfBudget`, `OutputSchemaInvalid`, `Cancelled`; attach retry classification. Never "retry" failed partial output with incremented root blindly. All physical failures map to a valid last committed frontier or a typed unrecoverable response.

## 4. Device/command graph boundary

**Only these runtime primitives are mandatory initially**:
`Reserve`, `AllocateFromPool`, `Submit`, `Event`, `WaitAtFirstConsumer`, `Ready`, `Quarantine/Release`.

The planner may represent physical work in CIR commands:
`Acquire/Copy/Import/Pack/Dispatch/Wait/Join/PublishResidency/Release`. A semantic `Commit` is **not** a free-standing device-visible state publication command: CIR may provide a `BarrierBeforePublication` and an event proving all required writes are complete. TaskCoordinator still owns `PublishRoot`.

A CIR plan has typed buffer aliases and explicit availability/failure dependency DAG. Host↔device events, CPU/NPU/SSD async IO and captured GPU streams may require different primitive mapping; the semantic result is not ready merely because the host submitted a callback. Device-specific constraints are checked at compile time where known and at step binding for dynamic buffer generations/capabilities.

**Cross-accelerator equivalence:** A generic tensor op split into independent output rows may overlap two compute devices; routing expert subsets across discrete/iGPU may overlap. In both cases `Join` waits on true first consumer; source quant and model top-k/position/state remain unchanged. GPU+NPU power/firmware qualification can **veto** an otherwise legal computation plan. No root-clock/suspend/IOMMU changes automatically.

## 5. Hierarchical physical budget and local allocation (zero global per-token lock)

```text
Hardware/TopologyProbe + conservative OS/cgroup physical RAM snapshot
              |
     PhysicalResourceArbiter (global admission)
       reserve permanent weights and task/session peak
       reserve per-device subpool budget, tenant fairness/headroom
              |
  Device A pool       Device B pool        Host/IO pool
  fast slab/pages      async allocator      pinned/SSD staging
  local generations    local generations    local generations
       \                  |                    /
        completion/retirement events → release global reservation
```

**Global owner** controls who has reserved capacity and handles low-memory pressure/eviction coordination at coarse boundaries; it does not synchronously allocate every GEMM scratch tile. GPU/CPU/NPU contexts manage their own fast local pools with explicit accountable reservation ticket. Resources include model weights, KV/recurrent/indexer/prefix, expert slots, adapter/draft, encoder/denoiser scratch, graph programs, transferred activations, PCIe and pinned DMA, storage IO cache and OS reserve.

**No impossible OS promise:** concurrent processes, kernel/driver pinned allocations, compaction, driver arena growth and temperature/power may change available capacity after probing. Admission with safe headroom + failure handler + cgroup metrics is the guarantee; a model/task requiring memory cannot be admitted if actual capacity/pressure cannot be bounded. Device allocation can fail safely, and optional cache released/replay model if appropriate. Do not assume virtual GTT = free DRAM.

**Allocator contract:** local `Allocate(reservation,shape,layout)` returns `BufferLease{allocation_id,generation,range,tier,readers,writer,last_event}`, no silent in-place alias conflicts. `Retire` queues asynchronous release after final reader fence. NPU/remote DMA timeout may be **unsafe to free**, so retain/quarantine/reset. Resource ownership doesn't grant state publication authority.

## 6. Model/task/plan compatibility keys

Keep distinct canonical identities instead of one giant runtime hash:

- **ArtifactKey:** exact source checkpoint/shards, tokenizer/template, adapter generation, encoder/projector/vocoder/media and quantization schema.
- **SemanticKey:** normalized operator equations/version, position and state effects, source weight logical roles, approved numerical mode.
- **TaskKey:** task control schema, sampling/grammar/voice/noise/stream semantics and external-action policy boundary.
- **StateCompatKey:** artifact+semantic+task relevant prefixes, full required state schemas, position/quant, adapter/sampler/tenant salt/medium fingerprints and snapshot generation.
- **BundleCacheKey:** semantic region/candidate, physical layout, shape band, device ISA, kernel compiler/runtime driver, numeric mode, eager/capture, graph pointer-layout and program version.
- **RuntimeBinding:** session epoch, current root, batch slot indices/generations, current expert table/prefix cache state, physical reservation. **Do not include runtime binding in compilation cache key** or variant explosion follows.

Persist source blobs, schemas and state snapshots with versioned checked formats; **never serialize raw pointers, device events or compiled driver object handles** in a durable model snapshot. Bundles are expendable caches rebuilt on toolchain/driver upgrade; a state snapshot is separately schema-versioned and may require explicit converter/replay.

## 7. Logical output commit versus transport delivery

A TaskSession publishes valid `StateRoot` before making emitted `OutputEvent(session_id,epoch,output_seq,kind,payload_ref)` visible. For real streaming protocols:
- **Default semantics:** ordered, **at-most-once live emission attempt** per connection, no false promise of exactly-once client receipt on disconnect. An app may have received an event whose acknowledgment was lost.
- **Optional resumable delivery:** client cursors and a bounded/durable replay buffer, with explicit identity/retention/auth/privacy and idempotency. A resumed stream can supply already committed events with sequence numbers, but the client must deduplicate.
- **Cancellation:** previous complete committed events remain valid, pending tentative state aborted, leases retired after physical quiescence; media stages have their own "complete chunk" boundary.
- **Distributed:** each shard/rank stages provisional results under one coordinator epoch, sends `Prepared` with actual device completion, publishes authoritative root only after the required cohort ready. Rank timeouts and lost ACK require **fencing generation and recovery policy**, not assuming instant all-rank atomic memory. A durable exactly-once guarantee needs separate log/replay consensus and is out of inference MVP.

The TaskCoordinator may use application-level exactly-once identifiers, but it doesn't control an external network consumer that disappears before ACK. This is a correctness and honesty requirement, not a performance feature.

## 8. Subtask control without a giant workflow runtime

The core TaskDriver is trusted Rust orchestration at **stage-level/loop-level**, not arbitrary executable user-defined program. A stage invokes a compiled semantic bundle and specifies `NextStage` or `Repeat` plus resource/commit guards. Its `poll`/completion transition may allocate/submit to a worker, but steady token or denoiser arithmetic can run in device-native StepProgram without repeating full TaskDriver graph search for every math op.

Task examples:
- language: tokenize→prefill→decode/verify→commit tokens; host transition at true sampler/tool/fairness boundary;
- encoder/pooling: batch→encode→pool→commit vector;
- audio/TTS: window/acoustic frame loop→vocoder→PCM chunk commit;
- image: denoising step loop→decode→image emit; intermediate latent private;
- video: frame/window stage, temporal cache and mux output commit;
- **future training**: gradient/parameter update task with stable source `ParameterVersion` and optimizer state, separate authorized update/weight publication; not built into ordinary LLM decode.

External tool execution and multi-agent workflows belong above TaskRequest in a separately permissioned application scheduler; they may invoke TaskPrograms but cannot be turned into arbitrary kernel effects.

## 9. Failure semantics matrix

| Event | Logical response | Physical response | User-visible behavior |
|---|---|---|---|
| kernel returns UnsupportedShape before launch | choose prepared generic plan | no new device work | mode receipt fallback reason |
| device fails during a provisional recurrent update | abort new root candidate; replay if safe | wait/drain/quarantine dirty pages | no emitted failed token/frame |
| NPU/IO timeout with possible active DMA | discard candidate and mark provider unhealthy | quarantine buffers or safe device reset | GPU-only future request if safe; no false auto recovery |
| client disconnect during stream | retain committed root to defined session policy; cancel pending | retire leases after fences | previous emitted sequence may have been received |
| unavailable memory after reserve | refuse or evict optional cache/replan from root | free safe retired buffers; don't OOM attempt | typed budget/fallback |
| mismatch artifact/adapter/quant/task schema | no shared state/cache/plan mutation | invalidate incompatible cache handles | report state compatibility failure |
| remote rank lost after prepare | stop commit until epoch fenced/recovery policy | drain or discard provisional peer work | no partial new output |
| model/device program source untrusted | refuse execution | no device load | typed untrusted artifact |
| device power/firmware profile unqualified | GPU/CPU compatible plan | never auto-tune privileged system settings | mode receipt unavailable device |
| approximate numeric plan not authorized | return strict/generic candidate | no lossy input packing | unchanged model/sampler quality contract |

## 10. First vertical slice acceptance for this document

Required tests:
1. Mock two StateDomains + RNG; second prepare failure leaves exact original `StateRoot` visible and no output emitted, no UAF.
2. Fake CUDA/HIP/XRT and CPU streams with delayed out-of-order completion, stale `TicketId` and BufferLease generation; ready means correct memory visibility before physical memory release.
3. Per-device slab allocations under one global reservation; 32 in-flight sessions don't take one global allocation mutex per output operation; no unaccounted pinned buffers.
4. GPU `StepProgram` two dependent provisional target steps with *zero* host router lookup or mandatory sampled-token transfer where backend supports it; reference/quality and sampler state match.
5. Output disconnect/retry tests show stable event `output_seq` and no premature output; no assumption of network exactly-once delivery.
6. Shape-band guard selects legal candidate for 1,7,128,4096 and ragged inputs, with graceful generic fallback on unsupported exact tile; no unbounded JIT.
7. Forced specialized whole region disabled: session/task continues through generic path and no duplicate serving runtime.
8. Optional multi-rank protocol injects lost ACK/timeouts; logical writer cannot publish before successful prepared cohort and fencing rules.
9. Memory pressure while NPU pinned buffers potentially active: safely refuses/replans or quarantines without touching privileged driver power controls.
10. Time critical path end-to-end compared with simple handwritten direct-dispatch reference; bounded host CPU overhead and fast async event wake.

**Anti-goal:** there is no requirement to build all ten tests before one dense model run; implement 1–3 plus 7 first, then expand. Keep the system minimal; every new primitive must close a demonstrated correctness/performance gap.

## v4.1 addendum — graph/dataflow accelerator readiness and state bridge

A DeviceWorker is not necessarily a CUDA-like GPU kernel submitter. Its `ExecutableBundle` may contain `NativeKernel`, `CapturedCommandGraph`, `AheadOfTimeDeviceGraph` or a **StaticallyScheduledDataflow** program, each with a capability-checked `ProgramLifecycle`, memory residency and external event/fence interface. **One submission can represent a whole compiled model stage or statically scheduled accelerator network**, with device-internal tile/collective schedule delegated to the compiled target program; host TaskCoordinator should not simulate its internal GEMM operations.

**State is the limiting correctness contract:** If a TPU-like graph compiler or remote wafer program owns opaque recurrent/KV/sampler data, its adapter must produce a versioned **provisional** state candidate, support explicit snapshot/restore/recompute or declare a stateless-only region. A service returning final text and hiding all state is **not** a qualified RegionExecutionProvider; it is an external TaskServiceAdapter with different semantics. TaskCoordinator remains the only `PublishRoot` authority after ready/fail event, per-rank epoch checks and correct effect schema.

**Physical resource accounting** includes compile-time code cache, warmed/running program image, on-chip SRAM allocations, HBM/UMA buffers, streamed external weight sources, graph execution scratch, transport bandwidth, inter-chip collective occupancy, and network/privacy/egress budgets. A static program's claimed memory **must be observed/probed**; aggregate rack memory is not one ordinary pointer. A short-lived stateless program can be pipelined with local GPU segments only where transfer and numeric/state layout bridges exist.

**Lifecycle:** `Compile→Load/Upload→Warm→Reserve→Submit→Ready/Failed→PrepareState→PublishRoot→Retire`. Compilation and upload can dominate cold start. Bundle caches are disposable, **no persisted device pointers**, bounded shape variants and support fallback on shape miss. On hardware firmware/SDK/driver upgrade invalidate executable fingerprints and state-snapshot compatibility as necessary, not the mathematical model. No assertion a native Rust TPU/wafer/LPU compiler exists until source/toolchain+hardware proof. Related full contract: [accelerator execution families](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## 11. v4.2 — per-session root publication across remote state shard owners

**A StateRoot may refer to local and remote prepared DomainVersionHandles** with StateShardGeneration, storage/compute owner, lease epoch and state compatibility. TaskCoordinator remains exactly one logical writer per session/shard across the entire cluster. Each RemoteRegionWorker validates model/quant/semantic/numeric/program/position and the expected StateRoot and PlacementEpoch before executing into **private provisional** state; it returns Ready/Failed plus a lease/fence reference. Device completion or network ACK is never a new visible session root.

**Publication sequence:** (1) pin old root and placement epoch; (2) reserve all destination state/resources/network credits; (3) execute closed regions or transfer compatible state; (4) require all participant Ready and matching semantic/frontier epochs; (5) one TaskCoordinator PublishRoot and sequenced output; (6) send idempotent CommitNotice/Release after completion to remote peers. Failure before publication retains old root; failures after publication require old retained checkpoint or correct compatible replay, not an unsupported exactly-once machine transaction. Expiring a remote shard lease or losing coordinator must not permit stale peer writers. If no durable coordinator quorum exists, **fail-stop/reconstruct** rather than promise HA.

A RemoteRegionWorker can own a layer's KV/SSM physical buffers for many tokens, so per-token cross-machine work transfers activation tensors at layer-group boundaries and **not** the complete cache. Attention may execute on the KV host with compact query/output exchange, or state may migrate once at a committed frontier. Remote full KV reads on a tunnel are generally a costly candidate. Typed leases/delayed DMA retirement continue across transport loss. A missing remote state snapshot cannot be faked by local text prefix hash.

Global physical budget remains coarse; authorized peers reserve locally and report headroom. Cluster capacity is not one globally coherent pointer. Program ownership stays in RegionExecutionProvider; control/metadata transport can be tunneled while physical data chooses another qualified path. **No new global distributed lock per device operation.** Read [remote state protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md).

## 12. v4.3 — model routing does not own token state or physical allocations

`ModelCatalog` resolves one immutable model/alias/profile revision at **admission**; `ModelRouter` chooses an eligible warmed or safely loadable resident `DeploymentId` and pins a `ResolvedRoute` for TaskCoordinator. **TaskCoordinator remains one logical writer per TaskSession**, DeviceWorker/RegionExecutionProvider remain owners of compiled kernel/graph execution, and ResourceArbiter remains sole physical admission owner; adding more models must not create an independent `EngineCore`/sampler/KV allocation authority per model.

An active session retains exact `ArtifactKey+SemanticKey+Quant+Tokenizer/Adapter/Template+NumericPolicy` in its StateCompatKey. Replica balancing across the **same logical source model** may route a new stateless request to a different model replica; stateful continuation sticks to a compatible state owner unless a full committed frontier snapshot/lease/placement migration is validated. Hot alias changes never rebind a running state root. Multi-model TaskProgram target/draft/encoder/vocoder stages hold separate registered model leases under one task's budget.

Model lifecycle controls run **outside the device-native StepProgram**: deduplicated source verification, physical reservation, loading/plan compiling/warming, ready→idle→drain→unload after quiescence. Idle weight eviction cannot discard a live unsnapshotable KV/GDN, in-flight speculative partial or DMA; memory stress may evict safe optional cache/idle models or queue/refuse. Registered catalog models need no resident allocation. One `run` invocation can create local in-process engine; `serve` wraps that same engine with API adapters. No per-token HTTP/gateway hop.

**Responses/client streaming** uses the same root-first, sequenced output/at-most-once *attempt* contract; API compatibility translates schemas, never commits its own state. Tool-call generation is data, not an external tool execution authority. [Router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [CLI/API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).

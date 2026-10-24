# Canonical Sourcebook — volume 1

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md

Original UTF-8 bytes: 23581; SHA256 `33ec1a0b63efff4fa71fc0c8e56766ed707680ac8ae9fc470328fa133719ba08`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — Universal region-execution backend contract (v4.1)

Design requirements below. Implemented/tested scope and unfinished work are
tracked only in [the plan TODO index](../plans/TODO.md).

**Status:** authoritative technical design refinement, 2026-10-12. **No GPU/TPU/wafer-scale/remote implementation, vendor compiler integration, driver install or benchmark was performed.** Hardware-specific first-party references and license caveats are in `ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md` (NON-NORMATIVE). The architectural goal is one **minimal provider interface** able to target CPU/GPU/NPU, local/rack graph processors and static dataflow fabrics without a model-specific runtime fork.

**Authority:** v4 `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`, `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md` and the master plan still govern mathematical semantics, task control, buffer ownership and one logical StateRoot publication. This specification **generalizes KernelProvider** to the full region/backend space; it does NOT introduce four more durable IR representations or distributed services.

## 1. The missing abstraction is *granularity*, not another framework

Some targets efficiently launch one hand-tuned tensor tile, some capture a fused device graph, others require a **whole bounded computation subgraph**, and some physically lay out a **static pipeline across a processor fabric**. Designing all targets as `launch(kernel, pointer, stream)` would force an inefficient host simulation on compiler-centric accelerators.

**One core abstraction: `RegionExecutionProvider`** (single registration/selection seam). The existing `KernelProvider` is its **native-kernel special case**, not a second competing engine. Every provider consumes **the same closed `SemanticRegion`** with exact input/output/shape/state effects and produces a device-executable **variant** bound to a physical command/resource DAG.

Allowed granularity is provider reported, not dictated by host `KernelProvider` assumptions:
- `Operation` — one tensor/GEMM/attention or few ops;
- `FusedRegion` — several pure+effect-compatible ops including recurrence;
- `ModelStage` — whole transformer layer, encoder, denoiser, vocoder or task stage;
- `ModelProgram` — most or all math for a prepared model execution regime;
- `PartitionedProgram` — statically scheduled multi-device/wafer/collective program, possibly with weight streaming, constrained by topology and authoritative state boundary.

**Closed** means all external observable inputs, outputs and effects are known and checked. A graph compiler can own more internal operations and opaque temporary layouts, but **cannot** take control of task authentication, model source integrity, session ownership or user-visible state publication.

## 2. Five physical execution kinds, one provider protocol

```rust
enum ExecutableKind {
    NativeKernel,             // direct tensor/tile/region entry point
    CapturedCommandGraph,     // driver graph handles, persistent parameter bindings
    AheadOfTimeDeviceGraph,   // whole supported subgraph via offline compiler
    StaticallyScheduledDataflow, // device/fabric micro-schedule preplaced by compiler
    ManagedDeviceSession,     // only if state/IO/effects contract is exposed and qualified
}
enum RegionScope {
    TensorOp,
    FusedRegion,
    ModelStage,
    ModelProgram,
    PartitionedProgram,
}
trait RegionExecutionProvider {
    fn capabilities(&self) -> ProviderCapabilities;
    fn candidate_regions(
        &self, closed: &ClosedSemanticRegion, topology: &TopologySnapshot,
        numerical: &NumericContract,
    ) -> Vec<RegionCandidate>;
    fn prepare(
        &self, region: &ClosedSemanticRegion, candidate: &RegionCandidate,
        compile_budget: &CompileBudget,
    ) -> Result<PreparedExecutable>;
}
```

This is **one** conceptual Rust provider trait; the actual first implementation may use an enum/table with static dispatch instead of dynamic trait objects. The provider can return an `UnsupportedOp/Shape/Precision/Topology/StateBridge` reason; only candidates passing gates reach dispatch.

`NativeKernel` and `CapturedCommandGraph` use local stream/launch/device graph APIs when supported. `AheadOfTimeDeviceGraph` can require vendor/compiler IR import/export of a legal subset and a compiler runtime instance. `StaticallyScheduledDataflow` may depend on a fabric-wide compiled layout and resident kernel network. `ManagedDeviceSession` is narrowly permitted when **provable** state/effect and cancellation/budget semantics exist; a mere opaque inference text API is **not** a qualified model-region provider.

No extra "wafer compiler IR" or "TPU semantic model" is introduced into SIR.

## 3. Provider capability contract is richer than kernel capability

```rust
struct ProviderCapabilities {
    granularity: SupportedRegionScopes,
    executable_kinds: ExecutableKindSet,
    semantic_op_set: VersionedOpCapabilitySet,
    tensor: TensorQuantShapeCapability,
    dynamic_shapes: ShapeDynamism, // static, bounded, symbolic, ragged
    control: DeviceControlCapability, // branches, loops, iteration bounds
    mutable_state: StateBridgeCapabilities,
    memory: MemoryDomainCapabilities,
    topology: FabricTopologyCapabilities,
    transport: HostDeviceTransportCapabilities,
    lifecycle: CompilationAndProgramResidency,
    reliability: FailureAndSafetyCapabilities,
    source_and_toolchain: BuildAndLicenseIdentity,
    qualification: EvidenceIdentity,
}
```

These are **capabilities advertised by measured probe and compiled program**, not marketing assumptions. An engine may accept one model's full graph but not a different model's recurrent operator, scalar routing, custom sparse top-k, dynamic context or quant codec. Support matrix labels `RECOGNIZED, CAN_LOWER, COMPILES, FUNCTIONAL, STATE_SAFE, QUALITY_VALIDATED, PERFORMANCE_QUALIFIED`; absent true devices cannot reach physical performance qualification.

**Compile granularity/shape** matters: a static graph needing `seq_len=4096` may accept a bounded dynamic variable only where its compiler supports it. Shape buckets, algebraic legal padding and ragged guards are compiled with explicit bounds and cost. Unsupported dynamic shape uses smaller portable closed region or a different topology; no unsafe hidden truncation or unbounded JIT.

## 4. One mathematical representation, many backend lowerings

The semantic program **owns mathematical truth** independently of compiler inputs or external graph dialects. `ClosedSemanticRegion` carries:
- exact operator schema versions and tensor logical shapes/layout constraints;
- checkpoint/adapter semantics, storage quant vs compute/numeric policy, position and sparse selection;
- state effects and expected immutable original `StateRoot` + provisional domain outputs;
- task phase and loop/stream boundary (prefill, decode, draft verify, encoder, denoiser, audio/video etc.);
- reference evaluator and portable fallback cut, plus candidate operator legalizations.

Provider lowering can generate:
1. **Rust direct low-level kernel** or compiled host/device code;
2. **device command graph** with kernel and async memory events;
3. **typed external compiler graph input** (only semantically compatible operations, known canonical quant/state wrappers; do not arbitrarily serialize unsupported custom ops);
4. **static graph/fabric layout program** with compiler-managed on-chip memory and link routes;
5. **remote closed-region protocol** *only if* exact ownership/state/quality rules can be satisfied.

No `ModelDefinition::if_backend_is_TPU...` condition. Any target-specific hand-tuned entire model region remains a provider candidate replacing the same closed semantics. Unhandled operator -> compiler partitions where legal to a reference/alternative provider, with *explicit* transfer/materialization cost and state conversion. Some graph devices require a whole-model eligibility decision because frequent tiny graph breaks erase their advantage; this does not imply a new server.

## 5. Device state bridge is essential for graph-only and external fabrics

**The universal hard problem** is a remote/graph device maintaining opaque mutable KV/recurrent/sparse state while our coordinator requires atomic logical **Prepare→PublishRoot** across domains.

```rust
enum StateBridgeMode {
    ExplicitProvisionalVersions, // private candidate state; preferred
    SnapshotAndRestore,          // verified bounded snapshot/rollback protocol
    RecomputeFromCheckpoint,     // last committed input history, bounded cost
    StatelessProgram,            // encoder/pure region with no mutable state
    OpaqueUnrecoverable,         // NOT qualified for stateful region reuse
}
struct StateBridgeContract {
    mode: StateBridgeMode,
    owned_domains: Vec<StateDomainId>,
    schema: StateSchemaId,
    export_import: StateSerializationCapability,
    fork_replay: FrontierAndRollbackCapability,
    memory_lifetime: ProviderLeaseContract,
    epoch_fencing: EpochCapability,
}
```

**No provider may publish semantic TaskSession state.** A vendor graph can mutate provisional device buffers privately and return `PreparedCandidateId + EventReady`; coordinator checks its logical state root's expected epoch and publishes as one. A device with only in-place opaque state and no verified snapshot/rollback might still run **stateless encoder** or a fully owned bounded read-only stage, but may not falsely claim safe speculative session migration/cancellation. It can be restricted to one whole-session opaque mode **only under separately disclosed guarantees**, not a default portable model runner.

A managed remote inference API that does not expose lowerable math, intermediate state, full KV/RNG/quant identity and buffer lifetime should be represented as an **application-level TaskServiceAdapter**, *not* a substitute for a native RegionExecutionProvider. It can provide alternative service results but is not evidence the engine runs kernels on that accelerator or can compose fine-grained state.

## 6. Physical topology is a graph of memory, compute, links and *compiler domains*

```rust
struct ComputeDomain {
    kind: ComputeEngineClass, // CPU SIMD, GPU matrix, NPU tile, static fabric, remote
    kernel_features: NumericAndOperationCaps,
    program_context: ProgramResidencyAndCompileIdentity,
    supported_execution: ExecutableKindSet,
    scheduling_owner: MicroScheduleOwner, // device compiler, kernel runner, host
}
struct MemoryDomain {
    kind: MemoryKind, // CPU RAM, coherent UMA, VRAM/HBM, on-chip SRAM, SSD, remote
    bytes_observed: CapacitySnapshot,
    ownership_and_coherence: CoherencyRules,
    buffer_mapping: BufferAddressability,
    reservations: QuotaAndSafetyRules,
}
struct TransportLink {
    src: ResourceDomainId,
    dst: ResourceDomainId,
    bandwidth_and_latency: CalibratedLinkCost,
    coherency_and_copy: TransferRules,
    collective: CollectiveCapability,
}
```

Hardware-system profiles differentiate:
- small **coherent CPU+GPU UMA**: one physical memory pool, shared bandwidth and no pretend HBM, direct CPU/GPU alias only where actual access/fences legal;
- **discrete VRAM/HBM GPU**: separate memory and PCIe/peer transfer, native low-bit instructions and graph capture, NVIDIA/AMD architecture-specific precision;
- **rack-scale multi-GPU**: NVLink/NVSwitch or Infinity/UAL fabric, rank group/collective placement, link topology and epoch fencing; aggregate HBM cannot be blindly considered one ordinary pointer;
- **compiler-centric TPU/NPU**: device graph-compiler/runtime program handles, program-owned local memory and approved compiler-runtime transport; *not* universally raw GPU kernel dispatch;
- **wafer/static dataflow**: compiler-owned mapping/PE grid/SRAM pipeline and streamed weight sources, with on-device program lifecycle and very different warmup/context costs;
- **remote accelerator**: transport/auth/privacy/egress/latency and opaque device program; fallback to app-service boundary if full contract unavailable.

All these physical systems share the **same SIR and TaskDriver**, but may have very different `ExecutableBundle` physical descriptions and eligibility.

## 7. Program lifecycle and cost are first-class

A graph-only accelerator may spend seconds/minutes compiling, reserve model SRAM/HBM/tensor program memory and reuse a compiled model graph over thousands of steps. A GPU might compile a small quant GEMV then issue many launches. A wafer-scale static pipeline might stream weights layer by layer or pin a trained fixed model over a group of processing elements.

```rust
struct ProgramLifecycle {
    build: BuildFingerprint, // compiler/vendor runtime/driver/program version
    compile_cost: CostRange,
    executable_size: u64,
    load_upload_cost: CostRange,
    warmup_cost: CostRange,
    persistent_device_memory: ResourceFootprint,
    parameter_weight_residency: ResidencyPolicy,
    shape_specializations: ShapeBandSet,
    teardown_eviction: LifetimeContract,
}
```

**Scheduler levels stay distinct:**
1. `TaskCoordinator` owns admissions/fairness/session committed state, output order and stage boundaries (Rust core).
2. `RegionExecutionProvider` at model load chooses closed region placements and compiles/loads device programs (may call approved vendor compiler/driver).
3. **Device-internal schedule** (warp/tile/ICI/wafer dataflow/collective) is owned by the target compiler/runtime *inside the compiled executable*, not simulated one micro-op at a time on the Rust host.
4. `DeviceBackend` submits executable program handles with device-local resource leases and typed event/epoch fence; returns Ready/Failed and provisional state to coordinator.

The runtime cannot assume every graph device has an asynchronous small-kernel callback or that early chunk output is available before a whole statically scheduled region finishes. Choose region/cut sizes by **host↔device transfer and compiler warmup cost**, not maximizing number of polymorphic provider calls.

**Cost objective:** quality/safety first, then measured whole TaskProgram latency/P95/throughput/energy/cold start/tenant fairness and egress; compile/load amortization and memory residency may dominate one model request, especially graph-only remote environments.

## 8. Quantization is a typed numeric encoding, not a hardware marketing label

`StoredQuant` from immutable checkpoint, `PhysicalWeightLayout` from target packing and `ComputeEncoding` (with rounding/block scaling/reduction) must be distinct. Examples of capabilities:
- FP32/BF16/FP16, FP8 and various accumulator modes;
- block-scaled FP4 including **different** 16-element FP8 scale vs 32-element power-of-two scale; never assign them one generic "FP4" binary codec;
- hardware-specific 4/6/8-bit microscaling, block-floating NPU layout and signed integer group formats;
- specialized GPU low-bit quantized GEMV vs device matrix GEMM and larger graph/pipeline numerical policy.

A provider must pass per-op/dequant/reference and full-model/task output quality, including sparse indexer selected IDs, state recurrence and speculative distribution if claimed. **No automatic conversion** of a high-quality checkpoint into a different lossy model merely to fit a specific accelerator; a new quant artifact is a separately fingerprinted model, user policy and quality tier.

## 9. Partition, collectives and remote/wafer failures

A `PartitionedProgram` may own several cooperating accelerators and internally route tensor shards/experts/layers. Need explicit:
- `TopologyId` and `CollectiveGroupId`, rank/chip set and allocation leases;
- static/interconnect compile layout, weight-stream source and stage ordering;
- per-rank versioned model/quant/semantic/StateCompatKey and TaskFrontier;
- physical event readiness, epoch fencing and lost-rank abort/replay;
- privacy/auth/egress and whether remote model operands can be sent legally;
- never reinterpret an opaque remote model's returned string as full authoritative target tensor/logit state.

A RegionExecutionProvider may compile a remote **graph program** via an authorized compiler/driver and submit compiled work, with our Rust TaskCoordinator retaining publication. This is an **architecture capability**, not a promise a SaaS model endpoint allows custom program execution. On a provider lacking source compiler/build and state API, only external TaskServiceAdapter is possible, and its semantics are explicitly different.

## 10. Seamless design vs actual hardware implementation

| Stage | v4.1 claim for every device family |
|---|---|
| **Architecture compatible** | can describe closed SIR region, task stage, program kind, physical tiers, memory/schedule/shape and state readiness; no new model server |
| **Backend feasible** | actual SDK/compiler/driver permits source-supported region lowering without unacceptable licensing/Rust-source violations |
| **Compiles** | produced real program for actual target and shape, with sourced BuildFingerprint and no unsupported op silently guessed |
| **Functional** | ran reference-verified region/model/task and handles shape/cancellation/memory safely |
| **State-safe** | correct snapshot/prepare/fork/replay or honest constrained stateless mode; no partial PublishRoot |
| **Performance-qualified** | matched actual target workloads, model/quant/driver/power/task latency/P95/energy; compared to generic and source peers |
| **Production-qualified** | resource/security/reliability/recovery/long-run multi-tenant tests pass for exact hardware/software profile |

**No physical source/toolchain support** means `Architecture compatible`, **not** `Functional` or `Performance-qualified`. A Rust-owned core using a vendor compiler is not a Rust-native device kernel compiler; which external compiler/FFI APIs are acceptable is open user policy. Preserve CPU/GPU portable functionality when TPU/wafer/LPU SDK not available.

## 11. Minimum implementation surface and extension drill

**Do not add one crate or trait per platform.** Introduce at most **one `RegionExecutionProvider` seam** that generalizes old `KernelProvider`, and one `ProgramLifecycle/StateBridge` capability carried on a candidate. A small enum dispatch table can encode `ExecutableKind` in one Rust module. CIR already understands async events, buffer leases, IO and remote; expand its **command payloads for SubmitProgram/Collective/ExternalEvent** only when a real backend proves need. Two enduring representations remain **SemanticProgram** and **ExecutableBundle**.

**Proof of composability:** implement a fake *graph-only* accelerator with a model-stage compiler and static shape guard, and a fake *stateless static-dataflow* fabric with scripted weight stream and high compile/warmup cost. Compare both against existing CPU/GPU direct kernel with:
1. same SemanticRegion/TaskDriver and exact reference tensors/output;
2. same TaskCoordinator and Prepare→PublishRoot where state supported;
3. no new server/tokenizer/model-specific allocator/StateDomain implementation;
4. shape miss fallback and compilation cache invalidation;
5. no hidden transfer of full user prompt or model to unmanaged remote endpoint;
6. correct device-internal collective graph without host micro-op interpretation;
7. transparent `NeedsBackendCompiler/StateBridgeUnsupported/LicenseUnapproved` status for devices without actual toolchain.

Later real hardware qualification: GB10 unified and Arm64 toolchain; consumer/discrete and B200/B300 native FP4; rack topology; TPU AOT graph; AWS Neuron-style graph; static dataflow/WSE/LPU if SDK accessible; AMD CDNA, Gaudi/TT and other regional accelerators as demand supports. This is a **roadmap, not a list of already implemented devices**.

## 12. Final rule

**Architecture cannot require a new inference engine when the minimum efficient device execution unit changes from a GEMM to a whole graph or wafer-scale program.** The same verified SIR and trusted TaskDriver must lower to multiple qualified physical execution families and obey one session-StateRoot publication/resource/quality contract. A generic opaque model API that cannot expose those semantics is a separate service, not a falsely universal kernel backend.

**Absorb the mechanism, not another vendor's entire serving runtime; never claim native execution until source/compiler/hardware and state contracts are proved.**

## 13. v4.2 — one RegionExecutionProvider across network topologies

A PartitionedProgram may cover GPUs/NPUs/graph accelerators **on different physical machines**, not just chips in a static rack. Its candidate includes independent **compute region**, model weight residency, per-session state shard owner, immutable cache location and TransferPlan under TopologyEpoch/PlacementEpoch; the provider may internally own fast collective/wafer micro-schedule, while the session TaskCoordinator retains PublishRoot, admission, output and authorization. A new remote machine does **not** require a new semantic model, new TaskDriver, or a separate model-specific serving process.

**Mandatory binding:** RemoteRegionWorker program/model/quant/semantic hash, supported state layout/snapshot/Prepare, device/fabric compiler and shape, trust/data handling and physical link transfer/queue limits. A remote model API that returns only text and hides mathematical state remains a TaskServiceAdapter, not a native region backend. If a session uses remote attention with state on B and FFN on A, region cuts and transfers are explicit in CIR; if B only stores immutable prefix, hydrate verified snapshot or recompute; if a fast fabric actually permits direct reads, quantify copies and coherence. Don't “optimize” a WAN tunnel as if it were NVLink.

**Topological state consistency:** mutable remote DomainVersionHandle includes StateShardGeneration and lease epoch, while content-addressed cached prefixes are optional hints. Dynamic placement changes only at safe committed frontier, peer-group fencing and buffer/event quiescence. Model state may be split by layer or valid attention token range, but an operator-specific partition algebra proof is necessary for compressed/indexed/recurrent types. RegionExecutionProvider can hand a whole compiled graph to a remote node without the Rust host issuing per-operator RPC calls.

**Normative:** [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) and [peer/tunnel state protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md).

## 14. U-048 — command transport within one execution provider

The [heterogeneous prefill/replay contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) refines a compiled native or
captured region with transport-specific preparation: ordinary driver launches,
vendor graph, retained queue dispatches or an attested native command body. A
public-HSA packet submitting retained PM4 is distinct from experimental direct
KMD/libdrm control. These are backend details inside the existing executable
kind/provider, not another durable IR or model runtime. Record exact loaded-code
ABI, all dispatch/effect coverage, bounded patches, route-positive physical
receipts and prepared-plan pointer ownership through idle intervals and teardown.
NPU programs similarly select qualified native-driver/optional-runtime transport
without assuming XRT or LLVM23 is universally required. Capability is still not
functional, state-safe, performance or production qualification.

````

## Source: .agents/docs/ASYNC_RESOURCE_COMMAND_IR.md

Original UTF-8 bytes: 19879; SHA256 `77ba19936ac47746988c05339eb5cd6b368ee631e362dfd0f19b3fa8c549bca0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — Async Command/Resource IR (CIR), ownership and compiled execution

**15 October 2026 — STORAGE-AGNOSTIC MATERIALIZATION EFFECTS:** The [first-class SSD streaming contract](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) specializes existing CIR Acquire/ReadModel/Verify/Transfer/PublishResidency/Dispatch/Wait/Release into resident-direct and bounded miss-capable prepared graph islands. It does **not** add a third persistent IR or universal tensor interpreter. A source TensorView retains its identity across GPU/UMA/RAM/SSD while a ReadyTensorLease binds physical generation, real upload event and reader lifetime. **Encoded/captured graph readers count as in-flight even before submit**; no slot reuse until all pending device consumers actually retire, preventing DS4 #384 deterministic wrong-expert logits. Fully resident islands compile away IO requests and host routing; disk misses yield/await bounded host IO, never spin GPU indefinitely. SSD-P0..P5 acceptance in [roadmap](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md).


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**Status:** proposed, unimplemented. Extends typed SIR/RIR/KernelProvider and direct compiled execution. It uses an independently defined, small Rust-native async command/resource graph exposing IO, memory and state dependencies across device and host. Named compiler research is indexed separately as prior art; it is not a dependency or runtime to clone.

## 1. Why a command IR is required

A conventional tensor graph says `C = MoE(A, W)`. It does not express that `W` is a routed expert, may be resident or streamed, occupies a generation-tagged pool slot, requires a valid host read, may overlap another layer, shares scratch with prefill, and has a state/transaction frontier that cannot be published until GPU events complete.

Without a typed command/resource representation, the same dependency appears differently in codegen, event loop, loader, scheduler, offloader, distributor and graph-capture logic; this creates duplicate policies and cross-component bugs.

**CIR separates**: math-equivalent RIR regions and their compiled kernel handles versus actual `allocate/reuse/copy/dispatch/fence/collective/commit` commands with dependency identities, resource leases and failure semantics. A CIR instance is compiled/validated at model load or plan revision; the scheduler instantiates handles and token row maps, not an interpreted CIR op graph per token.

Sources:
- Relevant public compiler/dataflow sources and exact URLs are in `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` and `COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md` (research-only).
- /home/pvl/spaces/abird/research/inference/analysis/synthesis/ZEROGRAVITY-ARCHITECTURE.md §2–5
- /home/pvl/spaces/abird/research/inference/analysis/ds4/ds4-backends-distributed.md

## 2. Type/identity vocabulary

```rust
type CommandId = u32;
type TimepointId = u32;
type BufferId = u64;
type LeaseGeneration = u64;
type KernelVariantId = u128;
type RegionId = u32;
type RankId = u16;

struct ResourceType {
    tier: MemoryTier,                 // device VRAM, UMA, pinned RAM, page RAM, SSD file
    bytes: SizeExpr,                   // checked symbolic/static bounds
    alignment: u32,
    access: AccessMode,                // immutable, read_write, staging, scratch
    lifetime: LifetimeClass,          // model, session, prefix, step, rank
    layout: PhysicalLayoutId,
    state_domain: Option<StateDomain>,
}
struct ResourceRef {
    buffer_id: BufferId,
    generation: LeaseGeneration,
    offset: u64,
    bytes: u64,
    alias_scope: AliasScopeId,
}
struct Timepoint {
    producer: CommandId,
    device: Option<DeviceId>,
    stream: Option<StreamId>,
    epoch: u64,
}
```

Rust signatures are illustrative; implementing packed u32/u64 indexes, lifetime regions and dynamic shape expressions is P04–P07 feasibility work. `BufferLease` is created by ResourceArbiter from a reservation, never through a kernel compiler directly.

## 3. CIR op classes and effects

| Op | Inputs and effects | Ordering guard |
|---|---|---|
| `Acquire(resource, bytes)` | consumes reservation, creates lease/generation | checked physical budget and device addressability |
| `Release(lease)` | consumes lease and final event | cannot recycle before every async consumer quiesces |
| `Alias/Subspan` | typed subview of existing allocation | no illegal mutable alias, no byte overflow |
| `Copy/Transfer(src,dst)` | input immutable read, output write, async completion | device/host access, layout and ordering |
| `ReadModel/LookupRows` | immutable artifact and row IDs, staging write | checksum/short read and model artifact epoch |
| `PrefetchExperts` | source→staging→candidate slots, no publish yet | lease for victim and in-flight read epoch |
| `PublishResidency` | updated expert slot ID/pointer/generation table | device acquire visibility; current layer pins retained |
| `DispatchKernel(variant)` | tensor/state reads/writes, scratch and compiled kernel | semantic effects and resource guards satisfied |
| `CaptureGraph/Replay` | capture-safe dispatch subset and stable pointers | no unapproved allocation under capture |
| `Wait/Signal/Barrier` | timepoint SSA edges | correct GPU queue/host memory order, no unbounded spinning |
| `PeerSend/Recv/Allreduce` | activated tensors and rank model/session epochs | matched collective shape, timeout, authentication |
| `Snapshot/ForkState` | committed state→frozen reference and optional COW | no mutable conflict with surviving parent/sibling |
| `ReadyForPublication(frontier)` | all domain writes/events completed → **prepared candidate** ready for TaskCoordinator | all state domains same causal frontier; CIR **does not publish session state** |
| `Abort/Quarantine` | provisional buffers/state invalidated on error | references held until safe quiescence |

Not all command types must be implemented at P00; define a stable extensible command effect interface, add commands only when real model needs them. Disk and remote operations may be managed by host event threads and compiled into host reactor tasks rather than GPU graphs. CIR describes dependencies, not the implementation of kernel mathematics.

## 4. Async dependency and memory legality

CIR verifier checks:
- each write's consumers see a matching timepoint/event;
- no two overlapping mutable aliases execute concurrently;
- every buffer read is preceded by successful materialization and generation match;
- lease lifetime spans last use and deferred GPU/device event completion;
- every device command is legal on the resource tier and target capability;
- IO errors abort dependent branches and prevent state commit;
- graph capture has stable pointer/resource mapping, and a read of dynamic expert table is legal to replay;
- source/destination alignments checked for DMA or fallback to legal buffered IO;
- mutable state effects are ordered according to SIR and speculative acceptance contract;
- explicit input and output snapshot conversion at any region fallback boundary.

**Important:** OS virtual memory overcommit cannot be proven away solely by static IR. ResourceArbiter still rechecks physical memory before executing an admitted plan and catches allocation errors. CIR checks logical reservation accounting and lifetime.

## 5. Two timelines: static plan and dynamic run

**Static graph**: region commands and dependencies (dense load, QSA, grouped MoE, output) with symbolic token rows. Compile/optimize/capture when weights and physical slots initialized. Cache keyed by semantic IR revision, kernel compiler/driver and layout/shape/quant/numerics.

**Dynamic run**: StepPlan binds session/committed frontier, actual token rows, resource lease handles, current expert pointer table generation, cache hit/miss IO events, and sampler/drafter policy. It is allowed to choose between prevalidated command variants but cannot add an unverified graph rewrite in token loop.

A GPU graph may be completely resident on normal decode. When an expert miss occurs, the safe graph region yields/waits on a bounded event and resumes only once the slot generation is published. The host scheduler never assumes `hipMemGetInfo` proves physical memory.

## 6. Lowering examples

### Resident MoE decode

```text
GDN(state[p])   -> event e1
QSA(latent)[p]  -> event e2
MoE router(e1/e2) -> DeviceExpertSlots(generation)
  -> fast device count/sort/tile -> group GEMV -> reduce
  -> ReadyForPublication(frontier p+1) signals event; TaskCoordinator validates all domains and PublishRoot
```

SIR effect edges determine whether GDN/QSA may overlap. Device routing knows expert pointer table, avoiding host synchronization on cache hits. CIR guarantees cache slot reads protected by GPU events.

### Missing expert

```text
GPU route produces missing expert_id
Host-mapped request ring (bounded credits)
   -> IO Read(artifact hash, offset,len) [error-aware]
   -> Staging BufferLease
   -> VictimSlot Lease waits all readers
   -> Device DMA Upload [copy event]
   -> PublishResidency(epoch,generation)
   -> compute missing expert [async dependency]
```

Prefetch hints may be issued earlier but never change routed IDs. GPU must not spin forever. Host crashes, SSD short reads and generation conflicts abort/replay from previous committed frontier.

### Prefill borrow scratch and restore cache

```text
ResourceArbiter reserves mandatory KV+state+model
Choose prefill chunk=1024/2048 if legal
Evict cold *optional* expert slots after readers quiesce
Borrow scratch lease from freed pool segments
Large GEMM prefill + async next expert stage
GPU complete -> scratch lease released
Optional prewarm decode experts if net benefit
State commit and ready-for-decode event
```

Timepoints and buffer lease checks eliminate false dependencies and prevent accidental overlap of active expert memory.

### Distributed

```text
rank(s) handshake model + semantic/schema/quant + frontier epoch
reserve / remote send / allreduce provisional results
all ranks acknowledge matched completion
commit same accepted frontier or fail-closed and rebuild
```

CIR can describe collectives, but a distributed protocol must additionally ensure authenticated peer/epoch and crash-consistent semantics.

## 7. Optimization opportunities created by CIR

1. **Eliminate redundant H2D/D2H copies** only if two resources share validated physical alias and numeric layout; UMA mapped RAM can be zero-copy but bandwidth isn't free.
2. **Overlap independent streams/IO** by timepoint analysis, eliminating accidental host fences; do not assume two HIP streams always achieve concurrent execution.
3. **Alias scratch** when lifetimes disjoint and state effects permit, otherwise reject; inspect peak allocations.
4. **Fuse or schedule all-important kernels** with registers/LDS and occupancy constraints from KIR, not mere launch-count minimization.
5. **Prefer graph islands** for fully-resident decode, and explicit graph breaks for IO/dynamic external state; choose eager path when capture slower or unsafe.
6. **Optimize per-runtime mode**: in serving-aware kernel-tuning research 2026, eager vs graph alters the winning tactic; CIR's capture/eager flag participates in KernelPlan key.
7. **Per-token partial fallback** if legal state format conversion exists; otherwise defer until canonical frontier with accurate cost estimates.

## 8. Maintainability constraints

- A small command enum/versioned typed effect registry; **do not implement an entire general-purpose external IR or stream compiler**.
- Descriptive debug dump and validation before execution, but no per-token AST walks; compile to direct command buffers/call sites.
- One device/backend module owns stream/event APIs; kernel providers return validated handles but never allocate persistent cache outside ResourceArbiter.
- One liveness/alias verifier; future NPU, FPGA, remote worker or alternative storage path adds resource kinds and commands, not model-core changes.
- Avoid op names like `HipMemCopy` or `NvmeRead` in SIR; CIR knows hardware, SIR does not.
- Must preserve Rust-only user source policy: CIR is Rust data; optional external compiled device drivers are runtime dependencies, not hidden model execution.
- If CIR abstraction adds measurable steady per-token overhead, precompile static command arrays or inline device graph launch and confirm via benchmark.

## 9. Proof and exit gates

- Fake dual-device+IO executor implements CIR ops with simulated timepoint, deterministic failures and multiple virtual RAM tiers; 1e6 random event schedules if feasible.
- Catch intentionally invalid alias, stale lease, early release, absent expert, duplicated completion and partial graph capture.
- Compare eager vs captured region execution using same model/quant/state and measured whole response, not just GPU timer.
- Prove CPU reference SIR ↔ lowered RIR+CIR output and each mutable state version match for GDN/KDA/indexer rollback.
- Prove `CIR` isn't imposing per-token interpretation/serialization cost by output assembly/disassembly or CPU perf trace.
- Demonstrate adding a new storage class or device without edits to SIR model definitions, model-specific quant handling, HTTP and session transaction code.
- Keep latest observable quality/latency/allocations with one click introspection plan dump and per-request ModeReceipt.

## 10. Open implementation details

Precise command serialization, event/timepoint ID encoding, capture support by backend, shape/batch variants, executor code generation vs indexed function pointers, IO worker and event ring technology, plan normalization/canonical hash, and move-graph optimization search budget. These are P00–P08 proof decisions; do not freeze before the first working model and physical GPU tests.

## 2026-10-12 integration — XDNA2/iGPU cooperative projection scheduling

The source-verified [Gufo fork NPU pipeline](../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md) supplies a real use case for CIR's async resource/timepoint semantics beyond GPU↔SSD and dGPU↔iGPU: **two distinct accelerators on one SoC execute disjoint output-row segments of the same projection** and join only at their first mathematical dependency. The model's SIR `Projection` remains unchanged; RIR forms validated row-slice regions, KIR chooses an XDNA2-compatible block-float GEMM, CIR compiles `GpuEncodeInput → PublishXrtBo → NpuDispatch` concurrently with `GpuProjectionRows` and `GpuIndependentWork` → `Wait/AcquireNpuOutput` → `Join` → state commit.

**CIR command extensions (proposed):** `PackForAccelerator`, `ImportMappedBuffer`, `ExternalRuntimeSubmit`, `ExternalRuntimeCompletion`, `WaitAtFirstUse`, `AssembleProjectionSlices`, and `QuarantineExternalDma` with target-neutral `AcceleratorKind::Npu`. These operations reuse the same timepoints/lease generation/alias checker and do not create a second inference scheduler. XRT host callbacks and BO sync are **not interchangeable with HIP stream events**; bridge cross-API memory visibility and failure atomically. If the NPU has a long-lived hardware context (the Gufo fork does), represent kernel/program slot ownership and bounded in-flight depth as resource reservations.

**Fallback:** absent NPU/driver/program, unqualified SoC safety, unsupported M/K/N or quant, insufficient pinned RAM or strict reference precision means the entire independent projection row group is assigned back to a verified GPU/CPU KIR candidate **before job submission**. A failed *midflight* XRT DMA or stale completion invalidates provisional rows, holds quarantined memory until safe, and can only replay from committed model state. It cannot blend partial NPU data with valid GPU data and declare a correct output.

**Topology and power as resources:** GPU/NPU can share physical LPDDR and a fabric/power transition domain. CIR knows `MemoryTier::MappedHost` and external DMA registration separately from GPU GTT; `ResourceArbiter` accounts pinned A/B/C and prepacked weights; `PowerFabricDomain` tracks whether concurrent operation is hardware-qualified. Software event correctness does not prevent an upstream SoC hard hang; **NPU opt-in only until qualified, and no automatic privileged clock/firmware changes**.

**Performance experiment:** on real `gfx1151+XDNA2`, instrument GPU-only held-clock baseline versus same-clock GPU+NPU with queued-but-unused and fully used NPU variants, measuring actual first-use critical path, XRT queue/cache sync, CPU callback blocking, memory contention, full logits and safety. Incorporate the fork's fixed 4096×2560×1024 BF8 tile as one candidate without hard-coding it as a universal NPU shape. See [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

## v4 refinement — command readiness, not semantic commit

CIR is a **compiled physical resource/timepoint view within ExecutableBundle**, not a fourth enduring serialized IR or a generic workflow engine. `Acquire/Copy/Import/Dispatch/Wait/Join/Release` and `PublishResidency` manage events/leases/IO. The former `CommitState` is now **`ReadyForPublication`**: it certifies all required candidate buffers/host/GPU/IO work completed, but **does not publish a new TaskSession state root**. TaskCoordinator must independently prepare/validate all StateDomains and publish one root. A device-facing command can remain in a graph after this event; memory isn't freed until all physical readers quiesce.

Preserve dynamic shape guard, generation-tagged parameter/row slots and stable pointer-table identity for captured device loops. A host callback submit is not automatically a ready device event and an XRT BO alias doesn't guarantee GPU/NPU coherent load; bridge availability at first actual consumer. The compiled worker hot path submits only direct code/graph handles, not a CIR AST per token.

**Fail-fast tests:** stale peer/IO completion, partial GPU writes, detached client and two provisional steps must never mutate already committed visible state or release live DMA buffers.

## U-048 — banked stages, terminal notification and retained-plan lifetime

The [banked pipeline/replay refinement](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) uses existing closed regions,
Alias/Subspan, materialization, Dispatch/Wait and ReadyForPublication. Prompt-bank
scheduling is distinct from same-op output-row splitting. Bind A→B edges at each
state domain's actual owner phase; preserve live-across-switch values and snapshot
cuts. A terminal failure signal may unblock a queue but is neither successful
tensor readiness nor DMA retirement; poisoned private results cannot publish.
Captured/retained addresses and loaded code stay owned for an idle prepared plan
or that plan is invalidated before eviction/rebind; outstanding work additionally
requires real retirement. Compile these checks into direct executable commands;
no extra persistent command IR, per-layer host fence or inference scheduler.

````

## Source: .agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md

Original UTF-8 bytes: 23822; SHA256 `8b5aaf34ae213f24fdbf25f784b4071239424367f63bddc5714ce36005fd30ff`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — v4.1 typed composition and region execution contracts

**15 October 2026 — ZERO-ADDITIONAL-ENGINE STORAGE EXTENSION:** Model/SourceTensor semantics remain invariant as BackingTier changes among verified GPU/UMA/host/SSD; existing CIR Read/Transfer/Ready and physical ResourceArbiter/Lease own materialization. Source tensor identity, codec, shape and numeric plan are independent of device pointer, page cache state or file offset. Use one compiled direct-ready fast path and a bounded miss-capable asynchronous path, **not** a VirtualTensor trait call per multiply or a new SSD inference runtime. Encoded-but-unexecuted GPU graph reads pin slots until completion (DS4 #384). See [full source-neutral contract](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md).


**15 Oct extension-minimality decision proposal:** [quant](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md), [source model/DraftSource](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md), and [gfx kernel](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) extend existing source adapter + TensorEncoding + verified auxiliary ModelAssembly + TaskDriver + region candidate. Do **not** add EXL3, MiMo, AMD or DFlash inference engines/allocators. Optional residual-activation steering is a **behavior-changing, source-versioned semantic model variant**, independently quality/safety reviewed and OFF by default; it is never a numerical-preserving physical accelerator, automatically loaded from untrusted weights or silently used as a default.


**2026-10-12 · authoritative illustrative Rust interface design · NOT IMPLEMENTED.** These are *minimal capability sketches*, not code to paste verbatim or a public ABI freeze. The reference architecture is `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`; complete state/worker and task/compiler protocols in `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md` and `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`. Rejected v3 sketches and their rationale are summarized in the [critical review](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md).

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

````

## Source: .agents/docs/COST_MODEL_AND_AUTOTUNING.md

Original UTF-8 bytes: 24473; SHA256 `f0408679f82134ff93c6c1daa4950138c9813bfc9a9fc050923861eef7e6561b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Joint cost model, online adaptation and compiler autotuning

**15 Oct 2026 — explicit joint SLO-goodput optimization:** [Source-verified candidate contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) requires a bounded Pareto table keyed by exact model/derived quant/ISA/shape/numeric mode and workload. Joint choices include batch compatibility and physical row packing, chunked prefill, MTP/DFlash acceptance, unique MoE expert bytes, GPU/UMA/RAM/SSD resident misses and DMA, KV/recurrent restore, kernel variant, peer/P-D placement, latency/fairness/energy and physical headroom. Select only prequalified legal candidates at safe boundaries; unknown shapes use safe generic path, not unbounded runtime autotune. Maximize valid **committed** task output meeting QoS under admitted arrival load, with rejected/timed-out requests and fairness counted, not raw proposed tokens/sec. Persist negative/noisy candidate evidence; no model-string heuristics. SERVOPT-20..25/33..36 cover optimizer correctness and measured comparisons.


**15 October 2026 — COSTED SSD/RESIDENT PLACEMENT:** A tensor SourceId is storage-tier independent, but hot path cost is not: direct GPU resident hit ≈ pointer/index + kernel; SSD miss adds IO queue/seek/length/BW, host staging, source verification, real H2D and eviction/replay, minus **proven** overlap along CIR critical path. [Full storage-tier contract](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) requires cost-aware admission before model load: mandatory dense/state/OS reserve first, remaining GPU/host **global bytes** for expert/row caches, branch by prevalidated resident or miss-capable compiled plan, no per-token global source lookup. Prefill reads can be coalesced/overlapped by routed layer; decode latency is random-miss critical; MTP draft depth and expert union must include storage miss costs. SSD size/capacity is not an interactive QoS guarantee. [Acceptance](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md).


**15 Oct specialization precision addendum:** [gfx1151/UMA cost and kernel layout](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) and [draft/MoE row cost](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) require cost of actual **padded expert-row chunks, unique experts, source trellis decode K, draft verification rows, recurrent state and physical memory**, not merely batch size, bitcount or estimated FLOPs. Source observation of 15 vs 18 verifier rows at a 16-row MoE group is a measured *candidate* cliff, never a universal constant. Restrict online selection to tested shape/driver/quant/ISA/numeric candidates; unseen image sizes get a bounded safe path, not unbounded first-use solver search. Prior-art negative CPU-offload, graph, smaller-bit and drafter-width results are retained as experimental evidence, not defaults.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**Novel research claim to test:** optimizing request batching, weight residency and speculation together may yield better system throughput/latency than tuning each independently. This is a hypothesis until experiments prove it. Keep a robust baseline schedule that avoids added complexity.

## 1. Objectives and constraints

For a selected model/hardware/workload, choose plan `p`, resource placement `m`, batch grouping `b`, speculative strategy `s` to minimize user-specific latency/energy or maximize throughput subject to:
- correct exact semantic and sampling contract;
- memory/addressability reservations, non-overlap of buffer leases, OS headroom;
- session state dependency/causal ordering;
- fairness, token budget and optional P95 SLO;
- viable device kernel, graph capture and quant format capabilities.

```text
maximize  useful_committed_output_tokens / wall_time
subject to numerics, physical-memory, latency-fairness, hardware constraints
```

For interactive mode optimize TTFT and P95 intertoken, not merely aggregate t/s. For server mode prefer total committed tokens/s with bounded latency. For memory-safe mode prefer capacity and stable operation while explicitly reporting performance sacrifice. The optimizer cannot switch to a lower precision checkpoint unless separately requested.

## 2. Critical path, not naive sum

True latency is longest dependent path through compute, transfer, disk and synchronization. If host starts reading expert weights for future layer while GPU executes current layer, some costs overlap. Model:
```text
nodes = {
  host_tokenize, prefix_load, io_read_expert, dma_upload,
  quant_prepare, gdn_step, attention_select, attention_eval,
  router, expert_gemv, shared_expert, reduce,
  lm_head, sample, speculative_rollback, stream_emit
}
edges = data/state dependencies, stream events, slot leases, host/IO readiness
cost = measured service time by hardware/model/shape/context/cache class
critical_path = longest path with contention/queue constraints
```

Approximate with calibrated per-op counters and overlap groups; exhaustive global optimization is likely intractable and shouldn't block P08 simple inference.

## 3. Cost model inputs

**Static:** model graph and tensor quant/layout, number of layers/heads/expert top-k, mandatory buffers, device arch/instruction throughput and measured memory bw, compiler/driver, network topology, per-kernel launch cost.

**Runtime:** new prompt tokens, existing context depth, active sessions, accepted draft history, router miss distribution, unique expert count, prefix hit probability, available physical RAM, scratch, GPU queues, disk read latency, power/frequency and workload priority.

**External pressure:** memory use by OS/other apps, SSD cache warmth, remote worker availability. A static plan cannot assume those stay constant.

## 4. Separate planning timescales

**Install/compile time:** load semantic ops, choose legal quant packing, kernel algorithm family, snapshot schema, graph segmentation and candidate shapes. Store validated candidate registry, not a plan for every possible context count.

**Model-load time:** device/driver probe, memory-fit and resident/disk tier policy, compile/load a bounded set of prefill/decode/verify variants and optional CUDA/HIP captures. Tune representative shapes if user explicitly enables startup tuning; cache profile keyed by versions.

**Request admission:** reserve mandatory state/scratch, evaluate prefix state, choose initial chunk size and queue placement, estimate TTFT/P95; reject/limit if resources unavailable.

**Scheduler cycles:** choose from existing candidates based on active rows, context depth, resource leases and learned cost; adapt speculation depth and expert cache *gradually*, without JIT compile or full model repack per token.

**Long-term background profiling:** only opt-in or low-priority safe windows; never compete with interactive requests or share dangerous memory allocations.

## 5. Kernel tuning database

Suggested key:
`(semantic_region_hash, model_shape, quant_id, kernel_source_version, compiler_version, backend_driver_version, device_arch, batch_band, context_band, numerical_mode, graph_capture_mode)`.

Stored evidence:
`candidate_impl, tested shapes, numeric pass IDs, median and confidence, memory/scratch, occupancy/ISA metadata, observed full-model speed, rejected alternatives, invalidation policy`.

On different driver/compiler or quant change, invalidate affected cells; never assume an old binary's codegen compatible. The database should support repeatable command-line export/import of profiles with source provenance.

## 6. Whole-model fusion decision

A fused region might save 20 launches and temporary bytes but add register spills that lower occupancy by 30%. Use:
- microkernel timing + measured bytes and compute;
- model-region timing including prior/next transfers;
- end-to-end prompt/AR/verify throughput and TTFT/P95;
- numerical correctness/quality under selected policy;
- memory peak and graph capture complexity.

Default to simplest correct region absent repeatable win.

## 7. MoE batch × cache co-planning

State includes unique experts, group token counts, cache hit count, predicted read-time and resident pool slots. Scheduler choices might be:
- dispatch immediately with 1 row and resident experts;
- wait briefly for compatible rows to share experts/launches (limited fairness);
- reduce batch to avoid scratch evicting critical expert cache;
- issue prefetch hints for likely upcoming experts while previous layer runs;
- choose CPU expert lane vs PCIe fetch when measured favorable;
- use larger prefill quantum to improve GEMM efficiency if no active decode starvation.

The system should estimate **delay to gain**: waiting for more rows may worsen TTFT more than weight reuse saves. Grouping is best considered at GPU router boundary to avoid host round-trip.

Example objective (illustrative, not final):
```text
score(batch) = expected_gpu_work_ms
             + expected_nonoverlapped_expert_miss_ms
             + queue_delay_penalty
             + memory_pressure_penalty
             - verified_weight_reuse_credit
```
Never double-count overlap; recalibrate from actual counters.

## 8. Speculation cost-aware policy

Beyond acceptance fraction `a`, use:
- proposal overhead/time and draft resident RAM;
- target verifier step time as function of row count and routed expert overlap;
- snapshot creation/restore cost and GDN/KDA/indexer size;
- correction sample and tool boundary rewind probability;
- draft-quality dependent acceptance distribution, not just mean.

Select `none / prompt_lookup / mtp_depth_n / DSpark` by predicted committed output tokens per critical-path ms with hysteresis. Stable greedy reference mode provides fixed policy for reproducible tests. Low-acceptance bypass must preserve target distribution and sampler state, not change output semantics.

## 9. Memory-pressure controller

Every adjustment requires a ResourceArbiter-approved reservation and safe device/IO lease state. On pressure:
1. stop accepting new requests or decrease token budget;
2. release expendable scratch/old caches;
3. demote cold frozen prefix snapshots or optional expert slots;
4. shrink batch/prefill chunk within valid model constraints;
5. if mandatory state cannot fit, reject new work with clear explanation.

Never solve overcommit by silently switching quant, discarding active state or relying on Linux OOM killer. Avoid frequent expand/shrink cycles that thrash page mappings on `gfx1151`; impose minimum dwell/hysteresis.

## 10. Data collection and causality

Profiler emits state/event timeline with plan hash, kernel ID/shape, allocation generation, route expert counts, IO miss events, GPU occupancy and queue latency. Requests may be sampled/redacted for performance telemetry, not actual prompts. Evaluate correlation versus causality by controlled A/B and identical request histories.

Kernel instrumentation overhead should be measured. Use non-instrumented end-to-end benchmarks as primary speed proof; separate instrumented diagnostic to attribute lost time.

## 11. Autotuner validation gates

Candidate cannot become auto-selected unless:
- op numerical parity and model quality gates pass;
- cache/snapshot/failure/cancel state tests pass;
- physical memory and no leaked allocation regression;
- repeated paired benchmarks show real gain for target work regime;
- no major p95/fairness regression on mixed workloads;
- fallback still works when candidate missing;
- plan cache version key correctly invalidates compiled binary.

When no credible benefit, keep current simpler plan and retain failed experiment only as research evidence.

## 12. Unknowns to test

Whether a global joint optimizer outperforms a few human-tuned simple heuristics; how often route overlap is predictable; speculation acceptance stationarity across reasoning/tool passages; cost-model refresh overhead and overfitting; kernel compiler variance; how much private recurrent state affects achievable batching; impacts of CPU/GPU LPDDR contention on UMA. No assumption these are solved by semantic IR alone.

## 2026-10-10 integrated v2 addendum — critical-path and serving-aware planning

**Measured engineering principles:** asynchronous resource/timepoint and allocation scheduling provide liveness/copy-overlap opportunities; serving-aware autotuning must distinguish eager from graph-captured execution because the fastest isolated GPU microkernel is not necessarily fastest in serving. Bounded offline kernel schedule search with persistent device/shape profiles provides adaptation; a published graph-rewrite Qwen3 MoE example warns that unconstrained rewrite systems can spend minutes without model benefit.

**Updated cost layers:** `KernelPlan` contains validated per-region candidate costs; the pure HardwarePolicy computes startup fit/residency, CPU/GPU/SSD topology and safe context/concurrency; CIR estimates dependency critical path with overlapped IO/compute/transfer; EngineCore's StepPlan chooses among existing variants under dynamic fairness, memory, queue and model state. Do **not** aggregate these as one mutable global plan or JIT optimize in every token loop.

Objective = maximize *committed tokens* per wall second or meet P95/TTFT/token-J objective **subject to** numerical fidelity, state legality, OS-safe RAM, request priority and verified kernel capabilities. Gather bytes/operation, expert group reuse, lookup/cache hit rates, transfer latency, device queue overlap, register occupancy and per-state snapshot cost. A theoretical bandwidth roofline or few-percent microbench alone is not a product target.

Candidate optimization hierarchy:
1. CPU/reference correct and stable;
2. measured kernel/GEMV/GEMM and fused region;
3. async command/resource scheduling and graph capture where legal;
4. token-budget scheduling and prefix cache;
5. expert-aware batch/residency and speculation as *opt-in experiments*;
6. more ambitious online policy only when shadow-mode experiments exceed simpler cost heuristic.

Policy hysteresis must avoid expert-cache thrash and compiled plan churn. `KernelPlan` cache key includes semantic region, state schema, source artifact/quant, compiler/kernel/backend/driver, target ISA, shape bucket, numeric contract, **captured-versus-eager** and memory topology where relevant. A new device/toolchain upgrade invalidates affected binaries and test verdicts, not model semantics.

**Falsifiability:** contrast offline best vs online policy at C1/2/4/8/16/32, uniform/skewed expert routing, MTP high/low acceptance, fixed 4K vs 128K context, SSD hit/miss and thermally stable power. Track p95 and UI TTFT, not just throughput. Reject joint optimizer if no repeatable whole-workload gain; simplest correct policy remains default.


## 2026-10-12 integration — costed multi-accelerator projection partition

Add a **compute-owner assignment** decision independent of weight/state residency. For an eligible projection `W` and prompt tokens, RIR can consider GPU-only, NPU row slices+GPU rest or (separately) dGPU+iGPU expert split. CIR captures host pack, NPU queue/compute/XRT sync, GPU concurrent independent work, read at first-use, output conversion and join. Estimate makespan by resource/timepoint **critical path**, not sum of two GEMM times or theoretical NPU peak TOPS; shared LPDDR bandwidth and fabric power clock coupling can degrade GPU work when NPU active.

Candidate selection is **phase/shape aware**: Gufo's `4096×2560×1024` block-float NPU kernel is only feasible for compatible multiples and prefill workloads, not arbitrary decode1 or novel model geometry. Inputs include quant/source format, NPU conversion overhead and shared pinned weight footprint, XRT call/context/warmup, clock/thermal/performance state, number of in-flight runs, NPU power and system safety qualification. The best split can be zero; reference or unqualified safety mode always chooses GPU-only.

**Measurement methodology:** matched UD-IQ4_XS 16K (GPU-only reported 1,698, GPU+NPU 1,821 tok/s) then 32K/40K cold served workload with **the same held fabric-clock and power state in both A and B**. Separate the effect of fabric clock pinning from NPU compute, and report repeated ABBA, raw request latency, P95, pinned RAM, watts/tokens/J, long soak and independent quality. 30 matching greedy prompts is a smoke test, not a complete logits/sampling-quality gate. Qualified candidate entry keyed by `(CPU/APU/NPU architecture, XRT+ROCm+firmware/BIOS, NPU program hash, shape, quant, numerical mode, power/fabric profile)`; automatically disable when key mismatches. **No online NPU experiment** on an interactive machine; physical power stability trials are opt-in and supervised.

Task-stage scheduler can separately price NPU GEMM work in ASR/TTS/vision encoders, but only after target op implementation and quality gates; otherwise the generic GPU/CPU stage remains intact.

## v4 refinement — three selection timescales and bounded cost

**Compile/load time:** validate SIR source truth, compile generic baseline modes and a small shape-band collection keyed by model/operator/quant/device/numeric mode; optional high-value candidate search **offline/bounded**. **Step runtime:** select only prevalidated guarded variants, reserving resources and parameter slots at session/batch boundary; no per-token optimizer graph traversal, exotic JIT or global allocator lock. **Physical telemetry:** use P95, real task completion and shared power/IO/bandwidth (not just synthetic FLOPs) to promote candidates, with hysteresis/disable path on regression.

ShapeGuard includes fixed-rank dynamic bounds, ragged row offsets, stride/quant alignment, selected regime (prefill, decode, verify, encoder/denoise/vocoder) and driver/profile qualification. `KernelPlan` is the candidate evidence table, not a new mutable execution engine. Bundle cache LRU is bounded; cache keys contain stable semantic/shape/compiler/driver/numeric identifiers but **exclude session IDs/active expert slot generations**, preventing cache explosion.

A generic reference/optimized plan must be available on cold unsupported shapes. Some fast device-native `StepProgram` modes can eliminate CPU sampled-token readback; where exact sampling/grammar forbids this, report the host-loop mode without claiming universal GPU-native execution.

## v4.1 — compile granularity, program warmup and topology cost

A universal optimizer must choose **region cut size and executable family**. `NativeKernel` may favor low kernel launch latency and direct quant GEMV; an AOT graph may prefer whole attention/layer/model stage to amortize graph compilation, graph parameter slots and device activation memory; static SRAM dataflow may prefer an entire placed/streamed model program and long residency. **Do not host-dispatch every op of a graph-only backend**, and do not claim a full static program has zero compilation or weight stream cost. Model load planning now includes program compilation/validation/upload/warmup, resident program SRAM/HBM, launch/collective cadence, shape bucket/cache invalidation, interchip topology and multi-session sharing.

Cost objective remains full **quality-constrained useful output**: single-request cold TTFT and warm TTFT/P95, throughput vs concurrency, program cache memory, device/host copy, compute/fabric bandwidth, energy, codegen latency, streaming weights and device safety. A local coherent memory SoC can be memory bandwidth bound at AR batch1, whereas HBM Blackwell/CDNA rack may be math/collective/latency bound. A TPU-like static graph may recompile on a new input shape: guard bounded dynamism, profile shape families and use safe generic fallback instead of unbounded per-sequence AOT.

Graph/dataflow compiler versions, device firmware/toolchain, model semantic/quant, StateBridge, shape program and topology are separate evidence/cache keys; session/request IDs are runtime data and must never generate unique executable plans. Source-specific hardware claims are research-only. See [backend contract](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## v4.2 — measured topology and independent state/cache placement cost

Placement solver considers independent **compute, weights, live state and immutable cache** locations, network TransferPlan, source numeric/semantics and task mode. Cost is the **longest dependency path** of region compute, remote activation/state transfers and collectives plus scheduling/queue, tail RTT/relay jitter, staging/crypto, program/cold model loading, cache miss/recompute, memory pressure, energy and failure recovery. A placement that fits *more* model capacity may deliver *slower* C1 decode; report capability vs latency/throughput gains separately.

**Per-topology strategy:** contiguous layer+KV colocated can minimize per-token network boundaries; attention remote with KV reduces full remote cache reads to compact query/result but adds per-layer RTT; remote prefix hydrate costs full state once, then saves future network; tensor/expert/sequence shards need fast links and batch or long context to amortize collectives. Full remote per-token KV reads over a relayed tunnel normally fail cost/latency/SLO. A P/D split may protect ITL but not necessarily increase throughput due to full hybrid-state migration. Hybrid attention can defeat a one-size-fits-all attention/FFN split; profile per layer group.

`TopologyEpoch` records **measured directional throughput by payload size, RTT/P95/jitter, direct/relay, effective MTU/loss/crypto cost, permission/egress** and physical node allocation. Offline placement search is bounded; online only eligible prequalified plans with hysteresis at legal root frontiers. Prevent router/chattering placement thrash; cache-directory hint misses re-evaluate quickly and never yield wrong state. Track TTFT/P95, committed output, cache transfer bytes, network wait ratio and network energy. [Placement contract](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md).

## v4.3 — model residency, replica and cache-affinity economics

Add coarse model-level admission decisions to the existing topology/phase optimizer: source model weight bytes, derived packed formats, adapter/draft/encoder sidecars, device program compilation/load/warm, per-session KV/SSM state and memory/energy, TTL/pin/cold load time, cache reuse/prefix affinity, SLO queue delay and peer-placement cost. **Registered catalog size is not concurrent physical model capacity.** Prewarm/idle eviction, when to create another equivalent replica and when to sharded-deploy one model are different choices.

One exact model request cannot be satisfied by a smaller/different quant or family merely because it's warm. If a user explicitly selected a model-choice alias, compare authorized candidates' declared quality/task/security conditions and report resolved actual source. Stateful session sticky affinity can dominate lowest raw queue; migrating full hybrid state to idle replica may cost more than waiting. Anti-thrash/hysteresis prevents alternating massive model loads.

Benchmark direct in-process `run` vs model router HTTP same TaskDriver under fixed model/quant/sampling for **no hot-path router overhead**, then 1–8 registered/resident models, giant/small mixed loads, preload/pins and C1/C8 latency/fairness. Record physical peak memory, model unload/transfer, energy, TTFT/p95 and throughput rather than headline aggregate across dissimilar checkpoints. Existing distributed v4.2 placement planner remains source of truth for remote replicas. [Router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md).

````

## Source: .agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md

Original UTF-8 bytes: 14326; SHA256 `bf2100d929ed6f2fd6b18981b13470f49dc3d6c17fd9162dfb125ceba69ca55e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — EXL3/QTIP and Source-Qualified Derived Quantization Contract

**2026-10-15 • Proposed design refinement, NOT a runnable codec or committed numerical equivalence claim.**
**Authority:** v4.3 SemanticProgram→ExecutableBundle; v5.1 ModelAssembly/SourceIdentity/TensorEncoding; source format and quantization are independent of model topology and hardware; no second inference engine.
**Source study:** published low-bit quantization pipelines, including an experimental per-tensor sensitivity optimizer and a source quantization format. Exact implementation is pinned to a source revision before any code begins. Performance numbers below are **upstream author measurements**, NOT Zerogravity results.

## A. Why this is not just add another QuantKind

EXL3 is an extremely-low-bit source quantization family based on QTIP concepts: structured transforms, codebook/trellis representations, source-trained calibration/error shaping and low-bit matrix multiplication. Source variants encode more than a bits_per_weight scalar, and on accelerator hardware their direct GEMV/GEMM execution path matters at least as much as compact storage. A decoder recognizing a packed byte sequence is not qualified to execute a full model; a source model compatible with GGUF Q4 is not automatically source-compatible with EXL3. **Never reinterpret packed codebook identifiers as ordinary Q2/Q4 or assume dequantization into FP16 is the only legal computation plan.**

### Storage, semantics, and physical layout must remain orthogonal

- SourceArtifact: original BF16/FP8 checkpoint with source hash, architecture, config, tokenizer, templates, optional vision/audio/MTP/n-gram assets.
- DerivedArtifact: a **different model-weight artifact** with its own source quantization/recipe/calibration/license and quality identity. Lossless packing of an already-quantized checkpoint is separate from a lossy derived model.
- SourceQuantCodec: EXL3/QTIP/trellis/codebook shape, bit-stream, grouping, scale/zero, sign/basis transforms and error-propagation constraints. Unsupported codebook/bit width is UnsupportedCodec, never unsafe byte reinterpretation.
- PackedDeviceLayout: native architecture/tile/coalescing/WMMA alignment, register/LDS plans; a reversible presentation of the **same derived mathematical weights** unless a separately verified arithmetic transform is declared.
- ActivationCodec: input activation pack/scale/rounding, optionally quantized INT8 decode kernel, with explicit source-relative numeric policy.
- ComputeCandidate: source-verified direct quant GEMV/GEMM or kernel-fused decode, fallback only if source compatible, accounting temporary buffers and dispatch.
- NumericContract: original model-vs-quantized checkpoint quality and exact/approved approximate backend kernel math are **two distinct proofs**.

Illustrative, not final ABI:

~~~rust
struct VerifiedQuantArtifact {
    original_artifact: ContentDigest,
    derived_artifact: ContentDigest,
    codec: RegisteredCodecAndVersion,       // EXL3 K/codebook family
    tensor_recipe: Digest,                  // complete per-tensor format map
    calibration: CalibrationProvenance,
    output_quality: QualityReceipt,         // original vs derived model
    processor_and_model: SemanticIdentity,
}
struct QuantTensorSpec {
    logical_shape: Shape, stored_shape: Shape,
    source_format: SourceCodecId,
    quant_group: ValidatedGroupGeometry,
    codebook: Option<ValidatedCodebookId>,
    basis: Option<VerifiedTransformId>,
    scale_encoding: ExactScaleEncoding,
    numerical_decode: RegisteredReductionContract,
}
struct PackedMatrixKernelKey {
    quant: QuantTensorSpec, device_isa: DeviceIsa,
    m_band: ShapeBand, k_band: ShapeBand, n_band: ShapeBand,
    codegen: CompilerDriverKey, precision: NumericContract,
    output_layout: PhysicalLayoutId,
}
~~~

Do not make EXL3 one hardcoded model family or a monolithic Exl3Model interpreter. Source semantics remain familiar Qwen/GLM/MiMo operators plus variant-specific recurrent/indexer/attention state; the physical codec can be replaced independently.

## B. Correctness and derived artifact identity

Check that every source tensor is accounted for exactly once:
- encoded bytes, expected block/codebook packing, codebook index bounds, endianness and alignment;
- logical axes vs padded GPU axes, special head and expert/tied-gate matrices, bias, per-channel/group scales, quantizer codebook and optional activation basis transform;
- all required inference sidecars, vision parts, MTP heads, lookup/PLE tables and template/processor hashes;
- corrupted and missing tensor/padding/codebook must fail **before** reserving physical GPU work;
- mixed-format models may have FP16/BF16 heads and codebook quantized experts; kernel selection is per tensor role and shape, not one pack-wide bit width;
- allow source-exact scalar dequant oracle for small golden vectors, but avoid mandatory full-model FP16 reconstruction that duplicates 100GiB packed weights in RAM.

An EXL3 pack copied from another library cannot silently assume its original model's tokenizer, attention variant, RoPE and lossless sample identity remain valid. Trust metadata only after cross-checking independently source-verified structure and numerical fixtures.

## C. Per-tensor calibration and sensitivity optimization

An experimental published pipeline goes beyond a uniform 2/3/4-bit recipe. Its workflow uses deployment-shaped self-generated model traces and per-tensor reference output sensitivity, with quantizer-error anchoring and quantizer-shaped noise perturbation; it then allocates bits within a size/quality budget, converts, and tests on separate held-out traces. **This is an optional offline developer pipeline**; it is not permitted to run hidden training/calibration inside an inference server.

Pipeline semantics:

1. Pin original checkpoint and authorized evaluation corpus plus source tokenizer/chat format. Use deployment-representative input **including reasoning/tool patterns** when permitted. Never store private user prompts by default.
2. Generate a **calibration** trace and an independent **held-out** evaluation trace. Record template/reasoning/temperature/RNG and prompt taxonomy. Do not measure quality on the same trace used for optimization.
3. Quantizer error anchor: per-tensor relative error distribution (not just universal quant noise). For EXL3 source variants, account for trellis/codebook and LDLQ error structure.
4. Source-weight tensor sensitivity: inject seeded **shaped** noise into unquantized tensor while keeping remaining unquantized model constant; measure downstream final-logit KLD. Do not claim ordinary isotropic noise is equivalent.
5. Build a constrained per-tensor bitrate solver under exact size plus kernel-group/fusion constraints. E.g. group gate/up projections only where a fused kernel truly requires matched coding, not as an assumed model law. Cost includes codec header, scale, packing, optional sidecar and compressed expert cache.
6. Emit a machine-verifiable recipe with every tensor listed, complete source digest, calibration/eval run IDs and codec version. Missing or duplicated tensors hard fail. Tuning intermediate files remain optional.
7. Convert deterministically where possible, verify exact tensor bytes/codebook decode and public model semantic shapes, then assess independent held-out NLL/KLD/logit differences, token accuracy and tool/long-context outcomes versus **original source**.
8. Independently qualify CPU and physical GPU on the **derived artifact**. A numerically exact kernel on a quantized target is not equivalent to an FP8/BF16 original unless independent quantization evaluation proves the chosen quality policy.
9. Export QualityReceipt and ModeReceipt, with warnings if calibration covers only a limited domain/language/context. No silent model replacement to fit memory.

Recommended offline optimizer inputs: tensor role, per-layer sensitivity, quantizer-error shape and cost, checkpoint topology, target size/quality cap, GPU codec/shape kernel cost, output heads/router/indexer/recurrent criticality and exact-source state. Optimizing KLD alone can choose an impractically slow codebook: use a **Pareto-aware output** of quality *and* measured decode/verify cost.

### Sources and realistic caution

A published experimental optimization guide labels its workflow **experimental and untested on sparse models**, despite positive demonstrations on dense models. Treat extrapolated tensor sensitivities, bitrate curves and source-domain transfer as hypotheses to reproduce, not universal safety guarantees for Qwen3.8/GLM/MiMo MoE. Quantization with its own calibration data is a new artifact, not an exact alternate KernelCandidate.

## D. Critical counterexample: lower bitrate may be slower

A published gfx1151 report found a 2.05 bpw Qwen pack ran around 25.7 generated tok/s versus 44.1 at 3.05 bpw in the corresponding tested MTP setting. The K2 trellis path tiled poorly, despite smaller disk weights and substantial draft acceptance. **Do not build an optimizer using only stored bytes.**

A legal candidate selection key includes per-tensor codec K and codebook, output-row/verify shape, quant-activation path, device ISA/compiler, physical pooling and image/prefill/AR task mode. Runtime cost considers packed decode operations, shared-memory bank conflicts, registers, launch cost, expert union and scratch as well as bytes.

Recommended profiler: codebook decode cycles/byte, index loads and branch cost, source dequant oracle discrepancy, full quant GEMV/GEMM latencies at M=1,2,3,4,8,16,32,64, prefill 128/512/2K/8K, route entropy/unique experts and actual committed tokens/sec with the target model.

## E. Correctness modes and regression policy

- **Source-derived pack quality:** original FP8/BF16 vs lossy EXL3 model. Report top1, KLD/JS or task metrics, calibrated confidence and original model state where comparable.
- **Kernel arithmetic quality:** a CPU/fast reference for the **same quant pack** vs accelerated candidate (include source exact signed zero/overflow/rounding, unique seeded stochastic distribution).
- **Speculative losslessness:** drafted+target-verified outputs vs target **using the same derived pack**. A quant pack with 93.8% top1 vs original FP8 can still have exact speculative output relative to that pack.
- **Batch concurrency determinism:** different legal row reductions may differ numerically even with 16bit KV; source-approved deterministic mode should explicitly control/fail those cases and never market unconditional token equality across all batch sizes.
- **Compilation/storage correctness:** archive integrity, codebook decoding, different context and source sample inputs; only recognize committed prepared bundles and semantic effects.

A reduction changed to escape FP16 overflow without source proof is not performance equivalent merely because PPL is similar. Follow the project strict numerical and StateRoot gates for currently qualified Qwen paths.

## F. Acceptance matrix for P03/P05/P07/P20/P22

| ID | Gate | Reject if |
|---|---|---|
| QTIP-01 | Parse/codebook/trellis source fixtures for each declared EXL3 K, block and scale; independent scalar source oracle | unsupported bit width or codebook yields plausible tensor |
| QTIP-02 | per-tensor mixed precision and special head/router/vision/sidecars; validate all tensor lists | model loads with missing or wrong auxiliary component |
| QTIP-03 | lossless repack roundtrips to same quant weights, preserving padded logical shape | weight identity conflated with new quantization |
| QTIP-04 | malformed codebook indexes/scales, huge bytes, overlapping shards and wrong alignment rejected before allocation | OOM/out-of-bounds/uncertain GPU work for host data error |
| QTIP-05 | CPU scalar quant GEMV output against source conversion reference for M=1 and verifier rows | only perplexity validates codec |
| QTIP-06 | ROCm gfx1151, CUDA and other permitted backend fast-kernel relative to same quant target on original source tensors | false exact numerical mode |
| QTIP-07 | 2.05 vs 3.05 bpw (when source/licensing allow): report both size, quality, GEMV and end-to-end MTP | equating smaller file with faster decode |
| QTIP-08 | calibrate on representative allowed trace, evaluate disjoint held-out trace including code/tools/multilingual | calibration/evaluation leakage |
| QTIP-09 | per-tensor sensitivity+LDLQ-shaped perturbation evidence vs unshaped control; sparse MoE remains experiment | speculative sensitivity promoted as fact |
| QTIP-10 | source model original vs derived pack quality and exactly targeted speculative output both reported separately | derived model labeled original quality |
| QTIP-11 | cancel host quant/repack during artifact staging: no half-install or mutated source hashes | partially written pack recognized as valid |
| QTIP-12 | feature disabled/missing GPU codec returns validated slow/reference alternative or honest Unsupported, never silent different quant | model-name-based unsafe fallback |
| QTIP-13 | actual RAM headroom and file page-cache accounting while loading >80GiB sharded quant pack; no OS pressure hang | GPU MemFree treated as physical available |
| QTIP-14 | one new codec can be added without editing TaskCoordinator/Router/public API parsing | added separate EXL3 inference engine |

## G. Implementation seam and phase order

- **P03**: independent VerifiedQuantCodec parser/reference, derived-artifact manifest/calibration schema, error checks. Minimal source tests; no GPU.
- **P05/P07**: quant region guarded conversion/direct codebook GEMV/GEMM kernels on genuine CPU/GPU with compiler/device evidence.
- **P10/P12**: packed expert cache and source-file/OS page cache, no duplicate GPU reservations; derive expert-specific packed costs.
- **P20**: experimental calibration, sensitivity solver, KLD/quality/benchmark harness (optional/offline).
- **P22**: full model/derived pack, long-context, tool/quality and throughput pass before auto-selecting candidate.

**No new persistent IR or external C++/Python inference engine is required.** Model-derived conversion scripts, if authorized later, may be independent offline tools with full source audit. No user asked to implement them in this design step.

````

## Source: .agents/docs/DEVICE_KERNEL_AND_CODEGEN.md

Original UTF-8 bytes: 23101; SHA256 `7c3631a2b503b9fbd7e9cbb5d9b582e437f1601f66b35dc5e20b3ca973a25d56`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Device abstraction, Rust GPU kernels, code generation and backend specialization

**15 Oct exact gfx1151 specialization addendum:** [hardware candidate and physical safety contract](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) makes EXL3/QTIP packed WMMA GEMV, Wave32 register/fragment map, LDS bank-swizzle/padding, v_dot4 INT8 and SAD trellis, skinny GDN GEMM, grouped expert fusions, and exact first-image patch-merge convolution→GEMM **candidate implementations** only. Each requires model/source/derived-quant, m/n/k, ISA/driver/compiler, physical LDS scratch shared across all warp callers, accumulation/numerics and region event/lease guard. Neither padding=5 nor grouping=16 may enter SIR as source semantics. Use reference closed region or typed Unsupported when no verified candidate; do not silently enable CPU expert offload or graph capture from external performance claims.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**Objective:** support fully Rust-maintained inference logic and custom kernels with peak GPU performance where physical hardware/driver allows, while preserving portable canonical operators. CubeCL/CubeK is a leading *candidate*, not approved mandatory dependency; physical `gfx1151` feasibility gates selection.

## 1. Backend interface and invariant

A backend handles **devices, memory, kernels, stream/event lifetimes, graph capture and errors**. It is not responsible for model interpretation, scheduling, tokenizer, speculative acceptance or resource policy.

Conceptual traits:

```rust
trait DeviceBackend {
    fn capabilities(&self) -> Result<DeviceCapabilities>;
    fn allocate(&self, request: BufferRequest) -> Result<BufferLease>;
    fn copy_async(&self, cmd: CopyCommand, deps: &[DeviceEvent]) -> Result<DeviceEvent>;
    fn compile(&self, kernel: &KernelSpec, opts: &CompileOptions) -> Result<CompiledKernel>;
    fn launch(&self, kernel: &CompiledKernel, args: &LaunchArgs,
              deps: &[DeviceEvent]) -> Result<DeviceEvent>;
    fn capture(&self, region: &CaptureRegion) -> Result<ExecutableGraph>;
    fn wait(&self, event: &DeviceEvent, deadline: Deadline) -> Result<()>;
}
```

Exact async Rust ownership/Send/Sync/ffi types must be determined by implementation. A `DeviceEvent` includes device/stream/generation and must not accidentally call a blocking synchronize on the scheduler executor.

## 2. Capability model

Record vendor, chip/architecture target (`gfx1151`, `gfx1100`, `gfx1201`, `sm_89/90/100/120/121`, Apple GPU generations), waves/warps, max threads, local/LDS/shared memory, VGPR/SM registers, matrix instructions (WMMA/MFMA/FP4/INT8/BF16/FP16), subgroup/shuffle, FP8/FP4 support, atomics, graph support, peer access, RAM/VRAM/pinned mapping limits, transfer bandwidth and versioned driver/runtime.

**Capabilities are verified by runtime probes**, not solely static tables. Distinguish a reported feature, successful compilation and successful execution of a small parity kernel. Unsupported precision may use validated conversion without silently changing quantization and quality.

## 3. Rust kernel source design

The user asked for all Rust. The engine must not turn into a Rust wrapper around another C++ model server. Rust-written kernels should compile to device ISA through a Rust GPU DSL/backend or custom compiler path; normal host GPU runtimes may be accessed via bindings depending on final O-003 decision.

Candidates:
- CubeCL/CubeK for expression, tiling, matmul, autotuning, CUDA/HIP/Metal/Vulkan;
- dedicated `gfx1151` Rust kernel lowerer with architecture intrinsics/LLVM when CubeCL misses key instructions;
- Rust-written CPU SIMD reference/optimized kernels;
- vendor BLAS as an *optional execution candidate* if final all-Rust policy permits driver/library calls, without making model execution dependent on another engine.

Every kernel `KernelCandidate` advertises exact supported shapes/quantization/position/state/memory alignment/numerical behavior/capture and scratch. Plan only runs a kernel if all preconditions hold.

## 4. Output contract and numerical precision

A kernel must specify:
- input logical/stored dtype, block quant scale and packed layout, strides and pointer alignment;
- output dtype/accumulator precision and any reduction/fusion order constraints;
- required zeroed/preserved scratch and whether it writes beyond logical shape padding;
- valid model geometry and row-count ranges;
- state input version and output transition;
- async event and synchronization semantics;
- numerical parity oracle with tensor and full-logit tolerances;
- deterministic property under fixed device plan where relevant;
- expected critical-path performance regime and fallback equivalent.

A successful compile is never sufficient. Tests include bit patterns at min/max quant, block boundaries, ragged rows, masks, zero/invalid expert IDs, NaN/Inf policy and old output guards.

## 5. Kernel families and target-specific strategies

### Dense FFN and projection
Decode1/low rows: memory-bound GEMV/MMVQ, group output rows and minimize load traffic. Prefill: shape-tuned GEMM/WMMA, reuse weights in shared memory. Speculative verifier: 2–8 row MMQ/compact GEMM; use graph capture to reduce launches. Q8 or mixed Q4/Q5/IQ2 format decoding inside tile rather than full materialize where beneficial. Ensure padded K dimensions not mistaken for mathematical inputs.

### Routed MoE
Separate GPU router top-k and weight table lookup; handle scoring bias/sigmoid/softmax exactly. Build token→expert assignment map and expert→token compact worklist entirely on device where possible. For small rows, per-expert GEMV; for many rows, group by expert and use grouped GEMM. Shared expert may run concurrently with routed expert and/or disk prefetch if no ordering hazard. Scatter and reduce in numerical-safe order. Deterministic mode may use two-pass reduction rather than atomics; speed policy requires measured numeric drift/quality checks. Distinct expert dense/down quant paths can be selected.

### Recurrent GDN/KDA
One-token state update is latency and register dependent; chunk prefill is matrix-friendly. GDN and KDA use model-specific gate types (scalar/per-channel) and conv history. Special fused kernels must preserve state after each token required for speculative rollback. Prefill chunked scan has to pass independent recurrence oracles and multiple chunk boundaries. Snapshot tiles can be written to small side buffers only at required rollback cuts.

### Qwen Sparse Attention
GPU indexer score/selection with causal masks, block grouping/pooling, top-k and empty/masked cells. Avoid CPU fallback to top-k in critical decode path if it creates context-depth scaling/round trips. Bounded scratch and register allocation important at 128K. Selection, attention and numerical semantics must match model; proof cannot silently change selected budget.

### GLM sparse MLA + KDA
Compressed latent KV `c_kv` (NoPE where appropriate) versus expanded head KV must be explicit incompatible layout types; absorption saves KV bytes but changes projection shapes. DSA score top-k, query grouping, FP16 latent shared-memory tile boundaries, and KDA recurrent step need separate tuning. Documented GLM has 34 KDA + 11 sparse attention layers, mHC repeated. Optimize prefill and decode separately.

### DeepSeek compressed attention
128-token raw sliding KV, older 4:1 / 128:1 compressed rows depending on layer; eligible indexer layers select historical rows, other layers use compressed history directly. Matrix kernels may fuse gather+softmax/value, but cannot discard required pooling history. User-visible context extension performance is measured not inferred from claimed max context.

### Hyperconnections
mHC/Sinkhorn and gated residual operations have distinct formulas. Kernel needs full width, mixing/normalization behavior, snapshot legal boundaries and FP32-sensitive numeric policy. Fusion allowed only after full-model checks.

### Quantized LM head and sampling
Support wide vocabulary head with direct row tiling, transparent top-k/top-p/min-p and stop/penalty semantics; sampled MTP proposals may need different candidate selection from target's exact distribution, but verifier corrects it. Avoid unqualified approximate shortlist as target logits.

### N-gram/Engram
Hash and row lookup often better on CPU/IO with small staging; table may be enormous and not GPU-resident. Keep row IDs and history deterministic; async pread/read compaction can overlap GPU work but must carry content/position identity and errors. Treat lookups as data-dependent operations, not giant dense embedding matrix multiplications.

## 6. Strix Halo gfx1151

Plan for Ryzen AI Max+395/Radeon 8060S (RDNA 3.5) and unified LPDDR. Use wave32 and carefully tested rocWMMA shapes where supported, Q2/IQ2/Q4 vector dequant, Q8/F16 projections, GPU sparse selector. Cited DwarfStar tuned prefill using 32-head/80-key tiles, specific x64/y64 IQ2 MMQ and per-block Q2_K staging. These are research references; our actual optimal tile shapes and generated ISA must be measured.

**Physical memory:** ROCm GTT/hipMemGetInfo may exceed safely available RAM; kernel code cannot solve OS memory overcommit. Device planner must interact with ResourceArbiter. CPU and GPU share memory bandwidth, so simultaneous CPU expert compute can reduce overall speed on UMA. Compare with discrete GPU CPU expert lanes instead of blindly applying Maya strategy.

**Graph capture:** distinguish driver HIP capabilities from DwarfStar's current eager ROCm implementation. Prove capture+replay state pointer safety and event semantics, or use low-overhead eager streams if capture isn't beneficial.

## 7. CUDA / NVIDIA and other targets

CUDA architectures differ on native FP4, BF16/FP8, tensor core instruction layouts, asynchronous copies and shared memory. Blackwell FP4 can make specific MoE quant kernels superior where hardware supports native FP4; Ada/Hopper may need different Q8/FP16 alternatives. Multi-GPU model sharding and library kernels optional; avoid losing shape-specialization by using generic BLAS always.

Metal on unified Apple memory: Metal graph capture/tapes and simdgroup matrix kernels, portable state sessions. Apple UMA is not identical to AMD gfx1151; model-specific memory residency and codegen should remain backend-specific.

Vulkan: broad portability fallback with subgroup/precision feature probes; extreme performance only where actual kernel features exist, not promised as default. CPU: SIMD portable reference and optional tuned architecture-specific vector loops; keep correctness and CPU-only server functional even without GPU.

## 8. Kernel planning and autotuning

Compile only bounded shape families; select candidate based on measured whole operation and end-to-end plan. Keys include chip, vendor driver/runtime, kernel build, model/op geometry, tensor packing and quant, batch/decode/prefill/verify, context and numeric policy. Static heuristics provide safe default; auto-tune with explicit cost/time budget or offline profiles, not surprise time-consuming warmup.

For every candidate gather kernel time, memory bytes, register/LDS occupancy, launches, GPU clock and power; then run a complete real-model request. Reject optimizations whose isolated improvement does not translate or causes quality/latency regression. Keep publicly documented rejected experiment records.

## 9. Direct execution design

After plan selection, host prepares flat launch parameters with stable pointers/offsets and events. GPU graph replay, if used, sees device-resident control values for changing position/next tokens to avoid rebuilding graph each step. No string name searches, tensor trait-object dynamic dispatch or per-op memory allocation at every token. Host callbacks reserved for IO/cache misses, cancellation boundaries and scheduler cycles.

**Graph safety:** captured memory addresses stable for graph lifetime; pointer updates only where supported. All scratch buffers preleased; no host CPU waiting in stream callback; device-side wait kernels bounded and failure-aware. If cache miss detected, park safely or exit graph region to host-driven event, then resume with new validated slot table.

## 10. Error handling and tests

Device loss/timeout invalidates provisional states. Slow IO cannot lead to spin forever. Buffer generation prevents use-after-free/evict. Host/device races guarded by fences/events; cross-queue copies ordered explicitly. Test with simulator/stub first, then physical GPU, sanitizer under CUDA/HIP where possible and valgrind/Miri where suitable for host ownership. Ensure NixOS loader paths and GPU permissions are handled without privileged operations.

## 11. Need-to-measure technical unknowns

- Can CubeCL generate efficient `gfx1151` WMMA and quantized matmul code, at competitive occupancy?
- How to expose wave/lane permutations, shared-memory barriers and low-level half/bfloat operations entirely from Rust-authored kernels?
- Cost and availability of low-overhead HIP graph capture and dynamic control tables.
- FP4/FP8 kernel availability on multiple CUDA generations.
- Workload-accurate SIMD CPU expert throughput on UMA vs discrete.
- Whether a dedicated custom Rust GPU IR/lowering would be warranted over CubeCL for extreme paths.

Do not answer these by architectural confidence alone; close P01 with disassembly, numerical parity and physical-device evidence.

## 2026-10-10 integrated v2 addendum — Rust GPU provider landscape and portability

**No GPU compiler is a permanent engine ABI.** Evaluate portable Rust GPU tensor DSLs, a stable Rust CUDA tile-programming frontend and optional experimental Rust SIMT/PTX source compilation, each on actually supported hardware with pinned driver/compiler capability and numerical tests. For AMD `gfx1151`, compare CubeCL HIP generated ISA with a Rust-authored direct AMDGPU/LLVM target kernel if its WMMA/MFMA/LDS/tile capability is insufficient. Vulkan portability toolchains and Rust CPU SIMD remain fallback options; Metal is a separate hardware backend, not automatically equal performance.

Source: https://github.com/tracel-ai/cubecl . **Do not equate vendor GPU runtime FFI with user permission to import C++/HIP source kernels**. The historical research `zg-vendor` escape tier remains a **blocked policy choice** until explicit source-language/licensing decision; prioritize truly Rust-authored target-kernel escape paths.

Backend registry now uses `KernelCapability` with region semantic hash, state effects, supported numerical precision, quant/storage layout, shape regime, hardware features, scratch, C IR event/lease contract, capture/eager mode, source/compiler/driver revision and verified evidence IDs. One model can mix providers at safe region boundaries. Compilation may use provider-specific internals, but model SIR/EngineCore cannot import their native types. A `CompiledKernelHandle` is held by `ExecutableBundle`.

**G0 physical proof:** on true `gfx1151`, Q8/IQ2/Q4 GEMV, wave32 low-bit grouped MoE, GDN state update and attention selector vs independent CPU+legal HIP reference; inspect emitted ISA, register/LDS, occupancy and bytes/token. On NVIDIA compatible GPU, test a stable Rust CUDA tile-programming frontend and a Rust CUDA SIMT compiler frontend as separate experimental paths, not whole-workspace pinned nightly requirement. Use real model kernel results and at least one end-to-end request to qualify. Graph captured islands must be tested with changing expert-resident device tables and fail-retire semantics.

**No hardware name assumptions:** older research described an RTX 3060/Cezanne 27GB dev box. P00 must probe actual reachable hardware before selecting test variants or demanding a new Strix machine. A 92/96GB UMA host may have different practical model fit from published 128GB numbers. See `ASYNC_RESOURCE_COMMAND_IR.md` for target-independent event/lease planning.


## 2026-10-12 integration — XDNA2 AIE/Rust source feasibility provider

Add **`AcceleratorKind::XDNA2`** to the device-capability design and a provisional **`NpuKernelProvider`** with explicit programmable AIE tile/BO constraints: fixed matrix input/output shapes and batch multiples, supported block-float encoding (Gufo example `v8bfp16ebs8`, 9 bytes per 8 signed mantissas + exponent), 32 AIE cores (8×4 in source example), kernel precompiled program/instruction fingerprints, resource queue-depth and XRT hardware-context capabilities. The Gufo fork uses `M=4096,K=2560,N=1024` and allows four NPU jobs in flight; these are **one candidate's shape limits**, not a model or KIR ABI constant. Control plane CPU, GPU and NPU all produce typed device events.

**Source language requirement is a real blocker to investigate:** the fork's AIE program is implemented through Python-generated AIE IR and C++ device source, with a C++ XRT host bridge and compiled `.xclbin`. Existing zerogravity user intent says **owned engine and kernels authored in Rust**. A Rust XRT runtime FFI binding by itself is not a Rust-authored AIE kernel. Investigate Rust-generated valid AIE device IR or a supported Rust→AIE compiler path with physical correctness and performance; if no viable Rust-owned path exists, label `NpuKernelProvider` BLOCKED, without using Gufo's non-Rust kernel source or relaxing policy. Explicit user approval is required for any alternative.

**Hardware safety and shared memory:** driver and device capability probe records XRT and ROCm, XDNA generation, IOMMU requirements, GPU-NPU shared LPDDR and fabric power domain. The supplied Reddit/README report **hard machine hangs** during concurrent NPU/GPU clock transitions. GPU-only stays the default; no startup `sudo`, FCLK pin, suspend override, BIOS/AGESA/IOMMU alteration, long-lived heartbeat or systemd deployment without explicit user action and hardware-specific validated procedure. A software NPU timeout cannot recover a totally frozen host. Track pinned import limits and DMA lease lifetime after failure.

**Validation:** independently compare NPU block-float projections with CPU reference and GPU-only Q8_0 calculations (full layers and logits), per-device worst-case numeric and MoE router/indexer sensitive margins, long context and selected model task quality; benchmark realistic (M=4096/8192) under held *same clock* baseline/experimental paths and C1–C8, plus GPU-only fallback for unknown NPU or shape. Report prefill tokens/s, P95, memory/energy and state safety, not implied decode t/s. Source details in [GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md).

## v4 refinement — true closed-region providers and device-native loop proof

A KernelProvider is selected at **compile/load time** by exact semantic region, numeric and state-effect contract, bounded shape/ragged constraints, compute encoding, target device/ISA and physical resource plan. It can implement an entire layer/model closed region as Rust-authored fused code, yet must **not own its own session/allocator/service**. KIR is a transient provider/codegen representation; compiled handles and device-resident parameter slots live in ExecutableBundle and DeviceBackend.

A performance-qualified device-native step must show that router IDs, draft acceptance, next-step token input and sampling/grammar data remain on device where semantically legal, not incur host sync each output token. `DeviceBackend` must support async param buffer updates, event readiness and local pool allocation under global reserve. Fallback safe on unsupported shape/context, arithmetic/driver/capture or unqualified SoC power state. GPU-only executable unaffected by absent NPU driver/source compiler.

Device/kernel Rust source policy stays strict; optional driver and vendor libraries are a separately unresolved user decision, and wrapping foreign-language device kernel doesn't itself make it Rust-authored.

## v4.1 — kernel backend is a specialized form of a region-program provider

The old `DeviceBackend::compile(kernel)` and `KernelProvider` sketches describe **native kernel execution**, not all target accelerators. The single `RegionExecutionProvider` contract now accepts a **closed semantic region of any validated scale** and returns either `NativeKernel`, `CapturedCommandGraph`, `AheadOfTimeDeviceGraph` or `StaticallyScheduledDataflow` (plus carefully gated managed-state programs). `DeviceBackend` handles low-level buffer and program submission/event/teardown for its target. **Don't emulate a TPU/wafer/graph compiler by host-interpreting each op** or require a device to offer a CUDA-like single-GEMM launch API.

Backend capability includes `SupportedRegionScope`, op/quant/numeric/shape/graph control, program resident memory, load/compile/warmup costs, topology/fabric placement, collectives and `StateBridgeContract`. A graph-only backend's compiler can accept an **ephemeral interchange graph** containing only known exact supported operations; it is not a new enduring zerogravity IR. An unsupported custom recurrent or sparse op must be partitioned at a legal state/layout cut or rejected with `NoLegalLowering`, not silently approximated. A new compiler/runtime or static device program requires separate Rust-source/vendor SDK/licensing review. Native CUDA/ROCm/Vulkan/Metal kernels remain useful high-performance provider candidates and CPU-only remains independent.

**Blackwell family tests:** coherent ARM+GPU SoC vs discrete HBM/GDDR vs high-bandwidth multi-GPU rack are separate `TopologyProfile`/power and memory classes; do not assume a 273 GB/s LPDDR host inherits datacenter HBM speed. Distinguish NVFP4 (16-value FP8 scale), MXFP4 (different block scaling) and FP8/FP6 compute, physical layout conversion and quality. Graph-only TPU-style accelerator tests need static/bounded shape and verified program state bridge; wafer/static SRAM device needs large-region placement/streamed weights and honest simulator-only status until real compiler/hardware.

Full normative specification: [accelerator execution families](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md). Source/hardware refs: [platform prior art](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md).

````

## Source: .agents/docs/DISTRIBUTED_PARALLELISM.md

Original UTF-8 bytes: 7780; SHA256 `c381673cbe696280aba4773921c3168d8becb2d6d7e3f13a9c5118a88d727b1c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Distributed and heterogeneous execution — v4.2 authoritative index

**2026-10-12 · DESIGN ONLY; no distributed runtime implemented.** This file is the stable subsystem entrypoint. Full *normative* technical specifications are [distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [peer/tunnel/state protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md), [universal region execution](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) and [single-root state/worker protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md). Named prior art in [network research](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md). Earlier distributed sketches are superseded by these placement and single-publisher contracts; their original bodies remain in Git history.

## Primary goal

**One AI model/task execution engine** can use a single CPU/GPU, several local accelerators, or multiple explicitly enrolled peer machines (including private encrypted tunnels) by choosing where **compute regions, model parameters, mutable model state and immutable reusable caches** live independently, and selecting verified communication. A distributed session never creates another source-model-specific server, tensor store, TaskDriver, ResourceArbiter or authority to publish output.

**Always preserve a zero-cluster fast path.** One-machine execution should not start discovery/remote heartbeats, peer control sockets or require consensus/transport libraries at CPU/GPU model launch. Distribution is optional and only selected if measured task performance/capacity/energy/cost/rights justify it.

## Four independent placement maps and one physical plan

- **ComputePlacement:** per-layer group, attention core, FFN/MoE, tokenizer/audio/vision stage, tensor/head and token-range partitions, optional whole AOT/fabric regions; one RegionExecutionProvider for each closed region.
- **ParameterPlacement:** source/packed trained weights, adapters, sparse experts and immutable massive lookup rows resident/streamed/cached per device and allowed node.
- **StatePlacement:** per-layer/session KV, sliding/latent/compressed attention, GDN/KDA, indexer/ngram, RNG/grammar/sampler, draft and TaskDriver media state. Remote shard is prepared/versioned, not another coordinator writer.
- **CachePlacement:** content-addressed immutable full-prefix snapshots, encoder features, compiled program caches and replicated model data across permitted peers/storage; directory presence is only a hint.
- **TransferPlan:** ordered/overlapped activations, KV snapshots, state migrations, expert partials and collectives with physical TransportProvider, bandwidth/RTT/priority/memory leases, source/numeric security and cancellation.

Together they are one candidate **PlacementPlan within ExecutableBundle/CIR** tied to a TopologyEpoch and TaskSession's placement epoch, not a new enduring IR.

## Required topologies

| Topology | Main property | Qualification caution |
|---|---|---|
| Single-device colocated | compute and live KV near weights | baseline, minimal latency |
| Local multi-device tensor/expert/model pipeline | peer GPU/CPU/NPU work | actual PCIe/UMA and collective cost |
| Layers on remote machines with their per-layer cache | contiguous layer shards, activation handoff | decode adds serial hops; prefill chunk pipeline can overlap |
| Layer compute and KV/state on different nodes | remote attention near KV or state hydrate once | naive remote full KV reads per token usually slow |
| Attention+KV separate from FFN/MoE | memory-bounded attention vs compute-rich experts | per-layer roundtrips, hybrid layer imbalance |
| P/D disaggregation | prefill produces full compatible state for decode | large transfer/TTFT and destination reservation |
| Token-range KV shards | execute partial attention beside state, merge numerical softmax summaries | only where op PartitionAlgebra proves legality; not generic GDN |
| Remote immutable prefix/encoder cache | reuse state across requests/workers | stale directory/hashes and transfer vs recompute |
| Multi-model/task distributed stage pipeline | encoder, denoiser, text decode, vocoder on different nodes | individual task frontier/privacy/output |
| Multi-GPU rack/static programmed fabric | compiler-owned internal collective/program | rank/epoch fencing and StateBridge |
| Trusted encrypted tunnel peers | private small cluster across NAT/WAN | direct vs relay RTT; overlay isn't RDMA; explicit data authorization |

The legal plan set is bounded by **semantic effects, model/quant/position, state snapshot/restore, numerical quality, device/transport permissions, physical reservations, latency and failure recovery**. The performance optimizer can favor **capacity** even when raw C1 decode gets slower, but must label that tradeoff.

## Authoritative publication and node failure

**Exactly one logical TaskCoordinator writer per TaskSession/shard**. Participating RemoteRegionWorkers execute under SessionEpoch/PlacementEpoch/StateShardGeneration, return Prepared/Ready/Failed and hold safe leases. TaskCoordinator validates all necessary region/state groups, then one logical StateRoot PublishRoot and sequenced output. Device/peer can neither individually commit session state nor emit provisional output as real.

A disconnected peer or stale rank must be fenced; partial DMA cannot be recycled until safe quiescence; failure may require replay from last compatible committed root. Losing coordinator can lose volatile work and leave network delivery uncertain. **Do not promise distributed exactly-once or transparent failover without durable quorum/log/ACK machinery**, which is deferred. Immutable prefix caches may be eventually discovered but cannot override authoritative state shard versions.

## Transport and security

Peer enrollment requires explicit authorized identity and model/tenant/data permissions. **No public remote inference endpoint or arbitrary remote shell by default**. A control plane can connect over authenticated QUIC/TCP through a private tunnel, while data can use a different permitted fast LAN/RDMA or bounded encrypted stream; keys/registered memory regions are scoped to peer/session/lease and never logged.

The planner measures **direct vs relayed** paths, RTT/P95 jitter, actual payload throughput, MTU/packet loss, staging/copy/crypto/egress cost. Tunnel does not confer cache coherence or GPUDirect. Slow relays are inappropriate for many per-token exchanges even when hardware memory is abundant. A remote prefix transfer or small number of layer boundaries may be acceptable.

## Minimal staging and tests

**No cluster implementation before real local model CPU/GPU vertical slice.** Then first distributed milestone: one explicit trusted peer, one remote layer group with its own state, activation boundary transfer, simple reliable encrypted stream, correct TaskSession/root after network delay and peer loss. Second: remote immutable prefix and full hybrid state transfer; third: independent attention/FFN and sequence sharded attention where fast links/model math justify. High-bandwidth collectives/rack, automatic placement/migration and robust leader recovery are **independent later milestones**, not core baseline.

Tests: exact model/logits/state across local vs split; 3 different state domains and proper source/adapter hashes; C1 and C8 P95 on direct LAN/private tunnel direct/relay; wrong quant/state version, partial transfer, duplicate frame, node restart, cancellation, coordinator failure and slow client. GPU-only fallback must remain unaffected.

**Full contracts, costs, wire identity, security and test plans are in the linked normative specs.** Nothing here indicates code or performance has been validated yet.

````

## Source: .agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md

Original UTF-8 bytes: 37796; SHA256 `2c47bae7d0dae9391c87e34335e9cc76b5b4b7b17815ad9ff31d99d5e72fa5c1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — Distributed placement, network and state execution contract (v4.2)

**15 Oct 2026 — P/D disaggregation and communication/compute overlap:** [Joint serving contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) treats separate prefill/decode stage placement as a **measured QoS/latency isolation candidate**, not an assumed throughput improvement. TransferPlan includes **all source-required** KV, SWA, GDN/KDA, MLA/mHC/indexer, RNG/grammar and media state with model/artifact/quant/epoch/tenant-compatible receipt. Async network/collectives may overlap useful model compute only when semantic effect DAG, physical device events and peer trusts allow it; remote publisher cannot independently alter TaskCoordinator StateRoot. WAN/tunnel RTT and burst tail may make collocated mixed P/D faster; local single-GPU fast path has no RPC overhead. See SERVOPT-28..30 and vLLM prior-art study for current baseline.


**AUTHORITATIVE ARCHITECTURE · 2026-10-12 · DESIGN ONLY.** This makes distributed and heterogeneous execution **a first-class property of one AI runtime**, not a second cluster engine. Existing normative foundations remain the single SemanticProgram, transient legal region/provider lowering, one ExecutableBundle/CIR, TaskDriver, PhysicalResourceArbiter and **one logical TaskCoordinator/StateRoot publisher per session/shard**. The universal [RegionExecutionProvider](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) can span devices or machines as one compiled PartitionedProgram. The detailed [source-attributed network research](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md) is non-normative.

**User goal U-020:** run any *semantically supported model/task* efficiently across available hardware, alone or together, including networked machines reached over explicitly authorized tunnels. Independently place compute layers, model weights, live KV/recurrent state and immutable/shared caches; optimize based on topology, workload, memory, network and failure cost. Support current and near-future execution patterns, yet keep the single-machine default as simple and high performing as possible.

## 1. Central design rule — four independent placement maps

A normal transformer layer contains compute, trained parameter tensors and mutable session state that are often colocated but are **not the same resource**. They must be independently placeable:

1. **ComputePlacement:** assigns closed semantic regions, operators, layer groups, attention cores, routed expert work, encoder/decoder/denoiser stages and partial tensor operations to qualified compute domains. An AOT graph or statically scheduled program can internally own multiple ranks.
2. **ParameterPlacement:** assigns immutable model weights, adapters, sparse routed experts, embedding/lookup rows, packed source tensors and graph program images to memory/storage and possible replicas. The execute owner may differ from the source storage owner.
3. **StatePlacement:** assigns **mutable authoritative per-session state shards** (layer KV, sliding/latent/compressed history, GDN/KDA conv/matrix, sparse indexer, n-gram, sampler/RNG/grammar and modality/task state) to memory and qualified stateful compute. Each has an immutable committed version and a provisional mutation scope.
4. **CachePlacement:** assigns *immutable/reusable* prefix snapshots, encoder features, checkpoint copies and other recomputable cached artifacts to local/peer/remote RAM/SSD/object storage; cache-directory entries are **hints** rather than an authority to publish state. Replicas need content/schema/tenant hashes and leases.

A **fifth derived object, TransferPlan**, connects mismatched compute, parameter, state and cache owners over the physical network topology, with explicit shape, quant, copy, conversion, source/tenant policy and completion events. Together they form **one PlacementPlan**, not four independent uncoordinated global schedulers.

*Why this matters:* “layer on GPU-A and cache on machine B” describes **at least three execution implementations**:
- **remote cache hydrate:** copy B's needed KV/SSM/indexer state to A once, then compute locally;
- **remote stateful operator:** send A's query/activation to a qualified state compute worker on B, execute attention beside KV, return compact output;
- **remote state reads:** A directly pulls B's cache bytes every step; may be valid on fast coherent/RDMA fabrics but generally disastrous across a tunnel or WAN.

The planner must distinguish and cost each; never treat remotely *stored* cache as if it were coherent GPU-local memory.

## 2. Three graphs, two enduring runtime representations

**SemanticExecutionGraph:** existing SIR + TaskDriver stage/region DAG and explicit effect/state boundaries (canonical, device independent).

**ResourceAndPlacementGraph:** a **view within the compiled physical plan** binding compute regions and exact parameter/state/cache resource handles to physical owners, copies/collectives and remote event edges. It is assembled by a bounded placement solver from SIR, resource estimates and current topology. **Not a third durable IR.**

**TopologyGraph:** runtime snapshot of machines, devices, memory, stores, transports, trust/power domains and measured links, with generation/time validity. It may change dynamically; a StepPlan binds a **validated placement epoch**, not a raw mutable global topology map.

```
ModelDefinition + TaskDriver           Hardware/Peer Topology Discovery
            |                                      |
         SemanticProgram (math/state effects)  Node/Link Capabilities
            |                                      |
     legal closed regions -------------------------+
            |                                      |
     PlacementCandidate(compute, weights, state, immutable caches)
            |                 + routing/phase/shape/cost/rights
     compiled ExecutableBundle + CIR transfers/collectives
            |
     TaskCoordinator(session root + placement epoch)
            |
     StepPlan across local and RemoteRegionWorkers
            |
     Prepared state shards + event-ready cohort
            |
     Coordinator alone PublishRoot -> sequenced committed output
```

Only SemanticProgram and physical ExecutableBundle endure as compilation representations. A RemoteRegionWorker is **the same trusted compiled-region executor** deployed on a peer, not a cloned inference server/API/task engine.

## 3. Physical topology, device graph and measured transport

Each node declares a signed/validated **NodeCapabilityRecord** with node identity, trusted execution policy, available executable providers, topology snapshot generation, CPU/GPU/NPU model/driver/compiler/program ABI, physical RAM/HBM/UMA/pinned/headroom, SSD/object cache capacity, link interfaces, OS/cgroup constraints, thermal/power, in-flight queue/resource quotas and optional model/weight caches. **Claims and observed measurements are separately recorded.**

Network **LinkCapability** contains both direction metrics (may be asymmetric), throughput by transfer size and concurrency, RTT and P95 jitter, packet loss/retransmits, MTU, direct vs relay, data-plane eligibility, encryption and host CPU crypto cost, collective/RDMA prerequisites, egress charges and versioned trust policy. Measure continuously/periodically outside the inference inner hot path; do not believe advertised 100G port means effective 100G payload after overlays.

Device and network form **different but connected graph layers**: one physical machine can contain CPU, integrated GPU, discrete GPU, NPU and shared memory with mixed coherency; two machines may connect via LAN, Thunderbolt, PCIe, NVLink-scale fabric, RoCE/IB, QUIC/TCP direct or encrypted overlay. A data path can have multiple hops and bottleneck links.

### 3.1 Separate control plane from data plane

**Control/metadata plane:** peer identity, discovery/approval, capabilities, placement epochs, lease renewal, heartbeat, failure, transfer rendezvous and per-session authority. Over authenticated/tunneled reliable channels, bounded messages. It need not carry every tensor.

**Bulk state/activation plane:** negotiated data transfers of model blobs, per-layer activations, parameter shards, KV/SSM snapshots, expert results, device partial attention summaries and collectives. On an eligible trusted local fabric choose direct peer copy/RDMA where actual NIC/driver memory access and security allow; on general WAN/tunnels choose bounded authenticated stream-based copies, with explicit staging if GPU direct isn't available.

**Telemetry/hints plane:** optional lower-priority link probes/cache-directory updates, never a source of mathematically authoritative state or raw credentials. Do not allow a lossy datagram to become a committed StateRoot or an unacknowledged checkpoint.

### 3.2 Tunnel requirements

Support **existing trusted encrypted overlay/tunnels or direct authenticated peer links** as transport *options*, not core engine security assumptions:
- private direct peer through an encrypted overlay using a standard secure tunneling protocol, with NAT traversal where available;
- peer-relayed or DERP-like relay if direct blocked, automatically **classified as high-RTT/constrained** and re-costed; never silently keep RDMA/collective assumptions;
- authenticated TCP/TLS or QUIC streams as eligible transport implementations with versioned protocol and stream priority, but don't mandate nested QUIC-over-VPN when inferior;
- on-host UDS/shared memory and trusted fast peer/RDMA/Thunderbolt/NVLink/IB/RoCE where qualified;
- explicit user-approved remote endpoints, identity pinning/rotation/revocation and per-node access scope. Discovery **does not authorize** arbitrary execution, user prompt sharing or model-weight redistribution.

**Hard rule:** a tunnel does **not** create RoCE, GPUDirect RDMA, cache-coherent remote VRAM or near-zero P95 RTT. Use buffered send/receive with strict bounds and encryption/authorization unless physical network+driver prove special transports. A relayed WAN tunnel may be useful for capacity-first layer split or asynchronous snapshot migration but usually cannot sustain per-layer per-token attention↔FFN calls at datacenter-like latency.

**Security:** control channel and data channel both cryptographically peer-authenticated/authorized (overlay encryption alone isn't blindly taken as application identity); no open inbound public HTTP inference server by default; no remote code execution or arbitrary shell just because machine joins. Remote worker only runs already trusted, allowed compiled program/model artifact IDs. RDMA memory registration capabilities scoped to approved segments/epochs and revoked on disconnect/fencing; don't log memory keys or raw pointer values. Tenant/model weight rights and user data privacy are explicit placement policy restrictions.

## 4. Placement modes all represented by the same compiler and remote worker

| Mode | Compute assignment | State / weight / cache placement | Typical network usage | Caution |
|---|---|---|---|---|
| **Colocated** | entire model/TaskProgram on one worker | weights, mutable state and hot caches local | none | performance baseline; no network tax |
| **Device-local partition** | tensor/layer/expert parallel across local CPU/GPU/NPU | state per owning op or local shared memory | peer DMA/PCIe | device state/alias/precision |
| **Layer pipeline** | contiguous or noncontiguous legal layer group on distinct machines | each layer's KV/recurrent state **colocated** with its compute by default | hidden activations and stage control at split boundaries | C1 AR decode waits serial hops; throughput via microbatch |
| **Remote immutable cache** | model remains local | prefix snapshots/encoder features stored elsewhere | one-shot hydrate or asynchronous prefetch | transfer can cost more than recompute |
| **P/D disaggregated** | prefill on one node/pool, decode elsewhere | **entire compatible KV+SSM+indexer+RNG/modalities** handed off under resource handshake | bulk full state at handoff | TTFT transfer cost, placement admission |
| **Attention vs FFN** | attention operator on bandwidth/cache owner, FFN/MoE on compute-rich machine | **attention KV colocated with attention**, experts resident with FFN | query/activation/result each affected layer | repeated inter-node RTT; async overlap/batch needed |
| **Remote state reads** | attention compute on A | live KV on B | large KV reads each token/step | often prohibit on LAN/WAN unless true direct coherent/read fabric |
| **Sequence-state sharding** | per-token-range attention partial on multiple state workers | disjoint KV sequence shards, weightless attention nodes | query broadcast plus compact softmax summaries/reduction | exact supported attention math; not arbitrary recurrence/index |
| **Expert sharding** | routed experts across peers | expert weights near their compute, KV elsewhere | selected activation rows, partial sums, routing metadata | tiny decode C1 may lose vs local fetching |
| **Tensor/head sharding** | matmul/attention heads split per layer/rank | weights and state per shard | allreduce/allgather every layer | requires very fast links, generally not relayed WAN |
| **Task/encoder/decoder split** | audio/image encoder, text decoder, vocoder/denoiser separate | stage-specific parameters and state colocated with stage | stage output/features, plus state if needed | distinct output/frontier/codec |
| **Replica/tenant pool** | independent full or part replicas | optional remote full-prefix share | cache location hints and task routing | consistency, memory duplication |
| **Static graph/fabric** | one `RegionExecutionProvider` owns compiled multi-device program | device compiler-owned SRAM/HBM/streamed weights | internal static collective + external program events | don't host-micro-schedule device; StateBridge mandatory |

These **compose**: e.g. node A prefill, node B attention+KV, node C FFN/expert, node D remote prefix object store, with tensor/head sharding within B/C on fast fabrics. A chosen plan must be a **legal and cost-qualified composition**, not an arbitrary Cartesian product enabled without tests.

## 5. Per-layer state vs compute placement is first-class

`RegionPlacement` can be **per layer/per operator group**:
- `LayerOwner(layer_range, compute_node)`;
- `StateDomainOwner(session, layer, state_kind, shard, node, epoch)`;
- `ParameterShardOwner(artifact, layer, tensor, replica, node)`;
- `CacheReplicaOwner(prefix_or_encoder_key, node, immutable_epoch)`;
- `ExecutionRole(node, task_stage_or_phase)`.

State owner is the device/machine physically holding **the authoritative mutable candidate and committed logical version** for that state shard—not a second independent TaskCoordinator. The global session root stores a versioned manifest of references to owned shard versions, with location + lease/fencing and compatible schema. The **TaskCoordinator alone publishes the new root** after required participant Ready responses.

For a stage/layer on a remote node, the minimal execution protocol is the *same* `ExecuteClosedRegion` work with source/versioned input activation, state shard handles and output to the next stage; the receiver validates semantic program, quant/precision, shape/layout, state epoch and physical reservations. A weight shard can be fetched once and retained, while mutable state remains at this layer worker over many tokens. **No full model or whole KV must be transferred per token** for ordinary layer-pipeline.

**Noncontiguous layer placement** is representable if source math and state boundaries allow and cost model tolerates extra hops; prefer contiguous groups when sparse networks because each extra boundary adds latency. Highly nonuniform hybrid layers might justify different per-group owners on fast fabrics.

## 6. Live state separate from compute: three physical plans

### 6.1 Cache hydrate once then local execution

Fetch an immutable compatible prefix snapshot or transfer a committed mutable state version from remote cache owner to compute node. Check `StateCompatKey`, resource reservations, page alignment, transport checksum and secure tenant scope; publish new location/version only at a committed task frontier. The new compute node owns local copy/lease for subsequent steps.

**Use when:** repeated future local compute will amortize a bulk transfer, link fast, and state layout compatible. **Reject when:** state huge, remote bandwidth constrained, recent prompt cache has short remaining useful life or full hybrid snapshot missing; recompute or remote operator may win.

### 6.2 Execute stateful operator beside remote state

Keep KV/SSM or sparse indexer on B and send **small activation/query** from A to the qualified attention/recurrent operator program on B. B executes into provisional state, returns compact output and readiness events. FFN/dense projections can stay on A.

**Use when:** remote cache is large, per-step query/result smaller, B has required math/ISA and per-layer network RTT + queue can be hidden or amortized by batching. **Reject when:** RTT per layer/token makes ITL unacceptable, B only storage without compute, state mutation isn't rollback-safe or slow tunnel relay path.

**For recurrent GDN/KDA,** do not replace exact time-dependent recurrence with ordinary remote softmax. The operator's own StateDomain schema, step ordering and snapshot support govern what may execute remotely. A state-only cache server may **not** be capable of running attention/SSM math; capability discovery must prove it.

### 6.3 Remote live cache byte reads per step

Only legal when physical remote memory access/read consistency/latency and owner leases can satisfy the exact operator, ideally a verified direct high-throughput fabric. Naively streaming full growing KV across ordinary network every token is **not** a default. Planner must estimate bytes and reject when bandwidth/latency costs defeat other placements or violate SLO. A tunnel path generally favors hydrate/recompute or remote compute rather than per-token full-KV reads.

## 7. Distributed sequence attention: compact partial-result composition

For mathematically ordinary softmax attention, token-range KV shards may execute attention partials **at the KV owners**. Let a fixed query have per-shard score values. Each worker produces:
- `m_i`: maximum valid attention logit for its causal/masked shard;
- `z_i`: sum of exponentials of score minus `m_i`;
- `n_i`: vector sum of exponentials times values under the same scale.

Combiner takes global `m = max_i(m_i)`, total `z = sum_i(exp(m_i-m) * z_i)`, total `n = sum_i(exp(m_i-m) * n_i)`, output `n / z`. This is **mathematically the same softmax weighted sum** in real arithmetic when shards partition valid keys with correct position/mask/scale. Finite precision/reduction order may differ; declared numeric and full-model quality gate required. Empty shards (all masked) have special zero contribution and well-defined `-infinity` maximum handling.

**Caveats:** multihead/GQA geometry, RoPE positional phase, causal mask, attention bias, sliding/local windows, latent indexer-selected keys and compressed KV semantics must be honored. Sparse global top-k selection may require global score candidate aggregation before sharded attention; a generic local fixed top-k can be wrong. Full softmax partition doesn't automatically extend to GDN/KDA recurrent state or deep learned compressor/indexer. **Operator registry must declare `PartitionAlgebra` and proof/reference**, else sequence sharding prohibited. This is a **capability extension**, not a baked-in mandatory attention algorithm.

Benefits: KV storage scales across attention nodes, full KV need not move every token, compact query/summary travels instead. Costs: broadcast/collective at each attention layer, P95 RTT, partial state update and memory pressure. May be profitable on fast local/rack fabric for long contexts, not automatically tunneled wide-area networking.

## 8. Placement plan validity, versioning and state migration

`PlacementPlan` binds:
- `SemanticProgramKey` and task stage/regime;
- `TopologyEpoch` and participating allowed nodes/link capabilities;
- `PlanEpoch` and immutable compiled RegionExecutionProvider candidates/program IDs;
- exact ComputePlacement, ParameterPlacement, StatePlacement, CachePlacement and TransferPlan;
- `StateCompatKey`, per-shard `StateShardLease` with ownership/fencing version;
- physical memory and network reservations, output/source numeric policy and tenant security;
- `FallbackPlan`: colocated, simpler layer split, portable provider, migrate/replay/refuse on failure.

A plan stays frozen for a **safe microstep/collective group**. Dynamic topology discovery can propose a new plan asynchronously; switching during an active speculative/remote collective would corrupt state unless a compiled protocol explicitly supports it. Plan changes happen at a **committed TaskFrontier** or an operator-specific snapshot/phase boundary where all old cohort events are quiesced, candidate shards copied/validated and owner leases atomically replaced in the logical StateRoot.

**Two-phase relocation (not a database):**
1. Reserve new owners + transfer immutable/committed compatible model/state snapshots, warm needed program.
2. Verify hashes/state codec and readiness, stop or fence old writer for that region at approved TaskFrontier, publish new placement and new remote state-shard leases in one root epoch. Retire old location only after late operations invalidated/fenced.
3. If migration fails: old authoritative committed root and old qualified placement remain, or reconstruct from latest canonical checkpoint/committed history; never mix half-old and half-new state.

A single cold **KV prefix-cache hint** may be stale without affecting correctness: revalidate bytes/hashes/availability on use, fetch elsewhere or recompute. A **mutable state shard owner** must have stronger lease/fencing epoch and cannot be selected from gossip without verification.

## 9. Minimal control plane and remote worker protocol

**Default installation:** one process on one host, zero remote peer discovery/network threads or unnecessary collective tax. A user may explicitly create an authorized private peer group; node participation is a **capability/authorization**, not automatic trust of subnet/broadcast/tunnel. Early implementation may use configured peer IDs and addresses; optional mDNS/overlay discovery later.

Minimal **PeerAgent** on each authorized machine:
- exposes signed `NodeCapabilities`, version, quotas and heartbeat;
- accepts scoped `PrepareExecutable(artifact/program/hash/shape)`, `ReserveResources`, `ExecuteRegion`, `TransferState/Activation/Weight`, `Cancel/EpochFence`, `Release`;
- uses same local `RegionExecutionProvider`, `DeviceBackend` and `ResourceArbiter` as standalone zerogravity;
- does **not** run a separate chat API/server, model orchestrator, state-root publisher or arbitrary remote shell on behalf of peers.

Illustrative wire identity, **not frozen ABI**:
```text
protocol/schema version + cluster membership epoch + peer identity
model artifact + semantic op + quant/shape/numeric and program hash
session id (opaque capability) + TaskFrontier + coordinator generation
placement plan id/epoch + state shard id/version + resource lease token
region id, input/output shape and transfer descriptors + operation sequence
deadline, cancellation epoch, checksum/authenticated payload and ACK level
```

Encode bounded typed frames with explicit endian/version/limits, replay protection, per-stream quota and cryptographic transport identity. Peer only executes verified/allowed model programs and tensor operations; **joining a cluster is never permission to run arbitrary arbitrary model-supplied code or arbitrary process commands**.

Control plane can use authenticated QUIC/TCP over an existing private tunnel, with coarse session/task metadata. On suitable fabric a distinct data plane can use direct memcpy/IPC/P2P/RDMA. `TransportProvider` chooses physical implementation from measured `LinkCapability`. Prioritize state transfer and token-critical activation over optional cache warming/telemetry via credits. No transfer may outlive its source/destination BufferLease or stale RDMA registration.

## 10. Coordinator, split brain and transport failure

**One logical TaskCoordinator session writer** remains the authority for root publication; it may be placed on a chosen machine independently of model compute. Local peer workers only `Prepared/Ready/Failed` state versions for the same expected `StateRoot`, attempt/placement epochs and required participant cohort. A remote buffer ready event does not mean a new token is committed; task output sequences are emitted **after** coordinator PublishRoot.

**Network partition/leader loss:** losing the coordinator can lose volatile prepared work and possibly recent published outputs not durably replicated. We do **not** promise transparent failover or exactly-once delivery. To permit recovery:
- **SessionLease / fencing epoch** from a trusted coordinator or future quorum/lease service; a node with expired/older epoch must reject writes.
- if supported, persist or replicate **immutable committed checkpoints** and output cursor, with explicit durability level and privacy; reconstruct from last safely known checkpoint and compatible StateCompatKey, replay where exact and affordable;
- in a partition, **fail-stop or serve only sessions whose coordinator lease remains valid**; do not create two independent writers with competing roots; late old coordinator fenced on rejoin;
- worker/collective timeout after partial GPU/remote DMA: quarantine affected buffers and abort prepared candidate, not “release because socket closed”;
- queue/graceful drain/rebalance at safe stage/frontier, avoid hot plan thrash based on transient P95 jitter.

**Consistency distinction:** immutable weight/model/prefix artifacts can be content-addressed and eventually discoverable; **mutable authoritative live KV/recurrent state must be linearizable under its single session-root writer** within the promised volatile epoch semantics. Durable exactly-once distributed output/consensus is optional future scope, not implicit from network tunnels.

## 11. Network-aware cost planner and performance policy

Choose the physically best plan **by task phase/workload, not static number of GPUs**. Input includes actual state bytes and model/quant, activation boundary bytes, model layer FLOPs and measured kernel time, cache hits/age and acquisition cost, output/token/denoise work unit, concurrency/fairness, link throughput/RTT/P95/jitter/loss, transfer staging+crypto+double tunnel overhead, power/energy/egress and failure risk.

Useful qualitative cost model:
```text
Per-step critical path =
    longest dependency-path of device compute + remote region waiting +
    peer control latency + data movement + state snapshot/fences
    minus legally overlapped independent async work

Placement objective =
    minimize quality-constrained P95 task latency / resource cost
    subject to physical/transport/tenant rights/StateBridge/SLO budgets.
```

A transfer `T_move` at minimum includes `bytes / effective_link_rate + RTT_or_handshake + buffer_copy + crypto/IO costs`; **real** asynchronous overlap is measured, not assumed. Multiple separate per-layer interactions multiply link latency; single prefill/decode state transfer has large bytes only once. The measured link and cache transfer are directional and affected by other tasks.

**Plan modes:**
- **Latency**: local/state-colocated, minimum serial network cuts, no unnecessary remote per-token action.
- **Capacity**: contiguous layer groups, cached shards and longer latency acceptable if model otherwise cannot fit.
- **Throughput**: multi-session pipeline, expert/tensor/sequence sharding on fast links, prefill/decode pools and multiple replicas.
- **Cost/Energy**: prioritize existing idle nodes/storage and model sharing, but never violate correctness/SLO/privacy or turn off CPU/GPU safety.
- **Private**: only nodes authorized for specific model and tenant content; remote bridge may be denied even if fastest.

**Minimum performance qualification:** no generic “more machines = faster” claim. Compare all-local same model/quant when it fits, forced offload, layer split, remote cache hydrate, attention/FFN split, sequence shards, P/D and hybrid placement with controlled link emulation and real LAN/tunnel/relay links at C1 and C8+ with long context. A capacity-only win can be useful while decode t/s decreases; label it honestly.

## 12. Distributed placement by task kind

- **Language generation**: layer and attention KV/SSM state, MoE experts, prefill/decode, trained MTP verification and tokens; coherent causal checkpoints and sampler/grammar owner.
- **Long agent context**: immutable full-prefix snapshots shared across requests, decode KV sequence shards, remote attention/core versus local FFN, reducing repeated KV movement; application/tool calls still permissioned outside core.
- **Embeddings/rerank**: stateless or windowed encoders sharded/replicated per batch, remote static graph program likely efficient; no language KV required.
- **ASR/TTS**: audio preprocess/encoder/acoustic/vocoder stages on different machines if feature/output size makes sense; per-PCM/window frontier, link codec/latency, private speaker/audio data.
- **Image/video diffusion**: denoiser/encoder/decoder stages can be placed separately with large latent tensors or frame streams; model weights may dominate RAM, transfer cost vs compilation/step loop and media data rights must be measured. Stage full-program graph on target accelerators when appropriate.
- **Future training/adapter fine-tuning**: ParameterVersion/gradient/optimizer sharding and fault semantics are independent future TaskDriver capabilities; inference sessions pinned to immutable parameter versions, no silent cross-task weight mutation.

A `RegionExecutionProvider` can own a remote closed program across multiple local chips and return one Ready event; the host **must not emulate every internal TPU/wafer graph operation over the tunnel**.

## 13. Security, model rights and user experience

Local-first: no outside hosts used without explicit user enrollment. Maintain **node identity, authorized model/artifact set, tenant data permissions, signed program/source/quant provenance, transport encryption, certificate/key rotation, lease expiry and clear revocation**. Different administrative domains are mutually distrustful by default; no automatic prompt/embedding/model weight export, despite valid tunnel routing. Weight licenses may prohibit redistribution; a host's spare GPU capacity doesn't override model rights.

Monitoring exposes per-peer capability, direct/relay transport, egress bandwidth and cost, actual layer/attention/FFN/weight/cache/state placement, memory reservations, qualifying state schema, observed network P95, output/compute owner, and why a requested topology is rejected or slower. The CLI/SDK can offer simple **one-machine / add trusted peers / auto measured topology / pin group / explain plan** modes, while advanced placement can be declarative/locked. Never expose raw user data or RDMA memory keys in traces. A tunnel peer need not open a public API.

**No implicit uptime promise:** a tunnel relay may drop, NAT change or worker vanish; fail-stop/recover from last valid committed root and report why. No distributed dynamic load balancer, membership consensus database or Kubernetes dependence required for first 2–3 private nodes. Scale control-plane sophistication only with measured need.

## 14. Proof/test program and staged delivery

**Stage D (design/simulator, independent of physical network):**
1. 3 fake heterogeneous nodes, different compute/weight/state/cache placements, exact same SIR/TaskDriver and StateRoot, no cloned inference service.
2. Layer pipeline with KV colocated per layer; capacity-first and C1 serial latency, two chunk prefill overlaps.
3. Remote immutable prefix cache: hit/miss/partial state, full hybrid GDN/SSM/indexer compatibility, transfer once vs recompute.
4. Attention versus FFN: KV colocated with attention node, selected activations/results over slow and fast links, hybrid attention cost.
5. Sequence-token KV shards: mathematically exact softmax shard summary combine and numerical quality, reject unsupported sparse/indexer/recurrence.
6. Per-step remote full KV read negative test: cost planner must reject tunnel/WAN unless measured valid, compare query remote attention.
7. Transport topology: direct LAN vs high-RTT/tunnel peer relay; RDMA capability never inferred from overlay; bandwidth/crypto/MTU effects.
8. Coordinator fail/reconnect, duplicate frames, old-epoch remote writes, lost Ready ACK, network partition, no split brain or uncommitted output.
9. State migration across machines at committed frontier, disabled fast path and recovery old placement, no memory leaks or zombie DMA.
10. Different TaskDriver (encoder/ASR/image) stage placement and privacy/tenant restrictions without new backend server.

**Physical qualification (future, only with permission and accessible machines):**
- 2 local heterogeneous machines on Ethernet or direct high-performance link, actual model/checkpoint/quant and generated tokens, multi-node layer split;
- same through explicitly enrolled private tunnel with direct/relay path fingerprint, measured P50/P95 and energy, correct fallback;
- independent remote cache/prefix + optional remote attention/FFN on verified link, snapshot migration and per-state math reference;
- higher-bandwidth rack RDMA/collectives/topology and eventual TPU/wafer/ASIC compiled partition only if SDK/device available;
- mixed node drain/rejoin/failures without leaked state, at-most-once live delivery attempt or explicit cursor resume.

**Implementation staging:** P00 local CPU/GPU no network overhead → P07 optional transport seam/fake peer → P09 trusted peer enrollment/metadata → P10 per-state handles/cache directory → P16 work planner cost → P19 remote region worker / layer split first, then remote prefix/P-D, attention/FFN and sequence shards on qualified links → P22 matched hardware tests → P23 new topology extension drill. Nonessential service orchestration/consensus/distributed KV datastore is **deferred** until needed. Existing 24-phase IDs preserved.

**Plan acceptance statuses:** ARCHITECTURE_ONLY, SIMULATED, TRANSPORT_TESTED, SEMANTICALLY_VALIDATED, STATE_SAFE, PHYSICALLY_QUALIFIED, PERFORMANCE_QUALIFIED, PRODUCTION_QUALIFIED. Do not mark functional because a network library crate compiles. A model/task unsupported by any remote accelerator must still work in available qualified local plan when possible.

## 15. Architectural completeness test

A new machine with a different GPU, NPU or topology should require **only** NodeCapabilities, RegionExecutionProvider backend and a qualified peer TransportProvider; it must not require a new model math description, TaskDriver, API server, StateRoot coordinator or independent distributed execution engine.

A new placement strategy (KV remote attention, per-layer cache shard, P/D transfer, token-range attention or expert route) should require **only** a legal region/StateDomain partition strategy plus TransferPlan and cost/failure policy, leaving other models/task kinds untouched.

**Core invariant:** distribute *where it helps* and colocate *where it is necessary*, but never trade away source mathematical truth, full causal state, physical memory safety, authenticated remote access or a minimal performant single-host fast path.

## 16. v4.3 — model router load balancing versus distributed one-model sharding

**ModelRouter operates at TaskRequest/model admission; distributed PlacementPlan operates within an already resolved exact model binding.** A single physical `ResidentModel` may shard one checkpoint across several machines (pipeline/tensor/expert/attention/state), whereas two different `ResidentModel` deployments of the same exact checkpoint/quant/TaskDriver/numerical policy form an **equivalent replica group** eligible for request-level load balancing. Two distinct model checkpoints on different nodes are **not** equivalent replicas, even if user alias or API format seems similar.

Route choice factors: source/tenant authorization and model weight redistribution rights, per-peer RAM/VRAM/prefix cache/StateShard locality, measured link direct/relay RTT/P95, cold load/graph warm and KV migration cost, available replica queue, model-parameter/quant source fidelity, task phase and requested SLO. Request/session binding to an already live distributed StateRoot remains sticky unless compatible full-state and placement epoch migration. A helper remote peer executing only one layer doesn't expose its own OpenAI server or independent model routing. When a target/draft/encoder pipeline uses several different models, the parent task reserves each model deployment under one admission budget and separate state/quant identities.

No automatic remote use from a local OpenAI API request: explicit permission to offload user content and licensed model weights required. A runtime may expose many remote qualified models through **one** local loopback/API endpoint, with `GET /v1/models` filtered by current tenant and actual task support. A model alias hot reload never changes active peer-owned StateShard generation or model source. [Router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md).

````

## Source: .agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md

Original UTF-8 bytes: 23782; SHA256 `5aaf6b9f69a004dfcd2354066d23ba486992dd3b0d2c4e052c91cc96f2dc4b07`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — Native Engine Core, Worker Protocol, KV policies and request lifecycles

**15 Oct 2026 — high-concurrency hot-path execution:** [Joint serving contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) defines **bounded Prepare → Dispatch → Retire** pipelines; CPU processor/source/SSD IO and GPU kernels overlap only under real device events and exact task-root dependencies. Active worker counts and queue selection differ from physical GPU submission/Ready. Compatible rows can form true batched kernel plans without per-token JSON, catalog routing, global resource snapshot or dynamic compiler search. Device streams and in-flight graph depth must be physically proven; the current one-physical-region CubeCL adapter is not evidence of a concurrent production GPU engine. Single TaskCoordinator still publishes StateRoot; client send/ACK cannot authorize early buffer release. SERVOPT-24..27 and 32 provide negative/hot-path gates.


**15 October 2026 — BOUNDED STORAGE MATERIALIZER, SAME ENGINE:** The source TensorView/physical BufferLease/DeviceWorker protocol is shared by fully resident and SSD-streamed model stages. Per-device worker remains authoritative for actual GPU events and slot lifetime, bounded asynchronous IO service owns source-file staging, and TaskCoordinator exclusively publishes SessionStateRoot after valid source data and kernel result. No separate SSD task coordinator/engine. Direct resident path has no filesystem scheduler per token; miss-capable graph island yields to bounded host IO and resumes at actual DMA completion. A captured GPU graph's **encoded-but-not-yet-executed** resident pointers are leased until GPU quiescence. Exact recipe and tests: [virtual residency](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md), [SSD roadmap](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md).


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**Working design, not implemented.** Combines the detailed single-owner, typed event and block-pool protocol from earlier architecture research with typed SIR state effects and CIR resource commands. This document resolves operational questions the original project scheduling document left underspecified.

## 1. One mutable owner, many concurrent hardware workers

One Rust logical `TaskCoordinator/EngineCore` is the **single writer per session or bounded session shard** for task cursors, request admission intent, committed state roots and speculative verification. It does not own every physical allocation: the global ResourceArbiter reserves at coarse boundaries and device-local subpools allocate/retire kernel buffers under those reservations. A dedicated event-driven owner thread per sharded cohort is one option; external service tasks, CPU processing and device workers send typed messages. Workers own GPU contexts/queues/graph caches, device-visible expert mappings and temp kernels. Async IO service threads own SSD read and staging queues. No `Arc<Mutex<Sequence>>` on every hot kernel or output token.

A dedicated owner is a working default, not permission for one global lock across all devices. Multi-GPU may use a coordinator and per-rank worker state with one canonical request owner; independent model instances can have independent owners.

## 2. Protocol envelopes and communication

```rust
struct EngineEnvelope {
    request_id: RequestId,
    session_id: Option<SessionId>,
    epoch: u64,
    body: EngineMsg,
}
enum EngineMsg {
    Add(PendingRequest),
    Cancel { id: RequestId, epoch: u64 },
    UpdatePriority { id: RequestId, priority: Priority },
    GrammarReady { id: RequestId, grammar: GrammarHandle },
    PrefixIoDone { id: IoId, outcome: Result<PrefixHandle> },
    ModelReady { model: ModelId },
    ReloadPolicy { generation: u64 },
    Shutdown,
}
enum EngineEvent {
    TokenCommitted { id: RequestId, token: TokenId, frontier: FrontierId },
    Finished { id: RequestId, reason: FinishReason, counts: UsageCounts },
    Error { id: RequestId, reason: EngineError },
    Metrics(MetricRecord),
}
struct StepPlan {
    step_epoch: u64,
    new_requests: Vec<RequestBinding>,
    cached_requests: Vec<PrefixHitBinding>,
    token_rows: Vec<RowAssignment>,
    preemptions: Vec<SessionId>,
    kv_allocs: Vec<BlockLeaseId>,
    state_forks: Vec<FrontierFork>,
    device_regions: Vec<ExecutableRegionHandle>,
    sampling: SamplingPlan,
    draft: Option<VerifyPlan>,
    resource_leases: Vec<BufferLeaseId>,
}
struct WorkerDelta {
    step_epoch: u64,
    completed: Vec<CompletedRegion>,
    tokens: Vec<ProvisionalTargetOutput>,
    state_updates: Vec<StateVersionHandle>,
    events: Vec<DeviceEventId>,
    errors: Vec<WorkerError>,
}
```

Types are sketches; target protocol must avoid cloning token histories or giant activation tensors. Use compact index/generation handles and buffer references, not serializing model tensors every step. `StepPlan` is the immutable command/delta for **one scheduled step**, not SIR/KernelPlan nor general object with all model semantics. Workers cannot commit visible state unilaterally.

`EngineCore` uses bounded ingress/egress and worker queues with backpressure. Priority queues and two in-flight steps require generation guards and isolated provisional states; forced grammar/token stop acts as barrier. Abort event may be processed immediately but physical leases retire after worker quiescence.

## 3. Step loop and critical-path discipline

```text
loop:
  drain incoming EngineMsg and IO/model completions
  process WorkerDelta events from finished device steps
  verify/commit all state domains for completed frontiers
  schedule next request group under token, memory and fairness budgets
  reserve CIR leases and bind precompiled variant handles
  if executable: submit StepPlan to worker, without blocking
  otherwise sleep on bounded event (not busy-spin)
```

The implementation may overlap preparing step n+1 while n executes only where state or grammar dependencies permit. Speculative placeholder scheduling needs explicit validation that no uncommitted sampler/RNG/indexer state leaks into the next step. A 2-step pipeline is an optimization, not a correctness requirement. Strict mode can run one synchronous step for replay.

"Run" and "serve" share this EngineCore library. They need not send loopback HTTP between themselves unless this is proven to simplify semantics at acceptable overhead; can use a direct local edge adapter while keeping the same tokenizer/request state pipeline.

## 4. Two budgets and three execution regimes

Research observes a unified token step `{seq_id: new_tokens}`; extend with explicit constraints for:
- **target compute token budget**: prefill/decode/verify target rows (including verification work for rejected proposals);
- **input/preprocess budget**: tokenization, image/audio encoder and pending input spans, often CPU or other devices;
- **resource budget**: physical RAM, GPU addressable/allocated bytes, KV and recurrent per-session, graph scratch, expert cache and provisional state;
- **fairness/latency budget**: queue residence, TTFT, P95 inter-token deadlines and priority.

Do not count draft proposals as committed tokens. Mixed prefill and decode uses bounded prefill chunks (128/2048 from research are tuning candidates, not global constants). Native batch kernels reuse weights across sessions; each session's state remains isolated.

**Latency profile:** avoid waiting for extra rows; eager/special decode and low TTFT. **Throughput profile:** aggregate work and expert reuse, protect P95. **Extreme low concurrency:** reserve-all state, preallocated contiguous KV and model-specific graph islands where measurements justify; no generic scheduler rewrite. **Quality-reference:** disable unvalidated numerical changes and speculative modes.

## 5. KV strategy under one resource arbiter

### Paged block pool (baseline)
Borrow research's `BlockMeta` slab with `ref_cnt`, block hash, free-list links, epoch/generation and in-flight lease state. Reuse cached blocks without allocation; free list with O(1) eviction/refcount and parent prefix hash chain. Separate **manager block granularity**, **kernel page/block granularity** and **semantic state alignment**—the latter is compressor/indexer-specific. A null block and checked page table mapping are useful, not mandatory external API.

No dynamic GPU buffer allocation for every token; page availability reserved/leased before step. Deferred-free fence protects in-flight steps. Partial KV page hit may be COW or recomputed at a legal boundary and must preserve state semantic alignment.

### Optional radix prefix index
Keep an optional radix/trie index of canonical token fingerprints that points to the **same** underlying pool; it is never a second allocator. Enable only on prefix-dense traffic when gain exceeds maintenance/fragmentation cost. Index supports tenant salt isolation, LoRA generation, tool-renderer/model hash, modality content fingerprints and exact token-ID prefix identity.

### Contiguous extreme pool
Separate implementation of `KvPool` for low-concurrency specialized graphs, with reservation of committed prompt plus answer room, safe growth/move at checkpoint, no preemption until plan transition. Admission mode `ReserveAll` and StepPlan changes are explicit; cannot flip one allocator flag mid-decode while pointers remain captured.

### General hybrid-domain coordinator
The actual prefix cache can include full KV, sliding KV, compressed MLA/CSA, GDN/KDA, indexer, n-gram, vision span, sampler and optional grammar state. Each domain publishes `CanResume`, required suffix and finalization alignment, a canonical state format and restore/recompute cost. Compute **common valid resume frontier** across *all* domains, not only token-budget serving research's limited two-family coordinator. A logical token prefix match without state availability is not a complete cache hit.

## 6. Persistent worker input batches and step diffs

Worker keeps an indexed `InputBatch` map by stable request slot (with generation). StepPlan specifies `new/cached/updated/preempted/finished` and row assignments; apply diffs in a deterministic order to avoid copying every request's entire token sequence. `condense` reclaims holes only when no in-flight graph still references affected slots. Shape-regime variant indices determine decode1, multi-req, verify-small and prefill-large compiled kernels.

Persistent device slot buffers and graph pointer table can be updated by a small validated device copy at safe epoch rather than graph recapture. Mode receipt logs eager/capture plan and shape bucket, quant, KV policy and fallback reason.

## 7. Sequence state and speculative transactions

Per session store:
- committed token count/frontier, running epoch and tokenizer/template identity;
- KV page table + recurrent/conv state + compressed indexer + n-gram history;
- multimodal spans and encoder fingerprint; token penalty+grammar FSM and sampler RNG;
- immutable shared prefix snapshot handles and private mutable tail;
- optional trained MTP/DSpark/EAGLE/small model draft state.

A `CommitPlan { FullDirect, Prefix(n), Replay(from) }` maps accepted tokens to **full state transaction**; a bounded `Frontier` ring can provide cheap logical offset snapshots while buffer generations guarantee no rejected mutations. O(1) logical frontier switches are not always O(1) physical GDN/KDA/indexer restoration. If impossible to keep all snapshots, recompute from canonical checkpoint, and record this in performance cost/receipt.

Exact sampled speculation needs rejection/correction respecting target `p` and proposal `q`, and the sampler's RNG history. Greedy "same plan" identity can be validated, but floating math across different GPUs/batch order may diverge near ties.

## 8. Interoperation with async CIR and resources

- Scheduler owns intent, not raw GPU pointers.
- ResourceArbiter owns **global budget/reservation authority**; DeviceBackend local pools own fast physical suballocation and issue generation-tagged BufferLeases within reserved budgets. Neither publishes logical TaskSession state.
- CIR executor owns event order, async IO and graph segments.
- `WorkerBackend` binds plan region and actual device capability.
- `ExpertStore` exposes source/prefetch/publish, never authorizes a new routed expert identity.
- `ModelRecipe` defines semantics and state, never an HTTP endpoint or kernel plan.

A required expert miss from GPU produces bounded host request and async upload; scheduling of unrelated sessions may continue but current dependent command waits safely. Per-layer expert route selection stays on device for fast resident path.

## 9. API consistency and mode receipts

A canonical `InferRequest` has model artifact hash, rendered prompt and token identity, output generation and sampling contract, modality, tools/grammar, priority, context/output budget, correctness mode and user options. OpenAI/Anthropic/Responses adapters translate into this; unsupported features return typed errors.

Response `ModeReceipt` records actual kernel tier/compiled variant, graph capture, quant/precision and checkpoint, KV layout/codec, expert tier/IO miss cost, speculation method/accepted distribution, cache reused tokens, model/context/SLO, fallback reason and quality policy. Do not embed private raw prompts/tokens into metrics by default.

## 10. Invariants and failure modes

- **Single logical writer** per session epoch; independent sessions may native batch with private state.
- GPU worker never publishes externally visible tokens before EngineCore commit.
- Active BufferLease generation matches every device/IO use; eviction/reuse only after event quiescence.
- `StepPlan` and `WorkerDelta` epochs verified; late/duplicate completions discarded.
- Scheduler never assumes future router expert IDs or unconfirmed speculative tokens.
- Deadlines/priority prevent long prefill or repeated spec from starving active decode.
- Resource plan cannot allocate above physical budget, regardless of HIP virtual GTT.
- HTTP disconnect and tool/sampling boundaries can abort provisional tokens and rewind consistently.
- No expansion of output stream beyond the committed frontier; real stream cancellation may retain already emitted tokens.
- Unknown mathematical op/quant/model configuration fails with explicit model/capability refusal.

## 11. Gating tests

1. Simulated scheduler + naive vs paged block pool differential, property/fuzz for block conservation/refcount.
2. 1e6 random session events (fork/prefix/cache hit/cancel/verify/replay) after fuzzer implemented.
3. Persistent InputBatch slot reuse under 2 in-flight steps, generated stale epoch cases.
4. Grammar/tool completion barrier during speculative verification and EOS.
5. Native batched vs ordered fallback labeling at 1/2/4/8/16/32 concurrency, P95+aggregate throughput.
6. Real CPU/GPU model full logits/state with chunked prefill and long prefix cache restoration.
7. IO miss, invalid expert lease and broken CUDA/HIP capture injection with no wrong output.
8. Reuse same compiled bundle across model sessions without leaking RNG/hidden state.

## 12. Open decisions

Exact event queue crate, internal protocol serialization, number of owner threads per GPU/model process, default KV block size, whether extreme reserve-all enables at C>1, grammar inference barrier and reusable command-list compilation strategy. These decisions require a functioning vertical slice and controlled benchmarks.

## 2026-10-12 integration — cross-device first-use barrier without extra EngineCore

An NPU execution provider is a **worker capability**, never a separate model/session scheduler. EngineCore keeps the single logical request owner and only publishes committed model state. StepPlan may reference an ExecutableBundle that starts NPU-projection rows after GPU packing, overlaps other GPU compute, then joins at true first use. WorkerDelta tracks XRT completion, input/output BO lease generations, errors and timing in the same epoch as GPU events.

The fork uses HIP stream host callbacks to enqueue NPU jobs on a dedicated worker thread and later wait on stream; this is source inspiration only, and our CIR may use a cleaner event bridge after proving XRT ordering. A timeout invalidates provisional results; it cannot imply host has stopped DMA, so buffers remain quarantined until safe. TaskProgram's ASR/audio/image/video stage orchestration reuses EngineCore, worker and memory ownership but has **stage-specific iteration and media output**, not always next-token speculation.

## v4 refinement — session owner, device-native StepProgram, output sequence

`EngineCore/TaskCoordinator` is a **single logical writer per TaskSession or bounded shard**, not one mandatory global request queue, memory allocator and CPU synchronization chokepoint. Its responsibilities are task cursor, attempt epoch, request policy, **StateRoot Publish**, and ordered committed output. ResourceArbiter reserves; device-local pools allocate. DeviceWorker reports `Submitted/Complete/Failed` with ticket+epoch+prepared candidate and readiness events, never publishes state.

A compiled `StepProgram` can chain target model steps, device-side route/group/verify/sampler and state update without an unconditional CPU transfer of routing IDs/sampled token at each step **only where sampled/grammar/causal effects have been validated**. Host TaskDriver is polled at real stage/output boundaries; a correct slower host-loop variant remains. Two in-flight scheduler steps are not legal when next inputs depend on uncommitted results unless StepProgram contains that dependency.

`CommittedOutput(session,epoch,seq)` is emitted only after PublishRoot, with slow client backpressure. A dropped network ACK means client receipt is unknown; offer at-most-once live **attempt** and optional cursor replay/dedup, not default exactly-once delivery. `WorkerDelta` must not contain raw huge tensor history or unversioned GPU pointers.

## v4.1 — DeviceWorker may submit a compiled graph or static pipeline

`EngineCore/TaskCoordinator` prepares **one WorkEnvelope pointing to a qualified closed region**. A native CPU/GPU worker launches direct kernels; a graph-only accelerator worker may submit **one AOT compiled model stage**; a static dataflow worker may launch a whole placed fabric program; each returns **Ready/Failed** with actual completion/event and prepared state. **Do not reinterpret a graph internally into hundreds of host StepPlans.** Device program owns its micro-schedule; coordinator owns queue fairness/admission, TaskDriver stage boundary, cancel epoch and StateRoot publication.

Compilation/cold load should be off request hot path and controlled by runtime ShapeGuard/ProgramLifecycle. A graph state must be private/provisional, snapshot/recompute capable or explicitly stateless. Opaque vendor inference API not exposing state/math belongs to TaskServiceAdapter, **not** an existing WorkerDelta pretender. The result/ModeReceipt includes executable kind, compiler/program version, shape mode, physical topology, state bridge and fallback reason. Same CPU/GPU requester remains functional without cloud TPU/wafer vendor SDK.

## v4.2 — TaskCoordinator and authorized remote RegionWorker

A `StepPlan` can address a **local DeviceWorker** or a **remote authorized RemoteRegionWorker** executing the *same* prepared semantic-region program. TaskCoordinator owns the request/session/TaskDriver, current StateRoot, placement epoch and committed output; a peer never becomes a second model-server API/sampler/session-root owner merely because it computes layers. A receiver validates cluster peer identity, model program/quant/numeric/StateCompatKey, attempt/placement epoch, bounded buffer shape and reserved resources, then executes into **provisional per-layer state**. It returns a typed Ready/Failed result and next activation/state shard handle; no output before all required peer cohorts Prepare and coordinator PublishRoot.

**Pipeline layer with local cache:** a remote stage holds its layer weights/KV/recurrent state across tokens and receives only activation tensors at boundary; no whole model/state transfer per token. **Disaggregated operator:** remote attention on KV owner and FFN/experts elsewhere incurs per-layer query/result RPC and must be costed/shape guarded. **Remote prefix cache or P/D:** bulk state hydrate/transfer is a separate asynchronous worker operation with full hybrid state ABI and recipient reservation. All routes share the same ModelDefinition, TaskDriver, resource/StateRoot semantics and ExecutableBundle.

Peer operations carry `SessionEpoch, PlacementEpoch, StateShardGeneration, OperationId` with idempotent duplicate handling and transport backpressure. If peer stops/responds late, fence old attempt and maintain previous committed root or safe replay; do not assume network exactly-once ACK/commit. Joining a tunnel does not authorize sensitive tensors or arbitrary remote code. [Remote protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md).

## v4.3 — one shared EngineCore under built-in model router

At ingress, `ModelRouter` verifies a named alias/profile and immutable checkpoint+quant+adapter+task/numeric identity. `ModelResidencyManager` prepares/warm-loads a resident model through the **existing global ResourceArbiter and DeviceBackend pools**, with bounded request waiting and fairness; `TaskCoordinator` then runs the same `TaskDriver/StepPlan/WorkerReply/Prepare→PublishRoot` as direct single-model CLI. A model-serving API route must **not** construct a second model-specific EngineCore, global KV allocator, token sampler or per-output HTTP proxy. The alias/model resolution is request-level, not per-token.

For multi-model concurrency, model-resident immutable weights/compiled bundles can be shared by many session roots; TaskCoordinator per session/shard owns mutable KV/indexer/RNG. Native batching combines only compatible admitted model/adapter/numeric/task shapes, not arbitrary different models. Per-model fairness/cold-load queue prevents one giant cold artifact from starving small warm interactive models. Replica balancing chooses a compatible ready deployment using physical budget and state/prefix/cache affinity; active state requires safe migration/fencing, not round-robin.

Per-model profile revision/source/quant/template/numeric mode is pinned before token rendering. A changed profile may require new deployment or safe session restart, not in-place mutation of compiled device input buffers. API protocol adapter maps `/v1/chat/completions`, embeddings, and optional Responses into canonical TaskRequest, returning **only committed** streamed events with model ID/usage. Admin catalog/load/unload endpoints have stricter access than generation. [Router contract](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md).

````

## Source: .agents/docs/EXAMPLE_EXECUTION_TRACES.md

Original UTF-8 bytes: 19718; SHA256 `46ad6832550ee6a070afa9567ea096b9a0b3569be2cd4ccbb788690f9faaaafb`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Design walkthroughs — concrete execution and fallback traces

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


These are **target behavioral examples**, not implemented traces. They force the system design to account for real effects across model, compiler, device, memory and serving subsystems.

## Example 1 — Dense Llama-like model on unsupported GPU specialization

1. User chooses known GGUF with valid tokenizer/config and model SHA.
2. Loader creates artifact/tensor map; parser checks dimensions, quant block, RoPE/heads.
3. Model recipe yields Semantic IR: embed, norm, QKV, causal GQA, residual, FFN, LM head.
4. Capability probe sees generic Vulkan backend but no architecture-specific quant MMQ.
5. Compiler selects portable attention and tested generic quant GEMV/GEMM; plan report names unsupported fast kernels.
6. ResourceArbiter computes dense weights+KV+scratch/OS headroom and reserves memory; load can refuse if model doesn't fit rather than silently changing quant.
7. Runtime prefills prompt in bounded chunks, builds KV state; decode uses direct compiled launches, no IR interpretation per token.
8. Target logits sampled with configured policy; frontier committed; HTTP stream emits.
9. Midstream cancellation invalidates provisional step and waits for events before buffer release.
10. Benchmark outcome measured versus llama.cpp on same GGUF; general correctness can be achieved before parity.

## Example 2 — Qwen3.8 Flash-Next on 128 GB gfx1151

**Input:** Qwen3.8 checkpoint with known qwen4exp recipe, supported quant, large n-gram table stored on local SSD, optional MTP; no automatic quant downgrade.

**Model load:** verify auxiliary n-gram token/row map, train-compatible MTP weights, 48 layers with GDN/QSA and 512 top-10 experts where config says so; allocate dense/expert and recurrent/attention/indexer pools using OS-safe global headroom. For model whose main weights and expert set fit, choose resident fast path. Separate SSD row reader for huge lookup table; a 95 GiB file does not become a 95 GiB resident mapping.

**Compile:** match fused norm/quant/projection/GDN and sparse attention regions; choose gfx1151 wave32 quantized grouped kernels for prefill and fast decode; verify MTP state variant; fallback to CPU reference on unsupported convolution op only if conversion/synchronization legal and benchmark shown acceptable.

**Prefill:** chunk prompt, hash token n-grams and asynchronously read needed embedding rows; device processes GDN, QSA and MoE using large-row specialized kernels; update provisional then committed sequence state; reuse previous immutable prefixes if compatible. Large prefill scratch leased and returned after GPU completion.

**Decode:** single-row quant MMVQ, GDN step, indexer and QSA selection execute GPU native without CPU route checks; trained MTP may propose 1–3 rows; target verifies, commits accepted states and drops others. If low acceptance, controller backs off without changing target sampling distribution.

**Multiple clients:** 4 unrelated requests may route into overlapping experts; route results grouped device-side; execution plan chooses grouped MoE where measured beneficial. Each GDN/indexer history remains per session. Memory manager reserves all live per-session states before admission.

**Metrics:** actual external same-checkpoint/quant comparison with Gufo and a closed tuned reference where legally runnable, fallback plan correctness; be explicit if that reference uses a different quant.

## Example 3 — GLM-5.3 Flash on 92/96 GB Strix Halo

**Artifact:** 321B-class GLM using a tested Q2 recipe, quant weight manifest. Model's 42 routed layers×288 experts cannot all necessarily fit beside dense/KDA/sparse MLA context and OS budget on smaller RAM.

**Resource plan:** commit dense, shared and recurrent/KDA/MLA structures; compute physical RAM reserve, prefill scratch and per-session frontier. Only remaining budget assigned to GPU resident expert slots. Missing expert blobs reside in file/mapped source; staging is bounded. Do not trust large HIP-reported GTT as free physical RAM.

**Decode layer:** device router selects top-8, consults table of `(layer, expert)->(slot,generation)`. Resident experts run immediately. On miss, service event requests exact blob, async read and safe slot replacement after readers complete; GPU waits at legal dependency, not arbitrary kernel synchronization. If transfer is expensive and CPU lane has demonstrated advantage, device/CPU work overlapped on discrete hardware, not assumed helpful on UMA.

**Context:** KDA fine-grain state and NoPE latent sparse MLA require exact position state. If a fallback attention kernel expects expanded KV, compiler performs explicit latent→expanded conversion at a legal boundary; otherwise reject candidate before launch.

**Perf:** compare cold and warm cache, stream and resident capacity, read miss stats and output correctness; Maya/DwarfStar quant recipes must be aligned or labeled incomparable.

## Example 4 — Shared-prefix coding agents with tool calls

- Eight clients submit 40K identical instructions/repository context, then diverge.
- Prefix cache key includes model SHA/quant/tokenizer/rendered history/pos policy and all modality fingerprints; load an immutable snapshot containing KV, GDN/indexer state and token-history.
- Each agent forks from immutable state, private mutable tail, independent sampler RNG.
- Tool JSON output and hidden reasoning retained in canonical token history; cannot retokenize changed formatting and claim exact prefix hit.
- One client changes images: fingerprint differs and affected multimodal spans/prefix state invalidate, while unrelated immutable prefix can be reused if valid.
- Another client cancels after one emitted token; no speculative uncommitted state survives. Unused leased scratch returned.
- Measure actual task latency/TTFT versus replaying 40K tokens independently.

## Example 5 — Training-free novel model op arrives

- New safetensors config declares operator `FooRecurrence` absent from registry.
- Loader recognizes artifact but returns `UnsupportedOperator(FooRecurrence)`, does not pretend ordinary attention works.
- Contributor adds a Rust reference state transition, tests and typed op schema; CPU model now functionally runs.
- Compiler generic GPU lowering added for supported backends; remaining known ops use existing extreme kernels.
- Later `gfx1151` fused kernel added; model semantics and server do not change.
- Offline quality/benchmarks validate and promote performance-qualified hardware cell only when evidence exists.

## Example 6 — Distributed TP failure during speculative verification

1. Two peers agree model/quant, stage graph and token frontier p.
2. Draft proposes 5 tokens, target verification begins over distributed expert/attention partials.
3. Worker disconnects after sending only some partial outputs.
4. Collective fails timeout/epoch check; no rank publishes frontier p+5 or emits proposals as accepted.
5. Quarantine incomplete buffers, close route; rebuild both peers from a common durable committed prefix p on a fresh handshake.
6. TCP/RDMA remoting must be explicitly trusted/authenticated per policy; no automatic cleartext public binding.

## Example 7 — Why compiler fusion might lose

A fused KDA + MoE kernel reduces launch count by 20 but spills registers, reduces occupancy and requires large shared scratch. Its per-operator microbench looks faster, but complete multi-client HTTP test reports worse P95 and no tokens/s gain. Retain the simpler plan as default, document rejection and leave candidate behind a lab build if useful for future GPUs. Numeric exactness is a separate gate from speed.

## Practical invariant crosswalk

Example 1 tests general fallback; example 2 tests model specialization and sampler rollback; example 3 tests physical memory and expert leases; example 4 tests full hybrid-prefix state; example 5 tests versioned semantic extensibility; example 6 tests distributed atomic frontier; example 7 tests proof-based plan selection.

Implementers should turn these scenarios into acceptance/integration fixtures as phases become active, preserving raw logs and actual performance evidence.

## 2026-10-10 integrated v2 addendum — async CIR and modern extension traces

**Trace 8: compilation and no hot-path IR**
1. Known model/quant tokenizer imported and validated into ModelRecipe.
2. SIR declares pure tensor ops + state-effect edges.
3. RIR optimizes legal subgraphs and reference fallback; KIR picks Rust-target CUDA/HIP provider candidates by shape/dtype and numeric contracts.
4. CIR assigns device streams, copy and IO timepoints, buffer lease generations, state-publish barriers, scratch lifetime.
5. Immutable executable bundle validated/cached. EngineCore schedules requests using StepPlan identifiers; GPU launches direct kernels/graph without walking SIR every token.
6. New driver invalidates code cache, **not** semantic model or trusted tokenizer.

**Trace 9: generalizable hybrid prefix intersection**
- Prefix state groups: full KV, sliding window KV, GDN recurrent matrix, compressed sparse QSA indexer, ngram token history and grammar RNG.
- Cache returns 50K token hash match, but compressed rows finalize only at a 2K-aligned frontier and one domain has only 44K valid state.
- Coordinator chooses a common legal frontier (at/before minimum valid constraints), restores compatible domains and recomputes remaining 6K in proper token order; no incorrect "50K hit" reported.
- COW fork to 8 agents remains immutable; adapter/image fingerprint invalidates incompatible branch state.
- Read source: `STATE_CACHE_AND_TRANSACTIONS.md`, `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`.

**Trace 10: new hardware with no model edits**
- A new NVIDIA target supports a custom Rust CUDA tile-kernel matmul but has no specialized KDA.
- DeviceProvider reports exact tile support; KIR compiles selected GEMM/projection regions, generic CPU/GPU KDA reference region at compatible conversion; CIR adds copy cost and context/phase plan.
- If KDA is too slow but correct, capability shown `functional, not qualified`. If no safe state conversion/kernel, reject selected backend for that model, then consider CPU whole plan.
- No changes needed to model recipe, tokenizer, tools or HTTP.

**Trace 11: async expert miss and cancellation**
- GPU route selects missing expert; CIR emits miss request with `model/layer/id/generation`; host reads exact immutable blob to pinned buffer under budget.
- User cancels while SSD read in flight; cancellation epoch invalidates dependent provisional step, but staging/slot leases remain until file/device completion. Completion with stale epoch is discarded, not published into a reused expert slot.
- EngineCore emits no further uncommitted output. Other unrelated sessions use hot expert cache, unaffected.
- A later request can resume from last committed full hybrid snapshot or from checked canonical token replay.

**Trace 12: an apparently fast kernel rejected**
- Grouped MoE kernel shows +12% isolated shape speed but full end-to-end at concurrency 8 worsens TTFT p95 by 18% due resident expert thrash and snapshot scratch pressure. Artifact/tactic remains in tuning lab but default KernelPlan retains simpler kernel; mode receipt names reason. No weakened numerical contract or silent precision change.

These traces are designed assertions, not evidence that zerogravity is already implemented; convert each into mock+real integration tests at the appropriate phase gate.


## 2026-10-12 trace 13 — Qwen cooperative prefill on an optionally qualified XDNA2 NPU

1. Device probe sees Strix Halo Radeon + XDNA2, XRT driver and exact matched firmware/power/FCLK qualification. If absent/unsafe, choose GPU-only plan and never wake NPU or modify host kernel settings.
2. ModelRecipe declares regular GDN z, QSA query/gate and shared FFN projections. RIR recognizes independent GEMM output row spans and the applicable 4096×2560×1024 BF8 program, preserving all source Q8_0 weight/quant metadata and numerical policy.
3. ResourceArbiter reserves pinned A/B/C and immutable NPU prepacked weight slices, instruction/queue and GPU state; CIR builds GPU pack→NPU enqueue versus GPU remaining projection work, with first-consumer external wait. Buffer generations must remain valid across HIP/XRT.
4. GPU continues GDN/attention/route work that does not depend on the NPU rows. NPU output arrives in mapped BO, GPU decodes/consumes in correct original row order, and all GDN/QSA/KV/indexer/sampler effects commit at a single causal frontier.
5. If unsupported chunk/numeric profile or NPU unqualified: immediate GPU-only kernel path, no extra model runtime. If XRT fails mid-compute: abort provisional token state and quarantine BO until truly quiescent; never fill missing rows with zeros or emit an invalid token.
6. Qualified benchmark captures source UD-IQ4_XS with matched held clocks and 16K prompt, actual first-use overlap, logit quality, P95/energy and safe device soak; author 1698→1821 tok/s remains research reference, not our result.

**Trace 14 — new media task:** An ASR model uses audio chunk preprocessing + encoder + model-native transcript generation; a TTS model uses acoustic stage/vocoder; an image model uses repeated denoise steps. Each TaskProgram runs through shared SIR/RIR/KIR/CIR and DeviceWorker but has independent stage frontier/output type, and optional NPU GEMM only after device and quality qualification. This prevents a media model from being forced into a fake text `decode_next_token` loop.

## v4 traces — prepared state, shape fallback and independent tasks

**Trace 15 — atomic all-domain state:** commit root R7 for KV+GDN+RNG. GPU prepares KV R8 and GDN R8; RNG Prepare fails or device event still pending. TaskCoordinator retains R7, emits no new output and quarantines pending buffers until quiescence. Recompute/RNG repair yields all prepared R8; coordinator swaps one root to R8 and then emits `OutputEvent(session,epoch,seq)`. Neither per-domain state nor physical event individually became public.

**Trace 16 — device-native StepProgram:** a text TaskDriver selects a legal compiled decode/verify island, binds persistent batch slot and current root epoch. GPU routes MoE and samples/accepts two provisional next tokens without host synchronizing IDs; event signals completed candidate and TaskCoordinator validates/publishes. If grammar needs host iteration or shape guard fails, choose correct smaller generic host-loop without claiming peak speed.

**Trace 17 — dynamic quant/shape and alternate task:** an ASR encoder stage with ragged audio frames chooses a guarded GPU kernel if `ShapeEnv` satisfies alignment, else CPU/GPU generic; it returns transcript chunks at audio time frontier, not fake target token budget. A future `TrainStep` produces new ParameterVersion without mutating existing inference session R7. Same resource/worker executor, no new model-specific serving engine.

## v4.2 distributed and tunneled execution traces

**Trace 18 — capacity-first layer split:** Node A owns TaskDriver and model layer group 0–k with its KV; authenticated Node B prewarms k+1–n weights and holds B-layer recurrent/KV. A sends only hidden activation at split boundary for each token. B returns next activation plus Prepared shard version; coordinator publishes one StateRoot referring to both after Ready. On B disconnect, stop/fence and recover from last canonical full-state checkpoint or fail clearly, with no wrong token output.

**Trace 19 — remote cache vs compute:** A compute GPU needs long-context attention state on B. Planner compares full compatible KV+GDN hydrate once to A, remote attention query execution on B returning small output, and naive per-step bulk KV remote read. On high-RTT tunneled link, last option fails cost; A can use its own local state/recompute or capacity-first layer placement. Immutable prefix cache hint stale yields verification miss, not wrong root.

**Trace 20 — attention token-range sharding:** three nodes own disjoint causal KV token ranges, compute local softmax max/denominator/weighted numerator and reduce stably to correct attention output, with source semantic positions/masks. A compressed/sparse/recurrent operator without registered PartitionAlgebra refuses same plan. All shard candidate versions/Ready must be prepared before TaskCoordinator root publication.

**Trace 21 — tunnel relay and failure:** enrolled peer path changes from direct to relayed while a layer group executes; current PlacementEpoch remains valid until safe barrier or attempt canceled. A new topology measurement proposes colocated or other peer at committed frontier; resource transfer/fence verifies model/tenant/state. No automatic arbitrary remote shell, no assumed RDMA over tunnel, no split-brain writer or phantom exactly-once network delivery.

## v4.3 multi-model router and one-command user traces

**Trace 22 — direct zero-config run:** user issues `zerogravity run ./local.gguf`. One engine verifies source model/tokenizer/quant, qualifies CPU/GPU, creates RegisteredModel and single ResidentModel lease, executes normal TaskCoordinator/TaskDriver and emits committed token stream; no HTTP service, model download or proxy per token. End closes/retains per configured in-process session and quiescent device leases.

**Trace 23 — many model aliases, limited resident capacity:** API `model:"coding"` resolves AliasRevision 7 to immutable source SHA/quant and finds warm GPU Replica 1. Simultaneously `model:"embedding"` requests an encoder load; ResidencyManager reserves physical RAM/VRAM and graph context, queues bounded if needed, without blocking warm coding output. A third huge model may force safe idle eviction or typed OutOfPhysicalBudget, never silent substitute/quant downgrade or active KV loss.

**Trace 24 — sticky replica, alias hot update:** two identical model replicas B and C each execute concurrent user sessions; a long-prefix session bound to B remains there for cache affinity. Admin changes alias `coding` to revision 8 targeting a different model; active revision 7 retains original tokenizer/KV/StateRoot and correct streamed `model` ID, new request resolves revision 8. Replica C may take an equivalent new session but cannot steal B's state without snapshot/epoch fencing.

**Trace 25 — OpenAI compatibility and native admin:** one `zerogravity serve` accepts local SDK Chat Completions SSE for `model:"coding"`, returns exact qualified model+usage/finish; embeddings endpoint only for actual encoder; Responses stateful/tool/cloud-only field unsupported => typed error. Native `models stop` requires admin capability and drains active model before physical unload. No model output causes server to invoke arbitrary tool, download model, publish private source paths or enroll remote peers.

````

## Source: .agents/docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md

Original UTF-8 bytes: 17081; SHA256 `b51c0c60c5d95b844ebce1c9c9903024b251c5b55c1a461d6b722627cd1bb80c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — Extreme Hybrid Model Recipes, Native MTP and DFlash Source Contracts

**2026-10-15 • Design-only refinement of v4.3 ModelRecipe/TaskDriver and P11–P16.**
**Primary sources:** published serving recipes for Qwen, GLM and MiMo hybrid checkpoints. Source semantics must be re-audited against exact original model checkpoint revision. These projects are prior art, NOT an embedded runtime.

## 0. Governing rule

**ModelRecipe defines source math/state; DraftSource proposes; the target verifies/publishes; KernelCandidate executes a source-qualified closed region.** None owns another ModelRouter, TaskCoordinator, StateRoot publisher or hidden external/Python inference product. Chat/reasoning/tool templates are model-source inputs, not global string hacks.

To add another extreme hybrid model, register exact immutable assembly (original/derived checkpoint, attention and MTP/DFlash/vision sidecars), source semantic operators, session StateDomains, verifier/rollback, hardware kernels and truthful capability/quality receipts. Avoid model-name branching in central queue, HTTP or quant loader.

## 1. Three distinct registered model recipes

| Dimension | Qwen3.8-Flash-Next | GLM-5.3-Flash | MiMo-V2.6-Flash |
|---|---|---|---|
| Identity | separately versioned Qwen3.8 | GLM5 Next | new MiMo2 semantic adapter, not generic Qwen |
| Attention/recurrent | GDN plus sparse QSA indexed attention | KDA plus MLA/DSA and mHC | source-specific SWA and global attention |
| State | GDN, QSA index, full KV/lookup | KDA, MLA compressed KV, DSA index, mHC stream | 39 SWA / 9 global in the tested checkpoint, ring and global KV |
| Routing | source-verified top10 / 512 experts | source-verified top8 / 288 experts | load source top-k/expert counts, don't copy GLM/Qwen |
| Draft | trained MTP + optional lookup table | trained MTP, optional eh_proj precision sidecar | shipped DFlash auxiliary model with its own attention |
| Quant | EXL3 derived pack, PLE, optional vision | EXL3, optional BF16 eh_proj | EXL3 MoE plus source E8M0 block scales |
| Current Zerogravity | planned source math, not full native implementation | planned source math, not full native implementation | **source recipe not yet implemented**, needs detailed semantic registration |

These layer/expert counts are inspected source variants, not universal model-family invariants. Working Qwen3.5 0.8/2B native CPU and selected GPU tensor regions do not prove any of these three whole models.

## 2. Qwen3.8 source recipe

- GDN DeltaNet source short-convolution, decay/gates and causal state update are distinct from dense attention.
- Sparse QSA indexer-selected blocks preserve exact selected positions, mask, position embedding and causal order. Indexer state cannot be thrown away when speculating.
- Routed experts/top-k/shared expert and GatedResidual mixing require exact scale, accumulation and intermediate precision.
- Immutable hashed-row n-gram/PLE lookup table may be enormous and file-backed. Lookups are source parameter access, not KV or draft token histories. Only needed rows are paged and guarded by byte/checksum metadata.
- Trained MTP and prompt lookup are independent DraftSources. The model can choose an explicitly registered policy to combine proposals, while target verification remains authoritative.
- A session root includes GDN, QSA, KV, accepted history, sampler/RNG/grammar and complete state versions, even if MTP or PLE is disabled.

## 3. GLM5.3 source recipe

- KDA fine-grained gated recurrence and short convolution must not be silently substituted with Qwen GDN scalar gate math.
- MLA compressed latent KV, NoPE and DSA index selection require separate source attention operators and mask/position semantics.
- mHC manifold-constrained/Sinkhorn stream mixing and residual order are semantic, not arbitrary fusion reorderings.
- Trained MTP sidecar may include an optional higher-precision BF16 eh_proj. A published source report notes a 64MiB BF16 copy improving draft acceptance; bind source hash, shape, license and numerical/derived-artifact mode. A missing sidecar can be valid but must report lower draft capability; never discover it from an unrelated filesystem path.
- Source-specific mutable KDA/MLA/mHC target history grows with number of speculative rows. Global physical resource admission counts those history slots independently of KV.

## 4. MiMo2.6 required source adapter (currently a documentation gap)

A published MiMo adaptation demonstrates a distinct recipe rather than a generic Qwen decoder. Checklist:

1. Exact source HF checkpoint/config/tokenizer/template/processor and architecture revision; source tensor/head names, expert counts and layer topology.
2. E8M0 block scales as explicit numerical encoding; original source scale values and fused weight readers checked independently of EXL3 codebook.
3. Source-specific optional value-head dimension: a published source reports 128 logical v_head_dim inside 192 padded KV head slots. Logical output is NOT the padded layout; no phantom attention channels in source math.
4. Sliding window attention with exact masks/window, and global paged attention. The 39 SWA / 9 global split is **specific to their MiMo2 checkpoint**; ring/cache format determined by source, not one global magic constant.
5. Native MLP expert intermediate overflow and paired conversion/runtime compensation, verified for the exact derived pack; a blanket activation clamp changes model semantics.
6. Source chat/thinking/tool instructions and generation defaults versioned separately from generic API output streaming; alternate medium reasoning is a declared prompt/profile variant.
7. DFlash model and source attention formula plus auxiliary codebook/embedding, not a substitute trained MTP head or unrelated target.
8. Stateful target verifier/rollback; target-only valid source baseline and exact sampler/correction distribution if claiming lossless speculation.
9. Reject incompatible scale, attention, draft asset or quant data before physical submission.
10. Original CPU scalar/layer/checkpoint source oracle, then exact GPU source/vocab, long-context and independent stochastic tests.

### MiMo layer47 FP16 overflow: semantic and derived-format requirement

A published conversion reports one converted pack's expert intermediate product around **84,000**, exceeding finite FP16 maximum **65,504**. Their specific conversion applies an up-projection /128 rescale and compensating routed scale ×128 later in FP32 (called interm_div=128), making the intermediate safe to store in FP16. This is a **paired derived artifact transform** with source-relative numeric proof, not a universal MiMo or MoE activation clamp.

Test specific numeric-sensitive source rows, check nonfinite intermediates, shape/routed expert paths, full logits, target PPL and tool/coding/reasoning. A near-identical PPL alone can miss a broken expert warp or near-tie difference. No kernel may silently switch to FP32 or saturating FP16 without matching source quality/numeric contract and mode receipt.

## 5. DraftSource families and target-authoritative verification

~~~rust
enum DraftFamily {
    TrainedMtp(MtpSource),
    DFlash(DFlashSource),
    IndependentSmallModel(DraftModelSource),
    PromptLookup(HistoryLookupSource),
    OtherVerified(RegisteredDraftOp),
}
struct DraftSource {
    source_id: VerifiedArtifactId,
    target_compatibility: SemanticAndTokenizerKey,
    algorithm: VerifiedProposalAlgorithm,
    mutable_state: DraftStateSchema,
    max_depth: u16,
    sidecars: Vec<VerifiedComponent>,
}
struct VerifyBlock {
    target_parent_root: StateRootId,
    proposals: Vec<TokenId>,
    provisional_state: TargetStateLease,
    target_logits: TargetLogitReceipt,
    rng_grammar_stop: SamplerStateReceipt,
    accepted_prefix_len: usize,
}
~~~

Drafts are *proposals*, never committed media/text or true model tokens. Target runs actual source mathematical forward for verification, accepts only legal prefix and applies exact correction sampling when stochastic mode requires it. On zero acceptance, cancellation or mismatched sidecar all provisional draft and target branch state is discarded, committed TaskCoordinator root unchanged. An optional no-draft target path remains viable.

### Qwen MTP + n-gram lookup
- MTP is source-trained auxiliary head, with exact feature/tap/position; prompt lookup is deterministic token-history proposal, no logits authority.
- Optional mixed policy must preserve tokenizer/lookup row identity and causal template. Failing hash lookup is not justification to propose arbitrary tokens.
- Auxiliary GDN/indexer/hidden state is versioned/leased; offloading draft to another device includes transfer in measured cost.

### GLM MTP and BF16 sidecar
- Auxiliary eh_proj stored as optional BF16 separate artifact, with hashed checkpoint revision and exact model/head dimensions.
- MTP verify history reserves all KDA/MLA/mHC state that could be pending. Tiny auxiliary weights do not imply tiny per-request state.
- No new decode/sampling semantics just because eh_proj uses different physical storage precision.

### MiMo DFlash attention: exact semantic fields

The shipped MiMo auxiliary drafter cannot simply be loaded as generic DFlash with defaults from a different training convention. Validate:

1. Partial RoPE factor and rotated Q/K dimensions. Full-128 rotation instead of source-defined partial rotation is a semantic error.
2. Per-head sink bias and value-scale, including an explicitly verified absorption of value scale into output projection where source math allows it.
3. **Bidirectional draft block** with source attention left/right window. A source is_causal=false must not be normalized into a plain causal left-only window. A published source reports left=1024 and right=block_size-1 in a MiMo drafter variant.
4. Source-learned mask embedding at draft positions, not zeros or a generic BOS.
5. Hidden target layer-tap indexing convention. A published source reports tap_shift=0 for MiMo where older DFlash defaults use +1.
6. Model-specific draft KV ring and attention cache: draft state cannot alias target or another session.
7. Correct proposal length, confidence policy and shape/precision source constraints; caps and thresholds have versioned semantics.
8. Derived/staged DFlash artifact may be generated offline, but source tokenizer/adapter/drafter hashes must validate before load.

**Critical failure:** a broken draft might still produce correct **target** tokens if the verifier rejects it all. Therefore draft semantic validation needs independent intermediate/attention and acceptance golden fixtures, not merely target perplexity or final token parity.

## 6. Distinct state and clock budgets for hybrid models

Do not equate all attention to one paged full-context KV cache. A source SWA window W with needed lookahead/overlap/guard pages may have a fixed ring of W+bound; source-defined ring size varies by checkpoint. Draft DFlash ring is different from target SWA and global KV.

The MiMo source model reports:
- 39 sliding layers with ~175.5MiB fixed per-slot ring and 9 global layers with ~27KiB/token paged KV at one FP16 mode;
- keeping all sliding layers full would be ~261KiB/token and drastically reduce max context;
- its draft 5 sliding layers use a ~1792-token ring of ~35MiB fixed rather than 20KiB/token.

These are **author measurements for an exact pack**, not engine constants. State spec includes window, head, position, padding, ring generation/ABA, causal/source lookahead, acceptance frontier, snapshot/COW and cross-session isolation.

Physical budget = resident target + aux draft+eh_proj + optional vision/PLE + global KV + target SWA ring × sessions + draft KV/ring × sessions + GDN/KDA/mHC/MLA and indexer state × verifier history + scratch + DMA + file-page cache/OS reserve.

## 7. Kernel/route-aware speculation and batching

For B active sequences and per-sequence proposed depths d_i, total target verifier rows R = Σ(1 + d_i). Time is not merely proportional to R: unique routed experts, per-expert packing granularity, padded tiles, cache/state traffic, cold expert fetch and row-independent launch costs all matter. The published observed regime roughly fit **T_ms=32.4+9.04×R**, but this is only its measured source pack/device/quant/shape.

A packed MoE kernel with 16-row chunks yielded a cliff at batch5 × depth2 = 15 rows (one chunk) versus batch6 × depth2 = 18 rows (two, the second mostly empty). Engine optimization should choose among **measured source-qualified shape bands** and consider both single-stream latency and aggregate committed tokens/s, never hardcode 16-row chunk in SIR.

Model-specific gate:
- evaluate target-alone vs draft+target full wall time, including draft residency/compute, verifier rows, accepted tokens, state rollback, scheduling wait and physical cache.
- dynamic backoff or no-draft on low acceptance; mixing prompt classes matters.
- hysteresis avoids repeated candidate flips; no compile/search of entire cost graph in per-token hot path.
- prompts for chat/prose/code/tools/long context and near-tie numeric cases; sampled target distribution, not just greedy tokens.

## 8. Numerical and behavior policy

| Capability claim | Proof |
|---|---|
| Original source semantic model | official checkpoint/reference per-layer/output/state/logit test |
| Derived EXL3 artifact vs original | held-out original-vs-derived pack error/quality separately |
| Lossless target speculation | target-only vs target+draft on the same derived pack, sampler/RNG/grammar and accepted-prefix state |
| Draft algorithm accuracy | tap/attention/mask/source golden and acceptance across prompt classes |
| Batch deterministic output | source numeric reduction/near-tie tests at B=1,2,4,8, not implied by batching |
| Steering or added reasoning style | named **behavior-changing** variant and explicit user mode, not hidden acceleration |

A published report notes some batched multisequence decode that changes greedy near-tie tokens even with FP16 cache, due to different attention reduction order. Therefore strict equality between batch and solo must be explicitly gated by validated numerical policy; an untested batch optimization cannot advertise deterministic equivalence.

Optional residual activation steering is a model-behavior change with unique artifact hash, affected layer/direction, security policy and measured quality; it is **not a source-preserving fused kernel** and is OFF by default. No automatic removal of model safeguards or hidden transform under quantization.

## 9. Acceptance matrix P11/P13/P14/P15/P16

| ID | Fixture and required assertion |
|---|---|
| HYB-01 | Qwen GDN+QSA exact attention/index/StateRoot against original source |
| HYB-02 | GLM KDA/MLA/DSA/mHC full state and prefill/verify numerical checks |
| HYB-03 | MiMo SWA/global head/value/padding/mask valid; no padded-axis semantic state |
| HYB-04 | E8M0 scale codec and mixed EXL3 source golden; corrupt scale rejected pre-GPU |
| HYB-05 | MiMo transformed layer47 source pack finite and source/derived quality verified; wrong compensation fails |
| HYB-06 | Qwen MTP/prompt lookup/mixed policy exact target accepted prefix and sample RNG |
| HYB-07 | GLM eh_proj optional BF16 sidecar correct model identity and performance budget |
| HYB-08 | DFlash wrong tap/partial RoPE causes source golden failure |
| HYB-09 | DFlash missing sink bias, value scale, mask embedding or wrong bidirectional window causes source golden failure |
| HYB-10 | DFlash accepted prefix 0..k and sampled target correction/rollback full StateRoot |
| HYB-11 | ring wrap and speculation fork/replay 8K/128K/256K with source physical state |
| HYB-12 | four sessions share immutable draft/target weights but never mutable rings, history, RNG or grammar |
| HYB-13 | verifier R=15 vs R=18 and many expert counts; cost candidate changes only when measured |
| HYB-14 | prose/chat/code differing acceptance disables net-negative drafting without changing target |
| HYB-15 | derived quant vs original and draft vs derived-target equivalence reported separately |
| HYB-16 | unknown MiMo source op/DFlash sidecar rejects rather than falsely claiming full support |
| HYB-17 | source thinking/tool/template controls preserved; no hidden behavior prompt |
| HYB-18 | batch B1/2/4/8 near-tie numeric contract explicit, not unconditional byte parity |

## 10. Existing phase mapping, no new engine

- **P11–P14** register exact KDA/MLA/QSA/GDN/mHC/MiMo semantics and state, CPU oracle first.
- **P10/P11** implement verified target + draft ring/global cache and physical credits, not one KV rule.
- **P15** draft family variants MTP/DFlash/small model/lookup with exact verifier and fallback.
- **P16** measured MoE padding/row/expert-aware scheduling and dynamic draft policy.
- **P22** whole-model/derived quality, long contexts, numeric near ties, multiple tasks, device and real sampling.

All remain future design obligations; no new media/text model is implemented by this document.

````

## Source: .agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md

Original UTF-8 bytes: 15167; SHA256 `220b5aeec5b6eb4ac87a19534c3bf0a206407d317255827134f88c76304d0809`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Model artifacts, quantization, tokenizer/template parity and safe loading

**15 October 2026 — VERIFIED SOURCE EXTENTS:** [Unified SSD/resident TensorView](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) binds each immutable source or derived-quant tensor slice to an opened verified ArtifactId+shard+offset+byte length+codec+transform, not a mutable filename. The same logical source tensor can materialize from bounded local SSD/host/UMA/GPU without changing its quant identity. Mixed-size Q2/Q4/EXL3 experts demand exact offset/scale integrity and one global byte budget, not full-pack re-dequantization or size-class cache resets. Qwen n-gram/DeepSeek Engram large row lookup is separately source-indexed and disk-primary, never required whole-table VRAM residency. Immutable GGUF file descriptors survive path replacement; partial/short/unsupported direct IO is a host error **before GPU submission**. See the source research notes, tests SSD-01..07, SSD-17..25.


**15 Oct specialization precision addendum (design, not implemented):** [EXL3/QTIP and derived quantization](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md) specifies packed trellis/codebook block/source scale and optional transforms independent of architecture, **three different identities** (original checkpoint, lossy derived quantization, lossless device packing), reference oracle and mixed per-tensor bitrates. Offline shaped LDLQ sensitivity, source-representative trace and held-out KLD/quality are experimental P20 evidence, not runtime auto-quantization. P03 must fail unknown EXL3 K/codebooks, E8M0 formats, missing sidecars or corrupt scale before physical GPU submission; quality, target speculation and raw weight fidelity are separate receipts. Tests QTIP-01..14 apply.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


## 1. Design constraints

Model format and numerical representation are orthogonal to the runtime's execution layout. Loading one GGUF or Hugging Face safetensors model should not require a Python inference server. However the loader can only execute **known semantic architectures and verified tensor layouts**. Never silently invent formula for an unknown operation or execute arbitrary `trust_remote_code`.

A model artifact fingerprint binds:
`architecture_id + config_semantic_hash + tokenizer/template_hash + tensor_manifest_hash + checkpoint_sha256 + quant_recipe_hash`.
The file's name is not identity. Partial/split GGUF or safetensors shards must be checked as a complete set, with offsets, hashes and duplicate/missing tensor detection.

## 2. Architecture of the loading pipeline

```text
Local path / HF model ID
  -> controlled artifact resolver/cache (explicit network when authorized)
  -> format detection and structural parser (bounded sizes)
  -> tensor manifest + tokenizer/template fingerprint
  -> adapter registry exact architecture validation
  -> SemanticModelIR recipe, state and position schema
  -> physical bytes/quant layout and conversion plan
  -> memory/resource reservation and packed device tensors
  -> executable compiler plan + reference validation if enabled
```

No GPU allocation or expensive network fetch should occur before basic model config/format validation and a credible safe memory estimate. Never use untrusted metadata to allocate `num_layers * num_experts * huge_dimension` without checked overflow/hard limits.

## 3. GGUF

Support recognized GGUF header fields, tensor metadata, alignment and quant schemes explicitly. Each tensor has logical dimensions, stored offset/bytes and quant block metadata, validated against model adapter's expected TensorSpec. GGUF is not automatically an executable graph.

Quant support should be incremental and model-verified: F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q2_K, IQ2_XXS, FP8, MXFP4 and additional types when needed and correctly implemented. Distinguish block format's logical input row length from stored padded stride. A 640 logical dim padded to 768 for an optimizer is not 768 semantic input features.

Format parsers need:
- little/big-endian correctly handled where specified;
- bounded key/value string/array lengths;
- non-overlapping tensor regions (or explicit format-supported alias);
- alignment and seek overflow safety;
- duplicate names and missing mandatory tensors detection;
- consistency with tokenizer/architecture config and manifest;
- checksum/written-partial-file safety and source provenance;
- support for mapped read-only weights when safe;
- clear error for uncommon tensor types rather than fallback to reinterpretation.

## 4. Safetensors / HF

Load explicit `config.json`, `tokenizer.json`/other declared tokenizer assets, safetensors index/shards and optional separate vision/audio/mtp modules. Validate tensor dimensions and shard order with a model adapter. Model-specific parameters such as `rope_theta`, sparse indexer geometry, conv widths, GDN/KDA gates, number of experts/top-k, mHC iterations and native context must be declared exactly.

HF model IDs should resolve to a pinned revision/commit or immutable cache digest. Downloads resume with atomic staging, checksum and bounded disk requirements; `--offline` mode uses local verified cache. Avoid implicit arbitrary external fetch of `image_url` or model repo hooks under service requests.

Hugging Face custom model Python code is not executed; a new mathematical op requires explicit Rust adapter/operator implementation. Use official Python/Transformers only in **offline developer reference QA**, not normal runtime serving.

## 5. Tokenizers, chat templates and agent semantics

Tokenizer parity is part of model correctness. Requirements:
- byte-level / BPE / SentencePiece / other supported tokenizer algorithms with verified Unicode/special token offsets;
- BOS/EOS/PAD and tool call sentinel handling;
- reasoning on/off variants and role delimiters;
- model-specific tool argument formatting and streaming parser;
- exact token prefix versus rendered text: hidden reasoning/tool turns may not match visible client text;
- Unicode normalization and invalid UTF-8 safe behavior;
- multimodal token spans and associated image/audio preprocessing hashes and positions.

Test official model tokenizer vectors, long tool histories, multiple languages/emoji, special-token injection and byte boundary streaming. Never assume OpenAI/Anthropic role schema maps one-to-one to every model's native template; adapters must render exact tokens.

## 6. Distinguish storage precision from compute format

`StoredQuant`: checkpoint's semantic quant bytes, scales and block encoding. `PackedLayout`: lossless reorder of those bytes for device locality, WMMA shape, alignment and group storage; must round-trip or have exact dequant parity. `RequantizedArtifact`: deliberately different precision, with its own checkpoint identity, imatrix/calibration/eval and quality loss; may change output.

A performance benchmark comparing an IQ2 checkpoint against BF16 changes both engine and model-quality conditions. Report quant bit counts/bytes, main/draft/table sizes, working set resident, and quality delta. Never auto downgrade from Q4 to Q2 under memory pressure; request a different file/artifact/policy explicitly.

Tensor role matters: dense always-active projections may be worth higher precision or faster packed layout; routed experts may tolerate lower bit widths; attention/indexers/routers/mHC may be sensitive to quant error. Model-specific imatrix-aware recipes can minimize quality loss but require independent evaluation. No one quant recipe fits Gemma, Llama, GLM and Qwen equally.

## 7. Conversion, calibration and verification pipeline (optional developer tooling)

- Source checkpoint manifest with SHA/model revision and license.
- Convert tensor geometry to GGUF or internal packed representation without changing numerical precision by default.
- Quantization tool may read calibration corpus/imatrix, apply chosen recipe, pack exact types, emit per-tensor quant metadata and audit.
- Verify every output tensor: shape, logical/padded dims, checked file offset, sample dequant error, full conversion manifest, tokenizer match.
- Publish quality table: reference NLL/KL/logit RMSE, first-token agreement, greedy prefix matching, coding/tool/multimodal long-context tests as appropriate.
- Save exact conversion toolchain/compiler revision and source dataset license.
- Avoid distributing copyrighted/unlicensed checkpoint weights or personal prompts automatically.

## 8. Multi-file and huge-file support

Weights can exceed physical RAM and span sharded files. Do not load all shards into an extra host arena before GPU upload. Use lazy mapped metadata and bounded read batches, appropriate readahead, checksums and verified staging. Storage performance needs measurement: n-gram BF16 table row reads vs 11 MiB expert blob reads have different access patterns.

Model switching must release safely after all sessions/events retired. A process may host multiple models only when global ResourceArbiter proves capacity and maintains separate model/checkpoint identities.

## 9. Layout transform and fallback

Compiler can select a device-specialized packed layout for fast Qwen/GLM kernels, but CPU/generic fallbacks require a canonical conversion contract or independently matched layout. Layout transitions need an explicit data movement cost and legal cut point; no direct reinterpretation of compressed vs expanded attention state. Converted tensors cache keyed by artifact hash, quant schema, kernel layout ID and compiler version.

## 10. Adversarial input tests

Malformed GGUF/safetensors, duplicate metadata, unknown type, overlarge dimensions, integer overflow, negative/NaN quant scale, broken shards, symlink/path traversal, truncated n-gram or expert table, invalid tokenizer UTF-8, emoji/special role injection, mismatched MTP sidecar, missing vision encoder, illegal context extension and stale artifact cache. Each must fail without OOM, arbitrary file read or running unsupported inference.

## 11. Open policy items

User-chosen automatic quant selection (if any), default HF network resolver behavior, cache retention, license distribution, trusted plugin adapters, maximum artifact size and default tokenizer library. All remain proposals, not explicit user decisions.

## 2026-10-10 integrated v2 addendum — complete product-model pipeline

An earlier architecture note and implementation blueprint provide many detailed contracts beyond initial file parsing. Adopt their **safe GGUF dialect/type rules, byte-exact tokenizer pre-types, Jinja/minijinja rendering, grammar-FSM/GBNF, content-addressed model store, imatrix calibration and derived acceleration-pack concept**, subject to current license/toolchain checks. See `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md` for consolidated owners and capability matrix.

An optional native expert/quant pack (historical working name "APK") is an acceleration **cache** referenced by original GGUF/safetensors hashes, never the only runnable input. Public extension/name not fixed; avoid Android APK collision. Its manifest can index one-blob-per-expert data, packed CPU/GPU layouts, KernelPlan hash and quality/speed overlays. Source and quant conversion must be independently verified for exactness or labeled as a *new quantized checkpoint*; don't silently choose lower precision to meet RAM.

**ModelStore obligations:** immutable SHA-256 blobs, versioned manifest list, sharded model source, model/LoRA/vision/MTP sidecar identity, resumable atomic downloads, offline mode and reachable-set GC, `--inspect/doctor` errors and privacy-safe prompt cache. All source weights/sidecars may carry separate license conditions. No `trust_remote_code` at runtime. Token IDs/template, adapters, special tokens and modality spans are part of semantic model identity and prefix compatibility.

**Old research defect:** its disk-KV ABI explicitly calls a listed 54-byte collection a 48-byte header and uses an impossible 5-byte string represented as u16. Do not copy/rewrite this broken struct into model loading or snapshot readers. Correct new version requires checked byte-count math, schema hash, collision-resistant identity and negative parsing tests.

**New model integration:** add `ArchDef/ModelRecipe` with exact tensor validation, SIR mathematical operators, CPU oracle and tokenizer/native template; attach to generic backend; only later register specialized KIR kernels. No model-specific `kernel_plan(hw)` inside semantics, no guarantee an arbitrary GGUF/HF file runs when mathematical ops are unknown.


## v4 refinement — independent artifact, semantic, state and bundle identities

Version **ArtifactKey** (checkpoint/shards/quant/tokenizer/template/adapter/sidecars), **SemanticKey** (op/formula/state/position contract), **StateCompatKey** (task/provenance/frontier/quant/adapter/media/tenant), and **BundleCacheKey** (compiled device/driver/shape/numeric/codegen). **Runtime session/batch slot indices must not appear in compiled bundle key**, or cache explodes. Compiled driver handles, GPU pointers and EventIds are never serialized with model weights or state snapshots.

Artifact loader is trusted data parser, not a loader of executable model code. Recognized GGUF/other tensor type does not imply registered mathematical op/TaskDriver support. An exact lossless repack differs from temporary low-precision arithmetic and from a new requantized source checkpoint; source/model quality must identify the difference. Default import lacks arbitrary external code execution and global kernel plugins.

## U-048 — packed quantization and NPU derived-encoding eligibility

[Packed-encoding qualification](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) covers packed affine/codebook quantization, transformed weights,
role-specific recipes, overlays and component sidecars as **source-qualified codec
and assembly candidates**, each with independent parsing and numerical qualification.
Independently validate every supported header/scale/zero/transform/padding and
sidecar binding. Material W8 requantization requires derived-encoding identity
and the permitted numeric/quality contract; native low-bit source bytes remain
immutable. KV compression is independently keyed state arithmetic. Source-known
or parseable formats do not imply model, device, quality or feature admission.

````

## Source: .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md

Original UTF-8 bytes: 377070; SHA256 `e4fed9cebd90a2f4d193fa73f8637c5254944e1ae2e1c295557961ea574c2ebb`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: GPT-6

# Zerogravity — comprehensive agent handoff, 17 October 2026

This is the requested context handoff for the live Rust repository at
`/home/pvl/spaces/abird/src/zerogravity`. It records the original vision, design
evolution, retained decisions, implementation history, accepted verification,
known failures, every currently open plan item and the two interrupted drafts.
Read it progressively; retain a short working TODO while moving through the
technical source references. The copyable entry prompt is
[NEXT_AGENT_PROMPT_2026-10-17.md](../plans/NEXT_AGENT_PROMPT_2026-10-17.md).

**Runtime implementation is PAUSED at the owner's request.** The current request
creates documentation only. Reading this handoff does not resume implementation,
launch a GPU, authorize recovery, acquire models/SDKs or publish anything. A new
explicit continuation instruction resumes the appropriate work within its scope.

<a id="contents"></a>

## Contents

1. [Checkpoint, authority, pause and reading order](#checkpoint)
2. [Original vision and architecture V1–V5.1](#architecture)
3. [Implementation chronology, actual tests and historical recovery](#history)
4. [Current acquisition contracts and failed experiments](#acquisition)
5. [Continuation, paused drafts and every open plan item](#continuation)
6. [Portable reconstruction, workspace and operating commands](#operations)
7. [Complete retained decision register, including later owner instructions](#decisions)
8. [Complete technical document and plan reading catalog](#catalog)
9. [Handoff validation and known limits](#handoff-validation)

<a id="checkpoint"></a>

## 1. Read this first: authority and checkpoint

### 1.1 Current source and working state

Recorded from live inspection on17 October2026:

| Item | Recorded fact |
| --- | --- |
| Canonical repository | `/home/pvl/spaces/abird/src/zerogravity` |
| Main branch | `main` |
| Verified implementation HEAD | `f197c9cf4a44e06a3c1cf3c532855545bed945fa` — explicit GGUF filename quant selection |
| Previous implementation | `f22f75972e63ea70771c59fbce065e371d5f29d4` — pinned metadata-only candidate discovery |
| Local tracking ref | `origin/main` at `fa6affa425089fdf9a15bfa10b330e03f738fec8`; MAIN was22 commits ahead,0 behind. This is a local ref observation; no new remote fetch/push was performed for this handoff. |
| Reachable main history | 77 commits, beginning with rewritten root `be95424`; it is not the original chronological implementation graph. |
| Last implementation acceptance | 975 focused executions, four strict artifact/CLI configurations, full isolated gate934 Rust passes/0 failures/69 intentional ignores/205 suite executions plus18 namespace tests, Nix source-integrity and both original trained-model cache routes. |
| Original trained routes | Separate exact calls: cache/direct17.60s; HF import23.70s. Both approved models, full248320-vocabulary logits, all42 state domains, two tokens/nonzero state, sessions/source/lease/mutation/reclamation exact. These are individual test durations, not a fabricated combined invocation or new8K/GPU qualification. |
| Signature | `git verify-commit f197c9c` succeeded during wrap-up. Signed commits remain required. |
| Dirty state before this documentation task | Two routine post-commit rolling TODO updates in `.agents/plans/TODO.md` and `.agents/plans/model-acquisition/TODO.md`, plus the interrupted filters run. Runtime source was committed; no staged changes. This task adds documentation and portable draft context, so do not expect the working tree to be clean. |
| Agent/process state | All implementation agents stopped; no MAIN Cargo/rustc process or process using either draft worktree remained at pause. Recheck rather than assuming this remains true. |
| Protected document | Root `README.md` is user-authored and must remain byte-identical unless explicitly requested otherwise. |

The earlier connector launches at `/home/pvl/spaces/abird`; through that connector
use `./src/zerogravity` as the working directory. In the current local shell the
working directory is already the canonical repository. Resolve it explicitly
before any mutation.

### 1.2 Authority order

1. The latest user instruction, including the current implementation pause and
   the immutable root README boundary.
2. [AGENTS.md](../../AGENTS.md) and [TODO-SPEC.md](../plans/TODO-SPEC.md).
3. [The plan index](../plans/TODO.md), the relevant live plan `TODO.md` and its
   current design/technical owners. These contain actual completed and open work.
4. The current source, tests, configuration and actual qualified source/device
   profile. A source identity, model hash, numerical limit or physical ownership
   boundary is not replaced by a handoff description.
5. This dated context snapshot, retained historical decision rows and old design
   documents. They explain reasoning and prior work; their historical status
   prose never overrides a later verified checklist or owner decision.

The handoff is a documentation snapshot, not a second live task ledger. Work
tracking remains one index, one TODO per plan and one own TODO per active agent.
Do not revive mandatory generated handoffs, evidence archives, staged blob/tree
receipts, per-file SHA inventories or archive-sealing duties. Actual content
hashes and numerical verification remain implementation requirements.

### 1.3 What changed since the original H4 takeover request

The original request began at historical handoff2125612, implementation0cad60a:
H4 was still hybrid, with566 physical completions across two tokens. H5 GDN
recurrence, attention/KV/RoPE/softmax/value, FFN/residual/embedding math, dynamic
activation quantization, resident state, device token feedback and native vendor
profiles were subsequently implemented and accepted within their documented
scope. Starting H5 again would discard substantial completed work.

The accepted baseline now includes original-model resident GPU execution through
512 consumed tokens and native CPP NVRTC/HIPRTC profiles. The old LLVM23 blocker
applies to a separate unqualified LLVM code-generation path; it is not a reason
to claim native CUDA/HIP remain universally unavailable. Local Renoir/gfx90c is
held after a later initialization/reset incident; historical Vulkan success does
not authorize using it now. CPU functional8K success, independent8K numerical
drift, resident512 qualification and deferred16K/32K are distinct scopes.

The latest implemented work is model acquisition, not another GPU phase. The
layout and thoughtful concurrency review are complete; no left-right migration
was justified. Finish acquisition convenience/provider gaps in bounded verified
steps when resumed, then the remaining CPU-first new-model and GPU optimization
plans. The full project is not complete.

### 1.4 Reading sequence

- Read this opening, the architecture and implementation chronology below, and
  the exact continuation/paused-draft section before editing.
- Read the live plan index, current acquisition TODO and
  [acquisition design](../plans/model-acquisition/README.md), including both plain
  completed-TODO archives. The handoff includes their source locations and the
  current exact open-item snapshot.
- Read the decision register snapshot below, especially its new U-056…U-070
  consolidation and the supersession notes. Historical U/P/O rows are retained
  to preserve reasoning, not to reopen obsolete evidence tasks.
- For core work, read the minimal execution spine, semantic/task/compiler,
  state/transactions, resource/accelerator and invariants contracts catalogued
  below; then inspect their actual crate owners.
- For a selected later plan, read that entire TODO and its referenced technical
  specifications/research before implementation. The complete document catalog
  ensures the remaining vision is not reduced to the latest acquisition task.
- Revalidate live HEAD/worktree/source/input/process/device state and relevant
  existing tests. Use one MAIN Cargo owner. Documentation checks do not qualify
  Rust runtime changes, new models, real devices or performance.

### 1.5 Historical recovery and its exact limits

The owner explicitly approved retiring completed raw evidence and replacing it
with concise tested completion lists and Git provenance. The repository was also
rewritten. Labels such as6d74f76 and bfe672c still name recorded approvals in plan
text, but their Git objects do not resolve in the current repository. The
original33-commit Git graph and later129/191-source aggregate versions were not
recovered as current standalone objects. Do not invent their ancestry or tell a
new agent to run `git show` on those unavailable labels.

Substantial earlier text DOES survive: reachable7e61cb6 contains the retired
multimodal aggregate, which embeds a complete older94-source implementation
handoff through U-040/P07-G3b. Its executive brief, early CPU/numerical/P07 phase
reports, older agent prompts and original design bodies can be read by the exact
source ranges catalogued below. The fourteen later deleted files plus one
replaced pre-product README body are also readable at7e61cb6/be95424. The
standalone early paths are absent; their absence does not imply all their text
is lost. Independent review caught and corrected that initial overly broad
conclusion before acceptance.

The retained canonical specifications, decision register, baseline/checklists,
current source and this recovered nested history together preserve the available
context. The history part separates positive recovery, unavailable Git objects
and unrecovered later versions; it includes all94 nested source reading ranges.
This handoff does not claim byte-complete recovery of the later129/191 snapshots
or the original33-node graph, and does not restore obsolete evidence/archive/SHA
maintenance as a duty.



<a id="architecture"></a>

## Original vision, design evolution and architecture

Documentation-only architecture context, integrated for this handoff. Inspected baseline: signed
`f197c9cf4a44e06a3c1cf3c532855545bed945fa` (2026-10-17). Implementation remains
**PAUSED**. This document records retained context; it does not authorize resuming
source work, running tests, downloading inputs, changing devices or publishing.
No implementation tests were run for this audit.

### 1. Authority and limits of the surviving record

Read [AGENTS.md](../../AGENTS.md), the [TODO spec](../plans/TODO-SPEC.md)
and [plan index](../plans/TODO.md) first. Latest user instructions govern;
current source plus applicable actually executed tests establish implemented
behavior; the plan TODOs distinguish accepted checkpoints and unfinished work.
The [engineering specification](../../docs/ENGINEERING_SPECIFICATION.md)
(ES-00–ES-15 and INV-01–INV-25) is the integrated design contract. The
[whitepaper](../../docs/WHITEPAPER.md) explains the argument, hypotheses and
limits. Neither makes every designed feature executable. Old phase statuses,
archive duties and illustrative Rust ABIs are historical context, not additional
tracking requirements or permission to act.

**Recovery distinguishes text from Git objects.** The owner-approved baseline
label6d74f76 and cleanup labelbfe672c survive in current TODOs but their Git
objects do not resolve in the rewritten graph. Original33-node ancestry and the
later129/191-source aggregate versions were not recovered. Standalone early
phase paths were already absent in rewritten rootbe95424; however the reachable
multimodal aggregate DOES embed an older complete94-source implementation
handoff throughU040/G3b, including early CPU/numerics/P07 reports and prompts.
The separate recovered-source catalog gives exact reading ranges. Fifteen later
retired bodies also have verified readable paths. Absence of a standalone file
must not be described as loss of every embedded copy. Old commands/results are
historical records, not newly executed tests. Initial blanket missing70-report
wording was corrected during independent review before handoff acceptance.

The [decision register](../plans/DECISION_REGISTER.md) preserves U-001–U-055,
P-001–P-096, O-001–O-074 and A-001–A-006; its dated owner consolidation adds
U-056–U-070. User requirements, assistant proposals, assumptions and open choices
are different categories. A proposal does not become a user decision through
repetition. Supersession changes the specified dimension, not every earlier
requirement. The multimodal-local ledger has its own historical U-040/U-041
namespace; cite that ledger explicitly rather than conflating them with the
canonical register's different U-040/U-041.

### 2. Beginning-to-current architectural evolution

| Stage | Original concern and retained result | Refinement that governs subsequent work |
| --- | --- | --- |
| V1, 2026-10-10 | One Rust-native semantic engine for ordinary models and extreme large/hybrid models; portable fallback plus hardware-specific kernels, exact state and source formats, easy local use and efficient serving. Research/design only at this date. | Semantic model truth must survive optimized lowering. Performance ambition is not a measured universal win, percentage target or deadline. See U-001–U-012 and recovered initial handoff in §11. |
| V2, 2026-10-10 onward | Combine previously separate compiler/research directions into one semantic/compiler/execution spine. Model recipes specify source math, not hardware kernel plans; include source formats, tokenizer/templates, batching, speculation and whole-state transactions. | SIR/RIR/KIR/CIR were useful compiler views, not four independent runtimes. See P-018–P-025 and [semantic IR addenda](SEMANTIC_IR_AND_COMPILER.md). |
| V2.1, 2026-10-11/12 | Heterogeneous GPU/NPU prefill and distinct ASR/TTS/image/video task loops become explicit. Partial output-row projection splitting is a placement optimization; it must not alter routing or recurrence. | Distinguish source encoding from temporary compute encoding and CPU/GPU-twin parity from source-model quality. Hardware fabric/power/hang risks require separate physical qualification. U-016, P-026–P-030. |
| V3, 2026-10-12 | Neutral, composable ModelDefinition/SemanticOp/StateDomain/TaskProgram/region/device/resource contracts replace project-named ownership and cloned model-specific engines. Preserve attributed research and file-specific licenses. | Broad workflow privileges, ambiguous domain commit ownership and an overgrown permanent IR stack were critically reviewed and narrowed in V4. U-017/U-018, P-031–P-035; recovered V3 snapshots are historical comparisons. |
| V4, 2026-10-12 | Minimal execution spine: two enduring representations, trusted minimal task control, prepare-only state domains, one root publisher, hierarchical physical admission and real completion before release. | The [minimal spine](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinement](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) supersede ambiguous older ABIs. P-036–P-045. |
| V4.1 | Accelerators participate at their natural closed execution region: kernels, captured graphs, AOT graphs, static dataflow or qualified managed sessions. | One model semantic program; no device-specific semantic IR or opaque vendor API masquerading as a native backend. U-019, P-046–P-055. |
| V4.2 | First-class multi-machine execution with independent compute, parameter, mutable-state and immutable-cache placement. Transport/topology and directional costs become first-class. | One semantic/state/resource authority across peers; remote work prepares private state, not independent roots. No automatic peer enrollment or transparent HA. U-020, P-056–P-070. |
| V4.3 | Native model routing, profiles, replicas and residency above the same executor; simple one-command local run/serve/library experience plus explicit advanced control. | Exact source/quant/profile resolution once per request, compatible sticky session ownership, original physical arbiter, no per-token router/proxy or silent model substitution. U-021, P-071–P-086. |
| V5, 2026-10-14 | Unified typed multimodal values, model assemblies, task drivers, state/media frontiers, quantization and finite-choice outputs. | MM-01–MM-11 extend V4.3; synthetic MM0 contract checks do not qualify trained audio/image/video execution. |
| V5.1, 2026-10-15 | Universal task-family and verified-assembly design, synchronized media, realtime sessions/playback, soft QoS and specialized source/placement policies. | Preserve original owners; expose unknown playback/provider outcomes and open policy choices. Do not claim every proposed interface or model is implemented. See §7. |
| Implementation and later acquisitions, through 2026-10-17 | Packaging → original CPU truth → source-exact resident GPU → shared-model/session/layout review → contention-first rejection → explicit acquisition/cache/source bridge → native providers/Xet and explicit filename quant selection. | Actual implementation is narrower than the architecture. Current accepted and unfinished scopes are in §§8–10 and the live TODOs. User pause remains binding. |

The current [master plan](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md)
§§1–15 is the V4.3 synthesis, not a frozen V1 snapshot despite its filename.
Its six concerns—model math, task control, compilation, execution, physical
resources, and service policy—remain separable. The
[reference architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md)
and [composition contracts](COMPOSITION_AND_EXTENSION_CONTRACTS.md)
explain extension points without authorizing parallel engines.

### 3. Exact ownership, representation and state publication

A verified artifact/config/tensor/tokenizer recipe builds one exact
**SemanticProgram**. A compiler produces **ExecutableBundle** objects containing
prepared code/regions, guarded shapes, bindings, local DAG readiness and resource
requirements. RIR/KIR are transient lowering views; CIR is executable dependency
organization, not a semantic commit owner. There is no per-token interpreter of
a universal workflow graph and no family-specific server required for a new
model. See [task/compiler refinement](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md)
§§1–5 and [semantic IR](SEMANTIC_IR_AND_COMPILER.md), V4 refinement.

The [minimal spine](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md)
§§1–10 defines exactly five owners: immutable semantics, immutable compiled plans,
logical session/TaskCoordinator (including TaskDriver), global physical resource
authority, and device/I/O execution authority. TaskDrivers express model-relative control:
token steps, denoising, audio segments or other bounded loops. They do not acquire
arbitrary workflow/tool/network privileges. Actual source currently implements
bounded pieces of this spine; the generic architecture is not fully delivered.

A step begins against an expected root/epoch; kernels and domains build private
proposals, COW versions and prepared outputs. Every participating domain must
validate compatibility/frontiers and prepare successfully. The **sole
TaskCoordinator publishes one root** only after genuine required device events.
A domain does not independently commit a partial model step. Output becomes
committed before external delivery; delivery attempt/ack/playback are separate.
Timeout, aborted Future, canceled I/O or drained socket is not proof DMA stopped.
Unknown completion retains actual handles and charge, fences the affected
service/card authority and refuses stale publication. See
[state/cache transactions](STATE_CACHE_AND_TRANSACTIONS.md)
§§3–8/V4 addendum and the minimal spine §§2–3, 7–10.

Compatibility is the meet of all relevant domains, not just a KV prefix: exact
source/quant/adapter generation, tokenizer/template/history, positions, attention
and recurrent/indexer state, lookup/ngram state, sampler/RNG/grammar and media
frontiers where applicable. Replay must reproduce the same declared numeric and
state profile; a portable trusted diagnostic checkpoint is not automatically
secure durable product persistence. Cross-source restore refuses.

### 4. Source, arithmetic and quality are distinct contracts

Recognized container → known semantic recipe → runnable selected backend →
numerically qualified → performance-qualified are separate statuses. Parsing a
GGUF/safetensors or recognizing metadata never licenses arbitrary custom model
code, unseen operators or new tokenizer behavior. A missing semantic op fails
closed; fallback requires a real legal reference implementation and sufficient
resources. New recipes should reuse exact existing ops rather than new engines.
See [model semantics](MODEL_SEMANTICS_AND_COMPATIBILITY.md) §§6–8,
[product pipeline](PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md), and
[immutable sharded-source specification](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md).

Immutable source identity has one issuer. A pathname, readonly descriptor,
readonly mmap or private-COW mapping does not freeze another writer's file.
The cache bridge hashes the **actual owned bytes** read through retained managed
descriptors, checks exact length and source metadata, reparses those same bytes
and retains admitted immutable-source ownership. Normal and cache preparation
use the same mathematical engine. Current executable source cap remains **4 GiB**;
large admitted BF16/F16 shards, complete index/header/name/dtype/offset/shape/hash
binding and safe host staging remain future work. Remote/provider acquisition
is data acquisition, not ModelReady or inference support.

[Numerical quality](NUMERICS_QUALITY_AND_BENCHMARKS.md) separates
formula/operator agreement, declared compute arithmetic, checkpoint/task quality
and distribution/state exactness. Scalar/FP64 references, pinned independent
source, Rust CPU, optimized devices and serving are different oracle layers.
Comparing an optimized implementation only to itself cannot establish original
source truth. Lossless repack, lossy derived quantization and temporary compute
encoding are different claims. A changed sparse mask/expert top-k/solver/guidance
schedule is changed semantics, not a free kernel optimization.

Retained original Qwen3.5 arithmetic constraints include source block/reduction
orders for Q4_0/Q4_1/Q8_0/Q6_K/F32 projections, explicit FP32 rounding stages,
GDN decay→prediction→beta→update→readout, ordered RMS/L2 and nonlinear policies.
Generic F64 multiply/add followed by F32 is not universal FMA emulation: midpoint
residual corrections and stored boundaries remain required where qualified.
Softmax's F64 reciprocal then F32 conversion is a two-round operation. Attention
value accumulation includes exactly the source's ceil(valid/256) padded FMA
prefix; canonical +0 future KV and signed zero are mathematical state. Preserve
partial IMRoPE's iterative source phase, polynomial versus scalar nonlinear
profiles, activation-quant rounding and finite-output checks. Greedy finite
f32 total ordering distinguishes +0/-0 and resolves equal-bit ties to lowest ID.
The authoritative detail is retained in the final section of the numerical
reference and executable source/tests; do not substitute a more convenient
formula because it is algebraically similar.

For new models, original HF grouped V axes versus converted tiled axes, converted
`-exp(A_log)`, non-GDN norm +1 and GDN SiLU versus attention sigmoid are explicit
Qwen3.8 source profiles. Gemma4 includes proportional RoPE, value norms/sliding
and shared-KV rules, E2B PLE/doubled FFN, and 26B raw K=V/MoE routing. Different
framework top-k/reduction/scale schedules need an independent choice and trained
qualification. Metadata exact-integer parsing and five analytic router cases
are not that proof. See model semantics' “Current metadata-only source profiles”
and the attributed [Gemma family contract](../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md).

Benchmark equal source/quant/input history/shape/context/sampler, power/driver,
physical placement and cold/warm conditions. Report useful committed whole-task
latency, TTFT/P95, memory and energy, not just launch count or a best microkernel.
Full-logit/all-state internal bit equality is stronger than sampled internal
checks but different from independent source full-vocabulary distribution proof;
partial top20 agreement cannot establish KL/TV. No universal faster-than claim
or imposed percentage/deadline was accepted (U-012).

### 5. Resources, accelerators and virtual residency

The single physical arbiter accounts models, private state, code/scratch,
activation, staging and output on actual physical domains. UMA aliases count
once; virtual address/GTT limits do not establish physical free RAM. Coarse
admission and fair workers own global policy; bounded local event-backed pools
serve hot operations without a global per-op mutex. Logical reservations must
state their actual coverage: a raw source lease is not all process RAM and a
cooperative cache quota is not filesystem-wide exclusive capacity. See
[memory/residency](MEMORY_RESIDENCY_AND_IO.md),
[engine protocol](ENGINE_CORE_EXECUTION_AND_PROTOCOL.md) and
[spine resource hierarchy](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md).

The [accelerator contract](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md)
§§1–8 defines NativeKernel, CapturedCommandGraph, AheadOfTimeDeviceGraph,
StaticallyScheduledDataflow and narrowly qualified ManagedDeviceSession. A
closed region exposes every external input/output/effect, guarded shape,
precision, lifecycle and state bridge. Opaque text APIs belong behind a
TaskServiceAdapter, not a purported device backend. Compilation/load/warmup,
code/scratch budgets, actual cancellation and checkpoint/replay are part of
qualification. TPU/wafer/coherent CPU-GPU/NPU descriptions are architectural
coverage, not executed hardware support.

Heterogeneous output-row prefill must preserve exact joins, source semantics,
state and source-versus-derived quality. CPU↔GPU twin↔NPU agreement on a derived
encoding does not prove quality against original weights. Retained replay needs
a compatible state frontier and actual completion; no automatic unqualified
power/fabric tuning. See [heterogeneous prefill/replay](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md)
and [GPU/UMA safety](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md).

The [virtual tensor/SSD contract](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md)
§§1–11 distinguishes stable source extents from generation/layout-specific
materialized views and Ready events. Expert blocks, lookup rows, frozen state
and media have distinct policies over the same I/O/event/arbiter owners.
Encoded/captured/queued consumers must pin before submission through actual
completion, not only once a kernel starts. Mixed-size/codec classes share a
global byte cap, preserve hot entries and charge staging peaks. Cancel does not
free in-flight DMA. A resident hit needs no mandatory host rematerialization;
SSD enables capacity, not a promise dense spilled inference remains interactive.
The SSD-01–SSD-32 gates, including queued readers and mixed cohorts, remain
planned, not an implementation completion inferred from cache acquisition.

### 6. Distributed placement, routing, serving and speculation

[Distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md)
§§1–16 separates compute, immutable parameters, mutable state and immutable
cache maps, with derived physical transfers in a bundle. Legal modes include
layer/state colocation, remote FFN/expert compute, explicit attention/state
separation, hybrid prefill/decode handoff and only operator-proven partition
algebra. Merging attention summaries requires valid masks and source numerical
semantics; it does not license generic GDN/KDA recurrence sharding. Directional
latency/bandwidth, relay/encryption/queue/copy costs and capacity versus speed
must be measured, not guessed from aggregate link bandwidth.

The [remote protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md)
binds source/schema/tenant/session/root/placement epoch/lease and actual events.
Peers prepare state; one coordinator publishes. Migration occurs at a safe
committed frontier with destination reservation, transfer validation and old
owner fencing. A trusted tunnel does not confer authentication, RDMA, model
redistribution rights or permission to use a peer. Local runs have zero automatic
peer overhead; fail-stop/replay is not transparent HA or a consensus cluster.
SSH fleet checks do not qualify distributed inference.

[Router/residency](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md)
§§1–13 separates RegisteredModel/source, revisioned Alias/Profile, physical
ResidentModel/replica and TaskSession/StateRoot. Resolve exact source/quant and
profile once at admission; route only among compatible replicas and keep live
session state sticky. An explicitly configured automatic model selector is a
changed target policy, not equivalent execution. Catalog count is not simultaneous
physical residency. The catalog remains cold control work: the attempted
left-right publication migration was rejected for lack of representative
contention; no accepted speedup or remaining dependency on that experiment.

[CLI/API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md) and
[runtime UX](RUNTIME_SERVING_AND_UX.md) specify one library engine for
run/serve/SDK, progressive configuration and explicit security/admin boundaries.
No mandatory daemon/download/config file; a miss never silently acquires.
Chat/completions/Responses/embeddings/SSE/tools must be explicitly tested subsets;
tool-call data is not permission to execute tools, and API resemblance is not
blanket compatibility. A reserved/disabled service is not a working production
server. Root README is user-owned and remains unchanged.

[Scheduling/speculation](SCHEDULER_BATCHING_AND_SPECULATION.md) and
[joint goodput](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md)
retain one scheduler and target authority: request/tenant admission, queued
prefill/residency and physical active steps are distinct valves. Chunking,
grouped experts, fair age/decode/media scheduling and speculation are costed
against useful committed goodput/P95. Draft work stays provisional; rejection
rolls back every relevant domain and RNG/history law. MTP/ngram/suffix and newer
drafting strategies require exact primary identity and independent numerical/
state qualification; an unresolved name or provisional capability count is not
full parity. No second model scheduler or proxy per token is permitted.

### 7. V5/V5.1 multimodal and realtime vision

The [V5 contract](UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md)
MM-01–MM-11 and [V5.1 universal architecture](UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md)
§§1–14 add typed tokens, PCM/features/RVQ, pixels/latents/frame PTS, packets,
transcripts, finite decisions and structured outputs to the same owners. Rational
rate clocks use checked wide arithmetic and preserve resampler remainder; codec
depth is not elapsed time. Media state is not disguised as text KV.

A [verified assembly](VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md)
binds exact component weights/config/tokenizer/processor/codec/transforms/license,
typed ports and source identities. It differs from a physical replica and from
a workflow joining different source models. Only identity-compatible immutable
parts can be shared. Specialization has independent semantic, physical,
task-control and placement/QoS axes; a codec or source quantizer cannot silently
change the solver/RNG/guidance/task law.

Task families include causal generation, encoding/pooling/ranking, Whisper-style
AR ASR, CTC collapse, RNNT/TDT, non-AR TTS, nested-RVQ/dual-AR generation,
acoustic flow, duplex, image flow/AR, temporal video, joint AV and calibrated
finite-choice outputs. Hosted opaque work is an explicitly authorized
TaskServiceAdapter with privacy/cost/idempotency/Unknown reconciliation; it is
not native model support or permission to invent proprietary internals.
Ternary/PQ-like source encodings include scales/transforms and are not merely a
plain low-bit matrix. Source-specific operators and actual checkpoints remain
necessary for every family.

[Realtime/QoS](REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md)
§§1–18 distinguishes capture/perception/response/output lanes and session/turn/
transport/physical epochs. Multitrack frontiers and AV CommitGroups prevent
partial RVQ, latent frames or unsynchronized outputs from being presented as
playable media. Model commit, server delivery, ACK, playback and playback
uncertainty are distinct. Barge-in changes the relevant turn/output epoch, not
necessarily the physical card fence; a slow client gets bounded queues, not
premature GPU release. Sync, progressive stream and bidirectional live regimes
have different delivery/reconnect contracts. Soft QoS is not a hard realtime
guarantee; measure first-valid playable output and useful media deadlines.

The [multimodal decision ledger](../research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md)
retains DMM-001–DMM-032 as proposals and OMM-001–OMM-022 as open choices.
Checkpoint choices (ASR/TTS/image/AR image/video/decision), media/device memory,
recovery, SLO/playback/consent/hosted cost, transport/midturn/persistence/licenses,
unverified model identity, ABI, session-vs-card cancellation, egress and compile
latency must not be silently decided by a generic illustration. The
[multimodal TODO](../plans/multimodal/TODO.md) and retained family dossiers are
the implementation/test frontier. Existing mock contracts do not establish
trained ASR/TTS/image/video quality or realtime operation.

### 8. User decisions, proposals and supersessions: complete ID crosswalk

Canonical source: [DECISION_REGISTER.md](../plans/DECISION_REGISTER.md).
These are summaries to navigate its exact statements, not replacements.

| IDs | Binding request or historical change |
| --- | --- |
| U-001–U-009 | Name, Rust-owned engine, broad and extreme models, portable legal fallback, attributed prior-art learning and semantic IR without hot-path abstraction tax. |
| U-010–U-015 | Initial research-only scope, complete decisions/pending context, no invented target deadline, corrected canonical path, unified designs and maintainable future research. U-013 replaces only U-002's destination. |
| U-016–U-022 | Heterogeneous prefill design, neutral/source-license policy, minimal critical review, natural accelerator granularity, first-class distribution, native router/simple UX and publication-quality design. None is blanket physical proof. |
| U-023–U-025 | Packaging-only implementation first; U-024 later authorizes reference implementation; U-025 fixes original .8B/2B CPU→CUDA/HIP/Vulkan before new Gemma/Qwen CPU→GPU order. |
| U-026–U-029 | Rigorous numerical/context gates; U-027 permits GPU work after genuine functional8K and defers16K/32K, without relaxing independent numerics; U-028 retains8K drift investigation; independent parallel streams are required without invented external-agent execution. |
| U-030–U-035 | Mixed source codecs, persistent residency, decode-once/static-vs-step split, typed async readback, uncertain-Future keepalive/quarantine and global physical/fair admission. Host proof at an early stage is not later hardware proof. |
| U-036–U-040 | Attested physical binding, real Vulkan workers, AMD native-F32 failure versus qualified WideF64, narrow adversarial fixes and immutable shared-model/independent-session catalog design. |
| U-041–U-049 | Absorb additive multimodal vision, repository ownership/user README preservation, H3/H4/H5 source operators, comprehensive context, heterogeneous retained replay and neutral core terminology. Retain source attribution. |
| U-050–U-055 | H7 source SiLU/output gates, H8 coefficients, H9 RoPE/current KV, H10 QK, H11 padded softmax and full resident GPU continuation/H12 source-padded value accumulation. Stage-local incomplete statuses are superseded by later accepted scope, not by weakening semantics. |
| U-056–U-060 | Finish qualified original GPU/native paths, first optimization fleet, parallel/disposable-source ownership, layout and contention-first pass; relocate reusable probes and preserve the sole product CLI. |
| U-061–U-064 | Explicit XDG HF/GitHub pull and finite pinned parity frontier; new model CPU basics first; full applicable Qwen/advanced decoding frontier; reject speculative cold-catalog concurrency migration. |
| U-065–U-068 | Publication authorization stays action-specific; checkpoint acceptance and simple TODOs supersede generated handoffs/evidence archives/SHA inventories. Production content hashes and numerical tests remain mandatory. |
| U-069–U-070 | Wrap/revalidate then PAUSE; new comprehensive context/start prompt is documentation-only and does not resume implementation or revive obsolete evidence duties. |

Proposal continuity covers **every P ID**, while preserving proposal status:
P-001–P-017 initial portable/tuned/extreme/compiler/state/format/UX proposals;
P-018–P-025 integrated compiler/execution spine; P-026–P-030 heterogeneous/task
research; P-031–P-035 neutral V3; P-036–P-045 minimized V4;
P-046–P-055 accelerator V4.1; P-056–P-070 distribution V4.2;
P-071–P-086 router/UX V4.3; P-087–P-091 public document/agent delivery;
P-092–P-096 packaging/build/release/module foundation. P-036 minimizes the old
permanent IR stack; prepare-only root publication replaces vague domain commit;
U-067/U-068 replace P-091's former hash/archive workflow. The six target build
definitions and installer checks do not themselves prove physical builds on six
platforms, signed releases or a working daemon.

All **O IDs** remain traceable, with narrow later decisions noted rather than
silently removing them: O-001–O-022 initial licensing/platform/FFI/toolchain/model/
quality/API/config/trust questions; O-023–O-026 NPU/source/compiler/hang/task
choices; O-027–O-034 reuse/training/release/delivery/allocator/plugin/control;
O-035–O-041 vendor graph/SDK/state/hardware/privacy/ambiguous accelerator name;
O-042–O-053 remote topology/enrollment/privacy/transport/recovery/placement/
partition/rights/SLA; O-054–O-066 CLI/config/catalog/API/service/replica/profile/
admin/residency/multi-role policy; O-067–O-074 publication/legal/channel/platform/
service qualification. Later U-023/U-025 and actual code settle specific bootstrap
and first-model choices without closing every general question. Historical
O-071/A-003 statements about no measurements belong to their research date;
they do not erase later bounded physical tests or establish universal superiority.
A-001 is superseded by U-013; A-002 requires refreshed source versions; A-004
requires real fallback semantics/resources; A-005/A-006 grant no ambient
publication/account/secret authority. Future decision records must identify
source/options/tradeoffs/tests/conclusion/superseded IDs and affected files.

### 9. Accepted implementation versus historical stage facts

[Accepted baseline](../plans/accepted-baseline/TODO.md) records owner-approved
original Qwen3.5 .8B/2B functional CPU Phase1 **16/16 through genuine8K** and
complete original-model resident GPU execution through **512 consumed tokens**.
Its accepted five backend profiles are CUDA local RTX3060/a1 RTX4060, HIP a1
gfx1150/x2 gfx1151 and SDK-free NVIDIA Vulkan: **not five completely qualified
physical cards**. The record lists135 profile checks plus12 host fingerprints.
It includes compressed embeddings, activation quantization, GDN/attention/FFN,
all42 private domains and compact token feedback. This paragraph relays retained
owner approval; the unavailable old graph prevents independent reconstruction of
its full original report sequence in this audit.

State-only prefill, exact Contiguous aliases, prepared range bindings and fused
View certification are not driver capture, GPU tokenization or a generic device
scheduler. Trusted diagnostic replay is not authenticated product persistence.
Independent source512 comparisons cover146 stages/model with zero nonzero-FP32
differences and56 recorded2B signed-zero distinctions; internal full-bit tests
remain strict. Native CPP scalar/libm/loader profiles are bounded; separate LLVM
device code generation is unqualified. Old statements “LLVM unavailable” or
“full GPU pending” describe earlier stages and must not overwrite later approval.

The failed cold-catalog left-right experiment was functionally reviewed but
rejected as an adoption target because no actual contention was established.
Production retained the simpler prior catalog; future adoption needs realistic
measured contention, lifetime/replay correctness and maintenance benefit.
Functional test success does not imply accepted migration or speedup.

[Model execution](../plans/model-execution/TODO.md) records Qwen3.8 27B and
Gemma4 E2B/26B **metadata/config/tensor/source-index contracts only**. No actual
trained new-model execution, tokenizer qualification or complete checkpoint
loading follows. This remains true despite extensive original-model acquisition
and loader tests. Official source links and subtle layout/math contracts are
retained in model semantics/Gemma references, not disposable historical prose.

The [numerics/context TODO](../plans/numerics-and-context/TODO.md) retains the
independent8K GDN problem: first final-token divergence at .8B layer4/2B layer1,
shared top20 logprob differences0.20372924/0.19678696 at native ubatch512 and
matching greedy7993. These do not prove full-vocabulary agreement. Historical
functional limits0.45/0.05 are not permission to change source-bit operators.
16K/32K whole-model tests and resident GPU beyond512 remain unqualified. The
separate historical AMD FMA failure is not an established cause of CPU GDN drift.

### 10. Current acquisition and remaining execution boundary

The [acquisition TODO](../plans/model-acquisition/TODO.md), its
[design](../plans/model-acquisition/README.md) and
[implementation specification](../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md)
record accepted source-preserving cache lifecycle, bounded explicit HTTP,
HF/GitHub files/releases/components, direct HTTPS contracts, OCI/GCS/S3/Azure,
ModelScope and native Xet, then metadata-only candidates and explicit filename
quant selection. These use the sole HTTP/cache/executor/publisher, original
arbiter and no implicit run/inspect fetch. Default CPU/cache-only closures stay
transport-free. Native provider/profile qualification is synthetic protocol
coverage unless an actual live source test is explicitly recorded; no claim of
complete upstream parity follows from a typed Unsupported frontier.

Exact and capped unknown lengths are separate. AtMost requires explicit opt-in
for auxiliary data only, real terminal EOF and actual length/hash; no percentage
against an unknown total or durable unknown-prefix resume. Required weights,
config/tokenizer/index/license closure stays exact. SHA trust distinguishes
caller/pinned expected digest from observed FirstSeen. Git OIDs, validator labels,
mutable tags and source filenames are not trusted raw content identities.

Local HF import is offline data-only copying from retained opened source FDs,
actual-byte verification, source/target disjointness and conservative qualified
filesystem policies. It never proves remote repository membership. Cache-backed
run preparation holds an admitted raw ImmutableWeights lease through retained
SourceWeights/Arc owners; sidecar copies have bounded staging through parse.
Managed descriptor protections survive dup/inherited handles until final OS
close. V2 exact-snapshot reclamation is cooperative; v1 migration, orphan/eviction
sweeps and arbitrary raw readers are not qualified. Named unique-inode charge is
not physical freed space. Raw run/inspect/File readers need external quiescence.

Native Xet authenticates complete file roots and exact raw length/SHA independently
through original primitives and bounded reconstruction. It preserves signed
interval sets/auth isolation/term order/replay/EOF and counts physical attempts,
observed bytes, chunks, leaves and refreshes across files/reissues. It does not
claim per-term partial authentication, performance gain, persistent native resume
or general required native-index support. Native default256MiB is a scoped
concurrent scratch+cache admission envelope; ordinary default128MiB and explicit
caller limits remain unchanged, not a global process RAM guarantee.

Current signed **f197c9c** accepts C2 explicit filename GGUF quant: exact terminal
basename tag/canonical complete split family, same pinned inventory/closure and
one fetch runtime; no auto quant ranking, whitelist, numeric eligibility or
fallback. C1 candidates atf22f759 list regular choices only; stored-Git/tree/LFS
versus verified-content length bases are explicit. No HEAD/index/model payload or
publication is implied by discovery. Quant offline verifies a requested exact
snapshot subset/current bounds/pointers/roles/closure; original Files offline
remains initial-selection/role-agnostic. Explicit components apply their stricter
Weights closure; generic cache verification still validates literal bytes only.

Recorded latest actual checks:975 focused passes/four strict configurations;
full frozen isolated gate934 Rust passes/0fail/69ignored/205suites plus18namespace
tests, Nix source integrity and26 local links. Original release cache_prepare
routes pass separately17.60s and23.70s. Each uses approved .8B/2B originals,
>=2 tokens, all248320 logits/all42 domain values bit-exact, independent sessions
and source IDs, retained raw lease, mutation/ref retarget and private reclamation.
These are separate test durations, not one aggregate invocation, and add no new
model/context/GPU qualification. Earlier per-phase counts are not additive totals.

**Paused unfinished drafts:** C3 filters in `/home/pvl/spaces/abird/src/zerogravity-filters-work`
has a matcher/oracle scaffold, is unwired/not compile-ready, with all Rust tests
unrun. C4 in `/home/pvl/spaces/abird/src/zerogravity-xet-index-work` changes only
HF/xet mod/budget draft paths; private callback/defaultfalse opt-in/all-counter
Result continuation and two unit drafts are uncompiled/unrun; CLI integration
and actual index tests are absent. Required future policy is same pin and bounded
metadata reader, trusted LFS rawSHA plus complete Xet root/actualEOF before index
parse, same admitted scratch and every planning/execution counter carried forward.
Default/dry-run must remain no-CAS; apply must explicitly opt in. Neither SIDE
is an implemented feature or a task to resume during this documentation request.

Other pending plans retain their own truth: [CPU/GPU optimization](../plans/cpu-gpu-optimization/TODO.md)
(measured hotspots, held Renoir/gfx90c, native numeric modes/recovery/fairness),
[memory/streaming](../plans/memory-and-streaming/TODO.md) (global host/UMA and
SSD/state/prefix policies), [serving/speculation](../plans/serving-and-speculation/TODO.md)
(production API/batching/advanced decoding), [multimodal](../plans/multimodal/TODO.md),
[distributed](../plans/distributed-execution/TODO.md) and
[platform/release](../plans/platform-and-release/TODO.md).
Full applicable pinned Qwen/provider parity needs finite enumerated source/API/
CLI/internal inventories with honest open rows; provisional counts and successful
small examples are not exhaustive closure. New models still need authorized
checkpoints/host capacity, immutable shards, trained CPU truth, then physical GPU.

### 11. Exact surviving historical recovery and unrecoverable material

Read-only recovery ref verified for the following bodies:
`7e61cb63efd0c1429aa96044f2f7d28fda1e59aa`, parent of archive deletion
`9239719c25128028fd78a6a60f817ce6b626517f`. These bodies also belong to the later
retired standalone design/archive set in the rewritten graph. Its multimodal aggregate also embeds the older94-source implementation archive, giving partial early-report recovery.
To inspect without restoring a file, use `git show REF:PATH`; do not recreate
archive directories, generated aggregates or obsolete source inventories.

| Historical body/path at that ref | Unique recoverable context and current semantic owner |
| --- | --- |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md` | 474lines; V2.1/NPU update then V2 synthesis; lines171–474 embed the original V1 initial handoff. Current master, vision, IR, model/state/resource/UX docs retain its live requirements. |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md` | V2 master snapshot; current master and minimal spine supersede old permanent-IR/kernel-plan ownership. |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_ROADMAP_V2.md` | Early implementation sequencing; current roadmap and plan TODOs distinguish delivered phases. |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_STAGE0_V2.md` | Original pre-framework feasibility gates; current experiment/acceptance/roadmap source keeps falsifiable ownership tests. |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_EXPERIMENTS_V2.md` | Early research measurements-to-do, not execution results; current numerical/experimental design owns them. |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_ACCEPTANCE_GATES_V2.md` | Historical quality/memory/source/failure gates; current engineering invariants and plan verification refine them. |
| `.agents/prior-art/DESIGN_EVOLUTION_V3_SNAPSHOT.md` | Three embedded V3 architecture/model-task references; current V4 composition/TaskDriver/prepare-only contracts resolve their overbroad sketches. |
| `.agents/prior-art/DESIGN_EVOLUTION_V3_HANDOFF_SNAPSHOT.md` | V3 decision/minimality/research transition; canonical register retains authority and supersessions. |
| `.agents/prior-art/PRIOR_ART_HISTORICAL_DISTRIBUTED_PARALLELISM_V4_1.md` | Pre-V4.2 topology snapshot; current distributed placement/protocol separates four maps and real transport costs. |
| `.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md` | 2,796,909byte/23,979line aggregate. Nested current contracts/dossiers and historical wrappers were consolidated; current multimodal references, ledger, realtime QoS, SSD and technical source references are semantic owners. Counts129/191 are historical aggregate inventories, not current physical file counts. |
| `.agents/research/MULTIMODAL_PARALLEL_RESEARCH_WORKSTREAMS_2026-10-14.md` | Early parallel agent briefs, negative fixtures and modality gate proposals; retained integrated acceptance/roadmap and multimodal TODO own pending tests. |
| `.agents/research/MULTIMODAL_V5_1_RESEARCH_WORKSTREAMS_AND_AGENT_BRIEFS_2026-10-15.md` | Expanded family/QoS/source-license briefs; current multimodal ledger/dossiers/contracts own decisions and open gates. |
| `.agents/research/SSD_STREAMING_FIRST_CLASS_ARCHITECTURE_AGENT_HANDOFF_2026-10-15.md` | SSD wrapper/context; retained virtual tensor/SSD contract and SSD acceptance roadmap own the technical plan. |
| `.agents/research/EXTREME_MODEL_SPECIALIZATION_DESIGN_AGENT_HANDOFF_2026-10-15.md` | Hybrid/replay/low-bit/GPU specialization context; retained extreme model, heterogeneous replay, GPU/UMA and quantization references own source math/gates. |
| `.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md` | Former README body. Current retained path is a deliberate research pointer required by the immutable root README; preserve it. Historical prose is not current product support. |

**Recovery limits:** current standalone early handoff/phase paths and the original
33-node Git graph were not recovered. The later129/191-source handoff versions
were not found in the inspected current/ref/reflog/unreachable and retained
archive bodies. The older94-source U040/G3b archive IS recoverable inside the
multimodal aggregate; its executive brief, early CPU/P07/report/prompt bodies are
not lost simply because their former paths are absent. Read its complete94-source
catalog in the following history-recovery supplement. Later H4+ implementation
context survives in current decisions, numerical/model technical references,
accepted-baseline/source and TODOs, without a claim of restoring every old verbatim
report/command or original commit ancestry. Do not infer unavailable SHA objects
from labels printed in recovered text. The initial coarse negative audit was
superseded by this positive nested-source discovery and independent review.

### 12. Phase and falsification map for preserving future work

The [P00–P23 roadmap](../plans/IMPLEMENTATION_ROADMAP.md) is a capability graph,
not a claim that each phase is done. Its complete phase vocabulary is:

| Phase | Intended scope |
| --- | --- |
| P00/P01 | Lean build/source foundation; actual architecture-specific Rust/hardware feasibility. |
| P02/P03 | Independent math/state oracle; trusted artifacts, codecs and rendering. |
| P04/P05/P06 | Typed semantic/task control, legal optimizer/async execution and useful portable backend. |
| P07/P08 | Accelerator streams/leases/providers; first complete real model/task vertical slice. |
| P09/P10/P11 | Coordinator/workers/serving; physical full-state memory/prefix management; hybrid/stateful operators. |
| P12/P13/P14 | Routed sparse/tiered parameters; first and independently different extreme model specializations. |
| P15/P16 | Target-authoritative speculation/transactions; continuous batching and joint resource cost. |
| P17/P18 | Broader families/adapters/embedding/rerank; actual audio/image/video/iterative tasks. |
| P19/P20 | Heterogeneous/remote execution; bounded compiler/autotuner and negative experiments. |
| P21/P22/P23 | Security/license/release; full fair quality/performance qualification; extension/change-radius drills. |

The [experiment program](../plans/EXPERIMENT_PROGRAM.md) preserves H01–H86:
H01–H14 kernel/cache/speculation/serving/memory/topology/task hypotheses;
H15–H25 compiler/effects/async/capture/search/frontier/modularity/malformed-input/
benchmark/hardware/runtime-cost tests; H26–H31 heterogeneous projection/NPU/
machine safety/source quality/non-token tasks; H32–H43 V4 root publication,
device progression/hierarchical resources/shapes/task minimality/regions/delivery/
epochs/future parameter version/cache migration/fairness/change-radius;
H44–H55 graph/dataflow/device-source composability; H56–H70 physical distributed
placement/transport/authority/failure/capacity; H71–H86 local run/catalog/residency/
replicas/profiles/API/security/roles/hot reload/minimality. Stage0 and
[release gates](../plans/RELEASE_ACCEPTANCE_GATES.md) retain the matching
negative tests and eligibility rules. Earlier “NOT RUN” section titles are
historical design status; no one should turn a similarly named later test into
blanket completion of those matrices.

**Identifier namespaces matter:** these design H01–H86 hypotheses are not the
same as implementation H3/H7/H12 operator slices or the later H25 workspace
layout milestone. Likewise design P07 contains many later bounded implementation
subphases. Preserve the exact source/namespace when reporting a pass, rather than
inferring coverage from a matching number.

The [multimodal acceptance/roadmap](../research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md)
and family dossiers retain MM-01–MM-24; ASR-01–12, TTS-01–12, IMG-01–12,
VID-01–15, DEC-01–06, BON-01–08 and SRV-01–12. Distinct negatives include bad
source/conditioner/codec masks, rational-clock overflow/discontinuity, incomplete
RVQ depth, incorrect solver/seed, temporal VAE/frame/GOP mismatch, uncalibrated
finite outputs, missing required Hadamard transform, stale ACK/reconnect cursor,
late physical events, mixed AV premature presentation, slow-client growth and
unauthorized egress/cross-tenant replay. These should stay attached to their
source-qualified task, not collapse into one generic “multimodal supported” tick.

The [implementation traceability](../plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md)
map connects user/engineering/invariant/phase IDs; the original source dossiers
provide license and equation context. They are complementary references, not
fresh evidence or an obligation to restore deleted archive bodies. Acceptance
requires each real input/backend/context/task cell and its relevant failure
cases, not merely counting retained Markdown IDs.

### 13. Coordinator integration and next-agent guardrails

Integrate this context under the coordinator-owned handoff plus current plan
links; keep original research/prior-art references in place. Preserve source and
license attribution through [the provenance policy](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md),
not project-branded core types or old engine clones. Do not manufacture a new
archive/generated-handoff maintenance system, extra evidence log hashes or
mandatory stage receipts. Content hashes remain where production integrity,
source identity and authorized original checkpoint tests require them.

A next agent must first read the current plan index/TODO rules, live acquisition
TODO and the two paused draft run TODOs, then verify live source/ownership if the
user actually asks to resume. Read source behavior and failed/interrupted tests;
merge only scoped reviewed bytes with fresh modification times (the documented
stale Cargo experiment is real). Keep one shared Cargo owner, preserve unrelated
work and protected README, and distinguish qualification inputs/profiles from
future capabilities. The immediate next action **now** is completing the requested
documentation and remaining paused, not implementing C3/C4 or running the gate.


<a id="history"></a>

## Implementation history, verification and recovery

Author: Codex history-audit agent (coordinator: root).
Provenance: signed `f197c9cf4a44e06a3c1cf3c532855545bed945fa`.
Status: documentation-only audit; runtime implementation remains **PAUSED**.
No implementation,
Cargo, model, SDK, provider, hardware, Git mutation or network action was run.
All historical test results below are transcribed from retained accepted plans;
this audit did not rerun them. Current source was read only to identify owners
and retained phase labels. A source assertion or a commit subject alone is not
an executed test result.

### Principal findings for the coordinator

1. The current local reachable history contains **77 commits**, not the original
   pre-rewrite 33-commit graph. Its first **46 commits** form an owner/test/vendor/
   tooling decomposition of already developed implementation. They must not be
   relabelled as the original 33 chronological implementation steps. All77 exact
   current IDs/titles are listed below. The original33 ledger cannot be recovered
   from the local objects inspected in this audit.
2. Accepted pre-rewrite results remain authoritative as owner-approved historical
   completion. Their labels, including `6d74f76`, are not current reachable hashes.
   Do not silently replace them with a convenient rewritten commit or imply a
   fresh gate was run on that rewritten commit.
3. Some retired design/archive bodies are recoverable from **reachable** commits
   `7e61cb6` and `be95424`. Standalone early handoff/phase paths were already absent from `be95424`, but its multimodal aggregate embeds an older complete94-source handoff throughU040/G3b. The positive nested recovery and all94 reading ranges supplement this audit below. Original33-node ancestry and later129/191-source versions were not recovered; do not confuse missing Git objects/standalone paths with missing embedded text.
4. CPU original-model functional acceptance is16/16 through genuine8K. Resident
   GPU original-model acceptance is512 consumed tokens on five backend profiles,
   not five physical cards. Independent8K GDN numerical drift and16K/32K whole
   model contexts remain open. Acquisition acceptance does not broaden these
   model/numerical/GPU limits.
5. Current quant selection is accepted at `f197c9c`; full gate934 Rust executions,
   69 ignored,205 suite executions plus18 namespace tests. Both original model
   preparation routes passed separately17.60s and23.70s. Two preserved filters/
   native-index SIDE drafts remain uncompiled/unqualified and runtime is paused.
6. The fixed root-README research link must remain valid. Do not blanket-move
   `.agents/research/`; the pre-product snapshot path is intentionally retained.
   Do not reintroduce the third-party project name the user requested removed,
   even in a historical-title quotation. No current77 subject contains that name.

### Current history, rewritten dates and available references

Read-only commands actually run: `git log --all --reverse`, `git show`,
`git ls-tree`, `git for-each-ref`, `git reflog show --all`,
`git fsck --no-reflogs --unreachable --no-progress`, and individual
`git cat-file -e OLD^{commit}` checks. No refs were created or fetched.

Observed local refs:

| Ref | Exact object |
| --- | --- |
| refs/heads/main | f197c9cf4a44e06a3c1cf3c532855545bed945fa |
| refs/remotes/origin/HEAD | fa6affa425089fdf9a15bfa10b330e03f738fec8 |
| refs/remotes/origin/main | fa6affa425089fdf9a15bfa10b330e03f738fec8 |

These are local remote-tracking refs only; no network remote state was checked.

Topological order is meaningful; rewritten calendar dates are not a trustworthy
execution timeline. Example: root `be95424` is dated2026-10-13, `57a9ad7` is
2026-11-05, and its child `1af2ac4` is2026-10-16. These are observed author AND
committer dates. Do not place the Nov5 owner decomposition after Oct17 provider
work because of those synthetic dates.

The first46 current commits package the accepted baseline by source owner. They
include the final GPU/model/layout/cache implementation, so a current commit
introducing a file is not proof that all its H-phases happened together. The old
approval labels and retained plans supply the historical development sequence.
The later31 commits are the reachable continuation, including documentation,
XDG namespace change and23 acquisition-phase commits through quant selection.

#### Entire reachable commit ledger

Exact subjects below are current Git metadata, not claims of independent tests.
The first46 rows are the rewritten source decomposition; rows47–77 continue it.

| # | Exact current commit | Exact current subject |
| --- | --- | --- |
| 1 | `be95424837deb7919adf34503db71d27d3ad3adb` | docs(design): product, architecture, prior-art and research plans |
| 2 | `1e10b5cc1d5cf34832477bc83ef39117a765c6a0` | feat(core): immutable source identity, state roots and physical resource ownership |
| 3 | `af9a5d3969c9a87951990223b6d289aba69c63fb` | feat(artifacts): sharded source artifacts and model identity |
| 4 | `b276fd01e421f1b7e416ae773610b5823b9da36a` | test(artifacts): cache transactions |
| 5 | `586d440b144727bdd9acea7455afb27f6c38ae25` | test(artifacts): qwen text contract |
| 6 | `782e6fc48f0924cc055232d56338da62b68b1a79` | feat(cpu): portable CPU math and weights |
| 7 | `1fd12da7840151ba47ab43368b4e4f435aac3d3b` | feat(model-qwen35): source-verified Qwen3.5 forward and tokenizer |
| 8 | `901ba9ae2f2ed531d992a6d657a37dc362518bd0` | test(model-qwen35): source layout contract |
| 9 | `e6ecf02bb7d951e01e66b430a24553c690b4ac5b` | feat(model-gemma4): Gemma contract, MoE and 26B variant |
| 10 | `81642a9af4d88d928327a1012af1594168704956` | test(model-gemma4): e2b contract |
| 11 | `7aa19c268026069ad916f6a69440dab602aaa94a` | test(model-gemma4): family contract |
| 12 | `65b026f902501e85afad5fd31f3ca53968a3e836` | test(model-gemma4): variant 26b contract |
| 13 | `b03c16451d65c33a05b3c495c1205240db4ccac0` | feat(data): vendored CubeCL license and provenance |
| 14 | `f491e34fe55e67eed434ac577c32d44eca51430e` | feat(data): vendor CubeCL cubecl-ir 0.11.0 |
| 15 | `627c58ec80ff7b33fa1ce5086a729946c5b1cd44` | feat(data): vendor CubeCL cubecl-cpp 0.11.0 |
| 16 | `3664f975e79242d8b1a12bc4eefe886f81a33c2f` | feat(data): vendor CubeCL cubecl-cuda 0.11.0 |
| 17 | `f77ae37ec0317f39a2e1ce4466d82c645cd344da` | feat(data): vendor CubeCL cubecl-hip 0.11.0 |
| 18 | `7d768a080a84b6132595f7cadbc84b75b69106e4` | feat(gpu): source-exact resident GPU execution and physical ownership |
| 19 | `a9723bceb8ad4aa4df639dbe660efcae6c22711b` | feat(runtime): facade, semantic program, task drivers and state publication |
| 20 | `c3d0d0b06fb1548c8bf3db10f2965e5af52be9bc` | test(runtime): gemma contract |
| 21 | `4add50a023d4fb5dd989989cab75ac4298d46d29` | test(runtime): layout api |
| 22 | `b4a36b829031fba498e45bebb71a6cd8f01e7ef9` | test(runtime): multimodal mm0 |
| 23 | `a691e21c040d2dcf46a52f2cbcf0e0eb934c8369` | test(runtime): qwen real model |
| 24 | `25ec3e27e3c208158e9763944026003005456461` | test(runtime): qwen shared model |
| 25 | `047e3b70bbcbd7acbb5176712208cc4bcc02dd29` | feat(cli): single-binary inspect and run |
| 26 | `52e37313bff57c2a7e831f50e9236fa983e13d02` | test(cli): CLI contract |
| 27 | `22e0cc5abb4a067a95324206c2881de9e3327427` | feat(tooling): workspace GPU qualification crate |
| 28 | `4d82c6704dad47875fa42bdec65f9a65de79722e` | test(tooling): independent reference oracle llama_reference_layers |
| 29 | `a2d6a02f1577554ea8dba669f77ba7b903c1bb29` | test(tooling): independent reference oracle llama_reference_logits |
| 30 | `a8ea8926b53d7b4a9c6d00f3d9be04906f8d64bb` | test(tooling): independent reference oracle llama_reference_qk |
| 31 | `6a0079eb167d1f1593fe2a9372039138cd949770` | test(tooling): independent reference oracle llama_reference_recurrent |
| 32 | `be88e30733f9459eda5f150bdbd70d9d0f8c6684` | test(tooling): independent reference oracle llama_reference_rope |
| 33 | `77778a2c258751c64c56c4ffd3dba2ff70f7296c` | test(tooling): independent reference oracle llama_reference_softmax |
| 34 | `6015241a5f5c2bb7ee65715a998988a026ff800c` | test(tooling): independent reference oracle llama_reference_value |
| 35 | `e190aff1fdd392696e8086010a7bbc1f12d3df4f` | test(tooling): independent reference oracle llama_reference_value_padding |
| 36 | `0aea8de66e3eb3b704e7d26687731a3e5b07d3bb` | test(tooling): build-release-artifacts.sh |
| 37 | `057987203d5de45ad268ba02cfc5084f76f48229` | test(tooling): check.sh |
| 38 | `c74450a27b1e3080e9576b0f95922f17c7cdca8b` | test(tooling): compare-attention-value.py |
| 39 | `9f54d2435fe6af15b3eaf28980bc1039ad02b559` | test(tooling): compare-qwen-layers.py |
| 40 | `7458ffc286f7e72c70ac18acfd3f776186aa204a` | test(tooling): cpu-qualification-gate.py |
| 41 | `73cb6e55e1795e2c13533f60621dc29619cf5e2c` | test(tooling): native-gpu.py |
| 42 | `80145a7f91724fb34a6f67e9972850a279982bf3` | test(tooling): test-install.sh |
| 43 | `ca1138b8cf866c81bbfb276358d03a62576df16a` | test(tooling): test-native-gpu.py |
| 44 | `5a0636c3f9c744c2f979ded40943a269ba5a1176` | test(tooling): test-numeric-resume.py |
| 45 | `9a6c504ebef037e09bd187d65e2bec7e2e456eef` | chore(workspace): restore the locked full workspace manifest |
| 46 | `57a9ad7fe78bf7c5e64d3870a6ba176ceba4cc07` | test(tooling): independent reference probe notes |
| 47 | `1af2ac434279cb5f4b635a51884c936f19b54aa3` | fix(nix): give module check a complete base system |
| 48 | `0452c594d99df6d3b4c972ceaae8df6ea56925e3` | docs(plan): record module eval check fix |
| 49 | `2afa32aa74aff7f35ec1b52e4caa7079a5f68980` | docs(readme): reorder engineering and FAQ sections |
| 50 | `a054d396e1ec12fc5e9b8e2ca60e8060954870c5` | docs(agents): document plan layout and TODO spec |
| 51 | `cc3cab27a2713fdb64bb3f1a42c8138c060c5dce` | docs(contributing): align guide and AI disclosure |
| 52 | `7e61cb63efd0c1429aa96044f2f7d28fda1e59aa` | docs(agents): gate pushes and destructive actions |
| 53 | `9239719c25128028fd78a6a60f817ce6b626517f` | docs: retire historical archive bodies |
| 54 | `6deebf22228315863f563723b5e6761ac6b51c5c` | refactor(cache): namespace model cache under abird/zerogravity |
| 55 | `fa6affa425089fdf9a15bfa10b330e03f738fec8` | feat(artifacts): add pinned model pulls |
| 56 | `3c97c52075563f65928c564818ce73502168778f` | feat(artifacts): resume interrupted pulls |
| 57 | `baa9d8818061958a63eabeb61a7239a103a738f0` | feat(artifacts): load pinned cache components |
| 58 | `20f449d20320b6f2912698c7b97cfdfd82472ba2` | feat(artifacts): recover missing pull sizes |
| 59 | `f44264c017856a015584a094566a077638939e55` | feat(cache): add protected snapshot removal |
| 60 | `ea700f67fca7ff7871613044a989e03b8ad6f9d7` | feat(pull)!: add capped auxiliary streaming |
| 61 | `03b69482a5fb10b8c1cc7032e7cbdf39e2e5fead` | feat(pull): pin multi-asset release components |
| 62 | `e44f4ee8020d5661dd8167cb07dfcb085906fcb7` | feat(cache): import local HF snapshots |
| 63 | `fce1353e649df2eb73fe6a39437043fc8551aef5` | feat(cache): qualify cross-filesystem imports |
| 64 | `e0073a5593e5641494c0f48e8d2dcc9e9e61a119` | feat(pull): add pinned HTTPS objects |
| 65 | `e82d76b8f8e91f7d4b49c211e26baa081ff523e5` | feat(pull): add pinned OCI GGUF layers |
| 66 | `f74a525366e6280a9806ecaabd7e1f4df46a6a97` | feat(pull): add version-pinned GCS objects |
| 67 | `833d68f716783c255b24691552a081a1d2b88e76` | feat(pull): add versioned S3 and SigV4 |
| 68 | `e62222276e0dac62ed554a3fec2f2ae86edad497` | feat(pull): add pinned Azure blobs |
| 69 | `87f6454fb7c4832262d36104a806baad9a9734a7` | feat(pull): add pinned ModelScope files |
| 70 | `223ca794c9d45a0e246ad1cd9ba8613f2f203cd4` | feat(pull): add Xet hash and chunk primitives |
| 71 | `a774ded154ed066c8f4b134ce7f5cba32ccdd09c` | feat(pull): add cloud content contracts |
| 72 | `837e77f5080c0c2c07fee90b2ef9fb791068a7d3` | fix(pull): enforce offline acquisition rules |
| 73 | `52e42fcdf0e7bead1ebefe55d8103501c7bd4f64` | feat(pull): add native Xet reconstruction |
| 74 | `46cf5a5830e7e8d8096234341ef51fafdf63cc16` | feat(pull): add conditional cloud reads |
| 75 | `4db5efbcce973d21bf1c8ca750471e6b2ab956ab` | feat(pull): recover pinned ModelScope sizes |
| 76 | `f22f75972e63ea70771c59fbce065e371d5f29d4` | feat(pull): list pinned file candidates |
| 77 | `f197c9cf4a44e06a3c1cf3c532855545bed945fa` | feat(pull): select explicit GGUF quant families |

#### How the rewritten baseline is split

- `be95424`: technical/design/research and retained acceptance/TODO records.
- `1e10b5c`: std-oriented core source identity, accelerator/program contracts,
  state/session roots, physical ResourceArbiter, events and fair device queue.
- `af9a5d3`: artifact parsing, original source allocation/identity, sharded source
  contracts and original local cache; `b276fd0`/`586d440` split artifact tests.
  The A1 accepted194-test result is retained under historical `3787519`; it is
  not an additional test result invented from these rewritten commit subjects.
- `782e6fc`: CPU math/weights. `1fd12da`/`901ba9a`: Qwen forward, checkpoint,
  tokenizer, model catalog and source-layout tests.
- `e6ecf02` and its three following test commits: Gemma config/tensor/MoE/26B
  contracts and official synthetic metadata, without trained new-family inference.
- `b03c164` through `f77ae37`: licensing/provenance plus four patched CubeCL0.11
  packages, not independent physical backend acceptance by themselves.
- `7d768a0`: all GPU implementation owners, including H1–H23 mechanisms,
  resident sessions, physical resources/events/quarantine and native backends.
- `a9723bc` through `25ec3e2`: facade plus its independent integration tests.
- `047e3b7`/`52e3731`: the sole product CLI and its contract tests.
- `22e0cc5`: GPU qualification tool; following reference/tool test commits split
  eight independent native-source probes and build/check/comparison/namespace/
  installer tools. `9a6c504` restores the locked full workspace manifest;
  `57a9ad7` retains independent reference-probe notes.
- `1af2ac4`: Nix module evaluation check needs a complete base system;
  `0452c59` records that bounded fix, not new model/GPU support.
- `2afa32a`: user-authored README engineering/FAQ ordering in its own historical
  commit. This does not authorize modifying those bytes during current work.
- `a054d39`/`cc3cab2`/`7e61cb6`: current TODO layout/contribution instructions and
  explicit push/destructive-action consent. Current runtime remains unchanged.
- `9239719`: archive retirement, described below. `6deebf2`: user cache namespace
  under `abird/zerogravity/models`, retained by subsequent A2 implementation.

### Recovery: what is really available and what is not

The following commands were tested successfully and are read-only. Use their
output for historical reading, without restoring archive files into the worktree:

```sh
git show be95424:.agents/plans/DECISION_REGISTER.md
git show be95424:.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md
git ls-tree -r --name-only 7e61cb6 -- .agents
git show 9239719 --name-status --format=
```

`git show be95424:PATH` or `git show 7e61cb6:PATH` recovers the following14
files deleted by `9239719` (their paths were verified by ls-tree):

- `.agents/prior-art/DESIGN_EVOLUTION_V3_HANDOFF_SNAPSHOT.md`
- `.agents/prior-art/DESIGN_EVOLUTION_V3_SNAPSHOT.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_ACCEPTANCE_GATES_V2.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_DISTRIBUTED_PARALLELISM_V4_1.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_EXPERIMENTS_V2.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_ROADMAP_V2.md`
- `.agents/prior-art/PRIOR_ART_HISTORICAL_STAGE0_V2.md`
- `.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md`
- `.agents/research/EXTREME_MODEL_SPECIALIZATION_DESIGN_AGENT_HANDOFF_2026-10-15.md`
- `.agents/research/MULTIMODAL_PARALLEL_RESEARCH_WORKSTREAMS_2026-10-14.md`
- `.agents/research/MULTIMODAL_V5_1_RESEARCH_WORKSTREAMS_AND_AGENT_BRIEFS_2026-10-15.md`
- `.agents/research/SSD_STREAMING_FIRST_CLASS_ARCHITECTURE_AGENT_HANDOFF_2026-10-15.md`

A fifteenth retired body is the pre-product README snapshot, whose fixed path
remains present with pointer content. Thus the precise current diff is14 file
removals plus one replaced historical body, not15 deleted paths. Current build/
release/Nix guidance moved to `docs/DEVELOPMENT.md`. Its original body remains
readable at the tested `be95424` reference above.

**Corrected recovery boundary:** the former standalone full-implementation
handoff, next-agent prompt and early phase paths are absent from the current
root tree. An older complete94-source full handoff throughU040/G3b survives
inside the reachable multimodal aggregate, including several of those reports
and prompts. The exact94-source catalog and old-label chronology follow in the
recovery supplement. Later129/191-source versions and original33-node Git
ancestry were not found. The live TODO redirections are navigation, not former
body copies, and printed historical commit labels do not recreate Git objects.
The earlier draft's blanket loss claim was rejected and corrected before final
handoff acceptance.

All14 individually tested historical labels below return128/unresolvable:

`6d74f76`, `bfe672c`, `3787519`, `aeff567`, `1fca893`, `023703f`, `8c39aa8`,
`0cad60a`, `2125612`, `4ab15a4`, `602c246`, `f28dc10`, `4d1b82f`, `53b360f`.

The retained pre-product body additionally names old `b283fb8` and the decision
register names `67dca72`; these were read as historical text, not individually
validated object lookups in this audit. Do not present them as tested recovery
references. The exact original33 commit list/titles is unavailable locally;
reconstructing a synthetic list would be incorrect.

Read-only fsck exit0 finds only two unreachable **recent workflow** commits,
`e2873a799ebc06ae9b97cf601e3266f68ab06357` and
`67b16e84da9ccb566f64aceb4beffb0ccf6716bf`, plus small recent document blobs.
The reflog goes back to rewritten-era `1af2ac4`; it does not contain the original
33-commit GPU/CPU development graph. Targeted inspection of unreachable blobs
finds no old full implementation aggregate or earlier phase reports. The two
unreachable workflow commits are not an alternative source of original model
history and could be garbage-collected; no recovery ref was created.

### Historical development sequence retained by accepted plans

Primary accepted/current records:

- [Accepted baseline](../plans/accepted-baseline/TODO.md).
- [Decision register](../plans/DECISION_REGISTER.md), especially U026–055 and
  consolidated U056 onward. Older rows deliberately preserve earlier status;
  do not mistake an old NOT VERIFIED paragraph for the final accepted boundary.
- [Numerics and context](../plans/numerics-and-context/TODO.md).
- [CPU/GPU optimization](../plans/cpu-gpu-optimization/TODO.md).
- [Model execution](../plans/model-execution/TODO.md).
- [Hardware targets](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md).
- Reachable historical pre-product README body from the recovery command above.

#### Design/bootstrap and CPU qualification

The original product direction was one Rust mathematical engine with immutable
source identity, private independently owned StateRoots, one authoritative
resource/event/publish path and explicit capability-qualified specialization.
P00–P23 design, compiler/placement/serving/multimodal/SSD hypotheses and fake
contract fixtures are not completion of the later real systems they propose.
Current surviving contracts supersede obsolete archive-generator and multi-writer
publication procedures.

The earlier native CPU milestone used exact original Qwen3.5 .8B/2B GGUF bytes,
co-located official config/tokenizer and a version-pinned independent native
reference. Early5 greedy/short-output checks and selected16-row GPU projections
were explicitly partial. Two12/32-token top20 failures and incomplete long-context
gates motivated independent raw-token/per-layer work rather than looser limits.

Recorded short CPU completion:8 prompts×32 tokens per model,256/256 raw greedy
choices, zero near-ties/hard mismatches and top20 logprob errors at most0.000001;
146-vector trained comparisons per model had zero nonzero-FP32 differences and
an earlier48 signed-zero distinctions for2B. The later accepted512-source scope
records56 signed-zero distinctions; these are different scopes, not an excuse to
merge result counts or relax internal equality.

Real mixed contexts advanced from .8B1024/2B512 to4096 both, then8192 both.
The recorded .8B8192 next token7993,20/20 shared top20 and0.203729244 maximum
shared logprob error passed the functional0.45 bound. Four actual16K/32K cases
were deferred by explicit user decision: accepted Phase1 is16/16 through8K;
extended gate remains16/20, not20/20. The independent8K GDN mismatch remains
open even though this functional gate passes.

#### Partial GPU owners and first physical acceptance

| Historical step | Recorded implementation/verification | Explicit limit at that step |
| --- | --- | --- |
| Initial source Q4 probe | Original first-layer16-row Q4_0 projection on CUDA4060, HIP890M and both vendor Vulkan profiles, roughly1e-8 maximum error against independent scalar oracle | Selected kernels only; no model GPU backend |
| P07-A | Source Q4_0×dynamicQ8_0 provider; host trained tensor and completed-region whole-model mock parity; fail-closed malformed/incomplete outputs and Unsupported-only fallback | No physical full-model inference |
| P07-B | Q4_1/Q8_1, Q8_0/Q8_0, Q6_K/Q8_K and F32; four original2B tensor projections and modeled248320-logit/state equality; MM0 fake contracts5/5 | Physical GPU not yet qualified; new modalities only mocks |
| P07-C | Process-unique source/tensor/codec/geometry keyed bounded weight residency, deterministic LRU, source cache and fencing; each model three host-modeled tokens gives one upload/two hits with full logits/state equality | Activation transfers, host preparation and synchronous coordination remain |
| P07-D | Decode immutable source descriptors once; six trained source tensors over four inputs;6 event tests plus MM06 barge-in mock (6/12 mock track) | Event protocol not yet actual async GPU worker |
| P07-E | Real CubeCL read_async bridge, prelaunch lease/event admission, finite/exact readback before Ready;5 source-free bridge tests and original-model host mocks | Physical readback/cancellation not yet verified |
| P07-F (old602c246) | Original source/dynamic/output GPU handles retained before launch and process-quarantined on uncertainty; combined cached-weight/transient cap;10 source-free and8 source-real host tests | Device execution then unavailable; no global worker integration yet |
| P07-G1 (old4d1b82f/f28dc10 labels) | ResourceArbiter physical-domain admission and bounded FairDeviceQueue;4 resource+4 queue tests | Eight control-plane tests are not physical GPU proof |
| P07-G2a | Attested PCI/LUID/registry domain and provider caps bound to same process-global arbiter; uncertain readback fences that device | Physical backend qualification still separate |
| P07-G2b | Real two-model NVIDIA Vulkan queue:4 physical completions,2 reused weights, exact selected source regions and one-token whole-model hybrid logits/state; AMD physically runs but strict FP32 differs and refuses StateRoot | AMD initially numerically unqualified, no complete GPU model |
| P07-G3a (old67dca72 text) | Explicit WideF64 makes original Q4_0 regions bit-exact on NVIDIA3060 and AMD Renoir Vulkan; four-token hybrid AMD parity; fair queue4/4; forced Future abandonment retains5 handles/256MiB, fences device and leaves root unchanged on both | NativeF32 diagnostic fails AMD; no driver recovery or full graph |
| U039 adversarial review | Pre-submit invalid input refusal avoids false terminal fence; scratch-aware idle LRU; one-descriptor GGUF parse/copy; CPU intermediate prefill elides discarded logits; selected/completed counters fixed; huge geometry and allocation hot-path guards | Actual trained-source/abort checks, not a scheduler/performance rewrite |
| P07-G3b | Arc<LoadedQwenModel>/ModelCatalog explicit alias/dedup/revoke; two independent sessions share one upload and two actual source-exact readbacks; advisory driver telemetry separated from logical charge | No full GPU GDN/attention/KV then; no global physical-free assertion |

Current owners: GPU `q4_q8.rs`, `mixed_ops.rs`, `host_descriptor.rs`,
`residency.rs`, `async_bridge.rs`, `physical_quarantine.rs`,
`physical_service.rs`, `physical_queue.rs`, `model_workers.rs`,
`model_sessions.rs`; core `resource_arbiter.rs`, `device_events.rs`,
`device_queue.rs`; Qwen `model_catalog.rs`. These sources are present in the
rewritten owner commits, not separate recoverable historical P07 commits.

#### H1–H25 progression into resident and native execution

This table distinguishes recorded earlier results from source-label evidence.
Original H-phase reports/commits are not locally recoverable; do not invent their
exact titles, IDs, total test counts or per-phase performance measurements.

| Step | Surviving mechanism/result | Evidence and qualification limit |
| --- | --- | --- |
| H1 | All187 original GGUF GEMV regions source-qualified on both original models and both Vulkan vendors, full logits/root | Decision U041; not full remaining operators |
| H2 | Source RMS rounding (f32 square, f64 sum, f32 mean/sqrt/reciprocal),187 matrices+49 norms=236 regions each model/card | Decision U041; independent GDN/attention still pending then |
| H3 (old4ab15a4) | Grouped per-head norms;530 actual completions over2tokens=266 selected/token minus2 final-only ops; complete248320 logits/state exact and zero outstanding logical reservation | U044; initial counter double-count fixed, no arithmetic relaxation; separate forced abandonment5 handles/256MiB fenced |
| H4 (old0cad60a) | GDN causal convolution raw pre-SiLU plus private3-frame history; source rounding fixed by stored FP32 products then ordered reduction | U045: one kernel failed1840/6144 second-token raw values with history exact; two-kernel fix yields566 actual completions/2tokens,36 raw/history checks and exact full logits/root both models/cards; forced abort6 handles/256MiB |
| H5 | GDN recurrent output AND proposed state on nonzero causal tokens; physical rounding boundaries | U047 and current `gdn_recurrent.rs`; source/current accepted completion, original standalone counts unavailable |
| H6 | GDN Q/K L2 normalization, rollback/admission/status/quarantine | U047 continuation/current `model_layer.rs`; no fabricated original hash/count |
| H7 | Polynomial source SiLU/output gate | U050/current `silu.rs` and model-layer source-bit arithmetic cases |
| H8 | Source GDN nonlinear coefficients exp/ln with exact rounding | U051/current `gdn_coefficients.rs`; stress source bits, no blanket libm portability claim |
| H9 | Attention RoPE and private current KV, nonzero positions/FMA rounding | U052/current `rope_kv.rs`; position32768 operator stress is not a32K model context |
| H10 | Bounded source QK scoring over private prefixes | U053/current `attention_qk.rs`; constructed prefix4096 operator scope is not whole-model context |
| H11 | Padded source softmax, exact exp/reductions/two-round normalization | U054/current `attention_softmax.rs`; constructed rows not long-model evidence |
| H12 | Padded value accumulation and independently proven signed-zero corner in prior CPU valid-only schedule | U055/current `attention_value.rs`; no tolerance relaxation |
| H13 | Remaining binary/residual/embedding operators | Current `operator_stress.rs` retains H13 marker; accepted baseline includes complete embedding/FFN math, original phase-specific gate count unavailable |
| H14 | Device dynamic activation quantization, nine source stress cohorts/codecs, no intermediate model readback | `activation_quant.rs`, `device_activation.rs`, `activation_stress.rs`; forced finite-input/nonzero status probe retains12 handles and fences root publication |
| H15 | Exact original phase title/ID not preserved locally | Compressed embeddings are accepted and implemented, but this audit does not assign that mechanism to H15 without a surviving label |
| H16 | One ForwardPlan binds resident values/leases, complete private state, GPU token feedback, exact full logits/state/ordered trace, restore/reset/stale-ready and independent sessions | `resident_program.rs`, `resident_inference/qualification.rs`; compact16-byte greedy feedback, no repeated source upload/intermediate activation transfer; accepted phase scope not general serving |
| H17 | State-only prefix execution and atomic/segmented prefill, skip output-only allocations/work, one final publication | `resident_prefill_qualification.rs`; original math schedule reused |
| H18 | Genuine continuous512 consumed tokens, milestones and254→257 replay across attention boundary; all42 state domains and full248320 logits; cap refuses beyond512 without root/admission change | `resident_context_qualification.rs`; snapshots survive provider with charge, final release zero; this is actual context acceptance distinct from synthetic long positions |
| H19 | Packed exact whole-range Contiguous aliases eliminate redundant physical values with one lease per original | `resident_alias_qualification.rs`; physical allocation audit, not general strided provenance |
| H20 | Cold prepared ranges/guard equivalence, malformed ranges refuse without root/admission mutation | `resident_range_qualification.rs`; timing threshold none, no model throughput claim |
| H21 | Portable v1 CPU/file/GPU state replay, exact42 domains/full logits/source/capacity/digest/Ready guards; actual upload-abort quarantine | `resident_checkpoint*.rs`; trusted diagnostic wire, not authenticated production persistence |
| H22 | Fused dense View bit-copy/certificate equivalence over original handles, guards/gaps/status/raw nonfinite bits | `resident_view_qualification.rs`; no root publication in fixture, general strided View remains future |
| H23 | Prepared projection bindings compare all10 handle coordinates to frozen H22/typed nodes, bounded host metadata | `resident_projection_qualification.rs`; host binding timing only, no GPU allocations/upload or throughput claim |
| H24 | Native CUDA/HIP CPP NVRTC/HIPRTC enablement and hardware profiles | Accepted baseline/hardware/native README; standalone LLVM code generation remains unqualified |
| H25 (old8c39aa8) | Promote reusable implementation into nine-package workspace/one lock/version/sole CLI, qualification under tools; frozen release-profile regression on five backend profiles | `RUST_WORKSPACE_AND_ENGINEERING.md`; no independent probes product; new-family libraries are source contracts |

Final accepted native/resident boundary: local RTX3060 CUDA, remote a1RTX4060
CUDA, a1gfx1150 HIP, x2gfx1151 HIP, and SDK-free NVIDIA Vulkan: **five profiles
on four healthy physical GPUs**. The fifth inventory GPU (local Renoir/gfx90c)
is held. Accepted135 profile checks+12 host fingerprints must not be summed with
earlier step checks as if they were disjoint. Separate main model merge recorded
135 ordinary Rust executions,23 original-model executions across22 distinct names,
two exact8-token CLI outputs and one official metadata case.

Native providers use the existing CubeCL C++ path and existing runtime/compiler
libraries/headers. Separate LLVM23 device codegen remains unqualified. The a1
HIP profile retains its measured glibc2.44/GCC16.2 loader/scalar/libm closure.
Current packaging release profile differs from the old independent probe;
actual newly packaged binaries were separately qualified, not inferred from old
binary hashes or compilation.

#### Hardware discovery and local AMD hold

Historical target inventory: three CPUs/five physical GPUs, not five accepted
GPU cards. Discovery is a capability inventory, not measured available memory
for current admission or live machine state.

| Host/profile | Retained discovery and boundary |
| --- | --- |
| Current CPU | Ryzen7 5800H,8C/16T,NUMA1,AVX2/FMA/F16C, noAVX512; L1d32KiB/L2512KiB/L316MiB |
| a1 CPU | RyzenAI9 HX370,12C/24T,4fast+8slow,NUMA1; AVX512/VNNI/BF16 capabilities; cache hierarchy48KiB/1MiB/16+8MiB |
| x2 CPU | RyzenAI MAX+395,16C/32T,NUMA1,48KiB/1MiB/2×32MiB; capability-specific optimization future |
| Current NVIDIA | RTX3060 Laptop sm86,6144MiB, PCI01:00.0; accepted CUDA/NVIDIA Vulkan profiles |
| Current AMD | Renoir gfx90c,8CU/wave64/shared RAM; later2B initialization SIGABRT and automated resets wedged; **no launches/resets/recovery experiments authorized now** |
| a1 NVIDIA | RTX4060 Laptop sm89,8188MiB,PCI64:00.0; accepted CUDA |
| a1 AMD | Radeon890M gfx1150,16CU/wave32,PCI66:00.0;512MiB aperture+15.23GiB GTT are NOT additive independent memory; accepted measured HIP closure |
| x2 AMD | Radeon8060S gfx1151,40CU/wave32,PCIc6:00.0; nominal64GiB shared pool/observed67.04GB decimal host; accepted HIP |

Remote AMD Mesa26.1.8/Vulkan1.4.354 and NVIDIA595.91.07/Vulkan1.4.329 advertised
features are discovery facts, not qualifications for every precision/shape.
Local AMD was physically used in earlier Vulkan phases but its subsequent hold
supersedes those historical examples. No current documentation task authorizes
using it or changing device/driver privileges. SSH discovery/temporary historical
qualification also does not authorize future downloads or persistent config.

#### Layout, new-family contracts and concurrency disposition

Nine-package layout: core, artifacts, CPU, Qwen3.5, Gemma4, GPU, facade, sole CLI,
and GPU qualification tooling. Patched CubeCL packages remain excluded vendor
members; family/text/CPU dependencies are absent from production GPU. Curated
`all-models` describes both family contracts, not executable support for every
checkpoint. One workspace version/lock; downstream patch declarations are not
transitive. Source/feature changes need their relevant fresh gates.

Qwen3.8/Gemma4 implementation is source metadata/geometry/tensor/routing contracts
only. Recorded Gemma family14 tests and official metadata do not establish trained
weights/tokenizers/CPU or GPU inference. Current owned GGUF executor retains4GiB
cap; larger BF16 sources exceed the local accepted source budget. Shard/index/
header/name/dtype/offset/shape/fullhash/admitted-owned-copy path remains separate
unfinished work; mutable mmap is not an accepted shortcut.

Thoughtful left-right review followed layout. The cold model-catalog proposal had
no demonstrated contention and was restored, with **no accepted migration or
speedup**. Future contention/publication/replay/reader-retirement evidence is
required. Immutable Arc sharing and existing narrow locks remain accepted.
No new dependency or unnecessary double-buffer publication was added to fill a
plan checkbox.

### Tracking/archive/checkpoint chronology

Historical checkpoint labels below are accepted record labels, not currently
resolvable commits. Reachable replacements/current bodies were checked separately.

1. Older generated full handoff evolved from89 embedded source records to an
   aggregate duplicating191 individual sources. User later explicitly replaced
   that process with plain TODOs; do not resurrect its generators/manifests/SHA
   inventories or impose obsolete byte-receipt duties on future implementation.
2. Historical `bfe672c` simple-TODO migration: single index/per-plan/agent-run
   hierarchy;5 archive/generator/SHA tools and4 check.sh invocations removed while
   every real implementation check was preserved. Retained final full gate194
   pass/27ignored, strict lint/release/installer/shell/Nix/reference plus18 namespace
   fixtures. Initial7 stale pre-layout links fixed;44 live links in13 workflow
   documents resolved. No implementation/dependency/root-README delta.
3. Historical `aeff567`: retire70 obsolete Markdown records:42 GPU phases/notes,
   20 CPU/model/concurrency/checkpoint reports,6 entrypoint aliases and2 takeover/
   layout records. Preserve source rounding/quantization/padding/new-family facts
   in technical docs and current TODOs; keep8K/schedules/deadlines/recovery/actual
   memory/View provenance limits open. Rejected-LR metadata JSON removed after
   ownership review; no model/SDK/build cache cleanup inferred. Full gate194/
   27ignored+18 namespace;114 current first-party document links and GPU144links/
   23documents pass.15 then-retained archive bodies matched the old checkpoint.
4. Current graph begins after those reports were removed; `be95424` already holds
   simplified current documents plus the remaining15 old design/brief bodies.
5. Reachable `a054d39`/`cc3cab2`/`7e61cb6` strengthen TODO/contribution/consent
   workflow. Reflog superseded two earlier workflow objects only.
6. Reachable `9239719` retires those remaining14 files plus replaced pre-product
   body after distinct rationale/open tests are integrated. Keep SSD01..32,
   QTIP/HYB/GFX, MM01..24 and77 family fixture requirements **planned**. One gate
   failure from a copied CLI test's embedded retired checkout was resolved by
   test-source mtime refresh with bytes unchanged. Final full gate passes; no
   additional result count invented where this section records none.
7. Current acquisition completed sections are plain Markdown archives, not a new
   evidence system: through-fce1353 holds10 completed groups; through-a774ded
   holds8 more; all unfinished tasks stay live. Rolling prior-commit TODO updates
   are carried with the next implementation commit, not separate self-reference
   commits. Runtime/user pause at current quant acceptance remains effective.

### Acquisition chronology and accepted gates

Sources read in full (bounded chunks):

- [First completed TODO archive](../plans/model-acquisition/archive/through-fce1353/TODO.md).
- [Second completed TODO archive](../plans/model-acquisition/archive/through-a774ded/TODO.md).
- [Live acquisition TODO](../plans/model-acquisition/TODO.md).
- [Acquisition design/frontier](../plans/model-acquisition/README.md).

Each row below is an implementation phase. Full-gate counts are actual Rust
**executions across feature configurations**, not distinct test names, and are
not additive across rows. Optional/child/helper ignores are intentional; ignore
counts do not mean those cases passed. Separate relevant physical/model selections
were run explicitly where stated. All final gates listed have0 failures.

| Commit/phase | Resulting behavior and mandatory boundary | Recorded complete frozen gate |
| --- | --- | --- |
| old3787519 / rewritten cache owners | A1 local Linux exact schema/path/admission/staging/fullhash/EOF/atomic manifest/ref publication. Trusted source identity/inference remains separate.18 operation types/38 callbacks exercised76 injected I/O+76 actual child exits; retained ref backing/private snapshot peak/hardlink ctime fixed. Cooperative quota is not global host/UMA budget; sameUID mutation not prevented. |194 pass/27ignored |
| fa6affa | A2 optional synchronous HTTPS owner with TLS/origin credentials/limits/retries/deadlines; pinned HF/GH commit/tree or release/asset; required shards/index/splits/config/tokenizer/sidecars; unknown raw digests first-seen fullhash. Offline-before-client; no default CPU HTTP/TLS. ActualTLS/range/redaction fixtures,40-boundary discovery publication matrix. |312/30ignored |
| 3c97c52 | Durable **proper** exact prefixes bound to immutable source/length/digest/strongETag/inode/prefixSHA; rehash+copy using original attempt/publisher.8MiB/1s checkpoints; file-before-atomic-record order. Exact-total prefix cannot bypass terminal EOF; zero-progress retry does not duplicate retained copies. Original dead stages remain charged. Full old+new+24KiB/file peak and one digest lock. Progress separates cached/resumed/downloaded; signals optional. |353/34ignored |
| baa9d88 | Explicit independently pinned component snapshots/role suppliers/per-label transports and cumulative budgets, no atomic composite source. Descriptor-based actual-owned GGUF copy hashes/reparses before sole issuer;4GiB cap and exact raw ImmutableWeights admission/lifetime. Positioned bounded config/tokenizer copies1MiB/64MiB and same Qwen math. HTTP-free run-cache/offline pure opening. |419/54ignored/75suites |
| 20f449d | Missing selected exact sizes from pinned LFS or body-free exact HEAD; missing≠zero, strongHEADvalidator retained. No transport-error GET fallback. Concurrent fill typed verified-hit under same locks avoids redundant payload. Cancellation reaches transaction hash/control-copy and online final open; legacy wrappers remain. |449/54ignored/81suites |
| f44264c | Fresh-root protected policyv2 actual OFD shared pins, whole-transaction activity barrier, safe exact unreferenced snapshot retirement/restart. Existingv1 byte/charge compatibility, no migration/removal. Typed Busy/Referenced/Durable/Incomplete and named-inode charge, no global physical-free claim.34 I/O+34 actual exits; independent charge/lazy/dup/inheritedFD tests. |518/56ignored/94suites |
| ea700f6 | Explicit positive AtMost for Auxiliary only, fullcap admission and max per trusted-digest cohort; exact observed manifest/receipt bytes at terminalEOF. No cap durable resume. HEAD fallback only validated200missingCL/405/501; cap+1 custom/HTTP read guard. Required/model closure stays exact. Intentional exact getters→Option API change. |553/57ignored/103suites |
| 03b6948 | Repeated release component aliases reuse typed first releaseID per provider/repo/tag; per-label asset listing/credentials and all cumulative budgets persist. Pure offline allows distinct assetIDs only same canonical releaseID. ReleaseID does not freeze mutable asset set or atomic cross-asset membership.147 scoped+14 independent executions. |569/57ignored/106suites |
| e44f4ee | Explicit local HF root/repo40commit/path/role/import. Readonly originalFDs, stable pre/postcopy checks; metadata-only source topology before any same-root destination initialization; no source locks/writes/hardlinks/refs. Persist local-hf-cache provenance, FirstSeen unlesscallerSHA. Knownhit still hashes source with copied0. Existing absolute0700 target only. |625/58ignored/116suites |
| fce1353 | Crossfs only distinct-dev+mount Btrfs↔tmpfs and distincttmpfs↔tmpfs, each tree/selectedFD on own dev+mount, ancestry/root/intersection checks. Complete source metadata inventory/hardlinks; sameUID races remain operation-local. Real6diagnostic+5production physical selections, not hiddenbind/subvolume/overlay proof. tmpfs fsync not reboot persistence. |633/68ignored/119suites |
| e0073a5 | URL-free HttpsObjectSpec: normalized bareorigin+logicalpath/role/exactbytes/callerSHA contract, secretURL/query onlymemory.1MiB admitted prepare before parse; noHEAD. Pure offline managed one-entry fullhash/trust/source/pointer check before URL/env/client. CLI bounded nofollow URL-file, no ref firstslice.160 API+35 independent focused. |653/68ignored/126suites |
| e82d76b | Native OCI/Docker schema2 unique/explicit rawGGUF descriptor, rawmanifestSHA/media/digest/exactsize checks. Explicit anonymous/bearer/fixedDockerHub pull broker; no arbitrarychallenge/login/refresh/container/index/compressedpaths. One originalHTTP/executor/publisher. Offline no independent retained-manifest membership/freshness claim. |672/68ignored/133suites |
| f74a525 | GCS JSON object exactgeneration+metageneration/size, locally derived querypin+precondition; optional-present generationheaders must agree. Media nofollow prevents query-loss. Identityencoding/keys/softdelete refusal; callerSHA orFirstSeen. BoundedAux prefix independent plans/noatomicbucket. Typedpins/receipts and pureoffline. |693/68ignored/140suites |
| 833d68f | Nonnullversioned S3 native/explicit compatible endpoints, boundedV2/versionlist, static/session SigV4. Opaque exactURI typedtarget+defaultUnsupported customtransport capabilities, zero redirects. Mandatoryexactversionheader/no-delete beforeeverybody, independentETag/range/hashEOF. Officialliteralvectors+actualwire signatures/rawpaths. |718/68ignored/147suites |
| e622222 | Native Azure BlockBlob explicitVersion/Snapshot, exactHEAD+transientETag, typed source/offlinepin. Mandatorytype/encoding/structured/key/copyguards; optionalpresentversionecho. Public/bearer/accountSharedKey/serviceSAS boundedprofile/API2023-11-03; rawSAScredentialowner only. Strict XML currentversionAux prefix and aggregate control tally. |749/68ignored/154suites |
| 87f6454 | ModelScope explicit40commit direct native boundedshallowtree/files and officialtoken Bearer+cookie in one originalHTTPowner. Nofollow/credentialprofile isolation; exactrawSHAifprovided elseFirstSeen. Source membership/revision serversemantics trusted, no aliasauthority. Digestofflinepointer refusal fixed RED→GREEN. |768/68ignored/161suites |
| 223ca79 | Pure then-unwired Xet32byte/text/keyedMerkle/hash/codec primitives and official/scalar/framegoldens. NONE/LZ4/BG4 boundeddecode/cancel; native transfer notyet enabled.13 actualprimitive tests before/after mechanicallint. |781/68ignored/164suites |
| a774ded | Explicit S3/Azure current-object **callerSHA+exactbytes** contentcontracts; nofakeversion, noHEAD during admittedprepare/dryrun. Originaltypedtargets/auth/executor/resume/fullhashEOF/publisher; distinctsource/offlineTrustedcontract. AzureSAS b/c currentcontent whileoldbv/bs scopesretained.488 focused/all4strict, nofix. |802/68ignored/171suites |
| 837e77f | Legacyoffline pointer/closure/indexbudget eligibility:5 intendedRED+3positives beforecode. Shared128B managedprefix; legacy ordinary Files role-agnostic semantics retained. ExplicitcomponentWeights enforce complete split/index/sidecars, one cumulative index-control budget.215focused/all4strict. |810/68ignored/174suites |
| 52e42fc | Explicit nativeHF Xet through same pin/tree/closure/HTTP/executor. SealedHub/CAS/signedinterval grants, scopedexpiry/nofollow/credential suppression; completeordered/repeatedterms/multipart/codec/EOF/filekeyedroot+rawSHA. Invocation-wide work/wire/control/decode/leaf/refresh caps+admittedscratch. Nativeephemeral fullrestart; ordinarysidecars retainresume.551artifact+175CLI/all4strict. |867/69ignored/181suites |
| 46cf5a5 | Explicit no-callerSHA conditionalcloud: admittedHEADexactbytes+strongETag, typedIf-Match signed on everyinitial/range/retry and exactresponseETag beforebody;412neverrefresh/fallback. FirstSeenactualrawhash, noimmutableversionclaim; pureofflinepin+snapshotDIGEST beforesecrets.872focused/all4strict. |886/69ignored/188suites |
| 4db5efb | ModelScope selected/index absentSize uses exactsamecommit nofollowHEADonce under originalcumulative metadataBudget. MalformedPRESENT/null stillrefuses,unselectedneverHEAD; strongETag checks,weak/absent noresume.891focused/all4strict; stalebuild resolutionbelow. |896/69ignored/191suites |
| f22f759 | Metadata-only HF/GH candidates through originalpinnedtree, noHEAD/index/payload/publication. Fullinventorycollision beforeprojection; observedsize provenance and cachedsubset scopehonest.943focused/all4strict; staticcollision fixbeforefirstexecution. |915/69ignored/198suites |
| f197c9c | ExplicitGGUF filenamequant withinsameoriginaltree/pin/closure/nativebudget; boundedASCII1..32tag, terminalbasename boundaries, canonical5digitcomplete splits beforeuniqueness; nofallback/eligibility. Strictnewquantoffline roles/source/fullhash/pointers/DIGEST beforesecrets.975focused/all4strict; nativequantfixture planonly. |934/69ignored/205suites |

All full phases use the isolated frozen command:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh
```

This includes feature/dependency isolation, formatting/strict lint/build/release/
installer,18 namespace-wrapper tests and comparator/resume self-tests as recorded
per phase. Actual Nix filtered-source integrity and local-link/rootREADME checks
were separately run; new retained fixture paths were explicitly curated. Fixture
HTTP/TLS/cloud traffic is owned local synthetic traffic, not live provider/account
or externally downloaded model qualification.

#### Original-model regression route ledger

Both approved model hashes remain:

- .8B `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`.
- 2B `91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6`.

A3 and later source tests use both originals,2tokens/model/nonzero state,
all248320 logits and all42 full state-domain vectors **bit-for-bit**, independent
sessions and immutable source allocation IDs, foreign-source restore refusal,
rawsource lease lifetime until last owner, ref/private-cache mutation isolation.
Protected-removal acceptance additionally removes the private cached snapshot/
weight while the owned model and its reservation survive and remain numerically
exact. HF-import route additionally copies into a synthetic explicit HF layout,
imports exact known hashes, removes the synthetic source before preparation and
reuses the same numerical/state/ownership tail. This does not requalify8K or a
new family/GPU filesystem execution path.

The original selected release test is
`approved_cached_models_match_direct_source_exactly`. The second is
`approved_hf_import_models_match_direct_source_exactly`, both in
`crates/zerogravity/tests/cache_prepare.rs`.

| Phase | Actual recorded selected release result/duration |
| --- | --- |
| baa9d88 | Both original models accepted; separate final selected test18.49s recorded in archive |
| 20f449d | Original cache route1test21.61s |
| f44264c | Enhanced actualremoval route1test19.81s |
| ea700f6 | Original route1test20.02s;43.49s prior compilation separate |
| 03b6948 | Original route1test20.03s;cachedcompile0.14s |
| e44f4ee | Bothroutes2tests45.15s;releasecompile45.18s separate |
| fce1353 | Bothroutes2tests42.20s;releasecompile45.24s separate |
| e0073a5 | Bothroutes2tests40.55s;preparation0.14s separate |
| e82d76b | Bothroutes2tests41.86s;compile0.15s separate |
| f74a525 | Bothroutes2tests41.83s |
| 833d68f | Bothroutes2tests41.06s |
| e622222 | Bothroutes2tests42.85s |
| 87f6454 | Bothroutes2tests42.91s |
| 223ca79 | Bothroutes2tests41.70s |
| a774ded | Bothroutes2tests40.55s |
| 837e77f | Bothroutes2tests42.34s |
| 52e42fc | Bothroutes2tests43.26s;44.87s rebuild separate |
| 46cf5a5 | Separateexactcalls17.40s and23.69s;not onecombined invocation |
| 4db5efb | Separateexactcalls17.64s and23.60s |
| f22f759 | Separateexactcalls17.47s and23.65s |
| f197c9c | Separateexactcalls17.60s and23.70s |

Recent exact commands (one invocation per NAME, accurately report each timing):

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py --   cargo test --release --locked --offline -p zerogravity --features cache   --test cache_prepare approved_cached_models_match_direct_source_exactly   -- --ignored --exact --nocapture
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py --   cargo test --release --locked --offline -p zerogravity --features cache   --test cache_prepare approved_hf_import_models_match_direct_source_exactly   -- --ignored --exact --nocapture
```

Earlier combined routes were selected with
`--test cache_prepare -- --ignored --test-threads=1 --nocapture` under the same
wrapper, not a fabricated single timing for the later filtered commands.
These are historical commands; **do not run while implementation is paused**.

#### Substantive failures/corrections that explain acceptance

- A2 transport initially misinterpreted whole-body timeouts and allowed TRACE
  URL/Location exposure. Header/total deadlines were separated; acquisition caps
  logTRACE and documents downstreamfeature constraint. A first gate stopped130
  for reviewed redaction; a next127 came from editing its active Bash file.
  Final script frozen and full312 passed;40-boundary matrix superseded an early
  incomplete four-boundary experiment.
- Durable-prefix review found terminalEOF bypass at savedbytes==total, repeated
  unchanged checkpoint copies, canceled returned chunk losing prior accepted
  prefix, and public known-digest private-copy quota escape. Fixes preserve
  proper-prefix-only checkpoints, original old+new peak, explicit checkpoints,
  accepted-prefix retention and stable digest authority.13cachepartial tests,
  16 I/O+16 real exits across8 initial/replaced-checkpoint boundaries pass.
- A3 review rejected publicly mutable geometry as authorization; normal/cache
  family constructors reuse validated config and reject ineligible family before
  rawallocation. Exact bounded owned-copy actualSHA/EOF/reparse precedes issuer.
- Protected-removal first fullgate intermittent absent-target retry was Busy at
  callback19 because unrelated test-thread childspawn inherited allprocess OFDs
  beforeexec. A test-only first-acquired/last-dropped mutex serializes the unrelated
  process fixtures;10 parallel reruns pass. No productionretry/timing weakening.
  Actualnamedcharge is not globalphysicalfree-space attestation.
- Cappedstream tests exposed custombody overread needing cap+1 clamp, and a final
  same-byte progress event was required onlyafterEOF/filefinish. Fixtures using
  `.bin` correctly triggeredWeights classification and were madeAux`.txt`; no
  rolepolicy weakened. No independent fullHTTPtrailers validation was invented.
- LocalHF initial2source failures were42-character supposed40hex fixture ID;
  constant/chain fixture corrected. Independent socket fixture exceededSUN_LEN;
  bind shortprivatepath thenrename retainedsame specialinode case. Two test-only
  default-field assignments fixedlint; no sourceacceptance relaxation.
- OCI CLI syntheticseed omitted TrustedSha256; one genuinefailure/two poisoned
  mutex cascades fixedprovenancefixtureonly. Signing attempt withoutSSHagent
  exit128 had nocommit; existingagentsocket succeeded, signing neverdisabled.
- S3 XML review caught outside-root references/repeateddeclaration/XML1.0invalid
  chars beforefirstcompile. Diagnostic originparser wronglyrejecting HTTPS was
  repaired beforeexecution. OneMaxKeys collapsible_if lint and onehelpcharacter
  failure+three mutexpoison cascades received mechanicalrepairs only.
- Azure firstcompile101 beforetests: wrongexistingcheck_cancel path and equivalent
  quickXML deprecatedwrapper. Subsequent shortbodyfixture omitted two legitimate
  originalrange retries; corrected scriptedrows, notruntime retry policy. Wrapped
  helpassertion+three mutexcascades fixed whitespace-normalizedtest; oneinclusive
  svwindow lint mechanical. Wirecanonical/HMAC expectations unchanged.
- ModelScope initialindex positive lacked mandatorytotal_size; actual-loopback
  tests neededdiagnostic-only cfg. Correct-hash manuallyseededLFS pointer first
  failed atintendednegative, thenmanagedpositionedprefix guard passed. No general
  CacheStoreliteral-byte validity was redefined as loadereligibility.
- PureXet independentgoldengenerator copied65/66hex XOFcharacters into32byte
  expectations beforeRust; fixedofficialprefixes first, verifiedofficialvectors,
  frozecustomgoldens. Official0..31textexample omitted0x10; correctsequence plus
  verbatimsavedsource vector retained. No expectedresult fittedtoRustoutput.
- NativeXet firstE0624 testaccessfixedtestonly; teeouterexit0 hadhiddenCargo101,
  henceinspectactualstatus. A payload.binauxfixture invokedrealmodelclosure:
 19failures correctedvalidAuxpath pluscompleteWeightsclosurepositive. Remaining
  StatevsLimit sanitizedexpectation and CLIgrammar correctedtestsonly. Ordinary
  sidecar retry restriction foundbyroot narrowedtoallowOrdinaryPayload only.
- ModelScope sizing firstdiagnostic run156librarypass/new1pass9fail contained an
  oldmissing-size literal absentfromlivesource butpresentin retainedrlib. Root's
  SIDEcopy2 preservedmtime olderthanpreviousconditionalbuild. Afterbyte-equality,
  only5mergedfile mtimesrefreshed; identical453pass rerun. No sourcepolicy/fixture
  expectationchanged; futuremergesusefreshmtimecopyfile, notworkspacecargo clean.
- Candidates fullinventoryregular-ancestor collision hiddenbyprojection caught
  beforeexecution; orderingfixedwithHF/GHnegatives/real-directorypositive.
  Quantdirectory`-of-`/tag counterexample repairedbybasename-onlyinspection before
  execution. Both finalphases report **no executedRED/sourcefix** fromthese static
  findings. Quantinterruption foundall11frozenMAINpathsunchanged.

### Still-open limits, not erased by accepted chronology

- CPU independent8KGDN numerical root cause: first .8Blayer4/2Blayer1 mismatch;
  nativeubatch512 sharedtop20 logprob errors0.20372924/0.19678696 remain. Greedy
  token7993 matching does not close it.2B native schedule changes; .8B schedule
  control differs. Compare pinned native0.6.0/build11429/d812350 CPU/F32KV/nonflash,
  rawteacherprefix512/1024/2048/4096/8192 andubatch1/64/256/512. Fullubatch1historical
  capture unfinished. Keep0.45sharedtop20/0.05toptokenprob functionalcriteria and
  strict sourcebitoperatorcriteria separate, never loosen to claim parity.
- ResidentGPU beyond512, genuine16K/32K, additionalmodels and five-device fleet
  optimization are unfinished. Synthetic32768positions/4096paddedrows are operator
  tests. Localgfx90c hold persists. NativeF32FMA/AVX2/AVX512/VNNI/specializedpaths
  and realthroughput gains remain separatelymeasured proposals; WideF64 accepted.
- Actualdriver cancellation/recovery/physicalfinalfree and multistream/deadline
  scheduling remainopen. Logicalfence/quarantine successes do not imply recovery;
  queueddeadline wakeups requireactualscheduling/poll opportunities.
- Trustedportablev1 replay is notauthcheckpoint persistence; tokenizer/sampler/
  grammar/media serialization/security UX remainsopen. ModelcatalogLRcontention
  notdemonstrated; noacceptedLR migration/performanceclaim.
- CloudcurrentcallerSHA, conditionalFirstSeen and versionedprofiles have distinct
  contracts. Broaderendpoints/partitions/ambientcredentials/checksumauthority/
  Azurelatest/SASprofiles/prefixdigestmaps stillopen; noETag/MD5/CRC/GitOID upgrade
  to trustedrawSHA. S3FULL_OBJECTchecksum qualification separate.
- ModelScope aliases/redirectCDN/mirror/refresh/nativehubparity remainopen; no
  immutablecommitauthority from callerrevisionecho/timestamp/filelastmodified.
- NativeXet requiredLFS/Xetindex302stillUnsupported; ordinaryindexworks. Native
  partial-xorb/pertermproof notinvented; nofulltrailersection/wholeoperationdeadline
  orcoalescing/throughputclaim. Signed-setoverfetch/refetch countsactualwork.
  Nativeprocessrestart retransfersfullnativepayload, ordinarydurableresume separate.
- v1cachemigration, deadstage/orphan/generalGC, furtherlazy/raw/mmapconsumerpins,
  unmanagedreaderremoval and additionalfilesystempairs remainUnsupported/unqualified.
  ManagedFDpin and observedunique-inodenamespacecharge do not provekernel/process-
  globalphysicalfree. Actual hiddenbind/subvolume/overlay fixtures neverrun;
  authorizing ordinaryexistingtmpfsfixtures did not authorize mounting.
- LocalHFimport is explicitreadonlyexternalcache selectedcopies, noimplicit
  discovery/closure/modelreadiness. SameUID stability checks areoperation-local,
  notanatomicexternal-tree snapshot. Additionallayouts/newdestinationcreation/
  missingmountidentities remainopen. Localoriginassertion isnotremoteHFmembership.
- Finiteprovider/loaderparity remainsopen despite many accepted slices. Newsource
  hashes/licenseauthorization do not authorize opaquepluginruntime or trained
  newfamily inference; noentirellama.cpp/vLLMunioncompletionclaim.

### Paused source drafts that must survive handoff

1. `/home/pvl/spaces/abird/src/zerogravity-filters-work`: C3 base4db5efb plus
   acceptedC1/C2overlays, onlynew`acquisition/filters.rs` differs. UNWIRED and NOT
   compile-ready; RusttestsUNRUN;8integrationpaths unchangedC2,2testfilesabsent.
   Preserve73primaryoraclecases+33345pairs. GrantedgrammarFullpathFnmatchCaseV1,
   includeunion→exclude→ambiguity, requiredveto/optionalskip; adjacentindex checked
   beforeGET.512KiBscratch/16Miwork andextra16MiBsameowneradmission stillneedactual
   integration/qualification. Formatting/whitespace success isnotRustacceptance.
2. `/home/pvl/spaces/abird/src/zerogravity-xet-index-work`: C4 basef22f759+frozenC2;
   onlyHF and`xet/{mod,budget}.rs` drafted. Explicitdefaultfalsebuilder/private
   callback/all-counterResultseed+2units UNCOMPILED/UNRUN;CLI optin/help/integration
   absent. RequiretrustedLFSrawSHA+completekeyedXetroot+EOF beforeJSON, carryall
   countersthroughresolution→execution, reconstructagainforpublication counted
   cumulatively. Default/dryrun noCAS; explicitapply optsoptionsclonein. Reconcile
   C2/eventualC3 overlaps beforeacceptance; currentnativeindexsupportnotadvertised.

CompletedquantSIDE+4integratedruns were retired after15exactcurrentbaseline/
ownedmatches, nochildprocessuse andpreservedusefulcolddesign. Interruptedfilters
MAIN/SIDEruns andbothunfinishedSIDEs remain. Do notreplace theseactualfrontiers
with an obsoletehandoffentrypoint orrestart alreadyacceptedquantwork.

### Audit limits and coordinator use

- This audit onlywritesitsown2runfiles. Parent-owned newhandoff/docs/TODO edits
  were observed in status and leftuntouched. NoGitstaging/commit/ref mutation.
- Oldacceptedlabels remainhistoricalauthority, and currentexact77IDs remain
  Gitprovenance. Testsareonlyclaimed fromcheckedretainedplans; currentimplementation
  audit hasnotfreshlyexecutedthem.
- Original33fullledger/perphaseH5–H23exactcommitIDs/titles and later129/191 aggregate versions
  cannotbe suppliedfrominspectedlocalhistory. A future externalbackup/explicitly
  authorizedfetch mayyieldthem; no suchsourcewasqueriedand norecoverypromised.
- Parentcanintegrate thiscontextintoitscomprehensivehandoff andthenretirethisrun
  afterallusefulresults/unknownsarepreserved. Do notreinstatearchive/SHAautomation.


<a id="recovered-history"></a>

## Recovered original94-source history: exact reading catalog

Author: Codex history-audit agent. Provenance: current implementation `f197c9c`;
historical reading reference `7e61cb63efd0c1429aa96044f2f7d28fda1e59aa`.
Documentation-only; runtime remains PAUSED. No source/Git mutation, Cargo, network,
model, SDK, device or SIDE operations. This file supersedes **only the overly
broad historical-recovery negatives** in the initial history-audit draft. The corrected recovery scope is integrated here.

### Correction and precise recovery boundary

The original audit checked standalone trees, refs, reflog and unreachable objects,
but failed to inspect implementation text nested inside the recoverable multimodal
aggregate. Its statement that the old implementation aggregate and all70 phase
reports were unrecoverable was too broad. Several earlier bodies survive inside
a reachable historical file despite missing standalone paths and original Git
objects. The coordinator was notified immediately during cold review.

The corrected distinction is:

- The current77-commit graph remains verified. The original33 exact commit ledger
  and pre-rewrite Git graph are not recovered. Historical labels in report text
  are not working Git object references.
- The recoverable outer15October multimodal aggregate embeds an **older full
  implementation handoff through P07-G3b**, comprising an executive brief and
  **94 numbered source sections**. It also has13 v5.1 appendices and3 then-live
  shared snapshots. These are historical text, not fresh runtime qualification.
- The later H4-era aggregate, including its original33 ledger and expanded
  inventories (129/191 in historical discussions), is not recovered. The70-report
  set removed in later cleanup is not recovered **as a complete set**. Several
  earlier members ARE recoverable below. Do not claim all70 bodies are missing.
- No H4/H5–H25 implementation report was found inside this94-source version.
  Later retained decision/accepted-plan summaries still supply those phase facts.
  H15 exact title/ID remains unknown. No original object was restored.

### Tested read-only commands and wrapper positions

All following line numbers refer to this specific historical Git object's output,
not the newly assembled handoff. Prefer numbered headings as stable reading aids.

```sh
# Older full implementation executive/index (read progressively).
git show 7e61cb6:.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md | sed -n '3931,4234p'
# Source1 example; every source has its own exact range below.
git show 7e61cb6:.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md | sed -n '4235,4555p'
# Concrete recovered G1, G3b and P00 reports.
git show 7e61cb6:.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md | sed -n '11875,11960p'
git show 7e61cb6:.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md | sed -n '12311,12367p'
git show 7e61cb6:.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md | sed -n '20563,20647p'
```

The nested implementation-history wrapper begins3931 and title3939. Its94-body
sequence begins4235. All94 start/END markers were checked. Inclusive ranges below
include original metadata/fences/end markers, without requiring any hash/byte
inventory. This is a reading catalog, not a revived archival workflow; do not
restore the old tree or execute obsolete command/generator instructions.

### Complete94 nested-source title and inclusive-line catalog

| No. | Exact historical section title | Start line | End line |
| --- | --- | ---: | ---: |
| 1 | README.md | 4235 | 4553 |
| 2 | .agents/plans/DECISION_REGISTER.md | 4556 | 4825 |
| 3 | .agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md | 4828 | 4999 |
| 4 | .agents/plans/IMPLEMENTATION_ROADMAP.md | 5002 | 5421 |
| 5 | .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md | 5424 | 5843 |
| 6 | docs/WHITEPAPER.md | 5846 | 6296 |
| 7 | docs/ENGINEERING_SPECIFICATION.md | 6299 | 7259 |
| 8 | .agents/docs/VISION_SCOPE_AND_PRINCIPLES.md | 7262 | 7397 |
| 9 | .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md | 7400 | 7590 |
| 10 | .agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md | 7593 | 7917 |
| 11 | .agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md | 7920 | 8173 |
| 12 | .agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md | 8176 | 8396 |
| 13 | .agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md | 8399 | 8551 |
| 14 | .agents/docs/SEMANTIC_IR_AND_COMPILER.md | 8554 | 8823 |
| 15 | .agents/docs/ASYNC_RESOURCE_COMMAND_IR.md | 8826 | 9043 |
| 16 | .agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md | 9046 | 9205 |
| 17 | .agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md | 9208 | 9532 |
| 18 | .agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md | 9535 | 9836 |
| 19 | .agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md | 9839 | 10109 |
| 20 | .agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md | 10112 | 10371 |
| 21 | the master handoff | 10374 | 10862 |
| 22 | .agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md | 10865 | 11238 |
| 23 | .agents/plans/the agent entrypoint | 11241 | 11398 |
| 24 | .agents/plans/CURRENT_STATUS_AND_TRACKING.md | 11401 | 11749 |
| 25 | .agents/plans/the next-agent prompt | 11752 | 11872 |
| 26 | the retired P07-G global-resource-arbiter handoff | 11875 | 11958 |
| 27 | the retired P07-G2 physical GPU-service handoff | 11961 | 12007 |
| 28 | the retired P07-G2b physical fair-GPU-workers handoff | 12010 | 12104 |
| 29 | the retired P07-G3a strict AMD Q4 and abort handoff | 12107 | 12196 |
| 30 | .agents/plans/FRESH_CODE_CORRECTNESS_ARCHITECTURE_REVIEW_2026-10-15.md | 12199 | 12308 |
| 31 | the retired P07-G3b shared-models handoff | 12311 | 12365 |
| 32 | .agents/prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md | 12368 | 12473 |
| 33 | .agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md | 12476 | 12711 |
| 34 | .agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md | 12714 | 12861 |
| 35 | .agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md | 12864 | 13198 |
| 36 | .agents/docs/COST_MODEL_AND_AUTOTUNING.md | 13201 | 13418 |
| 37 | .agents/prior-art/DESIGN_EVOLUTION_V3_HANDOFF_SNAPSHOT.md | 13421 | 13640 |
| 38 | .agents/prior-art/DESIGN_EVOLUTION_V3_SNAPSHOT.md | 13643 | 14541 |
| 39 | .agents/docs/DEVICE_KERNEL_AND_CODEGEN.md | 14544 | 14724 |
| 40 | .agents/docs/DISTRIBUTED_PARALLELISM.md | 14727 | 14791 |
| 41 | .agents/prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md | 14794 | 14863 |
| 42 | .agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md | 14866 | 15082 |
| 43 | .agents/docs/EXAMPLE_EXECUTION_TRACES.md | 15085 | 15251 |
| 44 | .agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md | 15254 | 15382 |
| 45 | .agents/prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md | 15385 | 15543 |
| 46 | .agents/docs/MEMORY_RESIDENCY_AND_IO.md | 15546 | 15808 |
| 47 | .agents/prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md | 15811 | 15896 |
| 48 | .agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md | 15899 | 16131 |
| 49 | .agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md | 16134 | 16368 |
| 50 | .agents/docs/OPERATOR_AND_FUSION_CATALOG.md | 16371 | 16565 |
| 51 | .agents/docs/PERFORMANCE_HYPOTHESES_AND_RISKS.md | 16568 | 16722 |
| 52 | .agents/prior-art/PRIOR_ART_HISTORICAL_ACCEPTANCE_GATES_V2.md | 16725 | 16841 |
| 53 | .agents/prior-art/PRIOR_ART_HISTORICAL_DISTRIBUTED_PARALLELISM_V4_1.md | 16844 | 16987 |
| 54 | .agents/prior-art/PRIOR_ART_HISTORICAL_EXPERIMENTS_V2.md | 16990 | 17153 |
| 55 | ../prior-art/PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md | 17156 | 17638 |
| 56 | .agents/prior-art/PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md | 17641 | 17930 |
| 57 | .agents/prior-art/PRIOR_ART_HISTORICAL_ROADMAP_V2.md | 17933 | 18341 |
| 58 | .agents/prior-art/PRIOR_ART_HISTORICAL_STAGE0_V2.md | 18344 | 18532 |
| 59 | .agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md | 18535 | 18681 |
| 60 | .agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md | 18684 | 18842 |
| 61 | .agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md | 18845 | 18991 |
| 62 | .agents/prior-art/RESEARCH_SOURCE_ATLAS.md | 18994 | 19126 |
| 63 | .agents/docs/RUNTIME_SERVING_AND_UX.md | 19129 | 19276 |
| 64 | .agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md | 19279 | 19484 |
| 65 | .agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md | 19487 | 19729 |
| 66 | .agents/docs/SECURITY_AND_RELIABILITY.md | 19732 | 19913 |
| 67 | .agents/docs/STATE_CACHE_AND_TRANSACTIONS.md | 19916 | 20152 |
| 68 | .agents/research/STRIX_HALO_OPTIMIZATION_DOSSIER.md | 20155 | 20317 |
| 69 | .agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md | 20320 | 20560 |
| 70 | the retired bootstrap handoff | 20563 | 20645 |
| 71 | .agents/plans/EXPERIMENT_PROGRAM.md | 20648 | 21024 |
| 72 | .agents/prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md | 21027 | 21138 |
| 73 | .agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md | 21141 | 21330 |
| 74 | .agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md | 21333 | 21496 |
| 75 | .agents/research/MULTIMODAL_PARALLEL_RESEARCH_WORKSTREAMS_2026-10-14.md | 21499 | 21547 |
| 76 | the retired P07-B mixed-quant handoff | 21550 | 21653 |
| 77 | the retired P07-C resident-GPU-weights handoff | 21656 | 21743 |
| 78 | the retired P07-D decode-once handoff | 21746 | 21841 |
| 79 | the retired P07-E async-readback handoff | 21844 | 21903 |
| 80 | the retired P07-F GPU-quarantine handoff | 21906 | 21969 |
| 81 | a retired phase-1 handoff | 21972 | 22065 |
| 82 | .agents/plans/QWEN35_8K_NUMERICAL_DRIFT_INVESTIGATION_2026-10-14.md | 22068 | 22147 |
| 83 | a retired Qwen CPU/GPU handoff | 22150 | 22261 |
| 84 | .agents/plans/QWEN35_CPU_LONG_CONTEXT_RESUME_2026-10-13.md | 22264 | 22370 |
| 85 | a retired Qwen numerics/context handoff | 22373 | 22535 |
| 86 | .agents/plans/QWEN35_EXACT_CPU_NUMERICS_LONG_CONTEXT_PHASE_2026-10-13.md | 22538 | 22779 |
| 87 | .agents/plans/QWEN35_PHASE1_8K_GATE_AND_P07_ACCELERATOR_HANDOFF_2026-10-14.md | 22782 | 22864 |
| 88 | .agents/plans/RELEASE_ACCEPTANCE_GATES.md | 22867 | 23075 |
| 89 | .agents/plans/THREE_STREAM_PROGRESS_2026-10-14.md | 23078 | 23187 |
| 90 | docs/PUBLICATION_AND_REVIEW_CHECKLIST.md | 23190 | 23284 |
| 91 | docs/README.md | 23287 | 23416 |
| 92 | docs/REFERENCES.md | 23419 | 23521 |
| 93 | the CPU reference probe README | 23524 | 23596 |
| 94 | the GPU probe crate README | 23599 | 23953 |

### Outer13 appendices and3 shared snapshots

These wrap the older94-source handoff; they are not another missing later
implementation archive. The end boundary is immediately before the next peer
section, so it may include separators.

| Section | Exact historical source title | Start line | End before next section |
| --- | --- | ---: | ---: |
| V5.1 source 1/13 | .agents/research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md | 73 | 239 |
| V5.1 source 2/13 | .agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md | 240 | 552 |
| V5.1 source 3/13 | .agents/docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md | 553 | 762 |
| V5.1 source 4/13 | .agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md | 763 | 1102 |
| V5.1 source 5/13 | .agents/research/MULTIMODAL_V5_1_RESEARCH_WORKSTREAMS_AND_AGENT_BRIEFS_2026-10-15.md | 1103 | 1245 |
| V5.1 source 6/13 | .agents/research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md | 1246 | 1413 |
| V5.1 source 7/13 | .agents/research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md | 1414 | 1627 |
| V5.1 source 8/13 | .agents/research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md | 1628 | 1833 |
| V5.1 source 9/13 | .agents/research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md | 1834 | 2050 |
| V5.1 source 10/13 | .agents/research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md | 2051 | 2250 |
| V5.1 source 11/13 | .agents/research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md | 2251 | 2457 |
| V5.1 source 12/13 | .agents/research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md | 2458 | 2623 |
| V5.1 source 13/13 | .agents/research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md | 2624 | 2890 |
| Live shared snapshot 1/3 | .agents/plans/DECISION_REGISTER.md | 2891 | 3160 |
| Live shared snapshot 2/3 | .agents/plans/IMPLEMENTATION_ROADMAP.md | 3161 | 3581 |
| Live shared snapshot 3/3 | .agents/plans/CURRENT_STATUS_AND_TRACKING.md | 3582 | 3930 |

### Additional nesting inspected for a second recovery oversight

- Source21 (10374–10864) preserves the earlier master handoff: P00, first source
  foundation, and V4.3→V4.2→V4.1→V4 design transitions. Older incomplete CPU
  statuses are superseded by later accepted results. It is recoverable text.
- Source23 is the then-agent entrypoint and25 the then-next-agent prompt. They
  survive as text; they do not authorize restarting G3/H5 or old evidence scripts.
- Sources37/38 are V3 snapshot versions. Source55 embeds V2/V2.1 plus the original
  V1 initial handoff, whose subsection begins at outerline17331. These design
  layers are not later trained-model/GPU tests.
- Source64 is the old workspace guide;70 P00 packaging;76–87 early implementation
  and numerics;93/94 old CPU/GPU-probe runbooks. Historical separate probe crates,
  locks and pre-layout paths do not override the accepted nine-package layout.
- All14 deleted archive/brief bodies were searched read-only for later H4,
  0cad60a/2125612/6d74f76 and explicit33-commit/129-source/191-source markers.
  No later H4-era implementation aggregate or exact original33 ledger was found.
  The pre-product README also retains earlier G3b-era status. The current decision
  register's later H4 summaries are different from full report-body recovery.

### Recovered historical phase labels and narrower test counts

The following labels are transcribed from executive chronology4063 onward. They
are **historical recorded labels**, not recovered objects or invented exact old
commit subjects. All nine new labels checked below returned128 to individual
`git cat-file -e LABEL^{commit}`; old602c246/4d1b82f/f28dc10 were already checked.

| Phase or baseline | Historical recorded label | Retained narrow meaning and checks |
| --- | --- | --- |
| P07-A |81a3e3c|Source Q4_0×dynamicQ8_0 seam; Unsupported-only fallback, fail-closed output; isolated physical projections are not full-model GPU|
| P07-B |2967edc|Mixed source quantization, host tensor/full-model completed-region simulation; no physical qualification yet|
| P07-C |836948b|Immutable allocation identity/residency; original .8B/2B3-token host tests one1/2MiB upload+2hits, complete logits/state equal|
| P07-D |e589be1|Decode fixed source blocks only on miss; DeviceEventLedger, host/event tests|
| P07-E |b283fb8|Real read_async completion bridge; actual hardware not yet verified in its own step|
| P07-F |602c246|Original physical handles retained on uncertainty, one static+transient cap;10 source-free+8 trained-GGUF host tests|
| P07-G1 |4d1b82f|4arbiter+4queue control-plane tests; no physical workers then|
| G2a starting baseline |f28dc10|Attested envelope;15GPU source-free+8GGUF host tests, not physical-model proof|
| G2b starting baseline |7e6e4f4|20GPU source-free+8GGUF host tests;4physicalNVIDIA completions/readbacks+2weight reuses; AMD F32 refuses root|
| G3a starting baseline |c3e5fca|WideF64 exact on both cards,4-token AMD hybrid; forced uncertainty5handles/256MiB fence, not driver recovery|
| Later review baseline |67dca72|Bounded adversarial source/geometry/queue/pre-submit/ownership fixes|
| G3b starting baseline |e9312ad|Shared Arc source/catalog and private session roots; measured source regions, not full resident/native CPP yet|

New individual lookup failures:81a3e3c,2967edc,836948b,e589be1,b283fb8,7e6e4f4,
c3e5fca,67dca72,e9312ad. These improve textual chronology without recovering the
old33 Git graph.

### Additional substantive early results and limitations

- P00 source70:4unit+3executable integration tests (7/7), native Linux release,
  strict clippy/fmt,4shell entrypoints,9Nix files parse/format and offline installer
  fresh/repeat/corruptSHA/wrongversion fixture. Early Nix evaluation was blocked by
  a read-only store-lock path; no six-target binaries/hosted CI/Windows/macOS
  physical validation/functional serving. Later reachable1af2ac4 fixes a different
  module-evaluation base-system issue; do not conflate the historical failures.
- Reference foundation source81:23passes (13library+5binary+5integration), source
  metadata/quant/config and failed three-domain transaction guards. No trained
  target weights/tokenizer/logits. Later original QwenCPU8K acceptance supersedes
  it; these first contracts do not establish27B/Gemma trained inference.
- CPU executive:8×32-token/256greedy short suites and146-vector trained comparisons
  permodel pass;2B48signed-zero distinctions in that earlier scope. Actual8192
  contexts bothgreedy7993/shared20top20 with logprob maxima.20372924/.19678696 and
  top-token probability errors.00215508/.0000536415. Functional limits are not
  exact source parity; GDN first .8Blayer4≈2.2538e-5/2Blayer1≈.01456 remained open.
  Later accepted512-source56zero signs is a different scope.
- G3b source31: shared vs old .8B3-token logits/state/root exact; two physical
  sessions share oneupload/onehit/two readbacks (AMD .8B/NVIDIA2B), independent
  roots/zero logical teardown charge. This is not later complete resident graph.
- G3b Q4 2048×1024 region measurements:2warmups+16readbacks, unoptimized debug build,
  H2D+kernel+readback included. NVIDIA NativeF32 median/p95 7115/13688µs vs WideF64
  7931/8897µs. AMD NativeF32 8499/10247µs vs WideF64 8473/11583µs. AMD F32 fails
  22896FP32 comparisons over18outputs/max2.3841858e-7, WideF64 exact. These are NOT
  modeltokens/sec, optimizedreleaseP95 or accepted NativeF32 policy. Device-wide
  VRAM/GTT counters remain elevated after zero logical charge: advisory pools/
  unrelated use, not exact model leak/free-space telemetry.
- G3b LLVM23 native blockers are historical. Later accepted CPP NVRTC/HIPRTC
  avoids the unused LLVM path without claiming independent LLVM device codegen.
  Historical AMD use does not lift the subsequent local hardware hold.

### Other assembled-handoff cross-checks

- Exact77 fullIDs+subjects match local `git log --all --reverse` in order. First46
  remain source-owner decomposition;23new acquisition commits+historicalA1 are
  not an invented original33 ledger.
- Acquisition full-gate counts, final934/69/205 and separate originalmodel17.60s/
  23.70s match retained canonical TODOs; no additive totals or combined timing.
- Five accepted backend profiles vs five target physical GPUs remain clear:
  four healthy cards, held local fifth. GPU512/functionalCPU8K/open independent
  8KGDN/16K/32K boundaries are distinct.
- All15 U056–U070 rows exactly match canonical register. Starting prompt keeps
  runtime PAUSED and requires explicit resume; old labels not current references.
- All assembled local Markdown file targets resolve. No other concrete chronology,
  count/timing/profile/pause blocker found in the scoped review.

Independent review required correction of the broad missing-aggregate/all70 claims,
including every inherited context paragraph. That correction and the recovery catalog are integrated in this handoff.


<a id="acquisition"></a>

## Model acquisition: complete current contracts and recent reasoning

The current detailed technical owner is
[model-acquisition/README.md](../plans/model-acquisition/README.md). Its live TODO
and two plain Markdown completed-checklist archives retain each implementation,
failed experiment, qualification command/result and pending extension. Acquisition
lives in `zerogravity-artifacts`; that crate also owns artifact parsing, immutable
source materialization, model/source metadata, quantization and the managed cache.
It is not an evidence-management crate.

### One source, transport, cache and preparation path

The optional Linux `cache` feature provides private managed content-addressed
blobs/manifests/snapshots, atomic refs, bounded locking/quota/admission, full-byte
verification and retained leases. The separate optional `acquisition` feature
adds one synchronous verified HTTPS owner, provider resolvers, exact plans,
streaming/progress/resume and explicit CLI. Default CPU/core/inspection builds
exclude HTTP/TLS/GPU dependencies. No Tokio service, Python loader/SDK, foreign
inference process, second cache or second model identity issuer is introduced.

The cache root is `${XDG_CACHE_HOME:-$HOME/.cache}/abird/zerogravity/models`;
relative XDG values are ignored and explicit absolute override wins. The initial
legacy root namespace was corrected before subsequent acquisition. Credentials
are separate from cache metadata. Paths and manifest/control bytes are bounded;
no absolute, traversal, duplicate, file/directory collision or symlink escape is
accepted. Root-relative retained descriptors protect traversal; pathname checks
alone are insufficient.

One original `ResourceArbiter` owns this process's transfer staging. Disk quota
serializes cooperating cache writers and includes true temporary/final/partial
peaks. Neither is a global host-RAM/free-space reservation. Stable lock inodes
must not be unlinked; a new inode would split authority. Immutable names/read-only
mode/open files do not freeze same-UID mutable backing. The actual source loader
hashes the bytes of an admitted private copy before the existing `SourceWeights`
issuer creates the process-local source identity. Manifest digest is durable
content identity, not that process identity or a runnable model.

Publication orders actual streaming/hash/EOF, file/directory synchronization,
verified dedup, immutable snapshot installation and atomic ref replacement.
Readers see the old complete snapshot or the new complete snapshot. Errors before
publication preserve the old ref; post-rename sync uncertainty has a distinct
`DurabilityUnknown` outcome. A file-complete progress callback is not publication.
A model source, device-ready object and session StateRoot are separate readiness
boundaries. Fault injection includes real process death, not only mocked errors.

Acquisition is explicit: run, inspect, catalog misses and unsupported model loads
do not download. Offline selects the local terminal branch before credentials,
client/DNS, admission or signals. It fully verifies actual retained bytes and
checks source/selection/current limits/pointers as required by the named mode.
Offline does not prove present remote authorization, license acceptance, alias
freshness, full remote membership or native Xet execution.

### Exact selection, closure and independent components

HF pins a branch/tag once to an exact commit, then uses one bounded complete
inventory for selected files, indices, HEAD recovery and sidecars. GitHub pins an
exact commit/tree or release/asset identity and refuses truncated trees. Git blob
object IDs are not raw SHA256. Ordinary `--file` retains its existing bounded
component-local byte `*`/`?` grammar, no slash crossing; `[]` is literal and `?`
consumes one byte. The newer Unicode/full-path filters remain an unwired draft.

Selected weights require complete model closure: config/tokenizer, complete
canonical ordered GGUF splits and safe safetensors index/shards. Indexed tensor
`total_size` is tensor payload, not shard file lengths including headers. Duplicate
keys, unsafe paths, mixed/ambiguous families and incomplete closures refuse.
Explicit components may supply separately authorized pinned config/tokenizer and
auxiliary snapshots. Each retains its own provider/repository/revision/manifest;
there is no invented atomic cross-source transaction, merged source identity or
composite ref. Same declared aliases reuse one pin; repeated release components
reuse one typed release pin without relisting each asset from mutable tags.

Git LFS pointer bytes are rejected as acquisition results, including correctly
hashed cached pointers. Generic cache inspect/verify still exposes literal stored
bytes; source eligibility is the stronger boundary. The legacy ordinary offline
API remains initial-selection/role-agnostic for compatibility. Component Weights
reuse strict closure with one cumulative metadata-byte/request budget across the
whole set. New strict quant/filter modes have their own explicit contracts.

### Transport, resume and bounded auxiliary bodies

Production HTTPS verifies TLS, identity encoding, framing, status and normalized
origin policy. No plaintext HTTP except controlled diagnostic literal-loopback
fixtures. Origin-scoped named credentials never appear in argv, plans, receipts,
errors or expiring signed locators. TRACE is compile-capped at DEBUG for the Rust
`log` facade because upstream ureq TRACE can expose query/Location secrets.
Foreign redirect origins lose credentials. Provider-specific opaque targets
preserve exact source/version/signature/condition authority in the same owner.
No arbitrary header/client/credential getter bypass is exposed.

Durable exact transfers retain proper-prefix checkpoints bound to immutable
source/path/length/strong validator, rehash the saved prefix, request/validate
Range and If-Range, and verify complete final hash/length/EOF. Changed validator,
status200 concatenation, invalid Content-Range and mismatched saved source refuse
or follow only the qualified owned restart path. Interrupted tails/older records
remain charged until a qualified owner reclaims them. Unknown-digest FirstSeen
transfers do not claim network single-flight. Typed cache hits verify actual data
before avoiding network work.

Exact lengths remain the default. Missing selected lengths use trusted pinned
LFS metadata or bounded body-free HEAD with exact representation rules. Chunked
bodies can transfer under an already authorized exact size. Explicit `AtMost`
streaming is Auxiliary-only: reserve the entire cap, distinguish unknown exact
size from the maximum, consume one terminal sentinel and publish actual observed
bytes. No required/model-closure cap, capped resume or independent full HTTP
trailer validation is claimed. Parsing/filesystem sync/blocked reads do not gain
real-time cancellation; original finite per-request deadlines remain binding.
SIGINT/SIGTERM registration is optional online CLI work, unregistered on return;
help/offline/cache/default CPU do not install it.

### Protected removal and explicit local HF import

Protected exact-snapshot removal excludes live readers through original managed
file/dup/inherited-child/lazy Artifact ownership. Graph ownership, locks,
ref/snapshot/blob sharing, actual retained handles, charge rebuild and process
faults are tested. Current removal does not imply automatic eviction, general
orphan/staging cleanup, legacy-root migration, arbitrary unmanaged-reader safety,
physical kernel/driver allocator release or global freed-space attestation.

`cache import-hf` is an explicit read-only external source capability: canonical
`models--owner--repo`, exact lowercase commit, explicit paths/roles, existing
absolute private destination and optional authorized raw hashes. Dry-run is local
metadata; apply creates independent verified owned copies through the original
transaction/publisher. It never adopts/deletes/chmods/renames/locks the external
cache or infers branch refs/config URLs. Open retained descriptors and rechecks
bind device/inode/size/mtime/ctime and exact EOF while copying; a separate owned
copy/load digest protects actual model execution from later mutation/removal.

Physical cross-filesystem qualification covers original same-mount plus distinct
Btrfs↔tmpfs and tmpfs↔tmpfs pairs, both directions, using existing filesystems.
Different mounts must have distinct devices and permitted retained-FD filesystem
types; selected files remain on their root mount/device. Same-device bind aliases,
Btrfs↔Btrfs, overlay/FUSE/network, hidden bind/subvolume fixtures and renamed layouts
remain unqualified. No mount or privilege changes were performed. tmpfs publication
acknowledgment is not persistence after unmount/reboot/power loss.

### Finite native provider profiles

| Surface | Qualified contract and explicit boundary |
| --- | --- |
| HF / GitHub | Exact pinned repository/release source, bounded inventory/closure/size recovery, named origin auth, original transfer/resume/progress/offline/cache. Git LFS object indirection, broader endpoints/env and complete new sidecar policy remain open. |
| HTTPS objects | Explicit origin/path/role/exact bytes/caller rawSHA, original one-entry managed verification; no implicit missing-digest or repository discovery. |
| OCI/Docker | Exact raw manifest digest and selected raw GGUF layer, tag pinning, public/preissued bearer and fixed Docker Hub anonymous broker. No image execution, compressed extraction, general index/platform/signature/private-login profile. |
| GCS | Native JSON generation/metageneration metadata/media, bounded auxiliary prefix/pageToken selections, public/preissued bearer. Independent object pins do not invent an atomic bucket snapshot. ADC/minting/requester-pays/custom endpoints remain open. |
| S3 | Explicit non-null versions, regional path/virtual-host addressing, exact initial/ranged representation, bounded V2/latest-version auxiliary prefixes, static/session SigV4. Additional current objects require caller SHA/exact-size contract or explicit strong-conditional observation. Broader partitions/endpoints/credential chains/FULL_OBJECT checksum authority remain open. |
| Azure | Explicit BlockBlob versions/snapshots and bounded current-version auxiliary listing, anonymous/bearer/SharedKey/ad-hoc serviceSAS, exact ranges. Current caller-SHA and conditional profiles are distinct. Latest/SAS/endpoint/default-managed-chain extensions remain open. |
| ModelScope | Explicit canonical40hex commit, bounded complete shallow walk and exact model closure; anonymous/official-token Bearer+session-cookie profile, digest-only offline. A direct directory hitting the native3000-row truncation boundary refuses. Alias echoes/timestamps/file-last-commit are not an alias→commit proof. |
| Native HF Xet | Exact HF pin/closure, header-only native announcement, scoped ephemeral CAS grant, exact signed range sets/multipart, ordered reconstruction/codecs/full root plus independent rawSHA/EOF through the original cache. At least one native file required; no compatibility substitution on failure. Native index reconstruction remains draft. |

Cloud caller-SHA contracts bind normalized locator/profile/path/role/size plus raw
content digest without inventing remote versioning. Conditional S3/Azure profiles
instead HEAD an exact strong ETag/length and send typed If-Match on every initial,
range and retry request. They remain FirstSeen rather than TrustedSha256. Range
validator contradictions refuse before signing/wire; HTTP412 never refreshes HEAD
or downgrades. SharedKey/SigV4 exact conditional signatures have independent literal
HMAC vectors. ETag/checksum labels are not automatically full raw content hashes.

Selected missing ModelScope file/index sizes use one same-full-commit body-free
exact200 no-follow HEAD through the existing cumulative metadata budget. Exact
identity Content-Length is required; no redirect/405/501/cap/GET fallback. Cache
index length once before its GET; unselected unknown lengths do not probe.
Malformed present/null blob sizes refuse, directory-null legacy behavior stays.
Strong ETag binds index/payload before bytes; weak/absent validators do not grant
resume authority. A later file-only CDN profile needs finite explicit origins,
exact bytes and authorized rawSHA, credential stripping on every delegated hop
including bounce-back, ephemeral redacted locators and same-pin/hash/size/range
refresh. Neither CDN nor alias/mirror authority has been qualified.

### Native Xet details that must survive continuation

The accepted protocol1.1 primitives use keyed BLAKE3 chunk/Merkle/file namespaces,
textual limb order and empty-root rules; rawSHA256 is separate content authority.
The allocation-free decoder bounds encoded chunks256KiB and decoded128KiB, checks
8-byte framing/exact consumption and supports NONE, framed LZ4 and BG4/LZ4 with
all tail remainders. Dictionary/legacy/skippable/concatenated frames refuse. No
absent per-term hash or partial-xorb root is invented as protocol authentication.

The native reader preserves ordered repeated/overlapping terms, exact file
coverage and complete signed interval sets, including arbitrary multipart order.
Only accepted logical chunks enter the file root/progress; edge/refetched work is
still decoded/charged. Replayed leaves must match prior accepted length/hash.
CAS v2 falls back to v1 only on404/501; scoped401/signed403 refresh is finite and
cannot change file identity or ordered semantic terms. Hub credentials never
reach CAS/signed payloads, even at the same origin, and grants never become
persisted model/source identity. Native targets refuse redirects.

One invocation bounds actual HTTP attempts, observed body/control wire bytes,
metadata, decoded chunks, logical leaves, refreshes and cumulative accepted
terms/groups/ranges across files/reissues. An observed overflow sentinel poisons
further work; exact-limit zero-byte EOF probes remain possible. Finite counters
and per-request deadlines are not physical TLS/kernel telemetry or one global
elapsed deadline. Native scratch is admitted before allocation and retained
alongside the cache transaction; native CLI defaults256MiB versus ordinary128MiB,
with explicit smaller limits still binding. Native object staging is ephemeral:
no HTTP-prefix checkpoint/resume; process exit starts a new full native transfer.
Ordinary sidecars retain their original durable resume/credential-redirect behavior.

Current dry-run is Hub metadata/announcement only, no CAS grant/signed body.
Offline verifies retained HF bytes but cannot attest native execution. The new
required-index draft needs complete counter continuation from pre-plan decoding
into execution: the current accepted seed carries only attempts/wire/metadata,
so simply enabling index decode without extending it would reset work limits.
The paused draft addresses this but is uncompiled/unqualified. Future measured
coalescing/chunk-cache/native persistent prefixes remain open; the current reader
may refetch complete signed sets and has no throughput superiority claim.

### Latest candidate and quant features

Candidate discovery is metadata-only through the same one pinned regular-file
inventory, with no family choice, HEAD, index/payload read or cache publication.
Complete inventory validation precedes subfolder projection, including regular
ancestor collisions. Returned scope is pinned-remote-tree or verified-cached-subset;
unknown lengths stay unknown and sizes are observations, not payload reservations.
JSON distinguishes stored Git blob sizes, HF tree/LFS metadata and verified retained
content. Git object IDs and cached FirstSeen digests never become expected rawSHA.
Cached discovery checks original source/current bounds/full hashes/pointers before
credentials/client/admission/signals. It cannot prove remote-global model choices.

Quant selection is an additive private-field `GgufQuantRequest` over that source,
not a change to public `Selection` or `PullRequest`. Tag grammar is1..32ASCII bytes,
letter-led then letters/digits/underscore, normalized uppercase. Exact lowercase
`.gguf`, optional strict five-digit split trailer and terminal basename `TAG`,
`-TAG` or `.TAG` match; underscores remain inside tags. Full path case/family
identity stays exact. Directory tag text never supplies a basename tag or split.
Every matching cohort must be complete before exactly-one-family selection;
malformed/missing/conflicting totals, case variants, singles+splits and multiple
model prefixes refuse. No default quant, numeric quality rank or first-file fallback.
Existing explicit files remain available for other filename conventions.

Selection happens inside the original provider tree before existing closure,
sizing and source construction. Shared logical-call bounds cover pin/tree/HEAD/
index/native discovery; original HTTP independently counts physical work. The CLI
uses one shared fetch/progress/cancellation/publisher path. `--file`, release assets
and `--quant` conflict explicitly. New quant offline requires snapshot digest and
source/current limits/pointers/selected closure plus Weights/Config/Tokenizer roles;
other retained data remains visible as the existing snapshot. Ordinary Files
retains its legacy offline semantics. The new native quant focused fixture proves
inventory/announcement/plan selection, not quant-specific CAS execution/inference.

### Failed experiments and lessons

The completed TODOs retain each failure and exact repair. Particularly important:

- Cache hits must read/hash actual bytes, not trust hash filenames, validators or
  a dedup label; cancellation and callback publication boundaries are independent.
- Legacy offline five intended failing assertions proved pointers/incomplete
  closure/reset budgets before the policy repair; other precompile static findings
  are not retroactively called executed RED tests.
- A native test pipeline with `tee` initially hid Cargo101 behind exit0; capture
  the actual command status. Synthetic `payload.bin` correctly triggered model
  closure, so auxiliary fixtures were renamed and a separate complete weights
  case added. Original sanitized State errors were not weakened to satisfy a
  raw Limit expectation; help do/does grammar was an assertion-only repair.
- Independent long-Xet oracle generation first copied65/66hex XOF prefixes;
  correct official32-byte prefixes and validate six official vectors before
  freezing custom128KiB/tree goldens. Never fit expected Rust results.
- Candidate review found a regular parent hidden by subfolder projection;
  full-inventory validation fixed it before first compile. Quant review found
  `-of-` in a directory falsely interpreted as a filename trailer; basename-only
  checks preserve full family identity. Both are static findings, not executed RED.
- ModelScope's new suite initially ran against an old normal rlib:1pass/9fail,
  with the removed error still inside the executable but absent from live source.
  `copy2` had preserved source mtimes older than Cargo outputs. Byte-equal narrow
  timestamp refresh on exactly five merged paths made the identical command pass.
  Future merges use fresh mtimes; do not relax math/protocol or clear all caches.
- The paused filters reference generator initially failed on Python3.14 strict-end
  regex spelling; only synonymous `\\z`→`\\Z` adaptation on installed3.13 was fixed
  before expected values were frozen. New Rust has never executed. The portable
  oracle copy is inert text because the original generator fetches source and
  overwrites an absolute SIDE path; do not run it as a continuation shortcut.



<a id="continuation"></a>

## Exact continuation and complete open-plan snapshot

Author: GPT-6 Codex (continuation audit).

This is the documentation-only continuation context, inspected on 2026-10-17. It is not authorization to resume and does not replace the live [plan index](../plans/TODO.md), [TODO rules](../plans/TODO-SPEC.md) or [AGENTS.md](../../AGENTS.md). The user explicitly requested wrapping the current phase and **PAUSING**. C2 is accepted; C3 and C4 remain unfinished source drafts. This later handoff request authorizes documentation only. Resume implementation only when the user asks.

### Starting point and verified scope

MAIN is `/home/pvl/spaces/abird/src/zerogravity`, branch `main`, inspected HEAD `f197c9cf4a44e06a3c1cf3c532855545bed945fa` (`feat(pull): select explicit GGUF quant families`). Subsequent coordinator documentation-only commits may advance HEAD: reread current files and status instead of resetting to this reference. At inspection the only MAIN tracked changes were the coordinator's rolling `.agents/plans/TODO.md` and model-acquisition TODO; own/other documentation runs were untracked. Preserve them. Do not treat an unclean status as permission to discard another writer's work.

Latest relevant accepted commits:

| Reference | Qualified slice |
| --- | --- |
| `6d74f76` (historical approval label; object unavailable) | Accepted original engine/source/state/resource baseline; remaining full-platform goals stayed open. |
| `837e77f` | Legacy offline managed-pointer checks and strict component model closure. |
| `52e42fc` | Native HF Xet reconstruction under the original HTTP/executor/cache owners. |
| `46cf5a5` | Strong conditional S3/Azure observations, FirstSeen raw hashing, original signing/range/publisher. |
| `4db5efb` | Selected ModelScope missing exact lengths via pinned HEAD. |
| `f22f759` | Metadata-only pinned candidate inventories, honest cache-subset coverage. |
| `f197c9c` | Explicit GGUF filename quant family selection, no silent fallback or tensor eligibility. |

These entries identify accepted records;6d74f76 is a retained approval label, not a currently resolvable object. The later implementation IDs are reachable. There is no new requirement to maintain SHA inventories, generated receipts or an evidence archive. Actual checkpoint/raw-content hashes remain implementation inputs where required.

The [acquisition TODO](../plans/model-acquisition/TODO.md) records C2 final975 focused executions/0 failures/16 intentional helper ignores and all4 strict artifact/CLI production/diagnostic configurations. Its frozen isolated full gate passed934 Rust executions/0 failures/69 ignored/205 suites plus18 namespace fixtures and the normal formatting/dependency/lint/build/release/installer/comparator/resume checks. Two separate exact release cache_prepare tests passed17.60s and23.70s, preserving both approved original Qwen3.5 models/all248320 logits/all42 state domains/two-token/session/source-lease/mutation/reclamation criteria. These are individual test durations, not a combined invocation. Actual Nix source-integrity,26 local links, protected root README and unchanged Cargo manifests/lock passed. This documentation audit ran none of those commands again.

Candidate coverage and quant/native scope remain deliberately narrow: candidate listing transfers no payload, HEAD or index bodies; quant chooses filenames, not tensor codecs or inference; new native-quant tests qualify shared Hub inventory/announcement/plan generation, not a new quant-specific CAS or inference gate. The original trained executable source boundary remains approved Qwen3.5 .8B/2B GGUF, current4GiB executable-source cap and the numerical/device scopes in their plans. Acquired safetensors/new model configs do not create runnable models.

### Non-negotiable working rules and existing authorization

- Read AGENTS, TODO-SPEC, index, relevant plan TODO/README and linked specifications before changes. Historical handoffs/roadmaps/status documents are design/history, never a second queue or proof of current support.
- One Rust mathematical engine; immutable original source identity with one checked issuer; independent StateRoots; one physical ResourceArbiter, fair workers and actual completion/quarantine. Do not add a downloader/runtime/cache/allocator/model engine merely to implement a provider or selection option.
- Keep the root `README.md` byte-exact: user-authored and outside agent edits. Acquisition design belongs in its plan README; root coordinates canonical docs. Preserve unrelated source/index/worktrees and each agent's own run file.
- Agents use only their assigned paths. Agree cold shared seams first; independent source/review/test work uses subagents, with one coordinator deciding/integrating. Serialize shared Cargo target ownership explicitly; no simultaneous Cargo/rustc/clippy or surprise SIDE builds. At this audit exact process-name inspection observed no cargo/rustc/clippy-driver process; recheck before a future grant. A prior grant is not a permanent target lock.
- Source/provider work authorized during the continuous task did not authorize arbitrary models, SDK downloads, private/gated license acceptance, external payload/account use, host mounts, driver/privilege changes, device reset/recovery or live cache mutations. Synthetic owned fixtures and already-approved original inputs were the qualification scope. Additional physical/network inputs need their applicable explicit authority; do not invent extra permission gates for ordinary reversible implementation already authorized after resume.
- The local Renoir/gfx90c AMD hold remains. Its aborted initialization/wedged recovery is not Drop-retirement/free-memory proof. No launches/resets/recovery on it without separate applicable authorization.
- No selected-GPU error→CPU fallback, numeric tolerance relaxation, implicit lower quant or hidden source change. Internal bit parity, independent oracle comparisons, model quality and throughput are separate acceptance dimensions.
- Offline must branch before credential env/file/FIFO/stdin/client/DNS/provider/admission/signal construction as each new profile specifies. `run`, `inspect`, cache/catalog misses never implicitly fetch. Credentials/tokens/signed URLs remain ephemeral, redacted and origin/profile-scoped; presets/config/model cards grant no transitive source or execution authority.
- Signed commits only; never disable signing. Root is current Git/dependency/lock/canonical/gate integrator. No push/force-push/tag/publish/deploy/destructive history/branch/worktree/cache operation without explicit authorization for that action. A source grant or request to commit is not permission to push.
- Main plan updates are one serialized coordinator write, integrating completed and unfinished results. Commit verified code/tests/plan together with explicit paths, record the actual commit in the next rolling update, retain unfinished work. No commit merely to record its own ID. Keep interrupted runs until recovered; remove finished runs only after useful contents are integrated.

#### Current acquisition environment and ownership boundaries

The explicit cache override is an absolute --cache path. Otherwise managed artifacts use `${XDG_CACHE_HOME:-$HOME/.cache}/abird/zerogravity/models`, ignoring relative/empty XDG overrides as implemented. This is the managed destination, never authority to inspect/delete an ambient external HF cache. HF import requires explicit repository-cache root/repository/canonical commit/selected files and an already-existing private destination; it copies verified bytes without modifying/locking/chmodding/hardlinking external data.

Legacy HF/GH credentials come only through HF_TOKEN/GITHUB_TOKEN and original normalized origins; GitHub explicitly scopes the accepted raw-content origin. Other provider handlers use caller-named environment variables or explicitly supplied bounded credential files/input only for their declared profiles. No ambient ADC/AWS/Azure/SDK metadata chain or automatic broker/login/refresh is implied. Never print values, dump the environment or paste signed URLs. Offline branches happen before those inputs; absence of online credentials does not invalidate a verified retained snapshot or attest current access. Native Xet cannot override Hub/CAS origins in CLI; transient CAS grants and signed range URLs never enter plans/manifests.

Ordinary CLI transfer memory default is128MiB; explicit native Xet default256MiB only when the caller did not specify --memory-limit-bytes. Library callers retain their own registered TransferAdmission. The process ResourceArbiter covers declared transfer/control/source allocations, not all host RAM/kernel/TLS memory or other processes. Blocking HTTP cancellation follows existing per-request deadlines/finite attempts, with no single whole-Xet-operation elapsed-time promise. Library cancellation stays an AtomicBool; CLI scoped SIGINT/SIGTERM flags feed that same owner. No new socket/async runtime is needed.

Cache completion/publication authority remains the original transaction/CommitOutcome. Exact sizes/trusted SHA and FirstSeen observations stay distinct; optional exact-size getters must not return cap maxima as known totals. Capped unknown streams are Auxiliary-only under explicit cap, ephemeral/no resume, true terminal-body/cap+1 checked; required model/index closure remains exact. Strong ETag conditions are observations, not immutable cloud versions or trusted rawSHA. HTTP decoder zero-chunk-line semantics and lack of independent full trailer validation remain documented, not silently strengthened by tests.

Components retain per-source immutable snapshots and verified leases; final component-set readiness is distinct from individual publication and is not a cross-snapshot atomic mutable reference. Protected v2 exact-snapshot reclamation only measures rebuilt managed namespace charge; it does not prove OS final-close/physical free bytes. Current refs/other manifests/shared blob reachability, actual managed descriptors+clones, activity/object/publication locks and retained stages remain authoritative. Legacy v1 migration/general orphan sweep/automatic eviction/unmanaged concurrent raw readers are NOT covered.

### Preserve the two unfinished worktrees

Only these two SIDE worktrees were listed at inspection. Completed quant/native/provider worktrees were already retired by the coordinator after source-preservation checks; do not reconstruct them as onboarding locations.

| Worktree | Actual detached base / overlay | Draft status |
| --- | --- | --- |
| `/home/pvl/spaces/abird/src/zerogravity-filters-work` | `4db5efbcce973d21bf1c8ca750471e6b2ab956ab` plus accepted C1/C2 source overlays | C3: only new `acquisition/filters.rs` differs from MAIN; unwired, not compile-ready. |
| `/home/pvl/spaces/abird/src/zerogravity-xet-index-work` | `f22f75972e63ea70771c59fbce065e371d5f29d4` plus accepted frozen C2 overlay | C4: only HF and xet/{mod,budget}.rs differ from MAIN; uncompiled, no integration/CLI tests. |

SIDE plan/rule copies may be older than MAIN: use current MAIN AGENTS/TODO-SPEC/canonical plans, then the SIDE own pause run and actual delta. Do not let a stale SIDE TODO or old source-grant wording override the current user pause.

Read-only continuation audit compared actual bytes: all15 C3 baseline Rust paths equal current MAIN; only new filters.rs has no MAIN counterpart. C4 has9 dirty/untracked overlay Rust paths equal MAIN and exactly3 declared production differences. Dirty/untracked status relative to an older detached base therefore includes accepted overlays, not just new work. Never copy the whole SIDE over MAIN or stage every dirty path. Inspect actual deltas against the accepted MAIN source and preserve newer fixes.

The15 C3 accepted overlay paths are artifacts acquisition/{components,github,huggingface,mod,selection,xet/mod,candidates,quant}.rs; artifact tests/acquisition_{candidates,quant}.rs; CLI src/{pull,pull_xet,pull_candidates}.rs; CLI tests/pull_{candidates,quant}.rs. C4's9 exact overlay paths are candidates/github/mod/selection/quant, artifact acquisition_quant test and CLI pull/pull_xet/pull_quant test. Its three new deltas are listed below.

Historical agent names: `transfer_resume` owned C3 matcher/integration proposal and independent reviews; `review_h5` owned C1/C2/CLI and C4 draft; `durable_cache` owned cache/provider seams and acted as current exclusive validator; root alone integrated/gated/committed. Names are context, not durable running processes. Reassign explicit ownership on a fresh session.

#### C3 filters — exact recoverable state

Authoritative paused run: [SIDE filters TODO](handoff-2026-10-17/PAUSED_FILTERS_CONTEXT.md). Main copy/design: [filters run](handoff-2026-10-17/PAUSED_FILTERS_CONTEXT.md). The SIDE pause section is newer/more complete than early cold-design checkboxes: source is already granted, but execution is paused. Do not ask again for those settled design choices after an actual user resume.

Only Rust delta: [unwired filters.rs](handoff-2026-10-17/paused-filters.patch). `acquisition/mod.rs` does NOT declare it. It contains the bounded matcher/API scaffold and inline unit expectations, but references NOT-YET-ADDED `quant::{validate_inventory,select_excluding,metadata_admission_extra}`, filtered provider entrypoints and `selection::close_model_selection_filtered`. It cannot be accepted/merged as a working feature. No integration test files were created. Scoped rustfmt/whitespace passed only; no Cargo/rustc/Rust test/lint or product matcher was executed. No child/session was retained when paused.

Frozen API/source grant:

```rust
FilterRequest::new(CandidateRequest, Vec<String> includes, Vec<String> excludes) -> Result<Self>
FilterRequest::for_quant(GgufQuantRequest, Vec<String> excludes) -> Result<Self>
// private fields; source/includes/excludes/quant_request read-only getters
filters::resolve(transport, &FilterRequest, &PullLimits, &TransferAdmission, &AtomicBool)
    -> Result<ResolvedPullPlan>
filters::validate_cached_selection(&SnapshotLease, &FilterRequest, &PullLimits, &AtomicBool)
    -> Result<()>
xet::resolve_filtered(transport, &FilterRequest, &PullLimits, &XetOptions,
                     &TransferAdmission, &AtomicBool) -> Result<ResolvedPullPlan>
```

Approved11 paths, all SIDE-only until integration: artifacts acquisition/{filters.rs NEW,quant.rs,selection.rs,huggingface.rs,github.rs,mod.rs,xet/mod.rs}; new artifacts/tests/acquisition_filters.rs; CLI src/{pull.rs,pull_xet.rs}; new CLI/tests/pull_filters.rs. `candidates.rs` needs a concrete agreed necessity before edits; none was planned. Existing private C2 names are InitialChoice and FetchRequest. Preserve public Selection/PullRequest/PullLimits shape and old byte/component `--file` grammar.

Settled semantics:

- Additive full-path Unicode-scalar case-exact FnmatchCaseV1: `*` spans slash; `?` consumes one scalar including slash; `**` has no extra grammar. No locale/case/Unicode normalization. Brackets/ranges/leading !/initial ]/literal ^/hyphens/descending ranges/literal malformed '[' follow fixed primary CPython grammar; backslash is literal. Inputs exclude control/NUL and lone-surrogate/bytes modes. Subfolder is a literal component-boundary scope; patterns match the relative suffix, so brackets in the folder never become pattern syntax.
- Include union → exclude union → family ambiguity → original required/optional model closure. Unmatched includes need not each match; empty final result refuses. Required config/tokenizer/shard/index exclusion refuses; optional README/license/tokenizer_config exclusions remain omitted. Removing an entire competing family may disambiguate; removing only a still-required split member fails. No silent numeric/source fallback.
- Explicit quant+excludes is supported composition; quant+includes and old file/release/asset+new filters refuse initially. Excludes alone is invalid except with quant base. Do not give unrelated providers/components/cloud modes implicit new grammar.
- Excluded adjacent index refuses BEFORE GET only when original closure would attempt to read that index. Do not blanket-refuse an unrelated index or GGUF selection. Retain separate HF and shared GH/ModelScope closure owners rather than unifying their historical .bin/source policies as a side effect.
- One pin/tree/remaining body/request/probe/admission owner. No discover_candidates→resolve repin/relist. Native thin route preserves original grants/raw SHA/root/EOF/actual budgets, >=1 announced native file and current required-native-index refusal until C4 qualifies.
- New offline mode is digest-only, current source/bounds/full managed bytes/pointer/roles/complete closure and EXACT retained closed snapshot subset: extras/excluded retained data must not be printed as if removed. It does not delete bytes/create a filtered snapshot or prove remote uniqueness/freshness/access/native execution.
- Pattern count<=4096; each nonempty<=4096 UTF8 bytes; aggregate pattern+inventory path bytes<=8MiB. Fixed cumulative16Mi compile/state/scalar/range work, no reset per pattern/file/closure; cancellation at most4096 transitions apart. One pattern compiled at a time, two NFA rows, no recursion/regex/global cache. <=512KiB scratch and extra16MiB SAME resolve-only admission were approved SUBJECT TO proving the actual retained representation. Do not reduce existing buffers or claim this is all host RAM. Tiny caller memory must refuse before HTTP; ordinary128/native256MiB defaults stay unchanged.

Independent expected values are frozen in the new file BEFORE the Rust matcher:73 literals plus585 patterns×57 names=33345 cases; literal SHA256 `c9f3b0c631f703ae44a4f2dafb5c8c71cb2715bfaa3249751cd299c73a40445f`. Generator is own MAIN [oracle.py](handoff-2026-10-17/paused-filters-oracle.py.txt), a disposable research helper. It executed pinned CPython v3.14.0 fnmatch source on installed3.13 with ONLY strict-end spelling `\z`→synonymous `\Z`. First attempt mistakenly searched two backslashes and failed before freezing outputs; corrected oracle then exited0. Never regenerate expectations from or fit them to Rust. No runtime Python dependency. Keep class edge behavior independently reviewed; all new Rust expectations remain UNRUN.

Resume tasks: inspect this draft first; implement missing private seams/default allow-all policy; prove scratch/collection/work/cancel bounds; wire both providers/closure and native path; add CLI grammar/shared FetchRequest path; author API/CLI/default-denial/offline/old-ref/native/legacy regression fixtures; independent review; only then exclusive compile/test/lint/gates. Adversarial classes/missing brackets/many patterns and an actual default admission test are still absent. No new public runtime or allocator is needed.

#### C4 required native Xet indexes — exact recoverable state

Read [SIDE implementation TODO](handoff-2026-10-17/PAUSED_XET_INDEX_CONTEXT.md) and [SIDE design TODO](handoff-2026-10-17/XET_INDEX_DESIGN.md). Design's old statement that it lives in retired quant SIDE and its recommendation for another resolver name are historical: the implementation pause section and coordinator-approved additive builder supersede them.

Three actual unfinished production paths only:

- `crates/zerogravity-artifacts/src/acquisition/huggingface.rs`: private pinned-index context/callback, after original exact length and before original parser.
- `crates/zerogravity-artifacts/src/acquisition/xet/mod.rs`: private index resolver reusing original discovery/Access/Invocation/XetReader/scratch; private defaultfalse XetOptions policy plus public additive `with_index_reconstruction(self)->Self`, same resolve/resolve_quant signatures.
- `crates/zerogravity-artifacts/src/acquisition/xet/budget.rs`: ALL-counter Result seed, poisoned/current-limit refusal and2 private unit cases.

All three are UNCOMPILED/UNRUN. Source-only syntax formatting passed. No new `acquisition_xet_indexes.rs`, no new CLI apply/helper/help edits and no actual index transfer/fault/rollback tests exist. Existing CLI still has no newly enabled index feature. Do not advertise native-index support or merge these three onto C3 blindly.

Approved behavior: default library options and metadata-only dry-run perform no CAS grant/signed body; required native index returns Unsupported before that activity. Only explicit builder opts bounded required-index reconstruction in, and CLI APPLY will use an opted-in options clone with no new flag/function proliferation. Ordinary non-LFS indexes keep existing metadata route. LFS index requires pinned trusted raw SHA independently of the complete Xet file root. Reconstruct full exact index plus terminal sentinel/root/EOF/raw SHA/pointer checks BEFORE parsing; exact `take(n)` without terminal read is insufficient. One pinned tree/selected-or-adjacent policy remains. Index length<=remaining metadata/maxfile/parser/addressable cap before allocations. No arbitrary URL callback or guessed HTTP response/source authority.

Use one existing Scratch sequentially for planning indexes and retained outer metadata control; no new allocator/body cache/preplan transaction. Keep path→XetSpec only to avoid duplicate discovery. Publication reconstructs the small index again through the original reader/installer and charges it again; executor's verified staged index reparse stays authoritative. Default simultaneous planning envelope is outer80MiB +scratch90.5625MiB =170.5625MiB before any later C3 extra control; native256 fits and128 must deny before signed body. Recompute checked formulas after composition, never infer all-RAM telemetry.

ALL cumulative counters must survive resolution→execution: actual attempts/wire/control, decoded chunks, logical leaves, refreshes and accepted terms/groups/ranges. Snapshot seed returns Result; poison/overflow/current tighter bounds cannot silently zero/reset. Final plan contains no grants/tokens/signed URLs. C4's unit cases alone cannot qualify this: add multiple individually-fit indexes+later bodies exhausting aggregate limits, actual owner index/raw SHA/root/EOF/budget/refusal/cancel/token isolation/default admission/dryrun/oldref fixtures and original regression cohorts. Native reader/hash/codec/multipart algorithms remain the existing owners.

C3 and C4 overlap HF/xet/mod and shared CLI. Root must choose serialized reconciliation order (natural continuation is qualify C3, then reconcile/qualify C4) and freeze exact seams before either shared merge. Preserve C3 excluded-index veto before invoking C4's reconstruction callback. Existing C2 source is accepted at f197c9c; never overwrite it with the older detached baseline.

### Commands, environment and failure lessons

These are future commands after user resume and coordinator target grant, not commands run by this documentation task. Use current checked-in scripts as the command owner.

Canonical complete gate:

```sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh
```

`native-gpu.py` always creates private user+network namespaces (`unshare --user --map-root-user --net`) and sets Cargo offline. Failure to establish isolation aborts. It does not install/download/copy/patch SDKs. LLVM verification is opt-in `--require-llvm` for an already available pinned bundle, never an installation authorization. Do not repurpose HOME/CODEX_HOME or invent a target resource key.

For focused commands use the SAME jobs2/Rayon4/native wrapper and `--locked --offline`; explicit packages/features, never workspace `--all-features` (that selects GPU diagnostic closures). Examples to adapt only after source/test names exist:

```sh
cargo test --locked --offline -p zerogravity-artifacts --no-default-features \
  --features acquisition,diagnostic-acquisition,diagnostic-cache \
  --lib --test acquisition_filters
cargo clippy --locked --offline -p zerogravity-artifacts --no-default-features \
  --features acquisition --all-targets -- -D warnings
cargo clippy --locked --offline -p zerogravity-artifacts --no-default-features \
  --features acquisition,diagnostic-acquisition,diagnostic-cache --all-targets -- -D warnings
cargo test --locked --offline -p zerogravity-cli --no-default-features \
  --features acquisition --test pull_filters
```

Commands above show inner Cargo selections; run inside the isolated shell. Actual socket fixtures additionally require enabling ONLY that private namespace's loopback via the exact `PYLOOPBACK` block in [scripts/check.sh](../../scripts/check.sh). It verifies single non-host UID mapping and network namespace owner before touching lo. Use that block inside the same wrapper/session before focused socket commands. Never enable host interfaces/routes, run a privileged bypass or drop isolation to make a fixture pass. Production URL policy remains HTTPS; literal HTTP loopback tests using ResolvedFile require diagnostic-acquisition, not merely cfg(test).

Both original model routes, separately under the same wrapper/default jobs2/Rayon4:

```sh
cargo test --release --locked --offline -p zerogravity --features cache \
  --test cache_prepare approved_cached_models_match_direct_source_exactly \
  -- --ignored --exact --nocapture
cargo test --release --locked --offline -p zerogravity --features cache \
  --test cache_prepare approved_hf_import_models_match_direct_source_exactly \
  -- --ignored --exact --nocapture
```

They read already-approved `target/models/qwen35-0.8b` and `qwen35-2b`, create private synthetic destination/source copies and retain existing exact identities. Do not fetch replacements if missing. Approved raw GGUF SHA256 values are in the [accepted baseline](../plans/accepted-baseline/TODO.md); the tests also bind config/tokenizer hashes. Model/SDK directories are NEVER disposable Cargo cleanup targets. Additional GPU/source/model qualification requires its distinct current scope and authorization.

After root includes new reviewed paths in Git's actual Nix source, run the actual offline source check (typical current target `nix build --offline --no-link .#checks.x86_64-linux.source-integrity`). Check current flake/system interface before use; no broad Nix fetch or cross-platform SDK acquisition is implicit. [nix/source.nix](../../nix/source.nix) and [source-integrity](../../nix/checks.nix) compare original versus actual filtered vendor/model/source/TLS/Azure fixture bytes, one root Cargo.lock and unchanged root README. An untracked new Rust file omitted from the flake is NOT tested by a passing old derivation. Root owns explicit add-N/staging and source-integrity integration; test agents do not mutate the index.

**Copy bytes with fresh modification times.** Do not use cp -p/cp -a/rsync -a to merge old SIDE timestamps over a freshly built MAIN. ModelScope sizing initially produced a stale-library Cargo101 despite byte-correct copied source; narrow affected timestamp refresh resolved it without source/policy/expectation repair. After verifying exact intended bytes and ownership, refresh only affected source mtimes or use a correctly scoped fresh build target, then repeat the same affected checks. Do not hide a genuine compiler failure with unconditional cleans or change test expectations. Moving a checkout can separately leave absolute CARGO_BIN_EXE paths embedded in tests; [development guide](../../docs/DEVELOPMENT.md) explains rebuilding affected outputs. No blanket deletion of target/models, SDKs or another owner's target.

Use `bash` with `set -o pipefail` when teeing logs. Native Xet's initial compiler error was hidden by outer tee exit0; the correct record is compilation failed before tests (E0624), not a captured Cargo101 unless the pipeline actually captured it. Record actual commands/results, failures and repairs; no RED/GREEN claim for static review, formatting or authored tests. Keep literal independent expected vectors fixed. Once appropriate checks pass, do not repeat/broaden them without new changes/failures/concerns; proceed to coordinator acceptance.

### Broader backlog, execution order and concrete external gaps

All9 active plans remain unchecked. The full exact unchecked inventory is appended below so no deferred item is dropped; the live originals remain authoritative. The three completed plans (accepted baseline, handoff cleanup, historical records) have no open checkbox and must not be reopened as new implementation duties.

After an actual user resume, finish acquisition convenience C3/C4 with current provider-frontier rows, then continue finite native provider/auth/layout/preset/sidecar gaps as explicitly qualified slices. Planning/disposition or Unsupported does not close parity. Do not invent57 missing row IDs: historical captures were retired and the live README table is a finite class frontier, not an exhaustive provider proof. Registered tensor/serialization/IPC/plugin/random/dummy loaders belong to loader/runtime work unless an inspected missing byte protocol is identified. No generic plugin/Python engine/SDK delegation is required to claim a concrete native protocol.

CPU-first model addition is next independent major priority: original/config/index/tokenizer/checkpoint binding, admitted immutable complete owned shards and real Qwen3.8 27B plus Gemma4 E2B/26B numerical/state/tokenizer execution. Current large BF16 checkpoints exceed available local RAM/current4GiB executable-source API. A capable admitted host and explicit authorized inputs are real gaps; do not download arbitrary sources, relax source length or call metadata contracts inference. Preserve original Qwen3.5 gates. New GPU execution follows real CPU truth and healthy qualified hardware. Renoir remains held.

Independent8K GDN drift is still unresolved; changing tolerances, matching greedy tokens or internal bit parity does not close the independent numerical oracle. Genuine resident GPU scope remains512 consumed tokens;16K/32K and broad corpus remain open. Physical memory/quarantine/unknown completion cannot be overridden by catalog status, counter observations or a successful teardown assumption.

Serving remains unimplemented (`serve` refuses; NixOS service definition stays disabled). API route/CLI option frontiers are unreviewed parity seeds, not behavior. Full pinned llama.cpp/vLLM union, source kernels, sampler/RNG/grammar/tools/stops/reasoning/logprobs, batching/speculation and true goodput gates all remain. DSpark2 exact primary source/version is unresolved. Runtime/platform/license/security/distributed/multimodal and physical stream/codec/backing qualifications are separate, not inferred from acquisition or mocks.

Real remaining external/authorization gaps include selected large original checkpoints/license/access and capable hosts; other target hardware/OS/driver/compiler availability; independently authorized recovery for held devices; explicit hosted/network/privacy/cost and peer deployment choices; hidden-bind/subvolume/overlay mount fixtures (not authorized by the existing no-mount tests); and any new SDK/download/publishing actions. Record the concrete gap and continue independent authorized work; do not manufacture generic approval loops or convert lack of test access into support.

### Research and prior-art placement for coordinator integration

The user's context-preservation request should be satisfied under current docs/plans through normal coordinator integration; this audit moves nothing. Keep normative contracts separate from attribution/research, retain unique source/decision details, repair references atomically if anything is later relocated, and preserve implementation/reference fixtures. Do not resurrect retired raw archives or require their old evidence/SHA machinery.

- Entry points: [public docs index](../../docs/README.md), [engineering spec](../../docs/ENGINEERING_SPECIFICATION.md), live plans above and `.agents/docs/` subsystem contracts. The architecture audit is integrated earlier in this handoff; its scope is architecture chronology, while this part preserves current resumable state/backlog/commands.
- Acquisition: [current design/frontier](../plans/model-acquisition/README.md), [A2/A3 historical design specification](../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md), [immutable shards specification](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md), [original acquisition research](../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md), [A1 contract](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md). Old specification headers still describe planned A2/A3; live TODO/README/current code decide implemented scope. The acquisition research's57-row reference is historical, NOT an extant exhaustive ledger.
- Model/numerical/parity: [Qwen basics](../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md), [Gemma contracts](../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md), [model semantics](MODEL_SEMANTICS_AND_COMPATIBILITY.md), [numerics](NUMERICS_QUALITY_AND_BENCHMARKS.md), [independent native probes](../../tests/reference/llama_cpp/README.md). Parity source pins llama.cpp2851b3bcbf8d993a2243ae751c0a7dfa8e944a40 and vLLMa8be1500bad93d3c459e435bf71194aa8062e49d differ deliberately from numerical oracle llama.cpp0.6.0/build11429/d812350. DeepSpec005e03b81cec38b7da6399833d609ee89a2587f2 and DFlash07ebd93db9f472af339b644bb70221ad8428328a are specific research pins. Current CLI/API CSV frontiers,105 groups/145 document seeds are unreviewed, not exhaustive proof. Regenerate selected frontier from pinned source only when work requires, no fresh payload/SDK dependency.
- Native protocol primary pins remain linked in acquisition README References: Hub docs39aa2452668744552cc6690c75cd239771148e58 /Xet corec3c726bed5cf54ded92a63fab892cfb7857c751a/LZ4ebb370ca83af193212df4dcbadcc5d87bc0de2f0; ModelScope client65255a78b451c3785f2dfbfaff97172552fea3af; official cloud/OCI/GitHub/HF contracts; cached ureq3.4.2/ureq-proto decoder boundary. No new research fetch occurred in this handoff.
- Fleet/optimization/streaming/multimodal/serving: each plan's Contracts links point to the substantive research and acceptance registers retained in `.agents/research/` and `.agents/docs/`. All24 research Markdown paths and12 prior-art paths are enumerated below for root's no-loss placement review. They are technical/provenance documents, not completed support or new tracking duties.
- `.agents/prior-art/RESEARCH_SOURCE_ATLAS.md`, comparative compiler/runtime, Gufo fork/XDNA2, accelerator, distributed/tunnel, router/API, SSD and vLLM concurrency studies retain unique attributed mechanisms/limitations/licenses. Their historical external corpus/local path references are starting points, not permission to inspect live external caches or assume those paths still exist. External author benchmark figures are not Zerogravity measurements. Proprietary core engine code is unavailable/not authorized to copy; NPU shared-fabric hang notes forbid automatic clock/privilege work. Root may relocate/distill while retaining source attribution, not silently discard those details.

### Exact open-item snapshot by plan

This section transcribes current unchecked/in-progress/blocked item text as ordinary bullets, grouped by its live source section; the authoritative checkboxes remain only in the live plan/run owners. It is a finite resumption snapshot, NOT a second plan to maintain or independently tick. Read the live target before acting. Completed rows and archives are not duplicated; checked future/rolling documentation changes belong to the coordinator.

#### accepted-baseline — 0 open items

Live source: [.agents/plans/accepted-baseline/TODO.md](../plans/accepted-baseline/TODO.md).

Completed plan; no open checkbox. Remaining work is assigned to the active plans.

#### cpu-gpu-optimization — 17 open items

Live source: [.agents/plans/cpu-gpu-optimization/TODO.md](../plans/cpu-gpu-optimization/TODO.md).

Uncommitted:

- Keep the local Renoir/gfx90c hold: 2B initialization aborted and automatic resets wedged it; no further launch/reset/recovery or Drop-retirement proof is implied.
- Establish controlled cold/warm request baselines across all five GPU/three CPU optimization targets; accepted five backend profiles do not mean five cards or fleet-wide maximal capability use.
- Qualify capability-selected AVX2/AVX512/VNNI quantized kernels, cache/thread/topology policies and source-derived repacking while preserving reduction/rounding/zero/subnormal rules.
- Measure native/Vulkan tiling, subgroup/vector loads, packing, source-exact fusion/prepared dispatch; gate tensor/MMA/WMMA/integer-dot use by exact arithmetic, not advertised support.
- Evaluate explicit NativeF32 projection policy only after physical FMA edge tests, all-codec trained vectors and whole-model/cancellation/replay gates; WideF64 stays default meanwhile.
- Profile then qualify captured graphs/replay, bounded device loops/persistent scheduling, CPU/GPU overlap and exact optional GPU tokenization/detokenization; prepared plans alone are not graph capture.
- Keep separate LLVM code-generation paths and broader accelerator backends unqualified until independently built and physically tested; accepted native paths use CPP NVRTC/HIPRTC.
- Qualify optional EXL3/QTIP/trellis/codebook/low-bit derived packs and calibration identity/quality separately from original source semantics. No hidden precision substitution or default steering.
- Promote only repeated complete-request gains with cold/warm cost, tails, memory, power, source correctness, negative results and qualified capability/version guards.
- Gate future compiler/driver/libm and fast-arithmetic profiles with scalar fingerprints, physical edge/stress and trained full-logit/state tests. WideF64 is qualified on measured inputs, not a universal IEEE FMA emulation.
- Qualify real deadline wakeups, concurrent/multi-stream scheduling and actual driver failure/recovery before broader worker claims; queue deadlines currently need scheduling/poll opportunities.
- Keep the local gfx90c hold. If a healthy/recovered profile is separately authorized, localize retained allocation/H2D/final-sync initialization before changing kernels; natural SIGABRT establishes neither Rust Drop quarantine nor reclaimed storage.
- Consider read-mostly publication only after representative contention and ownership evidence. If left-right fits, test deterministic dual replay without IO/RNG/admission/state/device effects, one writer, persistent non-Sync readers, short guards without await/load/device work and explicit publish/drain/shutdown. Account for both indexes/logs while sharing one payload; do not reopen the rejected cold-catalog migration.
- Resolve applicable SYK-011..016 source/derived-quality/LLVM/SLO/pressure/steering choices from the [specialization reference](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) before that slice; proposed SYK-002..010 are not frozen user approvals. Preserve negative candidates and the default disabled steering policy.

Tests and verification:

- QTIP-01..14: derived source codec/pack identity, transforms, held-out quality and physical kernel correctness pass independently of original model semantics.
- HYB-01..18: exact source routing/draft/tap/mask/reduction and target-authoritative verification pass, including cases where a wrong draft is rejected or deeper drafting slows work.
- GFX-01..18: WMMA/LDS caller/shape coverage, driver/compiler numeric gates and coherent-memory physical headroom pass; require matched complete-request gains. See the [50-test register](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md#4-acceptance-register-50-planned-tests-not-implemented).

#### distributed-execution — 7 open items

Live source: [.agents/plans/distributed-execution/TODO.md](../plans/distributed-execution/TODO.md).

Uncommitted:

- Preserve zero-network standalone execution; add capability/transport/placement seams through the same model/task/coordinator/resource authorities.
- Qualify optional local CPU/GPU/NPU split projection, expert ownership and prefill/decode placement with actual event/memory/power-fabric safety; Stage N is independent of CPU/GPU release.
- Begin authorized two-node layer-group plus local-state execution and immutable full-state hydration; then evaluate P/D and attention+state versus FFN/expert disaggregation.
- Extend tensor/head/pipeline/data/context/expert/sequence parallelism and collectives only where exact state partition algebra, source numerics and transport cost support them.
- Authenticate/version source/state/tenant/session epochs and complete transfers; loss/timeout/stale peers cannot publish partial roots or release unknown in-flight storage.
- Qualify topology changes only at committed frontiers and compare collocation against fast/slow/tunneled links, including real transfer/copy/crypto cost and fairness.
- Add a peer/transport/placement without rewriting model, task, coordinator or API; fleet SSH qualification alone is not distributed inference.

#### handoff-cleanup — 0 open items

Live source: [.agents/plans/handoff-cleanup/TODO.md](../plans/handoff-cleanup/TODO.md).

Completed plan; no open checkbox. Remaining work is assigned to the active plans.

#### historical-records — 0 open items

Live source: [.agents/plans/historical-records/TODO.md](../plans/historical-records/TODO.md).

Completed plan; no open checkbox. Remaining work is assigned to the active plans.

#### memory-and-streaming — 14 open items

Live source: [.agents/plans/memory-and-streaming/TODO.md](../plans/memory-and-streaming/TODO.md).

Uncommitted:

- Route legacy/new model allocations, CPU state/scratch/staging and APU/GPU shared backing into the same configured physical pools with OS/cgroup headroom; never sum overlapping UMA/VRAM/GTT counters.
- Implement paged/contiguous state policy and full-domain prefix identity/intersection/COW across KV/SWA/recurrent/indexer/draft/grammar/media state, source/template/adapter and tenant boundaries.
- Qualify committed snapshot persistence/restore, corruption/security/compatibility UX and restore-versus-recompute; diagnostic portable checkpoints are not product persistence.
- Add immutable source extents and bounded coalesced IO/materialization with stable logical tensor identity, genuine copy readiness and original IO/device reader leases.
- Protect encoded-but-unsubmitted/captured expert reads and generation ABA; support mixed-codec expert size classes under one byte cap, source-exact routing and hot-cache preservation.
- Add bounded PLE/Engram lookup rows and separately governed disk state/media tiers; fully resident execution must incur no mandatory IO or host cache dispatch.
- Test source-exact resident/streamed parity, short/corrupt IO, cancellation, safe victim refusal, SSD QoS and actual physical lifetime before eviction/free-space claims.
- Optional direct/async/platform IO needs equivalent semantics and explicit support; large memory-pressure/SSD experiments require approved hardware/source scope.
- Distinguish logical envelope retirement from driver allocator pages/pools, fragmentation, cross-process use and actual OS release. Driver VRAM/GTT usage/max counters are not reservations or immediately free bytes.
- Prove trusted teardown/recovery before reclaiming unknown-work handles or clearing a fence. Diagnostic service sync/readback permits inspection/exit only and cannot synthesize Ready, retire unknown events or authorize reuse.

Tests and verification:

- SSD-01..32: run the [full source/residency suite](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) for each claimed backing/backend; mock tiers cannot qualify physical IO or trained streaming.
- SSD-14/15: encoded-but-unsubmitted, queued and captured readers prevent generation-ABA slot reuse through actual GPU completion/cancellation.
- SSD-21/22/24: mixed Q2/Q4 expert sizes share one byte cap with safe victims and fair progress; hot-cache retention cannot bypass admission.
- SSD-32: an authorized original trained streamed model matches complete resident logits/StateRoot; measure cold/warm IO, staging, OS headroom and genuine resource lifetime.

#### model-acquisition — 20 open items

Live source: [.agents/plans/model-acquisition/TODO.md](../plans/model-acquisition/TODO.md).

Uncommitted:

- PAUSED at user request2026-10-17 after wrapping current quant acceptance. Resume only when asked; preserve the following unfinished source worktrees and run TODOs. Parent interruption revalidation found all11MAINquantpaths exact frozen draft and no duplicate Cargo; dedicated validator finishes only currentphase.
- Resume C3 filters from `/home/pvl/spaces/abird/src/zerogravity-filters-work/.agents/runs/model-acquisition-filters/TODO.md`: base4db5efb+exactC1/C2overlays, only new acquisition/filters.rs differs from supplied baseline. Bounded matcher/oracle/API scaffold is UNWIRED and NOT compile-ready; all Rust tests unrun. Eight existing integration paths match C2, two integration tests absent; no CLI/module/provider/closure/native wiring. Preserve fixed primary oracle73cases+33345pairs, source-granted fullpath FnmatchCaseV1, includeunion→exclude→ambiguity, required-veto/optional-skip and conservative adjacent-index-before-GET policy. Validate actual512KiB scratch/16Miwork and extra16MiB sameowner admission before acceptance; preserve old Files semantics. Scoped formatting/whitespace only passed; owner stopped with no child process.
- Resume C4 native required indexes from `/home/pvl/spaces/abird/src/zerogravity-xet-index-work/.agents/runs/model-acquisition-xet-index/TODO.md` and sibling model-acquisition-xet-index-design/TODO.md: basef22f759+frozenC2, actualdraftpaths only HF provider and xet/{mod,budget}.rs. Private callback/explicit defaultfalse builder/all-counter Result seed and2unitcases are UNCOMPILED/UNRUN; CLI opt-in/help and integration tests absent. Reuse original reader/scratch, require trusted LFSrawSHA+completeXetroot+actualEOF before parser, carry every counter through resolution→execution, reconstruct again for publication with cumulative charging. Default/dryrun noCAS; native apply explicitly opts options clone in. Reconcile accepted C2 then eventual C3 overlap before qualification; syntax/format only passed, owner stopped/no child. No native-index support advertised.
- Extend cloud profiles through separately qualified S3 FULL_OBJECT checksum authority, broader endpoints/partitions/credentials, Azure latest/SAS/managed chains and per-object digest-map prefixes. Keep explicit version, caller-SHA and conditional FirstSeen contracts distinct; never infer immutable versions or trusted raw hashes from validator/checksum labels.
- Extend ModelScope only through separately qualified alias authority and redirect/CDN/mirror/credential profiles. Selected missing sizes now use exact pinned HEAD; no alias-to-commit authority was established from revision echoes, timestamps or file last-modified IDs. The current profile does not close broader parity or independently prove server revision/membership behavior.
- Extend native Xet to required LFS/Xet index JSON reconstruction under the same source/metadata/admission authority. Ordinary non-LFS index closure is supported; native index302 currently refuses. Define dry-run authorization semantics explicitly before changing the current no-CAS-grant discovery path.
- Measure and qualify native Xet fetch coalescing/chunk caching and any persistent native-prefix protocol. The current bounded per-intersection complete-signed-set reader has counted overfetch/refetch and no performance claim; native process interruption restarts a full transfer. Keep per-term/partial-xorb proof limits, fixed CAS scope and existing elapsed-time/HTTP-trailer boundaries explicit.
- Close finite convenience/compatibility gaps: filter/exclude grammar disposition, required sidecar/adapter/sparse-quant policy, bounded auxiliary/presets, mirror/endpoint/env precedence and gate/license diagnostics. Preserve explicit source authority/no silent numeric fallback and separate loader/runtime plans.
- Qualify any future v1 migration only after separate authorization and proven reader/process quiescence; current removal supports originally protected roots only.
- Extend cleanup only through separately qualified exact ownership/recovery protocols for dead retained staging/general orphans and any later eviction policy. Current exact-snapshot operation deliberately retains them and every stable lock inode.
- Any additional lazy/raw-path/mmap consumer must adopt managed descriptor/open ownership or remain externally quiescent during removal. Observed named cache charge is not a process-global final-close/physical-free-space attestation.
- Qualify additional filesystem pairs before extending the explicit allowlist. Different st_dev values alone do not prove independent backing objects for Btrfs subvolumes, overlay/FUSE or virtualized views; retain branch-specific root device/mount checks and complete inode/ancestry exclusion. No unsupported pair gains acceptance from generic mocks.
- Separately authorize any needed namespace-only hidden-bind/subvolume/overlay filesystem fixtures before mount operations. Real target-root/unselected inode aliases and hidden bind aliases in both directions remain NOT RUN; no host mounts, block devices, new privileges or live cache actions are implied. Current tests use already existing distinct filesystems only.
- Independently qualify any broader local layouts, renamed/unnamespaced roots, missing mount identities or new destination-root creation before exposing support. Keep explicit selection, retained descriptors, bounded topology work and existing cancellation/rollback limits; real import still needs user-selected source/repository/commit/files and copying authorization.
- Finish the finite provider/loader parity ledger beyond qualified HF/GitHub/HTTPS/OCI/GCS/versioned-S3/Azure/ModelScope and explicit local-HF import slices: remaining external-cache layouts/filesystems, additional model hubs, unversioned/broader S3/Azure/compatible profiles, native Xet and broader OCI profiles. Opaque serialization/plugin loaders remain separate runtime work.
- Continue explicit source/license authorization and trusted-digest versus first-seen provenance through additional providers/loaders; acquisition grants no license or loading support.

Tests and verification:

- ACQ-A12: complete the finite provider/loader parity inventory and independently qualify remaining supported paths.
- Each remaining provider phase requires focused API/CLI/protocol/auth/version/range/offline/fault gates, full frozen isolated check.sh, relevant source validation, actual source-integrity and documentation/dependency/root README checks before signing.

Deferred:

- Unsupported plans fail closed until qualified: unknown-length required/model-closure files, capped resumable transfers, unqualified external-cache layouts/filesystems and remaining cloud/OCI profiles, legacy-root removal and unmanaged-reader concurrent reclamation remain explicit above. Independent full HTTP trailer validation is unqualified under the documented decoder boundary, rather than a reason these payloads universally refuse. Independent component data grants no additional loader eligibility.
- Linux local-filesystem semantics are qualified first. HTTP cancellation obeys configured whole-request bounds; old retained attempts consume real quota and arbitrary same-UID mutation remains outside descriptor isolation.

#### model-execution — 10 open items

Live source: [.agents/plans/model-execution/TODO.md](../plans/model-execution/TODO.md).

Uncommitted:

- Acquire/bind authorized original config/index/tokenizer/checkpoint identities and choose a capable admitted host. Large BF16 sources exceed current local RAM; current executable source limit stays 4 GiB.
- Implement explicitly admitted verified owned shards, exact complete index/header/name/dtype/offset/shape/hash binding and retained source leases; no unsafe mutable-file mapping shortcut or second issuer.
- Qwen3.8 27B: original HF grouped-head ordering and Q/K replication 16→48, explicit converted-profile permutations, untied head and source-derived norm/RoPE geometry.
- Gemma E2B/26B: shared GELU/multiply/scale/soft-cap, proportional RoPE, value norm/sliding/shared KV, E2B PLE/doubled FFN and 26B K=V/MoE branches; choose and qualify the exact reference-framework numerical schedule.
- Qualify actual trained CPU operators/full vocabulary/state/nonzero replay, official tokenization/templates/reasoning/stop behavior, sessions, rollback/checkpoints and resource refusal.
- Add the same source-bound forward plans and full physical GPU qualification per eligible device, preserving all state/actual-completion/cancellation guarantees.
- Keep distinct future heavy/hybrid recipes (Qwen Flash-Next, GLM, MiMo and source-defined QSA/MLA/KDA/indexers/hyperconnections), and broaden dense/encoder/pooling/reranking/LoRA support only through independent model/codec/backend gates.
- Preserve CPU-first family priorities; already-authorized parallel basics do not authorize model/SDK downloads or unsupported whole-model claims.
- Bind explicit original HF versus converted Qwen source profiles. Conversion permutes grouped/tiled V projections/constants/convolution/output, uses -exp(A_log) and adds1 to non-GDN norms; GDN swish is SiLU while full-attention gating remains sigmoid.
- Choose and independently qualify Gemma26B reference-framework routing/top-k/reduction/scale placement before production execution. Five analytic test-local routing cases are not trained source parity.

#### multimodal — 24 open items

Live source: [.agents/plans/multimodal/TODO.md](../plans/multimodal/TODO.md).

Uncommitted:

- Add typed sample/frame/codebook/latent/decision values, rational clocks, checked conversions and complete source-bound component/codec/processor assemblies.
- Prove a small real non-token encoder/decision task and bounded iterative task through the existing engine before broad media claims.
- Qualify CPU-first original ASR and non-AR TTS (Whisper/Parakeet and Kokoro candidates); then actual backend execution and measured streaming/audio quality.
- Add depth-AR/RVQ speech, dual-AR codebook TTS and genuine codec-based duplex neural audio (including Qwen-Omni) where source/license/resources permit; mock codecs or text cascades do not establish native support.
- Qualify seeded source-exact image flow/editing conditioner+VAE pipelines, then video temporal/3D-VAE/expert switching and synchronized AV commit groups; candidate families remain separately gated.
- Keep ternary PTQ/Hadamard codecs, schema/calibration decision heads and specialist encoders as independent codec/model/task gates, not aliases of Qwen.
- Implement RealtimeSession lanes/epochs, bounded media buffers, VAD/barge-in, presentation/ack/playback frontiers and transactional rollback without speculative/partial media leakage.
- Add per-model transport capabilities and authorized hosted TaskService adapters with privacy/cost/idempotency/Unknown reconciliation; no duplicate local runtime or guessed realtime guarantees.
- Complete MM-01..24 plus model-specific ASR/TTS/image/video/decision quality/clock/resource suites; no real media-model support follows from source-free mocks.

Tests and verification:

- MM-D1/MM-01/MM-15: checked rational sample/frame/codec clocks, 44.1/48 kHz to 16/24 kHz and 24000/1001 or 30000/1001 FPS, long-duration remainder, discontinuity, overflow and bounded parser/buffer refusal.
- MM-D2/MM-02: complete source/processor/codec/license assemblies; wrong conditioner/shape/mask, missing source transform and cross-session mutable leakage refuse before use.
- MM-D3/MM-03..09: seeded solver/RNG and direct finite decision state, CTC/RNNT/TDT distinction, complete RVQ depth and joint AV commit; interrupted/stale device work never publishes or exposes media.
- MM-ASR-01..12: actual frontend/window/blank/predictor/duration/language/timestamp/diarization source fixtures; offline-only checkpoint refuses streaming, malicious audio bounded before allocation, sessions isolated.
- MM-TTS-01..12: non-AR phoneme/duration/vocoder and exact PCM count; incomplete 9/10 RVQ frame unplayable, codec-rate swap refused, overlap/voice generation and duplex/barge-in/state/lease boundaries correct.
- MM-IMG-01..12: source conditioner/solver/CFG/seed/mask/VAE/color and generation-entrypoint traces; invalid source stage/crop/processor fails, cancel retains old root and unknown physical handles, source/license/hosted status truthful.
- MM-VID-01..15: temporal VAE/latent crop/frame count, Wan timestep experts, source tile masks/rectangular attention, AV cross-attention/joint clocks, invalid GOP/backpressure, hot reload, cancellation and whole-assembly admission.
- MM-DEC-01..06: finite and joint schema head options/order/null/refusal/calibration against original source; generated text is not source-equivalent, external egress refused without authorization.
- MM-BON-01..08: PTQ1_0/PQ2_0 packed source scales/signs/padding plus mandatory 1024 Hadamard activation transform; wrong codec or plain-Q2 path refused, original scalar/CPU/device source parity separately qualified.
- MM-13..17/MM-19: real per-session/turn/transport epochs; stale ACK ignored, duplicate reconnect cursor idempotent/authenticated/bounded, playback progress remains uncertain when missing, slow clients do not release device leases or grow queues indefinitely.
- MM-18/MM-22/MM-23: ten speech/five video mixed jobs, fair shared physical admission and legal solver/frame yield; no per-packet catalog/router/global snapshot or silent precision/step downgrade, coupled output stays unplayable until both tracks valid.
- MM-SRV-01..12/MM-20/21/24: exact versioned SDK/protocol/field/task conformance, hosted accepted-timeout Unknown/idempotent reconciliation, callback/signature/TTL/MIME bounds, malformed media/unauthorized egress/cross-tenant replay refused before side effects.
- MM-I0 change-radius drill: small encoder/finite task and iterative mock use existing source/state/resource owners; no central router/worker rewrite or duplicate runtime; original Qwen gates stay unchanged.
- Eligible source-real CPU→physical full-model→playable-stream→soft-QoS gates: exact licensed checkpoints/processors/codecs and original quality/state oracles, real negative cancellation/admission/replay tests and measured cold/warm tails/memory. Mock/reference/region/hosted results never promote another tier.
- Resolve realtime section18 policy choices explicitly (media numeric policy, first sources, SLO/clock uncertainty, mid-turn changes, persistence, biometric/asset consent, hosted cost/egress, transport packaging and safe physical recovery). User examples do not establish released model IDs or grant execution/provider/download authority.

#### numerics-and-context — 7 open items

Live source: [.agents/plans/numerics-and-context/TODO.md](../plans/numerics-and-context/TODO.md).

Uncommitted:

- Resolve independent original-source CPU 8K GDN numerical divergence without loosening tolerances or changing the declared source arithmetic.
- Extend genuine resident GPU context/corpus qualification beyond the currently accepted 512 consumed tokens, including capacity/window/padding/replay boundaries.
- Run genuine whole-model 16K/32K CPU/GPU gates only after prerequisites; functional 8K success is not independent long-context numerical closure.
- Keep internal full-logit/all-state/trace bit parity distinct from independent native-source stage/top20 comparisons, including the known 2B 56 signed-zero distinctions.
- Broaden prompts/tokenizers/precision profiles and sampling quality only with independent oracles, exact history and explicit qualified ranges.
- Diagnose identical teacher-forced prefixes512/1024/2048/4096/8192 against pinned llama.cpp0.6.0/build11429 `d812350`, CPU/F32 KV and Flash Attention off. Fix the physical microbatch/chunk/arithmetic schedule and compare per-head GDN state, gain/beta, quantization and AR-versus-bulk recurrence; distinguish replay correctness from arithmetic.
- Compare native ubatch1/64/256/512 only within bounded feasible runs; the historical full ubatch1 capture did not finish. Use raw token IDs to avoid incomplete UTF-8 HTTP rows. Recheck short corpus, layers, state and actual8K after any fix; retain original functional0.45 shared-top20-logprob/0.05 top-token-probability limits and strict source-bit operator gates.

#### platform-and-release — 13 open items

Live source: [.agents/plans/platform-and-release/TODO.md](../plans/platform-and-release/TODO.md).

Uncommitted:

- Generalize the working typed forward plan into verified effectful semantic regions/task control, closed provider substitution, bounded shape/kernel caches and explicit command/resource legality; no new persistent IR/runtime without a real need.
- Prove artifact→model→compiled task→coordinator→output and explain-plan for supported tasks, including source-qualified preselected fallback and no selected-GPU-error CPU substitution.
- Run extension drills for one stateful semantic op, device/provider, non-token TaskProgram, quant/storage/API adapter and optional ParameterVersion/TrainStep seam; future training is not thereby implemented.
- Finish per-file source/model/license/NOTICE/SBOM policy, malformed-input fuzzing and ownership/security review, authenticated APIs/private caches and qualified driver/device safety.
- Verify Nix/module/native/cross packages and installer/release behavior on actual Linux/Windows/macOS target access; packaging definitions alone do not prove platform support.
- Maintain minimal/default/composed CPU-free feature isolation, one lock/version authority and truthful unsupported capability behavior.
- Gate bounded compiler/autotune/AOT caches by source/device/driver/shape and retain negative candidates; no unbounded first-request search.
- Complete reproducible/signing/release policy and fair model/task/quality/performance matrices with cold/warm, p50/p95/p99, memory/energy and uncertainty. Push/tag/release remain separately authorized actions.
- Demonstrate compatibility/invalidation of operator/task/state/plan caches and bounded change radius before broader platform claims.
- Before a true zero-copy strided View is enabled, validate contiguity against stored value provenance and test public PlanValue retargeting. Current View physically materializes dense output; retain cached-element and dynamic range guards.
- Preserve one GPU-library service/queue/quarantine owner and cold generic-plan admission through packaging changes; verify actual final downstream links, not only compiler/type checks.

Tests and verification:

- Qualify all six [CPU release definitions](../../docs/DEVELOPMENT.md#cpu-cross-release-definitions) with actual target execution, installer checksum/version/refusal cases and platform signing/authenticity policy. Build definitions and same-host checks do not establish other-target support.
- Test enabled NixOS service lifecycle and runtime-secret handling after serving is implemented; the currently disabled service definition and module evaluation are not a running-server claim.

#### serving-and-speculation — 9 open items

Live source: [.agents/plans/serving-and-speculation/TODO.md](../plans/serving-and-speculation/TODO.md).

Uncommitted:

- Implement canonical TaskRequest/coordinator workers and CLI/SDK/API integration, model routing/residency lifecycle, explicit supported OpenAI/Anthropic interfaces, streaming/tools/reasoning/schema/sampling and truthful capabilities.
- Enforce separate API/tenant payload, queued-prefill/residency and physical active-step admission; bounded overload, cancellation, disconnect, stale epochs and output backpressure.
- Add real compatible continuous/grouped batches, aged chunked prefill, decode/media fairness and finite joint batch/draft/expert/kernel/placement plans; no per-token catalog/JSON/global-lock work.
- Finish Qwen basics before optimizations; enumerate the complete pinned llama.cpp/vLLM union, public and internal surfaces. Preserve 105 provisional groups/145 document seeds and CLI/API frontiers as unreviewed inputs, not exhaustive parity.
- Implement every applicable sampler/logits-processor/RNG/logprob/grammar/tool/stop/reasoning composition and its independently verified outputs.
- Qualify target-authoritative MTP, draft model, n-gram/suffix, EAGLE, DFlash/DFlash2, DSpark and related registered strategies; DSpark2 exact primary source/version remains unresolved.
- Test all accepted-prefix lengths, target-distribution correction, target/draft RNG/grammar ownership, full KV/recurrent/index/media rollback, physical branch admission and authentic cancellation/quarantine.
- Jointly benchmark material combinations, adaptive depth/backoff and all offered/rejected/canceled work: committed SLO-goodput, TTFT/ITL tails, fairness, memory and energy versus equivalent pinned vLLM/V1/Omni.
- Preserve zero untriaged frontier/disposition requirements before any full-parity claim; synthetic acceptance, fewer launches and competitor measurements do not establish local parity or speed.

### Retained research and prior-art source path index

For coordinator placement/context preservation only. No files were moved or copied, and no new mandatory archive exists. Source diagrams/code/fixtures remain in their actual owners.

#### .agents/research — 24 files

- [EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md)
- [GEMMA4_FAMILY_CONTRACTS_2026-10-16.md](../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md)
- [GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md](../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md)
- [HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md](../research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md)
- [INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md)
- [MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md](../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md)
- [MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md](../research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md)
- [MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md](../research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md)
- [MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md](../research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md)
- [NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md](../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md)
- [PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md)
- [QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md](../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md)
- [README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md](../research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md)
- [SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md](../research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md)
- [SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md)
- [STRIX_HALO_OPTIMIZATION_DOSSIER.md](../research/STRIX_HALO_OPTIMIZATION_DOSSIER.md)
- [VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md)
- [ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md](../research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md)
- [CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md](../research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md)
- [DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md](../research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md)
- [IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md](../research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md)
- [SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md](../research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md)
- [TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md](../research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md)
- [VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md](../research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md)

#### .agents/prior-art — 12 files

- [ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md)
- [ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md](../prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md)
- [ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md)
- [COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md](../prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md)
- [DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md)
- [FINAL_ARCHITECTURE_REVIEW_2026-10-12.md](../prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md)
- [GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md)
- [MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md)
- [PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md)
- [RESEARCH_SOURCE_ATLAS.md](../prior-art/RESEARCH_SOURCE_ATLAS.md)
- [SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md)
- [VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md](../prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md)

### First resumption checklist for the next coordinator

1. Obtain/recognize an actual user resume; this documentation-only request leaves execution paused. Read the current rules/index/relevant live TODOs and inspect current MAIN/status/worktree processes. Preserve rolling docs and other owners. Do not recreate retired worktrees or treat this snapshot as an authority to reset source.
2. Review the accepted f197c9c code/current delta and the two paused runs. Reassign named paths and one Cargo owner. C3 is a single unwired file; C4 has exactly3 unfinished deltas. Establish integration order for HF/xet/CLI overlap; preserve current C2 basename fix, old Files, native/default budgets and the sole publisher.
3. Resume the chosen smallest source slice in its preserved SIDE, keeping a current own run. Review independent goldens/design before edits; keep absent/unrun fixtures explicit. No source copy/merge/compile merely because a file formatted.
4. At a frozen reviewed boundary, coordinator copies only agreed owned changes with fresh mtimes, checks intended byte preservation and retains original source/dependency/README authority. Never copy an older complete SIDE baseline over newer MAIN or run two Cargo owners.
5. Run appropriate focused actual API/protocol/CLI/auth/offline/cancel/fault/process/admission/legacy tests and both strict configurations under canonical isolation; report actual failures before changing policy/expectations. Full frozen check.sh, actual Nix source-integrity/new-file inclusion/current links, protected README/dependency checks and applicable original source/model gates precede acceptance. Additional device/input/recovery/download authority remains separate.
6. Coordinator atomically integrates useful run results and every unfinished item into current canonical docs/plans, signs verified code/tests/plan with explicit paths and carries rolling commit provenance to the next update. Full plan stays unchecked until every item/gate is actually complete. Future continuation should not stop after a convenience milestone when the user has resumed the broader authorized task, but it must obey any new pause.

Documentation-only validation of this handoff cannot establish implementation correctness, feature support or successful recovery of either paused draft.


<a id="operations"></a>

## Practical reconstruction and handoff boundaries

### Portable unfinished source under the requested documentation roots

All new handoff material is under `.agents/docs/` or `.agents/plans/`. The
supporting directory [handoff-2026-10-17](handoff-2026-10-17/) contains:

- [PAUSED_FILTERS_CONTEXT.md](handoff-2026-10-17/PAUSED_FILTERS_CONTEXT.md): exact
  paused C3 design/grant/state/pending-test record.
- [paused-filters.patch](handoff-2026-10-17/paused-filters.patch): the one new,
  unwired `acquisition/filters.rs` relative to implementationf197c9c.
- [PAUSED_XET_INDEX_CONTEXT.md](handoff-2026-10-17/PAUSED_XET_INDEX_CONTEXT.md): exact
  C4 incomplete implementation and resumption record.
- [XET_INDEX_DESIGN.md](handoff-2026-10-17/XET_INDEX_DESIGN.md): complete cold
  protocol/admission/counter/verification design; the current builder decision in
  the implementation context supersedes its older named-resolver suggestion.
- [paused-xet-index.patch](handoff-2026-10-17/paused-xet-index.patch): exact three
  draft production-file deltas againstf197c9c.
- [paused-filters-oracle.py.txt](handoff-2026-10-17/paused-filters-oracle.py.txt):
  inert source of the independent expected-vector generator. Its original
  MAIN run copy fetches primary code and overwrites an absolute SIDE source path;
  do not execute it as a handoff step or replace fixed expectations with Rust.

These are unfinished source/context, not historical proof archives. Actual
`git apply --check`, application and byte comparison in an owned disposable
reconstruction tree reproduced all four draft files exactly fromf197c9c. That
proves recovery bytes, not type correctness, tests, support or merge readiness.
An initial check from a subdirectory accidentally discovered the parent Git root
and skipped the patch; the missing-file comparison caught it. A separate owned
temporary Git root fixed the check. MAIN/SIDE source was never modified by it.

Prefer inspecting the preserved actual SIDEs when they still exist. If they are
missing and continuation has been authorized, create separate disposable
worktrees atf197c9c and apply only the corresponding patch in each. Inspect
before applying; never apply drafts blindly onto a newer MAIN or combine them
without reconciling shared HF/quant/Xet/CLI seams. A mechanical reconstruction
example, not an instruction to resume during this documentation task:

```sh
git worktree add --detach ../zerogravity-filters-resume f197c9c
git -C ../zerogravity-filters-resume apply --check \
  /home/pvl/spaces/abird/src/zerogravity/.agents/docs/handoff-2026-10-17/paused-filters.patch
git -C ../zerogravity-filters-resume apply \
  /home/pvl/spaces/abird/src/zerogravity/.agents/docs/handoff-2026-10-17/paused-filters.patch
```

Use a different explicit worktree for the native-index patch. Detached bases are
context references, not permission to reset or overwrite current work. The
existing filters SIDE started earlier with accepted overlays; the portable patch
has already normalized its actual one-file delta ontof197c9c. The index patch
likewise excludes accepted C2 overlay changes. Both features are incomplete.

### Workspace owner map

| Owner | Responsibility and first source locations |
| --- | --- |
| `zerogravity-core` | Immutable source IDs, state/session roots, physical ResourceArbiter, device queues/events and typed accelerator programs; `crates/zerogravity-core/src/{source,state,session_root,resource_arbiter,device_queue,device_events,accelerator}.rs`. |
| `zerogravity-artifacts` | GGUF/config/safetensors/model contracts, quant/source weights, admitted immutable source materialization, managed cache/acquisition; `src/{artifact,model,sharded,source_weights,quant}.rs`, `src/{cache,acquisition}/`. It is substantive runtime/data ownership, not evidence tooling. |
| `zerogravity-cpu` | Portable source math/quantized weight execution; `src/{math,weights}.rs`. |
| `zerogravity-model-qwen35` | One Qwen model semantic schedule, immutable shared model/session/checkpoint/tokenizer/source-profile contracts; `src/{lib,forward,host_math,checkpoint,tokenize,model_catalog}.rs`. |
| `zerogravity-model-gemma4` | Config/tensor/variant/MoE/E2B/26B source contracts and their tests, not trained execution. |
| `zerogravity-gpu` | Reusable resident-plan execution, source buffers, prepared bindings, one service/queue/quarantine and Vulkan/CPP CUDA/HIP providers; production features default off, no duplicate Qwen loop. |
| `zerogravity` | Library facade/preparation/semantic task drivers/session coordination; `src/{lib,prepare}.rs` and original-model integration tests. |
| `zerogravity-cli` | Sole product executable; inspect/run/backend frontends/pull/cache; defaultCPU+Qwen+text. `serve` remains unimplemented. |
| `tools/gpu-qualify` | Workspace qualification tool linking the same GPU library. Legacy executable name `zerogravity-gpu-probe` may remain; there is no independent `probes/` workspace or lockfile. |

One root `Cargo.lock`, one workspace version authority, production defaults separated
from diagnostics, default-members facade+CLI. Vendored CubeCL0.11.0 IR/CPP/CUDA/HIP
packages are patched path dependencies under the GPU crate and excluded as
workspace members. Preserve license/NOTICE/provenance and downstream patches.

### Concrete build and qualification commands

The pinned toolchain is Rust1.95.0 with rustfmt/Clippy. See
[development guide](../../docs/DEVELOPMENT.md),
[GPU library runbook](../../crates/zerogravity-gpu/README.md),
[qualification tool](../../tools/gpu-qualify/README.md) and
[reference probes](../../tests/reference/llama_cpp/README.md).

```sh
# Revalidation: read current state, do not reset to a historical reference.
pwd
git status --short --branch
git log -5 --oneline
git worktree list --porcelain

# Locked/offline CPU build, using only existing authorized dependencies.
cargo build --release --locked --offline -p zerogravity-cli

# Complete implementation-phase gate, one MAIN target owner.
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
  python3 scripts/native-gpu.py -- bash scripts/check.sh

# Actual source-filter check; new source must be included in the Git flake first.
nix build --offline --no-link .#checks.x86_64-linux.source-integrity

# Original trained cache/direct route (both approved models in this one test).
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo test --release --locked --offline -p zerogravity --features cache \
  --test cache_prepare approved_cached_models_match_direct_source_exactly \
  -- --ignored --exact --nocapture

# Original trained external-HF-import route, separately and sequentially.
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- \
  cargo test --release --locked --offline -p zerogravity --features cache \
  --test cache_prepare approved_hf_import_models_match_direct_source_exactly \
  -- --ignored --exact --nocapture
```

Do not execute new implementations/tests/hardware work until resumed. The above
commands document the accepted workflow; this handoff documentation task does
not rerun model/build qualification unnecessarily. A new source phase must pass
its focused meaningful API/CLI/protocol/fault checks, all appropriate feature and
strict lint configurations, full frozen gate, actual Nix/source/document checks
and relevant trained/physical gates before acceptance. Source-free mocks and
compilation do not qualify a model or device.

Actual approved GGUF source hashes, also enforced by the existing tests:

| Original input | Raw SHA256 |
| --- | --- |
| Qwen3.5 0.8B | `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf` |
| Qwen3.5 2B | `91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6` |

Existing source/config/tokenizer fixtures live beneath `target/models/qwen35-0.8b`
and `target/models/qwen35-2b`; tests bind their own exact config/tokenizer bytes.
If absent or changed, record the missing approved input; do not download a
replacement or silently select another checkpoint. Source hashes are real
model integrity inputs, not a revived evidence-log inventory.

Use actual native user/network isolation, not Cargo's offline flag alone: build
scripts may otherwise access the network. The default native CPP guard uses
existing NVRTC/HIPRTC without LLVM. Explicit `--require-llvm`/`--verify-only` checks
existing pinned LLVM inputs; it never authorizes acquisition. GPU commands select
an explicit provider and must use a healthy qualified physical profile. Never
run held local AMD or destructive cancellation probes in a shared long-lived
process; unknown work stays quarantined/fenced until safe physical termination.

### Packaging and product boundaries

The six CPU cross-release definitions cover Linux x86_64/aarch64 musl, Windows
x86_64 GNU/aarch64 gnullvm and macOS x86_64/aarch64. Definitions, checksums and same
host installer fixtures do not establish physical per-OS model execution,
authenticity, Windows/macOS signing or permission to fetch Apple SDKs. GPU
binaries are explicit builds, not silently included in the six CPU assets.

Installers require an explicit repository/baseURL, verify checksum and version
and install no service. A checksum from the same host detects corruption but is
not independent authentication against compromise. Nix exports CPU packages,
source/module checks and a disabled service definition. Enabled `serve` currently
refuses and `Restart=no` avoids a restart loop; a service definition is not a
working OpenAI server. No automatic firewall exposure, persistent service
installation, secret ingestion, tag/release/publication or remote peer setup is
part of onboarding.

The owner requested comprehensive capabilities and superior measured performance,
without approving arbitrary percent targets/deadlines or hidden fallback. Native
CPU/GPU source semantics, model quality, request-level throughput/tails, fairness,
energy and memory must be qualified separately. “Full GPU pipeline” currently
means the accepted original-model resident scope; it does not imply GPU
tokenization, graph capture, all-context/model support, all hardware optimized or
complete serving/distributed/multimodal products.



<a id="decisions"></a>

## Complete retained decision register — dated context snapshot

The following reproduces the current canonical decision record so every user, proposal, open-question and assumption ID remains available in this handoff. Its canonical source is [DECISION_REGISTER.md](../plans/DECISION_REGISTER.md); update that owner for future decisions. Historical status and process statements retain their dates and are superseded where the current TODO rules, later owner decisions or verified implementation say so. In particular, older A2-planned/full-GPU-pending/LLVM-blocked/archive-SHA/all-history-in-Git text is not current support or workflow. The sole binary name and MIT project license were subsequently settled; third-party/model licenses, distribution/signing and broader platform questions retain their independent gates. A complete record of an OPEN row is not a claim that every part of it is still unresolved.


**TODO-USER-001 · USER ·16 October2026:** Replace the handoff/evidence process
with `.agents/plans/TODO.md`, one `.agents/plans/<plan>/TODO.md` and independent
`.agents/runs/<plan>-<session-or-agent-id>/TODO.md` files. Keep a provenance
commit, implementation/test checkboxes, completed commit sections and a rolling
Uncommitted section. The coordinator atomically integrates agent results into
main plans; finish a plan by checking its one index line. Archive only completed
TODO sections as needed. No mandatory generated handoffs, SHA inventories,
source snapshots or run receipts. [Workflow](../../AGENTS.md). This supersedes
older process/preservation/update duties below; substantive architecture and
actual model/source/numerical/state/resource requirements stay in force.


**PULL-A1-001 · VERIFIED ·16 October2026:** Keep local cache publication in the
existing artifact owner behind optional Linux cache/diagnostic features. Reuse
caller-owned physical admission, stable process locks and exact retained file
handles; report post-ref-rename sync failure as explicit DurabilityUnknown.
No executable readiness or source identity follows from data publication. A01..09
pass within [the phase's stated limits](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md);
A2 HTTPS/providers/CLI and A3 owned-copy loader integration remain next.


**EVIDENCE-USER-002 · USER ·16 October2026:** Remove SHA inventories for ordinary
evidence logs and simplify the evidence specification. Git commit, actual command,
result and tested scope establish the record. No global evidence catalog,
per-log hash or source-file hash inventory is required. Keep hashes at real
model/artifact/content or explicit byte-identity qualification boundaries.
[plan TODO index](../plans/TODO.md). This supersedes older log
inventory/sealing instructions; design-document SHA tooling remains separate.


**CONC-USER-002 · USER ·16 October2026:** Use left-right thoughtfully at
real contention sites. The cold model catalog is not a justified migration.
Reject its uncommitted implementation and restore the accepted production
source/dependencies; preserve the experiment as rejected evidence. Require
representative access/contended-wait evidence and ownership fit before adding
a concurrency primitive. Existing direct immutable sharing and authoritative
transactions retain their appropriate design. [Correction and current review](../plans/cpu-gpu-optimization/TODO.md).


**H25 promoted-product gate ·16 October2026:** Both original Qwen3.5 models
now pass27 checks on each of local CUDA/NVIDIA Vulkan, pvl-x2 HIP and pvl-a1
CUDA/HIP:135 actual checks, including cold prelaunch negatives, plus12 actual
scalar/libm tests. Genuine512 full logits/state/trace, replay and physical
cancellation/quarantine pass; the local gfx90c hold remains. Canonical archives
and exact retirement now preserve old records inside `.agents/` with original
phase/path/hash mappings. Final repository/catalog/documentation closure remains
in progress. New Qwen/Gemma source work is isolated and does not inherit trained
execution from these results. [Current phase](../plans/accepted-baseline/TODO.md).


**QWEN-PARITY-USER-001 · USER ·16 October2026:** For Qwen, finish the
source-correct basics first, then match every capability and optimization of
llama.cpp and vLLM, then add further appropriate MTP, speculative decoding,
n-gram, DSpark2/DFlash2 and related optimizations. Maintain a version-pinned,
exhaustively enumerated capability ledger; each applicable mechanism needs
functional/source/ownership qualification and measured performance. Methods
already present upstream remain parity gaps until their own gates pass.
No guessed model/drafter pairing or synthetic acceptance counts as inference.
[Ordered implementation, complete coverage and acceptance](../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md).


**MODEL-PARALLEL-USER-001 · USER ·16 October2026:** Begin Qwen3.8 27B,
Gemma4 E2B and Gemma4 26B source/CPU implementation in parallel while current
qualification runs. Use subagents and disposable isolated worktrees where
changes overlap; main coordinates, reviews, reconciles and merges verified
pieces. The two Gemma tracks share one family implementation. Preserve the
existing engine, source identity, state/resource authority and new-model
CPU-first acceptance. No new checkpoint/SDK downloads or GPU capability claims
are authorized. This adds parallel model work while shared layout → left-right
→ XDG acquisition → further GPU optimization sequencing remains binding.
[Ownership, source gates and integration plan](../plans/model-execution/TODO.md).


**LAYOUT-RETIRE-USER-001 · USER ·16 October2026:** Remove `probes/` entirely,
moving implementation, reference tests and helpers to `crates/`, `tests/` and
`tools/` according to their real responsibility. Audit sidetree worktrees and
clean up all that are merged or superseded while preserving unique work and
active users. Resnapshot and archive historical evidence in the existing
`.agents/` documentation model; all retained historical evidence must live
inside `.agents/`, never in external build directories. Store underlying
records and original-path/hash/disposition indexes under
[`.agents/plans/evidence/`](../plans/TODO.md); obsolete build caches and
compiled executable bodies are retired after their provenance is recorded.
This extends the final H25 layout gate and does not change the layout →
left-right → XDG acquisition → GPU optimization order.

**Next: [workspace/model/feature layout redesign](../plans/accepted-baseline/TODO.md), immediately followed by the [left-right concurrency redesign](../plans/cpu-gpu-optimization/TODO.md), then individually qualified CPU/GPU performance variants.** Preserve the [five-GPU/three-CPU target inventory](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md), existing model/serving priorities and [GPU text/device orchestration roadmap](../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md). Fleet-wide capability optimization is unfinished; [native FP32 FMA](../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md) remains a proposal, not an enabled policy. Capacity remains 512; broader GPU contexts/corpora, independent CPU8K GDN divergence, genuine16K/32K qualification and product checkpoint persistence/security UX remain open. The default CPU CLI and user-authored root README are unchanged. Earlier status entries below are preserved as historical evidence.

**CRATES-USER-001 · USER ·16 October2026:** After the native backend qualification boundary, promote cohesive implemented packages under `crates/`, keep a small public core/facade, and make backends, model families and product capabilities independently selectable with real minimal/composed/`all` build gates. The bootstrap crate count is not a permanent cap; extraction preserves the same engine, state authority, physical ownership and source bits. [Workspace/model/feature redesign roadmap](../plans/accepted-baseline/TODO.md) records proposed package seams and sequencing; none is implemented by this documentation.

**CONC-USER-001 · USER ·16 October2026:** Apply [jonhoo/left-right](https://github.com/jonhoo/left-right) broadly wherever it fits read-heavy published immutable data/indexes, replacing application read locks or mutexes where semantics and measured performance justify it. Conduct this concurrency redesign pass **immediately after the workspace/layout/structure redesign**. Reinventory every owner; preserve one writer, coherent publication, deterministic replay, existing session/resource/actual-completion authority and physical retirement. [Complete candidate inventory, migration and acceptance](../plans/cpu-gpu-optimization/TODO.md) includes hidden second-copy slot retention and unregister/drain/re-register source identity, bounded guards and writer delay. This request adds the roadmap/design direction now; dependency and runtime changes belong to its later verified implementation units.


**HW-FLEET-USER-001 · USER ·16 October2026:** Initial optimization targets are pvl-l5 (current), pvl-a1 and pvl-x2: five physical GPUs and all three CPUs. Explicit SSH work is authorized on the two remote machines. Optimize capability-selected execution paths, source-exact kernels, memory placement and complete-request performance for these targets first; advertised instruction support alone is not qualification. The user also explicitly requests subagents for each implementation going forward, using independent ownership and parallel work to keep the main thread focused and efficient. [Observed fleet/ordered optimization gates](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md) and [active native CUDA/HIP phase](../plans/accepted-baseline/TODO.md) retain scope/evidence. Temporary test artifacts and reuse of already-owned approved checkpoint bytes support that implementation; new external model/SDK acquisition, persistent device/service/privilege changes and publication retain existing restrictions. Root README stays unchanged.


**OPEN-001 · USER · 15 October 2026 (MIT, AI disclosure and open contributions):** The user selected **MIT** for the Zerogravity project source and documentation, requested the standard root `LICENSE`, and approved clearly recording `license = "MIT"` for the root and optional GPU package. README must include a succinct AI disclosure influenced by [ds4](https://github.com/antirez/ds4): **human-centric design with AI-centric implementation**, substantial AI assistance without unchecked autonomous control and without claiming human line-by-line inspection. The founder **Prasanna Loganathar** explicitly welcomes everyone to build the project openly and kindly; preferred contributions start as design/architectural issues or small, contained implementation changes with a documented plan under `.agents/plans/` before nontrivial coding. [Public contribution guide](../../CONTRIBUTING.md) and [full policy](../plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md) capture the workflow. MIT applies to this project's own material; model weights, dependencies, codecs and imported assets retain their independent licensing. User approval covers documentation/contributor/licensing metadata, not new inference implementations or external service access.


**SERVOPT-USER-001 · USER · 15 Oct 2026:** After a read-only vLLM concurrency review, the user explicitly instructed strengthening every recommended concurrent serving/design/implementation plan, then a complete product-focused README rewrite with Intro, Quick start, Architecture, FAQ, Implementation status/roadmap, concise engineering bullets and high-level diagrams including joint optimization. The user requested exactly: **The blazingly fast AI execution fabric - built for composable multi-modal inference specialization at every layer, from individual kernels to distributed systems**, and early explicit text/image/video/audio/realtime scope. Examples [dotlink](https://github.com/abird-ai/dotlink) and [agentc](https://github.com/abird-ai/agentc) are structural references, not required implementation copies. [Joint serving contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) adds accurate source-verified vLLM-class bounded concurrency and joint legal batch/spec/cache/kernel/placement/QoS; [benchmark/acceptance](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md) specifies 36 PLANNED, not-run tests and truthful comparison conditions. This is authorization for docs/README and minimal doc tooling only, not new GPU serving engines, benchmarks, downloads or deployment. Existing one-Rust-engine and no-hidden-quant policies remain unchanged.


**VST-USER-001 · USER · 15 October 2026:** The user explicitly authorized clean changes to architectural docs/plans **and the minimal documentation tooling needed to repair the nested full handoff archive**, overriding the earlier design-only restriction for these documentation artifacts, while also requiring **SSD streaming to become a first-class, nearly zero-abstraction-cost storage policy** under the same source tensor model as fully resident GPU memory. DS4/DwarfStar and related SSD-streaming strategies are inspirations; one physical ResourceArbiter/TaskCoordinator and Rust inference engine remain mandatory. Direct fully resident fast path should avoid extra software overhead, but actual SSD misses necessarily pay IO; source tensor identity, verification, GPU command reader pins and real completion must be preserved across GPU/UMA/RAM/SSD tiers. Detailed specs: [first-class virtual tensor residency](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md), [source research](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md), [SSD-P0..P5/SSD-01..32](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md). **Current user decision authorizes docs/tooling, not actual SSD inference implementation or large model downloads**. Proposed policies/tests are specified but not yet verified in actual hardware. This VST-prefixed local decision avoids earlier U-040/U-041 historical numbering collisions.


**15 October 2026 design-only source-specialization clarification:** After the explicit read-only specialist-source review, user requested **all its precision/specialization gaps resolved in documentation and implementation plans, cleanly and elegantly**. The [specialization implementation plan](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) preserves the one-Rust-engine and original model priority user commitments. New identifiers **SYK-001..010 are PROPOSED design obligations; SYK-011..016 are OPEN policy/source/permission questions**, NOT additional user-frozen ABI approvals. Do not conflate with earlier v5.1 addendum's historically reused U-040/U-041 or overwrite the shared U-040 native model implementation entry. No new code/model/driver changes were requested by this instruction.


**Read this before interpreting any spec.** Project date 2026-10-10; provenance is the originating conversation. Identifiers are stable for future design reviews. Status: `USER` = explicitly requested or endorsed; `PROPOSAL` = assistant-origin recommendation, not separately approved; `OPEN` = requires a measured or policy decision; `ASSUMPTION` = working interpretation, reversible.

### Explicit user decisions

| ID | Status | Decision, in the user's intent |
|---|---|---|
| U-001 | USER | Project name **zerogravity** (lowercase). |
| U-002 | USER, SUPERSEDED PATH | Originally requested comprehensive agent-readable plans, specs, decision history and takeover documentation under the relative `./zerogravity/.agents/{docs,plans}` path, via **dotlink l5**. Its location requirement is superseded by U-013; the documentation scope still applies. |
| U-003 | USER | The inference engine's own implementation is to be **entirely Rust**; record a strict desire for Rust-authored kernels, not a C++ inference engine wrapper. What counts as unavoidable vendor runtime bindings is clarified in O-003. |
| U-004 | USER | Support standard model families, expressly **Llama**, **Gemma 4**, **Qwen 3.5+** (dense/MoE/hybrid) and additional newer models, rather than specialize for one fixed checkpoint. |
| U-005 | USER | Support unusually large / specialized models including **Qwen3.8-Flash-Next (125B backbone + auxiliary table/MTP)** and **GLM-5.3-Flash (~321B total)**, especially on **Ryzen AI Max+ 395 / Radeon 8060S / AMD Strix Halo (gfx1151)**. |
| U-006 | USER | Target vLLM/SGLang-level peak server efficiency while retaining llama.cpp-like accessibility/portability and Maya/Gufo/DwarfStar-like hardware-specific extreme optimizations. |
| U-007 | USER | On suitable hardware activate maximal specializations; **otherwise fall back to general model execution**, without degrading unrelated optimized operations. |
| U-008 | USER | Combine architectural learning from vLLM, SGLang, llama.cpp, Maya, Gufo, Strata, DwarfStar (antirez/ds4), plus complementary runtimes and Rust projects. |
| U-009 | USER | Adopt the **Semantic IR-based** architecture as a way to express model computation once and allow specialized execution without a hot-path abstraction penalty; ambition is to *outclass* existing engines where demonstrably possible. |
| U-010 | USER | Prior stage was **research only**, then explicit request to refine architecture, now explicit request to persist comprehensive plans and handoffs. No code implementation was asked in this turn. |
| U-011 | USER | The record must preserve **all user decisions, detailed design decisions and pending questions**, so external LLM agents can implement from the files without relying on chat history. |
| U-012 | USER | Performance ambition is across both model breadth and specialized hardware. There is **no user-approved target performance percentage or deadline**; faster-than-everything is a goal, not a result. |
| U-013 | USER | **Canonical project location is `/home/pvl/spaces/abird/src/zerogravity/`**, explicitly correcting the original relative placement. Move existing documentation there, preserve files, update references and revalidate. This supersedes only the destination in U-002. |
| U-014 | USER | **Incorporate the best of the research/inference architecture and src/zerogravity Semantic IR architecture, adapting all plans to a unified stronger design.** The user approved the combination goal; this does not automatically approve each old council-specific Rust ABI, license or performance threshold. |
| U-015 | USER | Conduct **comprehensive new research** to design zerogravity for maximum potential to surpass existing inference projects while preserving long-term maintainability, clean adaptability to future technologies and robust extensibility at **every stage of the pipeline**. State measured evidence versus conjecture and update the authoritative plans. |
| U-016 | USER | Explicitly include the architectures and pipelines demonstrated by [mikealanni/gufo](https://github.com/mikealanni/gufo) and the October 11 Strix Halo Reddit report on **Qwen3.8 Flash-Next 1,800+ tok/s cooperative GPU+XDNA2 NPU prefill**, making these capabilities fit elegantly into zerogravity rather than becoming model-specific parallel runtimes. This is a **design/support requirement**, not proof that XDNA2 code is implemented or automatically safe to enable. |
| U-017 | USER | **Generalize all project plans, specs and architectural interfaces away from external project names**, with an exception permitting concrete naming/reference to verified MIT-licensed sources. Retain all upstream work as **identified, attributed prior art and research inspiration**, rather than as architectural base classes or cloned engines. Build zerogravity's **own original, modular and composable AI execution architecture** so new model/accelerator/quantization/task/pipeline specializations do not require rewriting new inference engines. Preserve model-family and hardware/protocol identifiers when materially necessary, and do not lose source provenance or license restrictions. |
| U-018 | USER | **Perform another comprehensive critical review/refinement of the entire architecture and detailed specifications**, aiming for the highest-performing and most composable runtime across AI execution tasks, while remaining **minimal whenever complexity has no proven value** and modular enough to incorporate novel models, devices, kernels and AI task pipelines without engine rewrites. This requests architecture/design refinement, not immediate implementation or proof of benchmark superiority. |
| U-019 | USER | **Verify and generalize accelerator/runtime architecture coverage** beyond local Strix Halo: explicitly include NVIDIA DGX Spark/GB10 coherent Grace–Blackwell, discrete and datacenter/rack-scale Blackwell, Google TPU, possibly Cerebras wafer-scale (user phrasing “cerberus” also corresponds to a separate parallel-decoding technique), and other programmable AI accelerators. Maintain one composable source-independent runtime that can absorb **native-kernel, AOT graph, deterministic/static dataflow and other specializations at their natural execution granularity** instead of cloning another engine. This is an architecture/research requirement, not proof of actual hardware support. |
| U-020 | USER | **Distributed and networked execution is an additional first-class zerogravity goal**, not just fastest inference on one machine. Support heterogeneous GPUs/CPU/NPU and programmable accelerators **across multiple machines, including trusted remote tunnels**, workload sharing/sharding and a range of topologies: layers and their cache colocated, different model layer groups on different machines, layer compute and its KV/recurrent/cache on different machines, and other contemporary/near-future combinations. Preserve performance, composability, auto-or-pin topology policy and mathematical/state correctness without writing a different runtime for each placement strategy. This requests architectural design/spec updates, not code/cluster deployment or permission to connect remote peers. |
| U-021 | USER | **Build a native multi-model model router into zerogravity**: resolve and load balance requests for as many models as needed, support separately configured model parameters/profiles and simultaneous serving where physical memory allows, with one OpenAI-compatible HTTP API interface. Preserve an **extremely simple one-command local run/serve experience** comparable to easy local model runtimes, while keeping every aspect of model/quant/placement/replicas/context/sampling/routing/API/device/network resource behavior configurable on demand. This is a **design requirement**, not implemented CLI/API or permission to download models/open network services. |
| U-022 | USER | Perform a **final comprehensive editorial and technical refinement** of zerogravity's full architecture/design specifications, organize them for clear authority and maintainability, prepare a **publication-quality technical whitepaper**, and deliver a **complete implementation handoff for LLMs and autonomous agents** to build and validate the system without reconstructing the chat. Retain all historical decisions/research/phase IDs and be honest about design-only status, prior art and unmeasured performance. No actual runtime, PDF, Git, model download or machine changes were expressly requested. |
| U-023 | USER | **Implement only zerogravity's project/bootstrap packaging foundation**, adapting the existing `/home/pvl/spaces/abird/src/dotlink` Rust/crane release/quality conventions as a **read-only reference**: a real minimal binary and project-defining `README.md`, Rust build/format/lint/test tooling, pinned Nix/crane flake and NixOS module, cross-build definitions for major Linux/Windows/macOS architectures, GitHub Actions build/test/release workflow, checksum-protected install scripts and reproducible release artifacts. **No model inference, tokenizer, serving/router/API, device kernels or distributed execution implementation in this phase.** Actual GitHub remote/release and physical cross-platform builds must be separately verified. |
| U-024 | USER | **Begin Phase 1 implementation now**, transitioning from packaging-only to the first real Rust model-source/reference foundation and ultimately running both real Qwen3.8-27B and Gemma 4 26B-A4B-it through one runtime. The user has **not** separately authorized a multi-gigabyte model download, device firmware or privilege changes, Git publish/release or remote peer enrollment. This supersedes U-023's *temporary implementation scope*, not its validated packaging infrastructure. Phase 1 partial implementation is recorded in the historical Phase 1 source/reference handoff. |

| U-025 | USER | **Reordered practical acceptance milestones**: first Qwen3.5 **0.8B** and **2B**, each CPU text inference first, then complete native **NVIDIA CUDA**, native **AMD ROCm/HIP**, and **Vulkan** on both physical GPU families; only after those, **Gemma 4 E2B**, **Gemma 4 26B-A4B**, and **Qwen3.8 27B** should receive source-correct **CPU execution first**, with their GPU work deferred. Preserve a shared Rust-native semantic execution engine and reference/fallback semantics; a physical specialized kernel proof does **not** qualify full-model GPU inference. See the Qwen3.5 CPU/GPU handoff for exact delivered partial implementation and open gates. |
| U-026 | USER | **Complete rigorous CPU numerical, prefill/state and genuine long-context qualification for Qwen3.5 0.8B and 2B before moving to full-model NVIDIA CUDA / AMD ROCm / Vulkan integration.** The new verification must distinguish actual model CPU token processing from synthetic 32K operator/position tests and independently trace model/reference mismatches, failing closed rather than widening thresholds to hide differences. Current status: substantial added tests and guarded chunked prefill, but the CPU gate is **PARTIAL/BLOCKED** on 2B strict distribution drift, missing per-layer reference, and 4K..32K real-model evidence. See the Qwen3.5 CPU numerics/context handoff. |
| U-027 | USER | **Move on to native GPU development once Qwen3.5 0.8B and 2B have verified real CPU semantics through 8K.** Defer 16K and 32K whole-model source/reference qualification to later repeatable local tests, rather than treating them as prerequisites for CUDA, ROCm and Vulkan integration. Keep all independent short numerical, layer, state, error and 8K context acceptance checks and thresholds intact; the deferred longer tests remain UNQUALIFIED. This supersedes only the 16K/32K-before-GPU ordering implied by U-026. Two explicit CPU scopes now exist: Phase 1 GPU-development admission (16/16 PASS) and deferred extended-context (16/20; four missing). The first native accelerator source-matvec integration seam has passed CPU fallback/failure transaction tests; physical complete GPU model execution is NOT yet qualified. See [Phase 1/P07 handoff](../plans/numerics-and-context/TODO.md). |
| U-028 | USER | Commit Qwen3.5 CPU Phase 1 and accelerate P07 GPU implementation while independently examining high 8K shared top-20 logprob errors (~0.2037/0.1968). Seek source-grounded improvements without widening numeric thresholds. Baseline initial phase commit is a965737; GPU physically qualified inference remains pending. See [8K numerical drift report](../plans/numerics-and-context/TODO.md). |
| U-029 | USER | Organize three parallel workstreams: GPU implementation, accuracy investigation, and multimodal architecture design with independent substreams/subagents where tools permit. Keep model design composable, native Rust and accurately source-grounded. Standalone Codex/Claude agents were discovered on host but authentication failed (Codex HTTP 401, Claude not logged in), so NO independent autonomous agent was claimed to have run. See [multimodal implementation checklist](../plans/multimodal/TODO.md). |
| U-030 | USER | Continue into the next implementation phase, preserving three parallel workstreams. Deliver actual native Rust/CubeCL source-quantized GPU regions beyond Q4_0, improve independent numerical understanding without relaxing parity thresholds, and advance additive multimodal architecture via non-disruptive mock contract tests. P07-B source-mixed checkpoint host parity and five MM0 checks were implemented; physical GPU devices and LLVM23 CUDA/HIP toolchain remain prerequisites for real backend validation. See [P07-B/MM0/numerics handoff](../plans/accepted-baseline/TODO.md). |
| U-031 | USER | Continue into P07-C GPU development: source-identity-safe persistent device weight residency, bounded physical buffer leases, strict source-only fallback and completion/event fencing, reducing repeated per-token uploads without turning CPU-only Zerogravity into a GPU-dependent binary. Preserve the 16/16 CPU Phase1 gate, defer 16K/32K and do not label unverified physical GPU execution as passed. See [P07-C exact implementation handoff](../plans/accepted-baseline/TODO.md). |
| U-032 | USER | Continue P07-D GPU development from P07-C: eliminate repeated host GGUF source-weight decoding on GPU cache hits, establish typed asynchronous device lease/event lifecycle and fail-closed completion/cancellation/epoch rules without perturbing CPU source math, and preserve clear separation between host tests and actual GPU hardware qualification. A decode-once StaticSource / DynamicStep split was implemented; device-independent DeviceEventLedger tests and MM-06 duplex/mock cancellation validation pass. Physical asynchronous CubeCL GPU integration remains later work. See [P07-D handoff](../plans/accepted-baseline/TODO.md). |
| U-033 | USER | Continue to P07-E from P07-D: integrate a real async CubeCL readback with the typed DeviceEventLedger, keep original synchronous CPU model semantics and sole TaskCoordinator StateRoot writer, and preserve explicit hardware qualification evidence. A source-typed async completion bridge and fail-closed future cancellation were implemented and host-tested; multi-device GPU concurrency/full-state execution remain later work. See [P07-E implementation handoff](../plans/accepted-baseline/TODO.md). |
| U-034 | USER | Recover interrupted P07-E state, validate the live committed implementation and continue the next GPU safety phase. P07-F retains the actual CubeCL source, dynamic and output handles on an uncertain/dropped Future in a process-wide service-quarantined keepalive registry, and accounts physical resident GPU weights plus activation/scratch/output under one per-provider budget. All CPU trained source and Phase1 8K acceptance remain unchanged; actual native device cancellation and global multi-model ResourceArbiter remain unqualified. See [P07-F handoff](../plans/accepted-baseline/TODO.md). |
| U-035 | USER | Revalidate and continue all Zerogravity work after the interrupted P07-E continuation, then author a complete comprehensive handoff markdown retaining the original idea/design, every phase decision/specification/implementation and pending item going back to project start, with a ready-to-run takeover prompt. Live baseline was already P07-F 602c246, not the old interrupted P07-E draft. P07-G1 adds source-neutral global device admission and bounded fair queue tests (CPU control plane only); physical GPU worker integration remains pending. Next agent must read the latest comprehensive handoff progressively in full before implementing. See [P07-G source scope](../plans/accepted-baseline/TODO.md). |

| U-036 | USER | Take over the live source-controlled Zerogravity repository at documented P07-G1 handoff commit `f28dc10`; revalidate the complete historical source/quality provenance and continue **P07-G2** binding attested physical CubeCL service/adapter identities to the single shared `ResourceArbiter` and P07-F/E residency, events, buffer keepalives and fail-closed cancellation. Preserve independently owned GDN accuracy and multimodal streams, canonical Qwen-first model order, 8K Phase1 acceptance and 16K/32K deferral. Do not claim hardware inference from compilation/mocks; keep concurrent agents' files, avoid bulk staging/stashing, prohibit unauthorized downloads/driver changes/push/releases. This implementation slice delivers conservative P07-G2a global physical envelopes and source-readback guards; actual cross-model fair GPU workers and full physical CUDA/ROCm/Vulkan model validation are still pending. See [P07-G2 phase handoff](../plans/accepted-baseline/TODO.md). |

| U-037 | USER | The user confirmed the `dotlink-l5` connector should now expose GPU hardware and requested immediate continuation to the next Zerogravity implementation phase, then explicitly requested revalidation/resumption after interruption. Revalidated clean `7e6e4f4` G2a baseline, real `/dev/nvidia*`, DRM render and KFD devices; physically exercised native CubeCL Vulkan on NVIDIA RTX 3060 and AMD Renoir, implementing P07-G2b shared per-physical-card bounded fair workers and strict original Qwen GGUF numerical/StateRoot gates. NVIDIA Vulkan source-region workers and partial-model Qwen0.8B/2B are bit-exact; AMD RADV physical readback is real but source FP32 arithmetic differs and therefore remains **numerically unqualified**, with failed Qwen StateRoot publication refused. CUDA/HIP LLVM23 development toolchain, full GPU GDN/KV/attention and physical cancellation remain pending; 16K/32K CPU contexts deferred. The verified phase evidence/limitations are in [P07-G2b](../plans/accepted-baseline/TODO.md). |

| U-038 | USER | After successful clean P07-G2b GPU worker commit `c3e5fca`, user requested **next implementation phase**. P07-G3a continued with REAL NVIDIA and AMD GPU access, resolving AMD Renoir RADV's observed original Qwen3.5 Q4_0×Q8_0 native-F32 fused-accumulation mismatch by a separate conservative WideF64 GPU compile-time specialization. BOTH physical cards now pass exact original GGUF source-bit four-dispatch 0.8B/2B GPU worker checks; AMD four-token real CPU+one-GPU Qwen hybrid complete-vocab logits/StateRoot identical to pure CPU. Original fast NativeF32 remains separately diagnosable and still FAILS strict source equality on AMD (PASS NVIDIA). Explicit opt-in feature-only physical submitted-Future-abort probe shows both GPUs retain five original buffers in process-lifetime quarantine, conservatively fence/retain 256MiB logical GPU provider charge, reject second model, and publish no stale StateRoot. Do **not** call this proof of driver cancellation/recovery, measured allocator VRAM, fast GPU inference or complete Qwen GPU graph. Native CUDA/HIP compilation needs unavailable LLVM23 and vendor SDK; CPU Phase1 16/16, extended 16/20 deferral, GDN long-context accuracy and MM0 v5 independent streams unchanged. See [P07-G3a physical evidence](../plans/accepted-baseline/TODO.md). |

| U-039 | USER | After `67dca72` G3a commit, requested a **fresh adversarial review** of bugs, correctness, regressions and a robust, simple, intuitive, future-proof, low-maintenance and composable software architecture. Review verified current live `main` and intentionally implemented bounded fixes instead of a wholesale rewrite: pre-submit GPU validation before physical terminal fencing; scratch-aware LRU of idle source weights; same-file-descriptor GGUF source loading/reparse (prevent equal-length pathname replacement source mix); normal CPU intermediate chunk prefill elides discarded logits with real 0.8B/2B parity; correct GPU fair-queue `selected` vs physically `completed` counters and renamed globally destructive card-epoch API; checked/limited untrusted model head geometry; and non-allocating per-card global fence hot path. Tests covered actual NVIDIA/AMD Vulkan positive dual-model work, actual invalid activation, invalid original FP16 scale, underbudget rejection and recovery, real Future-abort quarantine; preserve independent CPU GDN 8K/16K/32K and MM0 streams. The detailed prioritized findings, changes, test evidence, remaining production blockers and smallest safe next steps are in [fresh quality review](../plans/accepted-baseline/TODO.md). No unapproved drivers, model downloads, remote release or Git push. |

| U-040 | USER | After the U-039 fresh correctness and architecture review, requested **resolving remaining issues** with clean, intuitive, scalable, future-proof, maintenance-light, composable Rust design. Continued live repo work on clean `e9312ad` baseline: introduced one immutable `Arc<LoadedQwenModel>` for original GGUF+predecoded layer constants shared across independent Qwen session StateRoots, backward-compatible direct load, minimal explicit `ModelCatalog` alias/lazy per-source singleflight load/retirement; real concurrent source-pinned Qwen0.8B/2B sessions and original GPU shared-resident source reuse verified. Added source-audited physical F32/WideF64 Q4 region latency probes and device-wide GPU VRAM/GTT measurements keyed by attested PCI BDF, which revealed driver memory pools remain reported allocated after logical provider release. Offline native CUDA and HIP compiles still block at LLVM23 `LLVM_SYS_231_PREFIX`, independent CPU GDN 8K arithmetic cause remains open, full GPU model and native HTTP/multi-model router **not claimed complete**. Preserve original models, source/tolerances, MM0 and explicit 16K/32K deferral. See [U-040 implementation evidence and remaining honest gates](../plans/accepted-baseline/TODO.md). |

| U-041 | USER | Absorb the newly committed 15 Oct v5.1 universal multimodal, source specialization, joint serving and virtual SSD-tensor-residency design proposals, then continue real GPU execution in verified phases; remain one Rust source-verified engine, one TaskCoordinator/StateRoot publisher and one physical resource authority. P07-H1 physically source-qualified all 187 Qwen3.5 0.8B/2B GGUF GEMV tensor regions and full-vocab/StateRoot on AMD Renoir and NVIDIA RTX3060 Vulkan; P07-H2 added actual GPU source-compatible RMSNorm with exact original f32-square, f64 sum, f32 mean/sqrt/reciprocal rounding, yielding 187 GPU matrix + 49 GPU norm regions = 236 source-bit-equivalent operations on EACH checkpoint/device combination. The rest of GDN/attention/KV/recurrent and complete device graph are not yet implemented/qualified. [H2 source/hardware evidence](../plans/accepted-baseline/TODO.md). |
| U-042 | USER | Explicitly granted FULL Zerogravity repository ownership, removing the prior constraint to preserve or partition any other agent changes. Revalidate concurrent Git before editing, reconcile all v5.1 docs and code into one coherent implementation, clean up as appropriate, update current handoffs and roadmap, progress in safe validated GPU stages, and do not fabricate full-GPU/native CUDA/ROCm/server or source-accuracy claims. Preserve original source model checkpoints/quality gates and do not push or publish releases without authorization. |

| U-043 | USER | The user explicitly superseded any prior assistant-authored README copy edits: the live README is USER-OWNED and MUST be preserved exactly as the user updated it. Do not revise, replace, clean, stage or bundle README.md into a GPU phase commit. Documentation snapshots may reference its current bytes, but must not mutate README. Continue independently verified P07-H phases. |


| U-044 | USER | Requested validation and continuation after P07-H3 grouped per-head RMS normalization and interrupted two-token device-source acceptance. Revalidated Git `4ab15a4`, original Qwen3.5 0.8B/2B GGUF SHA hashes and AMD Renoir/NVIDIA RTX3060 Vulkan access. The initially failing two-token acceptance MAP was counting final-only vocabulary and output-RMS operations twice; fixed count without relaxing source math, GPU correctness or state acceptance. On BOTH checkpoints and BOTH physical Vulkan GPUs, the original Rust CPU Qwen model exactly matches **530 REAL physical GPU completed regions across two tokens** (266 selected: 187 source GEMV, 49 residual/final RMSNorm, 30 independently grouped GDN/attention head norms; subtract 2 final-only operations), all output bits, complete 248,320-vocabulary logits, committed root and state fingerprint, with ZERO outstanding global logical reservation after real readback. Physically forced post-launch grouped-RMS Future abandonment separately passed on BOTH cards: 5 original GPU handles quarantined, 256MiB retained/fenced and root unchanged; not driver cancellation/recovery. User-owned README MUST remain exactly as supplied SHA256 `736c794cf68d82cf47b20fc8409e4876ab107888a38f98805bdedbde10d2aaf8`. The actual GDN convolution/recurrent, attention/softmax/RoPE/KV, FFN SiLU and device-resident inference graph are NOT complete; do not invent entire-GPU qualification. See [H3 real hardware evidence and H4 continuation](../plans/accepted-baseline/TODO.md). |

| U-045 | USER | Continued physically validated GPU execution after P07-H3. Implemented P07-H4 first original-source **GDN causal convolution island** through an additive typed kernel returning BOTH the raw pre-SiLU activation and private proposed 3-frame history; the one CPU QwenState still owns the commit. One-kernel GPU output failed source-bit comparison on the SECOND original Qwen token (1,840/6,144 raw discrepancies, history exact); split source FP32 multiplication rounding into actual stored device products and separately ordered reduction (two real kernels). Both original Qwen3.5 0.8B/2B checkpoints on BOTH AMD Renoir RADV and NVIDIA RTX3060 Vulkan then passed **566 genuine physical GPU region completions over two tokens**, including 36 conv raw/next-history source bit checks, 187 matrices/49 residual norms/30 grouped norms per token, exact full vocab logits and StateRoot, and zero remaining provider logical charge. Real conv Future abandonment on both GPUs retained 6 original handles, fenced 256MiB charge, rejected another model and preserved Qwen root. This does **NOT** implement device-resident recurrent GDN matrix, attention/KV/FFN or full-GPU Qwen; native CUDA/HIP LLVM23 still blocked; independent CPU 8K GDN drift remains open; original 16K/32K deferred. User-owned README preserved bit-for-bit. [Detailed H4 physical results and H5 direction](../plans/accepted-baseline/TODO.md). |

| U-046 | USER | Requested a **FULL comprehensive agent takeover handoff Markdown** covering the original Zerogravity vision and whitepaper, every specification/design iteration, all decisions/proposals/negative experiments, every implementation/test/commit phase since beginning, incomplete work and exact instructions plus ready-to-paste next-agent prompt. Keep everything in `.agents/docs/` or `.agents/plans/` and do not skip historical sources; retain the user-owned README verbatim. Delivered a new dated, practical complete takeover guide [plan TODO index](../plans/TODO.md) and regenerated the canonical **byte-for-byte lossless historical archive** [FULL_COMPREHENSIVE_IMPLEMENTATION_HANDOFF_2026-10-14.md](../plans/TODO.md), which embeds the dated guide and every earlier versioned specification/decision/phase source with SHA/byte counts. H4 `0cad60a` remains last *implementation* phase; upcoming H5 GPU GDN recurrent state is implementation work, not completed by this documentation task. No unapproved downloads, code alterations, pushes, tags/releases or README changes. |

| U-047 | USER | Requested immediate continuation of the live repository from handoff2125612/H4 through **P07-H5 source-exact GPU GDN recurrence**, after complete takeover-guide reading and live repository/model/hardware/test revalidation. Preserve one Rust engine, immutable source identity, independent StateRoots, physical arbiter/fair workers and actual completion; compare output AND proposed state bits on trained nonzero causal tokens across both checkpoints and both Vulkan vendors, with rollback, admission and post-launch handle quarantine. Continue later attention/KV/FFN/resident/vendor/model milestones in small independently verified phases. Root README remains user-owned; no model/SDK downloads, privilege changes, push/tag/release. [H5 source and acceptance ledger](../plans/accepted-baseline/TODO.md); continued under the same authorization with [H6 Q/K L2 source/rollback/physical acceptance](../plans/accepted-baseline/TODO.md). |

| U-048 | USER | Requested architecture/design validation of native kernels, speculative execution, retained replay and heterogeneous prefill, including relevant task/product capabilities and measured performance objectives. Preserve the one-Rust-engine/source/StateRoot/resource boundaries. This is design qualification, not authorization for downloads, machine tuning or an unmeasured speed claim. [Architecture and planned gates](../research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md); [pipeline contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md). |

| U-049 | USER | Requested removing external implementation/release references from this design update while retaining its technical ideas and design constructs. Generalize document names, text, links and acceptance identifiers; remove imported comparison/provenance/benchmark narrative; regenerate the archive and SHA manifest. Runtime and root README remain unchanged. |

### Strongly recommended architectural choices (not independently approved yet)

| ID | Status | Proposed choice and why |
|---|---|---|
| P-001 | PROPOSAL | Three capability levels: portable semantic reference; tuned generic kernels; extreme model-family/hardware specialization. Graceful per-operation fallback with state-layout compatibility. |
| P-002 | PROPOSAL | Three compiler layers: Semantic IR (exact/effectful model math), Optimized Execution IR (placement/fusion/layout/lifetimes), Device Plan (kernels/events/capture/scratch). |
| P-003 | PROPOSAL | Semantic IR represents recurrent and sparse state and its mutation/rollback as **effects**, not just pure tensor DAG. |
| P-004 | PROPOSAL | Rust-owned kernel registry and compiler selection with CubeCL as candidate, but keep independent CPU/Rust reference and allow target-specific Rust GPU intrinsic/backend work if CubeCL codegen is inadequate. |
| P-005 | PROPOSAL | Rust-authoritative scheduling with mixed prefill/decode/verify batches, bounded admission, latency/throughput modes, no CPU routing round trips in resident MoE hot path. |
| P-006 | PROPOSAL | One global budget arbiter, many specialized caches: dense/experts/KV/recurrent/indexer/lookup/scratch; zero assumption that GPU-visible GTT is truly available RAM. |
| P-007 | PROPOSAL | GPU route-aware MoE grouping across sessions and speculative rows, overlap-safe expert slot management, heterogeneous placement, no universal cache policy. |
| P-008 | PROPOSAL | Persistent, versioned prefix-state snapshots including KV, GDN/KDA, indexer, n-gram history, multimodal positions, speculative frontier; COW where beneficial. |
| P-009 | PROPOSAL | Speculation algorithms behind an exact verifier: native MTP, DSpark where model-compatible, EAGLE/draft model, prompt-lookup, n-gram and disabled; adaptive policy based on accepted **committed** tokens per full cost. |
| P-010 | PROPOSAL | Strict default sampling distribution preservation when available; explicit opt-in for approximate/opportunistic sampling. Quantization changes never implicit. |
| P-011 | PROPOSAL | GGUF and Hugging Face safetensors readers, semantic model adapters, tokenizer/chat-template correctness; recognized metadata != runnable unknown architecture. |
| P-012 | PROPOSAL | Multiple device execution strategies: single-device, hetero CPU/GPU, pipeline, tensor parallel, expert parallel, prefill/decode disaggregation where economics justify. |
| P-013 | PROPOSAL | Rust core and kernel source; use platform GPU drivers, APIs and vendor math facilities via safe Rust wrappers when needed; avoid linking another *inference runtime* as core. |
| P-014 | PROPOSAL | Test-driven optimization with CPU/FP64 oracles, reference logits, checkpoint parity, actual device benchmarks, quality regression, controlled ABBA/BAAB tests; publish failed experiments. |
| P-015 | PROPOSAL | Keep front-end as single binary/SDK with OpenAI/Anthropic-style APIs, streaming, tools, reasoning semantics, observability and simple local CLI. |
| P-016 | PROPOSAL | Prefer clear modular Cargo workspace, no performance-sensitive per-token dynamic-dispatch overhead, few crates, minimal operational deps. |
| P-017 | PROPOSAL | A plan cache keyed by IR/kernel/driver/device/quant/shape/numeric policy; invalidated on relevant changes; no invasive speculative adaptation until measurement proves value. |
| P-018 | WORKING TECHNICAL BASELINE | Reconcile the two design programs with **SIR** state/math, **RIR** legal fusion/layout, independent **KIR** Rust kernel provider branch, **CIR** asynchronous resource/stream/IO command graph, validated immutable ExecutableBundle; HardwarePolicy and StepPlan distinct from KernelPlan. This is a design decision from comparison/research, not a new user-specific API approval. |
| P-019 | WORKING TECHNICAL BASELINE | Adopt research's EngineCore single logical writer, bounded EngineMsg/EngineEvent, persistent InputBatch and StepPlan/WorkerDelta, CPU/IO worker isolation and typed mode receipt, validated against effectful model-state transactions. |
| P-020 | WORKING TECHNICAL BASELINE | Implement generic paged KvPool + chained hashes with optional radix index over same pool; extreme contiguous reserve-all pool only when measured, both under one ResourceArbiter; generalize hybrid prefix sharing to 3+ state-domain groups. |
| P-021 | WORKING TECHNICAL BASELINE | Keep GGUF as core input, optional derived acceleration pack, content-addressed ModelStore, precise tokenizer/pretypes/chat/grammar FSM, encoder/embedding/rerank and LoRA adapter capabilities; scope each release independently. |
| P-022 | WORKING TECHNICAL BASELINE | Rust kernel provider ABI independent of CubeCL; evaluate 2026 independent Rust CUDA tracks, AMD Rust gfx1151 ISA path and CPU/Vulkan/Metal fallback. Vendor driver FFI/non-Rust kernel source policy remains OPEN. |
| P-023 | WORKING TECHNICAL BASELINE | Use first-party independent ML compiler streams, schedule search, the MLIR effect model, attention-kernel serving autotuning and equality-saturation/optimization engines as *architectural research sources*, not required large runtime dependencies. Optimize pure regions offline with bounded compile time; no hot-path search. |
| P-024 | WORKING TECHNICAL BASELINE | Combine research frontier ring/CommitPlan with project full-state causal transaction; semantic, cache schema and async lease/timepoint effects must all remain valid across speculation/fallback. |
| P-025 | WORKING TECHNICAL BASELINE | Deliver a Stage-0 architecture/kernel/format/CPU vertical slice before freezing dozens of crates and advanced API; detailed crosswalk in INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md; no phase done without actual tests. |
| P-026 | WORKING TECHNICAL BASELINE | Treat XDNA2 and future NPUs as capability-probed **compute devices** with external runtime/driver and memory topology, not separate language model runtimes. RIR may partition semantically independent GEMM output rows; KIR selects an NPU GEMM provider; CIR schedules prepack, NPU/GPU concurrent work, event/fence join and full state commit. Missing NPU or unsupported shapes retain GPU-only fallback. |
| P-027 | WORKING TECHNICAL BASELINE | Treat upstream Gufo's distinct ASR/TTS/image/video work as **TaskProgram stage-graph research**: model/task-specific control loops share SIR/KIR/CIR/device memory but encoder, vocoder, denoiser and media outputs cannot all be forced into the decoder-only next-token loop. Maintain separately qualified capabilities. |
| P-028 | WORKING TECHNICAL BASELINE | Because the supplied Gufo fork reports hard GPU/NPU **machine hangs during fabric clock transitions**, its NPU path is experimental/opt-in and GPU-only is the safe default until a **specific hardware/firmware/XRT/ROCm/power configuration** has demonstrated stable concurrency, numerical quality and end-to-end win. Do not automatically modify privileged fabric clocks, IOMMU, NPU suspend or systemd configuration. |
| P-029 | WORKING TECHNICAL BASELINE | Keep **compute owner vs weight residency owner vs state owner** independent: an independent dGPU↔iGPU expert execution design, Gufo-style same-SoC iGPU↔NPU GEMM slicing, and optional CPU/SSD are different CIR placement candidates under one physical ResourceArbiter. |
| P-030 | WORKING TECHNICAL BASELINE | For NPU block-float compute, store source checkpoint precision separately from **temporary device compute encoding**; the source research's nine-byte/eight-value block-float example has quantization error and MUST be validated by per-operator/full-logit and model-quality tests, not merely 30 matching greedy outputs. |
| P-031 | **AUTHORITATIVE V3 DESIGN** | Replace source-project-specific architecture claims with **original zerogravity contracts**: ModelDefinition, SemanticOp/StateDomain, TaskProgram, legal RIR, independent KIR provider, CIR async resource graph, ExecutableBundle, DeviceCapabilities, ResourceArbiter and TaskCoordinator/StepPlan. Prior art documented separately and source-named examples optional only when root/file MIT permits. |
| P-032 | **AUTHORITATIVE V3 DESIGN** | Model semantics and task control are orthogonal; a TaskProgram can compose text, audio ASR, speech synthesis/vocoder, image/video diffusion, encoder/pool/rerank, agent workflows and future compute without a second device runtime or fake decoder-only loop. Each has typed iteration/frontier/output and independent model/task qualification. |
| P-033 | **AUTHORITATIVE V3 DESIGN** | New model/operator/quant/device/kernel/IO/source/task/frontend extensions register narrow versioned capability interfaces with independent reference/fallback/evidence and cannot require changing unrelated API/session/scheduler/memory ownership. Adopt minimal modules and a mandatory architecture extension drill. |
| P-034 | **AUTHORITATIVE V3 DESIGN** | License-aware prior-art catalog owns external project/author links, repository- and per-file SPDX verification, source revision and benchmark provenance. Normative plans use mechanism names; a MIT-licensed reference is a possible source candidate, not a mandatory dependency or blanket permission to copy bundled differently licensed code/weights. |
| P-035 | HISTORICAL V3 BASELINE (REFINED BY V4) | Typed SIR math/effects → legal RIR regions → KIR candidates → CIR async resources → ExecutableBundle runs **directly**, while task control coordinates stages/loops. v4 retains conceptual compiler stages but minimizes persistent interfaces/formats; StateRoot owner/physical budget ownership refined in P-036…P-045. |
| P-036 | **V4 ARCHITECTURAL BASELINE** | Exactly two enduring compilation representations initially: **SemanticProgram** and **ExecutableBundle** (containing async CIR commands). RIR and KIR are transient optimization/provider views; no requirement to implement four serialized IR dialects or hot interpreters. |
| P-037 | **V4 ARCHITECTURAL BASELINE** | StateDomains prepare immutable candidate versions; only **TaskCoordinator per session/shard** performs one logical `PublishRoot` after all state domains and physical events validate. Device/CIR commands report ready/failed, never publish semantic session state. |
| P-038 | **V4 ARCHITECTURAL BASELINE** | Maintain **coarse global physical memory admission/reservation**, with fast device-local slab/page suballocators under generation/event-backed leases; no global per-token allocation lock or impossible promise that the OS cannot change available memory. |
| P-039 | **V4 ARCHITECTURAL BASELINE** | A small trusted built-in `TaskDriver` controls AI stage loops and typed outputs; no general workflow DSL or `ExternalAction` with filesystem/network privilege inside core. External RAG/agent/tool orchestration belongs above the TaskRequest boundary. |
| P-040 | **V4 ARCHITECTURAL BASELINE** | **Async-first device-native StepProgram** with persistent parameter/row buffers, routing/sampler/draft verification and output event transfer where supported, avoiding mandatory CPU↔GPU synchronization for every model step; host-loop reference/fallback remains correct. |
| P-041 | **V4 ARCHITECTURAL BASELINE** | Typed fixed-rank bounded dynamic shapes, ragged row layouts, runtime kernel shape/quant/ISA/precision guards and bounded shape-band/compiled-bundle cache; use portable fallback on unsupported shape rather than unlimited JIT. |
| P-042 | **V4 ARCHITECTURAL BASELINE** | Output after committed root uses sequence/epoch identity, ordered at-most-once **delivery attempt** semantics; optional resumable cursor/ACK/durable store separately scoped. Distributed peers require epoch fencing and prepared cohort; no unqualified exactly-once network or instant multi-GPU physical atomicity claims. |
| P-043 | **V4 ARCHITECTURAL BASELINE** | Preserve a **future optional** `GradientProgram/TrainStep` and mutable `ParameterVersion` semantic/task seam without importing a compulsory autograd/tape/trainer into inference-first core; inference sessions pin immutable checkpoint/adapter versions. |
| P-044 | **V4 ARCHITECTURAL BASELINE** | Start with **one Rust library and one binary** (or 2–4 crates if GPU backend build/separation proven necessary); no mandatory nine-crate design, dynamic plugin framework or generalized external workflow engine. Require measured benefit to add persistent abstractions. |
| P-045 | **V4 ARCHITECTURAL BASELINE** | Specialized entire model/layer closed regions can replace generic math through one provider contract and compatible state/numerical/resource boundaries; disable specialized path and still run portable model/task with no new server/scheduler/allocator. |
| P-046 | **V4.1 BACKEND CONTRACT** | Generalize native `KernelProvider` into one `RegionExecutionProvider` seam supporting NativeKernel, CapturedCommandGraph, AheadOfTimeDeviceGraph and StaticallyScheduledDataflow. The existing KernelProvider is the native implementation subset, **not** another model runtime or an extra IR. |
| P-047 | **V4.1 BACKEND CONTRACT** | Let the provider declare **minimum efficient closed-region granularity**: operator, fused region, whole stage/model or partitioned fabric program. Device compiler owns micro-scheduling while Rust TaskCoordinator retains session task/admission/output/StateRoot, with no host instruction-by-instruction emulation of TPU/wafer/static fabric. |
| P-048 | **V4.1 BACKEND CONTRACT** | StateBridge qualification explicitly requires privately prepared state, checked snapshot/restore, bounded replay or stateless region. An opaque in-place stateful device program without these is **not** an interchangeable general spec/rollback backend; no device-side PublishRoot authority. |
| P-049 | **V4.1 BACKEND CONTRACT** | ProgramLifecycle records compile/build, upload/load, warmup, resident code/scratch, source/driver/firmware, statically constrained shape, cache invalidation, code/weight stream and program teardown; graph-only devices need shape-band fallback and no per-token recompile. |
| P-050 | **V4.1 BACKEND CONTRACT** | Model hardware **compute domains, memory tiers, links, compiler domains and micro-schedule ownership** independently: coherent LPDDR CPU+GPU, discrete HBM/GDDR GPU, multi-GPU rack peer fabric, graph-compiled TPU/NPU, on-chip SRAM/static wafer fabric and authenticated remote transport. Aggregate chip memory never implies common pointer addressability. |
| P-051 | **V4.1 BACKEND CONTRACT** | Separate FP4 variants with 16-value FP8 block scale and two-level tensor scaling vs 32-value shared power-of-two microscale, native FP8/MX and specialized low-bit layouts; source stored quant vs physical repack vs lossy compute/changed model quality remain distinct with full-task tests. |
| P-052 | **V4.1 BACKEND CONTRACT** | A managed remote inference API without exact compiler lowering and StateBridge is **TaskServiceAdapter interoperability**, not a native RegionExecutionProvider; explicit auth/privacy/egress/session limitation and no claim of native device execution. |
| P-053 | **V4.1 BACKEND CONTRACT** | Report qualification ladder per model/task/shape/driver: architecture compatible → SDK feasible → compiles → functional → state-safe → performance-qualified → production-qualified. Missing SDK/physical hardware/source-policy yields BLOCKED/INCOMPATIBLE, not a phantom supported TPU/wafer product. |
| P-054 | **V4.1 RESEARCH** | Distinguish Cerebras wafer-scale accelerator from an independently published Cerberus parallel-decoding algorithm; the latter is an optional DraftSource/TargetVerifier and full-state rollback candidate, not a new hardware backend. User referent remains unconfirmed, both are researched. |
| P-055 | **V4.1 MINIMALITY** | Add no permanent new semantic IR, task runtime or model-specific server. Use one RegionExecutionProvider and one ExecutableBundle physical program enum; optional closed region graph compiler and statically scheduled fabric tests cannot block real CPU/GPU vertical slice. |
| P-056 | **V4.2 DISTRIBUTED BASELINE** | One **PlacementPlan** independently binds ComputePlacement, ParameterPlacement, mutable StatePlacement, immutable CachePlacement, and derived TransferPlan across topology, TaskDriver phases and committed roots; placement is a *view inside ExecutableBundle*, not another enduring IR or model engine. |
| P-057 | **V4.2 DISTRIBUTED BASELINE** | Support colocated baseline, contiguous/noncontiguous layer groups with per-layer cache colocated, remote layer compute with state elsewhere via hydrate/remote-attention/direct-read variants, tensor/head/expert/sharded-sequence, P/D, FFN/attention disaggregation, remote prefix and multi-task stage placements; every mode per-op/source/math legal and costed. |
| P-058 | **V4.2 DISTRIBUTED BASELINE** | Node/LinkTopology capability includes compute/memory/storage/driver, network **effective directional bandwidth, RTT/P95 jitter, direct vs relay, MTU/loss, auth/crypto/egress**, plus physical budget and failure domain; runtime placement epoch frozen for active compiled region/collective. |
| P-059 | **V4.2 DISTRIBUTED BASELINE** | **TransportProvider** supports local IPC/peer/DMA, physically qualified RDMA, authenticated TCP/QUIC or an existing private encrypted tunnel; control and bulk data planes may choose separate paths. VPN/tunnel is not RDMA, a relay is not direct, and private network routing is not peer authorization. |
| P-060 | **V4.2 DISTRIBUTED BASELINE** | Minimal opt-in authenticated **PeerAgent** reuses the same binary/RegionExecutionProvider/DeviceBackend, advertising allowed model/work/resources and executing closed region/state transfers. No second server, arbitrary shell, network public exposure or automatic export of private prompts, activations, KV or model weights. |
| P-061 | **V4.2 DISTRIBUTED BASELINE** | Each mutable remote per-session/layer state shard has owner/lease/generation/schema/placement epoch references in the **single TaskCoordinator-owned StateRoot**; peers only prepare/ready/fail and are fenced on lost cohort/epoch. Immutable prefix cache locations may be eventually discoverable hints and never substitute for live state authority. |
| P-062 | **V4.2 DISTRIBUTED BASELINE** | Replan/migrate live computation/state only at legal committed TaskFrontier or op-defined checkpoint, with destination reservation, compatible verified snapshot transfer, epoch fence/lease move and old-lease retirement; failure preserves old root or replays safe checkpoint. No automatic transparent failover or exactly-once WAN output without explicit durable protocol. |
| P-063 | **V4.2 DISTRIBUTED BASELINE** | **Remote attention next to its KV/state** and independent FFN/experts are distinct from remote raw KV byte reads per token; choose based on measured network critical path and StateBridge. Slow/relayed tunnels normally veto chatty A/FFN or full remote KV reads, favor contiguous layer or one-time cache hydration. |
| P-064 | **V4.2 DISTRIBUTED BASELINE** | Optional **token-range attention shard** computes local softmax summaries (max, normalization, weighted values) on state-owning nodes and composes stable result; requires operator-specific PartitionAlgebra proving positions/masks/global routing/numerics. Ordinary recurrence/sparse global indexer not presumed partitionable. |
| P-065 | **V4.2 DISTRIBUTED BASELINE** | Layer pipeline colocates its layer-local mutable caches by default, reducing network to activation handoffs. Decode C1 serial hops can be slower; multi-chunk prefill/throughput and model capacity may benefit. Distinguish capacity benefit from single-request t/s gain. |
| P-066 | **V4.2 DISTRIBUTED BASELINE** | Separate control/metadata, bulk data and telemetry; scoped node identity, per-tenant/model/weight authorization, checked stream/transfer length, registered memory lease and content integrity, transport priority/backpressure, no arbitrary remote executable shell or untrusted device program. |
| P-067 | **V4.2 DISTRIBUTED BASELINE** | Dynamic cluster membership/topology and state-owner **lease epochs are not the same**; no split brain and old coordinator writes after failed election; start fail-stop/manual peer recovery, defer durable quorum/consensus/autoscaler until product requirement. |
| P-068 | **V4.2 DISTRIBUTED BASELINE** | Joint placement cost based on real compute/shape/phase, model+state bytes, activation/collective/query payload, link P95/copy/crypto/transfer overlap, cold program/cache, failover, security and objective. Do **not** activate a remote machine merely because it has spare memory. |
| P-069 | **V4.2 MINIMALITY** | Standalone has **zero peer/cluster overhead** and no required network permission. Gradual proof from synthetic remote layer, 2 private peers with reliable transport, remote full-state cache, A/FFN/token shards on fast links; cluster DB, DHT and public super-node services not mandatory. |
| P-070 | **V4.2 VALIDATION** | Add optional S0-DIST-A…L and H56…H70 distributed fault/cost/network/topology experiments, preserving P00–P23; source implementation, model/quant, physical network direct/relay, causal state, quality, P95 and privacy required before functional/performance claims. |
| P-071 | **V4.3 ROUTER BASELINE** | **One built-in ModelRouter** sharing ModelCatalog/ArtifactStore, PhysicalResourceArbiter, TaskCoordinator, RegionExecutionProvider/DeviceWorkers with CLI/SDK/HTTP; no mandatory gateway proxy, second inference scheduler or independent model server for each checkpoint. Exact model choice happens on request admission, not on every token. |
| P-072 | **V4.3 MODEL IDENTITY** | Separate `RegisteredModel` (immutable artifact), versioned `ModelAlias/ModelProfile`, `ResidentModel` deployment/physical replica (may span nodes), and `TaskSession` full mutable StateRoot. No conflation of many catalog models, simultaneously resident physical models, same-model replicas or sharded one model. |
| P-073 | **V4.3 EXACT ROUTING** | Default model alias resolves an exact checkpoint/stored quant/tokenizer/adapter/profile revision and compatible TaskDriver/numeric contract. **No silent substitution of another family, quant or model** when overloaded; explicit opt-in policy aliases may select different models and must reveal actual source/model and fidelity. |
| P-074 | **V4.3 REPLICA ROUTING** | Same-model replica load balancing uses verified state/quant compatibility, sticky session/prefix locality, physical headroom, queue/P95 and tenant/peer data permissions. Stateful migration requires safe StateRoot snapshot/PlacementEpoch transfer at committed frontier, not random per-request model switch. |
| P-075 | **V4.3 RESIDENCY** | On-demand model load/verify/compile/warm and safe idle unload, keepalive/pin/preload/scale-to-zero as optional policies. Deduplicate concurrent loads and bound queue. Unlimited registered models are allowed; simultaneous resident models/replicas/contexts bounded by physical RAM/VRAM/OS reserve and distributed capacity, no automatic quant degradation. |
| P-076 | **V4.3 SHARED BUDGET** | Model-level coarse admission and resident eviction reuse PhysicalResourceArbiter/local pools; within-model native continuous batching stays in existing TaskCoordinator and DeviceWorkers. Router must not become per-token bottleneck or starve loaded interactive models during cold loading. |
| P-077 | **V4.3 PER-MODEL PROFILES** | Versioned typed configuration at global, model, alias, deployment, session and request scope, with deterministic effective-source precedence; immutable source/model semantics, hard security and resource limits always constrain overrides. Hot alias revision affects only future requests; active TaskSession remains pinned. |
| P-078 | **V4.3 SIMPLE UX** | Proposed one executable `zerogravity run <model>` / `zerogravity serve` with zero mandatory daemon/config/cloud account: local in-process run, loopback serving, safe auto-device/model plan, concise progress; `models list/ps/pull/inspect` and `config show --effective` provide progressive disclosure. Exact binary name/port subject to open policy. |
| P-079 | **V4.3 SOURCE POLICY** | Unknown/unavailable named source must return diagnostics/offer explicit import/pull, **never silently download a guessed model**. Explicit source/license/size consent and content-addressed atomic verification, no executable model repository code, no public remote peer use without authorization. |
| P-080 | **V4.3 OPENAI API** | One loopback-native API adapter supports a pinned, tested subset of `GET /v1/models`, `POST /v1/chat/completions`, embeddings once encoder qualified, optional completions and versioned Responses. Streaming/SSE/event/tool-call data, usage, stop and error semantics validated with actual SDKs; no blanket OpenAI feature parity promise. |
| P-081 | **V4.3 RESPONSES/PRIVACY** | Stateless Responses first; stateful `previous_response_id`, local `store`, tool execution or cloud built-ins only when separately implemented/authorized. Tool-call *generation* is output data; arbitrary tool execution is outside inference core. Conversation storage privacy-first and opt-in. |
| P-082 | **V4.3 API SECURITY** | Loopback-only default and native admin/control API segregated from public inference API. External bind requires explicit auth/TLS and tenant/model visibility, remote source/peer policy; no API field can bypass hard physical/numeric/tenant limits or fetch arbitrary media URL/local files. |
| P-083 | **V4.3 MULTI-MODEL COMPOSITION** | A TaskProgram may reference **multiple distinct model roles** (target/draft, encoder/projector, embedding/rerank, TTS/vocoder) with separate artifact identities but one coordinated resource/task budget; these aren't replicas or alternate target routing. |
| P-084 | **V4.3 CONFIG CONTROL** | Expose every valid override through discoverable typed model/global/request/profile controls and `inspect --explain-plan/config show --effective` with value/source/guard/fallback. Config revisions/driver/source changes trigger compatible drain/replan rather than unsafe mutation of live KV or model weights. |
| P-085 | **V4.3 VALIDATION** | Add optional S0-SERVE-A…J and H71…H86 tests for one-command local UX, multi-model load/eviction, equivalent-replica routing, exact alias/session consistency, memory pressure, config precedence, OpenAI Python/JS SDK and SSE, auth/privacy and distributed resource placement. None run. |
| P-086 | **V4.3 MINIMALITY** | `run`, `serve` and Rust SDK use **same library engine**. Router owns model request resolution/admission and does not re-tokenize, JIT, marshal or run a second token scheduler per output step. A separate process may later be a security fault-domain option, not the mandatory architecture. |
| P-087 | **PUBLICATION/DOC AUTHORITY** | Provide a public `docs/WHITEPAPER.md` (problem, proposed original architecture, related work, research limitations, falsifiable hypotheses) separate from an integrated normative `docs/ENGINEERING_SPECIFICATION.md` (ES-00…ES-15 exact owners, protocols, tests). The external research/engine names remain properly attributed only as prior art. |
| P-088 | **PUBLICATION/DOC AUTHORITY** | `docs/README.md` is the organized audience-specific document index; `.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md` is the concise agent entrypoint; latest user decisions and live verified code/tests supersede historical addenda, whose full records must be retained. Do not silently rewrite older source history to remove evidence. |
| P-089 | **AGENT DELIVERY** | Maintain `.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md` with bounded P00–P23 tickets, exact Phase0/negative-test and hardware/source/privacy ownership, handoff template, and `REQUIREMENTS_AND_TEST_TRACEABILITY.md` mapping user U requirements to ES/INV/phase/proof without invented test successes. |
| P-090 | **PUBLICATION SAFETY** | Whitepaper is a **proposed design**, not a benchmark-results paper or proof of universal hardware/model support. Provide numbered primary source citations and publication checklist, with unresolved byline, repository/document license, DOI and independent peer review required before final public distribution. No PDF/export or results artifacts promised without separate request. |
| P-091 | **FINAL REVIEW PROCESS** | Run structural Markdown/link/anchor, core source-name neutrality, stable U/P/O IDs, P00–P23 and H01–H86 plus Stage0 coverage, and exact SHA-256 manifest **regenerated last**. Any failure must be fixed rather than marked passing; no source/model/device/remote operations occur as part of design-only review. |
| P-092 | **PACKAGING BOOTSTRAP** | Use **one dependency-free Rust 2024 binary** pinned to available Rust 1.95.0, with `--version`, `--help`, and honest nonzero refusal for reserved `run/serve/models/inspect/config` until implementation. Never supply a fake HTTP service or model output merely to make packaging pass. |
| P-093 | **PINNED CRANE BUILD GRAPH** | Reuse the trusted *pattern* of pinned nixpkgs/crane/rust-overlay flake, separate dependency/build/test/clippy/fmt outputs, NixOS module and six Linux MUSL/Windows GNU+LLVM-MinGW/macOS Apple SDK cross target definitions. An unexecuted Nix cross derivation is not a verified platform build. |
| P-094 | **RELEASE AND INSTALLER BOUNDARY** | GitHub Actions checks/builds six assets, validates SHA-256 and exact Cargo version, creates a release only on matching v* tag; installers select supported OS/arch, verify sidecar/version and atomically replace the user binary. No default GitHub organization is invented or network/download performed by bootstrap. SHA sidecar is not a signature. |
| P-095 | **NIXOS MODULE HONESTY** | Expose `services.zerogravity.enable` disabled by default, safe journal/DynamicUser and future service options; since `serve` is not implemented, an explicitly enabled unit exits clearly and does not restart-loop, without claiming a functional daemon. |
| P-096 | **BOOTSTRAP SCOPE AND STATUS** | Record P00 tooling/binary subset as PARTIAL/IN_PROGRESS, no completed reference model or inference engine. Preserve source/design docs and agent handoff, do not change read-only dotlink/research, leave Git init/public release/license decisions separate. |

### Open decisions (do not fill silently)

| ID | Status | Question / recommended validation |
|---|---|---|
| O-001 | OPEN | What license for zerogravity? MIT/Apache-2.0 dual licensing is a *candidate*, not selected. Verify copied code license, attribution and weight redistribution separately. |
| O-002 | OPEN | Which accelerators must be production-grade at v1? Candidate first targets: Linux x86-64 CPU and AMD Strix Halo, then CUDA, Apple Metal, Vulkan; final order is not approved. |
| O-003 | OPEN | Exact interpretation of entirely Rust: are system CUDA/ROCm/Metal APIs, HIP/BLAS libraries and compiler-emitted non-Rust machine artifacts acceptable? Proposal: Rust-maintained source with driver/runtime FFI allowed; prove performance before forbidding vendor libraries. |
| O-004 | OPEN | Select primary GPU kernel toolchain after comparative proof (CubeCL, Rust GPU/AMDGPU, compiler extensions, vendor GPU APIs). No framework chosen definitively. |
| O-005 | OPEN | First complete model acceptance target: basic Llama-style dense vs Qwen3.5 hybrid vs Qwen Flash-Next vs GLM; phase gates recommend dense then a hybrid and giant MoE. |
| O-006 | OPEN | Which external format/quant combinations are promised at initial release? Capability matrix must be evidence-based, not optimistic. |
| O-007 | OPEN | How much sampling reproducibility across backends/batch schedules is required? Strong guarantee should at least cover same-plan, same-seed, same-schedule. Cross-plan bit identity often impossible without speed cost. |
| O-008 | OPEN | Choose stable versioned API/ABI and whether to ship C ABI, WASM/browser, FFI from other languages. |
| O-009 | OPEN | Distributed authentication/transport defaults and remote execution scope; do not expose unauthenticated RDMA/TCP by default. |
| O-010 | OPEN | Supported deployment platforms and packaging (NixOS, Linux distros, macOS, Windows, ARM) and GPU availability for CI. |
| O-011 | OPEN | Quality policy for low-bit checkpoint selection and acceptable evaluation delta. Do not make default quality regressions invisible. |
| O-012 | OPEN | Service SLOs and max concurrency/contexts for each hardware tier; derive from measured P50/P95 and memory admission. |
| O-013 | OPEN | Should external model definitions be declarative DSL, Rust code, or both? Prefer explicit versioned semantic registry and safe loading. |
| O-014 | OPEN | Whether to allow third-party Rust-authored kernel/model plugins; assess untrusted code and reproducibility before accepting dynamic execution. |
| O-015 | OPEN | Whether/when to initiate a Git repository, CI, automated releases and model benchmark hardware matrix. |
| O-016 | OPEN | Exact application/project branding, executable name, CLI flags and public version semantics. Working examples are illustrative only. |
| O-017 | OPEN | Formal Rust GPU source policy: are driver/vendor math FFI and separately licensed non-Rust kernel source allowed? User asked Rust-authored owned engine/kernels; research vendor-kernel escape is **not** automatically approved. |
| O-018 | OPEN | Final target kernel toolchain per device after physical benchmark: CubeCL gfx1151, target Rust AMDGPU lowering; independent Rust CUDA options are new as of Sep 2026 and require separate maturity/security/ISA assessment. |
| O-019 | OPEN | Should exact binary name zg, dual MIT/Apache-2.0 license, Rust MSRV and package name from old research be adopted? They remain council recommendations, not explicit user decisions. |
| O-020 | OPEN | Versioned acceleration-pack public filename and cache format (avoid overloading APK), plus correct disk-KV header/schema replacing the historical invalid 48/54-byte spec. |
| O-021 | OPEN | Thresholds for eager/capture kernel tactics, strict replay vs cross-plan numeric variance, online resource-cost policy hysteresis, cache tenant privacy and model-quality pass bands: establish by tests. |
| O-022 | OPEN | Prioritize first accepted dense, hybrid, MoE, embeddings/LoRA and multimodal release cells; user specified eventual breadth but not v1 feature commitments or CI hardware purchases. |
| O-023 | OPEN | Can a **Rust-authored XDNA2/AIE kernel** and program be built with a supported compiler/toolchain and XRT FFI, without bundling Gufo's Python MLIR generator/C++ AIE kernel as owned source? This is separate from the already open driver-math FFI policy. |
| O-024 | OPEN | How to qualify **same-SoC GPU+NPU concurrency safety** across firmware/AGESA, power states, GPU fabric clock and ROCm/XRT versions without privileged automatic system tuning? Unqualified configurations must remain GPU-only; physical hard-hang experiments require isolated test hardware. |
| O-025 | OPEN | First supported NPU workload and precision policy: fixed 4096×2560×1024 `v8bfp16ebs8` Qwen Flash-Next prefill versus generic GEMM shapes or encoder work; benchmark actual critical-path gain and quality under controlled clock/power. |
| O-026 | OPEN | TaskProgram production scope: text generation, ASR, TTS, image diffusion/edit and video/audio must have individual model/driver/license/quality/release cells. Prior-art examples do not imply zerogravity already supports these. |
| O-027 | OPEN | Which exact MIT-licensed source files, if any, to reuse as code rather than independently implementing the mechanism? Confirm file/embedded code/model weight licenses, required notices, Rust-source policy and user review before each import. Non-MIT source remains design research unless explicitly authorized. |
| O-028 | OPEN | How to scope external source attribution in public product documentation and distribute an independent legal source/SBOM table without introducing runtime dependencies on upstream project APIs? Current v4 keeps provenance in research docs, not normative interfaces. |
| O-029 | OPEN | Does “all AI execution” require **training/fine-tuning** as a first-release capability or only an extensible future GradientProgram/TrainStep/ParameterVersion seam? v4 chooses inference-first with future seam, user has not approved a training release scope. |
| O-030 | OPEN | Which task kinds beyond generation/embeddings are release-qualified first (ASR, TTS, image/video, agent-compute)? Supported TaskDriver *type* does not imply actual model/source/data/quality readiness. |
| O-031 | OPEN | Exact delivery guarantees for disconnected/remote clients: default ordered at-most-once live **attempt**; optional durable cursor/idempotent replay requires separate retention/privacy/ACK policy. |
| O-032 | OPEN | Physical memory/sharding strategy under device allocation pressure: choose per-device pool and coarse global lease/reservation/compaction, tuning based on real hardware and task workload rather than freezing one allocator. |
| O-033 | OPEN | When to introduce public plugin/IR ABI or external stable model import schemas? v4 keeps built-in Rust providers and minimal versioned SIR/StateRoot until real extension/use and security needs justify them. |
| O-034 | OPEN | How much fully GPU-native sampled control/grammar/device-loop execution is feasible on each target without sacrificing exactness, cancellation and request output streaming? Determine per model/shape/driver from physical proof, not universal promise. |
| O-035 | OPEN | Which external graph/device compiler/runtime backends and foreign device programs can be used while honoring Rust-owned semantics/kernel-source user policy? Graph lowerer through a vendor compiler is **not** a Rust-authored native TPU/wafer compiler; must decide SDK/license boundaries before implementation. |
| O-036 | OPEN | Which TPU and other graph accelerators offer actual documented source-op/state/shape/graph program APIs sufficient for Rust host lowering, snapshot/provisional state and task-loop execution? Architectural support is not real qualified compiler integration. |
| O-037 | OPEN | Can wafer-scale/static dataflow or deterministic LPU/RDU hardware SDKs be accessed and trusted for direct compiled-region execution with state lifecycle, or is only a remote opaque model API available? Treat latter as TaskService boundary, not native region implementation. |
| O-038 | OPEN | Which physical accelerator to qualify after baseline CPU/Strix/CUDA: available GB10 coherent ARM/Blackwell, workstation/discrete Blackwell, B200/B300 rack HBM, TPU7x/6e, cloud AOT, wafer/static fabric, AMD CDNA etc? Do not assume user owns expensive hardware or cloud capacity. |
| O-039 | OPEN | Cloud accelerator remote compilation/model artifact/privacy/egress/tenant authentication and prohibited third-party code import policy; TaskService integrations need separate user authorization and are not a low-level vendor device backend. |
| O-040 | OPEN | Compiler region granularity and device-resident state bridge: which programs may keep static mutable KV/recurrence/optimizer across calls while still supporting prepare/abort, rollback and common logical StateRoot? Must prove per backend, not infer from vendor API name. |
| O-041 | OPEN | Was user’s “cerberus” intended as **Cerebras** wafer-scale hardware, the **Cerberus** adaptive parallel-decoding paper, or both? Preserve both research paths without making an unsupported hardware claim or demanding clarification before design work. |
| O-042 | OPEN | Exact multi-node first release topology: two trusted heterogeneous LAN/tunnel machines with contiguous layer+state colocation, remote prefix, P/D, A/FFN or expert-first? Design supports all; prioritize physical hardware and measurable application workloads, not arbitrary feature count. |
| O-043 | OPEN | Preferred private peer connectivity and transport: direct authenticated TCP/QUIC vs existing WireGuard/Tailscale overlay tunnels, local RDMA/Thunderbolt when physical, NAT relay behavior and trust/double-encryption cost. **No approval to connect machines or install a tunnel is implied**. |
| O-044 | OPEN | Node identity/provisioning/authorization and tenant/model/checkpoint sharing permissions, weight license redistribution and remote prompt/hidden-state privacy policy; default require explicit enrollment. |
| O-045 | OPEN | Topology discovery: manual peer inventory first vs mDNS/optional overlay discovery, dynamic link measurement frequency and peer capability attestation. No automatic use of unknown peers. |
| O-046 | OPEN | Remote state shard transfer schema for KV+GDN/KDA+sparse indexer/ngram/grammar/RNG/modalities and versioned state snapshots: choose portable wire codec, delta/COW and layout conversion only after actual models/quant reference. |
| O-047 | OPEN | Coordinator crash/re-election and durability tier: fail-stop with replay vs durable checkpoints/lease service/quorum, output cursor and exactly-once requirements. v4.2 promises no automatic transparent HA until implemented. |
| O-048 | OPEN | Placement solver sophistication: static simple heuristic vs bounded search and online hysteresis; performance varies dramatically by direct/relay RTT, context, shape, per-layer operator type, task/power and physical cache residence. |
| O-049 | OPEN | Network transfer implementation for GPU direct, RDMA/UCX-style native libraries and fast collectives under Rust-owned code/vendor FFI policy, with fallback buffered TCP/QUIC. A VPN cannot guarantee GPU RDMA. |
| O-050 | OPEN | Which attention operators support exact token-range PartitionAlgebra and stable softmax summary combine? Need source-equation and numerical proof; compressed/selected sparse and GDN/KDA recurrence are NOT automatically sequence partitionable. |
| O-051 | OPEN | When to disaggregate mutable attention KV from FFN versus migrate cache once, keep whole layer+cache colocated or read remote state bytes: need measured per-link cost and safe migration/readiness/replay. |
| O-052 | OPEN | Model/task weights/adapter/quant cache distribution and encrypted storage, remote source permission and content-addressed trusted reference, cache directory presence hints versus authoritative remote mutable state generation. |
| O-053 | OPEN | Private WAN-tunneled machine capacity mode vs home LAN vs rack/cluster behavior and expected SLAs—must qualify separately, rather than one universal default topology. |
| O-054 | OPEN | Final binary name (`zerogravity` vs shorthand), exact command aliases and localhost default port; examples are proposed, not a stable CLI approval. |
| O-055 | OPEN | Per-model default context, keepalive/idle unload, max concurrent resident models and parallel sessions: adaptive resource safety vs predictable user controls, not fixed arbitrary source-project limits. |
| O-056 | OPEN | Config on-disk format/location/schema migration, global/model/alias/request/CLI/env precedence and secrets handling; v4.3 proposes versioned typed TOML but final user setting policy remains open. |
| O-057 | OPEN | Model catalog trusted source registry list, automatic local alias discovery, explicit `pull` network consent, license/size/checkpoint pin and offline semantics; avoid silent remote download. |
| O-058 | OPEN | Which OpenAI API subset to release first beyond chat/models, including completions, embeddings, stateless Responses, stateful previous_response_id and streaming schema/SDK version; no blanket compatibility promise. |
| O-059 | OPEN | `run` process lifecycle: always in-process, optionally connect trusted same-user resident daemon/socket, explicit auto-service mode, NixOS/system service setup, startup and history retention; no silent system service installation. |
| O-060 | OPEN | Exact replica load balancer cost and stickiness: session StateRoot affinity, cache hints, predicted P95/TTFT, per-tenant fairness, model load churn; when to prewarm/create replicas vs move a session. |
| O-061 | OPEN | Policy-defined different-target `auto` alias: whether/how to expose optional automatic model selection by quality/privacy/cost while disallowing unrequested model/quant substitutions for exact requests. |
| O-062 | OPEN | Model profile editable parameters and default modes, including runtime/reload safety boundaries; which sampler/grammar/context overrides should be possible per request versus require new TaskSession. |
| O-063 | OPEN | Native administrative model-management API (local socket vs authenticated HTTP, administrative scoping) and public exposed `/v1/models` catalog/alias privacy; model source paths/hardware/peer topology must not leak. |
| O-064 | OPEN | Many-model physical residency and eviction policy under unified RAM/GTT, VRAM, per-model context state, adapter/shared weights and remote replication; exact pin/keepalive/scale-to-zero guarantees and cold-start SLA. |
| O-065 | OPEN | Advanced cross-model TaskProgram roles and router SLOs: target/draft/encoder/reranker/vocoder on independently placed qualified devices with one tenant admission budget; actual model combos need explicit task quality. |
| O-066 | OPEN | OpenAI Responses-compatible state storage and session retention, privacy and stream cursor contract vs stateless-only initial endpoint; server-side tool execution is outside zerogravity inference core. |
| O-067 | OPEN | Public author/byline, affiliations and citation contact for whitepaper publication; do not attribute the design to unapproved individual or organization names. |
| O-068 | OPEN | Public repository code license and whitepaper/documentation copyright/distribution license; current publication draft must not assume MIT/Apache or CC-BY for original zerogravity source. |
| O-069 | OPEN | Publication channel/URL, DOI or stable version identifier, visual template/PDF/HTML export and accessibility standard; markdown manuscript exists but no publication artifact was requested/produced yet. |
| O-070 | OPEN | Independent technical peer review, legal/trademark review and final owner signoff before public whitepaper release; design arguments are not verified runtime implementation benchmarks. |
| O-071 | OPEN | Future results edition requires exact model/quant/source/driver/hardware/network/task baseline, quality/sampling/p95/energy, raw reproducibility and negative results; no current benchmark evidence exists. |
| O-072 | OPEN | Final GitHub repository OWNER/zerogravity, initial source license and signed-release/provenance policy before public release. Installers deliberately require explicit repository or release base URL rather than guessing a host. |
| O-073 | OPEN | Final tested support matrix for all six cross-built targets; Apple SDK license/source access, macOS signing/notarization, Windows ARM64 toolchain and real per-OS binary smoke tests. Rust native Linux build alone is not proof. |
| O-074 | OPEN | When the real serving runtime exists, update `services.zerogravity` execution arguments, restart behavior, listen/credentials and mutable model directories; currently the module's `serve` command intentionally fails and the unit is disabled by default. |

### Assumptions / boundaries

- A-001 (SUPERSEDED): initial relative-path interpretation placed `./zerogravity` under dotlink l5's launch directory `/home/pvl/spaces/abird`. User decision U-013 explicitly overrides it: canonical root is `/home/pvl/spaces/abird/src/zerogravity`.
- A-002: Research is a 2026-10-10 **snapshot**; named model geometries, release support and benchmark numbers require source pinning and re-verification.
- A-003: Reference benchmark reports are *external author measurements*. zerogravity has not executed tests or proved superiority.
- A-004: Full general fallback requires a known semantic implementation for every operator, sufficient resources and an available execution backend; unknown Python custom model code is not magically executable.
- A-005: The user has not requested using dotlink to modify existing projects or publish a GitHub repo.
- A-006: No specific secret, account, network endpoint or user credential is part of this project design.

### v3 architectural supersession and provenance rule — 2026-10-12

**U-017 remains the user-mandated project-neutral/source-license policy; U-018 is the latest comprehensive review/minimality instruction.** Source-specific implementation patterns mentioned in older U-00x user motivations, P-001…P-030 assistant proposals and old source-council D-xx records are **retained as historic evidence**, not live project-named architectural ownership. They are governed by this v4 distinction:

1. **Original normative design**: `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`, `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`, `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`, v4 `ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`, current roadmap, experiment and release plans. These use typed model/task/region/device/resource/service contracts; P-002's old three-IR abstraction was refined by P-018/P-035 and **minimized by P-036**. Prepared-root publication and hierarchical allocator supersede ambiguous v3 StateDomain/ResourceArbiter sketches.
2. **Named external sources**: `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` verifies root/file license evidence and URLs; Retained source investigations preserve attribution, source findings, decisions and reported benchmarks as non-normative research. Retired snapshot bodies remain in Git history. Exact model checkpoint and hardware identifiers can appear in validation matrices; they are not architecture type names.
3. **MIT exception**: named references and source-reuse candidates from verified MIT-licensed projects are allowed but are not required dependencies. Even an MIT repository can contain differently licensed subfiles/device kernels/model weights. Reuse remains subject to specific file license/NOTICE, source-language and user review. Non-MIT works provide attributed research and engineering ideas, not copied or named project-specific runtime interfaces by default.
4. **No composability exception for performance**: even a much faster optimized model device graph must implement a closed semantic region or TaskProgram stage through the same executor/state/resource contracts, never a cloned model-specific server.
5. **Future decisions**: source attribution/compatibility is a research record; user product/architecture acceptance needs explicit U-/D-style decision and independent model/hardware proof. No prior project's "resolved" option can overrule U-017.

**Policy goal:** external projects could disappear or change license tomorrow without invalidating any core zerogravity architectural interface. v4 adds the further requirement that an optional model/kernel/task specialization can be removed without cloning or corrupting the shared runtime.

### Conflict-resolution protocol

Write future decisions as `D-YYYYMMDD-NN` with source (user instruction/review), options, trade-offs, tests, conclusion, superseded IDs, and affected files. Never upgrade a PROPOSAL to USER on an assistant's own authority. Whenever a model/hardware benchmark contradicts a claim, record the measured evidence and change the claim.

| U-050 | USER | “Let's go, next phase”: resume implementation after the pipeline design/reference cleanup, preserving the single Rust engine and source/StateRoot/physical ownership boundaries. Next bounded H7 moves original GDN polynomial SiLU and output gating; source host exp/ln coefficients are the next separate numerical gate. Full build/lint/tests, trained source checks, AMD/NVIDIA physical qualification, rollback/quarantine and archived handoff/SHA updates precede an atomic local commit. README stays unchanged; no downloads, device changes or publication. |

| U-051 | USER | Requested the next implementation phase after H7. Continue H8 original-source GDN nonlinear coefficients through the existing Rust engine/provider and immutable source/session/resource ownership, with exact FP32 arithmetic and real trained AMD/NVIDIA physical qualification. Preserve the user README, record failures and acceptance, regenerate handoff/SHA, and commit verified work locally; acquisitions, system changes and publication remain unauthorized. |


| U-052 | USER | Requested the next implementation phase after H8. Continue bounded H9 source-exact attention RoPE and private current-token KV proposals through the existing Rust engine/provider, immutable source and independent session/resource ownership. Qualify actual trained AMD/NVIDIA operations, exact rounding and nonzero positions, rollback, admission and post-launch quarantine; retain failed experiments and independent source evidence, regenerate handoff/SHA and commit verified work locally. User README, download/device/publication restrictions remain in force. |


| U-053 | USER | Requested the next implementation phase after H9. Continue source-exact GPU attention through the existing Rust engine/provider and immutable source, independent session StateRoot and physical resource ownership. H10 qualifies bounded QK scoring before separately verified softmax/value stages, preserving exact source rounding and nonzero private prefixes; verify both trained checkpoints/cards, rollback/admission/quarantine and independent source evidence, then update archive/SHA and commit locally. README and acquisition/device/publication restrictions remain in force. |


| U-054 | USER | Requested the next implementation phase after H10. Continue original-source GPU attention through the existing Rust engine/provider, immutable source identity, private StateRoots and physical resource ownership. H11 qualifies padded attention softmax with exact exponential/reduction and two-round normalization, both trained checkpoints/cards, independent native captures, rollback/admission/quarantine, then updates the handoff/SHA and commits verified work locally. README and acquisition/device/publication restrictions remain in force. |


| U-055 | USER | Explicitly requested “let's go, next, let's finish up full gpu inference”, extending the preceding individually requested operator phases. Continue verified implementation through remaining attention/FFN/residual/embedding math, GPU dynamic activation quantization and a complete resident pipeline using the same Rust mathematical schedule and independent transactional StateRoots. Preserve original source bits/identity, fair physical workers/arbiter, actual completion, memory admission and quarantine; no selected-error CPU substitution or unsupported whole-GPU/performance claims. H12 adds source-padded value-cache accumulation and corrects a native-proven signed-zero corner in the former valid-only CPU schedule. Root README remains byte-identical; no models/SDKs, device privileges, push/tag/release authorized. [Full completion tracking](../plans/accepted-baseline/TODO.md), [H12 evidence](../plans/accepted-baseline/TODO.md). |


### Owner decisions after U-055 — consolidated 2026-10-17

These rows preserve the subsequent user instructions from the implementation
conversation. Earlier rows describe the state and authority at their own point
in time. Current implementation is tracked in the [plan index](../plans/TODO.md),
[accepted baseline](../plans/accepted-baseline/TODO.md) and individual plan TODOs.
U-067/U-068 supersede older archive/SHA/generated-handoff process duties;
U-069/U-070 leave runtime implementation paused during this documentation task.
Historical checkpoint labels may predate the rewritten Git graph: a recorded
approval is distinct from a currently resolvable Git object.

| ID | Type | Decision and current meaning |
| --- | --- | --- |
| U-056 | USER | Repeatedly continue verified GPU phases through full GPU inference, then enable native NVIDIA CUDA and AMD ROCm/HIP. Preserve one mathematical engine, original source identity, independent transactional StateRoots, physical admission, fair workers, actual completion and quarantine. A passed operator is not a full-model claim. The accepted original-model resident512-token scope and native CPP profiles are recorded in accepted-baseline; independent8K drift, larger contexts and held hardware remain open. |
| U-057 | USER | Include the current machine and SSH targets pvl-a1/pvl-x2 as the first optimization fleet: five physical GPU devices and the best paths for all three CPUs. Maximize demonstrated capability on these targets using measured, source-correct optimizations. Target discovery/SSH qualification does not qualify distributed inference or authorize model/SDK downloads, persistent configuration, driver/device privilege changes or held-device recovery. |
| U-058 | USER | Use subagents for implementation and independent work, parallelizing wherever efficient; the main agent primarily coordinates, gates, reconciles, decides and merges. Use disposable side worktrees for overlapping source work, preserve ownership, then merge only verified pieces and retire superseded work. One MAIN Cargo/target owner avoids build contention and stale cross-worktree results. |
| U-059 | USER | After the native-enablement checkpoint, do the workspace layout/structure redesign and then a thoughtful left-right concurrency pass before returning to GPU optimization. Add left-right where contention and immutable publication ownership justify it. Its later cold model-catalog proposal was challenged by the user because that area was not contended; the accepted result is no speculative migration. Future candidates need representative contention evidence and deterministic dual-replay/reader-lifetime/resource tests. |
| U-060 | USER | Remove the independent probes/ directory by relocating reusable implementation to crates/, qualification tools to tools/ and appropriate tests to their owners. Consolidate the workspace/lock/version and preserve the sole product CLI. Clean only merged/superseded side worktrees. The old29.7GB build cache and historical evidence binaries need not be preserved; prior resnapshot/archive directions were subsequently superseded by U-066/U-067. Root README remains unchanged. |
| U-061 | USER | Immediately after the concurrency pass, implement explicit model acquisition from GitHub and Hugging Face into the XDG cache, then cover the finite applicable pull capabilities of pinned llama.cpp/vLLM. Preserve source authority, complete required closure, offline-before-credentials, ownership/admission/cancellation and the single native transport/cache/loader boundaries. Additional provider/profile support requires its own tests; SDK/plugin delegation does not imply native equivalence. |
| U-062 | USER | In parallel with independent checks, begin Qwen3.8 27B, Gemma4 E2B and Gemma4 26B model support. First bind source/config/index/tokenizer identities and basic source-correct CPU execution, then GPU. Existing delivered contracts are not trained-model inference; no new model or SDK downloads are authorized by the model request. |
| U-063 | USER | For Qwen, finish basics first, then match every applicable capability and optimization of the pinned llama.cpp/vLLM union, including public and internal surfaces. Afterwards add appropriate MTP, speculative decoding, n-gram/suffix, DSpark2/DFlash2 and related strategies under target-authoritative state/distribution correctness. Exact primary identity/version for DSpark2 remains unresolved; provisional frontier counts are not an exhaustive parity claim. |
| U-064 | USER | Use concurrency primitives thoughtfully, not to fill cheap places arbitrarily. Specifically the model catalog is not demonstrated contention. Preserve a simple existing lock where it fits; performance claims require measured benefit, correct publication/lifetime ownership and minimal maintenance cost. |
| U-065 | USER | Requested commit/push at earlier checkpoints. Those particular actions do not authorize later publication. The archive split proposal initially selected verified ordinary-Git parts rather than LFS, but was superseded by the owner-approved checkpoint/evidence deletion decision. Current AGENTS requires explicit authorization for each push, force-push, tag, release, deployment or destructive Git action; local verified atomic signed commits remain the normal implementation workflow. |
| U-066 | USER | Completed phases may use a singular owner-approved checkpoint as accepted truth; historical evidence archives and binaries are unnecessary and can be removed after preserving completed/tested lists and remaining work. The repository was subsequently rewritten. Preserve actual source/checkpoint/content identities and implementation tests, while distinguishing old approval labels from reachable history. The earlier preference to retain a future evidence model was superseded by the later simpler TODO instructions. |
| U-067 | USER | Replace the entire handoff/evidence process with simple TODO.md files: plans/TODO.md indexes plans; plans/<plan>/TODO.md records provenance, committed implementation/tests and current unfinished work; runs/<plan>-<session-or-agent-id>/TODO.md holds each agent's own work. On completion the coordinator atomically integrates useful results and unfinished items into the main plan; completed plans get one checked index line. Carry unfinished items forward after every commit; archive only long completed Markdown TODO sections. Delete obsolete evidence records and historical archives after useful context is retained. |
| U-068 | USER | Remove SHA inventories and staged tree/blob receipt requirements from evidence logs. Actual implementation tests prove behavior; provenance alone does not prove commands or outputs. A test item must state its intended inputs/behavior and be checked only after execution. Git commit provenance and ordinary resumable TODOs replace mandatory per-file receipts. Actual model/content hashes, integrity checks and numerical acceptance contracts remain required where they are part of implementation. |
| U-069 | USER | After the latest interruption, wrap the current phase, revalidate live source/process/worktree state, then pause. Quant selection finished at signedf197c9c after975focused/fourstrict/full934Rust+18namespace/originalmodel passes; two source-only filters/Xet-index drafts remain unfinished and unmerged. All implementation agents stopped. Only routine rolling TODO updates remained dirty before this handoff documentation request; nothing was pushed. |
| U-070 | USER | Create a comprehensive Markdown handoff and ready-to-copy next-agent starting prompt covering context from the beginning: initial design, specifications, user decisions, implementation steps, real verification scope, all pending work and exact resumption instructions. Record the handoff and its retained context under .agents/docs or .agents/plans, preserve root README, and identify any irrecoverable originals honestly. This is a documentation request; it does not resume the paused implementation or reinstate obsolete evidence/SHA/archive automation. |


<a id="catalog"></a>

## Complete reading catalog and source ownership

All current normative architecture/specification and plan sources are listed here, with retained research/prior-art references for full equations, alternatives, attribution, negative cases and acceptance matrices. New handoff records and portable unfinished work are under `.agents/docs/` or `.agents/plans/`. Existing research paths remain in place to preserve live canonical links, including the immutable root README’s research-pointer target; their relevant context is integrated above. Do not relocate or regenerate them as an onboarding prerequisite.

Read the relevant original in full before implementing that subsystem; a brief catalog entry is navigation, not an additional support claim.


### Public engineering and publication context


- [Development, builds and release definitions](../../docs/DEVELOPMENT.md) — `docs/DEVELOPMENT.md`.

- [Zerogravity — Integrated Engineering Specification](../../docs/ENGINEERING_SPECIFICATION.md) — `docs/ENGINEERING_SPECIFICATION.md`.

- [Whitepaper publication and final architecture review checklist](../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md) — `docs/PUBLICATION_AND_REVIEW_CHECKLIST.md`.

- [Zerogravity documentation](../../docs/README.md) — `docs/README.md`.

- [References and source provenance](../../docs/REFERENCES.md) — `docs/REFERENCES.md`.

- [Zerogravity](../../docs/WHITEPAPER.md) — `docs/WHITEPAPER.md`.


### Normative subsystem architecture


- [zerogravity — Universal region-execution backend contract (v4.1)](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) — `.agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md`.

- [zerogravity — Async Command/Resource IR (CIR), ownership and compiled execution](ASYNC_RESOURCE_COMMAND_IR.md) — `.agents/docs/ASYNC_RESOURCE_COMMAND_IR.md`.

- [zerogravity — v4.1 typed composition and region execution contracts](COMPOSITION_AND_EXTENSION_CONTRACTS.md) — `.agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md`.

- [Joint cost model, online adaptation and compiler autotuning](COST_MODEL_AND_AUTOTUNING.md) — `.agents/docs/COST_MODEL_AND_AUTOTUNING.md`.

- [Zerogravity — EXL3/QTIP and Source-Qualified Derived Quantization Contract](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md) — `.agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md`.

- [Device abstraction, Rust GPU kernels, code generation and backend specialization](DEVICE_KERNEL_AND_CODEGEN.md) — `.agents/docs/DEVICE_KERNEL_AND_CODEGEN.md`.

- [Distributed and heterogeneous execution — v4.2 authoritative index](DISTRIBUTED_PARALLELISM.md) — `.agents/docs/DISTRIBUTED_PARALLELISM.md`.

- [zerogravity — Distributed placement, network and state execution contract (v4.2)](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) — `.agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md`.

- [zerogravity — Native Engine Core, Worker Protocol, KV policies and request lifecycles](ENGINE_CORE_EXECUTION_AND_PROTOCOL.md) — `.agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`.

- [Design walkthroughs — concrete execution and fallback traces](EXAMPLE_EXECUTION_TRACES.md) — `.agents/docs/EXAMPLE_EXECUTION_TRACES.md`.

- [Zerogravity — Extreme Hybrid Model Recipes, Native MTP and DFlash Source Contracts](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) — `.agents/docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md`.

- [Model artifacts, quantization, tokenizer/template parity and safe loading](FORMATS_QUANTIZATION_AND_LOADING.md) — `.agents/docs/FORMATS_QUANTIZATION_AND_LOADING.md`.

- [Zerogravity — gfx1151 ROCm Kernel Specialization, MoE Scheduling and Unified-Memory Safety Contract](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) — `.agents/docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md`.

- [Zerogravity — Heterogeneous prefill and retained replay contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) — `.agents/docs/HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md`.

- [Cross-cutting invariants, state lifetimes and failure semantics](INVARIANTS_AND_FAILURE_SEMANTICS.md) — `.agents/docs/INVARIANTS_AND_FAILURE_SEMANTICS.md`.

- [Zerogravity — Joint Concurrent Serving, Goodput and Combinatory Specialization Contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) — `.agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md`.

- [Global memory, expert residency, storage I/O and resource admission](MEMORY_RESIDENCY_AND_IO.md) — `.agents/docs/MEMORY_RESIDENCY_AND_IO.md`.

- [Minimal execution spine v4 — state, asynchronous workers, resource ownership and hot-path performance](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) — `.agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`.

- [zerogravity — Built-in model router, multi-model residency and serving contract (v4.3)](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) — `.agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md`.

- [Model semantics, model adapters and compatibility qualification](MODEL_SEMANTICS_AND_COMPATIBILITY.md) — `.agents/docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md`.

- [zerogravity — modular AI execution reference architecture v4.1](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md) — `.agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`.

- [zerogravity — NPU/GPU cooperative execution and multimodal TaskProgram design](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) — `.agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md`.

- [Numerical correctness, model-quality evaluation and benchmark methodology](NUMERICS_QUALITY_AND_BENCHMARKS.md) — `.agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md`.

- [Operator/region contracts, fusion patterns and model execution catalog](OPERATOR_AND_FUSION_CATALOG.md) — `.agents/docs/OPERATOR_AND_FUSION_CATALOG.md`.

- [Architectural risks, opportunities, trade-offs and falsification criteria](PERFORMANCE_HYPOTHESES_AND_RISKS.md) — `.agents/docs/PERFORMANCE_HYPOTHESES_AND_RISKS.md`.

- [zerogravity — Product, model loading and extension contract across the full pipeline](PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md) — `.agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`.

- [Zerogravity Real-Time Multimodal Serving, Sessions, QoS and API Contract — v5.1](REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md) — `.agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md`.

- [zerogravity — Remote peer transport, state handoff and failure protocol (v4.2)](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md) — `.agents/docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md`.

- [Runtime, APIs, developer experience and observability](RUNTIME_SERVING_AND_UX.md) — `.agents/docs/RUNTIME_SERVING_AND_UX.md`.

- [Rust workspace, interfaces, implementation practices and contributor workflow](RUST_WORKSPACE_AND_ENGINEERING.md) — `.agents/docs/RUST_WORKSPACE_AND_ENGINEERING.md`.

- [Scheduler, continuous batching, prefix reuse and speculative execution](SCHEDULER_BATCHING_AND_SPECULATION.md) — `.agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md`.

- [Security, reliability, resilience and safe defaults](SECURITY_AND_RELIABILITY.md) — `.agents/docs/SECURITY_AND_RELIABILITY.md`.

- [Semantic IR, state-effect system and ahead-of-time/JIT execution planning](SEMANTIC_IR_AND_COMPILER.md) — `.agents/docs/SEMANTIC_IR_AND_COMPILER.md`.

- [Session state, prefix caching, snapshots and transactional semantics](STATE_CACHE_AND_TRANSACTIONS.md) — `.agents/docs/STATE_CACHE_AND_TRANSACTIONS.md`.

- [v4 TaskProgram and compiler contracts — high throughput with a small semantic core](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) — `.agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`.

- [Zerogravity — Unified multimodal execution contracts (v5 design proposal)](UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md) — `.agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md`.

- [Zerogravity — First-Class Virtual Tensor Residency and High-Performance SSD Streaming](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) — `.agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md`.

- [Zerogravity Universal Multimodal Execution Architecture — v5.1 Design Specification](UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md) — `.agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md`.

- [Zerogravity Verified Model Assemblies, Source Codecs and Legal Specialization — v5.1](VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md) — `.agents/docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md`.

- [Vision, product scope, constraints and engineering philosophy](VISION_SCOPE_AND_PRINCIPLES.md) — `.agents/docs/VISION_SCOPE_AND_PRINCIPLES.md`.

- [zerogravity — One-command CLI, progressive configuration and OpenAI-compatible APIs (v4.3)](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md) — `.agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md`.


### Cross-cutting specifications and decision/workflow owners


- [Zerogravity — implementation design reference](../plans/AGENT_IMPLEMENTATION_PLAYBOOK.md) — `.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md`.

- [Current work](../plans/CURRENT_STATUS_AND_TRACKING.md) — `.agents/plans/CURRENT_STATUS_AND_TRACKING.md`.

- [Decision register — confirmed user intent versus recommended design](../plans/DECISION_REGISTER.md) — `.agents/plans/DECISION_REGISTER.md`.

- [zerogravity — Project-neutral experimental program, H01–H43 (v4 reviewed)](../plans/EXPERIMENT_PROGRAM.md) — `.agents/plans/EXPERIMENT_PROGRAM.md`.

- [Large immutable source loading: bounded basic-unit proposal](../plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md) — `.agents/plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md`.

- [Zerogravity — architecture roadmap P00–P23](../plans/IMPLEMENTATION_ROADMAP.md) — `.agents/plans/IMPLEMENTATION_ROADMAP.md`.

- [zerogravity — Independent Stage-0 feasibility and delivery gates (v4 reviewed)](../plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md) — `.agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`.

- [Acquisition next unit — bounded cache first, then HF/GitHub vertical slice](../plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md) — `.agents/plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md`.

- [Starting prompt — Zerogravity takeover, 17 October 2026](../plans/NEXT_AGENT_PROMPT_2026-10-17.md) — `.agents/plans/NEXT_AGENT_PROMPT_2026-10-17.md`.

- [Zerogravity — Human-Centric Design, AI-Centric Implementation, and Open Contributions](../plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md) — `.agents/plans/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md`.

- [zerogravity — Model-neutral release, quality and composability gates (v4 reviewed)](../plans/RELEASE_ACCEPTANCE_GATES.md) — `.agents/plans/RELEASE_ACCEPTANCE_GATES.md`.

- [Zerogravity — requirements and test design](../plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md) — `.agents/plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md`.

- [TODO spec — plans, checklists and evidence](../plans/TODO-SPEC.md) — `.agents/plans/TODO-SPEC.md`.

- [Zerogravity plans](../plans/TODO.md) — `.agents/plans/TODO.md`.

- [zerogravity — master architecture v4.3: composable AI execution, model routing and simple user experience](../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md) — `.agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`.


### Live plan TODO owners and completed plain-checklist archives


- [Accepted implementation baseline](../plans/accepted-baseline/TODO.md) — `.agents/plans/accepted-baseline/TODO.md`.

- [Cpu gpu optimization](../plans/cpu-gpu-optimization/TODO.md) — `.agents/plans/cpu-gpu-optimization/TODO.md`.

- [Distributed execution](../plans/distributed-execution/TODO.md) — `.agents/plans/distributed-execution/TODO.md`.

- [Simplify agent handoffs](../plans/handoff-cleanup/TODO.md) — `.agents/plans/handoff-cleanup/TODO.md`.

- [TODO — historical records](../plans/historical-records/TODO.md) — `.agents/plans/historical-records/TODO.md`.

- [Memory and streaming](../plans/memory-and-streaming/TODO.md) — `.agents/plans/memory-and-streaming/TODO.md`.

- [TODO — model acquisition](../plans/model-acquisition/TODO.md) — `.agents/plans/model-acquisition/TODO.md`.

- [Model execution](../plans/model-execution/TODO.md) — `.agents/plans/model-execution/TODO.md`.

- [Multimodal](../plans/multimodal/TODO.md) — `.agents/plans/multimodal/TODO.md`.

- [Numerics and context](../plans/numerics-and-context/TODO.md) — `.agents/plans/numerics-and-context/TODO.md`.

- [Platform and release](../plans/platform-and-release/TODO.md) — `.agents/plans/platform-and-release/TODO.md`.

- [Serving and speculation](../plans/serving-and-speculation/TODO.md) — `.agents/plans/serving-and-speculation/TODO.md`.

- [Model acquisition](../plans/model-acquisition/README.md) — `.agents/plans/model-acquisition/README.md`.

- [TODO — completed model acquisition phases through a774ded](../plans/model-acquisition/archive/through-a774ded/TODO.md) — `.agents/plans/model-acquisition/archive/through-a774ded/TODO.md`.

- [TODO — completed model acquisition through fce1353](../plans/model-acquisition/archive/through-fce1353/TODO.md) — `.agents/plans/model-acquisition/archive/through-fce1353/TODO.md`.


### Retained research and acceptance registers


- [Zerogravity — Extreme Model Specialization Integration Roadmap and Executable Acceptance Design](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) — `.agents/research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md`.

- [Gemma 4 family source contracts — 2026-10-16](../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md) — `.agents/research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md`.

- [GPU tokenization, execution control and text decoding roadmap](../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md) — `.agents/research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md`.

- [Zerogravity — Heterogeneous prefill and retained replay acceptance](../research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md) — `.agents/research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md`.

- [Initial hardware targets and capability optimization](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md) — `.agents/research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md`.

- [Model pull, XDG artifact cache and pinned acquisition parity](../research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md) — `.agents/research/MODEL_PULL_XDG_CACHE_AND_SOURCE_PARITY_2026-10-16.md`.

- [Zerogravity — Model-family research atlas and multimodal implementation roadmap](../research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md) — `.agents/research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md`.

- [Zerogravity Multimodal v5.1 — Complete Implementation Roadmap, Dependency Gates and Acceptance Matrix](../research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md) — `.agents/research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md`.

- [Zerogravity v5.1 Multimodal Decisions, User Requirements, Design Proposals and Open Questions](../research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md) — `.agents/research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md`.

- [Proposed native FP32 GEMV FMA qualification](../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md) — `.agents/research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md`.

- [A1 local cache — protocol and API reference](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md) — `.agents/research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md`.

- [Qwen basics, complete reference parity and advanced decoding](../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md) — `.agents/research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md`.

- [Development, builds and releases](../research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md) — `.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md`.

- [Zerogravity Specialization Prior Art — Specialist Source Atlas](../research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md) — `.agents/research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md`.

- [Zerogravity — First-Class SSD Streaming Implementation Roadmap and Acceptance Tests](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) — `.agents/research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md`.

- [AMD Strix Halo dossier — gfx1151 extreme inference targets](../research/STRIX_HALO_OPTIMIZATION_DOSSIER.md) — `.agents/research/STRIX_HALO_OPTIMIZATION_DOSSIER.md`.

- [Zerogravity — vLLM Parity and Combinatory Serving Acceptance Program](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md) — `.agents/research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md`.

- [Research Track B — ASR, Speech Translation and Streaming Speech Understanding](../research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md) — `.agents/research/multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md`.

- [Research Track H — Cross-Stream Adversarial Reconciliation and Minimal Architecture Delta](../research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md) — `.agents/research/multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md`.

- [Research Track F — Typed Decision Models, Ternary Formats and Specialist Text Architectures](../research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md) — `.agents/research/multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md`.

- [Research Track D — Image Generation, Editing, Flow/Diffusion and Autoregressive Image Models](../research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md) — `.agents/research/multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md`.

- [Research Track G — Real-Time and Multimodal Serving / API Prior Art and Design Reconciliation](../research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md) — `.agents/research/multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md`.

- [Research Track C — Text-to-Speech, Neural Audio Codecs, Voice and Full-Duplex Conversation](../research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md) — `.agents/research/multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md`.

- [Research Track E — Video, Temporal Diffusion, Synchronized Audio-Video and Frontier Providers](../research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md) — `.agents/research/multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md`.


### Prior art, critical review and license/source attribution


- [Accelerator execution prior art — Blackwell, TPU, wafer-scale, static dataflow and specialized runtimes](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md) — `.agents/prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md`.

- [zerogravity — Reconciled architecture and authority matrix](../prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md) — `.agents/prior-art/ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md`.

- [zerogravity — v4 critical architectural review, challenges and resolution record](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md) — `.agents/prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md`.

- [zerogravity — Comprehensive cross-stack research and design selection](../prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md) — `.agents/prior-art/COMPARATIVE_COMPILER_AND_RUNTIME_RESEARCH_2026-10-10.md`.

- [Distributed execution, state disaggregation and tunnel transport: prior-art research](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md) — `.agents/prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md`.

- [Zerogravity — Final Architecture, Editorial and Agent-Handoff Review](../prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md) — `.agents/prior-art/FINAL_ARCHITECTURE_REVIEW_2026-10-12.md`.

- [Gufo XDNA2 cooperative-prefill fork and October 2026 Strix Halo report — primary-source research](../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md) — `.agents/prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md`.

- [Multi-model local runtime, router, CLI and API prior-art research](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md) — `.agents/prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md`.

- [Prior art, provenance and architecture neutrality](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md) — `.agents/prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`.

- [Source atlas — architectural research preserved for implementers](../prior-art/RESEARCH_SOURCE_ATLAS.md) — `.agents/prior-art/RESEARCH_SOURCE_ATLAS.md`.

- [SSD Streaming Research Dossier — DS4/DwarfStar and related offload prior art](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md) — `.agents/prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md`.

- [vLLM Concurrency Prior Art and Zerogravity Architecture Reconciliation](../prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md) — `.agents/prior-art/VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md`.


### Current executable/qualification/reference runbooks


- [Working in Zerogravity](../../AGENTS.md) — `AGENTS.md`.

- [Contributing to zerogravity](../../CONTRIBUTING.md) — `CONTRIBUTING.md`.

- [Zerogravity CLI](../../crates/zerogravity-cli/README.md) — `crates/zerogravity-cli/README.md`.

- [Zerogravity resident GPU library](../../crates/zerogravity-gpu/README.md) — `crates/zerogravity-gpu/README.md`.

- [CubeCL native target and compiler dependency support](../../crates/zerogravity-gpu/vendor/README.md) — `crates/zerogravity-gpu/vendor/README.md`.

- [Explicit GPU qualification frontend](../../tools/gpu-qualify/README.md) — `tools/gpu-qualify/README.md`.

- [Independent CPU reference probes](../../tests/reference/llama_cpp/README.md) — `tests/reference/llama_cpp/README.md`.


The two [CLI-option frontier](../plans/qwen-parity/CLI_OPTION_FRONTIER.csv) and [API-route frontier](../plans/qwen-parity/API_ROUTE_FRONTIER.csv) tables are unreviewed pinned parity seeds, not certified exhaustive compatibility. The current acquisition README References pin source protocols independently from the numerical oracle; use the exact version appropriate to the question.


<a id="handoff-validation"></a>

## Handoff validation and remaining limitations

This documentation request did not resume implementation, execute model/GPU work, download inputs, modify devices/services/caches or publish. The last implementation test results above are recorded accepted results atf197c9c, not a claim of fresh runtime testing during handoff writing. New-document integrity, exact open-item/decision/catalog coverage, reachable-history recovery, draft patch reconstruction and preservation checks are separately recorded after review. All unrecoverable early history is explicitly identified; no document count, Git label or generated summary repairs that gap.

### Completed documentation validation — 17 October2026

- Independent architecture/decision, complete continuation and history/recovery
  audits reviewed the assembled document. The initial overly broad historical
  loss claim was corrected after finding the embedded older94-source handoff.
  Final history approval compared every94 source title/range, all13+3 outer
  sections and all77 current commit IDs/subjects directly with primary Git text.
- Mechanical coverage verified every121 live open item, all twelve plan owners,
  all current canonical source catalog paths, and complete U-001…070,
  P-001…096, O-001…074 and A-001…006 ranges. The assumptions are prose bullets,
  so the first table-only coverage assertion was corrected; no source decision
  or expected implementation result was changed to satisfy it.
- All current local-file links and handoff navigation anchors resolve, with
  no dependency on completed documentation run files. The missing acquisition
  README catalog entry and transient history-run link found in review are fixed.
- Two portable patches independently reconstruct exactly four paused draft
  source files fromf197c9c. Both actual disposable application/byte comparison
  and an independent in-memory hunk application passed. Three context files and
  the inert generator source preserve interrupted work beneath `.agents/docs/`.
- Root README is byte-equal to HEAD; the existing decision register prefix is
  unchanged, with only the dated owner-decision consolidation appended. Runtime,
  tests, models, dependency manifests/lock, vendor/Nix/CI and SIDE source remain
  unchanged. No MAIN index entries were staged and no commit/push was made for
  this documentation request.
- Completed documentation-agent scratch runs are retired after integration.
  The original interrupted filters run and both unfinished worktrees remain.
  Runtime implementation is PAUSED; this handoff's read-first prompt preserves
  that pause until the owner explicitly resumes it.

This requested handoff is a one-time context document. Future work continues
through the existing simple TODO rules; no handoff generator, archival proof
system or per-file SHA inventory is introduced.

````


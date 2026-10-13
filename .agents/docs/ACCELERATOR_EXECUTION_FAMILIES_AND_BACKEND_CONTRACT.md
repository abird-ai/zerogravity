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

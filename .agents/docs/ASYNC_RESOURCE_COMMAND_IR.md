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

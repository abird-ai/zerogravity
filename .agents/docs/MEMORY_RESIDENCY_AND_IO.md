# Global memory, expert residency, storage I/O and resource admission

Design requirements below. Implemented/tested scope and unfinished work are
tracked only in [the plan TODO index](../plans/TODO.md).

**15 Oct 2026 — multi-request physical memory and prefix-work accounting:** [Joint serving cost contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) makes admission count immutable source weights once, per-session KV/GDN/KDA/draft and prefix-COW state, real active + queued prefill work and actual device/pinned/OS/SSD scratch before kernels submit. Physical GPU/host caches may be shared under source/tenant-compatible leases; selected/published/Ready counters remain distinct. More speculative rows increase unique routed experts and physical byte pressure, potentially slowing output even when acceptance rises. A prefix restored from SSD is not zero-cost, and a cached expert used by encoded-but-not-yet-executed GPU commands cannot be evicted. SERVOPT-05/12..18/21/26 exercise those combinations.


**15 October 2026 — FIRST-CLASS STORAGE-BACKED EXECUTION (new user requirement):** [Unified Virtual Tensor Residency and SSD Streaming](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) elevates SSD streaming from an optional expert miss sketch into the same verified TensorView/physical BufferLease/DeviceWorker execution fabric as GPU-resident weights. One logical immutable SourceTensorId can materialize directly on GPU, mapped UMA, bounded host RAM, OS page cache or local SSD, with one global **byte-budgeted** ResourceArbiter and domain-local expert/row/frozen-state policies. Fast resident kernels incur no mandatory host/IO dispatch; SSD misses pay real bounded IO+DMA latency with asynchronous dependency overlap. Cache slots must be pinned from **GPU command encoding through actual completion**, including captured-but-unexecuted graph readers (DS4 #384); mixed Q2/Q4 size classes share ONE byte cap and prefill must not discard warm decode caches (DS4 #503). [SSD-01..SSD-32 acceptance and phase mapping](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md). Source integrity, file-descriptor identity, source bit output, UMA physical headroom, cache/QoS and actual device Ready are mandatory; no native SSD runner is claimed implemented.


**15 Oct UMA/source-loading precision addendum:** [gfx1151 and coherent-memory contract](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) distinguishes device address space/GTT max, driver reserved, host Linux MemFree, **MemAvailable**, cgroup limits/PSI, reclaimable clean source-file page cache, pinned H2D pages and physically resident model/draft/KV/SSM/lookup tables. On 128GiB-class coherent memory a buffered 80–100GiB model read can duplicate weight bytes in clean file cache before reclaim; must predict and preserve OS health reserve. Safe known-shard fadvise is optional/restricted, never global cache purge or default SIGKILL. First check physical headroom before GPU launch; model alias/source sharing and sidecars avoid duplicate resident bytes but *per-session* draft rings/states remain charged. QTIP-13 and GFX-09..11 are future physical stress gates.

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


## 1. Core architectural decision

Use **one authoritative memory-budget arbiter with specialized resource managers**. Model parameters, context state, experimental scratch, n-gram/Engram tables and disk I/O do not have the same lifecycle. A universal cache is wrong, but uncoordinated pools can fatally overcommit physical memory. This is especially critical for the 128 GB and 92/96 GB AMD Strix Halo targets.

This specification combines separately owned GPU/RAM/NVMe expert tiers, lease-safe fixed expert slabs, bounded immutable lookup-row IO, prefix-state cache tiers, hybrid snapshots and checked mapped quantized model sources. Named project provenance lives only in the prior-art atlas.

## 2. Distinguish addressability, residency and physical headroom

Measure and expose:
- installed and available physical DRAM;
- process RSS/working set, OS page cache and pinned/locked memory;
- device reported addressable VRAM, true separate VRAM, GTT/TTM and unified memory map limits;
- current HIP/CUDA/Metal allocations and driver-owned/staging bytes;
- cgroup/rlimit constraints, per-NUMA node memory and thread affinity;
- space needed for OS, file cache and untracked application load;
- host-to-device, device-to-host and peer bandwidth (possibly sharing same DRAM);
- local NVMe bandwidth, random latency and concurrency; filesystem/page cache influence.

**GPU virtual addressability is not free physical RAM.** For Strix Halo, the GPU may map over 100 GiB while `MemAvailable` and application safety margin are much smaller. Use conservative fail-closed preflight, dynamically recheck under external memory pressure and avoid mandatory unsafe firmware/boot tuning.

Illustrative budget (not fixed constants):
```text
safe_physical =
    max(0, MemAvailable_adjusted - OS_HEADROOM - safety_margin)
candidate_memory =
    mandatory_dense + mandatory_per_session_state + model_metadata
    + selected_expert_pool + IO_staging + temp_prefill + draft_weights
if candidate_memory > safe_physical OR any device limit:
    shrink optional pool / batch / context / prefill chunk in safe order
    else reject with exact shortfall
```

Use checked arithmetic and record request/lease context; never calculate with fixed total installed RAM and ignore other running processes.

## 3. Resource classes and default priorities

| Resource class | Owner / lifecycle | Demand characteristic | Placement principle |
|---|---|---|---|
| Dense model weights | model executable | read every layer/token | resident close to compute if possible |
| Shared expert and norms | model executable | active every applicable token | resident if beneficial |
| Routed MoE experts | model pool | sparse, skewed choices | dynamic cache by layer/class/cost |
| KV pages / sparse latent | session | grows with context | paged GPU then RAM/disk if valid |
| Recurrent GDN/KDA state | session | large mutable fixed shape | GPU resident active, frozen snapshots tiered |
| Sparse indexer history | session | grows/compresses with context | model-specific compressed ring/pools |
| MTP/draft weights | model option | only if activated | evaluate memory/benefit |
| Vision/audio weights | model option | sporadic | workload/budget tiered |
| N-gram/Engram table | large file | tiny row random accesses | disk primary; bounded row staging/cache |
| Prefill activation scratch | execution plan | short bursts | borrow from optional pools with leases |
| IO staging | device/IO plan | overlapping read/upload | fixed ring or slab with credits |
| Persistent prefix snapshots | prefix cache | shared immutable reads | GPU/host/disk hierarchy |

## 4. Authoritative resource arbiter

Proposed Rust interfaces:

```rust
struct ResourceEstimate {
    mandatory_bytes: u64,
    optional_bytes: u64,
    peak_scratch_bytes: u64,
    pinned_host_bytes: u64,
    device_addr_bytes: u64,
    gpu_compute_bytes: u64,
    num_sessions: u32,
}
enum ResidencyTier { Device, Unified, PinnedHost, PageableHost, FileBacked }
struct BufferLease {
    allocation_id: u64,
    generation: u64,
    tier: ResidencyTier,
    bytes: u64,
    /// physically released only after all referenced device/io events
}
trait ResourceArbiter {
    fn can_admit(&self, plan: &ResourceEstimate) -> Result<Reservation>;
    fn reserve(&mut self, plan: ResourceEstimate) -> Result<Reservation>;
    fn reclaim_optional(&mut self, target_bytes: u64) -> Result<ReclaimReport>;
}
```

These are **coarse global admission** sketches. In v4 real code, a DeviceBackend local allocator draws from an approved reservation and issues fast generation-tagged RAII leases with event-backed delayed release; the central ResourceArbiter must **not take a lock per token/kernel allocation**. Memory pressure notifies the coarse arbiter, which revises quotas/optional pools at safe boundaries. A reservation is a *cap*, not a guarantee that `malloc` cannot fail; execution must catch allocation failure and abort safely. A requested expert-cache size is not a physically guaranteed allocation.

Admission order:
1. validate artifact and desired context/modalities/quant; estimate mandatory tensor/state weights;
2. query physical/GPU topology and current use;
3. reserve OS/device headroom and mandatory buffers;
4. consider batch/session limits, context length, prefill chunk;
5. allocate remaining optional expert cache/draft/cache state based on expected benefit;
6. compile/capture plan with compatible stable pointers;
7. admit only when all mandatory leases ready; expose clearly chosen sizes and residual headroom.

On growing context or concurrency, rerun budget and shrink optional caches as legally possible; if mandatory state cannot fit, reject new request or provide explicit user-facing adjustment, not corruption.

## 5. Expert store organization

**Immutable backing source:** model file/GGUF shards, metadata offset/stride/type per `(layer, expert, projection)`, SHA/tensor manifest and optional imatrix/quant config. Read-only at runtime; page-aligned when useful.

**Mutable resident table:** `ExpertKey = (model_id, layer, expert_id, quant_layout_id)`; maps to slot `(allocation_id, generation, type/class, device addr, valid_until_event)`. GPU sees a compact current table for the model/region. The table should be updated atomically or via ordered generation publication so kernels never see new IDs with old pointers.

**Slot layout:** fixed-size class/layer slabs with deterministic padded expert sizes rather than allocate/free 11 MiB blobs per miss. Policies differ by backend: on PCIe discrete, pinned RAM can hold complementary expert set and CPU vector lane; on UMA, extra pinned copy often wastes scarce bandwidth and memory. Per-layer partitioning protects Maya-like hazards, but can reduce flexibility; measured capacity tradeoff vs pooled slots with leases.

**Residency policy:** use history (LFU/LRU/skew), miss and transfer cost, model layer/projection differences, multiple session working set and available memory. Preload known popularity profile where valid; dynamic adaptive predictions are hints not correctness. Correctness MUST NOT rely on expert being predicted. Pool can fill completely with all experts on large unified memory: avoid reserved idle spares unless needed for safe DMA replacement.

## 6. Fast resident expert path

```text
GPU router:
  route selected IDs & weights
  read device-resident slot table with generation validation
  group token rows by expert
  compute expert matrices from resident pointers
  aggregate results
  publish layer state/output event
```

No CPU copy of selected IDs, no per-layer host round trip, no global device synchronize merely to read routing. If all selected experts reside, CPU service thread can be asleep. Pointer table must be legal under captured graph replay (device-side data rather than stale baked host pointers, depending on graph design).

## 7. Expert miss path and async event protocol

```text
GPU router detects missing id(s)
  -> publish request (model/layer/IDs/expected table generation) to host-mapped ring
  -> safe dependency wait or exit current graph island
host IO service validates source + slot availability
  -> acquires lease on victim slot only after all readers/queued work are quiesced
  -> disk/pinned host read with checksum/length check (can use concurrent pread)
  -> async upload or CPU lane compute if measured superior on this topology
  -> update slot table + publish completion with release/acquire memory ordering
GPU resumes gated compute after valid completion, or aborts on failure/timeout
```

The GPU cannot spin indefinitely if host crashes or SSD fails; design bounded wait/progress counter and host-side cancellation/timeout recovery. Host and device memory ordering must be tested for HIP/CUDA specifically. Ring overflow, duplicate requests, eviction races and changing model/session epochs must be handled.

Do not over-evict active expert weights because a victim slot is being read by *another concurrent session*. Use generation leases and event completion, not only "victim from same layer" assumption, when batching across sessions.

## 8. Async SSD and NUMA

Options `mmap` + OS page cache, `pread`, `io_uring` and direct IO only after testing alignment/filesystem/queue behavior; portable fallback to bounded synchronous file reads on a worker pool is acceptable when async primitives unavailable.

For large prompt prefill, aggregate selected experts by layer and issue future reads while current layer is computing. Use multiple staged buffers only if their occupied RAM and queue depth yield net benefit. For decode, random expert misses are critical path; priority should favor current token and a known next layer.

NUMA-aware pinned memory placement and CPU pool threads can reduce PCIe costs on multi-socket machines. On Strix Halo, NUMA complexity often unnecessary; physical memory contention dominates.

Never automatically turn arbitrary model paths into network fetch operations. Downloaded model identity and checksums verified separately.

## 9. Disk-only lookup row service

Qwen n-gram and DeepSeek Engram tables are not giant MoE expert matrices. They require small indexed row reads, often BF16/native FP8 reconstruction. Data structures:
`LookupIndex{artifact_sha, tensor_offset, row_size, row_count, encoding, hash_table_schema}`
`LookupRequest{session_epoch, token_positions, row_ids, expected_generation}`
`LookupResult{verified_row_ids, staged_bytes, event}`.

Hash and row selection must match official model algorithm exactly (including history/window/padding). For repeated rows, deduplicate reads without changing output order. Use a bounded read cache sized by measured locality and physical RAM, not file size; on corruption or failed read, abort dependent transaction. Cache invalidation by model artifact/version, not file name.

DwarfStar Qwen Q2 example keeps ~95.37 GiB BF16 n-gram on disk while 41.73 GiB main/MTP weights can be resident; different model quant/quality than a closed tuned reference's lookup table. Never claim both have identical bandwidth need.

## 10. KV, prefix and recurrent state tiers

Attention KV may use page-size/batch optimized layout. Full context, sliding rings, latent compressed MLA and DeepSeek compressed history each have distinct address formula and eviction safety. Prefix cache uses immutable snapshot key and may share page memory across sessions; mutable tail COW protected. Recurrent state snapshot sizes and costs need explicit budget; offload only frozen copies, not in-flight mutable buffers. Offloaded state can often be cheaper to recompute from prefix than read from disk; scheduler compares costs.

GPU/host/disk movement is not always supported for every state type. Unsupported-tier fallback must be explicit. Never reinterpret expanded KV as latent KV. Do not evict a prefix still referenced by sibling session without COW/immutable ownership.

## 11. Dynamic scratch borrowing

Prefill may need GiB of temporary storage not used by decode. Arbiter can borrow cold optional expert cache capacity, but requires:
- prove no in-flight kernel holds victim leases;
- evict/demote optional experts safely;
- allocate bounded scratch with state generation and exact lifetime;
- run prefill;
- release scratch after GPU event;
- optionally prewarm experts for decode if justified;
- preserve mandatory state and OS headroom throughout.

Borrowing cannot be implemented as blindly mapping two overlapping buffers. Compare borrowing time/eviction cost versus smaller prefill chunk and reduced GEMM efficiency; select by measured benefit. Avoid second full scratch allocation if model can logically use tail of pool under strict exclusivity.

## 12. Safety and deployment

- default conservative memory budget, no OS-wide overcommit;
- warn on swap pressure; avoid unsafe `mlock` entire model;
- cgroup awareness and bandwidth competition;
- Linux `MemAvailable-CmaFree` adjustment where hardware kernels/CMA affect reporting; don't assume same signal on other OS;
- optional watchdog monitors physical memory threshold and aborts benchmark/load before host instability;
- protected cached prompt/session data on disk; user opts into persistence;
- pack source files immutable and checksum validated;
- model shards, staging/temp files placed under controlled paths with safe permissions.

## 13. Required test matrix

Synthetic 0/1/8/288/512 expert routes, duplicate and skewed routing, cold/hot alternation, cache sizes 0/1/half/all, forced corrupted SSD/read-only model, missing block, ABA slot reuse across concurrent requests, draft rollback during cache eviction, prefill scratch borrowing, context growth and server cancellation. Measure OOM-safe refusals. Physical tests on discrete PCIe, 92/96/128 GB UMA if available, and at least one different backend; record memory pool telemetry and clean teardown.

## 14. Known research hazards / unapproved constants

Default cache % (DwarfStar reports 80% safe in one ROCm setup), 1 GiB reserve/16 GiB headroom (Maya-specific examples), exact slot counts, pinned pool size, sparse top-k and prefill batch size are *reference configurations*, not portable zerogravity constants. Decide through measurements, hardware checks and explicit tested defaults.

## 2026-10-10 integrated v2 addendum — research-grade KV/expert algorithms governed by CIR

Adopt **research expert-tier specifics** as measured policy candidates: size-class slabs, per-layer allocations/ceilings, fill-only startup, decayed LFU+recency protections, profile-derived hotlists, device-side resident pointer table, pinned staging/O_DIRECT/io_uring vs mmap/read-ahead, one-blob expert source and host `publish` gate. Historical values such as 16 workers, 64 routes, 96 swaps, 20GiB reserve, 2ms coalescing and 3 misses are **research configurations**, not universal defaults; derive per machine profile by physical proof and source license.

**Allocator layering:** global `ResourceArbiter` controls physical GPU/UMA/CPU/OS/pinned reservations; separate `KvPool` implementations (paged baseline and contiguous extreme), expert slots, ngram/Engram disk row cache, prefix snapshots, encoder cache and scratch are all **clients**, each with domain-local eviction rules and global accounting. Radix prefix index points into the same paged pool; it must not create its own hidden allocator. The smallest valid cache frontier for hybrid models comes from `StateGroupCodec` intersection, not plain KV page count.

**CIR effect/timepoint integration:** `Acquire` and IO staging requests consume budget; `ReadArtifact` verifies immutable source identity and exact length; `Transfer` gets device event; `PublishResidency` updates slot pointer/generation only after all readers quiesce; `Dispatch` owns read leases; `Release` waits event. In a fully resident MoE decode, no per-token host route read. Miss path has bounded host ring and deadline; no GPU spin forever on IO failure. State snapshots are frozen session context and never treated as interchangeable model expert blobs.

**Policy decision:** host-visible mapped memory on Strix Halo shares LPDDR bandwidth with CPU; "promotion" from host to GPU may be a registration/alias change, **not** more physical RAM. Dynamic CPU expert lane only if measured net benefit. SSD row lookup table of 95/189GiB remains indexed file/optional tiny cache, not mandatory GPU residency. Scratch borrowing legal only when optional victim pool has no live leases and OS headroom remains valid.

**New experiment:** compare complete CIR scheduling of next-layer prefetch, resident/streamed expert grouping and prefill scratch reduction against handcoded host-stage baseline; measure critical-path latency and memory peak under cold/hot SSD and concurrent sessions. No claim of true asynchronous benefit until driver/event traces show overlap.


## 2026-10-12 integration — pinned AIE/NPU buffers and shared physical ownership

Extend global resource inventory for **XRT device/host buffer objects, GPU-mapped pinned host allocations, immutable prepacked NPU GEMM weight slices, A/B/C staging, instruction/program memory and outstanding DMA**. The Gufo fork imports `hipHostMallocMapped` pages into XRT user-pointer BO; an isolate mode copies to private XRT BO. Neither technique adds physical RAM: aliases may refer to the same LPDDR and **all pinned/duplicated allocations count toward OS-safe admission**. Shared-mapped memory requires explicit cross-API synchronization, not just a shared virtual address.

Each buffer has `(allocation, generation, owner, GPU/XRT mappings, last read/write event)` and stays pinned until both runtimes acknowledge quiescence. A timed-out NPU command can still be writing; quarantine the buffers or drain/reset correctly, never `free` based on elapsed time alone. NPU programs and weight slices may consume enough shared memory to reduce KV/expert/cache capacity, so cost planner chooses GPU-only when net TTFP/latency or safe physical memory deteriorates. The NPU can use its own program instruction BO and up to one driver-supported queue depth; do not share mutable staging inputs across active groups.

Mapped NPU buffer ownership is distinct from discrete-GPU/integrated-GPU *routed-expert ownership* and from SSD-backed n-gram tables; all coexist under one ResourceArbiter/ExecutionTopology without one universal residency policy. New `NpuBuffers` are an optional resource class, not a copy of model KV or sampler state. Early capability check can disable offload with no change to model semantics. Details in [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md).

## v4 refinement — reservation authority versus local pool

`PhysicalResourceArbiter` owns **coarse admission, peak bounds/headroom, tenant/model/session quotas and reclaim decisions**. Each `DeviceBackend` owns efficient **local suballocation** from an approved reservation: slab/page allocator, pinned buffers/IO staging and async event-backed retirement. A central mutex may be used for rare admission/rebalancing operations but **not for each emitted token/GEMM scratch tile**. Every allocation maps to an accountable reservation and BufferLease with generation, live readers and writer/completion event; optional cache eviction cannot touch active state.

Physical RAM/GTT/driver memory changes outside our control, so guarantee conservative engine admission + safe allocation failure and reclaim, **not** absolute no-OS-OOM behavior. Use local pressure watchers and release optional cache before changing mandatory task state. Imported GPU/NPU mapped buffers count physical RAM once per aliased allocation but require **both** API event/fence retirements; a timed-out DMA stays quarantined. A specialized contiguous cache can exist next to paged state pool under same global budget without a new model serving engine.

## v4.1 — weight/program residency spans GPU, graph and static dataflow

ResourceArbiter accounts **compiled program images**, graph code cache, on-chip SRAM scratch, device HBM/VRAM, shared coherent LPDDR (actual OS/headroom), pinned host staging, remote accelerator reserved quota, program warm/load cost, runtime collectives and streamed model weights independently. Native GPU kernel residency differs from **compiled graph program residency** and a **statically scheduled weight stream**; none should be forced to emulate page-by-page local KV. Static fabrics may keep compute program on chip while streaming immutable model layers, whose source, stage and transfer cost are explicit in CIR.

A `DeviceBackend` may own opaque physical handles that cannot be exported as native pointers; state snapshots must use typed `StateBridge` codec/replay or the region restricted to stateless/no-fork modes. The logical StateRoot remains authoritative and old versions unmutated before coordinator PublishRoot. A device with static on-chip memory and limited dynamic allocation requires **compiled reservation and bounded shape policy**; OS/device pressure and program upload must be gated before dispatch. Compiled device blobs and raw physical/collective handles are ephemeral and never serialized into a persistent prefix cache.

Coherent local host+GPU systems share physical RAM/bandwidth even if pointer alias works, while HBM rack GPUs and SRAM wafer arrays have different reuse/transfer costs. Reuse one hierarchy of global admission→device-local allocations and on-device compiler-owned layouts, **not** one cache policy for all. See [backend contract](ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## v4.2 — one physical accounting discipline across remote model, state and cache owners

Global ResourceArbiter is a **logical aggregate quota/admission authority**, not an all-nodes one-pointer allocator. Per-node ResourceArbiter verifies its own DRAM/HBM/UMA/pinned/SSD/OS headroom, issues local reservation tickets and fast DeviceBackend pool leases. A distributed PlacementPlan must reserve **all** mandatory per-node regions, staging and transfer credits *before* launching a multi-peer collective/state migration; workers report real pressure and independent failures. Aggregate cluster memory capacity is only capacity if the transfer topology and required state/compute semantics allow the shard.

Track four distinct owners: immutable ParameterPlacement (dense weights, experts, adapter, lookup source), mutable StatePlacement (KV/GDN/indexer), immutable CachePlacement (prefix/encoder/compiled program copies) and compute RegionPlacement, all with model/quant/tenant rights. Cache locality hints can be eventually consistent; mutable remote state version/lease cannot. A remote prefix “hit” that needs GB-sized hydrate over relayed WAN may be worse than recompute, so physical IO/crypto/RTT and latency participate in admission/cost, not just byte presence.

State and expert transfer resources have bounded staging, per-flow priorities/credits, checksum/length and lease event completion; TCP/QUIC write acknowledgment is not GPU DMA quiescence. Trusted RDMA only on physically eligible registered memory, with scoped keys revoked/fenced. A lost tunnel cannot free in-flight DMA or release authoritative state; quarantine and recover from compatible committed checkpoint if possible. [Distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md).

## v4.3 — multiple resident models under one physical ResourceArbiter

The ModelCatalog may describe very many registered immutable artifacts **without allocating GPU/system model weights**. `ResidentModel` deployment is admitted by PhysicalResourceArbiter for mandatory dense/shared/routed weights, adapter/draft/encoder, compiled program images, KV/SSM/indexer/session headroom and scratch/pinned buffers **across all selected local/remote devices**. Separate **replicas of the same semantic model** need their own actual physical bytes unless device/model source sharing is explicitly proven; one source model **sharded across devices** is one deployment, not N replicas.

Life cycle: verify→reserve→compile/load→warm→ready→idle/draining→evict after all outstanding TaskSession/DeviceWorker/StateRoot/IO/DMA generation leases are quiescent. `RegisteredModel` remains available after unload. Model prewarm/pin/keepalive/scale-to-zero policies are soft except an explicitly admitted hard pin under quota; no unlimited resident model promise and no unsafe GPU virtual GTT RAM inflation. Before unloading idle source weights, ensure any desired restorable session state has a valid frozen full-state snapshot/replay or a clear session expiration policy; **never evict an active unsnapshotable session** to serve another model.

Cold-load and model-swap cost is an explicit admission/scheduling term. Prevent thrash from alternating huge models via load dedup, hysteresis, warm TTL, prewarm priority and per-tenant fairness. Evict optional model/prefix caches and safe cold idle models before refusing, but don't silently switch source quant, trained top-k or route to another model due to OOM. A model may use multiple GPUs or authorized remote nodes, whose reservations/tenants and state/weight/cache placements are already governed by v4.2. [Router contract](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md).

# Zerogravity — First-Class Virtual Tensor Residency and High-Performance SSD Streaming

**15 Oct 2026 — cache residency × concurrency as one optimization:** [Joint serving and SLO-goodput contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) couples model/source weights, per-session recurrent/paged prefix state, routed expert hit/miss and actual SSD/host IO credits to batch/speculative verification and p95 admission. Prefetch hints never change selected experts or target math. A fully resident compatible kernel avoids per-token host IO; a cache miss legitimately blocks only its source-dependent stage and cannot evict expert slots referenced by encoded/captured/submitted GPU work. Larger MTP draft may increase expert union and lower net goodput even with higher acceptance. Serve transparent source/quant and honest memory/latency QoS receipts; no hidden re-quantization to meet deadlines.


**15 October 2026 · Governing design refinement, NOT implemented support.**
**Architecture:** v4.3 SemanticProgram → ExecutableBundle, verified ModelAssembly and SourceTensorId, one PhysicalResourceArbiter, per-device lease/event authority, one TaskCoordinator StateRoot publisher. Specialized asynchronous IO providers and cache policy clients, but NO additional inference engine.

## 0. User requirement and an exact promise

The user requires SSD streaming to be a first-class, natural capability: an immutable neural tensor can be fully GPU resident, host/unified-memory resident, cached in pinned RAM, file-backed, or asynchronously loaded on demand from SSD **without changing its mathematical identity or model operator**. Models too large to fit in GPU/CPU memory should load only needed experts, lookup rows or source-bound media components under bounded capacity and predictable performance.

**Near-zero-cost abstraction means near-zero software overhead on the fully resident path. It cannot mean zero physical latency for an SSD miss.** The resident fast path has no mandatory host routing readback, global model map lock, per-token filesystem stat, host scheduler wakeup, HTTP/JSON dispatch, dynamic allocation or source IR walk. The miss path pays actual SSD latency, source verification, staging and transfer costs, which are bounded and measured; may overlap safely with earlier layers.

Keep one model source, one tensor view, one legal source kernel and one task result. Physical residency is a property of materialization and lease, never a different model type or a silent quantization policy.

## 1. Independent identities and ownership

| Item | Owner | Exact identity/effect |
|---|---|---|
| SourceTensorId | ArtifactCatalog / verified ModelAssembly | immutable checkpoint or derived-quant SHA, component/tensor role, logical axes, source codec, transform, shape |
| TensorSlice | source tensor descriptor | bounds-checked logical byte/element subview: selected expert, projection, lookup row, image/video tile |
| SourceExtent | bounded read-only file descriptor | pinned artifact+shard ID, validated offset/length/alignment, integrity and source generation |
| ResidentSlot | physical storage manager | physical device/host address, layout, slot generation, readiness and read leases |
| TensorLease | existing BufferLease/event authority | source identity plus stable materialized slot and real dependency/fence until last GPU reader retires |
| Prepared plan | ExecutableBundle/CIR | source-qualified kernel, hit-only or miss-capable path and physical resource dependency graph |
| Mutable TaskSession | TaskCoordinator/StateDomain | KV/recurrent/draft/codec state, causal session epoch, distinct from immutable expert weights |

An immutable tensor can change backing tiers without invalidating its SourceTensorId. Requantizing weights produces a NEW derived artifact with its own numerical/quality evidence; reordering a physical layout without changing values is a separately verified lossless packing transform. A frozen session snapshot must carry state-format and session-compatibility identity, not reuse the immutable expert key.

### Illustrative Rust vocabulary — conceptual, not a frozen public ABI

~~~rust
struct TensorKey {
    artifact: VerifiedArtifactId,
    component: ComponentId,
    tensor: TensorId,
    source_codec: SourceCodecId,
    transform: Option<VerifiedTransformId>,
    geometry: TensorGeometry,
}
struct TensorView { key: TensorKey, offset: u64, size: u64, layout: LayoutId }
struct SourceExtent { artifact: VerifiedArtifactId, shard: ShardId,
    offset: u64, length: u64, alignment: u32, integrity: ExtentDigest }
enum BackingTier {
    DeviceLocal(DeviceId), UnifiedMapped(PhysicalPoolId),
    HostPinned(HostPoolId), HostPageCache(FileId), LocalSsd(SourceExtent),
}
struct ResidentSlot {
    source: TensorViewId, allocation: PhysicalAllocationId,
    generation: u64, layout: PhysicalLayoutId,
    upload_event: DeviceCompletionEvent,
    readers: PhysicalReaderLeaseSet,
}
struct ReadyTensorLease { slot: SlotId, generation: u64,
    source: TensorViewId, event: VerifiedCompletionId }
~~~

Do not implement a trait-object indirection per matrix element or media frame. These concepts are compiled/instantiated at source-loading/stage-admission boundaries, and become stable GPU-native descriptors plus already-held leases during hot execution.

## 2. Source-preserving fast and miss paths

~~~text
Exact model + immutable TensorView
      │
Precompiled source-equivalent region / legal KernelCandidate
      │
Validated resident slot directory / device pointer table
      ├── READY HIT
      │     Same GPU pointer or mapped memory address
      │     Existing generation/read lease, no OS IO
      │     Kernel launch → real event → retire lease
      │
      └── MISSING EXTENT
            Coalesced bounded request to source materializer
            Verified local SSD / host-cache read
            Exact length, source digest and codec verification
            Host staging or direct compatible mapped memory
            Async H2D transfer / verified event
            New resident slot table generation published
            Same source-equivalent kernel → real event
            Release only after every reader retires
      │
TaskCoordinator independently verifies causal state and publishes ONE StateRoot
~~~

**Resident hits:** ideally direct device address, validated source table ID or precompiled device-resident mapping. An SSD service thread can sleep. No per-layer synchronous CPU route ID copy or mandatory host callback. A prepared all-resident graph entirely omits the IO branch.

**Misses:** GPU routing may first generate real selected expert IDs into a compact device-visible list; if a missing ID occurs, yield at a validated graph-island boundary, post a bounded request ring, and resume when actual host IO and GPU DMA Ready. No unbounded GPU busy-spin or uncaught page fault. Predicted routes are hints only; routing cannot choose a different expert because one is cached.

A captured GPU graph must not hold an old raw slot pointer after reuse. Either keep all captured pointers pinned for the graph's full lifetime or dereference a correctly ordered generation-tagged device table. If capture cannot safely cope with misses, select a precompiled miss-capable eager island; no dynamic kernel compilation per token.

## 3. One residency fabric, domain-specific allocation policies

| Source data class | Default candidate tier and cost | Safe policy |
|---|---|---|
| Dense trunk, routers, layer norms, attention/shared projections | GPU/UMA when feasible; used by every step | keep resident first; slow staged whole-layer reload is opt-in capacity mode |
| Routed MoE expert matrices | SSD source + bounded GPU/host expert slab | demand selected experts, per-layout size class and adaptive cache |
| Gigantic Qwen n-gram or DeepSeek Engram table | immutable SSD file and OS page cache | tiny indexed row cache, dedup/coalesce reads; never upload all |
| Tokenizer/critical small sidecars | host/GPU resident | avoid predictable per-token SSD latency |
| Trained MTP/DFlash/vision encoders | workload-enabled component | stage residency by measured benefit; independently budget auxiliary state |
| Image/video VAE, diffusion predictors, upscalers | GPU while active, host/SSD between stages | source-verified stage/tile streaming; respect temporal overlap |
| Input audio/video files | bounded timed decoded buffers and codec IO | source media chunks, not MoE slab policy |
| Immutable committed prefix state | GPU while active; RAM/disk snapshot | tenant-private, source-compatible restart/recompute |
| Mutable active KV/GDN/KDA/mHC/codec state | device-local current step | never evict unconfirmed GPU state based only on disk write |

Different lifetimes require different **cache policies**, not separate engines: RoutedExpertCache, LookupRowCache, FrozenStateStore and MediaTileStore use the same SourceExtent, event/lease and global physical budget, but own specialized eviction and access patterns. One universal LRU for all is incorrect.

## 4. Expert source and cache: lifecycle and data layout

Key immutable expert by (model artifact and derived quant revision, layer, expert ID, gate/up/down projection group, tensor codec, physical layout). Source GGUF shards are opened by verified file descriptor with bounded exact offsets and independent source integrity, not by mutable pathname.

Expert residency lifecycle:

~~~text
Cold → Requested → Reading → Verified → Uploading/Mapping
     → READY published (generation increment)
     → Captured/Encoded reader leases
     → Queued/submitted/physical GPU readers
     → Complete (all device events retired)
     → Idle/Evictable → reused for another source, generation increment
~~~

Separate IO and GPU transfer events: SSD completion does not imply GPU upload Ready, and upload Ready does not imply all future readers quiesced.

### Mandatory correctness: lease includes ENCODED but UNEXECUTED GPU work

DS4 issue #384 isolated wrong logits under SSD expert cache pressure: an early layer's GPU command buffer was **encoded** referring to a slot; while encoding a later layer, the cache evicted/reused that slot; GPU executed the first command and read the *later* expert's bytes. The pointer was valid and CPU cache state internally consistent, so no hardware fault occurred.

**Zerogravity requirement: a slot becomes pinned as soon as a GPU command/captured graph records its reference, and stays protected until every possibly reading submitted command genuinely completes.** Merely protecting the currently encoding layer's experts is unsafe.

- All graph/batch commands pin their referenced slots across entire unexecuted, queued and submitted lifetime.
- If no safe victim, wait/flush at a legal graph island boundary, reduce staging/parallelism, use another safe backing tier or refuse with backpressure.
- No slot reuse until physical completion and generation checking; pointer-table publish uses device acquire/release ordering and actual Ready.
- Graph replay must revalidate slot generations or keep stable addresses pinned through replay.
- Unknown hardware completion means entire actual slot/physical handles quarantined; no timed release.

## 5. Mixed expert sizes and one shared byte budget

Alternating Q2/Q4/FP8/derived-trellis layers may store expert gate/down matrices with different sizes and layouts. DS4 issue #503 found a single-size CUDA cache repeatedly cleared its entire hot set on alternating mixed-quant expert sizes; a separate prefill path also threw away hot decode cache on short user prompts. The issue additionally warns that per-size-class count limits can collectively exceed real memory.

Use:
- small bounded size-class pools keyed by exact codec/shape, **one authoritative total byte reservation across ALL classes**;
- admission covers required scratch, pending staging, session KV and model nonrouted weights first;
- low-traffic class must not collapse or globally clear another class;
- prefill may use temporary independent staging or fill-only cache behavior, with evictions only after actual last reader;
- decode may access already ready slot directly, avoiding whole-expert device-to-device compact copy for every hit where legal and useful.

A high cache hit rate while repeatedly copying 2GB/token between device buffers is NOT low overhead. Test path with direct resident pointer and explicit copy path to evaluate correctness and speed.

## 6. Multi-tier IO implementation and platform reality

Default logical tiers: device-local or unified mapped, bounded pinned/hot host RAM, OS buffered file cache, SSD immutable source. Any intermediate tier can be absent. A host RAM second-level expert cache is useful when random O_DIRECT SSD access bypasses page cache; DS4 issue #629 discusses such a tier as proposed work, NOT necessarily merged production behavior.

IO provider choices (not runtime model semantics):
- Linux io_uring with alignment/capability validation and bounded submissions; fallback bounded pread worker pool.
- Linux O_DIRECT where alignment/device/filesystem and measured read size justify it; explicit compatible buffered fallback on Unsupported/invalid alignment.
- macOS dispatch IO/pread/mmap/Metal mapping under its own event and residency contract.
- Windows overlapped IO/IOCP or mapped file with exact lifetime and error handling.
- CUDA GPUDirect Storage/vendor direct SSD→GPU only after actual backend proof; no assumption that page-cache-free implies free GPU transfer.
- UMA mapped CPU/GPU handles may share one DRAM physical allocation (single physical byte charge), but contention and cache coherence remain real.
- Discrete GPU consumes host staging and PCIe/H2D budget; pinned cache doubles physical copies and cannot be called zero-copy.

Read tasks carry source offset, exact length, integrity, owner/tenant/source generation, cancel epoch, output buffer lease and deadline. Partial/torn reads and corrupted tensors never publish Ready. IO cancel cannot reclaim pinned bytes until the IO operation actually retired. Coalesce identical immutable source requests across model sessions (per-source singleflight) without sharing mutable state.

## 7. Prefill, decoding, and speculative expert reuse

### Prefill policy
Gather source-selected unique experts by layer across many token rows, coalesce reads, overlap future layers where dependencies permit, and protect existing hot decode slots. Large prefill scratch can borrow **idle** optional cache only after real event retirement. Small subsequent user turns must not indiscriminately flush all warmed expert cache. Consider shrinking legal prefill chunk rather than displacing mandatory dense weights/KV.

### Decode policy
The random expert miss often lies on critical path; keep route-hot experts warm, cost next-layer lookahead, and favor short bounded IO queues. Speculative decoding may increase distinct routed expert IDs and total candidate rows, so speculation width and batch depth must include miss probability, cache churn and storage-transfer cost, not merely model FLOPs. Prefetch hints never change mathematical router selection.

### Dynamic cache policies
- Full residency is special case with zero misses: no host IO or source-materialization callbacks in hot path.
- Across-layer dynamic hot expert cache with LFU/recency/byte cost under remaining budget.
- Fully resident prefix layers plus dynamic later experts when source stage/cost warrants. This is policy, not model type.
- Per-task adaptive cache partition/cold experts with bounded hysteresis; no unsafe eviction under queued GPU graph.
- May use host RAM as second tier, or keep source file page cache warm where OS headroom supports it.

Always reserve nonrouted trunk + mandatory state/OS health reserve before optional expert-cache bytes. Larger SSD cache is not universally faster when it displaces dense trunk/KV and forces expensive mandatory reload.

## 8. Four distinct uses of SSD under one substrate

**Routed weights:** cache expert slabs and materialize missing routed matrices as real device buffers. DS4 supports selected models/backends and cache sizes; it explicitly does NOT promise every arbitrary GGUF.

**Indexed giant lookup:** A specialized engine's large Qwen FP8 n-gram table and DS4 Qwen PLE/DeepSeek Engram table remain disk-file-backed and return only required source-registered hash rows. Use verified row layout, dedup, batched coalesced reads, OS page-cache or bounded row RAM cache. No full-table GPU residency expected.

**Frozen committed session/KV:** DS4/specialized-engine disk KV/prompt caches store source-compatible token/session state for reconnect or restart. Snapshot is mutable-derived private state with epoch/COW/compatibility and checksum, not just expert bytes; restore may require significant SSD→GPU transfer or source-compatible replay. Durable write can be asynchronous behind a committed prefix and must not block ordinary output unless durability was requested.

**Unpinned dense trunk:** One specialized engine documents disabling trunk pinning to free memory, trading substantial decode speed for memory fit and potentially 7–50s additional cold prompt reload. Treat as an explicit capacity/latency tradeoff, not a default "equally fast" tier mode.

These patterns share IO buffers, artifact identities, resource accounting and event contracts, but use distinct policies. The source model architecture stays independent.

## 9. Memory budgets and source performance cost

Physical capacity before loading:
mandatory dense/shared weights + Qwen/GLM/MiMo source state + active sessions KV/GDN/MLA/SWA + draft/vision/audio + graph/code + codec latents + output and scratch + OS/cgroup health reserve + pinned staging ring + optional expert/row caches + expected page-cache pressure. For UMA, GPU and CPU aliases to the same DRAM charged once physically, with separate budget policy; GPU virtual address max != safe RAM.

A cache count N cannot be applied independently to every expert size class. Global byte reservation and staging peaks must prevent physical driver/OOM hangs.

Latency model:
T_miss = queue + random SSD seek/IO + length/effective throughput + verification + optional dequant + host staging/mapping + GPU transfer and event wait + eviction/repair - **measured and legal overlap**.
T_hit = existing device pointer lookup + GPU compute, with no mandatory per-layer host thread.
Whole task cost is the dependency-graph **critical path**, not sum of nominal streams; expose effective hit/miss per layer and actual transfer/stall.

The scheduler should prefer fully resident when feasible automatically, then pick legal streamed placement when memory isn't sufficient, with truthful ModeReceipt: resident vs streamed ratio, effective byte budget, predicted/observed latency/QoS, cache/SSD fallback. If model will fit but be very slow, explicitly disclose; don't auto-change original quant or model identity.

## 10. Reliability, error handling and cancellation

| Event | Required result |
|---|---|
| Source extent/hash missing or invalid | pre-submission refuse, no false device fence |
| SSD short/corrupt read | cancel dependent stage, no published slot/StateRoot |
| O_DIRECT alignment or unsupported filesystem | measured buffered fallback or typed Unsupported |
| Cache full, every slot pinned in encoded GPU batch | legal wait/flush/backpressure or refuse, never overwrite |
| Pointer table ABA/reuse | generation/read lease prevents old graph seeing new expert |
| Wrong model/codec layout | exact model-specific Unsupported, no implicit requantization |
| Transfer partially submitted, canceled | retain staging/device buffer until actual completion or quarantine |
| GPU command reads streamed slot then canceled | physical lease stays until genuine quiescence |
| Slow SSD / rising p95 / OS page-cache thrash | backpressure/admission, optional quality-declared QoS downgrade only |
| Mixed quant source | byte-class-specific pool and one global budget, no cache reset |
| Slow client after output commit | bounded transport credits, no forced device buffer release |
| Disk snapshot incompatible with model/tokenizer/tenant | fail secure or authorized recompute, no foreign KV |

## 11. Verification and zero-overhead discipline

Target minimal implementation seams:
1. Verified source extent description in existing ArtifactCatalog, preserving open-descriptor lifetime and hashes.
2. Bounded materialization Future/result compatible with existing BufferLease and typed actual IO/device events, not another executor.
3. Compact physical resident pointer/generation table under existing DeviceWorker; no per-element trait objects.
4. Domain-local cache policies (expert slab, lookup rows, immutable encoder/latent, frozen state) with one global resource reservation.
5. Prepared resident-only and bounded miss-capable command variants; no dynamic IR interpretation per token.
6. Existing TaskCoordinator publishes root only after all actual IO+DMA+compute dependencies and source algorithms are proven.

Test ID SSD-01..SSD-32 in companion plan. No model/format/backend can advertise SSD streaming just because an arbitrary file path is present. Physical source test must include original trained model and full token/state parity in both fully resident and deliberately constrained cache cases.

## 12. Source and provenance references (reviewed 2026-10-15)

- DS4 SSD streaming: https://github.com/antirez/ds4/blob/main/docs/SSD_STREAMING.md
- DS4 SSD controller: https://github.com/antirez/ds4/blob/main/ds4_ssd.c
- DS4 GPU reader lifetime bug: https://github.com/antirez/ds4/issues/384
- DS4 mixed quant/cache clearing: https://github.com/antirez/ds4/issues/503
- DS4 proposed host RAM cache tier: https://github.com/antirez/ds4/issues/629
- DS4 session/KV serving: https://github.com/antirez/ds4/blob/main/docs/SERVER.md
- Specialized-engine resident trunk, lookup and disk KV (external reference removed during attribution cleanup).
- Specialized-engine detailed config (external reference removed during attribution cleanup).
- Specialized-engine quant/lookup (external reference removed during attribution cleanup).

All numerical figures and bug reports in companion research are original author/issue evidence, NOT native Zerogravity SSD performance or implementation proof. **Specified first-class architecture does not mean already implemented SSD execution.**

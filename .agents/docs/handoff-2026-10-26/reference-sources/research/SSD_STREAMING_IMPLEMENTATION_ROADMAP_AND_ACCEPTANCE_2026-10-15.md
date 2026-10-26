# Zerogravity — First-Class SSD Streaming Implementation Roadmap and Acceptance Tests

**15 October 2026 · DESIGN/PLANS ONLY, 32 planned tests, NOT run.**
**Status:** explicit user architectural requirement to treat SSD-backed model residency and streaming as first class across the same model and tensor structures; no user authorization to download enormous models or run physical memory-pressure experiments.
**Authorities:** .agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md and existing v4.3 ModelAssembly/SemanticProgram, CIR, DeviceBackend, ResourceArbiter, ModelRouter, TaskCoordinator. No model-specific disk engine, no new persistent IR.

Current implementation work is tracked in the [memory and streaming TODO](../plans/memory-and-streaming/TODO.md). This document retains the design and the SSD acceptance scenarios; their presence does not establish streamed-model support.

### Reading the design without the retired handoff

Use the [virtual tensor residency contract](../docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) for the source/materialization/lease protocol and the [source research](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md) for attributed findings. VST-USER-001 records the requirement for first-class SSD residency through the same tensor structures as RAM/GPU backing; it does not authorize large downloads, memory-pressure experiments or provider execution.

One verified source tensor binds to either an already-ready resident lease or a provisional source extent. A miss requires checked IO, admitted staging and actual upload completion before the same source kernel can consume it. Host read completion alone cannot publish a ready device slot or StateRoot. The resident path requires no per-token filesystem access, IO worker wakeup, cache lookup or hidden bulk copy; these are acceptance conditions to measure.

Keep four policies distinct over that shared substrate: routed expert slabs, bounded selected lookup rows, immutable committed session snapshots, and optional slower dense-trunk reload. A universal LRU across these lifetimes is not the design. The encoded-reader and mixed-codec regressions below remain required even when a cache reports high hit rates. Prefetch or lower storage use cannot change source routes, quantization, numerical policy or the committed result.

## 0. Implementation principle and current status

The old design already covered expert SSD miss path, disk-only n-gram/Engram rows, prefetched experts and disk KV snapshots conceptually. What needed strengthening is ONE verified immutable TensorView and source tensor identity independent of backing, clear materialization/IO/physical GPU lease contracts, zero-extra-IO resident fast path, whole-batch encoded reader pin lifetime, mixed-codec byte budgeting, IO provider portability and performance-truth admission. All are specified now; no SSD streaming path is hereby implemented in the current Rust binary.

Do not equate existence of an SSD-backed file with whole-model SSD streaming qualification. Physical backend support must be stated per exact source format, quant, checkpoint, task, region and real device.

## 1. Minimal changes to the existing engine — no new layers

- P03 ArtifactCatalog: SourceExtent and verified immutable file backing, logical TensorKey/View identity, exact tensor slice/shape/codec, file descriptor and derived quant hash.
- P05 prepared CIR/ExecutableBundle: one MaterializationEffect with ready/dependency/read-lease, and validated resident/miss-capable command alternatives; no new serialized compiler IR.
- P07 optional physical IO/DeviceBackend providers: bounded file/host IO worker and real H2D upload event, generation-tagged ResidentSlot and protected reader sets, atomic table publish; reuse device events and BufferLease instead of another lease authority.
- P10 ResourceArbiter and domain-cache clients: total OS/DRAM/VRAM/UMA/pinned/staging bytes, expert size-class cache, lookup-row cache, frozen StateSnapshot tier, per-shard source singleflight, OS page-cache headroom.
- P11/P12 exact source model operations: router, source expert and source hashed lookup access stay mathematical authority, independent of physical placement.
- P15/P16 physical miss-aware target/draft/batch cost, ready frame and safe prefetch, no hidden source routing change or cross-session mutable state sharing.
- P20/P22 tuning/validation: source original full model, forced cache miss and IO failure, real physical source parity, no uncaught memory-pressure failure, cold/warm performance and P95 QoS.
- v5.1 image/video/audio: same typed source extents, component-stage residency and media-file chunks where legal, but codec/window/finality policies distinct from expert cache. Audio source real-time QoS can REFUSE SSD placement that violates bound.

## 2. Implementation slices and gate definitions

### SSD-P0 — source-only contracts and a fake tiered executor
Define a source-verifiable logical TensorKey/View, immutable SourceExtent, BackingTier and lease-generation vocabulary. Create simulated GPU/host/SSD with controlled IO ordering (no actual source model or GPU). Test same source kernel outputs with 0/1/all resident and controlled misses; no hidden approximate data conversion. ModelRoot effect remains one owner.

**Gate:** SSD-01..04, 08 and 12: fake concurrent command queues, logical tensor identity preserved, source data exact, wrong shard/quant refused, no per-element trait dispatch.

### SSD-P1 — immutable file-backed tensor and bounded IO
Keep original ArtifactCatalog source descriptor open against same-size path swap; checked offsets, on-disk sorted expert/lookup/codec metadata, exact read length/integrity. Add a small Rust portable buffered pread worker pool; source may optionally support Linux io_uring, O_DIRECT where legal and macOS/Windows mapped/overlapped adapters, but no mandatory new OS dependencies.

**Gate:** SSD-05..07 and 17..20: partial, O_DIRECT invalid alignment, short read, simultaneous same-offset coalescing, cancellation IO lifetime and optional clean file-cache hints.

### SSD-P2 — resident GPU plus source-preserving demand materialization
On a real supported backend, bind already verified source tensor to either ready direct GPU pointer or asynchronously H2D-loaded slot, with exact transfer event and generation. A captured command must pin all expert buffers it references even before submission. On no safe victim, flush/quiesce/retry or bounded backpressure.

**Gate:** SSD-09..16 including source-real DS4 #384 encoded-not-executed regression; keep source logits/StateRoot invariant. No "Ready" from a completed host read alone.

### SSD-P3 — mixed-quant MoE streaming and nonrouted source residency
Implement small bounded per-quant-size-class expert slabs under ONE total byte budget and source route cache hotness policy, with optional resident prefix layers. Mandatory dense/attention/router/state stays resident where possible. Prefill staging can borrow only retired idle optional cache without wiping hot decode working set. On all-resident path, direct compiled kernel needs no host cache lookup.

**Gate:** SSD-13, 21..24, 26 and 29: Q2/Q4 alternating layers, cache warm after short follow-up, GPU direct-hit copies measured and honest cost, no unbounded host prefetch.

### SSD-P4 — disk-only indexed tables and frozen state
Source-specific huge Qwen PLE/DeepSeek Engram lookup row index with bounded coalescing and tiny RAM cache. Separately serialize exact committed StateDomain KV/SWA/GDN snapshots into private versioned disk cache with corruption/tenant/quant/processor and fork/ABA checks. These are separate cache policy clients over same source IO and ResourceArbiter/StateRoot.

**Gate:** SSD-25, 27, 28 and 30: row ordering, exact source hash mapping, disk KV private restore and invalid source rejection.

### SSD-P5 — adaptive placement, QoS and end-to-end integration
On 64/92/96/128GiB coherent RAM and at least one discrete GPU + NVMe machine, measure full model resident and forced streamed outputs, OS MemAvailable/PSI, driver pool, staging, cache ratios and end-to-end latency. Test multiple Qwen/GLM/DeepSeek specialist source recipes as source/weights permit; source manual approvals before 100GiB memory pressure. At model load choose direct resident when feasible, streaming if needed and source/device verified. Refuse real-time deadline that SSD misses cannot meet; never change model/quant silently.

**Gate:** SSD-31, SSD-32, entire corpus with actual source-trained model parity, memory headroom and cold/warm measured QoS. No blanket "every GGUF SSD supported" claim.

## 3. SSD-01..SSD-32 planned acceptance matrix

| ID | Independent scenario | Required source/physical safety result |
|---|---|---|
| SSD-01 | same immutable tensor in GPU, host RAM and file-only fake tier | same TensorKey/SourceId, logical math and state outputs |
| SSD-02 | fully resident model with instrumented filesystem/IO worker | zero per-token IO requests, cache worker wakeups and new allocations |
| SSD-03 | cache miss during source model region | same prepared semantic kernel after true materialization; no changed router |
| SSD-04 | file-backed tensor view with offsets and padded format | exact checked source bytes/geometry, no generic reinterpretation |
| SSD-05 | same-length atomic pathname replacement after source open | opened inode + verified model hash preserved; no mixed metadata/data |
| SSD-06 | short/torn read, corrupt model scale/codebook and negative extent | fail before GPU submit, no false Ready/Root or unrelated fence |
| SSD-07 | simultaneous sessions request same cold immutable expert | singleflight IO, both receive correct distinct reader leases |
| SSD-08 | cancel queued IO before any device submission | IO buffer lifetime valid until IO completion; no GPU fence |
| SSD-09 | CPU source exact model outputs at cache capacities 0,1,all | same model semantics, numeric comparison and accepted tokens |
| SSD-10 | GPU route hits ready slot 10K times | no mandatory CPU readback, per-token global lock, disk read or DMA |
| SSD-11 | cache miss H2D pending; CPU marks SSD read Done | slot unavailable until genuine GPU copy event Ready |
| SSD-12 | stale slot generation/pointer table ABA | old expected TensorView cannot read new resident tensor data |
| SSD-13 | model with 0, 1, 288/512 experts, long hot/cold/skewed routes | bounded pool and exact top-k routing; no guessed warm expert substitution |
| SSD-14 | DS4 #384: two-layer encoded batch, capacity one, later expert miss | earlier unexecuted GPU reader prevents slot reuse; full logits bit/source parity |
| SSD-15 | capture/graph replay with evicted pointer and stage-split fallback | stable pinned pointer or validated indirection generation, no stale address |
| SSD-16 | unknown GPU completion / cancel after upload or dispatch | physical buffers and staged IO retained until proved device quiescence |
| SSD-17 | O_DIRECT short alignment / unsupported filesystem | validated buffered fallback or typed Unsupported, no incorrect bytes |
| SSD-18 | Linux io_uring/worker pread, macOS mapped and Windows IOCP mocks | same original artifact/read/lease result, bounded queues and cancellations |
| SSD-19 | host RAM cache vs O_DIRECT/no page cache, repeated hot experts | source parity, byte budget and honest reduced NVMe misses |
| SSD-20 | concurrent IO cancellation while SSD read in flight | original host pinned IO buffer not freed before completion |
| SSD-21 | DS4 #503 alternating Q2/Q4 expert sizes across layers | bounded size-class caching without hot cache teardown or memory overcommit |
| SSD-22 | prefill after warmed decode working set | preserve hot entries when safe; scratch preflight and no unconditional flush |
| SSD-23 | 97% expert hit but hidden full packed compact copy | instrumentation catches physical copy bytes; direct hit variant compared |
| SSD-24 | cache budget N per different format sizes | one global **byte** cap prevents allocating N slots in each class beyond budget |
| SSD-25 | giant FP8 BF16 ngram/Engram lookup rows on SSD | exact source hash/row IDs, dedup and right reorder; no full table residency |
| SSD-26 | dense trunk pinned vs optional unpinned/reload | same source result but explicit latency/capacity ModeReceipt; no false zero-cost claim |
| SSD-27 | persist session KV after committed root and resume | exact compatible model/tokenizer/quant/tenant; no replay of uncommitted state |
| SSD-28 | disk KV encrypted/permissioned and source revision mismatch | reject foreign tenant/unsupported state; proper error/recompute policy |
| SSD-29 | Qwen target+DFlash/MTP and shared warm experts across 4 sessions | correct expert union, draft/target cache state and model SessionRoot isolation |
| SSD-30 | source VAE/video-tile/audio frontend offloaded by legal stage | exact frame/sample clock/codec output, no giant unsafely resident intermediate |
| SSD-31 | coherent 128GiB with hot 80–100GiB source page cache and active KV | admission keeps OS health reserve; no OOM killer/page-cache thrash |
| SSD-32 | full original trained MoE model SSD streamed vs fully resident on physical backend | complete logits/StateRoot source parity, cold/warm TTFT, P95, memory and actual IO receipts |

**Every SSD-XX above is PLANNED / NOT EXECUTED.** Source-free fake IO tests are not original model GPU qualification; actual hardware support must remain per device+model+quant.

## 4. Required negative-fixture detail — DS4 #384 / #503

### Encoded GPU use-after-slot-reuse (SSD-14, SSD-15)
- Construct two layers A and B, one reusable expert slab S, immutable source expert X then different expert Y.
- Encode A reading X from S, without submitting or waiting; start encoding B and request Y; allocator must see A's **encoded reader lease** and refuse to reclaim S.
- Only when A physically finishes may S be reused for Y; publisher generation must change and no A stale pointer may read Y.
- Compare full source tensor logits and state; simple CPU directory/key check is insufficient because wrong bytes produce valid GPU completion.
- Exercise queued, submitted, captured graph and cancellation of encoded-but-unsubmitted command. No timed release.

### Mixed-format thrash (SSD-21/22/24)
- Q2 expert gate/down 7MiB, Q4 expert 14MiB, alternate A-B-A-B across layers, maintain one global 64MiB byte budget.
- Per-class pools may persist simultaneously; no group switch clears unrelated source residency, and both pools combined never exceed 64MiB plus separately accounted mandatory scratch.
- Start short prefill on warm decoder; no unconditional cache purge; staging may evict truly idle optional victims only if it cannot fit, and must record chosen policy.
- Real device copy bytes on cache hits tracked; an extra 2GB/token device-to-device copy with 97% hits fails a zero-extra-IO claim even though source numerics pass.

## 5. Admission/placement and high-performance UX

Default profile: Auto, source identity and exact numerical mode unchanged, resident first when mandatory weights+all active state+OS reserve fit; otherwise source-qualified dynamic tiered mode where available, or typed Unsupported/InsufficientResources. Don't expose an additional user-facing model alias "ssd_model" unless a distinct quant/artifact is actually chosen.

Advanced optional controls (design proposals, not flags yet):
- operator maximum host/device memory and minimum OS headroom,
- expert cache byte allowance, pinned prefix/host slab, read queue depth,
- prefetch lookahead bytes/depth and chosen SSD buffered/direct tier,
- QoS class: throughput/batch, interactive, soft-realtime (admit/refuse based on measured miss),
- model/task cache warmup only when verified safe.

User-visible summary must say exact model, source/derived quant, resident mandatory bytes, effective optional GPU/host SSD cache bytes, source file table bytes, active state budget, estimated cold miss and expected QoS; don't claim SSD access itself is zero latency or use driver GTT addressability as free DRAM.

## 6. Benchmark and evidence reporting

Fields: SourceTensorId+artifact hash, size/quant classes, device driver/compiler/ISA, filesystem/SSD model+protocol, physical topology, OS MemAvailable/cgroup/PSI, initial cache hotness and expected lookup/expert routes, route entropy/unique expert count, prefill/decode/draft accepted tokens, actual IO bytes/ops/latency and staging/H2D bytes, cache hits/misses and evictions, protected encoded+GPU reader leases, pagecache or O_DIRECT mode, p50/p95/p99 TTFT and decode, Qwen/GLM/DeepSeek complete logits/StateRoot source parity and full startup/teardown resources.

Do **not** promote upstream DS4/prior-art author numbers into local performance promises. Compare independent resident, pagecache-warm, cold SSD, host-tier hot and mixed-quant modalities with equivalent full model/numeric task. Document negative/neutral experiments for later agents.

## 7. Dependency and next engineering plan

This roadmap is additive to original Zerogravity P00–P23, and existing Qwen3.5 0.8B/2B CPU→physical GPU followed by Gemma4/Qwen3.8 order is unchanged. Do not start 200GiB downloads before approved test/hardware/resources.

Ownership:
- Artifact source owner: P03, source-extents and schema.
- CIR/Device owner: P05/P07, IO/transfer/slot lease/graph correctness.
- Resource/storage owner: P10, bytes/headroom/cache and safe immutable sharing.
- Model math owner: P11–P14, source-exact routing and lookup.
- Draft and batching owner: P15/P16, acceptance, IO-aware expert union and QoS.
- Perf/release gate owner: P20/P22, physical original model source+telemetry.
- Realtime media owner: v5.1 TaskDriver and serving contracts; stage-specific offload cannot corrupt playable frontiers.

Other agents may be working in repository; use precise owned paths, no unrelated staging/stashes. Test via source-free mocks then original checkpoint CPU, then physical GPU, then measured SSD stress only on authorized machine. A separate SSD engine is a design failure.

## 8. User intent and design decisions

**Binding user instruction, 2026-10-15:** fix documentation archive generator cleanly and make SSD streaming natural, first-class, highly performant, reusable with GPU/RAM/SSD backing via similar structures. User authorized docs/plans and necessary minimal documentation tooling, not implementing a new native SSD inference engine or downloading large model weights.

**Design proposals:**
- SSDD-001: Same immutable logical TensorView across tiers, ready lease in existing BufferLease/CIR.
- SSDD-002: Resident direct fast path has effectively zero per-token software IO overhead; SSD miss paid honestly.
- SSDD-003: Expert, lookup and frozen state share physical/IO substrate but have separate eviction/finality policies.
- SSDD-004: Full encoded/captured/submitted GPU reader set pins slot before eviction; generation ABA test.
- SSDD-005: Mixed quant size classes share one global byte cap; preserve hot decode cache across prefill.
- SSDD-006: Physical OS/driver headroom before cache; direct IO and mmap chosen by measured platform.
- SSDD-007: Source route and quant/numeric identity never change due to cache status.
- SSDD-008: Committed StateSnapshot disk persistence private, source/version compatible and independent of weight cache.
- SSDD-009: Auto resident first, stream when required/qualified, with QoS/cold/warm mode receipt; no change of target model.
- SSDD-010: Same model and API independent of backing, no additional storage inference engine.

**OPEN:** which cache/popularity and IO algorithm performs best on each actual target; default minimum physical OS headroom; pressure stress authorization; optional GDS/remote storage policy; disk KV retention/encryption; mandatory realtime QoS; source model/backend/quant streaming qualification. These need measurements and/or explicit authorization and must not be filled with DS4/prior-art-specific constants.

## 9. References and implementation boundary

Use the [current plan](../plans/memory-and-streaming/TODO.md) and [repository workflow](../../AGENTS.md) for an assigned implementation slice. The [virtual residency contract](../docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md), [source research](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md), [physical memory design](../docs/MEMORY_RESIDENCY_AND_IO.md), [command/event contract](../docs/ASYNC_RESOURCE_COMMAND_IR.md) and [state transaction contract](../docs/STATE_CACHE_AND_TRANSACTIONS.md) supply the detailed technical requirements. Start with the bounded source-free scenarios, then original checkpoint CPU, actual GPU and authorized SSD-pressure gates. Mock IO cannot establish streamed-model or physical-lifetime qualification.

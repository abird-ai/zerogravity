# SSD Streaming Research Dossier — DS4/DwarfStar and related offload prior art

**15 October 2026 • Source-grounded design research, no Zerogravity SSD model execution.**
**Question:** how to make SSD-backed weights and state *naturally* available inside one native Rust tensor/resource fabric without sacrificing the fully resident fast path or GPU lifetime correctness?
**Answer:** bind exact source tensor identity independently of physical residency; source-specific expert/lookup/session storage policies share asynchronous IO, device event/lease safety, bounded physical memory and compiler prepared execution. Resident decoding becomes a zero-extra-IO specialization, while SSD streaming is an explicitly costed and verifiable materialization path.
**Governing design:** .agents/docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md.

## 1. Distinguish four source-observed techniques

1. **Dynamic routed expert weights**: DS4 pages missing experts from immutable GGUF source and caches hot experts in bounded GPU memory. It is not generic stream-every-weight prefill.
2. **File-backed sparse row lookup**: an independent file-backed Qwen 47.7GiB n-gram table and DS4 95GiB Qwen PLE/189GiB DeepSeek Engram may remain file-backed and fetch small selected rows; not a full tensor H2D copy.
3. **Persisted session/KV checkpoints**: both engines can save compatible committed conversation/prefix state to SSD and restore useful prefixes. Different mutability and tenant/lifetime rules from model weights.
4. **Unpinned dense trunk as capacity fallback**: the file-backed lookup design documents a model-trunk unpin option that frees memory but substantially lowers decode and adds cold reload latency. This must be opt-in/capacity tier, not a zero-cost equivalent.

These can be implemented over one SourceExtent/BackingTier/ReadyTensorLease and typed StateSnapshot substrate, but require **different caching and finality policies**.

## 2. Antirez DS4 / DwarfStar streaming

Official docs:
- https://github.com/antirez/ds4
- https://github.com/antirez/ds4/blob/main/docs/SSD_STREAMING.md
- https://github.com/antirez/ds4/blob/main/ds4_ssd.c
- https://github.com/antirez/ds4/blob/main/docs/SERVER.md

DS4 distinguishes fully resident from explicitly enabled SSD streaming, and says resident is faster when feasible. Its SSD mode reserves enough memory for nonrouted weights, attention KV/recurrent state, graph and prefill transient allocations, and spends only remaining budget on dynamic expert slots.

- Automatic cache byte budget and bounded expert count are requested capacities; effective cache may be lower because mandatory model/OS/graph memory comes first.
- Source backend/model coverage differs: Metal DeepSeek/GLM, CUDA routes and ROCm GLM5.2/5.3 (at documented version) are not proof every GGUF/quant streaming works.
- Optional full routed prefix layers are allocated first; dynamic cache otherwise may range across all layers.
- A larger cache can displace nonrouted mandatory weights or session KV on unified memory, slowing decode despite more expert hits.
- It preloads experts under supported paths; cold/preload controls are benchmarking tools, not universal speed defaults.
- Streaming speed is sensitive to cache hit rate, random SSD IOPS, verified expert byte group, staging and target verification width.

DS4's publicly reported M5 Max 128GB measurements (September 6 2026; author setup, **not Zerogravity**): GLM5.3 Flash Q4_K file 177.77GiB; its SSD-streamed initial prefill 121 tokens/s, continued prefill 104 tokens/s, generation after two frontiers 11.9/14.9 tokens/s. DeepSeek Flash Vision Exp 145.26GiB: initial 300, continued 263, generation 11.9/19.3 tokens/s in a single run. These are source workload-specific, not general SSD/target performance promises.

### 2.1 DS4 issue #384: encoded-but-not-executed GPU command uses evicted expert slab

https://github.com/antirez/ds4/issues/384

This is the most valuable correctness negative test for Zerogravity. DS4 recorded a deterministic argmax flip at constrained 16–20GiB cache in Metal layer-batched decode. The CPU producer encoded ALL layer commands first; a later layer's missing expert caused a prior layer's unsubmitted slot to be overwritten before its GPU command actually ran. The GPU pointer remained valid, source CPU cache directory consistent and GPU completion truthful, but **the bytes were a different expert**. Prefill didn't show the issue.

Required architecture:
- mark expert slab reader lease as soon as a GPU command records/reference/captures that slot, not just when it is submitted or executing;
- protect the full set referenced by **every command in the currently unexecuted batch**;
- combine slot generation and physical event lease, stable pointers or validated indirection table;
- do not recycle until GPU completion proves quiescence;
- when no eviction victim exists, safely flush/await current batch or return backpressure; never overwrite a queued GPU reader;
- this is a test with two layers, cache capacity 1, encoded batch + later miss, so same reuse bug cannot hide behind CPU source checks.

The DS4 issue was observed on one configuration; read its resolution/source revision when porting but keep the invariant regardless of vendor.

### 2.2 DS4 issue #503: mixed-quant cache reset and prefill evicting decode

https://github.com/antirez/ds4/issues/503

On a mixed Q2/Q4 model, an original CUDA expert cache used a single expert-size assumption and cleared all slots when gate/down bytes changed, leading to ~0.9 tokens/s and repeated expensive reset. Another prefill path released all warmed decode cache for every prompt, even short tool-result followups.

Reported patch proposal: per-size-class slabs keyed by gate/down bytes, try staging allocation before eviction, use existing warm cache for prefill, and avoid evicting decode working set in long prefill. A second hazard was discovered: per-class N slots can overcommit when **every** class independently requests N. Correct design is **one authoritative byte budget across heterogeneous classes** with reserved arena/scratch and per-class weights as secondary policy.

Additional performance ceiling: a 97% hit rate could still copy ~2GB/token device-to-device into a compact buffer. The goal should be direct source-valid resident-pointer consumption when legal, not merely high cache-hit count.

Do not claim the issue's proposed patch was merged or that the source author measurements are Zerogravity results.

### 2.3 DS4 issue #629: optional pinned host-RAM second tier

https://github.com/antirez/ds4/issues/629

Research/work-in-progress proposed copying source expert bytes into bounded host RAM slabs as a second-level cache. The published CUDA path used O_DIRECT for SSD reads, and on buffered fallback advised cache eviction, so ordinary OS page cache wasn't retaining useful expert hotness. On a host tier hit, stream from RAM to GPU without random SSD IO.

Lessons:
- host cache is distinct from OS page cache and requires real RAM/pinned byte admission;
- on Strix Halo UMA, RAM+GPU may alias DRAM, and a redundant pinned duplicate could hurt memory efficiency;
- on discrete GPUs or direct-IO systems, dedicated hot RAM slabs may be economically useful;
- size by observed hot routes and global physical byte credits, not every class allocating a full budget;
- host cache fill/read must never overwrite a pinned IO or GPU reader buffer prematurely.
- Treat issue status as a source idea/benchmark, not guaranteed merged functionality.

### 2.4 DS4 disk KV session cache and direct-file lookups

https://github.com/antirez/ds4/blob/main/docs/SERVER.md

DS4 uses disk prefix/KV to avoid expensive refilling when clients resend shared conversation history, and keeps source static lookups file-backed for models such as Qwen3.8 and DeepSeek Engram. This is **not identical to expert streaming**:
- KV state is private, mutable, session-versioned and compatibly snapshot after StateRoot publication.
- Immutable lookup rows are model-specific hashed addresses from a verified large table, indexed by source token history and row schema.
- On distributed execution, disk prefix restore may require recomputing/redistributing state on each rank rather than pointer restoration, and numerical mode can differ.
- Disk persistence requires explicit rights/tenant scope and source/processor/kernel state compatibility to prevent cross-user history disclosure.

## 3. File-backed disk-only lookup and optional dense reload, not a blanket SSD-expert engine

Source documentation:

A prior-art file-backed design emphasizes keeping the tuned Qwen3.8 Flash-Next model **trunk pinned/resident** on gfx1151 to preserve decode speed. Its distinct 47.7GiB FP8 n-gram source table remains file-backed and is read on demand with bounded indexed rows; a cold 32K prompt may read some rows from SSD when OS page cache is cold, then subsequent hits benefit from file cache.

A setting to unpin the model trunk deliberately exchanges capacity for dramatic decode slowdown and extra per-prompt layer reload. The documented example contrasts ~37 tokens/s pinned with ~6 tokens/s unpinned and cold prompt weight reloads 7s warm file cache versus 25–50s after disk eviction. This is a **capacity-over-latency** mode, not a universal transparent equivalent of resident GPU memory.

The same design also separately supports persistent disk prompt/KV cache with incremental committed state snapshots. It records model/weights/config-specific disk directories and limits (historically default 64GiB disk cache). This is a third lifecycle (session state), not another immutable tensor expert slab.

Its memory discussion notes that the n-gram source table relies on OS cache and that host reserve and KV pool compete with that cache. The correct global planner considers mandatory trunk, session pool, prefix state, reclaimed page cache, physical GPU allocations and OS reserve, not a simplistic GPU MemFree number.

Another optimization in that design: in-place prompt snapshots can avoid copying entire 20+GiB KV state when only small position-free state changes. This is an effect/compatibility optimization under committed StateRoot, not permission to make raw GPU pointer snapshots durable.

## 4. Prior offloading system: one system across GPU, RAM and disk, optimize task regime

Background paper: https://proceedings.mlr.press/v202/sheng23a.html

A 2023 offloading system used block scheduling/offloading and compression to run enormous dense models on limited GPUs, optimizing **throughput-oriented large-batch** workloads via LP policy across GPU/CPU/disk rather than low-latency single-user interaction. Its paper explicitly warns that SSD offloading under small-batch interactive workloads can be much slower than fully resident GPU.

Zerogravity should incorporate the **latency/throughput regime distinction and cost-based placement**, not replicate a new offload engine or giant per-token LP solver. Model load/plan may solve a coarse constrained placement; hot path follows precompiled resident/demand materialization candidates.

## 5. Activation-aware expert caching and source tracing

A known activation-aware design demonstrates routed expert offload across SSD and host RAM using activation-aware expert hotness, tracing/prefetching and multi-GPU per-device caches. Its current open-source edition differs from paper target and may expose unsupported/no-op features depending on checkpoint and backend. Treat its code as a policy experiment, not a guarantee every model/quant/prefix KV offload works.

Design lesson: source router computes expert IDs; cache hotness/traces are **hints** for physical residency only. Model math remains unchanged. Multi-GPU uses per-device expert slot maps with one physical-budget authority and real cross-device/peer transfer receipts; remote network storage is separate explicit authorization.

## 6. Source comparison and exact gaps closed

| Capability | DS4 | File-backed lookup | Offload project | Expert-offload project | Required Zerogravity |
|---|---|---|---|---|---|
| Fully resident fast path | yes | explicit primary | possible | possible | direct kernel pointer, no IO overhead |
| SSD expert demand | supported subsets | mostly unpin trunk (different tradeoff) | layer/weight offload | expert offload | same TensorView, typed asynchronous materialization |
| On-disk sparse lookup | Qwen/DeepSeek source tables | huge FP8 n-gram | model-dependent | model-dependent | source-specific row cache, small IO |
| Disk KV snapshot | server cache | persistent prompt state | KV offload | configuration-dependent | frozen versioned StateDomain, private |
| Mixed quant cache | bug case issue #503 | source 4-bit quant variant | low-bit compression | per-device policy | size class, shared bytes, no thrash |
| Reader protection | bug case issue #384 | pinned trunk | stage block schedule | async transfers | pin from encoding through GPU event |
| Host RAM tier | proposed issue #629 | OS file cache primary | planned multi-tier | yes | optional RAM cache for source hotness |
| Automatic capacity | cache constrained by other state | host reserve/KV pool | plan optimizer | dynamic experts | conservative physical+OS admission |
| Interactive service | cache-sensitive | optimized resident | weaker | backend/task specific | truthfully measured QoS and refusal |
| Dedicated engine | narrow C source | gfx1151-specific | Python research | Python tensor framework | NO; one Rust engine |

## 7. Numerical source and performance honesty

A fully resident model and its SSD-streamed twin should produce identical source-exact output given identical model/quant/numeric policy and state. But if device layouts/dtype/reduction order change with batch/route/IO mode, numerical equivalence requires separate source-based validation and ModeReceipt. GPU event Ready only proves command completion, not that an evicted expert slot contained the expected bytes (DS4 #384); original model logits/hidden-state tests under extreme cache pressure are mandatory.

Include cold/warm SSD, random vs sequential IO, page-cache hotness, host staged vs direct SSD, source extents, cache shape, active model count and session count in performance evidence. A model that fits on SSD may only generate a few tokens/s; that is a valid capacity feature with a truthful slowdown, not a failed abstraction.

## 8. Strictly not adopted

- No separate DS4 or other prior-art offload engine or Python wrapper in Zerogravity runtime.
- No default force of all model tensors onto SSD, no arbitrary GPU page fault inside unsupported kernel, no unbounded per-token IO worker or global source scans.
- No fixed magic 80% cache reservation from one ROCm result; memory/OS headroom is measured/conservative and workload specific.
- No automatic process killer or OS page-cache purge based on prior-art anecdote; optional known-source hints require explicit authorization.
- No assumption unpinning model trunk is performance-equivalent or that persisted KV can resume arbitrary different model/quant/session.
- No copying an upstream issue patch or library without license/source review and actual hardware/quality evidence.
- No claim implemented because old CIR spec mentions NVMe; each source tensor/backing/device combination must pass its own acceptance.

## 9. Upstream provenance and exact time

The publicly available DS4 docs and GitHub issues, the file-backed lookup project documentation, the offload paper and the expert-offload source were inspected on 2026-10-15. Source issues represent reproducible failure modes and improvement suggestions; whether a particular fix has been merged is not asserted from its issue alone. Author measured throughput, cache sizes, input contexts and device versions cannot be transferred to the user's Strix Halo or a Zerogravity GPU proof without independent reproduction.

This dossier supports the single unified virtual-residency design and its SSD-01..32 planned acceptance suite; **it is not an SSD-streamed Zerogravity model execution report**.

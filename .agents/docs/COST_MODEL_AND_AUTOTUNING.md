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

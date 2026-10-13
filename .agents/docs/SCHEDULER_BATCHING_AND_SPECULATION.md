# Scheduler, continuous batching, prefix reuse and speculative execution

**15 Oct 2026 — precise concurrent-serving requirements:** [Joint serving, goodput and combinatory specialization](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) clarifies unified needed-versus-computed task frontiers, bounded per-step token/verification/media work, decode/duplex latency protection **and prefill starvation prevention**, separate queued requests/queued prefill tokens/active sequences and resource credits. P15 MTP/DFlash target proposals and P16 source/quant-compatible batch plans jointly account expert-row padding, unique routes, cache residency and full rollback—not batch or draft depth alone. A small **prequalified Pareto candidate set** is chosen at safe boundaries with hysteresis; default target-only/reference path remains. [SERVOPT-01..36 future tests](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md). vLLM already supports continuous batching, hybrid KV and disaggregation; these requirements aim to compete using demonstrated *joint* optimization, not assert feature novelty.


**15 Oct speculation/batching precision addendum:** [exact MTP/DFlash source draft contract](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) defines TrainedMTP/DFlash/lookup proposal-specific attention, tap, mask and mutable state; target alone verifies and commits accepted prefix, including stochastic sampler, grammar and full GDN/KDA/MLA index rollback. Work rows equal sum of (1+each draft depth), but actual cost includes grouped expert quant kernel tile/padding, unique routed experts and per-session state. Policy may back off to **target-only** on low acceptance without changing model/quant. Distinct batch B and near-tie numerical modes require explicit receipts; benchmark interactive P95 separately from aggregate committed throughput. HYB-06..18 and GFX-07/08 qualify this contract.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**Architectural core:** One request scheduler controls token budgets, priority, admission and complete state commit; one device-plan executor runs fused kernels; one global ResourceArbiter reserves physical memory. The original runtime scheduler jointly considers token work, prefix reuse, routed expert/lookup IO, physical memory, speculative verification and task fairness. No guarantee that complex scheduling is superior without measured thresholds.

## 1. Entities and lifecycle

```text
Incoming API request
  -> parse/render/tokenize & validate model/modality/sampling
  -> lookup compatible live/durable prefix
  -> estimate per-session mandatory state/scratch budget
  -> WAITING (may cancel/expire)
  -> ADMITTED (reservation and state fork)
  -> PREFILLING (chunks) <--> RUNNING/DECODING
  -> optionally DRAFTING/VERIFYING
  -> commit verified frontier and STREAM emitted tokens
  -> COMPLETED | CANCELLED | FAILED
  -> release or snapshot state by cache policy
```

A request may have pending prompt tokens while older requests decode; no exclusive "prefill engine". Scheduler prepares execution groups, chooses plan variant, preleases memory, dispatches async GPU work and processes completed events. GPU execution should proceed with no host decision between resident MoE ops within a captured/specialized region.

### Request ledger fields

```rust
struct Request {
    id: RequestId,
    model_id: ModelId,
    queue_class: QueueClass,
    s_lo: LatencyPolicy,
    input_tokens: Vec<TokenId>,
    committed_frontier: u64,
    computed_prefix: u64,
    max_new_tokens: u32,
    sampling: SamplingContract,
    mode: WorkMode, // prefill, decode, verify
    session: SessionId,
    cancellation_epoch: u64,
    deadline: Option<Instant>,
    resource_reservation: Option<ReservationId>,
}
```

Code sketch only. Request data and state access must avoid copying massive token histories on each schedule.

## 2. Unified token-budget scheduling

Given global capacity per cycle (tokens, running sessions, launch budget, memory scratch, device resources), choose currently running requests and new admission in one bounded pass. For each request compute required new tokens from `pending_prefix_end - computed_frontier`; allocate up to token budget and candidate plan's supported row shape; established decode requests usually require 1 new target token or a verify batch.

Chunked prefill is a service fairness mechanism and kernel-shape optimization simultaneously. A fixed 128/256 token quantum may be good for latency but bad for large GEMM utilization; choose per hardware/workload mode, profile P95 and throughput, and retain bounded maximum waiting time.

Scheduling policy must consider:
- per-model tokenizer/encoder work and modality;
- prefix cache hit probability and compatible state availability;
- per-request priority/deadline and fairness;
- model active weights and context/snapshot residency;
- GPU compute critical-path time, queues and outstanding copy/IO;
- token rows available for grouped MoE;
- scratch and per-session KV/recurrent state budget;
- speculative verification rows and acceptance;
- whether changing plan requires state-layout conversion (cost + safety).

### Baseline fairness and admission

Use stable waiting/running queues with aging/priority and optional per-tenant quotas. Running requests must not be perpetually starved by fresh long prompts; new requests cannot evict mandatory running session buffers without safe preemption/checkpoint. Out-of-memory admission fails with actionable budget report; do not preempt by corrupting sessions.

## 3. Execution groups

A group has compatible device model, kernel plan, precision/quant, operator-region state contract and batch rows. Several sessions can share immutable weights but retain private state. Mix:
- initial/continued prefill with token rows;
- target decode with one row per sequence;
- speculative verification with several rows per sequence;
- attention/recurrent state private by sequence and position;
- routed MoE expert grouping after device router selects IDs.

Do not force every row shape into same GEMM kernel. Use separate prefill, small-batch decode and verifier plans as justified by measured performance.

## 4. Route-aware MoE execution and cost

At layer L, each sequence token selects `top_k` experts. A grouped kernel can reuse expert weights across all rows routed to the same expert. Example four Qwen/GLM sequences may yield only ~15 distinct expert IDs for 32 route assignments; the actual distribution may be much less skewed and grouping may cost more than it saves.

**Mechanism:**
1. GPU route operations produce ids, gates, numerical scores.
2. GPU builds compact assignment list `(expert, session row, gate)`; optionally radix/count sort by expert within a bounded scratch and occupancy constraint.
3. Device-side table maps expert to resident slot or miss event. Missing IDs batched without stale pointer usage.
4. Resident grouped expert computation processes rows in appropriate MMVQ/WMMA tiles; layer's shared expert may overlap legally.
5. Scatter/sum to original row according to numeric contract; update layer state.
6. Profiling records distinct experts, group occupancy, byte reuse, misses, scatter and launch overhead.

**When not to group:** single row, too many unique experts, small workload, little overlap, expensive scratch or numerics requiring strict fixed reduction not achievable efficiently. Choose simple per-row path.

**Important:** scheduler generally cannot know route IDs ahead of previous layer activations. Memory decisions can use expert popularity as hints, but correctness requires on-device routing and miss handling; no blocking host lookup every token.

## 5. Prefix-aware scheduling

A token-prefix index supplies computed token length and exact compatible state snapshot identity. Rank new requests partly by reusable state and available memory only where fairness allows. A 40K shared prefix may justify admitting a request without 40K recomputation; if state is offloaded to disk and takes longer than replay, choose recomputation based on cost.

Prevent cache hoarding by low-value/one-off prefixes. Eviction score can include future reuse, state bytes, prefix length, reconstruct time, branches and disk IO. Keep LRU/LFU details local to PrefixCache; scheduler consumes estimated cost and handles memory reservations.

## 6. Speculative decoding architecture

A **Drafter** proposes token IDs but does not authorize them. A **TargetVerifier** runs the authoritative model on proposals, returns logits and provisional state snapshots. A **Sampler** chooses accepted prefix/correction under declared numerical/sampling policy. A **Controller** chooses drafter and draft length based on measured net benefit.

Drafters:
- native MTP (Qwen/GLM) with model-matched weights;
- DSpark for compatible DeepSeek version (support checkpoint verified);
- EAGLE-like trained drafter if supported;
- small independent auxiliary model;
- prompt-lookup/copied spans/n-gram suggestions;
- disabled.

No two proposals should be combined without an algorithm with correct target verification. If drafter weights unavailable or incompatible, disable rather than invent.

### Exact sampling reminder

For target distribution `p(t)` and proposal distribution `q(t)`, stochastic speculative sampling requires acceptance/correction rules that preserve `p`; a common one-draft form accepts candidate `x ~ q` with probability `min(1,p(x)/q(x))`, and upon rejection samples from normalized positive residual `max(p-q,0)`. Multi-token/tree variants additionally require careful conditioning and RNG handling. Deterministic greedy mode validates against target argmax within selected numerical plan, not necessarily bitwise cross-kernel equality.

The exact algorithm and FP32/FP64 probability arithmetic must be independently reference tested. **DwarfStar's default opportunistic sampled mode changes target sampling distribution**; zerogravity proposal is exact mode by default with explicit approximation opt-in, pending user approval.

### Transaction for verify block

```text
take snapshot of all required state domains at frontier p
draft up to d tokens; optional prompt lookup
run target forward over d proposals in one state-provisional region
compute acceptance/correction and valid a
commit all KV/GDN/KDA/indexer/history/RNG at chosen a
discard rejected state versions, reclaim scratch, emit accepted/corrected output
maybe adjust drafter policy at request/step boundary
```

Correctness for stop/EOS/structured output/tool-call boundary: stop may occur inside verified block. Accept only up to actual semantic boundary, roll back rest; if sampling mode changes, restore appropriate pre-verify snapshot and re-evaluate after boundary rather than pretending the draft has same policy.

## 7. Speculation cost model and guardrails

Estimated benefit:
```text
committed_tps =
    E[useful committed tokens per cycle] /
    E[draft_compute + verify_compute + state_snapshot/rollback + memory/IO +
      scheduler overhead and critical-path non-overlapped time]
```

Use actual measured critical-path time, not sum where stages overlap. Controller policies: initialize conservative, sample acceptance for small number of cycles, expand/shrink depth gradually with hysteresis, bypass on poor acceptance or memory pressure, re-evaluate at context or workload transitions; do not disable speculation based on one noisy token. Minimize per-token controller CPU work and toggles that invalidate graph captures.

An extreme kernel plan can optimize verification groups using MoE weight reuse and shared projection data, but must not change sampler semantics. Draft weights compete with expert/prefix caches for physical RAM, so speculation is a resource-policy decision too.

## 8. Active batch vs queued concurrency

Explicit distinction:
- **HTTP concurrency:** multiple incoming requests being accepted/queued.
- **Session concurrency:** multiple live state buffers present.
- **Native GPU batch:** multiple independent rows computed together, potentially reusing weights.
- **Throughput:** aggregate committed token rate, with P50/P95 per-user latency.

Some source projects report concurrent sessions but execute sequentially; zerogravity must label `ordered-fallback` distinctly from `native-batched`.

## 9. Multi-model hosting and eviction

User goal implies running many model families as options; not necessarily resident simultaneously on 96 GB Strix Halo. Model manager can unload/evict model-specific immutable weights when inactive, using reference-counted session frontiers and optional checkpoint/replay. Do not evict a model that still has in-flight kernels or live sessions without explicit admission policy. Cold switching time and downloaded model disk size must be visible to the user.

## 10. Distributed scheduling and prefill/decode disaggregation

Optional P/D split and remote expert parallel when network and throughput justify transfers. Every request includes model/plan/quant and committed frontier identity; remote workers may not invent independent stochastic samples or proceed after a lost collective. Scheduler sizes network and memory budgets, and can keep local single-device fast path free from transport overhead.

## 11. Cancellation, priority and stop conditions

Scheduler assigns cancellation epochs. A request cancelled in WAITING simply releases admission estimate; ACTIVE request marks pending transaction invalid, waits for current kernel/IO events to settle, emits already committed tokens only, releases state or persists valid prefix if policy. Stop sequences apply to committed token stream; never emit speculative drafts. Slow streaming clients have bounded queues and can cause request cancellation/backpressure without retaining unlimited GPU scratch.

## 12. Metrics required for correct tuning

Per request: queue time, TTFT, prefill tokens/rate, decode and total tokens/rate, per-token P50/P95, context depth, prefix cache reused tokens and state domains, speculative drafter and acceptance distribution, num verifier forwards, output quality/error, terminal reason. Per engine: GPU active %, memory allocations, expert hit/miss, unique expert grouping, IO bytes/latency, graph capture vs eager, task fairness, request rejection, P95 under load, per-model weights resident.

Do not report generated drafts as output tokens; report committed tokens only. A 60 tokens/s repetitive draft workload does not imply 60 tokens/s unpredictable reasoning.

## 13. Tests and acceptance

- C1/2/4/8/16/32 mixed workload with different cached prefixes and route distributions.
- Simulated high-variance prompt arrivals, context depths 0/4K/32K/128K and output lengths.
- Cancellation during prefill, router miss, MTP verify and after partial tool boundary.
- Equal input model state with different scheduler interleavings: same-plan state/logits with numeric tolerances and deterministic sampling configuration.
- Draft lengths 0..8, acceptance 0..d, EOS early, RNG exact/correction distribution, draft mismatch, model mismatch.
- Forced OOM admission and scratch shrinking; no running state corruption.
- HTTP P50/P95 and aggregate throughput compared with raw engine independent of network.
- Cross-session isolation with shared weights and immutable prefix fork under simultaneous GPU batch.

## 14. Open questions for empirical resolution

Optimal fairness policy, prefill quantum by GPU and batch mix, GPU route grouping critical-path tradeoff, cost model sample frequency and hysteresis, prefix eviction vs expert cache priority, speculative exactness vs optional opportunistic modes, and time to checkpoint/restore for serving preemption. Avoid prematurely locking these constants.

## 2026-10-10 integrated v2 addendum — concrete research event loop and scheduler ownership

The implementation core is research's `EngineMsg/EngineEvent/StepPlan/WorkerDelta` single logical engine owner plus per-device worker with persistent `InputBatch` diffs, **not** a global locked session list. See `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`. A `StepPlan` references already-compiled `ExecutableBundle` variant handles, state epochs, row assignment, KV block leases and CIR timepoint dependencies. A device worker may complete computation but EngineCore alone commits externally visible token frontier across all state domains.

Use two independently constrained budgets: target compute work (new prefill/decode/verify rows) and input/encoder work; then physical resource reservations and fairness/latency rules. Research's 128/2048 mixed-prefill quanta and 2ms coalescing are **measured candidates**, not unconditional settings. A generic paged token-budget `KvPool` and low-concurrency extreme contiguous ReserveAll mode are explicit different admission/preemption policies, requiring different worker output and state region mobility; switching them is *not* a pure flag change.

Speculation retains research `DraftSource`, `VerifyMode`, per-hardware acceptance EMA+backoff, and `CommitPlan/Frontier` optimizations, but must obey the project's full typed state/RNG/grammar transaction. Adaptive depth uses **committed** tokens per critical-path time, and draft weights/extra verifier rows must be charged against global memory. Exact target-distribution sampled mode is the working fidelity default, with explicit opt-in when a mode intentionally changes output statistics. Model-specific MTP and DSpark source-artifact compatibility is mandatory.

**Joint optimization experiment:** GPU routes target/verifier candidate rows by expert; group rows and reuse matrix source only when more efficient and numerically legal. Scheduler may consider cache pressure, expert union, scratch and prefix hit, but it cannot know future expert selection before router activation; device table and bounded miss service remain authoritative. serving-aware kernel-tuning research 2026 demonstrates execution-mode tactic variation: record captured-versus-eager mode per plan.

New backpressure/failure tests: two overlapping StepPlans with grammar/tool boundary, cancel after first visible token, 0..k accepted draft, GPU cache miss and slot ABA, 32 concurrent requests and long prefill. No output until all state domains commit at same causal frontier. Emit ModeReceipt showing native vs ordered batch, actual draft acceptance, numeric policy and reason for fallback.


## 2026-10-12 integration — prefill-specific GPU+NPU candidate

Eligible large prefill matrix projections may use RIR output-row split across RDNA iGPU and XDNA2 when source quant, batch shape, NPU code, numeric quality and SoC concurrency are qualified. Gufo's fixed GEMM requires prompt groups of 4096 tokens; short/ragged prefill, decode1 and most speculative verifier batches should remain GPU-only unless new NPU kernels prove a gain. Scheduler selects among prevalidated plans, and CIR owns asynchronous GPU/NPU first-use joins.

Physical pinned memory, shared LPDDR bandwidth and NPU queue/power state count against the global budget. ModeReceipt reports actual row offload and wall-time benefit, not simply NPU presence. Speculative decoding target verifier and complete GDN/QSA/KV/RNG state remain authoritative. For ASR/TTS/image/video use typed TaskProgram stage budgets and output events without pretending all tasks are token generation.

## v4 refinement — mixed task cost, prepared state and bounded overlap

TaskCoordinator schedules **ready task stages**, not necessarily text tokens, using typed `WorkUnit` and physical budgets. Text generation stages can use target token/prefill/verification budgets, while image denoise, audio encoder/vocoder, video frames and embeddings need their own cost/frontier controls. Preserve fairness/age/latency targets and backpressure so one long prefill/task cannot starve ongoing interactive output. A request accepted draft sequence is only a provisional state until target verifier and all domain Prepare checks pass.

**No independent domain commit during speculation:** draft and verifier kernels produce prepared versions; logical `PublishRoot` and output happen once after GPU events and target distribution check. Device-native verification loops can avoid host trips when grammar/sampler state supported; host loop remains a correct fallback. Kernel candidates with unsupported shape, compute quant or state layout are gated before launch or cause checkpoint replay.

Physical resource reservations are coarse; one shared global lock per token/expert cache fill is prohibited. Dynamic prefetch/route results are **data**, not requests to JIT one kernel per expert ID. No unapproved top-k/mask/sampler reduction is an optimization.

## v4.2 — cross-machine task admission and stage scheduling without central hot-path RPC

TaskCoordinator owns **one session/shard TaskDriver and StateRoot** regardless of how many physical peers execute it. Stage admission checks per-node physical reservations and network credits before submitting a multi-peer StepPlan, preferring colocated execution when network cannot meet p95/SLO. Remote workers keep their layer weights/KV persistent; only bounded stage-boundary activation and Ready/Failed messages transit each work unit. The coordinator need not synchronize every tensor op across nodes if a remote RegionExecutionProvider owns a compiled whole-layer/stage/fabric program.

For P/D, reserve decode destination memory and transfer complete compatible KV/recurrent/indexer/modal state at phase boundary; for A/FFN split, route hidden activations and compact partials with per-layer queue/RTT accounting; for long-context attention shard, only mathematically qualified softmax partial summary reduction. Immutable remote prefix cache is used only when hydration beats recompute. Preempt/cancel on slow peer must **fence placement/session epochs** and keep authoritative root, not orphan selected tokens or draft recurrent state.

Fair sharing across nodes: per-task typed WorkUnits, weighted deadlines/credits, tenant access scope and backpressure; topology may become unavailable mid-session and require planned replay at last committed frontier. Speculative target correction remains authoritative and distributed draft branches cannot publish state on a worker. Transport metadata and peer-safety do not enter generic one-host token hot loop. [Remote protocol](REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md).

## v4.3 — multi-model route admission and native within-model batching

Keep two **timescales, not two inference schedulers**. `ModelRouter` resolves exact named source and selects/waits for compatible ResidentModel/Replica under ResourceArbiter at request admission, using load/warm cost, queue/p95, prefix/StateRoot affinity, effective memory, tenant/authorized peers and TaskKind. `TaskCoordinator/DeviceWorker` performs ongoing per-model/per-task decode/prefill/speculative verification batching in persistent kernel slots; do **not** select a new source/checkpoint or call HTTP router per token.

Multiple different models with unrelated state/weights can run concurrently on qualified devices if physical reservations permit; mathematically compatible sessions under *same model/adapter/precision* may batch. Never merge unrelated model token ID/position/grammar semantics into a GPU batch simply because shapes coincide. Target-draft or embedding-rerank stages are separate registered model role leases under **one TaskProgram budget**, not load balancing equivalent replicas.

Policies: bound global and per-model pending queues, cold model load wait and SLO priority to protect already warm low-latency traffic; deduplicate 100 concurrent requests for one model load; pin/cache-affine active sessions and drain before model eviction; automatic model-choice aliases only explicitly authorized. Model hot reload or replica migration requires full StateRoot/StateCompatKey and committed frontier, not a queue-level `model` string change. [Router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md).

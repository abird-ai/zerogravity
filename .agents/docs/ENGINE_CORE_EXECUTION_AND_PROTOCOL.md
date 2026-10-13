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

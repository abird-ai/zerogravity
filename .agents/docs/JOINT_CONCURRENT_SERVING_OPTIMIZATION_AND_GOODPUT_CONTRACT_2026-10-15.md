# Zerogravity — Joint Concurrent Serving, Goodput and Combinatory Specialization Contract

**15 October 2026 — DESIGN ONLY; not production serving or a vLLM speed claim.**

**Authority:** Additive to v4.3 SemanticProgram → ExecutableBundle, TaskDriver, one TaskCoordinator/StateRoot publisher, one PhysicalResourceArbiter, DeviceWorkers, ModelCatalog/Router, and v5.1 realtime/multimodal TaskSessions. No new model-specific inference engine, second resource authority or durable compiler IR.

**Companions:** VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md and VLLM_CONCURRENCY_PRIOR_ART_AND_ARCH_RECONCILIATION_2026-10-15.md in .agents/plans.

## 1. Design objective and competitive honesty

Preserve the concurrency fundamentals of vLLM-class systems—continuous token work scheduling, paged and hybrid state, prefix reuse, chunked prefill alongside decode, speculative verification, bounded admission, async worker pipelines, optional distributed P/D, and truthful metrics—and jointly specialize these **with**:
- exact model/source/processor and tensor encoding,
- quantized kernel geometry, fuse candidates and real device ISA,
- routed MoE expert overlap, padded physical row groups and source state,
- MTP/DFlash/lookup speculation and accepted tokens per expensive target step,
- GPU/UMA/host/SSD residency and real transfer completion,
- CPU/GPU/NPU/peer placement, power/thermal headroom and physical resource quotas,
- text TTFT/ITL, duplex audio latency, image/video deadline and tenant fairness.

The aim is to maximize **source-verified useful output meeting its declared service objective**, under safe physical budgets and latency, rather than optimizing unrelated component metrics independently.

Current Zerogravity implementation does NOT yet equal vLLM V1 production continuous batching; code for CPU Qwen3.5, catalog, fair resource arbitration and selected physical GPU regions is separate, narrower evidence. Any future assertion that Zerogravity outperforms vLLM requires a tuned comparable vLLM baseline with an identical model/quant/hardware/numerical setup and real concurrent request traces.

**Simplicity ceiling:** one native Rust engine, one request planner/control loop, one physical resource authority, prevalidated candidate StepPlans and cheap device-local batch execution. The joint optimizer is a planning/control view **within** the existing ExecutableBundle; not a new optimizer daemon, per-token graph search or giant dynamic dispatch hierarchy.

## 2. vLLM-class baseline — operational detail to preserve

1. **Continuous batching:** admit compatible work at safe token/stage frontiers instead of waiting for a whole request to finish.
2. **Unified token/work scheduling:** distinguish needed vs computed prompt/output/speculative tokens; allocate from bounded per-step work budget. For media, use source-typed encoder/acoustic/solver work rather than coercing every unit to text tokens.
3. **Chunked prefill/decode coexistence:** decode prioritized to protect ITL, long prefills served in legal chunks and aged to avoid starvation; chunk size may vary with available budget and source/state validity.
4. **Paged+hybrid KV/SSM:** lazily allocate physical KV/recurrent pages/state, protect in-flight reservations, reuse compatible committed prefix blocks and maintain exact source attention/recurrent state.
5. **Speculative decoding:** trained MTP/DFlash/lookup proposals remain provisional; target verifies whole state, sampling/grammar and accepted prefix before one StateRoot publication.
6. **Async worker:** overlap host preparation, source IO, transfers, GPU kernels and result delivery where mathematically legal; never hold global resource locks awaiting kernel/network.
7. **Distributed P/D or expert placement:** cost actual state transfer, network tail and warm/cold placement; latency isolation is distinct from increased throughput.
8. **Bounded overload:** request queue, live stateful sequences, prefill token backlog, device rows and memory each have separate admission limits.
9. **Multimodal coordination:** timed audio/video and finite decisions share global resource admission but retain own legal TaskDriver clock and output frontier.

vLLM V1 and vLLM-Omni already have sophisticated hybrid cache/expert/disaggregated/real-time capabilities; differentiation is **quality-grounded joint selection on specific devices/workloads**, not claiming those features are absent elsewhere.

## 3. Three independent admission limits

| Admission boundary | Bounded unit / owner | Before allocating what | Overload behavior |
|---|---|---|---|
| API and tenant queue | queued+inflight requests, payload bytes, tenant credits/weight, request deadline | any heavy model/session buffer | reject or bounded queue; typed 429/503-style overload with permissible retry advice |
| Model/state/prefill | immutable model assembly residency, private KV/GDN/KDA/draft, queued uncomputed prefill work, IO staging and OS+device quotas | active source session and state resources | defer, shrink legal prefill chunk, qualified preempt or refuse |
| Device StepPlan | active sequences, target verifier rows, token budget, media encoder units, physical kernel scratch/leases/driver depth | actual CPU/GPU/NPU submission | bounded batch selection or NoCapacity, never counterfeit Ready |

Queued HTTP requests are NOT active stateful sequences; a request with a 200K-token prompt is not one unit of GPU memory. Counting only active sequence slots allows unbounded prompt/work pressure. Counting only HTTP queue length ignores draft and video memory. Queued requests must not eagerly allocate full per-session KV.

Credits must be hierarchical for server/tenant/model/session and preserve fairness with aging. Decode and interactive audio may have protected latency priority, but long prefill cannot be starved permanently. Capacity admission must include active context memory, source weights/shared source ownership, DMA, host/SSD buffers and OS safety reserve. Releasing one reference to a shared loaded model does not free weights until all consumers genuinely quiesce.

**Cancellation contract:** waiting ticket cancel releases estimate; an unsubmitted prepared StepPlan releases preflight leases; a physically submitted command is retained until real completion or fenced/quarantined on unknown completion; request/turn cancellation never advances a card-wide physical epoch.

## 4. Unified request work frontier

The scheduling representation is an ephemeral typed **view** of existing model/session StateDomains; not a new persistent compiler IR.

~~~rust
// Non-binding Rust type sketch.
struct RequestWorkFrontier {
    session: SessionId,
    committed_root: StateRootId,
    source_model: VerifiedModelId,
    input_count: u64, verified_cached_prefix: u64,
    computed_count: u64, committed_output: u64,
    speculative_unverified: u32,
    input_media_stages: Vec<TypedWorkFrontier>,
    source_state: Vec<StateDomainFrontier>,
    generation: RequestEpoch,
}
struct StepBudget {
    active_sequences: u32,
    target_token_rows: u32,
    media_work_by_domain: Vec<DomainWorkBudget>,
    physical_scratch_bytes: u64,
    read_write_io_credits: u32,
    soft_deadline: Option<Deadline>,
}
~~~

At one scheduling opportunity:
1. Drop expired/stale/canceled waiting tasks before heavy admission.
2. Make progress on source-required already-running steps; inspect only ready/real events and source state frontiers.
3. Choose compatible decode, target verify and latency-constrained audio/video with protected credits; avoid one huge prefill blocking a live stream.
4. Fill spare target-row budget with legal prefill chunks after fairness/aging, respecting state/grammar/source processor boundaries.
5. Avoid splitting a codebook frame, video latent solver dependency or other atomic source task unit in the name of generic token budgeting.
6. Distinguish blocked-by-IO requests from executable work; SSD cache miss is a materialization Future, not a GPU source-ready token.
7. Construct a bounded compatible StepBatch; no source/quant/grammar conversion merely to merge rows from different models.
8. Dispatch with actual resource/event receipts; publish only source-valid committed frontier after device completion.

Rows proposed by MTP/DFlash are not output tokens. Required target verify rows for batch of B sequences with per-session depths d_i are conceptually R = sum(1+d_i), with physical expert-row/group padding and source state included in cost.

## 5. Hybrid prefix cache — exact state-domain frontier meet

A valid prefix hit must satisfy the *intersection of all source-required StateDomains*, not only global-attention KV hashes. Possible domains:
- full KV pages and position/RoPE masks;
- sliding window KV/rings and source lookahead/slot generation;
- GDN/KDA/SSM convolution/matrix state at a complete checkpoint boundary;
- MLA compressed KV, DSA/QSA indexer and required sparse selected positions;
- mHC stream state, draft model auxiliary tap/hidden and MTP state;
- source tokenizer/template/processor and multimodal encoder/codec alignment.

**Semantic key** (conceptual):
(original or derived artifact hash + exact component assembly, tokenizer/template/processor, source content/tokens, positional policy and state schema, numerical/reduction mode, tenant sharing authorization, causal StateRoot version and compatible restore recipe).

Physical device placement can change via a *verified state conversion/transfer*; pointer location is not the semantic key. A source-level prefix cache cannot simply share private user state because token hashes happen to match. Only immutable/share-authorized source features may be reused cross-tenant.

### Frontier meet algorithm
Each domain supplies a set/range of valid reusable committed checkpoints and what source context is necessary to resume there. Find the largest prefix frontier whose **full vector of domains** can be restored causally, with exact normalized input positions; if a GDN checkpoint is missing at that point, reduce length to previous complete joint checkpoint or recompute. For a sliding attention layer, retaining last W slots may be sufficient for source-compatible future continuation, but W alone is not an unconditional reusable prefix proof. For a source recurrent state, a single needed hidden matrix might force full replay despite global KV reuse.

A generalized multi-domain meet is required: avoid hardcoding that hybrid means just “full KV + one SWA cache”. Safe COW and reference-counted pages protect simultaneously queued/shared readers, including encoded-but-unexecuted GPU work. The candidate can choose **recompute vs RAM/SSD restore** by measured costs and integrity/tenant proof. An untrusted disk snapshot is never automatically a prefix hit.

## 6. Joint candidate selection: fast bounded Pareto lookup

Do not optimize batching, quant, MTP, cache and kernel independently. The planner filters a finite set of **prevalidated source/math/shape/device plans** and chooses among a small measured Pareto neighborhood at safe boundaries.

| Candidate dimension | Legal choices | Must preserve |
|---|---|---|
| Active work | B compatible sessions, target rows, prefill chunk length | tenant fairness, model and state domains |
| Speculation | none/MTP/DFlash/lookup, draft depth, exact target correction | accepted-token/source sampling distribution and full rollback |
| Physical weights | expert hot cache, GPU/RAM/SSD source extent, model component residency | source bytes/quant, cache readers pinned until Ready |
| Kernel | direct packed GEMV/GEMM, fused regions, row tile/ISA/numeric | qualified source tensor/output and reduction/effects |
| Placement | local CPU/GPU/NPU or trusted peer, collocated/disaggregated P/D | source artifact, state transfer validity and latency |
| QoS | TTFT, ITL, tail latency, audio first-chunk/jitter, video progress, energy | admitted advertised objective and error reporting |

Objective:

~~~text
p* = argmax over SOURCE-QUALIFIED feasible plans p:
    SLO-Goodput(p)
subject to:
    source/model/quant/numeric/effects correctness,
    physical GPU/OS/IO budgets and real completion,
    per-tenant fairness and active cancellation,
    declared tail latency and task deadlines.
~~~

**Text goodput** = useful accepted committed tokens meeting stated per-request TTFT+ITL/TPOT service targets over wall time, including overloaded/failed arrivals in reporting. **Media goodput** uses complete source-valid task units meeting its own latency objectives (PCM chunks, transcript segments, images or video segments); do not add heterogeneous unit counts as if one audio sample equaled one text token. A weighted multi-tenant score requires explicit normalized utility and per-class metrics; display class-specific goodput, admission losses and fairness even when optimizing a composite score.

Cost model dimensions: tensor codec/decode + physical expert-row padding + unique routed expert IDs; expected accepted target tokens and verifier rows; KV/GDN/MLA/indexer snapshot/COW; hot/cold GPU/RAM/SSD hit/miss and DMA; CPU/driver/IO stage overlap; batch/prefill interference; GPU graph reuse vs cold-shape JIT; physical resources and OS headroom; p95 queue/fairness, energy, backend/driver measured capability.

A small optimizer may update preferred candidates using offline/background evidence and hysteresis, not launch an exhaustive kernel search per token or during low-latency audio. Unknown hardware/model shapes fall back to simple qualified baseline. Repeatedly neutral/negative optimizations remain recorded, not forced.

## 7. Async worker pipeline and hot-path costs

Target architecture is bounded Prepare → Dispatch → Retire loops with source-verified state/physical isolation:
1. Host tokenizer/codec/verified preprocessing, IO and H2D may overlap with legal GPU compute on another admitted step; compiler/shape search must not block hot path.
2. Reuse preallocated per-shape command buffers, scratch arenas and model source descriptors; a fully resident token step must not require catalog JSON serialization or global arbiter snapshot.
3. A worker can have more than one GPU region in flight **only after** real per-stream event/lease isolation and driver recovery are physically tested. Current single worker region is *not* production continuous GPU concurrency.
4. Source backends supply true event Ready and exact output; scheduler selection is distinct from GPU submitted and actual retired kernel.
5. Provisional state for a later step may prepare ahead, but cannot publish ahead of its source predecessor or overwrite another session; root epoch/prefix validity checked at single TaskCoordinator.
6. Graph capture must protect ALL encoded/captured/queued source pointers and buffer generations until physical quiescence, not merely those in a presently executing layer (see DS4 issue #384).
7. Output network/media delivery is independently credit-bound; a slow socket never indefinitely retains GPU scratch.
8. Metrics distinguish parse/routing, queue, host prep, IO, H2D, kernel waiting, GPU execution, physical Done, StateRoot publication, media transport.

No mandatory CPU roundtrip for every layer if source-compatible device-native graph island can execute multiple steps; no false or fabricated readiness when IO/device events are missing.

## 8. Distributed prefill/decode and collective overlap

Retain the existing **single PlacementPlan** with compute, weight, mutable state and reusable cache owners and a typed TransferPlan. Disaggregated prefill/decode is **optional**: it can isolate TTFT from decode ITL, but is not automatically a throughput win.

- Prefill/decode model source+derived quant+template/numeric/state schema and causal positions must match; avoid duplicate source tokenization where permitted.
- Transfer **all** source-required state for hybrid models, not only global KV: GDN/KDA/SWA/MLA/mHC/indexer, sampler state, media track frontiers where relevant.
- Remote transfer receipt authenticates tenant, source model, state epoch, digest, version/layout and actual peer/device completion; no independent remote StateRoot publication.
- Stage/microbatch overlap of send/receive/collective and useful device compute must be legal in semantic dependency DAG.
- Compare tail ITL, TTFT, cross-device transfer bytes, queue and peak residency against collocated chunked mixed P/D; choose cheapest verified option.
- Local single-device hot path never opens a distributed RPC just to query placement; replicas/expert placement selected and pinned by request/phase.
- WAN/tunnel untrusted or high-RTT transfer may lose to local recomputation; explicit permissions and source identity control.

## 9. Failure, overload and fairness cases

- Large burst of queued requests: capacity valve, 429/503-like structured response or admitted bounded wait. No unbounded prefill work/state buffers.
- Decode-vs-prefill: protect ITL without starving long prompts, through aging/protected minimum work and bounded chunking.
- Prefix cached for KV but not recurrent/vision/codec state: exact bounded replay/restore, never report full hit.
- Draft target verifier full capacity, incompatible MTP sidecar or low acceptance: target-only fallback, identical stochastic distribution where lossless mode claimed.
- GPU cache slot being read by a *captured but not executed* command: cannot evict even if CPU cache directory marked idle; no premature reuse.
- SSD IO miss waiting: bounded staging and actual H2D Ready; cannot block all unrelated model requests under global lock.
- Target session canceled while other sessions on same physical card active: cancel only request/turn generation, not card epoch.
- Device uncertain submitted event: retain/quarantine real source/output handles and global fence until proven quiescent.
- Remote state transfer partially done: no prepared output publication; safe retry only if idempotence and model state/root support it.
- Malformed media or malicious payload: validation/refusal before expensive memory/device allocation.
- Hidden JIT/autotuning on first unfamiliar audio/image/video stage: bounded generic known-good candidate, not unbounded search.
- P95 latency misses, queue timeout and rejected requests: counted and disclosed, not silently excluded from goodput.

## 10. Metrics and honest performance claims

At minimum record per request: model artifact/quant/source processor version, task mode, arrival/admit/reject/timeouts, queue wait, live session count, committed tokens and tokens/s, TTFT/ITL/TPOT p50/p95/p99, actual prefix reused across each required domain and restoration cost, draft proposed/accepted/corrected, target verifier rows, unique experts/padded rows, GPU+OS+SSD physical pressure, IO/hot cache H2D, kernel scheduled/issued/ready counts, active/queued state, backend/driver/shape/numeric evidence, energy/power when available, tenant and tail fairness.

Avoid survivor bias: goodput denominator includes all offered work/attempted requests and rejection/timeout counts. Must gate service availability, fairness and rejection ratio alongside throughput. A server rejecting difficult requests is not “faster” by the goodput definition.

Benchmark current optimized vLLM V1 and vLLM-Omni on equal model/checkpoint/quant/tokenizer/input distribution/hardware/driver/capabilities and comparable latency policy. Different source quantization or lossy cache mode makes the result non-equivalent; report source quality alongside performance instead of treating it as a free optimization. Cold vs warm and single vs mixed workloads separate. vLLM-Omni may already support relevant multimodal scheduling, so don't infer new feature novelty solely from its task type.

## 11. Phase ownership and evidence status

| Roadmap | New explicit acceptance focus |
|---|---|
| P09 | three-level request/model/device admission; bounded queued prefill work; no unbounded API backlog |
| P10 | hybrid source-domain prefix frontier meet, tenant-safe COW/SSD restore, global memory credits |
| P12 | model/quant-compatible groups, source MoE expert count/padding and actual kernel row bands |
| P15 | exact MTP/DFlash/lookup proposals, target verification and measured acceptance/backoff |
| P16 | bounded measured joint Pareto candidates, batching/prefill fairness and goodput QoS |
| P20 | per-ISA/model/candidate cost evidence, cold-shape/negative results, no per-token tuning |
| P22 | pinned vLLM/vLLM-Omni benchmark parity and source-equivalent honest advantage proofs |
| v5.1 / distributed | realtime media QoS, remote state/collective overlap and protocol capability receipts |

Planned SERVOPT-01..SERVOPT-36 tests are specified in .agents/research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md. **They are not implemented or passed.**

## 12. Primary sources and version caveat

- vLLM V1 scheduler: https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/
- vLLM hybrid KV manager: https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/
- vLLM prefix cache: https://docs.vllm.ai/en/latest/design/prefix_caching/
- vLLM chunked prefill: https://docs.vllm.ai/en/stable/configuration/optimization/
- vLLM optional experimental disaggregated prefill: https://docs.vllm.ai/en/latest/features/disagg_prefill/
- vLLM config request queue controls: https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- vLLM-Omni: https://docs.vllm.ai/projects/vllm-omni/en/latest/
- Prior-art specialization source atlas: the specialization source atlas in the plans directory (2026-10-15).

**Checked 15 October 2026.** vLLM's hybrid KV support, disaggregated connectors and decoder features evolve; review at pinned upstream commit/driver before running comparisons. No vLLM binaries were run, no end-to-end Zerogravity model-serving performance tested in this design update.

## 13. U-048 — NPU banks and retained replay participate in joint choice

The [heterogeneous pipeline contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) adds bank/family mask, paired
prompt cuts, direct arena layout and actual dispatch transport to the existing
finite candidate set. Price fill/drain, padded NPU waves, pack/consume, causal
snapshot cuts, other-bank GPU overlap and shared LPDDR/power interference with
expert-row padding, speculation and SLO. Separate same-op row-split cost from
layer-major pipeline makespan. Reuse calibration only within its qualified
correctness/safety envelope; it cannot approve unknown firmware or alter exact
mode. Realized numeric schedule belongs to state compatibility. Admit the whole
feature/phase tuple; separate NPU, MTP, EP and replay support is not composition
proof. Compare useful accepted output and real latency against a qualified baseline
route before any outperformance claim; these are planned design requirements.

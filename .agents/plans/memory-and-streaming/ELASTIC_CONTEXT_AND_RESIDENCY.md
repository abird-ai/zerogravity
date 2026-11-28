Author: Project owner
Co-authors: GPT-6.2
Owner: @pvl

# Elastic context and residency

28 Nov 2026 — user-approved design direction; implementation and acceptance
remain open. Own physical allocation and reclamation in memory-and-streaming;
serving owns scheduling and native page consumption. Model semantics remain
in model crates. Follow the [ordered plan](../FULL_IMPLEMENTATION_PLAN.md)
and [programmable service foundations](../programmable-service/IMPLEMENTATION_PHASES.md).
This extends the existing owners, not a new runtime or a new PS-F1 prerequisite.

## 1. Objective and current boundary

Context capacity is a logical per-sequence maximum, independent of reserved,
resident and used bytes. Support many short sessions on a large-context
model, mixed-length continuous batching, and a few growing long sessions
without reserving a complete maximum-length KV slab for every admitted
sequence. Preserve predictable resource guarantees when requested; no
unaccounted physical allocation or promise that all elastic sessions can
simultaneously reach their maxima.

Current ordinary Qwen source paths (28 Nov):
- HIP uses contiguous state. CUDA supports contiguous and hybrid paged state.
  Startup smart configuration compares qualified candidates, favoring
  resident weights, native cohort width, then contiguous at a tie.
- Existing cold inventory accounts for bounded state capacity and original
  model/physical admission. ResidentPool begins with no allocated sets and
  checks out complete layout-sized sets lazily. ResidentPagePool uses the
  same backing authority for token pages and returns retired pages to idle
  reuse. Logical demand, backing allocation and idle retention differ.
- CUDA paged state includes fixed recurrent/conv domains and lazy,
  full-capacity contiguous KV staging sets for execution. It is not yet
  direct native paged attention with fully elastic backing.
- Accepted same-process HIP/CUDA context and model retirement remain
  authoritative. Existing 32K, prefix/COW and numerical results retain their
  actual scope; do not infer longer context, other backend or concurrency
  capability from the design.

## 2. Minimal shared contract

Keep three independent dimensions:

| Dimension | Meaning |
| --- | --- |
| Semantic context | Model-supported position/mask/window policy and maximum permitted sequence length; per-session effective limit can be lower. |
| Resource policy | Hard device/host limits, headroom, model/session allowances and optional guaranteed token capacity; independent of physical layout. |
| State representation | Contiguous, paged, sliding/ring, fixed recurrent/conv, or another qualified typed state layout; selected for actual geometry/backend/workload. |

One original StateRoot publishes complete committed state. Bind its typed
domains to admitted backing ranges with logical extents, valid positions,
allocation generations and exact source/numerical/layout identity. Do not
encode a model-specific state kind as a generic byte buffer just because
its extent matches. Compose per-domain policies rather than one monolithic
cache mode: token-indexed KV grows in pages, recurrent/conv uses bounded
fixed-size slots and explicit checkpoint semantics; sliding/ring state uses
the model's actual retention policy. Other indexer, latent KV, cross-modal
and speculative domains follow their real shape/effect contracts.

Reuse ResidentPool, ResidentPagePool, backing/lease IDs, physical queue and
ResourceArbiter, with a thin common state binding and capability contract.
Refactor ownership only where a real consumer needs it. No second page
allocator, scheduler, cache identity, model registry or completion authority.
Hot paths use compact typed handles/tables, contiguous metadata and
specialized native kernels; no dynamic trait/object traversal, per-token
host round trip, kernel recompilation or mandatory general graph interpreter
in the token loop. Host metadata also has an explicit bounded allocation.

## 3. Dynamic admission and residency

- Separate logical limits, guaranteed reservations, model/operation
  envelopes, committed live state, resident idle backing, transient
  workspace and physically available capacity. Represent guarantees and
  actual allocations in the same original ledger; a guaranteed credit
  converted into backing is not charged twice. Idle cached bytes remain
  physically charged until proven freed; unknown work stays charged.
- Default Auto: select qualified state/weight/workspace policies using
  actual device limits, source geometry, expected workload and observed
  pressure. Prefer measured resident fast paths where fit; select direct
  paging when it improves effective capacity/goodput or is needed to fit.
  Do not equate configured context with preallocated VRAM.
- Elastic sessions may consume available pool capacity on demand, subject
  to hard limits. Optional guaranteed mode reserves declared token/work
  capacity before admitting the request, including necessary completion
  resources. Guarantees must be real even during contention; elastic work
  cannot spend another owner's guarantee. Preserve contiguous/fixed-budget
  policies as qualified deterministic/reference alternatives.
- Cold model admission accounts for immutable weights, persistent device
  context/compiler, minimum operational workspace, fixed state and explicit
  guarantees. An elastic KV envelope is a bounded shared capacity, not a
  dedicated max-context slab for each session. Reconcile per-model quotas
  with the device owner's available capacity; permit actual reclamation/
  redistribution between models/providers without a new physical capacity
  attestation. Share UMA/host/device observations without double-counting the
  same physical RAM as independent free pools.
- Before prefill/decode/append/fork, derive exact incremental pages, COW,
  vector slots, table growth, scratch and in-flight completion costs.
  Admit the whole bounded operation or defer it before enqueue; release
  unused credit and failed private candidates on the original terminal
  path. Chunk prefill and adapt cohort membership/width to fit; retain
  original source, numerical and state contracts. Avoid serial per-page
  transactions on hot paths: reserve a batch of slots/ranges, then bind
  actual device tables and submit.
- Keep the logical-to-physical table compact; grow backing in bounded
  chunks/slabs where beneficial and acquire pages only for actual demand.
  Page sizes, alignment, packing, cache precision and native tile strategy
  are separate qualified capabilities; 64 tokens is today's CUDA
  candidate, not a universal hardware contract. Bound metadata and pool
  fragmentation; retain useful idle backing for reuse instead of forcing
  alloc/free at every step.
- On pressure, trim idle backing and evict unpinned reusable prefix/state
  entries through real completion and original lease release; retain live,
  queued, encoded or submitted readers and guaranteed credits. Use
  hysteresis and measured reuse/reclaim cost, not constant polling or
  eviction churn. Return quota only after physical retirement, then
  refresh observations. No global allocator reset.
- Scheduler must have a forward-progress policy under pressure: account for
  a next-step/retirement margin, protect decode and age deferred prefill,
  and defer admission when current work cannot progress. A parked session
  holding state is not automatically free capacity. Use qualified
  checkpoint/offload/recompute and bounded preemption only if their actual
  ownership permits it; otherwise explicit typed resource exhaustion or
  cancellation with observable retirement. No deadlock waiting for
  capacity held by the same unfinishable set of sessions, silent context
  truncation or selected-GPU-error CPU fallback.

## 4. Direct native execution and immutable sharing

- Lower original KV append/read and attention operations directly against
  bounded device block tables; support ragged positions, partial tails,
  page permutations, causal/window masks, head mapping, RoPE and the
  declared numerical/reduction contract. Use appropriate mature native
  kernels and fused page-aware lowering, or model-specific kernels where
  the measured contract fits. Providers select actual capability, not a
  backend label.
- The intended fast path never gathers/copies the full context into a
  contiguous staging slab per step. Temporary tiles, append staging and
  reductions are bounded by the real operation, not maximum context.
  Retain the current gather/contiguous route as an explicitly reported,
  separately admitted baseline until a direct route is qualified.
- Reuse the complete-domain prefix identity and scope authorization.
  Matching tokens alone are insufficient: source, adapter, tokenizer/
  processor, numerical/state contract, position and sharing scope must
  agree. Share immutable pages, COW the changed tail or domain, and fork
  independent roots. A hybrid hit is the common committed frontier across
  every required domain, including recurrent checkpoints. Missing
  recurrent/indexer state means restore an earlier complete frontier or
  recompute; never claim a KV-only full hit.
- Keep previous and private candidate roots alive until actual Ready and
  atomic publication. No pre-commit mutation of shared pages, stale
  generation reuse or reclamation with unresolved readers/completions.
  Fence/quarantine uncertain physical work. Physical reuse of device
  context, compiler and allocator never revalidates retired model or
  session IDs.
- Optional host/SSD tiers, transfer prefetch, compression/quantization and
  multi-device placement use the same source/state identities and
  completion boundaries. Add them only with independent numerical,
  migration and physical-pressure acceptance; not mandatory overhead on
  resident native execution.

## 5. Tiered state cache and configurable storage providers

**28 Nov follow-up — proposed, not implemented.** Existing explicit
state-checkpoint codecs and bounded local IO are the starting consumer,
not a transparent swapping implementation. In-process prefix retention
and eviction already exist on qualified CUDA paged paths. Explicit
committed persistence is qualified for bounded GGUF CPU and original
HIP contiguous F32 paths; preserve their actual compatibility scopes.
The current neural checkpoint does not by itself restore full task,
sampler/RNG, request or output continuation. A provider's ability to
store bytes does not qualify a state layout or resume path.

### Ownership and policy

- Separate reusable prefix cache from parked live continuations.
  Prefix entries are dispensable immutable cache; evict/recompute on
  miss. A suspended live session retains an authoritative continuation
  record and its exact committed frontier; never discard unrecoverable
  state or replay already emitted outputs. Use the original session/
  operation identity, checkpoint and continuation contracts.
- Maintain one logical state/cache identity, versioned immutable storage
  descriptors and placement observation in the existing owners. A state
  object can have a verified copy in more than one tier; count actual
  backing and temporary replicas separately. Storage residency does
  not make a second model/session or publish a new neural root.
- Eligible tiers: local accelerator memory/UMA, admitted host memory,
  local SSD/NVMe, other explicitly configured cache/storage providers,
  including remote service/object-backed stores where supported.
  Local tiers must work without a network. Distinguish original model
  artifact acquisition/storage from mutable session and derived prefix
  data. Reuse existing verified IO/storage mechanisms where applicable,
  not provider names or model-download authority as a state-store grant.
- A small typed storage contract exposes actual capability and cost:
  bounded read/write, immutable publish, exact extent/identity/integrity,
  availability, deletion/expiry and relevant range/atomicity support.
  Use native local-file and host backing first; optional remote adapters
  extend the same contract. Do not require a network daemon, Redis,
  a database, a remote filesystem or one universal object format.
- Default Auto retains active/hot state on the device, uses host cache
  for warm/parked state, and considers slower tiers only when explicitly
  enabled. User/SDK policy can configure named providers, roots/endpoints,
  allowed tier order, byte budgets, retention/TTL, minimum free space,
  IO bandwidth/concurrency and prefetch/eviction controls. External
  credentials are references to existing secret owners. Disable a tier
  by stopping new placement and draining/reconciling live references;
  removal is not implicit deletion of authoritative continuation data.

### Promotion, demotion and concurrency

- Demote immutable committed pages/checkpoints or park a complete
  continuation at a safe operation boundary. Coalesce compatible pages
  into bounded extents and retain source readers, destination quota and
  transfer scratch until actual copy/IO completion and integrity check.
  Publish a valid destination before releasing the last required source.
  A failed/cancelled transfer keeps the original authoritative copy;
  uncertain GPU work retains its existing fence/quarantine.
- On demand, resolve exact compatibility and authorized sharing scope,
  admit destination backing and asynchronous transfer workspace, and
  restore through the original private candidate/Ready/publication path.
  Prefetch likely next-use state and overlap bounded transfer with
  independent ready work; do not block unrelated decode on disk/network
  IO. Never advertise a partially uploaded page as ready. Deduplicate
  concurrent compatible fetches while preserving independent cancellation
  and lifetime ownership; no cross-scope information leak.
- Scheduler chooses keep, demote, restore or recompute by actual reuse
  probability, bytes, transfer latency/bandwidth, prefill cost, queue
  pressure and latency/SLO constraints. Protect active decode and
  guaranteed work; use age/fairness, progress margin, hysteresis and
  bounded IO queues. Slow storage is not free VRAM and is not universally
  faster than replay. Keep resident kernels free of mandatory tier
  lookup/remote calls in the token loop.
- Explicitly handle disk full, missing/corrupt objects, unavailable
  remote provider, credential revocation, quota reduction, incompatible
  revision and interrupted writes. Prefix miss can recompute; live
  continuation may resume only from a complete compatible checkpoint
  or an explicitly qualified replay path. Surface a typed refusal or
  recovery state otherwise. Never silently truncate, change arithmetic
  or restart a live session.

### Privacy, durability and evidence

Prompt-derived state may contain sensitive information. Local memory
only by default unless a slower tier is enabled; remote placement
requires explicit authorization and compatible sharing/tenant scope.
Use authenticated encrypted transport and appropriate at-rest encryption,
bounded integrity-checked codecs, expiry and deletion policy. Keep
credentials and payloads out of diagnostics. Durable recovery and
cross-process/host/device continuation are separate qualified capabilities;
a local overflow cache need not be durable across a restart.

Report cache hits by tier, actual live/idle/parked bytes, transfer/restore
and recompute costs, queued IO, evictions, resumed frontier, pressure and
refusal reason through the same Service/SDK/MCP and UI observations.
Qualify in order: local RAM/SSD prefix spill/restore; safely parked live
continuation with full applicable task state; optional external provider
and cross-owner compatibility. Reuse complete-domain prefix and existing
checkpoint evidence without broadening it. Compare resident-only,
recompute, host and SSD/remote policies under real mixed-length and
concurrent pressure; verify exact continuation/output, bounded staging,
scope isolation, cancellation, failure recovery, guaranteed capacity
and final physical retirement. Optional tier work follows direct resident
paging and elastic admission, not a new prerequisite for native parity.

## 6. Intuitive policy and programmable observability

User normally selects model, context limit and Auto execution. Advanced
settings can constrain memory budget/headroom, guarantee versus elastic
admission, state policy, allowed tiers and active capacity. Defaults are
derived, not manually pinned. Keep existing explicit backend/device and
profile overrides; refuse incompatible or unsupported requested policies
with reasons. Distinguish supported maximum context from available
guaranteed capacity and current elastic headroom. No new user-facing mode
for each allocator implementation.

Project actual configured/effective limits, state representation, committed
tokens, live/reserved/resident/idle/reclaimable bytes (non-overlapping
accounting), page utilization, sharing/COW, staging and pressure/refusal
reason. Track queue wait, TTFT, ITL, throughput/goodput, prefill/decode
cohorts and reclaim/transfer costs. Prefer counters at existing owner and
completion boundaries; detailed tracing is optional and bounded. Never
report a hypothetical capacity as observed residency or conflate cached
pages with free memory.

Use the programmable Service/Client operations and native observations;
SDK/MCP expose the same typed inspection and policy controls when their
lanes exist. Auto may choose among qualified prepared variants and adapt
allocation/scheduling at safe admission or operation boundaries. Changes
that affect layout or numerical/state identity require compatible
transition, admitted new owner and publication at a safe boundary;
otherwise drain/prepare a new revision. Lowering a resource cap below
held capacity blocks growth/drains, not invalidates live state. Preserve
explicit user overrides and rollback; experiments consume admitted quota,
not privileged hidden capacity. No online unbounded autotuning.

## 7. Delivery and acceptance

Follow existing PS-F1/F2/F3 order; use the shared contract when extracting
the original service and observation surfaces, but don't require native
paging or elastic overcommit to exit those foundations. Then integrate
common state/budget changes with one owner; ROCm and CUDA lanes can
qualify their native consumers independently. Vulkan and compiled providers
later use the same contracts. Preserve all accepted routes and evidence.

1. Establish observable, bounded dynamic credit/backing and per-domain
   layout contract on the actual existing pools/StateRoot. Prove allocation,
   return, guarantee isolation and reuse under pressure; keep unchanged
   contiguous route. Never add a second resource authority.
2. Native page-table append/attention, independent per-domain pools and
   complete-domain prefix/COW: ROCm on X2, CUDA on L5, later Vulkan. Remove
   full-context staging from selected direct paths. Qualify real ordinary
   model/serve consumers, not only isolated kernels.
3. Integrate elastic admission, incremental operation leases, shared
   device/model capacity and pressure-aware continuous batching. Retain
   guaranteed/deterministic mode; prove forward progress and recovery
   under competing requests, cancellation, reload, quota changes and
   failed/unknown completion. Optional offload/migration only after its
   separate acceptance.
4. Qualify Auto policy and actual performance/resource trade-offs with
   matched contiguous/current paged/direct paged/elastic workloads:
   existing original 0.8B/2B and 27B where supported, short and qualified
   long contexts, mixed lengths, shared prefixes, C1/C8 and higher offered
   concurrency within actual capacity. Report actual allocated/peak/idle
   bytes, context used, TTFT/ITL, tokens/s, SLO goodput, fragmentation,
   copy/dispatch and reclaim costs; compare matched vLLM where appropriate.
   No forced selector from a single favorable microbenchmark.
5. Correctness: exact source/contract identity and every required state
   domain, page boundary/partial tail/shuffled page, ragged and cohort
   transitions, append/fork/COW/scope/revocation, no stale readers, bounded
   physical pressure/refusal, guaranteed-vs-elastic behavior, retirement
   and post-unload/process idle delta. Specifically demonstrate that a
   large logical limit with short requests does not allocate full-length
   KV slabs or multiply full-context guarantees across elastic sessions;
   direct path has no hidden full-context staging. Test both allocation
   success and genuine exhaustion. Qualify larger context limits only
   with actual model/numerical/backend evidence. Finish the normal assembled
   gate and real source/cancellation/retirement tests before acceptance.

The final criterion is useful native goodput at safe memory pressure,
not paging for its own sake or maximum configured context. Keep the
contiguous and other measured winning choices available.

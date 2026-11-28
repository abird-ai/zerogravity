Author: Project owner
Co-authors: GPT-6.2; GPT-6 Astra Pro; GPT-6.1 Sol (resource, scheduling and cache review)
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

### One accounting authority

Keep logical context, physical backing and capacity commitments distinct.
Within each original physical domain, count unique allocated backing once,
plus unused guaranteed/progress credit and unused prepaid capacity.
Their sum stays within the admitted parent budget. Converting credit into
backing moves the charge; it does not duplicate it. A child/model/workload
quota subdivides its parent, not another independent pool of free capacity.

- Backing includes live state, immutable weights, device/compiler context,
  captures, idle slabs and submitted/quarantined allocations. Account peak
  private candidates, COW, metadata, workspace and transfer staging too.
  Live/idle/in-flight are ownership views; define exclusive ledger totals
  and report overlapping attributes separately. Reclaimable is a subset
  of retained bytes, never additional free capacity.
- Driver/OS/cgroup free-memory samples have timestamps and provenance;
  they constrain admission but do not create ledger credit. UMA/host/device
  aliases of the same physical RAM share a parent. Actual shared weights
  or pages count once through compatible original owners, not merely
  matching content digests.
- Existing model admission prepays a provider envelope. Returning a page
  to its idle pool does not release that envelope to another model.
  Dynamic growth/rebalancing must extend the original arbiter/lease and
  pool owners: batch reserve/checkout, rollback of unsubmitted candidates,
  bounded idle trim, and adjustment of unused parent credit after actual
  retirement. Existing Reserved-only lease partitioning is not permission
  to resize or reassign submitted/shared leases.
- Trimming reports selected cache entries, unique backing still pinned,
  charged idle backing, actually retired backing and unresolved owners.
  Prefix eviction can free zero physical bytes. Keep useful idle slabs
  with bounded hysteresis; no global allocator reset or alloc/free per token.

Cold admission still accounts for weights, contexts, fixed domains and
minimum operational resources. Elastic state uses a shared bounded envelope,
not dedicated maximum-context backing per session. Resident-only prepared
weights cannot be evicted behind an executor's back; a streamed-weight
alternative needs its supported preparation/transition. Weight streaming,
state transfers, experiments and normal inference obey the same parents.

### Queue, session and operation admission

| Boundary | Bounded resource and decision |
| --- | --- |
| Queue | Request count and retained input/metadata bytes, deadline and trusted workload allowance. Queue acceptance without admitted guarantee credit is not a completion guarantee. |
| Session | Fixed state and current retained pages; optional declared completion guarantee. Parked/output-blocked sessions still own charged state. |
| Operation | Exact incremental pages/COW/fixed candidates, table growth, peak workspace, transfer/output credits and completion lifetime for a bounded prefill/decode/restore step. |

Elastic requests may grow within available capacity and can later receive
typed resource exhaustion. A logical context maximum does not guarantee
completion. Guaranteed mode reserves the bound for the declared prompt,
maximum output/work and all required domains plus peak completion resources.
Elastic work cannot spend that credit; ordinary elastic pressure cannot
select guaranteed work as its victim. This is a capacity guarantee for the
qualified execution scope, not immunity to hardware/source failure or an
unconditional latency promise.

Use one executor demand description derived from actual prepared geometry,
layout, frontier and candidate cohort. Current worker scratch credit is a
fairness estimate, not physical workspace sizing. Estimates do not reserve;
the original owners recheck and admit the actual operation before enqueue.
Acquire the whole bounded demand or roll back/defer; never wait while holding
partial cross-domain reservations needed by peers. Reuse batch page/slot
checkout rather than serial allocator transactions per page. Failed private
candidates leave the committed root unchanged.

Keep admitted-session count, runnable cohort width and physical in-flight
limits distinct. Scan a bounded fair ready set, form compatible cohorts that
fit, and reduce a prefill quantum/cohort only on proven pre-submit capacity
refusal. Preserve queued age across retries; fit-first selection cannot
bypass a large request indefinitely. Give it a protected feasible quantum
or an explicit deadline/capacity outcome. Decode protection means bounded
service gaps under the admitted policy, not every resident row in every wave.
Token credits approximate fairness; bound long non-preemptible prefill work
using qualified chunk sizes and measured cost where useful across models.

Input/output/capacity/restore/dependency waits are explicit. Deferred rows
hold their required state but no unrelated step scratch. Completion and
credit availability wake scheduling; no agent polling or blocking storage
call on the execution worker. Genuine execution faults are not capacity
signals permitting replay or silent backend fallback.

### Progress under finite capacity

A one-step reserve alone is insufficient: repeated steps can convert it into
retained KV until every session waits for memory held by its peers. Before
admitting more retained growth, the native policy must preserve a concrete
exit: a selected request/set has its bounded completion envelope, or a
qualified park/recompute/reclaim path can release enough retained capacity,
or the elastic contract permits a deterministic capacity victim.

Reserve the execution/retirement and, when used, bounded transfer resources
needed for that exit within the original budget. If no feasible exit remains,
stop new admission/growth and finish the selected work or terminate a permitted
elastic victim with `resource_exhausted`; reclaim only after actual retirement.
Do not repeatedly restore/park peers without useful work. Protect guarantees,
retain age/fairness and report the decision. A fenced/unknown domain remains
unavailable instead of inventing reclaimed capacity. Simple conservative
bounds and a finite native policy are sufficient; no global optimization
solver or per-kernel scheduler is required.

Auto chooses among qualified state/weight/workspace policies using actual
geometry, workload and pressure. Prefer measured resident paths where they
fit; use direct paging when it improves capacity/goodput or enables a fit.
Grow payload backing in bounded chunks, with page size/alignment/precision
separate from logical context. Bound table storage and fragmentation.
Current page-table append clones/validates the complete table; measure its
long-context CPU/allocation cost and amortize checks/updates with compact
reused or shared chunks where justified. Cheap bounded metadata reservation
is acceptable; avoid replacing payload over-allocation with metadata churn.

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
  backing and temporary replicas separately. Publishing a verified replica
  does not advance neural state or invent a session. Explicit session
  restore/import/fork still uses the original replacement transaction,
  root binding and epoch; placement metadata cannot bypass that authority.
- A portable stored descriptor records verified source/semantic revision,
  numerical and per-domain schema/layout/codec, valid extents/positions,
  complete frontier, integrity and authorized sharing scope. It contains
  no usable device lease, process-local source/root binding or permit.
  Rehydration validates those facts and binds fresh original live owners.
  A content hash alone grants neither compatibility nor restore authority.
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
  IO bandwidth/concurrency and prefetch/eviction controls. Provider
  capabilities also declare durability and whether pinning prevents expiry;
  a tier whose retention cannot satisfy an authoritative continuation's
  lifetime cannot be its only required copy. External
  credentials are references to existing secret owners. Disable a tier
  by stopping new placement and draining/reconciling live references;
  removal is not implicit deletion of authoritative continuation data.

### Publication and transfer ownership

Reuse existing immutable artifact installation, held descriptors and verified
range IO where appropriate, without inheriting model-cache paths or grants.
The current checkpoint atomic-write helper is last-rename-wins, not immutable
multiwriter publication. Install verified immutable extents before publishing
their complete domain/frontier manifest with the provider's declared atomic
or conditional-publication contract. Incomplete sets are owned orphans, never
resumable continuations. Verified visibility and requested crash durability
are separate outcomes; reconcile uncertain publication rather than blindly
overwriting. Clean only exact-owned temporary/orphan records.

One existing service operation owns a transfer through queued, copying,
verified, published and source-retirement states. It holds source pins,
destination reservation, bounded workspace and exact object/provider/session
generations until actual completion. Recheck scope, deletion/drain generation
and any live session epoch at publication. Late completion cannot repopulate
a revoked scope or disabled tier. A scope invalidation that only removes
current prefix entries is not revocation; stop/fence producers when revoking.

Bound queued and in-flight bytes as well as operation/waiter counts and
workers. The admitted peak includes source + destination + page/manifest
metadata + pinned/bounce/codec staging, counting existing shared backing once.
Use a bounded blocking-IO lane or a real async provider; cancelling a future
does not finish its underlying syscall. Hold no pool/session lock or global
management slot across storage waits. Retain unknown submitted owners.
Single-flight keys include exact object, scope, codec and destination contract;
one cancelled waiter detaches without cancelling other useful waiters.

Current HIP checkpoint restore builds full-capacity planes and admits roughly
twice full-state bytes of host storage; CUDA paged execution still gathers
into full-capacity staging. These remain explicit baseline consumers.
Tight-memory local spill needs bounded native page/domain extraction and
reconstruction; do not flatten each prefix into another whole checkpoint
or claim existing codecs already implement efficient asynchronous swapping.

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
  bounded IO queues, minimum residency/cooldown and bounded speculative
  prefetch. Reserve the next useful execution/retirement margin with a
  restore; loading a session that cannot take a step is not progress.
  If full restore cannot fit, defer/refuse or use an explicitly qualified
  streaming/recompute path. Weight and state IO share finite transfer
  capacity with fairness. Slow storage is not free VRAM and is not universally
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
tokens, unique physical backing and unused reserved credit, with
live/idle/pinned/reclaimable ownership views as defined in §3. Include page
utilization, sharing/COW, staging, pressure, wait and refusal reasons. Track queue wait, TTFT, ITL, throughput/goodput, prefill/decode
cohorts and reclaim/transfer costs. Prefer counters at existing owner and
completion boundaries; detailed tracing is optional and bounded. Never
report a hypothetical capacity as observed residency or conflate cached
pages with free memory.

Use the programmable Service/Client operations and native observations;
SDK/MCP expose the same typed inspection and policy controls through
[resources()](../programmable-service/API_AND_PROTOCOLS.md#45-resource-policy-and-cache-control).
Keep one small facade: inspect/estimate, policy get/update, provider inspection
and bounded maintenance. Reuse configuration transactions for named storage
instances, existing operations for transfers/drain, and continuation controls
for qualified park/resume. No public page allocator, dispatch tool or
independent resource registry.

Saved defaults, desired/active resource policy generation, and model/mapping
revision are distinct. Saving configuration does not activate a policy.
Inspect returns all three where relevant. Policies and per-request limits
are clamped by trusted parent grants; a workload label is not sharing or
tenant authority. Quotes report exact target, known/unavailable demand and
observation time, and never compile, evict or reserve resources.

Auto may choose among qualified prepared variants and adapt
allocation/scheduling at safe admission or operation boundaries. Changes
that affect layout or numerical/state identity require compatible
transition, admitted new owner and publication at a safe boundary;
otherwise drain/prepare a new revision. Lowering a resource cap below
held capacity publishes a desired cap and visible drain deficit, blocks
new admission/unreserved elastic growth, and leaves existing guaranteed/progress
credit spendable under its admitted terms. The prior obligations remain charged
until retired; the lower target does not retroactively free or invalidate state. The active transition reports remaining obligations.
Pin source/layout and guarantee terms for admitted work; supported elastic
growth/fairness policy can change at safe operation boundaries with its
decision generation recorded. Rollback affects future decisions, not emitted
output or discarded replicas. Preserve explicit user overrides; experiments consume admitted quota,
not privileged hidden capacity. No online unbounded autotuning.

## 7. Delivery and acceptance

Follow existing PS-F1/F2/F3 order; use the shared contract when extracting
the original service and observation surfaces, but don't require native
paging or elastic overcommit to exit those foundations. Then integrate
common state/budget changes with one owner; ROCm and CUDA lanes can
qualify their native consumers independently. Vulkan and compiled providers
later use the same contracts. Preserve all accepted routes and evidence.

The dependency boundary is deliberately small:

| Work | Needs |
| --- | --- |
| F1/F2/F3 | Faithful current admission/observations; common policy/query types and supported controls; explicit unsupported elasticity/tier operations. |
| Shared memory/scheduling core | Foundation exit; one owner extends original accounting, batch checkout/trim and demand-aware fair scheduling. No PS-E optimizer dependency. |
| HIP/CUDA direct paged consumers | Foundation exit and the agreed common per-domain binding/demand contract; independent implementations and qualification. |
| Integrated elastic serving | Shared core plus that backend's qualified direct consumer. No all-backend, MCP or tier-store gate. |
| Tiered prefix cache | Applicable page/domain extraction and storage consumer; local RAM/SSD first. Native resident parity proceeds without it. |
| Live parking / remote tiers | Applicable PS-S full task continuation (PS-C edges for multi-stage tasks) / actual authorized provider. Neither follows from a neural checkpoint alone. |

The following are responsibilities within coherent implementation batches,
not a serial gate or a separate acceptance cycle per primitive:

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
   copy/dispatch, CPU/table allocation and reclaim/IO costs; include all
   offered/admitted/refused/failed/cancelled outcomes and compare matched
   vLLM where appropriate.
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
contiguous and other measured winning choices available. Fold pressure,
cache/IO failure and fairness coverage into the existing real serving/state
scenarios; reuse unchanged evidence and run the assembled gate once per
coherent implementation boundary, not a new matrix per policy or method.

## Source seams

- [Original admission](../../../crates/zerogravity-core/src/resource_arbiter.rs)
  and [Reserved-only partition](../../../crates/zerogravity-core/src/resource_arbiter/partition.rs).
- [Resident backing pool](../../../crates/zerogravity-gpu/src/resident_leases/backing_pool.rs),
  [paged state](../../../crates/zerogravity-core/src/paged_state.rs),
  [worker](../../../crates/zerogravity-engine/src/router/worker.rs) and
  [wave formation](../../../crates/zerogravity-engine/src/router/waves.rs).
- [Checkpoint publication](../../../crates/zerogravity-artifacts/src/checkpoint.rs),
  [immutable artifact installation](../../../crates/zerogravity-artifacts/src/cache/store.rs),
  [bounded range IO](../../../crates/zerogravity-artifacts/src/ssd_extent.rs) and
  [HIP checkpoint consumer](../../../crates/zerogravity-gpu/src/resident_inference/resident_text/checkpoint.rs).

Author: GPT-6 Astra
Co-authors: Codex coordinator
Owner: @pvl; coordinator reconciles live plans

# Pre-0.1 execution fabric: source-grounded decisions

23 Oct 2026. Parent plans: cpu-gpu-optimization, pipeline-composition,
model-execution, memory-and-streaming, serving-and-speculation and
distributed-execution. Source review plus one contained staged implementation;
not a new runtime, model qualification or performance claim.

## Decision

Keep one semantic `ForwardPlan`, one prepared physical execution plan and the
existing source → attempt → original flight → actual completion → original
SessionAuthority publication spine. Improve the boundaries before widening
families. Break internal/public pre-0.1 APIs where needed; avoid compatibility
facades that preserve the actual coupling.

```text
verified family source + exact family mathematical traversal
    ↓ shared semantic operations and optional region scopes
ForwardPlan: typed values, source operands, numerical contracts, state effects
    ↓ prepare finite compatible implementations + storage/transfer lifetimes
existing prepared physical plan: kernels / graphs / transfers / retained owners
    ↓ actual RootAttempt, values, generations, source and current epochs
original fair worker + ResourceArbiter + flight → actual Ready → SessionAuthority
```

Four implementation layers describe responsibilities, not four new crates:

| Layer | Owns | Must not acquire |
|---|---|---|
| Engine primitives | Backing/lease generations, ranges, physical admission, transfers, events, quarantine | Tensor role names, model geometry, task publication |
| Shared model operations | Typed projection/RMS/attention/state operations, preparation, liveness, row/feedback transport | A second mathematical traversal, family source assumptions |
| Family primitives | Tensor roles/ties, source config, exact mathematical traversal and source-specific math | Provider allocators, streams, worker queues |
| Model specialization | Verified actual source/config; finite qualified native candidates | A copy of shared math, independent storage or completion authority |

The goal is low complete-request cost under the declared quality and ownership
contract. Fewer types, launches or copies are useful mechanisms, not independent
acceptance criteria.

## Follow-up: shared physical ownership and the next native consumer

The [Astra architecture-only follow-up](../../runs/architecture-astra-next-native-review-20261023/DESIGN.md)
accepts prepared-binding8 and defines two substantial Sol HIGH units:

- Split source-neutral physical ownership from source-bound ResidentService:
  one context/envelope/queue/ledger/fence, explicit model sub-admissions,
  independent close/reload, retained real source/module/flight owners. Attaching
  a model cannot register larger caps or reuse the entire remaining budget.
- Record Qwen27B's existing family math through MathematicalBackend/ForwardPlan,
  with explicit MKL arithmetic identity,112 logical domains/128 ordinary planes
  and fullConv4. Short-HF42 domains/48planes and scalar GPU exp cannot be retagged.

Keep frozen Gemma single-source qualification separate. Multi-model serving,
device-resident stage edges, fusion and MoE reuse require their actual consumers
and physical/source/state proof. Astra proposes redesigns; Sol implements.

## Initial findings and concrete corrections

These findings describe the initial review baseline. Shared head extraction,
pools/rings and prepared semantics have since advanced; current implementation
and qualification status lives in [TODO.md](TODO.md). They remain design
constraints, not assertions that each original coupling is still present.

### 1. Shared core currently contains closed Qwen and Gemma model recipes

Source anchors:
- `crates/zerogravity-core/src/accelerator/program/full_model_profile.rs`: closed
  Qwen08/Qwen2B geometry and projection widths/vocabulary.
- `.../program/vllm_model.rs`: Vllm08Call, 24 layers, attention every fourth layer,
  original Qwen tensor member names, fixed projection memberships.
- `.../program.rs`: publicly exported family descriptors and TypedVllm08
  operations alongside shared primitives; MathematicalBackend has both generic
  operations and historically closed source/numerical operations.
- `.../program/bf16_text.rs`: nominally shared BF16 call had fixed Gemma
  vocabulary262144/cap64; recording still deliberately checks exact15 KV domains
  and finite candidate widths.
- `crates/zerogravity-model-qwen35/src/original_binding/window_model/full_model_source.rs`:
  actual source seal already authenticates original config/index, role geometry,
  tied output and complete family schedule. This is the correct family boundary.

Risk: another Qwen/Gemma size requires edits to shared core, unrelated numerical
matches and provider dispatch, encouraging another copied special case.

Correction in progress: shared `TextCallGeometry` carries source/vocabulary/
capacity metadata and validates token IDs plus the original attempt frontier.
Gemma derives it from the actual verified model, after its existing finite
source guard. Bf16TextCall retains its explicit cached-one phase rule; numerical
and source execution eligibility remain independent. This is a first useful
boundary extraction, not completion of all family leakage.

Next migration, in one assembled unit:
1. Family source owner builds immutable geometry/source-role bindings from actual
   metadata; validate full role joins/ties here once. Source-only identity reuse
   must retain the original owner and all collision/content checks.
2. Record generic semantic operation descriptors with explicit source matrices,
   dimensions, cast/store/reduction contracts and state effects. Remove model
   names from the fields consumed by shared transport/storage/liveness.
3. Existing native adapter matches exact operation-region identity and finite
   qualified shape/source/numerics. Keep .8B/2B distinctions where they select
   genuinely different compiled code or arithmetic; they are implementation
   eligibility, not universal mathematical geometry.
4. Move closed family builders/validators to Qwen; remove FullModelProfile from
   generic core only after consumers no longer need it. A rename or adding an
   Other variant to the enum does not accomplish this migration.

Do not replace explicit numerical contracts with a loose dtype enum or generic
floating-point tolerance. Source-specific reduction/FMA/rounding schedules are
real semantics, even when named after the reference that exposed them. Separate
family identity, arithmetic identity and prepared implementation identity.

### 2. Streamed head still combines family sealing and reusable execution

`crates/zerogravity-gpu/src/vllm_cuda_qualification/short_hf_head.rs` retains
OriginalWindowModelSource, QwenMatrixRole, node254, ShortHfModelBinding,
ModelValues and RMS owners in its streaming operation. These are legitimate
qualified-adapter constraints. They prevent another family from reusing the
whole transport/greedy operation without carrying Qwen's recipe.

Extract shared behavior behind the original prepared/flight objects:
- Family adapter seals the actual output matrix, optional embedding tie, selected
  source operation/value IDs, exact normalized input and arithmetic contract.
- Shared prepared head owns `VerifiedMatrix`/authenticated source owner, matrix
  layout, existing resident/streamed choice, bounded transfer retirement,
  output/status leases and a concrete prepared kernel implementation.
- Per-call binding supplies original attempt, input/output lease views and
  selection extent. Node254 and six particular rows remain in Qwen adapter
  validation, outside the shared page loop.
- The outer original flight owns all pages/kernels/status. Shared code must not
  construct a different completion event or publish a token.

ROOT's source-derived matrix layout/row transport and removal of .8B-only bytes
are the right foundation. Preserve the exact original2048 packet before any2B
residency claim. Transport reuse alone does not qualify Gemma head arithmetic.

### 3. Reuse must be physical ownership reuse, not hidden retained allocations

Current work in resident_leases/backing_pool, transfer_retirement,
resident_program/native_full/{value_storage,short_hf} and native value/RMS
owners addresses the correct common boundary. Review assembled code against:
- Physical reservation survives idle, checked-out, Pending/snapshot readers and
  unknown work; fresh logical checkout starts uninitialized with a new identity.
- One exact backing may have several views, but charge it once and retain every
  actual/encoded/captured reader until retirement. No refcount-only completion.
- Liveness is derived after region selection/fusion, including observations and
  escaped outputs. Sharing six geometry workspaces requires exclusive activation
  and no retained Ready/certificate view into an overwritten range.
- Pool/ring exhaustion yields bounded backpressure outside the workspace lock;
  cancellation cannot turn unknown work into available capacity.
- Native module/compiler/cache owners belong to actual backend/service lifetime;
  a model drop or managed-pool report cannot prove native memory release.

The shared SHARED_MEMORY_AND_BOUNDARIES.md already states these invariants.
Do not duplicate them as a second allocator protocol or receipt system.

### 4. Serving still dispatches by specific source route

`crates/zerogravity-cli/src/serve/gpu_engine/route.rs` has Gguf/Canonical/ShortHf
ModelEngine and RequestEngine variants, source-specific session construction,
prompt validation and presentation IDs. This bounded closed dispatch is not an
urgent performance bug. It is an extension seam to fix before adding family
branches throughout HTTP/worker logic.

Prepare a model execution adapter once, exposing existing operations: validate
request, create original session, bind/submit accepted step, inspect committed
output, cancel. A concrete enum or one coarse object per prepared model is fine;
no vtable per tensor element or projection. The generic route retains auth,
limits, output credits and errors; tokenizer/template adapters remain source
specific. Do not introduce another session, worker, state publisher or universal
TaskDriver before a second actual task needs it.

### 5. Advertised backend support must follow implementation tuples

Common plan/source/lease code is reusable across CUDA, HIP and Vulkan. Current
BACKEND_PARITY.md distinguishes tiny Q8/FP32 and BF16 transport results from
unqualified learned HF execution. Keep this distinction in runtime capabilities.

- Backend = physical command/memory/event implementation, not arithmetic mode.
- Kernel eligibility = source/semantic and numerical contract + physical layout +
  actual shape band + device/compiler/code identity + complete admitted owners.
- Share CubeCL Rust kernels where they preserve the selected arithmetic contract.
  Vendor native code remains an optional concrete candidate under the same plan.
- HIP module retention and Vulkan mapping/barrier/subgroup behavior require real
  implementations. Sharing a PCI budget is necessary; it does not equalize them.
- Failed/unsupported preparation may choose an already qualified alternative;
  unknown submitted work must fence/quarantine, never silently fall back.
- Keep the local gfx90c hold. No device qualification was run by this review.

## Nested regions, fusion and profiles

Preserve the 21Oct REGIONS_AND_PROFILES.md design:
- Region scopes are private metadata over the existing operation/value/effect
  graph. Deterministic flattening reproduces the ordinary reference schedule.
- Nested math and compound media tasks share composition principles, not one
  untyped port/frontier system. Media carries timestamps/segment clocks and
  task commit frontiers; it must not fake token offsets.
- Prepare bounded candidate regions outside the hot path. Fused regions preserve
  complete source/numeric/effect/observation contracts and original provisional
  state. Whole-model specialization remains legal through the same boundary.
- Joint liveness includes all consumers, traces and exported views. Fusion can
  lengthen lifetimes or reduce occupancy; retain forced ordinary decomposition.
- Same-device compatible stage edges retain typed leases/event dependencies.
  Stage boundaries alone introduce no payload copy or host wait. Real codec,
  device, host-decision or authorization boundaries use explicit bridges.
- Workload profile selects prepared compatible policies, separately from
  ProgramNumericalProfile. Changed math or lower expert top-k is an explicit
  approximate semantic variant with separate quality gates, never auto-tuning.
- Finite runtime selection uses prepared IDs/arrays and actual workload bindings.
  No strings, model catalog lookup or generic graph optimizer per token.

A new trait is justified by an actual interchangeable implementation boundary.
Use generic concrete code for inner kernels/operations, compact enums for finite
runtime choices, Arc immutable tables and indexed per-call binding workspaces.
Avoid monomorphizing every model × backend × shape × profile combination; code
size and compile pressure are also costs. No claim of zero overhead without
measurement: required ownership checks and device/driver calls still exist.

## Sharding, multiple devices and MoE residency

Keep placement as physical lowering of the same program. Device affinity and
source/state partition algebra are separate from model identity:
- Start with admitted layer groups and local state shards; transport carries
  source/profile/root/frontier/epoch and actual producer readiness.
- Tensor/head partitions need declared collective reduction order; attention
  summaries and recurrent state are different algebra. Quantized wire formats
  change the numerical contract unless exactness is proved.
- Each physical memory domain retains its original arbiter/accounting. Transfers
  hold source, destination, staging and events across devices; timeout is not
  evidence that DMA or remote execution finished. Original root publication
  waits for every required state shard, with no partial-root visibility.
- Topology/profile changes happen at committed frontiers and retain old readers.
  Distributed execution and remote model execution remain unimplemented here.

MoE intelligent residency should reuse the same source/pool/transfer machinery:
- Logical expert ID + immutable source tensor/member/range/codec identifies
  weights; a physical slot/generation/device identifies current placement.
  Routing IDs stay data, not a compiled variant per expert.
- Pin all encoded, captured, queued and in-flight readers before selecting
  victims; separate completion from last-reader release. Protect demand hotsets
  from transient prefill/prediction. Mixed encodings share one byte budget.
- Preparation policy may rank candidates by observed reuse, fetch cost, size and
  fairness; bounded prediction gets its own admitted bytes/IO credits and can
  be canceled. Demand progress must survive inaccurate prediction.
- Share immutable weight backing only when verified content/layout/encoding and
  authorization permit it; retain each live source binding. Never share mutable
  StateRoots, routing history or activation state because weights match.
- Migration/replication temporarily charges old+new storage and both readers;
  logical expert identity remains fixed. No hit-rate-only speed claim.

`core/.../text_moe.rs` explicitly implements an analytic contract only, while
`gpu/src/residency.rs` is a synchronous exclusive-provider cache. Neither is an
asynchronous trained expert cache. Evolve the latter only through real reader
leases/completion; copying its LRU table into a model would lose the safety
contract. Real sparse source/top-k/reduction qualification remains required.

## Priority and useful end-to-end units

1. **Finish the assembled storage/streaming unit now.** Shared backing/rings,
   six-geometry workspace and source-sized head. Keep source/state/fault/API
   checks and measured whole-request comparison together. Root pooling alone
   cannot repair the source-reported2B970-window/per-window-wait bottleneck.
2. **Consolidate source operations before additional family expansion.** Current
   call geometry extraction, source-sealed matrix/head operation and generic
   native prepared owner. Migrate .8B/2B and the actual Gemma candidate to these
   common APIs; preserve separate arithmetic qualification and current guards.
3. **Move remaining family builders out of core; simplify serving adapter.**
   Qwen source/tensor roles stay in Qwen, exact shared operators stay in core,
   finite compiled implementations stay in provider adapters. Demonstrate a new
   geometry without touching shared memory/transport/worker code.
4. **Add one useful region/profile vertical slice.** Forced ordinary versus
   resident chain/fused path with identical source/state and required observers;
   no generalized nesting runtime prerequisite. Extend meaningful profile bands
   and shared state only after concrete correctness/ownership closure.
5. **Then broader trained sparse/media/distributed features.** Keep design seams
   now; avoid implementing speculative frameworks before real source consumers.

## Prior design accounting

| Existing design | Disposition |
|---|---|
| MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md | Retain one fabric, two representations, whole-region substitution; illustrative interface names are not current runtime support |
| MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md | Retain sole publication, finite credits, coarse physical admission and actual Ready/quarantine |
| COMPOSITION_AND_EXTENSION_CONTRACTS.md | Retain typed closed region/state bridges; instantiate smallest concrete seams, not all proposed public traits |
| TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md | Retain semantic versus physical lowering and bounded specialization; hierarchy is metadata/view, not a new IR |
| REGIONS_AND_PROFILES.md | Retain nested scopes, joint liveness and device-resident stage edges; implement against current ForwardPlan |
| PROFILE_POLICY_AND_COMPONENT_REUSE.md | Retain explicit approximation and replaceable native components under existing authority |
| SHARED_MEMORY_AND_BOUNDARIES.md | Adopt as current pool/ring/source-preparation implementation contract |
| BATCHING_AND_PAGED_STATE.md | Retain complete hybrid state/paging/common-frontier semantics; no KV-only cache shortcut |
| Distributed and SSD design contracts/live TODOs | Retain source identity, reader pinning, remote epochs and explicit admission; generic geometry adds no execution grant |

Older handoffs describe historical checkpoints and preserved sources; they are
not additional tracking duties or current qualification. Live plans/source and
executed results control acceptance.

## Consolidated redesign review — 23 Oct closeout

### Boundaries required before new model merges

1. **Source geometry and family traversal must stop being shared-engine enums.**
   `core/src/accelerator/program/full_model_profile.rs` encodes two Qwen sizes;
   `vllm_model.rs` additionally owns24 layers, every-fourth attention, tensor
   names and projection member joins. Its `members_for_profile` treats every
   non-.8B profile as2B. Adding enum variants is unsafe architectural expansion.
   Family-owned verified config/roles/ties must supply concrete source operands
   and exact traversal to existing semantic operations. Preserve the enum only
   temporarily as a closed legacy/native eligibility adapter; do not add models
   to it or merely rename it. Move each recipe only with a real migrated consumer.
2. **Shared operations own math, not family permission.** Existing `TypedProject`,
   `TypedEmbed`, typed RMS/state operations and `SourceTensor` are the starting
   point in `core/src/accelerator/program.rs`; no competing operation IR. A
   projection/compound region carries explicit dimensions, source members and
   reduction/cast/store contract. Native eligibility matches that contract,
   actual source packet, shape, layout and code identity independently.
3. **Prepared matrix/head boundaries retain actual owners.** Current
   `gpu/src/vllm_cuda_qualification/short_hf_head.rs` still owns Qwen nodes253/254,
   ValueIds324/326, final producer and source ties. Those seals belong in its
   family adapter. Shared source windows/geometry, prepared kernel and W/X/Y
   leases are reusable. Extract the complete prepared head next through this
   actual consumer, retaining its original Run/ring/flight; do not introduce a
   generic head trait until another real lowerer consumes the same contract.
4. **Serving adapts prepared models once.**
   `cli/src/serve/gpu_engine/route.rs` currently branches Gguf/Canonical/ShortHf
   across model/session operations. Extend by a coarse prepared execution
   adapter over existing request validation/session/step/output/cancel methods,
   with the current fair worker and SessionAuthority unchanged. Do this when
   the next real family joins; no placeholder TaskDriver or per-tensor vtables.

`TextCallGeometry` and the actual Gemma consumer are a useful metadata seam,
not a universal numerical grant. `bf16_text.rs` still deliberately has finite
candidate capacity/domain/width rules. Do not infer arbitrary-model support
from accepting different token/vocabulary metadata.

### One explicit resource policy, independent physical domains

- Existing `optimization::MemoryFootprint` suffices for host/provider selection.
  `short_hf/loaded/inventory.rs` must compare actual host existing+retained+maximum
  transient against source host admission, independently of provider capacity.
- Root's current provider patch threads configured bytes through load, selection,
  `ResidentService` and original fair ticket. It retains4096MiB host and5120MiB
  physical caps. Invalid caps are rejected before rotary/native preparation.
- Complete2B source-derived provider bound4598390272B fits explicit4608MiB,
  leaving233447936B within that allowance. This is a bound, not observed usage.
  The larger prepaid provider envelope leaves512MiB within the physical cap
  for other admitted owners. It does not establish unchanged multi-model density.
- `physical_service.rs` charges the whole provider envelope once. Keep native
  images/opaque provider charges separate from managed aligned buffers inside
  that envelope; do not replace either with free-VRAM snapshots. Resident head
  upload remains bounded source windows, not a full host shadow.
- Old4096MiB benchmark and new4608MiB profile require distinct protocol/package
  bindings and same-protocol preflight. Stock requested4096/KV512MiB stays
  unchanged and has no hard device cap. Report this asymmetry explicitly.
- Source-neutral `HostWorkspace` consumes the original host reservation and
  retains source owner. Its metadata label is not source authenticity. HF
  `VerifiedTensorSource` stays a format adapter; GGUF must consume its own actual
  source/host owner, not acquire a fabricated HF wrapper or new allocator.

### Batch admission precedes batch execution claims

Reviewed numerical_source_audit's current donor under its continuous-HF-batching
run: physical BF16 row extents, descending prepared-width decomposition, and
existing cohort publication extraction are appropriate low-level seams.

- B4/B8 does not imply B2. Unmatched rows use prepared B1 without padding;
  fallback is prelaunch only. Unknown submitted work cannot be retried as B1.
- Current14-root bound is8 current +2 Pending +2 snapshots +2 displaced.
  B8 needs8 simultaneous candidates as well as current roots; replacing14 with16
  silently discards external reader allowances. Derive the full new peak or
  explicitly prove a narrower concurrent-owner policy at the existing API.
- A batch owns one complete activation/workspace epoch. Independent row views
  must not each reactivate/overwrite the same six-geometry storage. All escaped
  Ready/certificate readers retain the active owner through their true lifetime.
- Mixed frontiers require actual per-row position/mask/RoPE and state mappings
  in native packets. Scheduler grouping alone supplies none of this.
- Recheck all original row roots after physical completion, then use the one
  original all-or-none cohort publisher. No second token loop or batch authority.
- Original B4/B8 CUDA source/argument/state evidence remains required. MKL-HA or
  B1 receipts cannot stand in for batched GPU arithmetic.

### HIP native ownership: one concrete blocker, no permission by inventory

Reviewed peer draft in its Gemma worktree:
`vendor/cubecl-hip/src/prepared_native.rs`, `prepared_native/artifact.rs`,
`compute/context.rs`, `compute/server/prepared_native.rs`.

- Exact finite compiled set on the original actor, sealed-cache miss refusal,
  module admission before load, retained failed/null loads and all-stream
  completion before unload are the right shared backend boundary.
- **Concrete draft blocker reported to peer:** ordinary `HipContext` destruction
  drops `PreparedModules`/`LoadedNativeOwner` admission while raw modules may
  remain loaded or completion unknown. Remaining native owners/charges must be
  quarantined or deliberately retained on context drop; only successful explicit
  retirement may release them. Peer subsequently reports a conservative Drop
  retention fix plus graph/capture guards and six retention tests; this final
  architecture pass has not re-read or verified that correction. Sol must review
  the assembled owner graph before merge.
- ELF PT_LOAD bytes describe source segments, not total driver/module residency.
  Keep explicit conservative opaque driver/compiler envelopes. No caller total
  or ELF-only estimate may grant physical fit.
- Cache-held admission must not retain a service which owns that same cache.
  Use an acyclic original lease/native-owner graph. Current retirement holds all
  charges until every unload succeeds; document this conservative batch release.
- Complete Gemma specialization enumeration and actual admitted consumer remain
  prerequisites. A safe shared cache implementation alone does not remove a
  model's Unsupported guard or qualify HIP inference.

### Composition, sharding, MoE and optimization context

Retain the prior decisions above and existing REGIONS_AND_PROFILES and
BATCHING_AND_PAGED_STATE designs. Current `core/src/pipeline.rs` is a pure typed
configuration checker; current `text_moe.rs` is analytic; `gpu/src/residency.rs`
is a synchronous exclusive-owner cache. None implements general media execution,
asynchronous expert pinning, nested runtime scheduling or distributed inference.

- One immutable prepared model/source context; independent per-session roots;
  per-call actual attempt/leases; explicit operator resource/workload policy.
  No all-purpose mutable global execution context collecting these authorities.
- Nested regions are private scopes over ForwardPlan. Flattening preserves exact
  operations/effects; specialization substitutes only a closed equivalent region.
  Recompute liveness after fusion, including diagnostic and escaped consumers.
- Same-device task edges retain typed leases and producer dependencies. Media
  timestamps/segments and neural frontiers remain different typed state; shared
  immutable model instances do not imply shared mutable session state.
- Sharding is physical placement of the same source/state graph. Per-card
  arbiters, source+destination+staging reservations, explicit collective arithmetic
  and every required shard's real completion precede original publication.
- Expert pinning covers encoded, captured, queued and executing readers; eviction
  uses physical generations and one mixed-codec byte budget. Prediction has
  bounded separate credits, cannot starve demand, and changes no routing math.
- Optimize measured whole requests: residency/transfer overlap, then prepared
  fused regions and bounded shape bands. Keep ordinary decomposition and exact
  numerical contracts. Dynamic dispatch at coarse preparation is acceptable;
  avoid per-element abstraction and model×backend×shape monomorphization growth.

### Priority, draft disposition and acceptance scope

1. Coordinator finishes assembled pools/rings and fresh .8B4096MiB measurement.
2. Sol reviews/takes ownership of preserved drafts: neutral host metadata
   admission (already integrated by root), four-path BF16 matrix-source donor
   (CUDA head + real HIP/Vulkan upload), explicit4608MiB benchmark package.
   Current paths and run notes are preserved; Astra authors no further code.
3. Complete2B resident source/state/fault/API proof under explicit allowance;
   integrate generic source/head operations and fix HIP context-drop ownership.
4. Complete actual B4/B8 cohort admission/native semantics before serving enable;
   then benchmark and C16 under separately declared resource/scheduler policy.
5. Remove remaining closed family recipes from core by migrated operation/region
   consumers; introduce serving prepared adapter with the next actual family.
   Expand Vulkan/Qwen27B/Gemma only as their own arithmetic/source owners qualify.

A new model merge should demonstrate reuse of the same storage, transport,
prepared-native lifetime and worker/publication path. It may add source roles,
family math and finite backend candidates. It must not add another size match to
shared resource policy or another copy of a provider allocator/flight.

## Evidence and unfinished work

Historical work in this run: six core call/BF16 tests and an actual isolated,
offline/locked strict core+Gemma consumer check passed. The call seam was
integrated by root. Subsequent donor source review/rustfmt is not GPU validation.
Before the architecture-only correction, the cloned benchmark package's13 cold
Python fixtures passed; actual package verification covered37 files and tool
verification covered3 original tools. Original package bytes remained unchanged.
Those checks establish no inference, numerical parity or performance result.

Final pass was read-only except this owned DESIGN.md. No model/GPU/compiler job,
source/package patch or test was run after the user corrected Astra's role.
Root/Sol own integration, current assembled checks, physical/source/state/fault
closure, benchmark execution and final isolated scripts/check.sh. Existing donor
code is reviewable draft work until they accept it. Future architectural review
must verify peer fixes/current results rather than treating this snapshot as live
qualification.

## Read-only followup — prepared head and finite HIP policy

Architecture-only review after Sol's shared-head donor. No source/package edits,
compilation, tests or GPU work in this followup. Coordinator reports phase1
.8B/2B physical/source/state/fault closure and preliminary timings; those results
cover the frozen phase1 implementation, not the new extraction or HIP inference.

### Sol six-path head donor

Reviewed `.agents/runs/shared-integration-sol-high-20261023/head-donor`, especially
`gpu/src/vllm_cuda_qualification/{prepared_bf16_head,short_hf_head}.rs`.

- Approve the architectural split: concrete PreparedBf16Head/Bf16HeadCall own
  source operand, numerical profile, native manifest/images, WXY leases and
  transfer progress. FamilyHead retains Qwen source roles/tie, original record,
  nodes253/254, values324/326 and final-normalizer producer. Original typed RMS
  registration retains both family Run and shared HeadRun on the same flight.
- Unsafe shared register/enqueue are explicit internal lowering seams; the safe
  family adapter proves their producer/operation/access prerequisites. They do
  not introduce another completion or publication authority.
- **Fix before merge as reusable API:** resident weight acceptance currently
  calls `ResidentMatrix::require(source,service,rows,columns)`, which does not
  compare the actual tensor/member. Same-source same-shape matrixB can satisfy
  shared preparation for operandA. Require exact issuer+tensor/member/range+
  axes/codec/geometry at this join, with a hostile same-shape negative. Current
  Qwen TiedWeight adapter independently constrains its original tied table; this
  is a reusable boundary gap, not evidence phase1 Qwen used wrong weights.
- The six selected-row extents and248320×1024/2048 packets remain finite CUDA
  eligibility. They are not universal head semantics or B4/B8 request batching.
  Adding another family requires its actual producer/source seal and qualified
  arithmetic candidate; do not broaden these checks by inference from dtype.
- Shared source reads remain HF-specific. Portable HIP/Vulkan byte upload reuses
  that owner and geometry while accepting every BF16 bit pattern; finite-value
  checking is an explicit arithmetic requirement, not transport permission.
- Remaining debt before further model expansion is still family recipes in
  `core/.../full_model_profile.rs` and `vllm_model.rs`. Move exact traversal and
  role joins to family code using existing typed operations, not a new universal
  model trait or an additional descriptor hierarchy.

### Practical HIP policy: finite conservative qualification is sufficient

There is no requirement to prove a universal upper bound for every future HIP
runtime. CUDA's fixed opaque allowance is also a finite conservative policy
assumption. HIP may adopt a documented, pinned empirical policy with explicit
scope and margin. It must not present source ELF segments as total device usage,
copy CUDA's64MiB allowance, or derive entitlement from currently free memory.
The provisional Gemma128MiB allowance is not an architectural constraint.

**Owner and API responsibilities:**
- Backend supplies an immutable policy identity covering actual hardware/device
  class, HIP/driver/compiler versions and code identities, compiler flags, cache
  mode, permitted streams/concurrency, module/specialization bounds and supported
  dispatch/resource ranges. Begin with one exact tested device/runtime tuple;
  wider architecture-class claims require their own evidence.
- Backend derives module count, actual code bytes/load-segment extents, retained
  binary/metadata and compiler inputs from the complete concrete set. Explicit
  conservative opaque allowances cover driver context, loader/lazy allocations,
  compiler/RTC and dispatch scratch not owned by the managed pool. The policy
  distinguishes persistent costs from transient preparation peaks.
- Model adapter supplies immutable source, complete actual specialized kernels,
  intended workload bounds and original service/host owners. It never supplies
  a trusted byte total, edits opaque allowances or grants itself fit.
- Existing HipNativeAdmission callback consumes a selected backend policy and
  actual inventory, reserves through the SAME local ledger/global envelope and
  original host ResourceArbiter, and returns actual retained owners. Policy is
  budget/eligibility metadata; it is neither allocator nor source/math authority.
- Existing operator provider/device/host caps remain explicit independent limits.
  Whole-profile known+opaque+managed peak must fit before activation. A higher
  justified policy allowance is acceptable on the96GiB remote GPU; no automatic
  allocation of all free/remainder bytes. Policy mismatch or insufficient space
  refuses preparation before model launch, not a silent lower-assurance fallback.

**Lifetime correction required at the consumer join:**
- Detached NativeCacheBudget's original Arc<GlobalResourceLease>, shared ledger,
  source/service generation and fence are the right acyclic proof. No Client,
  actor or model-owner backedge belongs in cache-held admission.
- Successful stream completion+module unload retires module-private allocations.
  It does not prove HIP primary context, driver allocator cache, runtime heap or
  compiler process memory has disappeared. Do not release all such opaque charge
  simply because PreparedModules was removed.
- Retain persistent context allowance on the original backend context owner,
  using the same detached charged proof; release only at a genuinely established
  context destruction boundary. If the existing runtime cannot establish it,
  conservative process-lifetime retention is an honest initial policy.
- Compiler scratch retires after actual compiler/RTC object destruction; loaded
  binaries and their host charge follow module lifetime. Unknown load, launch,
  sync, unload or owner destruction retains applicable charges and fences.
- `reserve_preparation(kernels)` and `reserve_module(inventory)` alone do not
  document these different lifetimes. The trusted adapter must make the retained
  context/preparation/module ownership explicit before enabling Gemma, with no
  new dispatcher or independent completion mechanism. Existing context creation
  may precede model admission; report and charge bootstrap explicitly rather
  than claiming every driver byte was pre-admitted by a later module callback.

### Smallest actual qualification route

1. Run one **owned model-free backend unit** on the authorized remote HIP GPU.
   Pin runtime/compiler/device/policy identity. Use the actual prepared-cache
   implementation and real compiled artifacts, not mocked handles or caller
   byte totals. Exercise finite module/dispatch/resource bounds, lazy first
   launch, streams and compile concurrency allowed by that policy. It can use
   the actual Gemma specialization catalogue without granting Gemma semantics.
2. Observe cold context, compilation, load, first launch, warmed reuse, repeated
   prepare/retire and final process retirement. Compare known allocations with
   device and host observations. Choose and document conservative opaque margin
   for this finite envelope. Sampled peaks are observations, not exact peaks;
   report sampling limits and any unobserved allowance. Absence of OOM alone is
   not acceptance. No bounded-growth/leak claim from a single warm load.
3. Exercise original refusal, cancellation and unknown completion/unload/drop
   paths in fresh owned processes. Known retirement releases only its proven
   lifetime; uncertain paths retain original budget/handles and reject reuse.
   Retain source identity and exact cache set; sealed misses never compile late.
4. Then prepare the complete Gemma35 actual specialization set under that policy
   and whole source-derived resource inventory. Qualify full model source/math,
   all state domains, outputs, cancellation and ownership. Model-free transport/
   native lifetime evidence cannot replace this step. Actual model peak beyond
   the admitted finite policy fails that profile; revise/requalify policy rather
   than minting a larger budget during execution.

Backend qualification and complete-model qualification are separate useful
end-to-end units. Neither requires an unprovable guarantee for all HIP devices;
both require truthful finite assumptions and actual original ownership. The
local gfx90c hold remains independent of remote capability or profile results.

Batching continues independently: source-issued B4/B8 observation and a derived
22-root candidate envelope are source work until actual packets, mixed-frontier
state, workspace readers and original all-or-none publication are qualified.
No N1 enablement flag may substitute for those facts.

Coordinator disposition: Sol HIGH has authored the exact sealed issuer/member/
expert-range correction and real-source negative fixture in its guarded eight-
path donor. Review is complete; assembled compile/physical validation remains
pending. The HIP context partition/native owner implementation is a separate
Sol-owned unit with its actual consumer required before integration. Astra
continues architecture/planning only.

## Original cohort capture: public scheduling barrier and evidence limits

Architecture-only review, 2026-10-23. No implementation, compiler or GPU work.

Decision: first obtain finite B4/B8 reference packets with an explicitly declared
closed-cohort workload using existing public vLLM APIs:
`LLM.sleep(level=0, mode="keep")` → `LLM.enqueue(prompts, sampling_params)` →
`LLM.wake_up(tags=["scheduling"])` → `LLM.wait_for_completion()`.
Use this identical protocol in both reference lanes. Preserve original prompts,
greedy policy, EOS, feedback, all 320 source roles and actual scheduler decisions.
This is a workload boundary, not a scheduler, backend or cache override.

Source inspected: pinned approved vLLM overlay
`/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers/overlay/3d48be3a9656809e23715095eb422187fd959f80ff8ecdeab724e98dfcba213b/diff/usr/local/lib/python3.12/dist-packages/vllm/`:

- `entrypoints/llm.py:731` explicitly documents this all-enqueued barrier;
  public `sleep`/`wake_up` are at 809/842. `enqueue` alone is insufficient.
- `v1/engine/llm_engine.py:374`: level zero does not clear renderer cache.
- `v1/engine/core.py:925`: level zero pauses without cache clearing and returns
  before executor memory sleep. Scheduling-only wake skips executor memory wake.
- `v1/engine/core_client.py:993`: adds and utilities use the same input socket;
  synchronous utility calls wait for their result. Original engine pause completion
  includes actual device synchronization. No private engine call is needed.

Keep the eight existing whole-call collector slots if actual stock execution is
one complete cohort prefill plus seven complete cohort decodes. Prompt totals
233 (B4) and 471 (B8) fit the existing 512-token ceiling. The public queue barrier
is not permission to assume scheduler widths, absence of padding, or eight calls:
inspect and reject any unsupported actual trajectory. An early natural EOS may
honestly refuse this finite fixture; never mask EOS to obtain a batch. Barrier
startup and queueing belong to this named workload; its results do not establish
ordinary asynchronous arrival behavior or timings.

The proposed variable-width observation remains legitimate independent serving
work, with these boundaries:

- Separate global model-call ordinal, per-request logical step and native scope
  ordinal. Archive every actual request trajectory, feedback and state transition;
  map IDs to input ordinals and original cache ownership. Mixed prefill/decode
  metadata can reorder rows: prove the authoritative row/state permutation.
- Select only complete eligible unpadded, all-cached B4/B8 calls for finite native
  scopes using a declared deterministic rule. Never select rows out of a mixed
  call, truncate its event interval, or relabel it as an accepted width. Eight
  native scopes do not imply eight source calls. Bound metadata and archives for
  the declared request work; B8 can have up to 64 sampling calls if serialized
  under the current whole-prompt/single-token, eight-output fixture. Recompute,
  chunked prefill and additional scheduler events need their own bound or refusal.
- Plain/observed bitwise attribution requires identical actual scheduling after
  request-ID normalization: ordered request spans, logical frontiers, phases and
  effective arithmetic profile. A grouping mismatch is incomparable evidence,
  not a numerical failure and not permission to force dispatch. Current
  `validate_cohort.py` already compares normalized schedules; retain this gate.
- Both current lanes collect tensors/state. Their comparison establishes the
  incremental effect of native tracing, not absence of all observer effects.
  Record that distinction. Matching final tokens alone cannot establish full
  logits/state parity or resolve a schedule mismatch.
- Selected native packets establish their finite operation profile only.
  Unqualified mixed/tail prefixes cannot silently become an end-to-end Rust
  model grant, nor may archived states seed production roots. The sole existing
  ForwardPlan, independent request roots, whole-epoch publisher, derived 22-root
  admission and exact qualified [8,4,1] decomposition remain authoritative.

Lowest implementation seam: public barrier profile plus actual schedule checks
before a larger variable-trajectory archive redesign. No new runtime authority.

Closed-cohort initialization is also a distinct numerical profile: original
233/471-row prefill can produce different BF16 priors from independently executed
N1 prompt prefills. Before granting complete continuous serving, produce each
Rust initial root through verified source and genuine prompts under an admitted
initial path; compare every initial GDN/KV state, full-vocabulary logits and first
natural token. Carry those same independent roots into all cached cohort epochs
and the original all-or-none publisher. Existing N1 initialization is reusable
only with the required actual initial equivalence proof for this profile. If it
fails, qualify the actual cohort-prefill arithmetic or use a separately declared
reference schedule matching supported initialization. Decode-kernel captures and
matching tokens cannot substitute. Captured original priors may be explicit
isolated operation-test operands; importing them as production roots, or using
original feedback to repair divergent Rust state, never establishes end-to-end
support. Recording all eight calls is necessary evidence, not automatic initial
and cached qualification.

## 2026-10-23 focused review: cohort execution and reachable Gemma HIP v2

Architecture/source review only; no implementation, builds or physical jobs.
Reviewed MAIN resident execution/publication and Sol's batching DESIGN plus
Gemma37 `gemma-final-v2` against its guarded bases. Coordinator reports accepted
canonical47 B4 source discovery; full argument capture is separate and no Rust
B4 execution grant follows. Gemma37 reachable compilation is reported evidence,
not model or provisional-policy qualification. Earlier unequal-prompt cohort
numbers remain historical: this first current runtime unit is canonical47×4,
physical188 prefill followed by seven B4 decode calls.

### Cohort: approve a sealed body in the existing loop, with row-complete joins

Prefer `ResidentProgramBody::NativeCohort` plus separate physical cohort values.
Do not widen N1 `ModelValues` checks into accepting arbitrary multiplied extents.
The latter currently seals one recorded invocation and exact BF16 lease extent
in `vllm_cuda_qualification/model_values.rs:57-140`; that invariant is useful.
The cohort retains four independent logical255-node/327-value invocations and
maps each logical value to an explicit physical request-axis view. Physical188
is not a temporal188-token invocation. No first-row plan can authorize siblings.

Concrete changes required beyond adding an enqueue match:

- `resident_program.rs:1369-1474` runtime/state validation accepts one runtime and
  a unique `StateSlot` map. Add a closed Single/Cohort execution-binding variant;
  each cohort row binds its authentic record, runtime, prior/candidate bindings
  and borrowed phase permit. Preserve each row's domain namespace. Validate every
  row plus cross-row candidate-vs-all-prior/candidate/workspace range disjointness
  before the first clear/copy/kernel. Do not pass first-row placeholders through
  the single-row validator, or flatten duplicate slots into its map.
- `native_full.rs:601-690` requires exact invocation/phase, one launch and ordered
  original nodes. Preserve those checks for ALL cohort rows. One node order may
  be shared only after sealing identical source roles, mathematical kinds, SSA,
  numerical profile and state effects modulo explicit invocation/root/frontier
  bindings. Comparing only operation discriminants or dimensions is insufficient.
- Keep the sole loop at `resident_program.rs:1654`. Each node dispatches one
  qualified physical B operation; it does not walk each request's full model.
  Certify every actual B output range before liveness reuse. Integrate View's
  special completion path, storage-node finish, deferred certificate flush,
  untouched-domain carries, terminal candidate certification and output lowering.
  Existing terminal proposals at1759-1792 use one plan/binding set: return grouped
  per-row proposals tied to their real plans/roots, never a first-row result.
- `resident_leases.rs:497` already binds all roots on one flight; completion at
  1007 checks exact root order and splits authentic per-row Ready outputs. This
  physical mechanism accepting width2 does not grant B2 arithmetic. Preserve the
  original `resident_inference.rs:249` publication transaction: all fallible
  sibling validation/transitions before any infallible swap/commit.
- Cost and native eligibility precede activation: actual Bprefill/decode kernels,
  source/ABI/ranges/row permutations, certifiers, source controls, Bvalue arenas,
  RMS/GDN/FA scratch, all48 candidate planes per row, full[B,V], immutable12-byte
  packets, readback and host bindings. Reject a missing packet before submit;
  never scale an N1 grid or receipt. Native compilation/storage registration is
  preparation work, not an operation-node allocation.

Call-local body ownership is necessary for the22-root bound. Existing N1
completion installs its bound program into `workspace.program` and retains the
previous one (`native_fragment/short_hf/completion.rs:163-170`). Do not copy that
pattern for a cohort body carrying eight prior roots. Retain the body on the real
flight; after known readback drop it/prior owners before admitting another
candidate cohort. Pending keeps candidates/Ready/packets/input metadata and the
single cohort reservation through commit/drop. Unknown work keeps its original
owners and fences reuse. The current model admits at most eight live sessions
(`short_hf/loaded.rs:281`);22 means8current+8candidate+2externalPending+2snapshot+
2displaced, not permission for16 live current roots. Shared mutable workspace
reuse and diagnostic readers must obey that same reservation/lifetime proof.

Actual closed-cohort prefill is required. Source diagnostics report widespread
initial producer differences from N1; N1 initialization is not an equivalent
shortcut. Execute genuine188 prefill from source/prompts and private roots,
compare every required producer/state/fullV/first token, then retain those roots
through seven B4 calls using their own committed feedback. Isolated packet tests
may use explicitly labeled captured operands; production initialization may not.
Mixed frontiers, B8 and ordinary asynchronous arrivals remain distinct profiles.

### Share prepared semantics; bind request authority late

`short_hf.rs:213-219` currently charges4MiB and calls `bound.record()` for every
request/call. `RecordedVllm08Model` and `ForwardPlan` contain invocation provenance
(`core/.../vllm_model.rs:658-687,829`), so pointer-sharing an existing first-row
record would be wrong. Do not multiply whole-plan rebuilding/hashing by B as a
permanent hot-path design.

Minimal convention inside the existing record/binding API:

- Prepare immutable topology/SSA, source role joins, numerical contracts, storage
  schedule and finite native candidates once per actual shape/profile; retain by
  Arc. A semantic ValueId is resolved together with its row invocation, not alone.
- Late bind small original input/feedback, attempt, phase, state leases and runtime
  controls. Source-issued construction proves the binding belongs to that prepared
  recipe; node launch checks the indexed current binding and original permit.
- Separate invariant descriptor data from request provenance in the existing
  model record; do not add a template IR or second recipe interpreter. Retain
  source equality and original row-specific state effects. Cache keys cannot
  erase phase/layout/numerical differences merely because both are BF16.

Conservatively priced existing per-row records can bootstrap the first correctness
unit. Record their cost honestly and remove repeated immutable work before a
zero-allocation/performance claim; broad compatibility facades are unnecessary.

### Gemma37v2: concrete API correction and shared backend policy boundary

**Must fix:** donor `resident_inference/gemma_text.rs:77` exposes a safe public
`load_hip_qualification(..., source_owner: Arc<dyn Any>, ...)`. `Arc<()>` satisfies
that argument without retaining the managed SnapshotLease or close guard. The
facade in `zerogravity/src/gemma4_cpu.rs:182` correctly issues GpuSourceRetention
under its lifecycle lock, but the public lower API bypasses that proof. Sol
confirmed no other trusted boundary exists.

Smallest correction: explicitly unsafe low-level constructor whose documented
contract requires the authentic managed source/snapshot lifecycle guard bound to
this exact source, retained through native/flight/quarantine retirement. Keep the
facade safe and put one justified unsafe call after its actual context/model/life
checks. A sealed safe token is suitable only if an existing authentic lower-crate
issuer can carry this ownership without a new registry or crate cycle. A safe
public token constructor/trait implemented over arbitrary Any is not a fix.

**Concrete reuse correction:** donor `gemma_text/native_owners/hip.rs` combines
family catalogue with hardware identity and context/compiler/module policy.
Move the concrete finite policy/enforcement into a shared HIP backend preparation
module using the existing NativeCacheBudget and HipNativeAdmission. Family code
supplies genuine source, complete generated specialization catalogue and model
costs; it cannot choose opaque allowance or manufacture backend fit. No new
allocator, dispatcher, completion type or universal backend-policy framework is
needed. Keep fixed cold-process qualification distinct from normal operator
resource configuration; a family switch must not upgrade a live host domain.

The audited ownership spine is otherwise sound in direction:

- Actual generated catalogue in `math/prepared.rs` derives specializations from
  existing plan and projection tiles; runtime-only token/position/M/contraction
  parameters are not fake extra kernels. Sealed cache misses refuse. Constructor
  coverage still needs actual whole35/cancellation execution; compilation alone
  cannot prove the complete launch set.
- Native admission uses detached original ledger/global envelope/generation/fence,
  with no Client/model backedge. Admission retains source guard; persistent context
  retains source-neutral context charges. Original actor syncs all streams and
  unloads every module before releasing set charges; failed paths retain/fence.
- Provisional provider16GiB consists of context4GiB plus model ledger12GiB under
  physical64GiB. Module8GiB opaque+actual code, context host2GiB and compiler8GiB
  are fixed conservative policy assumptions, not native-four qualification or
  ELF-derived physical bounds. Native code/retained bytes are actually admitted.
- `layout.rs` computes checked family/model costs. `GemmaTextBody::prepare` subtracts
  already-admitted native bytes before comparing managed demand with remaining
  capacity; the outer loader compares full model/native cost with the model
  partition. Preserve this nonduplicated accounting. Report persistent context
  separately, but include it in complete process peaks against the same physical
  and host caps. Shared ResourceArbiter remains the ultimate admission boundary.
- Source-page authentic member/codec/axes/ranges and actual output-buffer sync
  govern staging reuse on the original flight. Gemma launch enters the same
  `launch_impl`; row StateRoot/Ready/publication remain original. Backend transport
  and lifetime reuse can later serve CUDA/Vulkan; numerical qualification cannot
  transfer merely because transport is portable.

Smallest assembled Gemma unit: close the public source-retention trust boundary,
extract this concrete backend policy, then actual pinned full-catalogue preparation
and whole35 source/producer/fullKV/logits/natural-state/cancel/unknown/retirement
qualification through the same facade. Keep bootstrap-before-context-partition
window explicit; successful model retirement leaves persistent context charge
until process retirement. No serving or general HIP grant from this review.

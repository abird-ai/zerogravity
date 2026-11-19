Author: Project owner
Co-authors: GPT-6
Owner: @pvl

# Batching, paged attention and hybrid prefix state

Design follow-up, 21 Oct 2026. Everything under implementation and acceptance
below remains pending; this document establishes no new runtime qualification.
Current source/tests outrank this inspection snapshot. Parent queues:
[serving](TODO.md), [memory](../memory-and-streaming/TODO.md).

## Existing scope and missing work

| Capability | Inspected scope | Remaining work |
| --- | --- | --- |
| Native request batching | Original GGUF .8B/2B CUDA B2/B4/B8; B1 survivors, unequal causal positions and compatible refill | Integrate and independently qualify every additional numerical profile/backend |
| Prompt processing | Native waves consume one teacher token per request; B1 supports bounded prompt quanta | Actual multi-token temporal prefill, qualified chunk partitions and fair prefill/decode scheduling |
| KV/recurrent ownership | Private per-session state, fixed-capacity contiguous KV allocations, original admission/leases/publication | Paged state allocation, block-table attention and complete-domain prefix reuse/COW |
| GPU HTTP scope | One worker, context512, at most8 active requests; fused mode is opt-in | Separately qualify larger contexts/concurrency; CPU8K results do not qualify GPU8K |
| Concurrency | Bounded request interleaving and actual multi-row kernels | Independent-stream/event-backed overlap, with physical lifetime tests |
| Performance | Retained bounded tests/measurements, with explicit limitations | Profile-matched short/long-context and cache/batching ablations against pinned vLLM |

These are source inspection plus retained test records, not a fresh physical
run. Accepted details remain in [FUSED_GPU_BATCHING.md](FUSED_GPU_BATCHING.md).
Do not treat its historical limitations, or those in the plan README, as
overriding later named acceptance in the live TODO.

## Shared engine and numerical profiles

- Keep one mathematical ForwardPlan, model/session lifecycle, ResourceArbiter,
  worker/queue and StateRoot publisher. Add typed cache/attention lowering to
  those paths; do not introduce a separate model or serving engine.
- Preserve the accepted GGUF/reference profile and separately qualify the
  pinned vLLM CUDA profile. Storage format, quantization, numerical arithmetic
  and batching policy are independent dimensions, not two fixed engine modes.
- Bind source/loaded tensor dtypes, activation quantization, numerical/phase
  contract, position policy and state schema to compiled-plan/cache identity.
  Select the profile when creating the model/session. Reject incompatible
  cache/state reuse or an unqualified mid-generation profile switch.
- Extend the actual CLI/session/HTTP path after whole-model qualification;
  isolated operators or a diagnostic driver cannot establish serving support.
  Keep bit-exact gates where required and existing declared tolerances
  elsewhere; new paging/fusion must not silently change either contract.

## Paged state allocator — memory plan

- Add a finite page pool and per-session logical-to-physical block tables
  behind existing state bindings. Record page generation, domain/layout,
  valid extent, source/profile compatibility and reader ownership. Admit
  tables, metadata, scratch and temporary/COW pages before allocation/submit.
- Allocate KV as context grows; define page size, alignment, tail validity,
  exhaustion and accounting. Preserve the contiguous implementation as an
  independently qualified baseline/policy, not a second session lifecycle.
- Keep recurrent matrices and convolution histories as typed state domains
  with their own checkpoint/layout policy. Do not pretend a GDN state matrix
  is a token-indexed KV page or infer compatibility from equal byte sizes.
- Committed shared state is immutable. Candidate writes copy only required
  shared pages/domains, including a partially filled tail. Keep prior and
  candidate roots separate until original Ready and atomic publication.
- Pin every encoded/captured/queued/submitted reader, not just executing
  kernels. Reuse a page generation only after all real readers retire.
  Cancellation/drop must preserve in-flight ownership; unknown completion
  retains original quarantine/fence and cannot manufacture reusable capacity.
- Define pool pressure/refusal and safe eviction of unpinned cached state.
  Eviction loses only reusable cache, never a live root or referenced page.
  CPU/SSD migration, persistence and recovery retain their separate physical
  qualification gates; paging alone does not establish them.

## Paged attention kernels — serving plan

- Lower original KV append/read and QK/softmax/AV through logical block tables,
  preserving each row's causal position, masks, RoPE, head mapping, dtype and
  reduction order. Physical page adjacency must not affect outputs.
- Preserve original numerical contracts, including inactive tails and signed
  zero where specified. A different FlashAttention/fusion/reduction route
  needs its own explicit qualification; paging is not permission to change
  arithmetic silently.
- A bounded gather-to-contiguous route may establish initial correctness, but
  must account scratch/copies and cannot be advertised as an efficient native
  paged kernel. Qualify direct page-table kernels and measure the difference.
- Verify appends across page boundaries, partial/empty tails, shuffled physical
  pages, row-specific lengths and B1/B2/B4/B8 transitions. No full-prefix copy
  on every append in the intended fast path; measure actual copies/dispatches.

## Complete-domain prefix reuse and COW — memory plan

- Build lookup identity from exact source/component and effective loader/quant
  profile, tokenizer/template/processor/adapter, input tokens/content, position
  policy, numerical contract, state schema and sharing authorization. Matching
  prompt text alone is insufficient. Physical pointer/device location is not
  semantic identity; transfers require qualified conversion and real Ready.
- Implement the greatest jointly restorable committed frontier across all
  model-required domains: full/sliding KV, recurrent/conv state, indexer,
  draft/grammar/media state where applicable. Missing GDN state forces an
  earlier complete checkpoint or recomputation, not a KV-only full-hit claim.
- For original small models, compare all42 domains; original27B requires
  all112 after that model is qualified. Define checkpoint cadence and bounds,
  partial-hit/recompute cost, eviction and invalidation. Token-page boundaries
  need not coincide with recurrent checkpoints.
- Share only immutable authorized state. Fork independent StateRoots; COW all
  changed shared domains before append. Preserve private sampler/RNG/grammar
  continuation and tenant/source boundaries. Sharing a prefix must not let a
  sibling's generation, cancellation or release change another session.
- Report actual common-frontier reused tokens and restore/recompute cost.
  Keep cache-disabled, partial-hit and complete-hit paths observable; a hash
  lookup or KV-only hit is not a full hybrid hit.

## Continuous batching, temporal prefill and overlap — serving plan

**15 Nov 2026 fair-interleaving sub-slice accepted:** the first ordered step
below—fair interleaving in the existing completion path—is implemented in the
shared engine rather than in a Qwen/CUDA adapter. `WavePolicy` carries finite
round-token and prefill-age controls. Wave selection protects every eligible
decode row, retains deferred prefill owners, services the oldest aged prefill
after two deferred rounds, and rejects a configuration that cannot hold one
maximum admitted prefill wave plus all active decode rows. Existing
source/phase/frontier grouping, native band authority, Ready/publication,
cancellation and backpressure semantics are unchanged. 18 wave tests, 14 worker
tests, strict engine clippy and production CUDA composition checks pass. The
strengthened staggered-arrival HTTP proof physically passes on both approved
0.8B/2B CUDA sources with exact CPU IDs/text/usage and final 41MiB.

Every command in `scripts/check.sh` has a clean terminal pass on these exact
bytes under isolated `native-gpu`, including the full GPU feature-boundary gate,
CPU workspace/clippy, serving/lifecycle, acquisition and release/installer/Nix
blocks. Repeated attempts to run the monolithic shell wrapper as one connector
job disappeared during ordinary cached builds without a compiler/test error,
surviving child, cgroup OOM or GPU residue; those wrapper invocations remain
incomplete rather than passes. The complete constituent command set is the
acceptance evidence for this scheduler-only slice.

**15 Nov 2026 temporal-prefill B2 functional acceptance:** the resident fused
layer has a reusable equal-chunk temporal transaction for B2/B4/B8. A chunk of
2..32 teacher tokens reuses the accepted native body repeatedly against private
candidate domains, retains the original root as the sole publication authority,
emits no intermediate logits/token packet and returns one terminal
`PreparedResidentCohort`; dropping/refusing the chain never publishes an
intermediate root. The ordinary cohort adapter consumes the receipt and advances
the scheduler cursor by the whole chunk.

The first full physical target remains original0.8B B2. The retained
`zerogravity-gpu` example is now the width-parameterized `temporal_prefill`;
B2 checks CPU/B1/all42 state, abandoned-candidate rollback, 4-token state-only
publication, 2-token terminal greedy producer lineage and final reserved0.
Production HTTP executes prompt10 as B2 chunks4+4+2 with CPU ID264/text ` a`,
usage10/1, exactly3 publications, no failures/refusals and final41MiB.

B4 and B8 retain full original0.8B debug qualifier passes, and original2B now
independently passes full B2/B4/B8 with the same CPU/B1/all42/root/producer
contract. 2B retained-band bytes are 1,589,873,428 / 1,833,045,908 /
2,319,390,868. Two unsupervised2B B4 attempts disappeared before temporal work;
the heartbeat-supervised B4 run passed fully.

Signed `eaec62d` promotes B4 and signed `8e6f14d` promotes **B8**. Production
HTTP passes both original0.8B and2B B4 with prompt10 -> exact chunks4+4+2, three B4 cohorts,
two state-only + one greedy, three root publications, mixed JSON/SSE CPU
text/usage exact and clean retirement; B8 passes the same exact production
contract on both sources, and retained2B B2 HTTP also passes. Production
therefore selects temporal4 for B2+B4+B8 with equal-frontier prefill. One earlier
supervised0.8B B8 HTTP attempt disappeared after test start without a semantic
result and remains incomplete.

These physical cells use the debug Rust profile; canonical release-profile B2 is
still open after a fresh heartbeat-supervised thin-LTO build disappeared during
CLI compile/link before test execution, without terminal child/Rust/OOM/Xid/
residue evidence. Do not loop that unstable build. Temporal/fair serving is now
the accepted foundation.

**15 Nov paged-state control foundation:** signed `0027c28` adds a
model-neutral `zerogravity-core::paged_state` primitive without selecting it in
resident execution. `PageCompatibility` requires the bound source plus
numerical/domain/layout identities; `PageGeometry` requires a positive page
extent that is an exact multiple of the caller's semantic append unit, so a KV
adapter cannot split one token row accidentally. `PagedBlockTable` is immutable:
full pages share their existing owner; a partial valid tail always COWs before
append; every fresh page is obtained before candidate-table mutation; allocation
failure therefore rolls back through the caller handle's RAII; and
`PageAppendPlan` retains source/target/write owners until the existing physical
copy/write authority accepts them. Prefix truncation shares only compatible
immutable pages and preserves a valid-tail extent. Eight focused tests cover
compatibility, geometry, full/tail pages, partial-prefix COW, exhaustion
atomicity and owner pinning; all178 core tests, strict core clippy and
`qwen35-cuda` composition pass.

**15 Nov physical page-owner follow-up:** signed `54340a6` adds a homogeneous
`ResidentPagePool` as a one-plane specialization of the existing
`ResidentPool`; checkout still returns the original `ResidentLease`, with
fresh allocation generation and the same backing ledger/event/quarantine path.
`ResidentPagedTable` binds that pool to `PagedBlockTable<ResidentLease>` and
rejects non-RecurrentState or physical/semantic alignment mismatch. Its physical
CUDA qualifier passes page_bytes2048/max_pages3: one partial-tail fork plans COW
onto fresh generations, the third backing page exhausts the finite pool without
mutating the prior table, a cloned table keeps its pages non-idle, an initialized
page remains pinned through actual GPU readback and returns only after completion,
and an actually submitted H2D under an armed Write which is then abandoned fences
the service/quarantines the page. Pool idle falls3->2 and stays2 after diagnostic
sync; 2 handles remain quarantined. The qualifier is ownership/lifetime evidence,
not a payload-COW numerical pass.

Allocator/COW below therefore remains **partially open**: physical page
allocation/reuse/pressure/readers/unknown-completion are accepted, but the actual
partial-tail copy/write was still open at `54340a6`.

**15 Nov COW materialization follow-up:** signed `1eb6e0f` binds the private
candidate and `PageAppendPlan` together so a foreign plan cannot be mixed with
another table. The adapter derives one deduplicated original-flight access set:
the committed partial tail is an initialized Read and every fresh page is a
Write. CUDA materialization first zeroes each fresh page (correctness-first
inactive-tail policy), then uses the existing runtime word transport for the
partial-tail device copy and overlays host append bytes into the planned ranges.
No new completion/Ready/root authority is introduced; the existing
`ResidentSubmission` owns all accesses and the existing completion path marks
the fresh pages initialized only after physical readback in the qualifier.

RTX3060 physical exactness passes with page_bytes2048/unit_bytes512:
first append produces valid1536 + zero tail; second append copies the immutable
1536-byte source tail, appends1024 across that COW page and one new page, yields
valid extents[2048,512], leaves the old page byte-exact unchanged, keeps the new
inactive tail zero and retires all3 pages. The first attempt passed all
copy/write/source checks but its final idle assertion correctly caught retained
qualifier-owned access/readback clones; those diagnostic owners were dropped at
their completion points and the exact rerun passed. The previous pressure/
reader/submitted-unknown physical cell also passes unchanged.

Allocator/COW is still **partially open at model composition**: the page owner and
payload COW materializer are accepted, but `StateLayout::Kv`/attention still
use contiguous `StateBuffers::Kv` and no paged attention/prefix cache is
selected.

**15 Nov typed paged-KV/gather follow-up:** signed `3ac2059` adds
`ResidentPagedKvDomain`, constructed only from a checked
`StateRequirement { StateLayout::Kv, .. }`. It preserves the model-declared
slab/capacity/physical type, binds page size to an integer token-row count,
requires row/page allocator alignment, gives key and value tables distinct
source/numerical/domain/schema compatibility identities, and keeps their logical
frontiers paired. `ResidentPagedKvAppend` composes the already accepted
candidate/COW materializers; the gather adapter takes initialized page Reads and
fresh exact-size contiguous K/V staging Writes in one original
`ResidentSubmission`, zeroes staging, then copies each valid page range in
logical order with the reusable runtime word transport. It is intentionally a
correctness adapter: copies/dispatches are observable and it is not a direct
paged-attention claim.

The retained RTX3060 boundary qualifier uses BF16 row_bytes512, four tokens per
2048-byte page, capacity7 and a strict four-page physical pool. It appends/gathers
frontiers3 (=page_size-1),4 (=page_size),5 (=page_size+1). Exact contiguous K/V
parity passes at all three frontiers with gather dispatches2/2/4. The3->4
transition makes two COW copies (K+V), while4->5 shares the full first K/V pages
and reuses the two retired physical backings rather than growing the pool. Both
retained prior domains remain byte-exact after child appends, inactive staging/
page tails stay zero, capacity overflow refuses and all four pages retire.

Two useful fail-closed observations are retained: (1) the first physical run
found that post-Ready `into_candidate` was incorrectly going through the
pre-submit “target must be uninitialized” check; plan identity and freshness are
now separate, so submit requires fresh targets and publication requires exact
initialized targets. (2) the next run passed data/gather checks but the test
expected six allocated pages; actual safe reuse kept the peak at four, so the
physical cap was tightened to four and the final-current run passed. A separate
link attempt hit ENOSPC because generated `target/debug/incremental` had grown
to about29GiB; only that disposable build cache was deleted (free space recovered
from about3.9GiB to24GiB). One later exact relink wrapper disappeared during
compile without a diagnostic; a single warmed retry completed and the exact
binary passed. Lower COW materialization and pressure/reader/submitted-unknown
tests remain green, as do production `qwen35-cuda` composition and strict
production clippy.

Allocator/COW remains **partially open at production session/model composition**.
Signed `53336b3` lands the canonical domain/root representation even though
ordinary session construction does not select it yet. `Domain` may retain a
`ResidentPagedKvDomain` and its exact page lease/generation cohort while
`StateBuffers::PagedKv` is a no-handle marker. `ResidentModel::validate_domains`
checks the model's declared `StateRequirement::Kv`, original source/service
generation, exact page-owner set and initialized state. Native batch transport,
range qualification and old diagnostic readback reject the marker until a
contiguous staging representation is explicitly bound. The single-session commit
path now calls `publish_domain_swap`, which is only a factorization of the
existing `publish_if_usable -> prepare_publish -> state swap -> commit` authority.
No second publisher or root lifecycle exists.

The retained RTX3060 root qualifier constructs physically Ready paged KV,
publishes frontier0->3 and3->4 through that exact helper, verifies root changes
and authority bindings, keeps only2 initialized page owners at each committed
frontier, and re-gathers both current frontier4 and displaced frontier3 exactly
after publication. Marker:
`P12_PAGED_KV_DOMAIN_PUBLISH_PASS frontiers=0,3,4 root_changes=2
canonical_marker=true frontier3_pages=2 frontier4_pages=2 boundary_cow=2
displaced_parent_exact=true original_session_authority=true
all_pages_retired=true`. The final-current bridge, 3/4/5 boundary, COW
materializer and submitted-unknown/quarantine tests also pass; production
`qwen35-cuda` composition and strict clippy pass.
Before that final ownership switch, signed `dc6a188` closes the missing
device-side bridge. `ResidentPagedKvStaging` is a transient pair of ordinary
resident leases validated against the same `StateRequirement::Kv`, and exposes
the old `StateBuffers::Kv` ABI without becoming committed state. A new scatter
path accepts a contiguous candidate lease already written by the same
`ResidentSubmission`: fresh target pages are zeroed, any partial committed tail
is copied from immutable prior pages, and only the append range is copied from
candidate staging into the planned target ranges. The flight must retain the
candidate staging Read (a submitted Write satisfies it), all COW source Reads and
all page Writes; no host readback or second completion authority is introduced.

The retained RTX3060 bridge qualifier starts from a physically Ready paged
frontier3, allocates full-capacity prior/candidate BF16 staging, gathers the old
K/V pages into prior staging, performs the exact two-plane contiguous prefix copy
plus one new-row write that stands in for the unchanged ordinary program, then
scatters that candidate row into a frontier4 paged child. One original flight
covers the whole bridge. Exact pass marker:
`P12_PAGED_KV_BRIDGE_PASS prior=3 candidate=4 page_tokens=4 row_bytes=512
gather_dispatches=2 prefix_copy_dispatches=2 scatter_cow_dispatches=2
scatter_append_transfers=2 gpu_only_after_input=true prior_immutable=true
contiguous_abi_exact=true all_pages_retired=true`. The final-current 3/4/5
boundary, lower COW materializer and submitted-unknown/quarantine tests all pass,
as do production `qwen35-cuda` composition and strict clippy. The materializer
counter is now named `append_transfers`, since the same plan can be populated
by host upload or device copy.

**Admission decision for the next slice:** do not carry both a full prior and a
full candidate staging pair on top of a paged committed root. Ordinary admission
counts a finite inventory of complete state roots: the base path is explicitly
prior+candidate, with additional roots charged as whole state extents.
Canonical paged state + prior staging + candidate staging would therefore become
three state-sized envelopes at full context and can exceed the provider cap.
The first production route should use one full-capacity **in-place** K/V staging
pair. Gather the committed paged prefix into it, bind an explicit paged-staging
mode that permits prior/candidate identity for KV only, skip the old KV
zero/full-prefix-copy in `KvAppend`, append new rows in place, let the existing
FlashAttention read that same contiguous candidate, then scatter only newly
produced rows into private pages under the original flight before the accepted
Ready/root swap. Conv/Recurrent state stays on the existing contiguous typed
path. Qualify continued page_size-1/page_size/page_size+1 and multi-page
full-state parity before direct block-table attention or complete-domain prefix
identity/intersection. Event/lease-backed physical overlap remains later.

**15 Nov in-place ordinary execution follow-up:** signed `ef9e4d4` implements
the execution half of that admission decision. `StateBinding` now carries an
explicit mode. Distinct remains the default everywhere, including native batch.
`KvInPlace` is accepted by ordinary CUDA only for a KV requirement whose prior
and candidate are the exact same two resident leases by ID, allocation
generation and extent; those staging leases must be uninitialized at validation,
so an already-published canonical root cannot be mutated through this mode.
Within the real `KvAppend` host path, `prepare_kv_candidate` preserves the old
Distinct zero+prefix-copy behavior but returns zero launches for KvInPlace.
The actual append kernel and FlashAttention candidate binding remain unchanged.

The retained physical RTX3060 test creates fresh resident BF16 K/V staging,
submits exactly one Write access per plane, writes a three-token committed
prefix + zero tail under that original flight, enters KvInPlace preparation,
then executes the real ordinary Qwen KV append for token4. Exact marker:
`P12_ORDINARY_KV_INPLACE_PASS base=3 rows=1 capacity=8 kv_width=512
preparation_launches=0 prefix_preserved=true new_row_exact=true
inactive_tail_zero=true unique_write_access=true all_staging_retired=true`.
The same final binary passes canonical paged root publication, GPU
gather/scatter, 3/4/5 page boundaries, lower COW materialization and
submitted-unknown quarantine; production `qwen35-cuda` composition and strict
clippy are green.

Production selection still requires three coupled changes; do not land only one:
1. split the existing complete-root pool into a **hybrid admitted inventory**
   whose physical sum stays within the already charged root bytes: contiguous
   Conv/Recurrent committed/candidate planes, finite KV page pools and one
   full-capacity K/V staging pair per active ordinary execution;
2. update `validate_prepared`/Ready coverage for immutable shared full pages:
   a candidate page may be inherited only when it is the exact initialized owner
   from the current root, while fresh/COW pages must be actual submitted Write
   outputs. Never turn shared pages into fake Writes merely to satisfy Ready;
3. implement paged snapshot/read/restore over page order/valid extents. The
   no-handle `StateBuffers::PagedKv` marker must not make snapshots silently
   empty or bypass source/generation checks.

After those gates, bind the accepted single staging pair, gather paged prior,
run ordinary KvInPlace append/attention, scatter only appended rows, and publish
through the existing root authority. Then qualify page_size-1/page_size/
page_size+1 and multi-page full-model continuations.

**15 Nov hybrid semantic/admission follow-up:** signed `93468b1` closes gates2
and3 above and turns gate1 into an explicit byte plan.
`ResidentPagedKvDomain::snapshot_planes` exposes K-then-V page owners with typed
physical layout + valid element counts. `ResidentSnapshot::read_domains` now
builds one explicit plane list for contiguous and paged domains, retains original
Read leases, validates finite values, checks partial-page inactive tails are +0
and reconstructs the same flat K-then-V active values as contiguous snapshots.
Restore uses the same factored `publish_domain_swap`; physical frontier3 partial
and frontier4 full snapshots plus4->3->4 restore are exact.

`canonical_write_outputs` is now the pre-publication cohort authority used by
`validate_prepared`. For paged KV it calls
`required_write_leases_from(current)`: source/numerical/domain/schema/page
geometry must match; inherited pages must be the exact same logical lease at the
same plane/index, must remain full with unchanged valid extent, and any current
page moved to another position is rejected. The retained boundary qualifier now
proves3->4 requires2 COW Writes, while4->5 owns4 candidate pages but Ready needs
only2 new page Writes because2 full pages are inherited.

`HybridStateInventoryPlan` is a pure admission contract, not a pool installer.
Inputs are the actual StateRequirements, old state_bytes/root count, number of
simultaneous staging sets, explicit page_tokens, explicit max_append_tokens and
allocator alignment. It derives:
- contiguous Vector plane layout and roots*vector_bytes maximum;
- one complete K/V staging layout and staging_sets*kv_bytes maximum;
- one homogeneous page layout and the remaining page-byte budget;
- pages per committed root plus worst candidate-page growth across **every KV
  domain** for every active staging set.
It refuses if page geometry is heterogeneous, state-byte accounting differs, or
remaining page capacity cannot hold all active committed roots plus candidate
forks. Unit proof with two KV domains/page4/append4 shows4 old roots -> vector
roots + one staging set +24 pages exactly equals the old byte cap; roots2 is
refused for missing fork headroom.

Next: construct the actual `ResidentPool`/ `ResidentPagePool` objects from this
plan using source-owned metadata, install them only in the ordinary source-call
path, and bind session allocation/execution. Page-token policy remains explicit;
the qualification value4 is not a production default.

**15 Nov physical hybrid-pool follow-up:** signed `cb57e1d` constructs the
pool substrate but deliberately does not select it in production serving.
`PooledStateStorage` has explicit Contiguous/Hybrid variants. The Hybrid owner
contains a finite Vector root `ResidentPool`, a finite full-K/V staging
`ResidentPool`, one homogeneous `ResidentPagePool`, the accepted inventory
plan and the numerical contract. The verified-source installer reserves distinct
metadata workspaces for all three pools before setting the one model storage
authority. Initial hybrid root checkout zeroes only Vector planes and constructs
empty paged KV domains; candidate checkout refuses until the current root can be
forked rather than fabricating an empty candidate.

Production compatibility identity is also no longer caller-invented:
`ResidentPagedKvDomain::empty_for_contract` maps each
`SourceNumericalContract` to a stable explicit tag and encodes
version/physical-type/slab/capacity into a checked collision-free 64-bit layout
identity within 24-bit slab/capacity bounds. Page byte/unit geometry remains
separate in `PageGeometry`.

Physical pool qualification holds all admitted backing simultaneously:
`P12_HYBRID_POOL_PASS old_root_budget_bytes=67584 vector_sets=4
staging_sets=1 max_pages=24 simultaneous_exact=true empty_paged_root=true
idle_reuse=true final_resident_bytes=0`. The same exact binary passes the six
previous paging/KvInPlace cells and production strict checks.

**15 Nov current-root execution follow-up:** signed `b9b04aa` lands that bundle
and integrates it into the ordinary executor while leaving the source loader on
the contiguous pool. `OrdinaryCandidateState` keeps the existing Contiguous
domain/binding path and adds Hybrid only when Hybrid storage was already
installed. The Hybrid bundle owns:
- one private Vector candidate checkout;
- one complete transient K/V staging checkout;
- every current paged KV owner needed for gather;
- one `ResidentPagedKvAppend` per KV domain until physical completion.

Its access cohort is deduplicated by exact lease ID/generation/extent/role/service;
Write dominates a duplicate Read only for the exact same lease. The original
ordinary submission then executes:
1. paged gather into staging;
2. unchanged ordinary source body using `KvInPlace` bindings;
3. paged scatter from staging into fresh/COW page targets;
4. original physical completion with Ready outputs = Vector candidates + only
   fresh/COW page targets;
5. append-plan -> canonical paged Domain conversion only after Ready.

State-effect receipts compare against each binding's actual candidate handles, so
the unchanged contiguous path still sees its normal candidate handles while
hybrid KV sees staging handles. Access deduplication is shared but fail-closed on
identity mismatch. Fresh/COW output order is deterministic per K/V plane and
matches post-completion `canonical_write_outputs`.

Physical RTX3060 composition with one BF16 Vector domain + two qwen35_08 KV
domains at base3 -> frontier4 passes:
`P12_HYBRID_CANDIDATE_PASS base=3 rows=1 kv_domains=2
gather_dispatches=4 scatter_dispatches=8 ready_outputs=5
canonical_ready_order=true paged_frontier=4 vector_ready=true
continuation_exact=true inactive_tail_zero=true
old_root_budget_bytes=135168 final_resident_bytes=0`.
The first run's staging-pool exhaustion was caused by the qualifier retaining its
submitted-access lease clones across a deliberate reuse checkout. The next run's
final-retirement failure was caused by the qualifier retaining its canonical
comparison lease clones. Dropping those test-owned proof vectors made both
lifetime checks pass; production bundle ownership did not change. Two broad
link wrappers also disappeared late without terminal Rust/OOM/Xid evidence; a
materially narrower lib-only link completed and the final-current eight physical
cells pass.

Next: make page policy explicit at source-call load time and select
`install_window_hybrid_state_pools` only for that policy. Keep the current
contiguous default. First qualify original0.8B/2B B1 full logits + all42 state at
page_size-1/page_size/page_size+1 and multi-page, with cancel-before-submit,
in-flight cancel, abandon-after-Ready/root rollback and snapshot/restore. Then
adapt native B2/B4/B8 state transport and production HTTP. Do not begin
complete-domain prefix reuse until those state gates pass, and do not begin
event/lease physical overlap until prefix/state ownership is accepted.

**15 Nov explicit source-policy follow-up:** signed `1413f72` lands the policy
seam without publishing a new high-level serving capability.
`CudaSourceStatePolicy` has two exact modes:
- `Contiguous`: the historical root-pool path;
- `Paged { page_tokens, max_append_tokens }`: hybrid Vector/staging/page pools.

`ResidentModel::load_ordinary_cuda_source` remains a Contiguous compatibility
wrapper. `load_ordinary_cuda_source_with_state_policy` is the explicit internal
selector. Paged currently requires native_rows=0; zero page/append bounds refuse.
`HybridStateInventoryPlan` rejects max_append_tokens above state capacity, and
`HybridPooledStates::candidate` returns `Limit` before checkout when an actual
call exceeds its admitted append bound. Production qwen source registration
passes Contiguous explicitly. No high-level paged `ModelLoadRequest` is exposed.

The source-real B1 differential opens one
authenticated original HF source, runs a complete Contiguous model first, shuts
it down, then runs the same source through Paged. It compares every F32
full-vocabulary logit bit and every all42 state-domain word at frontiers
`page_tokens-1`, `page_tokens`, `page_tokens+1`, and
`2*page_tokens+1`. The last chunk is one full page, so the final checkpoint is
genuinely multi-page. On success it prints
`P12_SOURCE_PAGED_B1_PASS ... full_vocab=exact all42=exact
contiguous_reference=true final_shutdown=true`.

The approved managed0.8B/2B snapshots are now visible in the real L5 XDG cache.
Original0.8B with page_tokens4/max_append_tokens4 physically passes the
differential at frontiers3,4,5,9 with every full-vocabulary logit and all42 state
word exact versus Contiguous, followed by clean shutdown.

**15 Nov paged source lifecycle qualification follow-up:** signed `398e446`
adds a second ignored original-HF target on the same explicit paged policy.
Without changing runtime code it verifies:
- cancellation set before prepare returns `CancelledBeforeLaunch` and preserves
  the exact initial root/all42 snapshot;
- dropping a completed Ready candidate publishes nothing and preserves all42;
- toggling cancellation after Ready causes `commit` to return `Discarded`
  with the original root/all42 unchanged;
- normal commits advance to page_size-1, page_size and page_size+1;
- snapshots saved at page_size-1 and page_size can each be restored exactly:
  restore publishes a fresh root/epoch at the saved frontier and all42 state
  matches the snapshot;
- final source-model shutdown succeeds.

Signed `3041457` corrects the original lifecycle qualifier bug: the old test
wrongly required restored `RootBinding == snapshot.binding()`, contradicting
`SessionAuthority` restore semantics. It now requires same source/session,
root+1, epoch+1, saved frontier and exact all42 bytes.

Cold original0.8B lifecycle is accepted from retained dotlink job
`9feb1394-1a3b-498b-8f92-fdd3e607c6b3`: exit0 after168.08s, no contiguous
warmup, full marker:
`P12_SOURCE_PAGED_LIFECYCLE_PASS page_tokens=4 contiguous_warmup=false
prelaunch_cancel=true abandoned_ready=true cancel_after_ready=true
restore_before_boundary=true restore_at_boundary=true all42=exact
final_shutdown=true`.
`e6aac53` removes the temporary warmup branch and stage logging; the retained
qualifier is cold-only and labels successful evidence `cold_load=true`.

Intermittent alternative runs that previously looked like cold-loader
disappearances are connector-infrastructure signals, not model failures.
Course correction after validating the **deployed dotlink v0.17.3**
`release/0.17` tree: the deployed build does not have main's newer
`SharedProcessExit::Signal` type. Its shell runner records
`JobCompletion::completed(status.code(), status.success(), ...)`; on Unix,
`ExitStatus::code()==None` means the process dotlink waited on was signalled,
but v0.17.3 discards the signal number. For sandboxed shell commands that waited
process is Bubblewrap. Dotlink timeout/cancel/spawn/stdin/spool failures use
distinct non-`Completed` statuses, and the `Completed` path disarms the
process-tree guard before recording completion. Controlled probes distinguish the
case: SIGKILL of an inner child yields shell status137, and SIGKILL of the
sandbox shell yields dotlink numeric137 because Bubblewrap survives; the
intermittent model jobs instead terminate with null exit code, meaning Bubblewrap
itself was signalled. At those failures there is no Rust panic, OOM/Xid/coredump,
surviving process or VRAM residue; `dotlink.service` cgroup counters show
`oom=0`, `oom_kill=0`, `pids.max events=0`, with no observed dotlink
restart/poll failure. The external signal source is not proven. A clean cold run
launched directly as `native-gpu -> env -> test binary`, with no inner
tmux/setsid/heartbeat supervisor, still reproduced the same null-exit shape, so
the older harness is not required. Treat null-exit runs as infrastructure-
incomplete and never as negative model evidence; do not add a warmup workaround.

**18 Nov original2B acceptance follow-up:** the sandbox/native-GPU execution
issue is sufficiently characterized to obtain valid evidence without weakening
the model gate. Signed `6b920a0` now owns the harness fix: existing Bubblewrap
PID/user namespaces are detected and reused rather than nesting another user
namespace. Existing private Bubblewrap networking executes directly; a
shared-network Bubblewrap gets a fail-closed socket/io_uring seccomp filter;
outside Bubblewrap the original user+network `unshare` path remains. Signed
`4688cba` additionally recognizes a caller-supplied child netns only when current
interfaces are exactly `lo`, PID1's `/proc/1/net/dev` differs and inherited socket
FDs are absent. `--isolation-info` exposes `bubblewrap-child-network-namespace`.
Direct shared-net L5 still refuses `--require-private-network`; wrapping the command
with `bwrap --unshare-net --bind / / --dev-bind /dev /dev --` supplies a proven
private loopback namespace without weakening the harness. 31/31 harness tests plus
real loopback+CUDA child-net smoke pass. The upgraded
connector reports external signal numbers directly and v0.17.4 logs every stop
it initiates. Follow-up connector forensics found exactly one stop in the
relevant window, the explicitly requested `job_cancel`; a silent `sleep300`
survived until that cancellation. Therefore earlier SIGKILL9s must not be
attributed to quiet output or a dotlink watchdog. They are surrounding
harness/orchestration artifacts outside dotlink's stop paths. Unlimited cgroup
memory, zero OOM/high/max/pids events, zero PSI and clean GPU retirement remain
supporting evidence that they were not model/resource failures. The accepted
qualification runs below used an independent stderr heartbeat sidecar only as
harness plumbing; periodic output is **not** part of the model acceptance
contract.

On approved original2B snapshot `88606494...`, page_tokens4/
max_append_tokens4:
- `38c1f321-0347-4a80-b2dc-c6d9d46d0249` exits0 in703.17s with
  `P12_SOURCE_PAGED_B1_PASS page_tokens=4 max_append_tokens=4
  frontiers=3,4,5,9 full_vocab=exact all42=exact
  contiguous_reference=true final_shutdown=true`;
- `744f36fd-b145-47b9-aa61-084e7d4b414b` exits0 in438.18s with
  `P12_SOURCE_PAGED_LIFECYCLE_PASS page_tokens=4 cold_load=true
  prelaunch_cancel=true abandoned_ready=true cancel_after_ready=true
  restore_before_boundary=true restore_at_boundary=true all42=exact
  final_shutdown=true`.

Thus original0.8B and2B both clear the source-real paged B1 boundary and
lifecycle gates.

**19 Nov true in-flight CUDA cancellation follow-up:** original0.8B page4/append4
now closes the missing cancellation phase on the ordinary paged source path. A
qualification-only launch witness pauses after gather + ordinary body + scatter
have actually been enqueued under the original `ResidentSubmission`, before any
physical readback/Ready. If cancellation wins at that cut, the original event is
marked cancellation-requested and a status readback is the positive physical
completion witness. The event/queue work retires without `record_completed_write`
and without constructing `ResidentReady`; candidate Vector/staging/page leases
therefore never become publication authority. The physical pass reports
`P12_SOURCE_PAGED_INFLIGHT_CANCEL_PASS ... submitted_handles=396
cancelled_before_ready=true root_unchanged=true all42=exact vector_held=true
staging_held=true pages_held=true completion=known retired_exact=true replay=true
final_shutdown=true`, and RTX3060 returns to37MiB.

The paired unknown-completion proof drops that exact submitted future at the same
launch cut. Existing `ResidentSubmission::drop` remains the authority: no root is
published, the service fences, one event remains pending, all396 submitted handles
and source completion owners enter process-lifetime quarantine, and hybrid pool
backing is not recycled. Marker:
`P12_SOURCE_PAGED_UNKNOWN_COMPLETION_PASS ... root_binding_unchanged=true
service_fenced=true pending_event=true handles_quarantined=true
completion_owners_retained=true pools_not_recycled=true recovery=false`.
Two earlier executions were externally SIGKILLed before terminal model semantics
and remain incomplete; the direct known-completion run and the final
heartbeat-assisted unknown-completion run each exit0. Periodic output is optional
transport plumbing only, not an acceptance requirement.

**19 Nov explicit high-level selector follow-up:** the public models API now owns
`QwenCudaStatePolicy::{Contiguous,Paged}` and a separate
`ModelLoadRequest::QwenCudaWithStatePolicy`. The existing `QwenCuda` request still
maps directly to Contiguous, so no default/current production caller silently
changes representation. The explicit request converts to the existing low-level
`CudaSourceStatePolicy` only at registration; zero/out-of-context page/append
bounds still fail before source IO. Native rows4/8 now use the same policy and
Hybrid state owner. Cohort execution does not learn page semantics: each lane is
an `OrdinaryCandidateState`, so Hybrid performs gather -> unchanged native cohort
body -> scatter under the same submission, Ready exposes only actual Vector/
fresh-COW Writes, and canonical paged Domains appear only after completion.
Contiguous B8 retains the specialized private-KV transition when bindings are
`Distinct`; paged `KvInPlace` deliberately uses the existing per-lane append to
keep the design simple until profiling justifies another fast path. Focused policy
tests, strict CUDA clippy and CUDA-serving composition check pass. The compact
original0.8B Contiguous-vs-Paged B2/B4/B8 prefill/decode/all42 cell is physically
accepted: exit0 in306.84s with exact prefill/decode/all42 and clean shutdown; two
earlier externally SIGKILLed attempts remain incomplete only. The next public-SDK
cell reuses the public library facade rather than a fixture corpus:8 concurrent
4-token page-boundary requests,2 outputs, explicit Paged/native_rows8 versus
Contiguous, actual native cohort width8 on both and identical token IDs; exit0 in
340.78s with clean shutdown. Public HTTP then passes the same workload under the
explicit child-private-net wrapper: actual B8 `{8}` on Contiguous/Paged, exact token
IDs, clean shutdown, `P12_SOURCE_PAGED_HTTP_PASS`, exit0 in358.07s. Strict
`fabric_sdk` target Clippy passes. No fresh vLLM result is claimed because no local
vLLM runtime is available and retained C8 timing is not workload-matched. Do not grow a
parallel fixture/evidence system.

**19 Nov complete-domain fork substrate:** `ResidentSession::fork_shared_prefix`
separates cross-session prefix reuse from same-session rewind. Only a pristine
root0/epoch0/frontier0 destination is accepted; snapshot source/service generation,
capacity and every physical domain are validated before the destination reserves
`begin_validated_replacement(frontier)`. Publication keeps the destination session
identity and swaps cloned immutable domain owners only—no token replay, host
readback or new physical Ready. `SourceCallSession` additionally requires the exact
same resident model. For paged KV, shared pages naturally use the existing append
COW path. Original0.8B page4 physically passes prefix3 -> fork -> append1 against an
independent replay: full logits/all42 bit-exact, source snapshot immutable,
independent session/root, clean shutdown (`P12_SOURCE_PAGED_PREFIX_FORK_PASS`,
179.13s). This is the payload/root substrate.

**19 Nov bounded semantic cache follow-up:** `zerogravity-engine::state_cache`
replaces the old unsupported stub with a backend-neutral `PrefixCache<U,T>`.
`PrefixCompatibility` binds a local source plus nonzero grouped fingerprints for
execution (loader/quant/numerical), input (tokenizer/template/processor/adapter),
position, complete state schema and sharing authorization. Entries retain exact
semantic units and immutable `Arc` payloads. Lookup is a bounded newest-first
linear scan with no allocation; the greatest exact prefix wins, newest wins an
equal frontier. Exact duplicate insertion refreshes recency, FIFO eviction enforces
both entry and caller-supplied weight bounds. The cache deliberately cannot publish
or validate physical state; a hit must still pass the resident complete-domain fork
checks above. This simple O(bound) design is preferred until serving measurements
justify an index/trie. Two focused tests, strict engine Clippy and production CUDA
composition pass.

Serving cache activation is still open because public requests currently have no
explicit tenant/sharing identity. Do not encode `sharing=1` or infer authorization
from model/source equality. Next plumb an authenticated sharing scope through the
request/executor owner, then wire worker restore/cache hit/miss. Event/lease-backed
physical overlap remains later.
Ordinary physical model cells can use the seccomp fallback; loopback-dependent
serving/full-gate work must require a private network namespace.

- Extend existing source-compatible grouping with numerical/phase/layout/device
  compatibility. Dynamically refill and regroup unpadded survivors; preserve
  B1 and smaller qualified bands after checked prelaunch capacity refusal.
  Unknown completion or numerical/ownership failure is not a retry signal.
- Distinguish batching independent requests from processing multiple prompt
  tokens in one temporal kernel. Qualify initial, continued and cached routes,
  actual chunk sizes and chunk-to-decode transitions; B1 token-loop correctness
  does not prove bulk arithmetic or temporal-prefill performance.
- Separate bounded queued-request/prefill, resident-state and active-step
  admission. Budget weights, pages, recurrent checkpoints, COW candidates,
  scratch, block tables, driver/graph resources and transfer staging without
  double-counting shared ownership. Preserve per-model/tenant fairness.
- Add aged chunked prefill with protected decode progress and finite output/IO
  credits. Test staggered arrivals, unequal prompts/generation, cancel/EOS,
  slow clients and a large prompt among short decodes. Long prompts must make
  progress; a blocked row must not retain unrelated scratch or stall peers.
- First qualify fair interleaving in the existing completion path. Separately
  qualify host/transfer/GPU overlap and any multiple in-flight GPU regions
  using actual stream events, independent buffers and original leases. Keep
  per-session dependencies/publication ordered; never infer overlap from
  scheduler membership, multiple threads or reduced launch counts.
- Source-ready cached pages may join a batch; restoring/IO-blocked pages may
  not. Preserve encoded graph readers across refill, cache eviction and COW.

## Acceptance and implementation order

These gates extend existing pending work, not the active main-thread order of
small-model numerical parity, matched benchmarks, then original27B. Keep each
allocator/kernel/cache/scheduler change independently reviewable and accepted
through the same engine. Extend supported contexts/profiles only after their
own admission, source/state and physical tests.

- [ ] **Baseline/profile integration:** freeze the actual source/profile and
  existing contiguous/B1/batched results. Preserve the accepted llama.cpp
  regression lane; qualify new vLLM profiles against their own pinned source.
  Reject wrong profile/source/schema/page generations before device access.
- [ ] **Allocator/COW:** test full/partial/tail pages, randomized physical block
  mapping, reference counts, simultaneous forks/readers, repeated reuse and
  exhaustion. Charge peak live+candidate+metadata bytes and verify final
  release or intentional quarantine under actual completion.
- [ ] **Paged kernels:** compare independent contiguous reference, B1 and native
  batches at page_size-1/page_size/page_size+1 and multi-page boundaries;
  include asymmetric heads/positions. Poisoned inactive tails must be masked
  or rejected according to the profile contract. Compare full affected tensors,
  logits and all required state, followed by another token.
- [ ] **Hybrid prefix reuse:** compare cache-disabled replay to miss/partial/full
  hit and forked continuations. Include KV hit with missing recurrent checkpoint,
  mismatched numerical profile, source/template/adapter and unauthorized tenant;
  invalidate stale epochs and corrupt/incompatible restore inputs safely.
- [ ] **Real model composition:** both approved .8B/2B sources, full248320 logits
  and all42 domains, initial/continued-prefill to decode, repeated turns,
  B1/B2/B4/B8, row reorder, refill and canceled neighbor. Original27B/all112
  remains a separately admitted later gate, not implied by small-model success.
- [ ] **Physical failures:** cancel before submit, in flight and after Ready
  before commit; abandoned candidates, stale roots, pressure, delayed/unknown
  completion and graph-held pages. No partial root publication, premature free,
  stale page reuse or clearing quarantine without separately approved recovery.
- [ ] **Worker/HTTP:** real JSON/SSE, raw token IDs, text/usage/EOS/finish, aliases,
  staggered arrivals, slow/disconnected clients and refill. Check independent
  roots, fairness, bounded queue/output credits and actual owned retirement.
- [ ] **Phase acceptance:** relevant source/state/cancel/physical tests, strict
  checks, both approved original release parity routes and the complete isolated
  gate from AGENTS.md. Compilation, mocks and documentation cannot close these.

## Performance qualification

- After correctness, use the already established short-context profiles and
  C1/2/4/8 before expanding to admitted, independently qualified long contexts.
  Pin source/tokenizer/loader/quant/numerical profile, binary/runtime/driver,
  hardware/resource caps, arrival trace and cold/warm policy. Revalidate the
  pinned upstream implementation when selecting algorithms; do not copy dated
  documentation as proof of current behavior.
- Ablate B1 vs fused, contiguous vs paged, cache disabled vs partial/full hit,
  chunk sizes, and fair interleaving vs physically qualified overlap. Qualify
  both numerical profiles independently; comparisons with different source
  precision/quality are capability/tradeoff studies, not equivalent speed wins.
- Compare equivalent pinned vLLM optimized serving settings; label eager/default
  routes separately. Repeat paired runs and retain regressions/neutral results.
  Actual batching benefits do not imply overall vLLM parity or a universal win.
- Record offered/admitted/rejected/failed/canceled requests, committed useful
  tokens, TTFT and real token ITL p50/p95/p99, SLO-goodput, fairness/starvation,
  queue waits, actual kernels/launches/copies and overlap. Socket chunk intervals
  are not automatically per-token ITL; separate commit and delivery clocks.
- Measure live/reserved/cached/peak memory, metadata, tail waste/fragmentation,
  COW/restore traffic, page hits/misses and pressure. Separate logical credits
  from allocator/driver/OS memory and unreclaimed unknown work. A new supported
  context size requires real admission and output/state tests, not a flag edit.

## References and code seams

- [Joint scheduling, prefix-frontier and lifetime contract](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md),
  especially sections4/5/7; [planned SERVOPT gates](../../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md).
- [Numerical contracts and same-engine integration](../numerics-and-context/CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md).
- [Worker](../../../crates/zerogravity-cli/src/serve/gpu_engine.rs),
  [wave formation](../../../crates/zerogravity-cli/src/serve/gpu_batch.rs),
  [session/state allocation and publication](../../../crates/zerogravity-gpu/src/resident_inference.rs),
  [single-row lowering](../../../crates/zerogravity-gpu/src/resident_program.rs),
  [native batch lowering](../../../crates/zerogravity-gpu/src/resident_program/batch.rs).

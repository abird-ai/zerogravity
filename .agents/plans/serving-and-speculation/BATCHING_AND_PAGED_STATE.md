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

Allocator/COW remains **partially open at session/model composition**:
`ResidentSession::Domain` still owns contiguous `StateBuffers::Kv`; the new
paged representation is not canonical and no root/serving policy changed.
**Next: integrate `ResidentPagedKvDomain` into KV `Domain` while Vector
recurrent/conv state stays contiguous. For the first end-to-end route, gather
committed pages into bounded contiguous prior staging, execute the unchanged
ordinary CUDA KV append/attention path into contiguous candidate staging, then
materialize/scatter that candidate into private pages before the existing
Ready/atomic root swap. Qualify continued page_size-1/page_size/page_size+1 and
multi-page full-state parity before direct block-table attention or
complete-domain prefix identity/intersection.** Event/lease-backed physical
overlap remains later.

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

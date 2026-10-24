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

# Independent critical design review

Owner: `/root/composable_design_review`. Date: 2026-10-21.
Parent: MAIN `.agents/plans/pipeline-composition/`.
Reviewed: ISO `composable-pipelines-profiles-design-20261021/DESIGN.md`
(616 lines) and `REVIEW.md` (177 lines), read completely.

## Verdict

- Accept the proposed direction as a small extension of existing contracts.
- No architecture blocker or need for another engine, allocator, scheduler,
  serialized IR, model traversal or publisher found.
- Complete natural/source/runtime parity, then the short concurrent baseline;
  choose the first useful accepted region from measured cost. Do not require
  general interfaces or every extension drill before current parity.
- One factual wording correction and one identity clarification were sent to
  ROOT and adopted in canonical `REGIONS_AND_PROFILES.md` during this review.
- No further must-fix identified. Actual inference, media, performance and
  physical qualification remain outside this design review.

## Corrections adopted by ROOT

1. **Shared traversal is not the owner of profile-specific equations.**
   Original DESIGN section2 calls `run_layer_sequence` the owner of model math.
   MAIN `crates/zerogravity-model-qwen35/src/forward.rs:15` explicitly describes
   a family-neutral layer traversal whose closure retains profile-specific
   mathematics. Describe the shared traversal/schedule seam accurately; never
   infer that all profiles have one interchangeable numerical layer body.

2. **Separate immutable content/code recipes from source-owned prepared objects.**
   MAIN `crates/zerogravity-core/src/accelerator.rs:22` defines `SourceIdentity`
   as one process-local immutable allocation identity; reopening the same bytes
   creates a fresh identity. ISO `crates/zerogravity-gpu/src/resident_program/native_full.rs:366`
   additionally rejects a different `Arc` source owner when binding the shared
   library. Late per-call binding must never rebind an already source-owned
   library merely because artifact hashes match. A reusable artifact recipe can
   resolve anew; an existing live prepared object retains its original issuer,
   service, leases and compatible resource ownership.

## Useful implementation checks, not design prerequisites

- **Joined completion and output ownership.** If one selected region later uses
  several streams/producers, its Ready must be minted by the original service
  only after actual dependency joins and exact output-write certification.
  MAIN `crates/zerogravity-gpu/src/resident_leases.rs:589` checks unique original
  submitted writes; `:609` mints Ready only after physical completion;
  `crates/zerogravity-gpu/src/resident_inference.rs:1222` requires the same
  service, root/attempt and complete exact output cohort, currently one event.
  Unioning child Ready flags or relaxing that event-count check is insufficient.
  Add a delayed/failing secondary producer drill only when this feature exists.
- **Persist recipes, never live handles.** Prepared-cache work should retain the
  existing rule in MAIN
  `.agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md:9`: serialized metadata
  and source/code fingerprints may survive; pointers, driver handles, events,
  source issuers and reservations are rebuilt and revalidated. No new cache
  format or public trait is needed for the first region.
- **Joint resource accounting after selection.** Keep the proposal's physical
  backing/generation live-set rule. Neither sum(child peaks) nor max(child
  peaks) proves a joint peak with aliasing, old+candidate state and captured
  readers. Retain encoded/queued/in-flight ownership through actual completion.
- **Keep the first delivery small.** One existing source/profile/shape, one
  legal closed replacement with forced decomposition, one resolved recipe and
  real complete-request comparison suffice. Broader task/device drills belong
  to their feature increments, not a preliminary framework project.

## Substantive checks

- Primitive semantics versus physical kernels: correct; nested scopes can
  flatten/in-line/out-line without forcing physical dispatches. Current
  `PlanOperation`/`ForwardPlan` are flat semantics, not launch counts
  (`accelerator/program.rs:510`, `:626`).
- Pure DAG/config validation: correctly separated from execution and authority.
  Current `pipeline.rs:349` and `:418` expose only deterministic structural
  ordering and effect-free validation; no recursive runtime is present.
- Numeric legality: required casts, dual F32/BF16 residual values, reduction/FMA
  order, phase/chunk differences, masks/tails, observations and exact gates are
  preserved. Same greedy IDs cannot replace all-state/full-logit continuation.
- Inner regions prepare only; outer model sessions retain their original
  `SessionAuthority` and immutable roots (`session_root.rs:57`, `:76`).
  Compound media tasks use explicit staged frontiers; delivery/playback and
  earlier committed stages cannot be rolled back by a later failure.
- Whole-model replacement: appropriately requires source/numerics/effects,
  original admission/completion and a tested state bridge; an opaque private
  model state machine cannot qualify through a typed wrapper.
- Profiles select compatible accepted choices and cannot mint grants, Ready,
  source identity, reservations or publication authority. Context snapshots
  require original freshness checks (`context_snapshot.rs:530`).
- Graphs: actual dispatch count is distinguished from replay/submission count;
  capture failure cannot silently retry on a fenced service. Current private
  graph code requires source/workspace/stream compatibility and safe retirement
  (`resident_inference/graph.rs:80`, `:159`, `:232`).
- Dense27B: first correct bounded streamed/heterogeneous path is permitted;
  expert cache assumptions cannot remove required dense matrices. Model order
  and acquisition/device holds remain explicit.

## Research coverage and limits

- Read the current master comparison, its 21-family dispositions/priorities,
  all 416 mechanism labels across eight report inventories, and report
  summaries/caveats/priority sections. Selected detailed rows were cross-checked;
  this is not a new line-by-line audit of every external cited implementation.
- Counts independently sum to 416: TensorFold72, vLLM69, llama.cpp54,
  Halogen65, DwarfStar60, Maya42, DeepInfer38, NVIDIA recipe16.
- No substantive mechanism family requires another runtime. Approximate state,
  changed quant/arithmetic, route skipping, shifted/truncated context, biased
  sampler/replay and model-native sparse/shared-KV behavior stay separately
  selected or deferred; no external speedup becomes local acceptance.
- MAIN current source and plans outrank historical memory and ISO snapshots.
  Source anchors establish inspected contracts only; no runtime result was
  rerun. The active .8B natural publication/native2B work remains open.
- No build, GPU launch, model/SDK download, source/shared-plan edit, staging,
  commit, publication or root README mutation performed.

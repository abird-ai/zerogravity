# TODO — Single-process model and pipeline composition

Status: existing scaffolds accepted; recursive primitive/pipeline/profile design active;
execution deferred to owner plans
Provenance: 4bf624e (original scaffolds); ac44e83/f28a62c (context source/filter)
Design: [README.md](README.md)
Owners: model-acquisition, model-execution, serving-and-speculation, multimodal,
memory-and-streaming. All model/source/media execution remains subject to
their existing numeric/source/physical acceptance gates.

## Uncommitted

### Shared HIP / model composition checkpoint — 24 Oct

- [ ] Natural independent-reference and explicit Gemma row-hit/transfer scopes;
  no public serving, current source21 x2, general-fit or performance grant.

### Model optimization reconciliation — 24 Oct

Input: [Qwen27/Gemma4 audit](../../runs/model-execution-qwen-gemma-vllm-audit-20261024/REPORT.md),
20Qwen/20Gemma/15shared source comparisons. Neither family has full applicable
vLLM optimization coverage; source113 remains distinct from qualified execution.

- [ ] Map each applicable strategy to existing primitive→model operation→family→
  model/profile owners: shared matrix storage/layout, compact outputs, prepared
  recipes, temporal work, pages/COW/prefix, graph ownership and target verifier
  once; Qwen GDN/state and Gemma PLE/KV-sharing/liveness select eligibility.
  No second registry/IR/runtime or per-token feature search. Reuse map/callers,
  source/math/backend/phase limits and complete costs precede profile selection.
- [ ] Preserve distinct numerical strategies: source113 CPU-order BF16 versus
  BLAS/TensorCore/split-K, packed casts and explicit lossy codecs. A common
  operation does not share its source permit, arithmetic grant or StateRoot.
  Genuine N1→B4 cross-width diagnostic first differs at QKVZ2; GdnOut7 has a
  separate projection difference even with exact input. Original same-prior B4
  comparison now passes all15transitions/30plain+observed/fullstate/fullV/24sums/
  natural bits, with genuine HTTP N1→mixedB4/refill separately exact. Explicit
  compositional proof, not an independently generated original public trajectory
  or universal N1/B4 bit equality. Preserve distinct strategies/source permits;
  no production reference imports, weakened limits or cross-width grant.
  [Astra contract review](../../runs/architecture-astra-cross-width-contract-20261024/DESIGN.md).
- [ ] Use temporal/ragged pipeline liveness for Gemma E2B/E4B shared-tail fast
  prefill; retain requested prompt logits/taps and every future KV dependency.
  It is inapplicable to12B/26B/31B. Request B4 is not temporal prefill.
- [ ] Adopt source-bound phase selection for2B cachedB4 through the existing
  family/native adapter: one selector governs geometry/storage/inventory,
  native/GDN/RMS definitions, readiness, input admission and scheduler. Genuine
  N1prefill/partialfallback; unsupported2BinitialB4 refused before preparation.
  Derive width4 root18 from actual8current+4candidate+2Pending+2snapshot+
  2displaced, preserving sole retained capacity and physical readers. Exact
  selected-phase costs within existing4608MiBprovider+512backend/4GiBHOST;
  authentic2B native/RMS/catalog and same-prior numerical proof required.
  Guarded source13 adoption now includes the shared model foundation and2B
  specialization together. Authentic CPU inventory/reopen reports root18,
  fullhead1,017,118,720B, provider upper4,767,472,640B and HOST4,080,276,698B;
  remaining64,365,568B/214,690,598B respectively. These are complete source-derived
  bounds, not themselves GPU evidence. Optimized13 CLOSED0; original19physical
  scopes and genuine2B15transition trajectory/export/healthyretirement PASS.
  Original2B SAMEPRIOR30plain+observed/8112sections/fullstate/fullV/natural bits
  now exact; first newcached fault fails an unchanged compound retention check,
  last2faults unrun. Fixture-only14 logs full actual snapshots without relaxed
  checks; API/HTTP/timing/full assembled acceptance remain held.
  [Astra architecture verdict](../../runs/model-execution-astra-phase-profiles-review-20261024/REVIEW.md).

### Owner additions — 23 Oct, Astra high architecture review

Contract: [shared execution architecture](SHARED_EXECUTION_ARCHITECTURE.md).
User permits breaking pre0.1 API corrections now, while preserving the original
source/state/admission/flight/publication spine and previous design intent.
New model integrations must compose the established .8B/2B mechanisms: engine
primitives → shared model operations → family primitives → model specialization.
Lift reusable behavior before copying it; retain specialized implementations only
for concrete source/numerical/kernel requirements. Review actual reuse and callers,
not only matching API names; no parallel allocator/dispatcher/publication spine.

- [ ] Implement the [Astra next-native redesign](../../runs/architecture-astra-next-native-review-20261023/DESIGN.md)
  through Sol HIGH: source-neutral physical owner plus source-bound services,
  same envelope/queue/ledger/fence, exact model sub-admissions and independent
  source/close/reload lifetimes. No per-model bootstrap/cap increase or source
  equality weakening. Source work isolated while frozen single-source Gemma
  actual qualification and binder316 checks continue.
  Physical13 now integrated MAIN/ISO345 with guarded original parents/modes/
  README; logical564/provider+default strict PASS, Astra architecture accepts/no
  concrete must-fix. Original local required release/full both-model CUDA/state/
  fault/API checks PASS; x2 native3/portable6/rawtransport12 and corrected shared4
  HIP/Vulkan healthy/unknown PASS. Fixture Oracle shadow aliases corrected only;
  production unchanged in345v3. Interrupted original controller and missing
  local13 closure preserved separately from successful fresh replay. Final gate
  pending after coherent contributions; measured release remains316.
- [ ] Extend the original HIP actor's bounded prepared-set lifecycle for multiple
  independently source-bound models on one physical owner. Current actor
  deliberately refuses a second live set/legacy/graph; shared admission alone
  does not grant simultaneous Gemma/Qwen native execution. Preserve exact source/
  arithmetic/specialization identities, original envelope/global fence and actual
  per-set module/flight retirement. No learned payload sharing by code identity.
  [Astra selected seam](../../runs/architecture-astra-shared-native-20261023/DESIGN.md):
  nonzero never-reused logical StreamId through all compile/cache/IO/driver joins,
  immutable source-service Client including internal enqueue paths, maxphysical
  streams1 retained. Cold32/64/32GiB provider/device/host remains a candidate;
  derive quotas from complete costs before bootstrap. The earlier two12GiB
  draft must not force Gemma's resident projections back to streamed storage.
  Current candidate Qwen12GiB/Gemma16GiB plus one4GiBcontext fits the32GiB
  provider envelope arithmetically; complete host/init/scoped-key peaks remain
  to qualify. Qwen9,917,743,073B is model-only; Gemma13,174,234,632B plus the
  shared context gives27,386,945,001B before new metadata. No duplicate context
  subtraction, live cap growth or optimistic compiler-host release. Sol HIGH
  implements; actual multi-set remains open.
  [Concrete ABI review](../../runs/architecture-astra-residency-and-set-key-20261023/DESIGN.md):
  private scoped key nests the FULL original KernelId; bounded full-Eq routing
  reaches actual driver lookup despite Command dropping logical StreamId.
  Fixed SingleCompat/Scoped mode preserves original callers without fallback
  mixing. Per-set source/module/admission and global unknown fence remain.
  [Astra HIGH draft-source/API review](../../runs/pipeline-composition-astra-hip-multiset-review-20261024/REPORT.md)
  approves the bounded direction: shared-only Qwen reload after actual final
  reader/native/service retirement; exact pinned-client preflight with a cold
  Arc/Weak identity, without cache backedges. Sol implements both corrections.
  Cold full-Eq routes and actual family consumers were inspected; completed
  fixes, full costs, checks and two-source physical lifecycle remain unqualified.
  ROOT reviewed the complete scoped-source delta and compiler successor: Shared
  retains up to three admitted8GiB compiler leases on the original persistent
  context (two initial sets plus one reload), including cancelled attempts; model
  unload gives no compiler-cache credit. Reserve before queue submission so a
  fourth ordinary refusal cannot fence a healthy sibling. SingleCompat keeps its
  existing policy. Conditional complete HOST upper33,668,886,252B includes two
  scoped-metadata maxima and128MiB fixture allowance, leaving690,852,116B under
  fixed32GiB; actual costs/fit and two-source lifecycle remain open. Grouped
  GPU-free successor checks run separately; no physical or performance grant.
  [Compiler lifetime correction](../../runs/pipeline-composition-hip-multiset-sol-high-20261024/compiler-successor-v2/REVIEW.md).
  Frozenv4 whole7stage CPU unit now PASS: all3strict,171resident/12core/7compiler
  tests and2actual-type source-expression controls. Actual compiled scoped metadata
  is observed per set and joined exactly, not inferred as a residual. Corrected
  fullHOST upper34,205,757,164B includes separate512MiB supervisor, leaving
  153,981,204B under32GiB; startup checks actual coldplan+fixture+controller.
  ROOT complete source/control/owner review; isolated x2 optimizedbuild active,
  genuine same-actor healthy reload/globalunknown cases still pending.
- [ ] Record Qwen27B's one family traversal through the existing backend/plan
  with explicit MKL arithmetic revision,112logical domains/128ordinary planes/
  fullConv4, before native/multi-model serving. Current scalar-exp/H16/short-HF
  binders refuse; accepted CPU INITIAL1→cached1 proof cannot grant their reuse.

- [ ] Qualify the integrated `TextCallGeometry` extraction and actual Gemma
  consumer. It separates source-bound call metadata from model geometry and
  finite numerical/backend eligibility; six copied-core tests pass only.
- [ ] Move remaining closed family builders/geometry out of core after generic
  semantic consumers migrate. Separate family, numerical and implementation
  identity; retain explicit rounding/reduction contracts and actual shape gates.
- [ ] Use one prepared-model serving adapter for validation/session/accepted
  step/committed output/cancel, retaining the original router/workers/roots.
- [ ] Implement sealed NativeCohort in the SAME node loop: authentic perrow
  record/runtime/prior/candidate/borrowedphase joins through validation, View/
  storage/finite certification and terminal grouped proposals, original physical
  cohort flight/Ready and single all-or-none publisher. Physical188/376 is an
  explicit request axis, not temporal tokens. Call-local body and enforcedmax8
  sessions preserve22root bound. Actual canonical47B4 source discovery passes;
  complete native arguments and genuine Rust188prefill→same-rootB4 remain open.
- [ ] Share immutable recipe/topology/SSA/storage/native candidates inside the
  existing record API; bind small source-issued invocation/root/phase data late.
  Current percall record charge4MiB must not multiply into permanent Bplan
  rebuild/hash overhead. No new template IR or firstrow provenance shortcut.
  Sol18 now integrated MAIN/ISO314selected source paths; Astra HIGH completed
  architecture-only review and accepted with no must-fix. Source/profile/shape
  cache40 retains original host charges; topology/source semantics Arc-shared,
  actual operation/provenance remains invocation-local; warm bind outsideMutex.
  Per-call record bound512KiB, conservative fullcache160MiB. Core201/Qwen87/
  GPU549/default+provider strict PASS. Original source/site/native validation and
  wrapper allocations remain. Both models' complete actual CUDA/state/fault/API
  plus streamed2B faults PASS; .8B quiet C1+70.73%/C4+28.81%/C8−2.58%, singlepass.
  Full assembled gate15 LAST PASS1752Rust/0failed/89ignored/269suites/allchecks;
  314sourcepaths/rootREADME unchanged. No zero allocation/general pipeline claim.
  [Architecture review](../../runs/architecture-astra-prepared-review-20261023/DESIGN.md).
  Residual Sol8 binder now integrated MAIN/ISO316: concrete existing ColdModel
  cache40, genuine invocation validation outside lock, original retained
  witness/site charges512KiB each +64KiB owner. GPU551/provider+default strict
  pass; original joint release and both models' full real source/state/fault/API
  plus streamed2B faults PASS. Quiet .8B C1+78.62%/C4+26.67%/C8−1.09%, singlepass;
  six original owner closures healthy. Final isolated gate16 PASS1752Rust/0failed/
  89ignored/269suites/allchecks;316sourcepaths/rootREADME unchanged. Astra
  next-native pass is architecture/planning only; Sol HIGH applies the redesign.
- [ ] Before Gemma HIP merge: retain authentic source/snapshot lifecycle through
  native unknown/retirement; unsafe lowlevel boundary with one validated safe
  facade caller, no safe Any-owner proof. Keep fixed hardware/context/compiler/
  module policy in shared backend preparation. Sol44v5 implements these Astra
  corrections and complete source/state/lifetime qualifier in donor; reachable
  strict/source tests pass. Original frozen MAIN314+44 on x2 now passes full
  seven-case/all35/state/fullV/natural/lifetime matrix and fresh actual unknown
  retention. Guarded Gemma44 integrated MAIN/ISO343; successor shared physical
  owner/serving/performance stay unqualified. Prior316 final gate16 passed;
  next assembled gate follows coherent successor/affected physical checks.
- [ ] Demonstrate one forced ordinary versus optimized region/profile slice
  with joint liveness/observers and device-resident typed stage edges. Build on
  existing region metadata and physical lowering, without a universal framework.
- [ ] Later multidevice/MoE reuse: source/range/codec identity separate from
  physical slot/generation/placement; pin queued/captured/inflight readers,
  charge migration old+new/staging, preserve fairness and complete root shards.
  Backend tuples qualify CUDA/HIP/Vulkan separately. No current sparse,
  distributed or media execution claim follows from this review.

### Owner additions — 21 Oct, resident stages and fusion

Contract: [resident values and cross-stage lowering](REGIONS_AND_PROFILES.md#resident-values-and-cross-stage-lowering--owner21-oct).
Design only; all implementation and acceptance below remain pending.

23Oct shared-engine allocation/boundary design: [pool, layout and transfer APIs](../cpu-gpu-optimization/SHARED_MEMORY_AND_BOUNDARIES.md).
Profiles select finite prepared storage/transfer policies through the existing
engine; model or nested-pipeline boundaries do not create allocator ownership,
copies or waits. Shared adoption remains in optimization/memory plans.

- [ ] Carry eligible stage outputs through typed leased device bindings with
  source/profile/root, owner/device/context/generation/range, layout/extents,
  access and actual producer dependencies. Stage boundaries alone introduce no
  host return, serialization, allocation, payload copy or synchronization.
- [ ] Prepare cross-stage resident handoff, device chains/graphs, exact fused
  regions and explicit bridges through existing ports/providers; force the
  accepted decomposition for comparison. Resolve finite compatible candidates
  outside the hot loop; no new interpreter, tensor runtime or execution IR.
- [ ] Plan joint placement/layout/liveness and admitted scratch/graph/staging
  across eligible producers and all consumers. Qualify caller-owned native
  component buffers and stream hooks; expose hidden pools/copies/waits and
  retain cases where a copy or smaller region improves complete-task cost.
- [ ] Keep qualified intermediates, model state and compact generation feedback
  on device. Use real event dependencies and bounded control/status transfers;
  preserve sampler/grammar, context grants, independent model publication,
  cancellation/fairness and output-credit boundaries. Require typed trained/
  codec bridges between models; graph composition does not erase host decisions.
- [ ] Explain and attribute each required materialization/copy/readback/wait and
  graph split. Compare forced ordinary, resident-chain, graph and fused paths
  under the same numerical/approximation contract, source and output consumers.
- [ ] Prove zero intermediate H2D/D2H payload traffic and no stage-only host wait
  on a qualified compatible fixture; test incompatible bridges/refusals,
  tails/ragged/multiple-consumer/debug/alias, cancel/backpressure/reload and
  unknown completion. Separately qualify trained complete-task results and
  actual physical peak/tails; start NVIDIA, then Strix Halo, without assuming UMA
  implies zero-copy. Appropriate real-source/physical and full check.sh gates.

### Owner additions — 21 Oct, approximation/components/shared policy

Contract: [profile policy and native component reuse](PROFILE_POLICY_AND_COMPONENT_REUSE.md).
These are pending implementation/acceptance tasks, not new runtime support.

- [ ] Support explicitly approximate profiles beyond derived weight precision:
  declared methods/ranges, changed semantics, held-out quality limits and
  permitted adaptive choices; separate variant correctness from original-model
  fidelity. Bind effective variant/state formats into cache/snapshot/graph keys.
- [ ] Permit interchangeable allocator/pool, provider, codec, scheduling,
  residency and transfer implementations under the original ownership spine.
  Define replacement boundaries and old-allocation retirement; test old/new
  coexistence, failure, graph readers, cancellation and unknown completion.
  Replace/rebuild unsuitable modules when real implementations justify it.
- [ ] Resolve one shared versioned configuration visible to participating
  modules, with explicit local overrides, bounded operational observations and
  explainable choices. Existing coordination admits joint batch/graph/draft/
  prefetch cost; test conflicts, stale observations and reload boundaries.
- [ ] Apply audited MIT/Apache static-native reuse first to every implementation
  and harness; otherwise directly port the smallest useful implementation and
  fixtures, establish faithful behavior, then integrate through our contracts.
  Preserve upstream notices; no Python/PyTorch production runtime is implied.
- [ ] Consolidate profile/phase/backend equivalence acceptance: solo/concurrent,
  B1/batched reorder/refill, draft off/on/every keep, fresh/restored/COW,
  ordinary/graph/fused and contiguous/paged. State required bitwise/error/
  distribution/changed-semantic relation; reject incompatible combinations.
- [ ] Exercise the first useful region/profile with one reused or ported
  component and the native correctness harness; preserve the baseline, complete
  state/physical gates and measured complete-request results. New abstractions
  must not delay current parity or require a second execution stack.

### Recursive composition and profiles — owner request21 Oct, design first

Independent [mechanism research](../../research/engine-optimization-comparison-2026-10-21/README.md)
is complete; parallel design runs without blocking active source parity/binding.
Existing YAML/DAG validation does not establish nested executable pipelines.
Proposed contract: [recursive regions and profiles](REGIONS_AND_PROFILES.md).

- [ ] Define primitive semantics separately from their kernel implementations;
  typed ports/shapes/source/numerical policy and declared state/read/write effects.
- [ ] Define a pipeline node that may reference a bounded subpipeline: one
  inference's operator/kernel regions and compound multimodal task stages use
  consistent composition laws while retaining their distinct value/frontier types.
- [ ] Define immutable profiles combining named pipelines and permitted source/
  arithmetic/workload/placement strategies. Separate defaults/preferences from
  non-overridable source, authorization, physical and completion requirements.
  A saved profile is not a live SourceIdentity, Ready or allocation permit.
- [ ] Compile hierarchical views into existing semantic/physical plans; legal
  flatten/in-line/out-line, exact region replacement, prepared shape selection,
  graph replay and optional fused kernels. Preserve two enduring representations,
  no new interpreter/runtime/model loop/allocator or per-token plan search.
- [ ] Specify child-region effects/exports, alias/liveness after fusion, late
  source/root/epoch/generation binding, cancellation, backpressure and completion.
  Inner mathematical regions prepare; original session writer publishes once.
  Compound task boundaries must not imply global rollback of already committed
  model/media stages or allow recursive independent publishers.
- [ ] Profile examples: resident interactive, concurrent throughput, long/shared
  prefix, memory-bounded dense27B; future sparse/derived-precision/media/distributed.
  Mark prerequisites, conflicts, unsupported/fallback and actual qualification.
- [ ] Compare minimal efficient kernel paths with ordinary decomposition and
  captured schedules; retain cases where fusion loses occupancy or overlap.
  Distinguish semantic nodes, submissions and actual device kernel invocations.
- [ ] After parity and matched baseline, measure source-only preparation
  (e.g. exact gamma widening/shift) and legal normalization-region fusion as
  early candidates; preserve casts, original model ownership and admitted bytes.
  Compare complete-request cost and actual dispatches before promotion.
- [ ] Review extension drills for a new primitive/model/provider/task without
  unrelated scheduler/API changes; define exact-source/all-state/physical and
  complete-request performance gates before implementing these abstractions.


## Committed

### `eb49fa0e945b` — source programs and shared native composition

Related source commits: `3e89f1919993` (gpu-resident-native-scc), `752642361cb4` (facade-shared-hip-lifecycle).
Aggregate source21 qualification applies to the assembled tree; no per-commit
build or general model/backend grant. Historical failures keep their source scope.

#### Shared HIP / model composition checkpoint — 24 Oct

- [x] Source20 local full gate/build, original CUDA23 physical cases and fresh
  full2B composition pass; these do not qualify current x2 shared execution.

- [x] ROOT reviewed current1650-source x2 payload and transport corrections:
  distinct packaged/hydrated closures, final local closure check and relocated
  bootstrap receipt import;15 grouped CPU controls pass. Original namespace1
  failed closed after hydration/import, before build/model work; preserved.

- [x] Fresh namespace2 source20 bootstrap/offline build and authenticated binary
  binding pass. Known shared reload and Qwen final-reader unknown close PASS;
  source20 scope, original limits and owner remain explicit.

- [x] Gemma feedback unknown closes PASS. ROOT independently reruns original
  typed validation for build and all3cases; exact source20 artifact51808f9e…,
  separate packaged/hydrated pins,70helddataFDs and both owner layers close.
  Scoped math/state/lifetime only; public serving, natural27B trajectory, current
  source21 x2 artifact, general fit and speed remain unqualified.

- [x] Original v7 known reload run CLOSED_FAILED1 with both groups reaped/gone,
  full EOF, all70 held source files and binary/control unchanged. Pi bounded
  read-only collection found aggregate fixture mismatch: original four-Qwen-call
  after() reused by a declared three-call shared scenario. All three Qwen
  per-call original comparisons were exact; original failure remains failed.

- [x] Observed Gemma scope is one full diagnostic plus three ordinary exact
  packet/fullKV calls. Retirement/reload/fourth-preparation refusal and final
  native/ledger events were observed, not a fresh accepted joint pass.

- [x] Pi corrects only shared-fixture original-oracle scope with exact declared
  proposal/record membership and negative controls; original full4 after()
  remains unchanged. Both original unknown cases remain unstarted.

- [x] Astra integration design: one HIP loader parameterized by original owner
  and readout, preserving existing shared teacher117; contextual Qwen natural13
  and Gemma reuse7 merge, no duplicate loader/allocator/source authority. See
  `.agents/runs/pipeline-composition-astra-three-unit-integration-20261024/TODO.md`.

- [x] ROOT independently reviewed contextual38-path composition, six Pi review
  dispositions and Astra design review; merged into main source19 preserving
  source18 CLI fixes and README. Combined strict and artifacts28/GPUCPU244/
  compiler7/core12/Qwen-oracle5/Gemma-oracle1 pass; final example cfg checks and
  complete main gate run separately. Source integration, not runtime acceptance.

- [x] Corrected source20 actual shared lifecycle and both unknown scopes pass
  with ROOT typed/source/state/owner review; fixed HOST/device limits and
  original failed scopes remain unchanged.

#### Model optimization reconciliation — 24 Oct

- [x] Before eager preparation, close actual context/compiler
  host/device admission and partial-failure retention on original owners.
  Compiler-lifetime7 provides raw-program cleanup/attestation; current12 adds
  actual512MiB DEVICE/384MiB HOST detached context owners on original domains,
  immutable environment, cumulative bounds, pooled RMS source definitions and
  same-actor compile-only exact recipes. Actual19/source/fault/retirement/API/
  genuineHTTP pass; livecompiler deltas0 in C1/C4/C8. Currentfullgate18 PASS.
  Finite SM86/profile scope, no arbitrary compiler allocation theorem, graph
  replay or other-backend grant. [Astra design](../../runs/architecture-astra-compiler-readiness-20261024/DESIGN.md).

#### Owner additions — 23 Oct, Astra high architecture review

- [x] Finish the assembled pool/ring/workspace/head implementation and physical
  checks first; apply the shared layers through existing objects, without a
  second dispatcher/IR/completion authority. Frozen storage/rings release2:
  both actual40natural/all5fault/API matrices exact; fresh short benchmarks and
  full isolated gate12 PASS1734Rust/0failed/allscriptchecks,287sourcepaths and
  rootREADME unchanged. C8 superiority and broader parity remain open.

- [x] Extract a source-sealed matrix/head operation from the Qwen Head254
  adapter: common source windows/residency/retirement and current lease binding;
  family owns tensor roles/tie, traversal/value IDs and exact arithmetic. Sol8
  integrated MAIN/ISO290sourcepaths, including exact issuer/member/expert-range
  correction. Strict/GPU544/real-source negative/default+CUDA+HIP+Vulkan builds
  pass; joint release and both models' complete actual CUDA/state/fault/API
  checks PASS. Full isolated gate13 PASS1737Rust/0failed/allscriptchecks.
  Affected HIP/Vulkan transport12 pass on the native/context successor; no
  portable head arithmetic or arbitrary family/backend qualification follows.

#### Recursive composition and profiles — owner request21 Oct, design first

- [x] Independent design plus coordinator source review: three conceptual layers,
  two enduring execution representations, exact effect/numeric/lifetime rules,
  staged compound-task commits, finite profile matrix and extension drills.
  Independent second source/coverage review passes after shared-traversal and
  content-recipe/live-source-owner corrections; actual producer completion join
  clarified. [Review](../../research/engine-optimization-comparison-2026-10-21/composition-review.md).
  No source/runtime/performance acceptance.

- [x] Canonical YAML schema `zerogravity.pipeline.v1` replaces TOML
  for the single-process/multi-model CLI config. Repeatable model/step/link/
  route/context flags lower into the same typed validator. Strict YAML
  256KiB source/expanded scalar, 16,384 nodes/32 depth, single document,
  deny-unknown-fields/custom-tag/recursive-alias protections; bounded
  anchors/aliases/shallow merges allowed. Ten sample YAML configurations
  cover text, audio, ASR/speaker diarization, chat+context filtering,
  NanoJev-like decisions, latent/complete image stages and video frames.
  `pipeline check` only validates graph structure; `pipeline run` still
  refuses. A maintained YAML parser replacement is a separate dependency
  review, since the already-cached `serde_yaml 0.9.34` is deprecated.

- [x] Add finite native `ComponentRegistry` with independently registered
  immutable typed port contracts, pure `Config::validate_with_registry`,
  duplicate/invalid/forged-context port refusal and source-neutral future
  stage kinds for audio feature/codebook conversion, image solve/decode,
  video frames and calibrated decisions. Every stage and source remains
  NOT IMPLEMENTED for runtime use until separately qualified.
  No per-model binary/process, runtime scheduler, GPU, transport, extra
  licensed source or user code loaded.

- [x] Frozen source-neutral architecture/YAML acceptance passed:
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  exit0, 1,073 Rust executions/0 failures/69 ignored/230 suites
  plus 18 namespace Python fixtures, strict production/diagnostic
  Clippy, CPU comparator self-tests, release CLI, installer, shell
  and Nix formatting. Both original approved Qwen3.5 checkpoints
  retain full 248320-vocabulary logits, 42 StateRoot domains,
  nonzero tokens, independent sessions and source/reclamation leases:
  cache/direct18.43s and HF-import23.39s independent release routes
  pass exit0. Nix offline source-integrity includes the ten
  YAML examples, typed parser and extension registry; root README,
  workspace Cargo.toml and owned model math/source paths unchanged.
  Acceptance is **pure configuration/registry scaffolding only**;
  trained multimodel execution, audio/video/image streams, NanoJev
  inference, source-real decoded shards and production APIs remain
  in the deferred owner plans. Sign exactly reviewed source/examples/docs;
  no push, no device launch, no model download.

- [x] Prior accepted source and family contracts remain in
  model-acquisition, model-execution, multimodal, serving, memory and
  earlier signed history; none of those prior unqualified model/runtime
  implementations are retroactively promoted.

### 4bf624e — accepted foundation

- [x] Source-neutral ordered model target matrix for Qwen3.5+/3.5/3.8,
  Gemma4, DeepSeek V4.1 Flash, GLM5.3 Flash/full, Muse Spark, Mistral/Llama,
  additional OSS, QwenImage2.1/QwenTTS/QwenASR/Whisper/Canary/Parakeet.
  Source name/layout/release status distinguished from executable support.
  Exact local 0.8B/2B Qwen GGUF baseline not promoted to arbitrary HF repo.
  DeepSeek V4.1 Flash/GLM5.3 Flash payload hold represented explicitly.

- [x] Pure typed ComponentContract, ModelInstance, Pipeline, Edge and ApiRoute
  types in existing neutral core. Compile bounded DAG with matching semantic
  ports, required producers, topology, source-to-output productivity,
  unambiguous exposed routes and shared model IDs. Output is NOT a server,
  loaded model or qualified media graph.

- [x] Default CLI `models [list|inspect]` static truthful registry and
  fail-closed `models run|stream`. initial TOML schema (now YAML) `zerogravity.pipeline.v1`
  and repeated `--model`/`--step`/`--link`/`--route` flags lower into
  exactly the same graph validator; config file size ≤256KiB. Both single-
  model and multi-model diagrams/example configs exist. `pipeline check`
  is pure and `pipeline run` is explicitly NOT IMPLEMENTED.

- [x] Bounded pinned HF/GitHub `--bundle --include` opaque Auxiliary-only
  data selection for Diffusers/multi-stage speech assets, reusing one Hub
  pin/tree, existing filters/admission/cache publisher and exact offline
  bundle-only source identity. No invented model/codec completeness, native
  Xet bundle, unknown-size cap, arbitrary subdir recursion or executable
  checkpoint interpretation.

- [x] Implement source-neutral explicit optional chat-context policy:
  per-pipeline enabled toggle (default absent/off), shared text LLM model
  instance, originally max1..16384 UTF-8 bytes and prior committed
  same-session visible turns (historical 4bf624e semantics, now superseded
  by the configurable source selection/filtering policy above).
  Typed optional context.value inputs reach ASR/diarization/
  transcript-text/speakers-text/translation. The graph still prevents
  same-turn cyclic feedback and unowned raw KV injection. Disabled policy leaves edges but delivers None.
  Source-real contextual speech/diarization/translation, privacy-safe
  snapshot capture and context A/B quality remain NOT IMPLEMENTED.
  initial TOML (now YAML) and repeated flags compile identical shape; include full
  contextual-voice-assistant example and adversarial negative tests.

- [x] Frozen source-only acceptance: exact MAIN
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  passed 1,032 Rust executions, zero failures, 69 intentional ignores
  across 230 suite runs, plus 18 namespace Python tests, release CLI,
  Nix/shell formatting, installer, CPU numeric self-tests and strict
  default/production/diagnostic Clippy. Actual Nix filtered source-integrity
  includes the new core/CLI modules and five initial TOML examples (migrated to YAML).
  Original Qwen3.5 release cache_prepare source/model parity independently
  passes for both models, all 248320 logits, all42 StateRoot domains,
  2 tokens, session/source leases, mutation/reclaim: direct 21.05s and
  HF-import 24.10s. Both successful after an initial aborted connector
  tool invocation and explicitly corrected two test-local shadowing errors
  plus a bad nested help quote; no numeric/source/runtime tolerances relaxed.
  Acceptance uses an exact-path signed local source commit. No device
  launch, data download, live server, license approval or real inference extension.

### ac44e83 — accepted foundation

- [x] Replace fixed prior-visible-only history with a reusable typed
  ContextFilterPlan for independently configurable user/assistant messages,
  tool calls, tool results, application-owned system/developer prompts and
  explicitly named shared/granted records. Default remains session
  user+assistant; filters include/exclude by record kind, exact tool name,
  source origin or text substring in deterministic order. Typed calibrated
  NanoJev-style decision stage references a separately declared Decision
  model, accepts threshold in basis points and configurable unavailable
  policy (omit/reject/keep_authorized). The latter never reintroduces
  excluded, missing or ungranted records. Supported initial TOML (now YAML) and repeatable CLI
  flags; bound source selectors/controls and configurable 4KiB default
  through 1MiB context budget. Configuration *requests* source rights;
  runtime source owner must verify actual authorization, not a hard-coded
  block against application prompts or authorized sharing. No model
  inference/automatic context history access is claimed.

- [x] Audited all 6 old detached side worktrees against signed C3/C4/C5,
  v1 and pipeline acceptance commits; preserved earlier unfinished portable
  source patches under .agents/docs and copied nine side-only run artifacts
  into existing MAIN .agents/runs before removing all six detached
  worktrees. One new active, clean worktree
  zerogravity-worktrees/agents-conventions at unmerged e0410c7 is
  intentionally retained. No branch or unrelated user work deleted.

### f28a62c — accepted foundation

- [x] Source-grant checked context owner/snapshot service: the Rust core
  now holds committed immutable records and exact revocable session,
  app-prompt and shared/cross-tenant source grants behind one trusted
  `ContextOwner`; separately minted read-only `ContextReader` capabilities
  cannot mutate grants or rebind to another principal. Per-session prior
  turn check, bounded latest-whole window, provenance IDs/source labels,
  source epoch retry/freshness validation, and filter-only exposure to
  verified candidates are implemented. Decision adapter receives record
  text only after source grant + previous filters; it returns a typed
  include/omit verdict or explicit unavailable/error. Snapshot records
  original-source attributes and decision audit (not raw text in audit).
  This is an in-process source owner, not an external chat connector
  or pretrained model executor. External source grants must be issued by
  the trusted application before minting a read-only reader. Separate
  tests prove that a policy selector alone cannot grant access.

## Tests and verification

- [x] Source-authoritative snapshot tests pass: exact tenant/application/
  session isolation, current-turn exclusion, cross-tenant shared grants,
  same-application prompts, independent tool-result filtering,
  authorized-only decision inputs, optimistic revoke during callbacks
  without deadlock, stale/foreign reader refusal, erasure/record provenance,
  bounded latest-whole 4MiB candidate and 1MiB output windows,
  full-or-empty source behavior, decision error propagation and
  secret-free decision audit. Full original scripts/check.sh frozen
  isolated gate passed 1,058 Rust executions/0 failures/69 ignored/230
  suite runs plus 18 Python namespace fixtures, strict lint, release
  build/installer, Nix-format and CPU comparator selftests.
  Independent release cache_prepare exact approved Qwen3.5 0.8B+2B
  parity passes full vocab/42 StateRoot domains/2 token sessions/leases:
  cache-direct17.36s and HF-import21.91s. Actual Nix offline
  source-integrity passed with new Rust module, root README and
  dependencies/lock unchanged; no live model/GPU/provider access.

- [x] Frozen isolated updated-context-filter source-only check.sh
  passed 1,047 Rust executions, 0 failed, 69 intentionally ignored
  across 230 suites, plus 18 Python namespace fixtures, four strict
  production/diagnostic feature lint gates, Nix formatting, release
  installer and CPU comparator gates. Two independently invoked original
  Qwen3.5 release cache_prepare routes pass both approved models,
  full248320 logits/all42 state domains/two tokens/source-lease/
  mutation/reclamation bit-exact: direct18.66s, HF-import23.24s.
  Real offline Nix filtered-source integrity includes two new paths;
  root README, workspace deps/Cargo.lock unchanged.
  This qualifies only config shape plus pure filtering over explicitly
  *provided* already-granted record views. Live source authorization,
  history retrieval, calibrated NanoJev model inference and real media
  execution are not implemented.

- [x] Filter pipeline: independent tool-call and tool-result opt-in/
  exclusion, context source selection for app prompts/shared grants,
  no silent permission acquisition, ordered predicates, user preference
  filters and decision confidence/fallback. Include/omit verdicts must
  not resurrect previously filtered entries or invent inaccessible text.
- [x] initial TOML (now YAML) + equivalent CLI filters and NanoJev-style typed decision
  references. No inferred upstream checkpoint or executable model. Reject
  duplicate/unknown scopes, malformed origin names, unsourced decision
  models, non-decision heads, invalid thresholds, invalid return variants,
  excessive source/filter/budget and unsupported executable pipelines.
- [x] Opt-in previous-turn context *shape* passes initial TOML (now YAML) and repeated flags,
  with toggling ON/OFF on the same graph, absent-by-default channel, and
  static bounds/unknown model/non-language source/no consumer/spoofed
  current-turn feedback refusal. Configuration checking has no
  data/transport access; a later runtime may include explicitly
  requested authorized app prompts or shared user records only under
  real source grants. Runtime content/biometric/clock tests remain open.
- [x] Core shape validation passes: one-stage Qwen, model sharing across
  two routes, ASR+diarization+optional contextual translation+LLM+TTS
  and image flow. Stages are semantic placeholders only.
- [x] Source-free negative tests pass: cycles, orphan stages, wrong
  typed ports, missing speaker source, duplicate model ID/API/path/input,
  unknown schema fields, unsupported model/media routes, invalid quant/role hold,
  context spoof/size/provider/current-turn linkage and disabled output.
- [x] Default CPU & acquisition CLI tests: offline `models list/inspect`,
  `pipeline check --config` and equivalent flags/ON-OFF context toggle;
  `run/stream` refuse without source/physical/API effects.
- [x] Acquisition and CLI gates: original HF/GH closed weights selection
  unchanged; `--bundle` separate Auxiliary role/source, original cache
  publisher and exact offline cache. Native Xet bundle refuses before
  Hub/CAS. No invented FirstSeen trust or guessed release payload;
  conflicting options fail before credentials.
- [x] Full frozen isolated `scripts/check.sh` 1,032/0 failures,
  four strict artifact/CLI profile lint and core CI, Nix filtered-source
  integrity and original two Qwen reference release regressions pass.

## Deferred

- [ ] OSS release/distribution naming and "latest" pointers remain
  observational until an exact immutable source and codec version is bound.
- [ ] Remote routed pipelines/peers, encrypted hosted gateways and
  implicit zero-copy cross-device are not enabled by source-free shapes.


### Runtime work deferred until after core model priorities

All entries below are still UNIMPLEMENTED. Their active owners are
[model-execution](../model-execution/TODO.md),
[serving-and-speculation](../serving-and-speculation/TODO.md),
[multimodal](../multimodal/TODO.md),
[memory-and-streaming](../memory-and-streaming/TODO.md) and
[model-acquisition](../model-acquisition/TODO.md).
The completed pipeline-scaffold plan is not a claim of serving/realtime
availability and does not keep pipeline development on the critical path.

- [ ] Connect authenticated application/conversation history providers to
  ContextOwner, with durable and revocable grants, owner-authenticated
  tenant+application+session identity, event/graph epoch binding to the
  existing TaskCoordinator and physical memory/residency admission. Wire
  source-real calibrated NanoJev/other decision heads and streaming media
  A/B tests before enabling an actual context-fed ASR/translation router.
  Do not infer inference support from the in-memory source fixture.

- [ ] Source execution: bind actual learned model, tokenizer, processor
  assembly and port formats in original ModelCatalog/TaskDriver before
  enabling any pipeline run or route listening.

- [ ] Single-process multi-model scheduler: immutable source residency,
  exact shared instance reuse/replicas, typed cross-model buffer leases,
  physical ready fencing, bounded ingress/admission/cancellation and
  model-specific batch/fairness. One TaskCoordinator publishes StateRoots.

- [ ] Pipeline runtime/hot reload: per-session model identity, graph generation,
  buffered media/source epoch, legal stage yields, failure rollback and
  atomic route transition. No stale audio, image or cross-tenant leakage.

- [ ] API serving: explicit auth/bind/route config and tested OpenAI
  chat, ASR, TTS, images; SSE/WS/WebRTC soft-realtime with content/size
  bounds and accurate NOT IMPLEMENTED capability reports until qualified.

- [ ] Context-aware source-real voice pipeline: stage-specific ASR prompt/
  vocab support or separate contextual transcript reranker, acoustic-first
  diarization and source-preserving contextual translation. Implement
  prior-committed same-session snapshot freezing, cancellation epochs,
  explicit consent, redaction, bounded history and no unauthorized
  cross-user disclosures.
  A/B compare WER/CER/DER/translation adequacy/latency for context
  ON/OFF; never override contradicted acoustic evidence. This does not
  follow from shape validation.

- [ ] Actual staged ASR + diarization (speaker-tagged timestamps), ASR
  -> Qwen -> TTS one coherent command; audio sample clocks/codebook
  playback/VAD/barge-in and image latent/complete preview frames with
  backpressure. Validate codec/latency/source on real media hardware.

- [ ] Optimize model execution: Qwen3.5 2B, Qwen3.8 27B
  resident+SSD, Gemma E2B/26B/31B, then Qwen Flash-Next QSA/GDN/MoE/
  learned n-gram embeddings/MTP and separate speculative n-gram/draft,
  with measured model-real speed/quality/physical safety gates.

- [ ] Complete remaining exact Q2/Q3.x/Q4 quant source pipelines,
  full native image/audio/NeMo assemblies, token/codec quality and
  range/resume/cancellation before declaring "all models supported".
  Keep DeepSeek V4.1 Flash and GLM5.3 Flash strictly pipeline-only now.

- [ ] Trained and streaming multi-model parity, real media correctness,
  device completion, SLO/fairness and zero-copy lease gates are owned by
  the model-execution/serving/multimodal/memory TODOs above.

### 24 Oct — parallel implementation checkpoint

- [x] Current2B cached trajectory original15-transition/30-call same-prior math exact; fixture15 actual conv/lastFA/head unknown-work cuts PASS. Source13 successful physics/math retained with explicit actual producer identities and unchanged serving bytes; original failed batch remains FAILED, no current serving/performance grant.
- [!] HIPv4 first healthy57492 refused before child/model/GPU at old-Q import receipt schema; actual source/binary/owner closure preserved, no inference result. Controlv5 real metadata-only preflight accepted both actual receipt schemas but independent review found missing finalizer import alarm_until; distinct control-only successor required before fresh actual serial cases. No source rebuild/import/copy/download needed.

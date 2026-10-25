# Model execution

Provenance commit: `6d74f76`.
Scope: P02/P03/P08/P11–P14/P17; CPU truth before GPU specialization.
Contracts: [large immutable sources](../IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md),
[Qwen basics](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md),
[Gemma family](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md).

## Uncommitted

Current24Oct: CLI and shared model units are integrated; Pi implements/executes,
Sol coordinates/reviews/merges, Astra designs/reviews architecture and math.
Local source21 shared state-certificate preparation passes the complete1793 CPU
gate, combined strict, five readiness contracts and optimized release build.
Current physical23/full2Bmath and both .8B/2B API/HTTP pass ROOT review;
current .8B/2B C1/C4/C8 singlepass rate AND wall wins are accepted. These
finite scopes do not grant fullmatrix/generalparity. Source20 x2's complete
shared HIP unit passes below. Independent Qwen natural feedback and learned
Gemma PLE hit/transfer coverage remain open; next whole-unit design proceeds in
parallel with local .8B→2B benchmark work. All78 signed commits are complete.
Latest user direction resumes all tracks now: separate Pi Qwen27B/Gemma4/
Vulkan/HIP/queue lanes alongside external C16 correction. C32/C64 follow shared
queue acceptance; physical jobs and quiet timings retain coordinated ownership.

Latest owner merge direction: complete the current Qwen27B/Gemma4/HIP/Vulkan/
queue units to a tested, mergeable checkpoint, assemble and reconcile shared
source, then integrate into main and retire completed worktrees. Keep original
reference roots while current checks use them. Broken source or a draft
controller is not a merge checkpoint; preserve pending physical scopes honestly.


### H/G/V physical checkpoint — 25 Oct pass 3

- [~] Three passes executed on pvl-x2 (full evidence:
  ../../runs/resume-all-hgv-checkpoint-execution-sol-20261025/LOG.md).
  Root causes of passes 1-2 fixed and verified on hardware
  (`06b89da` E0308, `c217c35` V parents, `e6ab074` census,
  `3941a3c` Gemma envelope). Pass 3: builds/admins/binds all PASS;
  h-known-qwen-reload + h-qwen-final-reader-unknown PASS (full
  semantic closures); three real failures preserved
  (h-gemma-feedback-unknown: session-state identity/policy/fence
  assert + `scope closes only between completed proposals`;
  g-resident: parity row-upload fixture contract vs legitimate PLE
  row dedup; v-resident: adapter validator requires a false-permit
  field the producer never emits — physical case itself passed);
  19 cases stopped as dependents. Fix sites pinned in
  [the PM handoff](../docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-25-PM.md) §2;
  pass-4 procedure §2 there. No product defect established
  (qualification-gated paths).

### Current shared source20 hardware checkpoint — 24 Oct

- [~] Qwen natural2-position independent original comparison and Gemma learned
  PLE row-hit/H2D/full-state scope, retaining existing quotas/owners. ROOT reviewed
  Astra's complete next-unit design; two isolated Pi implementation/controller
  preparation lanes active, no GPU/reference execution before review. Qwen old
  teacher capture requires independently verified original fullV/prior/source
  join; otherwise fresh natural reference. Gemma reuses its genuine natural
  matrix and adds an actual-hit full35/fullV comparison plus final-owner scopes.
  No new public
  serving, source21 x2 artifact, general-fit or speed grant from source20 fixture.
- [ ] Gemma PLE twofixture source draft and prepared controller need corrective
  review before merge/physical work. ROOT finds diagnosticD2H exactly1B gate
  instead of positive and regressed dataFD/bounds. Original valid source20
  math/lifetime results unchanged; new all-hit/final-reader scope unexecuted.


### Qwen27B and Gemma4 optimization audit — 24 Oct

- [ ] Close isolated source113 as one checked model unit, then actual x2 source/
  state/fault/lifetime qualification: Qwen full64/all112, Gemma resident20/all35/
  fullV/all15KV and observer-free natural feedback. Grouped isolated CPU/strict
  checks PASS:5strict configurations,907CPU library tests/2comparators,46explicitly
  ignored cases. Exact source113/P1/P2 and original400v4 base remain unchanged;
  contextual410 continuous/compiler reconciliation is reviewed and affected
  strict/166boundary/7compiler-owner checks pass with all jobs closed;21physical
  cases explicitly ignored. Later RMS3/source6 excluded. Gemma's finite actual
  Resident20 and Qwen full64/armed-producer fault results follow below; Qwen's
  final Pending/Snapshot reader fault coverage is now closed on frozen469 below.
  Keep current C8 assembly separate.
- [ ] Reconcile frozen models113/P1 hooks and Gemma final-feedback fix with current
  shared CUDA readiness source, preserve arithmetic/backend boundaries, perform
  affected assembled checks and final complete gate before model merge acceptance.
- [ ] Qwen27: qualify natural and temporal decode/prefill, residency/admitted
  prefetch, packed projections, eligible optimized GEMM/attention/recurrent GDN,
  hybrid prefix checkpoints and serving through shared engine primitives.
  Current native route requires streamed weights and finite teacherfrontiers0/1;
  shared64-lane CPU-order BF16 dot is not TensorCore/BLAS. Preserve source/cast/
  reduction profiles rather than inheriting small-model CUDA permits.
- [ ] Gemma E2B final-reader ownership: source466 review finds Session/Snapshot
  drop `loaded` before final committed feedback's independent12-byte Output
  lease. Native/modules/managed-source guard can retire before a true final
  feedback release refusal. Pending already has safe ordering. Isolated Sol
  fix+fixture unit authorized: place `loaded` after inner+feedback; Session/
  Snapshot × healthy/exact final Output-lease refusal, Weak-only diagnostics and
  original actual service/native/source/quarantine owners. Preserve frozen466,
  completed Qwen P1 and readiness sources. Frozen469 ten-path fix+fixture passes
  grouped6control/4probe/all-target strict checks; independent Sol review and ROOT
  complete source/control/cost/archive review pass. ROOT authorized one bounded
  fresh admin/offlinebuild/four-case unit. Admin33537/offline GPU-free build73520
  CLOSED0; first Session-known28952 FAILED_CLOSED at semantic verifier's exact
  `device_with_six_roots_bytes` formula after native child exited0. Actual groups/
  one-reap/fullEOF/source35FD/cache-parent closure healthy. ROOT and independent
  Sol identify verifier-only omission: greedy12336B+six256B alignment guards
  missing from both exact expected totals; production sums/caps are correct.
  Corrected local verifier6controls PASS; new saved Session-known revalidation
  PASS without GPU replay, original FAILED receipts unchanged. ROOT reviewed
  full225line local collector delta/17pins and authorized remaining3 serial
  native cases, exact-known-error-only/source35/cache/binary/originalowner
  closure required. All4actual scopes now close with corrected local acceptance;
  ROOT independently verifies every report/source/cache/original+reader owner.
  Known:184→0modules/sourcecloseOK/model0/context4GiBdevice+2GiBhost. Fault:
  actual12Bfinalreader refusal precedes native retirement,184modules/617handles
  retained/full20GiBfenced/sourceclose refused. Every full35/30planes/fullV/
  one natural commit; original failed remote collectors and SSH1 remain explicit.
  No Session replay/Rust rebuild/serving or performance grant. Existing source466
  cache reused with only actual publication/activity locks writable; no second
  10GB import. Current11+118path isolated combined8package/alltarget strict and
  189CPU PASS/33explicitphysicalignored; original source/owner/EOF closures pass,
  first controller-only compilerfile cap failure preserved. Current12 differs
  only by test-fixture rustfmt outside118. ROOT now adopted models118 plus the
  source-bound2B cachedB4 unit together:173selected/172effective/46new paths,
  fresh mtimes, original README/lock and all unrelated source preserved. Current13
  has1629full files/492selected; combined strict/20sharedCPU/5ABI-source controls/
  authentic2B CPU inventory/reopen pass. Original excluded-vendor Cargo refusals
  remain failed, not executed vendor-test evidence. Optimized13 CLOSED0 and all
  original19physical scopes/healthy2Btrajectory PASS; original2B same-prior30calls/
  8112sections/fullstate/fullV exact. New2B fault global-vs-flight count mismatch
  is diagnosed; fixture-only15 correction builds, preserving every graph/source/
  fence guard. Current assembled phase and final
  assembled gate still required. Prior.8B measured checkpoint remains source11/12.
  [Combined assembly results](../../runs/model-execution-current11-assembled-checks-sol-high-20261024/REPORT.md).
  [Four-case report](../../runs/model-execution-gemma-final-feedback-verifier-v2-sol-high-20261024/REPORT.md).
  [Independent source evidence](../../runs/model-execution-gemma-final-owner-review-sol-high-20261024/REVIEW.md).
- [ ] Gemma E2B/E4B: dependency-based shared-tail fast prefill, PLE row reuse,
  optimized packed projections and eligible attention, physical SWA reclamation.
  E2B15owners/20shared tail exists; E4B needs24/18 and a new functional route.
  Optional vLLM fast prefill is off by default, not a universal baseline feature.
- [ ] Add functional Gemma E4B and12B routes;12B uses Unified geometry/MM mapping,
  no PLE/shared tail. Extend source/profile contracts before learned qualification.
- [ ] Gemma26B-A4B: learned router/grouped experts/dense+expert composition, then
  admitted expert sharing/hotsets/EP. Gemma31B is dense; qualify its learned dense
  path separately. Reuse the existing single raw K projection for configured
  K=V layers, preserving distinct K/V transforms and state. Select the exact
  optimized-router tie/reduction/scale profile; full-softmax fallback is not an
  identical floating-point schedule. Geometry/analytic contracts alone do not
  implement either learned route.
- [ ] Both families: shared continuous serving/temporal work, paged KV/COW,
  full-domain compatible prefix intersection and graph/capture ownership after
  actual source qualification. Shared target verifier/all-prefix rollback before
  Qwen trained MTP or Gemma assistant/KV sharing/sparse draft vocabulary.
- [ ] Later explicit codec/distributed/MM/LoRA profiles: do not substitute lossy
  FP8/FP4/KV codecs for original BF16 or assume source/hardware eligibility.
  CUDA-only fusion and CDNA/gfx12 AITER are not unconditional gfx1151 gaps;
  current ROCm Triton speculative GDN needs actual eligibility checks. Generic
  MXFP4 registry falls back for dense Linear; DBO rejects relevant hybrid/MM routes.

### Current model/backend checkpoint — 23 Oct

- Owner clarification: Qwen27B/Gemma4 must reuse the established .8B/2B code
  wherever applicable. Lift common allocation/residency/transfer/native ownership,
  model operations and family behavior to their existing shared layers; compose
  those primitives in each model. Before merging, inspect callers and a concrete
  reuse map, remove parallel implementations, and justify remaining specialization
  by actual source layout, numerical contract or necessary optimized kernel path.
  Distinct arithmetic/backend eligibility must not be merged by superficial
  operator similarity. Sol HIGH implements; ROOT reviews/reconciles whole units.
  Actual audit found duplicated family orchestration in27B. Sol extracts one
  substantive HybridDecoder composition consumed by BOTH small-model and27B
  recorders, with shared role groups/bindings/state schema. Packed-vllm delayed
  residual versus separate Torch/BF16 arithmetic remain explicit adapters.
  [Astra initial review](../../runs/architecture-astra-qwen-pool-review-20261023/TODO.md)
  accepts the design; settled source/caller reuse and actual checks remain open.
- Shared physical successor now integrated MAIN/ISO345v3. Required local release
  and both small models' original full source/state/fault/API matrices PASS;
  x2 native3/portable6/rawtransport12 plus corrected shared4 HIP/Vulkan healthy/
  unknown PASS in finite scopes. Healthy source retirement and unknown source
  retention were rerun after fixture-only Oracle lifetime correction; production
  ownership unchanged. Original interrupted-controller/missing local13 proof
  remains explicit; fresh13 replay passed. No new learned large-model/performance
  grant. Qwen27B reachable native+managed facade compiles with Gemma; complete
  single-source qualifier/strict validation in progress. Next Gemma residency
  and multiple-native-set work follows Astra architecture-only review and Sol
  HIGH implementation; final assembled gate remains pending.
- Gemma optimized residency unit active in isolated Sol HIGH assembly, following
  [Astra architecture-only design](../../runs/architecture-astra-residency-and-set-key-20261023/DESIGN.md):
  reuse shared authenticated matrices/row transport, cold-select repeated full
  projections and tied embedding/head, widen output-row launches without changing
  full-K arithmetic, normal observer=None before D2H construction, original
  private packet/Ready/state publication. Complete source-shaped cold host/device
  inventory precedes fixed profile selection. PLE selected-row transport remains
  explicit. Implementation/actual numerical/serving/performance acceptance open.
- Shared native/context+Qwen opt-in Conv4 integrated301selected paths. Both small
  Qwen models' required release full physical/state/fault/API scopes and explicit
  streamed2B faults PASS; full isolated gate14 LAST PASS1747Rust/0failed/allchecks.
  Original x2 native3+HIP/Vulkan transport12 also pass their finite scopes only.
  Conv4 transport is not learned27B numerical closure.
- Astra HIGH architecture/design only; Sol HIGH owns implementation. Gemma44v5
  now guarded/fresh-mtime integrated MAIN/ISO343selected paths. Unsafe lowlevel
  source-retention contract has one validated safe facade; source-neutral finite
  HIP policy, full seven-cell/state/lifetime qualifier and shared round probe
  use the original engine/source/flight/publication owners. All parents/donors/
  modes/rootREADME preserved. Original CPU128MiB profile unchanged.
  Actual frozen MAIN314+44 executable on x2 PASS:7source-tokenized prompt cases,
  55completed comparisons/52commits/150rows/all35block boundaries/full262144V/
  all15KV final+prior pairs,46,858,240derived compared words/zero mismatches;
  real private feedback/decoded output, independent sessions/pressure/cancel/
  Ready discard/refusal/restore/reset/source close PASS. Actual shared-round16
  signed-subnormal/tie/cancellation cases and186native modules pass. Fresh
  unknown process:1completed producer/0roots/full16GiB fenced/source-close
  refused; original owners retain until process retirement. Healthy model close
  retains contextdevice4GiB/host2GiB until process exit. Both original local+
  remote source35/group/one-reap/fullEOF closures pass; sampled shared memory is
  not an isolated peak/bound. Three fresh-process ELF hashes differ explicitly.
  [Complete actual checkpoint](../../../../zerogravity-worktrees/gemma4-portable-gpu-whole35-20261023/.agents/runs/model-execution-native-sol-high-20261023/GEMMA_APPLY-v5-2.md).
  Single1MiB synchronous source transport is a correctness baseline; no serving/
  performance/general HIP permit. Binder316 and the new shared physical-owner
  split were not executed by that proof. Next assembled gate/affected physical
  checks follow the coherent successor; gate16 accepted preceding316 only.
- Qwen27B actual original MKL dispatch identified SSE EXHAynn. Sol's shared Rust
  explicit normal-domain128split-table/non-FMA profile matches the authenticated
  connected H48 cached producer, including both exp calls and all786432state
  words bitexact; legacy2/48decay +5539state negative control preserved. Five
  reviewed paths integrated MAIN/ISO302selected paths with fresh mtimes; CPU/Qwen
  strict and existing operator tests pass. No learned
  model/native/serving/performance grant from this fixture closure. Subsequent
  actual full64 replay3 now PASS: two independent sessions each one INITIAL
  token ID11 then one cached token ID353, all1040original records/240896000float words bitexact, all112
  state domains/fullV/ALL4Conv frames,12position words. Source/archive/owned
  closures, precancel/actualpressure/output-retention/finalhost0/unfenced pass.
  Generic profile/wholemodel/native/serving/performance flags remain false;
  natural feedback, wider contexts and GPU qualification still open. Four
  qualification-example corrections now integrated MAIN/ISO with fresh mtimes,
  preserving exact immutable replay evidence. Full assembled gate15 LAST PASS:
  1752Rust/0failed/89ignored/269suites/allscriptchecks,314sourcepaths/rootREADME
  unchanged.
  Prepared-recipe successor also passes both small models' full CUDA/state/
  fault/API scopes. Latest prepared-binding release1 quiet .8B C1+78.62%/
  C4+26.67%/C8−1.09% remains singlepass; final gate16 PASS1752Rust/0failed/
  89ignored/269suites/allchecks accepted316before Gemma integration.

### Parallel model work — contained integration, 22 Oct

- [ ] Continue Qwen27B/Gemma source/runtime integration independently; merge
  contained completed units promptly. Use pvl-x2 Radeon8060S/gfx1151 ROCm/HIP
  for their GPU qualification/benchmarks; preserve exact HF source identities,
  source-derived complete allocations and full logits/state/cancellation gates.
  Existing remote Ollama GGUF inputs are not approved-HF equivalents. Current
  Vulkan/CUDA parity work is a separate implementation/qualification obligation.
- [ ] Gemma follow-on: qualify tokenizer/ordinary learned route and broader
  prompt/cached geometry plus learned cancellation/state lifecycle through the
  same engine; selected diagnostic exactness alone grants no public route.
- [ ] Independent follow-on units continue while short-HF validation runs:
  Qwen authentic H48/128 GDN producer and full64/112-domain schedule, plus Gemma
  family source-bound CPU completion. Preserve actual source geometry and exact
  numerical limits; unavailable learned arithmetic must not use legacy fallback.
- [ ] Gemma trained Rust execution requires exact learned BF16 arithmetic and
  independent source/state comparison. Payload/tokenizer/original CPU oracle
  now exist; native/serving/performance acceptance remains separate.


### Source-specific optimization backlog — 21 Oct, deferred behind current order

Source details and limits: [independent comparison](../../research/engine-optimization-comparison-2026-10-21/README.md).
These are future operators/strategies in the same engine, not payload approval
or model support. Qwen parity/short benchmark/long→dense27B→Gemma remains binding.

- [ ] Dense27B: bind full trained source/state through bounded row/layer windows;
  amortize immutable reads across compatible prompt/request rows, account
  activation spill and genuine upload readiness. MTP metadata/tensors alone
  do not qualify target or drafter runtime; expert hotsets do not sparsify it.
- [ ] Gemma4 variants: source-real PLE, normalization/logit caps, SWA/global
  attention, configured K=V/cross-layer sharing, head256/512 and26B-A4B routing.
  Check graph padding cannot route experts or advance roots. Matching trained
  assistants may share target KV/embedding only under actual architecture maps;
  first qualify target execution and complete branch/precision ownership.
- [ ] Later sparse Qwen: stable finite top-k/tie/block ordering, source-selected
  expert grouping, hyperconnections and deterministic learned PLE lookup lane;
  runtime n-gram proposals are a separate stage. Source-dependent sparse masks
  and reduction/cast profiles cannot be grafted onto ordinary dense attention.
- [ ] Later DeepSeek: exact CSA/HCA/MLA/indexer/compressor/mHC/YOCO/Engram and
  checkpoint-matched DSpark/Markov/confidence contracts; typed dependency/liveness
  proof may eliminate dead prompt work only after every future state read is
  covered. Truncated bounded replay and perturbed sampling are explicit changed
  profiles, not exact defaults. Existing payload acquisition hold persists.
- [ ] Later GLM: distinguish KDA from GDN, MLA/DSA index/open-tail state, mHC and
  own trained MTP/taps. CPU/GPU placement, cached-expert group order and FP16
  prefill must have declared arithmetic; geometry or source names do not prove
  interchangeability. Existing payload/device holds persist.

- [ ] **Independent trained BF16/FLA checkpoint gate:** original
  Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 acquired
  through the existing bounded Rust cache pipeline; independent31-file/
  18-shard full hashes and1199-tensor header/index checks passed.
  Whole-model27B execution remains unqualified; bounded trained operator
  qualifications below do not establish full logits or112 StateRoot domains.
  Genuine original layer0 GDN reference now validates initial chunk64 and true
  cached continuation for teachers11/353,48 value heads and original BF16
  parameters. Plain/observed complete outputs, convolution/recurrent state,
  raw source readsets, full before/after source hashes and original owned
  retirement match. The serializer's V2 stride failure remains preserved;
  V3 raw-bit transport passes12 real PyTorch layouts. This is an independent
  public target. An isolated native composition now passes the authentic H48
  cached slice:804960 input and798720 output words, including the complete
  recurrent state, match independently before/after callers drop. The distinct
  BF16-gamma gated RMS slice also passes61632 intermediate,12288 final and
  24576 retained input words. Source hashes, original ownership, cancellation,
  real pressure and final reservations0 pass; strict/release builds and237
  focused tests pass. These operator changes remain isolated from MAIN;
  initial chunk64, native exp, complete GDN,112-domain state and full inference
  remain unqualified. A separate genuine public reference passes32 pure
  operator cases over16 roles,881856 raw BF16 input/output words, including
  complete248320-row head and17408-column down projection. The isolated
  native wide-role replay now matches all881856 words exactly; real pressure,
  pre-cancel, retained head/source after caller drop and final reservations0
  pass. Strict/release builds and17 source/relocation fixtures pass. External
  probes for other roles do not establish trained producers or full-model
  parity; these changes also remain isolated from MAIN.
  Six actual bounded native-provider startup/lifetime cases pass with the
  provider unloaded. Physical provider allocation/math admission remains open.
  Original
  Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc is now acquired
  by the same bounded Rust pipeline; independent official Git/LFS full12-file
  hashes,632 headers/320 text shapes,596BF16+36F32 tensors pass. Its4.548GB
  shard remains above the default4GiB complete-source guard. Genuine pinned
  official CPU two-profile reference capture now passes archive and owned
  cleanup validation: tokens11/353, four full248320 logits,168 canonical
  domains and60 actual intermediates. Official BF16 loading has617 BF16
  parameters; restoring only36 original F32 tensors gives581 BF16+36 F32.
  The profiles differ in47/49 raw tensors per step despite identical argmax;
  neither establishes Rust2B HF numerical qualification. First capture's
  cleanup-name bug is preserved; the repaired fresh run proves actual nonce
  container absence and process reap. Original Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17 pinned
  Transformers5.17/Torch2.13 CPU captures passed archive validation:
  tokens11,353, full248320 logits/all42 domains, actual layer0 stages.
  Keep official473-BF16 loader and source437-BF16+36-F32 profiles distinct;
  first-token argmax321 vs198. Explicit new CPU64 diagnostic vs source-mixed
  reference remains UNQUALIFIED: embeddings/direct layer0 norms and both
  QKV projections exact; layer0 convolution state exact,41/42 domains still
  differ. Full248320 logits differ214245/214285 words, maxabs .22119140625/
  .1875. First token exposes conv/SiLU signedzero; second exposes GDN state.
  Complete pinned convolution, chunk64/cached recurrent, coefficients and
  gated RMS arithmetic, then original GPU/FLA/state/streaming gates before enabling
  trained HF execution or27B serving. User authorizes required runtime/model
  acquisition/testing; dated source-free slice restrictions below are history.
- [ ] **Next trained reference gate:** qualify exact original HF BF16
  rounding/FLA order and independent source-trained logits/StateRoot;
  then bind the original profile to the existing per-op
  MathematicalBackend/SessionAuthority, with full physical
  device/SSD resource admission. Earlier source-free slices establish no
  full27B logits,112-domain state, native provider admission or GPU support;
  current owner authorization covers the required acquisition and testing.

- [ ] **Following ME-P0 real trained integration:** fuse the
  accepted analytical original source operators into the ONE
  existing forward_token/MathematicalBackend and
  SessionAuthority execution schedule, qualify original HF BF16
  intermediate casts and FLA GDN accuracy, source-trained exact
  full logits and StateRoot, tokenizer, physical mutable-state
  memory accounting, cancellation and device completion. No
  SSD/physical inference claim follows from the diagnostic bridge.

- [ ] **ME-P0 next after diagnostics:** wire the analytically verified
  Qwen original operators into the ONE existing forward-plan and
  TaskDriver/SessionAuthority (no second token loop or state issuer),
  with exact source BF16 intermediate cast and original FLA GDN kernels,
  real trusted tokenizer and full 248320 logits/42-or-more StateRoot
  domains against the pinned checkpoint. Only then begin
  SSD-resident streaming and measured Qwen3.8-27B physical throughput.

- [ ] **ME-P0 next numerical stage:** source-exact original Qwen
  full attention, GDN/MLP, grouped Q/K→V replication, A_log/gate
  transformation and BF16 intermediate numerical schedule using the
  shared original TaskDriver/StateRoot. Qualify tiny independent
  analytic operator/state gates before real approved trained Qwen3.8
  27B original full-vocabulary reference. Measured original Qwen3.5
  2B CPU/GPU end-to-end performance also remains open.

- [ ] **ME-P0 next:** bind source-approved original Qwen3.8 27B
  tensor/index/config and tokenizer to the existing mathematical engine
  with authentic head/axis geometry, strict trained CPU full logits/
  StateRoot, then source-verifiable immutable SSD extents with stable
  leased memory and measured decode/prefill. First finish measured
  Qwen3.5 2B CPU/approved GPU end-to-end performance and
  cancellation/physical readiness. Preserve no model/SDK downloads
  and held device launch policies without explicit authorization.

- [ ] **ACTIVE NEXT: ME-P0 Qwen3.5 2B then Qwen3.8 27B.**
  Begin with original qualified Qwen3.5 .8B/2B CPU and resident GPU
  source/state oracles, inspect release source ownership and exact
  model-2B profiling on the permitted CPU/GPU devices. Keep tests
  bounded to existing 8K context now; 16K/32K extensions remain
  separate numerical gates. Qualify a stable single-model fast path,
  original tokenizer/math/KV/recurrent/StateRoot and actual device
  completion first. Then implement checked immutable source shard leases
  above the present 4GiB executable-source gate and Qwen3.8 27B
  complete trained CPU logits, including SSD streaming and quant-aware
  physical placement. Do not fetch weights or launch held Renoir/gfx90c
  hardware without the corresponding permission and source integrity.
- [ ] **NEXT after Qwen: ME-P1 Gemma4 E2B→26B-A4B→31B**, independently
  source-bind tensors/reference semantics, PLE/MoE/dense quantization
  and trained CPU→GPU correctness before measured throughput
  optimization. **ME-P2 Qwen3.8 Flash-Next** follows with separately
  verified learned n-gram embedding/4B MTP heads, QSA/GDN/MoE and
  explicit Q2/~3.5bpw/Q4 source quant choices (NOT guessed files),
  speculative n-gram as a distinct runtime optimization. Other OSS,
  ASR/TTS/image/video/decision execution and multi-model router remain
  on their own deferred source-real gates.
- [ ] Preserve direct one-model-per-process execution and one
  shared-instance-many-pipelines as **one Rust fabric**; the accepted
  YAML DAG/compiler and extension schemas add no inference backend.
  Latest21Oct owner request authorizes parallel recursive pipeline/profile
  design after independent engine research. It must not delay ME-P0/P1 or
  imply new trained context/media/router execution.

- [ ] Enforce new owner priority: Qwen3.5 .8B regression, Qwen3.5
  2B highest-perf trained CPU/GPU, Qwen3.8 27B with SSD
  streaming/owned shard tensors, Gemma4 E2B, 26B-A4B, dense 31B,
  then Qwen3.8 Flash-Next QSA/GDN/MoE/gated residual/MTP
  and 51B learned n-gram embeddings. Every additional DeepSeek,
  GLM/Muse/Mistral/Llama family has a typed NOT IMPLEMENTED
  source contract until distinct trained CPU and GPU gates pass.
  No subfamily is generalized from metadata-only source support.
- [ ] Treat single-model local GGUF execution and multi-model
  pipeline-qualified execution as two frontends of ONE TaskDriver/
  SourceIdentity/StateRoot fabric. A reusable model instance
  serves multiple routes without another model process or implicit
  source promotion; named replicas/adapters require separate
  physical memory admission.

- [ ] After the accepted source-acquisition v1 cutover, use the existing
  pinned HF index-first path to inspect explicit original Qwen3.8-27B and
  Gemma4 26B-A4B index/config/tokenizer/chat/processor identities
  (see [acquisition v1 handoff](../model-acquisition/README.md)).
  Select an authorized exact revision and admitted local host. Do not treat
  synthetic >51GB transfer-plan tests, public model pages or cached bytes
  as authorization, execution-ready source, or trained CPU parity.
  2026-10-18 tokenless live metadata-only HF source observation resolved
  Qwen3.8-27B to `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  (18 shards, 55,586,113,293 selected bytes) and Gemma4-26B-A4B-it
  to `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
  (2 shards, 51,644,338,194 selected bytes); Gemma4 E2B-it commit
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
  (1 unsharded safetensors, 10,278,848,001 selected bytes).
  No trained weights downloaded, locally owned, header-audited or
  model-execution-qualified yet.

- [ ] Acquire/bind authorized original config/index/tokenizer/checkpoint identities and choose a capable admitted host. Large BF16 sources exceed current local RAM; current executable source limit stays 4 GiB.
- [ ] Qualify real large (>4GiB) original trained shard payload on an
  authorized capable host using the already verified owned-shard
  implementation from 93a5622; independently measure resident
  allocation/physical RAM pressure and immutable lease lifetime. Next,
  develop the separately admitted SSD streaming extent architecture.
  The bounded source-only synthetic tests do not establish real large
  learned weight loading.
- [ ] Qwen3.8 27B: original HF grouped-head ordering and Q/K replication 16→48, explicit converted-profile permutations, untied head and source-derived norm/RoPE geometry.
- [ ] Gemma E2B/26B: shared GELU/multiply/scale/soft-cap, proportional RoPE, value norm/sliding/shared KV, E2B PLE/doubled FFN and 26B K=V/MoE branches; choose and qualify the exact reference-framework numerical schedule.
- [ ] Qualify actual trained CPU operators/full vocabulary/state/nonzero replay, official tokenization/templates/reasoning/stop behavior, sessions, rollback/checkpoints and resource refusal.
- [ ] Add the same source-bound forward plans and full physical GPU qualification per eligible device, preserving all state/actual-completion/cancellation guarantees.
- [ ] Keep distinct future heavy/hybrid recipes (Qwen Flash-Next, GLM, MiMo and source-defined QSA/MLA/KDA/indexers/hyperconnections), and broaden dense/encoder/pooling/reranking/LoRA support only through independent model/codec/backend gates.
- [ ] Preserve CPU-first family priorities; already-authorized parallel basics do not authorize model/SDK downloads or unsupported whole-model claims.

- [ ] Bind explicit original HF versus converted Qwen source profiles. Conversion permutes grouped/tiled V projections/constants/convolution/output, uses -exp(A_log) and adds1 to non-GDN norms; GDN swish is SiLU while full-attention gating remains sigmoid.
- [ ] Choose and independently qualify Gemma26B reference-framework routing/top-k/reduction/scale placement before production execution. Five analytic test-local routing cases are not trained source parity.

## Committed

### `29a98c1` — HIP source-page events and witness (25Oct assembly)

25Oct integration: accepted H14 pending-event source plus final sticky witness
error projection is now copied into MAIN with fresh mtimes. ROOT independently
matched all14 files/524,658B and modes; README/index preserved. New diagnostic
bound2,640,926B consumes the existing128MiB fixture allowance; conditional cold
host33,530,474,220B is unchanged. Enabled-HIP typing and actual H6/G8/V10 physical
cases remain pending. V's fixed SSH adapter is source-reviewed only. CPU queue
16/32/64, cap/cancellation/HOST and actual P16 tokenizer checks pass separately;
they do not qualify GPU models or performance. The collected CUDA direct-route
ownership/host-cost repair landed in `52f76a0`/`efd7a34`; the final full CPU
gate passed rc0 25Oct and the five-lane batch landed as signed commits
(`8f01635`..`68e13a8`). Five-tree retirement remains pending.

- H14 accepted source is the transferred state behind this commit;
  enabled-HIP typing and the actual H6/G8/V10 physical cases remain open in
  Uncommitted scopes. Vulkan lane source landed in `19ff059` (final five-fix
  strict-SSH adapter source-reviewed; V10 physical pending); Qwen27B audit
  corrections folded into `e8db111`; Gemma4 examples/tools into `332038e`;
  queue serving into `5d40cf5`/`52f76a0`.


### `225fc5e1a520` — shared family sources and finite learned-model diagnostics

Related source commits: `bebfeaf2d817` (gemma4-text-reference), `9af407bf3ff5` (qwen35-27b-window-replays), `2058dcb1ea3f` (facade-qwen27b-hip-source), `71f496b190dc` (facade-gemma4-hip-drivers), `752642361cb4` (facade-shared-hip-lifecycle).
Aggregate source21 qualification applies to the assembled tree; no per-commit
build or general model/backend grant. Historical failures keep their source scope.

#### Current shared source20 hardware checkpoint — 24 Oct

- [x] ROOT accepts the actual x2 offline build/binary binding and complete
  finite known3Q/4G reload plus two unknown1Q/1G lifecycle cases. Revalidated
  original typed receipts, exact1650 source,70dataFD, artifact51808f9e…,
  global fence, both process-group/reap/EOF layers; all earlier failures kept.
  [Actual scope](../../runs/source20-x2-actual-namespace2-pi-20261024/REPORT.md).

- [x] ROOT authenticates original Qwen teacher receipt/manifest and independently
  scans allfour248320V BF16 records. Both original call0 select220, while held
  call1 consumes353; compositional natural continuation refuses correctly.
  Fresh natural reference producer/validator/controller preparation required,
  never relabel teacher call1. ROOT+Astra reject correction draft: signedzero
  stillwrong, fresh package table stale, scopes/staging incomplete; replacement
  controller has invented caps/reap-before-signals. No merge/physical grant.
  [Coherent corrective design](../../runs/model-execution-astra-qwen-natural-capture-correction-20261024/DESIGN.md).

#### Qwen27B and Gemma4 optimization audit — 24 Oct

- [x] Separate parallel read-only audit of actual vLLM0.31 benchmark sources,
  stable0.32 and frozen current upstream;20Qwen/20Gemma/15shared comparisons.
  [Report, exact model matrix and source links](../../runs/model-execution-qwen-gemma-vllm-audit-20261024/REPORT.md).
  Neither family currently has all applicable vLLM optimizations. Source dispatch,
  planning inventory and source113 assembly do not grant physical/serving/perf.

- [x] Independent full Gemma4 series review: G01–G20 × E2B/E4B/Unified12B/
  26B-A4B/31B, plus shared S01–S15 eligibility. Existing raw K=V projection
  reuse is already in the family schedule; final normalized/rotated KV stays
  distinct. Pinned optimized Triton MoE routing and Torch fallback have different
  reduction schedules. Current E2B GPU12,288-column limit does not admit12B/
  31B down projections; extend source-bound profiles/inventory, not inherited
  limits or permits. Variant KV owner/PLE costs and MAIN versus frozen118
  boundaries are explicit. No new runtime/performance qualification.
  [Complete matrix, corrections and reuse map](../../runs/model-execution-gemma-series-audit-review-sol-high-20261024/REPORT.md).

- [x] Source466 x2 readiness/control unit reviewed and executed through actual
  packs/builds/GPU-free imports; Gemma Resident20 healthy+fresh unknown PASS.
  Old namespaces/controls immutable; source466 excludes later RMS3/source6.
  [Commands, costs and limitations](../../../../zerogravity-worktrees/model-source410-checks-sol-high-20261024/.agents/runs/model-execution-x2-readiness-sol-high-20261024/READINESS.md).

- [x] Fresh Gemma source466 source-only pack reviewed:32,286,720B archive
  `28fcd32410715277668534efc14ce18aadf998b0fd5bbde6e759480ad482226d`,
  sourceclosure`b404cb9a831b896d5a179a07a08b27f9fe0ccf3ce9f8e2d5ec2f7839f5d19330`.
  All1576regular tar members' bytes/SHA, selected466 source/modes,11controls and
  13protected guards exact; source/tar owners closed. Local prep grants no fit.

- [x] Gemma fresh source upload/bootstrap/build and complete original P2/source
  closure PASS; actual binary/source identities retained below and in run results.

- [x] Qwen genuine local data/source preparation closed:35member data archive
  56,521,973,760B`8109be7cd03b1232569f312fac75496ee6dfa6da6ccffad13e30d8ec63b09036`,
  source32,276,480B`39ae07bde9a08afc0e8203b09762c4682e93aae934841620e2802dbb4ffdc2c8`.
  Original held source reads/copy hashes stable; all1575source tar members exact,
  selected466modes/10controls/13protected guards exact; shared1562source records
  match Gemma. No downloaded weights or data-derived expected outputs.

- [x] Both original fresh admin/bootstrap/build units CLOSED0; preparation/build
  alone do not establish GPU fit. Actual model scopes remain separate below.

- [x] Both source466 fresh x2 builds passed with actual held artifacts/source and
  complete original owner/EOF closure. Gemma readonly35FD authentication and
  genuine GPU-free new96GiB CacheStore import passed:8payloads10,278,848,001B,
  snapshot`ba32b7693d0f7dcbdf4f00f94d13bd61c838e2d7a927f1879817cd5e74fca8b1`.
  Installed user/PID/net/bwrap readonly35data+binaryFD namespace executed; old
  source/cache unchanged. Subsequent actual GPU dev-bind/Resident20 scope passed
  below; no whole-family or serving/performance grant.

- [x] Gemma E2B frozen466 actual x2 HIP Resident20 healthy+fresh unknown CLOSED0.
  All55diagnostic calls/all35blocks/full262144V/all15KV exact,52commits150rows;
  50normal observer=None calls/50commits145rows preserve actual state/private
  feedback with zero optional producer/KV D2H and repeated selected-weight H2D.
  PLE selected-row H2D remains2,598,400B; explicit state snapshots separate.
  Healthy model/source/native close retains honest4GiBdevice/2GiBhost context
  until process retirement. Freshunknown1producer/0roots/full20GiB fenced/source
  close refused and original owners retained until process retirement. Actual
  184native modules independently priced; complete device17,469,201,928B under
  unchanged20GiB profile.16physical round cases pass; all source/data/binary/
  original groups/one-reap/fullEOF closures pass. Shared sampledVRAM is not an
  isolated peak; seven fresh ELF hashes differ explicitly. No serving/speed or
  current late-RMS/source8 grant.
  [Complete actual scope](../../../../zerogravity-worktrees/model-source410-checks-sol-high-20261024/.agents/runs/model-execution-x2-readiness-sol-high-20261024/execute-source466-v1/PHYSICAL-SETUP-RESULT.md).

- [x] Qwen466 actual original receive+GPU-free new96GiB CacheStore import CLOSED0:
  31payloads55,586,113,293B, snapshot`98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e`.
  Same held binary1fec59d5/sourcea9d86cb4; all35dataFD/fullSHA/source and original
  local/remote owner/EOF closures stable. No model/GPU call from import.

- [x] Qwen466 original x2 healthy sixfull64/all112/two teacher-session matrix,
  then dependent freshunknown authorized. Healthy localSSH18683 CLOSED1 after
  wireless route loss; original local group/reap/EOF closed but remote GPU is NOT
  retired by that receipt. Subsequent configured strict-hostkey Tailscale read-only
  collection confirms original remote healthy PASS/1097.656s: sixfull64calls,
  fourcomparisons/commits/two independent teacher sessions,240,896,000floatwords/
  12positions/zero mismatches, fullV/all112domains128planes/Conv4/priors/finals.
  Ready discard/latecancel/pressure/restore/reset/lastoutput/sourceclose pass;
  117actual native modules/16physical round cases pass. Remote original supervisor/
  child absent, group/one-reap/fullEOF/source/data closure healthy. Local original
  transport failure stays FAILED_CLOSED255, not an end-to-end collector pass.
  Freshunknown88524 CLOSED0 through reviewed LOCAL-only configured Tailscale
  transport derivative:1actual producer/0roots/full16GiB fenced/sourceclose
  refused, original source/native/armedflight retained until process retirement.
  Both original local+remote owner/source/group/one-reap/fullEOF closures pass.
  No healthy replay/recovery. Mandatory native-gpu user/net and held
  binaryFD; supervisor-held35dataFD SHA/lifetime. Gemma's nested readonly-host/PID/
  dev-bind/dataFD namespace is not present in this original Qwen GPU controller;
  import's private GPU-free bwrap is separate. P1 final-owner Pending/Snapshot
  actual HIP fault coverage remains open regardless of finite matrix outcome.
  [Complete physical scopes and transport distinction](../../../../zerogravity-worktrees/model-source410-checks-sol-high-20261024/.agents/runs/model-execution-x2-readiness-sol-high-20261024/execute-source466-v1/QWEN-PHYSICAL-RESULT.md).

- [x] Frozen469 alias-v2 actual P1 final Pending/Snapshot reader retirement:
  all4 fresh HIP cases PASS; ROOT independently checked every actual report and
  original local/remote group/reap/fullEOF/source/binary/35FD/cache-parent closure.
  Internal aliases3/3/2/2, commits0/0/1/1, Candidate1/model1/workspace-target0;
  exactly one true original lease release precedes one native retirement.
  Known:117native modules retire/model0/sourcecloseOK/context4GiBdevice+2GiBhost.
  Fault:117modules/1602handles/service/source retained/full16GiBfenced and
  sourceclose refused until actual process retirement. Each real64layers/
  128proposedplanes/full248320V; unchanged numerical matrix reused, no replay/
  old-order negative/natural-temporal/serving/speed grant. Original failed469
  incomplete internal alias census remains failed; exact equality preserved in
  repair. Genuine own P1 cache reused with actual locks, no second55GB import.
  [Four-case results](../../../../zerogravity-worktrees/qwen-final-owner-alias-sol-high-20261024/.agents/runs/model-execution-qwen-final-owner-alias-sol-high-20261024/REPORT.md).

#### Current model/backend checkpoint — 23 Oct

- [x] **Original27B final63 tail, external boundary only:** pinned public CPU
  reference and the existing Rust `OriginalWindowOperators::execute_output_tail`
  agree exactly on all2,609,152 BF16 words: four external post-attention probes,
  independent1+3 rows, two repeats, nine complete streams including248320 head.
  Actual31-file/18-shard full source identities, held reference FDs, pre-cancel,
  real exhausted-host refusal, output retained after caller drops and final
  host0/unfenced pass. Nine Python fixtures, Rust example3/strict Clippy and
  complete isolated scripts/check.sh pass. Expected pin is sealed after actual
  independent validation; no expected bytes supply computed stages. Scope ends
  at this tail: full64-layer producer,112-domain StateRoot, GPU/serving and
  performance remain open. Source now integrated in MAIN and ROOT; fresh MAIN
  full checkgate passes. The external-tail numerical proof remains the original
  ROOT replay above; no fresh full27B execution follows from source convergence.

#### Parallel model work — contained integration, 22 Oct

- [x] Completed combined40-path CPU donor integrated into ROOT ISO with fresh
  bytes: Qwen27B final63 attention→existing output tail plus Gemma4 genuine
  26B-A4B MoE through the same mathematical backend/source-window/session paths.
  Reciprocal source review, combined CPU libraries/replays and strict all-target
  Clippy pass in isolation. Learned Qwen rotary/BMM parity remains unqualified
  (reference pinNone); Gemma learned checkpoint/oracle not yet qualified.
  Source readiness does not establish trained/native/serving/performance support.
  This and the follow-on CPU batch now converge into MAIN; final assembled
  isolated MAIN gate PASS1531 Rust/0failed/81ignored/260suites plus18 namespace
  fixtures and complete scripted release/installer/lint/qualification checks.

- [x] Follow-on CPU model source batch integrated into ROOT ISO: Gemma31B
  exact60-layer/60-owner analytic schedule and21504-column windows (113isolated
  CPU tests/strictClippy PASS); Qwen27B original64-layer traversal/shared H48
  GDN hook/private SessionAuthority diagnostic (176/82/74libraries+6replays,
  strictClippy PASS). Existing public trained/native routes remain refused.
  Qwen initial-one is not originalchunk64; minimal Conv history3 is distinct
  from original HF4. Real retired H48 exp differs2/48 and recurrent state
  5539/786432 words; generic f64 exp also fails. Preserve fatal exact comparison,
  no tolerance/correction or complete112-domain/full64 numerical claim.

- [x] Complete owner-authorized managed HF Gemma4 E2B pull at pinned
  3e22461f65e89153144f8adb70e3b8c2cc9845a7. Tokenizer/config sidecars are
  acquired and verified. Owner-approved canonical cache64→96GiB applied,
  preserving models; full8file/10,278,848,001B managed checkpoint closure
  published and acquisition hashes verified. Offline full closure verified;
  authentic263,952B header/2,011BF16 tensors and derived index prepared, all600
  text members match registered geometry. Acquisition grants no Rust/native or
  serving qualification; original learned reference/replay results follow below.

- [x] Gemma E2B original learned CPU reference executed and independently
  validated: full checkpoint, BF16/eager, prompt4 then two natural continuations,
  two independent cache cohorts,93 records/full262144 logits/all15KV owners and
  exact consumed priors. Original8-file source and owned lifetime close healthy
  under12GiB/4CPU. Strict Rust AnalyticF32TextV1 replay completes the full35-block
  first call, rejects logits[0] (`c1ce0000` vs `c1ce9ba0`), publishes no root,
  retains unchanged source and releases reservations to0. Neither numerical
  parity nor native/serving/performance support follows from this failure.

- [x] Gemma BF16 whole35-block candidate and producer-boundary tracing merged
  into ROOT ISO (20 paths); default analytic arithmetic and trained refusals
  preserved. Source-defined embedding/PLE scales, direct-gamma RMS, rotary,
  projection/BMM, residual and final casts use the same ReferenceText engine.
  CPU tests/strict Clippy pass. Fresh original reference2 passes all natural
  calls, fullV/all15KV/repeats/priors/source/lifetime,1981 producer records.
  Strict whole35 replay rejects first original-order boundary at layer0,
  position0,ffn.gelu[416]: source -0 (80000000), candidate b4240000.
  Root/frontier remain0, final reservations0; this is not learned parity.

- [x] Resolve Gemma candidate transcendental arithmetic at its source:
  faithful pinned BF16 AVX2 GELU/tanh schedule, same full-range formula and
  source saturation; no expected operands or input correction table. CPU batch
  passes; strict whole35 replay against unchanged healthy reference2 closes the
  selected producer prefix through810 complete records, then rejects layer12,
  position0,ple.post_norm[880] (3b830000 vs3b820000). Same1981 observations,
  root/frontier0, unchanged source/final0 and original owned closure. Seven
  source paths now converge MAIN/ISO. Pinned ATen pow(-.5) selects1/sqrt;
  candidate general powf differs. Shared source-shaped RMS correction is now
  merged MAIN/ISO; CPU batch/strict Clippy PASS. Actual whole35 replay matches
  all1981 selected initial producer records and initial+first cached fullV/all15
  KV pairs, commits greedy886→2915/frontiers4→5. Second cached call fatally
  differs logit0 (c1b00000 vs c1b10000); prior root2/frontier5 unchanged, source
  and actual owned closure healthy/final0. Its KV comparison was not reached.
  Detailed cached producers were not captured, so initial matches do not
  establish cached arithmetic. Extend all-call diagnostic comparison/tracing
  within existing caps, then compare a fresh independent producer reference.
  This was the intermediate diagnostic limit; later PV closure below supersedes
  its selected three-call failure. General trained/native/serving stays refused.

- [x] Qwen27B original full64 producer executed healthy: teachers11→353 with
  two independent cache cohorts, all851 raw source roles/67modules,1040 records/
  934,776,928B, authentic Conv4/FP32 recurrent/head-majorKV. All851 observed source roles
  are BF16, no source casts. All31 whole source
  identities/mapped bytes unchanged, owned exit0/group/container/EOF close.
  Eleven additive producer tools plus contained observed-exp causal test merged
  into ROOT ISO; source-free Python protocol8+4 fixtures PASS. Independent fresh
  original archive/source validation PASS all1040 records/exact repeats/priors/
  fullV/all112 domains and source/lifetime. Rust parity remains false.

- [x] Qwen original full64 archive connected to the existing Rust diagnostic,
  explicit HF Conv4→minimal3 mapping and complete available-record comparison.
  INITIAL1 earliest producer divergence and correction remain tracked below.

- [x] Separate cached Qwen GDN exponential correction, finite original profile:
  actual original SSE HA dispatch reconstructed with mathematically derived
  split128table/separateF32 operations; both A_log→A and logdecay→decay computed.
  Authenticated H48 connected producer/all786432state words bitexact; legacy
  2/48decay and5539state negative control preserved. Actual full64 replay3 closes
  initial+cached/2independent sessions/all1040records/240896000words including
  all4Conv/completeKV/fullV, source/ownership healthy. No golden operands or
  looser limit. Numerical primitive's explicit finite normal domain remains;
  general mode/tails/natural/long/native/serving/performance stay unqualified.

- [x] Qwen full64 Rust replay/source-owned scratch unit merged into ROOT ISO
  (13 paths): same traversal/SessionAuthority, explicit Conv4→minimal3 and
  head-major→token-major mappings, expected-only completed-boundary observer.
  H48 transient scratch retires after synchronous compute; output/state retains
  original ownership. Whole-replay host envelope768MiB, derived upper651296768B;
  old128MiB leaf limits unchanged. CPU105 checks/strict Clippy pass. ROOT sealed
  only the expected-target reference pin after validating actual retired raw
  archive/manifest/invocation/source. No kernel operand or runtime permission.

- [x] Recover the first learned Rust replay's pre-boundary GDN output capacity
  refusal with bounded exact output allocation below; no raised caps. Initial
  refusal opened no expected archive and kept root/final0/source/owned closure
  healthy. Subsequent full64 replays now reach complete numerical comparison;
  startup-only namespace/affinity refusals establish no numerical result.

- [x] Qwen bounded GDN output correction integrated MAIN/ISO: declare/reserve
  exact output before fallible collection, retain capacity and element bounds,
  reject short/excess/original errors. Covers activated/gated and analogous
  cached recurrent BF16 output, unchanged arithmetic/caps. H48 reproducer
  confirms prior capacity8192/len6144; related CPU47+whole-window5 and strict
  Clippy pass. Corrected learned replay5 completes the actual64-layer proposal,
  then strictly rejects layer9 output word13 (3cc80000 vs3cc60000). Embedding,
  layers0..8 and first13 words of layer9 match,51213 words in this reader order;
  no post-state/internal-stage comparison was reached, no cached call ran.
  Source before/after/full expected SHA/EOF, pre-cancel/real pressure and actual
  owned closure pass; root unchanged/commits0/final host0/unfenced. Observed
  boundary reservation517760000B is not an exact peak. No causal attribution
  to older cached exp results. Next same completed callback must compare all
  available state/boundary/fullV records before refusing, with bounded summaries.

- [x] Complete Qwen proposal comparison source merged MAIN/ISO;9 fixtures and
  strict Clippy PASS. Actual learned6 compares all196 initial records/39842304
  float words plus3 position metadata words. All20 Conv3/H/KV tensors through
  layer9 exact; first differing state is layer10 Conv after known layer9 output
  divergence. First bad layer9 output differs381/5120 words. FullV differs
  197638/248320; no commit/root unchanged/final host0/unfenced and full source/
  expected/owned closure pass. No hidden earlier state mismatch was found.

- [x] Qwen layer9 causal diagnostic source merged MAIN/ISO (20 paths): opt-in
  fifteen borrowed producers from the same full64 Rust traversal, completed
  full-proposal and48-record prefix comparison before numerical refusal. No
  arithmetic/default/native change; conservative upper664928256B under same
  768MiB cap. Python6/session6/example10/strict Clippy PASS. Original full64 pin
  unchanged; separate prefix pinNone until independent actual validation.

- [x] Original64-object prefix independently validated: two fresh initial
  caches/135 used roles/18 logical domains/96records/52355504B, all66 overlaps
  byte-exact to unchanged accepted full64 original. Stop after layer9 source
  unload; source whole checks/31FDs/actual process+container+group/reap/EOF close.
  ROOT checked artifacts and sealed distinct expected-only prefix receipt
  14593B SHA4caffb384f0a67b20bb3a81aded8fc151b4bfd949659aab28cb2d5bf7a02b657.
  Old full64 pin unchanged; no original final/head/full-model claim.

- [x] Actual full64 layer9 causal replay closes with strict numerical refusal:
  input/InputNorm/QKV/Z/B/A exact; first error GDN core2/6144 BF16 words,
  word806 actualb6d60000 vsoriginalb6d70000. NormGate exact, gated2→Out32→
  postresidual3/postnorm3→MLP divergences→381/5120 output differences. All20
  layer0..9 state tensors exact; first state error layer10Conv downstream.
  Full196-record comparison reproduces learned6. Root unchanged/commits0,
  source/31FDs/expected/actual owned closure healthy, finalhost0/unfenced.
  Observed boundary530355200B<derived664928256B<768MiB is not a peak.

- [x] Correct Qwen INITIAL1 GDN core at its source. Genuine original pads to64
  BEFORE intra-chunk F32 matmul; current product+sum_inner128 is not a proven
  equivalent reduction. Actual MKL SGEMM_BATCH(T,N)/four-thread dispatch and
  learned inner producers identify sequential FP32 FMA as the correction.
  Shared primitive and same-iterator diagnostics integrated; complete initial
  full64 replay passes below. Cached MKL exponential remains separate/open.

- [x] Qwen actual inner-prefix causal comparison closes: convolution/activation/
  Q/K/normalizedQ/K/scaledQ/betaV all exact; first mismatch QK36/48 F32 values,
  then4549/6144 rawcore and2 BF16core words. FMA candidate using only genuinely
  computed Rust scaledQ/K/betaV matches all48 QK and6144 rawcore words exactly.
  Original116-record capture joins all96 oldprefix+66full records; ROOT reviewed
  expected-only SHA9355c45eef22726c25710faaaf897777fa6f9ab47f7dec9780262289a7873c48
  and12 actual marked MKL dispatch intervals. Actual unpublishedprefix callback1/
  commits0/rootunchanged, source31FD/ownedclosure/final0/unfenced. Report
  SHA51b5d18f6209b2f2472e034814a593cbbce8baf522e2ec79e9e1e3c0b454851f.
  Shared INITIAL1 sequential-FMA primitive now wired and30 reviewed source
  paths merged MAIN/ISO. Actual full64 replay: two independent INITIAL1
  teacher11 proposals,196 records/39,842,304float words each; all79,684,608
  words exact including fullV and mapped state. Two callbacks/commits, actual
  pre-cancel/pressure/retained-output/source31FD/owned closure/finalhost0 pass.
  Replay204551B SHA34c81e5118c8df4ad275ff0833e372cbf70eca54def149f76c14476626e1465b.
  Complete Conv4 equivalence, cached353/MKLexp, natural/native/serving/performance
  remain unqualified. MAIN gate5 PASS1645Rust/0failed/86ignored/265suites;
  later common cache/backend drafts are outside this gate.

- [x] Gemma all-call diagnostics integrated MAIN/ISO (10 paths):3965 producers/
  22598144B within unchanged caps, complete numeric scans before refusal. Fresh
  original v2 capture/independent validators PASS; ordinary fullV/KV raw and all
  1981 initial producer records equal immutable reference2. Gemma19/example5/
  Python11/strict Clippy PASS. Actual Rust calls0/1 match all1981+992 producers,
  fullV/all15KV; call2 compares all992 producers/all31 fullV+KV records. Earliest
  actual mismatch is layer9/position5 attention.pv[2421] (head4/channel373), one
  4096-element PV word:3d460000 vs3d470000, with exact observed six-term inputs.
  Callback rejects despite matching greedy886; root2/frontier5 unchanged,
  source/owned closure healthy/final0. No full three-call parity promotion.

- [x] Gemma source-shaped BF16 PV correction merged MAIN/ISO (5 paths): pinned
  GEMM-NN four FP32 partials/tail/final order, source small-BMM branch and full
  masked-prefill contraction from actual RootAttempt call geometry. Default
  analytic/QK/GEMV, source identity, caps and native eligibility unchanged.
  PV4/reference11/Gemma19/example5/strict Clippy PASS. One learned whole35 replay
  against unchanged healthy v2 matches all3965 selected producers and all93
  fullV/KV records across three natural calls886→2915→886; roots1/2/3,
  frontiers4/5/6, source unchanged/final_reserved0 and actual owned closure.
  Report SHA199ca15df9d6d02017b2fb3fff94bcaeb1bd4dbc6a669bd4f5850486bc3fee6f.
  Scope: authentic4-token prompt plus two natural cached calls only; general
  trained/native/serving/performance support remains unqualified.

- [x] Gemma selected learned lifecycle source5 paths merged MAIN/ISO; existing
  completion callback handles late refusal/cancel with no new engine hook.
  Two actual sessions, pre-cancel/layer9 observer refusal/completed refusal/
  post-comparison cancel/same-root retry/natural survivor; full producer/fullV/
  all15KV comparison and actual root/KV/reservation invariants. Gemma20/example6/
  strict Clippy/release PASS. Same128MiB+8MiB limits; one actual learned run active.

- [x] Actual Gemma learned lifecyclePASS: two natural sessions886→2915→886,
  roots1/2/3/frontiers4/5/6. Pre-cancel/actuallayer9PV refusal have0completed
  callbacks; completed refusal+latecancel each compare992producers/fullV/all15
  KV then preserve root1/allKV hashes/reserved79441036B. Retry+survivor pass;
  all8callbacks/9914producers/248completed records exact. Source/actual owned
  closure/finalreserved0 healthy; selectedCPU only, generalflagsfalse.

- [x] Assembled108-path MAIN isolated fullcheckgate4PASS:1560Rust/0failed/
  82ignored/261suites,18Python namespace checks, complete scriptClippy/release/
  installer/lint/qualification/shell/Nix. Model learned runs above are separate
  real-source checks; neither test durations nor Qwen refusal grant support.

- [x] Gemma bounded tokenizer/text preparation source merged MAIN/ISO:
  approved32,169,626B tokenizer uses BPE262144 vocab/514906 merge pairs,
  bytefallback/fuse_unk, Replace spaces→▁/Split and identity single-sequence
  postprocessor; no implicit BOS. Reuse tokenizers0.22.2 algorithms via bounded
  typed streaming construction; full generic JSON graph is not a fit proof.
  Actual original/Rust comparison passes46 text and15 byte/control decode
  cases in both special-token modes. Requested GlobalAlloc construction peak
  94099192B/retained57113812B, notRSS; original source/owner closure and final0
  pass. Construction116MiB+8MiB control plus4MiB caller fits128MiB admission.
  Source review fixes serialize allocating operations nonblockingly, retire
  provenance before admission and verify final source stamps after hashing.
  Tokenize→drop leaves only64KiB IDs/provenance before model load. Focused
  Gemma24/featureexample7/legacyexample6/strictClippy/Python6 PASS. Opt-in
  --plain-text wiring uses actual tokenizer IDs and the same learned35-block
  engine. One actual replay PASS: all3965 producers and93 fullV/KV records exact,
  natural886→2915→886, roots1/2/3/frontiers4/5/6, unchanged source/final0 and
  heldbinary/process/group/reap/EOF closure. Report SHA
  716ee16ec29e1b80855055c873254f2421e57029a1766bb9bbf4abd569998354.
  Final assembled gate pending; default CPU gate now includes optionaltokenizer
  feature and six source-only Python contract tests.
  No chat/multimodal/generaltrained/native/serving/performance promotion.

- [x] Gemma ordinary plainCPU31-path candidate integrated MAIN/ISO: managed
  held SnapshotLease/source windows, bounded selected-ID decoder reusing source
  decoder algorithms without retained BPE graph, private same-session greedy
  feedback/EOS[1,106,50], explicit CLI profile. Derive whole128MiB coexistence;
  whole derived108773100B fits128MiB; bounded decoder61 actual cases in both
  special-token modes exact, original source/final0/owned closure healthy.
  Full assembled gate6b PASS1713Rust/0failed/88ignored/266suites plus scripted
  checks, including candidate CLI feature. This is source/decoder acceptance;
  broader ordinary runtime qualification remains open below.

- [x] Complete frozen actual seven-cell Gemma ordinary matrix: genuine plain
  M1/2/3/4/32/57 plus UnicodeM3, natural extent64/fullV/all15KV/consumedpriors,
  source/cancel/refusal/retry/sibling checks. Original captures and lossless
  BF16 archives within unchanged caps. Actual Rust PASS50natural committed
  calls +2completed rejected proposals,52 fullV/all15KV comparisons and10710
  actual35-block row-boundary events,0mismatchedwords. Pre-cancel/refusal/late-
  cancel/retry/independent survivor/pressure pass. Sourceafter true, prepared
  3670016B→caller3145728B→final0. Report255906B SHA
  cef7dd0dddc18710caff6e6a23c6b8046c401507e635b8acd68a8745b62ace1e;
  ROOT independently checked numerical counts/frontier64/report/source/owned
  closure. Frozen binary unchanged/FDclosed, groupabsent/reaped/EOF/cleanup[];
  same original7200s deadline including bounded855.9s pause, no recapture or
  expected operands. Finite ordinary CPU candidate scope only; generic trained,
  chat/native/HTTP/performance flags remain false.

- [x] CLI dispatch successor integrated MAIN/ISO: option-position scan preserves
  existing Qwen --prompt '--source-profile' and every existing value flag/alias,
  while selecting the explicit Gemma profile. Eight matching-feature CLI tests
  and strict Clippy PASS; both original parsers and frozen matrix unchanged.
  Assembled isolated gate7 PASS1715Rust/0failed/88ignored/266suites plus scripted
  checks; actual Gemma matrix retains its immutable earlier binary scope.

- [x] Qwen27B CPU source-window output tail integrated into ROOT worktree: final FFN norm → gate/up → staged BF16 SiLU/multiply → down → residual → final norm → full vocabulary, through existing MathematicalBackend/HostMath. Fourteen owned paths merged with fresh mtimes after exact parent checks; integrated model67/core17 tests PASS. Explicit post-attention input remains the boundary; no trained whole27B/112-domain/serving parity claim. [Scope and checks](../../../../zerogravity-worktrees/root-lowest-f32-20261020/.agents/runs/model-execution-qwen27b-root-integration-20261022/TODO.md).

- [x] Complete isolated CPU integration gate now covers Qwen27B + Gemma: scripts/check.sh PASS,1490 Rust test passes/0failed/80ignored across257 suites,18Python checks, strictClippy/release/installer/shell/Nix PASS,0warnings. Full trained inference/native/serving/performance qualifications remain open.

- [x] Gemma E2B source-bound AnalyticF32TextV1 CPU reference integrated: one generic full35-layer schedule,15KV owners/shared consumers,PLE/proportional rotary/sliding attention/finalnorm/tied full vocabulary and original SessionAuthority transactions. Eighteen owned paths contextually merged with fresh mtimes, preserving27B. Tiny genuine source-window fixtures/nonzero synthetic continuation/rollback and complete CPU integration gate PASS. [Scope and checks](../../../../zerogravity-worktrees/root-lowest-f32-20261020/.agents/runs/model-execution-gemma4-root-integration-20261022/TODO.md).

### fff1842 — trained BF16 operator qualification

- [x] **Native CUDA CPU-profile BF16 projection:** both original Qwen3.5
  sources, layer3 q/k/v windows and teachers11/353, B1/independent B1/B2/4/8:
  163840 raw F32 accumulation words match the accepted CPU64 operator and
  163840 rounded BF16 words match authentic public archives exactly.
  Same sealed full-source windows, original CUDA queue/service/leases/Ready;
  pressure, pre-cancel, source/axis refusals, final physical0 and fresh actual
  abandoned-flight source/completion/fence retention pass. Synthetic328
  case/band pairs cover source widths64..8192, signed zero, subnormals, RNE,
  FMA/reduction order and canaries; eight fresh FP32/BF16 overflow flights
  withhold Ready and retain original quarantine. Strict Clippy/five focused
  tests and independent trained/synthetic raw-byte validators pass. This is
  the explicit CPU-target profile, not vLLM CUDA arithmetic, full HF/GDN,
  model StateRoot or a throughput claim. Original failed build/import checks
  and their additive harness corrections remain preserved.
- [x] **Original GDN convolution/SiLU fragment:** both authentic HF sources,
  layer0 teachers11/353, initial padding and cached-tail inputs: all122880
  BF16 words across raw convolution, activated output and canonical next3
  history match independent public Torch operators exactly. Public outputs
  first reproduce original activated/K4 archives. Same ForwardPlan/HostMath,
  sealed raw BF16[C,1,4] SourceParameter and externally authenticated inputs;
  actual pressure/pre-cancel/wrong-role, retained output after callers drop,
  full parameter stage accounting and final physical0 pass. Core10/artifact2/
  model48 tests and strict CPU/CUDA/Vulkan checks pass; trained replay has
  independent all12-file signed-bit validation. Standalone JSON-macro syntax
  failure is preserved; fresh ROOTV7 changes only harness parentheses.
  Full HF/session/checkpoint guards remain closed. Initial FLA chunk64,
  cached recurrent state, coefficients/gated norm and full logits/StateRoot
  remain required before enabling original2B/27B inference.
- [x] **Composed BF16/conv phase:** complete isolated1251 Rust tests,
  zero failures/70 ignored/248 suites,18 Python tests, release/installer/Nix
  checks pass. Composed strict CUDA/qualifier/Vulkan and CPU-free binding8/
  GPU cold-refusal2 tests pass. Both approved original release cache-direct
  and HF-import routes reproduce full248320/all42 for both models and retain
  original source/lease/mutation/reclamation guarantees.

### fe2a342 — original attention and reference operators

- [x] **Original CPU attention fragment:** both authenticated sources at
  layer3, teachers11/353, source-36-F32-restored profile: all47104 raw BF16
  words across seven projection/view/QK-norm outputs match authentic original
  captures, including signed zero. Same HostMath/ForwardPlan and sealed source
  windows; real source-pool pressure, pre/mid-cancel, foreign input/provider,
  retained output after callers drop and final reservations0 pass. Explicit
  external hidden inputs establish no preceding producer or StateRoot.
  Focused16 tests, strict CPU/CUDA/Vulkan, both approved original release
  routes and complete isolated1243-test phase pass. Fresh helper strict build,
  full source scans and owned process/group/EOF cleanup also pass.
  Earlier metadata-list and omitted second4MiB binding-accounting failures
  remain preserved; corrections change only the replay harness. RoPE, KV,
  attention BMM/softmax, GDN/FLA, full-HF logits/StateRoot and27B stay open.

- [x] **Authentic vLLM projection operands:** read-only extraction from the
  accepted four-call eager-B1 capture preserves all original full496640-logit,
  320-source-audit, provider and cleanup guards. Six raw BF16 files,36992 bytes,
  bind actual teacher11/353 inputs/results, public GEMM pointers/settings and
  learned QKVZ(qkv,z)/BA(b,a) row hashes. Ten transport tests and independent
  rederivation pass; result f0981e62. No new model/GPU execution or dtype cast.
  Native BF16 arithmetic, opaque provider allocation admission and full-HF
  execution remain unqualified.

- [x] **Original2B paired attention observation:** fresh plain/observed CPU
  captures preserve all196 raw tensors/four full248320 logits/168 canonical
  domains/60 prior intermediates against the accepted two-profile reference.
  Observed capture adds216 genuine BF16 tensors across six attention layers,
  nine stages and teachers11/353 in both original loader profiles,688128 bytes.
  Exact signed-bit/shape/stride/source checks, unchanged archive validator,
  all632 source/617 loaded parameter audits and owned group/container/EOF
  cleanup pass. Ten parser fixtures pass. Diagnostic elapsed44.56s/42.01s;
  no benchmark or physical RAM admission claim. Result e45e1903 qualifies
  transparent reference-stage availability, not Rust attention output,
  full HF execution/StateRoot, GPU/FLA, long context or trained27B.

- [x] **Genuine pinned vLLM V1 eager B1 public GEMM observation:** four
  unchanged original0.8B layer0 QKVZ/BA projections at teachers11/353 use
  cublasGemmEx, BF16 A/B/C, FP32 compute, requested DEFAULT_TENSOR_OP,
  default math/host pointers/atomics disallowed, alpha1/beta0 and explicit
  8519680-byte workspace. Actual full tensor/workspace allocation extents,
  original provider versions/mapped SHA and loaded320 source roles before/
  after pass; plain/observed all496640 logit words are exact. Strict observer
  build and all three owned process/container/group/EOF cleanups pass.
  Original header-build and generic resolver failures remain preserved;
  additive exact already-loaded provider binding has no alternative lookup
  or math fallback. Prior resolver failure cause remains unproven.
  This records public requests, not hidden selected kernels, all42 state,
  wider/default graphs, Zerogravity GPU math/profile or matched performance.
  Qualifier result55d68c7a; explicit workspace alone does not qualify opaque
  library allocation admission. Pinned default vLLM loader/quant registries
  have no GGUF entry; original BF16 remains the matched-source route.

### bfe62df — leased source-window execution

- [x] **Shared source-window HostMath fragment:** original27B851-role
  namespace and same PlanBuilder/HostMath embedding→ordinary RMS→full-column
  projection now execute genuine trained bytes without source materialization.
  Both tokens11/353 match all40,960 independent BF16-widened reference words,
  including full5120-wide embedding/norm and10240-row QKV. All18 shards/
 55,563,006,776B are verified through retained FDs. All80 authentic128-row
  windows preserve5120 input columns; maximum real QKV staging charge8,388,656B
  and raw norm staging4,194,328B are measured, not inferred from requested bytes.
  Each output retains admitted64KiB Output+960KiB Scratch and source issuer
  through last drop. Real stage-triggered mid-cancel, pre-cancel, pool pressure,
  foreign-source/config/index/token refusal and retained output comparisons pass;
  final reservations0/unfenced. Actual92.74s includes90.79s source scan;
 18,736KiB observed RSS is not physical free-RAM admission evidence.
  Fresh MAIN-linked original0.8B three-operator regression also matches all
 16,384 words/final source0. Focused17 artifact/1CPU/5model, strict CPU/CUDA/
  Vulkan/qualifier Clippy and full isolated1243 Rust/0failed/70ignored/248suites
  plus18Python/release/installer/Nix pass. Both approved original cache-direct
 17.22s/HF-import22.74s release routes pass fullV/all42/reclamation for both
  models. Only four pointless borrowed-value test drops needed repair. This is three pure state-free operators,
  frontier delta0: full112-domain27B logits/StateRoot and streamed inference
  remain unqualified.


### 5350a4d — typed ordinary math and source windows

- [x] **Typed ordinary RMS in the shared schedule:** the same PlanBuilder/
  HostMath embedding→raw InputNorm(0)→GdnQkv(0) fragment preserves source
  parameter identity, BF16 precision, epsilon1e-6 and explicit packing.
  Fresh MAIN replay matches all16384 genuine original0.8B reference words,
  two tokens11/353, with final source reservation0. Unaware providers and
  GPU plans refuse before callbacks/source admission; full HF session,
  forward and checkpoint guards remain closed.
- [x] **Sealed rank1 parameter windows:** BF16/F16/F32 raw ranges retain
  original full descriptors, issuer, opened source and actual staging charge;
  source/cancel/corruption/capacity/drop tests pass. Partial ranges never
  become complete TensorSource views. This is transport, not27B inference.
- [x] **Closed original BF16 row-window projection:** authentic original
  full-matrix descriptors and complete input columns reuse the accepted
  CPU64 dot without changing its arithmetic. Output-row partition, width5120,
  foreign source/range/dtype/nonfinite/cancel/provider and lease tests pass.
  No partial matrix/provider substitution or final BF16 cast; caller must
  admit private output/scratch. Same-HostMath27B integration remains open.
- [x] **Closed pinned F32 softmax:** source Vec8 maximum/sum order, unchanged
  SLEEF exponential and F32 reciprocal/multiply, width1..512. Genuine public
  full3599-case replay matches1838736 output words and3 refusals; authenticated
  default72-case golden preserves11280 expected words/3 refusals. NaN/+Inf/
  all-masked rows refuse. No legacy GGUF or full HF attention enablement.
- [x] **Composed acceptance:** focused core9/artifact16/CPU47/model6 pass,
  compact default5 pass; full isolated gate1241 Rust/0failed/70ignored/248
  suite runs plus18Python/release/installer/Nix passes. Strict composed CUDA+
  HF+CPUcache, Vulkan serving and CUDA qualifier pass; CPU-free HF binding8
  and GPU cold refusal2 pass. Both original cache-direct/HF-import RELEASE
  routes pass(18.88s/24.75s), full248320/all42 and final source reclamation.
  Full27B Rust reference comparison, streamed model state and CUDA/FLA remain
  open. Compilation/type and Clippy-only first failures were fixed; no
  mathematical tolerance or full-profile guard changed.


### c5eb1f8 — closed pinned sigmoid and ordinary RMS

- [x] **Closed pinned BF16 sigmoid:** explicit contiguous iterator spans,
  vector32 prefix plus the original scalar tail, private SLEEF and licensed
  Arm/glibc-FMA arithmetic. Genuine public Torch and Rust replay match all
  326753 output words, including all65280 finite BF16 values in vector/scalar/
  mixed call shapes and both actual original beta16 calls. No host exp,
  fitted output table, numerical-profile enablement or GGUF math change.
- [x] **Closed pinned ordinary RMS:** raw immutable rank1 BF16/F32 learned
  weights, exact BF16 input, source four-Vec8 reduction/carry order at widths
  256/1024/2048/5120, F32 division and runtime(1+raw weight), then final BF16
  cast after the F32 product. Actual original .8B122 norm calls preserve98
  prior full-logit/cache checks;32 public width/dtype fixtures cover the
  larger geometries. Rust matches all154 cases/1694 full stage fields/
  1680488 words exactly. Width fixtures do not qualify2B/27B inference.
- [x] **Composed source-free gate:** complete isolated check1218 Rust passed,
  0failed/69ignored/248 suite runs plus18Python, release/installer/Nix;
  ordinary RMS7 focused tests and strict CPU Clippy pass. Independent source
  review authenticates every6792 compact golden word from genuine public
  archives. Strict final composed CUDA+HF, Vulkan serving and CUDA qualifier
  Clippy pass; both approved cache-direct/HF-import release routes pass
  (20.09s/24.88s), both original models, full248320/all42, independent sessions,
  source leases and reclamation. Original-HF full session/forward/checkpoint
  guards remain closed; coefficient exp, initial chunk64, attention and
  physical model/state integration are still required.

### 2a32aff — typed source composition and vector softplus

- [x] **Typed original-source operator integration:** explicit program
  profile and activation precision, BF16 OutputInput embedding/projection
  contracts and raw SourceParameter metadata use the existing PlanBuilder
  and HostMath. Private operator slices carry no domains, logits receipt
  or frontier advance. Unknown physical providers and resident GPU plans
  refuse before source reads/admission. Raw learned parameters stay borrowed
  from the same owner; existing RootBinding descriptors distinguish initial/
  cached phase, BF16 conv/KV and FP32 recurrent axis layouts. These metadata
  views grant no freshness, state transport or publication capability.
  Complete original-HF session/forward/checkpoint guards remain closed.
- [x] **Genuine typed operator qualification:** pinned public Torch
  F.embedding for original tokens11/353 followed directly by independent
  GdnQkv(0), FfnGate(0) and tied Output projections matches every Rust bit,
  including all248320 words of both output projections. Eight reference
  payloads preserve518144 values; same loaded source, raw matrix pointers,
  two-node recorder, zero initialized state and final source reservation0
  are checked. No input normalization, complete layer/model logits or
  GPU/27B inference is claimed by this pure operator composition.
- [x] **Closed pinned FP32 vector softplus:** source beta1/threshold20,
  finite general-F32 contiguous spans divisible16, explicit SLEEF3.8
  compensated log1p/exp/FMA order and BSL notice. Actual32 original biased
  alphas and28464 adversarial F32 values, reordered values and16/32/48/64
  spans match public Torch bit for bit. No scalar-tail, MKL coefficient
  exp, general profile or default GGUF arithmetic change.
- [x] **Typed/softplus composed gate:** full isolated check1213 Rust
  passed/0failed/69ignored/248 suite runs plus18Python, installer/release/Nix;
  focused core/CPU/model158 passed/2ignored, CPU-free binding10 passed,
  resident cold refusal2 passed. Strict composed CUDA+HF, CPU-free HF,
  Vulkan serving and CUDA qualification Clippy pass. Both approved original
  cache-direct/HF-import release routes pass (17.15s/23.12s), both original
  models, full248320/all42, independent sessions/source lifetime/reclaim.

### cadcbd9 — closed pinned original-HF CPU operators

- [x] **Closed pinned CPU original-HF operators:** source-bound BF16
  four-tap convolution, FP32 width128 cached GDN, raw-F32-gamma gated RMS
  prefix and contiguous vector SiLU. No model/session/initial-chunk permit;
  default GGUF/scalar profiles stay unchanged. Convolution preserves four
  independent FMA partials followed by direct adds and BF16 output. Cached
  GDN uses direct query division by sqrt128, actual key-major FP32 state,
  materialized key reductions and separate update multiply/add; observed
  decay is an explicit input. RMS casts normalized core before raw original
  F32 gamma, retaining the FP32 product. SiLU ports pinned SLEEF3.8 AVX2
  exp/FMA/scaling and FP32 division with source attribution/license, only
  finite exactly BF16-widened contiguous inputs with length divisible16.
  Twenty-eight focused Rust tests pass including old projection/reducers.
- [x] **Actual closed-operator Rust qualification:** untouched pinned
  Torch2.13 public SiLU with observed SLEEF AVX2 provider supplies exact
  outputs for ALL65280 finite BF16 inputs plus4096 actual source gates;
  Rust matches every F32 bit. Full16-head cached replay matches262144
  state words plus5*2048 normalized/scaled/core/output words. Gated RMS
  prefix matches8256 words across both actual tokens/all16heads.
  Convolution and composed vector-SiLU/BF16 match all6144 channels at
  four actual causal positions (49152 output words). These are pure
  operator replays of genuine source operands, not full-model/profile,
  GPU/FLA or27B numerical acceptance. Initial chunk64, coefficient exp/
  sigmoid/softplus, ordinary RMS and eager attention remain open.
- [x] **Composed verification:** complete isolated check1202 Rust passed/
  0failed/69ignored/248 suite runs plus18Python/release/installer/Nix.
  Strict CUDA+HF-binding+portable, Vulkan serving and CUDA-qualification
  Clippy passed; CPU-free original binding6 and CPU serving22 passed.
  Both approved original0.8B/2B cache-direct/HF-import RELEASE routes
  retain all248320 logits/all42 domains, source lifetime and reclamation.
  Independent long-context GGUF GDN divergence remains a separate required
  gate in [numerics-and-context](../numerics-and-context/TODO.md).

### 9815504 — original HF roles and closed CPU projection/reducers

- [x] **Original HF source roles and closed CPU profiles:** authentic rank/axes,
  raw source dtype and tied/untied heads bind to the same loaded owner;
  Safetensors output/input rows remain untransposed. Typed original-HF
  metadata carries the actual config/index/bundle and independent state
  geometry. Its execution/session/checkpoint-v1 entry points refuse before
  allocation/backend/I/O; metadata is not an executable-model promotion.
  Explicit Torch CPU AVX2 BF16 projection diagnostics retain the pinned
  64-lane FMA/reduction order and separate state/trace schema; the original
  scalar/GGUF profiles remain distinct. Actual original .8B layer0 QKV
  is exact6144/6144 for each of tokens11,353. Closed FP32 width128 L2 and
  key-axis reducers pass authentic compact bitwise goldens and adversarial
  head/order/finite/overflow cases. Real attention-input norm observations
  now cover both full and GDN layers. Focused composed suites185/0/2ignored,
  CPU-free binding8/0, strict CPU/CUDA/Vulkan/qualifier Clippy and full
  isolated check1164/0/69ignored/245suites+18Python passed. Both approved
  original .8B/2B cache-direct/HF-import release routes remain exact full
  logits/all42. Full original HF mathematics remains UNQUALIFIED.


### f26232e — authentic tied original output heads

- [x] **Original HF tied output head:** verified `tie_word_embeddings`
  selects the original embedding or independent `lm_head` projection;
  no alias tensor or fallback source. Source-free authenticated tied/untied
  fixtures test opposite-sign independent heads, exact F32/BF16 diagnostic
  logits, immutable source identity and final lease0. Full isolated gate:
  1148 Rust passed/0 failed/69 ignored/245 suites,18 Python. Both approved
  original0.8B/2B cache-direct/HF-import release routes retain exact full
  logits/all42. This fixes the diagnostic head; no trained HF promotion.

### 1c0bb41 — pinned numerical reference seam

- [x] **ME-P0 pinned original numerical reference gate:**
  extend one immutable source owner with canonical source-bundle
  SHA256 over pinned index and all authenticated shard bytes.
  Capture a bounded subset of original Qwen GDN, attention, FFN
  and logits through shared PlanObservation, with opt-in 32MiB
  private trace and a single selected GDN recurrent state layer.
  Strict typed independently SHA256-pinned JSON comparisons bind
  config/index/bundle, exact stage order/geometry/position/dtype,
  detect first mismatched element, and distinguish truly bit-exact
  BF16 output from bounded, tolerated FP32 state deviations.
  Source-free adversarial tests verify same-index/different-weight
  digest changes, duplicate/missing/reordered reference content,
  bad token/source/dtype, selected GDN state and immutable roots.
  Fix full-model BF16 residual result-cast omission by reusing
  its source-typed existing residual binary operator.
  This stage DOES NOT prove independence merely because a
  reference JSON asserts trained-framework origin.
  Complete isolated scripts/check.sh passed 1,135 Rust tests, 0 failures,
  69 intentionally ignored across 238 suites plus 18 Python fixtures,
  release CLI/installer and CPU numeric comparators. Strict default and
  composed CPU/cache/Qwen/acquisition Clippy -D warnings pass.
  Independent approved Qwen3.5 .8B/2B release cache/direct 17.45s and
  HF-import 21.97s preserve exact full 248320 logits, all 42 StateRoot
  domains, two-token independent sessions and source/lease correctness.
  Offline Nix source-integrity passed with the checked trace files.
  No trained Qwen3.8 reference or Torch/FLA runtime was accessed.

### 6d74f76

- [x] Qwen3.8 27B and Gemma4 E2B/26B configuration, tensor and source-index contracts; no new trained execution, tokenizer or checkpoint loading.

### Source contracts — 023703f

- [x] Qwen3.8 metadata/shard roles and CPU-free Gemma4 family contracts,14 Gemma tests and official metadata/config cases passed within metadata scope. No tokenizer, new learned weights, physical shard-header binding or trained inference follows.
- [x] Preserve Qwen/Gemma source axes, metadata precision and routing/KV facts in the [model semantics reference](../../docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md) and [Gemma technical contract](../../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md). The executable-source4GiB cap remains.

### 93a5622 — original verified immutable source shard owner

- [x] **ME-P0 LS-B01–B05, verified owned sharded-source allocation:**
  implement `SourceWeights::prepare_verified_shards` in the existing
  artifacts owner with exactly opened original Safetensors file
  descriptors, caller-expected index SHA256 and whole-shard SHA256/length,
  one original SafetensorsIndex parser and checked header name→shard,
  dtype/shape/offset/payload binding. Preflight limits and reserve whole
  captured file allocation plus a conservative metadata envelope via the
  original caller-attested host `ResourceArbiter` before copying. Reparse
  exact owned bytes, compare authenticated headers against preflight
  (including same-length mutation) before issuing the single original
  SourceIdentity. Arc owner lifetime pins all original slices and RAM
  charge; default GGUF `SourceWeights::open` keeps the 4GiB cap and
  original Qwen execution path.
  10 source-only integration tests pass: positive F32/F16/BF16, complete
  index closure, foreign tensor metadata rejection, wrong SHA/length,
  missing/extra/duplicate/wrong shard, same-descriptor pathname
  replacement, original inode mutation, verified copied immutability,
  multiple admitted/shared original leases and exact resource rollback.
  Sparse >5GiB header tests refuse before reads under both 4GiB
  default and physically insufficient explicit >4GiB budgets.
  **No actual >4GiB weight load, SSD streaming, trained original
  Qwen27B/Gemma math, tokenizer or physical GPU support is claimed.**

### b796a2e — original Qwen text source geometry and CPU value binding

- [x] **ME-P0 original Qwen3.8 source/value binding (small original
  format, not trained 27B):** bind caller-authenticated original
  QwenTextConfig SHA256 and pinned indexed-source SHA256 on the *same*
  immutable SourceWeights allocation. Check every ordinary-text
  name and original [output,input] tensor shape/encoding, including
  the untied output head; distinguish source extra vision/MTP from
  unregistered text roles. Reuse the existing CPU source MatVecRegion
  for HF packed F32/F16/BF16 output-first matrices with no transpose,
  plus the existing embedding/parameter-vector decoder through a
  source-bound QwenOriginalCpu façade. Small complete 4-layer hybrid
  original-format integration tests and negative hash/shape/missing-
  head/source-provider tests pass. Oversized HF source matrices
  refuse before the optional accelerator is invoked (observed no-dispatch).
  The complete frozen isolated
  `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`
  passes 1,103 Rust tests, 0 failures, 69 ignored across 238 suites,
  plus 18 Python namespace fixtures, strict lint, installer/Nix
  formatting and CPU numeric selftests.
  Both independent Qwen3.5 0.8B/2B original trained
  source-parity release routes pass AFTER the final safety guard:
  cache/direct 16.70s and HF-import 22.88s, each full 248320-vocabulary
  logits, all 42 StateRoot domains, two tokens, independent sessions
  and lease/mutation/reclamation exact. Default and
  acquisition/cache-CPU strict `cargo clippy --locked --offline
  --all-targets -- -D warnings` checks also pass. The original
  GPT-family GGUF CPU and session path is unchanged.
  Does NOT run trained 27B, create a Qwen session, implement
  grouped GDN permutations, or stream SSD.

### b11e58c — original source-bound Qwen analytical one-layer proposals

- [x] **ME-P0 original Qwen3.8 analytic one-layer operators:**
  source-bound ordinary zero-centered RMS gamma (1+weight),
  HF head-interleaved full Q/gate projection, per-head Q/K norm,
  source-configured partial RoPE, 24Q/4KV grouped lookup,
  QK/softmax/AV/sigmoid/out projection and candidate KV slabs under a
  source-and-layer-bound immutable `OriginalFullPrefix`. Source GDN
  four-tap causal conv and SiLU, exact 16→48 repeat_interleave of
  original Q/K, per-head L2, stored A_log→-exp(A_log),
  dt_bias/softplus/sigmoid beta through the existing coefficients and
  recurrent step, output GDN gated norm/out projection. An immutable
  `OriginalGdnPrefix` holds exact source/layer and private tentative
  conv/recurrent state, never replacing SessionAuthority. Source FFN
  post-attention RMSNorm, SwiGLU and residual reuse old math.
  Small independently derived analytical checks include zero-centered
  norms, 16→48 head signed-zero mapping, 24Q/4KV GQA six-query
  grouping, ln2 decay beta0.5/gain0.5, full attention gates/partial
  RoPE, GDN two-token private repeat/branch, source identity mismatch,
  finite and dimensional errors. The original 8K diagnostic bound and
  original default GGUF behavior remain. **No trained HF BF16 intermediate
  rounding, full layer/model StateRoot, tokenization, actual 27B learned
  weights or GPU/SSD support claimed.** Strict source-config-specific
  numerical reference and official checkpoint tests remain open.
  Final isolated `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4
  python3 scripts/native-gpu.py -- bash scripts/check.sh`
  passed 1,111 Rust test executions, 0 failures, 69 ignored across
  238 suites, plus 18 Python namespace fixtures, release CLI/installer
  and CPU numeric-comparator scripts; strict default and composed
  CPU/cache/Qwen/acquisition Clippy `-D warnings` passed.
  Actual offline Nix source-integrity passed. Two independently run
  Qwen3.5 .8B and 2B trained-source release routes remain exact:
  cache/direct 17.36s and HF-import 22.49s, each original complete
  248320 logits/42 StateRoot domains/two tokens/source leases.
  This diagnostic-only milestone is recorded in signed commit b11e58c.

### 6329cff — source-bound Qwen one-layer TaskCoordinator

- [x] **ME-P0 source-bound original Qwen StateRoot admission
  (one-layer diagnostic only):** the sole existing TaskCoordinator
  now supports an opt-in original immutable source identity plus
  a unique coordinator/session ID while retaining its generic
  publication API. A stateless source-bound one-layer bridge maps
  canonical ForwardPlan Conv/Recurrent/Kv StateSlots to versioned
  GDN or full-attention StateRoot domain bytes, with exact model,
  epoch/session, layer, root, schema and frontier verification.
  Immutable source-pinned numerical proposals publish via the
  existing TaskCoordinator only after complete finite-domain
  preparation, 8K token/64MiB diagnostic caps and early next-KV
  admission. Independent tests cover GDN two-domain and causal
  attention KV commit, abandoned and stale competing steps,
  previous immutable roots, corrupted/nonfinite state refusal,
  source/session/schema mismatch and generic coordinator parity.
  There is NO separate multi-layer token loop, model scheduler or
  new state issuer; this is not yet trained BF16 reference parity.
  Final isolated full scripts/check.sh passed 1,116 Rust tests,
  0 failed, 69 intentionally ignored across 238 suites plus
  18 Python namespace fixtures, release CLI, installer, shell/Nix
  formatting and CPU comparator selftests. Strict default and
  composed CPU/cache/Qwen/acquisition Clippy -D warnings passed;
  independently repeated Qwen3.5 .8B/2B release cache/direct
  17.11s and HF-import 22.27s retained exact 248320 logits,
  42 StateRoot domains, two tokens and source lease behavior.
  Offline Nix source-integrity passed before signing.

### e43ba4c — original Qwen full-model FP32 analytical forward

- [x] **ME-P0 full original-HF model analytical forward:** extract
  the one Qwen layer traversal from the existing GGUF forward_token,
  preserving its complete MathematicalBackend operator order.
  Use it also for original source-bound GDN/full-attention/FFN,
  each layer's private Conv/Recurrent/Kv state and residual output,
  and final RMSNorm/untied head. One opt-in TaskCoordinator
  owns **all** model state domains, not one root per layer.
  Validate all source identities/schema/finite bytes, 8K/128-layer/
  128MiB diagnostic bounds and complete logits before atomic
  publication. Tests: complete synthetic four-layer original
  checkpoint with seven state domains/two tokens, matching
  independent one-layer GDN, abandoned/competing/foreign session
  and late-corrupted state refusal, derived final logit ~8, and
  shared GGUF traversal ordering/fail-fast. The method is explicitly
  named `prepare_diagnostic_token_f32`: this is **not** trained BF16/
  FLA inference, production TaskDriver or an alternate runtime.
  Complete isolated scripts/check.sh passed 1,119 Rust test
  executions, 0 failed, 69 intentionally ignored across 238
  suites, plus 18 Python namespace fixtures, release installer/CLI,
  Nix/shell formatting and numeric comparator selftests.
  Strict default and composed CPU/cache/Qwen/acquisition Clippy
  -D warnings passed. Both independent original Qwen3.5 .8B/2B
  trained-source release routes preserve exact 248320 full logits,
  42 StateRoot domains, two tokens and source/session leases:
  cache/direct 17.88s and HF-import 22.59s, both exit0.
  Offline Nix source-integrity passed before the signed commit.

### aaf9ba6 — original Qwen BF16 source dtype diagnostics

- [x] **ME-P0 explicit original BF16 dtype diagnostic:**
  implement finite nearest-even BF16 rounding with signed-zero and
  overflow refusal; introduce optional BF16 source stages into the
  existing original Qwen source binder/CPU operators, without
  mutating the default FP32 path. Cast after ordinary FP32
  (1+gamma) RMSNorm, per-head Q/K norm, HF output-first
  projection, SwiGLU, residual, RoPE, BF16 QK matmul *and*
  scaling, post-FP32 softmax, attention value and gates.
  GDN follows BF16 conv/SILU, FP32 A_log/softplus, BF16 beta,
  FP32 recurrent state and BF16 core output, source gated RMS
  casts before gamma and at gated output. StateRoot schema
  variants disallow cross-dtype replay even for one allocation.
  Two-token four-layer synthetic BF16 whole-model integration
  plus source-independent NumPy BF16 bit-golden RMSNorm and
  gated RMSNorm, FLA two-step recurrent output/FP32 state
  oracle, sign/tie/overflows and F32 parity are required.
  This is explicit BF16 **diagnostics** only: NOT trained
  reference equivalence, a new model-run route or GPU/SSD.
  Source-independent NumPy BF16 goldens match exact bit patterns
  for ordinary/gated RMS and two-step GDN returned cores, with
  FP32 recurrent state tested within 2e-7. The full isolated
  scripts/check.sh gate passed 1,128 Rust test executions with
  0 failures and 69 intentional ignores across 238 suite runs,
  plus 18 Python fixtures, release installer/CLI and CPU numeric
  comparators. Rerunning on the final implementation still
  passes 1,128/0 across 238 suites. Both independent Qwen3.5
  .8B and 2B trained release routes retain complete 248320
  logits, all 42 StateRoot domains and source lease/mutation:
  direct16.26s and HF-import21.78s on the final source.
  Strict default/composed Clippy and Nix source-integrity pass;
  production model run is not enabled.

## Deferred — after Qwen/Gemma and Flash execution

- [ ] Qualify independently selected NanoJev-like calibrated decision
  model sources under the same model/source/cache/residency authority.
  Register finite choice/boolean/score or include/omit heads without
  assuming a specific research checkpoint. Validate calibrated
  probabilities and per-question source-bound state, batching, thresholds,
  accepted/rejected context records and decision-fallback semantics
  before enabling generic context-filter execution; pipeline check alone
  provides only typed static intent.

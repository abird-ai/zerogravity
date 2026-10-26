# Serving and speculation

Provenance commit: `6d74f76`.
Scope: P09/P12/P15/P16/P20/P22.
Contracts: [joint serving/goodput](../../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md),
[36 serving gates](../../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md),
[Qwen parity and decoding](../../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md).

## Uncommitted

### Signal-driven serving shutdown — 26 Oct

- [x] Install TERM/INT handling before source preparation; close admission,
  interrupt partial/backpressured sockets and original request receipts, join
  all HTTP handlers, then close/join the original CPU/GPU workers. Report handler/
  worker panic, new final GPU retirement failure or retained source readers as
  failure. Original source close/quarantine and physical publication are unchanged.
- [x] Actual model-free process/PID1 TERM+INT controls: exit0, no cleanup signals,
  no surviving members and serving ledger0. Actual approved cached Podman PID1
  stop also exit0/CLOSED/ledger0/no escalation; exact CID independently absent.
  Assembled full `scripts/check.sh` passes; CUDA/HIP/Vulkan feature compilation
  and model-free GPU retirement-failure control pass. No model/device execution
  or performance grant. [Coverage](../../runs/serving-shutdown-20261026/root/validation.json).
- [ ] Replay applicable original-model/backend shutdown and fault scopes under
  their existing source/device gates before a benchmark qualification claim.

### Current parity and post-parity optimization order — 23 Oct

- Current MAIN/ISO410: continuous34+dynamic2-key KV and compiler-lifetime7
  assembled; source/strict/ownership checks and optimized release5 pass. Actual
  finite B4/lifecycle/retirement/3fault and both-model N1 matrices/faults pass.
  Source6 shared input geometry fixes the old equal-frontier refusal; grouped
  strict/2401frontier checks and optimized release6 pass. Affected finite B4/
  lifecycle/zero-retirement/3fault cases pass. Genuine mixed N1→B4 reaches math
  but differs from same-prior N1 at SSA9/layer0 post-attention residual; earliest
  divergence was unresolved there. Fixture-only diagnostic7 now finds first
  difference at QKVZ2, with identical embedding/InputRMS and all48prior planes
  perrow. Subsequent original same-prior B4 comparison passed below; the old
  cross-width N1 diagnostic remains unchanged. All original owner/group/EOF
  closures verified. Native mixed/refill/fullstate and stock API evidence below
  are distinct from the separate genuine HTTP mixed/refill result below; fullgate
  remains open.
- Fixture-only source8 exported all15 genuine transitions/35events/1,648,529,280B
  including cancel/refill/permutation/N1 bridge. Healthy source/native retirement
  and final host/device0 verified; this is explicitly unqualified numerical data.
  Original same-prior reference3 PASS: all15transitions/30plain+observed calls,
  full42prior+candidate/fullV/24sums/natural bits/control exact, including first
  mixed B4/refill/N1bridge/permutation. All320loadedroles unchanged; final source/
  engine/process/container/group/EOF closure healthy. Explicit state-initialized
  comparison, not an independently generated original public trajectory. Earlier
  adapter failures and N1 cross-width diagnostic remain preserved. Fresh API8
  PASS24offers192tokens/provider/source/owner closure. Wave uses B4 prefill and
  cached B4; ragged remains N1. Separate actual event-driven HTTPv2 CLOSED0:
  5genuine N1prefills, A→cachedB4[50,48,48,48], completedA then replacementE in
  B4[51,51,51,48] through[54,54,54,51]; all5offers36natural tokens match actual
  same-cap original stock.5B4publications/20rows,21GPUcalls/36roots/peak4;
  original source/process/container/group/one-reap/fullEOF closes. Explicit
  A4+BCD/E8 correctness case; stock serial reference reused after v1's missing
  metadata run_id failure, no retag/performance/general cancellation grant.
  Quiet C1/C4 CLOSED0: ZG66.7151/17.5611 vsstock37.3407/49.0326tok/s; all32offers/
  256tokens and four original owners exact. Singlepass preliminary; C4 regresses,
  C8 held. Finite genuine HTTP proof now passed. Separate direct diagnostic4
  establishes79first-use trips consuming3.10167–3.10218s of its3.72236s instrumented
  wave;185total trips/1480spans, all8/64/clock/source/owners exact, later/second
  wave0compiler spans. No quiet-run fraction or new speed claim. Shared admitted
  cold readiness/context accounting is next, preserving original caps and real
  model/root owners; no fake warmup or unpriced cache. Fullgate
  remains last; latest accepted gate16 covers preceding316 only.
- Latest measured finiteB4 source400v4: all8actual calls/fullV/all42states/
  all24boundaries/24dual sums/32outputs and lifecycle PASS; submitted conv/lastFA/
  head unknown retention PASS. Matched API24offers192tokens/provider exact;
  wave8 uses16realB4 flights, ragged remains N1. Same-flight B4 record retirement
  correction now passes fresh actual healthy zero-retirement, lifecycle and all3
  unknown cuts. Fresh API24offers192tokens/provider/source/owner closures PASS.
  Quiet C1 ZG67.2592/stock37.1156tok/s; **C4 ZG11.7555/stock49.4752**, regression.
  All32offers256tokens exact; C8 held until corrected C4 superiority. First-use
  compile specialization is source-confirmed; no measured compiler fraction.
  Continuous33 canonical47 N1→mixedcached47..53/refill/cancel and dynamic2-key KV
  successor advance in isolated checks; actual continuous qualification pending.
  Wider prompts N1. Final fullgate pending, no accepted B4/performance promotion.
- Residual shared binding Sol8 integrated MAIN/ISO316selected paths; bounded40
  analyzed-site cache with real per-invocation provenance and original late
  physical/state authority. Retained witness/site metadata is charged; GPU551/
  provider+default strict PASS. Original release and both models' complete
  source/state/fault/API plus forced-streamed2B faults PASS. Fresh .8B C1/C4/C8
  ZG67.2018/63.8830/60.9713 vsstock37.6230/50.4311/61.6415tok/s
  (+78.62%/+26.67%/−1.09%); all48offers384tokens exact, six original source/
  owner/group/container/full-EOF closures healthy. Singlepass; C8 superiority
  remains unmet. Final isolated gate16 PASS1752Rust/0failed/89ignored/269suites/
  allscriptchecks;316sourcepaths/rootREADME unchanged. Native remains N1; genuine
  BF16 batching precedes further N1 microtuning.
- Previous prepared-recipe release1:314selected source paths; shared source/
  topology semantics with genuine per-invocation provenance, retained host
  charges and warm binding outside cacheMutex. Both models' complete actual
  source/state/fault/API and forced-streamed2B faults PASS. Quiet .8B C1/C4/C8
  ZG64.8742/63.5759/60.7789 vsstock37.9990/49.3562/62.3861tok/s
  (+70.73%/+28.81%/−2.58%);48offers384tokens exact, all6owner/source/group/
  container/EOF closures healthy. Singlepass; C8 superiority remains unmet.
  Full assembled gate15 LAST PASS1752Rust/0failed/89ignored/269suites/allchecks
  after four qualification-only27B example corrections;314sourcepaths/rootREADME
  unchanged. Native remains N1; true B4/B8 and wider parity remain open.
- Previous accepted source checkpoint: shared native/context ownership+Conv4,
  301selected paths; required release both models' complete physical/state/fault/
  API scopes and forced-streamed2B faults PASS. Full isolated gate14 LAST PASS
  1747Rust/0failed/88ignored/269suites/allscriptchecks; source/rootREADME unchanged.
  Its timings remain storage/rings release2. Actual x2 native3+transport12
  pass only their four-kernel/BF16-transport scopes. Canonical47B4 source discovery
  and immutable fullargument capture2 now accepted:7872complete native launches,
  4344records/1262868896B plain-observed exact/all24/all42/fullV/natural. Exact
  finite54tuple catalogue uses two independently healthy original FAimage
  variants; failedcapture1 stays unaccepted, no normalization/receipt retag.
  All original source/owner closures pass; this is expected/native metadata only.
  Genuine188/376-token prefill and same-root
  decode must be qualified before wider batching or performance claims.
- Previous frozen storage/rings release2,287 selected source paths: both actual
  301finite+40natural/fullV/all42states/all5fault matrices and matchedAPI24offers/
  192tokens perprovider exact. Forced-streamed2B headunknown/headcancel verifies
  actual ring lifetime/quarantine. Fresh .8B C1/C4/C8 ZG62.3484/60.7822/57.4249
  vsstock38.0862/49.9539/61.8017tok/s (+63.70%/+21.68%/−7.08%).2BC1
  ZG43.3738/stock37.6126tok/s (+15.32%); full resident head uses explicit native
  4608MiB vsstockrequest4096+KV512/no harddevicecap, host4096/device5120 unchanged.
  All original timing/source/group/EOF closures closed; singlepass/no confidence
  claim. C8 still needs genuine BF16 batching. Strict/core/GPU/provider checks
  pass; full isolated gate12 PASS1734Rust/0failed/88ignored/266suites/allscriptchecks,
  287selected sourcepaths/rootREADME unchanged. Astra handles architecture only;
  shared integration, model/native ownership and batching implementation use
  Sol HIGH in independent lanes. [Current measurements](../cpu-gpu-optimization/SHORT_HF_RESULTS.md).
- Previous frozenpair9: shared prepared keys/cold copy definitions/source metadata
  reuse,260MAIN/ISO sourcepaths, strict focused checks/release PASS. Original
 301edge+40natural/all5fault/matchedAPI checks exact, unchanged device counters/
  fault scalars, cold metadata +1400B within old bound. Quiet C1ZG61.5227/
  stock38.5419(+59.63%),C4ZG62.4174/stock49.6312(+25.76%),C8ZG62.5204/
  stock61.2769(+2.03%)tok/s. All48offers384tokens exact/all6 originalowners/
  source/EOFclosures reconciled. Singlepass only, small C8 margin unreplicated;
  no complete parity/stable-win claim. Full isolated gate11 PASS1724Rust/
  0failed/88ignored/266suites/allscriptchecks,260sourcepaths/rootREADME unchanged.
 2B actual301edges+40natural/all5faults/matchedAPI24offers192tokens each PASS.
  Quiet2BC1 ZG0.5298/stock37.0680tok/s (stock69.97x),16offers128tokens exact/
  both originalowner/source/EOFclosures reconciled. Further2B bands held for
  shared storage/streaming repair: source shows1.017GB head/970uploads+actual
  streamwaits perforward. No full attribution, fit or general parity claim.
- Latest owner sequence: shared-engine reuse first; superior matched .8B C1
  AND C4, thenC8; after allthree superior,2B. Qwen27B/Gemma integration stays
  parallel; their GPU qualification/benchmarks run on pvl-x2 ROCm/HIP. Vulkan
  parity with CUDA is also required. Backend availability grants no HF permit.
- Previous common196path optimization checkpoint: full isolated gate6b
  PASS1713Rust/0failed/88ignored/266suites and scripted checks; new frozenpair6
  real matrix40/all5faults and matchedAPI24offers192tokens perprovider exact,
  independent roots/active8/source/ownedclosure PASS. Quiet pair6 singlepass
  C1 ZG52.0304 vsstock37.1838tok/s(+39.93%), C4 51.6654 vs47.0443(+9.82%)
  opened C8: ZG52.6333 vsstock61.0984(-13.85%). All48offers384tokens exact,
  allsix original owners/full-source closures healthy. Actual peaks1/4/8,
  batchrowsmax1/zero fused rows; no repeated confidence/stable-tail claim.
  No2B promotion; remaining C8 gap is the next optimization unit. The frozen
  independent Gemma matrix resumed after C8 final closure under its original
  deadline. Backend admission changes are outside this frozen binary; latest
  source checkpoint follows below.
- Previous MAIN202source checkpoint adds the tested backend admission/CLI value-
  routing fixes; isolated gate7 PASS1715Rust/0failed/88ignored/266suites plus
  scriptchecks, exact source/rootREADME unchanged. No new timed binary or HIP/
  Vulkan physical support follows. C8 finite-check cohort successor and portable
  backend source/transport work continue in isolated worktrees.
- Latest accepted MAIN/ISO210selected changed-source unit: frozenpair7 actual
  301deviceedge+40natural calls/allfivefaults/matchedAPI24offers192tokens per
  provider PASS, fullsource/originalgroup/container/EOFclosure. All40 calls
  observe83finitecertificate enqueues vs327previous; exact math unchanged.
  Full isolated gate8b PASS1715Rust/0failed/88ignored/266suites/allscriptchecks;
  one test-module declaration formatting-only gate8failure preserved/fixed.
  Quiet matched .8B C1ZG55.0931/stock38.5359(+42.97%),C4ZG53.1955/stock49.3781
  (+7.73%),C8ZG53.4094/stock63.9834(-16.53%)tok/s. All48offers384tokens exact,
  all6 originalsource/ownerclosures healthy. Singlepass/no confidence claim;
  N1math/zero fusedrows and C8superiority unmet. No2B timing. Next source unit
  investigates shared submission/replay vs genuineHFmultirow execution.
  Newer reviewed portable matrix11 plus exact allocator-range correction are
  integrated MAIN/ISO after timed source closure. All12 actual HIP/Vulkan
  transport modes pass with independent ROOT owner/source/binary/EOF checks;
  no arithmetic/model/serving grant. Actor6 merged; frozen pair8 release and
  actual301+40/all5fault/API pass, every dispatch/finite/fault scalar equals
  pair7. Corrected assembled gate10 PASS1715Rust/0failed/88ignored/266suites/
  allscriptchecks, source/rootREADME unchanged. Quiet pair8 C1ZG60.8262/
  stock38.2050(+59.21%),C4ZG60.9697/stock50.6970(+20.26%),C8ZG60.4256/
  stock63.2563(−4.47%)tok/s, all48offers384tokens exact/all6 original owners/
  finalsource closure. ROOT reconciled actual records. Preliminary singlepass,
  C8superiority unmet/no2B promotion; current actor causal diagnostic next.
  The tiny portable6mode foundation passed separately; neither foundation nor
  transport compilation grants HF/backend/model/serving/performance parity.
- Expanded short-HF runtime converged into MAIN, preserving orderedSiLU/PTX
  work. Both actual native matrices PASS: .8B/2B ×47/57/62/67/72 ×8 natural
  calls, exact24 boundaries/42 state domains/fullV/feedback and independent
  roots, refusal/stale/cancel/survivor/final physical0 checks. Genuine67/72
  two-chunk prefill is covered; all327 independent observations remain open.
  Matched .8B HTTP preflight PASS: active8, both populations, all24 responses'
  eight IDs/text/usage/finish exact against stock. All five .8B cached fault
  scopes PASS for each model. Prior MAIN isolated checkpoint check.sh PASS:
  1531 Rust passes/0failed/81ignored/260suites plus18 namespace fixtures and
  complete scripted release/installer/lint/qualification checks. Matched2B HTTP
  now PASS all24 responses/full8 IDs/text/usage/finish/active8/owned closure.
  Owner23Oct reprioritized after quick .8B canonical C1/C4 baseline: optimize
  .8B to beat vLLM first; no2B timing, C8 canceled and full128arms deferred.
  Quick C1 output tok/s ZG0.9557 vsstock38.5021; C4 ZG0.9511 vsstock49.6408.
  All32 offers completed correctly, one warmed-process pass/configuration.
  No confidence claim. Timed owners retired; separate untimed profiling and
  admitted head-residency optimization now active, Qwen/Gemma parallel.
  Test durations are not serving measurements.
- Both new stock-only seq8 quality preflights pass. Stock2B sampled peak5443MiB
  exceeds native's5120MiB physical cap; original stock protocol does not enforce
  that hard device cap. Preserve resource asymmetry in results; no stock5GiB fit
  or equal-resource performance claim. No2B timed arm has executed.
- Previous finite checkpoint22Oct: original .8B canonical47+7 matches all24 boundaries,
  all42 state domains/fullV/actual feedback in the qualified native scope.
  Existing HTTP observes real active2, cancellation and healthy survivor.
  That accepted checkpoint converged MAIN/ROOT; fresh MAIN six SiLU/two original GGUF B2
  workers/canonical HTTP, strictClippy/fullgate and vendor104 guards PASS.
  Full all327 oracle/original GDN tuple, broader two-model/ragged active8 and
  matched performance remain open. [Current numerics](../numerics-and-context/TODO.md)
  supersedes dated partial-layer milestones below.
- Owner reconfirms binding/publication continuation, then matched vLLM timing
  after parity. Parallel recursive primitive/pipeline/profile design is authorized;
  no new framework blocks the current original-engine correctness units.
- [ ] Profile and fix the .8B gap first, then beat the same matched vLLM C1/C4
  baseline without changing source/quality/caps. Broader2B/C8/ragged/repeated
  timing deferred by explicit owner priority; no unnecessary baseline sweep.
  Resident-head candidate passes all40 actual numerical calls/unchanged limits
  and all five fault scopes after narrow verifier recognizer repair. Quiet
  host phase diagnostic8requests/64forwards exact: binding3.629112s/enqueue
  3.021396s, nestedHead0.005158s/zeroH2D/zero checkpoints; original owned closure
  clean. Follow-on host unit passes fresh40+all5faults/14metadata guards/API24
  per-provider exact; strict GPU Clippy PASS. Fresh unprofiled canonical wave8:
  C1 native9.7748 vsstock38.0061 tok/s (6.54748s vs1.68394s); C4 native9.7972
  vsstock49.1130 (6.53247s vs1.30312s). All32 offers/256tokens exact, actual
  peaks1/4, batchmax1, all original owners/source/EOF close. Preliminary one
  pass only; still3.89x/5.01x slower, no certified win. Combined CPU+phase
  diagnostic finds memcmp70.66% of532 user-cycle samples/0lost; actual process
  6.72user+0.66system over6.77396wall. Nested bind3.074109s/enqueue2.945825s,
  complete/readback0.149683s; Head64launches/zeroH2D/checkpoints. CPU fractions
  are not GPU or wall fractions. Investigate deep GDN static-source identity
  comparisons at their shared metadata owner; do not weaken equality/authority.
  Exact static-source equality now preserves all field/content semantics while
  bypassing same-pointer+length text scans. Metadata10/strictGPU Clippy/joint
  release PASS; frozen pair4 passes fresh actual matrix40/all5faults and matched
  API24/24 per provider, exact output/state/owner closure. Fresh unprofiled pair4:
  C1 ZG40.4089 vsstock37.0951 tok/s (1.58381s vs1.72530s), native+8.93%;
  C4 ZG39.2740 vsstock48.2272 (1.62958s vs1.32705s), native18.57% lower.
  All32/32 offers/256tokens exact, original owners/source/EOF closed. One
  preliminary pass, no confidence/tail/fullparity claim. Actual HF batchmax1;
  source-safe B4 integration is under separate investigation, not yet a causal
  percentage attribution. Subsequent pair5 residentembedding reuse passes
  fresh40calls/allfivefaults/matchedAPI24offers192tokens perprovider. Quiet
  C1 ZG48.9151 vsstock38.8841tok/s (+25.80%); C4 ZG49.9998 vsstock50.4721
  (-0.94%), all32offers/256tokens exact/allfour owners/sourceclosed. Native
 64roots=forwards=outputs/432steps perarm/peak1|4/batchmax1. Singlepass only,
  no C4win/confidence claim. Periodic146path MAIN↔ROOT convergence preserves
  parents/freshmtimes; assembled gate5 PASS1645Rust/0failed/86ignored/265suites
  plus complete scripted checks. Later common preparation/cache work requires
  its own assembled gate and fresh physical/API checks. Models parallel.
- [ ] Remaining .8B C4 optimization: source-safe HF physical coalescing/batching.
  Current CLI rejects native_rows, HFloaded batchNone/native_batchfalse and
  worker excludes originalHF; existing B2/B4/B8 BatchProgram requires GGUF F32.
  Reuse root-cohort validation/publication, not incompatible math admission.
  Unchanged N1 completioncohort needs genuine private lanes or registered
  cohort epochs/status taps; shared RMS/value/readback reuse must await actual
  completion. True BF16 B4 needs its own typed arithmetic/source/state lowering.
  Existing genuine47/57/62/67 B1 prefills then four distinct cached frontiers
  provide strong mixed-row/current-root oracles without another capture; test
  row permutations/all24boundaries/all42state/fullV/F32/feedback/fault closure.
  This is source planning; no B4 implementation or causal gap percentage claim.
- [ ] After source/state/natural-serving parity, execute established short
  C1/C2/C4/C8 comparisons with unchanged source/profile/EOS/caps and original
  lifecycle controls. Then apply measured substantial generic improvements and
  rerun those profiles before long contexts and dense27B. Research alone adds
  no throughput/TTFT/goodput result.
- [ ] After the already authorized matched benchmarks complete, implement
  graceful cancellation plus explicit forced/immediate cancellation with bounded
  escalation deadlines. Add hooks for detecting stalled progress, repeated model
  loops and runaway CPU work; allow the supervisor to break them cleanly. Define
  request/session/model scope, cooperative drain versus forced termination,
  timeout policy and observability. Release GPU/native/source resources only
  after confirmed completion/termination; keep unknown work fenced/quarantined.
  Cover disconnect, stuck execution, timeout/escalation and healthy-survivor
  recovery with real physical tests. Owner directive22Oct: defer this extension
  until after benchmarks; it must not delay the active parity/benchmark path.
- [ ] Compose stable leased rows/diff metadata, compact readout, qualified
  production graph regions, exact fusion and true temporal prefill inside the
  existing worker. Price extra graph pools/padding, output backpressure and
  prefill/decode fairness; unknown capture/launch completion cannot eager-fallback
  through a fenced service.
- [ ] Extend full-state branch verification before any proposer: prefix0..K,
  target correction, all recurrent/conv/KV/indexer/draft/RNG/grammar domains,
  EOS/stop and canceled/stale epochs. Compare n-gram/Recall/suffix, compatible
  trained MTP/block drafters and K=0 using useful complete-cycle work, not assumed
  acceptance or source kernel counts. Keep explicit sampler/distribution profiles.
- [ ] Retain profile/shape/layout/source/context-grant compatibility at every
  batch/prefix/graph/proposal boundary; dynamic policy chooses only already
  qualified compatible variants. The user-facing profile combines pipelines,
  while original roots, physical admission and actual completion authorize work.

Research/priorities: [mechanism comparison](../../research/engine-optimization-comparison-2026-10-21/README.md),
[optimization tasks](../cpu-gpu-optimization/TODO.md),
[recursive composition](../pipeline-composition/TODO.md).

### Batching and paged-state follow-up — 21 Oct design only

Implementation and acceptance detail: [batching/paged state](BATCHING_AND_PAGED_STATE.md).
Existing GGUF CUDA B2/B4/B8, refill and one-teacher-token prefix waves are
accepted only in their recorded scope; the inspected HTTP path remains
context512/active<=8. The following work is pending, not new qualification
or a change to the active correctness/benchmark/27B order.

- [ ] Integrate both accepted GGUF/reference and newly qualified vLLM numerical
  profiles through the same session/worker/HTTP path. Bind profile/phase/schema
  to compiled plans and cache compatibility; reject incompatible state reuse.
- [ ] Implement original-plan paged KV append and block-table QK/softmax/AV,
  preserving each profile's masks, positions, dtype/reductions and ownership.
  Qualify independent B1/B2/B4/B8, page boundaries and full logits/state;
  allocator/hybrid prefix/COW dependencies live in the
  [memory plan](../memory-and-streaming/TODO.md).
- [ ] Extend bounded refill/regroup into fair aged chunked prefill/decode
  scheduling; qualify actual multi-token temporal prefill separately from
  request batching. Admit any physical GPU overlap only after event/lease,
  cancellation/unknown-completion and independent-root tests.
- [ ] Run profile-matched batching/paging/prefix-hit/overlap ablations and genuine
  pinned-vLLM comparisons: established short C1/2/4/8 first, then qualified long
  contexts. Record quality/full-state acceptance, true ITL/TTFT tails, committed
  SLO-goodput, offered/rejected/canceled work, fairness, actual dispatch/copy
  counts and physical memory/fragmentation. Preserve old-profile regressions;
  code presence or fewer launches cannot close serving/performance parity.

### Resumed correctness-first continuation — 21 Oct

- Latest owner order: .8B/2B genuine vLLM whole-model/source/state parity,
  then benchmarks, then qwen:27b. Actual timing is held until parity closes.
- All4 original short-profile policy/quality preflights complete, including2B
  ragged. Original owners/cgroups/cleanup validate; IDs/text/usage still differ.
  Numerical equivalence and functional-quality parity remain unqualified.
- ROOT executed12 runner fault tests and5 aggregation fault tests: PASS.
  Timed C1/2/4/8 matrix remains unexecuted; no new concurrent speed,
  SLO-goodput or hardware-count claim. Existing87e1 GGUF profile inputs do not
  establish equivalence with the new HF CUDA correctness profile.
- Native MLP and activation leaves pass565248/247296BF16 words. Complete
  connectedlayer0 fivephases now PASS4071072BF16+2171024F32 exact with actual
  computedcache and separatecarriedresidual/sourceWunchanged/host0. Leaf
  cancel/pressure/unknown and4whole-blockfaults pass. Whole-model/
  all42state/serving parity remain open; timings held and27B not started.
- [ROOT checklist](../../runs/numerics-and-context-upstream-cuda-root-20261020/TODO.md).
  Rebind any future benchmark to its actual accepted source/profile/binary;
  preserve paused SiLU work. No new unit merged or committed.

- [ ] **P12 ordered SiLU integration:** private CUDA B1/B2/B4/B8 operator
  passes204 physical old-GPU/CPU differential cases,171562710 exact bit
  comparisons and five fresh failure/abandonment processes. All17 intermediate
  checkpoints, status, canaries and original lease/quarantine ownership pass.
  Independent source review and candidate CUDA/Vulkan/default strict checks
  plus two row-partition tests pass. Candidate library, qualifier, server,
  worker and HTTP release artifacts compile; the exact-source diagnostic
  file-transport unit passes. Paused at the owner's request before starting
  the prepared60-case physical matrix. Full-model integration remains unaccepted:
  complete PTX/loaded-code witnesses, model/StateRoot/serving regressions, both
  original release parity routes, isolated full gate and fresh timing/count
  comparisons remain pending. Existing truncated PTX dump filenames overwrite
  specializations; retain that failed evidence and qualify a collision-safe
  diagnostic before claiming complete instruction coverage.

- [ ] **Stock2B loader and matched benchmark:** genuine pinned stock2B
  V4 plain/observed loader and independent comparison now pass: all335
  stored tensors inventoried, exact15 MTP exclusions and320 backbone roles;
  both complete vocabularies,496640 native BF16 logit words, remain exact.
  Source/runtime and owned process/group/container/EOF cleanup pass; retain
  the original V3 observer refusal. Both model loaders are prerequisites,
  not cross-engine/full42/default-graph or throughput qualification.
  Actual0.8B eager HTTP policy passes for both providers: identical47 prompt
  IDs, original Rust tokenizer, effective greedy/EOS settings, observed
  cgroup12GiB/swap0/CPU4 caps and authenticated GPU process membership.
  Original owned cleanup closes. Podman unresolved HPIDs establish no
  ownership; every sampled GPU PID must match the numeric owned subset.
  Generated IDs differ under the declared activation/cast profiles. Actual
  0.8B full-output quality covers seven prompts; quality equivalence remains
  unqualified. First unprofiled eager/cold-process/solo comparison completes
  all eight ABBA/BAAB samples: four requests/32 raw tokens per provider.
  Median TTFT13.400s native versus1.561s stock; stock is faster here.
  Cold-process means first request after readiness, not storage-cache cold
  or complete startup latency. Both providers are internally stable, but
  cross-engine IDs/text/usage differ. One stock sample coalesces delivery;
  individual-token ITL and source-correct SLO-goodput remain unqualified.
  The2B eager/cold-process/solo run also completes all eight samples:
  four requests/32 tokens per provider, identical raw IDs/output hashes/
  usage/finish on that one prompt. Median TTFT14.540s native versus1.562s
  stock; all samples have individual token delivery, native ITL p95
  0.186–0.199s versus stock0.026–0.030s. Seven-prompt2B quality remains
  unqualified despite five matching full texts; this prompt establishes no
  numerical equivalence. Finish default-route, warm/concurrent workloads
  and physical counts;
  this bounded observation does not establish numerical or serving parity.

- [ ] **Production CUDA graph admission:** close every same-card constructor,
  transport, allocation, external event/storage and final-drop path through
  original queue/completion ownership; admit graph metadata/driver resources.
  Private graph replay does not enable production graph selection. Test-only
  host protocol14 cases/three ownership negatives pass. Separate original
  transport14 cases/two intended ownership failures, strict Clippy and no_std
  also pass in an isolated harness. Prepaid retirement and complete integration
  remain separate work, with participation unproven.

- [ ] **P12 prefix scheduling performance:** measure concurrent native prefix
  cold-process throughput/TTFT against the accepted max32-prefix implementation.
  Warm observations below cover both models plus explicit B1 scheduling cost;
  genuine equivalent-source vLLM comparison remains open.

- [ ] **P12 fused continuous GPU serving:** extend the accepted CUDA B2/B4/B8
  HTTP path with paged hybrid KV/recurrent cache, physically safe
  prefill/decode overlap and fairness. Unequal decode frontiers and
  compatible refill are accepted within the B7 source/context scope.
  Extend full-logit/StateRoot qualification to other advertised trained
  profiles/backends and full GPU prefill. Measure actual p50/p95 TTFT,
  true token ITL, committed SLO-goodput, fairness, memory and physical
  dispatch counts against genuine pinned vLLM V1 with equivalent
  source/quant/tokenizer/hardware, cold/warm conditions and caps.
  Local release smoke and fewer launches do not qualify performance.
  Separate native0.8B B2 CUPTI diagnostic records64328 actual kernels and
  64328 successful direct launch calls, zero dropped records and exact source
  counter delta; plain/observed complete outputs and owned retirement match.
  A fresh accepted-baseline0.8B canonical C1 diagnostic also validates all
  four native/stock plain/observed processes:236270 actual native kernels
  and direct launches equal the source delta, zero dropped records/errors;
  exact provider-internal outputs and original owned retirement pass.
  Stock public profiler records9318 kernel activities/8810 host launches,
  but its losslessness/all-process coverage remains unverified. Independent
  validators and ROOT/peer metadata reviews pass. These observations qualify
  the native windows only; complete matched hardware counts, profiler device
  allocation admission, source-correct goodput and profiled timing remain
  unqualified. Equivalent default-route/warm/concurrent vLLM timing stays open.

- [ ] Connect accepted core `ContextOwner`/`ContextReader` snapshot
  capability to the real router's authenticated session/turn frontier:
  source-scoped application/shared grants, per-application and per-tenant
  ownership, immutable complete context records, explicit ON/OFF context
  selection, short-lock optimistic source epochs and cancel/revoke
  invalidation. Recheck `require_fresh` at TaskCoordinator delivery and
  bind source/graph generations to model/media StateRoot publication.
  DecisionAdapter is a typed callback boundary, not trained NanoJev
  logits or actual ASR context support. Cross-session/host source sync,
  bounded CPU staging and QoS, media quality A/B and revocation/stream
  cancellation remain open.

- [ ] One router must snapshot each enabled, source-granted context
  stream at the declared turn/graph epoch before live ASR. Default only
  committed user/assistant session content; explicitly selected tool
  call/result, application-owned system/developer prompts and authorized
  shared/cross-user records are eligible. Strict source ownership and
  requested grants, not a blanket category blacklist, govern access.
  No same-turn LLM feedback loop or unauthorized sources/tenants, raw
  unowned KV pointers or tool-secret leaks. The ON/OFF toggle leaves
  typed graph topology stable; OFF performs no retrieval. Bound
  configurable 4KiB..1MiB plus physical admission, ordered filters,
  typed decision models, credit/backpressure, stale ACKs and source-real
  accuracy/consent tests before live mode.

- [ ] Make YAML/repeatable-flag compiled pipeline configs a first-class
  source-bound router input: many model IDs, many typed component DAGs,
  many explicit protocol/route bindings within ONE process; one model
  can serve standalone chat and ASR→diarization→LLM→TTS. Preserve
  direct one-model CLI execution with no config requirement. Current
  pipeline check is pure scaffolding; implement model residency,
  admission, TaskCoordinator execution, generation/hot reload and
  authenticated OpenAI/SSE/WebSocket/WebRTC transport, then enable
  pipeline run/serve only after source/physical tests.
- [ ] Distinguish model-trained MTP weights and Qwen Flash-Next
  learned n-gram embedding bank from runtime n-gram/suffix
  speculative decoding (no extra source required). Verify
  accept/reject and complete rollback/grammar RNG across
  QSA/GDN/MoE before any speed claim.

- [ ] Implement canonical TaskRequest/coordinator workers and CLI/SDK/API integration, model routing/residency lifecycle, explicit supported OpenAI/Anthropic interfaces, streaming/tools/reasoning/schema/sampling and truthful capabilities.
- [ ] Enforce separate API/tenant payload, queued-prefill/residency and physical active-step admission; bounded overload, cancellation, disconnect, stale epochs and output backpressure.
- [ ] Add real compatible continuous/grouped batches, aged chunked prefill, decode/media fairness and finite joint batch/draft/expert/kernel/placement plans; no per-token catalog/JSON/global-lock work.
- [ ] Finish Qwen basics before optimizations; enumerate the complete pinned llama.cpp/vLLM union, public and internal surfaces. Preserve 105 provisional groups/145 document seeds and CLI/API frontiers as unreviewed inputs, not exhaustive parity.
- [ ] Implement every applicable sampler/logits-processor/RNG/logprob/grammar/tool/stop/reasoning composition and its independently verified outputs.
- [ ] Qualify target-authoritative MTP, draft model, n-gram/suffix, EAGLE, DFlash/DFlash2, DSpark and related registered strategies; DSpark2 exact primary source/version remains unresolved.
- [ ] Test all accepted-prefix lengths, target-distribution correction, target/draft RNG/grammar ownership, full KV/recurrent/index/media rollback, physical branch admission and authentic cancellation/quarantine.
- [ ] Jointly benchmark material combinations, adaptive depth/backoff and all offered/rejected/canceled work: committed SLO-goodput, TTFT/ITL tails, fairness, memory and energy versus equivalent pinned vLLM/V1/Omni.
- [ ] Preserve zero untriaged frontier/disposition requirements before any full-parity claim; synthetic acceptance, fewer launches and competitor measurements do not establish local parity or speed.

## Committed

### `9999adb1a66d` — source-routed continuous BF16 cohort serving

Related source commits: `45570bdb3d7f` (serve-wave-planning), `e6b07efdbb2d` (serve-source-routing), `bc66207ce9c5` (facade-short-hf-loader), `08fc5935e124` (facade-short-hf-cohort-suite).
Aggregate source21 qualification applies to the assembled tree; no per-commit
build or general model/backend grant. Historical failures keep their source scope.

- [x] **MAIN convergence regression checkpoint,22Oct:** preserve orderedSiLU
  and collision-safe PTX dump while integrating the finite HF worker route.
  Fresh MAIN six SiLU modes/two original GGUF B2 workers/canonical HF47 HTTP,
  strictClippy/complete isolated checkgate and vendor104 guards PASS. Actual
  HTTP active2/cancel/survivor/400 and owned retirement are verified; wider
  HF two-model/ragged active8 and matched performance remain open.

#### Current parity and post-parity optimization order — 23 Oct

- [x] Execute quick .8B C1/C4 canonical overview on both CUDA engines; preliminary
  results above. C8 naturally retired before its cancellation lookup, retained
  separately; no2B timing started. All owners/groups/EOF/source closure complete.

#### Resumed correctness-first continuation — 21 Oct

- [x] **Finite original-HF cached worker route:** existing fair worker and CLI
  serve canonical .8B47+7 using real current-root/greedy feedback; generic bounded
  atomic-prefill scheduling charges47 within unchanged64-token waves. Actual
  active2 chat/rawSSE returns three complete8-ID chains, disconnect cancellation
  increments1 and fresh survivor succeeds; max9/wrongprompt400 preserve GPU
  counters and quiescent active0/queued0. Native source/state/normal retirement,
  submitted uncertainty/quarantine, independent reviews and complete isolated
  checkgate pass. HTTP foreground server uses owned SIGTERM/group/EOF cleanup,
  not observed in-process finalzero. Fixed source/prompt/active2 scope now in
  MAIN and ROOT; general two-model/C4/C8 and performance remain open. Broader
  short-HF parity precedes matched benchmarks; cancellation extension stays after.

### fb68d8f — exact state transport and certification

- [x] **Row-native state transport and scatter certification:** exact U32
  vector/KV transport and persistent finite/status certification preserve the
  original all42 preflight, leases, flight, Ready and atomic publication.
  All46 physical model/owner/private-graph/failure cases,10 operator/failure
  cases, complete isolated1251-test gate, strict checks and both approved
  original release parity routes pass. Fresh14 worker/12 HTTP regressions
  pass with exact frozen source/build and original owner/EOF retirement.
  Native-only unprofiled before/after runs complete32 fresh ABBA/BAAB
  processes,144/144 measured offers,1152 output tokens and32 ordinary B1
  warmups. Full IDs/text/usage/finish match exactly; all144 requests have
  individual-ID delivery/1008 actual intervals. C8 uses54 actual B8 cohorts
  per sample,432 independent steps/roots. Median raw token rates improve
  2.2506→2.28945/s(0.8B,+1.73%) and1.6268→1.6543/s(2B,+1.69%);
  C1 is nearly unchanged. Four samples/label, finite bursts, ordinary B1
  warmup and point memory samples establish no confidence-certified win,
  fully warm B8 scope, source-correct goodput or vLLM parity. Source dispatch
  counts292116→239952/284340→232176 are not hardware enumeration. All owned
  process/group/container/EOF retirement and independent ROOT/peer reviews pass.

### fff1842 — stock baseline prerequisites

- [x] **Genuine projector and stock0.8B loader prerequisite:** unchanged
  pinned llama converter produces source-audited BF16 visual projectors for
  both original models; independent complete tensor-byte checks pass.
  Stock pinned vLLM V1/official GGUF plugin loads original0.8B language-only
  weights; all320 backbone roles validate. Plain/observed B1 teacher runs
  preserve both full vocabularies,496640 native BF16 logit words, exactly.
  Native dtype and separate F32 observer representation are explicit; no
  sampler-cast, all42, cross-engine arithmetic or performance qualification.
  Owned process/group/container/EOF cleanup passes. Both projectors are
  genuine source conversions, not dummy parameters or a modified engine.

### 9a19901 — batched graph and baseline qualification

- [x] **Private full-model CUDA graph qualification:** both original models
  B2/4/8, StateOnly teacher prefixes (including genuine EOS input), future full
  logits and outside-body greedy sampling match independent CPU/B1/native
  full248320/all42 state. All six normal cases and18 fresh unknown-completion
  cases pass actual replay, stale/Ready/pressure refusal and reclamation or
  original quarantine. RoPE launch reduction additionally passes both models'
  170 ordered-angle cases and full native B2/4/8 normal/greedy comparisons.
  Physical32-case result87b85eff. Composed strict CUDA/Vulkan, complete
  isolated1243-test phase, both original release routes,14 worker and12 HTTP
  CUDA regressions pass; composed result e76248e8.

- [x] **Genuine GGUF baseline build prerequisite:** official out-of-tree
  vLLM GGUF plugin e2b8ad532b8b builds unchanged against pinned vLLM0.31/
  Torch2.13/CUDA12.9; wheel af9de736,129 RECORD entries/111 packaged Python
  files and full151-source pre/post pins validate. Owned container/group/EOF
  cleanup passes; no GPU/model ran. Stock text loading with a genuine visual
  projector and language-model-only mode, activation/cast/reduction profiles,
  numerical quality and matched benchmarks remain unqualified. Lack of an
  in-tree GGUF registry does not exclude this official plugin route.

- [x] **Bounded native-prefix warm comparison:** 0.8B strict B8 ABBA and 2B
  natural concurrency8 BAAB, each plus B1,16 serialized samples/72offers/
  72completed/576 output tokens. Exact full IDs/text/usage/finish and cleanup.
  Mean concurrent waves32.301205→18.33249s(0.8B,-43.2%) and
  42.19858→28.457115s(2B,-32.6%); actual forward dispatches
  1722640→302160(-82.5%) and1668496→294384(-82.4%). B1 means
  4.43954→4.42998s/5.60352→5.652635s, unchanged kernel counts.
  Both concurrent workloads still miss10s TTFT; no consistent ITL-tail gain,
  source-correct goodput unset. Original strict2B B8 attempt failed its
  declared scheduling guard and remains unqualified; fresh naturalC8 protocol
  permits actual2/4/8+B1, and both successful new samples happened to use54B8.
  Conditions/limits and observed memory in FUSED_GPU_BATCHING; no vLLM claim.

### f788929 — native teacher-prefix serving

- [x] **Bounded same-engine cold/warm comparison complete:** both original
  models, warm ABBA(0.8B)/BAAB(2B) and cold-process ABBA,16 serialized samples,
  128offers/128completed/1024 output tokens. Every sample has8 genuineB8
  cohorts, zero refusals/errors/discards, and identical complete IDs/text/usage/
  finish reason across providers. Provider4GiB/device5GiB on BOTH builds.
  Mean warm waves improve15.3%/6.1%; cold13.8%/6.0%. Actual dispatches fall
  16.9%/14.4%. No consistent ITL-tail improvement or10s TTFT-SLO success;
  observed goodput0/source-correct goodput unset. Exact values/conditions in
  FUSED_GPU_BATCHING. Initial equal4GiB attempt remains rejected evidence.
- [x] **P12 native teacher-prefix serving:** same full-model BatchProgram
  consumes one teacher token/request/wave with StateOnly readout, independent
  causal positions and original all-root publisher. Both modelsB2/4/8 pass
  per-teacher CPU+B1/all42/future full248320, trained EOS input, abandoned Ready,
  stale/invalid rows and final0. Two physical pressure and two fresh unknown
  completion cases pass; unknown retains1817/1712 handles and4GiB envelope,
  unchanged roots/all42 and fence, without recovery. All14 worker and12 HTTP
  scopes pass, including ragged prefix/tail partition, no prefix-generated IDs,
  Ready cancellation/regroup, output-credit fairness and final active/queue0.
  Three readout-phase health counters close against actual native bands and
  four sampler stages/greedy cohort. Complete isolated check1241 Rust/0failed/
 70ignored/248suites+18Python/release/installer/Nix; strict CUDA/Vulkan/qualifier
  and both approved original cache-direct/HF-import release parity routes pass.
  Two test-fixture/assertion failures and one missing-CUDA_PATH prelaunch
  refusal are preserved; no production arithmetic or acceptance limit changed.


### d0a6084 — source activation preparation and physical acceptance

- [x] **P12 block-local CUDA activation preparation:** one native32-lane
  workgroup preserves source ordered finite/max/scale/rounding/code/sum
  arithmetic for Q4_0/Q4_1/Q8_0/Q6_K projections. Original seven buffers,
  composite/status stages, leases, queue and completion remain authoritative.
  Cold checked bindings remove repeated warm handle/capability/alias setup;
  only explicit non-CUDA selection retains the old lowering. CUDA errors
  propagate. Projections use5 launches versus10 (Q6_K:8), before certification.
- [x] **Current physical acceptance:**272 preparation cases/all7 buffers/
  1,483,608 words exact; both original0.8B/2B complete B2/4/8 unequal-frontier
  graphs pass all248320 logits/all42 domains versus independent CPU AND B1,
  feedback/refill/abandoned Ready/stale packets/final reservation0. Two real
  pressure cases pass prelaunch refusal, B4 progress and B8 after release.
  Eight serving-worker and six HTTP cases pass both models B2/4/8, EOS,
  credit/cancel/disconnect/replay and final0. Four fresh graph/sampler unknown
  completions retain1824/1719 and1828/1723 handles, unchanged roots/all42,
 4GiB envelope, fence and no recovery. Complete isolated gate1218 Rust/0fail/
 69ignored/248 suites plus18Python/release/installer/Nix; strict CUDA/Vulkan/
  qualifier and both original release parity routes pass.
- [x] **Bounded same-engine0.8B warmed ABBA:**32 offers/32 completed/256
  output tokens across four fresh serialized samples; all eight requests/sample
  have individual token events and identical IDs/text/usage/finish reason.
  Each sample executes eight B8 cohorts with zero refusal/failure/cancel.
  Mean wave39.14206→33.163665s (-15.3%); dispatches2,072,360→1,722,640
  (-16.9%). Mean of per-sample client ITL p95:0.3911→0.3933s, no tail
  improvement established. Both miss the declared10s TTFT SLO; goodput0.
  Source-correct SLO goodput remains unset. This is ZG versus ZG only.
### 1b599df — serialized owned benchmark provider lifecycle

- [x] **P22 serialized owned provider lifecycle:** optional strict
  `--provider-lifecycle` specs launch one fresh foreground process per
  actual ABBA/BAAB sample, with literal loopback vacancy checks, bounded
  startup/shutdown/logs, exact model readiness, finite optional warmup and
  isolated receipts. `cold_process` means a fresh process without warmup;
  it does not mean evicted OS/storage/device caches. `warmed_process`
  adds one ordinary unmeasured request. Direct argv only; Podman binds
  the first fresh CID and stops only that owned container via the same
  local runtime prefix. Cleanup precedes evidence writes; receipt/log/
  reader/process/endpoint failures block the next sample. Environment
  names and redacted argv are recorded; private raw logs are capped/drained.
  Existing already-running-endpoint comparison remains available.
- [x] **Lifecycle acceptance:** source-free actual localhost ABBA/BAAB,
  exclusive ownership, warm/cold, startup/model/hang/warmup/measurement
  failures, TERM-to-KILL, CID overwrite/missing/stop failure and five
  persistent receipt-I/O failures passed; all owned readers/processes/
  endpoints retired before any next sample. Root independently passed
  comparator/benchmark self-tests and pycompile. Two real GPU providers
  also passed SERIAL startup, one warmup request and complete shutdown:
  accepted B7 ZG mixed-GGUF CUDA, then immutable pinned vLLM0.31.0-cu129
  original-HF BF16. Each returned GPU telemetry to baseline41MiB; live
  samples1059/4203MiB are advisory samples, not peaks or reservations.
  Those sources/quant/math are explicitly unmatched: this is lifecycle
  qualification ONLY, not a benchmark or numerical/performance parity.
  Complete isolated gate1202 Rust/0failed/69ignored/248 suites plus18Python,
  strict CUDA/Vulkan/qualifier Clippy and both approved release parity
  routes passed. Matched source-correct SLO goodput remains unset.


### dbb5f4f — unequal native decode frontiers and bounded token observations

- [x] **P12-B7 unequal-frontier native CUDA decode:** per-row positions,
  RoPE/KV append/masking and packed attention softmax lower
  the same full-model ForwardPlan. B2/B4/B8 retain independent original
  authorities, roots, packets, leases and all-row Ready publication.
  Native decode accepts unequal frontiers; prompt-tail grouping stays
  strict and other prefill uses B1. Actual original0.8B/2B engine tests
  passed all248320 logits/all42 domains versus independent CPU AND B1
  at mixed positions255/256/0/31/32/1/510/511, host/device feedback,
  B1 handoffs/refill, abandoned Ready, stale/foreign packets, capacity512,
  inactive AV poison/signed-zero/subnormal cases and final reservation0.
  This engine run preceded metadata/observability-only composition;
  fresh composed worker/HTTP/pressure/abort suites passed separately.
- [x] **P22 bounded publication/delivery observations:** opt-in
  `--enable-per-request-metrics` plus `return_token_ids:true`; bounded
  ASCII request IDs, at most512 non-EOS token records, exact-once IDs
  independent of UTF-8 emission. Observe commit immediately after original
  root publication, queue receive and socket-write completion separately;
  retain the original observation through output-credit parking. GPU
  producer frontiers and existing successful native publication sequence/
  width bind each row without creating a new authority. Standard vLLM
  scheduler-origin TTFT/queue fields stay unset; parsed-request timing is
  named separately. Tools validate IDs, clocks, usage, frontiers/cohorts;
  true client ITL requires one actual token per arrival. Grouped arrivals
  remain unresolved; source-correct SLO goodput remains unset.
- [x] **B7 physical composed serving acceptance:** eight release worker
  cases and six release HTTP cases passed both models at B2/4/8, exact
  CPU/B1 packets/all42 internally, CPU IDs/text/usage on wire, trained EOS,
  limits, Ready cancellation/regroup, capacity1 parking, disconnect,
  survivors/replay and final reservation/active/queued/failed0. New
  unequal-arrival cases offer the short request AFTER the long request's
  first actual ID and observe a shared native B2 publication at distinct
  frontiers. Each model completed5 native cohorts/10rows/935 native
  projection-dot launches/20 native sampler stages in that wire case.
  Two fresh real-pressure cases passed prelaunch B8 refusal without
  mutation, B4 progress under pressure and B8 after release. Four fresh
  full-graph/sampler unknown-completion probes retained all original
  handles in quarantine (1824/1719;1828/1723), unchanged prior roots/all42,
  refused old Ready/further work/other model, retained4GiB and no recovery.
  First abort invocation used a nonexistent package-named binary and
  launched no GPU work; only the four probes were rerun successfully.
- [x] **B7 verification:** complete isolated check1164 Rust passed/
  0failed/69ignored over245 suite runs plus18Python/release/installer/Nix;
  strict CPU/CUDA/Vulkan/qualification Clippy passed. Fresh source-free
  release CLI35/GPU58 passed; focused composed CPU/Core/Model/facade/
  serving185 passed/2ignored. Both approved original cache-direct and
  HF-import release routes passed full248320/all42 for both models.
  Independent launch audit reconciles the unequal0.8B HTTP counter exactly:
  26 B1 state-only graphs*5489 +6 B1 greedy*5510 +5 nativeB2*5728
  =204414 actual completed launches. This includes private certification
  and state/packet copies, excludes cold initialization/maintenance, and
  counts a native cohort once. No throughput claim follows from it.


### d9fa829 — native CUDA B4/B8 HTTP bands

- [x] **P12-B6 native CUDA B4/B8 serving:** finite unpadded8/4/2/1
  bands, live EOS/length/cancel/credit survivor regrouping and actual
  per-band Ready counters. Explicit `--batch-rows 2|4|8` (default2),
  CUDA/active>=rows checks before model/device; default interleaved.
  Checked physical capacity refusal is typed None BEFORE allocations or
  submission; only that result permits smaller bands. Errors/unknown
  completion retain original fence/quarantine. Real scratch leases prove
  cold/warm B8 refusal leaves queue/uploads/allocations/roots/all42/idle
  workspace untouched; B4 progresses under pressure, B8 after release,
  full248320 CPU/B1 logits/packets/all42 exact, final reservation0.
  Six same-process release worker cases (.8B/2B × B2/4/8) passed exact
  CPU/B1 state, unequal output limits, handoffs, actual after-Ready cancel,
  bounded capacity1 parking/resume and trained EOS. Initial combined run
  failed CUDA OOM after old-stream cached pools accumulated; cold-load
  cleanup now runs under original initialization/queue guards. Repetition
  physically released completed idle pools (2.51GB/2.13GB→0), all6 pass.
  Four release HTTP tests prove both models B2/B4/B8 JSON/SSE CPU text/
  usage, trained EOS, one terminal, active disconnect, survivors and replay;
  final active/queued/failed0. Source-free CLI26 and GPU53 passed;
  strict CPU/CUDA/Vulkan/qualifier Clippy and full isolated check1148/0/
  69ignored/245suites+18Python passed, both approved release cache-direct/
  HF-import routes exact full logits/all42. Initial direct HTTP invocation
  found a non-serve binary left by the full check; rebuilt explicit CUDA
  server and repeated all4 successfully. Same-frontier serving only;
  unequal-frontier/paged state, full-prefix native prefill and matched
  vLLM source-correct performance remain open.

### 6bfcdbe — bounded offered-work benchmark accounting

- [x] **P22 bounded benchmark accounting:** `serve-bench.py` records
  offered arrivals, client backlog/overload, HTTP acceptance/rejections,
  completion/timeout/cancel outcomes, tenant tails and the full offered
  denominator. SSE chunk intervals remain explicit proxies. Optional
  provider token traces require exact run/wire/provider IDs, complete
  ordered token IDs/timestamps and terminal usage equality; malformed or
  incomplete traces cannot qualify true ITL/SLO goodput. Raw loopback JSON
  snapshots remain provider claims. `serve-compare.py` executes actual
  alternating ABBA/BAAB orders, compares per-request text/usage/token IDs
  and keeps caller source declarations separate from independent parity.
  Both isolated self-tests/pycompile and full check.sh passed. Actual pinned
  vLLM0.31.0-cu129 V1 CUDA startup and four-request warm smoke passed with
  original HF0.8B; this is NOT a matched comparison. Common trained source
  arithmetic, tokenizer/generation policy, caps and real token observations
  remain open; qualified source-correct SLO goodput stays unset.

### 2677d8b — native CUDA B2 HTTP waves

- [x] **P12-B5 native CUDA B2 HTTP:** existing worker dispatches the
  original full-model native plan for compatible prompt tails/decode;
  unmatched rows and other prefill remain B1. Independent session/root/
  packet/physical leases and the all-row publisher remain authoritative.
  Present each committed packet once, retire EOS/length/canceled rows,
  park bounded output before feedback and retry discarded Ready cohorts
  without repeated text/usage/terminal. Actual original0.8B/2B RELEASE
  worker tests passed CPU/B1 compact ID/max/all42 parity, B1↔native
  handoffs, output limits2/5, real trained EOS, pre-feedback/after-Ready
  cancellation, capacity1 credit parking/recovery and final reservation0.
  Actual owned CUDA HTTP alias JSON/SSE tests passed exact CPU text/usage,
  trained EOS stop/EOS-excluded usage, one terminal/usage/Done, active
  disconnect, survivor/replay and final active/queued/failed0. Native
  completed cohorts9/8, projection launches1683/1496, sampler stages36/32;
  one Ready cohort discarded per model. Opt-in fused is CUDA B2 only,
  active≥2; default interleaved and non-CUDA refusals are preserved.
  Strict CPU/CUDA/Vulkan serve+cache and CUDA-qualifier Clippy passed;
  CPU serve14 and GPU53 tests passed (GPU3 intentionally ignored).
  Full isolated check.sh exit0:1147 Rust passed/0 failed/69 ignored over
  245 suite runs,18 Python, release/installer/Nix formatting. Both approved
  cache-direct/HF-import RELEASE routes passed full logits/all42 for both
  models. Real release serve-bench smoke4/4 requests16 usage tokens each,
  eight actual native cohorts each; comparator self-test passed. No true
  ITL/SLO/fairness or matched vLLM performance acceptance; vLLM absent.

### 54fd580 — native CUDA greedy/device feedback

- [x] **P12-B4 native CUDA greedy/device feedback:** same full-model
  batch ForwardPlan plus four native vocabulary-reduction stages,
  independent12-byte packets and one native device-input gather.
  Original packet producer is stamped only at all-row checked commit;
  exact source/session/epoch/root/frontier and physical lease bind
  consumption. Mutable public host ID/max fields are non-authoritative.
  Approved trained0.8B/2B mixed GGUF B2/B4/B8 RELEASE passed six
  source steps/four GPU-feedback steps per row: CPU/B1/native
  full-logit shadow all248320 logits and all42 domains exact;
  compact greedy ID/max bits/all42 domains exact,28/52/100 bytes
  per cohort, no warm value uploads and final reservation0.
  Swapped/duplicate/identical-ID foreign packets, isolated same-epoch
  old roots, restored epochs, invalid IDs, abandoned/reordered/stale
  proposals refuse without publication. Full-vocabulary synthetic
  ties/zero/subnormal/dense-bit/EOS winners and nonfinite/sticky-status
  refusals passed; trained EOS input is exact, HTTP EOS stop unqualified.
  Actual feedback launch/drop retained1827/1722 original handles,
  including input/output packets, all42 CPU prior domains and future
  KV zero tails exact; old Ready/further work/new service refused,
  original4GiB envelope fenced/charged after owners drop, no recovery.
  Both models repeated B2 imported-prefix valid1/256/257/512 boundary,
  exhausted-capacity and reclamation proof; no GPU full-prefix claim.
  GPU53 passed/0 failed/3 ignored; strict default/CUDA qualification
  and CUDA/Vulkan serve+cache Clippy passed. Final isolated check.sh
  exit0:1147 Rust passed/0 failed/69 ignored over241 suites,18 Python,
  releaseCLI/installer/Nixfmt; both approved cache-direct/HF-import
  release routes and benchmark/comparator self-tests passed.
  HTTP remains interleaved; no fused-serving/vLLM throughput acceptance.

### 35ebc5b — native full-model CUDA ForwardPlan

- [x] **P12-B3 native full-model CUDA ForwardPlan:** one traversal
  over B2/B4/B8, original shared source handles, admitted packed
  scratch and private per-session prior/candidate leases. One actual
  completion yields exact source/session/epoch/root-bound Ready
  receipts; existing all-row checked commit remains the publisher.
  Original trained0.8B/2B mixed GGUF profiles passed all248320
  logits/all42 domains per row versus independent CPU AND B1 over
  two decode steps, distinct histories, invalid/duplicate/foreign/
  reordered rows, abandoned proposals, stale epoch/root rollback,
  sibling continuation, no warm source/value uploads and reclamation.
  Release B2 for BOTH models passed CPU-hydrated prefix0/255/511
  decode at valid1/256/257/512, exact canonical KV tails, attention
  padding dispatch delta and capacity refusal. Actual launch/drop
  tests retained all1820/1715 original handles and the full4GiB
  envelope behind the existing quarantine; prior roots/all42 domains
  stayed exact, old Ready/next step/other model refused, no recovery.
  Source-free GPU51 passed/0 failed/3 ignored; strict default GPU,
  CUDA qualification and CUDA/Vulkan serve+cache Clippy passed.
  Frozen isolated check.sh exit0:1147 Rust passed/0 failed/69 ignored
  over241 suites,18 Python fixtures, release CLI/installer/Nixfmt.
  Both approved original0.8B/2B direct-cache and HF-import RELEASE
  full-logit/all42-domain parity routes passed; benchmark and
  comparator self-tests passed after private loopback setup.
  Same-frontier full-logit CUDA diagnostic only: no native row
  greedy/EOS, unequal-prefix/paged state, GPU full-prefix prefill,
  fused HTTP or vLLM performance acceptance. HTTP stays interleaved.

### 3ec431a — native quantized batch projections

- [x] **P12-B2 source-quantized native batch projections:**
  Q4_0, Q4_1, Q8_0 and Q6_K source-exact B2/B4/B8
  kernels call the SAME original B1 integer/F32 dot.
  Each physical RTX3060 CUDA test used one native B>1
  launch, bit-identical 128/256/512 F32 outputs versus
  original per-row B1 GPU reference. Original row-specific
  dynamic Q8 scales/offsets/codes, strict typed layouts,
  source-free bad-geometry tests and no CPU dot fallback.
  THESE are synthetic-source operator fixtures only.
  Strict source-free and CUDA Clippy pass; frozen gate
  1,147 Rust tests/0 failed over 241 suites, and trained
  Qwen3.5 .8B/2B direct-cache/HF-import source parity
  remains full logits/all42 StateRoot domains. Next:
  batch-native resident ForwardPlan integration and
  physical B2/4/8 trained full-model root parity.

### 08be9c6 — resident cohort publication

- [x] **P12-A1 source-bound resident GPU cohort:** ONE
  `ResidentModel` original-source method stages B2/B4/B8
  independently completed `ResidentReady` per-session proposals,
  checks all session RootAttempts and complete domain leases
  before one `ResidentService` fence-guarded publication.
  Invalid/foreign/abandoned/stale rows refuse all sibling
  root changes. Physical RTX3060 CUDA Qwen3.5 B2/B4/B8 matched
  full 248320 logits and all42 state domains per row against
  independent CPU; B4/B8 memory returned to baseline.
  Live OpenAI CUDA B2 short-prompt wave published two roots
  in one checked cohort; four separate GPU full-model forwards,
  six completed quanta, zero errors and `gpu_fused_batches=0`.
  Real batch-native ForwardPlan kernels remain pending.
  Acceptance: physical CUDA resident B2/B4/B8 exact all248320
  logits and all42 StateRoot domains per independent session,
  source/session state snapshots, abandoned/invalid/stale rollback,
  one source upload and lease reclamation; HTTP B2 cohort
  publication and active disconnect/survivor/replay pass.
  Strict CPU/CUDA/Vulkan feature Clippy -D warnings; full
  scripts/check.sh exit0: 1,147 Rust tests/0 failures/69 ignored
  across 241 suites plus release CLI/installer, 18 Python fixtures
  and Nix formatting; trained Qwen3.5 .8B/2B direct-cache and
  HF-import release parity both pass exact full logits/all42 domains.
  Separate offline Nix source-integrity cannot evaluate because
  pinned nixpkgs is absent from the local store and connector SSL
  CA path is unavailable. NOT a source-test failure.

### 9790fa2 — resident GPU OpenAI serving

- [x] **P09 GPU OpenAI serving / finite batch-wave scheduler:**
  opt-in CUDA/Vulkan/HIP resident model with independent physical
  StateRoots and device-side greedy feedback; no CPU inference
  fallback. Shared HTTP/SSE/greedy output, cancellation and
  original completion/quarantine. Bounded one-GPU worker,
  active/queue limits, source/phase/frontier wave grouping.
  0 fused GPU batches (no batch-native kernels yet).
  Typed GPU profile, CUDA/CCCL fail-fast preflight and reusable
  Nix split-SDK include view. Metrics separate GPU forward API calls (not kernel launches),
  successful quanta, attempted quanta, active/queued/canceled/error.
  Physical RTX3060 Qwen0.8B CUDA: 2 concurrent aliases under one
  model source, six completed quanta/four GPU forward API calls/zero errors,
  canceled active session retired safely and new session succeeded.
  SSE benchmark: 2/2 requests, 4 tokens, 20.02s debug wall,
  0.1998 token/s (NOT optimized/vLLM parity).

### 35b9093 — initial OpenAI CPU router and verified SSD extents

- [x] **First real OpenAI HTTP/CPU router milestone:** opt-in
  `zerogravity serve` with original Qwen3.5 GGUF sources, alias reuse
  (one resident immutable model/lease per canonical path), model list,
  health/admission, text and chat greedy completions and SSE usage.
  Bounded N CPU workers, independent source-specific StateRoots,
  round-robin 32-token prefill / single decode quanta, finite
  queue+active+connection admission, shared host ResourceArbiter
  weight and per-session state envelopes. Loopback-only with
  optional Bearer token; strict duplicate-key JSON and request
  framing, unsupported sampling/roles/media refusal, cancellation,
  SSE keepalives/deadlines and bounded backpressure. Real live 0.8B
  multi-alias + two concurrent text requests, authenticated health,
  host weight/state reservation release, unsupported parameter
  refusal and streaming passed. Full frozen gate:
  1,147 Rust passed/0 failed/69 ignored across 241 suites,
  18 Python fixtures and release/installer/CPU comparator checks.
  Both approved Qwen3.5 0.8B/2B original direct and HF-import
  release parity tests passed exact full 248320 logits/all 42
  StateRoot domains/two tokens/source lease and mutation.
  Strict default and composed serve+SSD/cache Clippy pass.
  The separate Nix offline source-integrity derivation is blocked:
  pinned nixpkgs 7a0f122 is not cached and the connector cannot
  fetch it with available SSL trust anchors; Nix formatting DID pass.
  Not fused GPU continuous batching or full OpenAI/vLLM API parity.

### 6d74f76

- [x] Shared immutable source, independent StateRoots, bounded fair physical workers and actual completion exist as foundations; production serving/parity/speculation is not complete.

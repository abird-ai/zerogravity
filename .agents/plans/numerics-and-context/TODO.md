# Numerics and context

Provenance commit: `6d74f76`.
Scope: P02/P11/P22, independently of optimization.
Contracts: [numerical/source profile](../../docs/NUMERICS_QUALITY_AND_BENCHMARKS.md),
[independent reference runbook](../../../tests/reference/llama_cpp/README.md).

Correctness-first continuation (20 Oct): [correctness fixes, arithmetic contracts,
three-model parity and subsequent vLLM performance plan](CORRECTNESS_FIRST_VLLM_PLAN_2026-10-20.md).
Owner resumed implementation: fix the lowest arithmetic, qualify each consumer,
then advance. The design alone closes no numerical or performance gate.
22 Oct owner directive: complete whole Initial47 end to end; then cached/current-root
feedback and worker/HTTP parity together, followed by short-context benchmarks,
long contexts and Qwen27B. Bindings/readers/helpers are internal steps, with focused
checks and one complete gate per larger unit; isolate only concrete failures.
Directive read in full: `.agents/runs/numerics-and-context-user-feedback-side-20261022/TODO.md`.

## Committed

### `cd4159d` — owned command streams

- [x] Shared reference/stop command owner retains intent, exact PID/PGID, one
  reap, separate bounded streams and EOF; return codes remain caller policy.
  DefaultSelector handles high descriptors without skipping owned cleanup.
- [x] Assembled isolated `scripts/check.sh`:2050 Rust PASS/0FAIL/91ignored,
  12 Python suites PASS, release/installer/shell/Nix PASS; natural0 and original
  group/reap/EOF closure. Strict CUDA/Vulkan all-target Clippy and284 CUDA/app
  host tests also pass. No physical model acceptance from these source checks.

### `5a18b5b` — capture scope ownership

- [x] Prepare model-owned prompt inventory and capture-owned cold roles/counts;
  both sources retain5prefills/40semantic calls/6physical scopes. B4 selects
  prefill47/cached1 by role; checked caches/storage/metadata preserve present
  charges. Native fresh48–50 remains unadmitted. Strict CUDA/Vulkan all-target
  Clippy passes after correcting baseline feature gates/private adapters/lints;
  all96 original GDN source tuples/12recipes/2catalogues remain exact. Whole
  Rust selector identity changed; no old binary qualification transfers.
- [x] Actual inventory4 and compiler-lifetime7 host controls PASS; uncertain
  destruction retains original source ownership. Current admitted shapes,
  resource charges and all96 CUDA tuples remain exact. Full gate above covers
  the assembled source; fresh native48–50 and device qualification remain open.

### `61f095c` — finite original prompt capture

- [x] Stage authenticated current-schema .8B/2B N1 bundles without changing
  resource bytes.28 reference controls and8 bundle controls pass. Failed packed47
  keeps six absent FA packets, empty norm archive and missing worker receipts;
  reviewed late-layout reader exports only authentic whole-image/formal facts
  for a fresh retry. Full packed/native model qualification remains open.
- [x] Shared original capture and bundle controls join the complete gate.
  Fresh47–50 original pairs passed independently; native/source/ABI acceptance
  still requires fresh full captures and per-shape implementation. Podman owns
  runtime isolation; source/build checks retain `native-gpu.py` isolation.

### `3612ecb` — production and test conventions

- [x] Placement is a rule/review responsibility; no additional convention tests.
  Runtime semantics and physical ownership keep meaningful behavioral coverage.

### `1f08e55` — rotary preparation and Rust test support

- [x] Replace ordinary text loading's rotary qualifier call and test counters
  with shared source initialization for the configured context. Retain the true
  context span; derive exact N1/B4 upload extent from real read-only ABI bindings.
  Preserve source/issuer/service/generation, admission and retirement. Strict
  CUDA/Vulkan all-target Clippy and affected CUDA/app/inventory host checks pass;
  complete assembled gate PASS0 (2050Rust/0fail/91ignored,12Python suites,
  release/installer/shell/Nix;202.101s/full original owner closure). Fresh
  physical validation remains open.
- [x] Move test-only GDN helper and cohort source reporter into Rust test support.
  Unused production constant removal changes the honest2B aggregate fingerprint
  from0938d435… to4d77cb66…; complete executable packets and GDN source stay exact.
## Uncommitted

### Ordinary source-derived continuation — 27 Oct

- [x] Qualification-only `.8B` fresh prefill 47/48/49/50 physically passes on
  RTX 3060: all24 boundaries, all42 post-state domains, full248320 logits,
  consumed priors, natural greedy248068, four independent roots, retirement and
  conservation. Commits `eb38dd5`..`e99c76e`; this grants no ordinary route.
- [x] Shared call-free `.8B`/`2B` `ForwardPlan` and retained authenticated source
  seal landed as `ad82feb`/`f7ff4f9`; exact24-layer/42-domain schedule,
  contexts512..32768 and chunks<=64. Legacy capture consumers explicitly refuse
  the new template.
- [x] Complete call-free parameterized GDN transport/state schedule landed as
  `fe228ab`/`52bfb51`: non-null slots1..4, explicit initial/final/cache state,
  alias guards and crate-private flight-owned launch preparation. Host/source
  gates pass; current-source numerical/device launch qualification remains open.
- [x] Resume the interrupted `29b491d` review and integrate only inert ordinary
  primitives. RMS/head-RMS overflow status, SiluUp word order, `byte_size()` cfg,
  alias rules, dynamic KV rows, RoPE theta, CUDA/NVRTC source portability and the
  previously untested feature matrix are independently closed. The reviewed
  series is integrated as `ea66b3a`/`68393e5`/`6b4a2ec`/`441fa64`; no backend or
  launch authority was added. See
  [the final review record](../../docs/REVIEW_ORDINARY_CANDIDATE_29B491D_2026-10-27.md).
- [ ] Finish source-derived non-GDN CUDA projection/attention lowering through
  the existing physical queue/flight/completion/quarantine owner and sole
  `StateRoot` publisher. Ordinary Qwen CUDA continues refusing before allocation.
- [ ] Requalify cached47→48→49→50, affected faults/cancellation/retirement and
  real-model shutdown on the assembled source; then `.8B` 512/1K/2K/4K/8K and
  16K/32K. Admit `2B` contexts only where the exact two-root resource plan fits.
- [ ] Run the final isolated `scripts/check.sh`, then the single serial matched
  campaign in
  `../serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md`. No 27 Oct
  performance cell has run.

### Reference capture and remaining runtime separation — 26 Oct

- [x] Four fresh packed47–50 original captures complete:408 same-request archives,
  31008 launches with complete argument fields, all24 layer boundaries/all42
  states/fullV/consumed priors/natural feedback and original owned cleanup.
  Earlier failed/incomplete parents remain failed. These are reference results.
- [x] Offline recipe emission from all three passing48–50 captures completes;
  81 added factories retain old packets/sites and exact source/ABI facts. Output
  remains qualification data; no production admission or native execution claim.
- [ ] Remove remaining shared rotary fault assertions/fixed-prompt qualifier
  dependency and case-driven model/GDN/native dispatch. Implement genuine
  parameterized kernels and explicit numerical configurations; do not move or
  rename test whitelists into runtime descriptors. Original CUDA scalar/source
  facts stay Rust fixtures. Parametric source proof/design is available only.

### Current owner order — 26 Oct

- [~] Sequential continuation after serving shutdown: fresh .8B prompt47/48/49/50
  originals, matching native profiles, real-model shutdown/faults, genuine long
  contexts/backend routes, then optimized matched C16 and27B. All23 approved
  model members freshly SHA-verified. New parameterized reference collector
  preserves original math/state/fullV/source validators and exact owner; source
  controls pass. Fresh47/48/49/50 original pairs now pass all24/all42, fullV,
  consumed priors and natural feedback;47 uses explicit separate postvalidation
  after its original controller report-size failure. Fresh complete packed
  v3 captures47–50 now pass; native comparisons remain pending. Native fresh48–50 remains
  unsupported until actual source/ABI/physical qualification.
- [x] Independently trace numerical-refusal retirement: D2H waits on the actual
  original stream; trusted comparison rejection precedes Ready and deliberately
  retains/fences the flight. No independent cleanup defect found; preserve the
  strict cross-batch failure and quarantine. Matching original/native profiles
  remain separate from optional batch invariance; no release or tolerance waiver.
- [~] Apply [all-model correctness policy](CORRECTNESS_POLICY.md), based on
  pinned vLLM/llama.cpp. Sol6.1 xhigh owns every lane; this supersedes older
  Pi/Astra assignments and performance-first continuations below.
- [ ] .8B initial47/48/49/50 and cached47→50: actual independent original and
  native outputs/state/physical closure. Original case5 strict FAIL101 remains;
  original N1/B4 QKVZ itself differs by six BF16 words at identical frontier47
  operands. The historical fifteen-transition archive lacks paired N1 frontier50;
  fresh47–50 original trajectories are now independently available. This explains
  the diagnostic, not native model acceptance.
- [ ] Genuine .8B512/1024/2048/4096/8192, then16K/32K context/state/feedback;
  corresponding original BF16 CPU/ROCm/Vulkan full-model routes are missing.
  GGUF, synthetic operators and finite other-model proofs stay separate. Local
  Renoir/gfx90c launch/recovery hold remains. Performance and27B follow this
  correctness sequence; earlier finite timing facts remain historical.

- [ ] **Next one short-HF unit, then matched benchmarks:** both approved .8B/2B
  sources, canonical/ragged47/57/62/67/72 plus genuine cached feedback, original
  active8/caps and existing worker. Stage generic bounded storage/liveness,
  checked runtime logical views/epochs, complete lowerer access/tap ownership and
  pooled primitive scratch; current metadata planner alone grants no alias/fit.
  Preserve old source/precision/native gates. Benchmark the whole qualified short
  matrix together, then extend long contexts; unrelated27B/Gemma continue.

- [ ] Actualcanonical47projectiondiscovery1 completed original PLAIN+OBSERVED/source/ownedcleanup, but exactcommon state.recurrent.after call0 differs. NewOBS cumsum/KKT/H/output selections differacceptedreference1; selectionvariance is verified, specificcause unproved. Independent actual metadata identity/layout gate now passes separately; no fullmodel/packedargs/native/perf acceptance.
- [ ] Qualify actual canonical short47input/max8output/EOS248046 and ragged57/62/67/72 prompts; exact source/profile/numerical schedule and original all24/all42/fullV/actual sampler. Extend genuine rotary/ABI/control coverage as required, then original worker/HTTP and C1/2/4/8. Old GGUF diagnostic shapes do not grant HF parity.
- [ ] Distinct native2B source opening/2048wide ordinary+dualRMS/projection+6144MLP/head/970window tied table+4096B source-row actor. Actual2B prefill PTX differs in cumsum, scaled-dot KKT, recompute W/U and chunk output; matching GDN geometry and different cache keys are not an execution permit. Preserve actual reference3/source/relative5GiB boundary; derive physical fit, then same fullmodel/natural/fault/serving closure. Matched short benchmarks, long contexts and dense27B follow actual parity. End-to-end benchmarks remain unexecuted.
- [ ] Investigate actual2B discovery2 PLAIN outlier: both lanes finish/source/ownedcleanup pass, OBSERVED matches acceptedreference3, but PLAIN first differs at call2/layer8 recurrent state and branch/residual, then consumed downstream/fullV. Conv.after is only its last3 cache rows; no proof all upstream conv outputs match. Ordinary layer0 RMS10boundaries/565248B match independently. Fullmodel discovery remains unaccepted; source audit confirms timing-based autotuning but no specific cause; unchanged PLAINrepeat1 completed with exact reproduction. Independent RMS leaf reference accepted after complete source/IR/byte gates (565248B boundaries/sourceembedding,4096B gamma); native2048 component physical comparisons/faults/fullgate20 now pass separately, explicit parent-failure boundary retained. Unchanged PLAINrepeat1 reproduces discovery2PLAIN exactly across113713152all42+11168652common+6117576final+565248RMS B, still548/21/3different records versusOBSERVED/reference3. Python-only actual selection2 diagnostic now completes18records/lane and strict source/common/final/prior/all42/RMS gates PASS despite four differing selected stageconfigs. Different configuration alone is insufficient; the earlier reproducible outlier remains unresolved.
### Binding continuation — 21 Oct 09:30 UTC

- [ ] Native continuation: genuine47 current-root/position/provider/cache coverage, cached2B histories and whole-model admission remain open. Isolated2B QKVZ/BA phase51 and actual six-node source prefix/GDN phase52 are accepted below; initial1/64 stops before GdnOut. Original full24 permit stays guarded.
- [ ] Current original2B KKT causal leaf1:29frozenchildren/23pure guards, actual original source/ownedcleanup and ordinaryRMS565248B pass. Four KKT configs/two repeats produce bit-identical A/F32+Ai/BF16 (all72comparisons0), but every solve uses the SAME2warps/4stages selected kernel: this does not compare solve variants. Strict original model reference still FAILS; current run reproduces prior PLAIN outlier byte-for-byte131564628B/27files. No tuner override or failure waiver; KKT config alone does not explain this case. Physical allocator peak remains unknown.
- [ ] Canonical47 failure narrowed in parallel: three separate PLAIN runs share the same all42 record differences; their OBSERVED lanes match acceptedreference1. Original late-H pair1 now reproduces oldbad PLAIN/good OBSERVED actual state bytes under healthy source/owned cleanup; original raw comparison remains FAIL. Inputs to H already differ: W59BF16words, U9BF16words (token34/head4); K/G/H0/history exact, Vnew=U, finalHT1002F32words differ. Later failed-pair solve-counterfactual1 directly captures equal operands/F32A and first solve.Ai differences, below. This earlier late-H capture alone did not localize the cause.
- [ ] Post-recompute47 investigation: attempt1 observer refused copied callable-grid assumption; original caller uses literal(1,16), failed attempt/cleanup retained. Narrowv2/28guards/peer review +48files/33parents/21runtime/3historypins PASS; actual attempt2 source/owned cleanup and original pair/common/fullV/all42 gates PASS. Seven actual K/V/beta/G/Ai/W/U records872320B/lane all equal good35; neither lane reproduces badW/U, so no historical cause/fix. Same-input original solve/recompute counterfactual proposed to test the actual observed warp2/warp4 solve selections without production tuner mutation.
- [ ] Repair47 exact arithmetic binding at existing provider/preparation seam; authenticate declared candidate and actual original code/ABI/launch histories, qualify finite owned solve variants, then connected full47+cached histories. Broad geometry/source equality cannot establish byte equivalence across two proven unequal solve implementations. No tuner forcing, expected-word selection, tolerance relaxation or failed-parent waiver; all18 GDN layer/eight-call identity coverage and fullmodel gates remain open.

- [ ] Retired2B postAttention1 parent gate remains CLOSED: bothlanes healthy/newlayer0pairedboundaries/source/native14op PASS. Fresh L8 pair3 source/lifetimes/full37paired archives verified after22observer+5host-schema guards: dual/RMS/QKVZ/BA/fullConv/preparedQKV/coefficients/cumulativeG/KKT A exact; FIRST solveAi14BF16differences, W96/U17/H2048. Actualsolve configs2warps/3stages versus4/4 and fullPTX differ; currentHT hashes reproduce prior same-lane split. Its original fullV/all42 failures remain fatal, RMSPASS. Separate new original MLP/ACT reference success and fixedOBS-only component successor do not accept this failedparent. Retired same-input originalsolve/cross-recompute leaf3 PASS: same actual794640B inputs; solve2warps/3stages reproducesPLAIN and4/4 reproducesOBS, Ai14/W96/U17 word split exact under BOTH recompute8/3 and8/4, twice; all12 returned completePTX/cachehashes equal actualoriginal variants. This isolates that split to solve configuration; it does not repair the original parent or prove earlier uncaptured inputs. Previous72 comparisons did not vary solve configuration.
- [ ] Next: all24/all42/fullV/feedback/cached47 histories and workerHTTP parity. New original2B MLP+activationcapture1 passes strict original common/fullV/prior/all42/RMS/postAttention/MLP reference gates and captures two6144activation packets; ROOTpostvalidation/native2B path/fullmodel remainpending. Earlier failed postAttention parents remainfailed; predeclared fixedOBSpost1 successor target unchanged. MatchedC1/2/4/8 benchmarks follow parity, then longcontexts and27B.
- [ ] Post-parity overhead follow-up: consider immutable GdnArithmetic same-object pointer shortcut before structural fallback; source-valid review only, no measured speedup or code change.

### Resumed continuation — 21 Oct

- Latest owner order: genuine .8B/2B vLLM source/state/full-model parity,
  then benchmarks, then qwen:27b. Timings held; layer0 cuts are component proofs.
  [Current ROOT checklist](../../runs/numerics-and-context-upstream-cuda-root-20261020/TODO.md).
- Earlier captured-image unit: combined CUDA/Vulkan/portable strict Clippy
  passes;20 focused GPU host tests and4 pure vendor captured-image tests pass.
  All15 original source-weight projection leaves compare638112 BF16 words
  exactly across19 native launches; captured operands are explicit leaf inputs.
- Connected five-call GDN run2 PASS3117216BF16+2100368F32 exact,15 native
  projection calls/19 kernels+50RMSstages+48GDNlaunches. Source-only operands,
  computed independent cache continuations, unchanged sourceW, host0 and original
  process/group/reap/EOF cleanup. This proves the GDN block, not decoderMLP/model.
  Failed run1 retained: helper512MiB host cap too small; derived peak533.94MiB
  from authenticated1MiB row windows. Caller-only576MiB cap; GPU caps unchanged.
- Original post-GDN oracle1 both owned captures complete and clean. Validator1
  wrongly assumed `_to_copy`; preserved failure. Separate validator2 and8fault
  tests PASS actual14 `aten.to.dtype`/arithmetic ops. Full153 prior records/
  25489440B unchanged; newplain/observed15692292B exact. ActualF32sum has30389
  nonzero-low16 words; BF16 outputs independently captured. Intermediate casts
  retain metadata only. ActualMLP10publicGemmEx calls observed; subsequent
  native ABI and leaf comparisons are recorded below.
- Connected fused residual/RMS run2 PASS3258528BF16+2171024F32 exact across
  all5calls, including actualunroundedF32sum and bothBF16outputs; sourceW
  unchanged, computed inputs/cache, host0 and original owned cleanup.28focused
  hosttests pass including7newstructural checks. Run1 retained: ROOT helper
  stale mode guard refused before arithmetic; helper13 fixes that guard.
  No full decoderMLP/model/serving qualification from this component proof.
- NativeMLP discovery1 and argumentcapture1+offlinevalidator3 pass10public
  GEMMs/13launches/6reflectedlayouts/180events,166resources/191138520B and all
  originalclosures/cleanup. Retain validator1/2 failures: valid unlaunched zero
  occupancy probes; two previously unclassified workspace clears. Newv3+9fault
  tests proves ordered32/64B zero-writes atworkspace0.8occupancy/control fault
  tests pass. Exact packed arguments and seven Rust templates now match all
  original assets. Actual standalone MLP run1 PASS565248BF16 words/10calls/
  13kernels/two real native32/64B clears; original W/input unchanged, host0
  and original owned cleanup. Captured inputs are explicitly leaf operands.
- Reserved typed residual/RMS+source-row metadata integrated:9coretests PASS;
  CPU/default/GPU execution stillUnsupported. Newfaultsource strict combined
  Clippy4 PASS and31focused GPU hosttests PASS. Actual earlycancel adds no CUDA
  driver/admission; pressure refuses beforebuffers/H2D/native and endsunfenced0.
  Unknown/latecancel each retain actual5completionowners/236handles/30readbacks,
  source/fusedprivateowner/native64MiB; pending1/fenced128MiB,0Ready/compares.
  Ten fused stages submitted forunknown,0forlatecancel; no completion/final0.
  Freshpositive postnorm3 undercurrenthelper15 passesall5fullcounts/host0.
- Required isolated full scripts/check.sh gate2 PASS1288Rust/0fail/76ignored
  across257invocations;18Python/release/installer/shell/Nix gates PASS. This
  covers core/postnorm/fault source, before new MLP/activation/connected code.
- Actual activation discovery/arguments1 PASS5launches/65events/one32BABI,
  source320/fullV/activecache/plain-observed closure and owned cleanup. Eight
  recorded-ABI faulttests and exact static assets PASS. Selected resource901664B
  is separately pinned; no loader-input/carrier/derivation equality claim.
  Native activation leaf1 PASS247296BF16 words/5calls/5kernels, input unchanged,
  host0 and original buffer/event/envelope/process cleanup.
- Current new source: vendor clear+activation hostcopy13 PASS18puretests;
  strict combined CUDA/Vulkan/portable Clippy6 PASS. Connected layer0 MLP code
  now reuses one set/5flights/computed cache and producer leases; distinct closed
  7image/12function/21record policy preserves old6/8/19 default. New144MiB
  provider/640MiB helper scope is source-budgeted, not a hard driver/RSS bound.
  Actualconnectedlayer0MLP1 PASS4071072BF16+2171024F32/5flights/37native/
  50RMS/50fused/48GDN/10MLP/5ACT/2realclears; sourcecomputed operands/cache,
  sourceW unchanged, separatecarriedresidual, host0/originalownedcleanup.
  Activation+MLPclear cancel/pressure/unknown sixfreshprocesses PASS: early
  cancel noCUDA/newadmission; pressure beforeH2D/native+host0; unknown retains
  actualimage/program/source/leases/native64MiB, pending1/fenced128MiB and
  externalhost112MiB or216MiB+192B;0Ready/readbacks/compared/no final0.
  Connectedwholeblock4faults nowPASS: earlycancelnoCUDA/host0;144MiB pressure
  beforeH2D/native+host0; latecancelafterGateUp+ACT versusunknownafterDown
  native6/8, both retain253handles/8owners/7images/21records/native64MiB/
  pending1/fenced144MiB/382041480Bhost. Active3MLPbuffers/oneCallowner stay;
  prepared15 is reported separately.0Ready/compared/no final0/sync/unload.
  Freshconnectedpositive2 andoldpostnormpositive4/unknown2 regressionsPASS.
  StrictcombinedClippy8 PASS;61componenttests and fullgate3 PASS:1288Rust/0fail/76ignored plus18Python/release/installer/shell/Nix.
  Combined strictClippy7 PASS; vendorhostcopy14 PASS40puretests including7new
  policy cases and original ownership/clear/cache guards. Firstattention
  reference capture1 PASS27095520B paired bytes, source320/fullV preserved,
  actual5eachFA2/fusedQK/KVupdate/implementation calls, layers1→2→3 branch/
  residual/state/KV continuity. Reference only; no replay/model permission.
  No whole-model/default/serving qualification.
- Actual captured-Out2 abandonment1 passes original retention/fence/pending1:
  source/program/image/native64MiB retained,0Ready/compared words. Process exit
  is not physical completion or final0 proof. MLPcollector build1 failed missing
  CPUparser dependency; retained. Freshbuild2 strictDLL51968B/13SHA+parser/scope
  hostguards passes; bothbuilds CPU-only with originalcleanup, no runtime proof.
- Prior full gate1:1279Rust/0fail/76ignored/257invocations plus18Python and
  release/installer/shell/Nix PASS; superseded by gate2 above. Repeat the complete
  gate after the current connected/fault source phase.
- All4 original short-profile untimed policy/quality preflights complete; wire
  outputs differ. No numerical/functional parity or accepted timing result.
- Bounded768/1024 helper release build and2 host tests pass; long GPU checks
  remain unexecuted. Production capacities256/512 unchanged.2048/4096 tiling
  is staged only;8K/default bulk/whole-model vLLM parity remain open.
- New unit is isolated; MAIN source composition, full gate and physical model/
  state/cancellation qualification remain required. Root README and paused SiLU
  work preserved. No commit/push/download/recovery authorization added.

- Subsequent ISO typed-native provenance/physical BF16 storage unit:127core and
  20resident host tests PASS; strict combined CUDA/Vulkan Clippy10 PASS.
  Same-original-window model metadata/source-owner unit:51host tests PASS,
  1trained-model integration ignored. These are metadata/ownership evidence;
  both real F32 source/state/abort regressions now PASS; full gate6 PASS1301Rust/0fail/76ignored plus18Python/release/installer/shell/Nix.
  Typed GDN/session lowering was subsequently linked in ISO and physically
  checked below; ROOT caught its ordinal-vs-actual-layer domain mapping error
  before execution.
- New storage unit physical closure in ISO: .8B/2B fullV/all42state/trace,
  restore/staleReady/greedy/concurrent roots/lastsnapshot→final0 PASS. Both actual
  aborts retain1624/1519 originalhandles and4GiBfenced envelope respectively;
  nonzeroprior/canonicalfutureKV/root exact, noReady/recovery/final0. H17 stale
  suffix-count expectation from earlier CUDAfusion was corrected at its test
  source (actual12bothCUDA, legacy17/15 retained); no numerical assertion relaxed.
  MAIN composition and new typed BF16 GDN execution remain pending.
- Actual first-attention driver discovery2 PASS25scopes/30launches/13layouts;
  45typed pointer queries/5dynamic-shared setups/3output copies, no unknown
  operations/refusals. Both original processes/cleanup and27095520Bpaired
  reference closure pass. Native argument capture1 preserves both originalprocesses/cleanup/paired
  closure but genericsemanticvalidation refuses newKVowner/29extra pointers.
  Offline QK/KV6tests andFA4tests PASS. Fresh semantic-v3 validation1 and7pure
  guards PASS all actual sourcefields/29private pointers/3copies with no typed
  refusals; original owner/context/image/reference guards retained. Native
  arithmetic/replay/model/performance remain unqualified.
- Fresh bounded availability audit: pinned original .8B/2B/27B HF sources all
  present; held regular FDs/extents, headers/index/EOF and small-sidecar SHA
  checks PASS. Weight payload SHA values were not freshly rescanned. Queued27B
  is Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, managed
  snapshot98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e;
  prior not-downloaded prose is stale. No acquisition gap; actual2B/27B native
  shapes and full42/112-domain execution remain open. Order stays parity,
  accepted benchmarks, then27B; no new downloads.


- Same-engine initial typedGDN1/64 physically PASS:20512/1312768BF16 +
  280576canonicalF32 each, exact all42prior/40carry/sourceW/status/final0,
  source-computed operands; frontier0/logitsNone/no model publication.
  Pre-cancel/pressure/unknown/true latecancel PASS; actual148leases/96state
  buffers/112readbacks retained,192MiBfenced,0Ready/compare. Helper supplied
  wrong six-image set in first failed attempt; corrected at caller to exact2/3
  and preserved512MiBcap. StrictClippy/defaultVulkan/purecohorts PASS; fullgate7
  PASS1307Rust/0fail/76ignored plus18Python/defaultrelease/installer/shell/Nix.
  Currentoldconnectedfivephase regression PASS4071072BF16+2171024F32,host0.
- Reviewed wholelayer0/prefix0..2 linked ISO only;136core tests,76component
  tests, strict combinedClippy3 and fullgate9PASS. Actual1/64-token complete
  layer0/prefix3, all12faults, source six-recipe regression and oldGDN/MLP
  checksPASS; exact scopes above and in ROOT checklist. No partial-prefix
  publication establishes a token-advanced model RootBinding.
- Actual final-head reference1 FAILpreserved at observer selected-row check;
  original cleanupclean. Corrected actualsample observer2 physicallyPASS:
 14finalnorm ops, unroundedF32sum, tiedBF16head/fullV/newV1forcedsampler,
  8025268pairedfinalB plus all153old25489440B unchanged. Originalbothowned
  groups/container/reap/EOFclean. Subsequent head5native discovery1PASS;
 160Bgemv2T fullpacked fields/nativearithmetic stillpending. Forced353fixture
  evidence leaves naturalgreedy/fullmodel/serving/benchmark gates OPEN.

### Bottom-up F32 repair — 20 Oct

- [ ] Qualify original-row Q4_0_8x8 QKV/Z using existing source projection math
  against native fields6/38 before changing any equation. Native operands are
  CPU_REPACK Q4_0, distinct from Q8_0/Zen BLAS; actual inner Q4 dispatch is
  source-derived, not freshly instrumented. QKV M6144 needs separate closed
  geometry; do not raise tinyBLAS limits. Then compose existing convolution,
  SiLU and grouped Q/K normalization with retained raw6144×3 history.
- [ ] Nonzero cached-prior, live producer, full temporal/default-bulk route/
  profile/state binding and default8K qualification. Retained native bulk64
  versus AR1 upstream inputs match2097152 words plus1572864 four-tap operand
  words; that comparison is not a Rust producer proof. Current
  Q/K/V/prior/Z remain explicit observations; gain lacks separate capture.
  Sliced CPU proofs confer no complete-model/vLLM or whole-host RAM qualification.
- [ ] MAIN full formatting still flags only the preserved SiLU draft
  `resident_program/silu.rs:91`; bytes match its pre-existing checkpoint.
  Do not alter another owner's draft.

### Remaining qualification

- [ ] **vLLM parity remains open:** implement and physically qualify its own
  CUDA arithmetic/cast/layout profile; complete initial/continued prefill and
  cached-decode GDN under that profile; close full .8B/2B/27B logits/state and
  long-context gates, then matched default/warm/concurrent serving performance.
  Current GGUF CPU operator proofs and historical cold-eager TTFT observations
  establish no completion percentage, cross-engine equivalence or new speedup.

- [ ] **Required three-model correctness gate (owner,20 Oct):** resolve
  independent original-source CPU8K GDN divergence for BOTH original
  Qwen3.5 models without loosening tolerances or changing declared source
  arithmetic. This is required alongside original Qwen3.8-27B correctness,
  matched serving/model benchmarks and shared-engine performance work.
  Short-context CPU/B1/native exactness and closed original-HF operator
  proofs do not close this independent GGUF long-context issue. Recover
  exact raw teacher prefixes and pinned native arithmetic/chunk settings,
  locate the first state/stage discrepancy, then rerun independent long
  context and the accepted short corpus after any implementation fix.
- [ ] Extend genuine resident GPU context/corpus qualification beyond the currently accepted 512 consumed tokens, including capacity/window/padding/replay boundaries.
- [ ] Run genuine whole-model 16K/32K CPU/GPU gates only after prerequisites; functional 8K success is not independent long-context numerical closure.
- [ ] Keep internal full-logit/all-state/trace bit parity distinct from independent native-source stage/top20 comparisons, including the known 2B 56 signed-zero distinctions.
- [ ] Broaden prompts/tokenizers/precision profiles and sampling quality only with independent oracles, exact history and explicit qualified ranges.

- [ ] Diagnose identical teacher-forced prefixes512/1024/2048/4096/8192 against pinned llama.cpp0.6.0/build11429 `d812350`, CPU/F32 KV and Flash Attention off. Fix the physical microbatch/chunk/arithmetic schedule and compare per-head GDN state, gain/beta, quantization and AR-versus-bulk recurrence; distinguish replay correctness from arithmetic.
- [ ] Compare native ubatch1/64/256/512 only within bounded feasible runs; the historical full ubatch1 capture did not finish. Use raw token IDs to avoid incomplete UTF-8 HTTP rows. Recheck short corpus, layers, state and actual8K after any fix; retain original functional0.45 shared-top20-logprob/0.05 top-token-probability limits and strict source-bit operator gates.


## Committed

### 6d85559 — shared correctness checks, 26 Oct

- [x] `tools/model-correctness/`: pinned vLLM exact/default generation checks,
  complete-vocabulary KL/NLL/probability metrics and llama.cpp candidate-energy
  NMSE. Explicit identities/history/context/closure; no implicit thresholds or
  inference qualification. Independent review corrections resolved.
- [x] CPU evidence gate refuses nonfinite/duplicate/malformed reports and truthy
  flags; requires genuine2048-token evidence for both models. Existing source
  digests and0.45/0.05/0.06/14-overlap limits unchanged;16K/32K deferred.
- [x] Focused23 shared +15 CPU tests/selftest and complete isolated
  `scripts/check.sh` actual0/EOF/group closure:1890 Rust passes/0fail/89ignored;
  Python/script/installer gates pass. Source1778 digest
  `686a1168a88b543563c9a775c22893bd9514e3ed534dfaaf0d57be4947ecb06c`.
  [ROOT evidence](../../runs/q08-correctness-20261026/TODO.md).
- [x] Both models' retained same-attempt C16 observations:16 actual requests each,
  exact wire IDs/stored text digests under both policies, four new comparator
  invocations0/EOF. No fresh inference/fullV/state/performance qualification;
  case5 and historical2B performance failures preserved. Runtime source remains
  byte-identical to signed a857d45; no model/math/profile substitution.

### `9dd1ebfb1862` — original source operators and numerical closure records

Related source commits: `ec30569bc6b9` (cpu-reference-math), `eb49fa0e945b` (core-program-recorders), `225fc5e1a520` (qwen35-source-program), `abca70fdebda` (gpu-source-arithmetic-kernels).
Aggregate source21 qualification applies to the assembled tree; no per-commit
build or general model/backend grant. Historical failures keep their source scope.

- [x] **Expanded short-HF physical checkpoint,23Oct:** MAIN now converged with
  ROOT's full .8B/2B short batch and independent CPU model contributions.
  Both native matrices PASS all80 natural calls: genuine47/57/62/67/72+7,
  exact24 boundaries/42 domains/fullV/consumed priors/greedy feedback, eight
  independent roots, pending/snapshot/stale/cancel/survivor and actual final
  host0/physical0. Both five-cut cached fault suites PASS; same private owners
  remain quarantined on unknown work, with no Ready/readback/publication.
  .8B matched HTTP active8/both populations/all24 responses exact against stock.
  Final assembled MAIN isolated gate PASS1531 Rust/0failed/81ignored,
  260suites/18 namespace fixtures and complete scripted checks. Matched2B API
  now PASS full24 responses/active8/owned closure. Quick .8B C1/C4 baseline
  completed; owner now prioritizes fixing the40–52x throughput gap and beating
  vLLM on these profiles before further2B/C8/full128 timing. No quality waiver.
  All327 independently observed remains false; no throughput result yet.

- [x] **Previous MAIN source convergence,22Oct:** all536 selected runtime/support paths
  now match ROOT bytes/modes; preserve MAIN orderedSiLU and collision-safe PTX
  dump, unrelated worktrees/changes and rootREADME. Fresh MAIN strictClippy,
  focused SiLU/canonical tests, release builds and complete isolated checkgate
  PASS:1490 Rust passes/0failed/80ignored,257suites,18Python plus packaging/style.
  Actual six SiLU modes and both original GGUF B2 workers PASS with exact source
  state/cancellation/credit/EOS and final reservations0. Fresh MAIN canonical
  HF47 HTTP again observes active2, three exact8-ID chains, actual cancellation
  and survivor,400/no added work and owned retirement. Vendor104 pure CPU guards
  PASS including PTX collision fixture; independent harness/results review.
  Earlier full native numerical scopes remain as recorded below; this merge
  adds no wider two-model/active8/all327-oracle/original-GDN-tuple/perf claim.

- [x] **Cached47+7 and existing-worker HTTP unit accepted in integration
  worktree:** .8B original255-node/327-value path, actual greedy feedback and
  independent roots; eight complete state/boundary/fullV comparisons exact.
  Interleaved sessions, restore/stale/cancel/pressure+survivor, three submitted
  cached cuts and two affected legacy regressions pass; unknown work retains
  original fenced/quarantined owners with no failed Ready/readback/publication.
  Existing CLI chat/rawSSE observes active2, exact8 IDs/frontiers47..54, actual
  disconnect cancellation and healthy survivor, prequeue400/no extra GPU work,
  owned process/group/EOF cleanup. Complete isolated checkgate2 and independent
  native/HTTP reviews pass. HTTP server retired via owned SIGTERM; native healthy
  finalzero is separate. Fixed47+7/active2 only; all327 independently observed,
  original GDN tuple, broader two-model parity and performance remain unqualified.
  Source now present in MAIN and ROOT; fresh convergence checks are above.
  ROOT cached-http TODO records original source fixes and numerical outcomes;
  root README unchanged.

- [x] Whole Initial47 end-to-end unit accepted in integration worktree: same original255-node/327-value engine, all24layers/finalnorm/fullV and all42active canonicalstate exact for the healthy observed trajectory (26,860,544B independently captured coverage; all327 observed=false). Genuine sourceW1008+tied485windows verifiedtwice; normalhost/resident/physical0. Coldcancel/pressure/positive/five submittedfaultcuts and focused affectedlegacy/currentRoot/prefix/2B regressions PASS; independentownedcleanup checks and completecheckgate40 PASS. Unpublished initialcall only: cachedfeedback/workerHTTP/broadtwo-model parity and servingperformance remainopen. Next ONE coherentcached47..53/current-root+workerHTTP unit, with unrelated27B/Gemma implementation inparallel. ROOT acceptance68; rootREADME unchanged.

- [x] Original full Initial47 expected reference255/327 accepted separately: SAMEhealthy discovery2OBS source/trajectory and all44metadata/64wholearchive identities, original unchanged common/all42/attention gates freshly PASS with disjoint newreceipt destinations; all24paired166576128B/3096records rechecked, fullheldsource before/after and tied508559360B/finalgamma2048B actualsourceword joins closed. Expectedcomparison26860544B covers48blockboundaries/all42active/finalnorm+F32sum/selectedrow46/fullV; original8callfeedback provenance only, all327/privateLSE/originalGDNtuple/native/model/serving/perffalse. Frozen21children219deps fullyread/reauthenticated,12isolatedPython guards and independent ROOT postvalidation PASS; newreceipt315134B/cce765be. Sixnewreader children remain uninstalled/pinNone. Next fullInitial runtime/seals/pool/head source drafts in parallel; no modelacceptance fromreference.

- [x] Original47 fullattention-layer expected reference43/54 accepted separately: fixed healthy discovery2OBS,10additional original boundaries2165760B/total7603304B/five sourcelearnedmember wholeword joins; fullsource/heldarchives/ownedlifetime gates remain mandatory and original failedparents unchanged. Frozen31children/198deps,8Python guards and independent ROOT postvalidation PASS; receipt82015B/e69b0f6a. Expected only; distinct43reader uninstalled/pinNone and native/model/serving/performance false. Distinct43reader now ROOT-pinned and bounded native43cut accepted below; no wholemodel/serving/performance authority.

- [x] Canonical47 complete layer3 phase67 accepted ISO: original43nodes/54computed values through first attention Out/RMS/MLP match coherent healthyOBS7603304B external values/state/positions incl2165760B full attention-layer boundaries+unroundedF32sum. Same35carries/168+168sourceWwindows/36native/317dispatch/81684272B D2H/finalhost-resident-physical0. Nine new+62old physical modes and independent owned-lifetime checks PASS; six actual tailcuts retain358graph/submitted handles/206descriptors/11images15records/pending1/fenced384MiB/resident361679071B/host425209496B afterexternaldrops,0modelReady/D2H/expected/publication. All3compiled Sig/Mul/Out completeABIs equal original112/121/132; exactresource/SASS/source/budget gates and independent reviews PASS. Tuple factory repairs privatepacket seam without widening oldfields; coldpressureOutcome boxed for strictClippy; failedbuilds preserved. StrictClippy/71canonical/8image/118helper/100vendor and completegate39 PASS1333Rust/18Python/0warnings. Acceptance67/helper67; bothREADMEs unchanged. No all54public/originalGDNtuple/privateLSE/fullmodel/serving/perf/MAIN promotion. Next ONE Initial47 full255node/327value all24/finalRMS/fullV unit, then cachedfeedback/HTTP/parity/benchmarks.

- [x] Canonical47 FA phase66 accepted ISO: same36node/46computed-value path matches coherent healthyOBS5630056B external words/state/positions, including192512B new FA output;35carries/142+142sourceWwindows/29native/292dispatch/79516464B D2H/finalhost-resident-physical0. Five new+57old physical modes and independent owned-lifetime postvalidation PASS. Both actual partialcuts retain337graph handles/194descriptors/9images13records/pending1/fenced320MiB/resident332903083B/host409200612B afterexternaldrops,0modelReady/D2H/expected/publication. Full472B ABI equals independent capture; finite valid-row/shared/provider/null/dummy/resource proofs close under explicit ISA/compiler inference. PaddedQ/K remain unwritten; privateLSE numericaloracle and originalGDNtuple remain unobserved. Helper291 prediction omitted existing Value45 outputcertificate; exact292 physicalreceipt/source fix accepted, failed6601/6602 retained unqualified. StrictClippy/65canonical/7image/109helper/98vendor and completegate38 PASS1333Rust/18Python/0warnings. Acceptance66/helper66c; bothREADMEs unchanged. No all46public/fullmodel/serving/perf/MAIN promotion. Next full layer3 cut43/54, then one Initial47 all24/finalRMS/fullV unit and cached feedback.

- [x] Canonical47 QK/KV phase65 accepted ISO: actual35-node/45computed-value path matches fixed healthyOBS5437544B external values/state/positions,35carries and142+142sourceWwindows;28native/290dispatch/79323916B D2H/finalhost-resident-physical0. All7new+50old physical modes and independent owned-lifetime checks PASS; four actual partialcuts retain331graph/submitted handles/189descriptors/8images/12kernels/pending1/fenced320MiB/resident332709031B/host404940772B after externaldrops,0modelD2H/Ready/expected/publication. Exact ABI/resource/memory/source bounds reviewed; reciprocal closure uses matching compiler-operation semantics, not original PTX lineage or measured MUFUbits. Newhelper Vec7 and exact7image validator repaired without changing oldcaps; failed attempts preserved. StrictClippy/56canonical/6image/101helper/96vendor and completegate37 PASS1333Rust/18Python/0warnings. Acceptance65/helper65c; bothREADMEs unchanged. No all45publicvalues/originalGDNtuple/FA/fullmodel/serving/perf/MAIN promotion. Next finiteFA36/46 and fullattentionlayer43/54; healthyreferencewords already available, memory/resource/ABI/runtime budgets remain separate.

- [x] Genuine2B eleven-node MLP phase64 accepted ISO: actualInitial1/64 coherent11selectedBF16values/unroundeddualSum/ConvH exact;13computedvalues with fullConv4/Z5 unobserved, all42/40carries,117625184B immutableW/parameters and73+73sourceWwindows/finalhost-resident-physical0. Finalhelper64b all26new+16old physicalmodes/ownedcleanup PASS; deliberatecuts retain164/166actualgraph/submitted/quarantinedhandles,121descriptors/3owners/pending1/fenced320MiB/resident267266785or279807881B/host529010408or614496104B afterexternaldrops. Down1partial6native/1main/Yunpromoted; source cuts follow realcompletedD2H, computeReady0. All14compiledABIrecords equal independentcaptures after narrownewunionWMMAwordfix; aftertaint witness and exact2BCUTLASS loader repaired without changingoldroutes/readiness. StrictGPUClippy/43MLP/18upload/13component/95helper/93vendor and completegate36 PASS1333Rust/18Python/0warnings. Acceptance64 preserves64a/failedpreflights/formatattempts/README. OriginalselectedGDNtuple differsowned4/3; fullmodel/serving/perf/MAINpromotion false. Next47QK-KV finitecompiler/reference/runtime qualification.

- [x] Genuine2B cold MLP weight transport phase63 accepted ISO: Initial1/64 real GateUp+Down72MiB/73H2D checkpoints and73serial sourceD2H windows/16KiB leadingRead match actual immutable source; normal host/resident/physical0. All24new+12old physicalmodes/ownedcleanup PASS; first/middle/last-read and validator/empty-compute refusals retain actual2W handles/source/record32MiBticket/plan/pending1/fenced72MiB/112459776Bhost after externaldrops. Source9+prerequisite7guards/16uploadRust/89helper/91vendor/strictClippy/completegate35 PASS1333Rust/18Python/0warnings. Acceptance63/helper63 preserves old62/README and failedtest-scope/vendor-setup attempts. Read-only completion/source verification only: nonemptycompute/nativeMLP/fullmodel/serving/perf false. Separate healthy coherent2B11 expected reference accepted after16Python guards/strictoriginalcommon+prior262578668B/all42pair113713152B/full4.548GBsource-before-after/75heldidentities337deps; expectedonly, reader remainsNone. Native2B11/47QK-KV continuation source-only in parallel.

- [x] Canonical47 computed QKV prefix phase62 accepted ISO: genuine33nodes/42computedvalues through original layer3 inputdual+QKV, stopbeforeQK/KV. Nine public boundaries +node31unroundedF32sum +sixcanonicalstates4,810,752B exact against fixedOBS; all42priors/36carries/142+142immutableWwindows/finalhost-resident-physical0. Actual26native/286dispatch/78879600B D2H. Seven new+12old physicalmodes/ownedcleanup PASS; partialmainunknown/reducercancel retain319actualgraph/submitted/quarantinedhandles/181descriptors/207uninitializedfuture/oneowner/pending1/fenced320MiB/397242604Bhost/326693771Bresident after externaldrops; exactlyoneQKVenqueue/Yunpromoted/0Ready+D2H+publication. Source/packet/resource/budget/independentreviews/48canonical/89helper/90vendor/7Pythonhelper/6budgetguards and completegate34 PASS1333Rust/0fail/76ignored/257suites+18Python/release/installer/shell/Nix/0warnings. Cache dispatch/format/diagnosticJSON repairs, failedattempts, frozenpackages/old61/README preserved. Acceptance62/helper62a; no all42publicvalue/originalGDNtuple/fullmodel/serving/perf/MAIN promotion.

- [x] Explicit2B postAttention phase61 accepted ISO: genuine Initial1/64 eight-node OwnedSolve4Recompute3 source paths match fixed originalOBS values0/1/2/3/6/7/8/9 (1333280BF16words),133120unroundedF32sum/36864canonicalConv/524288H words; all42priors/40carries/42127712B immutableW per shape/finalhost-resident-physical0. Finalhelper61b tennew+14old actualmodes/ownedprocesscleanup PASS; unknown/latecancel retain159/161graphhandles,118descriptors,158/161submitted-quarantinedhandles/twoowners/pending1/fenced224MiB/472190696or557676392host bytes after externaldrops,0D2H/Ready/publication. FixedOBS acceptor authenticates full4.548GB source before/after/320roles/wholearchives and preserves originalcall2 fatalparent; actual9stage OBS metadata differsownedtuple. Independent source review/20Python/83helper/strictGPUClippy/targetedRust and completegate33 PASS1333Rust/0fail/76ignored/257suites+18Python/release/installer/shell/Nix/0warnings. Two modulepaths/incorrect diagnostic pressurestatus literal repaired against existingsource; failedbuild/61a preserved. OldNone reader/60blauncher/rootREADME unchanged. No all10values/fullConv/Z/originaltuple/wholemodel/serving/perf qualification.

- [x] Fresh original canonical47 attention discovery2: PLAIN/OBSERVED common, all42 and attention raw comparisons PASS for natural47→seven cached1; coherent reference only. All56 scopes/64 launches/12 reflected layouts, original320roles/33primarypins/held owners and process/group/reap/container/EOF closure verified. Host-only observer quota and zero-based event-order refusals repaired against unchanged producers; failedattempts retained. Declared-argument capture remains in progress; scoped first3 acceptance below does not establish model/serving/performance parity.

- [x] Canonical47 first3 model cut phase60 accepted ISO: finalhelper60b five new physical modes +12old regressions PASS; same31node cursor/39computed readbacks/24native/271dispatch, six coherent original boundary values +six Conv-H states3,944,448B exact;132+132 immutableW windows/36carry domains/42copies/48certificates/finalhost-resident-physical0. Unknown/latecancel retain305actual prepared/submitted/quarantined handles,96state/39values/15Wgroups/174descriptors/195uninitializedfuture/5images/8kernels; pending1/fenced320MiB/373107256Bhost/314753367Bresident after external drops,0callback/readback/Ready/publication. Independent source/fault/helper reviews, strictClippy36canonical+fault/78helper/sourceguards and completegate32 PASS1333Rust/0fail/76ignored/257suites +18Python/defaultrelease/offlineinstaller/shell/Nix/0warnings. Acceptance60/helper60a+60b preserved. Whole39 numerical coverage=false, original selected GDN tuple unobserved, rootfrontier0/unpublished; no wholemodel/serving/performance qualification.

- [x] Original47 attention finite14 whole-image catalogue accepted for observation metadata only: complete existing fourFA images/equivalent code-resource-ABI sections checked;12original callbacks +two exact module variants,36host guards PASS. Failedcapture1 bit64/eight missingargumentpackets preserved; no hash normalization/reconstruction/native/math permission. Freshcapture2 next with original producer/owned controls.

- [x] Current isolated initial whole layer0 and three-layer GDN prefix: real1/64-token same-plan producers, independent next-boundary and canonical-state matches; all12cancel/pressure/unknown cuts, prior regressions and fullgate9PASS. ROOT run checklist records exact scopes/failures. No whole-model or MAIN composition acceptance.

- [x] Genuine source recipe capacity bug fixed at producer; actual six-recipe/44-descriptor original-source regressionPASS, unchanged admission guard.

- [x] Actual finalnorm/fullBF16head/newV1sampler capture2PASS under original353forced fixtures; source/publichead evidence only. Naturalgreedy/headnative/ZGmodel remainopen.

- [x] Isolated attention native leaves:1088000BF16exact/5calls/30kernels/3DtoDcopies/twocomputedKVhistories/sourceW/final0;4cancel/pressure/unknowncutsPASS.82component+46purevendor+strictClippy/fullgate10PASS. Authenticnormalized/cosSin/gatedOutinput remain explicit capturedleaves; connectedattention/model notqualified.

- [x] Actual all42active-state/24dual-boundary reference PASS106930176B across five phases; old153/finalnorm/fullV closures and original owned cleanup preserved. Reference values only, no ZG whole-model qualification.

- [x] Head5 native arguments PASS; Gate35 capture3 healthy/all40 payloads complete. Offline semanticv6 PASS all29private owners/3copies, no unresolved fields/driver operations; failed captures1/2 and semanticv5 alias failure preserved.

- [x] Isolated full485MiB tied-table upload:485H2D/checkpoints+485independentreadbacks/all508559360B exact/final0; all5cold/pressure/cancel/unknown modesPASS, originalretention/fence/cleanup checked. StrictClippy/6coverage+3vendor/initialprefix1+64regressions/fullgate11PASS. Transfer only, no nativehead/model acceptance.

- [x] Isolated cached layer0/prefix3 continuation: actual1→1/64→2→1/all32positive/cancel/pressure/unknown/restore modesPASS, all42prior/carry/computedcanonicalstate/sourceW/originalcleanup checked; oldinitialprefix1+64/connectedregressions/fullgate12PASS. Sameunpublishedrootfrontier0; no fullmodel/serving acceptance.

- [x] Isolated connectedattentiongate: actualFA→separateBF16sigmoid→multiply→Out/1370624BF16/5phases/40native/3copies/sourceW/host0; all5physicalmodes+6regressions/91component/50purevendor/strictClippy/fullgate13PASS. Normalized/positions/cosSin stillauthenticinputleaves; no fullmodel acceptance.

- [x] Actualrotaryconstructionarguments v3:15CUDAops/12completepackets/11layouts/4typedhostqueries/unknownAPIsempty/fulloldsource320/state/fullV closures+originalcleanupPASS; no replay/sourceproducer/model acceptance.

- [x] Isolated nativefinalnorm/phase-selected fullvocabularyhead: all10physicalmodes+6regressions/96component/58purevendor/helperguard/strictClippy/fullgate14PASS;1241600BF16fullV +70656eachunroundedF32sum/dualBF16 +5120selected exact,485MiBsourceWunchanged/final0; original576MiBunknownquarantine checked. Finalbranch/residual are stillauthenticinputleaves.

- [x] ActualunforcednaturalvLLM+all42/24boundaries reference:chains[198,285]/[72,18],fullV/feedback/numeric-ties/source320/106930176Bstate+boundarypairPASS. Offlinevalidator3+22faulttests/peerreviewPASS; originaltrace-onlyPLAINcheckerfailure andunexecutedv2 preserved. Referenceonly,noZGmodel/sampleracceptance.

- [x] Isolated full24source metadata foundation: original5recipes255nodes/327BF16values/42effects each/selected0,0,63,1,0; core144/strictClippy/helper34/2actualprefix-headregressions/fullgate15PASS1318Rust. No mathematicalmodel execution, Rootpublication or wholemodel acceptance.

- [x] Isolatedoriginalsource rotaryproducer: all9physicalmodes/4regressions/99component/65purevendor/strictClippy/fullgate16PASS;12nativefactories/full1Mgeometry/52364B11prefixes+8BI64exact/private67rowprefix afterretirement/final0,all6uncertainquarantinesretained. FailedfirstELFguard preserved;9exactheaders/fullSHA-beforeparse/structuralguards qualified. Noall1Mexpectedcoverage, allocatorpool0ormodelpromotion.

- [x] Prior isolated forced full24/all42/fullV/BF16greedy scope: actual positive3, original unknown/cancel cuts, regressions and fullgate17 PASS. This remains unpublished forced353 evidence; source-frozen helper39 does not establish natural feedback or serving.

- [x] Isolated natural current-root binding/publication: finalhelper43 all7physical modes PASS; actual5calls1/1/64/2/1 match all24dual boundaries/all42/fullV/greedy, original roots and chains[198,285]/[72,18], prefix64emits0. Normal/safe cuts sourceWunchanged/host-resident-physical0; uncertain cuts retain actualcompletegraph/noReady/noD2H/noPublication under originalfence. Restore is stale rejection only.8authority/13reader-mode-reducer/3slot-budget/81vendorhost/2runner tests, strictcombinedClippy2 and completegate18(1320Rust/0fail/76ignored) PASS. Original NVRTC startup failures retained; existing C++ runtime path corrected, temporary diagnostic removed. ISO only; no production/2B/performance promotion.

- [x] Isolated reusable same-ResidentModel BF16 storage: finalhelper45 seven new +nine old physical modes PASS; two genuine request pairs/10calls/four independent Sessions, same five scratch geometries,480fresh canonical buffers/10fresh packets,8committed packets physically preserved96B. All24/all42/fullV/feedback exact; actual2038native/96clears/174copies/13712dispatches. Duplicate candidate submission, cancellation retry, fresh post-restore inference and competing Ready/commit refusal pass. Unknown cuts retain original current graph/pending1/fence with no reuse/publication. Source1008/tied485 unchanged; normal/safe host-resident-physical0, unchanged host3170893824B/device5GiB caps.6new hostguards/18helper tests, strictcombinedClippy2 and completegate19(1320Rust/0fail/76ignored) PASS. One equivalent probe lint fixed and final source rebuilt/retested. ISO only; math still fixed1/64/2 coordinates/rotary0..67, no production serving or benchmark acceptance.

- [x] Original canonical47 HF.8B reference1: genuine unforced47→seven cached1 calls/max8/EOS248046 policy, unchanged PLAIN+OBSERVED and both validators PASS. Actual IDs[248068,198,90700,8340,25,271,16,13], fullV/actualV1sampler/common20256232B and all24dual/all42active166576128B exact. All34frozenchildren,33primarypins before/after,320loadedroles and two heldsourceowners/lane unchanged through original owned group/reap/container/EOF cleanup. ROOT postvalidation1 records acceptance REFERENCE ONLY; no native canonical/C1population/serving/timing qualification.

- [x] Bounded original2B RMS component in ISO: authentic sourceembedding→ordinary2048RMS and explicit captured finaldual leaf exact across1/1/64/2/1;141312words each output,100stages/10actualcompletions. Precancellation/cap+1/afterMeanunknown/latecancel and fiveexisting.8B physicalregressions PASS. Finalhelper47/strictcombinedClippy4/21helperhostchecks/completegate20 PASS1320Rust/0fail/76ignored +18Python/release/installer/shell/Nix,0warnings. Wholemodel/serving/performance remain open; finaldual operands are leaves.

- [x] Exact source-family metadata foundation in ISO: current helper49 authentic2B and old.8B source recipes PASS (five255-node/327-value recipes,24boundaries/42effects,320roles). Seven current physical regressions, strict combined Clippy5,21RMS/23helper host checks and complete gate21 PASS1326Rust/0fail/76ignored/257suites +18Python/release/installer/shell/Nix;0warnings.2B source identity/geometry metadata only, no native2B model permit.

- [x] Owned GDN source-path audit correction: current GDN executes translated visible PTX through original PinnedKernel, so an unused original private CUBIN ABI is not a prerequisite. Original opaque projection/FA replay keeps its actual ABI gates. Seven requested47/2B source families yield15descriptors/fiveCUDA sources; source generation only, no new phase/runtime permit. Static included output/KKT matches are not physical qualification.

#### Binding continuation — 21 Oct 09:30 UTC

- [x] Independent projection metadata V3:13pureguards, source review and both actual offline gates PASS.2B40scopes/45launches/698events; canonical47 64scopes/88launches/1155events. Exact resource/function/reflection layouts accepted only; packed arguments, scalar semantics and native math remain unqualified. Separate receipts preserve failed parent numerical comparisons.

- [x] First2B selection observer failure preserved: actual caller uses hidden_states keyword. V2 strict binder/15pureguards and fresh actual pair2 PASS; five calls/18actualrecords per lane, source/ownedcleanup/three strict original raw gates PASS. Cumsum/KKT/recompute/H configs differ while all42/fullV match. No configoverride or nativeZG/cause/serving claim; ROOT selection-postvalidation-2.json pins scope.

- [x] Bounded47 RMS component in ISO: independent ordinary leaf reference authenticates source rows/gamma,291004 boundary bytes and11ATen stages; parent recurrent-state failure preserved. Helper49 five physical modes PASS:48128 words each source embedding/ordinary/dualF32sum/dualBF16outputs exact,20stages/two completions; cancellation/pressure/afterMeanunknown/latecancel. Unknown cases retain12handles/source/prepared/pending1/fenced128MiB and131339036host bytes;0Ready/D2H/no recovery. Seven old physical regressions and complete gate21 PASS. Accepted49binary preserved; no canonical whole-model/GDN47/serving/performance promotion.

- [x] ISO semantic span/provenance stage1 accepted: same source recorder/traversal, immutable teacher issuer+mode, fallible legacy coverage and unchanged old constructors. Current helper50 authentic47 source route records8true47/47..53 spans with255nodes/327values/24boundaries/42effects each; realhost0/noCUDA. Eleven actualsource/physical regressions PASS including existing ten-call full24/all42/fullV natural reuse. Core157/helper26/corrected2B8host guards, strictcombinedClippy and completegate22 PASS1333Rust/0fail/76ignored/257suites +18Python/release/installer/shell/Nix,0warnings. Source metadata grants no native47/2B wholemodel permit, Session feedback, serving or performance acceptance.

- [x] Actual projection declared-argument capture1: original2B pair passes strict source/common/final/prior/all42/RMS gates; separate argument gate accepts40scopes/45launches/698events with zero common differences. Canonical47 both lanes/source/ownedcleanup complete; argument gate accepts64scopes/88launches/1155events, but original exact recurrent.after call0 fails (23common digest differences). Both independent receipts retain scalar/pointer/ABI/native/replay/performance refusals. No parent-failure waiver.

- [x] Current instrumented47 stage reference1:24pureguards/source review/42frozenchildren/33parent and21runtime pins PASS; actual PLAIN+OBS35records6153077B/lane exact, including embedding/RMS/QKVZ/BA/fullConv/prep/cumsum/A/Ai/W/U/H0/h/Vnew/HT/O. Original8-call pair/fullV/common20256232B/all42 gates PASS. ROOT streams both new all42 archives166576128B/lane exact to acceptedref1. Both current lanes now follow acceptedref1; original PLAIN failure is NOT fixed. Early copies/retention changed experimental conditions; next late-H-only observer must reproduce oldbadHT before causal interpretation. No native/serving/performance promotion.

- [x] Standalone2B QKVZ/BA unit accepted ISO only: actual paired projection+Gated reference1 source/owned cleanup/original common+prior/fullV/all42/RMS PASS; normalized141312/QKVZ565248/BA2208/Gated141312 BF16words paired exact. Actual expected pin sealed. Helper51 five physical modes PASS: five callbacks/11native launches/1134912projection bytes exact, full33685504B source-W unchanged, normal host/resident/physical0; uncertain retains18prepared buffers through five submitted handles/pending1/fenced128MiB/0Ready+D2H. Submitted5/prepared18 and pre-cancel64KiB owner fixes independently reviewed. Corrected6library/3helper guards, finalstrictClippy2, four old physical/source regressions and completegate23 PASS1333Rust/0fail/76ignored/257suites +18Python/release/installer/shell/Nix,0warnings; earlier function-count assertion failure preserved. No connectedGDN/fullmodel/serving/performance promotion.

- [x] Actual2B QKVZ/BA leaf reference prerequisite: fresh original projection+Gated pair1 passes all original raw/source/owned-cleanup gates; independent projection/Gated receipts pin actual payloads absent acceptedref3. Native helper51 physical projection and complete phase gate are accepted above.

- [x] Current failed47 pair causal localization: actual original pair `root-canonical47-solve-counterfactual-1` has equal first24 boundaries, including fullConv/prepared inputs/controls/F32KKT A; first difference is solve.Ai5BF16words. PLAIN naturally selected solve2/2 PTX8201e612…bb02, OBSERVED solve4/4 PTXfcae68be…c135; both actual original recompute8/4 PTX5731fd14…0c9a. W59/U9→HT1002 differences follow. Both retired lanes source/owned-cleanup pass; original raw FAIL remains fatal. ROOT streams all24/all42 actual166576128B/lane: PLAIN exactly reproduces prior bad trajectory, OBSERVED acceptedgood. No native/model repair acceptance.

- [x] Same-input original solve counterfactual: four repeated solve2/2 vs4/2 outputs followed by fixed recompute8/3 PTXd9c9fff7…1763, after all original calls; both lanes reproduce badW59/U9 for2warps and goodAi/W/U for4warps. Actual inputs/outputs and original Autotuner state preserved; no historical payload enters worker. This establishes current solve-variant arithmetic sensitivity, not universal catalogue equivalence or historical uncapturedA equality. Separate controlled recompute tuple is not the original8/4 tuple.

- [x] Actual2B six-node initial component phase52 accepted ISO: same original ResidentProgram loop, genuine source embedding2048/RMS/QKVZ8192/BA32/Conv/recurrent/Gated for1/64, all42 priors/candidates and40 exact unchanged carries. Whole normalized/QKVZ/BA/Gated and Conv18432BF16/H262144F32 exact;33735008B immutable source unchanged. Ten physical modes+five old regressions,2library/32helper guards, finalstrictcombinedClippy3 and completegate24 PASS1333Rust/0fail/76ignored/257suites plus18Python/release/installer/shell/Nix,0warnings. Unknown retains original source/graph/fullprepared owners after locals dropped; submitted144/147 and pending1/fenced192MiB/0Ready/D2H/publication. Diagnostic provider192MiB and helper576/704MiB are separate from unchanged old caps. Root unchanged/frontier0; stops before GdnOut; no cached2B/native47/wholemodel/serving/performance permit. Accepted executable52 preserved and phase-acceptance-52.json pins results.

- [x] Finite original47 solve variants phase53 accepted ISO:34sourcechildren/8dependencies/4parents/peer review/10pure guards and independent actual leaf reference PASS while exact original recurrent-after call0 fatal failure remains. Helper53 both original solve2/2 vs4/4 each x2 fresh zero cohorts +originalrecompute8/4:1925120B fullAi/W/U and2334272B immutable operands exact. Five physical modes+four oldRMS47/2B64prefix/naturalreuse regressions,6library/32helper guards/strictcombinedClippy2/completegate25 PASS1333Rust/0fail/76ignored/257suites plus18Python/release/installer/shell/Nix,0warnings. Unknown/latecancel actual19prepared/10submitted/pending1/fenced128MiB/96534528B retainedhost/0D2H+Ready+publication. Initial compile errors preserved/corrected/peer reviewed; binary53 preserved. PLAIN fullABI remains same-source reconstruction, capturedPTX separate. Leaf proves variant reproduction, no connected47/model/serving/performance permit.

- [x] Exact private GDN arithmetic source binding phase54 accepted ISO:26sourcechildren/36deps/7parents/4newchildren composed preserving53bridge; source-only compilation policy distinguished from actual toolchain/provider identity. Actual stage code/PTX/entry/threads/shared/grid/ordered ABI, sourcefamily08vs2B, phase/slot/typedflag, control bytes and zero/packing/state-copy implementation bound to preparedinstance; sameArc retained by model/2B wrappers, all18samephase layers and natural/reuse cold tuples checked.30GDNguards incl11new/32helper/strictcombinedClippy1/all12physical regressions/completegate26 PASS1333Rust/0fail/76ignored/257suites plus18Python/release/installer/shell/Nix,0warnings. Actual original caps/owners/quarantines/source/state/fullV/feedback preserved; no new47/default/whole2B/serving/performance permit. Helper54 and phase-acceptance-54.json preserved.

- [x] Canonical47 connected first-six frontier phase56 accepted ISO below, following finite47 projection55 and exact arithmetic54. Existing publicGemmExCreate status1/export-UUID issue remains blocked; no bypass. Original failed47 parent remains fatal and separate from bounded connected-prefix acceptance.

- [x] Canonical47 QKVZ/BA explicit normalized-input leaf phase55 accepted ISO: current source35children/27deps/6parents; independent QKVZ and BA address/branch audits, authentic small-original NVINFO corroboration and pinned CPU NVRTC→SASS predicate proof. Exact captured66c2 image/manifest, immutable sourceW, original actor/flight/caps retained. Finalhelper55b all5native modes+5oldRMS47/2Bprefix64/solve47/naturalreuse regressions PASS;385024QKVZ+1504BA+48128input BF16words exact,16842752B sourceW unchanged, normalhost/resident/physical0. Unknown/latecancel6handles/3owners/pending1/fenced128MiB/resident93340608B/retainedhost329843168B/0Ready+D2H+publication. Negative-single fixture/Linuxcfg fixes and initial unrecognized exactCUDA123 ELFheader failure preserved; onlynewexact7.8MB/64B header branch added, fullSHA/structuralguards unchanged. Actualparser9/5GPU/35helper guards/strictcombinedClippy5/completegate27 PASS1333Rust/0fail/76ignored/257suites +18Python/release/installer/shell/Nix,0warnings. Directvendorregistryguard remains authored/unrun because vendor not workspace-testable. Frozen drafts/priorhelpers/README/30worktrees preserved. No computed47GDN/wholemodel/serving/perf permit; phase-acceptance-55.json pins bounded acceptance.

- [x] Canonical47 source six-node connected prefix phase56 accepted ISO: actual embedding/RMS/QKVZ/BA/Conv/Z/Gated964064BF16words, eight GDN intermediate648672words and canonicalConv18432/H262144F32words exact versus independently authenticated originalOBS expected trajectory. All42fresh priors/40completeunchanged carries and16892256B immutableW verified;3native/134dispatch/86312100B readback. Five physical modes+9old47/2B1/2B64/naturalreuse regressions PASS. Unknown/latecancel147actual graphbuffers/119descriptors/pending1/fenced192MiB/retainedhost468406660B after externaldrops,0D2H/Ready/publication; normalhost/resident/physical0. Independent complete source review,18canonical47/32GDN/3component filter tests(overlap),44helper guards/strictClippy/completegate28 PASS1333Rust/0fail/76ignored/257suites +18Python/release/installer/shell/Nix,0warnings. Two private-module helper imports corrected to existing public reexports; failedbuild retained. Current OwnedCanonical47Solve4 code remains different from originalOBSselectedtuple despite actual fullword equality. Same engine/session/attempt; frontier0/stopsbeforeGdnOut. Acceptance56/helper56 preserved; no full47/cached47/whole2B/serving/perf permit.

- [x] Canonical47 eight-node through GdnOut/postAttention phase57 accepted ISO: same computed source prefix plus144384newBF16 Out/normalized/residual words and48128unroundedF32sum words exact against fixed originalOBS. All42priors/40carries,21088608B immutableW,4native/148dispatch/90991780B readback; frontier0/unpublished/stopsbeforeGateUp. Five47modes+eight2Bfaultmodes+11oldregressions PASS, including source/Out/futuredual/all42 Weak-only unknown retention161handles/126descriptors/pending1/fenced224MiB/510367920Bretainedhost; safe/normalhost-resident-physical0. Exact3packet finite memory proof/13guards,62helper+3Python guards/strictClippy/fullgate29 PASS1333Rust/0fail/76ignored/257suites +18Python/release/installer/shell/Nix. Source-role/private-import/Clippy fixes and first failed expectedguard (snapshotmanifest vs learnedbundle identity) preserved/corrected with negativeguard. Helper57b and acceptance57 preserved. Original47 pair stillfatal, code tuple stilldifferent; no wholelayer/model/serving/perf permit.

- [x] Canonical47 MLP11cut phase58 accepted ISO: actual source-produced eleven-node GPU path now PASS through GateUp/SiLU/Down, fixed observed original reference; computednew values8/10/11/12=48128/336896/168448/48128BF16words. Same42domains/40carries and43108704B immutableW, two independent22window source verifications,8native/155dispatch/114118820B readback; root0/unpublished/stopsbeforelayer1 and normalhost/resident/physical0. Five new physical modes plus13old regressions PASS with owned binary/argv/group/signals/reap/EOF postvalidation. Unknown/cancel-beforeACT retain166actualhandles/131descriptors/pending1/2owners/fenced224MiB/514693296B host after external/caller drops. Actual ABI8/full finite memory/SASS joins accepted;48vendor/71helper/3launcher/targetedRust guards and strictClippy PASS. Completegate30 PASS1333Rust/0fail/76ignored/257suites +18Python/defaultrelease/offlineinstaller/shell/Nix/0warnings; phase-acceptance-58.json retained. Original failed parents and distinct owned-vs-original code tuple remain explicit; no wholemodel/serving/performance permit.

- [x] Owned2B solve/arithmetic phase59 accepted ISO: explicit OwnedSolve4Recompute3 remains separate from originalOBS4/4. Two fresh native T64 cohorts compare all1310720B Ai/W/U and1589280B immutable inputs; whole6307872B expected archive SHA/EOF verified only after each actual completion. Five leaf modes and three explicit source-prefix modes PASS, including genuine computed Initial1/64 six-node paths/all42/40carries/33735008B unchangedW and coldcancel. Finalhelper59b also passes8old physical regressions with owned group/signals/reap/EOF postvalidation. Unknown/latecancel retain13prepared/10submitted, resident2105360B/pending1/1owner/fenced128MiB/96534528B host after external drops;0D2H/Ready/publication. Independent source review,33GDN/6leaf/8component/74helper/5sourcePython/4launcher checks and strictClippy PASS. Completegate31 PASS1333Rust/0fail/76ignored/257suites +18Python/defaultrelease/offlineinstaller/shell/Nix/0warnings. Acceptance59 pins exact final binary/source; source-only donor and prior failed builds remain preserved. Original failed fullmodel parent remains fatal; no whole2B/47model/serving/performance permit.

#### Bottom-up F32 repair — 20 Oct

- [x] Closed pinned Haswell tinyBLAS F32 primitive: one Vec8 FMA chain,
  exact horizontal/tail/zero order; unchanged RNE/masked/noFTZ/noDAZ required
  at entry/completion. Existing AR1 reductions remain unchanged.

- [x] Same PlanBuilder/HostMath sealed temporal matrix slice: exact original
  source/teacher issuer/layout, leased inputs/output, cancellation/pressure,
  retained clones/final0. Full-model/state publication and GPU lowering refuse.

- [x] Original .8B AV +2B alpha default-bulk captures: all132096 FP32 words
  exact debug/release; typed same-engine original-source replay also exact.

- [x] Connected original2B layer0 first64: alpha/beta projections2048 words
  → coefficients3072 → recurrence core131072/final-state262144 → grouped
  RMS/raw gamma/SiLU gate393216; every captured word exact debug/release.
  Existing recurrence equation and SiLU arithmetic remain unchanged.

- [x] Distinct original2B `ssm_out` Q8_0→F32 OpenBLAS leaf: native plain/GDB
  all131072 captured words exact. Actual0.3.33Zen/16threads,1792 kernel
  rectangles; each output receives K panels320×5+224+224 with even/odd FMA
  chains, rounded merge and alpha1 FMA write. Rust debug/release all131072
  bits and7 arithmetic/control/cancellation/pressure tests pass. TinyBLAS
  contract/limits stay unchanged; no OpenBLAS/FFI runtime dependency added.

- [x] Computed gated producer → original whole Q8 matrix through the same
  generic fragment/PlanBuilder/HostMath: all131072 output words exact
  debug/release; source/span/provider/full-route guards,3core tests,
  retained output/source ownership and final0 pass. No state publication.

- [x] Latest complete isolated gate:1276 Rust passes/0failures/76ignored/
  256 suite invocations;18 Python plus release/installer/shell/Nix checks.
  Strict CPU/core/model plus CUDA+Vulkan and all3 GPU cold-refusal tests pass.
  Both original cache-direct/HF-import release regressions pass full248320
  logits/all42 domains for both approved models. Independent reviews found
  no blocker in the qualified scope.

- [x] Compose40 owned code/test paths into MAIN with fresh mtimes. MAIN
  combined strict and all6 actual source comparisons pass;38 owned Rust
  files pass formatting. Original changes, worktrees, README and existing
  unaccepted SiLU binary preserved; no staging/commit/push or GPU execution.

- [x] Native probe initialization refusal retained: AS2GiB could not fit
  ctor16×128MiB reservation. Source-informed AS3GiB retry preserved environment/
  affinity; plain/observed maxRSS54/84MiB, owned groups/EOF retired. No benchmark
  or whole-host physical-memory qualification follows from these diagnostics.

- [x] Replace earliest external input: actual original2B first64 selected Q6_K
  embedding→GGUF RMS/raw effective gamma through the same generic recipe,
  PlanBuilder and HostMath. Native fields1/2/4:393216 words exact debug/release.
  Stored GGUF norm is already adjusted; no extra+1 or BF16 contract reuse.

- [x] Retain that actual normalized producer/issuer into coefficients without
  RHS copy or caller source/span/admission substitution; all922624 downstream
  words through Q8 ssm_out remain exact debug/release and after MAIN composition.
  Original source and all3 output buffers retain physical leases. Pre-cancel,
  pressure including private embedding→RMS refusal, unaware provider refusal,
  output clone/source lifetime and final0 reservations pass. No state publication.

#### Remaining qualification

- [x] **Full-runtime reference-control prerequisite:** actual public metadata
  identifies the rejected constructor as pinned BLAS (ACCEL), before CPU
  initialization. The diagnostic keeps its original pre-setter exit125 and
  produces no numerical capture. Strict build, exact source/build/invocation
  binding and owned process/group/EOF cleanup pass. This diagnoses a control
  harness refusal; it does not explain or close the long-context arithmetic
  divergence. Guarded public compute-route coverage remains pending; no BLAS
  disable, source arithmetic or tolerance change.

### fff1842 — complete two-model reference control

- [x] **Fresh2B whole-runtime reference control:** original8192-ID corpus,
  CPU8/F32 KV/FA off, batch/ubatch64, public use_ref and distinct CPU_Mapped
  intermediate placement pass all730 selected records/full248320 logits and
  all42 active causal domains,55,382,016 state words, zero bit differences
  versus accepted Rust. Plain/observed control runs preserve all730/fullV
  and whole serialized prior/post-state. Original CPU/BLAS providers remain
  live in their original order; no BLAS math is delegated. Actual public
  graph placement, buffer types, CPU plan/TLS route, source/build/invocation
  and owned process/group/EOF cleanup checks pass. V6/V7 pre-delegation
  exit125 failures remain preserved. This qualifies a declared reference
  control, not default bulk arithmetic or historical corpus recovery; the
  broader required numerical gate stays open. No production math or tolerance
  changed. Source freeze6f349a12; exact native state export38f96ba1.

### 9a19901 — original reference-control schedule

- [x] **Fresh0.8B whole-runtime reference control:** original8192-ID corpus,
  CPU8/F32 KV/FA off, batch/ubatch64 and public CPU use_ref control pass all730
  selected records/full248320 logits and all42 active causal domains,
  55,382,016 state words, zero bit differences including signed zero versus
  accepted Rust. Actual630 public CPU calls bind630 inner plans with use_ref
  true and exact TLS ownership; two BLAS lifecycle contexts execute no math.
  Original source/build/provider/observation and owned cleanup guards pass.
  This qualifies the explicit reference-control schedule, not default bulk
  arithmetic, historical corpus recovery or production arithmetic changes.

### bfe62df — independent source and reference-path qualification

- [x] Fresh mixed/Unicode teacher prefix: both original GGUF models at
  512/1024/2048/4096, pinned native AR1/F32 KV/FA off versus Rust: all730
  selected actual stage records/full248320 logits and independently decoded
  all42 active causal state domains bit-exact. Each4K state comparison covers
  30,216,192 words. Fresh native8K captures/55,382,016-word state exports are
  complete. BOTH fresh8K comparisons now pass all730/full248320 and all42
  active causal domains,55,382,016 words/model with zero differences;
  Rust6K captures alone are not independent6K comparisons. Historical8K raw
  corpus was not recovered; this fresh corpus does not establish its replay.
- [x] Unchanged0.8B first64 bulk-versus-AR1 capture narrows the first observed
  result gap to layer3 attention AV, position2 (530/2048 words), before the
  previously observed layer4 GDN gap. All131072 probability operands and
  1,064,960 causal V comparisons are exact;114432 future probabilities are
  positive zero. Earlier captured layers0–2/GDN states match. Plain/observed
  whole state and next AR1 full-vocabulary guards prove capture transparency.
- [x] Genuine public ggml AV replay:65 defaults/262144 words match their own
  captures. Public reference-control bulk differs57823 words from default
  bulk and matches all64 default AR1 outputs exactly. Transparent exported
  entry observation preserves all130 default/control outputs (524288 words)
  in both runs and records576 real calls:bulk64 takes64 true returns,
  AR1 takes512 false returns, reference control takes none. Exact opaque ABI,
  pinned DSO origin and bounded owned cleanup pass independent review.
  Internal compiled kernel/full-runtime cause is unqualified; no production
  arithmetic, source profile or tolerance changed.
- [x] Unchanged2B first64 bulk/AR1 capture: plain/observed whole post-state
  and next AR1 full248320/whole-state guards pass. All28 original physical
  learned operands remain unchanged after both modes; eight opaque CPU_REPACK
  payloads are transport-only, with no codec equivalence claim. The first
  observed result gap is layer0 F32 alpha projection, position0 (13/16 words).
  All960 preceding compared rows and131072 actual alpha-input words match;
  its full F32[2048,16] weight equals the original131072 GGUF bytes. Beta
  projection also differs before GDN. No production fix or tolerance change
  is inferred from this capture.
- [x] Genuine2B public alpha replay:65 defaults/2048 output words match their
  own original captures. Public reference-control bulk matches all64 default
  AR1 outputs exactly; reference AR1 outputs are unchanged. Transparent
  exported-entry observation preserves all130 default/control outputs,
 4096 words/run;520 calls comprise8 bulk true,512 AR1 false and zero control
  calls. Original inputs/full raw F32 weights, source/DSO identity and bounded
  owned cleanup pass. This narrows native batch-dependent arithmetic before
  GDN; internal compiled route/full-runtime cause and historical issue closure
  remain open. Rust/source arithmetic and numerical limits are unchanged.

- [x] Actual compiled-route observation for BOTH unchanged public replays:
  all130 default/control outputs, operand checks, exported-entry traces and
  complete before/after mapped-file sets/hashes remain exact. Each dataset
  records132 first named entries across392 stops. Default bulk enters the
  pinned tinyBLAS F32 outer/full/tail families; the other129 scopes enter
  ggml_vec_dot_f32. Static source/code pins identify one Vec8 partial/output
  versus four Vec8 partials/reduction, respectively. Only first named entries
  were observed, not every hit/instruction or private numerical arguments.
  Five actual owned-SID/pidfd/subreaper failure fixtures and actual debugger/
  inferior/group/EOF cleanup pass. First default-scheduler guard failure is
  preserved; additive explicit scheduler-locking off retains original math.
  This strengthens the fresh first64 batch-dependent arithmetic explanation;
  it does not recover the lost historical corpus, qualify full host-runtime
  closure or change production source math/tolerances.


### 6d74f76

- [x] Original CPU Phase 1 functional 16/16 through genuine 8K; bounded GPU genuine 512-token internal comparisons accepted.
- [x] Neither result closes the open independent numerical or larger-context gates.

### Accepted investigation findings — 6d74f76

- [x] Independent final-token divergence first appears in0.8B layer4 GDN and2B layer1 GDN, after preceding layers match. The separate historical AMD FMA failure is not a demonstrated CPU cause/fix.
- [x] At native ubatch512, shared-top20 logprob differences were0.20372924/0.19678696 (0.8B/2B); matching greedy7993 did not close numerical agreement.2B changes with native scheduling;0.8B did not. These partial-top20/activation observations are not full-vocabulary/KL/total-variation proof.
- [x] A large-position RoPE or padded-state fixture does not qualify a genuine long-context whole model. Canonical future-KV padding and zero signs remain source semantics.


### Full24 forced-input numerical closure — 21 Oct

- [x] ISO original-source .8B fivephase1/1/64/2/1 full24 execution: all3391488 dual-boundary BF16,1658880Conv BF16,23592960 recurrent F32,1228800activeKV BF16,1241600fullV BF16 and70656finalsum F32 words exact against independently authenticated vLLM archives; all5 greedy packets independently reduced/exact. Native1019/clears48/copies18;1008source+485tiedW windows checked;57999360futurecanonicalKV bytes preserved from actual prior. Finalresident/physical/host0.
- [x] Precise FA current-layerKVordinal1 read guard fixed after preserved firstphysicalfailure; five validated-operator tests PASS. Secondattempt failed expected-reader128MiB single ticket; fixed4×32MiB under existing API/unchangedtotal/hostcap. Fresh helper39 positive3 PASS; neither failedattempt promoted.
- [x] Currentfull24 entrycancel/pressure and afterConvunknown/beforeRecurrentcancel PASS. Actualuncertaincuts retain336statebuffers/1635values/115Wgroups/14images+head afterALLlocalsdrop, pending1/fenced2550136832B/host2302560620B;0Ready/D2H/completedphases. No recovery/unload/final0 claimed.
- [x] Freshcurrenthelper prefixlongpositive/recurrentcancel, connectedattentiongate, finalnormhead/fulltiedupload regressions PASS;10originalownedrun cleanup/pins checked in ROOTfull24-root-postvalidation-1.json. StrictcombinedClippy5,101focusedGPUhost+8greedy+4position+144core+1sourcegeometry+5FA+4expectedgreedy tests,69distinctpurevendor tests PASS. Requiredcompletegate17 PASS1320Rust/0fail/76ignored/257invocations +18Python/defaultrelease/installer/shell/Nix.
- [ ] Naturalfeedback/current-root publication and serving integration through originalSession/ResidentProgram. Forced353 diagnostics remainfrontier0/unpublished; candidate IDs[198,599,8,72,18] were not fed back. Separate source-staged per-call binding/publication unit in progress. No serving/performance/fullmodel-default-route acceptance; newimplementation remainsISOonly.
- [x] Original2B natural/all42 reference3: unchanged stock vLLM PLAIN+OBSERVED and both validators PASS; actual shortIDs[321,279], longIDs[84,2387], phases1/1/64/2/1 and genuine feedback. All320loaded source roles checked; all42active+24dualboundaries113713152B exact (9670656BF16/23592960F32), fullV/sampler paired. All24views alias ONE1062862848B backing in both lanes; original relative5GiB target unchanged. Source v3 freeze33cb4955…5214,52isolatedCPUtests and independent review PASS. Original owned cleanup, held-source closure and25primarypins before/after PASS in ROOT root-2b-natural-all42-postvalidation-3.json. Earlier pin-mapping and four-descriptor accounting failures retained; no math/cap change.
- [ ] Native2B original-source all-state/natural-feedback qualification and serving integration. Reference success does not establish Zerogravity inference, a driver hard cap or unknown transient highwater; actual baseline is separate from requested relative memory. Benchmarks/longcontexts/27B runtime remain OPEN.

- 22Oct freshattentioncapture2 retired both original lanes exit0/all56scopes64launches/full declared packets/12layouts/source33roles320/ownedCLI-group-container-EOF closure. Metadata remainsREFUSED:17projection+23FAprivate pointer candidates,7cachedcopies and3KVloader typedrefusals unresolved. Original common/all42/attention_raw pairFAIL remainsfatal. ROOT held-wholeSHA/recordSHA all24 archive diagnostic: newOBS==accepteddiscovery2OBS all3096records/166576128B; newPLAIN firstdiff layer0/call0H1002F32words(word66816), then1850records. Same8emittedIDs. OBSselectedKKT/H configurations changed but OBSoutputs areidentical; PLAINselection/firstinternaldivergence unobserved, no causal claim/newreference/math/native permit. Actual observedcall0GDN launches exist despite empty worker.selected; this does not seal all3layers/PLAIN tuples. First3 acceptedfixedOBS scope unaffected; complete47attention/fullmodel/benchmarks remainopen.

- Source diagnostic precision: capture2 preserved-input903 includes a COMPLETE1428640B b0ddb21d…420d blob and actual exactresource equality, alongside its separate64B headerdiagnostic. The earlier raw-header-only wording applies only to that diagnostic, not the whole input evidence. This does not prove KV0781 LoadData64 or DSO-carrier membership; source/native/replay flags remainfalse.

- Canonical47 semanticmetadata successor:14pureguards/source review PASS; actual retiredcapture2 revalidation exits1 with explicit REFUSED report, classifying40attention launches/23FAprivate pointers/7copies/KVregistration chain. Original three numerical failures and24projection/17workspace refusals remainfatal. Current fixed diagnostic does not grant native/math/model/perf acceptance; exact fresh-key schema guard required before generic reuse.
- Initial47call0 QKV/reducer ROOT resource inspection: wholeheld-image identity/ELF bank descriptors match actual reflected spans/base0x160; NVIDIA CPU reader confirms producer static49152 +actual dynamic24576=required73728B, reducer static0. Producer relocatable bank remains exactname/section-info join only, not a defined-symbol/linker claim. Four pure address/coverage guards PASS; reducer60..63 is unobserved ABI padding. Separate additive binding/numerical/physical gates pending; no new native/model/perf permit.

- New47 QKV fixedOBS reference prerequisite PASS:12sourceguards and actual offline source/33primary/320roles/twooriginalowners/44metadata/16wholearchives checks; values39/40/41 plus actual inputdual F32sum866304B already captured, no newobserver. New15312B/8ff503d8…fe1b receipt preserves original discovery2 and first3 receipts; all42 acceptance inherited, no newfull24 recompare. Packet source11children/52deps and allnonpointer/60B/21parameter comparisons match actual. Real33node/42value source composed in ISO; strictClippy/48canonical/90vendor checks PASS before final helper, six budget guards and independent326856780Bdevice/719156132Bhost bounds justify finite320/704MiB testing envelopes;319 is conservativeupper, not actualhandles. Two captured complete ABI seals materialized, actual partial-main faults authored. Physical33nodeQKV/source/state/fault/fullgate successor acceptedphase62 above; laterQK/KV/FA/fullmodel/serving/perf gates remainopen.

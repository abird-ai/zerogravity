# Populated Qwen HIP serving

Landing commit: `fix(hip): qualify populated Qwen serving`. [Live plan](../../TODO.md).

## Completed qualification and retained failures

- [x] Shared transport/profile checks admit up to32K only within actual source
  capacity; output remains≤512. Raw-text-only source policy now survives HTTP
  chat framing before enqueue. Affected host tests and strict builds pass.
- [x] Both models: twenty fresh SDK/private-loopback HTTP cells independently
  audited. SDK C1/4/8/16/32/64 completion, C64 cancellation/shutdown; HTTP C64
  JSON/SSE completion/disconnect plus SDK replay. Original IDs, advancing roots,
  aliases and final readers pass. Complete/HTTP-disconnect cases reach configured
  active1/4/8; larger offered populations queue. Native batch width remains1.
  Cancellation requested22 per C64 case; actual scheduler cancellation counts3
  for SDK and22 for HTTP on each model. No full-float or performance claim.
  Every cell retires native/source owners, retains only the original device
  context and final tokenizer/support host owners, unfenced. Invalid empty-
  cgroup observer sample retained; corrected exact owned-path teardown verified.
- [x] Recover completed1K/2K/4K/8K/16K originals without recapture: each model
  has1,012calls/3,720records, exact two-repeat checkpoint/logit equality and
  unchanged source/runtime pins. Independently verify5,906,781,184 /5,940,859,904
  archived bytes and2,670,267,392 /2,687,306,752 checkpoint float words.
  Nonce-bound journal commands and successful container exit/removal establish
  retired owners; original foreground reap/EOF receipts remain missing and live
  logs incomplete. Preserve that supervision gap, not a fabricated receipt.
  All ten authenticated Rust source/reference preflights pass; no GPU claim.
- [x] Complete authenticated native full-logit digests and sparse all-state
  checkpoints, SDK completion/cancel and populated JSON/SSE/disconnect. Fresh
  serial x2 campaign independently audited: eight native1K–8K passes, both16K
  full-logit digest failures and24unrun SDK/HTTP cells. Exact owned cgroup gone,
  KFD empty/GPU idle; this is cleanup, not numerical acceptance. LocalizerR2
  confirms both first failures at call216/base13824/rows64 after432 prior
  digest comparisons/model; exact localizer owner retired. That chunk contains
  the independent scalar/original-Torch cosine BF16 discrepancy at13852.
  General source sine/cosine reconstruction matches6,291,464 original CPU F32
  words; actual Rust host body and prepared GPU kernel both match all2,097,152
  RoPE F32 words. GPU64signed-zero words and original primitive retirement pass.
  Host regression/strict native+runtime builds pass. R3 independently audited:
  both models reach all six16K checkpoints,955,163,648/956,867,584 exact float
  words and448,512 KV-tail words/model, then fail diagnostic coverage accounting
  (248320-word slice shadowed by64hex chars). Both516-call loops complete their
  digest checks; in-process final retirement/source assertions were unrun.
  R3 failures retained; exact owned cgroup gone/KFD empty/GPU idle verified.
  Qualification-only fix/four host regressions, strict/release and three mixed-
  feature checks pass. Corrected nativeR4 independently verified both models:
  516calls/six checkpoints each;955,163,648/956,867,584 exact checkpoint float
  words,128,133,120 digest words and448,512 tail words/model. Final source checks
  and native/resident retirement pass; unfenced4GiB context only. Unitexit0,
  exact cgroup gone/KFD empty/GPU idle; unchanged R3 primitive proof reused.
  Runtime-only SDKr6 matrix independently audited:18passes/8failures/0unrun.
  Both models pass SDK1K/2K/4K/8K complete+cancel/replay and32K pre-native refusal.
  All four16K SDK cases hit1200s event wait; all four16K HTTP cases hit the180s
  generation deadline. Failed cases have no semantic/final-retirement record;
  parent unitexit2/exact cgroup gone/KFD empty/idle does not substitute for those.
  Review follow-up: full9728B trig allocation had a stale280B view. Production/
  diagnostic now share9636B derived from the actual table, with diagnostic extent
  check. NativeR5 independently passes2,097,152F32+64signed-zero words and native/
  resident retirement; exact unitexit0/idle. Math/table/owners unchanged; SDKr7
  built/staged, unrun. Existing evidence retains its frozen binary scope.
  Deadline diagnosisR6 independently audited: original0.8B state-only16K path
  completes258calls/three checkpoints,636,964,352 float words/744,960 digest words
  and430,080 KV-tail words, zero mismatches; native/source retirement and exact
  unitexit0/idle pass. Diagnostic elapsed740.654s; at8K, attention host intervals
  occupy2.434s of a2.945s chunk. Intervals include readback, not GPU timings or a
  benchmark. This establishes advancing execution with costly attention, not a
  worker stall. Candidate repair tiles independent query rows and bounds QK to
  actual KV tokens; per-query arithmetic, causal masking and full-call PV order
  unchanged. Scratch derives from tile geometry and stays under original caps.
  Geometry regression, native/runtime strict and release builds pass. NativeR7
  independently passes all four cells: both original66-call schedules preserve
  873,649,152/939,144,192 words and both16K trajectories preserve the R4 counts,
  all original source/native retirement assertions and exact unitexit0/idle.
  SDKr8 first16K HTTP completion still hits180s generation deadline; matrix
  stops after1failure/25unrun. No successful semantic/final-reader retirement
  record; unitexit2/exact cgroup gone/KFD empty/idle is separate cleanup evidence.
  R9 state-only diagnosis independently passes the unchanged R7 binary: same
  258calls/three checkpoints and all R6 numerical/retirement counts. Diagnostic
  elapsed350.049s versus R6's740.654s; at8K, attention still occupies0.876s of
  1.432s. Not a benchmark or serving pass. Next candidate specializes validated
  BF16 QK multiplicands: normal products are exact FP32, reusing source multiply/
  add; tiny/overflow/non-BF16/nonfinite cases retain general FMA. Actual-body host
  regression passes16,499,870 normal/6,122,580 fallback cases with exact bits;
  native/runtime strict and release builds pass. NativeR10 independently passes
  all four original-reference cells and exact original-source/native retirement.
  SDKr10/servingR11 still fails first16K HTTP completion at180s:1failed/25unrun,
  independently audited; parent cleanup is not semantic/final-reader success.
  Unchanged R10 diagnosisR12 passes all R6 numerical/retirement counts in359.314s;
  at8K attention occupies0.941s of1.452s. No demonstrated improvement over R9.
  R12 reused the old R7 shell receipt check; independent retrieval authenticated
  actual R10 executable/commands and retains that precheck limitation. Next:
  bounded diagnostic-only per-kernel timing via the existing runtime observer,
  bracketed by original producers; include outputless KV append and end only on
  the final readback chunk. R13 independently passes all198 sampled launches,
  original258-call/checkpoint counts and exact source/native/physical retirement.
  QK629,218,266ns/PV289,078,286ns dominate; serialized device-clock intervals
  include submission gaps, not GPU busy time or benchmark acceptance. R14 now
  guards native FP32 FMA/add/product to normal operands/results; denormal inputs,
  tiny/zero results, overflow and unsupported operands retain original arithmetic.
  Attention reduction/full-call PV order stays unchanged. Actual-body regression
  passes67,867,350 scalar checks with preserved denormals and the same number
  with simulated FTZ, plus ordered QK/PV sequences; zero mismatches. Native/runtime
  strict and releases pass. R14 independently accepts all4native scopes: both
  original66-call schedules and both16K516-call trajectories, unchanged numerical
  counts/IDs and exact source/native/unit/cgroup/KFD/idle retirement (audit2257).
  SDKr14 servingR15 independently audited2263:1pass(08HTTP16K JSON/SSE IDs,
  SDKreplay/final readers),1failure(2bHTTP180s deadline),24unrun; parent exit2,
  exact cgroup absent/KFDempty/idle is not failed-case semantic acceptance.
  Test548.28s spans requests, not per-request latency. Unchanged nativeR14 2b
  diagnosisR16 independently passes258calls/638,668,288checkpoint/744,960digest/
  430,080tail words,0mismatches and exact original retirement (audit2271).
  246,838ms; midpoint host incl.readback:Projection473.680ms,ResidualRms183.747ms,
  attention217.547ms. Not GPU time/benchmark. R17 groups independent projection
  and residual-RMS rows through original kernels; streamed/partial-tile fallback,
  reductions/owners preserved. Scratch explicitly priced; RMS row extents become
  bounded runtime data; source-call/source-text constructors updated. Corrected build2279
  passes4geometry/shared-backend/native/runtime strict and both releases; obsolete
  RMS weight-count constructor failure retained/fixed. NativeR17 independently
  audited2286: both original66-call schedules and both516-call16K trajectories,
  original words/IDs,0mismatches/exact retirement; remote binaries/scripts checked.
  SDKr17 servingR18 independently audited2290:0pass/1failed2bHTTP180s/25unrun;
  test205.63s/exact parent retirement, not failed-case semantics/final readers.
  Unchanged R17 2b diagnosisR19 independently audited2295: same258calls/3original
  checkpoints (638,668,288float/744,960digest/430,080tail),0mismatches/retirement;
  203,948ms. Midpoint host incl.readback:Projection371.452ms,ResidualRms16.558ms,
  attention224.646ms; not GPU/benchmark. R20 reorders independent projection groups
  for weight-row locality, retaining request-major writes/original FMA/reductions/
  flags. All four launch consumers' exact extents reviewed. Build2300 passes2actual
  mapping tests/shared-backend/native/runtime strict and both releases. R20 native
  independently audited2314: all4original scopes, exact counts/IDs,0mismatches,
  original source/native/unit/cgroup/KFD/idle retirement; actual binaries/executed
  scripts authenticated. R21 SDKr20 serving audited2319:0passed/1failed/25unrun;
  2b16KHTTP timeout180s/test205.72s. Exact parent cleanup, failed semantics/final
  readers unaccepted. R22 unchangedR20 2b diagnosis audited2325:258calls/
  638668288float/744960digest/430080tail words,0mismatch/exact retirement,194637ms.
  Midpoint host projection345549034/attention221736759/GDN122854333/QkRmsRotaryGated
  83876841ns; readback included, not device time/benchmark. R23 groups independent
  Q/K+GDN RMS head rows in existing≤64-group launches/already-priced scratch;
  unchanged reductions/gamma/rotary/gating/native images/owners. Build2327 passed
  actual range/overflow tests2+RMS geometry1, shared/native/runtime strict and both
  releases. R23 independently audited2340: all4original scopes/counts/IDs,
  0mismatches, original source/native/unit/cgroup/KFD/idle retirement; actual
  binary/executed-script identities verified. R24 SDKr23 serving audited2346:
  0passed/1failed/25unrun, first2b16KHTTPcomplete timeout180s/test206.14s; exact
  parent exit2/cgroup absent/KFD[]/busy0 but failed semantics/final readers
  unaccepted. R24 launch2341/monitor2342 ended. Watch6d6d3a4b
  handled2346;13849045/cd9be347/older watches handled. R25 actualHTTP diagnosis
  independently audited2364: load24062ms before timer;180030ms reaches255calls/
  16320of16382prompt tokens,0output/queue/parking/failures. Original runtime/native/
  tokenizer/unit/cgroup/KFD/idle retire; originalHTTPfailure101 rethrown. Not
  serving/performance acceptance; watch0fd1f67f handled. R26 batches four-row
  weight reuse in shared64-lane projection, all4callers and actual geometry tests;
  per-dot FMA/reduction/BF16/flags/owners/caps unchanged. Build2367 passed2actual
  mapping/bounds tests/shared-native-SDK strict/both releases. Actual Cargo
  artifacts frozen/staged. R26 ONE original2B16K screen independently audited
  2379:258calls/638668288float/744960digest/430080tail,0mismatches/exact original
  source-native-physical retirement;208831ms, midpoint projection471241594ns.
  REJECTED for deadline goal; no full4native/serving promotion. Diagnostic
  includes readback, not device time/benchmark. Watches4b79a306/4dd0bb2e handled.
  R27 removes only R26tiling/grids, preserves accepted R20mapping/R17/R23support;
  folds integer flags in existing32/16/8 stages/barriers, then8masks instead64.
  Float math/BF16/+0/statusbits/owners/caps unchanged. Build2386 passes3actual
  mapping/bounds/native-body flag tests/shared-native-SDK strict/both releases.
  Actual Cargo artifacts frozen/staged. R27 ONE original2B16Kscreen independently
  audited2397:258calls/638668288float/744960digest/430080tail,0mismatches/exact
  original retirement+identities,178825ms. Narrow margin; diagnostic/readback,
  not HTTP/device-time/benchmark acceptance. UnchangedR27 full4native R28
  independently accepted2406: both66-call schedules+both516-call16K trajectories,
  original words/IDs,0mismatches, source/native/exactphysical retirement and
  actualexecutables/receipts/scripts verified. R29 full26serving independently
  accepted by audit2417:24native SDK/HTTP cases and2pre-native32K refusals,
  26passed/0failed/0unrun. Both models pass SDK1K–16K complete/cancel/replay and
  16K HTTP JSON/SSE/disconnect/replay. Both16K SDK cancellations observed exactly
  once;7accepted output tokens/49150model-token steps. Original IDs, roots,
  aliases, counters and every original native/source/tokenizer final reader pass.
  Same R27 executables/receipt/scripts authenticated; HTTP180s/SDK1200s unchanged.
  Monitor2409 stopped before final receipt; historical terminal file is absent.
  Explicit later observation verifies the SAME invocation's exit0, exact cgroup
  absent/KFD[]/busy0. Audit retains `supervision_gap_retained=true` and
  `terminal_observation=post_monitor_stop`; current cleanup never substitutes
  for the per-case semantics/readers. No completed campaign/audit was repeated;
  watchc4ebba31 is absent, not armed. Original capture/R12 limitations retained.
  Cadence: actual bottleneck, assembled repair, one timing/state viability screen,
  then mandatory native/serving acceptance; no per-helper or speculative cycles.
  No position/value patch, tolerance change or expected-output change.
  HTTP has two sequential requests, not concurrent C2. Full intermediate-state
  and performance claims remain excluded.
- [x] Both fresh32K SDK controls refuse before native creation and retire source
  readers; no device budget/adapter/model. R3 passes exact host total, reservation
  count, allocation-class and model-owner equality; final owned cgroup absent,
  KFD empty/GPU idle. Retained R1(.8B), R2(both) baseline-accounting failures and
  initially unrun2B R1. Their1,950B delta was the original process-retained
  serving thread-control root, not Runtime metadata. Baseline initializes that
  same owner; no numeric allowance, budget increase or production change.
- [x] Complete assembled source review and final isolated gate. GateR1 exposed
  five stale count/groups arguments in separate `bf16_text` RMS preparation;
  all five repaired without changing geometry, kernels or the qualified Qwen path.
  Complete isolated gateR2 passes:2350Rust/0failed/101ignored,309suites including
  74empty;17Python suites/182cases/10skips. Earlier gate failure retained.

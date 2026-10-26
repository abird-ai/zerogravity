# Cpu gpu optimization

Provenance commit: `6d74f76`.
Scope: P01/P06/P07/P20/P22; follow acquisition and measure before selecting complexity.
Contracts: [fleet optimization](../../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md),
[GPU text/control](../../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md),
[FP32 FMA proposal](../../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md),
[derived quant contract](../../docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md).
Research: [independent engine mechanism comparison](../../research/engine-optimization-comparison-2026-10-21/README.md)
(416 overlapping source-scoped entries; research is not runtime qualification).
Current measurements: [short-HF results and profile limits](SHORT_HF_RESULTS.md).
Shared allocation/boundary direction: [reconciled pool, layout and transfer APIs](SHARED_MEMORY_AND_BOUNDARIES.md).
Architecture: [Astra high shared/family/model boundary review](../pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md).

## Uncommitted

### Current correctness-first order — 26 Oct

- [~] [Shared vLLM/llama.cpp correctness policy](../numerics-and-context/CORRECTNESS_POLICY.md)
  across all models; .8B initial/cached47–50, then genuine long contexts and
  CPU/CUDA/ROCm/Vulkan. All agents GPT-6.1 Sol xhigh. Earlier B4/C16/27B order
  below is historical; no new performance or27B capture before correctness.
- [ ] Preserve case5 FAIL101 and quarantine. Original default vLLM N1/B4 also
  has the same six QKVZ BF16 differences at frontier47 with identical inputs,
  input RMS and48 prior planes. Width-specific original agreement and model
  accuracy are distinct; no arithmetic substitution or failed-run relabel.

### Current B4 takeover — 25–26 Oct

- [ ] Current B4 → fresh optimized matched C16 (.8B +2B) → x2 original27B,
  latest owner order; no C2. Base `d0fff261`, all246 signatures verified;
  README unchanged. All current lanes: GPT-6.1 Sol xhigh, owner override.
  [Owned run](../../runs/b4-takeover-codex-20261025/TODO.md).
  [Final evidence index](../../runs/b4-takeover-codex-20261025/FINAL_REPORT.md).

#### a857d45 — validated fixture repairs

- Current l5 route legitimately reads all23 original model members and approved
  CUDA/source-reference inputs; no grant change/download. Original relay denial
  remains historical. A3 source1772 digest
  `3e336ffdc40efa93a46bd6d86584b016365b8d087e8ce39d72a7b4e667856fea`.
  Canonical opt3/thinLTO/codegen1 build CLOSED0; fmt,3transport+7typed pass.
  CLI `871e9fed…7117`, N1 `1679b955…8c3b`, B4 `00c6d4d7…7875`;
  exact CompilerArtifact/frozen alias lineage retained. Full isolated
  `scripts/check.sh` CLOSED0 (1890 passing Rust executions,89 ignored; source/
  installer/script gates pass), original TERM/KILL/single-reap/group/EOF closure,
  source/README stable. Serving-SDK qualification remains separate.
- Reviewed fixture fixes release completed access/readback/output aliases;
  retain every historical case plus finite/sticky/isolated-status/guard controls.
  Current grouped oracle physically PASS58grouped/232per-row, all12 independent
  mode closures resident0/events0/arbiter0/unfenced. CPU4344 + finite301 PASS.
- HOST fixtures use already declared8GiB and independently priced permanent
  tokenizer baseline, exact ledger/class/reservation closure; GPU quotas and
  mathematical predicates unchanged. A2 baseline B4 and N1 both FAIL101 under
  HOST4; actual B4 HOST5481465815B >4294967296B. Preserve both failures;
  historical queue cost is a distinct domain. A2 source1771 is separate.
- All original26 executed on A3:24 physical/CPU passes, case6 diagnostic2B
  export0 (unqualified), case5 continuous N1/B4 comparison FAIL101. All26
  process owners closed with actual EOF/TERM/KILL/single-reap/group absence.
  Case5 first projection differs; reported producer9 bytes differ and error
  retirement retains2469431001B/pending1/rootsets9/unusable. In-process close
  FAILED; process closure does not erase it. Exact original23 projection passes;
  this is explicitly NOT all26 green or complete continuous qualification.
- Separate actual .8B15-transition/35-event export CLOSED0, healthy source/model
  retirement resident0/events0/rootsets0. Both-family full original30 comparisons
  now CLOSED0: each15transitions/35events/4344current+8112original sections,
  576unrounded sums/fullstate/fullV/natural feedback exact, no ID normalization.
  This is width/schedule-specific composition, not an independent original
  public trajectory. Case5 error is deliberately fenced before Ready; no unsafe
  retirement patch. Exact26 remains false.

#### SDK replay and serving — unchanged a857d45 runtime

- Initial SDK qualification seam: physical proofs used overlay NVRTC7c67c6b5…,
  while unchanged original serving recipe pins Nix12.9.86 NVRTC4b8c4db1…;
  builtins also differ. Runtime compiles through NVRTC; matching version labels,
  counters or final tokens do not prove full-tuple equivalence. Serving-SDK
  physical/math replay preparation underway, with exact frozen A3 binary aliases
  and original owners; no substitution/retag. First Nix grouped test native0
  passes all12modes/58grouped232per-row; SDK identity observer REFUSES
  Btrfs maps-device0:36 versus stat-device0:38 (inode matches). Outer1,
  adapter0/test0, complete original owner closure. Preserve source55edcd92
  and this rejected attempt. Reviewed V2 retains held DSO/directory FDs,
  read-only Btrfs mount/filesystem identity, kernel VMA link/inode and stable
  namespaces/root. Root28 metadata controls and isolated CPU probe pass; no
  permission changes or blind device-check waiver. Fresh V2 case10 also native0/
  adapter0/outer1: all required actual DSOs witnessed, healthy process closure
  and postchecks, but two `/proc` observation errors (ENOENT/EINVAL) remain
  latched. V2 sources473156b3 and rejected receipts are immutable. Focused CPU
  lifecycle investigation must establish the observation stages before another
  adapter/replay; no missing identity or interrupted snapshot is accepted.
  CPU probes subsequently reproduced terminal and live VMA races; V3 stages
  snapshots transactionally and requires original WNOWAIT proof for terminal
  exceptions. Its41 controls/10 CPU lifecycles pass, but actual case10 again
  native0/adapter0/outer1:54 complete snapshots precede a libcuda VMA error and
  four child-root errors classified live. Source/post/FD closure passes; SDK
  remains unqualified. Preserve V3 `dc912480`, driver `f1f00a50` and actual
  root observation `426cfb77`. Review bounded live identity measurement and
  exact loader retention before another adapter; do not guess teardown cause.
  V4 explicitly bounds identity to one simultaneous live snapshot, with all SDK
  supply FDs held and checked through original cleanup. Source review and51
  controls pass; actual case10 still native0/adapter0/outer1: two earlier VMA-link
  ENOENT errors prevent establishment despite52 successful snapshots. Preserve
  sources `f289996d`, driver `4ba87ed6`, root observation `87638c6a`; SDK remains
  false. Investigate acquisition-time topology changes with independent source
  review and CPU evidence before another physical attempt. No retrospective
  classification or qualification of the rejected V4 run.
  V5 `f79a08a2` validates a separately witnessed acquisition discard, with no
  mappings/count credit, then requires a fresh stable simultaneous3 checkpoint.
  Held SDK/DIRFD SHA/identity checks and original process ownership continue
  through cleanup. ROOT+independent review,61 controls and actual integrated CPU
  lifecycle pass; all four older SDK attempts remain false. Fresh Nix-SDK exact26
  now executed:24 physical/CPU passes, case6 diagnostic export0, case5 native101
  with numerical and in-process retirement failure. All26 SDK/source/process
  closures pass; exact original23 projection passes. Separate .8B export0 and
  both full original30 comparisons pass on this tuple:15transitions/35events,
  4344current+8112original sections/576unrounded sums/allstate/fullV/own roots,
  no identifier normalization. Original26 and complete continuous certification
  remain false. Fresh both-model matched API/HTTP checks precede any C16 release;
  no current timing result or speedup claim.
  First API08 refuses the old23-key health schema against current Rust26 fields;
  actual outer1, original container/process/group/log closure healthy, no warmup.
  API2B stopped before provider launch (outer-15); both remain unqualified.
  External HTTPv6 strict current-schema adapter preserves raw observations,
  HOST model+backend+serving conservation, DEVICE, compiler/readiness and source
  initialization predicates. ROOT38HTTP+18C16 source controls pass; independent
  review clean. Separate source review confirms direct case1 genuine .8B
  initial4, both-family five-frontier N1 and original30 cached composition cover
  the declared finite request recipes; no original26/fullcontinuous grant.
  Fresh v6 .8B API CLOSED0:24requests/192tokens per provider exact, cold/whole/
  workload compiler0, both original owners/source/postproof closures healthy.
  API2B also CLOSED0 with the same24requests/192tokens per provider, compiler0
  and complete original owner/source/postproof closure. Fresh HTTP08 CLOSED0:
  actual N1→mixed cached4/refill, all5requests/36tokens fresh-stock exact,
  cold/whole/workload compiler0 and original owner/source/postproof closure.
  HTTP2B also CLOSED0 with the same five-request/36-token chronology, fresh
  stock equality, compiler0 and complete owner/source/postproof closure.
  ROOT released the exact reviewed finite-profile C16 commands for both models.
  Both fresh quiet SAME-ATTEMPT stock→ZG C16 populations now CLOSED; original
  rate AND wall predicates unchanged. Original26/complete continuous remain false.
  First controller refuses coordinator's overlapping output layout before any
  provider/timing; original record retained. Corrected disjoint sibling outputs
  pass the original guard for both models; source/profiles/qualification unchanged.
  A3 cudarc bindings select `cuda-13040`; fallback-latest is enabled, but the
  original selection cause is not independently captured. This is a separate
  identity from observed NVRTC12.9 and driver595.91.07. No ABI change
  or general compatibility grant follows from passing physical scopes.
  Full source gate passes. Fresh v6 serving API/HTTP pass for both models;
  original runtime P16 and matched provider P4 remain distinct.

#### Closed C16 measurements and remaining work — 26 Oct

- Actual corrected C16 ROOT64250 CLOSED1/full stdout+stderr EOF retrieved;
  .8B controller0,2B controller1. All four original provider owners/container/
  process-group/single-reap/full-log closures healthy; full original30 source
  proofs revalidated after each provider group. Each arm completed16 offers/
  128tokens,0errors, client peak16/native peak8, all native compiler intervals0.
  A3 opt3/thinLTO/codegen1 source1772 and frozen CLI871e9fed…7117 unchanged.
  No C2/C32/C64, extra samplers, predicate waiver or asynchronous completion.
- .8B stock98.2827tok/s,1.30237s; ZG111.1738tok/s,1.15135s. Original rate AND
  wall pass. Four genuine INITIAL4 publications +28cached B4 publications,
  16 independent initial roots/128committed rows/864physical model token steps.
- 2B stock96.8627tok/s,1.32146s; ZG84.1017tok/s,1.52197s. Original rate refusal
  retained; pair-index/final remain INCOMPLETE/false despite both measured arms
  and healthy cleanup.16 genuine N1 initializations +28cached B4 publications,
  128committed rows/864physical steps. No2B performance pass or both-model win.
  Single measured wave/model, no confidence-certified performance claim.
  [Both closed review](../../runs/b4-takeover-codex-20261025/matched-c16-v6/root-both-closed-review.json);
  actual-results SHA58ec6b242aac7cb4c3f07c41b29af32ffdb78b91b98e98680aa5c58dc769df79.
- Strict x2 SSH now actually verifies existing alias/address ED25519
  `SHA256:acPQLnW/MfwchrrYkM/TB9eVeTWrnaA5efNPrEJZ2U0`, via prescribed
  100.100.1.1/tailscale0 route; known_hosts bytes unchanged. NativeUID1000,
  machine8280a2085c2247e4bb49a5ac1bcd1582/boot40a8634b-bd7c-4cb0-b127-d0fde2a4f37d.
  Current kernel7.2.10/AMD MAX+395; CPU-visible33263382528B total,
  15343521792B available <16GiB gate; sysfs VRAM103079215104B is separate.
  Disk1769526714368B available. All31 exact Q27 members+source parent and exact
  image store/runroot absent. Existing host Nix Python3.13 metadata/RX verified,
  SHA85ee2b9c…a4a5, not executed; prescribed/PATH python3 absent. No device open,
  inference, downloads, store/recovery/privilege changes or memory repartition.
  [Current resource review](../../runs/b4-takeover-codex-20261025/x2-current-preflight/root-resource-review.json).
- [l5 benchmark container termination](BENCHMARK_CONTAINER_TERMINATION.md):
  original55 artifact pins revalidated;16original Rust files+Cargo manifest
  authenticated. Fresh corrected17-row journal query adds actual stop-actor
  scope start; container death follows by5.288610s on wall/boot-monotonic clocks.
  The pinned application ran as PID1 without a serving shutdown handler;
  owned cleanup used `stop --time 5`. Cleanup-timeout escalation is the supported
  leading inference. Later foreground PGID cleanup cannot explain the earlier
  container death. Fatal signal/sender and OOM remain unproven; hold stays.
  Native status137, stock0, both16/16 measured requests and healthy independent
  owner closures remain distinct facts. Exact-actor follow-up has2rows, no
  fatal-signal evidence; the earlier empty kernel query proves no OOM absence.
  That forensic review invoked no Podman, model/device execution or SSH.
  Preventive shutdown/evidence fixes now land as50abaf7/c84b987: real model-free
  process/PID1 controls and actual cached-image Podman stop pass0/no escalation,
  ledger0/exactCIDabsent; full gate0. This fixes new-source shutdown, not the
  historical missing signal/OOM trace. Original-model/backend qualification and
  capture holds remain separate; no benchmark or inference grant.
- Q27 frozen original9 revalidated; current owner-readback caller54641525 differs
  from frozen ae75a695. Unchanged validator calls source_open(plan,pins), while
  repaired caller requires state: independent validation ABI blocker. Preserve
  both authorities; next repair must be explicit owner-layer source, retaining
  all math/inputs/helper/resource/gates, with focused lifecycle/validator checks.
  Additional isolated natural-window source fixture CLOSED0:11tests pass,
  all source pins stable. It does not exercise this caller/validator ABI or
  establish capture/inference. Full runtime source gate remains the A3 result.
- Next: diagnose original26 case5 numerical/error-retirement failure without
  relaxing comparisons; review2B dispatch cost with phase evidence before any
  optimization; retain the container-termination hold pending actual signal/OOM
  evidence; replay required model/backend shutdown against the new source before
  future benchmarks, then establish the
  already-authorized exact x2 image+31 model files, verify a genuine CPU/RAM/
  GPU-owner window and all G1–G6 gates, then supervise original natural27B
  capture+independent validation. Capture remains unstarted; no short substitute.

### Current source checkpoint — 24 Oct

- [~] User extends both-model sequence to C16, thenC32 andC64; NO C2. Implement
  genuine populations and honest active8/B4 observations. Latest user direction:
  resume all parallel tracks now. The78 signed logical commits are complete;
  C16 correction, Qwen27B, Gemma4, Vulkan, HIP and shared queue have independent
  Pi lanes. ROOT coordinates builds/physical ownership and quiet timings.
  Default pending count unlimited, optional configured cap, actual HOST
  admission unchanged. C32/C64 follow queue acceptance. C16 correction and
  grouped controls run separately from product source; old drafts preserved.
  Audit/store latest and best verified matching historical C1/C4/C8 plus new
  bands, with source/profile and stock arm from SAMEattempt, no cherry-picking.
- Historical source20 first-use refusal remains failed:446→448 attempts after
  readiness. Source21 fixes the two observed state recipes and passes actual
  both-model API/HTTP/C1/C4/C8 with cold/whole compiler0; see Committed above.
  Preserve original failed20/source15 artifacts; no warmed substitution.
- [~] C16 both executed on frozen21. Clean2B: ZG78.7859tok/s,1.62466s;
  vLLM95.9974tok/s,1.33337s. Each16offers/16completed/128tokens/0errors,
  original semantic/source/owner closure healthy, actual active8/B4 and compiler0.
  Rate refusal is a measured performance loss, not numerical/cleanup failure.
  First.8B timing overlaps audit scan; one repeat105.7331vs96.6544tok/s has an
  extra sampler and remains diagnostic, not quiet/best evidence. No third rerun.
  [Scope review](../../runs/resume-all-parallel-source-supervision-sol-20261024/C16_FINAL_SCOPE_REVIEW.json).
Historical source13 checkpoint (superseded by source20 above): ROOT adopted models118 plus source-bound2B
cachedB4 as one guarded173path unit (172effective/46new), fresh mtimes and
README/lock preserved. Full1629/selected492 source equality verified MAIN/ISO/
donor; optimized joint CLI/N1/B4 release13 CLOSED0, both-family actual nonlaunch
admission CLOSED0. Current23 CLOSED1: original19 and genuine2B trajectory/export/
healthyretirement pass; first new2B cached-conv fault fails an unchanged compound
retention assertion without its actual snapshot, last2 faults unrun. Source-first
fixture diagnosis continues; original SAMEPRIOR comparison of the healthy
closed trajectory now passes all15transitions/30plain+observed calls,8112sections/
576unrounded sums/fullstate/fullV/natural bits. Source13 only; serving/benchmarks
stay held until fault qualification. Combined strict,20sharedCPU,
5ABI-source controls, format and authentic2B inventory/reopen pass; excluded
vendor-test Cargo refusals stay explicit. Source-derived provider upper
4,767,472,640B within4608MiB and HOST4,080,276,698B within4GiB; fullhead remains
resident and only cachedB4 is admitted for2B. Its actual successful trajectory
fits this finite admission and retires healthy; whole fault/math/serving/speed
qualification remains open. Diagnostic14 only prints actual fault snapshots,
retains all assertions and changes no production source. Source11/12 below
remains the last accepted measured checkpoint.

Earlier numerical checkpoint24Oct: fixture-only source8/412paths exported all15
genuine transitions/35events/1,648,529,280B, with healthy source/native retirement
and final host/device0. Original same-prior reference3 now PASS: all15transitions/
30plain+observed calls, fullstate/fullV/24sums/natural bits and all controls exact.
Loaded320roles unchanged; final source/engine/process/container/group/EOF closure
healthy. Explicit state-initialized compositional proof, not a separately generated
original public trajectory. Earlier adapter failures and cross-width N1 diagnostic
remain preserved. Fresh API8 PASS24offers192tokens/provider, wave16actualB4
flights/64rows; ragged N1. Separate event-driven HTTPv2 now PASS: all5 genuine
N1prefills, A→mixed cachedB4, completedA→replacementE/refilledB4,5publications/
20rows/36natural tokens exact against actual same-cap original stock outputs.
All original source/owners close; v1's metadata-only adapter failure preserved.
Finite A4+BCD/E8 correctness scope, not a timing or general cancellation grant.
Quiet C1/C4
CLOSED0: ZG66.7151/17.5611 versus vLLM37.3407/49.0326tok/s. All32offers256tokens,
four original owner/source closures exact; singlepass preliminary. C4 regresses,
C8 hard-held. Separate direct diagnostic4 CLOSED0:185compiler trips/1480spans,
all8offers64tokens/source/owner/clock closure exact. In its3.72236s wave,79first-use
trips consume3.10167–3.10218s parent union; NVRTC2.18533–2.18585s and module load
.84863–.84915s are nested. Later decode/second wave have0compiler spans. This is
instrumented attribution, not a fraction/updated result of the quiet benchmark.
Shared admitted exact-recipe readiness now passes14distinct CPU contracts and
intended all-target strict configurations; independent Sol review finds no
must-fix. ROOT merged52guarded paths with fresh mtimes into MAIN/ISO source9
(420selected/1583complete build files); rootREADME unchanged. Optimized joint
CLI/N1/B4 release9 CLOSED0. Its actual18-case physical batch CLOSED1:16 cases
refused before inference at the compiler ABI guard; two N1 matrices first passed
their301 standalone finite checks, then correctly refused installing a fresh
compiler policy on that already-used actor. All18 original owner/group/one-reap/
fullEOF closures healthy. Installed CubeCL deliberately sets
`CompiledKernel.io=None` for handwritten kernels (conservative access analysis),
while the private PinnedKernel factory already enforces physical ABI<=32. The
new policy conflated optional access metadata with the actual ABI. Reviewed
four-path successor10 now merged MAIN/ISO; nine contracts and intended full
integration strict pass. ROOT verified all1583build files MAIN/frozenWORK plus
420selected MAIN/ISO and unchanged rootREADME; optimized joint build88097 CLOSED0.
The unchanged301-case qualifier now has its own original-owned fresh process;
reviewed19-case adapter requires its success/closure before all18model scopes.
Preserve release9/failures. Actual19physics97239 CLOSED1: standalone301, both
whole numerical matrices and all4B4 scopes PASS;12N1faults stop at stale fixture
accounting after expected retained/fenced/no-publication outcomes. All19 original
owner/group/one-reap/fullEOF closures healthy. N1 deliberately admits4608MiB
provider even for.8B; actual512MiB backend adds to5GiB, rather than the old
provider-only expectation. Reviewed fixture-only successor11 derives exact sum
from actual healthy leases, checks bothDEVICEowners/classes plus originalHOST
source/backend owners before/after. One guarded file applied MAIN/ISO/WORK;
optimized affected N1build93562 CLOSED0, unchanged CLI10/B4/libraries reused
byte-exactly. Fresh19batch25323 CLOSED0: all19 source/state/fault/lifetime scopes
and original owners pass; all12 exact HOST/DEVICE fault ledgers emitted. API94011
CLOSED0:24offers192tokens/provider exact, original sources/owners close. Genuine
HTTP70470 CLOSED0:5N1prefills→mixedB4/refill,5offers36natural tokens exact;
same-cap actual stock reuse explicit, compiler attempts/keysource/PTX deltas0.
Quiet current11 C1/C4 unit24047 CLOSED0: ZG67.7996/101.984 versus
vLLM37.4208/49.023tok/s; C8 gate passed, unit39371 CLOSED0:100.46vs61.9783.
All48offers384tokens/source/sixoriginalowners exact, compilerattempt/keysource/
PTX delta0; actualC4/C8 each16B4flights64rows,C8peak8. Single-pass preliminary,
pinned0.31 eager/BF16 recipe, no full parity/confidence grant. Fullgate17 stops
only at owned fixture formatting; rustfmt-only source12 applied MAIN/ISO/WORK,
CLI11/production unchanged. Fullgate18session56242 CLOSED0:1756Rust/0failed/
89explicitignored/271suites/allscriptchecks; full1583MAIN/WORK and420selectedISO/
README unchanged after gate. Current.8B checkpoint accepted within its finite
source/profile/singlepass scope. No budget or
production change from fixture correction/formatting. Isolated2B source-bound
cachedB4 family specialization now active; N1prefill/partialfallback retained,
authentic2B RMS/projection/catalog/reference and exact fit required.
Original2B B4 discovery passes all8calls/all24boundaries/all42state domains.
Capture1's exact fullimage layout refusal64/48absent FApackets stays failed.
Reviewed capture-only catalog pins54discovery+twoactual late-reflected fullimages;
fresh capture2 CLOSED0, all selected arguments present/driverfail0/all8scopes/
fullV/natural4×8 plain-observed exact. ROOT verifies source320/632/fullpayload,
all64original package members and both original process/container/group/reap/
fullEOF closures. No image hash normalization. Expected-only original reference
and complete native metadata do not grant ZG math/fit/serving or N1→B4
continuation; the shared2B implementation and independent SAMEPRIOR proof remain
open. Original failed capture is unchanged.
No new physical/performance grant. Actual detached
backend384MiB HOST/512MiB DEVICE uses the original domains; model quotas remain
unchanged. Exact pooled RMS definitions replace duplicated source metadata;
anchored bounds fit, fresh live inventory remains required. Live same-actor
compiler counters and actual retained lease charges are exposed for qualification.
CLI6 measurements above are historical; current CLI10/11 results and fullgate18
are the accepted checkpoint. Run the next complete gate last on the assembled
2B/model contribution; gate16 covers preceding316 only.

### Ordered adoption — 21 Oct

Owner-authorized optimization is active for physically qualified short-HF
profiles; wider profiles require their own source/state/natural-serving parity.
Qwen27B/Gemma integration continues independently in parallel worktrees. Keep
large end-to-end units and distinct claims; run the full gate last on the
assembled batch rather than after each helper.

Stronger22Oct batching clarification re-read/acknowledged: related implementations
and parallel source contributions assemble before coordinated physical/HTTP
validation; collect related failures and rerun affected coverage. Next complete
check.sh follows the assembled portable-arithmetic/Gemma-contract/CUDA work,
after meaningful physical checks settle. Internal diagnostics and focused
compile guards resolve concrete blockers; no separate per-helper acceptance
cycle. Reuse unchanged oracle/source/lifecycle evidence.

Current owner order (23 Oct, supersedes older target/timing order below):

Owner's latest merge sequence:

1. Finish shared pools, transfer retirement rings and related C8 optimizations;
   qualify the assembled source and run fresh matched .8B C1/C4/C8 timings.
2. Resolve full2B head residency within the declared profile and integrate the
   Astra shared engine/model operations/family/model architecture changes.
3. Merge genuine BF16/HF continuous request batching; implementation proceeds
   independently now, preserving independent roots and partial-batch fallback.
4. Rerun matched benchmarks after batching: .8B first, include2B if ready.
5. Finish2B correctness/performance and add C16. C16 requires a real16-request
   workload and explicit server/admission profile; the old8-offer/active8
   protocol cannot establish C16 support by changing only client concurrency.
6. Merge Vulkan/HIP, Qwen27B and Gemma4 contained units as each becomes ready;
   continue isolated implementation now, without waiting for benchmark closure.

Three Sol HIGH implementation slots: prepared recipe/late binding; Qwen27B+
Gemma/HIP native ownership; genuine BF16 batching. Astra HIGH is architecture,
planning and redesign only. ROOT coordinates source merges and serial physical/
timing ownership.
Extra independent lanes stay queued when the three slots are occupied.

Owner resumes all tiers in parallel: shared physical13 is now guarded/integrated
MAIN/ISO345, with original Gemma/math/modes/README preserved. One physical owner,
summed explicit model quotas, detached authentic source/native retention and
namespace accounting; logical564/0/58ignored/provider+default strict PASS. Astra
architecture-only review accepts/no must-fix. Required local release2 and both
models' full original CUDA/source/state/all5fault/API scopes PASS, including
explicit streamed2B head faults. x2 native3/portable6/rawtransport12 PASS;
corrected shared4 HIP/Vulkan healthy/unknown PASS. Initial fixture Oracle
shadowing failures and interrupted original controller remain preserved; fresh
case13 replay restores coverage without inventing lost local closure. ROOT
reviewed/merged only fixture lifetime correction as345v3; production unchanged.
B4 body/policy/worker and Qwen27B native consumer continue in parallel; next
Astra residency/set-key pass is architecture only. Final assembled gate pending;
measured release stays316. [Physical scope](../../runs/cpu-gpu-optimization-physical-owner-root-20261023/TODO.md).

Current MAIN/ISO410 assembles continuous34 and compiler-lifetime7. Dynamic KV
uses two direction-only keys with existing readonly controls; genuine canonical47
N1 roots can enter mixed cached47..53 cohorts/refill. Grouped transport/key/bounds,
CUDA/default strict and compiler library562/lifetime7/CUDA-BLAS strict PASS;
Astra focused review finds no must-fix. Complete compiler units remain attested.
ROOT optimized joint release5 PASS. Coordinated actual finite B4/state/lifecycle/
zero-retirement and3unknown cuts PASS; both .8B/2B N1 full matrices/faults PASS.
The source5 mixed N1→B4 RMS equal-frontier refusal is corrected in source6 by
one shared input-geometry validator, preserving each source/root/permit. Grouped
strict checks and all2401 cached-frontier shape combinations pass. Optimized
release6 and affected finite B4/lifecycle/zero-retirement/3unknown cuts pass.
Actual mixed[47,48,49,50] now reaches arithmetic, then fails byte comparison of
same-prior N1 versus B4 at SSA value9 (layer0 post-attention residual; earliest
divergence then unresolved). All5 original process/group/one-reap/fullEOF
closures are verified; failed callback retains original fence. Sol traces the
lowest responsible operation; no tolerance/oracle substitution or continuous/
API/performance grant. Fresh API/quiet comparisons follow corrected fullstate
qualification. Independent models113 passes5strict cuts/907CPU/2comparators on
400v4; contextual410+113 affected strict/166boundary/7compiler-owner checks also
pass with all jobs closed. RMS3/source6 remains excluded from that donor proof.

Fixture-only source7 diagnostic now isolates the earliest observed N1/B4
difference at QKVZ projection2, before Conv/GDN: all4 embedding/InputRMS operands
and48 consumed physical prior planes/row exact; QKVZ differs6/7/4/1BF16 words.
Original owner/group/EOF closes completely. This does not establish original
B4 correctness. Sol prepares a bounded expected-only export plus original-runner
state-initialized same-prior continuation, sequentially after ZG closes. Exact
full24/24sums/42prior+candidate/fullV/natural checks remain required. No captured
values enter production or grant independent public-original trajectory parity.

Latest measured finite B4 source is400v4. Cold initialization
now performs a real admitted zero upload and preserves original initialized
Write/completion guards; complete source bound host4,285,247,927B and
provider2,935,637,504B stay under unchanged4GiB caps. Required release2/3 PASS;
release3 changes only the fixture's truthful two-reducer+four-packet-copy count,
CLI/N1 binaries byte-identical. Actual B188 prefill plus7cached calls PASS all24
boundaries/all42 priors+candidates/24unrounded dual sums/fullV/32natural outputs,
and capacity/precancel/latecancel/stale/all-none/lineage checks. All3 submitted
cached unknown scopes PASS original quarantine/fence/owner closure. Full matched
API PASS24offers192tokens perprovider including ragged/full source and process
closure; wave8 performs16actualB4 flights/64fused rows, ragged remains N1.
The six-path correction retires B4 records on the same original armed source
flight before the shared prepared set. Fresh release4 actual matrix/lifecycle,
healthy zero-retirement and all3 submitted unknown cuts PASS; host/device final0
on healthy close. Fresh API24offers192tokens/provider PASS with16realB4 flights/
64fused rows in wave8. Original failed retirement results remain preserved.
Quiet singlepass C1 ZG67.2592 vsstock37.1156tok/s; **C4 regresses to11.7555 vs
49.4752**. All32offers256tokens exact and four original owner/source closures
healthy. C8 remains held until C4 superiority; no profile promotion/full parity.
First four C4 requests take~5.128s versus second four~0.315s; source proves lazy
compilation, including60 position-specialized KV recipes. This is not a measured
compiler wall fraction. The Basic diagnostic matrix passes but stores no events:
with compilation caching off, records remain unpersisted observations. No cache
policy change or compile fraction is inferred. Dynamic controls reduce KV to two
recipe keys; isolated meaningful checks are underway with continuous33.
Genuine canonical47 N1→mixedcached47..53/refill/cancel source is frozen33paths;
CPU/strict/worker checks PASS, physical qualification pending. Wider prompts stay
N1 pending authentic native qualification. Final assembled gate LAST; gate16
still accepts preceding316 only.
[B4 unit](../../runs/cpu-gpu-optimization-cohort-b4-root-20261023/TODO.md).

- [ ] Shared compiler lifetime/readiness: fix checked all-exit raw NVRTC cleanup,
  retain/fence uncertain destruction and partial module loads; establish actual
  context-lifetime host/device ownership before eager exact-recipe preparation.
  No global DryRun, dummy roots or unpriced cache. Existing model-local128MiB
  loader is host scratch, not a persistent CUDA context charge. Numeric ceilings/
  complete fit remain unresolved; Sol HIGH implements the contained foundation.
  [Astra architecture](../../runs/architecture-astra-compiler-readiness-20261024/DESIGN.md).
- [ ] After current C8/refill and source113 qualification: shared prepared matrix/
  compact IO reuse, temporal work, optimized eligible GEMM/attention/GDN, pages/
  full-domain prefix/COW, admitted prefetch and graph replay. Preserve numerical
  profiles; source113 CPU-order dot is not TensorCore/BLAS parity. Family and new
  model gaps are queued in model-execution; no hardware eligibility by name.

Gemma44 diagnostic source now integrated MAIN/ISO343 after binder316 gate16.
Frozen MAIN314+44 actual x2 HIP matrix PASS7cases/55completed/52commits/all35/
fullV/all15KV, zero mismatches; fresh unknown0roots/full16GiB fence and original
source/process closures PASS. One1MiB synchronous source page remains a
correctness baseline. New source-neutral physical owner and simultaneous
prepared sets need their own actual checks; no Gemma serving/performance grant.
Current HIP actor still refuses a second live native set; bounded lifecycle
extension remains in the next shared native-consumer unit.
The [Astra follow-up](../../runs/architecture-astra-shared-native-20261023/DESIGN.md)
selects nonzero, never-reused logical streams through every actor cache/IO/driver
lookup, immutable source-service Client views, explicit cold32GiBprovider/
64GiBdevice/32GiBhost with two12GiB quotas and conservative compiler retention.
These are provisional design ceilings; complete cost/actual fit is unproved.
The [Astra residency/set-key follow-up](../../runs/architecture-astra-residency-and-set-key-20261023/DESIGN.md)
accepts a private key carrying logical set plus the FULL original KernelId,
bounded full-Eq actual-driver routing and immutable SingleCompat/Scoped modes.
Sol HIGH implements separately from single-source Qwen numerics. Gemma's next
parallel unit retains repeated projection weights, widens exact row launches,
skips optional diagnostics before readback construction and uses original private
feedback/publication. Exact source selection/complete cold peaks precede fixed
profile admission; old16GiB proof remains distinct. Tied head768MiB plus2MiB
scratch is design arithmetic only; PLE full table4480MiB is not needed to stop
repeated projection uploads. No measured speedup or new fit claim.

Latest accepted shared-head extraction:290selected source paths, real exact
issuer/member/expert-range binding; both models' complete physical/state/fault/
API checks PASS. Full isolated gate13 PASS1737Rust/0failed/88ignored/269suites/
allscriptchecks. Shared implementation owns operand/residency/native invocation;
family adapter owns roles/tie/SSA/publication. Timings below remain the earlier
storage/rings release2. Native14+core2 and opt-in Qwen full4 convolution-state
transport now accepted301selected paths:198core/86Qwen/549GPU/8nativeartifact
tests and strict complete-feature checks PASS. Required release both models'
complete physical/state/fault/API scopes plus forced-streamed2B faults PASS.
Full isolated gate14 LAST PASS1747Rust/0failed/88ignored/269suites/allscriptchecks;
source/rootREADME unchanged. Actual x2 native3 +affected HIP/Vulkan transport12
also PASS, scoped to four native kernels and BF16 transport. No Gemma/Qwen27B
inference or fresh performance grant.

Successor shared CPU arithmetic now reviewed/integrated302selected paths:
explicit original normal-domain exp profile closes authentic H48 cached producer
and all786432state words bitexact, preserving legacy negative control. Actual
full64 replay3 now passes two independent sessions, each one INITIAL token ID11
then one cached token ID353,
all1040records/240896000float words/fullV/all112domains/ALL4Conv frames bitexact;
source/owned closures and finalhost0 pass. Four qualification-example corrections
are integrated MAIN/ISO. Natural/long/GPU/native/serving/performance remain open.
Existing native301 qualification/timings stay scoped to their immutable sources.

Prepared recipe successor now reviewed/integrated314selected source paths.
Sol HIGH implements; Astra HIGH architecture-only review accepts/no must-fix.
Same record/ForwardPlan: shared topology/source semantics, genuine per-request
operation/provenance and original physical/state authority. Exact source cache40
retains4MiB/shape, conservatively160MiB; request record bound512KiB replaces4MiB.
Warm binding runs outside cacheMutex. Core201/Qwen87/GPU549/default+provider
strict PASS; original joint release and both models' complete real source/state/
fault/API scopes plus forced-streamed2B faults PASS. Fresh .8B C1/C4/C8:
ZG64.8742/63.5759/60.7789 vsstock37.9990/49.3562/62.3861tok/s
(+70.73%/+28.81%/−2.58%). All48offers384tokens exact/all6owner/source/group/
container/EOF closures healthy. Singlepass/no confidence claim; C8 superiority
still unmet. Source/site/native checks and wrapper allocations remain; no
zero-allocation claim. Full assembled gate15 LAST PASS1752Rust/0failed/89ignored/
269suites/allscriptchecks after the four qualifier example corrections;
314selected sourcepaths/rootREADME unchanged. Frozen release/timing retain
their original pre-qualifier snapshot, with unchanged CLI/library bytes.

Shared binder unit Sol8 now reviewed/integrated MAIN/ISO316:
existing ColdModel owns bounded40 immutable analyses; warm actual-call/operator/
producer checks run outside cacheMutex before scope/attempt. Original source/
member/SSA/role/head checks stay in cold analysis, downstream physical/state/
native authority remains late. Retained first-record witness +site metadata
guarded inside512KiB/entry, full20MiB+64KiB host inventory. GPU551/0failed/
54ignored/provider+default strict PASS. Original joint release and both models'
complete real source/state/fault/API scopes plus forced-streamed2B faults PASS.
Fresh .8B C1/C4/C8 ZG67.2018/63.8830/60.9713 vsstock37.6230/50.4311/61.6415
tok/s (+78.62%/+26.67%/−1.09%). All48offers384tokens exact; six original
source/owner/group/container/full-EOF closures healthy. Singlepass only; C8
superiority remains unmet. Final assembled isolated gate16 PASS1752Rust/0failed/
89ignored/269suites/allscriptchecks;316sourcepaths/rootREADME unchanged. Source
and performance claims remain scoped to this frozen release. B4 loaded/inventory
changes reconcile against this donor. Genuine batching precedes further N1
microtuning.

Previous timed frozen storage/rings release2:287 MAIN/ISO selected source paths;
strict6,193core+543GPU tests and standalone default/CUDA/HIP/Vulkan/all-provider
targets pass. Both actual301finite+40natural/fullV/all42state/all5fault matrices
and matchedAPI24offers192tokens perprovider pass. Explicit forced-streamed2B
mid-head unknown/cancel also verifies original transfer retention/quarantine.
Fresh .8B C1/C4/C8 ZG62.3484/60.7822/57.4249 vsstock38.0862/49.9539/61.8017
tok/s (+63.70%/+21.68%/−7.08%).2B C1 ZG43.3738 vsstock37.6126tok/s
(+15.32%): full1017118720B resident head, explicit native4608MiB vsstock
request4096MiB+KV512MiB/no harddevicecap. Host4096/device5120MiB unchanged.
All timing owners/source/group/EOF closures closed. Singlepass/no confidence
claim. Native remains N1; C8 superiority is unmet. Full isolated gate12 PASS
1734Rust/0failed/88ignored/266suites/allscriptchecks;287sourcepaths/rootREADME
unchanged. Genuine B4/B8 batching remains open.

Previous checkpoint: frozenpair9 shared prepared content keys, cold GDN/FA copy
definitions and immutable source-seal reuse assembled with portable21/Gemma13/
diagnostic6,260MAIN/ISO sourcepaths exact. Strict focused checks/joint release
and original301edge+40natural/all5fault/matchedAPI checks PASS. Device counts/
fault scalars unchanged; cold cohort metadata +1400B is priced. Quiet C1ZG61.5227/
stock38.5419(+59.63%),C4ZG62.4174/stock49.6312(+25.76%),C8ZG62.5204/
stock61.2769(+2.03%)tok/s. All48offers384tokens exact/all6 originalowner/source/
EOFclosures reconciled. Singlepass only; small C8 margin is not stable confidence.
Full isolated gate11 PASS1724Rust/0failed/88ignored/266suites/allscriptchecks,
260sourcepaths/rootREADME unchanged.2B actual301edges+40natural/all5faults and
matchedAPI24offers192tokens eachprovider PASS. Quiet2BC1 ZG0.5298/stock37.0680
tok/s (stock69.97x);16offers128tokens exact/both originalclosures reconciled.
Further2B bands held for shared storage/streaming repair. Source shows1.017GB/
970uploads+streamwaits perforward; full attribution/fit remains open. Parallel
pool/boundary audits and shared APIs linked above.

Previous source checkpoint: shared ordered-enqueue actor6 passes frozen pair8
original301edge+40natural/all5fault/API checks; actual dispatch/finite/fault
scalars equal pair7. Full isolated gate10 PASS1715Rust/0failed/88ignored/
266suites/allscriptchecks over220selectedchangedpaths, source/rootREADME
unchanged. Quiet pair8 C1ZG60.8262/stock38.2050(+59.21%), C4ZG60.9697/
stock50.6970(+20.26%), C8ZG60.4256/stock63.2563(−4.47%)tok/s. All48offers/
384tokens exact; ROOT checked all6 originalowner/source closures. Preliminary
singlepass, no confidence claim or2B promotion. Current actor CPU/GPU-envelope
diagnostic precedes next optimization; earlier pair7 remains historical.
Portable corrected transport12 actual HIP/Vulkan modes separately pass on x2;
arithmetic10 actual HIP/Vulkan modes pass exact finite CPU-profile boundaries
and original fault/ownership closure; reviewed21paths merged MAIN, combined
gate pending. Gemma13 source contracts also merged; whole-model GPU remains
unqualified. Actor8 diagnostic closed with64exactforwards/432steps/all owners,
whole device-clock envelope.615946s/nestedHead.099558s, hostforward.860439s.
258usercycle samples/0lost identify allocator scans, kernel-key hashing and
driver submission; no wall/GPU percentage attribution. Prepared immutable key
unit proceeds without math/cap changes; no2B timing.

- [ ] **Parallel implementation, third merge — genuine BF16/HF request batching
  (owner directive,23Oct).** Develop the physical multi-request unit now; merge
  after current storage/residency/architecture fixes, before further N1 microtuning.
  Start with genuine canonical47 B4/B8 initial prefill (188/376 total tokens)
  through SAME independent roots into decode and correct partial-batch fallback;
  extend mixed frontiers through shared engine
  cohort/binding/workspace APIs. Reuse existing root-cohort publication, but
  implement compatible BF16 attention/GDN/projection lowering rather than
  enabling the older GGUF/F32 path. Preserve independent row states, natural
  feedback, fair work, cancellation/quarantine and current resource caps.
  Validate row permutations/full logits/all states and physical/API behavior,
  then matched C1/C4/C8; preserve the C1 fast path. Request batching is distinct
  from operation fusion; no speedup or superiority claim before measurement.
- [ ] Lift reusable source identity, bounded immutable reads, prepared bindings,
  actual-flight indices and resident matrix transport into shared components.
  Select qualified implementations using complete host/device live costs;
  a family adapter may restrict mathematics without duplicating common policy.
- [ ] Shared resident-backing pool + immutable layout/access metadata + bounded
  transfer/frame retirement, from23Oct parallel source audits. Keep existing
  model traversal/publication and current root/lease freshness; price live,
  idle and unknown owners until actual retirement. Expose minimal common APIs
  through existing prepare/bind/submission/completion objects. CUDA/HIP pinned
  staging and Vulkan queue/mapping need their own physical proof; no inferred
  speedup or zero-copy claim. Follow the coherent units in the linked design.
- [ ] Demonstrate superior matched .8B C1 AND C4, then run C8. Only after all
  three wins, optimize/benchmark2B. Keep original quality/source/caps and report
  single-pass uncertainty. Latest prepared-binding release1 observes C1+78.62%/
  C4+26.67%/C8−1.09%; C8 superiority remains open.2B head residency is fixed;
  actual physical/API pass and C1+15.32% with the explicit4608MiB profile.
  Genuine batching precedes wider fresh timing bands; stable superiority is unproved.
- [ ] Merge Qwen27B/Gemma contained units as ready in parallel. Their later GPU
  qualification/benchmarks use pvl-x2's AMD gfx1151 through ROCm/HIP.
- [ ] Bring Vulkan implementations to CUDA parity independently; AMD uses HIP,
  not an implicit Vulkan/CPU substitute. Preserve one engine/physical service,
  independent roots, fair work and completion/quarantine through each provider.
  Read-only pvl-x2 inventory confirms96GiB GPU memory, existing ROCm7.2.4 and
  RADV/Vulkan. Availability/compilation is not numerical or performance parity.
  [Shared components and backend implementation units](BACKEND_PARITY.md).

- [ ] **OPT-01 matched short baseline:** after parity, established same-source
  C1/C2/C4/C8 profiles, original ABBA+BAAB/offered-load controls, real TTFT/true
  token ITL/E2E tails, correct committed/delivered SLO-goodput and failures.
  Separate unprofiled timing from CPU/driver/kernel/copy/IO diagnostics; pin
  actual binary/profile/caps/EOS rather than retagging old GGUF workloads.
  Quick .8B canonical wave8 C1/C4 baseline completed: ZG0.9557/0.9511 output
  tok/s vsstock38.5021/49.6408. Owner23Oct now requires fixing the gap and
  beating vLLM first;2B/full repeated/ragged timing deferred. C8 pair6 is now
  completed above; earlier C8 cancellation is superseded. Preliminary
  single-pass result, no repeated-run confidence or stable-tail claim.
- [ ] **OPT-02 reusable bindings:** profile then qualify stable leased row/
  workspace metadata, bounded diff uploads, compact device feedback and removal
  of redundant source/state/readout copies in the original actor. Source/root/
  epoch/physical generations and unknown-completion ownership stay authoritative.
  After the current short baseline, measure the1MiB streamed vocabulary head's
  actual source reads/copies/launches. Qualify admitted larger windows or full
  residency where the complete model/root/workspace inventory fits; retain
  streaming when it does not. Investigate sharing serial cold-geometry storage
  by genuine liveness/alias proofs rather than summing six separate cohorts.
  Both are performance hypotheses, not measured gains or permission to raise caps.
  Current source uses485/970 authenticated1MiB reads+H2D+kernel+same-stream
  checkpoints per .8B/2B Head. Existing .8B original full-table selected-head
  actor offers a reuse path. Raw508,559,360B .8B table addition stays below the
  present4GiB provider/host upper bounds, before deriving new exact overheads;
  original estimate is now superseded by actual .8B validation below. 2B cannot
  inherit that fit claim; evaluate larger tiles and proved workspace reuse.
  Active isolated resident-head unit reuses original full-table actor, keeps
  same arithmetic/source/private ownership and actual4GiB/5GiB caps. Separate
  warmed-process CPU profile now closed healthy: sampled user CPU cycles57.573%
  SHA256,15.202% BF16 finite scanning,10.195% actor spin,7.341% memcmp. Private
  stripped addresses classified from actual instructions. These are CPU cycle
  fractions, not wall/GPU percentages; perf-stat file was empty. Static30.31GiB/
  31,040sync per64forwards is not a measured time percentage.
  Assembled resident .8B passes actual40 natural calls/all5 prompt lengths with
  exact boundaries/fullV/all-state/feedback and pressure/stale/precancel/retirement.
  Actual resident head508,559,360B; source-derived provider upper2,625,406,976B/
  host upper3,397,702,556B fit unchanged4GiB limits. Conv/recurrent/lastFA fault
  cases pass; head-unknown exposed verifier's old streamed-only error recognizer.
  Exactly2 resident strings added behind unchanged error/index254/fault20|21/
  actual-enqueue gates; two affected Head faults await rebuilt fixture. Vendor117
  CPU guards pass. No speedup or complete new-batch acceptance claim yet.
  Follow-up physical Headunknown+cancel PASS; matrix40+all5faults complete.
  Quiet resident CLI2 diagnostic8requests/64forwards exact, owned source/retirement
  closed: forward6.907854s, bind_program3.629112s, enqueue3.021396s,
  complete/readback0.148894s. Nested Head0.005158s,0H2D/0checkpoints/1native per
  call; prepare includes forward, so totals must not be added. Instrumented
  wave7.115s is not an unprofiled performance win. Next substantial host unit
  addresses reusable cold construction and attempt-local state proofs while
  preserving dynamic root/session/generation/fence/cancel/ownership checks.
- [ ] **OPT-03 prepared optimized regions:** close existing production graph
  admission in the serving plan; qualify exact producer/consumer fusion,
  prologue/epilogue layouts, finite decode/prefill/verify shape variants and
  source-derived packs. Preserve materialized casts/reductions, aliases and
  effective fused read lifetimes. Fewer graph submissions are not fewer kernels.
- [ ] **OPT-04 temporal/fair work:** extend existing native sequence batches
  with actual multi-token dense/GDN prefill and fair mixed-work chunks; retain
  independent state and profile-specific arithmetic. Serving owns scheduling
  integration; memory owns pages/prefix/COW. Re-run short concurrent baselines
  after singles and meaningful combinations before larger contexts.
- [ ] **OPT-05 state/residency composition:** full-domain compatible prefix
  intersection, pages/COW, exact source chunk/pack sharing, safe scratch reuse
  and bounded transfer readiness through existing owners. Benchmark cache hits,
  cold arrivals, long prefill versus decode, memory displacement and fair tails.
- [ ] **OPT-06 larger dense source:** after long-context prerequisites, original
  27B execution using authenticated layer/row windows, coalescing/batch reuse
  and admitted prefetch. Price every required dense matrix plus activation IO;
  a sparse expert cache cannot turn this checkpoint into a sparse model.
- [ ] **OPT-07 optional proposals:** target verifier/all-prefix rollback first,
  then bounded n-gram/Recall/suffix and compatible trained MTP/block drafters.
  Joint finite cost policy considers batch/graph/depth/IO/retained state and
  selects K=0 when it loses. Exact distribution, seeded trace, numeric profile
  and quality changes are distinct gates. No acquisition/training implied.
- [ ] **OPT-08 nested profiles:** reconcile primitives→recursive pipelines→
  profiles with the pipeline-composition plan. Compile/in-line/specialize closed
  regions through the original engine; no per-token interpreter/search, new
  serialized IR, model loop, allocator or publisher. Structural design can
  proceed in parallel; new framework implementation must not delay parity.
- [ ] **OPT-09 deferred capabilities:** typed sparse selection, grouped experts/
  protected SLRU/hotsets/replicas, deterministic PLE/Engram IO, source-dependent
  shared-KV/liveness, hardware-specific MMA/PDL and derived weight/state codecs.
  Activate only for qualified Qwen/Gemma variants, then other model/device work.
- [ ] **OPT-10 promotion:** source/full-logit/all-state/natural-output, actual
  cancel/pressure/unknown-completion/fair serving and complete isolated check.sh;
  repeated complete-request benefit including cold/setup/memory/energy and
  negative cases. Do not multiply external gains or promise universal superiority.

### Owner follow-up — 21 Oct, reuse/harness/target coverage

Contract: [profiles, components and reuse](../pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md).
Implementation remains behind the ordered parity/baseline prerequisites above.

- [ ] Qualify [cross-stage residency/fusion](../pipeline-composition/REGIONS_AND_PROFILES.md#resident-values-and-cross-stage-lowering--owner21-oct)
  through the original engine after baseline: compare ordinary, resident-chain,
  graph and fused paths; retain intermediates/feedback on device where eligible,
  report actual H2D/D2H/device traffic, host waits and reasons for required
  bridges. Preserve arithmetic/declared approximation, original commits,
  lifetimes/cancel/fairness and measure complete-task benefit/negative cases.
- [ ] Complete the dedicated [ExLlamaV3 investigation](../../research/engine-optimization-comparison-2026-10-21/exllamav3-adoption-scope.md):
  pinned kernel/packing/transform/codec consumers, shape/model/backend limits,
  KV, batching/graphs, CPU offload and partition seams; exact file/dependency
  licenses and static C/Rust ABI versus direct-port disposition. Add independent
  fixtures and map mechanisms into existing plans; existing EXL3/QTIP calibration
  design is not missing and must not be duplicated. Initial scoping is not a
  completed audit or a new entry in the completed416-mechanism inventory.
- [ ] Implement native kernel-contract comparison/fixture support, reusing or
  directly porting suitable TensorFold harness/leaf code after license/ABI
  review. Add planted rounding/reduction/NaN/tail/no-write/alias/source defects
  and prove every expected rejection. Invalid hardware configurations refuse
  before launch. Keep broken controls test-only; independent oracles, actual
  model/all-state/physical checks and full isolated check.sh remain required.
- [ ] Implement the useful kernels/components identified by these investigations
  through existing providers, using static-native reuse first and faithful
  direct ports when unsuitable for linking. Qualify the implementation as well
  as the harness; an upstream passing test is not local runtime acceptance.
- [ ] Specialize first for the NVIDIA cards on both existing machines, then
  prioritize Strix Halo/gfx1151 ahead of all other new hardware and cluster
  work. Keep broad model/device/platform compatibility as the long-term target,
  with portable paths and actual per-target gates; retain the separate gfx90c
  hold. Hardware order does not alter parity→short→long→dense27B model order.

- [ ] Bench and hyper-optimize in strict user order:
  Qwen3.5 2B, dense Qwen3.8 27B including SSD/coalesced layer-window streaming;
  Gemma4 E2B/26B/31B; Qwen3.8 Flash-Next Q2/~3.5-bpw/Q4
  and MTP/drafter/n-gram pipeline. Optimize complete
  request goodput with the same model semantics, source quant
  calibration and StateRoot precision. Do not change the
  Renoir/gfx90c launch hold or assume measured superiority.
  Routed-expert streaming applies only to later source-defined MoE variants.

- [ ] Keep the local Renoir/gfx90c hold: 2B initialization aborted and automatic resets wedged it; no further launch/reset/recovery or Drop-retirement proof is implied.
- [ ] Establish controlled cold/warm request baselines across all five GPU/three CPU optimization targets; accepted five backend profiles do not mean five cards or fleet-wide maximal capability use.
- [ ] Qualify capability-selected AVX2/AVX512/VNNI quantized kernels, cache/thread/topology policies and source-derived repacking while preserving reduction/rounding/zero/subnormal rules.
- [ ] Measure native/Vulkan tiling, subgroup/vector loads, packing, source-exact fusion/prepared dispatch; gate tensor/MMA/WMMA/integer-dot use by exact arithmetic, not advertised support.
- [ ] Evaluate explicit NativeF32 projection policy only after physical FMA edge tests, all-codec trained vectors and whole-model/cancellation/replay gates; WideF64 stays default meanwhile.
- [ ] Profile then qualify captured graphs/replay, bounded device loops/persistent scheduling, CPU/GPU overlap and exact optional GPU tokenization/detokenization; prepared plans alone are not graph capture.
- [ ] Keep separate LLVM code-generation paths and broader accelerator backends unqualified until independently built and physically tested; accepted native paths use CPP NVRTC/HIPRTC.
- [ ] Qualify optional EXL3/QTIP/trellis/codebook/low-bit derived packs and calibration identity/quality separately from original source semantics. No hidden precision substitution or default steering.
- [ ] Promote only repeated complete-request gains with cold/warm cost, tails, memory, power, source correctness, negative results and qualified capability/version guards.

- [ ] Gate future compiler/driver/libm and fast-arithmetic profiles with scalar fingerprints, physical edge/stress and trained full-logit/state tests. WideF64 is qualified on measured inputs, not a universal IEEE FMA emulation.
- [ ] Qualify real deadline wakeups, concurrent/multi-stream scheduling and actual driver failure/recovery before broader worker claims; queue deadlines currently need scheduling/poll opportunities.
- [ ] Keep the local gfx90c hold. If a healthy/recovered profile is separately authorized, localize retained allocation/H2D/final-sync initialization before changing kernels; natural SIGABRT establishes neither Rust Drop quarantine nor reclaimed storage.
- [ ] Consider read-mostly publication only after representative contention and ownership evidence. If left-right fits, test deterministic dual replay without IO/RNG/admission/state/device effects, one writer, persistent non-Sync readers, short guards without await/load/device work and explicit publish/drain/shutdown. Account for both indexes/logs while sharing one payload; do not reopen the rejected cold-catalog migration.

- [ ] Resolve applicable SYK-011..016 source/derived-quality/LLVM/SLO/pressure/steering choices from the [specialization reference](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) before that slice; proposed SYK-002..010 are not frozen user approvals. Preserve negative candidates and the default disabled steering policy.

## Committed

### 6d85559 — correctness comparison gate, 26 Oct

- [x] All-model shared comparison helpers + fail-closed CPU report gate;
  genuine2048 requirement restored, existing numerical limits preserved.
  Full isolated `scripts/check.sh` actual0/EOF:1890 Rust passes/0fail/89ignored;
  new23+15 controls/selftest pass. No runtime/math source changes or fresh GPU
  inference. [Evidence](../../runs/q08-correctness-20261026/TODO.md).
- [x] Retained actual C16 IDs/text digests compare exact for both models under
  exact/default policies,16 requests/model. Original case5 N1/B4 failure and
  historical2B performance refusal remain failed; no new speedup claim.

### `5d40cf5` — shared pending queue and ingress credit

- [~] Shared queue checkpoint preserves application-owned HOST admission,
  unlimited default pending count and optional cap. Final batch corrects
  tokenizer construction, original Rayon reuse, HTTP framing/serialization
  and worker/last-reader ownership. HTTP handles and connection credit survive
  through join; original default thread stacks remain runtime-selected.
  Platform stacks/TLS/caches and OS/driver headroom remain separate, unquantified
  scope: application reservations are not process RSS or a full-process fit.
  [Contract decision](../../runs/resume-all-queue-production-bound-astra-review-20261024/MERGE_SCOPE_DECISION.md).
  CPU functional acceptance is complete; the final full CPU gate passed
  rc0 25Oct on the committed tree. Assembled GPU checks PASSED 25Oct
  (22/22 rc0: builds/stricts, cuda_compiler_context 4/4, loaded/inventory
  4/4, CLI baseline/serve-short_hf, host-profile umbrella, four ignored
  route cases alone, five actual source-cost cases) —
  [execution](../../runs/resume-all-takeover-queue-gpu-glm-20261025/LOG.md).
  Physical device scopes (CLI direct route, serving) remain open.
- [x] Optimized25Oct CPU population/cap/cancellation/HOST continuation passes
  under unchanged4096MiB,2workers×4active and unrestricted P16 affinity.
  All16/32/64 requests are offered before draining; actual greedy output,
  unique Done and final serving/worker/queue closure assertions pass. Seven
  ledger tests and three functional cases pass;1024MiB retained-stage refusal
  remains distinct from optional queue-cap429. Actual release CPU/serve build
  and strict checks pass. Captured successful child output does not expose
  exact peak/backing traces; this is CPU correctness, not GPU C16/32/64 timing.
  [Execution](../../runs/resume-all-tokenizer-vm-cpu-continuation-prep-pi-20261025/).
- Earlier optimized25Oct population control fails at offer12/16 under unchanged
  HOST4096MiB/active8: each pending request retains about240MB temporary tokenizer
  credit. Source review separates about224MB fixed VM scratch from about17MB
  cache growth; fixing cache growth alone is insufficient. No16/32/64 completion
  accepted at that failed attempt. Shared repair reserves serialized fixed scratch once per actual
  tokenizer in its typed Retained owner; the private mutex and synchronous
  single-string call graph establish concurrency1. Original run used affinity4;
  acceptance must cover unrestricted production P16, without narrowing affinity.
  Remove duplicate CPU/GPU request charges, preserve all prompt/output/state
  owners. The repaired CPU population batch passes above; GPU acceptance remains.
  [Reviewed design](../../runs/resume-all-queue-production-bound-astra-review-20261024/TOKENIZER_GROWTH_CREDIT_PLAN.md).

### `52f76a0` — short-HF tokenizer credit and HOST8

- [x] Explicit tokenizer text-feature coverage now passes: actual optimized
  `--features text` build/strict,14 bounds tests and one actual-plan test at P16.
  The minimal `super::plan(16)` test-only repair leaves production math unchanged.
  Earlier zero-test/default-feature and101 attempts remain failed/unaccepted.
  Actual current rows: R1,147,648,523/G17,054,684/V224,000,144B, plus actual
  lease inventory; the older supplied R was136B higher. No passing CPU replay.
  [Execution](../../runs/resume-all-tokenizer-text-fixed-coverage-pi-20261025/).
- [~] Direct short-HF route lifetime correction: preliminary tokenizer drops
  before model loading, but its static retained-credit vector never releases.
  Return tokenizer plus its actual lease; validate the exact legitimate
  process-global bootstrap baseline through route/model closure. No VM9/H14
  source overlap. Closed11-path source draft has the typed owner; ROOT/Astra
  required collected corrections for request payload/credit lifetimes,
  unconditional original close, exact bootstrap/backend baselines and tests.
  Those source repairs landed in `52f76a0`; the affected optimized/strict
  batch and the final full CPU gate passed rc0 25Oct; real source-cost/
  lifetime and GPU serving/ownership checks remain pending.
- [~] CUDA B4 HOST requirement needs correction before fresh acceptance.
  Source-derived P16 lower bound4,341,525,623B already exceeds4GiB by46,558,327B
  before bootstrap/native/other metadata; old source21 HOST fits omit the new
  R+G+V credit. Cost-only inventory currently bypasses that owner too. Preserve
  all real charges, GPU caps and CPU4096MiB population controls; review honest
  configured HOST requirements versus real owner/phase savings. Explicit
  short-HF8192MiB policy landed in `52f76a0`; CPU/default4096 and legacy4GiB,
  provider/physical and H/G/V caps remain unchanged. Measured 25Oct
  (application-owned configured-domain accounting at 8192MiB, gpu_executed
  false, no process-RSS or device claim): N1 .8B host_upper 4,075,700,811B,
  N1 2B 4,111,001,099B, B4 2B 4,741,293,239B, B4 .8B 4,736,259,779B — all
  fit 8GiB; legacy 4GiB refusal tied to authenticated host cost (excess
  446,325,943B, provider within cap). Real RAM/cgroup availability and
  device/serving execution checks remain open.

### 25Oct assembly and evidence records — `docs(plans)` (this commit)

- [x] Old-worktree cleanup:52 registered trees +5 redundant extracts removed;
  historical source/notes/evidence archived and independently rechecked.
  Separately removed30 obsolete merged branches; main and five active heads
  remain. Two historical registrations retired after exact metadata backup;
  their ordinary reference files remain at the required paths. Six registered
  trees remain: main and five active lanes. Main/rootREADME preserved.
  [Details](../../runs/resume-all-parallel-root-20261024/WORKTREE_CLEANUP.md).
- [~] Five-lane source is assembled in MAIN; the final affected batch completes
  portable core admission, actual spawn-capture ownership and direct-lease
  metadata. Original private-network HTTP checks now execute without a fallback.
  The final source freeze/full CPU gate passed rc0 25Oct and the five-lane
  batch landed as signed commits (`8f01635`..`68e13a8`); assembled GPU/
  physical checks, H6/G8/V10 cases and lane-tree retirement remain pending;
  no new benchmark yet.
- [x] Read-only `pvl-x2` metadata check25Oct: existing data/cache/SDK paths
  present, all eight proposed QUAL/STATE leaves absent, AMD PCI/render/KFD
  paths observed; original process/group/EOF closure healthy. No payload hashes,
  driver launch, physical-fit or inference qualification follows from metadata.
- [x] Retire the five active Qwen27B/Gemma4/HIP/Vulkan/queue lane trees at the
  reviewed checkpoint; their units are integrated (`8f01635`..`68e13a8`).
  DONE25Oct: prepared-B4 certification/state transport implemented per
  [Astra design](../../runs/cpu-gpu-optimization-astra-c16-shared-gap-20261024/DESIGN.md)
  and merged in `d3d21a3` (existing arithmetic, caps and independent roots
  kept; 22.13% remains a software-dispatch ceiling, not a wall-speedup
  claim). Unit green: 594 gpu lib tests + 17 new (real ABI counts 9/22/30,
  all ≤32). PENDING: the B4 physical GPU round (ignored matrices +
  actual_grouped_state_transport_oracle) and the fresh matched C16 for
  both models on the checkpoint tree; C32/C64 follow acceptance.
  DONE25Oct: five trees and branches retired after evidence preservation
  (zerogravity-worktrees/.preserved-lanes-20261025/); OLD-OIDs recorded there.



### `3e89f1919993` — shared resident execution and current finite checkpoints

Related source commits: `d9d4f8b9bb50` (vendor-cuda-actor), `68c2f5c05013` (gpu-shared-kernel-primitives), `08fc5935e124` (facade-short-hf-cohort-suite), `9999adb1a66d` (serve-short-hf).
Aggregate source21 qualification applies to the assembled tree; no per-commit
build or general model/backend grant. Historical failures keep their source scope.

#### Current source checkpoint — 24 Oct

- [x] ROOT reviewed and provisionally integrated source21's six-path shared
  state-certificate preparation, using source-issued plane layouts and the
  original compiler owner. Reused checked layout accessor preserves allocation
  geometry; no math kernel/launch, profile/B8 admission or budget change.
  Complete1650 MAIN/build files and524 selected MAIN/ISO paths match. Source
  integration reviewed; current21 physical acceptance and separate math/serving
  status follow below. Current21 complete physical/math/API/HTTP unit passes.

- [x] Combined source21 full gate1793PASS/0fail/89ignored, all-target strict,
  five pure CPU readiness contracts and optimized CLI/N1/B4 release build pass.
  ROOT checks complete1650files in both roots,524 selected MAIN/ISO paths and
  all three actual emitted/frozen binaries. Release ba1370ac…; no runtime grant.

- [x] Current21 whole correctness unit through Pi: physical23, full2B original
  composition and .8B API→HTTP. CPU composition/API may overlap; quiet benchmarks
  wait for actual outcomes and owner closure. No retries or warmed substitution.
  First attempt stays failed/incomplete:17 actual cases passed, then a concurrent
  Pi wrongly edited MAIN and the source guard refused. ROOT preserves all3edits,
  restores exact1650bytes/modes and verifies unchanged frozen binaries/README.
  New implementation workers have actual candidate cwd and kernel-enforced
  MAIN-readonly mounts. Fresh complete unit uses separate -21-2 outputs.

- [x] Fresh source21 physical23 passes. ROOT independently reruns unchanged
  original typed validation of all23 ordered source/native/binary/state/fault/
  owner records; result50f6c8c3… binds release ba1370ac… and selected524paths.
  Fresh2B export is diagnostic-only until complete numerical comparison;
  serving/compiler-zero/performance remain separate.

- [x] Current21 full2B numerical composition passes. ROOT independently rehashes
  133 pinned files including all53current raw members/1,702,109,056B:15transitions,
  35events/24rows/8112sections including576unrounded sums. Original30 reused,
  no new original calls or independent public-trajectory claim.

- [x] Current21 .8B API/HTTP pass with ROOT recheck:24offers192tokens/provider,
  genuine5arrival36token N1→mixedB4→refill outputs exact; original3owners close.
  Cold448/6084912source/5007769PTX equals whole; all3compiler deltas0 from BEFORE
  warmup, five prompts×eight outputs complete. This fixes the observed20 failure
  within its actual finite source/profile scope; old failed20 remains failed.

- [x] Fresh current21 .8B C1/C4/C8 passes; ROOT checks exact48offers384tokens,
  six original owners/source closures and compiler0. ZG67.7432/95.6083/92.9820
  vs vLLM38.8470/42.8328/54.8241tok/s; rate AND wall wins, singlepasspreliminary.
  ActualC4/C8 usesB4kernels. [Raw results and limits](SHORT_HF_RESULTS.md).
  All paused implementation workers resumed; Qwen correction checks released.

- [x] Current21 2B API→HTTP whole unit passes; ROOT reruns unchanged typed
  physical/fullcomposition/API/HTTP/source/owner validators. API24/192/provider,
  genuine fresh2B stock5cap reference and HTTP5/36 mixedB4/refill exact; all4
  owners close. Cold396/source5235651/PTX4320117 equals whole, allcompiler0.
  Provider4608MiB/physical5120MiB/HOST4096MiB unchanged; stockhardcap equivalence
  remains false. source21-api-http2b-root-review.json; no performance grant.

- [x] Quiet current21 2B C1/C4 firstpass passes ROOT typed c8_gate/output/owner/
  source review:45.0305vs37.7431 and80.8278vs50.2514tok/s, BOTHwallwins;
  all32offers256tokens/fourowners/compilerdeltas0, cachedB4fourwide. Raw1c2f393c…;
  singlepass/pinnedrecipe scope only.

- [x] Fresh current21 2B C8 passes ROOT independent original typed review:
  ZG80.7265 vs vLLM62.4442tok/s, walls.79280/1.02492s; all16offers128tokens,
  two original owners, exact output/source/native closure and compiler0.
  Actual14cachedB4flights56rows plus8N1prefills. Raw8b822cc6…;
  source21-c8-2b-root-review.json. Preliminary singlepass only.

- [x] Actual frozen20 one-request capture identifies both late kernels:
  `CertifyFiniteStrided`, dim32/Unchecked/U32, rows1, widths18432/262144.
  ROOT rehashes896 raw files and verifies446 cold pairs unchanged; deltas
  +2/+13233key-source/+4739PTX. They match source21's prepared constructors.
  Original first-use refusal remains failed. Bounded Debug collisions in old
  native descriptions are retained as ambiguity, never full-key equality.

- [x] Reviewed CLI34 and finite-readiness7 source units merged into main with
  fresh mtimes; README and source15 actual-fault fixture preserved. Shared
  immutable finite groups retain whole-group eligibility or individual checks;
  all six N1 geometries,108 GDN preparations and40 attention controls traversed.
  Loaded-object coverage now supplies serving readiness rows.

- [x] Astra/ROOT rejected Pi's skip-missing-certificate suggestion after checking
  complete original certificate factories and exact rebind invariants. Hard
  errors remain; grouping-complete is a narrow preparation statement.

- [x] Assembled source20 isolated full gate:1793 Rust PASS/0failed/89ignored,
  installer/script checks PASS; combined all-target strict PASS. Shared argmax
  kernels now compile independently for Qwen or Gemma without arithmetic changes.

- [x] Source20 optimized CLI/N1/B4 build and ROOT review pass. Exact emitted
  binaries, all1650 build inputs and523 selected MAIN/ISO paths verified.
  Subsequent AGENTS change is operational documentation only.

- [x] Fresh source20 complete physical23 PASS: standalone301, both N1 matrices,
  .8B B4, cancellation/unknown/streamed-head faults, genuine2B export and3cached
  faults. ROOT independently checks23 actual tests, source/native/binary and all
  original owner closures. No serving/performance grant from this unit.

- [x] External original-producer / finite-consumer bridge real request and .8B
  path corrected;36CPU controls pass and real original helper metadata works.
  ROOT reviewed corrected source; actual full composition remains separate.

- [x] Actual current2B full composition PASS. ROOT independently rehashes all53
  raw members/1,702,109,056B and authenticates original30 reuse:15transitions,
  24physical rows,4344observed (including576unrounded sums)+3768plain=8112.
  No new original calls; no independent public-trajectory or performance grant.

- [x] Source20 x2 complete finite shared unit passes: bootstrap/build, actual
  binary binding, known3Q/4G reload and both unknown1Q/1G cases. ROOT reruns
  original typed receipt validation and checks all local/remote source,70dataFD,
  binary, global-fence and owner closures. New artifact51808f9e…; all earlier
  failed namespaces preserved. This qualifies the finite source20 fixture;
  current source21, public serving, natural27B trajectory and speed stay open.

- [x] Current21 both-model API/HTTP, C1/C4 and subsequent C8 pass within their
  original finite source/physical scopes; actual current numbers above.

#### Ordered adoption — 21 Oct

- [x] Qwen27/Gemma4 vLLM optimization audit:20Qwen+20Gemma+15shared source
  comparisons, frozen benchmark0.31/current upstream/stable0.32 distinguished.
  Neither has all applicable optimizations; backend/codec/model gates explicit.
  [Report and priorities](../../runs/model-execution-qwen-gemma-vllm-audit-20261024/REPORT.md).

- [x] **Pools/rings and shared .8B→2B reuse.** Assemble
  charged backing/fresh logical roots, bounded transfer retirement, immutable
  recipes and exact mutually exclusive geometry views. Derive resident head
  upload/embedding/ABI/inventory from actual source geometry through existing
  engine→shared model operations→family→model adapters. Owner clarified the4GiB
  provider allowance is artificial: keep .8B comparison at4096MiB, explicitly
  configure2B at4608MiB inside the unchanged5120MiB physical envelope. The
  complete2B bound4598390272B fits this allowance; actual full resident-head
  correctness/fault/API checks now pass for both models.
  Qualify the assembled source/roots/states/faults/API, measure selected quiet
  timings, then run the full gate. No new performance claim from source edits.

- [x] Shared optimization batch integrated MAIN/ISO: common exact source
  equality, complete-cost residency selection, prepaid bounded immutable windows,
  original-flight/certificate indices, immutable prepared RMS/GDN bindings and
  source-derived resident BF16 matrix transport. No model-name cost policy,
  arithmetic/cap change, source bypass or new executor. Qualified adapters still
  restrict unavailable mathematics;2B remains streamed.196 assembled paths;
  full isolated gate6b PASS1713Rust/0failed/88ignored/266suites plus scripted
  checks. One formatting-only gate failure fixed; original donor kept intact.
  Real .8B40-call matrix/all5submittedfaults PASS with exact fullV/allstate/
  priors/naturalIDs, independent-root lifecycle and actual ownership closure.
  Matched API24offers192tokens perprovider PASS including active8, all IDs/
  text/usage/finish/policy and zeroerrors; source/ownedclosures healthy.
  Quiet pair6 matched singlepass: C1 ZG52.0304 vsstock37.1838tok/s;
  C4 ZG51.6654 vsstock47.0443; C8 ZG52.6333 vsstock61.0984. All48offers/
  384tokens exact, allsix original owners/full-source closures healthy.
  Actual peaks1/4/8 but batchrowsmax1/zero fused rows: throughput is nearly
  flat across bands. This establishes the measured gap, not its causal split.
  No repeated-run confidence, stable tails or whole-engine parity claim.
  The independent frozen Gemma matrix resumed immediately after C8 closure;
  backend preparation continues independently. Next unit investigates C8
  shared-engine/family costs and genuine certified multirow execution.

- [x] HIP/Vulkan admission successor:7path donor integrated MAIN/ISO; explicit
  compiled/runtime gates, missing-library private/public constructor refusal
  before source/client allocation, strict HIP+Vulkan/no-provider checks PASS.
  Assembled source gate7 passed below. The model-free stateful portable unit uses
  existing source/plan/session/physical flight and real fault/quarantine paths.
  Existing remote SDK is sufficient; approved offline Rust dependency closure
  prepared from current MAIN202source+locked cached inputs:541 checksum-verified
  archives,84.6MB, no missing package/Git source. Selected GPU-test offline Linux
  metadata resolves443packages/all9 workspace manifests; local portable cold/
  strictClippy PASS. Remote offline build and final six real HIP/Vulkan modes
  PASS: both healthy, both unknown, both shared-PCI fence directions. ROOT
  independently checked source/binary/owners/fullEOF and remote two-file source
  hashes against donor; qualification-only2paths merged MAIN. Existing matching
  glibc2.44 loader resolves installed LLVM22 requirement; prior loader/fixture
  owner failures preserved. Whole-model/BF16/serving/perf remain open.
  No SDK/model acquisition performed.

- [x] Complete assembled successor gate7: MAIN202sourcepaths unchanged through
  full isolated scripts/check.sh PASS1715Rust/0failed/88ignored/266suites and
  scripted checks. Backend7cold/strictHIP+Vulkan checks remain separate from
  physical support. GemmaCLI option-position regression included. RootREADME
  unchanged; no staging/commits. Frozenpair6 performance/physical results retain
  their exact196path older binary scope. Newer checkpoints are below.

- [x] Finite-cohort12path unit independently reviewed and physically verified:
  frozenpair7 passes301actual device cases+40exact natural calls/allfivefaults/
  matchedAPI24offers192tokens eachprovider. Independent roots/source/owner/
  group/EOF closure pass. All40 calls observe279 deferred/35cohorts+48immediate
  certificates,244 fewer enqueues per forward; no hardware activity enumeration.
  Portable foundation test2 also fresh-synced MAIN/ISO after source guards close.
  Assembled210selected changed-source paths exact MAIN/ISO; full isolated gate8b
  PASS1715Rust/0failed/88ignored/266suites/allscriptchecks, source/rootREADME
  unchanged. Gate8 stopped at test-module declaration formatting only; narrow
  correction retained the frozen binaries and actual evidence. Quiet pair7
  C1ZG55.0931/stock38.5359(+42.97%),C4ZG53.1955/stock49.3781(+7.73%),
  C8ZG53.4094/stock63.9834(-16.53%)tok/s. All48offers384tokens exact/all6
  originalsource/group/container/EOFclosures healthy. Singlepass; no repeated
  confidence,allthree superiority or2B promotion. Native remains N1/zero fused
  rows. Next substantive unit investigates shared submission/replay vs actual
  HF multirow lowering; no existing graph/fusion flag can grant its support.

- [x] Portable matrix11 reviewed by ROOT+independent peer,5source/geometry/ID
  guards+strictHIPVulkan+3CUDAgeometryguards+strictCUDA+allownedfmt PASS.
  All6exactparent/5new source+mode checks precede fresh MAIN/ISO integration,
  preserving CUDA adapter and all210prior changes/rootREADME. Actual remote
  matrix12modes and next assembled gate pending; this newer source is outside
  gate8b/frozenpair7 performance scope. Existing approved offline cache/SDK/
  loader and original owned limits reused, no acquisition/privilege changes.

- [x] Fresh frozenpair6 C8 diagnostic closed healthy before next implementation:
  exact64forwards/8offers/all original+two profiler owners/source/EOF closed.
  Nested forward1.134798s:bind.131585/preflight.036283/enqueue.796107/
  complete-readback.137668/post.031198; prepare includes forward, Head.005718
  includes64launches/zeroH2D/checkpoints.233usercycle samples/0lost, actor idle
  spin~29.48%self and memcmp~9.29%self; process1.47user+.33system over1.64010
  instrumented wall. CPU fractions/host phase durations are not GPU durations
  or removable wall gains. New bounded finite-check cohorts target actual327
  separate certificate launches/forward; sole-producer/range/status chronology
  must prove any deferred checks, while recycled/uncertified values stay immediate.

- [x] Finish the measured host equality cost: fresh C1/C4 .8B9.7748/9.7972
  vsstock38.0061/49.1130 outputtok/s, all32offers/256tokens exact and original
  source/owned closure. Combined diagnostic532samples/0lost: memcmp70.66%
  self usercycles, process6.72user+0.66system over6.77396wall; worker5.55user/
  .25system, actor1.16user/.40system. These are CPU samples, not GPU durations.
  Source audit locates recursive full static compiler/catalogue text equality
  in GdnFamilyIdentity/StageArithmetic at bind/enqueue. Confirm actual caller/
  compared extent, then preserve exact content equality with same immutable
  storage fast paths and full fallback for different storage. Keep negative
  source/profile/compiler/stage/ABI/control checks and actual current roots.

- [x] Exact .8B byte-comparison attribution completed without another GPU job:
  frozen ELF108-entry table byte-joined to5415018B original catalogue; actual
  sampled register pairs use the same table address. Same-pointer+length exact
  source equality repair covers all old compiler/family/stage fields and full
  different-storage byte fallback. Two paths merged MAIN/ISO;10 metadata guards
  PASS; strict GPU lint/joint fixture+CLI release and fresh matrix40/all5faults/
  API24-per-provider PASS. Fresh pair4 unprofiled C1 ZG40.4089 vsstock37.0951
  tok/s (+8.93%); C4 ZG39.2740 vsstock48.2272 (18.57%lower). All32offers/
  256tokens exact/actual owned closure. Preliminary singlepass; no confidence
  or fullparity claim. Current HF batchmax1; investigate qualified physicalB4
  integration separately. One assembled final full gate remains.

- [x] Fresh pair4 C4 CPU+phase diagnostic closes actual8requests/64forwards/
  432steps exact and original+perf source/group/container/EOF ownership.245
  usercycle samples/0lost: server idlebusywait hotinstructions~22% (source
  channel::Server::start/8192spinbudget), actual embedding range SHA~12.8%.
  Hash caller joined to ExtentSource::read_verified→VerifiedTensorSource rows,
  not GDN static compilation hashing. CPU1.78user+.35system over1.9563observed
  wave; nested bind.445470/enqueue.788747/readback.140948s, Head.005336s/
  64launches/0H2D/0checkpoints. CPU fractions are not wall/GPU gains.

- [x] Resident .8B tied-table actual-ID embedding D2D source/physical/API unit: reuse already
  authenticated immutable Head table/existing owned embedding copy; retain full
  tableRead in original flight/node0, source/record/currentroot/generation/
  service/stream/extents/nonalias/taint checks, diagnostic source comparisons
  and final485-window revalidation. Cover finite shortrow cohorts unchanged
  caps; preserve streamed2B/H2D. Source/targeted checks→reviewed merge→onejoint
  release→actual physical/API/quiet C1/C4 and finalassembledgate. No idle-spin
  tuning/newmath/kernel/permit bundled; no speedup claim before measurement.
  Reviewed six-path source merged MAIN/ISO; vendor119/GDN10/strictGPUClippy and
  joint frozenfixture+CLI5 release PASS. Actual40-call matrix/allstate lifecycle
  and allfive submittedfaults PASS/source/ownedclosure; matchedAPI24offers/
  192tokens perprovider PASS including actualactive8/zeroerrors/ownedclosure.
  Quiet matched timing follows below after parallel model/source scans retire.

- [x] Pair5 quiet matched .8B C1/C4 complete after all model/compiler/source
  scans retired. C1 ZG48.9151 vsstock38.8841tok/s (+25.80%); C4 ZG49.9998 vs
  stock50.4721 (0.94%lower). All32offers/256tokens exact,64actualforwards/roots
  pernative arm/432steps, peak1/4, batchmax1, zeroerrors and allfour original
  source/ownedclosures healthy. Preliminary unbalanced singlepass, no confidence
  or demonstratedC4win. Quick SHAa9acf925b575261ef578e107d4466165a7e022920796936c64abf0d21d5707ca.
  Earlier pair4 remains historical. Assembled gate5 PASS1645Rust/0failed/
  86ignored/265suites plus scripted checks. Subsequent shared cache/preparation
  drafts are outside that gate. Next C4
  binding reuse investigation remains source-only until causal scope reviewed.

### 6d74f76

- [x] Complete resident original-model pipeline and five bounded promoted profiles accepted; actual rollback/admission/cancellation/quarantine remain baseline requirements.

### Concurrency review — 1fca893

- [x] No demonstrated cold-catalog contention justified left-right. Its uncommitted candidate was restored and no speedup is accepted. Immutable Arc/borrows stay direct; a delayed status view cannot authorize admission, launch, release, StateRoot publication or fence clearing.

## Tests and verification

- [ ] Assembled .8B host batch: authenticate immutable114-weight cohort once
  per binding, reuse exact cold allocation analysis/certificates, seal fresh
  native roles once, index actual lease/access coverage. Current source/root/
  permit/generation/fence/cancel and physical range checks remain. Existing
  pre-node0 status clear permits removal of redundant4B upload/sync; no new
  sticky-status semantics or cap changes. Ten paths merged into ROOT ISO;
  frozen physical fixture passes full40+five-fault requalification, actual final
  host0/device0 and original owned closure. All14 pure metadata guards PASS,
  including three new lifetime/rebind/SSA/private-effect/alias guards. Matched
  API wave/ragged checks PASS full IDs/text/usage/finish and actual owned/source
  closure. Clean C1/C4 performance remains pending CPU/IO quiet clearance.
- [ ] Native harness passes positive fixtures and rejects all named planted
  defects; independent oracle and selected-profile equivalence matrix pass.
- [ ] Reused/ported implementation passes exact/approximate declared-contract,
  held-out quality where applicable, full-state/cancel/physical and native
  packaging/link checks independently of the harness's own controls.
- [ ] QTIP-01..14: derived source codec/pack identity, transforms, held-out quality and physical kernel correctness pass independently of original model semantics.
- [ ] HYB-01..18: exact source routing/draft/tap/mask/reduction and target-authoritative verification pass, including cases where a wrong draft is rejected or deeper drafting slows work.
- [ ] GFX-01..18: WMMA/LDS caller/shape coverage, driver/compiler numeric gates and coherent-memory physical headroom pass; require matched complete-request gains. See the [50-test register](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md#4-acceptance-register-50-planned-tests-not-implemented).

### 24 Oct — current 2B readiness checkpoint

- [x] Corrected fixture15 three genuine cached2B unknown-work cuts PASS: no publication, actual whole-flight/source/native owners retained, full5GiB physical fence, original supervisor retirement healthy. Original failed13/14 receipts preserved; correction is global-handle versus one-flight count scope, no runtime/math/capacity change.
- [x] Original2B source13 trajectory15transitions/30 original plain+observed calls/8112sections/576 actual unrounded sums exact; full source/loaded weights/native/owner closure accepted. Serving/N1 executables byte-identical13/14/15; explicit incremental prerequisites retain each actual producer identity (20 prior successes +3 fresh cuts), no failed-batch or independent-public-trajectory retag.
- [ ] Complete current15 fresh matched API/HTTP for .8B and2B; ROOT1254 running. Then quiet C1/C4 and C8 only after both rate+wall wins, no hot compiler activity; previous .8B measurements remain historical.
- [ ] Full assembled check.sh phase gate after CLI/source reconciliation; source12 gate18 remains the last accepted whole gate.

# ZeroGravity current comprehensive handoff

This is the canonical cross-window handoff for the repository.

It is deliberately not a generated concatenation of every historical plan, run log, source snapshot or per-file SHA catalogue. Those mechanisms were retired. Current source and tests are the implementation truth, the four maintained planning/status files are the navigation spine, the maintained subsystem and Fabric plans hold detailed design/work items, and Git history retains superseded material.

A successor should be able to start here without prior chat history, understand how the project evolved, identify what is implemented versus merely designed, and continue either the current Qwen correctness critical path or the separately authorized Fabric vNext transition.

## Functional transition accepted — `8d761c4` / `8c43844`

S0–S3/F0–F4 and scoped Q0/Q1/Q2 are complete. This closes migration of existing
functionality, not deferred Qwen contexts/concurrency/performance or future tasks.
The final reconciliation changes docs only; source remains the tested `8c43844`.


- User confirmed lifting GPU/trained-source test holds for existing approved
  sources/devices. Main only; no new subagents, acquisition, resets, privilege/
  recovery changes, gfx90c launches, pushes or recurrence restarts.
- CUDA repair signed `8d761c4`; Gemma diagnostics/SDK and accepted scopes signed
  `8c43844`, following `1f40880` / `fef7123`. Do not repeat extraction; older held
  statements describe historical checkpoints.
- Trained Gemma CPU SDK lifecycle passes91.78s: real completion/root, presentation
  failure, pre-cancel and original-owner retirement. Not independent numerical
  or serving qualification. CUDA/HIP/Vulkan scopes remain separate below.
- CUDA0.8B four-step cached47–50 full-logit/all42-domain natural-feedback parity
  passes against the original captured payload. Normal/cancel/unknown lifecycle
  pass, including exact restore, original quarantine/charges and final retirement.
  Fresh processes; no populated long-context or wider-model claim.
- Local RTX3060 and remote PCI `0000:c6:00.0` / gfx1151 identities inspected.
  Nix CUDA12.9 paths are absent; installed13.4 fails unsupported PTX. Existing
  sealed baseline12.9 compiler/headers plus installed GCC15 runtime resolve this
  without downloads/patching/driver changes. Failed environment scopes retained.
  Remote approved ROCm7.2.4 remains; current-source HIP build verified.
- HIP0.8B/2B:16 fresh-process numerical/lifetime cells verified, including all256
  call records, fullV/all-state, independent roots/reset/restore, final readers,
  known release faults and unknown quarantine. Each model128 calls at context64;
  1,396,408,320/1,399,816,192 exact words. Unit success/MainPID0/empty cgroup.
  Healthy model owners retire; original process context remains. Not serving.
- CUDA worker acceptance discovered a pre-existing strict-layout regression:
  12-byte greedy packet described as one word. Fresh B2 reports exact binding
  refusal. Signed `8d761c4` validates full3-word packets in single/cohort
  paths; one-word kernel read and all original owners/strict checks unchanged.
  All14 isolated workers now pass (R3 ten, R4 four corrected wide EOS cases),
  including exact prefix/tail/width/token accounting and final physical-device
  reservation0/unfenced. HTTP R4 stopped before execution on an obsolete direct-GPU
  CLI feature argument; corrected facade-only R5 passes all12 guarded HTTP cases:
  JSON/SSE, disconnect/replay, unequal arrivals and native prefix rows2/4/8.
  Original failed/panic-contaminated scopes retained; final host gate passes below.
- Trained Gemma CPU matrix:7 prompt cases,50 natural calls plus completed refusal/
  cancel; all52 comparisons exact, full boundaries/KV/logits/actual IDs through
  frontier64, source closed/final host0. Not serving/performance qualification.
- HIP SDK passes separate fresh completion/presentation-error children, exact
  pre-cancel and unchanged2GiB host/4GiB device baseline. Original cold-only warm
  refusal is asserted, not disabled. Caller wrapper retains matched ld.so via
  original argv executable. Loader/fixture/constant-assert lint failures retained;
  release guard preserved, cancellation assertion strengthened.
- Gemma HIP8/Vulkan10 matrix/unknown/final-reader/partial-native/same-PCI scopes
  independently verified. Both resident matrices preserve full35/all KV/fullV,
  actual source IDs,50 observer-free state checks and selected-row reuse. Known
  final readers unload184/183 modules; faults retain originals/fence20GiB; streamed
  HIP unknown fences16GiB. Vulkan retains two actually constructed partial modules;
  both same-PCI directions refuse peer work without root/queue/upload effects.
  Healthy Vulkan host baseline is10GiB (2GiB context+8GiB compiler), not global0.
  One unobserved Vulkan hit makes no observed-full35 claim. Initial stale BF16
  cancellation-label failure retained; production cancellation/math unchanged.
- Approved cache/private HF-import routes pass both GGUF models: fullV/all42
  domains, source mutation/reclamation isolation, independent sessions and final
  raw-source lease release. No acquisition or original input mutation.
- Finite CUDA SDK/CLI dry-run and whole-card pre-Client controls pass. Fresh legacy
  0.8B cached47–50 result is byte-identical to the original captured evidence;
  no reference repinning. Thin CLI→SDK47-row/raw2-token generation emits ` a a`
  and retires cleanly. Missing exact legacy2B CUDA image remains an explicit limit.
- Finite27B teacher/unknown/final-reader6 cases inspected:240,896,000 exact words,
  full64/fullV/all128 physical planes (112 logical domains), independent sessions;
  separate Ready-discard/cancel preserve root. Known retires117 modules; faults
  retain117/1,602 handles/source and fence16GiB.
- Finite27B natural6 modes independently verified: genuine11→220→16, four main
  comparisons240,896,000 words, restore/late-cancel80,114,176 each, committed112
  domains exact. Healthy9 full64 calls; cached unknown keeps its80,103,936
  unavailable words explicitly unqualified. Known final readers unload121 modules;
  faults retain121/1,477 handles/source/fence16GiB. No serving/general-context claim.
- All six shared-HIP scopes verified. Known reload preserves live Gemma while
  retiring/reloading Qwen:3 full64 comparisons/160,781,824 words; Gemma1 observed
  full35 plus3 ordinary packet/full-KV calls. Final native sets/resident bytes0,
  device4GiB; host26GiB+128MiB+64KiB includes24GiB compiler, not global zero.
  Both final-reader faults retain117/184 original modules and fence32GiB. Actual
  source-page query/wait/destroy injections refuse both sources/reentry before
  publication, with unchanged bindings/ledgers/native state. Those three cases
  complete no full model; destroy has no pending-window witness claim.
- Final19-process supervisor (SDK+18 finite scopes) succeeds/exit0. Unit collected,
  inactive/MainPID0/cgroup absent; no owned executable and GPU idle. Monitor stopped.
- Combined CUDA/HIP and shared-HIP strict pass; Vulkan examples retain two known
  unused-probe warnings. Final isolated gate passes2,340 Rust/0fail/97ignored/
  307 suites (72 empty); Python17/182/10skips. Source review complete; CUDA repair
  signed `8d761c4`, Gemma diagnostics/SDK signed `8c43844`. No math/owner/timeout/
  tolerance relaxation. All19 S/F/repair signatures through `8c43844` verified.
- Logs: XDG state `abird/zerogravity/fabric-acceptance-oEz7801w/`.
  Owning transition TODO records exact acceptance and remaining unsupported cells.
  No new matched performance or broader serving claim; deferred mission preserved.

## Previous signed checkpoint — source/host parity `fef7123`

- SDK `run::config::resolve` and typed PipelinePatch/ChatContextPatch now own run
  precedence, graph overrides, path anchoring and provenance. CLI keeps argv/XDG/
  environment/signals/presentation. Public example and genuine tiny CPU planning
  fixture use the same resolver; no source/device/network IO from config resolution.
- Current feature matrix source audit reconciles S1–S3/F0–F4 ownership: original
  codecs/numerics/layout, typed CPU/window/Gemma/resident/captured adapters, one
  ForwardPlan/source issuer, SessionAuthority/arbiter, shared worker and transport.
  No forced TensorSource for windows, shadow interpreter, scheduler or publisher.
- Whole API/provider/status/alias/CLI review found and closed the config policy gap.
  Per-model unload, paged restore, partial-region/placement compilation, derived
  quantization and future tasks remain unsupported/descriptive, not new capability.
- Seven SDK controls pass default/minimal; thirteen existing CLI controls pass;
  combined CUDA/HIP, shared-HIP and minimal strict pass, CPU-free/cache-free Vulkan
  compiles. Approved direct CPU18.19s and HTTP18.55s pass; cold example help passes.
- Final isolated host gate: 2,340 Rust executions/0fail/97ignored/305 suite runs (70 empty); 17 Python suites/182 cases/10 skips.
  RUST_TEST_THREADS=4. Counts are suite executions; ignored/empty scopes are not
  model/device proof. Initial new context-fixture failure retained and corrected
  without weakening validation. No timeout/assertion was relaxed.
- Source review preserves existing merge/validation bodies after provenance and
  formatting normalization. Model/CPU/GPU math, roots/resources, workers, serving,
  kernel images and source pins unchanged. Historical physical evidence stays old.
- **Fabric remains unfinished:** applicable Q0/Q2 trained-source and physical GPU
  numerical/state/cancel/fault/final-reader/serving acceptance needs separate
  authority. No current compile/host/source-ownership blocker is known; do not
  redo extraction or fabricate future features while physical acceptance is held.
- Main only; both recurrences cancelled. No GPU/trained-Gemma execution,
  acquisition, push, agent restart or recovery/privilege change. Protected README
  and external detached checkout unchanged. Qwen/context/C1..C64/matched-vLLM and
  E01–E10 remain deferred. Logs: `abird/zerogravity/fabric-config-eJobTiNW/`.

## Previous checkpoint — source-owned preparation `860ce38`

- Follows `83060e5`/`2f19224`/`51cd294`; earlier numerical/layout, catalog, serving,
  SDK/artifact and shared-scheduler ownership remains accepted.
- ModelProgram retains the actual family/source and original ForwardPlan Arcs.
  Typed provider/ExecutableBundle preparation invokes the existing resident loader;
  ordinary GGUF catalog loading consumes it. Equal source IDs or copied graphs
  cannot substitute for either original Arc. No second IR, scheduler or publisher.
- ModelDefinition stays storage-neutral; only the existing tensor-provider binding
  requires TensorSource. Window-bound/Gemma/finite execution keeps original adapters;
  generic partial-region/placement compilation is not implemented. CPU keeps its
  existing source-bound TextSession/Executor, not a replacement graph interpreter.
- Fixed standalone text+GPU without cache: tokenizer admission depends on text;
  its original physical ledger/body is unchanged. No new acquisition dependency.
- Four host owner-routing controls pass; combined CUDA/HIP strict and CPU-free,
  cache-free Vulkan compilation pass. Approved sparse GGUF recording/direct CPU/
  audit/final-reader case passes17.42s; SDK/catalog/HTTP lifecycle passes18.48s.
- Final isolated source/host gate: 2,326 Rust executions/0fail/97ignored/303 suites (70 empty); 17 Python suites/182 cases/10 skips.
  RUST_TEST_THREADS=4 bounds harness concurrency. An earlier full gate failed the
  unchanged50ms HTTP header test; exact+25 HTTP controls and the bounded full rerun
  pass without weakening deadlines. Failed log retained, not accepted.
  Ignored/empty suites are not execution evidence; counts are suite executions.
- Source review retains original Qwen traversal/recording and resident preflight,
  initialization, lowering/math/event bodies after cold Arc normalization only.
  CPU math, workers/cohorts, serving, root/arbiter and kernel/image pins unchanged.
- **Fabric remains unfinished:** remaining family/provider and whole-feature/
  root/transport/API/CLI reconciliation plus F4/Q acceptance. Per-model unload,
  paged restore and future tasks still refuse; status/admin intent is not capability.
- No GPU/trained-Gemma execution, acquisition, agents/recurrences, push or recovery
  change. Physical numerical/state/cancel/fault/final-reader/serving acceptance
  needs separate authority; no historical evidence retagged or speedup claimed.
- Logs: XDG state `abird/zerogravity/fabric-parity-A94JdkGR/`. Failed compile and
  feature scopes retained separately. Protected README/external checkout untouched;
  Qwen/context/C1..C64/vLLM and E01–E10 remain deferred, not discarded.

## Previous checkpoint — shared scheduler `ae7ec11`

- Signed source/host checkpoint `ae7ec11` follows foundation `7fc5d17`/`8591b52`
  and lifecycle `d78eb55`. CPU and GPU use one engine Router, alias registry,
  admission/event path, fair worker, parking and cohort policy. CLI scheduler
  copies and CPU/GPU transport dispatch are removed.
- Facade owns ordinary GGUF and separate captured-diagnostic typed adapters;
  original source/session/Ready/commit/feedback and physical arbiter remain the
  authority. Public `load_gpu_text` shares the runtime; cold refusals stay early.
  This is shared implementation, not simultaneous heterogeneous/multi-model support.
- New controls/scratch are admitted. Final-reader credit survives the model
  handle; tokenizer and retained inventory share one owner. CPU32-token quanta,
  original GPU wave limits/bands, cancellation/discard/fatal distinctions and
  complete publication remain explicit. No kernel/math/image/manifest changes.
- 85 engine controls, GPU loader/reader4, tokenizer6 and public GPU SDK3 pass;
  CUDA/captured and HIP strict scopes, Vulkan compilation and25 source-free GPU
  feature controls pass (2 physical cases ignored). Genuine CPU GGUF fixture
  cases4 pass: SDK/tasks, aliases, JSON/SSE, cancellation, server stop and retirement.
- Final isolated gate: 2,306 Rust passes/0fail/94ignored/300 suite runs;
  17 Python suites/182 cases/10 skips.
  Architecture, feature/minimal/strict, release/installer/shell/Nix pass. The two
  existing Vulkan diagnostic source-page warnings remain. Failed compile/lint
  attempts are retained; empty targets provide no execution evidence.
- **Physical GPU regression is unrun and required before qualifying moved
  execution.** No GPU launch, model acquisition, privilege/recovery change or push;
  no old source/ABI evidence retagged, trained-model or speedup claim.
- Main-only; no new subagents. Both recurrences remain cancelled; root README
  and protected external checkout unchanged. Remaining SDK/CLI business logic,
  active contracts, F4/Q0/Q1/Q2 and original Qwen/context/C1..C64/vLLM work stay open.

Recovery: [transition TODO](../plans/fabric-vnext-transition/TODO.md).
Evidence: XDG state `abird/zerogravity/fabric-scheduler-YeCrPeT8/`, especially
`assembled-gate-r1.log`, `cpu-http-r1.log`, `backend-feature-r1.log`,
`shared-strict-r4.log` and `source-review-r1.txt`. Earlier foundation/lifecycle
logs remain separate. Direct Router/implicit self-drop still need external
retirement; arbitrary callback reentrancy and caller deadlines remain open.

The following sections retain earlier checkpoints/design context. Latest source,
tests and the transition TODO outrank their superseded continuation instructions.

## Historical direction — Fabric first (31 Oct)

Owner authorizes [Fabric transition](../plans/fabric-vnext-transition/TODO.md): reconcile
both plans against signed`1c35172`, then complete compilable S0 and
feature-preserving S/F migration. Qwen remains deferred, not completed;
recurrence`17aafa71` stays cancelled. No new hardware campaign or production
route. Plan 1's full design is retained in Plan 2's S0–S3/F0–F4/Q0–Q2 ordering.
See [source delta](../plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md) and
[feature preservation](../plans/fabric-vnext-transition/FEATURE_PARITY_MATRIX.md).
Signed`7396984` lands verified S0;`c853ebb` records reconciled recovery.
Owner now requests continuing until the full runtime transition is finished.
S1/F0/F1 implementation is active; no new accepted milestone yet.

## Interruption recovery — active contracts after SDK/CLI ownership

Start with the transition TODO and current `main`/diff. The source/host checkpoint
above supersedes old S0/lane instructions below. No new agent or recurrence;
source/build ownership stays with main. Do not restart GPU work without separate
authority. Continue active-contract/provider/root and final parity work without redoing
accepted scheduler or SDK extraction. Physical/trained cases stay held; source
compilation and existing local files never substitute for execution authority.

## Historical Qwen pause — 31 Oct 2026

- Owner requested a pause at the next reasonable checkpoint; direction may change.
- Signed implementation checkpoint: `e2559b6` (native chunk GDN); CPU predecessor
  `55536ce`. Final native/operator, strict-feature and isolated full gates pass.
- Recurring mission`17aafa71` cancelled, not completed. No owned native/build job
  remains running; terminal evidence saved and owned native units stopped.
- Do not resume implementation, re-arm recurrences or rerun accepted scopes from
  stale automatic receipts. Fabric is now the explicit direction; Qwen stays deferred.
- Functional CPU/CUDA model coverage is the user's meaning of parity; numerical
  contracts may differ. Blanket coverage is not verified. Original HF CPU captures
  are reference evidence, not ZeroGravity CPU execution qualification.
- If resumed: multi-row source-model composition/call-free lowering is next;
  broader contexts, general27B, C1..C64 serving and matched vLLM remain open.
  Production stays disabled. Preserve exact CUDA reference/runtime authority gaps.

## 1. Authority and read order

Authority, highest first:

1. Current source and current tests.
2. AGENTS.md repository rules.
3. The four-file maintained planning/status spine:
   - ../plans/README.md
   - ../plans/TODO.md
   - ../plans/CURRENT_STATUS_AND_TRACKING.md
   - ../plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md
4. This handoff, for cross-window context and ordered navigation.
5. The owning subsystem plan/specification for the work being changed.
6. Fabric vNext / transition plans when the user explicitly asks for that architecture work.
7. Git history for superseded plans, old handoffs, removed run artifacts and old branches.

Do not resurrect deleted handoff archives, transient .agents/runs trees, old side branches or old worktrees as live provenance.

Current owned CUDA implementation checkpoint: `2983bb7` (signed). Exact `.8B`
cached47→50, lifecycle/capacity and source gates pass; broader reference scopes
remain open. Shared source binding `99f0232` and finite27B HIP natural/lifetime
checkpoint `825ade8` are signed. New signed `d4d5df4` fixes CPU attention PV;
`f6fa4b9` shares source traversal and actual HIP attention. Independent CPU/HIP
operator comparisons pass1,774,240 words through64 active tokens;150 native
modules retire and the durable parent exits0/empty cgroup. Final source gate
passes2,135 Rust/17 Python suites. Ordinary HIP/model contexts/serving/performance
remain open; operator data is not trained-model evidence. Signed `8a1c6b1`
restores initial GDN empty-state addition; `1f29a42` adds actual source-bound
0.8B/2B HIP through64 populated single-row positions, fullV/all state/two natural
replays and healthy lifetime controls. Final gate passes2,136 Rust/17 Python
suites. Canonical cache/native unknown/fault lifetimes now pass the inspected
r1/r2 qualification below and the final gate; signed checkpoint `85f33c7` landed.
Signed `55536ce` CPU padded GDN matches64,345,520 original words; its final gate
passes2,142 Rust/17 Python suites. Signed `e2559b6` adds native chunk GDN:
64,345,520 original words exact, four status-only Ready controls and native
retirement pass; final source gate passes. Full-model multi-row integration,
scalable ordinary execution, general27B, contexts/concurrency/vLLM remain open.
No pushes. Re-read git status/log.

## 2. Preserved Qwen checkpoint (deferred behind Fabric)

Immediate continuation is Fabric S1 in the transition TODO. The Qwen details
below preserve accepted evidence and future work; they are not restart orders.

**Paused owner mission (resume only on explicit direction):** finish CUDA `.8B`/`2B` on `pvl-l5`, ROCm `.8B`/`2B`/`27B`
on `pvl-x2`, genuine active-context fit, then C1/C4/C8/C16/C32/C64 serving and
measured vLLM superiority in every required feasible run. Use high concurrency
on sufficiently capable hardware, especially x2. Routine x2 resource use needs
no further permission. Correctness precedes timing; do not cherry-pick losses.
Check cached models first; needed2B/9B/27B model downloads into canonical XDG cache
are authorized. No subagents; stop to ask only for consequential external inputs
or decisions. The earlier blanket benchmark deferral is superseded, not numerical,
source/device or ownership requirements. Missing wider CUDA references do not
block independent ROCm work. Finite27B diagnostics alone do not finish the mission.
Current results are in section8 and the status/numerics TODO.

**Next lane if resumed:** ordinary0.8B/2B HIP. Signed `1f29a42`
assembles typed-SSA state lowering, shared resident matrices/workspace, cold
pricing and the original session/root machinery. Source arithmetic remains
bounded to64 positions; finite27B guards stay separate. Strict standalone/HIP
and shared Vulkan checks pass. Both actual-source cold plans pass64 positions
and11 refusal/accounting controls; this does not qualify C64 execution. The first
native attempt prepares785 modules, then refuses binding before initialization
Ready. Its exit1/empty cgroup remains failed. Post-Ready binding reaches actual
0.8B full-model comparison:303 mismatches, first after.recurrent/layer0 word50816
(`80000000` vs `00000000`). All785 modules retire, namespace/source close; parent
exit1/empty cgroup verified. The initial recurrence omitted original empty-state
addition; CPU/HIP now retain it, without canonicalizing observations. Four host
controls and standalone strict check pass. Fresh r3 passes both actual models:
64 populated positions × two natural replays, fullV/all24 layer outputs/all
populated state, all15,776 records/model.0.8B/2B compare1,396,408,320/
1,399,816,192 words exactly;785/781 native modules retire. Known cancellation,
discard, pressure, independent roots, restore/reset and shutdown pass. Final
assembled gate passes2,136 Rust/0fail/85ignored/283 suites,17 Python suites/178
cases plus release/installer/shell/Nix. Signed arithmetic repair:`8a1c6b1`;
source-bound execution checkpoint:`1f29a42`. New source/cache and unknown/fault
qualification follows below. Scalable ordinary paths and production/serving
remain unaccepted.
Independent original HF captures are now complete: each small model has64
populated positions × two exact replays, fullV/all layers/all populated state;
all15,776 records/model inspected, original source and parent/group/container/
EOF closure verified. The durable local reference unit completed and stopped.
Evidence:`ordinary-hip-20261031.Jc5XWh/small-original-r1-{08,2b}`. Do not repeat
finite27B/attention/reference capture for activity. All29 approved source/reference
members are transferred/verified on x2;0.8B is imported canonically with original
legacy provenance. `2B` is now canonically imported as the unchanged snapshot
`88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52`,12 exact
members, no download. Failed r1/r2 units remain stopped; r2 invocation
`cd0d5df490124e75b5f3ed7dc7884a79`, watch`e8a1de06` completed. Preserve
`source-model-physical-r2-08.log` and its source/binary/parent identities.
Completed r3 unit:`zg-source-models-20261031-r3.service`, invocation
`3839671566394ffa83c111b9d3c3dcb6`; watch`fa8b160b` succeeded. Archive SHA256
`1f79d6e982b62534a89a2632b8de6a42179145d6d8f567d28f87433c7df4bbef`.
`source-model-physical-r3-{08,2b}.log` and every call/control/retirement inspected
against original references in `source-model-all-scopes-r3.log`; parent exits0,
MainPID0/empty descendants. Completed owned r3 unit stopped. This is
single-row position-bound execution through64, not call-free scalable prefill,
concurrent serving or performance acceptance. Final local gate job766 passes:
CPU/diagnostic/HIP/shared-Vulkan strict checks, both actual-source cold-control
sets and isolated `scripts/check.sh`. Only the fail-closed identical-capacity
check changed after physical r3; fresh cold controls cover it, with valid model
arithmetic/storage/ownership unchanged. No physical replay claim for that guard.
**Qualified source-lifetime unit:** native source loading retains the actual
canonical `SnapshotLease` and duplicates its pinned descriptors. Native staging
refuses; cold preflight still permits staging. New diagnostic controls reuse
original pending-flight abandonment and final-reader release faults. Two host
schema/provenance/path controls and standalone/shared-HIP/shared-Vulkan strict
checks pass. All14 r1 scopes are now independently inspected: both models,
Session/Snapshot/Pending known/fault readers, unknown completion and canonical
pins. Healthy readers release785/781 modules and all11/12 pins; unknown/fault
retain original source/cache/native/service with a24GiB physical fence. Unknown
retains752/747 handles,189 completion owners and one pending event, no cached
root publication or logits claim. Parent exits0/MainPID0/empty descendants;
completed owned unit stopped. Watch`264d971a` succeeded after63 checks; a later
receipt is stale. Evidence:`source-lifetime-all-scopes-r1.log`.
Closed unit:`zg-source-lifetimes-20261031-r1.service`, invocation
`ce8715bf40f0403fbb4a01f916f5115c`. Source archive SHA256
`bdae47d11bbd8a1bf2c5f9a6815cca3a6bb7f0b46f88c005824ea5d7f3fa5395`;
binary`17e767bc9e21f9558d51543083c62507d04f7b529b1ac8c0599fc01e7440e3a5`.
Current main additionally checks healthy physical-fence clearance/other-owner
admission. That control was added after r1 froze; all six healthy-only r2 scopes
pass and are independently inspected. Original4GiB context charge remains,
physical fence is clear and other-owner admission succeeds. Source/cache/native
owners retire; parent exits0/MainPID0/empty descendants. Completed owned unit
stopped; x2 is free. Closed unit:`zg-source-lifetimes-20261031-r2.service`,
invocation`cad7ee91e0f44e83a7fbf284bed7661b`; watch`9a53c915` succeeded after25
checks, later receipt stale. Source archive
`82de51f7eb7207cbc784f07314a62cd02a1d7846dbdc9968397ad698b3ffc61b`;
binary`55f49e93ce190fd1f3b96370bdc0ef1628059786764b4fee38001c61a54d1b6a`.
Evidence:`source-lifetime-all-scopes-r2.log`. Final isolated gate job810 passes
2,136 Rust/0fail/85ignored/283 suites and17 Python suites/178 cases; existing
non-strict warnings/skips remain scoped. Signed checkpoint:`85f33c7`. Landing
plan reconciliation and the next arithmetic/test-support unit remain dirty; no push.
Lifetime arithmetic is unchanged; accepted r1 fault/unknown and full64/finite27B/
attention evidence is reused. Preserve historical failures/unrun scopes. Scalable ordinary execution
remains next: CPU GDN currently has only `InitialOne`/`CachedOne`; source records
and attention arithmetic cap64. Extending metadata/capacity alone is not prefill
or multichunk support. Production/serving remain disabled.

**Next implementation unit:** genuine multi-row source math and call-free HIP
lowering. Its independent original CPU references now pass complete inspection:
fresh47/48/49/50 plus two natural decode calls; uniform64-row and ragged47/64/17
prefill consume identical512 input IDs, then512→513→514. Two replays/model,
66 forwards/7,608 records each, full last-row V/all24 layers/all populated state.
0.8B/2B contain873,649,152/939,144,192 float words; every record/selection/source
mapping/replay and child/group/container/EOF closure inspected. No HIP claim.
Case/input data:`crates/zerogravity/tests/support/qwen_source_calls.json`.
Closed local unit:`zg-source-calls-original-20261031-r1.service`, invocation
`dca08936d8e649b686a6e0b06e758d0e`; watch`d35de2b5` succeeded33 checks.
Both child/parent exits0, MainPID0/empty recursive descendants saved; owned unit
stopped. Later receipt is stale. Run:`ordinary-hip-20261031.Jc5XWh`;
`source-calls-original-all-scopes-r1.log`, `source-calls-original-r1-{08,2b}/`
and `validate-source-calls-r1.py`. Original chunk-sensitivity analysis is separate:
uniform/ragged512 prefixes differ in7,563,855/7,532,508 state/logit words despite
both choosing11; same-plan replays are exact. No chunk-invariance/quality claim
or relaxed contract. No active local/x2 model supervisor remains.
Next backend work must reuse call-free source invocations/original owners,
not widen the64-body diagnostic catalogue. Actual multi-row GDN includes the
original padded64 triangular solves; CPU-source eager attention/BF16 residuals
also differ from CUDA contracts. Production/serving remain disabled.
**Current internal arithmetic work:** original padded-GDN capture is now fully
inspected:19 synthetic cases,484 stage/input/output records,74,948,432 float words,
exact two-replay results. r1 traced the wrapper rather than the original body and
failed; its exit1/closed ownership remains preserved. r2 observes
`inspect.unwrap(fn).__code__`, invokes the unchanged public function and succeeds.
Closed unit:`zg-gdn-chunk-original-20261031-r2.service`, invocation
`1deb9d18708b468282d3c78c406e4ab1`; child/parent0, empty descendants and complete
container/group/reap/EOF closure saved; owned unit stopped. Watch`9906f901` belongs
only to failed/closed r1. Evidence:`gdn-chunk-original-validation-r2.log`,
`gdn-chunk-original-r2/`, terminal/stop files under the same ordinary-HIP run.
New CPU `numerics/torch_cpu/gdn/chunk.rs` implements the padded unit-lower solve:
nonfused products, four lanes, `(s0+s2)+(s1+s3)`, scalar tail, one RHS subtraction.
Four host/refusal controls, strict CPU checks and all38 original solves/
7,077,888 words pass. Test data stays under Rust test support.
Matrix comparison matches114 source stages/21,233,664 words, including state
updates. Raw-dot attempt failed685 signed-zero words in stressed initialized
inter-chunk output. Original Zen GEMM loads/adds initialized C after its FMA
chain; retaining that arithmetic resolves them without normalizing observations.
Every stage inspected in `gdn-gemm-inspection-r2.log`; initial mode/ordinal
inspector mistake preserved.
Connected CPU `gdn/chunk/core.rs` now implements the whole padded chunk core:
source L2/scale, FP64 cumsum, causal decay, both solves, all matrix/state updates
and BF16 output. New nonpositive exp includes the original rare FP64 path
(`vsexp_cout_rare`, ELF0xb41a130), not scalar-libm substitution. Its64 exp2 split
pairs are independently derived at180 digits and match all128 original words.
Existing normal-profile guard |x|<=87 is unchanged. Original -inf causal masks
are explicit; supplied log-decay remains finite/nonpositive.
All19 cases/380 computed/returned fields/64,345,520 words compare exactly from
original inputs, including initialized state/strong decay; no intermediate
reference is used as an operand. Strict CPU and60 host tests pass/8ignored.
Retain initial iterator lint failure. Final review aligns observer order and
adds cumulative-overflow/observer-refusal controls. Fresh r3 original comparison
and CPU/diagnostic strict pass. Final isolated `scripts/check.sh` passes2,142
Rust/0fail/87ignored/283 suites,17 Python suites/178 cases plus release/installer/
shell/Nix. Existing feature warnings and skips remain scoped; no physical replay
is inferred. Signed checkpoint `55536ce` contains only this arithmetic, test support and
plans. Post-commit plan grouping is carried into the next substantive unit.
Evidence:`gdn-connected-{original,host,strict}-r{2,3}.log`,
`gdn-connected-diagnostic-strict-r1.log`, `gdn-connected-assembled-gate-r1.log`.
No model/call-free/HIP integration yet. Next implement actual HIP chunk math,
then multi-row projection/Conv4/attention and call-free lowering. Source
machine-code inspections are run-local, not a replacement runtime or permission
to call private ABIs. Local gate job875 completed0; no owned build/model job
remains. Production/serving stay disabled.
Read-only x2 inventory additionally resolves all25 dangling images' public name
histories: application/database/Ollama images, no named vLLM candidate. Only
public repository/name fields requested, no container/device work or download.
`x2-untagged-runtime-inventory-r1.log` is name metadata, not exhaustive package
inspection or baseline authority; compatible approved ROCm vLLM remains open.

**Native chunk continuation:** shared `source_gdn/chunk.rs` and separate rare-exp
GPU arithmetic now compile; no model lowering/production selection exists.
Test-only composition uses original source/native/service/queue/flight owners,
256MiB admitted host workspace and64MiB case scratch, streams independent input
records and compares all stages/returned state. It also reads back all inputs/
prior state and tests four numerical refusals without exporting math through Ready.
Original transport only widens BF16 bits/reorders records; SHA256
`0b314ec242346caec8d7f2ed2d0e45647cf8eed37649f97c66f907c5a657a071`,
299,795,904 bytes. R1 macro path and r3 ambiguous slice type compile failures
are retained; strict r4 passes. Frozen source archive SHA256
`79dac12c18259f5dea34b1199b15b0f6e32232a9050850b99551fd384cc0d5ad`,
40,560,640 bytes/2,148 files, base55536ce plus dirty implementation, transferred
with all bytes authenticated to bounded non-Git x2 scratch. Durable unit:
`zg-source-gdn-chunk-20261031-r1.service`, invocation
`fc0379a845354d57be6cc34f23ee5174`, initial MainPID974422. Watch`fba2bb67`
observed terminal failure: release build passed, native refused `HIP original
complete source catalogue metadata owner` before all19 cases. The diagnostic
requested4MiB against the exact1MiB original contract. Parent101/MainPID0/empty
recursive descendants, source/binary/log identities saved; unit stopped (failed
state retained), starter905 closed. No numerical/in-process retirement pass.
Caller now requests1MiB; original guard/arithmetic unchanged. R2 frozen archive
`e3387a9ed607518c54f8425e6c39f7d2484811222f2b92222e2aa2ff476543b2`,
invocation`ee5c37a887b94376a9c2e7837110f81b`, watch`39fc9f3d`: all19 cases/
380 fields/64,345,520 words exact; every input/prior unchanged, per-case resident
baseline restored. Independent all-scope inspector reconciles the pinned original
manifest. First negative control fails a fixture mistake: completed Writes are
physically initialized regardless of Ready export selection. Remaining3 controls
and final retirement unverified. Preserve failed101/empty descendants/evidence;
owned unit stopped. Fix diagnostic only: exact original service/event/root context
and export cohort, negative Ready contains only status/no math. No completion or
arithmetic change. R3 archive
`2afc28f5a1987ec04e84e4b858f2672a848fb625e499336ab533c0fc1ffa5715`, binary
`3c5db98e1c2dfdff7cc279aa6734d2d63aa5df85d99304a998e521fb6a69db79`, unit
`zg-source-gdn-chunk-20261031-r3.service`, invocation
`01091a34a8f54d6c92a2bb95ee11b8f0`, watch`90ef500e`: all19 cases/380 fields/
64,345,520 words exact; all4 numerical status controls retain inputs and export
only status, never math, through original Ready. Physical writes complete as
specified; no model publication exists. Per-case resident baseline intact, all20
native modules/code/host/device retire, namespace closes. All scopes independently
inspected; parent0/MainPID0/empty recursive descendants saved; unit stopped.
Shared HIP/Vulkan strict checks pass. Final isolated `scripts/check.sh` r2 passes
2,142 Rust/0fail/87ignored/283 suites,17 Python suites/178 cases plus packaging.
R1 formatting-only refusal retained; module declaration order corrected without
math/ownership changes. Signed implementation checkpoint:`e2559b6`. Existing dotlink.service GPU context
was idle at inspection and remains untouched; preserve the initial overbroad
empty-FD preflight refusal. Non-timed operator run only, no performance or
exclusive-device claim. No full-model, serving or Vulkan physical grant.
Old receipts d35de2b5/9906f901 are closed captures; fba2bb67/39fc9f3d are closed
native failures. R3 is inspected/stopped; do not repeat it on a stale receipt.
Details remain in model-execution/numerics plans.

Repository:

    /home/pvl/spaces/abird/src/zerogravity

Persistent topology:

- main is the only persistent local branch.
- main is the only authorized implementation checkout; do not create another.
- 31Oct preflight also found an external detached Kilo scratch checkout; it was
  not created, read or modified by this lane. Preserve other owners' work.
- origin/main is the only remote branch currently relevant.
- no stashes.
- do not create durable side branches/checkouts as coordination state.
- do not push/tag/release unless the user explicitly authorizes it.

Recommended startup:

    export CARGO_BUILD_JOBS=8
    export RAYON_NUM_THREADS=8
    export PYTHONDONTWRITEBYTECODE=1
    git status --short --branch
    git log --show-signature --oneline -20
    git worktree list --porcelain
    nvidia-smi

Then read, in order:

1. AGENTS.md
2. this file
3. ../plans/README.md
4. ../plans/TODO.md
5. ../plans/CURRENT_STATUS_AND_TRACKING.md
6. ../plans/REQUIREMENTS_AND_TEST_TRACEABILITY.md

For the current Qwen correctness critical path also read:

7. ../plans/serving-and-speculation/TODO.md
8. ../plans/numerics-and-context/CORRECTNESS_POLICY.md
9. ../plans/numerics-and-context/TODO.md
10. ../plans/model-execution/TODO.md
11. ../plans/cpu-gpu-optimization/TODO.md
12. ../plans/memory-and-streaming/TODO.md
13. ../plans/serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md

For Fabric vNext work also read:

14. ../plans/fabric-vnext/HANDOFF.md
15. ../plans/fabric-vnext/README.md
16. ../plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md
17. ../plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md
18. ../plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md
19. ../plans/fabric-vnext-transition/HANDOFF.md
20. ../plans/fabric-vnext-transition/README.md
21. ../plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md
22. ../plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md
23. ../plans/fabric-vnext-transition/STRUCTURAL_SKELETON.md
24. ../plans/fabric-vnext-transition/PHASED_IMPLEMENTATION.md
25. ../plans/fabric-vnext-transition/FEATURE_PARITY_MATRIX.md
26. ../plans/fabric-vnext-transition/QUANTIZATION_AND_NUMERICS.md
27. ../plans/fabric-vnext-transition/QUALIFICATION_AND_COMMIT_POLICY.md
28. ../plans/fabric-vnext-transition/ROADMAP_AFTER_TRANSITION.md
29. ../plans/fabric-vnext-transition/DECISIONS_RISKS_AND_ACCEPTANCE.md
30. ../plans/fabric-vnext-transition/TODO.md

That is the intended next-window set. Do not load every historical document up front.

## 3. Original mission and enduring product design

ZeroGravity began as a Rust-native inference compiler/runtime and evolved into a broader execution fabric.

The enduring product goal is one Rust-owned execution fabric that can support:

- text generation;
- embeddings/reranking and encoder tasks;
- image generation/editing;
- video/temporal generation;
- ASR;
- TTS and audio codecs;
- realtime/full-duplex media;
- combined multimodal programs;
- future optional training/update tasks where explicitly designed.

The user-facing aspiration is simple:

- one binary;
- one Rust SDK;
- local-first operation;
- automatic hardware selection when safe;
- explicit fit/capacity explanations;
- OpenAI-compatible serving where implemented;
- no Python inference server dependency for qualified local paths;
- the same runtime for portable and highly specialized execution rather than a separate server per model or modality.

The performance goal is measured Pareto leadership where specialization is justified, never a blanket “always fastest” claim.

The architectural idea from the beginning has been:

- model mathematics and source semantics are defined once;
- state/effects are explicit;
- portable/reference execution exists first;
- optimized hardware paths replace closed semantic regions only when legal and qualified;
- one logical session/root authority publishes committed state;
- one shared physical resource authority owns real memory/device budgets;
- device workers own actual asynchronous compute/events but cannot invent logical commits;
- serving, model loading, scheduling and resource ownership are not cloned for each specialization.

The initial vision and full early architecture remain in:

- VISION_SCOPE_AND_PRINCIPLES.md
- ../plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md
- the current public Engineering Specification and Whitepaper under ../../docs/
- current Fabric vNext plans for the next structural transition.

## 4. Engineering philosophy and non-negotiable invariants

Priority order:

1. semantic/state correctness;
2. source and quality fidelity;
3. safety, ownership and crash behavior;
4. compatibility and user clarity;
5. stable latency/throughput;
6. peak specialized speed;
7. experimental complexity only when evidence earns it.

Rust design preference:

- clean, robust, intuitive ownership;
- shared lowest layer when it genuinely benefits multiple callers;
- zero-cost abstractions where natural;
- hot-path optimization where simple and maintainable;
- prefer a clean design over convoluted optimization when the two conflict;
- no second runtime/allocator/state publisher hidden behind “optimization.”

Hard invariants:

- exact source/model identity remains explicit;
- model metadata or a registry slug never grants execution support;
- a finite captured testcase is qualification evidence, not general runtime authority;
- prompt IDs, prompt lengths, ordinals, packet contents, capture hashes or nearest-case tables must not select ordinary production math;
- one logical StateRoot publication authority;
- one process-shared physical ResourceArbiter;
- actual completion/event/fence governs release;
- completion-unknown work is quarantined, not silently recycled;
- source readers/owners survive until actual final consumers retire;
- cancellation cannot publish partial state;
- unsupported shape/source/backend is explicit Unsupported/refusal, never silent fallback to another implementation;
- a passing cleanup or final case cannot relabel an earlier failure;
- compile/source-only success does not imply physical inference support;
- performance follows correctness;
- no universal numeric epsilon; each operator/model/profile owns its declared contract.

The shared correctness policy is:

    ../plans/numerics-and-context/CORRECTNESS_POLICY.md

It deliberately combines the useful ideas from vLLM and llama.cpp without blindly copying either:

- exact semantics/source/state checks;
- independent operator numeric checks;
- model-level output/distribution/quality checks;
- determinism/batch-invariance as separate claims.

## 5. Important rejected patterns

Do not reintroduce:

- one model-specific server per family;
- duplicate resource wallets;
- duplicate StateRoot publishers;
- generic production dispatch based on testcase/capture identity;
- arbitrary Hugging Face remote Python execution;
- metadata-only “support” claims;
- universal tolerance constants;
- hidden precision changes to hit benchmark numbers;
- KV-only prefix reuse for models with additional recurrent/SSM/GDN state;
- benchmark claims before exact source/output/ownership gates;
- a giant permanent graph interpreter in the token hot path;
- per-kernel trait/object hierarchies with no demonstrated need;
- durable agent side branches/worktrees as coordination state;
- monolithic generated handoff archives or documentation SHA catalogues.

## 6. Architecture evolution from the beginning

### 10–12 Oct: initial architecture and v4 consolidation

The early design converged on:

- semantic/effect-aware model programs;
- portable CPU/reference execution;
- optimized device regions;
- extreme model/hardware specialization as optional qualified providers;
- TaskProgram/TaskDriver for non-autoregressive tasks;
- explicit model/session/resource/device boundaries;
- multi-model routing and OpenAI-compatible serving;
- local-first model store/cache/acquisition;
- future SSD and distributed placement.

The v4 minimality principle reduced durable representations to the semantic program and prepared executable/bundle; lower-level compiler/device views are transient.

### 13–16 Oct: source formats, quantization, model families and GPU ownership

Work established:

- GGUF/safetensors/sharded source handling;
- Qwen3.5 and Gemma source/model contracts;
- CPU source-exact math and quant paths;
- GPU backend separation for CUDA/HIP/Vulkan;
- physical ResourceArbiter and source/device ownership;
- exact source/state/fixture policies;
- XDG cache/source behavior;
- early model acquisition/cache foundations;
- multimodal/task architecture research.

Qwen .8B and 2B became the immediate correctness vehicles.

### 17–22 Oct: full-model numerics, GPU native paths and serving foundations

Work expanded through:

- full 24-layer Qwen execution/reference comparisons;
- all 42 Qwen state domains;
- full 248,320 logits;
- RMSNorm/GDN/attention/FFN numerical investigation;
- retained source/window ownership;
- GPU native image/launch qualification;
- cancellation/fault/quarantine;
- serving queues, streaming and shutdown;
- OpenAI-style API subset;
- preliminary matched vLLM comparisons.

Historical numeric differences remain evidence, not current acceptance.

### 23–26 Oct: integration, routing, acquisition and broader backend work

Work covered:

- device routing, including exact /dev selectors where appropriate;
- CLI/config/source precedence;
- acquisition providers and offline policies;
- cache transactions/reclamation;
- pipeline/config validation;
- Qwen27 and Gemma finite qualification lanes;
- HIP/Vulkan qualification work;
- serving concurrency/queue scaling;
- shared state/resource ownership audits;
- repeated fresh bug/regression reviews.

### 27 Oct: Fabric vNext design

The user requested a clean future architecture where the whole target structure becomes visible early in code, then current implementation migrates into it.

The first structural milestone is intended to introduce model-neutral engine and reusable serving crates, visible typed contracts and explicit unsupported stubs before deep migration.

That was the design-only checkpoint. Signed`7396984` now implements S0; functional migration remains open.

### 28 Oct: ordinary source-derived Qwen CUDA integration

Current source now has ordinary Qwen3.5 .8B/2B CUDA execution assembled under the existing resident owner:

- one call-free ForwardPlan;
- authenticated source owner;
- request-local materialization;
- cuBLAS projection path;
- source-exact attention;
- arbitrary-row parameterized GDN;
- F32/F32/BF16 two-root state;
- one queue/flight/completion/quarantine path;
- existing StateRoot publication authority.

It remains qualification-only. No production ordinary CUDA route is enabled.

### 28–29 Oct: evidence hardening, cache human aliases and repository normalization

Recent work also:

- made cache by-name aliases crash-consistent and explicitly non-authoritative;
- simplified ordinary parity evidence so the checked-in capture procedure defines the reference and only the evidence content digest crosses processes;
- removed code/doc SHA catalogues from parity evidence;
- made compare-existing independent of the historical native bundle;
- retired machine-local capture-only tests and dead run-artifact references;
- reduced topology to main-only;
- pruned superseded handoff archives;
- normalized current planning provenance;
- reviewed and fixed environment-sensitive PID1 serving signal tests;
- added Qwen27 layer9 source-contract coverage to the shared check gate;
- refreshed vNext handoff method to use this canonical pointer + maintained files + Git history.

## 7. Current repository/runtime structure

Current workspace owners remain:

- zerogravity-core
- zerogravity-artifacts
- zerogravity-cpu
- zerogravity-gpu
- zerogravity-model-qwen35
- zerogravity-model-gemma4
- zerogravity
- zerogravity-cli
- zerogravity-gpu-qualify
- zerogravity-engine (cold model-neutral shell; F0 migration pending)
- zerogravity-serving (cold embedded API; F2 protocol migration pending)

Signed`7396984` adds the two Fabric crates without enabling new model routes.

Current logical ownership:

- core: semantic/state/resource/device/pipeline contracts;
- artifacts: source formats, immutable sources, cache, acquisition;
- model crates: family math/source/state/tokenizer semantics;
- CPU/GPU: physical execution providers;
- zerogravity facade: current public composition surface;
- CLI: still owns more reusable application/serving policy than final vNext design wants;
- serving currently lives under CLI and is intended to move to a reusable library during vNext.

## 8. Current Qwen ordinary CUDA status

### What is implemented

Ordinary .8B/2B source-derived CUDA is integrated under the resident execution owner.

Source-free tests establish:

- ordinary cancellation checks before allocation, before flight, during source execution, after launch/completion and before commit;
- armed-flight cancellation enters existing unknown-completion/quarantine behavior;
- explicit provider/pool shutdown;
- unique model ownership before shutdown;
- zero resident bytes/events before namespace close;
- source/provider ownership survives until retirement;
- no alternate root publication path.

The ordinary production route remains intentionally refused until real-current-source qualification closes.

### Exact approved .8B source

Repository/model:

    Qwen/Qwen3.5-0.8B

Revision:

    2fc06364715b967f1860aea9cf38778875588b17

Visible approved source path:

    /home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17

Weight shard:

    model.safetensors-00001-of-00001.safetensors
    bytes: 1,746,942,600
    SHA256: 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696

The tracked source manifest is:

    crates/zerogravity/src/qwen35_cuda/source08.json

### Recovered reference and original blocker — 29 Oct

The old assembled bundle was purged, but its exact `.8B` N1 inputs are recovered
offline under XDG state. Retained canonical images plus two whole SM80
FlashAttention modules extracted from the pinned local reference image and the
original cuBLAS providers match current Rust identities. All 11 model members
verify; maintained bundle validation and isolated structural preflight pass.

Fresh phase-1 cached47→50 reference capture passes after selecting the existing
8 GiB reference-fixture host profile, with all 1,008 source windows retired.
The original 4 GiB refusal stays failed. The independently emitted evidence
content digest is retained in the owning numerics plan and external evidence.

Fresh ordinary comparison fails before model math: pinned `cublasCreate_v2`
returns status 1 after the owned allocator denies a private `cuGetExportTable`
call. Zero allocation attempts/charges and bounded resolver observations identify
this as a provider ownership blocker, not missing model/bundle bytes. Do not
forward opaque tables or preload cuBLAS around admission to make it run.

The current evidence and remaining work live in
`../plans/numerics-and-context/TODO.md`. `.8B` cached47→50 is exact; broader
lifecycle/context and production routing remain unqualified. Owner order is CUDA correctness first,
then ROCm on `pvl-x2`, with no subagents.

Maintained tooling:

- tools/qwen35-cuda-reference/
- tools/qwen35-cuda-bundle/
- scripts/qualify-qwen35-ordinary-cached47-50.sh

The Python discovery/packed-capture protocol still requires explicit retained
external capture sources. Exact offline bundle recovery is documented in the
bundle helper README; the Rust two-process parity wrapper uses that recovered
bundle. These are qualification inputs, not production authority.

### Owned execution checkpoint — `2983bb7`

Owner approved replacing ordinary opaque BLAS with shared driver-only GEMM.
The allocation-free tensor-core/SIMT primitive passes 180 synthetic cases and
eight authenticated prefill/cached projections exactly, including guards and
retirement. Typed SSA roles choose explicit reductions without captured node or
prompt lookup. Original compiler-context reservations and physical flights remain
the owners; pinned reference/source/resource limits are unchanged. Ordinary
comparison no longer requires cuBLAS paths.

Corrected the rotary admission omission (+280 bytes), GDN partition binding and
premature BF16 rounding of residual RMS sums. Row RMS retains Torch vector-four
FP32 boundaries; Q/K RMS instead follows the fused Triton adjacent-pair/warp
reduction. CUDA rotary now uses FP32 FMA and final BF16 rounding; the CPU-target
BF16 product contract is unchanged. All211 GPU host tests (43 ignored) and an
independent host-FMA rotary comparison at14 positions through32767 pass.

Owned tile-64 CUDA attention now preserves FlashAttention's BF16 unnormalized
probabilities, FTZ FP32 online softmax and tensor-core QK/PV. Runtime geometry,
not captured-case routing; no new device scratch or allocation authority. Both
independently authenticated prefill/cached attention captures match exactly.
Nine analytic causal/multitile cases through a32K active prefix and three
nonfinite refusals pass with guards and retirement. They are not long-context
model parity.

Fresh `.8B` cached47→48→49→50 passes exactly: all248,320 logits, all42 state
domains and natural feedback at every step, with normal model shutdown. The
reference bytes/digest are unchanged. The earlier229,459-logit /4,359,077-state
failure (max0.28125) remains failed historical evidence.

Replaced obsolete row/residual CPU-profile CUDA oracles with independent staged
CUDA/Torch comparisons:16 cases across widths1024/2048, rows1/2/17/64 and residual
modes pass exactly, including non-BF16 residual sums, guards and retirement.
Real post-launch cancellation and abandoned-future probes pass:396 original
handles quarantined, prior bits/tails intact, unchanged root, peer refusal and
retained charges after external drops. Normal lifecycle passes all13 admitted
capacities (`.8B` through32K,2B through16K), each at48 active tokens;2B32K refuses
at pure preflight. Long-active-context/2B numerical parity remains open. Final
isolated repository gate passes2,129 Rust executions/0fail/83ignored;14 Python
suites and ancillary gates pass. CUDA/native host675/0fail/54ignored and strict
CUDA/HIP/Vulkan feature checks pass.
The2B approved snapshot exists in the canonical cache; its extra reference image
must be recovered independently, not replaced by the `.8B` inventory.

Owned execution landed in signed main commit `2983bb7`; nothing is pushed and
production remains disabled. The extra exact2B reference image was not found
among1,252 extracted SM80 modules; retained original capture inputs/long-context
comparisons are also missing from the current qualification root. Do not replace
or repin them. ROCm is now active on `pvl-x2` gfx1151/96GiB, independently of
those CUDA input gaps. Current source is in bounded run-local execution scratch,
not another Git worktree. The exact31-member Qwen3.8-27B snapshot was imported and
verified in canonical XDG cache from existing inputs; no download was needed.
Fresh independent natural full64/fullV/all112-state reference capture validates
both seed11→220→16 cohorts and source/process/FD closure. Capture/validator
protocol repairs have14 Python controls;26 Rust comparator/admission controls and
narrow/shared HIP strict checks pass. Signed `99f0232` fixes rank3 Conv4 preflight
through the shared exact binding/30 source-window tests. SDK-matched child-only
loader resolves build-libc versus GLIBC2.44 runtime refusal without replacing
SDKs or changing shell libc. Actual teacher full4 and healthy natural9-call reports
match full64/fullV/all112-state exactly; all six repaired natural modes pass
semantic validation. Known reader paths retire121 modules; release faults retain
121 modules/original service/source/1,477 handles. Teacher unknown and independent
Pending/Snapshot final-owner scopes also pass. Earlier failed scopes stay failed.
The old healthy controller exit record was lost; that controller scope stays
incomplete. A fresh durable healthy-only replay passes with exit0 and no remaining
processes. Signed `825ade8` lands the bounded diagnostic repairs. Source gate
passes2,132 Rust/17 Python suites with explicit skips. Genuine ordinary contexts, small-model HIP, C1..C64
and performance remain open; production stays disabled. See the owning numerics
TODO for exact scopes and continuation.

Do not recreate deleted `.agents/runs` paths or treat old run names as provenance.

## 9. Ordinary cached47→50 parity protocol

Current parity design was deliberately simplified.

The checked-in reference capture procedure defines how phase 1 is produced.

Evidence schema v3 contains:

- source coordinates;
- prompt;
- 47→48→49→50 observations;
- full logits;
- all state domains;
- selected IDs/max-bit receipts;
- teardown/close receipt;
- an embedded corruption digest.

Cross-process trust:

- phase 1 emits the evidence content SHA256 in stdout;
- normal wrapper execution carries that digest in memory to phase 2;
- compare-existing requires the independently supplied trusted digest;
- the embedded digest is corruption detection, not a self-authenticating implementation identity.

Removed intentionally:

- reference implementation SHA graph;
- source-code SHA catalogues;
- migration pin catalogues;
- stage-receipt authority;
- code-hash-based evidence identity.

Phase 2 does not need the historical native bundle; it needs:

- captured evidence;
- trusted phase-1 evidence digest;
- current approved model source;
- ordinary CUDA providers.

Normal phase 1 still needs the qualified native/reference bundle.

The wrapper is:

    scripts/qualify-qwen35-ordinary-cached47-50.sh

Modes:

- normal: structural checks, ephemeral legacy staging if necessary, reference capture, digest handoff, ordinary compare;
- preflight-only: structural/environment checks only, no large copy;
- compare-existing: no historical bundle required.

## 10. Exact ordinary CUDA context admission

Under the unchanged 4608 MiB provider / 5120 MiB physical limits, totals include
cold model, two roots and retained conservative admission headroom. The owned
GEMM allocates no scratch; these are not claims of physical scratch usage.
The 30-Oct correction includes the previously omitted 280-byte rotary table.

### .8B

- 512: 1,652,916,585 B
- 1K: 1,667,662,185 B
- 2K: 1,697,153,385 B
- 4K: 1,756,135,785 B
- 8K: 1,874,100,585 B
- 16K: 2,110,030,185 B
- 32K: 2,581,889,385 B

All fit admission.

### 2B

- 512: 3,911,846,249 B
- 1K: 3,926,591,849 B
- 2K: 3,956,083,049 B
- 4K: 4,015,065,449 B
- 8K: 4,133,030,249 B
- 16K: 4,368,959,849 B
- 32K: 4,840,819,049 B

512 through 16K fit.

32K exceeds 4,831,838,208 B by 8,980,841 B and must refuse before physical allocation.

These are admission/fit results only. Every admitted context still needs real-model correctness, continuation, fault and ownership qualification.

## 11. Deferred Qwen mission

Fabric transition is now first. Do not rerun already accepted cached47→50,
bounded64 HIP, finite27B, lifetime or GDN scopes merely to restart the campaign.
After migration, resume genuine multi-row0.8B/2B HIP and call-free lowering,
then general27B/populated fit contexts/C1..C64/matched vLLM. Preserve the separate
CUDA2B/wider reference and ROCm baseline authority gaps. Exact current scope:
`../plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md` and owning
model-execution/numerics/serving TODOs. Production remains disabled.

## 12. Serving and router state

The project already has:

- bounded HTTP handling;
- health/models;
- OpenAI-style text/chat completions subset;
- JSON/SSE streaming;
- request/output usage accounting;
- bounded pending queue;
- cancellation/disconnect propagation;
- CPU route;
- finite/native CUDA routes where qualified;
- source/model route validation before side effects;
- graceful worker join/shutdown;
- source-free C16/C32/C64 queue/population coverage.

The source-free production router queue is not the blocker for ordinary Qwen CUDA; the model route is.

Recent fresh review found the process-signal fixture incorrectly assumed every environment could nest a bubblewrap PID1 namespace. The shared fixture now:

- always exercises ordinary SIGTERM/SIGINT process shutdown;
- preflights PID1 namespace capability;
- skips PID1 rows only for explicit namespace/permission unavailability;
- hard-fails unexpected bubblewrap/preflight errors;
- reports skip reason explicitly;
- Rust asserts ordinary rows cannot skip and PID1 rows pass when capability exists.

Current serve all-targets result after the fix:

- 89 passed;
- 0 failed;
- 4 intentionally ignored source/hardware fixtures.

## 13. Cache/acquisition current state

Canonical cache data is authoritative; human by-name naming is not.

Human projection design:

    models/
      snapshots/<digest>/...
      refs/...
      .by-name-initialized
      .by-name-state/
      by-name/

Key properties:

- runtime/model selection never reads by-name state;
- by-name projection is disposable;
- explicit aliases and automatic names live in durable human state;
- dirty modes recover projection/reverse/reconciliation;
- publication can leave a pending human update and later self-heal;
- missing state after prior initialization refuses rather than silently losing aliases;
- reset requires explicit alias discard;
- mount-boundary checks use mount identity, not just st_dev;
- process-exit/fault matrices exercise human metadata and reclamation recovery;
- canonical snapshot/ref availability is not downgraded by human metadata failure.

Acquisition/cache supports bounded, source-aware providers and offline exact-cache behavior for the implemented providers, including Hugging Face/GitHub/Xet/OCI/S3/GCS/Azure/ModelScope/object flows where the relevant feature is enabled.

No provider metadata or alias may upgrade source rights.

## 14. Device/CLI/config state

Current CLI/device work includes:

- explicit CPU/device routing;
- exact PCI/device identity;
- validated direct /dev selectors for supported DRM/NVIDIA nodes;
- NVIDIA minor is not treated as CUDA ordinal;
- aliases converge on one observed device identity;
- source/alias/device policy precedes execution side effects;
- bounded YAML pipeline/config checking;
- strict config/env/argv precedence;
- dry-run and refusal behavior.

Pipeline configuration is more complete than pipeline execution. A valid YAML graph does not imply the task pipeline is implemented.

## 15. Memory/SSD/state direction

Implemented foundations include:

- shared physical resource accounting;
- immutable source extents;
- bounded source windows;
- source descriptor retention;
- transfer staging admission;
- retained GPU readers through actual completion;
- state/root/cancellation/quarantine semantics.

Open work remains for:

- production paged complete-domain state;
- complete-domain prefix reuse/COW;
- product snapshot persistence/restore;
- real SSD model streaming;
- expert/source IO QoS;
- eviction under actual GPU completion;
- broader multi-model residency.

Do not call KV-only reuse a hybrid-state prefix hit when recurrent/GDN/SSM/other domains are missing.

## 16. Model-family status discipline

Qwen .8B/2B are the primary current ordinary CUDA correctness path.

Qwen27, Gemma4 and other families have substantial source/reference/finite backend work, but do not inherit .8B support automatically.

Keep these distinctions:

- source parser support;
- family mathematical support;
- CPU reference support;
- a finite backend qualification;
- a production route;
- long-context support;
- serving support;
- performance qualification.

They are separate claims.

Future/target families and tasks discussed in the design include broader Qwen, Gemma4, Llama/Mistral, GLM, DeepSeek, image, ASR, TTS, Whisper/Canary/Parakeet-style audio systems and multimodal pipelines. Treat these as roadmap targets unless current source/tests explicitly implement and qualify them.

## 17. Fabric vNext — current canonical design

Fabric vNext is active: S0 is signed in`7396984`; S1–S3/F0–F4/Q0–Q2 remain open.

User intent:

- make the final architecture visible in code immediately;
- break pre-v1 internal/public APIs when that produces a cleaner design;
- move current working code into the new ownership model instead of carrying compatibility layers forever;
- allow private compile-safe placeholders/Unsupported where implementation is not yet migrated;
- no reachable todo panic on an advertised supported path;
- preserve all current exact source/numeric/state/resource safety;
- qualify the assembled transition once, then fix regressions in small focused commits.

Target dependency/ownership idea:

- zerogravity-core: semantic/numeric/state/provider/physical contracts;
- zerogravity-artifacts: containers/codecs/source/acquisition/store;
- model crates: family source/math/state/tokenization/specialization;
- CPU/GPU: physical providers;
- new zerogravity-engine: model/backend-neutral runtime, scheduling, sessions, residency;
- new zerogravity-serving: reusable server/protocol layer depending on engine/core, not facade/CLI;
- zerogravity facade: composition root/public SDK;
- CLI: arguments/presentation/signals only.

The facade is the composition root. Engine must not depend on Qwen/Gemma or physical CUDA/CPU implementations. Serving must not depend on facade/CLI. CLI should call the public SDK.

### Planned transition sequence

S0 — full structural skeleton first

- create engine and serving crates;
- expose top-level target ownership/contracts;
- compile;
- explicit Unsupported for pending operations;
- architecture checker updated;
- no heavyweight model qualification yet.

S1 — exact source/quant/encoding migration.

S2 — family math + GPU provider/qualification separation.

S3 — canonical state/root/resource publication and ownership.

F0 — model-neutral engine.

F1 — complete public Rust SDK.

F2 — reusable serving library.

F3 — thin CLI and deletion of shadow runtime copies.

F4 — feature parity matrix across all currently real behavior/refusals.

Q0 — one assembled broad qualification.

Q1 — focused bugfix queue at lowest shared layers.

Q2 — final full gate, physical/source qualification and capability matrix.

Only after transition:

- broader models;
- vLLM-style continuous batching/paged state;
- larger model residency;
- SSD execution;
- Gemma broad native routes;
- additional quant formats;
- sparse/MoE/MLA/Engram/MTP;
- multimodal ASR/TTS/image/video;
- NPU/remote distributed placement;
- calibrated autotuning/benchmarking.

The vNext documents no longer require or promise a monolithic archive or documentation SHA index. This handoff is the cross-window pointer. Git history is the archive.

## 18. Current plan queue

`../plans/TODO.md` is authoritative. Owner promoted Fabric transition to P0:
S0 architecture-visible skeleton, S1–S3 lower-layer moves, F0–F4 complete
engine/SDK/serving/thin CLI, Q0/Q1/Q2 assembled acceptance. Preserve all current
Qwen outcomes and unfinished mission in their existing owning plans.
The original Fabric packet supplies full architecture, the transition packet
is its implementation superset. Historical “not authorized” wording is not a
current stop condition. No physical campaign or recurrence was restarted.

## 19. Fresh review findings and fixes immediately before this handoff

Fresh review found and resolved:

1. Serving shutdown test portability:
   - nested PID1 bubblewrap was assumed universally available;
   - capability is now preflighted;
   - ordinary TERM/INT always run;
   - PID1 skip is explicit and limited to permission/namespace unavailability;
   - unexpected bubblewrap errors still fail.

2. Shared check coverage:
   - qwen27b-layer9-prefix-reference source-contract tests were maintained but absent from scripts/check.sh;
   - they are now part of the shared source-only gate.

3. Repository policy contradiction:
   - repo-specific instructions said main-only topology but later recommended persistent side worktrees;
   - repo-specific policy now consistently uses main-only persistence, explicit file ownership and ephemeral external scratch.

4. Fabric vNext handoff drift:
   - live docs still promised combined single-file archives / documentation SHA catalogues;
   - those instructions are retired;
   - canonical handoff method is this file + four-file spine + maintained vNext plans + Git history.

5. Active Qwen plan drift:
   - three plans still said the approved .8B safetensor was inaccessible;
   - it is now visible and hash-verified;
   - plans now identify the true missing input: qualified native/reference bundle.

6. Active plan completion drift:
   - older top sections still listed non-GDN ordinary lowering and context-fit derivation as open;
   - later accepted implementation had already closed those;
   - the top open-work statements are reconciled.

7. Source hygiene:
   - ignored Python bytecode left by earlier tooling was removed.

No additional source/Cargo correctness regression was found in the fresh pass after those fixes.

## 20. Validation state

The repository was comprehensively exercised during the cleanup/review sequence.

Important passing scopes include:

- formatting;
- architecture dependency/DAG checks;
- architecture checker self-tests;
- HIP/Vulkan/CUDA feature-boundary compiles;
- ordinary Qwen route/refusal feature-boundary tests;
- Qwen qualification route tests;
- CPU dependency isolation;
- strict CPU clippy;
- full CPU workspace tests;
- minimal no-default feature checks;
- cache transaction tests;
- diagnostic cache process-exit/reclamation/by-name recovery tests;
- source-only model/reference contracts;
- CLI cache-only tests;
- shared CPU preparation/cache tests including real local CPU inference;
- acquisition base and diagnostic artifact/CLI tests;
- release CPU CLI build;
- installer integrity/idempotency/version fixture;
- shellcheck;
- Nix formatting;
- serving benchmark/comparator/matrix self-tests;
- numerical layer/attention/cpu-qualification/native-GPU script controls.

After the PID1 harness fix:

- full serve all-targets: 89 passed, 0 failed, 4 intentionally ignored;
- qwen27 initial-core source contract: 7 passed;
- qwen27 layer9-prefix source contract: 6 passed.

A previous monolithic scripts/check.sh invocation exceeded the connector's single-call wall while compiling, so the same sections were executed in bounded groups. Do not interpret the connector wall as a source failure.

The 29-Oct continuation recovered `.8B` native inputs and passed fresh reference
capture. The old opaque-provider startup still refuses a private driver
export-table call. The30-Oct driver-only path supersedes that blocker and passes
exact `.8B` cached47→50; broader physical scopes remain open (section8).

## 21. Git/topology cleanup history

Recent normalization commits include:

- 768b724 — serving Qwen CUDA qualification-boundary test alignment
- 932f564 — GDN recipe capture-tool migration
- 734d29f — retire obsolete Qwen27 replay examples
- 2ff3d49 — add Fabric vNext transition design
- 15f5bd3 — document retired worktree/run cleanup
- 9abd936 — retire machine-local capture dependencies
- ace934b — prune superseded handoff archives
- 7b5e50e — normalize current planning provenance
- 379d91d — capability-aware PID1 serving shutdown control
- ae8cd89 — Qwen27 layer9 source contract added to shared gate

The large archive prune removed hundreds of superseded handoff/sourcebook/run-history files. That material remains in Git history.

Final topology after cleanup:

- one persistent checkout;
- one local branch: main;
- origin/main remote;
- no stashes;
- no old pi-web-agent branches;
- no old worktree-pool references;
- no tracked .agents/runs provenance;
- root README remained byte-identical during cleanup.

## 22. How to continue safely

Latest owner direction is Fabric, not autonomous Qwen resumption. Read current
source/status, the two Fabric packets and source reconciliation. Continue the
active S/F phase with one implementation writer, exact source/state/resource
ownership and explicit Unsupported stubs. Preserve existing runtime paths until
replaced coherently; never claim the skeleton migrated or qualified them.
Resume Qwen only after this prioritized structural/feature-preserving work or a
new explicit priority change. Cancelled recurrences and stale receipts do not
change this order.

## 23. Do not do

- Do not push automatically.
- Do not tag/release automatically.
- Do not recreate old worktrees/branches.
- Do not recreate .agents/runs as tracked provenance.
- Do not rebuild the retired monolithic handoff archive.
- Do not make per-document SHA catalogues.
- Do not download/substitute model sources without authorization.
- Do not infer physical GPU support from compilation.
- Do not enable ordinary Qwen CUDA production routing before real source+native qualification.
- Do not benchmark before parity/lifecycle/context gates.
- Do not weaken source/state/resource semantics to make a context or benchmark fit.
- Do not modify the root README unless the user explicitly asks.

## 24. Current continuation

Fabric source resync → S0 → S1/S2/S3 → F0/F1/F2/F3/F4 → Q0/Q1/Q2,
then the preserved Qwen mission. Track accepted and uncommitted phase work in
`../plans/fabric-vnext-transition/TODO.md`; don't reset to historical design
snapshots or expand finite qualification catalogues into a production engine.

This document should remain concise enough to load in one window while still carrying the complete continuity story. Detailed facts belong in the maintained linked plans, not duplicated here. When a fact changes, update the owning plan/status file first and this handoff only when the cross-window continuation materially changes.

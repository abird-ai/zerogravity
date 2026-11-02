# Current status and tracking

This file is the stable status target used by the repository README. Cross-window context starts at [CURRENT_HANDOFF.md](../docs/CURRENT_HANDOFF.md).

## Authority

Current source and tests are the implementation truth. The active work queue is
[TODO.md](TODO.md), with plan conventions in [README.md](README.md). Git history
retains superseded plans, handoffs, checkouts and run evidence; they are not
live authority.

## Accepted functional transition — source `8c43844`

- S0–S3/F0–F4 and scoped Q0/Q1/Q2 complete: one Rust mathematical/source engine,
  original root/physical arbiter, shared CPU/GPU worker, public SDK, independent
  serving library and thin CLI. Future/descriptive capabilities still refuse.
- Signed repairs: `8d761c4` full12-byte feedback bounds; `8c43844` cold-only Gemma
  SDK and exact model-neutral diagnostics. No math/kernel/ABI/source pin, timeout,
  tolerance, cancellation or owner relaxation; original failed scopes retained.
- Verified separately: CUDA14 workers/12 real HTTP, ordinary0.8B parity/lifetimes,
  finite SDK/CLI/captured equality; HIP0.8B/2B16 source/lifetime cases; trained
  Gemma CPU52 comparisons, HIP8/Vulkan10 scopes and SDK lifecycle; cache/import;
  all18 finite27B/shared-HIP scopes. Each case's semantics and cleanup inspected.
- Final isolated gate:2,340 Rust/0fail/97ignored/307 suites (72 empty), Python17/
  182/10skips. Affected strict/features pass. Source reviewed; all19 S/F/repair
  commits through `8c43844` have verified good signatures. No source change after
  this gate; final reconciliation is documentation only.
- Finite27B natural remains bounded to two positions; source-page fault cases
  complete no full model and claim no numerics. Unknown retains/fences original
  owners. Healthy process/compiler baselines remain nonzero. Final remote unit
  success/exit0, collected/inactive/MainPID0/empty cgroup; no owned executable,
  GPU idle. Full scoped evidence and retained failures: [transition TODO](fabric-vnext-transition/TODO.md).
- Not qualified: missing legacy2B CUDA image, wider independent/populated contexts,
  general HIP/27B serving/residency, future codecs/tasks and matched performance.
  No new p95/TTFT/ITL or vLLM superiority claim. These remain the deferred Qwen/
  E01–E10 mission, not inferred from diagnostics or suite durations.
- Main only; recurrences cancelled. No acquisition, gfx90c launch, recovery/
  privilege change, push or release. Root README/external checkout preserved.

## Previous checkpoint — source/host parity `fef7123`

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

Continue from [transition TODO](fabric-vnext-transition/TODO.md). Compilation
and metadata inventory are not physical execution eligibility.

## Historical accepted state — Fabric S0 (31 Oct)

Owner authorizes [Fabric transition](fabric-vnext-transition/TODO.md): reconcile
both plans against signed`1c35172`, then complete compilable S0 and
feature-preserving S/F migration. Qwen remains deferred, not completed;
recurrence`17aafa71` stays cancelled. No new hardware campaign or production
route. Plan 1's full design is retained in Plan 2's S0–S3/F0–F4/Q0–Q2 ordering.
See [source delta](fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md) and
[feature preservation](fabric-vnext-transition/FEATURE_PARITY_MATRIX.md).
Signed`7396984` lands verified S0: new cold crates/contracts/SDK and initial
byte-preserving CLI moves; DAG18, strict and focused refusal/authority tests.
Final isolated gate:2,159 Rust/0fail/87ignored/292 suites;17 Python suites/182
cases (10 explicit source-input skips), release/installer/shell/Nix pass.
S1 exact codec/source moves are next. No new physical/model/production support;
F0/F2 shell operations remain Unsupported. Resume from the transition TODO.

## User-requested checkpoint and pause — 31 Oct

- Signed `e2559b6`: actual HIP chunk-GDN arithmetic matches all19 original
  cases/380 fields/64,345,520 words exactly. Four numerical status controls
  preserve inputs/prior and export status only through the original Ready.
  All20 modules/code/host/device retire; namespace closes; parent0 and empty
  recursive descendants verified. Failed r1/r2 scopes remain separate.
- Shared HIP/Vulkan strict checks pass. Final isolated gate:2,142 Rust/0fail/
  87ignored/283 suites,17 Python suites/178 cases plus packaging. No Vulkan
  physical, whole-model, serving or performance grant from operator evidence.
- Owner requested pause at this checkpoint, with possible direction change.
  Recurrence`17aafa71` cancelled; no owned native/build jobs remain running.
  Fabric is now the authorized direction; stale receipts must not resume Qwen.
- CPU/CUDA parity means functional model coverage, not identical arithmetic.
  CUDA0.8B ordinary cached47→50 is correct;2B ordinary numerical/wider populated
  context coverage remains open despite capacity/lifecycle execution. Bounded
  CPU/HIP27B/Gemma evidence does not qualify equivalent CUDA execution. Original
  HF CPU captures do not establish ZeroGravity CPU support for those calls.
- Next if resumed: genuine multi-row0.8B/2B HIP composition and call-free
  lowering, then general27B/contexts/C1..C64 and matched vLLM. Production remains
  disabled; broader CUDA references and approved ROCm baseline remain gaps.

## Paused owner mission — 30 Oct

- Finish CUDA `.8B`/`2B` on `pvl-l5`, ROCm `.8B`/`2B`/`27B` on `pvl-x2`,
  genuine active contexts through supported fit, and C1/C4/C8/C16/C32/C64 serving.
- Then measured optimization against pinned matched vLLM in every feasible run;
  high concurrency on capable hardware, especially x2. No hiding losses or
  inferring acceptance from padded capacity, a finite diagnostic or old timings.
- x2 resource use is authorized without routine prompts. Continue ready work;
  ask only for consequential external dependencies/decisions beyond the repo.
- The completed27B finite checkpoint is a prerequisite, not the whole mission. Ordinary small-model
  HIP, general27B contexts, model-specific concurrency and performance remain open.
  See [campaign](serving-and-speculation/BENCHMARK_CAMPAIGN_2026-10-27.md).

## Shared source execution foundation — 31 Oct

- Signed `d4d5df4` fixes eager CPU PV: original four-part BMM reduction, not
  projection/QK order. Original seven-token failure remains recorded; no relaxed
  tolerance.32 independent cases match65,536 words exactly.
- Signed `f6fa4b9` separates source traversal/recording from CPU diagnostics and
  shared HIP preparation from probes. Model geometry determines state extents;
  recording owns original metadata/source, not backend/publication authority.
- Independent original HF attention:40 cases through64 active tokens, nine
  fields each. CPU and actual gfx1151 each match1,774,240 words exactly. All150
  native modules retire, namespace closes, durable parent exits0/empty cgroup.
  This is operator evidence, not populated full-model or serving qualification.
- Final isolated gate:2,135 Rust passes/0fail/85ignored/283 suites;17 Python
  suites/178 reported cases, release/installer/shell/Nix pass. Selected CPU/HIP
  strict checks pass; existing non-strict warnings are unchanged.
- Approved `.8B`/`2B` inputs:11/12 members with genuine indexes. Independent
  original HF full-model CPU references now cover64 populated positions × two
  exact replays/model: fullV/all layers/all populated state,15,776 records/model,
  authenticated source and actual parent/group/container/EOF closure.
- Signed `1f29a42` HIP source assembly reuses typed SSA, shared resident storage and
  original session publication. Both cold64 plans and11 refusal/accounting
  controls pass; a root-pool130 quote is not C64 inference. First native attempt
  prepares785 modules then refuses premature matrix binding before Ready.
  Corrected pending binding reaches actual0.8B comparison, failing303 words;
  first initial recurrent word differs only in zero sign. All785 modules retire,
  namespace/source close, parent exits1/empty cgroup. Original initial recurrence
  addition restored in CPU/HIP; four host tests/strict check pass. Fresh r3
  passes both actual models through64 populated positions × two natural replays:
  fullV/all24 layers/all state;1,396,408,320/1,399,816,192 words exact. Known
  cancellation/discard, root isolation, pressure, restore/reset/shutdown pass;
  native785/781 modules retire, namespaces/source close, parent exit0/empty
  descendants. Owned unit stopped.2B canonically imported without download.
  Final assembled gate passes2,136 Rust/0fail/85ignored/283 suites,17 Python
  suites/178 cases plus packaging checks; CPU/diagnostic/HIP/shared-Vulkan strict
  checks and both current cold-control sets pass. Canonical unknown/fault lifetime subsequently passed in `85f33c7`;
  scalable ordinary execution and production/serving/performance remain open.
  CUDA-only seals and finite27B guards remain separate; wider contexts are open.

## Owned CUDA checkpoint — `2983bb7`

- Owner authorized continuation and completion. Signed owned CUDA execution
  checkpoint `2983bb7` is on the sole main worktree; nothing pushed.
- Owned driver-only GEMM replaces ordinary opaque BLAS under the original
  queue/compiler/arbiter. Pinned reference/source/limits stay unchanged.
  180 synthetic and eight authenticated projection cases pass exactly, with
  intact guards and retired leases/events.
- Corrected rotary admission (+280 bytes), GDN partition binding, FP32 residual
  normalization, fused Q/K reduction and CUDA rotary FMA rounding. CPU contracts
  remain unchanged. 211 GPU host tests and14-position rotary comparison pass;
  strict CUDA GPU/facade all-target Clippy and formatting checks pass.
- Fresh ordinary `.8B` cached47→48→49→50 now passes exactly: all248,320 logits,
  all42 state domains and natural feedback at every step, against the unchanged
  independently authenticated reference. Real-model normal shutdown also passes.
- Allocation-free tile-64 CUDA attention replaces the CPU-profile lowering;
  independent prefill/cached attention operands match exactly. Nine analytic
  causal/multitile cases through a32K active prefix and three nonfinite-refusal
  cases pass with intact guards and retirement. Admission totals are unchanged.
- 16 independent staged CUDA RMS cases pass. Real post-launch cancel/abandon
  preserves original quarantine/prior state. Normal lifecycle passes all13
  admitted capacities at48 active tokens;2B32K refuses before physical creation.
  Full active-context/2B numerical coverage remains open; production is disabled.
  Final source gate passes2,129/0failed/83ignored; CUDA/native host675/0failed,
  plus strict CUDA/HIP/Vulkan feature checks. Earlier failed scopes remain failed.
- Remaining input edge: exact2B reference image and trusted active-context
  comparison inputs; no model/SDK download, substitution or reference relaxation.

## Current ROCm finite checkpoint — `825ade8`

- Latest owner order: finish ROCm on `pvl-x2` now, without waiting for separate
  CUDA reference gaps. Needed2B/9B/27B model downloads are authorized; existing
  exact31-member27B source was verified/imported to canonical cache instead.
- Current-source HIP attempts run on gfx1151/96GiB. Fresh original natural
  full64/fullV/all112-state reference validates both11→220→16 cohorts and
  source/FD/process closure. Protocol14 and comparator/admission26 controls pass,
  as do narrow/shared HIP strict checks. Signed `99f0232` fixes rank3 Conv4's
  rank1 preflight routing with shared authenticated binding/30 source-window
  controls. SDK-matched
  child-only loader resolves old build-libc versus GLIBC2.44 runtime refusal.
  No SDK download, device permission change or production routing.
- Actual teacher full4 and natural healthy9-call diagnostics pass exact full64/
  fullV/all112 state; each full4 comparison has240,896,000 float words/0 mismatches.
  Healthy natural passes maintained semantic validation, including complete
  late-cancel scope rather than an invented publication packet.
- Teacher unknown and all4 independent Pending/Snapshot final-owner cases pass,
  individually inspected. All six repaired natural reports pass maintained semantic
  validation. Known final-reader paths retire121 modules; two release faults retain
  all121 modules, original service/source and1,477 quarantined handles. Cached
  unknown preserves the exact completed seed, refuses publication and retains the
  original fenced domain; its unfinished cached suffix is not numerical evidence.
- Five original natural failures remain failed. The earlier repaired batch lost
  its outer healthy exit record; that controller scope remains incomplete. A fresh
  healthy-only replay passes independently under durable supervision: exit0,
  no remaining processes, exact full4/ordinary committed state/replay/late-cancel
  and known retirement. Signed diagnostic checkpoint `825ade8` follows `99f0232`.
- Final isolated source gate passes2,132 Rust executions/0 failures/83 ignored,
  17 Python suites(10 source-input-dependent skips), release/installer/shell/Nix.
  No ordinary context/serving/performance grant; see [numerics](numerics-and-context/TODO.md).

## Historical pre-owned-execution baseline — 29 Oct 2026

- Historical reviewed implementation/source baseline: `ae8cd89`; the 29-Oct continuation adds reference-fixture/diagnostic repairs, not ordinary execution support.
- Ordinary Qwen3.5 `.8B`/`2B` CUDA is integrated under the existing resident owner but remains qualification-only; production routing still refuses before source/device/native load.
- Approved `.8B` source revision `2fc06364715b967f1860aea9cf38778875588b17` is visible at `/home/pvl/.cache/abird/zerogravity/external-baselines/models/Qwen--Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17`. The 1,746,942,600-byte safetensor hashes to `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
- The `.8B` N1 native/reference bundle is recovered offline under XDG state: retained canonical native images plus the two exact SM80 FlashAttention modules extracted from the retained pinned reference image, with the original cuBLAS providers. All 28 resources match current Rust identities; all 11 approved model members verify. Maintained bundle validation and isolated wrapper structural preflight pass. No downloads, repinning or deleted run-tree restoration.
- Fresh phase-1 cached47→50 capture passes after selecting the existing 8 GiB reference-fixture host profile; all 1,008 source windows retire. Ordinary host and all device limits remain unchanged. The original 4 GiB refusal stays failed.
- At this historical checkpoint, opaque cuBLAS initialization failed before math: status1, refused private export UUID `6bd5fb6c5bf4e74a8987d93912fd9df9`, zero allocation attempts/charges; legacy graph lookup also refused. That failure remains failed. `2983bb7` replaces ordinary execution's dependency, not the refusal or reference.
- That checkpoint's source-free lifecycle/admission/serving checks did not qualify real inference; current physical scopes are listed above.
- Fresh review fixed environment-sensitive PID1 serving shutdown control and added the maintained Qwen27 layer9 source-contract suite to `scripts/check.sh`.

## Repository topology

- Persistent checkout: main only.
- Local branch: main only.
- Accepted implementation is committed on main.
- Transient agent/run artifacts are not tracked as provenance.

## Current product boundary

The root README's implementation-status table is the concise public capability
summary. Detailed open/closed work remains in the subsystem TODOs linked from
[TODO.md](TODO.md). A capability is accepted only by current source plus its
tests/physical qualification where required; model names, metadata, old run
receipts or historical branches do not grant support.

## Current architecture direction

The maintained architecture transition is under
[fabric-vnext](fabric-vnext/README.md) and
[fabric-vnext-transition](fabric-vnext-transition/README.md). Current code wins
over older architecture prose when they disagree.

## Updating status

Land implementation and tests in logical signed commits, update the owning
subsystem TODO when the status materially changes, and keep this file concise.
Do not add generated handoffs, branch inventories, checkout paths or transient
run directories as status authority.

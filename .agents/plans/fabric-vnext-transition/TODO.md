# Fabric vNext transition — active implementation queue

Author: Project owner
Co-authors: GPT-6
Owner: @pvl — main-session coordinator

Base: `1c35172`

## Current direction — shared scheduling, then remaining Fabric ownership

- Shared CPU/GPU scheduler source/host checkpoint `ae7ec11` follows foundation
  and lifecycle `d78eb55`. Original source math, state, cancellation and ownership
  retained; physical GPU regression remains open. This is not full F0–F4/Q acceptance.
- **Main does all further work; no new subagents.** Previously running tasks
  finished without cancellation. Main reviewed/integrated their results.
- Main owns all files, compiler capacity, plan reconciliation and commits. No
  push, source acquisition, GPU launch, recovery or privilege changes authorized.
- Fabric/Qwen recurrences remain cancelled. Qwen expansion remains deferred;
  accepted model/operator/lifetime evidence is historical and tuple-specific.
- Signed`7fc5d17` lands source/state foundations on S0`7396984`/`c853ebb`.
  Signed`8591b52` lands the tested runtime/SDK/server batch. Root README and
  external checkout untouched; nothing pushed.

## Historical foundation evidence — assembled gate passed

Current logs: XDG state`abird/zerogravity/fabric-checkpoint-lEnE3DDm/`.
Earlier failures/source comparisons:`fabric-runtime-juOhHkRS/`.

- CPU/runtime/server/SDK/CLI strict checks pass; initial focused suites216 passes,
  1 ignored across34 suite runs (test executions, not distinct tests).
- Updated facade30/serving23 pass; four approved-tokenizer/full-vocabulary CPU
  fixture cases pass, including actual public SDK loading, shared source aliases,
  text streams, JSON/SSE, cancellation, server stop preserving SDK work and runtime
  shutdown. Fixtures are real executable GGUF sources, not trained-model proof.
- Held-descriptor pricing/materialization rejects source growth; trailing input
  no longer masks HTTP disconnect. Both regression scopes pass.
- CUDA native diagnostic-only, CUDA diagnostic+CLI, explicit captured serving,
  ordinary held serving, HIP/Vulkan and minimal library feature strict checks pass.
  These are compilation scopes, not hardware qualification.
- Source-free GPU grouping/profile/join/refusal controls pass; the initially
  zero-matching fixed-prompt filter was corrected to `active8_tests` and passed.
- Final isolated `scripts/check.sh` passes:2,267 Rust/0fail/94ignored across297
  suite runs;17 Python suites/182 cases, including10 explicit input skips.
  Architecture/feature/strict/installer/shell/Nix gates pass. Every terminal scope
  inspected; zero-test filters and physical/unrun scopes are not acceptance.
- Prior independent review authenticated169 original snapshots and preserved34
  CubeCL bodies, decoder and moved CPU math bodies. No tracked CUDA/image assets
  changed. Rust selector/import/cfg changes DO change broader GDN family identity;
  do not retag old physical evidence or silently repin it.

## Current batch and remaining boundaries

- Real stored encoding/codecs and CPU quantized/BF16 owners migrated; old
  QuantKind/quant producer removed. Q4_K decode works; exact CPU matvec refuses.
- Reference TaskCoordinator uses original SessionAuthority/RootAttempt, including
  caller epoch and complete-domain atomic publication. The numeric device-event
  bridge remains source-less/reference-only, never a source-bound Ready witness.
- CPU/GPU scheduler, queue, events and presentation are engine-owned; facade
  composes ordinary and explicitly diagnostic source/tokenizer/admission. Loading,
  joins and source destruction are unlocked;
  concurrent shutdown waits for cleanup and preserves failures. Explicit runtime
  shutdown from engine workers or its own loader/retirement callback now refuses
  before changing admission, rather than waiting on itself.
- Real reusable HTTP server and SDK pipeline checking/device APIs; no fabricated
  successful model preparation, placement, pipeline execution or future protocol.
- CLI GPU scheduler/route dispatch copies removed. Current uncommitted SDK batch
  also owns direct CPU/source planning, finite CUDA/Gemma diagnostics and artifact
  orchestration. Final batch gate/checkpoint and active contracts remain open;
  F0–F4/Q completion is not claimed.
- Qualification callers use real public admitted channels/host/cancel owners;
  no production test-support feature or exported isolated ledger was introduced.

## Next refinements — one coherent unit at a time

1. Continue from shared-scheduler checkpoint `ae7ec11`; reuse accepted evidence
   for unchanged paths. Do not redo the foundation or scheduler extraction.
2. Physical GPU regression remains required before qualifying moved execution.
   No launches are authorized by this source checkpoint. Preserve exact source,
   Ready/root, cancellation/fault, final-reader and serving scopes; no historical
   physical evidence is retagged. Fixed-prompt credit stays64; original captured
   Qwen credit is72×max(native_rows,1), with original prompt/B4 guards unchanged.
3. Complete remaining SDK/CLI ownership, active numerical/layout contracts and
   production/qualification separation, then F4/Q0/Q1/Q2 acceptance.
4. Explicit runtime self-join guards remain. Caller deadlines/nonblocking close
   are future work; direct Router/implicit last-owner drop needs external
   retirement, and short router callbacks remain non-reentrant.
5. Preserve the original model/context/C1..C64/matched-vLLM mission and broader
   model/media/SSD/distributed roadmap. Measure before tuning; no speedup claim.

## Committed

Historical documentation packet:`2ff3d49`; canonical handoff refresh:`ff89b87`.
Qwen commits are preserved inputs, not S/F/Q completion proof.

### `ae7ec11` — shared text scheduler, source/host checkpoint

- Main-only batch based on signed `d78eb55`; no GPU launch, source acquisition,
  recurrence restart, push or protected README/external-checkout modification.
- One Router, admission/event path and worker loop. Typed executor associated
  state/Ready/lease objects avoid Any/downcasts, reconstructed feedback and an
  adapter-owned scheduler. Shared policy owns fairness, grouping, output parking,
  band reduction and retry ordering; adapters retain original physical authority.
- Separate common model metadata from host-logit execution. A single admitted
  alias map is shared by routing and the typed worker; no duplicate registry.
- Shared active rows retain opaque physical session state and a final-reader
  lease after presentation/transport fields. GPU state remains charged by its
  original owner, never CPU state charging. Price new control/scratch owners.
- Preserve CPU one32-token-or-decode quantum/round and bounded output refusal;
  GPU full ready rounds, source-specific prompt quantum, 30s output parking and
  complete cohort publication, including non-fused GGUF final prompt tails.
- Preserve prelaunch capacity/cancel versus known-completed discard versus fatal/
  unknown. GGUF discard regroups immediately; captured-HF discard/cancel returns
  to the next fair round. Never drop submitted futures or reconstruct device
  feedback from presentation IDs. Exact route validators remain authoritative.
- Facade owns concrete loading, source/tokenizer owners, typed execution adapters
  and source health; CLI serve keeps configuration/presentation/signals only.
  Ordinary GGUF and captured CUDA use distinct executor types: composition must
  not depend on captured recipes. Explicit captured source profiles enter
  diagnostic composition, sharing the engine rather than a diagnostic scheduler.
  Ordinary CUDA refusal remains early.
- [x] Actual CPU/ordinary GPU/captured-diagnostic consumers use one typed Router,
  registry, worker and transport adapter. Remove CLI gpu_engine/gpu_batch copies.
  Physical qualification calls actual shared step/wave primitives, not a copied loop.
- [x] PreparedModel metadata separated from LogitModel; public load_gpu_text and
  cold backend/runtime-limit/closed refusals. TextAdapter reports compiled wiring,
  not source eligibility or qualification. No compatibility scheduler retained.
- [x] Admit new registry/stats, worker/retry/scratch and adapter controls. Bound
  retry progress; fatal/unknown never becomes capacity fallback. Reader-control
  credit survives the model handle; tokenizer+inventory share one actual owner.
- [x] Strict ordinary/captured CUDA and HIP consumers; Vulkan compilation;85
  engine tests; source-free GPU feature controls25 pass/2 physical cases ignored;
  GPU loader/reader4, tokenizer6 and public GPU SDK3 controls pass. Vulkan
  diagnostic-only builds retain the two existing source-page dead-code warnings.
- [x] Genuine CPU acceptance: all four approved-tokenizer GGUF fixture cases pass,
  including SDK/tasks, aliases, JSON/SSE, cancellation, independent server stop,
  host pressure and source retirement. Facade31/server23 plus integration/CLI
  scopes pass; empty feature-disabled targets are not execution evidence.
- [x] Source review: no CPU/GPU/model-crate math, kernel/image/manifest changed;
  original numerical comparison bodies and physical teacher cases retained.
  Issuer accessors return original source identities. Rust migration is not a
  physical/ABI requalification or trained-model/performance claim.
- [x] Final isolated scripts/check.sh:2,306 Rust passes/0fail/94ignored across300
  suite runs;17 Python suites/182 cases/10 skips. Architecture/features/CPU strict,
  minimal builds, release/installer/shell/Nix pass.68 empty feature-target suites
  are not execution evidence. `assembled-gate-r1.log` records the complete gate.
- [x] Signed source/host checkpoint `ae7ec11`; physical acceptance stays open.
- Logs: XDG state `abird/zerogravity/fabric-scheduler-YeCrPeT8/`;
  `shared-strict-r4.log`, `backend-feature-r1.log`, `cpu-http-r1.log`,
  `source-review-r1.txt`. Initial compile/lint failures remain preserved.

### `d78eb55` — runtime self-join refusal

- [x] Busy lifecycle phases retain their owning ThreadId; explicit shutdown by
  that loader/retirement callback returns State without closing admission.
- [x] Actual engine worker scope refuses blocking runtime shutdown, including
  during an external join. Scope covers capture destruction/unwind; no new heap
  owner or per-token/step check. External shutdown still joins outside the mutex,
  waits for completed cleanup and preserves terminal failures.
- [x] Four bounded regressions: loader open/raced close, real worker Ready/Joining,
  source destruction after load rejection/normal retirement, nested/unwound worker
  context. Refusing models exercise lifecycle only, never synthetic inference.
- [x] Strict engine/serving/facade checks;64 engine tests and exact-filter real CPU
  SDK/JSON/SSE/cancel/source-retirement test pass. Child success and completion
  marker checked by its parent. Initial two test-module compile failures corrected.
- [x] Final isolated `scripts/check.sh`:2,271 Rust passes/0fail/94ignored across297
  suite runs;17 Python suites/182 cases incl10 skips. Architecture/CPU strict/
  feature-boundary/installer/shell/Nix pass. The two Vulkan diagnostic dead-code
  warnings match the foundation gate. Logs:
  XDG state`abird/zerogravity/fabric-lifecycle-dfes2sdx/`.
- No model-source, arithmetic or kernel/ABI change, new hardware qualification
  or speedup claim. External-owner shutdown remains required; this is not arbitrary callback
  reentrancy, implicit self-drop support, unified GPU scheduling or F4/Q completion.

### `8591b52` — working CPU runtime, public SDK and embedded serving

- [x] One neutral CPU Router owns original queue, workers, cancellation, bounded
  events, text presentation and token observations; concrete source/tokenizer
  adapters and loading belong to facade. Aliases share immutable source owners.
- [x] Actual SDK CPU load/request/task-stream and embedded HTTP JSON/SSE paths;
  server stop preserves caller runtime/tasks, runtime shutdown closes clones and
  joins actual workers. Source/load/destruction joins are outside runtime mutex;
  concurrent shutdown observes terminal cleanup/sticky failures.
- [x] New tokenizer wrapper/control storage admitted before allocation and held
  through final source reader. One prepared model map; optional count ceiling
  never removes physical host admission.
- [x] Close inherited source admission pathname race with the same priced/opened
  descriptor and extent checks. Reject late HTTP bytes that mask client closure.
- [x] SDK pipeline schemas/validation and device inventory/policy extracted; PCI
  aliases deduplicated. Pipeline execution/placement/future APIs still refuse.
- [x] CLI delegates CPU scheduler/transport/pipelines/devices; obsolete import-shim
  files removed. Real-worker TERM/INT/PID1 controls preserved. GPU CLI remains.
- [x] GPU qualification fixtures use admitted public channels/cancel/host primitives,
  not leaked cfg(test) helpers. Explicit captured-serving gates match callers;
  ordinary CUDA still refuses before discovery/source/device work.
- [x] Strict CPU/minimal/CUDA/HIP/Vulkan feature checks, actual source-bound SDK/
  server/population cases, GPU source-free controls and final assembled gate pass.
  No new physical or trained-model qualification; F0/F1/F3/F4 and Q acceptance open.

### `7fc5d17` — source encoding, codecs, CPU math and canonical reference roots

- [x] Actual core StoredTensorEncoding/ScalarDType/QuantScheme, complete caller closure;
  remove old QuantKind/quant producer, retain exact external labels/format mappings.
- [x] Actual artifact scalar/q4/q6/q8 codecs and malformed/refusal tests; one decoder
  owner. CPU Q8 activation/exact quantized dots/BF16 projection moved unchanged.
- [x] Q4_K decoding retained; source-exact CPU matvec still Unsupported. No future
  codec or metadata-to-execution grant.
- [x] Reference TaskCoordinator and Qwen reference preparation use original
  SessionAuthority/RootAttempt; complete-domain/epoch/source/frontier checks and
  overflow-before-publication; original ordinary/GPU publishers and arbiter retained.
- [x] Narrow CUDA diagnostic cfg/import closure and unused host-observer cfg aligned
  to real callers. No kernel/ABI/image arithmetic change or new hardware acceptance.
- [x] Focused codec/CPU/root tests, CUDA/HIP/Vulkan strict checks and assembled gate
  above pass. Foundational APIs compile with original higher-layer callers removed
  or migrated in this same source group; runtime/transport landing follows separately.

### `7396984` — compilable Fabric base skeleton

- [x] Owner activates Fabric implementation; clean signed main at`1c35172` inspected, current source/plans reconciled. Main session owns integration; unrelated detached checkout preserved.
- [x] Create zerogravity-engine and zerogravity-serving as compiled independent Rust library crates, correct Cargo features and resource ownership (engine model-neutral, serving no facade/CLI).
- [x] Make core semantic/effects/encoding/numerics/model/task/typed media/provider/placement/root/physical contract modules visible in Rust with actual types and explicit capability/refusal.
- [x] Keep source exact existing ForwardPlan and original process-shared low-level ResourceArbiter, SessionAuthority, GPU event/lease as one real implementation each; avoid parallel placeholder owners.
- [x] Make artifact container/codec/source/acquisition/derived homes, CPU numerics/quant, GPU backend/quant/program/lease/transfer/qualification, family source/forward/state/specialization homes compile-visible.
- [x] Make public zerogravity API visible for all user functionality (RuntimeBuilder, models, run, acquisition/cache, devices, pipeline, serving, diagnostics) with typed Unsupported for unavailable future tasks.
- [x] Make CLI target args/commands/output/signals structure visible while preserving existing currently supported execution through source migration.
- [x] New pending operations use typed Unsupported; no new todo! or public panic path, fake model, zero-logit kernel, fabricated Ready or permit.
- [x] Update existing scripts/check-architecture.py/check fixtures for new DAG, no serving→facade cycle or engine→Qwen/GPU; ensure CPU-only default compile avoids GPU compiler/network stack.
- [x] S0 targeted strict/contracts and final isolated repository gate pass; no physical model/GPU qualification.
- [x] Signed structural source/tests and owning TODO checkpoint; nothing pushed.


## Uncommitted — SDK/CLI ownership source/host checkpoint

- Owner requests completing all remaining Fabric work. Main owns source, builds,
  review and plan reconciliation; prior launch/acquisition/push holds remain.
- [x] Rehome real direct-run/preparation/audit and cached-source business paths;
  keep one shared scheduling/physical authority and exact existing arithmetic.
- [x] Finish typed artifact operations and CLI delegation. Preserve provider,
  Xet/conditional/offline/ref/durability and cancellation semantics; environment,
  credentials, signals and human formatting stay caller-owned.
- [ ] Reconcile remaining active contracts/catalog/transport against actual
  consumers, not inert descriptors; keep future E-phase capabilities refused.
- [x] Focused SDK/current-feature regressions and final assembled source/host
  gate pass. Physical acceptance remains separate/unrun.
- [ ] Sign this source/host checkpoint and reconcile current recovery documents.
- Implemented: `artifacts::{HubPull,PreparedHubPull,ComponentDocument,
  ComponentClients,SnapshotFile}`, original fixed transfer/source pools and
  `PreparedQwen::from_cache`; no transport, publisher or source issuer replacement.
- Run ownership: `run::cpu::Plan`, `prepare::selection::PreparedCpu`,
  `diagnostics::{cpu_text,qwen_route,gemma_text}`; CLI keeps arguments, environment,
  signals, bounded stdin, lazy audit sinks and presentation. Unreachable alternate
  run/parser/GPU implementation removed; existing command behavior retained.
- CPU returned payloads retain the original envelope until their final owner
  drops; private fields prevent independently moving charged payloads out.
  Context refusal and lazy audit timing keep original source/control lifetimes.
- Finite native SDK owns allowlist validation; genuine fresh PCI/Client binding,
  source roles, native feedback, pre-Client admission and retirement stay original.
  Gemma returns operation plus source/accounting failures after source close, even
  for post-open setup or presentation errors. No trained/physical pass inferred.
- Accepted focused scopes so far: artifact SDK4/cache2/Xet1 controls; CPU input
  controls2; CLI all-target/provider/alias tests, including genuine sparse-source
  run forms; direct CPU source/audit/planning/final-output case passes16.84s.
  Public-only `examples/fabric_sdk.rs` builds and runs cold help.
- Combined CUDA/HIP, native-only, Gemma CPU-only, default/acquisition strict and
  minimal checks pass. Reviewed native controls12 pass/2 physical ignored;
  Gemma host control1 passes/1 trained-source CPU case ignored. SDK-only example
  builds; cold help and actual missing-source refusal pass.
- Final isolated `scripts/check.sh` passes:2,315 Rust executions/0fail/97ignored
  across302 suites (70 empty feature-target suites are not execution evidence);
  17 Python suites/182 cases/10 skips. Architecture/strict/feature/minimal,
  release/installer/shell/SSE/Nix gates pass; only the two existing Vulkan
  source-page dead-code warnings remain. See `final-gate-r1.log`.
- Source review confirms16 unchanged native policy/budget helper bodies after
  namespace/whitespace normalization. No core/CPU/GPU/model math, kernel/image,
  source manifest or root README bytes changed. No physical evidence retagged.
- `assembled-sdk-r1.log` was deliberately interrupted after completed compile/
  artifact/control stages, before model execution: trained Gemma remains deferred.
  `assembled-sdk-r2.log` finishes CLI plus the explicitly approved tiny CPU case.
  The new existing-source Gemma CPU lifecycle case compiles but stays ignored;
  source presence never grants execution permission. Initial failed logs retained.
- Logs: XDG state `abird/zerogravity/fabric-finish-b9SliwEt/`.

## Uncommitted — remaining S1 active contracts

- [ ] Bind activation/numerical/layout/derived contracts to genuine preparation consumers; real stored encoding/callers are migrated above.
- [ ] Bind existing GPU packed kernels/activation under source+variant+numeric+device constraints; no codec-specific model engine.
- [ ] Validate remaining active-contract integration; reuse accepted unchanged codec/numerical evidence.
- Concrete follow-up: active `SourceNumericalContract`/source operators remain
  authoritative. Newer `ActivationEncoding`, `NumericalContract`,
  `PreparedTensorLayout` and `DerivedModelManifest` are descriptions, not bound
  kernel/source/Ready authority. Generic engine preparation/catalog loading,
  per-model retirement, residency and paged state still explicitly refuse.
  Bind genuine consumers; do not rename metadata as execution or invent E-phase
  quantization/hot-reload/placement merely to close these boxes.

## Uncommitted — future S2 family math and GPU provider

- [ ] Keep one Qwen forward traversal and independent Gemma source math, one existing ForwardPlan recorded as canonical SemanticProgram.
- [ ] Preserve GPU production model dependency removal (263ff79) and concrete source-independent provider/ABI preparation.
- [ ] Separate CUDA GDN kernel implementation and exact code+driver/ISA/numerical identities from historical finite Qwen prompt fixtures/qualification source.
- [ ] Preserve newest captured fresh-prefill/reference layout identities, short contexts, original source permission; no automatic ordinary CUDA enable.
- [ ] Move only remaining genuine source/model wrappers; no giant unnecessary compiled .cu rename/rewrite.
- [ ] Independent CUDA/HIP/Vulkan feature source-only check, family and source refusal fixtures; signed commit(s), hardware later.

## Uncommitted — future S3 canonical root/resource

- [ ] Ensure CPU/GPU/Gemma commit only after source/epoch/root/full domain/real Ready; no per-domain public commit nor device stream direct PublishRoot.
- [ ] Keep existing process-shared PhysicalResourceArbiter low-level in core; engine will use its admitted physical ledger, no new wallet.
- [ ] Preserve buffer leases encoded/captured/queued/submitted/unknown and safe quarantine/retire, host/UMA alias accounting.
- [ ] S3 source-free state resource targeted checks and signed commit(s).

## Uncommitted — moved GPU physical acceptance

- [ ] Qualify moved execution with authorized real-source numerical/state,
  cancellation/fault/final-reader and serving checks. Compilation/host controls
  and source preservation do not qualify GPU execution; no old evidence retagged.

## Uncommitted — remaining F0 model-neutral engine

Shared source/host scheduling landed in `ae7ec11`; remaining requirements:
- [ ] One cold model/provider registration point in facade, same source alias source sharing and exact loaded ModelGeneration pin.
- [ ] Preserve CPU bounded prefill/decode and GPU native B1/B2/B4/B8 true batch semantics including ragged/late arrivals and source-specific eligibility.
- [ ] One ResourceArbiter and original worker Ready/lease/shutdown; source-valid typed committed output before HTTP.
- [ ] F0 tiny model-neutral mock/CPU tests, affected compile and signed commit(s).

## Uncommitted — remaining F1 public zerogravity Rust library

- [ ] Expose all real model/run/cancel/stream/inspect/model admin/alias functions through exported zerogravity API.
- [ ] Expose all artifact pull/download/providers/cache/source selections, devices/fit/dry-run/config and pipeline checking APIs without CLI.
- [ ] Preserve provider-specific HF/GitHub/Xet/OCI/S3/GCS/Azure/ModelScope/HTTP implementation, source path/symlink/shard/credential/etag/refusal handling.
- [ ] External Rust program depending only on zerogravity compiles/runs qualified local functionality and model-free tests; no CLI needed.
- [ ] F1 focused acquisition/cache/devices/run/pipeline feature tests and signed commit(s).

## Uncommitted — remaining F2 unified-engine transport integration

Shared Router/HTTP integration landed in `ae7ec11`; remaining requirements:
- [ ] Serve uses engine TaskRequest and CommittedEvent, not Qwen/GPU/allocator; ingress/auth/HTTP body/stream credits in serving, scheduler/state in engine.
- [ ] Preserve CPU and eligible native CUDA request routes, real cancellation/disconnect, bounded server pending queue and 50abaf7 worker join/owned shutdown.
- [ ] Validate remaining unified GPU/transport integration within applicable source/physical scopes; reuse accepted CPU/HTTP tests.

## Uncommitted — remaining F3 thin CLI / delete old runtime

- [ ] Move all CLI current run, serve, pull, models, cache, inspect, devices, pipeline logic to facade/API and delete old implementation copies.
- [ ] CLI retains argv/env/config precedence/path selection, human/JSON display, progress, process exit and Ctrl-C→cancel only.
- [ ] Enforce project-specific CLI normal imports only through zerogravity; remove legacy shadow scheduler/model/HTTP and no compatibility aliases.
- [ ] F3 commands/help/config/source/negative smoke and signed commit(s).

## Uncommitted — future F4 whole current feature completeness

- [ ] Review every row of FEATURE_PARITY_MATRIX.md, mark real supported, held/unsupported, migrated and targeted-tested.
- [ ] All current CPU/GGUF, source codecs, physical GPU qualified modules, source acquisition/cache, device selection, pipeline validation, HTTP JSON/SSE/worker lifecycle, state/leases operate on new owners.
- [ ] No reachable TODO panic on any supported API/CLI/serve path, no false Q4_K / Q1/2/3 / NVFP4 / general CUDA / Gemma/27B/media/distributed capability.
- [ ] Remove obsolete actual duplicate compatibility paths, address targeted small failures, signed source cleanup commit(s).
- [ ] Ready for Q0; no heavyweight test yet.

## Uncommitted — future Q0/Q1/Q2 consolidated qualification

- [ ] Q0 source/identity frozen, one full trusted scripts/check.sh and applicable independent model/CPU/GPU/HIP/Vulkan/codec/HTTP/SDK/CLI hardware qualification only with approved sources/devices. Report passed/failed/ignored/unexecuted/unsafe separately.
- [ ] Q1 create bounded transition bugfix queue; fix at lowest generic layer with focused regression source/test and signed commits, run only changed small tests (do NOT repeat huge suites per bug).
- [ ] Q2 one final complete gate and necessary permitted real model/device scopes, full current feature matrix, public SDK/server embedded tests, HTTP and CLI parity, no shadow/old runtime, no false benchmark claims.
- [ ] Present full signed phase commit SHA list, tests, actual capability/quant/device status, known held scopes, Pareto perf baseline where measured, and remaining roadmap. No push/tag/release without explicit authorization.

## Future expansion AFTER transition, not initial acceptance

- [ ] E01 original-HF general Qwen3.5/2B and real 27B; independent fresh47–50/long context CPU/CUDA/HIP/Vulkan.
- [ ] E02 vLLM continuous batching, persistent row slots, temporal prefill, all-domain paged/COW state, goodput; benchmark.
- [ ] E03 shared multi-model residency, replicas and hot reload.
- [ ] E04 true large dense SSD authenticated windows and physical streaming.
- [ ] E05 Gemma4 E2B/26B-A4B/31B complete native qualification, then new families.
- [ ] E06 Q4_K source kernel, Q1/Q2/Q3/NVFP4/EXL3/source derived quant and quality evidence.
- [ ] E07 sparse/QSA/GDN/MLA/Engram/MoE/MTP draft state and expert IO.
- [ ] E08 native ASR/TTS/codec/image/video/duplex and typed task pipelines.
- [ ] E09 NPU/graph/wafer/remote distributed placement and state bridge.
- [ ] E10 calibrated cost/autotune, matched vLLM/llama.cpp task benchmarks and release.

## Handoff and documentation maintenance

- [ ] Keep every transition document UTF-8, internally linked and consistent with the canonical current handoff plus the four maintained planning/status files.
- [ ] Revalidate source-dependent claims against current main; historical source SHAs are provenance only and never an implementation base.
- [ ] Do not recreate consolidated archives, per-file documentation SHA indexes or frozen handoff copies; Git history is the archive.
- [ ] Keep the key-path checkpoint and unfinished migration explicit; no new subagents or Qwen restart from stale receipts.

## Tests and verification

- [x] S0 DAG18 + isolated minimal strict + focused engine/server/core/SDK tests — `7396984`.
- [x] Final isolated repository gate2,159 Rust/0fail/87ignored/292 suites;
  17 Python suites/182 cases (10 explicit source-input skips), packaging pass — `7396984`.
- [x] Current codec/CPU/reference-root and CPU SDK/server batch: scoped tests and assembled gate above.
- [x] Post-`27ac278` lifecycle refusal: strict,64 engine tests, genuine CPU SDK/HTTP
  fixture and complete isolated gate2,271/0fail/94ignored pass; no GPU launch.
- [ ] Remaining S1–S3 active-contract/provider/source-bound physical integration coverage.
- [ ] F0–F4 complete SDK/engine/serving/CLI feature/refusal migration coverage.
- [ ] Q0/Q1/Q2 assembled transition/physical acceptance; unchanged evidence reused
  only within actual source/numerical/owner scopes. S0 is not Q0/Q2.

# Fabric vNext transition — active implementation queue

Author: Project owner
Co-authors: GPT-6
Owner: @pvl — main-session coordinator

Base: `1c35172`

## Current direction — checkpoint key paths, then refine

- Owner reauthorized the working foundation first: close CPU/SDK/server/CLI key
  paths, verify, sign a checkpoint, then evolve parts separately. Do not hold this
  checkpoint for the entire GPU scheduler rewrite or mark the full plan complete.
- **Main does all further work; no new subagents.** Previously running tasks
  finished without cancellation. Main reviewed/integrated their results.
- Main owns all files, compiler capacity, plan reconciliation and commits. No
  push, source acquisition, GPU launch, recovery or privilege changes authorized.
- Fabric/Qwen recurrences remain cancelled. Qwen expansion remains deferred;
  accepted model/operator/lifetime evidence is historical and tuple-specific.
- HEAD at start: signed`c853ebb`, S0 source`7396984`. Dirty migration preserved,
  integrated and tested; no new commit yet. Root README/external checkout untouched.

## Checkpoint evidence — assembled gate passed

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
- CPU scheduler/queue/events/presentation now engine-owned; facade composes actual
  Qwen source/tokenizer/admission. Loading, joins and source destruction are unlocked;
  concurrent shutdown waits for cleanup and preserves failures. Synchronous
  shutdown from its own loader/worker/destructor remains unsupported/documented.
- Real reusable HTTP server and SDK pipeline checking/device APIs; no fabricated
  successful model preparation, placement, pipeline execution or future protocol.
- Temporary CLI import-shim files removed. GPU scheduler and remaining run/pull/
  cache business paths still live in CLI; F0/F1/F3/F4 are NOT complete.
- Qualification callers use real public admitted channels/host/cancel owners;
  no production test-support feature or exported isolated ledger was introduced.

## Next refinements — one coherent unit at a time

1. Preserve this tested working foundation in signed logical checkpoints.
2. Harden lifecycle reentrancy/caller-deadline ergonomics at the shared engine
   layer; keep ordinary paths simple and measure before performance changes.
3. Extend the SAME engine Router/worker to GPU, preserving the reviewed contract:
   engine owns queue/fairness/parking/presentation; coarse adapters own typed
   source/session slots, device feedback and original prepare/Ready/commit.
   No adapter scheduler/thread or feedback reconstruction from host token IDs.
   CPU/GGUF32-token chunks, native GGUF teacher-token quanta, captured HF atomic
   prompts and non-fused GGUF final-tail cohorts remain distinct. Preserve B8/4/2/1
   GGUF and B4/1 captured-HF bands, ragged/late arrivals, all-row publication,
   cancellation/known-discard/unknown quarantine and 30s output parking. No
   workspace while parked, state-only output usage, duplicate state charge or
   ordinary CUDA activation. Existing67/72-token captured prompts exceed64 quota:
   retain/report the refusal rather than silently widen eligibility.
4. Complete remaining SDK/CLI ownership, active numerical/layout contracts and
   production/qualification separation; then full F4/Q0/Q1/Q2 acceptance.
5. Profile CPU scheduling/presentation and GPU cohorts independently. Prefer
   simple shared-layer/zero-cost improvements; preserve arithmetic/source/ABI/
   BF16/reduction/signed-zero and final-reader identities. No speedup claim yet.

## Committed

Historical documentation packet:`2ff3d49`; canonical handoff refresh:`ff89b87`.
Qwen commits are preserved inputs, not S/F/Q completion proof.

### Source owners — this commit

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


## Uncommitted — remaining S1 active contracts

- [ ] Bind activation/numerical/layout/derived contracts to genuine preparation consumers; real stored encoding/callers are migrated above.
- [ ] Bind existing GPU packed kernels/activation under source+variant+numeric+device constraints; no codec-specific model engine.
- [ ] Validate remaining active-contract integration; reuse accepted unchanged codec/numerical evidence.

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

## Uncommitted — future F0 model-neutral engine

- [ ] Move CLI/serve CPU+GPU model routing/scheduler/private state into one engine ModelInstance, PreparedModel, TaskSession and one scheduler.
- [ ] One cold model/provider registration point in facade, same source alias source sharing and exact loaded ModelGeneration pin.
- [ ] Preserve CPU bounded prefill/decode and GPU native B1/B2/B4/B8 true batch semantics including ragged/late arrivals and source-specific eligibility.
- [ ] One ResourceArbiter and original worker Ready/lease/shutdown; source-valid typed committed output before HTTP.
- [ ] F0 tiny model-neutral mock/CPU tests, affected compile and signed commit(s).

## Uncommitted — future F1 public zerogravity Rust library

- [ ] Expose all real model/run/cancel/stream/inspect/model admin/alias functions through exported zerogravity API.
- [ ] Expose all artifact pull/download/providers/cache/source selections, devices/fit/dry-run/config and pipeline checking APIs without CLI.
- [ ] Preserve provider-specific HF/GitHub/Xet/OCI/S3/GCS/Azure/ModelScope/HTTP implementation, source path/symlink/shard/credential/etag/refusal handling.
- [ ] External Rust program depending only on zerogravity compiles/runs qualified local functionality and model-free tests; no CLI needed.
- [ ] F1 focused acquisition/cache/devices/run/pipeline feature tests and signed commit(s).

## Uncommitted — future F2 standalone reusable serving library

- [ ] Implement existing HTTP/1.1 GET health/models, POST chat/completions/completions, constrained sampling, JSON/SSE/usage/EOS/finish; source-specific unsupported arguments refuse.
- [ ] Serve uses engine TaskRequest and CommittedEvent, not Qwen/GPU/allocator; ingress/auth/HTTP body/stream credits in serving, scheduler/state in engine.
- [ ] Preserve CPU and eligible native CUDA request routes, real cancellation/disconnect, bounded server pending queue and 50abaf7 worker join/owned shutdown.
- [ ] Public zerogravity::serving reexports an independent embedded ServerBuilder/Handle; serving depends engine/core only.
- [ ] F2 loopback tiny server/mock transport tests and signed commit(s).

## Uncommitted — future F3 thin CLI / delete old runtime

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
- [ ] S1–S3 affected exact codec/source/state/ownership coverage.
- [ ] F0–F4 complete SDK/engine/serving/CLI feature/refusal migration coverage.
- [ ] Q0/Q1/Q2 assembled transition/physical acceptance; unchanged evidence reused
  only within actual source/numerical/owner scopes. S0 is not Q0/Q2.

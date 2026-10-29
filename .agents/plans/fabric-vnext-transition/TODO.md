# Fabric vNext transition — held phase checklist and next-implementation owner plan

Author: Project owner
Co-authors: GPT-6
Owner: @pvl — successor coordinator not yet appointed

**Scope:** all future code phases held pending a new explicit implementation instruction. This plan was created as documentation only and is not promoted to the active .agents/plans/TODO.md queue. Initial read-only source HEAD fadf2b1123309a2118725f2cc005d3edc7ea39f2, moving; base for implementation is NOT frozen. No plan commit has been made.
**Reference authority:** HANDOFF.md → README.md → PHASED_IMPLEMENTATION.md → STRUCTURAL_SKELETON.md; old preserved ../fabric-vnext entire packet.

## Committed

None. No code/refactor/source/test commit authorized during plan writing. When implementation is separately approved, append logical signed SHA and acceptance evidence here or reconcile into current owning plan per AGENTS.md. Historical implementation commits are not new Fabric S/F/Q completion proof.

## Uncommitted — future S0 COMPLETE architectural skeleton FIRST

- [ ] User explicitly authorizes actual source implementation and integration owner; re-read live HEAD, source status, active owners/checkouts, AGENTS.md and all referenced docs; preserve unrelated work.
- [ ] Create zerogravity-engine and zerogravity-serving as compiled independent Rust library crates, correct Cargo features and resource ownership (engine model-neutral, serving no facade/CLI).
- [ ] Make core semantic/effects/encoding/numerics/model/task/typed media/provider/placement/root/physical contract modules visible in Rust with actual types and explicit capability/refusal.
- [ ] Keep source exact existing ForwardPlan and original process-shared low-level ResourceArbiter, SessionAuthority, GPU event/lease as one real implementation each; avoid parallel placeholder owners.
- [ ] Make artifact container/codec/source/acquisition/derived homes, CPU numerics/quant, GPU backend/quant/program/lease/transfer/qualification, family source/forward/state/specialization homes compile-visible.
- [ ] Make public zerogravity API visible for all user functionality (RuntimeBuilder, models, run, acquisition/cache, devices, pipeline, serving, diagnostics) with typed Unsupported for unavailable future tasks.
- [ ] Make CLI target args/commands/output/signals structure visible while preserving existing currently supported execution through source migration.
- [ ] Any todo! only private, not live/production/serve/public supported path. No fake model, zero-logit kernel, fabricated physical Ready or permit.
- [ ] Update existing scripts/check-architecture.py/check fixtures for new DAG, no serving→facade cycle or engine→Qwen/GPU; ensure CPU-only default compile avoids GPU compiler/network stack.
- [ ] S0 targeted fast package checks and small mock/refusal, one complete signed structural commit, no full model/GPU qualification.

## Uncommitted — future S1 source format/quant exactness

- [ ] Separate source stored ScalarDType/QuantScheme, dynamic ActivationEncoding, numerical/reduction policy, packed physical layout and derived lossy artifact identity.
- [ ] Move exact artifacts q4/q6/q8, F16/BF16 scalar codec source and malformed tests; preserve actual codec decoder only once.
- [ ] Move CPU source-exact Q4_0/Q4_1/Q6_K/Q8_0 projection + Q8 activation and BF16 math to proper owners; keep dequantized pure F32 reference separately.
- [ ] Retain Q4_K scalar decoding and explicit unqualified CPU physical matvec refusal; Q1/Q2/Q3/NVFP4/EXL3 unimplemented placeholders.
- [ ] Bind existing GPU packed kernels/activation under source+nvariant+numeric+device constraints; no codec-specific model engine.
- [ ] S1 small byte/source numerical/refusal test and affected cargo check; signed coherent commit(s).

## Uncommitted — future S2 family math and GPU provider

- [ ] Keep one Qwen forward traversal and independent Gemma source math, one existing ForwardPlan recorded as canonical SemanticProgram.
- [ ] Preserve GPU production model dependency removal (263ff79) and concrete source-independent provider/ABI preparation.
- [ ] Separate CUDA GDN kernel implementation and exact code+driver/ISA/numerical identities from historical finite Qwen prompt fixtures/qualification source.
- [ ] Preserve newest captured fresh-prefill/reference layout identities, short contexts, original source permission; no automatic ordinary CUDA enable.
- [ ] Move only remaining genuine source/model wrappers; no giant unnecessary compiled .cu rename/rewrite.
- [ ] Independent CUDA/HIP/Vulkan feature source-only check, family and source refusal fixtures; signed commit(s), hardware later.

## Uncommitted — future S3 canonical root/resource

- [ ] Converge reference TaskCoordinator and original real SessionAuthority RootBinding/RootAttempt under one logical complete-domain state publication protocol.
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

## Tests and verification — THIS documentation task ONLY

- [ ] Verify all new side transition files exist, are UTF-8, include full user/source/architecture/correctness/phase content and local links resolve. Perform read-only check before reporting completion.
- [ ] Create complete consolidated archive under .agents/docs that reproduces all 7 frozen original files and all new transition files in known order (not merely links), plus index and SHA-256 documentation hashes.
- [ ] Confirm Git status changes ONLY new side documentation paths and original frozen plan files remain byte-identical; no source, active TODO, build/test, commit or push.
- [ ] Present next-agent prompt in chat and do not begin S0 without a new explicit implementation instruction.

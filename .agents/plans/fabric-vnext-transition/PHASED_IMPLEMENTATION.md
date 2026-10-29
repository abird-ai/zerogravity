# Fabric vNext — precise implementation phases, logical units and signed-commit program

**Plan only as of 27 October 2026.** The first code milestone is a complete compilable skeleton, NOT several partial design-only commits. Then implement all currently real features through it, delete pre-v1 compatibility paths and run ONE consolidated heavyweight verification cycle. Reference source is moving; first actual implementation agent must re-read current HEAD/source before editing. Model numerical/physical safety is never silently waived.

## 0. Definitions and strict ordering

Tranches:
- **A / structural:** S0 (whole skeleton) → S1 (encoding/CPU/GPU quant) → S2 (model/provider/semantic boundaries) → S3 (state/resource/physical ownership). All of the architecture is already visible after S0; later S phases relocate real source and consolidate actual owners.
- **B / functionality:** F0 (single engine) → F1 (public SDK+acquisition/devices/config/pipeline) → F2 (independent serving library) → F3 (thin CLI/no compatibility) → F4 (complete existing feature coverage and targeted transition repair).
- **C / acceptance:** Q0 (one broad integrated suite and selectively authorized physical cases) → Q1 (focused bugfix commits without repeated huge suites) → Q2 (final integrated qualification/status baseline and closeout).
- **D / later product expansion:** see ROADMAP_AFTER_TRANSITION.md; not blockers of current refactor.

**Per-phase rule:** one *smallest coherent end-to-end responsibility unit*, tested with quick checks, **one signed commit after it is coherent**; do not make broken commits or giant mixed half-rewrites. If a phase exceeds reviewable boundaries, split suffix A/B/C (e.g. S1.artifact, S1.cpu) each with its own strict source/caller/test closure. Record original source test evidence and failure/refusal scope. No Git push/rebase/tag unless separately approved.

**Checking rule:** format, static dependency check, affected crate cargo check/clippy, bounded unit test(s) every phase. Defer full scripts/check.sh, 27B/Gemma trained model runs, long GPU fault sweeps, hardware physical tests, C16/C32/C64 perf, long-context and all model/driver matrices until final assembled Q0/Q2. The user specifically asked this. Detailed in QUALIFICATION_AND_COMMIT_POLICY.md.

## S0 — whole architecture in compilable Rust, FIRST code commit

**Input:** source at newer live HEAD, plan documents read, active owner claims checked. Existing in-progress CUDA/serving source must be preserved; no blind reference snapshot replay.

**Work:**
1. Add independent zerogravity-engine and zerogravity-serving Cargo workspace members. CPU-only base, feature forwarding, no dependency cycles or mandatory HTTP/GPU for direct use.
2. Add all top-level core SemanticProgram/effect/value/encoding/numeric/TaskDriver/media/model/provider/placement/state/resource contracts and actual module declarations. Existing ForwardPlan and physical Root/Arbiter remain real; don't add parallel runtime implementations.
3. Add engine Runtime/ModelInstance/Catalog/Router/Admission/Scheduler/TaskSession/PreparedBundle/Residency/Placement/Output modules and types.
4. Add serving server lifecycle/HTTP/OpenAI/Responses/SSE/ingress/auth/metrics modules; only actual supported protocols claim working; optional endpoints typed refuse.
5. Add facade complete Runtime, model/store/acquisition/cache, device/placement, direct run/events, pipeline, server, diagnostics API surface and one cold registration/composition point. Compiled future builders may typed refuse.
6. Add CLI args/commands/output/signals target directories, with old current command implementations temporarily still in use until F3. No permanent compat facade.
7. Add minimal test/fixture and architecture dependency guard modifications; model/source/device support ladder must remain truthful. All code imports resolve and the full tree is visible via IDE/Cargo/Rustdoc, not just docs or empty folders.
8. Private TODO macros ONLY unreachable/excluded; ordinary new API methods return typed Unsupported where not implemented; current supported paths continue working via old owner until migrated, never panic as a skeleton substitution.

**Commit recommendation:** signed `refactor(architecture): establish complete fabric vnext Rust skeleton`.

**Fast check:** cargo fmt, updated static architecture checker and fixtures; package-local offline Cargo check for core, artifacts, engine, serving, facade, CLI CPU-only (no trained model), and tiny public builder/typed Unsupported tests. Optional compile-only async example for external zerogravity dependency. No full check.sh or GPU device.

**Exit:** immediate complete architectural structure in actual compiling code, no second resource owner or model engine, no false capability; reversible exact source behavior not yet all moved.

## S1 — exact codec/quantization/precision hierarchy, REAL code moved

**Work:**
- Distinguish stored ScalarDType vs QuantScheme/StoredTensorEncoding, activation coding, NumericalContract and reversible PreparedTensorLayout in core.
- Move actual artifacts q4/q6/q8 scalar decoders and F16/BF16 conversions, original source metadata and tests into artifacts codec homes. Keep checksum, lengths and malformed refusal exact.
- Move CPU original GGUF source dot/activation packing and BF16 numerical paths to precise CPU ownership, preserving all reduction/cast/NaN/signed-zero behavior.
- Retain GPU quantization existing kernels by physical owner, bind typed source layout/activation/numeric eligibility; no unnecessary .cu body rewrite.
- Existing Q4_K source CPU matvec remains Unsupported; Q1/Q2/Q3/NVFP4/EXL3 remain README-only/unqualified; never generate fake new decoder.
- Update true callers/tests and remove any replaced old producer; no duplicate quantization runtime.

**Suggested commits:** S1.a artifacts, S1.b CPU, S1.c GPU if each has an independent compile closure, or one S1 if compact.

**Fast check:** architecture/format check, affected crate check, small original scalar codec fixtures, CPU q4/q6/q8 exact math tests, GPU source-only quant fixture. No original full-model/physical gate.

**Exit:** source→numeric→kernel representation and actual codec implementations cleanly separate and usable by Qwen/Gemma without a model named format parser.

## S2 — family semantics / prepared provider / production-vs-qualification ownership

**Work:**
- Use the already moved Qwen/Gemma family math/config/plan and shared ForwardPlan as one canonical SemanticProgram; no new source-independent duplicate operator interpreter.
- Remove remaining source-specific Qwen/Gemma eligibility from generic physical worker and cold application route: model owns exact source roles and state schemas, GPU owns physical kernel implementation and ABI, facade/engine selects only qualified pair.
- Prepare closed source/numeric/shape region executable and fallback via common interface; actual CUDA/GDN parameterized variants are preserved, with source image signatures and actual buffer/reader proof.
- Explicitly separate historical finite source prompts, expected snapshots, captured qualification case catalogs and fault-run decision trees into unit/integration/qualification support. Real reusable kernels/resources remain production.
- No GPU normal/build/optional model dependency (already fixed); test-only dev fixtures allowed and gated. Preserve HIP and Vulkan feature isolation and selected source identities.
- Work toward generated/native source separation where appropriate; don't rewrite hundreds of CUDA files solely to rename modules.
- Current CUDA original HF ordinary route may still refuse, even if finite qualified captures are present; real general profile requires separate future source/physical evidence.

**Suggested commits:** S2.a family plan/semantics; S2.b GPU provider source/binding; S2.c qualifier separation if needed.

**Fast check:** default and requested CUDA/HIP/Vulkan package/feature compile without physical device; source/plan recorded math unit, negative source/ABI/fixture tests, dependency checker. Avoid full GPU/x2 or long trained model run.

**Exit:** one semantic/program source truth; physical backend is model-neutral, qualification cannot advertise source capability, native optimized codes retained.

## S3 — one root protocol and one physical budget

**Work:**
- Converge core/state reference TaskCoordinator and core/session_root SessionAuthority/RootBinding/RootAttempt semantics under one StateRoot/epoch/PreparedDomainVersion/CommitCandidate/ReadyWitness/CommittedEvent protocol.
- Retain original physical SourceIdentity/RootBinding checks, monotonic counters, frozen restore, cancellation/stale epochs and actual device physical Ready/Unknown. No separate root publisher in HTTP or model family.
- Keep one process-shared ResourceArbiter in low-level core. Engine can own coarse source/model/session/tenant admission policy, but must never create a new spendable pool. Material allocations and pending/captured readers charged under original ledger.
- Establish all-or-none multi-domain reservation as a future extension where needed, but do not block today's working single-domain route on unproven multi-device implementation. Interface may exist now with typed refusal until implemented.
- Route CPU and selected GPU/Model family state through shared protocol, preserving logical vs physical Ready vs transport output.
- Bind actual source/tenant/quant/numerics/root/frontier state identity, future media typed domain placeholders, compatible prefix-state contract (paged complete-domain implementation deferred).

**Suggested commits:** S3.a state protocol; S3.b resource and physical receipts if independent.

**Fast check:** targeted reference root/cancel/stale/foreign-state and ResourceArbiter tests, GPU source-free physical lifecycle/quarantine fixtures where practical; compile affected crates. No full trained-model or remote/cross-device hard suite.

**Exit:** no competing logical root/physical wallet, all execution preconditions are explicit and source safe.

## F0 — model-neutral execution engine with current real model paths

**Work:**
- Move current CLI/application inference orchestration (serve/engine.rs, serve/gpu_engine, runtime session construction, Qwen source router) into engine's ModelInstance/PreparedModel/Session, generic TaskRequest and StepEnvelope.
- The facade registers compiled Qwen/Gemma/CPU/GPU capability descriptors at cold setup. Engine knows only model-neutral task/source/quant/shape/physical requirements; model family and GPU implement corresponding adapters.
- Same source model aliases share authenticated immutable weight lease; independent request sessions and roots. Resolve alias revision once at admission, pin session model generation, no per-token catalog hot lookup.
- Consolidate CPU serving fair queue/worker, resident GPU fused cohorts and native B1/B2/B4/B8 as task-aware scheduling policies under one engine; don't demote native multi-row hardware work into N loops over single-request inference.
- Admission has ingress quota interface, model/session/state/preload and device StepPlan constraints; only one physical ResourceArbiter pays costs. Persistent request slots and per-token metadata can be expanded later; current finite batching preserved now.
- Output events source-validated by sole coordinator before external transport. Worker final destruction/stop and queued/cancelled work retain original device/host owners.

**Suggested commits:** F0.a prepared model/registration; F0.b CPU engine session; F0.c GPU batch engine session, each with end-to-end owner and narrow tests.

**Fast check:** compile engine/facade and tiny mock model; CPU small request/StateRoot smoke and selected GPU source-only recorded packet/slot tests; no full trained GPU matrix.

**Exit:** one model-neutral engine accepts already qualified Qwen CPU and selected GPU execution through same session/alias/resource contracts, while HTTP remains a client.

## F1 — public zerogravity Rust library for ALL CLI capabilities

**Work:**
- Curated high-level API RuntimeBuilder/Runtime, ModelStore and model methods, Artifact/SourceInspector, devices inventory/fit/dry-run and placement, TaskRequest and CommittedEvent streams, cache/snapshot management, typed Pipeline check/compile/run (real only where actually supported), source acquisition/pull operations, admin status/metrics.
- Reuse actual existing artifacts/acquisition transport and immutable cache publisher, but lift CLI-only provider request normalization/protocol control into libraries. CLI keeps flags and human progress only.
- Pull candidates, revision pins, S3/GCS/Azure/OCI/ModelScope/Xet, filters and source quant selection/conditional resume keep error and caching behavior. No implicit acquisition from inspect/list when not requested.
- Source/window/processor and device path handling must remain bounded, secure and typed; library functions accept explicit policy/context rather than reading uncontrolled shell environment or exiting process.
- Define public errors/feature support receipts, cancellation tokens and async/sync usage ergonomics; no fake world-capability enum.
- Add external Rust example depending ONLY on zerogravity with local inspect/pull/dry-run/run and no CLI import. The example can avoid actual model bytes in cheap CI and check safe refusal if none available.

**Suggested commits:** F1.a runtime/run/devices; F1.b artifacts/pull/cache; F1.c pipeline/config/status, each complete.

**Fast check:** facade/model/artifacts/CPU package builds and small acquisition/cache/CLI-equivalent config fixtures, no network downloads, no full GPU/model run.

**Exit:** every actual CLI business operation has a typed public library equivalent, whether or not CLI is migrated yet.

## F2 — independent reusable zerogravity-serving library

**Work:**
- Extract working CLI serve HTTP parsing, OpenAI /v1/models, chat/completions, completions, constrained greedy sampling, strict error handling, SSE and stop/usage, health, backlog/connect deadlines, output credits and join-on-shutdown to serving library.
- Serving routes to model-neutral engine TaskRequest and consumes source-valid CommittedEvents; it does not own Qwen tokenizer/GPU sessions/ResourceArbiter/worker TokenScheduler.
- Preserve source- and backend-specific finite capabilities and explicit rejects; no false embeddings/Responses/tools/media support; future endpoints can return Unsupported or not be registered.
- Protocol ingress/auth/rate/tenant limits remain serving-controlled, but model/state/device work scheduling is engine-controlled; transport socket/ACK never publishes neural state.
- Lifecycle uses ordinary Rust ServerBuilder/ServerHandle, explicit bind/start/stop/drain/cancel; can be embedded in another Rust program. No process::exit or global SIGINT from library.
- Ensure server crate depends engine/core, facade reexports server builder, CLI depends only facade. CPU build with serve feature includes HTTP stack only if selected.

**Suggested commits:** F2.a HTTP/stream/error and parser conformance; F2.b serve loop/bind/auth/lifecycle; F2.c GPU and CPU transport adapters/feature forwarding if needed.

**Fast check:** test serving crate with fake engine and real small CPU fixture/loopback-only HTTP; existing serve-lifecycle test focused subset and engine worker shutdown test, no full pinned GPU/large SSE load.

**Exit:** user can import ONLY zerogravity, build Runtime, start/stop API server, send direct task and HTTP request concurrently, sharing loaded model/resource ledger.

## F3 — thin CLI and clean removal of old pre-v1 execution paths

**Work:**
- Replace CLI production command handlers with calls through public zerogravity facade: run, serve, inspect, pull, models, cache, devices, pipeline check/run, config and diagnostics; retain command-specific output formatter and process/signal bridge.
- Delete CLI serve/engine/gpu_engine/gpu_batch source after F0/F2 own them, and pull/source/device/pipeline duplicated business code after F1 moves it. No permanent legacy helper/alias shims or model-named runtime branches in CLI.
- Make CLI source feature graph library-only; compile architecture checker forbids importing GPU/model/artifact/engine/serving direct for project functionality. CLI may still import third-party serde/HTTP model JSON formatting or terminal dependencies as appropriate, but not implement HTTP server itself.
- Existing CLI command spellings may be adjusted pre-v1 for clean UX if necessary. User explicitly waived API compatibility but requires all *behavioral capabilities* previously supported to be accessible via the new API and CLI.

**Suggested commit:** `refactor(cli): use zerogravity SDK exclusively and remove legacy execution`, optionally split command families only if each is independently tested.

**Fast check:** CLI feature help/config/arg diagnostics and representative run/inspect/pull/serve/pipeline commands with test fixtures; no expensive trained model/network side effects beyond test loopback.

**Exit:** CLI is a *real thin facade*, no alternate hidden model engine or physical/HTTP service.

## F4 — current feature completeness and transition readiness

**Work:**
- Review FEATURE_PARITY_MATRIX.md row by row, including exact source formats/quant, CPU Qwen0.8B/2B, selected GPU CUDA/HIP/Vulkan diagnostics, CPU HTTP/OpenAI, fused GPU service, real cancellation/shutdown, artifact acquisition/cache source rights, device PCI aliases, pipeline check/refusal, numerical/source/physical fault fixtures, all current feature combinations.
- Fix remaining architectural gaps and compile/test regressions with small targeted patch commits. No temporary todo in any current supported route; any remaining future source/task/codec returns typed Unsupported as before.
- Remove obsolete compatibility aliases/module paths and duplicate old schedulers, allocators, stage controllers, source parsers. Verify no production code imports qualification datasets/features to choose ordinary execution.
- Build current features' source-level code; check no new unsupported model is advertised because a new module exists.
- Preserve old passing and explicit failed/ignored coverage, including x2/gfx90c hardware hold and public API limits.

**Suggested commits:** F4.a source and app/codec cleanup; F4.b remaining serving/runtime parity and feature cleanup, each with lightweight tests.

**Fast check:** relevant model/codec/API assertions, static checker, strict lints for affected crates, optional source-only CUDA/CPU package checks and model-free loopback tests. Save giant original-checkpoint physical suite for Q0.

**Exit:** all current features rehomed and code clean, no parallel legacy runtime, no false support, ready for broad verification.

## Q0 — ONE consolidated full post-migration correctness/feature gate

**Run only after S0–F4 source migration stabilizes**, using repository's approved isolated/controlled runner and source permissions:

- Recheck clean signed source baseline, source/model/driver/Cargo-lock exact identities; ensure no other agents are mutating test tree.
- Full `scripts/check.sh` / equivalent complete environment-required build+fmt+clippy+source/correctness/qualification tools, architecture checker, new engine+serving+SDK embedded tests.
- Full format/malformed/activation/reduction numerical suites; current supported Qwen and related Gemma/27B tests under exact original model/task/quant/shape/backend evidence; complete source/state/physical/fault/service tests.
- Real CUDA/HIP/Vulkan and trained source-model cases only on authorized safe hardware with actual models/toolchains; obey gfx90c hold, no unsafe resets/NPU clocks, no automatic remote enrollment.
- Real API JSON/SSE, worker shutdown, connection/cancel, aliases and direct SDK+embedded server parity, file cache/source acquisition and device selection with real negative-path checks.
- Record passed/failed/ignored/unsupported/unexecuted separately and attach source/driver/model/kernel/bin identities. Future unimplemented pipeline/media/codec features must continue refusing.

**One heavy cycle**, not full suite per S1/S2/F0 move. If physical hardware unavailable, Q0 source-free/host verification can pass, but relevant device cell remains unqualified; don't claim all required physical acceptance.

## Q1 — bugfix rounds after consolidated gate, NOT repeated full qualification

If Q0 reports regressions:
- Assign each defect to *lowest shared responsible layer* (codec, semantic, source/model family, physical provider, session/root, engine/route, serving/protocol, facade/CLI), not per affected model or endpoint.
- Fix causal bug with an independent, small failing regression test/fixture, then run only affected package/mode, source-free targeted checks, architecture checker, format and strict lints.
- Commit each coherent correction in signed bugfix units `fix(fabric): ...` or `fix(codec): ...` with test and exact evidence. Keep earlier broken diagnostic/negative receipts as failed history; never rewrite them green.
- Do not start broad physical/long suite again after every patch. Maintain a finite issue list and rerun one **final full gate** when all targeted defects are repaired and source is stable.
- If critical safety/refusal broken, do not defer correction even if heavy qualification postponed; hold unsafe public feature until fixed.

## Q2 — final integrated acceptance and closeout

- One last complete assembled check and only genuinely needed/authorized source-model/physical regression subset, with all scopes and exclusions named.
- Embedded Rust SDK direct model run AND independent server start/stop from same public dependency; same model aliases, immutable source and physical arbiter; independent sessions/root.
- Existing qualified HTTP CPU/GPU source requests, token/EOS/JSON/SSE/status, batching, cancellation, graceful shutdown and model/source/codec support matrix correct.
- Record performance baseline/p95/TTFT/true ITL, cold/warm runtime and physical memory for selected authorized source/hardware; do not assert vLLM parity without matched quality and device.
- Confirm no duplicate legacy HTTP/router/allocator/public runtime path, no permanent compatibility glue or reachable todo! in supported routes.
- Update docs, README/whitepaper and current plan ownership **only with source-backed facts** and explicit coordinator role; one logical signed closeout commit if warranted, no push without request.

**Transition complete** means all previously real functionality works correctly under new architecture; all unsupported/future capabilities still truthful. It does not mean all planned multimodal/distributed/SSD/model additions suddenly completed.

## 4. Commit/test matrix (overview)

| Unit | Primary source owner | Must pass before signed commit | Expensive deferred |
| --- | --- | --- | --- |
| S0 | Workspace/core/engine/serving/facade/CLI skeleton | fmt, Cargo feature graph, dependency check, small type/refusal tests | All complete model/physical suites |
| S1 | artifacts/CPU/GPU quant | codec byte tests, source CPU numeric fixtures, affected cargo check | trained model full GPU |
| S2 | family/GPU/compiler | source plan/ABI/qualifier guards and compile | actual CUDA model runs, 27B/Gemma |
| S3 | core state/resource/physical | stale-source/epoch/root/lease and alias/admission microtests | full device fault, long context |
| F0 | engine/registered providers | mock + tiny CPU session and native source-free cohort | end-to-end trained original model all sizes |
| F1 | public SDK/acquisition/devices | library source/cache/mock API and command-parity fixtures | large source download/IO |
| F2 | serving library | independent HTTP fake backend, CPU loopback SSE/shutdown | full GPU-serving stress |
| F3 | CLI | compile all representative features, CLI parse/format and source-free commands | full model benchmark |
| F4 | all owners cleanup | targeted current feature tests and architecture source audit | major suite |
| Q0 | integrated controlled runner | one major full check + authorized physically scoped tests | n/a |
| Q1 | bug owners | narrow regressions + fmt/dependency/package checks per bugfix | new major suite per patch forbidden |
| Q2 | integrated runner | final full gate once stable + selected physical evidence | future model/backend expansions not required |

## 5. Parallelism and conflict resolution

Safe after S0 schema:
- **Source/codec lane:** S1 artifact/CPU, Qwen/Gemma semantic source.
- **Physical lane:** S1 GPU/S2 CUDA GDN/native ABI/qualification separation.
- **Core/state lane:** S3 immutable RootBinding and original ResourceArbiter.
- **Engine lane:** F0 Runtime/ModelInstance/TaskScheduler and F1 SDK shape.
- **Serving/CLI lane:** F2 protocol extraction, F3 binary thinning, only after engine/SDK API interfaces freeze enough.

Coordinate all shared files through one integration owner, isolated worktrees with original source read-only per AGENTS. No concurrent model+GPU edits on same source/release root. Always preserve other agents' uncommitted work and signed old receipts. Re-evaluate source changes before applying historical line-number path descriptions.

## 6. Future milestones (not part of current structural closure)

After Q2, plan separate features:
- A1 real original-HF broader source/Qwen3.5 fresh47–50 and long contexts;
- A2 true temporal prefill and persistent paged hybrid prefix state, more vLLM-like goodput;
- A3 Qwen27B source/SSD and Gemma E2B/26B-A4B/31B general trained support;
- A4 source-exact Q1/Q2/Q3/NVFP4/EXL3 with quantizer/quality and physical gates;
- A5 extreme MoE, learned n-gram/Engram, MTP/drafter/sparse attention;
- A6 encoder/ASR/TTS/diffusion/image/video native TaskDrivers/codec/resident stage composition;
- A7 NPU/distributed/heterogeneous placement and trusted peer protocol;
- A8 matched vLLM/llama.cpp and vLLM-Omni whole-task throughput/fairness/latency, resource/energy/Pareto tuning.
Each future item inherits original acceptance and public-library/serving/CLI boundaries.

See ROADMAP_AFTER_TRANSITION.md for complete later feature ordering.

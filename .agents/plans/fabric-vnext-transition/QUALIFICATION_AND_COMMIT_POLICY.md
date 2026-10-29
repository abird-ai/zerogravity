# Fabric vNext — qualification, testing cadence, bugfix policy and signed commits

**User-directed testing priority (27 Oct 2026):** Implement the complete architectural skeleton and perform all major source relocation + engine/SDK/serving/CLI migration **before** running long all-model/real-GPU/benchmark qualification. Each phase still compiles and passes a small relevant test selection; **one integrated post-transition qualification** finds cross-layer issues, followed by low-level targeted bugfixes and **one final full gate** when stable. No repeated expensive full suites after every moved file.

This adapts, but does not annul, the existing all-model correctness policy at .agents/plans/numerics-and-context/CORRECTNESS_POLICY.md. Exact source arithmetic, buffer lifetime and source-rights checks are never waived. A failed critical structural check prevents that phase's commit; a long unrun hardware case stays explicitly unqualified until actually run.

## 1. Three tiers of validation

### Tier L — FAST per-commit structure/contract checks

Run for EVERY signed logical phase:
- `cargo fmt --all -- --check` (or scoped before full worktree integration).
- `python3 -B scripts/check-architecture.py` and its small checker fixtures; extend existing checker to know engine/serving/facade/CLI after S0.
- Cargo dependency feature tree and build checked at least for the changed affected crates with `--locked --offline` and `--no-default-features` where appropriate. Cargo compilation is not a model/device qualification.
- `cargo check` or targeted `cargo clippy --all-targets -- -D warnings` on affected packages/features only; do not treat intentionally disabled physical qualification as a release feature.
- A handful of focused source-free/original small fixtures for changed ownership: codec byte vectors/negative parser, root epoch/source guards, queue cancellation, server JSON/SSE loopback, CPU reference small geometry, assembly registration and dynamic shape refusal.
- No compiler/driver/device launch unless test is explicitly user-authorized, safe, quick and does not require long model qualification. Default no GPU device in this tranche.
- Verify supported production caller not connected to `todo!()`/fake SourceIdentity/Ready; capability status remains explicit. Test current error/Unsupported, not just structural imports.
- Inspect `git diff --stat` and scope, ensure no other owner's paths/staged changes; commit only owned, coherent source+targeted tests, signed as repo requires. No push unless separate authorization.

Tier L is intentionally brief, sufficient to keep Rust sources compilable and ownership boundaries sane during mass relocation. It is **not** a substitute for Tier H.

### Tier M — MEDIUM per-completed responsibility (selective, not every file)

Only when a coherent source behavior moved (S1 codec, S3 root, F2 service) and a small isolated suite is inexpensive:
- Artifacts codec/source and core/CPU exact small reference tests.
- Current GPU source-free finite kernel source/ABI/driver-image invariant tests without actual GPU launch.
- Mock-engine serving JSON/SSE parser, cancellation and shutdown lifecycle in a loopback test namespace.
- Runtime model-neutral fake ModelDefinition/Provider, independent TaskSessions and original real CPU tiny source if available without licensed model download.
- Source/CLI run_policy device path and config/pipeline structural cases, no remote IO.
- Standalone library embed/no-CLI tests.
- No matched full 0.8B/2B/27B/Gemma physical model matrix, C16/C32/C64 performance, long source-shard SSD streaming or exhaustive fault injection here.

If an M suite is unusually slow, source-private or requires hardware, move it to Q0 with an explicit held mark. Do NOT label phase fully qualified because an M suite was skipped.

### Tier H — HEAVY one consolidated post-migration integrated gate

Run once **after S0–S3 and F0–F4 sources/traits/interfaces settled**, using approved isolated runner, permitted source/model/driver and hardware.

Repository's documented first-party full gate can be invoked where authorized with its execution wrapper:

~~~sh
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 \
  python3 scripts/native-gpu.py -- bash scripts/check.sh
~~~

This command is an example of the existing full isolated gate; it is not an unconditional instruction to launch physical GPU work or change system settings. Read AGENTS.md, script semantics and current plan permits first. The wrapper/config may be revised in current source by execution time. Running in an unapproved environment that lacks nested namespace/device access must fail safely; never bypass the isolation with a direct unsandboxed substitute.

Tier H coverage:
- Strict compiled Cargo feature and target matrix: CPU-only default, source/family, artifacts/acquisition/cache, serving-only, full SDK/CLI, optional CUDA/HIP/Vulkan qualification combinations.
- Source codec and format exactness/malformed/overflow and scalar/special value/per-block action tests.
- Source-specific trained model semantics/reference where actual pinned checkpoints can be used, declared all-logit/operator/StateRoot and independent original checks, natural feedback (not only forced teacher).
- CUDA/HIP/Vulkan physical kernel/ABI/source/driver Ready/Unknown/quarantine/cancel tests only on authorized healthy hardware; no unsupported gfx90c work.
- Current native Qwen0.8B/2B CPU/selected CUDA HTTP and exact finite B2/B4/B8 request/worker pipeline; Gemma E2B/Qwen27B physical diagnostic gate only where original scopes and access satisfy source/driver holds.
- Original semantic H/G/V expected failures or unexecuted/ignored cases kept visible, no false "suite passed all" if hardware scopes missing.
- Embedded public Rust SDK direct model and independent server using same model+resource manager, CLI parity and cross-model alias/tenant failure cases.
- Cache/acquisition/portability/host memory/OS/driver/unknown-buffer retirement, bounded concurrency, cancel, shutdown and release packaging where actually applicable.
- Formal modes by source checkpoint/quant/shape/TaskKind/numeric/backend/compiler/driver; compare only meaningful matched whole-task/model metrics when performance is requested/permitted.

If a Tier H subcase cannot run, mark precisely Unexecuted or Held and leave the corresponding capability **unqualified**. Do not downgrade safety/refusal requirements to avoid time.

## 2. Practical phase-level testing

| Phase | Minimum Tier L | Optional Tier M | Tier H deferred |
| --- | --- | --- | --- |
| S0 whole skeleton | fmt + DAG + feature graph + engine/serve/facade/CLI cargo check | new embedded builder/refusal microtests | everything big |
| S1 codecs | format enum/source codec crate compile + type-move guards | exact q4/q6/q8 scalar, CPU activation numerics | real trained model/cross-GPU |
| S2 model+GPU | source-only CUDA/HIP/Vulkan feature compile, no forbidden imports | region source/ABI/qualification refusal fixtures | actual Qwen HF source physical/27B/Gemma |
| S3 state+resources | source-free root/fence/memory closure | stale root/alias/epoch/cancel/quarantine | full device fault/long state |
| F0 engine | engine/facade compile | two alias/mock models, CPU small session/queue | C16, real multi GPU performance |
| F1 SDK/artifacts | public library compile (without CLI) | model inspect/pull/cache/devices/dry-run/YAML error tests | live network large source |
| F2 serving | serving independent compile and strict imports | loopback JSON/SSE, shutdown/disconnect, mock engine | full original GPU model server load |
| F3 CLI | only public facade project dependency + run/serve compile | command/JSON output/parity and CLI error tests | full trained model end-to-end |
| F4 cleanup | full modified crate feature compilation/architecture/source negatives | representative current feature smoke/remaining test moves | one complete Q0 |
| Q0 integrated | all Tier H applicable | no partial fake pass | run once |
| Q1 bugfix | fixed component Tier L and targeted M | critical focused regression | don't repeat all H per fix |
| Q2 closeout | final Tier H on stable changes | material physical/model tests as access permits | report remaining held cells |

## 3. Q0/Q1/Q2 protocol

**Q0: initial broad transition qualification**
1. Snapshot exact commit/source identity and driver/Cargo/toolchain/model hash, confirm owner release from parallel changes; don't run a stale binary.
2. Run one full trusted check.sh and applicable GPU/model/source/serving suites once; record real passed, failed, explicitly ignored, unsupported and unrun rows, with diagnostic logs and source generation.
3. Separate defects by lowest responsible owner; never patch similar Qwen/Gemma/HTTP paths separately when one generic source/ABI/Root cause exists.
4. Build consolidated bugfix queue with precise smallest failing fixture, criticality and whether physical device needed.

**Q1: stabilization bugfix commits**
1. Group coherent low-layer faults; write regression test reproducing error, fix shared layer.
2. Run Tier L/M affected scopes and architecture DAG, not whole H again. A critical safety/root/source permit regression may require immediate disabling of affected runtime feature pending proof.
3. Commit signed `fix(fabric): ...` with accurate failure cause, source/model/physical evidence, original Q0 failure remains failure in historic record.
4. Proceed until integrated candidate is structurally/numerically plausible, active owners reviewed and no known broken Tier L/M.
5. Run a single next full H only when the bugfix wave is complete. If it still finds failures, bounded next Q1 cycle with justification, not automatic expensive loop on every attempted fix.

**Q2: final full acceptance**
1. Re-run complete H once on candidate with fresh changed binary/source hashes and appropriate real hardware/sources where permitted.
2. Mark every functional original row from FEATURE_PARITY_MATRIX, source/decoder/model/physical/client protocol and SDK/CLI proof; unexecuted cannot inherit old green.
3. Show new code dependency tree, zero duplicate legacy inference runner, no model-name switches in generic physical service/HTTP, and no reachable todo!() for supported paths.
4. Measure selected matched before/after complete request overhead and source fidelity; don't claim new vLLM superiority without equal artifact/numeric/hardware/driver.
5. Update architecture/status docs and accepted plan ledger, sign logical closeout if user has authorized implementation; no push/release without explicit permission.

## 4. Light-check budget and performance discipline

Why defer expensive checks: the user wants the architecture fully in place quickly and does not want hours of repeated GPU/large-model qualification for moved structs/methods. A source-only compile/short golden test should be sufficient to commit each coherent migration unit, with honest pending final certification.

**Bounded cost heuristics (not SLA promises):**
- Prefer cheap static Cargo graph and source-only check for S0 before building all native CUDA crates.
- Use an affected package with explicit features, not `--all-features` that inadvertently turns on qualification and GPU toolchain.
- One tiny reference golden/malformed/epoch scenario when a numerical/state function moved; no full model step test at each callsite.
- When a source/ABI identity is changed by a rename, record pending requalification, do not force a full physical rerun immediately.
- Use specific named Cargo tests rather than an exhaustive giant compilation workload every commit.
- Keep expensive model/test source assets read-only and access-controlled; no unauthorised checkpoint pull.
- Run real device only with approved source and known device holds, driver power safety and fences.
- When a fast check fails critically, stop, repair and rerun that fast check before committing; deferral applies to **expensive independent qualification**, not broken builds/memory/source safety.

## 5. Signed commit hygiene and parallel worker integration

Current AGENTS.md requires signed Git commits and explicit narrow ownership. Plan a commit for every coherent phase/subphase, **not one huge unreviewable final diff**:

~~~text
refactor(fabric): establish complete vnext workspace skeleton
refactor(codec): separate source encoding and exact decoders
refactor(cpu): own source numeric quant projections
refactor(gpu): bind qualified packed and recurrent region candidates
refactor(models): share one semantic program and source families
refactor(state): converge session publication and physical roots
refactor(engine): centralize model-neutral loaded runtime and scheduler
feat(sdk): expose complete embedded zerogravity API
refactor(serving): extract reusable OpenAI and streaming library
refactor(cli): use public zerogravity library exclusively
test(fabric): reconcile current feature parity before full acceptance
fix(fabric): <lowest-layer-Q0-regression>
docs(fabric): record final source/quality/physical acceptance
~~~

Messages are suggestions; actual unit should describe what source really changed. Record scoped source/test evidence and open Q0 gate after each S/F commit. Never falsely mark long H completed because a package builds. Do not disable commit signing, don't rewrite old source/test history or push on planning authorization. Other agents' staged or unstaged paths cannot be automatically included.

For parallel work: isolate candidate branches/worktrees with explicit read-only original/source policy; one coordinator merges, checks staged diff, and integrates source/code/release boundaries serially. Keep shared Cargo/engine/state/owner files under one writer; don't have four agents independently invent runtime TaskId/BufferLease types. Subagent results are not accepted without code review and actual targeted evidence.

## 6. Acceptance that must never be hidden by deferral

- No malformed codec/source bytes accepted as valid weights, no input/source hash forging.
- No CUDA/driver/source/device graph out-of-bounds/ragged/mismatched kernel admitted because case metadata says "present."
- No temporary buffer freed after Future drop if physical work may have started. Captured but not yet submitted source readers stay live.
- No duplicate state publisher or resource wallet; one session/root authority and process physical ledger.
- No accidental multi-device reset or gfx90c launch in test run.
- No HTML/HTTP 200 response with invented output from unsupported model or server stub.
- No unbounded network/client input or unrestricted external serve default; loopback/auth limits preserved.
- No silent quant/approximate model replacement; numerical and task quality independently validated.
- No "all current features pass" from source-only fixtures when requested physical model/driver gate is unavailable.

## 7. Closeout artifact expected from implementation agent

Report:
- list of signed structural and functional commit SHAs; each with changed owner and fast Tier L/M result;
- final Cargo dependency DAG, public SDK/serving/CLI imports and no legacy shadow engines;
- source codec/precision/derived model support table with separate recognized/decode/CPU/GPU/model/quality status;
- exact current feature parity matrix with CLI and standalone Rust SDK usage;
- one Q0 acceptance report, Q1 bugfix list/commits, Q2 final complete gate and explicit unrun/failed physical work;
- benchmark notes with actual request shapes/profile/driver/source and tests that are not performance proof;
- remaining future phase tasks, long-context/model/media/distributed/codec gaps and blockers.

The design-writing turn itself runs no Tier H and makes no code commits.

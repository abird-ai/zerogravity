# Fabric vNext transition — architecture and continuation

**Current source/host checkpoint:** `fef7123` — public SDK config/graph policy,
current feature/source ownership audit and final isolated host gate passed.
Original `860ce38` source preparation and earlier engine/numerical/catalog/serving
owners remain. GPU/trained acceptance is now explicitly authorized. CUDA/Gemma/
HIP/cache and final host gate pass; CUDA repair signed `8d761c4`. Finite27B
natural/shared-HIP and final signed reconciliation remain in progress. Whole Fabric stays open; Plan2 preserves Plan1 and deferred
Qwen/context/C1..C64/vLLM work. No acquisition/gfx90c/recovery/recurrence restart.

**Canonical current entrypoint:** `.agents/docs/CURRENT_HANDOFF.md`. Current source/tests and the four-file planning spine outrank historical source snapshots in this document.

**Created:** 27 October 2026. **Current status: authorized trained/physical acceptance; final Q2 open.**
**Repo:** /home/pvl/spaces/abird/src/zerogravity; connector **dotlink-l5** (host pvl-l5), not dotlink-a1.
**Source observation during this design pass:** HEAD fadf2b1123309a2118725f2cc005d3edc7ea39f2 at 06:50 UTC; 41 commits after prior deep-review baseline 195099ca, including codec split, model/CPU/GPU source refactors and recent fresh CUDA prefill source/oracle. Run current HEAD/status again before any implementation.
**This handoff's central mandate:** first write complete compilable Rust architecture skeleton with both new engine/serving crates and exported public library, then move ALL current functionality in tested logical signed commits, then one full integrated heavy qualification and targeted bugfixes plus final gate. No old pre-v1 API compatibility obligation. The historical planning turn alone did not authorize code work; the latest owner instruction now does.

## 1. All context is preserved in the following order

Mandatory current implementation sources:
1. [README.md](README.md) — current architecture, exact user directions, three-tranche implementation.
2. [DECISIONS_RISKS_AND_ACCEPTANCE.md](DECISIONS_RISKS_AND_ACCEPTANCE.md) — settled decisions, source-specific risks, safety gates.
3. [CURRENT_SOURCE_RECONCILIATION.md](CURRENT_SOURCE_RECONCILIATION.md) — 41-commit actual source delta, formats/GDN/CLI/server/core and bounded test evidence.
4. [ARCHITECTURE_AND_CRATE_CONTRACTS.md](ARCHITECTURE_AND_CRATE_CONTRACTS.md) — final Rust tree, Cargo DAG, SemanticProgram/ExecutableBundle, one resource/root authority, public SDK, independent serving.
5. [QUANTIZATION_AND_NUMERICS.md](QUANTIZATION_AND_NUMERICS.md) — stored codec/scalar vs activation vs source math vs physical vs derived loss, exact per-scheme status.
6. [STRUCTURAL_SKELETON.md](STRUCTURAL_SKELETON.md) — immediate first Rust skeleton S0 and S1–S3 actual source ownership.
7. [PHASED_IMPLEMENTATION.md](PHASED_IMPLEMENTATION.md) — coherent signed-commit phases S0–S3 / F0–F4 / Q0–Q2 and all exit gates.
8. [FEATURE_PARITY_MATRIX.md](FEATURE_PARITY_MATRIX.md) — all currently working features plus explicit unsupported and held scopes for functional preservation.
9. [QUALIFICATION_AND_COMMIT_POLICY.md](QUALIFICATION_AND_COMMIT_POLICY.md) — Tier L/M cheap tests each commit vs one consolidated heavy Q0 and one final Q2, bugfix Q1.
10. [ROADMAP_AFTER_TRANSITION.md](ROADMAP_AFTER_TRANSITION.md) — next expansion phases model/serving/SSD/codec/media/distribution/perf after transition.
11. [TODO.md](TODO.md) — held per-phase checklist and plan conventions.

Read the maintained earlier Fabric design set as architecture context:
- ../fabric-vnext/README.md
- ../fabric-vnext/PUBLIC_LIB_AND_SERVING.md
- ../fabric-vnext/PRIOR_ART_AND_REVALIDATION.md (full llama.cpp/vLLM source tree and model portability analysis)
- ../fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md
- ../fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md (verbatim user instructions and old decisions)
- ../fabric-vnext/HANDOFF.md (earlier phase status, superseded by newest user request for implementation plan)
- ../fabric-vnext/TODO.md

**Original existing engineering policy remains in:** AGENTS.md, .agents/plans/README.md, .agents/plans/TODO.md, source owners .agents/plans/{pipeline-composition,model-execution,serving-and-speculation,cpu-gpu-optimization,numerics-and-context,memory-and-streaming,model-acquisition,cli-device-routing}/TODO.md, and source-specific normative .agents/docs and P00–P23 roadmap. Do not lose existing accepted/failed/held evidence.

## 2. User decisions to follow without revisiting

- Everything pre-v1 may be broken or replaced for a clean intuitive maintainable architecture. **Do not keep compatibility shims** or two inference runtimes at final state.
- There is ONE semantic AI execution fabric, not a model-specific/quant/device/modality server farm.
- Both **zerogravity-engine** and **zerogravity-serving** should be concrete Rust library boundaries. Serving is independently reusable; it is not a submodule implemented in zerogravity-cli.
- Public **zerogravity** Rust crate is the only ordinary application dependency, exposing all run/load/pull/cache/devices/pipeline/diagnostics/server/stream functionality.
- **zerogravity-cli** is thin: parse argv/config paths/env, display terminal/status/JSON, bridge signal to SDK cancel, invoke public library. No model math/source parser/HTTP implementation/ResourceArbiter/scheduler in CLI.
- **Full architecture visible in compiled code immediately in S0**: crates/modules/interfaces and gated future task/codec/device structures, not just docs. Unimplemented private experimental modules may use todo!(), BUT ordinary supported SDK/server/CLI calls must never panic; delegate current working path or return typed Unsupported.
- Moving code and functional migration go in small coherent signed commits with quick checks; do NOT run long full check.sh/real GPU trained-model/long-context benchmarks after every move.
- When all S/F source transitions are assembled, run **ONE Q0 full qualification**, track regressions as targeted Q1 bugfix commits, then **ONE Q2 final gate**. No blanket future model/format/driver performance claims from source only.
- Preserve original exact source/numerical/state/physical evidence and all currently real user features; metadata/codec/test-case existence is not a capability.
- Historical boundary: creating this transition packet did not authorize implementation. Current implementation authority comes only from the latest user instruction reached through CURRENT_HANDOFF.md.

## 3. Core correction to apply before coding

**Physical ResourceArbiter:** keep one actual process-shared low-level authority in core or an even lower leaf if truly justified; don't move it to engine so GPU/CPU/artifacts import engine or mint another budget. Engine owns source/session/workload policy and reservations *through* original authority.

**State:** core/state TaskCoordinator/StateRoot and core/session_root SessionAuthority/RootBinding/RootAttempt must share **one logical root and publication law**, preserving real GPU physical Ready/fence/quarantine and source-bound state; do not add another publisher to server.

**Semantic:** build on current ForwardPlan / MathematicalBackend and source family forward, no parallel generic IR. Only SemanticProgram and ExecutableBundle endure. Region providers prepare qualified source/math/shape/numerical/physical kernels including specialized whole-model CUDA, not new model-specific engine.

**Formats:** source container, StoredTensorEncoding (scalar vs quant scheme), ActivationEncoding and numerical reduction, PreparedTensorLayout, DerivedModelArtifact and actual QualifiedKernelCandidate are separate. Q1/Q2/Q3/NVFP4 homes are READMEs only; Q4_K scalar decode does not give source CPU matvec; fresh prefill CUDA capture and parameterized GDN backend do not grant general ordinary CUDA.

**Serving:** server consumes engine TaskRequest/CommittedEvent with output credits, auth/HTTP/SSE/openai conformance and graceful shutdown; facade reexports serving; CLI calls facade. A server cannot publish neural state or run its own Qwen/GPU scheduler.

## 4. Historical initial sequence; use current TODO for continuation

Original sequence; current TODO records what is already landed:

1. **Read source/plan ownership:** use dotlink-l5 current HEAD/status/recent commits, AGENTS.md and all linked docs, active TODOs, other workers. No destructive Git operations, no unapproved model/device access, never overwrite other agent uncommitted files. Treat fadf2b1 only as the last *review snapshot*.
2. **Freeze small interface choices** and integrate shared source/type names with actual latest code. Appoint one coordinator for Cargo/runtime/root shared files and separate owners for independent lanes. Respect the repo's signed commits, explicit file ownership and main-only persistent topology.
3. **S0 first:** create full Rust workspace/module skeleton in a SINGLE logical commit with actual types/traits, engine/serving crates, public facade, CLI target, task/media/codec/placement contracts, static checker. Fast build/feature/architecture/unit/refusal checks; no expensive whole-model/HIP/CUDA GPU runs. User must SEE full architecture immediately in source.
4. **S1:** current actual q4/q6/q8 codecs, CPU source arithmetic and GPU quant binding into exact format vs numerical vs physical homes; preserve source identities and original q4_k refusal. Signed per coherent move.
5. **S2:** family/source math and same ForwardPlan, existing CUDA GDN parameterized/fresh-prefill native kernels and source/qualifier isolation into one provider contract. No repeating already merged GPU→Qwen dep extraction; signed source/test logical units.
6. **S3:** converge state/root/publication and resource/physical event protocol. Keep one process authority, no duplicate runtime/arbiter. Signed tests only relevant small root/lease cases.
7. **F0:** lift current CLI/serve/engine and GPU/CPU model-specific runtime into one model-neutral engine, preserve same-source alias/private roots/fair queue/native B2/B4/B8 accepted kernels and physical owners.
8. **F1:** public zerogravity SDK for all existing run/pull/cache/devices/config/pipeline/model admin and status/typed event functionality, with ordinary Rust embed test.
9. **F2:** move real HTTP/OpenAI JSON/SSE/ingress/stream/cancel/shutdown into independently reusable zerogravity-serving, sharing same engine. Test embedded server start/stop and CPU loopback.
10. **F3:** all CLI command handlers delegate public library, delete legacy server and business execution; no compatibility shims.
11. **F4:** reconcile EVERY row of FEATURE_PARITY_MATRIX, close small compile/feature/small source/refusal gaps and remove stubs on current supported paths; no hidden general CUDA model or unsupported codec claims.
12. **Q0:** on assembled stable code, run one authorized complete scripts/check.sh plus full scoped independent source/math/physical/API/SDK/CLI tests as permitted. Record exact passed/failed/ignored/unexecuted and checkpoint/driver/kernel IDs.
13. **Q1:** fix observed regressions in small lowest-layer signed bugfixes with targeted suites; do not repeatedly run long full qualification for each.
14. **Q2:** one final full gate when stable; results and source-support matrix; explicitly mark held model/GPU device scopes, no vLLM parity from source alone. Then future E phases in roadmap, separately triggered.

**Critical safety:** during structural S0 old functioning CLI code can remain temporarily so user features are not destroyed before F2/F3. But the *target architecture must be fully visible and compile*. At F4 there is only ONE live runtime/serving/CLI path and no old compatibility duplication.

## 5. Historical parallel guidance — current work is main-only

Use subagents for independent tasks with **clear module ownership**:
- codec/format source and CPU reference (S1);
- GPU code/ABI, quantum/native GDN/qualification separation (S1/S2);
- model family/program builder (S2);
- canonical state/resource/event (S3);
- engine/catalog/scheduler/session (F0);
- SDK/acquisition/device/config/pipeline (F1);
- serving/HTTP/SSE/lifecycle and CLI (F2/F3).
Only after S0 completes and shared interfaces are agreed. Avoid separate teams simultaneously creating independent RootBinding, ModelRegistration, ResourceArbiter or HTTP message types. One coordinator reviews source, integrates and signs commits. No other owner files touched without actual authorization.

Progress must be monotonic: no missing code tracked as passing, no uncommitted massive patch handed off as complete. Per phase build/lint/checker/targeted test only. Heavy GPU/bench jobs wait Q0. Stop early on broken source math, resource/physical safety, invalid state publication or architecture DAG even during skeleton.

## 6. Current implementation-launch method

Do not use the retired embedded implementation prompt from the 27-Oct design
window. Start at `.agents/docs/CURRENT_HANDOFF.md`, re-read current `main`, then
read this transition packet in the order in README.md.

Only if the user's latest instruction explicitly authorizes Fabric implementation:

1. revalidate source/ownership/current capability matrix;
2. continue the current TODO; do not repeat the accepted skeleton/foundation;
3. continue S1–S3 and F0–F4 in logical signed units;
4. run Q0 once assembled, fix bounded regressions in Q1, then run Q2;
5. preserve current source/numerical/state/resource/refusal semantics and do not
   mint support from historical capture or design metadata.

The canonical current handoff owns the copy/paste starting prompt. Historical
prompt text is retained only in Git history.

## 7. Current handoff/report method

This file is a maintained architecture handoff inside the vNext plan, not the global cross-window entrypoint. Start a new agent at `.agents/docs/CURRENT_HANDOFF.md`; it points to the four maintained planning/status files, this transition set and the earlier vNext architecture set. Git history retains retired snapshots. Do not recreate a single-file archive or documentation SHA catalogue. When handing off, print the ready-to-paste starting prompt in chat; the user controls further authorization; no implicit hardware campaign restart.

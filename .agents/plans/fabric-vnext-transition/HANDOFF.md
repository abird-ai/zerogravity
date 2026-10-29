# Fabric vNext transition — COMPLETE next-agent implementation handoff

**Created:** 27 October 2026. **STATUS: PLAN/HANDOFF COMPLETE; SOURCE IMPLEMENTATION NOT STARTED.**
**Repo:** /home/pvl/spaces/abird/src/zerogravity; connector **dotlink-l5** (host pvl-l5), not dotlink-a1.
**Source observation during this design pass:** HEAD fadf2b1123309a2118725f2cc005d3edc7ea39f2 at 06:50 UTC; 41 commits after prior deep-review baseline 195099ca, including codec split, model/CPU/GPU source refactors and recent fresh CUDA prefill source/oracle. Run current HEAD/status again before any implementation.
**This handoff's central mandate:** first write complete compilable Rust architecture skeleton with both new engine/serving crates and exported public library, then move ALL current functionality in tested logical signed commits, then one full integrated heavy qualification and targeted bugfixes plus final gate. No old pre-v1 API compatibility obligation. This phase writing documents alone does not authorize code work.

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

Read **every** original frozen complete-design source as well:
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
- No implementation, Git commit or push is requested in THIS plan-writing conversation.

## 3. Core correction to apply before coding

**Physical ResourceArbiter:** keep one actual process-shared low-level authority in core or an even lower leaf if truly justified; don't move it to engine so GPU/CPU/artifacts import engine or mint another budget. Engine owns source/session/workload policy and reservations *through* original authority.

**State:** core/state TaskCoordinator/StateRoot and core/session_root SessionAuthority/RootBinding/RootAttempt must share **one logical root and publication law**, preserving real GPU physical Ready/fence/quarantine and source-bound state; do not add another publisher to server.

**Semantic:** build on current ForwardPlan / MathematicalBackend and source family forward, no parallel generic IR. Only SemanticProgram and ExecutableBundle endure. Region providers prepare qualified source/math/shape/numerical/physical kernels including specialized whole-model CUDA, not new model-specific engine.

**Formats:** source container, StoredTensorEncoding (scalar vs quant scheme), ActivationEncoding and numerical reduction, PreparedTensorLayout, DerivedModelArtifact and actual QualifiedKernelCandidate are separate. Q1/Q2/Q3/NVFP4 homes are READMEs only; Q4_K scalar decode does not give source CPU matvec; fresh prefill CUDA capture and parameterized GDN backend do not grant general ordinary CUDA.

**Serving:** server consumes engine TaskRequest/CommittedEvent with output credits, auth/HTTP/SSE/openai conformance and graceful shutdown; facade reexports serving; CLI calls facade. A server cannot publish neural state or run its own Qwen/GPU scheduler.

## 4. First implementation sequence, exact action

Only after a **new explicit user implementation instruction**:

1. **Read source/plan ownership:** use dotlink-l5 current HEAD/status/recent commits, AGENTS.md and all linked docs, active TODOs, other workers. No destructive Git operations, no unapproved model/device access, never overwrite other agent uncommitted files. Treat fadf2b1 only as the last *review snapshot*.
2. **Freeze small interface choices** and integrate shared source/type names with actual latest code. Appoint one coordinator for Cargo/runtime/root shared files and separate owners for independent lanes. Respect repo's signed commits/isolated checkout rules.
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

## 5. Parallel agent guidance

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

## 6. Ready-to-paste NEXT AGENT PROMPT — implementation after approval

~~~text
Use the dotlink-l5 connector at /home/pvl/spaces/abird/src/zerogravity.
We have approved a full pre-v1 Fabric vNext architectural transition.
Everything is available for breaking changes if it results in a cleaner,
more maintainable, more composable and faster Rust execution fabric.

FIRST READ THE ENTIRE HANDOFF, not just README:
.agents/plans/fabric-vnext-transition/HANDOFF.md
.agents/plans/fabric-vnext-transition/README.md
.agents/plans/fabric-vnext-transition/DECISIONS_RISKS_AND_ACCEPTANCE.md
.agents/plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md
.agents/plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md
.agents/plans/fabric-vnext-transition/QUANTIZATION_AND_NUMERICS.md
.agents/plans/fabric-vnext-transition/STRUCTURAL_SKELETON.md
.agents/plans/fabric-vnext-transition/PHASED_IMPLEMENTATION.md
.agents/plans/fabric-vnext-transition/FEATURE_PARITY_MATRIX.md
.agents/plans/fabric-vnext-transition/QUALIFICATION_AND_COMMIT_POLICY.md
.agents/plans/fabric-vnext-transition/ROADMAP_AFTER_TRANSITION.md
.agents/plans/fabric-vnext-transition/TODO.md

ALSO READ the full frozen previous design:
.agents/plans/fabric-vnext/README.md
.agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md
.agents/plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md
.agents/plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md
.agents/plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md
.agents/plans/fabric-vnext/HANDOFF.md
and the combined comprehensive reference under .agents/docs.
Then read AGENTS.md, .agents/plans/README.md, .agents/plans/TODO.md
and active owner plans / current source tests.

DO NOT trust historic HEAD/path line numbers blindly. The last
read-only review observed fadf2b1 after 41 new commits since 195099ca.
Revalidate live HEAD, working tree, ongoing agents and current Cargo
graph before any source edits. Preserve others' work and source evidence.

The critical latest user mandate:
1. Make the ENTIRE target architecture VISIBLE immediately in compiled
   Rust code in the FIRST structural milestone S0. Add full crate/module
   structure, zerogravity-engine and standalone zerogravity-serving libraries,
   exported public zerogravity SDK, thin CLI target, semantic/codec/numeric/
   model/task/provider/placement/state/resource contracts and type-safe
   future capabilities. Private feature-gated todo! stubs are okay, but
   ordinary supported API/CLI/HTTP paths MUST NOT panic; return typed
   Unsupported for genuinely unimplemented future features.
2. Next S1-S3 move the ACTUAL current source/codec/CPU/GPU/model/state/
   qualification implementations into their final owners. Reuse the recent
   codec/precision split, GPU->Qwen dependency removal, CPU GDN math and
   parameterized CUDA GDN/fresh-prefill source; do not redo completed work
   or mistake finite captured kernels for general production.
3. Next F0-F4 lift real model-neutral session/scheduler/router/runtime,
   all model acquisition/cache/device/config/pipeline work and HTTP/SSE
   from CLI into engine, public zerogravity SDK and independent serving lib.
   zerogravity-cli must own ONLY args/env/UX/process signals and use
   zerogravity public APIs for all project functionality. Delete legacy
   pre-v1 compatibility/shadow runtimes. Preserve every current real
   feature, cancellation, source and physical safety/refusal behavior.
4. Keep one process-shared low-level ResourceArbiter in core (engine uses it,
   no reverse backend->engine dependencies), one logical TaskCoordinator/
   SessionAuthority StateRoot publish protocol, one original ForwardPlan-
   based SemanticProgram and one ExecutableBundle.
5. Format layers are distinct: stored scalar/quant scheme, source decoder,
   temporary activation encoding, numerical/reduction contract, reversible
   device layout, optional lossy DerivedArtifact, qualified physical kernel.
   Q1/2/3/NVFP4 folders are only placeholders; Q4_K source CPU matvec
   explicitly refuses. Do not grant unsupported execution.
6. Do this in LOGICAL UNIT SIGNED COMMITS. Every S/F phase gets cheap
   cargo format/check/clippy, architecture dependency and targeted small
   relevant unit/source/refusal tests. DO NOT run repeated heavyweight
   all-model/full GPU/check.sh/performance suites after each source move.
   Once the large structural and functional migration is assembled, do ONE
   integrated Q0 qualification; record failures, fix in Q1 lowest-layer
   bugfix commits with focused tests; then one final Q2 full gate.
   Keep any unavailable physical/model/hardware scope explicitly held,
   not passed; never bypass source/device safety or launch held gfx90c.
7. After Q2, future high-performance serving/paged state/SSD/27B/Gemma/
   more quant formats/sparse/MTP/multimodal/distributed milestones are
   separately planned in ROADMAP_AFTER_TRANSITION.md, not prerequisites
   for rehoming current working functionality.

Begin with S0 immediately after revalidating the current source and
ownership. Use subagents for independent substantial source lanes where
supported, with explicit ownership and one integration coordinator.
Do not add a second model engine, allocator, router per model or session
publisher. No dynamic compiler/token hot-path model registry/HTTP JSON.
No false support/performance claims. Signed coherent commits only,
source/tests reviewed. No push, tag, destructive reset/rebase, firmware
changes, model downloads or unsafe remote/device access without the
user's separately stated permissions.

Deliver every phase's commit SHA, exact changed owners, focused tests,
unsupported/held cases, final source/device/codec matrix and the
consolidated Q0/Q1/Q2 acceptance report. Start by making the FULL
architecture skeleton visible and compiling, then transition working
implementations quickly.
~~~

**This prompt is meant for the agent that has received explicit user authorization to implement. The current conversation is design/docs only and must finish after writing/verification, without starting S0 itself.**

## 7. Completion/report to user for THIS writing task

Report creation and verification of new sibling transition plan and complete single-file reference including original frozen context. State that source, active plans, build/test/physical usage and Git commits were not changed. Provide path of the handoff and display the ready-to-paste prompt in the response, then stop. User/next agent controls actual S0 start.

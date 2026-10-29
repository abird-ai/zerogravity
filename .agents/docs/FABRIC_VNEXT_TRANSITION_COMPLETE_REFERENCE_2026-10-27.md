# zerogravity Fabric vNext — COMPLETE IMPLEMENTATION TRANSITION AND ORIGINAL ARCHITECTURE REFERENCE

**Frozen planning/handoff archive: 27 October 2026; SOURCE CODE UNCHANGED.**

Contains the **complete verbatim contents** of all twelve new Fabric vNext transition plan files (first, authoritative for latest phase instructions) and all seven previous frozen Fabric vNext architecture/research/user-decision/serving files (after, original historical context). Section delimiters identify source; Markdown content within each section is reproduced word-for-word from the indicated file. No automatic implementation, source qualification, signed commit or push is implied. The old original single-file archive also remains intact.

**User's latest instructions:** all pre-v1 internals may break; first make entire architecture visible as compilable Rust skeleton, including independent zerogravity-engine and zerogravity-serving libraries, one fully exported zerogravity Rust SDK and a truly thin zerogravity-cli; then move all current source/codec/quant/numeric/model/GPU/state/HTTP implementations into the new layers in small tested signed commits; restore all current supported behavior; only after big migration run a consolidated full qualification, then targeted bugfixes and a final full gate. Current doc-writing turn must NOT start the source implementation.

**Last refreshed source observation:** dotlink-l5, /home/pvl/spaces/abird/src/zerogravity, fadf2b1123309a2118725f2cc005d3edc7ea39f2 (moving live repo, 27 Oct 2026). Previous original source review used 195099ca and earlier vNext source refresh used eb38dd5; physical/model/driver conclusions are time-scoped, not generally certified.

**Start with FILE: .agents/plans/fabric-vnext-transition/HANDOFF.md immediately below** to obtain agent instructions; do not stop after the first overview. User/implementation approval must be separately provided for Rust/source edits.

## Contents in this complete reference

- .agents/plans/fabric-vnext-transition/HANDOFF.md
- .agents/plans/fabric-vnext-transition/README.md
- .agents/plans/fabric-vnext-transition/DECISIONS_RISKS_AND_ACCEPTANCE.md
- .agents/plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md
- .agents/plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md
- .agents/plans/fabric-vnext-transition/QUANTIZATION_AND_NUMERICS.md
- .agents/plans/fabric-vnext-transition/STRUCTURAL_SKELETON.md
- .agents/plans/fabric-vnext-transition/PHASED_IMPLEMENTATION.md
- .agents/plans/fabric-vnext-transition/FEATURE_PARITY_MATRIX.md
- .agents/plans/fabric-vnext-transition/QUALIFICATION_AND_COMMIT_POLICY.md
- .agents/plans/fabric-vnext-transition/ROADMAP_AFTER_TRANSITION.md
- .agents/plans/fabric-vnext-transition/TODO.md
- .agents/plans/fabric-vnext/HANDOFF.md
- .agents/plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md
- .agents/plans/fabric-vnext/README.md
- .agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md
- .agents/plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md
- .agents/plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md
- .agents/plans/fabric-vnext/TODO.md


---

## BEGIN FILE: .agents/plans/fabric-vnext-transition/HANDOFF.md

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
- ../../docs/FABRIC_VNEXT_COMPLETE_REFERENCE_2026-10-27.md (complete old frozen single-file copy).

The combined NEW complete reference under .agents/docs/FABRIC_VNEXT_TRANSITION_COMPLETE_REFERENCE_2026-10-27.md copies all original frozen seven files AND all current transition files word for word (excluding self-referential duplicate text where noted by index). Read it if one context file is more convenient. The original frozen archive is still preserved separately; current implementation phase ordering in this file is newer and wins where plans conflict.

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
2. **Freeze small interface choices** and integrate shared source/type names with actual latest code. Appoint one coordinator for Cargo/runtime/root shared files and separate owners for independent lanes. Respect repo's signed commits/isolated worktree rules.
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

## END FILE: .agents/plans/fabric-vnext-transition/HANDOFF.md


---

## BEGIN FILE: .agents/plans/fabric-vnext-transition/README.md

Author: Project owner
Co-authors: GPT-6 (architecture research, 26–27 October 2026)
Owner: @pvl; later appointed implementation coordinator

# zerogravity Fabric vNext — transition architecture and implementation plan

**Created:** 27 October 2026. **STATUS: DESIGN/HANDOFF ONLY — no Rust implementation started.**
**Read-only source observation:** HEAD fadf2b1123309a2118725f2cc005d3edc7ea39f2 (l5, 27 Oct 06:50 UTC). The 26 Oct deep review had 195099ca; preceding refresh had eb38dd5. These are evidence timestamps, not frozen ongoing source. Recheck HEAD before coding.
**Physical root:** /home/pvl/spaces/abird/src/zerogravity via dotlink-l5.
**Side-plan policy:** Preserve existing frozen .agents/plans/fabric-vnext, existing user/agent code, active .agents/plans/TODO.md and other owner plans. This is a sibling plan, not a silent queue promotion.

## 0. Current user's binding implementation-order instruction

The user wants the fully revised architecture **made visible immediately in real compilable Rust code**, then current implementations *moved into it*, then all existing supported features functioning and correct, finally one consolidated full qualification and bounded bugfix rounds. Internal pre-v1 compatibility is not required. Build, commit and run **small focused checks for each logical phase**, but avoid repeated costly full model/physical/release suites until the large migration is assembled.

Exact latest user's critical language:

> "first set of phases are for the reorganizing the current code, in this new structure, move it all, have the full structural layers ready to go right away, with todo! impl in rust, where needed.. but the core goal is we have the full architecture visible immediately in code.."

> "then, once the architecture is visbile, then focus on the current code fitting into this architecture, then all of the current feature set working and correct in this new model with refactors fully completed. no need to preseve any compat changes, so just break it and redo where needed freely"

> "ensure to do qualification and verification, once we are done with the large changes in one go.. we can consider them as bugfixes in the large plan so that we don't repeatedly run qualification or such tests that take a long time before we finish the transition and skeleton, architecture and engine design"

The previous user binding decisions remain:
- Serving **must** be a reusable, independently compilable **zerogravity-serving** library, not buried in CLI.
- The public **zerogravity** Rust crate exports *all* user-facing functionality including server construction/lifecycle, model loading, pull/cache, run, routing, pipelines, device selection, configuration, metrics and errors.
- **zerogravity-cli** owns parsing, CLI UX/presentation, process/signal bridging only; calls the public zerogravity library for all actual work.
- Everything pre-v1 may be broken/restructured for maintainable and fast architecture; model/source/quality/physical safety and real tests must be kept.
- This turn authorizes **writing plan documents and handoff only**, NOT modifying Rust/code/build history.

## 1. Read the original whole-context archive

The preceding full architecture reference remains frozen:
- ../fabric-vnext/README.md — comprehensive semantic compiler + model/task/provider/resource/state/multimodal/distributed design.
- ../fabric-vnext/PRIOR_ART_AND_REVALIDATION.md — in-depth llama.cpp and vLLM file trees, model extensibility, vLLM Model Runner V2 and Omni patterns, detailed zerogravity v4/v5 docs crosswalk.
- ../fabric-vnext/PUBLIC_LIB_AND_SERVING.md — new independent serving library + complete exported Rust SDK.
- ../fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md — initial migration, user journeys, correctness/performance/physical gates.
- ../fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md — literal user instructions and design choices.
- ../fabric-vnext/HANDOFF.md — original context and read order.
- ../fabric-vnext/TODO.md — historical held work.
- ../../docs/FABRIC_VNEXT_COMPLETE_REFERENCE_2026-10-27.md — full single-file original snapshot (~198KB).

**Do not silently replace this historical material.** This transition pack supplements it with the second read-only current-code refresh, improved phase ordering, precise codec split, root/resource correction, complete feature inventory and actionable engineering commit sequence. A combined complete reference will reproduce both packs verbatim, so an LLM can load everything in one file.

## 2. Project's core product and architecture, retained

Build **one native Rust semantic AI execution fabric**. It can drive source-verified Qwen, Gemma, dense/MoE/sparse/recurrent families; CPU plus CUDA/HIP/Vulkan and future Metal/NPU/AOT/fabric/remote; GGUF/safetensors and explicit derived quant; single or many models; requests, stages and model assemblies for text, embeddings, ASR, TTS, image, video, realtime AV; high-throughput continuous batching, paged hybrid-state/prefix caching, speculative target verification, source-qualified whole-layer/whole-model fusion and global physical resource planning across VRAM/UMA/RAM/SSD.

**Performance is an evidence-based Pareto goal**, not a claim of universal vLLM superiority. A model name or scalar codec does not prove trained model operation. No imported unknown HF executable model code; no hidden source/quant changes; no second runtime per model/quant/device/modality; no user/agent tool workflow DSL within device stages; no per-token global resource lock or compiler search; no fake Ready.

**Exactly two enduring execution representations:**
1. **SemanticProgram:** actual source mathematical meaning, typed tensor/media shapes, numerical arithmetic, effect dependencies and logical model/task boundaries. Build from the existing real ForwardPlan and family math, never parallel independent IR interpretations.
2. **ExecutableBundle:** prepared kernels, closed fused regions, graphs/AOT/remote programs, guarded numeric/source/shape candidates, resource+command+copy DAG, state bridge and fallback; bind request epochs/slots/leases late, no per-token compilation or model-name dispatch. RIR/KIR are transient compiler views, CIR is within bundle.

**Five primary authorities:** model+source semantics; compiler and qualified region implementation; TaskCoordinator single session logical publisher; shared physical ResourceArbiter budget; physical Device/IO/RemoteWorker real completion and lifetime.

**Task-aware engine:** one global admission, fairness, model routing and physical authority; task-stage-specific schedulable work for text tokens, encoders, acoustic windows/codebooks, denoise/image/video, structured decisions. ModelAssembly vs mathematical region vs multi-model TaskPipeline are distinct; typed values/media clocks prevent text-token fake semantics.

**State:** all required KV/SWA/MLA/SSM/GDN/indexer/grammar/RNG/draft/media versions must prepare against one committed root and actual Ready events; the sole publisher swaps one RootBinding after checking epoch/root/frontier. Prefix reuse is a common restorable frontier across ALL participating state domains, not KV-only. Logical committed output != transport delivered != acknowledged != heard/played.

**Physical memory:** original ResourceArbiter is process-shared in core; do NOT move its low-level authority up into an engine crate that GPU/artifacts would then need to depend on. Engine owns *policies* while physical ledger stays below it. Device-local pools use prepaid envelopes; account aliases/UMA once, retain encoded/captured/queued readers, preserve unknown completion quarantine and correct real readiness. Four independent placement maps (compute, parameters, mutable state and reusable caches) plus derived TransferPlan form one physical plan.

## 3. Most important current-refactor adaptation

The normal/build GPU→Qwen dependency was already REMOVED in commit 263ff79; the reference planned extraction but it is now done. Avoid redoing large model/backend relocation. Existing format/precision directory split and CPU shared GDN convolution also landed. The new work is largely higher-level:
- Finish semantic/evidence separation of production kernels vs finite source qualification and updated CUDA GDN/fresh-prefill implementations.
- Consolidate distinct CPU TaskCoordinator/StateRoot and GPU SessionAuthority/RootBinding **semantics** in one canonical protocol without losing actual physical owners.
- Formalize explicit **StoredTensorEncoding, ScalarDType, QuantScheme, ActivationEncoding, NumericalContract, PreparedTensorLayout, DerivedModelArtifact, QualifiedKernelCandidate**. These are logical roles, not mandatory individual Rust traits/crates.
- Keep existing source codec homes; Q1/Q2/Q3/NVFP4 placeholder readmes have no executable codec; Q4_K scalar decode exists but original CPU source matvec is deliberately unsupported. No false support.
- Extract model-neutral Engine and dedicated Serving from CLI/application; public facade becomes capable high-level SDK.
- Keep all existing supported paths after migration and delete obsolete compatibility/internal duplicate branches. Distinguish full working features from metadata scaffolds and future planned work.

**Latest snapshot:** 41 commits from prior 195099ca baseline (39 through eb38dd5 plus ba7fb03 and fadf2b1). Recent final two touched fresh CUDA prefill generated capture and reference layout; treat as qualification-in-progress. At archive creation initial source tree had eight crates (core, artifacts, cpu, gpu, Qwen, Gemma, facade, CLI) plus gpu-qualify tool; two new structural crates intended: engine and serving. Existing source API is not general production HF CUDA, despite finite capture paths.

## 4. Implementation tranche contract

**Tranche A — full architecture in code before migration detail:**
- S0: **first single structural skeleton commit** creates engine and serving library crates, the full target module/package hierarchy in core/artifacts/model/cpu/gpu/facade/CLI, SDK APIs and typed contracts, feature wiring, new boundary tests and safe stubs. It MUST compile. Unimplemented backend operations are feature-gated private placeholders or explicit Unsupported, NEVER reachable todo! panics. This is the visible architecture milestone the user explicitly requests.
- S1–S3: move exact source/codec/precision code, model+GPU provider/qualification source, and resource/state/effect owner implementations into the new homes. Delete old pre-v1 APIs/callers in the same coherent units; no compatibility aliases except narrow temporary staging within an uncommitted unit. Commit each phase after tiny checks.

**Tranche B — current functionality on new structure:**
- F0 engine: move CLI/app source-specific model/session/request work to one prepared model interface, one source-bounded router and scheduling.
- F1 SDK: expose **all** current model, direct inference, acquisition/cache/device/config/pipeline/status operations through zerogravity public library.
- F2 serving: move real HTTP/SSE/OpenAI, bounded ingress, worker shutdown, CPU+GPU paths to standalone zerogravity-serving that uses same engine; optional API generations must be honest.
- F3 CLI: arguments/presentation only and remove old duplicate engines/serve/pull wrappers. No compatibility requirement.
- F4 source/test parity: assert all previously implemented behavior and deliberate Unsupported/ignored states are represented by new path, with focused checks and inexpensive behavior tests only.

**Tranche C — one full qualification, bugfix and final gate:**
- Q0: one consolidated assembled scripts/check.sh and applicable source/model/GPU/serving gates (actual hardware only when authorized/available); capture true missing/failed scopes.
- Q1: fix discovered regressions in dedicated small signed logical bugfix commits with focused tests; do not rerun long full suites per fix.
- Q2: final complete check and independent source/state/physical benchmarks/evidence where permitted; publish accepted/unsupported/held matrix and new baseline. If long device tests are impossible, do not mark their capabilities qualified; document precise blocker.

**After transition:** separate future model/batching/codec/SSD/multimodal/distributed feature phases, not prerequisites to moving current functionality.

## 5. Priority and dependencies

The first target source structure and model-neutral contracts are the main early deliverable. Avoid a giant one-shot code rewrite: **one first full scaffold, then small logical end-to-end responsibility moves**. Each phase has a named owner and should be a coherent signed commit, with a cheap check. Parallel lanes are allowed after S0 establishes interface names and contract tests: (i) source codecs and model adapters, (ii) physical GPU/qualification, (iii) engine/session/resources, (iv) library/API/serving/CLI. Shared files/active owners have serialized integration; source authority and project AGENTS.md still apply.

**Conflict policy:** status of upstream/competing live agents outranks the historical audit. Recheck live HEAD and in-progress work just before any implementation. This design phase changes nothing but new reference documents. The user may later say "begin transition"; the next agent should use the prompt in HANDOFF.md, not invoke broad code writes from an older snapshot.

## 6. Read the detailed actionable documents in this order

1. [HANDOFF.md](HANDOFF.md) — full context and copyable implementation prompt, immediate operating guard.
2. [CURRENT_SOURCE_RECONCILIATION.md](CURRENT_SOURCE_RECONCILIATION.md) — code/commit delta with full source status and concrete paths.
3. [ARCHITECTURE_AND_CRATE_CONTRACTS.md](ARCHITECTURE_AND_CRATE_CONTRACTS.md) — target complete crate/module tree, typed APIs and dependency graph.
4. [QUANTIZATION_AND_NUMERICS.md](QUANTIZATION_AND_NUMERICS.md) — detailed storage/codec/activation/calibration/qualification matrix.
5. [STRUCTURAL_SKELETON.md](STRUCTURAL_SKELETON.md) — **first compiled whole architecture**, safe todo! usage and detailed path moves.
6. [PHASED_IMPLEMENTATION.md](PHASED_IMPLEMENTATION.md) — each commit-sized phase S0–S3, F0–F4, Q0–Q2 with outcomes, tests, dependencies and boundaries.
7. [FEATURE_PARITY_MATRIX.md](FEATURE_PARITY_MATRIX.md) — preserve all current working behaviors and deliberate non-support.
8. [QUALIFICATION_AND_COMMIT_POLICY.md](QUALIFICATION_AND_COMMIT_POLICY.md) — cheap checks per phase, one full post-migration qualification, bugfix strategy, signed Git policy.
9. [ROADMAP_AFTER_TRANSITION.md](ROADMAP_AFTER_TRANSITION.md) — later feature phases and critical math/hardware targets.
10. [DECISIONS_RISKS_AND_ACCEPTANCE.md](DECISIONS_RISKS_AND_ACCEPTANCE.md) — exact commitments, rejected designs, risk/mitigation and stop gates.
11. [TODO.md](TODO.md) — held checklist in plan-manager convention.

A combined one-file archive will include all seven historical frozen design files plus all transition documents verbatim; both historical and updated architecture and source-level comparisons are preserved for other agents.

## 7. Minimum completion claims and guardrails

**Structural skeleton complete** does NOT mean runnable model support is new; it means all modules and crates exist and are compile-visible, dependencies enforce ownership, public API exists, and deferred operations fail closed.

**Migration complete** means all previously implemented supported behaviors, source/codec/CPU/GPU/CLI/serving/fixtures and their exact scoped safety are working under the new architecture, with no duplicate legacy runtime, no reachable todo! on a supported path, no missing current command or library equivalent.

**Physically qualified transition complete** requires the consolidated evidence and actual hardware/source tests for the relevant previously qualified tuples. Historical proof may guide expected results but cannot be retagged to a changed binary/toolchain.

No agent may assert current model/device/endpoint support from a bare trait, source codec, feature, captured test case, old report or attractive architecture diagram. No commits, pushes, tags, hardware probing/reset or untrusted source downloads merely from writing this plan.

## END FILE: .agents/plans/fabric-vnext-transition/README.md


---

## BEGIN FILE: .agents/plans/fabric-vnext-transition/DECISIONS_RISKS_AND_ACCEPTANCE.md

# Fabric vNext transition — settled decisions, risks, unresolved choices and acceptance law

This is an implementation-focused decision register made 27 October 2026. It **supplements** the fully preserved historical user/architecture decision ledger at ../fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md and the exact latest user instructions in this plan's README.md. It does not supersede source truth or active owner plans.

## 1. Explicitly binding user decisions

- User fully permits pre-v1 breaking of Rust internal/public APIs, crates, module names, implementations and old artifacts to achieve cleaner scalable maintainable design. **No compatibility shims** required at final transition. Correct source/math/device/state/lifecycle and existing supported user operations must survive.
- The architecture must be a **single reusable AI execution fabric**, not cloned inference engines per model/backend/format/modality; vLLM and llama.cpp lessons are incorporated but neither imported wholesale.
- **Dedicated independently usable zerogravity-serving Rust library** owns HTTP/OpenAI/Responses/SSE and future realtime protocols, with no serving implementation buried in CLI.
- **All functionality exported through one public zerogravity Rust library** for arbitrary external programs: model/load/inspect/pull/cache, devices, run, sessions, pipeline, diagnostics, server start/stop, eventually modalities/distributed. CLI only flags, terminal and presentation/signals.
- The **complete target architecture must be visible immediately as actual compiled Rust code** in the FIRST structural source milestone, with typed TODO stubs where necessary. File tree alone is not enough.
- Then physically move current source implementations into this structure, make current functionally supported feature set work and stay correct, delete old paths, commit in logical units as implementation proceeds.
- Each phase is committed and tested with cheap targeted checks. **Do not repeatedly run heavyweight model/GPU/long-context/release qualification before the large structural + functional transition is complete.** Run one consolidated complete qualification afterward; follow with specific bugfix commits and targeted tests; one final full gate when stable.
- User's **current request is to write detailed plan and handoff only**, not actually implement/refactor/commit Rust now. The ready-to-paste successor prompt will direct a future explicitly approved agent how to begin.

## 2. Accepted technical design refinements after 41-commit source review

| Decision | Reason and implementation rule |
| --- | --- |
| Add zerogravity-engine as a real model-neutral library | Extract model/session scheduler/router and task/physical preparation from CLI/facade; engine does not import model family or GPU directly |
| Add zerogravity-serving as independent library | Protocol and socket lifecycle are reusable; server→engine/core, facade→server, CLI→facade, no cycle |
| Existing physical ResourceArbiter stays low-level | CPU/GPU/artifacts need real lease/physical identity; moving arbiter wholesale to engine would create a reverse dependency, duplicate wallets |
| CPU TaskCoordinator and GPU SessionAuthority converge logically | One RootAttempt/prepared domains/physical Ready/PublishRoot protocol; preserve original physical signoffs, no second publisher |
| Keep existing ForwardPlan as foundation | One canonical SemanticProgram rather than a new unproved IR+interpreter |
| Only SemanticProgram and ExecutableBundle enduring | RIR/KIR transient, CIR embedded physical commands; no graph interpretation per token |
| Model families own source math | Qwen/Gemma config/source roles/traversal/state/specialization; GPU implements physically reusable kernels/ABI only |
| Existing GPU→Qwen dependency removal is already DONE | Commit 263ff79 and checker; don't waste time recreating that refactor |
| Quant/codec system is not one QuantKind | Container vs stored scalar/quant vs activation encoding vs numerical behavior vs packed layout vs derived lossy checkpoint vs physical qualification |
| Q4_K scalar decode is not CPU source-operator support | Preserve explicit source dot refusal until independently qualified |
| Q1/Q2/Q3/NVFP4 current dirs are placeholders | Keep capability false and typed Unsupported despite visible code homes |
| GPU CUDA GDN parameterized kernel implemented source ≠ ordinary model capable | Model source + backend exact implementation + numeric/ABI/shape proofs and physical Ready required, finite captures remain diagnostic |
| Resource and task schedulers are separate levels | One global fair admission policy, stage-specific work units; no separate capacity ledger or token-only scheduler for all media |
| Complete original functionality before future features | Existing GGUF/CPU/OpenAI/GPU finite/codec/acquisition/cache/device/pipeline checker semantics restored; new codecs/media/distributed are later |
| Heavy qualification once after major changes | S/F Tier L/M quick source checks and signed commits, Q0 merged broad gate, Q1 focused bugfix, Q2 final full gate |
| Source/user rights and hardware holds remain binding | No dangerous gfx90c/clock reset/unknown source download/remote peers from design or skeleton |

## 3. Major architectural rejection decisions

**Reject**:
- Separate per-model/per-format/per-API engine or "legacy worker" kept in parallel just to avoid breaking pre-v1 imports.
- A full runtime scheduler/ResourceArbiter in CLI or serving crate.
- Moving low-level ResourceArbiter *implementation* upwards without solving CPU/GPU/artifact dependency cycles; duplicating pools instead.
- A fake general ModelRunner that repeatedly calls a model-specific short-HF CUDA fixed schedule under the hood while claiming dynamic production.
- Generic Q1/Q2/Q3/Q4/NVFP4 codec recognition based on bits-per-weight without exact released format and numerical operators.
- Treating BF16 numerical precision and quantized storage as the same thing, or lossless source pack and lossy derived artifact as one checkpoint.
- Treating model checkpoint metadata/registry YAML/typed port schema as trained runtime capability.
- Reusing a passing finite GDN CUDA captured test to authorize arbitrary fresh 48–50/context/source GPU calls.
- Full SemanticProgram rewrite that duplicates current ForwardPlan or adds four durable compiler IRs without a proven use.
- CPU/GPU per-op host readback or generic stage serialization when qualified device-resident handoff is possible.
- Broad autograd/training/framework hierarchy, agent workflow DSL, dynamic executable plugins before real vertical slices.
- Running full scripts/check.sh and long GPU model/reference suites after every moved file.
- Using todo!() to deliver panics from supported SDK/HTTP/CLI call paths.
- Treating all generated .cu source as an architecture flaw that must be rewritten; generated source is fine with correct ABI/code provenance.
- Publishing/pushing/releasing based on a planning request alone.

## 4. Risks that can derail a fast transition

| Risk | Severity | Mandatory mitigation |
| --- | --- | --- |
| Skeleton becomes non-functional second runtime | High | It is typed structure; existing routes may remain temporarily but no new claims; F0–F4 fully migrate/delete old |
| Two ResourceArbiters give false available memory | Critical | Process-shared core authority; engine policies reserve original ledger; alias and UMA accounts once |
| CPU and GPU each publish separate StateRoot semantics | Critical | Canonical root/epoch/frontier/physical Ready gate and one publisher |
| Model math changes under a crate rename | Critical | Preserve original source, cast/reduction, exact operator fixtures; Q0 actual model proof |
| CUDA captures/encoded buffers freed early | Critical | Real physical lease retirement from encoding/queue/capture through device Ready, unknown quarantine |
| New codec enum falsely advertises unsupported Q4_K/Q1/Q2/NVFP4 | Critical | Separate decode/operator/backend/whole-model evidence, Unsupported-first |
| vLLM performance goals lost in generic abstraction | High | Prepared executable/row slots, no per-token name/JSON/plan/compiler/global lock, measured E02 |
| Serving facade cycle / CLI still owns HTTP | High | Enforced acyclic Cargo DAG and public Rust embed tests |
| Acquisition and security edges lost in CLI extraction | High | Full pull S3/GCS/Azure/HF/Xet/OCI/cache tests, source/privacy rights |
| New stub panic in supported runtime | Critical | Explicit Unsupported errors or delegate current path; unreachable todo only in internal gate |
| Broad full suites repeatedly block transition | Medium-high | Tier L/M focused checks per commit, Q0 one main run, Q1 fixes, Q2 final |
| Heavy qualification delayed hides obvious broken import or source guard | High | Every phase compile+small actual invariant tests; do not commit broken source safety |
| Historical binary evidence retagged after code image changes | Critical | Preserve source/driver/model fingerprints, fresh physical qualification when allowed |
| Live source moved by other agents while plan executed | High | Recheck HEAD/active TODO/owner, isolated worktrees, one merge coordinator |
| gfx90c/NPU safety hold bypassed by "full qualification" script | Critical | Review hardware access, no unsafe/unauthorized device launches or system tuning |
| Published UI claims all planned models/formats run | High | Capability matrix per source/quant/task/shape/backend/numerical/quality |
| One giant unreviewable commit | High | Single full structural S0 commit, subsequent smallest coherent logical signed commits |

## 5. Deliberately unresolved implementation details

Do not freeze illustrative Rust names if source suggests cleaner pre-v1 implementation:
- exact Runtime/Server/PreparedModel/ExecutionProvider public naming and sync/async method return types;
- whether all new structural top-level modules are physically separate files or grouped files, as long as ownership clear and compiled;
- whether resource identity warrants lower-level dedicated zerogravity-resource crate vs existing core module; default KEEP core to avoid unnecessary new package;
- whether a broad RegionExecutionProvider trait is required vs static prepared enum and narrow common contract; avoid trait zoo;
- how much of fixed-source CUDA dynamic parameterization can generalize without numerical/ABI changes; finite captures remain diagnostic;
- existing pipeline YAML v1/v2 schema changes allowed pre-v1 if old useful validation preserved; no forced backward syntax compat;
- whether serving HTTP uses current httparse-based implementation or a dedicated async HTTP library; reuse existing robustness until measured reason to replace;
- exact CLI flags/config precedence and error presentation; user functionality preserved but legacy spelling not binding;
- how to represent multiple Runtime handles over one process-global physical ResourceArbiter and preserve source/tenant visibility; avoid duplicated budgets;
- which short Tier M suite is inexpensive enough per phase; use measured build cost/availability and hard safety constraints;
- precise real hardware/driver/source access for Q0/Q2; source/user restrictions and previous holds remain;
- whether all proposed task/media/remote typed skeleton modules warrant public exports in S0 or private typed API; structure must remain visible and refuse;
- use of future workflows/provider apps or dynamic plugin ABI; not initial structural deliverable.

Any deviation from mandatory latest user decisions needs explicit agreement. These open choices may be resolved by the successor based on current Rust source and measured maintenance/perf/correctness evidence.

## 6. Phase acceptance law

**S0 Architecture-visible acceptance:**
- Two new crates engine/serving, public SDK and CLI target shape compile with correct Cargo DAG and strict source-only tests.
- All semantic, codec/numerical, session, physical, model/TaskDriver, placement, public API, server and CLI roles visible as real Rust modules.
- Future stubs Unsupported or private todo; no fake model/device/task support.
- Only cheap checks, one signed coherent commit; NOT declaring previous features migrated.

**S3 Structural relocation acceptance:**
- Source exact codec/CPU/GPU and model/forward semantics in correct owners; physical provider/qualification separate; one root/resource protocol.
- Exact source/quant/shape/numeric and known device GPU decoupling preserved; no duplicate source math.
- Each phase signed, source-only tests and regression/refusal checks, long model/driver suite deferred.

**F4 Functional migration acceptance:**
- All currently implemented features in FEATURE_PARITY_MATRIX functioning via engine, public zerogravity SDK, independent serving and thin CLI, or truthfully held at originally unsupported/unsafe scope.
- No old shadow runtime, no permanent compatibility alias tree, no missing CLI-to-lib operation, no supported-path todo panic.
- Focused checks only; ready for Q0.

**Q2 Final transition acceptance:**
- One Q0 broad gate performed, bugfixes per Q1 targeted/tests/commits, one final complete Q2 gate on stable source; safety/quality/physical and actual hardware/driver scopes separately recorded.
- Previously qualified exact model/source/quant/task/backend ranges independently verified or explicitly marked unexecuted/blocked, not retagged from old binary.
- No performance claims absent comparable matched whole-request tests; future E-series capabilities remain clearly separate.

## 7. Planning/documentation control

- New transition side plan is independent of .agents/plans/TODO.md active owner queue until a coordinator explicitly promotes it during authorized implementation.
- Existing older frozen ../fabric-vnext files and full archive remain unchanged; successor must read them as historical architecture/research, with user-requested updated phase ordering here.
- At actual code handoff, verify current source HEAD, AGENTS.md, active TODO plans, potential concurrent agent worktrees and partial runs. Do not automatically push/tag or include other agent files.
- This turn writes documents only. The copyable next-agent implementation prompt is in HANDOFF.md and provided to the user in chat.

## END FILE: .agents/plans/fabric-vnext-transition/DECISIONS_RISKS_AND_ACCEPTANCE.md


---

## BEGIN FILE: .agents/plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md

# Fabric vNext transition — current-source and 41-commit reconciliation

**Read-only baseline:** dotlink-l5, /home/pvl/spaces/abird/src/zerogravity, HEAD fadf2b1123309a2118725f2cc005d3edc7ea39f2, observed 2026-10-27 06:50 UTC. The read-only Python architecture checker passed at this HEAD. No full Cargo suites, trained-model test, GPU workload, full benchmark or physical qualification run was performed for this design.

## 1. Baseline and commit lineage

The original file-tree/revalidation research inspected HEAD 195099ca583ace766d075682927f590360c823a5 with a then-large moving uncommitted worktree. The first vNext source refresh inspected eb38dd5e892f2120ac29cc2fb79b2dbe137a37fe, **39 commits** after 195099ca, 1,220 files touched, 157,086 inserted and 103,429 deleted lines (includes moves, generated source and handoffs). Two later commits ba7fb03 and fadf2b1 bring the observed gap to **41 commits**. Earlier whole-repository history contained approximately 289 commits; do not mistake the last 41 for all project work.

| Commit | Concrete change | Fabric transition consequence |
| --- | --- | --- |
| 405a3e6 | Exact source codec and CPU/GPU quantization homes; BF16 precision separate | Retain format split, do not mint support from empty README |
| d9a8ecf | Name actual prefill/input/cache request geometry | Use shape/data, not fixture or ordinal names |
| fa844a6 | Family/model and backend ownership instructions | Enforce actual source/ABI/control boundaries |
| a5ed7de | Qwen/Gemma source/config/math moved into family owners | Do not redo already completed extraction |
| c90b8cd | GPU provider/native prepared bindings grouped | Reuse real physical GPU work, source evidence and lifecycle |
| fc73457 | Application model routes composed | Move orchestration up to model-neutral engine later |
| fe17625 | Four-tap GDN scalar math consolidated in CPU | Preserve tested arithmetic, no new duplicate code |
| 263ff79 | GPU→Qwen normal/build/optional model dependency removed; tests use dev dependency | Already satisfies the central Cargo boundary |
| 50abaf7 | HTTP worker join and shutdown ownership | Preserve in serving extraction |
| cd4159d / 5a18b5b | Native submitted commands and captured source leases retained | No premature free on cancellation/stop |
| 61f095c / d74f59a | Historical finite Qwen CUDA prompt and recipe fixtures | Diagnostics, not general production routing |
| 3612ecb | Production vs test/qualification code rules in AGENTS.md | Keep compiled production independent of fixture whitelist |
| 1f08e55, a1c51f9, 91a2958 | Fixed rotary preparation/capture separated and source identity guarded | Retain exact ABI and read-only source leases |
| 9ad9aac, d087012 | Parameterized CUDA GDN Conv and correct image/code identity | Reuse specialized kernel body and guard |
| 05baab5 | Prompt source semantics separated from qualification | Valid source geometry cannot be faked from testcase |
| 0c63d89, ae936a8, 7a910d9, ca0d2bb, f61a3ca | Qwen CUDA finite qualification separate, declarative implementation selection | Distinct source permission and backend kernel eligibility |
| 85b2724, 1fe6151, 2e3bbec | Prefill GDN stages parameterized | Preserve KKT/solve/recurrent/output variant contract |
| 1b898ff, c0bdbb1 | Sealed explicit kernel combinations, no recipe bridge in execution | Don't reintroduce test-case dispatch in production |
| eb38dd5 | Fresh-48 output variant | Not proof of general dynamic shape/model inference |
| ba7fb03 | Added fresh-prefill capture and generated source | Retain qualification source hashes/compiler/lease semantics |
| fadf2b1 | Fresh-prefill reference layout/oracle fixture corrections | Latest code snapshot; real hardware/model gate remains separate |

## 2. Actual current Cargo workspace, dependencies, ownership

~~~text
crates/
  zerogravity-core/
  zerogravity-artifacts/
  zerogravity-cpu/
  zerogravity-model-qwen35/
  zerogravity-model-gemma4/
  zerogravity-gpu/
  zerogravity/              public facade but not yet complete high-level SDK
  zerogravity-cli/          owns HTTP/service work + run/acquisition/pipeline CLI
tools/gpu-qualify/
~~~

The proposed zerogravity-engine and zerogravity-serving crates DO NOT EXIST at the snapshot. The user explicitly wants them and all other target structural layers visible in code in the first architectural skeleton commit.

Normal/build dependencies:
- core imports no model or backend.
- artifacts -> core.
- CPU -> core+artifacts.
- GPU -> core+artifacts, optional CPU; Qwen source fixtures use dev-dependency only.
- Qwen model -> core+artifacts, optional CPU.
- Gemma model -> core, optional artifacts/CPU/GPU for specific preparation/qualification.
- facade -> core+artifacts, optional CPU/GPU/families.
- CLI -> facade plus CLI-specific crates. It still owns core application behaviors even though it only depends on facade by name.
The original scripts/check-architecture.py enforces currently accepted normal/build/optional/target boundaries and passed at fadf2b1. Extend it in skeleton S0 for engine/serving/facade/CLI, not a new checker with a second policy.

Historic approximate source counts at refresh: GPU ~590 Rust files/~310,274 LoC (~120,414 generated source); artifacts 57/~32,268; Qwen 95/~36,517; Gemma 24/~6,677; CPU 43/~11,774; core 43/~19,383; facade 27/~5,924; CLI 46/~26,407. Includes test/qualification sources. These are indicators of refactor ownership, not numerical/model capability.

## 3. Existing portions that should be reused

### Exact semantic model and source

- core/src/accelerator/program.rs contains real ForwardPlan, PlanOperation, typed SourceOperator, state effects, value/parameter references, explicit source/precision and MathematicalBackend. This is the foundation for future canonical SemanticProgram, not a new redundant IR.
- model-qwen35/src/forward.rs performs single family mathematical traversal; plan/ and source/ own source input geometry, config/tensor interpretation and variants.
- model-gemma4/src/{config,tensors,text,execution}/ contains source/math/PLE/attention/MoE and candidate source integration; no unrelated Qwen inference authorization.
- core/src/pipeline.rs contains structural typed pipelines/ports; pipeline run is explicitly NOT IMPLEMENTED.
- core/src/targets.rs contains metadata-only model targets; binding family slug does not certify model execution.

### Source formats, scalar and physical computation

- core/src/quant.rs contains the old QuantKind representation with exact GGML GGUF codes and byte length: F32/F16/BF16, Q4_0, Q4_1, Q4_K, Q6_K, Q8_0. This broad enum mixes scalar and quantized storage identity and should be refined into two typed variants without weakening errors.
- artifacts/src/quant.rs and quant/q4.rs, quant/q6.rs, quant/q8.rs perform actual scalar decoding (including Q4_K); Q1/Q2/Q3/NVFP4 directories are only README placeholders, not Rust bodies. BF16 currently has real scalar precision conversion in shared source, but new precision/bf16 folders mainly house documentation.
- CPU source source.rs/projection.rs owns true source-specific activation quantization and GGML reduction (Q4_0 x Q8_0, Q4_1 x Q8_1, Q6_K x Q8_K, Q8_0 x Q8_0). Q4_K full source matvec explicitly refuses because the required original GGUF Q4_K x Q8_K physical dot has not passed independent qualification.
- GPU quantization.rs and quantization/{activation,activation_fused,projection,batch_projection,q4_0,q4_1,q6_k,q8_0} plus backend CUDA/HIP/Vulkan use actual packed kernels and physical queues/ABI, separately qualified by backend. No general Q1/Q2/Q3/NVFP4 computation granted.
- New CUDA GDN includes explicit TextGdnImplementation and GdnKernelConfig for KKT/Recompute/H/Output, source-bound geometries and retained code-image hashes; actual source/ABI/shape must be proven before general source promotion.

### State/resources and real completion

- core/resource_arbiter.rs: one process-shared physical ResourceArbiter, GlobalResourceLease, PhysicalDeviceKey, allocation classes. This is the existing low-level owner and stays below the engine.
- core/state.rs: reference TaskCoordinator/StateRoot/PreparedDomain/CommittedOutput; core/session_root.rs: SessionAuthority/RootAttempt/RootBinding. They need one logical root/epoch/publication protocol, preserving original test and real physical safety.
- GPU resident_program.rs, resident_inference.rs, resident_leases.rs, physical_service.rs, physical_queue.rs have accepted physical work, one selected StateRoot publication, ready witnesses, image/code ownership, unknown/cancel quarantine and finite multi-row programs. Do not discard in favor of a hypothetical executor.
- Device source and activation data may remain GPU-resident over legal stages; source reader leases must be held from encoding/capture through actual final completion.

### High-level still needing extraction

- zerogravity/src/lib.rs is mainly module/backend reexports with some model-specific Qwen CUDA adapters under qwen35_cuda.rs. It is not yet a complete public Runtime/Server/ModelStore API.
- CLI/serve/{mod.rs,engine.rs,gpu_engine.rs,gpu_batch.rs,pending.rs,http.rs} implements CPU/GPU HTTP routes, Qwen-only model/tokenizer fields, bounded queues/workers, real B2/B4/B8 native request cohorts, JSON/SSE/output credits and graceful worker shutdown.
- CLI/pull.rs + pull_azure/s3/gcs/object/modelscope/xet/oci etc. implements high-level provider selection and UX despite many transport primitives already living in artifacts/acquisition. Make library programmatically useful without changing source rights.
- CLI/pipeline.rs does bounded YAML v1/v2 parse and source-neutral graph check, but actual pipeline run refuses. Move library logic up without implying native execution.
- CLI/run_cli/source, execute, devices, config, qwen35_cuda contain model source/fit/selection and preflight business behavior; move reusable policy to facade/engine while retaining argument/env/printing in CLI.
- The serving API is a qualified subset only, not full OpenAI Responses/embedding/tool/multimodal support.

## 4. Historical outcomes and explicitly open claims

The current plan records include prior check.sh CPU/library/CUDA source/test passes and real finite GPU runs with specific model/driver/certificates, e.g. 263ff79 full 1,914 Rust tests with 87 explicit ignored cases and no new physical model run. That does not qualify fresh latest code or all models. Other plans contain current scoped B2/B4/B8 GPU API cases, historical C1/C4/C8 wins and regressions, and long-context gaps. They remain grounded historical evidence and must not be relabeled "all performance done."

Current truthful gaps:
- Q1/Q2/Q3/NVFP4: document homes, no real codec/CPU/GPU implementations.
- Q4_K: scalar decoder exists; source-exact CPU matvec explicitly Unsupported; no general trained model execution permit follows.
- Original HF CUDA: finite captured source diagnostic permits only; ordinary dynamic Qwen CUDA explicitly rejects unsupported GDN/source shapes. Fresh48 kernel source/capture/oracle updates are insufficient to promote.
- Qwen27B: sharded/source windows and finite CPU/HIP/qualification; not arbitrary trained whole-model, SSD optimized serving.
- Gemma4: family source, E2B etc. reference and finite HIP/Vulkan candidates; not generalized all model variants/tasks.
- Paged complete-domain KV/GDN/MLA prefix state, generic production chunked temporal prefill, unrestricted vLLM continuous goodput parity: not finished.
- Native image/video/audio/ASR/TTS/TTS-codec and peer-distributed inference: design/scaffolding only or separately bounded provider experiments.
- CPU GGUF serving and selected Qwen CUDA fused HTTP routes: valuable existing qualified subsets, preserve and revalidate.
- Source/parser/device/data/HTTP refusal and physical lifecycle tests: no waiver during transition.

## 5. Concrete old-to-new owner crosswalk

| Current source | Final owner/responsibility | Must preserve |
| --- | --- | --- |
| core/accelerator/program.rs | core/semantic/program and typed contracts | one plan, exact arithmetic/effects |
| core/quant.rs | stored encoding and typed scalar/quant schemes | byte length, GGML IDs, unknown refusal |
| artifacts/quant.rs & q4/q6/q8 | artifacts/codecs + source encoding | exact scalar decoder and malicious input guards |
| CPU source.rs/projection | CPU source-exact numerical operators | activation quant/reduction and reference |
| GPU quantization/ | GPU packed kernels and ABI | physical/kernel tests, per-backend qualifiers |
| Qwen source/plan/forward | family semantics and qualified recipes | original trained math/source maps |
| Gemma config/tensors/execution | family semantic assembly/candidate | true PLE/MoE/softcap/attention |
| GPU CUDA GDN/text and generated cases | reusable CUDA implementations + qualification-tool oracle | actual code/ABI/testcase provenance |
| core/resource_arbiter | shared low-level physical ownership | single process ledger, aliasing, real leases |
| core/state and session_root | canonical state protocol + optional engine coordination | one publisher per session, full domain version |
| GPU resident_program/inference/leases | device executor/physical service, compiled bundles | actual Ready/cancel/quarantine |
| facade prepare/qwen35_cuda/model | application model registry and public runtime adapters | no false production CUDA permit |
| CLI serve engine/gpu_engine/pending | engine model-neutral scheduling/admission; ingress-specific part serving | independent sessions, fairness, native cohorts |
| CLI serve http/protocols | separate zerogravity-serving | OpenAI JSON/SSE, error mapping, shutdown |
| CLI run/pull/cache/devices/pipeline | public library/engine/artifacts and truly thin CLI | same functionality and deliberate refusals |

## 6. Boundaries for successor agents

The source is actively changing and the latest observed model/GPU evidence can shift between turns. At future code start:
- Read HEAD/status/active worktrees, all .agents/plans owner TODO updates and AGENTS.md.
- Preserve signed commit history, retained source hashes, source model/driver/kernel code IDs and explicit negative tests.
- Don't rerun already completed low-level dependency extraction just to match a new filename; move responsibility only where still genuinely misplaced.
- Never execute held gfx90c GPU or unsafe system power tuning; model downloads and remote peers need separate authorization.
- No source code changes or Git commits are part of writing this plan. The structural-code phases require a later explicit implementation instruction.

## END FILE: .agents/plans/fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md


---

## BEGIN FILE: .agents/plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md

# Fabric vNext transition — complete target crate/module architecture and contracts

**Status:** target architecture for first structural source milestone. Source baseline fadf2b1. All APIs below are illustrative; full Rust names remain pre-v1 breakable. This design incorporates the recent codec and GPU refactors, proper resource ownership and library-first serving.

## 1. Enforce the real Cargo dependency graph

~~~text
                 zerogravity-cli
               args / UX / signals
                       |
                       v
                  zerogravity                  PUBLIC RUST SDK
            /         |         \
           /          |          \
  model-family    zerogravity-engine     zerogravity-serving
   Qwen/Gemma      model-neutral           protocol-neutral to models
      |                |                          |
      |                +----------+---------------+
      |                           |
      v                           v
  zerogravity-core <----- zerogravity-artifacts
      ^         ^                 ^
      |         |                 |
      +--- CPU/GPU providers -----+

Facade zerogravity is the ONLY composition root:
facade links model families and physical providers and registers them
into a cold model-neutral engine; facade reexports serving library.
Engine does not import Qwen/Gemma or physical CUDA/CPU implementations.
Serving depends on engine/core, NOT facade or CLI.
CLI depends on facade for ALL project functionality; no internal runtime.
~~~

**Important correction from the prior archive:** The live low-level process-shared `ResourceArbiter` stays in `zerogravity-core` (or a lower-level dedicated physical-resource crate ONLY if demonstrated necessary), since artifacts, CPU and GPU depend on it. Engine owns model/session admission and scheduling policies, using the same existing arbiter. Never create an independently spendable memory ledger in engine, serving or a model worker.

Current normal/build GPU→model-family dependency was REMOVED in commit 263ff79. Dev-only Qwen fixtures can remain outside runtime. Do not repeat that refactor. Extend `scripts/check-architecture.py` to enforce NEW engine/serving/facade/CLI layers and ensure opt-in GPU/acquisition/server feature isolation. Do not create another checker that disagrees.

## 2. Complete visible target Rust tree for S0

This is an ownership template, not an instruction to manufacture 100 empty modules. S0 creates all named *top-level responsibility modules and usable typed contracts*; add nested files when moved working implementation justifies them.

~~~text
crates/
  zerogravity-core/src/
    identity/        versioned Artifact/Model/Semantic/Task/Root/Bundle keys
    encoding/        ScalarDType, QuantScheme, StoredTensorEncoding
    numerics/        ActivationEncoding, NumericalContract, exact/approx profile
    semantic/
      program.rs     canonical SemanticProgram based on ForwardPlan
      op.rs          exact typed mathematical operators + source parameters
      effects.rs     StateEffect and dependency legality
      region.rs      closed replaceable semantic regions
      values.rs      shapes, tensor and media value descriptors
    model/           ModelDefinition, VerifiedModelAssembly, components
    task/            TaskDriver, StageIntent, typed TaskFrontier and media clocks
    state/           shared state domains + one root/version/attempt contract
    provider/        RegionExecutionProvider/Capability/PreparedExecutable contracts
    physical/        ORIGINAL ResourceArbiter, DeviceEvent, queue/lease interfaces
    placement/       compute/parameters/state/cache + derived transfers, topology
    errors.rs        typed Unsupported/Invalid/Capacity/DeviceFailure/Cancelled

  zerogravity-artifacts/src/
    containers/      GGUF/safetensors/indexed shard and bounded source parsing
    codecs/
      scalar/        F32, F16, BF16 decode and exact conversion
      quant/         existing exact q4/q6/q8 decoding, future q1/q2/q3/nvfp4
    source/          immutable SourceWeights/SourceTensorRef/verified windows
    acquisition/     HF, GitHub, Xet, OCI, S3, GCS, Azure, ModelScope, HTTPS
    store/           transactional model cache/snapshots/resolved manifests
    derived/         optional calibrated derived quantization artifact schemas

  zerogravity-cpu/src/
    reference/       CPU reference semantics and independent trained oracles
    operators/       matvec, embedding, attention, GDN, RMS, exact source math
    quantization/    source quant plus dynamic Q8 activation/reduction
    precision/       BF16/F16 numerical execution; ISA variants
    backend/         CPU region provider and local execution resources

  zerogravity-model-qwen35/src/
    config.rs        real variants/source geometry
    tensors.rs       roles, ties, validated source shapes
    source/          Qwen GGUF/HF immutable binding
    forward.rs       ONE source-exact mathematical traversal
    state.rs         GDN/KV/conv/hybrid state schema and effects
    tokenize/        tokenizer and chat/template
    plan/            record/evaluate Canonical SemanticProgram
    specialization/  Qwen source/shape/kernel eligibility recipes
    reference/       family oracle / regression (not production fixture logic)

  zerogravity-model-gemma4/src/
    config/ tensors/ source/ forward/ state/ tokenize/
    plan/ specialization/ reference/
    text/            retained where it groups real Gemma responsibilities

  zerogravity-gpu/src/
    backend/
      cuda/          driver compiler, native CUDA kernel images, ABI
      hip/           independent ROCm/HIP physical backend
      vulkan/        independent Vulkan physical backend
    kernels/         shared projection, attention, recurrence, MoE, other ops
    quantization/    EXISTING activation, q4_0/q4_1/q6_k/q8_0 and batch kernels
    precision/       exact selected BF16/compute contracts
    program/         prepared physical plan and closed region/graph compilation
    memory/          actual physical backing, leases, capture readers/pools
    queue/           actual device submission/event/fence/retirement
    transfer/        verified source H2D/D2H/peer/copy/IO readiness
    qualification/   test-only source/capture/fixturing (not live runtime permit)

  zerogravity-engine/src/          NEW model- and backend-neutral crate
    runtime.rs       shared logical runtime lifecycle/service
    catalog.rs       exact registered model/assembly/capability identities
    router.rs        name/alias/replica to source/Task binding
    model_instance.rs immutable loaded model and model-neutral session adapter
    prepare.rs       SemanticProgram -> ExecutableBundle from qualified candidates
    executable.rs    immutable physical prepared region/control plan
    scheduler/
      admission.rs   model/session/workload/tenant budgets using one arbiter
      dispatch.rs    persistent slots, queued ready work, device command binding
      policy.rs      task-local token/audio/encoder/diffusion work strategies
    task.rs          TaskDriver stage control, not workflow DSL
    session.rs       TaskCoordinator logical sole publisher; RootAttempt protocol
    state_cache.rs   domain-specific state pool, common-restorable-prefix frontier
    residency.rs     shared model/adapter/draft/expert/cache physical policy
    placement.rs     one physical plan with four placements and transfers
    request.rs       TaskRequest ingress validation and exact model binding
    output.rs        source-valid committed events, bounded output credits
    diagnostics.rs   effective profile, capacity, status/evidence and metrics

  zerogravity-serving/src/         NEW independent reusable library
    server.rs        builder, lifecycle, listener, stop/drain/readiness
    http.rs          HTTP/connection/framing
    ingress.rs       body/auth/rate/tenant/request limits
    openai.rs        models/completions/chat and qualified other endpoints
    responses.rs     separate typed Responses schema (future until implemented)
    stream.rs        SSE/JSON and bounded client egress
    auth.rs          loopback default, external-bind rights, scoped tokens
    metrics.rs       serving statistics and shared engine diagnostics
    admin.rs         restricted service management (future)
    realtime.rs      WebSocket/WebRTC/media/ACK/playback (future)
    tests/           official-client protocol and safety checks

  zerogravity/src/                 PUBLIC RUST SDK facade
    lib.rs           curated API; must not become giant implementation
    runtime.rs       RuntimeBuilder / Runtime
    models.rs        load/list/inspect/unload/register/source aliases
    artifacts.rs     pull/acquire/cache/source manifest
    run.rs           direct TaskRequest and typed output streams
    devices.rs       inventory, fit, explicit placement, dry-run
    pipelines.rs     bounded YAML/config and typed check/run
    serving.rs       public reexports/wrapper for zerogravity-serving
    diagnostics.rs   status/support matrix/resource usage/typed errors
    builtins.rs      ONE cold registration for compiled-in model/providers

  zerogravity-cli/src/
    main.rs          binary entrypoint
    args.rs          arguments, XDG/env/config-path precedence
    commands.rs      invoke only public zerogravity APIs
    output.rs        terminal/progress/JSON human presentation
    signals.rs       ctrl-C to public cancellation; no GPU/HTTP mechanics

tools/gpu-qualify/                independent actual hardware/known source tests
tests/                            end-to-end library/embed/CLI/server parity
~~~

Move existing sources and callers into these owners in S1–F3, avoiding duplicate ABI namespaces/long-lived wrappers. Source file names, module splits and trait shape are changeable without compatibility. Skeleton must be compile-visible immediately; unreachable/feature-gated pending modules are allowed, not advertised as functional.

## 3. Five authority domains

1. **Model authority:** verified source, family mathematics and state schema. May record one SemanticProgram but not choose model-specific HTTP queue or publish user root.
2. **Compilation authority:** valid closed region, selected candidate, shape/numerical/source/ABI and physical command planning; stores prepared executable bundle. Does not own live session.
3. **Logical session authority:** one TaskCoordinator/RootAttempt/StateRoot publication for each TaskSession or authorized shard. Device Ready is not logical committed output.
4. **Shared physical resource authority:** one actual process ledger and coarse reservations, device local pools under caps, backing aliases/OS+UMA counted once, error/unknown retention.
5. **Physical worker authority:** actual driver/image/upload/capture/IO/device/peer work, Ready/Failed/Unknown event/fence/quarantine and final reader release. Cannot publish user task root.

Router, task driver, provider registry and serving are *clients* of these authorities, not additional competing model execution engines.

## 4. Two enduring representations and exact extension boundaries

**SemanticProgram** is the exact source-verified typed model math: operator/schema revisions, immutable source operands, tensors/axes/dynamic extents, state effects/positions, explicit numerical rules, output/frontier schema. Build it using the existing real ForwardPlan. Reference CPU evaluator and optimized region recorder must share source math. New model with existing operators should touch family and cold registry/tests only. A novel semantic op adds genuine reference/numeric/state contract and optional backend implementations; no new serving engine.

**ExecutableBundle** is the immutable, prepared physical implementation for a model/task/shape/capability band: source+kernel+compiler+numeric guards, region executable/kernels/graph, physical backing/readers, transfers/events, optional fallback decomposition, parameter slots and prepared copy/readiness. Per-session root/slot/buffer generations bind late. Keep only RIR/KIR transient compiler views; CIR command/lifetime graph is within bundle. Compile once/prepare bounded variants outside per-token hot loop; no generic recursive interpreter.

Provider may prepare one op, fused layer, whole model step, AOT/static hardware stage or qualified multi-device region. Its contract covers all visible inputs/outputs and mutable effects. A fast model-specific fused CUDA code image is a legal provider candidate, not a new inference engine. Opaque external hosted text API is a separate explicitly named TaskServiceAdapter (future).

Illustrative coarse interfaces (not frozen ABI):

~~~rust
pub trait ModelDefinition: Send + Sync {
    fn identity(&self) -> &ModelIdentity;
    fn tasks(&self) -> TaskCapabilities;
    fn program(&self, task: &TaskSpec) -> Result<SemanticProgram>;
}

pub trait RegionExecutionProvider: Send + Sync {
    fn capabilities(&self) -> ProviderCapabilities;
    fn candidates(&self, closed: &ClosedRegion, ctx: &CompileContext)
        -> Result<Vec<RegionCandidate>>;
    fn prepare(&self, region: &ClosedRegion, candidate: &RegionCandidate)
        -> Result<PreparedExecutable>;
}

pub trait PreparedModel: Send + Sync {
    fn capabilities(&self) -> &ExecutionCapabilities;
    fn prepare_session(&self, task: &TaskRequest) -> Result<PreparedSession>;
}

// An ephemeral already-prepared step, NOT a compiler IR:
pub struct WorkEnvelope {
    pub session: SessionId,
    pub expected_root: RootBinding,
    pub execution: ExecutableId,
    pub slots: RowBindings,
    pub leases: LeaseBatchId,
    pub dependencies: EventSet,
}
~~~

Use small concrete implementations and enum/static dispatch if these real interfaces do not need distinct dynamic trait objects. No trait per tensor kernel, module, allocator slab or codec. A qualifying fixture schema is not a runtime permit.

## 5. Public SDK and reusable server interaction

~~~text
Rust app -> zerogravity::RuntimeBuilder -> Runtime
                               | -> models: pull/load/list/inspect/alias/unload
                               | -> run(TaskRequest): committed events/cancel
                               | -> devices: enumerate/explain/plan
                               | -> pipelines: check/compile/run when supported
                               | -> cache/snapshots/status/metrics
                               | -> serving::ServerBuilder(runtime)
                                               |
                                      zerogravity-serving
                                       OpenAI/SSE/HTTP/etc.
                                               |
                                    same zerogravity-engine
                                               |
                                    same ResourceArbiter,
                                    same exact model sources,
                                    independent TaskSessions

zerogravity-cli -> ONLY zerogravity public SDK calls,
                   CLI parsing/terminal/progress/exit/signal mapping
~~~

The serving crate **cannot import zerogravity facade** (cycle). It consumes engine/core types/handles and does not contain model family/GPU code. Facade imports/reexports serving, registers model/builtins and exposes coherent public runtime capabilities. No mandatory server process for direct run.

Example intended external Rust API:

~~~rust
use zerogravity::{Runtime, TaskRequest, serving};

async fn embedded() -> zerogravity::Result<()> {
    let runtime = Runtime::builder().build().await?;
    let model = runtime.models().load_local("model.gguf").await?;
    let stream = runtime.run(TaskRequest::text(model, "Hello")).await?;
    // Use committed events; stream drop/cancel cannot free live device work.
    let server = serving::Server::builder(runtime.clone())
        .localhost_default()
        .build()?;
    let handle = server.start().await?;
    handle.shutdown().await?;
    runtime.shutdown().await?;
    Ok(())
}
~~~

Not executable as written until API naming/implementations are selected; user wanted the *shape visible immediately*. Ordinarily callable methods without an implementation must return typed Unsupported/NotImplemented, not todo! panics.

## 6. Stateful runtime, resource, tasks and future placement

- **State:** Begin expected root/epoch -> private provisional state -> real Ready -> domain Prepare and compare -> sole PublishRoot -> typed CommittedEvent. CPU TaskCoordinator and GPU SessionAuthority become one logical protocol. Full required domains KV/SWA/GDN/KDA/SSM/MLA/indexer/draft/grammar/RNG/media; no KV-only false hybrid prefix hit. Prefix restore is greatest common legal checkpoint across domains and source/numeric/tenant compatible.
- **Resources:** existing process-shared ResourceArbiter, coarse model/session/step reservation; device local small pools under prepaid credit, no per-kernel global lock. Physical backing retained from command encoding/capture until last verified consumer retirement, including Unknown quarantine.
- **Scheduling:** ingress/tenant/request credits, model/session/source state admission, device actual work/row/IO budgets. One global policy/admission coordination; distinct stage work units for text/encoder/ASR/TTS/flow/video/decision. Native batching only among compatible same-contract sessions. No cheap multi-model request loop marketed as true fused batching.
- **Physical plan:** four orthogonal placements compute/parameters/mutable state/reusable cache and a derived TransferPlan; local-first, source and driver coherent. SSD requires real async IO readiness and immutable source extent leases. Native NPU/remote/AV remain disabled until separately qualified.
- **Multimodal:** typed Value/TimeBase/Codebook/Codec/Latents/VideoPTS and real TaskDriver; a source-defined multimodal ModelAssembly differs from ASR→LLM→TTS pipeline. Prepared, Committed, Delivered, ACK and Played frontiers remain distinct. No general workflow tools inside kernel runtime.
- **Correctness:** provenance of original model and derived quant source, temporary activation precision, storage layout, numerical equivalence, implementation guarded shapes and per-backend physical test must remain separate. No universal model-support boolean.

## 7. Structural and runtime gates

- S0: full visible module ownership, two new crates, public SDK+serving types, architecture checker, CPU-only/check/default builds, no reachable todo! on an advertised call.
- S1–S3: move exact codecs/source math and physical/state/qualification implementations, delete obsolete source APIs without hidden duplicates.
- F0–F3: one model-neutral engine, library-first all functionality, standalone serving lib, thin CLI.
- F4: feature matrix current supported paths/intentional refusals reconstructed under new imports, cheap relevant tests.
- Q0–Q2: ONE broad merged qualification and staged bugfixes with a final source/physical gate. New model/device/media/SSD expansions follow separately.

Detailed movement ownership and first structural milestone: STRUCTURAL_SKELETON.md. Individual commits and bounded tests: PHASED_IMPLEMENTATION.md. User-requested postponed heavy tests: QUALIFICATION_AND_COMMIT_POLICY.md.

## END FILE: .agents/plans/fabric-vnext-transition/ARCHITECTURE_AND_CRATE_CONTRACTS.md


---

## BEGIN FILE: .agents/plans/fabric-vnext-transition/QUANTIZATION_AND_NUMERICS.md

# Fabric vNext transition — quantization, codecs, numerical equivalence and ownership

**Contract status:** specification for code movement and new clean Rust type boundaries; source code only created after separate user instruction. Informed by commit 405a3e6 format homes, live artifact/CPU/GPU modules, .agents/plans/pipeline-composition/FORMATS.md, and the original complete Fabric vNext/derived-quant design. A directory, parser, source codec or CUDA image is NOT whole-model execution proof.

## 1. The six independent dimensions

Do not collapse quantization into a single `QuantKind` label used to infer everything else.

1. **Source container** — GGUF, safetensors/index shards, NeMo/OCI source assembly, etc.; metadata, configuration, tensors and process/processor provenance. The container is not an inference algorithm or physical kernel.
2. **StoredTensorEncoding** — exact mathematical source tensor byte representation: ScalarDType (F32, F16, BF16) OR exact named QuantScheme such as Q4_0, Q4_1, Q4_K, Q6_K, Q8_0, future NVFP4 or EXL3 trellis/codebooks. Stores block length, bytes/block, endianness, signed codes, scale/zero tables, padding and exceptional bits. Never assume any format from bit width alone.
3. **NumericalContract + ActivationEncoding** — original mathematical operator and per-phase/batch reduction/cast/accumulation/profile, intermediate dynamic Q8_0/Q8_1/Q8_K coding, source-specific rounding, FMA/parallel lane order and nonfinite state. These are not implied by source encoding. The CPU reference can match GGUF's original arithmetic rather than a dequantized-f32 ideal dot.
4. **PreparedTensorLayout** — the *physical representation* after reversible lossless packing: row/column/group/stride, SIMD/tile/WMMA descriptors, device alignment and view lease/generation; a lossless repack changes buffer layout but not weight semantics. If a cast/transform changes math, it is a separately qualified numerical strategy or derived artifact.
5. **Physical QualifiedKernelCandidate** — selected source/semantic/quant/activation/numerical/shape/batch/device/driver/compiler/ABI/capture+state candidate; actual events, buffers/reader liveness, preflight memory and cancellation. A kernel source body existing does not make it production qualified.
6. **DerivedModelArtifact** — a *different* source artifact obtained by lossy weight quantization/calibration, exact source lineage, quantizer version, per-tensor transformation, held-out task quality and licensing. Must never silently replace the user's original BF16/FP8 source or reuse unrelated state/prefix keys.

**Physical tensor placement and IO** is orthogonal to these six: immutable SourceTensorId may materialize in device RAM/UMA/host/SSD/remote, but changes are separately chargeable TransferPlans with verified source event readiness. SourceIdentity for a live original issuer is distinct from content hash; exact matching hashes cannot mint the same source owner, resident pointer or qualified physical permit.

## 2. Proposed exact Rust types (sketch, keep small)

~~~rust
pub enum ScalarDType {
    F32, F16, BF16, // later exact scalar types when sources exist
}

pub enum QuantScheme {
    GgufQ4_0,
    GgufQ4_1,
    GgufQ4_K,
    GgufQ6_K,
    GgufQ8_0,
    // Add only implemented + versioned schemes. Q1/2/3 and NVFP4
    // remain *declared future capabilities*, NOT executable enum variants
    // until exact codec/source validation exists.
}

pub enum StoredTensorEncoding {
    Scalar(ScalarDType),
    Quantized(QuantScheme),
}

pub struct QuantTensorLayout {
    pub logical_shape: Shape,
    pub source_encoding: StoredTensorEncoding,
    pub blocks: BlockGeometry,
    pub scales: ScaleEncoding,
    pub source: AuthenticatedTensor,
}

pub struct NumericalContract {
    pub accumulation: AccumulationPolicy,
    pub activation: ActivationEncoding,
    pub reduction_order: ReductionPolicy,
    pub rounding: RoundingPolicy,
    pub nonfinite: NonfinitePolicy,
    pub semantic_profile: SemanticNumericIdentity,
}

pub struct PreparedTensorLayout {
    pub logical: SourceTensorId,
    pub layout: PhysicalLayoutId,
    pub destination: PhysicalMemoryDomain,
    pub source_owner: SourceOwner,
    pub lease: BufferLease,
    pub ready: EventSet,
}

pub struct DerivedModelArtifact {
    pub original: ArtifactDigest,
    pub derived: ArtifactDigest,
    pub quantizer: QuantizerRecipeIdentity,
    pub calibration: CalibrationProvenance,
    pub quality: QualityReceiptId,
}
~~~

These are conceptual fields/names, not a requirement for a heap object/trait per tensor, nor permission to claim an unsupported format. Store small exact enum/record contracts in shared core, real decoding in artifacts, CPU arithmetic in CPU, GPU kernels and source-specific ABI in GPU, and model role/geometry in family. Model-recipes use generic source and math interfaces; router/HTTP never parses Q4 blocks.

### Support must be multidimensional, not one boolean

~~~text
source: recognized / bytes verified / tensor shape validated
codec: scalar decode / supported activation + operator
backend: source and shape compile / real device Ready / state safe
model: trained source whole-forward / full task quality
serving: scoped request API, output and cancellation
performance: matched complete-request qualified
~~~

For example Q4_K currently has source recognized + scalar decoder, but source-exact CPU matvec deliberately Unsupported and no general trained model eligibility. It remains Unsupported at that level after refactor. A Q6_K GPU implementation on one backend does not qualify an independently different HIP/Vulkan backend or CUDA graph profile.

## 3. Present source implementation map at fadf2b1

| Exact scheme | Artifact bytes/decoder | CPU source arithmetic | GPU source work | Whole-model support |
| --- | --- | --- | --- | --- |
| F32 | Exact original scalar codec | General source projection/reference and ordered lane variants | Backend-selected, source/profile-specific | Conditional actual model/run |
| F16 | Exact scalar codec | Source/transform selected mathematical path | Some physical operations; numerical/ISA dependent | Conditional |
| BF16 | Scalar exact decode; real arithmetic located outside new precision folder | Explicit BF16 numerics and vector paths | Native/CubeCL/CUDA/HIP/Vulkan partial variants, not universal | Source/driver/shape dependent |
| Q4_0 | 32 elements / 18 bytes exact q4 decoder | Original Q4_0 x dynamic Q8_0 and 8-row repack | Existing q4_0 kernel, source qualifications | Qualified by exact model/backend tuple only |
| Q4_1 | 32 / 20, signed offset and scale | Q4_1 x Q8_1 | Existing specialized source projection | Model/backend/shape specific |
| Q4_K | 256 / 144, scalar decoder exists | **Deliberate Unsupported** for unqualified source Q4_K x Q8_K, despite reference dequantized diagnostic | No corresponding complete GPU kernel in source-format map | NOT granted |
| Q6_K | 256 / 210, q6 decoder | Signed dynamic Q8_K/laned original source arithmetic | Existing source projector q6_k | Scope-specific |
| Q8_0 | 32 / 34, q8 decoder | Q8_0 x Q8_0; distinct pure f32 reference/OpenBLAS contract | Existing q8_0 kernel | Scope-specific |
| Q1 family | README home only | No implementation | No implementation | No |
| Q2 family | README home only | No implementation | No implementation | No |
| Q3 family | README home only | No implementation | No implementation | No |
| NVFP4 | README home only | No implementation | No implementation | No |
| EXL3/QTIP/trellis/codebook | Design/research only | Not qualified | Not qualified | No |

Location notes:
- core/src/quant.rs currently recognizes F32/F16/BF16 and exact Q4_0/Q4_1/Q4_K/Q6K/Q8_0; from_ggml/from_safetensors have checked Unsupported for other formats.
- artifacts/src/quant.rs uses q4.rs, q6.rs, q8.rs and f16/bf16 converter functions. New q1/q2/q3/nvfp4/precision/bf16 directories hold README placeholders, not executable Rust.
- CPU source.rs/projection.rs handles original activation quantization and source-reduction, even though new cpu/src/source/quantization/q* directories hold only README. Do not invent implementability from those folder names.
- GPU src/quantization/* actually groups activation, fused, batch and exact Q4_0/Q4_1/Q6_K/Q8_0 source projection code. CUDA/HIP/Vulkan body and ABI proof is separate.
- Family source roles are in model-Qwen/model-Gemma, no quant format is allowed to invent valid tensor/model geometry.

## 4. Immediate structural milestone S0 for encoding

Create compile-visible types/contracts under core/encoding and core/numerics and exact parser/codec namespaces under artifacts. Existing QuantKind and old module names may remain internal while moving code in S1, but do not freeze them into a new public API. The facade should expose source/codec capability inspection and explicit refusal. All new placeholder paths must fail closed if called; don't add dummy "Q1" or "NVFP4" decoders returning fake zeros.

Recommended behavior during S0:
- Current working scalar decoder/CPU quant existing APIs still callable, or a narrow forwarding implementation to them under new public facade (temporary within tranche). Source parse/refusal unchanged.
- New codec/quant branches not yet migrated are module stubs with typed Unsupported, never todo! in an ordinary call path.
- Exact codec schema is a versioned type identity, not a string lookup at each matrix multiply; direct kernel selected once at prepared region.
- No codec-specific model/runtime engine, allocator, HTTP route or standalone scheduling actor.
- Add an inspect-only codec capability table that honestly reports recognized format, scalar decode, CPU kernel, GPU kernel and whole-model readiness as distinct fields.
- Keep old data/normalization/test conversions until next code moving units, no accidental change in numerical expression ordering.

## 5. Code migration S1 — physical implementation, not just renames

Small logical units (can commit per coherent group or as one S1 with passing focused checks):

S1.a Artifact codec exactness:
1. Introduce core ScalarDType vs QuantScheme/StoredTensorEncoding with explicit source GGML code mapping and checked byte-length.
2. Move original quant decoder q4/q6/q8 scalar functions into artifacts new codec namespace, preserving F16/BF16 nonfinite/round-to-even and all source byte-size bounds exactly.
3. Migrate parser/SourceTensorRef/source weights/quant tests in one source-integrated unit. Keep only one actual source decoder implementation, delete old aliases after callers updated.
4. Add unknown/malformed/truncated/overlap and original block-format fixtures; keep Q1/2/3/NVFP4 Unsupported.

S1.b CPU numerical contract:
1. Separate pure f32 dequantized reference from source-exact GGML Q4_0xQ8_0, Q4_1xQ8_1, Q6_KxQ8_K, Q8_0xQ8_0 and source BF16 calculations.
2. Move real dot/repack code into cpu/quantization or precise owned operation modules; preserve eight-lane reductions, cast/power/subnormals/negative-zero and AVX2 source profiles. No new numerical tolerance due to file move.
3. Retain deliberate Q4_K source execution refusal; its scalar decoder does not grant operator capability.
4. CPU tests compare actual original golden vectors and malformed/unqualified source guards, not merely converted f32 approximations.

S1.c GPU packed operation ownership:
1. Keep existing gpu/quantization implementation in place if it already meets owner contract; relocate only new module exports and source binding description, not every .cu file.
2. Implement adapter to semantic SourceTensorEncoding/ActivationEncoding + numerical profile + typed prepared kernel guard; no model named dispatch or hidden code-image change.
3. Retain q4/q6/q8 kernel unit fixtures and per-CPU/CUDA/HIP/Vulkan separate qualification marks; never transfer permits on feature-name equality.
4. Confirm encoded/captured queued readers and physical buffer generations; do not evict source-backing from a prepared direct packed kernel.
5. Any future Q1/2/3/NVFP4/EXL3 remains separate feature expansion AFTER transition acceptance.

## 6. CUDA GDN changes and kernel profile crosswalk

The 27-Oct CUDA GDN work explicitly separates a backend configuration (GdnKktKernel variants, GdnRecomputeKernel, GdnStateKernel, GdnOutputKernel) and TextGdnImplementation from historical finite qualification table recipes. This should remain **a source/phase/ISA/numerics-qualified physical candidate**, not an alternative model-specific execution engine.

- User-facing model family supplies Qwen source operator/phase/GDN geometry and parameter/buffer state roles.
- Backend supplies reusable parameterized GDN implementations and actual code/ABI identities.
- Compiler/preparation joins only compatible SourceCall and GdnKernelConfig and checks full phase stage implementation prior to H2D or root reservation; refuses missing implementation.
- Qualified source evidence may certify exact fresh47 or 48 or cached mode; a source-kernel implementation existing is not proof of general dynamic source.
- Backend code may retain explicit specialized geometric constants as a guarded implementation (e.g. QKV/Z width) but NOT falsely treat them as generic GDN mathematical law.
- Fresh-prefill capture added in ba7fb03 and oracle reference layout amended in fadf2b1. Treat as **in-progress source/qualification**, not promoted ordinary dynamic CUDA. Keep code/compiler/driver/ABI hashes for comparison when source moves.
- Historic finite source recipes and expected-output comparison remain in independent qualification/test support; code running in production must not lookup testcase-specific names or prompt lists.

## 7. Derived low-bit/approximate models after transition

Explicit EXL3/QTIP and ultra-low-bit/NVFP4 may materially improve extreme-model capacity but are not structural skeleton prerequisites. Before implementation, enforce:
- original source hash/model tensor complete identity, trained sidecars and source licensing;
- per-tensor exact packed coding, scale/codebook/trellis/basis transforms/exception handling, including MoE/embedding/head special tensors;
- optional offline calibration on independently held-out corpus; deliberate lossy conversion creates new ModelArtifactId;
- separately qualified CPU decoder/reference and direct GPU packed kernel, not mandatory full FP16 reconstruction;
- source checkpoint vs derived model quality; backend arithmetic vs derived source; target speculative losslessness relative to the *selected derived source* are THREE different quality claims;
- actual profile includes memory/compute, decoder/tile/register/LDS/capture cost, not just bits/weight; smaller bpw can run slower;
- model, quant and backend profile remain immutable within TaskSession/StateRoot/Bundle compatibility without explicit verified conversion/replay.

Source index: original ../fabric-vnext/README.md `quant, ../fabric-vnext/PRIOR_ART_AND_REVALIDATION.md, project .agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md and pipeline-composition/FORMATS.md. Follow actual source IDs and current laws, not guide README marketing.

## 8. Codec acceptance for the final consolidated Q0/Q2

**At every S1 commit:** fmt, architecture checker, exact artifact/CPU/GPU source package check and SMALL codec fixtures, no full trained/GPU qualification.

**At Q0 integrated transition:**
- Exact container source pack decoding byte-for-byte where defined; unknown/new variants refuse, safe parsed extents and overflow limits.
- Scalar F16/BF16 special values, signed zeros, scale/zero offsets, q4/q6/q8 blocks and negative malformed/corrupted source vectors.
- Q4_0/Q4_1/Q6_K/Q8_0 source-exact CPU output under original activation coding/reduction and explicitly distinct ideal F32 reference.
- Independent per-backend physical GPU source/ABI/version/shape/driver/Ready/retirement/unknown fault for qualified tuples only.
- End-to-end previously qualified Qwen3.5 GGUF CPU and native GPU source executions, same complete logits and all required state according to original mathematical contract.
- Separate original HF Qwen, Gemma4 E2B/HIP/Vulkan, Qwen27B and user-held device scopes; report unavailable/unexecuted instead of promoting.
- Finite CUDA fresh-prefill 47–50 original model profile checks only when matching original source and actual physical ownership; last changed capture cannot inherit earlier binary receipt.
- Confirm codecs/precision execute through the new model-neutral engine without codec-specific schedulers or HTTP switches.

**Never relax numerical/stochastic/determinism acceptance just to pass refactor.** Latest .agents/plans/numerics-and-context/CORRECTNESS_POLICY.md governs distinct semantic safety, operator accuracy, whole model quality, and deterministic/batch consistency claims. Original failed widths N1 vs B4 remain failed if genuinely different even when each separately fits reference.

## END FILE: .agents/plans/fabric-vnext-transition/QUANTIZATION_AND_NUMERICS.md


---

## BEGIN FILE: .agents/plans/fabric-vnext-transition/STRUCTURAL_SKELETON.md

# Fabric vNext transition — FIRST full Rust skeleton and complete code relocation

**The user's highest-priority implementation milestone** is visible, compilable code structure of the ENTIRE Fabric vNext architecture immediately, with all crate/module ownership layers and small typed interfaces ready for implementation. That must precede the detailed implementation migration and expensive qualification. Design only in this current conversation; no Rust was edited.

## 1. S0 — complete structural skeleton in one signed logical commit

The first source implementation milestone must introduce:

- NEW `crates/zerogravity-engine`: model-neutral Runtime, ModelCatalog/ModelInstance, Router, TaskCoordinator/TaskDriver, Scheduler, PreparedExecutable/ExecutableBundle, Residency, Placement and committed output handling. Uses existing one shared physical ResourceArbiter, not a separate allocator.
- NEW `crates/zerogravity-serving`: independently reusable server library with ServerBuilder/ServerHandle, HTTP/OpenAI/SSE, auth/ingress/output lifecycle, metrics and future media protocol modules. Depends on engine/core, never facade/CLI/Qwen/Gemma/GPU.
- EXISTING `crates/zerogravity`: fully visible public SDK sections Runtime, models, artifacts/pull/cache, devices/placement/dry-run, direct run and events, pipelines/config, serving builders/reexports, diagnostics, built-in registration.
- EXISTING `zerogravity-cli`: target source-level thin shape args/commands/output/signals/entrypoint. Temporary old CLI implementations can coexist only during S0–F2; F3 deletes old runtime/HTTP/business implementation, and CLI source after F3 uses only public zerogravity library for all project work.
- EXISTING `core`: semantic program based on current ForwardPlan, source/quant/numerics/identity/task/state/provider/value/placement modules visible, including typed future media, NPU and remote capability schemas but not fake backends.
- EXISTING `artifacts`: containers, exact codecs and source reads, immutable model store/cache/acquisition and future derived quant manifest.
- EXISTING `CPU/GPU`: clear operator/quantization/precision/physical execution/transfer/memory/capture/qualifier ownership; existing bodies retained and moved only if needed.
- EXISTING `model-Qwen/Gemma`: authoritative family config/roles/source binding/forward/state/tokenizer/specialization/reference homes, not another engine.

Each source-visible top-level module must have at least one meaningful Rust type, function, enum or contract, not empty directories. All target crates and feature selections must compile, have one real acyclic Cargo DAG, and document their responsibilities. Import or link the complete expected tree from ARCHITECTURE_AND_CRATE_CONTRACTS.md.

**S0 is a single structural milestone** and should be one coherent signed commit following only fast architecture, Cargo-check, formatting and small contract/refusal tests. It does NOT claim new model functionality.

## 2. Safe TODO strategy in real Rust

The user explicitly allows `todo!()` placeholders where needed for the first structural skeleton. Enforce the safety boundary:

- Every ordinary public/CLI/HTTP/live Runtime operation must either delegate to a proven existing implementation or return a clear typed `Error::Unsupported` / `NotImplemented` including missing capability and status.
- A `todo!()` macro is acceptable only inside an explicitly unreachable, private, development-only or future-disabled module. It MUST NOT be on a production, release, default, "all" feature, supported device, or live request route. Never use it to construct fake Ready, tensor bytes, logits, source permits, working TaskSession, buffer lease, admission credit or authenticated model support.
- Future Q1/Q2/Q3/NVFP4 codec, multimodal TaskDriver, remote worker or graph-only accelerator APIs can have **types and explicit refusal** in S0 without physically implementing them.
- S0 may leave current supported HTTP/runners on the existing CLI implementation temporarily; F0–F3 migrate them without compatibility branches remaining at completion.
- A module exists and compiles in S0 even when its exact source/model/backend operation has not yet been implemented. Library capability reports must distinguish structural API availability from actual model/physical support.

Example permitted public contract:

~~~rust
pub fn prepare_unsupported_media(spec: &MediaTaskSpec) -> Result<PreparedTask> {
    let _ = spec;
    Err(Error::Unsupported("media TaskDriver not source-qualified".into()))
}

#[cfg(feature = "internal-stub-only")]
mod never_selected_by_runtime {
    fn future_kernel_lowering() -> () {
        todo!("private planned implementation")
    }
}
~~~

The internal-stub feature must not be included in release/all-features build surfaces unintentionally; best use typed refusal for most skeleton interfaces. Avoid a new dead code tree requiring clippy allowances everywhere: favor small public types over hundreds of placeholder functions.

## 3. S0 mechanical action order and integration

Before coding:
1. Recheck actual current HEAD/status, AGENTS.md, active plan ownership and concurrent agents; source may have advanced beyond fadf2b1. Never overwrite uncommitted docs or other agents' work.
2. Decide exact final crate/type names once against current source and avoid unnecessary public compatibility aliases. The user explicitly permits breaking pre-v1 APIs.
3. Freeze input source identities, existing actual capability inventory and the lightweight migration smoke commands. Do not run heavyweight model/device suites yet.

Structural source unit:
4. Add engine and serving crates to workspace with safe default feature graph; no concrete model/GPU import in engine; no facade/CLI import in serving; CLI depends only on facade for project behavior.
5. Add core semantic/model/task/encoding/numeric/provider/placement/root type homes, reusing original ForwardPlan and low-level ResourceArbiter/SessionAuthority implementations as the one actual source.
6. Add facade Runtime/TaskRequest/Models/Devices/Artifacts/Pipeline/Serving API surfaces and cold builtins registration; new methods either delegate or fail closed.
7. Add engine Scheduler/ModelInstance/TaskCoordinator/ExecutableBundle/CommittedEvent types and self-contained lifecycle; no automatic source loads or sockets.
8. Add serving ServerBuilder/ServerHandle/HTTP/SSE placeholder modules and a safe builder; do not bind without explicit start and actual engine capability.
9. Add typed task/media/clock/snapshot/stage/remote/optional accelerator descriptors and explicit future capability statuses; no functional media/remote claim.
10. Add CLI target args/commands/output/signals modules and path-map; no forced switch of all current production until corresponding migration phase.
11. Extend existing architecture checker to enforce all target crate edges and no production source imports of model test fixtures. Update relevant Cargo feature checks to include new packages, without making CPU-only build pull CubeCL/native GPU compiler, HTTP or acquisition dependencies by default.
12. Run small checks; review no panic on reachable route; signed structural commit, explicit S0 result in side plan.

**Exit:** all target structural layers *exist in actual compiled Rust*, module ownership recognizable, public crate exposes a coherent API with support-aware errors and independent serving compile capability. Existing supported code has not been silently discarded. Zero new physical/model runtime acceptance from skeleton.

## 4. S1 — codec/precision/source ownership

Move real source code and callers, not just declarations.

- core/src/quant.rs -> typed `ScalarDType`, `QuantScheme`, `StoredTensorEncoding` with exact GGML/safetensors mappings, checked byte length and unknown refusal.
- artifacts/src/quant.rs, quant/q4.rs, quant/q6.rs, quant/q8.rs -> exact artifacts source codecs. Retain F16/BF16 edge handling, source offsets, scales, malformed shape tests.
- CPU source.rs/projection exact original GGUF dynamic-activation Q8_0/Q8_1/Q8_K arithmetic and reduction -> CPU owned quantization/precision operators. Preserve pure F32 dequantized oracle as separately named diagnostic, not exact source dot.
- GPU quantization/ q4_0/q4_1/q6_k/q8_0, activation, batch projection -> keep physical kernel bodies; adapt code to shared encoding/numerics type and qualified source/shape candidate.
- Q1/Q2/Q3/NVFP4/EXL3 README-only homes remain unimplemented; Q4_K original CPU matvec still deliberately refuses despite scalar decode.
- Migrate all source/quant parser/tests and callsites coherently; delete obsolete pre-v1 legacy names once new APIs serve all callers, no shadow codecs.

Suggested one S1 signed commit or split S1.a artifacts, S1.b CPU, S1.c GPU only if each subunit compiles and has targeted fixture coverage.

## 5. S2 — family semantics and GPU/qualification

- Continue already accepted a5ed7de/263ff79 Qwen/Gemma family containment. Do not reintroduce model→GPU→model cycle or move all model-specific source into core.
- Keep one Qwen source-exact traversal and Gemma math with independent PLE/softcap/shared-KV/MoE as appropriate.
- Build one canonical SemanticProgram from existing ForwardPlan, capable of host reference or prepared closed-region implementations; no duplicate model traversal.
- GPU physical provider implements reusable actual kernels/ABI/leases; family source adapter owns original model eligibility and tests; application facade cold registration joins families and physical candidates.
- Preserve recent parameterized CUDA GDN Conv/KKT/Recompute/State/Output and fresh-prefill source code/image hashes without rewriting real CUDA math just for new module naming.
- Separate production implementation config from historical finite prompt/recipe cases; test-only source/fixture orchestration in tool/tests, production always requires actual semantic/source/profile/device/shape permits.
- Keep physical CUDA/HIP/Vulkan qualification independent, no code rename giving false capability. Existing frozen/held gfx90c and physical device scopes remain held.
- Delete obsolete direct model-specific routing from physical worker only when generic adapter exists and tests compile.

Suggested S2 signed commit or S2.a family+semantic, S2.b physical provider+qualification if needed for reviewable change.

## 6. S3 — canonical session + resource + event semantics

- Reconcile core/state.rs TaskCoordinator/StateRoot/PreparedDomain/CommittedOutput with core/session_root.rs SessionAuthority/RootBinding/RootAttempt into one logical StateRoot/attempt/epoch/provisional-domain/publication contract, keeping original physical completion, source identity and semantics.
- One publisher per TaskSession/shard. Source identity, expected previous root, epoch, frontier, domain schema and real ReadyWitness required. Device events/IO callbacks never publish logical root themselves.
- No new engine-only ResourceArbiter; existing process-shared low-level ledger stays below engine; introduce all-or-none bundle reservation only if actual multi-domain consumer/proof requires it, with one alias accounting and no duplicate host budgets.
- Preserve all physical readers encoded/captured/unsubmitted/in-flight and quarantine. Separate physical Ready, logical Commit, transport Delivery and client ACK/playback.
- Adapt CPU/Qwen/Gemma/GPU state callsites once to common contract; no permanent CPU-vs-GPU second logical root runtime. Preserve real source/driver-specific state capabilities and unsupported modes.
- Compile and run small test matrix for source/state authority and reservation/lifetime, without GPU device/model long suites.

One S3 signed commit or S3.a root, S3.b reservations if independent unit.

## 7. S0–S3 lightweight exit table

| Phase | Minimum check before commit | Deferred |
| --- | --- | --- |
| S0 | Cargo package/feature graph, fmt, architecture checker, new type/builder/refusal smoke, CPU-only cargo check | Full check.sh, hardware, trained models, performance |
| S1 | artifacts+CPU+GPU source-only checks and exact q4/q6/q8 malformed/golden fixtures | Full model/driver GPU and slow codec benchmark |
| S2 | family+GPU source/feature checks and representative semantics/source/ABI/capture fixture | Fresh actual CUDA full model, Gemma/27B, C16/C32 |
| S3 | CPU state/root/resource source-free targeted tests and affected crate compile | GPU fault injection, full root migration stress, long context |

Fix obvious failures before each commit, but one expensive integrated qualification only after F4. Do not waive source, state, memory or root safety merely because a structural test passes.

## 8. Transition halfway point

At the end of S0 architecture is VISIBLE; at end of S3 the lower boundaries are MOVED. Old legacy CLI serving and source business logic will be migrated in F0–F3, not retained as permanent competing runners.

Existing source correctness, acceptance identities, model-specific test results, original non-support and format guards must be preserved. A pre-v1 Rust type/name change is allowed; changing mathematical arithmetic, buffer lifetime or authorized capability without explicit new qualification is not.

After S3 proceed immediately to the model-neutral engine migration F0, public SDK F1, independent serving F2, thin CLI F3, present-feature coverage F4 and one consolidated Q0–Q2 integration gate. Full details in PHASED_IMPLEMENTATION.md.

## END FILE: .agents/plans/fabric-vnext-transition/STRUCTURAL_SKELETON.md


---

## BEGIN FILE: .agents/plans/fabric-vnext-transition/PHASED_IMPLEMENTATION.md

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

## END FILE: .agents/plans/fabric-vnext-transition/PHASED_IMPLEMENTATION.md


---

## BEGIN FILE: .agents/plans/fabric-vnext-transition/FEATURE_PARITY_MATRIX.md

# Fabric vNext — comprehensive current feature and behavior preservation matrix

**Source baseline:** HEAD fadf2b1 on dotlink-l5, 27 Oct 2026; source-scope evidence read from current crates and .agents/plans/TODO.md, not a new physical/test qualification. This is the **functional recovery checklist** for F4/Q0/Q2. Break/delete pre-v1 implementation and API names freely, but do not lose actually supported model, artifact, device, service or safety behavior. Current stubs/metadata only MUST remain explicitly unsupported until genuinely implemented.

## 1. Preservation policy and truth states

Record separately for every feature:
- **Runtime-functional:** supported on an exact model/task/source/quant/backend/shape, with actual output and state semantics.
- **Source/binary qualification only:** selected operator/capture/driver/source fixture has evidence, not general model/runtime authorization.
- **Structural/config validation only:** accepted YAML/media/model metadata, but no numerical execution or service.
- **Planned/not implemented:** typed API may exist, must refuse.
- **Held/unsafe:** explicit user/source/hardware authorization or device safety hold; cannot run to fill a gate.

For each row below: current owner, new owner, intended working behavior, required negative tests and final proof. Qualification state does not automatically improve because a file is moved or Cargo package compiles.

## 2. Direct model and source functionality

| Function | Historic current scope | New owner | F4/Q0 expected behavior |
| --- | --- | --- | --- |
| Qwen3.5 GGUF 0.8B and 2B direct CPU | Native source-exact text, tokenization, prefill/decode, state, known context scopes | Family Qwen + CPU provider + model-neutral engine + public Runtime | Same supported numerical/functional output, prompts, stop/cancel; no HTTP or Qwen-CLI dependency |
| Qwen3.5 source tokenizer/config | Qwen tokenizer + exact GGUF role/shape checks | Family/Qwen + public ModelInspector | Same source roles, chat template, Qwen control token sanitation |
| Qwen original HF BF16 | finite source/reference/diagnostic and CUDA candidate paths | Qwen family + source adapter + qualified provider | Preserve *finite* proof and explicit ordinary Unsupported outside qualified source/shape |
| Qwen3.8 27B | indexed/sharded source, CPU windows and physical HIP source/qualification work | Qwen family + artifacts SourceWeights + CPU/GPU | Same source/oracle/lease tests, not general trained/SSD serving |
| Gemma4 E2B | original-source config/tensors/source, CPU text/reference and finite HIP/Vulkan provider experiments | Gemma family + CPU/GPU provider + engine adapter | Preserve exact PLE/attention/state and source validation/qualified scopes; no universal model support |
| Gemma 26B-A4B /31B | separately coded variant geometry, MoE vs dense source work | Gemma family | Preserve variants and Unsupported where no physical/quality proof |
| Model aliases and loaded Qwen GGUF | CPU serving shared immutable source by canonical path, independent sessions per request | Engine ModelCatalog+Router+Residency | Same source share, private RootBinding, exact alias revision, no duplicate weights |
| Metadata-only model targets | core/targets.rs Qwen Flash, GLM, DeepSeek, Llama, image/media etc. | Facade compiled target catalog with exact statuses | Listing/inspection okay; cannot mint LoadedModel/serve permission |

New execution API must be callable from an external Rust crate importing only zerogravity. Any model not genuinely supported should produce structured Unsupported before allocating model GPU work. The facade cannot advertise future model as available merely because its family/format enum exists.

## 3. Formats, quant and source windows

| Capability | Existing scope | Required outcome |
| --- | --- | --- |
| GGUF tensor metadata, parser and bounded open | implemented artifacts source; strict typed tensor layout | same source-length, shape, type, malformed/unknown/overflow guards |
| Safetensors / indexed shards | implemented verified owned source/index and bounded tiny test fixtures | exact shard digest/source owner identity, no incomplete tensor permit |
| F32/F16/BF16 codec | exact source read scalar conversion; BF16 numerical profiles separate | source arithmetic unchanged, correct nonfinite/subnormal/negative zero |
| Q4_0 | decoder, source Q4_0xQ8_0 CPU/repack, selected GPU kernel | same exact blocks, scales, activation and reduction; backend gated |
| Q4_1 | decoder, source Q4_1xQ8_1 CPU and GPU implementation body | preserve signed offset and arithmetic bitwise/reference rules |
| Q6_K | decoder, source Q6_KxQ8_K CPU and selected GPU | preserve signed activation, reduction and original profile |
| Q8_0 | decoder, CPU Q8_0xQ8_0, GPU source body | same bytes/row geometry, source F32 separate |
| Q4_K | scalar decoder but deliberately Unsupported source-exact CPU dot | decoder still works; CPU/model run still refuses unless separately qualified later |
| Q1/Q2/Q3/NVFP4 | format directory readmes only | metadata status remains planned/no real codec; no generic false output |
| EXL3/QTIP/trellis | planning only | derived-source contract reserved, no automatic quantizer or performance claims |
| Activation packing | GGUF dynamic Q8 variants for particular stored encodings | source stored bytes vs compute precision and effective dynamic Q8 stay separate |
| Large immutable SourceWeights | verified_shards, owned source, cache/window, resource admission | source owner and tensor lease survive all consumers; no fake SSD streaming |
| Cache transactional model store | original cache/acquisition/pull features, file-safety and errors | same atomic publication/locking/resume/immutability with exposed public SDK |

Each codec must have exact separate recognition, scalar decode, source operator, native provider and whole-model capability fields. No one bool `supports_quantization` for everything.

## 4. Artifact acquisition and model management

The existing CLI pull/provider implementations and artifacts lower-level acquisition already cover the following categories, with separate permissions, pinned versions and bounded storage contracts:
- Hugging Face basic and indexed safetensors/pinned model selection, sidecars and cache.
- GitHub source trees/release content as applicable.
- Hugging Face Xet transport and explicit cache.
- ModelScope.
- AWS S3 path/bucket/revision/conditional source reads (correct SigV4 and endpoint/proxy policies).
- GCS and Azure object/prefix paths with exact conditional etags/revisions.
- OCI/container image source fetch or manifest selection.
- Generic HTTP(S) object transfer, resume/etag/range and cache views.
- Source GGUF quant basename/source selection (selection != conversion), candidate/filter/component bundle discovery, report/progress and typed unsupported/malformed paths.
- Cache import/manage/run/current inventory, source hashes, transactional snapshot store.

**New owner:** artifacts/src/acquisition/cache/transport for real source work; facade public `ModelStore`/`Acquisition`/`Cache` API for model/source selection, progress/status and callbacks; CLI only input/output. No forced network access in library constructor or inspect. Unsupported transport or unavailable source returns exact refusal, no fallback to untrusted Python or alternate source.

**Tests:** existing CLI pull_*.rs, cache_* integration fixtures should be moved/reused as Rust library tests and small CLI delegate parity tests. Do not drop subtle path traversal/symlink/malformed shard/conditional race test to make CLI tiny. A standalone Rust program must be able to perform the same source selection/pull/cache request without spawning CLI. Use controlled loopback fixture rather than real downloads for cheap phases; full acquisition/isolated loopback suite Q0.

## 5. Device inventory, fit and run configuration

- Linux GPU device selectors (including physical PCI identity, /dev validated mappings and rejected unrelated character/control nodes), local ordered allowlists and CPU fallback rules.
- Separate device identity, vendor backend (CPU, CUDA, HIP, Vulkan), actual physical device and stage role. Do not infer split/run from a matching name or /dev path alone.
- Qualified source task+device+shape and available real physical fit, host/UMA and cgroup headroom; GTT virtual range not free DRAM.
- Model/backing/host+GPU+compiler+scratch+state+output+IO complete peak budgeting, one process-shared ledger; multi-domain future reservations all-or-none before physical work.
- Config precedence, bounded YAML/JSON parser and CLI flags, source/profile/headroom/diagnostics, dry-run/explain and full errors.
- Per-model source/quant/numeric and effective qualified GPU strategy selection, no hidden model change under pressure.
- Public `Runtime::devices`/`RunPlan`/`explain` equivalent plus CLI thin pretty table and argv parser.

Use existing CLI-device-routing plan as exact behavior; no second device selection engine in CLI or server. Preserve disallowed gfx90c device launch and dangerous/unknown reset holds. Code can inspect device capability without launching unsafe GPU/remote operations, but such inspection is not acceptance evidence.

## 6. Serving and batching behavior

| Behavior | Existing accepted subset | New owner / proof |
| --- | --- | --- |
| CPU Qwen GGUF OpenAI HTTP/1.1 | loopback server, GET health/v1/models, POST completions/chat completions | serving lib HTTP and engine Qwen GGUF CPU; preserve request/response errors |
| Greedy sampler API | temp0/top_p1/n1/zero penalties, bounded output, strict unsupported params | source model/engine sampling and serving request validation, no silent ignore |
| JSON/SSE | streaming responses, finish_reason, usage, UTF-8/EOS, selected request metrics | server protocol emitter consumes engine committed output; different delivery clock |
| CPU concurrency | finite workers, active/queued, fair prefill/decode quanta, shared weights and private roots | model-neutral engine scheduler, same limits/behavior |
| GPU Qwen GGUF | one original resident source, independent GPU sessions, explicit device completion | GPU provider + engine coordinator |
| GPU native fused B2/B4/B8 | opt-in qualified same-source decode cohorts with differing frontiers, ragged/unknown fallback B1 | engine work formation using prepared native kernels; no fake multiple B1 work |
| Resource/host credit | bounded connections/request body/state/queued/scratch/credits | serving ingress plus engine physical one arbiter |
| HTTP cancellation | disconnect cancels future worker quanta, preserves submitted GPU backing until Ready/retire | server cancel scope, engine session epoch and physical worker |
| Slow-client backpressure | bounded response output packet, parking, timeouts and survivor handling | server transport credits + engine CommittedEvent stream |
| Server shutdown | recent 50abaf7 worker join and owned provider closure/stop evidence | serving ServerHandle, engine pending admission drain, no device leak |
| Model aliases | canonical shared source and source-bound CPU/GPU request root | engine alias generation and exact model pin |
| Unsupported API scopes | tool calls, grammar, embeddings, Responses, media etc. are not blanket implemented | separate typed server endpoint capability, Unsupported/refuse |
| vLLM goodput parity | not demonstrated by current finite native source cases | not part of transition support claim; later matched benchmark |

No hidden per-model HTTP server, no CPU-path token scheduler copied into serving crate, no independent Socket→StateRoot publisher. HTTP API through public zerogravity::serving and direct Rust run must share engine and model residency.

## 7. Pipeline and multimodal compatibility

- CLI pipeline v1/v2 YAML/flags typed schema, duplicate/unknown/size/depth/symlink/path refusal and type/port graph/DAG checker: already functional *structural* validation only. Move parser and validator to public SDK; CLI display only.
- Source-neutral builtin component kinds (text, asr, diarization, tts, image, decision) and typed ports/clock descriptors: preserve validation/registry, not imply model execution.
- Pipeline run remains deliberate NOT IMPLEMENTED until later real TaskDriver/provider/task-output acceptance. Safe typed library return and CLI error.
- Immutable chat-context grants and source/authorization/freshness typed snapshot views preserve data privacy and permission semantics, even though realtime model execution is future.
- Native actual media (ASR, TTS, image, video), external hosted TaskServiceAdapters, multi-peer state transfer are not current general functional features; skeleton has typed architecture, not a successful fake output.
- A ModelAssembly is not a cross-model TaskPipeline, and compositional pipelines cannot spawn new HTTP inference servers or silently merge different users' StateRoots.
- Prepared/Committed/Delivered/Ack/Played events distinct even in future transport.

## 8. Numerical, state and physical invariants to preserve

- Original task/model/quant/source/binary identity, independent reference profiles, no model or backend width grant inheritance.
- Source GGUF and original HF BF16 arithmetic are different contracts. Pure dequantized F32 oracle is distinct from original GGUF activation coding and per-lane rounding.
- Source-independent region replacement only when all state/output effects and numeric/shape guards pass.
- State root publication checks actual old root, epoch, session/source and all required prepared state domains; no foreign/stale root or two session publishers.
- First-use physical event Ready is *real* driver completion, not submission, Graph capture, worker queue dequeue or diagnostic output.
- ResourceArbiter one process shared physical budget with source ownership, actual backing aliases, preflight peaks, encoder captured readers and unknown/quarantine retention.
- Cancellation/exception handles do not free in-flight DMA reader or reset shared physical GPU for one request.
- Strict unsupported/invalid/limits and source mutation/refusal ordering remains. No false recovery by retrying an unknown submitted GPU operation on another provider.
- Signed historical source/model/backend proof stays attached to exact code identity. Renaming executable/source invalidates blanket transfer of physical acceptance until reconciled.

## 9. Final F4/Q0/Q2 parity table requirements

At F4, mark each row from ``2–8 as:
- retained and rehomed with real implementation reference;
- tested with focused source-free/loopback fixture and exact command;
- held for consolidated Q0 expensive qualification;
- not implemented originally (continue typed Unsupported);
- deliberately removed ONLY if it was a pre-v1 compatibility/API duplicate, not a previously supported behavior.

At Q0 run one full assembled check with source/codec/CPU/GPU/serving+SDK/CLI, preserving ignored/failed/unavailable cases. At Q1 fix at lowest appropriate layer and with small tests. At Q2 final accepted complete gate. No claim "all features working" unless every *originally supported* behavior from this matrix and its original gates is demonstrably present, while future-only cells remain clearly unsupported.

## 10. External library and release-facing coverage

An external Rust application with a **single public zerogravity dependency** can:
- inspect, load and run supported model/quant source;
- use original bounded acquisition/cache/pull/verify and source-manifest methods;
- list model capabilities and device/fit/explain;
- submit concurrent direct text TaskRequests, cancel and observe committed output;
- validate a pipeline YAML/typed graph, get Unsupported for unqualified execution;
- attach and start an independent HTTP OpenAI server via zerogravity::serving and stop it cleanly;
- share immutable model source while maintaining independent live sessions with the same low-level resource authority.

The CLI must be a *consumer of these same functions*, with no separate source importer, model-specific server, quant registry, TaskCoordinator or native kernel selection.

**Transition completion is not a v1 release.** Future model/media/SSD/distributed features need their own gates after architectural acceptance. The user expressly allows Rust name/signature breaking and requires no old compatibility shims at final state.

## END FILE: .agents/plans/fabric-vnext-transition/FEATURE_PARITY_MATRIX.md


---

## BEGIN FILE: .agents/plans/fabric-vnext-transition/QUALIFICATION_AND_COMMIT_POLICY.md

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

## END FILE: .agents/plans/fabric-vnext-transition/QUALIFICATION_AND_COMMIT_POLICY.md


---

## BEGIN FILE: .agents/plans/fabric-vnext-transition/ROADMAP_AFTER_TRANSITION.md

# Fabric vNext — feature-expansion roadmap AFTER structural/functional transition

**Not part of first S0–F4 / Q0–Q2 migration completion.** This document preserves all current zerogravity product targets and full future design threads, with an implementation ordering that starts only after a correct model-neutral engine, public SDK, standalone serving library and thin CLI exist. Every new capability is separately source/task/numeric/state/physical/performance qualified before being advertised.

## 1. Base prerequisite

Tranche S: complete immediate compiled Rust module skeleton and migrate current exact codecs, family math, GPU provider and one state/resource protocol.

Tranche F: route all currently supported Qwen/Gemma/source/cache/device/pipeline/API features through new engine, exposed through zerogravity public Rust SDK and reusable zerogravity-serving crate. Delete old shadow runners/compatibility.

Tranche Q: one broad assembled check, targeted bugfixes and one final complete acceptance. Preserve original negative/held hardware/driver/model cases. Never claim vLLM-equivalent performance or full multimodal task support merely because the new packages compile.

Only then broaden physical/codecs/multimodal/distributed.

## 2. E01 — production-safe ordinary Qwen CPU and CUDA HF paths

- Finish Qwen3.5 0.8B independent source correctness for *fresh* prompt lengths 47/48/49/50 separate from cached decode at 48–50 and historical 47 prefix.
- Promote general original BF16/HF CUDA only through actual source qualified dynamic GDN/attention/KV/reduction with native model natural-feedback, every full-vocabulary affected state and physical lease/cancel/unknown; no historical finite captured selection override.
- Extend Qwen2B independently with source/numeric/whole-model/state profiles (0.8B does not qualify 2B), then Qwen3.8 27B source/data and independent full model gates.
- CPU/GGUF and original BF16 source remain different selected numerics; CPU/CUDA/HIP/Vulkan separate independent profiles. Cross-width batch bit equality only when profile explicitly promises it; default vLLM source may differ N1 vs B4.
- Long contexts 512/1K/2K/4K/8K and later 16K/32K independently tested for actual semantic state, RoPE, KV/GDN history, output and memory fit, not a position-only test.

Exit: actual ordinary source model run and API through public lib + serving, not finite test-only route.

## 3. E02 — vLLM-class continuous batching, async execution and goodput

- Persistent generated token / request slots with generation identity, bounded incremental metadata and GPU resident buffers; no model catalog/JSON/compilation/allocator per-token work.
- True mixed temporal prompt chunk prefill/decode scheduling with fair old/new session admission, deadline/aging, select task-compatible cohorts, valid ragged masks and independent root.
- Native B2/B4/B8 now works in finite cohort paths; extend actual independently qualified widths/concurrency, dynamic arrivals, vLLM Model Runner V2 lessons and reduce CPU↔GPU sync and first-use compile churn.
- Paged and **complete-domain** full KV/SWA/GDN/KDA/MLA/indexer/grammar/draft/media prefix caching/COW with common-restorable-frontier checks; no KV-only false hit.
- Dynamic shape-band cache, qualified capture/replay, CUDA graph source reader lifetimes, GPU-side sample and feedback where source/grammar allow.
- Unified task-aware scheduler and physical reserve: queued input bytes and uncomputed prompt, active state, verifier rows, prefill chunks, expert union/padding, output credits, streaming tails and tenant fairness, no unbounded queues.
- Joint finite prevalidated Pareto selection for batch/prefill/graph/kernel/SSD-hotset/optional drafting, not exhaustive online compiler search.
- Match current vLLM V1/MRV2 source/profile/hardware/perf in actual task throughput, TTFT, true ITL P50/P95/P99, offered-load rejection, concurrency/fairness and memory/energy.

Exit: workload-real native concurrency and correct state/output, reproducible complete-request benchmark with named winning AND losing cases; no blanket win statement.

## 4. E03 — full multi-model residency and source-aware routing

- One shared model catalog, immutable alias generation, replica groups, source fingerprint, valid TaskDriver and per-replica prepared executable plan. Multiple models/aliases in one process share physical budget without separate per-model server.
- Lazy model load, warm reuse, optional source-qualified unload/eviction/prewarming, component residency, exact request alias pin while hot reload, independent per-session state and output rights.
- Source-aware routing between eligible equivalent replicas/placements; persistent sessions cannot silently change quant/adapter/source/numerical.
- GPU/CPU/UMA and host/SSD complete peaks, OS headroom, multi-domain all-or-none reservation as actual consumer emerges; fairness across tenants/models.
- Serving API handles many qualified models with independent caps, not one Qwen sample model or forced process-per-model.
- Model-catalog metadata targets remain intent only; exact runtime capability receipt required.

Exit: two genuinely different trained model families source-qualified concurrently through same embedded SDK+HTTP and one ResourceArbiter.

## 5. E04 — large dense 27B, first-class SSD and virtual tensor residency

- Real Qwen3.8 27B original pinned indexed shards, complete model config/tensor roles/axis, tokenizer, BF16 model-level logits and all state.
- Bounded verified immutable SourceWeights/source windows, true resident/streamed owned extents (source hashes, FD mutation refusal, partial short read), CPU/GPU first-use event and reader leases.
- Physical O(1) source identity and direct fully-resident execution, optional SSD source window fault/prefetch with actual IO+DMA, no hot-path virtual page table fetch unless required.
- Dense layer-major streaming and activation spill optimized by physical bandwidth and source math, not incorrectly treated as sparse MoE experts. Startup, page cache and UMA memory effects measured.
- Honest capacity vs usable tokens/sec; streaming allows fit when safe, does not promise fast decode if every layer rereads huge weights.
- Later Qwen Flash large learned lookup/experts/draft uses same storage/IO substrate but different policies.
- Local multiple GPU layer split only with exact composed source/state/numeric/physical transfer and cost proof.

Exit: at least one source-exact large dense model across actual admitted memory/SSD envelopes, with full logits/StateRoot and honest cold/warm performance.

## 6. E05 — Gemma4 models and novel architectures

- Gemma4 E2B model-quality/PLE/attention/KV sharing/softcap exact source, CPU reference and independently qualified CUDA/HIP/Vulkan original model paths.
- Gemma4 26B-A4B MoE first, dense 31B after, exact source routing/shared expert/state schemas and boundaries; do not treat MoE and dense as one numerically interchangeable model.
- Expand to Llama, Mistral, GLM, DeepSeek, Phi, Kimi and released actual hybrid/sparse families only with verified artifacts and correct new ops/attention/SSM/quant; metadata target registry is not capability.
- Model integration drill: new architecture using existing semantic ops touches only its model family, one registration and tests. Novel stateful mathematical op extends exact core + CPU reference + optional provider; no server/scheduler/allocator edit.
- Held model payloads (DeepSeek/GLM etc.) remain held until explicit rights/authorization.

Exit: new real model paths demonstrate small change radius and same public SDK/server/session resource infrastructure.

## 7. E06 — future source encodings, quantization, compression and extreme low bit

- Implement exact released Q1/Q2/Q3/NVFP4 variants (bitstream/scales/layout/finite exceptions), distinct CPU reference and physically tested CUDA/HIP/Vulkan kernels; no generic "Q3" format without exact variant.
- Complete Q4_K source-exact CPU Q8_K dot and GPU kernels with real source numerical profile, separate from existing reference scalar Q4_K decoder.
- EXL3/QTIP/trellis/codebook/3.5-bpw source encodings and any other derived quant families: exact source tensor manifest, codebook/basis/bias/shape groups, original source license, offline deterministic conversion and independent held-out quality.
- Distinguish storage compression, reversible device packing, activation/KV temporary quant, lossy derived checkpoint, approximated attention or expert-budget changes and source sampler distribution. User expressly selected approximation profile may trade quality; no silent fallback from exact.
- Tune actual direct quant GEMV/GEMM batch/shape/cache, avoid mandatory gigantic FP16 dequant and measure that lower bpw can be slower due to decode/trellis tiling overhead.
- Source format and numeric profiles remain separate from model family, physical device layout and serving tasks.

Exit: exact per-codec support matrix, end-to-end source/derived quality and performance with real hardware/driver, no format-specific engine.

## 8. E07 — hybrid sparse models, expert/lookup streaming and speculation

- Qwen Flash-Next source-real recurrent GDN/KDA, QSA/sparse masks/indexer, MoE, n-gram/Engram trained lookup tables, independent MTP/drafters; verify all auxiliary sources/sidecars and memory.
- GLM/DeepSeek/Kimi other attention variants only after source equations, independent CPU reference and sampled task quality.
- Expert cache grouped model-selected routing IDs, lookup row IO and shared finite byte budgets; physical prefetch/eviction pins captured readers, SSD/UMA, CPU/GPU overlap as actual cost.
- Target-authoritative speculation, draft acceptance and full rollback for recurrent, KV, sparse indices, grammar and RNG, not KV-only; wrong drafter rejected all tokens can still yield correct target output, so drafter separately validated.
- Joint scheduler considers target verifier rows, acceptance probability, drafted sidecar weight+state memory, unique expert union and bandwidth/IO, not only accepted tokens.
- Support exact target distribution for required sampler mode and separate explicit approximate policies.

Exit: new hybrid model full semantic/state/source/batch/memory and complete-request quality/performance proofs; no new inference engine.

## 9. E08 — native multimodal task drivers and model assemblies

- Real encoder/embed/rerank/decision task as first non-autoregressive proof through current model-neutral runtime and typed task output; no second token scheduler.
- Source-verified ModelAssembly components for Qwen Image, encoder/transformer/scheduler/VAE, actual image/video diffusion/flow stages and typed latents.
- Source-real ASR: Whisper/Canary/Parakeet/Qwen ASR exact audio processors, windows, CTC/RNN-T/TDT/AR stage semantics, timestamps and transcript finality, speaker/diarization only if source-qualified.
- TTS: phoneme/prosody/acoustic flow, RVQ/codec depth, vocoder frames, priming delay, speech audio and model-specific timing; text→PCM cannot be fabricated by generated JSON.
- Native realtime audio/video and synchronized AV, duplex barge-in, capture clock, rational timebases, codecs and output credits; distinguish committed, delivered, ACK and played; no guaranteed human attention from network receipt.
- TaskPipeline of independently registered ASR→LLM→TTS stages on shared memory/device region links, type-checked transforms and staged commit groups. ModelAssembly internal stages remain distinct from cross-model task pipeline. No general tool/agent workflow DSL in engine.
- Public SDK supports tasks/typed event streams without HTTP; standalone serving adapts OpenAI/Responses/WebSocket/WebRTC protocols only as each model/task/transport is independently qualified.

Exit: real trained source task non-token stage and one compound pipeline with source-valid output, physical liveness and complete timing; architecture extended without new engine.

## 10. E09 — heterogeneous and distributed compute, state and memory

- GPU+XDNA2 NPU same-SoC safe partial projection row split/cooperative prefill only with correct source equations/output reduction and known power/driver safety. No hidden root clocks, suspend/IOMMU tuning or unsafe gfx90c GPU resets.
- Multi-GPU layer/tensor/expert/data-parallel qualified physical execution, graph-only device program/AOT/static dataflow with true external program lifecycle and state bridge, not a pretend CUDA raw kernel.
- Four placements compute, parameters, mutable state, reusable immutable cache, plus TransferPlan for state/copy/IO/collective and frozen PlacementEpoch.
- Trusted remote peer enrollment with per-source/tenant authorization, authenticated region-execution worker, actual bandwidth/RTT, epoch/fault/fence/unknown and private session root, no arbitrary remote data egress or mandatory local RPC.
- Disaggregated prefill/decode must transfer *all required hybrid state*, not merely KV, and choose measured QoS isolation over assumed throughput; remote cache hydrate vs remote stateful operator vs RDMA reads separately costed.
- No distributed consensus/HA or true exactly-once client delivery until independently implemented, with lost ACK/reconnect failure cases.

Exit: source/task/model quality parity across a genuinely qualified multi-device program with measured performance and failure safety, while local direct run remains simple.

## 11. E10 — autotuning, release and whole-system evaluations

- Bounded offline kernel/capture/shape/quant tuning with negative experiments and saved qualified evidence keys; never a giant per-token search.
- Hot path cost model includes compiler warmup, source bytes, physical resource stalls, latency and QoS fairness; full-task critical path and accepted-goodput objective.
- Compare selected models/tasks/numerical policy/hardware against pinned llama.cpp, vLLM V1/Model Runner V2 and vLLM-Omni baseline with matched actual source/quant/tokenizer/driver/context and performance class. Publish losing and neutral cases.
- Optional parameter-version/gradient/update TaskDriver future seam without training/autograd as mandatory inference dependency; immutable active sessions pin model version.
- CPU-only/optional-GPU/serving crate/Rust SDK + single binary, Nix/build/cross installer/release/license/security and docs/whitepaper updated only from real source/qualification.
- Security: untrusted source parsers, no unknown model Python, opt-in external listener auth/TLS, tenant/source privacy, snapshot integrity, network rights and retained GPU reader safety.

Exit: verified release and research evidence, not only a specification.

## 12. Crosswalk with existing P00–P23 program

Do not create a second implementation queue that loses original open tasks. Once transition is expressly implemented and accepted, map later E phases back to owning actual existing plans:
- P00–P08: source, math, artifact, semantic program, providers and first model;
- P09/P16: serving, scheduler, API and goodput;
- P10–P12: state, SSD, sparse/hybrid resource execution;
- P13–P15: extreme hybrid families and drafting;
- P17–P18: broad models and multimodal TaskDrivers;
- P19–P20: heterogeneous/distribution/autotune;
- P21–P23: security, release, verified performance/extension.
A plan number is NOT a capability state. Existing plans/TODO owners stay in effect; side plan only integrates architecture/ordering for future approved source work.

## 13. Ordering rationale and stop conditions

Why skeleton/engine first: without one real public Runtime and serve crate, adding codec/model/media features risks more CLI/model-specific dispatch and duplicated source/session code. Why actual source math stays first: general API structure without matching mathematical and physical state does not confer correctness. Why heavyweight qualification deferred: user explicitly prioritizes a rapid coherent transition and one integrated full test; per-phase checks still prevent broken code. Why source/model expansion later: it is independently large and frequently requires licensed model/device.

Halt any new capability if it:
- creates another inference runtime, cache/spendable physical ledger or StateRoot publisher;
- depends on test/fixture branch for ordinary execution authorization;
- silently substitutes source/quantization/approximation or treats old physical proof as fresh;
- introduces per-token model registry/compiler/JSON/allocator global contention;
- interprets audio/image/video work as fake text tokens;
- violates approved hardware/device/peer/source holds;
- cannot be accessed from public Rust zerogravity crate or requires model-specific serving implementation.

The next execution agent must implement the structural/foundation first (S0–F4/Q0–Q2), only then continue this E-series roadmap as independently approved capability development.

## END FILE: .agents/plans/fabric-vnext-transition/ROADMAP_AFTER_TRANSITION.md


---

## BEGIN FILE: .agents/plans/fabric-vnext-transition/TODO.md

# Fabric vNext transition — held phase checklist and next-implementation owner plan

Author: Project owner
Co-authors: GPT-6
Owner: @pvl — successor coordinator not yet appointed

**Scope:** all future code phases held pending a new explicit implementation instruction. This plan was created as documentation only and is not promoted to the active .agents/plans/TODO.md queue. Initial read-only source HEAD fadf2b1123309a2118725f2cc005d3edc7ea39f2, moving; base for implementation is NOT frozen. No plan commit has been made.
**Reference authority:** HANDOFF.md → README.md → PHASED_IMPLEMENTATION.md → STRUCTURAL_SKELETON.md; old preserved ../fabric-vnext entire packet.

## Committed

None. No code/refactor/source/test commit authorized during plan writing. When implementation is separately approved, append logical signed SHA and acceptance evidence here or reconcile into current owning plan per AGENTS.md. Historical implementation commits are not new Fabric S/F/Q completion proof.

## Uncommitted — future S0 COMPLETE architectural skeleton FIRST

- [ ] User explicitly authorizes actual source implementation and integration owner; re-read live HEAD, source status, active owners/worktrees, AGENTS.md and all referenced docs; preserve unrelated work.
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

## END FILE: .agents/plans/fabric-vnext-transition/TODO.md


---

## BEGIN FILE: .agents/plans/fabric-vnext/HANDOFF.md

# Fabric vNext — complete next-agent handoff and starting prompt

**Date frozen:** 2026-10-27.
**Status:** DOCUMENTATION / REFERENCE ONLY. The user explicitly said to finish writing and report completion *before* any next phase. Do not initiate resync, source refactor, test, download, commit or deployment from this plan until their later instruction.
**Project:** zerogravity, /home/pvl/spaces/abird/src/zerogravity on connector **dotlink-l5** (pvl-l5).
**Initial 27-Oct archival-phase observed HEAD:** eb38dd5e892f2120ac29cc2fb79b2dbe137a37fe, worktree initially clean; this observation is provenance ONLY, not a readapted architecture review.
**Historical in-depth 26-Oct comparison checkout:** 195099ca583ace766d075682927f590360c823a5 plus a large moving worktree; source paths, file counts and status in comparative research refer ONLY to it.

## 1. Why this archive exists

The user previously requested:
1. In-depth source file-tree and architectural comparison of llama.cpp and vLLM then zerogravity, with new-model integration and what to adapt.
2. A second zerogravity design-validation reading .agents/docs and .agents/plans and proposing a clean next-gen AI execution fabric including llama.cpp/vLLM lessons.
3. Explicit approval to break all pre-v1 internals and redesign any part of zerogravity if better.
4. Architectural endorsement subject to mandatory refinement: make serving an independently reusable library and expose *everything* through exported public zerogravity Rust library; CLI must only consume this library for CLI/presentation duties.
5. Create a completely self-contained side planning reference with all discussion details, preserving exact language where materially important, and a handoff for future LLMs. **Do not resync or implement against the changed checkout yet.** User says they will separately initiate the next phase when ready.

This archive fulfills #5 and fully describes #1–4; the other phases have not been initiated. No code modification or architectural source adaptation is implied by these texts.

## 2. Mandatory reading sequence for the next agent

1. This HANDOFF.md — latest user decision, no-edit gate and exact next step.
2. [DECISIONS_AND_OPEN_QUESTIONS.md](DECISIONS_AND_OPEN_QUESTIONS.md) — verbatim user instructions, accepted vs recommended, rejected designs, unresolved API choices.
3. [README.md](README.md) — complete proposed Fabric vNext architecture including model/task/compiler/state/physical/memory/serving/multimodal/distributed.
4. [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md) — **mandatory new standalone zerogravity-serving + all public zerogravity API + thin CLI**.
5. [PRIOR_ART_AND_REVALIDATION.md](PRIOR_ART_AND_REVALIDATION.md) — upstream llama.cpp/vLLM source trees, prior design research, local historic architecture mapping and full original document crosswalk.
6. [IMPLEMENTATION_AND_ACCEPTANCE.md](IMPLEMENTATION_AND_ACCEPTANCE.md) — staged future migration, no-edit resync, exact acceptance and performance/correctness gates, risks.
7. [TODO.md](TODO.md) — held work, never treat as part of active queue unless coordinator later promotes it.
8. Existing source-of-truth instructions only when future resync is expressly requested: AGENTS.md; .agents/plans/README.md; .agents/plans/TODO.md; .agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md; relevant active TODOs and normative .agents/docs/ contracts (see PRIOR_ART_AND_REVALIDATION).

A combined single-file full-reference archive may exist under .agents/docs/FABRIC_VNEXT_COMPLETE_REFERENCE_2026-10-27.md. It is a lossless concatenated copy of the detailed plan contents for convenience; individual plan files remain the editable authoritative sections in a later *separately approved* documentation phase.

## 3. The latest architectural mandate

The core goal:

> A semantic, task-aware, hardware-adaptive Rust execution fabric with one runtime and composable qualified specialization, where models define exact mathematical programs, providers implement verified hardware-specific regions, schedulers choose compatible work, one ResourceArbiter owns physical admission, and each session has one authoritative StateRoot publisher.

**Additional binding user requirements (most recent):**
- **zerogravity-serving must be a separate reusable Rust library**, not server logic embedded in zerogravity-cli.
- **The public zerogravity library must expose everything** needed by other Rust programs: loading, inference, model registration/admin/cache/pull, device introspection, pipelines, runtime/scheduler observability, cancellation, serving start/stop/configuration and media/remote task APIs as they are qualified.
- **zerogravity-cli only uses the public zerogravity library** and owns command line argument interpretation and terminal/presentation behavior.
- All prior internal types/crates/modules/APIs can change pre-v1 for a genuinely better design. Do not assume compatibility with old names.
- Actual model quality, source integrity, state publication, physical events/leases, resource budgets and user rights remain inviolate safety/quality claims.
- Serving requests and SDK direct requests share one engine/source/state/resource stack.
- One small SemanticProgram and one ExecutableBundle; avoid trait zoo, second inference engine, per-token compiler overhead.
- Multiple models, task modalities, CPU/GPU/NPU/SSD/peer heterogeneous execution are composable goals with independently truthful qualification.
- Next phase is *first review new current source*, not blindly transplant old findings.

## 4. How to use in the next conversation

**If user merely asks about this archive:** quote/find files, explain decisions. Do NOT resync source or start implementation.

**If user says "we can begin resynchronization":**
- perform read-only live source checkout review with dotlink-l5, compare with frozen design, update findings and present proposed adaptation;
- inspect recent repo work/commits, active other agents, Cargo dependencies, model math, source identity, state, workers, tests and serving crate status;
- decide whether design modifications are needed, preserving endorsed requirements;
- provide an "already done / still needed / newly conflicts / revised plan" analysis;
- do NOT begin code changes unless later instruction expressly asks to implement.

**If user expressly says "implement after resync":** reconcile current source and design, then implement in smallest reviewable logical units, preserving active agent ownership and accepted test evidence. Follow AGENTS.md and designated owner coordination; do not alter other worktrees, git history or push without explicit permission.

## 5. Ready-to-paste next-agent prompt (resynchronization-only)

~~~text
Use the dotlink-l5 connector and work READ-ONLY in
/home/pvl/spaces/abird/src/zerogravity.

We previously completed a major architectural research/redesign for zerogravity:
llama.cpp vs vLLM source trees, vLLM Model Runner V2 and vLLM-Omni lessons,
zerogravity's full v4/v5 architecture goals, and a Fabric vNext design.

The user explicitly approved breaking any pre-v1 APIs if a cleaner architecture
results, and required TWO final refinements:
(1) serving lives in its OWN reusable zerogravity-serving Rust library;
(2) ALL useful functionality, including serving lifecycle, is exposed through
the public zerogravity Rust library. zerogravity-cli only owns CLI parsing and
presentation and calls that public library. NO server or model engine in CLI.

Start by reading ALL of the frozen design context:
.agents/plans/fabric-vnext/HANDOFF.md
.agents/plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md
.agents/plans/fabric-vnext/README.md
.agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md
.agents/plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md
.agents/plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md
.agents/plans/fabric-vnext/TODO.md
and the combined full reference under .agents/docs if useful.

Then read AGENTS.md, .agents/plans/README.md, .agents/plans/TODO.md,
the CURRENT normative architectural contracts, and the actual CURRENT
source, tests, Cargo dependency graph, features, Git history and working
tree. Do not assume that any historical 26 October file anchors or tests
still reflect the new checkout. No files may be overwritten, reverted or
changed by this read-only phase.

Do a comprehensive RESYNCHRONIZATION of the current zerogravity state
against the frozen Fabric vNext design. Identify which parts are already
implemented, which should be retained, moved, simplified or discarded,
and which need redesign. Explicitly check:
- family source/math vs generic SemanticProgram
- single source-valid state publisher and physical resource authority
- one qualified physical executable path and real Ready/leases
- GPU import direction and specialized kernel source ownership
- model-neutral engine crate and embedded Runtime/API
- standalone reusable zerogravity-serving library
- thin zerogravity-cli consuming only public library
- scheduler/persistent batches/continuous and hybrid state
- source, quant, memory/SSD, task/multimodal and distributed targets
- numerical/fault/performance qualification and honest capability status
- user pre-v1 breaking-change permission and newest serving SDK mandate

Deliver:
1. source-grounded current architecture diagram and Cargo dependency graph;
2. changed/current files and precise differences from the frozen reference;
3. what is already compliant, what is not and why;
4. refined smallest coherent architecture and module ownership plan;
5. safe staged migration/revalidation/acceptance gates;
6. unresolved design decisions and any justified modifications;
7. complete preservation of source/test/negative evidence without
   false support or benchmark claims.

DO NOT IMPLEMENT OR EDIT CODE in this resynchronization phase.
Wait for the user's next explicit instruction to begin implementation.
~~~

## 6. First implementation prompt template (ONLY after separate future authorization)

This is NOT permission to run now; retained for a later user-triggered phase.

~~~text
Continue at /home/pvl/spaces/abird/src/zerogravity using dotlink-l5.
Read the complete Fabric vNext reference and the approved UPDATED resync plan.
The user has explicitly authorized the implementation phase in this turn.

Make the smallest coherent source changes that establish the approved
model-neutral engine, library-first public API, standalone reusable
zerogravity-serving library, and truly thin CLI, while preserving all
source/numerical/physical/state safety evidence and independent backend
capability gates. Work with current code, not historical source paths.

Do not create a second allocator, model server, TaskCoordinator/StateRoot
publisher or inferred model permit. No giant trait zoo, hot-path compiler,
new per-family model runner or mandatory server dependency for direct run.
Separate genuine source/family math from physical kernels/ABI.

Validate affected crate graph/source/Cargo features/CPU tests/selected native
source and serving cases, report incomplete hardware/benchmark scopes.
No push, release or destructive Git operation absent separate explicit
permission. Follow AGENTS.md and active plan ownership rules.
~~~

## 7. State of the archival work

- This is **design documentation only**, not code implementation.
- No historical SOURCE/CUDA/HIP/ROCm/Vulkan acceptance or vLLM perf claim was upgraded by the review.
- No new builder/server/SemanticProgram/TaskDriver was compiled by this archive.
- Existing active TODO entries, owner run files, source artifacts and driver settings were not modified.
- The appropriate user response after archive verification is ONLY: "The reference and handoff have been written. Here are the paths. No implementation or resynchronization was started." User decides when next phase begins.

Preserve this explicit boundary.

## END FILE: .agents/plans/fabric-vnext/HANDOFF.md


---

## BEGIN FILE: .agents/plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md

# Fabric vNext — decision ledger, rejected alternatives, open questions and conversation provenance

**Frozen 2026-10-27. Reference only, no changes to the active implementation.** This file distinguishes explicit user decisions from design recommendations and implementation hypotheses. It includes essential exact user wording so future agents can recover the requested intent even without this chat.

## 1. User instructions in original chronology

### 1.1 Initial comparative research request (user, 26 October 2026)

> Do deep research on the file tree and overall architecture of llama cpp, vllm and then zerogravity.
>
> And then, compare how they are structured, how the overall design is, how new model adding conducivness is across the 3, and then also, compare their file tree organization and architecurally where goes what..
>
> For zerogravity, you can use the connector [$dotlink---a1] to access. It's at /home/pvl/spaces/abird/src/zerogravity.
>
> Do zerogravity in the end.. we might want to compare with multiple versions of zg, so first, just do deep research on vllm and llamacpp and then the initial round with the current state of zerogravity against it's design targets.. but focus on detailed analysis of the other 2 first with reference to where we need to get zerogravity to, and what we can adapt for zerogravity from the others
>
> No code change anywhere.. Just give me a full comparison, wherever you can visually.

Correction during research from user:

> sorry, use [$dotlink---l5]

The correct active connector for the zerogravity source is **dotlink-l5**, host pvl-l5, repository path /home/pvl/spaces/abird/src/zerogravity. dotlink-a1/pvl-a1 did not contain that checkout on the research date. Do not assume that physical host availability remains unchanged; verify next time.

### 1.2 User requests full architectural revalidation (26 October)

> do another round of validation of the above, reading through all of the architectural docs in zerogravity, in .agents/plans, and ensure that the above is the best and most elegant design for achieve all of zero gravity's targets, and then redesign this cleanly as the next-gen ai execution fabric with all of the vllm and llamacpp's learnings and then design it fresh cleanly on what are the adaptions to make to current zerogravity

This authorized *research and conceptual redesign*, not implementation. The analysis included v4/v4.3 and v5.1 architecture, plans, task states, GPU event/source precision, multimodal and distributed aspirations, in addition to upstream source trees.

### 1.3 Crucial pre-v1 permission to break all APIs (user, 26 October)

> note that anything and everything is up for change if it brings up a better design.. we are not v1 yet, so breaking anything is fine

**Decision U-VNEXT-01 — ACCEPTED:** No compatibility constraints for internal pre-v1 APIs, crate layout, prior proposals, traits, enum variants, model registration or implementation, if a cleaner, more correct, more scalable and more maintainable design results. This does not permit discarding *correctness*, source/model provenance, physical safety, tests, licensing, active agents' work or source history. We should avoid compatibility facades for old broken abstractions merely to maintain names.

### 1.4 User acceptance and amendment (user, 27 October)

> excellent, love it.. some minor adaptations:
>
> - make sure that the serving bits, are also in it's own library, not just all dumped into zerogravity-cli.
> - everything should be usable from a zerogravity lib that can be exported out and used from other programs, so, ensure that it's architectured that way, so zerogravity-cli simply just uses that lib and owns the cli bits and presentation bits.

**Decision U-VNEXT-02 — ACCEPTED:** Create a standalone reusable **zerogravity-serving** library for wire protocols and listeners; it must not be buried in the CLI.

**Decision U-VNEXT-03 — ACCEPTED:** Build **library-first**. All usable functionality must be programmatically available through the public **zerogravity** Rust library, including serving startup, lifecycle, streaming/SDK events, management and all CLI commands. The CLI is a thin, presentation/argument adapter and owns no source/mathematical/routing/GPU scheduler/HTTP protocol implementation.

These two decisions supersede any previous vNext proposed file tree with HTTP server logic residing inside CLI and any partial facade that provides only model execution but not the rest.

### 1.5 User's handoff and phase-separation instruction (user, 27 October)

> with all of these, makes sure to write the full plan comprehensively on the side, without skipping any of the details above, word to word where needed, to make sure entirely of this context above is preserved. the goal is the full context above preserved.
>
> i mentioned on the side because, we've made lots of changes to the current state of the repo.. as a next step, we'll come back and resync with the current changes, and refresh and look into any adaptions needed, but for now, we need the above reference to be cleanly recorded. So add the above plans with full hand off,so that another LLM can read this and then continue this
>
> once you finish writing, let me know, then and only then will we start the next phase to readapt our selves to the new changes

**Decision U-VNEXT-04 — ACCEPTED:** The current deliverable is an exhaustive **side documentation archive and handoff only**. Preserve all prior architecture and research context in new files under .agents without modifying, reinterpreting or implementing against a moving source tree. The immediate future stage is separately initiated resynchronization/revalidation; DO NOT continue into that stage automatically after writing the archive. No source files, active plans, benchmarks, model assets, devices, Git branches or releases were requested.

## 2. Settled architectural conclusions from accepted redesign

The user broadly approved the prior vNext direction subject to the above amendments. These are architectural recommendations/targets, not a promise the present implementation already supports them.

| ID | Design direction | Why |
| --- | --- | --- |
| D-01 | One native Rust AI computation/execution fabric, no separate runtimes per model/backend/modality | Sharing correctness, resource and serving primitives |
| D-02 | One canonical source-verified mathematical SemanticProgram | Avoid divergent model loops/reference vs optimized |
| D-03 | One physical ExecutableBundle; RIR/KIR ephemeral, CIR embedded | No four independently persisted IRs/compilers |
| D-04 | Five primary authorities (model, compilation, session, resource, device) | Clarifies who extends what |
| D-05 | StateDomain Prepare, one TaskCoordinator PublishRoot per session/shard | Atomic logical state and failure safety |
| D-06 | Physical ResourceArbiter with local prepaid allocators | Correct global budgets without per-kernel central lock |
| D-07 | Source-specific models define math and semantics; physical provider implements kernels | New-model extensibility and backend neutrality |
| D-08 | Generic RegionExecutionProvider supports op, fused region, stage, whole model, partitioned program | Supports CUDA and graph/static devices without duplicate engine |
| D-09 | TaskDriver stage algorithm separated from mathematical semantics and physical schedule | True ASR/TTS/diffusion/video without token-loop fakery |
| D-10 | One admission/fairness system with specialized task work formation | Serve heterogeneous AI tasks efficiently |
| D-11 | vLLM-class continuous scheduling, paged hybrid state, persistent slots and async workers | Throughput and latency under concurrency |
| D-12 | Multi-domain StateRoot and complete common-frontier prefix reuse | Correct GDN/KDA/MLA/codec/draft state, not KV-only |
| D-13 | Four independent placements (compute/parameters/live state/reusable cache) plus TransferPlan | Unified VRAM/UMA/host/SSD/NPU/remote optimization |
| D-14 | ModelAssembly vs mathematical region vs multi-model task pipeline distinct | Correct cross-modality composition and ownership |
| D-15 | Model metadata/intent ≠ source/model/backend/task quality support | Honest trust and capability report |
| D-16 | Model original/derived quant/compute precision/approximation separate | No silent quality regressions |
| D-17 | No external workflow/tool privilege in mathematical TaskDriver | Safety/privacy and engine minimalism |
| D-18 | Built-in one-process catalog/router/residency with exact aliases, replicas and hot reload pinned at admission | No source switch or duplicate model-serving engines |
| D-19 | Local-only direct execution has no compulsory network/service/GPU dependency | llama.cpp-quality local UX and embedded APIs |
| D-20 | Source- and state-qualified whole-model native kernels valid optional providers | Peak hardware specialization without model engine duplication |
| D-21 | Built-in zerogravity-serving reusable protocol library | 27-Oct explicit user amendment |
| D-22 | Public zerogravity Rust facade exports all core/sdk/server/cmd capabilities | 27-Oct explicit user amendment |
| D-23 | CLI as parsing/presentation/signal entry point and facade consumer only | 27-Oct explicit user amendment |
| D-24 | Pre-v1 APIs/layout may be broken to simplify | 26-Oct explicit permission |
| D-25 | Archive first; next phase independently triggered resync | 27-Oct explicit phase separation |

## 3. Word-for-word architectural formulations to preserve

From the first comparative conclusion:

> The most important architectural difference is not the programming language. It is what each project treats as its fundamental abstraction.

> llama.cpp is the stronger reference for the lower execution layers; vLLM is the stronger reference for serving and workload orchestration. Neither should be adopted wholesale for zerogravity.

From the first vNext synthesis:

> The objective is not maximal modularity. It is minimal coupling with maximum composability.

> The most important discovery is that zerogravity's existing design documents already describe a considerably more capable system than either llama.cpp or vLLM considered individually. The challenge isn't inventing a larger architecture. It's turning the existing design into the smallest coherent execution fabric that can actually deliver all those capabilities.

> The desired destination is a system where models define source-faithful mathematical programs, providers implement qualified operations, schedulers select compatible work, and one resource/state authority owns execution safety.

> Its fundamental object should be a verified executable AI task, built from semantically defined components and mapped onto a qualified physical execution plan.

From the first analysis of vLLM's new model runner:

> Borrow vLLM's newer architectural direction, not its historical implementation complexity.

From zerogravity governing v4 architecture:

> A model describes mathematics; a task describes control; a provider supplies an implementation; a plan schedules computation and physical resources; a coordinator owns visible state.

> Only SemanticProgram and compiled ExecutableBundle need enduring execution representations.

From the user-approved latest refinement, restated as a hard requirement:

> Everything should be usable from a zerogravity lib that can be exported out and used from other programs ... zerogravity-cli simply just uses that lib and owns the cli bits and presentation bits.

These formulations prevent a future agent from narrowing the task to merely migrating server.rs or renaming Rust crate modules.

## 4. Prior design proposals that are now superseded

1. **Server in zerogravity-cli/src/serve** — forbidden as end-state. Protocol machinery is implemented in new separate zerogravity-serving library and publicly reachable via zerogravity facade.
2. **Public facade as merely inference library** — insufficient. Must also expose serving, model/asset/cache management, devices, pipeline, source/profiles, diagnostics, lifecycle and future protocols/placement.
3. **Retain existing crates by default** — not a hard rule. Reconfigure crates if dependency direction and ownership improved; recommend coherent engine and serving crates.
4. **Plain KV-cache-centric generalized state** — too narrow; complete multi-domain state and common-frontier restore is required.
5. **Single token scheduler for all tasks** — incorrect for non-AR/modalities. One global coordinator with specialized work-formation strategies.
6. **One monolithic GPU provider containing model math** — separate model recipe/source eligibility from backend/physical ownership, while allowing model-tuned kernel code.
7. **One IR language per compiler pass** — unnecessary; only SemanticProgram and ExecutableBundle enduring, RIR/KIR/CIR views.
8. **A model's parser returning success means model is supported** — false; independent trained source/backend/context/numeric/task gates.
9. **Per-model subprocess router as entire multi-model abstraction** — insufficient for intended shared one-process resource/memory/state/task fusion; optional isolation may later exist.
10. **General plugin/trait factory for every operator or buffer** — rejected until concrete extension proves need; static built-ins and coarse typed prepared handles preferred.

## 5. Tradeoffs and deliberately unresolved design choices

The user did NOT specifically freeze exact public Rust APIs, HTTP crate/framework, feature list, naming or C ABI. Those are design/implementation decisions for later resync and approval, consistent with accepted requirements.

| Open question | Candidate approaches | Governing decision/constraint |
| --- | --- | --- |
| Public Runtime naming/API | Runtime, Engine, SessionManager with builder | Must be exported through zerogravity and ergonomic for embedded Rust; do not pre-v1 freeze early |
| Dedicated engine crate | zerogravity-engine vs cohesive facade-private engine module | Recommend crate for enforceable dependency; adjust only if fresh source proves less code/complexity another way |
| Serving crate name | zerogravity-serving | Recommended coherent Rust crate; MUST remain reusable separate library |
| Server framework | axum/hyper/custom async stack | Choose by actual conformance, security, dependencies, perf, packaging; no forced CLI integration |
| Sync vs async facade | async-first plus bounded sync adapter | All functionality reachable, safe cancellation; no hidden global executor mandatory |
| Model registry form | static table/enum vs registration descriptor | One cold registration point, no model names in scheduler; dynamic executable plugins not mandatory |
| Crate feature topology | cpu default, serve optional, GPU backends independent | No accidental compiler/link dependency or false model/device support |
| State root representation | IDs/handles/Arc/RCU/internal ownership | One logical writer; no unsound device in-place mutation; optimize when measured |
| TaskDriver object form | enum/small trait/coarse function | No per-token vtable zoo or huge workflow interpreter |
| Region provider implementation | enum/static generic, coarse traits, AOT | One seam and honest fallback; no dynamic math per scalar |
| Compiler search | bounded opt candidates, offline autotune, cold shape bands | No JIT/e-graph in token hot loop, avoid variant explosion |
| Native model/kernel code ownership | family math, provider kernel bodies, specialization recipe | No GPU→family import, avoid duplicated device allocation in family |
| Global multiple Runtime instances | process-shared arbiter/explicit shared context | Same physical backing/domain must not be double-admitted |
| All-or-none physical admission | multi-domain transactional reserve inside existing arbiter | No second allocator/ledger, minimize locking, stop safely on partial failure |
| Prefix snapshots | cache frontier meet vs lazy replay and restorers | All participating states, source/tenant identity, no false partial KV hit |
| Device memory strategy | local pool/ring/arena different per backend | One backing ledger, generations/real reader retirement |
| CPU operator/math reference | independent host reference and optimized CPU | source exactness and reproducibility first |
| Realtime network transport | WebSockets/WebRTC/SSE, codecs | typed media clocks/epochs, real playback uncertainty |
| Embedded server startup/shutdown semantics | explicit ServerHandle, graceful drain | CLI-independent, model-state publication and lease safe |
| Administration API | Rust-only initially vs HTTP separate scope | Credential/tenant controls, never bundled into public inference without auth |
| Model load/residency eviction | explicit user policy + measured cost candidates | immutable source ownership, alias reload source pin, physical budget proof |
| Multi-device placement first candidate | contiguous layers, output rows, expert groups | source/number/bridge/network cost proof; avoid guessing and unsafe fallbacks |
| Remote peer enrollment | user-authorized static peers first | no automatic machine scanning, no implicit data egress |
| Training/parameter updates | optional future task driver | never core inference dependency; immutable model parameter version for existing sessions |
| Long-term public ABI | Rust crate API, possible C bridge/plugin ABI later | no pre-v1 ABI freeze without user decision and tests |
| User-visible default profile | correctness-first, automatic qualified fast choice | no silent source/quant/approximation changes |

## 6. Non-negotiable invariants and restrictions

- Source original model and any changed/derived quant artifact are different identities with independent quality.
- CPU/reference correctness, GPU kernels, cache/source and backends separately qualified; no universal epsilon from vLLM or llama.cpp.
- No request/StateRoot publication by device, IO, media transport or remote peer; one logical publisher per authorized session/shard.
- No capacity minting from observation, GTT virtual addressability, immutable content digest or an aliased CPU/GPU view.
- Recorded, captured, queued, submitted GPU source readers remain pinned until proved actual retirement, including unknown completions.
- A simulator, syntax check, test fixture, model slug, compiled branch or performance report never grants real hardware/format/model functionality.
- No mandatory Python/HF custom code, arbitrary user workflow callbacks, remote networking, system firmware/power change or untrusted model execution.
- No additional model-specific servers or token/state allocators.
- No per-token central global allocation lock, recursive task interpreter, CLI network JSON translation loop or dynamic optimizer search.
- User explicitly told us to archive only; no next phase until separately initiated.
- Do not modify active .agents/plans/TODO.md or other plan owner sections to track this held reference without coordination.

## 7. Revalidation of old findings before implementation

The 26-Oct file-tree inventory was read-only on a moving and very dirty branch. The 27-Oct archive began after HEAD advanced significantly. The current source may already embody some design recommendations or may have different ownership. Therefore:
- Do not auto-open a PR for all proposed files from the historic paths.
- Run a fresh dependency graph/current interface inventory in the later resync phase.
- Classify every proposal as already implemented, still needed, modified/rejected or conflict with new evidence.
- Re-run affected model/numerical/physical tests after source moves, preserving historical comparison documents.
- Decide final public API shape, crate graph and migration units from real current compilation boundaries.
- Never skip the library-first and serving-only user amendments simply because old code differs.
- Obtain separate user authorization before coding, committing, pushing, releasing, model download, physical device runs, or destructive operations.

## 8. Research/design authority and handoff status

**Reference authority inside this side plan:** README = complete target architecture; PUBLIC_LIB_AND_SERVING = newest mandatory amendment; IMPLEMENTATION_AND_ACCEPTANCE = staged future proposal; PRIOR_ART_AND_REVALIDATION = historic external/local evidence; DECISIONS_AND_OPEN_QUESTIONS = literal user decisions; HANDOFF = how to resume; TODO = held next phase work.

**Existing active repository authority:** AGENTS.md and .agents/plans/README.md, TODO.md and owning live TODOs govern ongoing implementation. We deliberately leave them unchanged this phase; this side proposal does not replace their current status or user approvals.

**Work status:** side documentation archive created 27 October; no source changes, no code build/tests, no new model/device qualification, no benchmark, no active plan resync. The only next step (once explicitly requested) is refresh and compare the changed repository before any implementation.

Full next-agent starting prompt and document read order: [HANDOFF.md](HANDOFF.md).

## END FILE: .agents/plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md


---

## BEGIN FILE: .agents/plans/fabric-vnext/README.md

Author: Project owner
Co-authors: GPT-6, architectural research and revalidation sessions (26–27 October 2026)
Owner: @pvl; future implementation coordinator to be appointed only after resynchronization

# zerogravity Fabric vNext — complete architecture reference

**Design reference frozen:** 2026-10-27.
**Status:** REFERENCE ONLY. NOT APPROVED FOR IMPLEMENTATION OR REBASE. NOT an executable capability declaration.
**User's latest architectural instructions:** anything may change before v1; serving MUST be a reusable Rust library separate from the CLI; ALL useful SDK, serving, model, pipeline, engine, and administration functionality MUST be accessible through the public zerogravity Rust library; the CLI MUST be only a consumer of that library and own parsing, terminal presentation, and OS command interaction. Save the design separately. On the next explicit phase, FIRST resynchronize with the changed repository, review changes and revise this design; do not implement now.

This file is the normative *proposed* architecture reference for the next conversation, not a retroactive change to existing implementation requirements. The companion reports deliberately retain preceding research, counterexamples, original options and exact user decisions. The entire folder is self-contained for an agent without chat history.

Read [HANDOFF.md](HANDOFF.md) first for the next-agent workflow, [PRIOR_ART_AND_REVALIDATION.md](PRIOR_ART_AND_REVALIDATION.md) for source-level llama.cpp/vLLM/current-zerogravity evidence, [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md) for the 27 October public-library amendments, [IMPLEMENTATION_AND_ACCEPTANCE.md](IMPLEMENTATION_AND_ACCEPTANCE.md) for staged migration, [DECISIONS_AND_OPEN_QUESTIONS.md](DECISIONS_AND_OPEN_QUESTIONS.md) for commitments versus options, and [TODO.md](TODO.md) for held work. An optional single-file combined copy is referenced from the companion index in .agents/docs.

## 0. Mission, scope and fidelity

**Positioning:** zerogravity is a **semantic, task-aware, hardware-adaptive Rust AI execution fabric**: one native execution substrate spanning local inference, performance-specialized GPU execution, model routing, concurrent serving, model composition, multimodal generation/perception, memory/storage tiers, and eventually trusted heterogeneous/distributed placement. It is not three independent "portable", "vLLM-like", and "extreme" inference engines.

Preserve these project statements in meaning:

> Run any *supported and independently qualified* model locally with the single-binary Rust execution fabric for text, image, video, audio, real-time, and combined multimodal inference. Model-specific mathematics, hardware kernels, memory placement, scheduling, and serving compose under one engine—not a separate runtime for every model or modality.

The phrase "any model" is an ambition, not permission to execute unknown mathematics or arbitrary downloaded HF Python. A filename, model registry entry, tokenizer or parsed weight file is not proof of functional model support. User-facing output must state the exact model+checkpoint+quant+task+shape+device+numeric+quality status. A fast path must never silently substitute a different checkpoint/approximation.

The project was previously designed around a semantic compiler, state-effect system, native Rust kernels, global physical resource authority, CPU/GPU/NPU/peer workers, source-trusted assembly, and a TaskCoordinator that publishes only source-valid state. That direction remains, but pre-v1 every internal API, module, name, crate layout, registry and implementation is permitted to change if simplicity, soundness, composability, performance or maintainability improve. Retain *proven behavior and test evidence*, not awkward compatibility facades.

**Scope inherited from the complete preceding discussion and designs:**
- Dense, sparse, MoE, GQA/SWA/MLA, GDN/KDA/SSM, n-gram/Engram, mHC, learned draft/MTP, explicit speculative inference and independent state restoration.
- Reference CPU execution plus CUDA, ROCm/HIP, Vulkan, future Metal, NPUs such as XDNA2, graph-only/AOT accelerators, statically scheduled fabrics and permitted remote peers.
- Original GGUF and safetensors/other verified sources; multiple model component assemblies; exact codecs and optional user-authorized derived low-bit/approximate sources.
- GPU VRAM, discrete HBM, coherent UMA, ordinary host RAM, pinned RAM, page cache, SSD and remote storage; physical capacity and real transfer coherency matter, not virtual address space claims.
- High throughput: continuous batching, chunked prefill/decode coexistence, persistent request slots, paged state, prefix reuse, fused/captured/model-wide specialization, dynamic MoE hotsets, optional speculation, tail latency/fairness, quality-preserving performance.
- Image generation/editing, video, embeddings/reranking, typed finite decisions, ASR, TTS/neural audio, native full-duplex and synchronized real-time audio/video; distinct task drivers and typed media clocks.
- One-process multiple model instances and replicas; one model can be partitioned across devices; many typed task pipelines can share immutable source/model instances; original state remains session-private.
- CLI, stable embedded Rust API, SDK integration, HTTP OpenAI-compatible APIs, Responses, streaming, WebSocket/WebRTC/media eventually, model admin/control surfaces, trustworthy observability. **Every programmatic function must be reachable through the public zerogravity library; serving is a separate reusable library.**
- Local-first runtime with optional remote placement under explicit user trust/authorization, cost, link and epoch proofs; no mandatory coordinator RPC for local execution.
- Source/semantic/numeric/state/device/transport licensing and safety first; publish neutral/negative benchmark findings and unsupported capability cells.

**Not goals for the initial proof:** running arbitrary Python/C++ from model repositories, shipping a new GPU driver, mandatory heavyweight training/autograd, arbitrarily extensible agent workflow DSL, separate optimizer daemon, public unstable plugin ABI, hidden hosted inference pretending native kernels, or forcing distributed execution into the local hot path.

### 0.1 Design ordering

1. Semantic and task correctness, exact source identity.
2. Numerical/quality equivalence under declared exact or selected approximate policy.
3. Real physical resource lifetime, asynchronous failure safety, one logical committed state.
4. User-facing correctness, privacy, usability and clear refusals.
5. Sustainable extension boundaries and reproducibility.
6. Whole-request/service goodput, TTFT/ITL/P95/P99, memory/energy.
7. Peak specialized performance on proven compatible model/device/workload.
8. Complexity only with a correctness proof or measured benefit.

The target is NOT to implement the most architecture nouns, nor to win an unrelated tokens/s benchmark. It is to execute useful source-valid AI work with the lowest defensible end-to-end cost.

## 1. Research synthesis and what changes from the preceding design

**llama.cpp's reference strength:** tensor graphs, per-model architecture builders, small public inference API, GGML backend interface/scheduler, CPU/GPU portability, backends selected below model mathematics, and a mature simple local executable experience. Avoid a global proliferation of hard-coded architecture enum/metadata switch edits and a GGUF-conversion prerequisite for all sources. Its multi-model router can manage model child processes, but that is not the same as zerogravity's shared resource, one-process, task-aware fabric.

**vLLM's reference strength:** V1 EngineCore separates request scheduling/cache planning from model execution; explicit scheduler outputs; KV blocks and prefix reuse; persistent batches, async workers and GPU model runners; reusable neural layers, model registry and out-of-tree model extension; mature OpenAI serving. Model Runner V2 redesign recognizes overly complex old model runners and targets stable slots, incremental GPU state/metadata, async dispatch and a minimal shared runner. Avoid assuming PyTorch/CUDA is the one foundation, treating request-slot state as the sole kind of state, or importing an ever-growing model-specific branch hierarchy into serving.

**vLLM-Omni's additional lesson:** modality-specific stages, coders/decoders, diffusion and other tasks require their own work units, control loops and timing. Reuse scheduling/resource/transport infrastructure but do not pretend all work is the next LM token. A logical multi-stage pipeline does not require a separate model server per stage.

**zerogravity's distinctive design strength:** exact source/numerical profile, semantic effects, two enduring compiler representations, a generalized physical region provider with optional model-wide fusion, one resource authority, multi-domain StateRoot publication, independently placeable compute/weights/state/cache and typed media frontiers. This goes beyond either reference architecture as a unified abstraction but is only an architectural hypothesis until tests and workloads demonstrate it.

**Corrections to the earlier assistant proposal:**
- Original file-tree plan retained crates by default; user's later pre-v1 permission authorizes breaking any layer or crate. Recommend a **model-neutral engine crate** enforcing strong dependency direction.
- The original vNext proposal left HTTP/server code near the CLI. **This is superseded**: introduce a reusable **zerogravity-serving library**; make zerogravity CLI only argument/config entry, invocation and presentation.
- Original vNext proposal said library/facade should assemble capabilities but did not clearly require all functions through it. **This is superseded**: public **zerogravity library is the single programmatic SDK and stable conceptual entry point**. Every capability including serving configuration, lifecycle, requests, events, introspection, model administration and future media/distributed APIs goes through it. Advanced native crate APIs can remain implementation details; no Rust consumer should need CLI imports or fork.
- Make **global scheduling and physical resource budgeting** foundational, not retrofitted after text CPU reference.
- Generic state is not just paged KV. Model states include recurrent, sparse/indexer, sampler/RNG, grammar, media clocks and codec/solver states.
- Distinguish mathematical graph composition, one verified multimodal model assembly, and a task pipeline connecting separate models. All can be typed, but they do not share a recursive interpreter or logical state publisher.
- No blanket one global token scheduling algorithm. One global policy/admission system and stage-local work-forming algorithms.
- Source-specific whole-model GPU specializations can be first-class closed-region implementations; their math and eligibility live in families, actual kernels/compilation/physical lifetime below shared providers.
- One physical plan with four independent placements; do not add four separate placement schedulers or a new distributed engine.
- Exactly two architecturally enduring execution representations: SemanticProgram and ExecutableBundle. RIR/KIR/CIR are optimization views/embedded command DAG, not parallel serialized runtimes.
- Explicitly preserve library-first API and delivery output-state separation, including services started from embedded programs, not only shell.

## 2. The architecture — one engine, five authority domains

Conceptual user-to-execution flow:

~~~text
               CLIENTS / PROGRAMS
  Rust application  CLI  OpenAI HTTP  Responses  WebSocket  Realtime
         \           |           |            /
                     |
                 zerogravity
      public SDK, built-in registry, typed builders
                     |
            zerogravity-serving
     reusable optional protocol servers / client sessions
                     |
   RESOLVED REQUEST + EXACT REGISTERED MODEL/TASK
                     |
       MODEL CATALOG / ROUTER / ADMISSION
                     |
       TASK COORDINATOR (session authority)
      TaskDriver | scheduling policy | StateRoot
          /                           \
  VERIFIED MODEL ASSEMBLY        CURRENT RESOURCE
  + original artifacts           and topology facts
          \                           /
            SemanticProgram (typed exact meaning)
                     |
       REGIONS / QUALIFIED COMPILER CANDIDATES
                     |
               ExecutableBundle
     guarded code+graphs+resource/transfer templates
                     |
         SCHEDULER + PHYSICAL RESOURCE ARBITER
                     |
             WORK ENVELOPES / SLOTS
                     |
 CPU | CUDA | HIP | Vulkan | NPU | IO | trusted remote workers
                     |
  ACTUAL READY/FAIL/UNKNOWN + PREPARED PRIVATE STATE
                     |
         SESSION AUTHORITY PUBLISHES ONCE
                     |
    COMMITTED OUTPUT => BOUNDED DELIVERY + ACK/PLAYBACK
~~~

This is a logical contract flow. Static validation/preparation occurs on source admission, cold model load or off-critical-path tuning, not every generated token. Logical authorities are NOT one thread/process/trait/crate each.

### Five primary authorities

1. **Model/semantic authority.** Owns exact original artifact/assembly, model family config, tensor roles/ties, tokenizer/processor, mathematical definitions, state-domain schemas and source-qualified conversions. It does NOT own device memory, HTTP, routing, task publication or user process privileges.
2. **Compilation/plan authority.** Owns semantic legality, pure-region fusion/rewrites, guarded region candidate selection, executable preparation, shape bands, code cache and physical plan generation. It does NOT own mutable session frontier or pretend compilation is physical success.
3. **Logical session authority.** Per TaskSession or authorized shard one coordinator/writer owns TaskDriver progression, expected StateRoot, epoch, cancellation, provisional → committed transition and committed output sequence. No kernel, transport or individual StateDomain may publish the whole session independently.
4. **Physical resource authority.** One global physical ResourceArbiter accounts for shared addressable/physical memory domains, committed and future liabilities, DMA/IO/graph/software/OS headroom, quotas and coarse all-or-none reservations. Device-local pools allocate fast under prepaid envelopes, not under a global lock per op.
5. **Device/IO execution authority.** DeviceWorkers, IOWorkers and qualified RemoteRegionWorkers own driver contexts, code images, local pools, stream/event/transfer submission and true Ready/Failed/Unknown completion, fencing and retirement. They return private candidate state and receipts, never directly a user-visible TaskRoot.

Other components are coordinated roles, NOT competing owners: ModelRouter chooses an exact immutable target and compatible replica at request admission, a scheduler selects work from already-admitted sessions, TaskDrivers select next legal stage, serving protocol adapters implement wire semantics, delivery sessions track external transport progress.

**Hard invariants:** a model addition may not create another allocator/scheduler/state publisher/API server; a device addition may not change mathematical source truth; a new task may not rewrite the physical worker; a specialized whole region remains an optional provider candidate with a canonical decomposable baseline.

## 3. Three different graph topologies, never conflated

A) **Software dependency graph.** Cargo crates and internal modules; conceptual owner and access constraints.

B) **Mathematical/task graph.** Typed tensors/media values, semantic operations, state effects, nested mathematical scopes, verified model assembly and trusted task-stage dependencies. This defines meaning and legal composition.

C) **Physical placement/command graph.** Compute regions, parameter extents, state shards, reusable caches, device-local buffers, transfers, event dependencies, captured programs and remote link costs. It defines one bound physical execution.

One structure does not force another. A Rust crate edge does not require data copy; a model boundary does not require a server/process; a type-correct model port does not prove backend placement or quality; a Ready device event does not mean task state was published or packet played.

### Distinguish three senses of pipeline

| Composition | Example | Authority |
| --- | --- | --- |
| Mathematical region | RMS → attention → FFN, nested fused subregions | SemanticProgram / compiler |
| Single verified model assembly | image text encoder + transformer + scheduler + VAE, speech encoder+codec | ModelDefinition/Assembly |
| Task workflow/pipeline | separately registered ASR → LLM → TTS | TaskCoordinator/TaskDriver, compound typed stages |

Nested mathematical regions are compiler structure, not dynamically recursive execution. A verified model assembly is one source-identified machine-learned model with independently placeable components. A multi-model task pipeline coordinates *different* committed sessions with explicit stage or compound transaction boundaries; it is not identical to a single model's internal stages. External tools/actions remain above core behind permissions, not arbitrary GPU TaskDriver callbacks.

## 4. Two enduring representations

**SemanticProgram (SIR).** One canonical device-independent typed mathematical program: source-bound parameters; logical values/rank/axes/shapes; positions, attention variants/masks; exact operator formulas; cast/reduction, nonfinite and numerical profile; explicit immutable reads; typed mutable state effects; valid stage/frontier outputs. Pure operations can be rewritten only under declared algebraic/numeric legality. Stateful operations cannot cross causal/state boundaries unless a validated effect transformation exists.

**ExecutableBundle.** One immutable prepared physical plan for a versioned semantic program, workload regime, topological capabilities and chosen admissible candidates. Contains direct kernels, captured programs, AOT graph handles, guarded shape bands, resource/transfer/dependency commands, storage lifetime/leases, state bridge, fallback cut, code/compiler/driver identity and bound mutable parameter slots. Live request epochs/slots/physical leases bind **late**, not into the compiled identity.

Compiler-local representations are useful but NOT additional permanent standards: RIR = transient legal region/layout/fusion options; KIR = transient kernel/codegen candidate; CIR = command/resource/event DAG contained inside the physical bundle. A StepProgram or WorkEnvelope is a prepared execution handle/bindings, not a third semantic IR. Compile/load/tune outside the request hot loop; bounded prevalidated variants only in the immediate dispatch loop. A shape miss uses a correct proven baseline or typed refusal, not uncontrolled dynamic codegen.

**Closed-region specialization:** a physical provider may replace an op, fused block, model stage, whole-model mode or partitioned program only if **all** visible inputs/outputs/state effects, original artifact/semantic/numerical rules, provenance, shape guards, source access, physical reader/lease lifetime, real completion, cancellation/fork/replay and required observations are honored. If state layout changes, transition at a proven committed frontier with snapshot/conversion/recompute; never retag pointers or publish partial state. Whole-model optimized programs are first-class, but **not standalone model inference engines**.

## 5. Extensible execution providers, not a trait zoo

A single **RegionExecutionProvider** conceptual seam chooses and prepares qualified closed semantic regions. Native CPU/GPU kernels are one execution family; others are captured GPU graphs, AOT device graphs, statically scheduled dataflow/fabric programs, and qualified remote multi-device compiled regions. An opaque inference text API is an application-level TaskServiceAdapter, not an admitted device-region provider.

Provider advertises actual capabilities: semantic op/schema, executable granularity, tensor/encoding/precision/shape/raggedness, device control/loop support, state bridge, memory topology, transfers/coherency, code image/compiler/driver identity, failure/cancel/fencing, test evidence and source/license policy.

No generic requirement to emulate wafer/TPU internals through host-issued scalar kernels. Providers can own large device-native programs with bounded loops; actual task/session authority stays in Rust. Optional Rust-owned GPU kernels supported where feasible; approved vendor runtime FFI remains distinct from importing unreviewed third-party compiled kernels.

Do NOT require one trait per codec, operator, kernel, buffer or platform. Prefer static enum dispatch and concrete prepared structs until two real consumers justify a trait. No heap virtual dispatch per tensor element, no vtable in a hot projection loop; coarse per-prepared executable dispatch is fine.

### Execution/granularity table

| Candidate | Suitable job | Limitation |
| --- | --- | --- |
| Portable CPU | canonical reference and source-supported local execution | throughput may be limited |
| Native GPU op | GEMV/GEMM, attention tile, packed quant kernel | too many launches if forced on static accelerators |
| Fused native region | norm+projection, recurrence, grouped MoE, full layer | numerical/effect/observer legality must be proven |
| Captured command graph | repeated shape-bounded GPU stage | pointer generations/encoded readers/capture lifetime |
| AOT device graph | encoder, denoiser, layer, whole model graph | compile/load cost, shape bands and state export restrictions |
| Static dataflow | fabric/wafer/graph accelerators | device-owned schedule and state bridge |
| Partitioned program | multi GPU/NPU/remote layers/heads/experts | complete transfer/state/fencing/cost model |
| External hosted inference | explicit TaskServiceAdapter | NEVER counted as native region execution |

## 6. Task drivers and work policies

One small trusted **TaskDriver** describes high-level stage sequencing, iteration/frontier legality, model-specific stop/valid-output rules and stage-dependent state; it never becomes a general agent workflow DSL, task subprocess manager or provider allocator. Stages can be model-wide closed graph islands and bounded device-native loops, not a callback per token.

Tasks require *different work units*: AR token/target verification rows; encoder sequence/window; audio frontend samples/chunks; ASR hypothesis window; TTS acoustic frames and RVQ codebook groups; denoiser/flow timestep and spatial/temporal tile; image and video output boundaries; typed score/decision units. A token budget cannot assign meaning to a PCM sample or image solver step.

One common scheduler/service makes global fairness, budgets and admission decisions; stage-local policies generate compatible execution work:
- Decode/prefill/target-verifier token or row budgets; chunked prefill with protected ITL and aging to prevent starvation.
- Big-batch encoder throughput and bounded memory.
- Denoising/video spatial tiles, allowed timestep schedule and temporal dependencies.
- Live acoustic input and complete codec output under real-time soft deadline and jitter constraints.
- Finite/schema-specific decision scoring and atomic answer sets.
- Optional distributed stage and remote queue only when authorized and profitable.

Never mathematical-batch incompatible models because their tensor shapes happen to match. Different tasks/models can be *co-scheduled* on shared devices and physical pools; within one qualified model/adapter/numerical plan, compatible sessions may form native batches.

## 7. High-performance serving is an engine concern, serving protocols are a library concern

vLLM-class baseline preserved: continuous batching, prefill/decode coexistence, persistent stable slots and incremental GPU metadata, paged/hybrid cache, prefix sharing, async worker stages, optional speculation with target verifier, GPU-native sampling/feedback, admission backpressure, QoS, fairness, prefix/placement affinity and comparable latency/goodput measurement.

**Three admission layers:** ingress request/payload/tenant/deadline credits; model/session/full-state/source and uncomputed work capacity; device StepPlan rows/scratch/transfers/active submission. One ResourceArbiter owns physical bytes; these are three control checkpoints, not duplicate independently spendable capacities.

**Host hot-path contract:** accept compact already-validated step inputs, choose among finite prepared plan candidates, bind slot and lease generations, submit and retire; no JSON, no catalog search, no dynamic compiler search, no global allocation lock per kernel and no mandatory device→host→device roundtrip when validated device-native feedback is available. GPU execution may overlap host prep, IO, peer transfer and another legal stage while preserving real dependency/event fences.

**Persistent batching:** stable request slots and per-session versions; incremental token/mask/position/row pointer metadata; opaque prepared executor receives bounded deltas; compatible cohorts never pad inactive rows into visible state/sample. A batch is physically real only when its native kernels share work, not merely when the server loops N independent single-request calls.

**Bounded candidate selection:** joint optimize prefill chunk, active rows, speculation depth (including K=0), MoE expert union/padding, source encoding, shape-compatible kernel, hot cache/SSD transfer, topology and request QoS. Candidate set is prequalified and bounded; online change only at legal state/placement boundaries with hysteresis, fallback and truthful mode receipt. Never use an approximation without explicit policy and quality proof.

**Streaming:** immutable committed outputs are exposed via SDK events, then optionally serialized by serving adapters. Client delivery, ACK and media playback are separate credits/epochs; slow output transport must not retain GPU scratch forever or force other session stalls.

The new **zerogravity-serving library** owns protocol adapters, network lifecycle, supported HTTP/OpenAI/Responses semantics, streaming framing, access control at transport level, TLS/reverse proxy integration policy, metrics endpoints, request validation mapping, timeouts, listener/reload options and bounded connection buffers. It delegates all inference scheduling/model loading/state/admission to the engine through zerogravity's public runtime API. It is reusable from a Rust executable embedding an API server; CLI just configures and calls it. Neither CLI nor the serving crate is allowed to implement an inference engine.

Full detailed public API, crate dependencies, server lifecycle and conformance is in [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md).

## 8. State beyond paged KV

**Logical StateRoot** is an immutable committed version per TaskSession (or explicitly authorized shard), containing full required domain version handles and a typed TaskFrontier. Domains include full/sliding KV, GDN/KDA/SSM convolution and matrices, sparse/MLA/QSA indexer, n-gram/Engram and target/draft taps, grammar/RNG/decoder history, audio encoder frontend overlap, acoustic codec/duration/RVQ state, image/video solver/latent state and typed decision output. Physical policies vary; the transaction protocol does not.

Mandatory sequence:

~~~text
Begin(expected_root, epoch)
    -> private/COW provisional state and admitted resources
    -> prepared execution and actual device/IO completion events
    -> each required domain PREPARES a compatible version
    -> coordinator validates complete frontier/effects/epoch/old root
    -> exactly one PublishRoot
    -> ordered source-valid committed output events
    -> retire old readers after actual physical quiescence
~~~

No independent StateDomain::commit, no device-visible Command::Commit that publishes logical state, no unverified token emission, no hidden model mutation on cancellation. Private device buffers cannot overwrite the old committed state unless proven undo/recompute safe. A logical O(1) root swap is not O(1) recurrent state copying.

**Hybrid prefix reuse** requires a common restorable frontier intersection across *all required* model/task state domains, with source/checkpoint/quant/processor/tokenizer/template/position/numeric/state schema/tenant authorization. A KV hit without matching GDN or indexer checkpoint cannot be reported as a complete prefix hit; back off to the largest jointly restorable frontier or recompute. Compatible COW pages protect queued, encoded, captured and in-flight readers through true completion.

Speculation keeps independent draft and target state, target-authoritative verification of candidate tokens and sampler distribution, complete committed acceptance, and rollback/recompute for ALL effects including RNG/grammar, recurrence, sparse indices and media. Model-correct output from a broken always-rejected drafter does not prove the drafter correct.

## 9. Resource, memory and placement

Physical resources: GPU/CPU/NPU computation, model weights, ephemeral activations, per-session state, shared prefix snapshots, draft/MTP weights and state, read-only expert pool, pinned DMA windows, CPU threads, native modules/graphs/code images, SSD staging, OS/cgroup reserves and network/link credits.

The **single global PhysicalResourceArbiter** admits coarse reservations at model/session/step boundaries; device-local slabs/rings/pools allocate and retire only from prepaid sub-envelopes with generation-tagged real backing leases. Reserve **all necessary domains atomically** for a multi-GPU+host+pinned/IO plan; alias overlapping UMA domains and count backing once. A host/GPU/IO observation is advisory and can be stale; there is no impossible OOM guarantee against OS/driver changes.

**Four independently representable placements** within one PlacementPlan:
1. ComputePlacement: op/layer/model stage, partitioned outputs, experts, prefill/decode roles.
2. ParameterPlacement: immutable source weights/derived packs, adapter/MTP/lookup and graph images.
3. StatePlacement: per-session authoritative live KV/recurrent/sparse/decoder/media state.
4. CachePlacement: immutable/recomputable prefix snapshots, encoder features, expert caches and other retained assets.
A derived **TransferPlan** connects mismatched owners with exact source/version/tenant/layout, first-use dependency and real readiness, never a separate distributed scheduler.

Default local placement colocates work/parameters/state unless measured benefit and valid bridge justify separation. One logical tensor can be materialized in local VRAM, unified RAM, ordinary RAM, SSD or authorized remote peer; its source and semantic identity remain unchanged. Fully resident fast path incurs NO mandatory virtual tensor lookup/cache miss branch per projection. A real SSD miss or device transfer uses bounded asynchronous work with event fences, cancellation, source hash and lease retention; caches cannot free buffers that a recorded/captured/queued GPU command has already encoded as a reader.

UMA: GTT/VRAM addressable limits are NOT physical free RAM. Observe MemAvailable, actual driver memory, page cache, OS/cgroup headroom, pinned memory, external pressure and overlapping domains. Avoid unsafe clock/firmware/system tuning, especially for GPU+XDNA2 NPU cooperation. A model bigger than RAM may be correct but SSD-bound; do not claim high-speed without cold/hot IO measurements.

Cache policies must be specialized under one global budget:
- immutable model weights and source snapshots share only exact authenticated backing;
- paged KV needs page tables, refcounts, valid tails and typed state resume;
- recurrent state needs bounded checkpoints/COW/recompute;
- MoE experts need actual source-selected IDs, size classes and decode-hotset protection;
- lookup tables need bounded row IO;
- scratch needs liveness and aliasing under actual graph/capture readers;
- disk staging must have true async readiness and source mutation/short-read refusal.
A universal LRU is wrong; independent unaccounted caches are equally wrong.

## 10. Multimodal and realtime design

Typed value descriptors include Tensor, TokenSequence, CodebookSequence/RVQ, AudioPCM, AcousticFeatures, VideoFrames (PTS/DTS), ImagePixels/color metadata, MediaLatents/VAE identity, EncodedPackets, Transcript with speaker/timing/finality, DecisionScores with honest calibration, Structured(schema). ValueRef has content and lease identity, bounded extents, correct clocks. No implicit conversions based on equal byte shapes or guessed sample rates.

**ModelAssembly** binds a whole source-authored composite checkpoint (e.g. text encoder + denoiser + VAE; vision encoder + projector + text decoder; acoustic encoder + codec), required processors and state schemas. It is NOT replica group nor task pipeline across unrelated independently registered models.

TaskDrivers cover causal text, encode/pool/rank, Whisper-like encoder-decoder ASR, CTC, RNN-T/TDT, deterministic and codec-AR TTS, acoustic-flow TTS, native duplex audio, latent image flow, image AR codebook, temporal video, joint AV, finite typed decisions. Native stage math and codecs must be source-qualified; a fake textual transcript or model-name metadata never implies runnable media.

Typed task pipelines may nest descriptions, but compile stage structure and typed ports ahead of execution; no recursive per-token/stage interpreter. Allow device-resident stage edges with compatible producer/consumer, real events, generation-safe leases; materialize/convert only where needed. Default compound task behavior is legal staged commits, not fictitious atomic rollback across prior ASR/audio playback/external tools. Optional explicit joint multi-model transaction needs separate acceptance.

Media has distinct Prepared, Committed, Delivered, Acknowledged, Presented frontiers, plus received/decoded/consumed ingress. ClockDomainId, rational rates, exact conversion residue, track/lane IDs, generations and source delay prevent drift and stale playback. Turn/OutputGeneration barge-in invalidates future uncommitted playback without resetting physical card epoch or rewriting prior committed/root history. Transport reconnect never grants raw neural state. Playback/ACK cannot prove the human actually heard sound.

## 11. Distributed and heterogeneous evolution

Support CPU/GPU/NPU combined compute, expert placement, legal output-row projection slices, multi-GPU layer/tensor/expert sharding, optional remote closed-region workers, disaggregated prefill/decode and graph/static devices as *physical candidate plans*, not new model servers.

A new provider must expose accurate device/topology/compiler/runtime capabilities, explicit coherency, typed state bridge (private provisional versions, verified snapshot/restore, or safe recompute), real completion/failure/fencing and program lifecycle/memory cost. Opaque-unrecoverable state is ineligible for stateful region replacement. A vendor text API is an external hosted task service and never masquerades as a region provider.

Compute/parameter/state/cache placement may vary independently, with an exact TransferPlan. State migration or topology change occurs only at a verified committed frontier after reserve/transfer/check/fence. One authorized logical session publisher remains responsible, even if remote workers own provisional state shards. Remote peers require explicit enrollment/data rights, authentication, tenant scoping, link/latency measurement and failure epoch protection; a tunnel grants connectivity, not RDMA, privilege or low RTT. No consensus/transparent exactly-once network failover claim without an actual protocol.

NPU/GPU cooperative kernels need explicit power/fabric safety; documented host hard-hang reports forbid automatic unsafe tuning. Rust-hosted SDK bindings do not magically make vendor-written device code Rust-authored.

## 12. Model onboarding and correctness

**Target model contributor workflow:**
1. Add source/processor/format recognizer and exact component manifest with hashes/license.
2. Define family config, roles/ties, source tensor validation, one mathematical layer/forward traversal, state schema and tokenizer/template/media processor as appropriate.
3. Record/evaluate into canonical SemanticProgram; implement new *shared* operators only if mathematics genuinely requires them.
4. Prove CPU mathematical reference, original-source numerical fidelity, complete trained model logits/state and task-quality gates at declared scope.
5. Reuse already qualified generic provider/capabilities; register independently qualified specialized region/kernel/graph only when beneficial.
6. Register family/variant+task capability in the single cold-built catalog; do not claim broader GPU/quant/context/model support from metadata.
7. Reuse SAME engine state/scheduler/router/serving/public API. No Qwen/Gemma/http-specific request branches.
8. Benchmark real cold/warm workloads, including quality and negative results.

Extensibility drills: add (a) another model using all known ops; (b) a truly novel stateful op; (c) a fused region disabled without changing semantic output; (d) a new device with real event/lease behavior; (e) an encoder/ASR/denoiser task with non-token frontier; (f) a mock remote state bridge. Measure changed shared production files, LoC and model-specific branches—not merely number of traits.

**Four independent quality claims:** (1) semantic and source safety; (2) per-op declared numerical accuracy; (3) full-model task/distribution quality versus pinned reference; (4) determinism in explicitly advertised run/batch/hardware modes. Do not replace these by top1, perplexity, matching forced teacher trajectories, hash equality or an upstream tolerance imported without context. Keep GGUF and original HF, BF16/FP8 and quant, C1/C4 and CPU/CUDA/HIP/Vulkan evidence independent. A different source quant is a different model artifact. Performance follows correctness and complete resource/state tests.

Capability status ladder: recognized metadata, semantic/reference, compiled, full-functional, state-safe, quality-validated, performance-qualified, production-qualified, or explicit blocked/unsupported. Each cell qualified per exact source, task, shape, mode, device, compiler/driver and numeric.

## 13. Deployment, packaging and stable public API goals

The user should be able to:
- import one public Rust crate "zerogravity", configure/load/inspect/run any supported model and task, and stream events, with no CLI dependency;
- construct a library runtime with models, caching, routing, pipelines, memory/device policies, telemetry, qualified provider selection and lifecycle;
- build the **same** local daemon/server programmatically, without invoking the zerogravity CLI or reimplementing HTTP/model routing; use the **zerogravity-serving** library through public reexports/builders in zerogravity;
- use CLI as the thinnest interface: parsing, local config/presentation/signal dispatch, error display, invocation of library functions; no proprietary source/memory/scheduler logic in CLI;
- run binary "zerogravity run model.gguf" without compulsory daemon/config, "zerogravity serve ..." with same engine, introspect genuine capabilities/fit/refusals, obtain a clear dry run;
- embed without Python model runtime or vendor GPU SDK when using CPU only; optional backends feature-gated and accurately advertised.

**Library must own ALL functionality** the CLI exposes, including model pull/cache, inspect, devices list, model catalog/alias/loaded list, direct run, serve start/stop/configure, request processing, output streams, pipeline compile/check/execute, snapshots, support matrix, generated metrics, status and explanations. Supporting crates can implement it; facade exposes coherent public typed APIs. Do not make the facade one giant implementation crate; reexport a curated capability-oriented surface, private modules for complexity, versioning/semantic stability only when decided.

**Serving library** is an independent package with HTTP/stream protocols, connection lifecycle, auth/transport policy, routing protocol payloads to engine Requests and committed Events, bounded ingress/egress, SSE/chunk framing, cancellation and structured client conformance. It must work from ANY embedded Rust application, not only zerogravity-cli. It depends on model-neutral engine contracts and shared types, not on CLI, model-family concrete code or GPU specifics. The facade exposes its builders/types so ordinary users depend only on zerogravity. The dependency DAG must remain acyclic.

See [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md) for exact intended examples, API matrix, feature/dependency boundaries and acceptance.

## 14. Initial accepted design priority

**Next phase must FIRST resync/revalidate against actual repository HEAD and uncommitted code**, assess recent refactors/commits, run affected read-only source inspections and plan crosswalk, revise documents as needed, and only then ask/receive a separate user instruction for implementation. This 27 October archive deliberately does not claim the 26 October code inventory remains current. The previous inspected live tree (2026-10-26) was HEAD 195099ca and very dirty; by 2026-10-27 on dotlink-l5 HEAD has advanced to eb38dd5e and was clean at initial read. This reflects moving work, not a verified migration. Never transplant old file-line findings without rechecking.

After explicit implementation authorization, focus: make baseline reproducible; enforce source→semantic→physical ownership and Cargo boundaries; extract one common engine and robust library facade; split serving into standalone crate; have CLI depend exclusively on facade; qualify Qwen3.5 0.8B/2B source CPU and selected GPU with unchanged numerical evidence; register Gemma without copying runtime; then vLLM-class temporal serving and complete-domain state; then 27B SSD/dense, Gemma variants, broader tasks/distribution in accepted user sequence.

**Design acceptance gate:** simultaneously satisfy:
- Adding a model without a new op touches only its family, one registration point and tests.
- Adding a device touches only provider/backend and its capabilities/gates.
- Adding a task touches its TaskDriver/source/codec and typed ports, not a model-named token runner.
- Changing a cache strategy cannot mint new physical capacity, skip source checks, publish state or free live encoded readers.
- A new protocol endpoint does not create an inference engine.
- Every functionality is consumable via public zerogravity Rust library; separate zerogravity-serving crate can start a server via programmatic calls; CLI contains no engine/serving implementation.
- One complete second model or non-token task validates reuse with real source/fault/quality and no hidden per-model conditional explosion.
- Full-request performance and honest failure/support receipts replace isolated kernel-only success.

## 15. Cross-document map and handoff

Use this pack:
- [PRIOR_ART_AND_REVALIDATION.md](PRIOR_ART_AND_REVALIDATION.md): in-depth actual llama.cpp, vLLM and historic zerogravity file trees; design comparison and source pin caveats; architectural documents and outstanding gaps.
- [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md): precise 27 October requirements: library API, separate serving library, application assembly, CLI ownership, protocol conformance, lifetime/cancellation and API examples.
- [IMPLEMENTATION_AND_ACCEPTANCE.md](IMPLEMENTATION_AND_ACCEPTANCE.md): source-to-target file migration map, phased zero-runtime-change plan, gates, performance studies and risk reduction.
- [DECISIONS_AND_OPEN_QUESTIONS.md](DECISIONS_AND_OPEN_QUESTIONS.md): original/user amendments, settled architectural invariants, rejected alternatives, explicitly undecided policy choices.
- [HANDOFF.md](HANDOFF.md): mandatory read order, next conversation starting prompt, resync-only instructions and no-edit requirement.
- [TODO.md](TODO.md): held project-specific checklist, separate from existing .agents/plans/TODO.md owner queue.
- Existing reference docs: ../pipeline-composition/STRUCTURE_AND_NAMING.md, ../pipeline-composition/REGIONS_AND_PROFILES.md, ../../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md, ../../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md, and the full project documentation tree.

Do not edit source, active TODOs, prior qualification receipts, existing plans, firmware/device settings, model assets, HEAD, branch or push based on this archive. It is a **side reference** created to preserve future design context pending explicit later resynchronization.

## END FILE: .agents/plans/fabric-vnext/README.md


---

## BEGIN FILE: .agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md

# Fabric vNext — public zerogravity Rust library and reusable serving library

**User amendment (27 October 2026):** The user explicitly requires (1) **serving in its own library**, not inside zerogravity-cli, and (2) **everything usable from an exported zerogravity library** by arbitrary other Rust programs. zerogravity-cli must use that library and own only command-line input, OS terminal interaction and presentation. These are **accepted architectural decisions**, overriding any earlier proposed tree with serving code in the CLI or one public facade limited to inference.

**Status:** design-only frozen side reference; NO implementation/resync authorization. Internal code is pre-v1 breakable. The APIs below are illustrative semantic contracts, not promised Rust type names or a frozen ABI.

## 1. The foundational separation

### Public programmatic API: zerogravity

One ordinary dependency should suffice for a developer:

~~~toml
[dependencies]
zerogravity = { version = "...", features = ["cpu", "serve"] }
~~~

The version/feature names above are illustrative; do not publish/freeze them before source review.

**Public responsibilities exposed through this one crate:**
- inspect/load/exact-source-verify models and multi-component assemblies;
- parse and validate model manifests and pinned formats, download/pull source and caches under explicit authority;
- register/select native compiled families, task drivers, effective source/profile/quant/quality contracts;
- enumerate devices, resolve physical identities and capabilities, explain qualified fit/placement;
- construct/open/close the library Runtime/Engine, configure budgets/placement/headroom/tenant/session limits;
- run text/embedding/rerank/decision/media/pipeline tasks without a network listener;
- stream typed committed output/events and observe statuses; cancel and bound memory safely;
- manage registered models, loaded resident instances, aliases, replicas, adapters and intended load/reload/unload policies;
- manage prompt/codec input, tokenization, structured requests and source-bound task schemas;
- prepare/inspect graph programs and cached qualified execution variants;
- persist/restore exact compatible snapshots as the model/task capability permits;
- monitor resource ownership, allocator/device receipts, throughput and failure telemetry;
- construct embedded OpenAI/Responses/other supported protocol servers via serving library reexports;
- start/stop/reconfigure listeners (within safe model binding/epoch rules), obtain bound addresses and worker status, attach authentication/transport policy and admin endpoints;
- expose structured error/refusal/capability receipts and dry-run/diagnostics;
- future authorized remote peer enrollment/placement, multimodal realtime/speech/video stage control and async jobs.

A Rust program must **not** need to call a CLI subcommand, parse CLI stdout, import private zerogravity-cli types, exec the binary, or copy request/session/HTTP implementation to access these capabilities. Optional feature selection may legitimately return Unsupported where model/device/service is not built or independently qualified.

**Implementation versus facade:** The public zerogravity crate SHOULD remain a lightweight curated facade/integration point. Underlying crates implement codecs, models, CPU, GPU, live engine and network serving. Reexports and builders keep a coherent public namespace. "Everything accessible through zerogravity" does not mean everything must physically live in zerogravity/src/lib.rs or be eager-linked on every platform.

### Reusable protocol server: zerogravity-serving

**Separate Rust library crate**, suitable for direct programmatic embedding, CLI start, GUI/desktop/web app local listener, tests and other programs.

It owns:
- HTTP/OpenAI-compatible routing and request JSON/body validation;
- actual Responses and chat/completions/embeddings endpoint conformance by qualified feature;
- SSE streaming/event framing, client disconnect/cancellation, bounded output credits;
- readiness/health/models endpoints and standard error mapping;
- listener lifecycle (start, bind, serve, graceful shutdown, cancellation and metrics);
- protocol-level authentication, loopback default, address/TLS/reverse proxy policy, client rate/payload limits;
- HTTP transport adapters to/from engine Request/CommittedEvent;
- WebSocket and WebRTC signaling/media/transport *when implemented and independently qualified*;
- protocol versioning, content-type/error/header/streaming semantics and official SDK tests;
- optional admin plane endpoint adapters with separate rights, never unguarded inference rights;
- transport sessions, epochs, ACK/playback receipt and bounded replay cursor semantics;
- listener observability, connection timeouts, keepalive/backpressure and egress encoding.

It MUST NOT:
- load model-specific Qwen/Gemma tensor math;
- implement a second model catalog, PhysicalResourceArbiter, StateRoot publisher, work scheduler, KV manager or GPU device runner;
- require a shell, argument parser, terminal renderer or process-level SIGINT handler;
- call GPU kernels or mutate private model source math;
- deem HTTP success equal to model StateRoot publication or client playback;
- auto-download model weights or enable network bind merely because runtime object is created.

Serving invokes model-neutral engine APIs. The facade reexports/wraps the serving library's public builders so users can import from zerogravity only. Direct dependency on zerogravity-serving is also legitimate for deeply customized integrations; facade must not hide core capabilities.

### Thin binary: zerogravity-cli

CLI owns only:
- Clap/argument parsing, shell environment discovery, config file path selection and command-line precedence;
- human help text, progress bars, terminal formatting, console streaming and error presentation;
- textual/JSON presentation, ctrl-C/signal-to-public-cancel-token bridge and conventional process exit status;
- resolving interactive terminal input and CLI-specific defaults;
- invoking the public zerogravity API, including zerogravity::serving, models, cache, devices, run and pipeline.

CLI explicitly does **not** own HTTP semantics, routing work, model source acquisition protocol, numerical selection, model lifecycle, memory fit implementation, state or scheduler. A library caller must be able to reproduce every CLI operation using typed library calls without depending on environment variables or terminal state except when those are explicit optional inputs.

**Ownership check:** search CLI source for API endpoint router handlers, model-specific GPU session types, allocator, source transport or native batch policy. Any such production functionality in CLI is architectural debt to move to library owners. CLI-only tests can validate parsing and output; shared conformance tests must exercise library APIs independent of CLI.

## 2. Recommended crate graph

Recommended (not automatically implemented):

~~~text
zerogravity-cli ──────────────────────→ zerogravity (public SDK facade)
                                           │
                                           ├─→ zerogravity-serving
                                           │       ├─→ zerogravity-engine
                                           │       └─→ zerogravity-core
                                           │
                                           ├─→ zerogravity-engine
                                           ├─→ zerogravity-model-qwen35
                                           ├─→ zerogravity-model-gemma4
                                           ├─→ zerogravity-cpu
                                           ├─→ zerogravity-gpu (optional)
                                           └─→ zerogravity-artifacts
                                                     ↓
                                                zerogravity-core

zerogravity-engine ─────→ core + artifacts + abstract provider/model interfaces
model family crates ────→ core + artifacts (+ narrow optional provider adapter)
cpu/gpu crates ─────────→ core + artifacts (NO model family import)
serving crate ──────────→ engine + core (NO facade import, NO CLI import)
~~~

Dependency direction is acyclic: serving cannot depend on zerogravity facade if zerogravity facade itself reexports serving. Instead engine/core contain shared request/event/API types. Serving depends on engine contracts and core types; application facade depends on both.

**Engine construction from binaries and other Rust apps:** zerogravity::RuntimeBuilder (facade) can register builtins/qualified providers; serving receives an Arc/model-neutral handle to the already constructed engine (or builds a runtime from validated options through the same facade). Its server may add ingress limits but does not create a second physical global arbiter. A single model's local direct run uses the same runtime without listener overhead.

**Optional crate strategy:** Recommend a new cohesive zerogravity-engine crate to enforce no concrete Qwen/CPU/GPU dependencies in scheduler/resource/session. Recommend a new zerogravity-serving crate. If source review after resync proves equivalent boundaries better enforced with existing crate modules, preserve semantics and revisit exact package split. But **serving must remain a separately reusable library** and **public zerogravity must expose it**; those are not optional.

Features:
- default CPU-only portable entry should build without CUDA/ROCm/Vulkan compiler or a network service;
- serve enables core HTTP server library optionally; no server binds until explicitly invoked;
- CUDA/HIP/Vulkan/Metal/NPU/remote features are independent, physically qualified;
- task/codec features explicit; unknown capability is returned, not an unconditional compiled "all" promise;
- facade's features forward narrowly and do not create circular dependency;
- a generic embedded Rust app may use a runtime and a server simultaneously with the same model catalog and shared resource ledger.

## 3. Public user-facing API shape (illustrative Rust)

Names and signatures will be resolved *after* resync and extensions tests. This is a practical usage contract, not final code:

~~~rust
use zerogravity::{Runtime, RuntimeConfig, ModelSource, TaskRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::builder()
        .cpu_enabled(true)
        .source_policy(/* offline/pinned, explicit authorization */)
        .build()
        .await?;

    let model = runtime.models()
        .load(ModelSource::local("./my-model.gguf"))
        .await?;

    let mut stream = runtime
        .run(TaskRequest::text(model, "Hello"))
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            // committed source-valid event; inspect typed payload
            event => println!("{event:?}"),
        }
    }

    runtime.shutdown().await?;
    Ok(())
}
~~~

The direct local API does not need HTTP, bind sockets, network authorization, a daemon, Python or a mandatory GPU.

~~~rust
use zerogravity::{Runtime, serving};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::builder().build().await?;
    // Exact source model registration/loading uses the public model API.
    let server = serving::Server::builder(runtime.clone())
        .listen("127.0.0.1:8000".parse()?)
        .openai_compatibility(/* qualified endpoint configuration */)
        .build()?;

    let handle = server.start().await?;
    println!("bound address: {}", handle.local_addr());
    // Other program modules can concurrently use runtime.run().
    // handle shutdown/cancellation and graceful request draining independently.
    handle.shutdown().await?;
    runtime.shutdown().await?;
    Ok(())
}
~~~

***Important:*** The two snippets are examples of intended developer experience. Not all named types or features currently exist; do not copy code into source before final API design. No fake success or server startup claim.

~~~rust
use zerogravity::{Runtime, serving, pipeline};

async fn example() -> zerogravity::Result<()> {
    let rt = Runtime::builder().build().await?;
    let supported = rt.capabilities().inspect(/* source + task + device */).await?;
    let plan = rt.plan(/* typed model assembly or pipeline */).await?;
    let dry_run = plan.explain_resource_fit().await?;
    // validated shape/config alone never claims functional execution
    let maybe_server = serving::Server::builder(rt.clone());
    Ok(())
}
~~~

**Embedded API design quality:**
- Async first for IO/model load/execution/stream, with ergonomic bounded sync wrapper only when useful; no mandatory Tokio runtime constructed behind user's back if avoidable.
- Typed error with Unsupported, Malformed, Unqualified, Quota, State, Cancelled, UnknownPhysicalCompletion, DeviceLost etc., plus stage and retryability. No random string parsing required.
- Runtime cancellation/signal is explicit token/scope and epoch; library does not install global SIGINT or process::exit.
- Output streams are bounded, with source-typed committed events; dropping/aborting a Future does not free physical in-flight backing.
- Source binding requires explicit permission/provenance; library constructor cannot silently fetch remote content.
- Device introspection and plan explanation are nonlaunch operations, never side-effectful GPU qualification.
- SDK lifetimes and backend initialization are transparent enough for embedding in a server/desktop app with other GPU users.
- Multiple Runtime handles in one process share the single physical authority where their device domains alias; either one root or a process service with explicit join, not silent double budgets.
- Source snapshots and loaded models are immutable source-owned handles; active requests pin exact ModelGeneration through alias hot reload.
- Accept authorized credentials via explicit scoped settings; never echo secrets in trace/event errors.

## 4. Runtime and serving separation at request stage

~~~text
zerogravity-serving HTTP worker
    |
    | validate transport/request/size/auth/rate, parse client format
    v
zerogravity-engine::Ingress::submit(TaskRequest)
    |
    | resolve exact alias, source+task capability and quotas
    | acquire model/session acceptance; queue bounded work
    v
engine::TaskCoordinator / scheduler
    |
    | use one model-defined semantic program and prepared bundle
    | workers report real Ready, state domains Prepare
    v
engine::SessionAuthority PublishRoot + committed event stream
    |
    | engine stream outputs logical CommittedEvent, error/usage/stop
    v
zerogravity-serving
    |
    | encode compliant SSE/JSON/media output with transport credits
    v
Client
~~~

The serving layer may concurrently support multiple protocols over one engine. A source-valid response can be produced without any network interface. The same model loaded into the runtime must not be duplicated because the server was started.

**No duplication of scheduling:** Serving has network ingress queue/accounting but does not own the physical token selection or root publication. It can reject a request for client/auth/rate/body; it cannot change model semantics to fit. Engine has one general multi-model load/routing/admission authority and stage-specific scheduling. Physical provider has backend queue, not a second token scheduler.

### Concrete API/service obligations

The public library must expose all of:
- start server with already-built runtime;
- automatic localhost-only listen default, explicit bind/auth policy;
- inspect actual bound addresses/port (including ephemeral port for tests);
- graceful shutdown timeout/drain/abort according to real leases and roots;
- attach model aliases and safely reload mappings without switching active session source;
- create/update/read qualified runtime config, stored profiles, priorities and limits;
- submit direct TaskRequest and consume typed committed outputs;
- access serving metrics, session/request IDs, queue/backpressure snapshots, cancellation;
- run regular model pulls, inspect, dry run, cache management, device listing through library;
- obtain structured explanations for "no model", unsupported backend, insufficient memory, invalid payload, numerical profile mismatch;
- use optional external protocol extensions without changing model families or GPU.
These are APIs, not claims they exist today.

## 5. OpenAI and transport conformance

Start with honest working subset and expand only with tests:
- GET /health or /ready with clear process/device/model readiness semantics;
- GET /v1/models with exact authorized aliases and capabilities, not metadata-only false positives;
- POST /v1/completions and /v1/chat/completions with template, stop, streaming, usage, roles, finish_reason, token IDs and errors per accepted model;
- POST /v1/embeddings when model/task actually supports it;
- POST /v1/responses as a distinct schema/streaming event family, not a mislabeled chat endpoint;
- structured output/grammar, tool call schema, multimodal input/output only per independently qualified model/task;
- optional WebSocket, WebRTC, audio streaming, video, async jobs with typed media clocks/lane/watermarks/interruptions;
- administrator model pull/unload/reconfigure API separately authenticated and disabled/unexposed by default.

Conformance tests must compare official clients where available: both streaming and non-streaming; chunk boundaries, empty/partial yields, cancellation/disconnect, usage, EOS, finish reason, model names, error codes, payload quotas, generated token IDs. Do not guess undocumented protocol semantics. An unsupported parameter should fail explicitly, not silently ignored as acceptable.

**Auth and deployment:** default local loopback. External listen requires explicit owner authorization, appropriate TLS/auth and access enforcement; reverse proxy route trust must be configured, not assumed. Per tenant/source rights/quotas must protect cache and model state, including prefix-sharing authorization. A bearer token over cleartext external bind is not production security.

**Serving lifetime:** ServerBuilder::build validates settings but does not run model kernels, open network automatically or acquire unbounded memory. start binds and runs; shutdown closes ingress, drains/aborts bounded requests while respecting worker physical flight. Network disconnect/cancel cannot convert an unknown submitted GPU event to Ready or safely free backing.

## 6. Delivery events, backpressure and realtime surface

Keep **five explicit frontiers** in SDK and transport: Prepared, Committed, Delivered, Acknowledged, Presented/Played. Only committed outputs may be presented as finalized source-valid result; provisional ASR updates can be explicitly labeled, revision-bound and retractable where source permits.

SDK stream is independent of transport. Serving converts SDK CommittedEvent to JSON/SSE/media packet while tracking backpressure/credits. When a client is slow, bounded copied/retained immutable output packets and explicit output quota suffice; the HTTP socket must not hold accelerator scratch or live native graph lifetime unnecessarily. Avoid global blocking of other requests.

Media clocks are typed and versioned: rational rates, capture and presentation timebase, A/V lane, codebook depth, timestamps, PCM format, codec priming/remainder and turn/output generation. Server's transport/ACK cannot declare human playback. Barge-in cancels uncommitted future output on one TurnEpoch, leaves unrelated device/session epochs intact, and reconciles previously committed content with independently reported playback.

Library program can subscribe to task events and drive UI, microphone/audio sinks or application pipelines without any HTTP listener. A WebRTC server is one adapter to these same typed event/control streams, not a new speech engine.

## 7. CLI-to-lib command matrix

| CLI command / capability | Required library surface | CLI-only behavior |
| --- | --- | --- |
| run MODEL and prompt | Runtime.run / text task + source/model preparation | Prompt arg/stdin and human token streaming |
| inspect MODEL | Artifact/ModelInspector | Table/text/JSON presentation |
| models list / inspect / ps | Catalog, runtime residency/read-only snapshots | Pretty list |
| models pull | Authorized ModelStore acquisition/verification | Progress bar and CLI flags |
| models manage/load/unload | Runtime model lifecycle, checked admin rights | Confirmation prompt and formatted result |
| devices list / select | DeviceInventory/PlacementPlan/fit explanation | Table and -d parse presentation |
| cache list/import/manage | ModelStore snapshot/cache APIs | Progress and human sizes |
| pipeline check | Typed TaskPipeline compile/validate | Input YAML path and error display |
| pipeline run | Runtime.execute pipeline TaskRequest | CLI progress/results |
| serve | serving::ServerBuilder + runtime | Address/config flags and foreground signal handling |
| serve status/metrics | Runtime/ServingHandle metrics | Pretty status |
| config show --effective | ConfigResolver provenance | Text/JSON formatting |
| bench / diagnose / qualification | Library diagnostic APIs or separately available support harness | CLI invocation/report printing only |

A CLI-command operation that only works through CLI is an **architectural API completeness failure**. Tests should perform equivalent actions through Rust library (without executable spawn) and through CLI facade, comparing result semantics and explicit configs.

## 8. Dependencies and naming constraints for the future source refactor

**Never** create a facade↔serving cycle by making the serving crate depend on zerogravity facade. The build composition root is only the facade. Model-neutral engine contracts should be in engine/core and passed by Arc or equivalent scoped handle.

**Keep library code source-neutral:** server request parser can know OpenAI "model" string but not Qwen or Gemma architecture; resolve through engine's exact catalog/alias. It can know HTTP status and JSON framing but not CUDA streams or tensor strides. Facade can know built-in registry; it must not duplicate scheduler/HTTP implementation.

**One binary, many libraries:** internal Rust crates assemble into one published Rust library interface and one CLI binary; feature-gated serving crate is reusable via facade; independent native server binaries are optional packaging choices, not requirements. Packaging may use static/dynamic vendor runtime as actual platform warrants; a Rust host is not proof the full binary is statically linked.

**No initial public C/plugin ABI freeze** unless separately approved. Current user permits breaking all Rust internals and public unstable SDK APIs pre-v1; use clean ergonomic API instead of carrying historical names and aliases. Do not publish "stable" until source qualification and extension tests justify contract.

## 9. Tests proving the 27 October amendments

- [ ] Compile a standalone Rust program depending ONLY on zerogravity and run a permitted local source/model task end-to-end.
- [ ] Compile a standalone Rust program depending ONLY on zerogravity and embed/start/stop a live API server and submit a request with real emitted events.
- [ ] Separate zerogravity-serving crate builds and integration tests WITHOUT zerogravity-cli or Qwen/Gemma/GPU import.
- [ ] zerogravity-cli depends on zerogravity facade only (and conventional CLI/terminal/logging helpers); no private engine/serving internals.
- [ ] All existing CLI actions have programmatic API equivalent with matching errors, rights and result semantics.
- [ ] Server and direct SDK share loaded immutable model/physical backing and ResourceArbiter; no duplication by alias/listener.
- [ ] Server ingress/reload/shutdown while active sessions run does not change their source/model generation/root unexpectedly.
- [ ] Strict loopback/auth/body/stream conformance; slow/disconnected clients do not leak GPU buffers/HTTP queues.
- [ ] A source-specific novel model executes via HTTP with no model-specific server or CLI branches.
- [ ] Native media/structured task SDK output and server transport share typed event contracts; prepared vs committed vs delivered distinct.
- [ ] CPU-only build requires neither network stack nor GPU compiler and can run without an HTTP daemon.
- [ ] Rustdoc/examples plus crate/features docs show library-first usage before CLI-specific flags.

The strongest release test is a library embedding a Runtime, direct model task, two model aliases and live HTTP server concurrently, with shared state/resource accounting, correct cancellation and clean teardown.

## 10. Cross references

[README.md](README.md) supplies full overarching architecture. [PRIOR_ART_AND_REVALIDATION.md](PRIOR_ART_AND_REVALIDATION.md) preserves original research. [IMPLEMENTATION_AND_ACCEPTANCE.md](IMPLEMENTATION_AND_ACCEPTANCE.md) maps this amendment to specific future source moves and gates. [DECISIONS_AND_OPEN_QUESTIONS.md](DECISIONS_AND_OPEN_QUESTIONS.md) records hard decisions and open API policy.

**The next authorized phase is RESYNC AND DESIGN REVALIDATION ONLY**, not moving serve code now. Do not touch active source or shared plan TODOs merely because these docs describe the target.

## END FILE: .agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md


---

## BEGIN FILE: .agents/plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md

# Fabric vNext — full comparative research and zerogravity revalidation

**Research provenance:** Source-level investigation of ggml-org/llama.cpp and vllm-project/vllm performed 26 October 2026 through the GitHub connector; read-only inspection of zerogravity via dotlink-l5 on 26–27 October. The source research initially targeted HEADs of upstream default branches; they are moving. Revalidate exact upstream SHAs on future measurements. Historical zerogravity checkout used for file counts: HEAD 195099ca583ace766d075682927f590360c823a5 plus 26 October uncommitted working state. The later 27 October working tree is NOT included in the analysis. Current reference must not be mistaken for a new source audit.

**Preservation goal:** This document preserves detailed earlier research (file trees, architectural ownership, new-model integration, router/batching/cache layers, favorable/unfavorable patterns, measured code-size observations) and the second round of zerogravity architecture/plan review. It is not a new implementation status certification.

## 1. The central comparison

| Engine | Core abstraction | Main strength | Main limitation relevant to zerogravity |
| --- | --- | --- | --- |
| llama.cpp | Model-specific GGML tensor graph evaluated by GGML backends | Portable local execution, simple API and CLI, GGUF ecosystem, heterogeneous hardware | Shared model registry/converter coordination; less general joint multi-model/task resource scheduling |
| vLLM | Request scheduler + KV cache + workers/model runners | High-throughput GPU serving, continuous batching, async engine/worker split, rich model registry | CUDA/PyTorch-centric design and large model runner complexity; serving-first task assumptions |
| vLLM-Omni | Multiple modality stage executors/pipelines | Explicit non-AR modalities, audio/video/diffusion/codec stage diversity | Another framework's decomposition, not one Rust-native resource/state fabric |
| zerogravity (intended) | Verified semantic program + prepared physical executable + TaskDriver + transaction/state/resource authorities | Source-correct composable model/device/task/memory/serving specialization with one execution substrate | Complexity and unimplemented general compiler/serving/device/remote/media capabilities must be proven |
| zerogravity (26-Oct inspected implementation) | Rust ForwardPlan + model family crates + physical CUDA/HIP/Vulkan qualification + bounded serving | Real state/lease/source and GPU/CPU implementations beyond scaffolding | Still carries family/source-specific execution branches in GPU and serving, and incomplete production generic engine |

### The four most reusable lessons

1. **From llama.cpp:** mathematics constructs execution graphs; generic backend knows how to execute but does not need to know Qwen/Gemma identity.
2. **From vLLM:** EngineCore, SchedulerOutput, KVCacheManager, Executor, GPUModelRunner and ModelRegistry are distinct responsibilities; high throughput requires coherent scheduling, cache and physical execution design. The newer GPU MRV2 notes explicitly warn about expansion of a monolithic shared runner.
3. **From vLLM-Omni and zerogravity v5.1:** a model's internal audio/video codec stages and an application workflow spanning separate models are different. Non-text task work units cannot be shoehorned into language token rows.
4. **From zerogravity v4–v5:** semantic/effect legality, one logical publication root, one coarse physical resource arbiter, real completion/fencing and independent placement of compute, weights, state and cache distinguish a rigorous execution fabric from a serving-only runtime.

## 2. llama.cpp architecture: actual file tree and ownership

Source repository: https://github.com/ggml-org/llama.cpp
Upstream guide (moving default branch): https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md
Public llama interface: https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h
GGML backend interface: https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h
Model source directory: https://github.com/ggml-org/llama.cpp/tree/master/src/models
Server source: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
Multimodal source: https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd

### Structural layers

~~~text
llama.cpp/
├── include/
│   └── llama.h                      Public C inference API
├── src/                             libllama
│   ├── models/
│   │   ├── models.h                 Model base classes and helpers
│   │   ├── llama.cpp                Llama graph
│   │   ├── qwen3.cpp                Qwen3 graph
│   │   ├── qwen35.cpp               Qwen3.5/hybrid graph
│   │   ├── gemma4.cpp               Gemma4 graph
│   │   └── ...                      Architecture implementations
│   ├── llama-arch.{h,cpp}           Architecture and metadata mapping
│   ├── llama-model.cpp              Model lifecycle and selection
│   ├── llama-model-loader.cpp       GGUF reader and tensor binding
│   ├── llama-graph.{h,cpp}          Shared graph construction
│   ├── llama-context.cpp            Inference context and graph execution
│   ├── llama-cparams.cpp            Context parameters
│   ├── llama-batch.cpp              Batch representation
│   ├── llama-kv-cache*.cpp          KV policies/variants
│   ├── llama-memory*.cpp            State and recurrent memory
│   ├── llama-vocab.cpp              Vocabulary
│   ├── llama-sampler.cpp            Sampling
│   └── llama-quant.cpp              Quantization entrypoints
├── ggml/
│   ├── include/
│   │   ├── ggml.h                   Tensor/operation API
│   │   ├── ggml-backend.h           Backend API
│   │   └── gguf.h                   Container format
│   └── src/
│       ├── ggml.c / ggml.cpp
│       ├── ggml-backend.cpp         Backend scheduling
│       ├── ggml-backend-reg.cpp     Backend/device registry
│       ├── ggml-alloc.cpp           Allocation
│       ├── ggml-quants.c            Quant types
│       ├── ggml-cpu/
│       ├── ggml-cuda/
│       ├── ggml-hip/
│       ├── ggml-metal/
│       ├── ggml-vulkan/
│       └── ...                      SYCL, OpenCL, RPC, others
├── common/                          Application-level shared utilities
├── conversion/                      HF conversion support
├── gguf-py/
├── convert_hf_to_gguf.py
├── tools/
│   ├── server/
│   │   ├── server-context.cpp       Server inference execution
│   │   ├── server-queue.cpp         Request queue
│   │   ├── server-models.cpp        Model routing/child lifecycle
│   │   ├── server-http.cpp          HTTP
│   │   └── ...                      Streaming, chat, APIs
│   ├── cli/
│   ├── mtmd/
│   │   ├── mtmd*.cpp                Media/processor stage
│   │   └── models/
│   │       ├── gemma4v.cpp          Vision encoder
│   │       ├── gemma4a.cpp          Audio encoder
│   │       └── ...
│   └── ...
├── tests/
├── benches/
├── examples/
├── docs/
└── CMakeLists.txt
~~~

The source in 26-Oct lookup showed a dedicated src/models directory via src/CMakeLists.txt: file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp"), with a distinct llama core source list. This separation means a model graph builder can emit standard GGML operations, while the backend implements CPU/GPU memory and kernels without embedding that model's full forward loop.

**Conceptual inference sequence:** checkpoint → GGUF metadata and tensor decoder → model-specific graph builder → GGML graph → backend graph allocator/scheduler → chosen CPU/CUDA/HIP/Vulkan/etc. backend → context/state/sample → application. GGML need not know a Qwen architecture name.

**Two distinct schedulers:**
- GGML graph/backend scheduler partitions physical graph work and handles backend selection, buffers and copies.
- llama-server slot/request machinery assembles compatible sequences into decode/prefill batches and drives llama_decode.
The current server also has a router mode that manages multiple child serving instances; this provides useful lifecycle UX but is NOT one-process shared physical multi-model/task scheduling by default.

**New-model integration path:**
1. Interpret source HF model config and tensors through converter; GGUF support where applicable.
2. Extend architecture/metadata naming in llama-arch.h/cpp and gguf-py/gguf/constants.py.
3. Add src/models/<arch>.cpp: load_arch_hparams, load_arch_tensors, graph builders and model-specific state.
4. Extend llama-model.cpp/model declarations/factory appropriately.
5. Add backend primitives only if existing GGML ops cannot express actual math efficiently/at all.
6. Extend tools/mtmd where vision, audio or processor components require it.
7. Regression tests for schema/CPU/GPU/quant/long context/server.
The full guide is linked above. Integrating a nearby variant is cheap *after* all shared operations exist; a novel recurrent/attention math family requires larger changes.

**Concrete models inspected:**
- src/models/qwen35.cpp (actual graph plus recurrent memory use)
- src/models/gemma4.cpp
- src/models/qwen3.cpp
- tools/mtmd/models/gemma4v.cpp
- tools/mtmd/models/gemma4a.cpp

**Measured historical source inventory examples (NOT invariant upstream current totals):**
- llama-arch.h around 914 lines; llm_arch enum contained roughly 165 entries including UNKNOWN at lookup.
- src/models/models.h ~3,115 lines with model class declarations/helpers.
- src/llama-model.cpp ~4,022 lines; shared model dispatch/lifecycle.
These figures illustrate that even a relatively well-separated model/graph system can accumulate central registries and classes.

**What zerogravity should adopt:** per-architecture mathematical ownership; one stable public API; backend-independent model descriptions; GGML-like portable primitives and physical backend capability negotiation; simple local UX. **Avoid:** making GGUF conversion universally mandatory, sprinkling model-name switches throughout core model factory and graph scheduling, treating GGML op availability as trained model support, replicating child server processes to get logical model separation.

## 3. vLLM architecture: actual file tree and ownership

Source repository: https://github.com/vllm-project/vllm
Architecture design: https://docs.vllm.ai/en/latest/design/arch_overview/
New-model guide: https://docs.vllm.ai/en/latest/contributing/model/basic/
Model Runner V2: https://docs.vllm.ai/en/latest/design/model_runner_v2/
Registry: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py

### Source tree mapped by responsibility

~~~text
vllm/
├── vllm/
│   ├── entrypoints/
│   │   ├── openai/                   HTTP/OpenAI APIs, model endpoints
│   │   ├── cli/                      CLI parsing
│   │   └── launchers/
│   ├── v1/
│   │   ├── engine/
│   │   │   ├── core.py               EngineCore
│   │   │   ├── core_client.py        IPC
│   │   │   └── llm_engine.py         Engine-facing abstraction
│   │   ├── core/
│   │   │   ├── sched/
│   │   │   │   ├── scheduler.py       Schedule requests, token budgets
│   │   │   │   └── output.py          New/continuing/stopped step output
│   │   │   ├── kv_cache_manager.py   KV page ownership
│   │   │   ├── kv_cache_utils.py
│   │   │   └── ...                   Hybrid cache planning
│   │   ├── executor/
│   │   │   └── multiproc_executor.py Worker dispatch
│   │   ├── worker/
│   │   │   ├── gpu_worker.py         Physical worker lifecycle
│   │   │   ├── gpu_model_runner.py   Earlier V1 model runner
│   │   │   └── gpu/
│   │   │       ├── model_runner.py   Newer modular V2 path
│   │   │       └── input_batch.py    Persistent slot/batch structures
│   │   ├── attention/
│   │   │   ├── backend.py            AttentionBackend
│   │   │   └── backends/registry.py
│   │   ├── kv_cache_interface.py    Cache layouts / type schemes
│   │   └── spec_decode/              Draft/speculative strategies
│   ├── model_executor/
│   │   ├── models/
│   │   │   ├── registry.py           Model architectures mapped to implementations
│   │   │   ├── llama.py
│   │   │   ├── qwen3_5.py
│   │   │   ├── qwen3_next.py
│   │   │   ├── gemma4.py
│   │   │   └── ...                   Numerous model-specific types
│   │   └── layers/
│   │       ├── attention/
│   │       ├── quantization/
│   │       ├── linear/
│   │       └── ...                   Reusable neural layers
│   ├── multimodal/
│   │   └── registry.py              Processors/dummy inputs / frontend
│   ├── distributed/
│   ├── platforms/
│   │   └── interface.py             Device/platform capabilities
│   ├── compilation/
│   ├── config/
│   └── ...
├── csrc/                              Native kernels C++/CUDA/HIP
├── rust/
│   ├── Cargo.toml                     Separate Rust frontend workspace
│   └── src/ ...
├── tests/
├── benchmarks/
├── examples/
├── docs/
└── CMakeLists.txt
~~~

**Logical serving sequence:** HTTP/LLM frontend → tokenization/input processing → EngineCore → request Scheduler and KVCacheManager → scheduler output → Executor/Worker → shared GPUModelRunner → architecture-specific PyTorch model.forward() → shared attention/linear/MoE/quant layers + Triton/CUDA/HIP kernels → sampler/outputs → protocol streaming.

**Central vLLM distinctions:**
- Model mathematical implementation is model_executor/models/<family>.py.
- Reusable neural layers are model_executor/layers/.
- Hardware-specific attention and kernels are separate.
- Scheduler and KV cache allocate/admit state independent of model code.
- Executor communicates work to device workers and model runners.
- Frontend protocols do not define model math.
- vLLM-Omni extends stage structure beyond text, with modality-specialized computation.

**New-model options:**
A. Existing Transformers-compatible model using Transformers modeling backend (lowest-friction, may not be optimal).
B. Native optimized vLLM model implementation: forward pass, layer reuse, load_weights, architecture registry, multi-GPU/attention/quant integration, tests.
C. Out-of-tree model plugin via ModelRegistry registration, provided it complies with runtime contracts.
Novel attention/cache/state mechanisms and media processors can require modifications beyond simple model file addition. Support in registry does not guarantee arbitrary backend or quant modes.

**Concrete upstream model evidence:** vllm/model_executor/models/qwen3_5.py subclasses/uses Qwen3-Next components; gemma4.py contains its own model, attention, router, decoder structures. At 26-Oct lookup:
- llama.py about 586 lines;
- qwen3_5.py about 756 lines;
- gemma4.py about 1,507 lines;
- v1/core/kv_cache_manager.py about 930 lines;
- v1/worker/gpu/model_runner.py about 2,506 lines with prominent comment that it should contain ONLY behavior common to all models and should remain minimal.
- models/registry.py ~1,636 lines / ~67 KB with separate generation/embedding/late-interaction/reward/classification/multimodal/speculative mappings.
Historical registry excerpts contained 124 text-generation and 127 multimodal entries; these are category entries with overlaps/variants and MUST NOT be interpreted as 251 distinct fully qualified models.

**vLLM cache strategy:** paged logical KV block table separated from physical cache backing. Block pool, cache keys/refcounts and reusable prefix blocks reduce duplicate prefix computation and enable flexible admission/preemption. That should inform zerogravity's generic domain-specific state pool, but it is not a complete substitute for GDN/SSM/MLA/media history.

**vLLM Model Runner V2 lessons:** The old V1 runner grew substantial interleaving of persistent batch state, async control and model-specific hacks. MRV2 targets stable per-request slot assignments, sparse/incremental metadata updates such as staged writes, GPU-side index preparation/sampling where appropriate, reduced GPU/CPU synchronization, consistent graph capture, and single common runner with specialized behaviors elsewhere. Its documentation describes evolving/partially qualified functionality, not universal completion. This is a *design lesson*, not a suggestion to copy Python code.

**What zerogravity should adopt:** explicit scheduler output/data contracts; engine/worker separation; cache-aware admission and paged blocks; one source-independent model runner; persistent slots; multi-modal model processor reuse; source-independent public serving protocols; asynchronous CPU/GPU overlap; real task and request QoS/metrics. **Avoid:** PyTorch or CUDA as hard foundation; model or device assumptions in the request scheduler; huge Python model runner; extra process/network hops for direct embedded CPU inference; a server-named mandatory control-plane runtime for all modalities.

## 4. Comparative new-model and physical execution pathways

| Situation | llama.cpp | vLLM | zerogravity target |
| --- | --- | --- | --- |
| Another checkpoint of same architecture | Converter/metadata + compatible graphs | Same model class/weight loader | Source manifest + family adapter + scoped capability |
| Variant with new layer geometry | Model graph changes | Inheritance/shared layer/config | Extend family math, operation data, guards |
| New dense transformer | GGML model graph + metadata + tests | Optimized module or HF Transformers | Verified definition, CPU reference, canonical program, existing provider |
| Novel GDN/KDA/SSM or sparse attention | New graph and state handling, perhaps ggml kernels | New vLLM layer/cache/backend | New exact semantic op + domain schema + reference, add qualified provider |
| New quantization codec | GGUF types/converter, kernels | Config/method and kernels | Source/derived artifact codec, explicit compute encoding + independent quality |
| Vision/audio | mtmd processor and encoder graph | multimodal registry/models/Omni stages | Native verified assembly + typed media processor/task/clock |
| New GPU backend | GGML backend implementation | Platform + native kernels | RegionExecutionProvider/DeviceBackend with actual state bridge, lease/events |
| Multi-model service | Router/child processes | independent engines/replicas/frontends | One process native catalog/router/global resource + model-neutral engine |
| SDK embed | C library & integration | Python/Rust frontend paths | Single public zerogravity crate reexporting all capabilities, including serving |

**Extensibility ranking caveat:** vLLM is easiest for Transformers-compatible fallbacks, llama.cpp is direct for closely related architectures with existing GGML ops, zerogravity today has significant finite implementation/qualification setup. The target is not "zero code for any new model", but measured minimum change radius for known semantics and simple native source-trusted extension.

## 5. Historic zerogravity checkout and observed file tree (26-Oct baseline)

Historic read-only source root: /home/pvl/spaces/abird/src/zerogravity on dotlink-l5.
Original inspected SHA: 195099ca583ace766d075682927f590360c823a5, with hundreds of uncommitted refactor modifications/deletions. DO NOT use old path/line counts as current after resync.

~~~text
zerogravity/
├── Cargo.toml
├── README.md
├── AGENTS.md
├── crates/
│   ├── zerogravity-core/
│   │   └── src/
│   │       ├── accelerator.rs
│   │       ├── accelerator/program.rs
│   │       ├── resource_arbiter.rs
│   │       ├── session_root.rs
│   │       ├── device_events.rs
│   │       ├── device_queue.rs
│   │       ├── pipeline.rs
│   │       ├── targets.rs
│   │       └── ...
│   ├── zerogravity-artifacts/
│   │   └── src/
│   │       ├── artifact.rs
│   │       ├── source_weights.rs
│   │       ├── sharded.rs
│   │       ├── quant.rs
│   │       ├── acquisition/
│   │       └── cache/
│   ├── zerogravity-model-qwen35/
│   │   └── src/
│   │       ├── config.rs
│   │       ├── forward.rs
│   │       ├── plan/
│   │       ├── source/
│   │       ├── reference/
│   │       └── tokenize/
│   ├── zerogravity-model-gemma4/
│   │   └── src/
│   │       ├── config.rs
│   │       ├── tensors.rs
│   │       ├── variant_26b.rs
│   │       ├── variant_31b.rs
│   │       ├── text/
│   │       └── execution/
│   ├── zerogravity-cpu/
│   ├── zerogravity-gpu/
│   │   └── src/
│   │       ├── cuda/
│   │       ├── hip/
│   │       ├── vulkan/
│   │       ├── backend/
│   │       ├── resident_program/
│   │       ├── resident_inference/
│   │       ├── resident_provider.rs
│   │       ├── resident_leases/
│   │       └── physical_service/
│   ├── zerogravity/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model.rs
│   │       └── prepare.rs
│   └── zerogravity-cli/
│       └── src/
│           ├── main.rs
│           ├── run_cli/
│           ├── serve/
│           ├── pull/
│           └── pipeline.rs
├── tools/gpu-qualify/
├── tests/
├── docs/
└── .agents/
    ├── docs/
    ├── plans/
    ├── research/
    ├── prior-art/
    └── runs/
~~~

**Historical source counts**, source .rs under crate src (not production-only, generated/qualification included; vendor excluded):
| Crate | Rust files | Source lines | Other note |
| --- | ---: | ---: | --- |
| zerogravity-gpu | 590 | ~315,159 | ~120,412 from generated*.rs |
| zerogravity-model-qwen35 | 88 | ~34,276 | concrete family state/plan and tests |
| zerogravity-artifacts | 57 | ~32,246 | acquisition/store/format |
| zerogravity-cli | 43 | ~25,225 | current HTTP and source-specific request orchestration |
| zerogravity-core | 43 | ~19,316 | generic and historically special-case ops |
| zerogravity-cpu | 42 | ~11,686 | numerical/CPU source/reference |
| zerogravity-model-gemma4 | 24 | ~6,677 | source contract and candidate paths |
| zerogravity public facade | 16 | ~3,464 | model/config and source-specific entry |

Other historically inspected lengths: core/src/accelerator/program.rs ~3,002; gpu/src/resident_provider.rs ~2,576; gpu/src/resident_inference.rs ~2,081; gpu/src/resident_program.rs ~3,668; cli/src/serve/engine.rs ~2,473; model-qwen35/src/lib.rs ~2,298; cli/src/serve/http.rs ~1,648. Large generated files alone are not an architectural failure; *family-specific dependencies and duplicate lifecycle owners* are.

### Existing working architectural strengths

- Generic core exports MatVec/attention/RMS/GDN operator regions, MathematicalBackend, PlanOperation, ForwardPlan, value IDs, source/numeric precision, state effects and typed domains.
- Core includes ResourceArbiter with global physical key/reservations, DeviceEventLedger/quarantine, FairDeviceQueue, SessionAuthority/RootAttempt/RootBinding and typed pipeline structural validator.
- Artifact layer offers GGUF/safetensors/checkpoint/cache/immutable original-shard source ownership and Xet/HF/other transport.
- Qwen family forward.rs establishes one source-verified family traversal that evaluates or records typed math through MathematicalBackend; source roles, profiles and checkpoints are separate concerns.
- Gemma family is separated into config/tensors/variants/text/execution and source-binding candidates.
- CPU source and numerics are independent from optional CUDA/HIP/Vulkan providers.
- GPU has real resident inference programs, physical services, leases, typed ABI, source qualification and bounded actual CUDA cohorts.
- CPU router and GPU router have bounded pending queues, model aliases, source memory charges, context and streaming, with source-specific evidence and limits.
- Pipeline config is source-neutral structurally typed, with audio/image/video/text/decision ports; it explicitly refuses graph execution until qualified.

### Historical architectural coupling / changes proposed

| Historic source | Coupling observed | Better owner after redesign |
| --- | --- | --- |
| core/targets.rs | model-named catalog/status in generic core | cold builtins/catalog assembly; core keeps generic schemas |
| zerogravity/src/model.rs | Qwen/Gemma inspection enum | model definition registry at application composition |
| core/accelerator/program.rs | generic plan alongside distinct source-specific arithmetic contracts | model-specific exact recipes in family; shared math/effect types in core |
| gpu → model-qwen35 Cargo dep | reverse dependency from physical provider to semantic family | remove, bind both in facade or family specialization adapter |
| gpu/resident_program/{gemma_text,qwen_*} | source-specific recipes amid shared native physical workers | family eligibility/recipes outside shared provider; backend owns kernel/image/lifetime |
| gpu/resident_inference/{qwen_source,gemma_text} | source-specific session progression in physical crate | generic physical step + family adapter + engine SessionAuthority |
| cli/serve/gpu_engine/route.rs | Gguf/FixedPrompt/Qwen35Cuda model/request enums | model-neutral executable-model/session interface in engine |
| cli/serve/engine.rs | LoadedQwenModel and QwenTokenizer directly in HTTP request engine | model/session orchestration engine; HTTP protocol in serving lib |
| cli/serve/mod.rs | CPU/GPU routers, source profile and backend model switching | server crate for HTTP mapping, model/placement engine for execution |
| cli/serve/gpu_batch.rs | finite source-aware waves and widths | qualified task-specific batching policy and prepared provider bands in engine |
| core/resource_arbiter.rs | global live resource service and contracts in foundational crate | shared IDs/lease semantics in core; one live physical authority in engine |
| core/session_root.rs | live session logic in core | core root schema/effects, engine sole TaskCoordinator |
| core/pipeline.rs | typed ports and structural config plus model intent | core task/multimodal type contracts; library pipeline checker/preparer/runner |
| CLI acquisition, devices, configuration | reusable business functions embedded in CLI | public zerogravity library models/store/devices/config; CLI only presentation |

Those findings derive from the historic refactor state. They are a migration hypothesis requiring a fresh source read later; they are NOT instructions to remove now-useful code blindly.

**Crate dependency issue inspected:** core had no model/backend dependency; artifacts→core; cpu→core+artifacts; gpu→core+artifacts and optional model-qwen35; model-gemma4→core and optional gpu; facade→models/cpu/gpu; CLI→facade. Model→GPU plus GPU→Qwen is a dangerous extension-pattern shape even if Cargo features did not produce an unconditional cycle. Remove reverse dependency to maintain model-neutral provider.

**What existed vs target:** ForwardPlan and typed effects exist; the general "SemanticProgram" and "ExecutableBundle" names did not appear as public defined Rust types in the inspected source. A TaskDriver was mostly a design/typed-pipeline concept, not a generalized production virtual execution framework. Tests/selected Qwen GPU cohorts were meaningful but did not certify general C1..C64, all models, distributed, media or fully qualified vLLM-equivalent performance. Do not confuse accepted finite captures with arbitrary source-run support.

## 6. Zerogravity's design documentation validation (26-Oct round)

Research root: .agents/docs, .agents/plans, docs. The review inventoried around 56 non-archive docs entries in .agents/docs, 45 plan entries and six public docs. It read governing documents, heading maps, active TODOs and selected critical sections; NOT every historical 15K-line handoff verbatim. Below are the key authority relationships and decisions for future agents.

### Governing reference docs

- .agents/docs/VISION_SCOPE_AND_PRINCIPLES.md — model/task/hardware user goals and no unknown semantics.
- .agents/docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md — single execution fabric; minimal vocabulary; SIR+bundle.
- .agents/docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md — exactly five logical authorities; domain Prepare not commit; one PublishRoot; device completion real.
- .agents/docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md — semantic operators vs pure subsets, TaskDriver, compiled StepProgram, bounded optimization, no workflows.
- .agents/docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md — typed model/provider/state/task extension seams and effect-driven fusion, not multiple engines.
- .agents/docs/SEMANTIC_IR_AND_COMPILER.md — detailed SIR/RIR/KIR/CIR mathematical legality; latest v4 two durable representations override earlier three/quadruple persistent IR wording.
- .agents/docs/ASYNC_RESOURCE_COMMAND_IR.md — liveness/aliasing, copies, readiness, events, coherence, no semantic device Commit.
- .agents/docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md — one RegionExecutionProvider for NativeKernel/CapturedGraph/AOT/Dataflow/Partitioned; state bridge and code lifecycle essential.
- .agents/docs/DEVICE_KERNEL_AND_CODEGEN.md — portable kernel contracts, Rust-authored policy, GPU/ISA-specific fusion with physical qualification.
- .agents/docs/COST_MODEL_AND_AUTOTUNING.md — critical path, not naive sum; three timescales (load/offline/online bounded); goodput constraints; topology/profiles.
- .agents/docs/OPERATOR_AND_FUSION_CATALOG.md — attention, GDN/KDA, MoE, lookup, output, stateful fused region alternatives.
- .agents/docs/NUMERICS_QUALITY_AND_BENCHMARKS.md — four truth claims, independent reference and full model/task quality, sampling, quant and bench comparisons.
- .agents/docs/STATE_CACHE_AND_TRANSACTIONS.md — state schema, COW, prefix frontier intersection, snapshot/recompute, cancellation.
- .agents/docs/ENGINE_CORE_EXECUTION_AND_PROTOCOL.md — EngineCore/Worker separation, scheduler outputs, persistent batch metadata and async event lifecycle.
- .agents/docs/SCHEDULER_BATCHING_AND_SPECULATION.md — token/work budgeting, group eligibility, target-authoritative speculation, state-safe prefill/decode.
- .agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md — vLLM-class serving, admission layers and joint cost/SLO selection.
- .agents/docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md — catalog/registered model/resident deployment/replica/aliases, exact route binding and shared resource ownership.
- .agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md — one-command local defaults, progressively exposed config, exact APIs and failures.
- .agents/docs/RUNTIME_SERVING_AND_UX.md — native SDK, streaming, endpoint conformance, user diagnostics.
- .agents/docs/MEMORY_RESIDENCY_AND_IO.md and UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md — resident/streamed tensors under same lease/identity/arbiter, real asynchronous IO.
- .agents/docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md — lossy new source artifact separate from reversible device packing.
- .agents/docs/GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md — coherent host/GPU resources, OS page cache, WMMA and safety.
- .agents/docs/EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md — native MTP/n-gram/Engram, target-authoritative verification, derived-source checks.
- .agents/docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md and HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md — row-slice decomposition, true event first-use, no unsafe GPU+NPU system changes.
- .agents/docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md, UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md, VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md — native model components, codec/time/typed state, non-AR task drivers.
- .agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md — five output frontiers, clocks, codecs, turn/barge-in, session/reconnect and QoS.
- .agents/docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md and REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md — four placement maps, transfer plan and remote epoch/failure trust.
- .agents/docs/SECURITY_AND_RELIABILITY.md, INVARIANTS_AND_FAILURE_SEMANTICS.md — required trust/event/quarantine/lease/tenant protocol.
- .agents/docs/PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md, MODEL_SEMANTICS_AND_COMPATIBILITY.md, FORMATS_QUANTIZATION_AND_LOADING.md — source math, models, codec, task and extension proof.

### Current (historic) owner-plan documents

- .agents/plans/README.md / TODO.md — one owner/current implementation queue, NOT duplicate this new design file into active queue until authorized.
- .agents/plans/pipeline-composition/STRUCTURE_AND_NAMING.md — one owner, dependencies down, family source/math in model crates, base operator/allocator shared; pre-v1 can break APIs, per later user instruction.
- .agents/plans/pipeline-composition/REGIONS_AND_PROFILES.md — semantic primitive→nested regions/tasks→orthogonal workload profile, one flat canonical semantics + one prepared physical plan, no recursive interpreter.
- .agents/plans/pipeline-composition/SHARED_EXECUTION_ARCHITECTURE.md — source to semantic ForwardPlan to prepared physical plan to original attempt/Ready/SessionAuthority; remove Qwen/Gemma leakage and duplicate pools.
- .agents/plans/pipeline-composition/PROFILE_POLICY_AND_COMPONENT_REUSE.md — allowed explicit approximation profiles, replaceable allocator/implementation under common accounting; no unsafe silent change.
- .agents/plans/serving-and-speculation/BATCHING_AND_PAGED_STATE.md and FUSED_GPU_BATCHING.md — B2/B4/B8 finite evidence, temporal vs request batch, paged hybrid state, graph readiness, release gates.
- .agents/plans/cpu-gpu-optimization/BACKEND_PARITY.md, SHARED_MEMORY_AND_BOUNDARIES.md, SHORT_HF_RESULTS.md — independent CUDA/HIP/Vulkan semantics and physical safety; no copied performance claims.
- .agents/plans/numerics-and-context/CORRECTNESS_POLICY.md — semantic, per-op, full model quality, determinism distinct; exact independent reference.
- .agents/plans/cli-device-routing/README.md — -d device, backend vs physical identity, fit/UMA domains, all-or-none reservations, qualified split.
- .agents/plans/memory-and-streaming/TODO.md — source leases, pooled backings, SSD/windows, paged/hybrid state/eviction.
- .agents/plans/model-execution/README.md — Qwen2B→Qwen27B→Gemma E2B/26B/31B→Flash; metadata-only vs qualified source.
- .agents/plans/multimodal/TODO.md, distributed-execution/TODO.md — important unimplemented scopes, no premature claims.
- .agents/plans/IMPLEMENTATION_ROADMAP.md P00–P23; EXPERIMENT_PROGRAM.md H01+; RELEASE_ACCEPTANCE_GATES.md; INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md.

### Conflicts resolved in the vNext design

1. Older document prose suggests three permanent IRs (SIR/EIR/DPI) or four named stages; **latest v4 contract wins: SIR and Bundle only enduring**.
2. Older StateDomain commit examples vs atomic whole-root safety; **domain Prepare only, TaskCoordinator PublishRoot once**.
3. Some scheduler narratives imply CPU per-token host state update; **device-native bounded execution islands permitted under exact state/effect/sampling and real completion constraints**.
4. Some architecture sketches imply one default model/one server; **one model-neutral engine and independent serving library, public lib as only ordinary SDK entry, standalone direct run does not require server**.
5. Older 12-Oct public engineering specification had packaging-only status; historic 26-Oct repository contained much more actual source and tests. **Source/tests and latest plan completion outrank old document status.**
6. Model registry placeholders/metadata-only entries are not executable model authorization.
7. No general autograd/training or unrestricted external plugins just to future-proof.
8. Local run should not block on remote, NPU, wafer or SSI proof gates.
9. Enormous specialized native code is acceptable as a bounded qualified physical candidate, not an architecture-specific duplicate session/server.
10. The 27-Oct user explicitly permits breaking ANY pre-v1 APIs, yet design-only archival phase must not touch active source.
11. The 27-Oct user's new library/serving requirements supersede any previous file tree showing all HTTP code inside CLI.

## 7. Comparative ownership table (historic source vs new design)

| Concern | llama.cpp | vLLM | Historic zerogravity | Fabric vNext |
| --- | --- | --- | --- | --- |
| Family math | src/models | model_executor/models | model-* (partial) | model-* only |
| Source loader | llama-model-loader/gguf-py | model_loader/load_weights | artifacts + family | artifacts generic bytes; family verifies roles |
| Semantic ops | GGML | PyTorch/vllm layers | core/accelerator/program | core/semantic/program, model-specific recipes outside |
| CPU kernel | ggml-cpu | CPU platform | zerogravity-cpu | CPU reference/provider |
| GPU kernel | ggml-cuda/hip etc | csrc/Triton/attention | zerogravity-gpu | GPU backend/kernel providers, model eligibility elsewhere |
| Cache/state | llama-memory/kv | v1/core/kv_cache | core/session_root + family + gpu | generic state transaction in engine, paged KV domain strategy |
| Scheduler | server-context | v1/core/sched | device_queue + cli/gpu_batch | engine task-aware scheduler |
| Physical worker | GGML backend | v1/worker/executor | gpu/physical_service | physical provider/worker + engine admission |
| HTTP/stream | tools/server | entrypoints/openai | cli/serve | **zerogravity-serving standalone library** |
| Public Rust SDK | C lib; bindings separate | Python/Rust frontends | zerogravity facade incomplete | **zerogravity public library exposes all APIs** |
| Multi-model router | server-models/process | engines and replicas | CPU/GPU model-specific routers | engine catalog/residency/router, exact immutable binding |
| Multimodal | mtmd | multimodal/Omni | pipeline validation/family | typed ModelAssembly + TaskDriver + serving |
| Distributed | RPC backend | distributed | plans | one Bundle placement + trusted remote worker |

## 8. Original three-way recommendations retained

**Keep/adapt:**
- Model family as explicit unit of ownership and minimal diff radius.
- Canonical source-aware tensor math and independently selected hardware execution.
- vLLM-class continuous scheduling, KV block accounting, async dispatch and slots.
- A model-neutral executable-model interface; request and output protocols not family-specific.
- One co-located local process and service; no mandatory external proxies.
- Shared immutable weights but private session roots.
- Hardware capabilities/evidence, independent CPU+CUDA/HIP/Vulkan qualification.
- Correctness then performance on full mixed request/service workload.
- Fused whole region, batch/row and source-specific mathematical optimizations as optional guarded candidates.
- Simple command-line run and serve UX, with all implementation in libraries.

**Avoid:**
- mandatory inference process per model, or independent runtime per backend/modality.
- growing core family-name enum/registry and shared GPU runner.
- CPU per-op tensor copies or mandatory host sync between device-resident stages.
- false native model support from test fixture/shape or metadata-only entry.
- training/autograd/equality saturation in token critical path.
- global GPU scratch memory lock at each kernel.
- universal LRU spanning weights, state and SSD.
- external text APIs misrepresented as native region backend.
- silent output/model quality changes, wrong quant source or converted checkpoint identity.
- collapsing queued work, dispatched work, physical Ready, StateRoot publish and client delivery into a single success flag.

## 9. Remaining research/evidence gaps

The 26-Oct comparative analysis was a source-tree and architecture review, not a complete line-by-line verification of every upstream module or a throughput benchmark. Some upstream docs and code have changed by 27-Oct and will change again. Before performance implementation:
- Pin exact llama.cpp and vLLM commit/feature branch and vLLM-Omni release; verify MRV2 readiness and active scheduler/cache APIs, not merely design draft.
- Check precise model support and quantization/attention in the actual selected upstream versions.
- Verify current zerogravity tree/implementation after explicit resync; previous 590 GPU Rust file count/paths are historical only.
- Independently run semantics, state, physical/resource, API and matched performance gates.
- Avoid equating source code counts with maintainability or number of supported models.
- Compare vLLM M1/M2 C1/C4/C8/C16/C32/C64 and llama.cpp parallel slot/context runs only when same source/quality/task and hardware device/driver are admissible; include requests offered/rejected, TTFT/ITL, throughput, memory/energy and tails.

This is enough research to design the next-generation abstraction; it is not yet enough to claim zerogravity equals or exceeds other engines in performance.

## 10. Primary external links and exact file anchors

### llama.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/CMakeLists.txt
- https://github.com/ggml-org/llama.cpp/blob/master/src/models/models.h
- https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma4.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend-reg.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/CMakeLists.txt
- https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/CMakeLists.txt
- https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md

### vLLM
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma4.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/output.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/input_batch.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/registry.py
- https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py
- https://github.com/vllm-project/vllm/blob/main/vllm/platforms/interface.py
- https://github.com/vllm-project/vllm/blob/main/rust/Cargo.toml
- https://docs.vllm.ai/en/latest/design/arch_overview/
- https://docs.vllm.ai/en/latest/design/model_runner_v2/
- https://docs.vllm.ai/en/latest/contributing/model/basic/

These URLs are references to upstream default branches, NOT immutable benchmarks. Freeze actual commit SHA before quoting exact per-line source in future evaluations. Confirm source licenses before adapting code.

**Next step for another agent:** DO NOT start refactoring from these historic paths. Read HANDOFF.md and wait for the user's explicit next-phase resynchronization request.

## END FILE: .agents/plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md


---

## BEGIN FILE: .agents/plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md

# Fabric vNext — migration blueprint, acceptance gates and source-change crosswalk

**Status:** planned future work, execution HELD. User ordered 27 October 2026: archive architecture first; ONLY in a later explicitly initiated phase review/refit to the changed repository; only after that may the user authorize implementation. This plan may propose any pre-v1 breaking edit, but does not authorize edits or a rebase now.

Use [README.md](README.md) as the authoritative proposed architecture, [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md) for the new mandatory standalone serving library and public SDK and [PRIOR_ART_AND_REVALIDATION.md](PRIOR_ART_AND_REVALIDATION.md) for source investigations. The working zerogravity repository is changing quickly. File anchors below are historical examples, not current instructions.

## 0. Freeze provenance without disturbing the live owner

Historical 26 October reference inspected with dotlink-l5:
- /home/pvl/spaces/abird/src/zerogravity; Git HEAD 195099ca583ace766d075682927f590360c823a5.
- Live workspace had hundreds of modified/deleted/untracked artifacts while a family/source/runtime refactor was underway. Source tree analysis and file-line counts are from that particular state and may no longer apply.
- At the beginning of this 27 October archival phase, HEAD was eb38dd5e892f2120ac29cc2fb79b2dbe137a37fe and the worktree initially clean, as verified by read-only Git commands. That observation is NOT a full source re-audit, comparison, build or execution capability claim.
- Current active owner plans remain .agents/plans/TODO.md and per-plan TODOs. This new fabric-vnext folder is explicitly NOT integrated into their queue yet, to avoid colliding with ongoing work.
- Never check off old tests, reuse numerical permits, or equate a moved source file with previously qualified exact source/compiler/kernel identity.
- No destructive Git operations, pushes, branch rewrites, model downloads, device launches or service starts are implied by this reference.

## 1. Next user-authorized phase: resynchronization BEFORE redesign implementation

This is the first and ONLY authorized type of work when the user returns with "resync", "revalidate" or equivalent; even then, obey the exact later instruction.

1. Open AGENTS.md, .agents/plans/README.md and .agents/plans/TODO.md; detect in-flight ownership and relevant worktrees.
2. Inspect git status/HEAD/recent commits for the current working tree with dotlink-l5. Do not overwrite other agent edits, untracked files or abandoned source; no cleanup/reset/rebase.
3. Refresh Cargo workspace dependency graph, crate roles, source counts (exclude generated/vendor and distinguish test/qualification), compilation modes and feature maps.
4. Read current architecture docs and active plan updates at their actual revision, comparing any new decisions to this saved 27 October proposal.
5. Compare implementations for math, source binding, state, physical lease/event, model session, scheduler, routing, CLI/server and tests, including new adapters and code recently split/moved.
6. Identify already completed vNext items, newly introduced coupling, obsolete file anchors, safety/quality test evidence and unqualified future tasks. Preserve exact provenance of successful and failed runs.
7. Refresh 26-Oct upstream llama.cpp and vLLM pins/changes only where a materially new pattern matters; do not transfer their performance claims without source/profile comparison.
8. Produce a **diff between this frozen architecture reference and the new observed actual code**, marking Keep, Already Implemented, Adjust, Reject, Open and Blocked with explicit source paths, tests and reasons.
9. Reconsider target crate dependencies, especially the proposed new zerogravity-engine/zerogravity-serving crates and facade responsibilities, using up-to-date current code.
10. Present a revised prioritized implementation roadmap and boundary diagram for user approval. **Stop here unless a separate explicit user instruction authorizes source changes.**

The goals of this first phase are resync, context reconstruction, design validation, and adaptation. No broad implementation is silently bundled with it.

## 2. Proposed target crate layout and dependency graph

~~~text
crates/
  zerogravity-core/
    semantic/                         pure op/state/value schema
    program/                          exact SemanticProgram
    task/                             TaskDriver and typed frontier contracts
    model/                            generic ModelDefinition/Assembly contracts
    numeric/                          cast/reduction/source-equivalence contracts
    identity/                         versioned source/task/state/bundle IDs
    provider/                         small compile/execution capability types

  zerogravity-artifacts/
    formats/                          exact GGUF/safetensors/NeMo/etc.
    source/                           immutable source tensor windows
    quant/                            physical encoding/source-derived model identities
    acquisition/                      explicit trusted network retrieval
    store/                            caches and trusted files

  zerogravity-model-qwen35/
    config.rs                         family geometry and variants
    tensors.rs                        roles, shapes, ties, source-specific validation
    source/                           validated mapping from generic SourceWeights
    forward.rs                        single source-exact layer traversal
    state.rs                          GDN/KV/conv and effect schema
    tokenize/                         Qwen tokenizer/template
    plan/                             canonical SemanticProgram lowering
    specialization/                   exact source/numeric/shape recipes
    reference/                        independent CPU math fixtures

  zerogravity-model-gemma4/
    analogous components; PLE and shared-KV semantics distinct

  zerogravity-cpu/
    numerics/ kernels/ reference/ backend/

  zerogravity-gpu/
    backend/{cuda,hip,vulkan,...}/    physical device APIs and codegen
    kernels/                          model-neutral primitives
    compiler/                         guarded prepared region implementation
    memory/                           actual backing/buffer leases/physical pools
    transport/                        H2D/D2H and coherence
    worker/                           submission/event/ready/quarantine

  zerogravity-engine/                 proposed NEW cohesive crate
    catalog/                          exact metadata/qualified model binding
    router/                           alias→model instance, replica selection
    preparation/                      SIR→physical ExecutableBundle
    model_instance/                   loading, shared weights, hot reload
    scheduler/                        admission, persistent slots, stage strategies
    session/                          TaskCoordinator and authoritative StateRoot
    resource/                         unique live ResourceArbiter
    residency/                        weights, state, cache and SSD placement
    placement/                        compute/parameter/state/cache and transfers
    output/                           typed committed stream/output sequence

  zerogravity-serving/                REQUIRED NEW separate reusable library
    http/                             listener/request/response routing
    openai/                           API conformance
    responses/                        Responses API
    stream/                           SSE, bounded transport output
    websocket/                        when independently implemented
    realtime/                         media, clocks, barge-in
    auth/                             protocol/tenant ingress
    admin/                            separately authorized model management API
    lifecycle/                        bind/start/reload/stop/graceful drain

  zerogravity/                        public Rust SDK and builtin composition root
    lib.rs                            exported useful facade, not giant implementation
    runtime.rs                        public builder/adapters/reexports
    model.rs                          public model/store/inspect APIs
    run.rs                            public inference/task requests/streams
    serving.rs                        public reexports/builders for serving crate
    pipeline.rs                       public model/task composition APIs
    devices.rs                        public inventory/fit/placement/explain
    admin.rs                          public model/cache/alias/runtime management

  zerogravity-cli/                    THIN binary only
    args.rs                           flags/config/CLI precedence
    commands.rs                       call public zerogravity SDK
    output.rs                         terminal/JSON/progress formatting
    signals.rs                        ctrl-C → library cancellation
    main.rs                           process entrypoint

  tools/gpu-qualify                   optional qualification tools
  tests/                              cross-system tests
~~~

This tree illustrates responsibilities, not 70 mandatory new files. Avoid empty directories, excessive interface traits or a separate crate for each device/task. Model family adapters can use backend preparation interfaces through a one-way dependency; physical GPU library must not import a model family. The model-neutral engine imports only core/artifacts and abstract provider/model contracts; facade provides compiled model/backend registrations. The separate serving lib uses engine/core, not facade; facade uses serving. CLI imports facade and generic CLI dependencies, nothing internal for model/physical/HTTP.

**Crate DAG proof:** no cycles; cpu-only facade builds without GPU compiler or serving listener; serving-only crate tests compile without any model family, GPU, CLI or platform-specific experimental feature; independent embedded program depends only on zerogravity and can run a supported model/start API server; native static registries safe without runtime dynamic executable plugin ABI.

## 3. Legacy-to-target change radius (provisional, based on historical files)

| Historic source/area | Proposed move or redesign | Invariant during migration |
| --- | --- | --- |
| core/accelerator.rs | generic semantic ops/contracts in core | source/numeric/effects unaltered |
| core/accelerator/program.rs | canonical SemanticProgram + reusable typed op versions; family recipes removed | original ForwardPlan semantics and state versions preserved |
| core/accelerator/program/source_program.rs, typed.rs | separate registered arithmetic from model source-specific eligibility | named source arithmetic distinct from model identity |
| core/resource_arbiter.rs | core retains physical identity and lease types; live one arbiter moves to engine/resource | same global bytes/backing/epoch/alias proof |
| core/session_root.rs | core schema, engine/session live publication | exactly one writer and old-root/epoch compare |
| core/device_queue.rs | engine/scheduler global control plus device-local physical queue | queued ≠ submitted ≠ Ready ≠ publish |
| core/pipeline.rs | generic typed task/media port contracts and public pipeline checker; stage runner engine | shape/config check does not run source or network |
| core/targets.rs | generic schema only; model-slug catalog in facade/engine builtins | model intent is NOT runtime execution authorization |
| artifacts/model/* | family-specific source models to family crate; generic artifact parsing stays | exact source bytes/roles/auth ownership unchanged |
| model-qwen35/src/lib.rs | thin exports, state/math/loader into files | one Qwen model forward traversal, no duplication |
| model-qwen35/src/forward.rs | family authoritative exact math traversal | same GGUF/HF branches preserve arithmetic identity |
| model-qwen35/src/plan/* | family semantic program builders and complete effect/state schemas | no model-specific scheduler or GPU memory |
| model-gemma4/src/* | same family template with actual independent Gemma logic | PLE/shared-KV/MoE/attention distinct from Qwen |
| gpu/src/cuda/hip/vulkan | backend/compiler/native kernels/state event owners | independent device qualification and unsafe memory rules |
| gpu/src/resident_program/* | generic executable preparation, transfer/liveness and qualified physical candidates | encoded reader pinning, no new allocator |
| gpu/src/resident_inference/* | common resident step/flight, state bridge and events; model recipes in family | original publisher/root/lease behavior retained |
| gpu/src/physical_service/* | actual device command/event/lease/quarantine service | no fake CPU confirmation of GPU Done |
| cpu/reference_text, cpu/numerics | shared mathematical CPU kernel + reference service | no model source assumptions in common kernels |
| facade/src/model.rs | one compiled model registration and source inspection interface | metadata recognition not trained permit |
| facade/src/prepare.rs | adapter through engine/prepare, used by public Runtime | no duplicate model math |
| cli/src/serve/* | network/protocol code → zerogravity-serving; runtime scheduler/session → zerogravity-engine | no second model/session/arbiter |
| cli/src/run_cli/* | reusable model/source/device planning → public facade/engine; retain CLI parser only | no duplicate CLI-only implementation |
| cli/src/pull/cache_manage/devices | acquisition/model admin/devices behavior → facade and underlying libs | CLI simply displays progress/outputs |
| generated native GPU code, tests | keep grouped with exact backend provenance; move test orchestration to support | preserve code/ABI hash and negative receipts |
| docs/ENGINEERING_SPECIFICATION.md, README/whitepaper | after resync consolidate authority and update status | no historical false claim or overwritten handoff |

### Source-moving order

Avoid large mechanical renames before ownership is verified. Move complete *responsibility and all callers/tests together*. A file move may change module/type identity and compiler/kernel cache/historical provenance; preserve needed recorded source arithmetic names/aliases for tracing but do not keep an awkward duplicate runtime indefinitely.

1. Remove model→artifact inverse interpretation if generic parsers embed Qwen math.
2. Neutralize core family descriptors while leaving generic values, typed effects and numerical contracts.
3. Move source-specific GPU eligibility, complete source mapping and specialized mathematical recipes to family ownership.
4. Remove GPU crate's optional Qwen model dependency; ensure no new Gemma reverse dependency.
5. Extract model-neutral prepared executable/session interface before replacing router conditionals.
6. Create engine crate, lift common scheduler/state/resource/router operations from CLI and GPU-facing app code.
7. Create serving crate, lift HTTP/streaming conformance and listener code from CLI without changing existing external behavior unnecessarily.
8. Reexport all library-facing actions through zerogravity facade; port CLI commands to call only facade.
9. Qualify second family/second task without production model-name branches outside registry/specialization.
10. After basic correctness closure, advance persistent batching, paged hybrid state, speculative and placement variants in measured order.

## 4. Future implementation program (after separate authorization)

### Phase 0 — reproducibility and source correctness gate

- Agree frozen implementation starting revision, independent temp test run, no active owner conflicts.
- Make source/artifact/build identities and affected worker/provider model+quant+shape+numeric eligibility explicit.
- Preserve historical accepted results/failed holds and determine which tests need rerunning due to new module/code/ABI/driver identity.
- Build CPU minimal library and CLI, validate all currently supported model source/fit/refusal paths and original request outputs.
- Establish reusable acceptance runner for source/full-logit/StateRoot/transport with pinned reference, exact documented tolerance and negative cases.

Gate: old accepted behavior preserved; no "new model support" claim, no implementation status regression, no hidden model substitution.

### Phase 1 — core semantics and family boundaries

- Factor public typed SemanticProgram from existing ForwardPlan only as actual consumers need; no duplicate enduring graph or interpreter.
- Ensure Qwen full traversal is solely family-owned; Gemma independently implements actual PLE/shared-KV/GQA/MoE and does not reuse Qwen math by convenience.
- Formalize exact source tensor roles/ties/shapes, access, source identity, arithmetic/numerical profile and state effects; no model family strings inside core physical objects.
- Remove reverse imports in GPU and artifact/CPU shared primitives; use Cargo feature matrix to prove it.
- Include independent small CPU operator and full model state proofs before attempting optimized provider reuse.
- Separate source artifact quantization, temporary compute dtype and explicitly derived approximation model identities.

Gate: model with existing ops touches only model crate + one registration + tests; new op touches exact operator/CPU/reference plus optional provider, not serving/session.

### Phase 2 — model-neutral engine authority

- Introduce zerogravity-engine as compiled source-independent engine library.
- Consolidate actual ResourceArbiter physical backing identity, multi-domain all-or-none reservation and device-local pools under one authority.
- Consolidate per-session TaskCoordinator/StateRoot publisher; fake Prepare/Ready/Failed and cancellation/unknown cases.
- Introduce model source/variant/loaded instance/replica/alias catalog and exact atomically pinned ModelGeneration; aliases do not mutate in-flight sessions.
- Generic prepared executable model/session interface with validate, prepare, start, submit, inspect committed output, cancel, close. Coarse calls, no vtable per op.
- General ingress, scheduling and resource owner scopes, engine event queues and typed output stream.
- Existing CPU and CUDA Qwen routes use same engine without model-specific HTTP switch; physical workers retain original flight/lease evidence.

Gate: one Runtime handles multiple aliases sharing immutable weights and private roots; concurrent requests survive abort, unknown device work, alias reload and pressure safely.

### Phase 3 — first-class public Rust library

- Curate zerogravity public SDK Runtime, Models, ArtifactStore, DeviceInventory, Task/Pipeline, State/Cache/Snapshot, Explain/Policies, Metrics and Err types.
- Forward feature selection to engine/artifacts/models/backend; retain CPU-only base.
- Expose model pull/inspect/register/load/unload/alias, all CLI present actions, admin rights and placement/reason APIs.
- Provide direct async run and bounded events, cancellation, clean process-agnostic lifecycle (no SIGINT or process exit from library).
- Create an embedded Rust example with only dependency zerogravity, no CLI or server, running source-verified text.
- Verify Rustdoc, examples, no private project type leakage, no library import of CLI.

Gate: CLI "run"/"inspect"/"pull"/"models"/"devices"/"pipeline" semantics available programmatically, with typed results and errors.

### Phase 4 — separate zerogravity-serving library and CLI thinning

- Create zerogravity-serving independent library depending only on engine/core.
- Move HTTP parsing/endpoints, OpenAI client protocol mapping, SSE/streaming, listener/backpressure/security and graceful lifecycle.
- Support exact current qualified OpenAI subset then incrementally Responses, embeddings, media protocols under explicit feature/acceptance.
- Serving from embedded Rust app, with configurable address and auth, is functionally equivalent to CLI serve.
- Reexport serving APIs/builders in zerogravity library without facade↔serving cyclic dependency.
- Remove source-specific CPU/GPU model session routing and worker scheduling from CLI serving modules; engine owns that logic.
- CLI serve uses zerogravity::serving and owns only argument path + terminal output + ctrl-C.

Gate: a program depending only on zerogravity starts/stops server, manages aliases, submits independent direct run concurrently; CLI crate contains NO inference/source/router/protocol implementation.

### Phase 5 — second-family independent integration drill

- Bind Gemma4 E2B source-real CPU and independently qualified provider/candidate through engine. Preserve PLE, attention, softcap, architecture-specific state and original source checks.
- Verify Qwen and Gemma simultaneously registered; no new HTTP server, allocator, session coordinator or core branch.
- Prepare first code-level "new known-op family" extension test measuring touches beyond family/registration/tests.
- Include Qwen2B and/or Qwen27B compatibility as next source-real priorities, not magic inheritance of E2B/GGUF/source evidence.
- Use exact source and backend/driver, no inconclusive match based only on synthetic operation fixtures.

Gate: second model runs through common SDK + server with identical lifecycle; source-specific math remains independent.

### Phase 6 — production-class scheduling, state and GPU physical pipeline

- Stable persistent batch slots, generations and incremental GPU row metadata (vLLM MRV2 principle).
- True within-model native continuous batching, mixed chunked prefill/decode/verify under token/stage budgets, fair aging and output credits.
- Cross-stage CPU/tokenizer/SSD prep and GPU execution overlap under actual device event/fence; no per-token graph search/JIT.
- Generalize paged hybrid KV/state pools, common-frontier prefix reuse, COW/restore, complete domain transaction and reference replay.
- GPU-native sampling/feedback and optional bounded device-native loops with exact grammar/RNG/effect/stop contract.
- True driver graph/captured workload support with held encoded readers, stable pointers, correct failure release.
- Joint costed choices among validated prefill/batch/fuse/quant/draft/cache/residency candidates, with no silent quality degradation.
- Explicit supported CUDA/HIP/Vulkan feature and numerical/physical qualification; no cross-backend inheritance.
- Compare against pinned actual vLLM C1/C4/C8/C16... request traces under identical model/quant/context and OS/device.

Gate: real complete requests show correct outputs, fair active/queued metrics, independent roots and better or truthfully negative TTFT/ITL/goodput with no hidden copies/leaks.

### Phase 7 — large dense source, SSD and more model families

- Qwen3.8 27B source-real indexed safetensors, tokenizer and complete model CPU/logit/state correctness first; borrowed source pointer and sharded source lifetime.
- Big immutable source windows, bounded SSD staging, actual source mutation/short-read/IO cancel/refusal and warm/cold IO/latency accounting.
- Reuse one logical immutable tensor identity for resident/streamed; code/graph IO read pins valid through completion. No second SSD model engine.
- Legal explicit GPU→CPU or multi-GPU layer placement only when full composite source/semantics/numeric effects/state bridges/transfer/physical footprint qualify.
- Gemma4 26B-A4B sparse MoE and dense31B independent source/checkpoint/PLE/attention/routers.
- Qwen Flash-Next GDN/QSA/Engram/MTP, variant-specific codecs and source-trained aux model paths, with target-authoritative speculation; do not download held models without separate permission.
- Explicit Q1/Q2/Q3/Q4/Q6/Q8/NVFP4/EXL3 codecs and CPU/GPU physical paths only as independently implemented, source-verified options.
- N-gram/MTP/routed expert group/cache optimized jointly under real resource budgets and full output quality.

Gate: at least one complete >single-GPU-memory model operates with truthful source/IO/capacity limits; no claims of high throughput from theoretical streamed fit.

### Phase 8 — non-token tasks and pipelines

- Actual encoder/embedding or ASR with different frontier; reuse same engine ResourceArbiter, worker, TaskDriver and session state APIs.
- Source-real model assemblies, processor/codec semantics, clock/timebase, typed media buffers, correct component-to-component residency.
- Mixed ASR→language→TTS pipeline through library TaskPipeline; nested structure compiled, not interpreted recursively each frame.
- Proper staged commits and transport DeliverySession, ACK, playback, turn cancellation/reconnect/late replies.
- Real-time QoS fairness/latency measured by complete chunk/segment, not arbitrary tokens.
- Typed decision head with verified schema/calibration and no fabricated free-text JSON scores.
- Server exposes only source-verified completed modal endpoints with accurate support receipts.

Gate: novel task added without new model engine or global token scheduler branch; native end-to-end output and safe cross-stage device leases measured.

### Phase 9 — advanced placement, graph-only accelerators and distributed execution

- Add richer topological capability/physical program lifecycle; zero mandatory distributed overhead for direct local run.
- Add qualified NPU row-slice or encoder stage under safe SoC power/fabric policy; no automatic unsafe system tuning.
- Graph/AOT provider can execute a whole compatible semantic region with state bridge, no microkernel host emulation.
- Local two-GPU and GPU↔CPU placement with actual transfer and composed numerical/StateRoot proof.
- Remote peer authorized and authenticated; copied vs remote-stateful execution candidates with measured RTT, transfer cost and exact epochs.
- Independent compute/parameter/state/cache maps, actual TransferPlan, frozen PlacementEpoch; safe change only at committed boundaries.
- Verify stale peer, split brain, lost ACK, partial transfer, unknown DMA and fallback refusal. No false exactly-once external delivery.

Gate: reference model state/output equivalence, real physical remote receipts/fault recovery and total task cost compare with collocated baseline; preserve local single-device code path.

### Phase 10 — product/release/architecture simplification

- Release correct single-binary CLI plus public Rust library and separate reusable serving library.
- Quality/support matrix, documentation capability language, installation/Nix/multi-OS packaging only after actual target build/runs.
- Simplify or remove unused traits, redundant IR versions, old model-slice execution branches, duplicated constructors, dead test-only files and outdated documentation examples.
- Publish matching external benchmarks, including neutral/negative cases and model/source/driver pin, goodput, TTFT/ITL tails, memory/energy.
- Update normative docs and whitepaper from proven implementation; archive superseded snapshots under history instead of keeping them live instructions.

Gate: source-independent extension tests + clean CI/artifact/license/security gating; no incomplete capability advertised as production.

## 5. Execution-critical correctness and safety acceptance catalog

The following are *in addition* to original plan gates and cannot be diluted during refactoring.

### A. Source/model/semantic correctness

- exact verified checkpoint and full auxiliary assembly, role/tie/dtype/shape, tokenizer/template/processor revisions and original quant encoding;
- errors for missing/duplicate/extra tensors, alignment/rank/size bounds, bad checksums, truncated and malicious file layout;
- one family forward schedule matching source equations for prefill, decode, teacher/forced, natural feedback and seeded sampler reference under actually claimed numeric mode;
- final full-vocabulary logits, intermediate representative op checks, all relevant mutable domain values and following-token continuity;
- architecture-specific RoPE, GDN/KDA/SSM, sliding/full/cross attention, softcap, sparse indexer, MoE exact expert IDs and learned aux sidecars;
- no model support inherited from metadata, related family or incompatible source quant.

### B. Numerical and quality

- **four claims separately:** semantic/state safety, operator numerical accuracy, complete model/task quality, determinism/batch invariance only where advertised;
- original GGUF/CPU vs pinned appropriate reference; original HF BF16/HIP/CUDA/Vulkan separate, different batch widths separately accepted;
- source numerical reduction/cast/precision/exp/silu/softmax contracts preserved with nonfinite handling, no universal epsilon;
- explicit quantized derived source measured against original source quality with independent held-out corpora; kernel exactness relative to selected quant model is a different claim;
- target-authoritative speculative sampler distribution, full rollback and source-trained draft independently checked;
- no error claims based on an unrelated upstream source, forced teacher trajectory or only top-1 agreement.

### C. State and concurrency

- one authoritative root publication per session; all required domain versions agree on root epoch/source/frontier;
- stale attempt, double Ready, wrong root, canceled attempt, mixing model/quant profiles and branch/speculated candidate cannot publish;
- COW private state fork/snapshot/recompute after full/partial/hybrid prefix state; same-prefix across state domains verified at valid common frontier;
- graph/in-flight buffer readers retained including encoded and not yet submitted views; ABA generation poison tests;
- interleaved requests, mixed batch refills, ragged tails, masks, source-specific phase/width changes and active cancellations show no cross-session state;
- slot metadata changes incrementally, but model math and logical epochs never mutate absent source permit.

### D. Physical capacity/lifetime

- backing charged once despite aliases, views and unified domains; all physical growth and unmaterialized reservations included with OS headroom, cgroup and GPU driver realities;
- atomic multi-domain reservation for required GPU/host/transfer domains; no independent wallet overcommits;
- allocations/IO after explicit admission only; no hidden model bootstrap without accounting;
- real Submit/Ready/Failed/Unknown distinctions: dropping Future does not release DMA reader; unknown retains/quarantines until proven safe;
- backpressure avoids lock-held waits, cold IO, compiler storm, infinite per-request state, GPU-wide cancels from one turn;
- resident path no mandatory SSD lookup/H2D copying, and source-backed SSD path authenticates owner and IO readiness;
- separate CUDA/HIP/Vulkan capability/ABI/compiler/driver and graph launch proofs.

### E. Serving/public lib

- exact source model alias pinned on admission, active session survives catalog reload; no hidden target switch;
- direct library TaskRequest and HTTP request share engine and loaded physical model source;
- public facade exposes *all* run, serve, manage, pull, inspect, configure, pipeline, device, event, status and diagnostic capabilities;
- zerogravity-serving independent compile/tests; zero CLI imports; embeds in test/GUI/app from public zerogravity dependency;
- CLI only flags/terminal/config selection/JSON printing/signal bridge, not inference/server implementation;
- OpenAI/Responses API conformance with actual official clients for supported endpoints, stream and errors; explicit Unsupported parameter otherwise;
- correct auth/loopback default, TLS external configuration, proper admin rights, tenant quotas, no secret leaks or state cross-contamination;
- stream vs logical commit vs socket send/ACK/media playback distinct, output credits bounded; slow sockets never keep GPU scratch indefinitely;
- cancellation from public Rust drop/token/CLI signal/HTTP disconnect, graceful shutdown, unknown worker recovery.

### F. Model-extensibility and composition

- new existing-op architecture in model family + one registration + tests only, measured changed shared source files;
- new stateful operation adds exact semantic contract/CPU reference and optional provider, no new TaskCoordinator;
- new optimized fused op or whole model replaces same closed semantic region with valid fallback;
- new device/graph/NPU provider through one small provider interface with correct state bridge and physical ledger;
- new non-token task through TaskDriver and typed values, no per-task model server or numerical-correctness shortcuts;
- nested task/region flattening/inlining vs original schedule and guarded fusion, no infinite cycle/alias;
- cross-model pipeline binds existing model instances, avoids duplicate weights and preserves private roots;
- remote fake/actual worker cannot publish root or skip authenticated state transfer epochs.

### G. Performance and science

- compare exact same checkpoint and source quant, tokenizer/template, context, output/sampling, task/quality, hardware/driver and profile;
- warm/cold process load, compilation/capture, cache hits/misses and IO costs separately captured;
- offered, queued, admitted, rejected, canceled and completed requests counted (avoid survivor bias);
- TTFT, true inter-token latency, P95/P99, committed output per second, response completion, useful SLO-goodput, fairness, peak physical/OS/SSD, joules where measurable;
- B1/2/4/8/16/32/64 and ragged mixed requests only when actual qualified model/device capabilities allow; real prefill vs cached continuation distinguished;
- native batch proof requires shared physical work, not N independent single-request loop;
- measure ablations before joint candidate optimizer, publish losing variants; no unproven vLLM/llama.cpp performance advantage.

## 6. Decision gates and crosswalk with P00–P23

The original existing roadmap P00–P23 is an established owner work breakdown, not something this side plan may silently replace. During resync map new conceptual phases to relevant existing ones, preserving all open technical requirements:
- P00/01/02/03: source, kernel, reference and format foundation;
- P04/05/06/07: semantic program, legal specialization, generic CPU/GPU execution and actual physical event ownership;
- P08/09: first complete real model and native execution/serving;
- P10/11/12: state/prefix/SSD hybrid, source sparse/hybrid math, routed expert residency;
- P13/14/15: extreme hybrid model validation, second distinct large family, speculation/drafting;
- P16: live continuous batching/QoS/joint optimizer;
- P17/18: broad model families/task/media stages;
- P19/20: heterogeneous/distributed placement and bounded autotuning;
- P21–23: release/security/performance/extraction gates and later coverage.
Do not invent completion from phase names; resync source before assigning a concrete status.

**Architectural dependency ordering is more important than file move order:** avoid making an engine-generalization gate that blocks an already useful independently scoped numerical repair when both can proceed safely, but NEVER create a second state/allocator/server to work around unfinished boundaries.

## 7. Risk register

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Broad refactor loses accepted source arithmetic/state parity | Critical | Golden source fixtures, exact numeric profiles, targeted tests, original condition names and source hashes |
| Reusable serving library creates dependency cycle | High | serving→engine/core, facade→serving, CLI→facade; build graph test |
| "All functionality through facade" creates giant facade implementation | High | curated reexports and coarse builders, implementation remains in cohesive crates |
| Model family still embedded in GPU/HTTP/scheduler | High | dependency and changed-file extension drill |
| Semantic compiler added to token hot path | High | prepare once, measure per-step CPU sync/alloc, force portable baseline |
| Resource authority split across local runtime instances or device names | Critical | physical identity merging, process-wide original shared ledger, all-or-none multi-domain reserve |
| Logical model state publication conflated with GPU Ready | Critical | mandatory prepared-domain→single PublishRoot tests, poisoned physical events |
| Captured unsubmitted graph reader evicted | Critical | encoded/captured reader pin from command encoding to actual completion/graph retirement |
| vLLM-style paged KV wrongly considered universal state | High | state-domain interface + common-frontier restore and recurrent tests |
| Broad multi-modal TaskDriver grows into workflow DSL | High | trusted few stages and typed effects; external tools outside |
| Model ID/metadata treated as source/runtime qualification | Critical | finite capability matrices, typed Unsupported and source-real tests |
| Dynamic optimizer uses unqualified/approximate profile silently | High | immutable resolved profile, evidence IDs and safe fallback |
| C8 larger batches look fast but starve real-time/long prefill | High | offered-work goodput, aging, QoS and p95 with multi-tenant tests |
| UMA/GTT double counts available physical RAM | Critical | real OS/driver/cgroup headroom, one domain for coherent memory |
| Single source-specific "GPU serving" route grows endless branches | High | engine prepared-model adapter, serving protocol-only |
| Distributed mode becomes mandatory RPC | High | local no-peer fast path, frozen prepare + remote only when planned/approved |
| Historical false "all tests passed" after abbreviated gate | High | incomplete/timeout/held scopes explicitly reported |
| User intent to archive accidentally triggers implementation | Critical | this file's all source phases HELD; handoff requires later user authorization |

## 8. Self-evaluation rubric for the future architecture

Score qualitatively, then prove with tests and metrics:
1. **Findability:** an engineer knows exactly which module owns source role, task stage, kernel, buffer, or HTTP without chasing model-specific branches.
2. **Composability:** Qwen CPU/selected CUDA and Gemma can use one state, scheduler, serving API and resource authority.
3. **Correctness:** full source-model/numeric/StateRoot/physical/event proof, no silent fallback or model substitution.
4. **Extensibility:** measured minimal production change radius for new model/op/backend/quant/task/transport.
5. **Performance:** native fast path has no per-token model registry/compiler/JSON/global lock or avoidable H2D/D2H; measured tail/goodput.
6. **Physical safety:** cancellation/unknown DMA/eager/capture/residency honor ownership and actual release.
7. **Library ergonomics:** one dependency zerogravity for external Rust users; separate serving lib and thin CLI.
8. **Multimodal truth:** source clocks/codec/layout/solver/task-state/output boundaries are typed, not pretended text tokens.
9. **Maintainability:** no giant trait zoo or duplicate runtime; finite trusted fallback and clear proof stage.
10. **Evidence traceability:** no old source/driver/branch receipts reused after refactor without correct identity.

The end-state is a source-faithful, intrinsically task-aware execution substrate that can be tuned from a single CPU/GGUF model to a fused whole-model GPU program, SSD-backed dense inference, multi-model compositional service and authorized distributed audio/video AI—**all with one stable conceptual engine and one public Rust SDK**.

**This file authorizes no actions.** Await next-phase source resync and the user's later implementation instruction.

## END FILE: .agents/plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md


---

## BEGIN FILE: .agents/plans/fabric-vnext/TODO.md

# Fabric vNext — held future plan queue

Author: Project owner
Co-authors: GPT-6
Owner: @pvl — no implementation agent assigned
Base observed when archive started: eb38dd5e892f2120ac29cc2fb79b2dbe137a37fe (2026-10-27 read-only observation, NOT a frozen implementation base).

**STATUS: DOCUMENTATION COMPLETE ONCE VERIFIED; EXECUTION HELD UNTIL USER STARTS NEXT PHASE.**
This plan is deliberately *not* linked into or promoted on the active .agents/plans/TODO.md queue. The user asked for side documentation, preserving all context, so the current implementation owners can continue unaffected. The next step is user-initiated current-repo resync, not implementation. Do not check off tasks on historical test evidence.

Read [HANDOFF.md](HANDOFF.md), [DECISIONS_AND_OPEN_QUESTIONS.md](DECISIONS_AND_OPEN_QUESTIONS.md), [README.md](README.md), [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md), [PRIOR_ART_AND_REVALIDATION.md](PRIOR_ART_AND_REVALIDATION.md), [IMPLEMENTATION_AND_ACCEPTANCE.md](IMPLEMENTATION_AND_ACCEPTANCE.md).

## Committed

None in this plan. The reference is side documentation only; nothing has been committed/released/pushed on account of this request.

## Uncommitted — execution HOLD

### Phase 0: on next EXPLICIT user resynchronization request only

- [ ] Confirm the user has expressly initiated the **read-only resync/revalidation phase**.
- [ ] Read the full frozen Fabric vNext pack and literal user amendments, then AGENTS.md and current owner plans.
- [ ] Inspect current live HEAD, branch, worktree, recent commits, in-flight agents and true Cargo feature/dependency graph. Protect third-party work.
- [ ] Reconcile current source/architecture docs/active TODOs with historical 26-Oct analysis; classify old path anchors obsolete/new/already done.
- [ ] Reconfirm current in-source model families, generic semantics, numerical policies, physical provider, queues, state publication, resource services and source ownership.
- [ ] Reconfirm current model-neutral engine/SDK/CLI/serving responsibilities and look specifically for newly added server/engine library work.
- [ ] Re-evaluate public zerogravity facade + standalone zerogravity-serving crate + thin CLI against exact new code; revise module names/ownership for simplicity without weakening the newest user decision.
- [ ] Validate distinction among mathematical region, native ModelAssembly and multi-model TaskPipeline; complete-domain state and typed media clocks.
- [ ] Crosswalk every source/design gap with the current active P00–P23 workstreams, including source/quality/physical and model/backend holds.
- [ ] Produce source-grounded diagram, gap table, revised implementation plan, test evidence ledger and explicit remaining decisions. **Stop without coding.**

### Phase 1: separate future implementation authorization required

- [ ] Receive explicit user permission to start implementation after the source-grounded resync.
- [ ] Freeze a reproducible implementation/test baseline without corrupting active work.
- [ ] Introduce minimal generic SemanticProgram/closed-region and prepare/bundle paths from existing ForwardPlan; avoid second IR.
- [ ] Remove family math and source-specific source/ABI recipes from generic core, artifact, GPU and serving implementations.
- [ ] Refactor model family crates and reverse dependencies, no GPU → Qwen/Gemma import.
- [ ] Extract one model-neutral zerogravity-engine or equally enforceable cohesive engine boundary; one TaskCoordinator/StateRoot and physical ResourceArbiter.
- [ ] Provide source-qualified model instance/session/TaskDriver interfaces and cold built-in registration, not model-name switches in runtime.
- [ ] Implement public zerogravity Rust library for ALL functionality (run, inspect, load/pull/cache, admin, device planning, pipelines, metrics and serving).
- [ ] Create independent, embeddable **zerogravity-serving** Rust library for HTTP/OpenAI/Responses/stream/media service protocols and lifecycle; no CLI/model-kernel imports.
- [ ] Make zerogravity-cli ONLY call exported zerogravity library and own args/env/terminal/signal/output.
- [ ] Prove separate library embed and server start/stop, direct run and HTTP share same model/resource/state; strict auth/stream/errors.
- [ ] Prove Qwen3.5 0.8B/2B source-real CPU and selected GPU state/numeric/regression gates survive changes.
- [ ] Add Gemma source-real model through same engine and server without source-specific allocator/HTTP/scheduler branching.
- [ ] Record accepted and failed/unsupported paths per actual profile/driver/model/corpus and explicitly leave unrun hardware cases open.

### Phase 2: high-performance and model breadth after foundation gates

- [ ] Persistent GPU slots, incremental row metadata, actual async bounded step submission and retirement.
- [ ] Native source-qualified continuous batching, real temporal prefill/decode overlap and fairness/ageing.
- [ ] Paged and complete-domain hybrid state/COW/prefix checkpoint reconciliation including sparse and recurrent state.
- [ ] Region fusion/capture/device-native feedback; optional speculation and draft/target full state semantics.
- [ ] Unified physical residency and SSD streaming on one source identity with byte-budgeted cache and real completion.
- [ ] Qwen3.8 27B exact source/CPU/selected GPU, Gemma E2B→26B-A4B→31B independent functionality; then richer sparse/quant targets.
- [ ] First real non-token task and multi-model typed pipeline via same public library/engine/serving.
- [ ] Qualified CPU+GPU+NPU, local split and authorized remote region workers only with actual source/state/physical/latency proof.
- [ ] Matched vLLM/llama.cpp complete-request/quality/TTFT/ITL/P95/goodput/memory/energy benchmarks and negative findings.
- [ ] Minimality/extension drills for new model, novel stateful op, backend, task, transport with measured shared source change radius.
- [ ] Consolidate normative docs/public whitepaper/current status after actual implementation acceptance.

## Tests and verification — design archive only

- [ ] Verify the planned side-pack files exist, are UTF-8, complete, link internally and contain user amendments and no misleading runtime/perf qualification. This archive-only verification can be done NOW with a read-only check.
- [ ] Confirm **only new side documentation paths** changed during archive writing; source, active TODOs, binaries, models, device settings untouched. This archive-only verification can be done NOW.
- [ ] Confirm the next-agent handoff prompt includes explicit resync-first/no-source-change instruction.
- [ ] Later: enforce Cargo acyclic layering including serving→engine and facade→serving and CLI→facade.
- [ ] Later: public zerogravity-only embedded Rust direct run/server examples compile without CLI, with one shared model/ResourceArbiter and exact source states.
- [ ] Later: source/semantic/quality/physical/serving/multimodal/distributed acceptance gates defined in IMPLEMENTATION_AND_ACCEPTANCE.md and original project owners.
- [ ] Later: no vLLM/llama.cpp performance parity claim absent true matched workload/driver/profile/quality evidence.

## Historical scope distinction

The historic 26-Oct file-tree review and 27-Oct document revalidation do **not** certify the present checkout or all capabilities. Earlier old HEAD 195099ca and new archive-start observed eb38dd5e must not be conflated. Upstream source links are moving branches, not pinned benchmark artifacts. No physical GPU/NPU execution, source pull, corpus testing, code refactor or new server operation was performed for these design documents.

Do not promote this side plan in the active parent queue, open a code PR or start the resync now. Await the user's explicit next step.

## END FILE: .agents/plans/fabric-vnext/TODO.md


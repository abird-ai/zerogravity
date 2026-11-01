Author: Project owner
Co-authors: GPT-6 (architecture research, 26–27 October 2026)
Owner: @pvl; later appointed implementation coordinator

# zerogravity Fabric vNext — transition architecture and implementation plan

**Latest checkpoint:** signed`7fc5d17`/`8591b52` land source/state owners and working
CPU SDK/embedded serving; full isolated gate passes. Main only, no new subagents.
Continue contained refinements, not another skeleton extraction.
Plan 2 includes Plan 1; Qwen expansion deferred. Historical docs-only holds below
are superseded. [Current source/crosswalk](CURRENT_SOURCE_RECONCILIATION.md) ·
[active phase queue](TODO.md). No new physical/production grant.


**Created:** 27 October 2026. **Current status: working CPU/runtime/server checkpoint; full GPU/SDK/CLI migration remains open.**
**Read-only source observation:** HEAD fadf2b1123309a2118725f2cc005d3edc7ea39f2 (l5, 27 Oct 06:50 UTC). The 26 Oct deep review had 195099ca; preceding refresh had eb38dd5. These are evidence timestamps, not frozen ongoing source. Recheck HEAD before coding.
**Physical root:** /home/pvl/spaces/abird/src/zerogravity via dotlink-l5.
**Ownership:** owner explicitly promoted this transition in the shared queue. Preserve Plan 1's requirements and all unrelated source/owner work; the current main session serializes integration.

## Working key paths

Enable facade features `cpu,model-qwen35,text,cache,serve` for local CPU plus HTTP.
No automatic model acquisition; local GGUF, configuration and approved tokenizer
must be present together. Existing source/format/host-budget refusals still apply.

- `Runtime::load_text`: admitted local sources; aliases share immutable ownership.
- `Runtime::request_text` / `Runtime::run`: direct bounded events/task streams.
- `Runtime::server`: independent server sharing the same engine; explicit bind,
  limits and authentication policy. `stop`/`join` do not shut down SDK tasks.
- `Runtime::shutdown`: closes admission and joins actual workers. Explicit calls
  from engine workers or the same runtime's loader/retirement callback return
  State before changing admission. Use an external owner for real shutdown;
  short router callbacks remain non-reentrant, implicit self-drop unsupported.
- `pipelines::Document`: bounded YAML/schema/entrypoint checking, not execution.
- `Runtime::devices`: inventory/policy; placement execution remains Unsupported.

Executed public composition example/test:
`crates/zerogravity/src/builtins/text/tests/serving_population/checkpoint.rs`.
It uses a real full-vocabulary CPU GGUF fixture, not fabricated successful events;
this establishes runtime composition, not trained-model numerical qualification.

GPU backends/routes still use their original physical owners; CPU engine migration
is not unified GPU scheduling. Source-free feature checks do not qualify hardware.
Pipeline execution, broad model administration, placement, media/distributed tasks
and unfinished active-contract bindings remain the tracked follow-up scope.

## 0. Binding implementation order (27-Oct design; now authorized)

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
- The original 27-Oct turn authorized documentation only. The latest owner instruction authorizes Fabric implementation and logical signed checkpoints.

## 1. Read the original whole-context archive

The preceding full architecture reference remains frozen:
- ../fabric-vnext/README.md — comprehensive semantic compiler + model/task/provider/resource/state/multimodal/distributed design.
- ../fabric-vnext/PRIOR_ART_AND_REVALIDATION.md — in-depth llama.cpp and vLLM file trees, model extensibility, vLLM Model Runner V2 and Omni patterns, detailed zerogravity v4/v5 docs crosswalk.
- ../fabric-vnext/PUBLIC_LIB_AND_SERVING.md — new independent serving library + complete exported Rust SDK.
- ../fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md — initial migration, user journeys, correctness/performance/physical gates.
- ../fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md — literal user instructions and design choices.
- ../fabric-vnext/HANDOFF.md — original context and read order.
- ../fabric-vnext/TODO.md — historical held work.

**Do not silently replace this historical material.** This transition pack supplements it with the second read-only current-code refresh, improved phase ordering, precise codec split, root/resource correction, complete feature inventory and actionable engineering commit sequence. The plan files in this directory and ../fabric-vnext are the maintained architecture source.

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

**Historical 27-Oct snapshot:** 41 commits from prior 195099ca baseline (39 through eb38dd5 plus ba7fb03 and fadf2b1). Recent final two touched fresh CUDA prefill generated capture and reference layout; treat as qualification-in-progress. At archive creation initial source tree had eight crates (core, artifacts, cpu, gpu, Qwen, Gemma, facade, CLI) plus gpu-qualify tool; two new structural crates intended: engine and serving. Existing source API is not general production HF CUDA, despite finite capture paths.

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

No combined archive is maintained. The canonical cross-window entrypoint is `.agents/docs/CURRENT_HANDOFF.md`; it points to the four maintained planning/status files and this transition set. Git history preserves superseded snapshots.

## 7. Minimum completion claims and guardrails

**Structural skeleton complete** does NOT mean runnable model support is new; it means all modules and crates exist and are compile-visible, dependencies enforce ownership, public API exists, and deferred operations fail closed.

**Migration complete** means all previously implemented supported behaviors, source/codec/CPU/GPU/CLI/serving/fixtures and their exact scoped safety are working under the new architecture, with no duplicate legacy runtime, no reachable todo! on a supported path, no missing current command or library equivalent.

**Physically qualified transition complete** requires the consolidated evidence and actual hardware/source tests for the relevant previously qualified tuples. Historical proof may guide expected results but cannot be retagged to a changed binary/toolchain.

No agent may assert current model/device/endpoint support from a bare trait, source codec, feature, captured test case, old report or attractive architecture diagram. No commits, pushes, tags, hardware probing/reset or untrusted source downloads merely from writing this plan.

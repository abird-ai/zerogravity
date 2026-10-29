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
- [ ] Inspect current live HEAD, branch, checkout, recent commits, in-flight agents and true Cargo feature/dependency graph. Protect third-party work.
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

## Tests and verification — maintained design set

- [ ] Keep this plan set internally linked and consistent with the canonical current handoff and four-file planning spine.
- [ ] Revalidate source-dependent statements against current main before implementation; historical SHAs remain provenance only.
- [ ] Keep the next-agent prompt source-first: read the canonical handoff, current status/traceability, then this plan; do not reconstruct deleted archive snapshots.
- [ ] Later: enforce Cargo acyclic layering including serving→engine and facade→serving and CLI→facade.
- [ ] Later: public zerogravity-only embedded Rust direct run/server examples compile without CLI, with one shared model/ResourceArbiter and exact source states.
- [ ] Later: source/semantic/quality/physical/serving/multimodal/distributed acceptance gates defined in IMPLEMENTATION_AND_ACCEPTANCE.md and original project owners.
- [ ] Later: no vLLM/llama.cpp performance parity claim absent true matched workload/driver/profile/quality evidence.

## Historical scope distinction

The historic 26-Oct file-tree review and 27-Oct document revalidation do **not** certify the present checkout or all capabilities. Earlier old HEAD 195099ca and new archive-start observed eb38dd5e must not be conflated. Upstream source links are moving branches, not pinned benchmark artifacts. No physical GPU/NPU execution, source pull, corpus testing, code refactor or new server operation was performed for these design documents.

Do not promote this side plan in the active parent queue, open a code PR or start the resync now. Await the user's explicit next step.

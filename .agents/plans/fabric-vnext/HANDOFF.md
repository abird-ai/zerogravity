# Fabric vNext — architecture design context and handoff

**31-Oct reconciliation:** this complete architecture remains required by the
now-active [transition superset](../fabric-vnext-transition/README.md).
[Plan/source crosswalk](../fabric-vnext-transition/CURRENT_SOURCE_RECONCILIATION.md)
records recent accepted Qwen work, preserved gaps and S0-first ordering.
Historical plan-writing holds below are not current implementation authority.


**Canonical current entrypoint:** `.agents/docs/CURRENT_HANDOFF.md`. Current source/tests and the four-file planning spine outrank historical source snapshots in this document.

**Date frozen:** 2026-10-27.
**Status:** DOCUMENTATION / REFERENCE ONLY. The user explicitly said to finish writing and report completion *before* any next phase. Do not initiate resync, source refactor, test, download, commit or deployment from this plan until their later instruction.
**Project:** zerogravity, /home/pvl/spaces/abird/src/zerogravity on connector **dotlink-l5** (pvl-l5).
**Initial 27-Oct archival-phase observed HEAD:** eb38dd5e892f2120ac29cc2fb79b2dbe137a37fe, checkout initially clean; this observation is provenance ONLY, not a readapted architecture review.
**Historical in-depth 26-Oct comparison checkout:** 195099ca583ace766d075682927f590360c823a5 plus a large moving checkout; source paths, file counts and status in comparative research refer ONLY to it.

## 1. Why this design packet exists

The user previously requested:
1. In-depth source file-tree and architectural comparison of llama.cpp and vLLM then zerogravity, with new-model integration and what to adapt.
2. A second zerogravity design-validation reading .agents/docs and .agents/plans and proposing a clean next-gen AI execution fabric including llama.cpp/vLLM lessons.
3. Explicit approval to break all pre-v1 internals and redesign any part of zerogravity if better.
4. Architectural endorsement subject to mandatory refinement: make serving an independently reusable library and expose *everything* through exported public zerogravity Rust library; CLI must only consume this library for CLI/presentation duties.
5. Create a completely self-contained side planning reference with all discussion details, preserving exact language where materially important, and a handoff for future LLMs. **Do not resync or implement against the changed checkout yet.** User says they will separately initiate the next phase when ready.

This maintained plan set preserves the architectural work behind #1–4. Its creation did not itself authorize implementation; current source/status and the latest user instruction determine what happens next.

## 2. Mandatory reading sequence for the next agent

1. `.agents/docs/CURRENT_HANDOFF.md` — current cross-window status, authority and next-path decision; then return here for Fabric-specific design context.
2. [DECISIONS_AND_OPEN_QUESTIONS.md](DECISIONS_AND_OPEN_QUESTIONS.md) — verbatim user instructions, accepted vs recommended, rejected designs, unresolved API choices.
3. [README.md](README.md) — complete proposed Fabric vNext architecture including model/task/compiler/state/physical/memory/serving/multimodal/distributed.
4. [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md) — **mandatory new standalone zerogravity-serving + all public zerogravity API + thin CLI**.
5. [PRIOR_ART_AND_REVALIDATION.md](PRIOR_ART_AND_REVALIDATION.md) — upstream llama.cpp/vLLM source trees, prior design research, local historic architecture mapping and full original document crosswalk.
6. [IMPLEMENTATION_AND_ACCEPTANCE.md](IMPLEMENTATION_AND_ACCEPTANCE.md) — staged future migration, no-edit resync, exact acceptance and performance/correctness gates, risks.
7. [TODO.md](TODO.md) — held work, never treat as part of active queue unless coordinator later promotes it.
8. Always read current source-of-truth instructions before acting: AGENTS.md; .agents/plans/README.md; .agents/plans/TODO.md; current status/traceability; relevant active TODOs and normative contracts.

The individual plan files in this directory are the maintained authoritative architecture source.

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

## 4. How to use in a current conversation

- Start at `.agents/docs/CURRENT_HANDOFF.md`; do not infer the user's requested path from this 27-Oct document alone.
- If the user asks about Fabric design, use this packet to explain decisions and tradeoffs without changing source unless implementation is explicitly requested.
- If the user explicitly authorizes Fabric implementation, revalidate current `main`, reconcile the target design against current source, then follow the maintained transition S0→S3, F0→F4, Q0→Q2 sequence.
- Current AGENTS.md topology applies: one persistent `main`, explicit file ownership, ephemeral external scratch only, no automatic push/tag/release.

## 5. Current launch method

Do not paste the retired 27-Oct resynchronization-only or implementation prompt.
Start every new window at `.agents/docs/CURRENT_HANDOFF.md`. That file contains
the reviewed current source status, active-Qwen critical path, exact read order
and the chat prompt pattern. Read this Fabric handoff only when the user chooses
the Fabric path.

When Fabric implementation is explicitly authorized, revalidate current `main`
first and follow the S0→S3, F0→F4, Q0→Q2 transition plan. When implementation
is not authorized, this directory is architecture context only.

Historical 27-Oct prompt text remains available in Git history; it is not a live
launch surface.

# Fabric vNext — complete next-agent handoff and starting prompt

**Date frozen:** 2026-10-27.
**Status:** DOCUMENTATION / REFERENCE ONLY. The user explicitly said to finish writing and report completion *before* any next phase. Do not initiate resync, source refactor, test, download, commit or deployment from this plan until their later instruction.
**Project:** zerogravity, /home/pvl/spaces/abird/src/zerogravity on connector **dotlink-l5** (pvl-l5).
**Initial 27-Oct archival-phase observed HEAD:** eb38dd5e892f2120ac29cc2fb79b2dbe137a37fe, checkout initially clean; this observation is provenance ONLY, not a readapted architecture review.
**Historical in-depth 26-Oct comparison checkout:** 195099ca583ace766d075682927f590360c823a5 plus a large moving checkout; source paths, file counts and status in comparative research refer ONLY to it.

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

## 4. How to use in the next conversation

**If user merely asks about this archive:** quote/find files, explain decisions. Do NOT resync source or start implementation.

**If user says "we can begin resynchronization":**
- perform read-only live source checkout review with dotlink-l5, compare with frozen design, update findings and present proposed adaptation;
- inspect recent repo work/commits, active other agents, Cargo dependencies, model math, source identity, state, workers, tests and serving crate status;
- decide whether design modifications are needed, preserving endorsed requirements;
- provide an "already done / still needed / newly conflicts / revised plan" analysis;
- do NOT begin code changes unless later instruction expressly asks to implement.

**If user expressly says "implement after resync":** reconcile current source and design, then implement in smallest reviewable logical units, preserving active agent ownership and accepted test evidence. Follow AGENTS.md and designated owner coordination; do not alter other checkouts, git history or push without explicit permission.

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

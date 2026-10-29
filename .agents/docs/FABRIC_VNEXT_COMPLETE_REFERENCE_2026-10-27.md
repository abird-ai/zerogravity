# zerogravity Fabric vNext — COMPLETE SELF-CONTAINED REFERENCE

Frozen: 27 October 2026. DESIGN/HANDOFF ONLY. No source change, no current checkout resync and no implementation authorization.

All detailed files in .agents/plans/fabric-vnext are reproduced verbatim below, in reading order. The separate originals are authoritative for future edits; this combined file is a complete immutable-style archival copy created in one documentation-only operation. Do NOT confuse the historical 26 October source audit with new HEAD. The user explicitly requires a standalone serving library, a complete exported zerogravity Rust library and CLI only as a facade client, and a separately initiated future resync.



---

# FILE: .agents/plans/fabric-vnext/HANDOFF.md

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


---

# FILE: .agents/plans/fabric-vnext/DECISIONS_AND_OPEN_QUESTIONS.md

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


---

# FILE: .agents/plans/fabric-vnext/README.md

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


---

# FILE: .agents/plans/fabric-vnext/PUBLIC_LIB_AND_SERVING.md

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


---

# FILE: .agents/plans/fabric-vnext/PRIOR_ART_AND_REVALIDATION.md

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


---

# FILE: .agents/plans/fabric-vnext/IMPLEMENTATION_AND_ACCEPTANCE.md

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


---

# FILE: .agents/plans/fabric-vnext/TODO.md

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

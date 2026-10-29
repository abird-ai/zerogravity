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

**Decision U-VNEXT-04 — HISTORICAL, handoff mechanism superseded:** the 27-Oct turn intentionally produced documentation only and did not authorize implementation. Its architectural decisions remain useful, but the old side-archive/concatenated-handoff mechanism is retired. Current handoff authority is `.agents/docs/CURRENT_HANDOFF.md` plus the four maintained planning/status files and this live plan set; Git history retains the deleted archive snapshots.

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

**Work status:** the 27-Oct documentation-only phase is historical. Current main has since advanced substantially; use the canonical current handoff/status files and revalidate source-dependent assumptions before implementing this architecture.

Full next-agent starting prompt and document read order: [HANDOFF.md](HANDOFF.md).

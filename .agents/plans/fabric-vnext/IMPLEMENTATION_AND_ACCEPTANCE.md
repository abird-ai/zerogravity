# Fabric vNext — migration blueprint, acceptance gates and source-change crosswalk

**Status:** planned future work, execution HELD. User ordered 27 October 2026: archive architecture first; ONLY in a later explicitly initiated phase review/refit to the changed repository; only after that may the user authorize implementation. This plan may propose any pre-v1 breaking edit, but does not authorize edits or a rebase now.

Use [README.md](README.md) as the authoritative proposed architecture, [PUBLIC_LIB_AND_SERVING.md](PUBLIC_LIB_AND_SERVING.md) for the new mandatory standalone serving library and public SDK and [PRIOR_ART_AND_REVALIDATION.md](PRIOR_ART_AND_REVALIDATION.md) for source investigations. The working zerogravity repository is changing quickly. File anchors below are historical examples, not current instructions.

## 0. Freeze provenance without disturbing the live owner

Historical 26 October reference inspected with dotlink-l5:
- /home/pvl/spaces/abird/src/zerogravity; Git HEAD 195099ca583ace766d075682927f590360c823a5.
- Live workspace had hundreds of modified/deleted/untracked artifacts while a family/source/runtime refactor was underway. Source tree analysis and file-line counts are from that particular state and may no longer apply.
- At the beginning of this 27 October archival phase, HEAD was eb38dd5e892f2120ac29cc2fb79b2dbe137a37fe and the checkout initially clean, as verified by read-only Git commands. That observation is NOT a full source re-audit, comparison, build or execution capability claim.
- Current active owner plans remain .agents/plans/TODO.md and per-plan TODOs. This new fabric-vnext folder is explicitly NOT integrated into their queue yet, to avoid colliding with ongoing work.
- Never check off old tests, reuse numerical permits, or equate a moved source file with previously qualified exact source/compiler/kernel identity.
- No destructive Git operations, pushes, branch rewrites, model downloads, device launches or service starts are implied by this reference.

## 1. Next user-authorized phase: resynchronization BEFORE redesign implementation

This is the first and ONLY authorized type of work when the user returns with "resync", "revalidate" or equivalent; even then, obey the exact later instruction.

1. Open AGENTS.md, .agents/plans/README.md and .agents/plans/TODO.md; detect in-flight ownership and relevant checkouts.
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

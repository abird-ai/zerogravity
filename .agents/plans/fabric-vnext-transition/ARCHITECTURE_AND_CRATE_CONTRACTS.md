# Fabric vNext transition — complete target crate/module architecture and contracts

**31-Oct direction:** Fabric S0 landed in signed`7396984`; source/host checks pass.
Plan 2 includes Plan 1; Qwen expansion deferred. Historical docs-only holds below
are superseded. [Current source/crosswalk](CURRENT_SOURCE_RECONCILIATION.md) ·
[active phase queue](TODO.md). No new physical/production grant.


**Status:** target architecture for first structural source milestone. Historical design baseline fadf2b1; current source baseline is the reviewed main state referenced by CURRENT_HANDOFF.md / CURRENT_SOURCE_RECONCILIATION.md. All APIs below are illustrative; full Rust names remain pre-v1 breakable. This design incorporates the recent codec and GPU refactors, proper resource ownership and library-first serving.

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

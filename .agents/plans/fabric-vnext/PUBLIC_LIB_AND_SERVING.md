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

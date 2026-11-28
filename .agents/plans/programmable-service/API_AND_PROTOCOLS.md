Author: Project owner
Co-authors: GPT-6 Astra Pro; GPT-6.1 Sol (native SDK, composition and protocol review)
Owner: @pvl
Status: Implementation contract; proposed APIs, not implemented support
Decision: 28 November 2026 (Asia/Singapore)

# Programmable stack — API and protocol contracts

## 1. Scope and implementation order

This is the authoritative target API/protocol reference for the [foundation phases](IMPLEMENTATION_PHASES.md). The [service design](README.md) explains ownership; [composition](COMPOSITION_AND_ADAPTIVE_EXECUTION.md) explains mathematical/task semantics. Keep contracts here and completion in [TODO.md](TODO.md). Once implemented, exported Rust types, adapter schemas and their documented behavior become the authority; update these examples with the implementation.

Names below are target shapes. They describe useful operations and invariants, not a requirement for a crate, trait, registry or test suite per row. Convenience builders and adapters use the same checked operations.

| Delivery | Public contract available at that exit |
| --- | --- |
| F1 | Embedded Service/Client; source discovery/configuration, explicit pull, initial atomic load, run/stream/serve, inspect, complete results and joined shutdown. |
| F2 | Owned adapter-facing commands/results, grants/capabilities/errors, supported task/model/mapping drafts and revisions, durable commands/results and bounded observations. Exercise existing models, a serial task and a supported closed region. |
| F3 | Sequential registration, generation-checked activation, full admitted composition retention and resource-safe coexistence/retirement. Exact references are required wherever F1/F2 already use their corresponding objects. |
| M1 / M2 | Optional SDK-hosted MCP / native attachment, respectively. Either can ship first over the foundation contracts. |
| C / K / E / S / X | Broader task/model authoring; execution transforms; profiles/studies/policies; continuation; isolated extensions. Each adds supported payloads/implementations to existing operations. |

The foundation exit does not require full MCP, transport attachment, arbitrary graphs/kernels, universal checkpoint restore or backend performance parity.

## 2. Rust design and library reuse

Prefer efficient native Rust with simple ownership. A clean high-performance design wins when it stays intuitive; otherwise keep the design simple and optimize measured hot areas.

- Reuse and reexport existing semantic, source, state, provider, resource, receipt and artifact types. Extract genuinely shared behavior into its lowest appropriate existing library. Do not copy an implementation into Service, CLI, MCP and a backend.
- Use concrete types, ordinary enums, slices, moved/borrowed values, associated types and static dispatch where useful. Share immutable owners where actual lifetimes require it. Do not add a trait per helper, pervasive generics, unsafe shortcuts or compulsory allocations to pursue theoretical zero cost.
- Keep existing dynamic boundaries where heterogeneous providers, tasks or direct/attached clients need them. Choose the client transport at a management/request boundary; do not propagate transport dispatch into token/tensor arithmetic.
- Embedded calls do not serialize or use localhost HTTP. Borrowed advanced native APIs remain valid. Owned DTOs are for queued/retained data and process boundaries; they do not require every native type to implement Serialize/Deserialize.
- Resolve selectors, schemas, permissions, source identities, mapping, compilation and verbose diagnostics before execution. Prepared code uses original bound owners and compact native choices. No new graph interpreter, token scheduler, allocator or plugin registry.
- Keep defaults inexpensive. Reuse current counters; detailed traces/captures are explicit and budgeted. Move original terminal output into results where possible instead of collecting another full copy from deltas.
- Share implementation before sharing abstractions. Add a crate, generic extension interface, specialized kernel or caching layer when a real consumer or measured bottleneck justifies it.

### Existing homes

| Responsibility | Reuse / extend |
| --- | --- |
| Public application lifecycle and convenience API | Small modules in the existing zerogravity facade; existing Runtime and ControlPlane underneath |
| Discovery/configuration/acquisition | `discovery`, `configuration::ConfigFile`, original HubPull/PreparedHubPull and artifact publication |
| Model mathematics and source bindings | Model-owned adapters; core ModelProgram/ForwardPlan and RegionExecutionProvider |
| Physical preparation | Engine prepare/ExecutableBundle, original source issuer, selected provider, state/resource/retirement owners |
| Task execution | Existing task/pipeline driver and scheduler; extend supported coordination there |
| Waiting and results | Original charged event receiver/receipt and worker completion; blocking and async views of the same submission |
| Native HTTP transport | Existing hyper/tokio/serde/bytes ecosystem; optional facade transport handlers |
| MCP | Optional official rmcp dependency and a small typed projection over Client |

The serving crate already depends on engine and the facade depends on serving. Native Service handlers must not introduce a serving → facade cycle. Share neutral transport helpers only when both consumers need them; keep OpenAI protocol behavior in its existing owner. The CLI-private browser parser remains a restricted presentation adapter.

## 3. Public SDK

### 3.1 Ownership and ordinary use

```rust
use zerogravity::Service;

let service = Service::embedded().await?;
let zg = service.client();

let acquired = zg.pull("catalog:qwen3.5-0.8b").await?;
let model = zg.load(acquired).await?;
let answer = model.run("Explain speculative decoding.").await?;
println!("{}", answer.text());

service.shutdown().await?;
```

Examples omit application error-path cleanup; production callers join owner shutdown on success and error. Service is the non-cloneable owner of one Runtime. Client is a cheap cloneable view. Embedding opens no listener/UI or process-global hooks implicitly.

Builders are cold until awaited or started and use ordinary Future/IntoFuture behavior without boxed futures per token. Awaiting pull/load/run returns the completed typed result. Where useful, `.start().await` returns an accepted operation/run handle. Starting or awaiting the same consumed builder cannot submit twice. Blocking methods use the same operations and a long-lived driver, not a runtime per call.

| Surface | Input | Result and semantics |
| --- | --- | --- |
| `Service::embedded()` / builder | Existing effective config, limits and optional sinks | Application owner; `client()` creates a direct view; `shutdown().await` joins owned work and reports cleanup failures. |
| `Client::connect(endpoint)` / `connect_local()` | Explicit selected endpoint and credentials | Attached view after identity/version checks; M2. Auto embed belongs to process bootstrap, never connect(). |
| `zg.sources().list/browse/repository(...)` | Existing named discovery source, query/page/cursor, online policy or locator | Existing source/Page/Repository observations with source context and exact continuation limits. Browse does not pull or load. |
| `zg.sources().replace(...)`, `zg.settings().update(...)` | Typed change plus observed configuration revision | Original optimistic atomic publication or stale conflict; enforce local-root grants. Removing a source does not delete acquired data or loaded owners. |
| `zg.pull(selector)` | Existing selector, revision/file/quant/filter selection and transfer limits | `AcquiredSource`: exact resolved artifact/source identity plus original acquisition/publication receipt. This is distinct from named `discovery::Source`. |
| `zg.load(input)` | AcquiredSource, local path or existing cache/catalog selector; execution/load options | `ModelHandle` with exact registration, source and requested/effective preparation. Missing bytes require explicit acquisition policy. |
| `model.run(input)` | Exact admitted target, supported typed input and bounded run controls | Completed `RunResult`; `.stream()` or `.start()` select alternate consumption modes below. |
| `zg.capabilities()`, `model.inspect/program()` | Optional exact subject/filter | Actual supported operations/constraints and semantic/prepared observations; unavailable detail is explicit. |
| `zg.unload(model.reference())` | Exact registration | Logical unpublication plus actual retirement status; dropping a ModelHandle does not unload it. |
| `zg.serve(&model)` | Exact exposed registration/route set, bind/auth/ingress options | Service-owned `ListenerHandle`, actual address and exposed scope. Explicit stop checks its generation. |
| `zg.operations().get/list/cancel(...)` | Operation identity, bounded page/query or cancellation intent | Retained status/result or explicit expiry. A dropped observer does not cancel service-owned work. |

Keep exact registration allowlisting in the original serving owner: `serve(&model)` cannot silently expose other loaded registrations. It shares weights/scheduler/registration with native runs. Preserve the advanced scoped ServerHandle's existing Drop contract; the new service-owned listener view stops only through an explicit stop or owner shutdown.

F1 implements the existing initial atomic model/route-set preparation. Sequential independent loads become supported in F3; before then a second unsupported load refuses explicitly instead of replacing the first model. Advanced callers retain the existing Runtime/provider APIs.

### 3.2 Run, stream and cancellation

```rust
let mut stream = model.run("Explain Rust ownership.")
    .max_tokens(256)
    .stream()
    .await?;

while let Some(event) = stream.try_next().await? {
    if let Some(text) = event.text_delta() {
        print!("{text}");
    }
}
let result = stream.result().await?;
println!("{:?}", result.usage());
```

RunStream implements the runtime-neutral futures_core::Stream trait with typed Result events; try_next is a thin convenience over that same receiver. This keeps ordinary stream combinators available without imposing an async runtime on the engine. There is one owning consumer of the original charged receipt. Await-to-completion, stream and start are alternative submission/consumption modes.

- `run(...).await` drains the receipt and returns its original terminal result.
- `.stream()` yields committed events with original backpressure. `stream.result()` drains any remaining events and returns/caches the terminal result; it never submits again. Do not synthesize success from EOF.
- `.start()` gives a RunHandle with cancel/result access and a collector on the shared driver. The collector consumes the original receiver even before result() is called; waiting only on a terminal cell while an unread token queue is full would deadlock. RunHandle retains the foreground cancellation guard independently: dropping its last public request owner requests cancellation while the collector drains terminal completion. The collector does not make the request service-owned.
- Extend original receipt completion/wakeup for all terminal paths, including cancellation and worker loss. Completion notification must remain available independently of token-queue fullness or a dropped presentation. It is not another token transport or physical authority.
- A cancellation reply means requested. Actual execution outcome comes from the original worker/receipt. Its supervisor may report interrupted or completion-unknown after worker loss; neither that report nor cancellation establishes physical retirement. Submitted resources remain owned until original retirement/quarantine. Current OutputStream::cancel closes its view immediately, so a wrapper alone cannot satisfy this contract.
- Foreground runs/streams are request-scoped; dropping their final owning request handle requests cancellation. Client-view drop alone does not invalidate a retained run handle. Explicit service-owned/background inference uses a supervised collector and advertised retention, not an accidental detached stream.
- Observers are separate from the sole result consumer. Slow telemetry may lose samples; accepted inference output cannot silently lose tokens. If a result exceeds its configured in-memory budget, use an explicitly supported artifact/spill path or report the limit without mislabelling truncation.

RunResult carries original output, finish reason, usage, exact run/registration/revision identities and declared usage scope. Preserve length-limited finishes. Pipeline totals require collecting per-stage outcomes: today's serial driver forwards only the last stage's Done. Until extended, label that usage final-stage rather than whole-task. Committed output is distinct from network delivery, acknowledgement or playback.

## 4. Programmable authoring and live control

### 4.1 One lifecycle, three typed domains

Task definitions connect model/task ports; model definitions express source-bound mathematics/state; mapping definitions choose implementations for an exact model revision. Reuse a small lifecycle without erasing these domain-specific contracts into universal JSON nodes.

| Native operation | Input → result | Contract |
| --- | --- | --- |
| `drafts.create` | TaskDefinition / ModelDefinition / MappingDefinition → DraftRef | Incomplete definitions can be retained with declared durability/expiry. |
| `drafts.derive` | Exact RevisionRef → DraftRef | Copies authoring recipe/lineage; no new execution owner. |
| `drafts.edit` | DraftRef + expected version + typed edit batch → updated DraftRef | Atomic bounded batch; conflict or invalid/unauthorized edit publishes nothing. |
| `drafts.validate` | Exact draft version → ValidationReport | Structural/source/type/shape/numerical/effect/capability findings by node/port; no execution or qualification. |
| `drafts.freeze` | Exact draft version → RevisionRef | Recheck and resolve immutable recipe; no live admission. Do not silently freeze a newer draft. |
| `zg.prepare` | RevisionRef + PrepareOptions → PreparedHandle; `.start()` returns Operation<PreparedRef> | Authenticate sources and bind original owners, reusing compatible retained owners where supported; verify target and admit resources. Failure retains recipe/outcome. |
| `zg.execution().inspect` | PreparedRef or RunRef → ExecutionDescription | Distinguish requested, prepared and observed executed mapping; preserve unavailable provenance. |
| `zg.activate` | Binding + PreparedRef + expected generation + boundary → ActivationReceipt | Publish one new binding generation; default applies to future requests. |
| `zg.checkpoints()/experiments()` | Supported checkpoint/study identity and typed request | Reopen, resume, fork or rerun under the declared supported scope. Later S/E capabilities fill these operations. |

Domain accessors may provide ergonomic builders such as `pipelines().chain`, `models().draft`, `revisions().derive` and `implementations().for_region`. They lower to these same definition/edit types. Local Rust building is synchronous and has no hidden I/O. A convenience `.prepare().await` submits the built definition, checks/freezes its exact version and prepares it; shared agent drafts use explicit create/edit/freeze calls. A failed preparation does not discard the frozen recipe. Preparation builders follow pull/load: awaiting returns the completed prepared handle; start returns the accepted operation. A local derive builder records an exact base reference and edits without fetching it; awaited submission resolves that exact base. The shared drafts.derive command is asynchronous.

Task preparation returns a complete prepared task; model preparation returns its prepared model owner. A mapping candidate prepares the connected model/task with that mapping. Explicit region preparation declares its partial diagnostic/inspection scope and cannot activate a complete route.

Reuse the same reducers, validation and lowering in library and adapter consumers. Authoring closures run while building a checked definition; they are not serialized runtime callbacks. Model scopes expand into the existing ForwardPlan. Mapping recipes prepare original executables; they do not become a second canonical IR.

### 4.2 References and capability boundaries

| Reference | Meaning |
| --- | --- |
| ArtifactRef / AcquiredSource | Exact bytes/resolved source and provenance, not a live reader or execution permit. |
| RevisionRef with kind | Immutable task, model or mapping recipe. Mapping records its exact semantic model revision. |
| DraftRef + version | Editable object and optimistic concurrency precondition; returned retention/durability are explicit. |
| RegionRef / NodeRef / PortRef | Exact parent revision or draft/version plus scoped native ID. Bare indices have no cross-recording meaning. |
| PreparedRef / ModelRef / ListenerRef / RunRef | Service identity, live epoch and exact preparation/registration/listener generation/original request identity. Service resolves these to original retained owners. |
| OperationId / AttemptId | Logical accepted work / one execution attempt. Retries do not create attempts; explicit rerun does. |

These can be small newtypes over existing identities. The service owns live retention; a serialized PreparedRef merely names/preconditions that ownership. Neither copying a reference nor pinning a disk artifact allocates a device permit. Restart reauthenticates and prepares recipes with new live identities.

Capabilities describe supported combinations: task/definition kind, primitive/schema, shapes/numerics/effects, provider/target, transforms, observations and activation/restore boundaries. Distinguish recognized semantics, implemented preparation/execution, and current availability. Capabilities never guarantee future capacity.

F2 uses existing model definitions, serial tasks and actual supported ClosedRegion choices. Broader DAGs, new mathematics, kernel chains and multi-operation fusion arrive in C/K. Kernel artifacts name recognized interfaces and exact source/compiler/target identities. Backend selection alone does not identify a fusion compiler: require a supported provider/template or report the actual unambiguous choice. Unknown required semantics refuse before execution.

### 4.3 Composition and mapping examples

Simple supported text chaining stays short:

```rust
let pipeline = zg.pipelines()
    .chain("draft-review", [&writer, &reviewer])
    .prepare()
    .await?;
let answer = pipeline.run("Explain this proposal.").await?;
```

The builder records immutable model/task references; preparation binds exact registrations. Admission retains every stage owner. Terminal-value edges carry committed output with declared bounds; incremental edges require supported stream contracts. No silent truncation, session sharing, mandatory host copy or new scheduler follows from an edge.

A mapping change uses a region scoped to the actual semantic revision:

```rust
let program = model.program().await?;
let region = program.region(region_id)?; // inspected ID, not a guessed name

let candidate = zg.revisions()
    .derive(model.mapping_revision())
    .map_region(&region, &implementation_recipe)
    .prepare()
    .await?;

let description = zg.execution().inspect(candidate.reference()).await?;
```

A model exposes its semantic `revision()` separately from `mapping_revision()`. program() returns an inspected description; it grants no source-binding rights. A chain/fusion builder produces an immutable implementation recipe for map_region; it preserves every escaping output, state effect and numerical observation. See [deep authoring](COMPOSITION_AND_ADAPTIVE_EXECUTION.md#6-deep-execution-pipelines-chain-fuse-capture-and-place) for typed bindings.

### 4.4 Profiles, studies and activation

Keep an explicit finite study before introducing search strategies:

- ProfileRequest: subject, window, requested fields, observation budget and separate raw-input/tensor capture choices.
- ProfileResult: actual coverage, units/clocks, loss and per-field provenance; aggregates cannot claim exact event history.
- StudyRequest: baseline, finite candidates, authorized/synthetic workload, numerical/quality/performance acceptance, resource/time/concurrency budgets and matched observation.
- StudyResult: all offered/admitted/refused/failed/cancelled/incomplete outcomes, comparisons, tested scope and decision: keep baseline, qualified candidate, or no qualifying result.

These become E capabilities over F2 records/observations and F3 activation. A passing numerical comparison is not a speedup; a profile is not replay input. Portfolio policies are E2 preparation, not an automatic product of every E1 comparison.

```rust
let report = zg.experiments().compare(&baseline, &candidate)
    .workload(workload_artifact)
    .budget(study_budget)
    .acceptance(service_objectives)
    .check_inherited_contract()
    .record("norm-study")
    .run()
    .await?;

if report.accepts(candidate.reference())? {
    zg.activate("assistant", &candidate)
        .if_generation(active.generation())
        .for_new_requests()
        .await?;
}
```

accepts checks the recorded numerical/state contract and selection objectives against the requested candidate, including any required meaningful gain; a caller boolean cannot authorize activation. Service independently checks grants, scope, exact owners and capacity. Desired durable binding and actual publication remain distinct; the receipt reports actual generation. Admitted requests/sessions keep their pinned composition. Rollback selects a retained/reprepared old recipe for new work; it does not rewind committed state. Failed coexistence admission leaves the active binding usable.

## 5. Operations, errors and observation

### 5.1 Command continuity

Use a small typed Command enum for application mutations and typed Query inputs for observations; straightforward native methods/builders wrap their handlers. Do not build a dynamic command bus or serialize direct calls through JSON. Run submission uses its original fast request path rather than the serialized management slot.

A command carries its typed arguments, CommandKey and any expected draft/config/binding generation. Service derives the principal/grants from the trusted client context; fields in a request cannot grant rights.

1. Atomically claim/check the key under stable ServiceId and principal scope across concurrent callers, before re-resolving floating references/defaults or rechecking an already-applied generation. Recheck current authorization before returning an existing result/private details; a prior key does not bypass revoked or reduced grants.
2. Same key and canonical submitted intent returns the original result/operation. Different intent conflicts. Retain original submitted intent and its resolved recipe; a new upstream HEAD must not alter a retry.
3. First durable acceptance commits submitted intent, resolved arguments/recipe and operation before applying mutation/submission effects and before acknowledging durability. Ephemeral commands say so. A failed/unknown durable publication cannot return durable success.
4. Transport request IDs correlate messages; CommandKey deduplicates logical mutation; OperationId identifies accepted work; AttemptId identifies an actual execution. Retry after transport loss uses the same key. Explicit rerun uses a new command/attempt with lineage.
5. Advertise bounded retention and the deduplication horizon. Within it, retain enough result/tombstone information to refuse expired results without re-execution. Beyond it, the client must reconcile instead of blindly retrying an old mutation. Do not promise unlimited exactly-once execution.

Operation<T> is a typed view over common operation state. Expose queued/running/terminal outcome, cancellation requested and actual retirement where relevant without duplicating engine authority. Terminal results are retained independently of notifications. Restart distinguishes interrupted from completion-unknown and preserves affected physical fences. Ordinary ephemeral inference does not write a durable command log by default.

### 5.2 Errors and grants

Preserve native causal errors. Project an owned ErrorInfo for adapters: stable code, safe message, correlation/operation, typed context such as expected/current generation or requested/available resource, and reconciliation/retry guidance. Keep detailed causes behind the native grant context.

Reuse existing error codes where their meaning fits. Invalid input, unsupported capability, stale/conflicting identity, denied access, unavailable/expired data, capacity refusal, cancellation and interrupted/unknown completion must be distinguishable without parsing prose. A remote failure never silently selects another backend or starts another Service.

Trusted embedded ownership can create restricted Client views. Adapters authenticate and map to those views; requested budgets can only reduce granted limits. Observe, infer/manage, raw capture, experiment/code execution and activation are distinct permissions where needed, not separate identity-management frameworks.

### 5.3 Observations and results

`zg.observe(filter)` yields a scoped snapshot/cursor and bounded updates. Pair the control snapshot with a watermark from the same state stream; register observation before the handoff and resnapshot on overflow so changes cannot disappear between snapshot and subscription. Asynchronous device samples retain their own timestamps and provenance, not a fictitious global snapshot.

Use existing typed events/counters. An envelope adds only relevant service epoch, stream/sequence, causal IDs, time domain/units and payload. Native IDs remain compact; wire IDs/cursors are opaque strings. Ordering is per stream. Native replay is available only for advertised retained history; otherwise return a gap/resync indication.

Keep three contracts distinct: loss-tolerant observations, reliable inference/task data, and retained terminal results/artifacts. Coalesce/drop samples with loss counts. Inference keeps original backpressure or explicit failure. Trace/capture requested as complete reserves the required budget or fails visibly. Large outputs use authenticated artifact reads; prompts/tensors/secrets are never implied by basic metrics.

## 6. Native attachment protocol (M2)

Use HTTP/1.1 JSON plus SSE over a private Unix-domain socket initially, with the existing HTTP/async/serde libraries. The same application protocol can support explicit remote HTTPS later. This is ZeroGravity's management protocol, separate from MCP and OpenAI inference. No custom frame format or generic remote-code executor is needed.

| Route family, under `/zg/v1` | Contract |
| --- | --- |
| `GET /info` | Stable service identity, live epoch, API compatibility, actual capabilities, limits and retention. |
| `POST /queries` | Bounded typed read/inspection/browse request → typed result/page. No arbitrary expression language. |
| `POST /commands` | Closed typed mutation → completed result (200) or accepted operation (202), with command identity and durability. |
| `GET /operations/{id}` | Authorized status plus bounded terminal result/reference; pending, expired and unavailable remain distinct. |
| `POST /operations/{id}/cancel` | Idempotent cancel request through the original operation; terminal outcome queried separately. |
| `POST /runs` | Request-scoped native inference; JSON terminal result or SSE events plus terminal outcome. Uses original admission/receipt. |
| `POST /runs/{id}/cancel` | Exact live request cancellation; not a claim of physical retirement. |
| `GET /events?after=...` | Native bounded observation stream; cursor/retention/gap rules above. Snapshot handoff uses the typed observation query. |
| `POST /artifacts`, `GET /artifacts/{id}/bytes` | Explicit bounded streamed import/read, with content identity, length/integrity checks and supported ranges. No large base64 control payloads. |

A run SSE response begins with acceptance containing its exact RunRef before output deltas. Attached start returns that handle and collects the same stream internally. If transport fails before acceptance arrives, its closure cancels that request; do not blindly resubmit inference.

Routes expose the same typed handlers as direct Client. Do not hand-maintain an API per frontend. Native streaming disconnect requests cancellation only for its request-scoped run; disconnecting a command wait/observer does not cancel accepted service-owned work. Explicit background inference is a command with a supervised collector. A client run handle identifies the original request even when its output is collected asynchronously.

Command example, after the native identity/compatibility check:

```json
{
  "command_key": "activate-norm-study-1",
  "context": {"service_id": "service_1", "epoch": "epoch_8"},
  "kind": "activate",
  "arguments": {
    "binding": "assistant",
    "prepared": {
      "service_id": "service_1",
      "epoch": "epoch_8",
      "id": "prepared_42"
    },
    "expected_generation": "12",
    "boundary": "new_requests"
  }
}
```

Require the observed service identity/epoch on live mutations, carried in the typed request context; stale references refuse before effects. Durable command reconciliation checks stable service identity and command key first, so an accepted pre-restart command can still be queried. Use one documented ErrorInfo projection with suitable HTTP statuses; 202 means accepted, never completed.

Bound message/body/page depth and size, outstanding requests, wait deadlines, stream buffers and artifact storage with service limits. A command/operation observation deadline ends only that wait and does not cancel accepted service-owned work. Request-scoped run deadlines and disconnects follow their declared cancellation scope. Never automatically retry a mutation under a fresh key.

### Local ownership and authentication

Use the existing private XDG runtime namespace, owner-only socket/directory permissions and OS peer credentials. Verify the expected owner; authenticated service descriptions are not proof by display name. Restricted adapters receive native grant contexts, not authority from submitted principal strings. Attached filesystem paths name the service host; importing local client data is explicit.

Hold one instance lock throughout discoverable host lifetime. Bootstrap probes/creates under this ownership protocol. Only established absence allows embedding; starting, busy, unresponsive, unauthorized or incompatible owners return their actual state. Clean a stale socket only while holding the lock and verifying its ownership. Embedded Service construction itself opens no socket: only a process host explicitly publishes discovery/attachment.

Remote HTTPS and credentials are explicit later configuration; default native attachment remains local. Browser-facing endpoints retain origin/CSRF and safe-projection rules. Distinct instance names do not coordinate GPU memory: shared-device processes need original parent-authoritative envelopes or disjoint devices.

### Compatibility

Native wire major version, persisted document version and MCP protocol revision are independent. Rust crate SemVer is not a wire handshake or a stable binary ABI. Start with one supported native major; add compatibility adapters only for concrete consumers.

Reuse public owned request/result types for serde/schema generation where suitable. Keep native borrowed handles and privileged error causes out of wire schemas. Accept additive optional response metadata; reject unknown command fields/variants and required semantics rather than ignoring a misspelled mutation. Required capability/version declarations make semantic changes explicit. Wire identities/cursors are opaque strings; full-width integer counters/generations use decimal strings. Ordinary bounded numeric options keep JSON numbers. Document the owned DTO encoding once so JavaScript cannot round native identities or counters.

## 7. MCP projection (M1)

Use optional official [rmcp](https://github.com/modelcontextprotocol/rust-sdk) with the current compatible supported release and minimal needed features, resolved normally in Cargo.lock. It supplies protocol framing/version handling and standard transports. Keep schemas close to the owned request/result types; handlers invoke Client. No CLI shelling, separate job store or per-kernel tool catalogue.

```rust
let service = zerogravity::Service::embedded().await?;
zerogravity::mcp::Server::new(service.client())
    .serve_stdio()
    .await?;
service.shutdown().await?;
```

The same Server accepts an attached Client in M2. Closing the adapter releases its request scopes/subscriptions, not independently service-owned models or jobs.

| MCP projection | Native surface |
| --- | --- |
| `capabilities`, `inspect`, `sources_list/browse/repository` | Capability/inspection/discovery queries |
| `sources_replace`, `models_pull/load/unload`, `serve_start/stop` | Existing typed configuration/model/listener commands |
| `run` | Explicitly bounded native inference; foreground or supported service-owned mode is declared |
| `composition_create/edit/validate/freeze/prepare` | Same Task/Model/Mapping definitions, edits and revision/preparation lifecycle |
| `execution_inspect`, `transform_propose`, `revision_activate` | Actual mappings, supported implementation candidates and guarded activation |
| `workloads_profile`, `experiments_run/compare` | Bounded profile/study operations when E is available |
| `operations_get/list/cancel` | Same accepted operation/result/cancellation |
| `checkpoint/resume/fork/rerun`, artifact access | Supported continuation and bounded import/read |
| Authorized `zg://` resources | Operation/revision/artifact views bound to an identified service |

Names are a small stable domain vocabulary. Tool input/output schemas reflect actual supported native operations; a known but unimplemented capability is reported explicitly. List order is deterministic and authorization-scoped. IDs/URIs do not confer access. Tool results use small structured outcomes and resource links; raw traces/tensors use bounded artifact transfer.

### Verified protocol baseline

Official [current core](https://modelcontextprotocol.io/specification/2026-07-28) on 28 November 2026 is 2026-07-28: self-contained requests with per-request version/capabilities and server discovery. Use rmcp's supported-version handling; advertise legacy compatibility only when enabled and checked. Do not couple native schema evolution to MCP's dated revisions.

- [stdio](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio) carries newline-delimited JSON-RPC; stdout is protocol-only. [Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http) uses POST with JSON or request-scoped SSE. This revision has no MCP protocol session, standalone GET event stream or Last-Event-ID replay. Native /events is a separate protocol.
- Use supported core subscription notifications as invalidation hints. Detailed raw observation/replay remains the native contract. [Tasks and Events](https://modelcontextprotocol.io/extensions/overview) are optional Beta extensions, explicitly enabled and capability-checked; neither gates ordinary operations.
- Long work always supports a normal tool result carrying OperationId plus get/cancel tools. Optional [Tasks](https://tasks.extensions.modelcontextprotocol.io/specification/2026-07-28/tasks) maps that same durably accepted work, without another scheduler/store. Never return a Tasks result to a request lacking support.
- Current HTTP stream closure cancels its MCP request; stdio uses cancellation notifications. Stop request-local output, preserve already accepted service-owned work, and reconcile a lost acceptance receipt by CommandKey. Explicit operation cancellation remains separate.
- [Tool failures](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) use structured domain results with isError; malformed/unknown protocol requests use protocol errors. Native codes and authorized context remain available.
- Service grants are authoritative. MCP HTTP validates Origin and uses its supported [authorization contract](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization/index); a native local credential is not automatically interoperable MCP HTTP authorization. stdio follows host process credential policy.

Implement optional protocol features only for an actual supported workflow. Sampling, elicitation, Apps, Events and Tasks are not generic foundation requirements.

## 8. Validation without process overhead

The tables are API contracts, not a checklist demanding a separate test/fixture/receipt for every method, type, protocol field or helper.

Extend the existing phase scenarios: F1 embedded acquire/load/run/stream/serve/shutdown; F2 supported authoring plus accepted-command continuity; F3 live registration/activation with pinned owners and failure preservation. Fold cancellation, permission, stale identity, slow observers and error propagation into those behaviors where they exercise a real failure risk.

M1 proves the same supported operations over MCP with a direct Client; M2 proves native attachment/bootstrap. Their combined scenario establishes adapter equivalence when both exist. Use library protocol tests where applicable, and add focused interoperability checks for our projections, supported version/features and lifetimes. No Cartesian matrix of every method × transport × backend × failure.

Reuse existing fixtures, approved models and accepted unchanged-path evidence. Run targeted diagnostics to resolve concrete failures and the assembled repository gate at the coherent implementation boundary required by AGENTS.md. Do not requalify a backend for documentation, a pure projection or an unchanged path. Actual numerical/state/source/physical changes keep their affected real checks.

Benchmarks answer a concrete performance question on the actual hot path/workload. Prefer existing harnesses and full-workload measurements; retain correctness and complete outcomes. No performance threshold per accessor, speculative benchmark farm, per-helper acceptance cycle, mandatory evidence archive, historical source freeze or extra version pins.

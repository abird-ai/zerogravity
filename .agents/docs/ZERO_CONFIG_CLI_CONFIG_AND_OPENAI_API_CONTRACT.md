# zerogravity — One-command CLI, progressive configuration and OpenAI-compatible APIs (v4.3)

**AUTHORITATIVE UX/API DESIGN · 2026-10-12 · NOT IMPLEMENTED.** Accompanies [multi-model routing](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [model/artifact pipeline](PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md), [stateful execution](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [trusted distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md). External CLI and API products are named only in [attributed prior-art research](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md).

**UX philosophy:** *one useful command with almost no decisions*, and *every meaningful decision visible and configurable when needed*. A small terminal tool is the primary experience; device/quant/kernel/model/router/cluster controls appear through inspection and advanced settings. The commands and config examples below are **proposed**, not a finalized public binary/port/flag contract.

## 1. Default one-command user journey

### Interactive local single model

```sh
zerogravity run ./model.gguf
zerogravity run coding
zerogravity run coding --prompt "Summarize this"
zerogravity run coding --context 8192
```

Expected behavior: validate model or local alias → verify exact trained model/quant/tokenizer/source → safe physical resource admission and hardware probe → select best **qualified** numeric/kernel/placement candidate → load/warm model if needed → interactive terminal or one-shot output → stream **committed** output. No config file, mandatory server, GPU architecture, model conversion, external account, cloud endpoint, Python environment or internet access is required for a supported local file.

A local `run` invocation **runs in-process by default**. Optionally reuse an already-running **same-user trusted** service socket when explicitly requested or configured, using versioned process identity and model binding. Do not blindly send prompts to any local HTTP listener or silently start a permanent daemon/systemd/launchd service or install root/firmware settings. If no local model is found, explain available names and offer an explicit import/pull with source/size/license and a consent step; never silently fetch a guessed checkpoint.

If physical memory cannot fit the exact requested model/quant/context, explain the shortfall and available *explicit* alternatives (smaller context, selected alternate quant artifact, CPU streaming or approved peer). Do not quietly downgrade model weights, drop trained experts or route to another model.

### Multi-model router/service

```sh
zerogravity serve
zerogravity serve --model coding
zerogravity models list
zerogravity models ps
zerogravity models show coding
zerogravity models stop coding
zerogravity inspect coding --explain-plan
zerogravity config show --effective --model coding
```

Default: **one engine, one loopback API endpoint, many model aliases** and on-demand authorized local loading, sharing the same ModelCatalog/Router, ResourceArbiter, TaskCoordinator, RegionExecutionProvider and TaskDrivers as standalone CLI. No extra gateway proxy or independent model-specific serving process. `serve --model` can optionally preselect/prewarm an alias while keeping the same router if configured; names/port/flag choices are examples.

### Model import and profile

```sh
zerogravity models import ./model.gguf --name coding
zerogravity models pull <explicit-source-reference>
zerogravity models alias code coding
zerogravity models profile coding --set context=32768
```

Illustrative only. Pull from trusted source on explicit user request/interactive consent, with hash, safe size bounds, license and disk budget. Local import may link or copy a validated immutable blob, according to declared policy. Model profile is a **typed overlay**, not executable foreign code. Alias operations are versioned; old live sessions remain pinned to original model/profile revision.

## 2. Progressive disclosure, not a configuration tax

- **Level 0:** `run model`, `serve`, correct safe auto-plan. No mandatory questions about GPU offload, compiler or KV.
- **Level 1:** `models list`, `models ps`, `inspect` for available/loaded/unloaded, effective context, devices, memory usage and simple timings.
- **Level 2:** `inspect --explain-plan` and `config show --effective` expose actual source/quant/adapter, alias/profile, loaded replicas, backend/shape fallback, cache locality, residency, latency and why a faster/remote variant was refused.
- **Level 3:** user-readable versioned config and named profiles; scoped CLI flags to override allowed nonsecurity preferences.
- **Level 4:** experts can adjust per-model context, physical memory/kv/expert cache budgets, device/phase/quant precision, fused kernels/capture, sampling, draft/LoRA, replicas and SLO policies, routing strategies, remote layer/state/cache placements and tunnel/security/trace settings.

The default terminal output is concise and approachable. Verbose diagnostics are opt-in; hidden model-family-specific flags should not be needed to run an ordinary supported checkpoint.

## 3. Typed configuration, precedence and immutability

Every option carries a **typed scope, validation, effective-source, mutability boundary, security floor and cache invalidation rule**. The chosen on-disk representation can be TOML or another validated versioned format; the example below uses TOML with conventional XDG/OS user config paths.

**Values resolve deterministically**:
1. built-in safe capability defaults and immutable artifact/semantic constraints;
2. qualified observed hardware/shape/precision recommendations, subordinate to mandatory security and numerical floors;
3. administrator service policy (for serving), then allowed user global configuration;
4. source-verified model profile defaults;
5. alias/deployment-specific allowed overrides;
6. CLI command flags;
7. request-scoped allowlisted sampling/task parameters.

**Security and semantic invariants are hard constraints, not low-priority defaults.** A request cannot override data/tenant authorization, configured maximum physical admission, trusted model source/quant/adapter, or device-qualification gate. Model-load/placement overrides require a new warmed deployment or safe drain/StateRoot migration, not per-token source mutation. A request may set temperature/max output/stop only within supported sampler and limits.

Environment variables may be an **explicit documented source**, never mysterious values that override a CLI flag or disable auth. Final precedence, key migration, OS key-file/credential handling and API compatibility must be tested with two different model profiles and aliases.

### Proposed readable configuration

```toml
version = 1

[defaults]
profile = "balanced"
context = "auto"
device = "auto"
routing = "exact"
model_keep_warm = "auto"

[server]
host = "127.0.0.1"
port = 8080   # illustrative port, not a finalized default
remote_access = false

[models.coding]
source = "./models/code-model.gguf"
task = "chat"
context = 32768
keep_warm = "10m"

[models.embedding]
source = "./models/embedding-model.gguf"
task = "embeddings"

[aliases]
code = "coding"
embed = "embedding"

[router]
autoload_local = true
allow_unconfigured_source_download = false
policy = "cache-aware"
on_capacity = "queue-then-reject"

[residency]
physical_headroom = "auto"
max_resident_models = "auto"
idle_unload = "auto"

[network]
enabled = false
```

The *value of this example* is legibility, not binding the Rust implementation to these exact keys. No plaintext secrets should be printed by `config show`. Source model/profile must resolve deterministically, with no circular aliases or invalid family-specific options. An embedding model alias should not claim chat without a qualified chat TaskDriver. `config validate` should work offline without a model download or kernel compilation. `config explain` reports value/source/constraint/mutability/rejected overrides.

**Settings domains**:
- immutable checkpoint/model formula/quant/tokenizer and registered artifact;
- load-time model execution plan, devices/placement/quant packs/graph/replica group;
- residency and memory keepalive/pin/unload/queue/TTL;
- session context/LoRA/adapters, TaskDriver and full StateRoot;
- request sampler/seed/max tokens/response format/tool-output generation;
- router exact alias/resolved replica/cache affinity/explicit alternative model selector;
- network node trust/tunnels/layer and remote state/cache placements;
- API listener/TLS/auth/tenant/allowed model-source and security;
- diagnostics, tracing/quality/reference and source compiler experiments.

A setting change to immutable model identity creates a **new ModelGeneration/alias revision**, never mutates an active checkpoint session invisibly.

## 4. API compatibility matrix — explicitly qualified subset

**Compatibility means passing actual SDK/streaming/schema/error tests for an explicit endpoint/feature set, not claiming every current or future proprietary OpenAI API feature.** The API is a transport adapter to our canonical TaskRequest/TaskResult; it must not own sampling state or device kernels.

| Endpoint | Recommended release scope | Contract |
|---|---|---|
| `GET /v1/models` | Initial | authorized catalog models/aliases and compatible model IDs; native readiness separate |
| `POST /v1/chat/completions` | Initial | messages, model-native roles/rendering, supported sampler, SSE deltas/finish/usage, generated tool-call data |
| `POST /v1/embeddings` | Early when a real encoder qualifies | exact encoder/pooling/dimensions/usage, input batching; no fake decoder embeddings |
| `POST /v1/completions` | Subsequent | raw model-compatible text completion |
| `POST /v1/responses` | Incremental and versioned | stateless input/output first; stateful response_id/store/tool events only after explicit implemented conformance |
| `POST /v1/messages` | Optional Anthropic-compatible adapter after conformance | declared request/events/tool/reasoning/error subset mapped to the same TaskRequest and StateRoot |
| Finite-decision API | Optional source-qualified Decision TaskDriver | model-native label logits, method/calibration/session rules; explicit nonstandard compatibility profile |
| Native `/api/models` / `/api/health` | Privileged local management | alias/profile/load/unload/inspect/residency; never blindly public |
| Speech/image/media task endpoints | Only with qualified TaskDrivers/models | actual codec/formats/stage frontiers rather than placeholder HTTP 200 |
| Reranking/scoring | Native or clearly documented nonstandard extension | model-specific task semantics and score data |

`GET /v1/models` may return authorized registered **on-demand-loadable** aliases even when their physical models are not resident; never expose private filesystem paths, tenant-only model names or internal peer topology. A request's `model` is exact by default; omit model only with an explicitly configured default alias, not surprise model choice.

### Chat Completions

Expected request fields map to `ModelAliasRevision → exact Artifact/Quant/Adapter/ModelDefinition → PromptRenderer → TaskDriver(LanguageGeneration) → prepared StateRoot → committed output stream`. Validate role/content part/tool schema/stop/JSON format and tokenizer rules. For a model without a compatible tool/role template, report `ModelCapabilityUnsupported`; don't flatten different roles into misleading arbitrary strings.

**Streaming:** chat SSE chunks must follow chat-completion `data:` delta/finish conventions for the tested SDK/version, with correct UTF-8 incremental decode, stop across token boundaries, committed output only, usage and finish reason. Speculative drafts never leak into stream. A slow consumer has bounded queue/backpressure; errors after stream begins produce a typed terminal failure and consistent last committed output.

**Tool calls** are *model-generated structured data*. The runtime never executes tools, web search, arbitrary files or user-provided programs merely because it accepts a tool schema. Those are separately permissioned application integrations, not inference.

### Responses API (separate schema and events)

Responses streaming uses its own ordered `response.created`, output-item, text delta and final completion/failure events; it is **not** chat-completion SSE with a renamed URL. A stateless request can be supported first through TaskRequest; stateful `previous_response_id`, server-side store/conversation continuation and resumable output require a real authenticated StateRoot/session/retention contract.

**Privacy-first local default:** no persistent conversation store unless explicitly enabled; a request asking to `store=true` must be honored only when implemented and authorized. The behavior of a cloud provider's current `store` default is **not** automatically inherited. Built-in web/file/computer tools of cloud APIs are not promised through a compatibility endpoint. Return a typed unsupported feature error rather than claim those tools executed.

### Embeddings and other tasks

Embeddings use a qualified `EncodePoolScore` TaskDriver and actual vector outputs/dimensions/normalization. Speech transcription/synthesis and images use their own typed I/O and media codecs; OpenAI-like endpoint labels cannot imply support before real model/operator/reference/codec testing. Model aliases advertise actual task capability and overload/unsupported states without leaking private metadata.

### Authentication, permissions and resource limits

- default listener **loopback only**; any external interface needs explicit authenticated/TLS or approved reverse-proxy policy before start;
- a local SDK requiring a placeholder `api_key` header may be accepted syntactically if local auth is not required, but that string is **not a security token** and must not be logged; never ignore an invalid token on publicly reachable service;
- native model/admin control requires separate admin/local capability, not access granted to every generation client;
- all HTTP endpoints enforce bounded JSON/media/upload/request size, body and tokenizer limits, security-safe CORS, SSRF-resistant media fetch policy;
- remote peer identity and data/weight access are distinct from HTTP Bearer auth; a user API key cannot enroll an untrusted cluster node;
- model alias visibility, license rights, private snapshots and telemetry are tenant scoped.

## 5. Error, partial streaming and compatibility tests

Canonical errors include:
```text
ModelNotFound, ModelNotDownloaded, ModelLoading, ModelUnavailable,
UnsupportedModelOp, UnsupportedTask, UnsupportedApiParameter,
ModelAliasRevisionConflict, IncompatibleAdapter,
OutOfPhysicalBudget, NoQualifiedDevice, UnqualifiedRemotePeer,
ResourceBusy, RequestOverloaded, DeadlineExceeded,
BadRequest, Unauthorized, Forbidden, Cancelled,
StateIncompatible, ModelArtifactCorrupt, InternalFailure
```

HTTP adapter maps to stable tested envelope/status: 400 malformed, 401/403 auth, 404 unknown model, 409 model revision conflict, 422 unsupported capability, 429 quota, 503 queue/device overload, 500 internal; exact details verified against client libraries, not assumed. A loading model may wait within bounded deadline, then return retryable error and local progress, not hang indefinitely. For compatibility, optional harmless metadata can be handled only under an explicit documented policy; **unknown fields affecting sampling, output or security never silently ignored**.

Test against official Python and JS SDKs (pinned tested versions) for `GET /v1/models`, chat nonstream/stream, tool-call generation, embeddings, Responses stateless subset as implemented, precise token usage/output finish and UTF-8/stop/error. Conformance must include: unknown model, unsupported field, alias change during stream, slow consumer, interrupted stream, multi-model concurrent requests, malformed roles, bad JSON/tool, cancelled speculative decode and exact commit/usage. A 200 with a wrong schema is a failure.

## 6. Terminal UX details and advanced controls

- quick startup and concise model load/progress only when useful; interactive prompts support copy/paste/multiline text and Ctrl+C cancellation to last committed root (not killing other daemon sessions);
- stdin/stdout pipelines should work with no ANSI noise when redirected; useful noninteractive `--prompt`/output modes;
- proposed interactive commands `/help`, `/model`, `/status`, `/options`, `/clear` only when they add real clarity. Model switch starts a new explicit source binding/session, not an unsafe reuse of old KV/tokenizer;
- multimodal attachments are explicitly allowed local files; no arbitrary remote URL fetch or implicit prompt content exfiltration;
- concise `inspect` shows loaded state/context/device; optional `--explain-plan` exposes source/quant/adapter/alias, model replicas/physical memory, kernels and fallback reason, remote peer placement and estimated versus measured performance;
- per-model context/sampler defaults and keepalive are user configurable; a model's source fixed quant is not simply a config option to degrade silently;
- no login/account, cloud dependency or persistent daemon required for a local checkpoint; model pull may require explicit network/consent;
- no surprise global system services, root device tweaks, public network exposure or debug logs of prompts/API keys.

## 7. Advanced configuration, without complexity leakage

When explicitly enabled, users can configure model aliases and sources, task type, per-model profiles/LoRA/drafters, sampler/context/grammar, **multiple resident models and replicas**, keepalive/pins, physical RAM/VRAM/cgroup budgets, router queue/fairness and SLO policy, hardware/kernel/quant precision and graph capture, region placement by layer/attention/expert/cache, enrolled remote peers/tunnels/transport, tenant rights, debugging/profiling and exact API compatibility flags.

All effective values report their **source and security guard**. Changes with immutable model/program or StateRoot implications require a new deployment/profile revision or safe compatible committed frontier handoff, not in-place mutation. Source/quant/adapter/position identity is always visible. Some settings are not supported for particular models/hardware; explain with precise `UnsupportedParameter/NoQualifiedDevice` reason.

## 8. Same engine for CLI, server and SDK

`zerogravity run` can instantiate the library in process and issue TaskRequest directly; `zerogravity serve` runs the same ModelCatalog, ModelRouter, ResourceArbiter, TaskCoordinator and DeviceWorkers once and maps HTTP to canonical tasks. A Rust SDK embeds the same types. An optional trusted local socket can reuse a resident service after same-user identity/version verification, but is **not mandatory**.

Private distributed peers run the **same compiled-region worker** with authorized work scopes, not a public `/v1/chat/completions` gateway per accelerator. External cloud model APIs remain separate TaskServiceAdapters with explicit policy, rather than a false claim of native SIR execution.

## 9. Product acceptance checklist

- **One command:** a clean machine with a supported local model can run text without configuration, cloud, daemon or privileged change.
- **Multi-model:** one service lists authorized aliases, routes exact requested source, safely auto-loads available local model(s) when memory permits, shares model weights where compatible, and queues/rejects when full.
- **Profiles:** alias revisions, sampler/context/deployment settings predictable, no switching source in active session, different checkpoint quant remains separate artifact.
- **OpenAI:** passing pinned SDK nonstream/stream chat/models and qualifying embedding/Responses subsets with correct SSE/error/usage and tool-data semantics; not unsupported blanket compatibility.
- **Resource:** multi-model OOM pressure triggers safe unload/queue/refusal, not leaked KV or arbitrary quant.
- **Distributed:** sharded one model, equivalent replicas and many distinct models remain different model/placement constructs, authorized remote privacy/StateRoot and cache stickiness respected.
- **Performance:** standalone/direct `run` avoids per-token router/gateway overhead; model load/cold start/replica routing and C1/C8 P95 measured, not merely advertised.
- **Security:** loopback default, external authentication/admin controls, no model-supplied tools, no user-data URL/remote peer leakage, error paths safe.

**Status:** design targets only; no executable CLI, API endpoint, model router or benchmark exists yet.

## U-048 — task and frontend capabilities

The [task and frontend contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) covers qualified task and protocol capabilities:
Messages/Responses adapters, multi-vector embeddings, decision readout, image edits/generation, metrics and optional UI/TUI. Reuse canonical task
requests and one gateway; advertise only implemented/conformance-qualified subsets.
Multi-vector query/document embeddings retain token matrices and source markers;
never return a pooled vector as equivalent. UI/TUI are optional clients and do not
introduce another inference owner. No endpoint implementation is added here.

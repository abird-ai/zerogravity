# Next implementation iteration — full OpenAI-compatible serving

**24 Nov delivery order:** the [full implementation plan](../FULL_IMPLEMENTATION_PLAN.md)
sets phase priority. Shared foundations (including optional compiled-provider
scaffolding), CLI/TUI/Web and serving, ROCm, CUDA, Vulkan, then concrete
compiled providers. Native Hugging Face/Safetensors and GGUF are primary.
Older ordering below is context; valid owner constraints remain authoritative.

Status: **design only; OA-1 is next** (26 Nov, UX-P accepted).

26 Nov user decision: [UX-P native TUI polish](../cli-ux/TUI_POLISH_AND_MOUSE_INTERACTION.md)
was the prerequisite. The assembled P4B gate passed on ad7a4178;
public serving is unblocked. No new public API acceptance is implied.

26 Nov user follow-up: normal Cargo CLI defaults are the production
superset (all current model features, CPU/CUDA/ROCm/Vulkan, acquisition,
serving, TUI/Web), not diagnostic/qualification features. Existing
HTTP/SSE error boundary now keeps stable code/status, safe public
messages, original typed internal failures and payload-free operational
FailureDetails. The native document's serve.enable_error_details opt-in
requires authenticated loopback; only authenticated requests receive
bounded diagnostic text/causes in error.details. Preserve redaction
by default and pre-auth refusal; any future public/admin/debug scope
must not expose these details to ordinary untrusted clients. Logs use
the original bounded event ring and safe metadata, not raw prompts,
paths or credentials. No public serving acceptance implied.

This iteration runs **after**
`../cli-ux/NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md`
and **before**
`../cpu-gpu-optimization/NEXT_ITERATION_BACKEND_PARITY_AND_EXECUTION_PERFORMANCE.md`.

It is intentionally isolated from the live implementation TODOs and from the
later physical-execution optimization work.

The CLI iteration should leave behind a clean `zg serve MODEL...` command,
one ConfigDocument/profile model, one model resolver and one presentation/event
layer. This iteration takes that stable product surface and makes the server
behind it a comprehensive OpenAI-compatible local inference server.

No endpoint, model family or modality is considered implemented merely because
it appears in this design.

---

## 1. Objective

`zg serve` should mean:

> Start ZeroGravity's OpenAI-compatible local inference server over the models
> and pipelines actually loaded and qualified on this host.

A normal OpenAI client should be able to use ZeroGravity by changing only:

- base URL;
- API key if configured;
- model name.

No ZeroGravity-specific client adapter should be required for standard inference
features.

The compatibility target is the OpenAI **inference** surface, not the entire
hosted OpenAI business/control plane.

Do not clone:

- billing;
- organization/account management;
- hosted fine-tuning lifecycle;
- hosted eval infrastructure;
- hosted file storage as a generic cloud product;
- unrelated administrative APIs.

Implement OpenAI wire semantics only where they correspond to a real local
ZeroGravity inference/runtime capability.

---

## 2. Ordering

The intended iteration sequence is:

1. current live implementation iteration closes;
2. CLI/model-lifecycle/terminal UX iteration;
3. **this OpenAI-compatible serving iteration**;
4. backend execution-performance iteration.

Do not start physical performance work merely because this protocol iteration is
partially complete.

Do not pull model/backend implementation work into this server iteration solely
to make an endpoint return success.

---

## 3. Current server: strong bootstrap, not full compatibility

The current ZeroGravity serving crate already has a useful bounded foundation.

Implemented today:

- explicit listener ownership;
- bounded HTTP/1.1 ingress;
- optional Bearer authentication;
- GET /health;
- GET /v1/models;
- POST /v1/completions;
- POST /v1/chat/completions;
- non-streaming JSON responses;
- SSE/chunked streaming and [DONE];
- usage accounting;
- request cancellation on client disconnect;
- bounded ingress charges;
- graceful transport/runtime shutdown.

Current limitations are deliberate and material:

- Responses is explicitly unsupported;
- Embeddings is explicitly unsupported;
- Realtime WebSocket/WebRTC is explicitly unsupported;
- chat accepts only plain string content;
- HTTP transport builds Qwen ChatML itself;
- current generation surface is greedy-only;
- only one HTTP request is accepted per TCP connection;
- responses use Connection: close;
- requests are JSON-only;
- no multipart/form-data;
- no WebSocket transport;
- no general OpenAI response/conversation resource store.

Therefore the current implementation must be described as:

> partial OpenAI-compatible text serving

until this iteration closes.

---

## 4. Compatibility boundary

"Full OpenAI-compatible" means:

> Every inference capability ZeroGravity exposes uses the corresponding current
> OpenAI request/response object shapes, event grammar, error shape and client
> behavior for that capability.

It does **not** mean:

> Every OpenAI cloud endpoint must exist locally.

A route is not "supported" simply because the HTTP path parses.

Support requires a real loaded model/runtime capability.

Examples:

- /v1/embeddings requires a loaded embedding-capable execution adapter;
- audio transcription requires an ASR-capable model/pipeline;
- speech requires a TTS-capable model/pipeline;
- image generation requires an image-generation runtime;
- Realtime requires a streaming session/pipeline capable of servicing its event
  semantics.

Unsupported capability must return a canonical OpenAI-shaped error, not fake
success and not silent fallback.

When this iteration is activated, re-check the then-current OpenAI API
documentation before freezing the compatibility matrix. The OpenAI surface
changes over time; do not treat this document as a frozen wire schema.

---

## 5. Canonical endpoint families

The target local inference surface should cover the OpenAI endpoint families that
map naturally onto ZeroGravity.

### 5.1 Models

~~~text
GET /v1/models
GET /v1/models/{model}
~~~

The list contains models actually served by this process.

Do not return every model in the download catalog.

### 5.2 Responses

Responses is the primary text/multimodal request surface.

Target:

~~~text
POST   /v1/responses
GET    /v1/responses/{response}
DELETE /v1/responses/{response}
POST   /v1/responses/{response}/cancel
GET    /v1/responses/{response}/input_items
POST   /v1/responses/input_tokens
~~~

Implement additional response-resource operations required by the current API
contract when this iteration begins.

### 5.3 Conversations

Where current Responses semantics require server-managed conversation resources,
support the corresponding local conversation operations:

~~~text
POST   /v1/conversations
GET    /v1/conversations/{conversation}
POST   /v1/conversations/{conversation}
DELETE /v1/conversations/{conversation}

GET    /v1/conversations/{conversation}/items
POST   /v1/conversations/{conversation}/items
GET    /v1/conversations/{conversation}/items/{item}
DELETE /v1/conversations/{conversation}/items/{item}
~~~

These are protocol resources, not a new model-state engine.

### 5.4 Chat Completions

~~~text
POST /v1/chat/completions
~~~

Remain compatible because a large ecosystem still uses it.

It must become an adapter into the same semantic request model as Responses,
not a second execution implementation.

### 5.5 Legacy Completions

~~~text
POST /v1/completions
~~~

Keep for compatibility with older inference clients.

It remains a thin adapter.

### 5.6 Embeddings

~~~text
POST /v1/embeddings
~~~

Requires an embedding-capable loaded model.

### 5.7 Audio

~~~text
POST /v1/audio/transcriptions
POST /v1/audio/translations
POST /v1/audio/speech
~~~

These should route to ZeroGravity's common ASR/TTS execution machinery as those
model implementations become available.

### 5.8 Images

~~~text
POST /v1/images/generations
POST /v1/images/edits
POST /v1/images/variations
~~~

These should route to common image model execution, not an HTTP-specific image
engine.

### 5.9 Realtime

First server-side target:

~~~text
WebSocket /v1/realtime?model=...
~~~

Use the current OpenAI Realtime WebSocket event contract when this iteration is
activated.

WebRTC may follow as a separate transport over the same session/event authority.

### 5.10 Future inference families

If ZeroGravity later supports additional OpenAI inference families such as video
or moderation, add the canonical OpenAI surface then.

Do not pre-create endpoint stubs as "supported."

---

## 6. One semantic request model beneath every protocol

The most important architecture rule is:

**OpenAI protocol shapes are adapters, not execution authority.**

Do not implement Responses by translating it into today's Chat Completions JSON
and then parsing that again.

Do not keep model-specific ChatML construction inside HTTP code.

The desired shape is:

~~~text
OpenAI Responses ─────────┐
Chat Completions ─────────┤
Legacy Completions ───────┤
zg run ───────────────────┤
future native clients ────┘
                         │
                         ▼
          typed semantic Input / Conversation request
                         │
                         ▼
         model capability + tokenizer/template adapter
                         │
                         ▼
               shared Runtime / scheduler
~~~

The semantic request type should be expressive enough for:

- text input;
- conversation turns;
- typed content parts;
- images;
- audio;
- tool calls/results;
- structured output;
- generation controls;
- reasoning controls when supported;
- streaming.

The core semantic request does not need to be identical to OpenAI's JSON schema.

It needs to preserve enough typed meaning that protocol adapters do not lose
information.

---

## 7. Model-owned prompt/conversation lowering

The HTTP layer must not know Qwen ChatML tokens.

Each model/tokenizer/template adapter owns:

- role framing;
- generation prompt;
- tool-call serialization;
- multimodal placeholder semantics;
- tokenizer-specific special tokens;
- model-family conversation rules.

This is necessary for:

- Qwen;
- Gemma;
- Mistral;
- Llama;
- DeepSeek;
- future multimodal/audio models.

A protocol request can be valid OpenAI input while a specific model does not
support every content type. That is a capability error at the model adapter
boundary.

---

## 8. Loaded-model capability registry

Every served model exposes a small typed capability set derived from real
runtime/model qualification.

Possible capabilities include:

~~~text
TextGeneration
Chat
Responses
StructuredOutput
ToolCalls
Embeddings
VisionInput
AudioInput
Transcription
Translation
Speech
ImageGeneration
ImageEdit
ImageVariation
Realtime
~~~

Do not use one static marketing catalog for runtime routing.

Capability checks occur before expensive work.

Examples:

- an embeddings request to a text-only model refuses before scheduling;
- image content sent to a text-only chat model refuses before tokenization;
- Realtime refuses if the loaded model/pipeline has no streaming-session
  capability.

The capability set must remain model/backend truthful.

---

## 9. Responses API

### 9.1 Request semantics

Support the current locally meaningful Responses request fields when implemented
by the selected model/runtime, including:

- model;
- input string or typed item list;
- instructions;
- max output tokens;
- text output controls;
- structured output / JSON schema;
- tools;
- tool choice;
- function/tool call arguments;
- sampling controls;
- reasoning controls;
- multimodal input;
- streaming;
- conversation/previous response state;
- store semantics.

Do not silently ignore valid OpenAI fields.

For a valid OpenAI field unsupported by the selected local runtime, return an
OpenAI-shaped capability error naming the parameter/capability.

### 9.2 Hosted OpenAI tools

Do not silently emulate cloud-hosted tools such as:

- web search;
- hosted file search;
- hosted computer/container tools;

unless the user has explicitly configured a local/provider-backed ZeroGravity
tool with compatible semantics.

Otherwise return a clear unsupported-tool error.

### 9.3 Streaming

Responses streaming must emit the current Responses event grammar.

Do not reuse Chat Completion chunk objects under /v1/responses.

Internally use typed semantic events, then render the OpenAI event names/shapes at
the protocol boundary.

Disconnect cancellation maps onto the existing request/session cancellation
authority.

---

## 10. Response and conversation state

Responses/conversations may require server-managed protocol resources.

Implement one bounded protocol store.

It is not model StateRoot authority.

### Rules

- response IDs are opaque server-owned IDs;
- conversation IDs are opaque server-owned IDs;
- item IDs are opaque server-owned IDs;
- logical model state stays in Runtime/session ownership;
- protocol records retain only data needed for compatible retrieval/continuation;
- store count, bytes and age are bounded;
- store memory is resource-admitted/bounded;
- shutdown retention semantics are explicit;
- persistence, if added, is opt-in and separate from model cache;
- no unbounded history.

### Stateless requests

Where current OpenAI semantics permit it, store=false should avoid protocol-store
retention.

Do not retain conversation history merely because it is convenient.

---

## 11. Chat Completions compatibility

/v1/chat/completions should support the current OpenAI message/content grammar
applicable to loaded models.

As capabilities arrive this includes:

- system;
- developer;
- user;
- assistant;
- tool;
- string content;
- typed text content parts;
- image content;
- audio content;
- tool calls;
- tool results;
- structured output;
- streaming;
- standard sampling/output controls.

Chat remains a protocol adapter into the common semantic request.

No HTTP-layer ChatML.

No separate scheduler.

No separate tokenizer path.

---

## 12. Legacy Completions

/v1/completions remains useful for older clients.

It maps prompt input directly into the same typed generation request.

Keep compatibility semantics where sensible, but do not let this endpoint drive
new core architecture.

---

## 13. Embeddings

/v1/embeddings should support the current OpenAI-compatible input forms that the
loaded embedding model can truthfully accept.

Potential forms include:

- single string;
- array of strings;
- token-array inputs where current OpenAI semantics permit them;
- dimensions when the selected embedding model can expose that behavior.

Return standard embedding objects and usage.

Embedding batching belongs in the common runtime.

Do not create a special HTTP embedding executor.

---

## 14. Audio

Audio endpoints are strategically important because ZeroGravity intends to
support ASR/TTS pipelines in-process.

### Transcriptions

Map multipart audio + options into the common ASR runtime.

### Translations

Map into a translation-capable ASR/runtime path where supported.

### Speech

Map text/speech controls into the common TTS runtime.

### Streaming

When supported by the selected runtime, expose the current OpenAI streaming wire
semantics.

Do not build:

- HTTP-only Whisper;
- HTTP-only TTS;
- a second audio scheduler.

Multipart decoding and protocol response encoding live in
zerogravity-serving; tensors/model execution do not.

---

## 15. Images

Image generation/edit/variation endpoints route into common image execution.

The server owns:

- multipart decoding;
- request validation;
- response encoding;
- URL/base64 response selection where compatible.

The runtime owns:

- image tensors;
- model state;
- device execution;
- scheduling;
- cancellation.

Large image bodies require bounded streaming ingress.

Never buffer an unbounded multipart body simply because the endpoint accepts
files.

---

## 16. Realtime

Realtime should be a major strength of ZeroGravity rather than a sidecar server.

The desired path is:

~~~text
OpenAI Realtime WebSocket
        │
        ▼
typed realtime session/events
        │
        ▼
shared pipeline/session authority
        │
        ├── ASR
        ├── LLM
        ├── tools
        └── TTS
~~~

Support current Realtime semantics such as:

- session update;
- conversation/input item creation;
- input audio append;
- input audio commit;
- response creation;
- response cancellation;
- text/audio deltas;
- tool calls;
- terminal response events.

Do not create a second realtime inference runtime.

Realtime should reuse:

- model registry;
- pipeline graph;
- session ownership;
- cancellation;
- bounded queues;
- resource admission.

### WebRTC

WebRTC is a future transport over the same Realtime session/event semantics.

Do not couple model execution to WebSocket framing.

---

## 17. OpenAI objects stay clean

Canonical OpenAI requests/responses must not accumulate arbitrary ZeroGravity
fields.

Current ZeroGravity diagnostics such as:

- token IDs;
- per-token GPU observations;
- internal publication timing;

remain valuable, but should be exposed through explicit extensions.

Preferred extension surfaces:

~~~text
X-ZeroGravity-...
/zg/...
~~~

or an explicitly opted-in namespaced extension object when that cannot collide
with OpenAI fields.

Normal OpenAI SDK calls should see standard OpenAI objects.

Standard responses should not gain unexpected top-level
zerogravity_* fields by default.

---

## 18. Errors

Map internal typed errors into the canonical OpenAI error envelope.

Conceptually:

~~~json
{
  "error": {
    "message": "temperature is unsupported by this model",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": "unsupported_parameter"
  }
}
~~~

Maintain stable ZeroGravity internal error codes behind this boundary.

Do not expose:

- Rust type names;
- allocator internals;
- StateRoot IDs unless explicitly diagnostic;
- physical owner names;
- raw provider errors containing secrets.

---

## 19. Request identity

Every accepted request gets one server-generated stable request ID.

Use it consistently for:

- response headers;
- logs;
- stream diagnostics where appropriate;
- verbose CLI status;
- metrics correlation.

Do not require the caller to provide one.

If OpenAI's current API uses a specific request-id header, follow it.

If a caller provides its own correlation header, keep it distinct from the
server request ID.

---

## 20. Transport architecture

Do **not** evolve the current hand-written HTTP parser into a general-purpose web
stack.

It was the right bootstrap because it provided:

- bounded headers/bodies;
- duplicate-key refusal;
- explicit connection ownership;
- disconnect cancellation;
- exact application-owned ingress charging;
- simple shutdown semantics.

Full OpenAI compatibility requires mature protocol support:

- persistent HTTP/1.1 keep-alive;
- multiple sequential requests per connection;
- streaming request bodies;
- multipart/form-data;
- SSE;
- WebSocket upgrade and frames;
- robust backpressure;
- robust disconnect handling;
- standard client interoperability;
- potentially HTTP/2 where useful.

Use a mature native async Rust HTTP stack inside zerogravity-serving.

A likely implementation is:

~~~text
Tokio
  + Hyper
  + optional Axum typed routing/extractors where that reduces code
~~~

Choose the smallest composition that keeps the server clear and maintainable.

### Dependency boundary

Tokio/Hyper/Axum types must not leak into:

- zerogravity-core;
- model crates;
- engine scheduler;
- StateRoot/candidate state;
- artifact/cache;
- GPU execution.

The execution fabric remains transport independent.

---

## 21. Async server ownership

The async server must preserve explicit lifetime ownership.

Conceptually:

~~~text
Server
  ├── Listener owner
  ├── Connection/task set
  ├── OpenAI protocol store
  ├── Model/runtime handle
  └── Shutdown controller
~~~

Drop/shutdown must not detach request tasks.

Graceful shutdown should:

1. stop accepting new connections;
2. stop accepting new requests on existing connections;
3. stop new WebSocket sessions;
4. cancel or drain active requests according to configured graceful policy;
5. join all protocol tasks;
6. close protocol stores;
7. then permit Runtime/model retirement.

Do not let transport tasks outlive the server owner.

---

## 22. Resource accounting

Moving to a mature HTTP stack does not surrender ZeroGravity's bounded-resource
design.

Explicitly bound:

- connections;
- requests per connection;
- header bytes;
- request body bytes by endpoint;
- multipart file/part bytes;
- decoded JSON complexity where material;
- buffered response bytes;
- SSE queued data;
- WebSocket inbound/outbound queues;
- open Realtime sessions;
- response/conversation store bytes/count;
- server task count.

Application-owned retained buffers should be admitted/priced according to the
existing resource contract before they become long-lived.

Do **not** pretend to precisely account for all allocator/runtime internals inside
Tokio/Hyper.

Instead maintain a documented bounded external-runtime headroom allowance.

That is more truthful and maintainable.

---

## 23. Backpressure

Backpressure must be bounded end-to-end.

### SSE

Do not buffer an unbounded token/event queue when a client reads slowly.

Use bounded channels or direct awaitable write flow.

### WebSocket

Inbound and outbound event queues are bounded.

### Multipart

Stream file parts into bounded admitted staging rather than reading the entire
request into RAM.

### Runtime output

If client output credit is blocked beyond the configured deadline:

- cancel the affected request/session;
- preserve existing runtime completion/cancellation semantics;
- do not affect unrelated requests.

---

## 24. Connection semantics

Full compatibility should support normal persistent HTTP behavior.

One TCP connection may service multiple sequential requests.

Cancellation is per request, not per TCP connection.

Closing a connection cancels only live requests whose transport ownership depends
on that connection.

Do not tie one Runtime session permanently to one HTTP socket.

---

## 25. Authentication and exposure

Keep safe defaults.

### Default

Loopback only.

### Authentication

Optional Bearer token.

Credentials:

- never logged;
- never stored in normal config/profile files unless a future credential store is
  explicitly designed;
- compared safely.

### Remote exposure

A trusted TLS/auth reverse proxy remains a valid recommended production path.

Native TLS may be added later if useful.

Do not permit plaintext authenticated non-loopback exposure by accident.

---

## 26. ZeroGravity-native admin surface

Keep OpenAI compatibility clean.

Recommended namespace split:

~~~text
/v1/...     OpenAI-compatible inference API
/health     minimal health
/zg/...     ZeroGravity-specific diagnostics/admin
~~~

Possible /zg features later:

- admission details;
- runtime/device health;
- detailed model capabilities;
- internal metrics;
- token/GPU observations;
- scheduler diagnostics.

These do not alter /v1 semantics.

---

## 27. Serve command relationship

The previous CLI iteration owns the user-facing invocation:

~~~text
zg serve qwen3.5-2b
zg serve qwen3.5-0.8b qwen3.5-2b
zg --profile workstation serve qwen3.5-2b
~~~

This iteration should not redesign that CLI.

It implements the server behind the stable command.

The same:

- ModelRef;
- ConfigDocument;
- profile selection;
- device resolver;
- model lifecycle;
- rich/plain presentation;

remain in use.

---

## 28. Serving config

The common ConfigDocument should expose only high-value server policy.

Example:

~~~yaml
serve:
  listen: 127.0.0.1:8000
  active: 8
  max_pending: 128

  openai:
    responses_store:
      max_items: 1024
      max_bytes: 64MiB
      max_age: 1h

    realtime:
      max_sessions: 16
~~~

Do not dump every Hyper/Tokio tuning knob into user config.

Transport-internal tuning belongs in sensible defaults until real use requires a
public control.

---

## 29. Compatibility matrix

Maintain one concise table in serving documentation.

Example structure:

| Family | Wire compatibility | Runtime capability | Status |
| --- | --- | --- | --- |
| Models | GET list/retrieve | loaded registry | complete |
| Responses text | request/stream/store | text generation | complete |
| Chat | request/stream | chat generation | complete |
| Completions | request/stream | text generation | complete |
| Embeddings | request/response | embeddings | pending |
| Audio transcription | multipart/response | ASR | pending |
| Speech | request/audio | TTS | pending |
| Images | multipart/JSON | image model | pending |
| Realtime WS | event grammar | realtime pipeline | pending |

Wire support and runtime support are separate columns.

This prevents "route parses" from being mistaken for "feature works."

---

## 30. Conformance strategy

Do not vendor a giant OpenAI API test corpus.

Do not create hundreds of static JSON fixtures.

For each endpoint family:

1. focused protocol unit tests;
2. one raw-wire test where framing matters;
3. one official OpenAI SDK smoke pointed at ZeroGravity's base URL;
4. one smallest real loaded model capable of that family;
5. cancellation/backpressure where relevant;
6. clean shutdown/retirement.

### SDK smoke

The normal acceptance is:

~~~python
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="..."
)
~~~

and then the ordinary SDK call for that endpoint.

No ZeroGravity adapter.

Add another language SDK only when an actual SDK-specific interoperability bug
appears.

---

## 31. Avoid over-engineered evidence

Reuse existing:

- serving lifecycle tests;
- model/runtime correctness;
- source authentication tests;
- cancellation tests;
- ingress/resource tests.

Do not duplicate those in protocol tests.

Protocol tests prove:

- parsing;
- object shape;
- streaming framing;
- endpoint routing;
- capability refusal;
- SDK interoperability.

Real-model tests prove:

- one actual adapter reaches the shared runtime and returns correct output.

Keep the distinction clean.

---

## 32. Implementation slices

### OA-1 — async transport substrate

Replace the bootstrap one-request HTTP server with a mature async substrate.

Implement:

- keep-alive;
- bounded connection tasks;
- graceful shutdown;
- bounded request bodies;
- SSE support;
- streaming bodies;
- multipart primitives;
- WebSocket foundation;
- auth;
- ingress bounds;
- backpressure.

Keep current /v1/models, Chat and Completions semantics working during the
migration.

Do not add new model semantics yet.

### OA-2 — semantic request model and model template boundary

Introduce the shared typed semantic request/event representation.

Move model-specific chat/template lowering out of HTTP.

Adapt:

- CLI run;
- Chat Completions;
- legacy Completions;

onto the same semantic request where practical.

No Responses endpoint until this shared seam is sound.

### OA-3 — Models + Responses text

Implement:

- GET /v1/models;
- GET /v1/models/{id};
- POST /v1/responses;
- current Responses text request fields;
- Responses SSE event grammar;
- canonical errors;
- request IDs.

Use the smallest real text model canary.

### OA-4 — response/conversation resources

Add bounded:

- response store;
- cancellation;
- retrieval;
- deletion;
- input-item retrieval;
- conversations/items;

as required by the current API.

Keep protocol state separate from model StateRoot.

### OA-5 — Chat/Completions compatibility close

Expand Chat to current supported message/content/tool/structured-output semantics
over the common request model.

Keep legacy Completions compatibility.

Remove HTTP-layer ChatML.

### OA-6 — Embeddings

Add the wire adapter only with a real embedding-capable runtime/model.

### OA-7 — Audio

Add:

- transcription;
- translation;
- speech;

over common ASR/TTS runtime capabilities.

### OA-8 — Images

Add generation/edit/variation over common image execution.

### OA-9 — Realtime WebSocket

Implement current OpenAI Realtime WebSocket event semantics over the common
session/pipeline runtime.

Do not add WebRTC in the same slice unless it is already trivial and required.

### OA-10 — compatibility closeout

Run:

- concise endpoint matrix;
- official SDK smokes;
- shutdown/cancellation;
- standard-object audit for accidental ZeroGravity fields;
- docs/help updates.

Retire duplicated bootstrap transport paths only after equivalent bounded
behavior is established.

---

## 33. Performance constraints

The server should be efficient, but this is not the model-kernel performance
iteration.

Avoid avoidable protocol overhead:

- parse once;
- typed conversion once;
- bounded allocation;
- no unnecessary JSON reserialization between adapters;
- no per-token unbounded allocation;
- direct streaming where practical.

Do not begin CUDA/HIP kernel optimization here.

Do not distort model execution architecture to reduce HTTP microseconds.

---

## 34. Rust design principles

Keep the implementation:

- native Rust;
- async where protocol concurrency benefits;
- explicit ownership;
- typed;
- bounded;
- transport-independent below serving;
- zero-copy where simple and useful;
- allocation-aware without becoming convoluted.

Prefer standard mature protocol libraries over custom HTTP/WebSocket machinery.

Prefer simple typed adapters over a generic meta-framework.

No macros/type gymnastics merely to look abstract.

---

## 35. Acceptance criteria

This iteration closes when the server can truthfully be called a comprehensive
OpenAI-compatible local inference server for all inference capabilities
ZeroGravity actually implements at that point.

### Text

- official OpenAI SDK works with base_url override;
- /v1/responses is primary;
- Responses streaming matches current event grammar;
- Chat Completions remains compatible;
- legacy Completions remains compatible;
- model-owned conversation templates;
- structured/tool semantics capability-gated correctly.

### Model discovery

- list;
- retrieve;
- actual served model IDs only.

### Protocol state

- bounded response/conversation resources where required;
- cancel/retrieve/delete semantics;
- no conflict with StateRoot authority.

### Other modalities

For each runtime capability implemented by then:

- canonical OpenAI endpoint;
- official SDK smoke;
- same common model registry/runtime.

Unsupported modalities refuse canonically.

### Realtime

When the runtime capability exists:

- OpenAI Realtime WebSocket works;
- same session/pipeline authority;
- bounded queues/backpressure;
- clean cancellation.

### Transport

- normal keep-alive;
- SSE;
- multipart;
- WebSocket;
- bounded bodies;
- bounded tasks;
- graceful shutdown.

### Clean compatibility

- no accidental ZeroGravity request fields required;
- no accidental ZeroGravity response fields;
- canonical error envelope;
- stable request IDs;
- /zg namespace for extensions.

### Maintainability

- no second scheduler;
- no HTTP-specific model runtime;
- no HTTP-layer model templates;
- no custom general-purpose HTTP stack;
- no giant conformance fixture repository;
- one semantic request/event layer;
- one model capability registry.

---

## 36. Relationship to the performance iteration

The following backend execution-performance iteration should assume:

- the CLI/model lifecycle is stable;
- `zg serve` is stable;
- OpenAI protocol adapters are no longer changing the execution request model;
- Responses/Chat/Completions feed one semantic request seam;
- server transport is mature async infrastructure;
- performance measurements can use the stable public server without repeatedly
  changing interface semantics.

Physical execution optimization remains separate.

The performance iteration should optimize what happens **after** semantic request
admission, not redesign OpenAI protocol parsing.

---

## 37. First activation prompt

After the CLI/model-lifecycle UX iteration closes:

> Read the current zerogravity-serving implementation,
> `NEXT_ITERATION_CLI_UX_AND_MODEL_LIFECYCLE.md`, and
> `NEXT_ITERATION_OPENAI_COMPATIBLE_SERVING.md`. Preserve the CLI/model/config
> surface established by the prior iteration. Begin OA-1 only: replace the
> bootstrap one-request HTTP transport with a mature bounded async Rust transport
> while preserving existing /v1/models, Chat Completions, Completions, auth,
> cancellation, ingress bounds and shutdown behavior. Do not add model semantics,
> Responses translation hacks or a second scheduler in OA-1. Keep transport
> dependencies confined to zerogravity-serving and prove the existing text API
> still works before proceeding to the shared semantic request layer.

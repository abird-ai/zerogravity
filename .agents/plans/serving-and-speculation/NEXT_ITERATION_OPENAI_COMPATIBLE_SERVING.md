# Next implementation iteration — full OpenAI-compatible serving

**24 Nov delivery order:** the [full implementation plan](../FULL_IMPLEMENTATION_PLAN.md)
sets phase priority. Shared foundations (including optional compiled-provider
scaffolding), CLI/TUI/Web and serving, ROCm, CUDA, Vulkan, then concrete
compiled providers. Native Hugging Face/Safetensors and GGUF are primary.
Older ordering below is context; valid owner constraints remain authoritative.

Status: **OA-1–5 accepted** (27 Nov), scoped to actual supported text capabilities. OA-6A native BERT candidate workspace-qualified; OA-6B native shared Runtime/SDK workspace-qualified; OA-6C gated wire/SDK workspace-qualified; trained-source and built-in eligibility remain open. OA-7A/B original cached Gemma4 encoder and shared audio-conditioned native F32 decoder parity passed; actual speech/Runtime/ASR integration remains open. Later units remain ordered.

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

#### OA-1A — bounded async text transport (26 Nov)

The ordinary server owns one current-thread Tokio runtime and bounded Hyper
HTTP/1 connection tasks. Pinned, locally cached transport dependencies are
confined to the serving crate; the engine's charged, fixed-slot event channel
adds only standard-library async receiver readiness. No second queue, model
owner or inference thread per client. Successive/pipelined requests use the
same connection; stop closes ingress, cancels protocol-owned receipts, and
joins every task without stopping caller-owned SDK work. The existing auth,
request parsing, body admission, JSON/SSE responses and diagnostic redaction
remain the semantic boundary.

Explicit text policy: 8 KiB header and 32-header caps, ten-second ingress
deadlines, bounded connection count, 256 requests per connection, cumulative
wire output budget and 30-second stalled-write bound. Connection and server
runtime storage each receive conservative 64 KiB admission allowances,
not an exact heap census; existing application body/output owners stay charged.
A bounded read-only duplicate socket observes peer closure even behind a
buffered pipeline. Hyper owns framing; invalid framing may produce a
protocol-level 4xx rather than application JSON. Hyper normalizes identical
repeated Content-Length; conflicting lengths and all chunked/Expect input
remain refused in this slice. Ordinary output is directly polled from the
original channel. Opt-in token observations measure completed SSE payload
handoff via actual socket writes, never client receipt or body polling.

OA-1A is not a general public API, TLS, HTTP/2, or external network exposure
claim. No new model semantics or GPU performance claim.

#### OA-1B — remaining bounded ingress and upgrade primitives

Finish charged unknown-length streaming request bodies, multipart primitives
and WebSocket foundations before closing OA-1. Use approved mature cached
dependencies where available (multipart dependency availability must be
checked; no automatic downloads). Keep auth, independent route/tenant scope,
size/deadline limits, cancellation and owned shutdown; no stub modality
endpoints or Responses translation hacks. Qualify new boundaries independently
and preserve OA-1A's original text behavior.

**OA-1B current implementation (26 Nov).** The existing fixed-length text
path consumes a bounded decoded-body reader under one absolute deadline.
The reader supports known and unknown lengths; its contiguous staging owner
charges exact backing before allocation, bounded geometric growth for unknown
length, and both allocations during a growth overlap. It rejects excess bytes,
incomplete bodies and trailers. No extra scheduler, upload store or unbounded
receive queue. Unknown-length staging is qualified against actual Hyper TCP
frames in a private fixture; public text still requires Content-Length.

Cached tokio-tungstenite 0.29.0 (handshake only; no TLS/client connector)
provides the reusable WebSocket primitive. Existing HTTP authentication and
header/body policy precede handshake validation and adapter admission. A
successful upgrade retains the same connection task, wire output budget,
connection slot, peer-close observation and joined shutdown. A separately
charged bounded frame/message allowance precedes the handshake; message IO
is direct and deadline-bounded. The private adapter admission callback owns
any semantic session/state and retained-message charges. Only a test adapter
is installed today: ordinary servers continue refusing /v1/realtime, and
RealtimeTransport::require_implemented still refuses. No modality endpoint,
model support or external exposure is claimed.

**Public chunked text activation (26 Nov).** The cached checksum-verified
Hyper 1.13.0 registry source is vendored with one server request parser
patch: refuse mixed Transfer-Encoding/Content-Length (either order) and
repeated Content-Length (including identical values), before normalization
or application dispatch. Client parsing/encoding stays upstream. Preserve
the exact source revision/license, one root lock, and existing vendor
packaging; remove this patch when upstream supports the strict policy.
No duplicate wire parser. The original public text routes now accept
only single chunked transfer coding on POST; Expect, other codings,
trailers and upgrades with chunked bodies remain refused. Decoded frames
use the same charged staging owner and absolute deadline as fixed-length
JSON, with authorization before reads and complete JSON validation before
model admission. Persistent connection and SSE ownership are unchanged.
No new model semantics or modality capability.

The full isolated retry on 08a9d30e exited 0 (2,546/0/109 across 318
Rust suites; real cached CPU consumers and SDK passed). Exact Nix
source-integrity remains unverified because locked inputs are absent.

**Multipart foundation (27 Nov).** Acquired only the authorized
`multer 3.1.0` registry archive; SHA-256 matches the existing cached
registry index. Other resolved dependencies were already cached.
Cached form-data 0.6.0 remains rejected (truncated-input loop and invalid
termination). Multer owns the parser; a narrow vendored patch rejects
duplicate part headers and requires the initial delimiter at body start
and final delimiter, optional transport padding/CRLF, and actual decoded
HTTP EOF before success. Preamble/epilogue are deliberately outside this
HTTP form subset. No separate handwritten parser. Source/license and
patch provenance retained under the serving vendor tree.

The existing decoded-body reader feeds the parser cooperatively, one
frame at a time. Original aggregate byte cap, absolute deadline and
trailer refusal remain authoritative. Conservative parser headroom and
bounded retained metadata are admitted before parsing; each part uses the
same exact charged backing growth as JSON, including old/new overlap.
Part count and per-part bytes are bounded; completed forms retain bytes
and metadata together. No filename-to-path interpretation, implicit
transfer decoding, persistent upload store or detached parser task.
Private actual TCP fixtures exercise the shared header/auth validation,
fixed/chunked forms, fragmented delimiters, repeated fields, binary/empty
fields, invalid/truncated forms, actual host budget refusal, size/count
bounds, trailers/deadlines and cancellation retirement. Ordinary server
still refuses media endpoints and multipart on JSON text routes. No new
model semantics or public Realtime/media claims.

Focused serving 38/0/0 and strict lint passed. Actual cached CLI/TUI/Web
and SDK passed (76.57s and 14.07s). Full isolated assembled gate passed
(902.1s): 2,551/0/109 across 318 Rust suites, release TUI/Web, installer
and scripts. OA-1 accepted; exact Nix source-integrity remains separately
unverified because four locked inputs are unavailable. OA-2 is next.

### OA-2 — semantic request model and model template boundary

Introduce the shared typed semantic request/event representation.

Move model-specific chat/template lowering out of HTTP.

Adapt:

- CLI run;
- Chat Completions;
- legacy Completions;

onto the same semantic request where practical.

**OA-2 implementation (27 Nov).** Protocol-neutral ChatRole, ChatMessage
and TextInput (raw, one user, ordered conversation) live in core/task;
engine SemanticRequest carries the existing generation and observation
controls. Legacy Request and borrowed TextTask are adapters; the original
RequestLease/Scheduled/Active worker path retains typed input and existing
ResponseEvent publication/cancellation. No extra execution owner, registry,
scheduler or event queue. Existing text events are the semantic event
contract for this slice; additional content/control/event kinds need
real model capability, not placeholder wire coercion.

HTTP validates only supported JSON shape and admits retained turn-vector
and string copies. The exact selected model capability and tokenizer
measure/validate the prompt before enqueue; PromptBounds prices rendered
input, prompt token storage and original tokenizer scratch. Structured
chat retains the historical 1 MiB rendered cap and configured runtime
input limit; legacy single text retains its original content-length
contract. The worker lowers the admitted typed input on the original
tokenizer; Qwen owns the same qualified ChatML template, preserving role
order, UTF-8 and assistant generation prefix. Single-user convenience
uses that renderer; literal role delimiters in typed chat now refuse
(including single-user CLI/SDK chat). Explicit raw text remains raw.
No arbitrary Jinja evaluation or new model capability. Legacy-only
TextService/Tokenizer implementations refuse structured chat unless they
explicitly implement it; PreparedModel validation consumes SemanticRequest.
The facade exposes request_semantic with the original typed event receipt.
The existing direct CPU diagnostic runner also consumes the same model
text intent/renderer; its physical runner and audit path are unchanged.

Focused final engine/serving 163/0/0 and strict lint passed. Qwen tokenizer
suite 7/0/0; explicit cached approved-tokenizer test 1/0/0 verifies exact
single/multi-turn token IDs and measured prompt bytes against the previous
raw spelling. Actual cached CLI/TUI/Web and SDK regressions passed
(67.77s and 15.85s); full isolated assembled gate exited 0 (878.6s),
2,553/0/109 across 318 Rust suites plus release TUI/Web, installer and
scripts. OA-2 accepted; no Responses endpoint introduced in this slice.

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

**OA-3 implementation (27 Nov).** Rechecked the current
[create](https://developers.openai.com/api/reference/resources/responses/methods/create)
and [streaming](https://developers.openai.com/api/reference/resources/responses/streaming-events)
reference. Models list/retrieve use accepting registered aliases only.
Responses text input string or ordered explicit message items, plain
text parts, instructions (developer), metadata and greedy generation
controls lower directly to SemanticRequest. Local stateless default
store=false; explicit store=true, background, prior-response/conversation
references, tools, structured output, other modalities/sampling/reasoning
and unsupported fields refuse with named static parameters where known.
No implicit resource lookup from a replayed complete assistant item ID.

Borrowed typed wire views and charged bounded projection over original
ResponseEvent/RequestEvents; exact restored prefix from the executor
cursor is exposed as additive cached_prompt_tokens on Done and projected
into input_tokens_details. Separate reasoning capability not claimed.
Named ordered SSE lifecycle, length -> incomplete, backend failure ->
failed with already-delivered partial text and safe canonical code.
No Chat chunks or [DONE] sentinel. Same listener, queue/cancellation,
deadline, backpressure, socket cumulative budget. Server correlation ID
and response/item IDs; preserve explicit legacy metrics-stream wire ID.
Focused 167/0/0 and strict lint; cached official OpenAI SDK 2.33.0 strict
validation passed actual cached Qwen3.5 Models/retrieve, Responses JSON
and SSE, exact Chat output/usage parity, refusal and correlation. Native/
browser/official SDK lifecycle (80.88s) and Rust SDK lifecycle (15.02s)
passed. Full isolated gate 2,557/0/109 across 318 Rust suites (617.5s),
release TUI/Web, installer and scripts passed. OA-3 accepted.

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

**OA-4A foreground resource slice (27 Nov).** Explicit store=true, terminal
response retrieval/deletion, paginated normalized input items and
previous_response_id text continuation. Memory-only single listener
trust scope, bounded count (64), conservative reserved backing (32 MiB)
and terminal age (30 min); listener-owned expiry and final reader
ownership. Reserve before model admission; no retained model session,
disk storage, detached tasks or additional scheduler. Record only
normalized history and output, serialized terminal view. Omitted store
stays false. Prior request-local instructions are not inherited; child
snapshots survive parent deletion. Pending foreground responses are
not addressable before terminal; failed stream records cannot continue.
Explicit cancel of foreground record refuses (current API requires
background). See live handoff for exact bounds and retention policy.
Focused 170/0/0 and strict lint pass; real installed SDK 2.33.0,
CLI/TUI/Web (134.83s) and Rust SDK (15.18s) lifecycle passed.
Full isolated assembled gate 2,560/0/109 (318 suites, 690.5s),
release TUI/Web, installer and scripts passed. OA-4A accepted.

**OA-4B background polling/cancellation (27 Nov).** Rechecked current
[background mode](https://developers.openai.com/api/docs/guides/background)
and [cancel](https://developers.openai.com/api/reference/resources/responses/methods/cancel).
Nonstream background=true create/poll/cancel through listener-supervised
bounded JoinSet and original event receipt; no second inference queue.
Same protocol quota and charged immutable snapshots. Actual receipt
cancel and atomic active-entry transition, late terminal cannot overwrite;
repeated cancel returns final object. Active delete refuses. Safe failure
fallback even on terminal serialization overflow; original deadline
and shutdown retirement. Local store=true 30min with continuation,
background store=false temporary 10min with polling only, foreground
store=false remains unretained. Default/soonest completion window only;
background streaming/resume refuses until bounded replay exists.
Focused 172/0/0 and strict lint passed. Real installed SDK/CLI/TUI/Web
(130.88s) and Rust SDK (14.36s) passed, including actual cancelled
on the cached model. Full isolated gate 2,562/0/109 across 318 Rust
suites (377.3s), release TUI/Web, installer and scripts passed.
OA-4B accepted.

**OA-4C conversation resources (27 Nov).** Conversation create/retrieve/
metadata update/delete and item create/retrieve/delete/list use the same
listener-local authenticated protocol store, reservation ledger and lock
as responses. Current API reference:
https://developers.openai.com/api/reference/resources/conversations
and https://developers.openai.com/api/reference/resources/conversations/subresources/items.
Text messages only; normalize through the original Responses text parser.
Metadata update replaces the map (null or empty clears it). New item IDs
are server-owned. Pages retain exact IDs and ascending/descending cursors;
invalid/deleted cursors refuse. At most 20 items per explicit create batch,
128 retained turns, 2 MiB retained text (1 MiB semantic input replay bound).
Each conversation reserves 2 MiB + 128 KiB of the existing 64-slot/32 MiB
quota before creation. Age out 30min after the last successful mutation,
not a read; active leases defer expiry. Replies retain separately admitted
immutable serialization, not borrowed deleted records. Local deletion
retires the scoped items; independent stored response snapshots survive.
No durable/cloud item store or tenant isolation beyond the listener.

Accept conversation ID string or object; mutually exclusive with
previous_response_id. Replay existing history through the original
SemanticRequest and model-owned template. Request-local instructions
remain request-local. Omitted input can generate from existing context.
Reserve one terminal assistant item before admission. One exclusive
mutation lease per conversation; reads remain available and competing
generation/edit/delete returns 409 until lease retirement. On successful
terminal completed or max-token incomplete, append only this request's
new input and actual assistant output, preserving its original output ID
and status. Publish append and any retained response record under the same
lock; no replay duplication, stale replacement, lost updates or partial
append. Foreground store=false needs no response record; explicit
conversation association still appends context. Background uses the
original joined receipt and cancellation race; failure, refusal,
pre-terminal disconnect or cancellation appends nothing. Cancellation can
briefly leave a mutation conflict until joined consumer retirement.
No model session retention or additional execution authority.

Focused engine/serving 177/0/0, strict lint passed. Real installed OpenAI
SDK 2.33.0 conversation/response canary passed exact Chat parity and
actual cancellation. Full isolated assembled gate 2,567/0/109 across
318 suites (342s), release TUI/Web, installer and scripts passed.
Serial CLI/TUI/Web and official SDK retry passed (124.91s), Rust SDK
(15.56s). First combined consumer run encountered a resource/execution
bound refusal during concurrent workspace build; serial retry passed
unchanged. OA-4C accepted.

**OA-4D exact input-token count (27 Nov).** Current API:
https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens/methods/count.
POST /v1/responses/input_tokens returns response.input_tokens with the
exact full rendered prompt token count, including the selected model's
generation prefix. Explicit loaded model alias required. Shared prompt
lowering and prompt-affecting capability checks with Responses generation:
typed text, instructions, previous-response replay and committed
conversation context; no inherited request-local instructions. The local
input-required policy is unchanged except existing conversation-only input.
Unsupported media/tools/reasoning/truncation/other controls refuse; count
does not reserve output or accept generation-only controls. The read-only
conversation snapshot measures and copies under the existing store lock,
with admitted input storage; active mutation lease does not prevent reading
the last committed context. No retention refresh, append, protocol record
or lease for counting. Generation still holds its original mutation lease.

Router, engine Runtime, serving TextService and public facade Runtime expose
count_input_tokens(model, &TextInput). Acquire the exact registry reader,
share generation's semantic/rendered byte checks and model-owned prompt
bounds, reserve original host scratch and actual token backing before
tokenization, count the resulting IDs, release scratch/tokenizer/model
before retiring the reader. Enforce positive prompt within configured model
context and advertised tokenizer token bound; no generated-span reserve.
A count is not a guarantee that later generation will fit or pass
execution-specific shape/control validation. No generation ticket, engine request
ID, worker, session, StateRoot, prefix lookup, output allocation or
performance-counter update. Legacy embedders explicitly refuse unless
they implement the capability. Synchronous bounded tokenizer call under
original Runtime lifecycle; no new detached job or scheduler.

Focused engine/serving 179/0/0, strict lint passed. Real cached Qwen3.5
official SDK 2.33.0 exact count/usage parity and CLI/TUI/Firefox lifecycle
(162.61s), native Rust SDK (18.79s) passed. Full isolated assembled
gate 2,569/0/109 across 318 suites (783s), release TUI/Web, installer
and scripts passed. OA-4D accepted.

**OA-4E bounded background streaming/replay (27 Nov).** Current API:
https://developers.openai.com/api/docs/guides/background and
https://developers.openai.com/api/reference/resources/responses/methods/retrieve.
Default/soonest completion window accepts background=true, stream=true.
GET response?stream=true replays the originally published SSE bytes,
optionally starting strictly after a previously published sequence_number.
Omitted cursor starts at zero; unknown/future/malformed/duplicate cursors
refuse. Last terminal event cursor yields EOF. Nonstream background and
foreground records cannot retroactively stream. Only disabled obfuscation
is supported; no new modality or cloud scheduling policy.

One existing listener JoinSet consumer owns the original event receipt.
The same typed foreground SSE grammar and bounded serializer produce
immutable event bytes. Replay subscribers hold only shared bytes/credit,
not a model session, receipt, queue or producer. No per-reader event
buffer; original socket admission, cumulative wire cap and stalled-write
deadline apply independently. A disconnected/stalled subscriber does
not cancel background work. Shared log cap min(configured output bytes,
1 MiB), 520 events (four opening + 512 deltas + four terminal), no
eviction or renumbering. Opening and deterministic failure event are
prepared before ID publication. Reserve space and one event slot for
failure; reuse its preallocated buffer if successful terminal group
or ordinary failure projection cannot fit. Additional 6x log bound
plus 128 KiB metadata allowance covers retained encoding/backing,
fallback overlap and partial text, in the same 64-slot/32 MiB protocol
quota and original physical ledger. The original generation deadline
and cooperative joined consumer remain bounded.

Validate exact delta/terminal text, usage and terminal status. Prepare
all successful terminal frames before publishing response/context append;
the same store transition wins against cancellation and closes the log.
Overflow/worker error produces failed response and no successful partial
terminal group or conversation append. Explicit cancellation signals
the original receipt, publishes the canonical cancelled snapshot and
closes subscribers after already retained events. No invented
response.cancelled SSE event (not in current public streaming union);
poll or cancel for the cancelled status. Previously emitted deltas are
replayable but not a completed assistant output. Repeat cancel is
idempotent. Same terminal retention (store=true 30min, temporary
store=false 10min); delete/expire blocks new subscribers but retained
reader/frame credit follows the final reader. Shutdown joins original
connection and background tasks. No additional scheduler, model-state
store, persistent replay, obfuscation or detached execution.

Focused engine/serving 182/0/0, strict lint passed. Actual TCP tests:
disconnect, active/terminal replay, concurrent readers, exclusive cursor,
once-only conversation append, cancel/late Done, bounded terminal
overflow, expiry/reader quota and joined shutdown. Actual OpenAI SDK
2.33.0 passed cached-model stream/resume, exact output/usage, full replay,
conversation append and actual cancellation. First fixture auth
assertion corrected for its explicitly unauthenticated listener; real
bearer-authenticated replay/refusal is covered by focused socket test.
Broader browser resident-model load timed out under observed shared
host pressure (cause not established); Rust SDK lifecycle passed
(19.96s). Full assembled private-network gate 2,572/0/109 across
318 suites (548s), release TUI/Web, installer and scripts passed.
Final serial resident Firefox/CLI/TUI lifecycle retry passed (83.29s)
on unchanged 37723346. No weakened admission or timeout; reused passed
SDK and assembled evidence. OA-4 accepted.

### OA-5 — Chat/Completions compatibility close

Current API references:
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
and https://developers.openai.com/api/reference/resources/completions/methods/create/.

Current supported text semantics (27 Nov): separate http/completions.rs
adapter lowers string and nonempty typed text parts, in original order
and role, directly to TextInput::Conversation; legacy single nonempty
string prompt remains TextInput::Raw. Selected model owns all framing
and delimiter validation; no HTTP-layer ChatML. Admit measured text,
alias and turn vector before copying. Explicit neutral controls (greedy
temperature/top_p/penalties, n=1, tools=[], auto/none tool choice,
boolean parallel tool calls, response_format=text, modalities=[text],
store=false, empty metadata/bias, logprobs=false/top_logprobs=0,
default/auto service tier) have no unavailable effects. Assistant text
replay accepts null tool/audio/refusal/function fields or empty tool
calls only. Legacy best_of=1/echo=false and null optional controls;
existing max_completion_tokens alias and token metrics retained.
Unknown, cross-route and active unsupported controls refuse before
admission; positive output limit within actual configured capacity.
No fake tool use, JSON guarantees, audio/image, stop processing,
non-greedy sampling, participant names, logprobs, bias or stored Chat.

One checked terminal count/finish validation shared with Responses;
Chat/legacy JSON and SSE use actual cached prompt count, total and
zero separate reasoning tokens. Chat requested include_usage places
null on normal chunks and one final usage chunk. Preserve existing
IDs, metrics and engine receipt ownership. Focused 184/0/0 and strict
lint passed; real cached official SDK 2.33.0, CLI/TUI/Firefox (184.78s)
and native Rust SDK (13.99s) passed serial retry. Actual typed/string
Chat parity, JSON/SSE usage, legacy raw JSON/SSE and unavailable
capability refusals. Legacy SDK CompletionChoice strict schema
incorrectly excludes null nonterminal finish_reason; ordinary legacy
streaming client explicitly checks partial/null and terminal reason,
identity, complete text and usage. All other SDK checks remain strict.
Earlier original host admission refusal and test-only SDK copy option
failure are recorded in current handoff. Full isolated assembled gate
2,574/0/109 across 318 suites (306s), release TUI/Web, installer and
scripts passed. OA-5 accepted for current supported text capabilities.
Remaining capabilities explicitly gated by actual semantic/model
implementations, not by protocol shape alone.

### OA-6 — Embeddings

Add the wire adapter only with a real embedding-capable runtime/model.

**OA-6A (27 Nov): native BERT encoder candidate.** The resident cached
generation-model inventory lacks a qualified embedding executor. First
target: original sentence-transformers/all-MiniLM-L6-v2 (BERT, six
layers, hidden size 384, source sentence limit 256). Model family owns
source config and module closure, WordPiece tokenization, tensor binding,
bidirectional forward and mean pooling/L2 normalization. Share original
CPU source GEMV/lookup and noncausal primitives; no fake logits-derived
embedding, decoder KV state, private Runtime or queue. F32 reference
candidate accepts bounded unpadded sequences and text, refuses invalid
geometry/unsupported module chains/overlong input, retains output
credit and exact source identity. Single Safetensors owner verifies
actual bytes from the existing cache descriptor before binding and
charges directory and storage; I64 index buffers never enter floating
weight projection. Synthetic source-independent oracle and ownership
tests pass; default CLI compile passes. Previous interrupted gate
superseded by full unchanged 6a5e3518 isolated gate, 2,581/0/109 across
319 Rust suites (580s), acquisition/release/installer passed.

**OA-6B (27 Nov): native shared Runtime.** Model-neutral synchronous
EmbeddingEncoder plus bounded typed EmbeddingRequest/EmbeddingBatch
through the original registry, request-ID allocator, PendingQueue,
workers, RequestEvents terminal channel and CancelToken. Max 16 ordered
rows, 4096 dimensions, checked aggregate input bytes, vocabulary and
runtime/source context. Shared active-slot budget; one sequence quantum
per iteration alongside text rounds. Charge private request/output before
enqueue; exact source-domain admission, caller-owned vector storage,
finite output and actual token usage validation. Publish complete batch
or discard; retained output charge independent of model retirement.
No extra decoder state, queue or serving worker. Native snapshot reports
embedding dimensions; prefill/token-byte fields optional for non-text
owners. Text list/resolve and HTTP models remain text-only. Explicit
candidate feature and trusted cold load via existing Runtime slot;
engine mixed owners supported, facade encoder-only load. No automatic
qualified loader. Actual source-bound synthetic Rust SDK and controlled
mixed scheduling/lifecycle tests pass, 246/0/1 focused, strict lint.
Actual loopback embedding refusal verified. Assembled gate:
2,584/0/109 (321 Rust suites, 834s), complete isolated gate passed.

**OA-6C (27 Nov): capability-gated wire integration.** Original
registry captures explicit trusted EmbeddingEncoder public capability at
registration, default false. Native candidate remains ineligible.
Serving model list includes only original text and explicitly eligible
encoder aliases. Actual ingress admission rechecks gate and projected
width with original model reader. Existing service defaults refuse;
no new execution owner. Authenticated strict JSON fixed/chunked input:
string, string array, ID sequence or array of ID sequences (max 16 rows);
original Runtime/selected encoder limits still apply. Dimensions omitted,
null or exact fixed width; explicit unsupported reduction and non-null
user refusal. Float or base64 little-endian F32, ordered rows and exact
prompt/total usage. Original async receipt and cancellation/deadline,
shared bounded response writer and independent retained wire credit.
No partial successful projection, no HTTP-side math. Actual source-bound
synthetic BERT with test-only explicit public declaration and native-only
owner, real socket and installed OpenAI SDK 2.33.0 checks pass.
Focused 246/0/1 (26 suites), strict lint. Assembled gate:
2,585/0/109 (322 Rust suites, 575s), complete isolated gate passed. Built-in candidate remains unqualified for public
serving; no learned payload acquired.

**Remaining OA-6 acceptance:** authorize acquisition of the exact
pinned original MiniLM weights (~90.9 MB) and config/vocab/tokenizer/
module closure only. Compare actual IDs and trained vectors to
independent reference with source-specific limits and actual admission,
cancellation/retirement; bind qualified loader/catalog eligibility and
rerun official SDK with trained source. No synthetic fixture or protocol
pass is sufficient for trained-model acceptance. No automatic download,
fake vectors or second execution owner. OA-6 remains open.

### OA-7 — Audio

**OA-7A (27 Nov): original trained audio encoder.** OA-6 final MiniLM
acquisition not authorized; advance independent ready work on cached
complete google/gemma-4-E2B-it source. Explicit audio-reference feature,
original authenticated checkpoint/processor, source-shape/clip closure.
Shared CPU FFT, convolution, norm, softmax and source projection;
model-owned 16 kHz mono unpadded semicausal Hann magnitude log-mel,
two stride-2 subsampling convs and projection, complete 12-layer
USM Conformer with causal relative attention and light convolution,
learned 1536-dimensional language-space projection. No waveform
truncation/resampling, separate runtime or model-derived dummy text.
Finite source-bound feature/output owners and original resource
admission/cancellation; full original 10.25 GB source retained.
Actual cached learned source and independent installed PyTorch 2.11/
Transformers 5.5.4 F32 reference: 4/16 x1536 projected values, max
absolute 2.5749e-5/4.3988e-5, relL2 1.039e-6/1.594e-6; log-mel
4.77e-7 max abs. Reference explicitly boolean SDPA mask, original
blocked attention and padded/valid row equivalence. Cancellation and
result retirement pass. Focused 125/0/13, strict lint. Assembled:
2,587/0/110 (322 Rust suites, 570s), complete isolated gate passed. This is encoder/projector evidence, not ASR/TTS or
public eligibility.

**OA-7B (27 Nov): shared original audio/language execution.** Existing
verified source window owner shared between audio encoder and original
Gemma text, exact same issuer. Bounded complete scalar/conv parameters
retain original shape; no full resident copy or separate source.
Original audio placeholder span resolves PAD IDs for PLE lookup;
learned PLE projection uses merged audio row (not PAD projection).
Native F32 activation import and optional main embedding in same
family traversal, original StateRoot transaction/KV, ordinary cached
decode. Actual source-derived synthetic waveform + nine-position
prompt, full vocabulary prefill/continuation vs independent original
F32 Transformers: 524288 logits max abs 0.000320435/0.000189781,
relL2 0.000004470/0.000009288, exact argmax both. Combined retained
74,392,844 bytes; previous full audio-only 10,254,953,566. Active
cancellation and independent retirement, existing two audio cases
passed. Focused 222/0/14 strict lint. Assembled: 2,591/0/111 (322 Rust suites, 679s), complete isolated gate passed.
No trained speech accuracy, GPU parity or public eligibility.

**OA-7C (27 Nov): native transcription.** Explicit original source/
template/tokenizer, fixed supported instruction-first non-thinking
ASR turn, actual audio row count and BOA/EOA expansion. Bounded mono
16kHz PCM; deterministic greedy original stop IDs, complete private
prefill then one decode quantum per round. Same registry and finite
active queue as embeddings, shared generation FIFO/IDs/limits,
typed transcript receipt, usage/stop/length, charged input/output and
actual original decoder lifecycle/cancellation. Native-only speech
catalog; explicit Gemma4 CPU candidate, no default or public grant.
Independent original HF on local synthesized "Hello world" (12186
samples,19 audio rows,41 prompt IDs,3 generated incl stop); actual
native prompt exact match. Native learned: actual native SDK "Hello world", input=41, output=3 including stop, finish=Stop; 263.484s completion, full real test passed 292.62s (including cancellation/retirement).
Focused247/0/2 strict lint; assembled: 2,593/0/112 (324 Rust suites, 617s), complete isolated gate passed.
**OA-7D:** bounded multipart transcription and supported controls
through same native receipt with explicit eligibility, never fake
transcript. Natural speech/general quality qualification before built-in
production capability; translation/synthesis/streaming only real support.

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

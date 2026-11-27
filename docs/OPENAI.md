# Local OpenAI-compatible inference

ZeroGravity exposes a bounded, capability-gated subset of the OpenAI
inference API. Protocol adapters are not model capabilities: loading a text
model does not grant embeddings, speech or image generation. Discovery and
admission resolve the actual ready registration. No inference request
downloads a model or silently changes backend.

## Start a local server

Build with the `serve` feature and an appropriate native model/backend.
For an already available compatible GGUF:

```sh
zerogravity serve --model local=./model.gguf --listen 127.0.0.1:8000
```

The same runtime can be embedded through the Rust SDK's
`Runtime::server()` / `serving::ServerBuilder`. The CLI, SDK and local
control UI share the original model and resource owners. Cold
`serving::openai::Endpoint::require_implemented()` reports protocol-family
availability only; it does not establish a loaded model, listener or
qualified execution.

The inference listener is loopback-only. Set `ZEROGRAVITY_API_KEY` before
starting it to require bearer authentication. Control Web authentication is
separate. For untrusted remote access a separately secured proxy and
deployment policy are needed; this is not a hosted multi-tenant service.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-key",  # use the configured key when authentication is enabled
)
models = client.models.list()
response = client.responses.create(
    model="local", input="Explain resource ownership briefly.",
    store=False, temperature=0, max_output_tokens=64,
)
print(response.output_text)
```

## Supported surface

The scope below is the qualified subset as of the November 27, 2026
compatibility closeout. Wire support and learned-model qualification are
separate.

| Family | Routes and supported wire behavior | Actual execution / qualification |
| --- | --- | --- |
| Health | GET `/health` | Native diagnostic object, not an OpenAI object |
| Models | GET `/v1/models`, GET `/v1/models/{id}` | Ready served aliases; no remote or cache search |
| Responses | POST `/v1/responses`; text JSON/SSE, foreground and background, exact input count at POST `/v1/responses/input_tokens` | Original semantic text/template/receipt; actual cached Qwen SDK parity |
| Response resources | GET/DELETE `/v1/responses/{id}`; POST `.../cancel`; GET `.../input_items`; bounded background stream replay | Listener-local records; cancellation only for eligible background work |
| Conversations | POST `/v1/conversations`; GET/POST/DELETE `.../{id}`; GET/POST `.../{id}/items`; GET/DELETE `.../items/{item_id}` | Bounded text context, pagination and atomic terminal append; no retained decoder |
| Chat Completions | POST `/v1/chat/completions`; ordered text messages, JSON/SSE and optional final usage | Qualified text routes, actual SDK parity |
| Legacy Completions | POST `/v1/completions`; single raw text prompt, JSON/SSE and optional final usage | Same original text runtime |
| Embeddings | POST `/v1/embeddings`; string/strings or ID sequence/sequences; float or little-endian F32 base64 | Explicit eligible encoder adapters; source-bound synthetic BERT fixture + actual SDK; trained MiniLM parity and built-in public eligibility open |
| Transcriptions | POST `/v1/audio/transcriptions`; multipart WAV, JSON or text | Explicit eligible speech adapter; actual learned Gemma 4 E2B synthesized-utterance reference and SDK test; built-in candidate native-only |
| Translations | POST `/v1/audio/translations`; multipart WAV, English target JSON or text | Separately eligible translation adapter; actual learned Spanish-to-English synthesized-utterance reference and SDK test; built-in candidate native-only |
| Realtime | WebSocket `/v1/realtime?model={id}`; text session update, user item creation, response generation/deltas/done/cancel | Explicit eligible ordinary CPU Qwen text route; actual SDK/native parity and controlled lifecycle tests |
| Speech synthesis | POST `/v1/audio/speech` | Unsupported; no qualified learned synthesizer |
| Images | POST `/v1/images/generations`, `/edits`, `/variations` (under `/v1/images`) | Unsupported; no qualified learned image generator |
| Voice Realtime / WebRTC | Audio buffers, VAD, voice output, calls/client secrets, WebRTC | Unsupported; finite speech tasks are not a streaming voice pipeline |

### Text, resources and bounds

Ordinary supported text uses deterministic greedy execution and the
selected model's own tokenizer/template. Raw legacy completions and typed
chat input are distinct. Responses supports the accepted text items,
instructions, metadata, conversation and previous-response references.
Unsupported active sampling, structured output, tools, hosted tools,
reasoning controls, arbitrary media, logprobs and other unsupported
parameters refuse rather than being silently ignored. Neutral accepted
controls do not grant additional capabilities. Input counts and terminal
usage reflect the actual selected model; length-limited generation is
reported as incomplete or `finish_reason: "length"`, not a complete answer.

Protocol records are bounded by count, bytes, expiry and the original host
admission. They are listener-local, not durable model state or KV sessions.
`store: false` avoids ordinary foreground retention; a background operation
may need a bounded temporary polling/replay record. Conversations retain
only the accepted protocol items and append after the original terminal
receipt. Cancel, disconnect, listener shutdown and retirement use the
original request owner. Background work that is explicitly listener-owned
survives its creation connection and is joined at shutdown.

### Speech and Realtime

Finite speech accepts RIFF/WAVE PCM16 little-endian mono at 16 kHz, within
both transport and model limits. Only JSON or plain text is supported;
custom language/prompt, nonzero temperature, timestamps, diarization,
arbitrary codecs and streaming refuse. English translation returns only
the validated target text. JSON translation follows the SDK Translation
schema (text only); internal accounting retains the actual generated source,
target and stop tokens. A length terminal refuses as incomplete.

Text Realtime is a single connection-local conversation with bounded
history, one active response and explicit cancellation. Sessions select
one exact registered model; no silent retarget after replacement. Only
completed valid assistant turns enter the local history. Cancelled,
failed and length-limited turns do not append assistant history. Session
settings and output are text-only; this is not voice, VAD, session
resumption or a latency guarantee. See the SDK's `realtime.connect`
with the same local base URL and a qualified model alias.

### Ordinary serving benchmarks

Benchmark the same public listener rather than a separate execution
binary. With the already running server above:

~~~sh
python3 -B tools/serve-bench.py --endpoint http://127.0.0.1:8000 \
    --model local --request-kind responses --prompt "Say hello." \
    --requests 4 --concurrency 2 --max-tokens 8 \
    --provider-snapshot-url http://127.0.0.1:8000/health
~~~

Run from the repository root. The endpoint is the server origin (without
/v1). The existing ZEROGRAVITY_API_KEY environment variable is used when
set; --api-key-env selects another variable. The default request kind
remains chat. Choose responses for stateless single-assistant text
SSE, or completions for an explicitly supplied raw
prompt. Chat and Responses use the selected model's template. The
benchmark does not render a model-family template for raw Completions.

Use --workload FILE for an open-loop schedule rather than repeated
closed-loop prompts. For example:

~~~json
{"schema":"zerogravity.serving.workload.v1","requests":[
  {"request_id":"a","prompt":"Say hello.","arrival_seconds":0,"tenant":"group-a","max_tokens":8},
  {"request_id":"b","prompt":"Name a color.","arrival_seconds":0,"tenant":"group-b","max_tokens":8}
]}
~~~

The JSON report records the actual request kind, offered/dispatched/
completed requests, failure classification, output digest, finish reason,
and actual reported input, output, total and cached-input usage (unknown
counts remain null). A successful length-limited terminal counts as a
finished transport request, also reported separately; it is not a complete
semantic answer. Responses validates sequence, selected model, item and
response identities, terminal output and usage. Unsupported outputs such as
reasoning/tool/media items do not masquerade as a supported text sample.

Latency and throughput are end-to-end client observations. TTFT measures
time to first nonempty text delta (offered time, including open-loop client
backlog; service TTFT is also retained). SSE chunk intervals are not
individual token intervals. The optional health snapshots preserve actual
effective scheduler config, catalog and counters; they are raw provider
observations, not independent physical proof. In ZeroGravity, scheduler
attempted/completed count execution steps; use registration
performance.completed_requests for completed model requests. Workload
tenant labels are not authenticated server tenants.

The existing --return-token-ids extension is available only for explicitly
enabled legacy Chat/Completions diagnostics; Responses refuses that option.
Absent individual token observations do not create token ITL or qualified
goodput. The paired tools/serve-compare.py accepts the same --request-kind
for both providers, including warmups, and preserves that kind in its
report and workload-matrix summaries. It does not imply the other provider
implements that route or that a measured ratio establishes source/math/
hardware equivalence. Compare exact output/usage/finish and verify source,
tokenizer and execution separately. See --help on each tool.

### Compatibility and diagnostics

The existing listener bounds HTTP/1.1 bodies, headers, connections,
multipart parts, frames, output and lifetimes. JSON/SSE and Realtime
project the original typed receipts. Ordinary objects contain no
ZeroGravity-specific required fields; unsupported requests receive an
OpenAI-shaped error with a safe message, stable code and parameter when
available. Accepted HTTP requests have server-generated correlation IDs.
Existing opt-in legacy token metrics retain their explicit correlation
and observation contract; do not enable these extensions for ordinary
SDK interoperability.

`/health` and native control diagnostics expose ZeroGravity observations.
Trusted authenticated HTTP/SSE diagnostics and per-request token metrics
are explicit opt-ins (see [CLI error diagnostics](CLI.md#error-diagnostics)).
Realtime events retain their standard safe shape even when HTTP diagnostic
details are enabled; safe error metadata still reaches the operational
observer. Diagnostic error responses and operational events do not leak raw prompts,
credentials or arbitrary failure causes by default.

Qualification is against installed OpenAI Python SDK 2.33.0 and the
specific tested native sources/adapters, not every OpenAI parameter or
all SDK versions. The [official Responses wire examples](https://developers.openai.com/api/reference/resources/responses/methods/create)
include the boolean `store` field even though that SDK's generated Response
class omits it; the actual store contract remains explicit. The legacy
Completions SDK's strict schema excludes null nonterminal finish reasons;
ordinary streaming client behavior is qualified with explicit event
assertions. General natural-speech accuracy,
trained embeddings, image/voice capabilities and optimized physical
performance remain separate acceptance work.

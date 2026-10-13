# Runtime, APIs, developer experience and observability

**15 Oct 2026 — efficient serving UX and truthful metrics:** [Joint concurrency specification](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) requires three explicit capacity valves (API requests, queued prompt work/model/session budgets, active device step rows/scratch) with typed overload and no unbounded queue growth. One request binds exact model/quant/profile once; GPU-ready and root-committed outputs stream without per-token router lookup or hidden Python model-specific server. Report queue/admission/rejection fractions, committed tokens, TTFT/ITL/p95, cached state domains and actual saved recompute, draft accepted-vs-proposed, MoE expert/memory IO, energy and physical GPU completion. Claims of vLLM superiority require matching numeric/model/quant/hardware and loaded source; source-free demos do not qualify.


**15 October 2026 — NATURAL STORAGE MODE AND TRUTHFUL UX:** [Storage-neutral tensor residency](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) should be automatic: prefer full GPU/UMA residency when feasible and source/backend-qualified SSD expert/lookup streaming when a model cannot fit, without a second model/server/API alias. Report source identity, effective cache and mandatory memory, OS headroom, SSD hot/cold miss/copy budget, current residency ratio, source-compatible state snapshots and actual measured latency/QoS. Advanced controls only for cache bytes, IO policy, prefetch, user-selected capacity priority; no hidden quant, default SIGKILL or global page cache drop. A model may load from SSD but be too slow for live audio; state latency truthfully and refuse soft-realtime deadlines where appropriate. [SSD test matrix](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md).


**15 Oct extreme model serving precision addendum:** [extreme-specialization evidence/implementation source atlas](../research/SPECIALIZATION_SOURCE_ATLAS_2026-10-15.md) and [full specialization tests](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) reinforce one native Rust ModelRouter/TaskCoordinator/StateRoot, serving exact model/quant/sidecar and source tokenizer/think/tool template. Expose supported source mode, original-vs-derived quality, speculative accepted-only tokens, TTFT/latency/cold-start, model loading progress and physical OS headroom without claiming unknown success. A multi-session batch can alter near-tie greedy output through reduction-order changes: numeric capability must say so. Avoid dynamic solver search, graph compilation or global model catalog lookup on packet/token hot path; no new GLM/Qwen/MiMo Python proxy per family.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


The user expects llama.cpp-like simplicity despite a complex compiler and serving runtime. **Do not expose model kernel details as mandatory user flags.** Every public capability is either functional and tested or clearly experimental/unsupported.

## 1. Proposed executable and modes (illustrative, not finalized)

```sh
zerogravity run /models/model.gguf
zerogravity run hf:org/model --offline
zerogravity serve /models/model.gguf --host 127.0.0.1 --port 8080
zerogravity inspect /models/model.gguf --explain-plan
zerogravity bench /models/model.gguf --scenario single-ar
zerogravity devices
zerogravity models list
```

Auto plan selects backend/device, quant/layout, safe memory reservation, default context and supported token budget from model/hardware capabilities. CLI should print approximate model weight bytes, resident/SSD table bytes, context/recurrence state, scratch headroom and expected fallback reasons. Do not silently choose lower quant when requested weight file won't fit.

## 2. Rust library/SDK layer

Engine shared by CLI, HTTP and embedding in other Rust applications. Suggested API objects: `ModelRegistry`, `ModelHandle`, `Engine`, `Session`, `Prompt`, `GenerationOptions`, `TokenStream`, `ExecutionReport`, `BenchmarkRunner`. Engine owns devices, memory, plan registry and async executor. Sessions are mutable timelines; model handles immutable weights. Typestate or guarded APIs prevent using session after close or stale snapshot. No caller gets a raw device pointer by default.

## 3. HTTP compatibility matrix

Initial proposed endpoints:
- `GET /v1/models`: loaded model IDs, true capability and context.
- `POST /v1/chat/completions`: model-native template, tools, streaming.
- `POST /v1/completions`: raw prefix/text completion.
- `POST /v1/responses`: Responses-style continuation and tool history, if conformance complete.
- `POST /v1/messages`: Anthropic-style messages, if conformance complete.
- health/ready/metrics endpoints with policy-based information exposure.

OpenAI/Anthropic compatibility does not imply one text renderer for every model. Model adapter handles native tools, role sentinels and hidden reasoning. Traces must distinguish visible output from model-internal reasoning and prevent accidental disclosure of hidden state through APIs. Structured output grammar/JSON mode must not silently change sampling if unsupported.

Authentication, TLS and rate limits need safe defaults; local bind `127.0.0.1`, remote `0.0.0.0` explicit with warning/auth requirement. Requests cannot pass arbitrary local file path or remote image URL without explicit access policy and bounded fetch.

## 4. Streaming correctness

Stream only **committed** model output, not speculative draft proposals. Cancellation/stop must reconcile state frontier and output budget. Slow client streams have bounded queue and backpressure; never allow unlimited GPU buffer accumulation. Stop sequences can span token boundaries, so hold back appropriate text until detection; do not retract already streamed bytes. UTF-8 output can cross token boundaries, so incremental decode must handle incomplete codepoints.

Include deterministic request ID, position, terminal reason (eos/stop/length/cancel/failure), usage counters distinguishing prompt, cached and generated output, and speculative acceptance/latency in diagnostics (not necessarily every API response).

## 5. Tool and agent histories

Tool call delimiters and argument encoding must come from model-native template; use an intermediate structured message representation with opaque rendered token history. Reusing cached prefix based on only visible text can fail if previous hidden reasoning or tool IDs have different exact rendered tokens. Parse tool calls incrementally and validate escaping, invalid nesting, very large arrays/maps and unclosed blocks.

Agents may send same long system/tools repository context each turn; full hybrid prefix snapshots should avoid re-prefill where compatible. Preserve structured roles and consistent parent/branch session IDs. Optional persistent cache contains private conversations, so explicit user configuration and safe permissions are required.

## 6. Context and memory fit UI

`inspect --explain-plan` should show:
```text
artifact SHA / tokenizer / family / quant
device and backend, driver, architecture target
current memory: physical available, GPU addressable, OS reserve
weights: dense, shared, expert resident, expert backing storage
context: KV pages, GDN/KDA/indexer, snapshots, scratch per session
selected prefill/decode/verify kernel regions and numeric contract
speculative mode: none/MTP/other, exact sampled or approx opt-in
fallbacks and why (missing kernel, shape range, precision, memory)
maximum safe context and concurrency (estimated, not guaranteed)
```

On insufficient memory, explain specific bytes short, safe options (smaller context/batch/model/quant, SSD streaming if actually supported), and never automatically mutate system boot parameters or require root.

## 7. Multi-model and hot reload

A single engine can enumerate models and load one or several only if budgeted. Hot reload creates new immutable ModelHandle/plan and leaves old sessions bound to old model identity until retired or explicitly migrated by replay. Sessions cannot transparently reuse incompatible weights or tokenizer. Device graph/model unload waits on outstanding leases and events.

## 8. Metrics and telemetry

Core counters:
- queue, TTFT, prefill and decode duration, per-token latency P50/P95/P99;
- token counts by input/cached/output/draft/accepted/rejected;
- model/plan/backend/quant and context frontier; can be redacted or aggregated;
- expert cache hits/misses, unique routed experts and IO critical path;
- resident/pinned/virtual physical allocations and OOM/admission rejects;
- GPU execution vs idle, graph capture/launch count, kernel breakdown in debug;
- prefix reused tokens, COW snapshot size, persistent cache costs;
- errors/cancellations/restarts and distributed sync status.

Prompts, model personal data and tool outputs not sent to telemetry by default. Error messages must avoid dumping secrets and credentials. Profiling instrumentation changes performance; benchmark uninstrumented also.

## 9. Configuration hierarchy

Priority proposed: explicit request options > CLI flags > validated config file > environment for developer diagnostics > safe hardware/device defaults. Do not let undocumented environment flags change sampling/quality silently. Show effective configuration and explain any ignored option. Use versioned config schema with migrations; secrets handled separately from ordinary config and excluded from logs.

Power/perf modes: `interactive`, `throughput`, `balanced`, `memory-safe`, `quality-reference` (working names). Modes choose scheduling/resource strategy, not silent model quantization.

## 10. Local deployment and packaging

Portable CPU build and conditional GPU features, no Python required at serving runtime. Rust binaries and optional precompiled architecture-specific kernels cached with checksums; offline installation and deterministic reproduction. macOS Metal, Windows and NixOS packaging remain open priority; include artifact availability matrix and clear unsupported hardware errors.

NixOS considerations: GPU driver dynamic libraries/ld paths, system `/dev/kfd`, render group, ROCm version, glibc/LLVM linking. No assumption that conventional Ubuntu install instructions work on NixOS. A dev shell may supply build tools, but system drivers must remain host controlled.

## 11. Testing API usability

Run official API SDK smoke tests, malformed JSON/tokenizer error recovery, concurrency, very long tool histories, hidden/visible reasoning, image/text order and fingerprint, Unicode incremental output, network timeout, cancellation, server restart, old/new model hot reload, user-local bind default, auth and metrics. Agent integration needs real Pi/OpenCode/Codex or compatible request replay, but no embedded agent tool should access filesystem/network implicitly.

## 12. Pending policy

Executable flags/naming, stable Rust SDK/ABI, external authentication solution, default disk cache and retention, third-party plugin model schemas, logging redaction, default model selection and release channels remain open. Avoid treating CLI examples as approved final UX.

## 2026-10-10 integrated v2 addendum — concrete product and event contracts

Adopt research's precise serving backbone from `ENGINE_CORE_EXECUTION_AND_PROTOCOL.md`: frontends translate OpenAI/Anthropic/native requests into one canonical `InferRequest`; `EngineCore` is the single mutable owner; `EngineMsg/EngineEvent` and worker `StepPlan/WorkerDelta` channels provide asynchronous execution. Workers own device code/streams and only emit provisional outputs; stream clients see **committed** tokens after exact state/sampler checks. A `ModeReceipt` can be returned in diagnostics/metrics so true kernel tier, eager/capture, quant/graph, spec sampling fidelity and fallback are visible.

Research convenience features are part of the **product plan**: GGUF tokenizer pretypes, Jinja templates, a grammar library/JSON-schema grammar FSM, streamed family-specific tool parsers, content-addressed ModelStore with atomic pull/list/rm/GC/offline, model conversion/quantization/imatrix/optional acceleration pack, `/props`/doctor, and companion `run`/`serve`/Rust SDK. These are distinct from executing tools or MCP access: tool invocation requires a separate explicit authorization/sandbox system and is not promised by token generation.

The external API must distinguish task runners: text generation with KV/sampling; embeddings with encoder+PoolingSpec/Matryoshka normalization; rerank/score classification over paired inputs; multimodal encoder/projector with separate cost and span handling. LoRA adapters include request-specific generation in cache keys; unsupported methods like `best_of`/multi-sample are typed refusals until tested. `Responses`, `Anthropic` and tool schema fidelity must be implemented by conformance, not endpoint names alone.

**Mode/profile naming and version remain OPEN**, despite historical research selecting `zg`. Keep simple user-facing automatic plans with `inspect --explain-plan` and refusal naming one actionable adjustment. No startup user needs to know attention tile/WMMAs; no silent new quant or target sampler change. Do not accidentally expose hidden reasoning or private snapshot bytes through status endpoints or telemetry.

Integrate `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md` for model store/tokenizers/grammar/LoRA/embeddings and `ARCHITECTURE_RECONCILIATION_AND_AUTHORITY.md` for ownership and Rust-source policy.


## 2026-10-12 integration — NPU diagnostics and future media outputs

Device inspection should expose XDNA/NPU architecture, XRT presence, kernel/program compatibility, shared physical memory, and safety qualification state (unsupported, unqualified, experimental opt-in, qualified). **No automatic privileged change** to FCLK, power governor, runtime suspend, IOMMU or systemd. GPU-only works without XRT. ModeReceipt names GPU/NPU projection split, compute-encoding precision and fallback reason.

TaskProgram supports planned distinct client outputs for ASR transcript/audio timestamp, TTS PCM/audio frames, images/image editing, video frames and embedding/rerank scores, all under the existing request authorization and DeviceWorker/resource interfaces. An upstream Gufo model pipeline is research inspiration, not an already supported endpoint in zerogravity. Read [PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md](PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md).

## v4 refinement — truthful task streams and no hidden task workflow executor

One canonical TaskRequest maps protocol and SDK inputs to a built-in TaskDriver; it runs under the same coordinator/device/resource execution spine. Media and score outputs are typed independently of token streams. Session state is **published once after all domains Prepare**; only then may result/stream become visible. Live transport guarantees ordered **delivery attempts** with output_seq, not exactly-once remote receipt; optional authenticated cursor replay with retention/backpressure is a separate product feature. A slow client cannot indefinitely pin GPU KV or unbounded PCM/video buffers.

A TaskDriver cannot execute arbitrary filesystem/network/tool actions from model outputs; such external actions belong to an application-level authorized/sandboxed service. `inspect` and ModeReceipt should report whether a GPU-native step loop, host-controlled loop, compiled-region fast path or shape/precision/power safety fallback actually ran, without leaking prompt/media/private state. The mere presence of GPU/NPU/encoder data does not mean it is supported or performance-qualified.

## v4.3 authoritative multi-model service and zero-config UX refinement

**The current authoritative router/UX contracts** are [multi-model residency and routing](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) and [one-command CLI/config/OpenAI API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md). Earlier `serve /model.gguf`, `config precedence`, standalone service and `OpenAI compatible` sketches in this document are historical supporting notes where they differ from v4.3.

**User-visible default:** `zerogravity run <model>` starts local interactive/single-request model processing directly in the same Rust engine **without a mandatory service**; `zerogravity serve` runs one loopback model-aware router+API over that same engine. Models list/pull/ps/inspect, per-model typed profiles and an effective-config explanation are progressively discoverable. A friendly alias resolves exactly to model bytes/quant/adapter/template/profile revision; a no-resident local model can on-demand load subject to real ResourceArbiter admission. No surprise registry download, public bind, model/quant substitute, remote peer, daemon or system setting changes.

**Multiple models/replicas:** catalog vs resident deployment vs active TaskSession different ownership; one model sharded across peers is one deployment. Preserve loaded state and sticky KV reuse where compatible. Replica load balance and model-choice `auto` policy are distinct, the latter explicit only. Model keepwarm/pin/idle unload never frees live DMA/unsnapshotable session; source model/alias hot reload affects only future bindings. Router isn't a token scheduler or another HTTP inference proxy.

**API:** qualified subset of `GET /v1/models`, `POST /v1/chat/completions`, embeddings with real encoder, later completions and versioned stateless Responses; typed SSE events, usage, roles/tools data and errors verified with official SDKs. Responses does NOT automatically provide cloud web search/file search/store or tool execution. Native model/admin control separate and restricted; local loopback by default, external bind requires explicit authenticated/TLS policy. Versioned user config with global/model/alias/CLI/request scopes, hard security/numeric/memory overrides and `config show --effective`; sample flags and port still proposed, not finalized.

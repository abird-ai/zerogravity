# Research Track G — Real-Time and Multimodal Serving / API Prior Art and Design Reconciliation

**15 October 2026; source/architecture research, no Zerogravity network/API execution.**  
**Focus:** OpenAI Realtime/Responses/audio/image/video, hosted live multimodal APIs, local vLLM/SGLang patterns, WebRTC/WS/SIP, hosted voice/media providers and provider job adapters.  
**Governing proposed spec:** .agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md.

## 1. Executive verdict

A high-performance universal inference engine must not expose every task solely through /v1/chat/completions. Text generation, asynchronous image/video, file ASR, streaming TTS, native duplex speech and typed decisions have different request, stream, cancellation, finality, cost and state lifecycles.

The right implementation is a **canonical TaskRequest and typed MediaEvent** beneath multiple outward protocol adapters, connected to one ModelRouter/TaskCoordinator/ResourceArbiter. Each adapter translates only verified observable semantics, never creates its own model engine.

## 2. OpenAI public realtime model/protocol reality

At review time, OpenAI's public API model catalog (https://developers.openai.com/api/docs/models) lists GPT-Realtime-2.1, GPT-Live and additional realtime/transcription models. The user supplied example "gpt-realtime-2.5" could not be independently verified as an official published model ID; treat as desired future/realtime capability class, **not** a known architecture or immutable API name.

Official realtime docs: https://developers.openai.com/api/docs/guides/realtime-conversations
- WebRTC and SIP flows may have a server-side playback/output buffer whose interrupted audio can be truncated automatically.
- WebSocket clients manage their own audio playback; they must stop it and send an explicit conversation.item.truncate with actual played duration to reconcile the conversation.
- With VAD, speech_start may cancel current response; client should not confuse generated audio, sent audio and actually heard audio.
- A canceled response can still have earlier committed text/audio, and tools may need independent cancellation/late result policy.
- Session and response events have explicit event IDs, item/content indices and generation boundary; these cannot be represented as raw token strings.

Official client-event schema: https://developers.openai.com/api/reference/resources/realtime/client-events
- conversation.item.truncate is an observable protocol contract, NOT an instruction for GPU DeviceBackend to roll back physical DMA.
- Truncation of unplayed speech transcript is a **presentation/history** policy and should not physically pretend a user heard earlier generated words.
- Unsupported events or fields must result in explicit capability refusal, rather than fabricated transparent compatibility.

**Lesson:** One session needs neural committed root AND transport/presentation playback ledger, plus source-defined state replay for barge-in.

## 3. Hosted live multimodal API reference

Provider documentation:
- Hosted live API capability documentation.
- Hosted live API session management documentation.
- Hosted live API best-practices documentation.
- Hosted live API stream reference.

A public hosted live API streams audio/video/text on a persistent WebSocket, advertises VAD interruption, session resumption handles, periodic connection GoAway and context compression. The public documentation reviewed 2026-10-15 illustrates a hosted live model variant; exact model IDs, session limits and capabilities vary by date/model and must be obtained from provider metadata rather than compiled into Zerogravity.

Live media behavior highlights:
- Audio chunks can be 16kHz PCM inbound, 24kHz PCM outbound depending on selected model/API; do NOT hardcode as universal media clocks.
- Video capture frame rate may be throttled/model-limited, distinct from the model's semantic video-processing clock.
- Asynchronous function calls may be permitted by some models and disallowed/serialized by others; model capability manifests must distinguish this.
- Session resumption and content compression preserve conversation under certain provider rules, but not guaranteed lossless neural local StateRoot migration.
- Multiple modality streams have independent ordering; timestamp and causal ordering must be explicitly handled.

**Lesson:** A protocol adapter can implement hosted-live-API-like session semantics over canonical MediaEvents without altering the native TaskDriver or TaskCoordinator.

## 4. Native streaming audio prior art

An independent full-duplex audio project provides native full-duplex audio and a streaming neural codec. It has source Python-framework code, an Apple-silicon framework and a Rust implementation, demonstrating that audio decoding and playback often require a strongly typed codec frame/time state, not conventional next-token output. A correct native backend must keep simultaneous input and output lanes and codec state. Its Rust source is useful prior art, not a mandate to embed a second independent model-serving runtime in Zerogravity.

**Lesson:** Real-time endpoint architecture must support source-native duplex models **and** composed Whisper→LLM→TTS under the same session transport.

## 5. Hosted voice streaming

A hosted WebSocket TTS input can accumulate text and use chunk_length_schedule and flush; those settings trade text buffering/latency against output. Some provider behavior responds to end-of-turn and inactivity; use provider adapter version, not universal client assumption. Word-level alignment may be available for certain outputs. The API provider is opaque; source model architecture and native GPU execution cannot be inferred from WebSocket frame shape.

**Lesson:** A TaskServiceAdapter must map provider-specific text flush, generated audio chunks, voice settings and stop/usage events into canonical committed/delivery events with receipt. Remote cancellation may be best-effort; never assume dropped socket means billable job stopped.

## 6. Hosted long-running media jobs

A hosted video generation service uses a long-running generate_videos operation and explicit polling until done, then asset download. Hosted async image/generative APIs return a job ID/status URL and later asset URLs; cancellation and rate handling exist as separate stages.

**Serving implication:**
- JobStarted is not ImageCommitted or VideoPlayable.
- A network 504 after POST can mean remote service accepted job; status Unknown must be reconciled with provider operation/idempotency key before reissuing potentially billable work.
- Output assets need bounded signed download, MIME/size/codec integrity checks, license and expiry tracking.
- Remote provenance includes provider model alias/version and charged cost if reported; no local GPU mode receipt.
- User-required rights/content credentials should be represented where actually provided, not fabricated.

## 7. vLLM and SGLang useful patterns and caveats

Relevant public docs (version-specific):
- vLLM documentation: https://docs.vllm.ai/
- SGLang documentation: https://docs.sglang.ai/
- llama.cpp HTTP serving/prior art: https://github.com/ggml-org/llama.cpp

Research uses them for architectural principles rather than claiming Zerogravity's feature parity:
- continuous batching and paged state management, prefix caching, model-aware scheduling;
- adapters/processor pipelines for multimodal image/audio inputs;
- source-specific compile/cache policies and asynchronous local HTTP streaming;
- graph capture/fused-kernel fast path while keeping model/session state separate;
- distinctions between inference engine capability and HTTP API compatibility.

**What not to copy:** hidden Python engine wrappers, per-model HTTP subprocess, mandatory user flags for every kernel, one global GPU lock per PCM packet, blindly interpreted model blobs, conflation of text token streaming with playable audio. "vLLM supports this model" does NOT prove it exposes full-duplex speech through its standard endpoint.

## 8. Canonical API compatibility and behavior matrix

| Capability | Native model runner needs | External API needs | Failure mode to test |
|---|---|---|---|
| Text/completions | tokenizer/template, exact samples/grammar state | streaming SSE/HTTP, source model alias | unsupported tools/JSON silently ignored |
| Embeddings/rerank | source encoder+pool normalization | vector size, batch schema, errors | generic decoder logits used as embeddings |
| File STT | source frontend+decoder+clock/diarization | files, timestamps, translation langs | partial text mislabeled final |
| Realtime STT | cached encoder/transducer, VAD, interim/final | WS/WebRTC, gap/revision events | out-of-order segments, fake latency |
| TTS | acoustic+vocoder and complete frame | binary chunk/format, voice ID, playback status | incomplete RVQ group emitted as playable |
| Duplex voice | independent capture/generation states | live session handshake, barge-in, ACK | output cancel corrupts input/other session |
| Images/edit | verified predictor/solver/VAE | async/sync, input image masks, output URI | invalid mask/color/schedule silently changed |
| Video/AV | temporal latents, 3D VAE/codec, AV sync | async job status, assets, progress | raw incomplete latents delivered as final |
| Decisions | schema-conditioned head raw logits | typed choice/noul/score | made-up calibrated probability |
| Hosted provider | NONE if no weights/source | consented TaskServiceAdapter | vendor host mistaken for local GPU |

**Compatibility receipt:** protocol/profile revision, actual task mode, model artifact or remote provider identity, requested and effective parameters, input/output modalities, supported stream events, error policy, numerical/quality source evidence, latency-qualified mode (if any).

## 9. Transport-specific lifecycle templates

### HTTP file ASR
Receive bounded file → sniff/check MIME, decode sample clock → admit verified ASR model → process windows → commit final transcript/timestamps → return response. Large files may need async job; do not keep file bytes indefinitely in memory.

### HTTP SSE text
Admit exact model/session → committed token/text events from TaskCoordinator → SSE adapter buffers at most configured small window → stop/length/cancel terminal frame. UTF-8/stop sequences can span token boundaries; hold back uncommitted pieces.

### WebSocket live voice
Handshake auth + source model/codec capability → choose input/output lanes → binary audio frames and control events with independent sequence numbers → VAD/ASR native TaskDriver → source-valid model output → complete codec chunk → bounded playback buffer → client ACK/truncate on barge-in → reconcile turn epoch and conversation.

### WebRTC live voice/video
Secure ephemeral session/negotiation plus ICE/transport setup → RTP audio/video; data channel for reliable events → transport jitter/RTCP feedback maps into DeliverySession only → model TaskDriver receives exact timed media input and response. No direct RTP socket manipulation of neural StateRoot or GPU BufferLease.

### Async hosted video
Authenticate/authorize egress, license and cost → create provider job with idempotency key → persist provider receipt/job status → poll or verified webhook → validate output asset → canonical committed job result/reference → client downloads within TTL. Retry only after reconcile Unknown.

## 10. Security, rate, billing and privacy research learnings

- Remote bind defaults to safe loopback; public interface must require appropriate auth/TLS, scope and rate limiting.
- Binary media decompress/parse under bounded input size, time, resolution, frame count and entropy ratio; beware zip bombs and forged codecs.
- No arbitrary local file path or media URL fetch based solely on model prompt; user/admin supplies explicit allowed paths/domains/credentials.
- Separate local/inference resource quotas and external provider cost/rate limits. API request budget does not give permission to upload private speech/face to vendor.
- Voice cloning/ref audio requires verified consent and source-voice access policy; log hashes/metadata not raw voice samples by default.
- Provider secrets never embedded in checkpoint, generic task IR, diagnostics or client/browser payload; ephemeral tokens for allowed client direct provider APIs.
- Tenant session state/decoder/acoustic/attention caches isolated; immutable source weights may be shared only with license/provenance checks.
- Reconnect tokens expire and are tenant-bound; playback ACK isn't trusted evidence of human hearing.
- Observability redacts private prompts, source image pixels, raw audio and video unless explicit opt-in.

## 11. Minimality and performance learnings

To avoid multiplying complexity:
- one canonical TaskRequest and ModelBinding/ModeReceipt;
- one actual TaskCoordinator and physically qualified worker/arbiter;
- protocol adapters in separate small Rust modules sharing the same SessionGateway;
- one MediaEvent schema for native and provider outputs with explicit source category;
- per-device/prepared-bundle hot path with no catalog lookup, JSON serialization or global snapshot allocation per packet;
- bounded live ring buffers and credits to prevent GPUs outrunning playback while memory grows unbounded;
- model-specific TaskDriver handles source math; API transport merely moves verified values.

A "zero-copy" claim should be measured at the actual codec/PCI/network boundaries; a reference-counted BufferLease may still require one conversion or copy for transport security/format. Avoid forcing DMA buffers to outlive malicious or broken clients.

## 12. Adversarial serving acceptance

**MM-SRV-01:** official OpenAI SDK short text, streaming, tools subset; unsupported fields explicit.  
**MM-SRV-02:** real-time WebSocket playback ACK is behind delivered, then barge-in truncates only unheard audio and matching transcript.  
**MM-SRV-03:** WebRTC RTP clock conversion to source PCM/video timestamp exact across frame jitter; client interruption/RTCP cannot commit model state.  
**MM-SRV-04:** hosted-live-API-like session reconnect GoAway and output stream replay with client resume cursor; no invented full neural migration.  
**MM-SRV-05:** hosted voice-provider incremental text chunks require source flush/end; model variant doesn't silently change speaker/voice.  
**MM-SRV-06:** hosted video provider accepted POST, network timeout, retry/reconcile does not create duplicate external billable job.  
**MM-SRV-07:** hosted async image job callbacks validate host/signature/TTL; wrong asset MIME rejected.  
**MM-SRV-08:** two sessions on same PCI GPU, one barge-in does NOT advance card-wide physical epoch; other completes.  
**MM-SRV-09:** slow client and 5 video jobs leave bounded PCM/video packet buffers; fair audio response under declared SLO or typed admission miss.  
**MM-SRV-10:** no local DeviceBackend receives proprietary provider raw credential or claims opaque model GPU execution.  
**MM-SRV-11:** versioned native schema decision endpoint consumes finite options, returns raw logits/provenance not generated JSON heuristics.  
**MM-SRV-12:** model hot reload/alias update cannot alter active Session model revision without legal migration.

## 13. Implementation recommendation and scope

First implement source-free MediaEvent, SessionGateway, rational clocks, authoritative vs delivery frontiers and explicit capability matrix. Then protocol adapters and tests. No new model inference engine, no uncontrolled media pipeline DSL.

Prioritize:
1. text + file ASR/TTS adapters sharing EngineCore;
2. WS true duplex with source-free duplex mocked TaskDriver and playback cancellation;
3. WebRTC live media once clock/jitter/egress security model passes adversarial tests;
4. async image/video/hosted adapters, provider consent and cost;
5. high-concurrency workloads with real checkpoint/hardware QoS evidence.

Protocol support is separately qualified from native model support and from current provider marketing.

## 14. External sources / dates

- OpenAI official catalog and model status: https://developers.openai.com/api/docs/models (reviewed 2026-10-15)
- OpenAI Realtime conversations/truncation: https://developers.openai.com/api/docs/guides/realtime-conversations
- OpenAI event schema: https://developers.openai.com/api/reference/resources/realtime/client-events
- Hosted live API capability documentation.
- Hosted live API session management documentation.
- Hosted live API stream ordering documentation.
- Native streaming audio prior-art repository.
- Hosted real-time TTS provider documentation.
- Hosted long-running video provider operation documentation.
- Hosted asynchronous media provider documentation.
- vLLM docs: https://docs.vllm.ai/
- SGLang docs: https://docs.sglang.ai/
- llama.cpp: https://github.com/ggml-org/llama.cpp

The research is about API-observable semantics, not proof of vendor model weights or internal implementation. No service login/test calls were made during this design pass.

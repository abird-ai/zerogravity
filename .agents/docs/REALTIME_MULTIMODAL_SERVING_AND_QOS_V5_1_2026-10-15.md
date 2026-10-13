# Zerogravity Real-Time Multimodal Serving, Sessions, QoS and API Contract — v5.1

**15 Oct 2026 — real-time concurrent-goodput scheduling:** [Joint serving and three-level admission contract](JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) extends global scheduling beyond text: ASR/TTS/duplex/audio codec frames, video solver steps and typed decision batches keep independent clocks and atomic output readiness while sharing physical resources, credits, placement and admission. Interactive text ITL and first-audible/packet jitter have declared **soft** QoS, not promises of hard real-time. A video job yields only at source-valid solver boundaries; live barge-in revokes one turn generation, not the GPU card epoch. No text tokens/s summation with acoustic/video units in goodput; per-class SLO/attempt/rejection/fairness measured separately. SERVOPT-31/36 are planned source-real mixed-workload qualification.


**15 October 2026 — STORAGE TIER MUST RESPECT SOFT REALTIME:** [Same source TensorView with GPU/RAM/SSD backing](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) is a legal serving placement mechanism, not proof that random SSD misses can satisfy duplex audio or synchronized AV frame deadlines. QoS admission evaluates complete ModelAssembly state, codec, median/p95 cold-cache miss and physical staging/OS headroom; opt for resident model components or typed refusal/explicit weaker policy when latency cannot be met. Media delivery/ack/playback ledgers remain separate from immutable source cache and committed neural StateRoot; no SSD IO task can publish media finality. Resident audio hot path has no per-frame host IO callbacks. [SSD-30/31/32 acceptance](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md).


**15 October 2026 • DESIGN ONLY, no protocol implementation.**  
**Governing constraints:** single Rust EngineCore and model execution fabric; TaskCoordinator is sole logical model StateRoot publisher, physical device leases retire only after genuine completion, network transports never mutate neural state or silently substitute models.  
**Companions:** UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md, VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md, SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md, MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md.

## 0. Scope and goals

Support local Rust CLI/SDK, offline files, streaming HTTP, OpenAI/Anthropic-compatible **tested subsets**, WebSocket and WebRTC live conversations, optional telephony SIP gateway, image/video async jobs and non-token decision endpoints with ONE canonical execution/admission/result contract.

No assumption that HTTP protocol compatibility proves native model math equivalence; no model inference work or binary media serialization inside a transport's JSON encoder. A source-verified Moshi/Omni full-duplex model and a composed Whisper→Qwen→Kokoro assistant share the same task/session/transport infrastructure but their TaskDrivers remain distinct.

## 1. Serving decomposition without a second inference runtime

~~~text
User / SDK / CLI / Remote authorized client
          |
HTTP / SSE / WebSocket / WebRTC / optional SIP or native Rust callbacks
          |
ProtocolAdapter: auth, request/media limits, capability matching
          |
SessionGateway: session id, transport state, input clocks, resumability
          |
ModelRouter: model alias+revision, replica, allowed provider, requested QoS
          |
AdmissionManager: task cost, physical/card memory, buffer/queue credits
          |
EngineCore -> trusted TaskDriver -> prepared ExecutableBundle -> DeviceWorker(s)
          |
Prepared domains -> TaskCoordinator PublishRoot once -> CommittedMediaEvent
          |
DeliverySession: packetization, delivery/ACK/playback tracking, bounded credits
          |
Same input transport(s) or stored job output
~~~

Protocol adapters are stateless or lightly stateful formatting/handshake layers. They do NOT allocate a separate model server or schedule kernels. Request's model binding, source revision, processor/codec identities, task semantics and numeric policy become immutable for its current session epoch.

## 2. Canonical TaskRequest and response mode

~~~rust
struct TaskRequest {
    request_id: RequestId, tenant: TenantId,
    mode: TaskMode,                              // exact task kind, not a string heuristic
    model_binding: ResolvedModelBinding,         // exact artifact/assembly/profile revisions
    inputs: Vec<TimedInputPort>,                 // typed clocked values, security/size guards
    output: OutputContract,                      // required types, clock/codec/formats/finality
    qos: WorkloadQoS,                             // requested soft deadlines/admission mode
    authz: PermissionScope,                      // local/remote/provider and data egress
    lineage: Option<SessionCheckpointRef>,      // authenticated compatible state only
}
enum TaskMode {
    Text, Embed, Rerank, DecisionSchema, ASR, TTS, Image, Video,
    NativeAudioDuplex, NativeAudioVisual, CompoundTask,
}
struct OutputContract {
    modalities: Vec<OutputPortContract>,
    partial: PartialPolicy,
    ordering: OrderingPolicy,
    error_mode: StructuredErrorPolicy,
    delivery: DeliveryGuarantee,
}
~~~

**Per-task model support is not implied by a family name.** The same RegisteredModel can advertise text-in/audio-out and image-in/text-out while refusing unsupported video-out. Compatibility adapters MUST reject or explicitly report unsupported parameters (e.g. voices, sampling, duration, frame rate, JSON mode, beam, LoRA, response modalities, output format). No ignore-and-proceed.

## 3. RealtimeSession and transport state

~~~rust
struct RealtimeSession {
    id: SessionId,
    binding: ResolvedModelBinding,                  // pinned immutable source
    task: TaskDriverId,
    neural_root: StateRootId,                       // owned exclusively by TaskCoordinator
    transport_epoch: TransportEpoch,
    turn_epoch: TurnEpoch,
    lanes: Vec<MediaLane>,                          // capture, perception, response, etc.
    transport: DeliverySessionId,                   // independent from neural state
    qos: EffectiveQoS,                              // admitted soft SLO, not magical guarantee
}
struct MediaLane {
    id: LaneId, kind: LaneKind, timebase: RationalRate,
    capture_cursor: Option<MediaCursor>,
    commit_cursor: MediaCursor,                    // model-valid state
    delivery_cursor: Option<MediaCursor>,
    acknowledged_cursor: Option<MediaCursor>,
    playback_cursor: Option<MediaCursor>,
}
~~~

**Separate scopes of epoch:**
- PhysicalDeviceEpoch: globally invalidates all queued work on one physical card after driver failure/quiescence transition. NEVER advance this for a normal one-session barge-in.
- SessionEpoch: invalidates all old TaskDriver outputs in one session, preserving unrelated sessions.
- TurnEpoch/OutputGeneration: invalidates only a reply's future uncommitted playback after barge-in; other session input lanes may continue.
- TransportEpoch: reconnect/renegotiation generation; MUST NOT accidentally reset Neural StateRoot or grant replay of old private data.
- ModelGeneration: model/adapter hot reload pinned at request/session admission; old active sessions remain on original immutable revision unless trusted replay/migration.

## 4. Five distinct frontiers

| Frontier | Authoritative owner | What is guaranteed | What it does not guarantee |
|---|---|---|---|
| Prepared | device/codec provisional work | candidate buffers exist | source-valid, ready, committed or delivered |
| Committed | TaskCoordinator / committed root | exact task/model criteria passed; output eligible | bytes transferred or played |
| Delivered | DeliverySession transport | bytes handed to transport or accepted for transmission | client actually received or listened |
| Acknowledged | verified client ACK | client reports receiving/buffering to cursor | user actually heard or viewed |
| Played/presented | client/player signal where available | client claims media playback/display progress | guaranteed human attention; server cannot independently prove it |

Capture/input ingestion has analogous received/decoded/model-consumed frontiers. ACKs may be absent, delayed, reordered or untrusted; keep the reported watermark provenance and never infer played=delivered. No false exactly-once delivery promise over network partitions. Logical StateRoot publication is **not rolled back** merely because a client went offline; conversational presentation history is reconciled through a separate compatible policy.

## 5. MediaEvent stream

~~~rust
struct MediaEvent {
    version: SchemaVersion,
    session: SessionId, request: RequestId, turn: TurnId,
    event_seq: u64, lane: LaneId, generation: OutputGeneration,
    clock: ClockDomainId,
    capture_pts: Option<MediaTimestamp>,
    presentation_pts: Option<MediaTimestamp>,
    duration: Option<MediaDuration>,
    commit_group: CommitGroupId,
    phase: EventPhase,                          // provisional/update/committed/retracted/terminal
    payload: PayloadRef,                        // immutable leased bytes/URI, bounded
    media: ValueDescriptor,
    mode_receipt: CapabilityAndEvidenceReceipt,
}
~~~

Recommended event variants: SessionReady, InputAccepted, InputGap, ActivityStarted/Ended, TranscriptInterim(revision,parent), TranscriptFinal, DecisionCommitted, TextDelta/Final, AudioFramePrepared/Committed, VideoFrameCommitted, PacketReady, PlaybackProgress, ResponseStarted/Interrupted/Canceled/Completed, RootAdvanced, UsageUpdated, ToolCallProposed (not executed), ExternalJobStatus, TransportReconnected, TerminalError.

Ordering:
1. Every lane has monotonically increasing sequence and generation; cross-lane arrival is NOT globally ordered merely because frames share a WebSocket.
2. Causal dependency IDs and commit groups declare meaningful cross-track ordering.
3. Provisional ASR hypothesis may revise with the same segment identity and a higher revision; committed transcript cannot be rewritten without explicit correction event/version.
4. Output packet PTS continuity, codec priming delay and media clock must be verified before publishing a complete/playable chunk.
5. No retraction of audio already played; a barge-in terminates future scheduled output and updates conversation presentation state accordingly.
6. Output chunks are leased; final physical backing cannot be freed merely when serialized or ACKed while a device still owns DMA completion.

## 6. Input clocks and buffering

- Audio capture: monotonic local hardware clock where possible; PCM channel layout, sample format, duration and discontinuity markers; explicit resampling with checked rational conversions and retained fractional remainder.
- Video capture: frame PTS and decode/PREP timestamps, image orientation/color/metadata; frame drop selection is explicit task policy, never equate video capture with fixed 30Hz if upstream source jitter occurs.
- Input text: source tokens are not acoustic time, even when streamed alongside input PCM.
- Jitter buffer: bounded by bytes, duration and discontinuity policy. Too much latency becomes a first-class degraded service receipt or admission error, not a hidden infinite queue.
- Echo cancellation/VAD: own signal-processing state, activation/release threshold policy and source processor identity; do not let VAD become a trusted model authorizing unrelated filesystem/remote actions.
- Out-of-order input timestamp: reorder within allowed jitter window, otherwise mark gap/late/drop; never put late media into an earlier committed model frontier.
- Clock synchronization: use checked rational duration alignment and source-registered resampler delay, not magic media FPS constants.

## 7. Playback interruption/barge-in protocol

Canonical steps for server- or client-origin VAD:
1. Ingress event marks activity_start at exact input sample or best-known approximate watermark; attach source and trust level.
2. Session control marks current response output generation canceled and rejects new work from that generation. It does NOT call physical-card-wide epoch cancellation.
3. Drop **unsubmitted** queue tickets and bounded outgoing playback buffers. Already launched kernels keep physical leases until genuine readiness or terminal quarantine.
4. Stop client playback promptly; record best client observed played/acknowledged watermark, distinguishing true player-signal from approximate transport receipt.
5. Model conversation visibility is reconciled by a TaskDriver/source-compatible history policy. E.g. transcript for unplayed speech may be discarded or explicitly marked inaudible; cannot pretend emitted text was heard.
6. Retain already committed input samples and ongoing perception lanes when model permits simultaneous input/output.
7. New reply starts at a fresh TurnEpoch and source-verified model state; no stale prepared chunk, codec residue or old asynchronous tool callback may publish under new epoch.
8. Report interrupted terminal status for old response with high-water marks, dropped output count and uncertainty flags.

**No requirement to always replay the model on barge-in:** native duplex models may carry concurrent audio input state that cannot be reduced to text history. Use their source-verified state transformation or a safe full recompute; if no compatible replay exists, return typed refusal instead of corrupting state.

## 8. Reconnection and resume

Transport reconnect is not guaranteed session migration:
- Reconnect token binds authenticated tenant/session, model revision, output lane/sequence and expiry. Never reuse raw GPU handles or transmit private model state.
- A new connection negotiates codecs/capabilities and receives last *acknowledged committed* media items within bounded replay retention; it cannot request unlimited server-side audio history.
- Resumption must verify session epoch and state compatibility; if model session expired, offer a new session with a declared history import policy or refuse.
- Duplicated events are idempotent by (session, output_generation, lane, sequence, commit_group).
- Server cannot infer exact heard audio after a broken connection; preserve UnknownPlaybackProgress, not an invented complete acknowledgment.

Google Gemini Live publicly documents long-running session resumption, GoAway, compressed context and model-dependent speech interruption; these are API design *test cases*, not hidden engine dependencies. See research dossier.

## 9. Admission and QoS

~~~rust
enum QoSClass { RealtimeSoft, Interactive, Progressive, BatchMedia, Background }
struct WorkloadQoS {
    class: QoSClass,
    desired_first_output: Option<Duration>,
    max_inter_chunk_gap: Option<Duration>,
    max_end_to_end: Option<Duration>,
    weight: FairShareWeight,
    budget: TaskBudget,                          // RAM/VRAM, CPU, IO, output buffers
    graceful_miss: QoSMissPolicy,                 // refuse/degrade only when authorized
}
~~~

Admission must check **whole ModelAssembly** memory and resident components, codecs/encoders, session KV/prosody/latent scratch, driver allocation headroom, output rings, permitted peers and physical device interference. Do not promise hard real-time on general-purpose OS scheduling.

Two-level scheduling:
- Coarse: per-model/per-tenant fair queue + replica/placement choice at request/stage safe boundaries, reservation and deadline estimates.
- Fine: device-local streams/subpools and proved device-native step batches. Do not globally lock a model catalog/arbiter for every audio sample or token.
- Protected interactive windows: batching only compatible state/task/precision cohorts; reserve bounded resources for real-time audio so large video solves don't starve it.
- Stepwise interruption: diffusion model may yield at verified solver-step boundaries; not halfway through unknown physical GPU work.
- Queue capacity and first output latency are both bounded; codegen/tuning/cache loading must not block real-time hot path unexpectedly.
- If requested deadline cannot be met, refuse or choose an *explicitly authorized* alternative model/resolution/quality mode with ModeReceipt. NEVER silently lower precision/steps/voice/model to fake success.

## 10. Media-specific progress, metrics and SLOs

Text: TTFT, tokens/s, P95 decode, accepted-state correctness.  
ASR: real-time factor, word error rate (using source-aligned truth), time-to-first interim, endpoint-to-final latency, timestamp error, speaker diarization accuracy, revision count.  
TTS: first audible PCM latency, real-time factor for continued speech, chunk gap P50/P95/P99, packet underflow, speaker/style consistency, vocoder boundary artifacts.  
Live duplex: mouth-to-ear latency (with disclosed endpoint clocks), interruption detection-to-silence, delayed sample watermark divergence, inbound/outbound jitter and queue misses, full-session fairness.  
Image: full result latency, solver steps, source exactness, VRAM peak, qualitative reproducibility under pinned seed and setup.  
Video: first fully valid frame/segment, frames/s, causal window decode latency, source rate and A/V PTS skew, temporal consistency, peak VAE scratch.  
Decision: schema parse/admission and scored-choice latency, joint questions per batch, calibration error only if calibration exists.  
Hosted: actual provider response/request cost, job backlog, retries, upload/download egress and result validity.

Measure steady warm/cold start separately. Publish evidence hardware, driver, model variant, precision, input sequence, queue depth and percentile calculation. A research repo vendor claim is not a Zerogravity measurement.

## 11. Three fundamental delivery regimes

### 11.1 Local or one-shot request (sync)
- CLI/SDK in-process; no daemon required.
- File input/output with explicit resource source and permissions.
- API short jobs return full result if possible; not appropriate for multihour video.

### 11.2 Progressive streaming (HTTP/SSE or binary streaming)
- SSE for small structured committed events and text.
- PCM/Opus/audio/video bytes through binary frames, files or reference URLs; avoid base64 JSON for every 20–40ms live chunk on hot path.
- Transparent cap on event retention/queue length; server slow-client policy must not buffer unbounded device outputs.
- Transport backpressure propagates credits to task/session without forcing a full GPU lock across network send.

### 11.3 Bidirectional live (WebSocket/WebRTC)
- WebSocket: reliable control events and appropriate binary media framing. Client-managed playback ACK needed to truthfully truncate unplayed audio.
- WebRTC: preferred browser/mobile low-latency media path with negotiated codecs, RTCP feedback and separate reliable control channel; packet timestamps must map to model rational clocks.
- SIP: optional telephony gateway adapter; translate call lifecycle, DTMF, jitter, codec and hangup into the same session events. No direct neural model access.
- Native Rust callbacks: same canonical event contract without network serialization.

A user request for "gpt-realtime-2.5" is a **capability target**, not a verified publicly documented model ID as of this design date. OpenAI's public model catalog reviewed here lists GPT-Realtime-2.1 and GPT-Live variants; preserve model version labels only as live provider metadata. Do not hardcode proprietary internal architecture.

## 12. API matrix and versioning

| Endpoint family (illustrative) | Operation semantics | Must not fake |
|---|---|---|
| GET /v1/models | per-task/model/transport capability, exact artifact and revisions | inference support just from recognizable GGUF |
| POST /v1/chat/completions | source tokenizer/template, generation tools and streaming | fabricated tool call grammar or model substitution |
| POST /v1/completions | raw text continuation with bounded params | hidden chat templating |
| POST /v1/responses | only verified Responses-style event/tool schema | claiming full endpoint fidelity from text-only subset |
| POST /v1/embeddings | encoder pooling/normalization/dimensionality | auto using decoder logits |
| POST /v1/rerank | pairwise/cross-encoder scoring | text generative heuristic |
| POST /v1/audio/transcriptions | file/window ASR; language and timestamp policy | arbitrary partial ASR as final |
| POST /v1/audio/translations | actual source-target speech translation model | translate ASR via different model without disclosure |
| POST /v1/audio/speech | typed voice/speaker/sample-rate/format | partial invalid codec-frame response |
| WS /v1/realtime | strict implemented live session event subset | undocumented pseudo OpenAI fields |
| WebRTC realtime session negotiation | session/codec/token auth and ice/transport state | unbounded relayed media or leaked credentials |
| POST /v1/images/generations and edits | exact supported image model mode | imaginary unified image parameters |
| POST /v1/videos / jobs/{id} | async status/progress/cancel/output | long-running video blocking an HTTP request |
| POST /v1/decisions | typed schema question logits, choice and calibration | generated JSON masquerading as direct logits |
| POST /v1/tasks | native canonical TaskRequest with typed multimodal ports | executing arbitrary user workflow code |

**Compatibility profile** keys: protocol name/version; model identifier+revision; request field set and semantics; supported input/output modalities and codecs; streaming event types; limitation and refusal semantics; task/control algorithm; native/opaque service execution; quality evidence. Unknown request fields with semantic effect must fail or explicitly warn according to versioned compatibility policy; never silently discard safety-/quality-relevant fields.

## 13. Hosted opaque provider TaskServiceAdapter

External API providers: authorized hosted TTS/image/video generation services, proprietary decision services, hosted GPT-Realtime/Gemini Live and future models. A hosted provider exposes **capabilities and actual API observations**, not local SIR/TensorEncode semantics. No unverified backend region state or GPU memory leases. Provider model revisions can change; pin named snapshot if available, otherwise record alias/revision uncertainty.

~~~rust
trait TaskServiceAdapter {
    fn capabilities(&self) -> TrustedProviderCapabilityManifest;
    fn validate(&self, task: &TaskRequest, authz: &DataEgressPolicy) -> Result<ProviderRequest>;
    fn start(&self, req: ProviderRequest, key: IdempotencyKey) -> Future<ProviderJobHandle>;
    fn watch(&self, job: ProviderJobHandle, cursor: Option<ProviderCursor>)
        -> Stream<ProviderEvent>;
    fn cancel(&self, job: ProviderJobHandle) -> Future<CancelReceipt>;
    fn reconcile(&self, key: IdempotencyKey) -> Future<ProviderJobStatus>;
}
~~~

This is illustrative, not a fixed public trait. Provider job lifecycle: Created → Accepted → Queued → Running → Partial (only if provider guarantees validity) → Succeeded / Failed / Canceled / Unknown. A network timeout after submission does NOT imply job cancellation. Idempotency keys, cost bounds, upload retention and retries must avoid double billing.

Security:
- per-tenant secrets excluded from logs, model prompt and event payloads; no browser exposure of server API keys;
- explicit allowed external domains/region/data egress; signed URLs bound to scope and TTL; verify received asset MIME, size, integrity;
- upload voice samples and image/video with explicit consent/rights; avoid deceptive "on device" labels;
- content credentials and provider provenance recorded only when actually supplied, not invented;
- rate/cost limits and audit record without leaking private inputs.

## 14. Failure and availability rules

- **Transport disconnect:** stop/pause per declared session policy; physical CUDA work still held until Ready. Do not drop uncommitted StateRoot because socket died.
- **Client slow playback:** decrease producer credits, pause at legal frame/solver boundary or abort; no unlimited buffering.
- **Bad media:** reject excessive size, decompression ratio, corrupted timestamps, nonfinite samples, invalid color format, unsupported codec and impossible dimension before GPU launch.
- **Unsupported device:** portable source-verified candidate or typed refusal; no silently lowered precision.
- **One-session interruption:** cancel only its turn generation. Physical-device-wide epoch changes reserved for actual device-level failure/reinitialization.
- **Remote worker failure:** never trust remote "complete" without authenticated source/schema/event receipt; no locally forged StateRoot.
- **Hosted job 502/timeout:** job status Unknown and reconcile; don't retry an expensive video create unconditionally.
- **Observed API mismatch:** mark compatibility tier unsupported; explicitly reject that feature.

## 15. Media presentation receipts and truthful playback

A client "played up to sample N" event is weaker than physical playback proof; it may be absent/forged. Maintain:
- reported watermark and origin (client SDK, RTP feedback, player callback, inferred transport),
- delay bounds and uncertainty interval,
- output generation and codec clock,
- acknowledged vs reported played distinction,
- authenticated monotonic advancement; never decrease within one generation,
- action on client reconnect/late ACK (ignore stale generation).

In OpenAI Realtime WebSocket, documented conversation.item.truncate removes the unplayed audio part and corresponding transcript; in WebRTC/SIP output audio buffer handling is server-managed. These are adapter semantics, not universal model capabilities. Source: https://developers.openai.com/api/docs/guides/realtime-conversations (reviewed 2026-10-15).

## 16. Explicit compatibility prior art (not native implementation)

- OpenAI realtime: https://developers.openai.com/api/docs/guides/realtime-conversations
- OpenAI client event schema: https://developers.openai.com/api/reference/resources/realtime/client-events
- OpenAI public model catalog: https://developers.openai.com/api/docs/models
- Google Gemini Live: https://ai.google.dev/gemini-api/docs/live-api/capabilities
- Google session resumption, context and GoAway: https://ai.google.dev/gemini-api/docs/live-api/session-management
- Moshi (external reference removed during attribution cleanup)
- Hosted TTS live API and voice SDK (external reference removed during attribution cleanup)
- Speech toolkit prior art (external reference removed during attribution cleanup)
- Joint audio/video generation prior art (external reference removed during attribution cleanup).

All rate/latency/token values shown in other documents are per **upstream claim or model checkpoint**, not Zerogravity measured SLO or hard-coded defaults.

## 17. Acceptance drill extension

MM-13: duplex input while output; server and client VAD; turn cancellation stops audio without dropping input.  
MM-14: late playback ACK after new output generation is ignored.  
MM-15: 44.1k→16k/24k resampler and 30000/1001 video clock maintain exact timebase across prolonged stream.  
MM-16: no held GPU physical buffer released because network send resolved or timeout occurred.  
MM-17: slow WebSocket client causes bounded producer backpressure while other tenants remain live.  
MM-18: 5 video diffusion jobs plus 10 duplex streams share GPUs within declared safe quotas; fair admission without numerical downgrade.  
MM-19: websocket reconnect/resume with duplicate event ID returns idempotent committed history and consistent watermark.  
MM-20: hosted video-generation timeout after create remains Unknown and reconciles without double-submitting expensive job.  
MM-21: source-capability matrix rejects unsupported image edits/voice/decision fields and advertises actual backend.  
MM-22: no per-packet model routing or global resource snapshot allocation in critical live path.  
MM-23: source A/V joint generator cannot expose video without corresponding valid audio watermark when declared coupled.  
MM-24: malicious media, unauthorized egress and cross-tenant replay tokens rejected before heavy resource reservation.

**Before feature implementation:** design mocks MM-01..MM-24 with deterministic fault injection and explicit assertable counters. Then original source checkpoints and actual local device tests with observed latency. No raw model weights or untrusted live audio required for the mock phase.

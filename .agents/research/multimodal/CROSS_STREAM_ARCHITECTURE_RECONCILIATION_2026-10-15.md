# Research Track H — Cross-Stream Adversarial Reconciliation and Minimal Architecture Delta

**15 October 2026 · final design integration, no implementation.**  
**Inputs:** the five family research dossiers (ASR, TTS/duplex, image, video/AV, decision/ternary), the serving/API prior-art dossier, existing v4.3 core and v5 multimodal proposal, the user's universal specialization/serving requirements.

## 1. Scope and governance

This is a review of **design claims**, not verification of every source-model neural checkpoint. Official source URLs, inference algorithm evidence, mathematical identities, current release status and licenses must be re-checked per version at implementation. A sample API from an opaque provider is evidence of an API event contract, not a disclosed source model.

There is no actual autonomous multi-agent orchestration tool in this conversation environment and no separate research subagent execution was claimed. These named streams represent independent analysis and preserved task briefs. The previous 14 October workstream plan explicitly noted local Codex/Claude standalone authentication failures; we did not attempt credential workarounds.

## 2. Overarching verdict

**The v4.3 execution architecture is substantially correct; v5 predicted nearly all multimodal model type requirements. A narrow v5.1 refinement is needed for live transport/presentation frontiers, soft-realtime QoS/admission, canonical MediaEvent/API profiles and external async jobs.** Preserve one EngineCore/TaskCoordinator/PhysicalResourceArbiter/ModelRouter and native Rust-owned reference semantics. Do not turn every modality into a new inference server, fork the resource authority or make a general workflow interpreter.

## 3. Cross-stream conflict and disposition matrix

| Conflict / failure temptation | Disposition | Proof demanded |
|---|---|---|
| "Image output means latent denoising" vs an AR image-codec source | TaskDriver may run AR codec, flow/denoise or verified hybrid under one assembly | trace exact published generate_image source; refuse unknown stages |
| "Video is independent visuals + optional TTS" vs joint neural AV | coupled dual-stream model must share source solver/attention/effect frontiers | MM07/MM23 valid joint CommitGroup and perturbation negative test |
| "Every speech model can stream" vs offline Parakeet TDT variants | capability per checkpoint, decoder and implementation | unsupported streaming returns typed refusal |
| "One audio frame equals one audio token" vs RVQ codec depth and streaming neural codecs | rational acoustic frame clock × independent RVQ depth × decoded sample clock | incomplete RVQ frame never published |
| "Transcript is final as generated" vs streaming ASR revision/diarization | provisional revisioned hypotheses and stabilized segment publication | no stale or out-of-order final transcript |
| "A model response is what a user heard" vs WS playback reality | presentation ledger distinct from committed neural StateRoot | interruption truncation based on actual/unknown played watermark |
| "gpt-realtime-2.5 must exist and expose model internals" vs public API | preserve as user-supplied target name, verify actual public model catalogue, model as opaque provider | no invented local neural semantics |
| "All decisions are LLM JSON" vs hosted and encoder decision heads | dedicated finite-option TaskDriver + direct logits/typed schema | reject generated JSON as source-equivalent direct decision |
| "Qwen topology loads ternary weights" vs PTQ1/PQ2/Hadamard | orthogonal verified TensorEncoding and runtime transform, source-matched numeric reference | missing Hadamard/codec hard fails |
| "Open code means commercially redistributable weights" | independent per-checkpoint/processor/codec license and rights receipt | deny disallowed distribution/egress |
| "Vulkan GPU availability means FP32 exact numerics" vs AMD FMA | source/device/driver/precision evidence keys; do not silently widen tolerance | exact source vectors on actual device |
| "A future canceled means card can be freed" | await proven completion or quarantine submitted unknown buffers | real post-launch abort retains global fence/leases |
| "One physical epoch per session" vs shared GPU | PhysicalDeviceEpoch is card-wide, Session/TurnEpoch independent | barge-in does not cancel other session/card |
| "Global router runs per audio chunk" | bind model/task/replica once at admission; local device/native StepProgram fast loop | no JSON/catalog/physical global snapshot per audio frame |
| "External hosted video job becomes source StateRoot" | explicit TaskServiceAdapter job lifecycle; no local neural region claim | timeout after accepted POST remains Unknown |
| "We should add one generic TaskKind::Anything" | bounded trusted TaskDrivers with types/stages, not a giant dynamic graph DSL | adding model doesn't edit global router/worker |
| "One sample rate for audio" | source-specific rational clock and codec/processor version | 44.1/48k resample and 29.97FPS no drift |
| "Commit means delivered means played" | independent model commit / transport delivery / ACK / playback ledger | late ACK after barge-in cannot rewrite old generation |
| "Solver fewer steps is transparent fast kernel" | source model/quality mode/version distinct; seeded numerical policy | wrong schedule/CFG rejected |
| "GPU fit means latency target met" | assembly-wide resources plus measured physical headroom and QoS admission | mixed video/duplex stress and soft latency receipts |

## 4. Minimal delta against v4.3 + v5, not a redesign

Already correct in v4.3:
- SemanticProgram mathematical identity + prepared ExecutableBundle.
- Transient RIR/KIR and CIR embedded inside command/resource bundle; not several separately serialized dialects.
- TaskDriver stage/iteration control under one StateRoot publisher.
- RegionProvider/device execution families, shape/numeric/state compatibility.
- PhysicalResourceArbiter coarse admission and local leased fast subpools.
- built-in ModelRouter over same EngineCore, model identity/source exactness, legal distributed placement.

Already proposed in v5 (promote into future governing implementation design, do not claim implemented):
- ValueDescriptor with precise media axes and rational timebase;
- verified ModelAssembly and component ports;
- source-specific codecs and TaskDriver algorithm families;
- multitrack TaskFrontier and legal AV commit groups;
- orthogonal TensorEncoding/activation transforms and ternary low-bit case;
- explicit hosted TaskServiceAdapter separation.

**New v5.1 proposals only:**
1. RealtimeSession with independent capture/response lanes, TurnEpoch and TransportEpoch.
2. PresentationLedger tracking prepared/committed/delivered/acknowledged/played separately from StateRoot.
3. Canonical MediaEvent schema with lane/sequence/timebase/commit group/finality/payload reference.
4. WorkloadQoS for soft deadlines, fair-sharing, bounded media buffer/queue admission and typed deadline refusal.
5. ProtocolCapabilityManifest describing per-model/per-task/per-event fidelity under versioned HTTP/WS/WebRTC/SIP/async API.
6. ExternalTaskJob ledger with Unknown/reconcile/idempotency and explicit egress/credential/license costs.
7. MM-13..MM-24 mock conformance suite for barge-in, AV clocks, QoS, reconnect, hosted jobs and refusals.

**Do not create** a MediaEngine, VideoEngine, AudioEngine, BrokerService, GPUAllocator2, WorkflowRuntime or TaskRoot2. Logical helper types/modules can be introduced within current Rust crate and proven later.

## 5. Source-to-execution map

| New feature | Source/model registry delta | TaskDriver delta | Kernel/provider delta | Common engine delta |
|---|---|---|---|---|
| Whisper ASR | mel/encoder/decoder config | 30s window AR+timestamp merge | DSP/attention source kernels | none beyond typed audio/clock |
| Parakeet CTC/RNNT/TDT | feature encoder+heads | blank/predictor/duration | fused source conv/encoder/joiner | none |
| Kokoro TTS | G2P/speaker/vocoder | duration/prosody→PCM | FFT/vocoder operators | media output descriptor |
| Dual-AR RVQ TTS | 10-codebook RVQ codec | time/depth nested AR | embedding/codec/acoustic op | codebook clocks/finality |
| Full-duplex codec model | streaming neural codec and source attention | simultaneous input/output | stream codec/temporal kernels | RealtimeSession/played ledger |
| Rectified-flow image | conditioner+DiT+VAE | flow/guidance iteration | MMDiT/flow/latent VAE | typed latent result |
| Qwen-Image | variant-specific MMDiT + edit | denoise or exact source stage | image/attention/VAE | typed image result |
| AR/hybrid image | audited AR/hybrid components | checkpoint-specific AR+codec/flow | source ops | none beyond flexible driver |
| Wan2.2 video | two denoise phase experts | source noise phase switch | temporal DiT+3D VAE | video typed track |
| Asymmetric video | asymmetric text/video attention | source denoise | non-square projection, AsymmVAE | none |
| Joint AV diffusion | joint dual latents and codecs | coupled solver/AV join | audio/video cross-attn, dual VAE/vocoder | synchronized CommitGroup |
| Encoder decisions | encoder + schema head | finite choice score | source encoder/score math | decision output |
| Joint-schema decision | Qwen backbone+joint head | multi-question score | joint head | decision output |
| Ternary low-bit | packed ternary and Hadamard | Qwen-family AR | new codec/transform/kernel | no router/core rewrite |
| Hosted media providers | external provider manifest | external task status only | none | TaskServiceAdapter at task layer |

**Modularity acceptance law:** a new registered model adds only required semantic adapter/operator/driver/codec/provider, not rewrites of Router/TaskCoordinator/ResourceArbiter/network protocol core. Any exception needs a written measured change-radius justification.

## 6. Hard invariants to keep

### Single authority
Every model/task including multimodal speech/video publishes only through TaskCoordinator's epoch-checked PublishRoot. A codec/transport can prepare and deliver eligible committed buffers but cannot change neural state. Remote worker/authenticated device events are merely readiness receipts.

### Physical buffer safety
All submitted device commands carry physical card identity, numeric capability, source/model identity and real event/lease keepalives. A canceled/unknown submitted Future retains original buffers and fence, no timer-based memory reclaim. No transport cancel request can automatically "unfence" device.

### Semantics
Processor/tokenizer/codec/solver/guidance/quantization/activation transform and model specific masking are part of source semantics and quality policy; unspecified transformations cannot be silently optimized across GPU shape bands.

### Time
Audio sample, acoustic semantic frame, RVQ codebook depth, text token, solver iteration and video frame/latent clocks distinct; exact rational conversions and codec delays. Published AV groups obey both lanes' legal readiness.

### API truth
A named model can be recognized but not executable; CPU reference may be qualified while GPU source region only partial; a hosted provider can answer API but doesn't become locally loaded weights. The capability matrix must distinguish each case.

### Security
No untrusted custom checkpoint host code; no default remote egress; separately authorized voice cloning/face/rights; tenant session state isolation; sensitive media metadata redacted in logs by default.

## 7. Additional design recommendations still OPEN, not silently approved

- Which codec/media reference implementation family to start first (Whisper vs Parakeet; Kokoro vs dual-AR RVQ; source-real video checkpoint size)? The architecture can accept multiple, but practical first choice needs artifact/license/hardware feasibility.
- Should numeric-equivalence mode be strict bit-exact for every media decoder pixel/PCM sample, or define a source-verified task-specific quality tolerance? **Default reference exactness remains governing for current Qwen**; don't assume all media codecs have bit-exact cross-device outcomes. An explicit user-approved quality policy can be introduced with evidence.
- What is the default audio buffer QoS: desired mouth-to-ear P95 under a given device/load? User supplied no numeric SLA; design only requires measured configurable soft objective.
- What durable playback event retention and encryption policy is appropriate? Need privacy/cost decisions and measured memory tradeoffs before persistence.
- How to expose hosted provider cost/egress rules? Default deny until trusted authorization; no implicit provider install.
- Do multi-provider audio/video synchronous clients need one multiplexed WebRTC connection or distinct tracks/sessions? Allow both in protocol without binding EngineCore to a single transport.

## 8. Adversarial review questions to require on every implementation PR

1. Which actual source tensor/algorithm/processor proves this task?
2. Which source artifact/codec/assembly revision is immutable for a running session?
3. Which state domain is provisional? Who alone can commit it?
4. Is this a real GPU readback or only a host simulation/compiled shader?
5. Does another model on the same GPU retain its own state and fairness?
6. Does this callback use a global snapshot/lock/JSON encoder per packet/token?
7. Is the source API genuinely compatible or are parameters silently discarded?
8. Is this a native model, a trained multi-component assembly, an application pipeline or a hosted provider?
9. Do output clocks/timebase/codec and actual playback frontier agree?
10. Is a low-bit codec and activation transform mathematically required and checked?
11. Does disabling a streaming feature preserve correctness rather than output fake chunks?
12. Could cancellation/retry free an unconfirmed physical buffer or double bill a remote job?
13. Are immutable weights and codecs shared only when source hashes/legal terms match?
14. Is the new abstraction necessary, or can it be an existing TaskDriver/Provider extension?
15. Can a second LLM agent reconstruct required golden fixtures and tests from documents alone?
16. Did benchmark include end-to-end media tasks rather than isolated tokens/s or FLOPs?
17. Could an attacker supply unbounded input media, malformed codec, infinite revised transcript or arbitrary file path?
18. Was exact original device/driver/source checkpoint and capability tier recorded in a ModeReceipt?

## 9. Proposed acceptance result classification

A feature is **DESIGN SPECIFIED** when typed ports/algorithm/clocks/ownership/error and adversarial mock tests are documented. It becomes **SOURCE-MATH VERIFIED** when original model/processor/codecs and CPU reference vectors run; **PARTIAL PHYSICAL** when only a GPU region executes; **FULL PHYSICAL** when end-to-end source-equivalent model and state output execute on exact backend; **STREAMING QUALIFIED** when playable chunks, abort, memory and latency tests pass; **HOSTED API VALIDATED** when actual approved provider request/response lifecycle conforms. **No future model family is already runnable solely because a document exists.**

## 10. Reconciliation deliverables

Authoritative for agents:
- .agents/docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md
- .agents/docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md
- .agents/docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md
- Independent ASR, TTS, image, video, decision/ternary, serving research dossiers (this folder).
- .agents/research/MULTIMODAL_V5_1_DECISIONS_AND_PROVENANCE_2026-10-15.md
- .agents/research/MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md
- Current [multimodal checklist](../../plans/multimodal/TODO.md) and [TODO rules](../../plans/TODO-SPEC.md); earlier aggregate bodies remain in Git history.

All choices remain subject to explicit user approval where they are recommendations rather than user directives.

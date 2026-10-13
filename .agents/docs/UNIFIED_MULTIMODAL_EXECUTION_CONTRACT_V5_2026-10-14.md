# Zerogravity — Unified multimodal execution contracts (v5 design proposal)

**14 October 2026 | Proposed refinement, NOT implemented multimodal model support.**
This is additive to the v4.3 governing docs: MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md, COMPOSITION_AND_EXTENSION_CONTRACTS.md, TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md, and MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md. See companion model-family research in the plans directory.

## MM0 executable mock acceptance — 14 October 2026

**Six of twelve proposed MM acceptance drills now have passing
source-free mock tests** under tests/multimodal_mm0.rs: rational AV clocks
(MM-01), typed component ports (MM-02), atomic fake iterative solver
(MM-03), duplex barge-in with device epoch/lease fencing (MM-06),
synchronized media publication (MM-07), and finite-choice scoring (MM-08).
MM-06 uses the new Rust-only DeviceEventLedger and the existing
TaskCoordinator/StateRoot: a cancelled prior audio stream cannot free
device buffers before signaled completion or publish stale output after a
later user turn. These are mock backend events, not actual GPU hardware,
and add **no** new production inference engine or source-media codec.
The remaining six MM drills and source-verified real media/decision
inference remain unimplemented.

## Executive conclusion

**The v4.3 execution spine is correct but underspecified for comprehensive image/audio/video/decision/multimodel support.** Preserve one native Rust SemanticProgram/ExecutableBundle, trusted TaskDriver, TaskCoordinator sole PublishRoot, resource authority, region provider, executable placement graph and model catalog. Do NOT add independent model-family servers, a general workflow interpreter, or a new persistent IR for each modality.

Six narrow architectural amendments are required:

1. **Typed media values and rational timebases.** Tokens, image/video latents, audio PCM/mel/codebooks, structured decisions, compressed streams and stream clocks must have explicit meanings and constraints.
2. **Verified component assemblies.** A model may comprise encoders, transformers, VAEs, codecs, vocoders, acoustic heads, decision heads and diffusion solvers that can be placed independently while retaining one exact artifact identity.
3. **Generalized task and iteration semantics.** Distinguish decoder token AR, parallel classifier scoring, CTC/RNN-T/TDT transcription, nested audio codebook generation, diffusion/flow solvers, and temporal video reconstruction.
4. **Synchronized multi-track StateRoot and streaming.** Maintain independent audio/video/sample/codec/denoise frontiers with legal coordinated publication, backpressure, low-latency duplex cancellation and replay.
5. **Extensible physical tensor encodings and legal activation transforms.** Model topology must be independent of Q4/Q6/ternary bit packing, Hadamard transforms, RVQ audio codebooks, FP4 or numerical reduction schedule.
6. **Honest composition and service boundaries.** Native model regions, compound local models, multiple independent models, replicas and proprietary external TaskServiceAdapters are all different concepts.

These are concrete refinements to existing types/registries, not six independently owning services.

## MM-01. Typed value and clock contract

The existing generic ValueType must specify the physical/semantic axes. Suggested *illustrative* Rust data contracts:

~~~rust
enum ValueDescriptor {
    Tensor(TensorDesc),           // rank, dtype, layout, axes, source codec
    TokenSequence(TokenDesc),     // text / image / acoustic / interleaved
    CodebookSequence(CodebookDesc), // outer frame, inner RVQ codebook
    AudioSignal(MediaDesc),       // PCM, channels, sample rate, layout
    VideoFrames(MediaDesc),       // dimensions, pixel/color space, timestamps
    MediaLatents(LatentDesc),     // spatial × temporal VAE, source compressor
    EncodedPackets(CodecStreamDesc), // codec ID/config, PTS/DTS
    StructuredValue(SchemaKey),  // typed object or finite answers
    DecisionLogits(ChoiceSchema),// per-question option logits/calibration
}
struct Timebase { ticks_per_second_num: u64, ticks_per_second_den: u64 }
struct TrackCursor { track_id: TrackId, tick: u64, clock: ClockDomainId }
enum ClockDomainKind {
    TextToken, AudioSample, AcousticFrame, RVQCodebookDepth,
    VideoFrame, VideoLatentFrame, SolverStep, ExternalWallTime,
}
~~~

Rational sample/frame times are required. 44.1k/48kHz audio, 24/29.97/50fps video, 12.5/21/25Hz acoustic tokens and text token counts are NOT interchangeable clocks. Resampling/codec decoding must carry explicit source-registered transforms, bounded overlap buffers and rounding residue. Image color space/pixel format, video frame order and audio channel layout cannot silently disappear inside opaque byte blobs. BufferLease can still hold all values physically.

Treat decoded and encoded media as distinct semantic types. A VAE latent image or unfinished RVQ frame is not automatically a playable output. Transport-level media container, mux, packet timestamp, license/rights and finality must be validated.

## MM-02. Verified model-component graphs

A source model can have many components without becoming many independent serving engines:

~~~rust
struct ModelAssembly {
    source: ArtifactKey, semantic_key: SemanticKey,
    components: Vec<VerifiedComponent>,
    typed_edges: Vec<TypedPortEdge>,
    supported_tasks: TaskCapabilitySet,
}
enum ComponentClass {
    Tokenizer, TextEncoder, VisionEncoder, AudioFeatureExtractor,
    AudioEncoder, DecoderLM, SparseExperts, RecurrentCore,
    AcousticDecoder, MultiCodebookDecoder, FlowPredictor, Denoiser,
    VAEEncoder, VAEDecoder, Vocoder, WaveformCodec, UpScaler,
    TranscriptionHead, DecisionHead, TemporalSynchronizer,
}
struct TypedPortEdge {
    from: ComponentPortId, to: ComponentPortId,
    type_and_clock: ValueDescriptor,
    conversion: Option<RegisteredCodecId>,
    state_effects: StateEffectContract,
}
~~~

Every component carries an independently hash-verified weight source, trusted processor/codecs, numerical semantics, shape/precision guards, adapter version and license. Unknown/custom executable checkpoint code is never imported as arbitrary Rust host callbacks.

Examples: FLUX.2 separates a *checkpoint-specific* conditioner ([dev] Mistral-class, [klein] Qwen3 embedding) from flow DiT and VAE; one autoregressive multimodal MoE image model is described as such, but the exposed source generation path must still be audited for diffusion or other non-AR stages; not simply a generic image denoiser; Qwen3-Omni has separable Thinker/Talker and audio codecs; an audio generation model has slow temporal AR, fast RVQ depth AR and waveform decoding; a joint audio/video model has two cross-attending audio/video latent streams; a typed decision model has a Qwen backbone plus a typed schema-scoring head.

A component may be resident on a different CPU/GPU/peer with explicit typed edge transfers, event leases and admission. The TaskDriver chooses when a component runs; the compiler and placement policy decide where its closed semantic regions execute. Keep **component assembly** distinct from a user-authored external workflow and distinct from **replicas** of one model.

## MM-03. Distinct iteration policies under one TaskDriver

Keep one minimal trusted TaskDriver interface but register stage/iteration kinds rather than assuming one token loop:

~~~rust
enum StageIntent {
    Encode(RegionGroup), DecodeText(TokenPhase),
    DecodeAcoustic(CodebookPhase), DecodeTranscript(CTC_RNNT_TDT_BeamSpec),
    ScoreFiniteSchema(ChoiceSchema),
    Denoise(DenoisePhase), FlowStep(FlowPhase),
    Reconstruct(CodecStage), Synchronize(TrackBarrier),
    ParallelKnown(RegionGroup),
}
struct IterativePolicy {
    solver: RegisteredSolverId, schedule: RegisteredScheduleId,
    numeric_policy: NumericPolicyId, rng: RngDomainId,
    shape_bounds: IterationBounds, rollback: StateEffectContract,
}
struct TaskFrontier {
    epoch: TaskEpoch, tracks: Vec<TrackCursor>,
    iterations: Vec<StageCursor>, published_group: CommitGroupId,
}
~~~

Token decode, encoder/embedding pooling, classifier inference, streaming transcription, nested audio codebooks, iterative flow/diffusion, causal VAE and duplex conversation have genuinely distinct termination and validity conditions. A Device-native StepProgram may run many internal microsteps without host round trips, but host TaskCoordinator remains the ONLY publisher of committed state/output.

Diffusion/flow scheduler, timestep ordering, seed and RNG, guidance branches, noise prediction parameterization, distilled step count and conditional expert routing are part of the **task/model semantic contract**. Altering these for speed is a new quality mode/model, not a transparent fused kernel. Source-registered solver families can cover Euler, Heun, DPM-type or learned schedule where exact math/evidence is known.

Speech: an outer 12–25Hz acoustic temporal token and inner multi-codebook RVQ depth loop are two coordinates. Whisper's encoder-decoder audio/text attention is not the same task as CTC/RNN-T/TDT streaming transduction, including blank and beam state. A typed decision head scores finite options with NO free-form token generation.

## MM-04. Multitrack StateRoot, synchronized commit and interruption

Extend StateDomains, preserving **Prepare each → await all physical ready events → one TaskCoordinator PublishRoot**:

| Typed domain | Examples | Valid commit boundary |
|---|---|---|
| TextToken | token history, grammar, sampler, top-k RNG | accepted token or verified prefix |
| AttentionRecurrent | KV, sliding cache, GDN/SSM | model-specific complete step |
| AudioFrontend | resampler/STFT overlap, mel windows, VAD | exact input sample/window offset |
| Acoustic | codec frames, codebook depth, voice/style embedding | complete reconstructible acoustic frame |
| Transcript | text hypotheses, diarization, timestamps | committed transcript segment |
| SolverLatent | image/video latent, timestep, guidance, solver history, RNG | complete denoise/flow iteration |
| TemporalVideo | 3D VAE tiles, latent temporal cache, frame sequence | valid temporal frame window |
| AudioVideoSync | track cursors, lip-sync alignment, ready watermark | declared playable AV sync group |
| TypedDecision | schema, option logits, calibrated probabilities | atomic scored record |
| CrossModelEdge | source-verified stage outputs / ownership | complete typed edge |
| OutputDelivery | stream sequence, retention, credits, receipts | committed packet or chunk |

A scalar StateRoot position is insufficient for multiple clocks. CommitGroup declares which audio samples and video frames become final together; alignment is computed from exact rational clocks, source delay and resampler residual. It may publish one audio segment before video only if the model/stream format explicitly permits that validity, and cannot expose partially denoised frames or incomplete codec frames.

**Full duplex:** simultaneous audio input and output lanes with capture timestamp, playback watermark, VAD/endpointing, incremental ASR, TTS buffer, speaker-turn and interruption epoch. On user barge-in: cancel all uncommitted output after the turn boundary; already played audio cannot be recalled; fence in-flight GPU/codec writes; preserve/replay source-compatible conversation context and release leases only after actual completion. Bound input jitter, queued playback, and all latency/deadline budgets; no blanket hard-real-time guarantee on shared OS.

## MM-05. Denoising, temporal specialization, and task performance

Decompose image/video tasks into **conditioning encoders**, **denoiser/flow predictor**, **registered sampler+guidance**, and **VAE/media reconstruction**. Text/VLM conditioners can be independently resident; image and video latents have ranks and time axes, not fake token identifiers.

- Qwen-Image and SD3.5: MMDiT/DiT latent predictors, text encoder(s), guidance and VAE.
- FLUX.2: source-variant conditioner ([dev] Mistral-class; [klein] Qwen3 text embedding) + rectified flow transformer + independent VAE and multiple image references.
- An autoregressive multimodal image model: described as a unified multimodal autoregressive MoE; its released generate_image path needs audit for diffusion/other stages before choosing a mandatory AR-only or MMDiT-only workflow.
- Wan 2.1: spatiotemporal DiT/3D VAE; Wan 2.2: high/low noise-phase **whole-model expert** selection by denoising step, not an ordinary token-level FFN MoE.
- An autoregressive multimodal video model: 3D causal VAE plus selective/sliding tile sparse attention and optional superresolution. Sparse mask is SEMANTIC; cannot silently replace with a different local window.
- Mochi: asymmetric multimodal attention with non-square projections and separate video VAE.
- Joint audio/video models and similar: jointly conditioned, synchronized audio/video streams; cannot be modeled as independent TTS plus video with an arbitrary merge.

Media performance policy includes latency-to-first-playable-frame/audio, true real-time factor, peak latent scratch, VAE tile size, audio/video synchronization error, completed sample quality, context streaming, denoise step count and bounded device leases. Multi-model resident conditioner+denoiser+codec sharing requires resource budgets and eviction safety across concurrent tasks.

## MM-06. Orthogonal tensor encodings and ternary adversarial compatibility

**Model topology and source tensor encoding must not be conflated.** A ternary 27B model preserves the hybrid Qwen3.8 model architecture, but the PTQ1_0/PQ2_0 original source weights and runtime block Hadamard transformed activations differ fundamentally from Q4/Q6. A decoder recognizing Qwen3.8 attention does NOT imply it can load/execute that ternary model.

Proposed TensorEncoding contract (illustrative):

~~~rust
struct TensorEncoding {
    source_format: TensorSourceFormat,
    physical_codec: WeightCodecId, // Q4_0, PTQ1_0, PQ2_0, FP4...
    block_geometry: BlockShape,
    scale_encoding: ScaleEncoding,
    activation_transform: Option<RegisteredTransformId>,
    activation_codec: Option<ActivationCodecId>,
    accumulator: AccumulationContract,
    semantic_kernel_key: SemanticKernelKey,
}
enum RegisteredTransform {
    Identity, Cast, LayoutPermutation,
    BlockHadamard{group_size:usize,sign_scheme:SignedBasisId},
    PackCodebooks, CodecResample,
}
~~~

g128 ternary weights use group scales, runtime matching Hadamard activation transformation, direct bitpacked GEMV; source mismatch is hard load failure. Allow source-qualified backend kernels that fuse Hadamard+ternary dot without unpacking entire original checkpoint into FP16. Applicable equally to BitNet-style low-bit, FP4, dynamic activation formats, audio codec RVQ, and mixed GEMM quant policies. Current Phase 1 Qwen QuantKind enum is legitimately narrow; extend through a verified codec/transform registry when an actual new source checkpoint is admitted, rather than adding an unbounded trait zoo today.

## MM-07. Typed decision scoring, not JSON generation

Proprietary decision services and Qwen-backed decision models require a ScoreFiniteSchema TaskDriver. Inputs: JSON/text/media state and finite schema of yes/no, choice or score questions. A model returns an option-logit vector per question, with explicit normalization/calibration and schema version:

~~~rust
struct DecisionResult {
    schema: SchemaKey, questions: Vec<QuestionScores>,
    calibration: Option<CalibrationId>, source: ArtifactKey,
}
struct QuestionScores {
    question_id: QuestionId, options: Vec<ChoiceId>,
    logits: Vec<f32>, probabilities: Vec<f32>,
    confidence_semantics: ScoreInterpretation,
}
~~~

A ModernBERT-based decision encoder and a Qwen multimodal backbone+joint schema head are distinct assemblies under one task kind. A proprietary decision service is exposed only via a separately authorized external adapter. Never generate JSON tokens and call it source-equivalent to direct classification scores; never invent calibrated confidence if the model supplies only raw logits. Many questions may be processed jointly in one forward; task routing is allowed to choose a true decision model instead of a bigger text generator when policy/quality support it.

## MM-08. Native versus opaque external models

Hosted image, video and TTS generation services generally require **approved TaskServiceAdapter**, not native SemanticProgram execution, unless source weights/math become legally accessible. These adapters exist ABOVE TaskRequest and have independent credential/egress/privacy controls, asynchronous job polling/webhooks, file/bucket staging, cost accounting, provider model revision, optional C2PA and rights policies. They may participate in a higher application workflow, but are not a DeviceBackend and may never claim to run on local GPU or publish local neural StateRoots.

Keep OpenAI-style audio/image/video/response APIs only where endpoint semantics genuinely align. Use native typed endpoints for decisions, synchronized AV and compound tasks. Unsupported parameters are typed errors, not dropped. Typed output events carry lane, sequence, source timebase, media timestamp, commit group, encoding, finality and payload reference.

## MM-09. Four orthogonal dimensions of specialization

Any candidate closed-region implementation is legal only across these four checked dimensions:

1. **Semantic:** MMDiT, flow, sparse mask, hybrid GDN, AR codec, audio text alignment, attention variants, ternary transform and head semantics.
2. **Physical:** CPU SIMD/CUDA/HIP/Vulkan/Metal/NPU/AOT, tiling, compression, matmul group FMA, graph capture, tensor layout.
3. **Task/control:** encoder, decode1, acoustic codebook loop, CTC beam, denoise step, long temporal window, latency/deadline, typed decision batch.
4. **Placement/resources:** UMA/discrete device, resident conditioner/denoiser/codec, KV/latents/prosody state, staged transfers, remote privacy, memory/IO quotas.

Plan keys also include source checkpoint, semantic/codec version, numeric policy, legal state bridge, resource/event budgets and exact device evidence. A driver supporting only a native kernel or whole AOT graph can still implement a legally closed region; there is no model-name conditional in the core router.

## MM-10. Minimal fake acceptance drills, before feature implementation

These tests are **proposed, not currently implemented**.

- MM-01: 44.1kHz PCM and 29.97 FPS rational clocks, lossless resampler offset and exact AV mux ordering.
- MM-02: verified conditioner→flow predictor→VAE graph, independent device placement/admission, artifact revocation.
- MM-03: 20-step flow/DiT with seed, noise schedule, guidance and partial-step failure; wrong sampler must not pass.
- MM-04: audio-codebook-style 21 Hz slow AR + 10-codebook depth AR and complete codec frame commit.
- MM-05: Whisper encoder-decoder versus Parakeet CTC/RNN-T/TDT transducer; blank and beam state, timestamps, streaming recovery.
- MM-06: full duplex input/output, partial ASR, barge-in, in-flight device completion, bounded latency credits.
- MM-07: joint-audio/video-style synchronized generation with dual-stream attention, different clocks and matched output watermark.
- MM-08: Qwen-backed decision model / ModernBERT-based encoder with 20 typed schema questions in one forward pass, calibrated distribution shape and rejection of invalid choices.
- MM-09: Ternary PTQ1_0/PQ2_0 source bitpack and Hadamard activation transform, exact CPU/physical backend parity, reject missing transform.
- MM-10: fake CPU, native GPU, graph AOT and external service provider on same TaskRequest, no duplicate StateRoot publication.
- MM-11: long media streams with bounded memory, cancellation replay, backpressure, physical lease/event fencing.
- MM-12: support status distinguishes recognized source vs mathematical reference vs CPU running vs per-kernel GPU vs full-model GPU vs real-time qualified vs opaque provider capability.

**Architecture acceptance law:** adding a source-verified new audio codec, TTS generator, video predictor, low-bit tensor format or typed decision head must not require rewriting the central Router, TaskCoordinator, BufferLease/DeviceBackend event authority or HTTP inference runtime. Extend only narrow source model/codec/TaskDriver/region providers.

## MM-11. Implementation order without derailing P07

- P07 GPU native Qwen3.5 specialization proceeds as planned; do not preempt with arbitrary media work.
- MM0: design/reference atlas, typed ValueDescriptor+Timebase+TrackFrontier schemas and fake source fixtures.
- MM1: one non-token task (encoder/typed decision) and one multi-step flow/denoise fake task under the existing coordinator; test effect/failure invariants.
- MM2: one small real ASR (Whisper or Parakeet), Kokoro TTS; then a small open flow/DiT image model such as FLUX.2 Klein where feasible.
- MM3: dual-codebook TTS, Qwen3 Omni, Wan/autoregressive video and synchronized joint-audiovisual generation, based on actual checkpoint/hardware/resources.
- MM4: Full ternary activation transforms + exact source packing and remaining specialist hardware providers.
- Opaque proprietary task adapters can be added independently above native execution, with explicit credential/user-consent/data-egress policies.

The **implementation reality** today is the native Rust Qwen3.5 CPU model, independently passed Phase 1 8K source qualification, and source-correct P07-A partial CubeCL GPU region implementation pending physical tests. **None of the above image/audio/video/decision/ternary models are implemented in Zerogravity today.**

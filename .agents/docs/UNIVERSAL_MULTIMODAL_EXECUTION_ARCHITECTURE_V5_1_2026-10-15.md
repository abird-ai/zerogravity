# Zerogravity Universal Multimodal Execution Architecture — v5.1 Design Specification

**Date:** 15 October 2026 (Asia/Singapore)  
**Status:** Comprehensive, implementation-ready DESIGN PROPOSAL; no new runtime/code/hardware qualification.  
**Lineage:** v4.3 core execution authority + additive 14 October v5 multimodal proposal + 15 October modality/real-time research.  
**Authority:** This document is the preferred design for newly considered multimodal scope; the user explicitly requested design and documentation, **not a unilateral final approval of every illustrative ABI choice**. Where conflicts exist, preserve current user decisions and the v4.3 single-owner invariants.  
**Companions:** REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md, VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md, the detailed research tracks under .agents/plans/multimodal-research, and MULTIMODAL_V5_1_ACCEPTANCE_AND_ROADMAP_2026-10-15.md.

## 0. Nonnegotiable goals and non-goals

### Goals
1. A **single Rust-authored execution fabric** that understands verified mathematical semantics rather than guessing algorithms from model family names.
2. Text generation, text embedding, ranking, finite schema decisions, images, image editing, audio ASR/translation, TTS, full-duplex speech, video, synchronized AV, native multimodal perception and multimodel compositions all representable using one small set of contracts.
3. Per-model and per-stage **best-legal specialization** on CPU SIMD, CUDA, ROCm/HIP, Vulkan, Metal, suitable NPUs, static-dataflow/AOT accelerators and allowed remote peers, subject to source math, numerical and physical resource proofs.
4. Streaming with independent clocks and bounded backpressure; explicit customer-facing distinction between prepared, committed, delivered and played media.
5. One process may register many distinct models, load many within physical quotas, share immutable component weights and execute independent mutable TaskSessions without leaking source/model or user state.
6. Portable CPU mathematical reference path wherever a model is claimed runnable; unsupported source operations are typed refusal, not silently delegated to unrelated inference products.
7. Native built-in model router and unified CLI/SDK/HTTP/realtime/job serving; an adapter-specific API protocol MUST NOT become an execution engine.

### Non-goals
- No automatic claim to reproduce proprietary model internals: GPT-Realtime, Gemini Live, hosted image/video/TTS generation APIs and proprietary decision-service APIs are opaque external services unless verified locally executable sources become available.
- No replacing all media tasks with the same token-AR loop, and no equating CTC, RNN-T, diffusion, rectified flow, codec RVQ and direct decision heads.
- No four independently serialized intermediate representations. Stable semantic source program and prepared executable bundle remain the two durable identities; region/kernel/command views stay compiler-internal.
- No general user-programmable agent workflow DSL in a kernel worker. Privileged retrieval/tools/action execution belongs in a separately permissioned application orchestrator.
- No implicit remote credential forwarding, unknown checkpoint code execution, downloaded model weights, vendor runtime installation, networking or services as a side effect of reading a checkpoint.
- No forced synchronous CPU roundtrip per token, acoustic frame, video frame or denoising microstep; prepared Device-native step islands may advance locally within validated effect/budget bounds.
- No target model substitution, changed quantization, sparse attention, solver schedule, codec resampling, guidance or answer-calibration without a separately named numerical/semantic policy and evidence.

## 1. Architectural reconciliation and ownership boundaries

**Keep the v4.3 vocabulary**, extended conservatively:

| Authority | Owns | Must never own |
|---|---|---|
| ModelCatalog / immutable ArtifactManifest | exact checkpoint files, hashes, tokenizer/media processors, source semantics, artifact versions, license and trusted components | user sessions, networking, GPU command readiness |
| ModelDefinition + ModelAssembly | verified source component graph, tensor roles, port schemas, exact mathematical definitions and codec/transform identities | ad-hoc family-specific HTTP server |
| SemanticProgram (SIR) | semantic ops, tensor axes, modalities, mathematical equivalence, numerical constraints and state effects | driver pointers, sockets, mode routing |
| TaskDriver | coarse task-specific stage order, algorithm, iteration, checkpoints, output validity and stop conditions | physical allocator, remote authentication, general user workflows |
| RegionExecutionProvider / DeviceBackend | semantically closed and source-qualified CPU/GPU/NPU/AOT region candidates; physical buffers, command stream, completion events | StateRoot publication, automatic model substitution |
| ExecutableBundle | immutable prepared region candidates, guarded shape/codec/numeric variants and event/lease command templates | mutable session state or per-packet model search |
| PhysicalResourceArbiter / DeviceWorker | coarse global physical admission and fast local bounded allocations, deadlines/fairness and real command retirement | semantic model policy, fake ready events |
| TaskCoordinator | one logical writer per TaskSession or authorized shard; state root publication, media commit groups, epoch and output-order authority | raw driver reclamation before hardware quiescence |
| DeliverySession / PresentationLedger | transport output sequence, packet ACK, playback evidence, reconnect/replay cursors | altering neural model state; claiming ACK means listened/viewed |
| ModelRouter | exact model binding, task capability, replica/placement selection at safe boundaries | hidden hot-path per-token scheduling, hidden degradation |
| TaskServiceAdapter | opt-in provider API with async jobs, privacy, credential and billing policy | local DeviceBackend and claims of native model math |
| Client protocol adapter | external OpenAI/Anthropic/native/WebSocket/WebRTC/SIP semantics | model-specific numerical definitions, StateRoot writes |

A set of logical authorities does **not** require separate OS processes or crates. Initial implementation may be cohesive Rust modules and a few small traits. No universal abstract object per PCM sample or per GPU op.

## 2. Type-safe multimodal values

The existing v5 ValueDescriptor is adopted with explicit validation and ownership fields.

~~~rust
// Illustrative type-level design, NOT compiled Rust/API compatibility promise.
enum ValueDescriptor {
    Tensor(TensorDesc),             // logical rank/axes/layout/dtype/shape
    TokenSequence(TokenDesc),       // text/image/acoustic/interleaved units
    CodebookSequence(RvqDesc),      // time index × depth index, completion policy
    AudioPcm(PcmDesc),              // rate, integer/float format, channels
    AudioFeatures(FeatureDesc),     // mel/STFT embeddings, windows/overlap state
    VideoFrames(VideoFrameDesc),    // dimensions, pixel/color space, PTS/DTS
    ImagePixels(ImageDesc),         // crop, alpha, ICC, color space, gamma
    MediaLatents(LatentDesc),       // spatial/temporal axes and VAE identity
    EncodedPackets(EncodedStreamDesc), // codec config / framing / extradata
    Transcript(TranscriptDesc),     // speaker/timing/confidence/finality
    DecisionScores(ChoiceSchemaId), // logits and calibration identity
    Structured(SchemaId),          // strict validated result shape
}
struct RationalRate { numerator: NonZeroU64, denominator: NonZeroU64 }
struct TimePoint { tick: u64, domain: ClockDomainId }
struct TimedValue { value: ValueRef, capture: Option<TimePoint>,
                    presentation: Option<TimePoint>, duration: Option<TimePoint> }
~~~

**Semantic invariants:**
- A text token count is never an audio sample count; acoustic codebook depth is neither audio time nor video frame number.
- Clock domain MUST name source origin, epoch, rational rate, optional delay and resampler rounding residue. Use checked u128 intermediate products or equivalently proven exact arithmetic; no cumulative floating point drift.
- Audio sample rates are source-specific (e.g. 16k, 24k, 44.1k, 48k), with validated PCM bit depth, endian and channel ordering. Codec packet timestamps have an independent rate.
- Video 24000/1001 and 30000/1001 PTS values require exact rational comparisons with audio timestamps and explicit continuity/discontinuity markers.
- Color conversion preserves declared pixel format, primaries, transfer curve, matrix, range and alpha handling; latent VAE channels and temporal compression MUST carry verified decoder identity.
- A latent is not a playable image/frame. An incomplete RVQ depth group is not playable audio. A partial/stale transcript is not equivalent to a committed transcript.
- Tensor dimensions are bounded and checked for rank, strides, byte multiplication and allocation pressure before host/device submission.
- Value references carry a lease/lifetime and immutable content identity; packets carry transport-specific delivery references without changing underlying model meaning.

## 3. ModelAssembly: physically separable, semantically unified

~~~rust
struct ModelAssembly {
    assembly_key: VerifiedAssemblyId,       // manifest hash and semantic revision
    components: Vec<VerifiedComponent>,     // each hash/license/config/weights pinned
    typed_edges: Vec<TypedPortEdge>,        // validated descriptor and clocks
    capabilities: TaskCapabilitySet,        // truthful per mode/transport
}
struct VerifiedComponent {
    role: ComponentRole,                     // model-specific, not global name heuristic
    artifact: VerifiedArtifactId,
    source_adapter: SemanticAdapterId,
    processor: Option<VerifiedProcessorId>,
    state_schema: ComponentStateSchema,
    inputs: Vec<PortSchema>, outputs: Vec<PortSchema>,
    source_license: LicenseAndAccessReceipt,
}
struct TypedPortEdge {
    source: PortId, destination: PortId,
    value: ValueDescriptor,
    transform: Option<VerifiedTransformId>,
    effect: StateEffectContract,
}
~~~

A component may be a text/vision/audio encoder, tokenizer, semantic decoder, expert bank, flow predictor, latent decoder, G2P, VAE, neural codec, vocoder, schema head, temporal synchronizer, refiner or upscaler. The list is not a promise of implemented kernels. Treat an assembly as **one exact source model** when its author defines components jointly, even if placed across separate devices.

**Not the same as:**
- a group of replicas serving the same logical model;
- a cross-model workflow combining separately registered ASR → LLM → TTS;
- user action/tool orchestration outside the inference engine.

Per-edge legal conversions are explicit: cast/layout, VAE encode/decode, resample with delay/remainder, tokenizer, feature extraction, spatial resize, normalized schema conversion. Unknown conversions are rejected, never guessed from matching tensor widths.

Assembly invariants:
1. Validate the complete component manifest and license against requested execution. Do not assume checkpoint directory structure implies semantic graph.
2. Version every processor/tokenizer/G2P/codec/vocoder independently; immutable assembly identity binds all required revisions.
3. Allow component-level residency and placement without splitting logical state publication.
4. Each source role has a unique verified model+component+adapter key; don't confuse a shared conditioner with a replica.
5. When sharing an immutable encoder across different models, allow only if exact source weights, media processor semantics and numerical policy are identical.
6. Treat captioner, prompt rewriter or safety model as explicitly optional/required assembly components, not hidden side effects.

## 4. TaskDriver families and termination

A TaskDriver describes algorithmic **control**, not device kernel code. Candidate task families:

| Driver | Main stages | Critical state | Valid output frontier |
|---|---|---|---|
| CausalText | tokenize, prefill, decode, verify/speculate, sample | KV, SSM/GDN, grammar, RNG, history | accepted token/text boundary |
| EncodePoolRank | frontend, encoder, pooling/reranker | source attention, normalization, batch | vector/score with schema |
| WhisperASR | STFT, mel, encoder, AR text decoder, merge windows | acoustic overlap, prompt/task headers, timestamps | committed transcript window |
| CtcSpeech | feature frontend, encoder, CTC collapse + align | blank/beam/timestamp state | stabilized transcript segment |
| TransducerSpeech | streaming FastConformer, predictor/joiner or TDT duration | per-chunk acoustic and decoder caches | emitted hypothesis with revision/finality |
| DeterministicTTS | phonemes, align/duration/pitch, acoustic network, vocoder | speaker embedding, align, vocoder overlap | complete PCM/encoded chunk |
| CodecAutoregressiveTTS | slow temporal codebook AR, fast RVQ depth, codec | time × codebook state, prosody/style, codec | complete reconstructible codec group |
| AcousticFlowTTS | text→acoustic context→flow predictor→codec | solver RNG/timestep, chunk overlap | complete decoded acoustic chunk |
| DuplexAudio | input acoustic encoder and simultaneous text/output codec lanes | capture, playback, turn epochs, echo/barge-in | model+playback-valid chunk |
| LatentImageFlow | conditioners, seeded latent, solver/guidance, VAE | timestep, noise/CFG RNG, latents | completed decoded image |
| ImageARCodec | interleaved multimodal AR, image codebook, decoder | token/codec positions and verification | complete frame/image unit |
| TemporalVideo | spatial-temporal latents, solver, 3D VAE, mux | tile/temporal cache, frame cadence | valid ordered frames/segments |
| JointAudioVideo | dual coupled latents and synchronized solver/codec | independent clocks and solver state, A/V barrier | legal AV commit group |
| TypedDecisions | encode, finite-choice logits/head, calibrate if justified | question/option schema and normalization | atomic answer set |
| ExternalTask | request serialization, async provider job, receipt | job/retry/idempotency, privacy, cost | provider-verified result |

**Algorithm-specific mandatory details:**
- CTC: blank semantics, collapse rules, repetition, alignment and optional LM fusion.
- RNN-T: predictor/joiner state, hypotheses, beam semantics; TDT: duration outputs and timestamp contracts. Must NOT be normalized to Whisper token decode.
- Neural speech codec: frame rate versus RVQ depth, source-trained quantizers, delay, overlap, packet reconstruction and voice/speaker conditioning.
- Rectified flow vs diffusion: predictor parameterization (epsilon, velocity, x0), sigma/timestep schedule, solver integrator, guidance/CFG/STG, seed/Philox-like RNG identity, count, step ordering and precision. Altering these changes the model task.
- Video temporal: sparse tile/selective masks and 3D RoPE/lattice, temporal VAE receptive field, tile overlap, frame latency, super-resolution branches.
- Decision: closed schema, raw option logits and only genuinely calibrated probabilities, joint multi-question inference; never invent probability calibration or simulate with free-text JSON.
- Compound multimodal: known bounded task joins inside trusted TaskDriver; no implicit arbitrary tool execution.

## 5. StateRoot, multitrack commit and physical event authority

Keep exactly one TaskCoordinator-authorized PublishRoot per TaskSession epoch (or explicitly authorized state shard). StateDomain implementations PREPARE versions only. An audio writer, codec decoder, transport thread, DeviceWorker and remote peer cannot independently publish the model's logical session state.

TaskFrontier extends from scalar token position to a vector:
- text accepted-token cursor and grammar/sampler/hidden history,
- recurrent/KV and prefix/branch version,
- input audio sample cursor and frontend overlap,
- output acoustic frame and **RVQ depth completeness**,
- ASR hypothesis revision and stable transcript segment,
- image/video solver iteration and latent version,
- video temporal tile and frame PTS,
- audio/video decode progress and playable commit-group watermark,
- typed-decision schema generation,
- external stage committed result identity.

~~~text
Committed Session StateRoot
  -> provisional sub-steps/regions, potentially on many devices
  -> physical events prove each submitted dependency completed
  -> typed StateDomains prepare compatible versions
  -> model/task algorithm validates complete output boundary
  -> TaskCoordinator compares epoch + expected predecessor root
  -> ONE PublishRoot
  -> committed output sequence becomes eligible for transport
~~~

**Do not conflate logical commit with physical RAM/VRAM DMA completion or with successful client delivery.** Physical handles remain leased until genuine readback/event retirement. Unknown cancellation quarantines indefinitely until proved device quiescence; the global device card need not be canceled because one session requests barge-in if its other sessions are still active.

Synchronized A/V can publish a joint legal group only when its declared audio and video track watermarks, solver iteration, codec state and timestamps are compatible. Permit legitimately independent output lanes only when the exact model/protocol contract permits them. Partial video denoise latents or incomplete codec codebooks may never be labeled final media.

## 6. Numerical/specialization legality dimensions

Specialization is orthogonal across **FOUR** axes:
1. **Semantic**: attention variant, GDN/SSM, transducer predictor/blank, diffusion/flow parameterization, sparse temporal mask, autoregressive codec, multi-head schema scoring.
2. **Physical**: CPU, CUDA/HIP/Vulkan/Metal/NPU/AOT; tensor dtype/packing, FP64/FP32 reduction, tile/warp subgroup, captured graph, PCI fabric, stream/event scheduling.
3. **Task control**: prefill/decode, RVQ frame/depth, CTC beam, flow solver step, video window, low-latency bidirectional audio, decision batch.
4. **Placement/QoS**: assembly component residency, async transfer, memory cap, calibrated soft deadlines, fairness, remote consent/data locality.

Candidate legality key (conceptual): artifact+assembly revision, semantic graph/op hash, exact tensor codec and activation transform, numerical policy, shape/stride and state-effect schema, precision reduction order, device driver/compiler/ISA, resource and event plan, required request QoS and verification EvidenceId.

Kernels can fuse or replace a closed region ONLY if source semantics/effects/numeric contract and guarded shape match. An unqualified model or media transform returns a typed unsupported/failed operation, not silent approximation. For user-selected approximate quality modes, report provenance and measurable distribution/quality differences; never label those equivalent.

## 7. TensorEncoding and novel low-bit models

The stored weight representation does not define model topology. Example: a ternary 27B model maintains a Qwen-derived hybrid model architecture but needs PTQ1_0/PQ2_0 packed ternary weights, 128-group scales and matching 1024-block Hadamard rotation on runtime activation. Loading only its Qwen topology and treating these formats as regular Q2 is incorrect.

~~~rust
struct TensorEncoding {
    source_format: VerifiedFormatId,
    weight_codec: WeightCodecId,
    block: BlockGeometry,
    scale: ScaleCodecId,
    activation_transform: Option<RegisteredTransformId>,
    activation_quant: ActivationCodecId,
    accumulation: NumericReductionContract,
}
~~~

Codec/transform/shape/source identity are all verified at load and plan selection. A backend can execute packed ternary dot directly; it need not expand to FP16 weights merely because reference semantics are expressed independent of packing. Equally applicable to FP4/NVFP4, indexed memory, RVQ audio embeddings and future quant families, but no blind numerical fallback.

## 8. Media component placement, resource accounting, and scheduling

One global physical arbiter and device-local bounded subpools allocate weights, state, H2D/D2H, pinned codec buffers, attention/video latents, replay snapshots and queues. Global admission is coarse and infrequent; per-step stage scheduling uses prepaid envelope leases without a global lock or snapshot allocation per audio frame. Kernel completion and actual driver retirement authorize physical reuse.

**QoS kinds**:
- RealtimeSoft: bounded deadlines, jitter, first audible token/codec chunk, short device quanta, request preemption at validated safe boundaries, explicit admission refusal if no budget.
- Interactive: chat, decisions, short synthesis, prioritized low queue time.
- StreamProgressive: bulk ASR, long TTS, progressive valid output and bounded buffers.
- BatchMedia: images/videos, throughput and joules/quality, denoise-step scheduling, VAE tiling and offload.
- Background: prefetch/tuning/quantization only when non-interfering.

No preemption inside an unconfirmed physical kernel unless backend has proven physical cancellation and state rollback. Deadline misses are observable, not grounds to mutate model semantics. Adaptive placement must preserve the client's exact artifact/numerical/source license, unless a verified policy explicitly permits substitution and records it.

## 9. Distinct stages of composition

A. **Intra-model assembly**: FLUX [dev] conditioner → flow DiT → VAE; audio temporal/depth/vocoder; Qwen-Omni Thinker/Talker; a coupled A/V model. These are one verified assembly where source declares joint weights/state.

B. **Multi-model inference task**: Whisper → Qwen → Kokoro; embedding+rerank+decision; independent image enhancer plus video generator. Separate RegisteredModel identities and output tokens/leases are joined by a trusted bounded task policy or external orchestrator. Each stage must declare exact ports and privacy budget.

C. **Application/agent workflow**: retrieval, tool calls, approvals, external API, business decisions. Authorization/side effects remain above TaskRequest, not inside a GPU kernel or arbitrary TaskDriver callback. Device/computation may remain in the same native engine.

## 10. Artifact/model capability status and truthful UX

A catalog must distinguish:
1. Recognized name or format,
2. complete verified semantic recipe,
3. runnable exact native CPU reference,
4. source-qualified native GPU subregion(s),
5. complete original model/task qualified on device,
6. streaming/playable result certified,
7. soft-realtime objective measured under declared load,
8. external hosted provider capability with auth/consent.
A model card may simultaneously have several statuses by task, variant and backend. Never infer full model support from a successful Q4 matrix projection or a v5 mock.

## 11. Failure classes and recovery

| Error | Source action | Physical resource | Public behavior |
|---|---|---|---|
| Unsupported semantic op/codec/shape | fail or choose proven equivalent reference | no launch | precise Unsupported response |
| Malformed source/input/clock/schema | refuse before submit | no device fence | 400-like validation error |
| Insufficient memory/credits | reject/admit smaller declared batch | no unsafe allocation | typed admission error with available safe options |
| Future canceled before device submission | discard queue ticket and provisional state | no device fence | canceled request |
| Future canceled after possible physical command | mark unknown, quarantine buffers/leases | fence affected physical service until proven quiescent | no false Ready, no stale publication |
| Model numerical non-equivalence | source-grounded failure, no silent epsilon increase | safe completed buffers eventually retire | unsupported/quality-gate failure |
| Transport disconnect | retain bounded committed output cursor, cancel or pause by session policy | never assume client received media | reconnect policy / explicit terminal event |
| Playback interruption | suppress unplayed queued output and reconcile actual playback evidence | do not cancel other sessions/card globally | truncation/turn event |
| Remote provider timeout | status uncertain until provider job receipt, idempotent reconciliation | no local neural DeviceBackend state claim | job status unknown/retry policy |
| Partial coupled A/V solver failure | abort incompatible commit group, retain previous root | correct event/lease quarantine | no partial mislabeled final AV |

## 12. Complexity ceilings and future-proofing

- Do not add an enum variant to one universal hot-path router for each new model; use registered semantic schemas and TaskDriver selection at admission.
- Do not make a central interpreter evaluate general user-defined workflow graphs per media frame.
- Keep public interfaces small (verified ModelAssembly, TaskDriver, ValueDescriptor/Timebase, StateRoot, RegionProvider, ModeReceipt, native task stream).
- Keep model-specific semantics in one source adapter plus reusable operators; no separate model-serving subprocess.
- Component graph and stage control must be understandable by human maintainers and independent LLM agents without runtime-generated magic.
- No mandatory trait objects for each tensor or audio sample; capability registries and generics should compile into direct dispatch at prepared-plan boundaries.
- Separate reference math from fast kernel selection; staged evidence and model-specific numerical tests are non-negotiable.
- A genuinely new hardware class may execute AOT graphs, static dataflow or native kernels at its own granularity under one Event/Lease/PreparedRoot boundary.
- Document every additional persistent abstraction with a measured change-radius/latency justification and an explicit removal path if it fails.

## 13. Design acceptance (not yet implementation)

A new model/codec/TaskDriver must be introduced with changes confined to source adapter, task algorithm, semantic math, processor/codec and the best matching provider—not rewrite core Router, TaskCoordinator, BufferLease, or HTTP protocol handling. Test on source-free mocks first, then original checkpoint source, then actual hardware.

Required adversarial drills:
- Qwen-AR text and direct finite-scored decisions share state/runner but not output semantics.
- Whisper versus TDT streaming ASR share transport but not decoding algorithm.
- Kokoro without token LM; multi-depth RVQ audio; Moshi full duplex without intermediate text requirement.
- FLUX variants require correct conditioner; wrong variant, solver, guidance and VAE are rejected.
- Autoregressive multimodal image-model AR-versus-hybrid stages are based on inspected checkpoint, not family label.
- Wan2.2 phase-dependent denoising experts; Mochi unequal text/video projection widths; autoregressive video-model temporal tile masks.
- Mutually coupled A/V solver steps and exact clocks; arbitrary TTS-over-video composition is not equivalent.
- Packed ternary plus Hadamard; loading ternary weights without matching transform hard fails.
- Correct pre-submit rejection, physical cancellation quarantine, multi-model fairness and no false StateRoot publication.
- Host native inference versus independently permissioned external hosted jobs never conflated.
- A single request may include text+image+audio+video and valid typed output, with precise capability/transport refusal for unsupported combinations.

**Plan relation:** Existing MM-01..MM-12 remain governing initial drills; extend with explicit MM-13..MM-24 for real-time QoS, presentation, hosted job and API compatibility. See acceptance/roadmap document for exact fixtures and evidence gates.

## 14. Source references / provenance

Local normative prior documents: MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md; COMPOSITION_AND_EXTENSION_CONTRACTS.md; MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md; TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md; UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md; MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md; DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md.

External original-source research and specific variant cards are indexed in the modality dossiers and SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md. Source claims are dated to this research pass; repository updates or different checkpoint revisions must be reverified. Upstream open-source runtime code is prior art, not automatically authorized for embedding or redistribution.

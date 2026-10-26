# Zerogravity Multimodal v5.1 — Complete Implementation Roadmap, Dependency Gates and Acceptance Matrix

**15 October 2026 — MODEL COMPONENT STORAGE SUPPORT:** The [virtual tensor + SSD first-class contract](../docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) applies to immutable model components, video temporal VAE and frozen private session state without changing any model algorithm/codec/clock. Audio/AV high-performance realtime may require full resident components and MUST reject SSD placement if it violates measured QoS. SSD stage admission and actual completion are independent of playable media finality and TaskCoordinator StateRoot. Companion [SSD-P0..P5 planned acceptance](SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) has SSD-01..32; all planned, not executed, and do not supersede MM01–24.


**15 Oct additive extreme-text/quant implementation precision:** The source-bound [QTIP + MiMo/DFlash + gfx1151/UMA integration roadmap](EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) supplies **50 distinct planned QTIP/HYB/GFX acceptance fixtures** under existing P03–P22 without altering media MM-01..24 or any model implementation priority. Large video/audio/decision TaskDrivers remain independent; source artifact/codec+TensorEncoding/region kernel support remains orthogonal to each task. The specialist source atlas is source evidence, NOT hardware/quality proof for Zerogravity; vendor-specific memory page-cache and row grouping are candidate kernel/cost properties, not generic multimodal task semantics.


Technical implementation and acceptance reference. Live work and completed results
belong to [the multimodal TODO](../plans/multimodal/TODO.md), under
[TODO-SPEC](../plans/TODO-SPEC.md). The requirements below do not grant permission
to acquire checkpoints, invoke a provider or change devices. Real media/decision
support requires its original source, task/state and eligible-device gates.

## 0. Universal execution truth and success definition

The user requires broad, high-performance model/hardware support with strong specialization, simple UX, first-class multimodal/realtime serving, a built-in router and maintenance-light Rust-native architecture. A model is not "supported" by importing its name or format; support must be labeled at precise capability tier.

Support tiers (each scoped by model artifact, component assembly, task, backend, numerical mode, input shape and stream transport):
- **T0 Recognized:** model name/artifact format recognized, no source semantics qualification.
- **T1 Semantically verified:** exact source tensors/processor/ops/driver and reference recipes identified.
- **T2 CPU reference running:** trained checkpoint/source fixture and exact CPU task correctness including state effects.
- **T3 Partial accelerator:** selected closed semantic region physically executed with source-matched receipt; NOT whole model.
- **T4 Whole-model accelerator:** full task semantics/state root, original complete checkpoint, physical backend under exact numerical quality contract.
- **T5 Streaming/playable:** valid output media chunks and clocks/codec finality, state/cancel/teardown verified.
- **T6 Soft-realtime measured:** declared latency/jitter/memory/throughput under pinned hardware, load and transport, with admitted QoS.
- **TX External hosted:** provider endpoint and authorized lifecycle validated; never reported as native T2–T6 merely because API responds.

The catalog and CLI/SDK/API must show exact tier and missing gates, not a vague "multimodal supported" flag.

## 1. Scope and source authority

Preserve the user-directed breadth: standalone and native duplex speech, image
creation/editing, temporal video and coupled AV, finite decisions, unusual source
codecs, composable model assemblies, and high-performance serving in one Rust
engine. A direct neural duplex model and an ASR→language→TTS composition remain
distinct tasks. Hosted services expose authorized observable APIs, not invented
local weights or proprietary internals. No family name implies streaming,
commercial licensing, source-equivalent math or a runnable checkpoint.

Use the [retained source atlas](MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md),
the per-family dossiers below, the governing execution/assembly contracts and
current plan TODOs. Do not revive old baseline statuses or change the authorized
CPU-first model/acquisition priorities because a media design is ready. User
requirements remain distinct from assistant proposals and unresolved policy.

## 2. Design readiness gates MM-D0..MM-D5

### MM-D0 — source and design consistency
Deliver:
- v5.1 unified execution, model assembly/codec and real-time serving specifications.
- research dossiers for ASR/translation, TTS/duplex, image, video/AV, decision/ternary, protocol/provider prior art.
- U40/U41 explicit user decision provenance, research/design recommendation register and open options.
- independent reviewer reconciliation with remaining conflicts and falsification criteria.
- linked original-source references, unresolved decisions and modality-specific adverse fixtures in their retained technical owners; no duplicate handoff or checksum inventory.

Gate: contracts agree on one state/resource authority, algorithms/clocks and exact capability scope; references resolve and no model support is invented. A design consistency review does not qualify runtime behavior. Current TODOs record implementation and actual tests.

### MM-D1 — typed media and clocks (later CODE only if authorized)
- ValueDescriptor with exact sample/frame/codebook/latent/decision types and registered parser limits.
- ClockDomain, RationalRate, checked u128/sample/PTS/DTS conversions and residual resampler delay.
- independent per-lane capture/codec/solver clocks; strict invalid output finality.
- no new persistent IR; adapt existing SemanticProgram ValueType contracts.

Gate:
- all MM-01 and timebase cases; 44.1k/48k PCM to 16k/24k audio plus 24000/1001 and 30000/1001 video;
- no drift over millions of ticks, invalid packet format/clock fails, bounded buffers and overflow refuses;
- no dependence on GPU vendor runtime for source-free tests.

### MM-D2 — verified ModelAssembly and source codec registry
- source artifact + processor + per-component assembly manifest, typed ports, verified transforms, license/version/hashes.
- TensorEncoding independent of model topology, accumulation/activation transform, exact padding/shape and codec identities.
- source model can have separate conditioner/VAEs/vocoder or multiple text/audio heads without another EngineCore.

Gate:
- MM-02 typed components and port incompatibility tests;
- intentionally mismatched rectified-flow [dev]/[klein] conditioner fails;
- ternary codec without Hadamard fails;
- no arbitrary checkpoint Python execution;
- model capture variants and license ownership shown in receipts;
- one component resident across two sessions without mutable state leakage.

### MM-D3 — task algorithms and transactional multi-domain root
- trusted drivers for decoder token, encoder pooling, CTC/RNNT/TDT, codec depth AR, non-AR TTS, flow, video, schema choices.
- legal repeat/parallel-known/codec/mux stage control with source-defined iteration and state effects.
- TaskFrontier multi-axis and CommitGroup atomic publication.

Gate:
- MM-03 flow and MM-04 RVQ/ASR addition, MM-07 AV and MM-08 finite schema; wrong solver/codec/choice fails;
- interrupted device events never publish stale StateRoot;
- no arbitrary external action inside TaskDriver.

### MM-D4 — RealtimeSession and MediaEvent/PresentationLedger
- live input/output lanes, separate neural/transport/turn epochs, committed/delivered/acked/played watermarks;
- VAD/barge-in, revisioned transcript, output codec chunks and bounded buffers; per-session cancellation scope.
- make recording/ack knowledge honest when client isn't providing accurate playback feedback.

Gate:
- MM-06 duplex barge-in plus MM-13..MM-17 and MM-19; no cross-session GPU poisoning, no underflow/latency lie, no output media emitted before complete codec;
- sequence, generation, timer and clock assertions with deterministic mock Waker.

### MM-D5 — canonical serving and protocol conformance
- EngineCore reused by CLI, SDK, HTTP, SSE, WS, WebRTC, optional SIP gateway;
- per-model/per-transport capability profiles and refusal semantics;
- external TaskServiceAdapter asynchronous jobs with auth, privacy/egress, rate/cost/idempotency and result receipts.

Gate:
- MM-18..MM-24, official tested SDK API subset, malformed request/codec/media tests;
- accepted remote provider job timeout remains Unknown and reconciles without automatic double charge;
- no duplicate router/TaskCoordinator/DeviceBackend, no per-frame global hot-path snapshot or JSON-only binary transport.

## 3. Source-real implementation slices MM-I0..MM-I7 (proposed, sequencing requires approval)

### MM-I0 — small non-token model and codec mock under same core
Implement one encoder/finite direct decision head plus one iterative latent solver mock using source-free deterministic fixture. Reuse TaskCoordinator/ResourceArbiter/RegionProvider. Validate no change to actual Qwen CPU/GPU results. This is the first architecture *change-radius* proof, not a media product.

### MM-I1 — original ASR checkpoint, CPU first
Select one sufficiently small legally available Whisper or Parakeet source checkpoint. Register exact feature extractor/processor, model semantic op, ASR TaskDriver and transcript clock. CPU source golden tests, then GPU source-equivalent attention/conv/frontend on one physical backend. Publish accurately whether model supports offline only or real-time streaming. No RTT claims until measured.

### MM-I2 — Kokoro non-AR TTS, CPU first
G2P/phoneme, duration/F0/prosody, source vocoder. Source-defined PCM sample rate, complete codec chunk and real WAV output. This deliberately falsifies designs hardcoded around decoder LM. Validate source accuracy and speaker privacy and measure first audible sound. Reuse warmed source weights across sessions only under source/version match.

### MM-I3 — dual-AR/Qwen TTS depth AR and full-duplex mock
- First source-free RVQ nested time/depth and two-lane audio consumer mock; then licensed dual-AR or Qwen3-TTS original checkpoint as permitted.
- Prove source codebook, chunk/overlap, voice state and timebase.
- Add native full-duplex neural-codec or Qwen-Omni source model if feasible, with actual loaded input/output streams and interrupt handling; don't force text-only cascade.
- Hosted TTS adapter can proceed independently behind consent without native weights.

### MM-I4 — image flow generator and editing
Pick a small source-audited flow/DiT model (rectified-flow Klein if exact weights/license permit) and exact conditioner/VAE. CPU reference fixtures and actual GPU source region/whole-image validation, seeded solver/guidance, memory tiling. Then SD3.5 and Qwen-Image as variants, not one fake image backend. The autoregressive image MoE AR/hybrid source recipe only after careful generation code audit. A hosted creative-media service/GPT-Image adapters independent of local support.

### MM-I5 — video temporal / Wan expert switching
3D VAE and temporal latent reference, Wan2.1 small proof if feasible, then Wan2.2 high/low noise expert switching with source threshold. Source-correct model/video clocks, VAE overlap, tile/sparse attention and real end-to-end output. Add dual-stream video DiT 1.5 / asymmetric-attention attention/shape variants as verified components.

### MM-I6 — joint audio/video and frontier job workflows
Coupled A/V source stages, independent audio/video latents and synchronizing one commit group. Actual source/AV codec output and PTS sync evidence on feasible GPU(s), no post-hoc TTS masquerade. Separate hosted generation-service asynchronous TaskServiceAdapters, source/license/rights/cost consent, unknown timeout reconcile.

### MM-I7 — Ternary compression, specialist text, typed decisions at scale
- Ternary PTQ1/PQ2 + Hadamard transform original source golden reference, then CPU bitpacked/physical GPU legal specialization; no second Qwen engine.
- One open source-typed decision checkpoint; a shared Qwen backbone and trained schema head, source logits and versioned calibration.
- The hosted decision API adapter with consent and provenance, direct finite typed results.
- Additional low-bit/MTP/linear-attention families adopt same independent codecs, transforms and StateDomains.

**These later slices do not override user's already approved Qwen3.5→Gemma→Qwen CPU/GPU roadmap.** They are an extension program after or interleaved only when explicitly authorized and do not interrupt existing phase safety work.

## 4. Complete MM-01..MM-12 inherited acceptance matrix

| ID | Fixture and minimum observable assertion |
|---|---|
| MM-01 | exact rational 44.1k PCM↔29.97FPS clocks, correct residual over long durations |
| MM-02 | verified conditioner→generator→decoder ports; wrong processor/shape/format refused |
| MM-03 | 20-step dummy flow with seed, RNG/guidance and per-step atomic StateRoot |
| MM-04 | Dual-AR slow temporal ~21Hz × ten RVQ codebooks; only complete frame can decode |
| MM-05 | Whisper encoder/AR vs Parakeet CTC/RNNT/TDT blank/duration/timestamps |
| MM-06 | full-duplex input/out + barge-in/epoch and device lease/mock completion fencing |
| MM-07 | Coupled-style synchronized two-track A/V publication and failure |
| MM-08 | multiple typed finite schema decisions atomically scored, nonfinite/refusal |
| MM-09 | Ternary PTQ1/PQ2+Hadamard exact codec mock and source shape guard |
| MM-10 | one TaskRequest supports fake CPU/GPU/AOT/hosted with distinct truth tiers |
| MM-11 | long streams bounded, partial playback/cancel/rollback and event/resource safety |
| MM-12 | recognized vs CPU-only vs GPU-region vs full GPU vs realtime vs hosted capability status |

Existing synthetic foundations are in `crates/zerogravity/tests/multimodal_mm0.rs`. The matrix is a requirement inventory, not a new test result. Check current code and the multimodal TODO before extending it; synthetic clocks or dummy flow do not qualify trained media.

## 5. MM-13..MM-24 new realtime/API acceptance matrix (all not implemented)

| ID | Deterministic scenario | Required assertion |
|---|---|---|
| MM-13 | Native duplex audio while response streaming; VAD barge-in | old output stops, input state continues, unrelated GPU sessions unaffected |
| MM-14 | Playback ACK from prior turn arrives after turn epoch changed | ignored; never rewrites new played watermark |
| MM-15 | 48k/44.1k PCM and 30000/1001 video PTS over hour | exact residual/rounding, no invented integer FPS |
| MM-16 | Transport acknowledges packet while GPU command still outstanding | physical BufferLease remains until verified Ready |
| MM-17 | Slow WS/SSE media client with tiny credits | bounded queue/memory, backpressure/admission error not OOM |
| MM-18 | 10 live speech sessions plus 5 large video solver jobs | fair physical admission and honest missed-SLO, no silent quant/solver changes |
| MM-19 | WS connection dies and reconnects with duplicate cursor/event | bounded idempotent replay of committed media, source auth and epoch check |
| MM-20 | Remote hosted-video-service-like POST accepted but response times out | status Unknown; reconcile ID, no automatic duplicate billable create |
| MM-21 | Model capability supports TTS but not realtime interruption/voice clone | unsupported transport/task fields explicitly rejected |
| MM-22 | Every audio packet goes through prepared path | no global catalog/router/arbiter snapshot allocation per frame |
| MM-23 | Coupled A/V joint solver step with audio lagging behind video | no video/audio final group until declared ready/compatible |
| MM-24 | malformed media path, hostile dimensions/codec and unauthorized voice egress | rejected before GPU allocation or outbound provider call |

Mock tests are **precondition** for source-real media implementation; they are not evidence of source model quality or commercial product readiness.

## 6. Model-specific detailed acceptance suites

Research dossiers define named tests:
- MM-ASR-01..12 in ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md.
- MM-TTS-01..12 in TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md.
- MM-IMG-01..12 in IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md.
- MM-VID-01..15 in VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md.
- MM-DEC-01..06 and MM-BON-01..08 in DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md.
- MM-SRV-01..12 in SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md.

These roughly 77 more focused fixtures supplement MM01–24, not replace them. An agent must import *all* relevant suites for a source model, and explicitly classify source-free mock, original source trained CPU, physical GPU, hosted API, or soft-realtime measurement evidence.

## 7. Source binding and review obligations

For each actual model/variant, identify original config/tensors, tokenizer,
processor, codec, scheduler/solver and licenses; keep code, weights, voice/G2P
assets and provider rights separate. Model/content identities remain required.
Source code is reference input, not executable checkpoint Python. A source matrix
must say which processor/codecs/checkpoint APIs are still unverified.

Keep the distinct gates from the retired review briefs:

| Slice | Required source trace and negative cases |
|---|---|
| ASR/translation | Whisper window/encoder/AR; CTC blank-collapse; RNN-T predictor/joiner and TDT duration advances. Check frame/subsampling clocks, streaming checkpoint lookahead, language/task headers and provisional diarization corrections. Offline FastConformer is not automatically streaming; encoder classification is not CTC. |
| TTS/duplex | Non-AR phoneme/duration/prosody/vocoder, nested time/depth RVQ and truly simultaneous codec input/output traces. Bind G2P/voice assets separately; incomplete codebooks, stale overlap or changed voice/codec generation cannot become audible PCM. |
| Image/editing | Check exact conditioner, latent packing, VAE/crop/color/mask, seed/guidance and source solver. Dev/distilled conditioners and schedules differ. Audit the selected AR/hybrid image generation entrypoint instead of choosing AR or diffusion from its name. |
| Video/coupled AV | Check temporal 3D-VAE stride/overlap, rectangular attention, sparse tile selection and timestep expert transitions. Audio/video cross-attention and independent clocks must meet their joint commit group; unrelated TTS over video is a compound task. |
| Decisions/low-bit | Check direct option/joint-schema head, option order, null/refusal and calibrated versus raw scores; text JSON is not equivalent. PTQ1_0/PQ2_0 group scales and the source1024 Hadamard/sign transform must match; ordinary Q2 is not a valid substitute. |
| Serving/providers | Check model/task/transport field conformance, prepared binary packet path, bounded credits, reconnect epochs and actual playback uncertainty. Ambiguous accepted remote jobs stay Unknown until reconciled; no duplicate billable create or implicit egress. |

Each implementation review identifies private state and its sole publisher,
source/numeric reference, clocks/finality, cancellation/readback/lease retention,
missing support tier and extension change radius. Adding a model must not require
a second router, allocator, runtime, arbitrary workflow DSL or public plugin ABI
per operator. Record actual command/result and failures in the plan/run TODO;
ordinary evidence-file hashes, archive construction and generated status copies
are not acceptance requirements. Physical or provider tests remain required for
their own claims; documentation or a compiled kernel is insufficient.

## 8. Perf/maintainability gates

Test both speed and extension cost:
- Throughput: GPU utilization, inference tokens/s, image steps/s, video frames/s, TTS audio seconds/s with exact model and output quality, not meaningless mixed-unit throughput.
- Tail: TTFT/TTFA, ASR lag, user barge-in silence delay, audio packet dropout, video first playable frame, P50/P95/P99 by load/transport and thermal regime.
- Memory: weights+latent state+codec+buffers+peak driver allocation/fragmentation, UMA/GTT vs resident physical bytes.
- Multi-session: start/stop/stale session isolation; source weights shared where exactly same artifact; no shared sampler/SSM/acoustic state corruption.
- Routing: exact model only by default, no hidden downgrade; per-replica placement sticky state and safe migration.
- Change-radius: adding one new codec/head/task provider should not edit the central Router/TaskCoordinator/serving socket code; document any exception.
- Build: useful CPU-only local Rust binary without extra CUDA/ROCm/Python prerequisites; compiler/vendor features optional.
- Simplicity: all new interfaces are expressible in bounded model+task+physical and streaming contracts; no plugin trait per sample and no unnecessary service microprocess.

## 9. Dependencies and stop rules

| Work | Depends on | Stop if |
|---|---|---|
| Universal typed media values | source ValueDescriptor/Timebase proposal | raw float time can't be made exact, risk of ambiguous clocks |
| ModelAssembly | artifact hash/config processors/licensing, semantic adapters | source component cannot be verified or license disallows use |
| TaskDriver | source-stable model algorithm and reference oracle | task fallback changes answer meaning |
| GPU specialization | real hardware/support, physical buffer/effect leases, numeric source proof | kernel produces unapproved distribution drift |
| Realtime serving | typed committed MediaEvent + bounded presentation state | client/transport can commit neural model state |
| WebRTC/SIP | media auth/transport/clock/codec/ACK contract | media transport exposes credentials or unbounded buffers |
| Hosted providers | explicit allowed egress, cost, credential, idempotency | source model is opaque but claimed local, or provider job replay ambiguous |
| Full AV generator | video/audio decoder and joint latent source graph | source cross-attention ignored or timestamps don't align |
| Ternary compressed | source codec+Hadamard validated | activation transform absent |
| Whole native GPU model | all source semantics, recurrent/KV/multimodal and state actually on GPU | merely one Q4 region or fake Ready source-only evidence |

**Release gate:** a feature cannot move from proposal to "supported" without exact source math, output and state/clock verification and device/API evidence. The user has not approved any numerical latency SLA, provider account linking or proprietary weight substitution.

## 10. Implementation ownership and pending policy

Assign narrow source/driver/codec/provider work to disjoint owners; the coordinator
integrates reviewed changes and test results. Use the current multimodal TODO and
TODO-SPEC, not copied agent briefs or a static aggregate. Named research lanes are
not evidence that independent reviewers ran, and no future test is passed by
writing its specification.

Unanswered choices remain explicit: first licensed feasible ASR/TTS/image/video
checkpoints; source-bit versus a separately approved media-quality policy; exact
AR-image entrypoint; temporal VAE/memory-fit limits; physical headroom/quiescence;
measured mouth-to-ear/first-output/jitter objectives; and packaging of optional
transports or future component ABI. None follows from model marketing or user
examples of API names. The detailed live-session questions are in
[the realtime policy section](../docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md#18-open-policy-decisions-before-live-serving).

## 11. Technical references

- [Universal execution](../docs/UNIVERSAL_MULTIMODAL_EXECUTION_ARCHITECTURE_V5_1_2026-10-15.md), [verified assemblies](../docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md), [realtime QoS](../docs/REALTIME_MULTIMODAL_SERVING_AND_QOS_V5_1_2026-10-15.md).
- [ASR](multimodal/ASR_AND_STREAMING_SPEECH_UNDERSTANDING_2026-10-15.md), [TTS/duplex](multimodal/TTS_VOICE_CODECS_AND_FULL_DUPLEX_2026-10-15.md), [image](multimodal/IMAGE_GENERATION_EDITING_AND_FLOW_MODELS_2026-10-15.md), [video/AV](multimodal/VIDEO_TEMPORAL_MODELS_AND_SYNCHRONIZED_AV_2026-10-15.md).
- [Decisions/ternary](multimodal/DECISION_MODELS_TERNARY_AND_SPECIALIST_TEXT_2026-10-15.md), [serving prior art](multimodal/SERVING_API_AND_REALTIME_PRIOR_ART_2026-10-15.md), [adversarial reconciliation](multimodal/CROSS_STREAM_ARCHITECTURE_RECONCILIATION_2026-10-15.md).

The original authors' source/license references in those owners remain intact.
These are technical research and future gates, not claims of native trained media
execution or completed API/realtime acceptance.

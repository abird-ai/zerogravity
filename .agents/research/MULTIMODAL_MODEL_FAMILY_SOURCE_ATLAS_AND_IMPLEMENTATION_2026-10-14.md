# Zerogravity — Model-family research atlas and multimodal implementation roadmap

**14 October 2026 (Singapore).** Research-only model capability inventory and design decisions.
**Companion:** [Proposed unified multimodal execution contract](../docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md).
**Truth:** model names, released weight status, numerical method and API contracts are source-specific and time-sensitive. The external sources below identify the models; Zerogravity **does not currently run any listed media/decision/ternary checkpoint natively**. Only native Qwen3.5 0.8B and 2B CPU inference/8K acceptance and source-correct partial GPU region code exist. No model weights were downloaded.

## Research method and source hierarchy

Prefer (1) model authors' official open checkpoints, source repositories and papers, (2) upstream framework inference implementations, and (3) original vendor API contracts. A vendor statement about image quality/latency or license is not an independently reproduced Zerogravity result. Proprietary hosted model details below describe **API behavior**, not verifiable internal neural math. Some names have several architectures/variants; identify the **specific model checkpoint and release**, never just the family string. Multimodal timestamps/codecs can change independently of neural weights; pin both.

### Image generation / editing

| Model and availability | Source-backed execution family | Requirements for verified native model support | Primary source |
|---|---|---|---|
| **Qwen-Image 20B**, open-weight | Multi-modal Diffusion Transformer (MMDiT), conditioned latent image generation and editing; text rendering; text and image processors | MMDiT dual streams, image VAE and processor, text conditioning, sampler/flow schedule, guidance, multi-image editing, source diffusion noise | [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) |
| **Rectified-flow image model [dev] 32B**, weights under a non-commercial license; **[klein] 4B** Apache-2.0 (9B klein different license) | Variant-specific flow models: **[dev] uses a Mistral-class image/text conditioner, while [klein] 4B uses a Qwen3-4B embedding conditioner**; latent flow transformer + VAE, image references, and differing distillation/schedules | Bind conditioner/embedding type and dimensions to the exact checkpoint (do not reuse dev's Mistral configuration for klein's Qwen3 embedding), independent residency, latent packing, model-registered flow solver and guidance, VAE, source step budget | — |
| **Autoregressive multimodal image MoE / Instruct / distilled**, source and weight availability vary by variant | **The vendor describes a unified autoregressive multimodal 80B MoE** (64 experts, about 13B active), but **the exposed image-generation source path requires deeper auditing before declaring it purely AR versus diffusion/hybrid stages** | Register exact underlying image-generation stage semantics only after following the official generate_image and source causal-MM implementations; support both AR token/codecs and denoising components where actually present, rather than hardcoding either | — |
| **Stable Diffusion 3.5 Large/Turbo/Medium**, open-weight release under a source license | MMDiT-X / QK normalization, Large dual attention layers, multiple conditioner encoders, autoencoder and noise/timestep scheduler; Turbo distilled few-step | Per-variant checkpoint and licensing, multiple text encoders, MMDiT, VAE, guidance, scheduler, exact noise policy; distilled turbo steps are not transparent fast mode | — |

**Important split:** Qwen-Image, the rectified-flow image model and SD3.5 have distinct source latents/transformer/sampler semantics. The vendor describes the autoregressive image MoE as an AR multimodal MoE, but an upstream issue identifies diffusion-style paths in the published generation code; inspect the exact checkpoint and actual execution graph before claiming a pure autoregressive image decoder. An image TaskDriver must permit the correct source stages instead of forcing all image models into either one global denoising loop or one AR codec loop. Rectified-flow [dev] and [klein] differ by conditioner architecture, checkpoint, release terms and distillation; no imaginary unified tensor parser. Optional hosted frontier image models go through external TaskServiceAdapters with explicit egress, not untrusted model code.

### Speech-to-text, streaming recognition and speech understanding

| Family | Execution semantics | Specialization/quality contracts | Primary source |
|---|---|---|---|
| **OpenAI Whisper**, open source/weights | Log-Mel frontend, audio encoder and autoregressive text decoder with cross-attention; multilingual transcription/translation, timestamp and language tokens; default 30s windows | STFT/mel CPU/GPU DSP, encoder batching, cross-attn state, window/stride/overlap and timestamp alignment, dynamic beam, language task headers | [OpenAI Whisper](https://github.com/openai/whisper), [architecture](https://openai.com/index/whisper/) |
| **NVIDIA Parakeet**, vendor open weights | FastConformer audio encoder with **CTC, RNN-T or TDT** alternative decoding; v2/v3 language variants | cached encoder states, convolution subsampling, blank/predictor/joiner or duration transducer, beam decoding/timestamps, streaming state, no fake decoder LM | vendor speech toolkit, [checkpoint docs](https://docs.nvidia.com/nemo/speech/nightly/asr/asr_checkpoints.html), [Parakeet model](https://build.nvidia.com/nvidia/parakeet-tdt-0_6b-v2/modelcard) |
| **NVIDIA Canary**, vendor open weights | Multitask multilingual ASR/translation with prompt/task slots and encoder-decoder transcription; variants may differ in features | language-conditioned decoder policy, source/target language slots, timestamp/prosody task flags, transcoding | vendor inference guide |
| **VibeVoice-ASR / Streaming / BitNet**, Microsoft published model/repo | Long-form transcription with diarization+timestamp+content; streaming recognition emits chunks; BitNet variant edge quantization | long audio source sample cursor, diarization identity, contextual vocabulary, chunk partial/finality, specialized I8/I2 tensor codec for BitNet | [Microsoft VibeVoice official](https://github.com/microsoft/VibeVoice) |
| **NVIDIA Nemotron speech / streaming voice**, release-dependent | streaming cache-aware FastConformer, deadline-adjusted WER/latency; duplex voice variants couple ASR/TTS | per-stream persistent acoustic cache, variable endpointing/latency modes, 80ms–1s output policies, barge-in/echo/codec clock | vendor speech updates |
| **Qwen3-Omni** audio/video understanding | Audio/video processor, MoE **Thinker** with optional **Talker**; native real-time interactive multimodal input | dual-stage model residency, stable A/V timebases, streaming audio/video input partials, Thinker→Talker handoff, optional text-only Talker eviction | [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) |

Do not collapse all ASR into Whisper or all encoder scoring into next-token generation. In particular CTC, RNN-T and TDT have different input/output/state and blank/duration decoding semantics. Distinguish **partial** ASR hypotheses that might be revised from committed transcript segments; user-visible streaming must tag revision/finality and sample-accurate time spans.

### TTS, codec-token generation, real-time speech and voice cloning

| Family | Execution family | Required model/codec constraints | Primary source |
|---|---|---|---|
| **Kokoro-82M**, open weights, checkpoint terms | style-based **G2P/phonemes → text conditioning/embedding → text prosody/duration/F0 → acoustic generator/ISTFT vocoder**, no token-AR LM required | registered G2P and licenses, phoneme normalization, duration/alignment, prosody voice embedding, FFT/ISTFT overlap-add, exact PCM sample clock | official checkpoint family |
| **Dual-AR speech model**, open code/weights under a source research license | **Dual autoregressive** speech codec: ~21Hz slow temporal primary codebook plus 9 fast residual codebooks; waveform decoder; multi-speaker/prosody | independent AR clocks (time vs RVQ depth), codebook tensor/event shape, codec decoder, voice references, fine-grained control, source sample rate and streaming finality | — |
| **Qwen3-TTS**, open weights under model terms | Dual-track real-time LM; published 12Hz/25Hz speech tokenizers, low-latency multicodebook generation and/or block-wise diffusion reconstruction | bidirectional streaming text/audio, phoneme/voice descriptions, multi-level codec state, packet timebase, separate speech tokenizer and waveform decoder, delay/rollback | [Qwen3-TTS official](https://github.com/QwenLM/Qwen3-TTS), [technical paper](https://arxiv.org/abs/2601.15621) |
| **VibeVoice TTS / Realtime**, Microsoft repo caveat | Continuous acoustic/semantic speech tokens (~7.5Hz) with language context LM and diffusion acoustic head; streaming realtime 0.5B | codebook/continuous representation, diffusion step per acoustic chunk, multi-speaker, long-form ~90min; **original VibeVoice-TTS code was removed from repo in 2025**, inspect variant availability before claiming runnable | [VibeVoice official and availability notes](https://github.com/microsoft/VibeVoice) |
| **A hosted TTS service**, proprietary API | Hosted streaming TTS and separate text-to-dialogue / multi-speaker WebSocket behavior, model dependent | authorized external TaskServiceAdapter, voice ID registry, SSML/turn boundary, stream chunk flush/cancel, credentials and content consent, egress/tracking/cost; **not** a native weight-execution model | Realtime TTS, TTS vs TTD |
| **Qwen3-Omni Thinker/Talker**, open models | Native real-time audio/video understanding and audio output; MoE-based Thinker and multi-codebook Talker | double model stage, optional 10GB-class Talker memory, text+audio synchronized output, dynamic turn and codecs, full duplex/interrupt handling | [Qwen3-Omni official](https://github.com/QwenLM/Qwen3-Omni) |

**Streaming availability ≠ all runtimes implement streaming.** Qwen3-TTS project maintainers explicitly distinguished model architecture support from the early demo package's mainly offline path: [streaming implementation issue](https://github.com/QwenLM/Qwen3-TTS/issues/10). A source model capable of block-wise audio generation does not guarantee Zerogravity can emit early playable audio until the entire upstream codec/decoder path and state checkpoint have been implemented.

### Video and synchronized audiovisual generation

| Model | Source architecture/availability | Native runtime requirements | Primary source |
|---|---|---|---|
| **Wan2.1** 1.3B/14B and I2V, open weights | Spatiotemporal DiT, causal 3D Wan-VAE, text-guided diffusion, offload/sequence parallel examples | temporal latent shape, 3D VAE incremental/tiling, spatial/temporal position axes, scheduler/CFG/seed, source text conditioner | [Wan2.1 official](https://github.com/Wan-Video/Wan2.1) |
| **Wan2.2** T2V/I2V/TI2V, open weights | Diffusion **denoising-phase MoE**: high-noise and low-noise specialized transformer blocks or models selected by timestep; dense TI2V option | source noise/timestep guard for expert switch, separate expert residency and transition events, correct scheduler, 3D VAE, video output timestamp | [Wan2.2 official](https://github.com/Wan-Video/Wan2.2), [model card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |
| **Dual-stream video DiT 1.5**, open code/weights | 8.3B DiT, causal 3D VAE with 16x spatial and 4x temporal compression, Selective/Sliding Tile Attention, super-resolution stage | registered **attention mask semantics**, spatial+temporal tiles, 3D VAE state, conditioned SR, possible FP8 GEMM, overlapping model residency/offload | — |
| **Original dual-stream video DiT**, open weights | Dual-stream/single-stream multimodal DiT, causal 3D VAE, MLLM text encoder | accurate dual→single transition, processor, large conditioner, latent VAE, source model-specific attention | — |
| **Asymmetric-attention video DiT**, Apache 2.0 open weights | 10B Asymmetric Diffusion Transformer (AsymmDiT), 3D video AsymmVAE, unequal text/video hidden size and non-square projections | unequal tensor dimensions and joint attention, 3D positions, causal 3D VAE, context/tensor parallel and tiling | — |
| **Coupled audio/video model**, open model with license caveats; model-specific variants | Coupled **audio+video dual-stream DiT**, bidirectional A/V cross attention, jointly conditioned timestep, media-aware CFG | two clock domains, audio+video latents, synchronized cross-attention/solver, decoder output watermark and coupled resource residency | — |
| **A hosted video service 3.1**, proprietary Google hosted | Text/image-guided audiovisual video generation via asynchronous hosted API; supports sound, reference frames/image | external provider job lifecycle/authorized media upload, quota, billing, result retrieval, optional C2PA; cannot supply native semantic regions without weights | A hosted video service 3.1 |
| **A hosted creative-media image/video service**, proprietary hosted | Hosted creative media API, synchronous/asynchronous image and video job families | external TaskServiceAdapter, OAuth/client credentials, async asset IDs and retries, consent, rights and content credentials | Hosted creative-media API, asynchronous API |
| **A hosted joint audio-video service 1.5 Pro**, proprietary ByteDance hosted | Native joint audio-video two-branch MMDiT-style generator with cross-modal conditioning | remote job task adapter; when a future open comparable model exists, coupled A/V clocks + attention and diffusion model assembly | ByteDance announcement, technical abstract |
| **A hosted video service 3.0 / Omni**, proprietary hosted video-service API | Hosted audio/video generation, storyboarding, image/video references and async jobs | remote typed capability adapter; do not infer internal neural architecture from an API model brand | hosted video service developer API |

**Important accuracy note:** The cited coupled-A/V paper describes synchronized dual streams; its arXiv abstract and HTML differ in the exact substream parameter breakdown in source versions. Do not hardcode those counts as trusted weight geometry; load actual weight model and source manifest. Similarly Wan2.2's timestep MoE is NOT an ordinary per-token MoE route. The asymmetric-attention model's large token space needs context/spatial-temporal tiling and strong memory admission; “GPU fit” is not equivalent to true device throughput.

### Typed decision engines and model selection

| Model | Nature | How it maps to Zerogravity | Primary source |
|---|---|---|---|
| **Proprietary decision service** | Proprietary decision-only service for yes/no/choice/score under an explicit allowed-option schema and calibrated decision semantics | external TaskServiceAdapter; score schema, no generated-text parser, calibration/threshold metadata, independent privacy/policy | [decision study](https://arxiv.org/html/2609.28940v1) |
| **Open multilingual typed-decision model**, open code/weights, checkpoint-dependent license | Non-AR ModernBERT-large 421M / mmBERT-base 322M encoder and task scoring heads; choice/score/no-ul (yes/no) schema, router chooses per request | locally executable TaskDriver ScoreFiniteSchema, normal encoder residency and independent head, batched state/question logits | — |
| **Open joint-schema decision model**, open weights | 27B Qwen3.8-27B or 9B Qwen3.5-9B **multimodal Qwen backbone + joint-schema decision transformer head**; one logit per allowed option, multiple questions in one forward pass | reusable Qwen and vision model semantic components plus registered typed question scoring head, calibrated score output; not token-AR JSON | model card and source head (references removed) |

A decision model must carry raw-logit, probability-calibration, option enumeration and schema ownership distinctly. A 27B multimodal backbone does not automatically make its decision head interchangeable with a 421M encoder classifier. The decision router may choose based on observed accuracy, cost and task latency, but *model type* does not grant unsupported calibration claims.

### Ternary-compressed and source-native hybrid text models

**A ternary-compressed 27B hybrid model** (source family Qwen3.8 27B) is a direct test of the separation between **model topology** and **tensor representation**. It has about 75% linear attention and 25% full attention, with vision as an optional component, like the underlying hybrid model—but the weights are packed in native ternary PTQ1_0 or PQ2_0 encodings with F16 group scales, and the runtime applies a matching blockwise Hadamard transform to activations.

The source model card explicitly specifies:

- PTQ1_0 dense trit packing versus PQ2_0 2-bit trit slots; group size 128 and original source scales.
- 1024-value block Hadamard rotation and fixed sign scheme; matching runtime transformation is **required**.
- Custom ternary kernels for GPU and CPU rather than decompressing the entire model to high-precision tensors.
- Optional separately resident Q8_0 vision mmproj, and model/version compatibility checks.

Primary sources: model card and release notes (references removed).

**Design consequence:** a new quantization codec/activation-transform provider can support the ternary model using the same Qwen3.8 SemanticProgram. Model-dispatch code that hardcodes GGUF Q4_0/Q6_K into the model topology is an obstacle. Distinguish a source-compatible compressed artifact from an ordinary low-bit quantized derivative that silently changes quality; checkpoint identity and activation-basis metadata are mandatory.

Other hybrid/recurrent families (Gated DeltaNet, Mamba, SSM, partial full attention, grouped MoE) are operators/state domains, not new independent inference servers. Every new weight codec and attention variant is separately source-qualified; no generic “ternary fallback to llama.cpp” route.

## Cross-model composition examples

### Live voice assistant

Authenticated live PCM ingress (48kHz) → VAD/resampler and acoustic windows → streaming ASR encoder/cache (Parakeet/Canary/Whisper) or native Thinker → optional typed decision router (proprietary or open decision models) → text/richer reasoning model → streaming TTS (Kokoro/dual-AR/Qwen3-TTS/Talker) → completed codec frames/PCM with backpressure and low-latency playback.

The application-level *orchestration* connecting distinct models is above TaskRequest; each source model has its own exact identity/admission, and its own source math program. If a single **trained compound model** like Qwen3-Omni bundles Thinker/Talker, its internal assembly belongs to the verified ModelDefinition, not the external agent scheduler.

Barge-in must cancel old not-yet-played audio and fence in-flight GPU work without duplicating the committed session state root. Audio clock conversion and streaming revision are first-class, not inferred from text length.

### Image studio

Typed prompt and optional multi-image reference → condition encoders (VLM, CLIP/T5 or Mistral depending on source) → seeded latent/flow source step with registered CFG/noise policy → VAE decode/tiling → image color metadata and requested format. If prompt rewrite uses a separate LLM, it is an explicitly visible optional application composition, not an invisible change of user prompt.

### Video with soundtrack

Wan/dual-stream-video/hosted joint-AV-like source conditioner → temporal latent/DiT schedule (and optionally timestep MoE experts) → causal 3D VAE video frames. With an intrinsically joint A/V model such as the coupled audio/video architecture, audio/video cross-attention and shared timestep outputs are one coupled source assembly, and the StateRoot sync frontier must commit audio samples and video frames consistently. For a staged separate TTS+video workflow, an explicit synchronization adapter and separate quality metrics are needed; do not imply intrinsic lip-sync is guaranteed.

### Decision-heavy application

Typed request with 20 allowed questions → low-latency open decision-model logits → exact calibrated option result. Based on policy, low confidence can trigger a separate Qwen/Gemma reasoning TaskRequest; returned typed decision state and explanation have different source provenance. No arbitrary script/tool execution is embedded in tensor kernels.

## Proposed support tiers, not invented implementation

| Tier | Meaning | Example today |
|---|---|---|
| Recognized | artifact or API name recognized, architecture/weights may be untrusted | catalog research only for all listed media models |
| Semantics registered | exact source operators, codec/time, state domains have reference implementation | Qwen3.5 text only |
| CPU verified | trained checkpoint executes, exact source/mode numeric evidence | Qwen3.5 0.8B/2B only |
| Device region verified | specific GPU kernel on physical device with source-model numeric evidence | historical Q4_0 F32 projection probe, NOT complete GPU model |
| Hybrid execution verified | full model runs with at least one real physical GPU region and CPU fallback | P07-A provider compiles and has CPU host oracle; physical GPU not yet accessible |
| Full native accelerated model | complete model operates on GPU with timing, state and accepted outputs | none yet |
| Stream/duplex qualified | latency, bounded buffers, cancellation, synchronization, quality | none yet |
| External API supported | explicitly authorized, typed remote adapter with credentials/egress and lifecycle | none yet |

Do NOT equate “open-source code,” “open-weight checkpoint,” “commercially unrestricted license” or “hosted API” when choosing runtime support. License and availability can change per model release.

## Implementation roadmap — staged extensions, not a new engine

**P07 remains current:** finish physical CubeCL/CUDA/ROCm/Vulkan source-Q4_0×Q8_0 partial GPU-model test; add Q4_1/Q6_K/F32, persistent weight residency, buffer leases and full-model GPU step planning. The new multimodal design is a *parallel specification effort*, not a reason to reset this implementation.

**MM0 — source semantics & fake tests:** typed value/media clocks, component graph/ports, task frontier coordinates, seed/solver schema, known codec transforms and unsupported-provider mode receipts. Build one fake non-token encoder/decision, one fake denoiser, one streaming audio timeline under the same StateRoot coordinator. Verify 12 acceptance drills in the architecture contract.

**MM1 — practical open audio:** Whisper ASR then small streaming Parakeet/CTC/TDT and/or Kokoro/ISTFT TTS for a non-AR acoustic model. End-to-end waveform + transcript + timestamps against author reference.

**MM2 — image:** a manageable Qwen-Image or rectified-flow-Klein/SD3.5 pipeline using verified conditioner → DiT → scheduler → VAE, actual image output, end-to-end source comparison and peak scratch budgeting. Preserve sampler choices and license.

**MM3 — difficult speech:** dual-AR RVQ, Qwen3-TTS dual-track codec, VibeVoice continuous-token acoustic diffusion, Qwen3-Omni live Thinker/Talker and duplex barge-in. Actual time-to-first-audio, codebook decode and sample clock quality required.

**MM4 — video:** Wan2.1/2.2 (3D VAE, denoising phase MoE), dual-stream video DiT 1.5 (SSTA), asymmetric-attention DiT (AsymmDiT), coupled A/V (joint synchronized). Model-class task drivers remain reusable; introduce only source codecs/operators/conditioners/solver variants.

**MM5 — ternary/decision:** ternary PTQ1_0/PQ2_0 transformed source GEMV, open typed-decision option logits plus multimodal state, transparent small-model routing. Closed-model APIs only through authorized TaskServiceAdapter independent of the CPU/GPU model runtime.

## Architectural acceptance decision

**Keep the Zerogravity v4 execution spine and clarify six contracts; do not rewrite the core.** If the proposed value/clock/codec/assembly/TaskDriver extensions cannot support a new open source model without writing another model server, the extension design failed. The strongest litmus models are:

1. **Dual-AR codec model** — two simultaneous AR axes and codec/waveform staging.
2. **Coupled A/V model** — synchronized dual audio/video streams and source timestep semantics.
3. **Typed decision head** — non-token structured decision head on a multimodal backbone.
4. **Ternary-compressed model** — same Qwen model topology with incompatible packed ternary weight/activation-basis semantics.
5. **Autoregressive image MoE** — advertised autoregressive multimodal MoE with a disputed source-generation path; exact checkpoint/operator tracing is required before concluding AR-only or diffusion-only.
6. **Wan2.2** — source denoising timestep expert switch, not token router.

Passing these architectural mock drills would demonstrate composability across fundamentally different execution types; implementing named weights and physical backends requires separate real validation and licensing.

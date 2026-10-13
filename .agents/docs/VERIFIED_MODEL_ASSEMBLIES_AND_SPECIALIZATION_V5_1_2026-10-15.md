# Zerogravity Verified Model Assemblies, Source Codecs and Legal Specialization — v5.1

**15 October 2026 — FIRST-CLASS TIER-BACKED ASSEMBLY:** Any source-verified model component (MoE expert, text/vision conditioner, VAE, audio codec, lookup row, DFlash auxiliary) can have **different immutable physical backing** under the same ModelAssembly and VerifiedTensorEncoding. [Unified TensorView+materialization contract](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) does not add a separate model architecture or public opaque SSD operator. Iterative video/flow can stage source components at legal solver boundaries; real-time audio can refuse SSD-backed placement if measured first-audible/p95 misses violate admission. Never stream incomplete audio codec state or evict in-flight VAE temporaries as immutable source cache. Distinct expert/row/frozen-session/temporal-media policies share physical IO/resource safety only.


**15 Oct derived codec and hybrid family addendum:** [derived-quant artifact and source quality](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md), [MiMo/DFlash and Qwen/GLM source recipes](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) and [gfx1151 memory/codegen](GFX1151_ROCM_KERNEL_COST_AND_UNIFIED_MEMORY_SAFETY_CONTRACT_2026-10-15.md) are **orthogonal extensions to the same ModelAssembly**, not separate model/media engines. A lossy derived-quant pack is a new derived artifact, a compiled device layout can be lossless, and an auxiliary drafter is a separately versioned component with mutable proposal-only state. No new v5.1 stable ABI/StateRoot writer is introduced; follow derived-quant/HYB/GFX acceptance plans before support claims.


**15 October 2026; architectural design artifact only.** Preserves the v4.3 SemanticProgram→ExecutableBundle model, no new model engine, no dynamic model code execution, no implemented driver claim. Read with the v5.1 universal architecture and real-time serving specifications.

## 1. Central separation: model, task, data encoding, hardware

Four different decisions must remain orthogonal:

1. **Model math and source identity** — source checkpoint/config/processor/tensor roles, validated semantic attention/recurrent/codec/loss/normalization formula and attached component assembly. This defines the computation.
2. **Task algorithm and stop/frontier** — AR token generation, CTC blank collapse, RNN-T predictor, RVQ nested generation, flow solver, denoising, typed choice scoring, multimodal audio/video synchronization. This defines what to do and when output is valid.
3. **Physical encoding and legal numerical specialization** — FP32/BF16, Q4/Q6, NVFP4, ternary PTQ1/PQ2, blockwise transform, source scaling, reduction order. This defines how identical mathematical meaning can be implemented physically.
4. **Device, placement and QoS** — CPU/GPU/NPU/remote, availability, admission, shape band, async events/leases, preferred batch, latency, throughput. This defines where/how fast, never the unapproved model substitution.

**Anti-patterns:** all image models are denoisers; every ASR is Whisper; every TTS is an AR LLM; Qwen-derived = Qwen-supported binary; arbitrary JSON from LLM = calibrated decision head; media transport/codec packets = tensor values; a hosted generation or decision service as local DeviceBackend.

## 2. Artifact and verified assembly manifest (conceptual)

~~~yaml
manifest_version: 1
registered_model: "exact source variant and release, NOT marketing alias"
source:
  artifact_hashes: ["sha256:..."]
  config_hash: "sha256:..."
  tokenizer_or_processor: "sha256:..."
  model_semantics: "registered semantic adapter version"
  license: "exact checkpoint and each processor/weight dependency"
assembly:
  - component_id: "text_conditioner"
    role: "TextEncoder"
    artifact: "sha256:..."
    input_ports: ["text/tokenized"]
    output_ports: ["text_embedding/f32/shape/time-independent"]
    placement: "cpu|gpu|peer candidates selected by engine"
  - component_id: "generator"
    role: "FlowPredictor"
    artifact: "sha256:..."
    input_ports: ["conditioning", "noisy_latent", "timestep"]
    output_ports: ["predicted_velocity"]
  - component_id: "decoder"
    role: "VaeDecoder"
    artifact: "sha256:..."
    input_ports: ["final_latent"]
    output_ports: ["image_pixels/color_spec"]
task:
  driver: "RegisteredFlowSolver"
  numeric: "source exact or explicitly named quality policy"
  exact_schedule: "registered identifier + hash"
  interpolation: "approved enum/source version"
  rng: "specified seed/counter algorithm and fork policy"
ports:
  - ["text_conditioner:embedding", "generator:conditioning"]
  - ["generator:final_latent", "decoder:final_latent"]
capabilities:
  - "text_to_image"
evidence:
  source_semantics: "unverified|verified"
  cpu_reference: "unimplemented|tested"
  gpu_regions: []
  full_model: "unimplemented"
~~~

Do NOT use literal component names in this illustrative YAML as a universal grammar. Real manifests must match exact official source architecture. Assembly keys hash source processor revisions, mathematical schema and legal transforms, not just tensor file byte hashes.

Validation steps:
- enumerate required tensors, shapes, layer topology and internal state, not just file names;
- validate tensor byte size and little/big endian, block scales, codebooks, checkpoints/cross-attention widths;
- verify model-specific prompt templates and media feature extraction;
- verify licensing of tokenizer/G2P/codecs and external prompt upsamplers independently;
- validate required/optional components and output types;
- resolve all conversions through a named, source-verified transform; no silent CLIP-to-Qwen embedding reinterpretation;
- produce capability receipt with supported task, model variant, source identity, numerical policy, hardware proof and legally allowed external egress.

## 3. Operator registry and output shape

Semantic operator categories (not every one needs a public Rust trait):
- Tensor primitives: matmul, norm, activation, gather, quantize, pack, reshape and safe reduction.
- Attention: dense, causal/sliding, cross-modal, sparse spatial-temporal, selective tile, GDN/delta/linear recurrence, joint attention, non-square projections and masked padding.
- Specialists: FFN/experts, tensor-parallel collective, source GDN/SSM update, convolution/causal convolution, 3D RoPE and temporal sampling.
- Audio frontend: resample, STFT/ISTFT, log-mel, overlap-add, periodic convolution, VAD and stream codecs.
- Audio semantic: CTC collapse, RNN-T joiner, TDT duration step, G2P/prosody align, multi-level RVQ, temporal acoustic grouping, waveform decoding.
- Image: patch/latent encoding, MMDiT/DiT attention, CFG guidance pair, rectified flow prediction, Euler/Heun or source-specific solver integration, VAE encode/decode and color management.
- Video: 3D causal VAE, temporal compression, tiled attention/sparse masks, frame reassembly and synchronization.
- Decisions: schema-conditioned logit selection, finite option normalization/calibration with evidence.
- Format/quant: PTQ1_0, PQ2_0, FP4, FP8, Hadamard, residual codebooks, layout transforms, packed expert blocks.

A novel model op requires a new semantic schema and independent reference oracle. Existing closed regions can fuse many primitives or whole device graph if effects and contracts match.

## 4. Component-to-device specialization

At **model load / compile / safe step boundary**:
1. Validate each region's source identity and legal numerical/shape/codec contract.
2. Enumerate source-qualified candidates (portable CPU, native GPU kernel, fused layer, graph capture, AOT/static dataflow).
3. Filter by model/task/precision/shape bounds, currently tested compiler/driver, physical resources and state bridge.
4. Choose measured Pareto candidate for requested workload QoS under privacy/security and model identity constraints.
5. Reserve all required component weights, state, scratch, transfers and bound output buffers under one PhysicalResourceArbiter.
6. Prepare bundle/plan with real command DAG and event/lease dependencies.
7. Execute actual precompiled batch/device-native step island. Runtime may select among precompiled shape bands; no shader compilation or model string matching per audio packet.
8. Await actual completion; TaskCoordinator exclusively prepares and publishes typed StateRoot/valid outputs.
9. Record ModeReceipt and input/output revision without claiming missing device/accuracy proof.

**No forced dequantization to high-precision giant buffers:** source representations can be evaluated directly by specialized kernels (e.g. ternary) if numerical proofs exist. Portable CPU reference may use a slower faithful scalar path, but its mathematically defined result is the guard.

## 5. Source-specific required model assembly examples

### 5.1 Qwen3.5 text + optional vision
- tokenizer and source chat template; Qwen hybrid dense/full attention and GDN recurrent state; source model Q4/Q8 activations and exact reduction contract.
- Optional vision encoder/mmproj with separately declared component artifact and image processor.
- GPU proof of a single Q4 source region does NOT qualify attention/GDN/full-GPU inference; route unsupported fused semantics to source-correct portable implementation or refuse.
- Other Qwen models/versions require checkpoint-specific new adapters where semantics differ.

### 5.2 Ternary 27B model
- Base topology derived from Qwen3.8 27B hybrid attention, but stored ternary weight codecs PTQ1_0/PQ2_0, source FP16 per-group scales, runtime matching block-1024 Hadamard signed basis transformation; optional separate vision mmproj.
- A stock model loader recognizing Qwen topology but lacking PTQ/PQ2/Hadamard must **reject before execution**. Plain low-bit Q2 loading is not safe.
- Specific block/signed scheme and order of scale-transform accumulation are source codec identities, not optional platform tuning.
- Fast direct bitpacked CPU/GPU kernels can coexist with slower checked reference; precise device coverage is per codec/task/shape/input.
- Never equate source storage compression ratio with HBM and throughput. Hardware-specific pack performance must be measured.

### 5.3 Whisper/Parakeet/Canary
- Frontend audio extractor, feature window/cache; Whisper encoder + cross-attending AR text decoder; Parakeet CTC/RNN-T/TDT variant-specific decoder; Canary task/language-conditioned ASR/translation.
- Source audio frames and frontend windows are state domains, NOT generic text token offsets.
- Matmul and attention kernels shared when math matches; each task-specific blank/duration/beam policy owned by a separate driver.
- Diarization/timestamp may be a separate trained component or head, never fabricated from text alone.

### 5.4 Kokoro/Qwen TTS/Moshi
- Kokoro: phoneme frontend, language-dependent G2P, prosody/align/duration and vocoder. No decoder-LM requirement.
- Audio generation models: source codec, slow temporal codebook AR, fast residual depth AR, waveform codec with a legal frame boundary.
- Qwen TTS: checkpoint-specific acoustic tokenizer 12.5/25 Hz and model/codec generation recipe, speaker adaptation.
- Moshi: Mimi streaming neural audio codec plus simultaneous input and output audio lanes, text stream and distinct temporal/codec state. Must not force ASR→LLM→TTS flattening.
- Verify codec sample rates, quantizers, ownership of voice identity/reference audio, source architecture, length and overlap; model output isn't playable until the decoder's exact frame boundary.

### 5.5 Qwen-Image, FLUX.2, SD3.5
- Verified conditioner assembly with text processors, MMDiT or rectified flow block, latent VAE, source guidance and solver.
- FLUX.2 [dev] text conditioner differs from [klein] variant; do not infer from model brand; check actual checkpoint.
- SD3.5 may use CLIP-L, CLIP-G, T5-XXL and a 16-channel latent VAE depending on variant. Correct exact source text conditioning and masks essential.
- Image editing involves source image encoder and reference/conditioning graph, not a generic T2I mode flag.
- Turbo/distilled step counts and quality policies are checkpoint specific; don't hide as generic speed switch.

### 5.6 Autoregressive multimodal image model
- An autoregressive multimodal generation architecture is presented as such; exact generate-image source graph must be inspected at the selected checkpoint revision before declaring mandatory codec/AR or additional solver stages.
- Keep extensibility for codebook image tokens and hybrid AR+latent generation under ModelAssembly and TaskDriver, without preemptively hardcoding one architecture.

### 5.7 Wan, Mochi, joint audio/video models
- Wan 2.2 selects high/low denoise-phase experts based on timestep: **not** token-level FFN MoE. Transition is an authoritative TaskDriver/source iteration decision.
- Autoregressive multimodal video-model sparse/selective sliding tile attention is semantic (mask must not silently change); 3D VAE and optional SR are separate source components.
- Mochi's asymmetric text/video projections and 3D AsymmVAE must retain non-square head geometry and temporal compression, including shape/scratch.
- A joint audio/video model shares cross-attention between genuinely coupled audio and video latents each iteration; independent TTS-over-video is a different model/TaskProgram, not equivalent. Audio and video clocks remain separate until a declared AV group becomes complete.
- Large video tensors require bounded tile/windows, stage-wise offload and per-assembly memory admission, never source-latent unbounded matrices.

### 5.8 Typed finite decisions
- A ModernBERT-based decision encoder typed choice/score/yes-no; a Qwen-style backbone + direct joint-schema score head; proprietary hosted typed decisions.
- Choice option identity, schema revision, raw logits and any calibration id must be captured, not overwritten with unverified "confidence".
- The TaskDriver can score many questions per forward; no unnecessary autoregressive text sampling.
- A proprietary hosted decision service maps to TaskServiceAdapter and cannot claim native local score-head weights.

## 6. Provenance, licensing and external adapter separation

Model code/weights/codecs/processors may have **different** licenses (Apache-2.0/MIT/non-commercial/research-only/restricted) even within one assembly. Record source URL, revision/checksum/license, allowed use, and any required restricted term. Do not infer "GitHub source visible" means commercial redistributable weights. Pin optional encoder or voice-reference assets separately.

Proprietary APIs are opaque and their neural internal architecture cannot be inferred from media endpoint shape. Feature-compatible external adapters are above TaskRequest, with explicit user/admin authorization, credentials, bounded egress and idempotent asynchronous job tracking. Their outputs may be composed with native tasks, but are not substitute native regions.

## 7. Performance and memory hypotheses to falsify

- Optimizing a single GEMM may worsen audio first-audible latency due to longer tail latency or extra codec/transfer operations.
- FP64 reductions restored AMD Q4 source equality in a trained subset but can be costly on consumer NVIDIA; never automatically enable native F32 on AMD without comprehensive exact-source proof.
- Conditioner sharing across image/video tasks can save memory but may worsen PCI transfers or tensor conversion; measure whole-task and residency compatibility.
- High/low denoising expert switching can benefit prefetch if the timestep schedule is known, but scheduling a wrong expert even for one step is a semantic bug.
- Streaming ASR encoder chunk caching reduces compute only if frontend overlap and predictor state remain source-equivalent.
- VAE tiling can alter output boundaries if overlap/padding/crop are not source-matched; validate imagery and pixel-level metrics independently.
- Ternary packed models may improve memory fit yet require costlier runtime Hadamard; benchmark end-to-end and verify exact source.
- Audio/video joint generation may require anti-starvation resident resources even if a text-only GPU scheduler already exists.

Measure cold/warm, end-to-end, input/model/driver/shape/numerical identity, CPU/GPU memory, true media latency and quality. No synthetic throughput goal is a user-approved deadline.

## 8. Required interfaces for agents adding a model

A model implementation proposal must contain:
1. Exact variant and source URLs/hashes/license; list all required components and their independent processors/codecs.
2. Typed input/output ports with clocks/axes and validation bounds.
3. Exact mathematics and novel semantic operator list, with reference examples and numerical policy.
4. TaskDriver stage graph/iteration and legal state commit frontier; unsupported task modes.
5. Tensor encoding and activation transform; special-form codebook or low-bit requirements.
6. All mutable StateDomains and cancel/recompute/fork rules.
7. Candidate region/accelerator kernels and deterministic guard/fallback semantics.
8. Physical memory residency, scratch and transfer budgets for each component.
9. Source-real correctness fixtures and negative tests; device/driver qualified evidence.
10. Serving capability manifest and API transport variants, playback/media requirements if streaming.
11. License/privacy/security and permitted external egress.
12. Measured performance hypotheses with reproducible environment; update acceptance matrix and phase handoff.
13. Change-radius declaration: if implementing this model requires edits to central Router, global arbiter or transport socket parser, explain why a smaller adapter/driver cannot suffice.

## 9. Stop conditions and architectural falsification

- Model-specific code starts a hidden Python/C++ inference server, sets source math by vendor kernel name or mutates TaskCoordinator's root directly → REJECT.
- A new media codec forces token-only StateRoot or all video frames through text token event queues → REJECT.
- New transport protocol forces a different ModelRouter or serializes every PCM packet as JSON through a central engine lock → REJECT.
- A new source tensor packing is silently mapped to Q4/Q2 and outputs are merely visually plausible → REJECT.
- Unknown operation claims CPU reference fallback when no source reference exists → REJECT.
- Cross-modal stages publish independent output before both required components/clock barriers are valid → REJECT.
- Hosted model adapter claims GPU-local CUDA execution or numerical source fidelity without weights → REJECT.
- A revision changes original model sampler/guidance/quant without user-selected quality policy and mode receipt → REJECT.

Only approved source math, typed state effects, truthful device completions and correct task-specific finality may be optimized.

## U-048 — source assemblies and realized heterogeneous numeric provenance

The [heterogeneous pipeline contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) attaches packed codec/overlay/head/draft/
vision/lookup components through this same verified assembly. A supported model
family, component or accelerator is not proof every combination works. Derived
encoding, family mask, paired/tail numeric cuts and scale/epilogue semantics are
recorded in existing numeric/state compatibility and prefix receipts; source
ArtifactKey remains immutable. Late-interaction token embeddings, finite decision
heads and media solvers keep source-specific task/port contracts. No new model
engine, component-owned StateRoot writer or universal arbitrary model execution.

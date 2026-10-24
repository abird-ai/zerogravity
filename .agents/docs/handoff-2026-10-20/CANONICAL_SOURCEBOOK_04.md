# Canonical Sourcebook — volume 4

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/docs/VERIFIED_MODEL_ASSEMBLIES_AND_SPECIALIZATION_V5_1_2026-10-15.md

Original UTF-8 bytes: 19704; SHA256 `01057dffdccd1d9a92bf3c4451cb80784ed7b84b5b90f0c70c9ad0bc754dea1e`.
Resolve embedded relative paths against the original path, not this volume.

````text
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

````

## Source: .agents/docs/VISION_SCOPE_AND_PRINCIPLES.md

Original UTF-8 bytes: 14132; SHA256 `dd87cfe5d9db503c3806494343204d7efbca9fad2b8013ccb92926c18a24447a`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Vision, product scope, constraints and engineering philosophy

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**zerogravity** is a Rust-native inference compiler+runtime. Its promise is not that it can parse any file or always exceed every competing engine, but that for **each supported model** there is a verified correct portable execution path, and that the same model can use progressively more specialized execution on capable hardware without losing maintainability or exposing multiple incompatible runtimes.

## Directly expressed user requirements

- Build **zerogravity**, in Rust, with a comprehensive handoff archive.
- Combine portable model execution, token-budget continuous batching, prefix-aware serving, fused model-region specialization, tiered sparse weights, hybrid state and Rust-owned device kernels into one original system.
- Include everyday modern models: Llama, Gemma 4, Qwen 3.5+, dense/MoE and new families.
- Include unusually large models: Qwen3.8-Flash-Next ~125B language backbone plus auxiliary n-gram/MTP, GLM-5.3-Flash ~321B total, on AMD Strix Halo Ryzen AI Max+395 / Radeon 8060S specifically.
- Strive for best peak performance **on eligible model+hardware**, and general fallback elsewhere.
- Aspire to native high-throughput serving, portable model usability and extreme hardware-specific inference performance, measured without changing model quality, plus future composable AI tasks.
- Formal Semantic IR is the chosen architectural direction; precise IR design and kernel framework remain proposals pending proof.

## Foundational contract

Supported execution means all of the following: checkpoint and tokenizer parsed unambiguously; semantic graph defined; quantization/numerics declared; device backend lowers every required operator; memory available or supported streaming plan; state and sampling semantics preserved across prefill/decode/branch/cancel; observed outputs validated against independent references. **Not** merely model name recognized or a model file downloaded.

For each feature declare stages: `PLANNED`, `EXPERIMENTAL`, `FUNCTIONAL`, `VERIFIED`, `PERFORMANCE_QUALIFIED`. A model can be `VERIFIED` on CPU and experimental on ROCm without contradiction. A specific quant variant can be unsupported while another variant is supported. Modality support is independent of text support.

## Three plan levels

1. Portable/reference execution: stable semantics, robust fallbacks, correct state.
2. Optimized device execution: tuned kernels, batching, graph capture, efficient memory.
3. Extreme specialization: named model-family/hardware tensor shapes, target instructions, sparse/recurrent/MoE fusion; zero forced semantic graph overhead in hot path.

A specialized plan may encompass an entire layer or model region, not just an isolated op. But all inputs, effects and outputs must satisfy the verified region contract. A fast path can be bypassed without restarting the entire process; a state-layout change may require a safe checkpoint/conversion boundary.

## What makes zerogravity qualitatively different (hypotheses, not claims)

- Compile-time mathematical graph and state effects separated from runtime request decisions.
- One resource arbiter coordinating expert cache, prefix state, scratch, n-gram/Engram disk IO and speculation, instead of independent competing allocators.
- Plan candidates optimized for **real concurrency and route overlap**, not only isolated GPU GEMM.
- Full hybrid-state prefix sharing (KV + recurrent and sparse indexer) to reduce repeated agent-context prefill.
- Adaptive speculative policy aware of *verification and memory cost*, not acceptance alone.
- One stable interface for portable, tuned and extreme **Rust-authored** GPU kernels.
- Honest, published negative experiments; automatic selection only after whole-model validation.

## Non-goals

- Arbitrary execution of remote HF model Python code; open format != arbitrary semantics.
- Unlimited context at fixed performance; context/memory geometry matters.
- Magical 321B model in insufficient RAM: SSD tier can help capacity but IO determines usable speed.
- Drop precision silently to reach a throughput marketing metric.
- Force one CUDA/ROCm/Metal implementation for all GPU families.
- Claim speed dominance based on unrelated model/quant numbers.
- Implement a full GPU driver/OS stack in Rust; exact acceptable external vendor runtime interaction is O-003.
- Give client code low-level device pointers or require users to understand expert-cache mechanics.

## User journeys (product specs)

**Local developer:** downloads supported GGUF/safetensors; `zerogravity run model.gguf`; runs text or an explicitly supported multimodal model; sees automatic hardware selection, fit forecast and descriptive unsupported-op failure; no Python inference server dependency. Output streaming and Ctrl-C cancellation preserve integrity.

**Coding agent:** connects via local OpenAI/Responses-compatible endpoint; long system/tool prefixes reused, tool schema/rendered bytes preserved, concurrent sessions isolated, snapshots safely persisted/invalidated and incremental generation begins promptly.

**Strix Halo owner:** model supports `gfx1151`; runtime detects GTT vs physical RAM, computes safe model/context/scratch fit, selects quantized fused GDN/KDA/attention/MoE plan or routes missing experts to IO, displays performance/capacity tradeoffs. Never requires unsafe kernel parameters to function; documented optional tuning is separate.

**GPU server operator:** serves many users, continuous batching with low P95, fairness, quotas, prefix caching and metrics; separate tuning modes for low latency and aggregate throughput; kernel + model + drivers pinned for reproducibility.

**Model contributor:** provides tokenizer/config adapter and semantic ops; CPU reference passes first; portable GPU follows; extreme specialization optional. Does not touch HTTP routing/session lifecycle to add another attention type.

## Product experience requirements

- One coherent executable and Rust SDK (names/flags in other docs are illustrative).
- Clear `inspect` output: model, checkpoint identity, supported modalities, backend chosen, op/region fallback reasons, memory layout/budget, disk IO and speculative mode.
- Appropriate errors: unknown model op, wrong tensor name/shape, unsupported quant, corrupted shard, insufficient memory, unusable GPU, untrusted remote code, incompatible checkpoint, invalid prefix snapshot.
- Deterministic and auditable `--quality`/reference configuration for tests; explicit `--fast` policy cannot silently change model format.
- Cold/warm startup and memory reservation displayed honestly.
- Conservative default API bind localhost and bounded input/remote asset access.

## Quality constraints and optimization order

1. Semantic and state correctness.
2. Quality equivalence relative to chosen checkpoint + quant policy.
3. Safety, memory lifetime, crash handling.
4. Practical compatibility/user clarity.
5. Stable latency and throughput.
6. Peak specialized speed.
7. Experimental complexity only when proven helpful.

Do not invert this hierarchy by prioritizing a kernel headline over exact model state or system stability.

## Integration boundaries

- Frontend owns HTTP/SDK and prompt rendering; it does not own GPU state.
- Model registry owns mathematical definitions/weights and checkpoint provenance.
- Compiler owns execution region legality and plan generation.
- Runtime owns session frontier, scheduling and speculation.
- Resource manager owns buffer leases/IO/prefetch/eviction and physical budgets.
- Device backend owns actual async compute and events.
- Validator owns independent references and qualification evidence.
- No module may mutate another module's invariants through raw pointers without a versioned narrow contract.

## User decisions not inferred

No license, supported OS release priority, first model order, kernel compiler selection, precise numerical tolerance, context/concurrency targets, remote distribution policy, public milestones or deployment dates were selected. They remain explicit in DECISION_REGISTER; phase plan uses recommendations.

## Success definition

Achieve credible model breadth and convenience *plus* narrow performance-class wins where hardware specialization is valuable. Publish full quality/performance matrix including losing cells. The ultimate goal is a design that can improve across models and devices without accumulating incompatible model-specific servers.

## 2026-10-10 integrated v2 addendum — research synthesis and product boundaries

The unified product vision is **a semantic, effect-aware, hardware-adaptive inference compiler + Rust serving engine**, not three independently maintained "personalities". A single model recipe produces reference execution, tuned generic and extreme region variants; all feed the same model/session/worker state contracts. The added async resource-command layer enables optimizing copies, GPU graphs, expert miss I/O, KV sharing and scratch simultaneously without a graph interpreter in token hot path.

"Outclass all projects" is a **measured Pareto ambition**, not an all-model/per-hardware guarantee. Benchmark quality and exact sampling distribution separately from throughput, TTFT/p95, memory/energy, startup and ability to accept model families; publish losing cells and unsupported paths. A correct general fallback is guaranteed for *registered semantic operators* on an actually capable backend with sufficient memory, not arbitrary unknown HF custom code or GGUF metadata.

The prior-art corpus examines budgeted multi-request serving, precise portable formats, extreme quantized kernels, stateful hybrid inference, tiered sparse parameter execution, compiler rewrites and serving-mode-aware autotuning. Its named sources and licenses are preserved in `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` and historical research. The system defines original SIR/RIR/KIR/CIR and typed TaskProgram ownership instead of relying on any named upstream project.

**Product coverage expanded explicitly** to encoder-only embeddings/rerank, LoRA/adapters, grammar-constrained decoding, content-addressed ModelStore and multimodal encoder budgets, in addition to user-required modern text models. Each capability has a distinct acceptance/runner type, not a blanket promise of v1 support. No mandatory dynamic third-party executable plugin ecosystem or deeply duplicated model-specific server.


## 2026-10-12 capability expansion — Gufo NPU and task pipelines

User decision U-016 explicitly adds the **mikealanni Gufo fork's partial same-SoC XDNA2 NPU+iGPU cooperative prefill** (Qwen3.8 Flash-Next) and upstream Gufo ASR/TTS/image/video architecture research to zerogravity's design. This extends the product's hardware adaptability beyond GPU/CPU/dGPU+UMA expert tiers: a device may own **only a subset of a pure operator's output rows** while the main accelerator concurrently performs independent work, with CIR first-use barrier and typed effect/state commit. The model mathematics stays identical; missing NPU gets GPU-only fallback.

New TaskProgram contract covers noncausal encoder, speech/video output and iterative image/diffusion/vocoder stages without twisting them into the LM autoregressive scheduler. All such tasks are planned extensions with independent acceptance, not functional now. **Severe caveat:** source author reports hard GPU+NPU host freezes when fabric clocks change; NPU hardware qualification and explicit opt-in are essential, never root system tuning or silent auto-selection. A Rust-authored XDNA AIE kernel is not yet proven; the supplied fork's Python/C++ kernel cannot be imported as if it satisfied Rust-only implementation. See two new NPU/Gufo docs under this folder.

## v4 refined product scope and minimality principle

**Primary goal:** one extremely capable Rust-owned AI computation execution spine that permits specialization of closed mathematical regions and *trusted task-specific stage control* without duplicating model loaders, allocators, sampler state, server or device backends. Usable inference for standard known models first, then hybrid/sparse/MoE and multiple task modalities (speech, image, video, embeddings), with **future optional** gradient/parameter-update tasks. Do not build a generic workflow/agent permission runtime or a mandatory training/autograd compiler before a real model vertical slice.

**Minimality test:** only SemanticProgram and compiled ExecutableBundle need enduring execution representations; optional RIR/KIR views and CIR events are local compiler/plan work. Global physical budget with device-local fast pools; one session/shard writer publishes prepared state; GPU-native inner steps reduce host sync where exact semantics allow; one library+binary initial packaging. Complexity earns its place only if a correctness proof or measured whole-task performance/operational need requires it.

**Evidence:** matched model/quant/driver/context/task quality, TTFT/P95/energy/physical memory, source-license integrity and cross-device safety; one faster microkernel isn't proof of a better runtime. `ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md` contains exhaustive risks and source research, whereas normative interfaces are original project-neutral contracts.

````

## Source: .agents/docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md

Original UTF-8 bytes: 21399; SHA256 `ab569c147444ab72bdec107f3f433c4c05fabc43522f4832f58d8ed813adfa96`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — One-command CLI, progressive configuration and OpenAI-compatible APIs (v4.3)

**AUTHORITATIVE UX/API DESIGN · 2026-10-12 · NOT IMPLEMENTED.** Accompanies [multi-model routing](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [model/artifact pipeline](PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md), [stateful execution](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [trusted distributed placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md). External CLI and API products are named only in [attributed prior-art research](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md).

**UX philosophy:** *one useful command with almost no decisions*, and *every meaningful decision visible and configurable when needed*. A small terminal tool is the primary experience; device/quant/kernel/model/router/cluster controls appear through inspection and advanced settings. The commands and config examples below are **proposed**, not a finalized public binary/port/flag contract.

## 1. Default one-command user journey

### Interactive local single model

```sh
zerogravity run ./model.gguf
zerogravity run coding
zerogravity run coding --prompt "Summarize this"
zerogravity run coding --context 8192
```

Expected behavior: validate model or local alias → verify exact trained model/quant/tokenizer/source → safe physical resource admission and hardware probe → select best **qualified** numeric/kernel/placement candidate → load/warm model if needed → interactive terminal or one-shot output → stream **committed** output. No config file, mandatory server, GPU architecture, model conversion, external account, cloud endpoint, Python environment or internet access is required for a supported local file.

A local `run` invocation **runs in-process by default**. Optionally reuse an already-running **same-user trusted** service socket when explicitly requested or configured, using versioned process identity and model binding. Do not blindly send prompts to any local HTTP listener or silently start a permanent daemon/systemd/launchd service or install root/firmware settings. If no local model is found, explain available names and offer an explicit import/pull with source/size/license and a consent step; never silently fetch a guessed checkpoint.

If physical memory cannot fit the exact requested model/quant/context, explain the shortfall and available *explicit* alternatives (smaller context, selected alternate quant artifact, CPU streaming or approved peer). Do not quietly downgrade model weights, drop trained experts or route to another model.

### Multi-model router/service

```sh
zerogravity serve
zerogravity serve --model coding
zerogravity models list
zerogravity models ps
zerogravity models show coding
zerogravity models stop coding
zerogravity inspect coding --explain-plan
zerogravity config show --effective --model coding
```

Default: **one engine, one loopback API endpoint, many model aliases** and on-demand authorized local loading, sharing the same ModelCatalog/Router, ResourceArbiter, TaskCoordinator, RegionExecutionProvider and TaskDrivers as standalone CLI. No extra gateway proxy or independent model-specific serving process. `serve --model` can optionally preselect/prewarm an alias while keeping the same router if configured; names/port/flag choices are examples.

### Model import and profile

```sh
zerogravity models import ./model.gguf --name coding
zerogravity models pull <explicit-source-reference>
zerogravity models alias code coding
zerogravity models profile coding --set context=32768
```

Illustrative only. Pull from trusted source on explicit user request/interactive consent, with hash, safe size bounds, license and disk budget. Local import may link or copy a validated immutable blob, according to declared policy. Model profile is a **typed overlay**, not executable foreign code. Alias operations are versioned; old live sessions remain pinned to original model/profile revision.

## 2. Progressive disclosure, not a configuration tax

- **Level 0:** `run model`, `serve`, correct safe auto-plan. No mandatory questions about GPU offload, compiler or KV.
- **Level 1:** `models list`, `models ps`, `inspect` for available/loaded/unloaded, effective context, devices, memory usage and simple timings.
- **Level 2:** `inspect --explain-plan` and `config show --effective` expose actual source/quant/adapter, alias/profile, loaded replicas, backend/shape fallback, cache locality, residency, latency and why a faster/remote variant was refused.
- **Level 3:** user-readable versioned config and named profiles; scoped CLI flags to override allowed nonsecurity preferences.
- **Level 4:** experts can adjust per-model context, physical memory/kv/expert cache budgets, device/phase/quant precision, fused kernels/capture, sampling, draft/LoRA, replicas and SLO policies, routing strategies, remote layer/state/cache placements and tunnel/security/trace settings.

The default terminal output is concise and approachable. Verbose diagnostics are opt-in; hidden model-family-specific flags should not be needed to run an ordinary supported checkpoint.

## 3. Typed configuration, precedence and immutability

Every option carries a **typed scope, validation, effective-source, mutability boundary, security floor and cache invalidation rule**. The chosen on-disk representation can be TOML or another validated versioned format; the example below uses TOML with conventional XDG/OS user config paths.

**Values resolve deterministically**:
1. built-in safe capability defaults and immutable artifact/semantic constraints;
2. qualified observed hardware/shape/precision recommendations, subordinate to mandatory security and numerical floors;
3. administrator service policy (for serving), then allowed user global configuration;
4. source-verified model profile defaults;
5. alias/deployment-specific allowed overrides;
6. CLI command flags;
7. request-scoped allowlisted sampling/task parameters.

**Security and semantic invariants are hard constraints, not low-priority defaults.** A request cannot override data/tenant authorization, configured maximum physical admission, trusted model source/quant/adapter, or device-qualification gate. Model-load/placement overrides require a new warmed deployment or safe drain/StateRoot migration, not per-token source mutation. A request may set temperature/max output/stop only within supported sampler and limits.

Environment variables may be an **explicit documented source**, never mysterious values that override a CLI flag or disable auth. Final precedence, key migration, OS key-file/credential handling and API compatibility must be tested with two different model profiles and aliases.

### Proposed readable configuration

```toml
version = 1

[defaults]
profile = "balanced"
context = "auto"
device = "auto"
routing = "exact"
model_keep_warm = "auto"

[server]
host = "127.0.0.1"
port = 8080   # illustrative port, not a finalized default
remote_access = false

[models.coding]
source = "./models/code-model.gguf"
task = "chat"
context = 32768
keep_warm = "10m"

[models.embedding]
source = "./models/embedding-model.gguf"
task = "embeddings"

[aliases]
code = "coding"
embed = "embedding"

[router]
autoload_local = true
allow_unconfigured_source_download = false
policy = "cache-aware"
on_capacity = "queue-then-reject"

[residency]
physical_headroom = "auto"
max_resident_models = "auto"
idle_unload = "auto"

[network]
enabled = false
```

The *value of this example* is legibility, not binding the Rust implementation to these exact keys. No plaintext secrets should be printed by `config show`. Source model/profile must resolve deterministically, with no circular aliases or invalid family-specific options. An embedding model alias should not claim chat without a qualified chat TaskDriver. `config validate` should work offline without a model download or kernel compilation. `config explain` reports value/source/constraint/mutability/rejected overrides.

**Settings domains**:
- immutable checkpoint/model formula/quant/tokenizer and registered artifact;
- load-time model execution plan, devices/placement/quant packs/graph/replica group;
- residency and memory keepalive/pin/unload/queue/TTL;
- session context/LoRA/adapters, TaskDriver and full StateRoot;
- request sampler/seed/max tokens/response format/tool-output generation;
- router exact alias/resolved replica/cache affinity/explicit alternative model selector;
- network node trust/tunnels/layer and remote state/cache placements;
- API listener/TLS/auth/tenant/allowed model-source and security;
- diagnostics, tracing/quality/reference and source compiler experiments.

A setting change to immutable model identity creates a **new ModelGeneration/alias revision**, never mutates an active checkpoint session invisibly.

## 4. API compatibility matrix — explicitly qualified subset

**Compatibility means passing actual SDK/streaming/schema/error tests for an explicit endpoint/feature set, not claiming every current or future proprietary OpenAI API feature.** The API is a transport adapter to our canonical TaskRequest/TaskResult; it must not own sampling state or device kernels.

| Endpoint | Recommended release scope | Contract |
|---|---|---|
| `GET /v1/models` | Initial | authorized catalog models/aliases and compatible model IDs; native readiness separate |
| `POST /v1/chat/completions` | Initial | messages, model-native roles/rendering, supported sampler, SSE deltas/finish/usage, generated tool-call data |
| `POST /v1/embeddings` | Early when a real encoder qualifies | exact encoder/pooling/dimensions/usage, input batching; no fake decoder embeddings |
| `POST /v1/completions` | Subsequent | raw model-compatible text completion |
| `POST /v1/responses` | Incremental and versioned | stateless input/output first; stateful response_id/store/tool events only after explicit implemented conformance |
| `POST /v1/messages` | Optional Anthropic-compatible adapter after conformance | declared request/events/tool/reasoning/error subset mapped to the same TaskRequest and StateRoot |
| Finite-decision API | Optional source-qualified Decision TaskDriver | model-native label logits, method/calibration/session rules; explicit nonstandard compatibility profile |
| Native `/api/models` / `/api/health` | Privileged local management | alias/profile/load/unload/inspect/residency; never blindly public |
| Speech/image/media task endpoints | Only with qualified TaskDrivers/models | actual codec/formats/stage frontiers rather than placeholder HTTP 200 |
| Reranking/scoring | Native or clearly documented nonstandard extension | model-specific task semantics and score data |

`GET /v1/models` may return authorized registered **on-demand-loadable** aliases even when their physical models are not resident; never expose private filesystem paths, tenant-only model names or internal peer topology. A request's `model` is exact by default; omit model only with an explicitly configured default alias, not surprise model choice.

### Chat Completions

Expected request fields map to `ModelAliasRevision → exact Artifact/Quant/Adapter/ModelDefinition → PromptRenderer → TaskDriver(LanguageGeneration) → prepared StateRoot → committed output stream`. Validate role/content part/tool schema/stop/JSON format and tokenizer rules. For a model without a compatible tool/role template, report `ModelCapabilityUnsupported`; don't flatten different roles into misleading arbitrary strings.

**Streaming:** chat SSE chunks must follow chat-completion `data:` delta/finish conventions for the tested SDK/version, with correct UTF-8 incremental decode, stop across token boundaries, committed output only, usage and finish reason. Speculative drafts never leak into stream. A slow consumer has bounded queue/backpressure; errors after stream begins produce a typed terminal failure and consistent last committed output.

**Tool calls** are *model-generated structured data*. The runtime never executes tools, web search, arbitrary files or user-provided programs merely because it accepts a tool schema. Those are separately permissioned application integrations, not inference.

### Responses API (separate schema and events)

Responses streaming uses its own ordered `response.created`, output-item, text delta and final completion/failure events; it is **not** chat-completion SSE with a renamed URL. A stateless request can be supported first through TaskRequest; stateful `previous_response_id`, server-side store/conversation continuation and resumable output require a real authenticated StateRoot/session/retention contract.

**Privacy-first local default:** no persistent conversation store unless explicitly enabled; a request asking to `store=true` must be honored only when implemented and authorized. The behavior of a cloud provider's current `store` default is **not** automatically inherited. Built-in web/file/computer tools of cloud APIs are not promised through a compatibility endpoint. Return a typed unsupported feature error rather than claim those tools executed.

### Embeddings and other tasks

Embeddings use a qualified `EncodePoolScore` TaskDriver and actual vector outputs/dimensions/normalization. Speech transcription/synthesis and images use their own typed I/O and media codecs; OpenAI-like endpoint labels cannot imply support before real model/operator/reference/codec testing. Model aliases advertise actual task capability and overload/unsupported states without leaking private metadata.

### Authentication, permissions and resource limits

- default listener **loopback only**; any external interface needs explicit authenticated/TLS or approved reverse-proxy policy before start;
- a local SDK requiring a placeholder `api_key` header may be accepted syntactically if local auth is not required, but that string is **not a security token** and must not be logged; never ignore an invalid token on publicly reachable service;
- native model/admin control requires separate admin/local capability, not access granted to every generation client;
- all HTTP endpoints enforce bounded JSON/media/upload/request size, body and tokenizer limits, security-safe CORS, SSRF-resistant media fetch policy;
- remote peer identity and data/weight access are distinct from HTTP Bearer auth; a user API key cannot enroll an untrusted cluster node;
- model alias visibility, license rights, private snapshots and telemetry are tenant scoped.

## 5. Error, partial streaming and compatibility tests

Canonical errors include:
```text
ModelNotFound, ModelNotDownloaded, ModelLoading, ModelUnavailable,
UnsupportedModelOp, UnsupportedTask, UnsupportedApiParameter,
ModelAliasRevisionConflict, IncompatibleAdapter,
OutOfPhysicalBudget, NoQualifiedDevice, UnqualifiedRemotePeer,
ResourceBusy, RequestOverloaded, DeadlineExceeded,
BadRequest, Unauthorized, Forbidden, Cancelled,
StateIncompatible, ModelArtifactCorrupt, InternalFailure
```

HTTP adapter maps to stable tested envelope/status: 400 malformed, 401/403 auth, 404 unknown model, 409 model revision conflict, 422 unsupported capability, 429 quota, 503 queue/device overload, 500 internal; exact details verified against client libraries, not assumed. A loading model may wait within bounded deadline, then return retryable error and local progress, not hang indefinitely. For compatibility, optional harmless metadata can be handled only under an explicit documented policy; **unknown fields affecting sampling, output or security never silently ignored**.

Test against official Python and JS SDKs (pinned tested versions) for `GET /v1/models`, chat nonstream/stream, tool-call generation, embeddings, Responses stateless subset as implemented, precise token usage/output finish and UTF-8/stop/error. Conformance must include: unknown model, unsupported field, alias change during stream, slow consumer, interrupted stream, multi-model concurrent requests, malformed roles, bad JSON/tool, cancelled speculative decode and exact commit/usage. A 200 with a wrong schema is a failure.

## 6. Terminal UX details and advanced controls

- quick startup and concise model load/progress only when useful; interactive prompts support copy/paste/multiline text and Ctrl+C cancellation to last committed root (not killing other daemon sessions);
- stdin/stdout pipelines should work with no ANSI noise when redirected; useful noninteractive `--prompt`/output modes;
- proposed interactive commands `/help`, `/model`, `/status`, `/options`, `/clear` only when they add real clarity. Model switch starts a new explicit source binding/session, not an unsafe reuse of old KV/tokenizer;
- multimodal attachments are explicitly allowed local files; no arbitrary remote URL fetch or implicit prompt content exfiltration;
- concise `inspect` shows loaded state/context/device; optional `--explain-plan` exposes source/quant/adapter/alias, model replicas/physical memory, kernels and fallback reason, remote peer placement and estimated versus measured performance;
- per-model context/sampler defaults and keepalive are user configurable; a model's source fixed quant is not simply a config option to degrade silently;
- no login/account, cloud dependency or persistent daemon required for a local checkpoint; model pull may require explicit network/consent;
- no surprise global system services, root device tweaks, public network exposure or debug logs of prompts/API keys.

## 7. Advanced configuration, without complexity leakage

When explicitly enabled, users can configure model aliases and sources, task type, per-model profiles/LoRA/drafters, sampler/context/grammar, **multiple resident models and replicas**, keepalive/pins, physical RAM/VRAM/cgroup budgets, router queue/fairness and SLO policy, hardware/kernel/quant precision and graph capture, region placement by layer/attention/expert/cache, enrolled remote peers/tunnels/transport, tenant rights, debugging/profiling and exact API compatibility flags.

All effective values report their **source and security guard**. Changes with immutable model/program or StateRoot implications require a new deployment/profile revision or safe compatible committed frontier handoff, not in-place mutation. Source/quant/adapter/position identity is always visible. Some settings are not supported for particular models/hardware; explain with precise `UnsupportedParameter/NoQualifiedDevice` reason.

## 8. Same engine for CLI, server and SDK

`zerogravity run` can instantiate the library in process and issue TaskRequest directly; `zerogravity serve` runs the same ModelCatalog, ModelRouter, ResourceArbiter, TaskCoordinator and DeviceWorkers once and maps HTTP to canonical tasks. A Rust SDK embeds the same types. An optional trusted local socket can reuse a resident service after same-user identity/version verification, but is **not mandatory**.

Private distributed peers run the **same compiled-region worker** with authorized work scopes, not a public `/v1/chat/completions` gateway per accelerator. External cloud model APIs remain separate TaskServiceAdapters with explicit policy, rather than a false claim of native SIR execution.

## 9. Product acceptance checklist

- **One command:** a clean machine with a supported local model can run text without configuration, cloud, daemon or privileged change.
- **Multi-model:** one service lists authorized aliases, routes exact requested source, safely auto-loads available local model(s) when memory permits, shares model weights where compatible, and queues/rejects when full.
- **Profiles:** alias revisions, sampler/context/deployment settings predictable, no switching source in active session, different checkpoint quant remains separate artifact.
- **OpenAI:** passing pinned SDK nonstream/stream chat/models and qualifying embedding/Responses subsets with correct SSE/error/usage and tool-data semantics; not unsupported blanket compatibility.
- **Resource:** multi-model OOM pressure triggers safe unload/queue/refusal, not leaked KV or arbitrary quant.
- **Distributed:** sharded one model, equivalent replicas and many distinct models remain different model/placement constructs, authorized remote privacy/StateRoot and cache stickiness respected.
- **Performance:** standalone/direct `run` avoids per-token router/gateway overhead; model load/cold start/replica routing and C1/C8 P95 measured, not merely advertised.
- **Security:** loopback default, external authentication/admin controls, no model-supplied tools, no user-data URL/remote peer leakage, error paths safe.

**Status:** design targets only; no executable CLI, API endpoint, model router or benchmark exists yet.

## U-048 — task and frontend capabilities

The [task and frontend contract](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) covers qualified task and protocol capabilities:
Messages/Responses adapters, multi-vector embeddings, decision readout, image edits/generation, metrics and optional UI/TUI. Reuse canonical task
requests and one gateway; advertise only implemented/conformance-qualified subsets.
Multi-vector query/document embeddings retain token matrices and source markers;
never return a pooled vector as equivalent. UI/TUI are optional clients and do not
introduce another inference owner. No endpoint implementation is added here.

````

## Source: .agents/docs/handoff-2026-10-17/PAUSED_FILTERS_CONTEXT.md

Original UTF-8 bytes: 22153; SHA256 `8fb40a46d132879ae46ea597a4c74ed9a9e16cfd1f3e680dabde9734918144de`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — additive filters and exclusions cold design

Status: PAUSED at user request; partial unwired SIDE source draft preserved, all Rust checks UNRUN.
Provenance: C1 frozen9path candidate inventory + C2 source-only quant11paths at46cf5a5 overlay; MAIN conditional/candidate gates coordinated separately.
Main plan: [model acquisition](../../plans/model-acquisition/TODO.md).
Owned scope: approved11Rustpaths only in /home/pvl/spaces/abird/src/zerogravity-filters-work (4db5efb +exact C1/C2 overlays) plus this own MAIN run. No MAIN source/Cargo/Git/canonical writes, payload/provider/account/model/SDK/cache/device actions.

## Uncommitted

- [x] Read original byte-component */? matcher, closure and current candidate/quant initial-choice seams without treating drafting code as frozen proof.
- [x] Verify primary Python/HF/vLLM filename matching requirements and choose exact additive full-path UTF8 grammar, brackets/ranges/negation/malformed literals/slash/case.
- [x] Define deterministic nonexponential bounded work/memory/cancel, includeunion->exclude->familyambiguity and required-versus-optional closure exclusion policy.
- [x] Propose additive API/CLI, explicit quant/native composition and meaningful tests, preserving legacy Files grammar and sole provider inventory/budget/publisher.

## Primary requirements and chosen grammar

[Python3.14 fnmatch documentation](https://docs.python.org/3.14/library/fnmatch.html) establishes full-string, case-sensitive fnmatchcase semantics: separators and leadingperiods have no special status; ? consumes one character, * any sequence. [HF download documentation](https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download) supports combined allow/ignore lists. These justify an additive Unicode path matcher, not changing existing component-local byte Files behavior.

Use named fixed grammar `FnmatchCaseV1` on valid UTF8 scalar strings; no locale/case/Unicode normalization. `*` includes slash; consecutive stars collapse with no specialglobstar/recursive-directory operator. Literal slash remains literal, and ? may itself consume slash. Exact repository paths remain validated separately; patterns only choose among owned pinned paths, never authorize URL/path traversal or discovery of a new source. With a subfolder, enforce its literal component-boundary scope first and match the remaining full relative suffix; never interpolate the folder text into pattern syntax (folders can themselves contain brackets). Without it, match full repository-relative paths.

[CPython v3.14.0 matcher source](https://github.com/python/cpython/blob/v3.14.0/Lib/fnmatch.py) and [its fixed tests](https://github.com/python/cpython/blob/v3.14.0/Lib/test/test_fnmatch.py) define bracket edge behavior: [seq], leading! negation, Unicodeordinal ranges; ^ is literal. Initial ] inside a class is literal. Unclosed [ is literal, then remaining pattern parses normally. Hyphens at class ends are literal; descending closed ranges contribute nothing, and negating an empty normalized set matches any scalar. Nested/set-operation punctuation has no POSIX/intersection meaning. Backslash is literal, never a general escape. Port the bounded class normalization into token/range data, not a regex runtime, and validate unusual multiple-hyphen cases against the pinned tests before claiming parity.

Patterns: nonempty <=4096UTF8bytes each; combined include+exclude count<=4096, aggregate pattern bytes plus inventory path bytes<=8MiB. Reject NUL/control inputs before HTTP and do not support bytes/lone-surrogate Python inputs; actual sourcepaths already exclude them. No requirement that every include pattern matches individually: include lists are union, and unmatched excludes are harmless. Empty final selection is an explicit refusal, never select-all or first-file fallback. Excludes-only without an explicit quantbase also refuses; callers can deliberately request --include '*'.

Pinned [vLLM source a8be1500bad93d3c459e435bf71194aa8062e49d](https://github.com/vllm-project/vllm/blob/a8be1500bad93d3c459e435bf71194aa8062e49d/vllm/model_executor/model_loader/weight_utils.py#L399) forwards ignorepatterns and may choose a first matching format. Zerogravity retains explicit selection and refuses ambiguity; it does not inherit that numeric/source fallback.

## Ordered selection and closure

1. Validate bounded request/patterns before provider activity; acquire original operation controls and matcher scratch before allocation. Pin/list once through the same complete private inventory used by C1/C2; validate completeinventory before subfolder projection, preserving accepted collision fix.
2. Match all include patterns as a union over scoped regularpaths, then subtract union of excludes. Determine family ambiguity only AFTER this subtraction; calling current select_paths first would reject before exclusion can intentionally narrow choices. Reuse original GGUF family parser/complete1..N and safetensors/index logic, not new filename heuristics.
3. Apply original model closure with private exclusion policy/reason awareness. Required config/tokenizer, complete split/shard cohort and source index cannot be excluded: fail before any forbidden payload/plan/publication. Excluded optional known sidecars (README/LICENSE/tokenizer_config/etc) are skipped; they cannot be silently re-added by generic closure. If any path is both optional and structurally required, required wins. Include patterns constrain initialchoice; they do not forbid required dependencies. Excludes are the explicit veto.
4. Excluding every file of a whole competingfamily may disambiguate deliberately; excluding only one member of a stillselected splitfamily must fail requiredclosure, not produce a partial model. Excluding an index while its shards remain cannot erase the dependency. Preserve original ambiguousindex/family/regularmode/path checks and explicit external sidecar supplier authority.
5. Final exact per-file bounds/provenance/source identity, rawSHA/EOF/LFS/progress/cancel and sole original publisher remain unchanged. Selectionpolicy is not new content authority or tensor/loader support. No filteredplan with placeholders or postpublication file deletion.

Index policy decision for root before source: safest smallslice forbids reading an explicitly excluded index body. Existing closure discovers relevant indexes by reading adjacent candidates, so an excluded adjacent candidate needed to establish membership must cause typed refusal before GET, even when later parsing might have proved it unrelated. Do not skip it and silently weaken closure. Record this conservative excluded-candidate-index boundary explicitly; broader membership discovery can be separately qualified if useful. No user permission gate or new source authority is implied.

Private code seam: existing require(path) gets Required reason+policy; optional additions use add_if_present_allowed(path). Old Files/quant/component/default paths use AllowAll policy byte-for-byte behavior. HF's original closure and shared GH/ModelScope closure have differences (including legacy .bin roles), so do not unify all implementations under this feature merely to share excludes. Share only the new pure matcher and small exclusion decision; preserve each old owner and provider metadata callbacks.

## Bounded matcher and admission

- Compile one pattern at a time to Literal/Any/Star/Class tokens; no global cache of compiled patterns, no regex crate/runtime and no recursive backtracking. Two DP/NFA state rows scan Unicode scalar filename input with star epsilon transitions. Work polynomial in patternunits*pathscalars, with class-range comparisons charged too; parse/normalize including malformed bracket lookahead must consume work before proceeding.
- Fixed cumulative16Mi state/character/range/compile transitions per complete operation, never reset perpattern/path/include/exclude/closure/refresh. Exhaustion returns typed Limit, not false/no-match/fallback. Cancellation checked before allocations/HTTP and at most4096transitions apart. Collapse repeated stars and bound unmatched-bracket/range normalization work; avoid naive exponential star matching.
- Scratch ceiling512KiB for one <=4096byte token/class representation, two state rows and Unicode path buffer. Precompute exclusion membership once over inventory, then reuse checked lookup during required/optional closure rather than reevaluating patterns each callback. Bounds must account actual retained path keys/collectionnodes; use compact inventory indices or separately charged bounded copies, no uncharged second full inventory.
- Recommended conservative resolve-only extra16MiB control allowance in the SAME existing TransferAdmission/arbiter owner, covering scratch +worstbounded excludedpath keys/nodes. Combine with original operation reservation before new allocations; no new resource domain/allocator/ledger and no silent reduction of existing metadata buffers. Default original80MiB metadata+16MiB remains below ordinary128MiB, but explicit lower caller limits still legitimately refuse. This number must be reconciled with actual chosen representation and tested, not treated as physical all-RAM telemetry. Drop matcher/control state before payload/cache transaction; native256MiB policy unchanged.
- All existing logicalmetadata/body/probe/file/path/manifest bounds remain cumulative across the one pin/tree/closure operation; original HTTP still owns actual retries/deadlines. Never discover_candidates()->resolve(), reset budget inside a selectionpass or materialize a second catalog/cache.

## Additive public and CLI seam

Proposed `acquisition::filters` public private-field type:

```rust
FilterRequest::new(source: CandidateRequest, includes: Vec<String>, excludes: Vec<String>) -> Result<Self>
FilterRequest::for_quant(quant: GgufQuantRequest, excludes: Vec<String>) -> Result<Self>
filters::resolve(transport, &FilterRequest, &PullLimits, &TransferAdmission, &AtomicBool) -> Result<ResolvedPullPlan>
filters::validate_cached_selection(&SnapshotLease, &FilterRequest, &PullLimits, &AtomicBool) -> Result<()>
xet::resolve_filtered(transport, &FilterRequest, &PullLimits, &XetOptions, &TransferAdmission, &AtomicBool) -> Result<ResolvedPullPlan>
```

Names are a proposal until root freezes C2 interfaces. No Selection enum/public PullRequest/PullLimits change. Private initialchoice hook grows from C2 Files/Quant to Filtered(Patterns or Quantbase,excludes), sharing inventory and finish closure; normal old Files and quant route untouched. One new patternengine implementation serves HF/GH/newnative/offline paths.

CLI proposed additions on explicit HF/GH/native Xet file routes: repeated --include GLOB with optional repeated --exclude GLOB; require at least one include. New flags reject any --file/asset/release ambiguity before I/O, so old --file byte*/? grammar never silently changes. Explicit --quant TAG --exclude GLOB is a second base mode: match all requested-tag candidates, subtract excludes BEFORE familychoice, then enforcecompletefamily. Ban --quant+--include in the firstslice rather than invent their intersection/auxiliary semantics. Pure --quant without excludes retains qualified C2 behavior. --exclude alone is invalid. Candidates/release/components/cloud/ModelScope modes do not acquire new semantics implicitly.

Native HF applies filtering INSIDE original Hub/pin/tree metadata pass before discovery; reuse its exact request/body/control budgets, read grants/credential suppression/Xetroot/rawSHA/sourceEOF/ephemeral guard. Must retain >=1announcednative file and current native-index refusal; excluded nativeobjects cannot trigger compatibility fallback. No ordinaryfiltersresolve then second nativealiasresolve. If this thin path cannot be qualified in C3, reject its flags explicitly and keep nativefilter item unfinished.

Offline terminal branch precedes credentials/client/admission/signals, using retained fully verified original source lease/currentlimits/pointer policy and the same bounded matcher/strict closure under honest cachedsubset scope. It neither removes excluded cached data nor publishes a filtered snapshot. Validate whole requested result consistently with C2's final offline contract; if returning a whole SnapshotLease as exact result, reject manifestextras outside finalclosure instead of implying excludedbytes vanished. No remote-global-family completeness/freshness/privateaccess/nativeexecution proof from retainedsubset.

## Meaningful future qualification (all UNRUN)

- [ ] Golden primary fnmatchcases: UTF8scalar? vs original byte?, slashcrossing*/?/classes, dots, caseexactness, composed/decomposedUnicode distinction, collapsed**; classpositive/negation/ranges/nonASCII/leading]/literal^/hyphens/backslash/metacharquoting/unclosedbrackets/descendingranges and setpunctuation. Fixed expected cases from primarytests/independent Python oracle BEFORE Rust implementation, never regenerate expectations to fit new engine.
- [ ] Bounded exhaustive/differential short-string/alphabet tests versus pinned fnmatchcase oracle, with fixed seed/finite cases; no Python runtime dependency or compiler-generated expected values at productruntime. Reject malformedinputpolicy deviations explicitly in docs.
- [ ] Adversarial repeatedstar/no-finalmatch, longmissingbrackets, largeclasses and manyindividuallyfit patterns exhaust one sharedwork cap with typedLimit, bounded measured buffers and cancellation, not exponential hang or false no-match. Admission/explicit tiny-RAM refusal before HTTP/new allocations and default128/native256 envelope positives.
- [ ] HF/GH includeunion thenexclude narrows twoGGUFfamilies correctly, multipleunmatchedinclude patterns tolerated, emptyresult nofallback; old --file still componentlocal bytegrammar exactly. Split partialexclusion/missingmembers/conflictingtotals and sourceindex/config/tokenizer exclusions refuse; optionalREADME/tokenizer_config/LICENSE exclusion stays omitted, never re-added.
- [ ] Adjacentexcludedindex policy refuses before itsGET, no unqualified membershipskip; malformed/ambiguous/incomplete source closures retain oldcriteria. Exactcase/subfolders containingbrackets treated literally; samefolder componentboundary and fullinventory regularancestor collision refusal preserved.
- [ ] Movingbranch/fulltreepagination prove one pin/list and sourcecommit through HEAD/index/body; cumulativelogicalrequests/metadatabytes/header/probes/path/file/control checked once, no reset/extra catalog. Old provenance/digest/source identities equal corresponding explicitfile acquisition.
- [ ] Actual originalowner/native flow tokenisolation/metadataonlydryrun/defaults/cancel/root/fullSHA/EOF/oldref/rollback; cacheddata choice and explicitoffline before env/FIFO/stdin/client/admission/signals, source/currentlimit/pointer/closure/extras negatives, no data deletion/new snapshot.
- [ ] New CLI modes/conflicts/JSONsafe diagnostics/optionalfeatures and old API/exhaustivematch compatibility; strict production/diagnostic, original provider/component/native/legacy regressions, full isolated gate/sourceNix and original model support evidence stay coordinator-owned. No loader/tensor quant support claim.

## Remaining choices before implementation

- [ ] Root freeze final source-qualified C2 initialchoice/API names, excluded-adjacent-index policy and exact matcher control allowance. This is normal design ownership, not a requirement for new user authorization.
- [ ] Source grant, implementation, independent review and actual tests remain pending. C3 is not closed by this proposal or by typed Unsupported boundaries; broader filter compatibility must follow verified implementation.

## Approved C3 source grant

- [x] Root approved exact11path/API outline: artifact acquisition/{filters.rs NEW,quant.rs,selection.rs,huggingface.rs,github.rs,mod.rs,xet/mod.rs}, tests/acquisition_filters.rs NEW; CLI src/{pull.rs,pull_xet.rs}, tests/pull_filters.rs NEW. candidates.rs only if concrete need arises (none planned). API uses private FilterRequest::new(source,Vec includes,Vec excludes), ::for_quant(quant,Vec excludes), source/includes/excludes/quant_request getters; filters resolve/cachedvalidator and native resolve_filtered.
- [x] Root approved excluded-index refusal only where original closure would attempt that index; keep unrelated/GGUF paths unaffected. Fixed matcher bounds and conservative combined extra16MiB SAME admission; offline new mode exact closed snapshot subset and digest-only.
- [ ] Implement matcher and provider/closure/CLI/native reuse, independent fixture coverage, static formatting and independent source review. All Rust compile/tests/lints/gates UNRUN until MAIN grant.
- [ ] Oracle setup initial attempt failed before freezing any output: pinned Python3.14 fnmatch source uses new strict-end regex spelling \z; installed Python3.13 rejects it because initial replacement accidentally looked for two backslashes. Correct only the spelling adaptation to synonymous strict-end \Z, then freeze literals before Rust matcher. This is oracle tooling, not a Rust test failure/RED or fitted expected value.

## Paused interruption handoff — 2026-10-17

- [x] Coordinator relayed explicit user request to wrap/revalidate interruption and pause. Stopped implementation immediately after first new-file scaffold; no additional integration attempted.
- [x] Exact edited Rust path: crates/zerogravity-artifacts/src/acquisition/filters.rs NEW, only in zerogravity-filters-work. Other approved10 Rust paths (quant/selection/HF/GH/mod/Xet +API test and CLI3) remain supplied4db5efb+C1/C2 baseline; candidates.rs not edited. Root is qualifying C2 independently, whose future actual fixes must be synchronized before resuming C3.
- [x] Existing own MAIN run and disposable oracle.py were written before pause; no MAIN writes after pause. This own SIDE TODO copies their useful design/interruption context. No canonical/main source/dependency/lock/Git modifications by C3.
- [x] Frozen independent expected cases BEFORE Rust matcher implementation: CPython v3.14.0 Lib/fnmatch.py read as primary source and executed on installed Python3.13 with ONLY strict-end regex spelling \z -> synonymous \Z.73 literal class/Unicode/slash cases plus585 patterns *57 names =33345 pairs produce literal SHA256 c9f3b0c631f703ae44a4f2dafb5c8c71cb2715bfaa3249751cd299c73a40445f. Initial oracle setup failed on mistaken two-backslash replacement; corrected once before any expected output, then successful oracle command exit0. Expected values were not derived from new Rust or altered to fit it. Oracle is a disposable research generator, not a runtime dependency.
- [x] New file draft contains private fixed-work matcher with Unicode-scalar NFA rows, CPython-style class normalization, no recursive backtracking/regex/cache; FilterRequest constructors/getters, computed Exclusions and public resolve/cachedvalidator SCAFFOLDS. These references intentionally await private integration and make the file NOT COMPILE-READY. It is not wired from acquisition/mod.rs and establishes no filter support.
- [x] Read-only interruption comparison: all8 existing approved integration paths byte-match frozen C2 SIDE after basename repair; both new integration test paths are absent. New filters.rs is the only Rust delta.
- [x] Scoped rustfmt --edition2024 --config skip_children=true filters.rs exited0; own-file Python trailing-whitespace/final-newline check passed. No Cargo/rustc/test/feature checks and no execution of new Rust matcher, oracle assertions or imports. No active command/session/child process is retained.
- [ ] Resume by independently auditing matcher/class normalization and fixed oracle; check compiler/lint only after serialized target grant. Validate actual scratch bound with size_of and class/token worst case; additional adversarial work/cancel/input tests not yet written. Extra16MiB same-admission rationale currently an unverified design comment, not physical qualification.
- [ ] Add missing quant private seams validate_inventory/select_excluding/metadata_admission_extra; InitialChoice::Filtered; default-empty Exclusions behavior. Existing C2 stays unchanged until reconciled qualified baseline is available.
- [ ] Add filtered HF/GH entrypoints inside original one pin/tree owner; required veto and optional skip in BOTH original HF closure and shared selection closure; conservative excluded index refusal only where original closure would attempt reading it. Avoid blanket unrelated/GGUF index refusal. Preserve original Files and pure Quant behavior.
- [ ] Add thin xet::resolve_filtered and combined control admission before HTTP, preserving native >=1announcement/exact source/index boundary/actual budgets and no second runtime/publisher. New filters API currently references nonexistent seams, so do not wire the module prematurely or claim compile success.
- [ ] Implement offline exact final closed subset, source/current limits/roles/pointers/index budgets and digest-only CLI before credentials/client/admission; verify no excluded retained extras are presented as a filtered view. Scaffold uses shared pure closure/read_verified but remains uncompiled/unreviewed.
- [ ] Add CLI Action::Filtered/private FetchRequest extension, includes/excludes vs legacy --file/quant conflicts, literal subfolder suffix matching, same shared native/ordinary fetch defaults and completion. No CLI source edited yet.
- [ ] Author tests/acquisition_filters.rs and CLI tests/pull_filters.rs: include union/exclusion-before-ambiguity, quant complete/partial family, optional omission/required veto/index-before-body, source/budgets/pin equality, offline exact subset/pointer/role/source/limit/secrets, tiny RAM preHTTP/native defaults and unchanged legacy byte grammar. No integration fixture files exist yet.
- [ ] Independent cold review and all compile/tests/lints/full gate/source-integrity/model checks remain UNRUN and coordinator-owned after resume. No C3 support/parity/inference or native execution claim. Preserve SIDE and this unfinished run until recovered; no merge/commit/cleanup requested.

````

## Source: .agents/docs/handoff-2026-10-17/PAUSED_XET_INDEX_CONTEXT.md

Original UTF-8 bytes: 5809; SHA256 `8823513c28666b74e86e503b267f3dd50222d460b7ee772089f9277a486b405c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — native Xet required index reconstruction

Status: paused at coordinator/user request; incomplete source-only draft, all Rust checks UNRUN.
Provenance: f22f759 + exact11frozenC2overlay in disposable zerogravity-xet-index-work, fresh copied mtimes. Root integrates C2/C3 later; preserve quant draft frozen.
Main plan: [model acquisition](../../plans/model-acquisition/TODO.md).
Owned scope: acquisition/huggingface.rs, xet/{mod,budget}.rs, CLI src/{pull,pull_xet}.rs, new artifacts/tests/acquisition_xet_indexes.rs and this run. No MAIN/Cargo/Git/canonical/dependency/provider/model/SDK/device actions.

## Uncommitted

- [x] Read current rules and exact C2/native/HF/source/budget/helper seams. Root approved private defaultfalse XetOptions index policy with additive with_index_reconstruction(self) builder, preserving all resolve signatures.
- [~] One pinned HF callback: exactlength/LFS rawSHA/native complete root/actual EOF before original parser, same closure/tree and original reader/HTTP budget/scratch. No index body cache; reconstruct again during original publication and charge both.
- [ ] Full consumed-counter Result seed across planning/execution, poison/currentlimit refusal and focused private tests.
- [ ] CLI opts native clone into index reconstruction only for apply; dryrun/default keeps no CAS/grants/signed bodies, no added CLI flag or Files/Quant public API multiplication.
- [ ] Authored actual original-HTTP synthetic index+weight publication and scripted limits/root/hash/EOF/cancel/admission/oldref fixtures; all Rust tests UNRUN until exclusive grant.

## Tests and verification

- [ ] Source static comparison/review + scoped formatting/whitespace.
- [ ] New production/diagnostic source/API/CLI fixtures; all UNRUN.
- [ ] Old HF/GH/selection/C1/C2/native/cache/CLI regressions and strict configurations; root target grant required.
- [ ] Full isolated gate/Nix/approved originals after final merge, coordinator owned.

## Boundaries

No compatibility-body fallback or repin/relist; same mathematical engine/source issuer/cache/publisher. Required index is bounded metadata, not new model execution. Default library resolve/dryrun retains noCAS policy; explicit builder authorizes required native index data only. RawSHA independent from native complete root, all counters cumulative, process buffer cap not allRAM. No performance/inference claim.


### Pause handoff — user-requested wrap/interruption boundary

- [x] STOP source drafting after coordinator relayed explicit user pause. All tool commands have completed; no Cargo/rustc/build/test command or owned active child/session was started. No MAIN/Git/index/refs/merge/dependency/network/model action.
- [x] Actual edited production paths ONLY: `crates/zerogravity-artifacts/src/acquisition/huggingface.rs`, `crates/zerogravity-artifacts/src/acquisition/xet/mod.rs`, `crates/zerogravity-artifacts/src/acquisition/xet/budget.rs`. Quant SIDE remains frozen/unmodified; C2 overlay in index SIDE remains context only. CLI pull.rs/pull_xet.rs and acquisition_xet_indexes.rs have NOT been changed/created for this slice.
- [x] Preserve complete prior cold design at sibling `model-acquisition-xet-index-design/TODO.md` in THIS index SIDE. Its old additive resolver recommendation is superseded by the coordinator's builder decision below; useful protocol/accounting/tests remain applicable. Do not depend on retired quant SIDE for context.
- [x] Drafted PRIVATE defaultfalse XetOptions policy with public additive `with_index_reconstruction(self)->Self`; same resolve/resolve_quant signatures. Draft native/HF callback borrows original pinned index context, exact LFS rawSHA/length/validator, reuses discovery/Access/XetReader/Invocation scratch admission, exact Vec plus EOF sentinel/SHA before parser, no index body cache. Existing normal HF index GET remains when no callback/LFS. Indexed spec is retained for final plan without another announcement; later executor reconstructs index again.
- [x] Drafted ALL-counter private Seed with Result snapshot/poison refusal: attempts/wire/metadata/decoded/leaves/refreshes/accepted terms/groups/ranges, validation on restore. Two private budget unit cases authored (carry/currentlimit refusal and poison), both UNCOMPILED/UNRUN.
- [x] Source-only scoped rustfmt --edition2024 --config skip_children=true and corresponding --check on the three edited production files; git diff --check PASS. These establish formatting/syntax parsing only, NOT Rust type/lifetime correctness, protocol acceptance or runtime support.
- [ ] INCOMPLETE: no new acquisition_xet_indexes.rs integration tests yet, no actual original-HTTP synthetic index+weights publication, no counter/admission/root/SHA/EOF/cancel/rollback qualification, no CLI opted-in clone/native-apply wiring/help changes. Default stillfalse, so CLI has no newly enabled feature. Do not merge/advertise this draft.
- [ ] Resume with independent read/type/lifetime/source-preservation review of private callback/reader borrowing and error/cancellation dominance before first compile. In particular check original native announcement/validator behavior and no hidden source-policy change, same metadata body/counter budget and declared default256MiB envelope, all-counter seed poison/current-limit behavior, exact EOF/root before parser and priorref unchanged on any failure. No test/RED/GREEN result is claimed.
- [ ] Root must first reconcile accepted C2/C3 production overlap onto the eventual integration base; this draft starts f22f759+frozenC2, contains NO C3 changes. Preserve C3 selection/callback policy when integrating later. CLI apply opts in via native-options clone ONLY; dryrun/default retains noCAS/no-signed-body, no API function multiplication. No fresh source/Cargo work until coordinator/user resumes.

````

## Source: .agents/docs/handoff-2026-10-17/XET_INDEX_DESIGN.md

Original UTF-8 bytes: 13746; SHA256 `b1e0e5115c089a0363ba5f6edf254e3cb3434496e82e880228910c5cee1138c7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — native Xet required-index reconstruction design

Status: cold design; no source or runtime qualification.
Provenance: accepted native52e42fc; read current MAIN accepted implementation while C1 gates run. This run lives only in zerogravity-quant-work and is separate from frozen C2.
Main plan: [model acquisition](../../plans/model-acquisition/TODO.md).
Owned scope: this TODO only. No source/Cargo/MAIN/Git mutation, provider requests, payload/model/SDK/device/live-cache actions.

## Uncommitted

- [x] Read accepted native README and HF closure, Xet Hub/discovery/Access/Reader/Invocation/Budget and executor index validation.
- [x] Identify exact one-pin index-reader seam and verification/admission/control/cumulative work obligations below.
- [ ] Coordinator chooses dry-run/API policy and authorizes a source slice; no implementation authorized by this design.
- [ ] Implement/qualify only after that grant, reconciling accepted C1/C2 without changing legacy selection or native identities.

## Smallest useful slice

Support an HF LFS-backed selected/adjacent `.safetensors.index.json` required by ORIGINAL closure when its fixed commit URL announces native Xet. Reconstruct that small index through ORIGINAL Access + XetReader + hash/codec/multipart/HTTP owners before SafetensorsIndex parses it. Preserve the same one exact commit, pinned tree and remaining metadata bytes/logical requests. Keep non-LFS ordinary index GET behavior; no repin/relist, compatibility redirect/body fallback, new source issuer, cache/publisher, CAS URL/token authority or chunk runtime. Required model weights still transfer only after final closure and normal cache transaction admission.

### Private provider seam

1. Add an optional PRIVATE admitted index-reader callback at HF's existing exact-size/index GET point, after ensure_length and before original `SafetensorsIndex::from_json`. Supply a borrowed pinned context: original repository/commit/path/URL, exact length, LFS raw SHA and existing validator. Do not fake an HttpResponse/header or infer index status from an arbitrary URL in Hub::get.
2. Ordinary HF/compatibility Files and quant routes keep their original callback/GET path. Native entrypoint supplies the callback; it uses the SAME retained Hub budget and caller TransferAdmission. The callback cannot authorize another source/path or change the original selected/adjacent-index policy. GGUF quant normally needs no index; its shared initial-choice path must remain unchanged.
3. For native LFS index, factor/reuse the EXISTING announcement checker: exact fixed commit URL, header-only302, x-xet-hash parse, all present commit/size/rawSHA bindings checked. Require the pinned LFS rawSHA plus exact length; Git OID is not rawSHA. No follow/compatibility response body. A missing/changed announcement refuses.
4. Keep bounded path→XetSpec for an index already observed so final plan can attach that exact spec without another discovery call. Other final files use original discovery. No index-body cache or new persisted records. Original per-source plan/provenance/path identity stays unchanged.

### Exact bytes before parser

- Refuse before CAS/scratch allocation if index exact length exceeds caller max_file, remaining cumulative HF metadata bytes,8MiB parser bound, or addressable allocation. A selected unknown length must first use the already-approved exact metadata/HEAD policy; no cap/partial JSON mode.
- Lazily acquire ORIGINAL Scratch::reservation from caller's same registered transfer domain BEFORE constructing Scratch/Access/reconstruction vectors. Reuse one scratch sequentially for multiple indexes; retain the outer original resolution control lease. Do not create/register a new arbiter/key or weaken limits.
- Allocate the index Vec with exact checked capacity under the existing resolution control envelope, perform bounded reads/cancellation, incrementally compute actual SHA256. After exactly expected bytes, request the separate EOF sentinel: `take(expected)` alone is WRONG because it can skip XetReader's final full-root/drain/EOF verification. Require no extra logical byte and actual terminal native completion.
- The original XetReader must verify complete ordered file root and exact logical length, including extra edge decoding, signed complete interval set/drain/multipart terminal semantics and replay rules. Compare actual rawSHA with pinned LFS SHA independently. Neither Xet hash nor transport metadata replaces rawSHA.
- Reuse reject_pointer; only then return bytes to original static-error SafetensorsIndex parser and required-shard closure. Never parse tentative prefixes/unverified raw bytes; no shard URL/body before closure. Raw index bytes are metadata/data, not executable readiness or tensor total-size authority.
- Decrement ONE cumulative HF index/body byte budget and logical request budget across all selected/adjacent indexes. Charge decoded index bytes to the native semantic metadata budget (without pretending decompressed bytes are additional physical wire), while actual compressed/index/CAS/headers/retries remain charged by the original native HTTP work owner. Existing observed metadata budget includes headers, so limits may conservatively refuse before HF body-only budget is exhausted.

### Admission and lifetime accounting

Current defaults: outer metadata control envelope80MiB (`8*8MiB+16MiB`), native scratch90.5625MiB (includes40*262144 leaf storage, fixed scratch/control and8*metadata). During index resolution their simultaneous declared peak is170.5625MiB; existing native CLI256MiB can fit,128MiB must deny before signed index body. The bounded index Vec/parser is covered by the original outer metadata envelope; do not add an unaccounted second copy. Custom larger limits must use the actual checked formulas, not this default arithmetic. This scopes process transfer/control buffers, not all host RAM/model memory.

Drop resolution scratch, grants and index bytes after their last parser/closure use and before returning the final plan. The final payload invocation then uses original scratch+cache transaction envelopes (~162.6875MiB defaults). First slice may reconstruct the tiny required index again during publication using existing executor so it is actually included in the immutable snapshot; keep that second transfer bounded/charged and honest rather than inventing a temporary index-body cache. The same pinned rawSHA/spec must verify it again, and original executor reparses staged verified index closure before commit. No cache root/ref mutation during resolution; failures/cancellation leave old refs untouched.

### Mandatory cumulative seed change

CURRENT `budget::Seed` carries ONLY attempts/wire/metadata because accepted resolution has not decoded native bodies. Adding pre-plan index reconstruction without changing that seed would RESET terms/groups/ranges/decoded/leaves/refreshes at Invocation::new, exceeding declared whole-operation caps.

Extend the PRIVATE seed/snapshot to carry ALL consumed counters: attempts,wire,metadata,decoded chunks,accepted logical leaves,refreshes,accepted terms/groups/ranges. Budget::new validates every seeded count against the same options and starts from them. Budget::seed must return Result if reading the accepted Mutex can fail; never silently default/zero on poison. Final plan seed is captured only after all index reconstruction and final discovery. Reader scratch leaves can clear per file, but cumulative Budget leaf/decode counts cannot. Count retried/discarded edge chunks and accepted reconstruction entries on every reissue exactly as now. Every physical HTTP retry/redirect attempt retains original begin_attempt behavior; metadata logical calls are a distinct finite counter, not wire telemetry.

Public plan reuse still begins an independent admitted execution from its resolution seed, matching current semantics; do not claim one global cross-process grant/work counter. No grant/token/signed URL persisted in the plan.

## Root choices before source grant

1. **Recommended preserve existing dry-run promise:** existing `xet::resolve`/CLI dry-run remains Hub metadata+announcement only, no CAS grants/signed bodies. Add a named additive native resolver that explicitly permits required-index metadata reconstruction for APPLY (e.g. `resolve_with_indexes`, private shared implementation). CLI apply chooses it; dry-run that NEEDS native index returns explicit Unsupported before grant and cannot claim complete closure/plan. No new flag or silent dry-run network expansion. Alternatively root can deliberately revise dry-run to permit CAS for index metadata, but current help/README's explicit no-CAS/no-signed-body promise must change and be tested; that is a policy decision, not an implementation shortcut.
2. **Recommended first slice require trusted LFS rawSHA for native indexes.** Already qualified ordinary non-LFS index metadata remains allowed under existing pinned/TLS/FirstSeen rules. Native announced index lacking a pinned rawSHA refuses before CAS; do not promote a newly observed hash to expected authority.
3. **Recommended no pre-plan cache reuse/staging.** Reconstruct required index twice if needed (planning then normal immutable publication), charge both against cumulative limits, document the bounded cost. Avoid introducing a new cache ownership/transaction-before-closure protocol merely to save <=8MiB. Future retained verified index optimization is separate.

## Proposed source/test ownership after approval

Native/HF owner: HF private index callback and native entrypoints/discovery helper/Invocation scratch constructor; xet/budget.rs full seed; same reader/access APIs wherever a narrow helper permits reuse. Shared scratch/codec/hash/multipart stay exact, no duplicate parser or new transport. CLI owner: only apply versus metadata-only resolver choice and precise failure/help wording in existing shared run_fetch, preserving ordinary None and C2 Files|Quant source choice. Tests: new acquisition_xet_indexes.rs plus existing native/ordinary HF/GH/selection/legacy/candidate/quant suites. Root owns docs/gates/dependencies (none expected), integration/Git. Concrete paths/API finalized only after root's dry-run decision.

## Tests and verification — all proposed, UNRUN

- [ ] Golden tiny LFS-index native reconstruction produces verified JSON, original closure requires every same-commit shard+sidecar, native plans retain original per-file/source/provenance. Fixed one alias lookup and one tree listing; adjacent index uses same authority. No weight body before closure; required missing shard refuses without publication.
- [ ] Actual original HTTP owner synthetic Hub→read token→CAS reconstruction→signed ranges for index; Hub token never reaches CAS/ranges, CASgrant never Hub/ranges, signed URL auth suppression/no-follow unchanged; final model source roots/rawSHA valid independently. No live provider/account.
- [ ] Valid JSON with wrong rawSHA or wrong Xet root refuses BEFORE index parser/model body; correct root/SHA but malformed/unsafe/duplicate/overflow index refuses with static error. Truncated/missing signed interval/trailing data/replayed changed chunk/cancellation at exact final length refuse; original root/drain/EOF checks actually reached.
- [ ] Multiple required indexes share total logical request/body/cardinality limits; max_terms/groups/ranges/chunks/refreshes configured to allow each file individually but reject index+later payload cumulatively. Decode/refetch edge work and second planning/publication index copy consume remaining limits. Unit seed snapshot restores every counter and poisoned accepted owner returns error.
- [ ] Full-default128MiB admission fails BEFORE signed index body;256MiB succeeds same fixture and releases all declared memory after drops. Custom metadata/chunk bounds prove arithmetic, exact allocation/no geometric growth and no hidden globalRAM claim.
- [ ] Metadata-only dry-run performs zero CAS grants/signed bodies even when native index is required; apply explicitly reconstructs bounded metadata. Offline requires no credentials/client/provider/admission; existing cached data does not become native execution proof.
- [ ] Grant401/range403 retries charge original total attempts/reissues and preserve reconstruction contract; end-to-end old-ref preservation on each new preplan refusal/cancel. Existing publisher fault matrix remains unchanged; no redundant new generic archival/proof framework.
- [ ] Original HF/GH compatibility index/shard/sidecar, legacy offline, C1 discovery, C2 quant, native non-index and ordinary sidecar resume tests; strict feature/dependency matrices, full isolated gate/Nix, approved originals as root decides. No new inference/throughput claim.

## Current code anchors

- `acquisition/huggingface.rs::resolve_from` exact index length→metadata_checked→verify_metadata_digest→SafetensorsIndex parser, under one existing pinned tree/metadata_left.
- `xet/mod.rs::resolve`, `Invocation::new/reader` and `PlanState::seed`: existing original Hub/discovery and reader/admission owner.
- `xet/access.rs::Access::{reconstruction,fetch,refresh}`: scoped grants/typed target and cumulative accepted reconstruction owner.
- `xet/reader.rs::{Scratch::reservation,XetReader::finish_source}`: actual allocation envelope, terminal complete root/EOF.
- `xet/budget.rs::{Seed,Budget::new,seed,reconstruction,decode_chunk,file_leaf,refresh}`: seed currently partial, MUST extend before decoding during resolve.
- `acquisition/executor.rs::execute_with_progress`: sole transaction/importer and staged index reparse before commit. README explicitly leaves native LFS/index reconstruction open and says dry-run no CAS grants/signed bodies.

````

## Source: .agents/docs/handoff-2026-10-17/paused-filters-oracle.py.txt

Original UTF-8 bytes: 4440; SHA256 `052f5636b3bcf6ae86c35c0984ebf719a004cf98e5d69a6e8bbdd4f9bb6c8c28`.
Resolve embedded relative paths against the original path, not this volume.

````text
"""Disposable primary-source golden generator, never a product/runtime oracle."""
from pathlib import Path
import urllib.request, itertools, hashlib, json
url='https://raw.githubusercontent.com/python/cpython/v3.14.0/Lib/fnmatch.py'
src=urllib.request.urlopen(url,timeout=15).read().decode()
ns={'__name__':'pinned_fnmatch_reference'}
exec(compile(src.replace(chr(92)+'z',chr(92)+'Z'),url,'exec'),ns)
check=ns['fnmatchcase']
assert check('', '') and not check('a\n','a')
cases=[('?', 'é'),('??','é'),('*','a/b'),('a?b','a/b'),('a*b','a/b'),('**','a/b'),('.*','.hidden'),('*','.hidden'),('A*','abc'),('é','é'),('[a-z]','m'),('[a-z]','A'),('[!a-z]','é'),('[!a-z]','a'),('[é-ê]','ê'),('[é-ê]','e'),('[^a]','^'),('[^a]','b'),('[]a]',']'),('[]a]','a'),('[!]]','a'),('[!]]',']'),('[a-]','-'),('[-a]','-'),('[a-]','a'),('[z-a]','a'),('[!z-a]','x'),('[!z-a]','/'),('[a--b]','b'),('[a--b]','a'),('[a--b]','-'),('[a-b-c]','-'),('[a-b-c]','c'),('[a-b-c]','b'),('[a-b-c]','d'),('[a-b-c-d]','d'),('[a-b-c-d]','-'),('[!--]','a'),('[!--]','-'),('[--a]','_'),('[--a]','b'),('[a&&b]','&'),('[a~~b]','~'),('[a||b]','|'),('[[]','['),('[[]a','[a'),('[abc','[abc'),('[!abc','[!abc'),('[]','[]'),('[!]','[!]'),('a\\b','a\\b'),('[\\]','\\'),('[*]','*'),('[?]','?'),('a[[]b','a[b'),('[[]*','[anything'),('a/b','a/b'),('a/*','a/b/c'),('[!a]','/'),('[/]','/'),('[a-b-c-d-e]','e'),('[z-a-c]','-'),('[z-a-c]','c'),('[a-z-a]','-'),('[a-z-a]','a'),('[a--c-e]','b'),('[a--c-e]','e'),('[!a--b]','b'),('[!a--b]','x'),('[---]','-'),('[---]','.'),('[!-]','-'),('[!-]','x')]
patterns=[''.join(t) for n in range(4) for t in itertools.product('ab[]!-*?',repeat=n)]
names=[''.join(t) for n in range(3) for t in itertools.product('ab]-!é/',repeat=n)]
golden=hashlib.sha256(bytes(check(name,pattern) for pattern in patterns for name in names)).hexdigest()
print(dict(source=url, patterns=len(patterns),names=len(names),pairs=len(patterns)*len(names),sha256=golden,fixed_cases=len(cases)))
lines=['//! Additive full-path Unicode FnmatchCaseV1 selection; legacy Files is unchanged.','', '#[cfg(test)]','mod tests {','    use super::*;','    use sha2::{Digest, Sha256};','', '    // Frozen before Rust implementation from CPython v3.14.0 Lib/fnmatch.py.', '    // Oracle Python3.13 only adapts strict-end spelling \\z to synonymous \\Z.', '    #[test]', '    fn pinned_fnmatchcase_class_unicode_and_path_literals() {','        let cases = [']
for pattern,name in cases:
 lines.append(f'            ({json.dumps(pattern,ensure_ascii=False)}, {json.dumps(name,ensure_ascii=False)}, {str(check(name,pattern)).lower()}),')
lines+=['        ];','        let cancel = AtomicBool::new(false);','        let mut work = Work::new(&cancel);','        for (pattern, name, expected) in cases {','            let compiled = Pattern::compile(pattern, &mut work).unwrap();','            assert_eq!(compiled.matches(name, &mut work).unwrap(), expected, "{pattern:?} {name:?}");','        }','    }','    #[test]','    fn bounded_exhaustive_short_patterns_match_frozen_reference_digest() {','        fn words(alphabet: &[char], maximum: usize) -> Vec<String> {','            let mut all = vec![String::new()];','            let mut previous = all.clone();','            for _ in 0..maximum {','                let mut next = Vec::new();','                for prefix in &previous {','                    for ch in alphabet {','                        next.push(format!("{prefix}{ch}"));','                    }','                }','                all.extend(next.iter().cloned());','                previous = next;','            }','            all','        }',"        let patterns = words(&['a','b','[',']','!','-','*','?'], 3);", "        let names = words(&['a','b',']','-','!','é','/'], 2);",'        assert_eq!(patterns.len() * names.len(), 33345);','        let cancel = AtomicBool::new(false);','        let mut work = Work::new(&cancel);','        let mut hash = Sha256::new();','        for pattern in &patterns {','            let compiled = Pattern::compile(pattern, &mut work).unwrap();','            for name in &names {','                hash.update([u8::from(compiled.matches(name, &mut work).unwrap())]);','            }','        }',f'        assert_eq!(format!("{{:x}}", hash.finalize()), "{golden}");','    }','}']
Path('/home/pvl/spaces/abird/src/zerogravity-filters-work/crates/zerogravity-artifacts/src/acquisition/filters.rs').write_text('\n'.join(lines)+'\n')

````

## Source: .agents/docs/handoff-2026-10-17/paused-filters.patch

Original UTF-8 bytes: 21436; SHA256 `097d683b12ee27b7a30d93dd67f06070fab8a3bfa889da9052101d4571dba22b`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-artifacts/src/acquisition/filters.rs b/crates/zerogravity-artifacts/src/acquisition/filters.rs
new file mode 100644
--- /dev/null
+++ b/crates/zerogravity-artifacts/src/acquisition/filters.rs
@@ -0,0 +1,600 @@
+//! Additive full-path Unicode FnmatchCaseV1 selection; legacy Files is unchanged.
+
+use super::candidates::{CandidateRequest, MetadataCalls};
+use super::http::{HttpTransport, check_cancel};
+use super::quant::GgufQuantRequest;
+use super::{ExternalSidecars, Provider, PullLimits, ResolvedPullPlan};
+use crate::cache::{FileRole, Manifest, SnapshotLease, TransferAdmission};
+use crate::{Error, Result};
+use std::collections::{BTreeMap, BTreeSet};
+use std::sync::atomic::AtomicBool;
+
+const MAX_PATTERNS: usize = 4096;
+const MAX_PATTERN_BYTES: usize = 4096;
+const MAX_INPUT_BYTES: usize = 8 * 1024 * 1024;
+const MAX_WORK: usize = 16 * 1024 * 1024;
+pub(super) const CONTROL_BYTES: u64 = 16 * 1024 * 1024;
+
+/// Full-path Unicode-scalar, case-exact FnmatchCaseV1. Slash and initial dots
+/// are ordinary characters; backslash is literal and malformed '[' is literal.
+/// This additive selection never changes the old Files byte/component grammar.
+#[derive(Clone, Debug, Eq, PartialEq)]
+pub struct FilterRequest {
+    source: CandidateRequest,
+    quant: Option<GgufQuantRequest>,
+    includes: Vec<String>,
+    excludes: Vec<String>,
+}
+impl FilterRequest {
+    pub fn new(
+        source: CandidateRequest,
+        includes: Vec<String>,
+        excludes: Vec<String>,
+    ) -> Result<Self> {
+        if includes.is_empty() {
+            return Err(Error::Malformed("filter includes are required".into()));
+        }
+        validate_patterns(&includes, &excludes)?;
+        Ok(Self {
+            source,
+            quant: None,
+            includes,
+            excludes,
+        })
+    }
+    pub fn for_quant(quant: GgufQuantRequest, excludes: Vec<String>) -> Result<Self> {
+        validate_patterns(&[], &excludes)?;
+        Ok(Self {
+            source: quant.source().clone(),
+            quant: Some(quant),
+            includes: Vec::new(),
+            excludes,
+        })
+    }
+    pub fn source(&self) -> &CandidateRequest {
+        &self.source
+    }
+    pub fn includes(&self) -> &[String] {
+        &self.includes
+    }
+    pub fn excludes(&self) -> &[String] {
+        &self.excludes
+    }
+    pub fn quant_request(&self) -> Option<&GgufQuantRequest> {
+        self.quant.as_ref()
+    }
+    fn relative<'a>(&self, path: &'a str) -> Option<&'a str> {
+        match self.source.subfolder() {
+            None => Some(path),
+            Some(folder) => path.strip_prefix(folder)?.strip_prefix('/'),
+        }
+    }
+}
+fn bound() -> Error {
+    Error::Limit("filter work or input bounds exceeded".into())
+}
+fn validate_patterns(includes: &[String], excludes: &[String]) -> Result<usize> {
+    if includes
+        .len()
+        .checked_add(excludes.len())
+        .is_none_or(|n| n > MAX_PATTERNS)
+    {
+        return Err(bound());
+    }
+    let mut bytes = 0usize;
+    for pattern in includes.iter().chain(excludes) {
+        if pattern.is_empty()
+            || pattern.len() > MAX_PATTERN_BYTES
+            || pattern.chars().any(char::is_control)
+        {
+            return Err(Error::Malformed("invalid filter pattern".into()));
+        }
+        bytes = bytes
+            .checked_add(pattern.len())
+            .filter(|n| *n <= MAX_INPUT_BYTES)
+            .ok_or_else(bound)?;
+    }
+    Ok(bytes)
+}
+struct Work<'a> {
+    left: usize,
+    cancel: &'a AtomicBool,
+}
+impl<'a> Work<'a> {
+    fn new(cancel: &'a AtomicBool) -> Self {
+        Self {
+            left: MAX_WORK,
+            cancel,
+        }
+    }
+    fn charge(&mut self, count: usize) -> Result<()> {
+        let before = self.left;
+        self.left = self.left.checked_sub(count).ok_or_else(bound)?;
+        if count >= 4096 || before / 4096 != self.left / 4096 || before == MAX_WORK {
+            check_cancel(self.cancel)?;
+        }
+        Ok(())
+    }
+}
+#[derive(Debug)]
+enum Token {
+    Literal(char),
+    Any,
+    Star,
+    Class {
+        negative: bool,
+        ranges: Vec<(char, char)>,
+    },
+}
+struct Pattern {
+    tokens: Vec<Token>,
+}
+impl Pattern {
+    fn compile(pattern: &str, work: &mut Work<'_>) -> Result<Self> {
+        if pattern.len() > MAX_PATTERN_BYTES {
+            return Err(bound());
+        }
+        work.charge(pattern.len())?;
+        let chars: Vec<_> = pattern.chars().collect();
+        let mut tokens = Vec::with_capacity(chars.len());
+        let mut i = 0;
+        while i < chars.len() {
+            work.charge(1)?;
+            let c = chars[i];
+            i += 1;
+            match c {
+                '*' => {
+                    if !matches!(tokens.last(), Some(Token::Star)) {
+                        tokens.push(Token::Star);
+                    }
+                }
+                '?' => tokens.push(Token::Any),
+                '[' => {
+                    let mut end = i;
+                    if chars.get(end) == Some(&'!') {
+                        end += 1;
+                    }
+                    if chars.get(end) == Some(&']') {
+                        end += 1;
+                    }
+                    while end < chars.len() && chars[end] != ']' {
+                        work.charge(1)?;
+                        end += 1;
+                    }
+                    if end == chars.len() {
+                        tokens.push(Token::Literal('['));
+                    } else {
+                        tokens.push(Self::class(&chars[i..end], work)?);
+                        i = end + 1;
+                    }
+                }
+                _ => tokens.push(Token::Literal(c)),
+            }
+        }
+        Ok(Self { tokens })
+    }
+    // Same range normalization as CPython v3.14.0, represented as scalar
+    // intervals instead of regex. Literal hyphens are distinct from connectors.
+    fn class(chars: &[char], work: &mut Work<'_>) -> Result<Token> {
+        let mut chunks = Vec::<Vec<char>>::new();
+        let mut start = 0usize;
+        let mut search = if chars.first() == Some(&'!') { 2 } else { 1 };
+        while search < chars.len() {
+            work.charge(1)?;
+            if chars[search] == '-' {
+                work.charge(search - start)?;
+                chunks.push(chars[start..search].to_vec());
+                start = search + 1;
+                search += 3;
+            } else {
+                search += 1;
+            }
+        }
+        if start < chars.len() {
+            work.charge(chars.len() - start)?;
+            chunks.push(chars[start..].to_vec());
+        } else if let Some(last) = chunks.last_mut() {
+            last.push('-');
+        }
+        for at in (1..chunks.len()).rev() {
+            work.charge(1)?;
+            if chunks[at - 1].last() > chunks[at].first() {
+                work.charge(chunks[at].len() + chunks.len() - at)?;
+                let tail = chunks.remove(at);
+                chunks[at - 1].pop();
+                chunks[at - 1].extend_from_slice(&tail[1..]);
+            }
+        }
+        let mut units = Vec::<(char, bool)>::new();
+        for (index, chunk) in chunks.iter().enumerate() {
+            work.charge(chunk.len() + 1)?;
+            if index != 0 {
+                units.push(('-', true));
+            }
+            units.extend(chunk.iter().map(|ch| (*ch, false)));
+        }
+        let negative = units.first() == Some(&('!', false));
+        let mut i = usize::from(negative);
+        let mut ranges = Vec::new();
+        while i < units.len() {
+            work.charge(1)?;
+            if i + 2 < units.len() && units[i + 1].1 {
+                ranges.push((units[i].0, units[i + 2].0));
+                i += 3;
+            } else {
+                ranges.push((units[i].0, units[i].0));
+                i += 1;
+            }
+        }
+        Ok(Token::Class { negative, ranges })
+    }
+    fn epsilon(&self, row: &mut [bool], work: &mut Work<'_>) -> Result<()> {
+        for (i, token) in self.tokens.iter().enumerate() {
+            work.charge(1)?;
+            if matches!(token, Token::Star) {
+                row[i + 1] |= row[i];
+            }
+        }
+        Ok(())
+    }
+    fn matches(&self, name: &str, work: &mut Work<'_>) -> Result<bool> {
+        work.charge(self.tokens.len() + 1)?;
+        let mut current = vec![false; self.tokens.len() + 1];
+        let mut next = vec![false; current.len()];
+        current[0] = true;
+        self.epsilon(&mut current, work)?;
+        for ch in name.chars() {
+            next.fill(false);
+            for (i, token) in self.tokens.iter().enumerate() {
+                work.charge(1)?;
+                if !current[i] {
+                    continue;
+                }
+                let yes = match token {
+                    Token::Star => {
+                        next[i] = true;
+                        false
+                    }
+                    Token::Any => true,
+                    Token::Literal(literal) => *literal == ch,
+                    Token::Class { negative, ranges } => {
+                        let mut found = false;
+                        for (low, high) in ranges {
+                            work.charge(1)?;
+                            found |= *low <= ch && ch <= *high;
+                        }
+                        found != *negative
+                    }
+                };
+                next[i + 1] |= yes;
+            }
+            self.epsilon(&mut next, work)?;
+            std::mem::swap(&mut current, &mut next);
+        }
+        Ok(current[self.tokens.len()])
+    }
+}
+/// The only retained matcher result needed by closure. Bounded to4096 keys and
+/// <=8MiB original path bytes. Together with <=512KiB scratch and tree node/key
+/// overhead this fits the conservative extra16MiB admitted control envelope.
+#[derive(Default)]
+pub(super) struct Exclusions {
+    paths: BTreeSet<String>,
+}
+impl Exclusions {
+    pub(super) fn allows(&self, path: &str) -> bool {
+        !self.paths.contains(path)
+    }
+    pub(super) fn required(&self, path: &str) -> Result<()> {
+        if !self.allows(path) {
+            return Err(Error::Unsupported(
+                "filter excludes a required shard, index or sidecar".into(),
+            ));
+        }
+        Ok(())
+    }
+}
+fn union(
+    patterns: &[String],
+    entries: &BTreeMap<String, bool>,
+    request: &FilterRequest,
+    regular_only: bool,
+    work: &mut Work<'_>,
+) -> Result<BTreeSet<String>> {
+    let mut matched = BTreeSet::new();
+    for pattern in patterns {
+        let compiled = Pattern::compile(pattern, work)?;
+        for (path, regular) in entries {
+            work.charge(1)?;
+            if regular_only && !regular || matched.contains(path) {
+                continue;
+            }
+            if let Some(relative) = request.relative(path) {
+                if compiled.matches(relative, work)? {
+                    matched.insert(path.clone());
+                }
+            }
+        }
+    }
+    Ok(matched)
+}
+pub(super) fn select(
+    entries: &BTreeMap<String, bool>,
+    request: &FilterRequest,
+    limits: &PullLimits,
+    cancel: &AtomicBool,
+) -> Result<(BTreeSet<String>, Exclusions)> {
+    super::quant::validate_inventory(entries, limits, cancel)?;
+    let mut bytes = validate_patterns(&request.includes, &request.excludes)?;
+    for path in entries.keys() {
+        check_cancel(cancel)?;
+        bytes = bytes
+            .checked_add(path.len())
+            .filter(|n| *n <= MAX_INPUT_BYTES)
+            .ok_or_else(bound)?;
+    }
+    let mut work = Work::new(cancel);
+    let exclusions = Exclusions {
+        paths: union(&request.excludes, entries, request, false, &mut work)?,
+    };
+    let selected = if let Some(quant) = &request.quant {
+        super::quant::select_excluding(entries, quant, limits, cancel, &exclusions)?
+    } else {
+        let mut selected = union(&request.includes, entries, request, true, &mut work)?;
+        selected.retain(|path| exclusions.allows(path));
+        selected
+    };
+    if selected.is_empty() {
+        return Err(Error::Unsupported(
+            "filter selection is empty; no fallback".into(),
+        ));
+    }
+    check_cancel(cancel)?;
+    Ok((selected, exclusions))
+}
+pub fn resolve(
+    transport: &impl HttpTransport,
+    request: &FilterRequest,
+    limits: &PullLimits,
+    admission: &TransferAdmission,
+    cancel: &AtomicBool,
+) -> Result<ResolvedPullPlan> {
+    check_cancel(cancel)?;
+    let _control = super::quant::metadata_admission_extra(limits, admission, CONTROL_BYTES)?;
+    let transport = MetadataCalls::new(transport, limits.max_pages);
+    match request.source.provider() {
+        Provider::HuggingFace => super::huggingface::resolve_filtered_from(
+            &transport,
+            request,
+            limits,
+            cancel,
+            "https://huggingface.co/",
+        ),
+        Provider::GitHub => super::github::resolve_filtered(&transport, request, limits, cancel),
+    }
+}
+/// Eligibility of exactly the retained closed snapshot, not a filtered view or
+/// proof of remote-global uniqueness, freshness, authorization or native Xet.
+pub fn validate_cached_selection(
+    lease: &SnapshotLease,
+    request: &FilterRequest,
+    limits: &PullLimits,
+    cancel: &AtomicBool,
+) -> Result<()> {
+    limits.validate()?;
+    if limits.unknown_length_cap.is_some() {
+        return Err(Error::Unsupported(
+            "filters require exact source lengths".into(),
+        ));
+    }
+    check_cancel(cancel)?;
+    super::components::validate_cached_file_source(
+        lease,
+        request.source.provider(),
+        request.source.repository(),
+        request.source.revision(),
+    )?;
+    Manifest::new(
+        lease.manifest().source().clone(),
+        lease.manifest().entries().to_vec(),
+        &limits.cache,
+    )?;
+    super::executor::reject_cached_pointers(lease, cancel)?;
+    let entries = lease
+        .manifest()
+        .entries()
+        .iter()
+        .map(|e| (e.path().to_owned(), true))
+        .collect();
+    let (mut selected, exclusions) = select(&entries, request, limits, cancel)?;
+    let mut metadata = limits.max_metadata_bytes;
+    let mut pages = limits.max_pages;
+    super::selection::close_model_selection_filtered(
+        &entries,
+        &mut selected,
+        limits,
+        ExternalSidecars::default(),
+        &exclusions,
+        &mut |path| {
+            check_cancel(cancel)?;
+            pages = pages.checked_sub(1).ok_or_else(bound)?;
+            let entry = lease
+                .manifest()
+                .entries()
+                .iter()
+                .find(|e| e.path() == path)
+                .ok_or_else(|| Error::Malformed("cached filter index absent".into()))?;
+            let size = usize::try_from(entry.bytes())
+                .ok()
+                .filter(|n| *n <= metadata)
+                .ok_or_else(bound)?;
+            metadata -= size;
+            let bytes = lease.read_verified(path, size as u64)?;
+            check_cancel(cancel)?;
+            Ok(bytes)
+        },
+    )?;
+    if selected.len() != lease.manifest().entries().len()
+        || lease
+            .manifest()
+            .entries()
+            .iter()
+            .any(|e| !selected.contains(e.path()))
+    {
+        return Err(Error::Unsupported(
+            "cached snapshot contains files outside the requested filter closure".into(),
+        ));
+    }
+    for entry in lease.manifest().entries() {
+        check_cancel(cancel)?;
+        let expected = if entry.path().ends_with(".gguf") || entry.path().ends_with(".safetensors")
+        {
+            Some(FileRole::Weights)
+        } else if entry.path().ends_with(".safetensors.index.json") {
+            Some(FileRole::Index)
+        } else {
+            match entry.path().rsplit('/').next().unwrap_or(entry.path()) {
+                "config.json" => Some(FileRole::Config),
+                "tokenizer.json" => Some(FileRole::Tokenizer),
+                _ => None,
+            }
+        };
+        if expected.is_some_and(|role| role != entry.role()) {
+            return Err(Error::Malformed(
+                "cached filter required role mismatch".into(),
+            ));
+        }
+    }
+    check_cancel(cancel)
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use sha2::{Digest, Sha256};
+
+    // Frozen before Rust implementation from CPython v3.14.0 Lib/fnmatch.py.
+    // Oracle Python3.13 only adapts strict-end spelling \z to synonymous \Z.
+    #[test]
+    fn pinned_fnmatchcase_class_unicode_and_path_literals() {
+        let cases = [
+            ("?", "é", true),
+            ("??", "é", false),
+            ("*", "a/b", true),
+            ("a?b", "a/b", true),
+            ("a*b", "a/b", true),
+            ("**", "a/b", true),
+            (".*", ".hidden", true),
+            ("*", ".hidden", true),
+            ("A*", "abc", false),
+            ("é", "é", false),
+            ("[a-z]", "m", true),
+            ("[a-z]", "A", false),
+            ("[!a-z]", "é", true),
+            ("[!a-z]", "a", false),
+            ("[é-ê]", "ê", true),
+            ("[é-ê]", "e", false),
+            ("[^a]", "^", true),
+            ("[^a]", "b", false),
+            ("[]a]", "]", true),
+            ("[]a]", "a", true),
+            ("[!]]", "a", true),
+            ("[!]]", "]", false),
+            ("[a-]", "-", true),
+            ("[-a]", "-", true),
+            ("[a-]", "a", true),
+            ("[z-a]", "a", false),
+            ("[!z-a]", "x", true),
+            ("[!z-a]", "/", true),
+            ("[a--b]", "b", true),
+            ("[a--b]", "a", false),
+            ("[a--b]", "-", false),
+            ("[a-b-c]", "-", true),
+            ("[a-b-c]", "c", true),
+            ("[a-b-c]", "b", true),
+            ("[a-b-c]", "d", false),
+            ("[a-b-c-d]", "d", true),
+            ("[a-b-c-d]", "-", true),
+            ("[!--]", "a", true),
+            ("[!--]", "-", false),
+            ("[--a]", "_", true),
+            ("[--a]", "b", false),
+            ("[a&&b]", "&", true),
+            ("[a~~b]", "~", true),
+            ("[a||b]", "|", true),
+            ("[[]", "[", true),
+            ("[[]a", "[a", true),
+            ("[abc", "[abc", true),
+            ("[!abc", "[!abc", true),
+            ("[]", "[]", true),
+            ("[!]", "[!]", true),
+            ("a\\b", "a\\b", true),
+            ("[\\]", "\\", true),
+            ("[*]", "*", true),
+            ("[?]", "?", true),
+            ("a[[]b", "a[b", true),
+            ("[[]*", "[anything", true),
+            ("a/b", "a/b", true),
+            ("a/*", "a/b/c", true),
+            ("[!a]", "/", true),
+            ("[/]", "/", true),
+            ("[a-b-c-d-e]", "e", true),
+            ("[z-a-c]", "-", true),
+            ("[z-a-c]", "c", true),
+            ("[a-z-a]", "-", true),
+            ("[a-z-a]", "a", true),
+            ("[a--c-e]", "b", false),
+            ("[a--c-e]", "e", true),
+            ("[!a--b]", "b", false),
+            ("[!a--b]", "x", true),
+            ("[---]", "-", true),
+            ("[---]", ".", false),
+            ("[!-]", "-", false),
+            ("[!-]", "x", true),
+        ];
+        let cancel = AtomicBool::new(false);
+        let mut work = Work::new(&cancel);
+        for (pattern, name, expected) in cases {
+            let compiled = Pattern::compile(pattern, &mut work).unwrap();
+            assert_eq!(
+                compiled.matches(name, &mut work).unwrap(),
+                expected,
+                "{pattern:?} {name:?}"
+            );
+        }
+    }
+    #[test]
+    fn bounded_exhaustive_short_patterns_match_frozen_reference_digest() {
+        fn words(alphabet: &[char], maximum: usize) -> Vec<String> {
+            let mut all = vec![String::new()];
+            let mut previous = all.clone();
+            for _ in 0..maximum {
+                let mut next = Vec::new();
+                for prefix in &previous {
+                    for ch in alphabet {
+                        next.push(format!("{prefix}{ch}"));
+                    }
+                }
+                all.extend(next.iter().cloned());
+                previous = next;
+            }
+            all
+        }
+        let patterns = words(&['a', 'b', '[', ']', '!', '-', '*', '?'], 3);
+        let names = words(&['a', 'b', ']', '-', '!', 'é', '/'], 2);
+        assert_eq!(patterns.len() * names.len(), 33345);
+        let cancel = AtomicBool::new(false);
+        let mut work = Work::new(&cancel);
+        let mut hash = Sha256::new();
+        for pattern in &patterns {
+            let compiled = Pattern::compile(pattern, &mut work).unwrap();
+            for name in &names {
+                hash.update([u8::from(compiled.matches(name, &mut work).unwrap())]);
+            }
+        }
+        assert_eq!(
+            format!("{:x}", hash.finalize()),
+            "c9f3b0c631f703ae44a4f2dafb5c8c71cb2715bfaa3249751cd299c73a40445f"
+        );
+    }
+}

````

## Source: .agents/docs/handoff-2026-10-17/paused-xet-index.patch

Original UTF-8 bytes: 24553; SHA256 `1db1754e780d788f0636f417a6e1fb57e7d332941cae9ad06c0b7a904338c8c6`.
Resolve embedded relative paths against the original path, not this volume.

````text
diff --git a/crates/zerogravity-artifacts/src/acquisition/huggingface.rs b/crates/zerogravity-artifacts/src/acquisition/huggingface.rs
--- a/crates/zerogravity-artifacts/src/acquisition/huggingface.rs
+++ b/crates/zerogravity-artifacts/src/acquisition/huggingface.rs
@@ -34,6 +34,19 @@
     #[serde(skip)]
     validator: Option<String>,
 }
+
+/// Exact pinned index metadata. Callback receives no authority to select a
+/// different repository, commit or URL; original closure/parser remains owner.
+pub(super) struct PinnedIndex<'a> {
+    pub repository: &'a str,
+    pub commit: &'a str,
+    pub path: &'a str,
+    pub url: &'a str,
+    pub bytes: u64,
+    pub sha256: &'a str,
+    pub validator: Option<&'a str>,
+}
+pub(super) type IndexReader<'a> = dyn for<'b> FnMut(PinnedIndex<'b>) -> Result<Vec<u8>> + 'a;
 
 /// Internal resolver: the public front door holds metadata admission first.
 pub(super) fn resolve(
@@ -129,6 +142,7 @@
         cancel,
         endpoint,
         sidecars,
+        None,
     )
 }
 pub(super) fn resolve_quant_from(
@@ -153,17 +167,20 @@
         cancel,
         endpoint,
         ExternalSidecars::default(),
+        None,
     )
 }
-fn resolve_choice(
+pub(super) fn resolve_choice(
     transport: &impl HttpTransport,
     source: (&str, &str, super::quant::InitialChoice<'_>),
     limits: &PullLimits,
     cancel: &AtomicBool,
     endpoint: &str,
     sidecars: ExternalSidecars,
+    mut index_reader: Option<&mut IndexReader<'_>>,
 ) -> Result<ResolvedPullPlan> {
     let (repository, revision, choice) = source;
+    validate_repo(repository)?;
     let base = Url::parse(endpoint).map_err(|_| Error::Malformed("invalid HF endpoint".into()))?;
     if !base.username().is_empty()
         || base.password().is_some()
@@ -239,13 +256,30 @@
             cancel,
             &pin,
         )?;
-        let (body, _) = metadata_checked(
-            transport,
-            url,
-            &mut metadata_left,
-            cancel,
-            entry.validator.as_deref(),
-        )?;
+        let body = if let (Some(reader), Some(lfs)) = (index_reader.as_mut(), entry.lfs.as_ref()) {
+            let body = reader(PinnedIndex {
+                repository,
+                commit: &pin,
+                path: &path,
+                url: url.as_str(),
+                bytes: size,
+                sha256: &lfs.oid,
+                validator: entry.validator.as_deref(),
+            })?;
+            metadata_left = metadata_left
+                .checked_sub(body.len())
+                .ok_or_else(|| Error::Limit("HF cumulative index metadata bound".into()))?;
+            body
+        } else {
+            metadata_checked(
+                transport,
+                url,
+                &mut metadata_left,
+                cancel,
+                entry.validator.as_deref(),
+            )?
+            .0
+        };
         if body.len() as u64 != size {
             return Err(Error::Malformed("HF index length changed".into()));
         }
diff --git a/crates/zerogravity-artifacts/src/acquisition/xet/mod.rs b/crates/zerogravity-artifacts/src/acquisition/xet/mod.rs
--- a/crates/zerogravity-artifacts/src/acquisition/xet/mod.rs
+++ b/crates/zerogravity-artifacts/src/acquisition/xet/mod.rs
@@ -15,6 +15,9 @@
 use super::{Provider, PullLimits, PullRequest, ResolvedPullPlan};
 use crate::cache::{Manifest, SnapshotLease, TransferAdmission};
 use crate::{Error, Result};
+use sha2::{Digest, Sha256};
+use std::collections::BTreeMap;
+use std::io::Read;
 use std::sync::Arc;
 use std::sync::atomic::{AtomicBool, Ordering};
 use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};
@@ -53,7 +56,9 @@
 }
 
 /// Resolve one explicit native HF request. Metadata and discovery use the same
-/// bounded HTTP owner; this never acquires CAS grants or signed payload bodies.
+/// bounded HTTP owner. Default options never acquire CAS grants or signed
+/// bodies during resolution. Explicit with_index_reconstruction permits only
+/// required bounded index metadata through the same native reader.
 pub fn resolve(
     transport: &impl HttpTransport,
     request: &PullRequest,
@@ -144,22 +149,52 @@
         transport,
         budget: budget.clone(),
     };
-    let mut plan = match choice {
-        NativeChoice::Files(request) => super::huggingface::resolve_from(
-            &hub,
-            request,
-            limits,
+    let (mut plan, mut indexed) = {
+        let source = match choice {
+            NativeChoice::Files(request) => {
+                let super::Selection::Files { paths, subfolder } = &request.selection else {
+                    return Err(Error::Unsupported(
+                        "HF release assets are not supported".into(),
+                    ));
+                };
+                (
+                    &*request.repository,
+                    &*request.revision,
+                    super::quant::InitialChoice::Files {
+                        paths,
+                        subfolder: subfolder.as_deref(),
+                    },
+                )
+            }
+            NativeChoice::Quant(request) => (
+                request.source().repository(),
+                request.source().revision(),
+                super::quant::InitialChoice::Quant(request),
+            ),
+        };
+        let mut indexes = IndexReconstruction {
+            transport,
+            options,
+            budget: budget.clone(),
+            metadata: limits.max_metadata_bytes,
+            admission,
             cancel,
-            &format!("{}/", options.endpoints.hub),
-            super::ExternalSidecars::default(),
-        )?,
-        NativeChoice::Quant(request) => super::huggingface::resolve_quant_from(
-            &hub,
-            request,
-            limits,
-            cancel,
-            &format!("{}/", options.endpoints.hub),
-        )?,
+            invocation: None,
+            specs: BTreeMap::new(),
+        };
+        let plan = {
+            let mut read_index = |index| indexes.read(index);
+            super::huggingface::resolve_choice(
+                &hub,
+                source,
+                limits,
+                cancel,
+                &format!("{}/", options.endpoints.hub),
+                super::ExternalSidecars::default(),
+                Some(&mut read_index),
+            )?
+        };
+        (plan, indexes.specs)
     };
     let mut native = 0usize;
     for file in &mut plan.files {
@@ -173,47 +208,27 @@
                 ));
             }
         }
-        let expected = file
-            .bytes()
-            .ok_or_else(|| Error::Unsupported("native Xet requires exact file size".into()))?;
-        let request = HttpRequest::new(file.url(), limits.max_metadata_bytes as u64);
-        let target = HttpTarget::xet_hub(&request.url, budget.clone(), XetHubMode::Discovery)?;
-        let response = transport.get_target(&request, &target, cancel)?;
-        if response.status != 302 {
-            return Err(Error::Unsupported(
-                "pinned HF file did not announce native Xet".into(),
-            ));
-        }
-        if response
-            .header("x-repo-commit")
-            .is_some_and(|value| value != plan.source.revision())
-            || response
-                .header("x-linked-size")
-                .map(super::http::parse_u64)
-                .transpose()?
-                .is_some_and(|value| value != expected)
-            || response.header("x-linked-etag").is_some_and(|value| {
-                let value = value
-                    .strip_prefix('"')
-                    .and_then(|v| v.strip_suffix('"'))
-                    .unwrap_or(value);
-                file.sha256() != Some(value)
-            })
-        {
-            return Err(Error::State(
-                "native Xet discovery contradicts the pinned source".into(),
-            ));
-        }
-        let file_id = hash::XetHash::from_text(
-            response
-                .header("x-xet-hash")
-                .ok_or_else(|| Error::Malformed("native Xet file identity is absent".into()))?,
-        )?;
-        file.xet = Some(XetSpec {
-            file_id,
-            repository: plan.source.repository().to_string(),
-            commit: plan.source.revision().to_string(),
-        });
+        if let Some(spec) = indexed.remove(file.path()) {
+            file.xet = Some(spec);
+        } else {
+            let expected = file
+                .bytes()
+                .ok_or_else(|| Error::Unsupported("native Xet requires exact file size".into()))?;
+            file.xet = Some(discover(
+                transport,
+                Discovery {
+                    repository: plan.source.repository(),
+                    commit: plan.source.revision(),
+                    url: file.url(),
+                    bytes: expected,
+                    sha256: file.sha256(),
+                    validator: None,
+                },
+                budget.clone(),
+                limits.max_metadata_bytes,
+                cancel,
+            )?);
+        }
         native += 1;
     }
     if native == 0 {
@@ -224,10 +239,167 @@
     let mut result = ResolvedPullPlan::new(plan.source, plan.files, limits)?;
     result.xet = Some(PlanState {
         options: options.clone(),
-        seed: budget.seed(),
+        seed: budget.seed()?,
     });
     check_cancel(cancel)?;
     Ok(result)
+}
+
+struct Discovery<'a> {
+    repository: &'a str,
+    commit: &'a str,
+    url: &'a str,
+    bytes: u64,
+    sha256: Option<&'a str>,
+    validator: Option<&'a str>,
+}
+fn discover(
+    transport: &dyn HttpTransport,
+    contract: Discovery<'_>,
+    budget: Arc<budget::Budget>,
+    metadata: usize,
+    cancel: &AtomicBool,
+) -> Result<XetSpec> {
+    check_cancel(cancel)?;
+    let request = HttpRequest::new(contract.url, metadata as u64);
+    let target = HttpTarget::xet_hub(&request.url, budget, XetHubMode::Discovery)?;
+    let response = transport.get_target(&request, &target, cancel)?;
+    check_cancel(cancel)?;
+    if response.status != 302 {
+        return Err(Error::Unsupported(
+            "pinned HF file did not announce native Xet".into(),
+        ));
+    }
+    super::http::validate_validator(contract.validator, &response)?;
+    if response
+        .header("x-repo-commit")
+        .is_some_and(|value| value != contract.commit)
+        || response
+            .header("x-linked-size")
+            .map(super::http::parse_u64)
+            .transpose()?
+            .is_some_and(|value| value != contract.bytes)
+        || response.header("x-linked-etag").is_some_and(|value| {
+            let value = value
+                .strip_prefix('"')
+                .and_then(|v| v.strip_suffix('"'))
+                .unwrap_or(value);
+            contract.sha256 != Some(value)
+        })
+    {
+        return Err(Error::State(
+            "native Xet discovery contradicts the pinned source".into(),
+        ));
+    }
+    let file_id = hash::XetHash::from_text(
+        response
+            .header("x-xet-hash")
+            .ok_or_else(|| Error::Malformed("native Xet file identity is absent".into()))?,
+    )?;
+    Ok(XetSpec {
+        file_id,
+        repository: contract.repository.to_owned(),
+        commit: contract.commit.to_owned(),
+    })
+}
+/// Only one sequential, already admitted native reader; the bounded Vec is
+/// covered by the outer HF resolution control envelope. No staged body cache.
+struct IndexReconstruction<'a> {
+    transport: &'a dyn HttpTransport,
+    options: &'a XetOptions,
+    budget: Arc<budget::Budget>,
+    metadata: usize,
+    admission: &'a TransferAdmission,
+    cancel: &'a AtomicBool,
+    invocation: Option<Invocation>,
+    specs: BTreeMap<String, XetSpec>,
+}
+impl IndexReconstruction<'_> {
+    fn read(&mut self, index: super::huggingface::PinnedIndex<'_>) -> Result<Vec<u8>> {
+        check_cancel(self.cancel)?;
+        if !self.options.index_reconstruction {
+            return Err(Error::Unsupported("required native Xet index reconstruction needs explicit opt-in; metadata-only resolution never acquires CAS grants".into()));
+        }
+        if index.sha256.len() != 64 || !index.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
+            return Err(Error::Malformed(
+                "native Xet index requires pinned raw SHA256".into(),
+            ));
+        }
+        let length = usize::try_from(index.bytes)
+            .ok()
+            .filter(|n| *n <= self.metadata && *n <= 8 * 1024 * 1024)
+            .ok_or_else(|| Error::Limit("native Xet index metadata length".into()))?;
+        let spec = discover(
+            self.transport,
+            Discovery {
+                repository: index.repository,
+                commit: index.commit,
+                url: index.url,
+                bytes: index.bytes,
+                sha256: Some(index.sha256),
+                validator: index.validator,
+            },
+            self.budget.clone(),
+            self.metadata,
+            self.cancel,
+        )?;
+        // Conservative exact logical-metadata charge before CAS. It is separate
+        // from compressed physical body bytes charged by the original owner.
+        self.budget.charge_metadata(index.bytes)?;
+        if self.invocation.is_none() {
+            self.invocation = Some(Invocation::from_budget(
+                self.options,
+                self.budget.clone(),
+                self.metadata,
+                self.admission,
+                self.cancel,
+            )?);
+        }
+        let mut bytes = Vec::new();
+        bytes
+            .try_reserve_exact(length)
+            .map_err(|_| Error::Limit("native Xet index allocation refused".into()))?;
+        bytes.resize(length, 0);
+        let mut reader = self
+            .invocation
+            .as_mut()
+            .ok_or_else(|| Error::State("Xet index owner absent".into()))?
+            .reader_spec(self.transport, &spec, index.bytes, self.cancel)?;
+        let mut done = 0;
+        let mut digest = Sha256::new();
+        while done < length {
+            check_cancel(self.cancel)?;
+            let end = (done + 8192).min(length);
+            let read = reader
+                .read(&mut bytes[done..end])
+                .map_err(|_| Error::Malformed("native Xet index reconstruction failed".into()))?;
+            check_cancel(self.cancel)?;
+            if read == 0 {
+                return Err(Error::Malformed("native Xet index truncated".into()));
+            }
+            digest.update(&bytes[done..done + read]);
+            done += read;
+        }
+        let mut extra = [0];
+        if reader
+            .read(&mut extra)
+            .map_err(|_| Error::Malformed("native Xet index terminal verification failed".into()))?
+            != 0
+        {
+            return Err(Error::Malformed(
+                "native Xet index exceeded exact length".into(),
+            ));
+        }
+        check_cancel(self.cancel)?;
+        if !format!("{:x}", digest.finalize()).eq_ignore_ascii_case(index.sha256) {
+            return Err(Error::Malformed(
+                "native Xet index raw SHA256 mismatch".into(),
+            ));
+        }
+        super::executor::reject_pointer(&bytes[..bytes.len().min(128)])?;
+        self.specs.insert(index.path.to_owned(), spec);
+        Ok(bytes)
+    }
 }
 
 /// Offline eligibility only, after CacheStore has opened and fully verified
@@ -285,8 +457,28 @@
         let Some(state) = &plan.xet else {
             return Ok(None);
         };
+        let budget = Arc::new(budget::Budget::new(
+            &state.options,
+            plan.limits.max_metadata_bytes,
+            state.seed,
+        )?);
+        Ok(Some(Self::from_budget(
+            &state.options,
+            budget,
+            plan.limits.max_metadata_bytes,
+            admission,
+            cancel,
+        )?))
+    }
+    fn from_budget(
+        options: &XetOptions,
+        budget: Arc<budget::Budget>,
+        metadata: usize,
+        admission: &TransferAdmission,
+        cancel: &AtomicBool,
+    ) -> Result<Self> {
         check_cancel(cancel)?;
-        let bytes = reader::Scratch::reservation(&state.options, plan.limits.max_metadata_bytes)?;
+        let bytes = reader::Scratch::reservation(options, metadata)?;
         let lease = admission.arbiter.admit(
             &admission.resource,
             admission.owner.clone(),
@@ -295,17 +487,13 @@
                 bytes,
             }],
         )?;
-        Ok(Some(Self {
-            budget: Arc::new(budget::Budget::new(
-                &state.options,
-                plan.limits.max_metadata_bytes,
-                state.seed,
-            )?),
-            scratch: reader::Scratch::new(&state.options)?,
-            options: state.options.clone(),
-            metadata_limit: plan.limits.max_metadata_bytes,
+        Ok(Self {
+            budget,
+            scratch: reader::Scratch::new(options)?,
+            options: options.clone(),
+            metadata_limit: metadata,
             _lease: lease,
-        }))
+        })
     }
     pub(super) fn ordinary(&self, file: &super::ResolvedFile) -> Result<super::ResolvedFile> {
         Ok(file.clone().with_request_target(HttpTarget::xet_hub(
@@ -327,6 +515,15 @@
         let bytes = file
             .bytes()
             .ok_or_else(|| Error::Unsupported("native Xet requires exact length".into()))?;
+        self.reader_spec(transport, spec, bytes, cancel)
+    }
+    fn reader_spec<'a>(
+        &'a mut self,
+        transport: &'a dyn HttpTransport,
+        spec: &XetSpec,
+        bytes: u64,
+        cancel: &'a AtomicBool,
+    ) -> Result<reader::XetReader<'a>> {
         let mut access = access::Access::new(
             transport,
             spec,
@@ -364,6 +561,7 @@
     pub max_wire_bytes: u64,
     pub max_refreshes: usize,
     endpoints: Endpoints,
+    index_reconstruction: bool,
 }
 #[derive(Clone, Debug)]
 struct Endpoints {
@@ -381,6 +579,7 @@
             max_requests: 16384,
             max_wire_bytes: 64 * 1024 * 1024 * 1024,
             max_refreshes: 2,
+            index_reconstruction: false,
             endpoints: Endpoints {
                 hub: "https://huggingface.co".into(),
                 cas: "https://cas-server.xethub.hf.co".into(),
@@ -390,6 +589,14 @@
     }
 }
 impl XetOptions {
+    /// Explicitly permit bounded required native index metadata reconstruction
+    /// during resolution. Default remains metadata/discovery-only, with no CAS
+    /// grants or signed bodies. Index and later payload consume ONE work budget.
+    pub fn with_index_reconstruction(mut self) -> Self {
+        self.index_reconstruction = true;
+        self
+    }
+
     pub fn validate(&self) -> Result<()> {
         if self.max_terms == 0
             || self.max_terms > 65536
diff --git a/crates/zerogravity-artifacts/src/acquisition/xet/budget.rs b/crates/zerogravity-artifacts/src/acquisition/xet/budget.rs
--- a/crates/zerogravity-artifacts/src/acquisition/xet/budget.rs
+++ b/crates/zerogravity-artifacts/src/acquisition/xet/budget.rs
@@ -11,6 +11,10 @@
     attempts: u64,
     wire: u64,
     metadata: u64,
+    decoded: u64,
+    leaves: u64,
+    refreshes: u64,
+    accepted: Accepted,
 }
 #[derive(Debug)]
 pub(crate) struct Budget {
@@ -24,7 +28,7 @@
     refreshes: AtomicU64,
     accepted: Mutex<Accepted>,
 }
-#[derive(Debug, Default)]
+#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
 struct Accepted {
     terms: usize,
     groups: usize,
@@ -71,6 +75,12 @@
             || seed.attempts > limits.max_requests as u64
             || seed.wire > limits.max_wire_bytes
             || seed.metadata > metadata_limit as u64
+            || seed.decoded > limits.max_chunks as u64
+            || seed.leaves > limits.max_chunks as u64
+            || seed.refreshes > limits.max_refreshes as u64
+            || seed.accepted.terms > limits.max_terms
+            || seed.accepted.groups > limits.max_fetch_groups
+            || seed.accepted.ranges > limits.max_ranges
         {
             return Err(exhausted());
         }
@@ -80,10 +90,10 @@
             attempts: AtomicU64::new(seed.attempts),
             wire: AtomicU64::new(seed.wire),
             metadata: AtomicU64::new(seed.metadata),
-            decoded: AtomicU64::new(0),
-            leaves: AtomicU64::new(0),
-            refreshes: AtomicU64::new(0),
-            accepted: Mutex::new(Accepted::default()),
+            decoded: AtomicU64::new(seed.decoded),
+            leaves: AtomicU64::new(seed.leaves),
+            refreshes: AtomicU64::new(seed.refreshes),
+            accepted: Mutex::new(seed.accepted),
         })
     }
     pub(crate) fn begin_attempt(&self, cancel: &AtomicBool) -> Result<()> {
@@ -172,15 +182,83 @@
         super::check_cancel(cancel)?;
         checked_increment(&self.refreshes, self.limits.max_refreshes as u64)
     }
-    pub(super) fn seed(&self) -> Seed {
-        Seed {
+    pub(super) fn seed(&self) -> Result<Seed> {
+        self.ensure_live()?;
+        let accepted = *self
+            .accepted
+            .lock()
+            .map_err(|_| Error::State("Xet control counter owner failed".into()))?;
+        Ok(Seed {
             attempts: self.attempts.load(Ordering::Acquire),
             wire: self.wire.load(Ordering::Acquire),
             metadata: self.metadata.load(Ordering::Acquire),
-        }
+            decoded: self.decoded.load(Ordering::Acquire),
+            leaves: self.leaves.load(Ordering::Acquire),
+            refreshes: self.refreshes.load(Ordering::Acquire),
+            accepted,
+        })
     }
     #[cfg(test)]
     pub(crate) fn observed_wire_bytes(&self) -> u64 {
         self.wire.load(Ordering::Acquire)
     }
 }
+
+#[cfg(test)]
+mod seed_tests {
+    use super::*;
+    #[test]
+    fn resolution_seed_preserves_every_consumed_budget_and_refuses_current_limits() {
+        let mut limits = XetOptions::default();
+        limits.max_requests = 1;
+        limits.max_wire_bytes = 3;
+        limits.max_chunks = 1;
+        limits.max_refreshes = 1;
+        limits.max_terms = 1;
+        limits.max_fetch_groups = 1;
+        limits.max_ranges = 1;
+        let flag = AtomicBool::new(false);
+        let original = Budget::new(&limits, 3, Seed::default()).unwrap();
+        original.begin_attempt(&flag).unwrap();
+        original.charge_body(3, true).unwrap();
+        original.decode_chunk(&flag).unwrap();
+        original.file_leaf(&flag).unwrap();
+        original.refresh(&flag).unwrap();
+        original.reconstruction(1, 1, 1, &flag).unwrap();
+        let seed = original.seed().unwrap();
+        let recovered = Budget::new(&limits, 3, seed).unwrap();
+        assert!(recovered.begin_attempt(&flag).is_err());
+        assert!(recovered.decode_chunk(&flag).is_err());
+        assert!(recovered.file_leaf(&flag).is_err());
+        assert!(recovered.refresh(&flag).is_err());
+        assert!(recovered.reconstruction(1, 0, 0, &flag).is_err());
+        assert!(recovered.reconstruction(0, 1, 0, &flag).is_err());
+        assert!(recovered.reconstruction(0, 0, 1, &flag).is_err());
+        assert!(recovered.charge_body(1, false).is_err());
+        assert!(Budget::new(&limits, 2, seed).is_err());
+        let mut tighter = limits.clone();
+        tighter.max_refreshes = 0;
+        assert!(Budget::new(&tighter, 3, seed).is_err());
+        for field in 0..6 {
+            let mut excessive = Seed::default();
+            match field {
+                0 => excessive.decoded = 2,
+                1 => excessive.leaves = 2,
+                2 => excessive.refreshes = 2,
+                3 => excessive.accepted.terms = 2,
+                4 => excessive.accepted.groups = 2,
+                _ => excessive.accepted.ranges = 2,
+            }
+            assert!(Budget::new(&limits, 3, excessive).is_err());
+        }
+    }
+    #[test]
+    fn poisoned_accepted_owner_never_silently_resets_seed() {
+        let budget = Budget::new(&XetOptions::default(), 1024, Seed::default()).unwrap();
+        let _ = std::panic::catch_unwind(|| {
+            let _guard = budget.accepted.lock().unwrap();
+            panic!("test-only poison");
+        });
+        assert!(matches!(budget.seed(), Err(Error::State(_))));
+    }
+}

````

## Source: .agents/plans/EXPERIMENT_PROGRAM.md

Original UTF-8 bytes: 43912; SHA256 `28c362f37d22fcd85493958b6c41fec37cb4b3cd4a26219aa3b78f58cc67541d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — Project-neutral experimental program, H01–H43 (v4 reviewed)

**15 Oct 2026 — SERVOPT 36 future concurrency/benchmark experiments:** [vLLM parity and combinatory source acceptance](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md) adds SERVOPT-01..36: API queue and queued prefill token/byte/tenant capacity, source-correct KV/SWA/GDN/MLA hybrid prefix intersection, continuous decode + chunk prefill QoS/fairness, true physical kernel batch, target-only vs MTP/DFlash, SSD expert cache/source IO, captured graph pinning, distributed P/D state/collectives and multimodal task deadlines. Benchmark vLLM V1 and vLLM-Omni **with same exact model/quant/tokenizer/source semantics and comparable hardware/config**. Sweeps 1/2/4/8/16/32/64 requests, burst arrivals, prompt classes, 15-vs-18 grouped MoE rows, cache and cold/warm, full offered request denominator and rejected/timeouts. Optimize **valid committed SLO-goodput + fairness** not raw drafts or selected-work counts. No experiments were run under this documentation update.


**15 October 2026 — SSD STREAMING EXPERIMENTS (NOT YET RUN):** [SSD-01..32 source/physical tests](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) extend original H-series experiments: identical source tensor/StateRoot with full device residency, host RAM hot and cold SSD; real IO/H2D events and transient free, mock encoded-but-unexecuted GPU reader eviction race (DS4 #384), alternating Q2/Q4 expert sizes and prefill-cache wipe (DS4 #503), optional host hot slab vs O_DIRECT OS page-cache mode (#629), disk-only Qwen n-gram/Engram and private compatible disk KV, real OS MemAvailable/PSI/GTT plus active context, QoS P95. Benchmark **cold/warm, prefill/decode, 0/1/all resident, 1/4/8 sessions and direct-pointer versus compact-copy hit path**. Treat DS4/other prior-art published throughput as external baseline, not Zerogravity evidence; source-model and backend approval are required before disruptive 100GiB pressure tests.


**15 Oct experimental specialization extension (design, not executed):** [50 source-verified planned acceptance tests and experiment phases](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) add three complementary experiment families: **QTIP-01..14** packed derived quantization K2-K8 codecs, per-tensor shaped-noise calibration vs disjoint held-out source quality and original-vs-derived-vs-accelerated kernel error; **HYB-01..18** Qwen3.8/GLM5.3/MiMo2.6 source state, MTP/DFlash hidden-tap/partial RoPE/bidirectional/mask and rollback, prompt-specific acceptance and batch numeric near-ties; **GFX-01..18** gfx1151 Wave32/WMMA/LDS/int8/kernel tile, 15-vs-18 MoE row cliff, coherent-memory page-cache/host/driver headroom, source-valid first-image cold shape and physically canceled work. Required independent controls: same checkpoint+derived quant+tokenizer/template+source task, target-only versus draft+target, B=1/2/4/5/6/8, prompt types code/prose/chat/tools, active physical VRAM/OS MemAvailable, repeated ABBA and quality/accepted-token P95. The specialist-source author data are **external prior-art hypotheses** and its measured negative CPU MoE offload/graph/int8/lower-bit variants are retained as controls rather than automatic flags. No download, hardware pressure test or new architecture implementation in this documentation phase.


**2026-10-12 · NOT EXECUTED.** Every experiment below is a *falsifiable zerogravity hypothesis* and a means of deciding whether a capability belongs in the shared modular AI runtime. Source projects are cited separately as prior art in `../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` and historical research archives, not as components in our runtime. All performance figures discussed there are external author reports unless a real internal run exists.

## 0. Common scientific, safety and licensing protocol

Before every experiment define: exact hypothesis/mechanism, CPU/portable/reference baseline, candidate change, objective (TTFT/P95, committed outputs/time, memory/energy, quality), input/model/quant/task/position, device/topology, numerical modes and explicit failure/kill criteria.

**Comparable runs:** exact same checkpoint SHA, storage and effective compute quant mode or labeled quality delta, tokenizer/template, task inputs/context/prefill/output length/concurrency/draft acceptance, kernel/compiler/driver, GPU/NPU/CPU clock+power/firmware, physical available RAM and cache warm/cold. Warm/stabilize, use repeated ABBA/BAAB randomized blocks where relevant, report raw distributions with uncertainty and separate instrumentation overhead. An optional NPU trial must use the **same fixed fabric clock for ON and OFF arms**, with separate default-clock GPU-only result. Do not benchmark unqualified hardware that may hard hang or trigger physical OOM without consent and isolated safety measures.

**Evidence artifact:** test and source commit/diff, model/file/quant/adapter/program SHA, DeviceCapabilities, KernelPlan/CIR schema, topology/driver/firmware, task/regime, output/reference tensors and state hashes, numeric tolerances and quality, physical allocations/SSD bytes/cache hit/miss, event timeline, TTFT/P50/P95/P99/end-to-end, throughput in **committed outputs** (not proposed drafts), energy and power, repetitions and result `PASS|FAIL|INCONCLUSIVE|NEUTRAL`.

**License rule:** experiment about a prior-art algorithm may study any publicly available project and attribute it in the evidence atlas, but release Rust code must be independently written or explicitly reused from an audited MIT-licensed file after license/source-language/user policy approval. No extra model-specific engine may be embedded as fallback.

## Lane A — Kernel, compiler and execution legality

### H01 — Rust GPU architecture-specific kernel feasibility
**Compare:** scalar CPU oracle vs portable Rust GPU provider vs target-specialized Rust GEMV/GEMM and quant kernels on available real GPU. Shapes for decode1, 2–8 verification rows and large prefill; quant formats from actual source checkpoint. **Report:** generated ISA/occupancy/register/memory, numeric error, full model region speed and actual CPU overhead. **Kill:** impossible accurate ISA path or gross regression; keep generic provider.

### H02 — Compiled execution without IR hot-path tax
One model/operator graph through reference interpreter, compiled SIR→RIR→KIR/CIR bundle and hand-authored direct command control. Observe per-step host launch overhead and GPU idle; the typed compiler must not be traversed each token/frame. **Kill/pivot:** measurable unavoidable CPU/IR overhead, simplify direct bundle.

### H03 — Fusion vs occupancy and numerical legality
Benchmark norm→quant→projection, attention selector→compute, gated recurrent step, residual/hyperconnections, sampling/vocab head, decoder islands. Compare isolated micro and complete task; measure register pressure/LDS/occupancy, scratch, full-logit and state changes. Reject maximal fusion when slower or numerically unsafe.

### H04 — GPU-side routed expert row grouping
At same model/quant, vary uniform/skewed route overlap, C1/2/4/8/16/32 and decode/verify/prefill. Compare per-token expert execution vs device grouped assignments and numerically correct scatter. Capture unique expert counts and weight rereads, buffer use, P95 and compute. Reject if regroup overhead/latency dominates.

### H05 — Expert residency × batching × scratch tradeoff
Fixed resident/cold tier, CPU lane, SSD fetch, dynamic LFU/LRU/hotlist and scratch borrowing under one physical budget; vary cache size, route distribution and batch quantum. Measure *critical-path* expert misses and physical headroom rather than hit% alone. Revert any eviction or COW hazard.

### H06 — Huge immutable lookup tables with bounded IO
Random/scattered/repeated row reads from a very large table without demanding entire file in RAM; compare mmap/pread/async as supported, checksum/row hash and cold/warm SSD, dedup and GPU packing. Fuzz truncation/corruption; no silent fallback row zero or blocked GPU without error.

### H07 — Full hybrid-prefix cache sharing
40K repeated text/model prefix with 3+ state domains (KV, recurrence, indexer, n-gram, optional grammar/modality), 8 branches and modified tool/media spans. Compare full recompute vs verified immutable state snapshots vs COW/restore tiers. Same committed frontier/state/logits; quantify TTFT, cache bytes, cross-plan conversion and eviction/rebuild.

### H08 — Adaptive target speculation
No speculation vs trained multi-token/auxiliary draft, independent small draft and prompt-derived lookup at varied acceptance, context and concurrency. Measure *committed* tokens/total time, draft+verifier+rollback resource/energy, stochastic exactness and degradation on unpredictable output; disable when net worse.

### H09 — Mixed prefill/decode service fairness
C1/2/4/8/16/32 users with bursts, 256→32K input and output lengths; compare FIFO, token budget, chunking/priority and stage-aware group choices. Record TTFT/p95 inter-token/throughput/starvation with exact state.

### H10 — Long-context dense/sparse/latent/compressed attention
0/4K/8K/16K/32K/64K/128K/256K where physically possible, selection masks/ties, compression boundaries, chunk continuation. Compare GPU selector vs portable lowering and full official logits/long-retrieval quality. Do not reduce trained selection count to obtain speed.

### H11 — Physical unified/discrete memory and safe admission
Probe actual device RAM/VRAM/GTT/address space, OS/cgroup/caches/CMA, pinned staging and model/context estimates; simulate first then supervised real loads. No host OOM or kernel parameter changes. Compare CPU+GPU bandwidth contention against distinct PCIe dGPU tiers.

### H12 — Quant storage vs compute/derived model quality
Same source checkpoint with lossless packed format, lower-bit derived variant and temporary compute encodings; record per-tensor numerical errors, KL/NLL, greedy continuations, official task quality, memory and performance. Do not silently choose lower quality. Grouped expert/always-used router/mHC quality separately.

### H13 — Heterogeneous topology, activations and peer transport
Compare single device, layer/tensor/expert partition, fast/small GPU+large/slow GPU or CPU and optional remote links. Model exact mathematical expert selection. Measure activations/partial result traffic, collective cost, residency, communication and fault-recovery. A capacity improvement isn't necessarily a fair raw throughput comparison.

### H14 — Real multi-turn agent/task workload
Replay long common-prefix tool/reasoning/JSON sessions and optional images under model-native rendering, cache reuse, cancellation and concurrent requests; observe end-to-end task time, output quality and storage privacy. Don't publish private prompt content.

## Lane B — Compiler/state/architecture extension

### H15 — Multiple Rust kernel providers through one interface
Same semantic region and actual reference math compiled by distinct CPU/GPU codegen routes where available. Compare executable native instructions, semantic portability, compilation cost, ABI, safety and output. No model math/device coupling.

### H16 — Effect-token correctness under arbitrary graph rewriting
Generate legal and illegal operator reorderings with full KV, gated recurrent, sparse indexer and RNG/grammar. SIR verifier rejects effect conflict and preserves full state/logits. Test a zero accepted-prefix speculative branch and rollback after async IO failure.

### H17 — CIR async execution, liveness and deliberate failures
Fake multiple devices+IO with Acquire/Import/Pack/Dispatch/Wait/Join/Commit, random out-of-order completions, stale generations, early release, cancel and torn SSD reads; valid plans safe, invalid rejected. Physical overlap measured only after simulation.

### H18 — Actual eager vs device capture candidate choice
Identical source kernel variants under eager launch and graph replay with modeled pointer/dynamic residency; micro and full request may prefer different. Persist proven tactic keyed by source/driver/shape/mode. No captured stale allocation.

### H19 — Bounded pure-region optimizer/search
One pure fused region with no rewrite, finite hand heuristics and bounded equal-value rewrite search. Compare compilation RAM/time and full model performance, generated code size and legality. Turn off complex search when tiny or noisy gain.

### H20 — Common resumable prefix frontier for 3+ state groups
Frozen KV/recurrent/latent/indexer/media state at incompatible block finalization boundaries, model+quant/adapter/template+tenant hashes; select last common legal frontier and recompute suffix. Verify exact same-plan outputs and state; record cache cost.

### H21 — Modularity extension drill
Add a novel stateful operator, an unfamiliar device target and a new task program without editing unrelated HTTP, global allocator or model registries. Count changed crates/files/lines, compile time and all quality/fallback tests. **FAIL** if model-specific runtime fork needed.

### H22 — Product/task correctness and malformed inputs
ModelStore manifest/GC/download atomicity, GGUF dialect, tokenizer pre-type, grammar rollback, LoRA/adapter cache keys, embeddings/rerank pooling and image/audio spans. Fuzz malicious inputs, output streaming and privacy; task outputs only when supported.

### H23 — Accurate external benchmark comparison provenance
Separate same-model/checkpoint/quant/hardware/task from unmatched published source reports; record exact prior-art source URL and license in external research atlas and no false category/model matching. Do not use performance marketing thresholds without quality.

### H24 — Actual hardware safety profiles
Discover available device topology and driver instead of old assumed equipment; distinguish 64/92/96/128/192 GiB physical fit from virtual GPU GTT and latency. Optional multi-accelerator SoC requires firmware/clock/power risk gating and isolated tests. No automatic privilege escalation.

### H25 — Full model compiler runtime overhead
CPU reference/diagnostic IR interpreter vs precompiled direct bundle vs literal static runner across model prefill, decode and repeated task stages; measure hot-host dispatch, code cache, physical memory and full-model throughput with matched source. Must keep optimized model execution directly compiled.

## Lane C — Heterogeneous projection compute and additional task programs

### H26 — Intra-operator output row partition
A pure matrix projection partitioned by rows across synthetic independent compute providers, complete/ordered row-map join and optional first-use consumed spans; same tensor/state as unsplit oracle, including quant/bias/tails. No device/model name in operator math.

### H27 — Optional NPU-owned program/driver feasibility
Investigate Rust-authored accelerator program compilation/packaging and legal driver memory/callback API for a source-proven block-float tile shape. A foreign-language device kernel wrapped in Rust does **not** satisfy stated source-language policy. Report BLOCKED if no compatible Rust AIE/codegen exists; GPU-only remains normal.

### H28 — Cross-API DMA, machine-safety and power/fabric isolation
Fake mapped pinned buffers imported between GPU/NPU APIs, resource leases/event bridges/queue depth, deadline/quarantine, cancellations and stale completions. Real concurrent SoC GPU+NPU requires qualified firmware/driver/clock/power, dedicated recoverable test host and long soak; no automatic root clock/suspend changes.

### H29 — Same-source cooperative prefill benchmark
On physically qualified NPU+iGPU, compare selected independent projection row-slice variants vs GPU-only with identical model/checkpoint+quant, shapes, fabric clock/power/temperature and cache, including idle energy and p95. Separate clock-only effects; an upstream reported ~7% result is a hypothesis, not ours. Test long-context, C1/C4 and decode fallback.

### H30 — Block-float numerical and target task quality
Check packed exponent+mantissa bytes and reference dequant, full logits/recurrent/sparse state across partial GPU/NPU joins, strict numeric mode refusal and target sampled distribution where appropriate. Matching a small greedy-output set is not quality proof.

### H31 — Non-token TaskProgram pipeline reuse
One real or synthetic ASR/audio stream, embedding/pool, image-denoise iteration or TTS vocoder stage using the same SIR/RIR/KIR/CIR/ResourceArbiter and TaskCoordinator but **not** next-token-only state. Validate task-specific committed frontier, output schema, cancellation and device fallback. Separate model and codec release qualification.

## Promotion rules and evidence storage

- **PASS:** mathematically/legal/quality/resource-safe, and measured win or required correctness/usefulness with documented cost.
- **NEUTRAL:** correct but performance within noise; keep simpler default and explicit opt-in candidate.
- **FAIL:** changed model semantics/state, unsafe resource, unapproved numerical degradation, user-visible tail latency regression or source/licensing violation.
- **INCONCLUSIVE:** unsupported device, missing model checkpoint, insufficient repetitions or uncontrolled power/cache/hardware; no automatic promotion.
- **BLOCKED:** impossible required source-language, unavailable accelerator/driver or unqualified hardware safety; portable paths continue.

Future benchmark tree may include `bench/{inputs,configs,references,results,profiles,reports}`, with hash/provenance, redacted artifacts and machine metadata. Do not commit private prompts, licensed checkpoint weights, source-derived foreign kernels or credentials. For each completed experiment, update phase status, candidate registry, source prior-art catalog as needed and full agent handoff.

**Architectural success:** useful source-independent mechanisms can be tested and selectively plugged in; losing specialization does not leave model users without a runtime.

## v4 extension experiments — 2026-10-12, H32–H43

These tests are **NOT RUN** and extend existing H01–H31 without modifying their identifiers. They correspond to review findings R4-01…R4-24 and are written by independent mechanisms, not named external project internals. Use source artifact/quant/device/OS/license evidence; some tests need physical devices, others synthetic proof.

### H32 — Prepared StateRoot and atomic logical publication

Synthetic KV/recurrent/indexer/RNG plus a failed second `StateDomain::prepare`, out-of-order GPU events, cancellation and accepted speculative prefix length 0..k. Baseline old logical root read unchanged until all domain versions ready; exactly one root transition and stream output after publish. **Reject** domain-level public `commit`, false O(1) GPU recovery or partial output.

### H33 — GPU-native multi-step input progression

Compare per-token CPU router/sampler readback against prepared device-resident row map/stop/sampler and two or more device-native dependent provisional steps on a supported model/grammar mode. Measure host sync, GPU idle, P95, exact target sampling and GDN/indexer state. **Reject** fast-path claims when mandatory CPU synchronization still blocks every token; fall back honestly for unsupported modes.

### H34 — Hierarchical reserve and device-local allocation

Global ResourceArbiter reservation plus per-device asynchronous slab allocation, bounded 32-session concurrent loads, generation/epoch/lease correctness and delayed free. Compare single-global-mutex-per-operation baseline. Inject OS/cgroup pressure, driver buffer allocation failures and live pinned NPU buffers. **Reject** missing reservations, ABA and falsely claimed no-OOM guarantees.

### H35 — Dynamic shape polymorphism with guarded fallback

Rank-fixed bounded symbolic dimensions, ragged row offsets, sparse selected rows and prefill/decode/verify/encoder shape modes. Run 1,7,128,4096 rows with divisibility-required kernel and generic plan; invalid shape never reaches kernel. Measure code cache size and cold start. **Reject** unbounded variant explosion, unsafe padding or per-token compilation.

### H36 — Minimal TaskDriver versus generic workflow graph

Implement trusted text TaskDriver and mock image denoiser/encoder TaskDriver sharing semantic/kernel/resource/worker core. Compare CPU dispatch/alloc/complexity against a generic interpreted workflow DAG, count unrelated files touched. **Reject** a new model server or untrusted `ExternalAction` in core.

### H37 — Fused whole-region candidate with unchanged runtime

Register hand-optimized whole-layer/whole-model region as KIR candidate whose state/effects match generic SIR and whose driver handles are owned by DeviceWorker. Toggle candidate off and prove exact model/task continuation/fallback without new scheduler, sampler, service or allocator. Record full-task throughput and quality, not just tile GEMM.

### H38 — Output stream commit/reconnect semantics

TaskSession commits 1..N output chunks, client disconnects before/after acknowledgment, optional cursor replay and sequence dedup under output backpressure. Emit after PublishRoot; no uncommitted token/PCM/image chunk. Do **not** claim exactly-once network receipt without a durable ACK protocol.

### H39 — Distributed epoch fencing and partial failure

Two synthetic remote ranks prepare same model/adapter/state root, one ACK delayed/lost, coordinator restarts and stale attempt arrives. Ensure only fenced current epoch publishes state, output cannot be partially committed; document volatile vs durable recovery limits. This is P19 work, not Stage0 blocking GPU-only.

### H40 — Future parameter update compatibility (without training platform)

Synthetic `TrainStep` and GradientProgram update a new ParameterVersion plus optimizer state while pre-existing inference sessions stay pinned to immutable old model/adapter/compiled bundle. Compile reference forward/backward for one tiny op. **No** full autograd engine, training infra or public training promise required. Reject if SIR/TaskDriver/resource design must be rewritten.

### H41 — Key/cache migration and shape-band pruning

Upgrade driver/kernel compiler/semantic op schema and adapter generation. Verify separately versioned ArtifactKey, SemanticKey, StateCompatKey, BundleCacheKey and RuntimeBinding; invalidated code is rebuilt and old state safely converted/recomputed, no persisted raw pointers. Test bounded LRU and cold portable fallback.

### H42 — Fair mixed-task scheduling under resource/latency targets

Text prefill/decode, embedding encoder, synthetic denoiser and audio frame tasks share GPU/CPU/SSD; compare fixed batch vs dynamic target-time/fairness/pressure admission with same math and output. P50/P95/P99, throughput, memory, fairness, starvation and actual kernel overlap. Revert policy if no statistically significant whole-task benefit.

### H43 — v4 minimality and extension change-radius audit

Across H32–H42 count crates, direct implementation lines, new traits, circular deps, build time, hot-path allocation/CPU sync, actual codegen and model coverage. Add novel model op, new hardware provider and non-token task; prohibit duplicated model server/allocator/session owner. Compare one library+binary (or 2–4 crates) against over-modular alternative. **Reject** architectural complexity without needed correctness/performance benefit.

**Evidence:** all H32–H43 reports must list actual hardware/source SHA/model+quant/precision/state+shape+task, commands and failures. A mock does not qualify a physical NPU/GPU; a better microbenchmark does not qualify whole model performance, and a fully correct but slower provider can remain `FUNCTIONAL` without default selection.


## v4.1 cross-accelerator composability experiments H44–H55 (NOT RUN)

These experiments are **independent falsifiable source/hardware and fake-backend proof tasks**, not claims of working TPU/wafer/Graph ASIC support. Sources and product/protocol specifics are cited in `../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md`. Use one `RegionExecutionProvider` seam, exact model/task semantics and existing TaskCoordinator/ResourceArbiter; no separate model server or runtime.

### H44 — Coherent local ARM+GPU system versus HBM device admission

Simulate and (only if hardware accessible) physically probe a coherent local CPU+GPU ARM host with LPDDR/shared bandwidth and compare against discrete/HBM accelerator. Distinguish real OS/cgroup headroom, GPU addressability, zero-copy permissions and bandwidth. Cross-check source model/quant and native CUDA portable fallback. **Reject** single “Blackwell memory” capacity/throughput heuristic.

### H45 — Native block-scaled precision and multi-GPU topology

Reference-verify FP4 formats with different per-block FP8 vs power-of-two scaling and tensor-scale headers; separate source checkpoint identity, device packing and lossy arithmetic. Graph capture on a native GPU and 2-rank mock with peer/collective cost and missing rank failure. Optional physical discrete/hardware qualification, matched quality/P95/energy. **Reject** automatic lossy quant or falsely unified HBM.

### H46 — Fake graph-only accelerator (TPU-class execution contract)

Compile one **closed full model stage** through a simulated ahead-of-time graph compiler with static/guarded dynamic shapes and no exposed kernel-by-kernel launch. Return one program/event with prepared StateRoot and direct TaskDriver output; shape-miss fallback and cache fingerprint. **Reject** host micro-op interpreter or new service engine. Actual TPU XLA/PJRT/other compiler needs separate SDK+hardware approval.

### H47 — Fake wafer-scale programmed dataflow and streamed model weights

Simulate static tile/core placement and on-chip SRAM pipeline with immutable layer weight stream, compile/load/warm/weight IO critical path and stateless region output. Compare fake whole-stage stream to host issuing every GEMM. Correct semantic output with no new model code; stateful graph only if StateBridge passes. **No physical WSE claim without SDK/hardware.**

### H48 — Statically scheduled inter-chip dataflow and heterogeneous stages

Fake deterministic compiler-owned schedule across processing elements/on-chip SRAM with fixed link bandwidth and externally synchronized first-use barrier to CPU/GPU stage. Verify region cut, memory pressure, collective/RNG effect and no host per-op dispatch. **Reject** device pretending to support arbitrary dynamic op/branch when it does not.

### H49 — AOT accelerator backend with expensive graph compile

Simulate cloud ASIC graph compiler: bounded `CompileBudget`, static shape bands, optional custom low-level kernels, program upload, model/quant/source/revision pin, and runtime memory. Compare warm/cold TTFT and shape specialization explosion; ensure background compilation does not block first portable response. **Reject** claims that “model recognized” means `COMPILES`.

### H50 — Additional graph/low-level accelerator device classes

Add mock matrix/SRAM accelerator and low-level programmable dataflow device through same `RegionExecutionProvider` for one reference encoder+LLM subgraph; represent device ISA/firmware, graph and source constraints and optional Ethernet interconnect. Test provider qualification without copying foreign source/C++ compiler, and fallback. Actual physical target testing only after access.

### H51 — Opaque stateful graph refusal and exact snapshot bridge

One fake compiler graph with private provisional KV+GDN+RNG state, snapshot/restore/recompute and true event Ready/Failed; another with only unrecoverable opaque in-place model state. First can satisfy PreparedRoot, second **must not** be eligible for general stateful session/speculation. Stateless encoder can still work. Check accepted prefix, cancellation and late event.

### H52 — Region granularity and program residency optimizer

Same semantic model/tensor graph compiled into native small kernels vs captured GPU graph vs whole AOT stage vs static program with build/load/compilation amortization and external weight stream. Compare first output, P95, resident bytes, topology IO and full task; choose the lowest measured critical-path plan without inventing universal winner. Bound number of cached variants.

### H53 — Remote/cluster topology and epoch fencing

Fake rank/wafer/pod fabric, link congestion, collective groups, model/quant/state schema mismatch, lost ACK and coordinator epoch restart. No publication of partial rank state, no assumption of one HBM pointer across rack, no false exactly-once remote delivery. Remote creds/privacy/egress are admission restrictions.

### H54 — External model API versus native program boundary

Attempt to register a text-only remote inference API as native RegionExecutionProvider: it must be rejected for lacking lowerable model op/control and state compatibility. It may become explicit authorized `TaskServiceAdapter` with separate user privacy, task/stream model and cost, but must not be described as running zerogravity semantic kernels on remote hardware.

### H55 — Backend extensibility, licensing and alternate speculative algorithm

Add a **fake graph-only provider and fake static fabric provider** alongside a native CPU/GPU provider; verify no changes in ModelDefinition/TaskDriver, StateRoot owner, service/ResourceArbiter and no new durable IR. Source-language/driver/device compiler status must be truthful. As a distinct optional experiment, adaptive multihead parallel draft decoding (the Cerberus *algorithm*, not wafer-scale hardware) can register a DraftSource and run target distribution/cost tests without changing base model semantics or decoding engine.

**Promotion ladder:** architectural mock `ARCHITECTURE_COMPATIBLE` is not real SDK readiness, compilation, state safety or physical performance. Exact artifact/model/quant, driver, program compiler/SRAM/cache, shape and sample/quality, physical memory, measured p95 and power must be recorded for any genuine hardware claim. All H44–H55 remain NOT_STARTED until actual runs.

## v4.2 new independent distributed execution experiments H56–H70 (NOT RUN)

All experiments use one model/TaskDriver/StateRoot plus independent ComputePlacement, ParameterPlacement, StatePlacement, CachePlacement and TransferPlan, with authenticated peers and measured network. Model and topology names are descriptive capabilities, not imported source engines. No physical machines/tunnels may be accessed without explicit user authorization. Distinguish **simulated** from real source/checkpoint/physical measurements.

### H56 — Standalone zero-network fast path

Verify no peer discovery, tunnel handshake or global cluster allocation in standalone CPU/GPU work; compare StepProgram host/physical memory overhead before/after optional distribution module. Fail if cluster code is mandatory for one model.

### H57 — Two-node layer pipeline with layer-local state

Synthetic then actual two worker groups, disjoint layer ranges and matching per-layer KV/GDN/state, activation boundary transfer, same full logits/task output as single-node source oracle. Measure C1 sequential decode and chunked prefill overlap, extra RTT and capacity gain.

### H58 — Remote immutable prefix cache versus recompute

Remote prefix directory may lie/be stale, full-state source/quant/adapter/tenant hash validated, measure state hydrate bytes/time/p95 across direct and relayed links, compare local recompute and previous committed root when missing.

### H59 — Separate prefill and decode machines

Reserve compatible decode destination and transfer **entire required hybrid state** (KV+recurrent/indexer/ngram/grammar/modal), with chunk checksum/Ready/fail. Compare prefill/ITL tail isolation vs large network TTFT overhead under same task, not “always more throughput”.

### H60 — Remote attention beside KV versus remote KV read

GPU A does projections/FFN; node B owns KV and attention core. Compare per-layer compact query+output to raw KV bytes per-step and cache migration once, on simulated slow LAN/tunnel and real physically qualified links. Reject chatty plan with high p95.

### H61 — Sequence-token KV attention partition algebra

Shard causal KV token intervals among nodes; compute local max/Z/weighted numerator and global stable softmax result. Validate all-mask, head/GQA, position, numerical drift and state update; reject unknown sparse/compressed/global topk/GDN without separate algebra.

### H62 — Attention/FFN and routed expert owners

Source model with per-layer attention+KV state one node and FFN/MoE experts on another, GPU-side route stable, activations/partial returns, varying route overlap and hybrid attention type. Measure p95 and stage imbalance, reject slow-link negative.

### H63 — Tensor/head parallel and fast fabric collectives

Simulate and later physically qualify collective across 2+ GPU nodes, exact shape/model/quant, graph/driver/rank and epoch order, multi-hop bandwidth, lane congestion and wrong rank failure. Do not assume RDMA across ordinary encrypted tunnels.

### H64 — Task-stage distribution beyond text

One synthetic audio encoder/ASR and denoiser/vocoder TaskDriver across peer region stages with task-specific output/committed frontier, source/tenant permission and transport cost; no fake next-token stage or second model server.

### H65 — Encrypted tunnels, direct versus peer/relay topology

Transport emulator then approved real tunnel tests for direct, peer-relay, high RTT, loss/MTU/crypto CPU load and link failover. Compare bounded authenticated TCP/QUIC implementations, stream priority/credits and physical copy paths; VPN not GPUDirect.

### H66 — Trusted node enrollment and adversarial peers

Unknown peer/forged capability, revoked cert, wrong tenant/model license, malicious oversized frame, duplicated command, remote arbitrary shell attempt and untrusted device program all rejected before memory allocation/source export. Log redaction and scoped RDMA leases.

### H67 — Coordinator/network partition, epochs and state shards

Old coordinator lease expires, worker holds Prepared candidate, lost CommitNotice/ACK, late peer rejoin and provisional DMA. Exactly one session writer or fail-stop, old root intact/compatible replay; don't promise network exactly-once or durable failover without separate implementation.

### H68 — State shard migration and topology replan

Topology link degrades/disappears during active work, new plan proposed but not applied until committed TaskFrontier. Reserve new peer, transfer/verify state, fence old generation, PublishRoot ownership epoch and retire; inject failure before and after publish, no wrong output.

### H69 — Performance/capacity Pareto across multi-machine placements

Physical source checkpoint/quant/driver and measured link compare colocated if fits, contiguous layer split, remote prefix, attention/FFN, sequence shard and expert/tensor sharding at C1/C8 and 4K/32K/128K. Report capacity-only wins separate from raw tokens/s, TTFT/P95, energy and WAN crypto/egress.

### H70 — Future-topology composability change-radius audit

Add a new trusted accelerator node + TransportProvider + cache/state placement mode to **same** SIR/TaskDriver/RegionExecutionProvider/CIR/StateRoot/ResourceArbiter with no duplicate model service/allocator, no new durable IR, and zero-change standalone startup. Record changed modules, codegen/compile/run overhead and safety checks.

**Promotion levels:** model-architecture simulators cannot prove actual tunnel bandwidth; link-tested transfer isn't state-safe execution; mathematically correct model run isn't performance-qualified until physical p95/energy/batch tests. New remote worker profile remains opt-in and source/tenant-restricted until qualified. Normative docs: `../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md` and `../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md`.

## v4.3 multi-model local runtime, router, config and OpenAI API research H71–H86 (NOT RUN)

These are **unexecuted** independent tests for zerogravity's own built-in model router and user-facing experience. One TaskCoordinator/ResourceArbiter/RegionExecutionProvider, one TaskDriver per session and versioned exact model identities across `run`, SDK and `serve`; no external inference engine or mandatory gateway. Named CLI/API source comparisons are research-only in `../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md`.

### H71 — Zero-config one-command local model run

Clean supported machine and source-verified local GGUF/model sidecars, no config/account/cloud/daemon; `run <path>` uses CPU then actual eligible GPU, correct prompt rendering/logits/output state/usage with concise terminal. Compare CLI direct vs in-process library to detect hidden per-token proxy cost. Refuse unsupported model/quant/physical capacity honestly.

### H72 — ModelCatalog, immutable artifact and alias revision

Two different checkpoint SHA/quant/Tokenizer/Adapter artifacts, two profile aliases and one alias hot update during active stream; verify old TaskSession stays bound to original source/numerics/StateRoot, new request uses new revision, model deletion/GC preserves referenced blobs. No silent target change.

### H73 — Concurrent model load deduplication and cold-start fairness

100 simultaneous requests for one cold model produce **one verified load+compile/warm reservation**, bounded waiter queue/deadlines and correct first valid output. Meanwhile another warm model continues low-P95 serving. Simulate corrupt artifact, driver compile failure, cancel and partial load rollback; no 100x physical allocations.

### H74 — Many-model simultaneous physical residency

1/2/4/8 different registered models and 0–N physically resident models under finite HBM/UMA/RAM and per-session context pools; actual budget counted (weights+code+draft/adapter+KV+scratch+OS). Measure cache hit, warm time, TTFT, safe overcapacity queue/unload and no OOM or hidden lower-quant swap.

### H75 — Idle eviction, keepalive and pinned session safety

Warm/idle/pinned/scale-to-zero model with ongoing TaskSessions, speculative provisional state, device in-flight events and remote data; evict only after leases/quiescence and compatible session checkpoint retention. Alternate giant/small models to stress cache thrash; explicit pin can be refused if physical reserve cannot be granted.

### H76 — Equivalent replica balancing versus different-model choice

Two replicas exact same model/quant/adapter and a third different model; sticky long-session prefix StateRoot, request queue/P95 and device headroom. Compare round-robin to cache-aware/least predicted completion. Exact alias never chooses third model; user-configured policy alias may select it and reveals actual source.

### H77 — Sharded deployment versus independent source model replicas

One source model split across two trusted peers, two full replicas same artifact on different peers and two different models on same peer. Validate DeploymentId/ReplicaGroupId/ResourceArbiter accounting and remote StateShard owner affinity; no treating two shards as two independent load-balanced models.

### H78 — Model profile precedence and provenance

Built-in safe device defaults, global config, model/profile, alias/deployment, CLI and per-request sampler; test conflicting context/quant/adapter, invalid keys, hard model/tenant/physical limits, env precedence and ability to print `config show --effective` with value/source/guard. No silent immutable model mutation or ignored quality-affecting fields.

### H79 — OpenAI Chat Completions exact SDK conformance

Official pinned Python/JS client uses local base URL, `GET /v1/models`, chat nonstream/stream with correct ChatChunk SSE, model identifier, usage and stop/length/tool call data/finish. Compare raw generated sequence to direct `run` under same checkpoint/sampler and root. Different model roles rendered by exact verified templates; malformed/unsupported params get typed errors.

### H80 — Responses subset, stateful retention and event correctness

Stateless `/v1/responses` request and distinct typed `response.*` streaming events, correct created/item/delta/final output and failure. `store`, `previous_response_id`, conversation state and cloud built-ins refused when not implemented; when enabled later, authenticated StateRoot/retention/cursor exactly scoped. No hidden permanent history or tool execution.

### H81 — Embeddings and extra TaskDriver endpoint qualification

Actual source-verified encoder + pooling/dimensions/normalization and model alias/tenant, input batching, usage and correct SDK response; compare independent reference outputs. Rerank/audio/image endpoints unsupported unless their own TaskDriver/codec/model quality passes. Never fake vector embeddings from arbitrary text completion head.

### H82 — Local listener versus public API security and native admin

Loopback-only default/no network if `run`, remote bind denied unless explicitly configured auth/TLS/tenant policy; authorized inference cannot invoke model import/remove/unload/admin; unauthorized model names/paths/peer details hidden; reject malicious JSON, media URL SSRF, untrusted model code and auto remote pull. No dummy-key auth on public network.

### H83 — Multi-model parent TaskProgram roles and resource admission

One explicit target+draft and one encoder+reranker pipeline with independently named model artifacts and separate memory/compiled-plan identities; one parent task/session budget and one TaskCoordinator committed output. No replica-confusion, incompatible source artifacts rejected, semantic sampled verifier model intact.

### H84 — Warm/cold replica placement and dynamic per-model batching

C1/C2/C8/C16 active sessions across tiny model, giant MoE and encoder while a cold new model loads; measure TTFT/P50/P95/P99, real throughput, queue fairness, GPU device idle, resource headroom, native compatible row batching and per-model cold model load/evict policy. Kill global per-token routing mutex or starvation.

### H85 — Model/router hot reload, client disconnect and server restart

Change alias/source/profile/driver while streaming, unload/stop after cancellation and in-flight DMA, async SSE consumer disconnect, restart with immutable source catalog but **without** serialized raw GPU pointers. If safe SessionState persistence configured, validate full state restoration; else report session expiration rather than replay made-up token history.

### H86 — End-to-end minimality, parity and observability

One model `run` in process, `serve` same engine over loopback and Rust SDK TaskRequest produce same source-correct output and honest ModeReceipt under matched sampling. Measure per-token CPU/HTTP/router overhead and startup/config complexity. Add a second model/replica/embedding alias **without changing** SemanticProgram, KernelProvider, TaskCoordinator, DeviceBackend or independent serving engine; compare touched files/crates and CI. A straightforward user never requires choosing GPU CUDA tiling, NPU fabric mode or remote placement before running a supported model.

**Qualification:** a mocked catalog/router passes architecture-only; real CPU/GPU source model and official tested SDK versions required for functional/API marks; accurate P95/energy under same checkpoint/hardware for performance-qualified. All experiments H71–H86 are PLANNED, **none run**. New optional Stage S0-SERVE-A…J tests the small prerequisites.


````

## Source: .agents/plans/IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md

Original UTF-8 bytes: 12534; SHA256 `dda399a4b79a677d6b2ae1e15b790257720dcdcd345a786c01d613f8821a40b1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Large immutable source loading: bounded basic-unit proposal

Status: PARTIALLY IMPLEMENTED (18 October 2026).
The verified owned-shard source-only implementation now lives in
`crates/zerogravity-artifacts/src/verified_shards.rs` and
`source_weights.rs` with tiny original-format regression fixtures at
`crates/zerogravity-artifacts/tests/verified_shards.rs`.
The later original trained Qwen3.8 27B model, SSD streaming,
full original tensor/mathematical binding and physical-device
qualification are NOT IMPLEMENTED. See the
[model-execution TODO](model-execution/TODO.md) for accepted gates.
The historical proposal below is retained for implementation context. The [accepted checkpoint](accepted-baseline/TODO.md) settles historical verification; [plan TODO index](TODO.md) is required for implementation. Revalidate source observations against the live workspace before editing. Artifact files below belong to `crates/zerogravity-artifacts/`.

Status of original review: design only. No production change,
limit increase, dependency addition, benchmark, model acquisition or GPU work.

Propose **verified owned shards first** in the existing SourceWeights owner.
Preserve the current4GiB GGUF `open` policy. Add an explicitly budgeted, expected-
digest constructor using the same Artifact parser, SourceIdentity issuer,
TensorSource and ResourceArbiter. Do not replace the model/runtime or add a
parallel source cache. Mapped residency is a later separately qualified storage
implementation, not an implicit way around host admission.

## Findings and boundaries

- `source_weights.rs` currently materializes one GGUF descriptor into Vec and
  issues exactly one identity. It rejects Safetensors and >4GiB. Its borrowed
  bytes remain stable for the owner lifetime; SourceIdentity is allocation
  identity, not a digest or a pathname. Existing loaded-model Arc sharing stays.
- Artifact::open already parses bounded F32/F16/BF16 Safetensors directories.
  It rejects duplicate names, shape/byte mismatches, overlapping/gapped ranges
  and unindexed data. into_bounded_source reuses the original descriptor and
  reparses copied bytes. Its documentation explicitly excludes concurrent-write
  snapshot guarantees. Reuse these checks, not another Safetensors parser.
- The merged Qwen metadata unit provides a bounded SafetensorsIndex parser with
  exact integer total_size and safe basenames. Metadata contracts do not
  establish full shard/header verification; reuse the existing parser.
- Current TensorSource returns lifetime-bound slices. Any implementation must
  keep the entire backing region alive and unchanged for every borrow and
  prepared GPU owner. An evicting per-tensor buffer cache cannot implement this
  interface by returning a slice whose storage later changes.
- Native HF matrices use output-first source shapes; existing GGUF CPU and
  prepared-GPU bindings expect their own explicit columns/rows interpretation.
  BF16/F16 codecs exist, but that does not authorize reinterpreting raw HF axes,
  Qwen grouped heads or MoE layouts. Preserve original header order and add a
  checked family binding; do not silently reverse metadata or transpose bytes.
- Local read-only meminfo sample reports28,581,696KiB physical RAM. The published
  Gemma26B index declares51,611,872,412 payload bytes; Qwen27B55,562,855,904.
  Neither fits a fully owned resident copy on this local host. Available RAM,
  swap, address space and sparse file length are different resources. Refusal
  is the correct result until an appropriate admitted host/profile is available.

## Immutability decision

Opening read-only, holding a descriptor, advisory locks, permissions, inode/size/
mtime checks, MAP_PRIVATE and map_copy_read_only do not freeze another writer's
inode. Path replacement changes future opens, while in-place writes/truncation
can affect the existing descriptor/mapping. Hashing once then mapping the
mutable original remains unsafe. memmap2 explicitly makes all file-backed map
constructors unsafe for this reason, including COW maps. [Primary safety docs](https://docs.rs/memmap2/0.9.11/memmap2/struct.MmapOptions.html#safety).

For the first unit, copy from the already opened descriptor into exclusively
owned Rust bytes; compute SHA256 of those actual copied bytes, compare the
caller-authorized expected whole-shard digest and length, and reparse those same
bytes. A concurrent writer can cause rejection; it cannot change a published
Vec. Before/after stat checks are useful diagnostics, not an authenticity proof.
A self-computed digest without an expected trusted manifest identifies captured
bytes but does not establish the intended official checkpoint or a coherent
multi-file revision. Require expected config/index/shard digests for the new
verified API. Caller authorization/provenance is separate from cryptographic
integrity; a digest supplied by an attacker authenticates nothing by itself.

Future mapped options require an enforceable immutable backing and expected
identity: supported fs-verity files with verified measured digest/metadata, or
an owned backing created/copied and sealed against write/grow/shrink before any
read-only mapping. fs-verity verifies blocks and makes enabled files read-only;
memfd seals protect its own copied backing, not the mutable source file. Neither
option automatically proves expected model identity, fitting working set,
platform support or an admission policy. No seal/fs-verity activation, privilege
change or filesystem configuration is proposed here. [Kernel fs-verity](https://docs.kernel.org/filesystems/fsverity.html), [memfd semantics](https://man7.org/linux/man-pages/man2/memfd_create.2.html).

## Smallest concrete next CPU source unit

1. Add an internal immutable source-storage enum to SourceWeights: retain single
   owned bytes, add owned shard allocations. Each tensor binding stores its
   shard index and original TensorInfo. Keep metadata/private backing immutable;
   issue one existing checked SourceIdentity only after the bundle validates.
2. Add checked SourceOpenLimits/ExpectedSourceManifest and an opaque prepared
   source plan owning the opened Artifacts, expected metadata and resource
   lease. Public API shape: prepare_verified_sources(opened_shards, index,
   expected_manifest, limits, admission) -> SourceLoadPlan; then
   materialize(self) -> SourceWeights. One synchronous cold transaction.
   Do not add a session/root issuer, generic runtime or loader registry.
3. Initially accept explicitly supplied opened descriptors rather than infer
   filesystem authority from untrusted index paths. Artifact needs a narrow
   descriptor constructor (declared format; parser reused). A later path
   convenience wrapper must open relative to an owned directory capability,
   refuse symlink/magic-link/escape races, and retain those exact descriptors.
   A check-then-open canonicalize/symlink_metadata sequence is insufficient.
4. Bind every shard's actual header names to its declared index shard; reject
   missing/extra/misassigned/duplicate cross-shard tensors, unsupported dtype,
   incomplete shard sets and inconsistent expected file/payload totals. Header
   payload sums exclude file headers; whole-file lengths include them. Hashes
   cover whole files. Validate model tensor roles separately using the newly
   merged Gemma/Qwen contracts. Do not treat index names as header evidence.
5. Reserve the checked total owned payload/file allocation plus metadata/header
   and bounded I/O staging envelope BEFORE materialization. Use fallible reserve
   and a fixed read bound; no read_to_end growth beyond the admitted maximum.
   Account allocator capacity/rounding conservatively; avoid an extra full-shard
   Vec-to-box copy. Sequential per-shard filling still retains previous shards:
   peak is total retained bytes plus bounded transient work, not largest shard.
6. Reuse process_shared ResourceArbiter with an explicitly configured host-RAM
   physical key and original lease held by SourceWeights until its last Arc
   owner dies. Synchronous CPU copy does not submit a GPU event: do not fake
   Ready, fence devices, mint GPU completion or release a still-live source.
   Keeping a conservative peak envelope charged for the source lifetime is a
   bounded first implementation; do not invent partial-lease splitting now.
7. Legacy SourceWeights::open keeps its default4GiB rejection. Before claiming
   process-wide host admission, route legacy source allocations and CPU state/
   scratch/host staging into the SAME configured host domain or reserve their
   conservative existing occupancy explicitly. A new-loader-only ledger cannot
   account for older untracked allocations. UMA and discrete staging need an
   attested shared-memory-pool policy; do not invent independent full budgets
   for CPU RAM and GPU RAM when they consume the same physical pool.
8. Publication happens only after all fallible source/hash/header/index checks.
   Failure drops private buffers and synchronous reservations, exposes no
   TensorSource/cache entry and changes no session roots. Existing mutable
   `bytes(&TensorInfo)` slicing must bind the name back to this owner's exact
   stored header/shard, refusing foreign altered metadata; no ambiguous absolute
   offset may accidentally select a different shard.

New materialized Safetensors storage alone does not enable the old GGUF CPU
model constructor. Family code must bind original names/axes and reuse the
single shared mathematical engine under an explicit numerical profile. Direct
F16/BF16 decode to FP32 is not proof of a framework's BF16 intermediate rounding.
No silently selected CPU/GPU fallback or trained support label is warranted.

## File/API ownership proposal

- artifacts/src/artifact.rs: descriptor constructor and bounded exact read/hash
  seam; preserve existing parser behavior and same-descriptor tests.
- artifacts/src/source_weights.rs: owned shard backing/binding, new explicit
  constructors, TensorSource implementation and retained host lease.
- artifacts/src/sharded.rs: reuse incoming index codec, add binding verification
  beside it; do not make the metadata parser open weights.
- artifacts/Cargo.toml: exact already-locked/cached sha2=0.10.9 if approved.
  No core dependency or unsafe mmap needed for this first unit.
- CPU adapter: only explicit original-layout bindings/borrowed source access;
  preserve current source-math bodies. Family integration is a separately
  verified follow-up, not a changed generic CPU runtime.
- Core arbiter: no duplicate type/authority. Host budget setup needs an explicit
  owner/policy decision before production enabling; source unit must not pick a
  machine limit from transient MemAvailable or treat swap as physical capacity.

## Acceptance with tiny/sparse fixtures

LS-B01: two tiny BF16/F16/F32 shards; strict index/header/name/dtype/offset/shape/
payload/file/digest agreement, source bytes and decoder bits. Wrong/extra/missing
shard, foreign header, duplicate cross-shard role, wrong hash and truncated read.
LS-B02: same-length pathname replacement after open; loaded bytes come only
from original descriptors. Writer mutates/truncates during copy: reject digest
or read; writer mutates AFTER successful load: borrowed source stays bit-exact.
LS-B03: mocked/controlled allocation/read failures, concurrent admitted loads,
insufficient envelope and checked u64/usize overflow; exact charge rollback,
last-Arc source lifetime and no root/cache publication on rejection.
LS-B04: sparse >4GiB file/header plans exercise offsets/checked limits and
pre-allocation refusal without reading/hashing/materializing4GiB. This is not
an actual >4GiB load qualification. Tiny explicit-limit overrides exercise the
same positive/failure paths; default4GiB policy remains unchanged.
LS-B05: source identity unique per independently loaded bundle, shared per Arc;
old GGUF API/source bytes/math/checkpoint tests stay exact; CPU-free family
compiles and dependency graph stays SDK-free. Test symlink escapes only when
the capability-based path wrapper exists; descriptor API must not claim it.
LS-B06: later explicitly authorized checkpoint access plus sufficient attested
host budget: actual >4GiB whole-shard digests/headers, measured allocation peak
and lifecycle, then independent trained model numerical qualification. No model
is available by implication; the existing0.8B/2B GGUFs cannot prove the new large
BF16/sharded profile. Acquisition and permission remain separate prerequisites.

````

## Source: .agents/plans/IMPLEMENTATION_ROADMAP.md

Original UTF-8 bytes: 41396; SHA256 `a976ec6153dd882d37c5478944c7aa10bb63d28e55d9c236ce49b5f8ff6381fa`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — architecture roadmap P00–P23

This is a design reference. Current work and completion live only in
[the plan TODO index](TODO.md); use [AGENTS.md](../../AGENTS.md) for the workflow.
The phase IDs and architectural requirements below remain unchanged.

## A. Program objective and hard invariants

Build one native Rust **AI TaskProgram + semantic compiler + compiled device executor** that accepts independently verified model/artifact types and replaces closed computation regions with optional specialized implementations. All phases respect one session/state owner, one physical resource arbiter and one async command/event graph, without creating a new model-specific engine. Source projects and research publications are not code architecture dependencies; MIT code reuse is optional after per-file audit, non-MIT sources stay prior-art.

**Phase complete only when:** actual source/test/model artefacts exist, independent reference and numerical quality pass, devices/IO/state are safe, and documentation records exact commands and model/source SHA. A phase may be functionally VERIFIED without being PERFORMANCE_QUALIFIED. No unvalidated kernel/project benchmark counts as zerogravity results.

### Critical minimum path

```text
P00 workspace/source & truth
  ├── P01 device/codegen feasibility (CPU/real GPU; optional NPU independent)
  ├── P02 portable mathematical/state oracle
  └── P03 artifacts/tokenizer/quant
             |
      P04 minimal effectful SIR + TaskProgram stub
             |
      P05 legal RIR + kernel registry + CIR event/resource proof
             |
      P06 CPU direct/reference executor + P07 one GPU backend
             |
      P08 first REAL model→task→generated output vertical slice
             |
      P09 serving/stages + P10 state/physical resources
             |
      P11 stateful/sparse semantics + P12 routed experts
             |
      P13/P14 selected heavy model/device specializations
             |
      P15 target-authoritative speculation + P16 task-aware batches
             |
      P17 operator/model family growth + P18 multimodal AI TaskPrograms
             |
      P19 heterogeneous/distributed topology
             |
      P20 tuning, P21 security/release, P22 qualification, P23 extensibility
```

Dependencies are minimum contracts, not proof that every future operator needs all preceding phases. Work can overlap only where interfaces and validation are isolated; avoid premature construction of all modules.

## ZG-P00 — Lean foundation, source policy and reproducible build

**Goal:** build a minimal Rust workspace that genuinely compiles and runs tests on CPU; pin toolchain and dependencies reproducibly, including optional device features. Prepare versioned data/semantic/lease identity types without adding 25 empty crates.

**Deliverables:** project metadata, Cargo lock, dev shell/Nix where appropriate, CPU-only smoke executable, test fixture policy, source/NOTICE/license audit, benchmark harness skeleton and hardware capability probe. A model/toolchain/source SHA ledger and exact trusted artifact path policy. Distinguish device execution source-language requirement from vendor runtime FFI; not all policy questions can be resolved by code.

**Gate:** fresh checkout builds CPU without GPU, no unknown dependencies or unreviewed code imports, static source policies and verified file constraints, measured exact local device inventory. **Out of scope:** implement full IR/backend architecture in one massive scaffold.

## ZG-P01 — Architecture-specific Rust kernel and hardware feasibility

**Goal:** demonstrate at least one real physical GPU kernel compiled from Rust-authored source with correct reference parity, and evaluate access to critical ISA/quant/tiling.

**Deliverables:** small shape matrix (dense decode GEMV, prefill GEMM, low-bit packed blocks, route+gather, simple recurrent update), device capability/driver/physical memory probe, generated ISA/register occupancy and reference benchmark. Multiple optional Rust codegen/kernel-provider backends may coexist; experimental target compilers cannot force the whole core to use unstable toolchain. Driver and vendor math FFI policy explicit.

**Gate:** correct physical GPU execution and plausible path to native instruction efficiency; if an extreme backend fails, portable GPU/CPU plan remains possible, and specialization is marked blocked. Optional cross-accelerator NPU programming and power-fabric qualification run in **separate Stage N gates**, never as release prerequisite for CPU/GPU.

## ZG-P02 — Independent mathematical and state-transition oracle

**Goal:** canonical Rust implementation of all operators needed for the first full model plus small synthetic stateful pipeline.

**Deliverables:** scalar/FP32/FP64 diagnostic norms, quant matmul, causal attention, gating/router/activations/positioning and reference KV/recurrent/indexer effects; `StateTransaction` begin/commit/abort/fork/replay; reference RNG/sampling where needed. Golden vectors and generated property tests for chunk/branch state and illegal sparse indices.

**Gate:** independent official/reference parity and blocked invalid state/order transformations. New semantic ops always require reference truth before optimized kernels.

## ZG-P03 — Trusted artifacts, quant codecs and rendering

**Goal:** read a real model checkpoint and produce canonical token/media input without executing external model repository code.

**Deliverables:** bounded GGUF/safetensors + shard integrity, quant codec/block manifest, tokenizer/pre-type and model-native template vectors, special tokens and structured input; content-addressed artifact manifest/store skeleton and safe offline resolution. Distinguish checkpoint storage quant, lossless physical packing and optional lossy compute format.

**Gate:** malformed corpus fails safely; real checkpoint tensor names/shapes and tokenizer/role rendering match official fixtures. No model support from a parser alone.

## ZG-P04 — Typed Semantic IR and minimal TaskProgram

**Goal:** represent pure tensor math and mutable state as explicit effects, with a minimal typed task stage/loop wrapper.

**Deliverables:** SSA values, symbolic logical dimensions, numeric policy, state-effect tokens and schema, one simple autoregressive task and one synthetic non-token bounded iterative task; IR dumper/verifier. Canonical CPU reference lowering.

**Gate:** illegal reorders/alias/position breaks rejected, full state frontier after speculative branch correct, same task operators can be interpreted in diagnostic mode and lowered without semantic difference. Keep grammar/media and novel model ops outside initial scope unless needed by vertical slice.

## ZG-P05 — Legal region optimizer, kernel registry and async CIR

**Goal:** minimal executable compiler that can substitute a whole region without a hot-path semantic interpreter and can enforce async memory/resource safety.

**Deliverables:** RIR fusion and quant/layout transformation legality, shape-band kernel candidate registry (KIR), optional static provider, CIR event/resource command DAG with buffers/transfers/dispatch/wait/commit/abort, liveness/alias/generation checks; tiny `ExecutableBundle` cache and safe fallback conversions.

**Gate:** reference parity for optimized vs nonoptimized synthetic graph, forced specialized-kernel miss, simulated IO failure/stale mapped allocation/cancel, no compiler AST walk per token. Do not implement an unlimited equality-saturation optimizer at this phase.

## ZG-P06 — Useful portable Rust execution backend

**Goal:** correct CPU model executor with direct prepared kernels, bounded allocations and no GPU dependency. Optional SIMD optimized path after reference tests.

**Deliverables:** valid tensor ops/quant matvec, one text task session with prefill/decode and sampler, CPU StateDomainCodec and task loop, cancellation, diagnostic state checksums, resource refusal. Clear unsupported-op errors and first minimal SDK.

**Gate:** one real small model can generate meaningful reference-checked output on CPU, including prompt continuation and stop/cancel. Source quality reference recorded.

## ZG-P07 — Accelerator providers, streams, leases and kernel ownership

**Goal:** one real GPU DeviceProvider capable of physical allocation/copy/launch/events, optional graph capture, persistent kernel/layout handles and safe failure; second GPU target smoke to guard portability.

**Deliverables:** `DeviceCapabilities`, `ResourceMemoryAdapter`, `EventBridge`, typed leases, async worker and graph replay where supported. Portable tensor kernels and optional fast kernel variants. Generic device capability negotiation and fallback.

**Gate:** CPU vs physical GPU parity on real operators and device-loss/cancel tests; no stale buffer free and no hidden global driver synchronization. NPU/other coaccelerators may remain unqualified.

## ZG-P08 — First complete real model and TaskProgram vertical slice

**Goal:** connect **artifact→model definition→SIR→RIR→KIR/CIR→CompiledBundle→TaskCoordinator→output** for one model/task on CPU and eligible GPU.

**Deliverables:** source-verified checkpoint+quant, tokenization/template, complete architecture operator implementation, state+sampler, CLI smoke (optional HTTP), `inspect --explain-plan`, forced disable of a specialized region and correct general fallback.

**Gate:** actual model logits/continuations and state parity, repeated prompts and chunk boundaries; memory-fit estimate; no duplicated model server or hidden external inference runtime. This is the primary architectural viability milestone.

## ZG-P09 — Native TaskCoordinator, worker and serving protocols

**Serving precision:** Three independent admission valves—bounded API/tenant request count+payload, queued prefill work+active model/session residency, and physical per-step active sequences/token+media rows/scratch—with typed overload and fair cancellation. Requests do not preallocate full KV while waiting. Source-valid decode/duplex work and aged chunked prefill share real event-backed Prepare/Dispatch/Retire, avoiding per-token router/catalog/JSON overhead. SERVOPT-01..11, 24..27, 32.


**Goal:** production-minded service skeleton, one logical session owner, event-driven heterogeneous device workers and stage-aware TaskProgram execution without forced LM-specific internals.

**Deliverables:** `TaskRequest/TaskResult`, typed `EngineMsg/EngineEvent`, `StepPlan/WorkerDelta`, persistent batch state, bounded channels, streaming/cancellation, structured model/tool/grammar output, local-only safe API/SDK, metrics and mode receipts.

**Gate:** canonical API/CLI share one engine path, no global per-token locks, no provisional output to clients, correct fairness/recovery under multiple sessions and simple concurrent tasks.

## ZG-P10 — Global physical memory and full-state prefix manager

**Serving precision:** Prefix reuse is the largest **source-complete frontier across every required state domain** (full KV, SWA, GDN/KDA/SSM, MLA/indexers, draft and media processors); exact artifact/tokenizer/template/numeric/tenant key, COW, real GPU reader leases and measured RAM/SSD restore-versus-recompute. Prefix reuse must never claim success from KV-only bytes when another required domain is absent. SERVOPT-12..18, 26.


**Goal:** unify memory reservation across all device/host/IO resources while allowing distinct policy per model weights, sparse experts, KV/state, prefix snapshots, adapters, encoders and scratch.

**Deliverables:** physical RAM/cgroup/UMA/addressability budget+OS margin, paged state pool, optional contiguous low-concurrency pool, immutable prefix sharing, 3+ StateDomain intersection/resume alignment, host/disk snapshot integrity and strict private cache identity. Per-request admission and safe cache evictions.

**Gate:** no OS-host OOM in controlled quota tests, state exact after snapshot/fork/replay, no double free/in-flight slot overwrite, graceful admission refusal. Huge virtual GPU address space cannot be counted as spare physical memory.

## ZG-P11 — Hybrid/stateful attention and complex operators

**Goal:** expand semantic reference and optimized device implementations for recurrent attention, latent/sparse/compressed context and multimodal position tracking.

**Deliverables:** GDN/KDA-style exact recurrence variants, QSA/MLA/indexer-like sparse selection, learned compressed history, hyperconnections/residual variants as required by selected model fixtures; chunk prefill vs step decode, fixed state layouts and checkpoint boundaries.

**Gate:** official full layer/logit/quality parity at representative deep contexts, safe acceptance/rewind/recompute, strict sparse IDs/position and fallback at legal layout boundary.

## ZG-P12 — Routed sparse parameters and tiered expert execution

**Serving precision:** Compile source/quant-compatible grouped decode/prefill/verify rows; account routed unique experts, padded physical row tiles, actual encoded GPU slot pins and scratch/transfer budget. Multiple models may interleave at one worker but incompatible models cannot be fused into one invalid GEMM. SERVOPT-21..26.


**Goal:** mathematically faithful routed experts across differing residency and compute tiers, no GPU↔host sync on resident hot path.

**Deliverables:** exact router scoring/top-k/shared expert; GPU compact assignment/grouping, low-row and large prefill kernels, leased resident expert slots/generation/prefetch, host bounded miss ring, IO staging and optional CPU execution lane only if measured favorable. Separate disk lookup-row service for massive immutable table.

**Gate:** synthetic randomized route sequences, cold/hot/SSD corruption/cancel tests, real MoE source reference parity; no expert ID changes for performance absent explicit changed-model policy.

## ZG-P13 — First extreme hybrid sparse model specialization

**Goal:** select a complex hybrid/recurrent/sparse/MoE model checkpoint and reach a fully correct specialized hardware plan while preserving portable fallback.

**Deliverables:** model-specific operator graph and source data, fused region candidates, GPU-resident sparse index/route, long-context stable attention, optional trained multi-token predictor, bounded lookup row IO and first measured critical-path profile. Optional GPU/NPU split-projection rows only when qualified.

**Gate:** complete model validation and matched source checkpoint/quant/context/power performance against source evidence, plus a forced generic region fallback and intact state transaction.

## ZG-P14 — Second, substantially different heavy model specialization

**Goal:** prove architecture reuse on a model with different recurrence, indexed sparse attention, hyperconnections, expert number and quant geometry—without rewriting first model's server.

**Deliverables:** new model recipe/operator implementations/reference; reuse same resource arbiter, state/TaskProgram and kernel registry; adaptive resident/streamed experts as physical fit allows and optional heterogeneous compute owners.

**Gate:** correct complete model+state and controlled resource/SSD/P95, including smaller vs larger RAM device tiers; documented delta code footprint demonstrates composability.

## ZG-P15 — Target-authoritative speculation and model/task transactions

**Serving precision:** Versioned MTP/DFlash/lookup/small-model DraftSources are provisional; only target verifies source-correct accepted prefix with exact sampled correction, grammar/stop state and full rollback. Benchmark proposed-vs-accepted tokens, state overhead, expert-cache churn and low-acceptance disable. SERVOPT-19..21.


**Goal:** support optional trained predictors, draft models, prompt/token lookup proposals and sampled distribution-preserving verification without state leakage.

**Deliverables:** `DraftSource`, exact target verifier/correction, state frontier snapshots/CommitPlan mechanism, arbitrary accepted prefix, per-task RNG/grammar, source artifact compatibility and dynamic draft policy backoff. Task loops outside LM use their own verifier semantics, not automatically borrowed.

**Gate:** exact target-distribution correctness where claimed; all-state rollback, EOS/tool boundaries, cancellation, moderate/long context and no hidden token stream drafts; fail/disable when net slower.

## ZG-P16 — Continuous batching and cross-resource cost optimization

**Serving precision:** Choose from finite prevalidated Pareto StepPlans jointly spanning active rows, legal prefill chunk, draft depth, MoE expert grouping, GPU/RAM/SSD residency, backend kernel, placement and soft deadlines. Optimize **committed SLO-goodput with all offered/rejected requests counted and per-tenant fairness**, not raw draft rates or GPU utilization alone. Protect decode and live audio without starving long prefills; do not optimize per token with global locks. SERVOPT-01..11/19..27/31/34.


**Goal:** high-concurrency and multi-task serving with stage-aware capacity/fairness and no quality regression; co-optimize resident weights, partial compute owners and speculative verifier work.

**Deliverables:** token/encoder/frame/timestep work budgets and physical resource reservations; chunk prefill, native grouped MoE, prefix-reuse admission, multi-session step-deltas, P95-aware scheduling and optional pipeline/graph variants.

**Gate:** C1/2/4/8/16/32 workload charts with TTFT/P95, aggregate useful outputs, cache hit/miss, memory/energy and no starvation/session cross-talk. Disable complexity that loses versus simple policy.

## ZG-P17 — Broader model families, adapters, embedding and reranking

**Goal:** widen registered semantics and useful models without creating another runtime; add per-request model adapters and noncausal encoder/ranking tasks.

**Deliverables:** model adapters for agreed dense/hybrid families and newer variants, encoder task/pooling, cross-encoder scoring, LoRA artifact/budget/snapshot identity, source checkpoint+quant quality matrix and text/multimodal tokenizer support.

**Gate:** independently qualified model/runner/quant/backend cells, no model-name branches in generic worker/server, correct adapter cache invalidation and source numeric agreement.

## ZG-P18 — Media TaskPrograms: audio, image, video and iterative AI

**Goal:** support independently certified non-token task loops on the same semantic/device/resource spine.

**Deliverables:** audio preprocess/ASR transcription, acoustic synthesis/vocoder, image/vision encoder, iterative diffusion/flow/timestep, video temporal and media output stages, resource/latency estimates, streaming codecs and cancellation/partial output semantics.

**Gate:** one real accepted model per claimed task type with official/independent task quality vectors; no retrofitting media into text token state. Other media programs remain planned until tested.

## ZG-P19 — Heterogeneous accelerators, collectives and remote placement

**Goal:** topology-aware compute and memory placement across local GPU/NPU/CPU and optionally remote nodes with one semantic/task/runtime protocol.

**Deliverables:** validated `SplitProjectionCompute`, `AsymmetricExpertOwnership`, layer/tensor/expert parallel and prefill/decode disaggregation where feasible, cross-driver event bridge, leases and peer session epoch handshake. Device power-fabric safety and memory bandwidth/copy cost measured.

**Gate:** local cross-accelerator safe parity with reference, shared SoC safety profile, two-device/host failure tests with no partial commit and performance where link speeds justify. Optional device/remote path cannot break single-device defaults.

## ZG-P20 — Bounded compiler/autotuner and negative experiments

**Serving precision:** Persist valid+negative hardware/model/quant/shape/driver measurements and offline candidate Pareto sets; unknown shape takes a source-qualified generic fallback, not unbounded request-time JIT or autotune. Paired repeated whole-task goodput, p95 tail and physical memory measurements authorize candidate promotion. SERVOPT-23/24/33/34.


**Goal:** improve measured kernel and whole-task performance without adding unbounded compile/search or hot-path plan churn.

**Deliverables:** bench kernel/region/task/server, offline shape/graph tactic selection, ISA/occupancy and numerical quality evidence, compiler/driver/quant keyed plan cache, guarded online policy hysteresis, repeated A/B and rejected experiment registry.

**Gate:** stable repeated whole-request latency/throughput gain on at least one true target regime and no regressions or memory faults; no "future" unbenchmarked plan auto-enabled.

## ZG-P21 — Security, trust, licensing and reproducible releases

**Goal:** audited Rust ownership, model/task/device security, safe filesystem/network inputs and installable CPU/GPU packages.

**Deliverables:** per-file license/NOTICE/SBOM policy, no proprietary copying/untrusted executable model code, signed/reproducible build decisions, fuzz parsers and snapshots, permissioned external actions, API auth, private cache, NixOS/device/driver feature isolation and GPU/NPU power safety controls.

**Gate:** release security checklist, no critical UB/resource/OS-hang behavior under supported profiles, truthful feature flags and legal artifact rights; licensed prior art remains an index, not a hidden runtime dependency.

## ZG-P22 — Full performance and quality release qualification

**Competitive evidence gate:** Pin current optimized vLLM V1 and vLLM-Omni baselines and compare equivalent source model/quant/tokenizer, numerical mode, GPU, power, prompt arrival distribution, cache/cold-warm state and tasks. Report offered/admitted/rejected/canceled requests, correct committed output meeting TTFT+ITL/media deadlines, p50/p95/p99, fairness, physical memory and energy with repeated uncertainty. No claim of universally outperforming vLLM from a single kernel/model/prompt. SERVOPT-35/36.


**Goal:** benchmark the available portable, specialized and heterogeneous paths against independently qualified external baselines, with fair source/model/task comparisons.

**Deliverables:** model/checkpoint/quant/hardware/driver/context/concurrency/sampling/power matched result matrix; logits/quality metrics and state integrity, TTFT/P95/committed outputs, energy, physical RAM, cache hit/miss, startup and cold/warm IO. Publish wins/ties/losses, and explicitly incomparable or untested cells.

**Gate:** reproducible commands and raw artifacts, statistical confidence, correct source attribution/license; no advertising universal performance superiority from mismatched models.

## ZG-P23 — Composability and future-technology evolution qualification

**Goal:** prove the primary product promise: new models, devices, tasks and storage/kernels do not require rewriting a specialized inference engine.

**Deliverables:** scripted architecture extension drills: one new SemanticOp/StateDomain, one kernel provider/device, one non-token TaskProgram, one quant/storage or API adaptation, with total touched modules, recompilation cost, source approval, release compatibility and full fallback/quality tests. Version operator/task/state/plan caches and migration or invalidation.

**Gate:** demonstrated independent extension with unchanged core service/allocator/session semantics; no nested model-specific servers. A future feature that needs pervasive unrelated changes requires interface revision before release.

---

## B. Milestones (capability, not dates)

| Milestone | Closures | Product result |
|---|---|---|
| **M0 — Feasible design** | P00–P05 slices and physical kernel checks | precise math/state, kernel and CIR resource contracts proved |
| **M1 — Usable reference/runtime** | P06–P09 | one real model direct CPU/GPU execution and basic service |
| **M2 — Stateful general serving** | P10–P12 | real hybrid-state context, expert residency and multi-session operation |
| **M3 — Extreme specialization** | P13–P15 | two contrasting complex model families, safe drafts, no independent engines |
| **M4 — Concurrency and task breadth** | P16–P18 | high-load responsive service and accepted non-token model tasks |
| **M5 — Heterogeneity and performance** | P19–P22 | qualified multi-device and Pareto/matched quality/performance evidence |
| **M6 — Extension-proof platform** | P23 | add new model/device/task without server rewrite |

## C. New model / kernel / task checklists

**Model:** checkpoint+manifest/tokenizer/quant and exact config; registered mathematics and state effects; CPU reference; tensor/logit and task quality; known backend capability; forced fallback; checked source rights.

**Kernel:** target ISA, shapes/quant, arithmetic and scratch, state effects, CIR liveness, async event fences, exact/tolerant numeric proof, complete task benchmarks, executable code provenance and fallback.

**Task:** input/output typed stages, iteration/stop/frontier, model/encoder/vocoder/denoiser as relevant, resource budgets, privacy/security, streaming/cancellation, independent source quality. Never assume all tasks use language KV.

**NPU/hardware:** optional feature under exact firmware/driver/power qualification, no automatic privileged tuning, Rust codegen feasibility and per-device compiler/source review; simulated proof not substitute for physical safety; keep GPU-only.

**Memory:** OS-safe physical reserves, all in-flight buffer generations, immutable persistent source and task state, SSD short reads and stale pointers, complete abort/quarantine under GPU/IO failures.

**Serving:** canonical task requests, authorization and bounded work queue, no hidden state leakage, p95/fairness, honest capability/mode receipt, correct output commit, independent device worker/session owner.

## D. Decision and progress discipline

Track implementation, required tests, results and remaining work in the relevant
`<plan>/TODO.md`, linked from [TODO.md](TODO.md). Update a design specification
only when the design changes. P00–P23 are architecture scope, not a completion ledger.

## v4 roadmap refinement — 2026-10-12

**Do not create new milestone numbers; preserve P00–P23.** These are **implementation clarifications** from `ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md`, not new "already implemented" features. Full protocol and review gates in `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`, `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md` and `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md` (S0-V4-A…J).

| Stable phase | v4 required change in actual acceptance |
|---|---|
| **P00** | **one Rust library + one CLI binary** to start, with modules/types not dozen scaffold crates; source/license and CPU tests. Split at 2–4 crates only if GPU compiler/unsafe build boundary proves useful |
| **P01/P07** | actual GPU ISA/source language and **async-first device-native input/worker APIs**, preallocated argument slots, safe event bridge; no host read of router IDs every token in performance-qualified hot plan |
| **P02/P04** | replace independently mutating StateDomain commit with **Prepare→PublishRoot**, source model math/effects, bounded ShapeEnv; TaskDriver stub only one minimal trusted control loop |
| **P03** | source/compute quant distinction, canonical artifact/adapter hashes and normalized source semantics, not ad hoc model code/format trust |
| **P05** | SIR + physical ExecutableBundle are the **only enduring model compiler data shapes**; RIR/KIR are transient passes; CIR is typed resource/timepoint graph inside bundle; guard ragged/dynamic shapes and avoid one variant per length |
| **P06/P08** | real model reference CPU, then GPU same TaskDriver and TaskSession; fast whole-region substitution + forced portable fallback, no interpreted IR/per-op allocations in hot path |
| **P09** | logical TaskCoordinator per session/shard, no single global queue/per-token lock, WorkerReply provisional, output after root publication and output_seq/backpressure |
| **P10** | PhysicalResourceArbiter coarse reserve and per-device local suballocators; OS-pressure fail closed, full hybrid state and prefix compatible root |
| **P11–P14** | exact source attention/recurrence/MoE and state semantics; optimized closed kernels can cover entire operator/layer while retaining portable reference; no model-specific runtime fork |
| **P15** | trained/prompt draft verifier returns PreparedRoot, sampled correction and RNG rollback before logical publish, not a kernel-side commit |
| **P16** | typed task budgets and p95/fairness/age; GPU-native StepProgram and persisted batch slots; prefill/encoder/vocoder/denoise stage classes distinct |
| **P17–P18** | **small trusted TaskDriver** for encoder/audio/media, stage-specific output/frontier; no unrestricted ExternalAction or general workflow interpreter in core |
| **P19** | epoch fencing, Prepared/Ready cohorts and safe coordinator publication with lost ACK/peer failure tests; optional durable replay must be separately designed |
| **P20** | bounded offline pure-region tuning and plan variant LRU; no context-length-by-length JIT; performance measured complete model/task under quality/power |
| **P21–P22** | source-license+Rust device program review, driver/SoC safety, resource/stream protocol, matched full-task quality/p95/energy; declare unsupported matrix honestly |
| **P23** | add new SemanticOp, KernelProvider, non-token TaskDriver, quant/IO tier **and synthetic optional ParameterVersion/TrainStep seam** without changing unrelated API/allocator/model code; training itself is future |

**Design minimality gate:** if a new model family demands a second service/session/allocator, or GPU plan compilation is mandatory per emitted token, halt that phase and correct architectural ownership. If a runtime path is correct but cannot yet avoid host synchronization, label `FUNCTIONAL` rather than `PERFORMANCE_QUALIFIED`. External source performance figures remain research baselines, not acceptance facts.


## v4.1 universal region-backend implementation crosswalk — 2026-10-12

**Keep P00–P23 unchanged, no implementation claimed.** The core bootstrap remains one CPU reference + one physically tested GPU; **do not block M0/M1 on an inaccessible TPU, wafer-scale/managed LPU or proprietary graph compiler**. Architecture accommodates these by widening the **single KernelProvider seam** into `RegionExecutionProvider` (native kernel provider is one form) accepting any semantically closed and exact state-compatible region. Candidate physical kinds: native tensor kernel, captured direct graph, AOT graph program, statically scheduled on-chip/fabric dataflow program. A completely opaque managed inference API is **TaskService interoperability**, not another kernel backend.

| Phase | Specific v4.1 addition / acceptance |
|---|---|
| **P00/P01** | identify real device capabilities and kernel/source compiler vs external vendor compiler runtime policy; local coherent ARM/LPDDR, discrete GDDR/HBM and rack fabric are distinct; no extra crates before physical need |
| **P04/P05** | `ClosedSemanticRegion` includes exact operator+state/numeric/source contract; `RegionExecutionProvider` returns region-scope/kind, lifecycle, shape, StateBridge, topology capability and fallback; RIR/KIR transient, CIR in ExecutableBundle |
| **P06/P07** | DeviceBackend can `PrepareProgram/SubmitProgram/Ready/Retire` with one AOT whole-region event rather than require CUDA-like per-op launch; native GPU remains fast and source-owned |
| **P08** | end-to-end CPU/GPU real model plus mock **graph-only AOT program** and static stateless fabric candidate exercising same SIR/TaskDriver/StateRoot; no real hardware compatibility claim from mock |
| **P09/P10** | coarse admission for compiled program SRAM/HBM/LPDDR/remote quotas, weight source/stream, graph/code cache and actual physical headroom; TaskCoordinator one logical root publication regardless of internal device micro-schedule |
| **P11–P15** | special model graphs must lower exact recurrence/sparse/MoE effects or partition safely, with real `StateBridge`; no opaque in-place state for drafts/forks |
| **P16** | phase-aware cost considers compile/load/warmup, graph region size, transfer and link critical path, shape recompile risk, whole-task quality and p95 |
| **P17/P18** | graph compiler may execute entire encoder/denoise/ASR/TTS task stage; no new model/task engine and no forced token-loop host GEMM simulation |
| **P19** | network/fabric compiler owner, actual collective topology and rank epoch fencing; no one-pointer unified rack assumption, opaque remote model APIs remain separate TaskServiceAdapter |
| **P20** | bounded graph/program cache keyed by compiler/device firmware/model/quant/shape/state schema and numeric mode; no shape explosion or per-token AOT recompilation |
| **P21** | user Rust-owned device code policy, vendor compiler/driver/runtime/IP/remote-data privacy separate; `ARCHITECTURE_COMPATIBLE` never presented as native executable `SUPPORTED` |
| **P22** | actual GB10, native Blackwell discrete/HBM, TPU AOT, wafer-static, cloud ASIC performances only where physical source/toolchain and matching checkpoint/quant/quality exist; unmeasured cells marked blocked |
| **P23** | add fake graph-only TPU-like and static-dataflow program with **same model/TaskDriver/state/allocator/server**, and independently choose another Arm64 coherent/native GPU kernel provider |

**New optional stage:** `S0-FABRIC-A…H`. Existing 24 phases, Stage0/NPU/V4 tests remain intact. New **H44…H55** research experiments cover actual hardware families and compositional failure cases; none are run. Precise normative [region-backend contract](../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md); source-specific research [platform evidence](../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md).

## v4.2 distributed execution and tunneled-machine phase crosswalk (U-020)

**Keep ZG-P00…P23 unchanged; all NOT_STARTED.** Distributed execution is a *native option within the single TaskCoordinator/RegionExecutionProvider/CIR/ResourceArbiter spine*, not an independent cluster inference server. The full topology and remote protocol are [placement](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md), [peer/state handoff](../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md), and [updated distributed entrypoint](../docs/DISTRIBUTED_PARALLELISM.md). No node/tunnel installation or network changes are authorized by this design.

| Phase | Additional distributed responsibility |
|---|---|
| **P00–P05** | keep zero-network standalone and single SIR; define minimal compute/weight/mutable-state/cache placement view and state PartitionAlgebra/ShapeGuard without building a cluster |
| **P06–P08** | real CPU/GPU single-node vertical slice remains primary gate; mock multi-node closed region/layer with state co-location and prepared root before implementation |
| **P07/P09** | optional authenticated PeerAgent/TransportProvider using same compiled region worker and TaskCoordinator semantics, tiny checked wire protocol, explicit peer authorization; no arbitrary shell or public server |
| **P10** | remote StateShardOwner with lease/placement generation, immutable remote prefix/cache directory hints, state codec and source/adapter/model schema; coarse physical reservations per node |
| **P11–P15** | stateful attention/SSM/indexer correctness across layer/local remote, full hybrid P/D handoff and speculation target fork/replay, per-attention PartitionAlgebra for token-range shards |
| **P16** | costed placement/admission includes direct vs relayed RTT/P95/jitter, activation/KV/collective bytes, prefill/decode/encode/task shape, privacy, failure and egress; measured selection, no per-token placement optimizer |
| **P17/P18** | reuse same TaskDriver/RegionExecutionProvider on remote media/encoder/vocoder/denoiser stages, task-specific output frontier and trust; no model/task engine duplication |
| **P19** | **major distributed delivery phase**: (1) 2-node authorized layer-group+local KV split, (2) remote immutable prefix full-state hydration, (3) P/D, (4) attention+state vs FFN/experts, (5) sequence KV shards on verified operator, (6) optional tensor/head/collectives/rack, (7) safe topology reconfiguration at committed frontier; no quorum DB initially |
| **P20** | bounded measured placement search, transport link profiler, costed data transfer/compute overlap, mixed-task p95 and hysteresis, negative experiments on relayed tunnel/remote KV |
| **P21** | node identity/enrollment, tenant/model/weight permission, encrypted QUIC/TCP or approved overlay tunnel; proper physical RDMA/registered memory keys, transport framing and no automatic code/OS privilege |
| **P22** | test actual C1/C8 and long context capacity vs speed for LAN/direct overlay/relayed with source model/quant/quality, physical network link and coordinator fail/recovery |
| **P23** | add one new hardware peer, new TransportProvider and remote attention/state placement **without edits to ModelDefinition, TaskDriver, TaskCoordinator or API**; standalone startup still zero cluster cost |

**Proof gates:** new optional **S0-DIST-A…L** plus experiments **H56…H70**. CPU/GPU source reference and local fast model path cannot wait for remote access. Source prior art is in `../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md`, not an architectural dependency. User's tunneling requirement is capability support, not permission to touch networking credentials or machines.

## v4.3 integrated ModelCatalog, ModelRouter, ResidencyManager and API/UX crosswalk (U-021)

**Do not create another phase or engine**. Preserve ZG-P00…P23 and prior distributed/accelerator stage programs. Multi-model routing is a small **request/model-admission** layer using the same physical ResourceArbiter, compiled SemanticProgram/ExecutableBundle, TaskCoordinator and DeviceWorkers. The `run` CLI is simply another frontend to canonical TaskRequest, not an HTTP/gateway hop or an independent model-specific server.

| Stable phase | v4.3 model management and user-facing acceptance |
|---|---|
| **P00/P03** | clean CPU-only Rust CLI skeleton, content-addressed source/quant/tokenizer manifest and local model identity/alias registry, no arbitrary remote download; versioned safe config and source rights |
| **P04/P05** | TaskRequest validated source/TaskDriver/model profile and numeric contracts, no router logic/model-name branching in semantic IR or kernels |
| **P06–P08** | **one-command `run ./model.gguf` local in-process** with no config/daemon, actual reference/GPU model task and one committed output stream; loading progress/refusal meaningful |
| **P09** | single `serve` loopback native engine+router/API adapter, exact alias→artifact binding, GET models/chat SSE/usage/finish, authorized native model admin, Python+JS SDK conformance; no gateway process |
| **P10** | actual ResidentModel/Replica lifecycle, one global physical reserve, safe load/warm/keepalive/pin/unload, deduplicated concurrent model loads and full-session/graph/adapter/draft memory; no GTT overcommit or active KV eviction |
| **P11–P15** | multiple models with exact operators/quant/StateRoot, source-specific model profiles, trained draft+target as separate model roles, no incorrect load-balanced target model substitutions |
| **P16** | same-source replica balancing, sticky KV/prefix locality, per-model and per-tenant fairness/deadlines/queue, mixed resident model workloads and cold-load thrash prevention; router never on per-token hot path |
| **P17/P18** | embedding/rerank/ASR/TTS/image TaskDrivers and separate model roles available via truthful capability/endpoint subset; stateless Responses then qualified stateful retention, per-model adapters/config |
| **P19** | one distributed model deployment vs multiple replicas across explicitly authorized nodes; pooled/placement resource budget and StateRoot-compatible migration; remote model/source rights |
| **P20** | measured model-cold-load/warm/unload/caching and route cost, cache affinity, replica count/keepalive hysteresis and graph kernel plans, no quality or P95 regression |
| **P21** | local loopback default, external bind auth/TLS/admin separation, model catalog privacy, source pull consent, API/schema/stream security and source licensing |
| **P22** | real same-model direct `run` vs router `serve`, multi-resident/replica P95 throughput/TTFT/memory, SDK chat/Responses/embeddings conformance and source numeric fidelity |
| **P23** | add a new model profile, new exact model family and replica/router policy with **no edits to TaskCoordinator/device executor/SIR**, demonstrate minimal change radius and inspectable effective config |

**New optional Stage0 S0-SERVE-A…J and experiments H71…H86**: staged from one real model/CLI/API to multi-model physical loading and multi-model concurrent serving. Full specs [router](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) and [CLI/API](../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md). **All are design only.** User-visible command/port/profile syntax remains proposed, not a guarantee it is compiled.


````

## Source: .agents/plans/INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md

Original UTF-8 bytes: 28363; SHA256 `9b11a2bcace9cfcea129dc4a94be051d3f24d516d303d2dd23584ca1e77a3f4b`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — Independent Stage-0 feasibility and delivery gates (v4 reviewed)

**2026-10-12 · NOT STARTED.** This is the project-neutral execution plan for architecture feasibility. It retains stable IDs **S0-00…S0-14** and optional NPU IDs **S0-NPU-A…F**; all remain PLANNED/NOT_STARTED. Normative math/task/kernel/resource contracts are in `../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md` and `../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md`.

## 1. Hard problem and mitigation

Two dangerous extremes are (a) a collection of fixed per-model GPU kernels plus a separate model-specific server that cannot grow to new algorithms, and (b) a large general IR/optimizer/framework that has never executed a real model efficiently. Our proof process deliberately combines **physical kernel feasibility**, **independent CPU reference and state truth**, a *minimal effect-aware graph compiler*, a *small async lease/event executor* and one **complete model+TaskProgram→output vertical slice**.

Do not freeze dozens of crate APIs or an exclusive kernel compiler before demonstrating the vertical slice. Do not defer kernel codegen until an enormous generic framework exists. Optional experimental accelerators may fail without preventing portable CPU/GPU success.

## 2. Stage-0 work packages

| ID | Work | Smallest concrete artifact | Independent pass gate |
|---|---|---|---|
| **S0-00** | Reproducible Rust baseline | lean Cargo CPU build, pinned dev shell/toolchain, source/provenance/license ledger, hardware probe | clean CPU tests, no unreviewed dependencies, correct device/memory inventory |
| **S0-01** | Physical native GPU kernel viability | Rust-authored CPU oracle and simple low-bit GEMV, prefill GEMM/WMMA and state update on target GPU | physical result, generated ISA, numeric/quality proof and measured gap to reference |
| **S0-02** | Second accelerator/codegen path | same shape/reference through different device/compiler kernel provider | compile+physical parity or honest missing/blocked capability, no model recipe edits |
| **S0-03** | Model artifact and source truth | GGUF/safetensors checked reader, tokenizer and quant codec, small real checkpoint | official/reference token, tensor and logits agreement; parser fuzz rejects bad files |
| **S0-04** | Small typed SIR with mutable effects | pure SSA + KV/one recurrent state operator, verify/Fork/Commit/Abort | illegal effect reorder rejected; canonical state output passes |
| **S0-05** | Minimal RIR and KIR registry | closed region fuse or row split, kernel candidate fallback, layout conversion | same math/state with optimized and generic variants, accurate error |
| **S0-06** | Async CIR and leases | fake CPU/GPU/IO executor with Acquire/Dispatch/Copy/Wait/Commit and timed events | invalid alias, stale generation, premature free, lost event/short IO fail closed |
| **S0-07** | Coordinator/worker protocol and state pools | EngineMsg/Event, StepPlan/WorkerDelta, paged KV and simulation | block conservation, no state cross-talk or global hot mutex, admission budget correct |
| **S0-08** | Model/task quality and transactional recovery | teacher-forced full logits, draft accept/abort, RNG and state replay reference | exact/tolerant modes differentiated; rejected speculative tail leaves no effects |
| **S0-09** | Rendering, grammar, service contracts | exact token/template/structured output and mock TaskRequest, streaming, cancellation | independent role/token vectors; no unverified provisional emission |
| **S0-10** | Immutable ExecutableBundle and plan cache | versioned direct kernel handles, stable graph/eager modes and capture proof | no stale captured address, no per-token SIR interpreter, valid forced fallback |
| **S0-11** | Sparse parameter placement simulator | layer/route/skew/hit/miss trace, optional cache/prefetch candidates and global headroom | no unsafe overcommit/eviction; measured cost rather than guessed optimal |
| **S0-12** | **Real end-to-end model vertical slice** | one checkpoint→ModelDefinition→SIR→RIR/KIR/CIR→CPU+GPU worker→committed streamed result | real model numeric/reference/quality, full state, CLI or SDK, forced missing fast kernel |
| **S0-13** | Multi-domain prefix and task composition | three different state domains, synthetic non-token TaskProgram stage loop and stage-specific output | exact common resume frontier, safe branches, no forced text token loop |
| **S0-14** | Optional additional portability and draft | alternative backend and optional zero-weight draft lookup/fallback | correct output/model state on supported backend; measured benefit or disable |

**Hard gates before freezing architecture:** S0-00, S0-01 (or documented physical specialization blocker), S0-03/04/05/06/07/08/12. **Composition should also be proved by S0-13** before claiming the system is generally modular. S0-02/09/10/11/14 may be staged when relevant, with all skips clearly recorded. The numbers are IDs, **not guaranteed calendar or engineer estimates**.

## 3. Optional Stage N — shared-SoC GPU+NPU and partial-operator execution

A specific coaccelerator may have its own device program format, memory import rules and **unresolved hardware/firmware power safety**. The NPU track never blocks GPU-only runtime.

| ID | Work | Pass / blocked condition |
|---|---|---|
| **S0-NPU-A** | Model-independent split-projection math proof and fake dual-device CIR | disjoint row spans, complete output order, quant/bias, first-use barrier, correct generic fallback |
| **S0-NPU-B** | Rust-owned NPU/AIE program/compiler and safe driver FFI | source-language and license policy satisfied plus compiled/validated shape/encoding; otherwise BLOCKED_BY_POLICY |
| **S0-NPU-C** | Physical SoC concurrency safety | dedicated isolated hardware and exact firmware/driver/fabric/clock evidence, normal boot/suspend/resume and long soak; hard hang blocks automatic enablement |
| **S0-NPU-D** | Whole-model/quality proof | full official model/logits/state, CPU/GPU-only vs GPU+NPU split and compute-precision error, exact sampling where claimed |
| **S0-NPU-E** | Matched end-to-end benchmarks | same model/checkpoint/quant and held physical clocks ON/OFF ABBA; TTFT/P95/memory/energy/idle/power, beneficial critical-path overlap |
| **S0-NPU-F** | New model/task/device extension | add second eligible projection operator and distinct encoder/task stage without touching model service and resource runtime; unsupported/unsafe defaults GPU-only |

An upstream-author prefill result is a **test target**, not proof. **No automatic privileged fabric clock, BIOS/IOMMU, NPU power-mode, runtime suspend or system service modifications.** A whole-machine hang is not recoverable by a software timer and the NPU path stays disabled without actual platform qualification. It is permissible and correct to ship GPU-only if the Rust source compiler or hardware safety gate cannot be passed.

## 4. Gates by architectural responsibility

### G0-A — Semantic and model foundations
- A real checkpoint's tensor/quant/tokenizer/config/template is parsed safely.
- A reference operator and independent model logit/continuation fixture prove correctness; novel op can't be silently guessed.
- SIR carries exact value shape, model position, quant/computation dtype and mutation effects; illegal reorders rejected.
- TaskProgram emits one correct output and supports a different synthetic bounded task stage with independent frontier.
- Unknown format/operator and malicious file rejected clearly without OOM or arbitrary code execution.

### G0-B — Kernel codegen and resource execution
- At least one physical GPU backend executes correct Rust-owned kernel(s); generated ISA/occupancy and build provenance stored.
- RIR produces one legal variant and fully correct fallback region from identical semantic model.
- CIR manages lease generations, async events/IO, alias and failure before committing state; graphical memory safety simulated and hardware smoke.
- Compiled ExecutableBundle is direct call/graph, no token-hot SIR/RIR/CIR interpretation or unnecessary memory copy.
- NPU/peer optional backend cannot change reference model, API or default GPU runtime when absent.

### G0-C — Functional end-to-end vertical slice
- Real model file/quant and tokenizer→typed ModelDefinition and TaskProgram→legal regions→portable CPU plus one validated GPU→user result, with exact state and fixed plan.
- Forced disabling a fast kernel does not cause another serving engine or broken session state; performance tradeoff recorded.
- TaskCoordinator owns request/commit frontier and DeviceWorker owns device events; no premature output.
- Physical resource headroom accounts OS/cgroup, GPU virtual address vs real RAM, scratch and in-flight buffers.

### G0-D — Modularity proof
- Add second model/novel op by extending reference op and model config only, not rewrite runtime service or allocator.
- Add second hypothetical accelerator provider with identity/shape/capability and fallback without modifying SIR model math.
- Add non-token task (e.g. encode→pool or mocked denoise loop) through same resource/worker/TaskProgram infrastructure without a decoder-only workaround.
- Code churn and test coverage recorded; new model-specific duplicated server is a **FAIL**.

## 5. Later release gates (capabilities, not promised dates)

**G1 Portable useful:** model support matrix, CPU+one GPU, safe local service/tokenizer/tool/grammar, complete output, source math validated and clear fallback/refusal. **G2 Production scheduler/state:** physical resource arbiter, multi-session token/other stage budgets, 3+ state-group prefixes, no use-after-free, P95 under load. **G3 Specialized/heterogeneous:** complex recurrent/sparse/MoE, expert tiers, optional NPU/two-GPU work, source quality and safety gates. **G4 Broader AI tasks/distribution:** embeddings/rerank, ASR/TTS/image/video and other TaskPrograms per model, topology/transport fail-closed, long-run reliability/security and independent benchmarks. Each model/task/hardware cell must be individually qualified.

## 6. Evidence and kill/pivot criteria

Per experiment/phase record commit/source, actual machine topology (CPU/GPU/NPU/physical RAM/storage), checkpoint+quant, tokenizer/semantic schema, compiler/device program source and flags, driver/power/FCLK for shared SoC, input/output task geometry, numerical/state tests, failures, per-request p95, energy and retry/abort correctness. User-specific output privacy preserved.

**Pivot/stop:**
1. Semantic IR cannot represent a recurrent/media loop without hidden mutation → fix model/TaskProgram effects before kernels.
2. Rust GPU/NPU kernel backend cannot emit correct/competitive instructions → defer extreme plan, keep portable model functionality.
3. CIR causes per-token interpreter overhead → compile static direct commands and shrink abstraction.
4. Physical memory/lease/IO/clock unsafe → prevent serving affected profile.
5. Specialized plan changes source model route/top-k/quant/sampling without opt-in → reject variant.
6. An extension requires rewriting unrelated server/allocator/model core → fix capability boundary.
7. Optimization gives no repeated whole-task gain or worsens P95/quality → retain simpler policy.
8. Source is non-MIT and named-project-dependent, or file/license rights unverified → preserve idea as attributed prior art only, no automatic source import.

**Closure:** no phase or gate becomes VERIFIED from documents. Update status/decision/handoff/manifest only after actual code, tests and physical evidence. The prior-art archive preserves older experiments and named sources, but the canonical acceptance criteria are these independently expressed gates.

## v4 correction and dedicated proof gates — 2026-10-12

**Authority:** [Critical review](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md), [execution/state](../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md), [compiler/task](../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md). Stable S0-00…S0-14 and S0-NPU-A…F remain PLANNED; **S0-V4-A…J** are *review closure tests*, not extra compulsory crates or a new entire implementation phase. Fold their smallest proofs into P04–P09 during the first vertical slice; later tests are deferred appropriately.

| Review gate | Why | Minimum convincing proof |
|---|---|---|
| **S0-V4-A — single root publication** | Fix per-domain state commit leak | fake KV+GDN+RNG all Prepare then **one** PublishRoot, failure in second domain leaves old root fully readable, no client output |
| **S0-V4-B — device-native hot loop** | Avoid mandatory host sync | on supported GPU run two causally dependent provisional target steps/rows with device routing/sampler index buffers, same reference state, no forced sampled token readback; record host wait cost |
| **S0-V4-C — hierarchical allocator** | Remove global per-token allocator lock | global reserve→local device subpool→generation lease→event retire, fault/stale generation tests; O(1) frequent allocations, no central mutex per op |
| **S0-V4-D — symbolic shape guards** | Prevent invalid kernels/JIT explosion | rank fixed with bounded dynamic dims, ragged offsets and sizes 1,7,128,4096; eligible kernel or deterministic safe fallback without out-of-bounds |
| **S0-V4-E — non-token TaskDriver** | Prevent embedded workflow DSL | trusted built-in encoder/pool or mock denoise loop through existing TaskCoordinator/DeviceWorker, output frontier not token |
| **S0-V4-F — specialization containment** | Prove no parallel engine | closed whole-layer region kernel can be enabled/disabled with identical model/task/server/state/physical budget ownership |
| **S0-V4-G — output delivery semantics** | Avoid promise of exactly-once network | output sequence IDs after root publish, slow consumer credits, disconnect/resume optional replay/dedup test, no provisional leak |
| **S0-V4-H — distributed epoch fencing** | Fail closed after partial rank prepare | fake remote peer timeout/lost ACK/stale epoch; coordinator does not publish new root until correctly prepared and fenced; no false durability guarantee |
| **S0-V4-I — optional training upgrade seam** | Broad AI execution without core training bloat | synthetic ParameterVersion+GradientProgram/optimizer effect can be represented in SIR/TaskDriver without mutating pinned inference sessions; real training deferred |
| **S0-V4-J — compilation/cache minimality** | Avoid four durable IR formats and cold compile storms | SIR+ExecutableBundle authoritative, transient region/kernel views, no raw driver pointers persisted, cache LRU/driver invalidation and generic shape fallback |

**Architecture freeze minimum:** V4-A/C/D/F/J plus core S0-00/03/04/05/06/07/12 should accompany the first real model. V4-B/E should be demonstrated before announcing high-performance asynchronous and cross-task composition as fully proven. V4-G/H/I can be implemented at service, distribution and future-learning phases respectively, without holding up the first GPU reference engine. **All are NOT_STARTED**.

**Known constraints:** GPU memory pressure cannot be perfectly predicted on a multiprocess OS; an OS OOM due to unrelated work is not proof that strict physical safety was guaranteed. Device loss may leave DMA live; quarantine, don't recycle. Cross-rank logical root publication and network output ACK are *not* automatic durable distributed transactions.

**Stable read order:** the old subphase gate text remains valuable background for quantitative tests, but where it says independent state domain `commit` or forces every TaskProgram into a huge DSL, these v4 gates and their owner contracts supersede it.

## v4.1 optional Stage Fabric — graph-only, wafer/static dataflow and heterogeneous cluster execution

**Not a prerequisite for the portable CPU/GPU S0 vertical slice.** Existing S0-00…S0-14, S0-NPU-A…F and S0-V4-A…J stay unchanged. The new **S0-FABRIC-A…H** are design-proof/future-hardware gates for one **RegionExecutionProvider** capable of different *minimum efficient region scales*, from native single kernel to static compiled wafer-scale program. All are NOT_STARTED.

| Gate | Smallest falsifiable proof | Pass vs blocked |
|---|---|---|
| **S0-FABRIC-A — graph-only provider** | fake target that accepts only closed whole-model stages, returns one AOT compiled executable and event; same source SemanticProgram/TaskDriver/StateRoot | correctness and no new server; actual TPU/AOT qualification remains blocked without vendor compiler/hardware |
| **S0-FABRIC-B — static dataflow fabric** | simulator for multi-core statically scheduled SRAM pipeline + streamed immutable model weights; compile/load cost, one completion event | tensor/reference parity, real weight/memory critical path, no host issue-per-op; actual wafer performance unverified |
| **S0-FABRIC-C — safe state bridge** | fake graph provider with private provisional KV/recurrence/adapter state, Prepare+PublishRoot, fork/snapshot/recompute; also opaque in-place state without rollback | first passes, second **must be rejected** for speculative stateful reuse (stateless encoder allowed) |
| **S0-FABRIC-D — shape/program lifecycle** | static and bounded dynamic shape guards, compilation/cache/version fingerprints, cold source fallback and no token-by-token graph recompilation | cache bounded, source semantics/quant correct, no stale program handles |
| **S0-FABRIC-E — topology/fabric** | mock 2×GPU / compiler-managed interchip collective / programmed fabric with link cost and fault/lost ACK | correct placement/ready/rank fencing, cannot publish mismatched root or assume aggregate memory pointer |
| **S0-FABRIC-F — coherent vs HBM and low precision** | device capability/quant mock for coherent local ARM+GPU, discrete HBM and rack topology; distinct NVFP4/MXFP4/other block layout | math/codec/shape and physical resident/OS budget correct; real device benchmarks separate |
| **S0-FABRIC-G — opaque service boundary** | a fake text-only remote model API with no tensor/state bridge and untrusted checkpoint provenance | cannot masquerade as native RegionExecutionProvider, only explicit permissioned TaskServiceAdapter with limited guarantees |
| **S0-FABRIC-H — extension and Rust/toolchain policy** | attach two new provider kinds and program liveness variants while existing ModelDefinition/TaskDriver/TaskCoordinator/ResourceArbiter remain unchanged | no new server, vendor compiler authorization/license evidenced; missing compiler reported unsupported not qualified |

**Release/maturity ladder:** `ARCHITECTURE_COMPATIBLE → SDK_FEASIBLE → COMPILES → FUNCTIONAL → STATE_SAFE → PERFORMANCE_QUALIFIED → PRODUCTION_QUALIFIED`, each tied to a specific region/model/task/quant/driver/shape/topology. Mock backend qualifies **architecture only**, never a real TPU/wafer or Blackwell program. Native GB10/B200 GPU paths can be real P01/P07 device qualification if hardware available; graph/wafer/Trainium style requires authorized vendor compiler plus state/quant/shape tests.

**Main gating rule:** stateful full-model program must provide privately prepared versions or safe snapshot/recompute. A compiler that only exposes opaque in-place state, or an API that returns text, is not an interchangeable region backend. No source-unknown kernel copied, no unapproved foreign-language compiler/FFI, no new runtime servers. See `../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md` and separately sourced `../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md`.

**Phase mapping:** P01/P05 for RegionExecutionProvider + lowerer, P07 DeviceBackend program lifecycle, P08 POC graph fake, P10 memory/resident program, P16 cost+shape, P17–18 compiled task stages, P19 topology/collective/fabric, P20 compiler program cache/tuning, P21 vendor/trust/privacy, P22 physical performance, P23 new-accelerator extension drill. Existing task/runtime ownership remains unchanged.

## v4.2 optional Stage Distributed — topology, remote state and tunnel execution

**S0-DIST-A…L are NOT_STARTED architecture/transport proof gates, not additional mandatory crates or blockers for the first CPU/GPU end-to-end model.** They complement original S0-00…14, optional NPU A…F, v4 A…J and Fabric A…H. The authoritative rules are [distributed placement](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) and [remote peer/state handoff](../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md). Names from current research remain [prior-art only](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md).

| Gate | Objective proof | Honest limit |
|---|---|---|
| **S0-DIST-A — topology and resources** | fake 3 heterogeneous nodes, measured-link object with directional bandwidth/RTT/P95/direct-vs-relay, separate compute/parameters/mutable state/immutable cache, no new IR | simulator doesn't qualify physical tunnel |
| **S0-DIST-B — layer state colocation** | two layer groups on different fake workers, each owning correct per-layer KV/recurrent state, stage activation only; C1 serial cost vs prefill pipeline | no universal multi-node speedup |
| **S0-DIST-C — independent attention/FFN and cache** | remote attention executes on KV owner, FFN elsewhere; compare one-time cache hydration and naive remote KV reads with modeled bandwidth and state correctness | slow tunnel may reject plan |
| **S0-DIST-D — remote prefix and full hybrid state** | immutable cache hint verified by source/quant/position/tenant and all state domains, transfer/recompute choice and old snapshot root correctness | directory hint isn't mutable owner |
| **S0-DIST-E — P/D handoff** | prefill worker complete KV/GDN/indexer/grammar/adapter state to decode worker after resource reserve; failure rollback and TTFT link cost | ordinary KV-only handoff insufficient |
| **S0-DIST-F — sequence KV partition algebra** | ordinary masked/causal attention shard max/Z/weighted numerator merge, all-masked/numerical edge tests; selected sparse and GDN refuse without op-specific proof | no universal recurrence split |
| **S0-DIST-G — trusted peer enrollment** | no unauthorized peer executes/receives models/prompts/weights, no remote shell, scoped ID/tenant/model/lease; program integrity/rights checked | auto-discovery ≠ consent |
| **S0-DIST-H — transport paths** | fake direct LAN vs encrypted tunnel direct/peer-relay/relay RTT, bandwidth/crypto/packet loss; rejects impossible GPUDirect over overlay | physical tests later |
| **S0-DIST-I — remote coordinator fencing** | duplicate/out-of-order Ready, wrong checkpoint/epoch, old coordinator after partition; exactly one writer/root output or fail-stop | no durable leader election assumed |
| **S0-DIST-J — state shard migration** | reserve/hydrate complete compatible state → fence old owner → one root placement epoch change, or old root preserved on failure; live DMA quarantined | not arbitrary mid-GPU step migration |
| **S0-DIST-K — mixed task stage split** | encoder/ASR/image stage plus text task across fake peers with task-specific output frontier, correct stream/privacy, no separate server | real codecs/model quality separately |
| **S0-DIST-L — real two trusted machines** | physically run one reference-checked model over enrolled direct authenticated link, then explicitly tested private tunnel; record model/quant/state, P95, memory and failure | BLOCKED until user authorizes devices/network and code |

**Initial distributed implementation order:** simulator A/B/I/G, optional one real D-LAN layer split L, then D/E remote-prefix+P/D, then C/F remote attention/sequence shards, and multi-peer topology search only after measured need. A 2-node private cluster is sufficient proof; no mandatory DHT, container orchestrator, cluster-wide consensus DB or generic workflow engine.

**Safety:** do not connect or enroll user's machines without explicit authorization. Tunnel doesn't confer user data permission or hardware RDMA. Peer cache replicas are immutable location hints; mutable StateShardOwner carries lease+epoch. Any network partition during a provisional commit must fail closed or reconstruct from safe committed history. Simulated success is **ARCHITECTURE_SIMULATED**, not physical state/performance qualification.

## v4.3 optional Stage Serve — minimal CLI, model router and API conformance

**S0-SERVE-A…J are NOT_STARTED proof gates.** They add no new implementation phase IDs and do not block the original single-model CPU/GPU vertical slice. They validate that built-in multi-model serving can be added through existing TaskCoordinator/ResourceArbiter/RegionExecutionProvider with *no extra inference engine*. Source-attributed comparisons in `../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md`; normative [router](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [CLI/config/API](../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).

| Gate | Minimum convincing test | Unqualified/incorrect if |
|---|---|---|
| **S0-SERVE-A — zero-config local `run`** | one real supported local checkpoint, no config/server/cloud/account, trusted artifact/quant and committed terminal output | mandatory daemon, arbitrary source download or a second engine |
| **S0-SERVE-B — catalog and source binding** | two exact immutable model source hashes, alias/model-profile revisions, tensor/adapter/tokenizer/task validation | alias update changes a live StateRoot |
| **S0-SERVE-C — safe on-demand load** | fake model load with actual reservation/ready lifecycle, 100 concurrent same-model requests deduplicated, bounded waiter queue and timeouts | 100 duplicate GPU allocations or infinite waits |
| **S0-SERVE-D — multiple physical residents** | 2–4 fake models/real small compatible models share one physical arbiter; context/SSM scratch counted, idle unload/event retire safe | virtual GTT counted as RAM or unsnapshotable active session evicted |
| **S0-SERVE-E — equivalent replica routing** | 2 same-artifact replicas, prefix-affine sticky session, queue/fairness, resource/deadline and safe full StateRoot migration | request routed to different source quant/model or stale state |
| **S0-SERVE-F — per-model and request config** | global, model, alias, CLI, authorized request and hard policies resolved deterministically; effective provenance and conflicts shown; invalid keys rejected | request overrides security/quant/resource or silent ignored setting |
| **S0-SERVE-G — OpenAI SDK chat/model** | one loopback `GET /v1/models`, chat nonstream and SSE/usage/finish/tool data via pinned current Python+JS SDK; exact model identity and semantics | wrong stream schema or fake compatibility |
| **S0-SERVE-H — task/API subsets** | real supported embeddings model/pooling and stateless Responses events as separate conformance cells when implemented; stateful store/tool cloud fields explicitly refuse until qualified | claim endpoints work with no qualified TaskDriver |
| **S0-SERVE-I — local vs external security** | loopback default, no automatic external bind, admin model management restricted, public auth/TLS required and scoped model visibility | unauthenticated remote admin or arbitrary model/media/tool code |
| **S0-SERVE-J — distributed model/replica distinction** | one model sharded across two fake peers, two replicas of exact same source and two distinct source models; router admission and placement/state/tenant budget correct | model shard mistaken as replica, unauthorized remote transfer, second model server |

**Early core proof:** A+B+C+F+G can be integrated during P08/P09 once one model genuinely runs; D+E at concurrency/residency P10/P16, H with TaskDriver breadth P17/P18, J after distributed mock v4.2. Local `run` shall have no mandatory HTTP router/proxy startup or per-token JSON overhead. **All gates are design tests only until actual Rust model/device/API code exists.**

**Safety:** no downloaded model/source, public listener, service manager, account/tunnel setup or network permission without user instruction. Profile and model-selection changes cannot reduce original checkpoint/quant/target sampling quality implicitly.

````

## Source: .agents/plans/MODEL_ACQUISITION_IMPLEMENTATION_SPEC.md

Original UTF-8 bytes: 15314; SHA256 `73d4eb29a98eebd39166b179b8cb647eec961b849a79b7a749220a5dc3c96829`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Acquisition next unit — bounded cache first, then HF/GitHub vertical slice

Status: A1 local cache implementation VERIFIED; A2/A3 remain planned. The
[A1 acceptance](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md) records actual API, source
snapshot, tests, findings and limits. This specification retains the original
full acquisition design; provider/loader/parity gates do not pass by implication.
The [accepted checkpoint](accepted-baseline/TODO.md) settles the earlier
baseline; [plan TODO](model-acquisition/TODO.md) tracks subsequent implementation and tests.

The original design below assumed no source changes at that time. A1 is now
implemented behind the optional Linux cache feature in `zerogravity-artifacts`.
No HTTP/provider, CLI pull, eviction, trained-model addition or download is
implemented by A1. The contention-first review remains complete with no LR
migration; do not put an advisory immutable index in charge of cache ownership.

## Bounded implementation sequence

A1 (now verified) in `zerogravity-artifacts`, behind an optional `cache` feature:
local import, bounded manifest/schema/path validation, private XDG cache,
content-addressed blobs, immutable complete snapshots, atomic mutable refs,
read-only offline inspect/verify and snapshot leases. No HTTP in this unit.
Use tiny local files and child-process fixtures, not new weights. Exclude eviction
until lease/accounting gates pass; return explicit Unsupported for that operation.
This is a concrete prerequisite, not complete PULL-A or upstream parity.

Immediately follow with A2: shared bounded HTTPS + native HF and GitHub resolvers,
explicit CLI pull/dry-run/offline, complete sidecar/shard closure, auth/redirect/
resume/cancel fixtures. Add A3 for acquisition-to-existing-loader and safe removal.
Keep optional cloud/OCI/an alternative model hub/Xet rows visible as remaining work, not silently
closed by the first two providers.

Ownership proposed for three parallel implementation agents AFTER cache API freeze:
1. Artifact store owner: new `src/cache/{mod,manifest,paths,store,leases}.rs`,
   cache tests and exact optional sha2/rustix dependencies. One file publication
   protocol and disk reservation authority; no inference/runtime/catalog rewrite.
2. Transfer/HF owner: new `src/acquisition/{mod,http,huggingface,resume}.rs` and
   transport fixtures. `http` exports the narrow verified stream API shared by GH;
   sole transfer owner sets TLS/auth/redirect/timeouts, not two HTTP implementations.
3. GitHub/CLI owner: new `src/acquisition/github.rs`, `zerogravity-cli/src/pull.rs`,
   command tests and facade exports. Coordinator reconciles manifests/lock/features.
   Do not parallel-edit store/protocol types; agree those cold interfaces first.

No extra crate is needed initially: optional transport dependencies fit the
existing artifact owner and preserve CPU/minimal inspection isolation. A separate
crate needs an observed platform/dependency boundary, not an architectural label.

## Cold API, authority and loader boundary

- `CacheStore::open(CacheOptions)` holds a validated root directory capability;
  options resolve absolute XDG/explicit root, quotas, lock deadlines and limits.
- `ResolvedPullPlan` has validated private fields: provider, endpoint/profile,
  immutable revision/object locator, exact relative paths, component roles,
  required/optional status, expected lengths/hashes, source-provenance strength.
  Never deserialize untrusted JSON directly into a trusted plan without validation.
- `CacheStore::begin(plan, admission) -> CacheTransaction`; stream each object
  through bounded memory + SHA256, then `commit -> PublishedSnapshot/PullReceipt`.
  A complete artifact snapshot is not ModelReady, DeviceReady or a StateRoot.
- `open_snapshot(digest) -> SnapshotLease` owns its exact parsed manifest and
  opened object handles; refs resolve once to one immutable manifest digest.
  No raw mutable path, catalog guard, credential or provider callback becomes
  lifetime/permission authority. Source publication remains the existing owner.
- `PullRequest -> resolve -> PullPlan -> execute` is explicit acquisition. Run,
  inspect and executable-catalog miss never trigger a download. Offline branches
  before provider/client/credential construction: no metadata, DNS or auth calls.
- Keep process-local SourceIdentity issued only by SourceWeights after a verified
  owned source allocation. Manifest digest is durable content identity, not that ID.
  SafetensorsIndex is metadata only; Qwen27B/Gemma contracts are not inference.
- Runtime bridge is A3, not a cache shortcut: add a descriptor-taking Artifact
  constructor plus a SourceWeights verified-owned-copy entry using the existing
  materialize/reparse path and hash the ACTUAL copied bytes against the manifest
  digest before issuance. Keep existing4GiB GGUF limit and mathematical schedule.
  Bounded config bytes must likewise be verified before the existing parser.
  Refactor one cold preparation helper if needed; do not create a second loader.
- A blob hash filename has no `.gguf` extension, while current Artifact::open
  dispatches by extension. Preserve original logical filename or pass an explicit
  checked format hint with the opened descriptor; do not reopen an unchecked path.
- Raw-path lazy catalog registration does NOT currently retain a cache lease.
  A1 offers no automatic GC claim for registered paths. A3 direct CLI cache-load
  holds a snapshot lease through the verified owned copy. Any later lazy cache
  registration needs an explicit retained lease, not a second executable catalog.

## XDG, locking and transaction details

Root: `${XDG_CACHE_HOME:-$HOME/.cache}/abird/zerogravity/models`, ignoring relative XDG;
explicit absolute override wins. Root/attempt directories0700, records0600,
completed objects read-only by convention. Keep credentials/config outside cache.
Suggested names: `blobs/sha256/<digest>`, `manifests/<digest>.json`,
`snapshots/<digest>/`, `refs/<encoded-provider-repo-profile>/<encoded-selector>`,
`staging/<unique-attempt>/`, and stable `locks/` (not remote path strings).

Validate relative path components before allocation: no absolute/dot-dot/NUL/
backslash/alternate separator, duplicates, file-vs-directory collision or symlink
escape. Handle Unicode/case policy explicitly; Linux first, no untested Windows/
NFS guarantee. Root-relative openat/O_NOFOLLOW directory capabilities close the
check-then-open ancestor race; do not merely call canonicalize then open.
Bound JSON bytes/depth/entry counts/path bytes, per-file/aggregate u64 arithmetic
and required closure. Safetensors declared total_size is tensor payload bytes,
not the sum of shard file lengths including headers. Reuse its duplicate-key and
exact-u64 parser; validate required files separately. GGUF split filenames need
complete ordered1..N closure; acquisition still does not add split inference.

Use OS advisory file locks (Linux rustix is cached) on stable owner-private lock
inodes. Never unlink a lock inode while another process can hold/open it, which
would split the authority. Avoid holding multiple object locks; install each
verified blob independently, then serialize snapshot/ref commit. No catalog/read
guard spans disk/network/hash work. Separate long object-transfer single-flight
from short ref publication. Caller cancellation releases only its own attempt.
Use lock ownership + bounded waits, not PID-existence as crash/ABA authority.

Blob commit: stream into a same-filesystem private file; enforce bound/hash,
fsync file, install without replacing a conflicting existing object, fsync parent.
Dedup verifies existing object bytes from its opened handle, never just filename.
Build manifest and snapshot in private staging; sync files/directories, atomically
install the complete immutable snapshot, then atomically replace its small ref
and sync the ref parent. Ref readers see old complete or new complete snapshot.
Publish failure leaves the old ref exact; orphan verified blobs are not readiness.
Fault-inject every sync/link/rename and process death, then restart/validate.
Cache format/version/canonical manifest encoding must be explicit; original
config/tokenizer/license bytes are never normalized or rewritten for convenience.

Read-only files, hardlinks, mmap/MAP_PRIVATE and an open descriptor do NOT freeze
an externally writable inode. A same-UID writer can still change cache bytes.
Logical immutable naming/cooperative writer locks are not OS immutability. Verified
copied-byte digest at A3 closes the actual source-loading boundary; future mappings
need enforceable backing/leases, per the existing large-source audit. Hardlinked
snapshot aliases share blob bytes and must never be opened for writing.

Host TransferStaging uses the EXISTING process ResourceArbiter with a caller-bound
registered resource domain, bounded buffers/concurrency and explicit owner.
Do not instantiate a new arbiter per pull or invent per-request physical keys.
There is no current unified host-RAM or cross-process disk arbiter. Scoped transfer
reservations do not account all CPU models/UMA memory. State that limit plainly.
Disk quota/reservation records under cache-wide locking must include partials,
verified-but-unreferenced blobs, metadata and actual temporary/final peak; statvfs
is an observation, not reservation. Refuse unsupported cross-process quota scopes
rather than claiming the GPU ledger provides them. No unlink-based free-space
claim while descriptors/leases can retain bytes. Begin eviction only after that
separate physical-release/reservation protocol passes.

## A2 exact transport/provider contract

Use synchronous streaming HTTPS, not a new Tokio/runtime/background service.
Cached ureq3.4.2 +rustls is a candidate: defaults off (notably gzip), explicit
identity content encoding, TLS verification and capped redirects/timeouts.
Check complete selected TLS/C build closure; do not imply cached/locked means
qualified. No native OpenSSL/SDK dependency or optional network feature in core,
math, CPU/GPU or default inspection builds. sha2 0.10.9/rustix are already cached;
no dependency acquisition is needed for this audit. Keep Error ABI unchanged.

HF: resolve branch/tag once to exact commit; all listing/subfolder/filter/index/
sidecar requests use that pin. Failure to pin is an error or explicit offline pin,
not vLLM's best-effort mutable revision fallback. Discover exact requested quant;
show ambiguity/missing selection, never silently use first-file fallback. A
converted GGUF repo may lack official config/tokenizer: roles may name separate
explicitly authorized/pinned repos. Model-card URLs/config references do not grant
new authority. Optional auxiliaries stay data-only and separately pinned.

GitHub: resolve repository ref once; bound tree traversal and reject truncated
closure. Release asset ID plus expected digest (if available) binds content;
null-digest first-seen content is explicitly weaker. Git blob OID framing is not
raw SHA256. Recognize LFS pointer payload and either use a separately qualified
LFS resolver or typed Unsupported; never install it as requested model weights.

Resume: persist secret-free immutable source/validator/length/prefix metadata,
rehash saved bytes, request If-Range +Range, validate206 Content-Range start/total
and identity encoding, then append. A200 response restarts an OWNED partial or
fails, never concatenates. Changed revision/validator cannot mix snapshots. Final
full hash/size validation is mandatory. Bound retries/backoff/deadlines even for
unknown sizes; cancellation is observable between bounded reads/timeouts.

Credentials: env/profile inputs; avoid plaintext argv/receipts. Token scope is
provider+normalized origin/profile, never arbitrary redirect/mirror. No HTTPS
downgrade or forwarding Authorization across origin. Signed URL query/header
secrets must be scrubbed from errors/progress; do not persist expiring signed URLs
as object identity. Mirrors require explicit endpoint policy. Private/gated access
is not automatic license acceptance. Offline cannot check remote token revocation;
local private cached access policy must say so. No pickle/remote Python/hooks.

## Acceptance design (A1 results in model-acquisition/TODO.md; A2/A3 pending)

ACQ-A01 XDG unset/empty/relative/absolute, explicit override, permissions and roots.
ACQ-A02 duplicate/escaping/symlink-raced paths, JSON/size/count/arithmetic bounds.
ACQ-A03 tiny complete multi-file import, digest mismatch/truncation/unknown bounds;
         data receipt remains separate from executable eligibility.
ACQ-A04 every publication fault boundary + real child exit/restart: old/new/absent,
         never partial snapshot, mixed shard closure or unacknowledged ref success.
ACQ-A05 two processes same/different object and ref, dead publisher, lock-inode ABA,
         canceled waiter vs live owner, bounded waits and single logical binding.
ACQ-A06 snapshot lease through ref update and attempted removal; typed refusal
         until actual removal protocol qualifies, with accounting restored on error.
ACQ-A07 admission before body read, full disk/quota/metadata peaks and failure release;
         no fake global RAM/cross-process GPU admission claim.
ACQ-A08 real network-isolated offline hit/miss with a provider that panics if called.
ACQ-A09 default CPU/core/minimal inspect closure unchanged, optional cache builds,
         one source issuer/catalog/arbiter and no new inference family claim.
ACQ-A10 later A2 scripted+actual loopback HTTP fixtures cover redirects/auth/ranges/
         changed validators/error redaction; actual offline net isolation separate.
ACQ-A11 later A3 existing approved local .8B/2B import->same engine exact checks;
         no new checkpoint download,4GiB/sharded-runtime limits unchanged.
ACQ-A12 independent finite capability coverage audit and honest CLI supported/
         Unsupported table; no `every upstream capability` claim from A1/A2 alone.

## Inventory gaps / completion decision

The reviewed acquisition capability inventory exposes 57 capability rows and their first-unit
or deferred/loader/delegated status. None is implemented by this audit. Important
roadmap omissions/refinements: explicit HF Xet high-performance delegation,
non-weight/sparse-quant sidecars, cache-env compatibility, stable lock inode
retirement, release assets without digest, cached-token revocation semantics,
4GiB/same-descriptor verified-copy and raw-path catalog lease gaps.

Do not call57 exhaustive. Close finite frontiers before parity declaration:
all acquisition CLI/config/env/public entrypoints ->callsite/row; aliases and
intentional differences justified; selected Hub/Xet/alternative-model-hub/RunAI/serialized-tensor/
ModelExpress versions pinned and their concrete delegated API surfaces enumerated;
HTTP helper source location found (old common/http.cpp404 is not an absence proof);
source tests and provider protocol failure matrix mapped. Loader formats/plugins
remain visible in the separate execution ledger rather than invented transports.
The historical research review required no source fetch. Its raw captures are retired; the roadmap retains reference revisions and primary-source links. Revalidate source pins when closing future parity gates.

````

## Source: .agents/plans/NEXT_AGENT_PROMPT_2026-10-17.md

Original UTF-8 bytes: 4351; SHA256 `15734cf3bef345bed0b121768d695d0ffbad0974c758b781e8db40bec20520b8`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Starting prompt — Zerogravity takeover, 17 October 2026

Copy the following into the next agent. It onboards and revalidates the paused
repository; implementation resumes only when the owner explicitly requests it.

```text
You are taking over the live Zerogravity Rust repository at
/home/pvl/spaces/abird/src/zerogravity. This is an existing implementation,
not a new project or architecture rewrite. If using dotlink-l5 from
/home/pvl/spaces/abird, use ./src/zerogravity as the working directory.

Before implementing anything, read IN FULL:
.agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-17.md
It is large: read bounded chunks without truncation, retain working notes,
and follow its complete document/decision/history/plan source catalog.

Then read the current AGENTS.md, .agents/plans/TODO-SPEC.md,
.agents/plans/TODO.md, model-acquisition/TODO.md and README.md, and both
linked completed-TODO archives. Read the relevant core architecture/specs,
all active plan TODOs and the portable paused-draft context referenced
by the handoff before changing their code. TODO files remain the live queue;
the handoff is a dated context snapshot, not a second tracking system.

Revalidate live Git HEAD, branch/index/worktree changes, the two unfinished
SIDE worktrees, active processes/Cargo ownership and the exact source delta.
The last signed implementation checkpoint was f197c9c (explicit GGUF quant),
preceded by f22f759 (candidate discovery). Later handoff documentation and
rolling TODO changes may be uncommitted. Preserve them; do not reset or
blindly copy an old SIDE over MAIN. Some old owner-approved checkpoint labels
no longer resolve after history rewrite. Use the handoff's tested recovery
refs and explicit missing-history limits, not nonexistent original commits.

Current accepted original-model resident GPU/native CUDA/HIP support is
bounded to its documented profiles/512 consumed tokens. CPU functional8K,
independent8K GDN drift, deferred16K/32K and the local Renoir/gfx90c launch
hold remain distinct. Qwen3.8/Gemma4 metadata contracts are not trained
inference. Serving, advanced decoding, streaming and distributed/multimodal
runtime goals remain open. Do not restart the already completed H5 sequence.

C3 filters and C4 native Xet indexes are incomplete, unmerged and untested.
Read their exact paused TODOs and portable source patches under
.agents/docs/handoff-2026-10-17/. C3 has only an unwired, not-compile-ready
filters.rs. C4 has only three draft production files, no CLI integration
or integration tests. Preserve them. On a future explicit resume, finish
and qualify C3, reconcile the accepted baseline, then qualify C4 through
one existing provider/cache/engine path; continue other live plans without
inventing completeness or numerical/performance claims.

Use subagents for independent implementation/review/test work; act mainly
as coordinator, gate, reconciler, decider and merger. Give each agent an
own runs/<plan>-<agent>/TODO.md and explicit paths. Only one MAIN Cargo owner;
no SIDE builds without a serialized grant. Merge bytes with fresh mtimes.
Use the documented locked/offline native user/network isolation and actual
focused/full/source/model/physical checks for the change being qualified.

Preserve one Rust mathematical engine, immutable source identity,
independent StateRoots, one physical ResourceArbiter, fair workers,
actual completion/fencing and quarantine. Never widen tolerances,
substitute CPU after a selected GPU failure or claim unqualified support.
The root README.md is user-authored and must remain byte-identical.
Use the simple TODO workflow; do not restore evidence archives,
SHA inventories or staged tree/blob receipts. Actual model/content hashes
and source/numerical integrity checks remain necessary.

Do not download models/SDKs, modify device privileges/recovery, mounts,
persistent services/live caches, push/force-push/tag/publish/deploy or perform
destructive Git operations without applicable explicit authorization.
Do not launch or reset held local AMD hardware.

Get fully up to context, report the live checkpoint, any source/draft drift
and the next concrete work. Keep runtime implementation PAUSED until I
explicitly request resumption. This takeover prompt authorizes reading
and revalidation, not a new implementation or publication phase.
```

````

## Source: .agents/plans/NEXT_AGENT_PROMPT_2026-10-19.md

Original UTF-8 bytes: 8956; SHA256 `4566d713995969bf048ed03e65ef4895c5b2566dc66fb159468048f469be76ad`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Starting prompt — Zerogravity comprehensive takeover, 19 October 2026

Copy the text below into the next agent. The active task is the P12 batch-native resident GPU forward implementation; the handoff also preserves the complete original architecture and all later model/SSD/serving priorities. This is an implementation handoff, NOT the earlier 17 October documentation-only pause.

~~~text
You are taking over the EXISTING live Zerogravity Rust repository at:

  /home/pvl/spaces/abird/src/zerogravity

Use the connected @dotlink-l5 local machine connector; specify that exact directory in every tool call. This is an established, source-verified engine, not a greenfield project and not an invitation to rewrite it.

FIRST: Read the NEW complete handoff progressively, without truncating its context:

  .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md

Read its up-to-date first 16 sections, Appendix A containing the complete previous 3,430-line October 17 historical handoff and decision/recovery register, Appendix B containing live-plan TODO snapshots, and Appendix C containing the exhaustive docs/plans/research/prior-art reading catalog. The embedded October 17 pause and draft statuses are HISTORICAL and were superseded by the user's explicit implementation continuation. Do not resume older unmerged worktrees or the C3/C4 acquisition drafts.

Then read current:
  AGENTS.md
  .agents/plans/README.md
  .agents/plans/TODO.md
  .agents/plans/serving-and-speculation/{README,TODO,FUSED_GPU_BATCHING}.md
  .agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md
  .agents/research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md
  .agents/plans/IMPLEMENTATION_ROADMAP.md
  .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md

Consult other canonical specs listed in the handoff when their ownership is relevant. These source files and active TODOs outrank a dated handoff.

REVALIDATE BEFORE ANY CHANGE:
  git status --short --branch
  git log -12 --oneline
  git worktree list
  git diff --check
  ps for Cargo/rustc/active GPU-serving processes
  relevant source ownership and configured device/model/capacity
  active agent-owned changes/branches/logs; do not overwrite them.

At handoff creation, the authoritative signed HEAD was
  3ec431a2df62bbcae5091a048f285801f151da7e
  feat(gpu): qualify native batched GGUF quantized projections

MAIN was clean and nine commits ahead of local origin/main, with no other worktree. It may have moved; current Git wins. The root README is user-authored: DO NOT EDIT it.

OWNER'S CURRENT PRIORITIES, IN ORDER:
1. Genuine vLLM-equivalent fused continuous GPU batching and physically verified OpenAI-compatible serving, then actual source-matched vLLM goodput benchmark.
2. Source-preserving SSD/GPU/RAM tensor residency integrated into the SAME engine, with real GPU readiness/lease and trained resident-vs-SSD StateRoot/logit proof.
3. Original Qwen3.8 27B CPU/BF16/FLA model source, trained numerical and large-source/SSD GPU execution.
4. Gemma4 E2B, then 26B-A4B and dense 31B, each separately source/numerically qualified; Flash-Next and all other multimodal/distributed/speculative work remain on their own plans.

WHAT HAS ALREADY PASSED (DO NOT REIMPLEMENT):
- Original real Qwen3.5 0.8B/2B CPU GGUF and 512-consumed-token native resident GPU bounded profiles, source/tokenizer/greedy/state/cancel/quarantine and full-vocab/all42-domain parity.
- Real OpenAI local CPU HTTP and resident CUDA GPU serving, multiple parallel sessions/alias-sharing, strict greedy-only JSON/SSE, queues, cancellation and resource admission. HTTP state-cohort publication is now verified for compatible one-token final prefill; the GPU model graphs still execute serially per request.
- Native B=2/4/8 GPU CUDA fused isolated F32 and GGUF Q4_0/Q4_1/Q8_0/Q6_K projection operators, batched elementwise/norm/GDN/KV/attention, source-exact versus B1.
- Real CUDA Qwen3.5 0.8B B2/B4/B8 ResidentBatchRows and PreparedResidentCohort: exact 248,320 logits and all42 state domains PER independent session, source/Ready/epoch/root/state arbitration, aborted/stale/foreign rollback and reclaimed leases. Full model GPU kernel fusion is NOT yet implemented; HTTP must keep gpu_fused_batches=0, and batch_mode=fused must refuse.
- Bounded checksum-verified immutable SSD source chunk/window IO and leased staging; no trained model streamed from SSD.
- YAML model/pipeline compiler/context filters/grants/decision adapter scaffolds, original Qwen3.8 BF16 mathematical diagnostics and pinned reference-trace comparator. No trained 27B or Gemma4 inference.

THE IMMEDIATE TASK: Implement the next physically qualified P12 slice, not another documentation or mock-only phase.

1. Inspect exact current source:
   crates/zerogravity-core/src/accelerator/program.rs
   crates/zerogravity-core/src/session_root.rs
   crates/zerogravity-gpu/src/resident_program.rs
   crates/zerogravity-gpu/src/resident_inference.rs
   crates/zerogravity-gpu/src/resident_batch.rs
   crates/zerogravity-gpu/src/batch_{projection,quant_projection,elementwise,recurrent,kv,attention}.rs
   crates/zerogravity-cli/src/serve/{gpu_engine,gpu_batch,http,engine,mod}.rs
   tools/serve-bench.py, tools/serve-compare.py

2. Introduce the smallest source-qualified B>1 batch-native ForwardPlan/ResidentProgram execution slice that actually CALLS the accepted multi-row GPU kernels for compatible same-source model lanes. Preserve the immutable weight source once, independent per-row prior/candidate KV/GDN, masks and causal frontiers, original SessionAuthority, ResourceArbiter, exact BufferLease/Ready/fence/quarantine, and distinct per-request output/greedy. DO NOT fork the Qwen token loop or invent another StateRoot issuer.

3. Qualify original trained Qwen3.5 0.8B physical CUDA B2 full-model complete logits/all42 domains against the accepted CPU/B1 references, one real fused multi-row kernel per appropriate graph stage, unequal prefixes, EOS, cancellation/stale epoch/foreign row and physical completion. Expand to B4/B8, Qwen3.5 2B and qualified quant codecs only after their smaller source/physical gates pass. Report actual kernel-launch counts, not grouped scheduler waves or model forward API calls.

4. Keep HTTP batch_mode=fused disabled until the complete B>1 resident model graph has passed all advertised-backend source/state/cancellation gates. Later integrate source-safe paged/hybrid KV and finite fair continuous batching; compare to true vLLM V1 on same weights/quant/tokenizer/hardware under matched cold/warm offered and committed load. tools/serve-compare.py is only a harness; vLLM was NOT installed or running on this host at handoff. Do not create fake comparison results.

5. Run focused tests, then the complete isolated gate:
   CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh

   Then source-appropriate strict all-target Clippy -D warnings and ACTUAL physical GPU original-model tests if needed. Run the separate approved Qwen3.5 .8B/2B original cache/direct AND HF-import release parity routes from handoff section 12. Verify full-vocabulary logits/all42 domains and status/exit receipts; a compiler pass is not a device pass.

6. Observe approval boundaries: no model/SDK downloads, held Renoir/gfx90c AMD launches or resets, external host privilege changes, destructive Git operations, root README edits, push/tag/release/deploy unless separately authorized. CUDA runtime/CCCL/NVRTC are present on the qualified machine via already-installed Nix packages; set CUDA_PATH correctly before physical CUDA, not by acquiring new SDKs. Avoid unqualified CUDA/Vulkan fallback. The separate offline Nix source-integrity derivation has been blocked by an absent pinned nixpkgs cache/SSL trust, while Nix formatting passes; report this accurately.

7. Keep the plan workflow simple:
   .agents/plans/TODO.md = one live queue;
   .agents/plans/serving-and-speculation/TODO.md = current P0 owner;
   .agents/runs/<phase>-<agent>/TODO.md = work in progress.
   Commit only reviewed implemented source/tests/docs with signed commits and verify the exact staged paths. Do not push. Record genuinely remaining work as unchecked items, leaving already-completed items under their signed commits.

Use independent subagents for focused review, implementation and testing where available, with disjoint file scopes and one MAIN Cargo owner; otherwise do the work with the available tools. Make progress autonomously. On an interruption, inspect actual live tree and test logs and continue; never rewrite already-qualified work or claim a phase finished from partial results.

Report succinctly: current verified HEAD, physically passing acceptance, exact unqualified fusion/vLLM/SSD/model gates, test exits, changed paths and next execution step. Do not confuse multi-root atomic publication with native multi-row model fusion.
~~~

````

## Source: .agents/plans/README.md

Original UTF-8 bytes: 3099; SHA256 `a7c38b5a54b7cabb1ee71b7e542fd0fc5b8e1a1f227e54f3a879acaa7b4d0804`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Plans

This folder is the agent planning entry point. Start here, then work from
`TODO.md`. Git is the authoritative history (author, time, diff, provenance sha);
this folder keeps the live queue and per-plan checklists.

## Layout

```text
.agents/plans/
  README.md                 overview + spec
  TODO.md                   queue: Active / Icebox / Closed
  <plan>/README.md          plan details + author/co-authors
  <plan>/TODO.md            Base + Committed + Uncommitted + Tests
  <plan>/HISTORY.md         optional chronology (heavy plans)
  archive/<year>-Q<n>.md    quarterly rollup of closed plans
```

Plans are directories, flat under this folder; do not nest plans. One plan per
effort; split into sibling plans only when a design is too large for one TODO.

## Queue — `TODO.md`

One line per plan:

```markdown
## Active
- [~] P0 <plan> — @owner — <one-line> — [details](<plan>/README.md)

## Icebox
- [ ] <plan> — [details](<plan>/README.md)

## Closed
- [x] <plan> — <closing-sha> — <date>
```

- Active is sorted P0 → P2; moving a plan from Icebox to Active is the act of
  prioritising it.
- `[ ]` open · `[~]` active · `[x]` done · `[!]` blocked.
- P0 requires human promotion and a named incident coordinator.

## Plan — `README.md`

Open with the plan's authorship and ownership:

```markdown
Author: <user>
Co-authors: <comma-separated AI models used>
Owner: @<user|agent>
```

Then what the plan is and why it exists, plus **Design**, **Decisions** and
**References** sections as needed. Split a section into `DESIGN.md`,
`DECISIONS.md` or `REFERENCES.md` only when it grows large, and link back.

## Plan — `TODO.md`

```markdown
# <plan>

Base: `<sha>`

## Committed
### `<sha>` — <what this set landed>
- [x] <item>

## Uncommitted
- [ ] <item>
- [ ] <item> — <blocker/reason>

## Tests and verification
- [x] <test or gate> — `<sha>`
- [ ] <test or gate>
```

- `Base:` is the commit the plan started from.
- Completed work is grouped under the commit that landed it; append new groups.
- Tests are a flat list; tick each with the sha when the commit's provenance
  group is added.
- A plan is done when `## Uncommitted` is empty and every test is `[x]`.

## History — optional `HISTORY.md`

For heavy or multi-agent plans, keep an append-only chronology:

```markdown
- 2026-10-18 09:14 — @owner — opened; claimed P2
- 2026-10-18 09:40 — abc1234 — implemented parser; tests pass
- 2026-10-18 10:02 — @reviewer — reviewed; requested X
```

Light plans can skip it; the commits are the history.

## Parallel work

The owner on the queue line owns the plan directory; no one else edits it.
Parallel agents keep scratch under `.agents/runs/<plan>-<agent>/TODO.md` and
append back only when finished. The coordinator reconciles the queue line and
the plan's committed groups.

## Closeout

All tests `[x]` and `## Uncommitted` empty → move the queue line to `## Closed`
with the closing sha and date. Quarterly, roll older `## Closed` entries into
`archive/<year>-Q<n>.md` and prune their plan directories; git retains them.

````

## Source: .agents/plans/RELEASE_ACCEPTANCE_GATES.md

Original UTF-8 bytes: 27297; SHA256 `84377b654b99fe828442debb5810f13feac0600e961c34b04358fe3fa9a9b64c`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — Model-neutral release, quality and composability gates (v4 reviewed)

**2026-10-12 · NOT EXECUTED.** All stage and release gates are proposed verification rules for the original modular AI architecture, not declarations that code exists. Use `IMPLEMENTATION_ROADMAP.md`, `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`, the modular AI reference architecture and composition contracts as authority.

## 1. Approval states

Each capability, not only each model name, has `PLANNED`, `REFERENCE`, `EXPERIMENTAL`, `FUNCTIONAL`, `VERIFIED`, `PERFORMANCE_QUALIFIED`, `UNSAFE_DISABLED` or `BLOCKED` status by (model architecture/checkpoint/quant, task, device/backend/driver, numeric contract, context/phase). A GPU kernel may be performance-qualified for prefill but not decode or a different quant, and an accelerator physically present but power-unsafe is not eligible for automatic scheduling.

A source repo claims a speedup ≠ zerogravity measured a speedup. Unknown trained operator and unsupported model artifact must fail honestly.

## 2. Foundational release gates

| Gate | Evidence | Hard failure |
|---|---|---|
| **G0-A — semantic correctness** | safe real checkpoint/config/tokenizer/quant, Rust scalar/CPU oracle and exact trained ops/position/state | wrong math, unknown op silently guessed, mutable state leak |
| **G0-B — kernel and command resources** | physical GPU Rust-authored kernel parity, legal RIR fallback, CIR alias/lease/timepoint checks and compiled direct dispatch | stale buffer/device pointer, illegal rewritten state, interpreted IR in token hot path |
| **G0-C — useful vertical slice** | model→SIR→compiled CPU/GPU→TaskCoordinator→committed task output and accurate ModeReceipt; forced missing optimized kernel | no real verified output, separate model-specific server required |
| **G0-D — composability** | novel model op + device provider + non-token TaskProgram added without unrelated service/allocator/core rewrite | architecture requires new inference engine |
| **G1 — portable product** | supported models/formats, real task outputs, local safe SDK/API, CPU+validated GPU and strict fallback | incomplete adapter, unsafe input, fake support |
| **G2 — serving/memory** | multi-session and heterogeneous task stages with physical admission, prefix sharing, fair P95 and cancellation recovery | host overcommit, state cross-talk, starvation or uncommitted output |
| **G3 — extreme specialized** | representative dense/hybrid/sparse/MoE/multidevice profile with independent quality and matched speed, device safe, no other engine process | hidden approximation, unsupported/unsafe kernel, false speed ranking |
| **G4 — broader AI tasks/distribution** | accepted ASR/TTS/image/video/embedding/rerank TaskPrograms and remote/multi-device only per verified cell | fake next-token task, irrecoverable task state, partial peer commit |
| **G5 — sustainable release** | security/ABI/schema/source license/reproducibility/performance evidence and extension drill | untrusted executable model code, unreviewed source, broken upgrade |

**No gate passes from synthetic simulation alone where physical GPU/NPU or official model quality is required.**

## 3. Required semantic and numerical quality controls

- Source checkpoint hash, tensor manifest, tokenizer/template and quant dialect verified before inference.
- Mathematical operator/position/masking/routing/compression/recurrence semantics independently implemented and tested.
- New stateful op has state transition/fork/snapshot/restore/commit/abort tests and SIR effect verification.
- Strict `Reference` vs validated numerical-tolerance vs approximate opt-in must be visible and not conflated with stored quant.
- Full logits and layer metrics where available (maxabs, RMS, KL, greedy match, sampled target distribution, context/long-task quality).
- Encoder pooling/reranking, ASR audio timestamps/WER/quality, TTS speaker/acoustic/PCM quality, image generation/edits, and video fidelity require model/task-specific independent acceptance. Do not use language perplexity as universal task score.
- Ongoing tool/reasoning/grammar and partial output/state read/write boundaries tested for cancellation and safety.
- Every specialization can be disabled to compare against the exact portable model; safe fallback may require layout conversion/replay and will report cost.

## 4. Resource and async safety

- Physical RAM/VRAM/OS headroom and real unified/device addressability distinguished, cgroup and pinned staging included; no dangerous default overcommit.
- All GPU/NPU/CPU/SSD/remote buffers have validated ResourceLease and generation, read/write aliases, timepoint/event quiescence and failure scope.
- Expert source loads, disk lookup rows, snapshot files and media streams validate exact artifact/offset/length/schema; EOF or cache generation mismatch fails closed.
- A trained routed expert or sparse attention selection cannot be skipped merely because its parameter tier is cold.
- Temporary prefill/diffusion scratch overlaps cache slabs only after old readers retired and lease fencing.
- Device lost/timeouts may leave DMA outstanding; quarantine until safe, no stale pointer reuse.
- Distributed collectives must agree model/quant/semantic state/accepted frontier or abort/rebuild.
- NPU and GPU sharing a known unsafe fabric/power domain stay OFF until physical safety qualification; no automatic root/sysfs/firmware/BIOS/IOMMU/clock changes. A process watchdog cannot recover whole-machine hangs.

## 5. Task and service acceptance

A task must declare `TaskKind`, logical input/output ValueTypes, stage DAG, iteration/control and output frontier/cancel behavior, model/adapter/codec identity and resource budgets. At least one non-token task must pass reuse of the same compiled region/device/TaskCoordinator without a model-specific runner fork.

Service/SDK:
- canonical TaskRequest and typed output, model/error/capability reports;
- streamed text/audio/image/video/structured results only when committed/valid according to task frontier;
- sampled text proposal never emitted before authoritative target verification;
- request cancellation/timeout leaves previously emitted valid output intact, rolls back uncommitted state and returns correct terminal reason;
- tool external actions require their own permission sandbox; no arbitrary model-file remote executable code;
- local-only bind by default, remote authentication/quota, private prefix/media state and redacted metrics;
- correct token counts versus audio frames, video frames and diffusion steps, accurate ModeReceipt for kernel/device/quality/fallback.

## 6. Long-run composability release test

Required demonstrator under controlled real code:
1. Add a model architecture composed of existing SemanticOps with only ModelDefinition/tensor manifest addition.
2. Add a genuinely new recurrent/attention operator with CPU reference, state schema and at least one generic backend.
3. Add a kernel provider for a new device ISA, with no ModelDefinition or service type modifications.
4. Add an encoder or iterative TaskProgram with distinct stage unit/output and no decoder-only loop.
5. Add a new storage/IO tier or async command mapping without changing math model and task code.
6. Disable extreme providers; semantically supported model/task runs portable/reference with correct state or honest physical capacity refusal.
7. Quantify unrelated modules changed, schema churn, Cargo build time, runtime CPU overhead, portability degradation and kernel performance.

**Pass rule:** functionality/quality/state/resources correct, no duplicate serving engine and no project-named core types, and ordinary serving remains operational after each extension. If patch changes every layer, revise the interface rather than add a model-name exception.

## 7. Performance qualification

Every performance claim has:
- matched source model, checkpoint/quant and task/control/sampler, prompt/context/regime and output units;
- actual CPU/GPU/NPU/device/fabric/driver/firmware and physical memory/cache state;
- direct baseline (same general path) and external peer reference when legitimately comparable;
- repeated paired ABBA/BAAB, raw time distribution, p50/p95/p99 TTFT/interunit/completion, energy, memory and source quality;
- distinct tokens-per-second for autoregression vs prefill vs speculative accepted output; distinct audio/video/encoder and image throughput units;
- explicit numerical error and resource safety failure logs; no favorable hardware-clock confounding.

**Promotion:** performance-qualified specialization becomes eligible for plan selection only in its validated shape/driver/model/quality/safety cell. Unqualified kernel remains experimental and not default. Worsened p95 or quality can outweigh isolated GEMM speedup. Source-author claims remain in prior-art evidence, never baseline "achieved" by zerogravity.

## 8. Source/license/security qualification

- Exact code file license/SPDX/header, required NOTICE and source provenance/commit reviewed independently of repository root badge.
- MIT-licensed candidates considered only after audit; any different embedded file/model-weight/device executable license handled separately.
- Non-MIT source-specific APIs/code do not become normative interfaces or copied implementation without explicit user policy resolution. Public mechanism research can inspire independent design.
- User's Rust-owned implementation policy: a foreign GPU/NPU kernel wrapped in Rust does not alone qualify as Rust-authored owned kernel.
- Compiler source/driver device-program trust, toolchain provenance, cache invalidation, snapshot schema and model artifact hash verified.
- No hidden third-party inference engine. Source project names retained in prior-art records, not architecture types.
- No release despite incomplete critical parser/security/physical-memory/async ownership checks.

## 9. Evidence/report schema (conceptual)

```text
result: PASS | FAIL | NEUTRAL | INCONCLUSIVE | BLOCKED
phase, gate, run_timestamp, git_sha or source_hash
model/task/quant/tokenizer/semantic_op/state_schema IDs
kernel/plan/driver/device/clock/fabric program identity
input/output task geometry, concurrency and mode receipt
reference numerical and full task quality outcomes
committed units/time, TTFT/P95, physical/pinned memory, energy
resource IO/cache metrics, safety/failure, privacy/authorization
raw artifacts paths, repetitions, uncertainty, skipped reasons
next decision / guarded fallback / reviewer source license
```

An unknown metric remains null/omitted rather than a misleading zero. Tests, raw data and hardware/OS error reports link the exact implementation state. No user private prompts or copyrighted weights in public artifacts.

**Implementation closure:** follow [TODO-SPEC.md](TODO-SPEC.md). Record actual gate commands/results, source/model-content identities, hardware scope, failures and remaining work in the owning plan/run TODOs. Commit verified implementation, tests and relevant design/policy updates together; completed work is grouped under its actual landing commit and unfinished work remains in Uncommitted. Plans alone do not pass a gate. No generated handoff, archive or documentation SHA manifest is required.

## v4 mandatory architecture acceptance refinements — 2026-10-12

These rules supersede older shorthand claims of “all domains commit atomically” and “one owner for all allocations” where those phrases did not specify physical mechanics. Add to G0/G1/G2/G5 without inventing new product release dates.

1. **Single publication owner:** no mutating per-domain public `commit`; prepared versions of all required state domains validate same TaskFrontier, event readiness, schema and numerical contract; TaskCoordinator/shard performs one `PublishRoot`, only then emits committed output sequence. CIR/device commands do not publish semantic session state.
2. **Async-first hot path:** performance-qualified device-native StepProgram avoids mandatory host router/sampler/token readback and semantic IR interpretation per inner iteration when model/support permits; correct host-loop fallback is allowed and honestly labeled `FUNCTIONAL`.
3. **Hierarchical memory:** global admissions reserve conservative physical memory, device-local suballocators own fast generation-tagged leases; no global lock per op/token. OS memory can change externally; no absolute no-OOM promise. Inject allocation failure, stale lease and delayed DMA.
4. **Shape guards and bounded compilation:** fixed-rank dynamic dimensions, ragged offsets, divisibility/context/quant/ISA constraints checked before launch; no unbounded JIT/variant explosion. Cold unsupported shape selects safe prepared portable plan or explicit diagnostic.
5. **One minimal TaskDriver:** no uncontrolled agent/workflow language or `ExternalAction` in core. Non-token model stages have their own frontier/output and share worker/resource backend, without fake LM sampling state.
6. **Fast specialization containment:** a full fused layer/model candidate is replaceable at a closed region with equivalent values/state/effects/numeric contract and can be forcibly disabled with portable model still supported.
7. **Distributed and transport honesty:** coordinator epoch fencing and Ready cohort precede logical root publish; output event sequence/backpressure/cursor documented, no network exactly-once delivery claim without explicit durable protocol.
8. **Training is future optional:** architecture allows ParameterVersion/GradientProgram/optimizer state TaskDriver without modifying inference state owner or forcing autograd/training dependencies into first release. Synthetically prove the seam, do not claim trained models run.
9. **Fewest mandatory abstractions:** SIR and compiled ExecutableBundle are the only enduring representations; region/kernel/command views are compile internals. One library+binary (or few crates) preferred over a nine-crate stub hierarchy; extra abstractions require measured value.
10. **Project neutrality and evidence:** source-name/foreign kernel dependencies confined to audited prior art, actual GPU driver/FP8/BFP8 numerical/provider choices validated and explicitly qualified; no silent trusted model approximation or unsafe privileged host changes.

**Dedicated tests:** `S0-V4-A…J` in integrated Stage0 and `H32…H43` in experiment program. Inability to do one of these tests at current hardware is **BLOCKED/SKIPPED**, not VERIFIED. Architecture is still **design only**.


## v4.1 backend portability and honesty gates — 2026-10-12

1. **Universal executable kind:** a `RegionExecutionProvider` candidate can cover a native kernel, captured graph, AOT whole-region graph or static dataflow fabric program, returning an `ExecutableBundle` and typed async ready/fail event. No new model-specific server/allocator/session writer and no device-specific semantic IR.
2. **Full state bridge:** stateless region can be qualified without mutable snapshot; **stateful region** requires privately prepared state, snapshot/restore or bounded recompute/checkpoint under exact schema/epoch/cancellation. Opaque in-place unrecoverable state must be rejected for general speculative/forkable decoding even if program compiles. Device program cannot publish StateRoot.
3. **Graph shape/precision proof:** bounded fixed-rank dynamic dimensions, ragged/sparse offsets, native quant scale/block representation and compiler region shape; unsupported layout/op/quant gets legal smaller region or typed refusal, not uncontrolled AOT recompilation or wrong model arithmetic.
4. **Program lifecycle:** compile/codegen/upload/warm/resident/teardown measured and budgeted, compiler/runtime/driver/firmware versioned; recompilation/cache invalidation tested. No leaked compiled handles or raw pointers in persisted snapshots.
5. **Physical topology and collectives:** coherent SoC vs discrete GPU vs multi-GPU rack vs on-chip SRAM/static fabric have distinct memory/IO/transport/link resources, rank/epoch fault fencing; one system marketing memory capacity cannot be treated as one pointer.
6. **Remote API separation:** vendor inference endpoint without trusted exact model op lowering and StateBridge is an independently authorized external **TaskServiceAdapter**, with visible privacy/credential/egress/model-sampling differences. It is **not** proof of native zerogravity inference on that accelerator.
7. **Rust-source/license capability:** using a vendor graph compiler for device executable is not evidence low-level kernels/compiler were written in Rust; classify it as optional authorized external toolchain policy, with exact source/program provenance and license. Missing SDK/compiler/hardware reports `ARCHITECTURE_COMPATIBLE` not `FUNCTIONAL`.
8. **Physical qualification:** each hardware/model/quant/context/task/driver/compiler variant needs official/independent semantic/state/quality and matched real end-to-end throughput/p95/energy tests before `PERFORMANCE_QUALIFIED`. Theoretical peak and source-vendor reported speedups aren't our evidence.

**New optional tests:** S0-FABRIC-A…H and experiments H44…H55. The portable CPU/GPU model remains valid when no external compiler/wafer/TPU/ASIC hardware can be accessed. Details: `../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md` and source-attributed `../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md`.

## v4.2 distributed placement and networked execution acceptance

**New distributed release cells are optional and separately qualified** from local model/kernel execution. Architecture support for a placement topology is not real distributed functionality.

1. **Independent placement proof:** same exact SemanticProgram/TaskDriver can independently map compute, immutable parameters, mutable KV/recurrent/indexer task state and immutable prefix caches to distinct authorized nodes, with one TransferPlan and StateRoot schema/lease.
2. **No runtime forks:** RemoteRegionWorker runs the same compiled closed region and DeviceBackend as local, without another API server/model-specific allocator/TaskCoordinator. Single host runs with zero cluster daemon/discovery/network hot-path overhead.
3. **Layer + cache correctness:** per-layer-group remote model worker keeps the layer's KV/recurrent state, receives only necessary activation boundary; model logits/full state match all-local reference after chunked prefill and C1 decoder, including cancellation.
4. **Separated compute and state:** remote cache hydrate vs remote attention beside KV vs direct remote KV reads have different cost/StateBridge requirements and must not be conflated. Slow/relayed WAN per-token KV readers ordinarily refused when worse, not silently used.
5. **Operator-specific token sharding:** mathematically exact masked/causal attention softmax shard summaries reference-tested, no unproved generic GDN/KDA or sparse selected indexer partition.
6. **Full hybrid P/D:** destination reserve and verified complete KV+recurrent+indexer+RNG/grammar+modal state transfer with committed frontier, or explicit replay/fallback.
7. **Tunnel, peer trust and privacy:** explicit authorized node identities, private transfer policy for source weights/prompts/KV/media; authenticated bounded control and reliable bulk, direct/relay cost measured and no assumed RDMA over VPN. No automatic remote shell, public service, credentials or model weights redistribution.
8. **Epoch/failure:** stale StateShard lease, wrong quant/model/adapter, lost ACK, network partition, stale coordinator or live DMA failure cannot publish incorrect TaskSession root or leak output. No unimplemented durable exactly-once or automatic HA claims.
9. **Topology plan lifecycle:** compiled plan frozen during active collective/attempt, safe relocation only at committed TaskFrontier with reserve/transfer/hash/epoch fence, prepared cohort and physical buffer retire; fallback to old root or replay.
10. **Real benchmarks:** two authenticated physical nodes and direct LAN and optional tunnel variants require exact model/quant/task/hardware, bytes/step, P95 RTT/bandwidth/jitter, CPU encryption overhead, TTFT/ITL/committed throughput, energy, device headroom, failure and quality. A larger fitting model is a **capacity** achievement even when C1 t/s lower.

**New optional Stage0 S0-DIST-A…L** and **experiments H56…H70**, all PLANNED/NOT RUN. Node/tunnel connection/deployment requires explicit user authorization and separate security test. Source research remains non-normative; see [distributed placement](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md).

## v4.3 multi-model router, simple CLI and API conformance release gates

**Product correctness is an explicit gate, not just a convenient CLI sketch.** A release must show the **same inference engine** serves direct in-process CLI and one multi-model HTTP endpoint with independent physical model lifecycles, no extra per-model serving engine.

1. **Blissfully simple local:** one installed Rust binary plus a supported local checkpoint runs with `run model` and no config, daemon, internet, remote/tunnel/root change or mandatory model registry authentication. Hardware auto-plan is qualified; unfit model gets precise actionable refusal, not silent quant/model downgrade.
2. **Exact model catalog/aliases:** immutable source SHA+model math+stored quant+tokenizer/template+adapter, separate ModelAlias/Profile revision, ResidentModel/Replica and TaskSession. Hot alias changes affect only future requests; old active sessions keep exact source and full StateCompatKey.
3. **Many-model physical safety:** catalog may hold many models; simultaneously resident models/replicas only when actual RAM/HBM/UMA/OS reserve+graph programs+per-session context fit. Load/warm/idle unload deduplicated and bounded, never evict active unsnapshotable state, live DMA or prepared speculative buffers. User keepwarm/pin cannot override safety.
4. **Exact replica load balancing:** route among equivalent semantic/quant/adapter models with sticky StateRoot/cache locality, queue/P95 and permitted device/tenant/peer; safe explicit full-session migration at committed root only. A configured different-model selector cannot masquerade as an exact model request.
5. **Fair multi-model execution:** cold giant-model wait cannot starve warm small interactive traffic, no unbounded model-load thrashing, bounded per-model and global queue with typed overload errors, existing native within-model batching and per-task scheduler unchanged.
6. **Per-model configuration:** versioned human-editable effective defaults/global/model/alias/CLI/request scopes, deterministic precedence and provenance, invalid/unsupported option diagnostics, immutable/security/physical/precision guards never overridden. Profile changes requiring recompile/load create new deployment; no in-place live model mutation.
7. **OpenAI-compatible subset conformance:** actual tested `GET /v1/models`, `POST /v1/chat/completions` streaming/nonstream JSON/usage/finish/tool-data, Python/JS SDK current pinned versions; embedding if real encoder qualified, Responses stateless subset with genuine distinct event schema if implemented, stateful store/tools separately approved. Unsupported request fields not silently ignored.
8. **API security/privacy:** loopback default, remote binding requires configured auth/TLS/tenant/cors and separate authorized native admin; no private model names/paths, filesystem/media SSRF, user prompt telemetry/credential leakage, automatic remote/tunnel/cloud enrollment.
9. **Multi-model TaskProgram:** target/draft, embedding/reranker, vision encoder/text model and TTS/vocoder are separately bound model artifacts with shared parent task budget; cannot treat different source/quant as replicas.
10. **Cost/performance:** direct local `run` vs router `serve` must not incur a **per-token gateway/second-inference-scheduler** overhead; measured real model/quant/driver C1/C8 TTFT/p95, cold/warm load, memory/energy, cache affinity, distributed placement and denied unknown source responses required.
11. **Client state/stream honesty:** model ID/alias/profile revision and committed output seq pinned, slow client backpressure/cancel, no rejected speculative output or fabricated exactly-once delivery. Stateful Responses/conversation storage needs explicit authenticated retention.
12. **Status discipline:** failed API conformance = unsupported endpoint/field/capability, not “fully OpenAI-compatible”. No model download, daemon/service installation or public API bind without user-approved policy.

**Evidence:** S0-SERVE-A…J and H71…H86; real supported model/TaskDriver/hardware+SDK needed before functionality or performance marks, synthetic architecture proofs only qualify design. The prior-art source UI/SDK research is non-normative; see [model router](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) and [CLI/API](../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).

## P00 packaging-only Rust binary and release artifact gates — U-023

This is a **separate bootstrap acceptance class**, not a claim the model runtime or the main product release is ready.

| Gate | Status (2026-10-12) | Required test |
|---|---|---|
| Cargo toolchain and package metadata | **PASS (local)** | Rust 1.95.0, Cargo.lock offline, no runtime dependencies, version 0.1.0 |
| CLI honest bootstrap behavior | **PASS (local)** | `--help/--version` success; `run/serve` explicit nonzero unimplemented refusal |
| Rust format/unit/integration/lint | **PASS (local)** | 4 unit + 3 CLI tests, rustfmt and clippy -D warnings |
| Local Linux x86-64 release build | **PASS (local)** | `cargo build --locked --offline --release`; `zerogravity --version` exact |
| Shell and Nix source static checks | **PASS (local)** | Shellcheck four scripts; nine Nix file parse and Alejandra checks |
| POSIX installer offline integrity | **PASS (fixture)** | file:// checksum, successful install/repeat, malformed sidecar and wrong version refused |
| Windows installer runtime | **NOT RUN** | Windows PowerShell x86-64/ARM64 source fixture and real release |
| Nix flake build/check/module eval | **BLOCKED BY CONNECTOR** | `nix flake check`, `nix build .#zerogravity`, `nix eval` hit read-only Nix-store source lock |
| Crane six cross release outputs | **NOT RUN** | actual Linux MUSL x86/ARM, Windows GNU/ARM64, macOS x86/ARM and per-OS smoke |
| GitHub Actions/release publish | **NOT RUN** | repo/owner/license and real tag not established; successful hosted CI/tag release required |
| NixOS future service | **DESIGN/SYNTAX ONLY** | module disabled default, no restart loop; real `serve` not implemented |

**Completion rule:** P00 packaging may be described as a *working local Rust CLI bootstrap*, **not** as a fully cross-platform published release or complete P00 source-policy/model-runtime foundation until the outstanding gates are exercised. Installer SHA256 is corruption detection, not a release authenticity signature. The project uses MIT; release ownership/repository and Apple/Windows signing policy follow the [current platform/release TODO](platform-and-release/TODO.md).

````

## Source: .agents/plans/TODO.md

Original UTF-8 bytes: 920; SHA256 `08328d17f8e331e910897c117f4f5a2a60f86681386d3857c28d9c84284c4a1d`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Plan queue

## Active

- [~] P0 serving-and-speculation — @pvl — [details](serving-and-speculation/TODO.md)
- [~] P1 numerics-and-context — @pvl — [details](numerics-and-context/TODO.md)
- [~] P1 memory-and-streaming — @pvl — [details](memory-and-streaming/TODO.md)
- [~] P2 model-execution — @pvl — [details](model-execution/TODO.md)
- [~] P2 model-acquisition — @pvl — [details](model-acquisition/TODO.md)

## Icebox

- [ ] cpu-gpu-optimization — [details](cpu-gpu-optimization/TODO.md)
- [ ] multimodal — [details](multimodal/TODO.md)
- [ ] distributed-execution — [details](distributed-execution/TODO.md)
- [ ] platform-and-release — [details](platform-and-release/TODO.md)

## Closed

- [x] accepted-baseline — `6d74f76` — 2026-10-18
- [x] handoff-cleanup — `bfe672c` — 2026-10-18
- [x] historical-records — `9239719` — 2026-10-18
- [x] pipeline-composition — 2026-10-18

````

## Source: .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md

Original UTF-8 bytes: 55253; SHA256 `e663490c312d8eca92669286468ddd904aa5a44217f6bb0065f8a8fa99bcd9b1`.
Resolve embedded relative paths against the original path, not this volume.

````text
# zerogravity — master architecture v4.3: composable AI execution, model routing and simple user experience


**Design baseline:** reviewed and extended 2026-10-12 (v4.3 local-first UX, native multi-model router, residency, profiles and API compatibility). This is an architectural specification, not an implementation-status ledger. [The plan index](TODO.md) and [accepted baseline](accepted-baseline/TODO.md) govern current completion; [numerics](numerics-and-context/TODO.md), [model execution](model-execution/TODO.md) and [CPU/GPU optimization](cpu-gpu-optimization/TODO.md) retain their unfinished gates. The October 13 progress prefixes have been retired rather than presented as live status.

**Authority:** v4.3 supersedes the older v2/v3 sketches where they differ. The [critical review](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md) retains the rejected-design rationale; earlier document versions remain in Git. Work tracking follows [AGENTS.md](../../AGENTS.md) and [TODO-SPEC.md](TODO-SPEC.md), not archived handoff or manifest procedures.

**Latest integrated, publication-facing authority:** `../../docs/ENGINEERING_SPECIFICATION.md` (ES-00…ES-15 model/task/compiler/state/resource/distribution/router/API implementation contracts), `../../docs/WHITEPAPER.md` (technical argument and limitations), `../../docs/README.md` (documentation categories and authority), and [the current workflow](../../AGENTS.md) with [TODO-SPEC.md](TODO-SPEC.md) (implementation and verification tracking). These do **not** invalidate the exact mathematical/operator/protocol detail below; they resolve cross-document ambiguity and establish the shortest implementation reading path.

**Read-first detailed subsystem references:**
1. `../docs/MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md` — short governing ontology and invariants.
2. `../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md` — illustrative minimal Rust API contracts.
3. `../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md` — universal native-kernel, AOT graph and static-dataflow region provider, lifecycle/state bridges and hardware topology.
4. `../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md` — independent compute/weight/mutable state/cache placement across heterogeneous local and networked machines.
5. `../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md` — authorized tunnel peering, transport, remote state lease/epoch and migration/failure semantics.
6. `../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md` — built-in catalog/alias/model profile, multi-model residency and exact replica/TaskSession routing.
7. `../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md` — one-command CLI, typed configuration, single API endpoint and tested OpenAI-compatible subset.
8. `../docs/MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md` — precise state publication, async workers, hierarchical memory ownership and delivery semantics.
9. `../docs/TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md` — two owning representations, dynamic shape/region guards, TaskDriver, optional gradients, performance.
10. `../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md` — review findings R4-01…R4-24 and independently researched external prior art.
11. `DECISION_REGISTER.md`, `INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md`, `IMPLEMENTATION_ROADMAP.md`, `EXPERIMENT_PROGRAM.md`, `RELEASE_ACCEPTANCE_GATES.md` — user requirements and practical work/evidence.
12. The other subsystem specifications, with v4 supersession note, and separate `../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md`.

## 1. Mission, scope and central design argument

Build the **best practical Rust-owned modular AI execution runtime**, in the combined sense of useful model/task breadth, source/trained math fidelity, low-latency peak/aggregate throughput, physical resource efficiency, security, reliable operations and cost of evolving new model/device technologies.

The differentiator is **reuse of a single fast execution spine**:
- a mathematical op is defined **once**, with effects and a CPU/reference oracle;
- a task defines stage control/output and exposes no particular device/quantization;
- the compiler selects **closed semantically legal regions** and can replace a tensor/layer/model/partition region with a qualified Rust-native kernel, captured graph, externally compiled device program or statically scheduled dataflow plan, while retaining generic fallback;
- device/resource capabilities and async work are explicitly planned and measured;
- a task session has **one logical state publication authority**, independently of physical device workers;
- the service, output delivery, model loader, scheduler and physical budget **are not cloned for every specialization**.

**Scope:** inference-first but **modality- and task-general** execution: text generation, model encoders and rerank, speech recognition/synthesis, image/video iterative generation and multimodal model programs, plus optional future gradient/fine-tuning and agent-app orchestration **above** the trusted compute boundary. “All AI execution” is a *pluggable semantics and TaskDriver goal*, not instant support for arbitrary unknown model files or every optimizer/training algorithm.

**Performance ambition:** compare at equal source checkpoint/quant/semantic mode/hardware/context/batch/power/task quality; specialize where measured better. No architecture alone proves universal supremacy; lower TTFT, P95 and committed output cost must be demonstrated on actual devices.

## 2. Six composable architectural concerns (not six plugin engines)

```text
   Trusted artifacts and typed ModelDefinition         TaskRequest/TaskDriver
                 |                                             |
                 +---------------- SemanticProgram (SIR) -----+
                              math + typed state effects
                                         |
                   RIR pure-region legality / KIR candidates
                                  (transient compiler views)
                                         |
                  CIR physical resource and command lowering
                                         |
                   Immutable compiled ExecutableBundle
                (shape guards, direct kernels, graph islands)
                                         |
              PhysicalResourceArbiter / Device/TopologyProfile
              coarse admission → fast device-local suballocators
                                         |
             TaskCoordinator(session writer) → StepPlan → Workers
                                         |
                  Provisional domain results + readiness
                                         |
                Prepare all → Publish one StateRoot
                                         |
                Typed committed TaskResult/stream + ModeReceipt
```

**Only two enduring compiler representations:** (A) `SemanticProgram` for exact model math/effects, (B) `ExecutableBundle` for physical execution—including direct kernels and external/graph/dataflow program handles—whose command/resource DAG is CIR. `RIR` and `KIR` are useful internal transformations/provider candidates, **not mandatory independent serialized IR ecosystems or runtime interpreters**. This reduces surface, keeps model-load compile bounded and eliminates semantic IR walks in the hot output loop.

**Separate task control from model math.** Use one minimal **trusted Rust TaskDriver** to issue stage intents/loop frontier/output, while repeated per-token/per-timestep mathematical work can be a **device-native StepProgram** without host synchronization. Do not create a generic user-supplied workflow language or embedded privileged agent/tool executor inside inference.

## 3. Correct model math and mutable-state transactions

### 3.1 Model and operator support

ModelDefinition checks trained checkpoint config/tensor roles/position/quant/architecture and emits high-level operators for attention (full/sliding/sparse/latent/compressed), recurrence/delta gates/conv, MoE router/shared experts, n-gram/history, multi-stream residual, modality encoders, acoustic/vocoder/denoise as each is actually defined. Pure local math may lower into lower-level arithmetic/quant/matmul/gather/reduction inside a closed region, but op source semantics and state effects remain provable.

Model names/versions (e.g. Llama, Gemma 4, Qwen3.5+, Qwen3.8 Flash-Next, GLM-5.3 Flash, DeepSeek V4 variants) are **artifact/test targets**, not names of core Rust IR types or kernel schedulers. New checkpoint using known ops adds model definition/fixture; genuinely new math needs operator schema+reference code. A file format parser recognizing tensors is not evidence any unknown op can run.

### 3.2 One atomic logical StateRoot publication

**v4 corrects an important v3 type bug.** Old `StateDomain::commit` should **never** directly mutate visible session data. Each domain computes provisional updates and prepares a version; a TaskCoordinator owns the **single** logical writer for a TaskSession/shard. It checks all required domain versions/events at the same frontier and performs a one-root publication. No CIR `Command::Commit` can independently publish user-visible state; device events signal **ready** or **failed**, not semantic commit.

This is atomic *logical metadata visibility* across KV, recurrence, sparse index, n-gram, RNG/sampler, grammar, adapters, latent/timestep/media state. Physical GPU buffers require private/COW/undo or safe checkpoint replay until publish; rollback may cost more than O(1). Distributed peer state uses epoch fencing/ready agreement/recovery, not an unsubstantiated instantaneous physical transaction.

**Output streaming:** committed root precedes `(session,epoch,output_seq)` output event. Default live transport offers ordered, at-most-once **attempted** delivery, not exactly-once receipt across disconnect/lost ACK. Resumable cursor/idempotency is an optional separate protocol with bounded/durable log and privacy policy. Cancellation cannot leak speculative tokens or partially decoded media state.

### 3.3 Full-state prefix/cache compatibility

A shared prefix can be resumed only when each required state group has compatible checkpoint/schema/position/quant/adapter/media/grammar history at the same legal causal frontier. Multi-domain finalization alignment may force a shorter prefix and suffix recompute. Immutable snapshots + COW support agent forks; active in-flight version/generation leases prevent ABA, wrong lookup or cross-tenant data reuse.

## 4. Specialization and fast-path architecture

### 4.1 Closed-region substitution

Every `RegionCandidate` (including a native `KernelCandidate` implementation) declares exact mathematical region, inputs/outputs/effects, source quant vs compute precision, supported symbolic shape/ragged band, target ISA/compiler/driver, physical buffer layout/scratch/event timing, graph/eager mode and independent quality evidence. Compiler can choose a whole attention layer/entire region without dictating its internal implementation. Fallback remains reference/generic at a proven boundary or canonical state replay.

**Fastest region need not be fastest task.** Performance qualification evaluates full-task TTFT/P95/throughput/energy/physical memory and state correctness, not only microkernel FLOPs. Reassociation, BF8/FP8 transforms and quant arithmetic are explicitly numerically qualified; intentional trained top-k/mask/diffusion schedule changes require explicit changed-model opt-in, never silent optimization.

### 4.2 Shape, graph and hot scheduling discipline

Typed ShapeEnv and runtime guards support bounded dynamic dimensions, ragged batches/route offsets, strides/alignment, prefill vs decode vs verifier vs encoder/denoiser/vocoder bands and per-device ISA. Dynamic expert IDs and sampled tokens can stay device-side; do not compile one variant per token context length or expert ID.

Prepared `ExecutableBundle` and `StepProgram` use persistent device input/parameter slots and direct launches/captured graphs; no SIR/RIR/CIR interpretation per token. A task driver is polled at **stage boundaries**, not necessarily after every mathematical operation or emitted token. GPU-native routing, accepted speculative target results, sampling/grammar where supported, and state continuation avoid CPU synchronization; other exact task/control modes can fall back to honest host steps.

On resource/shape miss, use a **prepared general kernel**; optional background optimization never blocks the current request or silently changes numeric mode. Kernel/search cache bounded by physical/source/shape key and LRU; invalidated by code/driver/op schema changes.

## 5. Coarse resource admission, local physical execution

**Global ResourceArbiter:** conservative OS/cgroup/UMA/VRAM/GTT/SSD and device memory budget with headroom, session/model reservations, fairness and optional evictable caches. It does **not** serialize every kernel temporary alloc. **DeviceBackend:** fast pre-reserved local slab/page/pool and stream/event infrastructure issuing generation-tagged BufferLeases, with asynchronous retire once all readers quiesce. Device workers own GPU/NPU contexts/programs/captured graph handles; TaskCoordinator owns only logical state publication.

Resources include immutable weights, draft/adapters, routed experts, full/sliding KV and GDN/KDA/indexer history, frozen prefix snapshots, model code/graph programs, scratch/quant staging, pinned GPU/NPU BO, encoders/vocoders and media streams, SSD lookup row buffers and collectives. Different cache eviction strategies share **one physical admission ledger**, not one universal cache policy.

**OS safety:** no general guarantee physical memory stays constant after probing. Engine conservatively reserves, watches pressure and fails safely; CPU/GPU/NPU kernel allocation errors are first-class. Virtual GPU addressability is not necessarily physical free RAM. A timed-out external DMA may still touch memory: do not recycle until quiesced or quarantine/reset.

## 6. Heterogeneous device and source-language strategy

**Capability-based physical topology:** CPU SIMD, integrated/discrete GPU, NPU and future accelerator, pinned host/system RAM, storage/remote, copy and event bridges and shared power/firmware domains. `SplitProjectionCompute` may assign independent output rows to GPU+NPU; `AsymmetricExpertOwnership` may assign top-k routed experts across fast/small and slow/large owners. Both are **physical region placement alternatives under the same model semantics**, not second model engines.

A device declared present is not automatically **safe or numerically qualified**. A particular shared SoC NPU/GPU power/fabric behavior may hard hang a host; default unqualified offload OFF, no privileged system clock/BIOS/IOMMU/suspend changes. Rust-authored kernel source remains user preference; XRT/CUDA/ROCm vendor runtime FFI vs foreign kernel language is a separate unresolved product policy. An absent or unsafe NPU cannot block CPU/GPU execution of a supported model.

**Distributed:** peers handshake exact checkpoint/semantic/state schema and attempt epoch, stage provisional results, coordinator fences lost ranks/epochs before publish. No claim of physical distributed transaction magic or end-to-end exactly-once network delivery. Remote resource/collective is optional and should not tax single device.

## 7. Task kinds and optional training compatibility

TaskDrivers for:
- text autoregression, target verified draft and structured outputs;
- embedding/pool and cross-encoder rerank;
- speech recognition, acoustic frames and TTS/vocoder;
- image flow/diffusion generation/edit and video/audio temporal stage pipelines;
- compound trusted **compute** stages;
- **future optional** `TrainStep` with gradient/optimizer/ParameterVersion state, independent of default inference build.

Agent tools/RAG/external workflows are above the permissioned TaskRequest layer, not core GPU commands. A new model's Tensor semantics can be reused in any TaskDriver but requires task-specific validation. “All AI execution” means **extensible supporting mathematics/control/device**, not guaranteed released training/ASR/video on day one.

**Training seam:** ParameterVersion and optimizer state can be a future transactional state domain; running inference sessions pin one immutable model+adapter version and only switch on explicit safe boundary. Activation checkpointing, gradients, distributed optimizer and training durability are deferred until proven needed; do not bloat the inference hot path for them now.

## 8. Minimal Rust modules and release/process strategy

**Keep cohesive ownership boundaries**, splitting packages only when backend toolchain, unsafe boundary, compile profile or feature isolation justifies it. The original one-library/one-binary bootstrap advice was not a permanent crate limit; the accepted workspace layout is recorded in the [baseline](accepted-baseline/TODO.md). Organize internal modules for artifacts+models, semantics+state, task drivers, compiler+kernel registry, device+resource, runtime/coordinator, optional service/validation. Avoid mandatory dynamic plugins, giant abstract factories, generalized agent workflows or per-model servers. Compile-time feature flags isolate unavailable CUDA/ROCm/XRT drivers while CPU tests run on ordinary hosts.

**Future developer test:** independently add one new stateful op/model, another kernel device, an unusual quant, a non-token TaskDriver and an alternate IO tier. Record unrelated files/modules touched, compile time and hot-path overhead. If the change duplicates scheduler/model loader/allocator or requires broad type switches throughout unrelated code, **the architecture fails maintainability**.

**Focus implementation in a vertical slice:** one real source-verified model and TaskDriver from artifact through SIR/reference, legal region, KIR candidate, compiled CIR plan, CPU/one GPU backend and state/output publication; forcibly disable fast kernel and verify general fallback. Pair with mock three-domain state transactions, dynamic shape/async event lease and one non-token synthetic task. Only then expand optimized MoE, exotic models, experimental NPU, speech/media, speculative tuning and distributed breadth.

## 9. Quality and performance governance

Separate (1) semantic formula/state, (2) reference-to-kernel numerical tolerance/precision, (3) checkpoint-level task quality, (4) stochastic target sampling/distribution and valid committed output. A limited generated-string comparison is not a full quality proof; cross-backend bit-identical output isn't universal unless contract/test says so.

Benchmark relevant device/task:
- kernel ISA/register/occupancy, bytes/time and exact quality;
- full program prefill/AR target t/s, TTFT, P50/P95/P99 under native C1/2/4/8/16/32+ work;
- task-specific units (ASR WER/timestamps, TTS audio, image/video sampling latency/quality, embeddings/rerank, future gradients);
- real memory/IO/cache/quant/prepacked weights/prefix reuse and energy;
- source checkpoint/quant/driver/compiler/fabric/power and before/after methods, with raw evidence;
- physical hardware safety, failure injection, source/license rights and mode receipt honesty.

Optimization can be promoted only inside measured capability cell and safety/quality constraints. A no-gain policy should be disabled; no universal "fastest on all models" assertion. Publish known losses/skips to avoid overfitting benchmarks.

## 10. Staged delivery and source/research separation

**Stable phases ZG-P00–P23** are maintained; v4 review introduces additional **S0-V4** protocol and minimality proof gates and new **H32–H43** falsifiable experiments (see integrated program). Their presence is **planning**, not implementation completion.

**Prior art:** retained source research records external URLs/revisions, licenses, benchmark author claims and limitations; superseded v2/v3 document bodies are recoverable from Git, not mandatory live archives; core architecture and type/module names are **original project-neutral contracts**. Verified MIT source may be considered only after file-level notice/weight/device code licensing and Rust ownership review. Non-MIT projects remain attributable research; no copied source without explicit approval. The external research corpus remains read-only.

**Developer operating rule:** start with [the plan index](TODO.md), its relevant TODO and linked contracts. Use the decision register and this master for design context, inspect live source/tests, and record implementation and verification under actual commits using [the TODO rules](TODO-SPEC.md). No generated handoff, raw-evidence archive or SHA inventory is a delivery requirement.

## 11. Success criterion and practical limitations

**Correct model mathematics and full mutable state + a tiny effective runtime that can execute independently authored high-performance specializations across tasks and devices without multiplying engines.**

If v4 cannot (A) avoid host synchronization on supported device-native steps, (B) guard dynamic shape/precision correctly, (C) publish all-state roots atomically, (D) keep per-device allocations out of a global lock, and (E) add a second model/device/TaskDriver without unrelated rewrites, then we must **simplify or revise the interface before building breadth**.

This master states the **design ambition**, not a verified fastest-runtime claim. The accepted implementation is narrower; outstanding extension, quality and physical matched-benchmark gates remain in the current plans.

## 12. v4.1 — universal region execution across GPU, TPU and static dataflow fabrics

**Latest hardware-composability request U-019:** account for local coherent Blackwell SoCs, discrete/rack-scale FP4/HBM multi-GPUs, compiler-centric TPU/ASIC graph devices, wafer-scale/static dataflow accelerators, specialized sparse/model execution systems, and future device generations. Preserve the minimal **one semantic program → one compiled ExecutableBundle → one TaskCoordinator/StateRoot** core. Do not require a new model runtime per hardware vendor.

### 12.1 The decisive abstraction is *minimum efficient execution granularity*

Old `KernelProvider::compile(one kernel)` models ordinary CPU/GPU dispatch and some NPUs. **It does not by itself model an accelerator whose natural submitted object is a compiled whole model graph, pipelined static fabric program or wafer-scale weight stream.** Generalize it to a single **`RegionExecutionProvider`** whose candidate may cover:
- **NativeKernel:** one operation or fused Rust-authored region;
- **CapturedCommandGraph:** pre-recorded direct GPU/CPU/NPU commands;
- **AheadOfTimeDeviceGraph:** graph compiler lowering a verified SIR subgraph/model stage to an executable program;
- **StaticallyScheduledDataflow:** a large placed accelerator/fabric program with compile-time communication and explicit weight residency/streaming;
- **ManagedDeviceSession:** only when device SDK exposes enough source math, provisional state, snapshot/fork/replay, epoch and numeric/control semantics to qualify as actual lowerable work. A vendor serving endpoint with opaque text output belongs to a **separate TaskServiceAdapter**, not a fake native backend.

This is **one candidate/capability interface, not five new engines**. Native `KernelProvider` is its simplest implementation form, not an independently maintained runtime. No source model definition or TaskDriver should mention a hardware vendor. The provider's micro-kernel/program schedule belongs to the device/backend compiler, while the Rust coordinator owns task admission, logical StateRoot publication, output and resource quota.

### 12.2 Platform categories this represents

| Hardware category | Major physical distinction | Candidate mode and proof |
|---|---|---|
| Local coherent CPU+GPU SoCs (including small Arm/Blackwell and x86/RDNA hosts) | LPDDR shared physical RAM and CPU↔GPU coherence, limited bandwidth/OS headroom | Native/Captured graph with shared-memory lease, actual safe OS reserve and Arm64 or x86 ISA |
| Discrete GPU workstation and datacenter HBM GPU | external VRAM/HBM, tensor low-bit formats, PCIe and native graph execution | Rust native precision-specific kernels and capture, exact source quant/matrix scale proof |
| Multi-GPU rack fabric | peer-to-peer, switch/interconnect/collectives, sharded HBM | Partitioned graph/collective with actual topology costs, epoch fencing and all-domain state |
| Compiler-centric cloud TPU/accelerator | requires ahead-of-time graph lowering and often static/bounded dynamic shapes | AOTGraph, graph program lifecycle, shape bands, persistent device state bridge and full model verification |
| Wafer-scale and deterministic dataflow ASIC | static on-chip SRAM/core placement, stream weights/data and device-owned schedule | StaticDataflow, large closed stage/model regions, compiler source rights, controlled program load/weight streams |
| Heterogeneous GPU + NPU / GPU + specialized processor | different compute owners, memory formats, power/fabric and link bandwidth | one CIR graph with device-owned subprograms and explicit first consumer/IO readiness, no unsafe auto offload |
| Managed external model service | no mathematical intermediate program or trusted KV/RNG state API | **External TaskServiceAdapter** with independently declared model/session/privacy semantics, never native backend equivalence |

Examples and verified first-party URLs—**NVIDIA DGX Spark/GB10; Blackwell RTX, B200/B300, GB200/GB300; Google TPU v6e/v7x Ironwood; Cerebras CS-3/WSE-3; AWS Trainium/Inferentia; Groq LPU; AMD Instinct CDNA; Intel Gaudi; Tenstorrent; SambaNova and others**—are catalogued exclusively as source-attributed **prior art**, not mandatory architecture types: `../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md`. The user said “cerberus”; this is likely Cerebras but a distinct Cerberus parallel-decoding algorithm also exists, which fits the existing `DraftSource/TargetVerifier` seam and requires algorithm-specific quality proof.

### 12.3 New required capability detail

`RegionExecutionProvider` must expose region scope, op/shape/quant/numeric/fabric support, **Compile→Load/Upload→Warm→Submit→Ready/Fail→Retire**, first-use memory and links, **who controls internal micro-scheduling**, optional device loops, collectives and parameter/weight residency. The compiled program may be costly to rebuild or warm: retain bounded graph/program cache and shape variants, provide cold portable fallback, never JIT a whole TPU model on every generated token.

**State bridge is a hard gate:** a graph/wafer provider with hidden in-place KV/recurrence state that cannot fork/prepare/snapshot/recompute is not automatically safe for speculative decoding, cancellation or hybrid prefix caching. Stateless encoders may be eligible earlier; stateful program must return **private provisional state + real ready events**, while TaskCoordinator alone publishes the new StateRoot. External hosting/serving API is not a substitute for that contract.

**Model precision:** Blackwell NVFP4 (16-value E4M3 microblocks plus per-tensor scale), MXFP4 (different block scaling) and CPU/TPU/AMD custom formats are *different stored/compute/physical encodings*. No silent quant downgrade or inferred compatibility. Physical topology includes coherent UMA vs HBM/VRAM, on-chip SRAM/wafer weight streams, rack interconnect, cloud access control and egress cost.

### 12.4 Rust-first and feasibility truth

The **semantic/task/resource/worker engine remains Rust-owned**. Vendor device compilers/runtime/firmware and proprietary graph IR lowering may require external binaries/FFI not yet approved by the user. A Rust host adapter that submits a vendor-compiled graph does **not** imply we authored that vendor's device compiler or kernel in Rust; classify support transparently as `architecture-compatible`, `SDK/toolchain-feasible`, `compiles`, `functional`, `state-safe`, `performance-qualified`, or `production-qualified` **for each model/task/quant/shape/device**.

Never claim TPU, wafer-scale, Groq/LPU, Trainium or other ASICs actually run zerogravity merely because the abstraction can describe their program. Provider SDK access, source-license policy, numerical oracle, device/state failure behavior and physical hardware are independent qualification steps.

### 12.5 Proof program and scope

New optional **S0-FABRIC-A…H** fake graph/dataflow backend tests and **H44…H55** extension experiments cover graph program lifecycle, topology, state snapshots/refusal, external service misclassification, per-quant source correctness, cold compile/shape bands, rack collectives and source+vendor compiler policy. These **do not delay CPU/GPU Stage-0 vertical slice**. Existing P00–P23 unchanged; P01/P05/P07/P08/P10/P16/P19/P20/P21/P22/P23 now explicitly carry these backend family acceptance gates in the roadmap.

**Reference:** `../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md` (normative, precise execution model), `../prior-art/ACCELERATOR_PRIOR_ART_AND_PLATFORM_RESEARCH_2026-10-12.md` (source-specific, not normative). Existing v4 state, TaskDriver, shape guard, CIR and physical allocator contracts remain authoritative.

**Success test:** a second backend that requires a **whole static compiled model graph** and one stateless **streamed-weight dataflow program** can run source-verified regions through the same TaskCoordinator, StateRoot/lease/admission and TaskDriver, without forking source model math or service. A device lacking a compiler or suitable state bridge stays explicitly unsupported rather than being falsely represented as “seamless.”


## 13. v4.2 — One distributed AI execution fabric across hardware, machines and tunnels

**Latest user requirement U-020** extends the project beyond peak single-device runtime: *independently place mathematical execution, model layers, their parameters, live KV/recurrent/indexer state, reusable prefix caches and task stages across heterogeneous local and networked machines, including explicitly authorized tunnels*. Prefer the best topology for **each checkpoint/quant/task/context/concurrency/physical network**, while retaining a truly minimal one-machine fast path. No model-specific distributed inference engines.

**New definitive references:** [Distributed placement and network execution](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) (full architecture and performance/correctness), [remote peer transport and state handoff](../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md) (trusted tunnel, leases and failure protocol), [distributed subsystem entrypoint](../docs/DISTRIBUTED_PARALLELISM.md), and [distributed prior-art source research](../prior-art/DISTRIBUTED_PRIOR_ART_AND_TUNNEL_RESEARCH_2026-10-12.md) (attributed references only). The placement and peer contracts supersede earlier rank-local commit sketches: peers prepare and report actual readiness; only the session coordinator publishes the logical root. Earlier text remains available in Git, not as an independent live specification.

### 13.1 Four independent placement axes and one derived transfer schedule

- **ComputePlacement:** direct kernel, full graph, layer groups, core attention, FFN/MoE, tensor/head, token-range attention shards and TaskDriver/media stages; legal closed SIR regions only.
- **ParameterPlacement:** immutable model weight tensor shards/replicas, sparse experts, adapters, program images and external model/lookup sources, independently of compute location.
- **StatePlacement:** each session/layer/state-domain mutable KV, recurrence GDN/KDA, sparse indexer, n-gram/RNG/grammar/sampler and modality/iterative state has one versioned authoritative physical state-shard owner under a logical TaskSession root.
- **CachePlacement:** immutable shareable prefix/checkpoint/encoder-feature caches on GPU/CPU/SSD/other machines, with validated content hash/schema/tenant and locality directory **hints only**.
- **TransferPlan:** physical movement of activations, state snapshots, compact attention/expert partials, model weights and collectives, annotated with actual link RTT/P95 throughput, encryption/transport, bandwidth, buffer leases, source rights and ready events.

A *single* `PlacementPlan` is compiled from these into the existing `ExecutableBundle`/CIR **within the current topology and placement epoch**. No third durable IR, new service kernel runtime or different mathematical ModelDefinition. A `RegionExecutionProvider` may own a whole multi-machine/static graph program, with model/task/root publication remaining ours.

### 13.2 The placement topologies that must be representable

| Placement | Core execution pattern | When considered |
|---|---|---|
| **Colocated baseline** | all model layers and live state on one device/host | default latency/perf comparison, no cluster overhead |
| **Layer+state colocated per peer** | model layer groups on machines A/B/C, each holds their own layer KV/SSM; only activation boundaries sent | run larger-than-one-host model, LAN/tunnel capacity mode |
| **Separate layer compute and cache storage** | full compatible prefix hydrate once or run operator near cache with query/activation exchange | remote cache useful but full KV transfer per token expensive |
| **Attention/KV vs FFN/experts split** | stateful attention executed with cache on bandwidth/capacity-rich nodes, FFN/MoE on compute-rich nodes | long context/batches and fast links where frequent layer crossings worthwhile |
| **Sequence attention cache sharding** | split context by token ranges, send query to state owners, combine compact stable softmax summaries | very long context on qualified ordinary softmax operators and faster fabric |
| **P/D split** | prefill vs decode machines/pools, full hybrid state handoff at committed frontier | protect TTFT/ITL via phase separation when state transfer affordable |
| **Tensor/head or expert sharding** | collectives/partial results across local/rack ranks | low latency/high bandwidth and acceptable numeric reductions |
| **Remote immutable cache and model source** | share prefix or model tensor blobs across enrolled peers; hydrate/recompute | avoid recomputing long repeated prompts/duplicating parameters |
| **Task-stage partition** | audio encoder, text decode, vocoder, denoiser/decoder on different workers | specialize task workloads when transferring outputs is cheaper than moving parameters |
| **Heterogeneous local + tunneled network** | trusted CPU/GPU/NPU or AOT graph workers on local and remote private peer group | capacity or throughput subject to real RTT, data permissions and partial failure |

These are **eligible plan candidates**, not simultaneous mandatory modes; mixes are legal only if the exact state-domain partition and real network resource/delay budget permit them. A “cache on machine B, attention on A” topology is *not* semantically/physically equivalent to “attention + KV both on B”; compute beside KV can replace full remote KV reads with compact query and result, while a remote prefix cache may be best **hydrated once**.

### 13.3 Topology transport and trusted private tunnels

**One optional PeerAgent capability** reuses the same binary/library, RegionExecutionProvider and local ResourceArbiter to advertise verified devices/resources and execute a scoped closed region, not another chat server. Enroll peers explicitly. Control/metadata can travel through an authorized encrypted private tunnel, while the data plane chooses an available trusted direct interconnect, RDMA or authenticated reliable stream. **A tunnel cannot turn TCP/QUIC into GPUDirect or make relayed WAN latency disappear.** Link capabilities must distinguish direct UDP, peer relay and remote relay, loss/MTU, effective bandwidth/P95 RTT, copy/crypto overhead, egress and link permission.

Model weights, private prompts, hidden activations, KV and modalities cannot be sent to unknown peers merely because they advertise free GPU memory. Remote code/agent shell execution is not part of joining the cluster. Authenticate/authorize node, tenant, model and each sensitive data class; remote memory keys and state-owner leases are scoped/fenced. Start with a **small manually enrolled cluster**; peer auto-discovery, leader election, distributed metadata DB and autoscaling only if product measurements justify them.

### 13.4 One logical StateRoot across remote state shards

A SessionEpoch has exactly **one TaskCoordinator writer**. `StateRoot` can contain remote state-shard references, but every remote worker returns **Prepared/Ready/Failed** with buffer leases, StateShardGeneration and matching placement epoch; the coordinator **alone** validates/publishes the new logical root and output sequence. Device worker or peer does not independently publish a sample/new token. Losing a rank during prefill/collective aborts or safely replays from last compatible committed snapshot; a speculative branch cannot leave rejected KV/GDN/indexer/RNG state on any peer.

**Replanning at committed task frontier only:** reserve/warm new compute owner, copy/verify compatible state version, fence old writers/in-flight DMA, publish new placement/ownership epoch in a single root update; hold/quarantine invalid memory until physical completion. Peer departure or network partition cannot create two independent session writers. Without a durable lease/quorum/checkpoint protocol, **fail-stop rather than promise automatic transparent failover**. User output delivery remains ordered attempt with optional authenticated cursor replay; no claim of network exactly-once receipt.

Immutable remote prefix and model cache directories can be eventually updated and stale; they are revalidated on use and never override mutable state authority.

### 13.5 Topology-aware mathematical and cost legality

- Source SIR effects decide which attention, recurrent, latent/sparse indexer and media states can move or be partitioned.
- **Softmax attention token-range sharding** is algebraically composable using local logit max/denominator/weighted-value numerator and numerically stable global rescale. Correct positional/mask/routing schema mandatory. Unknown compressed, global selected-top-k and recurrent GDN/KDA do **not** automatically inherit this partition; add an operator-specific `PartitionAlgebra` with reference proof first.
- **Attention–FFN split** can reduce compute/memory coupling but adds per-layer network RTT. On a relayed tunnel it can easily be much slower than a colocated layer group. More nodes can improve *model capacity*, not necessarily C1 decode speed.
- A costed placement uses actual device compute and resident bytes, per-boundary activation/state/collective size, measured direct/relay link throughput/P95, phase/shape/cold warmup, batching and fairness, privacy and failure recovery. Candidate selection optimizes the **whole critical path of valid committed task outputs**, with capacity/energy modes separate from latency/throughput.
- Remote full KV reads per token across ordinary WAN/tunnel should normally be rejected; prefer local state migration, remote attention beside state, compact partial attention, recompute or simpler pipeline. Fast RDMA must be physically verified, never inferred from a VPN interface.

### 13.6 Delivery and proof without multiplying engine complexity

**Do not block P00–P08 local CPU/GPU vertical slice.** Expand existing P19 distributed phase incrementally:
1. simulated 2–3 peer layer placement with per-layer state, correctness/failure;
2. actual trusted two-node layer split via bounded encrypted stream with exact model/quant and slow-link tests;
3. remote prefix and full hybrid state handoff, including P/D and cache hit;
4. independent attention/FFN ownership and sequence-sharded KV where qualified fast link and operator math justify;
5. automatic measured placement across mixed devices and tunnel topologies, optional rack compiled collectives, graceful rebalancing and larger cluster management.

Optional **S0-DIST-A…L** design/transport/failure gates and **H56…H70** research experiments now track these capabilities; all are **PLANNED, not run**. Existing stable ZG-P00…P23 stay unchanged. Higher-performance / production claims require actual source-verified checkpoint, task, network direct/relay profile, numerics/full state, P95, physical memory and recovery evidence.

**Architecture acceptance:** adding a new private peer, transport link, remote state cache provider or layer/attention placement requires only `NodeCapabilities`, scoped `PeerTransport`, `PlacementPlan`/state-domain partition capability and actual measured tests; must **not** require cloning ModelDefinition, TaskDriver, TaskCoordinator, any API server or global allocator.

**This becomes a core zerogravity capability goal alongside peak single-device performance, not a side project or external model-serving integration.**


## 14. v4.3 — one-command local UX, built-in multi-model router and OpenAI-compatible serving

**Latest explicit user requirement U-021:** zerogravity must be both an extremely capable **hardware/model-specialized execution system** and a **joyfully simple local model tool**, with a built-in router/load balancer that can manage and serve many different model families and per-model settings through **one OpenAI-compatible API**. From a user's viewpoint, one `run model` command should work with sane defaults; every choice can be inspected and overridden safely when needed.

**Authoritative design files:** [model router and residency](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md) and [zero-config CLI/config/API](../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md). Names, source project implementations and current API features remain in [dated prior-art research](../prior-art/MULTI_MODEL_ROUTER_AND_API_PRIOR_ART_RESEARCH_2026-10-12.md) (non-normative).

### 14.1 One engine across CLI, SDK, API and many models

```text
 CLI run / Rust SDK / API compatibility / native task endpoints
                         |
                 Auth + TaskRequest
                         |
     ModelCatalog + versioned exact aliases & ModelProfiles
                         |
               ModelRouter (request admission)
                         |
          ModelResidencyManager (load/warm/evict)
                         |
       existing ResourceArbiter + PlacementPlan selector
                         |
             TaskCoordinator / TaskDriver
                         |
             ExecutableBundle / DeviceWorkers
                         |
                Prepared all-domain state
                         |
                PublishRoot once + stream
```

**No extra inference server**: ModelRouter does not run its own token sampler, device allocator or task-output stream protocol. It resolves and admits work at **request/model/session/phase boundaries**, while the existing TaskCoordinator batches and executes directly. A single model bound to in-process `run` has no mandatory HTTP/daemon/network hop; `serve` exposes the same engine plus API adapters. Optional trusted same-user local socket reuse is a convenience, not a required service or automatic root/systemd change.

### 14.2 Four entities must remain distinct

1. **RegisteredModel:** immutable content-addressed artifact/checkpoint/quant/semantic model and allowed tasks. A catalog can describe thousands without allocating their weights.
2. **ModelAlias + ModelProfile:** human-friendly versioned name, role and safe per-model settings (context, sampling, planned device/perf/cache/resident policy). Alias revision changes affect **new requests only**.
3. **ResidentModel / Replica:** actually loaded/warmed physical checkpoint/adapter/program under resource reservation. **One sharded deployment across several GPUs/machines is one replica; multiple independent deployments of an equivalent logical model are multiple replicas**; shared base weights/adapters may be physically deduplicated where safe.
4. **TaskSession:** user/task-specific mutable KV/SSM/indexer/RNG/grammar/media state, pinned source/quant/adapter and current StateRoot; replica/session affinity and future migration governed by identical v4.2 StateCompatKey/PlacementEpoch rules.

Many models can be **registered**, while simultaneous resident/active models/replicas are limited by actual physical RAM/HBM/UMA, model/adapter weights, compiled code, scratch, context/state, device queues, remote reservations and OS margin. On-demand loading, warming, idle eviction and configurable keepalive/pins/TTL happen under the same ResourceArbiter—**no unbounded resource claim**.

### 14.3 Exact-model routing versus model choice

**Default exact:** `run coding` or an OpenAI `model:"coding"` resolves deterministic alias/revision to one **precise checkpoint/quant/adapter/TaskDriver** and may load-balance among semantically/numerically compatible replicas based on queue, p95, cache affinity, device/remote safety and tenant policy. Stateful sessions remain sticky or migrate only after verified full StateRoot handoff at committed frontier. A router must never silently answer using another language model or lower quant because an exact model is busy.

**Explicit selector:** a configured `auto` alias or named pool may choose a different target source by an authorized policy, but must report the actual selected model and meet task/privacy/quality conditions. This is **not equivalent** to replica balancing and is not default for exact models.

**Multi-model TaskProgram:** target+draft, encoder+language model, embedding+reranker or TTS+vocoder can load several separately pinned artifacts under one coordinated task and budget. They are **not replicas**. Each may be placed locally or on trusted remote workers through v4.2, with source rights and full-state effects preserved.

### 14.4 Resident lifecycle and model switching

A prepared model moves through `registered→verified→admitting→loading→warming→ready→busy/idle/draining→unloaded`, with typed failed/unsupported states. Deduplicate concurrent loads of same binding, reserve before GPU allocations, continue serving other warm models while cold load runs, bound waiter queues and timeouts and avoid unload/reload thrashing. **Never evict active unsnapshotable KV or in-flight GPU/DMA.** Allow optional keepalive, preload/pin, idle scale-to-zero and safe lease-drained model unload, with actual physical capacity authoritative.

A hot alias/config update publishes a new immutable revision for future routes; old sessions remain on old source/template/adapter/plan until completion or explicitly legal migration. A source artifact deletion cannot destroy active session resources; model store GC respects references. A model cannot silently swap to a different source due to alias reuse mid-generation.

### 14.5 Configuration without upfront complexity

**Level 0:** `zerogravity run <model>`, `zerogravity serve` (illustrative final spelling). No mandatory daemon, config file, external service, account/cloud or manual hardware flags for local supported file.

**Advanced:** typed human-editable global config, per-model profile, alias/deployment policy, per-session and allowlisted request overrides, physical resources/graph/kernel/quant, sampling/grammar, model pin/warm/eviction, authorized multi-machine sharding, routing strategy/replicas and API/TLS/auth. **Hard security/semantic/memory constraints always outrank overrides.**

Users can inspect **effective values and why** via `inspect --explain-plan` and `config show --effective --model <name>`, including model source/quant, profile revision, resident replicas, task type, device/state/cache placement and performance evidence. No silent quant downgrade or unknown model pulls; remote fetch only with explicit trusted source/license/size/authorization. Proposed config is small versioned TOML; exact keys/path/port are open user/implementation decisions.

### 14.6 APIs, security and complete streaming semantics

Recommended first interoperable subset: **`GET /v1/models`**, **`POST /v1/chat/completions`** with correct model-template rendering, sampler, SSE deltas/finish, usage/tool-call **generation** and **`POST /v1/embeddings`** when real encoder qualified. Then qualified completions and `POST /v1/responses` with a **separately versioned stateless subset** first, stateful response storage/previous-response and typed Responses stream events only when genuine StateRoot/session retention and conformance exist. No claim every cloud built-in tool/search feature works; tools are only output data until an external authorized agent/workflow adapter executes them.

**Default local-only:** loopback bind, no auth required solely for same-user local trusted CLI unless configured; external bind requires explicit authentication/TLS and separate restricted native admin model-management API. Tenant/allowed model aliases and peer permissions do not transfer from any arbitrary OpenAI API token. Private source/model weights/media/prefix/remote state and log data stay protected. No privileged driver tuning or public remote server implied by `run`.

**Compatibility is a tested matrix** (endpoint × task/model × field × streaming × SDK version × numeric/usage behavior), not a boolean. Unsupported features return typed errors, not HTTP 200 with silently changed semantics.

### 14.7 Performance and long-term minimality

Router selection uses warm replica cost, physical model residency, cache locality, per-task context/StateRoot, queue fairness/SLO, peer/hardware/authorization and cold compilation/startup. **No router in per-token GPU hot loop, no per-output HTTP proxy tax, no global allocation mutex per kernel**. Multiple models have distinct queues and fairness; within a model, already existing native batching handles compatible sessions. Model cold load, on-demand eviction and warm pins are coarse physical-budget decisions, with measured P95/TTFT/cold startup/energy and safe backpressure.

**Proof:** one zero-config in-process model, one shared multi-model service, a cold+warm model concurrent load, safe idle evict without StateRoot corruption, two aliases of same source, equivalent replica balancing with sticky state, one explicit model-choice alias, model/task multi-role pipeline and correct official OpenAI SDK nonstream/stream tests. Additional S0-SERVE-A…J and H71…H86 are optional acceptance/evidence tracks; all NOT RUN. ZG-P00…P23 remain unchanged.

**Non-goals:** copying an existing inference gateway, starting independent model servers for every loaded artifact, automatically downloading unknown models, silently switching quant/source on overload, pretending all models fit simultaneously, running arbitrary tool actions inside an API handler, or implementing every provider's proprietary cloud-only Responses behavior.

**Final UX success:** users see **one command and one endpoint**; expert developers see composable and controllable model profiles, replica placements, multi-hardware routing, real source fidelity, full model-state safety and transparent performance. Both experiences run the same original zerogravity engine.


## 15. Final editorial and implementation authority (publication dossier 0.1, U-022)

**The public project documents are now structured by audience rather than chronological appended notes.** Source architecture remains v4.3; the October 12, 2026 publication dossier adds *no new runtime subsystem or experimental result*. The user asked for a complete whitepaper manuscript and full agent-deliverable design specification.

**Start here:**
- `../../docs/WHITEPAPER.md`: externally readable manuscript, attributed prior art and falsifiable hypotheses, **design only**.
- `../../docs/ENGINEERING_SPECIFICATION.md`: integrated normative implementation acceptance **ES-00…ES-15**, spanning precise source semantics, region execution, all-domain prepared state, hierarchical resources, distributed placement, multi-model router, CLI/API, source/security and implementation tests.
- `../../docs/REFERENCES.md`: numbered academic and primary source citations, no presumed code-copy permission.
- `../../docs/README.md`: documentation authority/navigation map, separating active contracts from non-normative historical/source reports.
- [AGENTS.md](../../AGENTS.md) and [TODO-SPEC.md](TODO-SPEC.md): bounded implementation, appropriate source/test/hardware/security gates and commit-grouped TODOs. Historical playbook templates impose no extra tracking duties.
- `REQUIREMENTS_AND_TEST_TRACEABILITY.md`: explicit U-001…U-022 user intents mapped to ES/INV/phase/test obligations.
- `../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md`: claims, author/license/public release signoff and future physical benchmark/reproducibility.
- `FINAL_ARCHITECTURE_REVIEW_2026-10-12.md`: final editorial/technical consistency and residual risk ledger.

**Current architecture invariants do not change:** one SemanticProgram and compiled ExecutableBundle; one trusted TaskDriver; one RegionExecutionProvider for native/graph/static programs; one physical admission authority with device-local pools; full StateDomains Prepare and one per-session TaskCoordinator PublishRoot; optional costed separate compute/parameter/live-state/immutable-cache distributed placements and authenticated peer transport; one exact-model catalog/router/residency layer and same-engine in-process CLI/loopback HTTP API.

**Evolve the working Rust model/task execution spine without building a framework for every research paper.** P00…P23 remain coverage references, not a blanket NOT_STARTED status: completed slices and genuine unfinished work are tracked in [the active plans](TODO.md). Stage0/Experiment/Release documents supply applicable acceptance criteria, not a demand to rerun completed historical phases. Design type sketches do not override tested source APIs.

**Publication policy:** manuscript is eligible for external *technical design review* but not experimental-results publication; named author/byline, original code/document license, stable publisher/DOI, independent reviewer and final public release must be approved separately (O-067…O-071). No PDF was generated or source/model/hardware/network action taken during this documentation pass. Any future performance edition must bring raw matched checkpoint/model/task/device/network data and negative evidence.

**Agent delivery success:** a new agent can identify exact authority, select one bounded phase, implement source-verified math and a compiled legal region with safe state/resources and fallback, prove a real task result and report exact tests/blocked issues without rereading a previous chat. A source-project-branded interface or cloned model runtime is evidence of architectural regression, not of innovation.


````

## Source: .agents/plans/archive/history/AGENT_IMPLEMENTATION_PLAYBOOK.md

Original UTF-8 bytes: 18003; SHA256 `bc7233914a8b22ba5a64709efd024254f772302465c5b782b6fa3ff5db8c97a7`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — implementation design reference

Read [the plan TODO index](TODO.md) for current work and
[AGENTS.md](../../AGENTS.md) for the complete work-tracking process. This file
retains phase/invariant guidance; its historical ownership sketches are design
references, not another handoff, tracker or mandatory document-update list.

## 2. Begin with a bounded phase contract

For a new agent task, record a small **work ticket** before changes:

```text
Ticket:
  phase / requirements IDs:
  user authorization and scope:
  exact current source revision / dirtiness:
  owner modules and files:
  upstream model/source/quant or device capabilities:
  expected new interface and invariants:
  portability/reference fallback:
  positive tests and deliberate negative tests:
  source permissions and physical safety:
  measurable performance hypothesis, if any:
  blocked policy/hardware requirements:
  rollout/status and handoff artifacts:
```

Do not spend weeks scaffolding all proposed backends, every TaskKind, a generic plugin ABI or all model families before a **real one-model CPU reference** passes. The project wants **a small number of coherent modules with strong tests**, not a sprawling architecture visualization implemented as empty traits.

## 3. Minimal Rust module ownership (first release, not frozen public ABI)

A healthy starting structure is one library crate and one CLI binary (or a second GPU/toolchain crate only if actual build isolation requires). These module names are illustrative:

| Module | Owns | Must NOT own |
|---|---|---|
| `artifact` | bounded GGUF/safetensors source/manifest/quant/tokenizer/token mapping | GPU execution or HTTP routing |
| `semantic` | registered exact SemanticOps, state effects, reference evaluator and model recipes | backend compiler choice or session mutability |
| `task` | trusted TaskDrivers, typed input/output frontiers and stage control | arbitrary external tool execution or driver memory |
| `compile` | legal region transformations, shape guards, RegionExecutionProvider selection and bundle metadata | HTTP, active session root or global model registry |
| `device` | CPU/GPU/NPU optional compiled handles, streams/events and local fast memory pools | TaskSession state publication |
| `resource` | global physical admission/reservations/pressure and typed resource lease policy | per-kernel global mutex |
| `runtime` | TaskCoordinator/StateRoot prepare and publish, request batching, StepPlan/WorkerReply | mutable checkpoint source or second model-specific server |
| `catalog` | immutable source and alias/profile registry + resident model lifecycle | per-token sampling/math |
| `router` | exact source/replica selection, queue/admission policy and state/cache affinity | its own device workers/sampler |
| `service` | CLI/Rust SDK and optional HTTP/SSE/API/admin adapters | model math or external tool execution |
| `network` *optional* | peer identity, topology, TransportProvider and RemoteRegionWorker messages | new semantic IR or second state writer |
| `validation` *test-only* | golden vectors, compatibility, property/fault tests and benchmark evidence | shipped driver/runtime dependency in CPU-only build |

Avoid literal per-module crates before compiler and source tests justify it. Types may live together to avoid circular dependencies. Make code review focus on **ownership** rather than the number of modules created.

## 4. Implementation sequence, Phase P00–P23

Every phase describes intended work, **not a completion claim**. The authoritative [Roadmap](IMPLEMENTATION_ROADMAP.md) has expanded tasks; this playbook states *first deliverable, required test and safe skip* per phase.

### P00 — reproducible project foundation

**Deliver:** minimal Cargo CLI binary (optional library added only with first actual engine), pinned toolchain/Cargo.lock, CPU-only build, `rustfmt/clippy/test` gates, dev-shell/Nix compatibility, test fixture/data rights policy and source/provenance ledger. **The packaging subset now exists and is locally tested (U-023); P00 remains PARTIAL until Nix release/module verification and remaining source/rights gates. Do not misclassify `--help` as an inference runtime.** Probe actual CPU/OS/physical memory **without altering drivers or root services**.

**Pass:** clean checkout builds and tests without GPU/NPU SDK, no foreign inference engine linked, documentation status updated honestly. **Do not:** create 12 empty crates or auto-download a model.

### P01 — physical accelerator source/codegen feasibility

**Deliver:** tiny Rust-authored native GPU source kernel on actually available hardware for representative GEMV/GEMM/quant/recurrent update shapes; record driver/compiler ISA and golden CPU reference.

**Pass:** actual GPU output/logits for tested kernel are correct; register/occupancy/kernel timing reported with source revision. If driver/compiler/hardware inaccessible mark **BLOCKED** and continue portable CPU reference. Additional AOT/wafer/NPU SDKs are optional, not gatekeeping CPU.

### P02 — reference operator/state truth

**Deliver:** exact math for the selected model's attention/projection/position/sampler and one effectful state domain, CPU independent oracle. Extend to reference recurrent/sparse as models demand.

**Pass:** source model tensor/vector/logit tests, branch/rollback and invalid state effect-order tests. **Do not:** substitute plausible but different attention or reduce routed top-k for speed.

### P03 — verified artifact, quant and tokenizer

**Deliver:** safe bounded real model source parser and tokenizer/template, tensor manifest and source SHA, relevant one quant codec. Explicit file limits and allocation overflow checks; no custom model Python execution.

**Pass:** official token pretypes/chat template, exact tensor dims/quant dequant/source vectors and malformed corpus; unsupported type returns typed error.

### P04 — effect-aware SemanticProgram and trusted TaskDriver

**Deliver:** minimal typed values/shape/effects and one text-generation TaskDriver, plus fake second StateDomain. The reference can execute a simple semantic program.

**Pass:** illegal effect reordering rejected; task loop/frontier valid; a second synthetic non-token task can be represented without fake target sampling. **Do not:** implement a generic agent workflow DSL.

### P05 — region candidates, dynamic guards and async compiled plan

**Deliver:** one legal optimized region, an explicit portable fallback, ShapeEnv/stride/ragged checks, prepared native kernel candidate and typed resource/event command DAG within ExecutableBundle.

**Pass:** legal optimized vs reference output/state, illegal shape rejected before launch, mock invalid lease/event/order tests, **no IR walk per token** in compiled path.

### P06 — useful CPU-only model executor

**Deliver:** source-verified small real model full forward/prefill/decode, CPU quant kernels and sampler, TaskSession state and simple CLI through same TaskCoordinator.

**Pass:** meaningful reference-checked user output/stop/usage, source quant/dtype parity and cancellation. **Do not:** declare GPU speed metrics or unsupported quant/model success.

### P07 — real GPU worker, local pools and event safety

**Deliver:** actual optional GPU backend with buffer generations, stream/dispatch/event callbacks, local subpool within global reservation and first direct compiled model regions.

**Pass:** full-source GPU/CPU parity for eligible shapes; delayed DMA and stale event generation tests; no global per-token allocator lock or obligatory CPU sampled-token round trip where native StepProgram supports it.

### P08 — first complete real-model vertical slice

**Deliver:** `run ./model.gguf` or equivalent from trusted model file through exact tokenizer, SIR, legal compiled region, device worker, state Prepare→PublishRoot and committed terminal output.

**Pass:** real checkpoint+reference, CPU and qualified GPU if feasible, explicit fallback when fast kernel disabled, no mandatory server/cloud/config or model-source download. **This is the key architecture viability gate.**

### P09 — canonical service/SDK plus model router entry

**Deliver:** same Engine library serving CLI and one loopback HTTP endpoint, exact model alias binding, ModelCatalog with immutable revision and basic on-demand local load. Implement tested `/v1/models` and nonstream/stream `/v1/chat/completions` as supported.

**Pass:** official SDK pinned-version conformance, semantic/source parity between direct CLI and HTTP, SSE usage/stop/UTF-8 and bounded cancellation, local binding/admin/auth baseline. **Do not:** add separate model API gateway.

### P10 — physical residency + full-state snapshots

**Deliver:** global source/resident context budget and page/slab pools, safe state snapshot/partial cache prefix and model load/warm/idle drain/evict. Distinguish registered model, resident model, replica and mutable TaskSession.

**Pass:** 3+ full state domains resumed correctly, real memory reservations and pressure/eviction tests, no active unsnapshotable KV or graph freed to load another model.

### P11 — complex hybrid attention/recurrent semantics

**Deliver:** source-exact registered latent/sliding/selected sparse and recurrence/GDN/KDA operators as demanded by chosen source fixture, state codecs and legal chunk vs decode.

**Pass:** long-context official logits/quality, snapshot/replay/speculative accepted-prefix, invalid position/indexer/tied state rejection. **Do not:** assume ordinary softmax partition algebra for SSM/learned indexers.

### P12 — routed sparse weights and tiered IO

**Deliver:** exact routed/shared expert math, grouped device assignments, resident expert slabs/generation and bounded host/SSD misses with correct state/IO.

**Pass:** randomized route overlaps, cold/expert cache failure/replacement and source model reference math. **Do not:** silently change trained router selection count to improve speed.

### P13 — first extreme hybrid model region

**Deliver:** one very different complex model's legal source operators plus hand-tuned/compiled whole-region candidates, realistic prefill/decode/verify and state/layout fallback.

**Pass:** full model/task quality and resource state, independent matched performance where tested, forced generic fallback without a new model runtime.

### P14 — second complex family confirms modularity

**Deliver:** second mathematically different model, e.g. different sparse/recurrent/attention family, using same Artifact/Task/Region/Device/StateRoot and service.

**Pass:** real source quality, bounded new operator/recipe code change and no modifications to unrelated model server/allocator. Track module/line delta.

### P15 — target-authoritative speculative decoding

**Deliver:** optional trained/independent/prompt draft source, exact target verifier/correction and all-domain provisional/rollback with per task RNG/grammar.

**Pass:** sampling distribution/reference tests, rejected tail state exact, no preliminary draft output and net full-task gain before default promotion.

### P16 — high-concurrency multi-model and stage-aware scheduling

**Deliver:** native batched rows, model-specific queues, warm/cold priority fairness, equivalent replica selection, state-affine routing and typed stage budgets (token/encoder/audio/denoise), resource/tunnel cost-aware placement policy.

**Pass:** C1/2/4/8/16/32 P95 and useful outputs/energy, many resident model load/eviction under pressure, no global per-token router dispatch or starvation. Revert scheduling complexity that loses against simple policy.

### P17 — broad model roles, adapters, embeddings

**Deliver:** more validated ModelDefinitions, adapters with exact keys and encoder/pooling/rerank task interfaces; composed target+draft and encoder+rerank model roles with separately reserved artifacts.

**Pass:** new source/adapter/model quality, independent reference vectors and model alias/task/endpoint capability matrix. Reject incompatible KV source/adapters.

### P18 — ASR/TTS/image/video TaskDrivers

**Deliver:** one true supported source model per claimed task, trusted stage loop, task-specific stream/out units, codec/state and source quality vectors on same executor.

**Pass:** task-specific official/independent reference and cancellation/output validity; do not fake audio/image endpoints with LM tokens.

### P19 — networked multi-device, layer/state and fabric

**Deliver in safe order:** fake remote closed-region peers then **authorized** real two-node layer+KV colocated pipeline, full state transfers/remote prefix, P/D and attention/FFN when cost-qualified, token-range attention with registered algebra, optional RDMA/multi-GPU/graph fabric and remote state/epoch recovery. Coarse router may balance equivalent replicas or select a sharded deployment.

**Pass:** reference logits/full state, wrong quant/peer/lost ACK/timeout, no split brain, physical link direct/relay P95 and capacity-vs-speed honesty. **Do not:** connect actual machines or tunnels absent explicit user authorization, nor build full cluster consensus as first implementation.

### P20 — bounded offline optimization and cost model

**Deliver:** measured variants, full task/region cost and shape/program cache, cold/warm/placement optimizer with hysteresis and negative experiments. Separate Compile/Load/Warm from StepPlan dispatch.

**Pass:** stable matched model/task real performance improvement; bounded optimizer memory/compile overhead and no per-token JIT/search. Unqualified candidate remains disabled.

### P21 — security, source rights and reproducible release

**Deliver:** hardened parser, model source/license policies, HTTP loopback/auth/TLS/admin model control, remote peer identity/privacy, rust-owned kernels and vendor SDK/foreign code provenance, Nix/OS build/release packaging and optional model state retention policy.

**Pass:** fuzz/denied remote/SSRF/prompt secrecy/driver error recovery and release/source NOTICE evidence. No hidden cloud account or auto privileged power changes.

### P22 — release-quality numerical and physical performance matrix

**Deliver:** full source checkpoint/operator/quant/driver/hardware/task/serialization quality tests and benchmark harness with raw confidence, TTFT/ITL/P95/committed output, memory/energy, local/run/API overhead and eligible multi-peer cost.

**Pass:** published wins/ties/losses with matched baselines, no untested TPU/NPU/wafer model “supported” badges, official SDK and verified model source. Hardware not available is **BLOCKED**, not silently assumed pass.

### P23 — long-term extension and future technology proof

**Deliver:** deliberately add a new ModelDefinition, new registered SemanticOp/StateDomain, another device/program provider, new model/alias profile, a non-token TaskDriver and a TransportProvider/topology while keeping old code functional. Optional synthetic TrainStep/ParameterVersion seam demonstrates future training compatibility without training product scope.

**Pass:** measured change radius, no duplicated serving engine/session writer/memory manager, safe source/state migrations or cache invalidation and complete regression suite.

---

## 5. Test programs as grouped work, not 86 mandatory day-one tasks

The [Integrated Stage0](INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md) includes:
- **S0-00…S0-14:** fundamental source kernel/format/semantic/state/compiler/end-to-end proofs;
- **S0-NPU-A…F:** optional hardware source/compiler/safety/quality/perf;
- **S0-V4-A…J:** prepared StateRoot, device-native hot loop, hierarchical allocation, shape and minimality;
- **S0-FABRIC-A…H:** fake graph-only/static dataflow provider then optional real compiler/hardware;
- **S0-DIST-A…L:** independent placements, peer state/tunnel and failure;
- **S0-SERVE-A…J:** one-command local, model/alias, residency, equivalent replica, config and API conformance.

The [H01…H86 Experiment Program](EXPERIMENT_PROGRAM.md) contains research hypotheses and benchmark matrices. **Do not run every experiment by default**; choose only relevant hypotheses for the phase, with a documented baseline, code version, hardware and quality controls. Simulated resource/transport and source model physical tests have different promotion levels.

The [Requirements Traceability](REQUIREMENTS_AND_TEST_TRACEABILITY.md) links important ES/INV and U requirements to stages and test obligations.

## 6. Implementation priority / hard dependencies

```text
P00 (lean Rust CPU build + fixture rights)
   +-- P01 (GPU source feasibility, independent optional blocker)
   +-- P02 (math oracle) -- P03 (real format/tokenizer) 
                    \      /
                     P04 (effects + TaskDriver)
                       |
                     P05 (compiled plan + leases)
                       |
                    P06 -> P07
                        |
                       P08 (REAL model through CLI)
                        |
                 P09 model router/API
                        |
             P10 physical context/residency
                        |
        P11/P12 hybrid + sparse correct math
                        |
             P13/P14 extreme models
                        |
             P15/P16 speculate/batch
                        |
             P17/P18 breadth of model/tasks
                        |
             P19 distribution, optional SDK fabrics
                        |
              P20/P21/P22 release & qualification
                        |
              P23 re-run extension drill
```

This is a **rough dependency spine**. Some work can parallelize when owned interfaces and tests are independent: parser fuzz and CPU source math, shape guard and resource event simulator, SDK adapter mock conformance and model residency simulation, CLI documentation and source license audit. Avoid freezing interface prematurely or stubbing all phases at once.

````

## Source: .agents/plans/archive/history/CURRENT_STATUS_AND_TRACKING.md

Original UTF-8 bytes: 180; SHA256 `fa73c0751e8106d902962a87725e017a3de0b912fe1f3a2f713cc831756859ed`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Current work

Use [the plan TODO index](TODO.md) and [the agent workflow](../../AGENTS.md).
The old handoff/status process is retired; this path remains only for existing links.

````

## Source: .agents/plans/archive/history/DECISION_REGISTER.md

Original UTF-8 bytes: 107297; SHA256 `2b5651d458aeff95ef86350dd13ae57ccae85df6f1164bb7a64604eba54b38af`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Decision register — confirmed user intent versus recommended design

**TODO-USER-001 · USER ·16 October2026:** Replace the handoff/evidence process
with `.agents/plans/TODO.md`, one `.agents/plans/<plan>/TODO.md` and independent
`.agents/runs/<plan>-<session-or-agent-id>/TODO.md` files. Keep a provenance
commit, implementation/test checkboxes, completed commit sections and a rolling
Uncommitted section. The coordinator atomically integrates agent results into
main plans; finish a plan by checking its one index line. Archive only completed
TODO sections as needed. No mandatory generated handoffs, SHA inventories,
source snapshots or run receipts. [Workflow](../../AGENTS.md). This supersedes
older process/preservation/update duties below; substantive architecture and
actual model/source/numerical/state/resource requirements stay in force.


**PULL-A1-001 · VERIFIED ·16 October2026:** Keep local cache publication in the
existing artifact owner behind optional Linux cache/diagnostic features. Reuse
caller-owned physical admission, stable process locks and exact retained file
handles; report post-ref-rename sync failure as explicit DurabilityUnknown.
No executable readiness or source identity follows from data publication. A01..09
pass within [the phase's stated limits](../research/PULL_A1_LOCAL_XDG_CACHE_2026-10-16.md);
A2 HTTPS/providers/CLI and A3 owned-copy loader integration remain next.


**EVIDENCE-USER-002 · USER ·16 October2026:** Remove SHA inventories for ordinary
evidence logs and simplify the evidence specification. Git commit, actual command,
result and tested scope establish the record. No global evidence catalog,
per-log hash or source-file hash inventory is required. Keep hashes at real
model/artifact/content or explicit byte-identity qualification boundaries.
[plan TODO index](TODO.md). This supersedes older log
inventory/sealing instructions; design-document SHA tooling remains separate.


**CONC-USER-002 · USER ·16 October2026:** Use left-right thoughtfully at
real contention sites. The cold model catalog is not a justified migration.
Reject its uncommitted implementation and restore the accepted production
source/dependencies; preserve the experiment as rejected evidence. Require
representative access/contended-wait evidence and ownership fit before adding
a concurrency primitive. Existing direct immutable sharing and authoritative
transactions retain their appropriate design. [Correction and current review](cpu-gpu-optimization/TODO.md).


**H25 promoted-product gate ·16 October2026:** Both original Qwen3.5 models
now pass27 checks on each of local CUDA/NVIDIA Vulkan, pvl-x2 HIP and pvl-a1
CUDA/HIP:135 actual checks, including cold prelaunch negatives, plus12 actual
scalar/libm tests. Genuine512 full logits/state/trace, replay and physical
cancellation/quarantine pass; the local gfx90c hold remains. Canonical archives
and exact retirement now preserve old records inside `.agents/` with original
phase/path/hash mappings. Final repository/catalog/documentation closure remains
in progress. New Qwen/Gemma source work is isolated and does not inherit trained
execution from these results. [Current phase](accepted-baseline/TODO.md).


**QWEN-PARITY-USER-001 · USER ·16 October2026:** For Qwen, finish the
source-correct basics first, then match every capability and optimization of
llama.cpp and vLLM, then add further appropriate MTP, speculative decoding,
n-gram, DSpark2/DFlash2 and related optimizations. Maintain a version-pinned,
exhaustively enumerated capability ledger; each applicable mechanism needs
functional/source/ownership qualification and measured performance. Methods
already present upstream remain parity gaps until their own gates pass.
No guessed model/drafter pairing or synthetic acceptance counts as inference.
[Ordered implementation, complete coverage and acceptance](../research/QWEN_BASICS_PARITY_AND_ADVANCED_DECODING_ROADMAP_2026-10-16.md).


**MODEL-PARALLEL-USER-001 · USER ·16 October2026:** Begin Qwen3.8 27B,
Gemma4 E2B and Gemma4 26B source/CPU implementation in parallel while current
qualification runs. Use subagents and disposable isolated worktrees where
changes overlap; main coordinates, reviews, reconciles and merges verified
pieces. The two Gemma tracks share one family implementation. Preserve the
existing engine, source identity, state/resource authority and new-model
CPU-first acceptance. No new checkpoint/SDK downloads or GPU capability claims
are authorized. This adds parallel model work while shared layout → left-right
→ XDG acquisition → further GPU optimization sequencing remains binding.
[Ownership, source gates and integration plan](model-execution/TODO.md).


**LAYOUT-RETIRE-USER-001 · USER ·16 October2026:** Remove `probes/` entirely,
moving implementation, reference tests and helpers to `crates/`, `tests/` and
`tools/` according to their real responsibility. Audit sidetree worktrees and
clean up all that are merged or superseded while preserving unique work and
active users. Resnapshot and archive historical evidence in the existing
`.agents/` documentation model; all retained historical evidence must live
inside `.agents/`, never in external build directories. Store underlying
records and original-path/hash/disposition indexes under
[`.agents/plans/evidence/`](TODO.md); obsolete build caches and
compiled executable bodies are retired after their provenance is recorded.
This extends the final H25 layout gate and does not change the layout →
left-right → XDG acquisition → GPU optimization order.

**Next: [workspace/model/feature layout redesign](accepted-baseline/TODO.md), immediately followed by the [left-right concurrency redesign](cpu-gpu-optimization/TODO.md), then individually qualified CPU/GPU performance variants.** Preserve the [five-GPU/three-CPU target inventory](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md), existing model/serving priorities and [GPU text/device orchestration roadmap](../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md). Fleet-wide capability optimization is unfinished; [native FP32 FMA](../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md) remains a proposal, not an enabled policy. Capacity remains 512; broader GPU contexts/corpora, independent CPU8K GDN divergence, genuine16K/32K qualification and product checkpoint persistence/security UX remain open. The default CPU CLI and user-authored root README are unchanged. Earlier status entries below are preserved as historical evidence.

**CRATES-USER-001 · USER ·16 October2026:** After the native backend qualification boundary, promote cohesive implemented packages under `crates/`, keep a small public core/facade, and make backends, model families and product capabilities independently selectable with real minimal/composed/`all` build gates. The bootstrap crate count is not a permanent cap; extraction preserves the same engine, state authority, physical ownership and source bits. [Workspace/model/feature redesign roadmap](accepted-baseline/TODO.md) records proposed package seams and sequencing; none is implemented by this documentation.

**CONC-USER-001 · USER ·16 October2026:** Apply [jonhoo/left-right](https://github.com/jonhoo/left-right) broadly wherever it fits read-heavy published immutable data/indexes, replacing application read locks or mutexes where semantics and measured performance justify it. Conduct this concurrency redesign pass **immediately after the workspace/layout/structure redesign**. Reinventory every owner; preserve one writer, coherent publication, deterministic replay, existing session/resource/actual-completion authority and physical retirement. [Complete candidate inventory, migration and acceptance](cpu-gpu-optimization/TODO.md) includes hidden second-copy slot retention and unregister/drain/re-register source identity, bounded guards and writer delay. This request adds the roadmap/design direction now; dependency and runtime changes belong to its later verified implementation units.


**HW-FLEET-USER-001 · USER ·16 October2026:** Initial optimization targets are pvl-l5 (current), pvl-a1 and pvl-x2: five physical GPUs and all three CPUs. Explicit SSH work is authorized on the two remote machines. Optimize capability-selected execution paths, source-exact kernels, memory placement and complete-request performance for these targets first; advertised instruction support alone is not qualification. The user also explicitly requests subagents for each implementation going forward, using independent ownership and parallel work to keep the main thread focused and efficient. [Observed fleet/ordered optimization gates](../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md) and [active native CUDA/HIP phase](accepted-baseline/TODO.md) retain scope/evidence. Temporary test artifacts and reuse of already-owned approved checkpoint bytes support that implementation; new external model/SDK acquisition, persistent device/service/privilege changes and publication retain existing restrictions. Root README stays unchanged.


**OPEN-001 · USER · 15 October 2026 (MIT, AI disclosure and open contributions):** The user selected **MIT** for the Zerogravity project source and documentation, requested the standard root `LICENSE`, and approved clearly recording `license = "MIT"` for the root and optional GPU package. README must include a succinct AI disclosure influenced by [ds4](https://github.com/antirez/ds4): **human-centric design with AI-centric implementation**, substantial AI assistance without unchecked autonomous control and without claiming human line-by-line inspection. The founder **Prasanna Loganathar** explicitly welcomes everyone to build the project openly and kindly; preferred contributions start as design/architectural issues or small, contained implementation changes with a documented plan under `.agents/plans/` before nontrivial coding. [Public contribution guide](../../CONTRIBUTING.md) and [full policy](OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md) capture the workflow. MIT applies to this project's own material; model weights, dependencies, codecs and imported assets retain their independent licensing. User approval covers documentation/contributor/licensing metadata, not new inference implementations or external service access.


**SERVOPT-USER-001 · USER · 15 Oct 2026:** After a read-only vLLM concurrency review, the user explicitly instructed strengthening every recommended concurrent serving/design/implementation plan, then a complete product-focused README rewrite with Intro, Quick start, Architecture, FAQ, Implementation status/roadmap, concise engineering bullets and high-level diagrams including joint optimization. The user requested exactly: **The blazingly fast AI execution fabric - built for composable multi-modal inference specialization at every layer, from individual kernels to distributed systems**, and early explicit text/image/video/audio/realtime scope. Examples [dotlink](https://github.com/abird-ai/dotlink) and [agentc](https://github.com/abird-ai/agentc) are structural references, not required implementation copies. [Joint serving contract](../docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md) adds accurate source-verified vLLM-class bounded concurrency and joint legal batch/spec/cache/kernel/placement/QoS; [benchmark/acceptance](../research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md) specifies 36 PLANNED, not-run tests and truthful comparison conditions. This is authorization for docs/README and minimal doc tooling only, not new GPU serving engines, benchmarks, downloads or deployment. Existing one-Rust-engine and no-hidden-quant policies remain unchanged.


**VST-USER-001 · USER · 15 October 2026:** The user explicitly authorized clean changes to architectural docs/plans **and the minimal documentation tooling needed to repair the nested full handoff archive**, overriding the earlier design-only restriction for these documentation artifacts, while also requiring **SSD streaming to become a first-class, nearly zero-abstraction-cost storage policy** under the same source tensor model as fully resident GPU memory. DS4/DwarfStar and related SSD-streaming strategies are inspirations; one physical ResourceArbiter/TaskCoordinator and Rust inference engine remain mandatory. Direct fully resident fast path should avoid extra software overhead, but actual SSD misses necessarily pay IO; source tensor identity, verification, GPU command reader pins and real completion must be preserved across GPU/UMA/RAM/SSD tiers. Detailed specs: [first-class virtual tensor residency](../docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md), [source research](../prior-art/SSD_STREAMING_AND_RELATED_PRIOR_ART_2026-10-15.md), [SSD-P0..P5/SSD-01..32](../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md). **Current user decision authorizes docs/tooling, not actual SSD inference implementation or large model downloads**. Proposed policies/tests are specified but not yet verified in actual hardware. This VST-prefixed local decision avoids earlier U-040/U-041 historical numbering collisions.


**15 October 2026 design-only source-specialization clarification:** After the explicit read-only specialist-source review, user requested **all its precision/specialization gaps resolved in documentation and implementation plans, cleanly and elegantly**. The [specialization implementation plan](../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) preserves the one-Rust-engine and original model priority user commitments. New identifiers **SYK-001..010 are PROPOSED design obligations; SYK-011..016 are OPEN policy/source/permission questions**, NOT additional user-frozen ABI approvals. Do not conflate with earlier v5.1 addendum's historically reused U-040/U-041 or overwrite the shared U-040 native model implementation entry. No new code/model/driver changes were requested by this instruction.


**Read this before interpreting any spec.** Project date 2026-10-10; provenance is the originating conversation. Identifiers are stable for future design reviews. Status: `USER` = explicitly requested or endorsed; `PROPOSAL` = assistant-origin recommendation, not separately approved; `OPEN` = requires a measured or policy decision; `ASSUMPTION` = working interpretation, reversible.

## Explicit user decisions

| ID | Status | Decision, in the user's intent |
|---|---|---|
| U-001 | USER | Project name **zerogravity** (lowercase). |
| U-002 | USER, SUPERSEDED PATH | Originally requested comprehensive agent-readable plans, specs, decision history and takeover documentation under the relative `./zerogravity/.agents/{docs,plans}` path, via **dotlink l5**. Its location requirement is superseded by U-013; the documentation scope still applies. |
| U-003 | USER | The inference engine's own implementation is to be **entirely Rust**; record a strict desire for Rust-authored kernels, not a C++ inference engine wrapper. What counts as unavoidable vendor runtime bindings is clarified in O-003. |
| U-004 | USER | Support standard model families, expressly **Llama**, **Gemma 4**, **Qwen 3.5+** (dense/MoE/hybrid) and additional newer models, rather than specialize for one fixed checkpoint. |
| U-005 | USER | Support unusually large / specialized models including **Qwen3.8-Flash-Next (125B backbone + auxiliary table/MTP)** and **GLM-5.3-Flash (~321B total)**, especially on **Ryzen AI Max+ 395 / Radeon 8060S / AMD Strix Halo (gfx1151)**. |
| U-006 | USER | Target vLLM/SGLang-level peak server efficiency while retaining llama.cpp-like accessibility/portability and Maya/Gufo/DwarfStar-like hardware-specific extreme optimizations. |
| U-007 | USER | On suitable hardware activate maximal specializations; **otherwise fall back to general model execution**, without degrading unrelated optimized operations. |
| U-008 | USER | Combine architectural learning from vLLM, SGLang, llama.cpp, Maya, Gufo, Strata, DwarfStar (antirez/ds4), plus complementary runtimes and Rust projects. |
| U-009 | USER | Adopt the **Semantic IR-based** architecture as a way to express model computation once and allow specialized execution without a hot-path abstraction penalty; ambition is to *outclass* existing engines where demonstrably possible. |
| U-010 | USER | Prior stage was **research only**, then explicit request to refine architecture, now explicit request to persist comprehensive plans and handoffs. No code implementation was asked in this turn. |
| U-011 | USER | The record must preserve **all user decisions, detailed design decisions and pending questions**, so external LLM agents can implement from the files without relying on chat history. |
| U-012 | USER | Performance ambition is across both model breadth and specialized hardware. There is **no user-approved target performance percentage or deadline**; faster-than-everything is a goal, not a result. |
| U-013 | USER | **Canonical project location is `/home/pvl/spaces/abird/src/zerogravity/`**, explicitly correcting the original relative placement. Move existing documentation there, preserve files, update references and revalidate. This supersedes only the destination in U-002. |
| U-014 | USER | **Incorporate the best of the research/inference architecture and src/zerogravity Semantic IR architecture, adapting all plans to a unified stronger design.** The user approved the combination goal; this does not automatically approve each old council-specific Rust ABI, license or performance threshold. |
| U-015 | USER | Conduct **comprehensive new research** to design zerogravity for maximum potential to surpass existing inference projects while preserving long-term maintainability, clean adaptability to future technologies and robust extensibility at **every stage of the pipeline**. State measured evidence versus conjecture and update the authoritative plans. |
| U-016 | USER | Explicitly include the architectures and pipelines demonstrated by [mikealanni/gufo](https://github.com/mikealanni/gufo) and the October 11 Strix Halo Reddit report on **Qwen3.8 Flash-Next 1,800+ tok/s cooperative GPU+XDNA2 NPU prefill**, making these capabilities fit elegantly into zerogravity rather than becoming model-specific parallel runtimes. This is a **design/support requirement**, not proof that XDNA2 code is implemented or automatically safe to enable. |
| U-017 | USER | **Generalize all project plans, specs and architectural interfaces away from external project names**, with an exception permitting concrete naming/reference to verified MIT-licensed sources. Retain all upstream work as **identified, attributed prior art and research inspiration**, rather than as architectural base classes or cloned engines. Build zerogravity's **own original, modular and composable AI execution architecture** so new model/accelerator/quantization/task/pipeline specializations do not require rewriting new inference engines. Preserve model-family and hardware/protocol identifiers when materially necessary, and do not lose source provenance or license restrictions. |
| U-018 | USER | **Perform another comprehensive critical review/refinement of the entire architecture and detailed specifications**, aiming for the highest-performing and most composable runtime across AI execution tasks, while remaining **minimal whenever complexity has no proven value** and modular enough to incorporate novel models, devices, kernels and AI task pipelines without engine rewrites. This requests architecture/design refinement, not immediate implementation or proof of benchmark superiority. |
| U-019 | USER | **Verify and generalize accelerator/runtime architecture coverage** beyond local Strix Halo: explicitly include NVIDIA DGX Spark/GB10 coherent Grace–Blackwell, discrete and datacenter/rack-scale Blackwell, Google TPU, possibly Cerebras wafer-scale (user phrasing “cerberus” also corresponds to a separate parallel-decoding technique), and other programmable AI accelerators. Maintain one composable source-independent runtime that can absorb **native-kernel, AOT graph, deterministic/static dataflow and other specializations at their natural execution granularity** instead of cloning another engine. This is an architecture/research requirement, not proof of actual hardware support. |
| U-020 | USER | **Distributed and networked execution is an additional first-class zerogravity goal**, not just fastest inference on one machine. Support heterogeneous GPUs/CPU/NPU and programmable accelerators **across multiple machines, including trusted remote tunnels**, workload sharing/sharding and a range of topologies: layers and their cache colocated, different model layer groups on different machines, layer compute and its KV/recurrent/cache on different machines, and other contemporary/near-future combinations. Preserve performance, composability, auto-or-pin topology policy and mathematical/state correctness without writing a different runtime for each placement strategy. This requests architectural design/spec updates, not code/cluster deployment or permission to connect remote peers. |
| U-021 | USER | **Build a native multi-model model router into zerogravity**: resolve and load balance requests for as many models as needed, support separately configured model parameters/profiles and simultaneous serving where physical memory allows, with one OpenAI-compatible HTTP API interface. Preserve an **extremely simple one-command local run/serve experience** comparable to easy local model runtimes, while keeping every aspect of model/quant/placement/replicas/context/sampling/routing/API/device/network resource behavior configurable on demand. This is a **design requirement**, not implemented CLI/API or permission to download models/open network services. |
| U-022 | USER | Perform a **final comprehensive editorial and technical refinement** of zerogravity's full architecture/design specifications, organize them for clear authority and maintainability, prepare a **publication-quality technical whitepaper**, and deliver a **complete implementation handoff for LLMs and autonomous agents** to build and validate the system without reconstructing the chat. Retain all historical decisions/research/phase IDs and be honest about design-only status, prior art and unmeasured performance. No actual runtime, PDF, Git, model download or machine changes were expressly requested. |
| U-023 | USER | **Implement only zerogravity's project/bootstrap packaging foundation**, adapting the existing `/home/pvl/spaces/abird/src/dotlink` Rust/crane release/quality conventions as a **read-only reference**: a real minimal binary and project-defining `README.md`, Rust build/format/lint/test tooling, pinned Nix/crane flake and NixOS module, cross-build definitions for major Linux/Windows/macOS architectures, GitHub Actions build/test/release workflow, checksum-protected install scripts and reproducible release artifacts. **No model inference, tokenizer, serving/router/API, device kernels or distributed execution implementation in this phase.** Actual GitHub remote/release and physical cross-platform builds must be separately verified. |
| U-024 | USER | **Begin Phase 1 implementation now**, transitioning from packaging-only to the first real Rust model-source/reference foundation and ultimately running both real Qwen3.8-27B and Gemma 4 26B-A4B-it through one runtime. The user has **not** separately authorized a multi-gigabyte model download, device firmware or privilege changes, Git publish/release or remote peer enrollment. This supersedes U-023's *temporary implementation scope*, not its validated packaging infrastructure. Phase 1 partial implementation is recorded in the historical Phase 1 source/reference handoff. |

| U-025 | USER | **Reordered practical acceptance milestones**: first Qwen3.5 **0.8B** and **2B**, each CPU text inference first, then complete native **NVIDIA CUDA**, native **AMD ROCm/HIP**, and **Vulkan** on both physical GPU families; only after those, **Gemma 4 E2B**, **Gemma 4 26B-A4B**, and **Qwen3.8 27B** should receive source-correct **CPU execution first**, with their GPU work deferred. Preserve a shared Rust-native semantic execution engine and reference/fallback semantics; a physical specialized kernel proof does **not** qualify full-model GPU inference. See the Qwen3.5 CPU/GPU handoff for exact delivered partial implementation and open gates. |
| U-026 | USER | **Complete rigorous CPU numerical, prefill/state and genuine long-context qualification for Qwen3.5 0.8B and 2B before moving to full-model NVIDIA CUDA / AMD ROCm / Vulkan integration.** The new verification must distinguish actual model CPU token processing from synthetic 32K operator/position tests and independently trace model/reference mismatches, failing closed rather than widening thresholds to hide differences. Current status: substantial added tests and guarded chunked prefill, but the CPU gate is **PARTIAL/BLOCKED** on 2B strict distribution drift, missing per-layer reference, and 4K..32K real-model evidence. See the Qwen3.5 CPU numerics/context handoff. |
| U-027 | USER | **Move on to native GPU development once Qwen3.5 0.8B and 2B have verified real CPU semantics through 8K.** Defer 16K and 32K whole-model source/reference qualification to later repeatable local tests, rather than treating them as prerequisites for CUDA, ROCm and Vulkan integration. Keep all independent short numerical, layer, state, error and 8K context acceptance checks and thresholds intact; the deferred longer tests remain UNQUALIFIED. This supersedes only the 16K/32K-before-GPU ordering implied by U-026. Two explicit CPU scopes now exist: Phase 1 GPU-development admission (16/16 PASS) and deferred extended-context (16/20; four missing). The first native accelerator source-matvec integration seam has passed CPU fallback/failure transaction tests; physical complete GPU model execution is NOT yet qualified. See [Phase 1/P07 handoff](numerics-and-context/TODO.md). |
| U-028 | USER | Commit Qwen3.5 CPU Phase 1 and accelerate P07 GPU implementation while independently examining high 8K shared top-20 logprob errors (~0.2037/0.1968). Seek source-grounded improvements without widening numeric thresholds. Baseline initial phase commit is a965737; GPU physically qualified inference remains pending. See [8K numerical drift report](numerics-and-context/TODO.md). |
| U-029 | USER | Organize three parallel workstreams: GPU implementation, accuracy investigation, and multimodal architecture design with independent substreams/subagents where tools permit. Keep model design composable, native Rust and accurately source-grounded. Standalone Codex/Claude agents were discovered on host but authentication failed (Codex HTTP 401, Claude not logged in), so NO independent autonomous agent was claimed to have run. See [multimodal implementation checklist](multimodal/TODO.md). |
| U-030 | USER | Continue into the next implementation phase, preserving three parallel workstreams. Deliver actual native Rust/CubeCL source-quantized GPU regions beyond Q4_0, improve independent numerical understanding without relaxing parity thresholds, and advance additive multimodal architecture via non-disruptive mock contract tests. P07-B source-mixed checkpoint host parity and five MM0 checks were implemented; physical GPU devices and LLVM23 CUDA/HIP toolchain remain prerequisites for real backend validation. See [P07-B/MM0/numerics handoff](accepted-baseline/TODO.md). |
| U-031 | USER | Continue into P07-C GPU development: source-identity-safe persistent device weight residency, bounded physical buffer leases, strict source-only fallback and completion/event fencing, reducing repeated per-token uploads without turning CPU-only Zerogravity into a GPU-dependent binary. Preserve the 16/16 CPU Phase1 gate, defer 16K/32K and do not label unverified physical GPU execution as passed. See [P07-C exact implementation handoff](accepted-baseline/TODO.md). |
| U-032 | USER | Continue P07-D GPU development from P07-C: eliminate repeated host GGUF source-weight decoding on GPU cache hits, establish typed asynchronous device lease/event lifecycle and fail-closed completion/cancellation/epoch rules without perturbing CPU source math, and preserve clear separation between host tests and actual GPU hardware qualification. A decode-once StaticSource / DynamicStep split was implemented; device-independent DeviceEventLedger tests and MM-06 duplex/mock cancellation validation pass. Physical asynchronous CubeCL GPU integration remains later work. See [P07-D handoff](accepted-baseline/TODO.md). |
| U-033 | USER | Continue to P07-E from P07-D: integrate a real async CubeCL readback with the typed DeviceEventLedger, keep original synchronous CPU model semantics and sole TaskCoordinator StateRoot writer, and preserve explicit hardware qualification evidence. A source-typed async completion bridge and fail-closed future cancellation were implemented and host-tested; multi-device GPU concurrency/full-state execution remain later work. See [P07-E implementation handoff](accepted-baseline/TODO.md). |
| U-034 | USER | Recover interrupted P07-E state, validate the live committed implementation and continue the next GPU safety phase. P07-F retains the actual CubeCL source, dynamic and output handles on an uncertain/dropped Future in a process-wide service-quarantined keepalive registry, and accounts physical resident GPU weights plus activation/scratch/output under one per-provider budget. All CPU trained source and Phase1 8K acceptance remain unchanged; actual native device cancellation and global multi-model ResourceArbiter remain unqualified. See [P07-F handoff](accepted-baseline/TODO.md). |
| U-035 | USER | Revalidate and continue all Zerogravity work after the interrupted P07-E continuation, then author a complete comprehensive handoff markdown retaining the original idea/design, every phase decision/specification/implementation and pending item going back to project start, with a ready-to-run takeover prompt. Live baseline was already P07-F 602c246, not the old interrupted P07-E draft. P07-G1 adds source-neutral global device admission and bounded fair queue tests (CPU control plane only); physical GPU worker integration remains pending. Next agent must read the latest comprehensive handoff progressively in full before implementing. See [P07-G source scope](accepted-baseline/TODO.md). |

| U-036 | USER | Take over the live source-controlled Zerogravity repository at documented P07-G1 handoff commit `f28dc10`; revalidate the complete historical source/quality provenance and continue **P07-G2** binding attested physical CubeCL service/adapter identities to the single shared `ResourceArbiter` and P07-F/E residency, events, buffer keepalives and fail-closed cancellation. Preserve independently owned GDN accuracy and multimodal streams, canonical Qwen-first model order, 8K Phase1 acceptance and 16K/32K deferral. Do not claim hardware inference from compilation/mocks; keep concurrent agents' files, avoid bulk staging/stashing, prohibit unauthorized downloads/driver changes/push/releases. This implementation slice delivers conservative P07-G2a global physical envelopes and source-readback guards; actual cross-model fair GPU workers and full physical CUDA/ROCm/Vulkan model validation are still pending. See [P07-G2 phase handoff](accepted-baseline/TODO.md). |

| U-037 | USER | The user confirmed the `dotlink-l5` connector should now expose GPU hardware and requested immediate continuation to the next Zerogravity implementation phase, then explicitly requested revalidation/resumption after interruption. Revalidated clean `7e6e4f4` G2a baseline, real `/dev/nvidia*`, DRM render and KFD devices; physically exercised native CubeCL Vulkan on NVIDIA RTX 3060 and AMD Renoir, implementing P07-G2b shared per-physical-card bounded fair workers and strict original Qwen GGUF numerical/StateRoot gates. NVIDIA Vulkan source-region workers and partial-model Qwen0.8B/2B are bit-exact; AMD RADV physical readback is real but source FP32 arithmetic differs and therefore remains **numerically unqualified**, with failed Qwen StateRoot publication refused. CUDA/HIP LLVM23 development toolchain, full GPU GDN/KV/attention and physical cancellation remain pending; 16K/32K CPU contexts deferred. The verified phase evidence/limitations are in [P07-G2b](accepted-baseline/TODO.md). |

| U-038 | USER | After successful clean P07-G2b GPU worker commit `c3e5fca`, user requested **next implementation phase**. P07-G3a continued with REAL NVIDIA and AMD GPU access, resolving AMD Renoir RADV's observed original Qwen3.5 Q4_0×Q8_0 native-F32 fused-accumulation mismatch by a separate conservative WideF64 GPU compile-time specialization. BOTH physical cards now pass exact original GGUF source-bit four-dispatch 0.8B/2B GPU worker checks; AMD four-token real CPU+one-GPU Qwen hybrid complete-vocab logits/StateRoot identical to pure CPU. Original fast NativeF32 remains separately diagnosable and still FAILS strict source equality on AMD (PASS NVIDIA). Explicit opt-in feature-only physical submitted-Future-abort probe shows both GPUs retain five original buffers in process-lifetime quarantine, conservatively fence/retain 256MiB logical GPU provider charge, reject second model, and publish no stale StateRoot. Do **not** call this proof of driver cancellation/recovery, measured allocator VRAM, fast GPU inference or complete Qwen GPU graph. Native CUDA/HIP compilation needs unavailable LLVM23 and vendor SDK; CPU Phase1 16/16, extended 16/20 deferral, GDN long-context accuracy and MM0 v5 independent streams unchanged. See [P07-G3a physical evidence](accepted-baseline/TODO.md). |

| U-039 | USER | After `67dca72` G3a commit, requested a **fresh adversarial review** of bugs, correctness, regressions and a robust, simple, intuitive, future-proof, low-maintenance and composable software architecture. Review verified current live `main` and intentionally implemented bounded fixes instead of a wholesale rewrite: pre-submit GPU validation before physical terminal fencing; scratch-aware LRU of idle source weights; same-file-descriptor GGUF source loading/reparse (prevent equal-length pathname replacement source mix); normal CPU intermediate chunk prefill elides discarded logits with real 0.8B/2B parity; correct GPU fair-queue `selected` vs physically `completed` counters and renamed globally destructive card-epoch API; checked/limited untrusted model head geometry; and non-allocating per-card global fence hot path. Tests covered actual NVIDIA/AMD Vulkan positive dual-model work, actual invalid activation, invalid original FP16 scale, underbudget rejection and recovery, real Future-abort quarantine; preserve independent CPU GDN 8K/16K/32K and MM0 streams. The detailed prioritized findings, changes, test evidence, remaining production blockers and smallest safe next steps are in [fresh quality review](accepted-baseline/TODO.md). No unapproved drivers, model downloads, remote release or Git push. |

| U-040 | USER | After the U-039 fresh correctness and architecture review, requested **resolving remaining issues** with clean, intuitive, scalable, future-proof, maintenance-light, composable Rust design. Continued live repo work on clean `e9312ad` baseline: introduced one immutable `Arc<LoadedQwenModel>` for original GGUF+predecoded layer constants shared across independent Qwen session StateRoots, backward-compatible direct load, minimal explicit `ModelCatalog` alias/lazy per-source singleflight load/retirement; real concurrent source-pinned Qwen0.8B/2B sessions and original GPU shared-resident source reuse verified. Added source-audited physical F32/WideF64 Q4 region latency probes and device-wide GPU VRAM/GTT measurements keyed by attested PCI BDF, which revealed driver memory pools remain reported allocated after logical provider release. Offline native CUDA and HIP compiles still block at LLVM23 `LLVM_SYS_231_PREFIX`, independent CPU GDN 8K arithmetic cause remains open, full GPU model and native HTTP/multi-model router **not claimed complete**. Preserve original models, source/tolerances, MM0 and explicit 16K/32K deferral. See [U-040 implementation evidence and remaining honest gates](accepted-baseline/TODO.md). |

| U-041 | USER | Absorb the newly committed 15 Oct v5.1 universal multimodal, source specialization, joint serving and virtual SSD-tensor-residency design proposals, then continue real GPU execution in verified phases; remain one Rust source-verified engine, one TaskCoordinator/StateRoot publisher and one physical resource authority. P07-H1 physically source-qualified all 187 Qwen3.5 0.8B/2B GGUF GEMV tensor regions and full-vocab/StateRoot on AMD Renoir and NVIDIA RTX3060 Vulkan; P07-H2 added actual GPU source-compatible RMSNorm with exact original f32-square, f64 sum, f32 mean/sqrt/reciprocal rounding, yielding 187 GPU matrix + 49 GPU norm regions = 236 source-bit-equivalent operations on EACH checkpoint/device combination. The rest of GDN/attention/KV/recurrent and complete device graph are not yet implemented/qualified. [H2 source/hardware evidence](accepted-baseline/TODO.md). |
| U-042 | USER | Explicitly granted FULL Zerogravity repository ownership, removing the prior constraint to preserve or partition any other agent changes. Revalidate concurrent Git before editing, reconcile all v5.1 docs and code into one coherent implementation, clean up as appropriate, update current handoffs and roadmap, progress in safe validated GPU stages, and do not fabricate full-GPU/native CUDA/ROCm/server or source-accuracy claims. Preserve original source model checkpoints/quality gates and do not push or publish releases without authorization. |

| U-043 | USER | The user explicitly superseded any prior assistant-authored README copy edits: the live README is USER-OWNED and MUST be preserved exactly as the user updated it. Do not revise, replace, clean, stage or bundle README.md into a GPU phase commit. Documentation snapshots may reference its current bytes, but must not mutate README. Continue independently verified P07-H phases. |


| U-044 | USER | Requested validation and continuation after P07-H3 grouped per-head RMS normalization and interrupted two-token device-source acceptance. Revalidated Git `4ab15a4`, original Qwen3.5 0.8B/2B GGUF SHA hashes and AMD Renoir/NVIDIA RTX3060 Vulkan access. The initially failing two-token acceptance MAP was counting final-only vocabulary and output-RMS operations twice; fixed count without relaxing source math, GPU correctness or state acceptance. On BOTH checkpoints and BOTH physical Vulkan GPUs, the original Rust CPU Qwen model exactly matches **530 REAL physical GPU completed regions across two tokens** (266 selected: 187 source GEMV, 49 residual/final RMSNorm, 30 independently grouped GDN/attention head norms; subtract 2 final-only operations), all output bits, complete 248,320-vocabulary logits, committed root and state fingerprint, with ZERO outstanding global logical reservation after real readback. Physically forced post-launch grouped-RMS Future abandonment separately passed on BOTH cards: 5 original GPU handles quarantined, 256MiB retained/fenced and root unchanged; not driver cancellation/recovery. User-owned README MUST remain exactly as supplied SHA256 `736c794cf68d82cf47b20fc8409e4876ab107888a38f98805bdedbde10d2aaf8`. The actual GDN convolution/recurrent, attention/softmax/RoPE/KV, FFN SiLU and device-resident inference graph are NOT complete; do not invent entire-GPU qualification. See [H3 real hardware evidence and H4 continuation](accepted-baseline/TODO.md). |

| U-045 | USER | Continued physically validated GPU execution after P07-H3. Implemented P07-H4 first original-source **GDN causal convolution island** through an additive typed kernel returning BOTH the raw pre-SiLU activation and private proposed 3-frame history; the one CPU QwenState still owns the commit. One-kernel GPU output failed source-bit comparison on the SECOND original Qwen token (1,840/6,144 raw discrepancies, history exact); split source FP32 multiplication rounding into actual stored device products and separately ordered reduction (two real kernels). Both original Qwen3.5 0.8B/2B checkpoints on BOTH AMD Renoir RADV and NVIDIA RTX3060 Vulkan then passed **566 genuine physical GPU region completions over two tokens**, including 36 conv raw/next-history source bit checks, 187 matrices/49 residual norms/30 grouped norms per token, exact full vocab logits and StateRoot, and zero remaining provider logical charge. Real conv Future abandonment on both GPUs retained 6 original handles, fenced 256MiB charge, rejected another model and preserved Qwen root. This does **NOT** implement device-resident recurrent GDN matrix, attention/KV/FFN or full-GPU Qwen; native CUDA/HIP LLVM23 still blocked; independent CPU 8K GDN drift remains open; original 16K/32K deferred. User-owned README preserved bit-for-bit. [Detailed H4 physical results and H5 direction](accepted-baseline/TODO.md). |

| U-046 | USER | Requested a **FULL comprehensive agent takeover handoff Markdown** covering the original Zerogravity vision and whitepaper, every specification/design iteration, all decisions/proposals/negative experiments, every implementation/test/commit phase since beginning, incomplete work and exact instructions plus ready-to-paste next-agent prompt. Keep everything in `.agents/docs/` or `.agents/plans/` and do not skip historical sources; retain the user-owned README verbatim. Delivered a new dated, practical complete takeover guide [plan TODO index](TODO.md) and regenerated the canonical **byte-for-byte lossless historical archive** [FULL_COMPREHENSIVE_IMPLEMENTATION_HANDOFF_2026-10-14.md](TODO.md), which embeds the dated guide and every earlier versioned specification/decision/phase source with SHA/byte counts. H4 `0cad60a` remains last *implementation* phase; upcoming H5 GPU GDN recurrent state is implementation work, not completed by this documentation task. No unapproved downloads, code alterations, pushes, tags/releases or README changes. |

| U-047 | USER | Requested immediate continuation of the live repository from handoff2125612/H4 through **P07-H5 source-exact GPU GDN recurrence**, after complete takeover-guide reading and live repository/model/hardware/test revalidation. Preserve one Rust engine, immutable source identity, independent StateRoots, physical arbiter/fair workers and actual completion; compare output AND proposed state bits on trained nonzero causal tokens across both checkpoints and both Vulkan vendors, with rollback, admission and post-launch handle quarantine. Continue later attention/KV/FFN/resident/vendor/model milestones in small independently verified phases. Root README remains user-owned; no model/SDK downloads, privilege changes, push/tag/release. [H5 source and acceptance ledger](accepted-baseline/TODO.md); continued under the same authorization with [H6 Q/K L2 source/rollback/physical acceptance](accepted-baseline/TODO.md). |

| U-048 | USER | Requested architecture/design validation of native kernels, speculative execution, retained replay and heterogeneous prefill, including relevant task/product capabilities and measured performance objectives. Preserve the one-Rust-engine/source/StateRoot/resource boundaries. This is design qualification, not authorization for downloads, machine tuning or an unmeasured speed claim. [Architecture and planned gates](../research/HETEROGENEOUS_PREFILL_AND_REPLAY_ACCEPTANCE_2026-10-15.md); [pipeline contract](../docs/HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md). |

| U-049 | USER | Requested removing external implementation/release references from this design update while retaining its technical ideas and design constructs. Generalize document names, text, links and acceptance identifiers; remove imported comparison/provenance/benchmark narrative; regenerate the archive and SHA manifest. Runtime and root README remain unchanged. |

## Strongly recommended architectural choices (not independently approved yet)

| ID | Status | Proposed choice and why |
|---|---|---|
| P-001 | PROPOSAL | Three capability levels: portable semantic reference; tuned generic kernels; extreme model-family/hardware specialization. Graceful per-operation fallback with state-layout compatibility. |
| P-002 | PROPOSAL | Three compiler layers: Semantic IR (exact/effectful model math), Optimized Execution IR (placement/fusion/layout/lifetimes), Device Plan (kernels/events/capture/scratch). |
| P-003 | PROPOSAL | Semantic IR represents recurrent and sparse state and its mutation/rollback as **effects**, not just pure tensor DAG. |
| P-004 | PROPOSAL | Rust-owned kernel registry and compiler selection with CubeCL as candidate, but keep independent CPU/Rust reference and allow target-specific Rust GPU intrinsic/backend work if CubeCL codegen is inadequate. |
| P-005 | PROPOSAL | Rust-authoritative scheduling with mixed prefill/decode/verify batches, bounded admission, latency/throughput modes, no CPU routing round trips in resident MoE hot path. |
| P-006 | PROPOSAL | One global budget arbiter, many specialized caches: dense/experts/KV/recurrent/indexer/lookup/scratch; zero assumption that GPU-visible GTT is truly available RAM. |
| P-007 | PROPOSAL | GPU route-aware MoE grouping across sessions and speculative rows, overlap-safe expert slot management, heterogeneous placement, no universal cache policy. |
| P-008 | PROPOSAL | Persistent, versioned prefix-state snapshots including KV, GDN/KDA, indexer, n-gram history, multimodal positions, speculative frontier; COW where beneficial. |
| P-009 | PROPOSAL | Speculation algorithms behind an exact verifier: native MTP, DSpark where model-compatible, EAGLE/draft model, prompt-lookup, n-gram and disabled; adaptive policy based on accepted **committed** tokens per full cost. |
| P-010 | PROPOSAL | Strict default sampling distribution preservation when available; explicit opt-in for approximate/opportunistic sampling. Quantization changes never implicit. |
| P-011 | PROPOSAL | GGUF and Hugging Face safetensors readers, semantic model adapters, tokenizer/chat-template correctness; recognized metadata != runnable unknown architecture. |
| P-012 | PROPOSAL | Multiple device execution strategies: single-device, hetero CPU/GPU, pipeline, tensor parallel, expert parallel, prefill/decode disaggregation where economics justify. |
| P-013 | PROPOSAL | Rust core and kernel source; use platform GPU drivers, APIs and vendor math facilities via safe Rust wrappers when needed; avoid linking another *inference runtime* as core. |
| P-014 | PROPOSAL | Test-driven optimization with CPU/FP64 oracles, reference logits, checkpoint parity, actual device benchmarks, quality regression, controlled ABBA/BAAB tests; publish failed experiments. |
| P-015 | PROPOSAL | Keep front-end as single binary/SDK with OpenAI/Anthropic-style APIs, streaming, tools, reasoning semantics, observability and simple local CLI. |
| P-016 | PROPOSAL | Prefer clear modular Cargo workspace, no performance-sensitive per-token dynamic-dispatch overhead, few crates, minimal operational deps. |
| P-017 | PROPOSAL | A plan cache keyed by IR/kernel/driver/device/quant/shape/numeric policy; invalidated on relevant changes; no invasive speculative adaptation until measurement proves value. |
| P-018 | WORKING TECHNICAL BASELINE | Reconcile the two design programs with **SIR** state/math, **RIR** legal fusion/layout, independent **KIR** Rust kernel provider branch, **CIR** asynchronous resource/stream/IO command graph, validated immutable ExecutableBundle; HardwarePolicy and StepPlan distinct from KernelPlan. This is a design decision from comparison/research, not a new user-specific API approval. |
| P-019 | WORKING TECHNICAL BASELINE | Adopt research's EngineCore single logical writer, bounded EngineMsg/EngineEvent, persistent InputBatch and StepPlan/WorkerDelta, CPU/IO worker isolation and typed mode receipt, validated against effectful model-state transactions. |
| P-020 | WORKING TECHNICAL BASELINE | Implement generic paged KvPool + chained hashes with optional radix index over same pool; extreme contiguous reserve-all pool only when measured, both under one ResourceArbiter; generalize hybrid prefix sharing to 3+ state-domain groups. |
| P-021 | WORKING TECHNICAL BASELINE | Keep GGUF as core input, optional derived acceleration pack, content-addressed ModelStore, precise tokenizer/pretypes/chat/grammar FSM, encoder/embedding/rerank and LoRA adapter capabilities; scope each release independently. |
| P-022 | WORKING TECHNICAL BASELINE | Rust kernel provider ABI independent of CubeCL; evaluate 2026 independent Rust CUDA tracks, AMD Rust gfx1151 ISA path and CPU/Vulkan/Metal fallback. Vendor driver FFI/non-Rust kernel source policy remains OPEN. |
| P-023 | WORKING TECHNICAL BASELINE | Use first-party independent ML compiler streams, schedule search, the MLIR effect model, attention-kernel serving autotuning and equality-saturation/optimization engines as *architectural research sources*, not required large runtime dependencies. Optimize pure regions offline with bounded compile time; no hot-path search. |
| P-024 | WORKING TECHNICAL BASELINE | Combine research frontier ring/CommitPlan with project full-state causal transaction; semantic, cache schema and async lease/timepoint effects must all remain valid across speculation/fallback. |
| P-025 | WORKING TECHNICAL BASELINE | Deliver a Stage-0 architecture/kernel/format/CPU vertical slice before freezing dozens of crates and advanced API; detailed crosswalk in INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md; no phase done without actual tests. |
| P-026 | WORKING TECHNICAL BASELINE | Treat XDNA2 and future NPUs as capability-probed **compute devices** with external runtime/driver and memory topology, not separate language model runtimes. RIR may partition semantically independent GEMM output rows; KIR selects an NPU GEMM provider; CIR schedules prepack, NPU/GPU concurrent work, event/fence join and full state commit. Missing NPU or unsupported shapes retain GPU-only fallback. |
| P-027 | WORKING TECHNICAL BASELINE | Treat upstream Gufo's distinct ASR/TTS/image/video work as **TaskProgram stage-graph research**: model/task-specific control loops share SIR/KIR/CIR/device memory but encoder, vocoder, denoiser and media outputs cannot all be forced into the decoder-only next-token loop. Maintain separately qualified capabilities. |
| P-028 | WORKING TECHNICAL BASELINE | Because the supplied Gufo fork reports hard GPU/NPU **machine hangs during fabric clock transitions**, its NPU path is experimental/opt-in and GPU-only is the safe default until a **specific hardware/firmware/XRT/ROCm/power configuration** has demonstrated stable concurrency, numerical quality and end-to-end win. Do not automatically modify privileged fabric clocks, IOMMU, NPU suspend or systemd configuration. |
| P-029 | WORKING TECHNICAL BASELINE | Keep **compute owner vs weight residency owner vs state owner** independent: an independent dGPU↔iGPU expert execution design, Gufo-style same-SoC iGPU↔NPU GEMM slicing, and optional CPU/SSD are different CIR placement candidates under one physical ResourceArbiter. |
| P-030 | WORKING TECHNICAL BASELINE | For NPU block-float compute, store source checkpoint precision separately from **temporary device compute encoding**; the source research's nine-byte/eight-value block-float example has quantization error and MUST be validated by per-operator/full-logit and model-quality tests, not merely 30 matching greedy outputs. |
| P-031 | **AUTHORITATIVE V3 DESIGN** | Replace source-project-specific architecture claims with **original zerogravity contracts**: ModelDefinition, SemanticOp/StateDomain, TaskProgram, legal RIR, independent KIR provider, CIR async resource graph, ExecutableBundle, DeviceCapabilities, ResourceArbiter and TaskCoordinator/StepPlan. Prior art documented separately and source-named examples optional only when root/file MIT permits. |
| P-032 | **AUTHORITATIVE V3 DESIGN** | Model semantics and task control are orthogonal; a TaskProgram can compose text, audio ASR, speech synthesis/vocoder, image/video diffusion, encoder/pool/rerank, agent workflows and future compute without a second device runtime or fake decoder-only loop. Each has typed iteration/frontier/output and independent model/task qualification. |
| P-033 | **AUTHORITATIVE V3 DESIGN** | New model/operator/quant/device/kernel/IO/source/task/frontend extensions register narrow versioned capability interfaces with independent reference/fallback/evidence and cannot require changing unrelated API/session/scheduler/memory ownership. Adopt minimal modules and a mandatory architecture extension drill. |
| P-034 | **AUTHORITATIVE V3 DESIGN** | License-aware prior-art catalog owns external project/author links, repository- and per-file SPDX verification, source revision and benchmark provenance. Normative plans use mechanism names; a MIT-licensed reference is a possible source candidate, not a mandatory dependency or blanket permission to copy bundled differently licensed code/weights. |
| P-035 | HISTORICAL V3 BASELINE (REFINED BY V4) | Typed SIR math/effects → legal RIR regions → KIR candidates → CIR async resources → ExecutableBundle runs **directly**, while task control coordinates stages/loops. v4 retains conceptual compiler stages but minimizes persistent interfaces/formats; StateRoot owner/physical budget ownership refined in P-036…P-045. |
| P-036 | **V4 ARCHITECTURAL BASELINE** | Exactly two enduring compilation representations initially: **SemanticProgram** and **ExecutableBundle** (containing async CIR commands). RIR and KIR are transient optimization/provider views; no requirement to implement four serialized IR dialects or hot interpreters. |
| P-037 | **V4 ARCHITECTURAL BASELINE** | StateDomains prepare immutable candidate versions; only **TaskCoordinator per session/shard** performs one logical `PublishRoot` after all state domains and physical events validate. Device/CIR commands report ready/failed, never publish semantic session state. |
| P-038 | **V4 ARCHITECTURAL BASELINE** | Maintain **coarse global physical memory admission/reservation**, with fast device-local slab/page suballocators under generation/event-backed leases; no global per-token allocation lock or impossible promise that the OS cannot change available memory. |
| P-039 | **V4 ARCHITECTURAL BASELINE** | A small trusted built-in `TaskDriver` controls AI stage loops and typed outputs; no general workflow DSL or `ExternalAction` with filesystem/network privilege inside core. External RAG/agent/tool orchestration belongs above the TaskRequest boundary. |
| P-040 | **V4 ARCHITECTURAL BASELINE** | **Async-first device-native StepProgram** with persistent parameter/row buffers, routing/sampler/draft verification and output event transfer where supported, avoiding mandatory CPU↔GPU synchronization for every model step; host-loop reference/fallback remains correct. |
| P-041 | **V4 ARCHITECTURAL BASELINE** | Typed fixed-rank bounded dynamic shapes, ragged row layouts, runtime kernel shape/quant/ISA/precision guards and bounded shape-band/compiled-bundle cache; use portable fallback on unsupported shape rather than unlimited JIT. |
| P-042 | **V4 ARCHITECTURAL BASELINE** | Output after committed root uses sequence/epoch identity, ordered at-most-once **delivery attempt** semantics; optional resumable cursor/ACK/durable store separately scoped. Distributed peers require epoch fencing and prepared cohort; no unqualified exactly-once network or instant multi-GPU physical atomicity claims. |
| P-043 | **V4 ARCHITECTURAL BASELINE** | Preserve a **future optional** `GradientProgram/TrainStep` and mutable `ParameterVersion` semantic/task seam without importing a compulsory autograd/tape/trainer into inference-first core; inference sessions pin immutable checkpoint/adapter versions. |
| P-044 | **V4 ARCHITECTURAL BASELINE** | Start with **one Rust library and one binary** (or 2–4 crates if GPU backend build/separation proven necessary); no mandatory nine-crate design, dynamic plugin framework or generalized external workflow engine. Require measured benefit to add persistent abstractions. |
| P-045 | **V4 ARCHITECTURAL BASELINE** | Specialized entire model/layer closed regions can replace generic math through one provider contract and compatible state/numerical/resource boundaries; disable specialized path and still run portable model/task with no new server/scheduler/allocator. |
| P-046 | **V4.1 BACKEND CONTRACT** | Generalize native `KernelProvider` into one `RegionExecutionProvider` seam supporting NativeKernel, CapturedCommandGraph, AheadOfTimeDeviceGraph and StaticallyScheduledDataflow. The existing KernelProvider is the native implementation subset, **not** another model runtime or an extra IR. |
| P-047 | **V4.1 BACKEND CONTRACT** | Let the provider declare **minimum efficient closed-region granularity**: operator, fused region, whole stage/model or partitioned fabric program. Device compiler owns micro-scheduling while Rust TaskCoordinator retains session task/admission/output/StateRoot, with no host instruction-by-instruction emulation of TPU/wafer/static fabric. |
| P-048 | **V4.1 BACKEND CONTRACT** | StateBridge qualification explicitly requires privately prepared state, checked snapshot/restore, bounded replay or stateless region. An opaque in-place stateful device program without these is **not** an interchangeable general spec/rollback backend; no device-side PublishRoot authority. |
| P-049 | **V4.1 BACKEND CONTRACT** | ProgramLifecycle records compile/build, upload/load, warmup, resident code/scratch, source/driver/firmware, statically constrained shape, cache invalidation, code/weight stream and program teardown; graph-only devices need shape-band fallback and no per-token recompile. |
| P-050 | **V4.1 BACKEND CONTRACT** | Model hardware **compute domains, memory tiers, links, compiler domains and micro-schedule ownership** independently: coherent LPDDR CPU+GPU, discrete HBM/GDDR GPU, multi-GPU rack peer fabric, graph-compiled TPU/NPU, on-chip SRAM/static wafer fabric and authenticated remote transport. Aggregate chip memory never implies common pointer addressability. |
| P-051 | **V4.1 BACKEND CONTRACT** | Separate FP4 variants with 16-value FP8 block scale and two-level tensor scaling vs 32-value shared power-of-two microscale, native FP8/MX and specialized low-bit layouts; source stored quant vs physical repack vs lossy compute/changed model quality remain distinct with full-task tests. |
| P-052 | **V4.1 BACKEND CONTRACT** | A managed remote inference API without exact compiler lowering and StateBridge is **TaskServiceAdapter interoperability**, not a native RegionExecutionProvider; explicit auth/privacy/egress/session limitation and no claim of native device execution. |
| P-053 | **V4.1 BACKEND CONTRACT** | Report qualification ladder per model/task/shape/driver: architecture compatible → SDK feasible → compiles → functional → state-safe → performance-qualified → production-qualified. Missing SDK/physical hardware/source-policy yields BLOCKED/INCOMPATIBLE, not a phantom supported TPU/wafer product. |
| P-054 | **V4.1 RESEARCH** | Distinguish Cerebras wafer-scale accelerator from an independently published Cerberus parallel-decoding algorithm; the latter is an optional DraftSource/TargetVerifier and full-state rollback candidate, not a new hardware backend. User referent remains unconfirmed, both are researched. |
| P-055 | **V4.1 MINIMALITY** | Add no permanent new semantic IR, task runtime or model-specific server. Use one RegionExecutionProvider and one ExecutableBundle physical program enum; optional closed region graph compiler and statically scheduled fabric tests cannot block real CPU/GPU vertical slice. |
| P-056 | **V4.2 DISTRIBUTED BASELINE** | One **PlacementPlan** independently binds ComputePlacement, ParameterPlacement, mutable StatePlacement, immutable CachePlacement, and derived TransferPlan across topology, TaskDriver phases and committed roots; placement is a *view inside ExecutableBundle*, not another enduring IR or model engine. |
| P-057 | **V4.2 DISTRIBUTED BASELINE** | Support colocated baseline, contiguous/noncontiguous layer groups with per-layer cache colocated, remote layer compute with state elsewhere via hydrate/remote-attention/direct-read variants, tensor/head/expert/sharded-sequence, P/D, FFN/attention disaggregation, remote prefix and multi-task stage placements; every mode per-op/source/math legal and costed. |
| P-058 | **V4.2 DISTRIBUTED BASELINE** | Node/LinkTopology capability includes compute/memory/storage/driver, network **effective directional bandwidth, RTT/P95 jitter, direct vs relay, MTU/loss, auth/crypto/egress**, plus physical budget and failure domain; runtime placement epoch frozen for active compiled region/collective. |
| P-059 | **V4.2 DISTRIBUTED BASELINE** | **TransportProvider** supports local IPC/peer/DMA, physically qualified RDMA, authenticated TCP/QUIC or an existing private encrypted tunnel; control and bulk data planes may choose separate paths. VPN/tunnel is not RDMA, a relay is not direct, and private network routing is not peer authorization. |
| P-060 | **V4.2 DISTRIBUTED BASELINE** | Minimal opt-in authenticated **PeerAgent** reuses the same binary/RegionExecutionProvider/DeviceBackend, advertising allowed model/work/resources and executing closed region/state transfers. No second server, arbitrary shell, network public exposure or automatic export of private prompts, activations, KV or model weights. |
| P-061 | **V4.2 DISTRIBUTED BASELINE** | Each mutable remote per-session/layer state shard has owner/lease/generation/schema/placement epoch references in the **single TaskCoordinator-owned StateRoot**; peers only prepare/ready/fail and are fenced on lost cohort/epoch. Immutable prefix cache locations may be eventually discoverable hints and never substitute for live state authority. |
| P-062 | **V4.2 DISTRIBUTED BASELINE** | Replan/migrate live computation/state only at legal committed TaskFrontier or op-defined checkpoint, with destination reservation, compatible verified snapshot transfer, epoch fence/lease move and old-lease retirement; failure preserves old root or replays safe checkpoint. No automatic transparent failover or exactly-once WAN output without explicit durable protocol. |
| P-063 | **V4.2 DISTRIBUTED BASELINE** | **Remote attention next to its KV/state** and independent FFN/experts are distinct from remote raw KV byte reads per token; choose based on measured network critical path and StateBridge. Slow/relayed tunnels normally veto chatty A/FFN or full remote KV reads, favor contiguous layer or one-time cache hydration. |
| P-064 | **V4.2 DISTRIBUTED BASELINE** | Optional **token-range attention shard** computes local softmax summaries (max, normalization, weighted values) on state-owning nodes and composes stable result; requires operator-specific PartitionAlgebra proving positions/masks/global routing/numerics. Ordinary recurrence/sparse global indexer not presumed partitionable. |
| P-065 | **V4.2 DISTRIBUTED BASELINE** | Layer pipeline colocates its layer-local mutable caches by default, reducing network to activation handoffs. Decode C1 serial hops can be slower; multi-chunk prefill/throughput and model capacity may benefit. Distinguish capacity benefit from single-request t/s gain. |
| P-066 | **V4.2 DISTRIBUTED BASELINE** | Separate control/metadata, bulk data and telemetry; scoped node identity, per-tenant/model/weight authorization, checked stream/transfer length, registered memory lease and content integrity, transport priority/backpressure, no arbitrary remote executable shell or untrusted device program. |
| P-067 | **V4.2 DISTRIBUTED BASELINE** | Dynamic cluster membership/topology and state-owner **lease epochs are not the same**; no split brain and old coordinator writes after failed election; start fail-stop/manual peer recovery, defer durable quorum/consensus/autoscaler until product requirement. |
| P-068 | **V4.2 DISTRIBUTED BASELINE** | Joint placement cost based on real compute/shape/phase, model+state bytes, activation/collective/query payload, link P95/copy/crypto/transfer overlap, cold program/cache, failover, security and objective. Do **not** activate a remote machine merely because it has spare memory. |
| P-069 | **V4.2 MINIMALITY** | Standalone has **zero peer/cluster overhead** and no required network permission. Gradual proof from synthetic remote layer, 2 private peers with reliable transport, remote full-state cache, A/FFN/token shards on fast links; cluster DB, DHT and public super-node services not mandatory. |
| P-070 | **V4.2 VALIDATION** | Add optional S0-DIST-A…L and H56…H70 distributed fault/cost/network/topology experiments, preserving P00–P23; source implementation, model/quant, physical network direct/relay, causal state, quality, P95 and privacy required before functional/performance claims. |
| P-071 | **V4.3 ROUTER BASELINE** | **One built-in ModelRouter** sharing ModelCatalog/ArtifactStore, PhysicalResourceArbiter, TaskCoordinator, RegionExecutionProvider/DeviceWorkers with CLI/SDK/HTTP; no mandatory gateway proxy, second inference scheduler or independent model server for each checkpoint. Exact model choice happens on request admission, not on every token. |
| P-072 | **V4.3 MODEL IDENTITY** | Separate `RegisteredModel` (immutable artifact), versioned `ModelAlias/ModelProfile`, `ResidentModel` deployment/physical replica (may span nodes), and `TaskSession` full mutable StateRoot. No conflation of many catalog models, simultaneously resident physical models, same-model replicas or sharded one model. |
| P-073 | **V4.3 EXACT ROUTING** | Default model alias resolves an exact checkpoint/stored quant/tokenizer/adapter/profile revision and compatible TaskDriver/numeric contract. **No silent substitution of another family, quant or model** when overloaded; explicit opt-in policy aliases may select different models and must reveal actual source/model and fidelity. |
| P-074 | **V4.3 REPLICA ROUTING** | Same-model replica load balancing uses verified state/quant compatibility, sticky session/prefix locality, physical headroom, queue/P95 and tenant/peer data permissions. Stateful migration requires safe StateRoot snapshot/PlacementEpoch transfer at committed frontier, not random per-request model switch. |
| P-075 | **V4.3 RESIDENCY** | On-demand model load/verify/compile/warm and safe idle unload, keepalive/pin/preload/scale-to-zero as optional policies. Deduplicate concurrent loads and bound queue. Unlimited registered models are allowed; simultaneous resident models/replicas/contexts bounded by physical RAM/VRAM/OS reserve and distributed capacity, no automatic quant degradation. |
| P-076 | **V4.3 SHARED BUDGET** | Model-level coarse admission and resident eviction reuse PhysicalResourceArbiter/local pools; within-model native continuous batching stays in existing TaskCoordinator and DeviceWorkers. Router must not become per-token bottleneck or starve loaded interactive models during cold loading. |
| P-077 | **V4.3 PER-MODEL PROFILES** | Versioned typed configuration at global, model, alias, deployment, session and request scope, with deterministic effective-source precedence; immutable source/model semantics, hard security and resource limits always constrain overrides. Hot alias revision affects only future requests; active TaskSession remains pinned. |
| P-078 | **V4.3 SIMPLE UX** | Proposed one executable `zerogravity run <model>` / `zerogravity serve` with zero mandatory daemon/config/cloud account: local in-process run, loopback serving, safe auto-device/model plan, concise progress; `models list/ps/pull/inspect` and `config show --effective` provide progressive disclosure. Exact binary name/port subject to open policy. |
| P-079 | **V4.3 SOURCE POLICY** | Unknown/unavailable named source must return diagnostics/offer explicit import/pull, **never silently download a guessed model**. Explicit source/license/size consent and content-addressed atomic verification, no executable model repository code, no public remote peer use without authorization. |
| P-080 | **V4.3 OPENAI API** | One loopback-native API adapter supports a pinned, tested subset of `GET /v1/models`, `POST /v1/chat/completions`, embeddings once encoder qualified, optional completions and versioned Responses. Streaming/SSE/event/tool-call data, usage, stop and error semantics validated with actual SDKs; no blanket OpenAI feature parity promise. |
| P-081 | **V4.3 RESPONSES/PRIVACY** | Stateless Responses first; stateful `previous_response_id`, local `store`, tool execution or cloud built-ins only when separately implemented/authorized. Tool-call *generation* is output data; arbitrary tool execution is outside inference core. Conversation storage privacy-first and opt-in. |
| P-082 | **V4.3 API SECURITY** | Loopback-only default and native admin/control API segregated from public inference API. External bind requires explicit auth/TLS and tenant/model visibility, remote source/peer policy; no API field can bypass hard physical/numeric/tenant limits or fetch arbitrary media URL/local files. |
| P-083 | **V4.3 MULTI-MODEL COMPOSITION** | A TaskProgram may reference **multiple distinct model roles** (target/draft, encoder/projector, embedding/rerank, TTS/vocoder) with separate artifact identities but one coordinated resource/task budget; these aren't replicas or alternate target routing. |
| P-084 | **V4.3 CONFIG CONTROL** | Expose every valid override through discoverable typed model/global/request/profile controls and `inspect --explain-plan/config show --effective` with value/source/guard/fallback. Config revisions/driver/source changes trigger compatible drain/replan rather than unsafe mutation of live KV or model weights. |
| P-085 | **V4.3 VALIDATION** | Add optional S0-SERVE-A…J and H71…H86 tests for one-command local UX, multi-model load/eviction, equivalent-replica routing, exact alias/session consistency, memory pressure, config precedence, OpenAI Python/JS SDK and SSE, auth/privacy and distributed resource placement. None run. |
| P-086 | **V4.3 MINIMALITY** | `run`, `serve` and Rust SDK use **same library engine**. Router owns model request resolution/admission and does not re-tokenize, JIT, marshal or run a second token scheduler per output step. A separate process may later be a security fault-domain option, not the mandatory architecture. |
| P-087 | **PUBLICATION/DOC AUTHORITY** | Provide a public `docs/WHITEPAPER.md` (problem, proposed original architecture, related work, research limitations, falsifiable hypotheses) separate from an integrated normative `docs/ENGINEERING_SPECIFICATION.md` (ES-00…ES-15 exact owners, protocols, tests). The external research/engine names remain properly attributed only as prior art. |
| P-088 | **PUBLICATION/DOC AUTHORITY** | `docs/README.md` is the organized audience-specific document index; `.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md` is the concise agent entrypoint; latest user decisions and live verified code/tests supersede historical addenda, whose full records must be retained. Do not silently rewrite older source history to remove evidence. |
| P-089 | **AGENT DELIVERY** | Maintain `.agents/plans/AGENT_IMPLEMENTATION_PLAYBOOK.md` with bounded P00–P23 tickets, exact Phase0/negative-test and hardware/source/privacy ownership, handoff template, and `REQUIREMENTS_AND_TEST_TRACEABILITY.md` mapping user U requirements to ES/INV/phase/proof without invented test successes. |
| P-090 | **PUBLICATION SAFETY** | Whitepaper is a **proposed design**, not a benchmark-results paper or proof of universal hardware/model support. Provide numbered primary source citations and publication checklist, with unresolved byline, repository/document license, DOI and independent peer review required before final public distribution. No PDF/export or results artifacts promised without separate request. |
| P-091 | **FINAL REVIEW PROCESS** | Run structural Markdown/link/anchor, core source-name neutrality, stable U/P/O IDs, P00–P23 and H01–H86 plus Stage0 coverage, and exact SHA-256 manifest **regenerated last**. Any failure must be fixed rather than marked passing; no source/model/device/remote operations occur as part of design-only review. |
| P-092 | **PACKAGING BOOTSTRAP** | Use **one dependency-free Rust 2024 binary** pinned to available Rust 1.95.0, with `--version`, `--help`, and honest nonzero refusal for reserved `run/serve/models/inspect/config` until implementation. Never supply a fake HTTP service or model output merely to make packaging pass. |
| P-093 | **PINNED CRANE BUILD GRAPH** | Reuse the trusted *pattern* of pinned nixpkgs/crane/rust-overlay flake, separate dependency/build/test/clippy/fmt outputs, NixOS module and six Linux MUSL/Windows GNU+LLVM-MinGW/macOS Apple SDK cross target definitions. An unexecuted Nix cross derivation is not a verified platform build. |
| P-094 | **RELEASE AND INSTALLER BOUNDARY** | GitHub Actions checks/builds six assets, validates SHA-256 and exact Cargo version, creates a release only on matching v* tag; installers select supported OS/arch, verify sidecar/version and atomically replace the user binary. No default GitHub organization is invented or network/download performed by bootstrap. SHA sidecar is not a signature. |
| P-095 | **NIXOS MODULE HONESTY** | Expose `services.zerogravity.enable` disabled by default, safe journal/DynamicUser and future service options; since `serve` is not implemented, an explicitly enabled unit exits clearly and does not restart-loop, without claiming a functional daemon. |
| P-096 | **BOOTSTRAP SCOPE AND STATUS** | Record P00 tooling/binary subset as PARTIAL/IN_PROGRESS, no completed reference model or inference engine. Preserve source/design docs and agent handoff, do not change read-only dotlink/research, leave Git init/public release/license decisions separate. |

## Open decisions (do not fill silently)

| ID | Status | Question / recommended validation |
|---|---|---|
| O-001 | OPEN | What license for zerogravity? MIT/Apache-2.0 dual licensing is a *candidate*, not selected. Verify copied code license, attribution and weight redistribution separately. |
| O-002 | OPEN | Which accelerators must be production-grade at v1? Candidate first targets: Linux x86-64 CPU and AMD Strix Halo, then CUDA, Apple Metal, Vulkan; final order is not approved. |
| O-003 | OPEN | Exact interpretation of entirely Rust: are system CUDA/ROCm/Metal APIs, HIP/BLAS libraries and compiler-emitted non-Rust machine artifacts acceptable? Proposal: Rust-maintained source with driver/runtime FFI allowed; prove performance before forbidding vendor libraries. |
| O-004 | OPEN | Select primary GPU kernel toolchain after comparative proof (CubeCL, Rust GPU/AMDGPU, compiler extensions, vendor GPU APIs). No framework chosen definitively. |
| O-005 | OPEN | First complete model acceptance target: basic Llama-style dense vs Qwen3.5 hybrid vs Qwen Flash-Next vs GLM; phase gates recommend dense then a hybrid and giant MoE. |
| O-006 | OPEN | Which external format/quant combinations are promised at initial release? Capability matrix must be evidence-based, not optimistic. |
| O-007 | OPEN | How much sampling reproducibility across backends/batch schedules is required? Strong guarantee should at least cover same-plan, same-seed, same-schedule. Cross-plan bit identity often impossible without speed cost. |
| O-008 | OPEN | Choose stable versioned API/ABI and whether to ship C ABI, WASM/browser, FFI from other languages. |
| O-009 | OPEN | Distributed authentication/transport defaults and remote execution scope; do not expose unauthenticated RDMA/TCP by default. |
| O-010 | OPEN | Supported deployment platforms and packaging (NixOS, Linux distros, macOS, Windows, ARM) and GPU availability for CI. |
| O-011 | OPEN | Quality policy for low-bit checkpoint selection and acceptable evaluation delta. Do not make default quality regressions invisible. |
| O-012 | OPEN | Service SLOs and max concurrency/contexts for each hardware tier; derive from measured P50/P95 and memory admission. |
| O-013 | OPEN | Should external model definitions be declarative DSL, Rust code, or both? Prefer explicit versioned semantic registry and safe loading. |
| O-014 | OPEN | Whether to allow third-party Rust-authored kernel/model plugins; assess untrusted code and reproducibility before accepting dynamic execution. |
| O-015 | OPEN | Whether/when to initiate a Git repository, CI, automated releases and model benchmark hardware matrix. |
| O-016 | OPEN | Exact application/project branding, executable name, CLI flags and public version semantics. Working examples are illustrative only. |
| O-017 | OPEN | Formal Rust GPU source policy: are driver/vendor math FFI and separately licensed non-Rust kernel source allowed? User asked Rust-authored owned engine/kernels; research vendor-kernel escape is **not** automatically approved. |
| O-018 | OPEN | Final target kernel toolchain per device after physical benchmark: CubeCL gfx1151, target Rust AMDGPU lowering; independent Rust CUDA options are new as of Sep 2026 and require separate maturity/security/ISA assessment. |
| O-019 | OPEN | Should exact binary name zg, dual MIT/Apache-2.0 license, Rust MSRV and package name from old research be adopted? They remain council recommendations, not explicit user decisions. |
| O-020 | OPEN | Versioned acceleration-pack public filename and cache format (avoid overloading APK), plus correct disk-KV header/schema replacing the historical invalid 48/54-byte spec. |
| O-021 | OPEN | Thresholds for eager/capture kernel tactics, strict replay vs cross-plan numeric variance, online resource-cost policy hysteresis, cache tenant privacy and model-quality pass bands: establish by tests. |
| O-022 | OPEN | Prioritize first accepted dense, hybrid, MoE, embeddings/LoRA and multimodal release cells; user specified eventual breadth but not v1 feature commitments or CI hardware purchases. |
| O-023 | OPEN | Can a **Rust-authored XDNA2/AIE kernel** and program be built with a supported compiler/toolchain and XRT FFI, without bundling Gufo's Python MLIR generator/C++ AIE kernel as owned source? This is separate from the already open driver-math FFI policy. |
| O-024 | OPEN | How to qualify **same-SoC GPU+NPU concurrency safety** across firmware/AGESA, power states, GPU fabric clock and ROCm/XRT versions without privileged automatic system tuning? Unqualified configurations must remain GPU-only; physical hard-hang experiments require isolated test hardware. |
| O-025 | OPEN | First supported NPU workload and precision policy: fixed 4096×2560×1024 `v8bfp16ebs8` Qwen Flash-Next prefill versus generic GEMM shapes or encoder work; benchmark actual critical-path gain and quality under controlled clock/power. |
| O-026 | OPEN | TaskProgram production scope: text generation, ASR, TTS, image diffusion/edit and video/audio must have individual model/driver/license/quality/release cells. Prior-art examples do not imply zerogravity already supports these. |
| O-027 | OPEN | Which exact MIT-licensed source files, if any, to reuse as code rather than independently implementing the mechanism? Confirm file/embedded code/model weight licenses, required notices, Rust-source policy and user review before each import. Non-MIT source remains design research unless explicitly authorized. |
| O-028 | OPEN | How to scope external source attribution in public product documentation and distribute an independent legal source/SBOM table without introducing runtime dependencies on upstream project APIs? Current v4 keeps provenance in research docs, not normative interfaces. |
| O-029 | OPEN | Does “all AI execution” require **training/fine-tuning** as a first-release capability or only an extensible future GradientProgram/TrainStep/ParameterVersion seam? v4 chooses inference-first with future seam, user has not approved a training release scope. |
| O-030 | OPEN | Which task kinds beyond generation/embeddings are release-qualified first (ASR, TTS, image/video, agent-compute)? Supported TaskDriver *type* does not imply actual model/source/data/quality readiness. |
| O-031 | OPEN | Exact delivery guarantees for disconnected/remote clients: default ordered at-most-once live **attempt**; optional durable cursor/idempotent replay requires separate retention/privacy/ACK policy. |
| O-032 | OPEN | Physical memory/sharding strategy under device allocation pressure: choose per-device pool and coarse global lease/reservation/compaction, tuning based on real hardware and task workload rather than freezing one allocator. |
| O-033 | OPEN | When to introduce public plugin/IR ABI or external stable model import schemas? v4 keeps built-in Rust providers and minimal versioned SIR/StateRoot until real extension/use and security needs justify them. |
| O-034 | OPEN | How much fully GPU-native sampled control/grammar/device-loop execution is feasible on each target without sacrificing exactness, cancellation and request output streaming? Determine per model/shape/driver from physical proof, not universal promise. |
| O-035 | OPEN | Which external graph/device compiler/runtime backends and foreign device programs can be used while honoring Rust-owned semantics/kernel-source user policy? Graph lowerer through a vendor compiler is **not** a Rust-authored native TPU/wafer compiler; must decide SDK/license boundaries before implementation. |
| O-036 | OPEN | Which TPU and other graph accelerators offer actual documented source-op/state/shape/graph program APIs sufficient for Rust host lowering, snapshot/provisional state and task-loop execution? Architectural support is not real qualified compiler integration. |
| O-037 | OPEN | Can wafer-scale/static dataflow or deterministic LPU/RDU hardware SDKs be accessed and trusted for direct compiled-region execution with state lifecycle, or is only a remote opaque model API available? Treat latter as TaskService boundary, not native region implementation. |
| O-038 | OPEN | Which physical accelerator to qualify after baseline CPU/Strix/CUDA: available GB10 coherent ARM/Blackwell, workstation/discrete Blackwell, B200/B300 rack HBM, TPU7x/6e, cloud AOT, wafer/static fabric, AMD CDNA etc? Do not assume user owns expensive hardware or cloud capacity. |
| O-039 | OPEN | Cloud accelerator remote compilation/model artifact/privacy/egress/tenant authentication and prohibited third-party code import policy; TaskService integrations need separate user authorization and are not a low-level vendor device backend. |
| O-040 | OPEN | Compiler region granularity and device-resident state bridge: which programs may keep static mutable KV/recurrence/optimizer across calls while still supporting prepare/abort, rollback and common logical StateRoot? Must prove per backend, not infer from vendor API name. |
| O-041 | OPEN | Was user’s “cerberus” intended as **Cerebras** wafer-scale hardware, the **Cerberus** adaptive parallel-decoding paper, or both? Preserve both research paths without making an unsupported hardware claim or demanding clarification before design work. |
| O-042 | OPEN | Exact multi-node first release topology: two trusted heterogeneous LAN/tunnel machines with contiguous layer+state colocation, remote prefix, P/D, A/FFN or expert-first? Design supports all; prioritize physical hardware and measurable application workloads, not arbitrary feature count. |
| O-043 | OPEN | Preferred private peer connectivity and transport: direct authenticated TCP/QUIC vs existing WireGuard/Tailscale overlay tunnels, local RDMA/Thunderbolt when physical, NAT relay behavior and trust/double-encryption cost. **No approval to connect machines or install a tunnel is implied**. |
| O-044 | OPEN | Node identity/provisioning/authorization and tenant/model/checkpoint sharing permissions, weight license redistribution and remote prompt/hidden-state privacy policy; default require explicit enrollment. |
| O-045 | OPEN | Topology discovery: manual peer inventory first vs mDNS/optional overlay discovery, dynamic link measurement frequency and peer capability attestation. No automatic use of unknown peers. |
| O-046 | OPEN | Remote state shard transfer schema for KV+GDN/KDA+sparse indexer/ngram/grammar/RNG/modalities and versioned state snapshots: choose portable wire codec, delta/COW and layout conversion only after actual models/quant reference. |
| O-047 | OPEN | Coordinator crash/re-election and durability tier: fail-stop with replay vs durable checkpoints/lease service/quorum, output cursor and exactly-once requirements. v4.2 promises no automatic transparent HA until implemented. |
| O-048 | OPEN | Placement solver sophistication: static simple heuristic vs bounded search and online hysteresis; performance varies dramatically by direct/relay RTT, context, shape, per-layer operator type, task/power and physical cache residence. |
| O-049 | OPEN | Network transfer implementation for GPU direct, RDMA/UCX-style native libraries and fast collectives under Rust-owned code/vendor FFI policy, with fallback buffered TCP/QUIC. A VPN cannot guarantee GPU RDMA. |
| O-050 | OPEN | Which attention operators support exact token-range PartitionAlgebra and stable softmax summary combine? Need source-equation and numerical proof; compressed/selected sparse and GDN/KDA recurrence are NOT automatically sequence partitionable. |
| O-051 | OPEN | When to disaggregate mutable attention KV from FFN versus migrate cache once, keep whole layer+cache colocated or read remote state bytes: need measured per-link cost and safe migration/readiness/replay. |
| O-052 | OPEN | Model/task weights/adapter/quant cache distribution and encrypted storage, remote source permission and content-addressed trusted reference, cache directory presence hints versus authoritative remote mutable state generation. |
| O-053 | OPEN | Private WAN-tunneled machine capacity mode vs home LAN vs rack/cluster behavior and expected SLAs—must qualify separately, rather than one universal default topology. |
| O-054 | OPEN | Final binary name (`zerogravity` vs shorthand), exact command aliases and localhost default port; examples are proposed, not a stable CLI approval. |
| O-055 | OPEN | Per-model default context, keepalive/idle unload, max concurrent resident models and parallel sessions: adaptive resource safety vs predictable user controls, not fixed arbitrary source-project limits. |
| O-056 | OPEN | Config on-disk format/location/schema migration, global/model/alias/request/CLI/env precedence and secrets handling; v4.3 proposes versioned typed TOML but final user setting policy remains open. |
| O-057 | OPEN | Model catalog trusted source registry list, automatic local alias discovery, explicit `pull` network consent, license/size/checkpoint pin and offline semantics; avoid silent remote download. |
| O-058 | OPEN | Which OpenAI API subset to release first beyond chat/models, including completions, embeddings, stateless Responses, stateful previous_response_id and streaming schema/SDK version; no blanket compatibility promise. |
| O-059 | OPEN | `run` process lifecycle: always in-process, optionally connect trusted same-user resident daemon/socket, explicit auto-service mode, NixOS/system service setup, startup and history retention; no silent system service installation. |
| O-060 | OPEN | Exact replica load balancer cost and stickiness: session StateRoot affinity, cache hints, predicted P95/TTFT, per-tenant fairness, model load churn; when to prewarm/create replicas vs move a session. |
| O-061 | OPEN | Policy-defined different-target `auto` alias: whether/how to expose optional automatic model selection by quality/privacy/cost while disallowing unrequested model/quant substitutions for exact requests. |
| O-062 | OPEN | Model profile editable parameters and default modes, including runtime/reload safety boundaries; which sampler/grammar/context overrides should be possible per request versus require new TaskSession. |
| O-063 | OPEN | Native administrative model-management API (local socket vs authenticated HTTP, administrative scoping) and public exposed `/v1/models` catalog/alias privacy; model source paths/hardware/peer topology must not leak. |
| O-064 | OPEN | Many-model physical residency and eviction policy under unified RAM/GTT, VRAM, per-model context state, adapter/shared weights and remote replication; exact pin/keepalive/scale-to-zero guarantees and cold-start SLA. |
| O-065 | OPEN | Advanced cross-model TaskProgram roles and router SLOs: target/draft/encoder/reranker/vocoder on independently placed qualified devices with one tenant admission budget; actual model combos need explicit task quality. |
| O-066 | OPEN | OpenAI Responses-compatible state storage and session retention, privacy and stream cursor contract vs stateless-only initial endpoint; server-side tool execution is outside zerogravity inference core. |
| O-067 | OPEN | Public author/byline, affiliations and citation contact for whitepaper publication; do not attribute the design to unapproved individual or organization names. |
| O-068 | OPEN | Public repository code license and whitepaper/documentation copyright/distribution license; current publication draft must not assume MIT/Apache or CC-BY for original zerogravity source. |
| O-069 | OPEN | Publication channel/URL, DOI or stable version identifier, visual template/PDF/HTML export and accessibility standard; markdown manuscript exists but no publication artifact was requested/produced yet. |
| O-070 | OPEN | Independent technical peer review, legal/trademark review and final owner signoff before public whitepaper release; design arguments are not verified runtime implementation benchmarks. |
| O-071 | OPEN | Future results edition requires exact model/quant/source/driver/hardware/network/task baseline, quality/sampling/p95/energy, raw reproducibility and negative results; no current benchmark evidence exists. |
| O-072 | OPEN | Final GitHub repository OWNER/zerogravity, initial source license and signed-release/provenance policy before public release. Installers deliberately require explicit repository or release base URL rather than guessing a host. |
| O-073 | OPEN | Final tested support matrix for all six cross-built targets; Apple SDK license/source access, macOS signing/notarization, Windows ARM64 toolchain and real per-OS binary smoke tests. Rust native Linux build alone is not proof. |
| O-074 | OPEN | When the real serving runtime exists, update `services.zerogravity` execution arguments, restart behavior, listen/credentials and mutable model directories; currently the module's `serve` command intentionally fails and the unit is disabled by default. |

## Assumptions / boundaries

- A-001 (SUPERSEDED): initial relative-path interpretation placed `./zerogravity` under dotlink l5's launch directory `/home/pvl/spaces/abird`. User decision U-013 explicitly overrides it: canonical root is `/home/pvl/spaces/abird/src/zerogravity`.
- A-002: Research is a 2026-10-10 **snapshot**; named model geometries, release support and benchmark numbers require source pinning and re-verification.
- A-003: Reference benchmark reports are *external author measurements*. zerogravity has not executed tests or proved superiority.
- A-004: Full general fallback requires a known semantic implementation for every operator, sufficient resources and an available execution backend; unknown Python custom model code is not magically executable.
- A-005: The user has not requested using dotlink to modify existing projects or publish a GitHub repo.
- A-006: No specific secret, account, network endpoint or user credential is part of this project design.

## v3 architectural supersession and provenance rule — 2026-10-12

**U-017 remains the user-mandated project-neutral/source-license policy; U-018 is the latest comprehensive review/minimality instruction.** Source-specific implementation patterns mentioned in older U-00x user motivations, P-001…P-030 assistant proposals and old source-council D-xx records are **retained as historic evidence**, not live project-named architectural ownership. They are governed by this v4 distinction:

1. **Original normative design**: `MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md`, `COMPOSITION_AND_EXTENSION_CONTRACTS.md`, `MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md`, `TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md`, v4 `ZEROGRAVITY_MASTER_PLAN_2026-10-10.md`, current roadmap, experiment and release plans. These use typed model/task/region/device/resource/service contracts; P-002's old three-IR abstraction was refined by P-018/P-035 and **minimized by P-036**. Prepared-root publication and hierarchical allocator supersede ambiguous v3 StateDomain/ResourceArbiter sketches.
2. **Named external sources**: `PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md` verifies root/file license evidence and URLs; Retained source investigations preserve attribution, source findings, decisions and reported benchmarks as non-normative research. Retired snapshot bodies remain in Git history. Exact model checkpoint and hardware identifiers can appear in validation matrices; they are not architecture type names.
3. **MIT exception**: named references and source-reuse candidates from verified MIT-licensed projects are allowed but are not required dependencies. Even an MIT repository can contain differently licensed subfiles/device kernels/model weights. Reuse remains subject to specific file license/NOTICE, source-language and user review. Non-MIT works provide attributed research and engineering ideas, not copied or named project-specific runtime interfaces by default.
4. **No composability exception for performance**: even a much faster optimized model device graph must implement a closed semantic region or TaskProgram stage through the same executor/state/resource contracts, never a cloned model-specific server.
5. **Future decisions**: source attribution/compatibility is a research record; user product/architecture acceptance needs explicit U-/D-style decision and independent model/hardware proof. No prior project's "resolved" option can overrule U-017.

**Policy goal:** external projects could disappear or change license tomorrow without invalidating any core zerogravity architectural interface. v4 adds the further requirement that an optional model/kernel/task specialization can be removed without cloning or corrupting the shared runtime.

## Conflict-resolution protocol

Write future decisions as `D-YYYYMMDD-NN` with source (user instruction/review), options, trade-offs, tests, conclusion, superseded IDs, and affected files. Never upgrade a PROPOSAL to USER on an assistant's own authority. Whenever a model/hardware benchmark contradicts a claim, record the measured evidence and change the claim.

| U-050 | USER | “Let's go, next phase”: resume implementation after the pipeline design/reference cleanup, preserving the single Rust engine and source/StateRoot/physical ownership boundaries. Next bounded H7 moves original GDN polynomial SiLU and output gating; source host exp/ln coefficients are the next separate numerical gate. Full build/lint/tests, trained source checks, AMD/NVIDIA physical qualification, rollback/quarantine and archived handoff/SHA updates precede an atomic local commit. README stays unchanged; no downloads, device changes or publication. |

| U-051 | USER | Requested the next implementation phase after H7. Continue H8 original-source GDN nonlinear coefficients through the existing Rust engine/provider and immutable source/session/resource ownership, with exact FP32 arithmetic and real trained AMD/NVIDIA physical qualification. Preserve the user README, record failures and acceptance, regenerate handoff/SHA, and commit verified work locally; acquisitions, system changes and publication remain unauthorized. |


| U-052 | USER | Requested the next implementation phase after H8. Continue bounded H9 source-exact attention RoPE and private current-token KV proposals through the existing Rust engine/provider, immutable source and independent session/resource ownership. Qualify actual trained AMD/NVIDIA operations, exact rounding and nonzero positions, rollback, admission and post-launch quarantine; retain failed experiments and independent source evidence, regenerate handoff/SHA and commit verified work locally. User README, download/device/publication restrictions remain in force. |


| U-053 | USER | Requested the next implementation phase after H9. Continue source-exact GPU attention through the existing Rust engine/provider and immutable source, independent session StateRoot and physical resource ownership. H10 qualifies bounded QK scoring before separately verified softmax/value stages, preserving exact source rounding and nonzero private prefixes; verify both trained checkpoints/cards, rollback/admission/quarantine and independent source evidence, then update archive/SHA and commit locally. README and acquisition/device/publication restrictions remain in force. |


| U-054 | USER | Requested the next implementation phase after H10. Continue original-source GPU attention through the existing Rust engine/provider, immutable source identity, private StateRoots and physical resource ownership. H11 qualifies padded attention softmax with exact exponential/reduction and two-round normalization, both trained checkpoints/cards, independent native captures, rollback/admission/quarantine, then updates the handoff/SHA and commits verified work locally. README and acquisition/device/publication restrictions remain in force. |


| U-055 | USER | Explicitly requested “let's go, next, let's finish up full gpu inference”, extending the preceding individually requested operator phases. Continue verified implementation through remaining attention/FFN/residual/embedding math, GPU dynamic activation quantization and a complete resident pipeline using the same Rust mathematical schedule and independent transactional StateRoots. Preserve original source bits/identity, fair physical workers/arbiter, actual completion, memory admission and quarantine; no selected-error CPU substitution or unsupported whole-GPU/performance claims. H12 adds source-padded value-cache accumulation and corrects a native-proven signed-zero corner in the former valid-only CPU schedule. Root README remains byte-identical; no models/SDKs, device privileges, push/tag/release authorized. [Full completion tracking](accepted-baseline/TODO.md), [H12 evidence](accepted-baseline/TODO.md). |


## Owner decisions after U-055 — consolidated 2026-10-17

These rows preserve the subsequent user instructions from the implementation
conversation. Earlier rows describe the state and authority at their own point
in time. Current implementation is tracked in the [plan index](TODO.md),
[accepted baseline](accepted-baseline/TODO.md) and individual plan TODOs.
U-067/U-068 supersede older archive/SHA/generated-handoff process duties;
U-069/U-070 leave runtime implementation paused during this documentation task.
Historical checkpoint labels may predate the rewritten Git graph: a recorded
approval is distinct from a currently resolvable Git object.

| ID | Type | Decision and current meaning |
| --- | --- | --- |
| U-056 | USER | Repeatedly continue verified GPU phases through full GPU inference, then enable native NVIDIA CUDA and AMD ROCm/HIP. Preserve one mathematical engine, original source identity, independent transactional StateRoots, physical admission, fair workers, actual completion and quarantine. A passed operator is not a full-model claim. The accepted original-model resident512-token scope and native CPP profiles are recorded in accepted-baseline; independent8K drift, larger contexts and held hardware remain open. |
| U-057 | USER | Include the current machine and SSH targets pvl-a1/pvl-x2 as the first optimization fleet: five physical GPU devices and the best paths for all three CPUs. Maximize demonstrated capability on these targets using measured, source-correct optimizations. Target discovery/SSH qualification does not qualify distributed inference or authorize model/SDK downloads, persistent configuration, driver/device privilege changes or held-device recovery. |
| U-058 | USER | Use subagents for implementation and independent work, parallelizing wherever efficient; the main agent primarily coordinates, gates, reconciles, decides and merges. Use disposable side worktrees for overlapping source work, preserve ownership, then merge only verified pieces and retire superseded work. One MAIN Cargo/target owner avoids build contention and stale cross-worktree results. |
| U-059 | USER | After the native-enablement checkpoint, do the workspace layout/structure redesign and then a thoughtful left-right concurrency pass before returning to GPU optimization. Add left-right where contention and immutable publication ownership justify it. Its later cold model-catalog proposal was challenged by the user because that area was not contended; the accepted result is no speculative migration. Future candidates need representative contention evidence and deterministic dual-replay/reader-lifetime/resource tests. |
| U-060 | USER | Remove the independent probes/ directory by relocating reusable implementation to crates/, qualification tools to tools/ and appropriate tests to their owners. Consolidate the workspace/lock/version and preserve the sole product CLI. Clean only merged/superseded side worktrees. The old29.7GB build cache and historical evidence binaries need not be preserved; prior resnapshot/archive directions were subsequently superseded by U-066/U-067. Root README remains unchanged. |
| U-061 | USER | Immediately after the concurrency pass, implement explicit model acquisition from GitHub and Hugging Face into the XDG cache, then cover the finite applicable pull capabilities of pinned llama.cpp/vLLM. Preserve source authority, complete required closure, offline-before-credentials, ownership/admission/cancellation and the single native transport/cache/loader boundaries. Additional provider/profile support requires its own tests; SDK/plugin delegation does not imply native equivalence. |
| U-062 | USER | In parallel with independent checks, begin Qwen3.8 27B, Gemma4 E2B and Gemma4 26B model support. First bind source/config/index/tokenizer identities and basic source-correct CPU execution, then GPU. Existing delivered contracts are not trained-model inference; no new model or SDK downloads are authorized by the model request. |
| U-063 | USER | For Qwen, finish basics first, then match every applicable capability and optimization of the pinned llama.cpp/vLLM union, including public and internal surfaces. Afterwards add appropriate MTP, speculative decoding, n-gram/suffix, DSpark2/DFlash2 and related strategies under target-authoritative state/distribution correctness. Exact primary identity/version for DSpark2 remains unresolved; provisional frontier counts are not an exhaustive parity claim. |
| U-064 | USER | Use concurrency primitives thoughtfully, not to fill cheap places arbitrarily. Specifically the model catalog is not demonstrated contention. Preserve a simple existing lock where it fits; performance claims require measured benefit, correct publication/lifetime ownership and minimal maintenance cost. |
| U-065 | USER | Requested commit/push at earlier checkpoints. Those particular actions do not authorize later publication. The archive split proposal initially selected verified ordinary-Git parts rather than LFS, but was superseded by the owner-approved checkpoint/evidence deletion decision. Current AGENTS requires explicit authorization for each push, force-push, tag, release, deployment or destructive Git action; local verified atomic signed commits remain the normal implementation workflow. |
| U-066 | USER | Completed phases may use a singular owner-approved checkpoint as accepted truth; historical evidence archives and binaries are unnecessary and can be removed after preserving completed/tested lists and remaining work. The repository was subsequently rewritten. Preserve actual source/checkpoint/content identities and implementation tests, while distinguishing old approval labels from reachable history. The earlier preference to retain a future evidence model was superseded by the later simpler TODO instructions. |
| U-067 | USER | Replace the entire handoff/evidence process with simple TODO.md files: plans/TODO.md indexes plans; plans/<plan>/TODO.md records provenance, committed implementation/tests and current unfinished work; runs/<plan>-<session-or-agent-id>/TODO.md holds each agent's own work. On completion the coordinator atomically integrates useful results and unfinished items into the main plan; completed plans get one checked index line. Carry unfinished items forward after every commit; archive only long completed Markdown TODO sections. Delete obsolete evidence records and historical archives after useful context is retained. |
| U-068 | USER | Remove SHA inventories and staged tree/blob receipt requirements from evidence logs. Actual implementation tests prove behavior; provenance alone does not prove commands or outputs. A test item must state its intended inputs/behavior and be checked only after execution. Git commit provenance and ordinary resumable TODOs replace mandatory per-file receipts. Actual model/content hashes, integrity checks and numerical acceptance contracts remain required where they are part of implementation. |
| U-069 | USER | After the latest interruption, wrap the current phase, revalidate live source/process/worktree state, then pause. Quant selection finished at signedf197c9c after975focused/fourstrict/full934Rust+18namespace/originalmodel passes; two source-only filters/Xet-index drafts remain unfinished and unmerged. All implementation agents stopped. Only routine rolling TODO updates remained dirty before this handoff documentation request; nothing was pushed. |
| U-070 | USER | Create a comprehensive Markdown handoff and ready-to-copy next-agent starting prompt covering context from the beginning: initial design, specifications, user decisions, implementation steps, real verification scope, all pending work and exact resumption instructions. Record the handoff and its retained context under .agents/docs or .agents/plans, preserve root README, and identify any irrecoverable originals honestly. This is a documentation request; it does not resume the paused implementation or reinstate obsolete evidence/SHA/archive automation. |

````

## Source: .agents/plans/archive/history/OPEN_CONTRIBUTION_AI_DEVELOPMENT_AND_MIT_POLICY_2026-10-15.md

Original UTF-8 bytes: 4268; SHA256 `bf208cd7fb2fc5d83de2c38079cd30065365b0b95cdb5e27b5200e39134b0421`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — Human-Centric Design, AI-Centric Implementation, and Open Contributions

**15 October 2026 · Public contributor policy.** User-approved README, MIT and contribution guidance; no inference implementation change.

## 1. How we build

**Zerogravity is deeply human-designed and extensively AI-implemented.**

- Humans own the vision, architectural boundaries, tradeoffs, technical decisions, and acceptance standards.
- AI agents assist with source research, implementation, testing, benchmarking, review and documentation.
- We do **not** let autonomous agents drive architectural direction unchecked, and do **not** claim a human manually reads every generated line.
- Source correctness, reproducible tests, reviewed design contracts, independent real model/hardware evidence and transparent provenance are our quality controls. AI output is not proof on its own.
- The development process is open to contributors regardless of whether they use AI tools; collaboration is judged on thoughtful design, correctness and evidence.

The README states this succinctly as **human-centric design, AI-centric implementation**.

## 2. Contribution workflow

The user explicitly wishes to build Zerogravity publicly together with everyone. Invite friendly architectural debate, questions, source research and small well-contained improvements.

1. Open a **design/architecture issue** explaining the problem, source references, constraints and proposed approach.
2. For nontrivial code, contribute a concise plan under `.agents/plans/<topic>.md` FIRST: alternatives, affected interfaces/files, existing invariants, source/numeric/physical safety, tests, and fallback/rollback.
3. Send a focused implementation PR with source references, tests and relevant performance/hardware evidence. Avoid unrelated rewrites.
4. Update the plan/evidence when a finding changes. Source-free mocks are not original model tests; partial GPU kernels are not full model inference.
5. Documentation fixes and small typos can go straight to a PR. Negative benchmark findings and better simpler designs count as valuable contributions.

No requirement that contributors personally author every line. Never overwrite concurrent agents' staging, and avoid hidden model/server backends that duplicate core Rust ownership.

## 3. License selection

- The user approved the **MIT license** for the Zerogravity project; see root `LICENSE` and Rust package manifest `license = "MIT"`.
- Copyright (c) 2026 Prasanna Loganathar and contributors.
- Model weights, processors, external assets, dependencies and imported source retain their independent licenses. MIT does not automatically relicense third-party code.
- For external source reuse, pin the upstream revision, inspect per-file notices and provide attribution/licensing evidence; architectural study does not imply source import authorization.

## 4. README and implementation evidence policy

Root README is a **product entry point**, not the running implementation log. Include the exact user tagline, clear text/image/video/audio/realtime ambitions, minimal highlights, simple local quick start, compact fabric and joint optimization diagrams, FAQ, high-level honest status/roadmap, contribution invite, AI disclosure and MIT license.

The root-README-linked `.agents/research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md` remains a navigation pointer to current build/release/Nix guidance and plan TODOs. The old snapshot body remains in Git history. Current plan checklists and verified code/tests govern implementation status.

No proposed architecture, benchmark or generated explanation establishes support for original trained image/audio/video, full GPU Qwen or production vLLM-class serving absent source-real tests.

## 5. Decision scope

The license/AI disclosure/contribution approach is **explicit user direction**, unlike proposed future model ABI and scheduling options. User authorization here concerns documentation, repository license metadata and public-facing contributor guidance—not driver permissions, downloaded model weights, external provider account access or live deployment.

Maintain original P00–P23 phase order, one Rust engine, one resource authority and one TaskCoordinator publisher.

````

## Source: .agents/plans/archive/history/REQUIREMENTS_AND_TEST_TRACEABILITY.md

Original UTF-8 bytes: 20924; SHA256 `30f9b3ae2108d2d311bb1ae012ce5a64ebc9901ff3dd3d96ca6341e3e1a1139e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Zerogravity — requirements and test design

This is a reference mapping of requirements and intended tests. Implementation
status and actual results live in [the plan TODO index](TODO.md) and its plans.

## 1. Explicit user intent U-001…U-026

| User ID | Requirement interpreted without overclaim | Authoritative design | Primary acceptance/work |
|---|---|---|---|
| **U-001** | Project name and independent Rust-owned identity | [Whitepaper](../../docs/WHITEPAPER.md), [Engineering Spec ES-00](../../docs/ENGINEERING_SPECIFICATION.md) | P00 project metadata; do not rename silently |
| **U-002** | Maintain complete agent-readable plans and decisions | [Agent Playbook](AGENT_IMPLEMENTATION_PLAYBOOK.md), this matrix, [Index](../../docs/README.md) | Docs integrity/manifest, complete phase handoff; path superseded by U-013 |
| **U-003** | Rust runtime + owned kernel source | [ES-01/04/05](../../docs/ENGINEERING_SPECIFICATION.md), [Device codegen](../docs/DEVICE_KERNEL_AND_CODEGEN.md) | P00–P01 CPU build, GPU kernel source/provenance; vendor FFI O-003 remains open |
| **U-004** | Broad known model-family support | [ES-02](../../docs/ENGINEERING_SPECIFICATION.md), [Model Semantics](../docs/MODEL_SEMANTICS_AND_COMPATIBILITY.md) | P02–P03/P17 exact registered ops and model source vectors |
| **U-005** | Extreme sparse/hybrid large-model support | [ES-02/04/08](../../docs/ENGINEERING_SPECIFICATION.md), [Memory/IO](../docs/MEMORY_RESIDENCY_AND_IO.md) | P11–P14 model/reference/sparse/expert qualification on real feasible machines |
| **U-006** | High performance alongside simple usability | [Whitepaper §3/9/11](../../docs/WHITEPAPER.md), [ES-09](../../docs/ENGINEERING_SPECIFICATION.md) | P07–P09/P16/P22; benchmark direct run/serve + C1/8/32 P95 |
| **U-007** | Fallback from extreme to general execution | [ES-04](../../docs/ENGINEERING_SPECIFICATION.md), [Composition](../docs/COMPOSITION_AND_EXTENSION_CONTRACTS.md) | P05/P08 forced specialized kernel disable; no new server |
| **U-008** | Learn comprehensively from external engines | [Prior Art Atlas](../prior-art/RESEARCH_SOURCE_ATLAS.md), [References](../../docs/REFERENCES.md) | Reproducible source attribution, file license review; not copied engine |
| **U-009** | Semantic IR with high-performance specialization | [ES-02/04](../../docs/ENGINEERING_SPECIFICATION.md), [Compiler](../docs/SEMANTIC_IR_AND_COMPILER.md) | P04–P05/P08/P20 two enduring representations, no hot SIR walker |
| **U-010** | Prior research-only, later authorized design/docs | [Status](CURRENT_STATUS_AND_TRACKING.md), [Playbook](AGENT_IMPLEMENTATION_PLAYBOOK.md) | No code/Git/driver/peer work without user authorization |
| **U-011** | Preserve decisions/pending items/handoffs | [Decision Register](DECISION_REGISTER.md), [plan TODO index](TODO.md) | Retained decisions and plan TODOs cover requirements/pending work; Git retains removed historical records |
| **U-012** | Benchmark ambition not fabricated numeric deadline | [Whitepaper §11](../../docs/WHITEPAPER.md), [Numeric quality](../docs/NUMERICS_QUALITY_AND_BENCHMARKS.md) | P22 matched hardware/model/quant/task and raw evidence |
| **U-013** | Canonical path corrected to source tree | [plan TODO index](TODO.md), [README](../../README.md) | Existing files at canonical path, no obsolete active root links |
| **U-014** | Combine research corpus with semantic architecture | [ES-00 authority](../../docs/ENGINEERING_SPECIFICATION.md), [Prior-art policy](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md) | No duplicated model runtime, research tree read-only |
| **U-015** | Adaptable at all pipeline layers and across near-future technology | [ES-14 extension matrix](../../docs/ENGINEERING_SPECIFICATION.md) | P23 change-radius drill and H43/H55/H70/H86 |
| **U-016** | Explicit GPU+NPU partial prefill and multi-task paths | [NPU/Task Pipelines](../docs/NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) | Optional S0-NPU A–F, source/numeric/host-hang safety |
| **U-017** | Independent project-neutral architecture and license-aware prior art | [Prior Art License Policy](../prior-art/PRIOR_ART_LICENSE_AND_PROVENANCE_POLICY.md), [ES-10](../../docs/ENGINEERING_SPECIFICATION.md) | No non-MIT project-branded core interfaces; file-level source review |
| **U-018** | Minimal, high-performance composable architecture critical review | [v4 critical review](../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md), [ES-01/14](../../docs/ENGINEERING_SPECIFICATION.md) | S0-V4 A–J, two enduring representations, one lib+bin until justified |
| **U-019** | GPU/Blackwell, graph TPU and wafer/ASIC execution family coverage | [Accelerator Provider](../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md) | S0-FABRIC A–H; mock architecture vs real SDK/state-safe distinction |
| **U-020** | Multi-machine topology, separate compute/parameter/state/cache and tunnels | [ES-08](../../docs/ENGINEERING_SPECIFICATION.md), [Distributed Placement](../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md) | S0-DIST A–L; P19 and H56–H70 authorized real two-node validation |
| **U-021** | One-command local, built-in router, many models/profiles and OpenAI APIs | [ES-07/09](../../docs/ENGINEERING_SPECIFICATION.md), [Router](../docs/MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [CLI/API](../docs/ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md) | S0-SERVE A–J, P08–P10/P16, H71–H86 tested CLI+SDK |
| **U-022** | Final coherent design review and **publishable whitepaper + full LLM-agent implementation handoff** | [Whitepaper](../../docs/WHITEPAPER.md), [Engineering Spec ES-00…15](../../docs/ENGINEERING_SPECIFICATION.md), [Public Index](../../docs/README.md), [Publication Checklist](../../docs/PUBLICATION_AND_REVIEW_CHECKLIST.md) | Design/editorial review; original handoff/SHA process superseded by [TODO workflow](../../AGENTS.md); owner public release approvals OPEN |
| **U-023** | **Packaging-only Rust binary foundation**: README, Cargo checks, Nix/crane cross build, NixOS module, GitHub Actions releases, checksum installers; no AI runtime logic | [Bootstrap Handoff](platform-and-release/TODO.md), [Root README](../../README.md), current P00 | Rust **7/7** + release CLI + shellcheck/nixfmt/Nix parse and offline installer fixture PASS; **Nix full build, Windows/macOS, module evaluation and GitHub release BLOCKED/NOT RUN** |
| **U-024** | Begin Phase 1 real inference development, targeting Qwen3.8-27B and Gemma 4 26B-A4B-it as two source-correct families | [Engineering Spec ES-02/06](../../docs/ENGINEERING_SPECIFICATION.md), [Phase 1 handoff](accepted-baseline/TODO.md) | **PARTIAL:** Rust artifact/config/quant/math/state reference + 23 tests; **BLOCKED/NOT_STARTED:** real model tensors/tokenizer/logits/generation, GPU and Nix cross builds |

| **U-025** | **Prioritize Qwen3.5 0.8B and Qwen3.5 2B: first source-correct CPU, then CUDA, ROCm and Vulkan on both GPUs; Gemma 4 E2B, Gemma 4 26B-A4B and Qwen3.8 27B CPU first, GPU after** | [Current Qwen handoff](accepted-baseline/TODO.md), [Physical GPU probe](../../crates/zerogravity-gpu/README.md) | **PARTIAL:** actual CPU text inference; independent greedy 5/5 12-token (0.8B), 5/5 8-token (2B); 28 core tests+1 opt-in checkpoint oracle; real Q4_0 source projection matches CUDA/HIP/NVIDIA Vulkan/AMD Vulkan on physical GPUs. **NOT_IMPLEMENTED:** all-device full-model GPU decoding, StateRoot/device leases and remaining CPU model families |

| **U-026** | Comprehensive independent CPU numerical/per-layer/long-context qualification of both Qwen3.5 0.8B and 2B before full CUDA/ROCm/Vulkan inference | [CPU numerical and context handoff](numerics-and-context/TODO.md) | **PARTIAL/BLOCKED:** real 1K 0.8B and 512 2B prefill reference parity; guarded private state/rollback, synthetic 32K operator tests; **2B strict numerical failure**, no full-model 4K..32K, no per-layer independent source parity, no GPU model inference |
| **U-027** | GPU-development admission after both Qwen3.5 CPU models independently qualify through 8K; defer but do not pass/drop 16K/32K local tests | [Phase 1 GPU gate and P07 integration handoff](numerics-and-context/TODO.md), [CPU gate](../../scripts/cpu-qualification-gate.py), [Accelerator region](../../crates/zerogravity-core/src/accelerator.rs), [Real model parity tests](../../crates/zerogravity/tests/qwen_real_model.rs) | **CPU Phase 1 PASSED 16/16** (both source-model 8K, short numerics, independent layer captures and state). **Extended context DEFERRED 16/20** (four 16K/32K still unqualified). **P07 STARTED**: shared Rust source-typed matvec dispatcher, explicit unsupported-only fallback and fail-closed GPU errors; real-checkpoint fallback/logit/state test PASSED on both models. Complete physical GPU inference NOT YET IMPLEMENTED. |
| **U-028** | GPU provider development and independent 8K numerical accuracy investigation, no threshold relaxation | [8K drift report](numerics-and-context/TODO.md), [source GPU Q4×Q8 provider](../../crates/zerogravity-gpu/src/q4_q8.rs) | Source Q4×Q8 host numerical parity and 3-token completed-region real-model tests pass on both models. 8K source/GDN layer divergence localized. Physical CUDA/ROCm/Vulkan provider inference NOT VERIFIED. |
| **U-029** | Three parallelizable implementation/research streams and multimodal architecture substreams, with independent agents only when authenticated | [v5 source design](../docs/UNIFIED_MULTIMODAL_EXECUTION_CONTRACT_V5_2026-10-14.md), [family atlas](../research/MULTIMODAL_MODEL_FAMILY_SOURCE_ATLAS_AND_IMPLEMENTATION_2026-10-14.md), [multimodal implementation checklist](multimodal/TODO.md) | Core architecture v4.3 preserved; proposed media clocks, compound models, non-token TaskDrivers, AV StateRoots, codec transforms, decisions; actual media inference not yet implemented. Standalone Codex/Claude tools unavailable due authentication; no false subagent work claimed. |
| **U-030** | Next phase: source-correct GPU P07-B Q4_1/Q8_1, Q8_0/Q8_0, Q6_K/Q8_K and F32 matrices with host parity, multimodal MM0 fake clocks/decision/flow/AV tests, and 8K GDN reference microbatch investigation | [P07-B handoff](accepted-baseline/TODO.md), [GPU mixed providers](../../crates/zerogravity-gpu/src/mixed_ops.rs), [MM0 tests](../../crates/zerogravity/tests/multimodal_mm0.rs) | Real checkpoint host arithmetic matched bitwise for all four 2B tensors and each source projection produced identical model logits/StateRoot when simulated as Completed. MM0 5/5 fake tests pass. Independent source 8K tensors: 0.8B ubatch256 vs 512 identical 158/158, 2B 149/158 differ (first layer1 GDN ~4.9e-7); causal drift is still under investigation. Actual GPU hardware execution NOT VERIFIED. |
| **U-031** | Bounded process-unique source-matched GPU weight buffers, failed-inflight fences, no repeated immutable H2D upload and complete GPU readback before StateRoot; no new numerical relaxation | [P07-C handoff](accepted-baseline/TODO.md), [residency policy](../../crates/zerogravity-gpu/src/residency.rs), [actual source provider](../../crates/zerogravity-gpu/src/resident_provider.rs), [core source ID](../../crates/zerogravity-core/src/accelerator.rs) | Host-tested local 0.8B and 2B GGUF sources each produce a single source weight upload and two cache hits in 3-token real forward, identical model logits and StateRoot. Bound/eviction/failed-upload/fence tests pass. CubeCL Vulkan builds; PHYSICAL GPU, asynchronous TaskCoordinator and GPU KV/GDN state are still unqualified. |
| **U-032** | P07-D source-weight decode-once, typed physical device-event admissions, cancellation/epoch fencing and fail-closed completion | [P07-D handoff](accepted-baseline/TODO.md), [StaticSource](../../crates/zerogravity-gpu/src/host_descriptor.rs), [event ledger](../../crates/zerogravity-core/src/device_events.rs) | All five GPU source tensor codecs host-verified and six device-event tests pass; physical async CubeCL execution not yet qualified. |
| **U-033** | P07-E actual CubeCL async readback completion bridge tied to pre-launch event/scratch admission; cancellation Drop fencing and source-model parity | [P07-E handoff](accepted-baseline/TODO.md), [bridge guard](../../crates/zerogravity-gpu/src/async_bridge.rs), [CubeCL async provider](../../crates/zerogravity-gpu/src/resident_provider.rs) | Async readback compiles, host Future-drop/lease tests pass, real pinned 0.8B/2B model event-mock parity and fail-closed error tests pass. No physically tested GPU kernel, no concurrent multi-stream task engine. |
| **U-034** | Recover P07-E, retain source/dynamic/output GPU CubeCL handles through uncertain cancellation, fence the entire service after dropped Future, and enforce static+transient per-provider GPU capacity without false physical qualifications | [P07-F handoff](accepted-baseline/TODO.md), [physical buffer quarantine](../../crates/zerogravity-gpu/src/physical_quarantine.rs), [event/handle guard](../../crates/zerogravity-gpu/src/async_bridge.rs), [budget](../../crates/zerogravity-gpu/src/residency.rs) | Optional GPU Vulkan source builds and strict Clippy pass, source-free GPU tests 10/10 and real Qwen3.5 opt-in GGUF host tests 8/8 pass. Native CPU Phase1 16/16 and deferred 16K/32K unchanged. Physical CUDA/HIP/Vulkan cancellation and global device scheduler remain NOT VERIFIED. |
| **U-035** | Validate live P07-F committed state, continue with P07-G control-plane, create full comprehensive phase-0-through-current handoff with original designs, all decisions, pending work and next agent prompt | [P07-G control plane](accepted-baseline/TODO.md), [ResourceArbiter](../../crates/zerogravity-core/src/resource_arbiter.rs), [fair device queue](../../crates/zerogravity-core/src/device_queue.rs), [full takeover document](TODO.md) | Live 602c246 P07-F baseline quality gate passed; P07-G1 4 resource tests + 4 queue tests pass. Comprehensive full-history handoff snapshot being generated. Physical CubeCL GPU model inference and source GDN accuracy changes remain unqualified. |

**New user requests after U-026 require a new U identifier and exact scope; do not silently reinterpret older decisions.** Root U-002 relative path was superseded by U-013 and must not be revived.

## 2. Operational core requirements and integrated invariant mapping

| Core contract | Engineering sections | Primary pass test (proposed; **not a file that exists yet**) | Related phases |
|---|---|---|---|
| Semantic source validation and exact op | ES-02; INV-01/02 | `test_artifact_source_and_unknown_ops`; reference tensors/logits | P02, P03, P06, P11 |
| No silent numeric/model change | ES-02/04; INV-02/23 | `test_quant_storage_compute_vs_changed_model`; route top-k | P03, P05, P12, P22 |
| Portable direct compiled fallback | ES-04; INV-02/09/10 | `test_kernel_guard_and_forced_fallback` | P05, P07, P08 |
| Prepared one-root logical commit | ES-06; INV-03/04/05 | `test_all_domains_prepare_then_single_publish` | P02, P04, P06, P15 |
| Token/grammar/draft exactness | ES-06; INV-05/06/23 | `test_target_distribution_and_rejected_branch_state` | P08, P15 |
| Resource leases and memory safety | ES-05; INV-07/08 | `test_delayed_dma_generation_and_pool_reservation` | P05, P07, P10 |
| Minimal TaskDriver and non-token output | ES-03; INV-11/20 | `test_text_and_encoder_task_reuse_one_engine` | P04, P17, P18 |
| Distinct model/alias/replica/session | ES-01/07; INV-12/13/14 | `test_alias_revision_replica_sticky_state` | P09, P10, P16 |
| Many-model concurrent admission | ES-07; INV-08/13 | `test_cold_load_dedup_and_idle_eviction` | P09, P10, P16 |
| Trusted API adapters and SSE | ES-09; INV-06/15/22 | `test_chat_streaming_python_js_qualified_subset` | P09, P17, P18, P21 |
| Exact peer/data/trust boundaries | ES-08/10; INV-15/16/18 | `test_peer_enrollment_and_wrong_model_reject` | P19, P21 |
| Remote full-state compatibility | ES-06/08; INV-04/16/17 | `test_distributed_prepare_fence_and_migrate` | P10, P19 |
| Token-sharded attention only when valid | ES-08; INV-19 | `test_partition_softmax_mask_numerics` | P11, P19 |
| AOT/static dataflow StateBridge | ES-04/08; INV-21 | `test_graph_only_stateful_bridge_or_reject` | P01, P05, P19 |
| Future learning seam without training bloat | ES-02/03; INV-24 | `test_parameter_version_inference_isolation` | P23 |
| Zero network/service tax for local run | ES-09; INV-13/25 | `test_run_local_no_daemon_http_or_tunnel` | P06, P08, P09, P23 |
| Full physical/measured performance evidence | ES-10/13 | `bench_exact_model_task_shape_driver_quality_matrix` | P20, P22 |

These are **suggested descriptive test names**, not an instruction to create test stubs without executable subject code. Each test must record source version, software/hardware, expected data, what fails, and whether simulated or real.

## 3. Phase-to-requirement closure map

| Stable phase | Primary ES sections | Minimum closure evidence | Notes |
|---|---|---|---|
| ZG-P00 | ES-00/01 | CPU-only clean Cargo executable/test and source policy | Small workspace first |
| ZG-P01 | ES-04/05 | real owned GPU kernel + CPU oracle or BLOCKED with reason | physical compiler optional for CPU |
| ZG-P02 | ES-02/06 | exact source op and provisional state reference | no hidden effect |
| ZG-P03 | ES-02 | bounded artifact/tokenizer/quant test vectors | no unsafe file parsers |
| ZG-P04 | ES-02/03/06 | effect-aware SemanticProgram + minimal TaskDriver | no general workflow |
| ZG-P05 | ES-04/05 | legal prepared region and CIR simulated lease/event | no SIR interpreter hot |
| ZG-P06 | ES-02/03/06 | real model CPU output and sampler/state | source-correct |
| ZG-P07 | ES-04/05 | real GPU event/buffer worker and model parity | physical safety |
| ZG-P08 | ES-03/04/06/09 | **one-command local real model output** with fallback | top viability gate |
| ZG-P09 | ES-07/09 | same-engine API catalog/chat/SSE + SDK conformance | no router engine |
| ZG-P10 | ES-05/06/07 | full-state cache + physical memory/residency | no active eviction |
| ZG-P11 | ES-02/06/08 | hybrid source recurrent/attention model quality | no fake partition |
| ZG-P12 | ES-02/05 | correct expert router/tier/IO | exact top-k |
| ZG-P13 | ES-04/05 | full complex model extreme region + fallback | baseline and quality |
| ZG-P14 | ES-02/04/14 | second complex model extension without server fork | change radius |
| ZG-P15 | ES-02/06 | target-authoritative speculation/rollback | target distribution |
| ZG-P16 | ES-05/07/08 | multi-model native batch/replicas/fairness | P95 and pressure |
| ZG-P17 | ES-02/03/07/09 | adapters, embeddings/rerank source math/API | real encoder |
| ZG-P18 | ES-03/09 | non-text media task source/quality | codecs, frontiers |
| ZG-P19 | ES-04/08 | trusted remote layer/state/transport + optional fabric | actual peer auth for physical |
| ZG-P20 | ES-04/05/10 | bounded offline autotuner/cost model and negative tests | model/task-level benefit |
| ZG-P21 | ES-09/10 | parser/auth/license/privacy build/release tests | no privilege surprise |
| ZG-P22 | ES-02/09/10/13 | matched model/quant/device/task quality+performance/API | raw reports |
| ZG-P23 | ES-14 | extension drill new op/backend/task/model/transport | no engine duplication |

## 4. Suggested milestone-level review gates

**M0 — Engine correctness feasibility:** source parser/CPU reference SIR, state root protocol, fake CIR and one native kernel source proof, no large framework scaffolding. **M1 — Useful local model:** an actual checkpoint produces a correct CLI response with safe local resources and general fallback. **M2 — Shared serving:** same engine behind model router/API, full-state cache, multi-model resident and safe users. **M3 — Advanced semantics:** hybrid/recurrent/MoE, speculation and specialized whole-layer kernels. **M4 — Task/modality breadth:** embedding/rerank, ASR/TTS/image/video only with real source quality. **M5 — Heterogeneous/distributed:** actual authorized multi-device/peer planning and proven state migration, AOT where compiler accessible. **M6 — Qualification and extension:** security, matched p95/energy and repeatable new model/device/task without engine rewrite.

The full Stage0 IDs, experiments and detailed model/hardware matrices remain in their [original files](INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md); don't flatten or renumber them to match this editorial organization.

## 5. Recording actual results

For each implemented item, record the relevant test command, actual result and
remaining failures in its plan TODO. Keep model/backend/context and numerical
limits precise; a mock or build is not physical inference. The phase checkbox
is complete only when its implementation and required tests pass. Git commits
provide source provenance. See [AGENTS.md](../../AGENTS.md) for agent integration
and the rolling Uncommitted section; no separate evidence ledger is required.

````

## Source: .agents/plans/archive/history/accepted-baseline/TODO.md

Original UTF-8 bytes: 3128; SHA256 `968831b1b3183e027222842127791b88509781adce2e08432acf0cc1eaef0a51`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Accepted implementation baseline

Provenance commit: `6d74f76`.
Owner-approved completion is recorded below; original reports are available in Git before their retirement.

## Uncommitted

None. Remaining features are separate plans in [the index](../TODO.md).

## Committed

### 6d74f76 — owner-approved tested baseline

- [x] Original Qwen3.5 CPU0.8B/2B functional Phase1:16/16 through genuine8K.
- [x] Complete original-model resident GPU pipeline and source/state/resource/cancellation boundaries through the qualified512-token scope.
- [x] CUDA local RTX3060/a1 RTX4060 and HIP a1 gfx1150/x2 gfx1151; SDK-free NVIDIA Vulkan is the fifth backend profile.135 profile checks plus12 host fingerprints accepted; this is not five fully qualified physical cards.
- [x] Nine-package layout, reference relocation and superseded worktree/cache cleanup; root README unchanged.
- [x] Qwen3.8/Gemma4 source contracts; no trained new-model inference.
- [x] Contention-first concurrency review completed; no left-right migration justified. Require measured contention and ownership fit for future proposals.
- [x] Completed historical evidence retired by owner authority. Preserve tests and technical designs; Git retains previous source/history.

Independent CPU8K GDN drift, larger contexts, held local AMD hardware, actual new
models, acquisition, serving and further optimization remain unchecked in their
own plans. This completed baseline does not close those tasks.

### Accepted detail retained from retired reports — 6d74f76

- [x] Original-model resident execution includes compressed embeddings, device activation quantization, GDN/attention/FFN and all42 private state domains; one mathematical schedule and compact token feedback.
- [x] State-only prefill, exact whole-range Contiguous aliases, prepared range/binding metadata and fused dense View certification are delivered. This is not driver graph capture, GPU tokenization or a general device scheduler.
- [x] Portable CPU/device v1 replay is qualified within its trusted diagnostic scope; authenticated persistence and full tokenizer/sampler/grammar/media serialization remain open.
- [x] Independent native-source512 checks compare146 stages/model: zero nonzero-FP32 differences and56 existing2B signed-zero distinctions. Internal full-logit/all-state/trace comparisons remain strictly bit-exact; independent full-vocabulary source equality is not inferred.
- [x] Main model-merge regression:135 ordinary Rust executions; separate23 original-model executions/22 distinct names, two exact eight-token CLI outputs and one official Qwen metadata case. Earlier phase counts are not additive accepted totals.
- [x] Native CPP profiles retain their measured scalar/libm/loader scope, including a1 gfx1150's explicit glibc2.44/GCC16.2 closure. Separate LLVM device code generation remains unqualified.

Approved original model content identities (still required for actual model inputs):

- Qwen3.5 0.8B SHA256: `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`.
- Qwen3.5 2B SHA256: `91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6`.

````

## Source: .agents/plans/archive/history/handoff-cleanup/TODO.md

Original UTF-8 bytes: 3564; SHA256 `0d92b720ff0ece86e003d4235dde95683eae408bbf2def229d2839459130433e`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Simplify agent handoffs

Provenance commit: `bfe672c`.

## Uncommitted

None for this completed plan. Continue [model acquisition](../model-acquisition/TODO.md).

## Committed

### bfe672c — simple TODO handoffs

- [x] Replace the handoff process with the owner's plan/run TODO hierarchy.
- [x] Define per-commit completed sections and a rolling unfinished checklist in root AGENTS.md.
- [x] Remove obsolete archive/generator/checksum scripts and their check.sh calls; preserve implementation tests.
- [x] Consolidate all material pending work into the plan index and per-plan TODOs.
- [x] Retire redundant handoff/status/provenance records and fix live navigation.
- [x] Reconcile independent agent checklists into this plan in one coordinator update.
- [x] Validate links, preserved architecture/source/root README and full repository checks.
- [x] Commit the cleanup, record its commit and finish this plan in the index.


### 3787519 — starting implementation

- [x] A1 cache implementation and tests exist. This cleanup changes only
  work tracking, documentation and obsolete tracking tools; it adds no inference
  capability and changes no mathematical engine or artifact implementation.

## Integrated agent results

- Plan inventory: nine plans retain all material P00–P23 work and original
  numerical/source/model/hardware limits. A1 is complete; A2/A3 are next.
- Tooling: removed five archive/generator/SHA tools and their four check.sh
  invocations. Exact comparison retained every real implementation check.
  Shell syntax, ShellCheck and AST parsing of18 remaining Python scripts passed.
- Preservation review: generated aggregate duplicated191 individual sources.
  Removed that aggregate and generated manifests; kept substantive current
  design documents and static V2/V3/multimodal historical versions. Authored
  retired handoff/status bodies remain available in Git at `3787519`; old
  entrypoint paths are small compatibility links to the TODO index.
- Navigation: all links in live index,11 plan TODOs, AGENTS.md, contributor
  guide and public document index resolve. No Rust, Cargo, GPU tools, model
  fixtures, Nix, CI or root README source has changed in this cleanup.
- Old A1 evidence recorder/patches/receipts are retired. Its actual194-test
  completion, publication/recovery cases and important limits are retained in
  model-acquisition/TODO.md and the A1 technical reference.

## Final checks

- [x] `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`: exit0;194 Rust executions passed,27 optional executions ignored. Strict Clippy, release build, installer, shell/Nix formatting, reference self-tests and18 native-wrapper fixtures passed.
- [x] One-off local-link check of35 current/new/modified documents: all targets resolve. The first pass found seven old pre-layout source paths in requirements; corrected to their crate owners and reran successfully.
- [x] Both independent final reviews: all11 plans preserve scope and pending work; all44 links in the13 TODO/workflow documents resolve. No review blocker.
- [x] Root README bytes and all crates/tests/tools/Cargo/Nix/CI sources match `3787519`. Only tracking docs, attributes and obsolete tracking scripts changed.
- [x] Finished per-agent run notes are integrated above and retired. The next agent starts from the plan index, not those completed working files.

Next implementation: model-acquisition/TODO.md — A2 shared bounded HTTPS,
pinned Hugging Face/GitHub resolution and explicit CLI, then A3 loader bridge.

````

## Source: .agents/plans/archive/history/historical-records/TODO.md

Original UTF-8 bytes: 10493; SHA256 `2ac60f20d2823af7d309c7867a3cb2cb64469dc42d6f82c011a834b655c3ecda`.
Resolve embedded relative paths against the original path, not this volume.

````text
# TODO — historical records

Status: done
Provenance: 7e61cb6

Retire superseded historical bodies after preserving distinct useful design,
source attribution and open implementation/test gates. Earlier completed groups
retain their pre-rewrite commit labels; this continuation starts at the rewritten
HEAD above. The root README and implementation remain outside this change.

## Committed

### aeff567 — retire legacy evidence reports

- [x] Translate completed evidence into concise committed checkboxes and real pending tasks in the existing plan TODOs.
- [x] Preserve useful source/numerical/ownership details in current technical references.
- [x] Delete obsolete phase reports, working notes and duplicate completion records after their useful content is integrated.
- [x] Update references to deleted records; keep root README and implementation/fixtures unchanged.
- [x] List historical archive deletion candidates, replacement references and any extraction still needed. Prepare only; preserve unique archives until this is done.
- [x] Run link/source checks, independent reviews and the complete repository check; record actual results briefly.
- [x] Commit this migration and carry unfinished archive work forward.
- [x] Identify the archive set and current replacement contracts; Git at `bfe672c` retains the original source history. Current specs/research and production/reference fixtures are outside this evidence deletion.
- [x] Remove the obsolete rejected-LR metadata JSON files; both were owned unchanged JSON with no active descriptor observed. Production/reference inputs and model/SDK/build caches are untouched.

#### Migration details

- [x] Retire 70 obsolete Markdown records: 42 GPU phase/working-note reports, 20 CPU/model/concurrency/checkpoint reports, six obsolete entrypoint aliases and two completed takeover/layout records. Their original contents remain in Git at `bfe672c`; no copies or new archives are created.
- [x] Integrate known accepted completion/results into accepted-baseline, model-execution and CPU/GPU TODOs. Preserve actual model content hashes, source512/internal-bit scope, native profiles and no-migration outcome.
- [x] Carry actionable 8K/source-schedule work, deadline/recovery, real physical memory release and future strided View provenance guards into current unchecked TODO items.
- [x] Preserve unique source rounding/quantization/padding and new-model axes/routing semantics in the current numerical/model technical references. Keep Gemma official revision URLs; remove obsolete result/evidence prose from that technical contract.
- [x] Redirect current documentation and test/tool runbook links to the plan owning the surviving results/tasks. Keep root-README-linked current-status and pre-product paths valid. Historical archive bodies are unchanged and their inbound/outbound link cleanup belongs to the queued archive step.

#### Validation

- [x] `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh`: exit 0; 194 Rust executions passed, 0 failed, 27 explicitly ignored. Strict lint, release build, installer, shell/Nix formatting, CPU/reference self-tests and 18 native-wrapper fixtures passed. No model/GPU launch or download occurred.
- [x] Current first-party navigation: 114 Markdown documents have no missing local targets. The three unchanged missing links in the vendored CubeCL README are outside this cleanup. Static historical archive/brief links stay queued for their removal pass.
- [x] Independent GPU review: 144 links across 23 selected TODO/spec/runbook files resolve; math/ownership/View guard extraction retained and no completed phase reopened.
- [x] Independent CPU/model review: preserve geometry/source/math details. Corrected two wording ambiguities: shard byte-count parsing refuses all exponent notation; WideF64 GEMV has no residual buffers, while separately qualified FMA operators use FastTwoSum correction.
- [x] All 15 historical archive/brief bodies and root README match `bfe672c`; all non-Markdown implementation/dependency/test-fixture/vendor/CI/Nix files are unchanged.
- [x] Ignored target top-level audit found no further old evidence after the two LR inventories were removed. Hardware discovery file listings were stale and are removed; source/model/reference/SDK/build caches remain outside deletion.
- [x] Finished agent findings are integrated here and into their owning plans; completed run files and temporary logs are retired. No new raw-evidence/archive/provenance pipeline is introduced.

### bfe672c — starting workflow

- [x] Single plan index, per-plan TODOs and separate agent run TODOs exist.
- [x] New work uses these checklists; old duplicated evidence does not become a fresh verification requirement.

### 9239719 — retire historical archive bodies

#### Historical archive retirement

- [x] Retire `DESIGN_EVOLUTION_V3_SNAPSHOT.md` after retaining useful rejected-design rationale in the [current critical review](../../prior-art/ARCHITECTURE_V4_CRITICAL_REVIEW_2026-10-12.md).
- [x] Retire `DESIGN_EVOLUTION_V3_HANDOFF_SNAPSHOT.md` after updating inbound links. StateDomains prepare; the current coordinator alone publishes roots.
- [x] Retire `PRIOR_ART_HISTORICAL_MASTER_PLAN_V2.md`, using [the current master](../ZEROGRAVITY_MASTER_PLAN_2026-10-10.md).
- [x] Retire `PRIOR_ART_HISTORICAL_MASTER_HANDOFF_V2.md`, keeping current user/open decisions and source attribution in the decision register and research/license documents.
- [x] Retire `PRIOR_ART_HISTORICAL_ROADMAP_V2.md`, using [P00–P23](../IMPLEMENTATION_ROADMAP.md).
- [x] Retire `PRIOR_ART_HISTORICAL_STAGE0_V2.md`, using [current delivery gates](../INTEGRATED_PHASE_0_AND_DELIVERY_GATES.md).
- [x] Retire `PRIOR_ART_HISTORICAL_EXPERIMENTS_V2.md`, using [current experiments](../EXPERIMENT_PROGRAM.md).
- [x] Retire `PRIOR_ART_HISTORICAL_ACCEPTANCE_GATES_V2.md`, using [current release gates](../RELEASE_ACCEPTANCE_GATES.md).
- [x] Retire `PRIOR_ART_HISTORICAL_DISTRIBUTED_PARALLELISM_V4_1.md`, using the current placement/peer contracts; preserve one PublishRoot across participants.
- [x] Retire `COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md` after accounting for its useful user/open decisions against the separately retained v5/v5.1 contracts, acceptance plan and seven research dossiers. Do not reproduce embedded old status/version copies.
- [x] Migrate useful current build/release/Nix guidance from [the pre-product snapshot](../../research/README_PRE_PRODUCT_REWRITE_SNAPSHOT_2026-10-15.md), then replace its body with a useful pointer. Keep that exact path valid because the immutable root README links it. Separate GPU lock/probe instructions are obsolete.
- [x] Retire `SSD_STREAMING_FIRST_CLASS_ARCHITECTURE_AGENT_HANDOFF_2026-10-15.md` after moving live links to its retained residency contract, SSD acceptance and source atlas. Preserve queued-reader and mixed-codec quota regressions.
- [x] Retire `EXTREME_MODEL_SPECIALIZATION_DESIGN_AGENT_HANDOFF_2026-10-15.md` after moving live links to its retained integration/source atlas and EXL3/QTIP/hybrid/GFX contracts. Preserve SYK proposal/open status and negative findings.
- [x] Retire `MULTIMODAL_PARALLEL_RESEARCH_WORKSTREAMS_2026-10-14.md` after retaining their useful unanswered questions/negative fixtures in the active multimodal TODO and dossiers.
- [x] Retire `MULTIMODAL_V5_1_RESEARCH_WORKSTREAMS_AND_AGENT_BRIEFS_2026-10-15.md` after preserving any distinct modality acceptance gates.
- [x] Before archive removal, update active inbound links from public engineering/index documents, the master/critical review/composition/placement specs, phase/release gates and decisions. Leave the root README bytes unchanged.

Keep historical rank/state commit, four enduring IR and obsolete archive-generator
instructions superseded. Current API/auth/trust/SLO/plugin/quality questions stay
in their active plan TODOs.

#### Integrated continuation

- [x] Preserve rejected-design rationale and one-publisher/compiler/physical ownership in the critical review; replace stale implementation prefixes in the master with current plan authority.
- [x] Retain all SSD-01..32, QTIP/HYB/GFX and MM-01..24 plus 77 family fixture requirements in their current technical owners; carry distinct pending tests/policies into the owning plan TODOs. None is promoted by this document cleanup.
- [x] Cross-check the multimodal aggregate's 13 current source owners, three shared snapshots and 94 nested source bodies. All 25 retired/alias bodies are already covered by the accepted earlier migration or this reviewed retirement; no unique current modality source is missing.
- [x] Replace the pre-product body with current navigation; move verified build/release/Nix interfaces into docs/DEVELOPMENT.md. Root README bytes remain unchanged.
- [x] Recover the initial gate failure caused by a copied CLI test embedding the retired checkout path. Refresh only test-source mtime and recompile; source bytes and the gate command are unchanged.
- [x] Reconcile independent final review, resolve leftover archive-preservation claims, and replace remaining active generated-handoff/SHA closeout duties with TODO-SPEC. Preserve historical user-decision rows, current source identifiers and numerical/physical tests.
- [x] Integrate completed run findings, retire their temporary files, commit this phase with signing and close the index line.

## Uncommitted

None. Continue [model acquisition A2](../model-acquisition/TODO.md).

## Tests and verification

- [x] Independently compare all nine prior-art snapshots against current contracts; retain distinct rationale, decisions and attribution.
- [x] Independently compare SSD/specialization/build wrappers against retained references and current source commands.
- [x] Account for all multimodal aggregate source sections and both briefs; preserve distinct acceptance gates and unanswered questions.
- [x] All current first-party Markdown local links resolve; no live reference claims a deleted archive is still preserved.
- [x] Root README bytes, runtime/source/dependencies, vendor/reference fixtures and recorded model-content identifiers are unchanged from 7e61cb6.
- [x] `CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh` passes without downloading models/SDKs or launching held GPUs.
- [x] Review explicit paths, create a signed commit, integrate finished agent work and retire its temporary run files.

````

## Source: .agents/plans/cpu-gpu-optimization/TODO.md

Original UTF-8 bytes: 5349; SHA256 `c49fa766463002fc5f0b13aee9ce841dcc8acd597fa5f1c5eec336dd351e23d0`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Cpu gpu optimization

Provenance commit: `6d74f76`.
Scope: P01/P06/P07/P20/P22; follow acquisition and measure before selecting complexity.
Contracts: [fleet optimization](../../research/INITIAL_HARDWARE_TARGETS_AND_OPTIMIZATION_2026-10-16.md),
[GPU text/control](../../research/GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md),
[FP32 FMA proposal](../../research/NATIVE_FP32_FMA_QUALIFICATION_PROPOSAL_2026-10-16.md),
[derived quant contract](../../docs/DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md).

## Uncommitted

- [ ] Bench and hyper-optimize in strict user order:
  Qwen3.5 2B, Qwen3.8 27B including SSD/coalesced expert streaming;
  Gemma4 E2B/26B/31B; Qwen3.8 Flash-Next Q2/~3.5-bpw/Q4
  and MTP/drafter/n-gram pipeline. Optimize complete
  request goodput with the same model semantics, source quant
  calibration and StateRoot precision. Do not change the
  Renoir/gfx90c launch hold or assume measured superiority.

- [ ] Keep the local Renoir/gfx90c hold: 2B initialization aborted and automatic resets wedged it; no further launch/reset/recovery or Drop-retirement proof is implied.
- [ ] Establish controlled cold/warm request baselines across all five GPU/three CPU optimization targets; accepted five backend profiles do not mean five cards or fleet-wide maximal capability use.
- [ ] Qualify capability-selected AVX2/AVX512/VNNI quantized kernels, cache/thread/topology policies and source-derived repacking while preserving reduction/rounding/zero/subnormal rules.
- [ ] Measure native/Vulkan tiling, subgroup/vector loads, packing, source-exact fusion/prepared dispatch; gate tensor/MMA/WMMA/integer-dot use by exact arithmetic, not advertised support.
- [ ] Evaluate explicit NativeF32 projection policy only after physical FMA edge tests, all-codec trained vectors and whole-model/cancellation/replay gates; WideF64 stays default meanwhile.
- [ ] Profile then qualify captured graphs/replay, bounded device loops/persistent scheduling, CPU/GPU overlap and exact optional GPU tokenization/detokenization; prepared plans alone are not graph capture.
- [ ] Keep separate LLVM code-generation paths and broader accelerator backends unqualified until independently built and physically tested; accepted native paths use CPP NVRTC/HIPRTC.
- [ ] Qualify optional EXL3/QTIP/trellis/codebook/low-bit derived packs and calibration identity/quality separately from original source semantics. No hidden precision substitution or default steering.
- [ ] Promote only repeated complete-request gains with cold/warm cost, tails, memory, power, source correctness, negative results and qualified capability/version guards.

- [ ] Gate future compiler/driver/libm and fast-arithmetic profiles with scalar fingerprints, physical edge/stress and trained full-logit/state tests. WideF64 is qualified on measured inputs, not a universal IEEE FMA emulation.
- [ ] Qualify real deadline wakeups, concurrent/multi-stream scheduling and actual driver failure/recovery before broader worker claims; queue deadlines currently need scheduling/poll opportunities.
- [ ] Keep the local gfx90c hold. If a healthy/recovered profile is separately authorized, localize retained allocation/H2D/final-sync initialization before changing kernels; natural SIGABRT establishes neither Rust Drop quarantine nor reclaimed storage.
- [ ] Consider read-mostly publication only after representative contention and ownership evidence. If left-right fits, test deterministic dual replay without IO/RNG/admission/state/device effects, one writer, persistent non-Sync readers, short guards without await/load/device work and explicit publish/drain/shutdown. Account for both indexes/logs while sharing one payload; do not reopen the rejected cold-catalog migration.

- [ ] Resolve applicable SYK-011..016 source/derived-quality/LLVM/SLO/pressure/steering choices from the [specialization reference](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md) before that slice; proposed SYK-002..010 are not frozen user approvals. Preserve negative candidates and the default disabled steering policy.

## Committed

### 6d74f76

- [x] Complete resident original-model pipeline and five bounded promoted profiles accepted; actual rollback/admission/cancellation/quarantine remain baseline requirements.

### Concurrency review — 1fca893

- [x] No demonstrated cold-catalog contention justified left-right. Its uncommitted candidate was restored and no speedup is accepted. Immutable Arc/borrows stay direct; a delayed status view cannot authorize admission, launch, release, StateRoot publication or fence clearing.

## Tests and verification

- [ ] QTIP-01..14: derived source codec/pack identity, transforms, held-out quality and physical kernel correctness pass independently of original model semantics.
- [ ] HYB-01..18: exact source routing/draft/tap/mask/reduction and target-authoritative verification pass, including cases where a wrong draft is rejected or deeper drafting slows work.
- [ ] GFX-01..18: WMMA/LDS caller/shape coverage, driver/compiler numeric gates and coherent-memory physical headroom pass; require matched complete-request gains. See the [50-test register](../../research/EXTREME_MODEL_SPECIALIZATION_INTEGRATION_AND_ACCEPTANCE_2026-10-15.md#4-acceptance-register-50-planned-tests-not-implemented).

````

## Source: .agents/plans/distributed-execution/TODO.md

Original UTF-8 bytes: 1782; SHA256 `21b25cbb3b8dc97a88e83fc7cc930da2a35b50983c05a815819a0941d0f038da`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Distributed execution

Provenance commit: `6d74f76`.
Scope: P19 plus P04/P05/P09/P10/P16/P23 extensions.
Contracts: [placement](../../docs/DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md),
[peer/state protocol](../../docs/REMOTE_PEER_TRANSPORT_AND_STATE_HANDOFF_PROTOCOL.md),
[accelerator families](../../docs/ACCELERATOR_EXECUTION_FAMILIES_AND_BACKEND_CONTRACT.md).

## Uncommitted

- [ ] Preserve zero-network standalone execution; add capability/transport/placement seams through the same model/task/coordinator/resource authorities.
- [ ] Qualify optional local CPU/GPU/NPU split projection, expert ownership and prefill/decode placement with actual event/memory/power-fabric safety; Stage N is independent of CPU/GPU release.
- [ ] Begin authorized two-node layer-group plus local-state execution and immutable full-state hydration; then evaluate P/D and attention+state versus FFN/expert disaggregation.
- [ ] Extend tensor/head/pipeline/data/context/expert/sequence parallelism and collectives only where exact state partition algebra, source numerics and transport cost support them.
- [ ] Authenticate/version source/state/tenant/session epochs and complete transfers; loss/timeout/stale peers cannot publish partial roots or release unknown in-flight storage.
- [ ] Qualify topology changes only at committed frontiers and compare collocation against fast/slow/tunneled links, including real transfer/copy/crypto cost and fairness.
- [ ] Add a peer/transport/placement without rewriting model, task, coordinator or API; fleet SSH qualification alone is not distributed inference.

## Committed

### 6d74f76

- [x] Hardware fleet discovery and independent per-host profiles accepted; distributed/heterogeneous model execution is not established by those results.

````

## Source: .agents/plans/memory-and-streaming/README.md

Original UTF-8 bytes: 3106; SHA256 `1f67e268e85c23fad327f0be354a4b44d280bcab9513299f14ec893d3083f19d`.
Resolve embedded relative paths against the original path, not this volume.

````text
Author: Project owner
Co-authors: GPT-6
Owner: @pvl

# Source-preserving SSD residency

## Implemented: immutable original SSD source-extent IO

`zerogravity-artifacts::ssd_extent` under original `cache` feature.

- Already-opened file descriptor + independently pinned SHA256 of strict `zerogravity.extents.v1` JSON with complete contiguous chunk offsets, bounded bytes and per-chunk hashes; manifest also *claims* whole-file SHA256 but never pretends it is verified merely by opening.
- Sorted unique chunk or bounded arbitrary tensor-window requests. Coalesced exact positional IO, staging charged *before* allocation/read to existing caller-provided `ResourceArbiter`, hash every chunk of owned bytes before exposing any slice.
- Max 4 MiB/chunk, 32 MiB/read, 32 chunks/read, 16 MiB index, 200k chunks, 1 TiB file metadata. Returned `StagedExtents`/`StagedSlice` retains both immutable bytes and physical transfer-staging lease through final drop.
- Tests: exact first/last/window reads, damaged manifest, incomplete/overlapping geometry, original source inode mutation, source file rename, truncated reads and physical budget refusal; sparse >5 GiB metadata without materializing giant bytes.
- Resident model hot paths remain unchanged: there is no mandatory SSD read per token.

## Still pending

- Model execution through accepted original-index tensor windows, asynchronous prefetch and source-verified copy-to-GPU completion with stable lease ownership. Gemma source/runtime qualification remains separate.
- Eviction/ABA/cancellation/backpressure, per-expert codec placement, fair IO and physical GPU/RAM/SSD budget integration; never return a borrowed TensorSource slice from an evictable cache.
- Authorized original trained resident-versus-SSD complete logits/StateRoot correctness and cold/warm TTFT/ITL, IO amplification and memory-pressure benchmarks.

**This is source-byte IO, not yet working SSD-streamed inference.**


## Original-index tensor windows

`SourceWeights::prepare_verified_windows` authenticates the complete pinned
original index, whole shard hashes and actual headers using bounded scans.
Its sealed owner exposes original matrix descriptors and immutable owned
full-width row windows, each retaining source identity, actual staging and
metadata leases. It supplies no whole borrowed TensorSource or model permit.
Existing resident-source APIs retain their4GiB policy.

Actual original Qwen3.8-27B qualification scanned all18 weight files
(55,563,006,776B), matched12 raw BF16 row samples including4 extent splits
and released every reservation after the final window dropped. Process
peakRSS was17,328KiB under a512MiB address cap. The configured128MiB host
ledger is not independently attested physical RAM admission; kernel page
cache and generic allocator overhead are separate. Nine window and seven
extent fixtures, full isolated check and approved original release parity
routes pass. See [TODO.md](TODO.md) for exact scope and remaining work.

**Still source-byte IO: no streamed model, GPU upload/Ready or StateRoot
publication is qualified by these results.**

````

## Source: .agents/plans/memory-and-streaming/TODO.md

Original UTF-8 bytes: 8082; SHA256 `0d0990395ef44a98e3b643e5d6612e8849164cd7a69aee18e1f987155c5d2c42`.
Resolve embedded relative paths against the original path, not this volume.

````text
# Memory and streaming

Provenance commit: `6d74f76`.
Scope: P03/P05/P07/P10/P12/P16.
Contracts: [unified residency/SSD](../../docs/UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md),
[SSD gates](../../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md),
[large-source admission](../IMMUTABLE_SHARDED_SOURCE_IMPLEMENTATION_SPEC.md).

## Uncommitted

- [x] **Original-index owned tensor windows:** existing SourceWeights
  factory prepares a sealed VerifiedTensorSource with one original source
  identity, authenticated complete index/shard/header/bundle closure and
  retained opened descriptors. Whole SHA256 scanning is bounded4MiB;
  only full-width original OutputInput matrix rows can materialize.
  TensorWindow owns immutable staged bytes, exact original kind/shape/
  offset/row geometry and source owner through final drop. No borrowed
  whole TensorSource implementation or55GB resident allocation. Existing
  resident4GiB path remains unchanged. Original host ResourceArbiter
  admission precedes scan/window allocation; charge includes retained
  extent-offset metadata and carried descriptor vector capacity. Metadata
  is capped128MiB; labels/geometry/read counts are bounded. Cancellation
  checks retain synchronous IO owners until completed; pre-cancel/drop
  coverage does not prove asynchronous mid-read cancellation.
- [x] **Actual original Qwen3.8-27B source-window qualification:** exact
  revision1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, acquired snapshot
  98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e.
  One actual55,563,006,776-byte scan verifies all18 weights shards,
  original1199-tensor index/header closure and unchanged bundle digest.
  Twelve actual BF16 full-width row samples across QKV/embedding/last
  down-projection/head include four authenticated extent crossings;
  same-open-inode independent raw reads match every byte. Peak selected
  staging8,388,656B includes retained offsets; final reservation0 after
  the last window drops its source/metadata/FD ownership. Header/source/
  occupied-ledger refusals, pre-cancel, wrong axes/rank/name/ranges and
  attempted whole embedding materialization refuse without leaked charge.
  Root scan completed82.847s under512MiB address cap; measured process
  peakRSS17,328KiB. Configured metadata64MiB/host pool128MiB exercises
  the real shared ledger; it does NOT independently attest free RAM,
  account kernel page cache or qualify physical host admission/inference.
  Initial direct-rustc LTO link failed on dependency bitcode before payload
  execution; locked/offline isolated harness build and actual run passed.
- [x] **Window verification:** nine new synthetic ownership/source tests
  and seven SSD extent tests pass; exact mutation/short-read/rename,
  staging/metadata refusal and retained lifetime cases. Complete isolated
  check1202 Rust/0failed/69ignored/248 suites plus18Python/release/installer/
  Nix; strict CUDA/Vulkan/qualification Clippy and both approved original
  cache-direct/HF-import release routes pass full248320/all42. No GPU
  upload/Ready or model-root publication is performed by this source API.

- [ ] **Next SSD source execution:** integrate accepted original-index
  owned windows and an async I/O/readiness model with
  original CPU/GPU TensorSource and BufferLease, cancellation,
  cache/eviction/ABA and independently trained full-logit/StateRoot
  parity on real large weights.

- [ ] Shared one-process pipeline residency: source-exact immutable
  model/component buffers, per-session private StateRoots, one
  ResourceArbiter (UMA/GPU/RAM/SSD and device queue), cross-model
  lease/zero-copy only when physical placement, codec and lifetime
  permit it. Do not replicate weights per route or release a
  GPU buffer before actual completion/quarantine retirement.
- [ ] Prioritize Qwen3.5 2B resident then Qwen3.8 27B SSD
  streaming and Gemma4 E2B/26B/31B, then sparse Qwen Flash-Next
  learned 51B n-gram bank and 4B MTP head placements. Compare
  source-real logits, resident/streamed StateRoot, cold/warm
  TTFT/ITL, I/O/read amplification and throughput with full
  physical/cancel/error gates; live audio's soft deadline may
  demand resident codecs and refuse cold SSD layouts.

- [ ] Route legacy/new model allocations, CPU state/scratch/staging and APU/GPU shared backing into the same configured physical pools with OS/cgroup headroom; never sum overlapping UMA/VRAM/GTT counters.
- [ ] Implement paged/contiguous state policy and full-domain prefix identity/intersection/COW across KV/SWA/recurrent/indexer/draft/grammar/media state, source/template/adapter and tenant boundaries.
- [ ] Qualify committed snapshot persistence/restore, corruption/security/compatibility UX and restore-versus-recompute; diagnostic portable checkpoints are not product persistence.
- [ ] Add immutable source extents and bounded coalesced IO/materialization with stable logical tensor identity, genuine copy readiness and original IO/device reader leases.
- [ ] Protect encoded-but-unsubmitted/captured expert reads and generation ABA; support mixed-codec expert size classes under one byte cap, source-exact routing and hot-cache preservation.
- [ ] Add bounded PLE/Engram lookup rows and separately governed disk state/media tiers; fully resident execution must incur no mandatory IO or host cache dispatch.
- [ ] Test source-exact resident/streamed parity, short/corrupt IO, cancellation, safe victim refusal, SSD QoS and actual physical lifetime before eviction/free-space claims.
- [ ] Optional direct/async/platform IO needs equivalent semantics and explicit support; large memory-pressure/SSD experiments require approved hardware/source scope.

- [ ] Distinguish logical envelope retirement from driver allocator pages/pools, fragmentation, cross-process use and actual OS release. Driver VRAM/GTT usage/max counters are not reservations or immediately free bytes.
- [ ] Prove trusted teardown/recovery before reclaiming unknown-work handles or clearing a fence. Diagnostic service sync/readback permits inspection/exit only and cannot synthesize Ready, retire unknown events or authorize reuse.

## Committed

### 35b9093 — source-only verified SSD extents

- [x] **Verified source-only SSD extent foundation:** original opened
  descriptor, SHA256-pinned complete chunk manifest, checked source
  offsets and per-range hashes, 4MiB chunk/32MiB read bounds,
  coalesced positional file IO and original ResourceArbiter
  transfer-staging admission before allocating/reading. Exact
  byte-window materialization returns immutable leased slices, never
  unsafe borrowed mutable-file views. Tests cover exact reads,
  truncation/hash mutation, path rename, rollback, insufficient
  host memory and sparse >5GiB metadata-only source.
  Six SSD-focused tests pass, and the entire isolated workspace
  gate passes 1,147/0 across 241 Rust suites. The Nix filtered
  source-integrity derivation cannot run offline because the
  pinned nixpkgs flake input is missing (not a source test failure).
  No trained resident-versus-SSD model, device copy or eviction yet.

### 6d74f76

- [x] Physical arbiter, original GPU leases/quarantine and committed session/checkpoint foundations accepted; global host/UMA, full-state prefix manager and SSD model streaming remain open.

## Tests and verification

- [ ] SSD-01..32: run the [full source/residency suite](../../research/SSD_STREAMING_IMPLEMENTATION_ROADMAP_AND_ACCEPTANCE_2026-10-15.md) for each claimed backing/backend; mock tiers cannot qualify physical IO or trained streaming.
- [ ] SSD-14/15: encoded-but-unsubmitted, queued and captured readers prevent generation-ABA slot reuse through actual GPU completion/cancellation.
- [ ] SSD-21/22/24: mixed Q2/Q4 expert sizes share one byte cap with safe victims and fair progress; hot-cache retention cannot bypass admission.
- [ ] SSD-32: an authorized original trained streamed model matches complete resident logits/StateRoot; measure cold/warm IO, staging, OS headroom and genuine resource lifetime.

````


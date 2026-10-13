# zerogravity — Product, model loading and extension contract across the full pipeline

> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


**Status:** consolidated working design. Implements lessons from prior format-architecture and cross-cutting multimodal research and the project SIR/state/serving docs. Defines what an agent adding a new format, model, modality, endpoint or quantization must change—and what must remain untouched.

## 1. Source/format are not model mathematics

The model-loading contract should be:
```text
Resolver{path | verified HF revision | ModelStore tag}
  -> immutable ModelArtifact + complete SHA/manifest
  -> bounded GGUF/safetensors parser and tokenizer/rendering source
  -> ArchDef(ModelRecipe) maps declared tensors to verified semantic ops
  -> typed SemanticModelIR + StateContracts
  -> Compiler/ResourcePolicy/Kernels
  -> independent EngineCore/Frontend
```

No loader should directly choose `gfx1151` fused kernels. Model adapter declares exact trained formulas, all tensors and layout constraints (could be declarative for repetitive transformer families) while backends know only semantic ops/shapes/quant. Unknown model architecture, quant, tokenizer pre-type or new positional math gets a diagnostic refusal, never reinterpretation.

## 2. ModelStore and manifest versions

Borrow Ollama-like content-addressed blobs and manifest lists:
- immutable binary layers addressed by SHA-256;
- named model tag maps to manifest specifying base weights, shards, tokenizer/config/template, optional vision projector, LoRA/adapters, MTP sidecar, calibration/imatrix, license and **derived acceleration pack**;
- atomic resumable download with bounded size/cryptographic hash; explicit offline mode, no arbitrary code execution;
- store's garbage collector computes reachable blobs from referenced manifests, pins and active sessions—not file mtime alone;
- model switching/unload waits until GPU graph and buffers release their leases; model identity tied to actual bytes and recipe schema;
- user-effective file/cache locations XDG-aware with defaults subject to UX decision, private prefix snapshot cache separated from public model weights.

**Optional accelerated pack:** contains losslessly packed device tensors, per-expert one-blob index, layout descriptors, kernel plan provenance, and possibly explicit quality overlays which are *separate derived model artifacts*. Original GGUF/safetensors remains loadable without pack; pack invalidates if source checkpoint/compiler/layout revision changes. The research called it "APK"; retain only as historical label to avoid conflating with Android installation packages. Define a stable public extension after quality/format tests, not now.

## 3. GGUF and quantization schema

GGUF parser contract: v2/v3 header, LE integers, tensor metadata, name uniqueness, alignment power-of-two, checked 64-bit dimension product/offset+length, per-file dialect and tensor type registry, multi-shard completeness and tokenizer metadata. `ne[0]` is stored contiguous in GGUF, not necessarily a row-major host matrix. Must not treat GGUF type id from ik dialect as mainline without dialect detection. Unknown types refused with type id and precise fix.

Quant sources: native F32/F16/BF16/FP8/MXFP4, GGUF Q8_0/Q4_0/Q4_K/Q5_K/Q6_K and IQ2-family as needed; supported types are an evidence-based matrix, not every known type promised at v1. Store `logical_dims`, `stored_block_shape`, `scale/zero metadata`, `packed_layout_id` and `calibration_id` distinctly.

Internal packing:
- lossless reorder/transpose may be automatically selected when validation proves dequant/oracle parity;
- deliberate quantization changes produce a new source/derived checkpoint fingerprint and quality bands, never silently chosen to fit memory;
- model-sensitive precision allocations: always-on norm/router/mHC/indexer/latent state may require more precision than selected routed experts; not a one-bpw-for-all policy;
- imatrix calibration collects per-channel activations/routed expert histograms from an explicit offline `Tune` graph mode, not a surprise production telemetry path; independent calibration dataset license and reference quality measured;
- overlay quality/speed patch keyed to base artifact and declared sensitivity tradeoff, not silently applied.

## 4. Tokenizer, chat templates and constrained outputs

Product use depends on token fidelity as much as GEMM speed.

`Tokenizer`: `encode` with special policy, `decode` and byte-exact `token_to_piece`, vocabulary size, EOG and GGUF pre-tokenization type. Golden fixtures against official source and pinned llama.cpp; Unicode/invalid UTF-8/special-token injection/role boundaries.

`PromptRenderer`: model-native Jinja-compatible templates (`minijinja` possible), strict undefined behavior, control-token-safe input marking, tool and reasoning headers; returns structured rendered token IDs plus canonical hash. No arbitrary template code with host file/network access.

`ToolParser`: incremental model-specific Hermes/Qwen, Llama, Mistral, Harmony, DeepSeek/GLM, generic JSON/tool fenced formats as supported; separate hidden reasoning versus visible assistant response; stops across token/UTF-8 boundaries; versioned capability matrix.

`GrammarFSM`: constrained decoding from JSON schema/grammar (a grammar library via Rust binding is candidate), token trie/pretype matching, compiled asynchronous with bounded CPU time/memory; accept/mask/rollback state is part of Session transaction; masking order follows exact selected model sampler distribution. Unsupported arbitrary GBNF should refuse explicitly rather than silently accept incomplete grammar.

Strong user-facing requirement: `/props` or `inspect --explain-plan` returns template/tool/grammar/vision/embedding/quant/backend capability and important typed unsupported reasons. No fake feature claims.

## 5. Text generator and encoder/pooler are different runner modes

**Decoder-only generation:** KV/recurrent state, token-by-token sampling and speculative verification, token-budget batching and stream. **Embeddings/encoder-only models:** no autoregressive decoder/sampler/KV in ordinary pooling path; batch by length/encoder budget, hidden-state pooling, normalization/dimension truncation and precision reference. **Cross-encoder rerank:** model of query/document pair, score head, document batching, template length/truncation and calibration. These are semantic graph recipes that share kernels, devices, formats and resource arbiter, but not a fake common "generation loop".

`PoolingSpec` should encode `Cls | Last | Mean | Token | Multi`, last-token side, model-specific L2 normalize and Matryoshka dimension ordering, padding/masks and output dtype. API `/v1/embeddings` and `/v1/score`/`/rerank` only when exact model/response schemas tested. Streaming input semantics for video/audio encoders can reuse modality graph but is not guaranteed by a text executor.

## 6. Multimodal and modalities as typed state

For each input `MmItem`: content fingerprint, raw/preprocess URI or validated blob, modality type, pixel/sample limits, resize/patch/encode geometry; `MmSpan`: token positions, embeddings, M-RoPE/axes, causal/bidirectional window. Encoders/projectors are model-like graphs with their own `ModelRecipe`, `DevicePlan`, computation/memory budgets and exact state/cache fingerprints.

**Critical:** a noncausal/bidirectional image span cannot be split arbitrarily into prefill chunks as if it were ordinary decoder tokens. Scheduler declares `supports_partial` and span chunk boundaries; refer to the cross-cutting multimodal-embedding research. Vision embedding cache and KV/recurrent prefix cache have related but distinct identities; both must be available to claim full prefix reuse. Image edit invalidates affected snapshot, even when visible text prefix unchanged.

Speculation with image/video histories must be separately verified; can be disabled by capability until state restoration is correct. Do not promise MTP support merely because text-only model uses it.

**Future modalities:** audio input and live streaming, video temporal positions, new vision patch schemes, action/robotics/embeddings, potential diffusion/flow engines; integrate new semantic operators and loop/sampler policy without forcing old decoder-only code to become a monolithic switch statement. The product does not promise text-to-image/diffusion at initial release.

## 7. LoRA/adapter model (previous project gap)

LoRA must not be a simple mutable `adapter_name` global on a loaded model. Required contract:
- adapter artifact references exact compatible base model tensor layout, rank, scaling and affected projections;
- per-request/session `AdapterSet{generation, weights, weight_multiplier, compatibility_digest}`, plus safe LoRA kernel candidate or portable reference implementation;
- prefix cache key includes adapter identity/generation and any request-time blending; cannot share cached state across incompatible adapters;
- memory manager budgets adapters separately from base immutable weights and in-flight requests; hot swap preserves old sessions until references retire;
- batching may group compatible adapter rows for fused kernels; opposite adapter combinations cannot accidentally share projections;
- dual quant/LoRA transformations must be independently quality-tested with official known vectors;
- trained MTP/drafters may be incompatible with adapted target distributions, so speculative drafter needs explicit compatibility or falls back.

Defer complex multi-adapter batched fusion until base generation and state contracts validated; CPU reference for adapters comes first.

## 8. Platform and hardware extension contract

New AMD/NVIDIA/Apple/Intel/ARM/NPU/remote accelerator path adds `DeviceCapabilities` probe and driver+stream+event/memory tier backend. It can select generic kernel provider where supported or implement target-specific Rust-authored kernels. It never changes `ArchDef` or user prompt parsing. NPU sidecars for encode/prefill are only feasible if state movement, quant support, driver coexistence and physical bandwidth/latency are proven; do not disable IOMMU or security system for speculative speed gains automatically.

Hardware detection uses actual architecture/capabilities and OS/driver status, not marketing product names. Quant and device ISA matters at least as much as numerical model family.

## 9. Serving facade

One canonical `InferRequest` with optional `TextGeneration`, `Embedding`, `Rerank`, `Multimodal` task kind; frontends for OpenAI/Anthropic formats where conformance passes. Edge performs validation, deterministic rendering/tokenization, auth and quotas, then sends request to EngineCore. Engine core performs scheduling/commit; worker model execution; frontend streams only verified output. A tool executor/MCP app is an *independent high-risk permissioned subsystem*, not implied by text-generation support. Avoid web fetch/file access from a model input by default.

CLI has simple `run/serve/models/inspect/doctor/bench/convert/quantize/pack` candidates, default simple local use, offline model trust controls and optional XDG model store. Use one structured error/refusal taxonomy with fixes, no arbitrary low-level flag burden.

## 10. Conformance matrix for every extension

For a **new model**: checkpoint and tokenizer/template hashes, SIR operator registry and state effects, CPU oracle logits, model task quality, generic backend, optimized candidate, context/quant/multimodal limit and fallback.

For a **new format**: negative/malformed vectors, odd alignment/multishard/mapping, model conversion roundtrip exactness or labeled quality if quantized, data license.

For **new GPU**: actual physical driver setup, dtype/ISA capability, buffer alloc+copy/event/capture, generic kernel smoke, upstream comparable benchmark, explicit unsupported ops and memory cost.

For **new endpoint/grammar**: official client schema, data/model state and error tests, streamed output/stop/timeout, privacy, auth and capacity.

For **new adapter/modality**: state cache identity, downstream sampling and position logic, memory reservation, snapshots, exact parity and cancellation.

## 11. Explicit non-goals and open points

No direct execution of unknown HF Python architectures; no automatic quant quality downgrade; no untrusted executable model plugins; no unconditional claim that 156+ llama.cpp families work; no mixing session/adapter cache across tenants; no public server without auth by default; no promise of universal MTP/LoRA multimodal support. First model families and extensions should be prioritized by real demand and feasible validation—user explicitly selected modern families but not v1 support matrix.

**Strong recommendation:** preserve this as the single product extension map; use existing `MODEL_SEMANTICS_AND_COMPATIBILITY.md`, `FORMATS_QUANTIZATION_AND_LOADING.md`, `RUNTIME_SERVING_AND_UX.md`, `RUST_WORKSPACE_AND_ENGINEERING.md` for detailed subtype contracts, not duplicate entire parser/runtime implementations.

## 2026-10-12 integration — upstream Gufo task pipelines and reusable TaskProgram

The latest upstream Gufo model suite is broader than decoder-only text: **Qwen3.8-27B/DFlash2 vision, Flash-Next QSA/GDN/MTP, DeepSeek V4/DSpark, Qwen3-ASR, Qwen3-TTS (speech synthesis/voice cloning), Qwen-Image-2.1 generation/editing and MiniMax H3 text-to-video/audio in progress**. Pin each upstream model's version and quality/benchmark independently. A unified zerogravity `TaskProgram` adds explicit `Preprocess/Encoder/TokenDecode/VerifyDraft/Pooling/Vocoder/DenoiseStep/DecodeMedia/OutputMux` stage semantics and task-specific iteration/output frontiers, while reusing the **same SIR/RIR/KIR/CIR, KernelPlan, device registry, memory arbiter and artifact store**. Do not force all audio/image/video into `EngineCore::next_token` or declare work-in-progress upstream models supported in zerogravity.

**XDNA2 NPU as generic stage/slice owner:** no new model-specific NPU server. For a qualifying image/text/audio encoder, RIR may split independent projection output rows or whole eligible GEMM stages onto NPU while GPU performs unrelated computation, through typed task-stage dependencies and memory/fabric qualification. The currently source-demonstrated Gufo fork only accelerates specific **Qwen3.8 Flash-Next prefill GEMM slices**, so all other NPU uses are unverified hypotheses. Each stage must have its own valid numeric/model quality and model artifact fingerprint, cost/occupancy and checkpoint/cancellation policy.

Audio/image/video stage loops differ: streaming ASR with encoder/CTC or seq2seq as model-defined, TTS acoustic autoregression plus vocoder and PCM stream, image generation with iterative flow/diffusion/conditioning and denoise timestep state, video with frame/temporal conditioning and A/V sync. Some models have no KV cache, some have noncausal spans or long-lived latent state; `TaskProgram` models this explicitly and `CIR` schedules GPU/NPU/CPU resources. ModelStore supports encoder/projector/vocoder/decoder sidecars and individual licenses. All unsupported task paths refuse clearly while text-only remains usable.

See [NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md](NPU_GPU_COOPERATIVE_EXECUTION_AND_TASK_PIPELINES.md) for the full stage graph and [GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md](../prior-art/GUFO_FORK_AND_XDNA2_PREFILL_SOURCE_RESEARCH_2026-10-12.md) for source citations.

## v4 refinement — TaskDriver, not universal workflow DSL

The shared product execution library exposes trusted built-in **TaskDrivers** for source-qualified text, encoder/embedding/rerank, ASR, TTS/vocoder, image denoise/edit and video temporal processing. Each has its own typed stage frontier and output stream; device step repetition can be compiled into a StepProgram instead of polling a generic workflow graph for every token/frame. `ExternalAction` from older TaskNode sketches is **removed from the inference core**: retrieval, tool execution, agent workflows and file/network actions are separate permissioned *application adapters* above TaskRequest.

Task requirements are model-independent but cannot imply a task works without source adapter, codecs, semantic operators and task quality tests. Optional future `TrainStep`/gradient update can operate on a new ParameterVersion and optimizer state, while ordinary inference keeps immutable model views and no mandatory autograd/training runtime. ModelStore remains a simple immutable manifest+blob source, not a workflow scheduler.

## v4.2 — TaskDriver distribution without forcing token-only topology

The same TaskDriver can run its stages locally or on authorized remote RegionExecutionProviders; the task's **committed frontier and output type remain unchanged**. Image/video/ASR/TTS may place encoder, transformer/denoiser and decoder/vocoder on different machines when stage output transfer is cheaper than weight loading or resource contention; no forced causal token/KV semantics for media. Distinct parameter/model sidecars and source rights matter when shipping staged model weights or private audio/images across peers.

For language tasks, prefill/decode disaggregation transfers **all required hybrid session state**, attention/FFN disaggregation transmits per-layer activations, remote prefix cache validates StateCompatKey, and token-range KV can be partitioned only for mathematical attention schema with a valid summary reduction. TaskRequest/application agent tool authorization does not expand because a tunnel peer exists; network execution remains an explicitly permitted compute action, no arbitrary remote process permissions.

Program and cache identity includes source checkpoint, adapter, quant, position, media fingerprint and state schema, not only model name. A device lacking remote state snapshot can execute a **stateless encoder** but cannot pretend full interactive decoding support. [Placement](DISTRIBUTED_PLACEMENT_AND_NETWORK_EXECUTION_CONTRACT.md).

## v4.3 — registry model profile vs immutable source and execution residency

The ModelStore's immutable content-addressed artifacts/tensor/quant/tokenizer/sidecars are **RegisteredModels**, while a human-friendly `ModelAlias/ModelProfile` is a versioned typed overlay and a warmed `ResidentModel/Replica` is a physical model deployment with executable bundle/resource leases. Model loader never owns HTTP routing, and a changed `alias` revision cannot change a live TaskSession's `ArtifactKey/StateCompatKey` source/adapter/tokenizer. Two profiles of the **same source** may differ in sampling/context or load placement but still obey exact model identity and compatible compiled/state/quant policy; a different quant is a different verified immutable artifact, not a runtime sampler option.

Native catalog operations proposed `import/pull/list/show/alias/profile/rm`, atomic manifest and digest, license/size/source policy and GC preserving referenced blobs; no implicit remote download of unknown model names or arbitrary model-supplied code. Model autoload is *only* from authorized locally available sources unless user explicitly initiates download. A `ModelProfile` may name a trained draft, LoRA adapter, encoder, reranker or vocoder as **separate logical model roles** under a TaskProgram admission; they are not replicas/automatic target-model switches.

Provider Device/Task capability matters: unsupported architecture/quant/TaskKind shouldn't appear READY. Registry can expose model source metadata without allocating weights, while ModelResidencyManager performs safe physical reserve/warm/unload via existing global arbiter. [Model router](MODEL_ROUTING_RESIDENCY_AND_MULTI_MODEL_SERVING_CONTRACT.md), [zero-config UX/API](ZERO_CONFIG_CLI_CONFIG_AND_OPENAI_API_CONTRACT.md).

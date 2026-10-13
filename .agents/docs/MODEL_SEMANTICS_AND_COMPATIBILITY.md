# Model semantics, model adapters and compatibility qualification

**15 Oct model-family precision addendum:** [Qwen3.8/GLM5.3/MiMo2.6 recipes and drafting](EXTREME_HYBRID_MODELS_AND_SPECULATIVE_DRAFTER_SEMANTICS_2026-10-15.md) makes **MiMo-V2.6-Flash a separate registered architecture to implement**, not another Qwen model alias. MiMo must identify source E8M0 scales, source-specific SWA/global layers and asymmetric value-head width, its independent DFlash partial RoPE/tap/mask/attention, and exact derived-quant paired layer overflow correction. Qwen's GDN/QSA/PLE/MTP and GLM's KDA/MLA/DSA/mHC/MTP remain distinct. Model recognition, host parser support and a single GPU matvec never imply runnable whole model. HYB-01..18 are required source-real acceptance fixtures before any capability promotion.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


## 1. Model support contract

A model descriptor is validated source metadata **plus** explicit semantic graph recipe and tokenizer/rendering contract. Config parsing alone is insufficient. Every new architecture must state:
- model/checkpoint revision, official config and tensor manifest;
- token ID/BOS/EOS/tool/reasoning special token representation, text and multimodal template;
- layer topology, stateful vs pure ops, special attention and MoE formulas;
- quantized or native precision per tensor family, numerical accumulation and storage;
- supported context ranges, positional schemes, masking and cache layouts;
- trained draft/MTP compatibility; snapshot/restore semantics;
- supported input modalities, encoder projection and image/audio position calculations;
- output head/logit and sampler numerical contract;
- CPU reference, backend capability requirements and quality vectors.

A loader must distinguish `UnknownModel`, `KnownModelUnsupportedVariant`, `UnsupportedOp`, `UnsupportedQuant`, `InsufficientResources` and `MalformedArtifact`. No silent generic transformer guess for a novel model.

## 2. Canonical operator registry

**Foundational:** embedding, norms (RMS/Layer/RMS variants), RoPE/GQA/MQA/local position, projections, gated activations, FFN, matmul, quantize/dequantize, softmax, route/topk/scatter, logits/sampling.

**Full attention:** causal MHA/GQA/MQA with paged KV, sliding/local and mixed global, model-specific RoPE scale and sections, quant KV, FlashAttention, grouped multi-query.

**Hybrid recurrence:** Gated DeltaNet with learned gates, state matrices, short convolution, chunk prefill/step decode; Kimi Delta Attention with fine-grained gates/decay and independent recurrence; keep gated recurrence variants separately versioned (not a generic SSM approximation). Both require exact state snapshot + rollback.

**Sparse compressed attention:** Qwen QSA with indexer-selected blocks; GLM DeepSeek Sparse Attention over MLA latent KV (NoPE where defined); DeepSeek V4 CSA/HCA time-axis compressed keys and special indexers; independent positional/causal, softmax and quantized attention semantics. Sparse indexer top-k changes model computation and must not be hidden as a performance optimization.

**MoE/residual:** explicit router logits/scoring (`softmax`/`sigmoid`, bias, scaling, noaux etc.), top-k or grouped selection, shared expert, gate/up/down weights and sum ordering; simple gated residual vs manifold-constrained hyperconnections (mHC) not interchangeable. Effect order must preserve causal layer frontier.

**Lookup/draft:** token n-gram/Engram lookup tables with historical token state and hashed-row read; separate trained MTP/DSpark/EAGLE models; target verifier maps speculative draft to authoritative state.

**Multimodal:** vision/video encoder, image patch/resize/merge, projector, modality token spans, 3D/MRoPE positions, audio encoder and special prompt template. No model is considered multimodal functional without full encoder/position/continuation tests.

## 3. Candidate first model families

### Llama and dense baseline
Common decoder-only dense causal/GQA, RoPE, RMSNorm, SwiGLU variants; baseline for portable graph, KV, tokenization and GPU GEMM performance. Specific new Llama variants need exact config; not all use identical RoPE/attention heads. Build complete CPU & GPU before deploying advanced IR.

### Gemma 4
Includes sliding-window vs global attention, proportional/other model-specific position rules, output transformations and, for 26B-A4B, routed MoE with 128 experts and top-8 + shared expert according to inspected model docs. Correct Gemma tokenizer/multimodal pathways as applicable. Read released config and official implementation for every exact variant (E2B/E4B/26B-A4B etc.), do not infer identical semantics from name.

### Qwen3.5+
A broad hybrid family with GDN and periodic full attention; both dense and MoE/vision variants exist. Treat Qwen3.5 and Qwen3.8-Flash-Next as different architecture IDs, even when sharing GDN primitives. Native model config must determine gate/state dimensions, number of attention layers, residual and positional scheme.

### Qwen3.8-Flash-Next
Research snapshot: language backbone ~125B, ~6B active/token (count excludes large auxiliary 51B n-gram parameter table + ~4B MTP); 48 layers, approx 36 Gated DeltaNet and 12 QSA, 512 routed experts with top 10, gated residual/hyperconnections, n-gram lookups and one MTP predictor. Native context ~262144 tokens in DwarfStar's documented graph; extensions such as YaRN are policy with possible quality changes. Different quant artifacts: DwarfStar Q2 main/MTP ~41.73 GiB resident and ~95.37 GiB BF16 n-gram disk-only; a closed reference native checkpoint ~62.1 GiB plus ~47.7 GiB paged n-gram; Gufo bench UD-Q4_K_XL target and Q8 draft sidecar. **Not equal precision or a controlled speed comparison**. DwarfStar Qwen supports Metal/CUDA, not ROCm at inspected snapshot; the closed reference and Gufo are Strix `gfx1151` references.

### GLM-5.3-Flash
From Maya's config study: ~321,323,031,390 total, 18B activated/token; `glm5-next`; 45 layers, hidden 4096; layers 0–2 dense, layers 3–44 routed MoE => 42 x 288 = 12,096 expert blobs; top 8, shared expert; 34 KDA and 11 sparse MLA+DSA (layers 3,7,...,43); NoPE latent MLA with 512-wide KV latent, DSA top-k 2048, 4-way mHC/Sinkhorn, 1 trained MTP. Quant Q2 Maya-S ~96.5 GiB and related options, DwarfStar GLM Q2 ~90 GiB, different quant recipes/quality. Attention and recurrent state cannot be lowered as ordinary GQA. Context up to 1M advertised does not imply cache or accuracy parity at 1M.

### DeepSeek V4 Flash / V4.1 Flash
DwarfStar's model research: V4 Flash ~284B total/~13B active, ~43 layers, 128-token raw sliding window and older history compressed by layer-specific 4:1 indexed (top-k up to 512) or 128:1 unindexed regimes; mHC, shared/routed MoE and trained DSpark support. V4.1 Flash has a large ~189 GiB Engram table in DwarfStar's Q2 file kept disk-only; different checkpoint/graph and tokenizer. Keep variants separate and checksum draft support files. Do not confuse DwarfStar quantized Q2 (81 GiB V4 Flash main) with official mixed FP4/FP8 model.

## 4. Initial capability matrix (PLANNED — not supported yet)

| Family | CPU semantic | AMD gfx1151 | Discrete AMD | NVIDIA CUDA | Apple Metal | Vulkan |
|---|---|---|---|---|---|---|
| Dense Llama | PLAN | PLAN | PLAN | PLAN | PLAN | PLAN |
| Gemma 4 dense/MoE | PLAN | PLAN | PLAN | PLAN | PLAN | PLAN |
| Qwen3.5 hybrid | PLAN | PLAN | PLAN | PLAN | PLAN | PLAN |
| Qwen3.8-Flash-Next | PLAN | extreme research target | PLAN | PLAN | PLAN | optional |
| GLM-5.3-Flash | PLAN | extreme research target | PLAN | PLAN | PLAN | optional |
| DeepSeek V4 family | PLAN | PLAN | PLAN | PLAN | PLAN | optional |
| Vision/audio | PLAN | separately qualified | separately qualified | separately qualified | separately qualified | separately qualified |

`PLAN` is not feature support. Exact model/quant/version/backend cells must migrate to `FUNCTIONAL`/`VERIFIED`/ `PERFORMANCE_QUALIFIED` only after tests. A generic GPU backend may be incomplete; CPU reference should be first for each semantic operator.

## 5. Model adapter shape (conceptual Rust)

```rust
struct ModelIdentity {
    family: ArchitectureId,
    checkpoint_sha256: [u8; 32],
    tokenizer_sha256: [u8; 32],
    quant_manifest_sha256: [u8; 32],
    semantic_version: u32,
}
struct ModelRecipe {
    identity: ModelIdentity,
    layers: Vec<LayerRecipe>,
    tensors: Vec<TensorSpec>,
    states: Vec<StateSpec>,
    positions: PositionRules,
    prompt_contract: PromptContract,
    modalities: ModalitySupport,
    draft: Option<DraftRecipe>,
}
trait ModelAdapter {
    fn recognize(&self, verified_metadata: &ModelMetadata) -> Result<ModelRecipe>;
    fn build_semantic_graph(&self, recipe: &ModelRecipe) -> Result<ModelGraph>;
}
```

Implementation should avoid storing model weight buffers inside `ModelRecipe`; it describes topology and stable TensorIds, while the ModelArtifact store owns backing bytes, and the runtime manager owns resident allocations. A `Session` owns only mutable state, not a copy of all weights.

## 6. Integration steps for an unfamiliar model

1. Pin official source model card/config and tokenizer code; hash the intended checkpoint and its legal distribution.
2. Produce a tensor manifest (names, shapes, stored quant, offsets, dimensions), reject mismatches.
3. Decompose layers into existing exact semantic ops; list missing ops explicitly.
4. Implement missing reference ops including state and multimodal positions; add deterministic small fixture.
5. Build Semantic IR recipe and validate shapes, masks and state effects.
6. Compare teacher-forced per-layer activations and full logits against independent reference in FP32/BF16/FP8 depending on artifact, then quantized format.
7. Add general backend lowerer and correct session prefill/decode, chunks, cache, sampling and cancellation.
8. Add optimized candidate kernels *after* measuring bottlenecks, validate output and quality.
9. Re-evaluate support across backend x context x quant x modality cells.
10. Publish known limitations; never claim blanket model family compatibility based on one small variant.

## 7. Numerical contracts to encode

LayerNorm/RMS epsilon and accumulator dtype; RoPE theta/sections/scaling; attention masks/top-k/ties/softmax reduction precision; router sigmoid/softmax/scaling/shared expert order; quant block types/scales/outlier layouts; hyperconnection and state update order; model's sample filtering and position advancement. A stored model's precision and a compute repack are separate. Fallback cannot arbitrarily reinterpret latent KV or recurrent matrices.

## 8. Future compatibility principle

A new model can run as soon as its novel math/state semantics have a verified reference implementation plus any required backend lowering. It need not wait for extreme kernels. But if no backend supports an operation, it must fail clearly rather than produce plausible nonsense.

## 2026-10-10 integrated v2 addendum — model recipe/compiler decoupling

The research original `ArchDef + ModelSpec` has been adopted as an excellent **model registration and validation** mechanism, but `ModelSpec::kernel_plan(hw)` is removed from authoritative ownership. A recipe declares the operator/state graph and exact tensor roles; the compiler/kernel provider selects implementation. Registry status distinguishes **recognized artifact**, **semantic model supported**, **runnable on selected backend**, **numerically validated**, **performance-qualified**.

**Modern explicit families remain required goals:** Llama, Gemma 4, Qwen3.5+, Qwen3.8-Flash-Next, GLM-5.3-Flash and DeepSeek V4 variants. The older research's baseline Llama/Qwen2/Qwen3/Gemma3/Mistral/Phi is a pragmatic early CPU/portability test set, **not** a replacement for user-required newer families. New model geometry comes from version-pinned official source/config; do not assume 48/45-layer examples apply to adjacent versions. Gufo is an additional `gfx1151` Qwen source absent in older research corpus.

**Model capability must include runner type:** decoder-only text generation, encoder-only embedding, cross-encoder rerank, multimodal processor/encoder/projector, and adapters (LoRA) with distinct per-session cache identities and possibly draft compatibility constraints. `PoolingSpec` requires no generation/KV loop. Vision/audio spans have explicit causal/bidirectional attention behavior and modality fingerprint; a new video/image model cannot just inject fake text token IDs.

**Loading invariant:** reading a valid GGUF/safetensors with 43+ known type IDs is *not* evidence any arbitrary trained op is implemented. For a truly unknown semantic op, report `UnsupportedOperator` and require a new reference op/state contract. No implicit HF remote-code execution. GGUF dialect, tokenizer pre-type, quant layout, template and source checkpoint must match the model recipe.

**Model admission proof:** per checkpoint+quant+backend produce golden layer activations/logits, tokenizer exactness, native positions and state-after-every-chunk, tool/grammar and (where applicable) image encoder/rerank quality. Include an explicit forced kernel-missing fallback test for each new semantic family. See `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md`.


## 2026-10-12 integration — Gufo-family semantic reuse and task control graphs

**Upstream Gufo is source evidence, not a new mandatory runtime.** The original `gufo-org/gufo` model implementations include Qwen3.8 Flash-Next hybrid/GDN/QSA, Qwen3.8-27B and DFlash2, DeepSeek V4/DSpark, ASR, TTS and emerging diffusion/video workloads. A `ModelRecipe` still declares exact semantic operators, tensor shapes and mutable state independent of GPU/NPU/compiler. The specified `mikealanni/gufo` fork's XDNA2 offload is a **kernel placement choice for pure projection subsets** within one Qwen recipe, not a new `NpuQwen` mathematical architecture or different MoE top-k.

Add model-aware **`TaskProgram`** definition distinct from core mathematical SIR: `TaskKind::{TextGenerate,AudioAsr,AudioTts,ImageGenerate,ImageEdit,VideoGenerate,Embedding,Rerank}`, stage DAG and stage-specific control loop (autoregressive tokens, encoder segments, vocoder frames, flow/denoise timesteps), mutable states, output stream and quality contract. TaskKind is extensible and a status/capability, not evidence any one model works. `TaskProgram` shares SIR/RIR/KIR/CIR but does not force different models into causal text KV or default sampler semantics.

For multi-device NPU acceleration, a compatible `Projection` shape may be split by **output row spans** under RIR legality and a partial-result layout join; exact model config, quant scales and all downstream GDN/indexer state operations remain invariant. An unknown AIE kernel or unsupported numerics must fall back GPU/CPU, not change model semantics. Reference tests should include source Gufo Qwen hybrid projection/ASR/TTS model family vectors where redistributable, and full GP U-only parity before any NPU candidate.

## v4 refinement — registered exact mathematical models and future parameter versions

`ModelDefinition` builds one versioned `SemanticProgram` from verified checkpoint config, tensors, tokenizer/positions and registered exact operators, independent of target kernel/device/task. Two model variants with different attention gate, trained router, compressed indexer or vocabulary semantics require distinct exact op schemas and CPU reference; a shared marketing family is not enough. A novel model built from existing semantics needs only manifest/recipe and fixtures, not a new server or scheduler.

Model **compatibility matrix** now separates semantic source, TaskDriver stage/control, numeric precision, shape modes and kernel qualification; one `FUNCTIONAL` model/device pair may not be `PERFORMANCE_QUALIFIED`. An unsupported dynamic shape may lower to generic reference if correct operator implementation exists, not trigger uncontrolled JIT. Future optional ParameterVersion/TrainStep cannot mutate immutable weights of active inference sessions; new adapter/model generation creates a separate cache/compiled bundle compatibility key.

## Current metadata-only source profiles

These profiles bind configuration/layout, not executable support. Current
implementation and required trained tests live in [model-execution/TODO.md](../plans/model-execution/TODO.md).

**Qwen3.8 27B:** official `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, declared qwen3_5_text,64 layers
(48 GDN/16 full), hidden5120, FFN17408,24Q/4KV width256,16 GDN key/48 value
heads width128, vocabulary248320 and untied output head. The metadata index has
1199 tensors/18 shards, with851 ordinary text roles. This does not verify actual
shard headers/bytes; vision and MTP are separately admitted components.

Original HF grouped V axes differ from converted tiled axes. The converted
profile permutes V-related projections/constants/convolution/output, applies
-exp(A_log) and adds1 to non-GDN norm weights. Bind the selected profile
explicitly. `output_gate_type=swish` denotes GDN SiLU; full attention still uses
sigmoid. Exact shard-metadata integer parsing accepts `9007199254740993.0`
without precision loss. It rejects all exponent notation (including `1e3`),
nonzero fractional parts and u64 overflow.

**Gemma4:** preserve official source revisions in the
[family technical contract](../research/GEMMA4_FAMILY_CONTRACTS_2026-10-16.md).
E2B has35 layers,15 KV owners and20 consumers reusing compatible producers13/14,
plus packed per-layer embeddings and doubled later FFN width. A missing optional
weight index is a single-file observation, not complete header/name validation.

The26B variant has hidden2816,30 layers,128 experts/top8, expert width704 plus
dense2112 in every layer. Five sliding layers then one full: sliding16Q/8KV
width256/window1024; full16Q/2KV width512 with shared raw K/V and separate norms.
Full RoPE rotates128/512;26B has neither PLE nor shared-KV-layer ownership. Expert
gate_up `[128,1408,2816]`, down `[128,2816,704]` and router scale roles stay exact.

Routing dataflow is residual, unscaled RMSNorm, learned router scale,
inverse-sqrt-width, projection, FP32 full-row softmax, top8, selected
normalization and expert scale. Experts have separate pre-FFN norm and
GELU-tanh gate×up; dense/expert postnorms precede common postnorm, residual and
layer scalar. Framework-specific top-k/reduction/scale placement differ: choose and
independently qualify the original arithmetic before implementing routing.
Analytic test-local cases do not establish trained numerical parity.

These contracts retain the executable-source4GiB limit. Large sources need
admitted immutable owned shards with actual header/dtype/name/offset/content
binding, not a mutable mmap/private-COW shortcut or another source issuer.

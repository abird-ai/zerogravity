# Research Track D — Image Generation, Editing, Flow/Diffusion and Autoregressive Image Models

**15 October 2026 · Design/source research, not runnable model support.**  
**Scope:** Qwen-Image, rectified-flow image variants (large dev and small distilled/base), AR/hybrid image models, multimodal DiT latent-diffusion models and newer/future frontier image providers. Source variants are not interchangeable.  
**Architecture authority:** Unified v5.1 model assembly, source-legal TaskDriver iteration, typed ValueDescriptor and single TaskCoordinator.

## 1. Verdict

An image model is **not** equivalent to "input text, iterate diffusion and return JPEG." Contemporary open architectures include MMDiT diffusion/flow, rectified flow with variant-specific LLM text conditioners, image edit reference branches, causal/multimodal autoregressive image/codebook generation and potentially mixed-stage reconstruction. The reusable engine must support semantic latent tensors, source-registered samplers and complete decoded images, without a separate image inference runtime.

The most important insight: **flow schedule, denoising prediction target, seed/RNG, guidance and conditioner/decoder identities are source-model semantics**. Changing step count or text encoder to make a smaller GPU run faster may alter output distribution, not constitute an equivalent physical kernel.

## 2. Source model matrix and component inventory

| Family and variant | Verified public architecture characteristics | Assembly and exact-variant risks | Official sources |
|---|---|---|---|
| Qwen-Image original (~20B MMDiT) | image generation/editing with multimodal diffusion transformer and text conditioning, strong rendered text | image editing image-reference processor and latent VAE, MMDiT exact sequence, Qwen family source tensors | model documentation |
| Qwen-Image later 2.0 variants | upstream release adds new image model/configuration variants | do NOT assume original 20B checkpoint attention, processor or numeric schedule apply unchanged | model documentation |
| Rectified-flow image model (large dev variant, 32B) | rectified-flow multimodal image generator/editing, Mistral-class text conditioner, latent generator and VAE | large conditioner residency and reference images; non-commercial weight license; source step rules | model documentation |
| Rectified-flow image model (small distilled/base variants, 4B/9B) | different conditioner based on Qwen3 embedding, distilled 4-step vs base many-step recipes; variant licensing | Do not pair the [dev] Mistral encoder with the distilled variant's Qwen embedding, do not silently treat distillation as equivalent solver | model documentation |
| Multimodal DiT latent-diffusion model (large/medium/turbo variants) | multimodal DiT with multiple text encoders (CLIP-L/OpenCLIP-G/T5-XXL), dedicated VAE latent decoder; version-distinct ControlNet | exact text encoders, QK/norm and timestep/CFG semantics, 16-channel VAE path, source license for weights | model documentation |
| Unified multimodal AR MoE image model | upstream documentation states unified multimodal AR MoE; source generation graph is checkpoint/branch specific | needs exact source audit; don't hardcode either pure AR image-codec or generic diffusion stages | model documentation |
| Hosted GPT-Image and other hosted image providers | opaque provider generation/editing APIs | TaskServiceAdapter with exact input asset/rights/usage/cost/async policy, not native regions | provider documentation |

**Release and license verification required:** The Qwen-Image source repository includes multiple generations; the small rectified-flow variants may use different terms from the larger and dev variants; source code availability ≠ commercially redistributable image weights. The AR/hybrid image model has divergent discussions of its AR image generation pipeline; inspect exact generate_image call tree for the **selected checkpoint revision** before freezing source semantics.

## 3. Verified image component assembly contract

Typical flow/diffusion assembly:
~~~text
Source input text (+ mask/reference images)
  -> validated tokenizer / text conditioner(s) (CLIP / T5 / Mistral / Qwen family)
  -> conditioned embeddings (+ image reference encoding / attention masks)
  -> sampled initial latent + source RNG/seed
  -> source timestep/sigma schedule, solver integrator and guidance branch(es)
  -> MMDiT / DiT / flow predictor repeated at validated stages
  -> final semantically valid latent state
  -> checkpoint-bound VAE decoder / image processing
  -> pixels with explicit color primaries, channels and output dimensions
  -> encode image packet/container if requested
  -> StateRoot single publication and output receipt
~~~

**No automatic architectural assumption**: for an AR image/codebook model, replace the solver loop with the source-verified image token/codec loop and decoder. For a hybrid AR→latent predictor, the assembly has both semantic stages. TaskDriver chooses the correct algorithm, not the endpoint named images/generations.

Version and hash required for every node: checkpoint, encoder processor/tokenizer(s), VAE, scheduler, optional edit encoder, LoRA/ControlNet, prompt optimizer and any external host dependencies. Allowed type edges may include pixel→VAE latent or raw text→tokenized embed; dimension match alone does NOT mean compatible.

## 4. Latent tensor and image value semantics

~~~rust
struct ImageValue {
    width: u32, height: u32, channels: ChannelLayout,
    component_type: PixelType, color_space: ColorSpaceId,
    alpha: AlphaMode, orientation: Orientation,
    data: PixelBufferLease,
}
struct ImageLatent {
    batch: u32, channels: u32, latent_width: u32, latent_height: u32,
    patch_layout: RegisteredLatentLayout, vae: VerifiedDecoderId,
    source_rng: RngStateId, iteration: SolverCursor,
}
~~~

Image tensor shapes need bounded element count, alignment, stride, patch packing, masks/crops and safe decoded-byte checks before device allocation. Image edits may have one or more reference images with different dimensions and region masks. Pixel color space, ICC, gamma and alpha cannot be silently discarded when decoding from latent.

A latent is **not** final output; never stream partial denoising state as a finished image. Progressive previews are allowed only when client explicitly opts into provisional preview semantics and model TaskDriver can construct them without leaking uncommitted neural state.

## 5. Flow and denoising semantics must be exact

For each checkpoint store:
- prediction target: velocity / epsilon / x0 or registered alternative;
- timestep/sigma schedule and indexing; exact start/end and interpolation;
- integrator (Euler, Heun, DPM family or source schedule), update order and required precision;
- noise RNG algorithm, counter/key, seed, shape, negative-conditioning and guidance branches;
- CFG weight and any "guidance distillation" convention; text/image/reference conditioning masks;
- learned embeddings and positional/rotary coordinate semantics;
- VAE latent scaling/shifting and source decoder/crop;
- whether step distillation/turbo supports alternate step count and under which quality mode.

**Wrong but plausible image is a failure.** A GPU candidate that uses Euler where original reference uses Heun, flips noise direction, omits guidance, changes VAE scaling or silently truncates a reference image must fail source qualification, even if visually acceptable.

A source TaskDriver may run many solver steps in a prepared device-native region or graph, then return prepared latent version and actual GPU completion. TaskCoordinator alone publishes root and final image. When a request is canceled midway, device kernel completes or quarantines; previous committed image/state remains uncorrupted.

## 6. Qwen-Image source-specific trace

1. Validate exact Qwen-Image variant/release, 20B original versus later Qwen-Image2.0 family.
2. Load source-bound text processor/conditioner and image editing reference processor if applicable.
3. Encode prompt preserving text layout/casing/Unicode direction needed for typography.
4. Prepare source latent representation, timestep/guidance and RNG state.
5. Execute MMDiT with correct cross-modal attention, normalization, positional embeddings and input/reference masks.
6. Run exact diffusion/flow source scheduler; after each full legal iteration, mark provisional latent state with a monotonically checked solver cursor.
7. Decode through matching VAE; source output dimensions/color and trim only by declared processor.
8. Publish one final image and source/numeric evidence; optional previews tagged provisional and never conflated with final.
9. For editing, changes to reference image or mask invalidate related source conditioning cache keys; no reuse of incompatible embeddings.

**Negative:** wrong image processor/text layout, mismatched VAE, truncated reference array, changed guidance schedule, incorrect timestep ordered expert/head.

## 7. Rectified-flow image source-specific trace

**Large dev variant, 32B**: source-backed flow transformer with a Mistral-class conditioner and substantial memory requirement. Text/image reference processing and prompt upsampling must be **explicit**. A remote conditioner can be a separately authorized stage, not covert egress.

**Small distilled/base variants, 4B/9B**: conditioner based on Qwen3 embeddings and distinct model sizes. Distilled 4-step variants and base ~50-step variants change training/sampler assumptions. Must pin exact Qwen3 embedding revision and dimension; using the large variant's encoder is an ABI error, not acceptable fallback.

Common trace:
1. Resolve source manifest → variant-specific VerifiedModelAssembly.
2. Encode prompt and references; retain original conditioning embeddings with verified source/processor keys.
3. Seed noise under allowed random policy; bind latents to source VAE geometry.
4. Predict flow velocity with source DiT and timestep conditioning, apply registered integrator and any source guidance.
5. Decode/crop/format final image and publish. Cache warmness/residency across requests must not leak user reference images.
6. Track conditioner GPU residency independently of denoiser to support offload and device placement; admission accounts the whole assembly.

**Falsification:** selecting the image model by name alone and injecting the [dev] Mistral embedding into the distilled variant (or vice versa) must refuse before any kernel. Distilled/base scheduler cannot silently be swapped because same tensor hidden size.

## 8. Multimodal DiT latent-diffusion source-specific trace

The original reference implementation identifies CLIP-L/14, OpenCLIP bigG, T5-XXL, 16-channel VAE and MM-DiT core. Not every deployment keeps every encoder resident or uses identical prompt pooling; source variant manifest must identify required components.

- Distinguish Large, Medium and Large-Turbo weight/semantic/numeric configurations.
- Text encoders are independent verified models with their own tokenizer, mask, pooling and output axes; combine exactly per source.
- VAE decode latent normalization, image channels, output cropping and color range are source-defined.
- QK normalization and attention-layer variants matter to mathematical identity; a generic Transformer block must not silently omit them.
- ControlNets and adapters are source-versioned components, not arbitrary tensor patches; their injection paths and residual scales are part of assembly state.
- Turbo fewer-step outputs are distinct quality model modes; do not market as equivalent fused implementation.

## 9. AR/hybrid image model ambiguity protocol

The upstream README describes AR multimodal generation with MoE. However general high-level media architecture descriptions are insufficient to decide the exact released \`generate_image\` pipeline for all variants.

Before native implementation:
1. Pin exact branch/commit, weight files, variants (Instruct/distilled/source).
2. Trace official entrypoint into model forward, image special token/codebook decoding and any extra denoise/flow/refiner or reconstruction.
3. Capture source equations and necessary processors for each stage.
4. Determine exactly which operations are AR, which are one-shot decoder, which are iterative solver; document mandatory RNG/state effects.
5. Register a ModelAssembly and TaskDriver matching that trace. **Never** force a global DiT solver or pure decoder AR based only on branding.
6. Verify synthetic source-specific shape/time/codec tests followed by original checkpoint tests. Until complete mark algorithm details Unknown/UnderAudit rather than inventing.

This model is an adversarial architectural proof that one coherent fabric can support distinct algorithms under one request without a new server.

## 10. Image editing and composability

Potential components: prompt language understanding, mask segmentation, text/vision encoders, image-to-latent, reference-conditioned generator, inpaint/outpaint attention/mask, VAE and optional super-resolution/upscaler. One task may combine these under a verified original assembly or app-level workflow depending on source. An external "remove background then generate" is NOT silently equivalent to a trained joint editing model.

Typed edits should include:
- input image(s) count, required ordering and roles;
- dimensions, crop coordinates, alpha/mask semantics, orientation and color space;
- allowed transformation, output dimensions/aspect and source step/guidance;
- whether input content leaves device/local machine;
- output rights and provider receipt for hosted APIs.

## 11. Physical memory and performance strategy

Cost dimensions: conditioner weights, denoiser weights, VAE weights, reference encoder, source latent+CFG doubling, attention scratch, tile reconstruction and output pixels. A 20B/32B text+image model cannot fit solely because the DiT weights fit; the complete assembly must be budgeted. Discrete/UMA storage and offload should be planned at stage boundaries.

Potential optimizations, all subject to proof:
- Independently resident conditioner shared by exact source manifest, memoized on permissible public prompt/cache key.
- Device-native iterative steps in compiled bundle to reduce host per-step scheduling while retaining typed solver state.
- VAE tiling with verified overlap/padding and deterministic output comparison.
- Tensor/sequence parallel large MMDiT with correct cross-device ordered dependencies.
- Quantized checkpoint original encoding and source numerical policy; not silent "use a cheaper quant".
- Long-running batch job QoS with video/voice tasks coexisting via legal scheduling step boundaries.
- Explicit warm/cold performance and peak physical resource reporting.

## 12. Required image acceptance

**MM-IMG-01:** Qwen-Image original vs later source variant require distinct manifests; mixed processors fail.  
**MM-IMG-02:** [dev] Mistral and distilled-variant Qwen3 conditioner mismatch refuses at load.  
**MM-IMG-03:** multimodal DiT multiple text encoders with different shape/mask; skip required encoder hard fails.  
**MM-IMG-04:** exact solver/CFG/seed/step count changes reproduce expected source reference; different schedule reported as distinct quality mode.  
**MM-IMG-05:** incorrect VAE crop/scaling produces error before final image; no unverified preview publication.  
**MM-IMG-06:** inpaint input mask/alpha/ICC survives image processor semantics, doesn't leak ignored/reference frames.  
**MM-IMG-07:** whole ModelAssembly physical memory admission; conditioner/latents/VAE all charged, never double-admitted.  
**MM-IMG-08:** dropped GPU Future mid-solver fences physical handles without stale StateRoot, retains last valid image.  
**MM-IMG-09:** parallel images from two users share immutable conditioner weights but not latent noise/RNG/private references.  
**MM-IMG-10:** AR/hybrid image model exact generate_image graph audited before labeling AR/Diffusion support.  
**MM-IMG-11:** output PNG/JPEG/WebP color/orientation and alpha validated against pixel type.  
**MM-IMG-12:** unsupported external hosted "frontier image" source not masquerading as local inference; egress consent enforced.

## 13. Recommended implementation order

1. Source-free validated conditioner→flow predictor→VAE mock assembly with source-bound RNG/solver.
2. Small verifiable open flow model checkpoint with original CPU reference and GPU-tiling correctness; candidate small distilled rectified-flow checkpoint if license/hardware permits.
3. Multimodal DiT latent-diffusion and Qwen-Image variant adapters using reusable ops while preserving their distinct source processors.
4. Editing/multi-reference and source-specific ControlNet/LoRA components.
5. The AR/hybrid image model's exact audited AR/hybrid recipe and optimizer proof, not before source semantic clarification.
6. Hosted frontier providers through TaskServiceAdapter independently from native image math.

The existing user CPU→GPU Qwen3.5/Gemma/Qwen3.8 implementation priority is NOT automatically reordered by this research roadmap.

## 14. Primary sources checked

- Qwen-Image model documentation.
- Rectified-flow image model documentation and license.
- Multimodal DiT latent-diffusion source.
- AR/hybrid image model documentation.
- Hosted image provider API documentation.
- OpenAI model/API catalog: https://developers.openai.com/api/docs/models

Research gathered and reconciled 15 Oct 2026. No checkpoint weights downloaded/tested, no license inferred from GitHub repo existence, no native image support implemented.

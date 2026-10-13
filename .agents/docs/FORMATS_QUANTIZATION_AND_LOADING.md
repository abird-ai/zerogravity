# Model artifacts, quantization, tokenizer/template parity and safe loading

**15 October 2026 — VERIFIED SOURCE EXTENTS:** [Unified SSD/resident TensorView](UNIFIED_VIRTUAL_TENSOR_RESIDENCY_AND_SSD_STREAMING_2026-10-15.md) binds each immutable source or derived-quant tensor slice to an opened verified ArtifactId+shard+offset+byte length+codec+transform, not a mutable filename. The same logical source tensor can materialize from bounded local SSD/host/UMA/GPU without changing its quant identity. Mixed-size Q2/Q4/EXL3 experts demand exact offset/scale integrity and one global byte budget, not full-pack re-dequantization or size-class cache resets. Qwen n-gram/DeepSeek Engram large row lookup is separately source-indexed and disk-primary, never required whole-table VRAM residency. Immutable GGUF file descriptors survive path replacement; partial/short/unsupported direct IO is a host error **before GPU submission**. See the source research notes, tests SSD-01..07, SSD-17..25.


**15 Oct specialization precision addendum (design, not implemented):** [EXL3/QTIP and derived quantization](DERIVED_QUANTIZATION_CONTRACT_2026-10-15.md) specifies packed trellis/codebook block/source scale and optional transforms independent of architecture, **three different identities** (original checkpoint, lossy derived quantization, lossless device packing), reference oracle and mixed per-tensor bitrates. Offline shaped LDLQ sensitivity, source-representative trace and held-out KLD/quality are experimental P20 evidence, not runtime auto-quantization. P03 must fail unknown EXL3 K/codebooks, E8M0 formats, missing sidecars or corrupt scale before physical GPU submission; quality, target speculation and raw weight fidelity are separate receipts. Tests QTIP-01..14 apply.


> **v4 governing contract (2026-10-12):** Detailed technical mechanisms in this supporting spec remain useful, but the [minimal architecture](MODULAR_AI_SYSTEMS_REFERENCE_ARCHITECTURE.md), [corrected composition interfaces](COMPOSITION_AND_EXTENSION_CONTRACTS.md), [execution/state protocol](MINIMAL_EXECUTION_SPINE_AND_PROTOCOL_V4.md) and [task/compiler refinements](TASK_PROGRAM_AND_COMPILER_REFINEMENT_V4.md) **supersede older sketches** whenever they conflict. In particular: **StateDomains only Prepare**; TaskCoordinator performs one PublishRoot; CIR only reports readiness; global resource admission is coarse with device-local allocation; TaskDriver is trusted minimal stage control; RIR/KIR are transient compiler views; GPU-only fallback and numeric/safety gates remain. No implementation is asserted.


## 1. Design constraints

Model format and numerical representation are orthogonal to the runtime's execution layout. Loading one GGUF or Hugging Face safetensors model should not require a Python inference server. However the loader can only execute **known semantic architectures and verified tensor layouts**. Never silently invent formula for an unknown operation or execute arbitrary `trust_remote_code`.

A model artifact fingerprint binds:
`architecture_id + config_semantic_hash + tokenizer/template_hash + tensor_manifest_hash + checkpoint_sha256 + quant_recipe_hash`.
The file's name is not identity. Partial/split GGUF or safetensors shards must be checked as a complete set, with offsets, hashes and duplicate/missing tensor detection.

## 2. Architecture of the loading pipeline

```text
Local path / HF model ID
  -> controlled artifact resolver/cache (explicit network when authorized)
  -> format detection and structural parser (bounded sizes)
  -> tensor manifest + tokenizer/template fingerprint
  -> adapter registry exact architecture validation
  -> SemanticModelIR recipe, state and position schema
  -> physical bytes/quant layout and conversion plan
  -> memory/resource reservation and packed device tensors
  -> executable compiler plan + reference validation if enabled
```

No GPU allocation or expensive network fetch should occur before basic model config/format validation and a credible safe memory estimate. Never use untrusted metadata to allocate `num_layers * num_experts * huge_dimension` without checked overflow/hard limits.

## 3. GGUF

Support recognized GGUF header fields, tensor metadata, alignment and quant schemes explicitly. Each tensor has logical dimensions, stored offset/bytes and quant block metadata, validated against model adapter's expected TensorSpec. GGUF is not automatically an executable graph.

Quant support should be incremental and model-verified: F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q2_K, IQ2_XXS, FP8, MXFP4 and additional types when needed and correctly implemented. Distinguish block format's logical input row length from stored padded stride. A 640 logical dim padded to 768 for an optimizer is not 768 semantic input features.

Format parsers need:
- little/big-endian correctly handled where specified;
- bounded key/value string/array lengths;
- non-overlapping tensor regions (or explicit format-supported alias);
- alignment and seek overflow safety;
- duplicate names and missing mandatory tensors detection;
- consistency with tokenizer/architecture config and manifest;
- checksum/written-partial-file safety and source provenance;
- support for mapped read-only weights when safe;
- clear error for uncommon tensor types rather than fallback to reinterpretation.

## 4. Safetensors / HF

Load explicit `config.json`, `tokenizer.json`/other declared tokenizer assets, safetensors index/shards and optional separate vision/audio/mtp modules. Validate tensor dimensions and shard order with a model adapter. Model-specific parameters such as `rope_theta`, sparse indexer geometry, conv widths, GDN/KDA gates, number of experts/top-k, mHC iterations and native context must be declared exactly.

HF model IDs should resolve to a pinned revision/commit or immutable cache digest. Downloads resume with atomic staging, checksum and bounded disk requirements; `--offline` mode uses local verified cache. Avoid implicit arbitrary external fetch of `image_url` or model repo hooks under service requests.

Hugging Face custom model Python code is not executed; a new mathematical op requires explicit Rust adapter/operator implementation. Use official Python/Transformers only in **offline developer reference QA**, not normal runtime serving.

## 5. Tokenizers, chat templates and agent semantics

Tokenizer parity is part of model correctness. Requirements:
- byte-level / BPE / SentencePiece / other supported tokenizer algorithms with verified Unicode/special token offsets;
- BOS/EOS/PAD and tool call sentinel handling;
- reasoning on/off variants and role delimiters;
- model-specific tool argument formatting and streaming parser;
- exact token prefix versus rendered text: hidden reasoning/tool turns may not match visible client text;
- Unicode normalization and invalid UTF-8 safe behavior;
- multimodal token spans and associated image/audio preprocessing hashes and positions.

Test official model tokenizer vectors, long tool histories, multiple languages/emoji, special-token injection and byte boundary streaming. Never assume OpenAI/Anthropic role schema maps one-to-one to every model's native template; adapters must render exact tokens.

## 6. Distinguish storage precision from compute format

`StoredQuant`: checkpoint's semantic quant bytes, scales and block encoding. `PackedLayout`: lossless reorder of those bytes for device locality, WMMA shape, alignment and group storage; must round-trip or have exact dequant parity. `RequantizedArtifact`: deliberately different precision, with its own checkpoint identity, imatrix/calibration/eval and quality loss; may change output.

A performance benchmark comparing an IQ2 checkpoint against BF16 changes both engine and model-quality conditions. Report quant bit counts/bytes, main/draft/table sizes, working set resident, and quality delta. Never auto downgrade from Q4 to Q2 under memory pressure; request a different file/artifact/policy explicitly.

Tensor role matters: dense always-active projections may be worth higher precision or faster packed layout; routed experts may tolerate lower bit widths; attention/indexers/routers/mHC may be sensitive to quant error. Model-specific imatrix-aware recipes can minimize quality loss but require independent evaluation. No one quant recipe fits Gemma, Llama, GLM and Qwen equally.

## 7. Conversion, calibration and verification pipeline (optional developer tooling)

- Source checkpoint manifest with SHA/model revision and license.
- Convert tensor geometry to GGUF or internal packed representation without changing numerical precision by default.
- Quantization tool may read calibration corpus/imatrix, apply chosen recipe, pack exact types, emit per-tensor quant metadata and audit.
- Verify every output tensor: shape, logical/padded dims, checked file offset, sample dequant error, full conversion manifest, tokenizer match.
- Publish quality table: reference NLL/KL/logit RMSE, first-token agreement, greedy prefix matching, coding/tool/multimodal long-context tests as appropriate.
- Save exact conversion toolchain/compiler revision and source dataset license.
- Avoid distributing copyrighted/unlicensed checkpoint weights or personal prompts automatically.

## 8. Multi-file and huge-file support

Weights can exceed physical RAM and span sharded files. Do not load all shards into an extra host arena before GPU upload. Use lazy mapped metadata and bounded read batches, appropriate readahead, checksums and verified staging. Storage performance needs measurement: n-gram BF16 table row reads vs 11 MiB expert blob reads have different access patterns.

Model switching must release safely after all sessions/events retired. A process may host multiple models only when global ResourceArbiter proves capacity and maintains separate model/checkpoint identities.

## 9. Layout transform and fallback

Compiler can select a device-specialized packed layout for fast Qwen/GLM kernels, but CPU/generic fallbacks require a canonical conversion contract or independently matched layout. Layout transitions need an explicit data movement cost and legal cut point; no direct reinterpretation of compressed vs expanded attention state. Converted tensors cache keyed by artifact hash, quant schema, kernel layout ID and compiler version.

## 10. Adversarial input tests

Malformed GGUF/safetensors, duplicate metadata, unknown type, overlarge dimensions, integer overflow, negative/NaN quant scale, broken shards, symlink/path traversal, truncated n-gram or expert table, invalid tokenizer UTF-8, emoji/special role injection, mismatched MTP sidecar, missing vision encoder, illegal context extension and stale artifact cache. Each must fail without OOM, arbitrary file read or running unsupported inference.

## 11. Open policy items

User-chosen automatic quant selection (if any), default HF network resolver behavior, cache retention, license distribution, trusted plugin adapters, maximum artifact size and default tokenizer library. All remain proposals, not explicit user decisions.

## 2026-10-10 integrated v2 addendum — complete product-model pipeline

An earlier architecture note and implementation blueprint provide many detailed contracts beyond initial file parsing. Adopt their **safe GGUF dialect/type rules, byte-exact tokenizer pre-types, Jinja/minijinja rendering, grammar-FSM/GBNF, content-addressed model store, imatrix calibration and derived acceleration-pack concept**, subject to current license/toolchain checks. See `PRODUCT_MODEL_PIPELINE_AND_EXTENSIONS.md` for consolidated owners and capability matrix.

An optional native expert/quant pack (historical working name "APK") is an acceleration **cache** referenced by original GGUF/safetensors hashes, never the only runnable input. Public extension/name not fixed; avoid Android APK collision. Its manifest can index one-blob-per-expert data, packed CPU/GPU layouts, KernelPlan hash and quality/speed overlays. Source and quant conversion must be independently verified for exactness or labeled as a *new quantized checkpoint*; don't silently choose lower precision to meet RAM.

**ModelStore obligations:** immutable SHA-256 blobs, versioned manifest list, sharded model source, model/LoRA/vision/MTP sidecar identity, resumable atomic downloads, offline mode and reachable-set GC, `--inspect/doctor` errors and privacy-safe prompt cache. All source weights/sidecars may carry separate license conditions. No `trust_remote_code` at runtime. Token IDs/template, adapters, special tokens and modality spans are part of semantic model identity and prefix compatibility.

**Old research defect:** its disk-KV ABI explicitly calls a listed 54-byte collection a 48-byte header and uses an impossible 5-byte string represented as u16. Do not copy/rewrite this broken struct into model loading or snapshot readers. Correct new version requires checked byte-count math, schema hash, collision-resistant identity and negative parsing tests.

**New model integration:** add `ArchDef/ModelRecipe` with exact tensor validation, SIR mathematical operators, CPU oracle and tokenizer/native template; attach to generic backend; only later register specialized KIR kernels. No model-specific `kernel_plan(hw)` inside semantics, no guarantee an arbitrary GGUF/HF file runs when mathematical ops are unknown.


## v4 refinement — independent artifact, semantic, state and bundle identities

Version **ArtifactKey** (checkpoint/shards/quant/tokenizer/template/adapter/sidecars), **SemanticKey** (op/formula/state/position contract), **StateCompatKey** (task/provenance/frontier/quant/adapter/media/tenant), and **BundleCacheKey** (compiled device/driver/shape/numeric/codegen). **Runtime session/batch slot indices must not appear in compiled bundle key**, or cache explodes. Compiled driver handles, GPU pointers and EventIds are never serialized with model weights or state snapshots.

Artifact loader is trusted data parser, not a loader of executable model code. Recognized GGUF/other tensor type does not imply registered mathematical op/TaskDriver support. An exact lossless repack differs from temporary low-precision arithmetic and from a new requantized source checkpoint; source/model quality must identify the difference. Default import lacks arbitrary external code execution and global kernel plugins.

## U-048 — packed quantization and NPU derived-encoding eligibility

[Packed-encoding qualification](HETEROGENEOUS_PREFILL_AND_RETAINED_REPLAY_CONTRACT_2026-10-15.md) covers packed affine/codebook quantization, transformed weights,
role-specific recipes, overlays and component sidecars as **source-qualified codec
and assembly candidates**, each with independent parsing and numerical qualification.
Independently validate every supported header/scale/zero/transform/padding and
sidecar binding. Material W8 requantization requires derived-encoding identity
and the permitted numeric/quality contract; native low-bit source bytes remain
immutable. KV compression is independently keyed state arithmetic. Source-known
or parseable formats do not imply model, device, quality or feature admission.

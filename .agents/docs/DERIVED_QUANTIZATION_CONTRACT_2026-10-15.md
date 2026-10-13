# Zerogravity — EXL3/QTIP and Source-Qualified Derived Quantization Contract

**2026-10-15 • Proposed design refinement, NOT a runnable codec or committed numerical equivalence claim.**
**Authority:** v4.3 SemanticProgram→ExecutableBundle; v5.1 ModelAssembly/SourceIdentity/TensorEncoding; source format and quantization are independent of model topology and hardware; no second inference engine.
**Source study:** published low-bit quantization pipelines, including an experimental per-tensor sensitivity optimizer and a source quantization format. Exact implementation is pinned to a source revision before any code begins. Performance numbers below are **upstream author measurements**, NOT Zerogravity results.

## A. Why this is not just add another QuantKind

EXL3 is an extremely-low-bit source quantization family based on QTIP concepts: structured transforms, codebook/trellis representations, source-trained calibration/error shaping and low-bit matrix multiplication. Source variants encode more than a bits_per_weight scalar, and on accelerator hardware their direct GEMV/GEMM execution path matters at least as much as compact storage. A decoder recognizing a packed byte sequence is not qualified to execute a full model; a source model compatible with GGUF Q4 is not automatically source-compatible with EXL3. **Never reinterpret packed codebook identifiers as ordinary Q2/Q4 or assume dequantization into FP16 is the only legal computation plan.**

### Storage, semantics, and physical layout must remain orthogonal

- SourceArtifact: original BF16/FP8 checkpoint with source hash, architecture, config, tokenizer, templates, optional vision/audio/MTP/n-gram assets.
- DerivedArtifact: a **different model-weight artifact** with its own source quantization/recipe/calibration/license and quality identity. Lossless packing of an already-quantized checkpoint is separate from a lossy derived model.
- SourceQuantCodec: EXL3/QTIP/trellis/codebook shape, bit-stream, grouping, scale/zero, sign/basis transforms and error-propagation constraints. Unsupported codebook/bit width is UnsupportedCodec, never unsafe byte reinterpretation.
- PackedDeviceLayout: native architecture/tile/coalescing/WMMA alignment, register/LDS plans; a reversible presentation of the **same derived mathematical weights** unless a separately verified arithmetic transform is declared.
- ActivationCodec: input activation pack/scale/rounding, optionally quantized INT8 decode kernel, with explicit source-relative numeric policy.
- ComputeCandidate: source-verified direct quant GEMV/GEMM or kernel-fused decode, fallback only if source compatible, accounting temporary buffers and dispatch.
- NumericContract: original model-vs-quantized checkpoint quality and exact/approved approximate backend kernel math are **two distinct proofs**.

Illustrative, not final ABI:

~~~rust
struct VerifiedQuantArtifact {
    original_artifact: ContentDigest,
    derived_artifact: ContentDigest,
    codec: RegisteredCodecAndVersion,       // EXL3 K/codebook family
    tensor_recipe: Digest,                  // complete per-tensor format map
    calibration: CalibrationProvenance,
    output_quality: QualityReceipt,         // original vs derived model
    processor_and_model: SemanticIdentity,
}
struct QuantTensorSpec {
    logical_shape: Shape, stored_shape: Shape,
    source_format: SourceCodecId,
    quant_group: ValidatedGroupGeometry,
    codebook: Option<ValidatedCodebookId>,
    basis: Option<VerifiedTransformId>,
    scale_encoding: ExactScaleEncoding,
    numerical_decode: RegisteredReductionContract,
}
struct PackedMatrixKernelKey {
    quant: QuantTensorSpec, device_isa: DeviceIsa,
    m_band: ShapeBand, k_band: ShapeBand, n_band: ShapeBand,
    codegen: CompilerDriverKey, precision: NumericContract,
    output_layout: PhysicalLayoutId,
}
~~~

Do not make EXL3 one hardcoded model family or a monolithic Exl3Model interpreter. Source semantics remain familiar Qwen/GLM/MiMo operators plus variant-specific recurrent/indexer/attention state; the physical codec can be replaced independently.

## B. Correctness and derived artifact identity

Check that every source tensor is accounted for exactly once:
- encoded bytes, expected block/codebook packing, codebook index bounds, endianness and alignment;
- logical axes vs padded GPU axes, special head and expert/tied-gate matrices, bias, per-channel/group scales, quantizer codebook and optional activation basis transform;
- all required inference sidecars, vision parts, MTP heads, lookup/PLE tables and template/processor hashes;
- corrupted and missing tensor/padding/codebook must fail **before** reserving physical GPU work;
- mixed-format models may have FP16/BF16 heads and codebook quantized experts; kernel selection is per tensor role and shape, not one pack-wide bit width;
- allow source-exact scalar dequant oracle for small golden vectors, but avoid mandatory full-model FP16 reconstruction that duplicates 100GiB packed weights in RAM.

An EXL3 pack copied from another library cannot silently assume its original model's tokenizer, attention variant, RoPE and lossless sample identity remain valid. Trust metadata only after cross-checking independently source-verified structure and numerical fixtures.

## C. Per-tensor calibration and sensitivity optimization

An experimental published pipeline goes beyond a uniform 2/3/4-bit recipe. Its workflow uses deployment-shaped self-generated model traces and per-tensor reference output sensitivity, with quantizer-error anchoring and quantizer-shaped noise perturbation; it then allocates bits within a size/quality budget, converts, and tests on separate held-out traces. **This is an optional offline developer pipeline**; it is not permitted to run hidden training/calibration inside an inference server.

Pipeline semantics:

1. Pin original checkpoint and authorized evaluation corpus plus source tokenizer/chat format. Use deployment-representative input **including reasoning/tool patterns** when permitted. Never store private user prompts by default.
2. Generate a **calibration** trace and an independent **held-out** evaluation trace. Record template/reasoning/temperature/RNG and prompt taxonomy. Do not measure quality on the same trace used for optimization.
3. Quantizer error anchor: per-tensor relative error distribution (not just universal quant noise). For EXL3 source variants, account for trellis/codebook and LDLQ error structure.
4. Source-weight tensor sensitivity: inject seeded **shaped** noise into unquantized tensor while keeping remaining unquantized model constant; measure downstream final-logit KLD. Do not claim ordinary isotropic noise is equivalent.
5. Build a constrained per-tensor bitrate solver under exact size plus kernel-group/fusion constraints. E.g. group gate/up projections only where a fused kernel truly requires matched coding, not as an assumed model law. Cost includes codec header, scale, packing, optional sidecar and compressed expert cache.
6. Emit a machine-verifiable recipe with every tensor listed, complete source digest, calibration/eval run IDs and codec version. Missing or duplicated tensors hard fail. Tuning intermediate files remain optional.
7. Convert deterministically where possible, verify exact tensor bytes/codebook decode and public model semantic shapes, then assess independent held-out NLL/KLD/logit differences, token accuracy and tool/long-context outcomes versus **original source**.
8. Independently qualify CPU and physical GPU on the **derived artifact**. A numerically exact kernel on a quantized target is not equivalent to an FP8/BF16 original unless independent quantization evaluation proves the chosen quality policy.
9. Export QualityReceipt and ModeReceipt, with warnings if calibration covers only a limited domain/language/context. No silent model replacement to fit memory.

Recommended offline optimizer inputs: tensor role, per-layer sensitivity, quantizer-error shape and cost, checkpoint topology, target size/quality cap, GPU codec/shape kernel cost, output heads/router/indexer/recurrent criticality and exact-source state. Optimizing KLD alone can choose an impractically slow codebook: use a **Pareto-aware output** of quality *and* measured decode/verify cost.

### Sources and realistic caution

A published experimental optimization guide labels its workflow **experimental and untested on sparse models**, despite positive demonstrations on dense models. Treat extrapolated tensor sensitivities, bitrate curves and source-domain transfer as hypotheses to reproduce, not universal safety guarantees for Qwen3.8/GLM/MiMo MoE. Quantization with its own calibration data is a new artifact, not an exact alternate KernelCandidate.

## D. Critical counterexample: lower bitrate may be slower

A published gfx1151 report found a 2.05 bpw Qwen pack ran around 25.7 generated tok/s versus 44.1 at 3.05 bpw in the corresponding tested MTP setting. The K2 trellis path tiled poorly, despite smaller disk weights and substantial draft acceptance. **Do not build an optimizer using only stored bytes.**

A legal candidate selection key includes per-tensor codec K and codebook, output-row/verify shape, quant-activation path, device ISA/compiler, physical pooling and image/prefill/AR task mode. Runtime cost considers packed decode operations, shared-memory bank conflicts, registers, launch cost, expert union and scratch as well as bytes.

Recommended profiler: codebook decode cycles/byte, index loads and branch cost, source dequant oracle discrepancy, full quant GEMV/GEMM latencies at M=1,2,3,4,8,16,32,64, prefill 128/512/2K/8K, route entropy/unique experts and actual committed tokens/sec with the target model.

## E. Correctness modes and regression policy

- **Source-derived pack quality:** original FP8/BF16 vs lossy EXL3 model. Report top1, KLD/JS or task metrics, calibrated confidence and original model state where comparable.
- **Kernel arithmetic quality:** a CPU/fast reference for the **same quant pack** vs accelerated candidate (include source exact signed zero/overflow/rounding, unique seeded stochastic distribution).
- **Speculative losslessness:** drafted+target-verified outputs vs target **using the same derived pack**. A quant pack with 93.8% top1 vs original FP8 can still have exact speculative output relative to that pack.
- **Batch concurrency determinism:** different legal row reductions may differ numerically even with 16bit KV; source-approved deterministic mode should explicitly control/fail those cases and never market unconditional token equality across all batch sizes.
- **Compilation/storage correctness:** archive integrity, codebook decoding, different context and source sample inputs; only recognize committed prepared bundles and semantic effects.

A reduction changed to escape FP16 overflow without source proof is not performance equivalent merely because PPL is similar. Follow the project strict numerical and StateRoot gates for currently qualified Qwen paths.

## F. Acceptance matrix for P03/P05/P07/P20/P22

| ID | Gate | Reject if |
|---|---|---|
| QTIP-01 | Parse/codebook/trellis source fixtures for each declared EXL3 K, block and scale; independent scalar source oracle | unsupported bit width or codebook yields plausible tensor |
| QTIP-02 | per-tensor mixed precision and special head/router/vision/sidecars; validate all tensor lists | model loads with missing or wrong auxiliary component |
| QTIP-03 | lossless repack roundtrips to same quant weights, preserving padded logical shape | weight identity conflated with new quantization |
| QTIP-04 | malformed codebook indexes/scales, huge bytes, overlapping shards and wrong alignment rejected before allocation | OOM/out-of-bounds/uncertain GPU work for host data error |
| QTIP-05 | CPU scalar quant GEMV output against source conversion reference for M=1 and verifier rows | only perplexity validates codec |
| QTIP-06 | ROCm gfx1151, CUDA and other permitted backend fast-kernel relative to same quant target on original source tensors | false exact numerical mode |
| QTIP-07 | 2.05 vs 3.05 bpw (when source/licensing allow): report both size, quality, GEMV and end-to-end MTP | equating smaller file with faster decode |
| QTIP-08 | calibrate on representative allowed trace, evaluate disjoint held-out trace including code/tools/multilingual | calibration/evaluation leakage |
| QTIP-09 | per-tensor sensitivity+LDLQ-shaped perturbation evidence vs unshaped control; sparse MoE remains experiment | speculative sensitivity promoted as fact |
| QTIP-10 | source model original vs derived pack quality and exactly targeted speculative output both reported separately | derived model labeled original quality |
| QTIP-11 | cancel host quant/repack during artifact staging: no half-install or mutated source hashes | partially written pack recognized as valid |
| QTIP-12 | feature disabled/missing GPU codec returns validated slow/reference alternative or honest Unsupported, never silent different quant | model-name-based unsafe fallback |
| QTIP-13 | actual RAM headroom and file page-cache accounting while loading >80GiB sharded quant pack; no OS pressure hang | GPU MemFree treated as physical available |
| QTIP-14 | one new codec can be added without editing TaskCoordinator/Router/public API parsing | added separate EXL3 inference engine |

## G. Implementation seam and phase order

- **P03**: independent VerifiedQuantCodec parser/reference, derived-artifact manifest/calibration schema, error checks. Minimal source tests; no GPU.
- **P05/P07**: quant region guarded conversion/direct codebook GEMV/GEMM kernels on genuine CPU/GPU with compiler/device evidence.
- **P10/P12**: packed expert cache and source-file/OS page cache, no duplicate GPU reservations; derive expert-specific packed costs.
- **P20**: experimental calibration, sensitivity solver, KLD/quality/benchmark harness (optional/offline).
- **P22**: full model/derived pack, long-context, tool/quality and throughput pass before auto-selecting candidate.

**No new persistent IR or external C++/Python inference engine is required.** Model-derived conversion scripts, if authorized later, may be independent offline tools with full source audit. No user asked to implement them in this design step.
